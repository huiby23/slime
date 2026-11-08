"""Portable TIM configuration/data/source tests; no Torch, Docker, or GPU needed.

Run: python3 -m unittest discover -s tests -p test_tim_protocol.py -v
"""

import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ENTRY = Path(__file__).resolve().parents[1] / "examples/tim"


def load(name):
    spec = importlib.util.spec_from_file_location("tim_test_" + name, ENTRY / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


protocol, runner = load("protocol"), load("run")
container_driver = load("container_driver")


def preset(name="bf16_lr1e6"):
    return protocol.load_config(ENTRY / "configs" / (name + ".json"))


def options(command):
    return {
        item: command[i + 1] if i + 1 < len(command) and not command[i + 1].startswith("--") else None
        for i, item in enumerate(command)
        if item.startswith("--")
    }


def record(prompt, correct=None, source_index=None):
    metadata = {}
    if correct is not None:
        metadata.update(screen_samples=8, screen_correct_count=correct)
    if source_index is not None:
        metadata["source_index"] = source_index
    return dict(prompt=[{"role": "user", "content": prompt}], label="2", metadata=metadata)


def dataset(path, records):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    return path


class ConfigurationTests(unittest.TestCase):
    def test_final_update_requires_periodic_or_early_evaluation(self):
        config = preset()
        config["training"]["updates"] = 2
        with self.assertRaisesRegex(ValueError, "final update"):
            protocol.validate_config(config)
        config["evaluation"].update(early_updates=2, retain_early=2)
        protocol.validate_config(config)
        config["evaluation"].update(interval=1, early_updates=0, retain_early=0)
        protocol.validate_config(config)

    def test_inheritance_preserves_shared_protocol_and_lr_variants(self):
        for name, lr in [("bf16_lr1e6", 1e-6), ("bf16_lr1e5", 1e-5), ("bf16_lr1e3", 1e-3)]:
            config = preset(name)
            self.assertEqual(config["optimizer"]["lr"], lr)
            self.assertEqual(config["optimizer"]["weight_decay"], 0.01)
            self.assertEqual(config["training"]["prompts"] * config["training"]["answers"], 128)
            self.assertEqual(config["training"]["updates"], 500)
            self.assertEqual(config["data"]["min_correct"], 2)
            self.assertEqual(config["data"]["max_correct"], 6)
        updated = protocol.load_config(ENTRY / "configs/bf16_lr1e6.json", ["optimizer.lr=0.00002"])
        self.assertEqual(updated["optimizer"]["lr"], 2e-5)
        self.assertEqual(preset()["optimizer"]["lr"], 1e-6)

    def test_unknown_keys_and_malformed_overrides_are_rejected(self):
        for override in ["optimizer.misspelled=1", "optimizer.lr", "missing.lr=1"]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                protocol.load_config(ENTRY / "configs/base.json", [override])
        config = preset()
        for section, key in [(None, "unknown"), ("training", "unknown")]:
            broken = copy.deepcopy(config)
            (broken if section is None else broken[section])[key] = 1
            with self.subTest(section=section), self.assertRaises(ValueError):
                protocol.validate_config(broken)

    def test_strict_types_ranges_and_incompatible_modes(self):
        invalid = [
            ("schema_version", None, True),
            ("runtime", "image", 42),
            ("runtime", "image", ""),
            ("training", "updates", True),
            ("training", "answers", 0),
            ("optimizer", "lr", True),
            ("optimizer", "lr", float("nan")),
            ("optimizer", "beta2", 1),
            ("optimizer", "weight_decay", -1),
            ("rollout", "top_k", -2),
            ("rollout", "max_running_requests", True),
            ("topology", "colocate", 1),
            ("topology", "sampler_replicas", 4),
            ("evaluation", "early_updates", 11),
            ("evaluation", "precision", "nvfp4_sim"),
            ("trainer", None, "nvfp4_sim"),
            ("data", "screen_samples", True),
            ("data", "min_correct", 7),
            ("training", "response_tokens", 2048),
        ]
        for section, key, value in invalid:
            config = preset()
            if key is None:
                config[section] = value
            else:
                config[section][key] = value
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                protocol.validate_config(config)
        noisy = preset("fixed_noise_s005")
        noisy["topology"] = dict(colocate=False, sampler_replicas=1)
        with self.assertRaises(ValueError):
            protocol.validate_config(noisy)

    def test_inheritance_cycles_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.json").write_text('{"extends":"b.json"}')
            (root / "b.json").write_text('{"extends":"a.json"}')
            with self.assertRaisesRegex(ValueError, "cycle"):
                protocol.load_config(root / "a.json")

    def test_commands_keep_optimizer_sampling_and_recompute_contract(self):
        for name, lr in [("bf16_lr1e6", 1e-6), ("bf16_lr1e5", 1e-5), ("bf16_lr1e3", 1e-3)]:
            command = protocol.native_command(preset(name))
            opts = options(command)
            self.assertEqual(command[:3], ["python", "-m", "slime_plugins.tim.bf16.train"])
            self.assertEqual(float(opts["--lr"]), lr)
            self.assertEqual(float(opts["--min-lr"]), lr)
            self.assertEqual(opts["--lr-decay-style"], "constant")
            self.assertEqual(opts["--lr-warmup-iters"], "0")
            self.assertEqual(opts["--rollout-batch-size"], "16")
            self.assertEqual(opts["--n-samples-per-prompt"], "8")
            self.assertEqual(opts["--global-batch-size"], "128")
            self.assertEqual(opts["--optimizer"], "adam")
            self.assertEqual(float(opts["--adam-beta1"]), 0.9)
            self.assertEqual(float(opts["--adam-beta2"]), 0.98)
            for disabled in (
                "--use-tis",
                "--use-rs",
                "--use-rollout-logprobs",
                "--async-save",
                "--no-save-optim",
                "--no-save-rng",
            ):
                self.assertNotIn(disabled, opts)
            self.assertIn("--get-mismatch-metrics", opts)
            self.assertNotIn("--custom-model-provider-path", opts)

    def test_noise_changes_sampler_only_and_uses_clean_evaluation(self):
        config = preset("fixed_noise_s005")
        command = options(protocol.native_command(config))
        self.assertEqual(config["trainer"], "bf16")
        self.assertEqual(config["evaluation"]["precision"], "bf16")
        self.assertNotIn("--custom-model-provider-path", command)
        self.assertEqual(command["--eval-function-path"], "slime_plugins.tim.fixed_noise.eval.generate_rollout")
        override = json.loads(command["--sglang-json-model-override-args"])
        self.assertEqual(override["fixed_noise_scale"], 0.05)
        self.assertEqual(override["fixed_noise_bank_manifest"], "/noise-bank/manifest.json")
        self.assertEqual(override["architectures"], ["FixedNoiseQwen3ForCausalLM"])
        self.assertEqual(
            protocol.runtime_environment(config)["SGLANG_EXTERNAL_MODEL_PACKAGE"],
            "slime_plugins.tim.fixed_noise.sglang_models",
        )

    def test_nvfp4_topology_and_simulated_training_are_explicit(self):
        config = preset("nvfp4_rollout4_fresh")
        self.assertEqual(protocol.gpu_count(config), 5)
        command = options(protocol.native_command(config))
        self.assertEqual(command["--actor-num-gpus-per-node"], "1")
        self.assertEqual(command["--rollout-num-gpus"], "4")
        self.assertEqual(command["--rollout-num-gpus-per-engine"], "1")
        self.assertEqual(command["--router-policy"], "round_robin")
        self.assertNotIn("--colocate", command)
        override = json.loads(command["--sglang-json-model-override-args"])
        self.assertTrue(override["nvfp4_sim_quantize_weights"])
        self.assertTrue(override["nvfp4_sim_quantize_activations"])
        self.assertTrue(override["nvfp4_sim_reference_arithmetic"])
        simulated = options(protocol.native_command(preset("nvfp4_train_rollout")))
        self.assertEqual(simulated["--custom-model-provider-path"], "slime_plugins.nvfp4_sim.megatron.model_provider")


class DatasetTests(unittest.TestCase):
    def check(self, root, train, evaluation):
        config = preset()
        config["data"].update(train_questions=len(train), eval_questions=len(evaluation))
        return runner.check_datasets(
            config, dataset(root / "train.jsonl", train), dataset(root / "eval.jsonl", evaluation)
        )

    def test_only_training_is_screened_and_unicode_embedded_lines_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = [record("含有分隔符\u2028和\u2029的题", 2, 1), record("另一题", 6, 2)]
            evaluation = [record("固定验证题", 0, 3)]
            report = self.check(root, train, evaluation)
            self.assertEqual(report["train_questions"], 2)
            self.assertEqual(report["screening_applies_to"], "training_only")
            self.assertTrue(report["evaluation_is_not_difficulty_filtered"])
            self.assertEqual(runner.rows(root / "train.jsonl"), train)

    def test_prompt_or_source_index_overlap_and_duplicates_are_rejected(self):
        train = [record("a", 2, 1), record("b", 6, 2)]
        for evaluation in [[record("a", 0, 3)], [record("c", 0, 1)]]:
            with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "overlap"):
                self.check(Path(temporary), train, evaluation)
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "duplicate"):
            self.check(Path(temporary), [train[0], train[0]], [record("c", 0, 3)])

    def test_missing_or_out_of_range_training_screening_is_rejected(self):
        for correct in [None, 1, 7, True]:
            with tempfile.TemporaryDirectory() as temporary, self.subTest(correct=correct), self.assertRaisesRegex(
                ValueError, "screening"
            ):
                self.check(Path(temporary), [record("a", correct, 1)], [record("b", 0, 2)])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = record("a", 2, 1)
            row["metadata"]["screen_samples"] = 16
            with self.assertRaisesRegex(ValueError, "screening"):
                self.check(root, [row], [record("b", 0, 2)])


class FrozenSourceAndMountTests(unittest.TestCase):
    def test_gpu_indices_and_uuid_aliases_cannot_double_allocate_one_device(self):
        report = "0, GPU-first, 0, 0\n1, GPU-second, 0, 0\n"
        with patch.object(runner.subprocess, "check_output", side_effect=[report, ""]):
            with self.assertRaisesRegex(ValueError, "duplicate physical"):
                runner.idle_gpu_uuids(["0", "GPU-first"])
        with patch.object(runner.subprocess, "check_output", side_effect=[report, ""]):
            self.assertEqual(runner.idle_gpu_uuids(["0", "GPU-second"]), ["GPU-first", "GPU-second"])
        for gpu_report, active in [
            (report, "GPU-first\n"),
            (report.replace("0, GPU-first, 0, 0", "0, GPU-first, 65, 0"), ""),
        ]:
            with patch.object(runner.subprocess, "check_output", side_effect=[gpu_report, active]):
                with self.assertRaisesRegex(RuntimeError, "busy"):
                    runner.idle_gpu_uuids(["0"])

    def test_container_workdir_keeps_extension_namespace_importable(self):
        config = preset()
        plan = dict(
            config=config,
            paths=dict(
                output="/output-host",
                model_mount="/model-host",
                train="/train-host",
                evaluation="/eval-host",
                noise_bank=None,
            ),
            environment=protocol.runtime_environment(config),
            image_id="sha256:local",
            gpus=[],
        )
        command = runner.docker_command(plan, "preflight")
        self.assertNotIn("--gpus", command)
        workdir = command[command.index("--workdir") + 1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native, extension = root / "runtime/slime", root / "runtime/extensions"
            (native / "slime_plugins").mkdir(parents=True)
            (native / "slime_plugins/__init__.py").write_text("")
            (native / "train.py").write_text("native_marker = True\n")
            (extension / "slime_plugins/tim").mkdir(parents=True)
            (extension / "slime_plugins/__init__.py").write_text(
                "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"
            )
            (extension / "slime_plugins/tim/__init__.py").write_text("")
            cwd = root / workdir.lstrip("/")
            cwd.mkdir(parents=True, exist_ok=True)
            environment = dict(os.environ, PYTHONPATH=str(extension) + os.pathsep + str(native))
            process = subprocess.run(
                [sys.executable, "-c", "import slime_plugins.tim; import train; assert train.native_marker"],
                cwd=str(cwd),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)

    def test_output_cannot_live_in_copied_sources_or_model_assets(self):
        model = Path("/local/model")
        for output in [ENTRY / "runs/new", runner.REPO / "slime_plugins/tim/new", model / "runs/new"]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                runner.check_output_location(output.resolve(), model)
        runner.check_output_location(Path("/separate/output"), model)

    def test_hf_snapshot_mount_contains_relative_blob_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models--Qwen--Qwen3-0.6B"
            blob = root / "blobs" / "weights"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"local weights")
            snapshot = root / "snapshots" / "version"
            snapshot.mkdir(parents=True)
            (snapshot / "model.safetensors").symlink_to("../../blobs/weights")
            mounted, inside = runner.model_mount(snapshot)
            self.assertEqual(mounted, root)
            self.assertEqual(inside, "/model/snapshots/version")
            (snapshot / "bad").symlink_to("../../../outside")
            with self.assertRaisesRegex(ValueError, "symlink"):
                runner.model_mount(snapshot)

    def test_standalone_model_mount_and_escaping_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            self.assertEqual(runner.model_mount(model), (model, "/model"))
            (root / "external").write_bytes(b"x")
            (model / "weights").symlink_to("../external")
            with self.assertRaisesRegex(ValueError, "symlink"):
                runner.model_mount(model)

    def test_archive_rejects_path_traversal_and_escaping_links(self):
        cases = [
            ("file", "../outside", None),
            ("symlink", "dir/link", "../../outside"),
            ("symlink", "dir/link", "/outside"),
            ("hardlink", "weights", "../outside"),
        ]
        for kind, name, link in cases:
            with self.subTest(kind=kind, name=name, link=link), tempfile.TemporaryDirectory() as temporary:
                payload = io.BytesIO()
                with tarfile.open(fileobj=payload, mode="w") as archive:
                    member = tarfile.TarInfo(name)
                    if kind in ("symlink", "hardlink"):
                        member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                        member.linkname = link
                    archive.addfile(member)
                payload.seek(0)
                with tarfile.open(fileobj=payload) as archive, self.assertRaises(ValueError):
                    runner.safe_extract(archive, Path(temporary))

    def test_archive_allows_owned_relative_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w") as archive:
                body = b"owned payload"
                member = tarfile.TarInfo("dir/payload")
                member.size = len(body)
                archive.addfile(member, io.BytesIO(body))
                link = tarfile.TarInfo("dir/link")
                link.type, link.linkname = tarfile.SYMTYPE, "payload"
                archive.addfile(link)
            payload.seek(0)
            root = Path(temporary)
            with tarfile.open(fileobj=payload) as archive:
                runner.safe_extract(archive, root)
            self.assertEqual((root / "dir/link").read_bytes(), b"owned payload")

    def test_prepare_writes_disabled_corrections_and_never_starts_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = preset()
            plan = dict(
                config=config,
                paths=dict(output=str(root / "output"), model="/local/model"),
                slime_commit=protocol.SLIME_COMMIT,
            )
            args = SimpleNamespace(slime_repo="/local/source")
            with patch.object(runner, "check_model", return_value={"passed": True}), patch.object(
                runner, "copy_sources", return_value={"source.py": "hash"}
            ), patch.object(runner.subprocess, "check_output", return_value="sha256:local-image") as inspect:
                runner.prepare(args, plan)
            settings = json.loads((root / "output/mismatch.json").read_text())
            self.assertFalse(settings["use_tis"])
            self.assertFalse(settings["use_rs"])
            inspect.assert_called_once()
            self.assertEqual(inspect.call_args[0][0][:3], ["docker", "image", "inspect"])
            status = json.loads((root / "output/status.json").read_text())
            self.assertEqual(status["gpu_jobs_launched"], 0)
            with self.assertRaises(FileExistsError):
                runner.prepare(args, plan)


class CompletionEvidenceTests(unittest.TestCase):
    def test_completion_requires_one_successful_record_for_every_update(self):
        records = [
            dict(rollout_id=index, step_id=0, optimizer_success=True, active_learning_step=False, global_grad_norm=0)
            for index in range(3)
        ]
        self.assertTrue(container_driver.complete_optimizer_records(records, 3))
        self.assertTrue(container_driver.complete_optimizer_records(list(reversed(records)), 3))
        # Reaching the last rollout ID is insufficient if intervening updates are absent.
        self.assertFalse(container_driver.complete_optimizer_records(records[-1:], 3))
        self.assertFalse(container_driver.complete_optimizer_records(records + records[-1:], 3))
        self.assertFalse(container_driver.complete_optimizer_records([records[0], records[0], records[2]], 3))
        for change in [dict(step_id=1), dict(optimizer_success=False)]:
            broken = copy.deepcopy(records)
            broken[1].update(change)
            with self.subTest(change=change):
                self.assertFalse(container_driver.complete_optimizer_records(broken, 3))


if __name__ == "__main__":
    unittest.main()
