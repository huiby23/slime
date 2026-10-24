"""Host-only checks for idle scheduling and ownership-safe cleanup."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "nvfp4_gpu_watch", Path(__file__).resolve().parents[1] / "examples/nvfp4_sim/watch_gpu.py"
)
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


def card(idle=True, uuid="gpu-a", index=0):
    return {"uuid": uuid, "index": index, "idle_now": idle, "compute_pids": []}


class GPUWatchTests(unittest.TestCase):
    def test_existing_container_monitor_only_inspects_and_keeps_running(self):
        alive = subprocess.CompletedProcess([], 0, json.dumps([{"State": {"Running": True, "ExitCode": 0}}]), "")
        done = subprocess.CompletedProcess([], 0, json.dumps([{"State": {"Running": False, "ExitCode": 0}}]), "")
        with patch.object(watch.subprocess, "run", side_effect=[alive, done]) as run:
            process = watch.RunningContainer("owned", Path("/unused"))
            self.assertIsNone(process.poll())
            self.assertEqual(process.wait(timeout=1), 0)
            self.assertEqual([call.args[0] for call in run.call_args_list], [["docker", "inspect", "owned"]] * 2)

    def test_removed_container_requires_successful_training_status(self):
        removed = subprocess.CompletedProcess([], 1, "", "Error: No such container")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results").mkdir()
            with patch.object(watch.subprocess, "run", return_value=removed):
                process = watch.RunningContainer("owned", root)
                self.assertEqual(process.poll(), 1)
                watch.save(root / "results/status.json", {"successful_training": True})
                self.assertEqual(process.poll(), 0)

    def test_resume_rejects_missing_model_or_dataset_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "No durable checkpoint"):
                watch.resume_checkpoint(root)
            checkpoint = root / "results/checkpoints"
            checkpoint.mkdir(parents=True)
            (checkpoint / "latest_checkpointed_iteration.txt").write_text("49\n")
            model = checkpoint / "iter_0000049"
            model.mkdir()
            (model / ".metadata").write_text("model and optimizer metadata")
            (model / "__0_0.distcp").write_text("parameters and optimizer state")
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                watch.resume_checkpoint(root)
            (checkpoint / "rollout").mkdir()
            with zipfile.ZipFile(checkpoint / "rollout/global_dataset_state_dict_49.pt", "w") as archive:
                archive.writestr("data.pkl", "dataset cursor")
            info = watch.resume_checkpoint(root)
            self.assertEqual(info["completed_updates"], 50)
            self.assertEqual(info["step"], 49)

    def test_validation_gate_waits_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(watch.validation_dependency(root, root, root)["state"], "waiting")
            watch.save(root / "status.json", {"state": "running", "completed_updates": 1})
            self.assertEqual(watch.validation_dependency(root, root, root)["completed_updates"], 1)
            watch.save(root / "status.json", {"state": "failed"})
            self.assertEqual(watch.validation_dependency(root, root, root)["state"], "failed")

    def test_validation_gate_requires_all_evidence_and_matching_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt, template = root / "smoke", root / "formal"
            (attempt / "results").mkdir(parents=True)
            template.mkdir()
            training = {
                key: "same"
                for key in (
                    "model",
                    "prompts_per_update",
                    "samples_per_prompt",
                    "global_batch_size",
                    "response_token_limit",
                    "seed",
                    "learning_rate",
                    "importance_correction",
                )
            }
            for directory in (attempt, template):
                watch.save(directory / "config.json", {"training": training, "image_id": "pinned-image"})
                watch.save(
                    directory / "manifest.json",
                    {"baseline_commit": "baseline", "simulator_commit": "sim", "recipe": {"wgrad": "NVFP4"}},
                )
                (directory / "mismatch.yaml").write_text("same-quantizer-config")
            watch.save(
                root / "status.json",
                {
                    "state": "completed",
                    "completed_updates": 2,
                    "exit_code": 0,
                    "current_attempt": {"directory": str(attempt)},
                },
            )
            watch.save(
                attempt / "results/status.json",
                {"successful_training": True, "training_validation_passed": True, "completed_updates": 2},
            )
            report = {
                "passed": True,
                "verified_rollout_ids": [0, 1],
                "checks": dict.fromkeys(watch.VALIDATION_CHECKS, True),
            }
            watch.save(attempt / "results/training-validation.json", report)
            with patch.object(watch, "command", return_value=""):
                self.assertEqual(watch.validation_dependency(root, template, root)["state"], "passed")
                report["checks"]["two_active_optimizer_updates"] = False
                watch.save(attempt / "results/training-validation.json", report)
                self.assertEqual(watch.validation_dependency(root, template, root)["state"], "failed")
                report["checks"]["two_active_optimizer_updates"] = True
                watch.save(attempt / "results/training-validation.json", report)
                (template / "mismatch.yaml").write_text("different-quantizer-config")
                self.assertEqual(watch.validation_dependency(root, template, root)["state"], "failed")
            (template / "mismatch.yaml").write_text("same-quantizer-config")
            with patch.object(watch, "command", side_effect=subprocess.CalledProcessError(1, ["git", "diff"])):
                self.assertEqual(watch.validation_dependency(root, template, root)["state"], "failed")

    def test_strict_threshold_and_independent_cards(self):
        tracker = watch.IdleTracker()
        for second in range(0, 301, 5):
            candidates = tracker.update([card(), card(False, "gpu-b", 1)], second)
            self.assertEqual(candidates, [])
        candidates = tracker.update([card(), card(False, "gpu-b", 1)], 305)
        self.assertEqual([gpu["uuid"] for gpu in candidates], ["gpu-a"])
        self.assertEqual(candidates[0]["idle_observed_seconds"], 305)

    def test_activity_query_gaps_and_disappearance_reset_timer(self):
        tracker = watch.IdleTracker()
        tracker.update([card()], 0)
        tracker.update([card()], 5)
        tracker.update([card(False)], 10)
        gpu = card()
        tracker.update([gpu], 15)
        self.assertEqual(gpu["idle_observed_seconds"], 0)
        tracker.update([card()], 20)
        gpu = card()
        tracker.update([gpu], 50)
        self.assertEqual(gpu["idle_observed_seconds"], 0)
        tracker.update([], 55)
        tracker.update([card()], 60)
        self.assertEqual(tracker.since, {"gpu-a": 60})
        tracker.reset()  # Called on an nvidia-smi failure, so unknown time is never credited.
        gpu = card()
        tracker.update([gpu], 1000)
        self.assertEqual(gpu["idle_observed_seconds"], 0)

    def test_snapshot_does_not_treat_occupied_zero_utilization_as_idle(self):
        with patch.object(
            watch,
            "command",
            side_effect=[
                "gpu-b, 99",
                "0, gpu-a, A100, 1, 0\n1, gpu-b, A100, 1, 0\n" "2, gpu-c, A100, 1000, 0\n3, gpu-d, A100, 1, 10",
            ],
        ):
            self.assertEqual([gpu["idle_now"] for gpu in watch.snapshot()], [True, False, False, False])

    def test_prepare_attempt_preserves_template_and_isolates_new_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "template"
            template.mkdir()
            config = {
                "training": {"updates": 2},
                "container_run_dir": "/nvfp4-run/results",
                "wall_time_limit_seconds": 7200,
                "environment": {"TORCHINDUCTOR_CACHE_DIR": "/nvfp4-run/cache/torchinductor"},
            }
            original = {
                "simulator_commit": "old-commit",
                "docker_command": [
                    "docker",
                    "run",
                    "--rm",
                    "--gpus",
                    "device=7",
                    "--mount",
                    "type=bind,src=/old,dst=/nvfp4-run",
                    "image",
                    "supervisor.py",
                ],
            }
            watch.save(template / "config.json", config)
            watch.save(template / "manifest.json", original)
            watch.save(template / "configuration-check.json", {"passed": True})
            for filename in ("mismatch.yaml", "validation.jsonl"):
                (template / filename).write_text("original\n")
            gpu = card()
            gpu["idle_observed_seconds"] = 305
            directory = root / "attempt-001"
            argv = watch.prepare_attempt(template, directory, gpu, "our-container", "our-watch", "new-commit")
            self.assertEqual(json.loads((template / "manifest.json").read_text()), original)
            self.assertEqual(argv[argv.index("--gpus") + 1], "device=gpu-a")
            self.assertEqual(argv[argv.index("--label") + 1], watch.LABEL + "=our-watch")
            self.assertIn("type=bind,src=%s,dst=/nvfp4-run" % directory, argv)
            prepared = json.loads((directory / "config.json").read_text())
            self.assertEqual(prepared["physical_gpus"], [0])
            self.assertEqual(prepared["assigned_gpu_uuids"], ["gpu-a"])
            self.assertEqual(prepared["training"]["updates"], 2)
            self.assertFalse(prepared["gpu_allocation_managed_by_sequence"])
            self.assertTrue((directory / "results/logs").is_dir())
            self.assertEqual(prepared["wall_time_limit_seconds"], 7200)
            self.assertTrue((directory / "cache/torchinductor").is_dir())
            with self.assertRaises(AssertionError):
                watch.prepare_attempt(template, directory, gpu, "our-container", "our-watch", "new-commit")
            config.update(experiment_mode="formal", requires_passed_validation=True, wall_time_limit_seconds=96 * 3600)
            config["training"]["updates"] = 100
            watch.save(template / "config.json", config)
            (template / "validation.jsonl").unlink()  # Formal eval uses the original 512-question asset.
            directory = root / "formal-attempt"
            watch.prepare_attempt(template, directory, gpu, "formal-container", "our-watch", "new-commit")
            prepared = json.loads((directory / "config.json").read_text())
            self.assertEqual(prepared["training"]["updates"], 100)
            self.assertEqual(prepared["wall_time_limit_seconds"], 96 * 3600)
            config.update(
                image_id="pinned-image",
                training_command=[
                    "python",
                    "train.py",
                    "--load",
                    "initial-HF",
                    "--finetune",
                    "--no-load-optim",
                    "--no-load-rng",
                ],
            )
            original.update(baseline_commit="baseline", recipe={"wgrad": "NVFP4"})
            watch.save(template / "config.json", config)
            watch.save(template / "manifest.json", original)
            source = root / "previous-formal-attempt"
            source.mkdir()
            watch.save(source / "config.json", config)
            watch.save(source / "manifest.json", original)
            (source / "mismatch.yaml").write_bytes((template / "mismatch.yaml").read_bytes())
            selected = {"directory": str(source / "results/checkpoints"), "step": 49, "completed_updates": 50}
            directory = root / "resumed-attempt"
            with patch.object(watch, "resume_checkpoint", return_value=selected):
                argv = watch.prepare_attempt(
                    template, directory, gpu, "resumed-container", "our-watch", "new-commit", source
                )
            resumed = json.loads((directory / "config.json").read_text())
            command = resumed["training_command"]
            self.assertEqual(command[command.index("--load") + 1], "/nvfp4-resume")
            self.assertEqual(command[command.index("--start-rollout-id") + 1], "50")
            self.assertEqual(command[command.index("--ckpt-step") + 1], "49")
            self.assertTrue(all(flag not in command for flag in ("--finetune", "--no-load-optim", "--no-load-rng")))
            self.assertIn("type=bind,src=%s,dst=/nvfp4-resume,readonly" % selected["directory"], argv)
            self.assertIn("--finetune", json.loads((template / "config.json").read_text())["training_command"])

    def test_cleanup_refuses_a_container_with_a_different_owner(self):
        result = subprocess.CompletedProcess(
            [], 0, json.dumps([{"Config": {"Labels": {watch.LABEL: "someone-else"}}}]), ""
        )
        with patch.object(watch.subprocess, "run", return_value=result) as run:
            with self.assertRaisesRegex(RuntimeError, "ownership label mismatch"):
                watch.stop_own_container("container", "our-watch")
            self.assertEqual(run.call_count, 1)

    def test_cleanup_handles_container_disappearing_between_inspect_and_stop(self):
        inspect = subprocess.CompletedProcess([], 0, json.dumps([{"Config": {"Labels": {watch.LABEL: "ours"}}}]), "")
        removed = subprocess.CompletedProcess([], 1, "", "Error: No such container")
        with patch.object(watch.subprocess, "run", side_effect=[inspect, removed]) as run:
            watch.stop_own_container("container", "ours")
            self.assertEqual(run.call_args_list[1][0][0], ["docker", "stop", "--timeout", "15", "container"])

    def test_foreign_detection_handles_current_children_and_reused_pids(self):
        gpu = card()
        gpu["compute_pids"] = [10, 20, 30, 40]
        known = {30}  # A stale owned PID must not exempt a new foreign process.
        with patch.object(watch, "command", side_effect=[json.dumps([{"Id": "full-own-container-id"}]), "PID\n10"]):
            with patch.object(
                Path,
                "read_text",
                side_effect=["0::/docker/full-own-container-id", "0::/user.slice", FileNotFoundError()],
            ):
                self.assertEqual(watch.foreign_pids("ours", gpu, known), [30])
        self.assertEqual(known, {10, 20})


if __name__ == "__main__":
    unittest.main()
