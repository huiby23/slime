"""CPU behavioral checks; fake SGLang loader, actual torch/noise arithmetic."""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from slime_plugins.tim.fixed_noise.bank import FixedNoiseBank, file_sha, tensor_sha


class FakeBase(torch.nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.pp_group = types.SimpleNamespace(world_size=1)
        self.quant_config = None
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Module()
        self.model.embed_tokens.weight = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))
        self.model.norm = torch.nn.Module()
        self.model.norm.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        self.lm_head = self.model.embed_tokens

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, value in weights:
            if name in params:
                params[name].data.copy_(value)

    def forward(self):
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])


def install_fake_sglang():
    for name in ["sglang", "sglang.srt", "sglang.srt.models"]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    distributed = types.ModuleType("sglang.srt.distributed")
    distributed.get_tensor_model_parallel_world_size = lambda: 1
    sys.modules[distributed.__name__] = distributed
    qwen = types.ModuleType("sglang.srt.models.qwen3")
    qwen.Qwen3ForCausalLM = FakeBase
    sys.modules[qwen.__name__] = qwen


with mock.patch.dict(sys.modules):
    install_fake_sglang()
    from slime_plugins.tim.fixed_noise.sglang_models.qwen3 import FixedNoiseQwen3ForCausalLM

_spec = importlib.util.spec_from_file_location(
    "tim_prepare_noise_bank", Path(__file__).resolve().parents[1] / "examples/tim/prepare_noise_bank.py"
)
prepare_bank = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_bank)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.clean = {
            "model.embed_tokens.weight": torch.tensor([[0.25, -0.5], [1.0, 0.125]], dtype=torch.bfloat16),
            "model.norm.weight": torch.tensor([1.0, 0.75], dtype=torch.bfloat16),
        }
        gen = torch.Generator(device="cpu").manual_seed(20261030)
        entries = {}
        for index, (name, value) in enumerate(self.clean.items()):
            eta = torch.randn(value.shape, generator=gen) * value.float()
            path = self.root / ("noise" + str(index) + ".pt")
            torch.save(eta, path)
            entries[name] = dict(
                file=path.name,
                sha256=file_sha(path),
                eta_tensor_sha256=tensor_sha(eta),
                shape=list(eta.shape),
                dtype="torch.float32",
            )
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    passed=True,
                    seed=20261030,
                    source_alias_equalityverified=True,
                    canonical_tensor_count=2,
                    canonical_aliases={"lm_head.weight": "model.embed_tokens.weight"},
                    tensors=entries,
                )
            )
        )

    def tearDown(self):
        self.temp.cleanup()

    def model(self, name="run"):
        directory = self.root / name
        return FixedNoiseQwen3ForCausalLM(
            types.SimpleNamespace(
                fixed_noise_bank_manifest=str(self.manifest),
                fixed_noise_scale=0.05,
                fixed_noise_audit_path=str(directory / "noise-audit.json"),
                fixed_noise_control_path=str(directory / "noise-control.json"),
                num_hidden_layers=0,
                tie_word_embeddings=True,
            )
        )

    def test_fixed_delta_repeated_fresh_updates_and_rng(self):
        state = torch.get_rng_state().clone()
        bank = FixedNoiseBank(self.manifest, 0.05)
        name = "model.embed_tokens.weight"
        fresh = self.clean[name].clone()
        original = fresh.clone()
        _, first, _ = bank.transform(name, fresh)
        fingerprint = bank.records[name]["delta_fingerprint"]
        _, repeated, reused = bank.transform(name, fresh)
        self.assertTrue(reused)
        self.assertTrue(torch.equal(first, repeated))
        self.assertTrue(torch.equal(fresh, original))
        next_weight = (fresh.float() + 0.1875).to(torch.bfloat16)
        _, changed, _ = bank.transform(name, next_weight)
        self.assertTrue(torch.equal(changed, (next_weight.float() + bank._deltas[name]).to(torch.bfloat16)))
        self.assertEqual(fingerprint, bank.records[name]["delta_fingerprint"])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_partial_buckets_order_alias_and_no_accumulation(self):
        first, second = self.model("first"), self.model("second")
        items = list(self.clean.items())
        for item in items:
            first.load_weights([item])
        second.load_weights(list(reversed(items)))
        self.assertTrue(torch.equal(first.forward(), second.forward()))
        expected = first.forward().clone()
        first.load_weights([("lm_head.weight", self.clean["model.embed_tokens.weight"]), *items])
        self.assertTrue(torch.equal(expected, first.forward()))
        self.assertEqual(first.lm_head.weight.data_ptr(), first.model.embed_tokens.weight.data_ptr())
        self.assertEqual(first._fixed_noise_state["input_untouched_failures"], 0)
        self.assertEqual(first._fixed_noise_state["input_untouched_checks"], 5)

    def test_clean_evaluation_and_exact_noisy_restore(self):
        from slime_plugins.tim.fixed_noise.eval import generate_rollout

        model = self.model()
        model.load_weights(list(self.clean.items()))
        noisy = model.forward().clone()
        clean = torch.cat([x.reshape(-1) for x in self.clean.values()])
        native = types.ModuleType("slime.rollout.sglang_rollout")
        native.generate_rollout = lambda *a, **kw: model.forward().clone()
        fake_modules = {native.__name__: native}
        for name in ["slime", "slime.rollout"]:
            module = types.ModuleType(name)
            module.__path__ = []
            fake_modules[name] = module
        args = types.SimpleNamespace(
            sglang_disable_radix_cache=True,
            sglang_json_model_override_args=json.dumps(
                dict(
                    fixed_noise_scale=0.05,
                    fixed_noise_audit_path=str(self.root / "run/noise-audit.json"),
                    fixed_noise_control_path=str(self.root / "run/noise-control.json"),
                )
            ),
        )
        with mock.patch.dict(sys.modules, fake_modules):
            result = generate_rollout(args, 0, None, evaluation=True)
        self.assertTrue(torch.equal(result, clean))
        self.assertTrue(torch.equal(model.forward(), noisy))
        self.assertEqual(model._fixed_noise_state["clean_mode_transitions"], 1)
        self.assertEqual(model._fixed_noise_state["noisy_mode_transitions"], 1)
        self.assertGreater(model._fixed_noise_state["clean_forward_calls"], 0)
        marker = json.loads((self.root / "run/clean-evaluation.jsonl").read_text())
        self.assertTrue(marker["completed"])
        self.assertEqual(marker["restore_requested"], "noisy")
        # The pristine copies survive the clean/noisy switching roundtrip.
        self.assertTrue(all(torch.equal(model._fixed_noise_clean[k], v) for k, v in self.clean.items()))

    def test_name_keyed_preparation_preserves_rng_and_initial_weights(self):
        initial = self.clean["model.embed_tokens.weight"].clone()
        state = torch.get_rng_state().clone()
        first = prepare_bank.make_eta(initial, 123, "model.embed_tokens.weight")
        second = prepare_bank.make_eta(initial, 123, "model.embed_tokens.weight")
        other_name = prepare_bank.make_eta(initial, 123, "model.other.weight")
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other_name))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(initial, self.clean["model.embed_tokens.weight"]))
        self.assertTrue(torch.equal(prepare_bank.make_eta(initial * 2, 123, "model.embed_tokens.weight"), first * 2))

    def test_seed_is_configurable_and_zero_sigma_is_exact_control(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["seed"] = 1234
        self.manifest.write_text(json.dumps(manifest))
        bank = FixedNoiseBank(self.manifest, 0)
        fresh = self.clean["model.embed_tokens.weight"]
        _, actual, _ = bank.transform("lm_head.weight", fresh)
        self.assertTrue(torch.equal(actual, fresh))
        self.assertNotEqual(actual.data_ptr(), fresh.data_ptr())

    def test_shape_and_non_bf16_payload_are_rejected(self):
        bank = FixedNoiseBank(self.manifest, 0.05)
        with self.assertRaises(ValueError):
            bank.transform("model.norm.weight", torch.ones(3, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            bank.transform("model.norm.weight", torch.ones(2, dtype=torch.float32))

    def test_failed_clean_evaluation_still_requests_noisy_restore(self):
        from slime_plugins.tim.fixed_noise.eval import generate_rollout

        model = self.model("failed-eval")
        model.load_weights(list(self.clean.items()))
        noisy = model.forward().clone()
        native = types.ModuleType("slime.rollout.sglang_rollout")

        def fail(*args, **kwargs):
            model.forward()
            raise RuntimeError("evaluation failed")

        native.generate_rollout = fail
        args = types.SimpleNamespace(
            sglang_disable_radix_cache=True,
            sglang_json_model_override_args=json.dumps(
                dict(
                    fixed_noise_scale=0.05,
                    fixed_noise_audit_path=str(self.root / "failed-eval/noise-audit.json"),
                    fixed_noise_control_path=str(self.root / "failed-eval/noise-control.json"),
                )
            ),
        )
        fake_modules = {native.__name__: native}
        for name in ("slime", "slime.rollout"):
            module = types.ModuleType(name)
            module.__path__ = []
            fake_modules[name] = module
        with mock.patch.dict(sys.modules, fake_modules), self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            generate_rollout(args, 3, None, evaluation=True)
        self.assertTrue(torch.equal(model.forward(), noisy))
        marker = json.loads((self.root / "failed-eval/clean-evaluation.jsonl").read_text())
        self.assertFalse(marker["completed"])

    def test_atomic_control_replacement_detected_with_same_mtime(self):
        from slime_plugins.tim.fixed_noise.eval import _write_mode

        model = self.model("same-mtime")
        model.load_weights(list(self.clean.items()))
        noisy = model.forward().clone()
        control = self.root / "same-mtime/noise-control.json"
        _write_mode(control, "clean", 0)
        model.forward()
        before = control.stat()
        self.assertEqual(model._fixed_noise_mode, "clean")
        _write_mode(control, "noisy", 0)
        os.utime(control, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(control.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertNotEqual(control.stat().st_ino, before.st_ino)
        self.assertTrue(torch.equal(model.forward(), noisy))

    def test_preparation_is_atomic_reusable_and_checks_tied_source(self):
        from safetensors.torch import save_file

        model = self.root / "local-model"
        model.mkdir()
        config = dict(model_type="qwen3", num_hidden_layers=1, tie_word_embeddings=True)
        (model / "config.json").write_text(json.dumps(config))
        tensors = {name: torch.ones(2, 2, dtype=torch.bfloat16) for name in prepare_bank.expected_names(config)}
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
        source_file = model / "model.safetensors"
        save_file(tensors, str(source_file))
        bank_path = self.root / "new-bank"
        state = torch.get_rng_state().clone()
        first = prepare_bank.prepare(model, bank_path, 31)
        manifest_hash = file_sha(bank_path / "manifest.json")
        second = prepare_bank.prepare(model, bank_path, 31)
        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(file_sha(bank_path / "manifest.json"), manifest_hash)
        bank = FixedNoiseBank(bank_path / "manifest.json", 0.05)
        for name, initial in tensors.items():
            canonical = bank.canonical(name)
            _, actual, _ = bank.transform(name, initial)
            expected = (initial.float() + 0.05 * prepare_bank.make_eta(initial, 31, canonical)).bfloat16()
            self.assertTrue(torch.equal(actual, expected))
        tensors["lm_head.weight"].add_(1)
        save_file(tensors, str(source_file))
        with self.assertRaisesRegex(ValueError, "different source"):
            prepare_bank.prepare(model, bank_path, 31)
        self.assertEqual(file_sha(bank_path / "manifest.json"), manifest_hash)
        with self.assertRaisesRegex(ValueError, "Tied source alias is unequal"):
            prepare_bank.prepare(model, self.root / "unequal-alias-bank", 31)

    def test_bank_tampering_is_rejected(self):
        manifest = json.loads(self.manifest.read_text())
        target = self.root / next(iter(manifest["tensors"].values()))["file"]
        with target.open("ab") as stream:
            stream.write(b"changed")
        bank = FixedNoiseBank(self.manifest, 0.05)
        with self.assertRaises(ValueError):
            bank.transform("model.embed_tokens.weight", self.clean["model.embed_tokens.weight"])


if __name__ == "__main__":
    unittest.main()
