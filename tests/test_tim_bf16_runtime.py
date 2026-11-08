"""CPU protocol checks: python3 -m unittest discover -s tests -p test_tim_bf16_runtime.py.

These tests need only Python's standard library. No GPU, Slime import, Ray,
Megatron, or Torch installation is required.
"""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1] / "slime_plugins/tim/bf16"


def load(name):
    spec = importlib.util.spec_from_file_location("tim_bf16_" + name, BASE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver, policy, validation = load("train"), load("checkpoint_policy"), load("training_validation")


def arguments(lr=1e-6, **overrides):
    values = dict(
        bf16=True,
        fp8=None,
        fp4=None,
        optimizer="adam",
        lr=lr,
        min_lr=lr,
        lr_decay_style="constant",
        lr_warmup_iters=0,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.98,
        save_interval=1,
        async_save=False,
        no_save_optim=False,
        no_save_rng=False,
        num_rollout=500,
        eval_interval=25,
        skip_eval_before_train=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class Adam:
    adam_w_mode = 1
    config = SimpleNamespace(decoupled_weight_decay=True)

    def __init__(self, args):
        self.param_groups = [
            dict(
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(args.adam_beta1, args.adam_beta2),
                wd_mult=1,
                params=[object()],
            ),
            dict(lr=args.lr, weight_decay=0, betas=(args.adam_beta1, args.adam_beta2), wd_mult=0, params=[object()]),
        ]


def pending(root, rollout_id, complete=True):
    path = root / (".pending-" + str(rollout_id))
    model = path / ("iter_%07d" % rollout_id)
    model.mkdir(parents=True)
    (path / "latest_checkpointed_iteration.txt").write_text(str(rollout_id))
    (model / ".metadata").write_text("full model/optimizer/RNG metadata")
    (model / "0.distcp").write_text("full model/optimizer/RNG payload")
    if complete:
        (path / "rollout").mkdir()
        (path / "rollout" / ("global_dataset_state_dict_%d.pt" % rollout_id)).write_text("cursor")
    return path


class OptimizerContractTests(unittest.TestCase):
    def test_each_baseline_lr_and_native_zero_decay_group(self):
        for lr in (1e-6, 1e-5, 1e-3):
            with self.subTest(lr=lr):
                args = arguments(lr)
                report = validation.inspect_optimizer(Adam(args), args)
                self.assertEqual([group["lr"] for group in report["param_groups"]], [lr, lr])
                self.assertEqual([group["weight_decay"] for group in report["param_groups"]], [0.01, 0])

    def test_betas_decay_and_group_multipliers_follow_arguments(self):
        args = arguments(3e-5, weight_decay=0.02, adam_beta1=0.8, adam_beta2=0.999)
        optimizer = Adam(args)
        optimizer.param_groups[0].update(lr=6e-5, lr_mult=2, weight_decay=0.01, wd_mult=0.5)
        report = validation.inspect_optimizer(optimizer, args)
        self.assertEqual(report["param_groups"][0]["betas"], [0.8, 0.999])
        self.assertEqual(report["param_groups"][0]["lr_mult"], 2)
        for field, wrong in [("lr", 1e-6), ("betas", (0.9, 0.98)), ("weight_decay", 0.1)]:
            with self.subTest(field=field):
                previous = optimizer.param_groups[0][field]
                optimizer.param_groups[0][field] = wrong
                with self.assertRaises(AssertionError):
                    validation.inspect_optimizer(optimizer, args)
                optimizer.param_groups[0][field] = previous

    def test_reject_nonconstant_lr_quantized_trainer_and_coupled_adam(self):
        for override in [
            dict(lr_decay_style="cosine"),
            dict(lr_warmup_iters=1),
            dict(lr_warmup_fraction=0.01),
            dict(fp4="e2m1"),
            dict(bf16=False),
            dict(optimizer="sgd"),
            dict(min_lr=0),
            dict(weight_decay=-0.1),
        ]:
            with self.subTest(override=override), self.assertRaises(AssertionError):
                validation.expected_optimizer_settings(arguments(**override))
        args = arguments()
        optimizer = Adam(args)
        optimizer.config = SimpleNamespace(decoupled_weight_decay=False)
        with self.assertRaises(AssertionError):
            validation.inspect_optimizer(optimizer, args)


class PeriodicProtocolTests(unittest.TestCase):
    def test_early_eval_never_enables_same_interval_checkpoint_branch(self):
        seen = []

        def native(rollout_id, interval, per_epoch=None, total=None):
            seen.append((rollout_id, interval, per_epoch, total))
            return (rollout_id + 1) % interval == 0

        periodic = driver.make_periodic_action(native, 25, 10)
        self.assertTrue(periodic(0, 25, 100))
        self.assertFalse(periodic(0, 25, 100, 500))
        self.assertFalse(periodic(10, 25, 100))
        self.assertTrue(periodic(24, 25, 100))
        self.assertEqual(len(seen), 4)

    def test_preflight_checks_exact_native_ast_and_complete_checkpoint_flags(self):
        source = """def train(args, num_rollout_per_epoch, rollout_id):
    should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout)
    should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch)
"""
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"TIM_EARLY_EVAL_UPDATES": "10", "TIM_RETAIN_EARLY_CHECKPOINTS": "10"}
        ):
            path = Path(temporary) / "train.py"
            path.write_text(source)
            report = driver.runtime_protocol_check(arguments(), path)
            self.assertTrue(report["native_periodic_calls"]["passed"])
            self.assertEqual(
                report["evaluation_completed_updates_without_epoch_triggers"][:12], list(range(11)) + [25]
            )
            path.write_text(source.replace("num_rollout_per_epoch, args.num_rollout", "num_rollout_per_epoch"))
            with self.assertRaises(RuntimeError):
                driver.runtime_protocol_check(arguments(), path)
            for override in [dict(no_save_optim=True), dict(no_save_rng=True), dict(async_save=True)]:
                with self.subTest(override=override), self.assertRaises(AssertionError):
                    driver.runtime_protocol_check(arguments(**override))
            with patch.dict(os.environ, {"TIM_RETAIN_EARLY_CHECKPOINTS": "9"}):
                with self.assertRaises(AssertionError):
                    driver.runtime_protocol_check(arguments())


class CheckpointRetentionTests(unittest.TestCase):
    def test_initial_and_first_update_axes_and_pinned_early_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"TIM_RETAIN_EARLY_CHECKPOINTS": "2"}):
            root = Path(temporary)
            initial = policy.record_evaluation(root, 0, 0.6, 512)
            self.assertEqual(initial["latest_evaluation"]["completed_updates"], 0)
            self.assertFalse((root / "best").exists())
            policy.publish_last(root, pending(root, 0), 0)
            first = policy.record_evaluation(root, 0, 0.7, 512)
            self.assertEqual(first["latest_evaluation"]["completed_updates"], 1)
            policy.publish_last(root, pending(root, 1), 1)
            policy.publish_last(root, pending(root, 2), 2)
            policy.publish_last(root, pending(root, 3), 3)
            self.assertFalse((root / ".snapshots/step_0000002").exists())
            self.assertEqual((root / "early/step_0000001").resolve(), (root / ".snapshots/step_0000000"))
            self.assertEqual((root / "early/step_0000002").resolve(), (root / ".snapshots/step_0000001"))
            policy.record_evaluation(root, 3, 0.8, 512)
            self.assertEqual((root / "best").resolve(), (root / "last").resolve())
            self.assertEqual(len(list((root / ".snapshots").iterdir())), 3)

    def test_incomplete_snapshot_cannot_replace_last_or_pin_early(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"TIM_RETAIN_EARLY_CHECKPOINTS": "2"}):
            root = Path(temporary)
            policy.publish_last(root, pending(root, 0), 0)
            previous = (root / "last").resolve()
            with self.assertRaises(AssertionError):
                policy.publish_last(root, pending(root, 1, complete=False), 1)
            self.assertEqual((root / "last").resolve(), previous)
            self.assertFalse((root / "early/step_0000002").exists())
            state = json.loads((root / "checkpoint-policy.json").read_text())
            self.assertEqual(state["last"]["completed_updates"], 1)


if __name__ == "__main__":
    unittest.main()
