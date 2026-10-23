"""Regression checks for the zero-reward smoke's false-success condition."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "nvfp4_run_validation", Path(__file__).resolve().parents[1] / "examples/nvfp4_sim/run_validated.py"
)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


class RunValidationTests(unittest.TestCase):
    def prepare(self, directory):
        config = {"training": {"updates": 2, "prompts_per_update": 8}}
        status = {
            "trainer_exit_code": 0,
            "completed_updates": 2,
            "evaluations": [{"eval/gsm8k": 0.25, "completed_updates": 2}],
            "ray_cleanup_exit_code": 0,
            "successful_training": True,
        }
        (directory / "status.json").write_text(json.dumps(status))
        (directory / "training-metrics.json").write_text(json.dumps([{"train/grad_norm": 1}, {"train/grad_norm": 1}]))
        (directory / "logs").mkdir()
        text = "Timer update_weights end\n" * 3
        for step in range(2):
            text += (
                "rollout_metrics.py:123 - perf %d: {'rollout/zero_std/count_0': 3, 'rollout/zero_std/count_1': 2, 'rollout/truncated_ratio': 0.25}\n"
                % step
            )
            checkpoint = directory / "checkpoints" / ("iter_%07d" % step)
            checkpoint.mkdir(parents=True)
            (checkpoint / "0.distcp").write_text("checkpoint placeholder")
        (directory / "logs/train.log").write_text(text)
        record = {
            "optimizer_success": True,
            "grad_finite": True,
            "global_grad_norm": 1.0,
            "projection_grad_norm": 0.1,
            "master": {"dtype": "torch.float32"},
            "master_delta_finite": True,
            "master_max_abs_delta": 1e-6,
            "master_changed_elements": 20,
            "recipe_counter_deltas": dict.fromkeys(
                ("fprop", "dgrad", "wgrad", "rht_x", "rht_g", "sr_dgrad", "sr_wgrad"), 113
            ),
        }
        records = [dict(copy.deepcopy(record), rollout_id=step) for step in range(2)]
        self.write_records(directory, records)
        return config, records

    def write_records(self, directory, records):
        (directory / "optimizer-validation.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))

    def test_complete_learning_evidence_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self.prepare(root)
            report = validation.validate_run(root, config)
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["rollouts"][0]["mixed_reward_groups"], 3)

    def test_zero_gradients_fail_even_with_parameter_decay_and_clean_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, records = self.prepare(root)
            for record in records:
                record.update(global_grad_norm=0.0, projection_grad_norm=0.0, active_learning_step=True)
            self.write_records(root, records)
            self.assertFalse(validation.validate_run(root, config)["passed"])

    def test_each_required_signal_is_independent(self):
        changes = [
            {"global_grad_norm": float("nan")},
            {"master_max_abs_delta": 0.0},
            {"master": {"dtype": "torch.bfloat16"}},
            {"recipe_counter_deltas": {"fprop": 1}},
            {"optimizer_success": False},
            {"projection_grad_norm": 0.0},
        ]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, records = self.prepare(root)
                records[0].update(change)
                self.write_records(root, records)
                self.assertFalse(validation.validate_run(root, config)["passed"])

    def test_equal_rewards_or_missing_observer_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self.prepare(root)
            path = root / "logs/train.log"
            path.write_text(path.read_text().replace("'rollout/zero_std/count_0': 3", "'rollout/zero_std/count_0': 6"))
            self.assertFalse(validation.validate_run(root, config)["passed"])
            (root / "optimizer-validation.jsonl").unlink()
            self.assertFalse(validation.validate_run(root, config)["passed"])


if __name__ == "__main__":
    unittest.main()
