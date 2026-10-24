"""Best/last retention must preserve the previous complete checkpoint on failure."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "policy", Path(__file__).resolve().parents[1] / "slime_plugins/nvfp4_sim/checkpoint_policy.py"
)
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


def pending(root, step, complete=True):
    path = root / (".pending-%d" % step)
    (path / ("iter_%07d" % step)).mkdir(parents=True)
    (path / "latest_checkpointed_iteration.txt").write_text(str(step))
    (path / ("iter_%07d" % step) / ".metadata").write_text("complete model metadata")
    (path / ("iter_%07d" % step) / "0.distcp").write_text("optimizer and parameters")
    if complete:
        (path / "rollout").mkdir()
        (path / "rollout" / ("global_dataset_state_dict_%d.pt" % step)).write_text("dataset cursor")
    return path


class CheckpointPolicyTests(unittest.TestCase):
    def test_only_best_and_last_payloads_remain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy.publish_last(root, pending(root, 0), 0)
            policy.record_evaluation(root, 0, 0.5, 512)
            policy.publish_last(root, pending(root, 1), 1)
            policy.record_evaluation(root, 1, 0.4, 512)
            self.assertEqual((root / "best").resolve().name, "step_0000000")
            self.assertEqual((root / "last").resolve().name, "step_0000001")
            policy.publish_last(root, pending(root, 2), 2)
            self.assertFalse((root / ".snapshots/step_0000001").exists())
            policy.record_evaluation(root, 2, 0.6, 512)
            self.assertEqual((root / "best").resolve(), (root / "last").resolve())
            self.assertEqual(len(list((root / ".snapshots").iterdir())), 1)

    def test_partial_save_cannot_replace_last(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy.publish_last(root, pending(root, 0), 0)
            previous = (root / "last").resolve()
            with self.assertRaises(AssertionError):
                policy.publish_last(root, pending(root, 1, False), 1)
            self.assertEqual((root / "last").resolve(), previous)
            self.assertTrue((previous / ".metadata").exists() or (previous / "iter_0000000/.metadata").exists())

    def test_initial_evaluation_and_stock_logging_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(save=str(root))
            self.assertFalse(policy.eval_log_hook(0, args, {"gsm8k": {"rewards": [0, 1]}}))
            self.assertFalse((root / "best").exists())
            policy.publish_last(root, pending(root, 2), 2)
            policy.eval_log_hook(2, args, {"gsm8k": {"rewards": [1, 1]}})
            self.assertEqual(json.loads((root / "checkpoint-policy.json").read_text())["best"]["score"], 1.0)
            with self.assertRaises(ValueError):
                policy.record_evaluation(root, 2, float("nan"), 512)


if __name__ == "__main__":
    unittest.main()
