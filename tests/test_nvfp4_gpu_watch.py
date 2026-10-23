"""Host-only checks for idle scheduling and ownership-safe cleanup."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
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
            config = {"training": {"updates": 2}, "container_run_dir": "/nvfp4-run/results"}
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
            with self.assertRaises(AssertionError):
                watch.prepare_attempt(template, directory, gpu, "our-container", "our-watch", "new-commit")

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
