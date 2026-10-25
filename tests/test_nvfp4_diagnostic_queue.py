import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "diagnostic_queue", Path(__file__).parents[1] / "examples/nvfp4_sim/watch_diagnostics.py"
)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class DiagnosticQueueTests(unittest.TestCase):
    def test_own_reservation_is_usable_but_foreign_pid_blocks(self):
        reservation = {"GPU-test": {"pids": [100, 200]}}
        with patch.object(queue.subprocess, "check_output", side_effect=["2, GPU-test, 487, 0\n", "100\n200\n"]):
            self.assertEqual(queue.free_gpus(reservation), [(2, "GPU-test")])
        with patch.object(queue.subprocess, "check_output", side_effect=["2, GPU-test, 487, 0\n", "100\n300\n"]):
            self.assertEqual(queue.free_gpus(reservation), [])

    def test_gpu_job_replaces_hidden_cuda_and_stays_network_isolated(self):
        template = ["docker", "run", "-d", "--name", "cpu", "--gpus", '"device=GPU-old"',
                    "--network", "none", "--env", "CUDA_VISIBLE_DEVICES=", "--entrypoint", "sleep", "image", "infinity"]
        command = queue.docker_command(template, "job", "GPU-new", "id", "/script.py")
        self.assertEqual(command[command.index("--gpus") + 1], '"device=GPU-new"')
        self.assertIn("CUDA_VISIBLE_DEVICES=0", command)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertEqual(command[-2:], ["/script.py", "/diagnostic/jobs/id/job.json"])


if __name__ == "__main__":
    unittest.main()
