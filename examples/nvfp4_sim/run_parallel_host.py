"""Launch the prepared DP4 container and preserve an inspectable host status."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from prepare_baseline import assert_gpu_idle
from watch_gpu import save, stamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    config = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert json.loads((root / "configuration-check.json").read_text())["passed"]
    assert not (root / "results/status.json").exists()
    for gpu in config["physical_gpus"]:
        if config.get("reservation_host_pids"):
            pids = subprocess.check_output(
                ["nvidia-smi", "--id=" + str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                text=True,
            )
            assert {int(line) for line in pids.splitlines() if line.strip()} <= set(config["reservation_host_pids"])
        else:
            assert_gpu_idle(gpu)
    base = config["resume_checkpoint"]["completed_updates"]
    status = {
        "state": "starting",
        "pid": os.getpid(),
        "started_at": stamp(),
        "planned_updates": 100,
        "completed_updates": base,
        "resumed_from_updates": base,
        "physical_gpus": config["physical_gpus"],
        "checkpoint_policy": "best-last",
        "current_attempt": {"directory": str(root), "container": manifest["container"]},
    }
    save(root / "status.json", status)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (root / "launch.log").open("w") as log:
        process = subprocess.Popen(
            manifest["docker_command"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
        )
        while process.poll() is None:
            ready = root / "results/gpu-contexts-ready.json"
            if ready.exists() and config.get("reservation_containers") and not status.get("gpu_handoff_completed"):
                for name in config["reservation_containers"]:
                    info = json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]
                    assert info["Config"]["Labels"].get("lab.nvfp4-reservation") == "20261025"
                    subprocess.run(["docker", "stop", "--timeout", "5", name], check=True)
                status["gpu_handoff_completed"] = True
            path = root / "results/status.json"
            if path.exists():
                runtime = json.loads(path.read_text())
                status.update(state=runtime["state"], completed_updates=max(base, runtime["completed_updates"]))
            policy = root / "results/checkpoints/checkpoint-policy.json"
            if policy.exists():
                status["checkpoints"] = json.loads(policy.read_text())
            status["updated_at"] = stamp()
            save(root / "status.json", status)
            if stopping:
                info = json.loads(subprocess.check_output(["docker", "inspect", manifest["container"]], text=True))[0]
                assert info["Config"]["Labels"].get("lab.nvfp4-parallel") == root.name
                subprocess.run(["docker", "stop", "--timeout", "30", manifest["container"]], check=True)
                break
            time.sleep(5)
        process.wait()
    runtime = json.loads((root / "results/status.json").read_text()) if (root / "results/status.json").exists() else {}
    status.update(
        state="completed" if process.returncode == 0 and runtime.get("successful_training") else "failed",
        completed_updates=max(base, runtime.get("completed_updates", 0)),
        exit_code=process.returncode,
        trainer_exit_code=runtime.get("trainer_exit_code"),
        finished_at=stamp(),
    )
    save(root / "status.json", status)
    if status["state"] == "failed" and not stopping:
        # Keep the user's assigned GPUs visible for recovery after a trainer
        # failure. This retains contexts; it does not provide exclusive access.
        name = "huibingyu-nvfp4-reserve-" + root.name
        code = (
            "import torch,time; "
            "pins=[torch.empty(16*1024*1024,device=f'cuda:{i}',dtype=torch.float32) "
            "for i in range(torch.cuda.device_count())]; "
            "print('Training failed; GPU contexts retained for recovery',flush=True); "
            "exec('while True: time.sleep(5)')"
        )
        command = [
            "docker", "run", "-d", "--rm", "--init", "--name", name,
            "--label", "lab.nvfp4-reservation=20261025", "--network", "none",
            "--gpus", '"device=' + ",".join(config["assigned_gpu_uuids"]) + '"',
            "--entrypoint", "python3", config["image_id"], "-u", "-c", code,
        ]
        try:
            status["recovery_reservation_id"] = subprocess.check_output(command, text=True).strip()
            status["recovery_reservation_container"] = name
        except Exception as exc:
            status["recovery_reservation_error"] = str(exc)
        save(root / "status.json", status)


if __name__ == "__main__":
    main()
