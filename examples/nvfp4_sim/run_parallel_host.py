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
        finished_at=stamp(),
    )
    save(root / "status.json", status)


if __name__ == "__main__":
    main()
