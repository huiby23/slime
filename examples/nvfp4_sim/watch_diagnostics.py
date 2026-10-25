"""Run prepared diagnostic jobs on individually idle GPUs; preserve running training."""

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def free_gpus(reservations):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    free = []
    for row in output.splitlines():
        index, uuid, memory, utilization = [x.strip() for x in row.split(",")]
        allowed = set(reservations.get(uuid, {}).get("pids", []))
        if int(memory) > (2048 if allowed else 64) or int(utilization) != 0:
            continue
        pids = subprocess.check_output(["nvidia-smi", "--id=" + uuid, "--query-compute-apps=pid",
                                       "--format=csv,noheader,nounits"], text=True).strip()
        if not ({int(pid) for pid in pids.splitlines() if pid.strip()} - allowed):
            free.append((int(index), uuid))
    return free


def docker_command(template, name, gpu_uuid, job_id, script):
    command = copy.deepcopy(template)
    command.remove("-d")
    command[command.index("--name") + 1] = name
    command[command.index("--gpus") + 1] = '"device=' + gpu_uuid + '"'
    for index, token in enumerate(command):
        if token == "--env" and command[index + 1].startswith("CUDA_VISIBLE_DEVICES="):
            command[index + 1] = "CUDA_VISIBLE_DEVICES=0"
    index = command.index("--entrypoint")
    image = command[index + 2]
    command[index:] = ["--label", "lab.nvfp4-diagnostic-job=" + job_id,
                      "--env", "TMPDIR=/diagnostic/jobs/" + job_id + "/tmp",
                      "--env", "XDG_CACHE_HOME=/diagnostic/cache/xdg",
                      "--env", "TORCHINDUCTOR_CACHE_DIR=/diagnostic/cache/torchinductor",
                      "--env", "TRITON_CACHE_DIR=/diagnostic/cache/triton",
                      "--entrypoint", "python3", image, script, "/diagnostic/jobs/" + job_id + "/job.json"]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    for job in plan["jobs"]:
        assert json.loads((root / "jobs" / job["job_id"] / "configuration-check.json").read_text())["passed"]
    template = json.loads((root / "cpu-container.json").read_text())["docker_command"]
    status = {"state": "watching", "pid": os.getpid(), "started_at": time.time(),
              "jobs": {job["job_id"]: {"state": "queued", "kind": job["kind"]} for job in plan["jobs"]}}
    save(root / "status.json", status)
    running = {}
    reservations = {item["uuid"]: item for item in plan.get("reservations", [])}
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            formal = Path(plan["formal_run"])
            policy = json.loads((formal / "results/checkpoints/checkpoint-policy.json").read_text())
            if policy["last"]["completed_updates"] >= 100 and not (root / "hf/nvfp4-step100/export-validation.json").exists():
                snapshot = root / "snapshots/nvfp4-step100"
                if not snapshot.exists():
                    def copy_or_link(src, dst):
                        try:
                            return os.link(src, dst)
                        except OSError:
                            return shutil.copy2(src, dst)
                    shutil.copytree((formal / "results/checkpoints/last").resolve(), snapshot, copy_function=copy_or_link)
                with (root / "export-nvfp4-100.log").open("w") as log:
                    result = subprocess.run(["docker", "exec", plan["cpu_container"], "python3",
                        "/quant/examples/nvfp4_sim/export_diagnostic_hf.py", "--input",
                        "/diagnostic/snapshots/nvfp4-step100/iter_0000099", "--origin", plan["origin"],
                        "--output", "/diagnostic/hf/nvfp4-step100"], stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError("Final checkpoint export failed; inspect export-nvfp4-100.log")
            for job_id, (process, log, name) in list(running.items()):
                if process.poll() is None:
                    child_path = root / "jobs" / job_id / "status.json"
                    if child_path.exists():
                        status["jobs"][job_id]["progress"] = json.loads(child_path.read_text())
                        uuid = status["jobs"][job_id]["gpu_uuid"]
                        reservation = reservations.pop(uuid, None)
                        if reservation:
                            subprocess.run(["docker", "stop", "--timeout", "5", reservation["container"]], check=True)
                    continue
                log.close()
                record = status["jobs"][job_id]
                if process.returncode == 75:
                    record.update(state="queued", reason="GPU became busy before CUDA initialization")
                else:
                    record.update(state="completed" if process.returncode == 0 else "failed",
                                  exit_code=process.returncode, finished_at=time.time())
                del running[job_id]
            available = free_gpus(reservations)
            active_gpus = {status["jobs"][key]["physical_gpu"] for key in running}
            available = [gpu for gpu in available if gpu[0] not in active_gpus]
            for job in plan["jobs"]:
                record = status["jobs"][job["job_id"]]
                if record["state"] != "queued":
                    continue
                if job.get("export_dependency") and not (root / job["export_dependency"]).exists():
                    record["reason"] = "waiting for frozen model export"
                    continue
                if not available:
                    break
                gpu, uuid = available.pop(0)
                job_path = root / "jobs" / job["job_id"] / "job.json"
                config = json.loads(job_path.read_text())
                config.update(physical_gpu=gpu, expected_gpu_uuid=uuid)
                config["reservation_host_pids"] = reservations.get(uuid, {}).get("pids", [])
                save(job_path, config)
                name = "huibingyu-nvfp4-diag-" + job["job_id"]
                command = docker_command(template, name, uuid, job["job_id"], job["script"])
                save(job_path.parent / "launch.json", {"docker_command": command})
                log = (job_path.parent / "launch.log").open("w")
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                running[job["job_id"]] = process, log, name
                record.update(state="running", physical_gpu=gpu, gpu_uuid=uuid, container=name, started_at=time.time())
                record.pop("reason", None)
            unfinished = any(v["state"] in ("queued", "running") for v in status["jobs"].values())
            status.update(state="running" if running else ("waiting_for_idle_gpu" if unfinished else "completed"),
                          updated_at=time.time())
            save(root / "status.json", status)
            if not unfinished:
                break
            time.sleep(5)
    finally:
        for process, log, name in running.values():
            info = json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]
            assert info["Config"]["Labels"].get("lab.nvfp4-diagnostics") == "20261025"
            subprocess.run(["docker", "stop", "--timeout", "30", name], check=True)
            process.wait()
            log.close()
        if stopping:
            status.update(state="stopped", updated_at=time.time())
            save(root / "status.json", status)


if __name__ == "__main__":
    main()
