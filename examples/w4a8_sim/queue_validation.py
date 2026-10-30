"""Queue small real-backend checks on continuously idle GPUs.

This controller never starts 30B training. Passing these tests is only one
prerequisite for a separate full-size smoke. It never stops foreign processes.
"""

import argparse
import fcntl
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def gpu_rows():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True, timeout=10
    )
    busy = {line.split(",")[0].strip() for line in processes.splitlines() if "," in line}
    result = []
    for line in output.splitlines():
        index, gpu_uuid, memory, util = [s.strip() for s in line.split(",")]
        result.append(
            dict(
                index=int(index),
                uuid=gpu_uuid,
                memory_mib=float(memory),
                utilization=float(util),
                idle=float(memory) <= 64 and float(util) == 0 and gpu_uuid not in busy,
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--runtime", default="huibingyu-slime-v040")
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    lock = (args.work / "validation-queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    image = subprocess.check_output(["docker", "inspect", args.runtime, "--format", "{{.Image}}"], text=True).strip()
    repo = Path(__file__).resolve().parents[2]
    pinned = Path("/data2/huibingyu/slime-v040/code/slime")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    stages = [("rollout-tp2", 2), ("megatron-ep4", 4), ("rollout-tp4", 4)]
    state = dict(
        run_id=run_id,
        phase="waiting_for_idle_gpus",
        formal_training_started=False,
        full_size_smoke_required=True,
        completed=[],
    )
    since, last, reuse = {}, None, set()
    deadline = time.monotonic() + args.hours * 3600
    try:
        for stage, count in stages:
            destination = args.work / ("queued-" + stage + "-" + run_id)
            while True:
                now = time.monotonic()
                if now >= deadline:
                    state.update(phase="expired_waiting_for_resources", pending_stage=stage)
                    return
                try:
                    rows = gpu_rows()
                except Exception as error:
                    since.clear()
                    state.update(phase="waiting_for_gpu_query", error=str(error), pending_stage=stage)
                    atomic_json(args.work / "validation-queue.json", state)
                    time.sleep(5)
                    continue
                if last is not None and now - last > 20:
                    since.clear()
                last = now
                for row in rows:
                    key = row["uuid"]
                    if row["idle"]:
                        since.setdefault(key, now)
                    else:
                        since.pop(key, None)
                        reuse.discard(key)
                    row["continuous_idle_seconds"] = now - since.get(key, now)
                    row["qualified"] = row["idle"] and (row["continuous_idle_seconds"] > 300 or key in reuse)
                qualified = [r for r in rows if r["qualified"]]
                state.update(
                    phase="waiting_for_idle_gpus",
                    pending_stage=stage,
                    gpus=rows,
                    at=datetime.now(timezone.utc).isoformat(),
                )
                atomic_json(args.work / "validation-queue.json", state)
                if len(qualified) >= count:
                    selected = qualified[:count]
                    # Close the observation-to-launch window with a fresh check.
                    fresh = {r["uuid"]: r for r in gpu_rows()}
                    if all(fresh[r["uuid"]]["idle"] for r in selected):
                        break
                time.sleep(5)
            name = "huibingyu-w4a8-" + stage + "-" + run_id
            command = [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--label",
                "lab.w4a8=queued-validation",
                "--gpus",
                '"device=' + ",".join(r["uuid"] for r in selected) + '"',
                "--shm-size",
                "4g",
            ]
            environment = {
                "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(count)),
                "PYTHONPATH": "/quant/slime_plugins:/workspace/slime:/root/Megatron-LM",
                "OMP_NUM_THREADS": "2",
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            for key, value in environment.items():
                command += ["--env", key + "=" + value]
            for source, target in [
                (repo, "/quant"),
                (pinned, "/workspace/slime"),
                (args.work, "/work"),
                (args.tokenizer.resolve(), str(args.tokenizer.resolve())),
            ]:
                command += ["--mount", f"type=bind,src={source},dst={target}"]
            command += ["--workdir", "/workspace/slime", "--entrypoint", "python3", image]
            if stage.startswith("megatron"):
                command += [
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={count}",
                    "/quant/examples/w4a8_sim/validate_megatron.py",
                ]
            else:
                command += [
                    "/quant/examples/w4a8_sim/validate_rollout.py",
                    "--tp",
                    str(count),
                    "--tokenizer",
                    str(args.tokenizer.resolve()),
                ]
            command += ["--output", "/work/" + destination.name]
            container_id = subprocess.check_output(command, text=True).strip()
            state.update(
                phase="running_small_integration",
                stage=stage,
                container=name,
                container_id=container_id,
                selected_gpus=selected,
                command=command,
                output=str(destination),
            )
            atomic_json(args.work / "validation-queue.json", state)
            while True:
                running = (
                    subprocess.check_output(
                        ["docker", "inspect", name, "--format", "{{.State.Running}}"], text=True
                    ).strip()
                    == "true"
                )
                if not running:
                    break
                # Record subsequent foreign contention; never cancel training.
                try:
                    state["observed_gpus"] = gpu_rows()
                except Exception as error:
                    state["observation_error"] = str(error)
                state["at"] = datetime.now(timezone.utc).isoformat()
                atomic_json(args.work / "validation-queue.json", state)
                time.sleep(5)
            with (args.work / (destination.name + ".log")).open("w") as stream:
                subprocess.run(["docker", "logs", name], stdout=stream, stderr=subprocess.STDOUT, check=True)
            code = int(
                subprocess.check_output(["docker", "inspect", name, "--format", "{{.State.ExitCode}}"], text=True)
            )
            if (
                code
                or not (destination / "validation.json").is_file()
                or not json.loads((destination / "validation.json").read_text()).get("passed")
            ):
                raise RuntimeError(f"{stage} failed: container {name}, exit {code}; inspect its log")
            state["completed"].append(dict(stage=stage, output=str(destination), gpus=selected))
            # Continue the same owned validation workload on freshly released GPUs.
            reuse = {r["uuid"] for r in selected}
            since.clear()
            last = None
        state.update(phase="small_integrations_passed", pending_stage=None)
    except Exception as error:
        state.update(phase="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.work / "validation-queue.json", state)


if __name__ == "__main__":
    main()
