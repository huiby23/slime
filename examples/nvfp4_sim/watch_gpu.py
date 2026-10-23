"""Wait for >5 minutes of observed GPU idleness, then run the prepared NVFP4 experiment.

Standard-library-only host process. Persistent state/logs live in --output.
Each retry gets a fresh directory. Stop only containers carrying our watch label;
never signal a process belonging to another user. A STOP file cancels the watcher.
"""

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOCAL_TIMEZONE = timezone(timedelta(hours=8))
LABEL = "lab.nvfp4-watch"
VALIDATION_CHECKS = (
    "runtime_completed",
    "training_metrics_complete_and_finite",
    "two_active_optimizer_updates",
    "checkpoints_present",
    "weight_syncs_completed",
    "final_evaluation_completed",
    "ray_cleanup_completed",
)


def stamp():
    return datetime.now(LOCAL_TIMEZONE).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def command(argv):
    return subprocess.check_output(argv, text=True, timeout=30).strip()


def snapshot():
    def query(fields):
        return list(csv.reader(command(["nvidia-smi", fields, "--format=csv,noheader,nounits"]).splitlines()))

    processes = {}
    for uuid, pid in query("--query-compute-apps=gpu_uuid,pid"):
        processes.setdefault(uuid.strip(), []).append(int(pid.strip()))
    cards = []
    for index, uuid, name, memory, utilization in query("--query-gpu=index,uuid,name,memory.used,utilization.gpu"):
        uuid = uuid.strip()
        memory, utilization = int(memory.strip()), int(utilization.strip())
        pids = processes.get(uuid, [])
        cards.append(
            {
                "index": int(index.strip()),
                "uuid": uuid,
                "name": name.strip(),
                "memory_used_mib": memory,
                "utilization_percent": utilization,
                "compute_pids": pids,
                "idle_now": not pids and memory <= 64 and utilization == 0,
            }
        )
    return cards


class IdleTracker:
    def __init__(self, threshold=300, maximum_gap=20):
        self.threshold = threshold
        self.maximum_gap = maximum_gap
        self.since = {}
        self.last_sample = None

    def reset(self):
        self.since.clear()
        self.last_sample = None

    def update(self, cards, now):
        if self.last_sample is not None and (now < self.last_sample or now - self.last_sample > self.maximum_gap):
            self.since.clear()
        self.last_sample = now
        present = {card["uuid"] for card in cards}
        self.since = {uuid: start for uuid, start in self.since.items() if uuid in present}
        for card in cards:
            uuid = card["uuid"]
            if card["idle_now"]:
                self.since.setdefault(uuid, now)
            else:
                self.since.pop(uuid, None)
            elapsed = now - self.since[uuid] if uuid in self.since else 0
            card["idle_observed_seconds"] = round(elapsed, 1)
            card["qualified"] = uuid in self.since and elapsed > self.threshold
        return sorted(
            (card for card in cards if card["qualified"]),
            key=lambda card: (-card["idle_observed_seconds"], card["index"]),
        )


def prepare_attempt(template, directory, gpu, name, watch_id, simulator_commit):
    """Reuse the validated CLI; change only GPU assignment and host output path."""
    config = json.loads((template / "config.json").read_text())
    manifest = json.loads((template / "manifest.json").read_text())
    assert json.loads((template / "configuration-check.json").read_text())["passed"]
    assert config["training"]["updates"] in (2, 100), "Use a two-update smoke or 100-update formal experiment"
    if config["training"]["updates"] == 100:
        assert config.get("experiment_mode") == "formal" and config.get("requires_passed_validation")
    assert not directory.exists(), "Preserve prior attempts"
    directory.mkdir(parents=True)
    config.update(
        run_id="nvfp4-simulation-" + name,
        host_run_dir=str(directory / "results"),
        physical_gpus=[gpu["index"]],
        assigned_gpu_uuids=[gpu["uuid"]],
        gpu_allocation_managed_by_sequence=False,
        wall_time_limit_seconds=config.get("wall_time_limit_seconds", 1800),
    )
    for filename in ("mismatch.yaml", "validation.jsonl", "configuration-check.json", "baseline-comparison.json"):
        if (template / filename).exists():
            (directory / filename).write_bytes((template / filename).read_bytes())
    for relative in ("results/logs", "cache", "tmp/sg-a"):
        (directory / relative).mkdir(parents=True, exist_ok=True)
    for key in ("XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        value = config.get("environment", {}).get(key, "")
        if value.startswith("/nvfp4-run/"):
            (directory / value[len("/nvfp4-run/") :]).mkdir(parents=True, exist_ok=True)
    (directory / "cache/xdg/torch/kernels").mkdir(parents=True, exist_ok=True)
    save(directory / "config.json", config)
    docker = copy.deepcopy(manifest["docker_command"])
    docker[docker.index("--gpus") + 1] = "device=" + gpu["uuid"]
    found_mount = False
    for index, value in enumerate(docker):
        if value == "--mount" and "dst=/nvfp4-run" in docker[index + 1].split(","):
            docker[index + 1] = "type=bind,src=%s,dst=/nvfp4-run" % directory
            found_mount = True
    assert found_mount, "Validated output mount missing"
    assert docker[:2] == ["docker", "run"]
    assert "--name" not in docker, "Use the unchanged preflight template, not a prior attempt"
    docker[2:2] = ["--name", name, "--label", LABEL + "=" + watch_id]
    manifest.update(
        docker_command=docker,
        idle_observation=gpu,
        prepared_at=stamp(),
        template_simulator_commit=manifest["simulator_commit"],
        simulator_commit=simulator_commit,
    )
    save(directory / "manifest.json", manifest)
    return docker


def validation_dependency(watch_directory, template, repo):
    """Fail closed until the same recipe has passed the complete two-step gate."""
    status_path = watch_directory / "status.json"
    if not status_path.exists():
        return {"state": "waiting", "reason": "Validation watcher has not written status"}
    state = json.loads(status_path.read_text())
    if state["state"] in ("failed", "stopped"):
        return {"state": "failed", "reason": "Validation watcher ended with " + state["state"]}
    if state["state"] != "completed":
        return {"state": "waiting", "completed_updates": state.get("completed_updates", 0)}
    try:
        directory = Path(state["current_attempt"]["directory"])
        report = json.loads((directory / "results/training-validation.json").read_text())
        result = json.loads((directory / "results/status.json").read_text())
        previous = json.loads((directory / "config.json").read_text())
        previous_manifest = json.loads((directory / "manifest.json").read_text())
        config = json.loads((template / "config.json").read_text())
        manifest = json.loads((template / "manifest.json").read_text())
        checks = {
            "two_update_gate_passed": state.get("completed_updates") == 2
            and state.get("exit_code") == 0
            and report.get("passed") is True
            and all(report.get("checks", {}).get(key) is True for key in VALIDATION_CHECKS)
            and report.get("verified_rollout_ids") == [0, 1],
            "training_status_passed": result.get("successful_training") is True
            and result.get("training_validation_passed") is True
            and result.get("completed_updates") == 2,
            "same_runtime_and_recipe": previous_manifest["baseline_commit"] == manifest["baseline_commit"]
            and previous_manifest["recipe"] == manifest["recipe"]
            and previous["image_id"] == config["image_id"],
            "same_training_dimensions": all(
                previous["training"][key] == config["training"][key]
                for key in (
                    "model",
                    "prompts_per_update",
                    "samples_per_prompt",
                    "global_batch_size",
                    "response_token_limit",
                    "seed",
                    "learning_rate",
                    "importance_correction",
                )
            ),
            "same_quantizer_configuration": (directory / "mismatch.yaml").read_bytes()
            == (template / "mismatch.yaml").read_bytes(),
        }
        command(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--exit-code",
                "--quiet",
                previous_manifest["simulator_commit"],
                "--",
                "slime_plugins/nvfp4_sim",
            ]
        )
        checks["same_simulator_source"] = True
        return {
            "state": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "watch_directory": str(watch_directory),
            "validated_attempt": str(directory),
            "validated_simulator_commit": previous_manifest["simulator_commit"],
        }
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        return {"state": "failed", "reason": "Cannot establish completed validation provenance: " + str(exc)}


def stop_own_container(name, watch_id):
    result = subprocess.run(["docker", "inspect", name], text=True, capture_output=True, timeout=15)
    if result.returncode:
        if "No such" in result.stderr:
            return
        raise RuntimeError(result.stderr)
    info = json.loads(result.stdout)[0]
    if info["Config"].get("Labels", {}).get(LABEL) != watch_id:
        raise RuntimeError("Container ownership label mismatch: " + name)
    result = subprocess.run(["docker", "stop", "--timeout", "15", name], text=True, timeout=30, capture_output=True)
    if result.returncode and "No such" not in result.stderr:
        raise RuntimeError(result.stderr)


def foreign_pids(name, card, known_owned):
    """Query after the GPU snapshot, so newly spawned own workers are included."""
    info = json.loads(command(["docker", "inspect", name]))[0]
    container_id = info["Id"]
    rows = command(["docker", "top", name, "-eo", "pid"]).splitlines()[1:]
    known_owned.clear()
    known_owned.update(int(row.strip()) for row in rows if row.strip())
    foreign = []
    for pid in card["compute_pids"]:
        if pid in known_owned:
            continue
        try:
            cgroup = Path("/proc/%s/cgroup" % pid).read_text()
        except FileNotFoundError:
            continue
        except PermissionError:
            cgroup = ""
        if container_id in cgroup:
            known_owned.add(pid)
        else:
            foreign.append(pid)
    return foreign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-seconds", type=float, default=300)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument(
        "--after-validation", type=Path, help="Require this smoke watcher's full success before launch"
    )
    args = parser.parse_args()
    if args.idle_seconds < 300 or not 1 <= args.poll_seconds <= 10:
        parser.error("Require at least 300 idle seconds and a 1-10 second poll interval")
    root, template = args.output.resolve(), args.template.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "watch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "STOP").exists():
        raise RuntimeError("STOP is present; choose a fresh watch directory to restart")
    if (root / "status.json").exists():
        raise RuntimeError("Watch state already exists; choose a fresh directory to preserve it")
    assert json.loads((template / "configuration-check.json").read_text())["passed"]
    config = json.loads((template / "config.json").read_text())
    planned = config["training"]["updates"]
    assert planned in (2, 100)
    if config.get("requires_passed_validation"):
        assert args.after_validation is not None, "Formal runs require --after-validation"
    repo = Path(__file__).resolve().parents[2]
    baseline_source = Path("/data2/huibingyu/slime-v040/code/slime")
    manifest = json.loads((template / "manifest.json").read_text())
    assert command(["git", "-C", str(baseline_source), "rev-parse", "HEAD"]) == manifest["baseline_commit"]
    source_files = list((repo / "slime_plugins/nvfp4_sim").rglob("*.py"))
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    watch_id = "%s-%s" % (root.name, os.getpid())
    tracker = IdleTracker(args.idle_seconds, max(20, args.poll_seconds * 3))
    state = {
        "state": "waiting_for_validation" if args.after_validation else "waiting_for_idle_gpu",
        "pid": os.getpid(),
        "watch_id": watch_id,
        "started_at": stamp(),
        "idle_threshold_seconds": args.idle_seconds,
        "poll_interval_seconds": args.poll_seconds,
        "criterion": "No compute PIDs, memory <=64 MiB, utilization 0%, strictly >300 observed seconds",
        "planned_updates": planned,
        "experiment_mode": config.get("experiment_mode", "smoke"),
        "completed_updates": 0,
        "attempts": [],
        "simulator_commit": command(["git", "-C", str(repo), "rev-parse", "HEAD"]),
    }
    (root / "watch.pid").write_text(str(os.getpid()) + "\n")
    save(root / "status.json", state)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    def event(kind, **details):
        with (root / "events.jsonl").open("a") as log:
            log.write(json.dumps({"at": stamp(), "event": kind, **details}) + "\n")
        print(stamp(), kind, json.dumps(details), flush=True)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    process = log_handle = active_name = active_dir = active_gpu = None
    owned_pids = set()
    dependency_passed = args.after_validation is None
    event("monitor_started", threshold_seconds=args.idle_seconds)
    try:
        while not stopping and not (root / "STOP").exists():
            candidates = []
            try:
                cards = snapshot()
                candidates = tracker.update(cards, time.monotonic())
                state["gpus"] = cards
                state.pop("query_error", None)
            except Exception as exc:
                tracker.reset()
                state.update(gpus=[], query_error=str(exc))
            state["updated_at"] = stamp()
            if not dependency_passed:
                dependency = validation_dependency(args.after_validation.resolve(), template, repo)
                state["validation_dependency"] = dependency
                if dependency["state"] == "failed":
                    state.update(state="validation_failed", finished_at=stamp())
                    event("validation_dependency_failed", dependency=dependency)
                    break
                dependency_passed = dependency["state"] == "passed"
                if dependency_passed:
                    save(root / "validation-dependency.json", dependency)
                    state["state"] = "waiting_for_idle_gpu"
                    event("validation_dependency_passed", dependency=dependency)
                else:
                    candidates = []
            if process is not None:
                status_file = active_dir / "results/status.json"
                if status_file.exists():
                    result = json.loads(status_file.read_text())
                    state.update(job_state=result["state"], completed_updates=result.get("completed_updates", 0))
                if process.poll() is None:
                    card = next((card for card in state["gpus"] if card["uuid"] == active_gpu["uuid"]), None)
                    if card is not None:
                        try:
                            foreign = foreign_pids(active_name, card, owned_pids)
                        except subprocess.CalledProcessError:
                            foreign = []  # Container is starting/exiting; the supervisor also checks before CUDA init.
                        if foreign:
                            event("gpu_contention", gpu=active_gpu["index"], foreign_pids=foreign)
                            stop_own_container(active_name, watch_id)
                            process.wait(timeout=30)
                            state["attempts"][-1].update(
                                outcome="gpu_contention", foreign_pids=foreign, finished_at=stamp()
                            )
                            log_handle.close()
                            process = log_handle = active_name = None
                            tracker.reset()
                            state["state"] = "waiting_for_idle_gpu"
                if process is not None and process.poll() is not None:
                    code = process.returncode
                    log_handle.close()
                    result = json.loads(status_file.read_text()) if status_file.exists() else {}
                    state["attempts"][-1].update(exit_code=code, finished_at=stamp())
                    deferred = (active_dir / "results/allocation-deferral.json").exists()
                    if deferred:
                        state["attempts"][-1]["outcome"] = "allocation_deferred"
                        event("allocation_deferred", gpu=active_gpu["index"])
                        tracker.reset()
                        state["state"] = "waiting_for_idle_gpu"
                    else:
                        success = (
                            code == 0
                            and result.get("successful_training")
                            and result.get("completed_updates") == planned
                        )
                        if json.loads((active_dir / "config.json").read_text()).get("require_training_validation"):
                            report_path = active_dir / "results/training-validation.json"
                            validation = (
                                json.loads(report_path.read_text()) if report_path.exists() else {"passed": False}
                            )
                            state["training_validation"] = validation
                            success = success and validation["passed"]
                        state.update(state="completed" if success else "failed", finished_at=stamp(), exit_code=code)
                        state["attempts"][-1]["outcome"] = state["state"]
                        event(state["state"], exit_code=code, completed_updates=result.get("completed_updates", 0))
                        save(root / "status.json", state)
                        process = active_name = log_handle = None
                        break
                    process = active_name = log_handle = None
            elif candidates:
                gpu = candidates[0]
                current = next(card for card in snapshot() if card["uuid"] == gpu["uuid"])
                if not current["idle_now"]:
                    tracker.reset()
                    continue
                for path, expected in source_hashes.items():
                    assert (
                        hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
                    ), "Simulator changed while waiting"
                number = len(state["attempts"]) + 1
                active_name = "huibingyu-nvfp4-watch-%s-%03d" % (os.getpid(), number)
                active_dir = root / ("attempt-%03d" % number)
                argv = prepare_attempt(template, active_dir, gpu, active_name, watch_id, state["simulator_commit"])
                log_handle = (active_dir / "launch.log").open("w")
                process = subprocess.Popen(argv, stdout=log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                active_gpu, owned_pids = gpu, set()
                attempt = {
                    "directory": str(active_dir),
                    "container": active_name,
                    "physical_gpu": gpu["index"],
                    "gpu_uuid": gpu["uuid"],
                    "launch_idle_seconds": gpu["idle_observed_seconds"],
                    "started_at": stamp(),
                }
                state["attempts"].append(attempt)
                state.update(state="running", current_attempt=attempt, completed_updates=0)
                event("experiment_started", **attempt)
            save(root / "status.json", state)
            for _ in range(int(args.poll_seconds * 10)):
                if stopping or (root / "STOP").exists():
                    break
                time.sleep(0.1)
    except Exception as exc:
        state.update(state="failed", error=str(exc), finished_at=stamp())
        event("failed", error=str(exc))
        raise
    finally:
        try:
            if active_name is not None:
                stop_own_container(active_name, watch_id)
            if process is not None:
                process.wait(timeout=30)
        except Exception as exc:
            state.update(state="failed", cleanup_error=str(exc), finished_at=stamp())
            event("cleanup_failed", error=str(exc))
        finally:
            if log_handle is not None:
                log_handle.close()
            if (stopping or (root / "STOP").exists()) and "cleanup_error" not in state:
                state.update(state="stopped", finished_at=stamp())
            save(root / "status.json", state)


if __name__ == "__main__":
    main()
