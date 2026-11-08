"""Isolated container entry: preflight, own Ray lifetime, and passive status.

This script only stops Ray inside its own container. It never contacts a host
Ray cluster or installs packages. Model/data/source mounts are read-only.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/output")


def write(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def optimizer_records():
    path = ROOT / "optimizer-validation.jsonl"
    if not path.exists():
        return []
    # A running worker may be in the middle of appending its last line.
    records = []
    with path.open() as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return records


def status(plan, state, returncode=None):
    records = optimizer_records()
    completed = max((row["rollout_id"] + 1 for row in records if row.get("optimizer_success")), default=0)
    record = dict(
        state=state,
        completed_updates=completed,
        planned_updates=plan["config"]["training"]["updates"],
        optimizer_records=len(records),
        active_optimizer_updates=sum(bool(row.get("active_learning_step")) for row in records),
        finite_optimizer_records=all(row.get("grad_finite") and row.get("master_delta_finite") for row in records),
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
        returncode=returncode,
        source_bundle_sha256=plan["source_bundle_sha256"],
        image_id=plan["image_id"],
        training_precision=plan["config"]["trainer"],
        rollout_precision=plan["config"]["rollout"]["precision"],
        evaluation_precision=plan["config"]["evaluation"]["precision"],
        note="Execution health and model quality are separate; zero-gradient updates remain observable.",
    )
    write(ROOT / "status.json", record)
    return record


def complete_optimizer_records(records, updates):
    return (
        len(records) == updates
        and sorted(row.get("rollout_id", -1) for row in records) == list(range(updates))
        and all(row.get("optimizer_success") and row.get("step_id") == 0 for row in records)
    )


def main(action):
    plan = json.loads((ROOT / "resolved-config.json").read_text())
    if action not in ("preflight", "train"):
        raise ValueError("Unknown container action")
    for key in (
        "HF_HOME",
        "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH",
        "TMPDIR",
        "RAY_TMPDIR",
        "TENSORBOARD_DIR",
        "WANDB_DIR",
    ):
        Path(plan["environment"][key]).mkdir(parents=True, exist_ok=True)
    if action == "preflight":
        from preflight import validate

        report = validate(plan)
        write(ROOT / "configuration-check.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    check = json.loads((ROOT / "configuration-check.json").read_text())
    if (
        not check["passed"]
        or check["source_bundle_sha256"] != plan["source_bundle_sha256"]
        or check["image_id"] != plan["image_id"]
    ):
        raise RuntimeError("Matching final-bundle CPU preflight is required")
    if optimizer_records() or (ROOT / "checkpoints").exists():
        raise RuntimeError("Fresh launch only: refusing to overwrite an existing training trajectory")
    # Ignore inherited cluster addresses; this container owns its Ray processes.
    os.environ.pop("RAY_ADDRESS", None)
    cpus = plan["config"]["runtime"]["cpus"]
    ray_command = [
        "ray",
        "start",
        "--head",
        "--node-ip-address=127.0.0.1",
        "--port=6379",
        "--num-gpus=" + str(plan["required_gpus"]),
        "--num-cpus=" + str(cpus),
        "--object-store-memory=2147483648",
        "--disable-usage-stats",
        "--include-dashboard=false",
    ]
    status(plan, "starting")
    try:
        subprocess.run(ray_command, check=True)
        os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"
        with (ROOT / "logs/train.log").open("w") as log:
            process = subprocess.Popen(plan["training_command"], stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                status(plan, "running")
                time.sleep(5)
            code = process.wait()
        record = status(plan, "validating", code)
        final_eval = False
        evaluation_path = ROOT / "evaluations-by-completed-updates.jsonl"
        if evaluation_path.exists():
            with evaluation_path.open() as stream:
                evaluations = [json.loads(line) for line in stream if line.strip()]
            final_eval = bool(evaluations) and evaluations[-1]["completed_updates"] == record["planned_updates"]
        complete_records = complete_optimizer_records(optimizer_records(), record["planned_updates"])
        passed = code == 0 and complete_records and record["finite_optimizer_records"] and final_eval
        write(
            ROOT / "training-validation.json",
            dict(
                passed=passed,
                execution=record,
                complete_optimizer_records=complete_records,
                final_evaluation_present=final_eval,
                scope="Runtime completion/finite updates, not evidence of accuracy or convergence",
            ),
        )
        status(plan, "completed" if passed else "failed", code)
        return 0 if passed else (code or 1)
    except BaseException:
        status(plan, "failed")
        raise
    finally:
        subprocess.run(["ray", "stop", "--force"], check=False)


def restore_output_ownership():
    owner = ROOT.stat()
    if os.geteuid() != 0:
        return
    for directory, dirs, files in os.walk(ROOT, followlinks=False):
        for path in [Path(directory)] + [Path(directory) / name for name in dirs + files]:
            os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)


if __name__ == "__main__":
    try:
        result = main(sys.argv[1])
    finally:
        restore_output_ownership()
    raise SystemExit(result)
