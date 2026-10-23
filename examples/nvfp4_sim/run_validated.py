"""Run the unchanged BF16-study supervisor, then gate success on learning evidence.

The validator is standard-library-only and can also audit a completed run on the
host. A clean process exit alone never passes this validation.
"""

import ast
import json
import math
import re
import runpy
import sys
from pathlib import Path


def validate_run(directory, config):
    status_path = directory / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    log_path = directory / "logs/train.log"
    text = re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text(errors="replace")) if log_path.exists() else ""
    records_path = directory / "optimizer-validation.jsonl"
    records = (
        [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        if records_path.exists()
        else []
    )
    rollouts = {}
    for match in re.finditer(r"rollout_metrics\.py:\d+ - perf (\d+): (\{[^\n]+\})", text):
        metrics = ast.literal_eval(match.group(2))
        zero_groups = sum(value for key, value in metrics.items() if key.startswith("rollout/zero_std/count_"))
        rollouts[int(match.group(1))] = {
            "mixed_reward_groups": config["training"]["prompts_per_update"] - zero_groups,
            "truncated_ratio": metrics.get("rollout/truncated_ratio"),
            "mean_response_tokens": metrics.get("rollout/response_len/mean"),
        }
    planned = config["training"]["updates"]
    metrics_path = directory / "training-metrics.json"
    training_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    keys = ("fprop", "dgrad", "wgrad", "rht_x", "rht_g", "sr_dgrad", "sr_wgrad")
    verified_steps = []
    for record in records:
        norm = record.get("global_grad_norm", float("nan"))
        projection_norm = record.get("projection_grad_norm", float("nan"))
        delta = record.get("master_max_abs_delta", float("nan"))
        active = (
            record.get("optimizer_success")
            and record.get("grad_finite")
            and math.isfinite(norm)
            and norm > 0
            and math.isfinite(projection_norm)
            and projection_norm > 0
            and record.get("master", {}).get("dtype") == "torch.float32"
            and record.get("master_delta_finite")
            and math.isfinite(delta)
            and delta > 0
            and record.get("master_changed_elements", 0) > 0
            and all(record.get("recipe_counter_deltas", {}).get(key, 0) > 0 for key in keys)
        )
        mixed = rollouts.get(record.get("rollout_id"), {}).get("mixed_reward_groups", 0) > 0
        if active and mixed:
            verified_steps.append(record["rollout_id"])
    checkpoints = [path for path in (directory / "checkpoints").glob("iter_*") if any(path.glob("*.distcp"))]
    syncs = len(re.findall(r"Timer update_weights end", text))
    checks = {
        "runtime_completed": status.get("trainer_exit_code") == 0 and status.get("completed_updates") == planned,
        "training_metrics_complete_and_finite": len(training_metrics) == planned
        and all(
            math.isfinite(value)
            for metric in training_metrics
            for value in metric.values()
            if isinstance(value, (int, float))
        ),
        "two_active_optimizer_updates": len(records) == planned and sorted(verified_steps) == list(range(planned)),
        "checkpoints_present": len(checkpoints) == planned,
        "weight_syncs_completed": syncs >= planned + 1,
        "final_evaluation_completed": any(
            item.get("completed_updates") == planned for item in status.get("evaluations", [])
        ),
        "ray_cleanup_completed": status.get("ray_cleanup_exit_code") == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "verified_rollout_ids": verified_steps,
        "rollouts": rollouts,
        "optimizer_records": records,
        "weight_sync_count": syncs,
        "scope": "Two-update full-model GRPO integration; not convergence or accuracy validation",
    }


def main():
    config_path = Path(sys.argv[1])
    config = json.loads(config_path.read_text())
    assert config["require_training_validation"]
    sys.argv = ["/study/scripts/run_online_training.py", str(config_path)]
    runpy.run_path(sys.argv[0], run_name="__main__")
    directory = Path(config["container_run_dir"])
    report = validate_run(directory, config)
    (directory / "training-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    status_path = directory / "status.json"
    status = json.loads(status_path.read_text())
    status.update(
        runtime_completed=status.get("successful_training", False),
        successful_training=report["passed"],
        training_validation_passed=report["passed"],
        state="completed" if report["passed"] else "validation_failed",
    )
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(status_path)
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
