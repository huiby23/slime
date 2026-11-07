"""Live training-reward and validation-accuracy exports for W4A8 runs."""

import csv
import json
import math
import os
from io import StringIO
from pathlib import Path
from time import perf_counter, time_ns


TRAIN_FIELDS = ("rollout_id", "step", "sample_count", "valid_token_count", "train_reward")
VALID_FIELDS = ("step", "total", "correct", "accuracy", "truncated", "eval_seconds")

SPECS = {
    "train_reward": {
        "title": "Online training reward",
        "definition": "Mean binary math reward of freshly generated training answers before this update.",
        "unit": "fraction",
        "aggregation": "All sampled answers, including truncated answers",
        "phase": "SGLang generation before the indicated optimizer update",
        "sampling": "Four prompts x four answers per rollout; 16 sampled answers",
    },
    "valid_metrics": {
        "value_key": "accuracy",
        "title": "GSM8K validation accuracy",
        "definition": "Correct greedy answers / total prompts; fixed validation quality under serving inference.",
        "unit": "fraction",
        "aggregation": "All fixed prompts, including truncated answers",
        "phase": "After completed optimizer updates; step 0 is before training",
        "sampling": "Fixed validation prompts; one answer per prompt",
    },
}


def _metrics_dir():
    configured = os.environ.get("W4A8_METRICS_DIR")
    if configured:
        path = Path(configured)
    elif os.environ.get("TENSORBOARD_DIR"):
        path = Path(os.environ["TENSORBOARD_DIR"]).parent
    else:
        path = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _attention_metadata():
    mode = os.environ.get("W4A8_ATTENTION", "c16")
    training_mode = os.environ.get("W4A8_TRAINING_MODE", "unknown")
    return {
        "mode": mode,
        "training_mode": training_mode,
        "backend": "Transformer Engine Flash" if mode == "c16" else "W4A8 attention simulation",
    }


def _read_rows(path, fields):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):
            raise ValueError(f"Unexpected metric CSV columns: {path}")
        rows = []
        for row in reader:
            parsed = dict(row)
            for key in fields:
                if key in ("accuracy", "eval_seconds", "train_reward"):
                    parsed[key] = float(row[key])
                else:
                    parsed[key] = int(row[key])
            rows.append(parsed)
        return rows


def _append_row(path, fields, row, expected_index, index_key):
    rows = _read_rows(path, fields)
    if len(rows) != expected_index or (rows and rows[-1][index_key] != expected_index - 1):
        raise ValueError(f"Metric history is missing or duplicating {index_key}={expected_index}: {path}")
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not rows:
            writer.writeheader()
        writer.writerow(row)
    return rows + [row]


def prepare_resume(args):
    """Align this run's metric history with its committed native last checkpoint."""
    checkpoint = Path(args.load).resolve(strict=True)
    root = Path(args.save)
    if checkpoint != (root / "last").resolve(strict=True):
        raise ValueError("Metric resume must load this run's committed last checkpoint")
    if args.finetune or args.no_load_optim or args.no_load_rng:
        raise ValueError("Resume requires native optimizer and RNG state")
    iteration = int((checkpoint / "latest_checkpointed_iteration.txt").read_text().strip())
    start = iteration + 1
    policy = json.loads((root / "checkpoint-policy.json").read_text())
    if policy["last"]["completed_updates"] != start or args.start_rollout_id not in (None, start):
        raise ValueError("Resume start does not match the committed checkpoint")
    if not 0 < start < args.num_rollout:
        raise ValueError("Resume must precede the requested total rollout count")
    if not (checkpoint / "rollout" / f"global_dataset_state_dict_{iteration}.pt").is_file():
        raise ValueError("Resume checkpoint is missing its rollout dataset cursor")
    args.start_rollout_id = start

    directory = _metrics_dir()
    changes = []
    metric_rows = {}
    for name, fields, key, boundary in (
        ("train_reward", TRAIN_FIELDS, "rollout_id", start - 1),
        ("valid_metrics", VALID_FIELDS, "step", start),
    ):
        path = directory / f"{name}.csv"
        rows = _read_rows(path, fields)
        kept = [row for row in rows if row[key] <= boundary]
        expected = (
            list(range(start))
            if name == "train_reward"
            else [step for step in _evaluation_steps(args) if step <= start]
        )
        if [row[key] for row in kept] != expected:
            raise ValueError(f"Missing or duplicate history before resume boundary: {path}")
        metric_rows[name] = kept
        if len(kept) != len(rows):
            stream = StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
            changes.append((path, path.read_text(), stream.getvalue()))

    audit_dir = Path(os.environ["W4A8_TRAINING_AUDIT_DIR"])
    for path in sorted(audit_dir.glob("optimizer-rank-*.jsonl")):
        original = path.read_text()
        lines = original.splitlines(keepends=True)
        kept = [line for line in lines if json.loads(line)["rollout_id"] < start]
        if len(kept) != len(lines):
            changes.append((path, original, "".join(kept)))

    # Save originals together before replacing anything; repeat preparation is a no-op.
    if changes:
        archive = directory / f"resume-before-{start}-{time_ns()}.json"
        _atomic_json(archive, {
            "checkpoint": str(checkpoint),
            "start_rollout_id": start,
            "original_files": {str(path): original for path, original, _ in changes},
        })
        for path, _, content in changes:
            temporary = path.with_name(path.name + ".resume.tmp")
            temporary.write_text(content)
            temporary.replace(path)
        for name, rows in metric_rows.items():
            _write_json(name, rows, args)
        print(f"W4A8 resume: archived superseded history to {archive}", flush=True)
    print(f"W4A8 resume: load {checkpoint}; next rollout_id={start}; total={args.num_rollout}", flush=True)


def _evaluation_steps(args):
    if getattr(args, "skip_eval_before_train", False):
        steps = list(range(args.eval_interval, args.num_rollout + 1, args.eval_interval))
        if not steps or steps[-1] != args.num_rollout:
            steps.append(args.num_rollout)
        return steps
    return list(range(0, args.num_rollout + 1, args.eval_interval))


def _write_json(name, rows, args):
    spec = SPECS[name]
    path = _metrics_dir() / f"{name}.json"
    if name == "valid_metrics":
        expected = len(_evaluation_steps(args))
        value_key = "accuracy"
    else:
        expected = args.num_rollout
        value_key = "train_reward"
    records = []
    for row in rows:
        value = row[value_key]
        status = "finite" if math.isfinite(value) else "nan" if math.isnan(value) else "inf"
        records.append({**row, value_key: value if math.isfinite(value) else None, "value_status": status})
    payload = {
        "schema_version": 1,
        "metric": name,
        **spec,
        "run": str(_metrics_dir().resolve()),
        "evaluation_steps": _evaluation_steps(args) if name == "valid_metrics" else None,
        "attention": _attention_metadata(),
        "sources": [f"{name}.csv"],
        "plot": f"{name}.png" if name != "valid_metrics" else "valid_accuracy.png",
        "plot_style": "matplotlib Agg; blue line with circular markers; no smoothing",
        "expected_points": expected,
        "complete": len(rows) == expected,
        "all_finite": all(record["value_status"] == "finite" for record in records),
        "records": records,
    }
    _atomic_json(path, payload)


def _plot_curve(steps, values, path, *, title, subtitle, ylabel, xlabel, max_step, percentage, note):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    surface, ink, secondary, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "text.color": ink}):
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150, facecolor=surface)
        ax.set_facecolor(surface)
        ax.plot(
            steps,
            [value if math.isfinite(value) else math.nan for value in values],
            color="#2a78d6",
            linewidth=1.5,
            marker="o",
            markersize=6,
            markeredgewidth=1.5,
            markeredgecolor=surface,
        )
        tick_interval = max(5, math.ceil(max_step / 40) * 5)
        ticks = sorted({*range(0, max_step + 1, tick_interval), max_step})
        ax.set(xlim=(-0.035 * max_step, 1.1 * max_step), xticks=ticks)
        if percentage:
            ax.set(ylim=(-0.03, 1.05), yticks=[i / 5 for i in range(6)])
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel, color=secondary)
        ax.set_ylabel(ylabel, color=secondary)
        ax.set_title(title, loc="left", color=ink, pad=30, fontweight="bold")
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=secondary)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=grid, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(axis)
            ax.spines[spine].set_linewidth(0.6)
        ax.tick_params(colors=secondary, length=0, pad=6)
        if values and math.isfinite(values[-1]):
            ax.annotate(
                f"{values[-1]:.1%}" if percentage else f"{values[-1]:.3g}",
                (steps[-1], values[-1]),
                xytext=(8, 8),
                textcoords="offset points",
                color=ink,
            )
        fig.text(0.1, 0.025, note, fontsize=8, color=secondary)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(path, facecolor=surface)
        plt.close(fig)


def _plot_train(rows, args):
    if not rows:
        return
    path = _metrics_dir() / "train_reward.png"
    note = "Observed points only; no smoothing. Raw values: train_reward.csv / .json."
    if len(rows) < args.num_rollout:
        note = f"Partial run: {len(rows)}/{args.num_rollout} updates. " + note
    _plot_curve(
        [row["step"] for row in rows],
        [row["train_reward"] for row in rows],
        path,
        title=SPECS["train_reward"]["title"],
        subtitle=f"{_attention_metadata()['mode']} · Fresh sampled answers · temperature {getattr(args, 'rollout_temperature', 1.0)} · BF16 serving",
        ylabel="Mean binary reward",
        xlabel="Optimizer update (value measured before update)",
        max_step=args.num_rollout,
        percentage=True,
        note=note,
    )


def _plot_valid(rows, args):
    if not rows:
        return
    path = _metrics_dir() / "valid_accuracy.png"
    expected = len(_evaluation_steps(args))
    note = "Observed points only; no smoothing. Raw counts and truncations: valid_metrics.csv."
    if len(rows) < expected:
        note = f"Partial run: {len(rows)}/{expected} evaluation points. " + note
    _plot_curve(
        [row["step"] for row in rows],
        [row["accuracy"] for row in rows],
        path,
        title=SPECS["valid_metrics"]["title"],
        subtitle=f"Fixed {rows[0]['total']} prompts · one answer each",
        ylabel=f"Correct answers / {rows[0]['total']}",
        xlabel="Completed optimizer updates",
        max_step=args.num_rollout,
        percentage=True,
        note=note,
    )


def generate_eval(args, rollout_id, data_source, evaluation=False):
    if not evaluation:
        raise ValueError("generate_eval must only be used for evaluation")
    from slime.rollout.sglang_rollout import generate_rollout

    start = perf_counter()
    output = generate_rollout(args, rollout_id, data_source, evaluation=True)
    output.metrics = {**(output.metrics or {}), "valid/eval_seconds": perf_counter() - start}
    return output


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time, *, weight_version=None):
    del rollout_extra_metrics, rollout_time, weight_version
    if not samples:
        raise ValueError("Cannot record training reward for an empty rollout")
    rewards = [float(sample.get_reward_value(args)) for sample in samples]
    if any(not math.isfinite(reward) for reward in rewards):
        raise ValueError("Training rewards must be finite")
    row = {
        "rollout_id": rollout_id,
        "step": rollout_id + 1,
        "sample_count": len(samples),
        "valid_token_count": sum(sample.effective_response_length for sample in samples),
        "train_reward": sum(rewards) / len(rewards),
    }
    path = _metrics_dir() / "train_reward.csv"
    rows = _append_row(path, TRAIN_FIELDS, row, rollout_id, "rollout_id")
    _write_json("train_reward", rows, args)
    _plot_train(rows, args)
    return False


def _summarize_eval(rollout_id, args, data, extra_metrics):
    if len(data) != 1:
        raise ValueError("Expected exactly one evaluation dataset")
    result = next(iter(data.values()))
    rewards, samples = result["rewards"], result["samples"]
    total = len(rewards)
    if total != 50 or len(samples) != total or [sample.index for sample in samples] != list(range(total)):
        raise ValueError("Validation must return exactly one answer for each of the 50 fixed prompts")
    if any(reward not in (0, 1) for reward in rewards):
        raise ValueError("Validation rewards must be binary")
    interval = args.eval_interval
    if interval is None or interval <= 0 or args.num_rollout <= 0:
        raise ValueError("Metric export requires a positive rollout count and eval interval")
    step = 0 if rollout_id == 0 and not getattr(args, "skip_eval_before_train", False) else rollout_id + 1
    expected_steps = _evaluation_steps(args)
    if step < 0 or step > args.num_rollout or step not in expected_steps:
        raise ValueError(f"Invalid evaluation step: {step}")
    eval_seconds = float((extra_metrics or {}).get("valid/eval_seconds", 0.0))
    if not math.isfinite(eval_seconds) or eval_seconds < 0:
        raise ValueError("Evaluation time must be finite and non-negative")
    return {
        "step": step,
        "total": total,
        "correct": int(sum(rewards)),
        "accuracy": sum(rewards) / total,
        "truncated": sum(sample.status.value == "truncated" for sample in samples),
        "eval_seconds": eval_seconds,
    }


def log_eval(rollout_id, args, data, extra_metrics=None):
    row = _summarize_eval(rollout_id, args, data, extra_metrics)
    path = _metrics_dir() / "valid_metrics.csv"
    expected_steps = _evaluation_steps(args)
    rows = _read_rows(path, VALID_FIELDS)
    expected_index = expected_steps.index(row["step"])
    if len(rows) != expected_index:
        raise ValueError(f"Missing or duplicate validation point: step={row['step']}")
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=VALID_FIELDS)
        if not rows:
            writer.writeheader()
        writer.writerow(row)
    rows.append(row)
    _write_json("valid_metrics", rows, args)
    _plot_valid(rows, args)
    return False
