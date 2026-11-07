"""Manual CPU checks for same-run W4A8 metric resume; no model or Ray startup."""

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0


def setup_resume(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "slime_plugins/w4a8_sim/metrics.py"
    spec = importlib.util.spec_from_file_location("w4a8_resume_metrics", source)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    root = tmp_path / "checkpoints"
    checkpoint = root / ".snapshots/step_0000024"
    (checkpoint / "rollout").mkdir(parents=True)
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("24")
    (checkpoint / "rollout/global_dataset_state_dict_24.pt").touch()
    (root / "last").symlink_to(checkpoint)
    (root / "checkpoint-policy.json").write_text(json.dumps({"last": {"completed_updates": 25}}))
    work = tmp_path / "work"
    audit = work / "training-audit"
    audit.mkdir(parents=True)
    monkeypatch.setenv("W4A8_METRICS_DIR", str(work))
    monkeypatch.setenv("W4A8_TRAINING_AUDIT_DIR", str(audit))
    for name, fields, rows in (
        ("train_reward", metrics.TRAIN_FIELDS, [(i, i + 1, 20, 100, 0.75) for i in range(26)]),
        ("valid_metrics", metrics.VALID_FIELDS, [(i, 50, 45, 0.9, 0, 0.0) for i in range(0, 26, 5)]),
    ):
        with (work / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)
    (audit / "optimizer-rank-0.jsonl").write_text(
        "".join(json.dumps({"rollout_id": i, "optimizer_success": True}) + "\n" for i in range(26))
    )
    args = SimpleNamespace(
        load=str(checkpoint), save=str(root), start_rollout_id=25, num_rollout=50,
        eval_interval=5, skip_eval_before_train=False, finetune=False, no_load_optim=False, no_load_rng=False,
    )
    return metrics, args, work


def test_resume_archives_tail_and_allows_replayed_rollout(tmp_path, monkeypatch):
    metrics, args, work = setup_resume(tmp_path, monkeypatch)
    reward = work / "train_reward.csv"
    audit = work / "training-audit/optimizer-rank-0.jsonl"
    original_reward, original_audit = reward.read_text(), audit.read_text()
    original_valid = (work / "valid_metrics.csv").read_bytes()
    metrics.prepare_resume(args)
    archives = list(work.glob("resume-before-25-*.json"))
    assert len(archives) == 1
    archived = json.loads(archives[0].read_text())["original_files"]
    assert archived[str(reward)] == original_reward
    assert archived[str(audit)] == original_audit
    assert len(metrics._read_rows(reward, metrics.TRAIN_FIELDS)) == 25
    assert len(audit.read_text().splitlines()) == 25
    assert (work / "valid_metrics.csv").read_bytes() == original_valid
    assert len(json.loads((work / "train_reward.json").read_text())["records"]) == 25
    metrics.prepare_resume(args)
    assert list(work.glob("resume-before-25-*.json")) == archives
    row = dict(zip(metrics.TRAIN_FIELDS, (25, 26, 20, 100, 0.8)))
    assert len(metrics._append_row(reward, metrics.TRAIN_FIELDS, row, 25, "rollout_id")) == 26


def test_mismatched_resume_does_not_touch_history(tmp_path, monkeypatch):
    metrics, args, work = setup_resume(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}
    args.start_rollout_id = 26
    with pytest.raises(ValueError, match="does not match"):
        metrics.prepare_resume(args)
    assert {p: p.read_bytes() for p in work.rglob("*") if p.is_file()} == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
