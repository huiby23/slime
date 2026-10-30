"""Multiple-filesystem publication must preserve last on failed/incomplete saves."""

import json
from pathlib import Path

import pytest

from slime_plugins.w4a8_sim.checkpoint_policy import allocate_pending, publish_last, record_evaluation, storage_roots

NUM_GPUS = 0


def complete(pending, step):
    (pending / "latest_checkpointed_iteration.txt").write_text(str(step))
    model = pending / f"iter_{step:07d}"
    model.mkdir()
    (model / ".metadata").write_text("metadata")
    (model / "rank0.distcp").write_text("model+optimizer+RNG")
    (pending / "rollout").mkdir()
    (pending / "rollout" / f"global_dataset_state_dict_{step}.pt").write_text("cursor")


def test_best_and_last_survive_cross_root_publication_and_incomplete_save(tmp_path, monkeypatch):
    logical = tmp_path / "logical"
    a, b = tmp_path / "disk-a", tmp_path / "disk-b"
    monkeypatch.setenv("W4A8_CHECKPOINT_ROOTS", json.dumps([str(a), str(b)]))
    storage_roots(logical)
    p0 = a / ".pending-zero"
    p0.mkdir()
    complete(p0, 0)
    best = publish_last(logical, p0, 0)
    record_evaluation(logical, 0, 0.8, 512)
    p1 = b / ".pending-one"
    p1.mkdir()
    complete(p1, 1)
    last = publish_last(logical, p1, 1)
    record_evaluation(logical, 1, 0.7, 512)
    assert (logical / "best").resolve() == best
    assert (logical / "last").resolve() == last
    incomplete = allocate_pending(logical, 1)
    with pytest.raises(FileNotFoundError):
        publish_last(logical, incomplete, 2)
    assert best.exists() and last.exists()
    p2 = a / ".pending-two"
    p2.mkdir()
    complete(p2, 2)
    new = publish_last(logical, p2, 2)
    assert best.exists() and not last.exists()
    record_evaluation(logical, 2, 0.9, 512)
    assert not best.exists() and new.exists()
    assert (logical / "best").resolve() == (logical / "last").resolve() == new


def test_refuse_other_runs_or_unowned_directories(tmp_path, monkeypatch):
    physical = tmp_path / "physical"
    monkeypatch.setenv("W4A8_CHECKPOINT_ROOTS", json.dumps([str(physical)]))
    storage_roots(tmp_path / "first")
    with pytest.raises(RuntimeError, match="another run"):
        storage_roots(tmp_path / "second")
    other = tmp_path / "other"
    other.mkdir()
    (other / "step_unrelated").mkdir()
    monkeypatch.setenv("W4A8_CHECKPOINT_ROOTS", json.dumps([str(other)]))
    with pytest.raises(RuntimeError, match="nonempty"):
        storage_roots(tmp_path / "third")


def test_fail_before_save_when_no_single_filesystem_has_space(tmp_path, monkeypatch):
    monkeypatch.setenv("W4A8_CHECKPOINT_ROOTS", json.dumps([str(tmp_path / "physical")]))
    with pytest.raises(RuntimeError, match="No checkpoint filesystem"):
        allocate_pending(tmp_path / "logical", 2**100)


def test_weight_only_best_does_not_retain_full_optimizer_snapshot(tmp_path, monkeypatch):
    from slime_plugins.w4a8_sim import checkpoint_policy as policy

    root, physical = tmp_path / "run", tmp_path / "disk"
    monkeypatch.setenv("W4A8_CHECKPOINT_ROOTS", json.dumps([str(physical)]))
    monkeypatch.setenv("W4A8_BEST_WEIGHTS_ONLY", "1")

    def export(root, last):
        target = physical / f"best_step_{last['rollout_id']:07d}"
        target.mkdir()
        (target / "model.safetensors").write_text("weights only")
        return target

    monkeypatch.setattr(policy, "export_best_weights", export)
    pending = allocate_pending(root, 1)
    complete(pending, 0)
    first = publish_last(root, pending, 0)
    state = record_evaluation(root, 0, 0.8, 32)
    best = (root / "best").resolve()
    assert best != first and state["best"]["optimizer_included"] is False
    pending = allocate_pending(root, 1)
    complete(pending, 1)
    second = publish_last(root, pending, 1)
    assert not first.exists() and best.exists() and second.exists()
    record_evaluation(root, 1, 0.7, 32)
    assert (root / "best").resolve() == best

    def fail(*args):
        raise RuntimeError("export interrupted")

    monkeypatch.setattr(policy, "export_best_weights", fail)
    with pytest.raises(RuntimeError, match="export interrupted"):
        record_evaluation(root, 1, 0.9, 32)
    assert (root / "last").resolve() == second and (root / "best").resolve() == best
