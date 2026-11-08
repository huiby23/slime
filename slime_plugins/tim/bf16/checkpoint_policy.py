"""Publish complete best/last checkpoints using Slime's native save operation.

The trainer, optimizer, RNG, and matching dataset cursor are saved together in a
new staging directory. Only after completion does an atomic symlink publish
`last`; evaluations can publish `best`. Configured early checkpoints have stable
completed-update links and survive pruning. This module never changes updates,
sampling, or loss functions.
"""

import json
import math
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _state(root):
    path = root / "checkpoint-policy.json"
    if path.exists():
        return json.loads(path.read_text())
    count = int(os.environ.get("TIM_RETAIN_EARLY_CHECKPOINTS", "0"))
    if count < 0:
        raise ValueError("TIM_RETAIN_EARLY_CHECKPOINTS must be nonnegative")
    return {
        "policy": "best-last-early" if count else "best-last",
        "best": None,
        "retain_early_completed_updates": count,
        "early": {},
    }


def _point(root, name, target):
    destination = root / name
    if destination.exists() and not destination.is_symlink():
        raise RuntimeError("Refusing to replace a non-symlink checkpoint: " + str(destination))
    temporary = root / ("." + name + "." + uuid.uuid4().hex)
    temporary.symlink_to(os.path.relpath(target, root), target_is_directory=True)
    temporary.replace(destination)


def prune(root):
    """Delete owned snapshots except best, last, and pinned early targets."""
    snapshots = root / ".snapshots"
    keep = set()
    for name in ("best", "last"):
        pointer = root / name
        if pointer.is_symlink():
            target = pointer.resolve(strict=True)
            target.relative_to(snapshots.resolve())
            keep.add(target)
    for pointer in (root / "early").glob("step_*"):
        if not pointer.is_symlink():
            raise RuntimeError("Refusing an unmanaged early checkpoint: " + str(pointer))
        target = pointer.resolve(strict=True)
        target.relative_to(snapshots.resolve())
        keep.add(target)
    for path in snapshots.glob("step_*"):
        if path.is_dir() and not path.is_symlink() and path.resolve() not in keep:
            shutil.rmtree(path)


def publish_last(root, pending, rollout_id):
    root, pending = Path(root), Path(pending)
    iteration = int((pending / "latest_checkpointed_iteration.txt").read_text().strip())
    assert iteration == rollout_id, (iteration, rollout_id)
    model = pending / f"iter_{rollout_id:07d}"
    assert (model / ".metadata").is_file() and any(model.glob("*.distcp"))
    assert (pending / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt").is_file()
    snapshots = root / ".snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    target = snapshots / f"step_{rollout_id:07d}"
    if target.exists():
        raise FileExistsError("Checkpoint snapshots are immutable: " + str(target))
    pending.rename(target)
    _point(root, "last", target)
    state = _state(root)
    state["last"] = {"rollout_id": rollout_id, "completed_updates": rollout_id + 1, "snapshot": str(target)}
    early_count = state.get("retain_early_completed_updates", 0)
    if rollout_id + 1 <= early_count:
        early_root = root / "early"
        early_root.mkdir(exist_ok=True)
        # Public early link names count completed optimizer updates, not native
        # zero-based rollout ids. Their targets are the same complete snapshots.
        name = f"step_{rollout_id + 1:07d}"
        _point(early_root, name, target)
        state.setdefault("early", {})[str(rollout_id + 1)] = dict(state["last"], link=str(early_root / name))
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(root / "checkpoint-policy.json", state)
    prune(root)
    return target


def record_evaluation(root, rollout_id, score, questions):
    root = Path(root)
    if not math.isfinite(score):
        raise ValueError("Cannot select best using a nonfinite evaluation")
    root.mkdir(parents=True, exist_ok=True)
    state = _state(root)
    last = state.get("last")
    # Initial eval and post-first-update eval both use native rollout_id=0.
    # A matching fully published checkpoint distinguishes them without changing
    # the native RPC id, evaluation seed, or log parser contract.
    completed = rollout_id + 1 if last is not None and last["rollout_id"] == rollout_id else 0
    if completed == 0 and rollout_id != 0:
        raise RuntimeError("Evaluation has no matching complete checkpoint")
    evaluation = {
        "rollout_id": rollout_id,
        "completed_updates": completed,
        "score": score,
        "questions": questions,
        "metric": "eval/gsm8k",
    }
    state["latest_evaluation"] = evaluation
    # Initial evaluation has no corresponding trained checkpoint yet.
    if last is not None and last["rollout_id"] == rollout_id:
        best = state.get("best")
        if best is None or score > best["score"]:
            _point(root, "best", Path(last["snapshot"]))
            state["best"] = dict(last, **evaluation)
    atomic_json(root / "checkpoint-policy.json", state)
    prune(root)
    return state


def eval_log_hook(rollout_id, args, data, extra_metrics=None):
    rewards = data["gsm8k"]["rewards"]
    state = record_evaluation(args.save, rollout_id, sum(rewards) / len(rewards), len(rewards))
    evaluation = state["latest_evaluation"]
    completed = evaluation["completed_updates"]
    # Dedicated tags use the model's completed-update axis, preserving initial0
    # and trained1 as separate TB points even though native eval tags share0.
    from slime.observability import logging_utils

    metrics = {
        "eval_completed_updates/gsm8k": evaluation["score"],
        "eval_completed_updates/questions": len(rewards),
        "eval_completed_updates/step": completed,
    }
    if samples := data["gsm8k"].get("samples"):
        metrics["eval_completed_updates/response_len/mean"] = sum(
            sample.effective_response_length for sample in samples
        ) / len(samples)
    if "truncated" in data["gsm8k"]:
        truncated = data["gsm8k"]["truncated"]
        metrics["eval_completed_updates/truncated_ratio"] = sum(truncated) / len(truncated)
    logging_utils.log(args, metrics, step_key="eval_completed_updates/step")
    row = dict(evaluation, metrics=metrics, recorded_at_utc=datetime.now(timezone.utc).isoformat())
    with (Path(args.save).parent / "evaluations-by-completed-updates.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    return False  # Keep Slime's normal metrics and TensorBoard logging.


def set_trainer_save(actor, path):
    from megatron.training import get_args

    previous = actor.args.save
    assert get_args().save == previous
    actor.args.save = get_args().save = path
    return previous


def set_rollout_save(manager, path):
    previous = manager.args.save
    objects = [manager, manager.data_source, manager.batch_builder]
    for obj in objects:
        if hasattr(obj, "args"):
            assert obj.args.save == previous
    for obj in objects:
        if hasattr(obj, "args"):
            obj.args.save = path
    return previous


def save_last(args, rollout_id, actor_model, critic_model, rollout_manager, **kwargs):
    import ray
    from slime.data.checkpoint import save_checkpoint as native_save

    assert not args.async_save, "Best/last publication requires synchronous native writes"
    assert not args.use_critic and critic_model is None
    root = Path(args.save)
    root.mkdir(parents=True, exist_ok=True)
    pending = root / (".pending-" + uuid.uuid4().hex)
    pending.mkdir()
    previous = str(root)
    handlers = actor_model._actor_handlers
    trainer_roots = []
    rollout_root = None
    try:
        trainer_roots = ray.get([actor.__ray_call__.remote(set_trainer_save, str(pending)) for actor in handlers])
        rollout_root = ray.get(rollout_manager.__ray_call__.remote(set_rollout_save, str(pending)))
        assert all(path == previous for path in trainer_roots) and rollout_root == previous
        args.save = str(pending)
        actor_model.args.save = str(pending)
        native_save(args, rollout_id, actor_model, critic_model, rollout_manager, **kwargs)
        return publish_last(root, pending, rollout_id)
    finally:
        args.save = actor_model.args.save = previous
        if trainer_roots:
            ray.get(
                [actor.__ray_call__.remote(set_trainer_save, path) for actor, path in zip(handlers, trainer_roots)]
            )
        if rollout_root is not None:
            ray.get(rollout_manager.__ray_call__.remote(set_rollout_save, rollout_root))
