"""Atomic, complete best/last checkpoints on multiple owned filesystems.

Each pending directory and its final snapshot are on the same filesystem.
Only the small best/last symlinks are replaced in the logical save directory.
An ownership marker prevents pruning an unrelated experiment's files.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

try:
    from slime_plugins.nvfp4_sim.checkpoint_policy import atomic_json, _point, _state, set_trainer_save, set_rollout_save
except ModuleNotFoundError:
    from nvfp4_sim.checkpoint_policy import atomic_json, _point, _state, set_trainer_save, set_rollout_save


def storage_roots(root):
    root = Path(root).resolve()
    paths = json.loads(os.environ.get("W4A8_CHECKPOINT_ROOTS", "[]")) or [str(root / ".snapshots")]
    result = []
    for path in paths:
        physical = Path(path)
        if not physical.is_absolute() or physical.is_symlink():
            raise ValueError("Checkpoint storage roots must be absolute, physical directories")
        physical.mkdir(parents=True, exist_ok=True)
        marker = physical / ".w4a8-owner.json"
        owner = {"logical_save_root": str(root)}
        if marker.exists():
            if json.loads(marker.read_text()) != owner:
                raise RuntimeError(f"Checkpoint storage belongs to another run: {physical}")
        else:
            if any(physical.iterdir()):
                raise RuntimeError(f"Refusing to adopt nonempty checkpoint storage: {physical}")
            atomic_json(marker, owner)
        result.append(physical.resolve())
    if len(set(result)) != len(result):
        raise ValueError("Duplicate checkpoint storage root")
    return result


def allocate_pending(root, required_bytes):
    roots = storage_roots(root)
    if os.environ.get("W4A8_STORAGE_FIRST_FIT") == "1":
        selected = next((p for p in roots if shutil.disk_usage(p).free >= required_bytes), None)
        if selected is None:
            raise RuntimeError(f"No checkpoint filesystem has {required_bytes} available bytes")
        pending = selected / (".pending-" + uuid.uuid4().hex)
        pending.mkdir()
        return pending
    choices = sorted(((shutil.disk_usage(p).free, str(p), p) for p in roots), reverse=True)
    available, _, selected = choices[0]
    if available < required_bytes:
        raise RuntimeError(f"No checkpoint filesystem has {required_bytes} available bytes; largest has {available}")
    pending = selected / (".pending-" + uuid.uuid4().hex)
    pending.mkdir()
    return pending


def prune(root):
    root = Path(root)
    roots = storage_roots(root)
    keep = set()
    for name in ("best", "last"):
        pointer = root / name
        if pointer.is_symlink():
            target = pointer.resolve(strict=True)
            if target.parent not in roots or not target.name.startswith(("step_", "best_step_")):
                raise RuntimeError(f"Checkpoint pointer outside owned storage: {pointer}")
            keep.add(target)
    for physical in roots:
        for path in list(physical.glob("step_*")) + list(physical.glob("best_step_*")):
            if path.is_dir() and not path.is_symlink() and path.resolve() not in keep:
                shutil.rmtree(path)


def publish_last(root, pending, rollout_id):
    root, pending = Path(root), Path(pending)
    if pending.resolve().parent not in storage_roots(root):
        raise RuntimeError("Pending checkpoint is not in this run's owned storage")
    iteration = int((pending / "latest_checkpointed_iteration.txt").read_text().strip())
    if iteration != rollout_id:
        raise RuntimeError("Checkpoint step does not match optimizer update")
    model = pending / f"iter_{rollout_id:07d}"
    if not (model / ".metadata").is_file() or not any(model.glob("*.distcp")):
        raise RuntimeError("Incomplete native distributed checkpoint")
    if not (pending / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt").is_file():
        raise RuntimeError("Missing dataset cursor")
    target = pending.parent / f"step_{rollout_id:07d}"
    if target.exists():
        raise FileExistsError(target)
    pending.rename(target)
    root.mkdir(parents=True, exist_ok=True)
    _point(root, "last", target)
    state = _state(root)
    state.update(last={"rollout_id": rollout_id, "completed_updates": rollout_id + 1, "snapshot": str(target)},
                 policy="best-last-multiple-filesystems")
    atomic_json(root / "checkpoint-policy.json", state)
    prune(root)
    return target


def record_evaluation(root, rollout_id, score, questions):
    root = Path(root)
    if not math.isfinite(score):
        raise ValueError("Nonfinite checkpoint selection metric")
    root.mkdir(parents=True, exist_ok=True)
    state = _state(root)
    evaluation = dict(rollout_id=rollout_id, score=score, questions=questions, metric="eval/gsm8k")
    state["latest_evaluation"] = evaluation
    last, best = state.get("last"), state.get("best")
    if last is not None and last["rollout_id"] == rollout_id and (best is None or score > best["score"]):
        if os.environ.get("W4A8_BEST_WEIGHTS_ONLY") == "1":
            destination = export_best_weights(root, last)
            _point(root, "best", destination)
            state["best"] = dict(evaluation, completed_updates=rollout_id + 1,
                snapshot=str(destination), format="hf-bf16-weights", optimizer_included=False)
        else:
            _point(root, "best", Path(last["snapshot"]))
            state["best"] = dict(last, **evaluation)
    atomic_json(root / "checkpoint-policy.json", state)
    prune(root)
    return state


def export_best_weights(root, last):
    """Export only trained dense Qwen tensors in a separate CPU subprocess.

    Evaluation owns a CUDA-visible rollout actor. The exporter must create no
    CUDA context and must not read optimizer tensors into memory.
    """
    pending = allocate_pending(root, int(os.environ["W4A8_BEST_BYTES"]))
    # The existing exporter requires a fresh output directory.
    pending.rmdir()
    script = Path(__file__).resolve().parents[2] / "examples/nvfp4_sim/export_diagnostic_hf.py"
    step = last["rollout_id"]
    source = Path(last["snapshot"]) / f"iter_{step:07d}"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    subprocess.run([sys.executable, str(script), "--input", str(source), "--origin",
        os.environ["W4A8_INITIAL_MODEL"], "--output", str(pending)], env=environment, check=True)
    report = json.loads((pending / "export-validation.json").read_text())
    if not report.get("passed") or report.get("optimizer_exported") is not False:
        raise RuntimeError("Best export did not pass weight-only validation")
    if report["iteration"] != step or not (pending / "model.safetensors").is_file():
        raise RuntimeError("Best export does not match the selected trained step")
    atomic_json(pending / "precision-policy.json", dict(
        training_mode=os.environ.get("W4A8_TRAINING_MODE", "w4a8"),
        stored_weight_dtype="bfloat16", optimizer_included=False,
        inference_requires_explicit_precision_selection=True))
    target = pending.parent / f"best_step_{step:07d}"
    if target.exists():
        raise FileExistsError(target)
    pending.rename(target)
    return target


def eval_log_hook(rollout_id, args, data, extra_metrics=None):
    rewards = data["gsm8k"]["rewards"]
    record_evaluation(args.save, rollout_id, sum(rewards) / len(rewards), len(rewards))
    return False


def save_last(args, rollout_id, actor_model, critic_model, rollout_manager, **kwargs):
    import ray
    from slime.data.checkpoint import save_checkpoint as native_save

    if args.async_save or args.use_critic or critic_model is not None or args.no_save_optim or args.no_save_rng:
        raise ValueError("Complete synchronous actor/model/optimizer/RNG checkpoints are required")
    root = Path(args.save)
    required = int(os.environ["W4A8_CHECKPOINT_BYTES"])
    pending = allocate_pending(root, required)
    previous, trainers = str(root), []
    rollout_previous = None
    handlers = actor_model._actor_handlers
    try:
        trainers = ray.get([a.__ray_call__.remote(set_trainer_save, str(pending)) for a in handlers])
        rollout_previous = ray.get(rollout_manager.__ray_call__.remote(set_rollout_save, str(pending)))
        if any(p != previous for p in trainers) or rollout_previous != previous:
            raise RuntimeError("Distributed checkpoint roots were inconsistent")
        args.save = actor_model.args.save = str(pending)
        native_save(args, rollout_id, actor_model, critic_model, rollout_manager, **kwargs)
        return publish_last(root, pending, rollout_id)
    finally:
        args.save = actor_model.args.save = previous
        if trainers:
            ray.get([a.__ray_call__.remote(set_trainer_save, p) for a, p in zip(handlers, trainers)])
        if rollout_previous is not None:
            ray.get(rollout_manager.__ray_call__.remote(set_rollout_save, rollout_previous))
