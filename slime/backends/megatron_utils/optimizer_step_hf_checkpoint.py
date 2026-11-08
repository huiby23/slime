"""Opt-in HF snapshots at successful optimizer boundaries, independent of serving."""
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


def _rank_zero_call(dist, group, operation):
    error = None
    if dist.get_rank() == 0:
        try:
            operation()
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
    result = [error]
    dist.broadcast_object_list(result, src=0, group=group)
    if result[0] is not None:
        raise RuntimeError(f"Optimizer-step HF checkpoint failed: {result[0]}")


def _clear_export_memory():
    from slime.utils.memory_utils import clear_memory

    clear_memory()
    if os.environ.get("SLIME_NPU_TRIM_UNUSED_CPU_HEAP") == "1":
        from slime.utils.npu_optimizer_preallocation import trim_unused_cpu_heap

        trim_unused_cpu_heap()


def after_optimizer_step(args, model, *, rollout_id, step_id, successful, grad_norm,
                         disable_pre_hook=None, enable_pre_hook=None, optimizer=None, scheduler=None, data_iterator=None):
    """Count actual successful updates; all actor ranks enter the same save boundary.

    Called only after train_one_step has released its gradients and frame. No
    updater/serving object is used: the rollout's policy version remains fixed.
    A failed write leaves a .partial directory; only complete HF files are renamed.
    """
    interval = getattr(args, "optimizer_step_hf_save_interval", 0)
    if not interval or getattr(model[0], "role", "actor") != "actor":
        return None
    if interval < 1:
        raise ValueError("Optimizer-step HF checkpoint interval must be positive")

    import torch
    import torch.distributed as dist
    from slime.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    offset = int(args.optimizer_step_hf_completed_offset)
    completed = getattr(args, "_optimizer_step_hf_completed_updates", offset)
    finite_gradient = math.isfinite(float(grad_norm))
    stateful = bool(getattr(args, "hybrid_restart_store", None))
    status = (int(rollout_id), int(step_id), completed, bool(successful), finite_gradient,
              (float(grad_norm) > 0) if stateful else True)
    statuses = [None] * dist.get_world_size()
    dist.all_gather_object(statuses, status, group=group)
    if any(item != status for item in statuses):
        raise RuntimeError(f"Actor ranks disagree at optimizer boundary: {statuses}")
    if stateful:
        if not successful or not finite_gradient or float(grad_norm) <= 0:
            raise RuntimeError("Stateful GRPO requires a successful finite positive-gradient update")
        if completed != int(rollout_id) * args.num_steps_per_rollout + int(step_id):
            raise RuntimeError("Real update count and prepared minibatch cursor disagree")
    if not successful:
        return None
    if not finite_gradient:
        raise ValueError("A successful update must have a finite gradient norm")
    completed += 1
    args._optimizer_step_hf_completed_updates = completed
    if completed % interval:
        return None

    from slime.backends.megatron_utils.hf_checkpoint_saver import save_hf_model_to_path

    root = Path(args.optimizer_step_hf_save_dir)
    final = root / f"checkpoint_step_{completed:06d}"
    partial = root / (final.name + ".partial")

    def prepare():
        root.mkdir(parents=True, exist_ok=True)
        if final.exists() or partial.exists():
            raise FileExistsError(f"Refusing to overwrite optimizer snapshot {final}")
        partial.mkdir()

    _rank_zero_call(dist, group, prepare)
    # Matching model.save semantics forces any outstanding parameter all-gather
    # to finish. Restore the same pre-hook state before the next microbatch.
    if disable_pre_hook is not None:
        disable_pre_hook()
    try:
        _clear_export_memory()
        with torch.no_grad():
            save_hf_model_to_path(args, partial, model,
                                 progress_desc=f"Save HF optimizer step {completed}")
        dist.barrier(group=group)
        _clear_export_memory()

        bundle = None
        if stateful:
            from slime.utils.hybrid_restart import save_bundle
            staging, bundle = save_bundle(args, model, optimizer, scheduler, data_iterator, rollout_id,
                                 step_id, completed, dist, group)

        def publish():
            index_path = partial / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            shards = set(index["weight_map"].values())
            if not shards or not (partial / "config.json").is_file():
                raise ValueError("HF checkpoint is missing its index, tensors, or config")
            if any(not (partial / name).is_file() for name in shards):
                raise ValueError("HF checkpoint is missing an indexed shard")
            metadata = {
                "completed_optimizer_steps": completed,
                "parent_completed_optimizer_steps": offset,
                "new_successful_optimizer_steps": completed - offset,
                "rollout_id": int(rollout_id), "step_id": int(step_id),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "format": "huggingface_safetensors", "complete": True,
                "optimizer_state_saved": False, "rng_state_saved": False,
                "source_cursor_saved": False, "hf_only": True,
                "parent_hf_checkpoint": str(args.hf_checkpoint),
                "serving_weight_version_changed": False,
                "tensor_keys": len(index["weight_map"]), "shard_files": len(shards),
                "model_tensor_bytes": index.get("metadata", {}).get("total_size"),
            }
            if bundle is not None:
                metadata.update(optimizer_state_saved=True, rng_state_saved=True, source_cursor_saved=True,
                                hf_only=False, restart_bundle_at_save=str(bundle),
                                restart_retention=args.hybrid_restart_retention)
            if getattr(args, "attention_simulation", None) is not None:
                metadata["attention_simulation"] = args.attention_simulation
            (partial / "optimizer-step-checkpoint.json").write_text(
                json.dumps(metadata, indent=2) + "\n")
            if bundle is not None:
                for item in partial.iterdir():
                    if item.is_file():
                        with item.open("rb") as saved:
                            os.fsync(saved.fileno())
            os.replace(partial, final)
            if bundle is not None:
                from slime.utils.hybrid_restart import advance_latest, check_next_checkpoint_space, sync_dir
                sync_dir(final)
                sync_dir(root)
                os.rename(staging, bundle)
                sync_dir(bundle.parent)
                advance_latest(bundle)
                check_next_checkpoint_space(args, bundle, index["metadata"]["total_size"])

        _rank_zero_call(dist, group, publish)
        dist.barrier(group=group)
    finally:
        if enable_pre_hook is not None:
            enable_pre_hook()
    if dist.get_rank() == 0:
        print(f"OPTIMIZER_STEP_HF_CHECKPOINT={final}", flush=True)
    return str(final)
