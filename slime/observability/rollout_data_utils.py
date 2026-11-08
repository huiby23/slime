import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from slime.data.tensor import TensorRef, materialize_tensor_refs
from slime.utils.routed_experts import validate_routed_experts_value
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_ROLLOUT_DATA_TENSOR_DTYPES = {
    "tokens": torch.long,
    "loss_masks": torch.int,
    "rollout_log_probs": torch.float32,
    "rollout_topk_token_ids": torch.int32,
    "rollout_topk_log_probs": torch.float32,
    "rollout_top_p_token_ids": torch.int32,
    "rollout_top_p_token_offsets": torch.int32,
    "rollout_top_p_log_probs": torch.float32,
    "teacher_log_probs": torch.float32,
    "rollout_routed_experts": None,
}


def _cpu_tensor(value, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    tensor = torch.as_tensor(value, dtype=dtype) if dtype is not None else torch.as_tensor(value)
    return tensor.detach().cpu().contiguous()


def tensorize_rollout_data_for_training(rollout_data: dict[str, Any]) -> None:
    for key, dtype in _ROLLOUT_DATA_TENSOR_DTYPES.items():
        if key in rollout_data:
            rollout_data[key] = [
                value if isinstance(value, TensorRef) else _cpu_tensor(value, dtype=dtype)
                for value in rollout_data[key]
            ]

    if "multimodal_train_inputs" in rollout_data:
        rollout_data["multimodal_train_inputs"] = [
            (
                {
                    key: _cpu_tensor(value) if isinstance(value, (np.ndarray, torch.Tensor)) else value
                    for key, value in mm_dict.items()
                }
                if mm_dict is not None
                else None
            )
            for mm_dict in rollout_data["multimodal_train_inputs"]
        ]

    if "rollout_mask_sums" in rollout_data:
        rollout_data["rollout_mask_sums"] = _cpu_tensor(
            rollout_data["rollout_mask_sums"],
            dtype=torch.float32,
        )


def validate_rollout_routed_experts_for_replay(
    routed_experts: list[torch.Tensor | TensorRef],
    args,
    expected_rows: list[int] | None = None,
) -> None:
    """Reject incomplete PP routing captures before R3 consumes them."""
    if not routed_experts:
        raise ValueError("R3 is enabled but no rollout routed-experts tensors were returned.")

    for sample_idx, experts in enumerate(routed_experts):
        validate_routed_experts_value(
            experts,
            args,
            sample_index=sample_idx,
            expected_rows=None if expected_rows is None else expected_rows[sample_idx],
        )


def validate_rollout_id_annotated(node, depth=0):
    """Walk the rollout function's nested output and validate ``rollout_id`` only
    when a compact / subagent pattern is detected.

    "Compact" = the rollout function wraps multiple training samples from one
    rollout execution into a ``list[Sample]``. In slime's convention the
    default rollout shape is ``list[list[Sample]]`` (depth-2: prompt × rollout)
    so its leaf ``list[Sample]`` lands at depth 1 and we skip validation,
    preserving backward compatibility. A compact rollout adds a third level:
    ``list[list[list[Sample]]]`` (prompt × rollout × samples-from-one-rollout),
    so the leaf ``list[Sample]`` lands at depth ≥ 2. At that point we require
    every sibling to carry a non-None ``rollout_id`` and to share the same
    value, so the loss reducer counts the rollout once instead of N times.
    """
    if isinstance(node, Sample):
        return
    assert isinstance(node, list), f"unexpected rollout output node type: {type(node).__name__}"
    if node and isinstance(node[0], Sample):
        if depth >= 2 and len(node) > 1:
            rids = [sample.rollout_id for sample in node]
            missing = [i for i, rollout_id in enumerate(rids) if rollout_id is None]
            assert not missing, (
                f"Compact rollout returned {len(node)} samples but rollout_id is unset on "
                f"positions {missing}. Set Sample.rollout_id on every sibling so the loss "
                "reducer can aggregate them as one rollout instead of N."
            )
            assert len(set(rids)) == 1, f"Sibling samples from one compact rollout must share rollout_id; got {rids}."
        return
    for item in node:
        validate_rollout_id_annotated(item, depth + 1)


def load_debug_rollout_data(path_template, *, rollout_id: int, subsample_ratio=None) -> list[Sample]:
    path = path_template.format(rollout_id=rollout_id)
    if path.endswith(".straw.json"):
        from slime.data.archive import RolloutArchive

        with RolloutArchive(path) as archive:
            data = archive.load_samples()
    else:
        data = torch.load(path, weights_only=False)["samples"]
        data = [Sample.from_dict(sample) for sample in data]
    if subsample_ratio is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * subsample_ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            "Subsample loaded debug rollout data using ratio=%s and change num rows %s -> %s",
            subsample_ratio,
            original_num_rows,
            len(data),
        )
    return data


def save_debug_rollout_data(
    path_template, data, *, rollout_id: int, evaluation: bool, args=None, reference=None
) -> None:
    if path_template is None:
        return

    path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
    logger.info(f"Save debug rollout data to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if str(path).endswith(".straw.json"):
        from slime.data.archive import RolloutArchive

        samples = [sample for info in data.values() for sample in info["samples"]] if evaluation else data
        RolloutArchive.save(
            path, samples, rollout_id=rollout_id, evaluation=evaluation, args=args, reference=reference
        )
        return

    if evaluation:
        samples = [sample.to_dict() for info in data.values() for sample in info["samples"]]
    else:
        samples = [sample.to_dict() for sample in data]

    for sample in samples:
        sample.pop("_queue_original_input", None)
    dump_data = {"rollout_id": rollout_id, "samples": samples}
    torch.save(materialize_tensor_refs(dump_data), path)


def save_pre_abort_rollout_backup(directory, groups, *, rollout_id: int, args) -> None:
    """Persist accepted CPU samples before network abort/drain can fail.

    This is a data archive only. It does not enable trainer recovery, aggregate
    optimizer state, or change any Sample fields, reward, token, or log-prob.
    """
    import json
    import os
    from datetime import datetime, timezone

    samples = list(groups)
    while samples and isinstance(samples[0], list):
        samples = [sample for group in samples for sample in group]
    if not samples or not all(hasattr(sample, "to_dict") for sample in samples):
        raise ValueError("Pre-abort backup requires accepted in-memory Sample groups")
    target = Path(directory) / f"rollout_{rollout_id}.pt"
    temporary = target.with_suffix(".pt.partial")
    save_debug_rollout_data(str(temporary), samples, rollout_id=rollout_id, evaluation=False, args=args)
    os.replace(temporary, target)
    metadata = {
        "stage": "accepted-before-abort", "rollout_id": rollout_id,
        "groups": len(groups), "samples": len(samples),
        "schema": "slime-debug-rollout", "optimizer_checkpoint": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(target),
    }
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def save_incremental_accepted_rollout_backup(directory, groups, *, rollout_id: int, args) -> None:
    """Atomically retain accepted CPU groups; never advertise a complete rollout.

    The .pt carries its own counts/stage so it remains self-describing if a
    process dies between the two file renames. The complete pre-abort archive
    uses a different filename and remains unchanged.
    """
    import json
    import os
    from datetime import datetime, timezone

    if getattr(args, "rollout_data_transport", None) != "object-store":
        raise ValueError("Incremental accepted backup requires object-store CPU Sample groups")
    if not groups or len(groups) > args.rollout_batch_size:
        raise ValueError("Incremental accepted backup has an invalid accepted-group count")
    if any(not isinstance(group, list) or len(group) != args.n_samples_per_prompt for group in groups):
        raise ValueError("Incremental accepted backup requires complete prompt groups")
    samples = [sample for group in groups for sample in group]
    if not all(hasattr(sample, "to_dict") for sample in samples):
        raise ValueError("Incremental accepted backup requires in-memory Samples")
    sample_dicts = [sample.to_dict() for sample in samples]
    visited = set()

    def require_cpu(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("Incremental accepted backup refuses non-CPU tensors")
            return
        if isinstance(value, TensorRef):
            raise ValueError("Incremental accepted backup refuses deferred tensor references")
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, dict):
            for key, item in value.items():
                require_cpu(key)
                require_cpu(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                require_cpu(item)
        elif isinstance(value, np.ndarray) and value.dtype.hasobject:
            for item in value.flat:
                require_cpu(item)
        elif hasattr(value, "__dict__"):
            require_cpu(vars(value))

    require_cpu(sample_dicts)
    target = Path(directory) / f"rollout_{rollout_id}.partial.pt"
    sidecar = target.with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    metadata_temporary = sidecar.with_name(sidecar.name + ".tmp")
    previous = target.with_name(target.name + ".previous")
    metadata = {
        "stage": "accepted-incremental-partial", "complete": False,
        "rollout_id": rollout_id, "groups": len(groups), "samples": len(samples),
        "target_groups": args.rollout_batch_size,
        "target_samples": args.rollout_batch_size * args.n_samples_per_prompt,
        "schema": "slime-debug-rollout", "optimizer_checkpoint": False,
        "source_cursor_included": False, "cpu_only": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "path": str(target),
    }
    payload = {"rollout_id": rollout_id, "samples": sample_dicts, "backup_metadata": metadata}
    replaced_payload = False
    had_previous = target.exists()
    try:
        # Finish both writes before touching either prior committed filename.
        with temporary.open("wb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        with metadata_temporary.open("w") as file:
            json.dump(metadata, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        previous.unlink(missing_ok=True)
        if had_previous:
            os.link(target, previous)
        os.replace(temporary, target)
        replaced_payload = True
        os.replace(metadata_temporary, sidecar)
    except Exception as error:
        if replaced_payload:
            # Roll back a failed sidecar commit without copying any sample data.
            if had_previous:
                os.replace(previous, target)
            else:
                target.unlink(missing_ok=True)
        raise RuntimeError(
            f"Incremental accepted backup failed for rollout={rollout_id}, "
            f"groups={len(groups)}, samples={len(samples)}, path={target}; "
            "previous committed partial was preserved"
        ) from error
    finally:
        for path in (temporary, metadata_temporary, previous):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not clean incremental backup temporary %s", path)
