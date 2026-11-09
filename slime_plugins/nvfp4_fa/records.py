"""Receipts from native evaluation, optimizer and synchronous checkpoint paths."""

import io
import json
import math
import os
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def serving_receipt(layer, batch):
    if layer.layer_id != 35:
        return
    mode = batch.forward_mode.name
    recorded = getattr(layer, "_fa_recorded", set())
    if mode not in recorded:
        root = Path(os.environ["NVFP4_FA_RUN"])
        write_json(root / "coverage" / f"sg_{os.getpid()}_{mode}.json", dict(pid=os.getpid(), mode=mode, layer=35))
        layer._fa_recorded = recorded | {mode}


def trainer_coverage(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    from torch import distributed as dist
    from .megatron import PackedAttention

    if step_id != 0:
        raise ValueError("Expected one optimizer update per rollout")
    if getattr(optimizer, "_fa_checked", False):
        return
    count = sum(isinstance(module, PackedAttention) for chunk in model for module in chunk.modules())
    if count != 36:
        raise RuntimeError(f"Expected 36 NVFP4 training layers, got {count}")
    write_json(
        Path(args.save).parent / "coverage" / f"trainer_{dist.get_rank()}.json",
        dict(rank=dist.get_rank(), layers=count),
    )
    optimizer._fa_checked = True


def evaluation(rollout_id, args, data, extra_metrics=None):
    step = 0 if rollout_id == 0 else rollout_id + 1
    root = Path(args.save).parent
    if len(list((root / "coverage").glob("sg_*_DECODE.json"))) < 8:
        raise RuntimeError("Missing native NVFP4 decode evidence from eight rollout processes")
    group = data["math500"]
    answers = []
    for sample, reward, truncated in zip(group["samples"], group["rewards"], group["truncated"], strict=True):
        if reward not in (0, 1) or sample.metadata.get("source") != "MATH-500/test":
            raise ValueError("Unexpected MATH-500 sample")
        answers.append(
            dict(
                index=sample.metadata["source_index"],
                id=sample.metadata["unique_id"],
                label=sample.label,
                response=sample.response,
                reward=reward,
                truncated=truncated,
                output_tokens=sample.response_length,
            )
        )
    if sorted(a["index"] for a in answers) != list(range(500)) or len({a["id"] for a in answers}) != 500:
        raise ValueError("MATH-500 evaluation needs every official question exactly once")
    write_json(
        Path(args.save).parent / "evaluation" / f"step_{step:03}.json",
        dict(
            step=step,
            questions=500,
            accuracy=sum(a["reward"] for a in answers) / 500,
            truncation_rate=sum(a["truncated"] for a in answers) / 500,
            answers=answers,
        ),
    )
    return False  # Keep native Slime metrics as well.


def checkpoint(run, step):
    """Validate DCP tensor metadata and file extents, rollout cursor, and HF export."""
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    from safetensors import safe_open

    attempt = step - 1
    ckpt = run / "checkpoints" / f"iter_{attempt:07d}"
    meta = FileSystemReader(ckpt).read_metadata()
    names = list(meta.state_dict_metadata)
    for suffix in ("exp_avg", "exp_avg_sq"):
        moments = [meta.state_dict_metadata[k] for k in names if "optimizer" in k and suffix in k.split(".")]
        if not moments or any(m.properties.dtype != torch.float32 for m in moments):
            raise ValueError(f"Missing FP32 Adam {suffix} at step {step}")
    if not any("rng_state" in n for n in names):
        raise ValueError("Missing native RNG state")
    sizes = {}
    for loc in meta.storage_data.values():
        shard = ckpt / loc.relative_path
        if shard not in sizes:
            sizes[shard] = shard.stat().st_size
        if loc.offset + loc.length > sizes[shard]:
            raise ValueError(f"Truncated DCP shard: {shard}")
    # The pinned Megatron torch_dist format stores the scheduler in the
    # shared state record. Only deserialize checkpoints from a trusted run.
    common = next(loc for index, loc in meta.storage_data.items() if index.fqn.startswith("common_state/"))
    with (ckpt / common.relative_path).open("rb") as stream:
        stream.seek(common.offset)
        state = torch.load(io.BytesIO(stream.read(common.length)), map_location="cpu", weights_only=False)
    shared = state[0] if isinstance(state, list) else state
    if shared["iteration"] != attempt or not shared.get("opt_param_scheduler"):
        raise ValueError(f"Incomplete scheduler or iteration mismatch at step {step}")
    cursor = run / "checkpoints" / "rollout" / f"global_dataset_state_dict_{attempt}.pt"
    if not cursor.is_file() or not cursor.stat().st_size:
        raise ValueError("Missing rollout data-source state")
    hf = run / "hf" / f"rollout_{attempt:03d}"
    index = json.loads((hf / "model.safetensors.index.json").read_text())["weight_map"]
    if len(index) != 398 or not (hf / "config.json").is_file():
        raise ValueError("Incomplete Qwen3-4B HF export")
    for filename in set(index.values()):
        with safe_open(hf / filename, framework="pt", device="cpu") as f:
            if set(f.keys()) != {k for k, v in index.items() if v == filename}:
                raise ValueError(f"HF shard differs from index: {filename}")
    return dict(step=step, iteration=attempt, adam=True, scheduler=True, rng=True, cursor=str(cursor), hf=str(hf))


def grad_norm(run, step):
    import torch

    value = float(torch.load(run / "grad_norm" / f"actor_{step-1}_0.pt", map_location="cpu", weights_only=True))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid optimizer gradient norm at step {step}: {value}")
    return value


def verified_save(native, args, rollout_id, *actors, **kwargs):
    native(args, rollout_id, *actors, **kwargs)
    run = Path(args.save).parent
    step = rollout_id + 1
    result = checkpoint(run, step)
    result["grad_norms"] = [grad_norm(run, i) for i in range(step - 4, step + 1)]
    write_json(run / "saved" / f"step_{step:03d}.json", result)
