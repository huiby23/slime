"""Opt-in native AIME logging; shares the standalone HTTP trajectory writer."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite raw evaluation value")
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if type(value).__module__.startswith("torch") and hasattr(value, "tolist"):
        if getattr(getattr(value, "device", None), "type", None) != "cpu":
            raise TypeError("Validation logger must not copy accelerator tensors")
        return _jsonable(value.tolist())
    raise TypeError("Unsupported raw evaluation value: " + type(value).__name__)


def _shared(args):
    path = Path(args.validation_protocol_helper)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != args.validation_protocol_helper_sha256:
        raise ValueError("Shared validation protocol source changed after recipe validation")
    spec = importlib.util.spec_from_file_location("slime_validation_trajectory_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sampling(args, config):
    result = {
        "temperature": config.temperature, "top_p": config.top_p,
        "top_k": config.top_k, "max_new_tokens": config.max_response_len,
        "stop": config.stop if config.stop is not None else args.rollout_stop,
        "stop_token_ids": config.stop_token_ids if config.stop_token_ids is not None else args.rollout_stop_token_ids,
        "skip_special_tokens": config.skip_special_tokens if config.skip_special_tokens is not None else args.rollout_skip_special_tokens,
        "no_stop_trim": config.no_stop_trim if config.no_stop_trim is not None else True,
        "spaces_between_special_tokens": False,
    }
    if config.repetition_penalty is not None:
        result["repetition_penalty"] = config.repetition_penalty
    minimum = config.min_new_tokens
    if minimum is None:
        minimum = getattr(args, "eval_min_new_tokens", None)
    if minimum is not None:
        result["min_new_tokens"] = minimum
    return result


def validate_native_protocol(args):
    """Validate actual resolved native arguments before expensive model startup."""
    shared = _shared(args)
    configs = args.eval_datasets
    if len(configs) != 1 or configs[0].name != "aime2024":
        raise ValueError("AIME audit requires exactly the fixed AIME2024 dataset")
    config = configs[0]
    if config.n_samples_per_eval_prompt != 1 or args.rm_type != "math" or args.custom_rm_path is not None:
        raise ValueError("AIME audit requires one real math-scored answer per problem")
    template = config.apply_chat_template_kwargs
    if template is None:
        template = args.apply_chat_template_kwargs
    apply_template = config.apply_chat_template
    if apply_template is None:
        apply_template = args.apply_chat_template
    if not apply_template or not isinstance(template, dict) or template.get("enable_thinking") is not False:
        raise ValueError("Resolved AIME chat template must disable thinking")
    hf_config = json.loads((Path(args.hf_checkpoint) / "config.json").read_text())
    actual_dtype = hf_config.get("dtype") or hf_config.get("torch_dtype")
    declared_dtype = getattr(args, "sglang_dtype", "auto")
    if actual_dtype != "bfloat16" or declared_dtype not in ("auto", "bfloat16", "bf16"):
        raise ValueError("The actual HF and declared serving precision must resolve to BF16")
    if getattr(args, "sglang_quantization", None) is not None:
        raise ValueError("Quantized serving would change the BF16 validation protocol")
    protocol, protocol_sha = shared.build_protocol(
        tokenizer_path=args.hf_checkpoint, eval_data_path=config.path,
        sampling_params=_sampling(args, config), enable_thinking=False,
        backend={
            "device": args.sglang_device, "tp_size": args.sglang_tp_size,
            "attention_backend": args.sglang_attention_backend,
            "sampling_backend": args.sglang_sampling_backend,
            "dtype": "bfloat16" if args.bf16 else "unsupported",
            "disable_cuda_graph": args.sglang_disable_cuda_graph,
        },
    )
    if hashlib.sha256(Path(config.path).read_bytes()).hexdigest() != args.validation_dataset_sha256:
        raise ValueError("Held-out dataset changed after recipe validation")
    return shared, config, protocol, protocol_sha


def capture_native(rollout_id, args, data, extra_metrics, *, summary_writer=None):
    """Capture complete real native results using the shared HTTP summary schema."""
    completed = (extra_metrics or {}).get("validation/completed_optimizer_steps")
    if type(completed) is not int or completed not in (0, 20, 40, 60, 80, 100):
        raise ValueError("AIME audit requires explicit completed optimizer updates")
    offset = args.validation_optimizer_step_offset
    difference = completed - offset
    if difference < 0 or difference % args.num_steps_per_rollout:
        raise ValueError("Evaluation is not a completed rollout boundary")
    expected_version = str(1 + difference // args.num_steps_per_rollout)
    if set(data) != {"aime2024"}:
        raise ValueError("AIME audit requires exactly the fixed AIME2024 dataset")
    shared, config, protocol, protocol_sha = validate_native_protocol(args)
    with Path(config.path).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = {row["metadata"]["source_index"]: row for row in rows}
    info = data["aime2024"]
    samples, rewards, truncated = info["samples"], info["rewards"], info["truncated"]
    if len(samples) != 30 or len(rewards) != 30 or len(truncated) != 30:
        raise ValueError("A valid curve point requires all 30 native answers")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, local_files_only=True)
    results = []
    for sample, reward, cut in zip(samples, rewards, truncated, strict=True):
        raw = _jsonable(sample.to_dict() if hasattr(sample, "to_dict") else sample)
        index = raw.get("metadata", {}).get("source_index")
        if type(index) is not int or index not in rows or type(cut) is not bool:
            raise ValueError("Missing stable original question identity/truncation flag")
        row = rows[index]
        prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if raw["prompt"] != prompt or raw["label"] != row["label"]:
            raise ValueError("Native question/label does not match the fixed held-out problem")
        if reward not in (0, 1) or type(reward) not in (int, float, bool) or raw["reward"] != reward:
            raise ValueError("Expected the actual unnormalized binary math reward")
        if not raw.get("weight_versions") or {str(v) for v in raw["weight_versions"]} != {expected_version}:
            raise ValueError("Native answer has an unexpected serving weight version")
        if raw.get("status") not in ("completed", "truncated") or (raw["status"] == "truncated") != cut:
            raise ValueError("Generation status disagrees with truncation evidence")
        length = raw.get("response_length")
        if type(length) is not int or not 0 < length <= 4096 or len(raw["tokens"]) < length:
            raise ValueError("Native completion token evidence is incomplete")
        results.append({
            "question_index": index, "sample_id": row["metadata"]["sample_id"],
            "dataset_metadata": row["metadata"], "label": row["label"],
            "messages": row["messages"], "rendered_prompt": prompt,
            "optimizer_step": completed, "expected_weight_version": expected_version,
            "actual_weight_version": expected_version, "status": "success",
            "response_text": raw["response"], "output_token_ids": raw["tokens"][-length:],
            "completion_tokens": length, "truncated": cut, "math_score": int(reward),
            "native_sample": raw,
        })
    results.sort(key=lambda item: item["question_index"])
    if [item["question_index"] for item in results] != list(range(30)):
        raise ValueError("Expected exactly one answer for each original AIME question")
    model = Path(args.update_weight_disk_dir) / f"weight_v{int(expected_version):06d}"
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(not (model / filename).is_file() for filename in shards):
        raise ValueError("Evaluated policy HF checkpoint is incomplete")
    identity = {
        "model_path": str(model), "index_path": str(index_path),
        "index_sha256": shared.sha(index_path), "indexed_shards": len(shards),
        "full_weight_sha256_verified_by_report": False,
        "verification_scope": "index SHA256, indexed files, successful native export, and per-answer serving version; not a full tensor SHA/finite scan",
    }
    correct = sum(item["math_score"] for item in results)
    point = {
        "optimizer_step": completed, "model_path": str(model), "weight_version": expected_version,
        "weight_identity": identity, "protocol": protocol, "protocol_sha256": protocol_sha,
    }
    point_sha = shared.canonical_sha(point)
    for item in results:
        item.update(model_path=str(model), point_sha256=point_sha)
    root = Path(args.validation_trajectory_dir)
    directory = root / f"step_{completed:06d}"
    summary = {
        **point, "point_sha256": point_sha,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "valid_curve_point": True, "correct": correct, "questions": 30,
        "successful_requests": 30, "errors": 0,
        "truncated": sum(item["truncated"] for item in results),
        "accuracy": correct / 30, "accuracy_percent": correct / 30 * 100,
        "execution_settings": {
            "transport": "Slime native evaluation and native log hook",
            "tokenizer_path": str(args.hf_checkpoint),
            "declared_dtype": getattr(args, "sglang_dtype", "auto"),
            "resolved_hf_dtype": protocol["backend"]["dtype"],
            "mem_fraction_static": args.sglang_mem_fraction_static,
            "max_total_tokens": getattr(args, "sglang_max_total_tokens", None),
            "max_running_requests": args.sglang_max_running_requests,
            "context_length": args.sglang_context_length,
            "rollout_id": rollout_id, "lineage": args.validation_model_identity,
        },
        "raw_results": str(directory / "results.jsonl"),
        "curve_scope": "All 30 fixed AIME problems; standard real math rewards; no training/filtering use.",
    }
    if shared.valid_point_exists(root, completed):
        previous = json.loads((directory / "summary.json").read_text())
        existing_raw = (directory / "results.jsonl").read_text()
        proposed_raw = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in results)
        if previous.get("point_sha256") != point_sha or existing_raw != proposed_raw:
            raise FileExistsError("Refusing to replace an existing valid point with different evidence")
        return previous
    # Reject protocol mismatch before creating raw files for a purported new point.
    for other in root.glob("step_*/summary.json"):
        old = json.loads(other.read_text())
        if old.get("valid_curve_point") and old["protocol_sha256"] != protocol_sha:
            raise ValueError("Cannot merge a different evaluation protocol")
    questions = directory / "questions"
    questions.mkdir(parents=True, exist_ok=True)
    for item in results:
        shared.atomic_json(questions / f"{item['question_index']:02d}.json", item)
    temporary = directory / "results.jsonl.partial"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in results))
    os.replace(temporary, directory / "results.jsonl")
    shared.atomic_json(directory / "protocol.json", {**point, "point_sha256": point_sha})
    shared.write_summary(root, summary)
    if summary_writer is not None:
        for name in ("accuracy", "correct", "questions", "truncated"):
            summary_writer.add_scalar(f"validation/aime2024/{name}", summary[name], completed)
        summary_writer.flush()
    return summary


def log_eval(rollout_id, args, data, extra_metrics):
    """Native callback returns False so the original Slime logging still runs."""
    if getattr(args, "validation_trajectory_dir", None) is None:
        return False
    # Use an independent writer; Slime's Tensorboard adapter is a singleton.
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(log_dir=str(Path(args.validation_trajectory_dir) / "tensorboard")) as writer:
        capture_native(rollout_id, args, data, extra_metrics, summary_writer=writer)
    return False
