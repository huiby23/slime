"""AIME2024 evaluation identity shared by checkpoint manifests and assessors."""
import hashlib
import json
import os
from pathlib import Path

AIME_SHA = "76a1ecaab458f041f89f43fd98f882ec69c4d16b60bb14d38c003daf22e0d36a"
SAMPLING = {
    "temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 4096,
    "stop": None, "stop_token_ids": None, "skip_special_tokens": False,
    "no_stop_trim": True, "spaces_between_special_tokens": False,
}
BACKEND = {
    "device": "npu", "tp_size": 8, "attention_backend": "hby_c8v16_sim",
    "sampling_backend": "pytorch", "dtype": "bfloat16", "disable_cuda_graph": True,
}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_protocol(tokenizer_path, eval_data_path, sampling_params, enable_thinking, backend):
    data, tokenizer = Path(eval_data_path), Path(tokenizer_path)
    if file_sha(data) != AIME_SHA:
        raise ValueError("AIME2024 requires the same full 30-question input")
    sample = SAMPLING | {key: value for key, value in sampling_params.items() if key in SAMPLING or value is not None}
    if sample != SAMPLING or enable_thinking is not False or backend != BACKEND:
        raise ValueError("AIME must use greedy, nonthinking W16A16C8V16 TP8 with 4096 answer tokens")
    workspace = Path(os.environ["HBY_C8V16_WORKSPACE"])
    repository = Path(os.environ["HBY_C8V16_REPOSITORY"])
    base = workspace / "models/Qwen3-30B-A3B"
    tokens = {name: file_sha(tokenizer / name) for name in ("tokenizer.json", "tokenizer_config.json")}
    if any(file_sha(base / name) != digest for name, digest in tokens.items()):
        raise ValueError("AIME checkpoint tokenizer differs from the original Qwen3 tokenizer")
    policy = json.loads(Path(os.environ["HBY_C8V16_POLICY_FILE"]).read_text())
    from c8v16_profile import implementation_sha256
    source_hash = implementation_sha256()
    if canonical_sha(policy) != os.environ["HBY_C8V16_POLICY_SHA256"] or source_hash != os.environ["HBY_C8V16_IMPLEMENTATION_SHA256"]:
        raise ValueError("AIME precision policy or implementation changed since launch")
    math_dir = repository / "slime/rollout/rm_hub"
    identity = {
        "protocol_version": "aime2024-30-greedy-c8v16-v1",
        "questions": 30, "dataset_sha256": AIME_SHA, "samples_per_question": 1,
        "sampling_params": sample, "enable_thinking": False, "add_generation_prompt": True,
        "tokenizer_add_special_tokens": False, "tokenizer_sha256": tokens,
        "reward": {"rm_type": "math", "custom_rm_path": None, "function": "grade_answer_verl(response, label)",
                   "math_utils_sha256": file_sha(math_dir / "math_utils.py"),
                   "dispatcher_sha256": file_sha(math_dir / "__init__.py")},
        "backend": backend, "attention_simulation": {"policy": policy, "policy_sha256": canonical_sha(policy),
                                                    "implementation_sha256": source_hash, "quantized_backward": False},
    }
    return identity, canonical_sha(identity)
