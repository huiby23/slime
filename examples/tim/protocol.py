"""Strict experiment configuration and native Slime command construction.

This module uses the standard library only. Dataset/model/output locations and
physical GPU selection belong to the launch command, not versioned presets.
"""

import copy
import json
import math
from pathlib import Path

SLIME_COMMIT = "0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e"
SECTIONS = {
    "schema_version": None,
    "name": None,
    "trainer": None,
    "rollout": {"precision", "temperature", "top_p", "top_k", "mem_fraction", "max_running_requests"},
    "training": {"updates", "prompts", "answers", "micro_batch_size", "response_tokens", "context_tokens", "seed"},
    "optimizer": {"lr", "weight_decay", "beta1", "beta2"},
    "evaluation": {"interval", "early_updates", "retain_early", "precision"},
    "topology": {"colocate", "sampler_replicas"},
    "noise": {"sigma", "seed"},
    "data": {"train_questions", "eval_questions", "screen_samples", "min_correct", "max_correct"},
    "runtime": {"image", "slime_commit", "cpus", "memory_gib"},
}


def merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, overrides=(), _parents=()):
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError("Configuration inheritance cycle: " + str(path))
    raw = json.loads(path.read_text())
    parent = raw.pop("extends", None)
    config = merge(load_config(path.parent / parent, _parents=_parents + (path,)), raw) if parent else raw
    for expression in overrides:
        key, separator, value = expression.partition("=")
        if not separator:
            raise ValueError("Overrides require section.key=JSON_VALUE")
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            if not isinstance(target.get(part), dict):
                raise ValueError("Unknown configuration key: " + key)
            target = target[part]
        if parts[-1] not in target:
            raise ValueError("Unknown configuration key: " + key)
        target[parts[-1]] = json.loads(value)
    validate_config(config)
    return config


def validate_config(config):
    if set(config) != set(SECTIONS):
        raise ValueError("Missing/unknown configuration sections: " + str(set(config) ^ set(SECTIONS)))
    for section, keys in SECTIONS.items():
        if keys is not None and (not isinstance(config[section], dict) or set(config[section]) != keys):
            raise ValueError("Missing/unknown keys in " + section)
    if (
        type(config["schema_version"]) is not int
        or config["schema_version"] != 1
        or config["runtime"]["slime_commit"] != SLIME_COMMIT
    ):
        raise ValueError("This protocol requires schema1 and its pinned Slime commit")
    if not isinstance(config["runtime"]["image"], str) or not config["runtime"]["image"].strip():
        raise ValueError("runtime.image must name a locally available Docker image")
    if (
        not isinstance(config["name"], str)
        or not config["name"]
        or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in config["name"])
    ):
        raise ValueError("Experiment name must use letters, digits, '-' or '_'")
    if config["trainer"] not in ("bf16", "nvfp4_sim"):
        raise ValueError("Unsupported trainer precision")
    mode = config["rollout"]["precision"]
    if mode not in ("bf16", "nvfp4_sim", "fixed_noise"):
        raise ValueError("Unsupported rollout precision")
    expected_eval = "bf16" if mode == "fixed_noise" else mode
    if config["evaluation"]["precision"] != expected_eval:
        raise ValueError("Evaluation precision must be explicit and consistent with the selected integration")
    train, ev, topo = config["training"], config["evaluation"], config["topology"]
    integers = [
        train[k] for k in ("updates", "prompts", "answers", "micro_batch_size", "response_tokens", "context_tokens")
    ]
    integers += [ev["interval"], topo["sampler_replicas"], config["runtime"]["cpus"], config["runtime"]["memory_gib"]]
    if any(type(x) is not int or x <= 0 for x in integers):
        raise ValueError("Batch, length, interval, topology and resource sizes must be positive integers")
    for value in (ev["early_updates"], ev["retain_early"], train["seed"], config["noise"]["seed"]):
        if type(value) is not int or value < 0:
            raise ValueError("Seeds and early-update counts must be nonnegative integers")
    if not ev["early_updates"] <= ev["retain_early"] <= train["updates"]:
        raise ValueError("Early checkpoint retention must cover every early evaluation")
    if train["updates"] > ev["early_updates"] and train["updates"] % ev["interval"]:
        raise ValueError(
            "The final update must have an evaluation: choose a dividing interval or cover it with early_updates"
        )
    if train["response_tokens"] >= train["context_tokens"]:
        raise ValueError("Context must leave space for the question")
    if train["prompts"] * train["answers"] % train["micro_batch_size"]:
        raise ValueError("Global batch size must be divisible by micro batch size")
    if type(topo["colocate"]) is not bool or (topo["colocate"] and topo["sampler_replicas"] != 1):
        raise ValueError("Colocated baseline supports exactly one sampler")
    if mode == "fixed_noise" and (not topo["colocate"] or config["trainer"] != "bf16"):
        raise ValueError("Fixed noise currently requires one colocated sampler and clean BF16 training")
    if config["trainer"] == "nvfp4_sim" and mode != "nvfp4_sim":
        raise ValueError("The simulated-training preset uses matching simulated rollout")
    opt = config["optimizer"]
    numeric = list(opt.values()) + [
        config["noise"]["sigma"],
        config["rollout"]["temperature"],
        config["rollout"]["top_p"],
        config["rollout"]["mem_fraction"],
    ]
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in numeric):
        raise ValueError("Numeric settings must be finite")
    if opt["lr"] <= 0 or opt["weight_decay"] < 0 or not all(0 <= opt[k] < 1 for k in ("beta1", "beta2")):
        raise ValueError("Invalid AdamW hyperparameters")
    if not 0 <= config["noise"]["sigma"] <= 1:
        raise ValueError("Noise sigma must be in [0,1]")
    if (
        config["rollout"]["temperature"] <= 0
        or not 0 < config["rollout"]["top_p"] <= 1
        or not 0 < config["rollout"]["mem_fraction"] < 1
    ):
        raise ValueError("Invalid rollout sampling/memory setting")
    if type(config["rollout"]["top_k"]) is not int or config["rollout"]["top_k"] == 0:
        raise ValueError("Rollout top_k must be -1 or positive")
    if (
        config["rollout"]["top_k"] < -1
        or type(config["rollout"]["max_running_requests"]) is not int
        or config["rollout"]["max_running_requests"] <= 0
    ):
        raise ValueError("Invalid rollout top_k or concurrency")
    data = config["data"]
    for key in ("train_questions", "eval_questions"):
        if data[key] is not None and (type(data[key]) is not int or data[key] <= 0):
            raise ValueError("Expected dataset counts must be positive integers or null")
    if not (
        type(data["screen_samples"]) is int
        and type(data["min_correct"]) is int
        and type(data["max_correct"]) is int
        and 0 <= data["min_correct"] <= data["max_correct"] <= data["screen_samples"]
    ):
        raise ValueError("Invalid data screening bounds")
    return config


def gpu_count(config):
    return 1 if config["topology"]["colocate"] else 1 + config["topology"]["sampler_replicas"]


def native_command(config, model="/model", output="/output"):
    """One optimizer update per generated prompt group; no correction is enabled."""
    validate_config(config)
    t, o, r, e, p = (config[k] for k in ("training", "optimizer", "rollout", "evaluation", "topology"))
    command = ["python", "-m", "slime_plugins.tim.bf16.train"]

    def add(flag, value=None):
        command.append("--" + flag)
        if value is not None:
            command.append(str(value))

    for key, value in {
        "actor-num-nodes": 1,
        "actor-num-gpus-per-node": 1,
        "rollout-num-gpus": p["sampler_replicas"],
        "rollout-num-gpus-per-engine": 1,
        "num-layers": 28,
        "hidden-size": 1024,
        "ffn-hidden-size": 3072,
        "num-attention-heads": 16,
        "num-query-groups": 8,
        "normalization": "RMSNorm",
        "norm-epsilon": "1e-6",
        "rotary-base": 1000000,
        "vocab-size": 151936,
        "kv-channels": 128,
        "hf-checkpoint": model,
        "load": model,
        "prompt-data": "/datasets/train.jsonl",
        "input-key": "prompt",
        "label-key": "label",
        "apply-chat-template-kwargs": '{"enable_thinking":false}',
        "rm-type": "math",
        "num-rollout": t["updates"],
        "rollout-batch-size": t["prompts"],
        "n-samples-per-prompt": t["answers"],
        "rollout-max-response-len": t["response_tokens"],
        "rollout-max-context-len": t["context_tokens"],
        "rollout-temperature": r["temperature"],
        "rollout-top-p": r["top_p"],
        "rollout-top-k": r["top_k"],
        "rollout-seed": t["seed"],
        "seed": t["seed"],
        "global-batch-size": t["prompts"] * t["answers"],
        "micro-batch-size": t["micro_batch_size"],
        "seq-length": t["context_tokens"],
        "advantage-estimator": "grpo",
        "kl-coef": 0,
        "kl-loss-coef": 0,
        "entropy-coef": 0,
        "eps-clip": 0.2,
        "eps-clip-high": 0.28,
        "optimizer": "adam",
        "lr": o["lr"],
        "min-lr": o["lr"],
        "lr-decay-style": "constant",
        "lr-warmup-iters": 0,
        "lr-decay-iters": t["updates"],
        "weight-decay": o["weight_decay"],
        "weight-decay-incr-style": "constant",
        "adam-beta1": o["beta1"],
        "adam-beta2": o["beta2"],
        "tensor-model-parallel-size": 1,
        "pipeline-model-parallel-size": 1,
        "context-parallel-size": 1,
        "expert-model-parallel-size": 1,
        "expert-tensor-parallel-size": 1,
        "attention-dropout": 0,
        "hidden-dropout": 0,
        "attention-backend": "flash",
        "sglang-mem-fraction-static": r["mem_fraction"],
        "sglang-context-length": t["context_tokens"],
        "sglang-max-running-requests": r["max_running_requests"],
        "sglang-attention-backend": "flashinfer",
        "sglang-chunked-prefill-size": -1,
        "custom-tis-function-path": "examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp",
        "custom-config-path": output + "/mismatch.json",
        "eval-interval": e["interval"],
        "n-samples-per-eval-prompt": 1,
        "eval-temperature": 1,
        "eval-top-p": 1,
        "eval-top-k": 1,
        "eval-max-response-len": t["response_tokens"],
        "eval-max-context-len": t["context_tokens"],
        "save": output + "/checkpoints",
        "save-interval": 1,
        "num-gpus-per-node": gpu_count(config),
        "rollout-data-transport": "object-store",
        "custom-eval-rollout-log-function-path": "slime_plugins.tim.bf16.checkpoint_policy.eval_log_hook",
        "tb-project-name": "tim",
        "tb-experiment-name": config["name"],
    }.items():
        add(key, value)
    command.extend(["--eval-prompt-data", "gsm8k", "/datasets/eval.jsonl"])
    for flag in (
        "swiglu",
        "group-query-attention",
        "use-rotary-position-embeddings",
        "disable-bias-linear",
        "qk-layernorm",
        "no-load-optim",
        "no-load-rng",
        "finetune",
        "bf16",
        "apply-chat-template",
        "rollout-shuffle",
        "accumulate-allreduce-grads-in-fp32",
        "attention-softmax-in-fp32",
        "sglang-disable-prefill-cuda-graph",
        "sglang-disable-decode-cuda-graph",
        "get-mismatch-metrics",
        "no-offload-train",
        "no-offload-rollout",
        "sglang-disable-radix-cache",
        "sglang-disable-overlap-schedule",
        "use-tensorboard",
    ):
        add(flag)
    if p["colocate"]:
        add("colocate")
        add("check-weight-update-equal")
    else:
        for key, value in {
            "update-weight-mode": "full",
            "update-weight-transport": "nccl",
            "router-policy": "round_robin",
        }.items():
            add(key, value)
    hook = "slime_plugins.tim.bf16.training_validation.before_train_step"
    if config["trainer"] == "nvfp4_sim":
        add("custom-model-provider-path", "slime_plugins.nvfp4_sim.megatron.model_provider")
        hook = "slime_plugins.nvfp4_sim.training_validation.before_train_step"
    add("custom-megatron-before-train-step-hook-path", hook)
    if r["precision"] == "nvfp4_sim":
        override = dict(
            architectures=["NVFP4SimQwen3ForCausalLM"],
            nvfp4_sim_reference_arithmetic=True,
            nvfp4_sim_weight_cache=True,
            nvfp4_sim_include_output_layer=True,
            nvfp4_sim_quantize_weights=True,
            nvfp4_sim_quantize_activations=True,
            nvfp4_sim_audit_path=output + "/rollout-quantization.json",
        )
        add("sglang-json-model-override-args", json.dumps(override))
    elif r["precision"] == "fixed_noise":
        override = dict(
            architectures=["FixedNoiseQwen3ForCausalLM"],
            fixed_noise_scale=config["noise"]["sigma"],
            fixed_noise_bank_manifest="/noise-bank/manifest.json",
            fixed_noise_audit_path=output + "/noise-audit.json",
            fixed_noise_control_path=output + "/noise-control.json",
        )
        add("sglang-json-model-override-args", json.dumps(override))
        add("eval-function-path", "slime_plugins.tim.fixed_noise.eval.generate_rollout")
    return command


def runtime_environment(config, model="/model"):
    env = {
        "PYTHONPATH": "/runtime/extensions:/runtime/slime:/root/Megatron-LM",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "MAX_JOBS": "4",
        "CMAKE_BUILD_PARALLEL_LEVEL": "4",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "RAY_USAGE_STATS_ENABLED": "0",
        "WANDB_MODE": "offline",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "QWEN3_06B_PATH": model,
        "TIM_EARLY_EVAL_UPDATES": str(config["evaluation"]["early_updates"]),
        "TIM_RETAIN_EARLY_CHECKPOINTS": str(config["evaluation"]["retain_early"]),
        "TENSORBOARD_DIR": "/output/tensorboard",
        "TMPDIR": "/output/tmp/work",
        "RAY_TMPDIR": "/output/tmp/ray",
        "WANDB_DIR": "/output/logs",
        "WANDB_CACHE_DIR": "/output/cache/wandb",
    }
    for key, folder in {
        "HF_HOME": "hf",
        "PIP_CACHE_DIR": "pip",
        "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "extensions",
        "TRITON_CACHE_DIR": "triton",
        "FLASHINFER_CACHE_DIR": "flashinfer",
        "XDG_CACHE_HOME": "xdg",
        "CUDA_CACHE_PATH": "cuda",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
    }.items():
        env[key] = "/output/cache/" + folder
    mode = config["rollout"]["precision"]
    if mode == "nvfp4_sim":
        env["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "slime_plugins.tim.nvfp4_models"
    elif mode == "fixed_noise":
        env["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "slime_plugins.tim.fixed_noise.sglang_models"
    return env
