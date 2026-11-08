"""Two-step Qwen3-0.6B Ascend GRPO installation validation.

Uses its own local Ray cluster and temporary directory. It does not stop or
attach to any existing Ray cluster. Rewards are synthetic; this is not an
accuracy, convergence, or throughput benchmark. Weight synchronization uses
full Hugging Face checkpoints on disk for a conservative NPU baseline.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=os.getenv("SLIME_WORKSPACE", "/root/hby"))
    parser.add_argument("--response-length", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=2)
    options, extra_train_args = parser.parse_known_args()
    workspace = Path(options.workspace).resolve()
    repo = workspace / "slime"
    model = workspace / "models/Qwen3-0.6B"
    data = workspace / "data/smoke-open-prompts.jsonl"
    if not (model / "config.json").is_file() or not data.is_file():
        raise SystemExit(f"Missing smoke model or prompts: {model}, {data}")
    if options.rollouts < 2:
        raise SystemExit("Use at least two rollouts to exercise updated policy weights")
    devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 2 or any(not item.strip() for item in devices):
        raise SystemExit("Set SMOKE_NPU_DEVICES to exactly two available NPU IDs")
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    output = workspace / "logs" / ("slime-npu-smoke-" + run_id)
    output.mkdir(parents=True)
    # Ray's AF_UNIX sockets include a long session suffix; keep the parent
    # path short enough for Linux's 107-byte socket path limit.
    ray_tmp = workspace / ".r" / uuid.uuid4().hex[:8]
    ray_tmp.mkdir(parents=True)
    server_config = output / "sglang-loopback.json"
    server_config.write_text(json.dumps({"sglang": [{
        "name": "actor",
        "update_weights": True,
        "num_gpus_per_engine": 1,
        "server_groups": [{
            "worker_type": "regular", "num_gpus": 1,
            "overrides": {"host": "127.0.0.1"},
        }],
    }]}, indent=2) + "\n")
    os.environ["SLIME_ACCELERATOR"] = "npu"
    os.environ["SLIME_HOST_IP"] = "127.0.0.1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    sys.path.insert(0, str(repo))
    import ray
    import torch
    import torch_npu

    if not torch.npu.is_available() or torch.npu.device_count() != 2:
        raise RuntimeError("The smoke environment must expose exactly two NPUs")
    print(f"SMOKE_OUTPUT={output}", flush=True)
    print("SMOKE_ONLY: synthetic alternating rewards; no quality claims", flush=True)
    # Ray owns this local cluster. shutdown() below only terminates this cluster.
    try:
        context = ray.init(
            address="local",
            namespace="slime-npu-smoke-" + run_id,
            num_cpus=8,
            resources={"NPU": 2},
            object_store_memory=2 * 1024**3,
            include_dashboard=False,
            _node_ip_address="127.0.0.1",
            _temp_dir=str(ray_tmp),
        )
        resources = ray.cluster_resources()
        print(f"SMOKE_RAY_RESOURCES={resources}", flush=True)
        if resources.get("NPU", 0) != 2:
            raise RuntimeError("Ray did not register both NPU resources")
        model_args = shlex.split(
            "--swiglu --num-layers 28 --hidden-size 1024 --ffn-hidden-size 3072 "
            "--num-attention-heads 16 --group-query-attention --num-query-groups 8 "
            "--use-rotary-position-embeddings --disable-bias-linear "
            "--normalization RMSNorm --norm-epsilon 1e-6 --rotary-base 1000000 "
            "--vocab-size 151936 --kv-channels 128 --qk-layernorm"
        )
        arguments = model_args + [
            "--bf16", "--hf-checkpoint", str(model), "--load", str(model), "--ref-load", str(model),
            "--save", str(output / "checkpoints"), "--save-interval", "1",
            "--ckpt-format", "torch_dist",
            "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1",
            "--rollout-num-gpus", "1", "--rollout-num-gpus-per-engine", "1",
            "--num-gpus-per-node", "2", "--no-offload-train", "--no-offload-rollout",
            "--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1",
            "--context-parallel-size", "1", "--expert-model-parallel-size", "1",
            "--expert-tensor-parallel-size", "1", "--use-cpu-initialization",
            "--prompt-data", str(data), "--input-key", "prompt", "--label-key", "label",
            "--apply-chat-template", "--apply-chat-template-kwargs", '{"enable_thinking": false}',
            "--num-rollout", str(options.rollouts), "--rollout-batch-size", "1",
            "--n-samples-per-prompt", "2", "--global-batch-size", "2", "--micro-batch-size", "1",
            "--rollout-max-response-len", str(options.response_length), "--rollout-temperature", "1.0",
            "--custom-rm-path", "slime_plugins.ascend_smoke.synthetic_reward",
            "--custom-megatron-before-train-step-hook-path",
            "slime_plugins.ascend_smoke.record_attention_backend",
            "--advantage-estimator", "grpo", "--kl-coef", "0", "--entropy-coef", "0",
            "--optimizer", "adam", "--lr", "1e-6", "--lr-decay-style", "constant",
            "--weight-decay", "0", "--attention-dropout", "0", "--hidden-dropout", "0",
            "--attention-softmax-in-fp32", "--accumulate-allreduce-grads-in-fp32",
            "--seq-length", "256", "--max-position-embeddings", "32768", "--attention-backend", "flash",
            "--sglang-config", str(server_config),
            "--router-prometheus-host", "127.0.0.1",
            "--sglang-device", "npu", "--sglang-attention-backend", "ascend",
            "--sglang-sampling-backend", "pytorch", "--sglang-disable-cuda-graph",
            "--sglang-context-length", "256", "--sglang-max-running-requests", "2",
            "--sglang-mem-fraction-static", "0.25",
            "--update-weight-mode", "full", "--update-weight-transport", "disk",
            "--update-weight-disk-dir", str(output / "weight-updates"),
            "--update-weight-disk-keep-files",
            "--ci-save-grad-norm", str(output / "grad_norm_{role}_{rollout_id}_{step_id}.pt"),
            "--save-debug-rollout-data", str(output / "rollout_{rollout_id}.pt"),
        ]
        if extra_train_args[:1] == ["--"]:
            extra_train_args = extra_train_args[1:]
        sys.argv = [str(repo / "train.py")] + arguments + extra_train_args
        (output / "invocation.json").write_text(json.dumps(sys.argv, indent=2) + "\n")
        from slime.utils.arguments import parse_args
        from slime.observability.logging_utils import configure_logger
        from train import main as train_main

        args, restore_plan = parse_args(return_restore_plan=True)
        if not args.bf16 or args.fp16:
            raise RuntimeError("The Ascend installation smoke requires BF16 training")
        configure_logger()
        train_main(args, restore_plan)
        # These norms are saved by the real Megatron training loop after a
        # successful optimizer.step(); the loop asserts update_successful.
        gradient_norms = []
        for rollout_id in range(options.rollouts):
            rollout_path = output / f"rollout_{rollout_id}.pt"
            norm_path = output / f"grad_norm_actor_{rollout_id}_0.pt"
            if not rollout_path.is_file() or not norm_path.is_file():
                raise RuntimeError(f"Missing completed rollout or optimizer evidence: {rollout_id}")
            norm = float(torch.load(norm_path, map_location="cpu", weights_only=True))
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError(f"Invalid/non-updating synthetic smoke gradient: {rollout_id}: {norm}")
            gradient_norms.append(norm)
        weight_versions = []
        for version in range(1, options.rollouts + 2):
            version_path = output / "weight-updates" / f"weight_v{version:06d}"
            if not any(version_path.glob("*.safetensors")):
                raise RuntimeError(f"Missing full-weight synchronization checkpoint: {version_path}")
            weight_versions.append(version)
        result = {
            "status": "completed", "rollouts": options.rollouts, "model": str(model),
            "physical_npus": devices, "reward": "synthetic-index-parity",
            "weight_update_transport": "disk", "precision": "bf16",
            "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "optimizer_steps": len(gradient_norms), "gradient_norms": gradient_norms,
            "weight_versions": weight_versions,
            "training_attention": [
                json.loads(path.read_text()) for path in sorted(output.glob("attention_*.json"))
            ],
            "quality_or_throughput_benchmark": False,
        }
        (output / "success.json").write_text(json.dumps(result, indent=2) + "\n")
        print("SLIME_NPU_SMOKE_COMPLETED=" + str(output / "success.json"), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
