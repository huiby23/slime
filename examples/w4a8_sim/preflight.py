"""Parse the prepared command in the pinned runtime without allocating CUDA."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch


def main():
    if torch.cuda.is_available():
        raise RuntimeError("Run CPU preflight with CUDA_VISIBLE_DEVICES empty")
    path = Path(sys.argv[1])
    plan = json.loads(path.read_text())
    dense = "arms" in plan
    mode = sys.argv[2] if len(sys.argv) > 2 else "w4a8"
    if dense:
        shared = plan
        plan = dict(shared, **shared["arms"][mode])
    os.environ.update(plan["environment"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    command = plan["training_command"]
    world_size = int(command[command.index("--actor-num-gpus-per-node") + 1])
    os.environ.update(WORLD_SIZE=str(world_size), RANK="0", LOCAL_RANK="0")
    sys.argv = plan["training_command"][1:]
    import train

    # Only CLI validation queries capability; the actual A100 kernels and EP
    # collectives are checked separately by the CUDA integration program.
    with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
        args, _ = train.parse_args(return_restore_plan=True)
    if not args.bf16 or args.fp8 or args.fp4:
        raise ValueError("Native FP4/FP8 must stay disabled on A100")
    assert args.expert_model_parallel_size == (1 if dense else 4) and args.tensor_model_parallel_size == 1
    assert args.optimizer_cpu_offload and args.use_precision_aware_optimizer
    assert args.offload_train and args.offload_rollout and args.save_interval == 1
    assert not args.no_save_optim and not args.no_save_rng and not args.async_save
    if dense:
        assert os.environ["TENSORBOARD_DIR"] == args.tensorboard_dir
        assert args.global_batch_size % world_size == 0
        from w4a8_sim.sglang_models.qwen3 import W4A8SimQwen3ForCausalLM

        assert W4A8SimQwen3ForCausalLM.__name__ == "W4A8SimQwen3ForCausalLM"
        assert not args.untie_embeddings_and_output_weights and args.num_layers == 36
        assert bool(args.custom_model_provider_path) == (mode == "w4a8")
    else:
        from w4a8_sim.sglang_models.qwen3_moe import W4A8SimQwen3MoeForCausalLM

        assert W4A8SimQwen3MoeForCausalLM.__name__ == "W4A8SimQwen3MoeForCausalLM"
    model = Path(plan["model_path"])
    index = json.loads((model / "model.safetensors.index.json").read_text())
    missing = [f for f in set(index["weight_map"].values()) if not (model / f).is_file()]
    report = dict(
        argument_parse_passed=True,
        backend_imports_passed=True,
        weights_complete=not missing,
        missing_weight_shards=sorted(missing),
        no_cuda_allocation=not torch.cuda.is_initialized(),
        full_model_smoke_required=True,
        formal_launch_ready=False,
    )
    (path.parent / (f"preflight-{mode}.json" if dense else "preflight.json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
