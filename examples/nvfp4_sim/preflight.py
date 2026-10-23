"""Resolve the original baseline's real CLI/YAML and both simulator entry points on CPU."""

import json
import os
import sys
from pathlib import Path

import torch


def main():
    root = Path(sys.argv[1])
    config = json.loads((root / "config.json").read_text())
    os.environ.update(config["environment"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    assert not torch.cuda.is_available(), "Run preflight without visible CUDA devices"
    sys.argv = config["training_command"][1:]
    from slime.utils.arguments import parse_args
    from slime.utils.misc import load_function

    args = parse_args()
    assert args.bf16 and args.fp4 is None and args.fp8 is None
    assert args.custom_model_provider_path == "nvfp4_sim.megatron.model_provider"
    assert callable(load_function(args.custom_model_provider_path))
    assert args.nvfp4_sim_include_output_layer and args.nvfp4_sim_chunk_blocks > 0
    assert args.num_rollout == config["training"]["updates"]
    assert args.use_tis == config["training"]["importance_correction"]
    assert args.sglang_disable_decode_cuda_graph and args.sglang_disable_prefill_cuda_graph
    assert args.tensor_model_parallel_size == args.pipeline_model_parallel_size == args.context_parallel_size == 1
    architecture = json.loads(args.sglang_json_model_override_args)["architectures"]
    from sglang.srt.models.registry import ModelRegistry

    model_class, _ = ModelRegistry.resolve_model_cls(architecture)
    assert model_class.__name__ == "NVFP4SimQwen3ForCausalLM"
    report = {
        "passed": True,
        "baseline_slime": __import__("slime").__file__,
        "training_provider": args.custom_model_provider_path,
        "rollout_model": model_class.__name__,
        "bf16": args.bf16,
        "native_fp4": args.fp4,
        "native_fp8": args.fp8,
        "include_output_layer": args.nvfp4_sim_include_output_layer,
        "updates": args.num_rollout,
        "use_tis": args.use_tis,
        "cpu_only": True,
    }
    (root / "configuration-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
