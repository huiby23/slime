"""Two optimizer updates and checkpoint round-trip with real Megatron/TE on CUDA.

Run in the baseline's runtime image on an exclusively available GPU. The small
model checks integration; it is not an accuracy/convergence experiment.
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from slime_plugins.nvfp4_sim.megatron import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires an exclusively available CUDA GPU")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as work:
        dist.init_process_group("nccl", init_method=f"file://{work}/distributed", rank=0, world_size=1)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(42)
        try:
            config = TransformerConfig(
                num_layers=2,
                hidden_size=64,
                num_attention_heads=4,
                ffn_hidden_size=128,
                bf16=True,
                params_dtype=torch.bfloat16,
                normalization="RMSNorm",
                qk_layernorm=True,
                add_bias_linear=False,
                gated_linear_unit=True,
                activation_func=F.silu,
                gradient_accumulation_fusion=True,
                hidden_dropout=0.0,
                attention_dropout=0.0,
            )
            model = GPTModel(
                config=config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
                vocab_size=128,
                max_sequence_length=32,
                share_embeddings_and_output_weights=True,
                position_embedding_type="rope",
            ).cuda()
            before_keys = set(model.state_dict())
            install(model)
            assert set(model.state_dict()) == before_keys
            recipe = model._nvfp4_recipe
            ddp = DistributedDataParallel(
                config,
                DistributedDataParallelConfig(
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=False,
                    use_distributed_optimizer=True,
                ),
                model,
                disable_bucketing=True,
            )
            opt_config = OptimizerConfig(
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                lr=1e-3,
                min_lr=1e-3,
                weight_decay=0.0,
            )
            with torch.no_grad():
                optimizer = get_megatron_optimizer(opt_config, [ddp])
            parameter = model.decoder.layers[0].mlp.linear_fc1.weight
            initial_weight = parameter.detach().clone()
            steps = []
            for step in range(2):
                ddp.zero_grad_buffer()
                optimizer.zero_grad()
                tokens = torch.randint(0, 128, (1, 19), device="cuda")
                positions = torch.arange(19, device="cuda").unsqueeze(0)
                mask = torch.triu(torch.ones(1, 1, 19, 19, device="cuda", dtype=torch.bool), diagonal=1)
                logits = ddp(tokens, positions, mask)
                loss = F.cross_entropy(logits.float().reshape(-1, 128), tokens.roll(-1, 1).reshape(-1))
                loss.backward()
                ddp.finish_grad_sync()
                gradients = [p.main_grad for p in model.parameters() if p.requires_grad]
                assert all(torch.isfinite(g).all() for g in gradients)
                assert parameter.main_grad.abs().sum() > 0
                success, grad_norm, _ = optimizer.step()
                assert success and torch.isfinite(torch.as_tensor(grad_norm))
                steps.append({"step": step, "loss": loss.item(), "grad_norm": float(grad_norm)})
            assert not torch.equal(parameter, initial_weight), "Optimizer did not update simulated weights"
            for key in ("fprop", "dgrad", "wgrad", "rht_x", "rht_g", "sr_dgrad", "sr_wgrad"):
                assert recipe.counters[key] >= len(model._nvfp4_modules) * 2, (key, recipe.counters)
            args.output.mkdir(parents=True, exist_ok=True)
            checkpoint = args.output / "checkpoint.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    # Distributed optimizer state_dict contains metadata, not
                    # the separate distributed parameter/moment checkpoint.
                    "optimizer_metadata": optimizer.state_dict(),
                    "rng": torch.cuda.get_rng_state(),
                },
                checkpoint,
            )
            restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(restored["model"])
            optimizer.load_state_dict(restored["optimizer_metadata"])
            torch.cuda.set_rng_state(restored["rng"])
            report = {
                "passed": True,
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "patched_linears": list(model._nvfp4_modules),
                "recipe_counters": dict(recipe.counters),
                "steps": steps,
                "weights_changed": True,
                "model_checkpoint_round_trip": True,
                "optimizer_metadata_round_trip": True,
                "checkpoint": str(checkpoint),
            }
            (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
