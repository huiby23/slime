"""Small real MoE integration: EP, FP32 gradients, CPU Adam and full DCP resume.

Run with torchrun on qualified idle GPUs. This does not evaluate the 30B model.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from w4a8_sim.megatron import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(expert_model_parallel_size=world)
    model_parallel_cuda_manual_seed(42)
    config = TransformerConfig(
        num_layers=2, hidden_size=128, num_attention_heads=2, num_query_groups=1,
        ffn_hidden_size=128, num_moe_experts=4, moe_ffn_hidden_size=128,
        moe_router_topk=2, moe_router_dtype="fp32", moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall", expert_model_parallel_size=world,
        expert_tensor_parallel_size=1, bf16=True, params_dtype=torch.bfloat16,
        normalization="RMSNorm", qk_layernorm=True, add_bias_linear=False,
        gated_linear_unit=True, activation_func=F.silu, gradient_accumulation_fusion=True,
        hidden_dropout=0.0, attention_dropout=0.0, moe_aux_loss_coeff=0.0,
    )
    model = GPTModel(config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=4, moe_grouped_gemm=True, qk_layernorm=True),
        vocab_size=256, max_sequence_length=32, share_embeddings_and_output_weights=False,
        position_embedding_type="rope").cuda()
    original_keys = set(model.state_dict())
    install(model)
    assert set(model.state_dict()) == original_keys
    ddp = DistributedDataParallel(config, DistributedDataParallelConfig(
        grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=True),
        model, disable_bucketing=True)
    opt_config = OptimizerConfig(bf16=True, params_dtype=torch.bfloat16,
        use_distributed_optimizer=True, lr=1e-3, min_lr=1e-3, weight_decay=0.0,
        optimizer_cpu_offload=True, use_torch_optimizer_for_cpu_offload=True,
        optimizer_offload_fraction=1.0,
        use_precision_aware_optimizer=True,
        overlap_cpu_optimizer_d2h_h2d=False)
    optimizer = get_megatron_optimizer(opt_config, [ddp])
    for part in getattr(optimizer, "chained_optimizers", [optimizer]):
        assert part.optimizer.offload_fraction == 1.0
        assert part.optimizer.cpu_optimizers and part.optimizer.gpu_optimizer is None
    token_generator = torch.Generator(device="cuda").manual_seed(1000 + rank)
    tokens = torch.randint(0, 256, (1, 19), device="cuda", generator=token_generator)
    positions = torch.arange(19, device="cuda").unsqueeze(0)
    mask = torch.triu(torch.ones(1, 1, 19, 19, device="cuda", dtype=torch.bool), diagonal=1)
    initial = {n:p.detach().clone() for n,p in model.named_parameters()}

    def step():
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        logits = ddp(tokens, positions, mask)
        loss = F.cross_entropy(logits.float().reshape(-1,256), tokens.roll(-1,1).reshape(-1))
        loss.backward()
        ddp.finish_grad_sync()
        assert all(p.main_grad.isfinite().all() for p in model.parameters() if p.requires_grad)
        assert any(p.main_grad.norm()>0 for n,p in model.named_parameters() if ".experts." in n)
        success, norm, _ = optimizer.step()
        assert success and torch.isfinite(torch.as_tensor(norm))
        return dict(loss=float(loss.detach()), grad_norm=float(norm))

    steps = [step(), step()]
    assert any(not torch.equal(initial[n],p) for n,p in model.named_parameters() if ".experts." in n)
    assert any(not torch.equal(initial[n],p) for n,p in model.named_parameters() if "linear_qkv.weight" in n)
    metadata = {"distrib_optim_sharding_type": "fully_reshardable"}

    def checkpoint_state(loading=False):
        state = model.sharded_state_dict()
        return {"model":state, "optimizer":optimizer.sharded_state_dict(state, is_loading=loading, metadata=metadata)}

    def optimizer_tensors():
        result = {}
        chain = getattr(optimizer, "chained_optimizers", [optimizer])
        for i, part in enumerate(chain):
            inner = part.optimizer
            inner.state_dict()  # Public hook syncs offloaded optimizer state.
            for j, group in enumerate(inner.param_groups):
                for k, parameter in enumerate(group["params"]):
                    for name, value in inner.state.get(parameter, {}).items():
                        if isinstance(value, torch.Tensor):
                            assert value.device.type == "cpu", (name,value.device)
                            result[f"{i}/{j}/{k}/{name}"] = value.detach().cpu().clone()
        return result

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "full-checkpoint"
    if rank == 0:
        path.mkdir(exist_ok=False)
    dist.barrier()
    dist_checkpointing.save(checkpoint_state(), str(path))
    saved_parameters = {n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    saved_optimizer = optimizer_tensors()
    saved_rng = torch.cuda.get_rng_state()
    expected_step = step()
    expected = {n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    restored = dist_checkpointing.load(checkpoint_state(loading=True), str(path))
    model.load_state_dict(restored["model"])
    optimizer.load_state_dict(restored["optimizer"])
    restored_optimizer = optimizer_tensors()
    diffs = {k:float((v-restored_optimizer[k]).abs().max()) for k,v in saved_optimizer.items()}
    model_diffs = {n:float((p.detach().cpu()-saved_parameters[n]).abs().max()) for n,p in model.named_parameters()}
    diagnostic = dict(optimizer_keys=list(saved_optimizer),optimizer_max_difference=diffs,model_max_difference=model_diffs,
                      expected_step=expected_step)
    (args.output/f"resume-audit-rank-{rank}.json").write_text(json.dumps(diagnostic,indent=2)+"\n")
    assert saved_optimizer and all(d==0 for d in diffs.values()), diffs
    assert all(d==0 for d in model_diffs.values()), model_diffs
    torch.cuda.set_rng_state(saved_rng)
    resumed_step = step()
    diagnostic["resumed_step"] = resumed_step
    (args.output/f"resume-audit-rank-{rank}.json").write_text(json.dumps(diagnostic,indent=2)+"\n")
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach().cpu(), expected[name], rtol=0, atol=0)
    counters = dict(model._w4a8_recipe.counters)
    assert all(counters.get(k,0)>0 for k in ("fprop","dgrad","wgrad","activation_e4m3","gradient_e5m2","weight_nvfp4"))
    qkv = model.decoder.layers[0].self_attention.linear_qkv.weight
    report = dict(passed=True, rank=rank, world_size=world, expert_parallel_size=world,
        steps=steps, expected_step=expected_step, resumed_step=resumed_step,
        exact_optimizer_resume=True, full_distributed_checkpoint=True, cpu_optimizer_offload=True,
        counters=counters, projection_modules=list(model._w4a8_modules),
        expert_modules=list(model._w4a8_grouped_modules),
        qkv_sha256=hashlib.sha256(qkv.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
    (args.output/f"validation-rank-{rank}.json").write_text(json.dumps(report,indent=2)+"\n")
    reports = [None]*world
    dist.all_gather_object(reports, report)
    assert len({r["qkv_sha256"] for r in reports}) == 1
    if rank==0:
        (args.output/"validation.json").write_text(json.dumps(dict(passed=True,ranks=reports),indent=2)+"\n")
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__=="__main__":
    main()
