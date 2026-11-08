"""Pinned Megatron/MindSpeed integration probe, NOT a GRPO training run.

Default --inspect-only constructs the real config and module spec, without a
model, distributed communicator or accelerator kernel. --execute-npu is only
for the parent-controlled idle-device test window. It uses one random tiny
MoE layer, causal GQA 8:1/head128, BF16 backbone, C4V8 forward QDQ + high-
precision identity-STE backward, and one real FP32-master Hybrid Adam update.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--inspect-only', action='store_true')
    mode.add_argument('--execute-npu', action='store_true')
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--physical-device', type=int, default=1)
    parser.add_argument('--core-module',choices=('device_attention','fast_attention'),default='device_attention')
    parser.add_argument('--sequence-length', type=int, default=128)
    options = parser.parse_args()
    if not 8 <= options.sequence_length <= 129:
        raise ValueError('Tiny adaptation probe sequence length must be 8..129')
    if options.execute_npu and os.environ.get('HBY_C4V8_PROBE_NPU_RESERVED') != str(options.physical_device):
        raise RuntimeError('Parent must first reserve the idle physical NPU and set HBY_C4V8_PROBE_NPU_RESERVED')
    if options.execute_npu and os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != str(options.physical_device):
        raise RuntimeError('Probe must expose exactly its reserved physical NPU')
    options.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', kind='tiny random-model adaptation; NOT RL',
                  real_grpo_updates=0, execution_mode='npu' if options.execute_npu else 'spec-inspection-only',
                  started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                  quantized_backward=False, backward='FP32 replay/identity-STE; BF16 leaf gradients',
                  physical_npu=options.physical_device if options.execute_npu else None)
    report_path = options.output_dir / 'report.json'

    def save():
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    initialized = False
    try:
        policy = json.loads(options.policy.read_text())
        policy_hash = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        from numerical_profile import implementation_profile,python_paths,digest as profile_digest
        workspace=Path(__file__).resolve().parents[2]
        implementation=implementation_profile(workspace,options.core_module)
        sys.path[:0]=list(map(str,python_paths(workspace,options.core_module)))
        os.environ['PYTHONPATH']=os.pathsep.join(map(str,python_paths(workspace,options.core_module)))+os.pathsep+os.environ.get('PYTHONPATH','')
        os.environ['HBY_C4V8_CORE_MODULE'] = options.core_module
        os.environ['HBY_C4V8_IMPLEMENTATION_ID'] = implementation['implementation_id']
        os.environ['HBY_C4V8_IMPLEMENTATION_SHA256'] = profile_digest(implementation)
        if options.core_module=='fast_attention' and os.environ.get('HBY_C4V8_ALLOW_UNVALIDATED_FUSION')!='1':
            raise ValueError('Fast probe requires explicit candidate fusion opt-in')
        os.environ['HBY_C4V8_POLICY_FILE'] = str(options.policy.resolve())
        os.environ['HBY_C4V8_POLICY_SHA256'] = policy_hash
        os.environ['HBY_C4V8_TRAIN_AUDIT_DIR'] = str(options.output_dir/'trainer-backend-audit')
        report.update(policy=policy, policy_sha256=policy_hash,implementation=implementation,
                      implementation_sha256=profile_digest(implementation))
        # Real MindSpeed feature registration sees the same relevant model
        # choices as production. Script-only options must not become features.
        sys.argv = [sys.argv[0], '--transformer-impl', 'transformer_engine', '--bf16',
                    '--no-gradient-accumulation-fusion', '--moe-grouped-gemm',
                    '--context-parallel-size', '1']
        import mindspeed.megatron_adaptor  # noqa: F401
        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state, tensor_parallel
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
        from megatron.core.optimizer import Adam, CPUAdam
        from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
        from megatron.core.packed_seq_params import PackedSeqParams
        import megatron_spec
        import device_attention
        import device_quant
        import exact_fp32
        from megatron_spec import C4V8CoreAttention, get_spec

        config = TransformerConfig(
            num_layers=1, hidden_size=1024, num_attention_heads=8, num_query_groups=1,
            kv_channels=128, ffn_hidden_size=128, moe_ffn_hidden_size=128,
            num_moe_experts=4, moe_router_topk=2, moe_router_dtype='fp32',
            moe_router_load_balancing_type='none', moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=True, moe_token_dispatcher_type='alltoall',
            normalization='RMSNorm', layernorm_epsilon=1e-6, qk_layernorm=True,
            activation_func=torch.nn.functional.silu, gated_linear_unit=True,
            add_bias_linear=False, add_qkv_bias=False,
            hidden_dropout=0.0, attention_dropout=0.0,
            bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
            use_cpu_initialization=True, perform_initialization=True,
            gradient_accumulation_fusion=False, bias_activation_fusion=False,
            bias_dropout_fusion=False, masked_softmax_fusion=False,
            apply_rope_fusion=False, sequence_parallel=False,
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            expert_model_parallel_size=1, expert_tensor_parallel_size=1,
            context_parallel_size=1)
        report['model_config'] = {k: str(getattr(config, k)) for k in (
            'num_layers', 'hidden_size', 'num_attention_heads', 'num_query_groups', 'kv_channels',
            'num_moe_experts', 'moe_ffn_hidden_size', 'moe_router_topk', 'params_dtype', 'qk_layernorm')}
        # The public rank setters permit inspecting the real stage-local spec
        # without constructing a communicator. They do not initialize a model.
        if not options.execute_npu:
            parallel_state.set_pipeline_model_parallel_rank(0)
            parallel_state.set_pipeline_model_parallel_world_size(1)
        else:
            torch.npu.set_device(0)
            dist.init_process_group('hccl', init_method='file://' + str((options.output_dir/'hccl-init').resolve()),
                                    rank=0, world_size=1, timeout=timedelta(minutes=5))
            initialized = True
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1, expert_model_parallel_size=1,
                expert_tensor_parallel_size=1, context_parallel_size=1,
                create_gloo_process_groups=False)
            torch.manual_seed(20261108)
            tensor_parallel.model_parallel_cuda_manual_seed(20261108)
        args = argparse.Namespace(num_experts=4, transformer_impl='transformer_engine', fp8=None, fp4=None)
        spec = get_spec(args, config)
        report['trainer_fp32_simulator_mode']=dict(actual_allow_hf32=bool(torch.npu.matmul.allow_hf32),
            scope='Actual process property readback; device execution only when execution_mode is npu')
        original = get_gpt_decoder_block_spec(config, use_transformer_engine=True)
        assert len(spec.layer_specs) == 1
        assert spec.layer_specs[0].submodules.self_attention.submodules.core_attention.module is C4V8CoreAttention
        original_core = original.layer_specs[0].submodules.self_attention.submodules.core_attention
        assert getattr(original_core, 'module', original_core) is not C4V8CoreAttention
        report['spec'] = dict(real_pinned_module_spec=True, core_replaced=True, original_spec_unchanged=True,
            layer=str(spec.layer_specs[0].module),
            qkv=str(spec.layer_specs[0].submodules.self_attention.submodules.linear_qkv),
            mlp=str(spec.layer_specs[0].submodules.mlp))
        objects = [TransformerConfig, GPTModel, get_gpt_decoder_block_spec, HybridDeviceOptimizer,
                   megatron_spec, device_attention, device_quant, exact_fp32]
        report['source_sha256'] = {str(Path(inspect.getfile(obj)).resolve()):digest(inspect.getfile(obj)) for obj in objects}
        report['source_sha256'][str(Path(__file__).resolve())] = digest(__file__)
        report['source_sha256'].update(implementation['source_sha256'])
        if not options.execute_npu:
            report.update(status='passed', model_constructed=False, npu_kernels_executed=False,
                          actual_optimizer_steps=0)
            save()
            print('C4V8_PINNED_SPEC_INSPECT_PASS=' + str(report_path), flush=True)
            return

        start = time.monotonic()
        model = GPTModel(config=config, transformer_layer_spec=spec, vocab_size=256,
                         max_sequence_length=options.sequence_length, parallel_output=False,
                         share_embeddings_and_output_weights=False,
                         position_embedding_type='rope', rotary_percent=1.0, rotary_base=1000000)
        model = model.to(device='npu', dtype=torch.bfloat16).train()
        cores = [m for m in model.modules() if isinstance(m, C4V8CoreAttention)]
        assert len(cores) == 1, len(cores)
        calls = []
        hooks = [m.register_forward_hook(lambda module, inp, out: calls.append(dict(
            q=list(inp[0].shape), k=list(inp[1].shape), output=list(out.shape)))) for m in cores]
        ids = torch.arange(options.sequence_length, device='npu', dtype=torch.long)[None] % 256
        positions = torch.arange(options.sequence_length, device='npu', dtype=torch.long)[None]
        logits = model(ids, positions, attention_mask=None)
        assert logits.shape == (1, options.sequence_length, 256), logits.shape
        assert bool(torch.isfinite(logits).all())
        target = ((ids+1)%256).flatten()
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1,256), target)
        assert bool(torch.isfinite(loss))
        (0.5*loss).backward()
        # Exercise the genuine GPT->SelfAttention THD path as well as SBHD.
        # This is two separate causal documents, not one sequence with a reset
        # position index. A single optimizer step consumes both probe losses.
        split = 97 if options.sequence_length>=128 else options.sequence_length//2
        report['sequence_coverage']=dict(sbhd_tokens=options.sequence_length,
            thd_document_tokens=[split,options.sequence_length-split],
            purpose='Exercise128-key tree and separate packed boundaries; not4096 throughput evidence')
        cu = torch.tensor([0, split, options.sequence_length], dtype=torch.int32, device='npu')
        packed = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu,
            max_seqlen_q=max(split,options.sequence_length-split),
            max_seqlen_kv=max(split,options.sequence_length-split))
        assert packed.max_seqlen_q==max(report['sequence_coverage']['thd_document_tokens'])
        assert packed.max_seqlen_kv==packed.max_seqlen_q
        packed_positions = torch.cat([torch.arange(split, device='npu'),
            torch.arange(options.sequence_length-split, device='npu')])[None]
        packed_logits = model(ids, packed_positions, attention_mask=None, packed_seq_params=packed)
        assert packed_logits.shape == logits.shape and bool(torch.isfinite(packed_logits).all())
        packed_loss = torch.nn.functional.cross_entropy(packed_logits.float().reshape(-1,256), target)
        assert bool(torch.isfinite(packed_loss))
        (0.5*packed_loss).backward()
        torch.npu.synchronize()
        parameters = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        active = [(n,p) for n,p in parameters if p.grad is not None]
        assert active and {len(call['q']) for call in calls} == {3,4}, calls
        norms = []
        for name,p in active:
            assert bool(torch.isfinite(p.grad).all()), name
            norms.append(float(p.grad.float().square().sum().cpu()))
        gradnorm = sum(norms)**0.5
        assert gradnorm > 0
        before = {name:p.detach().cpu().clone() for name,p in parameters}
        optimizer = HybridDeviceOptimizer(
            [{'params':[p for _,p in parameters], 'lr':0.002, 'betas':(0.9,0.999),
              'eps':1e-8, 'weight_decay':0.0, 'bias_correction':True, 'fused':True, 'foreach':False}],
            offload_fraction=0.35, cpu_optimizer_cls=CPUAdam, gpu_optimizer_cls=Adam,
            param_update_in_fp32=True, pin_cpu_grads=False, pin_cpu_params=False,
            overlap_cpu_optimizer_d2h_h2d=False)
        optimizer.step()
        torch.npu.synchronize()
        changed = []
        for name,p in parameters:
            assert bool(torch.isfinite(p).all()), name
            count = int((p.detach().cpu()!=before[name]).sum())
            if count: changed.append(dict(name=name, changed_values=count))
        assert changed, 'The actual BF16 model did not change'
        assert any('linear_qkv' in item['name'] for item in changed), 'Attention projection weights must change'
        master = list(optimizer.param_to_fp32_param.values())
        states = [v for state in optimizer.state.values() for k,v in state.items()
                  if k in ('exp_avg','exp_avg_sq') and isinstance(v,torch.Tensor)]
        assert master and states and all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in master+states)
        for path,value in report['source_sha256'].items():
            if digest(path)!=value:raise ValueError('Actual probe source changed during execution: '+path)
        if implementation_profile(workspace,options.core_module)!=implementation:
            raise ValueError('Numerical source or installed toolchain changed during actual probe')
        report.update(status='passed', model_constructed=True, npu_kernels_executed=True,
                      actual_optimizer_steps=1, real_grpo_updates=0, core_forward_calls=calls,
                      loss_sbhd=float(loss.detach().cpu()), loss_thd=float(packed_loss.detach().cpu()),
                      gradient_norm=gradnorm,
                      parameter_count=sum(p.numel() for _,p in parameters), changed_parameters=changed,
                      fp32_master_and_adam_states=True, optimizer_cpu_offload_fraction=0.35,
                      peak_npu_allocated_bytes=torch.npu.max_memory_allocated(),
                      elapsed_seconds=time.monotonic()-start)
        for hook in hooks:hook.remove()
        save()
        print('C4V8_MEGATRON_TINY_ADAPTATION_PASS=' + str(report_path), flush=True)
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        save()
        raise
    finally:
        if initialized:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
