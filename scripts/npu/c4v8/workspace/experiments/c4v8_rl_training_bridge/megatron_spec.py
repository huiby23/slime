"""Opt-in Megatron core-attention replacement; no global framework patches.

`get_spec` matches the pinned slime callable-spec interface (args, config,
vp_stage). It uses the normal MoE block spec and replaces only core_attention.
Production launch remains gated by explicit kernel-module and JSON policy
configuration; creating this file does not enable a model or change precision.
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import os
import hashlib
from pathlib import Path

import torch
from layouts import attention_sbhd, attention_thd


def configure_npu_matmul():
    """FP32 reference GEMMs must not silently use HF32 on the real trainer."""
    if not hasattr(torch, 'npu'):
        return {'available': False, 'allow_hf32': None, 'reason': 'CPU torch has no NPU namespace'}
    control = getattr(torch.npu, 'matmul', None)
    if control is None or not hasattr(control, 'allow_hf32'):
        raise RuntimeError('Cannot explicitly control NPU FP32 matmul semantics')
    before = bool(control.allow_hf32)
    control.allow_hf32 = False
    if bool(control.allow_hf32):raise RuntimeError('Failed to disable NPU matmul HF32')
    return {'available': True, 'allow_hf32_before': before, 'allow_hf32': False,
            'scope': 'FP32 simulated FA GEMMs; BF16 backbone dtype unchanged'}


def record_trainer_policy(policy, matmul, implementation=None):
    directory = os.environ.get('HBY_C4V8_TRAIN_AUDIT_DIR')
    if not directory:
        if matmul['available']:raise RuntimeError('Real trainer requires an explicit C4V8 audit directory')
        return
    directory = Path(directory);directory.mkdir(parents=True,exist_ok=True)
    policy_sha = hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
    if os.environ.get('HBY_C4V8_POLICY_SHA256') != policy_sha:
        raise ValueError('Trainer policy differs from the declared shared policy SHA')
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
    path = directory/f'trainer-policy-pid{os.getpid()}.json'
    value = dict(pid=os.getpid(),rank=rank,policy=policy,policy_sha256=policy_sha,
        fp32_matmul=matmul,quantized_backward=False,
        core_module=os.environ.get('HBY_C4V8_CORE_MODULE'),implementation=implementation,
        implementation_sha256=os.environ.get('HBY_C4V8_IMPLEMENTATION_SHA256'),
        spec_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if path.exists():
        previous=json.loads(path.read_text())
        if any(previous.get(k)!=value[k] for k in ('policy_sha256','spec_source_sha256','rank','core_module','implementation_sha256')):
            raise ValueError('Trainer precision identity changed within a process')
    else:
        with path.open('x') as file:json.dump(value,file,indent=2)


def _mask_name(value):
    return getattr(value, 'name', str(value)).lower().split('.')[-1]


class C4V8CoreAttention(torch.nn.Module):
    def __init__(self, config, layer_number, attn_mask_type, attention_type='self',
                 cp_comm_type=None, softmax_scale=None, pg_collection=None, *,
                 kernel, precision_config):
        super().__init__()
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.kernel = kernel
        self.precision_config = precision_config
        if getattr(config, 'context_parallel_size', 1) != 1:
            raise ValueError('Experimental C4V8 adapter is validated only for CP1')
        if getattr(config, 'attention_dropout', 0.0) != 0.0:
            raise ValueError('Replay assumes deterministic attention without dropout')
        if attention_type != 'self' or not precision_config.causal:
            raise ValueError('This training adapter only implements causal self attention')
        if getattr(config, 'window_size', None) is not None:
            raise ValueError('Sliding-window attention is not implemented')
        if getattr(config, 'apply_query_key_layer_scaling', False):
            raise ValueError('Layer-dependent QK scaling is not implemented')
        d = getattr(config, 'kv_channels', None)
        if softmax_scale is not None and (d is None or not math.isclose(softmax_scale, 1/math.sqrt(d))):
            raise ValueError('Only the standard 1/sqrt(head_dim) softmax scale is implemented')
        if _mask_name(attn_mask_type) != 'causal':
            raise ValueError('Only the causal Megatron mask enum is accepted')

    def forward(self, query, key, value, attention_mask=None, *, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        if query.device.type in ('npu','privateuseone') and bool(torch.npu.matmul.allow_hf32):
            raise RuntimeError('C4V8 FP32 simulation requires matmul.allow_hf32=False')
        mask_type = self.attn_mask_type if attn_mask_type is None else attn_mask_type
        if _mask_name(mask_type) != 'causal' or attention_bias is not None:
            raise ValueError('Custom masks/bias are outside the verified C4V8 adapter')
        if attention_mask is not None:
            # For the pinned packed path this should be None. A supplied mask
            # must be exactly the ordinary boolean causal mask; never ignore
            # padding, prefix, document or arbitrary additive mask information.
            if packed_seq_params is not None or query.ndim != 4:
                raise ValueError('Packed attention requires its explicit lengths and no separate mask')
            n = query.shape[0]
            if (attention_mask.dtype != torch.bool or attention_mask.ndim != 4
                    or attention_mask.shape[-2:] != (n, n)
                    or attention_mask.shape[0] not in (1, query.shape[1])
                    or attention_mask.shape[1] != 1):
                raise ValueError('Unsupported explicit attention mask shape/type')
            expected = torch.ones((n, n), dtype=torch.bool, device=attention_mask.device).triu(1)
            if not bool((attention_mask == expected).all()):
                raise ValueError('Explicit attention mask is not the exact ordinary causal mask')
        if packed_seq_params is None:
            return attention_sbhd(query, key, value, kernel=self.kernel, config=self.precision_config)
        p = packed_seq_params
        if p.qkv_format != 'thd' or getattr(p, 'local_cp_size', None) not in (None, 1):
            raise ValueError('Only CP1 THD packed metadata is implemented')
        return attention_thd(
            query, key, value, cu_seqlens_q=p.cu_seqlens_q,
            cu_seqlens_kv=p.cu_seqlens_kv,
            cu_seqlens_q_padded=p.cu_seqlens_q_padded,
            cu_seqlens_kv_padded=p.cu_seqlens_kv_padded,
            kernel=self.kernel, config=self.precision_config)


def replace_core_attention(block_spec, *, kernel, precision_config, module_spec_type):
    """Deep copy before replacing; preserve every BF16 projection/MoE/norm spec."""
    result = copy.deepcopy(block_spec)
    if not getattr(result, 'layer_specs', None):
        raise ValueError('Expected a nonempty Megatron TransformerBlockSubmodules spec')
    for layer in result.layer_specs:
        attention = layer.submodules.self_attention
        if not hasattr(attention.submodules, 'core_attention'):
            raise ValueError('Unexpected Megatron attention module spec')
        attention.submodules.core_attention = module_spec_type(
            module=C4V8CoreAttention,
            params={'kernel': kernel, 'precision_config': precision_config})
    return result


def get_spec(args, config, vp_stage=None):
    """Use via slime --spec megatron_spec get_spec with explicit environment.

    HBY_C4V8_CORE_MODULE selects an importable module with AttentionConfig and
    attention. HBY_C4V8_POLICY_FILE is a JSON object of AttentionConfig kwargs.
    Launcher must separately verify kernel SHA, numerical agreement and device
    resource gates; this helper does not start work or assert those passed.
    """
    if not getattr(args, 'num_experts', None):
        raise ValueError('This provider was prepared for a MoE decoder block')
    if getattr(args, 'fp8', None) or getattr(args, 'fp4', None):
        raise ValueError('Global backbone quantization must remain disabled')
    module_name = os.environ.get('HBY_C4V8_CORE_MODULE')
    policy_path = os.environ.get('HBY_C4V8_POLICY_FILE')
    if not module_name or not policy_path:
        raise RuntimeError('Explicit C4V8 kernel module and policy file are required')
    core = importlib.import_module(module_name)
    policy = json.loads(Path(policy_path).read_text())
    if not isinstance(policy, dict):
        raise TypeError('C4V8 policy must be a JSON object')
    implementation=None
    if module_name=='fast_attention' or os.environ.get('HBY_C4V8_IMPLEMENTATION_ID'):
        from numerical_profile import implementation_profile, digest
        implementation=implementation_profile(Path(__file__).resolve().parents[2],module_name)
        if (implementation['implementation_id']!=os.environ.get('HBY_C4V8_IMPLEMENTATION_ID')
                or digest(implementation)!=os.environ.get('HBY_C4V8_IMPLEMENTATION_SHA256')):
            raise ValueError('Actual trainer implementation differs from its declared source/toolchain identity')
    record_trainer_policy(policy, configure_npu_matmul(),implementation)
    precision_config = core.AttentionConfig(**policy)
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.transformer.spec_utils import ModuleSpec
    kwargs = {'use_transformer_engine': args.transformer_impl == 'transformer_engine'}
    if vp_stage is not None:
        kwargs['vp_stage'] = vp_stage
    original = get_gpt_decoder_block_spec(config, **kwargs)
    return replace_core_attention(original, kernel=core.attention,
                                  precision_config=precision_config, module_spec_type=ModuleSpec)
