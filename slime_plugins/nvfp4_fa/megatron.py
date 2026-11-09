"""Megatron THD core attention for dense Qwen3 (single-rank TP/PP/CP)."""

import torch

from .attention import attention


class PackedAttention(torch.nn.Module):
    def __init__(self, config, layer_number, attn_mask_type, attention_type, **kwargs):
        super().__init__()
        if (
            attention_type != "self"
            or attn_mask_type.name not in ("causal", "padding_causal")
            or config.attention_dropout
            or config.context_parallel_size != 1
            or config.tensor_model_parallel_size != 1
            or getattr(config, "window_size", None)
        ):
            raise ValueError("NVFP4 FA requires causal, dense self-attention with TP=CP=1 and zero dropout")
        scale = kwargs.get("softmax_scale")
        if scale is not None and scale != config.kv_channels**-0.5:
            raise ValueError("Unexpected Qwen3 attention scaling")
        torch.backends.cuda.matmul.allow_tf32 = False

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        **kwargs
    ):
        p = packed_seq_params
        if attention_mask is not None or attention_bias is not None or p is None or p.qkv_format != "thd":
            raise ValueError("Expected Slime's packed THD self-attention with implicit causal mask")
        cu = getattr(p, "cu_seqlens_q_cpu", None)
        kv = getattr(p, "cu_seqlens_kv_cpu", None)
        if cu is None:
            cu = p.cu_seqlens_q.tolist()
        if kv is None:
            kv = cu if p.cu_seqlens_kv is p.cu_seqlens_q else p.cu_seqlens_kv.tolist()
        if tuple(cu) != tuple(kv) or cu[0] != 0 or cu[-1] != len(query) or len(key) != len(query):
            raise ValueError("Invalid self-attention sequence offsets")
        for padded in (p.cu_seqlens_q_padded, p.cu_seqlens_kv_padded):
            if padded is not None and not torch.equal(padded, p.cu_seqlens_q):
                raise ValueError("Different padded THD offsets are unsupported")
        out = []
        for start, stop in zip(cu[:-1], cu[1:], strict=True):
            if stop <= start:
                raise ValueError("Empty or reversed packed sequence")
            q, k, v = (x[start:stop].transpose(0, 1).unsqueeze(0) for x in (query, key, value))
            out.append(attention(q, k, v).squeeze(0).transpose(0, 1))
        return torch.cat(out, 0).contiguous()


def get_spec(args, config, vp_stage=None):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.spec_utils import ModuleSpec

    if args.num_experts or not args.qk_layernorm or args.pipeline_model_parallel_size != 1:
        raise ValueError("This recipe supports dense Qwen3, PP=1")
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=args.num_experts, qk_layernorm=True)
    spec.submodules.self_attention.submodules.core_attention = ModuleSpec(module=PackedAttention)
    return spec
