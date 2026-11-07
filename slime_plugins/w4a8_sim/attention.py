"""Numerical reference for the four Qwen3-4B W4A8 attention routes.

C16 keeps the native Transformer Engine attention spec. The software routes
use THD-packed causal attention with unpacked BF16 fake-quantized operands and
FP32 reductions. They are A100-compatible numerical references, not native
FP8/FP4 kernels.
"""

import inspect
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from slime_plugins.nvfp4_sim.recipe import round_e2m1, round_e4m3
except ModuleNotFoundError:
    from nvfp4_sim.recipe import round_e2m1, round_e4m3


FP8_MAX = 448.0
FP8_BLOCK_SIZE = 64
NVFP4_BLOCK_SIZE = 16


@dataclass(frozen=True)
class AttentionRoute:
    name: str
    bits: int
    qkv_format: str
    p_format: str
    block_size: int
    backward_rule: str
    qat: bool = False


ROUTES = {
    "c16": AttentionRoute("c16", 16, "bf16", "bf16", 1, "transformer-engine"),
    "naive-c8": AttentionRoute(
        "naive-c8", 8, "fp8-e4m3", "fp8-e4m3", FP8_BLOCK_SIZE, "ordinary-softmax"
    ),
    "naive-c4": AttentionRoute(
        "naive-c4", 4, "nvfp4-e2m1", "nvfp4-e2m1", NVFP4_BLOCK_SIZE, "ordinary-softmax"
    ),
    "attn-qat-c4": AttentionRoute(
        "attn-qat-c4", 4, "nvfp4-e2m1", "nvfp4-e2m1", NVFP4_BLOCK_SIZE, "attn-qat-consistency", True
    ),
}
ROUTE = ROUTES["naive-c8"]


def _validate_floating(tensor, name):
    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"{name} must be BF16, FP16 or FP32")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Nonfinite {name} input")


@torch.no_grad()
def fp8_fake_quant(tensor, block_size=FP8_BLOCK_SIZE):
    """Fake-quantize the last axis to FP8 E4M3 with FP32 block scales."""
    _validate_floating(tensor, "FP8")
    if block_size < 1:
        raise ValueError("FP8 block size must be positive")
    shape = tensor.shape
    columns = shape[-1]
    if columns == 0:
        return tensor.clone()

    x = tensor.float().reshape(-1, columns)
    padded_columns = ((columns + block_size - 1) // block_size) * block_size
    blocks = F.pad(x, (0, padded_columns - columns)).reshape(-1, padded_columns // block_size, block_size)
    scales = blocks.abs().amax(-1) / FP8_MAX
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    scaled = (blocks / scales[..., None]).clamp(-FP8_MAX, FP8_MAX)
    magnitude = scaled.abs()
    exponent = torch.floor(torch.log2(magnitude.clamp_min(2.0**-6))).clamp_min(-6)
    step = torch.exp2(exponent - 3)
    payload = torch.copysign((torch.round(magnitude / step) * step).clamp_max(FP8_MAX), scaled)
    result = (payload * scales[..., None]).reshape(x.shape[0], padded_columns)[:, :columns]
    return result.bfloat16().reshape(shape)


@torch.no_grad()
def nvfp4_fake_quant(tensor, block_size=NVFP4_BLOCK_SIZE):
    """Fake-quantize the last axis with NVFP4 E2M1 and local row scales.

    Each flattened row is independent. For Q/K/V this is one token/head row;
    for P it is one query/head row. Causal future entries are zeroed before
    this function is called, so they do not affect a row's amax.
    """
    _validate_floating(tensor, "NVFP4")
    if block_size != NVFP4_BLOCK_SIZE:
        raise ValueError("NVFP4 block size must be 16")
    shape = tensor.shape
    columns = shape[-1]
    if columns == 0:
        return tensor.clone()

    x = tensor.float().reshape(-1, columns)
    padded_columns = ((columns + block_size - 1) // block_size) * block_size
    blocks = F.pad(x, (0, padded_columns - columns)).reshape(-1, padded_columns // block_size, block_size)
    row_amax = x.abs().amax(-1, keepdim=True)
    global_encode = torch.div(x.new_tensor(2688.0), row_amax)
    global_encode = torch.where(row_amax == 0, torch.ones_like(global_encode), global_encode)
    global_decode = global_encode.reciprocal()
    block_amax = blocks.abs().amax(-1)
    scales = round_e4m3((block_amax * global_encode / 6.0).clamp_max(torch.finfo(torch.float32).max))
    denominator = scales * global_decode
    encode = torch.where(denominator == 0, torch.zeros_like(denominator), denominator.reciprocal())
    payload = round_e2m1(blocks * encode[..., None])
    decoded = payload * scales[..., None] * global_decode[..., None]
    return decoded.reshape(x.shape[0], padded_columns)[:, :columns].bfloat16().reshape(shape)


def fake_quant(tensor, bits=8):
    """Return an unpacked BF16 fake-quantized tensor for C4, C8 or C16."""
    if bits == 16:
        return tensor.contiguous()
    if bits == 8:
        return fp8_fake_quant(tensor)
    if bits == 4:
        return nvfp4_fake_quant(tensor)
    raise ValueError(f"Unsupported attention precision: C{bits}")


def quantize_qkv(query, key, value, bits=8):
    """Quantize Q/K/V independently along each token/head head_dim row."""
    return tuple(fake_quant(tensor, bits) for tensor in (query, key, value))


def _sequence_ranges(cu_seqlens, total_tokens):
    if cu_seqlens is None or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("Expected cumulative sequence lengths for THD attention")
    if (
        cu_seqlens.device != total_tokens.device
        or cu_seqlens.dtype != torch.int32
        or not cu_seqlens.is_contiguous()
    ):
        raise ValueError("Sequence offsets must be contiguous int32 tensors on the attention device")
    offsets = cu_seqlens.detach().cpu().tolist()
    if offsets[0] != 0 or offsets[-1] != total_tokens.shape[0] or any(
        right < left for left, right in zip(offsets, offsets[1:])
    ):
        raise ValueError("Invalid packed sequence offsets")
    return tuple((int(left), int(right)) for left, right in zip(offsets, offsets[1:]))


def _validate_inputs(query, key, value, packed_seq_params):
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Low-precision attention requires THD Q/K/V tensors")
    if key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError("Incompatible THD Q/K/V shapes")
    if query.shape[1] == 0 or key.shape[1] == 0 or query.shape[1] % key.shape[1]:
        raise ValueError("The number of query heads must be divisible by KV heads")
    if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
        raise ValueError("Low-precision attention requires THD packed sequences")
    cu_q = packed_seq_params.cu_seqlens_q
    cu_kv = packed_seq_params.cu_seqlens_kv
    if cu_q is None or cu_kv is None or not torch.equal(cu_q, cu_kv):
        raise ValueError("Self-attention requires shared Q/KV sequence offsets")
    if packed_seq_params.cu_seqlens_q_padded is not None or packed_seq_params.cu_seqlens_kv_padded is not None:
        raise ValueError("Padded packed sequence offsets are not supported")
    if packed_seq_params.cp_group is not None or packed_seq_params.local_cp_size not in (None, 1):
        raise ValueError("Context parallelism is not supported by the low-precision reference")
    for tensor, name in ((query, "query"), (key, "key"), (value, "value")):
        _validate_floating(tensor, name)
    return _sequence_ranges(cu_q, query)


def _attention_forward(query, key, value, cu_seqlens, scale, bits=8, qat=False):
    qf, kf, vf = (tensor.float() for tensor in quantize_qkv(query, key, value, bits))
    outputs, records = [], []
    for start, end in _sequence_ranges(cu_seqlens, query):
        q, k, v = qf[start:end], kf[start:end], vf[start:end]
        if k.shape[1] < 1 or q.shape[1] < k.shape[1] or q.shape[1] % k.shape[1]:
            raise ValueError("The number of query heads must be divisible by KV heads")
        groups = q.shape[1] // k.shape[1]
        k_for_q = k.repeat_interleave(groups, dim=1)
        v_for_q = v.repeat_interleave(groups, dim=1)
        length = end - start
        causal = torch.ones((length, length), dtype=torch.bool, device=query.device).tril()
        scores = torch.einsum("lhd,mhd->hlm", q, k_for_q) * scale
        probabilities = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        probabilities = probabilities.masked_fill(~causal, 0.0)
        probability_fake = fake_quant(probabilities, bits).float().masked_fill(~causal, 0.0)
        output = torch.einsum("hlm,mhd->lhd", probability_fake, v_for_q)
        auxiliary = torch.einsum("hlm,mhd->lhd", probabilities, v_for_q) if qat else None
        outputs.append(output.to(query.dtype))
        records.append((q, k, v, probabilities, probability_fake, output.float(), auxiliary))
    if not outputs:
        return query.new_empty(query.shape), records
    return torch.cat(outputs, dim=0), records


class _SimulatedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, cu_seqlens, softmax_scale, bits, qat):
        output, records = _attention_forward(query, key, value, cu_seqlens, softmax_scale, bits, qat)
        ctx.records = records
        ctx.cu_seqlens = cu_seqlens
        ctx.softmax_scale = softmax_scale
        ctx.qat = qat
        ctx.query_shape, ctx.key_shape, ctx.value_shape = query.shape, key.shape, value.shape
        ctx.query_dtype, ctx.key_dtype, ctx.value_dtype = query.dtype, key.dtype, value.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        dq = torch.zeros(ctx.query_shape, device=grad_output.device, dtype=ctx.query_dtype)
        dk = torch.zeros(ctx.key_shape, device=grad_output.device, dtype=ctx.key_dtype)
        dv = torch.zeros(ctx.value_shape, device=grad_output.device, dtype=ctx.value_dtype)
        for (start, end), (q, k, v, probabilities, probability_fake, output, auxiliary) in zip(
            _sequence_ranges(ctx.cu_seqlens, grad_output), ctx.records, strict=True
        ):
            grad = grad_output[start:end].float()
            groups = q.shape[1] // k.shape[1]
            k_for_q = k.repeat_interleave(groups, dim=1)
            v_for_q = v.repeat_interleave(groups, dim=1)
            delta_source = auxiliary if ctx.qat else output
            delta = (grad * delta_source).sum(-1).transpose(0, 1)
            d_probability = torch.einsum("lhd,mhd->hlm", grad, v_for_q)
            d_score = probabilities * (d_probability - delta[..., None])
            d_query = torch.einsum("hlm,mhd->lhd", d_score, k_for_q) * ctx.softmax_scale
            d_key_per_q_head = torch.einsum("hlm,lhd->hmd", d_score, q) * ctx.softmax_scale
            d_value_per_q_head = torch.einsum("hlm,lhd->mhd", probability_fake if ctx.qat else probabilities, grad)
            length = end - start
            d_key = d_key_per_q_head.reshape(k.shape[1], groups, length, k.shape[2]).sum(1)
            d_value = d_value_per_q_head.reshape(v.shape[1], groups, length, v.shape[2]).sum(1)
            dq[start:end] = d_query.to(ctx.query_dtype)
            dk[start:end] = d_key.transpose(0, 1).to(ctx.key_dtype)
            dv[start:end] = d_value.transpose(0, 1).to(ctx.value_dtype)
        return dq, dk, dv, None, None, None, None


class SimulatedAttention(torch.nn.Module):
    """THD-packed causal self-attention for one software precision route."""

    def __init__(
        self,
        config,
        layer_number,
        attn_mask_type,
        attention_type,
        cp_comm_type="p2p",
        softmax_scale=None,
        pg_collection=None,
        route="naive-c8",
    ):
        super().__init__()
        if pg_collection is None or getattr(pg_collection, "tp", None) is None:
            raise ValueError("An explicit ProcessGroupCollection with TP is required")
        tp_group = pg_collection.tp
        cp_group = getattr(pg_collection, "cp", None)
        if tp_group.size() != 1 or (cp_group is not None and cp_group.size() != 1):
            raise ValueError("Reference attention only supports single-rank TP/CP")
        if route not in ROUTES or route == "c16":
            raise ValueError(f"Unsupported simulated attention route: {route}")
        if config.tensor_model_parallel_size != 1 or config.context_parallel_size != 1:
            raise ValueError("Reference attention only supports TP=CP=1")
        if attention_type != "self" or getattr(attn_mask_type, "name", None) not in ("causal", "padding_causal"):
            raise ValueError("Reference attention only supports causal self-attention")
        if getattr(config, "attention_dropout", 0) != 0 or getattr(config, "window_size", None) is not None:
            raise ValueError("Dropout and windowed attention are not supported")
        if getattr(config, "softmax_type", "vanilla") != "vanilla" or getattr(
            config, "apply_query_key_layer_scaling", False
        ):
            raise ValueError("Only vanilla softmax without layer-dependent scaling is supported")
        if getattr(config, "qk_clip", False) or getattr(config, "log_max_attention_logit", False):
            raise ValueError("QK clipping and max-logit logging are not supported")
        self.route = ROUTES[route]
        self.softmax_scale = softmax_scale if softmax_scale is not None else config.kv_channels**-0.5
        self.layer_number = layer_number

    def forward(self, query, key, value, attention_mask, attn_mask_type=None, attention_bias=None, packed_seq_params=None):
        if attention_mask is not None or attention_bias is not None:
            raise ValueError("Reference attention only supports implicit packed causal masks")
        _validate_inputs(query, key, value, packed_seq_params)
        return _SimulatedAttention.apply(
            query,
            key,
            value,
            packed_seq_params.cu_seqlens_q,
            self.softmax_scale,
            self.route.bits,
            self.route.qat,
        )


def _base_spec_kwargs(args):
    return dict(
        num_experts=getattr(args, "num_experts", 0),
        moe_grouped_gemm=getattr(args, "moe_grouped_gemm", False),
        qk_layernorm=getattr(args, "qk_layernorm", False),
        multi_latent_attention=getattr(args, "multi_latent_attention", False),
    )


def _te_spec(args):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

    provider = get_gpt_layer_with_transformer_engine_spec
    kwargs = _base_spec_kwargs(args)
    if "moe_use_legacy_grouped_gemm" in inspect.signature(provider).parameters:
        kwargs["moe_use_legacy_grouped_gemm"] = getattr(args, "moe_use_legacy_grouped_gemm", False)
    return provider(**kwargs)


def _build_spec(args, route):
    from megatron.core.transformer.spec_utils import ModuleSpec

    if getattr(args, "num_experts", 0) or getattr(args, "multi_latent_attention", False) or not getattr(
        args, "qk_layernorm", False
    ):
        raise ValueError("The simulated attention spec requires dense Qwen3 Q/K-norm attention")
    spec = _te_spec(args)
    spec.submodules.self_attention.submodules.core_attention = ModuleSpec(
        module=SimulatedAttention, params={"route": route}
    )
    return spec


def get_c16_reference_spec(args, config, vp_stage):
    """Return the unchanged Transformer Engine C16 control spec."""
    return _te_spec(args)


def get_naive_c8_spec(args, config, vp_stage):
    return _build_spec(args, "naive-c8")


def get_naive_c4_spec(args, config, vp_stage):
    return _build_spec(args, "naive-c4")


def get_attn_qat_c4_spec(args, config, vp_stage):
    return _build_spec(args, "attn-qat-c4")


__all__ = [
    "ROUTE",
    "ROUTES",
    "SimulatedAttention",
    "_SimulatedAttention",
    "_attention_forward",
    "fake_quant",
    "fp8_fake_quant",
    "get_attn_qat_c4_spec",
    "get_c16_reference_spec",
    "get_naive_c4_spec",
    "get_naive_c8_spec",
    "nvfp4_fake_quant",
    "quantize_qkv",
]
