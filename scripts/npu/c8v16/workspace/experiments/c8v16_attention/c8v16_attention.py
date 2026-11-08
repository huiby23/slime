"""Causal W16A16C8V16 attention for Megatron and SGLang.

Q and K use MXFP8 blocks along head features. Unnormalized probabilities and
V use MXFP8 blocks along logical keys. In a query's last 32-key block, a V
scale sees only keys visible to that query. This gives training and decoding
the same causal scaling rule for BF16 KV storage.

Vector operations are evaluated in FP32 and rounded to BF16 at the documented
boundaries. Both matrix multiplications accumulate in FP32. QDQ and BF16
rounding use identity STE. The training adapter supplies query-tile replay.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from device_quant import mxfp8_values


MX_WIDTH = 32


@dataclass(frozen=True)
class AttentionConfig:
    query_block: int = 128
    key_block: int = 128
    vector: str = 'bf16'
    quantize_operands: bool = True
    causal: bool = True
    reduction: str = 'mixed_fp32_reduction'

    def __post_init__(self):
        if type(self.query_block) is not int or self.query_block < 1:
            raise ValueError('query_block must be a positive integer')
        if type(self.key_block) is not int or self.key_block < MX_WIDTH or self.key_block % MX_WIDTH:
            raise ValueError('key_block must be a positive multiple of the 32-key MXFP8 group')
        if (self.causal is not True or self.quantize_operands is not True
                or self.vector != 'bf16' or self.reduction != 'mixed_fp32_reduction'):
            raise ValueError('Expected causal C8 MXFP8 operands and the V16 BF16 reduction recipe')


class _RoundBF16(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.bfloat16().float()

    @staticmethod
    def backward(ctx, grad):
        return grad


def _npu_mx(x):
    from fused_mxfp8 import qdq_kernel

    # Zero pad each logical row before packing many independent 32-value groups
    # into one Triton program. This handles long V columns during decode while
    # bounding each program's working set.
    width = x.size(-1)
    padded = F.pad(x, (0, (-width) % MX_WIDTH)).contiguous()
    flat = padded.reshape(-1)
    launch_width = min(1024, max(MX_WIDTH, 1 << (flat.numel() - 1).bit_length()))
    packed = F.pad(flat, (0, (-flat.numel()) % launch_width))
    rows = packed.numel() // launch_width
    values = torch.empty_like(packed)
    codes = torch.empty_like(packed, dtype=torch.uint8)
    scales = torch.empty(packed.numel() // MX_WIDTH, dtype=torch.uint8, device=x.device)
    flags = torch.empty(rows, dtype=torch.int32, device=x.device)
    qdq_kernel[(rows,)](packed, values, codes, scales, flags, launch_width, launch_width, False,
                        num_warps=4, enable_fp_fusion=False)
    if bool((flags != 0).any()):
        raise FloatingPointError('MXFP8 quantization flagged a nonfinite input or overflow')
    return values[:flat.numel()].reshape(padded.shape)[..., :width]


class _RoundMX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.device.type == 'cpu':
            return mxfp8_values(x).values
        return _npu_mx(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


bf = _RoundBF16.apply
mx = _RoundMX.apply


def _validate(q, k, v, config, offset):
    if not isinstance(config, AttentionConfig):
        raise TypeError('Expected the C8V16 AttentionConfig')
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError('Expected BHSD inputs')
    if q.dtype not in (torch.bfloat16, torch.float32) or any(x.dtype != q.dtype for x in (k, v)):
        raise TypeError('Expected matching BF16 inputs or their FP32 replay values')
    if q.device.type not in ('cpu', 'npu', 'privateuseone') or any(x.device != q.device for x in (k, v)):
        raise ValueError('Expected Q/K/V on the same CPU or Ascend device')
    if q.device.type != 'cpu' and torch.npu.matmul.allow_hf32:
        raise ValueError('Disable NPU HF32 for the FP32 accumulation reference')
    if (min(q.shape) < 1 or min(k.shape) < 1 or k.shape != v.shape or q.size(0) != k.size(0)
            or q.size(-1) != k.size(-1) or q.size(1) % k.size(1)):
        raise ValueError('Invalid Q/K/V or GQA head shapes')
    if type(offset) is not int or offset < 0 or offset + q.size(2) > k.size(2):
        raise ValueError('Invalid absolute query_offset')


def _pv(prob, full_v, raw_v, query_start, key_start, group_size, qdq):
    """Multiply the quantized operands, recomputing the last visible V group per row."""
    rows, width = prob.shape[-2:]
    key_end = key_start + width
    if query_start >= key_end:
        # Every query sees every key in this tile, including all complete groups.
        return prob @ full_v

    # A diagonal query can share all groups strictly before its own. Query rows
    # below this tile share the entire tile; rows above it have zero probability.
    qidx = torch.arange(query_start - key_start, query_start - key_start + rows, device=prob.device)
    kidx = torch.arange(width, device=prob.device)
    group = qidx // group_size
    shared = (kidx[None, :] // group_size < group[:, None]) | (qidx[:, None] >= width)
    result = prob.masked_fill(~shared, 0) @ full_v

    first = max(0, key_start - query_start)
    last = min(rows, key_end - query_start)
    if last <= first:
        return result
    local = qidx[first:last]
    diagonal_groups = local // group_size
    pad = (-width) % group_size
    b, h, _, d = raw_v.shape
    v_blocks = F.pad(raw_v, (0, 0, 0, pad)).reshape(b, h, -1, group_size, d)
    v_tail = v_blocks.index_select(2, diagonal_groups)
    visible = torch.arange(group_size, device=prob.device) <= (local % group_size)[:, None]
    # [B,H,Q,D,32] makes each scale local to one query and one V feature.
    v_tail = qdq(v_tail.transpose(-1, -2).masked_fill(~visible[:, None, :], 0.0))
    p_index = diagonal_groups[:, None] * group_size + torch.arange(group_size, device=prob.device)
    p_index = p_index[None, None].expand(b, h, -1, -1)
    p_tail = F.pad(prob[:, :, first:last], (0, pad)).gather(-1, p_index)
    tail = (p_tail.unsqueeze(-2) @ v_tail.transpose(-1, -2)).squeeze(-2)
    return result + F.pad(tail, (0, 0, first, rows - last))


def attention(q, k, v, config=AttentionConfig(), *, query_offset=0):
    """Apply causal C8V16 to [batch, heads, tokens, features], returning BF16."""
    _validate(q, k, v, config, query_offset)
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _attention(q, k, v, config, query_offset)


def _attention(q, k, v, config, offset, *, qdq=mx, group_size=MX_WIDTH):
    end = offset + q.size(2)
    q_quant = qdq(bf(q.float()))
    k_quant = qdq(bf(k[:, :, :end].float()))
    v16 = bf(v[:, :, :end].float())
    # All complete groups can be used directly by every later query.
    v_quant = qdq(v16.transpose(-1, -2)).transpose(-1, -2)
    repeat = q.size(1) // k.size(1)
    if repeat > 1:
        k_quant, v_quant, v16 = (x.repeat_interleave(repeat, dim=1) for x in (k_quant, v_quant, v16))

    chunks = []
    for qs in range(0, q.size(2), config.query_block):
        qe = min(q.size(2), qs + config.query_block)
        query = q_quant[:, :, qs:qe]
        qpos = torch.arange(offset + qs, offset + qe, device=q.device)
        maximum = query.new_full((*query.shape[:-1], 1), -math.inf)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(query)
        for ks in range(0, offset + qe, config.key_block):
            ke = min(ks + config.key_block, offset + qe)
            scores = bf((query @ k_quant[:, :, ks:ke].transpose(-1, -2)) * (q.size(-1) ** -0.5))
            masked = torch.arange(ks, ke, device=q.device)[None, :] > qpos[:, None]
            scores = scores.masked_fill(masked, -math.inf)
            new_max = bf(torch.maximum(maximum, scores.amax(dim=-1, keepdim=True)))
            alpha = bf(bf(maximum - new_max).exp())
            prob = bf(bf(scores - new_max).exp())
            # The denominator uses BF16 probabilities. The Cube receives the QDQ values.
            tile_sum = bf(prob.sum(dim=-1, keepdim=True))
            denominator = bf(bf(denominator * alpha) + tile_sum)
            contribution = _pv(qdq(prob), v_quant[:, :, ks:ke], v16[:, :, ks:ke],
                               offset + qs, ks, group_size, qdq)
            numerator = bf(bf(numerator * alpha) + bf(contribution))
            maximum = new_max
        chunks.append((numerator / denominator).bfloat16())
    out = torch.cat(chunks, dim=2)
    if not bool(torch.isfinite(out).all()):
        raise FloatingPointError('Nonfinite C8V16 attention output')
    return out
