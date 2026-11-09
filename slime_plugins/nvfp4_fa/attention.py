"""Causal W16A16C4V16 attention shared by Megatron and SGLang.

Q/K/P use one upper scale per head-row (P per online-softmax key tile).
V uses one upper scale per head and aligned 16-token by head_dim slab.
Each actual E4M3 scale covers 16 elements along the contraction dimension.
"""

import torch
import torch.nn.functional as F

from .pv import pv
from slime_plugins.nvfp4_sim.recipe import round_e2m1, round_e4m3

BLOCK = 16
TILE = 128


class _BF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.bfloat16().float()

    @staticmethod
    def backward(ctx, grad):
        return grad


class _NVFP4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, blocks):
        if blocks.is_cuda:
            from .quant_cuda import qdq

            return qdq(blocks)
        # Input is already a BF16 boundary value, represented in FP32.
        peaks = blocks.abs().amax(-1, keepdim=True)
        upper_peak = peaks.amax(-2, keepdim=True)
        upper = torch.where(upper_peak == 0, 1.0, upper_peak / (448 * 6))
        scales = round_e4m3(peaks / (6 * upper))
        scale = scales * upper
        codes = round_e2m1(blocks / torch.where(scale == 0, 1.0, scale))
        return codes * scale

    @staticmethod
    def backward(ctx, grad):
        return grad


bf = _BF.apply


def rows(x):
    width = x.size(-1)
    groups = F.pad(x, (0, -width % BLOCK)).unflatten(-1, (-1, BLOCK))
    return _NVFP4.apply(groups).flatten(-2)[..., :width]


def values(slabs):
    # [..., aligned 16 tokens, features] -> [..., features, 1 x 16].
    blocks = slabs.transpose(-1, -2).contiguous()
    return _NVFP4.apply(blocks).transpose(-1, -2)


def causal_pv(p, v, slabs, query_start, key_start):
    # A query inside this tile must quantize its last V group using only
    # its visible values. Softmax masking alone cannot hide a future scale.
    groups = p.unflatten(-1, (-1, BLOCK))
    parts = pv(groups, slabs)
    first = max(0, key_start - query_start)
    stop = min(p.size(-2), key_start + p.size(-1) - query_start)
    if first < stop:
        pos = torch.arange(first, stop, device=p.device) + query_start - key_start
        indices = pos // BLOCK
        tail = v.index_select(-3, indices)
        visible = torch.arange(BLOCK, device=p.device) <= (pos % BLOCK)[:, None]
        tail = values(tail.masked_fill(~visible[..., None], 0))
        tail = tail.repeat_interleave(p.size(1) // tail.size(1), 1)
        idx = indices.view(1, 1, -1, 1, 1)
        prob = groups[:, :, first:stop].gather(-2, idx.expand(*groups[:, :, first:stop].shape[:3], 1, BLOCK))
        contribution = pv(prob.transpose(2, 3), tail).transpose(2, 3)
        corrected = parts[:, :, first:stop].scatter(-2, idx.expand_as(contribution), contribution)
        parts = torch.cat((parts[:, :, :first], corrected, parts[:, :, stop:]), dim=2)
    return parts.sum(-2)


def _operands(q, k, v):
    repeat = q.size(1) // k.size(1)
    k = rows(k.float()).repeat_interleave(repeat, 1)
    raw = F.pad(v.float(), (0, 0, 0, -v.size(2) % TILE)).unflatten(2, (-1, BLOCK))
    quant = values(raw)
    return rows(q.float()), k, raw, quant.repeat_interleave(repeat, 1)


def _validate(q, k, v):
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.size(0) != k.size(0):
        raise ValueError("Expected BHSD self-attention with matching KV")
    if min(*q.shape, *k.shape) < 1 or q.size(1) % k.size(1) or q.size(-1) != k.size(-1):
        raise ValueError("Incompatible GQA shape")
    if any(x.device != q.device or x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise ValueError("Attention requires BF16 Q/K/V on one device")
    if q.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Disable TF32 for C4V16 FP32 accumulation")


def attention(q, k, v, *, offset=0):
    """Training or cached prefill. Offsets index the request's logical sequence."""
    _validate(q, k, v)
    if offset < 0 or offset + q.size(2) != k.size(2):
        raise ValueError("Q must be the suffix of the supplied KV history")
    with torch.autocast(q.device.type, enabled=False):
        q, k, raw, quant = _operands(q, k, v)
        chunks = []
        for qs in range(0, q.size(2), TILE):
            query = q[:, :, qs : qs + TILE]
            start = offset + qs
            stop = start + query.size(2)
            positions = torch.arange(start, stop, device=q.device)
            maximum = torch.full_like(query[..., :1], -torch.inf)
            denom = torch.zeros_like(maximum)
            num = torch.zeros_like(query)
            for ks in range(0, stop, TILE):
                ke = min(ks + TILE, stop)
                score = bf((query @ k[:, :, ks:ke].transpose(-1, -2)) * (q.size(-1) ** -0.5))
                score = score.masked_fill(torch.arange(ks, ke, device=q.device) > positions[:, None], -torch.inf)
                highest = torch.maximum(maximum, score.amax(-1, keepdim=True))
                alpha = bf(bf(maximum - highest).exp())
                prob = bf(bf(score - highest).exp())
                prob = F.pad(prob, (0, TILE - (ke - ks)))
                denom = bf(bf(denom * alpha) + bf(prob.sum(-1, keepdim=True)))
                p4 = rows(prob)
                g = slice(ks // BLOCK, (ks + TILE) // BLOCK)
                update = causal_pv(p4, raw[:, :, g], quant[:, :, g], start, ks)
                num = bf(bf(num * alpha) + bf(update))
                maximum = highest
            chunks.append((num / denom).bfloat16())
        return torch.cat(chunks, dim=2)


@torch.no_grad()
def decode(q, k, v, lengths):
    """Heterogeneous final queries, using zero padding only inside this call."""
    _validate(q, k, v)
    if q.size(2) != 1 or lengths.shape != (q.size(0),):
        raise ValueError("One decode query and length per request required")
    with torch.autocast(q.device.type, enabled=False):
        valid = torch.arange(k.size(2), device=q.device) < lengths[:, None]
        padding = ~valid[:, None, :, None]
        q, k, _, v4 = _operands(q, k.masked_fill(padding, 0), v.masked_fill(padding, 0))
        score = bf((q @ k.transpose(-1, -2)) * (q.size(-1) ** -0.5)).squeeze(2)
        score = score.masked_fill(~valid[:, None, :], -torch.inf)
        tiles = F.pad(score, (0, -score.size(-1) % TILE), value=-torch.inf).unflatten(-1, (-1, TILE))
        highest = tiles.amax(-1).cummax(-1).values
        old = F.pad(highest[..., :-1], (1, 0), value=-torch.inf)
        alpha = bf(bf(old - highest).exp())
        prob = bf(bf(tiles - highest.unsqueeze(-1)).exp())
        sums = bf(prob.sum(-1))
        p4 = rows(prob).reshape(*prob.shape[:2], 1, -1, BLOCK)
        parts = pv(p4, v4).squeeze(2)
        contrib = bf(parts.unflatten(2, (-1, TILE // BLOCK)).sum(-2))
        den = torch.zeros_like(sums[..., :1])
        num = torch.zeros_like(q.squeeze(2))
        for t in range(sums.size(-1)):
            a = alpha[..., t : t + 1]
            den = bf(bf(den * a) + sums[..., t : t + 1])
            num = bf(bf(num * a) + contrib[:, :, t])
        return (num / den).bfloat16().unsqueeze(2)
