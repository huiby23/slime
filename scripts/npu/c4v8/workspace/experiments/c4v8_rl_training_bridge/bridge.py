"""Query-tile recomputation for an explicitly supplied QDQ/STE attention core.

This reduces saved attention graph memory; it is not a fused FlashAttention
kernel or low-precision backward. The core must scale rows independently and
have identical FP32 internal arithmetic for FP32 and BF16 input values, apart
from the final output cast. It must be deterministic and must not use dropout,
stateful amax, or RNG. Its interface is kernel(q,k,v,config,query_offset=offset),
with [batch,head,token,feature] inputs and GQA handled by the core.
"""
from __future__ import annotations

from typing import Callable, Any

import torch
from torch.autograd.function import once_differentiable


def _validate(q, k, v, config, query_offset):
    if any(x.ndim != 4 or x.dtype not in (torch.float32, torch.bfloat16, torch.float16)
           for x in (q, k, v)):
        raise ValueError('Expected FP32/BF16/FP16 BHND inputs')
    if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError('Q/K/V device and storage dtype must agree')
    if (k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]
            or min(q.shape) < 1 or min(k.shape) < 1 or q.shape[1] % k.shape[1]):
        raise ValueError('Invalid Q/K/V or GQA shapes')
    if not isinstance(query_offset, int) or query_offset < 0 or query_offset + q.shape[2] > k.shape[2]:
        raise ValueError('Invalid absolute query offset')
    if not isinstance(config.query_block, int) or config.query_block < 1:
        raise ValueError('Positive fixed query tile size is required')


class _RecomputedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, kernel, config, query_offset):
        ctx.kernel = kernel
        ctx.config = config
        ctx.query_offset = query_offset
        ctx.save_for_backward(q, k, v)
        # Function.forward already runs without grad. Explicitly disable
        # autocast in both passes so replay cannot select another arithmetic.
        with torch.autocast(device_type=q.device.type, enabled=False):
            return kernel(q, k, v, config, query_offset=query_offset)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v = ctx.saved_tensors
        needs = ctx.needs_input_grad[:3]
        dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device) if needs[0] else None
        dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device) if needs[1] else None
        dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device) if needs[2] else None
        # BF16 leaf gradients must be cast after cross-query accumulation,
        # rather than rounded once for each query tile then summed in BF16.
        q32, k32, v32 = (x.detach().float() for x in (q, k, v))
        starts = list(range(0, q.shape[2], ctx.config.query_block))
        # Reverse order follows ordinary backward's last-query-first order;
        # floating reductions may nevertheless differ in their exact order.
        for qs in reversed(starts):
            qe = min(q.shape[2], qs + ctx.config.query_block)
            stop = ctx.query_offset + qe if ctx.config.causal else k.shape[2]
            with torch.enable_grad(), torch.autocast(device_type=q.device.type, enabled=False):
                leaves = [q32[:, :, qs:qe].detach().requires_grad_(needs[0]),
                          k32[:, :, :stop].detach().requires_grad_(needs[1]),
                          v32[:, :, :stop].detach().requires_grad_(needs[2])]
                out = ctx.kernel(*leaves, ctx.config, query_offset=ctx.query_offset + qs)
                active = [x for x, need in zip(leaves, needs) if need]
                gradients = torch.autograd.grad(
                    out, active, grad_output[:, :, qs:qe].float(),
                    retain_graph=False, create_graph=False, allow_unused=False)
            by_input = iter(gradients)
            if needs[0]:
                dq[:, :, qs:qe].copy_(next(by_input))
            if needs[1]:
                dk[:, :, :stop].add_(next(by_input))
            if needs[2]:
                dv[:, :, :stop].add_(next(by_input))
            del out, leaves, gradients, active, by_input
        return (dq.to(q.dtype) if dq is not None else None,
                dk.to(k.dtype) if dk is not None else None,
                dv.to(v.dtype) if dv is not None else None,
                None, None, None)


def recomputed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
                         kernel: Callable[..., torch.Tensor], config: Any,
                         query_offset: int = 0) -> torch.Tensor:
    """Apply the supplied deterministic QDQ core with one-query-tile replay.

    Forward values are those of the core; first-order gradients are its STE
    surrogate gradients. The bridge does not add any backward quantization.
    Peak retained replay graph is O(query_block * key_length), plus linear
    Q/K/V/gradient storage. This is not FA's strictly tile-local backward and
    can still be large or slow. No production/NPU readiness is implied.
    """
    _validate(q, k, v, config, query_offset)
    if not callable(kernel):
        raise TypeError('An explicit validated attention core is required')
    return _RecomputedAttention.apply(q, k, v, kernel, config, query_offset)
