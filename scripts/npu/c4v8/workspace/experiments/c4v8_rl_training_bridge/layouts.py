"""GQA-compatible SBHD/THD layout adapters with explicit packed boundaries."""
from __future__ import annotations

import torch
from bridge import recomputed_attention


def _offsets(value, name):
    if (not isinstance(value, torch.Tensor) or value.ndim != 1
            or value.dtype not in (torch.int32, torch.int64) or value.numel() < 2):
        raise ValueError(f'{name} must be a one-dimensional int32/int64 cumulative-length tensor')
    result = value.detach().cpu().tolist()
    if result[0] != 0 or any(b < a for a, b in zip(result, result[1:])):
        raise ValueError(f'{name} must start at zero and be nondecreasing')
    return result


def _apply(q, k, v, *, kernel, config, recompute):
    if recompute:
        return recomputed_attention(q, k, v, kernel=kernel, config=config)
    return kernel(q, k, v, config, query_offset=0)


def attention_sbhd(q, k, v, *, kernel, config, recompute=True, flatten_heads=True):
    """Self attention on [sequence,batch,head,dim], returning SB(HD) by default."""
    if (any(x.ndim != 4 for x in (q, k, v)) or q.shape[:2] != k.shape[:2]
            or k.shape != v.shape or q.shape[-1] != k.shape[-1]):
        raise ValueError('SBHD adapter supports self attention with matching sequence/batch/feature dimensions')
    values = [x.permute(1, 2, 0, 3).contiguous() for x in (q, k, v)]
    output = _apply(*values, kernel=kernel, config=config, recompute=recompute)
    output = output.permute(2, 0, 1, 3).contiguous()
    return output.flatten(-2) if flatten_heads else output


def attention_thd(q, k, v, *, cu_seqlens_q, cu_seqlens_kv, kernel, config,
                  cu_seqlens_q_padded=None, cu_seqlens_kv_padded=None,
                  recompute=True, flatten_heads=False):
    """Independent packed self attention, preserving zero-valued padding slots.

    Compact cumulative lengths specify valid lengths. Optional padded offsets
    specify physical starts/ends. Q and KV may have distinct physical padding,
    but the valid self-attention length of each sequence must match. Outputs
    are THD by default because Megatron flattens packed core output itself.
    """
    if (any(x.ndim != 3 for x in (q, k, v)) or k.shape != v.shape
            or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]):
        raise ValueError('Expected valid THD/GQA tensors')
    cq, ck = _offsets(cu_seqlens_q, 'cu_seqlens_q'), _offsets(cu_seqlens_kv, 'cu_seqlens_kv')
    pq = _offsets(cu_seqlens_q_padded, 'cu_seqlens_q_padded') if cu_seqlens_q_padded is not None else cq
    pk = _offsets(cu_seqlens_kv_padded, 'cu_seqlens_kv_padded') if cu_seqlens_kv_padded is not None else ck
    if (len(cq) != len(ck) or len(cq) != len(pq) or len(cq) != len(pk)
            or pq[-1] != q.shape[0] or pk[-1] != k.shape[0]):
        raise ValueError('Packed metadata does not cover the physical Q/KV tensors')
    pieces = []
    for i in range(len(cq)-1):
        nq, nk = cq[i+1]-cq[i], ck[i+1]-ck[i]
        qs, qe, ks, ke = pq[i], pq[i+1], pk[i], pk[i+1]
        if nq != nk or nq > qe-qs or nk > ke-ks:
            raise ValueError('Packed self-attention lengths differ or exceed their physical segments')
        if nq:
            values = [q[qs:qs+nq], k[ks:ks+nk], v[ks:ks+nk]]
            values = [x.transpose(0, 1).unsqueeze(0).contiguous() for x in values]
            out = _apply(*values, kernel=kernel, config=config, recompute=recompute)
            pieces.append(out.squeeze(0).transpose(0, 1).contiguous())
        if qe-qs > nq:
            # Multiplication by zero keeps the output graph valid even for an
            # all-empty pack, with exact zero gradients for padding positions.
            pieces.append(q[qs+nq:qe] * 0)
    output = torch.cat(pieces, dim=0) if pieces else q * 0
    return output.flatten(-2) if flatten_heads else output
