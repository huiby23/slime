"""Adapt native SGLang BF16 paged KV to request-local attention operands."""

import torch

from .attention import attention, decode


def addresses(batch, table):
    # ForwardBatch lives for one model forward. Cache addresses across layers,
    # never values, and never carry them across requests or weight versions.
    found = getattr(batch, "_nvfp4_fa_addresses", None)
    if found is not None:
        return found
    mode = batch.forward_mode.name
    lengths = batch.seq_lens_cpu.tolist()
    rows = batch.req_pool_indices.long()
    if len(lengths) != rows.numel() or min(lengths) < 1:
        raise ValueError("Invalid SGLang request metadata")
    if mode == "DECODE":
        lens = batch.seq_lens.long()
        pos = torch.arange(max(lengths), device=rows.device)[None, :]
        slots = table[rows[:, None], torch.minimum(pos, lens[:, None] - 1)].long()
        found = (slots, lens)
    elif mode == "EXTEND":
        query_lens = list(batch.extend_seq_lens_cpu)
        if len(query_lens) != len(lengths) or any(not 0 < n <= l for n, l in zip(query_lens, lengths)):
            raise ValueError("Invalid cached prefill lengths")
        found = [
            (table[row, :length].long(), length - n, n)
            for row, length, n in zip(rows.tolist(), lengths, query_lens, strict=True)
        ]
    else:
        raise ValueError(f"Unsupported SGLang forward mode {mode}")
    batch._nvfp4_fa_addresses = found
    return found


def paged_forward(layer, q, k, v, batch, backend):
    head_dim = layer.head_dim
    q = q.reshape(-1, layer.tp_q_head_num, head_dim)
    k = k.reshape(-1, layer.tp_k_head_num, head_dim)
    v = v.reshape_as(k)
    if q.shape[0] != k.shape[0] or q.size(0) != batch.out_cache_loc.numel():
        raise ValueError("Unexpected KV write count")
    cache = backend.token_to_kv_pool
    if cache.dtype != torch.bfloat16 or k.dtype != cache.dtype or v.dtype != k.dtype:
        raise ValueError("The W16A16 recipe requires a BF16 KV pool")
    cache.set_kv_buffer(layer, batch.out_cache_loc, k, v)
    keys, vals = cache.get_key_buffer(layer.layer_id), cache.get_value_buffer(layer.layer_id)
    layout = addresses(batch, backend.req_to_token_pool.req_to_token)
    if batch.forward_mode.name == "DECODE":
        slots, lens = layout
        k, v = (x[slots].transpose(1, 2) for x in (keys, vals))
        result = decode(q.unsqueeze(2), k, v, lens).squeeze(2)
    else:
        pieces = []
        start = 0
        for slots, offset, n in layout:
            k, v = (x.index_select(0, slots).transpose(0, 1).unsqueeze(0) for x in (keys, vals))
            query = q[start : start + n].transpose(0, 1).unsqueeze(0)
            pieces.append(attention(query, k, v, offset=offset).squeeze(0).transpose(0, 1))
            start += n
        if start != len(q):
            raise ValueError("Packed queries differ from SGLang's extend metadata")
        result = torch.cat(pieces)
    return result.reshape(q.size(0), -1)


def radix_forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
    from sglang.srt.model_executor.forward_context import get_attn_backend

    if forward_batch.forward_mode.name == "IDLE" and q.numel() == 0:
        return q.reshape(0, self.tp_q_head_num * self.head_dim)
    if not save_kv_cache or kwargs or k is None or v is None:
        raise ValueError("NVFP4 requires explicit native Q/K/V and KV writes")
    from .records import serving_receipt

    result = paged_forward(self, q, k, v, forward_batch, get_attn_backend())
    serving_receipt(self, forward_batch)
    return result
