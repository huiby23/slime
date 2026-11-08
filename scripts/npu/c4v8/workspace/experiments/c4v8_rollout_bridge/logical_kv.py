"""Request-isolated logical-KV bridge; not yet a registered SGLang backend.

This module has no SGLang or torch_npu import and never changes model projection
layers. A verified caller supplies token-addressable BF16 cache views and CPU
metadata. KV values remain BF16; all numerical quantization is delegated to the
same attention core used by training. No accelerator is initialized on import.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import math
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class RequestSpan:
    request_index: int
    prefix_length: int
    query_length: int

    def __post_init__(self):
        values = (self.request_index, self.prefix_length, self.query_length)
        if any(type(value) is not int for value in values):
            raise TypeError('Request metadata must be CPU Python integers')
        if self.request_index < 0 or self.prefix_length < 0 or self.query_length < 1:
            raise ValueError('Invalid request index, prefix, or query length')

    @property
    def sequence_length(self) -> int:
        return self.prefix_length + self.query_length


def spans_from_lengths(request_indices: Sequence[int], sequence_lengths: Sequence[int],
                       query_lengths: Sequence[int]) -> tuple[RequestSpan, ...]:
    """Build extend spans, including decode where every query length is one.

    Metadata must come from the scheduler's CPU copies. Passing device tensors
    is intentionally rejected instead of hiding a synchronizing .cpu() transfer.
    """
    arrays = (request_indices, sequence_lengths, query_lengths)
    if any(isinstance(values, torch.Tensor) for values in arrays):
        raise TypeError('Use already available CPU scheduler metadata')
    if len({len(values) for values in arrays}) != 1:
        raise ValueError('Batch metadata lengths differ')
    spans = tuple(RequestSpan(req, seq - qlen, qlen)
                  for req, seq, qlen in zip(*arrays))
    if len({span.request_index for span in spans}) != len(spans):
        raise ValueError('Duplicate request indices; speculative branching is unsupported')
    return spans


def token_cache_view(cache: torch.Tensor, *, page_size: int | None = None) -> torch.Tensor:
    """Accept only documented logical layouts, never guess an Ascend native ABI.

    [physical_token, KV_head, feature] is accepted directly. A contiguous
    [page, page_token, KV_head, feature] test/adapter layout can be flattened with
    an explicit page_size. Native NZ/blocked-head formats need a verified reader.
    """
    if cache.ndim == 3 and page_size is None:
        return cache
    if (cache.ndim == 4 and type(page_size) is int and page_size > 0
            and cache.shape[1] == page_size and cache.is_contiguous()):
        return cache.view(-1, cache.shape[2], cache.shape[3])
    raise ValueError('Cache layout is not an explicitly supported token view')


def _check_common(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                  req_to_token: torch.Tensor, spans: tuple[RequestSpan, ...], scale: float,
                  require_bf16_cache: bool) -> None:
    if q.ndim != 3 or key_cache.ndim != 3 or key_cache.shape != value_cache.shape:
        raise ValueError('Expected Q [tokens,Qheads,D] and matching KV [slots,KVheads,D]')
    if min(q.shape[1:]) < 1 or min(key_cache.shape) < 1:
        raise ValueError('Head/feature/cache dimensions must be positive')
    if q.shape[-1] != key_cache.shape[-1] or q.shape[1] % key_cache.shape[1]:
        raise ValueError('Invalid head dimension or GQA head relationship')
    if q.shape[0] != sum(span.query_length for span in spans):
        raise ValueError('Query token count differs from request spans')
    if len({span.request_index for span in spans}) != len(spans):
        raise ValueError('Each request must occur once in this batch')
    if req_to_token.ndim != 2 or req_to_token.dtype not in (torch.int32, torch.int64):
        raise ValueError('req_to_token must be an integer [request,logical_token] table')
    if any(span.request_index >= req_to_token.shape[0] or
           span.sequence_length > req_to_token.shape[1] for span in spans):
        raise ValueError('Request span exceeds mapping table')
    tensors = (q, key_cache, value_cache, req_to_token)
    if any(value.device != q.device for value in tensors):
        raise ValueError('Q, KV views, and address table must be on the same device')
    if q.dtype != key_cache.dtype or q.dtype != value_cache.dtype:
        raise ValueError('Q and KV storage dtypes differ')
    if require_bf16_cache and key_cache.dtype != torch.bfloat16:
        raise ValueError('This experiment requires BF16 KV storage')
    if not q.is_floating_point():
        raise ValueError('Q must be floating point')
    expected_scale = 1.0 / math.sqrt(q.shape[-1])
    if not math.isfinite(scale) or not math.isclose(scale, expected_scale, rel_tol=1e-12):
        raise NotImplementedError('Core currently supports only post-QK 1/sqrt(D) scaling')


def _logical_slots(req_to_token: torch.Tensor, span: RequestSpan,
                   cache_slots: int) -> torch.Tensor:
    slots = req_to_token[span.request_index, :span.sequence_length].to(torch.int64)
    # These checks can synchronize a device. They are deliberate validation gates
    # in this functional bridge, not a claim of graph-safe/high-throughput serving.
    if bool(((slots < 0) | (slots >= cache_slots)).any()):
        raise ValueError('Invalid physical KV address in active logical sequence')
    if torch.unique(slots).numel() != slots.numel():
        raise ValueError('A logical sequence repeats physical token slots')
    return slots


def validate_cache_write(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                         req_to_token: torch.Tensor, spans: Sequence[RequestSpan],
                         new_key: torch.Tensor, new_value: torch.Tensor,
                         out_cache_locations: torch.Tensor) -> torch.Tensor:
    """Validate native cache writes before a caller invokes set_kv_buffer.

    Shape/device/scale checks common to attention must also precede this call.
    Returns packed int64 addresses without modifying any cache.
    """
    if not spans:
        raise ValueError('An idle batch must not write KV')
    logical = [_logical_slots(req_to_token, span, key_cache.shape[0]) for span in spans]
    expected_shape = (q.shape[0], key_cache.shape[1], key_cache.shape[2])
    if new_key.shape != expected_shape or new_value.shape != expected_shape:
        raise ValueError('New KV tokens do not match packed query spans')
    if (new_key.dtype != key_cache.dtype or new_value.dtype != value_cache.dtype
            or new_key.device != q.device or new_value.device != q.device):
        raise ValueError('New KV storage differs from the cache')
    if (out_cache_locations.device != q.device or out_cache_locations.ndim != 1
            or out_cache_locations.numel() != q.shape[0]
            or out_cache_locations.dtype not in (torch.int32, torch.int64)):
        raise ValueError('Invalid packed cache-write locations')
    locations = out_cache_locations.to(torch.int64)
    expected = torch.cat([slots[span.prefix_length:] for slots, span in zip(logical, spans)])
    if not torch.equal(locations, expected):
        raise ValueError('Write locations disagree with request logical mapping')
    if torch.unique(locations).numel() != locations.numel():
        raise ValueError('New KV writes alias within this batch')
    prefixes = torch.cat([slots[:span.prefix_length] for slots, span in zip(logical, spans)])
    if prefixes.numel() and bool(torch.isin(locations, prefixes).any()):
        raise ValueError('New KV writes alias a cached prefix in this batch')
    return locations


@torch.no_grad()
def forward_cached(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                   req_to_token: torch.Tensor, spans: Sequence[RequestSpan], *,
                   core: Callable, config, scale: float,
                   new_key: torch.Tensor | None = None,
                   new_value: torch.Tensor | None = None,
                   out_cache_locations: torch.Tensor | None = None,
                   require_bf16_cache: bool = True,
                   sliding_window: int | None = None,
                   logit_softcap: float | None = None,
                   group_same_length_decode: bool = False) -> torch.Tensor:
    """Run complete attention independently for each request's new Q tokens.

    If the caller already stored KV through its verified native cache API, omit
    all three write arguments. Otherwise all three are required and are checked
    against the logical mapping before any mutation. New-token addresses may not
    alias another request's cached prefix in this batch. Shared prefixes are
    allowed for reads. The core receives [1,H,Q,D], [1,HKV,S,D], query_offset=P.

    Every invocation starts key tiles at logical token zero, independent of
    physical pages, query batch packing, and prefill chunks. This is the same
    numerical attention function as training, not native SDPA plus QKV hooks.
    """
    spans = tuple(spans)
    if any(not isinstance(span, RequestSpan) for span in spans):
        raise TypeError('spans must contain RequestSpan entries')
    if group_same_length_decode and any(span.query_length != 1 for span in spans):
        raise ValueError('Equal-length grouping is explicit single-query decode only')
    if sliding_window not in (None, -1) or logit_softcap not in (None, 0.0):
        raise NotImplementedError('Sliding window and logit softcap are not this profile')
    if not getattr(config, 'causal', False):
        raise ValueError('The rollout bridge requires causal attention')
    _check_common(q, key_cache, value_cache, req_to_token, spans, scale, require_bf16_cache)
    logical = [_logical_slots(req_to_token, span, key_cache.shape[0]) for span in spans]
    write_args = (new_key, new_value, out_cache_locations)
    if any(value is not None for value in write_args):
        if not all(isinstance(value, torch.Tensor) for value in write_args):
            raise ValueError('All three KV-write arguments must be supplied together')
        locations = validate_cache_write(q, key_cache, value_cache, req_to_token, spans,
                                         new_key, new_value, out_cache_locations)
        key_cache.index_copy_(0, locations, new_key)
        value_cache.index_copy_(0, locations, new_value)
    if not spans:
        return torch.empty_like(q)
    if group_same_length_decode:
        return _forward_grouped_decode(q, key_cache, value_cache, spans, logical, core=core, config=config)
    output = torch.empty_like(q)
    offset = 0
    for span, slots in zip(spans, logical):
        end = offset + span.query_length
        query = q[offset:end].transpose(0, 1).unsqueeze(0)
        keys = key_cache.index_select(0, slots).transpose(0, 1).unsqueeze(0)
        values = value_cache.index_select(0, slots).transpose(0, 1).unsqueeze(0)
        result = core(query, keys, values, config, query_offset=span.prefix_length)
        expected = (1, q.shape[1], span.query_length, q.shape[2])
        if result.shape != expected or result.dtype != q.dtype or result.device != q.device:
            raise ValueError('Attention core returned incompatible output')
        output[offset:end].copy_(result.squeeze(0).transpose(0, 1))
        offset = end
    return output


def _forward_grouped_decode(q,key_cache,value_cache,spans,logical,*,core,config):
    """Called after original validation/write path; q row i is spans[i]'s query."""
    buckets=OrderedDict()
    for index,span in enumerate(spans):
        buckets.setdefault(span.sequence_length,[]).append(index)
    output=torch.empty_like(q)
    for length,indices in buckets.items():
        if len(indices)==1:
            # Keep the existing singleton B1 shape/stride path unchanged.
            index=indices[0];slots=logical[index]
            query=q[index:index+1].transpose(0,1).unsqueeze(0)
            keys=key_cache.index_select(0,slots).transpose(0,1).unsqueeze(0)
            values=value_cache.index_select(0,slots).transpose(0,1).unsqueeze(0)
            result=core(query,keys,values,config,query_offset=length-1)
            expected=(1,q.shape[1],1,q.shape[2])
            if result.shape!=expected or result.dtype!=q.dtype or result.device!=q.device:
                raise ValueError('Attention core returned incompatible output')
            output[index:index+1].copy_(result.squeeze(0).transpose(0,1))
            continue
        packed_rows=torch.tensor(indices,dtype=torch.int64,device=q.device)
        # Different requests may legally share prefix slots for reads. There is
        # no cross-request reduction and no flattening across head-dimension rows.
        slots=torch.stack([logical[index] for index in indices])
        group=len(indices);kvheads=key_cache.shape[1];dimension=key_cache.shape[2]
        query=q.index_select(0,packed_rows).unsqueeze(2)
        keys=key_cache.index_select(0,slots.reshape(-1)).reshape(group,length,kvheads,dimension).transpose(1,2)
        values=value_cache.index_select(0,slots.reshape(-1)).reshape(group,length,kvheads,dimension).transpose(1,2)
        result=core(query,keys,values,config,query_offset=length-1)
        expected=(group,q.shape[1],1,q.shape[2])
        if result.shape!=expected or result.dtype!=q.dtype or result.device!=q.device:
            raise ValueError('Grouped attention core returned incompatible output')
        output.index_copy_(0,packed_rows,result.squeeze(2))
    return output


def bridge_metadata() -> dict:
    return {
        'registered_sglang_backend': False,
        'native_cache_layout_verified': False,
        'npu_validated': False,
        'cache_storage': 'BF16, unchanged by attention QDQ',
        'supported_modes': ['causal extend with optional prefix', 'single-token decode'],
        'key_tiles': 'logical position zero, independent of page/chunk layout',
        'precision_source': 'injected shared attention core and explicit config',
        'limitations': ['full logical KV gather per request', 'Python per-request dispatch',
                        'validation checks can synchronize device', 'no speculative branching',
                        'no sliding window/softcap/CUDA graphs'],
    }
