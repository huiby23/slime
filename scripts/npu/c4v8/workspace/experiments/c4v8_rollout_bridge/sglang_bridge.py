"""Explicit experimental SGLang bridge for the audited pinned interfaces.

Importing this module does not import/patch SGLang. Installation is explicit and
must happen in every scheduler process before ServerArgs/default selection.
This file is prepared code, not evidence that a real SG server has executed it.
"""
from __future__ import annotations

import functools
from typing import Callable

import torch

from logical_kv import (DecodeAddresses, _check_common, _check_new_values, _forward_grouped_decode,
                        forward_cached, spans_from_lengths,
                        token_cache_view, validate_cache_write)


BACKEND_NAME = 'hby_c4v8_sim'


def _cpu_values(value, name: str, *, permit_device_copy: bool = False) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f'{name} must be a one-dimensional integer tensor')
        if value.device.type != 'cpu':
            if not permit_device_copy:
                raise ValueError(f'{name} must use the scheduler CPU copy')
            # The pinned ForwardBatch has no req_pool_indices_cpu. This explicit
            # one-per-forward transfer is recorded as a bridge cost, not hidden
            # inside every layer or quantizer.
            value = value.detach().to('cpu')
        return value.tolist()
    if isinstance(value, (tuple, list)) and all(type(item) is int for item in value):
        return list(value)
    raise TypeError(f'{name} is missing or not CPU integer metadata')


def make_backend_class(base_class, *, core: Callable, config, pool_type=None, audit_callback=None,
                       snapshot_decode_addresses=False):
    """Inject the exact attention core and optionally the actual NPU pool type."""
    class C4V8AttentionBackend(base_class):
        def __init__(self, model_runner):
            super().__init__()
            args = model_runner.server_args
            if not args.disable_cuda_graph or getattr(args, 'enable_torch_compile', False):
                raise ValueError('C4V8 functional bridge requires graphs and torch.compile disabled')
            if getattr(args, 'speculative_algorithm', None) not in (None, 'NONE'):
                raise NotImplementedError('Speculative generation is outside this bridge')
            if getattr(args, 'enable_mixed_chunk', False):
                raise NotImplementedError('Disable mixed prefill/decode batches for this first bridge')
            if getattr(model_runner, 'use_mla_backend', False):
                raise NotImplementedError('This bridge only covers ordinary GQA/MHA')
            if not config.causal:
                raise ValueError('A causal numerical profile is required')
            self.page_size = model_runner.page_size
            if type(self.page_size) is not int or self.page_size < 1:
                raise ValueError('Invalid configured page size')
            self.forward_metadata = None
            self.metadata_batch = None
            self.decode_addresses = None
            self.metadata_device_to_host_transfers = 0
            self._audited_forward_modes = set()
            actual_hf32=None
            if hasattr(torch,'npu'):
                actual_hf32=bool(torch.npu.matmul.allow_hf32)
                if actual_hf32:
                    raise RuntimeError('C4V8 simulator requires FP32 matmul with HF32 disabled')
            if audit_callback is not None:
                audit_callback('backend_created', {
                    'backend_class': type(self).__qualname__,
                    'core_module': core.__module__,
                    'core_callable': core.__name__,
                    'tp_rank': getattr(model_runner, 'tp_rank', None),
                    'pp_rank': getattr(model_runner, 'pp_rank', None),
                    'page_size': self.page_size,
                    'actual_allow_hf32':actual_hf32,
                    'actual_selectors': {name: getattr(args, name, None) for name in
                                         ('attention_backend', 'prefill_attention_backend', 'decode_attention_backend')},
                })

        def init_forward_metadata(self, forward_batch):
            self.decode_addresses = None
            mode = forward_batch.forward_mode
            if mode.is_idle():
                self.forward_metadata = ()
                self.metadata_batch = forward_batch
                return
            if getattr(mode, 'name', None) not in ('EXTEND', 'DECODE'):
                raise NotImplementedError('Only ordinary extend/decode is audited')
            if (mode.is_mixed() or mode.is_target_verify() or
                    mode.is_draft_extend(include_v2=True)):
                raise NotImplementedError('Mixed/speculative attention is not supported')
            indices = forward_batch.req_pool_indices
            if isinstance(indices, torch.Tensor) and indices.device.type != 'cpu':
                self.metadata_device_to_host_transfers += 1
            requests = _cpu_values(indices, 'req_pool_indices', permit_device_copy=True)
            seqs = _cpu_values(forward_batch.seq_lens_cpu, 'seq_lens_cpu')
            if mode.is_decode():
                qlens = [1] * len(requests)
            elif mode.is_extend():
                qlens = _cpu_values(forward_batch.extend_seq_lens_cpu, 'extend_seq_lens_cpu')
                prefixes = _cpu_values(forward_batch.extend_prefix_lens_cpu, 'extend_prefix_lens_cpu')
                if len(prefixes) != len(qlens) or any(p + q != s for p, q, s in zip(prefixes, qlens, seqs)):
                    raise ValueError('CPU prefix/query/sequence lengths disagree')
            else:
                raise NotImplementedError(f'Unsupported forward mode {mode!r}')
            if len(requests) != forward_batch.batch_size:
                raise ValueError('Metadata request count differs from batch size')
            self.forward_metadata = spans_from_lengths(requests, seqs, qlens)
            self.metadata_batch = forward_batch

        def _run(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
            if any(value is not None for value in kwargs.values()):
                raise NotImplementedError('MLA/rope/sink auxiliary attention arguments are unsupported')
            if self.metadata_batch is not forward_batch or self.forward_metadata is None:
                raise RuntimeError('init_forward_metadata was not called for this batch')
            if (layer.is_cross_attention or str(getattr(layer.attn_type, 'value', layer.attn_type)) != 'decoder'
                    or layer.qk_head_dim != layer.v_head_dim or layer.tp_k_head_num != layer.tp_v_head_num):
                raise NotImplementedError('Only decoder GQA with equal QK/V dimensions is supported')
            if layer.logit_cap != 0 or layer.sliding_window_size not in (-1, None):
                raise NotImplementedError('Softcap/sliding-window semantics are not this profile')
            if getattr(layer, 'quant_method', None) is not None:
                raise NotImplementedError('Do not combine this profile with a quantized KV attention method')
            pool = forward_batch.token_to_kv_pool
            if pool_type is not None and not isinstance(pool, pool_type):
                raise TypeError('Unexpected native KV pool layout')
            if getattr(pool, 'use_fia', False):
                raise NotImplementedError('This audited path requires ASCEND_USE_FIA=False')
            # Audited NPUMHATokenToKVPool creates contiguous [page,page_size,H,D].
            raw_k = pool.get_key_buffer(layer.layer_id)
            raw_v = pool.get_value_buffer(layer.layer_id)
            kc = token_cache_view(raw_k, page_size=self.page_size)
            vc = token_cache_view(raw_v, page_size=self.page_size)
            query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
            table = forward_batch.req_to_token_pool.req_to_token
            spans = self.forward_metadata
            _check_common(query, kc, vc, table, spans, layer.scaling, True)
            decode = forward_batch.forward_mode.is_decode()
            addresses = None
            if snapshot_decode_addresses and decode:
                locations = forward_batch.out_cache_loc if save_kv_cache else None
                if self.decode_addresses is None:
                    self.decode_addresses = DecodeAddresses(table, locations, spans, kc.size(0), pool)
                addresses = self.decode_addresses
                addresses.verify(table, locations, kc.size(0), pool)
            if save_kv_cache:
                if k is None or v is None:
                    raise ValueError('Missing KV for native cache update')
                keys = k.reshape(-1, layer.tp_k_head_num, layer.qk_head_dim)
                values = v.reshape(-1, layer.tp_v_head_num, layer.v_head_dim)
                if addresses is None:
                    locations = forward_batch.out_cache_loc
                    validate_cache_write(query, kc, vc, table, spans, keys, values, locations)
                else:
                    _check_new_values(query, kc, vc, keys, values)
                    locations = addresses.locations
                pool.set_kv_buffer(layer, locations, keys, values)
            if addresses is None:
                output = forward_cached(query, kc, vc, table, spans, core=core, config=config,
                                        scale=layer.scaling, group_same_length_decode=decode)
            else:
                output = _forward_grouped_decode(query, kc, vc, spans, addresses.logical, core=core, config=config)
            mode = forward_batch.forward_mode.name
            if audit_callback is not None and mode not in self._audited_forward_modes:
                audit_callback('first_' + mode.lower(), {
                    'layer_id': layer.layer_id, 'query_shape': list(query.shape),
                    'cache_shape': list(raw_k.shape), 'cache_dtype': str(raw_k.dtype),
                    'core_output_shape': list(output.shape),
                    'core_module': core.__module__, 'core_callable': core.__name__,
                    'metadata_device_to_host_transfers': self.metadata_device_to_host_transfers,
                })
                self._audited_forward_modes.add(mode)
            return output.reshape(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

        def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
            if not forward_batch.forward_mode.is_extend():
                raise ValueError('forward_extend received a non-extend batch')
            return self._run(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

        def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
            if not forward_batch.forward_mode.is_decode():
                raise ValueError('forward_decode received a non-decode batch')
            return self._run(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

        def support_triton(self):
            return False

    return C4V8AttentionBackend


def wrap_npu_defaults(original: Callable) -> Callable:
    """Preserve every normal default; override selection only on explicit opt-in.

    The pinned set_default_server_args unconditionally selects ascend. Keeping
    its memory/page/all-reduce behavior, then restoring our requested selector,
    avoids changing the production ascend source file or any unrelated request.
    """
    @functools.wraps(original)
    def selected(args):
        requested = args.attention_backend == BACKEND_NAME
        if requested and any(getattr(args, name, None) not in (None, BACKEND_NAME)
                             for name in ('prefill_attention_backend', 'decode_attention_backend')):
            raise ValueError('C4V8 must cover both prefill and decode')
        original(args)
        if requested:
            args.attention_backend = BACKEND_NAME
            args.prefill_attention_backend = BACKEND_NAME
            args.decode_attention_backend = BACKEND_NAME
    selected._hby_c4v8_selector = True
    return selected


def install_in_sglang(*, core: Callable, config, audit_callback=None):
    """Explicit per-process registration; production source files stay unchanged.

    Must be invoked before ServerArgs is instantiated in each relevant process.
    A dedicated entrypoint/worker bootstrap must arrange this; no automatic
    sitecustomize or server startup is provided by this module.
    """
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES, add_attention_backend_choices
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
    from sglang.srt.hardware_backend.npu import utils as npu_utils
    if BACKEND_NAME in ATTENTION_BACKENDS:
        raise RuntimeError('C4V8 backend already installed; reject ambiguous profile replacement')
    if getattr(npu_utils.set_default_server_args, '_hby_c4v8_selector', False):
        raise RuntimeError('NPU selector was already wrapped unexpectedly')
    from hardware_compat import install_hardware_compat
    install_hardware_compat(audit_callback=audit_callback)
    cls = make_backend_class(AttentionBackend, core=core, config=config,
                             pool_type=NPUMHATokenToKVPool, audit_callback=audit_callback)
    ATTENTION_BACKENDS[BACKEND_NAME] = lambda runner: cls(runner)
    if BACKEND_NAME not in ATTENTION_BACKEND_CHOICES:
        add_attention_backend_choices([BACKEND_NAME])
    npu_utils.set_default_server_args = wrap_npu_defaults(npu_utils.set_default_server_args)
    return cls
