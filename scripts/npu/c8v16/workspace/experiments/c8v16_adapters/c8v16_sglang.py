"""Register the named SGLang backend and retain the native Ascend BF16 KV pool."""

import copy
from functools import wraps
import sys

from c8v16_profile import BACKEND_NAME, IS_HIF4, load_module


def make_backend_class(base, core, config, pool_type=None, audit_callback=None):
    # Reuse the logical address translation, GQA dispatch, native writes and
    # request isolation code. Its numerics are injected through the core callback.
    load_module("logical_kv")
    factory = load_module("sglang_bridge").make_backend_class
    shared = factory(base, core=core, config=config, pool_type=pool_type, audit_callback=audit_callback,
                     snapshot_decode_addresses=IS_HIF4)

    class C8V16AttentionBackend(shared):
        pass

    return C8V16AttentionBackend


def wrap_defaults(original):
    @wraps(original)
    def defaults(args):
        requested = args.attention_backend == BACKEND_NAME
        if requested and any(getattr(args, key, None) not in (None, BACKEND_NAME)
                             for key in ("prefill_attention_backend", "decode_attention_backend")):
            raise ValueError("C8V16 requires both prefill and decode to use the shared core")
        result = original(args)
        if requested:
            for key in ("attention_backend", "prefill_attention_backend", "decode_attention_backend"):
                setattr(args, key, BACKEND_NAME)
        return result
    return defaults


def wrap_pool(original, pool_type, allocator_type, audit_callback):
    @wraps(original)
    def allocate(runner, *args, **kwargs):
        requested = runner.server_args
        if requested.attention_backend != BACKEND_NAME:
            return original(runner, *args, **kwargs)
        if (str(runner.device).split(":")[0] not in ("npu", "privateuseone")
                or runner.use_mla_backend or runner.mambaish_config is not None or runner.is_hybrid_swa
                or getattr(runner, "hybrid_gdn_config", None) is not None or runner.is_draft_worker
                or requested.speculative_algorithm is not None or requested.disaggregation_mode not in (None, "null")
                or runner.token_to_kv_pool_allocator is not None):
            raise ValueError("Unsupported C8V16 native KV pool configuration")
        # The pinned NPU memory-pool implementation selects its allocator by
        # the backend name. Give it a private args view only for allocation.
        hw = copy.copy(requested)
        hw.attention_backend = "ascend"
        runner.server_args = hw
        try:
            result = original(runner, *args, **kwargs)
        finally:
            runner.server_args = requested
        if not isinstance(runner.token_to_kv_pool, pool_type) or not isinstance(runner.token_to_kv_pool_allocator, allocator_type):
            raise TypeError("Expected the native NPU MHA pool and paged allocator")
        if audit_callback:
            audit_callback("hardware_kv_pool_created", dict(pool_class=pool_type.__name__, allocator_class=allocator_type.__name__))
        return result
    return allocate


def install(profile, audit):
    from sglang.srt.hardware_backend.npu import utils
    from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.mem_cache import common
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin
    from c8v16_bootstrap import register_cli_choice

    if BACKEND_NAME in ATTENTION_BACKENDS:
        raise RuntimeError(f"Attention backend already installed: {BACKEND_NAME}")
    cls = make_backend_class(AttentionBackend, profile.core.attention, profile.config, NPUMHATokenToKVPool, audit)
    register_cli_choice()
    ATTENTION_BACKENDS[BACKEND_NAME] = lambda runner: cls(runner)
    if IS_HIF4:
        from sglang.srt.model_executor.model_runner import ModelRunner
        check_before_return = load_module("hif4_checks").check_before_return
        original_forward = ModelRunner.forward

        @wraps(original_forward)
        def forward(runner, forward_batch, *args, **kwargs):
            if (runner.server_args.attention_backend == BACKEND_NAME and forward_batch.forward_mode.is_decode()):
                with check_before_return():
                    return original_forward(runner, forward_batch, *args, **kwargs)
            return original_forward(runner, forward_batch, *args, **kwargs)

        ModelRunner.forward = forward
    utils.set_default_server_args = wrap_defaults(utils.set_default_server_args)
    ModelRunnerKVCacheMixin.init_memory_pool = wrap_pool(
        ModelRunnerKVCacheMixin.init_memory_pool, NPUMHATokenToKVPool, NPUPagedTokenToKVPoolAllocator, audit)
    original = common.get_last_loc
    torch_impl = common.get_last_loc_torch

    @wraps(original)
    def last_location(*args, **kwargs):
        if common.get_global_server_args().attention_backend == BACKEND_NAME:
            return torch_impl(*args, **kwargs)
        return original(*args, **kwargs)

    # The pinned SGLang imports this function by value in some modules.
    for name, mod in list(sys.modules.items()):
        if name.startswith("sglang.srt.") and mod is not None and vars(mod).get("get_last_loc") is original:
            mod.get_last_loc = last_location
    common.get_last_loc = last_location
