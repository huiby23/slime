"""Opt-in hardware plumbing for a named attention backend on pinned SGLang.

The backend name controls both numerical attention and NPU pool selection in
this version. During ONLY native pool initialization, use a private shallow
copy of ServerArgs with hardware attention_backend='ascend'. The actual shared
ServerArgs remains the experimental selector, and is restored in finally even
if a real pool/device error propagates. No production source file is changed.
"""
import copy
import functools


BACKEND_NAME='hby_c4v8_sim'


def wrap_init_memory_pool(original,*,pool_type,allocator_type,audit_callback=None):
    @functools.wraps(original)
    def initialize(runner,*args,**kwargs):
        requested=runner.server_args
        if requested.attention_backend!=BACKEND_NAME:
            return original(runner,*args,**kwargs)
        if str(runner.device).split(':',1)[0] not in ('npu','privateuseone'):
            raise ValueError('C4V8 native pool adapter requires the NPU device')
        if (runner.use_mla_backend or runner.mambaish_config is not None
                or runner.is_hybrid_swa or getattr(runner,'hybrid_gdn_config',None) is not None
                or runner.is_draft_worker or requested.speculative_algorithm is not None):
            raise ValueError('C4V8 native pool adapter only covers ordinary Qwen GQA/MHA')
        if requested.disaggregation_mode not in ('null',None):
            raise ValueError('Disaggregated serving is outside the audited C4V8 pool path')
        if runner.token_to_kv_pool_allocator is not None:
            raise ValueError('Unexpected preexisting allocator in non-draft C4V8 initialization')
        # The inspected non-speculative method does not mutate ServerArgs.
        # A private view avoids changing the global/shared selectors at all.
        hardware_args=copy.copy(requested)
        hardware_args.attention_backend='ascend'
        runner.server_args=hardware_args
        try:
            result=original(runner,*args,**kwargs)
        finally:
            runner.server_args=requested
        if not isinstance(runner.token_to_kv_pool,pool_type):
            raise TypeError('Native initialization did not create the audited NPU MHA KV pool')
        if not isinstance(runner.token_to_kv_pool_allocator,allocator_type):
            raise TypeError('Native initialization did not create the NPU paged allocator')
        if audit_callback is not None:
            audit_callback('hardware_kv_pool_created',{
                'pool_class':type(runner.token_to_kv_pool).__name__,
                'allocator_class':type(runner.token_to_kv_pool_allocator).__name__,
                'page_size':runner.page_size,'cache_dtype':str(runner.kv_cache_dtype),
                'enable_memory_saver':requested.enable_memory_saver,
                'hardware_selection_scope':'private ServerArgs view during native init_memory_pool only',
                'actual_selectors':{name:getattr(requested,name,None) for name in
                    ('attention_backend','prefill_attention_backend','decode_attention_backend')},
            })
        return result
    initialize._hby_c4v8_hardware_pool=True
    return initialize


def wrap_get_last_loc(original,torch_impl,get_server_args):
    @functools.wraps(original)
    def get_last_loc(*args,**kwargs):
        if get_server_args().attention_backend==BACKEND_NAME:
            return torch_impl(*args,**kwargs)
        return original(*args,**kwargs)
    get_last_loc._hby_c4v8_last_loc=True
    return get_last_loc


def install_hardware_compat(*,audit_callback=None):
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
    from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache import common
    import sys
    if getattr(ModelRunnerKVCacheMixin.init_memory_pool,'_hby_c4v8_hardware_pool',False):
        raise RuntimeError('C4V8 native pool adapter already installed')
    ModelRunnerKVCacheMixin.init_memory_pool=wrap_init_memory_pool(
        ModelRunnerKVCacheMixin.init_memory_pool,pool_type=NPUMHATokenToKVPool,
        allocator_type=NPUPagedTokenToKVPoolAllocator,audit_callback=audit_callback)
    original=common.get_last_loc
    replacement=wrap_get_last_loc(original,common.get_last_loc_torch,common.get_global_server_args)
    # Existing from-imports may precede bootstrap. Rebind only this exact native
    # function identity inside already loaded SGLang modules; other functions
    # and all non-custom calls retain their original implementations.
    for name,module in list(sys.modules.items()):
        if name.startswith('sglang.srt.') and module is not None:
            if vars(module).get('get_last_loc') is original:
                module.get_last_loc=replacement
    common.get_last_loc=replacement
