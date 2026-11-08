"""CPU tests execute the full actual pinned init_memory_pool method with fake allocators."""
import ast
from contextlib import ExitStack
import logging
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
from hardware_compat import BACKEND_NAME,wrap_init_memory_pool,wrap_get_last_loc

ROOT=Path(__file__).resolve().parent


class FakePool:
    def __init__(self,*args,**kwargs):self.args=args;self.kwargs=kwargs


class FakeAllocator(FakePool):pass


def original_init():
    path=ROOT/'pinned-source/model_runner_kv_cache_mixin.py'
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='init_memory_pool')
    scope=dict(torch=torch,logging=logging,logger=logging.getLogger(__name__),
               get_available_gpu_memory=lambda *a:64.,get_attention_tp_size=lambda:1,
               is_deepseek_nsa=lambda c:False,is_float4_e2m1fn_x2=lambda d:False,
               _is_npu=True,ReqToTokenPool=FakePool)
    exec(compile('from __future__ import annotations\n'+ast.unparse(node),str(path),'exec'),scope)
    return scope['init_memory_pool']


def runner():
    args=NS(attention_backend=BACKEND_NAME,prefill_attention_backend=BACKEND_NAME,
            decode_attention_backend=BACKEND_NAME,max_running_requests=4,max_total_tokens=128,
            page_size=128,speculative_algorithm=None,speculative_num_draft_tokens=None,
            disaggregation_mode='null',enable_memory_saver=True)
    return NS(server_args=args,device='npu',gpu_id=1,page_size=128,
              profile_max_num_token=lambda mem:256,model_config=NS(context_len=128,hf_config=NS(),
                  get_num_kv_heads=lambda tp:2,head_dim=16),mambaish_config=None,
              spec_algorithm=NS(is_none=lambda:True),pp_size=1,is_hybrid_swa=False,
              req_to_token_pool=None,use_mla_backend=False,kv_cache_dtype=torch.bfloat16,
              num_effective_layers=1,start_layer=0,end_layer=1,is_draft_worker=False,
              token_to_kv_pool_allocator=None,hybrid_gdn_config=None)


class HardwareCompatibilityTests(unittest.TestCase):
    def stub_modules(self,pool_type=FakePool):
        pool=ModuleType('sglang.srt.hardware_backend.npu.memory_pool_npu');pool.NPUMHATokenToKVPool=pool_type
        allocator=ModuleType('sglang.srt.hardware_backend.npu.allocator_npu');allocator.NPUPagedTokenToKVPoolAllocator=FakeAllocator
        return patch.dict(sys.modules,{pool.__name__:pool,allocator.__name__:allocator})

    def test_full_pinned_method_selects_native_pool_allocator_and_restores_args(self):
        r=runner();shared=r.server_args;events=[]
        wrapper=wrap_init_memory_pool(original_init(),pool_type=FakePool,allocator_type=FakeAllocator,
                                      audit_callback=lambda *a:events.append(a))
        with self.stub_modules():wrapper(r,64)
        self.assertIs(r.server_args,shared)
        self.assertEqual(shared.attention_backend,BACKEND_NAME)
        self.assertIs(type(r.token_to_kv_pool),FakePool)
        self.assertIs(type(r.token_to_kv_pool_allocator),FakeAllocator)
        self.assertTrue(r.token_to_kv_pool.kwargs['enable_memory_saver'])
        self.assertIs(r.token_to_kv_pool_allocator.kwargs['kvcache'],r.token_to_kv_pool)
        self.assertEqual(events[0][0],'hardware_kv_pool_created')
        self.assertEqual(events[0][1]['actual_selectors']['attention_backend'],BACKEND_NAME)

    def test_original_error_propagates_and_shared_selectors_are_restored(self):
        class BrokenPool:
            def __init__(self,*a,**k):raise RuntimeError('real pool failure sentinel')
        r=runner();shared=r.server_args
        wrapper=wrap_init_memory_pool(original_init(),pool_type=FakePool,allocator_type=FakeAllocator)
        with self.stub_modules(BrokenPool),self.assertRaisesRegex(RuntimeError,'real pool failure sentinel'):
            wrapper(r,64)
        self.assertIs(r.server_args,shared);self.assertEqual(shared.attention_backend,BACKEND_NAME)

    def test_noncustom_calls_original_without_changing_identity(self):
        r=runner();r.server_args.attention_backend='ascend';shared=r.server_args;seen=[]
        original=lambda x,*a,**k:seen.append(x.server_args) or 'native'
        self.assertEqual(wrap_init_memory_pool(original,pool_type=FakePool,allocator_type=FakeAllocator)(r,64),'native')
        self.assertIs(seen[0],shared)

    def test_unsupported_path_rejected_before_native_allocation(self):
        r=runner();r.use_mla_backend=True;calls=[]
        wrapper=wrap_init_memory_pool(lambda *a:calls.append(1),pool_type=FakePool,allocator_type=FakeAllocator)
        with self.assertRaises(ValueError):wrapper(r,64)
        self.assertEqual(calls,[])

    def test_last_location_custom_uses_pinned_torch_other_names_original(self):
        path=ROOT/'pinned-source/mem_cache_common.py';tree=ast.parse(path.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_last_loc_torch')
        scope={'torch':torch};exec(compile(ast.unparse(node),str(path),'exec'),scope)
        args=NS(attention_backend=BACKEND_NAME)
        wrapper=wrap_get_last_loc(lambda *a:'original',scope['get_last_loc_torch'],lambda:args)
        table=torch.tensor([[1,4,8],[2,5,9]])
        result=wrapper(table,torch.tensor([0,1]),torch.tensor([0,2]))
        self.assertEqual(result.tolist(),[-1,5])
        args.attention_backend='ascend';self.assertEqual(wrapper(None,None,None),'original')


if __name__=='__main__':unittest.main()
