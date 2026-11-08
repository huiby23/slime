"""CPU interface tests against AST-extracted pinned base/enum, not a live server."""
from abc import ABC, abstractmethod
import ast
from enum import IntEnum, auto
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'c4v8_device_core'))
from device_attention import AttentionConfig, attention
from sglang_bridge import BACKEND_NAME, make_backend_class, wrap_npu_defaults


def pinned_class(file, name):
    tree = ast.parse((ROOT / 'pinned-source' / file).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    source = 'from __future__ import annotations\n' + ast.unparse(cls)
    scope = dict(ABC=ABC, abstractmethod=abstractmethod, torch=torch,
                 IntEnum=IntEnum, auto=auto, is_npu=lambda: False)
    exec(compile(source, file, 'exec'), scope)
    return scope[name]


Base = pinned_class('base_attn_backend.py', 'AttentionBackend')
Mode = pinned_class('forward_batch_info.py', 'ForwardMode')


class FakePool:
    def __init__(self, hk, d):
        self.k = torch.zeros((16, 4, hk, d), dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)
        self.use_fia = False
        self.writes = 0

    def get_key_buffer(self, layer): return self.k
    def get_value_buffer(self, layer): return self.v

    def set_kv_buffer(self, layer, loc, k, v):
        self.writes += 1
        self.k.view(-1, *self.k.shape[2:])[loc] = k
        self.v.view(-1, *self.v.shape[2:])[loc] = v


class SGLangInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = AttentionConfig(query_block=5, key_block=8)
        self.runner = NS(page_size=4, use_mla_backend=False, server_args=NS(
            disable_cuda_graph=True, enable_torch_compile=False,
            speculative_algorithm=None, enable_mixed_chunk=False))
        self.cls = make_backend_class(Base, core=attention, config=self.config, pool_type=FakePool)
        self.backend = self.cls(self.runner)
        self.hq, self.hk, self.d = 4, 2, 17
        self.layer = NS(tp_q_head_num=4, tp_k_head_num=2, tp_v_head_num=2,
                        qk_head_dim=17, v_head_dim=17, scaling=1/math.sqrt(17),
                        is_cross_attention=False, attn_type='decoder', layer_id=0,
                        logit_cap=0.0, sliding_window_size=-1, quant_method=None)
        gen = torch.Generator().manual_seed(1408)
        self.q, self.k, self.v = (torch.randn((19, h, self.d), generator=gen).bfloat16()
                                  for h in (self.hq, self.hk, self.hk))
        self.pool = FakePool(self.hk, self.d)
        self.table = torch.full((3, 24), -1, dtype=torch.int32)
        self.slots = torch.randperm(63, generator=gen)[:19] + 1
        self.table[2, :19] = self.slots.to(torch.int32)

    def batch(self, start, end, mode):
        return NS(forward_mode=mode, batch_size=1, req_pool_indices=torch.tensor([2]),
                  seq_lens_cpu=torch.tensor([end]), extend_seq_lens_cpu=[end-start],
                  extend_prefix_lens_cpu=[start], out_cache_loc=self.slots[start:end],
                  token_to_kv_pool=self.pool, req_to_token_pool=NS(req_to_token=self.table))

    def test_pinned_base_dispatch_with_native_write_api(self):
        expected = attention(*(x.transpose(0,1).unsqueeze(0) for x in (self.q,self.k,self.v)),
                             self.config).squeeze(0).transpose(0,1).flatten(1)
        outputs=[]
        for start,end,mode in [(0,7,Mode.EXTEND),(7,17,Mode.EXTEND),
                               (17,18,Mode.DECODE),(18,19,Mode.DECODE)]:
            batch=self.batch(start,end,mode)
            self.backend.init_forward_metadata(batch)
            outputs.append(self.backend.forward(self.q[start:end].flatten(1),self.k[start:end],
                            self.v[start:end],self.layer,batch,True))
        torch.testing.assert_close(torch.cat(outputs),expected,atol=0,rtol=0)
        self.assertEqual(self.pool.writes,4)
        self.assertEqual(self.backend.metadata_device_to_host_transfers,0)
        torch.testing.assert_close(self.pool.k.view(-1,self.hk,self.d)[self.slots],self.k,atol=0,rtol=0)

    def test_metadata_required_and_unsupported_paths_rejected(self):
        batch=self.batch(0,7,Mode.EXTEND)
        with self.assertRaises(RuntimeError):
            self.backend.forward(self.q[:7].flatten(1),self.k[:7],self.v[:7],self.layer,batch)
        for mode in (Mode.MIXED,Mode.TARGET_VERIFY,Mode.DRAFT_EXTEND,Mode.DLLM_EXTEND,Mode.SPLIT_PREFILL):
            with self.assertRaises(NotImplementedError):
                self.backend.init_forward_metadata(self.batch(0,7,mode))
        self.assertEqual(self.pool.writes,0)

    def test_unsupported_layer_cannot_write_cache(self):
        batch=self.batch(0,7,Mode.EXTEND)
        self.backend.init_forward_metadata(batch)
        self.layer.logit_cap=30
        with self.assertRaises(NotImplementedError):
            self.backend.forward(self.q[:7].flatten(1),self.k[:7],self.v[:7],self.layer,batch)
        self.assertEqual(self.pool.writes,0)

    def test_native_write_locations_must_match_logical_mapping(self):
        batch=self.batch(0,7,Mode.EXTEND)
        batch.out_cache_loc=batch.out_cache_loc.flip(0)
        self.backend.init_forward_metadata(batch)
        with self.assertRaisesRegex(ValueError,'disagree'):
            self.backend.forward(self.q[:7].flatten(1),self.k[:7],self.v[:7],self.layer,batch)
        self.assertEqual(self.pool.writes,0)

    def test_explicit_selector_preserves_normal_defaults(self):
        calls=[]
        def native(args):
            calls.append(args.attention_backend)
            args.attention_backend=args.prefill_attention_backend=args.decode_attention_backend='ascend'
            args.page_size=128
            args.disable_custom_all_reduce=True
        wrapped=wrap_npu_defaults(native)
        normal=NS(attention_backend='ascend',prefill_attention_backend=None,decode_attention_backend=None)
        wrapped(normal)
        self.assertEqual(normal.attention_backend,'ascend')
        selected=NS(attention_backend=BACKEND_NAME,prefill_attention_backend=None,decode_attention_backend=None)
        wrapped(selected)
        self.assertEqual(selected.attention_backend,BACKEND_NAME)
        self.assertEqual(selected.prefill_attention_backend,BACKEND_NAME)
        self.assertEqual(selected.decode_attention_backend,BACKEND_NAME)
        self.assertEqual(selected.page_size,128)
        self.assertTrue(selected.disable_custom_all_reduce)
        selected.decode_attention_backend='ascend'
        with self.assertRaises(ValueError):wrapped(selected)
        self.assertEqual(len(calls),2)


if __name__=='__main__':unittest.main()
