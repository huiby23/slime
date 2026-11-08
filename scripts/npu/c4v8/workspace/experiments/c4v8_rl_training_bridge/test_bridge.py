import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile
import json
import hashlib

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'nvfp4_mxfp8_attention_sim'))
import torch
from attention import attention, AttentionConfig
from bridge import recomputed_attention
from layouts import attention_sbhd, attention_thd
from megatron_spec import C4V8CoreAttention, replace_core_attention, configure_npu_matmul, record_trainer_policy


def inputs(n=13, dtype=torch.float32):
    g = torch.Generator().manual_seed(9721)
    return tuple(torch.randn(1, h, n, 17, generator=g).to(dtype) for h in (4, 2, 2))


def arms():
    return [AttentionConfig(query_block=4, key_block=8),
            AttentionConfig(query_block=4, key_block=8, reduction='mixed_fp32_reduction'),
            AttentionConfig(query_block=4, key_block=8, vector='fp32', quantize_operands=False,
                            reduction='mixed_fp32_reduction')]


class BridgeTests(unittest.TestCase):
    def test_device_core_callback_on_cpu(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'c4v8_device_core'))
        from device_attention import attention as device_attention
        for dtype in (torch.float32, torch.bfloat16):
            a = tuple(x.requires_grad_() for x in inputs(dtype=dtype))
            b = tuple(x.detach().clone().requires_grad_() for x in a)
            config = arms()[0]
            x = attention(*a, config)
            y = recomputed_attention(*b, kernel=device_attention, config=config)
            torch.testing.assert_close(x, y, atol=0, rtol=0)
            x.square().sum().backward(); y.square().sum().backward()
            for aa, bb in zip(a, b):
                torch.testing.assert_close(aa.grad, bb.grad, atol=4e-6, rtol=4e-5)

    def test_forward_and_gradients_match_reference(self):
        for dtype in (torch.float32, torch.bfloat16):
            for config in arms():
                with self.subTest(dtype=dtype, config=config):
                    a = tuple(x.requires_grad_() for x in inputs(dtype=dtype))
                    b = tuple(x.detach().clone().requires_grad_() for x in a)
                    expected = attention(*a, config)
                    actual = recomputed_attention(*b, kernel=attention, config=config)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    go = torch.randn(actual.shape, generator=torch.Generator().manual_seed(87)).to(dtype)
                    expected.backward(go); actual.backward(go)
                    for original, replay in zip(a, b):
                        torch.testing.assert_close(replay.grad, original.grad, atol=3e-6, rtol=4e-5)

    def test_partial_grad_requirements_and_query_offset(self):
        for requirements in ((True, False, False), (False, True, True), (True, True, False)):
            a = list(inputs(n=16))
            a[0] = a[0][:, :, -5:]
            a = tuple(x.requires_grad_(r) for x, r in zip(a, requirements))
            b = tuple(x.detach().clone().requires_grad_(r) for x, r in zip(a, requirements))
            cfg = arms()[0]
            x = attention(*a, cfg, query_offset=11)
            y = recomputed_attention(*b, kernel=attention, config=cfg, query_offset=11)
            torch.testing.assert_close(x, y, atol=0, rtol=0)
            x.square().sum().backward(); y.square().sum().backward()
            for aa, bb, needed in zip(a, b, requirements):
                if needed: torch.testing.assert_close(aa.grad, bb.grad, atol=3e-6, rtol=4e-5)
                else: self.assertIsNone(bb.grad)

    def test_only_linear_inputs_saved_in_forward(self):
        data = tuple(x.requires_grad_() for x in inputs(n=33))
        stored = []
        def pack(x): stored.append(tuple(x.shape)); return x
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x:x):
            recomputed_attention(*data, kernel=attention, config=arms()[0])
        self.assertEqual(stored, [tuple(x.shape) for x in data])

    def test_rng_unchanged(self):
        data = tuple(x.requires_grad_() for x in inputs())
        state = torch.random.get_rng_state().clone()
        recomputed_attention(*data, kernel=attention, config=arms()[0]).square().sum().backward()
        torch.testing.assert_close(state, torch.random.get_rng_state(), atol=0, rtol=0)

    def test_noncausal_reference_and_gradient(self):
        config = AttentionConfig(query_block=4, key_block=8, causal=False)
        a = tuple(x.requires_grad_() for x in inputs())
        b = tuple(x.detach().clone().requires_grad_() for x in a)
        x = attention(*a, config)
        y = recomputed_attention(*b, kernel=attention, config=config)
        torch.testing.assert_close(x,y,atol=0,rtol=0)
        x.square().sum().backward(); y.square().sum().backward()
        for aa,bb in zip(a,b): torch.testing.assert_close(aa.grad,bb.grad,atol=4e-6,rtol=4e-5)


class LayoutTests(unittest.TestCase):
    def test_explicit_npu_hf32_control_and_missing_control_rejected(self):
        fake=SimpleNamespace(matmul=SimpleNamespace(allow_hf32=True))
        with patch.object(torch,'npu',fake,create=True):
            report=configure_npu_matmul()
            self.assertTrue(report['allow_hf32_before'])
            self.assertFalse(fake.matmul.allow_hf32)
        with patch.object(torch,'npu',SimpleNamespace(),create=True):
            with self.assertRaisesRegex(RuntimeError,'control'):configure_npu_matmul()

    def test_trainer_policy_audit_rejects_changed_policy(self):
        policy={'query_block':8}
        digest=hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{
            'HBY_C4V8_TRAIN_AUDIT_DIR':tmp,'HBY_C4V8_POLICY_SHA256':digest}):
            record_trainer_policy(policy,{'available':False,'allow_hf32':None})
            record_trainer_policy(policy,{'available':False,'allow_hf32':None})
            self.assertEqual(len(list(Path(tmp).glob('*.json'))),1)
            with self.assertRaisesRegex(ValueError,'SHA'):record_trainer_policy({'query_block':16},{'available':False})

    def test_sbhd_gqa_forward_and_gradient(self):
        a = tuple(x.permute(2,0,1,3).contiguous().requires_grad_() for x in inputs())
        b = tuple(x.detach().clone().requires_grad_() for x in a)
        x = attention_sbhd(*a,kernel=attention,config=arms()[0],recompute=False)
        y = attention_sbhd(*b,kernel=attention,config=arms()[0])
        self.assertEqual(y.shape,(13,1,68))
        torch.testing.assert_close(x,y,atol=0,rtol=0)
        x.square().sum().backward();y.square().sum().backward()
        for aa,bb in zip(a,b):torch.testing.assert_close(aa.grad,bb.grad,atol=4e-6,rtol=4e-5)

    def test_packed_isolation_outputs_and_gradients(self):
        values=tuple(x.squeeze(0).transpose(0,1).contiguous() for x in inputs(n=12))
        a=tuple(x.requires_grad_() for x in values)
        b=tuple(x.detach().clone().requires_grad_() for x in a)
        cu=torch.tensor([0,5,12],dtype=torch.int32)
        actual=attention_thd(*a,cu_seqlens_q=cu,cu_seqlens_kv=cu,kernel=attention,config=arms()[0])
        expected=[]
        for start,end in ((0,5),(5,12)):
            chunk=[x[start:end].transpose(0,1).unsqueeze(0) for x in b]
            expected.append(attention(*chunk,arms()[0]).squeeze(0).transpose(0,1))
        expected=torch.cat(expected,0)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        actual[:5].square().sum().backward();expected[:5].square().sum().backward()
        for aa,bb in zip(a,b):
            torch.testing.assert_close(aa.grad,bb.grad,atol=4e-6,rtol=4e-5)
            self.assertEqual(torch.count_nonzero(aa.grad[5:]).item(),0)

    def test_padded_pack_exact_zero_padding_gradient(self):
        a=tuple(x.squeeze(0).transpose(0,1).contiguous().requires_grad_() for x in inputs(n=14))
        cu=torch.tensor([0,4,11],dtype=torch.int32)
        padded=torch.tensor([0,6,14],dtype=torch.int32)
        out=attention_thd(*a,cu_seqlens_q=cu,cu_seqlens_kv=cu,
            cu_seqlens_q_padded=padded,cu_seqlens_kv_padded=padded,kernel=attention,config=arms()[0])
        self.assertEqual(torch.count_nonzero(out[[4,5,13]]).item(),0)
        out.square().sum().backward()
        for x in a:self.assertEqual(torch.count_nonzero(x.grad[[4,5,13]]).item(),0)

    def test_invalid_packed_lengths_rejected(self):
        a=tuple(x.squeeze(0).transpose(0,1).contiguous() for x in inputs(n=12))
        bads=[([0,5,11],[0,5,12]),([0,5,12],[0,4,12]),([1,5,12],[0,5,12])]
        for cq,ck in bads:
            with self.assertRaises(ValueError):
                attention_thd(*a,cu_seqlens_q=torch.tensor(cq,dtype=torch.int32),
                    cu_seqlens_kv=torch.tensor(ck,dtype=torch.int32),kernel=attention,config=arms()[0])

    def test_megatron_forward_shape_and_guard(self):
        c=SimpleNamespace(context_parallel_size=1,attention_dropout=0.,kv_channels=17)
        module=C4V8CoreAttention(c,1,'causal',kernel=attention,precision_config=arms()[0])
        x=tuple(v.permute(2,0,1,3).contiguous() for v in inputs())
        self.assertEqual(module(*x).shape,(13,1,68))
        ordinary=torch.ones(1,1,13,13,dtype=torch.bool).triu(1)
        torch.testing.assert_close(module(*x,ordinary),module(*x),atol=0,rtol=0)
        bad=ordinary.clone();bad[:,:,3,1]=True
        with self.assertRaises(ValueError):module(*x,bad)
        with self.assertRaises(ValueError):C4V8CoreAttention(
            SimpleNamespace(context_parallel_size=2),1,'causal',kernel=attention,precision_config=arms()[0])

    def test_spec_replacement_preserves_original_and_backbone(self):
        class FakeSpec:
            def __init__(self,module,params):self.module=module;self.params=params
        sub=SimpleNamespace(core_attention='old',linear_qkv='bf16_qkv',linear_proj='bf16_o')
        layer=SimpleNamespace(submodules=SimpleNamespace(self_attention=SimpleNamespace(submodules=sub),mlp='bf16_moe'))
        original=SimpleNamespace(layer_specs=[layer],layer_norm='original_norm')
        changed=replace_core_attention(original,kernel=attention,precision_config=arms()[0],module_spec_type=FakeSpec)
        self.assertEqual(original.layer_specs[0].submodules.self_attention.submodules.core_attention,'old')
        self.assertIs(changed.layer_specs[0].submodules.self_attention.submodules.core_attention.module,C4V8CoreAttention)
        self.assertEqual(changed.layer_specs[0].submodules.mlp,'bf16_moe')
        self.assertEqual(changed.layer_specs[0].submodules.self_attention.submodules.linear_qkv,'bf16_qkv')


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
