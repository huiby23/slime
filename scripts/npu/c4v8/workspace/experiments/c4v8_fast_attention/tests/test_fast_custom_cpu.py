"""CPU oracle injection exercises the actual custom Functions, NOT CPU fallback.

The optional attention entry is the exact original AST with only three hardware
dispatch/guard statements adapted for CPU execution. _Pre/_Post/_MX are always
the unmodified imported classes. This proves graph/shape/STE semantics with an
oracle forward API; it does not test Triton kernels or Ascend numerics.
"""
import ast
import copy
import hashlib
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS=ROOT.parent
sys.path[:0]=[str(ROOT),str(EXPERIMENTS/'c4v8_device_core'),str(EXPERIMENTS/'c4v8_rl_training_bridge')]
import torch
import fast_attention as fast
import device_attention as reference
import device_quant as quant
from exact_fp32 import divide_positive
from bridge import recomputed_attention
from layouts import attention_thd,attention_sbhd


def vr(x,masked=False):
    return quant.mxfp8_qdq(x,allow_masked_negative_infinity=masked)


def pre_oracle(scores,maximum,denominator):
    scores=vr(scores,True)
    return reference.online_softmax_step(scores,maximum,denominator,
        reference.AttentionConfig(query_block=4,key_block=8))


def post_oracle(old,part,alpha):
    return vr(vr(old*alpha)+vr(part))


class OracleAPI:
    def __init__(self):
        self.counts=dict(mx=0,pre=0,post=0,cube=0)
        self.inject_bad_stage=None

    def qdq(self,x,*,allow_mask=False,check=True):
        self.counts['mx']+=1
        if torch.is_grad_enabled() and x.requires_grad:
            raise AssertionError('Forward-only API called with a tracked tensor')
        y=vr(x,allow_mask)
        return y,None,None,torch.zeros(x.numel()//x.shape[-1],dtype=torch.int32)

    def pre(self,scores,maximum,denominator,*,check=True):
        self.counts['pre']+=1
        if torch.is_grad_enabled() and any(x.requires_grad for x in (scores,maximum,denominator)):
            raise AssertionError('Forward-only pre API has a live gradient graph')
        out=pre_oracle(scores,maximum,denominator)
        flags=torch.zeros(maximum.numel(),dtype=torch.int32)
        if self.inject_bad_stage=='pre':flags[0]=1
        return (*out,flags)

    def post(self,old,part,alpha,*,check=True):
        self.counts['post']+=1
        if torch.is_grad_enabled() and any(x.requires_grad for x in (old,part,alpha)):
            raise AssertionError('Forward-only post API has a live gradient graph')
        flags=torch.zeros(alpha.numel(),dtype=torch.int32)
        if self.inject_bad_stage=='post':flags[0]=1
        return post_oracle(old,part,alpha),flags

    def cube(self,x,*,check=True):
        self.counts['cube']+=1
        return quant.nvfp4_qdq(x,check=check)

    def normalize(self,numerator,denominator,*,check=True):
        if torch.is_grad_enabled() and any(x.requires_grad for x in (numerator,denominator)):
            raise AssertionError('Forward-only normalization API has a live graph')
        divided=quant._copy_sign(divide_positive(numerator.abs(),denominator),numerator)
        flags=torch.zeros(denominator.numel(),dtype=torch.int32)
        return vr(divided),flags

    def api(self):return self.qdq,self.pre,self.post,self.cube


def cpu_dispatch_entry():
    """Change only device dispatch. No copied loop/gradient implementation."""
    source=ROOT/'fast_attention.py'
    module=ast.parse(source.read_text())
    node=copy.deepcopy(next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='attention'))
    removed=[]
    new_body=[]
    for statement in node.body:
        if isinstance(statement,ast.If):
            condition=ast.unparse(statement.test)
            if condition in ("q.device.type == 'cpu'",'torch.npu.matmul.allow_hf32'):
                removed.append(condition);continue
            if condition.startswith("q.device.type not in ('npu', 'privateuseone')"):
                statement.test.values[0].comparators[0].elts.append(ast.Constant(value='cpu'))
        new_body.append(statement)
    assert sorted(removed)==["q.device.type == 'cpu'",'torch.npu.matmul.allow_hf32']
    node.body=new_body
    code=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    namespace=dict(fast.__dict__)
    # Preserve late API injection; a copied namespace must not accidentally
    # freeze the original hardware imports before setUp patches them.
    namespace['_apis']=lambda:fast._apis()
    namespace['_normalize_api']=lambda:fast._normalize_api()
    exec(compile(code,str(source)+':cpu_device_dispatch_only','exec'),namespace)
    return namespace['attention']


CPU_ENTRY=cpu_dispatch_entry()


def clone_inputs(values,needs):
    return tuple(x.detach().clone().requires_grad_(needed) for x,needed in zip(values,needs))


def finite_loss(outputs):
    return sum((out*torch.linspace(0.1,0.9,out.numel()).reshape(out.shape)).sum() for out in outputs)


class CustomFunctions(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.api=OracleAPI()
        self.context=patch.object(fast,'_apis',self.api.api)
        self.context.start()
        self.normalize_context=patch.object(fast,'_normalize_api',lambda:self.api.normalize)
        self.normalize_context.start()
    def tearDown(self):self.normalize_context.stop();self.context.stop()

    def states(self,initial=False):
        g=torch.Generator().manual_seed(1728)
        scores=torch.randn(2,3,5,13,generator=g)
        scores[...,9:]=-float('inf')
        maximum=torch.full((2,3,5,1),-float('inf')) if initial else torch.randn(2,3,5,1,generator=g)
        denominator=torch.zeros_like(maximum) if initial else torch.rand(2,3,5,1,generator=g)+1
        return scores,maximum,denominator

    def test_pre_actual_custom_forward_and_all_ste_inputs(self):
        for initial in (False,True):
            for needs in itertools.product((False,True),repeat=3):
                if not any(needs):continue
                with self.subTest(initial=initial,needs=needs):
                    a=clone_inputs(self.states(initial),needs);b=clone_inputs(self.states(initial),needs)
                    actual=fast._Pre.apply(*a);expected=pre_oracle(*b)
                    self.assertFalse(actual[-1].requires_grad)
                    for x,y in zip(actual,expected):torch.testing.assert_close(x,y,atol=0,rtol=0)
                    # Differentiating an initial -inf sentinel is meaningless;
                    # production initial maximum never requires grad. Compare
                    # it too, but require exact matching nonfinite classification.
                    finite_loss(actual[:-1]).backward();finite_loss(expected).backward()
                    for x,y,need in zip(a,b,needs):
                        if need:torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-6,equal_nan=True)
                        else:self.assertIsNone(x.grad)
                    if needs[0]:self.assertEqual(torch.count_nonzero(a[0].grad[...,9:]).item(),0)
        self.assertGreater(self.api.counts['pre'],0)

    def test_pre_each_output_and_partial_gradient_path(self):
        for selected in range(4):
            a=clone_inputs(self.states(),(True,True,True));b=clone_inputs(self.states(),(True,True,True))
            x=fast._Pre.apply(*a);y=pre_oracle(*b)
            x[selected].sum().backward();y[selected].sum().backward()
            for aa,bb in zip(a,b):
                expected=torch.zeros_like(bb) if bb.grad is None else bb.grad
                torch.testing.assert_close(aa.grad,expected,atol=2e-6,rtol=2e-6)

    def test_post_actual_custom_forward_and_partial_requires_grad(self):
        g=torch.Generator().manual_seed(82)
        data=(torch.randn(2,4,7,17,generator=g),torch.randn(2,4,7,17,generator=g),torch.rand(2,4,7,1,generator=g))
        for needs in itertools.product((False,True),repeat=3):
            if not any(needs):continue
            a=clone_inputs(data,needs);b=clone_inputs(data,needs)
            x,flags=fast._Post.apply(*a);y=post_oracle(*b)
            self.assertFalse(flags.requires_grad)
            torch.testing.assert_close(x,y,atol=0,rtol=0)
            x.square().sum().backward();y.square().sum().backward()
            for aa,bb,needed in zip(a,b,needs):
                if needed:torch.testing.assert_close(aa.grad,bb.grad,atol=0,rtol=0)

    def test_mx_mask_ste_and_noncontiguous_input(self):
        x=torch.randn(3,7).T.contiguous().T.requires_grad_(True)
        x.data[0,1]=-float('inf')
        a=fast._MX.apply(x,True)
        b=vr(x.detach(),True)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        a.sum().backward()
        expected=torch.ones_like(x);expected[0,1]=0
        torch.testing.assert_close(x.grad,expected,atol=0,rtol=0)

    def test_actual_normalize_forward_and_broadcast_ste(self):
        g=torch.Generator().manual_seed(771)
        numerator=torch.randn(2,3,5,17,generator=g)
        numerator[0,0,0,0]=-0.0
        denominator=torch.rand(2,3,5,1,generator=g)+0.125
        for needs in ((True,True),(False,True),(True,False)):
            a=clone_inputs((numerator,denominator),needs);b=clone_inputs((numerator,denominator),needs)
            actual,flags=fast._Normalize.apply(*a)
            expected=vr(b[0]/b[1])
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            self.assertFalse(flags.requires_grad)
            weights=torch.randn(actual.shape,generator=g)
            actual.backward(weights);expected.backward(weights)
            for aa,bb,need in zip(a,b,needs):
                if need:torch.testing.assert_close(aa.grad,bb.grad,atol=3e-5,rtol=3e-6)

    def test_score_error_flags_keep_one_scalar_per_tile(self):
        data=self.inputs(n=33,batch=1)
        seen=[];original=torch.cat
        def track(values,*args,**kwargs):
            if all(x.dtype==torch.int32 for x in values):seen.extend(x.numel() for x in values)
            return original(values,*args,**kwargs)
        with patch.object(torch,'cat',track):
            CPU_ENTRY(*data,fast.AttentionConfig(query_block=4,key_block=8))
        self.assertTrue(seen)
        self.assertEqual(max(seen),1,'Error flags must not retain every score element')

    def inputs(self,n=19,d=17,dtype=torch.float32,heads=(4,2,2),batch=2):
        generator=torch.Generator().manual_seed(99)
        return tuple(torch.randn(batch,h,n,d,generator=generator).to(dtype) for h in heads)

    def test_real_attention_body_gqa_offsets_and_gradients(self):
        for dtype in (torch.float32,torch.bfloat16):
            for offset in (0,7):
                for needs in ((True,True,True),(True,False,False),(False,True,True)):
                    data=list(self.inputs(dtype=dtype));data[0]=data[0][:,:,offset:]
                    a=clone_inputs(data,needs);b=clone_inputs(data,needs)
                    cfg=fast.AttentionConfig(query_block=4,key_block=8)
                    x=CPU_ENTRY(*a,cfg,query_offset=offset);y=reference.attention(*b,cfg,query_offset=offset)
                    torch.testing.assert_close(x,y,atol=0,rtol=0)
                    x.float().square().sum().backward();y.float().square().sum().backward()
                    for aa,bb,need in zip(a,b,needs):
                        if need:torch.testing.assert_close(aa.grad,bb.grad,atol=4e-6,rtol=4e-5)
        self.assertGreater(self.api.counts['pre'],0);self.assertGreater(self.api.counts['post'],0)

    def test_query_tile_recompute_matches_custom_graph_and_saves_only_qkv(self):
        for dtype in (torch.float32,torch.bfloat16):
            a=clone_inputs(self.inputs(n=33,dtype=dtype,batch=1),(True,True,True))
            b=clone_inputs(a,(True,True,True));cfg=fast.AttentionConfig(query_block=4,key_block=8)
            saved=[]
            def pack(x):saved.append(tuple(x.shape));return x
            with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):
                x=recomputed_attention(*a,kernel=CPU_ENTRY,config=cfg)
            self.assertEqual(saved,[tuple(t.shape) for t in a])
            y=CPU_ENTRY(*b,cfg)
            torch.testing.assert_close(x,y,atol=0,rtol=0)
            x.float().square().sum().backward();y.float().square().sum().backward()
            for aa,bb in zip(a,b):torch.testing.assert_close(aa.grad,bb.grad,atol=4e-6,rtol=4e-5)

    def test_actual_custom_functions_inside_packed_gqa_replay(self):
        values=tuple(x.squeeze(0).transpose(0,1).contiguous() for x in self.inputs(n=21,batch=1))
        a=clone_inputs(values,(True,True,True));b=clone_inputs(values,(True,True,True))
        cfg=fast.AttentionConfig(query_block=4,key_block=8)
        cu=torch.tensor([0,5,17],dtype=torch.int32);padded=torch.tensor([0,8,21],dtype=torch.int32)
        options=dict(cu_seqlens_q=cu,cu_seqlens_kv=cu,cu_seqlens_q_padded=padded,cu_seqlens_kv_padded=padded,config=cfg)
        x=attention_thd(*a,kernel=CPU_ENTRY,**options)
        y=attention_thd(*b,kernel=reference.attention,**options)
        torch.testing.assert_close(x,y,atol=0,rtol=0)
        x[:5].square().sum().backward();y[:5].square().sum().backward()
        for aa,bb in zip(a,b):
            torch.testing.assert_close(aa.grad,bb.grad,atol=4e-6,rtol=4e-5)
            self.assertEqual(torch.count_nonzero(aa.grad[5:]).item(),0)
        self.assertEqual(torch.count_nonzero(x[[5,6,7,20]]).item(),0)

    def test_actual_flags_reject_invalid_stage_before_output_acceptance(self):
        data=self.inputs(n=9,batch=1);cfg=fast.AttentionConfig(query_block=4,key_block=8)
        for stage in ('pre','post'):
            self.api.inject_bad_stage=stage
            with self.assertRaisesRegex(FloatingPointError,'invalid'):CPU_ENTRY(*data,cfg)
        self.api.inject_bad_stage=None

    def test_float32_only_state_api_and_no_higher_order_claim(self):
        a=clone_inputs(self.states(),(True,True,True))
        out=fast._Pre.apply(*a)
        gradients=torch.autograd.grad(finite_loss(out[:-1]),a,create_graph=True)
        self.assertTrue(all(not x.requires_grad for x in gradients))


if __name__=='__main__':unittest.main(verbosity=2)
