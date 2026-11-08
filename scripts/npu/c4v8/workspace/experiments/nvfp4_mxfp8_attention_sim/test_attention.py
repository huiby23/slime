import math
import unittest
import torch
from attention import AttentionConfig,attention,mxfp8_tree_reduce,vector_round,online_softmax_step
from test_mxfp8 import scalar_row


def dense(q,k,v,causal=True):
    repeat=q.shape[1]//k.shape[1]
    scores=q@k.repeat_interleave(repeat,1).transpose(-1,-2)/math.sqrt(q.shape[-1])
    if causal:scores=scores.masked_fill(torch.ones(scores.shape[-2:],dtype=torch.bool).triu(1),-math.inf)
    return scores.softmax(-1)@v.repeat_interleave(repeat,1)


def inputs(n=19,d=35):
    g=torch.Generator().manual_seed(1208)
    return tuple(torch.randn((1,h,n,d),generator=g) for h in (4,2,2))


def configurations(qb=7,kb=32):
    return [AttentionConfig(qb,kb,p,reduction='mixed_fp32_reduction') for p in ('fp32','bf16','mxfp8')]+[
        AttentionConfig(qb,kb,'mxfp8',reduction='mxfp8_tree_reduction')]


class AttentionTests(unittest.TestCase):
    def test_unquantized_output_and_gradient_oracle(self):
        first=tuple(x.requires_grad_() for x in inputs(n=11,d=9))
        second=tuple(x.detach().clone().requires_grad_() for x in first)
        actual=attention(*first,AttentionConfig(5,7,'fp32',False,reduction='mixed_fp32_reduction'))
        expected=dense(*second)
        torch.testing.assert_close(actual,expected,atol=6e-7,rtol=4e-6)
        actual.square().sum().backward();expected.square().sum().backward()
        for a,b in zip(first,second):torch.testing.assert_close(a.grad,b.grad,atol=4e-6,rtol=4e-5)

    def test_future_perturbation_does_not_leak_through_scale(self):
        q,k,v=inputs(n=37)
        kk,vv=k.clone(),v.clone();kk[:,:,13:]*=4;vv[:,:,13:]+=100
        for config in configurations():
            with self.subTest(config=config):
                a=attention(q,k,v,config);b=attention(q,kk,vv,config)
                torch.testing.assert_close(a[:,:,:13],b[:,:,:13],atol=0,rtol=0)

    def test_prefix_loss_has_zero_future_gradients(self):
        for config in configurations():
            q,k,v=(x.requires_grad_() for x in inputs(n=19))
            attention(q,k,v,config)[:,:,:9].square().sum().backward()
            for x in (k,v):torch.testing.assert_close(x.grad[:,:,9:],torch.zeros_like(x.grad[:,:,9:]),atol=0,rtol=0)

    def test_prefill_decode_same_tile_policy(self):
        q,k,v=inputs(n=19,d=17)
        for config in configurations(qb=1,kb=8):
            full=attention(q,k,v,config)
            pieces=[attention(q[:,:,t:t+1],k[:,:,:t+1],v[:,:,:t+1],config,query_offset=t) for t in range(19)]
            torch.testing.assert_close(full,torch.cat(pieces,2),atol=0,rtol=0)

    def test_backward_finite_nonzero_and_zero_v(self):
        for config in configurations():
            for dtype in (torch.float32,torch.bfloat16):
                values=tuple(x.to(dtype).requires_grad_() for x in inputs(n=11,d=17))
                out=attention(*values,config);out.float().square().mean().backward()
                for x in values:
                    self.assertTrue(bool(torch.isfinite(x.grad).all()));self.assertGreater(x.grad.float().norm().item(),0)
                q,k,v=values
                torch.testing.assert_close(attention(q,k,torch.zeros_like(v),config),torch.zeros_like(q),atol=0,rtol=0)

    def test_mxfp8_and_bf16_vector_boundaries_are_exercised(self):
        x=inputs();fp32,bf16,mx,_=[attention(*x,c) for c in configurations()]
        self.assertGreater((fp32-bf16).abs().max().item(),0)
        self.assertGreater((fp32-mx).abs().max().item(),0)

    def test_pairwise_tree_against_independent_scalar_levels(self):
        row=[1.,.0625,.0625,.0625,.0625,.125,.00390625]
        expected,_=scalar_row(row)
        while len(expected)>1:
            if len(expected)%2:expected.append(0.)
            expected,_=scalar_row([a+b for a,b in zip(expected[::2],expected[1::2])])
        actual=mxfp8_tree_reduce(torch.tensor([row]),'sum')
        self.assertEqual(actual.item(),expected[0])
        # This chosen order loses each small contribution at its own tree level.
        row=torch.tensor([[1.,.0625,.0625,.0625]])
        tree=mxfp8_tree_reduce(row,'sum')
        terminal=vector_round(row.sum(-1,keepdim=True),'mxfp8')
        self.assertNotEqual(tree.item(),terminal.item())

    def test_max_tree_negative_padding_and_structural_mask(self):
        x=torch.tensor([[-5.,-3.,-7.,-math.inf,-4.]])
        self.assertEqual(mxfp8_tree_reduce(x,'max').item(),-3.)
        all_masked=mxfp8_tree_reduce(torch.full((1,5),-math.inf),'max')
        self.assertTrue(bool(torch.isneginf(all_masked).all()))

    def test_reduction_policy_is_explicit(self):
        mixed=AttentionConfig(vector='mxfp8',reduction='mixed_fp32_reduction').metadata()
        tree=AttentionConfig(vector='mxfp8',reduction='mxfp8_tree_reduction').metadata()
        self.assertIn('denominator sum and running update',mixed['fp32_exceptions'])
        self.assertNotIn('denominator sum and running update',tree['fp32_exceptions'])
        self.assertFalse(tree['native_bit_exact']);self.assertFalse(tree['fully_8bit_vector_arithmetic'])
        with self.assertRaises(ValueError):AttentionConfig(vector='bf16',reduction='mxfp8_tree_reduction')

    def test_tree_lengths_and_signed_scale_against_scalar_oracle(self):
        for length in (1,2,3,31,32,33,64):
            for operation in ('sum','max'):
                row=[(-1 if i%3 else 1)*(i+1)/13 for i in range(length)]
                expected,_=scalar_row(row)
                while len(expected)>1:
                    if len(expected)%2:expected.append(0. if operation=='sum' else -math.inf)
                    expected=[a+b if operation=='sum' else max(a,b) for a,b in zip(expected[::2],expected[1::2])]
                    expected,_=scalar_row(expected)
                self.assertEqual(mxfp8_tree_reduce(torch.tensor([row]),operation).item(),expected[0])

    def test_tree_max_scale_regrouping_underflow_is_not_hidden(self):
        scores=torch.tensor([[12288.]*32+[-1e9]*32])
        scores=vector_round(scores,'mxfp8')
        self.assertEqual(mxfp8_tree_reduce(scores,'max').item(),16384.)
        with self.assertRaisesRegex(FloatingPointError,'denominator'):
            online_softmax_step(scores,torch.full((1,1),-math.inf),torch.zeros(1,1),AttentionConfig())
        mixed=AttentionConfig(reduction='mixed_fp32_reduction')
        _,denominator,_,_=online_softmax_step(scores,torch.full((1,1),-math.inf),torch.zeros(1,1),mixed)
        self.assertGreater(denominator.item(),0)

    def test_online_states_and_all_masked_row(self):
        config=AttentionConfig()
        maximum=torch.full((3,1),-math.inf);denominator=torch.zeros(3,1)
        tiles=[torch.tensor([[-3.,-2.,-1.],[0.,.01,.02],[-20.,-21.,-22.]]),
               torch.tensor([[-4.,-5.,-6.],[.03,.03,.02],[-19.,-19.,-18.]])]
        for tile in tiles:
            new,denominator,alpha,p=online_softmax_step(vector_round(tile,'mxfp8'),maximum,denominator,config)
            self.assertTrue(bool((new>=maximum).all()))
            self.assertTrue(bool(((alpha>=0)&(alpha<=1)).all()))
            self.assertTrue(bool((denominator>0).all()));self.assertTrue(bool(torch.isfinite(p).all()))
            maximum=new
        with self.assertRaisesRegex(FloatingPointError,'all-masked'):
            online_softmax_step(torch.full((1,3),-math.inf),torch.full((1,1),-math.inf),torch.zeros(1,1),config)

    def test_arithmetic_negative_infinity_is_not_a_mask(self):
        for policy in ('fp32','bf16','mxfp8'):
            with self.assertRaises((ValueError,FloatingPointError)):
                vector_round(torch.tensor([-math.inf]),policy)
        q=torch.full((1,1,1,1),1e30);k=-q;v=torch.ones_like(q)
        with self.assertRaisesRegex(FloatingPointError,'unmasked QK'):
            attention(q,k,v,AttentionConfig(quantize_operands=False))
        with self.assertRaisesRegex(FloatingPointError,'subtraction overflow'):
            online_softmax_step(torch.tensor([[-3e38]]),torch.tensor([[3e38]]),torch.ones(1,1),
                AttentionConfig(vector='fp32',reduction='mixed_fp32_reduction'))


if __name__=='__main__':unittest.main()
