import math
from pathlib import Path
import sys
import unittest
import torch

REFERENCE = Path(__file__).resolve().parent.parent/'nvfp4_mxfp8_attention_sim'
sys.path.insert(0,str(REFERENCE))
import nvfp4 as cpu_c4
import mxfp8 as cpu_v8
import attention as cpu_attn
from device_quant import (round_e4m3fn,e4m3fn_bytes,round_e2m1,nvfp4_values,
                          mxfp8_values,nvfp4_qdq,mxfp8_qdq)
from device_attention import attention,AttentionConfig,operation_estimate
from exact_fp32 import divide_positive


def exact(a,b):
    torch.testing.assert_close(a,b,atol=0,rtol=0)


class CoreTests(unittest.TestCase):
    def test_exact_scale_division_over_fp32_exponent_range(self):
        g=torch.Generator().manual_seed(1339)
        for numerator in (1.,2688.,None):
            den=torch.randint(1,0x7f800000,(200000,),generator=g,dtype=torch.int32).view(torch.float32)
            num=(torch.randint(1,0x7f800000,(200000,),generator=g,dtype=torch.int32).view(torch.float32)
                 if numerator is None else torch.full_like(den,numerator))
            a=divide_positive(num,den);b=num/den
            self.assertTrue(torch.equal(a.view(torch.int32),b.view(torch.int32)))

    def test_e4m3_all_codes_and_midpoint_neighbors(self):
        codes=torch.arange(256,dtype=torch.uint8)
        values=codes.view(torch.float8_e4m3fn).float()
        finite=torch.isfinite(values);values=values[finite];codes=codes[finite]
        self.assertTrue(torch.equal(e4m3fn_bytes(values),codes))
        positive=values[values>=0].unique().sort().values
        mids=(positive[:-1]+positive[1:])/2
        x=torch.cat([values,mids,-mids,torch.nextafter(mids,torch.zeros_like(mids)),
                     torch.nextafter(mids,torch.full_like(mids,math.inf))])
        expected=x.clamp(-448,448).to(torch.float8_e4m3fn)
        exact(round_e4m3fn(x),expected.float())
        self.assertTrue(torch.equal(e4m3fn_bytes(x),expected.view(torch.uint8)))

    def test_e2m1_midpoints_saturation_and_signed_zero(self):
        x=torch.tensor([0.,-0.,.25,.75,1.25,1.75,2.5,3.5,5.,7.,100.])
        x=torch.cat([x,-x,torch.nextafter(x,torch.zeros_like(x)),
                     torch.nextafter(x,torch.full_like(x,math.inf))])
        expected=cpu_c4.round_e2m1(x)
        actual,codes=round_e2m1(x,return_codes=True)
        exact(actual,expected)
        self.assertTrue(torch.equal(torch.signbit(actual),torch.signbit(expected)))
        self.assertTrue(torch.equal(codes,cpu_c4._e2m1_codes(x)))

    def test_nvfp4_random_value_and_byte_parity(self):
        g=torch.Generator().manual_seed(98723)
        for dtype in (torch.float32,torch.bfloat16,torch.float16):
            for n in (1,15,16,17,35,128):
                x=(torch.randn(3,n,generator=g)*7).to(dtype)
                r=cpu_c4.quantize(x);a=nvfp4_values(x,logical_bytes=True)
                exact(a.values,cpu_c4.dequantize(r))
                self.assertTrue(torch.equal(a.payload_bytes,r.codes))
                self.assertTrue(torch.equal(a.scale_bytes,r.block_scale_bytes))
                exact(a.upper_encode,r.global_encode_scale);exact(a.upper_decode,r.global_decode_scale)

    def test_mxfp8_random_value_and_byte_parity(self):
        g=torch.Generator().manual_seed(9823)
        for dtype in (torch.float32,torch.bfloat16,torch.float16):
            for n in (1,31,32,33,67,128):
                x=(torch.randn(3,n,generator=g)*7).to(dtype)
                r=cpu_v8.quantize(x);a=mxfp8_values(x,logical_bytes=True)
                exact(a.values,cpu_v8.dequantize(r))
                self.assertTrue(torch.equal(a.payload_bytes,r.payload_bytes))
                self.assertTrue(torch.equal(a.scale_bytes,r.scale_bytes))

    def test_mxfp8_actual_scale_boundaries(self):
        for exponent in (-127,-120,-40,-1,0,1,20,100,117):
            point=torch.tensor(448.*2.**exponent,dtype=torch.float32)
            vals=torch.stack([torch.nextafter(point,torch.tensor(0.)),point,
                              torch.nextafter(point,torch.tensor(math.inf))]).view(-1,1)
            r=cpu_v8.quantize(vals);a=mxfp8_values(vals,logical_bytes=True)
            exact(a.values,cpu_v8.dequantize(r))
            self.assertTrue(torch.equal(a.scale_bytes,r.scale_bytes))

    def test_zeros_tiny_values_and_extreme_range(self):
        cases=[torch.zeros(2,35),torch.tensor([[0.,-0.,1e-44,1e-40,1e-35]]),
               torch.tensor([[1e-20,1.,1e20]])]
        for x in cases:
            for reference,fn in ((cpu_c4,nvfp4_values),(cpu_v8,mxfp8_values)):
                actual=fn(x);expected=reference.dequantize(reference.quantize(x))
                exact(actual.values,expected)

    def test_ste_and_mask_gradient(self):
        x=torch.randn(2,35,requires_grad=True);grad=torch.randn_like(x)
        for fn in (nvfp4_qdq,mxfp8_qdq):
            out=fn(x);got=torch.autograd.grad(out,x,grad)[0];exact(got,grad)
        z=torch.tensor([[1.,-math.inf,2.]],requires_grad=True)
        out=mxfp8_qdq(z,allow_masked_negative_infinity=True)
        got=torch.autograd.grad(out,z,torch.ones_like(z))[0]
        exact(got,torch.tensor([[1.,0.,1.]]));self.assertTrue(torch.isneginf(out[0,1]))

    def test_nonfinite_never_silently_saturates(self):
        for fn in (nvfp4_values,mxfp8_values):
            for value in (math.inf,-math.inf,math.nan):
                x=torch.tensor([[value,1.]])
                with self.assertRaises(ValueError):fn(x)
                self.assertFalse(bool(torch.isfinite(fn(x,check=False).values).all()))
        with self.assertRaises(ValueError):mxfp8_qdq(torch.tensor([[-math.inf]]))

    def test_attention_values_and_gradients_against_cpu_reference(self):
        g=torch.Generator().manual_seed(7172)
        for dtype in (torch.float32,torch.bfloat16):
            for reduction in ('mixed_fp32_reduction','mxfp8_tree_reduction'):
                xs=tuple(torch.randn(1,h,19,17,generator=g).to(dtype).requires_grad_() for h in (4,2,2))
                ys=tuple(x.detach().clone().requires_grad_() for x in xs)
                a=attention(*xs,AttentionConfig(query_block=5,key_block=8,reduction=reduction))
                b=cpu_attn.attention(*ys,cpu_attn.AttentionConfig(query_block=5,key_block=8,reduction=reduction))
                exact(a,b);a.float().square().sum().backward();b.float().square().sum().backward()
                for x,y in zip(xs,ys):exact(x.grad,y.grad)

    def test_causal_prefix_and_decode_consistency(self):
        g=torch.Generator().manual_seed(717)
        q,k,v=(torch.randn(1,h,17,19,generator=g) for h in (4,2,2))
        config=AttentionConfig(query_block=1,key_block=8)
        full=attention(q,k,v,config)
        incremental=torch.cat([attention(q[:,:,t:t+1],k[:,:,:t+1],v[:,:,:t+1],config,query_offset=t)
                               for t in range(17)],2)
        exact(full,incremental)
        kk,vv=k.clone(),v.clone();kk[:,:,7:]*=20;vv[:,:,7:]+=10
        exact(full[:,:,:7],attention(q,kk,vv,config)[:,:,:7])

    def test_static_call_estimate_is_not_throughput(self):
        e=operation_estimate(4096,AttentionConfig(query_block=128,key_block=128))
        self.assertEqual(e['query_key_tile_pairs_per_layer'],528)
        self.assertGreater(e['approximate_vector_qdq_calls_all_layers'],500000)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
