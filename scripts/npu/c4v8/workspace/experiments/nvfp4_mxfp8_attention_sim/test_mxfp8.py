import bisect
import math
import unittest
import torch
from mxfp8 import quantize,dequantize,fake_quantize_ste


def independent_e4m3_table():
    values=[]
    for code in range(127):
        e,m=divmod(code,8)
        values.append(m*2**-9 if e==0 else (1+m/8)*2**(e-7))
    return values


TABLE=independent_e4m3_table()


def scalar_code(value):
    sign=128 if math.copysign(1.0,value)<0 else 0
    value=abs(value)
    at=bisect.bisect_left(TABLE,value)
    if at==0: code=0
    elif at==len(TABLE): code=126
    else:
        low,high=at-1,at
        dlo,dhi=value-TABLE[low],TABLE[high]-value
        code=low if dlo<dhi else high if dhi<dlo else low if low%2==0 else high
    return sign|code


def scalar_row(row):
    answer=[]; scales=[]
    for offset in range(0,len(row),32):
        block=row[offset:offset+32];amax=max(map(abs,block))
        if amax==0: e=0
        else:
            mantissa,exp=math.frexp(amax/448)
            e=max(-127,min(127,exp-1 if mantissa==.5 else exp))
        scale=2.0**e;scales.append(e+127)
        for value in block:
            code=scalar_code(value/scale)
            answer.append(math.copysign(TABLE[code&127]*scale,-1 if code&128 else 1))
    return answer,scales


class MXFP8Tests(unittest.TestCase):
    def test_every_finite_e4m3_value_and_midpoint(self):
        values=TABLE+[(a+b)/2 for a,b in zip(TABLE[:-1],TABLE[1:])]
        values=values+[-x for x in values]
        x=torch.tensor([[x,448.0] for x in values])
        result=quantize(x)
        self.assertEqual(result.payload_bytes[:,0].tolist(),[scalar_code(x) for x in values])
        self.assertTrue(bool((result.scale_bytes==127).all()))

    def test_scale_jumps_and_480_conversion_distinction(self):
        a=torch.tensor([224.,448.])
        up=torch.nextafter(a,torch.full_like(a,math.inf))
        q=quantize(torch.stack((a,up)).flatten().reshape(-1,1))
        self.assertEqual(q.scale_bytes.flatten().tolist(),[126,127,127,128])
        q480=quantize(torch.tensor([[480.]]))
        self.assertEqual(q480.scale_bytes.item(),128)
        self.assertEqual(dequantize(q480).item(),480)

    def test_random_rows_against_independent_scalar(self):
        g=torch.Generator().manual_seed(2208)
        x=torch.randn((7,73),generator=g)*torch.tensor([.001,.01,.1,1,10,100,1000])[:,None]
        q=quantize(x);got=dequantize(q)
        for i,row in enumerate(x.tolist()):
            expected,scales=scalar_row(row)
            torch.testing.assert_close(got[i],torch.tensor(expected),atol=0,rtol=0)
            self.assertEqual(q.scale_bytes[i].tolist(),scales)

    def test_zero_signed_zero_tail_and_row_isolation(self):
        x=torch.zeros(3,35);x[0,0]=-0.;x[1,34]=17.;x[2,:]=1000.
        q=quantize(x);y=dequantize(q)
        self.assertEqual(q.padded_last_dim,64)
        self.assertTrue(torch.signbit(y[0,0]))
        self.assertEqual(q.scale_bytes[0].tolist(),[127,127])
        self.assertEqual(q.payload_bytes[1,35:].count_nonzero().item(),0)
        torch.testing.assert_close(y[:2],dequantize(quantize(x[:2])),atol=0,rtol=0)

    def test_mask_preserved_and_excluded_from_amax(self):
        x=torch.tensor([[1.,2.,-math.inf,3.]],requires_grad=True)
        out=fake_quantize_ste(x,allow_masked_negative_infinity=True)
        finite=fake_quantize_ste(torch.tensor([[1.,2.,0.,3.]]))
        torch.testing.assert_close(out[:,[0,1,3]],finite[:,[0,1,3]],atol=0,rtol=0)
        self.assertTrue(torch.isneginf(out[0,2]))
        out[:,[0,1,3]].sum().backward()
        torch.testing.assert_close(x.grad,torch.tensor([[1.,1.,0.,1.]]),atol=0,rtol=0)

    def test_identity_ste_and_no_rng_use(self):
        x=torch.tensor([[.12345,3.14159,-1.23456]],requires_grad=True)
        g=torch.tensor([[.7,.3,-.2]])
        state=torch.random.get_rng_state().clone()
        (fake_quantize_ste(x)*g).sum().backward()
        torch.testing.assert_close(x.grad,g,atol=0,rtol=0)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    def test_minimum_scale_underflow_and_overflow_fail_loudly(self):
        x=torch.tensor([[torch.finfo(torch.float32).tiny/4096]])
        q=quantize(x)
        self.assertEqual(q.scale_bytes.item(),0)
        self.assertEqual(dequantize(q).item(),0)
        with self.assertRaises(FloatingPointError):
            dequantize(quantize(torch.tensor([[torch.finfo(torch.float32).max]])))

    def test_invalid_inputs_rejected(self):
        for bad in (float('nan'),float('inf'),-float('inf')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                quantize(torch.tensor([bad]))
        for bad in (float('nan'),float('inf')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                fake_quantize_ste(torch.tensor([bad]),allow_masked_negative_infinity=True)
        with self.assertRaises(ValueError):fake_quantize_ste(torch.tensor([-math.inf]))


if __name__=='__main__':unittest.main()
