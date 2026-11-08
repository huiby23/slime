"""Correctly rounded positive finite FP32 division via integer significands.

Experimental helper; CPU tests and NPU integer-division validation are required
before using it. It is not optimized. No float64 or CPU tensor transfers occur.
"""
import torch


@torch.no_grad()
def divide_positive(numerator, denominator):
    denominator=denominator.float()
    numerator=torch.as_tensor(numerator,dtype=torch.float32,device=denominator.device)
    numerator,denominator=torch.broadcast_tensors(numerator,denominator)
    def components(x):
        raw=x.contiguous().view(torch.int32)
        exponent=torch.bitwise_and(torch.bitwise_right_shift(raw,23),255)
        subnormal=exponent==0
        scaled=torch.where(subnormal,x*16777216.0,x)
        bits=scaled.contiguous().view(torch.int32)
        normal_exponent=torch.bitwise_and(torch.bitwise_right_shift(bits,23),255)
        e=normal_exponent-subnormal.to(torch.int32)*24
        mantissa=(torch.bitwise_and(bits,0x7fffff)+0x800000).to(torch.int64)
        return e,mantissa
    en,mn=components(numerator);ed,md=components(denominator)
    ratio_ge_one=mn>=md
    base_shift=torch.where(ratio_ge_one,23,24)
    exponent=en-ed+127-(~ratio_ge_one).to(torch.int32)
    shift=base_shift+torch.minimum(exponent-1,torch.zeros_like(exponent))
    # For exponent<-23 the rounded subnormal is necessarily zero. Clamp its
    # internal arithmetic so that all executed integer shifts remain bounded.
    scaled_n=torch.bitwise_left_shift(mn,shift.clamp(0,24).to(torch.int64))
    scaled_d=torch.bitwise_left_shift(md,(-shift).clamp(0,1).to(torch.int64))
    q=torch.div(scaled_n,scaled_d,rounding_mode='floor')
    remainder=scaled_n-q*scaled_d
    twice=2*remainder
    up=(twice>scaled_d)|((twice==scaled_d)&(torch.bitwise_and(q,1)!=0))
    rounded=q+up.to(torch.int64)
    bits=torch.bitwise_left_shift((exponent.clamp_min(1)-1).to(torch.int64),23)+rounded
    bits=torch.where(exponent<-23,torch.zeros_like(bits),bits)
    bits=torch.where((exponent>254)|(bits>=0x7f800000),torch.full_like(bits,0x7f800000),bits)
    bits=torch.where(numerator==0,torch.zeros_like(bits),bits)
    bits=torch.where(denominator==0,torch.full_like(bits,0x7f800000),bits)
    invalid=(numerator<0)|(denominator<0)|~torch.isfinite(numerator)|~torch.isfinite(denominator)|((numerator==0)&(denominator==0))
    bits=torch.where(invalid,torch.full_like(bits,0x7fc00000),bits)
    return bits.to(torch.int32).contiguous().view(torch.float32)
