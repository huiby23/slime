"""CPU operation oracle for the experimental fusion, not a speed implementation.

The integer/FP32 algorithm mirrors fused_mxfp8.py and independently checks the
existing device_quant encoder. This module does not import Triton or torch_npu.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class Result:
    values: torch.Tensor
    payload: torch.Tensor
    scales: torch.Tensor
    flags: torch.Tensor


def power2(e):
    e=e.to(torch.int32)
    bits=torch.where(e==-127,torch.full_like(e,1<<22),(e+127)<<23)
    return bits.contiguous().view(torch.float32)


def round_e4(x):
    raw=x.contiguous().view(torch.int32)
    a=(raw & 0x7fffffff).clamp_max(0x43e00000).contiguous().view(torch.float32)
    exponent=((a.contiguous().view(torch.int32)>>23)&255)-127
    step_exp=(exponent-3).clamp_min(-9)
    units=a*power2(-step_exp)
    whole=units.to(torch.int32)
    fractional=units-whole.to(torch.float32)
    rounded=whole+((fractional>0.5)|((fractional==0.5)&((whole&1)!=0))).to(torch.int32)
    rounded=(rounded.float()*power2(step_exp)).clamp_max(448.0)
    rbits=rounded.contiguous().view(torch.int32)
    normal=(((rbits>>23)&255)-120)*8+((rbits>>20)&7)
    subnormal=(rounded*512.0).to(torch.int32)
    code=torch.where(rounded<2.0**-6,subnormal,normal)
    code=code+((raw<0).to(torch.int32)*128)
    value=(rbits|(raw&-2147483648)).contiguous().view(torch.float32)
    return value,code.to(torch.uint8)


@torch.no_grad()
def qdq(x,allow_mask=False):
    if x.device.type!='cpu' or x.dtype!=torch.float32 or x.numel()==0:
        raise ValueError('CPU nonempty FP32 only')
    shape=x.shape; width=shape[-1]; padded=(width+31)//32*32
    rows=x.reshape(-1,width)
    raw=rows.contiguous().view(torch.int32)
    structural=(raw==-8388608) if allow_mask else torch.zeros_like(rows,dtype=torch.bool)
    invalid=(~torch.isfinite(rows))&~structural
    clean=torch.where(invalid|structural,0.,rows)
    groups=F.pad(clean,(0,padded-width)).reshape(-1,padded//32,32)
    maximum=groups.abs().amax(-1)
    exponent_bits=(maximum.contiguous().view(torch.int32)>>23)&255
    candidate=(exponent_bits-135).clamp(-127,119)
    threshold=power2(candidate)*448.0
    exponent=(candidate+(maximum>threshold).to(torch.int32)).clamp(-127,127)
    exponent=torch.where(maximum==0,0,exponent)
    normalized=groups*power2(-exponent).unsqueeze(-1)
    rounded,payload=round_e4(normalized)
    values=rounded*power2(exponent).unsqueeze(-1)
    # Explicitly preserve signed zero even if an accelerator multiply does not.
    values=(values.contiguous().view(torch.int32)&0x7fffffff | (groups.contiguous().view(torch.int32)&-2147483648)).contiguous().view(torch.float32)
    values=values.reshape(rows.shape[0],padded)[:,:width]
    overflow=~torch.isfinite(values)
    flags=invalid.any(-1).to(torch.int32)|(overflow.any(-1).to(torch.int32)*2)
    values=torch.where(structural,-float('inf'),values)
    values=torch.where(invalid,float('nan'),values)
    return Result(values.reshape(shape),payload.reshape(rows.shape[0],padded),
                  (exponent+127).to(torch.uint8),flags)


def tree(x,operation):
    partial=qdq(x,operation=='max').values
    while partial.shape[-1]>1:
        if partial.shape[-1]%2:
            partial=F.pad(partial,(0,1),value=-float('inf') if operation=='max' else 0.)
        partial=(torch.maximum(partial[...,::2],partial[...,1::2]) if operation=='max'
                 else partial[...,::2]+partial[...,1::2])
        partial=qdq(partial,operation=='max').values
    return partial


def fixed_width_tree(x,operation):
    """Mirror fixed-size lane/gather layout used by the Triton tree candidate."""
    width=x.shape[-1];block=max(32,1<<(width-1).bit_length())
    identity=-float('inf') if operation=='max' else 0.
    partial=qdq(F.pad(x,(0,block-width),value=identity),operation=='max').values
    idx=torch.arange(block)
    for level in range((width-1).bit_length()):
        left=partial[...,2*idx%block];right=partial[...,(2*idx+1)%block]
        values=torch.maximum(left,right) if operation=='max' else left+right
        count=(width+(1<<(level+1))-1)>>(level+1)
        partial=qdq(torch.where(idx<count,values,identity),operation=='max').values
    return partial[...,:1]


def pre_p(scores,maximum,denominator):
    """Includes score QDQ and every max/sum tree boundary; returns m,l,a,P."""
    s=qdq(scores,True).values
    m=qdq(torch.maximum(maximum,tree(s,'max')),True).values
    alpha=qdq(torch.exp(qdq(maximum-m,True).values)).values
    p=qdq(torch.exp(qdq(s-m,True).values)).values
    l=qdq(qdq(denominator*alpha).values+tree(p,'sum')).values
    if not torch.isfinite(m).all() or not (torch.isfinite(l)&(l>0)).all():
        raise FloatingPointError('Invalid softmax state; no fallback')
    return m,l,alpha,p


def post_pv(accumulator,contribution,alpha,denominator=None):
    value=qdq(qdq(accumulator*alpha).values+qdq(contribution).values).values
    if denominator is not None:value=qdq(value/denominator).values
    return value
