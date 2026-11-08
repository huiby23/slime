"""Explicit forward-only fused candidates. Not installed into any active backend.

Calling this API requires authorization to use the NPU and separately requires
HBY_C4V8_ALLOW_UNVALIDATED_FUSION=1 until the real-device audit is complete.
Every call returns device error flags; check=True synchronizes ONCE and rejects
all invalid arithmetic. check=False is for an explicitly audited caller that
combines flags and MUST check them before accepting an attention result.
"""
import os
import torch
from fused_mxfp8 import (qdq_kernel,pre_p_kernel,post_pv_kernel,normalize_kernel,
                        pre_p_prepare_kernel,pre_p_denominator_kernel)


def _input(x):
    if os.environ.get('HBY_C4V8_ALLOW_UNVALIDATED_FUSION')!='1':
        raise RuntimeError('Unvalidated fusion requires explicit opt-in')
    if x.device.type not in ('npu','privateuseone') or x.dtype!=torch.float32:
        raise ValueError('This candidate requires NPU FP32 tensors')
    if x.ndim<1 or x.numel()==0 or not x.is_contiguous():
        raise ValueError('Nonempty contiguous input required')
    if torch.is_grad_enabled() and x.requires_grad:
        raise ValueError('Forward-only candidate: no implicit autograd/STE substitution')
    if x.shape[-1]>4096:
        raise ValueError('Unreviewed fusion tile larger than4096')
    width=x.shape[-1];rows=x.numel()//width;block=max(32,1<<(width-1).bit_length())
    return rows,width,block


def _same(x,*others):
    for other in others:
        if other.device!=x.device or other.dtype!=torch.float32 or not other.is_contiguous():
            raise ValueError('All inputs must be contiguous FP32 on the same device')
        if torch.is_grad_enabled() and other.requires_grad:
            raise ValueError('Forward-only candidate does not implement autograd')


def check_flags(flags):
    if bool((flags!=0).any()):
        # Only a scalar is transferred; no Q/K/V, probabilities or cache copies.
        raise FloatingPointError('Fused C4V8 vector stage flagged invalid data/arithmetic; no fallback')


def qdq_forward(x,*,allow_mask=False,check=True):
    rows,width,block=_input(x)
    stored=x.reshape(rows,width)
    if width!=block:stored=torch.nn.functional.pad(stored,(0,block-width))
    y=torch.empty((rows,block),dtype=x.dtype,device=x.device)
    codes=torch.empty((rows,block),dtype=torch.uint8,device=x.device)
    scales=torch.empty((rows,block//32),dtype=torch.uint8,device=x.device)
    flags=torch.empty((rows,),dtype=torch.int32,device=x.device)
    qdq_kernel[(rows,)](stored,y,codes,scales,flags,block,block,allow_mask,
                        num_warps=4,enable_fp_fusion=False)
    if check:check_flags(flags)
    padded=(width+31)//32*32
    return y[:,:width].contiguous().reshape(x.shape),codes[:,:padded].contiguous(),scales[:,:padded//32].contiguous(),flags


def pre_p_forward(scores,maximum,denominator,*,check=True):
    rows,width,block=_input(scores);_same(scores,maximum,denominator)
    expected=(*scores.shape[:-1],1)
    if maximum.shape!=expected or denominator.shape!=expected:
        raise ValueError('One independent running-state scalar per logical row required')
    new_m=torch.empty_like(maximum);new_l=torch.empty_like(denominator)
    alpha=torch.empty_like(maximum);p=torch.empty((rows,block),device=scores.device,dtype=torch.float32)
    stored=scores.reshape(rows,width)
    if width!=block:stored=torch.nn.functional.pad(stored,(0,block-width),value=-float('inf'))
    flags=torch.empty((rows,),dtype=torch.int32,device=scores.device)
    # Store P before starting the sum/denominator stage. This preserves every
    # boundary while avoiding the unaccepted monolithic live-value corruption.
    pre_p_prepare_kernel[(rows,)](stored,maximum,new_m,alpha,p,flags,
                                  block,(block-1).bit_length(),num_warps=4,enable_fp_fusion=False)
    pre_p_denominator_kernel[(rows,)](p,denominator,alpha,new_l,flags,
                                      block,(block-1).bit_length(),num_warps=4,enable_fp_fusion=False)
    if check:check_flags(flags)
    return new_m,new_l,alpha,p[:,:width].contiguous().reshape(scores.shape),flags


def post_pv_forward(accumulator,contribution,alpha,denominator=None,*,check=True):
    rows,width,block=_input(accumulator);_same(accumulator,contribution,alpha)
    if contribution.shape!=accumulator.shape or alpha.shape!=(*accumulator.shape[:-1],1):
        raise ValueError('Contribution/alpha shape mismatch')
    if denominator is not None:
        _same(accumulator,denominator)
        if denominator.shape!=alpha.shape:raise ValueError('Denominator shape mismatch')
    old=accumulator.reshape(rows,width);part=contribution.reshape(rows,width)
    if width!=block:
        old=torch.nn.functional.pad(old,(0,block-width));part=torch.nn.functional.pad(part,(0,block-width))
    result=torch.empty((rows,block),device=accumulator.device,dtype=torch.float32)
    flags=torch.empty((rows,),dtype=torch.int32,device=accumulator.device)
    post_pv_kernel[(rows,)](old,part,alpha,alpha if denominator is None else denominator,
                            result,flags,block,block,denominator is not None,
                            num_warps=4,enable_fp_fusion=False)
    if check:check_flags(flags)
    return result[:,:width].contiguous().reshape(accumulator.shape),flags


def normalize_forward(numerator,denominator,*,check=True):
    rows,width,block=_input(numerator);_same(numerator,denominator)
    if denominator.shape!=(*numerator.shape[:-1],1):
        raise ValueError('One denominator per logical output row required')
    stored=numerator.reshape(rows,width)
    if width!=block:stored=torch.nn.functional.pad(stored,(0,block-width))
    output=torch.empty((rows,block),device=numerator.device,dtype=torch.float32)
    flags=torch.empty((rows,),device=numerator.device,dtype=torch.int32)
    normalize_kernel[(rows,)](stored,denominator,output,flags,block,block,
                              num_warps=4,enable_fp_fusion=False)
    if check:check_flags(flags)
    return output[:,:width].contiguous().reshape(numerator.shape),flags
