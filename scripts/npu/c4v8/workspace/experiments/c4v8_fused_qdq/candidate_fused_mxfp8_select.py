"""Unvalidated Triton-Ascend 3.2 forward candidates; explicit opt-in only.

No installation, device probe, allocation, or JIT compilation occurs at import.
Uses integer mantissas for E4M3 RNE and bitcast E8M0 scales, no FP8 cast/pow/log.
Checked wrappers must inspect returned flags before accepting outputs. The
pre-P kernel leaves NVFP4 P quantization and QK/PV outside this module.
"""
import triton
import triton.language as tl
from triton.triton_patch.language.core import gather as ascend_gather
from nvfp4_triton import _divide_positive_rne


@triton.jit
def _power2(e):
    bits=tl.where(e==-127,1<<22,(e+127)<<23).to(tl.int32)
    return bits.to(tl.float32,bitcast=True)


@triton.jit
def _finite(x):
    return (x.to(tl.int32,bitcast=True)&0x7fffffff)<0x7f800000


@triton.jit
def _e4(x):
    raw=x.to(tl.int32,bitcast=True)
    a=tl.minimum(raw&0x7fffffff,0x43e00000).to(tl.float32,bitcast=True)
    exponent=((a.to(tl.int32,bitcast=True)>>23)&255)-127
    step_exp=tl.maximum(exponent-3,-9)
    units=a*_power2(-step_exp)
    whole=units.to(tl.int32)
    frac=units-whole.to(tl.float32)
    rounded=whole+((frac>0.5)|((frac==0.5)&((whole&1)!=0))).to(tl.int32)
    rounded=tl.minimum(rounded.to(tl.float32)*_power2(step_exp),448.0)
    bits=rounded.to(tl.int32,bitcast=True)
    normal=(((bits>>23)&255)-120)*8+((bits>>20)&7)
    subnormal=(rounded*512.0).to(tl.int32)
    code=tl.where(rounded<0.015625,subnormal,normal)+((raw<0).to(tl.int32)*128)
    return (bits|(raw&-2147483648)).to(tl.float32,bitcast=True),code


@triton.jit
def _mx_groups(x,ALLOW_MASK:tl.constexpr):
    # Input [groups,32]; groups are within a single logical token/head row.
    raw=x.to(tl.int32,bitcast=True)
    mask=(raw==-8388608) & ALLOW_MASK
    invalid=(~_finite(x))&(~mask)
    clean=tl.where(mask|invalid,0.0,x)
    amax=tl.max(tl.abs(clean),1)
    exponent_bits=(amax.to(tl.int32,bitcast=True)>>23)&255
    candidate=tl.minimum(tl.maximum(exponent_bits-135,-127),119)
    threshold=_power2(candidate)*448.0
    exponent=tl.minimum(tl.maximum(candidate+(amax>threshold).to(tl.int32),-127),127)
    exponent=tl.where(amax==0.0,0,exponent)
    normalized=clean*_power2(-exponent)[:,None]
    payload,codes=_e4(normalized)
    values=payload*_power2(exponent)[:,None]
    # Sign-preserving bit composition also protects negative zero.
    out_bits=(values.to(tl.int32,bitcast=True)&0x7fffffff)|(clean.to(tl.int32,bitcast=True)&-2147483648)
    values=out_bits.to(tl.float32,bitcast=True)
    overflow=~_finite(values)
    flags=tl.max(invalid.to(tl.int32),1)|(tl.max(overflow.to(tl.int32),1)*2)
    values=tl.where(mask,-float('inf'),values)
    values=tl.where(invalid,float('nan'),values)
    return values,codes,exponent+127,flags


@triton.jit
def _mx_vector(x,BLOCK:tl.constexpr,ALLOW_MASK:tl.constexpr):
    if BLOCK>=32:
        y,c,s,f=_mx_groups(tl.reshape(x,(BLOCK//32,32)),ALLOW_MASK)
    else:
        # One short logical block; omitted padding is allzero and cannot change
        # its amax. No values from other query rows enter this scale group.
        y,c,s,f=_mx_groups(tl.reshape(x,(1,BLOCK)),ALLOW_MASK)
    return tl.reshape(y,(BLOCK,)),tl.max(f,0)


@triton.jit
def _mx_scalar(x,ALLOW_MASK:tl.constexpr):
    # Algebraic form of a one-element block padded with31zeros. No cross-row
    # scale grouping, and no integer-bit reduction to extract a scalar.
    raw=x.to(tl.int32,bitcast=True);mask=(raw==-8388608)&ALLOW_MASK
    invalid=(~_finite(x))&(~mask);clean=tl.where(mask|invalid,0.0,x)
    amax=tl.abs(clean)
    candidate=tl.minimum(tl.maximum(((amax.to(tl.int32,bitcast=True)>>23)&255)-135,-127),119)
    exponent=candidate+(amax>(_power2(candidate)*448.0)).to(tl.int32)
    exponent=tl.where(amax==0.0,0,exponent)
    payload,code=_e4(clean*_power2(-exponent))
    value=payload*_power2(exponent)
    value=((value.to(tl.int32,bitcast=True)&0x7fffffff)|(raw&-2147483648)).to(tl.float32,bitcast=True)
    flag=invalid.to(tl.int32)|((~_finite(value)).to(tl.int32)*2)
    value=tl.where(mask,-float('inf'),value);value=tl.where(invalid,float('nan'),value)
    return value,flag


@triton.jit
def _tree(x,BLOCK:tl.constexpr,WIDTH:tl.constexpr,LEVELS:tl.constexpr,IS_MAX:tl.constexpr):
    idx=tl.arange(0,BLOCK)
    identity:tl.constexpr=-float('inf') if IS_MAX else 0.0
    x=tl.where(idx<WIDTH,x,identity)
    x,flags=_mx_vector(x,BLOCK,IS_MAX)
    # Fixed aligned buffers: Ascend3.2 rejected the compact reshape/split tree
    # with a real storage-alignment compiler error. The two-stage pre-P API
    # separates P lifetime from this sum tree; mathematical boundaries stay.
    for level in tl.static_range(0,LEVELS):
        left=ascend_gather(x,(2*idx)%BLOCK,0)
        right=ascend_gather(x,(2*idx+1)%BLOCK,0)
        partial=tl.maximum(left,right) if IS_MAX else left+right
        partial=tl.where(idx<((WIDTH+(1<<(level+1))-1)>>(level+1)),partial,identity)
        x,flag=_mx_vector(partial,BLOCK,IS_MAX)
        flags=flags|flag
    # Select only the existing finalized x[0]; other lanes are -inf. This
    # max does not alter the specified adjacent-pair quantized tree.
    selected=tl.max(tl.where(idx==0,x,-float('inf')),0)
    return tl.reshape(selected,(1,)),flags


@triton.jit
def qdq_kernel(X,Y,Codes,Scales,Flags,WIDTH:tl.constexpr,BLOCK:tl.constexpr,ALLOW_MASK:tl.constexpr):
    row=tl.program_id(0)
    offsets=tl.arange(0,BLOCK)
    x=tl.load(X+row*WIDTH+offsets,offsets<WIDTH,other=0.0).to(tl.float32)
    y,c,s,f=_mx_groups(tl.reshape(x,(BLOCK//32,32)),ALLOW_MASK)
    tl.store(Y+row*WIDTH+offsets,tl.reshape(y,(BLOCK,)),offsets<WIDTH)
    tl.store(Codes+row*BLOCK+offsets,tl.reshape(c,(BLOCK,)).to(tl.uint8))
    group=tl.arange(0,BLOCK//32)
    tl.store(Scales+row*(BLOCK//32)+group,s.to(tl.uint8))
    tl.store(Flags+row,tl.max(f,0))


@triton.jit
def _pre_p_vector(score,old_m,old_l,WIDTH:tl.constexpr,BLOCK:tl.constexpr,LEVELS:tl.constexpr):
    score,flags=_mx_vector(score,BLOCK,True)
    local_m,flag=_tree(score,BLOCK,WIDTH,LEVELS,True);flags=flags|flag
    new_m,flag=_mx_scalar(tl.maximum(old_m,local_m),True);flags=flags|flag
    flags=flags|tl.where(_finite(new_m),0,4)
    old_mask=(old_m.to(tl.int32,bitcast=True)==-8388608)
    score_mask=(score.to(tl.int32,bitcast=True)==-8388608)
    md=old_m-new_m;sd=score-new_m
    flags=flags|tl.where(_finite(md)|old_mask,0,8)
    flags=flags|tl.max(tl.where(_finite(sd)|score_mask,0,8),0)
    md,flag=_mx_scalar(md,True);flags=flags|flag
    alpha,flag=_mx_scalar(tl.exp(md),False);flags=flags|flag
    sd,flag=_mx_vector(sd,BLOCK,True);flags=flags|flag
    p,flag=_mx_vector(tl.exp(sd),BLOCK,False);flags=flags|flag
    local_l,flag=_tree(p,BLOCK,WIDTH,LEVELS,False);flags=flags|flag
    scaled_l,flag=_mx_scalar(old_l*alpha,False);flags=flags|flag
    new_l,flag=_mx_scalar(scaled_l+local_l,False);flags=flags|flag
    flags=flags|tl.where(_finite(new_l)&(new_l>0.0),0,16)
    return new_m,new_l,alpha,p,flags


@triton.jit
def pre_p_kernel(Scores,OldM,OldL,NewM,NewL,Alpha,P,Flags,
                 WIDTH:tl.constexpr,BLOCK:tl.constexpr,LEVELS:tl.constexpr):
    row=tl.program_id(0);idx=tl.arange(0,BLOCK)
    score=tl.load(Scores+row*WIDTH+idx,idx<WIDTH,other=-float('inf'))
    scalar_lane=tl.arange(0,1)
    old_m=tl.load(OldM+row+scalar_lane);old_l=tl.load(OldL+row+scalar_lane)
    new_m,new_l,alpha,p,flags=_pre_p_vector(score,old_m,old_l,WIDTH,BLOCK,LEVELS)
    tl.store(NewM+row+scalar_lane,new_m);tl.store(NewL+row+scalar_lane,new_l);tl.store(Alpha+row+scalar_lane,alpha)
    tl.store(P+row*WIDTH+idx,p,idx<WIDTH);tl.store(Flags+row+scalar_lane,flags)


@triton.jit
def pre_p_prepare_kernel(Scores,OldM,NewM,Alpha,P,Flags,BLOCK:tl.constexpr,LEVELS:tl.constexpr):
    row=tl.program_id(0);idx=tl.arange(0,BLOCK);one=tl.arange(0,1)
    score=tl.load(Scores+row*BLOCK+idx);old_m=tl.load(OldM+row+one)
    score,flags=_mx_vector(score,BLOCK,True)
    local_m,flag=_tree(score,BLOCK,BLOCK,LEVELS,True);flags=flags|flag
    new_m,flag=_mx_scalar(tl.maximum(old_m,local_m),True);flags=flags|flag
    flags=flags|tl.where(_finite(new_m),0,4)
    md=old_m-new_m;sd=score-new_m
    old_mask=(old_m.to(tl.int32,bitcast=True)==-8388608)
    score_mask=(score.to(tl.int32,bitcast=True)==-8388608)
    flags=flags|tl.where(_finite(md)|old_mask,0,8)
    flags=flags|tl.max(tl.where(_finite(sd)|score_mask,0,8),0)
    md,flag=_mx_scalar(md,True);flags=flags|flag
    alpha,flag=_mx_scalar(tl.exp(md),False);flags=flags|flag
    sd,flag=_mx_vector(sd,BLOCK,True);flags=flags|flag
    p,flag=_mx_vector(tl.exp(sd),BLOCK,False);flags=flags|flag
    tl.store(NewM+row+one,new_m);tl.store(Alpha+row+one,alpha)
    tl.store(P+row*BLOCK+idx,p);tl.store(Flags+row+one,flags)


@triton.jit
def pre_p_denominator_kernel(P,OldL,Alpha,NewL,Flags,BLOCK:tl.constexpr,LEVELS:tl.constexpr):
    row=tl.program_id(0);idx=tl.arange(0,BLOCK);one=tl.arange(0,1)
    p=tl.load(P+row*BLOCK+idx);old_l=tl.load(OldL+row+one);alpha=tl.load(Alpha+row+one)
    flags=tl.load(Flags+row+one)
    local_l,flag=_tree(p,BLOCK,BLOCK,LEVELS,False);flags=flags|flag
    scaled_l,flag=_mx_scalar(old_l*alpha,False);flags=flags|flag
    new_l,flag=_mx_scalar(scaled_l+local_l,False);flags=flags|flag
    flags=flags|tl.where(_finite(new_l)&(new_l>0.0),0,16)
    tl.store(NewL+row+one,new_l);tl.store(Flags+row+one,flags)


@triton.jit
def post_pv_kernel(OldO,Contribution,Alpha,Denominator,NewO,Flags,
                   WIDTH:tl.constexpr,BLOCK:tl.constexpr,NORMALIZE:tl.constexpr):
    row=tl.program_id(0);idx=tl.arange(0,BLOCK)
    old=tl.load(OldO+row*WIDTH+idx,idx<WIDTH,other=0.0)
    contribution=tl.load(Contribution+row*WIDTH+idx,idx<WIDTH,other=0.0)
    alpha=tl.load(Alpha+row+tl.arange(0,1))
    old,flags=_mx_vector(old*alpha,BLOCK,False)
    contribution,flag=_mx_vector(contribution,BLOCK,False);flags=flags|flag
    result,flag=_mx_vector(old+contribution,BLOCK,False);flags=flags|flag
    if NORMALIZE:
        denominator=tl.load(Denominator+row+tl.arange(0,1))
        flags=flags|tl.where(_finite(denominator)&(denominator>0.0),0,16)
        normalized=_divide_positive_rne(tl.abs(result),denominator)
        normalized=(normalized.to(tl.int32,bitcast=True)|(result.to(tl.int32,bitcast=True)&-2147483648)).to(tl.float32,bitcast=True)
        result,flag=_mx_vector(normalized,BLOCK,False);flags=flags|flag
    tl.store(NewO+row*WIDTH+idx,result,idx<WIDTH)
    tl.store(Flags+row+tl.arange(0,1),flags)


@triton.jit
def normalize_kernel(Numerator,Denominator,Output,Flags,WIDTH:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);idx=tl.arange(0,BLOCK);one=tl.arange(0,1)
    numerator=tl.load(Numerator+row*WIDTH+idx,idx<WIDTH,other=0.0)
    denominator=tl.load(Denominator+row+one)
    flags=tl.where(_finite(denominator)&(denominator>0.0),0,16)
    normalized=_divide_positive_rne(tl.abs(numerator),denominator)
    normalized=(normalized.to(tl.int32,bitcast=True)|(numerator.to(tl.int32,bitcast=True)&-2147483648)).to(tl.float32,bitcast=True)
    output,flag=_mx_vector(normalized,BLOCK,False);flags=flags|flag
    tl.store(Output+row*WIDTH+idx,output,idx<WIDTH);tl.store(Flags+row+one,flags)
