"""Unvalidated small-pass kernels. Error flags use independent GM slots.

Do not hold score flags across a complex tree or combine them in a long-lived
integer accumulator: Ascend3.2 has a real external flag-OR/liveness counterexample.
All checks remain present and the wrapper checks every initialized slot.
"""
import triton
import triton.language as tl
from fused_mxfp8 import _finite,_mx_vector,_mx_scalar,_tree
from nvfp4_triton import _divide_positive_rne


@triton.jit(do_not_specialize=['TILES','SEQ'])
def scores_local_max(Scores,QuantizedScores,LocalMaximum,Flags,TILES,SEQ,BLOCK:tl.constexpr,STATE_STRIDE:tl.constexpr):
    program=tl.program_id(0);tile=program%TILES
    lane=tl.arange(0,BLOCK);one=tl.arange(0,1);flag_base=program*STATE_STRIDE+one
    score=tl.load(Scores+program*BLOCK+lane);valid=tile*BLOCK+lane<SEQ
    bad=tl.max((valid&(~_finite(score))).to(tl.int32),0)*32
    tl.store(Flags+flag_base,bad)
    score=tl.where(valid,score,-float('inf'))
    score,flag=_mx_vector(score,BLOCK,True)
    tl.store(Flags+flag_base+1,flag)
    tl.store(QuantizedScores+program*BLOCK+lane,score)
    maximum,flag=_tree(score,BLOCK,BLOCK,7,True)
    tl.store(LocalMaximum+program*STATE_STRIDE+one,maximum)
    tl.store(Flags+flag_base+2,flag)


@triton.jit(do_not_specialize=['TILES'])
def maximum_scan(LocalMaximum,Maximum,Alpha,Flags,TILES,STATE_STRIDE:tl.constexpr):
    row=tl.program_id(0);one=tl.arange(0,1)
    previous=tl.full((1,),-float('inf'),tl.float32)
    for tile in range(TILES):
        address=(row*TILES+tile)*STATE_STRIDE+one
        local=tl.load(LocalMaximum+address)
        current,flag=_mx_scalar(tl.maximum(previous,local),True)
        tl.store(Flags+address,flag)
        tl.store(Flags+address+1,tl.where(_finite(current),0,4))
        difference=previous-current
        old_mask=previous.to(tl.int32,bitcast=True)==-8388608
        tl.store(Flags+address+2,tl.where(_finite(difference)|old_mask,0,8))
        difference,flag=_mx_scalar(difference,True)
        tl.store(Flags+address+3,flag)
        alpha,flag=_mx_scalar(tl.exp(difference),False)
        tl.store(Flags+address+4,flag)
        tl.store(Maximum+address,current);tl.store(Alpha+address,alpha)
        previous=current


@triton.jit
def probability_pass(Scores,Maximum,P8,Flags,BLOCK:tl.constexpr,STATE_STRIDE:tl.constexpr):
    program=tl.program_id(0);lane=tl.arange(0,BLOCK);one=tl.arange(0,1)
    address=program*STATE_STRIDE+one
    score=tl.load(Scores+program*BLOCK+lane);maximum=tl.load(Maximum+address)
    structural=score.to(tl.int32,bitcast=True)==-8388608
    difference=score-maximum
    flag=tl.max(tl.where(_finite(difference)|structural,0,8),0)
    tl.store(Flags+address,flag)
    difference,flag=_mx_vector(difference,BLOCK,True)
    tl.store(Flags+address+1,flag)
    probability,flag=_mx_vector(tl.exp(difference),BLOCK,False)
    tl.store(Flags+address+2,flag)
    tl.store(P8+program*BLOCK+lane,probability)


@triton.jit
def probability_sum(P8,LocalDenominator,Flags,BLOCK:tl.constexpr,STATE_STRIDE:tl.constexpr):
    program=tl.program_id(0);lane=tl.arange(0,BLOCK);one=tl.arange(0,1)
    probability=tl.load(P8+program*BLOCK+lane)
    total,flag=_tree(probability,BLOCK,BLOCK,7,False)
    tl.store(LocalDenominator+program*STATE_STRIDE+one,total)
    tl.store(Flags+program*STATE_STRIDE+one,flag)


@triton.jit(do_not_specialize=['TILES'])
def final_state_scan(Alpha,LocalDenominator,Contribution,Output,FinalDenominator,Flags,NormalizeFlags,TILES,D:tl.constexpr,STATE_STRIDE:tl.constexpr):
    row=tl.program_id(0);lane=tl.arange(0,D);one=tl.arange(0,1)
    denominator=tl.full((1,),0.0,tl.float32)
    accumulator=tl.full((D,),0.0,tl.float32)
    for tile in range(TILES):
        address=row*TILES+tile;state_address=address*STATE_STRIDE+one
        alpha=tl.load(Alpha+state_address);local_l=tl.load(LocalDenominator+state_address)
        contribution=tl.load(Contribution+address*D+lane)
        scaled_l,flag=_mx_scalar(denominator*alpha,False)
        tl.store(Flags+state_address,flag)
        denominator,flag=_mx_scalar(scaled_l+local_l,False)
        tl.store(Flags+state_address+1,flag)
        tl.store(Flags+state_address+2,tl.where(_finite(denominator)&(denominator>0.0),0,16))
        scaled_o,flag=_mx_vector(accumulator*alpha,D,False)
        tl.store(Flags+state_address+3,flag)
        accumulator,flag=_mx_vector(scaled_o+contribution,D,False)
        tl.store(Flags+state_address+4,flag)
    normalized=_divide_positive_rne(tl.abs(accumulator),denominator)
    normalized=(normalized.to(tl.int32,bitcast=True)|(accumulator.to(tl.int32,bitcast=True)&-2147483648)).to(tl.float32,bitcast=True)
    output,flag=_mx_vector(normalized,D,False)
    tl.store(NormalizeFlags+row*STATE_STRIDE+one,flag)
    tl.store(Output+row*D+lane,output)
    tl.store(FinalDenominator+row*STATE_STRIDE+one,denominator)
