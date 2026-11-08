"""Unactivated fallback candidate: isolate quantized max tree from exponent region."""
import triton
import triton.language as tl
from fused_mxfp8 import _mx_vector,_mx_scalar,_tree,_finite

@triton.jit
def maximum_stage(Scores,OldM,NewM,Flags,BLOCK:tl.constexpr,LEVELS:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK);one=tl.arange(0,1)
    score=tl.load(Scores+row*BLOCK+i);old=tl.load(OldM+row+one)
    score,flags=_mx_vector(score,BLOCK,True)
    value,flag=_tree(score,BLOCK,BLOCK,LEVELS,True);flags=flags|flag
    maximum,flag=_mx_scalar(tl.maximum(old,value),True);flags=flags|flag
    flags=flags|tl.where(_finite(maximum),0,4)
    tl.store(NewM+row+one,maximum);tl.store(Flags+row+one,flags)

@triton.jit
def probability_stage(Scores,OldM,NewM,Alpha,P,Flags,BLOCK:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK);one=tl.arange(0,1)
    score=tl.load(Scores+row*BLOCK+i);old=tl.load(OldM+row+one);maximum=tl.load(NewM+row+one)
    flags=tl.load(Flags+row+one)
    # Identical deterministic score QDQ is recomputed rather than materialized.
    # It does not consume a newly rounded value or alter a scale group.
    score,flag=_mx_vector(score,BLOCK,True);flags=flags|flag
    md=old-maximum;sd=score-maximum
    old_mask=(old.to(tl.int32,bitcast=True)==-8388608)
    score_mask=(score.to(tl.int32,bitcast=True)==-8388608)
    flags=flags|tl.where(_finite(md)|old_mask,0,8)
    flags=flags|tl.max(tl.where(_finite(sd)|score_mask,0,8),0)
    md,flag=_mx_scalar(md,True);flags=flags|flag
    alpha,flag=_mx_scalar(tl.exp(md),False);flags=flags|flag
    sd,flag=_mx_vector(sd,BLOCK,True);flags=flags|flag
    p,flag=_mx_vector(tl.exp(sd),BLOCK,False);flags=flags|flag
    tl.store(Alpha+row+one,alpha);tl.store(P+row*BLOCK+i,p);tl.store(Flags+row+one,flags)

class ThreeStagePrepare:
    def __getitem__(self,grid):
        def launch(scores,old_m,new_m,alpha,p,flags,block,levels,**options):
            maximum_stage[grid](scores,old_m,new_m,flags,block,levels,**options)
            return probability_stage[grid](scores,old_m,new_m,alpha,p,flags,block,**options)
        return launch
