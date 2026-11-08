"""Experimental device-capable FA forward QDQ + STE; not a native low-bit kernel.

This port preserves the CPU reference operation graph. It is intentionally
strict: assertions synchronize a device, so performance is a measured limitation.
NPU execution must be authorized only after existing jobs release resources.
"""
from dataclasses import asdict, dataclass
import math
import torch
from device_quant import nvfp4_qdq as c4_qdq
from device_quant import mxfp8_qdq as v8_qdq


@dataclass(frozen=True)
class AttentionConfig:
    query_block: int = 8
    key_block: int = 32
    vector: str = 'mxfp8'
    quantize_operands: bool = True
    causal: bool = True
    reduction: str = 'mxfp8_tree_reduction'

    def __post_init__(self):
        if self.query_block<1 or self.key_block<1 or self.vector not in ('fp32','bf16','mxfp8'):
            raise ValueError('Invalid tile sizes or vector policy')
        if self.reduction not in ('mixed_fp32_reduction','mxfp8_tree_reduction'):
            raise ValueError('Unknown explicit reduction recipe')
        if self.reduction=='mxfp8_tree_reduction' and self.vector!='mxfp8':
            raise ValueError('MXFP8 tree policy requires the MXFP8 vector arm')

    def metadata(self):
        return dict(**asdict(self),
            cube='NVFP4 E2M1/block16 E4M3FN/FP32 per-token row scale; FP32 reference matmul after QDQ',
            vector_format='MXFP8 E4M3FN/block32 E8M0/NVIDIA-documented ceil-amax when mxfp8 selected',
            experimental_role=('forward_boundary_baseline' if self.reduction=='mxfp8_tree_reduction'
                else 'fp32_reduction_ablation'),
            vector_boundaries=['masked scores','max-difference exp input','alpha rescale',
                'shifted-score exp input','unnormalized probabilities','PV contribution',
                'old-output rescale','output sum','final normalized output'],
            fp32_exceptions=(['running maximum','max reduction','denominator sum and running update']
                if self.reduction=='mixed_fp32_reduction' else [])+
                ['underlying elementary operation before boundary QDQ','dequantized reference matmul accumulation',
                 'scale estimation in FP32 only','exact scalar constants and structural mask/index arithmetic'],
            reduction_recipe=('FP32 max/sum/running states' if self.reduction=='mixed_fp32_reduction' else
                'Adjacent pairwise binary tree along last key axis; pad odd level with zero for sum or -inf for max; MXFP8 QDQ each level. Running max merge, denominator multiply and denominator add each followed by QDQ.'),
            scale_groups='Within a query row: key axis for scores/probabilities, feature axis for O. Never across query tokens.',
            scalar_vector_states='One-element final-axis blocks padded with zeros; no grouping across query tokens',
            masks='Apply causal mask before score QDQ; -inf is preserved out-of-band and excluded from amax',
            p_quantization='Unnormalized online-softmax P per query within each tile; not unspecified HiF4 first-order rule',
            v_layout='Per-token features, not transposed native-GEMM layout',
            backward='ordinary autograd; identity STE through QDQ; no quantized backward GEMM claim',
            efficient_backward=False, production_ready=False, npu_validated=False, float8_cast=False, strict_device_synchronization=True,
            native_bit_exact=False, hif4=False, fully_8bit_vector_arithmetic=False,
            quantized_reduction_boundaries=self.reduction=='mxfp8_tree_reduction',
            reduction_semantics='Experiment-defined operation order; MXFP8 format alone does not define addition or exp arithmetic')


class _Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x): return x.bfloat16().float()
    @staticmethod
    def backward(ctx,g): return g


def vector_round(x,policy,*,allow_structural_negative_infinity=False):
    if policy=='mxfp8':
        return v8_qdq(x,allow_masked_negative_infinity=allow_structural_negative_infinity)
    mask=torch.isneginf(x) if allow_structural_negative_infinity else torch.zeros_like(x,dtype=torch.bool)
    if not bool((torch.isfinite(x)|mask).all()):
        raise FloatingPointError('Nonfinite ordinary Vector input; not a structural mask')
    result=_Round.apply(x) if policy=='bf16' else x
    if not bool((torch.isfinite(result)|mask).all()):
        raise FloatingPointError('Nonfinite Vector rounding result; no silent fallback')
    return result


def mxfp8_tree_reduce(x,operation='sum'):
    """Every tree-level result is requantized, never a FP32 reduction then cast.

    Pair [0,1], [2,3], ... along the final key axis. Each level's partials are
    independently grouped by final-axis32 within each query row. Returning
    a scalar still uses a padded one-element block, never groups query rows.
    """
    if operation not in ('sum','max') or x.ndim<1 or x.shape[-1]<1:
        raise ValueError('Expected nonempty last axis and sum/max operation')
    partial=vector_round(x,'mxfp8',allow_structural_negative_infinity=operation=='max')
    identity=0.0 if operation=='sum' else -math.inf
    while partial.shape[-1]>1:
        if partial.shape[-1]%2:
            partial=torch.cat([partial,torch.full_like(partial[...,:1],identity)],-1)
        left,right=partial[...,0::2],partial[...,1::2]
        partial=left+right if operation=='sum' else torch.maximum(left,right)
        partial=vector_round(partial,'mxfp8',allow_structural_negative_infinity=operation=='max')
    return partial


def online_softmax_step(scores,maximum,denominator,config):
    """One explicitly ordered tile update; fails instead of rescuing an unstable recipe.

    `scores` already includes the structural causal mask and score boundary QDQ.
    Empty local rows keep prior finite state; a never-initialized all-masked row
    has zero mass and is rejected. This reference does not implement empty rows.
    """
    vr=lambda x,masked=False:vector_round(x,config.vector,allow_structural_negative_infinity=masked)
    if config.reduction=='mxfp8_tree_reduction':
        local_maximum=mxfp8_tree_reduce(scores,'max')
        new_maximum=vr(torch.maximum(maximum,local_maximum),masked=True)
    else:
        new_maximum=torch.maximum(maximum,scores.amax(-1,keepdim=True))
    if not bool(torch.isfinite(new_maximum).all()):
        raise FloatingPointError('Uninitialized/all-masked softmax row; no silent fallback')
    maximum_difference=maximum-new_maximum
    score_difference=scores-new_maximum
    if not bool((torch.isfinite(maximum_difference)|torch.isneginf(maximum)).all()):
        raise FloatingPointError('Ordinary running-maximum subtraction overflow')
    if not bool((torch.isfinite(score_difference)|torch.isneginf(scores)).all()):
        raise FloatingPointError('Ordinary score subtraction overflow')
    alpha=vr(torch.exp(vr(maximum_difference,masked=True)))
    probabilities=vr(torch.exp(vr(score_difference,masked=True)))
    if config.reduction=='mxfp8_tree_reduction':
        new_denominator=vr(vr(denominator*alpha)+mxfp8_tree_reduce(probabilities,'sum'))
    else:
        new_denominator=denominator*alpha+probabilities.sum(-1,keepdim=True)
    if not bool((torch.isfinite(new_denominator)&(new_denominator>0)).all()):
        raise FloatingPointError('Nonpositive/nonfinite softmax denominator; no silent fallback')
    return new_maximum,new_denominator,alpha,probabilities


def attention(q,k,v,config=AttentionConfig(),*,query_offset=0):
    if q.device.type not in ('cpu','npu','privateuseone') or any(x.device!=q.device for x in (k,v)):
        raise ValueError('Expected matching CPU or explicitly authorized NPU devices')
    if any(x.ndim!=4 or not x.is_floating_point() for x in (q,k,v)):
        raise ValueError('Expected floating [batch,head,sequence,feature] tensors')
    if any(not bool(torch.isfinite(x).all()) for x in (q,k,v)):
        raise ValueError('Nonfinite attention input')
    if q.dtype!=k.dtype or q.dtype!=v.dtype:
        raise ValueError('Input dtypes must agree')
    b,h,n,d=q.shape; nk=k.shape[2]
    if (min(b,h,n,d)<1 or k.shape!=v.shape or (k.shape[0],k.shape[3])!=(b,d)
            or k.shape[1]<1 or h%k.shape[1] or not isinstance(query_offset,int)
            or query_offset<0 or query_offset+n>nk):
        raise ValueError('Invalid attention/GQA/position shapes')
    def quant(x): return c4_qdq(x.float()) if config.quantize_operands else x.float()
    def vr(x): return vector_round(x,config.vector)
    qq,kk,vv=(quant(x) for x in (q,k,v))
    repeats=h//k.shape[1]
    if repeats!=1:
        kk=kk.repeat_interleave(repeats,1);vv=vv.repeat_interleave(repeats,1)
    result=[]
    for qs in range(0,n,config.query_block):
        qe=min(n,qs+config.query_block);qi=qq[:,:,qs:qe]
        state=(b,h,qe-qs,1)
        maximum=torch.full(state,-math.inf,dtype=torch.float32,device=q.device)
        denominator=torch.zeros(state,dtype=torch.float32,device=q.device)
        accumulator=torch.zeros((b,h,qe-qs,d),dtype=torch.float32,device=q.device)
        stop=query_offset+qe if config.causal else nk
        for ks in range(0,stop,config.key_block):
            ke=min(stop,ks+config.key_block)
            scores=qi@kk[:,:,ks:ke].transpose(-1,-2)/math.sqrt(d)
            if config.causal:
                qi_positions=torch.arange(query_offset+qs,query_offset+qe,device=q.device).unsqueeze(-1)
                ki_positions=torch.arange(ks,ke,device=q.device).unsqueeze(0)
                causal_mask=ki_positions>qi_positions
                if not bool(torch.isfinite(scores.masked_fill(causal_mask,0)).all()):
                    raise FloatingPointError('Nonfinite unmasked QK score; not a structural mask')
                scores=scores.masked_fill(causal_mask,-math.inf)
            elif not bool(torch.isfinite(scores).all()):
                raise FloatingPointError('Nonfinite QK score; not a structural mask')
            # Scale must not observe a masked future key, even if its logit is huge.
            scores=vector_round(scores,config.vector,allow_structural_negative_infinity=config.causal)
            new_maximum,denominator,alpha,probabilities=online_softmax_step(scores,maximum,denominator,config)
            contribution=quant(probabilities)@vv[:,:,ks:ke]
            accumulator=vr(vr(accumulator*alpha)+vr(contribution))
            maximum=new_maximum
        result.append(vr(accumulator/denominator).to(q.dtype))
    output=torch.cat(result,2)
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError('Nonfinite reference attention output; no silent fallback')
    return output


def operation_estimate(sequence_length,config=AttentionConfig(),layers=48):
    """Static call-count estimate, not a performance prediction or kernel count."""
    qtiles=(sequence_length+config.query_block-1)//config.query_block
    pairs=sum(((min(sequence_length,(i+1)*config.query_block)+config.key_block-1)//config.key_block)
              if config.causal else ((sequence_length+config.key_block-1)//config.key_block)
              for i in range(qtiles))
    tree_levels=math.ceil(math.log2(config.key_block)) if config.key_block>1 else 0
    vector_per_pair=(13+2*tree_levels if config.reduction=='mxfp8_tree_reduction' else 8)
    return dict(sequence_length=sequence_length,query_tiles=qtiles,query_key_tile_pairs_per_layer=pairs,
                matmuls_per_layer=2*pairs,c4_qdq_calls_per_layer=3+pairs,
                approximate_vector_qdq_calls_per_layer=vector_per_pair*pairs+qtiles,
                layers=layers,approximate_vector_qdq_calls_all_layers=layers*(vector_per_pair*pairs+qtiles),
                warning='Counts Python QDQ calls, each expands into many torch ops and strict device checks; not measured NPU throughput')
