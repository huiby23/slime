"""Independent unvalidated staged decode candidate; never registered in SGLang.

Every128-key online quantization boundary remains. P8 controls the denominator;
P4 controls PV. FP32 GEMMs and scalar arithmetic are simulation, not native
low-precision hardware. Forward-only: no implicit backward substitution.
"""
import hashlib,math,os
from pathlib import Path
import torch
import triton
from staged_kernels import scores_local_max,maximum_scan,probability_pass,probability_sum,final_state_scan
from nvfp4_triton import quantized_values
from forward_api import qdq_forward

_BINARY_CACHE={}
LAST_KERNEL_IDENTITIES={}


def _record(name,compiled):
    if os.environ.get('HBY_C4V8_AUDIT_KERNEL_CACHE')!='1':return
    if compiled.hash not in _BINARY_CACHE:
        _BINARY_CACHE[compiled.hash]=hashlib.sha256(compiled.kernel).hexdigest()
    LAST_KERNEL_IDENTITIES[name]={'triton_hash':compiled.hash,'binary_sha256':_BINARY_CACHE[compiled.hash]}


def source_fingerprints():
    import fused_mxfp8,nvfp4_triton,forward_api,staged_kernels
    paths=[Path(__file__),*(Path(module.__file__) for module in [staged_kernels,fused_mxfp8,nvfp4_triton,forward_api])]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def staged_decode_forward(q,k,v,*,check=True,return_intermediates=False):
    if os.environ.get('HBY_C4V8_ALLOW_UNVALIDATED_FUSION')!='1':
        raise RuntimeError('Explicit unvalidated candidate opt-in required')
    if q.device.type not in ('npu','privateuseone') or any(x.device!=q.device for x in (k,v)):
        raise ValueError('Matching NPU tensors required; no CPU fallback')
    if any(x.dtype!=q.dtype or x.dtype not in (torch.float32,torch.bfloat16) or x.ndim!=4 for x in (q,k,v)):
        raise ValueError('Expected matching FP32/BF16 BHND Q/K/V')
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,v)):
        raise ValueError('Forward-only candidate has no backward implementation')
    if torch.npu.matmul.allow_hf32:raise RuntimeError('HF32 must be disabled for FP32 accumulation')
    batch,heads,queries,dim=q.shape;kvheads=k.shape[1];seq=k.shape[2]
    if queries!=1 or dim!=128 or k.shape!=v.shape or k.shape[0]!=batch or k.shape[-1]!=dim or min(batch,heads,kvheads,seq)<1 or heads%kvheads:
        raise ValueError('Expected decode Q[B,H,1,128], K/V[B,HK,SEQ,128], integer GQA')
    block=128;state_stride=32;tiles=triton.cdiv(seq,block);padded=tiles*block;rows=batch*heads
    input_bad=(~(torch.isfinite(q).all()&torch.isfinite(k).all()&torch.isfinite(v).all())).to(torch.int32).reshape(1)
    q4,k4,v4=(quantized_values(x.float().contiguous(),check=False)[0] for x in (q,k,v))
    if heads!=kvheads:
        repeats=heads//kvheads;k4=k4.repeat_interleave(repeats,1);v4=v4.repeat_interleave(repeats,1)
    # One full-sequence FP32 QK. Its backend accumulation can differ from
    # tile-shaped GEMM; audit this separately instead of changing thresholds.
    raw_scores=(q4@k4.transpose(-1,-2)/math.sqrt(dim)).reshape(rows,seq)
    raw_scores=torch.nn.functional.pad(raw_scores,(0,padded-seq),value=-math.inf).contiguous()
    score8=torch.empty((rows,padded),device=q.device,dtype=torch.float32)
    #128-byte physical state rows keep separate programs from sharing short
    #DMA destinations. This is a candidate precaution, not a proven root cause.
    local_m=torch.empty((rows,tiles,state_stride),device=q.device,dtype=torch.float32)
    maximum=torch.empty_like(local_m);alpha=torch.empty_like(local_m);local_l=torch.empty_like(local_m)
    p8=torch.empty_like(score8)
    flags_a=torch.empty((rows,tiles,state_stride),device=q.device,dtype=torch.int32)
    flags_b=torch.empty_like(flags_a);flags_c=torch.empty_like(flags_a);flags_d=torch.empty_like(flags_a)
    compiled=scores_local_max[(rows*tiles,)](raw_scores,score8,local_m,flags_a,tiles,seq,block,state_stride,enable_fp_fusion=False)
    _record('scores_local_max',compiled)
    compiled=maximum_scan[(rows,)](local_m,maximum,alpha,flags_b,tiles,state_stride,enable_fp_fusion=False)
    _record('maximum_scan',compiled)
    compiled=probability_pass[(rows*tiles,)](score8,maximum,p8,flags_c,block,state_stride,enable_fp_fusion=False)
    _record('probability_pass',compiled)
    p4=quantized_values(p8.reshape(rows,tiles,block),check=False)[0]
    compiled=probability_sum[(rows*tiles,)](p8,local_l,flags_d,block,state_stride,enable_fp_fusion=False)
    _record('probability_sum',compiled)
    v_stored=v4.reshape(rows,seq,dim)
    if padded!=seq:v_stored=torch.nn.functional.pad(v_stored,(0,0,0,padded-seq))
    raw_contribution=torch.matmul(p4.unsqueeze(-2),v_stored.reshape(rows,tiles,block,dim)).squeeze(-2).contiguous()
    contribution,_,_,flags_pv=qdq_forward(raw_contribution,check=False)
    output=torch.empty((rows,dim),device=q.device,dtype=torch.float32)
    denominator=torch.empty((rows,state_stride),device=q.device,dtype=torch.float32)
    flags_final=torch.empty((rows,tiles,state_stride),device=q.device,dtype=torch.int32)
    flags_normalize=torch.empty((rows,state_stride),device=q.device,dtype=torch.int32)
    compiled=final_state_scan[(rows,)](alpha,local_l,contribution,output,denominator,flags_final,flags_normalize,tiles,dim,state_stride,enable_fp_fusion=False)
    _record('final_state_scan',compiled)
    result=output.reshape(batch,heads,1,dim).to(q.dtype)
    result_bad=(~torch.isfinite(result).all()).to(torch.int32).reshape(1)
    flags=torch.cat([input_bad,flags_a[...,:3].reshape(-1),flags_b[...,:5].reshape(-1),flags_c[...,:3].reshape(-1),flags_d[...,0].reshape(-1),flags_pv.reshape(-1),flags_final[...,:5].reshape(-1),flags_normalize[...,0].reshape(-1),result_bad])
    if check and bool((flags!=0).any()):raise FloatingPointError('Staged C4V8 decode invalid values/arithmetic; no fallback')
    intermediates=None
    if return_intermediates:
        intermediates={'q4':q4,'k4':k4,'v4':v4,'raw_scores':raw_scores,'score8':score8,'local_m':local_m[...,0],'maximum':maximum[...,0],
                       'alpha':alpha[...,0],'p8':p8,'p4':p4,'local_l':local_l[...,0],'raw_contribution':raw_contribution,
                       'contribution':contribution,'denominator':denominator[:,:1],'output_fp32':output,
                       'stage_flags':{'score':flags_a[...,:3],'max_scan':flags_b[...,:5],'probability':flags_c[...,:3],'sum':flags_d[...,0],'pv':flags_pv,'final':flags_final[...,:5],'normalize':flags_normalize[...,0]}}
    return result,flags,intermediates


def metadata():
    return {'candidate':True,'production_registered':False,'npu_validated':False,'native_low_precision_hardware':False,
            'scope':'Forward-only decode, original NVFP4/MXFP8 boundaries, fixed128 logical key tiles, head128',
            'p8':'MXFP8 probabilities used by MXFP8 tree sum and denominator',
            'p4':'NVFP4 probabilities used by PV only',
            'runtime_dimensions':['SEQ','TILES','grid rows'],'physical_state_stride_elements':32,'source_fingerprints':source_fingerprints(),
            'accumulation_audit':'Full-sequence QK and batched padded128-key PV may differ in FP32 rounding from original per-tile GEMM',
            'memory':'Linear in sequence length; expanded GQA K/V retained in this initial candidate; no persistent KV cache',
            'error_checks':'Independent per-stage/per-tile GM slots, every initialized slot checked; no external flag OR held across complex trees',
            'backward':'Not implemented; caller with grad-requiring input is rejected'}
