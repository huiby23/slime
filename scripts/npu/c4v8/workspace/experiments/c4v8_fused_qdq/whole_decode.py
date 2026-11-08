"""Unvalidated whole-decode C4V8 kernel with fixed online-keytile semantics.

One program owns one batch/query-head row and loops all logical key tiles.
QK/PV use16-key microchunks solely for FP32 arithmetic/storage pressure. P C4
sees the COMPLETE keytile; all Vector/tree QDQ boundaries remain intact.
No import-time launch, active-backend installation or quantized-backward claim.
"""
import os
import hashlib
import torch
import triton
import triton.language as tl
from triton.triton_patch.language.core import gather as ascend_gather
from nvfp4_triton import nvfp4_value_2d,_divide_positive_rne
from fused_mxfp8 import _finite,_mx_vector,_pre_p_vector

LAST_KERNEL_IDENTITY=None
_BINARY_IDENTITIES={}


@triton.jit(do_not_specialize=['SEQ'])
def _whole_decode(Q,K,V,Output,Flags,H:tl.constexpr,HK:tl.constexpr,
                  SEQ,D:tl.constexpr,KEY_TILE:tl.constexpr,
                  MICRO:tl.constexpr,LEVELS:tl.constexpr,PREQUANTIZED:tl.constexpr):
    row=tl.program_id(0);batch=row//H;head=row%H;kv_head=head//(H//HK)
    features=tl.arange(0,D);key_lanes=tl.arange(0,KEY_TILE);micro_lanes=tl.arange(0,MICRO)
    query=tl.load(Q+row*D+features).to(tl.float32)
    flags=tl.max((~_finite(query)).to(tl.int32),0)*32
    if not PREQUANTIZED:
        query,invalid=nvfp4_value_2d(tl.reshape(query,(1,D)))
        flags=flags|tl.max(invalid.to(tl.int32),0)*32
        query=tl.reshape(query,(D,))
    maximum=tl.full((1,),-float('inf'),tl.float32)
    denominator=tl.full((1,),0.0,tl.float32)
    accumulator=tl.full((D,),0.0,tl.float32)
    for tile in range(tl.cdiv(SEQ,KEY_TILE)):
        tile_start=tile*KEY_TILE
        scores=tl.full((KEY_TILE,),-float('inf'),tl.float32)
        for micro in range(KEY_TILE//MICRO):
            logical=tile_start+micro*MICRO+micro_lanes
            address=((batch*HK+kv_head)*SEQ+logical[:,None])*D+features[None,:]
            keys=tl.load(K+address,logical[:,None]<SEQ,other=0.0).to(tl.float32)
            flags=flags|tl.max(tl.max((~_finite(keys)).to(tl.int32),1),0)*32
            if not PREQUANTIZED:
                keys,invalid=nvfp4_value_2d(keys)
                flags=flags|tl.max(invalid.to(tl.int32),0)*32
            partial=tl.sum(keys*query[None,:],1)*(1.0/tl.sqrt(float(D)))
            # Replicate16-lane fragment to fulltile, then replace only its own
            # contiguous slice. Does not advance the online softmax boundary.
            repeated=ascend_gather(partial,key_lanes%MICRO,0)
            scores=tl.where((key_lanes>=micro*MICRO)&(key_lanes<(micro+1)*MICRO),repeated,scores)
        valid=tile_start+key_lanes<SEQ
        flags=flags|tl.max(tl.where(valid&(~_finite(scores)),32,0),0)
        scores=tl.where(valid,scores,-float('inf'))
        maximum,denominator,alpha,p,flag=_pre_p_vector(scores,maximum,denominator,KEY_TILE,KEY_TILE,LEVELS)
        flags=flags|flag
        p4,invalid=nvfp4_value_2d(tl.reshape(p,(1,KEY_TILE)))
        p4=tl.reshape(p4,(KEY_TILE,));flags=flags|tl.max(invalid.to(tl.int32),0)*64
        contribution=tl.full((D,),0.0,tl.float32)
        for micro in range(KEY_TILE//MICRO):
            logical=tile_start+micro*MICRO+micro_lanes
            address=((batch*HK+kv_head)*SEQ+logical[:,None])*D+features[None,:]
            values=tl.load(V+address,logical[:,None]<SEQ,other=0.0).to(tl.float32)
            flags=flags|tl.max(tl.max((~_finite(values)).to(tl.int32),1),0)*32
            if not PREQUANTIZED:
                values,invalid=nvfp4_value_2d(values)
                flags=flags|tl.max(invalid.to(tl.int32),0)*32
            probabilities=ascend_gather(p4,micro*MICRO+micro_lanes,0)
            contribution=contribution+tl.sum(probabilities[:,None]*values,0)
        old,flag=_mx_vector(accumulator*alpha,D,False);flags=flags|flag
        current,flag=_mx_vector(contribution,D,False);flags=flags|flag
        accumulator,flag=_mx_vector(old+current,D,False);flags=flags|flag
    flags=flags|tl.where(_finite(denominator)&(denominator>0.0),0,16)
    normalized=_divide_positive_rne(tl.abs(accumulator),denominator)
    normalized=(normalized.to(tl.int32,bitcast=True)|(accumulator.to(tl.int32,bitcast=True)&-2147483648)).to(tl.float32,bitcast=True)
    output,flag=_mx_vector(normalized,D,False);flags=flags|flag
    converted=output.to(Output.dtype.element_ty)
    flags=flags|tl.max((~_finite(converted.to(tl.float32))).to(tl.int32),0)*2
    tl.store(Output+row*D+features,converted)
    tl.store(Flags+row+tl.arange(0,1),flags)


def whole_decode_forward(q,k,v,*,key_tile=128,micro_keys=16,prequantized=False,check=True):
    if os.environ.get('HBY_C4V8_ALLOW_UNVALIDATED_FUSION')!='1':
        raise RuntimeError('Whole-decode candidate requires explicit unvalidated opt-in')
    if q.device.type not in ('npu','privateuseone') or any(x.device!=q.device for x in (k,v)):
        raise ValueError('This candidate is NPU-only')
    if any(x.dtype!=q.dtype or not x.is_contiguous() for x in (q,k,v)):
        raise ValueError('Matching contiguous dtypes required')
    if q.dtype not in (torch.float32,torch.bfloat16):raise ValueError('BF16/FP32 only')
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,v)):
        raise ValueError('Forward-only candidate; no implicit unquantized/missing backward')
    if q.ndim!=4 or q.shape[2]!=1 or k.ndim!=4 or k.shape!=v.shape:
        raise ValueError('Decode expects Q[B,H,1,D], K/V[B,HK,N,D]')
    b,h,_,d=q.shape;hk=k.shape[1];n=k.shape[2]
    if d!=128 or key_tile!=128 or micro_keys!=16:
        raise ValueError('First bounded candidate fixes D128/K128/micro16')
    if k.shape[0]!=b or k.shape[-1]!=d or hk<1 or h%hk or n<1:
        raise ValueError('Invalid GQA/cache shape')
    if prequantized and q.dtype!=torch.float32:
        raise ValueError('Prequantized operands must be exact dequantized FP32 values')
    out=torch.empty_like(q)
    flags=torch.empty((b*h,),dtype=torch.int32,device=q.device)
    compiled=_whole_decode[(b*h,)](q,k,v,out,flags,h,hk,n,d,key_tile,micro_keys,7,prequantized,
                          num_warps=4,enable_fp_fusion=False)
    if os.environ.get('HBY_C4V8_AUDIT_KERNEL_CACHE')=='1':
        # Pinned3.2 CompiledKernel exposes hash and binary bytes; hash the binary
        # once, not on every inference call. This does not inspect tensor data.
        global LAST_KERNEL_IDENTITY
        if compiled.hash not in _BINARY_IDENTITIES:
            _BINARY_IDENTITIES[compiled.hash]=hashlib.sha256(compiled.kernel).hexdigest()
        LAST_KERNEL_IDENTITY={'triton_hash':compiled.hash,
            'binary_sha256':_BINARY_IDENTITIES[compiled.hash],
            'sequence_length':n,'prequantized':prequantized,
            'sequence_is_runtime_argument':True,'do_not_specialize_sequence':True}
    if check and bool((flags!=0).any()):
        raise FloatingPointError('Whole-decode C4V8 candidate flagged invalid arithmetic; no fallback')
    return out,flags
