"""Whole-decode forward oracle with explicit16-key FP32 GEMV accumulation.

The only intended change to the existing operation graph is FP32 QK/PV
accumulation order. C4 P scale still observes the COMPLETE logical key tile;
microchunks do not become online-softmax or quantization tiles.
"""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'c4v8_device_core'))
from device_quant import nvfp4_values
from device_attention import vector_round,online_softmax_step,AttentionConfig


@torch.no_grad()
def decode(q,k,v,config=AttentionConfig(key_block=128),*,micro_keys=16,return_audit=False):
    if q.device.type!='cpu' or any(x.device!=q.device for x in (k,v)):
        raise ValueError('This implementation is a CPU semantics oracle')
    if q.ndim!=4 or q.shape[2]!=1 or k.shape!=v.shape or k.ndim!=4:
        raise ValueError('Decode expects Q[B,H,1,D], K/V[B,HK,N,D]')
    b,h,_,d=q.shape;hk=k.shape[1];n=k.shape[2]
    if h%hk or k.shape[0]!=b or k.shape[-1]!=d or config.key_block%micro_keys:
        raise ValueError('Invalid GQA/microchunk dimensions')
    if config.vector!='mxfp8' or not config.quantize_operands or config.reduction!='mxfp8_tree_reduction':
        raise ValueError('Full C4V8 profile required')
    qq,kk,vv=(nvfp4_values(x.float()).values for x in (q,k,v))
    kk=kk.repeat_interleave(h//hk,1);vv=vv.repeat_interleave(h//hk,1)
    maximum=torch.full((b,h,1,1),-float('inf'));denominator=torch.zeros_like(maximum)
    out=torch.zeros((b,h,1,d));audits=[]
    vr=lambda x:vector_round(x,'mxfp8')
    for start in range(0,n,config.key_block):
        stop=min(n,start+config.key_block);keys=kk[:,:,start:stop];values=vv[:,:,start:stop]
        scores=torch.cat([(qq*keys[:,:,s:s+micro_keys]).sum(-1) for s in range(0,stop-start,micro_keys)],-1)
        scores=scores.unsqueeze(-2)/(d**0.5)
        reference_scores=qq@keys.transpose(-1,-2)/(d**0.5)
        raw_scores=scores.clone()
        scores=vr(scores)
        maximum,denominator,alpha,p=online_softmax_step(scores,maximum,denominator,config)
        p4=nvfp4_values(p).values
        contribution=torch.zeros_like(out)
        for s in range(0,stop-start,micro_keys):
            # FP32 reduction within16keys, then ordered FP32 microchunk add.
            contribution=contribution+(p4[...,s:s+micro_keys].squeeze(-2).unsqueeze(-1)*values[:,:,s:s+micro_keys]).sum(-2).unsqueeze(-2)
        reference_contribution=p4@values
        out=vr(vr(out*alpha)+vr(contribution))
        if return_audit:
            audits.append({'key_start':start,'key_stop':stop,
                           'qk_float_max_abs':float((raw_scores-reference_scores).abs().max()),
                           'qk_float_rel_l2':float(torch.linalg.vector_norm(raw_scores-reference_scores)/torch.linalg.vector_norm(reference_scores).clamp_min(1e-30)),
                           'score_quant_mismatch_count':int((vr(reference_scores)!=scores).sum()),
                           'pv_float_max_abs':float((contribution-reference_contribution).abs().max()),
                           'pv_float_rel_l2':float(torch.linalg.vector_norm(contribution-reference_contribution)/torch.linalg.vector_norm(reference_contribution).clamp_min(1e-30)),
                           'pv_quant_mismatch_count':int((vr(contribution)!=vr(reference_contribution)).sum())})
    result=vr(out/denominator).to(q.dtype)
    return (result,audits) if return_audit else result
