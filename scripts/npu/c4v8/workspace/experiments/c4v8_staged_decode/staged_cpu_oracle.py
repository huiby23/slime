"""CPU-only staged scheduling/BLAS oracle; not a fallback execution backend."""
import math
from pathlib import Path
import sys
import torch
CORE=Path(__file__).resolve().parent.parent/'c4v8_device_core'
if str(CORE) not in sys.path:sys.path.insert(0,str(CORE))
from device_attention import vector_round,mxfp8_tree_reduce
from device_quant import nvfp4_qdq


def staged_cpu(q,k,v,*,batched_gemm=True,return_intermediates=False):
    if any(x.device.type!='cpu' for x in (q,k,v)):raise ValueError('CPU reference only')
    if q.ndim!=4 or q.shape[2]!=1 or q.shape[-1]!=128 or k.shape!=v.shape or q.shape[1]%k.shape[1]:raise ValueError('Expected head128 causal GQA decode')
    batch,heads,_,dim=q.shape;seq=k.shape[2];rows=batch*heads;tiles=(seq+127)//128;padded=tiles*128
    vr=lambda x,masked=False:vector_round(x,'mxfp8',allow_structural_negative_infinity=masked)
    q4,k4,v4=(nvfp4_qdq(x.float()) for x in (q,k,v))
    if heads!=k.shape[1]:
        repeat=heads//k.shape[1];k4=k4.repeat_interleave(repeat,1);v4=v4.repeat_interleave(repeat,1)
    if batched_gemm:raw=(q4@k4.transpose(-1,-2)/math.sqrt(dim)).reshape(rows,seq)
    else:raw=torch.cat([q4@k4[:,:,start:min(seq,start+128)].transpose(-1,-2)/math.sqrt(dim) for start in range(0,seq,128)],-1).reshape(rows,seq)
    raw_scores=torch.nn.functional.pad(raw,(0,padded-seq),value=-math.inf).reshape(rows,tiles,128)
    score8=vr(raw_scores,True)
    local_m=mxfp8_tree_reduce(score8,'max').squeeze(-1)
    maximum=torch.empty_like(local_m);alpha=torch.empty_like(local_m)
    previous=torch.full((rows,1),-math.inf)
    for tile in range(tiles):
        current=vr(torch.maximum(previous,local_m[:,tile:tile+1]),True)
        maximum[:,tile:tile+1]=current
        alpha[:,tile:tile+1]=vr(torch.exp(vr(previous-current,True)))
        previous=current
    p8=vr(torch.exp(vr(score8-maximum.unsqueeze(-1),True)))
    p4=nvfp4_qdq(p8)
    local_l=mxfp8_tree_reduce(p8,'sum').squeeze(-1)
    values=v4.reshape(rows,seq,dim)
    if batched_gemm:
        stored=torch.nn.functional.pad(values,(0,0,0,padded-seq)).reshape(rows,tiles,128,dim)
        raw_contribution=torch.matmul(p4.unsqueeze(-2),stored).squeeze(-2)
    else:
        parts=[]
        for tile in range(tiles):
            start=tile*128;stop=min(seq,start+128);length=stop-start
            parts.append(p4[:,tile:tile+1,:length]@values[:,start:stop])
        raw_contribution=torch.cat(parts,1)
    contribution=vr(raw_contribution)
    denominator=torch.zeros((rows,1));accumulator=torch.zeros((rows,dim))
    for tile in range(tiles):
        a=alpha[:,tile:tile+1]
        denominator=vr(vr(denominator*a)+local_l[:,tile:tile+1])
        accumulator=vr(vr(accumulator*a)+contribution[:,tile])
    output_fp32=vr(accumulator/denominator)
    output=output_fp32.reshape(batch,heads,1,dim).to(q.dtype)
    details={'q4':q4,'k4':k4,'v4':v4,'raw_scores':raw_scores.reshape(rows,padded),'score8':score8.reshape(rows,padded),
             'local_m':local_m,'maximum':maximum,'alpha':alpha,'p8':p8.reshape(rows,padded),'p4':p4,'local_l':local_l,
             'raw_contribution':raw_contribution,'contribution':contribution,'denominator':denominator,'output_fp32':output_fp32}
    return (output,details) if return_intermediates else output
