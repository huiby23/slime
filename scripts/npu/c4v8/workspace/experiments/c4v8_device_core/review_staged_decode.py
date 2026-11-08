"""CPU proof for a staged C4V8 decode fallback; no NPU implementation/claim."""
from pathlib import Path
import hashlib,json,math
import torch
from device_attention import attention,AttentionConfig,vector_round,mxfp8_tree_reduce
from device_quant import nvfp4_qdq


def staged(q,k,v,config):
    vr=lambda x,masked=False:vector_round(x,'mxfp8',allow_structural_negative_infinity=masked)
    q4,k4,v4=(nvfp4_qdq(x.float()) for x in (q,k,v))
    if q.shape[1]!=k.shape[1]:
        repeats=q.shape[1]//k.shape[1];k4=k4.repeat_interleave(repeats,1);v4=v4.repeat_interleave(repeats,1)
    # PhaseA: independent tile scores and local max tree. Keep original GEMM
    # shapes/order in this proof so it tests scheduling, not alternate BLAS.
    tiles=[]
    for ks in range(0,k.shape[2],config.key_block):
        ke=min(k.shape[2],ks+config.key_block)
        score=vr(q4@k4[:,:,ks:ke].transpose(-1,-2)/math.sqrt(q.shape[-1]))
        local_m=mxfp8_tree_reduce(score,'max')
        tiles.append({'score':score,'local_m':local_m,'ks':ks,'ke':ke})
    # PhaseB: ordered maximum-only scan; this recurrence does not depend on L/O.
    maximum=torch.full((*q.shape[:-1],1),-math.inf)
    for tile in tiles:
        new_m=vr(torch.maximum(maximum,tile['local_m']),True)
        tile['new_m']=new_m
        tile['alpha']=vr(torch.exp(vr(maximum-new_m,True)))
        maximum=new_m
    # PhaseC: independent probabilities, P4, sum-tree(P8), and tile PV. All
    # original boundaries remain. P8 and P4 are distinct mathematical arrays.
    for tile in tiles:
        p8=vr(torch.exp(vr(tile['score']-tile['new_m'],True)))
        p4=nvfp4_qdq(p8)
        tile['local_l']=mxfp8_tree_reduce(p8,'sum')
        tile['contribution']=vr(p4@v4[:,:,tile['ks']:tile['ke']])
    # PhaseD: ordered L/O scans using exactly the recorded alphas. Neither may
    # be replaced with one FP32/global reduction. Final division uses final L.
    denominator=torch.zeros((*q.shape[:-1],1));accumulator=torch.zeros(q.shape)
    for tile in tiles:
        denominator=vr(vr(denominator*tile['alpha'])+tile['local_l'])
        accumulator=vr(vr(accumulator*tile['alpha'])+tile['contribution'])
    return vr(accumulator/denominator).to(q.dtype)


def main():
    torch.set_num_threads(1);generator=torch.Generator().manual_seed(19019)
    config=AttentionConfig(query_block=64,key_block=128);results=[]
    for dtype in (torch.float32,torch.bfloat16):
        for length in (17,33,128,129,256):
            q=torch.randn(1,8,1,128,generator=generator).to(dtype)
            k=torch.randn(1,1,length,128,generator=generator).to(dtype)
            v=torch.randn(1,1,length,128,generator=generator).to(dtype)
            expected=attention(q,k,v,config,query_offset=length-1)
            actual=staged(q,k,v,config)
            exact=torch.equal(actual.float().view(torch.int32),expected.float().view(torch.int32))
            row={'dtype':str(dtype),'length':length,'output_bit_exact':exact,'max_abs':float((actual.float()-expected.float()).abs().max())}
            results.append(row)
            assert exact,row
    out=Path(__file__).parent/'evidence/staged-decode-cpu-scheduling-proof.json'
    out.write_text(json.dumps({'passed':True,'scope':'CPU scheduling factorization only, same underlying per-tile GEMM operations; no NPU code or speed claim','cases':results,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({'passed':True,'cases':len(results),'path':str(out)}))

if __name__=='__main__':main()
