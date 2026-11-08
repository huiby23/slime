"""Authorized NPU1 short stage diagnostic; values remain in this experiment."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['TRITON_CACHE_DIR']='/root/hby/cache/triton-c4v8'
os.environ['TMPDIR']='/root/hby/tmp'
os.environ['MAX_COMPILE_CORE_NUMBER']='1'
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parent/'c4v8_fused_nvfp4'),str(ROOT.parent/'c4v8_device_core')]
import torch,triton
import triton.language as tl
from fused_mxfp8 import _mx_vector,_mx_scalar,_tree


@triton.jit
def stage_probe(Scores,Output,BLOCK:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK)
    raw=tl.load(Scores+row*BLOCK+i)
    s,f=_mx_vector(raw,BLOCK,True)
    m,f=_tree(s,BLOCK,BLOCK,7,True)
    m,f=_mx_scalar(m,True)
    diff=s-m
    dq,f=_mx_vector(diff,BLOCK,True)
    exp=tl.exp(dq)
    p,f=_mx_vector(exp,BLOCK,False)
    tl.store(Output+row*6*BLOCK+i,s)
    tl.store(Output+row*6*BLOCK+BLOCK+i,m)
    tl.store(Output+row*6*BLOCK+2*BLOCK+i,diff)
    tl.store(Output+row*6*BLOCK+3*BLOCK+i,dq)
    tl.store(Output+row*6*BLOCK+4*BLOCK+i,exp)
    tl.store(Output+row*6*BLOCK+5*BLOCK+i,p)


def main():
    out=ROOT/'evidence/vector-stage-diag-v1'
    out.mkdir(exist_ok=False)
    import torch_npu
    torch.npu.set_device(1);torch.npu.set_per_process_memory_fraction(.05,1)
    torch.npu.matmul.allow_hf32=False;torch.npu.set_compile_mode(jit_compile=False)
    generator=torch.Generator().manual_seed(1111)
    x=torch.randn((8,128),generator=generator);x[::2,65:]=-float('inf')
    y=torch.empty((8,6,128),device='npu:1')
    stage_probe[(8,)](x.to('npu:1'),y,128,enable_fp_fusion=False)
    result=y.cpu();torch.save({'input':x,'stages':result},out/'stages.pt')
    for i,name in enumerate(('score_qdq','max','difference','difference_qdq','exp','p_qdq')):
        print(name,'first16',result[0,i,:16].tolist(),'nonfinite',int((~torch.isfinite(result[:,i])).sum()),flush=True)


if __name__=='__main__':main()
