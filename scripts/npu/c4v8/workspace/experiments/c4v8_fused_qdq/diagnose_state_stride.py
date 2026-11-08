"""NPU1 probe of scalar state load/store alignment versus grid size."""
import os,sys,json,re,subprocess,hashlib
from pathlib import Path
os.environ.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',TRITON_CACHE_DIR='/root/hby/cache/triton-c4v8',TMPDIR='/root/hby/tmp',MAX_COMPILE_CORE_NUMBER='1',OMP_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parent/'c4v8_fused_nvfp4')]
import torch,triton
import triton.language as tl
from fused_mxfp8 import _mx_vector,_mx_scalar,_tree,_finite
from cpu_oracle import qdq,tree

@triton.jit
def maximum_stride(X,Old,Y,Flags,Broadcast,STRIDE:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,128);one=tl.arange(0,1)
    score=tl.load(X+row*128+i);old=tl.load(Old+row*STRIDE+one)
    score,flags=_mx_vector(score,128,True)
    value,flag=_tree(score,128,128,7,True);flags=flags|flag
    maximum,flag=_mx_scalar(tl.maximum(old,value),True);flags=flags|flag
    flags=flags|tl.where(_finite(maximum),0,4)
    tl.store(Y+row*STRIDE+one,maximum);tl.store(Flags+row*STRIDE+one,flags)
    tl.store(Broadcast+row*128+i,maximum)

def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
    before=subprocess.check_output(['npu-smi','info'],text=True);(out/'npu-before.txt').write_text(before)
    assert sorted(set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)',before))))==list(range(8))
    for name in ('diagnose_state_stride.py','fused_mxfp8.py'):(out/('source-'+name)).write_bytes((ROOT/name).read_bytes())
    import torch_npu
    torch.set_num_threads(1);torch.npu.set_device(1);torch.npu.set_per_process_memory_fraction(.05,1);torch.npu.matmul.allow_hf32=False;torch.npu.set_compile_mode(jit_compile=False)
    report={'tests':[]};g=torch.Generator().manual_seed(1111)
    source=torch.randn((16,128),generator=g);source[::2,65:]=-float('inf')
    for rows in (4,8,16):
        x=source[:rows].contiguous();expected=qdq(tree(qdq(x,True).values,'max'),True).values
        for stride in (1,32):
            old=torch.full((rows,stride),-float('inf'),device='npu:1');y=torch.full_like(old,-123.);flags=torch.full((rows,stride),-99,dtype=torch.int32,device='npu:1');b=torch.empty((rows,128),device='npu:1')
            maximum_stride[(rows,)](x.to('npu:1'),old,y,flags,b,stride,num_warps=4,enable_fp_fusion=False)
            yc=y.cpu();fc=flags.cpu();bc=b.cpu()
            result={'rows':rows,'stride':stride,'scalar_equal':torch.equal(yc[:,:1].view(torch.int32),expected.view(torch.int32)),'broadcast_equal':torch.equal(bc.view(torch.int32),expected.expand(-1,128).contiguous().view(torch.int32)),'scalar':yc[:,0].tolist(),'expected':expected[:,0].tolist(),'flags':fc[:,0].tolist()}
            report['tests'].append(result);print(json.dumps(result),flush=True)
            torch.save({'y':yc,'flags':fc,'b':bc,'x':x,'expected':expected},out/f'rows{rows}-stride{stride}.pt')
            (out/'report.partial.json').write_text(json.dumps(report,indent=2))
    (out/'report.json').write_text(json.dumps(report,indent=2));print('DONE',flush=True)
if __name__=='__main__':main()
