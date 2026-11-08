"""Explicit single NPU1 tree diagnostic; CPU-reference only, source manifests."""
import os,sys,json,hashlib,math,re,subprocess,time
from pathlib import Path
os.environ.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',TRITON_CACHE_DIR='/root/hby/cache/triton-c4v8',TMPDIR='/root/hby/tmp',MAX_COMPILE_CORE_NUMBER='1',OMP_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parent/'c4v8_fused_nvfp4')]
import torch,triton
import triton.language as tl
from triton.triton_patch.language.core import gather as gather
from fused_mxfp8 import _mx_vector,_mx_scalar,_tree
from cpu_oracle import qdq,tree

@triton.jit
def gather_probe(X,Y,BLOCK:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK)
    x=tl.load(X+row*BLOCK+i)
    a=gather(x,(2*i)%BLOCK,0);b=gather(x,(2*i+1)%BLOCK,0)
    z=gather(x,tl.full((1,),0,tl.int32),0)
    tl.store(Y+row*3*BLOCK+i,a);tl.store(Y+row*3*BLOCK+BLOCK+i,b)
    tl.store(Y+row*3*BLOCK+2*BLOCK+i,z)

@triton.jit
def tree_stages(X,Y,F,BLOCK:tl.constexpr,LEVELS:tl.constexpr,IS_MAX:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK)
    identity:tl.constexpr=-float('inf') if IS_MAX else 0.
    x=tl.load(X+row*BLOCK+i);x,flag=_mx_vector(x,BLOCK,IS_MAX)
    tl.store(Y+row*(LEVELS+1)*BLOCK+i,x);tl.store(F+row*(LEVELS+1),flag)
    for level in tl.static_range(0,LEVELS):
        a=gather(x,(2*i)%BLOCK,0);b=gather(x,(2*i+1)%BLOCK,0)
        p=tl.maximum(a,b) if IS_MAX else a+b
        p=tl.where(i<(BLOCK>>(level+1)),p,identity)
        x,flag=_mx_vector(p,BLOCK,IS_MAX)
        tl.store(Y+row*(LEVELS+1)*BLOCK+(level+1)*BLOCK+i,x)
        tl.store(F+row*(LEVELS+1)+level+1,flag)

@triton.jit
def tree_final(X,Old,S,Broadcast,F,BLOCK:tl.constexpr,LEVELS:tl.constexpr,IS_MAX:tl.constexpr,WITH_OLD:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK);one=tl.arange(0,1)
    x=tl.load(X+row*BLOCK+i)
    value,flag=_tree(x,BLOCK,BLOCK,LEVELS,IS_MAX)
    if WITH_OLD:
        old=tl.load(Old+row+one)
        value=tl.maximum(old,value) if IS_MAX else old+value
        value,flag2=_mx_scalar(value,IS_MAX);flag=flag|flag2
    tl.store(S+row+one,value);tl.store(Broadcast+row*BLOCK+i,value);tl.store(F+row+one,flag)

def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
    before=subprocess.check_output(['npu-smi','info'],text=True);(out/'npu-before.txt').write_text(before)
    assert sorted(set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)',before))))==list(range(8))
    for name in ('diagnose_tree.py','fused_mxfp8.py'):(out/('source-'+name)).write_bytes((ROOT/name).read_bytes())
    report={'tests':[],'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'diagnose_tree.py',ROOT/'fused_mxfp8.py')},'portable_npu_reference':False}
    def check(name,a,b):
        a=a.cpu();b=b.cpu();r={'name':name,'bit_equal':torch.equal(a.view(torch.int32),b.view(torch.int32)),'different':int((a.view(torch.int32)!=b.view(torch.int32)).sum())}
        report['tests'].append(r);print(json.dumps(r),flush=True);(out/'report.partial.json').write_text(json.dumps(report,indent=2))
    import torch_npu
    torch.set_num_threads(1);torch.npu.set_device(1);torch.npu.set_per_process_memory_fraction(.05,1);torch.npu.matmul.allow_hf32=False;torch.npu.set_compile_mode(jit_compile=False)
    gen=torch.Generator().manual_seed(1111)
    for block in (32,64,128):
        x=torch.arange(4*block,dtype=torch.float32).reshape(4,block);nx=x.to('npu:1');y=torch.empty((4,3,block),device='npu:1')
        gather_probe[(4,)](nx,y,block,enable_fp_fusion=False);ix=torch.arange(block)
        check(f'gather-{block}',y,torch.stack((x[:,2*ix%block],x[:,(2*ix+1)%block],x[:,:1].expand(-1,block)),dim=1))
        for op in ('max','sum'):
            x=torch.randn((4,block),generator=gen)
            if op=='max':x[::2,block//2+1:]=-float('inf')
            else:x=x.abs()
            nx=x.to('npu:1');levels=int(math.log2(block));y=torch.empty((4,levels+1,block),device='npu:1');flags=torch.empty((4,levels+1),device='npu:1',dtype=torch.int32)
            tree_stages[(4,)](nx,y,flags,block,levels,op=='max',enable_fp_fusion=False)
            part=qdq(x,op=='max').values;expected=[part];identity=-float('inf') if op=='max' else 0.
            for lv in range(levels):
                l=part[:,2*ix%block];r=part[:,(2*ix+1)%block];p=torch.maximum(l,r) if op=='max' else l+r
                part=qdq(torch.where(ix<(block>>(lv+1)),p,identity),op=='max').values;expected.append(part)
            expected=torch.stack(expected,dim=1)
            torch.save({'x':x,'actual':y.cpu(),'expected':expected,'flags':flags.cpu()},out/f'{op}-{block}.pt')
            for lv in range(levels+1):check(f'tree-stages-{op}-{block}-level{lv}',y[:,lv].contiguous(),expected[:,lv].contiguous())
            old=torch.full((4,1),-float('inf') if op=='max' else 0.);ns=torch.empty((4,1),device='npu:1');nb=torch.empty((4,block),device='npu:1');nf=torch.empty((4,1),device='npu:1',dtype=torch.int32)
            for wo in (False,True):
                tree_final[(4,)](nx,old.to('npu:1'),ns,nb,nf,block,levels,op=='max',wo,enable_fp_fusion=False)
                check(f'tree-final-{op}-{block}-old{wo}-scalar',ns,part[:,:1].contiguous());check(f'tree-final-{op}-{block}-old{wo}-broadcast',nb,part[:,:1].expand(-1,block).contiguous())
    report['passed']=all(t['bit_equal'] for t in report['tests']);(out/'report.json').write_text(json.dumps(report,indent=2));print('DONE',report['passed'],flush=True)
if __name__=='__main__':main()
