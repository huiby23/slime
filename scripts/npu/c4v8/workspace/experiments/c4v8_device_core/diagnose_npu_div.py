import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,re,subprocess
from pathlib import Path
import torch
from device_quant import round_e4m3fn,round_e2m1


def stages(x,reciprocal):
    rows=x.reshape(-1,x.shape[-1]);blocks=rows.reshape(rows.shape[0],-1,16)
    amax=rows.abs().amax(-1,keepdim=True)
    encode=torch.div(torch.full_like(amax,2688.),amax)
    decode=reciprocal(encode)
    multiplier=encode.unsqueeze(-1)*torch.div(torch.ones_like(encode),torch.full_like(encode,6.)).unsqueeze(-1)
    pre=blocks.abs().amax(-1,keepdim=True)*multiplier
    scale=round_e4m3fn(pre.clamp(0,448))
    bd=scale*decode.unsqueeze(-1)
    be=reciprocal(bd).clamp_max(torch.finfo(torch.float32).max)
    norm=(blocks*be).clamp(-6,6)
    val,code=round_e2m1(norm,return_codes=True)
    return dict(amax=amax,encode=encode,decode=decode,pre_scale=pre,scale=scale,
                block_decode=bd,block_encode=be,normalized=norm,codes=code)


def rcp(x):return torch.div(torch.ones_like(x),x)
def nr(x):
    y=rcp(x)
    return y+y*(1-x*y)
def nr2(x):
    y=nr(x)
    return y+y*(1-x*y)
def fma(x):
    y=rcp(x)
    return y+y*torch.addcmul(torch.ones_like(x),-x,y)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    snap=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
    (a.output/'npu-before.txt').write_text(snap.stdout+snap.stderr)
    idle=sorted(set(int(i) for i in re.findall(r'No running processes found in NPU\s+(\d+)',snap.stdout)))
    if snap.returncode or idle!=list(range(8)):raise RuntimeError('NPU idle gate failed')
    import torch_npu
    torch.set_num_threads(1);torch.npu.set_device(1);torch.npu.set_per_process_memory_fraction(.05,1)
    trace=torch.load(Path(__file__).parent/'evidence/npu-stage-trace-v1/trace.pt',weights_only=False)
    x=trace['cpu_events'][46]['input'];ref=stages(x,rcp)
    report={}
    for name,fn in [('div',rcp),('reciprocal',torch.reciprocal),('newton1',nr),('newton2',nr2),('addcmul',fma)]:
        actual=stages(x.to('npu:1'),fn);row={}
        for key,v in actual.items():
            v=v.cpu();r=ref[key];different=v!=r
            row[key]={'different':int(different.sum()),'total':v.numel(),
                      'examples':[{'index':i.tolist(),'npu':float(v[tuple(i)]),'cpu':float(r[tuple(i)])}
                                  for i in different.nonzero()[:4]]}
        report[name]=row
    # Probe whether integer division used for exact rounding is actually exact.
    numerator=torch.tensor([1<<47,(1<<47)+1,(1<<48)-1],dtype=torch.int64)
    denominator=torch.tensor([(1<<23)+1,12345679,(1<<24)-1],dtype=torch.int64)
    got=torch.div(numerator.to('npu:1'),denominator.to('npu:1'),rounding_mode='floor').cpu()
    expected=torch.div(numerator,denominator,rounding_mode='floor')
    report['int64_division']={'exact':torch.equal(got,expected),'npu':got.tolist(),'cpu':expected.tolist()}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
