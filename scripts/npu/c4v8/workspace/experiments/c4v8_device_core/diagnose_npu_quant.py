"""Bounded NPU1 format diagnostic; no model, no Ray, no production changes."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,re,subprocess,time
from pathlib import Path
import torch
from device_quant import _power_two,round_e4m3fn,e4m3fn_bytes,round_e2m1,mxfp8_values


def compare(a,b):
    a=a.detach().cpu();b=b.detach().cpu();d=a.float()-b.float()
    nonzero=(a!=b)
    indices=nonzero.nonzero()[:16]
    return {'exact':torch.equal(a,b),'different_count':int(nonzero.sum()),
            'total':a.numel(),'max_abs':float(d.abs().max()),
            'examples':[{'index':i.tolist(),'npu':float(a[tuple(i)]),'cpu':float(b[tuple(i)])} for i in indices]}


def bit_power_two(exponents):
    exp=exponents.to(torch.int32)
    bits=torch.where(exp==-127,torch.full_like(exp,1<<22),torch.bitwise_left_shift(exp+127,23))
    return bits.contiguous().view(torch.float32)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    snap=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
    (a.output/'npu-before.txt').write_text(snap.stdout+snap.stderr)
    idle=sorted(set(int(i) for i in re.findall(r'No running processes found in NPU\s+(\d+)',snap.stdout)))
    if snap.returncode or idle!=list(range(8)):raise RuntimeError('NPU idle gate failed')
    import torch_npu
    torch.set_num_threads(1);torch.npu.set_device(1);device=torch.device('npu:1')
    torch.npu.matmul.allow_hf32=False;torch.npu.set_per_process_memory_fraction(.05,device)
    report={};start=time.perf_counter()
    e=torch.arange(-127,128,dtype=torch.float32)
    report['native_power_two']=compare(torch.pow(torch.full_like(e.to(device),2.0),e.to(device)),_power_two(e))
    report['implemented_power_two']=compare(_power_two(e.to(device)),_power_two(e))
    report['bit_power_two']=compare(bit_power_two(e.to(device)),_power_two(e))
    x=torch.arange(-16,17,dtype=torch.float32)/2
    report['round_halfway']=compare(torch.round(x.to(device)),torch.round(x))
    codes=torch.arange(256,dtype=torch.uint8);values=codes.view(torch.float8_e4m3fn).float()
    values=values[torch.isfinite(values)];positive=values[values>=0].unique().sort().values
    mids=(positive[:-1]+positive[1:])/2
    x=torch.cat([values,mids,-mids,torch.nextafter(mids,torch.zeros_like(mids)),torch.nextafter(mids,torch.full_like(mids,float('inf')))])
    report['e4m3_decoded']=compare(round_e4m3fn(x.to(device)),round_e4m3fn(x))
    report['e4m3_codes']=compare(e4m3fn_bytes(x.to(device)),e4m3fn_bytes(x))
    mid4=torch.tensor([0.,-0.,.25,.75,1.25,1.75,2.5,3.5,5.,7.])
    report['e2m1_values']=compare(round_e2m1(mid4.to(device)),round_e2m1(mid4))
    x=torch.tensor([[1.,.5,.25,.125,.0625,.03125,.015625,.0078125]]).repeat(5,1)
    x=x*torch.tensor([.5,1,2,4,16]).unsqueeze(-1)
    report['dyadic_mxfp8']=compare(mxfp8_values(x.to(device)).values,mxfp8_values(x).values)
    transitions=[]
    for exponent in (-127,-120,-40,-1,0,1,20,100,117):
        point=torch.tensor(448.*2.**exponent,dtype=torch.float32)
        vals=torch.stack([torch.nextafter(point,torch.tensor(0.)),point,
                          torch.nextafter(point,torch.tensor(float('inf')))]).view(-1,1)
        cpu=mxfp8_values(vals,logical_bytes=True);npu=mxfp8_values(vals.to(device),logical_bytes=True)
        transitions.append({'exponent':exponent,'scales':compare(npu.scale_bytes,cpu.scale_bytes),
                            'payload':compare(npu.payload_bytes,cpu.payload_bytes),
                            'values':compare(npu.values,cpu.values)})
    report['mxfp8_scale_transitions']=transitions
    report['elapsed_seconds']=time.perf_counter()-start
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
