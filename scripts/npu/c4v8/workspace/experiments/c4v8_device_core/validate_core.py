"""CPU acceptance / explicitly authorized single-NPU numerical experiment."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import unittest

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MAX_COMPILE_CORE_NUMBER','1')
import torch
from device_quant import nvfp4_values,mxfp8_values,metadata
from device_attention import attention,AttentionConfig,operation_estimate
from exact_fp32 import divide_positive


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def synchronize(device):
    if device.type != 'cpu':torch.npu.synchronize(device)


def metrics(a,b):
    x=a.detach().float().cpu();y=b.detach().float().cpu()
    return {'exact':bool(torch.equal(x,y)),'finite':bool(torch.isfinite(x).all()),
            'max_abs':float((x-y).abs().max()),
            'relative_l2':float((x-y).norm()/y.norm().clamp_min(1e-30))}


def run(args,report):
    torch.set_num_threads(1)
    if args.device == 'cpu':
        suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test_device_core.py')
        with (args.output/'unittest.log').open('w') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        report['tests']={'run':result.testsRun,'passed':result.wasSuccessful(),
                         'failures':[(str(t),msg) for t,msg in result.failures+result.errors]}
        report['passed']=result.wasSuccessful()
        return
    if args.device != 'npu:1' or not args.authorize_npu1:
        raise ValueError('Only explicit --device npu:1 --authorize-npu1 is permitted')
    snapshot=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
    (args.output/'npu-before.txt').write_text(snapshot.stdout+snapshot.stderr)
    idle=sorted(set(int(s) for s in re.findall(r'No running processes found in NPU\s+(\d+)',snapshot.stdout)))
    if snapshot.returncode or idle!=list(range(8)):
        raise RuntimeError('All-eight-idle gate failed; no NPU initialized')
    report['all_eight_idle_before']=True
    import torch_npu
    device=torch.device(args.device)
    torch.npu.set_device(device)
    report['torch_npu']=str(torch_npu.__version__)
    report['packages']={name:importlib.util.find_spec(name) is not None for name in ('torchair','triton','triton_ascend')}
    if hasattr(torch.npu,'matmul') and hasattr(torch.npu.matmul,'allow_hf32'):
        torch.npu.matmul.allow_hf32=False
        report['allow_hf32']=False
    else:
        report['allow_hf32']='API not present; no precision toggle made'
    torch.npu.set_per_process_memory_fraction(0.05,device)
    report['npu_memory_fraction_limit']=0.05
    division_generator=torch.Generator().manual_seed(1339)
    report['scale_division']=[]
    for numerator in (1.0,2688.0,None):
        den=torch.randint(1,0x7f800000,(65536,),generator=division_generator,dtype=torch.int32).view(torch.float32)
        num=(torch.randint(1,0x7f800000,(65536,),generator=division_generator,dtype=torch.int32).view(torch.float32)
             if numerator is None else torch.full_like(den,numerator))
        got=divide_positive(num.to(device),den.to(device)).cpu()
        expected=num/den
        different=int((got.view(torch.int32)!=expected.view(torch.int32)).sum())
        report['scale_division'].append({'numerator':numerator,'cases':65536,'bit_differences':different})
        save(args.output/'report.partial.json',report)
        if different:raise AssertionError('Exact FP32 scale division does not match CPU RNE')
    g=torch.Generator().manual_seed(20261108)
    x=torch.randn(3,67,generator=g)*7
    report['quantizers']={}
    for name,fn in [('nvfp4',nvfp4_values),('mxfp8',mxfp8_values)]:
        cpu=fn(x,logical_bytes=True)
        start=time.perf_counter();actual=fn(x.to(device),logical_bytes=True);synchronize(device)
        item=metrics(actual.values,cpu.values)
        item['payload_exact']=bool(torch.equal(actual.payload_bytes.cpu(),cpu.payload_bytes))
        item['scale_exact']=bool(torch.equal(actual.scale_bytes.cpu(),cpu.scale_bytes))
        item['first_call_seconds']=time.perf_counter()-start
        if actual.upper_encode is not None:
            item['upper_encode']=metrics(actual.upper_encode,cpu.upper_encode)
            item['upper_decode']=metrics(actual.upper_decode,cpu.upper_decode)
        item['value_acceptance']='exact logical bytes and reconstructed values against CPU format reference; not native Blackwell parity'
        report['quantizers'][name]=item
        save(args.output/'report.partial.json',report)
        torch.testing.assert_close(actual.values.cpu(),cpu.values,rtol=0,atol=0)
        if not(item['exact'] and item['payload_exact'] and item['scale_exact']):
            raise AssertionError(f'{name} NPU quantization differs from CPU reference: {item}')
    report['attention']=[]
    for length in args.lengths:
        if length<1 or length>256:raise ValueError('NPU smoke lengths must be 1..256')
        config=AttentionConfig(query_block=min(16,length),key_block=32)
        dtype=getattr(torch,args.dtype)
        cpu=tuple(torch.randn(1,h,length,args.head_dim,generator=g).to(dtype).requires_grad_()
                  for h in (args.heads,args.kv_heads,args.kv_heads))
        acc=tuple(t.detach().to(device).requires_grad_() for t in cpu)
        expected=attention(*cpu,config)
        expected.float().square().mean().backward()
        synchronize(device);start=time.perf_counter()
        actual=attention(*acc,config);synchronize(device)
        forward_seconds=time.perf_counter()-start
        start=time.perf_counter();actual.float().square().mean().backward();synchronize(device)
        backward_seconds=time.perf_counter()-start
        item={'length':length,'input_dtype':args.dtype,'heads':args.heads,'kv_heads':args.kv_heads,
              'head_dim':args.head_dim,'config':config.metadata(),'forward_seconds':forward_seconds,
              'backward_seconds':backward_seconds,'output':metrics(actual,expected),
              'gradients':[metrics(a.grad,b.grad) for a,b in zip(acc,cpu)],
              'allocated_bytes':int(torch.npu.memory_allocated(device)),
              'peak_allocated_bytes':int(torch.npu.max_memory_allocated(device))}
        item['gradient_acceptance']=('one BF16 ULP or abs<=1e-8 after FP32 STE backend arithmetic'
                                     if dtype==torch.bfloat16 else 'rtol=2e-5,atol=1e-8 for FP32 STE backend arithmetic')
        report['attention'].append(item)
        save(args.output/'report.partial.json',report)
        if not item['output']['finite'] or not all(m['finite'] for m in item['gradients']):
            raise AssertionError('NPU attention produced nonfinite output/gradients')
        torch.testing.assert_close(actual.cpu(),expected,atol=0,rtol=0)
        for aa,bb in zip(acc,cpu):
            if dtype==torch.bfloat16:
                got=aa.grad.cpu();ref=bb.grad
                low=torch.nextafter(ref,torch.full_like(ref,-float('inf')))
                high=torch.nextafter(ref,torch.full_like(ref,float('inf')))
                accepted=((got>=low)&(got<=high))|((got.float()-ref.float()).abs()<=1e-8)
                if not bool(accepted.all()):raise AssertionError('BF16 STE gradient differs by more than one ULP / 1e-8')
            else:
                torch.testing.assert_close(aa.grad.cpu(),bb.grad,atol=1e-8,rtol=2e-5)
        del acc,cpu,actual,expected
    if args.layout_checks:
        vals=tuple(torch.randn(1,h,32,128,generator=g).to(torch.bfloat16).to(device) for h in (8,1,1))
        config=AttentionConfig(query_block=16,key_block=32)
        with torch.no_grad():
            full=attention(*vals,config)
            split=torch.cat([attention(vals[0][:,i:i+4],vals[1],vals[2],config) for i in (0,4)],1)
            decoded=torch.cat([attention(vals[0][:,:,i:i+1],vals[1][:,:,:i+1],vals[2][:,:,:i+1],config,query_offset=i)
                               for i in range(32)],2)
        report['layout_checks']={'shape':[1,8,32,128],'dtype':'bfloat16',
                                 'tp_head_split':metrics(split,full),'prefill_decode':metrics(decoded,full)}
        save(args.output/'report.partial.json',report)
        torch.testing.assert_close(split,full,atol=0,rtol=0)
        torch.testing.assert_close(decoded,full,atol=0,rtol=0)
    report['passed']=True
    report['acceptance_scope']='Portable QDQ and tiny attention outputs exact against CPU; STE gradients within explicit FP32 tolerance; not 30B or native low-bit parity'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu')
    p.add_argument('--authorize-npu1',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lengths',type=int,nargs='+',default=[32,64,128])
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    p.add_argument('--heads',type=int,default=4);p.add_argument('--kv-heads',type=int,default=2)
    p.add_argument('--head-dim',type=int,default=32)
    p.add_argument('--layout-checks',action='store_true');args=p.parse_args()
    if not(1<=args.heads<=8 and 1<=args.kv_heads<=8 and args.heads%args.kv_heads==0 and 1<=args.head_dim<=128):
        p.error('Bounded test shapes require 1..8 heads, divisible GQA, head_dim 1..128')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'started_utc':datetime.now(timezone.utc).isoformat(),'device':args.device,
            'torch':str(torch.__version__),'python':sys.version,'precision':metadata(),
            'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
            'estimate_4096':operation_estimate(4096,AttentionConfig(query_block=128,key_block=128)),
            'passed':False}
    start=time.perf_counter();code=1
    try:
        run(args,report);code=0 if report['passed'] else 1
    except BaseException as error:
        report['error']=type(error).__name__+': '+str(error)
        (args.output/'failure-traceback.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        save(args.output/'report.json',report);(args.output/'exit').write_text(str(code)+'\n')
        print(json.dumps({'output':str(args.output),'passed':report['passed'],'elapsed_seconds':report['elapsed_seconds']}))
    sys.exit(code)
