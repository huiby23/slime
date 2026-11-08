"""NPU1-only inference/autograd probe, never calls portable reference on NPU."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import argparse,hashlib,json,re,subprocess,sys,time,traceback
from pathlib import Path
import torch

def main(out,r):
    before=subprocess.check_output(['npu-smi','info'],text=True)
    (out/'npu-before.txt').write_text(before)
    idle=sorted(set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)',before))))
    if idle!=list(range(8)):raise RuntimeError('All-eight idle gate failed')
    torch.set_num_threads(1)
    sys.path.insert(0,str(Path(__file__).parent.parent/'c4v8_device_core'))
    from device_quant import nvfp4_values
    g=torch.Generator().manual_seed(44413)
    x=torch.randn(512,128,generator=g)
    print('PHASE cpu_reference_begin',flush=True)
    expected=nvfp4_values(x,logical_bytes=True)
    print('PHASE cpu_reference_end',flush=True)
    import torch_npu
    torch.npu.set_device(1);torch.npu.set_per_process_memory_fraction(.05,1)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.matmul.allow_hf32=False
    from nvfp4_triton import quantized_values,nvfp4_qdq,_qdq
    r['jit_compile']=False;r['hf32']=bool(torch.npu.matmul.allow_hf32);r['cases']=[]
    r['dynamic_rows_cache']=[]
    for rows in (13,17,257,13):
        print('PHASE dynamic_rows_cache_begin rows='+str(rows),flush=True)
        before_keys={hashlib.sha256(str(k).encode()).hexdigest() for k in _qdq.cache[1]}
        y,codes,scales=quantized_values(x[:rows].to('npu:1'),logical_bytes=True)
        torch.npu.synchronize(1)
        after_keys={hashlib.sha256(str(k).encode()).hexdigest() for k in _qdq.cache[1]}
        item={'rows':rows,'new_specializations':len(after_keys-before_keys),'cache_key_sha256':sorted(after_keys),
              'values_exact':torch.equal(y.cpu().view(torch.int32),expected.values[:rows].view(torch.int32)),
              'codes_exact':torch.equal(codes.cpu(),expected.payload_bytes[:rows]),
              'scales_exact':torch.equal(scales.cpu(),expected.scale_bytes[:rows])}
        r['dynamic_rows_cache'].append(item)
        if not all(item[k] for k in ('values_exact','codes_exact','scales_exact')):raise AssertionError('dynamic row mismatch')
        if len(r['dynamic_rows_cache'])>1 and item['new_specializations']!=0:raise AssertionError('NROWS still recompiles the kernel')
        print('PHASE dynamic_rows_cache_end rows='+str(rows),flush=True)
    for inference in (False,True):
        print('PHASE fused_only_begin inference='+str(inference),flush=True)
        with torch.inference_mode(inference):
            npu=x.to('npu:1');y,codes,scales=quantized_values(npu,logical_bytes=True)
            torch.npu.synchronize(1)
            item={'inference':inference,'values_exact':torch.equal(y.cpu().view(torch.int32),expected.values.view(torch.int32)),
                  'codes_exact':torch.equal(codes.cpu(),expected.payload_bytes),'scales_exact':torch.equal(scales.cpu(),expected.scale_bytes)}
        r['cases'].append(item)
        if not all(item[k] for k in ('values_exact','codes_exact','scales_exact')):raise AssertionError('fused-only numerical mismatch')
        print('PHASE fused_only_end inference='+str(inference),flush=True)
    print('PHASE fused_ste_begin',flush=True)
    npu=x.to('npu:1').requires_grad_();dy=torch.randn_like(npu)
    dx=torch.autograd.grad(nvfp4_qdq(npu),npu,dy)[0]
    r['ste_identity_exact']=torch.equal(dx,dy)
    if not r['ste_identity_exact']:raise AssertionError('STE identity changed')
    print('PHASE fused_ste_end',flush=True)
    start=time.perf_counter()
    for _ in range(10):quantized_values(npu,logical_bytes=True)
    torch.npu.synchronize(1);r['ten_repeat_mean_seconds']=(time.perf_counter()-start)/10
    r['peak_allocated_bytes']=int(torch.npu.max_memory_allocated(1))
    r['numerical_passed']=True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    r={'numerical_passed':False,'backend_fallback_audit':'Inspect full process stdout/stderr separately; script never invokes portable NPU code.',
       'device':'npu:1','sources':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['nvfp4_triton.py','validate_fused_only.py']}}
    for name in r['sources']:(a.output/('source-'+name)).write_bytes((Path(__file__).parent/name).read_bytes())
    start=time.perf_counter();code=1
    try:main(a.output,r);code=0
    except BaseException as e:r['error']=type(e).__name__+': '+str(e);traceback.print_exc()
    finally:
        r['elapsed_seconds']=time.perf_counter()-start
        (a.output/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
        (a.output/'exit').write_text(str(code)+'\n')
        print(json.dumps(r),flush=True)
    sys.exit(code)
