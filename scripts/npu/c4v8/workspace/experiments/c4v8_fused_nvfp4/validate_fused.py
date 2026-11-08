"""Explicitly gated NPU1 validation of the independent Triton candidate."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MAX_COMPILE_CORE_NUMBER','1')
import argparse,hashlib,json,re,subprocess,sys,time,traceback
from pathlib import Path
import torch


def save(path,r):Path(path).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')


def main(args,r):
    snap=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
    (args.output/'npu-before.txt').write_text(snap.stdout+snap.stderr)
    idle=sorted(set(int(i) for i in re.findall(r'No running processes found in NPU\s+(\d+)',snap.stdout)))
    if snap.returncode or idle!=list(range(8)):raise RuntimeError('All-eight idle gate failed')
    import torch_npu
    torch.npu.set_device(1);torch.npu.set_per_process_memory_fraction(.05,1);torch.set_num_threads(1)
    sys.path.insert(0,str(Path(__file__).parent.parent/'c4v8_device_core'))
    from device_quant import nvfp4_values
    from nvfp4_triton import division_probe,quantized_values,nvfp4_qdq
    r['division']=[];g=torch.Generator().manual_seed(1339)
    for numerator in (1.,2688.,None):
        den=torch.randint(1,0x7f800000,(65536,),generator=g,dtype=torch.int32).view(torch.float32)
        num=(torch.randint(1,0x7f800000,(65536,),generator=g,dtype=torch.int32).view(torch.float32)
             if numerator is None else torch.full_like(den,numerator))
        start=time.perf_counter();got=division_probe(num.to('npu:1'),den.to('npu:1')).cpu()
        expected=num/den;different=got.view(torch.int32)!=expected.view(torch.int32)
        item={'numerator':numerator,'cases':65536,'different':int(different.sum()),'seconds':time.perf_counter()-start}
        if different.any():
            indices=different.nonzero().flatten()[:8]
            item['examples']=[{'numerator':float(num[i]),'denominator':float(den[i]),'actual':float(got[i]),'expected':float(expected[i])} for i in indices]
        r['division'].append(item);save(args.output/'report.partial.json',r)
        if item['different']:raise AssertionError('Triton integer-significand division is not CPU RNE-equivalent')
    cases=[]
    for dtype in (torch.float32,torch.bfloat16):
        for width in (1,5,16,31,32,35,127,128,129):cases.append((f'random-{dtype}-{width}',torch.randn(13,width,generator=g).to(dtype)))
    cases.append(('zero',torch.zeros(5,128)))
    cases.append(('tiny',torch.tensor([[0.,-0.,1e-44,1e-40,1e-35]])))
    trace_path=Path(__file__).parent.parent/'c4v8_device_core/evidence/npu-stage-trace-v1/trace.pt'
    if trace_path.exists():
        trace=torch.load(trace_path,weights_only=False)
        cases.append(('real-P-halfway-counterexample',trace['cpu_events'][46]['input']))
    r['quantization']=[]
    for name,x in cases:
        expected=nvfp4_values(x,logical_bytes=True)
        start=time.perf_counter();y,codes,scales=quantized_values(x.to('npu:1'),logical_bytes=True)
        torch.npu.synchronize(1)
        actual=y.cpu();item={'name':name,'shape':list(x.shape),'dtype':str(x.dtype),
                            'values_exact':torch.equal(actual.view(torch.int32),expected.values.view(torch.int32)),
                            'codes_exact':torch.equal(codes.cpu(),expected.payload_bytes),
                            'scales_exact':torch.equal(scales.cpu(),expected.scale_bytes),
                            'max_abs':float((actual-expected.values).abs().max()),
                            'seconds':time.perf_counter()-start}
        r['quantization'].append(item);save(args.output/'report.partial.json',r)
        if not all(item[k] for k in ('values_exact','codes_exact','scales_exact')):
            torch.save({'input':x,'actual_values':actual,'actual_codes':codes.cpu(),'actual_scales':scales.cpu(),
                        'expected_values':expected.values,'expected_codes':expected.payload_bytes,'expected_scales':expected.scale_bytes},
                       args.output/'quantization-failure-tensors.pt')
            raise AssertionError('Fused NVFP4 disagrees with portable exact reference')
    r['inference_mode']=[]
    for width in (35,128):
        x=torch.randn(9,width,generator=g)
        expected=nvfp4_values(x,logical_bytes=True)
        with torch.inference_mode():
            y,codes,scales=quantized_values(x.to('npu:1'),logical_bytes=True)
            actual=y.cpu()
        item={'width':width,'values_exact':torch.equal(actual.view(torch.int32),expected.values.view(torch.int32)),
              'codes_exact':torch.equal(codes.cpu(),expected.payload_bytes),'scales_exact':torch.equal(scales.cpu(),expected.scale_bytes)}
        r['inference_mode'].append(item)
        if not all(item[k] for k in ('values_exact','codes_exact','scales_exact')):raise AssertionError('inference_mode mismatch')
    bad=torch.zeros(1,128,device='npu:1');bad[0,3]=float('nan')
    try:quantized_values(bad)
    except ValueError:r['checked_invalid_rejected']=True
    else:raise AssertionError('checked invalid input accepted')
    propagated=quantized_values(bad,check=False)[0]
    r['unchecked_invalid_row_nan']=bool(torch.isnan(propagated).all())
    if not r['unchecked_invalid_row_nan']:raise AssertionError('invalid input silently discarded')
    x=torch.randn(7,128,generator=g).to('npu:1').requires_grad_();dy=torch.randn_like(x)
    y=nvfp4_qdq(x);dx=torch.autograd.grad(y,x,dy)[0]
    r['ste_identity_exact']=torch.equal(dx,dy)
    if not r['ste_identity_exact']:raise AssertionError('Fused STE gradient changed')
    x=torch.randn(512,128,generator=g).to('npu:1')
    # Warm both candidates before the measured repeats; identical strict checks.
    for fn in (lambda:nvfp4_values(x),lambda:quantized_values(x)):
        fn();torch.npu.synchronize(1)
    timings={}
    for name,fn in [('portable',lambda:nvfp4_values(x)),('fused',lambda:quantized_values(x))]:
        start=time.perf_counter()
        for _ in range(5):fn()
        torch.npu.synchronize(1);timings[name]=(time.perf_counter()-start)/5
    r['timing_512x128_seconds']=timings;r['speedup']=timings['portable']/timings['fused']
    r['peak_allocated_bytes']=int(torch.npu.max_memory_allocated(1));r['passed']=True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    r={'passed':False,'device':'npu:1','native_fp4_hardware':False,'full_model_test':False,
       'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}}
    for filename,expected_sha in r['sources'].items():
        source=(Path(__file__).parent/filename).read_bytes()
        assert hashlib.sha256(source).hexdigest()==expected_sha
        (args.output/('source-'+filename)).write_bytes(source)
    start=time.perf_counter();code=1
    try:main(args,r);code=0
    except BaseException as e:
        r['error']=type(e).__name__+': '+str(e);(args.output/'failure-traceback.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        r['elapsed_seconds']=time.perf_counter()-start;save(args.output/'report.json',r)
        (args.output/'exit').write_text(str(code)+'\n');print(json.dumps({'passed':r['passed'],'elapsed_seconds':r['elapsed_seconds'],'output':str(args.output)}))
    sys.exit(code)
