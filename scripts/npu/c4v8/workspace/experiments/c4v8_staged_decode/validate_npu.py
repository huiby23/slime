"""Explicitly authorized NPU1 diagnostic only. Never launches itself on import."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MAX_COMPILE_CORE_NUMBER','1')
import argparse,hashlib,json,re,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path[:0]=[str(ROOT),str(BASE/'c4v8_device_core'),str(BASE/'c4v8_fused_nvfp4'),str(BASE/'c4v8_fused_qdq')]
import torch
from staged_cpu_oracle import staged_cpu
from device_attention import attention,AttentionConfig
from validate_cpu import compare


def json_save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(args,r):
    snapshot=subprocess.check_output(['npu-smi','info'],text=True)
    (args.output/'npu-before.txt').write_text(snapshot)
    if sorted(set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)',snapshot))))!=list(range(8)):
        raise RuntimeError('All8 NPUs must be idle before this explicitly authorized probe')
    torch.set_num_threads(1);generator=torch.Generator().manual_seed(208814)
    config=AttentionConfig(query_block=64,key_block=128)
    prepared=[]
    print('PHASE cpu_reference_only_begin',flush=True)
    for dtype,seq in [(torch.bfloat16,n) for n in args.lengths]+[(torch.float32,args.lengths[0])]:
        q=torch.randn(1,8,1,128,generator=generator).to(dtype)
        k=torch.randn(1,1,seq,128,generator=generator).to(dtype)
        v=torch.randn(1,1,seq,128,generator=generator).to(dtype)
        expected=attention(q,k,v,config,query_offset=seq-1)
        staged,details=staged_cpu(q,k,v,batched_gemm=True,return_intermediates=True)
        prepared.append({'qkv':[q,k,v],'expected':expected,'staged_expected':staged,'details':details,'seq':seq,'dtype':str(dtype)})
    torch.save(prepared,args.output/'cpu-references.pt')
    print('PHASE cpu_reference_only_end',flush=True)
    import torch_npu
    torch.npu.set_device(1);torch.npu.set_per_process_memory_fraction(.05,1)
    torch.npu.set_compile_mode(jit_compile=False);torch.npu.matmul.allow_hf32=False
    os.environ['HBY_C4V8_ALLOW_UNVALIDATED_FUSION']='1';os.environ['HBY_C4V8_AUDIT_KERNEL_CACHE']='1'
    import staged_decode as candidate
    r['metadata']=candidate.metadata();r['hf32']=False;r['jit_compile']=False;r['cases']=[]
    baseline_kernel_ids=None;observations={}
    with torch.inference_mode():
        for index,case in enumerate(prepared):
            print('PHASE staged_candidate_begin seq='+str(case['seq'])+' dtype='+case['dtype'],flush=True)
            qkv=[x.to('npu:1') for x in case['qkv']]
            torch.npu.synchronize(1);started=time.perf_counter()
            # Preserve every flag and stage on first failure, then reject it.
            output,flags,details=candidate.staged_decode_forward(*qkv,check=False,return_intermediates=True)
            torch.npu.synchronize(1);elapsed=time.perf_counter()-started
            actual={key:( {k:v.cpu() for k,v in value.items()} if isinstance(value,dict) else value.cpu()) for key,value in details.items()}
            actual['flags']=flags.cpu();actual['output']=output.cpu()
            torch.save(actual,args.output/('case-'+str(index)+'-stages.pt'))
            identity=dict(candidate.LAST_KERNEL_IDENTITIES)
            row={'seq':case['seq'],'dtype':case['dtype'],'seconds_including_first_shape_compilation':elapsed,
                 'nonzero_flags':int((actual['flags']!=0).sum()),'flags_unique':actual['flags'].unique().tolist(),
                 'output_vs_original_cpu':compare(actual['output'],case['expected']),
                 'output_vs_batched_cpu':compare(actual['output'],case['staged_expected']),
                 'stages_vs_batched_cpu':{key:compare(actual[key],expected) for key,expected in case['details'].items()},
                 'kernel_identities':identity}
            if baseline_kernel_ids is None:baseline_kernel_ids=identity
            row['same_kernel_identities_as_first_sequence']=identity==baseline_kernel_ids
            r['cases'].append(row);json_save(args.output/'report.partial.json',r)
            if row['nonzero_flags']:raise FloatingPointError('Staged candidate set error flags; stage tensors retained')
            if not bool(torch.isfinite(actual['output']).all()):raise FloatingPointError('Staged candidate output nonfinite')
            # Explicit head partition test stays on the candidate, never portable NPU.
            parts=[candidate.staged_decode_forward(qkv[0][:,start:start+4].contiguous(),qkv[1],qkv[2])[0] for start in (0,4)]
            row['head4_partition_vs_head8']=compare(torch.cat(parts,1).cpu(),actual['output'])
            started=time.perf_counter()
            for _ in range(3):candidate.staged_decode_forward(*qkv)
            torch.npu.synchronize(1);row['three_repeat_mean_seconds']=(time.perf_counter()-started)/3
            json_save(args.output/'report.partial.json',r)
            print('PHASE staged_candidate_end seq='+str(case['seq']),flush=True)
        bad_q=qkv[0].clone();bad_q[0,0,0,0]=float('nan')
        try:candidate.staged_decode_forward(bad_q,qkv[1],qkv[2])
        except FloatingPointError:r['nonfinite_input_rejected']=True
        else:raise AssertionError('Nonfinite candidate input accepted')
    # This check occurs before device execution; no autograd substitution is allowed.
    grad_q=qkv[0].detach().clone().requires_grad_()
    try:candidate.staged_decode_forward(grad_q,qkv[1],qkv[2])
    except ValueError:r['gradient_required_rejected']=True
    else:raise AssertionError('Unimplemented backward silently accepted')
    r['all_output_cpu_bit_exact']=all(c['output_vs_original_cpu']['bit_equal'] for c in r['cases'])
    r['sequence_did_not_change_kernel_identity']=all(c['same_kernel_identities_as_first_sequence'] for c in r['cases'])
    r['peak_allocated_bytes']=int(torch.npu.max_memory_allocated(1))
    r['diagnostic_completed']=True
    r['acceptance']='Not production acceptance: audit exact predicates, FP32 accumulation differences, full process fallback log, and real runtime roles separately.'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lengths',type=int,nargs='+',default=[33,128,129,256]);args=parser.parse_args()
    if any(n<1 or n>256 for n in args.lengths):raise ValueError('Bounded probe only allows lengths1..256')
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[*ROOT.glob('*.py'),BASE/'c4v8_device_core/device_attention.py',BASE/'c4v8_device_core/device_quant.py',
             BASE/'c4v8_device_core/exact_fp32.py',BASE/'c4v8_fused_nvfp4/nvfp4_triton.py',
             BASE/'c4v8_fused_qdq/fused_mxfp8.py',BASE/'c4v8_fused_qdq/forward_api.py']
    hashes={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for p in sources:
        dest=args.output/'sources'/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    r={'diagnostic_completed':False,'device':'npu:1','cpu_reference_only':True,'source_sha256':hashes}
    start=time.perf_counter();exit_code=1
    try:main(args,r);exit_code=0
    except BaseException as e:r['error']=type(e).__name__+': '+str(e);traceback.print_exc();(args.output/'failure.txt').write_text(traceback.format_exc())
    finally:
        r['elapsed_seconds']=time.perf_counter()-start
        r['sources_unchanged']=all(hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
        if not r['sources_unchanged']:exit_code=1;r['source_change_error']='Source changed during audit'
        json_save(args.output/'report.json',r);(args.output/'exit').write_text(str(exit_code)+'\n')
        print(json.dumps({'diagnostic_completed':r['diagnostic_completed'],'elapsed_seconds':r['elapsed_seconds'],'error':r.get('error')}),flush=True)
    sys.exit(exit_code)
