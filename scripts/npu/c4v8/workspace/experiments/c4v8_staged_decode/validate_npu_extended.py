"""Strict standalone staged-decode audit; never registered into a model backend.

Only this probe changes. Shared numerical modules are snapshotted and must stay
unchanged. All batches have the same true KV length and query position SEQ-1;
this does not validate padding mixed-prefix requests into one batch.
"""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MAX_COMPILE_CORE_NUMBER','1')
import argparse,hashlib,json,math,re,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path[:0]=[str(ROOT),*(str(BASE/n) for n in ['c4v8_device_core','c4v8_fused_nvfp4','c4v8_fused_qdq','c4v8_fast_attention'])]
import torch
from device_attention import attention as cpu_attention,AttentionConfig,vector_round,mxfp8_tree_reduce
from device_quant import nvfp4_values
from staged_cpu_oracle import staged_cpu
from validate_cpu import compare


def save_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def to_cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:to_cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [to_cpu(v) for v in value]
    return value


def replay_actual_gemms(qkv,actual):
    """CPU-only full Vector replay using actual scaled QK/PV inputs, O(SEQ)."""
    if any(x.device.type!='cpu' for x in qkv):raise ValueError('CPU replay only')
    q,k,v=qkv;b,h,_,d=q.shape;seq=k.shape[2];rows=b*h;tiles=(seq+127)//128;padded=tiles*128
    vr=lambda x,masked=False:vector_round(x,'mxfp8',allow_structural_negative_infinity=masked)
    q4,k4,v4=(nvfp4_values(x.float()).values for x in qkv)
    if h!=k.shape[1]:
        k4=k4.repeat_interleave(h//k.shape[1],1);v4=v4.repeat_interleave(h//k.shape[1],1)
    raw=actual['raw_scores'].reshape(rows,tiles,128)
    score8=vr(raw,True);local_m=mxfp8_tree_reduce(score8,'max').squeeze(-1)
    maximum=torch.empty_like(local_m);alpha=torch.empty_like(local_m);previous=torch.full((rows,1),-math.inf)
    for tile in range(tiles):
        current=vr(torch.maximum(previous,local_m[:,tile:tile+1]),True)
        maximum[:,tile:tile+1]=current;alpha[:,tile:tile+1]=vr(torch.exp(vr(previous-current,True)));previous=current
    p8=vr(torch.exp(vr(score8-maximum.unsqueeze(-1),True)));p4=nvfp4_values(p8).values
    local_l=mxfp8_tree_reduce(p8,'sum').squeeze(-1)
    contribution=vr(actual['raw_contribution']);denominator=torch.zeros((rows,1));accumulator=torch.zeros((rows,d))
    for tile in range(tiles):
        a=alpha[:,tile:tile+1];denominator=vr(vr(denominator*a)+local_l[:,tile:tile+1])
        accumulator=vr(vr(accumulator*a)+contribution[:,tile])
    output_fp32=vr(accumulator/denominator)
    details={'q4':q4,'k4':k4,'v4':v4,'score8':score8.reshape(rows,padded),'local_m':local_m,'maximum':maximum,'alpha':alpha,
             'p8':p8.reshape(rows,padded),'p4':p4,'local_l':local_l,'contribution':contribution,'denominator':denominator,'output_fp32':output_fp32}
    return output_fp32.reshape(b,h,1,d).to(q.dtype),details


def strict_case_predicates(row):
    checks={'flags_zero':row['nonzero_flags']==0,'output_finite':row['output_finite'],
            'actual_FP32_inputs_CPU_replay_output_exact':row['cpu_replay_output']['bit_equal'],
            'all_quantization_vector_reduction_stages_exact':all(m['bit_equal'] for m in row['cpu_replay_stages'].values()),
            'kernel_identity_present':set(row['kernel_identities'])=={'scores_local_max','maximum_scan','probability_pass','probability_sum','final_state_scan'},
            'same_kernel_identity_across_sequences':row['kernel_identity_equals_first'],
            'head_partition_exact':row['head_partition']['bit_equal'],
            'repeated_same_input_exact':row['warm_repeated_output']['bit_equal']}
    if row['batch']>1:checks['batch_partition_exact']=row['batch_partition']['bit_equal']
    if 'same_npu_generic_decode' in row:checks['same_npu_generic_decode_exact']=row['same_npu_generic_decode']['bit_equal']
    return checks


def main(args,r):
    initial=subprocess.check_output(['npu-smi','info'],text=True);(args.output/'npu-before.txt').write_text(initial)
    if sorted(set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)',initial))))!=list(range(8)):
        raise RuntimeError('Exclusive all-eight-idle gate failed')
    import torch_npu
    torch.npu.set_device(1);torch.npu.set_per_process_memory_fraction(.05,1)
    torch.npu.set_compile_mode(jit_compile=False);torch.npu.matmul.allow_hf32=False;torch.set_num_threads(1)
    os.environ['HBY_C4V8_ALLOW_UNVALIDATED_FUSION']='1';os.environ['HBY_C4V8_AUDIT_KERNEL_CACHE']='1'
    import staged_decode as candidate
    from fast_attention import attention as generic_npu
    cfg=AttentionConfig(query_block=64,key_block=128);rng=torch.Generator().manual_seed(208815)
    specs=[(torch.bfloat16,1,8,1,n) for n in (33,128,129,256,128)]
    specs += [(torch.bfloat16,2,4,2,n) for n in (128,129,256,128)]
    specs += [(torch.float32,2,4,2,33)]
    specs += [(torch.bfloat16,1,8,1,n) for n in args.long_lengths]
    r.update({'cases':[],'metadata':candidate.metadata(),'strict_validation_completed':False,'hf32':False,'jit_compile':False,
              'batch_contract':'Each batch member has identical true SEQ; query is SEQ-1; mixed-prefix padding is not supported by this test.',
              'long_reference_complexity':'Decode query length1, so CPU reference and replay use O(SEQ), no NxN attention allocation.'})
    baseline_ids=None
    with torch.inference_mode():
        for index,(dtype,batch,heads,kvheads,seq) in enumerate(specs):
            folder=args.output/f'case-{index:02d}-n{seq}-b{batch}-h{heads}-hk{kvheads}';folder.mkdir()
            print(f'PHASE cpu_reference_only index={index} seq={seq} batch={batch}',flush=True)
            q=torch.randn(batch,heads,1,128,generator=rng).to(dtype)
            k=torch.randn(batch,kvheads,seq,128,generator=rng).to(dtype);v=torch.randn(batch,kvheads,seq,128,generator=rng).to(dtype)
            qkv=[q,k,v];expected=cpu_attention(q,k,v,cfg,query_offset=seq-1)
            cpu_batched,cpu_stages=staged_cpu(q,k,v,batched_gemm=True,return_intermediates=True)
            torch.save({'qkv':qkv,'original_output':expected,'batched_output':cpu_batched,'batched_stages':cpu_stages},folder/'cpu-reference.pt')
            print(f'PHASE staged_npu index={index} seq={seq}',flush=True)
            nqkv=[x.to('npu:1') for x in qkv];torch.npu.synchronize(1);start=time.perf_counter()
            output,flags,stages=candidate.staged_decode_forward(*nqkv,check=False,return_intermediates=True)
            torch.npu.synchronize(1);elapsed=time.perf_counter()-start
            actual=to_cpu(stages);actual['flags']=flags.cpu();actual['output']=output.cpu();torch.save(actual,folder/'actual-npu-stages.pt')
            identity=dict(candidate.LAST_KERNEL_IDENTITIES)
            if baseline_ids is None:baseline_ids=identity
            row={'index':index,'seq':seq,'batch':batch,'heads':heads,'kvheads':kvheads,'dtype':str(dtype),
                 'first_observed_seconds_including_any_compile':elapsed,'nonzero_flags':int((actual['flags']!=0).sum()),
                 'flags_unique':actual['flags'].unique().tolist(),'output_finite':bool(torch.isfinite(actual['output']).all()),
                 'vs_original_CPU':compare(actual['output'],expected),'vs_batched_CPU':compare(actual['output'],cpu_batched),
                 'kernel_identities':identity,'kernel_identity_equals_first':identity==baseline_ids}
            r['cases'].append(row);save_json(args.output/'report.partial.json',r)
            if row['nonzero_flags'] or not row['output_finite']:raise FloatingPointError('Flags/nonfinite; actual stage tensors saved')
            print(f'PHASE cpu_actual_gemm_replay index={index}',flush=True)
            replay,replay_stages=replay_actual_gemms(qkv,actual);torch.save({'output':replay,'stages':replay_stages},folder/'cpu-actual-gemm-replay.pt')
            row['cpu_replay_output']=compare(replay,actual['output'])
            row['cpu_replay_stages']={name:compare(value,actual[name]) for name,value in replay_stages.items()}
            row['cpu_original_stages']={name:compare(actual[name],value) for name,value in cpu_stages.items()}
            # Head partitions preserve the mapping of query groups to KV heads.
            if kvheads==1:
                parts=[candidate.staged_decode_forward(nqkv[0][:,s:s+heads//2].contiguous(),nqkv[1],nqkv[2])[0] for s in (0,heads//2)]
            else:
                group=heads//kvheads
                parts=[candidate.staged_decode_forward(nqkv[0][:,j*group:(j+1)*group].contiguous(),nqkv[1][:,j:j+1].contiguous(),nqkv[2][:,j:j+1].contiguous())[0] for j in range(kvheads)]
            row['head_partition']=compare(torch.cat(parts,1).cpu(),actual['output'])
            if batch>1:
                parts=[candidate.staged_decode_forward(*(x[j:j+1].contiguous() for x in nqkv))[0] for j in range(batch)]
                row['batch_partition']=compare(torch.cat(parts,0).cpu(),actual['output'])
            if seq<=256:
                baseline=generic_npu(*nqkv,cfg,query_offset=seq-1).cpu();torch.save(baseline,folder/'generic-same-npu-output.pt')
                row['same_npu_generic_decode']=compare(baseline,actual['output'])
            torch.npu.synchronize(1);start=time.perf_counter()
            for _ in range(args.warm_repeats):warm=candidate.staged_decode_forward(*nqkv)[0]
            torch.npu.synchronize(1);row['warm_mean_seconds_including_flag_checks']=(time.perf_counter()-start)/args.warm_repeats
            row['warm_repeated_output']=compare(warm.cpu(),actual['output'])
            row['strict_predicates']=strict_case_predicates(row)
            save_json(folder/'report.json',row);save_json(args.output/'report.partial.json',r)
            failed=[name for name,passed in row['strict_predicates'].items() if not passed]
            if failed:raise AssertionError('Strict staged gate failed: '+','.join(failed))
            print(f'PHASE case_passed index={index} seq={seq} warm_seconds='+str(row['warm_mean_seconds_including_flag_checks']),flush=True)
        invalid=nqkv[0].clone();invalid[0,0,0,0]=float('nan')
        try:candidate.staged_decode_forward(invalid,nqkv[1],nqkv[2])
        except FloatingPointError:r['nonfinite_input_rejected']=True
        else:raise AssertionError('Nonfinite input accepted')
    requiring_grad=nqkv[0].detach().clone().requires_grad_()
    try:candidate.staged_decode_forward(requiring_grad,nqkv[1],nqkv[2])
    except ValueError:r['unimplemented_backward_rejected']=True
    else:raise AssertionError('Forward-only implementation silently accepted required gradients')
    r['peak_allocated_bytes']=int(torch.npu.max_memory_allocated(1));r['strict_validation_completed']=True
    r['acceptance_scope']='Only the observed bounded decode shapes, fixed128 key policy, forward quantization and CPU replay exactness. CPU original output bit equality is reported separately because FP32 GEMM/scaling can differ. No production registration, no low-precision backward, no mixed-length batching.'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--long-lengths',type=int,nargs='*',default=[1024,4096,8192]);p.add_argument('--warm-repeats',type=int,default=3);args=p.parse_args()
    if any(n<257 or n>8192 for n in args.long_lengths) or args.warm_repeats<1 or args.warm_repeats>5:raise ValueError('Bounded long lengths257..8192 and repeats1..5')
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'staged_cpu_oracle.py',ROOT/'staged_decode.py',ROOT/'staged_kernels.py',ROOT/'validate_cpu.py',
             *(BASE/path for path in ['c4v8_device_core/device_attention.py','c4v8_device_core/device_quant.py','c4v8_device_core/exact_fp32.py',
                                      'c4v8_fused_nvfp4/nvfp4_triton.py','c4v8_fused_qdq/fused_mxfp8.py','c4v8_fused_qdq/forward_api.py','c4v8_fast_attention/fast_attention.py'])]
    hashes={str(path.relative_to(BASE)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for path in sources:
        dest=args.output/'sources'/path.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    r={'strict_validation_completed':False,'cpu_reference_only':True,'source_sha256':hashes,'device':'npu:1'}
    start=time.perf_counter();code=1
    try:main(args,r);code=0
    except BaseException as e:r['error']=type(e).__name__+': '+str(e);traceback.print_exc();(args.output/'failure.txt').write_text(traceback.format_exc())
    finally:
        r['elapsed_seconds']=time.perf_counter()-start;r['sources_unchanged']=all(hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
        if not r['sources_unchanged']:code=1
        save_json(args.output/'report.json',r);(args.output/'exit').write_text(str(code)+'\n')
        print(json.dumps({'strict_validation_completed':r['strict_validation_completed'],'elapsed_seconds':r['elapsed_seconds'],'error':r.get('error')}),flush=True)
    sys.exit(code)
