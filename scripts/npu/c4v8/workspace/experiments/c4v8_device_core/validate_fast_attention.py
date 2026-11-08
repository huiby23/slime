"""Bounded whole-attention candidate audit; CPU reference is never run on NPU.

Reports exact predicates and difference metrics without silently relaxing a
threshold. Completion of this diagnostic is distinct from numerical acceptance.
"""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MAX_COMPILE_CORE_NUMBER','1')
import argparse,hashlib,json,re,subprocess,sys,time,traceback
from pathlib import Path
import torch

BASE=Path(__file__).resolve().parent.parent
for directory in ['c4v8_device_core','c4v8_fused_nvfp4','c4v8_fused_qdq','c4v8_fast_attention','c4v8_rl_training_bridge']:
    sys.path.insert(0,str(BASE/directory))
from device_attention import AttentionConfig,attention as cpu_reference
from bridge import recomputed_attention


def metrics(actual,expected):
    a=actual.detach().cpu().float();b=expected.detach().cpu().float()
    if a.shape!=b.shape:raise AssertionError('shape mismatch')
    delta=(a-b).abs();finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result={'shape':list(a.shape),'finite':finite,'bit_exact':torch.equal(a.view(torch.int32),b.view(torch.int32)),
            'nonzero_differences':int((a!=b).sum()),'count':a.numel()}
    if finite:
        result.update(max_abs=float(delta.max()),relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30)))
        flat=delta.flatten();inds=torch.topk(flat,min(4,flat.numel())).indices
        result['largest']=[{'index':int(i),'actual':float(a.flatten()[i]),'expected':float(b.flatten()[i])} for i in inds if float(flat[i])!=0.0]
    return result


def save(out,r):
    (out/'report.partial.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')


def cpu_cases(lengths,config,out,r):
    cases=[];torch.set_num_threads(1)
    for n in lengths:
        print('PHASE cpu_reference_begin n='+str(n),flush=True)
        generator=torch.Generator().manual_seed(667700+n)
        q=torch.randn(1,8,n,128,generator=generator).bfloat16()
        k=torch.randn(1,1,n,128,generator=generator).bfloat16()
        v=torch.randn(1,1,n,128,generator=generator).bfloat16()
        dy=torch.randn(q.shape,generator=generator).bfloat16()
        leaves=[x.detach().requires_grad_() for x in (q,k,v)]
        if any(x.device.type!='cpu' for x in (*leaves,dy)):
            raise AssertionError('CPU reference must never run on NPU')
        start=time.perf_counter();expected=cpu_reference(*leaves,config)
        gradients=torch.autograd.grad(expected,leaves,dy)
        cases.append({'n':n,'qkv':[x.detach() for x in leaves],'dy':dy,'output':expected.detach(),
                      'gradients':[x.detach() for x in gradients],'cpu_seconds':time.perf_counter()-start})
        print('PHASE cpu_reference_end n='+str(n),flush=True)
    torch.save(cases,out/'cpu-reference.pt')
    r['cpu_reference_saved']=str(out/'cpu-reference.pt')
    return cases


def synchronized(fn):
    torch.npu.synchronize(1);start=time.perf_counter();result=fn();torch.npu.synchronize(1)
    return result,time.perf_counter()-start


def main(args,r):
    out=args.output
    before=subprocess.check_output(['npu-smi','info'],text=True)
    (out/'npu-before.txt').write_text(before)
    idle=sorted(set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)',before))))
    if idle!=list(range(8)):raise RuntimeError('All-eight idle gate failed')
    # Register the private-device backend BEFORE the first CPU autograd engine
    # run. With autoload disabled, late registration can leave ready queues
    # sized for CPU only. CPU reference tensors still remain entirely on CPU.
    import torch_npu
    torch.npu.set_device(1);torch.npu.set_per_process_memory_fraction(.05,1)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.matmul.allow_hf32=False
    if hasattr(torch.npu,'conv'):torch.npu.conv.allow_hf32=False
    r['backend_initialization']={'torch_npu_registered_before_cpu_backward':True,
        'reason':'Register private device before PyTorch autograd engine first initializes',
        'cpu_reference_tensors_device':'cpu','npu_context_initialized_before_cpu_reference':True}
    print('PHASE npu_backend_registered_and_context_initialized_before_cpu_backward',flush=True)
    config=AttentionConfig(query_block=64,key_block=128)
    r['config']=config.metadata()
    cases=cpu_cases(args.lengths,config,out,r)
    os.environ['HBY_C4V8_ALLOW_UNVALIDATED_FUSION']='1'
    from fast_attention import attention as candidate,metadata
    r['candidate_metadata']=metadata();r['hf32']=bool(torch.npu.matmul.allow_hf32);r['jit_compile']=False
    r['cases']=[]
    evidence={}
    for case in cases:
        n=case['n'];item={'n':n,'cpu_seconds':case['cpu_seconds']};r['cases'].append(item)
        print('PHASE npu_fast_direct_forward_begin n='+str(n),flush=True)
        leaves=[x.to('npu:1').requires_grad_() for x in case['qkv']];dy=case['dy'].to('npu:1')
        output,seconds=synchronized(lambda:candidate(*leaves,config))
        item['direct_forward_seconds']=seconds;item['direct_output_vs_cpu']=metrics(output,case['output']);save(out,r)
        torch.save(output.detach().cpu(),out/('n'+str(n)+'-forward.pt'))
        print('PHASE npu_fast_direct_backward_begin n='+str(n),flush=True)
        grads,seconds=synchronized(lambda:torch.autograd.grad(output,leaves,dy))
        item['direct_backward_seconds']=seconds
        item['direct_gradients_vs_cpu']=[metrics(a,b) for a,b in zip(grads,case['gradients'])];save(out,r)
        direct_saved=[x.detach().cpu() for x in grads];full_output=output.detach()
        print('PHASE npu_fast_recomputed_begin n='+str(n),flush=True)
        replay_leaves=[x.detach().requires_grad_() for x in leaves]
        replay,seconds=synchronized(lambda:recomputed_attention(*replay_leaves,kernel=candidate,config=config))
        item['recomputed_forward_seconds']=seconds;item['recomputed_output_vs_direct']=metrics(replay,full_output)
        replay_grads,seconds=synchronized(lambda:torch.autograd.grad(replay,replay_leaves,dy))
        item['recomputed_backward_seconds']=seconds
        item['recomputed_gradients_vs_cpu']=[metrics(a,b) for a,b in zip(replay_grads,case['gradients'])]
        item['recomputed_gradients_vs_direct']=[metrics(a,b) for a,b in zip(replay_grads,direct_saved)];save(out,r)
        print('PHASE npu_inference_policy_begin n='+str(n),flush=True)
        plain=[x.detach() for x in leaves]
        with torch.inference_mode():
            inference,seconds=synchronized(lambda:candidate(*plain,config))
            item['inference_forward_seconds']=seconds;item['inference_vs_direct']=metrics(inference,full_output)
            decode,seconds=synchronized(lambda:candidate(plain[0][:,:,-1:,:].contiguous(),plain[1],plain[2],config,query_offset=n-1))
            item['generic_decode_seconds']=seconds;item['generic_decode_vs_prefill']=metrics(decode,full_output[:,:,-1:,:])
            item['generic_decode_vs_cpu_last_row']=metrics(decode,case['output'][:,:,-1:,:])
            if args.include_whole_decode:
                from whole_decode import whole_decode_forward
                whole,seconds=synchronized(lambda:whole_decode_forward(plain[0][:,:,-1:,:].contiguous(),plain[1].contiguous(),plain[2].contiguous(),check=True)[0])
                item['whole_decode_seconds']=seconds;item['whole_decode_vs_prefill']=metrics(whole,full_output[:,:,-1:,:])
                item['whole_decode_vs_generic_decode']=metrics(whole,decode)
                item['whole_decode_vs_cpu_last_row']=metrics(whole,case['output'][:,:,-1:,:])
            extension=candidate(plain[0][:,:,-17:,:].contiguous(),plain[1],plain[2],config,query_offset=n-17)
            item['offset17_vs_prefill']=metrics(extension,full_output[:,:,-17:,:])
            pieces=[candidate(plain[0][:,s:s+4].contiguous(),plain[1],plain[2],config) for s in (0,4)]
            item['head4_split_vs_prefill']=metrics(torch.cat(pieces,1),full_output)
            prefix=17;k_changed=plain[1].clone();v_changed=plain[2].clone()
            k_changed[:,:,prefix:]*=8;v_changed[:,:,prefix:]*=-8
            prefix_original=candidate(plain[0][:,:,:prefix].contiguous(),plain[1],plain[2],config)
            prefix_changed=candidate(plain[0][:,:,:prefix].contiguous(),k_changed,v_changed,config)
            item['causal_future_invariance']=metrics(prefix_changed,prefix_original)
        # Compare the same generic one-query implementation under autograd
        # and inference roles; requires_grad is not a backend selector.
        generic_q=plain[0][:,:,-1:,:].contiguous().requires_grad_()
        generic=candidate(generic_q,plain[1],plain[2],config,query_offset=n-1)
        item['generic_one_query_vs_prefill']=metrics(generic,full_output[:,:,-1:,:])
        item['generic_no_grad_vs_grad_one_query']=metrics(decode,generic)
        if args.include_whole_decode:item['whole_decode_vs_generic_one_query']=metrics(whole,generic)
        evidence[str(n)]={'npu_output':full_output.cpu(),'direct_gradients':direct_saved,
                          'recomputed_gradients':[x.detach().cpu() for x in replay_grads],
                          'generic_decode':decode.cpu(),'generic_one_query':generic.detach().cpu()}
        if args.include_whole_decode:evidence[str(n)]['whole_decode']=whole.cpu();del whole
        save(out,r)
        del leaves,replay_leaves,replay,output,grads,replay_grads,plain,full_output,dy,generic_q,generic,inference,decode,extension,pieces,prefix_original,prefix_changed,k_changed,v_changed
        print('PHASE npu_case_end n='+str(n),flush=True)
    torch.save(evidence,out/'npu-observations.pt')
    r['peak_allocated_bytes']=int(torch.npu.max_memory_allocated(1))
    r['same_npu_shared_policy_exact']=all(c['generic_decode_vs_prefill']['bit_exact'] and c['generic_no_grad_vs_grad_one_query']['bit_exact'] for c in r['cases'])
    if args.include_whole_decode:r['whole_decode_shared_policy_exact']=all(c['whole_decode_vs_prefill']['bit_exact'] and c['whole_decode_vs_generic_one_query']['bit_exact'] for c in r['cases'])
    r['causality_exact']=all(c['causal_future_invariance']['bit_exact'] for c in r['cases'])
    r['strict_cpu_output_exact']=all(c['direct_output_vs_cpu']['bit_exact'] for c in r['cases'])
    r['diagnostic_completed']=True
    r['acceptance']='Diagnostic only: audit all exact predicates and any explained FP32 accumulation differences before production; no relaxed tolerance is applied here.'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--lengths',type=int,nargs='+',default=[33,128,256]);p.add_argument('--include-whole-decode',action='store_true');a=p.parse_args()
    if any(n<17 or n>256 for n in a.lengths):raise ValueError('Bounded probe supports lengths17..256')
    a.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),BASE/'c4v8_fast_attention/fast_attention.py',BASE/'c4v8_device_core/device_attention.py',
             BASE/'c4v8_device_core/device_quant.py',BASE/'c4v8_device_core/exact_fp32.py',BASE/'c4v8_fused_nvfp4/nvfp4_triton.py',
             BASE/'c4v8_fused_qdq/forward_api.py',BASE/'c4v8_fused_qdq/fused_mxfp8.py',BASE/'c4v8_fused_qdq/whole_decode.py',
             BASE/'c4v8_rl_training_bridge/bridge.py']
    hashes={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for path in sources:
        dest=a.output/'sources'/path.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    r={'diagnostic_completed':False,'include_whole_decode':a.include_whole_decode,'device':'npu:1','cpu_reference_only':True,'native_low_precision_hardware':False,'sources':hashes}
    started=time.perf_counter();code=1
    try:main(a,r);code=0
    except BaseException as e:r['error']=type(e).__name__+': '+str(e);traceback.print_exc();(a.output/'failure.txt').write_text(traceback.format_exc())
    finally:
        r['elapsed_seconds']=time.perf_counter()-started
        r['sources_unchanged']=all(hashlib.sha256((BASE/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
        if not r['sources_unchanged']:
            code=1;r['source_change_error']='Source changed during diagnostic; no acceptance'
        (a.output/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');(a.output/'exit').write_text(str(code)+'\n')
        print(json.dumps({'diagnostic_completed':r['diagnostic_completed'],'error':r.get('error'),'elapsed_seconds':r['elapsed_seconds']}),flush=True)
    sys.exit(code)
