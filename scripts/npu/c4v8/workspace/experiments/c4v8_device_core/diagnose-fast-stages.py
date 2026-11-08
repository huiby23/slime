"""Bounded stage attribution. CPU reference is never executed on NPU.

Uses saved whole-probe inputs, preserves all tensors, and only attributes an
output difference to FP32 GEMM after actual raw QK/PV CPU replay reproduces all
non-GEMM stages and the instrumented NPU path reproduces the frozen core.
"""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MAX_COMPILE_CORE_NUMBER','1')
import argparse,hashlib,json,math,re,subprocess,sys,time,traceback
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(BASE/name) for name in ['c4v8_device_core','c4v8_fast_attention','c4v8_fused_nvfp4','c4v8_fused_qdq']]
from device_attention import AttentionConfig,online_softmax_step,vector_round
from device_quant import nvfp4_values,mxfp8_values


def metric(a,b):
    a=a.detach().cpu();b=b.detach().cpu()
    if a.shape!=b.shape:return {'shape_mismatch':[list(a.shape),list(b.shape)],'bit_exact':False}
    af=a.float();bf=b.float();valid=torch.isfinite(af)&torch.isfinite(bf)
    delta=torch.where(valid,af-bf,0)
    exact=torch.equal(af.contiguous().view(torch.int32),bf.contiguous().view(torch.int32))
    return {'bit_exact':exact,'shape':list(a.shape),'count':a.numel(),'different_values':int((af!=bf).sum()),
            'max_finite_abs':float(delta.abs().max()),'relative_finite_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(torch.where(valid,bf,0)).clamp_min(1e-30)),
            'nonfinite_pattern_equal':torch.equal(torch.isfinite(af),torch.isfinite(bf))}


def vr(x,masked=False):return vector_round(x,'mxfp8',allow_structural_negative_infinity=masked)


def cpu_trace(qkv,config,overrides=None):
    q,k,v=qkv;dtype=q.dtype;b,h,n,d=q.shape
    q4,k4,v4=(nvfp4_values(x.float()).values for x in qkv)
    if h!=k.shape[1]:
        repeat=h//k.shape[1];k4=k4.repeat_interleave(repeat,1);v4=v4.repeat_interleave(repeat,1)
    trace={'tiles':[],'normalizations':[]};outputs=[];index=0
    for qs in range(0,n,config.query_block):
        qe=min(n,qs+config.query_block);state=(b,h,qe-qs,1)
        m=torch.full(state,-math.inf);l=torch.zeros(state);o=torch.zeros((b,h,qe-qs,d))
        for ks in range(0,qe,config.key_block):
            ke=min(qe,ks+config.key_block)
            raw_qk_matmul=q4[:,:,qs:qe]@k4[:,:,ks:ke].transpose(-1,-2)
            raw_qk=raw_qk_matmul/math.sqrt(d)
            if overrides is not None:raw_qk=overrides['tiles'][index]['raw_qk']
            mask=torch.arange(ks,ke)[None,:]>torch.arange(qs,qe)[:,None]
            masked=raw_qk.masked_fill(mask,-math.inf);score8=vr(masked,True)
            old_m,old_l,old_o=m,l,o
            m,l,alpha,p8=online_softmax_step(score8,m,l,config)
            p4=nvfp4_values(p8).values
            raw_pv=p4@v4[:,:,ks:ke]
            if overrides is not None:raw_pv=overrides['tiles'][index]['raw_pv']
            contribution8=vr(raw_pv);scaled_old8=vr(o*alpha);o=vr(scaled_old8+contribution8)
            trace['tiles'].append({'qs':qs,'qe':qe,'ks':ks,'ke':ke,'raw_qk_matmul':raw_qk_matmul,'raw_qk':raw_qk,'masked_scores':masked,'score8':score8,
                'old_m':old_m,'old_l':old_l,'old_o':old_o,'m':m,'l':l,'alpha':alpha,'p8':p8,'p4':p4,'raw_pv':raw_pv,
                'contribution8':contribution8,'scaled_old8':scaled_old8,'post':o})
            index+=1
        raw_norm=o/l;normalized=vr(raw_norm)
        trace['normalizations'].append({'qs':qs,'qe':qe,'numerator':o,'denominator':l,'raw_normalized':raw_norm,'normalized':normalized})
        outputs.append(normalized.to(dtype))
    trace['output']=torch.cat(outputs,2)
    return trace


def format_expected(x,kind,masked=False):
    if kind=='c4':
        r=nvfp4_values(x.float(),logical_bytes=True)
        return r.values,r.payload_bytes,r.scale_bytes,{'upper_encode':r.upper_encode,'upper_decode':r.upper_decode}
    structural=torch.isneginf(x) if masked else torch.zeros_like(x,dtype=torch.bool)
    r=mxfp8_values(x.float().masked_fill(structural,0),logical_bytes=True)
    return r.values.masked_fill(structural,-math.inf),r.payload_bytes,r.scale_bytes,{}


def npu_trace(qkv,config,out,report):
    from nvfp4_triton import quantized_values
    from forward_api import qdq_forward,pre_p_forward,post_pv_forward,normalize_forward
    from fast_attention import attention as frozen_attention
    trace={'tiles':[],'normalizations':[],'formats':[]};checks=[]
    def quant(x,kind,name,masked=False):
        if kind=='c4':values,codes,scales=quantized_values(x.float().contiguous(),logical_bytes=True)
        else:
            values,codes,scales,flags=qdq_forward(x.float().contiguous(),allow_mask=masked,check=False)
            if bool((flags!=0).any()):raise FloatingPointError('Format flags at '+name)
        xc=x.detach().cpu();vc=values.cpu();cc=codes.cpu();sc=scales.cpu()
        ev,ec,es,extra=format_expected(xc,kind,masked)
        comparison={'name':name,'kind':kind,'values':metric(vc,ev),'codes':metric(cc,ec),'scales':metric(sc,es)}
        trace['formats'].append({'name':name,'kind':kind,'input':xc,'actual_values':vc,'actual_codes':cc,'actual_scales':sc,
                                 'expected_values':ev,'expected_codes':ec,'expected_scales':es,**extra})
        checks.append(comparison);report['same_input_format_checks']=checks
        if not all(comparison[k]['bit_exact'] for k in ('values','codes','scales')):
            torch.save(trace,out/'partial-trace.pt');raise AssertionError('Same-input quantization mismatch at '+name)
        return values
    nqkv=[x.to('npu:1') for x in qkv];q,k,v=nqkv;b,h,n,d=q.shape
    qq,kk,vv=[quant(x,'c4',name) for x,name in zip(nqkv,['Q_C4','K_C4','V_C4'])]
    if h!=k.shape[1]:
        repeat=h//k.shape[1];kk=kk.repeat_interleave(repeat,1);vv=vv.repeat_interleave(repeat,1)
    outputs=[]
    for qs in range(0,n,config.query_block):
        qe=min(n,qs+config.query_block);state=(b,h,qe-qs,1)
        m=torch.full(state,-math.inf,device=q.device);l=torch.zeros(state,device=q.device);o=torch.zeros((b,h,qe-qs,d),device=q.device)
        for ks in range(0,qe,config.key_block):
            ke=min(qe,ks+config.key_block);label='q'+str(qs)+'-k'+str(ks)
            print('PHASE npu_stage '+label,flush=True)
            raw_qk_matmul=qq[:,:,qs:qe]@kk[:,:,ks:ke].transpose(-1,-2)
            raw_qk=raw_qk_matmul/math.sqrt(d)
            scaling_expected=raw_qk_matmul.cpu()/math.sqrt(d)
            report.setdefault('same_input_qk_scaling_checks',[]).append(metric(raw_qk.cpu(),scaling_expected))
            mask=torch.arange(ks,ke,device=q.device)[None,:]>torch.arange(qs,qe,device=q.device)[:,None]
            masked=raw_qk.masked_fill(mask,-math.inf)
            score8=quant(masked,'v8',label+'-score8',True)
            old_m,old_l,old_o=m,l,o
            m,l,alpha,p8,pre_flags=pre_p_forward(masked.contiguous(),m.contiguous(),l.contiguous(),check=False)
            p4=quant(p8,'c4',label+'-P4')
            raw_pv=p4@vv[:,:,ks:ke]
            contribution8=quant(raw_pv,'v8',label+'-contribution8')
            scaled_old8=quant(o*alpha,'v8',label+'-scaled_old8')
            o,post_flags=post_pv_forward(o.contiguous(),raw_pv.contiguous(),alpha.contiguous(),check=False)
            event={'qs':qs,'qe':qe,'ks':ks,'ke':ke,'raw_qk_matmul':raw_qk_matmul,'raw_qk':raw_qk,'masked_scores':masked,'score8':score8,'old_m':old_m,'old_l':old_l,
                   'old_o':old_o,'m':m,'l':l,'alpha':alpha,'p8':p8,'p4':p4,'raw_pv':raw_pv,'contribution8':contribution8,
                   'scaled_old8':scaled_old8,'post':o,'pre_flags':pre_flags,'post_flags':post_flags}
            trace['tiles'].append({key:(value.cpu() if isinstance(value,torch.Tensor) else value) for key,value in event.items()})
            if bool((pre_flags!=0).any()|(post_flags!=0).any()):
                torch.save(trace,out/'partial-trace.pt');raise FloatingPointError('Fused stage flags at '+label)
        normalized,flags=normalize_forward(o.contiguous(),l.contiguous(),check=False)
        oc,lc=o.cpu(),l.cpu();raw_expected=oc/lc;expected=vr(raw_expected)
        trace['normalizations'].append({'qs':qs,'qe':qe,'numerator':oc,'denominator':lc,'raw_normalized_cpu_same_inputs':raw_expected,
                                      'normalized':normalized.cpu(),'flags':flags.cpu(),
                                      'same_input_normalize_expected':expected})
        report.setdefault('same_input_normalize_checks',[]).append(metric(normalized.cpu(),expected))
        if bool((flags!=0).any()):raise FloatingPointError('Normalize flags')
        outputs.append(normalized.to(q.dtype))
    trace['output']=torch.cat(outputs,2).cpu()
    # Guard against extra diagnostic launches or logical-byte stores changing
    # execution: the traced path must reproduce the frozen actual callable.
    actual=frozen_attention(*nqkv,config).cpu()
    trace['frozen_core_output']=actual
    report['instrumented_vs_frozen_core']=metric(trace['output'],actual)
    return trace


def compare_traces(left,right):
    keys=['raw_qk_matmul','raw_qk','score8','m','alpha','p8','p4','l','raw_pv','contribution8','scaled_old8','post']
    rows=[]
    for a,b in zip(left['tiles'],right['tiles']):
        rows.append({'qs':a['qs'],'ks':a['ks'],'stages':{name:metric(a[name],b[name]) for name in keys}})
    return {'tiles':rows,'normalizations':[metric(a['normalized'],b['normalized']) for a,b in zip(left['normalizations'],right['normalizations'])],
            'output':metric(left['output'],right['output'])}


def main(args,report):
    before=subprocess.check_output(['npu-smi','info'],text=True);(args.output/'npu-before.txt').write_text(before)
    if sorted(set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)',before))))!=list(range(8)):
        raise RuntimeError('Explicit exclusive all-eight-idle gate failed')
    import torch_npu
    torch.npu.set_device(1);torch.npu.set_per_process_memory_fraction(.05,1)
    torch.npu.set_compile_mode(jit_compile=False);torch.npu.matmul.allow_hf32=False;torch.set_num_threads(1)
    os.environ['HBY_C4V8_ALLOW_UNVALIDATED_FUSION']='1'
    payload=torch.load(args.input_run/'cpu-reference.pt',map_location='cpu',weights_only=False)
    config=AttentionConfig(query_block=64,key_block=128);report['cases']=[]
    with torch.inference_mode():
        for length in args.lengths:
            case=next(c for c in payload if c['n']==length)
            out=args.output/('n'+str(length));out.mkdir()
            row={'n':length};report['cases'].append(row)
            print('PHASE cpu_original_trace_begin n='+str(length),flush=True)
            original=cpu_trace(case['qkv'],config)
            row['cpu_trace_vs_saved_reference']=metric(original['output'],case['output'])
            if not row['cpu_trace_vs_saved_reference']['bit_exact']:raise AssertionError('CPU trace does not reproduce saved reference')
            torch.save(original,out/'cpu-original.pt')
            print('PHASE npu_fused_trace_begin n='+str(length),flush=True)
            npu=npu_trace(case['qkv'],config,out,row);torch.save(npu,out/'npu-fused.pt')
            saved_path=args.input_run/('n'+str(length)+'-forward.pt')
            if saved_path.exists():row['trace_vs_saved_npu_forward']=metric(npu['output'],torch.load(saved_path,map_location='cpu',weights_only=False))
            print('PHASE cpu_actual_raw_gemm_replay_begin n='+str(length),flush=True)
            replay=cpu_trace(case['qkv'],config,overrides=npu);torch.save(replay,out/'cpu-replay-actual-npu-qk-pv.pt')
            row['cpu_original_vs_npu']=compare_traces(original,npu)
            row['cpu_actual_gemm_replay_vs_npu']=compare_traces(replay,npu)
            nongemm=['score8','m','alpha','p8','p4','l','contribution8','scaled_old8','post']
            replay_stages=row['cpu_actual_gemm_replay_vs_npu']
            row['all_replayed_nongemm_stages_bit_exact']=all(tile['stages'][name]['bit_exact'] for tile in replay_stages['tiles'] for name in nongemm)
            row['gemm_only_attribution_supported']=(row['all_replayed_nongemm_stages_bit_exact'] and replay_stages['output']['bit_exact']
                and all(x['bit_exact'] for x in replay_stages['normalizations']) and row['instrumented_vs_frozen_core']['bit_exact']
                and all(x['bit_exact'] for x in row['same_input_normalize_checks'])
                and all(x['bit_exact'] for x in row['same_input_qk_scaling_checks'])
                and row.get('trace_vs_saved_npu_forward',{'bit_exact':True})['bit_exact'])
            (out/'report.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
            (args.output/'report.partial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print('PHASE attribution_complete n='+str(length)+' supported='+str(row['gemm_only_attribution_supported']),flush=True)
    report['peak_allocated_bytes']=int(torch.npu.max_memory_allocated(1));report['diagnostic_completed']=True
    report['all_lengths_gemm_only_attribution_supported']=all(c['gemm_only_attribution_supported'] for c in report['cases'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lengths',type=int,nargs='+',default=[33,128,256]);args=parser.parse_args()
    if any(n<1 or n>256 for n in args.lengths):raise ValueError('Bounded stage trace allows at most256 tokens')
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),BASE/'c4v8_device_core/device_attention.py',BASE/'c4v8_device_core/device_quant.py',BASE/'c4v8_device_core/exact_fp32.py',
             BASE/'c4v8_fast_attention/fast_attention.py',BASE/'c4v8_fused_nvfp4/nvfp4_triton.py',BASE/'c4v8_fused_qdq/fused_mxfp8.py',BASE/'c4v8_fused_qdq/forward_api.py']
    hashes={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for p in sources:
        dest=args.output/'sources'/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    report={'diagnostic_completed':False,'cpu_reference_only':True,'source_sha256':hashes,'input_run':str(args.input_run)}
    started=time.perf_counter();exit_code=1
    try:main(args,report);exit_code=0
    except BaseException as e:report['error']=type(e).__name__+': '+str(e);traceback.print_exc();(args.output/'failure.txt').write_text(traceback.format_exc())
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        report['sources_unchanged']=all(hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
        if not report['sources_unchanged']:exit_code=1
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(args.output/'exit').write_text(str(exit_code)+'\n')
        print(json.dumps({'diagnostic_completed':report['diagnostic_completed'],'elapsed_seconds':report['elapsed_seconds'],'error':report.get('error')}),flush=True)
    sys.exit(exit_code)
