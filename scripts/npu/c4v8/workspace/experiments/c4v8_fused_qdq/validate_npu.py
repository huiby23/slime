"""Bounded, explicitly authorized single-NPU experiment; never starts a model.

Run ONLY after the coordinating agent releases NPU1. Code/profile manifests and
failure traces are preserved. A semantic mismatch is recorded as needs_review,
never silently promoted to an accepted precision policy.
"""
import argparse,hashlib,json,os,re,shutil,subprocess,sys,time,traceback
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--phase',choices=('qdq','vector','qdq-vector','decode','all'),default='qdq')
    parser.add_argument('--device',type=int,choices=(1,),default=1)
    parser.add_argument('--iterations',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.iterations<=10:parser.error('iterations must be1..10')
    out=args.output_dir.resolve()
    if not str(out).startswith('/root/hby/'):parser.error('All remote artifacts must be under/root/hby')
    if out.exists() and any(out.iterdir()):parser.error('Choose a fresh evidence directory')
    out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parent
    os.environ['TRITON_CACHE_DIR']='/root/hby/cache/triton-c4v8'
    os.environ['TMPDIR']='/root/hby/tmp'
    Path(os.environ['TRITON_CACHE_DIR']).mkdir(parents=True,exist_ok=True)
    Path(os.environ['TMPDIR']).mkdir(parents=True,exist_ok=True)
    os.environ['HBY_C4V8_ALLOW_UNVALIDATED_FUSION']='1'
    os.environ['HBY_C4V8_AUDIT_KERNEL_CACHE']='1'
    os.environ['MAX_COMPILE_CORE_NUMBER']='1'
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
    snapshot=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
    (out/'npu-before.txt').write_text(snapshot.stdout+snapshot.stderr)
    idle=sorted(set(int(i) for i in re.findall(r'No running processes found in NPU\s+(\d+)',snapshot.stdout)))
    if snapshot.returncode or idle!=list(range(8)):
        raise RuntimeError('Coordinated all-eight idle gate failed; no NPU allocation started')
    sys.path[:0]=[str(root),str(root.parent/'c4v8_device_core'),str(root.parent/'c4v8_fused_nvfp4')]
    import torch
    import torch_npu
    from device_quant import mxfp8_values,nvfp4_values
    from device_attention import AttentionConfig,attention,online_softmax_step,vector_round
    from cpu_oracle import pre_p,post_pv
    from decode_cpu_oracle import decode
    from forward_api import qdq_forward,pre_p_forward,post_pv_forward,normalize_forward
    from whole_decode import whole_decode_forward
    import whole_decode
    torch.set_num_threads(1)
    report={'phase':args.phase,'device':args.device,'passed':False,'status':'running',
            'production_activated':False,'tests':[],'timings':[],
            'uses_portable_npu_reference':False,
            'cache_dir':os.environ['TRITON_CACHE_DIR'],'tmpdir':os.environ['TMPDIR']}
    sources=list(root.glob('*.py'))+list((root.parent/'c4v8_device_core').glob('*.py'))
    sources+=[root.parent/'c4v8_fused_nvfp4'/'nvfp4_triton.py']
    report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for p in root.glob('*.py'):shutil.copyfile(p,out/('source-'+p.name))
    shutil.copyfile(root.parent/'c4v8_fused_nvfp4'/'nvfp4_triton.py',out/'source-nvfp4_triton.py')
    def save():
        (out/'report.partial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    def compare(name,actual,expected,strict=False):
        a=actual.detach().float().cpu();b=expected.detach().float().cpu()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise FloatingPointError(name+': unexpected nonfinite output')
        diff=a-b
        row={'name':name,'bit_equal':torch.equal(a.view(torch.int32),b.view(torch.int32)),
             'max_abs':float(diff.abs().max()),'relative_l2':float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(b).clamp_min(1e-30)),
             'different_values':int((a!=b).sum()),'elements':a.numel(),'strict':strict}
        report['tests'].append(row);save()
        if strict and not row['bit_equal']:raise AssertionError(name+': strict encoding/value mismatch')
    def time_call(name,fn):
        # Existing warm result must have been validated by the caller first.
        torch.npu.synchronize();start=time.monotonic()
        for _ in range(args.iterations):last=fn()
        torch.npu.synchronize()
        report['timings'].append({'name':name,'iterations':args.iterations,
                                  'wall_ms_per_call':1000*(time.monotonic()-start)/args.iterations,
                                  'includes_host_flag_validation':True})
        save();return last
    try:
        torch.npu.set_device(args.device);device=f'npu:{args.device}'
        torch.npu.set_per_process_memory_fraction(.05,args.device)
        torch.npu.matmul.allow_hf32=False
        torch.npu.set_compile_mode(jit_compile=False)
        report['actual_allow_hf32']=bool(torch.npu.matmul.allow_hf32)
        report['torch_npu_jit_compile']=False
        if report['actual_allow_hf32']:raise RuntimeError('HF32 simulator mode was not disabled')
        if args.phase in ('qdq','qdq-vector','all'):
            print('PHASE fused_MXFP8_only CPU_reference',flush=True)
            gen=torch.Generator().manual_seed(1108)
            for width in (32,128,17):
                x=torch.randn((8,width),generator=gen);x[0,0]=0.;x[0,1]=-0.
                expected=mxfp8_values(x,logical_bytes=True)
                nx=x.to(device)
                actual,codes,scales,flags=qdq_forward(nx)
                compare(f'mxfp8-width{width}',actual,expected.values,strict=True)
                if not torch.equal(codes.cpu(),expected.payload_bytes) or not torch.equal(scales.cpu(),expected.scale_bytes):
                    raise AssertionError('MXFP8 payload/scale code mismatch')
                time_call(f'mxfp8-width{width}',lambda:qdq_forward(nx))
            # Cover scale boundaries and all finite FP32 exponent ranges.
            vals=[]
            for exponent in range(-127,120):
                b=torch.tensor(448.0*2.0**exponent)
                vals.extend([b,torch.nextafter(b,torch.tensor(0.)),torch.nextafter(b,torch.tensor(float('inf')))])
            x=torch.stack(vals).reshape(-1,1).repeat(1,32)
            expected=mxfp8_values(x,check=False,logical_bytes=True)
            a,c,s,f=qdq_forward(x.to(device),check=False)
            # These selected boundaries do not reconstruct overflow.
            compare('mxfp8-scale-boundaries',a,expected.values,strict=True)
            if not torch.equal(c.cpu(),expected.payload_bytes) or not torch.equal(s.cpu(),expected.scale_bytes):
                raise AssertionError('Scale boundary logical code mismatch')
            invalid=torch.tensor([[float('nan'),float('inf'),0.,-0.]*8],device=device)
            *_,flags=qdq_forward(invalid,check=False)
            if not bool((flags!=0).all()):raise AssertionError('Invalid inputs were not flagged')
            masked=torch.tensor([[-float('inf'),-0.,1.,-1.]*8],device=device)
            a,_,_,flags=qdq_forward(masked,allow_mask=True)
            if not bool(torch.isneginf(a[0,::4]).all()):raise AssertionError('Structural masks lost')
            report['tests'].append({'name':'invalid-and-mask-flags','passed':True});save()
        if args.phase in ('vector','qdq-vector','all'):
            print('PHASE fused_Vector_only CPU_reference',flush=True)
            gen=torch.Generator().manual_seed(1111)
            for width in (128,17,32,64):
                scores=torch.randn((8,width),generator=gen);scores[::2,width//2+1:]=-float('inf')
                maximum=torch.full((8,1),-float('inf'));denominator=torch.zeros((8,1))
                expected=pre_p(scores,maximum,denominator)
                inputs=[x.to(device) for x in (scores,maximum,denominator)]
                actual=pre_p_forward(*inputs,check=False)
                if bool((actual[-1]!=0).any()):
                    bundle={'inputs_cpu':(scores,maximum,denominator),
                            'actual_cpu':tuple(a.detach().cpu() for a in actual),'expected_cpu':expected}
                    torch.save(bundle,out/f'preP-width{width}-invalid.pt')
                    report['invalid_preP']={'width':width,'flags':actual[-1].cpu().tolist(),
                        'maximum':actual[0].cpu().flatten().tolist(),
                        'alpha':actual[2].cpu().flatten().tolist()}
                    # Nonfinite diagnostics live in torch artifact; JSON stays strict.
                    report['invalid_preP'].pop('maximum');report['invalid_preP'].pop('alpha')
                    save()
                    raise FloatingPointError('Fused pre-P emitted invalid flags; full states saved')
                for label,a,b in zip(('maximum','denominator','alpha','probabilities'),actual[:4],expected):
                    compare(f'preP-width{width}-{label}',a,b)
                # Retain finite mismatches as well as flagged failures. A cleared
                # flag never constitutes numerical acceptance on its own.
                torch.save({'inputs_cpu':(scores,maximum,denominator),
                            'actual_cpu':tuple(a.detach().cpu() for a in actual),
                            'expected_cpu':expected},out/f'preP-width{width}-states.pt')
                time_call(f'preP-width{width}',lambda:pre_p_forward(*inputs))
                carried_maximum=torch.rand((8,1),generator=gen)*3.0
                carried_denominator=torch.rand((8,1),generator=gen)*40.0+.25
                carry_expected=pre_p(scores,carried_maximum,carried_denominator)
                carry_inputs=[x.to(device) for x in (scores,carried_maximum,carried_denominator)]
                carry_actual=pre_p_forward(*carry_inputs)
                for label,a,b in zip(('maximum','denominator','alpha','probabilities'),carry_actual[:4],carry_expected):
                    compare(f'preP-carry-width{width}-{label}',a,b)
                torch.save({'inputs_cpu':(scores,carried_maximum,carried_denominator),
                            'actual_cpu':tuple(a.detach().cpu() for a in carry_actual),
                            'expected_cpu':carry_expected},out/f'preP-carry-width{width}-states.pt')
            old=torch.randn((8,128),generator=gen);contribution=torch.randn((8,128),generator=gen)
            alpha=torch.rand((8,1),generator=gen);den=torch.rand((8,1),generator=gen)+.2
            inputs=[x.to(device) for x in (old,contribution,alpha,den)]
            actual,_=post_pv_forward(*inputs)
            compare('postPV-normalized',actual,post_pv(old,contribution,alpha,den),strict=True)
            time_call('postPV-normalized',lambda:post_pv_forward(*inputs))
            for width in (17,128):
                numerator=torch.randn((8,width),generator=gen);numerator[0,0]=-0.
                denominator=torch.rand((8,1),generator=gen)*2.+.01
                expected=mxfp8_values(numerator/denominator).values
                normal_inputs=[x.to(device) for x in (numerator,denominator)]
                actual,_=normalize_forward(*normal_inputs)
                compare(f'normalize-width{width}',actual,expected,strict=True)
            # A finite negative score can reconstruct as -inf during score QDQ.
            # Later masked stages would treat it as structural, so its FIRST
            # stage overflow flag must survive independently to final checking.
            bad=torch.zeros((8,128));bad[3,0]=-torch.finfo(torch.float32).max
            state_m=torch.full((8,1),-float('inf'));state_l=torch.zeros((8,1))
            bad_inputs=[x.to(device) for x in (bad,state_m,state_l)]
            bad_output=pre_p_forward(*bad_inputs,check=False)
            actual_flags=bad_output[-1].cpu()
            torch.save({'inputs':(bad,state_m,state_l),'flags':actual_flags},out/'initial-score-overflow-negative.pt')
            row_flags=(actual_flags.reshape(8,-1)!=0).any(-1)
            if row_flags.tolist()!=[False,False,False,True,False,False,False,False]:
                raise AssertionError('Initial score overflow flag lost or leaked across rows')
            try:pre_p_forward(*bad_inputs)
            except FloatingPointError:pass
            else:raise AssertionError('Checked wrapper accepted initial score overflow')
            report['tests'].append({'name':'initial-score-overflow-survives-and-rejects','passed':True,
                                    'flags':actual_flags.tolist()});save()
        if args.phase in ('decode','all'):
            print('PHASE fused_whole_decode_only CPU_reference',flush=True)
            config=AttentionConfig(query_block=64,key_block=128)
            for n in (128,256):
                gen=torch.Generator().manual_seed(448+n)
                q=torch.randn((1,8,1,128),generator=gen).bfloat16()
                k=torch.randn((1,1,n,128),generator=gen).bfloat16()
                v=torch.randn((1,1,n,128),generator=gen).bfloat16()
                expected,cpu_audit=decode(q,k,v,config,return_audit=True)
                report.setdefault('cpu_accumulation_audit',{})[str(n)]=cpu_audit
                inputs=[x.to(device) for x in (q,k,v)]
                actual,_=whole_decode_forward(*inputs)
                report.setdefault('kernel_identities',[]).append(dict(whole_decode.LAST_KERNEL_IDENTITY))
                compare(f'whole-decode-raw-n{n}',actual,expected)
                time_call(f'whole-decode-raw-n{n}',lambda:whole_decode_forward(*inputs))
                prequant=[nvfp4_values(x.float()).values.to(device) for x in (q,k,v)]
                pq,_=whole_decode_forward(*prequant,prequantized=True)
                report.setdefault('kernel_identities',[]).append(dict(whole_decode.LAST_KERNEL_IDENTITY))
                compare(f'whole-decode-prequant-n{n}',pq.bfloat16(),expected)
                time_call(f'whole-decode-prequant-n{n}',lambda:whole_decode_forward(*prequant,prequantized=True))
        if args.phase in ('decode','all'):
            for variant in (False,True):
                identities=[r for r in report['kernel_identities'] if r['prequantized']==variant]
                if len({(r['triton_hash'],r['binary_sha256']) for r in identities})!=1:
                    raise AssertionError('Decode length caused unexpected binary recompilation')
            report['decode_runtime_sequence_binary_reuse_verified']=True
        mismatches=[r for r in report['tests'] if r.get('bit_equal') is False]
        report['status']='needs_numerical_review' if mismatches else 'passed'
        report['passed']=not mismatches
        report['peak_npu_allocated_bytes']=torch.npu.max_memory_allocated()
    except BaseException as error:
        report['status']='failed';report['error_type']=type(error).__name__;report['error']=str(error)
        (out/'failure-traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        save();(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':report['status'],'passed':report['passed'],'output_dir':str(out)}))
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()
