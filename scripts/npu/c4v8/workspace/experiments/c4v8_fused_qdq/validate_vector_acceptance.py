"""Acceptance of the actual default forward API and frozen source, no monkeypatch."""
import os,sys,json,hashlib
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root),str(root.parent/'c4v8_fused_nvfp4'),str(root.parent/'c4v8_device_core')]
import forward_api

from validate_npu import main
out=Path(sys.argv[sys.argv.index('--output-dir')+1])
try:
    main()
    import torch
    from cpu_oracle import pre_p
    r=json.loads((out/'report.json').read_text())
    for seed,scale in enumerate((.001,1.,32.,100.)):
        rows=(1,4,13,16)[seed]
        gen=torch.Generator().manual_seed(8180+seed)
        score=torch.randn((rows,128),generator=gen)*scale
        mask=torch.rand((rows,128),generator=gen)<.35;mask[:,0]=False
        score[mask]=-float('inf')
        maximum=torch.rand((rows,1),generator=gen)*scale
        den=torch.rand((rows,1),generator=gen)*30.+.25
        expected=pre_p(score,maximum,den)
        actual=forward_api.pre_p_forward(*(x.to('npu:1') for x in (score,maximum,den)))
        torch.save({'inputs':(score,maximum,den),'actual':tuple(x.cpu() for x in actual),'expected':expected},out/f'random-mask-{seed}.pt')
        for label,a,b in zip(('maximum','denominator','alpha','probabilities'),actual,expected):
            a=a.cpu();eq=torch.equal(a.view(torch.int32),b.view(torch.int32))
            r['tests'].append({'name':f'random-mask-{seed}-{label}','bit_equal':eq,'different':int((a.view(torch.int32)!=b.view(torch.int32)).sum()),'max_abs':float((a-b).abs().max())})
    for kind in ('nan','positive_inf','all_masked'):
        x=torch.zeros((8,128));m=torch.full((8,1),-float('inf'));l=torch.zeros((8,1))
        if kind=='all_masked':x[3,:]=-float('inf')
        else:x[3,0]=float('nan') if kind=='nan' else float('inf')
        args=[t.to('npu:1') for t in (x,m,l)]
        actual=forward_api.pre_p_forward(*args,check=False);flags=actual[-1].cpu()
        selected=(flags.reshape(8,-1)!=0).any(-1).tolist()
        if selected!=[False,False,False,True,False,False,False,False]:raise AssertionError('Invalid row flags lost or cross-row pollution: '+kind)
        try:forward_api.pre_p_forward(*args)
        except FloatingPointError:pass
        else:raise AssertionError('Invalid state accepted: '+kind)
        r['tests'].append({'name':'negative-'+kind,'passed':True,'flags':flags.tolist()})
    r['peak_npu_allocated_bytes']=torch.npu.max_memory_allocated()
    r['passed']=all(t.get('bit_equal',t.get('passed',True)) for t in r['tests'])
    r['status']='passed' if r['passed'] else 'needs_numerical_review'
    (out/'report.json').write_text(json.dumps(r,indent=2))
    print(json.dumps({'acceptance_status':r['status'],'default_api_used':True,'total_tests':len(r['tests'])}),flush=True)
    if not r['passed']:raise SystemExit(2)
except BaseException as exc:
    if not isinstance(exc,SystemExit) or exc.code not in (0,2):
        import traceback
        (out/'acceptance-failure-traceback.txt').write_text(traceback.format_exc())
        p=out/'report.json'
        d=json.loads(p.read_text()) if p.exists() else {}
        d.update(passed=False,status='failed',acceptance_error=str(exc),acceptance_error_type=type(exc).__name__)
        p.write_text(json.dumps(d,indent=2))
    raise
finally:
    for name in ('report.json','report.partial.json'):
        p=out/name
        if p.exists():
            d=json.loads(p.read_text());d['pre_p_stages']=2;d['default_api_used']=True;d['fused_source_sha256']=hashlib.sha256((root/'fused_mxfp8.py').read_bytes()).hexdigest();p.write_text(json.dumps(d,indent=2))
