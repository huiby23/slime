"""Probe-only three-stage pre-P adapter with explicit source/evidence identity."""
import os,sys,json,hashlib
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root),str(root.parent/'c4v8_fused_nvfp4'),str(root.parent/'c4v8_device_core')]
import forward_api
from candidate_forward_slots import pre_p_forward
forward_api.pre_p_forward=pre_p_forward
from validate_npu import main
out=Path(sys.argv[sys.argv.index('--output-dir')+1])
try:
    main()
    import torch
    from cpu_oracle import pre_p
    r=json.loads((out/'report.json').read_text())
    for seed,scale in enumerate((.001,1.,32.,100.)):
        gen=torch.Generator().manual_seed(8180+seed)
        score=torch.randn((8,128),generator=gen)*scale
        mask=torch.rand((8,128),generator=gen)<.35;mask[:,0]=False
        score[mask]=-float('inf')
        maximum=torch.rand((8,1),generator=gen)*scale
        den=torch.rand((8,1),generator=gen)*30.+.25
        expected=pre_p(score,maximum,den)
        actual=forward_api.pre_p_forward(*(x.to('npu:1') for x in (score,maximum,den)))
        torch.save({'inputs':(score,maximum,den),'actual':tuple(x.cpu() for x in actual),'expected':expected},out/f'random-mask-{seed}.pt')
        for label,a,b in zip(('maximum','denominator','alpha','probabilities'),actual,expected):
            a=a.cpu();eq=torch.equal(a.view(torch.int32),b.view(torch.int32))
            r['tests'].append({'name':f'random-mask-{seed}-{label}','bit_equal':eq,'different':int((a.view(torch.int32)!=b.view(torch.int32)).sum()),'max_abs':float((a-b).abs().max())})
    r['peak_npu_allocated_bytes']=torch.npu.max_memory_allocated()
    r['passed']=all(t.get('bit_equal',t.get('passed',True)) for t in r['tests'])
    r['status']='passed' if r['passed'] else 'needs_numerical_review'
    (out/'report.json').write_text(json.dumps(r,indent=2))
    if not r['passed']:raise SystemExit(2)
finally:
    for name in ('report.json','report.partial.json'):
        p=out/name
        if p.exists():
            d=json.loads(p.read_text());d['probe_pre_p_stages']=2;d['independent_flag_slots']=2;d['candidate_sha256']={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('candidate_fused_mxfp8_slots.py','candidate_forward_slots.py')};p.write_text(json.dumps(d,indent=2))
