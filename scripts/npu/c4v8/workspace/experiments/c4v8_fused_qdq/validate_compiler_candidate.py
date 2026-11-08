"""Probe-only compiler-option injection; no production function/source changes."""
import os,sys,json
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root),str(root.parent/'c4v8_fused_nvfp4'),str(root.parent/'c4v8_device_core')]
mode=sys.argv.pop(1)
options={'inject_barrier_all':True} if mode=='barrier' else {'multibuffer':False} if mode=='singlebuffer' else None
if options is None:raise ValueError('Expected barrier or singlebuffer')
import forward_api
class OptionsKernel:
    def __init__(self,kernel):self.kernel=kernel
    def __getitem__(self,grid):
        def launch(*a,**kw):
            for k,v in options.items():
                if k in kw and kw[k]!=v:raise ValueError('Conflicting option')
                kw[k]=v
            return self.kernel[grid](*a,**kw)
        return launch
for name in ('pre_p_prepare_kernel','pre_p_denominator_kernel'):
    setattr(forward_api,name,OptionsKernel(getattr(forward_api,name)))
from validate_npu import main
out=Path(sys.argv[sys.argv.index('--output-dir')+1])
try:main()
finally:
    for name in ('report.json','report.partial.json'):
        p=out/name
        if p.exists():
            d=json.loads(p.read_text());d['probe_compiler_options']=options;p.write_text(json.dumps(d,indent=2))
