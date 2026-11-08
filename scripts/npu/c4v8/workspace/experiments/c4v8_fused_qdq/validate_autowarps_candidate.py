"""Probe only: use pinned Ascend compiler's num_warps=-1 auto policy."""
import os,sys,json
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root),str(root.parent/'c4v8_fused_nvfp4'),str(root.parent/'c4v8_device_core')]
import forward_api
class AutoKernel:
    def __init__(self,kernel):self.kernel=kernel
    def __getitem__(self,grid):
        def launch(*a,**kw):
            kw.pop('num_warps',None)
            return self.kernel[grid](*a,**kw)
        return launch
for name in ('pre_p_prepare_kernel','pre_p_denominator_kernel','post_pv_kernel','normalize_kernel'):
    setattr(forward_api,name,AutoKernel(getattr(forward_api,name)))
from validate_npu import main
out=Path(sys.argv[sys.argv.index('--output-dir')+1])
try:main()
finally:
    for name in ('report.json','report.partial.json'):
        p=out/name
        if p.exists():
            d=json.loads(p.read_text());d['probe_compiler_options']={'num_warps':'omitted; Ascend default -1'};p.write_text(json.dumps(d,indent=2))
