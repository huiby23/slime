"""Probe-only kernel injection: isolates final tree gather-to-one replacement."""
import os,sys
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root),str(root.parent/'c4v8_fused_nvfp4'),str(root.parent/'c4v8_device_core')]
import forward_api
import candidate_fused_mxfp8_select as candidate
# Explicit local process only. Main files and formal role selectors are unchanged.
forward_api.pre_p_prepare_kernel=candidate.pre_p_prepare_kernel
forward_api.pre_p_denominator_kernel=candidate.pre_p_denominator_kernel
from validate_npu import main
if __name__=='__main__':main()
