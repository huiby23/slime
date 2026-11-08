"""Stdlib-only explicit implementation identity, separate from math policy.

Reading a candidate profile never imports torch, Triton or a numerical core.
Source/toolchain identity is not evidence of compiled-kernel correctness.
"""
import hashlib
from importlib import metadata
import json
from pathlib import Path

IMPLEMENTATION_IDS={
    'device_attention':'device_attention_forward_ste_v1',
    'fast_attention':'fast_attention_fused_forward_ste_staged_decode_v2',
}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def source_files(workspace,module):
    if module not in IMPLEMENTATION_IDS:raise ValueError('Unreviewed numerical implementation module: '+str(module))
    root=Path(workspace)/'experiments'
    paths=[root/'c4v8_device_core'/name for name in ('device_attention.py','device_quant.py','exact_fp32.py')]
    if module=='fast_attention':
        paths += [root/'c4v8_fast_attention/fast_attention.py',
            root/'c4v8_fused_nvfp4/nvfp4_triton.py',
            *[root/'c4v8_fused_qdq'/name for name in ('forward_api.py','fused_mxfp8.py','whole_decode.py')],
            *[root/'c4v8_staged_decode'/name for name in ('staged_decode.py','staged_kernels.py')]]
    paths.append(root/'c4v8_rl_training_bridge/numerical_profile.py')
    return paths


def python_paths(workspace,module):
    if module not in IMPLEMENTATION_IDS:raise ValueError('Unreviewed numerical module')
    root=Path(workspace)/'experiments'
    names=['c4v8_rl_training_bridge','c4v8_device_core','c4v8_rollout_bridge']
    if module=='fast_attention':names+=['c4v8_fast_attention','c4v8_fused_nvfp4','c4v8_fused_qdq','c4v8_staged_decode']
    return [root/name for name in names]


def implementation_profile(workspace,module):
    files=source_files(workspace,module)
    hashes={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    canonical_helper=Path(workspace)/'experiments/c4v8_rl_training_bridge/numerical_profile.py'
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=hashes[str(canonical_helper.resolve())]:
        raise ValueError('Prepared implementation helper differs from the canonical helper source')
    versions={}
    for name in ('torch','torch-npu','triton','triton-ascend'):
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]=None
    return dict(core_module=module,implementation_id=IMPLEMENTATION_IDS[module],
        source_sha256=hashes,toolchain_distribution_versions=versions,
        arithmetic=(dict(final_normalization='signed exact FP32 RNE division then MXFP8 QDQ',
                         decode_reduction='staged decode uses ordinary framework FP32 QK/PV GEMM with the same NVFP4/MXFP8 boundaries',
                         decode_execution='staged_decode only for one query at final KV position, no input requires_grad, D128 and key_block128; all other calls use generic tiled execution',
                         training_execution='generic tiled forward and high-precision STE backward, including query-tile replay',
                         whole_decode_candidate='monolithic whole_decode source inventoried but not routed; distinct from the validated staged_decode implementation')
                    if module=='fast_attention' else dict(final_normalization='framework FP32 division then MXFP8 QDQ',
                                                         decode_reduction='ordinary framework FP32 GEMM')),
        backward='High-precision identity-STE; no low-bit backward',
        compiled_kernel_validation='Separate actual-device/source-matched launch evidence required')
