"""Explicit parent/worker bootstrap. No installation or hardware use at import."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path

from sglang_bridge import BACKEND_NAME


_INSTALLED = None


def register_cli_choice():
    """Driver-side parser preparation only; no numerical core or NPU-pool import."""
    from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES, add_attention_backend_choices
    if BACKEND_NAME not in ATTENTION_BACKEND_CHOICES:
        add_attention_backend_choices([BACKEND_NAME])


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _load_profile():
    module_name = os.environ.get('HBY_C4V8_CORE_MODULE')
    policy_name = os.environ.get('HBY_C4V8_POLICY_FILE')
    audit_name = os.environ.get('HBY_C4V8_ROLLOUT_AUDIT_DIR')
    if module_name not in ('device_attention','fast_attention') or not policy_name or not audit_name:
        raise ValueError('Explicit reviewed core, shared policy file, and audit directory are required')
    policy_path = Path(policy_name).resolve(strict=True)
    policy = json.loads(policy_path.read_text())
    required = {'query_block', 'key_block', 'vector', 'quantize_operands', 'causal', 'reduction'}
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError('Policy must explicitly contain exactly all six AttentionConfig fields')
    if (policy['vector'] != 'mxfp8' or policy['quantize_operands'] is not True or
            policy['causal'] is not True or policy['reduction'] != 'mxfp8_tree_reduction'):
        raise ValueError('This launcher is for the declared full-forward C4V8 baseline only')
    requested_policy_sha = os.environ.get('HBY_C4V8_POLICY_SHA256')
    if requested_policy_sha is not None and requested_policy_sha != _digest(policy):
        raise ValueError('Shared policy identity differs from the training/launcher policy hash')
    root = Path(__file__).resolve().parent
    workspace=root.parent.parent
    helper_path=workspace/'experiments/c4v8_rl_training_bridge/numerical_profile.py'
    helper_spec=importlib.util.spec_from_file_location('hby_c4v8_numerical_profile',helper_path)
    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    implementation=helper.implementation_profile(workspace,module_name)
    implementation_sha=helper.digest(implementation)
    expected_id=os.environ.get('HBY_C4V8_IMPLEMENTATION_ID')
    expected_implementation_sha=os.environ.get('HBY_C4V8_IMPLEMENTATION_SHA256')
    if module_name=='fast_attention' and (expected_id is None or expected_implementation_sha is None
            or os.environ.get('HBY_C4V8_ALLOW_UNVALIDATED_FUSION')!='1'):
        raise ValueError('Fused candidate requires explicit implementation ID/hash and opt-in')
    if expected_id is not None and expected_id!=implementation['implementation_id']:
        raise ValueError('Implementation ID differs from the selected core')
    if expected_implementation_sha is not None and expected_implementation_sha!=implementation_sha:
        raise ValueError('Implementation source/toolchain identity differs from the training launcher')
    core = importlib.import_module(module_name)
    config = core.AttentionConfig(**policy)
    core_path = Path(core.__file__).resolve(strict=True)
    if str(core_path) not in implementation['source_sha256']:
        raise ValueError('Imported core is not the canonical implementation source')
    files = [root/'logical_kv.py', root/'sglang_bridge.py', root/'hardware_compat.py',
             root/'c4v8_bootstrap.py', root/'c4v8_runtime.py', root/'python-c4v8']
    source_hashes = dict(implementation['source_sha256'])
    source_hashes.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    identity = {'backend': BACKEND_NAME, 'policy': policy, 'sources': source_hashes,
                'implementation_sha256':implementation_sha}
    runtime_sha = _digest(identity)
    expected = os.environ.get('HBY_C4V8_EXPECTED_RUNTIME_SHA256')
    if expected is not None and runtime_sha != expected:
        raise ValueError('Source/policy identity differs between parent and spawned worker')
    os.environ['HBY_C4V8_EXPECTED_RUNTIME_SHA256'] = runtime_sha
    audit = Path(audit_name).resolve()
    audit.mkdir(parents=True, exist_ok=True)
    return {'core': core, 'config': config, 'policy': policy,
            'policy_path': str(policy_path), 'policy_sha256': _digest(policy),
            'source_sha256': source_hashes, 'runtime_sha256': runtime_sha,
            'implementation':implementation,'implementation_id':implementation['implementation_id'],
            'implementation_sha256':implementation_sha,'core_module':module_name,
            'audit_dir': audit}


def _record(profile, event, data):
    # One process owns each file; O_EXCL rejects accidental evidence replacement.
    data=dict(data)
    if data.pop('core_module',profile['core_module'])!=profile['core_module']:
        raise ValueError('Actual backend core differs from the installed profile')
    report = dict(event=event, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  pid=os.getpid(), ppid=os.getppid(), backend=BACKEND_NAME,
                  policy=profile['policy'], policy_sha256=profile['policy_sha256'],
                  runtime_sha256=profile['runtime_sha256'],
                  implementation_id=profile['implementation_id'],
                  implementation_sha256=profile['implementation_sha256'],
                  core_module=profile['core_module'],
                  execution_validation_kind=('cpu_stubbed_sglang' if os.environ.get('HBY_C4V8_CPU_SPAWN_PROBE') == '1'
                                             else 'actual_process'),
                  precision_scope='full-forward NVFP4/MXFP8 QDQ; training backward remains explicit STE',
                  **data)
    name = profile['audit_dir'] / f'{event}-pid{os.getpid()}.json'
    with name.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def install_current_process(role, ranks=None):
    global _INSTALLED
    profile = _load_profile()
    if _INSTALLED is not None:
        if _INSTALLED['runtime_sha256'] != profile['runtime_sha256']:
            raise RuntimeError('Cannot replace an installed numerical profile')
        return _INSTALLED
    from sglang_bridge import install_in_sglang
    if os.environ.get('HBY_C4V8_CPU_SPAWN_PROBE') == '1':
        fp32_mode={'applied':False,'reason':'explicit CPU stubbed probe; no NPU import',
                   'requested_allow_hf32':False,'actual_allow_hf32':None}
    else:
        # Process-local C4V8 simulation needs FP32 GEMM, matching its numerical
        # reference probe. Do not rely on the library's ambient HF32 default.
        import torch
        import torch_npu
        torch.npu.matmul.allow_hf32=False
        actual=bool(torch.npu.matmul.allow_hf32)
        if actual:raise RuntimeError('Could not disable HF32 for C4V8 simulator GEMM')
        fp32_mode={'applied':True,'requested_allow_hf32':False,'actual_allow_hf32':actual,
                   'scope':'this explicit C4V8 process FP32 matmul; no BF16 dtype change'}
    def audit(event, values):
        _record(profile, event, dict(role=role, ranks=ranks or {}, **values))
    install_in_sglang(core=profile['core'].attention, config=profile['config'], audit_callback=audit)
    _record(profile, 'bootstrap_registered', {
        'role': role, 'ranks': ranks or {}, 'source_sha256': profile['source_sha256'],
        'policy_path': profile['policy_path'],'fp32_simulator_mode':fp32_mode,
        'implementation':profile['implementation'],
        'environment': {name: os.environ.get(name) for name in
                        ('PYTORCH_NPU_ALLOC_CONF', 'MAX_COMPILE_CORE_NUMBER', 'HCCL_BUFFSIZE',
                         'SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK', 'SGLANG_LOGITS_PROCESSER_CHUNK_SIZE',
                         'SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION')},
    })
    _INSTALLED = profile
    return profile


def verify_server_args(server_args):
    selected = {name: getattr(server_args, name, None) for name in
                ('attention_backend', 'prefill_attention_backend', 'decode_attention_backend')}
    if set(selected.values()) != {BACKEND_NAME}:
        raise ValueError(f'Actual backend selectors are not all {BACKEND_NAME}: {selected}')
    if server_args.dtype not in ('bfloat16', 'bf16') or server_args.kv_cache_dtype not in ('auto', 'bfloat16'):
        raise ValueError('Explicit BF16 model and BF16/auto KV dtype are required')
    if not server_args.disable_cuda_graph or server_args.enable_torch_compile:
        raise ValueError('Graphs and torch.compile must be disabled')
    if server_args.enable_mixed_chunk or server_args.speculative_algorithm not in (None, 'NONE'):
        raise ValueError('Mixed/speculative generation is not supported by this profile')
    if getattr(server_args, 'enable_pdmux', False) or getattr(server_args, 'enable_two_batch_overlap', False):
        raise ValueError('PD multiplexing and two-batch overlap are outside this first bridge')
    if server_args.device != 'npu':
        raise ValueError('This audited serving launcher targets the Ascend NPU path')
    if server_args.quantization is not None:
        raise ValueError('Do not apply backbone/weight quantization in this FA-only experiment')
    if server_args.dp_size != 1 or server_args.pp_size != 1 or server_args.nnodes != 1:
        raise ValueError('Initial audited serving path is one node with TP only')
    if server_args.encoder_only or server_args.grpc_mode:
        raise ValueError('Initial audited serving path is ordinary causal HTTP serving')
    return dict(selected, dtype=server_args.dtype, kv_cache_dtype=server_args.kv_cache_dtype,
                tp_size=server_args.tp_size, pp_size=server_args.pp_size, dp_size=server_args.dp_size,
                enable_memory_saver=server_args.enable_memory_saver,
                device=server_args.device, model_path=server_args.model_path,
                context_length=server_args.context_length,
                weight_version=server_args.weight_version)


def audit_effective_server_args(server_args, role):
    if _INSTALLED is None:
        raise RuntimeError('No numerical backend installed in this process')
    actual = verify_server_args(server_args)
    _record(_INSTALLED, 'effective_server_args', {'role': role, 'actual': actual})


def run_scheduler_process(server_args, port_args, gpu_id, tp_rank, moe_ep_rank,
                          pp_rank, dp_rank, pipe_writer):
    """Top-level picklable spawn target; registration is repeated in each child."""
    ranks = dict(gpu_id=gpu_id, tp_rank=tp_rank, moe_ep_rank=moe_ep_rank,
                 pp_rank=pp_rank, dp_rank=dp_rank)
    profile = install_current_process('scheduler', ranks)
    audit_effective_server_args(server_args, 'scheduler')
    # Imported only after this worker has its own registration and NPU selector.
    from sglang.srt.managers.scheduler import run_scheduler_process as original
    try:
        return original(server_args, port_args, gpu_id, tp_rank, moe_ep_rank,
                        pp_rank, dp_rank, pipe_writer)
    except BaseException as error:
        _record(profile, 'scheduler_wrapper_error', {'role': 'scheduler', 'ranks': ranks,
                                                    'error_type': type(error).__name__, 'error': str(error)})
        raise
