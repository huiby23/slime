"""Fail-closed configuration/provenance gates for the dedicated C4V8 recipe.

Stdlib only: reading this module never loads torch, Ray, or an accelerator.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from numerical_profile import implementation_profile, python_paths, source_files, IMPLEMENTATION_IDS

BACKEND='hby_c4v8_sim'
LABEL='FA NVFP4 C4 + MXFP8 V8 forward QDQ; high-precision identity-STE backward'
GATE_CONTRACT_VERSION=2
GATE_CHECKS=('npu_core_forward_ste_backward','megatron_core_npu_smoke',
             'attention_adapter_teacher_forcing_agreement','sg_prefill_decode_policy_verified',
             'real_single_optimizer_update','evaluation_uses_same_policy',
             'representative_length_throughput','no_unexpected_tensor_cpu_fallback')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_digest(policy):
    return hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def storage_plan(model, destination, *, steps=100, interval=5, rollouts=5,
                 reserve_gib=64, free_bytes=None):
    """Conservative full-HF retention requirement; never deletes or prunes.

    The original every5 recipe keeps every rollout weight version as well as
    every step snapshot. Final publication is a same-filesystem hardlink, but
    no deduplication between separately exported versions is assumed.
    """
    model,destination=Path(model),Path(destination)
    index=json.loads((model/'model.safetensors.index.json').read_text())
    shards=sorted(set(index['weight_map'].values()))
    if not shards or any(Path(s).name!=s for s in shards):
        raise ValueError('Expected local HF shard basenames')
    tensor_bytes=sum((model/s).stat().st_size for s in shards)
    if steps<1 or interval<1 or steps%interval or rollouts<1:
        raise ValueError('Inexact checkpoint plan')
    # Include configs/tokenizer/metadata plus a generous 2% per-export margin.
    export_bytes=(tensor_bytes*102+99)//100
    snapshots=steps//interval
    versions=rollouts+1
    temporary_exports=1
    required=(snapshots+versions+temporary_exports)*export_bytes+int(reserve_gib*1024**3)
    if free_bytes is None:free_bytes=shutil.disk_usage(destination).free
    return dict(tensor_bytes_per_hf=tensor_bytes,estimated_bytes_per_export=export_bytes,
        retained_step_snapshots=snapshots,retained_rollout_versions=versions,
        temporary_exports=temporary_exports,reserve_bytes=int(reserve_gib*1024**3),
        required_free_bytes=required,available_bytes=free_bytes,
        shortfall_bytes=max(0,required-free_bytes),passed=free_bytes>=required,
        policy='All every5 full snapshots retained; no deletion, pruning or speculative deduplication')


def source_inventory(workspace, recipe, module='device_attention'):
    workspace=Path(workspace)
    roots=[workspace/'experiments/c4v8_rl_training_bridge',
           workspace/'experiments/c4v8_device_core',
           workspace/'experiments/c4v8_rollout_bridge']
    sources=[Path(recipe),Path(__file__),Path(recipe).with_name('c4v8_validation_protocol.py'),
             Path(recipe).with_name('run-c4v8-runtime.sh'),
             Path(recipe).with_name('run-c4v8-dapo100-every5.sh')]
    sources.extend(source_files(workspace,module))
    sources.append(Path(recipe).with_name('numerical_profile.py'))
    for root in roots:
        # Top-level runtime source only: exclude mutable evidence and caches.
        sources.extend(root.glob('*.py'))
        sources.extend(root.glob('*.sh'))
    sources.extend([
        roots[2]/'python-c4v8',
        workspace/'slime/scripts/npu/qwen30b_dapo.py',
        workspace/'slime/slime/backends/megatron_utils/model_provider.py',
        workspace/'deps/sglang/python/sglang/srt/model_loader/loader.py',
        workspace/'deps/sglang/python/sglang/srt/model_executor/model_runner.py',
        workspace/'deps/sglang/python/sglang/srt/managers/scheduler_update_weights_mixin.py',
        workspace/'setup/preallocate-hybrid-adam.py'])
    missing=[str(p) for p in sources if not p.is_file()]
    inventory={str(p.resolve()):sha(p) for p in sources if p.is_file()}
    return inventory,missing


def prepare_run_context(options, extra, recipe):
    w=Path(options.workspace).resolve()
    model=Path(options.model).resolve() if options.model else w/'models/Qwen3-30B-A3B'
    if model != w/'models/Qwen3-30B-A3B':raise ValueError('C4V8 comparison starts from the official original model')
    expected={'parent_completed_optimizer_steps':0,'rollouts':5,'prompts':5,'samples':16,
        'minibatch':4,'hf_checkpoint_every_steps':5,'offload_fraction':0.35,
        'response_length':4096,'sequence_length':8192,'learning_rate':1e-6,'seed':1234}
    for key,value in expected.items():
        if getattr(options,key)!=value:raise ValueError(f'Changed fixed C4V8 comparison option: {key}')
    if (options.parent_run_manifest or options.offline_parent_run_manifest
            or not options.offline_checkpoint_validation or not options.keep_all_weight_versions
            or options.lineage_target_optimizer_steps!=100 or not options.validation_trajectory_dir):
        raise ValueError('Require fresh100, every5 retained HF snapshots and offline0..100 evaluation')
    if any(name.startswith('SLIME_NPU_REPLAY0_') and value for name,value in os.environ.items()):
        raise ValueError('BF16-policy cached rollout must not be replayed as a C4V8-policy rollout')
    policy_path=Path(os.environ['HBY_C4V8_POLICY_FILE']).resolve()
    policy=json.loads(policy_path.read_text())
    keys={'query_block','key_block','vector','quantize_operands','causal','reduction'}
    if (set(policy)!=keys or policy['vector']!='mxfp8' or policy['quantize_operands'] is not True
            or policy['causal'] is not True or policy['reduction']!='mxfp8_tree_reduction'
            or type(policy['query_block']) is not int or type(policy['key_block']) is not int
            or min(policy['query_block'],policy['key_block'])<1):
        raise ValueError('Explicit complete forward C4V8 baseline policy required')
    module=os.environ.get('HBY_C4V8_CORE_MODULE')
    if module not in IMPLEMENTATION_IDS:raise ValueError('An explicit reviewed core module is required')
    implementation=implementation_profile(w,module)
    if os.environ.get('HBY_C4V8_IMPLEMENTATION_ID')!=implementation['implementation_id']:
        raise ValueError('Core module and explicit implementation ID differ')
    if module=='fast_attention' and (policy['query_block'] not in (64,256,512,1024) or policy['key_block']!=128):
        raise ValueError('This fast candidate requires an explicit Q64/256/512/1024 and K128 policy; launch evidence must match the exact policy SHA')
    if module=='fast_attention' and os.environ.get('HBY_C4V8_ALLOW_UNVALIDATED_FUSION')!='1':
        raise ValueError('Candidate fusion requires explicit opt-in; launch gates still apply')
    dirs=python_paths(w,module)
    env={'HBY_C4V8_CORE_MODULE':module,'HBY_C4V8_POLICY_FILE':str(policy_path),
         'HBY_C4V8_IMPLEMENTATION_ID':implementation['implementation_id'],
         'HBY_C4V8_IMPLEMENTATION_SHA256':policy_digest(implementation),
         'HBY_C4V8_POLICY_SHA256':policy_digest(policy),
         'TRITON_CACHE_DIR':str(w/'cache/triton-c4v8'),
         'SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION':'false',
         'PYTHONPATH':os.pathsep.join(map(str,dirs))+os.pathsep+os.environ.get('PYTHONPATH','')}
    if module=='fast_attention':env['HBY_C4V8_ALLOW_UNVALIDATED_FUSION']='1'
    os.environ.update(env)
    # PYTHONPATH changes only future interpreters; this driver is already live.
    sys.path[:0]=[str(path) for path in dirs if str(path) not in sys.path]
    inventory,missing=source_inventory(w,recipe,module)
    capacity=storage_plan(model,w)
    dry=options.print_command or options.validate_arguments
    gate_path=os.environ.get('HBY_C4V8_LAUNCH_GATE')
    gate=json.loads(Path(gate_path).read_text()) if gate_path and Path(gate_path).is_file() else {}
    validated=(gate.get('status')=='passed' and gate.get('gate_contract_version')==GATE_CONTRACT_VERSION
               and gate.get('policy_sha256')==env['HBY_C4V8_POLICY_SHA256']
               and gate.get('implementation_sha256')==env['HBY_C4V8_IMPLEMENTATION_SHA256']
               and gate.get('source_sha256')==inventory and not missing
               and all(gate.get('checks',{}).get(name) is True for name in GATE_CHECKS))
    audit=dict(label=LABEL,policy=policy,policy_file=str(policy_path),policy_sha256=env['HBY_C4V8_POLICY_SHA256'],
        core_module=module,implementation=implementation,implementation_sha256=env['HBY_C4V8_IMPLEMENTATION_SHA256'],
        policy_file_sha256=sha(policy_path),
        source_sha256=inventory,missing_runtime_sources=missing,launch_gate_file=gate_path,
        launch_gate_passed=validated,gate_contract_version=GATE_CONTRACT_VERSION,
        teacher_forcing_scope='same post-QKnorm/RoPE QKV through actual FA adapters; not whole-model logprobs',
        whole_model_logprob_agreement='unknown',
        prelaunch_optimizer_update_scope='real small-model Adam update; not a30B GRPO update',
        first_30b_optimizer_update='runtime pilot observation; never fabricated as prelaunch evidence',
        capacity=capacity,backbone_precision='BF16',
        backward_precision='FP32 arithmetic/gradient accumulation with identity STE; not low-bit backward',
        source_mode='fresh original Qwen3-30B-A3B; all rollout answers generated by C4V8 policy',
        prepared_utc=datetime.now(timezone.utc).isoformat(),dry_run=dry)
    if not dry:
        if not capacity['passed']:
            raise RuntimeError('Insufficient disk for required every5 full-HF retention: '+json.dumps(capacity))
        if not validated:
            raise RuntimeError('C4V8 launch evidence incomplete/stale; preserve policy and actual source SHA')
    return env,audit


def register_driver_backend():
    from c4v8_bootstrap import register_cli_choice
    register_cli_choice()


def verify_resolved_config(args, options, env):
    required={'bf16':True,'num_rollout':5,'rollout_batch_size':5,'n_samples_per_prompt':16,
        'global_batch_size':4,'micro_batch_size':1,'num_steps_per_rollout':20,
        'rm_type':'math','advantage_estimator':'grpo','optimizer_offload_fraction':0.35,
        'offload_train':False,'offload_rollout':True,'disable_actor_cpu_backup':True,
        'optimizer_cpu_offload':True,'use_precision_aware_optimizer':True,
        'tensor_model_parallel_size':4,'pipeline_model_parallel_size':2,
        'expert_model_parallel_size':4,'expert_tensor_parallel_size':1,'context_parallel_size':1,
        'rollout_max_response_len':4096,'seq_length':8192,
        'sglang_attention_backend':BACKEND,'sglang_prefill_attention_backend':BACKEND,
        'sglang_decode_attention_backend':BACKEND,'sglang_disable_cuda_graph':True}
    for key,value in required.items():
        if getattr(args,key,None)!=value:raise ValueError(f'Resolved C4V8 policy/config mismatch: {key}')
    if args.spec!=['megatron_spec','get_spec']:
        raise ValueError('Trainer does not select the reviewed C4V8 core spec')
    if args.custom_rm_path is not None or args.save_debug_rollout_data is not None or args.load_debug_rollout_data is not None:
        raise ValueError('Real math rewards and real C4V8 rollout generation are required')
    if args.dynamic_sampling_filter_path!='slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std':
        raise ValueError('The strict nonzero reward-variance filter changed')
    chat=args.apply_chat_template_kwargs
    if isinstance(chat,str):chat=json.loads(chat)
    if chat!={'enable_thinking':False}:raise ValueError('Nonthinking training must match the BF16 baseline')
    for key in ('main_params_dtype','exp_avg_dtype','exp_avg_sq_dtype'):
        if str(getattr(args,key,None)).split('.')[-1] not in ('float32','fp32'):
            raise ValueError('FP32 master parameters and Adam states must be retained: '+key)
    actual=args.train_env_vars
    if isinstance(actual,str):actual=json.loads(actual)
    if any(actual.get(k)!=v for k,v in env.items()):raise ValueError('Trainer did not receive the exact shared policy environment')
    if getattr(args,'fp8',None) or getattr(args,'fp4',None):raise ValueError('Global backbone quantization is forbidden')
