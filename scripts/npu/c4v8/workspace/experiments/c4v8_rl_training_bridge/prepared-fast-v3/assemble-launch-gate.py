#!/usr/bin/env python3
"""Read actual bounded evidence and bind it to the final runtime source inventory.

Default writes only an acceptance review. --write-gate is an explicit root
decision after every check succeeds; missing/failed evidence never writes a gate.
No accelerator import, server, optimizer, or training process is started here.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

PREPARED=Path(__file__).resolve().parent
BRIDGE=PREPARED.parent
WORKSPACE=BRIDGE.parents[1]
sys.path.insert(0,str(BRIDGE))
from numerical_profile import implementation_profile,digest
# The launcher imports its sibling copy, whose __file__ is part of inventory.
# Loading the top-level helper gives a superficially valid but incomplete gate.
_consumer_spec=importlib.util.spec_from_file_location('prepared_gate_consumer',PREPARED/'recipe_support.py')
consumer=importlib.util.module_from_spec(_consumer_spec);_consumer_spec.loader.exec_module(consumer)
source_inventory=consumer.source_inventory
GATE_CHECKS=consumer.GATE_CHECKS
GATE_CONTRACT_VERSION=consumer.GATE_CONTRACT_VERSION

FALLBACK=('npu_cpu_fallback','fall back to run on the CPU')


def require(condition,message):
    if not condition:raise ValueError(message)


def finite_numbers(value):
    if isinstance(value,float):require(math.isfinite(value),'Nonfinite JSON number')
    elif isinstance(value,dict):
        for item in value.values():finite_numbers(item)
    elif isinstance(value,list):
        for item in value:finite_numbers(item)


def hash_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_gate_consumer(gate):
    """Exercise the actual launch guard before publishing, without importing ML.

    dry=False intentionally exercises exact gate, capacity and lineage checks.
    prepare_run_context is stdlib-only and cannot construct/train a model.
    """
    options=SimpleNamespace(workspace=str(WORKSPACE),model=str(WORKSPACE/'models/Qwen3-30B-A3B'),
        parent_completed_optimizer_steps=0,rollouts=5,prompts=5,samples=16,minibatch=4,
        hf_checkpoint_every_steps=5,offload_fraction=.35,response_length=4096,sequence_length=8192,
        learning_rate=1e-6,seed=1234,parent_run_manifest=None,offline_parent_run_manifest=None,
        offline_checkpoint_validation=True,keep_all_weight_versions=True,lineage_target_optimizer_steps=100,
        validation_trajectory_dir=str(WORKSPACE/'evaluations/qwen3-30b-c4v8-fast-forward-qdq-ste-every5'),
        print_command=False,validate_arguments=False)
    environment=os.environ.copy();search_path=sys.path[:]
    try:
        with tempfile.NamedTemporaryFile(mode='w',suffix='.json',prefix='consumer-check-',dir=PREPARED) as f:
            json.dump(gate,f);f.flush()
            os.environ.update(HBY_C4V8_CORE_MODULE='fast_attention',
                HBY_C4V8_IMPLEMENTATION_ID='fast_attention_fused_forward_ste_staged_decode_v2',
                HBY_C4V8_ALLOW_UNVALIDATED_FUSION='1',
                HBY_C4V8_POLICY_FILE=str(PREPARED/'precision-policy.json'),HBY_C4V8_LAUNCH_GATE=f.name)
            for key in tuple(os.environ):
                if key.startswith('SLIME_NPU_REPLAY0_'):os.environ.pop(key)
            _,audit=consumer.prepare_run_context(options,[],PREPARED/'qwen30b_dapo_c4v8.py')
            require(audit['launch_gate_passed'] is True and audit['dry_run'] is False,
                    'Actual prepared launch consumer rejected candidate gate')
            return dict(passed=True,consumer_path=str(Path(consumer.__file__).resolve()),
                consumer_sha256=hash_file(consumer.__file__),dry_run=False,
                source_inventory_sha256=digest(audit['source_sha256']),capacity_passed=audit['capacity']['passed'],
                model_constructed=False,process_started=False,optimizer_updates=0)
    finally:
        os.environ.clear();os.environ.update(environment);sys.path[:]=search_path


def idle_text(text):
    cards=sorted(set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)',text))))
    require(cards==list(range(8)),'Release evidence does not show all8 idle')


def bit_comparison(value):
    finite=value.get('both_finite',value.get('finite',False))
    equal=value.get('bit_exact',value.get('bit_equal',False))
    signed=value.get('pure_signed_zero_difference',value.get('signed_zero_only',False))
    require(finite is True and (equal is True or
        (signed is True and value.get('numeric_equal') is True)),
        'Comparison is nonfinite or differs beyond exact signed zero')


class Evidence:
    def __init__(self,profile,policy):
        self.profile=profile;self.policy=policy;self.files={};self.results={}
    def text(self,path):
        path=Path(path).resolve();raw=path.read_bytes();value=hashlib.sha256(raw).hexdigest()
        require(str(path) not in self.files or self.files[str(path)]==value,'File changed during acceptance: '+str(path))
        self.files[str(path)]=value
        return raw.decode()
    def json(self,path):
        result=json.loads(self.text(path));finite_numbers(result);return result
    def log(self,path):
        text=self.text(path)
        require(not any(token in text for token in FALLBACK),'Unexpected tensor CPU fallback: '+str(path))
    def owned(self,root):
        root=Path(root);result=self.json(root/'result.json')
        require(result.get('status')=='exited' and result.get('child_exit_code')==0
            and result.get('execution_completed_cleanly') is True and result.get('after_all_eight_idle') is True,
            'Owned controller failed/incomplete: '+str(root))
        require(not result.get('tensor_cpu_fallback_warnings') and not result.get('cleanup_error'),
            'Owned controller has fallback/cleanup error')
        self.log(root/'outer.log')
        require(result['outer_log_sha256']==self.files[str((root/'outer.log').resolve())],'Outer log SHA mismatch')
        release=self.json(root/'npu-after.json');require(release.get('all_eight_idle') is True,'Incomplete release')
        idle_text(release['stdout'])
        owner=self.json(root/'owned-process.json')
        require(owner==result['owner'] and owner['pid']==owner['pgrp']==owner['sid'],'Owned identity mismatch')
        require('--owned-child' in owner['argv'],'Missing owned supervisor command')
    def source_subset(self,sources,names):
        actual={str((WORKSPACE/'experiments'/key).resolve()) if not Path(key).is_absolute() else str(Path(key).resolve()):value
                for key,value in sources.items()}
        for name in names:
            path=str((WORKSPACE/'experiments'/name).resolve())
            require(path in actual and actual[path]==hash_file(path),'Source mismatch/missing: '+name)
            self.text(path)  # Rechecked after every validator, closing cross-check source races.
            if path in self.profile['source_sha256']:
                require(actual[path]==self.profile['source_sha256'][path],'Numerical profile source mismatch: '+name)
    def full_identity(self,report):
        require(report.get('implementation')==self.profile and report.get('implementation_sha256',digest(report['implementation']))==digest(self.profile),
            'Implementation/source/toolchain profile mismatch')
        require(report.get('policy')==self.policy and report.get('policy_sha256')==digest(self.policy),'Policy mismatch')
    def recorded_probe_source(self,source,expected,snapshot):
        # A measurement harness may evolve after a run. Its exact archived
        # bytes prove that run; this exception never applies to runtime sources.
        chosen=Path(source) if Path(source).is_file() and hash_file(source)==expected else Path(snapshot)
        self.text(chosen);require(self.files[str(chosen.resolve())]==expected,'Probe source/snapshot SHA mismatch: '+source)


RUNTIME_NUMERICS=('c4v8_fast_attention/fast_attention.py','c4v8_device_core/device_attention.py',
 'c4v8_device_core/device_quant.py','c4v8_device_core/exact_fp32.py','c4v8_fused_nvfp4/nvfp4_triton.py',
 'c4v8_fused_qdq/forward_api.py','c4v8_fused_qdq/fused_mxfp8.py',
 'c4v8_staged_decode/staged_decode.py','c4v8_staged_decode/staged_kernels.py')


def components(e):
    base=WORKSPACE/'experiments/c4v8_fused_qdq';root=base/'evidence/npu-default-finite-v1'
    r=e.json(root/'report.json')
    require(r.get('passed') is True and r.get('status')=='passed' and len(r.get('tests',[]))==60,'Need actual60 component tests')
    require(r.get('default_api_used') is True and r.get('uses_portable_npu_reference') is False
        and r.get('actual_allow_hf32') is False and r.get('torch_npu_jit_compile') is False,'Wrong component execution path')
    for row in r['tests']:
        if 'bit_equal' in row:require(row['bit_equal'] is True,'Component bit mismatch')
        if 'passed' in row:require(row['passed'] is True,'Component case failed')
    # The legacy60-case proof predates the generic owned wrapper. Do not invent
    # one, or require its unrelated historical fast-attention hash to be current.
    shared=[name for name in RUNTIME_NUMERICS if not name.startswith(('c4v8_fast_attention/','c4v8_staged_decode/'))]
    e.source_subset(r['source_sha256'],shared)
    require(e.text(base/'logs/npu-default-finite-v1.exit').strip()=='0','Component exit !=0')
    e.log(base/'logs/npu-default-finite-v1.log');idle_text(e.text(base/'evidence/npu-after-default-finite-v1.txt'))
    nv=WORKSPACE/'experiments/c4v8_fused_nvfp4/evidence/npu-fused-only-v4'
    n=e.json(nv/'report.json')
    require(n.get('numerical_passed') is True and n.get('ste_identity_exact') is True and
        n.get('hf32') is False and n.get('jit_compile') is False,'NVFP4 component failed/wrong precision')
    require(n['sources']['nvfp4_triton.py']==hash_file(WORKSPACE/'experiments/c4v8_fused_nvfp4/nvfp4_triton.py'),'NVFP4 source changed')
    require(n.get('cases') and all(all(row.get(k) is True for k in ('values_exact','codes_exact','scales_exact')) for row in n['cases']),
        'NVFP4 values/codes/scales mismatch')
    require(e.text(nv/'exit').strip()=='0','NVFP4 process exit !=0')
    e.log(nv/'complete-process.log');audit=e.json(nv/'backend-log-audit.json')
    require(audit.get('passed') is True and not audit.get('matches') and audit['log_sha256']==e.files[str((nv/'complete-process.log').resolve())],
        'NVFP4 log audit mismatch')
    idle_text(e.text(nv/'npu-after-process-exit.txt'))
    return dict(passed=True,scope='Unchanged NVFP4/MXFP8 primitive proofs; legacy explicit exit/log/release evidence, no invented owned wrapper')


def postcheck(e,root):
    e.owned(root);r=e.json(root/'validation/report.json')
    require(r.get('diagnostic_completed') is True and r.get('sources_unchanged') is True
        and r.get('strict_post_change_passed') is True and r.get('strict_decode_route_passed') is True
        and r.get('all_finite') is True and r.get('device')=='npu:1','Final core postcheck failed')
    require(r.get('new_query_block')==e.policy['query_block'] and r.get('key_block')==e.policy['key_block'],'Postcheck tile mismatch')
    e.source_subset(r['source_sha256'],RUNTIME_NUMERICS+('c4v8_rl_training_bridge/bridge.py',))
    require({140,513,1025}<={row['length'] for row in r['cases']},'Postcheck length coverage incomplete')
    for row in r['cases']:
        require(row.get('all_forward_and_gradient_values_finite') is True,'Nonfinite postcheck row')
        for name in ('same_q1024_output','new_staged_vs_new_generic_q1','new_staged_q1_vs_new_prefill_last'):
            bit_comparison(row[name])
        require(len(row['same_q1024_gradients'])==3,'Missing QKV gradient comparisons')
        for result in row['same_q1024_gradients']:bit_comparison(result)
    return dict(passed=True,scope=r['scope'],full_model_logprob_agreement='unknown')


def adapter(e,root):
    e.owned(root);r=e.json(root/'adapter/report.json');e.full_identity(r)
    require(r.get('passed') is True and r.get('npu_executed') is True and
        r.get('attention_adapter_teacher_forcing_agreement') is True,'Adapter test not actual/passed')
    require(r['source_sha256']==r['source_sha256_after'],'Adapter sources changed during execution')
    for value in r['checks'].values():
        if isinstance(value,bool):require(value,'Adapter boolean check failed')
        else:
            for item in value if isinstance(value,list) else [value]:bit_comparison(item)
    for field in ('full_audit','prefix_audit'):
        require(all(r[field].get(k) is True for k in ('cached_prefix_unchanged','final_cache_exact','invalid_write_rejected_before_mutation')),
                'Adapter cache/append identity failed')
    e.source_subset(r['source_sha256'],RUNTIME_NUMERICS+('c4v8_rl_training_bridge/bridge.py','c4v8_rl_training_bridge/layouts.py',
        'c4v8_rl_training_bridge/megatron_spec.py','c4v8_rollout_bridge/sglang_bridge.py','c4v8_rollout_bridge/logical_kv.py'))
    return dict(passed=True,scope='Post-QKnorm/RoPE same-QKV attention adapter teacher forcing only; not whole-model logprobs')


def tiny(e,root):
    e.owned(root);r=e.json(root/'tiny/report.json');e.full_identity(r)
    require(r.get('status')=='passed' and r.get('execution_mode')=='npu' and r.get('npu_kernels_executed') is True,'Tiny did not execute NPU')
    require(r.get('actual_optimizer_steps')==1 and r.get('real_grpo_updates')==0 and r.get('quantized_backward') is False,'Wrong tiny scope')
    require(r.get('fp32_master_and_adam_states') is True and r.get('optimizer_cpu_offload_fraction')==.35,'Tiny Adam precision/offload changed')
    require(r['gradient_norm']>0 and r['changed_parameters'] and all(x['changed_values']>0 for x in r['changed_parameters']),
        'No finite positive gradient/parameter change')
    require(r['spec']['real_pinned_module_spec'] is True and r['spec']['core_replaced'] is True and r['spec']['original_spec_unchanged'] is True,'Tiny spec mismatch')
    require(r['trainer_fp32_simulator_mode']['actual_allow_hf32'] is False,'Tiny HF32 not disabled')
    e.source_subset(r['source_sha256'],RUNTIME_NUMERICS+('c4v8_rl_training_bridge/megatron_spec.py',))
    return dict(passed=True,scope='One real tiny random-model Adam update; 0 real GRPO updates',gradient_norm=r['gradient_norm'])


def performance(e,root):
    e.owned(root);r=e.json(root/'measurements/report.json')
    require(r.get('status')=='measurement_completed' and r.get('measurement_device')=='physical_npu1'
        and r.get('all_requested_cases_completed') is True and r.get('sources_unchanged') is True
        and r.get('actual_hf32') is False and r.get('policy')==e.policy,'Performance report incomplete/wrong source/policy')
    e.source_subset(r['source_sha256'],RUNTIME_NUMERICS+('c4v8_rl_training_bridge/bridge.py',))
    require({'recomputed-n4096','recomputed-n8192','decode-prefix4096','decode-prefix8192'}<={x['id'] for x in r['cases']},'Missing representative length')
    for row in r['cases']:
        require(row.get('completed') is True,'Incomplete performance case')
        for iteration in row['iterations']:
            require(iteration.get('output_finite') and all(iteration['output_finite']),'Nonfinite benchmark output')
            if row['kind']=='training':
                require(len(iteration['gradient_finite'])==3 and all(iteration['gradient_finite']) and
                    all(iteration['gradient_nonzero']),'Invalid representative QKV gradient')
            require(iteration['forward']['seconds']>0,'Invalid forward timing')
        require(row['warm_forward']['samples']>=1,'No warm measurement')
    return dict(passed=True,scope=r['timing_scope'],limitation=r['feasibility_caveat'])


def evaluator(e,path):
    r=e.json(path);m=e.json(r['manifest']);a=m['attention_simulation'];e.full_identity(a)
    require(r.get('passed') is True and r.get('dataset_questions')==30 and r.get('expected_steps')==list(range(0,101,5))
        and r.get('actual_optimizer_steps')==0 and r.get('completed_validation_points')==0,'Wrong evaluator artifact scope')
    require(r['policy_sha256']==digest(e.policy) and r['protocol_sha256']==m['validation_protocol_sha256'],'Evaluator policy mismatch')
    require(hash_file(m['validation_protocol_helper'])==m['validation_protocol_helper_sha256'],'Evaluator frozen protocol changed')
    e.text(m['validation_protocol_helper'])
    # Controller-only evidence-label edits do not invalidate the unchanged
    # numerical protocol/client/grader test. Every actual evaluator source does.
    names=('eval-c4v8-trajectory.py','eval-c4v8-client.py','eval_support.py','eval_deadlines.py','c4v8_eval_protocol.py',
        'c4v8_eval_inherited.py','c4v8_eval_runtime_support.py','run-c4v8-eval-server.sh')
    e.source_subset(a['source_sha256'],tuple('c4v8_rl_training_bridge/'+n for n in names))
    grader=Path(path).with_name('c4v8-evaluation-reward-preflight.json');g=e.json(grader)
    require(g.get('passed') is True and g.get('full_fixed_eval_dataset_rows')==30 and hash_file(grader)==r['reward_preflight_sha256'],
        'Actual grader artifact mismatch')
    return dict(passed=True,scope='Actual CPU evaluator/protocol/grader artifacts; 0 evaluated model points')


def sg_pilot(e,root):
    e.owned(root);p=root/'pilot';r=e.json(p/'manifest.json');e.full_identity(r)
    require(r.get('completed') is True and r.get('requests_completed') is True and r.get('actual_http_performed') is True
        and r.get('npu_process_started') is True and r.get('optimizer_updates')==0,'30B SG pilot not completed')
    require(r.get('model')==str(WORKSPACE/'models/Qwen3-30B-A3B') and r['server_args']['tp_size']==8,'Wrong actual model/TP')
    require(not r.get('tensor_cpu_fallback_warnings') and not r.get('cleanup_error'),'SG fallback/cleanup error')
    e.log(p/'server.log');owner=e.json(p/'owned-process.json');info=e.json(p/'server-info.json')
    require(info['status']==200,'No actual server info')
    spec=importlib.util.spec_from_file_location('actual_audit_verifier',WORKSPACE/'experiments/c4v8_rollout_bridge/verify_backend_audit.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    audit=verifier.verify(p/'backend-audit',owner['pid'],8,server_info=info['body'])
    require(audit['policy_sha256']==digest(e.policy) and audit['implementation_sha256']==digest(e.profile),'Actual8rank policy/implementation mismatch')
    for source,value in r['runtime_source_sha256'].items():
        e.text(source);require(e.files[str(Path(source).resolve())]==value,'SG runtime/probe source changed: '+source)
    for source,value in r['probe_source_sha256'].items():
        relative=Path(source).relative_to(WORKSPACE)
        e.recorded_probe_source(source,value,p/'sources'/relative)
    require(info['body'].get('model_path')==str(WORKSPACE/'models/Qwen3-30B-A3B'),'Actual HTTP model identity mismatch')
    for path in (p/'backend-audit').glob('*.json'):
        row=e.json(path)
        if row.get('event')=='bootstrap_registered' and row.get('role')=='http_parent':
            require(row.get('fp32_simulator_mode',{}).get('actual_allow_hf32') is False,'Parent HF32 readback missing')
    concurrency=r.get('concurrency',1)
    require(concurrency in (1,4) and r['server_args']['max_running_requests']==concurrency
        and info['body'].get('max_running_requests')==concurrency,'Actual SG concurrency mismatch')
    require(len(r.get('measurements',[]))==6*concurrency,'Need2 genuine prompts x3 waves x recorded concurrency')
    require(len({row['name'] for row in r['measurements']})==6*concurrency,'Duplicate SG request records')
    if concurrency>1:
        require(len(r.get('waves',[]))==6 and all(wave.get('concurrency')==concurrency and
            len(wave.get('request_names',[]))==concurrency for wave in r['waves']),'Missing concurrent wave evidence')
    for row in r['measurements']:
        require(row.get('logprob_finite') is True and row.get('input_token_logprob_alignment') is True,'Missing finite/aligned SG logprobs')
        request=e.json(p/(row['name']+'-request.json'));response=e.json(p/(row['name']+'-response.json'))
        require(hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==row['request_sha256'],'SG request identity mismatch')
        meta=response['meta_info'];count=request['sampling_params']['max_new_tokens']
        require(count in (16,32) and meta['completion_tokens']==count and meta['prompt_tokens']==len(request['input_ids']),
            'Actual SG token count mismatch')
        require(len(meta['output_token_logprobs'])==count and all(isinstance(x[0],(int,float)) and math.isfinite(x[0]) for x in meta['output_token_logprobs']),
            'SG output logprob not finite')
        lp=meta['input_token_logprobs'];require(len(lp)==len(request['input_ids']),'SG teacher forcing length mismatch')
        require(all(v[1]==request['input_ids'][i] and (v[0] is not None or i==0) for i,v in enumerate(lp)),
            'SG teacher forcing token alignment mismatch')
    return dict(passed=True,scope='Actual original30B TP8 C4V8 SG prefill/decode and finite aligned logprobs; no trainer-vs-SG full-model comparison',
        measurements=r['measurements'],whole_model_logprob_agreement='unknown')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=WORKSPACE/'experiments/c4v8-integration-probes/evidence'
    p.add_argument('--postcheck-root',type=Path,default=base/'owned-finalcore-postcheck-v1')
    p.add_argument('--adapter-root',type=Path,default=base/'owned-attention-adapters-staged-q1024-v1')
    p.add_argument('--tiny-root',type=Path,default=base/'owned-tiny-staged-q1024-v1')
    p.add_argument('--performance-root',type=Path,default=base/'owned-final-performance-q1024-v1')
    p.add_argument('--sg-pilot-root',type=Path,default=base/'owned-30b-sg-staged-q1024-v1')
    p.add_argument('--eval-artifact',type=Path,required=True)
    p.add_argument('--review-output',type=Path,required=True)
    p.add_argument('--write-gate',type=Path,help='Explicit root decision; only created after all actual checks pass')
    args=p.parse_args();require(not args.review_output.exists(),'Preserve prior review output')
    if args.write_gate:require(not args.write_gate.exists(),'Preserve prior gate output')
    policy=json.loads((PREPARED/'precision-policy.json').read_text());profile=implementation_profile(WORKSPACE,'fast_attention')
    e=Evidence(profile,policy);errors=[]
    jobs=[('components',lambda:components(e)),('postcheck',lambda:postcheck(e,args.postcheck_root)),
        ('adapter',lambda:adapter(e,args.adapter_root)),('tiny',lambda:tiny(e,args.tiny_root)),
        ('performance',lambda:performance(e,args.performance_root)),('evaluation',lambda:evaluator(e,args.eval_artifact)),
        ('sg_pilot',lambda:sg_pilot(e,args.sg_pilot_root))]
    for name,fn in jobs:
        try:e.results[name]=fn()
        except (ValueError,KeyError,TypeError,OSError) as error:errors.append(dict(check=name,type=type(error).__name__,message=str(error)))
    inventory,missing=source_inventory(WORKSPACE,PREPARED/'qwen30b_dapo_c4v8.py','fast_attention')
    if missing:errors.append(dict(check='runtime_inventory',missing=missing))
    for name,value in e.files.items():
        if hash_file(name)!=value:errors.append(dict(check='evidence_drift',path=name))
    if implementation_profile(WORKSPACE,'fast_attention')!=profile:errors.append(dict(check='implementation_drift'))
    gate=None;consumer_acceptance=None
    if not errors:
        checks={key:True for key in GATE_CHECKS}  # All actual evidence validators have succeeded.
        gate=dict(status='passed',gate_contract_version=GATE_CONTRACT_VERSION,policy_sha256=digest(policy),
            implementation_sha256=digest(profile),source_sha256=inventory,checks=checks,
            evidence_sha256=e.files,evidence_scopes=e.results,whole_model_logprob_agreement='unknown',
            real_single_optimizer_update_scope='Tiny-model Adam only; first30B update remains runtime observation',
            assembler_sha256=hash_file(__file__),checked_utc=datetime.now(timezone.utc).isoformat())
        try:consumer_acceptance=verify_gate_consumer(gate)
        except (ValueError,KeyError,TypeError,OSError,RuntimeError) as error:
            errors.append(dict(check='actual_prepared_gate_consumer',type=type(error).__name__,message=str(error)))
    passed=not errors
    review=dict(status='verified_for_root_review' if passed else 'rejected_or_pending',passed=passed,launch_performed=False,
        gate_written=False,checked_utc=datetime.now(timezone.utc).isoformat(),errors=errors,evidence_scopes=e.results,
        evidence_sha256=e.files,implementation_sha256=digest(profile),policy_sha256=digest(policy),
        source_sha256=inventory,source_inventory_sha256=digest(inventory),whole_model_logprob_agreement='unknown',
        actual_grpo_updates=0,assembler_sha256=hash_file(__file__),consumer_acceptance=consumer_acceptance)
    if passed and args.write_gate:
        gate['consumer_acceptance']=consumer_acceptance
        with args.write_gate.open('x') as f:json.dump(gate,f,indent=2,allow_nan=False);f.write('\n')
        review.update(gate_written=True,gate_path=str(args.write_gate.resolve()),gate_sha256=hash_file(args.write_gate))
    with args.review_output.open('x') as f:json.dump(review,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=passed,errors=errors,gate_written=review['gate_written'])));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
