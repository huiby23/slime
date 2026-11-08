"""Validate one server parent's rank-specific C4V8 evidence, never launch a job."""
import argparse
import hashlib
import json
from pathlib import Path


BACKEND='hby_c4v8_sim'
SELECTORS=('attention_backend','prefill_attention_backend','decode_attention_backend')


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify(audit_dir,parent_pid,tp_size,*,require_forward=True,allow_cpu_probe=False,
           server_info=None,rehash_sources=True):
    rows=[json.loads(path.read_text()) for path in Path(audit_dir).glob('*.json')]
    rows=[r for r in rows if r.get('pid')==parent_pid or r.get('ppid')==parent_pid]
    if not rows:raise ValueError('No audit belongs to the specified server parent')
    expected_kind='cpu_stubbed_sglang' if allow_cpu_probe else 'actual_process'
    if any(r.get('execution_validation_kind')!=expected_kind for r in rows):
        raise ValueError('Evidence kind mismatch; CPU stubbed evidence cannot pass production acceptance')
    profiles={r.get('runtime_sha256') for r in rows}
    policies={r.get('policy_sha256') for r in rows}
    if len(profiles)!=1 or len(policies)!=1 or None in profiles or None in policies:
        raise ValueError('Parent and workers did not use one source/policy identity')
    if any(_digest(row['policy'])!=row['policy_sha256'] for row in rows):
        raise ValueError('Recorded policy hash does not describe the recorded policy')
    parent=[r for r in rows if r.get('event')=='bootstrap_registered' and
            r.get('role')=='http_parent' and r.get('pid')==parent_pid]
    if len(parent)!=1:raise ValueError('Expected exactly one parent installation record')
    identity={'backend':BACKEND,'policy':parent[0]['policy'],'sources':parent[0]['source_sha256']}
    if 'implementation_sha256' in parent[0]:
        identity['implementation_sha256']=parent[0]['implementation_sha256']
        if _digest(parent[0]['implementation'])!=parent[0]['implementation_sha256']:
            raise ValueError('Implementation hash differs from its actual source/toolchain profile')
        if len({r.get('implementation_sha256') for r in rows})!=1:
            raise ValueError('Parent/workers used different numerical implementations')
    if _digest(identity)!=parent[0]['runtime_sha256']:
        raise ValueError('Runtime identity does not describe the parent source/profile snapshot')
    children=[r for r in rows if r.get('event')=='bootstrap_registered' and r.get('role')=='scheduler']
    if len(children)!=tp_size or {r['ranks']['tp_rank'] for r in children}!=set(range(tp_size)):
        raise ValueError('Missing/duplicate scheduler rank installation')
    child_pids={r['pid'] for r in children}
    if len(child_pids)!=tp_size:raise ValueError('Scheduler ranks were not distinct processes')
    summaries=[]
    for child in children:
        pid=child['pid']
        selected=[r for r in rows if r['pid']==pid]
        args=[r for r in selected if r['event']=='effective_server_args']
        created=[r for r in selected if r['event']=='backend_created']
        if len(args)!=1 or len(created)!=1:raise ValueError('Missing actual args or backend instance per rank')
        if any(args[0]['actual'].get(k)!=BACKEND or created[0]['actual_selectors'].get(k)!=BACKEND
               for k in SELECTORS):raise ValueError('A rank selected a different backend')
        expected_core=parent[0].get('core_module','device_attention')
        if created[0]['core_module']!=expected_core or created[0]['core_callable']!='attention':
            raise ValueError('A rank instantiated a different numerical core')
        if created[0].get('tp_rank')!=child['ranks']['tp_rank']:
            raise ValueError('Backend instance rank disagrees with spawn rank')
        seen={r['event'] for r in selected}
        if not allow_cpu_probe:
            if (child.get('fp32_simulator_mode',{}).get('applied') is not True
                    or child['fp32_simulator_mode'].get('actual_allow_hf32') is not False
                    or created[0].get('actual_allow_hf32') is not False):
                raise ValueError('Actual C4V8 simulator HF32-disable audit missing or incorrect')
            pools=[r for r in selected if r['event']=='hardware_kv_pool_created']
            if len(pools)!=1:raise ValueError('Missing actual NPU KV pool/allocator audit per rank')
            if (pools[0].get('pool_class')!='NPUMHATokenToKVPool'
                    or pools[0].get('allocator_class')!='NPUPagedTokenToKVPoolAllocator'
                    or pools[0].get('cache_dtype')!='torch.bfloat16'
                    or any(pools[0]['actual_selectors'].get(k)!=BACKEND for k in SELECTORS)):
                raise ValueError('Actual NPU KV hardware plumbing differs from the audited path')
        if require_forward and not {'first_extend','first_decode'} <= seen:
            raise ValueError('Each rank must execute both prefill and decode through the numerical core')
        for row in selected:
            if row['event'] in ('first_extend','first_decode') and row.get('cache_dtype')!='torch.bfloat16':
                raise ValueError('Actual KV storage was not BF16')
        summaries.append({'tp_rank':child['ranks']['tp_rank'],'pid':pid,
                          'backend_class':created[0]['backend_class'],
                          'prefill_observed':'first_extend' in seen,'decode_observed':'first_decode' in seen})
    if rehash_sources:
        for path,expected in parent[0]['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                raise ValueError('Source changed since parent startup: '+path)
    if server_info is not None:
        if any(server_info.get(k)!=BACKEND for k in SELECTORS):
            raise ValueError('HTTP server_info selectors differ from actual rank audits')
        if server_info.get('tp_size')!=tp_size:raise ValueError('HTTP TP size differs')
    return {'passed':True,'parent_pid':parent_pid,'tp_size':tp_size,
            'runtime_sha256':next(iter(profiles)),'policy_sha256':next(iter(policies)),
            'implementation_id':parent[0].get('implementation_id'),
            'implementation_sha256':parent[0].get('implementation_sha256'),
            'evidence_kind':expected_kind,'ranks':sorted(summaries,key=lambda r:r['tp_rank']),
            'http_server_info_verified':server_info is not None,
            'sources_rehashed':rehash_sources,'both_forward_modes_required':require_forward,
            'native_npu_pool_allocator_verified':not allow_cpu_probe,
            'native_kernel_bit_parity_verified':False,'training_or_rl_verified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--parent-pid',type=int,required=True)
    parser.add_argument('--tp-size',type=int,required=True)
    parser.add_argument('--server-info-json',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-cpu-probe',action='store_true')
    parser.add_argument('--initialization-only',action='store_true')
    args=parser.parse_args()
    info=json.loads(args.server_info_json.read_text())
    result=verify(args.audit_dir,args.parent_pid,args.tp_size,
                  require_forward=not args.initialization_only,
                  allow_cpu_probe=args.allow_cpu_probe,server_info=info)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
