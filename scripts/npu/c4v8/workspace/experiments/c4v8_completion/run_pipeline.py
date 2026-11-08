#!/usr/bin/env python3
"""One prepared run -> acceptance -> same-policy 21 points -> plots.

--prepare is read-only with respect to training/models/devices. --execute is a
separate explicit operation. No scheduler, retries, training restart or notices.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import time

ROOT=Path('/root/hby')
HERE=Path(__file__).resolve().parent
BRIDGE=ROOT/'experiments/c4v8_rl_training_bridge'
PYTHON=ROOT/'envs/slime/bin/python'
PLOT_PYTHON=HERE/'plot-env/bin/python'
BINDING_KEYS=('run_id','training_driver_pid','initial_model','hf_model_destination',
    'parent_completed_optimizer_steps','optimizer_step_hf_checkpoint_directory',
    'optimizer_step_hf_checkpoint_interval','optimizer_step_hf_expected_global_steps',
    'validation_trajectory_directory','validation_checkpoint_model_paths','validation_expected_global_steps',
    'validation_protocol','validation_protocol_sha256','validation_protocol_helper',
    'validation_protocol_helper_sha256','attention_simulation','argv')


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as file:
        while block:=file.read(8*1024**2):result.update(block)
    return result.hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def binding(manifest):return {key:manifest[key] for key in BINDING_KEYS}
def proc(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),state=s[0],pgid=int(s[2]),sid=int(s[3]),startticks=int(s[19]),
            argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v])
    except (FileNotFoundError,ProcessLookupError):return None
def same_process(current,expected):
    return current is not None and all(current[k]==expected[k] for k in ('pid','pgid','sid','startticks','argv'))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module
def verify_sources(sources):
    for path,value in sources.items():require(sha(path)==value,'Source changed: '+path)


def pidfd_open(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
    # This pinned Python omits os.pidfd_open despite its Linux5.10 kernel
    # supporting it. syscall434 is pidfd_open on these two explicit ABIs.
    require(sys.platform=='linux' and os.uname().machine in ('aarch64','x86_64'),
            'No supported Linux pidfd ABI; no polling fallback')
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    fd=libc.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
    if fd<0:
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    return fd


def prepare(canonical_path,output):
    canonical_path=canonical_path.resolve();output=output.resolve()
    require(output.is_relative_to(HERE),'Preparation output must be inside this independent completion directory')
    canonical=read(canonical_path);controller=Path(canonical['controller_dir']).resolve()
    require(controller.is_relative_to(BRIDGE),'Unexpected training controller directory')
    status=read(controller/'controller.json');run=Path(status['training_run']).resolve()
    require(run.is_relative_to(ROOT/'runs/qwen3-30b-a3b-dapo-math-17k'),'Unexpected training run')
    manifest=read(run/'manifest.json');current=proc(canonical['pid'])
    expected=dict(pid=canonical['pid'],pgid=canonical['pgid'],sid=canonical['sid'],
                  startticks=canonical['startticks'],argv=canonical['command'])
    require(same_process(current,expected) and current['state']!='Z','Fixed controller identity is no longer live')
    require(manifest['initial_model']==str(ROOT/'models/Qwen3-30B-A3B') and
            manifest['parent_completed_optimizer_steps']==0,'Need this original-model trajectory')
    require(sha(canonical['gate'])==canonical['gate_sha256'],'Accepted gate changed')
    gate=read(canonical['gate'])
    require(gate['status']=='passed' and gate['source_sha256']==manifest['attention_simulation']['source_sha256'],
            'Training manifest and accepted gate source inventory differ')
    sources=dict(gate['source_sha256'])
    for path in [ROOT/'setup/compare-initial-qwen30b.py',ROOT/'setup/scan-hf-finite-sha256.py',
                 HERE/'run_pipeline.py',HERE/'plot_curve.py',HERE/'plot-requirements.txt']:
        sources[str(path)]=sha(path)
    verify_sources(sources)
    descriptor=pidfd_open(current['pid']);os.close(descriptor)  # Actual read-only kernel capability check.
    output.mkdir(parents=True,exist_ok=False)
    snapshots=output/'source-snapshot';snapshots.mkdir()
    snapshot_index={}
    for path,value in sources.items():
        destination=snapshots/Path(path).relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(Path(path).read_bytes())
        require(sha(destination)==value,'Source changed during snapshot: '+path)
        snapshot_index[path]=str(destination)
    write(output/'manifest-at-preparation.json',manifest)
    write(output/'canonical-at-preparation.json',canonical)
    plan=dict(status='prepared',prepared_utc=utc(),execution_started=False,
        controller=expected,controller_dir=str(controller),canonical_path=str(canonical_path),canonical_sha256=sha(canonical_path),
        training_run=str(run),manifest_binding=binding(manifest),manifest_binding_sha256=digest(binding(manifest)),
        gate=canonical['gate'],gate_sha256=canonical['gate_sha256'],source_sha256=sources,source_snapshots=snapshot_index,
        trajectory_root=str(Path(manifest['validation_trajectory_directory'])/manifest['run_id']),
        python=str(PYTHON),plot_python=str(PLOT_PYTHON),
        evaluation_timeouts=dict(startup=1800,request=10800,point=43200,total=604800),
        stages=['pidfd_wait_fixed_controller','accept_same_run100','fresh_release_gate','sampled_changes',
                'full_finite_sha_published','full_finite_sha_step100','full_finite_sha_base',
                'same_policy21_point_evaluation','verify21_actual_points','plots_and_final_manifest'],
        no_training_restart=True,no_periodic_automation=True,whole_model_logprob_agreement='unknown')
    plan['plan_identity_sha256']=digest(plan)
    write(output/'plan.json',plan)
    print(json.dumps(dict(status='prepared',plan=str(output/'plan.json'),execution_started=False)))
    return plan


def wait_controller(expected):
    current=proc(expected['pid'])
    if current is None or current['state']=='Z':return 'already_exited'
    require(same_process(current,expected),'Controller PID reused/identity changed; refusing to follow another run')
    try:fd=pidfd_open(expected['pid'])
    except ProcessLookupError:return 'exited_before_pidfd_open'
    try:
        current=proc(expected['pid'])
        if current is not None and current['state']!='Z':
            require(same_process(current,expected),'Controller identity changed while opening pidfd')
        poll=select.poll();poll.register(fd,select.POLLIN)
        poll.poll()  # Blocking kernel wait; no timer, retry loop, or CPU polling.
        return 'pidfd_exit_observed'
    finally:os.close(fd)


def verify_scan_report(path,model):
    """Only reuse a full finite report for this same directory and actual bytes."""
    path,model=Path(path),Path(model).resolve();r=read(path)
    require(r.get('passed') is True and r.get('complete_scan') is True and r.get('nonfinite_count')==0,
            'Incomplete/failed/nonfinite weight scan')
    require(Path(r['hf_dir']).resolve()==model,'Finite report belongs to a different model directory')
    index=model/'model.safetensors.index.json'
    require(r.get('index_sha256')==sha(index),'Finite report index SHA mismatch')
    shard_names=set(read(index)['weight_map'].values())
    require(len(r['shards'])==len(shard_names) and {s['file'] for s in r['shards']}==shard_names,
            'Finite report does not cover exactly the indexed shards')
    for row in r['shards']:
        require(Path(row['file']).name==row['file'],'Unexpected shard path')
        p=model/row['file'];before=p.stat();actual=sha(p);after=p.stat()
        require((before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Model changed during SHA verification')
        require(after.st_size==row['file_bytes'] and actual==row['sha256'],'Actual shard bytes differ from finite report')
    return dict(passed=True,report=str(path),report_sha256=sha(path),hf_dir=str(model),
                actual_shard_bytes_rehashed=True,shard_sha256={s['file']:s['sha256'] for s in r['shards']})


def tensor_content_hashes(model):
    """All indexed tensor bytes, independent of shard layout/JSON metadata.

    Called only after full finite/index scanner acceptance in the pipeline.
    No dtype conversion, sample selection, or accelerator/framework import.
    """
    model=Path(model);weight_map=read(model/'model.safetensors.index.json')['weight_map'];tensors={}
    for filename in sorted(set(weight_map.values())):
        require(Path(filename).name==filename,'Unexpected tensor shard path')
        path=model/filename;before=path.stat()
        with path.open('rb') as file:
            prefix=file.read(8);require(len(prefix)==8,'Truncated safetensors header size')
            size=struct.unpack('<Q',prefix)[0];require(2<=size<=64*1024**2,'Invalid/oversized safetensors header')
            header=json.loads(file.read(size));base=8+size
            for name in sorted((n for n,s in weight_map.items() if s==filename),key=lambda n:header[n]['data_offsets'][0]):
                entry=header[name];start,end=entry['data_offsets']
                require(type(start) is int and type(end) is int and 0<=start<=end and base+end<=before.st_size,
                        'Invalid indexed tensor byte range')
                file.seek(base+start);remaining=end-start;value=hashlib.sha256()
                while remaining:
                    block=file.read(min(remaining,8*1024**2));require(bool(block),'Truncated tensor data')
                    value.update(block);remaining-=len(block)
                tensors[name]=dict(dtype=entry['dtype'],shape=entry['shape'],bytes=end-start,sha256=value.hexdigest())
        after=path.stat()
        require((before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Tensor export changed while comparing')
    return tensors


def compare_exports(first,second,first_scan,second_scan):
    one=read(Path(first)/'model.safetensors.index.json')['weight_map']
    two=read(Path(second)/'model.safetensors.index.json')['weight_map']
    require(set(one)==set(two),'Published and HF100 tensor name sets differ')
    if one==two and first_scan['shard_sha256']==second_scan['shard_sha256']:
        return dict(passed=True,method='All actual shard bytes and complete tensor map are identical',tensor_count=len(one))
    a,b=tensor_content_hashes(first),tensor_content_hashes(second)
    mismatches=[name for name in sorted(set(a)|set(b)) if a.get(name)!=b.get(name)]
    require(not mismatches,'Published/HF100 full tensor bytes or dtype/shape differ: '+str(mismatches[:5]))
    return dict(passed=True,method='Every indexed tensor dtype/shape/all raw bytes match despite serialization differences',
                tensor_count=len(a),tensor_content_sha256=digest(a),total_tensor_bytes=sum(v['bytes'] for v in a.values()))


def verify_truncation_aggregate(trajectory,step):
    directory=Path(trajectory)/f'step_{step:06d}'
    paths=sorted((directory/'questions').glob('*.json'))
    require([p.name for p in paths]==[f'{i:02d}.json' for i in range(30)],'Need30 raw truncation records')
    flags=[read(path)['truncated'] for path in paths]
    require(all(type(value) is bool for value in flags),'Raw truncation flags must be bool')
    reported=read(directory/'summary.json')['truncated']
    require(type(reported) is int and reported==sum(flags),'Summary truncation count differs from30 actual raw answers')
    return dict(passed=True,truncated=reported,questions=30,truncation_rate=reported/30)


class Stages:
    def __init__(self,output,budget):self.output=output;self.budget=budget
    def run(self,name,command,*,timeout=21600,is_evaluation=False):
        directory=self.output/name;directory.mkdir()
        record=dict(status='starting',command=list(map(str,command)),started_utc=utc(),timeout_seconds=timeout)
        write(directory/'stage.json',record);child=owner=None;started=time.monotonic()
        try:
            with (directory/'stdout.log').open('x') as log:
                with self.budget.defer_termination():
                    child=subprocess.Popen(record['command'],stdout=log,stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,start_new_session=True,
                        env=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                    owner=proc(child.pid)
                    require(owner is not None and owner['pid']==owner['pgid']==owner['sid'],'Child session not owned')
                    record.update(status='running',owner=owner);write(directory/'stage.json',record)
                code=child.wait(timeout=timeout)
                record['exit_code']=code
                require(code==0,'Stage failed: '+name+' exit '+str(code))
            record['status']='completed'
        except BaseException as error:
            record.update(status='failed',error_type=type(error).__name__,error=str(error))
            raise
        finally:
            with self.budget.owned_cleanup():
                if child is not None and child.poll() is None:
                    current=proc(child.pid)
                    if same_process(current,owner):
                        # Eval owns a separate SG session and has a tested TERM
                        # handler. Give it enough time for client + SG cleanup.
                        child.terminate()
                        try:child.wait(timeout=120 if is_evaluation else 10)
                        except subprocess.TimeoutExpired:
                            if is_evaluation:
                                record['cleanup_error']='Evaluator did not exit after TERM; preserve its owned-server evidence for operator cleanup'
                            elif same_process(proc(child.pid),owner):
                                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
                    else:record['cleanup_error']='Owned child identity changed; no unrelated PID/group signaled'
                if child is not None:record['exit_code']=child.poll()
                record.update(finished_utc=utc(),elapsed_seconds=time.monotonic()-started)
                write(directory/'stage.json',record)
                (directory/'exit').write_text(str(record.get('exit_code'))+'\n')
        return record


def execute(plan_path):
    plan_path=plan_path.resolve();plan=read(plan_path)
    require(plan_path.parent.is_relative_to(HERE),'Plan must belong to this completion workspace')
    require(plan['status']=='prepared' and plan['execution_started'] is False,'Unexpected plan state')
    identity=plan.pop('plan_identity_sha256');require(digest(plan)==identity,'Plan identity changed')
    plan['plan_identity_sha256']=identity
    verify_sources(plan['source_sha256'])
    require(sha(plan['canonical_path'])==plan['canonical_sha256'],'Canonical pointer changed to another task')
    require(sha(plan['gate'])==plan['gate_sha256'],'Launch gate changed')
    output=plan_path.parent/'execution';output.mkdir(exist_ok=False)  # Atomic single invocation claim.
    summary=dict(status='waiting',complete=False,plan=str(plan_path),plan_sha256=sha(plan_path),
        started_utc=utc(),controller=plan['controller'],training_run=plan['training_run'],
        training_updates_accepted=0,completed_validation_points=0,stages=[])
    write(output/'completion.json',summary)
    sys.path.insert(0,str(BRIDGE))
    from eval_deadlines import EvaluationBudget
    with EvaluationBudget(None) as budget:
        stages=Stages(output,budget)
        try:
            summary['controller_exit_observation']=wait_controller(plan['controller'])
            verify_sources(plan['source_sha256'])
            require(sha(plan['canonical_path'])==plan['canonical_sha256'],'Canonical pointer changed while waiting')
            controller=read(Path(plan['controller_dir'])/'controller.json')
            require(controller.get('status')=='training100_finished_pending_final_acceptance' and
                    controller.get('child_exit_code')==0 and controller.get('npu_after',{}).get('all_eight_idle') is True,
                    'Training/controller failed or incomplete; pipeline will not restart it')
            run=Path(plan['training_run']);manifest=read(run/'manifest.json')
            require(binding(manifest)==plan['manifest_binding'],'Training run identity/configuration changed')
            from pilot_evidence import read_progress
            progress=read_progress(run,minimum_file_age=0)
            require(manifest.get('status')=='completed' and manifest.get('optimizer_steps')==100 and
                    manifest.get('new_optimizer_steps')==100 and progress['gradient_norm_records']==100 and
                    progress['successful_updates_proven_by_complete_hf']==100 and progress['final_hundred_checkpoint_complete'],
                    'Need exactly100 real updates with complete HF100 and100 positive finite gradients')
            from eval_support import configure_from_manifest,fresh_resource_gate
            configure_from_manifest(run/'manifest.json')
            import c4v8_eval_inherited as inherited
            _,models=inherited.plan_models(run/'manifest.json')
            summary.update(status='accepting_weights',training_updates_accepted=100,training_acceptance=progress,
                final_training_manifest_sha256=sha(run/'manifest.json'),controller_result_sha256=sha(Path(plan['controller_dir'])/'controller.json'))
            write(output/'completion.json',summary)
            fresh_resource_gate(output/'resources-before-scans.json',manifest,inherited.proc)
            final=Path(manifest['hf_model_destination']);base=Path(manifest['initial_model'])
            compare=output/'published-vs-base-sampled-changes.json'
            summary['stages'].append(stages.run('sampled-changes',[plan['python'],ROOT/'setup/compare-initial-qwen30b.py',
                base,final,'--expect-sampled-changes','--output',compare]))
            comparison=read(compare);require(comparison.get('status')=='passed','Sampled numerical-change acceptance failed')
            scans={}
            for name,model,report in [('published',final,run/'final-hf-finite-sha256.json'),
                ('step100',models[100],run/'eval-step100-finite-sha256.json'),
                ('base',base,run/'base-hf-finite-sha256.json')]:
                reused=report.exists()
                if not reused:
                    summary['stages'].append(stages.run('scan-'+name,[plan['python'],ROOT/'setup/scan-hf-finite-sha256.py',
                        model,'--output',report,'--block-mib','8']))
                scans[name]=verify_scan_report(report,model);scans[name]['reused_existing_report']=reused
                summary['finite_scans']=scans;write(output/'completion.json',summary)
            summary['published_matches_checkpoint100']=compare_exports(final,models[100],scans['published'],scans['step100'])
            write(output/'published-vs-checkpoint100-full-content.json',summary['published_matches_checkpoint100'])
            fresh_resource_gate(output/'resources-before-evaluation.json',manifest,inherited.proc)
            summary['status']='evaluating21';write(output/'completion.json',summary)
            t=plan['evaluation_timeouts']
            command=[plan['python'],BRIDGE/'eval-c4v8-trajectory.py','--manifest',run/'manifest.json',
                '--base-sha-report',scans['base']['report'],'--final-sha-report',scans['step100']['report'],
                '--server-start-timeout',str(t['startup']),'--request-timeout',str(t['request']),
                '--client-timeout',str(t['point']),'--total-timeout-seconds',str(t['total'])]
            summary['stages'].append(stages.run('evaluate21',command,timeout=t['total']+180,is_evaluation=True))
            fresh_resource_gate(output/'resources-after-evaluation.json',manifest,inherited.proc)
            trajectory=Path(plan['trajectory_root']);client=inherited.client_module();rows=client.load_data(inherited.DATA)
            points=[]
            for step,model in models.items():
                point=inherited.validate_point(client,trajectory,model,step,rows)
                point['raw_truncation_aggregate']=verify_truncation_aggregate(trajectory,step)
                points.append(point)
            require(len(points)==21 and [p['optimizer_step'] for p in points]==list(range(0,101,5)),
                    'Need21 actual same-lineage points')
            curve=read(trajectory/'accuracy-curve.json')
            require(curve.get('complete') is True and len(curve['valid_points'])==21,'Incomplete curve')
            summary.update(status='plotting',completed_validation_points=21,points=points,
                curve_path=str(trajectory/'accuracy-curve.json'),curve_sha256=sha(trajectory/'accuracy-curve.json'))
            write(output/'completion.json',summary)
            summary['stages'].append(stages.run('plots',[plan['plot_python'],HERE/'plot_curve.py',
                '--curve',trajectory/'accuracy-curve.json','--output',output/'plots'],timeout=600))
            verify_sources(plan['source_sha256'])
            summary.update(status='completed',complete=True,model_path=str(final),
                plot_report=read(output/'plots/plot-report.json'),evaluation_accuracy=points[-1]['accuracy'],
                evaluation_truncation_rate=points[-1]['truncation_rate'],
                scope='BF16 backbone + C4V8 forward QDQ/STE;21 real nonthinking C4V8 evaluation points; no claim of low-bit backward')
        except BaseException as error:
            summary.update(status='failed',complete=False,error_type=type(error).__name__,error=str(error))
        finally:
            summary['finished_utc']=utc();write(output/'completion.json',summary)
            (output/'pipeline.exit').write_text('0\n' if summary['complete'] else '1\n')
    print(json.dumps({k:summary[k] for k in ('status','complete','training_updates_accepted','completed_validation_points')}))
    return 0 if summary['complete'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare',action='store_true');choice.add_argument('--execute',type=Path)
    parser.add_argument('--canonical',type=Path,default=ROOT/'logs/c4v8-dapo100-every5-controller.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.prepare:
        if args.output is None:parser.error('--prepare requires fresh --output')
        prepare(args.canonical,args.output);return 0
    return execute(args.execute)


if __name__=='__main__':raise SystemExit(main())
