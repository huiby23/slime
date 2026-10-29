"""Run gated full-size smoke/resume, then matched BF16 and W4A8 experiments.

Uses only observed idle GPUs. It never kills foreign processes. Every Docker
attempt uses frozen source and an isolated local Ray cluster. A failed gate
stops the queue with a visible error; it never promotes a failed smoke.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def replace_option(command,flag,value):
    command=list(command)
    if flag in command:
        command[command.index(flag)+1]=str(value)
    else:
        command += [flag,str(value)]
    return command


def eligible_gpus(status,processes,reuse=()):
    """Accept only fresh idle observations with no current compute owner."""
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(status['at'])).total_seconds()
    if not 0 <= age < 20:
        return []
    busy={line.split(',')[0].strip() for line in processes.splitlines() if ',' in line}
    return [g for g in status.get('gpus',[]) if g['idle'] and g['uuid'] not in busy
        and (g['qualified'] or g['uuid'] in reuse)]


def stage_arm(plan,mode,work,save,smoke=False,resume=False):
    arm=copy.deepcopy(plan['arms'][mode])
    original_work,original_save=arm['work'],arm['save']
    work,save=str(work),str(save)
    arm['environment']={key:value.replace(original_work,work).replace(original_save,save)
        for key,value in arm['environment'].items()}
    arm['environment']['W4A8_RUN_NAMESPACE']=mode+'-'+('smoke' if smoke else 'formal')
    arm['environment']['RAY_TMPDIR']='/data2/huibingyu/r4b/'+mode+('-s' if smoke else '-f')
    command=replace_option(arm['training_command'],'--save',save)
    command=replace_option(command,'--tensorboard-dir',str(Path(work)/'tensorboard'))
    if smoke:
        for flag,value in {'--num-rollout':2 if resume else 1,'--rollout-batch-size':32,'--n-samples-per-prompt':4,
            '--global-batch-size':128,'--seq-length':2048,'--rollout-max-response-len':1024,
            '--eval-max-response-len':1024,'--eval-interval':1}.items():
            command=replace_option(command,flag,value)
        index=command.index('--eval-prompt-data')
        command[index+2]=str(Path(work).parent/'validation16.jsonl')
        command += ['--skip-eval-before-train']
    if resume:
        command=replace_option(command,'--load',str(Path(save)/'last'))
        if smoke:
            # The smoke extends its planned duration (and may broaden its
            # batch). Keep the new constant schedule while restoring consumed
            # samples, model weights and every Adam state from the checkpoint.
            command += ['--override-opt-param-scheduler']
        if '--start-rollout-id' in command:
            index=command.index('--start-rollout-id')
            del command[index:index+2]
    arm.update(training_command=command,work=work,save=save)
    return arm


def validate_run(arm,gpu_count,mode,completed,require_nonzero=True):
    work,save=Path(arm['work']),Path(arm['save'])
    policy=json.loads((save/'checkpoint-policy.json').read_text())
    if policy['last']['completed_updates']!=completed:
        raise RuntimeError('Full-size run did not complete the expected updates')
    last=(save/'last').resolve(strict=True)
    step=completed-1
    if not (last/f'iter_{step:07d}/.metadata').is_file():
        raise RuntimeError('Missing complete native last checkpoint')
    best=(save/'best').resolve(strict=True)
    exported=json.loads((best/'export-validation.json').read_text())
    if not exported['passed'] or exported['optimizer_exported']:
        raise RuntimeError('Best must contain only validated trained model weights')
    audits=[]
    nonzero=[]
    for rank in range(gpu_count):
        records=[json.loads(line) for line in (work/f'training-audit/optimizer-rank-{rank}.jsonl').read_text().splitlines()]
        current=[r for r in records if r['rollout_id']==step]
        if not current or not current[-1]['optimizer_success'] or not current[-1]['sampled_masters_finite']:
            raise RuntimeError(f'Missing successful optimizer update on rank {rank}')
        nonzero.append(any(r['grad_norm'] and r['sampled_master_max_update']>0 for r in records))
        if require_nonzero and not nonzero[-1]:
            raise RuntimeError(f'No nonzero trained update on rank {rank}; inspect reward diversity')
        if mode=='w4a8' and current[-1]['quantized_modules']!=144:
            raise RuntimeError('Incomplete full-size linear W4A8 coverage')
        audits.append(current[-1])
    if mode=='w4a8':
        paths=list((work/'rollout-audit').rglob('rollout-rank-*.json'))
        for path in paths:
            report=json.loads(path.read_text())
            if len(report['projections'])!=144:
                raise RuntimeError('Incomplete full-size rollout W4A8 coverage')
            if report['counters'].get('weight_cache_misses',0)<288:
                raise RuntimeError('No rollout cache invalidation after trained weight synchronization')
        engine_count=int(arm['training_command'][arm['training_command'].index('--rollout-num-gpus')+1])
        if len(paths)<engine_count:
            raise RuntimeError('Missing quantized rollout audit for an engine')
    return dict(passed=True,completed_updates=completed,training_mode=mode,
        last=str(last),best=str(best),best_export=exported,rank_audits=audits,
        nonzero_gradient_seen=all(nonzero),nonzero_gradient_required=require_nonzero)


def retire_smoke(arm,report,image):
    """Remove only validated owned smoke state, including root-owned files."""
    if not report.get('passed') or not report.get('nonzero_gradient_seen'):
        raise RuntimeError('Refusing to retire unvalidated smoke state')
    save=Path(arm['save'])
    physical=save/'.snapshots'
    last=Path(report['last'])
    policy=json.loads((save/'checkpoint-policy.json').read_text())
    owner=json.loads((physical/'.w4a8-owner.json').read_text())
    if last.parent!=physical.resolve() or not last.name.startswith('step_') or owner!={'logical_save_root':str(save.resolve())}:
        raise RuntimeError('Refusing to retire a checkpoint outside owned smoke storage')
    best=(save/'best').resolve(strict=True)
    if best==last or best!=Path(report['best']) or not (best/'model.safetensors').is_file():
        raise RuntimeError('A separate validated weights-only best must survive smoke retirement')
    if policy.get('last') is None and not last.exists() and policy.get('retired_smoke_last',{}).get('snapshot')==str(last):
        return
    if policy.get('last',{}).get('snapshot')!=str(last):
        raise RuntimeError('Smoke retirement no longer matches the current last')
    pointer=save/'last'
    if pointer.is_symlink() and pointer.resolve(strict=True)!=last:
        raise RuntimeError('Smoke last pointer does not match the validated report')
    metadata=last/f"iter_{report['completed_updates']-1:07d}/.metadata"
    write_json(Path(arm['work'])/'retired-smoke-last.json',dict(
        snapshot=str(last),metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        reason='Full-size resume validated; best weights and reports retained'))
    # Training containers create root-owned directories. The CPU-only helper
    # sees just this owned snapshot root and removes only the validated step.
    subprocess.run(['docker','run','--rm','--network','none','--env','NVIDIA_VISIBLE_DEVICES=void',
        '--mount',f'type=bind,src={physical},dst=/snapshot','--entrypoint','python3',image,
        '-c','import pathlib,shutil,sys; p=pathlib.Path("/snapshot")/sys.argv[1]; '
        'assert p.parent==pathlib.Path("/snapshot") and p.name.startswith("step_") and not p.is_symlink(); '
        'shutil.rmtree(p)',last.name],check=True)
    if pointer.is_symlink():
        pointer.unlink()
    policy['retired_smoke_last']=policy.pop('last')
    policy['last']=None
    write_json(save/'checkpoint-policy.json',policy)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--frozen-source',type=Path,required=True)
    parser.add_argument('--all-idle-gpus',action='store_true',
        help='Choose every qualified idle GPU before the first smoke; retain that topology for both arms')
    parser.add_argument('--resume-bf16-smoke',action='store_true',
        help='Reuse a completed first BF16 smoke, then require a nonzero-gradient resumed update')
    parser.add_argument('--resume-after-bf16-validation',action='store_true',
        help='Reuse passed BF16 smoke/resume reports, finish retirement, then continue with W4A8')
    parser.add_argument('--resume-after-validation',action='store_true',
        help='Reuse passed full-size smoke/resume reports for both arms and launch formal runs')
    args=parser.parse_args()
    work=args.work.resolve()
    lock=(work/'experiment.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((work/'training-plan.json').read_text())
    state=dict(phase='preparing',formal_training_started=False,completed_stages=[],model=plan['model'])
    image=subprocess.check_output(['docker','inspect','huibingyu-slime-v040','--format','{{.Image}}'],text=True).strip()
    source=args.frozen_source.resolve()
    reuse=set()
    if args.resume_bf16_smoke or args.resume_after_bf16_validation or args.resume_after_validation:
        previous=json.loads((work/'experiment-status.json').read_text())
        owned=json.loads(subprocess.check_output(['docker','inspect',previous['container']],text=True))[0]
        if owned['Config']['Labels'].get('lab.w4a8')!='qwen3-4b' or owned['State']['Running']:
            raise RuntimeError('Only an exited owned smoke with a validated checkpoint may be reused')
        reuse={g['uuid'] for g in previous['selected_gpus']}
        state.update(allocation_frozen=True,allocation=plan['topology'],physical_gpus=plan['physical_gpus'])

    def publish():
        state['at']=datetime.now(timezone.utc).isoformat()
        write_json(work/'experiment-status.json',state)

    def acquire():
        while True:
            status=json.loads((work/'gpus/idle-status.json').read_text())
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader'],text=True,timeout=10)
            eligible=eligible_gpus(status,processes,reuse)
            if args.all_idle_gpus and not state.get('allocation_frozen') and eligible:
                from prepare_qwen3_4b import prepare
                previous=list(plan['physical_gpus'])
                updated=prepare(Path(plan['model_path']),work,work/'checkpoints',
                    [g['index'] for g in eligible])
                write_json(work/('training-plan-before-allocation-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.json'),plan)
                plan.clear()
                plan.update(updated)
                write_json(work/'training-plan.json',plan)
                state.update(allocation_frozen=True,previous_planned_gpus=previous,
                    allocation=plan['topology'],physical_gpus=plan['physical_gpus'])
                publish()
            required=len(plan['physical_gpus'])
            if len(eligible)>=required and (not args.all_idle_gpus or state.get('allocation_frozen')):
                return eligible[:required]
            state.update(phase='waiting_for_idle_gpus',gpus=status.get('gpus',[]))
            publish()
            time.sleep(10)

    def run(stage,mode,smoke=False,resume=False):
        nonlocal reuse
        gpus=acquire()
        destination=work/'smoke'/mode if smoke else Path(plan['arms'][mode]['work'])
        save=work/'smoke'/'checkpoints'/mode if smoke else Path(plan['arms'][mode]['save'])
        destination.mkdir(parents=True,exist_ok=True)
        save.mkdir(parents=True,exist_ok=True)
        arm=stage_arm(plan,mode,destination,save,smoke,resume)
        # Each SGLang engine uses TP=1 and owns its own audit file.
        arm['environment']['W4A8_ROLLOUT_AUDIT_DIR']=str(destination/'rollout-audit')
        # Framework caches are created inside the container. CPU preflight can
        # create root-owned 0700 cache parents that the host cannot traverse.
        Path('/data2/huibingyu/r4b').mkdir(parents=True,exist_ok=True)
        name='huibingyu-4b-'+stage+'-'+datetime.now().strftime('%m%d-%H%M%S')
        command=['docker','run','-d','--name',name,'--label','lab.w4a8=qwen3-4b',
            '--gpus','"device='+','.join(g['uuid'] for g in gpus)+'"','--shm-size','16g',
            '--env','CUDA_VISIBLE_DEVICES='+','.join(str(i) for i in range(len(gpus)))]
        for key,value in arm['environment'].items():
            command += ['--env',key+'='+value]
        for src,dst in [(source/'quant','/quant'),(source/'slime','/workspace/slime'),
            (Path('/data2/huibingyu/slime-v040'),'/experiment'),(work,str(work)),
            (Path(plan['model_path']),plan['model_path']),
            (Path('/data2/huibingyu/r4b'),'/data2/huibingyu/r4b')]:
            readonly=',readonly' if dst in ('/quant','/workspace/slime','/experiment',plan['model_path']) else ''
            command += ['--mount',f'type=bind,src={src},dst={dst}{readonly}']
        command += ['--workdir','/workspace/slime','--entrypoint','python3',image]+arm['training_command'][1:]
        manifest=dict(stage=stage,mode=mode,gpus=gpus,command=command,arm=arm,image=image,
            frozen_source=str(source),at=datetime.now(timezone.utc).isoformat())
        manifest_path=work/(name+'.json')
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        write_json(manifest_path,manifest)
        cid=subprocess.check_output(command,text=True).strip()
        state.update(phase=stage,container=name,container_id=cid,selected_gpus=gpus,
            work=str(destination),save=str(save))
        state.pop('checkpoint_policy',None)
        if not smoke:
            state['formal_launched']=True
        publish()
        with (work/(name+'.log')).open('w') as log:
            reader=subprocess.Popen(['docker','logs','-f',name],stdout=log,stderr=subprocess.STDOUT)
            while subprocess.check_output(['docker','inspect',name,'--format','{{.State.Running}}'],text=True).strip()=='true':
                policy=save/'checkpoint-policy.json'
                if policy.is_file():
                    state['checkpoint_policy']=json.loads(policy.read_text())
                    if not smoke and state['checkpoint_policy'].get('last',{}).get('completed_updates',0)>0:
                        state['formal_training_started']=True
                state['observed_gpus']=json.loads((work/'gpus/idle-status.json').read_text()).get('gpus',[])
                publish()
                time.sleep(10)
            reader.wait(timeout=30)
        code=int(subprocess.check_output(['docker','inspect',name,'--format','{{.State.ExitCode}}'],text=True))
        if code:
            raise RuntimeError(f'{stage} failed with exit {code}; inspect {name}.log')
        expected=(2 if resume else 1) if smoke else int(arm['training_command'][arm['training_command'].index('--num-rollout')+1])
        report=validate_run(arm,plan['topology']['train_dp'],mode,expected,
            require_nonzero=not (smoke and not resume))
        write_json(destination/(stage+'-validation.json'),report)
        state['completed_stages'].append(dict(stage=stage,report=str(destination/(stage+'-validation.json'))))
        reuse={g['uuid'] for g in gpus}
        return arm,report

    try:
        for path in [work/'preflight-bf16.json',work/'preflight-w4a8.json']:
            if not json.loads(path.read_text())['argument_parse_passed']:
                raise RuntimeError('CPU preflight failed')
        for path in [work/'cuda-dense-dp4-a3/validation.json',work/'cuda-dense-rollout-a1/validation.json']:
            if not json.loads(path.read_text())['passed']:
                raise RuntimeError('Small real-backend validation failed')
        while not json.loads((work/'model-download.json').read_text()).get('completed'):
            downloaded=json.loads((work/'model-download.json').read_text())
            if downloaded.get('error'):
                raise RuntimeError(downloaded['error'])
            state.update(phase='downloading_initial_weights')
            publish()
            time.sleep(10)
        if shutil.disk_usage(work).free<plan['estimates']['two_arm_peak_including_shared_initial_bytes']*1.15:
            raise RuntimeError('Insufficient free NVMe storage for both experiments and atomic saves')
        smoke=work/'smoke'
        smoke.mkdir(exist_ok=True)
        dataset=Path('/data2/huibingyu/slime-v040/datasets/gsm8k-seed42-val512/validation.jsonl')
        (smoke/'validation16.jsonl').write_text(''.join(dataset.read_text().splitlines(keepends=True)[:16]))
        for mode in ['bf16','w4a8']:
            if args.resume_after_validation:
                arm=stage_arm(plan,mode,work/'smoke'/mode,work/'smoke/checkpoints'/mode,smoke=True,resume=True)
                path=Path(arm['work'])/('resume-'+mode+'-validation.json')
                report=json.loads(path.read_text())
                if (not report.get('passed') or not report.get('nonzero_gradient_seen')
                    or report.get('completed_updates')!=2 or report.get('training_mode')!=mode):
                    raise RuntimeError(f'Missing completed full-size {mode} resume validation')
                retire_smoke(arm,report,image)
                for stage in ['smoke-'+mode,'resume-'+mode]:
                    state['completed_stages'].append(dict(stage=stage,report=str(Path(arm['work'])/(stage+'-validation.json'))))
                continue
            if mode=='bf16' and args.resume_after_bf16_validation:
                arm=stage_arm(plan,mode,work/'smoke'/mode,work/'smoke/checkpoints'/mode,smoke=True,resume=True)
                path=Path(arm['work'])/'resume-bf16-validation.json'
                report=json.loads(path.read_text())
                if report.get('completed_updates')!=2 or report.get('training_mode')!='bf16':
                    raise RuntimeError('Missing completed BF16 resume validation')
                retire_smoke(arm,report,image)
                for stage in ['smoke-bf16','resume-bf16']:
                    state['completed_stages'].append(dict(stage=stage,report=str(Path(arm['work'])/(stage+'-validation.json'))))
                state['bf16_validation_topology']=dict(plan['topology'])
                if args.all_idle_gpus:
                    state['allocation_frozen']=False
                continue
            if mode=='bf16' and args.resume_bf16_smoke:
                first=stage_arm(plan,mode,work/'smoke'/mode,work/'smoke/checkpoints'/mode,smoke=True)
                report=validate_run(first,plan['topology']['train_dp'],mode,1,require_nonzero=False)
                path=Path(first['work'])/'smoke-bf16-validation.json'
                write_json(path,report)
                state['completed_stages'].append(dict(stage='smoke-bf16',report=str(path)))
            else:
                run('smoke-'+mode,mode,smoke=True)
            arm,report=run('resume-'+mode,mode,smoke=True,resume=True)
            # Retain best weights and validation reports; retire only the owned
            # temporary full optimizer snapshot after successful real resume.
            retire_smoke(arm,report,image)
            if mode=='bf16':
                state['bf16_validation_topology']=dict(plan['topology'])
                if args.all_idle_gpus:
                    state['allocation_frozen']=False
        state.update(phase='full_size_smokes_passed')
        state['full_size_validated']=True
        publish()
        for mode in ['bf16','w4a8']:
            run('formal-'+mode,mode)
        state.update(phase='complete')
    except Exception as error:
        state.update(phase='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        publish()


if __name__=='__main__':
    main()
