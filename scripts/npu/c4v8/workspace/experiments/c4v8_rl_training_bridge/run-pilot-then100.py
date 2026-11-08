#!/usr/bin/env python3
"""Observe first5 of the existing100-update process, then continue the same Adam.

Both time budgets are explicit and must follow measured representative speed.
This controller does not alter the5x20 schedule, reward, sampling, or checkpoint
interval. The existing recipe's full launch gates are always enforced.
"""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from pilot_evidence import read_progress

ORIGINAL_DEVICE_AUTOLOAD=os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
ROOT=Path('/root/hby')


def write(path,value):
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot-timeout-seconds',type=int,required=True)
    parser.add_argument('--run-timeout-seconds',type=int,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if not 60<=args.pilot_timeout_seconds<=args.run_timeout_seconds<=7*24*3600:
        raise ValueError('Require60<=pilot budget<=total budget<=7days; derive budgets from actual throughput')
    prepared=args.prepared_dir.resolve();out=args.output.resolve()
    if (not prepared.is_relative_to(ROOT/'experiments/c4v8_rl_training_bridge')
            or not out.is_relative_to(ROOT/'experiments/c4v8_rl_training_bridge')):
        raise ValueError('Only dedicated /root/hby experiment paths are accepted')
    launcher=prepared/'run-c4v8-dapo100-every5.sh'
    if not launcher.is_file():raise FileNotFoundError(launcher)
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='prepared',pilot_accept_after_actual_updates=5,first_gradient_is_runtime_observation=True,
        pilot_timeout_seconds=args.pilot_timeout_seconds,run_timeout_seconds=args.run_timeout_seconds,
        schedule='5 real rollouts x5 prompts x16 answers /globalbatch4 =100 actual updates',
        optimizer_state_preserved_across_pilot_boundary=True,command=['/bin/bash',str(launcher)],
        whole_model_logprob_agreement='unknown',post_training_model_and21point_acceptance_complete=False)
    write(out/'controller.json',report)
    if args.prepare_only:print(json.dumps(report));return 0
    module_path=ROOT/'experiments/c4v8-integration-probes/run-owned-probe.py'
    spec=importlib.util.spec_from_file_location('owned_runtime',module_path)
    owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    command,_=owned.validate_command(report['command'])
    before=owned.npu_snapshot(out/'npu-before.json')
    if not before['all_eight_idle']:
        report.update(status='resource_gate_rejected',npu_before=before,child_started=False)
        write(out/'controller.json',report);print(json.dumps(report));return 1
    child=None;owner=None;run=None;started=time.monotonic();first_seen=False
    def interrupt(*_):raise InterruptedError('Controller termination requested')
    signal.signal(signal.SIGTERM,interrupt)
    try:
        with (out/'outer.log').open('w') as log:
            # The CPU-only evidence reader must not change the validated child
            # initialization environment. Restore its original autoload setting.
            child_environment=os.environ.copy()
            if ORIGINAL_DEVICE_AUTOLOAD is None:child_environment.pop('TORCH_DEVICE_BACKEND_AUTOLOAD',None)
            else:child_environment['TORCH_DEVICE_BACKEND_AUTOLOAD']=ORIGINAL_DEVICE_AUTOLOAD
            child=subprocess.Popen([sys.executable,str(module_path),'--owned-child','--',*command],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=child_environment)
            owner=owned.proc(child.pid);owned.verify_owner(owner,owner)
            write(out/'owned-process.json',owner)
            report.update(status='pilot_running',started_utc=datetime.now(timezone.utc).isoformat())
            while child.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed>args.run_timeout_seconds:raise TimeoutError('Total training time budget exceeded')
                if report['status']=='pilot_running' and elapsed>args.pilot_timeout_seconds:
                    raise TimeoutError('Bounded first5-update/checkpoint pilot deadline exceeded')
                if run is None:
                    matches=re.findall(r'^QWEN30B_RUN_DIR=(.+)$',(out/'outer.log').read_text(errors='replace'),re.MULTILINE)
                    if matches:
                        if len(set(matches))!=1:raise ValueError('Multiple run identities in one owned process')
                        candidate=Path(matches[0]).resolve()
                        if not candidate.is_relative_to(ROOT/'runs/qwen3-30b-a3b-dapo-math-17k'):
                            raise ValueError('Unexpected training output directory')
                        try:manifest=json.loads((candidate/'manifest.json').read_text())
                        except json.JSONDecodeError:
                            time.sleep(.2);continue  # A concurrent small manifest rewrite is not a failed job.
                        driver=owned.proc(int(manifest['training_driver_pid']))
                        if driver['pgrp']!=owner['pgrp']:raise ValueError('Training driver is not in the owned group')
                        if manifest.get('initial_model')!=str(ROOT/'models/Qwen3-30B-A3B') or manifest.get('parent_completed_optimizer_steps')!=0:
                            raise ValueError('Pilot must be the original fresh-model trajectory')
                        run=candidate;report['training_run']=str(run)
                if run is not None:
                    progress=read_progress(run)
                    if progress['first_gradient_observed'] and not first_seen:
                        write(out/'first-gradient-observed.json',progress);first_seen=True
                    if report['status']=='pilot_running' and progress['pilot_passed']:
                        write(out/'pilot-accepted.json',progress)
                        report['status']='continuing_same_process_to100'
                        report['pilot_accepted_elapsed_seconds']=elapsed
                    report['progress']=progress
                report['elapsed_seconds']=elapsed;write(out/'controller.json',report)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:pass
        if child.returncode!=0:raise RuntimeError('Owned training process exited '+str(child.returncode))
        if run is None:raise ValueError('Training exited without an identified run')
        progress=read_progress(run,minimum_file_age=0)
        manifest=json.loads((run/'manifest.json').read_text())
        if manifest.get('status')!='completed' or progress['gradient_norm_records']!=100 or not progress['final_hundred_checkpoint_complete']:
            raise ValueError('Training exit0 is not a completed100-update trajectory')
        report.update(status='training100_finished_pending_final_acceptance',progress=progress)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
    finally:
        if child is not None:
            if owner is not None and child.poll() is None:
                try:report['owned_cleanup']=owned.stop_owned(child,owner,out)
                except BaseException as error:report['cleanup_error']=str(error)
            report['child_exit_code']=child.poll()
        report['npu_after']=owned.npu_snapshot(out/'npu-after.json')
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        write(out/'controller.json',report)
    good=(report['status']=='training100_finished_pending_final_acceptance'
          and report['npu_after']['all_eight_idle'] and 'cleanup_error' not in report)
    print(json.dumps(report));return 0 if good else 1

if __name__=='__main__':raise SystemExit(main())
