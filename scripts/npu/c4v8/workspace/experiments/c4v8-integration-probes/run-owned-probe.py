#!/usr/bin/env python3
"""Bound one root-owned probe; never stops existing tasks or reads environment secrets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

WORKSPACE=Path('/root/hby')


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_command(command):
    if command[:1]==['--']:command=command[1:]
    if not command:raise ValueError('An explicit argv command is required after --')
    executable=Path(command[0])
    if command[0] in ('/bin/bash','/usr/bin/bash'):
        if len(command)<2 or not Path(command[1]).is_absolute():raise ValueError('Bash requires a script under /root/hby; no -c')
        source=Path(command[1]).resolve(strict=True)
        if not source.is_relative_to(WORKSPACE):raise ValueError('Only /root/hby scripts are allowed')
    else:
        source=executable.resolve(strict=True)
        if not executable.is_absolute() or not source.is_relative_to(WORKSPACE):
            raise ValueError('Only explicit executables under /root/hby are allowed')
    return command,source


def proc(pid):
    base=Path('/proc')/str(pid)
    fields=(base/'stat').read_text().rsplit(') ',1)[1].split()
    return dict(pid=pid,state=fields[0],pgrp=int(fields[2]),sid=int(fields[3]),startticks=int(fields[19]),
                argv=[x.decode() for x in (base/'cmdline').read_bytes().split(b'\0') if x])


def verify_owner(owner,current):
    for name in ('pid','startticks','pgrp','sid','argv'):
        if owner[name]!=current[name]:raise RuntimeError('Owned process identity changed: '+name+'; no signal sent')
    if current['pgrp']!=current['pid'] or current['sid']!=current['pid']:
        raise RuntimeError('Refusing to signal a non-independent group')
    if str(Path(__file__).resolve()) not in current['argv'] or '--owned-child' not in current['argv']:
        raise RuntimeError('Refusing to signal an unrelated command')


def npu_snapshot(path):
    executable=shutil.which('npu-smi') or '/usr/local/Ascend/driver/tools/npu-smi'
    try:
        result=subprocess.run([executable,'info'],capture_output=True,text=True,timeout=30)
        idle=sorted(set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)',result.stdout))))
        report=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,idle_cards=idle,
                    all_eight_idle=result.returncode==0 and idle==list(range(8)))
    except Exception as error:
        report=dict(all_eight_idle=False,error_type=type(error).__name__,error=str(error))
    report['utc']=datetime.now(timezone.utc).isoformat()
    write(path,report)
    return report


def stop_owned(child,owner,out):
    if child.poll() is not None:return dict(already_exited=True,exit_code=child.returncode)
    current=proc(child.pid);verify_owner(owner,current)
    write(out/'before-term.json',current)
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        current=proc(child.pid);verify_owner(owner,current)
        write(out/'before-kill.json',current)
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
    return dict(already_exited=False,exit_code=child.returncode)


def owned_child(command):
    # A stable supervisor PID keeps exact argv across shell exec transitions.
    # On TERM it remains identifiable until the parent can KILL the same group;
    # the actual probe receives TERM normally and has the grace period to exit.
    if os.getsid(0)!=os.getpid() or os.getpgrp()!=os.getpid():
        raise RuntimeError('Internal supervisor must own its independent session')
    term=[False]
    signal.signal(signal.SIGTERM,lambda *_:term.__setitem__(0,True))
    code=subprocess.call(command)
    while term[0]:time.sleep(1)
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--timeout-seconds',type=int,default=600)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--owned-child',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command,source=validate_command(args.command)
    if args.owned_child:return owned_child(command)
    if not args.output or not 10<=args.timeout_seconds<=3600:
        raise ValueError('Fresh --output and timeout10..3600 required')
    out=args.output.resolve()
    if not out.is_relative_to(WORKSPACE/'experiments/c4v8-integration-probes'):
        raise ValueError('Evidence must remain under the dedicated probe directory')
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='prepared',command=command,timeout_seconds=args.timeout_seconds,
                command_source=str(source),command_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                started_utc=datetime.now(timezone.utc).isoformat(),child_started=False,
                numerical_acceptance_claimed=False,actual_grpo_updates=0)
    write(out/'result.json',report)
    if args.prepare_only:
        print(json.dumps(report),flush=True);return 0
    child=None;owner=None;started=time.monotonic()
    def interrupted(*_):raise InterruptedError('Root-owned controller received termination')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        before=npu_snapshot(out/'npu-before.json')
        if not before['all_eight_idle']:raise RuntimeError('All-eight NPU idle gate rejected; no task stopped or started')
        with (out/'outer.log').open('w') as output:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--owned-child','--',*command],
                stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            owner=proc(child.pid);verify_owner(owner,owner)
            write(out/'owned-process.json',owner)
            report.update(child_started=True,owner=owner,status='running');write(out/'result.json',report)
            try:
                child.wait(timeout=args.timeout_seconds)
                report['status']='exited'
            except subprocess.TimeoutExpired:
                report['status']='timed_out'
                report['cleanup']=stop_owned(child,owner,out)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
    finally:
        if child is not None:
            if owner is not None and child.poll() is None:
                try:report['cleanup']=stop_owned(child,owner,out)
                except BaseException as error:report['cleanup_error']=dict(type=type(error).__name__,message=str(error))
            report['child_exit_code']=child.poll()
        after=npu_snapshot(out/'npu-after.json')
        report.update(after_all_eight_idle=after['all_eight_idle'],elapsed_seconds=time.monotonic()-started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        if (out/'outer.log').exists():
            report['outer_log_sha256']=hashlib.sha256((out/'outer.log').read_bytes()).hexdigest()
            report['tensor_cpu_fallback_warnings']=[dict(line=i,text=line) for i,line in enumerate(
                (out/'outer.log').read_text(errors='replace').splitlines(),1)
                if 'npu_cpu_fallback' in line or 'fall back to run on the CPU' in line]
        report['execution_completed_cleanly']=(report['status']=='exited' and report.get('child_exit_code')==0
                                               and after['all_eight_idle'] and 'cleanup_error' not in report)
        write(out/'result.json',report)
    print(json.dumps(report),flush=True)
    return 0 if report['execution_completed_cleanly'] else 1

if __name__=='__main__':raise SystemExit(main())
