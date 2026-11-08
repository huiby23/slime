#!/usr/bin/env python3
"""Prepared full30B end-to-end SG performance pilot; never training data or RL."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback
import urllib.request
import urllib.error

WORKSPACE=Path(__file__).resolve().parents[2]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


SMOKE_PATH=WORKSPACE/'experiments/c4v8-integration-probes/sg_smoke.py'
smoke=module('pilot_sg_smoke',SMOKE_PATH)
AUDIT_PATH=WORKSPACE/'experiments/c4v8_rollout_bridge/verify_backend_audit.py'
audit=module('pilot_backend_audit',AUDIT_PATH)
sha=smoke.sha
write=smoke.write


def load_fixture(path):
    fixture=json.loads(path.read_text())
    if (fixture.get('kind')!='actual_dapo_nonthinking_token_ids_performance_only'
        or fixture.get('model_path')!=str(WORKSPACE/'models/Qwen3-30B-A3B')
        or fixture.get('synthetic') is not False or fixture.get('may_be_used_for_training') is not False
        or fixture.get('chat_template_kwargs')!={'enable_thinking':False,'add_generation_prompt':True}
        or len(fixture.get('prompts',[]))!=2):raise ValueError('Expected the actual two-row DAPO nonthinking fixture')
    for source,digest in fixture['tokenizer_source_sha256'].items():
        if not Path(source).resolve().is_relative_to(WORKSPACE/'models/Qwen3-30B-A3B') or sha(source)!=digest:
            raise ValueError('Actual tokenizer identity changed')
    if sha(fixture['data_path'])!=fixture['data_sha256']:raise ValueError('DAPO data changed')
    with open(fixture['data_path']) as f:
        for expected,row in enumerate(fixture['prompts']):
            raw=next(f)
            if row['source_line_index']!=expected or hashlib.sha256(raw.encode()).hexdigest()!=row['source_line_sha256']:
                raise ValueError('DAPO fixture source-row identity mismatch')
            ids=row['input_ids']
            if not isinstance(ids,list) or len(ids)!=row['prompt_tokens'] or not 1<=len(ids)<=8160:
                raise ValueError('Invalid prompt IDs/length')
            if any(type(x) is not int or not 0<=x<fixture['vocabulary_size'] for x in ids):raise ValueError('Invalid actual tokenizer ID')
    return fixture


def configuration(out,policy_path,port,concurrency=1):
    # Same identity construction and transport controls as the proven sg_smoke,
    # with explicit TP8/30B sizing. Unlike its tiny-only helper, query block is
    # read from the final shared policy at execution, never hard-coded to64.
    numerical=smoke.helper(WORKSPACE); policy=json.loads(policy_path.read_text())
    if (set(policy)!={'query_block','key_block','vector','quantize_operands','causal','reduction'}
        or policy['query_block'] not in (64,256,512,1024) or policy['key_block']!=128
        or policy['vector']!='mxfp8' or policy['quantize_operands'] is not True
        or policy['causal'] is not True or policy['reduction']!='mxfp8_tree_reduction'):
        raise ValueError('Full C4V8 policy with query64/256/512/1024 and key128 required')
    core='fast_attention'; implementation=numerical.implementation_profile(WORKSPACE,core)
    implementation_sha=numerical.digest(implementation);policy_sha=numerical.digest(policy)
    sources=dict(implementation['source_sha256']);bridge=WORKSPACE/'experiments/c4v8_rollout_bridge'
    sources.update({str(bridge/n):sha(bridge/n) for n in smoke.BRIDGE_FILES})
    runtime_sha=numerical.digest(dict(backend=smoke.BACKEND,policy=policy,sources=sources,implementation_sha256=implementation_sha))
    env=os.environ.copy()
    env.pop('HBY_C4V8_CPU_SPAWN_PROBE',None)
    env.update(ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        PYTORCH_NPU_ALLOC_CONF='expandable_segments:False',SLIME_NPU_SGLANG_ALLOC_CONF='expandable_segments:False',
        MAX_COMPILE_CORE_NUMBER='1',MAX_JOBS='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HCCL_BUFFSIZE='128',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0',ASCEND_USE_FIA='False',
        SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION='false',SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK='1',SGLANG_LOGITS_PROCESSER_CHUNK_SIZE='64',
        HBY_C4V8_CORE_MODULE=core,HBY_C4V8_POLICY_FILE=str(policy_path),HBY_C4V8_POLICY_SHA256=policy_sha,
        HBY_C4V8_IMPLEMENTATION_ID=implementation['implementation_id'],HBY_C4V8_IMPLEMENTATION_SHA256=implementation_sha,
        HBY_C4V8_EXPECTED_RUNTIME_SHA256=runtime_sha,HBY_C4V8_ALLOW_UNVALIDATED_FUSION='1',
        HBY_C4V8_ROLLOUT_AUDIT_DIR=str(out/'backend-audit'),XDG_CACHE_HOME=str(WORKSPACE/'cache'),
        TORCH_HOME=str(WORKSPACE/'cache/torch'),TORCH_EXTENSIONS_DIR=str(WORKSPACE/'cache/c4v8-sg-probe-extensions'),
        TRITON_CACHE_DIR=str(WORKSPACE/'cache/triton-c4v8'),TMPDIR=str(WORKSPACE/'tmp'))
    env['PYTHONPATH']=os.pathsep.join(map(str,numerical.python_paths(WORKSPACE,core)))+os.pathsep+env.get('PYTHONPATH','')
    server=dict(model_path=str(WORKSPACE/'models/Qwen3-30B-A3B'),device='npu',dtype='bfloat16',kv_cache_dtype='auto',
        host='127.0.0.1',port=port,tp_size=8,dp_size=1,pp_size=1,nnodes=1,base_gpu_id=0,
        context_length=8192,max_total_tokens=12288,max_running_requests=concurrency,chunked_prefill_size=256,
        mem_fraction_static=.135,enable_memory_saver=True,disable_cuda_graph=True,enable_torch_compile=False,
        enable_mixed_chunk=False,disable_overlap_schedule=True,skip_server_warmup=True,
        attention_backend=smoke.BACKEND,prefill_attention_backend=smoke.BACKEND,decode_attention_backend=smoke.BACKEND,
        enable_metrics=True,enable_metrics_for_all_schedulers=True,stream_interval=1,watchdog_timeout=600)
    own_files=[Path(__file__),Path(__file__).with_name('prepare_prompts.py'),Path(__file__).with_name('run-owned-command.sh'),SMOKE_PATH,AUDIT_PATH]
    report=dict(kind='full30B_SG_short_generation_performance_only',model=server['model_path'],core_module=core,
        policy=policy,policy_sha256=policy_sha,policy_file_sha256=sha(policy_path),implementation=implementation,
        implementation_sha256=implementation_sha,runtime_sha256=runtime_sha,runtime_source_sha256=sources,
        probe_source_sha256={str(p):sha(p) for p in own_files},server_args=server,
        actual_http_performed=False,npu_process_started=False,completed=False,optimizer_updates=0,
        may_be_used_for_training=False,accuracy_evaluation=False,quantized_backward=False,
        measurements=[],waves=[],concurrency=concurrency,limitations=['Short fixed16/32 output tokens; not full math completion or RL throughput.',
        'Warm repeats use identical real DAPO token IDs and greedy ignore_eos generation, not training sampling.',
        'return_logprob=True,start_len=0 prevents prefix reuse; cached_tokens must confirm0.',
        'HTTP TTFT includes transport, queue, prefill and first-token work; not pure kernel prefill time.',
        'Finite logprob and backend internal checks do not constitute a full tensor finite scan.'])
    return env,server,report


def require_supervisor():
    parent=os.getppid();argv=(Path('/proc')/str(parent)/'cmdline').read_bytes().split(b'\0')
    if (str(WORKSPACE/'experiments/c4v8-integration-probes/run-owned-probe.py').encode() not in argv
        or b'--owned-child' not in argv or os.getpgid(parent)!=parent):
        raise RuntimeError('Pilot requires root-approved existing run-owned-probe supervisor')


def stop_owned_budgeted(child,owner,out,*,trigger):
    """Finish nested SG cleanup inside the outer controller's10-second grace.

    Reuse pinned PID/startticks/PGID/session/cmdline verification before each
    signal; never signal an unverified process group. This private deadline is
    independent of the long-running HTTP startup/request budget.
    """
    started=time.monotonic();deadline=started+8.0
    result=dict(trigger=trigger,term_grace_seconds=5,kill_wait_max_seconds=3,
                cleanup_budget_seconds=8,outer_term_grace_seconds=10,actions=[])
    def verify_before_signal(label):
        # poll and /proc are separate observations: a leader can exit between
        # them and expose an empty zombie cmdline. Confirm/reap that child;
        # never waive command identity for a still-live process.
        if child.poll() is not None:return False
        try:
            actual=smoke.proc(child.pid)
            fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(') ',1)[1].split()
            actual['state']=fields[0]
            actual['stat_identity_consistent']=(int(fields[19]),int(fields[2]),int(fields[3])) == (actual['startticks'],actual['pgrp'],actual['sid'])
        except FileNotFoundError:
            child.wait(timeout=min(.5,max(.001,deadline-time.monotonic())))
            result.setdefault('exit_races',[]).append(dict(stage=label,observed='proc_disappeared',exit_code=child.returncode))
            return False
        write(out/('before-sg-'+label+'.json'),actual)
        if child.poll() is not None:
            result.setdefault('exit_races',[]).append(dict(stage=label,observed=actual['state'],exit_code=child.returncode))
            return False
        if not actual['stat_identity_consistent']:
            raise RuntimeError('Owned SG identity changed during snapshot; refusing group signal')
        if actual['state'] in ('Z','X'):
            pid=owner['pid']
            if (actual['pid'],actual['startticks'],actual['pgrp'],actual['sid']) != (pid,owner['startticks'],pid,pid):
                raise RuntimeError('Exited SG identity changed; refusing cleanup confirmation')
            child.wait(timeout=min(.5,max(.001,deadline-time.monotonic())))
            result.setdefault('exit_races',[]).append(dict(stage=label,observed=actual['state'],exit_code=child.returncode))
            return False
        try:smoke.verify_owner(owner,actual)
        except RuntimeError:
            # Last bounded reaping check for exit after the state snapshot.
            if child.poll() is None:raise
            result.setdefault('exit_races',[]).append(dict(stage=label,observed='exited_after_snapshot',exit_code=child.returncode))
            return False
        return True

    def signal_verified(label,sig):
        if not verify_before_signal(label):return False
        try:os.killpg(child.pid,sig)
        except ProcessLookupError:
            child.wait(timeout=min(.5,max(.001,deadline-time.monotonic())))
            result.setdefault('exit_races',[]).append(dict(stage=label,observed='group_disappeared',exit_code=child.returncode))
            return False
        result['actions'].append(dict(signal=signal.Signals(sig).name,elapsed_seconds=time.monotonic()-started,identity_reverified=True))
        write(out/'cleanup.json',result)
        return True

    previous_handler=signal.getsignal(signal.SIGTERM)
    # A second controller TERM during finally must not abort cleanup and leave
    # the independently sessioned SG alive. Outer SIGKILL remains effective.
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    try:
        if child.poll() is not None:
            result.update(leader_already_exited=True,exit_code=child.returncode,
                          all_descendants_released='requires outer all-eight-idle gate')
            return result
        signal_verified('term',signal.SIGTERM)
        try:
            child.wait(timeout=min(5,max(.001,deadline-time.monotonic())))
        except subprocess.TimeoutExpired:
            # The pid may have exited immediately after wait timed out. If it
            # did, do not signal a recycled identity; let the outer idle audit
            # detect any unexpected surviving descendant instead.
            if child.poll() is None:
                if signal_verified('kill',signal.SIGKILL):
                    result['forced_stop_reason']='verified SG leader did not exit within5-second TERM grace'
                    child.wait(timeout=min(3,max(.001,deadline-time.monotonic())))
        result.update(leader_already_exited=False,exit_code=child.returncode,
                      all_descendants_released='requires outer all-eight-idle gate')
        return result
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['finished_before_outer_grace']=result['elapsed_seconds']<10
        write(out/'cleanup.json',result)
        signal.signal(signal.SIGTERM,previous_handler)


def stream_request(port,request,path,deadline,request_timeout):
    started=time.monotonic();events=[];first=None;first_count=None;last_token=None;last_count=0;final=None;done=False
    req=urllib.request.Request(f'http://127.0.0.1:{port}/generate',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
    with path.open('w') as event_file:
        with urllib.request.urlopen(req,timeout=min(request_timeout,deadline-started)) as response:
            if response.status!=200:raise RuntimeError('Non200 generate response')
            while True:
                if time.monotonic()>deadline or time.monotonic()-started>request_timeout:raise TimeoutError('Generation deadline exceeded')
                line=response.readline()
                if not line:break
                if not line.startswith(b'data: '):continue
                data=line[6:].strip()
                if data==b'[DONE]':done=True;break
                body=json.loads(data);elapsed=time.monotonic()-started
                event={'client_elapsed_seconds':elapsed,'body':body};event_file.write(json.dumps(event,allow_nan=False)+'\n');event_file.flush()
                if 'error' in body:raise RuntimeError('SG stream returned error: '+str(body['error']))
                count=body.get('meta_info',{}).get('completion_tokens',0)
                if count<last_count:raise ValueError('Completion count moved backwards')
                if count>last_count:
                    if first is None:first=elapsed;first_count=count
                    last_token=elapsed;last_count=count
                final=body;events.append({'elapsed_seconds':elapsed,'completion_tokens':count})
    if not done or final is None or first is None:raise RuntimeError('Incomplete SG stream')
    meta=final['meta_info'];completed=meta['completion_tokens']
    if completed!=request['sampling_params']['max_new_tokens']:raise ValueError('Fixed performance token count not honored')
    if meta.get('prompt_tokens')!=len(request['input_ids']) or meta.get('cached_tokens')!=0:
        raise ValueError('Unexpected prompt length or prefix reuse')
    lp=meta.get('output_token_logprobs')
    if not isinstance(lp,list) or len(lp)!=completed or any(not isinstance(x[0],(int,float)) or not math.isfinite(x[0]) for x in lp):
        raise FloatingPointError('Missing/nonfinite actual output token logprobs')
    prompt_lp=meta.get('input_token_logprobs',[])
    if len(prompt_lp)!=len(request['input_ids']):raise ValueError('Prompt logprob coverage missing')
    for i,item in enumerate(prompt_lp):
        if item[1]!=request['input_ids'][i] or (item[0] is None and i!=0):raise ValueError('Input logprob token alignment mismatch')
        if item[0] is not None and not math.isfinite(item[0]):raise FloatingPointError('Nonfinite prompt logprob')
    elapsed=time.monotonic()-started
    decode=last_token-first
    return final,dict(client_e2e_seconds=elapsed,client_first_nonempty_chunk_seconds=first,
        first_chunk_completion_tokens=first_count,client_first_token_seconds=first if first_count==1 else None,
        client_first_to_last_chunk_seconds=decode,client_decode_tokens_per_second=(completed-first_count)/decode if decode>0 else None,
        client_completion_tokens_per_e2e_second=completed/elapsed,actual_meta_info=meta,
        stream_events=events,logprob_finite=True,input_token_logprob_alignment=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--prompts',type=Path,required=True);p.add_argument('--max-new-tokens',type=int,choices=[16,32],default=16)
    p.add_argument('--concurrency',type=int,choices=[1,4],default=1)
    p.add_argument('--max-seconds',type=int,default=1200);p.add_argument('--startup-seconds',type=int,default=480)
    p.add_argument('--request-seconds',type=int,default=240)
    modes=p.add_mutually_exclusive_group(required=True);modes.add_argument('--prepare-only',action='store_true');modes.add_argument('--execute-after-resource-gate',action='store_true')
    a=p.parse_args();out=a.output.resolve();policy=a.policy.resolve();prompts=a.prompts.resolve()
    if not out.is_relative_to(WORKSPACE/'experiments/c4v8-integration-probes'):raise ValueError('Use dedicated fresh integration evidence path')
    if not 60<=a.startup_seconds<a.max_seconds<=3500 or not 10<=a.request_seconds<=a.max_seconds:raise ValueError('Invalid bounded deadlines')
    fixture=load_fixture(prompts);env,server,report=configuration(out,policy,0,a.concurrency)
    report.update(prompt_fixture=str(prompts),prompt_fixture_sha256=sha(prompts),max_new_tokens=a.max_new_tokens,
                  total_budget_seconds=a.max_seconds,startup_budget_seconds=a.startup_seconds,request_budget_seconds=a.request_seconds)
    out.mkdir(parents=True,exist_ok=False)
    if a.prepare_only:
        report['kind']='CPU preparation only: no NPU/process/HTTP';write(out/'manifest.json',report)
        print(json.dumps({'prepared':True,'server_args':server,'prompt_tokens':[x['prompt_tokens'] for x in fixture['prompts']]}));return
    require_supervisor()
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));server['port']=sock.getsockname()[1]
    for name in ('XDG_CACHE_HOME','TORCH_HOME','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TMPDIR'):Path(env[name]).mkdir(parents=True,exist_ok=True)
    for path in report['runtime_source_sha256']|report['probe_source_sha256']:
        source=Path(path);target=out/'sources'/source.relative_to(WORKSPACE);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
    write(out/'manifest.json',report);child=None;owner=None;deadline=time.monotonic()+a.max_seconds
    def progress(phase,**extra):write(out/'progress.json',dict(phase=phase,utc=time.time(),remaining_seconds=deadline-time.monotonic(),**extra))
    def interrupted(*_):raise InterruptedError('Owned pilot received TERM; stopping only its own SG process group')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with (out/'server.log').open('w') as log:
            command=[str(WORKSPACE/'experiments/c4v8_rollout_bridge/python-c4v8'),'-m','slime.backends.sglang_utils.runtime']
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,text=True)
            owner=smoke.proc(child.pid)
            if owner['pgrp']!=child.pid or owner['sid']!=child.pid:raise RuntimeError('SG does not own its session')
            write(out/'owned-process.json',owner);report['npu_process_started']=True
            child.stdin.write(json.dumps({'server_args':server,'parent_pid':os.getpid()}));child.stdin.close();progress('startup')
            startup_deadline=min(deadline,time.monotonic()+a.startup_seconds)
            while time.monotonic()<startup_deadline:
                if child.poll() is not None:raise RuntimeError('SG exited during startup')
                try:
                    health=smoke.fetch(server['port'],'/health',timeout=2)
                    if health['status']==200:write(out/'health.json',health);break
                except (urllib.error.URLError,TimeoutError,ConnectionError):pass
                time.sleep(2)
            else:raise TimeoutError('SG startup deadline exceeded')
            report['actual_http_performed']=True
            for prompt in fixture['prompts']:
                request=dict(input_ids=prompt['input_ids'],sampling_params={'temperature':0.,'max_new_tokens':a.max_new_tokens,'ignore_eos':True},
                             stream=True,return_logprob=True,logprob_start_len=0)
                request_hash=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
                for iteration in range(3):
                    wave_name=f"{prompt['name']}-{'warm' if iteration==0 else 'timed'+str(iteration)}"
                    progress('generate',request=wave_name,concurrency=a.concurrency)
                    if time.monotonic()>=deadline:raise TimeoutError('Pilot deadline before next request')
                    names=[wave_name if a.concurrency==1 else wave_name+'-request'+str(i) for i in range(a.concurrency)]
                    for name in names:write(out/(name+'-request.json'),request)
                    barrier=threading.Barrier(a.concurrency+1)
                    pool=ThreadPoolExecutor(max_workers=a.concurrency)
                    def generate_one(name):
                        barrier.wait(timeout=10)
                        return name,stream_request(server['port'],request,out/(name+'-events.jsonl'),deadline,a.request_seconds)
                    try:
                        futures=[pool.submit(generate_one,name) for name in names]
                        wave_started=time.monotonic();barrier.wait(timeout=10)
                        for future in as_completed(futures):
                            name,(answer,metrics)=future.result()
                            write(out/(name+'-response.json'),answer)
                            report['measurements'].append(dict(name=name,wave=wave_name,prompt_name=prompt['name'],warmup=iteration==0,request_sha256=request_hash,**metrics))
                            write(out/'manifest.json',report)
                        elapsed=time.monotonic()-wave_started
                        report['waves'].append(dict(name=wave_name,warmup=iteration==0,concurrency=a.concurrency,
                            total_completion_tokens=a.concurrency*a.max_new_tokens,client_wave_seconds=elapsed,
                            aggregate_completion_tokens_per_second=a.concurrency*a.max_new_tokens/elapsed,
                            request_names=names))
                        write(out/'manifest.json',report)
                    finally:
                        # Never wait for HTTP threads here on controller TERM:
                        # the outer finally must stop owned SG within10seconds.
                        # Closing SG then wakes any outstanding local requests.
                        barrier.abort();pool.shutdown(wait=False,cancel_futures=True)
                    # get_internal_state has no receive-loop bootstrap. First
                    # generation must precede server_info (proven pinned issue).
                    if len(report['waves'])==1:
                        info=smoke.fetch(server['port'],'/server_info',timeout=min(20,deadline-time.monotonic()))
                        write(out/'server-info.json',info)
                        if info['status']!=200:raise RuntimeError('server_info failed after first generation')
                        report['backend_audit']=audit.verify(out/'backend-audit',owner['pid'],8,server_info=info['body'])
                        for row_path in (out/'backend-audit').glob('*.json'):
                            row=json.loads(row_path.read_text())
                            if row.get('runtime_sha256')!=report['runtime_sha256']:raise ValueError('Actual runtime differs from planned identity')
                            if row.get('event')=='bootstrap_registered' and row.get('role')=='http_parent' and row.get('fp32_simulator_mode',{}).get('actual_allow_hf32') is not False:
                                raise ValueError('Parent HF32 mode missing')
                        for name,value in [('model_path',server['model_path']),('tp_size',8),('context_length',8192),('max_total_tokens',12288),('chunked_prefill_size',256),('max_running_requests',a.concurrency)]:
                            if info['body'].get(name)!=value:raise ValueError('Actual server configuration differs: '+name)
            smoke.check_frozen_sources(report,policy,WORKSPACE)
            if sha(prompts)!=report['prompt_fixture_sha256']:raise ValueError('Fixture changed during execution')
            load_fixture(prompts);report['requests_completed']=True
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error));(out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        try:
            if child is not None and owner is not None:
                report['cleanup']=stop_owned_budgeted(child,owner,out,
                    trigger=report.get('error_type','normal_request_completion'))
        except BaseException as error:report['cleanup_error']=str(error);raise
        finally:
            logs=(out/'server.log').read_text(errors='replace') if (out/'server.log').exists() else ''
            report['tensor_cpu_fallback_warnings']=[dict(line=i,text=line) for i,line in enumerate(logs.splitlines(),1) if 'npu_cpu_fallback' in line or 'fall back to run on the CPU' in line]
            report['completed']=bool(report.get('requests_completed') and report.get('cleanup') and not report.get('cleanup_error') and not report['tensor_cpu_fallback_warnings'])
            report['remaining_seconds']=deadline-time.monotonic();write(out/'manifest.json',report)
    if not report['completed']:raise RuntimeError('Pilot did not pass completion/fallback gates')
    print(json.dumps({'completed':True,'requests':len(report['measurements']),'optimizer_updates':0,'output':str(out)}))


if __name__=='__main__':main()
