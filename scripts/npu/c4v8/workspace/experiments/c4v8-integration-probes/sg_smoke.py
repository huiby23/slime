#!/usr/bin/env python3
"""Bounded real SG integration probe; --prepare-only uses stdlib and no hardware.

Every invocation creates fresh evidence. The parent reserves the one NPU before
--execute-after-resource-gate; this tool never kills another task. Implementation
identity is computed at invocation, not copied from an older prepared candidate.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

BACKEND = 'hby_c4v8_sim'
BRIDGE_FILES = ('logical_kv.py', 'sglang_bridge.py', 'hardware_compat.py',
                'c4v8_bootstrap.py', 'c4v8_runtime.py', 'python-c4v8')


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def helper(workspace):
    path = Path(workspace)/'experiments/c4v8_rl_training_bridge/numerical_profile.py'
    spec = importlib.util.spec_from_file_location('sg_probe_numerical_profile', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_configuration(workspace, out, policy_path, core_module, port, physical_device=1):
    """Pure preparation: no torch/NPU, socket access, or process construction."""
    workspace, out, policy_path = map(lambda p: Path(p).resolve(), (workspace, out, policy_path))
    numerical = helper(workspace)
    policy = json.loads(policy_path.read_text())
    if set(policy) != {'query_block','key_block','vector','quantize_operands','causal','reduction'}:
        raise ValueError('Exactly the six numerical policy fields are required')
    if (policy['vector'] != 'mxfp8' or policy['causal'] is not True
            or policy['quantize_operands'] is not True or policy['reduction'] != 'mxfp8_tree_reduction'):
        raise ValueError('This probe requires the full forward NVFP4/MXFP8 numerical policy')
    if core_module == 'fast_attention' and (policy['query_block'] not in (64,256,512,1024) or policy['key_block'] != 128):
        raise ValueError('The explicit fast candidate requires query64/256/512/1024 and key128; this probe does not pass a training launch gate')
    implementation = numerical.implementation_profile(workspace, core_module)
    implementation_sha = numerical.digest(implementation)
    policy_sha = numerical.digest(policy)
    runtime_sources = dict(implementation['source_sha256'])
    bridge = workspace/'experiments/c4v8_rollout_bridge'
    runtime_sources.update({str(bridge/name): sha(bridge/name) for name in BRIDGE_FILES})
    runtime_sha = numerical.digest(dict(backend=BACKEND, policy=policy, sources=runtime_sources,
                                        implementation_sha256=implementation_sha))
    env = os.environ.copy()
    for key in ('HBY_C4V8_CPU_SPAWN_PROBE','HBY_C4V8_ALLOW_UNVALIDATED_FUSION'):
        env.pop(key, None)
    env.update(ASCEND_RT_VISIBLE_DEVICES=str(physical_device), CUDA_VISIBLE_DEVICES=str(physical_device),
        PYTORCH_NPU_ALLOC_CONF='expandable_segments:False', SLIME_NPU_SGLANG_ALLOC_CONF='expandable_segments:False',
        MAX_COMPILE_CORE_NUMBER='1', MAX_JOBS='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        HCCL_BUFFSIZE='128', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        ASCEND_USE_FIA='False', SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION='false',
        SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK='1', SGLANG_LOGITS_PROCESSER_CHUNK_SIZE='64',
        HBY_C4V8_CORE_MODULE=core_module, HBY_C4V8_POLICY_FILE=str(policy_path),
        HBY_C4V8_POLICY_SHA256=policy_sha, HBY_C4V8_IMPLEMENTATION_ID=implementation['implementation_id'],
        HBY_C4V8_IMPLEMENTATION_SHA256=implementation_sha, HBY_C4V8_EXPECTED_RUNTIME_SHA256=runtime_sha,
        HBY_C4V8_ROLLOUT_AUDIT_DIR=str(out/'backend-audit'),
        XDG_CACHE_HOME=str(workspace/'cache'), TORCH_HOME=str(workspace/'cache/torch'),
        TORCH_EXTENSIONS_DIR=str(workspace/'cache/c4v8-sg-probe-extensions'),
        TRITON_CACHE_DIR=str(workspace/'cache/triton-c4v8'), TMPDIR=str(workspace/'tmp'))
    if core_module == 'fast_attention':
        env['HBY_C4V8_ALLOW_UNVALIDATED_FUSION'] = '1'
    env['PYTHONPATH'] = os.pathsep.join(map(str,numerical.python_paths(workspace,core_module))) + os.pathsep + env.get('PYTHONPATH','')
    server = dict(model_path=str(workspace/'models/Qwen3-0.6B'), device='npu', dtype='bfloat16',
        kv_cache_dtype='auto', host='127.0.0.1', port=port, tp_size=1, dp_size=1, pp_size=1, nnodes=1,
        base_gpu_id=0, context_length=256, max_total_tokens=512, max_running_requests=1,
        chunked_prefill_size=128, mem_fraction_static=0.15, enable_memory_saver=True,
        disable_cuda_graph=True, enable_torch_compile=False, enable_mixed_chunk=False,
        disable_overlap_schedule=True, skip_server_warmup=True,
        attention_backend=BACKEND, prefill_attention_backend=BACKEND, decode_attention_backend=BACKEND,
        watchdog_timeout=300)
    report = dict(validation_kind='real_SGLang_small_model_integration', model=server['model_path'],
        physical_device=physical_device, core_module=core_module, policy=policy, policy_sha256=policy_sha,
        implementation=implementation, implementation_sha256=implementation_sha,
        runtime_sha256=runtime_sha, runtime_source_sha256=runtime_sources,
        probe_source_sha256={str(Path(__file__).resolve()):sha(__file__)},
        policy_file_sha256=sha(policy_path), optimizer_updates=0, production_30b_validated=False,
        server_args=server, started=time.time(), passed=False, actual_http_performed=False,
        npu_process_started=False, quantized_backward=False,
        selected_environment={name:env[name] for name in env if name.startswith('HBY_C4V8_') or name in
            ('PYTHONPATH','TRITON_CACHE_DIR','XDG_CACHE_HOME','TORCH_HOME','TORCH_EXTENSIONS_DIR','TMPDIR',
             'ASCEND_RT_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','MAX_COMPILE_CORE_NUMBER','HCCL_BUFFSIZE',
             'PYTORCH_NPU_ALLOC_CONF','SLIME_NPU_SGLANG_ALLOC_CONF')})
    return env, server, report


def fetch(port, path, data=None, timeout=20):
    body = None if data is None else json.dumps(data).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=body,
                                     headers={'Content-Type': 'application/json'})
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read().decode()
        try: payload = json.loads(raw) if raw else None
        except json.JSONDecodeError: payload = raw
        return dict(path=path, status=response.status, body=payload)


def proc(pid):
    base = Path('/proc') / str(pid)
    stat = (base/'stat').read_text().rsplit(') ',1)[1].split()
    return dict(pid=pid,pgrp=int(stat[2]),sid=int(stat[3]),startticks=int(stat[19]),
                cmdline=(base/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())


def verify_owner(owner, actual):
    pid = owner['pid']
    if (actual['pid'],actual['startticks'],actual['pgrp'],actual['sid']) != (pid,owner['startticks'],pid,pid):
        raise RuntimeError('Owned SG PID/PGID/session identity changed; refusing group signal')
    if not any(name in actual['cmdline'] for name in ('c4v8_runtime','python-c4v8')):
        raise RuntimeError('Owned SG command identity changed; refusing group signal')


def stop_owned(child, owner, out):
    if child.poll() is not None:
        return dict(leader_already_exited=True, exit_code=child.returncode)
    actual = proc(child.pid)
    verify_owner(owner,actual)
    write(out/'stop-owned-process.json',actual)
    os.killpg(child.pid,signal.SIGTERM)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        actual=proc(child.pid)
        verify_owner(owner,actual)
        os.killpg(child.pid,signal.SIGKILL)
        child.wait(timeout=10)
    return dict(leader_already_exited=False,exit_code=child.returncode)


def verify_audit(directory, expected):
    paths=sorted(Path(directory).glob('*.json'))
    records=[json.loads(path.read_text()) for path in paths]
    if not records:raise ValueError('No actual SG backend audit was produced')
    for row in records:
        for name in ('policy_sha256','implementation_sha256','runtime_sha256','core_module'):
            if row.get(name)!=expected[name]:raise ValueError('SG audit identity mismatch: '+name)
        if row.get('implementation_id')!=expected['implementation']['implementation_id']:
            raise ValueError('SG implementation ID mismatch')
        if row.get('execution_validation_kind')!='actual_process':raise ValueError('CPU-stub SG evidence rejected')
        for path,value in row.get('source_sha256',{}).items():
            if expected['runtime_source_sha256'].get(path)!=value or sha(path)!=value:
                raise ValueError('Actual SG source identity changed')
    modes={}
    for role in ('http_parent','scheduler'):
        selected=[row for row in records if row['event']=='bootstrap_registered' and row.get('role')==role]
        if not selected:raise ValueError('Missing real process bootstrap: '+role)
        for row in selected:
            if row.get('source_sha256')!=expected['runtime_source_sha256']:
                raise ValueError('Bootstrap did not record the complete actual runtime source identity')
            mode=row['fp32_simulator_mode']
            if mode.get('applied') is not True or mode.get('actual_allow_hf32') is not False:
                raise ValueError('Missing actual HF32=False readback: '+role)
        modes[role]=[dict(pid=row['pid'],mode=row['fp32_simulator_mode']) for row in selected]
    for event in ('backend_created','first_extend','first_decode'):
        rows=[row for row in records if row['event']==event and row.get('role')=='scheduler']
        if {row.get('ranks',{}).get('tp_rank') for row in rows}!={0}:
            raise ValueError('Missing actual TP1 prefill/decode evidence: '+event)
    effective=[row for row in records if row['event']=='effective_server_args']
    if {row.get('role') for row in effective}!={'http_parent','scheduler'}:
        raise ValueError('Missing real parent/child effective settings')
    for row in effective:
        if any(row['actual'].get(key)!=BACKEND for key in
            ('attention_backend','prefill_attention_backend','decode_attention_backend')):
            raise ValueError('Actual runtime selected a different attention backend')
        actual=row['actual']
        if (actual.get('model_path')!=expected['model'] or actual.get('tp_size')!=1
                or actual.get('dtype') not in ('bf16','bfloat16')
                or actual.get('kv_cache_dtype') not in ('auto','bfloat16')):
            raise ValueError('Actual runtime model, TP, or dtype differs from the probe')
    return dict(parent_and_scheduler_hf32=modes,evidence_sha256={str(p):sha(p) for p in paths},
                actual_prefill_and_decode=True)


def check_frozen_sources(report, policy_path, workspace):
    for path,value in (report['runtime_source_sha256']|report['probe_source_sha256']).items():
        if sha(path)!=value:raise ValueError('Probe source changed during execution: '+path)
    if sha(policy_path)!=report['policy_file_sha256']:raise ValueError('Policy changed during execution')
    numerical=helper(workspace)
    if numerical.implementation_profile(workspace,report['core_module'])!=report['implementation']:
        raise ValueError('Numerical source or installed toolchain changed during execution')


def load_long_prompt(path, model):
    """Read an already CPU-tokenized request; validate actual tokenizer bytes."""
    fixture=json.loads(Path(path).read_text())
    if fixture.get('kind')!='offline_real_tokenizer_request_fixture' or fixture.get('model_path')!=model:
        raise ValueError('Long prompt fixture must belong to this actual model tokenizer')
    ids=fixture['input_ids']
    if (not isinstance(ids,list) or len(ids)!=fixture['prompt_tokens'] or not 129<=len(ids)<=240
            or any(type(x) is not int or not 0<=x<fixture['vocabulary_size'] for x in ids)):
        raise ValueError('The long prompt must contain129..240 authentic token IDs')
    sources=fixture['tokenizer_source_sha256']
    if not sources:raise ValueError('Tokenizer source identities are required')
    for name,value in sources.items():
        if not Path(name).resolve().is_relative_to(Path(model).resolve()) or sha(name)!=value:
            raise ValueError('Actual tokenizer source changed')
    return fixture


def check_generation(answer, *, expected_prompt_tokens=None):
    meta=answer.get('meta_info',{})
    if not isinstance(answer.get('text'),str) or meta.get('completion_tokens',0)<2:
        raise ValueError('Probe must execute two genuine output tokens, including decode')
    if expected_prompt_tokens is not None:
        actual=meta.get('prompt_tokens')
        if type(actual) is not int or not 129<=actual<=240 or actual!=expected_prompt_tokens:
            raise ValueError('Actual long-request prompt length did not cover the128-key tree')
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--policy',required=True)
    parser.add_argument('--core-module',choices=('device_attention','fast_attention'),default='device_attention')
    parser.add_argument('--physical-device',type=int,default=1)
    parser.add_argument('--prompt-fixture',type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--execute-after-resource-gate',action='store_true')
    args=parser.parse_args()
    workspace=Path(__file__).resolve().parents[2]
    out=Path(args.output).resolve()
    if not out.is_relative_to(workspace/'experiments/c4v8-integration-probes'):
        raise ValueError('Evidence must stay in the dedicated probe directory')
    if out.exists():raise ValueError('Choose a fresh evidence directory')
    if not (args.prepare_only or args.execute_after_resource_gate):
        raise RuntimeError('Use CPU --prepare-only, or let the parent reserve NPU before explicit execution')
    port=0
    if args.execute_after_resource_gate:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env,server,report=build_configuration(workspace,out,args.policy,args.core_module,port,args.physical_device)
    fixture_path=(args.prompt_fixture or workspace/'experiments/c4v8_rl_training_bridge/role-probe-prompts-v1.json').resolve()
    long_prompt=load_long_prompt(fixture_path,server['model_path'])
    report['long_prompt_fixture']=dict(path=str(fixture_path),sha256=sha(fixture_path),
                                     prompt_tokens=long_prompt['prompt_tokens'])
    report['length_coverage']='Short request plus real129..240-token request; not4096-token throughput validation'
    out.mkdir(parents=True)
    if args.prepare_only:
        report['validation_kind']='CPU preparation only; no HTTP, subprocess, model or NPU'
        write(out/'manifest.json',report)
        print(json.dumps(report),flush=True)
        return
    for name in ('XDG_CACHE_HOME','TORCH_HOME','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TMPDIR'):
        Path(env[name]).mkdir(parents=True,exist_ok=True)
    write(out/'manifest.json',report)
    command=[str(workspace/'experiments/c4v8_rollout_bridge/python-c4v8'),'-m','slime.backends.sglang_utils.runtime']
    with (out/'server.log').open('w') as log:
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,
                               env=env,start_new_session=True,text=True)
        owner=proc(child.pid)
        if owner['pgrp']!=child.pid or owner['sid']!=child.pid:
            raise RuntimeError('SG did not start in an independent session')
        write(out/'owned-process.json',owner)
        report['npu_process_started']=True
        try:
            execution_deadline=time.monotonic()+560
            child.stdin.write(json.dumps({'server_args':server,'parent_pid':os.getpid()}));child.stdin.close()
            deadline=time.monotonic()+360
            while time.monotonic()<deadline:
                if child.poll() is not None:raise RuntimeError(f'SG exited during initialization: {child.returncode}')
                try:
                    health=fetch(port,'/health',timeout=2)
                    if health['status']==200:
                        report['health']=health;write(out/'health.json',health);break
                except (urllib.error.URLError,TimeoutError,ConnectionError):pass
                time.sleep(2)
            else:raise TimeoutError('SG startup deadline exceeded')
            report['actual_http_performed']=True
            report['generations']={}
            for name,prompt,expected_tokens in (
                    ('short',dict(text='What is 2 + 3? Answer briefly.'),None),
                    ('long',dict(input_ids=long_prompt['input_ids']),long_prompt['prompt_tokens'])):
                request=prompt|dict(sampling_params={'temperature':0.0,'max_new_tokens':2,'ignore_eos':True},
                                    return_logprob=True,logprob_start_len=0)
                write(out/f'generation-{name}-request.json',request)
                remaining=execution_deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Overall probe deadline exceeded')
                start=time.monotonic()
                result=fetch(port,'/generate',request,timeout=min(180,remaining))
                write(out/f'generation-{name}-http.json',result)
                if result['status']!=200 or not isinstance(result['body'],dict):raise RuntimeError('Generation failed: '+name)
                answer=result['body'];write(out/f'generation-{name}.json',answer)
                meta=check_generation(answer,expected_prompt_tokens=expected_tokens)
                report['generations'][name]=dict(seconds=time.monotonic()-start,
                    actual_prompt_tokens=meta.get('prompt_tokens'),completion_tokens=meta['completion_tokens'])
                if name=='short':write(out/'generation.json',answer)
                write(out/'manifest.json',report)
            # Pinned skip-warmup runtime starts its tokenizer receive loop on
            # the first generate, not on /health or server-info. Query the
            # canonical metadata endpoint only after both real generations.
            remaining=execution_deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Overall probe deadline exceeded before server info')
            info=fetch(port,'/server_info',timeout=min(20,remaining))
            if info['status']==404:
                write(out/'server-info-404.json',info)
                remaining=execution_deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Overall probe deadline exceeded before legacy server info')
                info=fetch(port,'/get_server_info',timeout=min(20,remaining))
            write(out/'server-info.json',info)
            if info['status']!=200:raise RuntimeError('Actual server info endpoint failed after generation')
            report['server_info_endpoint']=info['path']
            if sha(fixture_path)!=report['long_prompt_fixture']['sha256']:
                raise ValueError('Prompt fixture changed during execution')
            load_long_prompt(fixture_path,server['model_path'])
            report['backend_audit']=verify_audit(out/'backend-audit',report)
            check_frozen_sources(report,args.policy,workspace)
            report['integration_passed']=True
        except BaseException as error:
            report.update(error_type=type(error).__name__,error=str(error))
            raise
        finally:
            try:report['cleanup']=stop_owned(child,owner,out)
            except BaseException as error:
                report['cleanup_error']=dict(type=type(error).__name__,message=str(error))
                raise
            finally:
                log.flush()
                report['tensor_cpu_fallback_warnings']=[dict(line=i,text=line) for i,line in
                    enumerate((out/'server.log').read_text(errors='replace').splitlines(),1)
                    if 'npu_cpu_fallback' in line or 'fall back to run on the CPU' in line]
                report.update(server_exit_code=child.returncode,finished=time.time())
                report['passed']=(bool(report.get('integration_passed')) and 'cleanup' in report
                                  and not report['tensor_cpu_fallback_warnings'])
                write(out/'manifest.json',report)
    if not report['passed']:raise RuntimeError('Real integration ran but Tensor CPU fallback remains; see manifest')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
