#!/root/hby/envs/slime/bin/python
"""One synchronous offline evaluation of a completed base->100 training run.

Owns one SG process session, all 21 fresh points, then exits. No retries,
daemon, training, Ray initialization, or direct accelerator framework imports.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from contextlib import nullcontext
from eval_deadlines import positive_seconds
from c4v8_eval_protocol import (
    STEPS, PROTOCOL_SHA256, atomic_json, canonical_sha, validate_summary, rebuild_curve)

SETUP = Path('/root/hby/setup')
BASE = Path('/root/hby/models/Qwen3-30B-A3B')
DATA = Path('/root/hby/data/aime-2024/eval.jsonl')
CLIENT = Path(__file__).with_name('eval-c4v8-client.py')
STARTER = Path(__file__).with_name('run-c4v8-eval-server.sh')
PYTHON = '/root/hby/envs/slime/bin/python'
URL = 'http://127.0.0.1:15115'
POINT_FIELDS = ('optimizer_step', 'model_path', 'weight_version', 'weight_identity',
                'protocol_sha256', 'protocol', 'execution_settings', 'experiment_identity_sha256')


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def client_module():
    spec = importlib.util.spec_from_file_location('every5_cpu_client', CLIENT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def proc(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return {'pid': int(pid), 'state': fields[0], 'ppid': int(fields[1]),
                'pgid': int(fields[2]), 'session': int(fields[3]), 'start_ticks': int(fields[19]),
                'argv': [value.decode(errors='replace') for value in argv if value]}
    except (FileNotFoundError, ProcessLookupError): return None


def group_members(pgid):
    result = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            info = proc(int(entry.name))
            if info and info['pgid'] == pgid and info['state'] != 'Z': result.append(info)
    return result


def port_busy():
    with socket.socket() as connection:
        connection.settimeout(0.5)
        return connection.connect_ex(('127.0.0.1', 15115)) == 0


def plan_models(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('status') != 'completed' or manifest.get('optimizer_steps') != 100
            or manifest.get('new_optimizer_steps') != 100
            or manifest.get('parent_completed_optimizer_steps') != 0
            or Path(manifest.get('initial_model', '')).resolve() != BASE.resolve()
            or manifest.get('optimizer_step_hf_checkpoint_interval') != 5
            or manifest.get('optimizer_step_hf_expected_global_steps') != list(STEPS[1:])
            or manifest.get('offline_checkpoint_validation') is not True
            or manifest.get('validation_execution') != 'post_training_sequential_http'
            or manifest.get('validation_expected_global_steps') != list(STEPS)
            or manifest.get('validation_protocol_sha256') != PROTOCOL_SHA256):
        raise ValueError('Require a completed official-base run with 100 real new updates and all20 every5 snapshots')
    norms = manifest.get('gradient_norms', [])
    if len(norms) != 100 or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in norms):
        raise ValueError('Completed training must prove100 finite positive-gradient updates')
    checkpoint_root = Path(manifest['optimizer_step_hf_checkpoint_directory']).resolve()
    if checkpoint_root != manifest_path.parent / 'optimizer-step-checkpoints':
        raise ValueError('Every5 checkpoint directory must belong to this training run')
    models = {0: BASE.resolve()}
    for step in STEPS[1:]:
        model = checkpoint_root / f'checkpoint_step_{step:06d}'
        metadata = json.loads((model / 'optimizer-step-checkpoint.json').read_text())
        if (metadata.get('completed_optimizer_steps') != step
                or metadata.get('parent_completed_optimizer_steps') != 0
                or metadata.get('new_successful_optimizer_steps') != step
                or metadata.get('complete') is not True or metadata.get('hf_only') is not True
                or metadata.get('serving_weight_version_changed') is not False
                or metadata.get('optimizer_state_saved') is not False
                or metadata.get('rng_state_saved') is not False
                or metadata.get('source_cursor_saved') is not False):
            raise ValueError('Snapshot metadata is incomplete or has wrong successful-update identity: ' + str(model))
        models[step] = model
    listed = [str(Path(path).resolve()) for path in manifest.get('optimizer_step_hf_checkpoints', [])]
    if len(listed) != 20 or set(listed) != {str(models[step]) for step in STEPS[1:]}:
        raise ValueError('Manifest must list exactly its20 completed independent HF snapshots')
    declared = manifest.get('validation_checkpoint_model_paths', {})
    if declared != {str(step):str(model) for step,model in models.items()}:
        raise ValueError('Manifest validation point-to-model mapping differs from snapshots')
    return manifest, models


def assert_resources_released(report_path, manifest_path):
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    stamp = datetime.datetime.fromisoformat(report['checked_utc'].replace('Z', '+00:00'))
    age = (datetime.datetime.now(datetime.timezone.utc) - stamp).total_seconds()
    if (report.get('passed') is not True or report.get('all_npu_idle') is not True
            or report.get('owned_training_processes_released') is not True
            or Path(report.get('training_run_directory', '')).resolve() != Path(manifest_path).resolve().parent
            or not 0 <= age <= 600):
        raise ValueError('Need a matching fresh <=10min all-NPU-idle and owned-training-release gate report')
    if report.get('detail_report') and report.get('detail_report_sha256'):
        if sha(report['detail_report']) != report['detail_report_sha256']:
            raise ValueError('Resource-release detail report changed')
    if port_busy(): raise RuntimeError('Port15115 is already owned; refusing to reuse or stop its server')
    return {'path': str(report_path), 'sha256': sha(report_path), 'report': report}


def launch_server(root):
    from c4v8_eval_runtime_support import server_payload
    log = (root/'server.log').open('x')
    try:
        process = subprocess.Popen(['/bin/bash',str(STARTER)], stdin=subprocess.PIPE,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,text=True)
    finally: log.close()
    identity = proc(process.pid)
    if not identity or identity['pgid'] != process.pid or identity['session'] != process.pid:
        raise RuntimeError('Failed to establish exclusive SG session')
    atomic_json(root/'server-owned-identity.json',identity)
    (root/'server.pid').write_text(str(process.pid)+'\n')
    process.stdin.write(json.dumps(server_payload(os.getpid())))
    process.stdin.close()
    return process,identity


def verify_server_identity(process, identity, *, require_native=True):
    current = proc(process.pid)
    if (not current or current['start_ticks'] != identity['start_ticks']
            or current['pgid'] != process.pid or current['session'] != process.pid):
        raise RuntimeError('Owned server PID/session identity changed')
    argv = current['argv']
    # Native runtime takes its full ServerArgs through the recorded stdin JSON.
    if require_native and 'c4v8_runtime' not in argv:
        raise RuntimeError('Owned process is not the dedicated numerical runtime')
    if not require_native and not ('c4v8_runtime' in argv or str(STARTER) in argv):
        raise RuntimeError('Owned process cmdline differs from this starter/runtime')
    return current


def wait_ready(client, process, identity, timeout):
    deadline = time.monotonic()+timeout
    last = 'starting'
    while time.monotonic() < deadline:
        if process.poll() is not None: raise RuntimeError('Owned SG exited before readiness')
        try:
            info, _ = client.http_json(URL+'/model_info', timeout=5)
            client.check_server(info, BASE.resolve(), BASE.resolve(), '0')
            verify_server_identity(process, identity)
            return info
        except (OSError, ValueError, TimeoutError) as error:
            last = type(error).__name__+': '+str(error)
        time.sleep(5)
    raise TimeoutError('Owned SG startup readiness timeout: '+last)


def reload_model(client, model, step):
    payload = {'model_path': str(model), 'load_format':'auto', 'weight_version':str(step),
               'abort_all_requests':False, 'is_async':False, 'torch_empty_cache':False,
               'keep_pause':False, 'recapture_cuda_graph':False, 'token_step':step, 'flush_cache':True}
    result, raw = client.http_json(URL+'/update_weights_from_disk', payload, timeout=900)
    if not isinstance(result, dict) or result.get('success') is not True:
        raise RuntimeError('Disk reload failed; no retry: '+raw)
    info, _ = client.http_json(URL+'/model_info', timeout=60)
    client.check_server(info, model, BASE.resolve(), str(step))
    return {'request':payload, 'response':result, 'model_info_after':info}


def run_point_client(root, model, step, report, timeout, *, request_timeout=3600, budget=None):
    timeout = positive_seconds(timeout)
    request_timeout = positive_seconds(request_timeout)
    command = [PYTHON, '-u', str(CLIENT), 'eval', '--server-url', URL,
               '--model-path', str(model), '--step', str(step), '--expected-weight-version',str(step),
               '--tokenizer',str(BASE), '--eval-data',str(DATA), '--output-root',str(root), '--concurrency','4',
               '--timeout',str(request_timeout)]
    if report: command += ['--weight-sha-report',str(report)]
    prefix = root/f'client-step-{step:06d}'
    atomic_json(prefix.with_suffix('.command.json'), command)
    environment = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', USE_TORCH='0', USE_TF='0',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    with prefix.with_suffix('.log').open('x') as log:
        child = None
        try:
            with budget.defer_termination() if budget else nullcontext():
                child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                prefix.with_suffix('.pid').write_text(str(child.pid)+'\n')
            try: code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise TimeoutError(f'Point{step} client timed out; no retry') from None
        finally:
            # Any failure, total deadline or TERM reaps only the CPU child
            # created above. The outer controller then releases its owned SG.
            with budget.defer_termination() if budget else nullcontext():
                if child is not None:
                    if child.poll() is None:
                        child.terminate()
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
                    prefix.with_suffix('.exit').write_text(str(child.returncode)+'\n')
    if code != 0: raise RuntimeError(f'Point{step} client exit{code}; preserved raw evidence, no retry')
    return command


def validate_point(client, root, model, step, rows):
    directory = root/f'step_{step:06d}'
    summary = json.loads((directory/'summary.json').read_text()); validate_summary(summary)
    if (summary.get('valid_curve_point') is not True or summary['optimizer_step'] != step
            or Path(summary['model_path']).resolve() != model or summary['weight_version'] != str(step)):
        raise ValueError('Incomplete or misidentified actual evaluation point')
    point = {key:summary[key] for key in POINT_FIELDS}
    if summary['point_sha256'] != canonical_sha(point): raise ValueError('Point identity hash mismatch')
    protocol = json.loads((directory/'protocol.json').read_text())
    if any(protocol.get(key) != point[key] for key in POINT_FIELDS): raise ValueError('Point protocol metadata mismatch')
    if protocol.get('point_sha256') != summary['point_sha256']: raise ValueError('Protocol point SHA mismatch')
    files = sorted((directory/'questions').glob('*.json'))
    if [path.name for path in files] != [f'{i:02d}.json' for i in range(30)]:
        raise ValueError('Must retain exactly30 complete original per-question raw files')
    with (directory/'results.jsonl').open() as file: aggregate = [json.loads(line) for line in file]
    raw_results = []
    for index, path in enumerate(files):
        raw = json.loads(path.read_text()); row=rows[index]
        if (raw.get('question_index') != index or raw.get('dataset_metadata') != row['metadata']
                or raw.get('messages') != row['messages'] or raw.get('label') != row['label']
                or raw.get('status') != 'success' or raw.get('math_score') not in (0,1)
                or raw.get('optimizer_step') != step or raw.get('model_path') != str(model)
                or raw.get('point_sha256') != summary['point_sha256']
                or raw.get('actual_weight_version') != str(step)
                or raw.get('expected_weight_version') != str(step)):
            raise ValueError('Raw answer identity or actual score evidence is incomplete')
        meta, tokens, cut = client.parse_completion(raw['response_json'], str(step))
        if (raw['response_text'] != raw['response_json']['text'] or raw['output_token_ids'] != tokens
                or raw['completion_tokens'] != len(tokens) or raw['truncated'] != cut
                or json.loads(raw['response_raw_text']) != raw['response_json']):
            raise ValueError('Raw native response differs from recorded completion evidence')
        raw_results.append(raw)
    if aggregate != raw_results or sum(raw['math_score'] for raw in raw_results) != summary['correct']:
        raise ValueError('Summary/aggregate does not equal all30 raw answers')
    return {'optimizer_step':step,'model_path':str(model),'weight_version':str(step),
            'summary_path':str(directory/'summary.json'),'summary_sha256':sha(directory/'summary.json'),
            'raw_question_files':30,'correct':summary['correct'],'questions':30,'accuracy':summary['accuracy'],
            'point_sha256':summary['point_sha256'],
            'truncated':summary['truncated'], 'truncation_rate':summary['truncated']/30,
            'generation_wall_seconds':(max(datetime.datetime.fromisoformat(r['completed_utc']) for r in raw_results)-min(datetime.datetime.fromisoformat(r['started_utc']) for r in raw_results)).total_seconds(),
            'raw_question_sha256':{path.name:sha(path) for path in files},
            'full_weight_sha256_verified_by_report':summary['weight_identity']['full_weight_sha256_verified_by_report']}


def stop_server(root, process, identity):
    verified=False; errors=[]
    try:
        current=proc(process.pid)
        if current and current['state'] != 'Z':
            verify_server_identity(process,identity,require_native=False);verified=True
        else:
            # A dead leader may leave its own descendants in this unique session.
            recorded=json.loads((root/'server-owned-identity.json').read_text())
            if recorded != identity: raise RuntimeError('Saved server ownership evidence changed')
            members=group_members(process.pid)
            if any(m['session'] != process.pid or m['start_ticks'] < identity['start_ticks'] for m in members):
                raise RuntimeError('Unexpected process in owned server group')
            verified=True
        if group_members(process.pid):
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        deadline=time.monotonic()+30
        while group_members(process.pid) and time.monotonic()<deadline: time.sleep(.5)
        if group_members(process.pid):
            # The group/session was created solely by this controller.
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            deadline=time.monotonic()+10
            while group_members(process.pid) and time.monotonic()<deadline:time.sleep(.5)
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:pass
    except Exception as error: errors.append(type(error).__name__+': '+str(error))
    remaining=group_members(process.pid); busy=port_busy()
    report={'passed':verified and not errors and not remaining and not busy,
            'checked_utc':utc(),'owned_server_pid':process.pid,'owned_process_group':process.pid,
            'original_proc_start_ticks':identity['start_ticks'],'verified_owned_cmdline':verified,
            'owned_group_live_pids_after':[item['pid'] for item in remaining],
            'port15115_released':not busy,'errors':errors}
    atomic_json(root/'server-owned-release.json',report)
    return report
