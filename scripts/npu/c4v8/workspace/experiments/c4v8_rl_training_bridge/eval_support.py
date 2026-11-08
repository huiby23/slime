"""CPU-only provenance, byte identity and real backend evidence for C4V8 eval."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import re
import shutil
import socket
import subprocess
from numerical_profile import implementation_profile, python_paths, digest

ROOT = Path('/root/hby/experiments/c4v8_rl_training_bridge')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        while block := file.read(8*1024*1024): h.update(block)
    return h.hexdigest()


def full_weight_identity(identity, model):
    """Index alone is insufficient: all training steps have the same index."""
    model = Path(model).resolve()
    index = json.loads((model/'model.safetensors.index.json').read_text())
    rows = []
    for name in sorted(set(index['weight_map'].values())):
        if Path(name).name != name: raise ValueError('Unexpected HF shard path')
        path = model/name
        before = path.stat()
        value = sha(path)
        after = path.stat()
        signature = lambda stat:(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if signature(before) != signature(after): raise ValueError('Weight changed while hashing: '+str(path))
        rows.append(dict(file=name, file_bytes=after.st_size, sha256=value))
    identity = dict(identity)
    references = identity.get('shard_sha256_references')
    if references is not None and sorted(references,key=lambda row:row['file']) != rows:
        raise ValueError('Actual weight bytes differ from the accepted full finite/SHA report')
    identity.update(full_weight_sha256_computed=True, full_weight_sha256=rows,
        weights_content_sha256=canonical(rows),
        verification_scope='All actual shard bytes SHA256 plus optional independent finite-value scan report')
    return identity


def configure_from_manifest(path, *, require_completed=True):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if require_completed and manifest.get('status') != 'completed':
        raise ValueError('Evaluation requires a completed training manifest')
    if manifest.get('parent_completed_optimizer_steps') != 0:
        raise ValueError('This C4V8 comparison requires a fresh original-model lineage')
    attention = manifest['attention_simulation']
    policy = Path(attention['policy_file']).resolve()
    if canonical(json.loads(policy.read_text())) != attention['policy_sha256']:
        raise ValueError('Training policy changed')
    inventory = attention['source_sha256']
    for source,value in inventory.items():
        if sha(source) != value: raise ValueError('Audited training source changed: '+source)
    helper = Path(manifest['validation_protocol_helper']).resolve()
    if sha(helper) != manifest['validation_protocol_helper_sha256']:
        raise ValueError('Training evaluation helper changed')
    module=attention.get('core_module','device_attention')
    implementation=attention.get('implementation')
    if implementation is not None:
        current=implementation_profile(Path(manifest['initial_model']).resolve().parent.parent,module)
        if current!=implementation or digest(current)!=attention.get('implementation_sha256'):
            raise ValueError('Selected implementation or toolchain/source identity changed')
        os.environ.update(HBY_C4V8_IMPLEMENTATION_ID=current['implementation_id'],
                          HBY_C4V8_IMPLEMENTATION_SHA256=digest(current))
    elif module!='device_attention':raise ValueError('A fast evaluation requires explicit implementation identity')
    else:
        os.environ.pop('HBY_C4V8_IMPLEMENTATION_ID',None)
        os.environ.pop('HBY_C4V8_IMPLEMENTATION_SHA256',None)
    if module=='fast_attention':os.environ['HBY_C4V8_ALLOW_UNVALIDATED_FUSION']='1'
    else:os.environ.pop('HBY_C4V8_ALLOW_UNVALIDATED_FUSION',None)
    workspace=Path(manifest['initial_model']).resolve().parent.parent
    os.environ['PYTHONPATH']=os.pathsep.join(map(str,python_paths(workspace,module)))+os.pathsep+os.environ.get('PYTHONPATH','')
    identity = dict(training_manifest=str(path), training_manifest_sha256=sha(path),
        run_id=manifest['run_id'], initial_model=manifest['initial_model'],
        checkpoint_paths=manifest['validation_checkpoint_model_paths'],
        policy_sha256=attention['policy_sha256'], source_sha256=inventory,
        core_module=module,implementation=implementation,
        evaluation_protocol_sha256=manifest['validation_protocol_sha256'])
    os.environ.update(HBY_C4V8_CORE_MODULE=module, HBY_C4V8_POLICY_FILE=str(policy),
        HBY_C4V8_POLICY_SHA256=attention['policy_sha256'],
        HBY_C4V8_EVAL_PROTOCOL_MODULE=str(helper),
        HBY_C4V8_EVAL_PROTOCOL_SOURCE_SHA256=manifest['validation_protocol_helper_sha256'],
        HBY_C4V8_EVAL_PROTOCOL_SHA256=manifest['validation_protocol_sha256'],
        HBY_C4V8_EXPERIMENT_SHA256=canonical(identity))
    return manifest, identity


def verify_backend_audit(directory, expected_inventory, *, require_forward=True):
    directory = Path(directory)
    records = [json.loads(p.read_text()) for p in directory.glob('*.json')]
    expected_policy = os.environ['HBY_C4V8_POLICY_SHA256']
    for row in records:
        if row['policy_sha256'] != expected_policy or row.get('execution_validation_kind') != 'actual_process':
            raise ValueError('Backend evidence is stale, simulated or from a different policy')
        if os.environ.get('HBY_C4V8_IMPLEMENTATION_SHA256') and (
            row.get('implementation_sha256')!=os.environ['HBY_C4V8_IMPLEMENTATION_SHA256']
            or row.get('implementation_id')!=os.environ['HBY_C4V8_IMPLEMENTATION_ID']):
            raise ValueError('Backend implementation identity differs from this training run')
        for source,value in row.get('source_sha256',{}).items():
            if expected_inventory.get(source) != value or sha(source) != value:
                raise ValueError('SG runtime source does not match training: '+source)
    events = ['bootstrap_registered','effective_server_args','backend_created']
    if require_forward: events += ['first_extend','first_decode']
    for event in events:
        rows = [r for r in records if r['event']==event and r.get('role')=='scheduler']
        ranks = {r.get('ranks',{}).get('tp_rank') for r in rows}
        if ranks != set(range(8)):raise ValueError('Missing actual TP8 backend audit: '+event)
    settings = [r['actual'] for r in records if r['event']=='effective_server_args']
    for setting in settings:
        if any(setting.get(k)!='hby_c4v8_sim' for k in ('attention_backend','prefill_attention_backend','decode_attention_backend')):
            raise ValueError('SG selected a different attention backend')
    return dict(actual_process=True, expected_tp_ranks=list(range(8)), events=events,
        evidence_sha256={p.name:sha(p) for p in directory.glob('*.json')})


def fresh_resource_gate(path, manifest, proc):
    """Read-only copy of the proven thinking-run idle guard, with C4V8 port.

    Never kills an existing process or imports an accelerator framework.
    Called immediately before creating the new SG session, after resume hashes.
    """
    command = shutil.which('npu-smi') or '/usr/local/Ascend/driver/tools/npu-smi'
    result = subprocess.run([command,'info'],capture_output=True,text=True,timeout=30)
    idle = sorted({int(value) for value in re.findall(r'No running processes found in NPU\s+(\d+)',result.stdout)})
    memory = {line.split(':',1)[0]:int(line.split(':',1)[1].strip().split()[0])*1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    ray = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        item = proc(int(entry.name))
        if not item or item['state']=='Z':continue
        if '/root/hby' in ' '.join(item['argv']) and any(Path(arg).name in ('raylet','gcs_server') for arg in item['argv'][:1]):
            ray.append(item)
    driver = proc(int(manifest['training_driver_pid']))
    driver_alive = bool(driver and driver['state']!='Z')
    with socket.socket() as connection:
        try:connection.bind(('127.0.0.1',15115));port_available=True
        except OSError:port_available=False
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr,idle_cards=idle,all_npu_idle=result.returncode==0 and idle==list(range(8)),
        mem_available_bytes=memory.get('MemAvailable',0),minimum_available_bytes=110*1024**3,
        port_available=port_available,hby_ray_processes=ray,training_driver_alive=driver_alive,
        owned_training_processes_released=not ray and not driver_alive)
    report['passed'] = report['all_npu_idle'] and port_available and not ray and not driver_alive and report['mem_available_bytes']>=report['minimum_available_bytes']
    Path(path).write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise RuntimeError('Resources not released; no process stopped: '+str(path))
    return report
