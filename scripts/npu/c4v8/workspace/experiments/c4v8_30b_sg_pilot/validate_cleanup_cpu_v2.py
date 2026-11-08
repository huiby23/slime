#!/usr/bin/env python3
"""CPU-only real process tests for pilot cleanup; no SG/model/NPU."""
import hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pilot',HERE/'pilot.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
OUT=HERE/'evidence/cleanup-cpu-v2';OUT.mkdir(parents=True,exist_ok=False)
original_proc=pilot.smoke.proc
rows=[]
dummy_code="""import os,signal,time,sys
from pathlib import Path
ready=Path(sys.argv[1]);mode=sys.argv[2]
if mode=='ignore':signal.signal(signal.SIGTERM,signal.SIG_IGN)
ready.write_text('ready')
while True:
 if mode=='exec' and ready.with_suffix('.exec').exists():os.execv('/bin/sleep',['sleep','60'])
 time.sleep(.005)
"""
def spawn(name,mode='normal'):
    out=OUT/name;out.mkdir()
    ready=out/'ready'
    child=subprocess.Popen([sys.executable,'-c',dummy_code,str(ready),mode,'c4v8_runtime_cpu_cleanup_dummy'],start_new_session=True)
    until=time.monotonic()+3
    while not ready.exists():
        if time.monotonic()>until:raise TimeoutError('fixture startup')
        time.sleep(.005)
    return child,original_proc(child.pid),out,ready
def zombie(child):
    os.kill(child.pid,signal.SIGKILL)
    until=time.monotonic()+2
    while (Path('/proc')/str(child.pid)/'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z':
        if time.monotonic()>until:raise TimeoutError('fixture zombie')
        time.sleep(.001)
def final(child):
    if child.poll() is None:child.kill()
    child.wait(timeout=2)
def run(name,mode='normal',injection=None,expect_refusal=False):
    child,owner,out,ready=spawn(name,mode);calls=0
    try:
        if injection=='reaped':child.terminate();child.wait(timeout=2)
        if injection=='zombie_before':zombie(child)
        if injection=='exec':
            ready.with_suffix('.exec').write_text('exec')
            until=time.monotonic()+2
            while 'c4v8_runtime' in original_proc(child.pid)['cmdline']:
                if time.monotonic()>until:raise TimeoutError('fixture exec')
                time.sleep(.001)
        def proc(pid):
            nonlocal calls
            calls+=1
            if injection in ('zombie_during_proc','gone_during_proc') and calls==1:
                zombie(child)
                if injection=='gone_during_proc':child.wait(timeout=2)
            if injection=='zombie_after_term_timeout' and calls==2:zombie(child)
            return original_proc(pid)
        pilot.smoke.proc=proc
        failed=None
        try:result=pilot.stop_owned_budgeted(child,owner,out,trigger='cpu_real_process_test')
        except RuntimeError as e:
            failed=str(e);result=json.loads((out/'cleanup.json').read_text())
        if expect_refusal:
            assert failed and 'command identity changed' in failed
            assert child.poll() is None
            assert result['actions']==[]
        else:
            assert failed is None,failed
            assert child.returncode is not None
            assert result['elapsed_seconds']<8.5
            if injection in ('zombie_during_proc','gone_during_proc','zombie_after_term_timeout'):
                assert result.get('exit_races'),result
                assert not any(a['signal']=='SIGKILL' for a in result['actions'])
            if mode=='ignore' and injection is None:
                assert [a['signal'] for a in result['actions']]==['SIGTERM','SIGKILL']
        rows.append(dict(name=name,passed=True,expected_live_identity_refusal=expect_refusal,result=result))
    finally:
        pilot.smoke.proc=original_proc;final(child)
for args in [
 ('normal_term',),
 ('stubborn_term_then_kill','ignore'),
 ('already_reaped','normal','reaped'),
 ('already_zombie','normal','zombie_before'),
 ('zombie_between_poll_and_snapshot','ignore','zombie_during_proc'),
 ('gone_between_poll_and_snapshot','ignore','gone_during_proc'),
 ('zombie_after_term_grace','ignore','zombie_after_term_timeout'),
 ('altered_live_command_rejected','exec','exec',True),
]:run(*args)
assert not any(k=='torch' or k.startswith('torch.') or k=='torch_npu' for k in sys.modules)
report=dict(passed=True,cpu_real_process_tests=True,npu_used=False,tests=rows,
 pilot_sha256=hashlib.sha256((HERE/'pilot.py').read_bytes()).hexdigest(),
 historical_failure='suspected exit race; original run did not record failed-snapshot state; preserved unchanged')
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(passed=True,tests=len(rows),pilot_sha256=report['pilot_sha256'],report=str(OUT/'report.json'))))
