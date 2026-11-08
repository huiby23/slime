"""CPU fixtures only; no real checkpoint scan, evaluator, Ray or NPU."""
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest

import run_pipeline as p
from plot_curve import validated_rows
sys.path.insert(0,str(Path(__file__).parents[1]/'c4v8_rl_training_bridge'))
from eval_deadlines import EvaluationBudget


def curve_fixture():
    return dict(complete=True,all_point_summaries=[dict(optimizer_step=step,
        valid_curve_point=True,questions=30,successful_requests=30,errors=0,correct=10+step//10,
        truncated=step//20,accuracy=(10+step//10)/30,model_path='/cpu-fixture/'+str(step),
        experiment_identity_sha256='cpu-fixture',protocol_sha256='cpu-fixture') for step in range(0,101,5)])


class CompletionFixtures(unittest.TestCase):
    def test_same_process_rejects_pid_reuse_and_changed_argv(self):
        owner=dict(pid=1,pgid=1,sid=1,startticks=123,argv=['/fixed/script'])
        self.assertTrue(p.same_process(owner,owner))
        self.assertFalse(p.same_process(owner|{'startticks':124},owner))
        self.assertFalse(p.same_process(owner|{'argv':['unrelated']},owner))

    def test_model_report_requires_actual_same_path_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir()
            (model/'one.safetensors').write_bytes(b'CPU fixture, not model weights')
            (model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'w':'one.safetensors'}}))
            r=dict(passed=True,complete_scan=True,nonfinite_count=0,hf_dir=str(model),
                index_sha256=p.sha(model/'model.safetensors.index.json'),shards=[dict(file='one.safetensors',
                file_bytes=(model/'one.safetensors').stat().st_size,sha256=p.sha(model/'one.safetensors'))])
            report=root/'report.json';p.write(report,r)
            self.assertTrue(p.verify_scan_report(report,model)['actual_shard_bytes_rehashed'])
            p.write(report,r|{'hf_dir':str(root/'other')})
            with self.assertRaisesRegex(ValueError,'different model directory'):p.verify_scan_report(report,model)
            p.write(report,r);(model/'one.safetensors').write_bytes(b'x'*(model/'one.safetensors').stat().st_size)
            with self.assertRaisesRegex(ValueError,'Actual shard bytes'):p.verify_scan_report(report,model)

    def test_failed_nonfinite_or_partial_reports_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.json'
            for extra in ({'passed':False},{'complete_scan':False},{'nonfinite_count':1}):
                p.write(path,dict(passed=True,complete_scan=True,nonfinite_count=0)|extra)
                with self.assertRaisesRegex(ValueError,'Incomplete/failed/nonfinite'):p.verify_scan_report(path,Path(tmp))

    def test_source_drift_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.py';path.write_text('before');sources={str(path):p.sha(path)}
            p.verify_sources(sources);path.write_text('after')
            with self.assertRaisesRegex(ValueError,'Source changed'):p.verify_sources(sources)

    def test_full_tensor_equality_allows_layout_but_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            def model(name,order,changed=False):
                root=Path(tmp)/name;root.mkdir();payload=b'';header={'__metadata__':{'fixture':name}}
                for key in order:
                    value=(b'xxxx' if changed and key=='b' else {'a':b'abcd','b':b'efgh'}[key])
                    header[key]=dict(dtype='BF16',shape=[2],data_offsets=[len(payload),len(payload)+len(value)])
                    payload+=value
                encoded=json.dumps(header).encode();shard=root/'fixture.safetensors'
                shard.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
                p.write(root/'model.safetensors.index.json',{'weight_map':{n:shard.name for n in order}})
                return root,{'shard_sha256':{shard.name:p.sha(shard)}}
            first,a=model('first',['a','b']);second,b=model('second',['b','a'])
            r=p.compare_exports(first,second,a,b)
            self.assertTrue(r['passed']);self.assertIn('Every indexed tensor',r['method'])
            bad,c=model('bad',['b','a'],changed=True)
            with self.assertRaisesRegex(ValueError,'full tensor bytes'):p.compare_exports(first,bad,a,c)

    @unittest.skipUnless(sys.platform=='linux','Linux pidfd required')
    def test_pidfd_waits_only_fixed_cpu_process(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.2)'],start_new_session=True)
        try:
            owner=p.proc(child.pid)
            with self.assertRaisesRegex(ValueError,'identity changed'):
                p.wait_controller(owner|{'startticks':owner['startticks']+1})
            self.assertEqual(p.wait_controller(owner),'pidfd_exit_observed')
            self.assertEqual(child.wait(timeout=2),0)
        finally:
            if child.poll() is None:child.kill();child.wait()

    @unittest.skipUnless(Path('/proc/self/stat').exists(),'Linux process identities required')
    def test_stage_timeout_records_exit_and_reaps_own_cpu_child(self):
        with tempfile.TemporaryDirectory() as tmp,EvaluationBudget(None) as budget:
            root=Path(tmp);stages=p.Stages(root,budget)
            with self.assertRaises(subprocess.TimeoutExpired):
                stages.run('timeout',[sys.executable,'-c','import time;time.sleep(30)'],timeout=.1)
            record=p.read(root/'timeout/stage.json')
            self.assertEqual(record['status'],'failed');self.assertLess(record['exit_code'],0)
            with self.assertRaises(ProcessLookupError):os.kill(record['owner']['pid'],0)

    def test_curve_requires21_real_same_identity_points(self):
        curve=curve_fixture();self.assertEqual(len(validated_rows(curve)),21)
        for changed in (dict(complete=False),dict(all_point_summaries=curve['all_point_summaries'][:-1])):
            with self.assertRaises(ValueError):validated_rows(curve|changed)
        curve['all_point_summaries'][1]['experiment_identity_sha256']='other-run'
        with self.assertRaisesRegex(ValueError,'Different experiments'):validated_rows(curve)

    def test_curve_rejects_wrong_denominator_score_and_errors(self):
        for change in ({'questions':29},{'successful_requests':29},{'errors':1},{'accuracy':float('nan')},{'accuracy':.99}):
            curve=curve_fixture();curve['all_point_summaries'][0].update(change)
            with self.assertRaisesRegex(ValueError,'Invalid actual30'):validated_rows(curve)

    def test_raw_truncation_aggregate_rejects_wrong_count_and_nonbool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'step_000005';questions=directory/'questions';questions.mkdir(parents=True)
            for i in range(30):p.write(questions/f'{i:02d}.json',{'truncated':i<3})
            p.write(directory/'summary.json',{'truncated':3})
            self.assertEqual(p.verify_truncation_aggregate(root,5)['truncated'],3)
            p.write(directory/'summary.json',{'truncated':4})
            with self.assertRaisesRegex(ValueError,'Summary truncation count'):p.verify_truncation_aggregate(root,5)
            p.write(directory/'summary.json',{'truncated':3});p.write(questions/'00.json',{'truncated':1})
            with self.assertRaisesRegex(ValueError,'must be bool'):p.verify_truncation_aggregate(root,5)


if __name__=='__main__':unittest.main()
