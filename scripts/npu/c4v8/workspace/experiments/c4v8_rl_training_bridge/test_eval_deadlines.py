"""CPU process/argument tests; no SG, torch, Ray, model, or NPU is started."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from eval_deadlines import EvaluationBudget

HERE = Path(__file__).parent


def atomic_json(path, value):
    Path(path).write_text(json.dumps(value))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluationDeadlines(unittest.TestCase):
    def setUp(self):
        self.protocol = types.ModuleType('c4v8_eval_protocol')
        self.protocol.STEPS = tuple(range(0,101,5))
        self.protocol.PROTOCOL_SHA256 = 'protocol'
        self.protocol.atomic_json = atomic_json
        self.protocol.canonical_sha = lambda value: 'identity'
        self.protocol.validate_summary = lambda value: None
        self.protocol.rebuild_curve = lambda root: {'complete':False}
        self.protocol.build_protocol = lambda: ({'test':'CPU'},'protocol')
        self.modules = patch.dict(sys.modules,{'c4v8_eval_protocol':self.protocol})
        self.modules.start()
        self.inherited = load('deadline_test_inherited',HERE/'c4v8_eval_inherited.py')
        self.trajectory = load('deadline_test_trajectory',HERE/'eval-c4v8-trajectory.py')
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.client = self.root/'cpu-client.py'
        self.client.write_text('import time\ntime.sleep(30)\n')
        self.inherited.PYTHON = sys.executable
        self.inherited.CLIENT = self.client

    def tearDown(self):
        self.modules.stop()
        self.tmp.cleanup()

    def test_parser_compatible_defaults_and_explicit_limits(self):
        args=self.trajectory.parse_args(['--manifest','/test','--validate-artifacts'])
        self.assertEqual((args.request_timeout,args.client_timeout,args.total_timeout_seconds),(3600,14400,None))
        args=self.trajectory.parse_args(['--manifest','/test','--validate-artifacts',
            '--request-timeout','10800','--client-timeout','43200','--total-timeout-seconds','604800'])
        self.assertEqual((args.request_timeout,args.client_timeout,args.total_timeout_seconds),(10800,43200,604800))
        for value in ('0','-1','nan','inf'):
            with self.assertRaises(SystemExit):
                self.trajectory.parse_args(['--manifest','/test','--validate-artifacts','--request-timeout',value])

    def test_request_timeout_passed_to_actual_child_argv(self):
        self.client.write_text('import sys,json\nprint(json.dumps(sys.argv[1:]))\n')
        command=self.inherited.run_point_client(self.root,self.root,0,None,10,request_timeout=10800)
        actual=json.loads((self.root/'client-step-000000.log').read_text())
        self.assertEqual(actual,command[3:])
        self.assertEqual(actual[actual.index('--timeout')+1],'10800.0')
        self.assertEqual((self.root/'client-step-000000.exit').read_text().strip(),'0')

    def test_point_deadline_reaps_only_cpu_child(self):
        with self.assertRaisesRegex(TimeoutError,'Point0 client timed out'):
            self.inherited.run_point_client(self.root,self.root,0,None,.1)
        pid=int((self.root/'client-step-000000.pid').read_text())
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)
        self.assertNotEqual((self.root/'client-step-000000.exit').read_text().strip(),'0')

    def test_total_deadline_during_child_wait_reaps_cpu_child(self):
        with self.assertRaisesRegex(TimeoutError,'Total 21-point'):
            with EvaluationBudget(.1) as budget:
                self.inherited.run_point_client(self.root,self.root,0,None,30,budget=budget)
        pid=int((self.root/'client-step-000000.pid').read_text())
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_pending_signal_delivered_after_child_handle_assignment(self):
        child=None
        try:
            with self.assertRaisesRegex(TimeoutError,'Total 21-point'):
                with EvaluationBudget(.02) as budget:
                    with budget.defer_termination():
                        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
                        time.sleep(.04)
            self.assertIsNotNone(child)
            # The parent's deferral must not block SIGTERM in its child.
            child.terminate();self.assertEqual(child.wait(timeout=2),-signal.SIGTERM)
        finally:
            if child is not None and child.poll() is None:child.kill();child.wait()

    @unittest.skipUnless(Path('/proc/self/stat').is_file(),'Real owned session evidence requires Linux /proc')
    def test_actual_trajectory_deadline_releases_owned_cpu_server(self):
        self._actual_trajectory_interrupt('deadline')

    @unittest.skipUnless(Path('/proc/self/stat').is_file(),'Real owned session evidence requires Linux /proc')
    def test_actual_trajectory_term_releases_owned_cpu_server(self):
        self._actual_trajectory_interrupt('term')

    def _actual_trajectory_interrupt(self, mode):
        manifest_path=self.root/'manifest.json'
        manifest_path.write_text('{}')
        manifest={'validation_protocol':{'test':'CPU'},'validation_protocol_sha256':'protocol',
                  'attention_simulation':{'source_sha256':{}},'run_id':'cpu_fixture',
                  'validation_trajectory_directory':str(self.root/'curves')}
        client=types.SimpleNamespace(load_data=lambda _:[],weight_identity=lambda *_:{})
        def launch_cpu(root):
            process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)','c4v8_runtime'],
                start_new_session=True)
            identity=self.inherited.proc(process.pid)
            atomic_json(root/'server-owned-identity.json',identity)
            return process,identity
        runtime=types.ModuleType('c4v8_eval_runtime_support')
        runtime.server_payload=lambda _:{}
        args=self.trajectory.parse_args(['--manifest',str(manifest_path),
            '--base-sha-report',str(manifest_path),'--final-sha-report',str(manifest_path),
            '--request-timeout','10800','--client-timeout','30'])
        # The production orchestration and production stop_server run unchanged.
        # Only the SG service is substituted by an owned CPU sleeper.
        def wait_ready(*_):
            if mode=='term':os.kill(os.getpid(),signal.SIGTERM)
            return {}
        with patch.dict(sys.modules,{'c4v8_eval_inherited':self.inherited,'c4v8_eval_runtime_support':runtime}), \
             patch.object(self.trajectory,'configure_from_manifest',return_value=(manifest,{'test':'fixture'})), \
             patch.object(self.trajectory,'fresh_resource_gate',return_value={'cpu_fixture':True}), \
             patch.object(self.trajectory,'verify_backend_audit',return_value={}), \
             patch.object(self.inherited,'plan_models',return_value=(manifest,{0:self.root})), \
             patch.object(self.inherited,'client_module',return_value=client), \
             patch.object(self.inherited,'launch_server',side_effect=launch_cpu), \
             patch.object(self.inherited,'wait_ready',side_effect=wait_ready), \
             patch.object(self.inherited,'port_busy',return_value=False):
            with EvaluationBudget(.2 if mode=='deadline' else 30) as budget:
                code=self.trajectory.evaluate(args,budget)
        self.assertEqual(code,1)
        attempt=next((self.root/'curves/cpu_fixture').glob('execution_*'))
        report=json.loads((attempt/'trajectory-summary.json').read_text())
        self.assertFalse(report['complete'])
        self.assertEqual(report['error']['type'],'TimeoutError' if mode=='deadline' else 'InterruptedError')
        self.assertTrue(report['owned_server_release']['passed'])
        self.assertEqual(report['owned_server_release']['owned_group_live_pids_after'],[])
        self.assertEqual((attempt/'trajectory.exit').read_text().strip(),'1')
        pid=report['owned_server_release']['owned_server_pid']
        with self.assertRaises(ProcessLookupError):os.kill(pid,0)


if __name__=='__main__':unittest.main()
