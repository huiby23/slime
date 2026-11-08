"""Pure CPU preparation/audit checks; fixtures are never device evidence."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('actual_sg_smoke',ROOT/'experiments/c4v8-integration-probes/sg_smoke.py')
sg=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(sg)
POLICY=dict(query_block=64,key_block=128,vector='mxfp8',causal=True,quantize_operands=True,reduction='mxfp8_tree_reduction')

class ProbePreparationTests(unittest.TestCase):
    def config(self,root,module='fast_attention'):
        policy=root/'policy.json';policy.write_text(json.dumps(POLICY))
        return sg.build_configuration(ROOT,root/'output',policy,module,0)

    def test_actual_builder_is_stdlib_only_and_sets_fast_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);env,server,report=self.config(root)
            self.assertEqual(env['HBY_C4V8_CORE_MODULE'],'fast_attention')
            self.assertEqual(env['HBY_C4V8_IMPLEMENTATION_SHA256'],report['implementation_sha256'])
            self.assertEqual(env['HBY_C4V8_ALLOW_UNVALIDATED_FUSION'],'1')
            self.assertEqual(server['attention_backend'],'hby_c4v8_sim')
            self.assertEqual(server['tp_size'],1)
            for name in ('c4v8_fast_attention','c4v8_fused_nvfp4','c4v8_fused_qdq'):
                self.assertIn(name,env['PYTHONPATH'])
            self.assertFalse(report['npu_process_started'])
            self.assertIn('not routed',report['implementation']['arithmetic']['whole_decode_candidate'])
            code=f"import importlib.util,sys;from pathlib import Path;s=importlib.util.spec_from_file_location('probe',{str(Path(sg.__file__))!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.build_configuration({str(ROOT)!r},{str(root/'output')!r},{str(root/'policy.json')!r},'fast_attention',0);assert not any(n in sys.modules for n in ('torch','torch_npu','triton','fast_attention'));print('NO_NUMERICAL_IMPORT')"
            self.assertIn('NO_NUMERICAL_IMPORT',subprocess.check_output([sys.executable,'-c',code],text=True))

    def test_portable_clears_stale_fusion_flag_and_invalid_policy_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'HBY_C4V8_ALLOW_UNVALIDATED_FUSION':'1','HBY_C4V8_CPU_SPAWN_PROBE':'1'}):
            root=Path(tmp);env,_,_=self.config(root,'device_attention')
            self.assertNotIn('HBY_C4V8_ALLOW_UNVALIDATED_FUSION',env)
            self.assertNotIn('HBY_C4V8_CPU_SPAWN_PROBE',env)
            policy=POLICY|{'query_block':8};(root/'policy.json').write_text(json.dumps(policy))
            with self.assertRaisesRegex(ValueError,'query64'):
                sg.build_configuration(ROOT,root/'out',root/'policy.json','fast_attention',0)

    def test_source_policy_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,report=self.config(root)
            sg.check_frozen_sources(report,root/'policy.json',ROOT)
            (root/'policy.json').write_text(json.dumps(POLICY)+' ')
            with self.assertRaisesRegex(ValueError,'Policy changed'):
                sg.check_frozen_sources(report,root/'policy.json',ROOT)

    def test_process_identity_gate_rejects_foreign_reused_or_shared_group(self):
        owner=dict(pid=44,pgrp=44,sid=44,startticks=99,cmdline='python -m c4v8_runtime')
        sg.verify_owner(owner,owner)
        for changed in ({'startticks':100},{'pgrp':45},{'sid':45},{'cmdline':'other_training.py'}):
            with self.assertRaises(RuntimeError):sg.verify_owner(owner,owner|changed)

    def test_actual_role_audit_requires_hf32_readback_and_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,report=self.config(root);audit=root/'audit';audit.mkdir()
            common={k:report[k] for k in ('policy_sha256','implementation_sha256','runtime_sha256','core_module')}
            common.update(implementation_id=report['implementation']['implementation_id'],execution_validation_kind='actual_process',pid=123,ranks={'tp_rank':0})
            records=[]
            for role in ('http_parent','scheduler'):
                records += [common|dict(event='bootstrap_registered',role=role,source_sha256=report['runtime_source_sha256'],fp32_simulator_mode={'applied':True,'actual_allow_hf32':False}),
                            common|dict(event='effective_server_args',role=role,actual={k:sg.BACKEND for k in ('attention_backend','prefill_attention_backend','decode_attention_backend')}|dict(model_path=report['model'],tp_size=1,dtype='bfloat16',kv_cache_dtype='auto'))]
            records += [common|dict(event=e,role='scheduler') for e in ('backend_created','first_extend','first_decode')]
            for i,row in enumerate(records):sg.write(audit/f'{i}.json',row)
            self.assertTrue(sg.verify_audit(audit,report)['actual_prefill_and_decode'])
            sg.write(audit/'0.json',records[0]|{'fp32_simulator_mode':{'applied':True,'actual_allow_hf32':True}})
            with self.assertRaisesRegex(ValueError,'HF32'):sg.verify_audit(audit,report)
            sg.write(audit/'0.json',records[0]);(audit/'6.json').unlink()
            with self.assertRaisesRegex(ValueError,'first_decode'):sg.verify_audit(audit,report)

    def test_tiny_file_digest_remains_bytes_not_profile_json(self):
        path=Path(__file__).with_name('probe_megatron_tiny.py')
        module_spec=importlib.util.spec_from_file_location('tiny_preparation',path)
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        self.assertEqual(module.digest(path),sg.sha(path))
        tree=ast.parse(path.read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        imports=[a for n in ast.walk(main) if isinstance(n,ast.ImportFrom) and n.module=='numerical_profile' for a in n.names if a.name=='digest']
        self.assertEqual([a.asname for a in imports],['profile_digest'])

    def test_long_generation_requires_actual_129_to_240_prompt_tokens(self):
        good=dict(text='ok',meta_info={'prompt_tokens':140,'completion_tokens':2})
        self.assertEqual(sg.check_generation(good,expected_prompt_tokens=140)['prompt_tokens'],140)
        for count in (None,128,141,241):
            bad=good|{'meta_info':{'prompt_tokens':count,'completion_tokens':2}}
            with self.assertRaisesRegex(ValueError,'128-key'):
                sg.check_generation(bad,expected_prompt_tokens=140)
        with self.assertRaisesRegex(ValueError,'two genuine'):
            sg.check_generation(good|{'meta_info':{'prompt_tokens':140,'completion_tokens':1}},expected_prompt_tokens=140)

    def test_long_prompt_fixture_hash_and_length_guards(self):
        # Clearly synthetic CPU fixture checks validation only; no tokenizer or NPU claim.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tokenizer=root/'tokenizer.json';tokenizer.write_text('CPU validation fixture')
            fixture=dict(kind='offline_real_tokenizer_request_fixture',model_path=str(root),
                input_ids=list(range(140)),prompt_tokens=140,vocabulary_size=200,
                tokenizer_source_sha256={str(tokenizer):sg.sha(tokenizer)})
            path=root/'fixture.json';sg.write(path,fixture)
            self.assertEqual(sg.load_long_prompt(path,str(root))['prompt_tokens'],140)
            tokenizer.write_text('changed fixture')
            with self.assertRaisesRegex(ValueError,'source changed'):sg.load_long_prompt(path,str(root))

    def test_actual_tiny_packed_constructor_uses_longest_document(self):
        from types import SimpleNamespace as NS
        tree=ast.parse(Path(__file__).with_name('probe_megatron_tiny.py').read_text())
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='packed' for t in n.targets))
        constructor=compile(ast.Expression(assignment.value),'<actual PackedSeqParams constructor>','eval')
        for total,split in ((128,97),(129,97),(16,8)):
            packed=eval(constructor,{'PackedSeqParams':NS,'cu':[0,split,total],
                                    'options':NS(sequence_length=total),'split':split})
            self.assertEqual(packed.max_seqlen_q,max(split,total-split))
            self.assertEqual(packed.max_seqlen_kv,packed.max_seqlen_q)

    def test_actual_sg_request_order_starts_receive_loop_before_info(self):
        tree=ast.parse(Path(sg.__file__).read_text())
        calls={}
        for node in ast.walk(tree):
            if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='fetch'
                    and len(node.args)>1 and isinstance(node.args[1],ast.Constant)):
                calls[node.args[1].value]=node.lineno
        self.assertLess(calls['/health'],calls['/generate'])
        self.assertLess(calls['/generate'],calls['/server_info'])
        self.assertLess(calls['/server_info'],calls['/get_server_info'])

if __name__=='__main__':unittest.main()
