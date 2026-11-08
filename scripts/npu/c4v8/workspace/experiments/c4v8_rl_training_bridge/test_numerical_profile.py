"""Source/implementation identity is explicit and independent of math policy."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numerical_profile as profile
from recipe_support import policy_digest
from eval_support import configure_from_manifest,sha

WORKSPACE=Path(__file__).resolve().parents[2]


class NumericalProfileTests(unittest.TestCase):
    def test_fast_sources_cover_actual_custom_dependencies_without_imports(self):
        report=profile.implementation_profile(WORKSPACE,'fast_attention')
        paths=set(Path(p).name for p in report['source_sha256'])
        self.assertEqual(paths,{'fast_attention.py','device_attention.py','device_quant.py','exact_fp32.py',
            'nvfp4_triton.py','forward_api.py','fused_mxfp8.py','whole_decode.py','numerical_profile.py','staged_decode.py','staged_kernels.py'})
        self.assertEqual(report['implementation_id'],'fast_attention_fused_forward_ste_staged_decode_v2')
        output=subprocess.check_output([sys.executable,'-c',
            'import sys;from numerical_profile import implementation_profile;'
            f"implementation_profile({str(WORKSPACE)!r},'fast_attention');"
            "assert not any(n in sys.modules for n in ['torch','torch_npu','triton','fast_attention']);print('CPU_STDLIB_ONLY')"],
            cwd=Path(__file__).parent,text=True)
        self.assertIn('CPU_STDLIB_ONLY',output)

    def test_policy_and_implementation_identities_are_separate(self):
        math={'query_block':64,'key_block':128,'vector':'mxfp8','quantize_operands':True,
              'causal':True,'reduction':'mxfp8_tree_reduction'}
        old=profile.implementation_profile(WORKSPACE,'device_attention')
        fast=profile.implementation_profile(WORKSPACE,'fast_attention')
        self.assertNotEqual(profile.digest(old),profile.digest(fast))
        self.assertEqual(profile.digest(math),policy_digest(math))
        with self.assertRaises(ValueError):profile.implementation_profile(WORKSPACE,'automatic_fallback')

    def test_eval_restores_fast_choice_and_rejects_stale_implementation(self):
        implementation=profile.implementation_profile(WORKSPACE,'fast_attention')
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=False):
            root=Path(tmp);policy=root/'policy.json';policy.write_text('{"query_block":64,"key_block":128}')
            helper=root/'helper.py';helper.write_text('cpu protocol fixture')
            manifest=dict(status='prepared',parent_completed_optimizer_steps=0,run_id='fast-source-test',
                initial_model=str(WORKSPACE/'models/Qwen3-30B-A3B'),validation_checkpoint_model_paths={'0':'original'},
                attention_simulation=dict(core_module='fast_attention',implementation=implementation,
                    implementation_sha256=profile.digest(implementation),source_sha256=implementation['source_sha256'],
                    policy_file=str(policy),policy_sha256=policy_digest(json.loads(policy.read_text()))),
                validation_protocol_helper=str(helper),validation_protocol_helper_sha256=sha(helper),
                validation_protocol_sha256='c'*64)
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            configure_from_manifest(path,require_completed=False)
            self.assertEqual(os.environ['HBY_C4V8_CORE_MODULE'],'fast_attention')
            self.assertEqual(os.environ['HBY_C4V8_ALLOW_UNVALIDATED_FUSION'],'1')
            self.assertEqual(os.environ['HBY_C4V8_IMPLEMENTATION_SHA256'],profile.digest(implementation))
            manifest['attention_simulation']['implementation_sha256']='0'*64
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'implementation'):configure_from_manifest(path,require_completed=False)


if __name__=='__main__':unittest.main()
