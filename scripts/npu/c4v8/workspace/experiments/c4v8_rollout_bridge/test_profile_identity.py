import importlib.util
import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from c4v8_bootstrap import _load_profile,_digest

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent.parent
spec=importlib.util.spec_from_file_location('identity_test_helper',ROOT.parent/'c4v8_rl_training_bridge/numerical_profile.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


class ProfileIdentityTests(unittest.TestCase):
    def test_fast_profile_requires_optin_identity_and_matches_shared_helper(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'evidence',prefix='profile-test-') as temporary:
            path=Path(temporary);policy={'query_block':64,'key_block':128,'vector':'mxfp8',
                'quantize_operands':True,'causal':True,'reduction':'mxfp8_tree_reduction'}
            (path/'policy.json').write_text(json.dumps(policy))
            identity=helper.implementation_profile(WORKSPACE,'fast_attention')
            env={'HBY_C4V8_CORE_MODULE':'fast_attention','HBY_C4V8_POLICY_FILE':str(path/'policy.json'),
                'HBY_C4V8_ROLLOUT_AUDIT_DIR':str(path/'audit'),'HBY_C4V8_POLICY_SHA256':_digest(policy)}
            with patch.dict(os.environ,env),patch.object(sys,'path',
                [str(ROOT.parent/'c4v8_fast_attention'),str(ROOT.parent/'c4v8_device_core')]+sys.path):
                for name in ('HBY_C4V8_IMPLEMENTATION_ID','HBY_C4V8_IMPLEMENTATION_SHA256',
                             'HBY_C4V8_ALLOW_UNVALIDATED_FUSION','HBY_C4V8_EXPECTED_RUNTIME_SHA256'):
                    os.environ.pop(name,None)
                with self.assertRaisesRegex(ValueError,'explicit implementation'):_load_profile()
                os.environ.update(HBY_C4V8_IMPLEMENTATION_ID=identity['implementation_id'],
                    HBY_C4V8_IMPLEMENTATION_SHA256=helper.digest(identity),HBY_C4V8_ALLOW_UNVALIDATED_FUSION='1')
                result=_load_profile()
                self.assertEqual(result['implementation_sha256'],helper.digest(identity))
                self.assertEqual(result['core_module'],'fast_attention')
                self.assertTrue(any(p.endswith('/whole_decode.py') for p in result['source_sha256']))
                os.environ['HBY_C4V8_IMPLEMENTATION_SHA256']='0'*64
                with self.assertRaisesRegex(ValueError,'source/toolchain identity'):_load_profile()
                self.assertNotIn('torch_npu',sys.modules)

    def test_unknown_core_is_never_imported(self):
        with patch.dict(os.environ,{'HBY_C4V8_CORE_MODULE':'unreviewed_core'}):
            with self.assertRaisesRegex(ValueError,'reviewed core'):_load_profile()


if __name__=='__main__':unittest.main()
