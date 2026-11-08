import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval_support import canonical, sha, full_weight_identity, configure_from_manifest, verify_backend_audit


class EvalGuards(unittest.TestCase):
    def test_identical_indexes_different_weight_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            identities=[]
            for i in range(2):
                model=root/str(i);model.mkdir()
                (model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'w':'one.safetensors'}}))
                (model/'one.safetensors').write_bytes(bytes([i])*128)
                identities.append(full_weight_identity({'index_sha256':sha(model/'model.safetensors.index.json')},model))
            self.assertEqual(identities[0]['index_sha256'],identities[1]['index_sha256'])
            self.assertNotEqual(identities[0]['weights_content_sha256'],identities[1]['weights_content_sha256'])
            with self.assertRaisesRegex(ValueError,'accepted full finite/SHA'):
                full_weight_identity({'shard_sha256_references':identities[0]['full_weight_sha256']},root/'1')

    def test_manifest_attempt_policy_and_source_immutable(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=False):
            root=Path(tmp);policy=root/'policy.json';source=root/'source.py';helper=root/'helper.py'
            policy.write_text('{"query_block":8}');source.write_text('original');helper.write_text('original helper')
            manifest=dict(status='completed',parent_completed_optimizer_steps=0,run_id='run_one',initial_model='/base',
                validation_checkpoint_model_paths={'0':'/base','5':'/run_one/step5'},
                attention_simulation=dict(policy_file=str(policy),policy_sha256=canonical(json.loads(policy.read_text())),
                    source_sha256={str(source):sha(source)}),validation_protocol_helper=str(helper),
                validation_protocol_helper_sha256=sha(helper),validation_protocol_sha256='p'*64)
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            _,first=configure_from_manifest(path)
            manifest['run_id']='run_two';path.write_text(json.dumps(manifest))
            _,second=configure_from_manifest(path)
            self.assertNotEqual(canonical(first),canonical(second))
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'source changed'):configure_from_manifest(path)
            source.write_text('original');policy.write_text('{"query_block":16}')
            with self.assertRaisesRegex(ValueError,'policy changed'):configure_from_manifest(path)

    def test_audit_requires_real_all_ranks_and_modes(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'HBY_C4V8_POLICY_SHA256':'policy'}):
            root=Path(tmp);sources={}
            for event in ['bootstrap_registered','effective_server_args','backend_created','first_extend','first_decode']:
                for rank in range(8):
                    row=dict(policy_sha256='policy',execution_validation_kind='actual_process',event=event,
                        role='scheduler',ranks={'tp_rank':rank},source_sha256={},actual={k:'hby_c4v8_sim' for k in
                        ('attention_backend','prefill_attention_backend','decode_attention_backend')})
                    (root/f'{event}-{rank}.json').write_text(json.dumps(row))
            self.assertTrue(verify_backend_audit(root,sources)['actual_process'])
            (root/'first_decode-7.json').unlink()
            with self.assertRaisesRegex(ValueError,'first_decode'):verify_backend_audit(root,sources)
            path=root/'first_extend-0.json';row=json.loads(path.read_text());row['execution_validation_kind']='cpu_stubbed_sglang'
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'simulated'):verify_backend_audit(root,sources)


if __name__=='__main__':unittest.main()
