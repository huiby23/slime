import json
from pathlib import Path
import tempfile
import unittest

from verify_backend_audit import verify


ROOT=Path(__file__).resolve().parent
PROBE=ROOT/'evidence/spawn-20261108-v2'


class BootstrapAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.parent=json.loads((PROBE/'spawn-validation.json').read_text())['parent_pid']
        self.info={'attention_backend':'hby_c4v8_sim','prefill_attention_backend':'hby_c4v8_sim',
                   'decode_attention_backend':'hby_c4v8_sim','tp_size':2}

    def test_cpu_spawn_records_pass_only_as_cpu_evidence(self):
        result=verify(PROBE/'audit',self.parent,2,allow_cpu_probe=True,server_info=self.info,
                      rehash_sources=False)
        self.assertTrue(result['passed'])
        self.assertEqual(result['evidence_kind'],'cpu_stubbed_sglang')
        with self.assertRaisesRegex(ValueError,'CPU stubbed evidence'):
            verify(PROBE/'audit',self.parent,2,server_info=self.info,rehash_sources=False)

    def test_missing_rank_and_http_backend_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'rank installation'):
            verify(PROBE/'audit',self.parent,8,allow_cpu_probe=True,rehash_sources=False)
        self.info['decode_attention_backend']='ascend'
        with self.assertRaisesRegex(ValueError,'HTTP server_info'):
            verify(PROBE/'audit',self.parent,2,allow_cpu_probe=True,server_info=self.info,
                   rehash_sources=False)

    def test_missing_actual_decode_rejected(self):
        with tempfile.TemporaryDirectory(prefix='audit-negative-',dir=ROOT/'evidence') as temporary:
            temp=Path(temporary)
            for source in (PROBE/'audit').glob('*.json'):
                row=json.loads(source.read_text())
                if row['event']=='first_decode' and row['ranks']['tp_rank']==1:continue
                (temp/source.name).write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'both prefill and decode'):
                verify(temp,self.parent,2,allow_cpu_probe=True,rehash_sources=False)


if __name__=='__main__':unittest.main()
