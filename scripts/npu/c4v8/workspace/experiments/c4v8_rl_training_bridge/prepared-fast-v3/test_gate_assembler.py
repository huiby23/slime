"""CPU negative fixtures; these are never accepted as real NPU evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('assembler',Path(__file__).with_name('assemble-launch-gate.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


class NegativeEvidence(unittest.TestCase):
    def test_inventory_uses_actual_prepared_launch_consumer(self):
        expected=(g.PREPARED/'recipe_support.py').resolve()
        self.assertEqual(Path(g.consumer.__file__).resolve(),expected)
        inventory,_=g.source_inventory(g.WORKSPACE,g.PREPARED/'qwen30b_dapo_c4v8.py','fast_attention')
        self.assertEqual(inventory[str(expected)],g.hash_file(expected))
        self.assertIn(str((g.BRIDGE/'recipe_support.py').resolve()),inventory)

    def test_policy_and_source_identity_mismatch_rejected(self):
        profile={'source_sha256':{'fixture':'a'}};policy={'query_block':1024}
        e=g.Evidence(profile,policy)
        exact={'implementation':profile,'implementation_sha256':g.digest(profile),'policy':policy,'policy_sha256':g.digest(policy)}
        e.full_identity(exact)
        for changed in ({'policy':{'query_block':64}}, {'implementation_sha256':'0'*64},
                        {'implementation':{'source_sha256':{'fixture':'b'}}}):
            with self.assertRaises(ValueError):e.full_identity(exact|changed)

    def test_shared_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(g,'WORKSPACE',Path(tmp)):
            p=Path(tmp)/'experiments/core.py';p.parent.mkdir();p.write_text('initial')
            hashes={str(p):g.hash_file(p)};e=g.Evidence({'source_sha256':hashes},{})
            e.source_subset(hashes,['core.py']);p.write_text('different')
            with self.assertRaisesRegex(ValueError,'Source mismatch'):e.source_subset(hashes,['core.py'])

    def owned_fixture(self,root,*,status='exited',fallback=False,last_card=7):
        owner=dict(pid=42,pgrp=42,sid=42,argv=['explicit_cpu_fixture','--owned-child'])
        outer='explicit CPU fixture only\n'+('npu_cpu_fallback unexpected\n' if fallback else '')
        (root/'outer.log').write_text(outer)
        result=dict(status=status,child_exit_code=0,execution_completed_cleanly=True,after_all_eight_idle=True,
            tensor_cpu_fallback_warnings=[],owner=owner,outer_log_sha256=g.hash_file(root/'outer.log'))
        (root/'result.json').write_text(json.dumps(result));(root/'owned-process.json').write_text(json.dumps(owner))
        (root/'npu-after.json').write_text(json.dumps(dict(all_eight_idle=True,
            stdout='\n'.join('No running processes found in NPU '+str(i) for i in range(last_card+1)))))

    def test_failed_owned_report_not_rescued_by_passed_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.owned_fixture(root,status='failed')
            with self.assertRaisesRegex(ValueError,'Owned controller failed'):g.Evidence({},{}).owned(root)

    def test_log_and_raw_release_rescanned(self):
        for flags in ({'fallback':True},{'last_card':6}):
            with self.subTest(flags=flags),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);self.owned_fixture(root,**flags)
                with self.assertRaises(ValueError):g.Evidence({},{}).owned(root)

    def test_signed_zero_does_not_allow_arbitrary_allclose_or_nonfinite(self):
        g.bit_comparison(dict(finite=True,bit_exact=False,signed_zero_only=True,numeric_equal=True))
        for bad in (dict(finite=True,bit_exact=False,signed_zero_only=False,numeric_equal=False),
                    dict(finite=False,bit_exact=True),dict(finite=True,bit_exact=False,signed_zero_only=True,numeric_equal=False)):
            with self.assertRaises(ValueError):g.bit_comparison(bad)

    def test_nonfinite_json_number_rejected(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):g.finite_numbers({'gradient':[float('nan')]})

    def test_source_change_between_checks_not_silently_refreshed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.py';p.write_text('initial');e=g.Evidence({},{});e.text(p)
            p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed during acceptance'):e.text(p)

    def test_historical_probe_requires_exact_saved_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            current=Path(tmp)/'probe.py';saved=Path(tmp)/'saved-probe.py'
            current.write_text('new harness');saved.write_text('old harness');old=g.hash_file(saved)
            g.Evidence({},{}).recorded_probe_source(str(current),old,saved)
            saved.write_text('not the original harness')
            with self.assertRaisesRegex(ValueError,'snapshot SHA mismatch'):
                g.Evidence({},{}).recorded_probe_source(str(current),old,saved)


if __name__=='__main__':unittest.main()
