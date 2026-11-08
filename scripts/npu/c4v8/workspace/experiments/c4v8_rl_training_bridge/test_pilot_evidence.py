"""CPU artifact fixtures only; no real optimizer/model update claims."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from pilot_evidence import read_progress

class PilotEvidenceTests(unittest.TestCase):
    def records(self,root,count=5):
        for i in range(count):torch.save(float(i+1)/10,root/f'grad_norm_actor_{i//20}_{i%20}.pt')
    def checkpoint(self,root,completed=5,complete=True,step=5):
        path=root/f'optimizer-step-checkpoints/checkpoint_step_{step:06d}';path.mkdir(parents=True)
        (path/'optimizer-step-checkpoint.json').write_text(json.dumps(dict(completed_optimizer_steps=completed,complete=complete,hf_only=True,parent_completed_optimizer_steps=0,new_successful_optimizer_steps=completed)))
        (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'fixture':'fixture.safetensors'}}))
        (path/'fixture.safetensors').write_bytes(b'not a real model; metadata test fixture')
    def test_first_and_fifth_update_are_observations_not_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.records(root,1)
            first=read_progress(root,minimum_file_age=0)
            self.assertTrue(first['first_gradient_observed']);self.assertFalse(first['pilot_passed']);self.assertEqual(first['successful_updates_proven_by_complete_hf'],0)
            self.records(root);self.checkpoint(root)
            fifth=read_progress(root,minimum_file_age=0)
            self.assertTrue(fifth['pilot_passed']);self.assertEqual(fifth['gradient_norm_records'],5)
            self.assertFalse(fifth['optimizer_state_touched']);self.assertEqual(fifth['successful_updates_proven_by_complete_hf'],5);self.assertFalse(fifth['final_hundred_checkpoint_complete'])
    def test_bad_norm_or_gap_never_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);torch.save(float('nan'),root/'grad_norm_actor_0_0.pt')
            with self.assertRaisesRegex(ValueError,'Nonfinite'):read_progress(root,minimum_file_age=0)
            (root/'grad_norm_actor_0_0.pt').unlink();torch.save(1.,root/'grad_norm_actor_0_1.pt')
            with self.assertRaisesRegex(ValueError,'contiguous'):read_progress(root,minimum_file_age=0)
    def test_complete_checkpoint_must_have_correct_step_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.records(root);self.checkpoint(root,completed=4)
            with self.assertRaisesRegex(ValueError,'provenance'):read_progress(root,minimum_file_age=0)
    def test_partial_checkpoint_does_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.records(root);self.checkpoint(root,complete=False)
            self.assertFalse(read_progress(root,minimum_file_age=0)['pilot_passed'])
    def test100_norm_records_do_not_replace_complete100_hf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.records(root,100);self.checkpoint(root)
            before=read_progress(root,minimum_file_age=0)
            self.assertEqual(before['gradient_norm_records'],100)
            self.assertEqual(before['successful_updates_proven_by_complete_hf'],5)
            self.assertFalse(before['final_hundred_checkpoint_complete'])
            self.checkpoint(root,completed=100,step=100)
            after=read_progress(root,minimum_file_age=0)
            self.assertEqual(after['successful_updates_proven_by_complete_hf'],100)
            self.assertTrue(after['final_hundred_checkpoint_complete'])

if __name__=='__main__':unittest.main()
