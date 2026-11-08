"""Exercise lossless frames, including chunk edges, and cross-process Adam recovery on CPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from slime.utils import hybrid_restart as restart


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(65, 16).to(torch.bfloat16))
        self.register_buffer("b", torch.arange(15, dtype=torch.int16)[::2])


class Scheduler:
    def __init__(self):
        self.num_steps = 0

    def state_dict(self):
        return {"num_steps":self.num_steps}

    def load_state_dict(self,s):
        self.num_steps += s["num_steps"]


def do_phase(root, resume):
    torch.set_num_threads(1)
    torch.manual_seed(443 if not resume else 9923)
    net = Model()
    master = torch.nn.Parameter(net.w.float().detach().clone())
    inner = torch.optim.AdamW([master], lr=.005)
    inner.state[master].update(step=torch.tensor(0.), exp_avg=torch.zeros_like(master),exp_avg_sq=torch.zeros_like(master))
    # Same live object bindings as a hybrid optimizer; this is a CPU serialization fixture.
    inner.state[master]["master_param"] = master
    hybrid = SimpleNamespace(sub_optimizers=[inner],param_groups=[{"params":[net.w], "lr":.005}],
                             param_to_inner_param={net.w:master})
    wrapper = SimpleNamespace(optimizer=hybrid,model_param_group_index_map={net.w:(0,0)})
    sch = Scheduler()
    it = SimpleNamespace(offset=0,rollout_data={"old_logp":torch.arange(12,dtype=torch.float32)})
    def one():
        val = it.rollout_data["old_logp"][it.offset]
        loss = (master + torch.rand_like(master)*.01-val).square().mean()
        loss.backward()
        inner.step()
        inner.zero_grad()
        with torch.no_grad():
            net.w.copy_(master)
            net.w[0,0] += 1  # Force a nonzero BF16/model residual and test it independently.
        sch.num_steps += 4
        it.offset += 1
        return loss.detach().item()
    if not resume:
        for _ in range(5):
            one()
        proof = restart.save_rank(root,0,[net],wrapper,sch,[it],rng=restart.rng_state())
        (root/"rank.json").write_text(json.dumps(proof))
    else:
        proof = json.loads((root/"rank.json").read_text())
        rng, offsets, batch = restart.load_rank(root,proof,[net],wrapper,sch)
        it.offset, = offsets
        it.rollout_data = restart.map_tensors(batch,saving=False)
        restart.restore_rng(rng)
    before = net.w.clone()
    assert it.offset == 5
    loss = one()
    results = [before,net.w,master,inner.state[master]["exp_avg"],inner.state[master]["exp_avg_sq"],inner.state[master]["step"],net.b,sch.num_steps,it.offset,loss]
    if resume:
        expected = torch.load(root/"reference.pt", weights_only=True)
        assert all(torch.equal(x,y) if torch.is_tensor(x) else x==y for x,y in zip(results,expected,strict=True))
    else:
        torch.save(results,root/"reference.pt")


class RecoveryTests(unittest.TestCase):
    def test_fresh_process_next_update_and_corruption(self):
        with TemporaryDirectory() as tmp:
            for mode in ("save","resume"):
                subprocess.run([sys.executable,__file__,mode,tmp],check=True)
            path = Path(tmp)/"rank-0.zst"
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
            failed=subprocess.run([sys.executable,__file__,"resume",tmp],capture_output=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertRegex(failed.stderr.decode(),"mismatch|corrupt|decompress|Invalid")

    def test_source_cursor_buffer_and_epoch_shuffle(self):
        from slime.data.data_source import RolloutDataSourceWithBuffer
        from slime.utils.data import Dataset
        from slime.utils.types import Sample

        def source():
            result = RolloutDataSourceWithBuffer(SimpleNamespace(prompt_data=None, n_samples_per_prompt=16,
                rollout_shuffle=True, buffer_filter_path=None))
            ds = Dataset.__new__(Dataset)
            ds.origin_samples = [Sample(prompt=f"probe-{i}", label=str(i)) for i in range(17)]
            ds.seed, ds.epoch_id, ds.samples = 1234, -1, ds.origin_samples
            ds.shuffle(0)
            result.dataset = ds
            return result

        with TemporaryDirectory() as tmp:
            old = source()
            old.get_samples(14)
            reserved = old.get_samples(1)
            old.add_samples(reserved)
            args = SimpleNamespace(optimizer_step_hf_save_dir=str(Path(tmp) / "snapshots"))
            digest = restart.save_source(args, 0, old)
            path = Path(tmp) / "rollout-source/rollout-0.pt"
            self.assertEqual(digest, restart.digest(path))
            restored = source()
            restored.load_state_dict(torch.load(path, weights_only=False)["source"])
            for n in (1, 3, 4):
                left, right = old.get_samples(n), restored.get_samples(n)
                fields = lambda samples: [(s.prompt,s.label,s.group_index,s.index) for group in samples for s in group]
                self.assertEqual(fields(left), fields(right))
            self.assertEqual(old.epoch_id, 1)

    def test_publish_and_prune_only_committed_state(self):
        for retention in ("all", "latest"):
            with self.subTest(retention=retention), TemporaryDirectory() as tmp:
                run = Path(tmp) / "run"
                run.mkdir()
                (run / "manifest.json").write_text(json.dumps({"status":"running"}))
                store = run / "store"
                store.mkdir()
                for count in (5, 10):
                    bundle = store / f"step-{count:06d}-run"
                    bundle.mkdir()
                    (bundle/"restart.json").write_text(json.dumps(dict(completed=count,run=str(run),
                        fingerprint="test",retention=retention)))
                    restart.advance_latest(bundle)
                self.assertEqual((store/"LATEST").read_text().strip(),bundle.name)
                self.assertEqual((store/"step-000005-run").is_dir(),retention=="all")
                self.assertEqual(json.loads((run/"manifest.json").read_text())["latest_durable_optimizer_steps"], 10)

    def test_retention_change_after_new_checkpoint_is_durable(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run/"manifest.json").write_text(json.dumps({"status":"running"}))
            store = run/"store"
            store.mkdir()
            old = None
            for count in (5,10,15):
                if count == 15:
                    (store/"retention.json").write_text(json.dumps(dict(fingerprint="matching",retention="latest")))
                mode = restart.retention_policy(store,"all","matching")
                current = store/f"step-{count:06d}-run"
                current.mkdir()
                (current/"restart.json").write_text(json.dumps(dict(completed=count,run=str(run),fingerprint="matching",
                                                    retention=mode,previous_committed=old)))
                restart.advance_latest(current)
                old = current.name
                self.assertEqual(len(list(store.glob("step-*"))),{5:1,10:2,15:1}[count])
            with self.assertRaises(ValueError):
                restart.retention_policy(store,"all","unrelated")

    def test_frame_chunk_edges(self):
        with TemporaryDirectory() as tmp:
            original = torch.arange(restart.CHUNK+1, dtype=torch.int32)
            with open(Path(tmp)/"f","wb") as f:
                restart.Frames(f,True).tensor(original)
            loaded = torch.empty_like(original)
            with open(Path(tmp)/"f","rb") as f:
                restart.Frames(f,False).tensor(loaded)
            self.assertTrue(torch.equal(loaded,original))


if __name__=="__main__":
    if len(sys.argv)==3:
        do_phase(Path(sys.argv[2]),sys.argv[1]=="resume")
    else:
        unittest.main()
