import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

from slime_plugins.nvfp4_fa.attention import attention, decode, rows, values
from slime_plugins.nvfp4_fa.megatron import PackedAttention
from slime_plugins.nvfp4_fa.serving import paged_forward


class Core(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False

    def tensors(self, n):
        return tuple(torch.randn(1, h, n, 32, device=self.device).bfloat16() for h in (4, 2, 2))

    def test_nvfp4_codes_scales_and_zero(self):
        # With upper=1/448 and local=448, reconstructed E2M1 values have scale 1.
        row = torch.tensor(
            [[0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -6, -1, 2, 3, 4, 1.5, -0.5]], device=self.device
        )
        expected = row.new_tensor([[0, 0, 1, 1, 2, 2, 4, 4, 6, -6, -1, 2, 3, 4, 1.5, -0.5]])
        torch.testing.assert_close(rows(row), expected)
        self.assertEqual(rows(torch.zeros_like(row)).abs().max().item(), 0)
        # Fractional E4M3 scale shared by a lower-amplitude group in the same row.
        row2 = torch.cat([row, row * 0.33], dim=-1)
        out = rows(row2)
        block_scale = torch.tensor(448 * 0.33).to(torch.float8_e4m3fn).float().item() / 448
        self.assertAlmostEqual(out[0, 24].item(), 6 * block_scale, places=5)
        slab = torch.stack((row, row * 0.33), dim=-1)
        torch.testing.assert_close(values(slab)[0, :, 1].abs().max(), row.new_tensor(6 * block_scale))

    def test_chunked_prefill_ragged_decode_and_future_isolation(self):
        q, k, v = self.tensors(145)
        pre = attention(q, k, v)
        for n in (1, 15, 16, 17, 65, 127, 128, 129, 145):
            prefix = attention(q[:, :, :n], k[:, :, :n], v[:, :, :n])
            torch.testing.assert_close(prefix, pre[:, :, :n], atol=0, rtol=0)
        # The longest row spans two tiles; padded slots deliberately have unrelated large values.
        lengths = [17, 128, 145]
        batchq = torch.cat([q[:, :, n - 1 : n] for n in lengths])
        bk, bv = k.repeat(3, 1, 1, 1), v.repeat(3, 1, 1, 1)
        for i, n in enumerate(lengths):
            bk[i, :, n:] = 3000
            bv[i, :, n:] = -3000
        result = decode(batchq, bk, bv, torch.tensor(lengths, device=self.device))
        torch.testing.assert_close(result, torch.cat([pre[:, :, n - 1 : n] for n in lengths]), rtol=0, atol=0)
        torch.testing.assert_close(attention(q[:, :, 119:], k, v, offset=119), pre[:, :, 119:], rtol=0, atol=0)

    def test_qwen3_4b_head_geometry(self):
        q, k, v = [torch.randn(1, heads, 131, 128, device=self.device).bfloat16() for heads in (32, 8, 8)]
        out = attention(q, k, v)
        single = decode(q[:, :, -1:], k, v, torch.tensor([131], device=self.device))
        torch.testing.assert_close(single, out[:, :, -1:], rtol=0, atol=0)

    def test_long_causal_decode(self):
        q, k, v = self.tensors(3070)
        for length in (512, 1024, 2048, 3070):
            query = q[:, :, length - 1 : length]
            expected = attention(query, k[:, :, :length], v[:, :, :length], offset=length - 1)
            actual = decode(query, k[:, :, :length], v[:, :, :length], torch.tensor([length], device=self.device))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_ste_gradient_causal_and_finite(self):
        q, k, v = self.tensors(145)
        for t in (q, k, v):
            t.requires_grad_()
        out = attention(q, k, v)
        out[:, :, 42].float().square().sum().backward()
        for t in (q, k, v):
            self.assertTrue(t.grad.isfinite().all() and t.grad.abs().sum() > 0)
            self.assertEqual(t.grad[:, :, 43:].abs().max().item(), 0.0)

    def test_packed_trainer_and_native_request_addresses(self):
        q, k, v = self.tensors(25)
        split = (0, 9, 25)
        thd = [x.squeeze(0).transpose(0, 1) for x in (q, k, v)]
        cu = torch.tensor(split, dtype=torch.int32, device=self.device)
        p = SimpleNamespace(
            qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=None, cu_seqlens_kv_padded=None
        )
        cfg = SimpleNamespace(attention_dropout=0, context_parallel_size=1, tensor_model_parallel_size=1)
        core = PackedAttention(cfg, 1, SimpleNamespace(name="causal"), "self")
        trainer = core(*thd, None, packed_seq_params=p)
        perm = torch.randperm(25, device=self.device)
        table = torch.zeros(2, 16, device=self.device, dtype=torch.long)
        table[1, :9], table[0, :16] = perm[:9], perm[9:]
        keys, vals = torch.empty_like(thd[1]), torch.empty_like(thd[2])

        class Cache:
            dtype = torch.bfloat16

            def set_kv_buffer(self, layer, slots, k, v):
                keys[slots] = k
                vals[slots] = v

            def get_key_buffer(self, layer):
                return keys

            def get_value_buffer(self, layer):
                return vals

        backend = SimpleNamespace(token_to_kv_pool=Cache(), req_to_token_pool=SimpleNamespace(req_to_token=table))
        layer = SimpleNamespace(layer_id=0, head_dim=32, tp_q_head_num=4, tp_k_head_num=2)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(name="EXTEND"),
            seq_lens_cpu=torch.tensor([9, 16]),
            extend_seq_lens_cpu=[9, 16],
            req_pool_indices=cu.new_tensor([1, 0]),
            out_cache_loc=perm,
        )
        result = paged_forward(layer, *thd, batch, backend)
        torch.testing.assert_close(result.reshape_as(trainer), trainer, rtol=0, atol=0)
        selected = [8, 24]
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(name="DECODE"),
            seq_lens_cpu=torch.tensor([9, 16]),
            seq_lens=cu.new_tensor([9, 16]),
            req_pool_indices=cu.new_tensor([1, 0]),
            out_cache_loc=perm[selected],
        )
        result = paged_forward(layer, *(t[selected] for t in thd), batch, backend)
        torch.testing.assert_close(result.reshape_as(trainer[selected]), trainer[selected], rtol=0, atol=0)


class Protocol(unittest.TestCase):
    def test_eight_card_config_and_resume(self):
        from examples.nvfp4_fa_math.run import arguments

        root = Path("/tmp/experiment")
        args = arguments(root / "hf", root / "data", root / "new", root / "parent", 5)
        val = lambda name: args[args.index(name) + 1]
        self.assertEqual(val("--load"), str(root / "parent/checkpoints"))
        self.assertEqual(val("--start-rollout-id"), "5")
        self.assertEqual(val("--ckpt-step"), "4")
        self.assertEqual(val("--num-rollout"), "100")
        self.assertEqual(int(val("--rollout-batch-size")) * int(val("--n-samples-per-prompt")), 40)
        self.assertEqual(val("--global-batch-size"), "40")
        self.assertEqual(val("--save-interval"), val("--eval-interval"))
        self.assertNotIn("--no-save-optim", args)
        self.assertEqual(val("--rollout-num-gpus"), "8")

    def test_evaluation_identity_and_range(self):
        from slime_plugins.nvfp4_fa.records import evaluation

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "coverage").mkdir()
            for i in range(8):
                (root / "coverage" / f"sg_{i}_DECODE.json").touch()
            batch = dict(
                samples=[
                    SimpleNamespace(
                        metadata=dict(source="MATH-500/test", source_index=i, unique_id=f"test/{i}"),
                        label="1",
                        response="1",
                        response_length=1,
                    )
                    for i in range(500)
                ],
                rewards=[1.0] * 250 + [0.0] * 250,
                truncated=[True] + [False] * 499,
            )
            args = SimpleNamespace(save=root / "checkpoints")
            evaluation(4, args, {"math500": batch})
            point = json.loads((root / "evaluation" / "step_005.json").read_text())
            self.assertEqual((point["accuracy"], point["truncation_rate"]), (0.5, 1 / 500))
            batch["samples"][-1] = batch["samples"][0]
            with self.assertRaises(ValueError):
                evaluation(9, args, {"math500": batch})

    def test_block_pv_gradient(self):
        from slime_plugins.nvfp4_fa.pv import pv

        torch.manual_seed(9)
        p = torch.randn(1, 2, 3, 4, 16, requires_grad=True)
        v = torch.randn(1, 2, 4, 16, 5, requires_grad=True)
        a = pv(p, v)
        b = torch.einsum("bhqgi,bhgid->bhqgd", p, v)
        upstream = torch.randn_like(a)
        ga = torch.autograd.grad(a, (p, v), upstream)
        gb = torch.autograd.grad(b, (p, v), upstream)
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CUDACore(Core):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
