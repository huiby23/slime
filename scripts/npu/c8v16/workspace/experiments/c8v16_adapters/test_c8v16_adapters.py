"""Exercise the two adapters with distinct physical and logical KV order on CPU."""

import dataclasses
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from c8v16_megatron_spec import C8V16CoreAttention, replace_attention
from c8v16_profile import BACKEND_NAME, _digest, implementation_sha256, load_module, load_profile
from c8v16_sglang import make_backend_class, wrap_defaults, wrap_pool

core = load_module("c8v16_attention")
HERE = Path(__file__).resolve().parent


class NativePool:
    def __init__(self, rows, heads, dim):
        self.keys = torch.zeros((rows // 4, 4, heads, dim), dtype=torch.bfloat16)
        self.values = torch.zeros_like(self.keys)
        self.use_fia = False

    def get_key_buffer(self, layer):
        return self.keys

    def get_value_buffer(self, layer):
        return self.values

    def set_kv_buffer(self, layer, loc, k, v):
        self.keys.flatten(0, 1)[loc] = k
        self.values.flatten(0, 1)[loc] = v


class Mode:
    def __init__(self, name):
        self.name = name

    def is_idle(self):
        return False

    def is_mixed(self):
        return False

    def is_target_verify(self):
        return False

    def is_draft_extend(self, include_v2=False):
        return False

    def is_decode(self):
        return self.name == "DECODE"

    def is_extend(self):
        return self.name == "EXTEND"


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = core.AttentionConfig(query_block=16, key_block=32)
        self.attention = C8V16CoreAttention(NS(kv_channels=32), 1, "causal",
                                           kernel=core.attention, precision_config=self.config)

    def qkv(self, n):
        g = torch.Generator().manual_seed(43)
        return tuple(torch.randn((n, h, 32), generator=g).bfloat16() for h in (4, 2, 2))

    def test_cached_sg_chunked_and_decode_equal_megatron_full(self):
        q, k, v = self.qkv(39)
        v[30] *= 1024
        dense = self.attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)
        pool = NativePool(64, 2, 32)
        slots = torch.randperm(64, generator=torch.Generator().manual_seed(88))[:39]
        table = slots.reshape(1, -1).int()
        events = {}
        cls = make_backend_class(object, core.attention, self.config, NativePool, events.__setitem__)
        args = NS(disable_cuda_graph=True, enable_torch_compile=False, speculative_algorithm=None,
                  enable_mixed_chunk=False, attention_backend=BACKEND_NAME,
                  prefill_attention_backend=BACKEND_NAME, decode_attention_backend=BACKEND_NAME)
        backend = cls(NS(server_args=args, page_size=4))
        self.assertEqual(events["backend_created"]["core_module"], "c8v16_attention")
        layer = NS(tp_q_head_num=4, tp_k_head_num=2, tp_v_head_num=2, qk_head_dim=32, v_head_dim=32,
                   scaling=1/math.sqrt(32), is_cross_attention=False, attn_type="decoder", layer_id=0,
                   sliding_window_size=-1, logit_cap=0.0, quant_method=None)
        outputs = []
        for start, end, mode in [(0, 17, "EXTEND"), (17, 30, "EXTEND"), (30, 31, "DECODE"),
                                 (31, 38, "EXTEND"), (38, 39, "DECODE")]:
            batch = NS(forward_mode=Mode(mode), batch_size=1, req_pool_indices=[0], seq_lens_cpu=[end],
                       extend_seq_lens_cpu=[end-start], extend_prefix_lens_cpu=[start],
                       out_cache_loc=slots[start:end], token_to_kv_pool=pool, req_to_token_pool=NS(req_to_token=table))
            backend.init_forward_metadata(batch)
            forward = backend.forward_extend if mode == "EXTEND" else backend.forward_decode
            outputs.append(forward(q[start:end], k[start:end], v[start:end], layer, batch))
        torch.testing.assert_close(torch.cat(outputs), dense, atol=0, rtol=0)
        self.assertEqual(events["first_extend"]["core_module"], "c8v16_attention")
        self.assertEqual(events["first_decode"]["core_module"], "c8v16_attention")
        torch.testing.assert_close(pool.values.flatten(0, 1)[slots], v, atol=0, rtol=0)

    def test_packed_trainer_forward_backward_and_padding(self):
        q, k, v = [x.requires_grad_() for x in self.qkv(14)]
        cu = torch.tensor([0, 5, 12], dtype=torch.int32)
        padded = torch.tensor([0, 6, 14], dtype=torch.int32)
        params = NS(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                    cu_seqlens_q_padded=padded, cu_seqlens_kv_padded=padded)
        out = self.attention(q, k, v, packed_seq_params=params)
        for start, end in [(0, 5), (6, 13)]:
            expected = self.attention(*(x[start:end].unsqueeze(1) for x in (q,k,v)))
            torch.testing.assert_close(out[start:end].flatten(1), expected.squeeze(1), atol=0, rtol=0)
        out.float().square().mean().backward()
        for t in (q, k, v):
            self.assertTrue(torch.isfinite(t.grad).all() and t.grad.abs().max() > 0)
            self.assertEqual(t.grad[[5, 13]].abs().sum().item(), 0)

    def test_selector_override_preserves_native_pool_behavior_and_restores_on_error(self):
        def native_defaults(args):
            args.page_size = 128
            args.attention_backend = args.decode_attention_backend = args.prefill_attention_backend = "ascend"

        args = NS(attention_backend=BACKEND_NAME, prefill_attention_backend=None, decode_attention_backend=None,
                  speculative_algorithm=None, disaggregation_mode=None)
        wrap_defaults(native_defaults)(args)
        self.assertEqual(args.prefill_attention_backend, BACKEND_NAME)
        self.assertEqual(args.page_size, 128)
        runner = NS(server_args=args, device="npu", use_mla_backend=False, is_hybrid_swa=False,
                    mambaish_config=None, is_draft_worker=False, token_to_kv_pool_allocator=None)
        def native_pool(runner):
            self.assertEqual(runner.server_args.attention_backend, "ascend")
            raise RuntimeError("native allocation test")
        with self.assertRaisesRegex(RuntimeError, "native allocation test"):
            wrap_pool(native_pool, NativePool, type("Allocator", (), {}), None)(runner)
        self.assertIs(runner.server_args, args)
        self.assertEqual(args.attention_backend, BACKEND_NAME)
        args.decode_attention_backend = "ascend"
        with self.assertRaises(ValueError):
            wrap_defaults(native_defaults)(args)

    def test_moe_spec_replaces_only_attention(self):
        block = NS(layer_specs=[NS(submodules=NS(mlp="bf16-moe", self_attention=NS(submodules=NS(core_attention="base"))))])
        spec = replace_attention(block, self.config, core, NS)
        self.assertEqual(block.layer_specs[0].submodules.self_attention.submodules.core_attention, "base")
        self.assertEqual(spec.layer_specs[0].submodules.mlp, "bf16-moe")
        selected = spec.layer_specs[0].submodules.self_attention.submodules.core_attention
        self.assertIs(selected.module, C8V16CoreAttention)
        self.assertIs(selected.params["kernel"], core.attention)

    def test_deployment_hashes_and_shim_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = dataclasses.asdict(self.config)
            file = root / "policy.json"
            file.write_text(json.dumps(policy))
            env = {"HBY_C8V16_POLICY_FILE": str(file), "HBY_C8V16_POLICY_SHA256": _digest(policy),
                   "HBY_C8V16_IMPLEMENTATION_SHA256": implementation_sha256()}
            with patch.dict(os.environ, env):
                self.assertIs(load_profile().core, core)
                os.environ["HBY_C8V16_IMPLEMENTATION_SHA256"] = "old"
                with self.assertRaisesRegex(ValueError, "implementation SHA"):
                    load_profile()
            show = root / "show-argv"
            show.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            show.chmod(0o700)
            shimenv = dict(os.environ, HBY_C8V16_REAL_PYTHON=str(show))
            result = subprocess.check_output([str(HERE/"python-c8v16"), "-m", "slime.backends.sglang_utils.runtime"],
                                             env=shimenv, text=True)
            self.assertEqual(result.splitlines(), ["-m", "c8v16_runtime"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
