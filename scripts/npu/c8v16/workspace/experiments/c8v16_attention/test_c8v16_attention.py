"""CPU numerical, gradient and causal replay tests for the C8V16 profile."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
import torch.nn.functional as F

here = Path(__file__).resolve().parent
for device_core in (
    here.parent / 'c4v8_device_core',
    here.parents[3] / 'c4v8/workspace/experiments/c4v8_device_core',
):
    if device_core.is_dir():
        sys.path.insert(0, str(device_core))
        break

from c8v16_attention import AttentionConfig, _npu_mx, attention


def ref_bf(x):
    rounded = x.detach().bfloat16().float()
    # Preserve the structural negative infinity in the first softmax shift.
    return rounded + (x - x.detach()).nan_to_num(0)


def ref_mx(x):
    """Independent E8M0 scale computation and torch native E4M3FN rounding."""
    width = x.size(-1)
    raw = F.pad(x.detach(), (0, -width % 32))
    blocks = raw.reshape(*x.shape[:-1], -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True)
    scale = torch.where(amax == 0, 1., (amax / 448).log2().ceil().exp2())
    quantized = ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).reshape_as(raw)
    return quantized[..., :width] + (x - x.detach())


def reference(q, k, v, offset, key_block):
    """One causal prefix at a time; groups never contain unseen V values."""
    rows = []
    for batch in range(q.size(0)):
        heads = []
        for qh in range(q.size(1)):
            kh = qh // (q.size(1) // k.size(1))
            tokens = []
            for i in range(q.size(2)):
                stop = offset + i + 1
                query = ref_mx(ref_bf(q[batch, qh, i].float()))
                maximum = query.new_tensor(-float('inf'))
                denominator = query.new_zeros(())
                numerator = torch.zeros_like(query)
                for start in range(0, stop, key_block):
                    end = min(start + key_block, stop)
                    key = ref_mx(ref_bf(k[batch, kh, start:end].float()))
                    val = ref_mx(ref_bf(v[batch, kh, start:end].float()).T).T
                    scores = ref_bf((key @ query) * (q.size(-1) ** -0.5))
                    new_max = ref_bf(torch.maximum(maximum, scores.max()))
                    alpha = ref_bf(ref_bf(maximum - new_max).exp())
                    p = ref_bf(ref_bf(scores - new_max).exp())
                    numerator = ref_bf(ref_bf(numerator * alpha) + ref_bf(ref_mx(p) @ val))
                    denominator = ref_bf(ref_bf(denominator * alpha) + ref_bf(p.sum()))
                    maximum = new_max
                tokens.append((numerator / denominator).bfloat16())
            heads.append(torch.stack(tokens))
        rows.append(torch.stack(heads))
    return torch.stack(rows)


def inputs(b=1, h=2, hk=1, n=6, nk=43, d=35):
    g = torch.Generator().manual_seed(2305)
    def rand(*shape):
        return torch.randn(*shape, generator=g).bfloat16()
    return rand(b, h, n, d), rand(b, hk, nk, d), rand(b, hk, nk, d)


class C8V16Tests(unittest.TestCase):
    def test_fused_launch_packing_preserves_independent_scale_groups(self):
        class KernelFixture:
            def __getitem__(self, grid):
                def run(packed, values, codes, scales, flags, width, block, mask, **options):
                    self.assertions(grid, packed, width, block)
                    values.copy_(ref_mx(packed.reshape(-1, width)).reshape(-1))
                    flags.zero_()
                return run

            @staticmethod
            def assertions(grid, packed, width, block):
                if block != width or width % 32 or width > 1024 or grid != (packed.numel() // width,):
                    raise AssertionError('invalid Triton launch or block layout')

        g = torch.Generator().manual_seed(3412)
        with patch.dict(sys.modules, {'fused_mxfp8': SimpleNamespace(qdq_kernel=KernelFixture())}):
            for width in (1, 17, 32, 33, 129, 4097, 8192):
                # Adjacent rows have very different magnitudes, revealing scale leakage.
                x = torch.randn(3, 2, width, generator=g) * torch.tensor([1., 32768.])[:, None]
                torch.testing.assert_close(_npu_mx(x), ref_mx(x), rtol=0, atol=0)

    def test_native_float8_reference_and_first_order_gradients(self):
        config = AttentionConfig(query_block=4, key_block=32)
        xs = [t.float().requires_grad_() for t in inputs()]
        ys = [t.detach().clone().requires_grad_() for t in xs]
        expected = reference(*ys, 30, 32)
        actual = attention(*xs, config, query_offset=30)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weight = torch.linspace(-1, 1, actual.numel()).reshape_as(actual)
        grad = torch.autograd.grad((actual * weight).sum(), xs)
        ref_grad = torch.autograd.grad((expected * weight).sum(), ys)
        for got, ref in zip(grad, ref_grad):
            torch.testing.assert_close(got, ref, rtol=3e-5, atol=2e-6)
            self.assertTrue(bool(got.abs().any()))

    def test_prefill_extend_decode_and_future_value_outlier(self):
        q, k, v = inputs(n=70, nk=70, d=17)
        config = AttentionConfig(query_block=11, key_block=64)
        full = attention(q, k, v, config)
        for n in (1, 13, 31, 32, 33, 63, 64, 65, 70):
            out = attention(q[:, :, n-1:n], k[:, :, :n], v[:, :, :n], config, query_offset=n-1)
            torch.testing.assert_close(out, full[:, :, n-1:n], rtol=0, atol=0)
        qstart = 26
        changed = v.clone()
        changed[:, :, qstart+1:] *= 32768
        changed_output = attention(q, k, changed, config)
        torch.testing.assert_close(changed_output[:, :, :qstart+1], full[:, :, :qstart+1], rtol=0, atol=0)
        self.assertFalse(torch.equal(changed_output[:, :, qstart+1:], full[:, :, qstart+1:]))
        extension = attention(q[:, :, 27:67], k[:, :, :67], v[:, :, :67], config, query_offset=27)
        torch.testing.assert_close(extension, full[:, :, 27:67], rtol=0, atol=0)

    def test_qk_and_pv_groups_use_their_contraction_axes(self):
        q, k, v = inputs(b=2, h=4, hk=2, n=1, nk=67, d=36)
        # A large outlier only in a single V column/32-token group; quantizing
        # V over features instead would suppress other columns of this token.
        v[:, :, 35, 3] = 4096
        actual = attention(q, k, v, query_offset=66)
        torch.testing.assert_close(actual, reference(q, k, v, 66, 128), rtol=0, atol=0)

    def test_suffix_gradients_zero_and_replay_matches_bf16_inputs(self):
        config = AttentionConfig(query_block=3, key_block=32)
        q, k, v = inputs(n=7, nk=50, d=16)
        leaves = [t.requires_grad_() for t in (q, k, v)]
        out = attention(*leaves, config, query_offset=26)
        gr = torch.autograd.grad(out[:, :, :2].sum(), leaves)
        for tensor in gr[1:]:
            self.assertEqual(torch.count_nonzero(tensor[:, :, 28:]), 0)
        torch.testing.assert_close(out, attention(*(t.float() for t in leaves), config, query_offset=26),
                                   rtol=0, atol=0)

    def test_wrong_profile_offsets_shapes_and_nonfinite_fail(self):
        q, k, v = inputs()
        for kw in ({'key_block':31}, {'causal':False}, {'vector':'mxfp8'}, {'quantize_operands':False}):
            with self.assertRaises(ValueError):
                AttentionConfig(**kw)
        with self.assertRaises(ValueError):
            attention(q, k, v, query_offset=40)
        with self.assertRaises(TypeError):
            attention(q.half(), k.half(), v.half())
        k[0, 0, 0, 3] = float('nan')
        with self.assertRaises((ValueError, FloatingPointError)):
            attention(q, k, v)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
