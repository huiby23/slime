"""CPU tests against independent scalar IEEE-format reference arithmetic.

These tests validate this Python numerical profile, not parity with TE/CUDA.
No accelerator API is called, and test random data uses private CPU generators.
Run: python -m unittest -v test_nvfp4
"""

from __future__ import annotations

import math
import struct
import unittest

import torch

from nvfp4 import (
    NVFP4Tensor,
    dequantize,
    fake_quantize_ste,
    quantize,
    round_e2m1,
    round_e4m3fn,
)


def f32(value: float) -> float:
    """Independent IEEE binary32 rounding, with explicit overflow handling."""
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


F32_MAX = f32(float.fromhex("0x1.fffffep+127"))
E2_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def scalar_e2m1(value: float) -> tuple[float, int]:
    value = max(-6.0, min(6.0, value))
    magnitude = abs(value)
    # Enumerate all candidates rather than sharing bucket boundaries with code.
    code = min(range(8), key=lambda i: (abs(E2_MAG[i] - magnitude), i & 1))
    negative = math.copysign(1.0, value) < 0
    return math.copysign(E2_MAG[code], value), code | (8 if negative else 0)


def e4m3_positive(code: int) -> float:
    exponent, fraction = code >> 3, code & 7
    if exponent == 0:
        return fraction * 2.0 ** -9
    return (1.0 + fraction / 8.0) * 2.0 ** (exponent - 7)


def scalar_e4m3(value: float) -> tuple[float, int]:
    value = max(-448.0, min(448.0, value))
    magnitude = abs(value)
    # E4M3FN uses 0x7f for NaN. The 127 nonnegative finite codes are monotonic.
    code = min(range(127), key=lambda i: (abs(e4m3_positive(i) - magnitude), i & 1))
    negative = math.copysign(1.0, value) < 0
    return math.copysign(e4m3_positive(code), value), code | (128 if negative else 0)


def scalar_reference(x: torch.Tensor, scope: str) -> dict[str, torch.Tensor]:
    """Independent Python loops and scalar FP32 rounding at every operation."""
    rows = x.detach().float().reshape(-1, x.shape[-1]).tolist()
    width = len(rows[0])
    padded_width = ((width + 15) // 16) * 16
    tensor_amax = max(abs(v) for row in rows for v in row)
    codes, block_bytes, output = [], [], []
    global_encodes, global_decodes = [], []
    for row in rows:
        upper_amax = max(abs(v) for v in row) if scope == "token" else tensor_amax
        encode = 1.0 if upper_amax == 0 else min(f32(2688.0 / upper_amax), F32_MAX)
        decode = f32(1.0 / encode)
        global_encodes.append(encode)
        global_decodes.append(decode)
        multiplier = f32(encode * f32(1.0 / 6.0))
        padded = row + [0.0] * (padded_width - width)
        row_codes, row_scales, row_output = [], [], []
        for j in range(0, padded_width, 16):
            block = padded[j:j + 16]
            block_amax = max(abs(v) for v in block)
            before_cast = max(0.0, min(448.0, f32(block_amax * multiplier)))
            scale, scale_code = scalar_e4m3(before_cast)
            product = f32(scale * decode)
            reciprocal = math.inf if product == 0 else f32(1.0 / product)
            block_encode = min(reciprocal, F32_MAX)
            row_scales.append(scale_code)
            for value in block:
                q, code = scalar_e2m1(f32(value * block_encode))
                row_codes.append(code)
                row_output.append(f32(f32(q * scale) * decode))
        codes.append(row_codes)
        block_bytes.append(row_scales)
        output.append(row_output[:width])
    upper_shape = (-1, 1) if scope == "token" else ()
    return {
        "codes": torch.tensor(codes, dtype=torch.uint8),
        "block_scale_bytes": torch.tensor(block_bytes, dtype=torch.uint8),
        "global_encode": torch.tensor(global_encodes if scope == "token" else global_encodes[0], dtype=torch.float32).reshape(upper_shape),
        "global_decode": torch.tensor(global_decodes if scope == "token" else global_decodes[0], dtype=torch.float32).reshape(upper_shape),
        "dequantized": torch.tensor(output, dtype=torch.float32).reshape(x.shape),
    }


class NVFP4Tests(unittest.TestCase):
    def assert_float_bits_equal(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        self.assertTrue(torch.equal(actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32)), (actual, expected))

    def assert_reference(self, x: torch.Tensor, scope: str) -> NVFP4Tensor:
        reference = scalar_reference(x, scope)
        result = quantize(x, scope)
        self.assertTrue(torch.equal(result.codes, reference["codes"]))
        self.assertTrue(torch.equal(result.block_scale_bytes, reference["block_scale_bytes"]))
        self.assert_float_bits_equal(result.global_encode_scale, reference["global_encode"])
        self.assert_float_bits_equal(result.global_decode_scale, reference["global_decode"])
        self.assert_float_bits_equal(dequantize(result), reference["dequantized"])
        return result

    def test_all_e2m1_codes_and_signed_zero(self):
        values = list(E2_MAG) + [-v for v in E2_MAG]
        source = torch.tensor(values, dtype=torch.float32)
        rounded = round_e2m1(source)
        self.assert_float_bits_equal(rounded, source)
        # Fix the block scale to 1 via upper scaling of amax=6 and check all 16 codes.
        result = self.assert_reference(source, "tensor")
        self.assertTrue(torch.equal(result.codes[0], torch.arange(16, dtype=torch.uint8)))

    def test_e2m1_midpoints_and_adjacent_float32_values(self):
        middle = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        cases = torch.cat((
            torch.nextafter(middle, torch.full_like(middle, -math.inf)),
            middle,
            torch.nextafter(middle, torch.full_like(middle, math.inf)),
        ))
        cases = torch.cat((cases, -cases))
        expected = torch.tensor([scalar_e2m1(v)[0] for v in cases.tolist()])
        self.assert_float_bits_equal(round_e2m1(cases), expected)
        self.assertTrue(torch.equal(round_e2m1(middle), torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])))

    def test_e2m1_saturates_finite_overflow(self):
        self.assertTrue(torch.equal(round_e2m1(torch.tensor([-F32_MAX, F32_MAX])), torch.tensor([-6.0, 6.0])))

    def test_e4m3_all_codes_subnormals_and_midpoints(self):
        finite = torch.tensor([e4m3_positive(i) for i in range(127)], dtype=torch.float32)
        self.assert_float_bits_equal(round_e4m3fn(finite), finite)
        self.assert_float_bits_equal(round_e4m3fn(-finite), -finite)
        middle = (finite[:-1] + finite[1:]) * 0.5
        cases = torch.cat((
            middle,
            torch.nextafter(middle, torch.full_like(middle, -math.inf)),
            torch.nextafter(middle, torch.full_like(middle, math.inf)),
            -middle,
        ))
        expected = torch.tensor([scalar_e4m3(v)[0] for v in cases.tolist()])
        self.assert_float_bits_equal(round_e4m3fn(cases), expected)

    def test_subnormal_block_scales_and_zero_scale_underflow(self):
        source = torch.zeros(1, 48)
        source[0, 0] = 2688.0  # Makes upper encode/decode scales exactly one.
        source[0, 16] = 6.0 * 2.0 ** -9
        source[0, 32] = 6.0 * 2.0 ** -10  # E4M3 scale ties to zero.
        result = self.assert_reference(source, "tensor")
        self.assertTrue(torch.equal(result.block_scale_bytes, torch.tensor([[126, 1, 0]], dtype=torch.uint8)))
        self.assertEqual(dequantize(result)[0, 32].item(), 0.0)
        self.assertTrue(torch.isfinite(dequantize(result)).all())

    def test_random_tensor_and_token_profiles_match_scalar_reference(self):
        generator = torch.Generator(device="cpu").manual_seed(9017)
        for shape in ((17,), (2, 37), (2, 3, 31)):
            base = torch.randn(shape, generator=generator)
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                x = base.to(dtype)
                for scope in ("token", "tensor"):
                    with self.subTest(shape=shape, dtype=dtype, scope=scope):
                        self.assert_reference(x, scope)

    def test_token_scope_is_independent_per_last_axis_row(self):
        source = torch.tensor([[1.0] * 17, [1000.0] * 17, [0.0] * 17])
        combined = quantize(source, "token")
        for i in range(3):
            individual = quantize(source[i], "token")
            self.assertTrue(torch.equal(combined.codes[i], individual.codes[0]))
            self.assertTrue(torch.equal(combined.block_scale_bytes[i], individual.block_scale_bytes[0]))
            self.assert_float_bits_equal(combined.global_decode_scale[i], individual.global_decode_scale[0])

    def test_padding_and_noncontiguous_inputs(self):
        source = torch.arange(1, 43, dtype=torch.float32).reshape(7, 6).t()
        self.assertFalse(source.is_contiguous())
        result = self.assert_reference(source, "token")
        self.assertEqual(result.padded_last_dim, 16)
        self.assertEqual(result.metadata["right_padding"], 9)
        self.assertTrue(torch.equal(result.codes[:, 7:], torch.zeros_like(result.codes[:, 7:])))
        self.assertEqual(dequantize(result).shape, source.shape)

    def test_zero_scope_rule_and_signed_zeros(self):
        source = torch.tensor([[0.0, -0.0] * 9, [-0.0, 0.0] * 9])
        for scope in ("token", "tensor"):
            result = self.assert_reference(source, scope)
            self.assertTrue(torch.equal(result.global_encode_scale, torch.ones_like(result.global_encode_scale)))
            self.assertTrue(torch.equal(result.global_decode_scale, torch.ones_like(result.global_decode_scale)))
            self.assertTrue(torch.equal(result.block_scale_bytes, torch.zeros_like(result.block_scale_bytes)))
            self.assert_float_bits_equal(dequantize(result), source)

    def test_extreme_finite_inputs(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            limits = torch.finfo(dtype)
            source = torch.tensor([[limits.max, -limits.max] + [0.0] * 14, [limits.tiny, -limits.tiny] + [0.0] * 14], dtype=dtype)
            for scope in ("token", "tensor"):
                with self.subTest(dtype=dtype, scope=scope):
                    result = self.assert_reference(source, scope)
                    self.assertTrue(torch.isfinite(dequantize(result)).all())

    def test_nonfinite_inputs_are_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            source = torch.tensor([bad, 1.0])
            for operation in (quantize, round_e2m1, round_e4m3fn, fake_quantize_ste):
                with self.subTest(bad=bad, operation=operation.__name__):
                    with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                        operation(source)

    def test_invalid_shapes_dtypes_scopes_are_rejected(self):
        for source in (torch.empty(0), torch.empty(2, 0), torch.tensor(1.0)):
            with self.assertRaises(ValueError):
                quantize(source)
        for source in (torch.tensor([1]), torch.tensor([1.0], dtype=torch.float64)):
            with self.assertRaises(TypeError):
                quantize(source)
        with self.assertRaises(ValueError):
            quantize(torch.ones(16), scope="channel")

    def test_ste_forward_exact_and_backward_identity(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for scope in ("token", "tensor"):
                source = torch.tensor([[0.17, 0.61, 1.03, -1.27, 3.9] * 4], dtype=dtype, requires_grad=True)
                output = fake_quantize_ste(source, scope)
                expected = dequantize(quantize(source, scope), dtype=dtype)
                self.assertEqual(output.dtype, dtype)
                self.assertTrue(torch.equal(output, expected))
                self.assertFalse(torch.equal(output, source))
                upstream = torch.arange(20, dtype=dtype).reshape(1, 20) / 7
                gradient, = torch.autograd.grad(output, source, upstream)
                self.assertTrue(torch.equal(gradient, upstream))
                self.assertTrue(torch.isfinite(gradient).all())

    def test_no_global_rng_or_input_mutation(self):
        generator = torch.Generator(device="cpu").manual_seed(314)
        source = torch.randn((3, 35), generator=generator).requires_grad_()
        before_data = source.detach().clone()
        before_rng = torch.random.get_rng_state().clone()
        first = quantize(source)
        second = quantize(source)
        fake_quantize_ste(source).square().sum().backward()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))
        self.assertTrue(torch.equal(source.detach(), before_data))
        self.assertTrue(torch.equal(first.codes, second.codes))
        self.assertTrue(torch.equal(first.block_scale_bytes, second.block_scale_bytes))
        self.assertIsNone(first.codes.grad_fn)
        self.assertIsNone(first.global_decode_scale.grad_fn)

    def test_metadata_exposes_simulation_boundary(self):
        result = quantize(torch.ones(2, 17))
        self.assertFalse(result.metadata["native_kernel_validated"])
        self.assertFalse(result.metadata["stochastic_rounding"])
        self.assertFalse(result.metadata["random_hadamard_transform"])
        self.assertEqual(result.metadata["block_size"], 16)
        self.assertEqual(result.metadata["block_scale_format"], "E4M3FN")


if __name__ == "__main__":
    unittest.main(verbosity=2)
