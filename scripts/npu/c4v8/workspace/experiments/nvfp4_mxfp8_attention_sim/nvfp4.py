"""CPU reference NVFP4 QDQ for numerical experiments, not a native FP4 kernel.

Profile: E2M1 round-to-nearest ties-to-even (RNE), one E4M3FN scale per
16 consecutive elements along the last dimension, and a FP32 upper scale.
``scope='token'`` independently chooses that upper scale for each last-axis
row; ``scope='tensor'`` uses one upper scale for the complete input tensor.
There is no stochastic rounding, RHT, weight quantization, or 4over6 search.

The FP32 operation order follows ``ref_fp4_quant_te`` and its global-scale
helper from FlashInfer's TE-reference tests, rather than treating algebraically
equivalent divisions/multiplications as bitwise equivalent. Sources:
https://github.com/flashinfer-ai/flashinfer/pull/3387
https://github.com/flashinfer-ai/flashinfer/blob/main/tests/test_helpers/utils_fp4.py
https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html

This implementation has not been compared against TE/CUDA/NPU execution.
Codes and scales have a simple linear, unpacked CPU layout, not a device ABI.
The token scope is an explicitly selected per-row extension to tensor scaling.
For an all-zero scope, upper encode/decode scales are both 1, block scales
are 0, and the original signs of zeros are preserved. No global RNG is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import torch
import torch.nn.functional as F


BLOCK_SIZE = 16
FP4_MAX = 6.0
FP8_MAX = 448.0
PROFILE = "nvfp4_e2m1_e4m3fn_rne_1d_no_rht_no_sr_v1"
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _require_cpu_float(x: torch.Tensor, name: str) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU for this reference implementation")
    if x.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must have float16, bfloat16, or float32 dtype")
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} contains NaN or infinity")


def _e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Encode finite FP32 values; overflow saturates and signed zero survives."""
    absolute = x.abs()
    boundaries = x.new_tensor(_E2M1_MIDPOINTS)
    # At a midpoint, bucketize initially chooses its lower neighbor. Increment
    # only odd lower codes so that the retained significand/code is even.
    lower = torch.bucketize(absolute.contiguous(), boundaries, right=False)
    midpoint = boundaries[lower.clamp_max(len(_E2M1_MIDPOINTS) - 1)]
    tie_to_upper = (absolute == midpoint) & ((lower & 1) != 0)
    magnitude = (lower + tie_to_upper.to(lower.dtype)).to(torch.uint8)
    return magnitude | (torch.signbit(x).to(torch.uint8) << 3)


def _decode_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device="cpu")
    magnitude = table[(codes & 7).to(torch.long)]
    return torch.where((codes & 8) != 0, -magnitude, magnitude)


@torch.no_grad()
def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Return E2M1 RNE/saturating values in FP32, without scaling."""
    _require_cpu_float(x, "E2M1 input")
    return _decode_e2m1_codes(_e2m1_codes(x.to(torch.float32)))


@torch.no_grad()
def round_e4m3fn(x: torch.Tensor) -> torch.Tensor:
    """Return saturating E4M3FN RNE values in FP32, including subnormals."""
    _require_cpu_float(x, "E4M3FN input")
    return x.to(torch.float32).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()


@dataclass(frozen=True)
class NVFP4Tensor:
    """Inspectable logical NVFP4 representation; tensors are detached CPU data.

    ``codes``: [flattened_rows, padded_last_dim], one uint8 nibble per element.
    ``block_scale_bytes``: [flattened_rows, padded_last_dim / 16], E4M3FN bytes.
    Upper scales are scalar for tensor scope, or [flattened_rows, 1] for token.
    Right padding contains zero codes and never crosses a last-axis row.
    """

    codes: torch.Tensor
    block_scale_bytes: torch.Tensor
    global_encode_scale: torch.Tensor
    global_decode_scale: torch.Tensor
    original_shape: tuple[int, ...]
    input_dtype: torch.dtype
    padded_last_dim: int
    scope: str

    @property
    def block_scales(self) -> torch.Tensor:
        """Decode the logical E4M3FN block scales to FP32."""
        return self.block_scale_bytes.contiguous().view(torch.float8_e4m3fn).float()

    @property
    def metadata(self) -> dict[str, Any]:
        """JSON-compatible description; numeric tensor payloads remain separate."""
        return {
            "profile": PROFILE,
            "scope": self.scope,
            "original_shape": list(self.original_shape),
            "input_dtype": str(self.input_dtype),
            "padded_last_dim": self.padded_last_dim,
            "right_padding": self.padded_last_dim - self.original_shape[-1],
            "block_size": BLOCK_SIZE,
            "data_format": "E2M1",
            "block_scale_format": "E4M3FN",
            "upper_scale_format": "FP32",
            "rounding": "nearest_ties_to_even",
            "zero_scope_rule": "upper_encode=upper_decode=1; block_scale=0",
            "layout": "linear_unpacked_uint8_nibbles; no_hardware_swizzle",
            "stochastic_rounding": False,
            "random_hadamard_transform": False,
            "native_kernel_validated": False,
        }


@torch.no_grad()
def quantize(x: torch.Tensor, scope: str = "token") -> NVFP4Tensor:
    """Quantize finite CPU input using the documented deterministic profile.

    Padding extends only the last dimension to a multiple of 16 using zeros.
    Upper scales and all arithmetic before the E4M3/E2M1 conversions are FP32.
    Extremely small blocks can round to zero E4M3 scale, intentionally losing
    those blocks; this is format behavior, not a replacement of invalid input.
    """
    _require_cpu_float(x, "NVFP4 input")
    if scope not in ("token", "tensor"):
        raise ValueError("scope must be 'token' or 'tensor'")
    if x.ndim < 1 or x.numel() == 0 or x.shape[-1] == 0:
        raise ValueError("NVFP4 input must be nonempty and have a last dimension")
    shape = tuple(x.shape)
    width = shape[-1]
    padded_width = ((width + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    rows = x.detach().to(torch.float32).reshape(-1, width)
    padded = F.pad(rows, (0, padded_width - width))
    blocks = padded.reshape(rows.shape[0], -1, BLOCK_SIZE)
    amax = rows.abs().amax(dim=-1, keepdim=True) if scope == "token" else rows.abs().amax()
    maximum = torch.finfo(torch.float32).max
    # Keep the explicit reciprocal order of the TE-reference implementation.
    # The all-zero rule differs only in metadata scaling of an exactly zero row.
    # Explicit torch.div avoids Python scalar/Tensor reciprocal rewrites.
    global_encode = torch.div(torch.tensor(FP8_MAX * FP4_MAX, dtype=torch.float32), amax).clamp_max(maximum)
    global_encode = torch.where(amax == 0, torch.ones_like(global_encode), global_encode)
    global_decode = torch.div(torch.ones_like(global_encode), global_encode)
    encode_broadcast = global_encode.unsqueeze(-1) if scope == "token" else global_encode
    decode_broadcast = global_decode.unsqueeze(-1) if scope == "token" else global_decode
    reciprocal_six = torch.div(torch.tensor(1.0, dtype=torch.float32), torch.tensor(FP4_MAX, dtype=torch.float32))
    global_encode_multiplier = encode_broadcast * reciprocal_six
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale_before_cast = (block_amax * global_encode_multiplier).clamp(0.0, FP8_MAX)
    fp8_scales = scale_before_cast.to(torch.float8_e4m3fn)
    decoded_scales = fp8_scales.float()
    block_decode = decoded_scales * decode_broadcast
    block_encode = torch.div(torch.ones_like(block_decode), block_decode).clamp_max(maximum)
    # Overflow can occur while scaling a value in a zero-scale block. Saturating
    # to E2M1's finite range is intentional; its decoded value is still zero.
    normalized = (blocks * block_encode).clamp(-FP4_MAX, FP4_MAX)
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("NVFP4 normalization produced a nonfinite value")
    codes = _e2m1_codes(normalized).reshape(rows.shape[0], padded_width)
    return NVFP4Tensor(
        codes=codes,
        block_scale_bytes=fp8_scales.squeeze(-1).contiguous().view(torch.uint8),
        global_encode_scale=global_encode,
        global_decode_scale=global_decode,
        original_shape=shape,
        input_dtype=x.dtype,
        padded_last_dim=padded_width,
        scope=scope,
    )


@torch.no_grad()
def dequantize(q: NVFP4Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Decode logical NVFP4 values with FP32 products, then optionally cast."""
    if not isinstance(q, NVFP4Tensor):
        raise TypeError("q must be an NVFP4Tensor")
    if dtype not in _FLOAT_DTYPES:
        raise TypeError("output dtype must be float16, bfloat16, or float32")
    rows = prod(q.original_shape[:-1])
    if q.scope not in ("token", "tensor"):
        raise ValueError("invalid NVFP4 scope")
    if q.codes.device.type != "cpu" or q.codes.dtype != torch.uint8:
        raise ValueError("NVFP4 codes must be CPU uint8")
    if q.codes.shape != (rows, q.padded_last_dim) or bool((q.codes > 15).any()):
        raise ValueError("invalid NVFP4 code shape or nibble value")
    if q.block_scale_bytes.device.type != "cpu" or q.block_scale_bytes.dtype != torch.uint8:
        raise ValueError("NVFP4 block-scale bytes must be CPU uint8")
    if q.block_scale_bytes.shape != (rows, q.padded_last_dim // BLOCK_SIZE):
        raise ValueError("invalid NVFP4 block-scale shape")
    scales = q.block_scales
    if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
        raise ValueError("NVFP4 block scales must be finite and nonnegative")
    _require_cpu_float(q.global_decode_scale, "NVFP4 global decode scale")
    expected_upper_shape = (rows, 1) if q.scope == "token" else ()
    if q.global_decode_scale.shape != expected_upper_shape or bool((q.global_decode_scale <= 0).any()):
        raise ValueError("invalid NVFP4 global decode scale")
    data = _decode_e2m1_codes(q.codes).reshape(rows, -1, BLOCK_SIZE)
    upper = q.global_decode_scale.unsqueeze(-1) if q.scope == "token" else q.global_decode_scale
    reconstructed = (data * scales.unsqueeze(-1)) * upper
    result = reconstructed.reshape(rows, q.padded_last_dim)[:, :q.original_shape[-1]].reshape(q.original_shape)
    result = result.to(dtype)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("NVFP4 reconstruction/cast produced a nonfinite value")
    return result


class _FakeQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scope: str) -> torch.Tensor:
        return dequantize(quantize(x, scope), dtype=x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Explicit identity STE, not a claim to reproduce quantized backward
        # matmuls or derivatives of scale/round/clip operations.
        return grad_output, None


def fake_quantize_ste(x: torch.Tensor, scope: str = "token") -> torch.Tensor:
    """NVFP4 QDQ forward in input dtype and an explicitly identity STE backward."""
    return _FakeQuantizeSTE.apply(x, scope)
