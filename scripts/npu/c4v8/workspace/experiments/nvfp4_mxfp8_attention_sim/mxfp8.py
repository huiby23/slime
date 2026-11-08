"""CPU MXFP8 E4M3/E8M0 reference, using NVIDIA-documented ceil-amax scaling.

Each last-axis block of 32 elements shares s=2**ceil(log2(amax/448)).
This is a specified conversion recipe, not the only conversion allowed by OCP.
OCP's suggested floor(log2(amax))-8 conversion can yield different results.
The code has not been compared with Blackwell, Transformer Engine, or Ascend.
"""
from dataclasses import dataclass
from math import prod
import torch
import torch.nn.functional as F

BLOCK_SIZE = 32
FP8_MAX = 448.0
PROFILE = 'mxfp8_e4m3fn_e8m0_nvidia_documented_ceil_amax_rne_cpu_v1'
DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def require_input(x):
    if not isinstance(x, torch.Tensor) or x.device.type != 'cpu':
        raise ValueError('This numerical reference accepts CPU tensors only')
    if x.dtype not in DTYPES or x.ndim < 1 or x.numel() == 0 or x.shape[-1] == 0:
        raise ValueError('Expected a nonempty floating tensor with a last axis')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('MXFP8 data must be finite; structural masks live outside the format')


@dataclass(frozen=True)
class MXFP8Tensor:
    payload_bytes: torch.Tensor
    scale_bytes: torch.Tensor
    original_shape: tuple
    input_dtype: torch.dtype
    padded_last_dim: int

    @property
    def scales(self):
        return torch.pow(torch.tensor(2.0, dtype=torch.float64),
                         self.scale_bytes.to(torch.int16) - 127).to(torch.float32)

    @property
    def metadata(self):
        return dict(profile=PROFILE, payload_format='E4M3FN', scale_format='E8M0',
            block_size=32, scale_axis='last dimension independently per row',
            scale_rule='2**ceil(log2(amax/448)), clamped to E8M0 exponents [-127,127]',
            scale_calculation='FP64 frexp of FP32 amax/448 avoids exponent-boundary log2 rounding',
            rounding='nearest ties to even', zero_block='scale=1, zero payload',
            tail='zero-pad last axis to32; never cross a row',
            underflow='finite tiny inputs may round to zero at the minimum positive scale',
            reconstruction_overflow='raise; never silently replace a nonfinite result',
            original_shape=list(self.original_shape), padded_last_dim=self.padded_last_dim,
            layout='linear logical bytes, no device ABI/swizzle',
            stochastic_rounding=False, native_kernel_validated=False,
            ocp_suggested_floor_conversion=False)


@torch.no_grad()
def quantize(x):
    require_input(x)
    shape = tuple(x.shape); width = shape[-1]
    padded_width = ((width + 31)//32)*32
    rows = x.detach().float().reshape(-1, width)
    padded = F.pad(rows, (0, padded_width-width))
    blocks = padded.reshape(rows.shape[0], -1, 32)
    amax = blocks.abs().amax(-1)
    # E8M0 has no zero. All-zero blocks use a conventional scale of one.
    ratio = amax.double()/448.0
    mantissa, exponent = torch.frexp(ratio)
    ceil_exponent = exponent - (mantissa == 0.5).to(exponent.dtype)
    ceil_exponent = torch.where(amax == 0, torch.zeros_like(exponent), ceil_exponent)
    ceil_exponent = ceil_exponent.clamp(-127, 127)
    scale_bytes = (ceil_exponent+127).to(torch.uint8)
    scales = torch.pow(torch.tensor(2.0, dtype=torch.float64), ceil_exponent).float()
    normalized = torch.div(blocks, scales.unsqueeze(-1))
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError('MXFP8 normalization produced nonfinite data')
    # Clamp is explicit saturating conversion; ordinary in-range FP32 amax
    # fits by construction. The minimum-scale edge may deliberately underflow.
    payload = normalized.clamp(-448,448).to(torch.float8_e4m3fn)
    return MXFP8Tensor(payload.contiguous().view(torch.uint8).reshape(rows.shape[0],padded_width),
        scale_bytes, shape, x.dtype, padded_width)


@torch.no_grad()
def dequantize(value, dtype=torch.float32):
    if not isinstance(value, MXFP8Tensor) or dtype not in DTYPES:
        raise ValueError('Invalid MXFP8 value or requested floating dtype')
    rows = prod(value.original_shape[:-1])
    if (value.payload_bytes.device.type != 'cpu' or value.payload_bytes.dtype != torch.uint8
            or value.payload_bytes.shape != (rows,value.padded_last_dim)
            or value.scale_bytes.device.type != 'cpu' or value.scale_bytes.dtype != torch.uint8
            or value.scale_bytes.shape != (rows,value.padded_last_dim//32)
            or bool((value.scale_bytes == 255).any())):
        raise ValueError('Invalid logical payload/scale layout or E8M0 NaN code')
    data = value.payload_bytes.contiguous().view(torch.float8_e4m3fn).float()
    if not bool(torch.isfinite(data).all()):
        raise ValueError('Logical FP8 payload contains NaN')
    reconstructed = data.reshape(rows,-1,32)*value.scales.unsqueeze(-1)
    output = reconstructed.reshape(rows,value.padded_last_dim)[:, :value.original_shape[-1]]
    output = output.reshape(value.original_shape).to(dtype)
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError('MXFP8 reconstruction/cast overflow; no silent fallback')
    return output


class _QDQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,allow_masked_negative_infinity):
        if x.device.type != 'cpu':
            raise ValueError('CPU-only reference; NPU/CUDA integration is unvalidated')
        if bool(torch.isnan(x).any()) or bool(torch.isposinf(x).any()):
            raise ValueError('NaN/+infinity are not structural attention masks')
        mask = torch.isneginf(x)
        if bool(mask.any()) and not allow_masked_negative_infinity:
            raise ValueError('Negative infinity requires an explicit structural-mask policy')
        finite_input = torch.where(mask, torch.zeros_like(x), x)
        result = dequantize(quantize(finite_input), dtype=x.dtype)
        ctx.save_for_backward(mask)
        # A mask sentinel is out-of-band metadata, not a representable E4M3 value.
        return result.masked_fill(mask, -float('inf'))

    @staticmethod
    def backward(ctx,grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output.masked_fill(mask,0), None


def fake_quantize_ste(x, *, allow_masked_negative_infinity=False):
    """MXFP8 QDQ with identity STE on finite inputs, zero derivative for masks."""
    return _QDQ.apply(x, allow_masked_negative_infinity)
