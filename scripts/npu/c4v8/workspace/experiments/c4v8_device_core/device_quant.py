"""Portable NVFP4/MXFP8 numerical QDQ, using only FP32/integer torch ops.

This is an experimental numerical implementation, not a native low-bit kernel.
There is no float8 tensor conversion and no device-to-CPU tensor copy. Logical
bytes can be requested for parity tests, but runtime QDQ operates on values.
The backward is identity STE (zero on structural masks), NOT quantized backward.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from exact_fp32 import divide_positive


FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
NVFP4_PROFILE = 'nvfp4_e2m1_e4m3fn_rne_per_row_fp32_v1'
MXFP8_PROFILE = 'mxfp8_e4m3fn_e8m0_nvidia_ceil_amax_rne_fp32_v1'


def _input(x, check):
    if not isinstance(x, torch.Tensor) or x.dtype not in FLOAT_DTYPES:
        raise TypeError('Expected float32/float16/bfloat16 tensor')
    if x.ndim < 1 or x.numel() == 0 or x.shape[-1] == 0:
        raise ValueError('Expected nonempty tensor with a last dimension')
    if x.device.type not in ('cpu', 'npu', 'privateuseone'):
        raise ValueError('Only CPU and explicit Ascend experimental devices accepted')
    if check and not bool(torch.isfinite(x).all()):
        raise ValueError('Ordinary quantization data must be finite')


def _power_two(exponent):
    # Ascend native pow(2,e) is approximate even at integer e. Encoding the
    # IEEE754 bits is exact and prevents those errors changing halfway RNE.
    e = exponent.to(torch.int32)
    bits = torch.where(e == -127, torch.full_like(e, 1 << 22),
                       torch.bitwise_left_shift(e + 127, 23))
    return bits.contiguous().view(torch.float32)


def _signbit(x):
    # Native NPU signbit currently treats negative zero as positive.
    return x.float().contiguous().view(torch.int32) < 0


def _copy_sign(magnitude, source):
    absolute_bits = torch.bitwise_and(magnitude.float().contiguous().view(torch.int32), 0x7fffffff)
    sign_bits = torch.bitwise_and(source.float().contiguous().view(torch.int32), -2147483648)
    return torch.bitwise_or(absolute_bits, sign_bits).contiguous().view(torch.float32)


@torch.no_grad()
def round_e4m3fn(x):
    """Finite saturating RNE E4M3FN decoded values, no float8 cast.

    Arithmetic nonfinite input propagates as NaN instead of being silently
    converted to a finite saturation value. Signed zeros are preserved.
    """
    x = x.float()
    absolute = x.abs().clamp_max(448.0)
    safe = torch.where(absolute > 0, absolute, torch.ones_like(absolute))
    exponent = torch.floor(torch.log2(safe))
    step_exponent = (exponent - 3).clamp_min(-9)
    step = _power_two(step_exponent)
    rounded = torch.round(absolute * _power_two(-step_exponent)) * step
    rounded = rounded.clamp_max(448.0)
    result = _copy_sign(rounded, x)
    return torch.where(torch.isfinite(x), result, torch.full_like(result, float('nan')))


@torch.no_grad()
def e4m3fn_bytes(x):
    """Encode the rounded values into linear logical E4M3FN bytes."""
    y = round_e4m3fn(x)
    absolute = y.abs()
    safe = torch.where(absolute > 0, absolute, torch.ones_like(absolute))
    exponent = torch.floor(torch.log2(safe))
    normal = (exponent + 7) * 8 + torch.round(absolute * _power_two(-exponent) * 8) - 8
    subnormal = torch.round(absolute * 512)
    magnitude = torch.where(absolute < 2.0**-6, subnormal, normal).to(torch.int32)
    magnitude = torch.where(torch.isfinite(y), magnitude, torch.full_like(magnitude, 127))
    sign = _signbit(x).to(torch.int32) * 128
    return (magnitude + sign).to(torch.uint8)


@torch.no_grad()
def round_e2m1(x, *, return_codes=False):
    """Saturating E2M1 RNE using comparisons; no bucketize/table gather."""
    absolute = x.abs().clamp_max(6.0)
    step = torch.where(absolute < 2, 0.5, torch.where(absolute < 4, 1.0, 2.0))
    inverse_step = torch.where(absolute < 2, 2.0, torch.where(absolute < 4, 1.0, 0.5))
    value = torch.round(absolute * inverse_step) * step
    value = _copy_sign(value, x)
    value = torch.where(torch.isnan(x), torch.full_like(value, float('nan')), value)
    if return_codes:
        code = torch.zeros_like(absolute, dtype=torch.int32)
        for index, midpoint in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
            code = code + ((absolute > midpoint) | ((absolute == midpoint) & (index % 2 == 1))).to(torch.int32)
        return value, (code + _signbit(x).to(torch.int32) * 8).to(torch.uint8)
    return value


@dataclass
class QuantizedValues:
    values: torch.Tensor
    payload_bytes: torch.Tensor | None
    scale_bytes: torch.Tensor | None
    upper_encode: torch.Tensor | None = None
    upper_decode: torch.Tensor | None = None


def _blocks(x, width):
    shape = tuple(x.shape)
    last = shape[-1]
    padded = ((last + width - 1) // width) * width
    rows = x.detach().float().reshape(-1, last)
    data = F.pad(rows, (0, padded-last)).reshape(rows.shape[0], -1, width)
    return shape, last, padded, rows, data


@torch.no_grad()
def nvfp4_values(x, *, check=True, logical_bytes=False):
    _input(x, check)
    shape, last, padded, rows, blocks = _blocks(x, 16)
    maximum = torch.finfo(torch.float32).max
    amax = rows.abs().amax(-1, keepdim=True)
    upper_encode = divide_positive(2688.0, amax).clamp_max(maximum)
    upper_encode = torch.where(amax == 0, torch.ones_like(upper_encode), upper_encode)
    upper_decode = divide_positive(1.0, upper_encode)
    # Preserve the validated CPU reference's explicit FP32 operation order.
    reciprocal_six = torch.full_like(upper_encode, 1.0/6.0)
    multiplier = upper_encode.unsqueeze(-1) * reciprocal_six.unsqueeze(-1)
    block_amax = blocks.abs().amax(-1, keepdim=True)
    before_scale_round = (block_amax * multiplier).clamp(0, 448)
    block_scale = round_e4m3fn(before_scale_round)
    block_decode = block_scale * upper_decode.unsqueeze(-1)
    block_encode = divide_positive(1.0, block_decode).clamp_max(maximum)
    normalized = (blocks * block_encode).clamp(-6, 6)
    if logical_bytes:
        payload_values, codes = round_e2m1(normalized, return_codes=True)
    else:
        payload_values = round_e2m1(normalized)
        codes = None
    reconstructed = (payload_values * block_scale) * upper_decode.unsqueeze(-1)
    # If unchecked ordinary inputs were nonfinite, propagate an invalid row.
    invalid_row = ~torch.isfinite(rows).all(-1, keepdim=True)
    output = reconstructed.reshape(rows.shape[0], padded)[:, :last]
    output = torch.where(invalid_row, torch.full_like(output, float('nan')), output).reshape(shape)
    if check and not bool(torch.isfinite(output).all()):
        raise FloatingPointError('NVFP4 reconstruction is nonfinite')
    return QuantizedValues(output,
        codes.reshape(rows.shape[0], padded) if codes is not None else None,
        e4m3fn_bytes(before_scale_round).squeeze(-1) if logical_bytes else None,
        upper_encode, upper_decode)


@torch.no_grad()
def mxfp8_values(x, *, check=True, logical_bytes=False):
    _input(x, check)
    shape, last, padded, rows, blocks = _blocks(x, 32)
    amax = blocks.abs().amax(-1)
    # FP32-only ceil(log2(amax/448)): choose a candidate binade and compare
    # against its EXACT power-of-two multiple of448. This avoids underflow in
    # amax/448 and avoids a log2 rounding error at the actual 448*2**e boundary.
    safe_amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    candidate = (torch.floor(torch.log2(safe_amax)) - 8).clamp(-127, 119)
    threshold = _power_two(candidate) * 448.0
    exponent = candidate + (amax > threshold).to(torch.float32)
    exponent = exponent.clamp(-127, 127)
    exponent = torch.where(amax == 0, torch.zeros_like(exponent), exponent)
    scales = _power_two(exponent)
    normalized = blocks * _power_two(-exponent).unsqueeze(-1)
    payload_values = round_e4m3fn(normalized)
    reconstructed = payload_values * scales.unsqueeze(-1)
    output = reconstructed.reshape(rows.shape[0], padded)[:, :last].reshape(shape)
    if check and not bool(torch.isfinite(output).all()):
        raise FloatingPointError('MXFP8 reconstruction/cast overflow')
    return QuantizedValues(output,
        e4m3fn_bytes(normalized).reshape(rows.shape[0], padded) if logical_bytes else None,
        (exponent + 127).to(torch.uint8) if logical_bytes else None)


class _NVFP4STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, check):
        return nvfp4_values(x, check=check).values.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _MXFP8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, allow_masked_negative_infinity, check):
        mask = torch.isneginf(x) if allow_masked_negative_infinity else torch.zeros_like(x, dtype=torch.bool)
        ordinary = torch.where(mask, torch.zeros_like(x), x)
        result = mxfp8_values(ordinary, check=check).values.to(x.dtype)
        ctx.save_for_backward(mask)
        return result.masked_fill(mask, -float('inf'))

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output.masked_fill(mask, 0), None, None


def nvfp4_qdq(x, *, check=True):
    return _NVFP4STE.apply(x, check)


def mxfp8_qdq(x, *, allow_masked_negative_infinity=False, check=True):
    return _MXFP8STE.apply(x, allow_masked_negative_infinity, check)


def metadata():
    return dict(nvfp4=NVFP4_PROFILE, mxfp8=MXFP8_PROFILE,
                float8_cast=False, cpu_tensor_copies=False, native_lowbit_arithmetic=False,
                fp32_scale_division='RNE positive FP32 division via integer significands; avoids native reciprocal approximation moving FP4 ties',
                backward='identity STE in incoming gradient dtype; structural-mask derivative zero',
                quantized_backward=False, npu_validated=False, production_ready=False)
