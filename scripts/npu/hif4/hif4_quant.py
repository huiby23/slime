"""BF16 to HiFloat4 simulation, with a straight-through gradient.

The last tensor axis is the contraction axis. Each 64 values have one E6M2
base scale, eight micro-exponents for 8-value subgroups and sixteen for 4-value
subgroups. Scale rounding is nearest-even. S1P2 rounds ties away from zero, as
in GCC-HiFloat's published NumPy To_HiFX reference.
"""

import torch
import torch.nn.functional as F

GROUP = 64
INV_SEVEN_BF16 = 0.142578125
MIN_SCALE = 2.0 ** -48
MAX_SCALE = 49152.0


def _padded_groups(x):
    if x.device.type not in ("cpu", "npu", "privateuseone"):
        raise ValueError("HiF4 simulation requires a CPU or Ascend tensor")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Expected BF16 activation or FP32 replay activation")
    if x.ndim == 0 or not x.numel():
        raise ValueError("Expected a nonempty contraction axis")
    values = x.to(torch.bfloat16).float()
    return F.pad(values, (0, -x.shape[-1] % GROUP)).reshape(-1, 16, 4)


def _e6m2(scale):
    # Clamp to finite normal E6M2, then round the FP32 significand to two
    # fraction bits. This also rounds the upper midpoint into the next binade.
    bits = scale.clamp(MIN_SCALE, MAX_SCALE).contiguous().view(torch.int32)
    rounded = (bits + 0xFFFFF + ((bits >> 21) & 1)) & -0x200000
    return rounded.view(torch.float32)


@torch.no_grad()
def reference_qdq(x):
    """Finite-input software conversion, returning reconstructed FP32 values."""
    groups = _padded_groups(x)
    if not bool(torch.isfinite(groups).all()):
        raise FloatingPointError("Nonfinite BF16 input to training HiF4 converter")
    peaks4 = groups.abs().amax(dim=-1)
    peaks8 = peaks4.reshape(-1, 8, 2).amax(dim=-1)
    peak = peaks8.amax(dim=-1, keepdim=True)
    scale = _e6m2((peak * INV_SEVEN_BF16).bfloat16().float())
    inverse = scale.reciprocal().bfloat16().float()
    factor8 = torch.where(peaks8 * inverse >= 4, 2., 1.).repeat_interleave(2, dim=-1)
    factor4 = torch.where(peaks4 * inverse / factor8 >= 2, 2., 1.)
    local = (factor8 * factor4).unsqueeze(-1)
    magnitude = (groups.abs() * inverse.unsqueeze(-1) * (4. / local) + .5).floor().clamp_max(7.)
    restored = torch.copysign(magnitude, groups) * (.25 * scale.unsqueeze(-1) * local)
    return restored.reshape(*x.shape[:-1], -1)[..., :x.shape[-1]]


class _HiF4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.device.type == "cpu":
            return reference_qdq(x)
        if x.device.type not in ("npu", "privateuseone"):
            raise ValueError("Unsupported HiF4 device")
        from hif4_triton import ascend_qdq
        return ascend_qdq(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


quantize = _HiF4.apply
