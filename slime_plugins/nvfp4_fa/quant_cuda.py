"""Software E2M1/E4M3 conversion on Ampere. Each program owns one upper scale."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qdq(X, Y, N: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, WIDTH)
    x = tl.load(X + row * N + col, mask=col < N, other=0.0).to(tl.float32)
    blocks = tl.reshape(tl.abs(x), (WIDTH // 16, 16))
    peaks = tl.max(blocks, 1)
    upper_peak = tl.max(peaks, 0)
    upper = tl.where(upper_peak == 0.0, 1.0, tl.div_rn(upper_peak, 2688.0))
    s = tl.minimum(tl.div_rn(peaks, 6.0 * upper), 448.0)
    # Three mantissa bits for normal E4M3; subnormals have fixed spacing 2^-9.
    bits = s.to(tl.int32, bitcast=True)
    normal = ((bits + 0x7FFFF + ((bits >> 20) & 1)) & -0x100000).to(tl.float32, bitcast=True)
    scaled = s * 512.0
    low = scaled.to(tl.int32)
    fraction = scaled - low.to(tl.float32)
    add = (fraction > 0.5) | ((fraction == 0.5) & ((low & 1) != 0))
    sub = (low + add.to(tl.int32)).to(tl.float32) * (1.0 / 512.0)
    scale = tl.where(s < 0.015625, sub, normal) * upper
    a = tl.div_rn(blocks, tl.where(scale == 0.0, 1.0, scale)[:, None])
    # Alternating strict/inclusive comparisons implement even ties at each midpoint.
    mag = (
        (a > 0.25).to(tl.float32) + (a >= 0.75).to(tl.float32) + (a > 1.25).to(tl.float32) + (a >= 1.75).to(tl.float32)
    ) * 0.5
    mag += (a > 2.5).to(tl.float32) + (a >= 3.5).to(tl.float32) + 2.0 * (a > 5.0).to(tl.float32)
    result = tl.reshape(mag * scale[:, None], (WIDTH,))
    signed = result.to(tl.int32, bitcast=True) | (x.to(tl.int32, bitcast=True) & -2147483648)
    tl.store(Y + row * N + col, signed.to(tl.float32, bitcast=True), mask=col < N)


def qdq(blocks):
    x = blocks.contiguous()
    n = x.size(-2) * 16
    out = torch.empty_like(x)
    _qdq[(x.numel() // n,)](x, out, n, triton.next_power_of_2(n), enable_fp_fusion=False)
    return out
