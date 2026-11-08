"""Fused finite-input BF16 -> HiF4 -> FP32 conversion for Ascend Triton."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bf16_rne(value):
    # Keep the BF16 conversion visible to the Ascend compiler. A narrowing
    # cast immediately widened back to FP32 can otherwise disappear at a tie.
    bits = value.to(tl.int32, bitcast=True)
    rounded = (bits + 32767 + ((bits >> 16) & 1)) & -65536
    special = (bits & 0x7f800000) == 0x7f800000
    return tl.where(special, bits, rounded).to(tl.float32, bitcast=True)


@triton.jit
def _convert(X, Y, INVALID, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i).to(tl.float32)
    # W16A16 operands carry BF16 values; the trainer replays them in FP32.
    x = _bf16_rne(x)
    mag = tl.abs(x)
    peak4 = tl.max(tl.reshape(mag, (BLOCK // 4, 4)), 1)
    peak8 = tl.max(tl.reshape(peak4, (BLOCK // 8, 2)), 1)
    peak64 = tl.max(tl.reshape(peak8, (BLOCK // 64, 8)), 1)
    target = _bf16_rne(peak64 * 0.142578125)
    bits = tl.minimum(tl.maximum(target, 3.552713678800501e-15), 49152.).to(tl.int32, bitcast=True)
    base_bits = (bits + 0xFFFFF + ((bits >> 21) & 1)) & -0x200000
    base = base_bits.to(tl.float32, bitcast=True)
    # Exact rounded BF16 reciprocal significands for E6M2 mantissas
    # 1, 1.25, 1.5 and 1.75. Scale by an exactly encoded power of two.
    mant = (base_bits >> 21) & 3
    inv_pow2 = ((254 - (base_bits >> 23)) << 23).to(tl.float32, bitcast=True)
    inv = inv_pow2 * tl.where(mant == 0, 1., tl.where(mant == 1, .80078125,
                              tl.where(mant == 2, .66796875, .5703125)))
    i8 = tl.reshape(tl.broadcast_to(inv[:, None], (BLOCK // 64, 8)), (BLOCK // 8,))
    f8 = tl.where(peak8 * i8 >= 4., 2., 1.)
    inv4 = tl.reshape(tl.broadcast_to(i8[:, None], (BLOCK // 8, 2)), (BLOCK // 4,))
    f8x2 = tl.reshape(tl.broadcast_to(f8[:, None], (BLOCK // 8, 2)), (BLOCK // 4,))
    f4 = tl.where(peak4 * inv4 >= 2. * f8x2, 2., 1.) * f8x2
    unscale = tl.reshape(tl.broadcast_to((inv4 * tl.where(f4 == 1., 4., tl.where(f4 == 2., 2., 1.)))[:, None], (BLOCK // 4, 4)), (BLOCK,))
    quant = tl.minimum(tl.floor(mag * unscale + .5), 7.)
    scale4 = tl.reshape(tl.broadcast_to(base[:, None], (BLOCK // 64, 16)), (BLOCK // 4,))
    units = tl.reshape(tl.broadcast_to((scale4 * .25 * f4)[:, None], (BLOCK // 4, 4)), (BLOCK,))
    decoded = (quant * units).to(tl.int32, bitcast=True) | (x.to(tl.int32, bitcast=True) & -2147483648)
    tl.store(Y + i, decoded.to(tl.float32, bitcast=True))
    nonfinite = (mag.to(tl.int32, bitcast=True) & 0x7f800000) == 0x7f800000
    tl.store(INVALID + pid, tl.max(nonfinite.to(tl.int32), 0))


def ascend_qdq(x):
    if x.dtype not in (torch.bfloat16, torch.float32) or not x.ndim or not x.numel():
        raise ValueError("Expected a nonempty BF16/FP32 tensor")
    padded = F.pad(x, (0, -x.size(-1) % 64)).contiguous()
    block = 256
    packed = F.pad(padded.reshape(-1), (0, -padded.numel() % block))
    programs = packed.numel() // block
    out = torch.empty_like(packed, dtype=torch.float32)
    flags = torch.empty(programs, dtype=torch.int32, device=x.device)
    _convert[(programs,)](packed, out, flags, block, enable_fp_fusion=False)
    if bool((flags != 0).any()):
        raise FloatingPointError("Nonfinite BF16 input to training HiF4 converter")
    return out[:padded.numel()].reshape(padded.shape)[..., :x.size(-1)]
