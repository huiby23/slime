"""Single-query NPU decode. Tile boundaries match the full causal attention."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from hif4_checks import check_invalid
from hif4_quant import GROUP, quantize
from hif4_triton import _bf16_rne


@triton.jit(do_not_specialize=['TILES'])
def _update_states(State, PV, Numerator, Denominator, TILES, DIM: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, DIM)
    one = tl.arange(0, 1)
    num = tl.full((DIM,), 0., tl.float32)
    den = tl.full((1,), 0., tl.float32)
    for tile in range(TILES):
        index = row * TILES + tile
        alpha = tl.load(State + index * 32 + one)
        local_sum = tl.load(State + index * 32 + 1 + one)
        contribution = tl.load(PV + index * DIM + lane)
        den = _bf16_rne(_bf16_rne(den * alpha) + local_sum)
        num = _bf16_rne(_bf16_rne(num * alpha) + _bf16_rne(contribution))
    tl.store(Numerator + row * DIM + lane, num)
    tl.store(Denominator + row * 32 + one, den)


def _bf(x):
    return x.bfloat16().float()


def decode(q, k, v):
    """All keys are visible to the final query; preserve 128-key online updates."""
    batch, heads, _, dim = q.shape
    count = k.size(2)
    block = 128
    tiles = triton.cdiv(count, block)
    pad = tiles * block - count
    repeat = heads // k.size(1)
    rows = batch * heads
    query = quantize(q)
    key = quantize(k)
    value = quantize(v.transpose(-1, -2)).transpose(-1, -2)
    if repeat > 1:
        key = key.repeat_interleave(repeat, 1)
        value = value.repeat_interleave(repeat, 1)

    scores = _bf((query @ key.transpose(-1, -2)) * (dim ** -.5))
    scores = F.pad(scores.reshape(rows, count), (0, pad), value=-torch.inf).reshape(rows, tiles, block)
    local_max = scores.amax(-1)
    # Prefix max over <=64 past tiles. The matrix is tiny relative to KV.
    positions = torch.arange(tiles, device=q.device)
    future = positions[None, :] > positions[:, None]
    maximum = local_max[:, None, :].masked_fill(future, -torch.inf).amax(-1)
    previous = F.pad(maximum[:, :-1], (1, 0), value=-torch.inf)
    alpha = _bf(_bf(previous - maximum).exp())
    prob = _bf(_bf(scores - maximum.unsqueeze(-1)).exp())
    local_sum = _bf(prob.sum(-1))
    p4 = quantize(prob)
    value = F.pad(value.reshape(rows, count, dim), (0, 0, 0, pad)).reshape(rows, tiles, block, dim)

    # The shared prefill routine separates the final query's diagonal 64-key
    # group from its preceding groups, then adds the two FP32 contributions.
    last_p = p4[:, -1].reshape(rows, 2, 1, GROUP)
    last_v = value[:, -1].reshape(rows, 2, GROUP, dim)
    pair = (last_p @ last_v).squeeze(-2)
    last = (pair[:, :1] + pair[:, 1:])
    if tiles > 1:
        earlier = (p4[:, :-1, None, :] @ value[:, :-1]).squeeze(-2)
        contributions = torch.cat((earlier, last), dim=1)
    else:
        contributions = last
    state = F.pad(torch.stack((alpha, local_sum), -1), (0, 30)).contiguous()
    num = torch.empty((rows, dim), device=q.device, dtype=torch.float32)
    den = torch.empty((rows, 32), device=q.device, dtype=torch.float32)
    _update_states[(rows,)](state, contributions.contiguous(), num, den, tiles, dim, enable_fp_fusion=False)
    output = (num / den[:, :1]).bfloat16().reshape(batch, heads, 1, dim)
    check_invalid(torch.isfinite(output).all().logical_not(), "Nonfinite HiF4 decode")
    return output
