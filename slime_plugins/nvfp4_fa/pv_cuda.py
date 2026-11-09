import torch
import triton
import triton.language as tl


@triton.jit
def _block_pv(P, V, Out, Q, G, D: tl.constexpr, W: tl.constexpr):
    row = tl.program_id(0)
    bh = row // (Q * G)
    group = row % G
    col = tl.arange(0, W)
    value = tl.full((W,), 0, tl.float32)
    for k in tl.static_range(16):
        prob = tl.load(P + row * 16 + k)
        v = tl.load(V + ((bh * G + group) * 16 + k) * D + col, mask=col < D, other=0.0)
        value = value + prob * v
    tl.store(Out + row * D + col, value, mask=col < D)


def block_pv(p, v):
    b, h, q, g, _ = p.shape
    d = v.size(-1)
    out = p.new_empty((b, h, q, g, d))
    _block_pv[(b * h * q * g,)](p, v, out, q, g, d, triton.next_power_of_2(d), enable_fp_fusion=False)
    return out
