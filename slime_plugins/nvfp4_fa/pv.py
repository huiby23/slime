"""A fixed FP32 reduction order for each 16-key NVFP4 PV block.

Different GEMM shapes otherwise round exact BF16 midpoints to opposite sides
when the scheduler combines requests. The CUDA kernel runs the 16 ordered products and sums in one launch.
"""

import torch


class BlockPV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, v):
        # p=[B,H,Q,G,16]; v=[B,H,G,16,D]. No query-dependent V expansion.
        ctx.save_for_backward(p, v)
        p = p.contiguous()
        if p.is_cuda:
            from .pv_cuda import block_pv

            return block_pv(p, v.contiguous())
        out = p.new_zeros((*p.shape[:-1], v.size(-1)))
        for i in range(16):
            out = out + p[..., i : i + 1] * v[:, :, None, :, i, :]
        return out

    @staticmethod
    def backward(ctx, grad):
        p, v = ctx.saved_tensors
        g = grad.transpose(2, 3)
        dp = (g @ v.transpose(-1, -2)).transpose(2, 3)
        dv = p.transpose(2, 3).transpose(-1, -2) @ g
        return dp, dv


pv = BlockPV.apply
