"""Keep stock Qwen3 BF16 projections, weight loaders and RoPE; replace FA only."""

from types import MethodType

import torch
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.models.qwen3 import Qwen3ForCausalLM

from slime_plugins.nvfp4_fa.serving import radix_forward


class NVFP4FAQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        torch.backends.cuda.matmul.allow_tf32 = False
        layers = [m for m in self.modules() if isinstance(m, RadixAttention)]
        if len(layers) != config.num_hidden_layers or len(layers) != 36 or self.quant_config is not None:
            raise ValueError("Expected all 36 unquantized Qwen3-4B layers on this rollout engine")
        for layer in layers:
            if (layer.tp_q_head_num, layer.tp_k_head_num, layer.head_dim) != (32, 8, 128):
                raise ValueError("This rollout configuration requires Qwen3-4B with TP=1")
            layer.forward = MethodType(radix_forward, layer)


EntryClass = NVFP4FAQwen3ForCausalLM
