"""Instrumented model, selected explicitly by frozen alignment jobs only."""

import torch

from .qwen3 import NVFP4SimQwen3ForCausalLM
from ..diagnostic_trace import canonical_name, capture


class NVFP4AlignmentQwen3ForCausalLM(NVFP4SimQwen3ForCausalLM):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._alignment_config = config
        self._alignment_active = False
        self._alignment_done = False
        self._alignment_records = {}
        for name, module in self.named_modules(remove_duplicate=False):
            tag = canonical_name(name)
            if tag is None:
                continue
            method = module.quant_method
            original = method.apply

            def observed(layer, x, bias=None, *, _original=original, _tag=tag):
                output = _original(layer, x, bias)
                if self._alignment_active and _tag not in self._alignment_records:
                    self._alignment_records[_tag] = capture(_tag, x, layer.weight, output,
                        len(config.nvfp4_alignment_trace_tokens))
                return output

            method.apply = observed

    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        expected = self._alignment_config.nvfp4_alignment_trace_tokens
        self._alignment_active = (not self._alignment_done and input_ids.numel() == len(expected)
                                  and input_ids.detach().cpu().tolist() == expected)
        try:
            result = super().forward(input_ids, positions, forward_batch, *args, **kwargs)
            if self._alignment_active:
                torch.save(self._alignment_records, self._alignment_config.nvfp4_alignment_trace_path)
                self._alignment_done = True
            return result
        finally:
            self._alignment_active = False


EntryClass = NVFP4AlignmentQwen3ForCausalLM
