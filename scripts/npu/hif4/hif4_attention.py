"""HiF4 Cube operands and the shared BF16/FP32 Vector implementation."""
from dataclasses import dataclass

import torch

import c8v16_attention as vector
from hif4_checks import check_invalid
from hif4_quant import GROUP, quantize


P_METHOD = "single_hif4_qdq_before_pv"


@dataclass(frozen=True, kw_only=True)
class AttentionConfig(vector.AttentionConfig):
    cube_format: str = "hif4_e6m2_s1p2_64_8_4"
    p_method: str

    def __post_init__(self):
        super().__post_init__()
        if self.key_block % GROUP:
            raise ValueError("HiF4 key tiles must contain complete 64-value scale groups")
        if self.p_method != P_METHOD or self.cube_format != "hif4_e6m2_s1p2_64_8_4":
            raise ValueError("Unknown HiF4 conversion or P rule")


def attention(q, k, v, config, *, query_offset=0):
    if not isinstance(config, AttentionConfig):
        raise TypeError("HiF4 requires an explicit HiF4 attention configuration")
    vector._validate(q, k, v, config, query_offset)
    with torch.autocast(device_type=q.device.type, enabled=False):
        if (q.device.type != "cpu" and q.shape[-2:] == (1, 128) and config.key_block == 128
                and query_offset + 1 == k.size(2) and not any(t.requires_grad for t in (q, k, v))):
            from hif4_decode import decode
            return decode(q, k, v)
        return vector._attention(q, k, v, config, query_offset, qdq=quantize, group_size=GROUP, check_invalid=check_invalid)
