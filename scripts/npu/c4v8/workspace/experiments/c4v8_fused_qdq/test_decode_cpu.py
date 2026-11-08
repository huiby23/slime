import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'c4v8_device_core'))
from device_attention import attention,AttentionConfig
from decode_cpu_oracle import decode


class DecodeCPUCase(unittest.TestCase):
    def test_whole_decode_128_256_matches_declared_full_profile(self):
        torch.set_num_threads(1)
        config=AttentionConfig(query_block=64,key_block=128)
        for n in (128,256):
            generator=torch.Generator().manual_seed(448+n)
            q=torch.randn((1,8,1,128),generator=generator).bfloat16()
            k=torch.randn((1,1,n,128),generator=generator).bfloat16()
            v=torch.randn((1,1,n,128),generator=generator).bfloat16()
            got,audit=decode(q,k,v,config,return_audit=True)
            expected=attention(q,k,v,config,query_offset=n-1)
            # For these fixtures allquantized boundaries coincide. Future NPU
            # tests must report, not hide, any FP32-induced boundary crossing.
            torch.testing.assert_close(got,expected,atol=0,rtol=0)
            self.assertTrue(all(a['qk_float_rel_l2']<1e-6 and a['pv_float_rel_l2']<1e-6 for a in audit))


if __name__=='__main__':unittest.main()
