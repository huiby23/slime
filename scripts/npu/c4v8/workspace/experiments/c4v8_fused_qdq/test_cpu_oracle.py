import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['OMP_NUM_THREADS']='1'
import sys
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'c4v8_device_core'))
from device_quant import mxfp8_values,round_e4m3fn,e4m3fn_bytes
from device_attention import AttentionConfig,online_softmax_step,vector_round
from cpu_oracle import qdq,round_e4,tree,fixed_width_tree,pre_p,post_pv


class FusedOracleTests(unittest.TestCase):
    def test_fixed_lane_gather_tree_equals_compact_tree(self):
        gen=torch.Generator().manual_seed(329)
        for width in (1,3,7,16,17,32,63,64,65,127,128):
            x=torch.randn((9,width),generator=gen)
            for op in ('sum','max'):
                torch.testing.assert_close(fixed_width_tree(x,op),tree(x,op),atol=0,rtol=0)
                block=max(32,1<<(width-1).bit_length())
                padded=torch.nn.functional.pad(x,(0,block-width),value=-float('inf') if op=='max' else 0.)
                torch.testing.assert_close(tree(padded,op),tree(x,op),atol=0,rtol=0)

    def test_e4_all_midpoints_and_neighbors(self):
        levels=torch.cat([torch.arange(8)*2.0**-9]+[
            torch.arange(8,16)*2.0**(e-3) for e in range(-6,9)])
        levels=levels[levels<=448].sort().values.unique()
        mids=(levels[1:]+levels[:-1])*0.5
        x=torch.cat([levels,mids,torch.nextafter(mids,torch.full_like(mids,float('inf'))),
                     torch.nextafter(mids,torch.zeros_like(mids)),torch.tensor([0.,-0.])])
        x=torch.cat([x,-x]).float()
        value,code=round_e4(x)
        torch.testing.assert_close(value.view(torch.int32),round_e4m3fn(x).view(torch.int32),atol=0,rtol=0)
        torch.testing.assert_close(code,e4m3fn_bytes(x),atol=0,rtol=0)

    def test_scale_boundaries_entire_fp32_range(self):
        values=[]
        for e in range(-127,120):
            b=torch.tensor(448.0*2.0**e,dtype=torch.float32)
            values.extend([b,torch.nextafter(b,torch.tensor(0.)),torch.nextafter(b,torch.tensor(float('inf')))])
        values+=list(torch.tensor([0.,-0.,2.0**-149,2.0**-126,torch.finfo(torch.float32).max]))
        x=torch.stack(values).reshape(-1,1).repeat(1,32)
        x[:,1::2]=-x[:,1::2]
        got=qdq(x);ref=mxfp8_values(x,check=False,logical_bytes=True)
        torch.testing.assert_close(got.values.view(torch.int32),ref.values.view(torch.int32),atol=0,rtol=0)
        torch.testing.assert_close(got.payload,ref.payload_bytes,atol=0,rtol=0)
        torch.testing.assert_close(got.scales,ref.scale_bytes,atol=0,rtol=0)

    def test_random_bits_and_partial_blocks(self):
        gen=torch.Generator().manual_seed(331)
        for width in (1,17,32,63,128):
            raw=torch.randint(-(1<<31),(1<<31)-1,(4096,width),generator=gen,dtype=torch.int32)
            x=raw.view(torch.float32)
            x=torch.where(torch.isfinite(x),x,torch.zeros_like(x))
            got=qdq(x);ref=mxfp8_values(x,check=False,logical_bytes=True)
            torch.testing.assert_close(got.values.view(torch.int32),ref.values.view(torch.int32),atol=0,rtol=0)
            torch.testing.assert_close(got.payload,ref.payload_bytes,atol=0,rtol=0)
            torch.testing.assert_close(got.scales,ref.scale_bytes,atol=0,rtol=0)

    def test_nonfinite_flags_and_masked_negative_infinity(self):
        x=torch.tensor([[0.,-0.,float('-inf'),2.],[float('nan'),float('inf'),1.,-1.]])
        got=qdq(x,True)
        self.assertEqual(got.flags.tolist(),[0,1])
        self.assertEqual(got.values[0,1].view(torch.int32).item(),-2147483648)
        self.assertTrue(torch.isneginf(got.values[0,2]))
        self.assertTrue(torch.isnan(got.values[1,:2]).all())
        self.assertEqual(qdq(x,False).flags.tolist(),[1,1])

    def test_pre_p_all_boundaries_and_partial_widths(self):
        gen=torch.Generator().manual_seed(337)
        config=AttentionConfig(key_block=128)
        for width in (1,7,17,32,64,127,128):
            scores=torch.randn((9,width),generator=gen)
            scores[::2,width//2+1:]=-float('inf')
            for initialized in (False,True):
                old_m=torch.randn((9,1),generator=gen) if initialized else torch.full((9,1),-float('inf'))
                old_l=torch.ones((9,1)) if initialized else torch.zeros((9,1))
                got=pre_p(scores,old_m,old_l)
                ref=online_softmax_step(vector_round(scores,'mxfp8',allow_structural_negative_infinity=True),old_m,old_l,config)
                for a,b in zip(got,ref):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_post_pv_all_boundaries(self):
        gen=torch.Generator().manual_seed(339)
        for width in (16,128):
            old=torch.randn((11,width),generator=gen);c=torch.randn((11,width),generator=gen)
            alpha=torch.rand((11,1),generator=gen);den=torch.rand((11,1),generator=gen)+0.1
            v=lambda x:vector_round(x,'mxfp8')
            ref=v(v(old*alpha)+v(c))
            torch.testing.assert_close(post_pv(old,c,alpha),ref,atol=0,rtol=0)
            torch.testing.assert_close(post_pv(old,c,alpha,den),v(ref/den),atol=0,rtol=0)

    def test_all_masked_never_initialised_rejected(self):
        with self.assertRaises(FloatingPointError):
            pre_p(torch.full((1,32),-float('inf')),torch.tensor([[-float('inf')]]),torch.zeros(1,1))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
