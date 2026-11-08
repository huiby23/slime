"""CPU-only independent algebra review of aligned fixed-width MXFP8 trees."""
import json,math,hashlib
from pathlib import Path
import torch
from device_attention import mxfp8_tree_reduce
from device_quant import mxfp8_qdq

def qdq(x,masked):
    return mxfp8_qdq(x,allow_masked_negative_infinity=masked)

def fixed(x,operation):
    width=x.shape[-1];block=max(32,1<<(width-1).bit_length())
    identity=-math.inf if operation=='max' else 0.
    padded=torch.nn.functional.pad(x,(0,block-width),value=identity)
    current=qdq(padded,operation=='max');idx=torch.arange(block)
    for level in range((block-1).bit_length()):
        left=current[..., (2*idx)%block];right=current[..., (2*idx+1)%block]
        partial=torch.maximum(left,right) if operation=='max' else left+right
        partial=torch.where(idx<((width+(1<<(level+1))-1)>>(level+1)),partial,identity)
        current=qdq(partial,operation=='max')
    gather=current[...,:1]
    extraction=torch.where(idx==0,current,-math.inf).amax(-1,keepdim=True)
    return gather,extraction

def main():
    torch.set_num_threads(1);g=torch.Generator().manual_seed(691413)
    widths=[1,2,3,5,7,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257]
    records=[]
    for width in widths:
        for kind in ['signed','probability','zero']:
            if kind=='signed':x=torch.randn(3,width,generator=g)*torch.pow(2.,torch.randint(-20,20,(3,1),generator=g).float())
            elif kind=='probability':x=torch.rand(3,width,generator=g)
            else:x=torch.zeros(3,width);x[0]=-0.
            for operation in ['sum','max']:
                data=x.clone()
                if operation=='max' and kind!='zero' and width>1:data[:,1::3]=-math.inf
                expected=mxfp8_tree_reduce(data,operation).detach()
                aligned,selected=fixed(data,operation)
                item={'width':width,'kind':kind,'operation':operation,
                      'fixed_tree_bit_exact':torch.equal(expected.view(torch.int32),aligned.view(torch.int32)),
                      'fixed_tree_values_equal':torch.equal(expected,aligned),
                      'selector_bit_exact':torch.equal(aligned.view(torch.int32),selected.view(torch.int32))}
                if not item['fixed_tree_values_equal'] or not item['selector_bit_exact']:raise AssertionError(item)
                if not item['fixed_tree_bit_exact']:
                    item['known_exception']='Negative-zero-only sum may gain +0 padding additions; not produced by exp-based P probability inputs'
                    assert operation=='sum' and kind=='zero'
                records.append(item)
    out=Path(__file__).parent/'evidence/fixed-tree-independent-review.json'
    out.write_text(json.dumps({'passed_for_probability_sum_and_max':True,'all_finite_numeric_values_equal':True,
                              'signed_zero_sum_exceptions':sum(not x['fixed_tree_bit_exact'] for x in records),
                              'scope':'CPU algebra only; does not validate Ascend gather/layout/compiler',
                              'cases':len(records),'records':records,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({'passed_for_probability_sum_and_max':True,'cases':len(records),'signed_zero_sum_exceptions':sum(not x['fixed_tree_bit_exact'] for x in records),'path':str(out)}))

if __name__=='__main__':main()
