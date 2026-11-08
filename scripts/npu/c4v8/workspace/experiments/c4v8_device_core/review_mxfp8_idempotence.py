"""CPU-only values/encoding idempotence audit; does not change any boundary."""
import hashlib,json
from pathlib import Path
import torch
from device_quant import mxfp8_values,mxfp8_qdq

def main():
    torch.set_num_threads(1);g=torch.Generator().manual_seed(720021)
    cases=[]
    cases.append(('signed-zero',torch.tensor([[0.,-0.]*32])))
    sub=torch.randint(1,0x800000,(1024,32),generator=g,dtype=torch.int32).view(torch.float32)
    cases.append(('subnormal-random',sub))
    bits=torch.randint(1,0x7f800000,(8192,32),generator=g,dtype=torch.int32)
    signs=torch.randint(0,2,bits.shape,generator=g,dtype=torch.int32)*(-2147483648)
    cases.append(('all-exponent-random',(bits|signs).view(torch.float32)))
    cases.append(('ordinary-random',torch.randn(1024,128,generator=g)))
    cases.append(('bf16-source-random',torch.randn(1024,128,generator=g).bfloat16().float()))
    tiny=torch.tensor([0.,-0.,2.**-149,2.**-148,2.**-140,2.**-130,2.**-126])
    cases.append(('tiny-tail',tiny.reshape(1,-1)))
    edge=[]
    for exponent in range(-127,120):
        boundary=torch.tensor(224.*(2.**exponent))
        above=torch.nextafter(boundary,torch.tensor(float('inf')))
        below=torch.nextafter(boundary,torch.tensor(0.))
        edge.append(torch.tensor([float(below),float(boundary),float(above),-float(above)]).repeat(8))
    cases.append(('scale-can-shrink-boundaries',torch.stack(edge)))
    mask=torch.randn(64,128,generator=g);mask[:,::3]=-float('inf');mask[0]=-float('inf')
    cases.append(('structural-mask',mask))
    rows=[];failures=[]
    for name,x in cases:
        structural=torch.isneginf(x) if name=='structural-mask' else torch.zeros_like(x,dtype=torch.bool)
        ordinary=x.masked_fill(structural,0.)
        first=mxfp8_values(ordinary,check=False,logical_bytes=True)
        valid=torch.isfinite(first.values).all(-1)
        rejected_rows=int((~valid).sum())
        a=first.values[valid];valid_mask=structural[valid]
        second=mxfp8_values(a,logical_bytes=True)
        first_values=a.masked_fill(valid_mask,-float('inf'));second_values=second.values.masked_fill(valid_mask,-float('inf'))
        bitdiff=first_values.contiguous().view(torch.int32)!=second_values.contiguous().view(torch.int32)
        code_diff=first.payload_bytes[valid]!=second.payload_bytes
        scale_diff=first.scale_bytes[valid]!=second.scale_bytes
        row={'name':name,'shape':list(x.shape),'ordinary_finite_inputs':bool(torch.isfinite(ordinary).all()),
             'first_reconstruction_nonfinite_rows':rejected_rows,'valid_rows_compared':int(valid.sum()),
             'values_compared':first_values.numel(),'value_bit_differences':int(bitdiff.sum()),
             'payload_byte_differences':int(code_diff.sum()),'scale_byte_differences':int(scale_diff.sum())}
        if rejected_rows:
            try:mxfp8_values(ordinary[~valid])
            except FloatingPointError:row['invalid_reconstruction_rejected_by_checked_api']=True
            else:raise AssertionError('overflow not rejected')
        if bitdiff.any():failures.append({'name':name,'input':x[valid],'first':first_values,'second':second_values})
        rows.append(row)
    invalid=[]
    for value in [float('nan'),float('inf'),-float('inf')]:
        try:mxfp8_values(torch.tensor([[value]]))
        except ValueError:invalid.append({'value':str(value),'ordinary_input_rejected':True})
        else:raise AssertionError('nonfinite ordinary input accepted')
    out=Path(__file__).parent/'evidence/mxfp8-idempotence-cpu-review.json'
    result={'scope':'CPU FP32 simulator reconstructed values only; exact encoding and NPU/compiler identity are separate claims',
            'finite_reconstructed_values_idempotent_in_test':not failures,'cases':rows,'invalid_inputs':invalid,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'no_boundaries_removed':True}
    out.write_text(json.dumps(result,indent=2)+'\n')
    if failures:torch.save(failures,out.with_suffix('.pt'))
    print(json.dumps(result))

if __name__=='__main__':main()
