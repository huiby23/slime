"""Independent CPU proof; reports BLAS order changes separately from scheduling."""
import argparse,hashlib,json,sys,traceback
from pathlib import Path
import torch
from staged_cpu_oracle import staged_cpu
from device_attention import attention,AttentionConfig


def compare(a,b):
    a=a.float();b=b.float();finite=torch.isfinite(a)&torch.isfinite(b)
    diff=torch.where(finite,a-b,0)
    return {'bit_equal':torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32)),
            'different_values':int((a!=b).sum()),'max_finite_abs':float(diff.abs().max()),
            'relative_finite_l2':float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(torch.where(finite,b,0)).clamp_min(1e-30))}


def main(output):
    torch.set_num_threads(1);generator=torch.Generator().manual_seed(928844)
    config=AttentionConfig(query_block=64,key_block=128);cases=[]
    for dtype in (torch.float32,torch.bfloat16):
        for seq in (1,17,33,128,129,255,256):
            for batch,heads,kvheads in ((1,8,1),(2,4,2)):
                q=torch.randn(batch,heads,1,128,generator=generator).to(dtype)
                k=torch.randn(batch,kvheads,seq,128,generator=generator).to(dtype)
                v=torch.randn(batch,kvheads,seq,128,generator=generator).to(dtype)
                reference=attention(q,k,v,config,query_offset=seq-1)
                sequential,seq_details=staged_cpu(q,k,v,batched_gemm=False,return_intermediates=True)
                batched,batch_details=staged_cpu(q,k,v,batched_gemm=True,return_intermediates=True)
                row={'dtype':str(dtype),'seq':seq,'batch':batch,'heads':heads,'kvheads':kvheads,
                     'same_gemm_scheduling_vs_reference':compare(sequential,reference),
                     'batched_output_vs_reference':compare(batched,reference),
                     'batched_qk_vs_tile_qk':compare(batch_details['raw_scores'],seq_details['raw_scores']),
                     'batched_pv_vs_tile_pv':compare(batch_details['raw_contribution'],seq_details['raw_contribution'])}
                assert row['same_gemm_scheduling_vs_reference']['bit_equal'],row
                cases.append(row)
    out=output;out.mkdir(parents=True,exist_ok=False)
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    report={'scheduling_proof_passed':True,'scope':'CPU-only; NPU/compiler/throughput not validated','cases':cases,
            'batched_output_all_exact':all(c['batched_output_vs_reference']['bit_equal'] for c in cases),
            'sources':source,'npu_validated':False,'production_registered':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'scheduling_proof_passed':True,'cases':len(cases),'batched_output_all_exact':report['batched_output_all_exact']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
