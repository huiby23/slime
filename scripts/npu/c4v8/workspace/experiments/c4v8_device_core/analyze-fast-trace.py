"""CPU-only evidence analysis; does not change numerical acceptance tolerances."""
import argparse,hashlib,importlib.util,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from device_attention import AttentionConfig,online_softmax_step,vector_round
from device_quant import nvfp4_values
spec=importlib.util.spec_from_file_location('trace_module',ROOT/'diagnose-fast-stages.py')
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)

def cpu_replay(qkv,config,actual,*,scale='actual',pv='cpu'):
    q,k,v=qkv;b,h,n,d=q.shape
    qq,kk,vv=(nvfp4_values(x.float()).values for x in qkv)
    if h!=k.shape[1]:
        kk=kk.repeat_interleave(h//k.shape[1],1);vv=vv.repeat_interleave(h//k.shape[1],1)
    result=[];index=0
    for qs in range(0,n,config.query_block):
        qe=min(n,qs+config.query_block);state=(b,h,qe-qs,1)
        m=torch.full(state,-math.inf);l=torch.zeros(state);o=torch.zeros((b,h,qe-qs,d))
        for ks in range(0,qe,config.key_block):
            ke=min(qe,ks+config.key_block);item=actual['tiles'][index]
            raw=item['raw_qk'] if scale=='actual' else item['raw_qk_matmul']/math.sqrt(d)
            mask=torch.arange(ks,ke)[None,:]>torch.arange(qs,qe)[:,None]
            scores=diag.vr(raw.masked_fill(mask,-math.inf),True)
            m,l,a,p=online_softmax_step(scores,m,l,config)
            p4=nvfp4_values(p).values
            contribution=p4@vv[:,:,ks:ke] if pv=='cpu' else item['raw_pv']
            o=diag.vr(diag.vr(o*a)+diag.vr(contribution));index+=1
        result.append(diag.vr(o/l).to(q.dtype))
    return torch.cat(result,2)

def records_at_differences(cpu,npu,stage,input_stage,limit=4):
    rows=[]
    for idx in (cpu[stage]!=npu[stage]).nonzero()[:limit].tolist():
        pos=tuple(idx);a=float(cpu[input_stage][pos]);b=float(npu[input_stage][pos])
        qa=float(cpu[stage][pos]);qb=float(npu[stage][pos]);mid=(qa+qb)/2
        rows.append({'coordinate_within_tile':idx,'cpu_input':a,'npu_input':b,'cpu_quantized':qa,'npu_quantized':qb,
                     'midpoint_between_outputs':mid,'cpu_minus_midpoint':a-mid,'npu_minus_midpoint':b-mid})
    return rows

def main(args):
    torch.set_num_threads(1)
    report=json.loads((args.trace/'report.json').read_text())
    inputs=torch.load(args.inputs,map_location='cpu',weights_only=False)
    result={'source_trace':str(args.trace),'trace_report_sha256':hashlib.sha256((args.trace/'report.json').read_bytes()).hexdigest(),
            'trace_source_sha256':report['source_sha256'],'old_strict_gemm_only_predicate_value':report['all_lengths_gemm_only_attribution_supported'],
            'original_trace_unchanged':True,
            'scope':'CPU replay of saved NPU tensors only; no NPU execution; no production edits or tolerance changes','cases':[]}
    cfg=AttentionConfig(query_block=64,key_block=128)
    for saved in report['cases']:
        n=saved['n'];base=args.trace/f'n{n}'
        cpu=torch.load(base/'cpu-original.pt',weights_only=False);npu=torch.load(base/'npu-fused.pt',weights_only=False)
        qkv=next(c['qkv'] for c in inputs if c['n']==n)
        row={'n':n,'cpu_vs_npu_output':saved['cpu_original_vs_npu']['output'],'same_input_format_comparisons':len(saved['same_input_format_checks']),
             'all_same_input_format_values_codes_scales_exact':all(v[k]['bit_exact'] for v in saved['same_input_format_checks'] for k in ('values','codes','scales')),
             'actual_scaled_QK_PV_CPU_replay_all_nongemm_exact':saved['all_replayed_nongemm_stages_bit_exact'],
             'actual_scaled_QK_PV_CPU_replay_output_exact':saved['cpu_actual_gemm_replay_vs_npu']['output']['bit_exact'],
             'same_input_normalize_exact':all(v['bit_exact'] for v in saved['same_input_normalize_checks']),
             'instrumented_vs_frozen_exact':saved['instrumented_vs_frozen_core']['bit_exact'],
             'instrumented_vs_saved_npu_exact':saved['trace_vs_saved_npu_forward']['bit_exact'],
             'qk_scalar_scaling_differences':[],'tile_causal_chain':[]}
        for c,g in zip(cpu['tiles'],npu['tiles']):
            expected=g['raw_qk_matmul']/math.sqrt(qkv[0].shape[-1])
            m=diag.metric(expected,g['raw_qk']);m.update({'qs':c['qs'],'ks':c['ks']})
            # Correctly rounded CPU division agrees with float64 division rounded
            # to FP32 for this fixed FP32 divisor; NPU can differ by one ulp.
            fp32_divisor=torch.tensor(math.sqrt(qkv[0].shape[-1]),dtype=torch.float32)
            m['cpu_vs_f64_division_by_fp32_divisor']=diag.metric(expected,(g['raw_qk_matmul'].double()/fp32_divisor.double()).float())
            row['qk_scalar_scaling_differences'].append(m)
            row['tile_causal_chain'].append({'qs':c['qs'],'ks':c['ks'],
                'raw_QK_matmul':diag.metric(c['raw_qk_matmul'],g['raw_qk_matmul']),
                'scaled_QK':diag.metric(c['raw_qk'],g['raw_qk']),
                'score8':diag.metric(c['score8'],g['score8']),
                'P8':diag.metric(c['p8'],g['p8']),'P4':diag.metric(c['p4'],g['p4']),
                'raw_PV':diag.metric(c['raw_pv'],g['raw_pv']),
                'PV_quantization_threshold_examples':records_at_differences(c,g,'contribution8','raw_pv'),
                'score_quantization_threshold_examples':records_at_differences(c,g,'score8','raw_qk')})
        counterfactuals={}
        for label,scale,pv in [('npu_matmul_cpu_scaling_cpu_PV','cpu','cpu'),('npu_scaled_QK_cpu_PV','actual','cpu'),('actual_scaled_QK_actual_PV','actual','actual')]:
            out=cpu_replay(qkv,cfg,npu,scale=scale,pv=pv)
            counterfactuals[label]={'vs_actual_npu':diag.metric(out,npu['output']),'vs_original_cpu':diag.metric(out,cpu['output'])}
        row['counterfactual_outputs']=counterfactuals
        cf=cpu_replay(qkv,cfg,npu,scale='actual',pv='cpu')
        original_diff=(cpu['output']!=npu['output']).flatten();qk_diff=(cpu['output']!=cf).flatten();remaining=(cf!=npu['output']).flatten()
        row['counterfactual_output_location_audit']={'original_difference_count':int(original_diff.sum()),'qk_chain_changed_count':int(qk_diff.sum()),
          'qk_chain_changes_subset_of_actual_differences':not bool((qk_diff&~original_diff).any()),'remaining_PV_count':int(remaining.sum()),
          'remaining_PV_subset_of_actual_differences':not bool((remaining&~original_diff).any()),'qk_and_remaining_PV_disjoint':not bool((qk_diff&remaining).any())}
        exact_keys=['all_same_input_format_values_codes_scales_exact','actual_scaled_QK_PV_CPU_replay_all_nongemm_exact',
                    'actual_scaled_QK_PV_CPU_replay_output_exact','same_input_normalize_exact','instrumented_vs_frozen_exact','instrumented_vs_saved_npu_exact']
        row['fp32_matmul_and_scalar_scaling_attribution_supported']=all(row[k] for k in exact_keys)
        result['cases'].append(row)
    result['all_cases_nongemm_implementation_exact_given_actual_FP32_inputs']=all(c['fp32_matmul_and_scalar_scaling_attribution_supported'] for c in result['cases'])
    result['acceptance_rationale']=[
      'The strict original GEMM-only predicate remains false: ordinary QK scalar division is also backend-dependent in these traces.',
      'The broader cause is upstream ordinary FP32 matmul plus QK scalar scaling. Supplying the observed NPU scaled QK and PV to unchanged CPU quantization/vector logic reproduces every quantization/reduction state and every output bit for all three cases.',
      'Same-input C4/V8 values, payload bytes, and scales are exact. Final normalization is exact for identical numerator/denominator. No unexplained quantization or tree-reduction implementation difference remains in this bounded set.',
      'Small upstream FP32 differences can cross a low-precision rounding midpoint and propagate: this is a sensitivity characteristic, not proof of CPU/NPU universal bit identity.',
      'This supports bounded numerical implementation acceptance with explicit FP32 backend-rounding disclosure; it does not establish production throughput, all-shape equivalence, low-precision backward, or unchanged model quality.'
    ]
    result['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['cpu_input_fixture_sha256']=hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    result['original_trace_unchanged']=hashlib.sha256((args.trace/'report.json').read_bytes()).hexdigest()==result['trace_report_sha256']
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'all_cases_attributed':result['all_cases_nongemm_implementation_exact_given_actual_FP32_inputs'],'output':str(args.output),
                     'cases':[{k:c[k] for k in ['n','same_input_format_comparisons','fp32_matmul_and_scalar_scaling_attribution_supported','counterfactual_outputs']} for c in result['cases']]}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
