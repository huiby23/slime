"""Short real-candidate B4 decode audit; requires an explicit exclusive NPU lease."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import argparse,hashlib,importlib.util,json,math,re,subprocess,sys,time,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path[:0]=[str(BASE/n) for n in ['c4v8_device_core','c4v8_fast_attention','c4v8_fused_nvfp4','c4v8_fused_qdq','c4v8_staged_decode','c4v8_rollout_bridge','c4v8_decode_batch_candidate','c4v8_query_tiling_probe']]
from validate_npu_extended import replay_actual_gemms
from validate_final_core_npu import metric
from device_attention import AttentionConfig
from logical_kv import RequestSpan,forward_cached as original_forward_cached


def json_save(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def cpu(value):
 if isinstance(value,torch.Tensor):return value.detach().cpu()
 if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [cpu(x) for x in value]
 return value

def cat_stages(stages):
 return {key:({k:torch.cat([x[key][k] for x in stages],0) for k in stages[0][key]} if isinstance(stages[0][key],dict) else torch.cat([x[key] for x in stages],0)) for key in stages[0]}


class Recorder:
 def __init__(self):self.calls=[]
 def __call__(self,q,k,v,config,query_offset=0):
  from staged_decode import staged_decode_forward
  if q.shape[2]!=1 or query_offset!=k.shape[2]-1:raise AssertionError('Not final-position decode')
  out,flags,details=staged_decode_forward(q,k,v,check=False,return_intermediates=True)
  item={'qkv':cpu((q,k,v)),'query_offset':query_offset,'output':cpu(out),'flags':cpu(flags),'stages':cpu(details)}
  self.calls.append(item)
  if bool((flags!=0).any()):raise FloatingPointError('Staged flags; captured actual tensors')
  return out


def run(args,report):
 before=subprocess.check_output(['npu-smi','info'],text=True);(args.output/'npu-before.txt').write_text(before)
 if sorted(set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)',before))))!=list(range(8)):raise RuntimeError('All-eight idle required')
 import torch_npu
 torch.npu.set_device(1);torch.npu.set_per_process_memory_fraction(.05,1);torch.npu.set_compile_mode(jit_compile=False);torch.npu.matmul.allow_hf32=False;torch.set_num_threads(1)
 os.environ['HBY_C4V8_ALLOW_UNVALIDATED_FUSION']='1'
 from equal_length_decode import forward_cached as grouped_forward_cached
 from fast_attention import attention as live_core
 cfg=AttentionConfig(query_block=1024,key_block=128);rng=torch.Generator().manual_seed(741173)
 report.update({'cases':[],'batch':4,'q_heads':4,'kv_heads':1,'head_dim':128,'dtype':'bfloat16','prefix_convention':'KV length is prefix+1, including the current decode token.',
  'scope':'Real proposed forward_cached grouping/stack/gather/scatter with frozen staged core; no SGLang model, no GRPO update.'})
 with torch.inference_mode():
  for index,prefix in enumerate(args.prefixes):
   seq=prefix+1;folder=args.output/f'prefix{prefix}';folder.mkdir();print(f'PHASE fixture prefix={prefix}',flush=True)
   reqids=[5,1,7,3];spans=tuple(RequestSpan(i,prefix,1) for i in reqids)
   cache_slots=4*seq+128;slots=torch.randperm(cache_slots-1,generator=rng)[:4*seq]+1
   table=torch.full((8,seq),-1,dtype=torch.int32)
   key_cache=torch.full((cache_slots,1,128),13,dtype=torch.bfloat16);value_cache=torch.full_like(key_cache,-7)
   for j,req in enumerate(reqids):
    positions=slots[j*seq:(j+1)*seq];table[req]=positions.to(torch.int32)
    key_cache[positions]=torch.randn(seq,1,128,generator=rng).bfloat16();value_cache[positions]=torch.randn(seq,1,128,generator=rng).bfloat16()
   q=torch.randn(4,4,128,generator=rng).bfloat16()
   torch.save({'packed_q':q,'key_cache':key_cache,'value_cache':value_cache,'table':table,'spans':spans},folder/'cpu-cache-fixture.pt')
   nq,nk,nv,nt=[x.to('npu:1') for x in (q,key_cache,value_cache,table)];kwargs={'config':cfg,'scale':1/math.sqrt(128)}
   batch_recorder=Recorder();single_recorder=Recorder()
   print(f'PHASE actual_grouped_bridge prefix={prefix}',flush=True)
   grouped=grouped_forward_cached(nq,nk,nv,nt,spans,core=batch_recorder,group_same_length_decode=True,**kwargs).cpu()
   print(f'PHASE original_four_B1_bridge prefix={prefix}',flush=True)
   singles=original_forward_cached(nq,nk,nv,nt,spans,core=single_recorder,**kwargs).cpu()
   torch.save({'grouped_calls':batch_recorder.calls,'single_calls':single_recorder.calls,'grouped_output':grouped,'single_output':singles},folder/'actual-device-stages.pt')
   row={'prefix':prefix,'kv_length':seq,'grouped_calls':len(batch_recorder.calls),'single_calls':len(single_recorder.calls),'grouped_vs_4B1':metric(grouped,singles)};report['cases'].append(row)
   if len(batch_recorder.calls)!=1 or batch_recorder.calls[0]['qkv'][0].shape[0]!=4 or len(single_recorder.calls)!=4:raise AssertionError('Unexpected grouping cardinality')
   actual_batch=batch_recorder.calls[0];combined=cat_stages([x['stages'] for x in single_recorder.calls])
   row['grouped_vs_independent_stages']={key:metric(actual_batch['stages'][key],value) for key,value in combined.items() if not isinstance(value,dict)}
   row['CPU_actual_scaled_QK_PV_replays']=[]
   for label,calls in [('grouped',batch_recorder.calls),('independent',single_recorder.calls)]:
    for j,call in enumerate(calls):
     replay,replay_details=replay_actual_gemms(call['qkv'],call['stages'])
     metrics={name:metric(value,call['stages'][name]) for name,value in replay_details.items()}
     entry={'path':label,'call':j,'output':metric(replay,call['output']),'stages':metrics,'all_stages_bit_exact':all(v['bit_exact'] for v in metrics.values())}
     row['CPU_actual_scaled_QK_PV_replays'].append(entry)
     torch.save({'output':replay,'stages':replay_details},folder/f'cpu-replay-{label}-{j}.pt')
   print(f'PHASE request_permutation_and_canonical_core prefix={prefix}',flush=True)
   order=[2,0,3,1];order_npu=torch.tensor(order,device=nq.device,dtype=torch.int64)
   permuted=grouped_forward_cached(nq.index_select(0,order_npu),nk,nv,nt,tuple(spans[j] for j in order),core=live_core,group_same_length_decode=True,**kwargs).cpu()
   row['request_order_permutation']=metric(permuted,grouped[order])
   canonical=grouped_forward_cached(nq,nk,nv,nt,spans,core=live_core,group_same_length_decode=True,**kwargs).cpu()
   row['diagnostic_vs_canonical_routing']=metric(canonical,grouped)
   torch.save({'permuted':permuted,'canonical':canonical},folder/'canonical-permuted-output.pt')
   # Timing uses the actual proposed bridge and current canonical core. No
   # recording tensors/CPU copies occur inside the timed calls.
   torch.npu.synchronize(1);start=time.perf_counter()
   for _ in range(3):grouped_forward_cached(nq,nk,nv,nt,spans,core=live_core,group_same_length_decode=True,**kwargs)
   torch.npu.synchronize(1);row['warm_grouped_bridge_mean_seconds']=(time.perf_counter()-start)/3
   start=time.perf_counter()
   for _ in range(3):original_forward_cached(nq,nk,nv,nt,spans,core=live_core,**kwargs)
   torch.npu.synchronize(1);row['warm_4B1_bridge_mean_seconds']=(time.perf_counter()-start)/3
   row['strict_predicates']={'grouped_vs_4B1_bitexact':row['grouped_vs_4B1']['bit_exact'],
    'request_order_permutation_bitexact':row['request_order_permutation']['bit_exact'],'canonical_routing_bitexact':row['diagnostic_vs_canonical_routing']['bit_exact'],
    'all_actual_input_CPU_quantization_vector_replays_bitexact':all(e['all_stages_bit_exact'] and e['output']['bit_exact'] for e in row['CPU_actual_scaled_QK_PV_replays']),
    'finite_outputs':row['grouped_vs_4B1']['both_finite']}
   json_save(folder/'report.json',row);json_save(args.output/'report.partial.json',report)
   print(json.dumps({'prefix':prefix,'strict_predicates':row['strict_predicates'],'group_seconds':row['warm_grouped_bridge_mean_seconds'],'single_seconds':row['warm_4B1_bridge_mean_seconds']}),flush=True)
 report['peak_allocated_bytes']=int(torch.npu.max_memory_allocated(1));report['diagnostic_completed']=True
 report['strict_all_cases_passed']=all(all(c['strict_predicates'].values()) for c in report['cases'])
 if not report['strict_all_cases_passed']:raise AssertionError('Exact grouped baseline/route/replay mismatch; actual tensors preserved, no tolerance relaxation')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prefixes',type=int,nargs='+',default=[129,1024,4096]);args=p.parse_args()
 if any(n<0 or n>4096 for n in args.prefixes):raise ValueError('Bounded prefixes0..4096')
 args.output.mkdir(parents=True,exist_ok=False)
 sources=[Path(__file__),*(BASE/name for name in ['c4v8_query_tiling_probe/validate_final_core_npu.py','c4v8_staged_decode/validate_npu_extended.py','c4v8_staged_decode/staged_decode.py','c4v8_staged_decode/staged_kernels.py','c4v8_decode_batch_candidate/equal_length_decode.py','c4v8_rollout_bridge/logical_kv.py','c4v8_fast_attention/fast_attention.py','c4v8_device_core/device_attention.py','c4v8_device_core/device_quant.py','c4v8_device_core/exact_fp32.py','c4v8_fused_nvfp4/nvfp4_triton.py','c4v8_fused_qdq/fused_mxfp8.py','c4v8_fused_qdq/forward_api.py'])]
 hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
 for i,path in enumerate(sources):
  dest=args.output/'sources'/f'{i:02d}-{path.parent.name}-{path.name}';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
 report={'diagnostic_completed':False,'source_sha256':hashes,'NPU':'physical1','memory_fraction':.05};start=time.perf_counter();code=1
 try:run(args,report);code=0
 except BaseException as e:report['error']=type(e).__name__+': '+str(e);traceback.print_exc();(args.output/'failure.txt').write_text(traceback.format_exc())
 finally:
  report['elapsed_seconds']=time.perf_counter()-start;report['sources_unchanged']=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in hashes.items())
  if not report['sources_unchanged']:code=1
  json_save(args.output/'report.json',report);(args.output/'exit').write_text(str(code)+'\n');print(json.dumps({'completed':report['diagnostic_completed'],'exit':code,'error':report.get('error')}),flush=True)
 sys.exit(code)
