import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import pathlib,sys,json,hashlib,torch
BASE=pathlib.Path('/root/hby/experiments')
sys.path[:0]=[str(BASE/n) for n in ['c4v8_device_core','c4v8_fast_attention','c4v8_fused_nvfp4','c4v8_fused_qdq']]
import torch_npu
torch.npu.set_device(1);torch.npu.set_per_process_memory_fraction(.05,1)
torch.npu.set_compile_mode(jit_compile=False);torch.npu.matmul.allow_hf32=False;torch.set_num_threads(1)
from fast_attention import attention,AttentionConfig
source=BASE/'c4v8-integration-probes/evidence/owned-staged-decode-extended-v1/validation'
out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
report={'scope':'Same NPU generic decode vs saved strict staged decode, identical QKV','cases':[],'actual_grpo_updates':0}
with torch.inference_mode():
 for seq in [1024,4096,8192]:
  folders=list(source.glob(f'case-*-n{seq}-b1-h8-hk1'));assert len(folders)==1
  folder=folders[0]
  cpu=torch.load(folder/'cpu-reference.pt',map_location='cpu',weights_only=False)
  saved=torch.load(folder/'actual-npu-stages.pt',map_location='cpu',weights_only=False)
  qkv=[x.to('npu:1') for x in cpu['qkv']]
  actual=attention(*qkv,AttentionConfig(query_block=64,key_block=128),query_offset=seq-1).cpu()
  expected=saved['output'];delta=actual.float()-expected.float()
  row={'seq':seq,'bit_exact':torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)),'finite':bool(torch.isfinite(actual).all()),'different_values':int((actual!=expected).sum()),'max_abs':float(delta.abs().max()),'input_sha256':hashlib.sha256((folder/'cpu-reference.pt').read_bytes()).hexdigest()}
  report['cases'].append(row);torch.save({'actual':actual,'saved':expected},out/f'n{seq}.pt');print(row,flush=True)
report['passed']=all(c['bit_exact'] and c['finite'] for c in report['cases'])
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if report['passed'] else 1)
