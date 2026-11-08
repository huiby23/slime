"""Small-tensor stage diagnostic. CPU transfers here are intentional tracing only."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,inspect,json,re,subprocess,time
from pathlib import Path
import torch
import device_attention as core
from device_quant import nvfp4_values,mxfp8_values


def compare(a,b):
    valid=torch.isfinite(a)&torch.isfinite(b)
    delta=(a[valid]-b[valid]).float()
    neq=(a!=b)&~(torch.isnan(a)&torch.isnan(b))
    indices=neq.nonzero()[:4]
    return {'different_count':int(neq.sum()),'numel':a.numel(),
            'max_abs':float(delta.abs().max()) if delta.numel() else 0.,
            'relative_l2':float(delta.norm()/b[valid].float().norm().clamp_min(1e-30)) if delta.numel() else 0.,
            'examples':[{'index':i.tolist(),'npu':float(a[tuple(i)]),'cpu':float(b[tuple(i)])} for i in indices]}


def record(q,k,v,config):
    original_c4,original_v8=core.c4_qdq,core.v8_qdq
    events=[]
    def wrapper(fn,kind):
        def execute(x,**kwargs):
            out=fn(x,**kwargs)
            finite=torch.where(torch.isneginf(x),torch.zeros_like(x),x)
            data=(nvfp4_values if kind=='c4' else mxfp8_values)(finite,logical_bytes=True)
            events.append({'kind':kind,'shape':list(x.shape),
                           'caller':[f'{s.function}:{s.lineno}' for s in inspect.stack(context=0)[1:5]],
                           'input':x.detach().cpu(),'output':out.detach().cpu(),
                           'codes':data.payload_bytes.cpu(),'scales':data.scale_bytes.cpu()})
            return out
        return execute
    try:
        core.c4_qdq=wrapper(original_c4,'c4');core.v8_qdq=wrapper(original_v8,'v8')
        output=core.attention(q,k,v,config)
    finally:
        core.c4_qdq,core.v8_qdq=original_c4,original_v8
    return output.detach().cpu(),events


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    p.add_argument('--heads',type=int,default=4);p.add_argument('--kv-heads',type=int,default=2)
    p.add_argument('--head-dim',type=int,default=32)
    p.add_argument('--query-block',type=int,default=16);p.add_argument('--key-block',type=int,default=32)
    p.add_argument('--layout-checks',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    snap=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
    (a.output/'npu-before.txt').write_text(snap.stdout+snap.stderr)
    idle=sorted(set(int(i) for i in re.findall(r'No running processes found in NPU\s+(\d+)',snap.stdout)))
    if snap.returncode or idle!=list(range(8)):raise RuntimeError('NPU idle gate failed')
    import torch_npu
    torch.set_num_threads(1);torch.npu.set_device(1);device=torch.device('npu:1')
    torch.npu.matmul.allow_hf32=False;torch.npu.set_per_process_memory_fraction(.05,device)
    g=torch.Generator().manual_seed(20261108);torch.randn(3,67,generator=g)
    vals=tuple(torch.randn(1,h,32,a.head_dim,generator=g).to(getattr(torch,a.dtype)) for h in (a.heads,a.kv_heads,a.kv_heads))
    config=core.AttentionConfig(query_block=a.query_block,key_block=a.key_block)
    cpu,ce=record(*vals,config);npu,ne=record(*(x.to(device) for x in vals),config)
    torch.save({'inputs':vals,'cpu_events':ce,'npu_events':ne,'cpu_output':cpu,'npu_output':npu},a.output/'trace.pt')
    assert len(ce)==len(ne)
    rows=[]
    for i,(c,n) in enumerate(zip(ce,ne)):
        assert c['kind']==n['kind'] and c['shape']==n['shape']
        rows.append({'event':i,'kind':c['kind'],'shape':c['shape'],'caller':c['caller'],
                     'input':compare(n['input'],c['input']),'output':compare(n['output'],c['output']),
                     'code_differences':int((n['codes']!=c['codes']).sum()),
                     'scale_differences':int((n['scales']!=c['scales']).sum())})
    report={'output':compare(npu,cpu),'event_count':len(rows),'events':rows,
            'config':config.metadata(),'dtype':a.dtype,'heads':a.heads,'kv_heads':a.kv_heads,'head_dim':a.head_dim,
            'scope':'Diagnostic CPU copies at every QDQ boundary, not production execution or a throughput measurement'}
    if a.layout_checks:
        inputs=tuple(x.to(device) for x in vals)
        with torch.no_grad():
            full=core.attention(*inputs,config)
            decoded=torch.cat([core.attention(inputs[0][:,:,i:i+1],inputs[1][:,:,:i+1],inputs[2][:,:,:i+1],config,query_offset=i)
                               for i in range(32)],2)
            report['same_npu_decode']=compare(decoded.cpu(),full.cpu())
            if a.heads==8 and a.kv_heads==1:
                split=torch.cat([core.attention(inputs[0][:,i:i+4],inputs[1],inputs[2],config) for i in (0,4)],1)
                report['same_npu_head_split']=compare(split.cpu(),full.cpu())
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':report['output'],'events':len(rows),'first_code_differences':[r for r in rows if r['code_differences']][:5]},indent=2))
