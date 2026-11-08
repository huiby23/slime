"""CPU-only subprocess test with explicitly stubbed SGLang/hardware modules.

Never run this as a real server. Artifacts are labelled cpu_stubbed_sglang.
The spawned target is the real c4v8_bootstrap.run_scheduler_process function.
"""
from __future__ import annotations

import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'

import argparse
import ast
from dataclasses import dataclass
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'c4v8_device_core'))
sys.path.insert(0,str(ROOT))


def _stub(name):
    module=ModuleType(name)
    module.__path__=[]
    sys.modules[name]=module
    if '.' in name:
        parent,leaf=name.rsplit('.',1)
        if parent not in sys.modules:_stub(parent)
        setattr(sys.modules[parent],leaf,module)
    return module


def _ast_function(path,name,scope):
    parsed=ast.parse(path.read_text())
    node=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile('from __future__ import annotations\n'+ast.unparse(node),str(path),'exec'),scope)
    return scope[name]


def _fake_native_scheduler(server_args,port_args,gpu_id,tp_rank,moe_ep_rank,pp_rank,dp_rank,pipe_writer):
    from test_sglang_bridge import FakePool,Mode
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang_bridge import BACKEND_NAME
    cls=ATTENTION_BACKENDS[BACKEND_NAME]
    runner=NS(page_size=4,use_mla_backend=False,server_args=server_args,tp_rank=tp_rank,pp_rank=pp_rank)
    backend=cls(runner)
    layer=NS(tp_q_head_num=2,tp_k_head_num=1,tp_v_head_num=1,qk_head_dim=16,v_head_dim=16,
             scaling=.25,is_cross_attention=False,attn_type='decoder',layer_id=0,
             logit_cap=0.,sliding_window_size=-1,quant_method=None)
    pool=FakePool(1,16)
    mapping=torch.full((1,8),-1,dtype=torch.int32)
    mapping[0,:3]=torch.tensor([7,11,3])
    gen=torch.Generator().manual_seed(1700+tp_rank)
    q,k,v=(torch.randn((3,h,16),generator=gen).bfloat16() for h in (2,1,1))
    outputs=[]
    for start,end,mode in ((0,2,Mode.EXTEND),(2,3,Mode.DECODE)):
        batch=NS(forward_mode=mode,batch_size=1,req_pool_indices=torch.tensor([0]),
                 seq_lens_cpu=torch.tensor([end]),extend_seq_lens_cpu=[end-start],
                 extend_prefix_lens_cpu=[start],out_cache_loc=mapping[0,start:end],
                 token_to_kv_pool=pool,req_to_token_pool=NS(req_to_token=mapping))
        backend.init_forward_metadata(batch)
        outputs.append(backend.forward(q[start:end].flatten(1),k[start:end],v[start:end],layer,batch))
    from device_attention import attention,AttentionConfig
    policy=json.loads(Path(os.environ['HBY_C4V8_POLICY_FILE']).read_text())
    expected=attention(*(x.transpose(0,1).unsqueeze(0) for x in (q,k,v)),AttentionConfig(**policy))
    expected=expected.squeeze(0).transpose(0,1).flatten(1)
    torch.testing.assert_close(torch.cat(outputs),expected,atol=0,rtol=0)
    pipe_writer.send(dict(status='ready',test_only=True,tp_rank=tp_rank,pid=os.getpid(),
                          registered=list(ATTENTION_BACKENDS),forward_cpu_equal=True,
                          torch_npu_imported='torch_npu' in sys.modules))
    pipe_writer.close()


def install_stubs():
    from test_sglang_bridge import Base,FakePool
    base=_stub('sglang.srt.layers.attention.base_attn_backend');base.AttentionBackend=Base
    registry=_stub('sglang.srt.layers.attention.attention_registry');registry.ATTENTION_BACKENDS={}
    args=_stub('sglang.srt.server_args');args.ATTENTION_BACKEND_CHOICES=['ascend']
    args.add_attention_backend_choices=lambda choices:args.ATTENTION_BACKEND_CHOICES.extend(choices)
    pool=_stub('sglang.srt.hardware_backend.npu.memory_pool_npu');pool.NPUMHATokenToKVPool=FakePool
    allocator=_stub('sglang.srt.hardware_backend.npu.allocator_npu')
    allocator.NPUPagedTokenToKVPoolAllocator=type('FakeNPUAllocator',(),{})
    mixin=_stub('sglang.srt.model_executor.model_runner_kv_cache_mixin')
    mixin.ModelRunnerKVCacheMixin=type('FakeKVMixin',(),{'init_memory_pool':lambda *a,**k:None})
    common=_stub('sglang.srt.mem_cache.common')
    common.get_last_loc=lambda *a:None
    common.get_last_loc_torch=lambda *a:None
    common.get_global_server_args=lambda:NS(attention_backend='hby_c4v8_sim')
    utils=_stub('sglang.srt.hardware_backend.npu.utils')
    utils.set_default_server_args=_ast_function(ROOT/'pinned-source/npu_utils.py','set_default_server_args',
                                              {'get_npu_mem_capacity':lambda:65536,
                                               'get_npu_memory_capacity':lambda:65536})
    scheduler=_stub('sglang.srt.managers.scheduler');scheduler.run_scheduler_process=_fake_native_scheduler


if os.environ.get('HBY_C4V8_CPU_SPAWN_PROBE')=='1':
    # Spawn imports __mp_main__ before unpickling its real target. Unlike fork,
    # the child receives empty registries and must perform its own installation.
    install_stubs()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):parser.error('Choose fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    policy={'query_block':5,'key_block':8,'vector':'mxfp8','quantize_operands':True,
            'causal':True,'reduction':'mxfp8_tree_reduction'}
    path=out/'policy.json';path.write_text(json.dumps(policy))
    os.environ.update(HBY_C4V8_CORE_MODULE='device_attention',HBY_C4V8_POLICY_FILE=str(path),
                      HBY_C4V8_ROLLOUT_AUDIT_DIR=str(out/'audit'),HBY_C4V8_CPU_SPAWN_PROBE='1')
    os.environ.pop('HBY_C4V8_EXPECTED_RUNTIME_SHA256',None)
    torch.set_num_threads(1)
    install_stubs()
    from c4v8_bootstrap import install_current_process,audit_effective_server_args,run_scheduler_process
    from sglang_bridge import BACKEND_NAME
    install_current_process('http_parent')
    from sglang.srt.hardware_backend.npu import utils
    server=NS(attention_backend=BACKEND_NAME,prefill_attention_backend=None,decode_attention_backend=None,
              page_size=4,chunked_prefill_size=128,cuda_graph_max_bs=None,tp_size=2,
              enable_hierarchical_cache=False,dtype='bfloat16',kv_cache_dtype='auto',disable_cuda_graph=True,
              enable_torch_compile=False,enable_mixed_chunk=False,speculative_algorithm=None,
              quantization=None,dp_size=1,pp_size=1,nnodes=1,encoder_only=False,grpc_mode=False,
              enable_memory_saver=True,device='npu',model_path='/CPU_PROBE_NO_MODEL',context_length=128,
              weight_version='cpu_probe')
    utils.set_default_server_args(server)
    audit_effective_server_args(server,'http_parent')
    ctx=mp.get_context('spawn')
    procs=[];readers=[]
    for rank in range(2):
        reader,writer=ctx.Pipe(duplex=False)
        child=ctx.Process(target=run_scheduler_process,args=(server,None,rank,rank,0,0,None,writer))
        child.start();writer.close();procs.append(child);readers.append(reader)
    messages=[]
    for child,reader in zip(procs,readers):
        if not reader.poll(30):raise TimeoutError('CPU probe child did not complete')
        messages.append(reader.recv());reader.close();child.join(30)
        if child.is_alive():
            child.terminate();child.join(5)
            raise TimeoutError('CPU probe child remained alive')
        if child.exitcode!=0:raise RuntimeError(f'CPU probe child exited {child.exitcode}')
    rows=[json.loads(p.read_text()) for p in (out/'audit').glob('*.json')]
    created=[r for r in rows if r['event']=='backend_created']
    assert {r['ranks']['tp_rank'] for r in created}=={0,1}
    assert len({r['runtime_sha256'] for r in rows})==1
    assert all(r['execution_validation_kind']=='cpu_stubbed_sglang' for r in rows)
    assert len([r for r in rows if r['event']=='bootstrap_registered'])==3
    assert len([r for r in rows if r['event']=='first_extend'])==2
    assert len([r for r in rows if r['event']=='first_decode'])==2
    assert all(not row['torch_npu_imported'] and row['forward_cpu_equal'] for row in messages)
    report={'passed':True,'process_start_method':'spawn','worker_count':2,
            'parent_pid':os.getpid(),'worker_results':messages,'audit_records':len(rows),
            'scope':'real bootstrap/registry installation/spawn path with explicitly stubbed SG/hardware dependencies',
            'live_sglang_or_npu_validated':False}
    (out/'spawn-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
