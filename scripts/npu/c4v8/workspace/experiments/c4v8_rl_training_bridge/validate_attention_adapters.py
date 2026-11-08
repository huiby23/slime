"""Same-QKV teacher-forcing across actual adapters; NOT whole-model logprobs.

Actual SGLang adapter body is used with its pinned AST base/enum and an explicit
CPU test pool by default. Explicit --execute-npu requires a root-reserved device
and uses the real pinned SG base/enum/native pool plus the frozen fast core.
No HTTP/model/logprob claim is made by this attention-adapter-only test.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import re
import inspect
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD","0")
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[2]
for name in ('c4v8_device_core','c4v8_rollout_bridge','c4v8_rl_training_bridge'):
    sys.path.insert(0,str(ROOT/'experiments'/name))
import torch
from device_attention import AttentionConfig,attention
from layouts import attention_sbhd,attention_thd
from megatron_spec import C4V8CoreAttention
from sglang_bridge import make_backend_class
from test_sglang_bridge import Base,Mode


class CpuPool:
    """Explicit CPU fixture, not a genuine accelerator allocation."""
    def __init__(self,hkv,dim):
        self.k=torch.full((4,128,hkv,dim),13,dtype=torch.bfloat16)
        self.v=torch.full_like(self.k,-7);self.use_fia=False;self.writes=0
    def get_key_buffer(self,layer):return self.k
    def get_value_buffer(self,layer):return self.v
    def set_kv_buffer(self,layer,locations,k,v):
        self.writes+=1
        self.k.view(-1,*self.k.shape[2:])[locations]=k
        self.v.view(-1,*self.v.shape[2:])[locations]=v


def compare(a,b):
    a,b=a.detach().cpu().contiguous(),b.detach().cpu().contiguous()
    if a.dtype!=b.dtype or a.shape!=b.shape:raise ValueError('Comparison tensors differ in shape/dtype')
    integer=torch.int16 if a.element_size()==2 else torch.int32
    bitdiff=a.view(integer)!=b.view(integer)
    numeric=a!=b;zeros=(a==0)&(b==0)
    delta=(a.float()-b.float()).abs()
    return dict(bit_exact=not bool(bitdiff.any()),numeric_equal=not bool(numeric.any()),
                signed_zero_only=bool(bitdiff.any()) and not bool((bitdiff&~zeros).any()),
                shape=list(a.shape),finite=bool(torch.isfinite(a).all()),
                differing_values=int(numeric.sum()),differing_bits_at_values=int(bitdiff.sum()),
                max_abs=float(delta.max()))


def run_sg(documents,stages,config):
    hq,hkv,dim=documents[0][0].shape[1],documents[0][1].shape[1],documents[0][0].shape[2]
    pool=Pool(hkv,dim);request_indices=[2,0]
    device=documents[0][0].device
    maximum=max(len(q) for q,_,_ in documents)
    table=torch.full((3,maximum),-1,dtype=torch.int32,device=device)
    all_slots=(torch.randperm(511,generator=torch.Generator().manual_seed(20261109))+1).to(device)
    cursor=0;slots=[]
    for index,(q,_,_) in enumerate(documents):
        current=all_slots[cursor:cursor+len(q)];cursor+=len(q);slots.append(current)
        table[request_indices[index],:len(q)]=current.to(torch.int32)
    calls=[]
    def kernel(q,k,v,cfg,query_offset=0):
        calls.append(dict(query_tokens=q.shape[2],kv_tokens=k.shape[2],query_offset=query_offset))
        return attention(q,k,v,cfg,query_offset=query_offset)
    cls=make_backend_class(Base,core=kernel,config=config,pool_type=Pool)
    runner=NS(page_size=128,use_mla_backend=False,server_args=NS(disable_cuda_graph=True,
        enable_torch_compile=False,speculative_algorithm=None,enable_mixed_chunk=False))
    backend=cls(runner)
    layer=NS(tp_q_head_num=hq,tp_k_head_num=hkv,tp_v_head_num=hkv,qk_head_dim=dim,v_head_dim=dim,
        scaling=1/math.sqrt(dim),is_cross_attention=False,attn_type='decoder',layer_id=0,
        logit_cap=0.,sliding_window_size=-1,quant_method=None)
    outputs=[torch.zeros_like(q) for q,_,_ in documents];prefix_unchanged=True
    for spans,mode in stages:
        lengths=[end-start for _,start,end in spans]
        q,k,v=[torch.cat([documents[i][which][start:end] for i,start,end in spans]) for which in range(3)]
        locations=torch.cat([slots[i][start:end] for i,start,end in spans])
        prefix_locations=torch.cat([slots[i][:start] for i,start,_ in spans])
        prefix_k=pool.k.view(-1,hkv,dim)[prefix_locations].clone()
        prefix_v=pool.v.view(-1,hkv,dim)[prefix_locations].clone()
        batch=NS(forward_mode=mode,batch_size=len(spans),
            req_pool_indices=torch.tensor([request_indices[i] for i,_,_ in spans],device=device),
            seq_lens_cpu=torch.tensor([end for _,_,end in spans]),extend_seq_lens_cpu=lengths,
            extend_prefix_lens_cpu=[start for _,start,_ in spans],out_cache_loc=locations,
            token_to_kv_pool=pool,req_to_token_pool=NS(req_to_token=table))
        backend.init_forward_metadata(batch)
        value=backend.forward(q.flatten(1),k,v,layer,batch,True).reshape(-1,hq,dim)
        offset=0
        for (index,start,end),length in zip(spans,lengths):outputs[index][start:end]=value[offset:offset+length];offset+=length
        prefix_unchanged &= torch.equal(prefix_k,pool.k.view(-1,hkv,dim)[prefix_locations])
        prefix_unchanged &= torch.equal(prefix_v,pool.v.view(-1,hkv,dim)[prefix_locations])
    cache_matches=all(torch.equal(pool.k.view(-1,hkv,dim)[slots[i]],k) and
        torch.equal(pool.v.view(-1,hkv,dim)[slots[i]],v) for i,(_,k,v) in enumerate(documents))
    # Invalid physical append locations must fail before any cache mutation.
    before=pool.writes
    wrong=NS(**vars(batch));wrong.out_cache_loc=locations+1
    backend.init_forward_metadata(wrong)
    try:backend.forward(q.flatten(1),k,v,layer,wrong,True)
    except ValueError:negative=pool.writes==before
    else:negative=False
    return outputs,dict(core_calls=calls,cached_prefix_unchanged=prefix_unchanged,
        final_cache_exact=cache_matches,invalid_write_rejected_before_mutation=negative,
        metadata_device_to_host_transfers=backend.metadata_device_to_host_transfers)


Pool=CpuPool


def main():
    global Pool,Base,Mode,attention
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute-npu',action='store_true')
    parser.add_argument('--policy',type=Path,help='Explicit source-matched numerical policy; required for actual NPU')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    actual_sources=[];device='cpu';implementation=None
    if args.execute_npu and args.policy is None:raise ValueError('Actual NPU test requires an explicit policy file')
    policy_value=json.loads(args.policy.read_text()) if args.policy else AttentionConfig(query_block=64,key_block=128).__dict__
    if set(policy_value)!={'query_block','key_block','vector','quantize_operands','causal','reduction'}:raise ValueError('Incomplete/unknown numerical policy')
    if policy_value['query_block'] not in (64,256,512,1024) or policy_value['key_block']!=128:raise ValueError('Unreviewed probe tile policy')
    if args.execute_npu:
        if os.environ.get('HBY_C4V8_PROBE_NPU_RESERVED')!='1':raise RuntimeError('Root must reserve physical NPU1')
        before=subprocess.check_output(['npu-smi','info'],text=True)
        (args.output/'npu-before.txt').write_text(before)
        idle=sorted(set(map(int,re.findall(r'No running processes found in NPU\s+(\d+)',before))))
        if idle!=list(range(8)):raise RuntimeError('All-eight idle gate failed')
        import torch_npu
        torch.npu.set_device(1);torch.npu.set_per_process_memory_fraction(.05,1)
        torch.npu.matmul.allow_hf32=False
        os.environ['ASCEND_USE_FIA']='False'
        os.environ['HBY_C4V8_ALLOW_UNVALIDATED_FUSION']='1'
        from numerical_profile import implementation_profile,python_paths
        implementation=implementation_profile(ROOT,'fast_attention')
        sys.path[:0]=list(map(str,python_paths(ROOT,'fast_attention')))
        from fast_attention import attention
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend as Base
        from sglang.srt.model_executor.forward_batch_info import ForwardMode as Mode
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
        class AuditedNativePool(NPUMHATokenToKVPool):
            def __init__(self,hkv,dim):
                super().__init__(size=512,page_size=128,dtype=torch.bfloat16,head_num=hkv,
                    head_dim=dim,layer_num=1,device='npu:1',enable_memory_saver=False,
                    enable_alt_stream=False,enable_kv_cache_copy=False)
                self.k=self.get_key_buffer(0);self.v=self.get_value_buffer(0);self.writes=0
            def set_kv_buffer(self,*args,**kwargs):
                self.writes+=1
                return super().set_kv_buffer(*args,**kwargs)
        Pool=AuditedNativePool;device='npu:1'
        actual_sources=[Path(inspect.getfile(obj)) for obj in (Base,Mode,NPUMHATokenToKVPool)]
    sources=[Path(__file__),ROOT/'experiments/c4v8_rl_training_bridge/megatron_spec.py',*actual_sources,ROOT/'experiments/c4v8_rl_training_bridge/layouts.py',ROOT/'experiments/c4v8_rl_training_bridge/bridge.py']
    sources += [ROOT/'experiments/c4v8_rollout_bridge'/name for name in ('sglang_bridge.py','logical_kv.py','test_sglang_bridge.py','pinned-source/base_attn_backend.py','pinned-source/forward_batch_info.py')]
    sources += [ROOT/'experiments/c4v8_device_core'/name for name in ('device_attention.py','device_quant.py','exact_fp32.py')]
    if args.policy:sources.append(args.policy)
    if implementation is not None:sources += [Path(name) for name in implementation['source_sha256']]
    source_before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    generator=torch.Generator().manual_seed(690913)
    documents=[tuple(torch.randn(n,h,128,generator=generator).bfloat16() for h in (8,1,1)) for n in (140,97)]
    config=AttentionConfig(**policy_value)
    trainer=C4V8CoreAttention(NS(context_parallel_size=1,attention_dropout=0.,kv_channels=128),
        layer_number=1,attn_mask_type="causal",kernel=attention,precision_config=config)
    torch.save(dict(documents=documents,config=config.__dict__),args.output/'post-rope-qkv-fixture.pt')
    documents=[tuple(x.to(device) for x in values) for values in documents]
    checks={};full=[]
    for q,k,v in documents:
        mask=torch.ones((1,1,len(q),len(q)),dtype=torch.bool,device=device).triu(1)
        full.append(trainer(q[:,None],k[:,None],v[:,None],attention_mask=mask).squeeze(1).reshape_as(q))
    try:
        q,k,v=documents[0]
        trainer(q[:,None],k[:,None],v[:,None],attention_mask=torch.zeros((1,1,len(q),len(q)),dtype=torch.bool,device=device))
    except ValueError:checks['noncausal_explicit_mask_rejected']=True
    else:checks['noncausal_explicit_mask_rejected']=False
    sg_full,audit_full=run_sg(documents,[([(0,0,140),(1,0,97)],Mode.EXTEND)],config)
    checks['same_qkv_batched_full_sg_vs_sbhd']=[compare(a,b) for a,b in zip(sg_full,full)]
    stages=[([(0,0,63)],Mode.EXTEND),([(0,63,129)],Mode.EXTEND),
            ([(0,129,139)],Mode.EXTEND),([(0,139,140)],Mode.DECODE)]
    sg_prefix,audit_prefix=run_sg(documents[:1],stages,config)
    checks['same_qkv_prefix_append_decode_vs_sbhd']=compare(sg_prefix[0],full[0])
    cu=torch.tensor([0,140,237],dtype=torch.int32,device=device)
    pq=torch.tensor([0,144,244],dtype=torch.int32,device=device);pk=torch.tensor([0,148,248],dtype=torch.int32,device=device)
    qpad=torch.zeros(244,8,128,dtype=torch.bfloat16,device=device)
    kpad=torch.zeros(248,1,128,dtype=torch.bfloat16,device=device);vpad=torch.zeros_like(kpad)
    for i,(q,k,v) in enumerate(documents):
        qpad[pq[i]:pq[i]+len(q)]=q;kpad[pk[i]:pk[i]+len(k)]=k;vpad[pk[i]:pk[i]+len(v)]=v
    packed=trainer(qpad,kpad,vpad,packed_seq_params=NS(qkv_format="thd",local_cp_size=1,
        cu_seqlens_q=cu,cu_seqlens_kv=cu,cu_seqlens_q_padded=pq,cu_seqlens_kv_padded=pk))
    checks['same_qkv_padded_thd_vs_batched_sg']=[compare(packed[pq[i]:pq[i]+len(q)],sg_full[i]) for i,(q,_,_) in enumerate(documents)]
    checks['thd_padding_zero']=bool((packed[140:144]==0).all() and (packed[241:244]==0).all())
    q,k,v=documents[0];changed_k=k.clone();changed_v=v.clone();changed_k[129:]*=3;changed_v[129:]*=-3
    changed=trainer(q[:,None],changed_k[:,None],changed_v[:,None]).squeeze(1).reshape_as(q)
    checks['future_kv_does_not_change_prefix']=compare(changed[:129],full[0][:129])
    changed_documents=[documents[0],(documents[1][0],documents[1][1]*3,documents[1][2]*-3)]
    changed_sg,_=run_sg(changed_documents,[([(0,0,140),(1,0,97)],Mode.EXTEND)],config)
    checks['sg_request_isolation']=compare(changed_sg[0],sg_full[0])
    def accepted(value):
        if isinstance(value,bool):return value
        if isinstance(value,list):return all(accepted(x) for x in value)
        return value.get('finite',False) and (value.get('bit_exact',False) or value.get('signed_zero_only',False))
    accepted_checks=all(accepted(v) for v in checks.values())
    accepted_audits=all(a[k] for a in (audit_full,audit_prefix) for k in
        ('cached_prefix_unchanged','final_cache_exact','invalid_write_rejected_before_mutation'))
    source_after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    checks['source_identity_unchanged']=source_before==source_after
    accepted_checks=accepted_checks and checks['source_identity_unchanged']
    torch.save(dict(sbhd=[x.detach().cpu() for x in full],sg_full=[x.detach().cpu() for x in sg_full],
                    sg_prefix=sg_prefix[0].detach().cpu(),thd=packed.detach().cpu()),args.output/'outputs.pt')
    policy=config.__dict__
    report=dict(scope='Attention-adapter same-post-RoPE-QKV teacher forcing only',
        execution_device=device,implementation=implementation,policy=policy,
        policy_sha256=hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        outer_log_fallback_and_owned_release_review_still_required=True,
        checks=checks,full_audit=audit_full,prefix_audit=audit_prefix,passed=accepted_checks and accepted_audits,
        actual_adapter_functions_exercised=True,pinned_base_enum_ast_fixture=not args.execute_npu,cpu_cache_fixture=not args.execute_npu,
        live_sglang_executed=False,megatron_model_executed=False,
        actual_megatron_core_attention_exercised=True,npu_executed=args.execute_npu,
        attention_adapter_teacher_forcing_agreement=bool(args.execute_npu and accepted_checks and accepted_audits),
        projection_qknorm_rope_validated=False,whole_model_logits_or_logprobs_validated="unknown",
        train_rollout_teacher_forcing_agreement_gate=False,
        source_sha256=source_before,source_sha256_after=source_after)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
