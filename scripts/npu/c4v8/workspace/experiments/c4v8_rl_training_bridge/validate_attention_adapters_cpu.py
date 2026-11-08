"""Same-QKV CPU teacher-forcing across actual adapters; NOT whole-model logprobs.

Actual SGLang adapter body is used with its pinned AST base/enum and an explicit
CPU test pool of the documented native shape. No live SG, NPU or fused kernel.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[2]
for name in ('c4v8_device_core','c4v8_rollout_bridge','c4v8_rl_training_bridge'):
    sys.path.insert(0,str(ROOT/'experiments'/name))
import torch
from device_attention import AttentionConfig,attention
from layouts import attention_sbhd,attention_thd
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
    delta=(a.float()-b.float()).abs()
    return dict(bit_exact=torch.equal(a,b),shape=list(a.shape),finite=bool(torch.isfinite(a).all()),
                differing_values=int((a!=b).sum()),max_abs=float(delta.max()))


def run_sg(documents,stages,config):
    hq,hkv,dim=documents[0][0].shape[1],documents[0][1].shape[1],documents[0][0].shape[2]
    pool=CpuPool(hkv,dim);request_indices=[2,0]
    maximum=max(len(q) for q,_,_ in documents)
    table=torch.full((3,maximum),-1,dtype=torch.int32)
    all_slots=torch.randperm(511,generator=torch.Generator().manual_seed(20261109))+1
    cursor=0;slots=[]
    for index,(q,_,_) in enumerate(documents):
        current=all_slots[cursor:cursor+len(q)];cursor+=len(q);slots.append(current)
        table[request_indices[index],:len(q)]=current.to(torch.int32)
    calls=[]
    def kernel(q,k,v,cfg,query_offset=0):
        calls.append(dict(query_tokens=q.shape[2],kv_tokens=k.shape[2],query_offset=query_offset))
        return attention(q,k,v,cfg,query_offset=query_offset)
    cls=make_backend_class(Base,core=kernel,config=config,pool_type=CpuPool)
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
            req_pool_indices=torch.tensor([request_indices[i] for i,_,_ in spans]),
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


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(690913)
    documents=[tuple(torch.randn(n,h,128,generator=generator).bfloat16() for h in (8,1,1)) for n in (140,97)]
    config=AttentionConfig(query_block=64,key_block=128)
    torch.save(dict(documents=documents,config=config.__dict__),args.output/'post-rope-qkv-fixture.pt')
    checks={};full=[]
    for q,k,v in documents:
        full.append(attention_sbhd(q[:,None],k[:,None],v[:,None],kernel=attention,config=config).squeeze(1).reshape_as(q))
    sg_full,audit_full=run_sg(documents,[([(0,0,140),(1,0,97)],Mode.EXTEND)],config)
    checks['same_qkv_batched_full_sg_vs_sbhd']=[compare(a,b) for a,b in zip(sg_full,full)]
    stages=[([(0,0,63)],Mode.EXTEND),([(0,63,129)],Mode.EXTEND),
            ([(0,129,139)],Mode.EXTEND),([(0,139,140)],Mode.DECODE)]
    sg_prefix,audit_prefix=run_sg(documents[:1],stages,config)
    checks['same_qkv_prefix_append_decode_vs_sbhd']=compare(sg_prefix[0],full[0])
    cu=torch.tensor([0,140,237],dtype=torch.int32)
    pq=torch.tensor([0,144,244],dtype=torch.int32);pk=torch.tensor([0,148,248],dtype=torch.int32)
    qpad=torch.zeros(244,8,128,dtype=torch.bfloat16)
    kpad=torch.zeros(248,1,128,dtype=torch.bfloat16);vpad=torch.zeros_like(kpad)
    for i,(q,k,v) in enumerate(documents):
        qpad[pq[i]:pq[i]+len(q)]=q;kpad[pk[i]:pk[i]+len(k)]=k;vpad[pk[i]:pk[i]+len(v)]=v
    packed=attention_thd(qpad,kpad,vpad,cu_seqlens_q=cu,cu_seqlens_kv=cu,
        cu_seqlens_q_padded=pq,cu_seqlens_kv_padded=pk,kernel=attention,config=config)
    checks['same_qkv_padded_thd_vs_batched_sg']=[compare(packed[pq[i]:pq[i]+len(q)],sg_full[i]) for i,(q,_,_) in enumerate(documents)]
    checks['thd_padding_zero']=bool((packed[140:144]==0).all() and (packed[241:244]==0).all())
    q,k,v=documents[0];changed_k=k.clone();changed_v=v.clone();changed_k[129:]*=3;changed_v[129:]*=-3
    changed=attention_sbhd(q[:,None],changed_k[:,None],changed_v[:,None],kernel=attention,config=config).squeeze(1).reshape_as(q)
    checks['future_kv_does_not_change_prefix']=compare(changed[:129],full[0][:129])
    changed_documents=[documents[0],(documents[1][0],documents[1][1]*3,documents[1][2]*-3)]
    changed_sg,_=run_sg(changed_documents,[([(0,0,140),(1,0,97)],Mode.EXTEND)],config)
    checks['sg_request_isolation']=compare(changed_sg[0],sg_full[0])
    sources=[Path(__file__),ROOT/'experiments/c4v8_rl_training_bridge/layouts.py',ROOT/'experiments/c4v8_rl_training_bridge/bridge.py']
    sources += [ROOT/'experiments/c4v8_rollout_bridge'/name for name in ('sglang_bridge.py','logical_kv.py','test_sglang_bridge.py','pinned-source/base_attn_backend.py','pinned-source/forward_batch_info.py')]
    sources += [ROOT/'experiments/c4v8_device_core'/name for name in ('device_attention.py','device_quant.py','exact_fp32.py')]
    def accepted(value):
        if isinstance(value,bool):return value
        if isinstance(value,list):return all(accepted(x) for x in value)
        return value.get('bit_exact',False) and value.get('finite',False)
    accepted_checks=all(accepted(v) for v in checks.values())
    accepted_audits=all(a[k] for a in (audit_full,audit_prefix) for k in
        ('cached_prefix_unchanged','final_cache_exact','invalid_write_rejected_before_mutation'))
    report=dict(scope='CPU attention-adapter same-post-RoPE-QKV teacher forcing only',
        checks=checks,full_audit=audit_full,prefix_audit=audit_prefix,passed=accepted_checks and accepted_audits,
        actual_adapter_functions_exercised=True,pinned_base_enum_ast_fixture=True,cpu_cache_fixture=True,
        live_sglang_executed=False,megatron_model_executed=False,npu_validated=False,fused_candidate_validated=False,
        projection_qknorm_rope_validated=False,whole_model_logits_or_logprobs_validated=False,
        train_rollout_teacher_forcing_agreement_gate=False,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
