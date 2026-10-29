"""Generate with a small genuine Qwen3 MoE through the real W4A8 SGLang TP path."""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--tp',type=int,default=4)
    parser.add_argument('--dense',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    config_cls, model_cls = (Qwen3Config,Qwen3ForCausalLM) if args.dense else (Qwen3MoeConfig,Qwen3MoeForCausalLM)
    config=config_cls(vocab_size=151936,hidden_size=256,intermediate_size=256,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,head_dim=128,
        num_experts=8,num_experts_per_tok=2,moe_intermediate_size=256,
        max_position_embeddings=128,attention_dropout=0.0,torch_dtype='bfloat16')
    model=model_cls(config).bfloat16()
    original=args.output/'tiny-model'
    model.save_pretrained(original)
    del model
    os.environ['SGLANG_EXTERNAL_MODEL_PACKAGE']='w4a8_sim.sglang_models'
    os.environ['W4A8_ROLLOUT_AUDIT_DIR']=str(args.output/'audit')
    from sglang.srt.entrypoints.engine import Engine
    engine=Engine(model_path=str(original),tokenizer_path=str(args.tokenizer),tp_size=args.tp,
        dtype='bfloat16',context_length=128,mem_fraction_static=0.1,max_total_tokens=512,
        max_running_requests=2,disable_cuda_graph=True,
        disable_prefill_cuda_graph=True,disable_decode_cuda_graph=True,
        disable_overlap_schedule=True,moe_runner_backend='triton',moe_a2a_backend='none',
        json_model_override_args=json.dumps({'architectures':[
            'W4A8SimQwen3ForCausalLM' if args.dense else 'W4A8SimQwen3MoeForCausalLM']}))
    try:
        results=engine.generate(input_ids=[[1,2,3],[4,5,6]],sampling_params={
            'temperature':0,'max_new_tokens':4,'ignore_eos':True})
        assert len(results)==2 and all(r['meta_info']['completion_tokens']==4 for r in results)
        records=[json.loads(p.read_text()) for p in (args.output/'audit').glob('rollout-rank-*.json')]
        assert len(records)==args.tp
        for record in records:
            if args.dense:
                assert len(record['projections'])==8
            else:
                assert len(record['projections'])==4 and len(record['expert_banks'])==2 and len(record['fp32_routers'])==2
                assert record['counters'].get('expert_nonempty_fprop',0)>0
            assert all(record['counters'].get(k,0)>0 for k in ['fprop','weight_nvfp4','activation_e4m3'])
            if args.tp>1: assert record['counters']['weight_amax_reductions']>0
        (args.output/'validation.json').write_text(json.dumps(dict(passed=True,tp_size=args.tp,
            results=results,coverage=records,dense=args.dense,
            scope='Small random model integration, not full-size accuracy'),indent=2)+'\n')
    finally:
        engine.shutdown()


if __name__=='__main__':
    main()
