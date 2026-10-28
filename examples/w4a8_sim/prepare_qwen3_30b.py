"""Prepare an inspectable four-GPU EP=4 W4A8 run without allocating GPUs.

The GPU launcher must separately verify the integration, initialization download,
runtime arguments and idle ownership. Estimates are not a peak-memory benchmark.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def plan(model, output, data3, data4, gpus, updates=100):
    config = json.loads((model / "config.json").read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())
    if config["model_type"] != "qwen3_moe" or config.get("quantization_config"):
        raise ValueError("Expected original BF16 Qwen3 MoE")
    if len(gpus) != 4 or len(set(gpus)) != 4 or config["num_experts"] % 4:
        raise ValueError("This plan requires four distinct GPUs and divisible expert count")
    weight_bytes = index["metadata"]["total_size"]
    params = weight_bytes // 2
    experts = 3 * config["hidden_size"] * config["moe_intermediate_size"] * config["num_experts"] * config["num_hidden_layers"]
    shared = params - experts
    local_params = experts // 4 + shared
    save_root = data4 / "run"
    storage = [save_root / ".snapshots", data3 / "snapshots"]
    state_bytes = params * 14
    safety_bytes = int(state_bytes * 1.05)
    model_args = ["--num-layers",str(config["num_hidden_layers"]),"--hidden-size",str(config["hidden_size"]),
        "--ffn-hidden-size",str(config["intermediate_size"]),"--num-attention-heads",str(config["num_attention_heads"]),
        "--group-query-attention","--num-query-groups",str(config["num_key_value_heads"]),
        "--kv-channels",str(config["head_dim"]),"--qk-layernorm","--swiglu","--disable-bias-linear",
        "--normalization","RMSNorm","--norm-epsilon",str(config["rms_norm_eps"]),
        "--position-embedding-type","rope","--rotary-base",str(int(config["rope_theta"])),
        "--vocab-size",str(config["vocab_size"]),"--untie-embeddings-and-output-weights",
        "--num-experts",str(config["num_experts"]),"--moe-ffn-hidden-size",str(config["moe_intermediate_size"]),
        "--moe-router-topk",str(config["num_experts_per_tok"]),"--moe-router-score-function","softmax",
        "--moe-router-dtype","fp32","--moe-layer-freq","1","--moe-grouped-gemm",
        "--moe-token-dispatcher-type","alltoall","--moe-aux-loss-coeff","0"]
    command = ["python3","/quant/examples/w4a8_sim/train_best_last.py"] + model_args + [
        "--hf-checkpoint",str(model),"--load",str(model),"--save",str(save_root),
        "--custom-model-provider-path","w4a8_sim.megatron.model_provider",
        "--actor-num-nodes","1","--actor-num-gpus-per-node","4","--rollout-num-gpus","4","--num-gpus-per-node","4",
        "--tensor-model-parallel-size","1","--pipeline-model-parallel-size","1","--context-parallel-size","1",
        "--expert-model-parallel-size","4","--expert-tensor-parallel-size","1",
        "--colocate","--offload-train","--offload-rollout","--optimizer-cpu-offload","--optimizer-offload-fraction","1.0",
        "--use-torch-optimizer-for-cpu-offload","--use-precision-aware-optimizer","--bf16","--use-distributed-optimizer",
        "--main-grads-dtype","fp32","--main-params-dtype","fp32","--exp-avg-dtype","fp32","--exp-avg-sq-dtype","fp32",
        "--optimizer","adam","--lr","1e-6","--lr-decay-style","constant","--weight-decay","0.1",
        "--adam-beta1","0.9","--adam-beta2","0.98","--adam-eps","1e-8","--clip-grad","1.0",
        "--seed","42","--num-rollout",str(updates),"--rollout-batch-size","8","--n-samples-per-prompt","8",
        "--global-batch-size","64","--micro-batch-size","1","--seq-length","2048",
        "--rollout-max-response-len","1024","--eval-max-response-len","1024","--rollout-temperature","1.0",
        "--rollout-top-p","1.0","--rollout-top-k","-1","--apply-chat-template",
        "--apply-chat-template-kwargs",json.dumps({"enable_thinking":False}),"--rollout-shuffle",
        "--input-key","prompt","--label-key","label","--rm-type","math",
        "--prompt-data","/experiment/datasets/gsm8k-seed42-val512/train.jsonl",
        "--eval-prompt-data","gsm8k","/experiment/datasets/gsm8k-seed42-val512/validation.jsonl",
        "--n-samples-per-eval-prompt","1","--eval-interval","25","--eval-top-k","1","--eval-top-p","1.0",
        "--advantage-estimator","grpo","--eps-clip","0.2","--eps-clip-high","0.28",
        "--kl-coef","0","--kl-loss-coef","0","--entropy-coef","0",
        "--attention-dropout","0","--hidden-dropout","0","--attention-backend","flash","--attention-softmax-in-fp32",
        "--recompute-granularity","full","--recompute-method","uniform","--recompute-num-layers","1",
        "--save-interval","1","--distrib-optim-sharding-type","fully_reshardable",
        "--custom-eval-rollout-log-function-path","w4a8_sim.checkpoint_policy.eval_log_hook",
        "--use-tensorboard","--tensorboard-dir",str(output/"tensorboard"),
        "--rollout-num-gpus-per-engine","4","--sglang-mem-fraction-static","0.6",
        "--sglang-moe-runner-backend","triton","--sglang-moe-a2a-backend","none",
        "--sglang-disable-cuda-graph","--sglang-disable-piecewise-cuda-graph","--sglang-disable-overlap-schedule",
        "--sglang-max-running-requests","8","--sglang-context-length","2048",
        "--sglang-json-model-override-args",json.dumps({"architectures":["W4A8SimQwen3MoeForCausalLM"]}),
        "--tokenizer-path",str(model)]
    environment = {"PYTHONPATH":"/quant/slime_plugins:/workspace/slime:/root/Megatron-LM",
        "SGLANG_EXTERNAL_MODEL_PACKAGE":"w4a8_sim.sglang_models","CUDA_DEVICE_MAX_CONNECTIONS":"1",
        "W4A8_CHECKPOINT_ROOTS":json.dumps([str(p) for p in storage]),"W4A8_CHECKPOINT_BYTES":str(safety_bytes),
        "SLIME_DESTROY_WORLD_PROCESS_GROUP":"0","OMP_NUM_THREADS":"2","NVTE_ALLOW_NONDETERMINISTIC_ALGO":"0",
        "CUBLAS_WORKSPACE_CONFIG":":4096:8","TMPDIR":str(output/"tmp"),"RAY_TMPDIR":str(output/"ray"),
        "TORCHINDUCTOR_CACHE_DIR":str(output/"cache/inductor"),"TRITON_CACHE_DIR":str(output/"cache/triton")}
    return dict(model="Qwen/Qwen3-30B-A3B", model_path=str(model), physical_gpus=gpus,
        precision={"name":"custom-w4a8-simulation","weights":"NVFP4 E2M1, E4M3 scales, FP32 tensor scale, 16x16",
                   "activations":"FP8 E4M3, 1x64 FP32 scales","backward_gradients":"FP8 E5M2, 1x64 FP32 scales",
                   "attention_core":"baseline BF16/FP32", "router":"FP32", "lm_head":"BF16",
                   "native_low_precision_instructions":False},
        topology={"train_tp":1,"train_pp":1,"train_cp":1,"train_ep":4,"train_etp":1,"rollout_tp":4,"rollout_ep":1},
        estimates={"parameters":params,"bf16_weights_bytes":weight_bytes,"expert_parameters":experts,
                   "per_gpu_weights_and_main_grads_bytes":local_params*6,"total_fp32_master_adam_cpu_bytes":params*12,
                   "single_complete_checkpoint_bytes":state_bytes,"checkpoint_required_free_bytes":safety_bytes,
                   "scope":"Excludes activations, runtime buffers, quantization temporaries; requires measured peak validation"},
        environment=environment, training_command=command, checkpoint_storage_roots=[str(p) for p in storage],
        formal_launch_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkpoint-primary",type=Path,required=True)
    parser.add_argument("--checkpoint-overflow",type=Path,required=True)
    parser.add_argument("--gpus",type=int,nargs=4,default=[1,2,3,6])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result=plan(args.model.resolve(),args.output.resolve(),args.checkpoint_overflow.resolve(),args.checkpoint_primary.resolve(),args.gpus)
    result['storage']={str(p):dict(available_bytes=shutil.disk_usage(p).free,writable=os.access(p,os.W_OK))
                       for p in [args.model,args.checkpoint_primary,args.checkpoint_overflow]}
    result['simulator_revision']=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    (args.output/'training-plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['estimates'],indent=2))


if __name__=='__main__':
    main()
