"""Prepare matched native BF16 and linear W4A8 runs for original Qwen3-4B.

This produces inspectable commands, not a claim that full-size CUDA validation
has passed. Both arms use the same initialization, data, topology and seed.
"""

import argparse
import json
from pathlib import Path


def prepare(model, output, checkpoint_root, gpus, updates=100):
    config = json.loads((model / "config.json").read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())
    if config["model_type"] != "qwen3" or config.get("quantization_config"):
        raise ValueError("Expected original BF16 dense Qwen3 initialization")
    if config["num_hidden_layers"] != 36 or config["hidden_size"] != 2560 or not config["tie_word_embeddings"]:
        raise ValueError("This experiment requires original tied-embedding Qwen3-4B")
    if len(gpus) != len(set(gpus)) or not 1 <= len(gpus) <= 8:
        raise ValueError("Use between one and eight distinct GPUs")
    # Keep the matched 64-sample global batch for every allocation. All GPUs
    # generate samples; training uses the largest DP degree dividing that batch.
    train_gpus = max(n for n in range(1, len(gpus) + 1) if 64 % n == 0)
    parameters = index["metadata"]["total_size"] // 2
    common = ["--num-layers",str(config["num_hidden_layers"]),"--hidden-size",str(config["hidden_size"]),
        "--ffn-hidden-size",str(config["intermediate_size"]),"--num-attention-heads",str(config["num_attention_heads"]),
        "--group-query-attention","--num-query-groups",str(config["num_key_value_heads"]),
        "--kv-channels",str(config["head_dim"]),"--qk-layernorm","--swiglu","--disable-bias-linear",
        "--normalization","RMSNorm","--norm-epsilon",str(config["rms_norm_eps"]),
        "--position-embedding-type","rope","--rotary-base",str(int(config["rope_theta"])),
        "--vocab-size",str(config["vocab_size"]),"--hf-checkpoint",str(model),"--load",str(model),
        "--actor-num-nodes","1","--actor-num-gpus-per-node",str(train_gpus),
        "--rollout-num-gpus",str(len(gpus)),"--num-gpus-per-node",str(len(gpus)),
        "--tensor-model-parallel-size","1","--pipeline-model-parallel-size","1",
        "--context-parallel-size","1","--expert-model-parallel-size","1","--expert-tensor-parallel-size","1",
        "--colocate","--offload-train","--offload-rollout","--optimizer-cpu-offload",
        "--optimizer-offload-fraction","1.0","--use-torch-optimizer-for-cpu-offload",
        "--use-precision-aware-optimizer","--bf16","--use-distributed-optimizer",
        "--main-grads-dtype","fp32","--main-params-dtype","fp32","--exp-avg-dtype","fp32","--exp-avg-sq-dtype","fp32",
        "--optimizer","adam","--lr","1e-6","--lr-decay-style","constant","--weight-decay","0.1",
        "--adam-beta1","0.9","--adam-beta2","0.98","--adam-eps","1e-8","--clip-grad","1.0",
        "--seed","42","--start-rollout-id","0","--num-rollout",str(updates),"--rollout-batch-size","8","--n-samples-per-prompt","8",
        "--global-batch-size","64","--micro-batch-size","1","--seq-length","2048",
        "--rollout-max-response-len","1024","--eval-max-response-len","1024",
        "--rollout-temperature","1.0","--rollout-top-p","1.0","--rollout-top-k","-1",
        "--apply-chat-template","--apply-chat-template-kwargs",json.dumps({"enable_thinking":False}),
        "--rollout-shuffle","--input-key","prompt","--label-key","label","--rm-type","math",
        "--prompt-data","/experiment/datasets/gsm8k-seed42-val512/train.jsonl",
        "--eval-prompt-data","gsm8k","/experiment/datasets/gsm8k-seed42-val512/validation.jsonl",
        "--n-samples-per-eval-prompt","1","--eval-interval","25","--eval-top-k","1","--eval-top-p","1.0",
        "--advantage-estimator","grpo","--eps-clip","0.2","--eps-clip-high","0.28",
        "--kl-coef","0","--kl-loss-coef","0","--entropy-coef","0",
        "--attention-dropout","0","--hidden-dropout","0","--attention-backend","flash","--attention-softmax-in-fp32",
        "--recompute-granularity","full","--recompute-method","uniform","--recompute-num-layers","1",
        "--save-interval","1","--distrib-optim-sharding-type","fully_reshardable",
        "--custom-eval-rollout-log-function-path","w4a8_sim.checkpoint_policy.eval_log_hook",
        "--custom-megatron-before-train-step-hook-path","w4a8_sim.training_audit.before_step",
        "--use-tensorboard","--rollout-num-gpus-per-engine","1","--sglang-mem-fraction-static","0.55",
        "--sglang-disable-cuda-graph","--sglang-disable-prefill-cuda-graph","--sglang-disable-decode-cuda-graph",
        "--sglang-disable-overlap-schedule","--sglang-max-running-requests","8","--sglang-context-length","2048",
        "--tokenizer-path",str(model)]
    result = dict(model="Qwen/Qwen3-4B",model_path=str(model),physical_gpus=gpus,
        formal_launch_ready=False,scope="Linear W4A8 simulation; FA core retains baseline precision",
        topology=dict(train_tp=1,train_dp=train_gpus,train_ep=1,rollout_tp=1,rollout_engines=len(gpus)),
        estimates=dict(parameters=parameters,full_checkpoint_bytes=parameters*14,
            best_weights_bytes=parameters*2,two_arm_peak_including_shared_initial_bytes=parameters*48,
            inference_fp32_quantized_weight_cache_bytes=parameters*4),arms={})
    for mode in ("bf16","w4a8"):
        work, save = output/mode, checkpoint_root/mode
        work.mkdir(parents=True,exist_ok=True)
        save.mkdir(parents=True,exist_ok=True)
        command=["python3","/quant/examples/w4a8_sim/train_best_last.py"] + common + [
            "--save",str(save),"--tensorboard-dir",str(work/"tensorboard")]
        if mode=="w4a8":
            command += ["--custom-model-provider-path","w4a8_sim.megatron.model_provider",
                "--sglang-json-model-override-args",json.dumps({"architectures":["W4A8SimQwen3ForCausalLM"]})]
        environment={"PYTHONPATH":"/quant/slime_plugins:/workspace/slime:/root/Megatron-LM",
            "SGLANG_EXTERNAL_MODEL_PACKAGE":"w4a8_sim.sglang_models","W4A8_TRAINING_MODE":mode,
            "W4A8_INITIAL_MODEL":str(model),"W4A8_BEST_WEIGHTS_ONLY":"1",
            "W4A8_STORAGE_FIRST_FIT":"1","W4A8_CHECKPOINT_ROOTS":json.dumps([str(save/".snapshots")]),
            "W4A8_CHECKPOINT_BYTES":str(int(parameters*14*1.05)),"W4A8_BEST_BYTES":str(int(parameters*2*1.05)),
            "W4A8_ROLLOUT_AUDIT_DIR":str(work/"rollout-audit"),"SLIME_DESTROY_WORLD_PROCESS_GROUP":"0",
            "W4A8_TRAINING_AUDIT_DIR":str(work/"training-audit"),
            "W4A8_AUDIT_PER_ENGINE":"1",
            "W4A8_ISOLATED_RAY":"1","W4A8_RUN_NAMESPACE":"qwen3-4b-"+mode,
            "TENSORBOARD_DIR":str(work/"tensorboard"),
            "OMP_NUM_THREADS":"2","NVTE_ALLOW_NONDETERMINISTIC_ALGO":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8",
            "CUDA_DEVICE_MAX_CONNECTIONS":"1","PYTHONDONTWRITEBYTECODE":"1",
            "TMPDIR":str(work/"tmp"),"RAY_TMPDIR":"/data2/huibingyu/r4b/"+mode,
            "TORCHINDUCTOR_CACHE_DIR":str(work/"cache/inductor"),"TRITON_CACHE_DIR":str(work/"cache/triton")}
        result["arms"][mode]=dict(training_command=command,environment=environment,save=str(save),work=str(work))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,required=True)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result=prepare(args.model.resolve(),args.output.resolve(),args.checkpoints.resolve(),args.gpus)
    (args.output/'training-plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['estimates'],indent=2))
