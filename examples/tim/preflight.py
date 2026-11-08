"""CPU-only validation of the final frozen bundle against pinned Slime."""

import dataclasses
import json
import os
import sys
from pathlib import Path


def validate(plan):
    import torch
    from megatron.core.optimizer import OptimizerConfig
    from slime.utils.arguments import parse_args
    from slime.utils.misc import load_function
    from slime_plugins.tim.bf16.train import runtime_protocol_check
    from slime_plugins.tim.bf16.training_validation import expected_optimizer_settings

    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Preflight must run without visible CUDA devices")
    sys.argv = ["tim-preflight"] + plan["training_command"][3:]
    args, _ = parse_args(return_restore_plan=True)
    config = plan["config"]
    t, o, e = (config[k] for k in ("training", "optimizer", "evaluation"))
    assert args.bf16 and not args.fp8 and not args.fp4
    assert args.finetune and args.no_load_optim and args.no_load_rng and args.start_rollout_id == 0
    assert args.hf_checkpoint == args.load == plan["paths"]["model_container"]
    assert args.num_rollout == t["updates"] and args.rollout_batch_size == t["prompts"]
    assert args.n_samples_per_prompt == t["answers"] and args.global_batch_size == t["prompts"] * t["answers"]
    assert args.micro_batch_size == t["micro_batch_size"]
    assert args.rollout_max_response_len == args.eval_max_response_len == t["response_tokens"]
    assert args.lr == args.min_lr == o["lr"] and args.lr_decay_style == "constant" and args.lr_warmup_iters == 0
    assert args.weight_decay == args.start_weight_decay == args.end_weight_decay == o["weight_decay"]
    assert args.adam_beta1 == o["beta1"] and args.adam_beta2 == o["beta2"]
    assert args.eval_interval == e["interval"] and not args.skip_eval_before_train
    assert not args.use_tis and not args.use_rs and not args.use_stateless_adam
    assert not args.use_rollout_logprobs and args.get_mismatch_metrics
    assert args.colocate == config["topology"]["colocate"] and not args.offload_train and not args.offload_rollout
    optimizer = OptimizerConfig(
        **{
            field.name: getattr(args, field.name)
            for field in dataclasses.fields(OptimizerConfig)
            if hasattr(args, field.name)
        }
    )
    assert optimizer.decoupled_weight_decay and args.optimizer == "adam"
    assert callable(load_function(args.custom_megatron_before_train_step_hook_path))
    assert callable(load_function(args.custom_eval_rollout_log_function_path))
    if config["trainer"] == "nvfp4_sim":
        assert args.custom_model_provider_path == "slime_plugins.nvfp4_sim.megatron.model_provider"
        assert callable(load_function(args.custom_model_provider_path))
    else:
        assert not args.custom_model_provider_path
    override = json.loads(args.sglang_json_model_override_args or "{}")
    mode = config["rollout"]["precision"]
    noise_validation = None
    if mode == "bf16":
        assert not override and not os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
    elif mode == "nvfp4_sim":
        assert override["architectures"] == ["NVFP4SimQwen3ForCausalLM"]
        assert override["nvfp4_sim_quantize_weights"] and override["nvfp4_sim_quantize_activations"]
        assert override["nvfp4_sim_weight_cache"] and override["nvfp4_sim_reference_arithmetic"]
        assert os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] == "slime_plugins.tim.nvfp4_models"
    else:
        from prepare_noise_bank import source_identity, verify_bank

        assert override["fixed_noise_scale"] == config["noise"]["sigma"]
        assert args.eval_function_path == "slime_plugins.tim.fixed_noise.eval.generate_rollout"
        assert os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] == "slime_plugins.tim.fixed_noise.sglang_models"
        bank = verify_bank(Path("/noise-bank"), source_identity(Path(args.hf_checkpoint)), config["noise"]["seed"])
        noise_validation = dict(
            passed=True,
            seed=bank["seed"],
            canonical_tensors=bank["canonical_tensor_count"],
            elementcount=bank["elementcount"],
            runtime_scale=config["noise"]["sigma"],
        )
    from slime.observability.tensorboard_utils import SummaryWriter

    assert SummaryWriter is not None and args.use_tensorboard
    protocol = runtime_protocol_check(args, "/runtime/slime/train.py")
    assert not torch.cuda.is_initialized(), "CPU preflight initialized CUDA"
    return dict(
        passed=True,
        gpu_jobs_launched=0,
        cuda_initialized=False,
        source_bundle_sha256=plan["source_bundle_sha256"],
        image_id=plan["image_id"],
        training_precision=config["trainer"],
        rollout_precision=mode,
        evaluation_precision=e["precision"],
        optimizer=expected_optimizer_settings(args),
        protocol=protocol,
        dataset=plan["data"],
        noise_validation=noise_validation,
    )
