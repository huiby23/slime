"""Passive evidence of native BF16 AdamW steps; never modifies a gradient."""

import json
import math
from pathlib import Path
from types import MethodType


def optimizer_tree(optimizer):
    pending, seen, result = [optimizer], set(), []
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        result.append(item)
        for name in ("optimizer", "optimizers", "chained_optimizers"):
            child = getattr(item, name, None)
            if child is None:
                continue
            pending.extend(child if isinstance(child, (list, tuple)) else [child])
    return result


def expected_optimizer_settings(args):
    """Read the constant AdamW contract from the parsed Slime arguments.

    This function and ``inspect_optimizer`` are CPU-only and import no Torch,
    Ray, or Megatron modules, so the final parsed configuration can be checked
    during preflight. Parameter groups may retain native LR/WD multipliers.
    """
    assert args.bf16 and not args.fp8 and not args.fp4, "Native BF16 trainer required"
    assert args.optimizer == "adam", "AdamW validation requires the Adam optimizer"
    assert args.lr_decay_style == "constant" and args.lr_warmup_iters == 0
    assert not getattr(args, "lr_warmup_fraction", None), "Warmup is not constant LR"
    lr, wd = float(args.lr), float(args.weight_decay)
    betas = (float(args.adam_beta1), float(args.adam_beta2))
    assert math.isfinite(lr) and lr > 0, lr
    assert math.isfinite(wd) and wd >= 0, wd
    assert all(math.isfinite(beta) and 0 <= beta < 1 for beta in betas), betas
    minimum = getattr(args, "min_lr", None)
    assert minimum is None or math.isclose(float(minimum), lr, rel_tol=1e-7, abs_tol=1e-14)
    assert getattr(args, "weight_decay_incr_style", "constant") == "constant"
    return {"lr": lr, "weight_decay": wd, "betas": betas}


def inspect_optimizer(optimizer, args):
    expected = expected_optimizer_settings(args)
    objects = optimizer_tree(optimizer)
    types = [type(x).__module__ + "." + type(x).__name__ for x in objects]
    adam_modes = [int(x.adam_w_mode) for x in objects if hasattr(x, "adam_w_mode")]
    config_modes = [
        bool(x.config.decoupled_weight_decay)
        for x in objects
        if getattr(x, "config", None) is not None and hasattr(x.config, "decoupled_weight_decay")
    ]
    is_torch_adamw = any(
        cls.__name__ == "AdamW" and cls.__module__.startswith("torch.optim.")
        for obj in objects
        for cls in type(obj).__mro__
    )
    assert config_modes and all(config_modes), "Megatron decoupled AdamW required"
    assert (adam_modes and all(x == 1 for x in adam_modes)) or is_torch_adamw, types
    groups = []
    for group in optimizer.param_groups:
        betas = tuple(float(x) for x in group.get("betas", ()))
        assert betas == expected["betas"], (betas, expected["betas"])
        lr, wd = float(group["lr"]), float(group["weight_decay"])
        lr_multiplier = float(group.get("lr_mult", 1))
        assert math.isclose(lr, expected["lr"] * lr_multiplier, rel_tol=1e-7, abs_tol=1e-14), lr
        multiplier = float(group.get("wd_mult", 0 if wd == 0 else 1))
        assert math.isclose(wd, expected["weight_decay"] * multiplier, rel_tol=1e-7, abs_tol=1e-12), (wd, multiplier)
        groups.append(
            dict(
                lr=lr,
                lr_mult=lr_multiplier,
                betas=list(betas),
                weight_decay=wd,
                wd_mult=multiplier,
                parameters=len(group["params"]),
            )
        )
    return dict(
        implementation_types=types, adam_w_mode=adam_modes, decoupled_weight_decay=config_modes, param_groups=groups
    )


def master_witness(optimizer):
    import torch

    fallback = None
    for group_id, group in enumerate(optimizer.param_groups):
        for param_id, param in enumerate(group["params"]):
            if param.dtype == torch.float32 and param.numel() > 0:
                info = dict(
                    group=group_id,
                    parameter=param_id,
                    numel=param.numel(),
                    observed_numel=min(param.numel(), 1_048_576),
                    dtype=str(param.dtype),
                )
                if 4096 <= param.numel() <= 8_000_000:
                    return param, info
                if fallback is None or param.numel() > fallback[0].numel():
                    fallback = param, info
    if fallback is None:
        raise RuntimeError("No nonempty FP32 optimizer master parameter")
    return fallback


def before_train_step(args, rollout_id, step_id, models, optimizer, scheduler):
    import torch

    expected_optimizer_settings(args)
    if not hasattr(optimizer, "_bf16_validation_context"):
        assert not any(
            vars(module).get("_nvfp4_recipe") is not None for model in models for module in model.modules()
        ), "Unexpected QDQ recipe"
        projections = [
            (name, p)
            for model in models
            for name, p in model.named_parameters()
            if name.endswith("mlp.linear_fc1.weight")
        ]
        assert projections, "Missing transformer projection witness"
        master, description = master_witness(optimizer)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        context = dict(
            master=master,
            master_description=description,
            projection=projections[0],
            path=Path(args.save).parent
            / ("optimizer-validation.jsonl" if rank == 0 else f"optimizer-validation-rank-{rank}.jsonl"),
            rank=rank,
        )
        original_step = optimizer.step

        def observed_step(self, *step_args, **step_kwargs):
            with torch.no_grad():
                initial = context["master"].detach().reshape(-1)[: description["observed_numel"]].clone()
                name, projection = context["projection"]
                gradient = projection.main_grad
                grad_finite = bool(torch.isfinite(gradient).all())
                projection_norm = float(gradient.float().norm())
            hyperparameters = inspect_optimizer(self, args)
            outcome = original_step(*step_args, **step_kwargs)
            with torch.no_grad():
                delta = context["master"].detach().reshape(-1)[: description["observed_numel"]] - initial
                delta_finite = bool(torch.isfinite(delta).all())
                changed = int(torch.count_nonzero(delta))
                delta_max = float(delta.abs().max())
            global_norm = float(outcome[1]) if outcome[1] is not None else float("nan")
            record = dict(
                rollout_id=context["rollout_id"],
                step_id=context["step_id"],
                optimizer_success=bool(outcome[0]),
                rank=context["rank"],
                world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
                global_grad_norm=global_norm,
                grad_finite=grad_finite and math.isfinite(global_norm),
                projection=name,
                projection_grad_norm=projection_norm,
                master=description,
                master_delta_finite=delta_finite,
                master_max_abs_delta=delta_max,
                master_changed_elements=changed,
                optimizer=hyperparameters,
                precision="native_bf16",
                quantization_recipe_modules=0,
            )
            record["active_learning_step"] = bool(
                record["optimizer_success"]
                and record["grad_finite"]
                and global_norm > 0
                and projection_norm > 0
                and delta_finite
                and changed > 0
            )
            context["path"].parent.mkdir(parents=True, exist_ok=True)
            with context["path"].open("a") as stream:
                stream.write(json.dumps(record, allow_nan=True) + "\n")
            print("BF16 optimizer validation: " + json.dumps(record, allow_nan=True), flush=True)
            return outcome

        optimizer.step = MethodType(observed_step, optimizer)
        optimizer._bf16_validation_context = context
    optimizer._bf16_validation_context.update(rollout_id=rollout_id, step_id=step_id)
