"""Observe real Megatron optimizer steps without changing gradients or updates.

Enabled by the before-step hook. The FP32 witness is an optimizer-owned master
tensor, or a bounded sample of a large local shard. Records cover
one transformer projection's gradient, the global gradient norm, master deltas,
and all seven NVFP4 recipe counters during each actual forward/backward step.
"""

import json
import logging
import math
from pathlib import Path
from types import MethodType

import torch

KEYS = ("fprop", "dgrad", "wgrad", "rht_x", "rht_g", "sr_dgrad", "sr_wgrad")
logger = logging.getLogger(__name__)


def _witness(optimizer):
    fallback = None
    for group_id, group in enumerate(optimizer.param_groups):
        for param_id, parameter in enumerate(group["params"]):
            if parameter.dtype == torch.float32 and parameter.numel() > 0:
                description = {
                    "group": group_id,
                    "parameter": param_id,
                    "numel": parameter.numel(),
                    "dtype": str(parameter.dtype),
                    "observed_numel": parameter.numel(),
                }
                if 4096 <= parameter.numel() <= 8_000_000:
                    return parameter, description
                if fallback is None or parameter.numel() > fallback[0].numel():
                    fallback = parameter, description
    if fallback is not None:
        # DP ranks may own only a large embedding shard or small norm shards.
        # Retain the actual master parameter and reread its sample after step().
        parameter, description = fallback
        description["observed_numel"] = min(parameter.numel(), 1_048_576)
        return parameter, description
    raise RuntimeError("No nonempty FP32 optimizer master parameter found")


def _master_sample(context):
    return context["master"].detach().reshape(-1)[: context["master_description"]["observed_numel"]]


def before_train_step(args, rollout_id, step_id, models, optimizer, scheduler):
    """Slime's existing pre-step hook installs a passive observer once."""
    if not hasattr(optimizer, "_nvfp4_validation_context"):
        recipes = {}
        projections = []
        for chunk in models:
            for module in chunk.modules():
                recipe = vars(module).get("_nvfp4_recipe")
                if recipe is not None:
                    recipes[id(recipe)] = recipe
            projections.extend(
                (name, parameter)
                for name, parameter in chunk.named_parameters()
                if name.endswith("mlp.linear_fc1.weight")
            )
        if not recipes or not projections:
            raise RuntimeError("Training validation requires the installed NVFP4 model")
        master, description = _witness(optimizer)
        context = {
            "recipes": list(recipes.values()),
            "projection": projections[0],
            "master": master,
            "master_description": description,
            "path": Path(args.save).parent
            / (
                "optimizer-validation.jsonl"
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
                else f"optimizer-validation-rank-{torch.distributed.get_rank()}.jsonl"
            ),
        }
        original_step = optimizer.step

        def observed_step(self, *step_args, **step_kwargs):
            with torch.no_grad():
                initial = _master_sample(context).clone()
                name, projection = context["projection"]
                gradient = projection.main_grad
                projection_finite = bool(torch.isfinite(gradient).all())
                projection_norm = float(gradient.float().norm())
            outcome = original_step(*step_args, **step_kwargs)
            with torch.no_grad():
                delta = _master_sample(context) - initial
                delta_finite = bool(torch.isfinite(delta).all())
                delta_max = float(delta.abs().max())
                delta_count = int(torch.count_nonzero(delta))
            global_norm = float(outcome[1]) if outcome[1] is not None else float("nan")
            current = {key: sum(r.counters[key] for r in context["recipes"]) for key in KEYS}
            record = {
                "rollout_id": context["rollout_id"],
                "step_id": context["step_id"],
                "optimizer_success": bool(outcome[0]),
                "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
                "global_grad_norm": global_norm,
                "grad_finite": projection_finite and math.isfinite(global_norm),
                "projection": name,
                "projection_grad_norm": projection_norm,
                "master": context["master_description"],
                "master_delta_finite": delta_finite,
                "master_max_abs_delta": delta_max,
                "master_changed_elements": delta_count,
                "recipe_counter_deltas": {key: current[key] - context["counter_start"][key] for key in KEYS},
            }
            record["active_learning_step"] = (
                record["optimizer_success"]
                and record["grad_finite"]
                and global_norm > 0
                and projection_norm > 0
                and delta_finite
                and delta_count > 0
                and all(value > 0 for value in record["recipe_counter_deltas"].values())
            )
            context["path"].parent.mkdir(parents=True, exist_ok=True)
            with context["path"].open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            logger.warning("NVFP4 optimizer validation: %s", json.dumps(record))
            return outcome

        optimizer.step = MethodType(observed_step, optimizer)
        optimizer._nvfp4_validation_context = context
    context = optimizer._nvfp4_validation_context
    context.update(rollout_id=rollout_id, step_id=step_id)
    context["counter_start"] = {key: sum(r.counters[key] for r in context["recipes"]) for key in KEYS}
