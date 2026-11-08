"""Evaluate cached clean actor weights in the same BF16 SGLang engine."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _write_mode(path, mode, rollout_id):
    record = dict(mode=mode, rollout_id=rollout_id, updated_at_utc=datetime.now(timezone.utc).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temp.write_text(json.dumps(record) + "\n")
    temp.replace(path)


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if not evaluation:
        raise ValueError("Clean evaluation wrapper is evaluation-only")
    if not args.sglang_disable_radix_cache:
        raise ValueError("Clean/noisy switching requires disabled radix cache")
    overrides = json.loads(args.sglang_json_model_override_args)
    audit = Path(overrides["fixed_noise_audit_path"])
    control = Path(overrides.get("fixed_noise_control_path", str(audit.with_name("noise-control.json"))))
    from slime.rollout.sglang_rollout import generate_rollout as native_generate

    _write_mode(control, "clean", rollout_id)
    completed = False
    try:
        result = native_generate(args, rollout_id, data_source, evaluation=True)
        completed = True
        return result
    finally:
        # The first subsequent noisy forward restores pristine+fixednoise.
        # No inference request is fabricated, so sampler RNG is untouched.
        _write_mode(control, "noisy", rollout_id)
        marker = dict(
            rollout_id=rollout_id,
            completed=completed,
            evaluation_precision="clean_bf16",
            noise_scale=overrides["fixed_noise_scale"],
            restore_requested="noisy",
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        with audit.with_name("clean-evaluation.jsonl").open("a") as stream:
            stream.write(json.dumps(marker) + "\n")
