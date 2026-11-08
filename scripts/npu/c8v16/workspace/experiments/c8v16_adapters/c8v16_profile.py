"""Shared policy and implementation identity for Megatron and spawned SGLang workers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CORE_DIR = HERE.parent / "c8v16_attention"
CUBE_FORMAT = os.environ.get("SLIME_FA_CUBE_FORMAT", "mxfp8")
if CUBE_FORMAT not in ("mxfp8", "hif4"):
    raise ValueError(f"Unknown FA Cube format {CUBE_FORMAT!r}")
IS_HIF4 = CUBE_FORMAT == "hif4"
BACKEND_NAME = "hby_hif4_c4v16_sim" if IS_HIF4 else "hby_c8v16_sim"
PRECISION = "W16A16C4V16" if IS_HIF4 else "W16A16C8V16"
CORE_MODULE = "hif4_attention" if IS_HIF4 else "c8v16_attention"
HIF4_DIR = HERE.parents[3] / "hif4"


def _shared_experiments():
    # A deployed overlay puts both families in the same experiments directory.
    if (HERE.parent / "c4v8_device_core").is_dir():
        return HERE.parent
    # In the Git checkout each experiment has its own workspace overlay.
    return HERE.parents[3] / "c4v8" / "workspace" / "experiments"


SHARED = _shared_experiments()
MODULES = {
    "c8v16_attention": CORE_DIR,
    "device_quant": SHARED / "c4v8_device_core",
    "exact_fp32": SHARED / "c4v8_device_core",
    "bridge": SHARED / "c4v8_rl_training_bridge",
    "layouts": SHARED / "c4v8_rl_training_bridge",
    "megatron_spec": SHARED / "c4v8_rl_training_bridge",
    "sglang_bridge": SHARED / "c4v8_rollout_bridge",
    "logical_kv": SHARED / "c4v8_rollout_bridge",
}

if IS_HIF4:
    MODULES.update({name: HIF4_DIR for name in ("hif4_quant", "hif4_triton", "hif4_attention", "hif4_decode", "hif4_checks")})
else:
    MODULES.update(fused_mxfp8=SHARED / "c4v8_fused_qdq", nvfp4_triton=SHARED / "c4v8_fused_nvfp4")


def module_paths():
    return tuple(dict.fromkeys(MODULES.values()))


def _canonical_import(name):
    module = importlib.import_module(name)
    expected = (MODULES[name] / f"{name}.py").resolve(strict=True)
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError(f"Conflicting source for {name}; expected {expected}")
    return module


def load_module(name):
    for directory in reversed(module_paths()):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    return _canonical_import(name)


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def implementation_sha256():
    sources = [directory / f"{name}.py" for name, directory in MODULES.items()]
    sources.extend(HERE / filename for filename in (
        "c8v16_profile.py", "c8v16_megatron_spec.py", "c8v16_bootstrap.py",
        "c8v16_runtime.py", "c8v16_sglang.py", "python-c8v16",
    ))
    return _digest({f"{p.parent.name}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sources})


@dataclass(frozen=True)
class Profile:
    core: object
    config: object
    policy: dict
    policy_sha256: str
    implementation_sha256: str


def load_profile():
    policy_file = Path(os.environ["HBY_C8V16_POLICY_FILE"])
    policy = json.loads(policy_file.read_text())
    required = {"query_block", "key_block", "causal", "vector", "reduction", "quantize_operands"}
    if IS_HIF4:
        required.update(("p_method", "cube_format"))
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError("Incomplete or unknown numerical policy fields")
    if (policy["causal"] is not True or policy["quantize_operands"] is not True
            or policy["vector"] != "bf16" or policy["reduction"] != "mixed_fp32_reduction"):
        raise ValueError("Expected causal quantized Cube and BF16 Vector policy")
    policy_sha = _digest(policy)
    implementation_sha = implementation_sha256()
    if os.environ["HBY_C8V16_POLICY_SHA256"] != policy_sha:
        raise ValueError("C8V16 policy SHA differs from the launcher")
    if os.environ["HBY_C8V16_IMPLEMENTATION_SHA256"] != implementation_sha:
        raise ValueError("C8V16 implementation SHA differs from the launcher")
    # Check even transitive generic module names so an earlier import cannot
    # substitute a different implementation under an otherwise valid digest.
    for name in ("exact_fp32", "device_quant"):
        load_module(name)
    shared = load_module("c8v16_attention")
    if IS_HIF4:
        load_module("hif4_quant")
    core = load_module(CORE_MODULE)
    if hasattr(shared.torch, "npu"):
        for name in (("hif4_triton", "hif4_decode") if IS_HIF4 else ("nvfp4_triton", "fused_mxfp8")):
            load_module(name)
    return Profile(core, core.AttentionConfig(**policy), policy, policy_sha, implementation_sha)


def record_event(profile, audit_env, event, **details):
    directory = Path(os.environ[audit_env])
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(event=event, pid=os.getpid(), implementation_sha256=profile.implementation_sha256,
                  policy_sha256=profile.policy_sha256, policy=profile.policy, **details)
    path = directory / f"{event}-pid{os.getpid()}.json"
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
