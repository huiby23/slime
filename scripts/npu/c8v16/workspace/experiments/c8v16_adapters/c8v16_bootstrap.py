"""HTTP and child scheduler entry points for the dedicated C8V16 serving path."""

import os

from c8v16_profile import BACKEND_NAME, load_profile, record_event

_installed = None


def register_cli_choice():
    from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES, add_attention_backend_choices

    if BACKEND_NAME not in ATTENTION_BACKEND_CHOICES:
        add_attention_backend_choices([BACKEND_NAME])


def install_current_process(role, ranks=None):
    import torch
    import torch_npu  # registers the pinned torch.npu matmul controls
    from c8v16_sglang import install

    profile = load_profile()
    identity = (os.getpid(), profile.implementation_sha256, profile.policy_sha256, role)
    global _installed
    if _installed is not None:
        if _installed != identity:
            raise RuntimeError("Process already registered with a different C8V16 identity")
        return profile
    torch.npu.matmul.allow_hf32 = False
    if torch.npu.matmul.allow_hf32:
        raise RuntimeError("C8V16 requires FP32 matmul with HF32 disabled")

    def audit(event, details):
        if details.get("core_module", profile.core.__name__) != profile.core.__name__:
            raise RuntimeError("Actual SGLang attention differs from the loaded C8V16 core")
        record_event(profile, "HBY_C8V16_ROLLOUT_AUDIT_DIR", event, role=role,
                     backend=BACKEND_NAME, ranks=ranks or {}, **details)

    install(profile, audit)
    audit("bootstrap_registered", {"core_module": profile.core.__name__, "allow_hf32": False})
    _installed = identity
    return profile


def audit_effective_args(profile, args, role):
    selected = {k: getattr(args, k) for k in ("attention_backend", "prefill_attention_backend", "decode_attention_backend")}
    if set(selected.values()) != {BACKEND_NAME}:
        raise ValueError(f"Unexpected attention backends {selected}")
    if (args.dtype not in ("bfloat16", "bf16") or args.kv_cache_dtype not in ("auto", "bfloat16")
            or args.quantization is not None or args.device != "npu"
            or not args.disable_cuda_graph or args.enable_torch_compile or args.enable_mixed_chunk
            or args.speculative_algorithm not in (None, "NONE")
            or getattr(args, "enable_pdmux", False) or getattr(args, "enable_two_batch_overlap", False)
            or args.dp_size != 1 or args.pp_size != 1 or args.nnodes != 1 or args.encoder_only or args.grpc_mode):
        raise ValueError("Unsupported W16A16C8V16 SGLang configuration")
    record_event(profile, "HBY_C8V16_ROLLOUT_AUDIT_DIR", "effective_server_args",
                 role=role, backend=BACKEND_NAME, selectors=selected, dtype=args.dtype,
                 kv_cache_dtype=args.kv_cache_dtype, tp_size=args.tp_size)


def run_scheduler_process(server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, dp_rank, pipe_writer):
    ranks = dict(gpu_id=gpu_id, tp_rank=tp_rank, moe_ep_rank=moe_ep_rank, pp_rank=pp_rank, dp_rank=dp_rank)
    profile = install_current_process("scheduler", ranks)
    audit_effective_args(profile, server_args, "scheduler")
    from sglang.srt.managers.scheduler import run_scheduler_process as run_pinned_scheduler
    return run_pinned_scheduler(server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, dp_rank, pipe_writer)
