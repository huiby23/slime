"""Allocate fresh Hybrid Adam state without performing an optimizer update."""
import types

def preallocate_hybrid_adam_state(hdo, *, preallocate_cpu_grads=True):
    import torch
    assert hdo.param_update_in_fp32
    assert not hdo.pin_cpu_params and not hdo.pin_cpu_grads
    assert not hdo.overlap_cpu_optimizer_d2h_h2d
    if getattr(hdo, "_hby_adam_preallocation_done", False):
        return hdo._hby_adam_preallocation_summary
    # Fresh-only guard: never reset a previously used optimizer.
    for opt in hdo.sub_optimizers:
        assert not opt.state, "Preallocation requires a fresh Hybrid optimizer"
        assert all("step" not in group for group in opt.param_groups)
        for group in opt.param_groups:
            assert not group.get("amsgrad", False), "AMSGrad is outside this fixed configuration"
            for inner in group["params"]:
                assert inner.dtype == torch.float32
                assert inner.grad is None
    summary = {"cpu_parameters": 0, "npu_parameters": 0, "moment_bytes": 0,
               "cpu_gradient_bytes": 0, "performed_optimizer_steps": 0}
    reserved = {}
    for opt in hdo.sub_optimizers:
        cpu = any(opt is value for value in hdo.cpu_optimizers)
        assert cpu or opt is hdo.gpu_optimizer
        for group in opt.param_groups:
            if not cpu:
                device = group["params"][0].device
                group["step"] = torch.zeros((), dtype=torch.int64, device=device)
            for inner in group["params"]:
                assert inner.device.type == ("cpu" if cpu else "npu")
                state = opt.state[inner]
                state["exp_avg"] = torch.zeros_like(inner, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(inner, dtype=torch.float32)
                if cpu:
                    state["step"] = torch.zeros((), dtype=torch.float32, device="cpu")
                    if preallocate_cpu_grads:
                        reserved[inner] = torch.zeros_like(inner, dtype=torch.float32, device="cpu")
                        assert not reserved[inner].is_pinned()
                        summary["cpu_gradient_bytes"] += inner.numel() * 4
                summary["cpu_parameters" if cpu else "npu_parameters"] += 1
                summary["moment_bytes"] += inner.numel() * 8
    hdo._sync_sub_optimizers_state_to_hdo()
    # Reserve memory now, bind only when the original parameter really has a grad.
    if reserved:
        original = hdo._set_sub_optimizer_grads
        def bind_reserved_grads(instance):
            for inner, buffer in instance._hby_reserved_cpu_grads.items():
                orig = instance.cpu_copys_map_gpu_param[inner]
                grad = getattr(orig, "decoupled_grad", orig.grad)
                if grad is not None and inner not in instance.cpu_copy_map_grad:
                    instance.cpu_copy_map_grad[inner] = buffer
                    inner.grad = buffer
            return original()
        hdo._hby_reserved_cpu_grads = reserved
        hdo._set_sub_optimizer_grads = types.MethodType(bind_reserved_grads, hdo)
    for orig, inner in hdo.param_to_fp32_param.items():
        assert hdo.state[orig]["master_param"] is inner
    hdo._hby_adam_preallocation_summary = summary
    hdo._hby_adam_preallocation_done = True
    return summary
