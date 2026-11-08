"""Synthetic reward used only to validate the Ascend RL installation.

This is deliberately not a mathematical correctness reward or a benchmark.
Alternating rewards produce different advantages inside each sampled group.
"""


async def synthetic_reward(args, sample, **kwargs):
    if isinstance(sample, list):
        return [await synthetic_reward(args, item, **kwargs) for item in sample]
    if sample.index is None:
        raise ValueError("The installation smoke test requires assigned sample indices")
    return 1.0 if int(sample.index) % 2 else -1.0

def record_attention_backend(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    """Record actual training attention modules and their autograd graph."""
    if step_id != 0:
        return
    import json
    from collections import deque
    from pathlib import Path

    candidates = [
        (name, module)
        for chunk in model
        for name, module in chunk.named_modules()
        if type(module).__name__ == "FlashAttention"
        and type(module).__module__.startswith("mindspeed.te.")
    ]
    output = Path(args.ci_save_grad_norm.format(
        role="actor", rollout_id=rollout_id, step_id=step_id
    )).parent / f"attention_{rollout_id}.json"
    report = {
        "rollout_id": rollout_id,
        "matched_module_count": len(candidates),
        "module": None,
        "fused_attention_backward_in_graph": False,
        "autograd_node_types": [],
    }
    if candidates:
        name, module = candidates[0]
        report["module"] = f"{name}: {type(module).__module__}.{type(module).__name__}"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("SMOKE_ATTENTION_MODULE=" + str(report["module"]), flush=True)
    if not candidates:
        return

    holder = {}

    def observe(_module, _inputs, result):
        node = getattr(result, "grad_fn", None)
        if node is None:
            return
        queue, seen, names = deque([node]), set(), set()
        while queue and len(seen) < 128:
            current = queue.popleft()
            if current is None or current in seen:
                continue
            seen.add(current)
            names.add(type(current).__name__)
            if "NpuFusionAttentionBackward" in type(current).__name__:
                break
            queue.extend(next_node for next_node, _index in current.next_functions)
        report["autograd_node_types"] = sorted(names)
        report["fused_attention_backward_in_graph"] = any(
            "NpuFusionAttentionBackward" in name for name in names
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        print("SMOKE_ATTENTION_AUTOGRAD=" + json.dumps(report), flush=True)
        holder["handle"].remove()

    holder["handle"] = candidates[0][1].register_forward_hook(observe)
