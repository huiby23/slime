"""Test-only injection for a same-input dual attention pass in pinned SGLang."""
import builtins
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch


def reference_modules(repo):
    paths = {"c8v16_attention":repo/"scripts/npu/c8v16/workspace/experiments/c8v16_attention",
             "sglang_bridge":repo/"scripts/npu/c4v8/workspace/experiments/c4v8_rollout_bridge",
             "logical_kv":repo/"scripts/npu/c4v8/workspace/experiments/c4v8_rollout_bridge"}
    paths.update({name:repo/"scripts/npu/hif4" for name in ("hif4_triton", "hif4_quant", "hif4_decode", "hif4_attention")})
    result = {}
    regular_import = builtins.__import__
    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        if not level and name in paths:
            return get(name)
        return regular_import(name, globals, locals, fromlist, level)
    def get(name):
        if name in result:
            return result[name]
        spec = importlib.util.spec_from_file_location("hby_baseline_"+name, paths[name]/(name+".py"))
        mod = importlib.util.module_from_spec(spec)
        result[name] = mod
        sys.modules[spec.name] = mod
        mod.__dict__["__builtins__"] = vars(builtins) | {"__import__":scoped_import}
        spec.loader.exec_module(mod)
        return mod
    return get("sglang_bridge"), get("hif4_attention")


def make_paired(candidate, base, config, pool_type):
    from dataclasses import asdict
    old_sg, old_core = reference_modules(Path(os.environ["HIF4_TWIN_REFERENCE"]).resolve())
    ref = old_sg.make_backend_class(base, core=old_core.attention,
                                    config=old_core.AttentionConfig(**asdict(config)), pool_type=pool_type)
    class Paired(candidate):
        def __init__(self, runner):
            super().__init__(runner)
            self.reference = ref(runner)
            self.rank = runner.tp_rank
            self.log = Path(os.environ["HBY_C8V16_ROLLOUT_AUDIT_DIR"]).parent / f"paired-tp{self.rank}.jsonl"
        def init_forward_metadata(self,batch):
            super().init_forward_metadata(batch)
            self.reference.init_forward_metadata(batch)
        def _run(self, q, k, v, layer, batch, save_kv_cache=True, **kwargs):
            # Use the same model activations, native cache, slots and query offsets.
            # Both implementations perform their normal write to this layer's identical slots.
            actual = super()._run(q,k,v,layer,batch,save_kv_cache,**kwargs)
            expected = self.reference._run(q,k,v,layer,batch,save_kv_cache,**kwargs)
            same = torch.equal(actual.view(torch.int16),expected.view(torch.int16))
            event = dict(rank=self.rank, mode=batch.forward_mode.name, layer=layer.layer_id, elements=actual.numel(),
                         max_prefix=max(s.prefix_length for s in self.forward_metadata), identical=same)
            if not same:
                event["mismatches"]=int((actual.view(torch.int16)!=expected.view(torch.int16)).sum())
                event["max_error"]=float((actual.float()-expected.float()).abs().max())
            with self.log.open("a") as file:
                file.write(json.dumps(event)+"\n")
            if not same:
                raise AssertionError(f"Reference/candidate attention mismatch: {event}")
            return actual
    return Paired
