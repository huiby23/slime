"""Preserve Qwen3's loader/update paths while simulating NVFP4 linear GEMMs."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod, UnquantizedLinearMethod
from sglang.srt.models.qwen3 import Qwen3ForCausalLM

from slime_plugins.nvfp4_sim.recipe import NVFP4Recipe, gemm, quantize


class NVFP4SimLinearMethod(UnquantizedLinearMethod):
    def __init__(self, recipe, cache_weights=False, quantize_weights=True, quantize_activations=True):
        self.recipe = recipe
        self.cache_weights = cache_weights
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.invalidate_weight_cache()

    def invalidate_weight_cache(self):
        self._weight_key = self._weight_cache = None

    def apply(self, layer, x, bias=None):
        if x.dtype != torch.bfloat16 or layer.weight.dtype != torch.bfloat16:
            raise ValueError("NVFP4 rollout simulation requires BF16 weights and inputs")
        # Activations are quantized every call. Optional cached weight casts are
        # invalidated after stock loading and both baseline update APIs.
        with torch.no_grad():
            matrix = x.reshape(-1, x.shape[-1])
            qx = (
                quantize(matrix, reference_arithmetic=self.recipe.reference_arithmetic)
                if self.quantize_activations
                else None
            )
            try:
                version = layer.weight._version
            except RuntimeError:  # Inference tensors intentionally have no version counter.
                version = None
            key = (
                id(layer.weight),
                layer.weight.data_ptr(),
                version,
                layer.weight.device,
                layer.weight.dtype,
                tuple(layer.weight.shape),
            )
            if not self.quantize_weights:
                qw = None
            elif not self.cache_weights or key != self._weight_key:
                qw = quantize(layer.weight, weight=True, reference_arithmetic=self.recipe.reference_arithmetic)
                if self.cache_weights:
                    self._weight_key, self._weight_cache = key, qw
            else:
                qw = self._weight_cache
            if qx is not None and qw is not None:
                output = gemm(qx, qw, chunk_blocks=self.recipe.chunk_blocks)
            else:
                # Diagnostic W4A16 / W16A4 arms. Never used by the default
                # W4A4 path; dequantized operands remain FP32 before GEMM.
                left = qx.dequantize() if qx is not None else matrix.float()
                right = qw.dequantize() if qw is not None else layer.weight.float()
                output = left @ right.T
            if bias is not None:
                output.add_(bias.float())
            self.recipe.record("fprop")
            return output.to(x.dtype).reshape(*x.shape[:-1], layer.weight.shape[0])


class NVFP4SimEmbeddingLinearMethod(UnquantizedEmbeddingMethod):
    """Use BF16 embedding lookup and NVFP4 output projection on the same weight."""

    def __init__(self, recipe, cache_weights=False, quantize_weights=True, quantize_activations=True):
        NVFP4SimLinearMethod.__init__(self, recipe, cache_weights, quantize_weights, quantize_activations)

    apply = NVFP4SimLinearMethod.apply
    invalidate_weight_cache = NVFP4SimLinearMethod.invalidate_weight_cache


class NVFP4SimQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if self.pp_group.world_size != 1 or get_tensor_model_parallel_world_size() != 1:
            raise ValueError("The baseline NVFP4 rollout simulation requires TP=PP=1")
        if getattr(self, "quant_config", None) is not None:
            raise ValueError("NVFP4 simulation loads original BF16 weights; disable native quantization")
        recipe = NVFP4Recipe(reference_arithmetic=getattr(config, "nvfp4_sim_reference_arithmetic", True))
        names = []
        selected = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head")
        include_head = getattr(config, "nvfp4_sim_include_output_layer", True)
        # The tied lm_head aliases model.embed_tokens. Include that alias while
        # preserving the embedding() implementation on their shared module.
        for name, module in self.named_modules(remove_duplicate=False):
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in selected or (leaf == "lm_head" and not include_head):
                continue
            expected_method = (
                (UnquantizedLinearMethod, UnquantizedEmbeddingMethod) if leaf == "lm_head" else UnquantizedLinearMethod
            )
            if not isinstance(getattr(module, "quant_method", None), expected_method):
                raise TypeError(f"Expected an unquantized linear method at {name}")
            method_cls = (
                NVFP4SimEmbeddingLinearMethod
                if isinstance(module.quant_method, UnquantizedEmbeddingMethod)
                else NVFP4SimLinearMethod
            )
            module.quant_method = method_cls(
                recipe,
                cache_weights=getattr(config, "nvfp4_sim_weight_cache", False),
                quantize_weights=getattr(config, "nvfp4_sim_quantize_weights", True),
                quantize_activations=getattr(config, "nvfp4_sim_quantize_activations", True),
            )
            names.append(name)
        expected = config.num_hidden_layers * 4 + int(include_head)
        if len(names) != expected:
            raise RuntimeError(f"NVFP4 rollout coverage mismatch: expected {expected}, got {names}")
        self._nvfp4_recipe = recipe
        self._nvfp4_modules = tuple(names)
        # CPU-only bookkeeping. No tensors are copied, reduced or synchronized
        # for this observer, and it never samples or changes the recipe.
        audit_path = getattr(config, "nvfp4_sim_audit_path", None)
        # Replica workers must not overwrite one another's final evidence.
        # The configured path names an audit family; a host-side monitor may
        # aggregate these files into that path after checking every worker.
        audit_base = Path(audit_path) if audit_path else None
        self._nvfp4_audit_path = (
            audit_base.parent / (audit_base.stem + ".workers") / f"worker-{os.getpid()}.json"
            if audit_base is not None
            else None
        )
        self._nvfp4_audit_base_path = str(audit_base) if audit_base is not None else None
        # Reading an existing parameter's device is metadata-only: no CUDA
        # operation, allocation, reduction, synchronization or RNG access.
        device = next(self.parameters()).device
        self._nvfp4_worker_device = dict(
            cuda_device=str(device),
            cuda_logical_device=device.index,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            nvidia_visible_devices=os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        )
        self._nvfp4_audit_expected = expected
        self._nvfp4_audit_pending_forward = True
        self._nvfp4_audit_fprop_at_load = 0
        self._nvfp4_audit_state = dict(
            load_weights_calls=0,
            load_weights_failures=0,
            weight_generation=0,
            loads_after_quantized_forward=0,
            cache_invalidations_verified=0,
            latest_invalidated_linears=0,
            cache_rebuilds_verified=0,
            latest_rebuilt_linears=0,
            last_rebuilt_weight_generation=-1,
        )
        self._write_quantization_audit("installed")
        logging.getLogger(__name__).warning("NVFP4 rollout simulation installed on %d linears", len(names))

    def _quantization_audit_methods(self):
        return [self.get_submodule(name).quant_method for name in self._nvfp4_modules]

    def _write_quantization_audit(self, stage, **extra):
        path = self._nvfp4_audit_path
        if path is None:
            return
        try:
            methods = self._quantization_audit_methods()
            fprop_calls = int(self._nvfp4_recipe.counters.get("fprop", 0))
            record = dict(
                schema_version=2,
                updated_at_utc=datetime.now(timezone.utc).isoformat(),
                stage=stage,
                worker_pid=os.getpid(),
                worker_device=self._nvfp4_worker_device,
                audit_base_path=self._nvfp4_audit_base_path,
                audit_path=str(path),
                model_class=type(self).__name__,
                precision="nvfp4_w4a4_simulation",
                reference_arithmetic=bool(self._nvfp4_recipe.reference_arithmetic),
                quantize_weights=all(method.quantize_weights for method in methods),
                quantize_activations=all(method.quantize_activations for method in methods),
                weight_cache_enabled=all(method.cache_weights for method in methods),
                installed_linears=len(methods),
                expected_linears=self._nvfp4_audit_expected,
                modules=list(self._nvfp4_modules),
                fprop_calls=fprop_calls,
                fprop_calls_since_latest_load=fprop_calls - self._nvfp4_audit_fprop_at_load,
                pending_forward_after_load=self._nvfp4_audit_pending_forward,
                cached_linears=sum(method._weight_cache is not None for method in methods),
                activation_block_shape=[1, 16],
                weight_block_shape=[16, 16],
                outer_scale="per_matrix_tensor_amax",
                payload="E2M1",
                block_scale="E4M3FN",
                hardware_emulation=False,
                **self._nvfp4_audit_state,
                **extra,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(path)
            # The concise history keeps load/rebuild evidence without repeating
            # all 113 module names on each flattened weight bucket.
            history = {key: value for key, value in record.items() if key != "modules"}
            with path.with_suffix(".jsonl").open("a") as stream:
                stream.write(json.dumps(history) + "\n")
        except Exception:
            # This observer must not change the model's execution or turn an
            # otherwise successful load into an exception. Postflight requires
            # the audit artifact and will reject missing/incomplete evidence.
            logging.getLogger(__name__).exception("Cannot publish NVFP4 rollout quantization audit")

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        if self._nvfp4_audit_path is not None and self._nvfp4_audit_pending_forward:
            methods = self._quantization_audit_methods()
            rebuilt = sum(method._weight_cache is not None for method in methods)
            state = self._nvfp4_audit_state
            state["latest_rebuilt_linears"] = rebuilt
            if rebuilt == self._nvfp4_audit_expected:
                state["cache_rebuilds_verified"] += 1
                state["last_rebuilt_weight_generation"] = state["weight_generation"]
                self._nvfp4_audit_pending_forward = False
            self._write_quantization_audit("first_forward_after_load")
        return result

    def load_weights(self, *args, **kwargs):
        # Both the baseline's regular and flattened-bucket updates call this
        # method. Their loaders may write through .data without a version bump.
        succeeded = False
        try:
            result = super().load_weights(*args, **kwargs)
            succeeded = True
            return result
        finally:
            for module in self.modules():
                method = getattr(module, "quant_method", None)
                if isinstance(method, (NVFP4SimLinearMethod, NVFP4SimEmbeddingLinearMethod)):
                    method.invalidate_weight_cache()
            if getattr(self, "_nvfp4_audit_path", None) is not None:
                state = self._nvfp4_audit_state
                state["load_weights_calls"] += 1
                if not succeeded:
                    state["load_weights_failures"] += 1
                else:
                    state["weight_generation"] += 1
                fprop_calls = int(self._nvfp4_recipe.counters.get("fprop", 0))
                if fprop_calls > 0:
                    state["loads_after_quantized_forward"] += 1
                methods = self._quantization_audit_methods()
                invalidated = sum(method._weight_cache is None and method._weight_key is None for method in methods)
                state["latest_invalidated_linears"] = invalidated
                if succeeded and invalidated == self._nvfp4_audit_expected:
                    state["cache_invalidations_verified"] += 1
                state["latest_rebuilt_linears"] = 0
                self._nvfp4_audit_fprop_at_load = fprop_calls
                self._nvfp4_audit_pending_forward = True
                self._write_quantization_audit("load_weights", load_succeeded=succeeded)


EntryClass = NVFP4SimQwen3ForCausalLM
