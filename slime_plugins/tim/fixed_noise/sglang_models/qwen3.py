"""Apply the same fixed offset to each fresh HF weight synchronization."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.models.qwen3 import Qwen3ForCausalLM

from slime_plugins.tim.fixed_noise.bank import FixedNoiseBank


class FixedNoiseQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if self.pp_group.world_size != 1 or get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Fixed-noise baseline requires TP=PP=1")
        if self.quant_config is not None:
            raise ValueError("Fixed-noise experiment requires unquantized BF16 inference")
        self._fixed_noise_bank = FixedNoiseBank(config.fixed_noise_bank_manifest, config.fixed_noise_scale)
        self._fixed_noise_expected = config.num_hidden_layers * 11 + 2
        if len(self._fixed_noise_bank.entries) != self._fixed_noise_expected:
            raise ValueError("Unexpected Qwen3 logical parameter coverage in noise bank")
        if not config.tie_word_embeddings:
            raise ValueError("This baseline explicitly preserves tied embedding/head weights")
        self._fixed_noise_audit_path = Path(config.fixed_noise_audit_path)
        self._fixed_noise_control_path = Path(
            getattr(
                config, "fixed_noise_control_path", str(self._fixed_noise_audit_path.with_name("noise-control.json"))
            )
        )
        self._fixed_noise_control_signature = None
        self._fixed_noise_clean = {}
        self._fixed_noise_mode = "noisy"
        self._fixed_noise_pending_forward = False
        self._fixed_noise_pending_clean_forward = False
        self._fixed_noise_control_rollout_id = None
        self._fixed_noise_state = dict(
            load_weights_calls=0,
            load_weights_failures=0,
            loaded_tensor_events=0,
            input_untouched_checks=0,
            input_untouched_failures=0,
            noise_application_count=0,
            noise_bank_reuse_count=0,
            noisy_forward_calls=0,
            clean_forward_calls=0,
            clean_mode_transitions=0,
            noisy_mode_transitions=0,
            synchronized_noisy_forward_generations=0,
            latest_load_executed=False,
        )
        self._write_audit("installed")

    def _write_audit(self, stage, **extra):
        bank = self._fixed_noise_bank
        record = dict(
            schema_version=1,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
            worker_pid=os.getpid(),
            model_class=type(self).__name__,
            scale=bank.scale,
            noise_seed=bank.manifest["seed"],
            bank_manifest_sha256=bank.manifest_sha256,
            expected_canonical_tensors=self._fixed_noise_expected,
            logical_tensors_seen=len(bank.records),
            clean_cached_tensors=len(self._fixed_noise_clean),
            fixed_delta_fingerprints_verified=True,
            accumulative_noise=False,
            source="fresh_sync_payload_out_of_place",
            mode=self._fixed_noise_mode,
            control_rollout_id=self._fixed_noise_control_rollout_id,
            stage=stage,
            **self._fixed_noise_state,
            tensors=bank.records,
            **extra
        )
        path = self._fixed_noise_audit_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
        temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temp.replace(path)
        with path.with_suffix(".jsonl").open("a") as stream:
            stream.write(json.dumps({k: v for k, v in record.items() if k != "tensors"}, allow_nan=False) + "\n")

    @torch.no_grad()
    def load_weights(self, weights):
        prepared, probes = [], []
        state = self._fixed_noise_state
        state["load_weights_calls"] += 1
        state["latest_load_executed"] = False
        try:
            for name, fresh in weights:
                canonical = self._fixed_noise_bank.canonical(name)
                if canonical not in self._fixed_noise_bank.entries:
                    # Stock loader may include rotary buffers; they are not
                    # learnable model parameters and must stay unperturbed.
                    prepared.append((name, fresh))
                    continue
                probe_size = min(fresh.numel(), 128)
                probes.append((fresh, fresh.reshape(-1)[:probe_size].detach().clone()))
                self._fixed_noise_clean[canonical] = fresh.detach().clone()
                _, noisy, reused = self._fixed_noise_bank.transform(name, fresh)
                evidence = self._fixed_noise_bank.records[canonical]
                evidence["fresh_loads"] = evidence.get("fresh_loads", 0) + 1
                prepared.append((name, noisy))
                state["loaded_tensor_events"] += 1
                state["noise_application_count"] += 1
                state["noise_bank_reuse_count"] += int(reused)
            result = super().load_weights(prepared)
            for source, before in probes:
                state["input_untouched_checks"] += 1
                if not torch.equal(source.reshape(-1)[: before.numel()], before):
                    state["input_untouched_failures"] += 1
                    raise RuntimeError("A fresh trainer weight payload was modified")
            self._fixed_noise_mode = "noisy"
            self._fixed_noise_pending_forward = True
            self._write_audit("fresh_weights_loaded")
            return result
        except BaseException:
            state["load_weights_failures"] += 1
            self._write_audit("load_failed")
            raise

    def _control_mode(self):
        try:
            stat = self._fixed_noise_control_path.stat()
            # Control files are atomically replaced. Some shared filesystems
            # report equal mtimes for rapid clean/noisy writes, so include the
            # replacement inode instead of relying only on timestamp precision.
            signature = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            return "noisy"
        if signature != self._fixed_noise_control_signature:
            control = json.loads(self._fixed_noise_control_path.read_text())
            if control.get("mode") not in ("clean", "noisy"):
                raise ValueError("Invalid fixed-noise mode request")
            self._fixed_noise_requested_mode = control["mode"]
            self._fixed_noise_control_rollout_id = control.get("rollout_id")
            self._fixed_noise_control_signature = signature
        return self._fixed_noise_requested_mode

    def _switch_mode(self, requested):
        if requested == self._fixed_noise_mode:
            return
        if len(self._fixed_noise_clean) != self._fixed_noise_expected:
            raise RuntimeError("Cannot switch precision with incomplete clean weight coverage")
        if requested == "clean":
            # Restore from independent pristine copies; subtracting noise in
            # BF16 would introduce a destructive rounding roundtrip.
            prepared = list(self._fixed_noise_clean.items())
        else:
            prepared = [
                (name, self._fixed_noise_bank.transform(name, fresh)[1])
                for name, fresh in self._fixed_noise_clean.items()
            ]
        super().load_weights(prepared)
        self._fixed_noise_mode = requested
        self._fixed_noise_state[requested + "_mode_transitions"] += 1
        if requested == "clean":
            self._fixed_noise_pending_clean_forward = True
        self._write_audit("mode_changed")

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        self._switch_mode(self._control_mode())
        result = super().forward(*args, **kwargs)
        mode = self._fixed_noise_mode
        self._fixed_noise_state[mode + "_forward_calls"] += 1
        if mode == "noisy" and self._fixed_noise_pending_forward:
            self._fixed_noise_state["synchronized_noisy_forward_generations"] += 1
            self._fixed_noise_state["latest_load_executed"] = True
            self._fixed_noise_pending_forward = False
            self._write_audit("first_noisy_forward_after_load")
        elif mode == "clean" and self._fixed_noise_pending_clean_forward:
            self._fixed_noise_pending_clean_forward = False
            self._write_audit("first_clean_forward")
        return result


EntryClass = FixedNoiseQwen3ForCausalLM
