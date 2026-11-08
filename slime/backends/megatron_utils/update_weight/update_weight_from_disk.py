from __future__ import annotations

import json
import os
import shutil
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
import torch.distributed as dist
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

from ..hf_checkpoint_saver import save_hf_model_to_path


class UpdateWeightFromDisk:
    """Full-weight sync through a shared filesystem and SGLang disk reload."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}
        self.rollout_engines: Sequence[ActorHandle] = []
        self.rollout_engine_lock: ActorHandle | None = None
        # Post-write hook: object-store-backed shared filesystems lack cross-host
        # read-after-write consistency, so written files need an explicit step
        # (e.g. uploading them to the backing object store) before the engines can see them.
        self._post_write_hook: Callable | None = None
        if args.custom_update_weight_post_write_path:
            from slime.utils.misc import load_function

            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

    def disconnect_rollout_engines(self) -> None:
        return

    def pop_metrics(self) -> dict[str, float]:
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    @torch.no_grad()
    def update_weights(self) -> None:
        self.weight_version += 1
        version_dir = Path(self.args.update_weight_disk_dir) / f"weight_v{self.weight_version:06d}"

        group = get_gloo_group()
        completed = getattr(self.args, "_optimizer_step_hf_completed_updates",
                            getattr(self.args, "optimizer_step_hf_completed_offset", 0))
        reuse = (getattr(self.args, "hybrid_restart_store", None) and completed > 0
                 and completed == (self.weight_version - 1) * self.args.num_steps_per_rollout)
        if reuse:
            from ..optimizer_step_hf_checkpoint import _rank_zero_call
            def link_completed_model():
                checkpoint = Path(self.args.optimizer_step_hf_save_dir) / f"checkpoint_step_{completed:06d}"
                proof = json.loads((checkpoint / "optimizer-step-checkpoint.json").read_text())
                if not proof["complete"] or proof["completed_optimizer_steps"] != completed or not proof["optimizer_state_saved"]:
                    raise ValueError("HF model does not match current completed optimizer update")
                stage = version_dir.with_suffix(".partial")
                shutil.copytree(checkpoint, stage, copy_function=os.link,
                                ignore=shutil.ignore_patterns("optimizer-step-checkpoint.json"))
                os.rename(stage, version_dir)
            version_dir.parent.mkdir(parents=True, exist_ok=True)
            _rank_zero_call(dist, group, link_completed_model)
        else:
            if dist.get_rank() == 0:
                shutil.rmtree(version_dir, ignore_errors=True)
            dist.barrier(group=group)
            version_dir.mkdir(parents=True, exist_ok=True)
            save_hf_model_to_path(
                self.args,
                version_dir,
                self.model,
                model_name=self.model_name,
                quantization_config=self.quantization_config,
                progress_desc="Save HF weights for update from disk",
            )
        dist.barrier(group=group)

        # every rank runs the hook (it gates itself): each container must publish
        # its own writes
        if self._post_write_hook is not None:
            self._post_write_hook(self.args, str(version_dir), list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())

        # SGLang reload is orchestrated by RayTrainGroup after the checkpoint
        # is fully written, so training-side lifecycle can decide whether
        # Megatron actors are still alive.
