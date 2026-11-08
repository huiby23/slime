"""Lossless same-topology restart for the NPU hybrid FP32 Adam recipe."""
import hashlib
import json
import os
from pathlib import Path
import random
import struct
from dataclasses import dataclass

import numpy as np
import torch


CHUNK = 1024 * 1024  # elements; at most 4 MiB per FP32 staging buffer


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_metadata(path, state):
    with path.open("xb") as file:
        torch.save(state, file)
        file.flush()
        os.fsync(file.fileno())
    return digest(path)


def rng_state(*, trainer=False):
    state = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state())
    if trainer:
        from megatron.core.tensor_parallel import get_cuda_rng_tracker
        state.update(npu=torch.npu.get_rng_state(), tracker=get_cuda_rng_tracker().get_states())
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if "npu" in state:
        from megatron.core.tensor_parallel import get_cuda_rng_tracker
        torch.npu.set_rng_state(state["npu"])
        get_cuda_rng_tracker().set_states(state["tracker"])


@dataclass
class SavedTensor:
    device: str
    value: torch.Tensor


def map_tensors(obj, *, saving):
    if saving and isinstance(obj, torch.Tensor):
        value = obj.detach().cpu().contiguous()
        if value.untyped_storage().nbytes() != value.numel() * value.element_size():
            value = value.clone()
        return SavedTensor(obj.device.type, value)
    if not saving and isinstance(obj, SavedTensor):
        return obj.value.to(obj.device)
    if isinstance(obj, dict):
        return {k: map_tensors(v, saving=saving) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(v, saving=saving) for v in obj)
    return obj


class Frames:
    """Independently compressed bounded chunks with a whole-stream digest."""
    def __init__(self, file, saving):
        import zstandard
        self.file = file
        self.sha = hashlib.sha256()
        self.codec = zstandard.ZstdCompressor(level=1) if saving else zstandard.ZstdDecompressor()
        self.saving = saving

    def tensor(self, tensor, master=None):
        # Model residuals use BF16 bytewise XOR. All-zero chunks need only a header.
        if not tensor.is_contiguous():
            if tensor.numel() > CHUNK:
                raise ValueError(f"Unvalidated large noncontiguous tensor {tensor.shape}")
            flat = tensor.detach().contiguous().view(-1)
        else:
            flat = tensor.detach().view(-1)
        for pos in range(0, flat.numel(), CHUNK):
            piece = flat[pos:pos+CHUNK]
            size = piece.numel() * piece.element_size()
            expected = None
            if master is not None:
                expected = master.detach().view(-1)[pos:pos+piece.numel()].cpu().to(tensor.dtype).view(torch.uint8)
            if self.saving:
                raw = piece.cpu().view(torch.uint8)
                if expected is not None:
                    raw = raw ^ expected
                data = b"" if expected is not None and not torch.any(raw) else self.codec.compress(memoryview(raw.numpy()))
                header = struct.pack("<I", len(data))
                self.file.write(header)
                self.file.write(data)
                self.sha.update(header)
                self.sha.update(data)
            else:
                header = self.file.read(4)
                if len(header) != 4:
                    raise ValueError("Truncated Adam frame header")
                length, = struct.unpack("<I", header)
                if length > size + 65536 or (length == 0 and expected is None):
                    raise ValueError("Invalid Adam frame length")
                data = self.file.read(length)
                if len(data) != length:
                    raise ValueError("Truncated Adam frame")
                self.sha.update(header)
                self.sha.update(data)
                if length:
                    decoded = bytearray(self.codec.decompress(data, max_output_size=size))
                    if len(decoded) != size:
                        raise ValueError("Invalid decoded Adam frame size")
                    value = torch.frombuffer(decoded, dtype=torch.uint8)
                    if expected is not None:
                        value = value ^ expected
                else:
                    value = expected
                piece.copy_(value.view(tensor.dtype))
        if not self.saving and not tensor.is_contiguous():
            tensor.copy_(flat.view(tensor.shape))


def rank_layout(model, optimizer):
    """Bind named local model shards to the tensors the two live Adam kernels update."""
    names = {param: f"{i}.{name}" for i, chunk in enumerate(model) for name, param in chunk.named_parameters()}
    arrays = []
    groups = {}
    known = set()
    optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    for oi, optim in enumerate(optimizers):
        hybrid = optim.optimizer
        mapped = optim.model_param_group_index_map
        bindings = {}
        for param, (group_idx, index) in mapped.items():
            orig = hybrid.param_groups[group_idx]["params"][index]
            master = hybrid.param_to_inner_param[orig]
            if param not in names or master.numel() != param.numel() or param in known:
                raise ValueError("Expected a complete, unique local optimizer shard (DP1)")
            bindings[master] = (names[param], orig, param)
            known.add(param)
        def add_group(name, group, members):
            groups[name] = (group, members)
        for gi, group in enumerate(hybrid.param_groups):
            add_group(f"{oi}/outer/{gi}", group, [bindings[hybrid.param_to_inner_param[p]][0] for p in group["params"]])
        handled = set()
        for si, sub in enumerate(hybrid.sub_optimizers):
            kind = type(sub).__module__ + "." + type(sub).__qualname__
            for gi, group in enumerate(sub.param_groups):
                add_group(f"{oi}/{si}/{kind}/{gi}", group, [bindings[p][0] for p in group["params"]])
                for master in group["params"]:
                    name, orig, param = bindings[master]
                    state = sub.state[master]
                    if master in handled or state.get("master_param") is not master:
                        raise ValueError("Hybrid optimizer live master binding mismatch")
                    if set(state) - {"master_param", "exp_avg", "exp_avg_sq", "step"}:
                        raise ValueError(f"Unrecognized Adam state for {name}")
                    handled.add(master)
                    for key, tensor in (("master", master), ("m", state["exp_avg"]), ("v", state["exp_avg_sq"])):
                        if tensor.dtype != torch.float32 or tensor.numel() != param.numel():
                            raise ValueError(f"Expected complete FP32 {key} for {name}")
                        arrays.append((name+"/"+key, tensor, None))
                    if "step" in state:
                        arrays.append((name+"/adam-step", state["step"], None))
                    arrays.append((name+"/model", param, master))
                    if orig.data_ptr() not in (param.data_ptr(), master.data_ptr()):
                        arrays.append((name+"/optimizer-shard", orig, master))
        if handled != set(bindings):
            raise ValueError("Missing live Adam shards")
    arrays.extend((name+"/unchanged", p, None) for p, name in names.items() if p not in known)
    for i, chunk in enumerate(model):
        arrays.extend((f"{i}.{name}/buffer", value, None) for name, value in chunk.named_buffers())
    layout = [(name, tuple(t.shape), str(t.dtype), t.device.type) for name, t, _ in arrays]
    return arrays, groups, layout


def group_values(group):
    return map_tensors({k: v for k,v in group.items() if k != "params"}, saving=True)


def restore_group(group, values):
    if set(group) - {"params"} != set(values):
        raise ValueError("Live Adam param group structure differs")
    for key, value in values.items():
        restored = map_tensors(value, saving=False)
        if isinstance(group[key], torch.Tensor):
            if not isinstance(restored, torch.Tensor) or restored.dtype != group[key].dtype:
                raise ValueError(f"Adam group tensor type differs for {key}")
            group[key].copy_(restored)
        else:
            group[key] = restored


@torch.no_grad()
def save_rank(root, rank, model, optimizer, scheduler, iterators, *, rng=None):
    arrays, groups, layout = rank_layout(model, optimizer)
    path = root / f"rank-{rank}.zst"
    state = dict(layout=layout, groups={n:(members, group_values(g)) for n,(g,members) in groups.items()},
                 scheduler=scheduler.state_dict(), rng=rng if rng is not None else rng_state(trainer=True),
                 offsets=[it.offset for it in iterators],
                 batch=map_tensors(iterators[0].rollout_data, saving=True))
    with path.open("xb") as file:
        frames = Frames(file, True)
        for _, tensor, master in arrays:
            frames.tensor(tensor, master)
        file.flush()
        os.fsync(file.fileno())
    meta = root / f"rank-{rank}.pt"
    return dict(rank=rank, bytes=path.stat().st_size, raw_tensor_bytes=sum(t.numel()*t.element_size() for _,t,_ in arrays),
                stream_sha256=frames.sha.hexdigest(),
                metadata_sha256=write_metadata(meta, state))


@torch.no_grad()
def load_rank(root, expected, model, optimizer, scheduler):
    rank = expected["rank"]
    meta = root / f"rank-{rank}.pt"
    if digest(meta) != expected["metadata_sha256"]:
        raise ValueError("Rank metadata SHA mismatch")
    state = torch.load(meta, map_location="cpu", weights_only=False)
    arrays, groups, layout = rank_layout(model, optimizer)
    if state["layout"] != layout or set(groups) != set(state["groups"]):
        raise ValueError("Live model/optimizer rank layout differs from checkpoint")
    path = root / f"rank-{rank}.zst"
    if path.stat().st_size != expected["bytes"]:
        raise ValueError("Rank stream size mismatch")
    with path.open("rb") as file:
        frames = Frames(file, False)
        for _, tensor, master in arrays:
            frames.tensor(tensor, master)
        if file.read(1) or frames.sha.hexdigest() != expected["stream_sha256"]:
            raise ValueError("Rank stream SHA mismatch")
    # Megatron's load_state_dict advances the schedule relative to num_steps.
    # Initialize its counter at zero so a repeated restore does not add it twice.
    scheduler.num_steps = 0
    scheduler.load_state_dict(state["scheduler"])
    if scheduler.state_dict() != state["scheduler"]:
        raise ValueError("Restored learning-rate scheduler differs from its saved state")
    for name, (group, members) in groups.items():
        saved_members, values = state["groups"][name]
        if members != saved_members:
            raise ValueError("Optimizer group parameter identities differ")
        restore_group(group, values)
    return state["rng"], state["offsets"], state["batch"]


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_source(args, rollout_id, source):
    directory = Path(args.optimizer_step_hf_save_dir).parent / "rollout-source"
    directory.mkdir(exist_ok=True)
    path = directory / f"rollout-{rollout_id}.pt"
    partial = path.with_suffix(".partial")
    checksum = write_metadata(partial, dict(rollout=rollout_id, source=source.state_dict(), rng=rng_state()))
    os.replace(partial, path)
    sync_dir(directory)
    return checksum


def retention_policy(root, default, fingerprint):
    # An operator can apply an agreed retention decision at the next durable
    # boundary without restarting expensive on-policy generation.
    override = root / "retention.json"
    if not override.exists():
        return default
    record = json.loads(override.read_text())
    if record.get("fingerprint") != fingerprint or record.get("retention") not in {"latest", "all"}:
        raise ValueError("Invalid recovery retention policy for this lineage")
    return record["retention"]


def save_bundle(args, model, optimizer, scheduler, data_iterator, rollout_id, step_id, completed, dist, group):
    from slime.backends.megatron_utils.optimizer_step_hf_checkpoint import _rank_zero_call
    root = Path(args.hybrid_restart_store)
    run = Path(args.optimizer_step_hf_save_dir).parent
    destination = root / f"step-{completed:06d}-{run.name}"
    staging = root / f".{destination.name}.partial"
    source = run / "rollout-source" / f"rollout-{rollout_id}.pt"
    def prepare():
        root.mkdir(exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        staging.mkdir()
        os.link(source, staging / "source.pt")
    _rank_zero_call(dist, group, prepare)
    rank = dist.get_rank()
    try:
        report = save_rank(staging, rank, model, optimizer, scheduler, data_iterator)
    except Exception as exc:
        report = {"rank":rank, "error":f"{type(exc).__name__}: {exc}"}
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, report, group=group)
    if any("error" in row for row in reports):
        raise RuntimeError(f"Failed to write hybrid Adam shards: {reports}")
    def publish():
        pointer = root / "LATEST"
        record = dict(format=1, run=str(run), completed=completed, rollout=rollout_id, step=step_id,
                      previous_committed=(pointer.read_text().strip() if pointer.exists() else None),
                      retention=retention_policy(root, args.hybrid_restart_retention, args.hybrid_restart_fingerprint), ranks=reports,
                      fingerprint=args.hybrid_restart_fingerprint, source_sha256=digest(source))
        path = staging / "restart.json"
        with path.open("x") as file:
            json.dump(record, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        sync_dir(staging)
    _rank_zero_call(dist, group, publish)
    return staging, destination


def advance_latest(bundle):
    """Called only after both the HF and all-rank Adam snapshots have been published."""
    import shutil

    root = bundle.parent
    pointer = root / "LATEST"
    previous = pointer.read_text().strip() if pointer.exists() else None
    record = json.loads((bundle / "restart.json").read_text())
    temp = root / ".LATEST.new"
    with temp.open("w") as file:
        file.write(bundle.name+"\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, pointer)
    sync_dir(root)
    run_manifest = Path(record["run"]) / "manifest.json"
    manifest = json.loads(run_manifest.read_text())
    if manifest["status"] == "running":
        manifest.update(optimizer_resume_available=True, latest_restart_bundle=str(bundle),
                        latest_durable_optimizer_steps=record["completed"], hybrid_restart_retention=record["retention"])
        pending = run_manifest.with_suffix(".json.restart-new")
        with pending.open("w") as file:
            json.dump(manifest, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(pending, run_manifest)
        sync_dir(run_manifest.parent)
    upper_step = record["completed"]
    while previous and previous != bundle.name and record["retention"] == "latest":
        if Path(previous).name != previous or not previous.startswith("step-"):
            raise ValueError("Invalid run-owned previous restart pointer")
        old = root / previous
        if not old.exists():
            break  # already pruned after a previous successful latest-only publication
        prior = json.loads((old / "restart.json").read_text())
        if old.is_symlink() or previous != f'step-{prior["completed"]:06d}-{Path(prior["run"]).name}' or prior["fingerprint"] != record["fingerprint"] or prior["completed"] >= upper_step:
            raise ValueError("Refusing to prune unrelated restart state")
        previous, upper_step = prior.get("previous_committed"), prior["completed"]
        shutil.rmtree(old)
        sync_dir(root)


def check_next_checkpoint_space(args, current, hf_bytes):
    """Stop at a durable boundary if the next complete uncompressed write cannot fit."""
    import shutil

    record = json.loads((current / "restart.json").read_text())
    if record["completed"] >= args.num_rollout * args.num_steps_per_rollout:
        return
    raw = sum(report["raw_tensor_bytes"] for report in record["ranks"])
    # zstd frames have a small worst-case expansion, and the next prepared
    # rollout and tokenizer/index metadata require space in addition to tensors.
    needed = raw + raw // 100 + int(hf_bytes) + 1024**3
    free = shutil.disk_usage(current.parent).free
    if free < needed:
        evidence = dict(latest=str(current), completed=record["completed"], free_bytes=free,
                        next_checkpoint_upper_bound_bytes=needed, measured_previous_raw_bytes=raw)
        path = Path(args.optimizer_step_hf_save_dir).parent / "waiting-for-checkpoint-storage.json"
        path.write_text(json.dumps(evidence, indent=2))
        raise OSError(f"Saved step {record['completed']}; free disk is below the next HF and Adam checkpoint uncompressed bound")


def configure(args, manifest, output, *, resume, retention, link_history):
    """Start a new attempt, optionally adopting a durable prefix of the same experiment."""
    fields = ("tensor_model_parallel_size", "pipeline_model_parallel_size", "expert_model_parallel_size",
              "expert_tensor_parallel_size", "context_parallel_size", "global_batch_size", "micro_batch_size",
              "num_rollout", "num_steps_per_rollout", "rollout_batch_size", "n_samples_per_prompt",
              "seed", "rollout_seed", "seq_length", "lr", "optimizer_offload_fraction", "optimizer", "adam_beta1", "adam_beta2",
              "adam_eps", "clip_grad", "rollout_temperature", "rollout_top_p", "rollout_top_k", "advantage_estimator")
    bound = {name:getattr(args, name) for name in fields}
    bound.update(base_path=args.hf_checkpoint, base_index=digest(Path(args.hf_checkpoint) / "model.safetensors.index.json"),
                 train_sha256=digest(manifest["train_data"]), attention=manifest["attention_simulation"])
    args.hybrid_restart_fingerprint = hashlib.sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest()
    if (args.use_critic or getattr(args, "use_routing_replay", False) or getattr(args, "use_rollout_routing_replay", False)
            or args.offload_train or args.rollout_data_transport != "object-store"):
        raise ValueError("Hybrid restart currently supports the synchronous resident actor GRPO recipe")
    args.hybrid_restart_retention = retention
    args.hybrid_restart_store = str(output / "hybrid-restart")
    args.hybrid_resume_from = None
    args.hybrid_resume_in_round = False
    if resume is not None:
        root = Path(resume).resolve(strict=True)
        data = json.loads((root / "restart.json").read_text())
        completed = data["completed"]
        per_round = args.num_steps_per_rollout
        if (data["format"] != 1 or data["fingerprint"] != args.hybrid_restart_fingerprint
                or not 0 < completed < args.num_rollout*per_round
                or completed % args.optimizer_step_hf_save_interval or len(data["ranks"]) != 8
                or [row["rank"] for row in data["ranks"]] != list(range(8))
                or (data["rollout"], data["step"]) != divmod(completed-1, per_round)
                or root.name != f"step-{completed:06d}-{Path(data['run']).name}" or digest(root / "source.pt") != data["source_sha256"]):
            raise ValueError("Restart boundary/configuration mismatch")
        if data["retention"] == "latest" and (root.parent / "LATEST").read_text().strip() != root.name:
            raise ValueError("Selected checkpoint is not the committed latest recovery state")
        parent = Path(data["run"])
        args.start_rollout_id, within = divmod(completed, per_round)
        args.hybrid_resume_in_round = bool(within)
        args.update_weight_start_version = args.start_rollout_id + int(bool(within))
        args.hybrid_resume_from = str(root)
        args.optimizer_step_hf_completed_offset = completed
        if retention == "latest":
            args.hybrid_restart_store = str(root.parent)
        if link_history:
            history = Path(args.optimizer_step_hf_save_dir)
            history.mkdir()
            for step in range(args.optimizer_step_hf_save_interval, completed+1, args.optimizer_step_hf_save_interval):
                name = f"checkpoint_step_{step:06d}"
                previous = parent / "optimizer-step-checkpoints" / name
                proof = json.loads((previous / "optimizer-step-checkpoint.json").read_text())
                if not proof["complete"] or proof["completed_optimizer_steps"] != step:
                    raise ValueError(f"Invalid previous HF boundary {step}")
                (history / name).symlink_to(previous.resolve(), target_is_directory=True)
            for step in range(completed):
                r, s = divmod(step, per_round)
                name = f"grad_norm_actor_{r}_{s}.pt"
                os.link(parent / name, output / name)
            history_weights = output / "weight-updates"
            history_weights.mkdir(exist_ok=True)
            for version in range(1, args.update_weight_start_version+1):
                name = f"weight_v{version:06d}"
                (history_weights / name).symlink_to((parent / "weight-updates" / name).resolve(strict=True), target_is_directory=True)
            if within:
                directory = output / "rollout-source"
                directory.mkdir()
                os.link(root / "source.pt", directory / f"rollout-{data['rollout']}.pt")
        manifest.update(resumed_from=str(root), parent_completed_optimizer_steps=completed,
                        target_new_optimizer_steps=args.num_rollout*per_round-completed, optimizer_state_rebuilt=False,
                        optimizer_state_restored=True, parent_run_manifest=str(parent / "manifest.json"),
                        source_cursor_restarted=False)
    manifest.update(optimizer_checkpoint_requested=True, optimizer_step_hf_checkpoint_format="HF weights plus separate lossless fixed-topology hybrid Adam recovery",
                    hybrid_restart_store=args.hybrid_restart_store, hybrid_restart_retention=retention,
                    hybrid_restart_fingerprint=args.hybrid_restart_fingerprint)
