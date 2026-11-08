"""Two-process NPU Adam continuation probe on a small Megatron DDP model."""
from argparse import ArgumentParser
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

p = ArgumentParser(description=__doc__)
p.add_argument("phase", choices=["save","resume"])
p.add_argument("output", type=Path)
options = p.parse_args()
world = int(os.environ.get("WORLD_SIZE","1"))
rank = int(os.environ.get("RANK","0"))
local_rank = int(os.environ.get("LOCAL_RANK","0"))
if world not in (1,8):
    raise ValueError("This probe supports one device or the production topology")
tp, pp, ep = (4,2,4) if world == 8 else (1,1,1)
out = options.output.resolve()
if world > 1:
    out = out / f"rank-{rank}"
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.argv = [sys.argv[0], "--transformer-impl", "transformer_engine", "--bf16", "--no-gradient-accumulation-fusion",
            "--moe-grouped-gemm", "--context-parallel-size", "1"]
import mindspeed.megatron_adaptor  # noqa:F401
import numpy as np
import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer.transformer_config import TransformerConfig
from slime.utils import hybrid_restart as restart
from slime.backends.megatron_utils.data import DataIterator
from megatron.core.tensor_parallel import set_defaults_if_not_set_tensor_model_parallel_attributes


def build():
    config = TransformerConfig(num_layers=pp, hidden_size=128, num_attention_heads=8, bf16=True,
                               params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16, context_parallel_size=1,
                               tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, gradient_accumulation_fusion=False,
                               num_moe_experts=4, expert_model_parallel_size=ep, expert_tensor_parallel_size=1,
                               sequence_parallel=world>1)
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layers = torch.nn.Sequential(torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,64))
            if world > 1:
                for p in self.layers[-1].parameters():
                    p.allreduce = False  # exercise the same separate expert optimizer chain as the MoE recipe
            self.register_buffer("probe_buffer", torch.arange(48, dtype=torch.int32))
        def forward(self, x):
            return self.layers(x)
    net = Module().to(device="npu", dtype=torch.bfloat16)
    for param in net.parameters():
        set_defaults_if_not_set_tensor_model_parallel_attributes(param)
    ddp = DistributedDataParallel(config=config, ddp_config=DistributedDataParallelConfig(
        use_distributed_optimizer=True, grad_reduce_in_fp32=True, overlap_grad_reduce=False),module=net)
    optim = get_megatron_optimizer(OptimizerConfig(optimizer="adam", bf16=True, lr=.003, min_lr=.001,
        use_distributed_optimizer=True, optimizer_cpu_offload=True, optimizer_offload_fraction=.35,
        use_precision_aware_optimizer=True, pin_cpu_params=False, pin_cpu_grads=False,
        exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32, main_params_dtype=torch.float32),
        [ddp], use_gloo_process_groups=True)
    spec = importlib.util.spec_from_file_location("preallocate", "/root/hby/setup/preallocate-hybrid-adam.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    count = {"cpu_parameters":0,"npu_parameters":0}
    for sub in getattr(optim,"chained_optimizers",[optim]):
        report = helper.preallocate_hybrid_adam_state(sub.optimizer)
        for k in count:
            count[k] += report[k]
    assert min(count.values()) > 0, count
    sch = OptimizerParamScheduler(optim, .001, .003, .001, 2, 10, "linear", 0., 0., 10, "constant")
    return ddp, optim, sch, count


def step(net, opt, sch, it):
    row = it.get_next(["x","target"])
    x = row["x"][0] + torch.rand((),device="npu",dtype=torch.bfloat16)*.01
    loss = (net(x).float()-row["target"][0]).square().mean()
    loss.backward()
    net.finish_grad_sync()
    successful, norm, _ = opt.step()
    assert successful and norm > 0
    sch.step(1)
    net.zero_grad_buffer()
    opt.zero_grad()
    torch.npu.synchronize()
    return float(loss.detach()), float(norm)


def fingerprint(net, opt, sch):
    import hashlib
    rows, groups, _ = restart.rank_layout([net], opt)
    def hash_tensor(t):
        return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()
    result = {key:hash_tensor(t) for key,t,_ in rows}
    for key, (group, _) in groups.items():
        for k,v in group.items():
            if k != "params":
                result[f"group:{key}:{k}"] = hash_tensor(v) if isinstance(v,torch.Tensor) else repr(v)
    return dict(tensors=result, scheduler=sch.state_dict())


def main():
    torch.npu.set_device(local_rank)
    torch.distributed.init_process_group("hccl", init_method="env://" if world>1 else "file://"+str(out/f"hccl-{os.getpid()}"),
                                         rank=rank, world_size=world, timeout=timedelta(minutes=3))
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep, expert_tensor_parallel_size=1, context_parallel_size=1, order="tp-cp-ep-pp-dp",
            create_gloo_process_groups=True)
        seed = 172 if options.phase == "save" else 8301
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        tensor_parallel.model_parallel_cuda_manual_seed(seed)
        net, opt, sch, counts = build()
        restart.verify_live_layout(out/"layout",rank,[net],opt)
        if options.phase == "save":
            batch = dict(x=[torch.randn(8,128,device="npu",dtype=torch.bfloat16) for _ in range(6)],
                         target=[torch.randn(8,64,device="npu") for _ in range(6)])
            it = DataIterator(batch, [[i] for i in range(6)])
            for _ in range(5):
                step(net,opt,sch,it)
            proof = restart.save_rank(out,rank,[net],opt,sch,[it])
            (out/"proof.json").write_text(json.dumps(dict(pid=os.getpid(),rank=proof,counts=counts)))
        else:
            proof = json.loads((out/"proof.json").read_text())
            assert proof["pid"] != os.getpid()
            rng, offsets, packed = restart.load_rank(out,proof["rank"],[net],opt,sch)
            it = DataIterator(restart.map_tensors(packed,saving=False), [[i] for i in range(6)])
            it.offset, = offsets
            restart.restore_rng(rng)
        assert it.offset == 5
        obs = dict(next_update=step(net,opt,sch,it), state=fingerprint(net,opt,sch), next_offset=it.offset,
                   draws=[random.random(),float(np.random.random()),float(torch.rand(())),
                          float(torch.rand((),device="npu"))])
        if options.phase == "save":
            (out/"uninterrupted.json").write_text(json.dumps(obs))
        else:
            actual = json.loads(json.dumps(obs))
            expected = json.loads((out/"uninterrupted.json").read_text())
            (out/"resumed.json").write_text(json.dumps(actual))
            assert actual == expected, [key for key in expected if actual[key] != expected[key]]
            (out/"verified.json").write_text(json.dumps(dict(passed=True, save_pid=proof["pid"],restore_pid=os.getpid(),
                shard_sha=proof["rank"]["stream_sha256"],cpu_npu=counts,
                rank=rank,world=world,train_tp=tp,train_pp=pp,expert_parallel=ep,
                matched_states=len(actual["state"]["tensors"]),next_update=actual["next_update"],next_cursor=it.offset),indent=2))
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
