"""Small passive audit of native optimizer updates and quantized GEMM coverage."""

import json
import math
import os
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist


def _masters(optimizer):
    samples=[]
    for part in getattr(optimizer,'chained_optimizers',[optimizer]):
        inner=part.optimizer
        if inner.offload_fraction!=1.0 or inner.gpu_optimizer is not None:
            raise RuntimeError('Expected fully CPU-offloaded native Adam')
        for cpu_optimizer in inner.cpu_optimizers:
            for group in cpu_optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.device.type!='cpu' or parameter.dtype!=torch.float32:
                        raise RuntimeError('CPU Adam must update FP32 master parameters')
                    samples.append(parameter.detach().reshape(-1)[::max(1,parameter.numel()//65536)][:65536].clone())
                    if len(samples)==4:
                        return torch.cat(samples)
    if not samples:
        raise RuntimeError('No FP32 CPU master parameters were found')
    return torch.cat(samples)


def before_step(args,rollout_id,step_id,model,optimizer,opt_param_scheduler):
    optimizer._w4a8_context=(rollout_id,step_id)
    if hasattr(optimizer,'_w4a8_audited'):
        return
    mode=os.environ['W4A8_TRAINING_MODE']
    recipes=[]
    modules=[]
    for chunk in model:
        core=chunk
        while hasattr(core,'module'):
            core=core.module
        if hasattr(core,'_w4a8_recipe'):
            recipes.append(core._w4a8_recipe)
            modules.extend(core._w4a8_modules)
    if mode=='w4a8' and len(modules)!=4*args.num_layers:
        raise RuntimeError(f'Incomplete W4A8 training coverage: {len(modules)}')
    if mode=='bf16' and modules:
        raise RuntimeError('BF16 control unexpectedly contains quantized linears')
    original=optimizer.step

    def step(self,*positional,**keywords):
        before=_masters(self)
        result=original(*positional,**keywords)
        after=_masters(self)
        if not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise RuntimeError('Nonfinite sampled master parameters')
        success,grad_norm,_=result
        if grad_norm is not None and not math.isfinite(float(grad_norm)):
            raise RuntimeError('Nonfinite global gradient norm')
        counters={}
        for recipe in recipes:
            for key,value in recipe.counters.items():
                counters[key]=counters.get(key,0)+value
        if mode=='w4a8' and not all(counters.get(k,0)>0 for k in ['fprop','dgrad','wgrad']):
            raise RuntimeError('Missing quantized forward/backward GEMMs')
        record=dict(rollout_id=self._w4a8_context[0],step_id=self._w4a8_context[1],
            rank=dist.get_rank(),training_mode=mode,optimizer_success=bool(success),
            grad_norm=None if grad_norm is None else float(grad_norm),
            sampled_master_elements=before.numel(),sampled_master_max_update=float((after-before).abs().max()),
            sampled_masters_finite=True,cpu_optimizer_offload_fraction=1.0,
            quantized_modules=len(modules),counters=counters,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        directory=Path(os.environ['W4A8_TRAINING_AUDIT_DIR'])
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/f'optimizer-rank-{dist.get_rank()}.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        return result
    optimizer.step=MethodType(step,optimizer)
    optimizer._w4a8_audited=True
