"""Frozen NVFP4 gradient replay with merged DP data and paired stochastic RNG.

Derived from the existing BF16 replay; no optimizer is created or stepped.
"""
import argparse
import copy
from datetime import datetime, timezone, timedelta
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

import torch


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def parameters(model):
    return [(str(chunk)+'.'+name,p) for chunk,module in enumerate(model) for name,p in module.named_parameters() if p.requires_grad]


def parameter_hash(params):
    digest=hashlib.sha256()
    for name,param in params:
        digest.update(name.encode())
        digest.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def capture_gradients(params):
    result={}
    for name,param in params:
        grad=getattr(param,'main_grad',None)
        if grad is None:
            grad=param.grad
        assert grad is not None,name
        assert torch.isfinite(grad).all(),name
        result[name]=grad.detach().float().cpu().clone()
    return result


def gradient_metrics(current, reference=None):
    square=reference_square=dot=difference_square=0.
    per_layer={}
    for name,grad in current.items():
        flat=grad.reshape(-1)
        ref=reference[name].reshape(-1) if reference is not None else None
        layer_square=0.
        for offset in range(0,flat.numel(),1048576):
            x=flat[offset:offset+1048576].double()
            layer_square+=torch.dot(x,x).item()
            if ref is not None:
                y=ref[offset:offset+1048576].double()
                reference_square+=torch.dot(y,y).item()
                dot+=torch.dot(x,y).item()
                delta=x-y
                difference_square+=torch.dot(delta,delta).item()
        square+=layer_square
        per_layer[name]=math.sqrt(layer_square)
    result={'norm':math.sqrt(square),'parameter_gradient_norms':per_layer}
    if reference is not None:
        result.update({'reference_norm':math.sqrt(reference_square),
            'cosine_similarity':dot/math.sqrt(square*reference_square) if square and reference_square else None,
            'relative_l2_difference':math.sqrt(difference_square/reference_square) if reference_square else None,
            'norm_ratio':math.sqrt(square/reference_square) if reference_square else None})
    return result


def prepare_data(job):
    payload=torch.load(job['batch_path'],map_location='cpu',weights_only=False)
    assert len(payload['samples'])==64
    # Reassemble all four training partitions into the same global batch for a
    # single-GPU diagnostic. Saved rewards, masks, advantages and old probs stay fixed.
    ordered=sorted(payload['samples'],key=lambda row:row['rollout_position'])
    layout={'partition':list(range(64)), 'num_microbatches':[64], 'global_batch_sizes':[64],
            'sample_indices':[row['sample_index'] for row in ordered],
            'micro_batch_indices':[[i] for i in range(64)]}
    by_position={sample['rollout_position']:sample for sample in payload['samples']}
    assert set(by_position)==set(range(64))
    keys=set(by_position[0])-{'data_parallel_rank','sample_index','rollout_position'}
    data={key:[by_position[position][key] for position in layout['partition']] for key in keys}
    data['partition']=layout['partition']
    data['sample_indices']=layout['sample_indices']
    data['micro_batch_indices']=layout['micro_batch_indices']
    data['num_microbatches']=layout['num_microbatches']
    data['global_batch_sizes']=layout['global_batch_sizes']
    assert len(layout['partition'])==64 and len(layout['micro_batch_indices'])==64
    for row in payload['samples']:
        assert len(row['tokens'])==row['total_lengths']
        assert len(row['log_probs'])==len(row['rollout_log_probs'])==len(row['loss_masks'])==row['response_lengths']
    return data,by_position


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job',type=Path)
    parser.add_argument('--check-only',action='store_true')
    cli=parser.parse_args()
    job=json.loads(cli.job.read_text());output=Path(job['container_run_dir'])
    output.mkdir(parents=True,exist_ok=True)
    if not cli.check_only:
        # CPU preflight has already validated args. Acquire this process's CUDA context promptly,
        # so framework imports are visibly part of the experiment's GPU use.
        expected=job.get('expected_gpu_uuid')
        assert expected, 'A GPU UUID must be assigned by the idle-only launcher.'
        occupancy=subprocess.check_output(['nvidia-smi','--id='+expected,
            '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
        allowed=set(job.get('reservation_host_pids',[]))
        if {int(pid) for pid in occupancy.splitlines() if pid.strip()}-allowed:
            raise SystemExit(75)
        memory,util=[int(v.strip()) for v in subprocess.check_output(['nvidia-smi','--id='+expected,
            '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15).split(',')]
        if memory>(2048 if allowed else 64) or util!=0:
            raise SystemExit(75)
        assert torch.cuda.device_count()==1
        actual=str(torch.cuda.get_device_properties(0).uuid)
        assert actual.replace('GPU-','')==expected.replace('GPU-','')
        torch.cuda.set_device(0)
        torch.cuda.init()
        save_json(output/'status.json',{'state':'preparing_framework','job_id':job['job_id'],
            'started_at_utc':datetime.now(timezone.utc).isoformat(),'optimizer_steps':0,'gpu_uuid':actual})
    sys.argv=job['megatron_argv']
    from slime.utils.arguments import parse_args
    args=parse_args()
    assert args.bf16 and not args.fp8 and args.debug_train_only
    assert not args.use_tis and not args.use_rs and not args.use_score_centering
    assert not args.overlap_grad_reduce and not args.overlap_param_gather
    assert args.num_rollout==0 and args.global_batch_size==64 and args.micro_batch_size==1
    assert args.kl_coef==args.kl_loss_coef==args.entropy_coef==0
    data,by_position=prepare_data(job)
    counts={subset:{'samples':0,'nonzero_advantage_samples':0} for subset in ['all','short','long']}
    for row in by_position.values():
        active=bool(torch.as_tensor(row['advantages']).count_nonzero())
        for subset in ['all','short' if row['response_lengths']<=256 else 'long']:
            counts[subset]['samples']+=1
            counts[subset]['nonzero_advantage_samples']+=int(active)
    if cli.check_only:
        save_json(output/'configuration-check.json',{'passed':True,'counts':counts,'checkpoint':args.load,
            'rollout_id':job['rollout_id'],'bf16':args.bf16,'optimizer_will_be_created':False})
        print('GRADIENT_REPLAY_CONFIGURATION_CHECK_PASSED',flush=True)
        return
    status={'state':'initializing','started_at_utc':datetime.now(timezone.utc).isoformat(),
        'job_id':job['job_id'],'rollout_id':job['rollout_id'],'optimizer_steps':0,
        'sample_counts':counts,'gpu_uuid':actual,'cases':{}}
    save_json(output/'status.json',status)
    import torch.distributed as dist
    from megatron.core import mpu
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.utils import get_model_config
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model import initialize_model_and_optimizer
    from slime.backends.megatron_utils.data import DataIterator,get_batch
    from slime.backends.megatron_utils.loss import loss_function,enable_log_prob_capture,drain_captured_log_probs
    from slime.utils.distributed_utils import init_gloo_group
    from slime.utils.eval_config import EvalDatasetConfig
    torch.serialization.add_safe_globals([EvalDatasetConfig])
    torch.cuda.set_device(0)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    os.environ.update({'MASTER_ADDR':'127.0.0.1','MASTER_PORT':str(port),'WORLD_SIZE':'1','RANK':'0','LOCAL_RANK':'0'})
    try:
        dist.init_process_group('nccl',rank=0,world_size=1,timeout=timedelta(minutes=5))
        init_gloo_group();args.rank=0;args.world_size=1;args.num_gpus_per_node=1
        init(args)
        model,optimizer,scheduler,iteration=initialize_model_and_optimizer(args,'actor')
        assert optimizer is None and scheduler is None
        for module in model:
            module.train()
        config=get_model_config(model[0]);config.finalize_model_grads_func=finalize_model_grads
        config.grad_scale_func=None;config.timers=None
        params=parameters(model);fingerprint=parameter_hash(params)
        for key,values in list(data.items()):
            if isinstance(values,list) and values and isinstance(values[0],torch.Tensor):
                data[key]=[value.to('cuda') for value in values]
        originals=[torch.as_tensor(value,device='cuda',dtype=torch.float32).clone() for value in data['advantages']]
        expected_logprobs={position:torch.as_tensor(row['log_probs'],device='cpu') for position,row in by_position.items()}
        keys=['tokens','multimodal_train_inputs','packed_seq_params','total_lengths','response_lengths',
            'loss_masks','log_probs','ref_log_probs','values','advantages','returns','rollout_log_probs',
            'rollout_topk_token_ids','rollout_topk_log_probs','rollout_top_p_log_probs','teacher_log_probs',
            'rollout_mask_sums','partition']
        references={}
        cases=[('all','baseline',1042),('all','repeat_same_rng',1042),
               ('all','repeat_other_rng',1043),('all','tis',1042)]
        for subset,method,replay_seed in cases:
            torch.manual_seed(replay_seed)
            torch.cuda.manual_seed_all(replay_seed)
            status['state']='computing_gradients';status['current_case']=subset+'_'+method
            save_json(output/'status.json',status)
            args.use_tis=method=='tis'
            args.use_rs=False;args.tis_mode='truncate';args.tis_upper_bound=2.;args.tis_batch_normalize=False
            for module in model:
                module.zero_grad(set_to_none=True);module.zero_grad_buffer()
            data['advantages']=[adv if subset=='all' or (length<=256)==(subset=='short') else torch.zeros_like(adv)
                for adv,length in zip(originals,data['response_lengths'],strict=True)]
            # Keep original masks and global-batch normalization when isolating a length group's contribution.
            iterator=[DataIterator(data,data['micro_batch_indices'])]
            completed_microbatches=0
            def forward_step(iterator,model,return_schedule_plan=False):
                nonlocal completed_microbatches
                assert not return_schedule_plan
                batch=get_batch(iterator,keys,args.data_pad_size_multiplier,args.allgather_cp)
                logits=model(input_ids=batch['tokens'],position_ids=None,attention_mask=None,labels=None,
                    packed_seq_params=batch['packed_seq_params'],loss_mask=batch['full_loss_masks'])
                completed_microbatches+=1
                if completed_microbatches%16==0:
                    status['completed_microbatches']=completed_microbatches
                    save_json(output/'status.json',status)
                    print('FORWARD_PROGRESS',subset,method,completed_microbatches,'/64',flush=True)
                return logits,partial(loss_function,args,batch,64,64)
            enable_log_prob_capture()
            get_forward_backward_func()(forward_step_func=forward_step,data_iterator=iterator,model=model,
                num_microbatches=64,seq_length=args.seq_length,micro_batch_size=1,
                decoder_seq_length=args.decoder_seq_length,forward_only=False)
            torch.cuda.synchronize()
            captured=drain_captured_log_probs();assert set(captured)==set(range(64))
            max_error=max((captured[p].cpu()-expected_logprobs[p]).abs().max().item() for p in captured)
            gradients=capture_gradients(params)
            comparison=gradient_metrics(gradients,references.get(subset))
            comparison['max_recomputed_logprob_error']=max_error
            comparison['peak_device_memory_bytes']=torch.cuda.max_memory_allocated()
            assert max_error<.01, 'Replayed model probabilities differ from the saved trainer snapshot.'
            if subset=='all' and method=='baseline':
                comparison['original_run_grad_norm']=job['expected_grad_norm']
                comparison['grad_norm_relative_error']=abs(comparison['norm']/job['expected_grad_norm']-1)
                # DP=4 to DP=1 changes the SR streams. The paired-RNG repeat is
                # the reproducibility control; source norm is reported, not asserted.
            status['cases'][subset+'_'+method]=comparison
            save_json(output/'status.json',status)
            print('GRADIENT_CASE',subset,method,json.dumps({key:value for key,value in comparison.items() if key!='parameter_gradient_norms'}),flush=True)
            if method=='baseline':
                references[subset]=gradients
            else:
                del gradients
        assert parameter_hash(params)==fingerprint, 'Weights changed during a gradient-only replay.'
        status.update({'state':'completed','passed':True,'weights_unchanged':True,
            'parameter_sha256':fingerprint,'parameter_count':sum(p.numel() for _,p in params),
            'finished_at_utc':datetime.now(timezone.utc).isoformat()})
        save_json(output/'status.json',status)
        save_json(output/'gradient-comparison.json',status)
    except BaseException as exc:
        import traceback
        status.update({'state':'failed','passed':False,'error':str(exc),'traceback':traceback.format_exc()})
        save_json(output/'status.json',status)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for path in output.glob('*.json'):
            os.chown(path,1042,1043)
        os.chown(output,1042,1043)


if __name__=='__main__':
    main()
