"""The two arms must differ only in precision hooks and artifact destinations."""

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def module(name):
    path=Path(__file__).resolve().parents[1]/'examples/w4a8_sim'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def model_metadata(tmp_path):
    model=tmp_path/'model'
    model.mkdir()
    config=dict(model_type='qwen3',num_hidden_layers=36,hidden_size=2560,
        tie_word_embeddings=True,intermediate_size=9728,num_attention_heads=32,
        num_key_value_heads=8,head_dim=128,rms_norm_eps=1e-6,rope_theta=1000000,vocab_size=151936)
    (model/'config.json').write_text(json.dumps(config))
    (model/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':8044936192}}))
    return model


def test_matched_dense_arms_and_resume_preserve_model_and_data(tmp_path):
    model=model_metadata(tmp_path)
    plan=module('prepare_qwen3_4b').prepare(model,tmp_path/'work',tmp_path/'checkpoints',[0,1,2,3])
    def normalized(command):
        result=list(command)
        for flag in ['--save','--tensorboard-dir','--custom-model-provider-path','--sglang-json-model-override-args']:
            if flag in result:
                index=result.index(flag)
                del result[index:index+2]
        return result
    assert normalized(plan['arms']['bf16']['training_command'])==normalized(plan['arms']['w4a8']['training_command'])
    run=module('run_dense_experiment')
    resumed=run.stage_arm(plan,'w4a8',tmp_path/'smoke/w4a8',tmp_path/'smoke/checkpoints/w4a8',True,True)
    command=resumed['training_command']
    assert '--start-rollout-id' not in command
    assert '--override-opt-param-scheduler' in command
    assert command[command.index('--load')+1]==str(tmp_path/'smoke/checkpoints/w4a8/last')
    assert command[command.index('--hf-checkpoint')+1]==str(model)
    assert '--untie-embeddings-and-output-weights' not in command
    assert resumed['environment']['W4A8_BEST_WEIGHTS_ONLY']=='1'
    assert resumed['environment']['TENSORBOARD_DIR']==str(tmp_path/'smoke/w4a8/tensorboard')
    assert json.loads(resumed['environment']['W4A8_CHECKPOINT_ROOTS'])==[str(tmp_path/'smoke/checkpoints/w4a8/.snapshots')]
    option=lambda name:command[command.index(name)+1]
    assert int(option('--rollout-batch-size'))*int(option('--n-samples-per-prompt'))==int(option('--global-batch-size'))==128
    assert int(option('--rollout-max-response-len'))==1024


@pytest.mark.parametrize('available,train_count',[(3,2),(7,4),(8,8)])
def test_use_every_gpu_for_rollout_without_changing_the_training_batch(tmp_path,available,train_count):
    model=model_metadata(tmp_path)
    plan=module('prepare_qwen3_4b').prepare(model,tmp_path/'work',tmp_path/'checkpoints',list(range(available)))
    for mode in ['bf16','w4a8']:
        arm=plan['arms'][mode]
        command=arm['training_command']
        option=lambda name:command[command.index(name)+1]
        assert int(option('--actor-num-gpus-per-node'))==train_count
        assert int(option('--rollout-num-gpus'))==available
        assert int(option('--global-batch-size'))==64
        assert int(option('--rollout-batch-size'))*int(option('--n-samples-per-prompt'))==64
        assert arm['environment']['TENSORBOARD_DIR']==option('--tensorboard-dir')


def test_idle_selection_rechecks_compute_owners_and_rejects_stale_observations():
    run=module('run_dense_experiment')
    now=datetime.now(timezone.utc)
    status={'at':now.isoformat(),'gpus':[
        {'uuid':'GPU-free','idle':True,'qualified':True},
        {'uuid':'GPU-raced','idle':True,'qualified':True},
        {'uuid':'GPU-reuse','idle':True,'qualified':False},
        {'uuid':'GPU-active','idle':False,'qualified':False}]}
    processes='GPU-raced, 123\nGPU-active, 456\n'
    selected=run.eligible_gpus(status,processes,{'GPU-reuse'})
    assert [g['uuid'] for g in selected]==['GPU-free','GPU-reuse']
    status['at']=(now-timedelta(seconds=30)).isoformat()
    assert not run.eligible_gpus(status,processes,{'GPU-reuse'})


def test_zero_gradient_smoke_never_satisfies_the_formal_training_gate(tmp_path):
    run=module('run_dense_experiment')
    work,save=tmp_path/'work',tmp_path/'save'
    last,best=save/'step_0000000',save/'best_step_0000000'
    (last/'iter_0000000').mkdir(parents=True)
    (last/'iter_0000000/.metadata').touch()
    best.mkdir()
    (best/'export-validation.json').write_text(json.dumps({'passed':True,'optimizer_exported':False}))
    (save/'last').symlink_to(last)
    (save/'best').symlink_to(best)
    (save/'checkpoint-policy.json').write_text(json.dumps({'last':{'completed_updates':1}}))
    audit=work/'training-audit'
    audit.mkdir(parents=True)
    record={'rollout_id':0,'optimizer_success':True,'sampled_masters_finite':True,
        'grad_norm':0.0,'sampled_master_max_update':5.96e-8}
    (audit/'optimizer-rank-0.jsonl').write_text(json.dumps(record)+'\n')
    arm={'work':str(work),'save':str(save)}
    first=run.validate_run(arm,1,'bf16',1,require_nonzero=False)
    assert first['passed'] and not first['nonzero_gradient_seen']
    with pytest.raises(RuntimeError,match='No nonzero trained update'):
        run.validate_run(arm,1,'bf16',1)


def test_extended_smoke_keeps_native_adam_state_and_constant_learning_rate():
    import copy
    import torch
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    def scheduler(optimizer,total,override):
        return OptimizerParamScheduler(optimizer,init_lr=0,max_lr=1e-6,min_lr=0,
            lr_warmup_steps=0,lr_decay_steps=total,lr_decay_style='constant',
            start_wd=.1,end_wd=.1,wd_incr_steps=total,wd_incr_style='constant',
            use_checkpoint_opt_param_scheduler=False,override_opt_param_scheduler=override)
    parameter=torch.nn.Parameter(torch.tensor([1.0]))
    optimizer=torch.optim.Adam([parameter],lr=1e-6)
    first=scheduler(optimizer,32,False)
    parameter.grad=torch.tensor([.2])
    optimizer.step()
    first.step(32)
    saved=copy.deepcopy(optimizer.state_dict())
    resumed_parameter=torch.nn.Parameter(parameter.detach().clone())
    resumed_optimizer=torch.optim.Adam([resumed_parameter],lr=1e-6)
    resumed_optimizer.load_state_dict(saved)
    resumed=scheduler(resumed_optimizer,256,True)
    resumed.load_state_dict(first.state_dict())
    assert resumed.num_steps==32 and resumed.lr_decay_steps==256
    for name in ['exp_avg','exp_avg_sq','step']:
        assert torch.equal(resumed_optimizer.state[resumed_parameter][name],optimizer.state[parameter][name])
    resumed_parameter.grad=torch.tensor([.3])
    resumed_optimizer.step()
    resumed.step(128)
    assert resumed.num_steps==160
    assert resumed_optimizer.param_groups[0]['lr']==1e-6


def test_retire_owned_smoke_keeps_best_and_can_be_retried(tmp_path,monkeypatch):
    import shutil
    run=module('run_dense_experiment')
    work,save=tmp_path/'work',tmp_path/'save'
    work.mkdir()
    physical=save/'.snapshots'
    last,best=physical/'step_0000001',physical/'best_step_0000000'
    (last/'iter_0000001').mkdir(parents=True)
    (last/'iter_0000001/.metadata').write_text('validated native checkpoint')
    best.mkdir()
    (best/'model.safetensors').write_text('retained trained weights')
    (physical/'.w4a8-owner.json').write_text(json.dumps({'logical_save_root':str(save.resolve())}))
    (save/'last').symlink_to(last)
    (save/'best').symlink_to(best)
    (save/'checkpoint-policy.json').write_text(json.dumps({'last':{'snapshot':str(last),'completed_updates':2}}))
    report={'passed':True,'nonzero_gradient_seen':True,'last':str(last),'best':str(best),'completed_updates':2}
    arm={'work':str(work),'save':str(save)}
    calls=[]
    def helper(command,check):
        assert '--gpus' not in command and 'none' in command
        assert command[command.index('--mount')+1]==f'type=bind,src={physical},dst=/snapshot'
        calls.append(command)
        shutil.rmtree(physical/command[-1])
    monkeypatch.setattr(run.subprocess,'run',helper)
    with pytest.raises(RuntimeError,match='unvalidated'):
        run.retire_smoke(arm,dict(report,nonzero_gradient_seen=False),'pinned-image')
    with pytest.raises(RuntimeError,match='outside owned'):
        run.retire_smoke(arm,dict(report,last=str(tmp_path/'foreign/step_0000001')),'pinned-image')
    assert not calls and last.exists()
    run.retire_smoke(arm,report,'pinned-image')
    assert not last.exists() and (best/'model.safetensors').is_file()
    assert not (save/'last').is_symlink()
    assert json.loads((save/'checkpoint-policy.json').read_text())['last'] is None
    run.retire_smoke(arm,report,'pinned-image')
    assert len(calls)==1
