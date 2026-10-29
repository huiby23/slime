"""The two arms must differ only in precision hooks and artifact destinations."""

import importlib.util
import json
from pathlib import Path

NUM_GPUS=0


def module(name):
    path=Path(__file__).resolve().parents[1]/'examples/w4a8_sim'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_matched_dense_arms_and_resume_preserve_model_and_data(tmp_path):
    model=tmp_path/'model'
    model.mkdir()
    config=dict(model_type='qwen3',num_hidden_layers=36,hidden_size=2560,
        tie_word_embeddings=True,intermediate_size=9728,num_attention_heads=32,
        num_key_value_heads=8,head_dim=128,rms_norm_eps=1e-6,rope_theta=1000000,vocab_size=151936)
    (model/'config.json').write_text(json.dumps(config))
    (model/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':8044936192}}))
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
    assert command[command.index('--load')+1]==str(tmp_path/'smoke/checkpoints/w4a8/last')
    assert command[command.index('--hf-checkpoint')+1]==str(model)
    assert '--untie-embeddings-and-output-weights' not in command
    assert resumed['environment']['W4A8_BEST_WEIGHTS_ONLY']=='1'
    assert json.loads(resumed['environment']['W4A8_CHECKPOINT_ROOTS'])==[str(tmp_path/'smoke/checkpoints/w4a8/.snapshots')]
