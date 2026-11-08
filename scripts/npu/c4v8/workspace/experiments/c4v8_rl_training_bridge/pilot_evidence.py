"""Read actual first-update evidence; never samples data or changes optimizer state."""
import json
import math
import os
from pathlib import Path
import re
import time


def read_progress(run, *, minimum_file_age=2):
    run=Path(run);now=time.time();records=[]
    files=[]
    for path in run.glob('grad_norm_actor_*.pt'):
        match=re.fullmatch(r'grad_norm_actor_(\d+)_(\d+)\.pt',path.name)
        if not match:raise ValueError('Unexpected gradient record name')
        rollout,step=map(int,match.groups())
        if not 0<=rollout<5 or not 0<=step<20:raise ValueError('Gradient record outside this100-update schedule')
        if now-path.stat().st_mtime<minimum_file_age:continue
        files.append((rollout*20+step+1,path))
    if files:
        os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
        import torch
        for step,path in sorted(files):
            value=float(torch.load(path,map_location='cpu',weights_only=True))
            if not math.isfinite(value) or value<=0:raise ValueError('Nonfinite/nonpositive actual gradnorm: '+str(path))
            records.append(dict(completed_step=step,gradient_norm=value,path=str(path)))
    if [row['completed_step'] for row in records]!=list(range(1,len(records)+1)):
        raise ValueError('Actual gradient records are not one contiguous fresh-model sequence')
    complete_checkpoints={}
    for metadata_path in sorted((run/'optimizer-step-checkpoints').glob('checkpoint_step_*/optimizer-step-checkpoint.json')):
        checkpoint=metadata_path.parent
        match=re.fullmatch(r'checkpoint_step_(\d{6})',checkpoint.name)
        if not match:raise ValueError('Unexpected checkpoint directory name')
        step=int(match.group(1))
        if step not in range(5,101,5):raise ValueError('Checkpoint outside every5 schedule')
        metadata=json.loads(metadata_path.read_text())
        if metadata.get('complete') is not True:continue
        index=json.loads((checkpoint/'model.safetensors.index.json').read_text())
        shards=set(index['weight_map'].values())
        if (metadata.get('completed_optimizer_steps')!=step or metadata.get('hf_only') is not True
                or metadata.get('parent_completed_optimizer_steps')!=0
                or metadata.get('new_successful_optimizer_steps')!=step
                or not shards or any(Path(s).name!=s or not (checkpoint/s).is_file() for s in shards)):
            raise ValueError('Complete HF checkpoint has wrong provenance or missing shards')
        complete_checkpoints[step]=str(checkpoint)
    confirmed=max(complete_checkpoints,default=0)
    if confirmed>len(records):
        # A complete checkpoint can be published while its last gradient file
        # is younger than the read-stability window; count it on a later check.
        confirmed=max((n for n in complete_checkpoints if n<=len(records)),default=0)
    return dict(gradient_norm_records=len(records),first_gradient_observed=bool(records),
        gradient_record_scope='Observed finite positive gradnorm files; not independently proof of optimizer success',
        successful_updates_proven_by_complete_hf=confirmed,
        first_five_gradients=records[:5],first_five_checkpoint_complete=5 in complete_checkpoints,
        first_five_checkpoint=complete_checkpoints.get(5),
        final_hundred_checkpoint_complete=100 in complete_checkpoints,
        final_hundred_checkpoint=complete_checkpoints.get(100),
        pilot_passed=len(records)>=5 and 5 in complete_checkpoints,
        optimizer_reinitialized_by_controller=False,optimizer_state_touched=False)
