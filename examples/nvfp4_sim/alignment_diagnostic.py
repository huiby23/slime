"""Frozen same-token Megatron/SGLang alignment, with no optimizer updates."""

import argparse
from datetime import timedelta
import importlib
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import torch

from replay_diagnostic import prepare_data, parameter_hash, parameters, save_json


def acquire(job):
    uuid = job['expected_gpu_uuid']
    pids = subprocess.check_output(['nvidia-smi', '--id='+uuid, '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True)
    if {int(p) for p in pids.splitlines() if p.strip()} - set(job.get('reservation_host_pids', [])):
        raise SystemExit(75)
    assert torch.cuda.device_count() == 1
    assert str(torch.cuda.get_device_properties(0).uuid).lower().replace('gpu-', '') == uuid.lower().replace('gpu-', '')
    return torch.empty(1, device='cuda')


def initialize(job):
    from slime.utils.arguments import parse_args
    from slime.utils.distributed_utils import init_gloo_group
    from slime.utils.eval_config import EvalDatasetConfig
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model import initialize_model_and_optimizer
    import torch.distributed as dist
    torch.serialization.add_safe_globals([EvalDatasetConfig])
    sys.argv = job['megatron_argv']
    args = parse_args()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), WORLD_SIZE='1', RANK='0', LOCAL_RANK='0')
    dist.init_process_group('nccl', rank=0, world_size=1, timeout=timedelta(minutes=5))
    init_gloo_group(); args.rank=0; args.world_size=1; args.num_gpus_per_node=1
    init(args)
    model, optimizer, scheduler, _ = initialize_model_and_optimizer(args, 'actor')
    assert optimizer is None and scheduler is None and len(model) == 1
    model[0].eval()
    return args, model


def megatron(job, output, status):
    from nvfp4_sim.diagnostic_trace import canonical_name, capture
    import nvfp4_sim.megatron as adapter
    from slime.backends.megatron_utils.data import DataIterator, get_batch
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
    args, model = initialize(job)
    params = parameters(model); fingerprint = parameter_hash(params)
    data, rows = prepare_data(job)
    for key, values in list(data.items()):
        if isinstance(values, list) and values and isinstance(values[0], torch.Tensor):
            data[key] = [v.to('cuda') for v in values]
    weight_names = {id(module.weight): canonical_name(name) for name, module in model[0].named_modules()
                    if canonical_name(name) and getattr(module, 'weight', None) is not None}
    for name, param in params:
        if name.endswith('embedding.word_embeddings.weight'):
            weight_names[id(param)] = 'head'
    records = {}; current = {}; original = adapter.nvfp4_linear

    def observed(x, weight, bias=None, *, recipe):
        result = original(x, weight, bias, recipe=recipe)
        tag = weight_names.get(id(weight))
        if current.get('trace') and tag and tag not in records:
            records[tag] = capture(tag, x, weight, result, current['valid_rows'], megatron=True)
        return result

    adapter.nvfp4_linear = observed
    iterator = DataIterator(data, data['micro_batch_indices'])
    results = []
    with torch.no_grad():
        for i, position in enumerate(data['partition']):
            batch = get_batch(iterator, ['tokens', 'total_lengths', 'response_lengths', 'loss_masks', 'partition'],
                              args.data_pad_size_multiplier, args.allgather_cp)
            current.update(trace=i == 0, valid_rows=rows[position]['total_lengths'])
            logits = model[0](input_ids=batch['tokens'], position_ids=None, attention_mask=None, labels=None,
                              packed_seq_params=batch['packed_seq_params'], loss_mask=batch['full_loss_masks'])
            _, probabilities = get_log_probs_and_entropy(logits.float(), args=args,
                unconcat_tokens=batch['unconcat_tokens'], total_lengths=batch['total_lengths'],
                response_lengths=batch['response_lengths'], with_entropy=False)
            lp = probabilities['log_probs'][0].float().cpu()
            expected = torch.as_tensor(rows[position]['log_probs']).float()
            results.append(dict(position=position, tokens=rows[position]['tokens'].tolist(),
                response_length=rows[position]['response_lengths'], log_probs=lp.tolist(),
                source_max_error=float((lp-expected).abs().max()), padded_tokens=batch['tokens'].numel()))
            status.update(state='forward_alignment', completed_samples=i+1)
            save_json(output/'status.json', status)
    assert parameter_hash(params) == fingerprint
    torch.save(records, output/'traces.pt')
    result=dict(passed=True, optimizer_steps=0, weights_unchanged=True, samples=results,
                trace_keys=list(records), parameter_sha256=fingerprint)
    save_json(output/'megatron-alignment.json', result)
    torch.distributed.destroy_process_group()
    return result


def errors(first, second):
    first=torch.as_tensor(first).float(); second=torch.as_tensor(second).float()
    assert first.shape == second.shape, (first.shape, second.shape)
    delta=(first-second).abs()
    return dict(mean_absolute=float(delta.mean()), maximum_absolute=float(delta.max()),
                relative_l2=float((first-second).norm()/first.norm().clamp_min(1e-30)))


def reference_check(traces):
    from nvfp4_sim.recipe import quantize
    ref=importlib.import_module('transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4')
    cls=ref.NVFP4QuantizerRef
    results={}
    for name, values in traces.items():
        if not name.startswith('layer0.'):
            continue
        for key, weight in [('reference_input',False), ('reference_weight',True)]:
            x=values[key]; x=torch.nn.functional.pad(x,(0,(-x.shape[-1])%16,0,(-x.shape[0])%16))
            q=quantize(x,weight=weight,reference_arithmetic=False)
            corrected=quantize(x,weight=weight,reference_arithmetic=True)
            method=cls._quantize_blockwise_reference
            args=(x,x.abs().max().float().view(1),16,16 if weight else 1)
            if next(iter(inspect.signature(method).parameters))=='cls':args=(cls,*args)
            payload,scale=method(*args,pow_2_scales=False,eps=0.)
            decoded=ref.cast_from_fp4x2(payload,torch.float32)
            results[name+'.'+key]=dict(payload_equal=bool(torch.equal(q.values.float(),decoded)),
                block_scales_equal=bool(torch.equal(q.scales,scale.float())),
                payload_mismatches=int((q.values.float()!=decoded).sum()),
                block_scale_mismatches=int((q.scales!=scale.float()).sum()),
                corrected_payload_equal=bool(torch.equal(corrected.values.float(),decoded)),
                corrected_block_scales_equal=bool(torch.equal(corrected.scales,scale.float())))
    return results


def sglang(job, output, status):
    import sglang as sg
    from evaluate_diagnostic import prepare
    base=json.loads(Path(job['megatron_result']).read_text())
    samples=base['samples']; traces=torch.load(Path(job['megatron_result']).with_name('traces.pt'),weights_only=False)
    _,_,kwargs,_=prepare(job)
    comparisons={}; engine=None
    try:
        for cache in [True,False]:
            overrides=json.loads(kwargs['json_model_override_args'])
            overrides.update(architectures=['NVFP4AlignmentQwen3ForCausalLM'],nvfp4_sim_weight_cache=cache,
                nvfp4_alignment_trace_tokens=samples[0]['tokens'],
                nvfp4_alignment_trace_path=str(output/('traces-cache-'+str(cache)+'.pt')))
            kwargs['json_model_override_args']=json.dumps(overrides)
            engine=sg.Engine(**kwargs)
            for batched in [False,True]:
                responses=engine.generate(input_ids=[s['tokens'] for s in samples],
                    sampling_params={'temperature':1.,'top_k':1,'max_new_tokens':1},
                    return_logprob=True,logprob_start_len=0) if batched else [engine.generate(
                        input_ids=s['tokens'],sampling_params={'temperature':1.,'top_k':1,'max_new_tokens':1},
                        return_logprob=True,logprob_start_len=0) for s in samples]
                rows=[]
                for source,response in zip(samples,responses,strict=True):
                    raw=response['meta_info']['input_token_logprobs']
                    assert len(raw)==len(source['tokens']), (len(raw),len(source['tokens']))
                    assert [r[1] for r in raw]==source['tokens']
                    values=[r[0] for r in raw[-source['response_length']:]]
                    rows.append(dict(position=source['position'],log_probs=values,
                                     difference_from_megatron=errors(source['log_probs'],values)))
                name='cache-'+str(cache)+'-batch-'+str(batched)
                comparisons[name]=rows;status.update(state='forward_alignment',current_case=name)
                save_json(output/'status.json',status)
            if cache:
                source=samples[0];start=len(source['tokens'])-source['response_length']
                prefix=[]
                for position in range(start,min(start+4,len(source['tokens']))):
                    response=engine.generate(input_ids=source['tokens'][:position+1],
                        sampling_params={'temperature':1.,'top_k':1,'max_new_tokens':1},
                        return_logprob=True,logprob_start_len=max(0,position-1))
                    raw=response['meta_info']['input_token_logprobs']
                    assert raw[-1][1]==source['tokens'][position]
                    prefix.append(raw[-1][0])
                comparisons['cached_prefix_extensions']=dict(position=source['position'],log_probs=prefix,
                    difference_from_megatron=errors(source['log_probs'][:len(prefix)],prefix),
                    note='Repeated growing teacher-forced prefixes; extension/prefix-cache sensitivity, not a free-running decode accuracy test.')
            engine.shutdown();engine=None
        sg_traces=torch.load(output/'traces-cache-True.pt',weights_only=False)
        layer_errors={name:dict(input=errors(traces[name]['input'],values['input']),
            output=errors(traces[name]['output'],values['output']) if values['output'] is not None else None,
            weights_equal=traces[name]['weight_sha256']==values['weight_sha256'],
            megatron_amax=traces[name]['input_amax'],sglang_amax=values['input_amax'],
            megatron_rows=traces[name]['rows'],sglang_rows=values['rows']) for name,values in sg_traces.items()}
        ref=reference_check(traces)
        cache_errors=[errors(a['log_probs'],b['log_probs']) for a,b in zip(
            comparisons['cache-True-batch-False'],comparisons['cache-False-batch-False'],strict=True)]
        batch_errors=[errors(a['log_probs'],b['log_probs']) for a,b in zip(
            comparisons['cache-True-batch-False'],comparisons['cache-True-batch-True'],strict=True)]
        result=dict(passed=True,optimizer_steps=0,comparisons=comparisons,layers=layer_errors,
                    cache_toggle_difference=cache_errors,batching_difference=batch_errors,
                    nvidia_software_reference=ref,
                    note='Same frozen weights and teacher-forced tokens. This does not establish hardware bitwise equivalence.')
        result['reference_exact_match']=bool(ref) and all(v['payload_equal'] and v['block_scales_equal'] for v in ref.values())
        result['weights_match']=all(v['weights_equal'] for v in layer_errors.values())
        save_json(output/'alignment-comparison.json',result)
        # Numerical mismatches are the measurement in this diagnostic. Preserve
        # the complete report instead of discarding it through an assertion.
        return result
    finally:
        if engine is not None:engine.shutdown()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job',type=Path);parser.add_argument('--preflight',action='store_true')
    cli=parser.parse_args();job=json.loads(cli.job.read_text());output=Path(job['output']);output.mkdir(exist_ok=True)
    if cli.preflight:
        assert job['component'] in ('megatron','sglang')
        data,rows=prepare_data(job)
        save_json(output/'configuration-check.json',dict(passed=True,sample_positions=list(rows),optimizer_steps=0))
        return
    pin=acquire(job)
    status=dict(state='initializing',job_id=job['job_id'],started_at=time.time(),optimizer_steps=0)
    save_json(output/'status.json',status)
    os.environ['SGLANG_EXTERNAL_MODEL_PACKAGE']='nvfp4_sim.sglang_models'
    try:
        result=(megatron if job['component']=='megatron' else sglang)(job,output,status)
        status.update(state='completed',result=result)
    except BaseException as exc:
        status.update(state='failed',error=repr(exc));raise
    finally:save_json(output/'status.json',status)


if __name__=='__main__':main()
