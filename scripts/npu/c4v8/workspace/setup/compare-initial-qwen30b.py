#!/usr/bin/env python3
"""CPU-only, bounded-memory sampled comparison of initial HF Qwen3-30B exports.

Uses safetensors indexes + read-only mmap; never imports torch/NPU/Ray or loads
whole tensors. Default PASS means sampled equality; --expect-sampled-changes instead requires real sampled weight changes. Neither is a full-file comparison.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import mmap
from pathlib import Path
import random
import struct
import sys

WIDTHS={'BOOL':1,'U8':1,'I8':1,'U16':2,'I16':2,'F16':2,'BF16':2,
        'U32':4,'I32':4,'F32':4,'U64':8,'I64':8,'F64':8,
        'F8_E4M3':1,'F8_E5M2':1,'F8_E4M3FN':1,'F8_E4M3FNUZ':1,'F8_E5M2FNUZ':1}
DEFAULT_LAYERS=[0,23,24,47]
DEFAULT_EXPERTS=[0,31,32,63,64,95,96,127]


class IndexedSafeTensors:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.index_path=self.root/'model.safetensors.index.json'
        if not self.index_path.is_file():
            raise ValueError(f'Missing completed HF index: {self.index_path}')
        raw=self.index_path.read_bytes()
        self.index_sha256=hashlib.sha256(raw).hexdigest()
        self.index=json.loads(raw)
        self.weight_map=self.index.get('weight_map')
        if not isinstance(self.weight_map,dict) or not self.weight_map:
            raise ValueError(f'Empty/invalid weight_map: {self.index_path}')
        missing=[name for name in sorted(set(self.weight_map.values())) if not (self.root/name).is_file()]
        if missing:
            raise ValueError(f'Incomplete directory, missing indexed shards: {missing[:10]}')
        self.files={}

    def close(self):
        for file,mapping,header,data_start in self.files.values():
            mapping.close();file.close()
        self.files.clear()

    def tensor(self,name):
        if name not in self.weight_map:
            raise KeyError(f'Tensor absent from HF index: {name}')
        filename=self.weight_map[name]
        if filename not in self.files:
            path=self.root/filename
            file=path.open('rb')
            try:
                mapping=mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)
                if len(mapping)<8:
                    raise ValueError(f'Truncated safetensors file: {path}')
                header_size=struct.unpack('<Q',mapping[:8])[0]
                if header_size>64*1024*1024 or 8+header_size>len(mapping):
                    raise ValueError(f'Invalid/truncated safetensors header: {path}')
                header=json.loads(mapping[8:8+header_size])
                self.files[filename]=(file,mapping,header,8+header_size)
            except BaseException:
                if 'mapping' in locals():mapping.close()
                file.close()
                raise
        file,mapping,header,data_start=self.files[filename]
        if name not in header:
            raise KeyError(f'Index/header mismatch for {name} in {filename}')
        spec=header[name]
        dtype=spec['dtype'];shape=spec['shape'];offsets=spec['data_offsets']
        width=WIDTHS.get(dtype)
        if width is None:
            raise ValueError(f'Unsupported raw element width for dtype {dtype}')
        count=math.prod(shape)
        begin,end=offsets
        if begin<0 or end<begin or end-begin!=count*width or data_start+end>len(mapping):
            raise ValueError(f'Invalid/truncated tensor storage for {name} in {filename}')
        return {'name':name,'file':filename,'dtype':dtype,'shape':shape,'numel':count,
                'width':width,'start':data_start+begin,'mapping':mapping}


def targets(layers,experts):
    names=['model.embed_tokens.weight','lm_head.weight']
    for layer in layers:
        names.extend(f'model.layers.{layer}.self_attn.{proj}_proj.weight' for proj in ['q','k','v','o'])
        for expert in experts:
            names.extend(f'model.layers.{layer}.mlp.experts.{expert}.{proj}_proj.weight' for proj in ['gate','up','down'])
    return names


def positions(count,per_kind,seed,name):
    if count<=0:return []
    n=min(count,per_kind)
    indices=set(range(n))|set(range(count-n,count))
    if n==1:indices.add(count//2)
    elif n>1:indices.update(i*(count-1)//(n-1) for i in range(n))
    derived=seed^int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little')
    indices.update(random.Random(derived).sample(range(count),n))
    return sorted(indices)


def value_at(tensor,index):
    start=tensor['start']+index*tensor['width']
    return tensor['mapping'][start:start+tensor['width']]


def decode(raw,dtype):
    if dtype=='BF16':return struct.unpack('<f',struct.pack('<I',int.from_bytes(raw,'little')<<16))[0]
    formats={'F16':'<e','F32':'<f','F64':'<d','I8':'<b','U8':'<B','I16':'<h','U16':'<H',
             'I32':'<i','U32':'<I','I64':'<q','U64':'<Q','BOOL':'<?'}
    try:
        return struct.unpack(formats[dtype],raw)[0] if dtype in formats else None
    except (struct.error,ValueError):return None


def coordinates(index,shape):
    result=[]
    for dim in reversed(shape):
        index,remainder=divmod(index,dim)
        result.append(remainder)
    return list(reversed(result))


def finite_sample(raw,dtype):
    if dtype in {'BF16','F16','F32','F64'}:
        return math.isfinite(decode(raw,dtype))
    if dtype.startswith('F8'):
        # Never silently accept an unvalidated floating format.
        raise ValueError(f'Sampled finiteness validation is not implemented for {dtype}')
    return True


def json_value(raw,dtype):
    value=decode(raw,dtype)
    return str(value) if isinstance(value,float) and not math.isfinite(value) else value


def compare(base,candidate,names,per_kind,seed):
    rows=[]
    for name in names:
        row={'name':name}
        try:
            left=base.tensor(name);right=candidate.tensor(name)
            row.update(base_file=left['file'],candidate_file=right['file'],
                       base_shape=left['shape'],candidate_shape=right['shape'],
                       base_dtype=left['dtype'],candidate_dtype=right['dtype'])
            row['shape_equal']=left['shape']==right['shape']
            row['dtype_equal']=left['dtype']==right['dtype']
            if not row['shape_equal'] or not row['dtype_equal']:
                row.update(sampled_exact_equal=False,valid=False,error='shape_or_dtype_mismatch')
            else:
                indices=positions(left['numel'],per_kind,seed,name)
                mismatch=[];nonfinite=[]
                changed_count=0;numeric_changed_count=0;nonfinite_count=0
                for index in indices:
                    a=value_at(left,index);b=value_at(right,index)
                    pair_finite=finite_sample(a,left['dtype']) and finite_sample(b,right['dtype'])
                    if not pair_finite:
                        nonfinite_count+=1
                        if len(nonfinite)<12:
                            nonfinite.append({'flat_index':index,'coordinates':coordinates(index,left['shape']),
                                              'base_hex':a.hex(),'candidate_hex':b.hex(),
                                              'base_value':json_value(a,left['dtype']),'candidate_value':json_value(b,right['dtype'])})
                    if a!=b:
                        changed_count+=1
                        if pair_finite and decode(a,left['dtype'])!=decode(b,right['dtype']):
                            numeric_changed_count+=1
                        if len(mismatch)<12:
                            mismatch.append({'flat_index':index,'coordinates':coordinates(index,left['shape']),
                                             'base_hex':a.hex(),'candidate_hex':b.hex(),
                                             'base_value':json_value(a,left['dtype']),'candidate_value':json_value(b,right['dtype'])})
                row.update(sample_count=len(indices),sample_indices=indices,mismatch_count=changed_count,
                           changed_sample_count=changed_count,numeric_changed_sample_count=numeric_changed_count,
                           nonfinite_sample_count=nonfinite_count,first_nonfinite_samples=nonfinite,
                           sampled_finite=nonfinite_count==0,valid=nonfinite_count==0,
                           first_mismatches=mismatch,sampled_exact_equal=changed_count==0)
                if nonfinite_count:row['error']='nonfinite_sampled_weights'
        except Exception as exc:
            row.update(sampled_exact_equal=False,valid=False,error=type(exc).__name__+': '+str(exc))
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base_hf_dir',type=Path)
    parser.add_argument('initial_export_hf_dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expect-sampled-changes',action='store_true',
                        help='Require at least one changed sampled weight element; all keys/shapes/dtypes/finite checks remain strict. Default requires sampled equality.')
    parser.add_argument('--seed',type=int,default=20261106)
    parser.add_argument('--samples-per-kind',type=int,default=32,
                        help='Element count each for first/last/uniform/random (deduplicated)')
    parser.add_argument('--layers',nargs='+',type=int,default=DEFAULT_LAYERS)
    parser.add_argument('--experts',nargs='+',type=int,default=DEFAULT_EXPERTS)
    args=parser.parse_args()
    if args.samples_per_kind<1:parser.error('--samples-per-kind must be positive')
    report={'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'base_hf_dir':str(args.base_hf_dir.resolve()),'initial_export_hf_dir':str(args.initial_export_hf_dir.resolve()),
            'seed':args.seed,'samples_per_kind':args.samples_per_kind,'layers':args.layers,'experts':args.experts,
            'expectation':'sampled_changes' if args.expect_sampled_changes else 'same',
            'method':'indexed safetensors raw-element bitwise comparison via read-only mmap; CPU only; no full tensor load or full weight hash',
            'scope':('PASS proves structurally valid finite sampled weights with at least one bitwise change; not a full-model comparison or training-quality metric' if args.expect_sampled_changes else 'PASS proves finite sampled equality for selected tensors, not full-model equality'),
            'tensor_key_policy':'strict same HF tensor names; missing/packed/renamed keys are reported as errors, never silently skipped'}
    base=candidate=None
    try:
        base=IndexedSafeTensors(args.base_hf_dir)
        candidate=IndexedSafeTensors(args.initial_export_hf_dir)
        report['base_index_sha256']=base.index_sha256
        report['candidate_index_sha256']=candidate.index_sha256
        names=targets(args.layers,args.experts)
        rows=compare(base,candidate,names,args.samples_per_kind,args.seed)
        if args.expect_sampled_changes:
            for row in rows:
                if 'first_mismatches' in row:row['first_changes']=row.pop('first_mismatches')
                row.pop('mismatch_count',None)
        report['tensors']=rows
        valid=[row for row in rows if row.get('valid',False)]
        changed=[row for row in valid if row.get('changed_sample_count',0)>0]
        unchanged=[row for row in valid if row['sampled_exact_equal']]
        changed_samples=sum(row.get('changed_sample_count',0) for row in valid)
        report.update(target_tensor_count=len(rows),valid_tensor_count=len(valid),invalid_tensor_count=len(rows)-len(valid),
                      changed_tensor_count=len(changed),unchanged_tensor_count=len(unchanged),
                      changed_sample_count=changed_samples,
                      numeric_changed_sample_count=sum(row.get('numeric_changed_sample_count',0) for row in valid),
                      nonfinite_sample_count=sum(row.get('nonfinite_sample_count',0) for row in rows),
                      passed_tensor_count=len(valid) if args.expect_sampled_changes else len(unchanged),
                      total_sampled_elements=sum(row.get('sample_count',0) for row in rows))
        if len(valid)!=len(rows):
            report.update(status='failed',reason='missing_key_shape_dtype_or_nonfinite_sample')
        elif args.expect_sampled_changes:
            report.update(status='passed' if changed_samples>0 else 'failed',
                          reason='sampled_weight_changes_observed' if changed_samples>0 else 'no_sampled_weight_changes')
        else:
            report.update(status='passed' if changed_samples==0 else 'failed',
                          reason='sampled_weights_equal' if changed_samples==0 else 'unexpected_sampled_weight_changes')
    except Exception as exc:
        report.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        if base is not None:base.close()
        if candidate is not None:candidate.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_name(args.output.name+'.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(args.output)
    marker=('TRAINED_QWEN30B_SAMPLED_CHANGES_' if args.expect_sampled_changes else 'INITIAL_QWEN30B_SAMPLED_COMPARE_')+report['status'].upper()
    print(marker+'='+str(args.output),flush=True)
    print(json.dumps({k:v for k,v in report.items() if k in {'status','target_tensor_count','passed_tensor_count','total_sampled_elements','changed_tensor_count','changed_sample_count','nonfinite_sample_count','reason','expectation','error'}}),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
