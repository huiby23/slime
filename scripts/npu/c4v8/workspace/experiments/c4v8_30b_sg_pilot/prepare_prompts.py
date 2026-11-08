#!/usr/bin/env python3
"""CPU tokenizer preparation only: first two real DAPO rows, no model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path('/root/hby/models/Qwen3-30B-A3B'))
    p.add_argument('--data',type=Path,default=Path('/root/hby/data/dapo-math-17k/train.jsonl'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Choose a fresh fixture path')
    os.environ.update(USE_TORCH='0',USE_TF='0',TORCH_DEVICE_BACKEND_AUTOLOAD='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    rows=[]
    with a.data.open() as f:
        for index in range(2):
            raw=next(f); item=json.loads(raw); messages=item['messages']
            if not messages or any(m.get('role') not in ('system','user','assistant') for m in messages):
                raise ValueError('Unexpected actual DAPO message structure')
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if not 1<=len(ids)<=8160 or any(type(x) is not int or not 0<=x<len(tokenizer) for x in ids):
                raise ValueError('Tokenized real prompt cannot fit8192 context plus32 output')
            rows.append(dict(name=f'dapo-row-{index}',source_line_index=index,source_line_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                messages=messages,input_ids=ids,prompt_tokens=len(ids),rendered_prompt=tokenizer.decode(ids,skip_special_tokens=False)))
    files=[a.model/n for n in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt'] if (a.model/n).is_file()]
    result=dict(kind='actual_dapo_nonthinking_token_ids_performance_only',model_path=str(a.model.resolve()),
        data_path=str(a.data.resolve()),data_sha256=sha(a.data),tokenizer_source_sha256={str(x.resolve()):sha(x) for x in files},
        vocabulary_size=len(tokenizer),chat_template_kwargs={'enable_thinking':False,'add_generation_prompt':True},
        prompts=rows,synthetic=False,optimizer_updates=0,may_be_used_for_training=False,
        torch_imported='torch' in sys.modules,npu_imported='torch_npu' in sys.modules,model_constructed=False)
    if result['torch_imported'] or result['npu_imported']:raise RuntimeError('Tokenizer-only process imported a device backend')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'prompt_tokens':[r['prompt_tokens'] for r in rows],'torch_imported':False,'npu_imported':False}))


if __name__=='__main__':main()
