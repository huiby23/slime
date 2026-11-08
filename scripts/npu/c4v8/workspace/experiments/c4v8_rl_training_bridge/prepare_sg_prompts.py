"""Prepare a real tokenizer-derived140-token request; CPU only, no model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('/root/hby/models/Qwen3-0.6B'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Choose a fresh prompt fixture path')
    os.environ.update(USE_TORCH='0',USE_TF='0',TORCH_DEVICE_BACKEND_AUTOLOAD='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    text=('Consider a sequence of positive integers. Each next number is two greater than the previous one. '
          'Write the first few terms, then identify the pattern and add the terms carefully. ')*12
    all_ids=tokenizer.encode(text,add_special_tokens=True)
    ids=all_ids[:140]
    assert len(ids)==140 and all(type(x) is int and 0<=x<len(tokenizer) for x in ids)
    paths=[args.model/name for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt') if (args.model/name).exists()]
    report=dict(kind='offline_real_tokenizer_request_fixture',model_path=str(args.model.resolve()),
        tokenizer_class=type(tokenizer).__name__,vocabulary_size=len(tokenizer),input_ids=ids,
        prompt_tokens=len(ids),decoded_prompt=tokenizer.decode(ids,skip_special_tokens=False),
        original_text=text,selection='first140 tokens from authentic offline tokenizer; request sends input_ids directly',
        tokenizer_source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        torch_imported='torch' in sys.modules,npu_imported='torch_npu' in sys.modules,model_constructed=False)
    assert not report['torch_imported'] and not report['npu_imported']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_ids','original_text','decoded_prompt','tokenizer_source_sha256')}))

if __name__=='__main__':main()
