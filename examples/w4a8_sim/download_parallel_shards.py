"""Resume pinned ModelScope shards using verified HTTP byte ranges.

Completed shards are SHA256-verified before publication. Existing contiguous
partials are retained. Run only after stopping this run's serial downloader.
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    metadata=json.loads((args.work/'model-info-modelscope.json').read_text())
    files=[f for f in metadata['Data']['Files'] if f['Path'].endswith('.safetensors')]
    chunks=args.work/'download-parts'
    chunks.mkdir(exist_ok=True)
    jobs=[]
    prefixes={}
    for f in files:
        target=args.destination/f['Path']
        if target.is_file() and digest(target)==f['Sha256']:
            continue
        partial=target.with_name(target.name+'.partial')
        prefix=partial.stat().st_size if partial.is_file() else 0
        if prefix>f['Size']:
            raise RuntimeError('Existing partial exceeds pinned file size')
        prefixes[f['Path']]=prefix
        for start in range(prefix,f['Size'],32*1024**2):
            jobs.append((f,start,min(start+32*1024**2,f['Size'])-1))
    report=dict(total_chunks=len(jobs),completed_chunks=0,completed=False)
    def fetch(job):
        f,start,end=job
        target=chunks/f"{f['Path']}.{start}-{end}"
        if target.is_file() and target.stat().st_size==end-start+1:
            return target
        temporary=target.with_name(target.name+'.tmp')
        headers=target.with_name(target.name+'.headers')
        url=f"https://modelscope.cn/api/v1/models/Qwen/Qwen3-4B/repo?Revision={f['Revision']}&FilePath={quote(f['Path'])}"
        subprocess.run(['curl','-fLsS','--retry','3','--connect-timeout','10','--max-time','1200',
            '--range',f'{start}-{end}','--dump-header',str(headers),'--output',str(temporary),url],check=True)
        ranges=re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)',headers.read_text(),flags=re.I)
        if not ranges or tuple(map(int,ranges[-1]))!=(start,end,f['Size']) or temporary.stat().st_size!=end-start+1:
            raise RuntimeError(f'Invalid HTTP range for {f["Path"]} at {start}')
        temporary.replace(target)
        return target
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for _ in pool.map(fetch,jobs):
                report['completed_chunks']+=1
                (args.work/'parallel-download.json').write_text(json.dumps(report,indent=2)+'\n')
        for f in files:
            if f['Path'] not in prefixes:
                continue
            target=args.destination/f['Path']
            partial=target.with_name(target.name+'.partial')
            prefix=prefixes[f['Path']]
            with partial.open('ab') as stream:
                for start in range(prefix,f['Size'],32*1024**2):
                    end=min(start+32*1024**2,f['Size'])-1
                    part=chunks/f"{f['Path']}.{start}-{end}"
                    with part.open('rb') as source:
                        for block in iter(lambda:source.read(8*1024**2),b''):
                            stream.write(block)
            if partial.stat().st_size!=f['Size'] or digest(partial)!=f['Sha256']:
                raise RuntimeError(f'Checksum mismatch after range assembly: {f["Path"]}')
            partial.replace(target)
            print('verified',f['Path'],flush=True)
            for part in chunks.glob(f['Path']+'.*'):
                part.unlink()
        # Reuse the main downloader's complete metadata/config/index validation.
        script=Path(__file__).with_name('download_model.py')
        subprocess.run(['python3',str(script),'--model','Qwen/Qwen3-4B','--source','modelscope',
            '--destination',str(args.destination),'--report',str(args.work/'model-download.json')],check=True)
        report['completed']=True
    except Exception as error:
        report['error']=str(error)
        status=json.loads((args.work/'model-download.json').read_text())
        status['error']='Parallel download: '+str(error)
        (args.work/'model-download.json').write_text(json.dumps(status,indent=2)+'\n')
        raise
    finally:
        (args.work/'parallel-download.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
