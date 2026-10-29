"""Mark GPUs of this work's running containers during slow CPU initialization.

Only containers bearing our label AND this exact work mount are eligible. This
does not select additional GPUs, kill foreign owners, or restart training. The
copied auxiliary script and its hash are recorded separately from frozen model
sources. It keeps 64 MiB per visible device until that container exits.
"""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=604800)
    args=parser.parse_args()
    work=args.work.resolve()
    script=Path(__file__).with_name('gpu_claim.py')
    digest=hashlib.sha256(script.read_bytes()).hexdigest()
    directory=work/'gpu-claims'
    directory.mkdir(exist_ok=True)
    known={}
    deadline=time.monotonic()+args.seconds
    while time.monotonic()<deadline:
        try:
            ids=subprocess.check_output(['docker','ps','-q','--filter','label=lab.w4a8=qwen3-4b'],text=True).split()
            for cid in ids:
                if cid in known:
                    continue
                info=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
                if not any(m['Source']==str(work) and m['Destination']==str(work) for m in info['Mounts']):
                    continue
                target=f'/tmp/w4a8_gpu_claim_{digest[:12]}.py'
                report=directory/(info['Name'].lstrip('/')+'.json')
                if not report.exists():
                    subprocess.run(['docker','cp',str(script),cid+':'+target],check=True,stdout=subprocess.DEVNULL)
                    subprocess.run(['docker','exec','-d',cid,'python3',target,
                        '--parent-pid','1','--report',str(report)],check=True)
                known[cid]=dict(container=info['Name'].lstrip('/'),report=str(report),script_sha256=digest,
                    auxiliary_script=target,at=datetime.now(timezone.utc).isoformat())
                (directory/(cid+'-manifest.json')).write_text(json.dumps(known[cid],indent=2)+'\n')
            (directory/'watcher-status.json').write_text(json.dumps(dict(
                at=datetime.now(timezone.utc).isoformat(),containers=known),indent=2)+'\n')
        except Exception as error:
            (directory/'watcher-error.json').write_text(json.dumps(dict(
                at=datetime.now(timezone.utc).isoformat(),error=f'{type(error).__name__}: {error}'),indent=2)+'\n')
        time.sleep(1)


if __name__=='__main__':
    main()
