"""Export measured21-point AIME accuracy/truncation; no interpolated checkpoints."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def validated_rows(curve):
    if curve.get('complete') is not True:raise ValueError('Refuse to plot an incomplete curve as completed')
    points=curve['all_point_summaries']
    if [p['optimizer_step'] for p in points]!=list(range(0,101,5)):raise ValueError('Need exactly21 ordered measured points')
    identity={p['experiment_identity_sha256'] for p in points}
    protocol={p['protocol_sha256'] for p in points}
    if len(identity)!=1 or len(protocol)!=1:raise ValueError('Different experiments/protocols cannot form one curve')
    result=[]
    for p in points:
        if (p.get('valid_curve_point') is not True or p.get('questions')!=30 or p.get('successful_requests')!=30
            or p.get('errors')!=0 or type(p['correct']) is not int or type(p['truncated']) is not int
            or not 0<=p['correct']<=30 or not 0<=p['truncated']<=30
            or not math.isfinite(p['accuracy']) or not math.isclose(p['accuracy'],p['correct']/30,abs_tol=1e-12)):
            raise ValueError('Invalid actual30-question result')
        result.append(dict(optimizer_step=p['optimizer_step'],correct=p['correct'],questions=30,
            accuracy_percent=p['correct']/30*100,truncated=p['truncated'],truncation_percent=p['truncated']/30*100,
            model_path=p['model_path'],experiment_identity_sha256=p['experiment_identity_sha256'],
            protocol_sha256=p['protocol_sha256']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curve',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-fixture',action='store_true',help='Mark every artifact as synthetic CPU test data')
    args=parser.parse_args();raw=args.curve.read_bytes();curve=json.loads(raw);rows=validated_rows(curve)
    args.output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,6.5),sharex=True,layout='constrained')
    steps=[r['optimizer_step'] for r in rows]
    for ax,key,label,color in [(axes[0],'accuracy_percent','Accuracy (%)','#176b87'),
                              (axes[1],'truncation_percent','Truncated answers (%)','#b75a28')]:
        ax.plot(steps,[r[key] for r in rows],marker='o',markersize=4,color=color,linewidth=1.5)
        ax.set_ylabel(label);ax.set_ylim(-2,102);ax.set_yticks(range(0,101,20));ax.grid(alpha=.22)
        ax.spines[['top','right']].set_visible(False)
    axes[1].set_xlabel('Successful optimizer updates');axes[1].set_xticks(range(0,101,10))
    fig.suptitle('Qwen3-30B-A3B | AIME2024 (30 questions per checkpoint)\n'
        'BF16 backbone; FA NVFP4/MXFP8 forward simulation; nonthinking',fontsize=12)
    if args.cpu_fixture:fig.text(.5,.5,'CPU FIXTURE — NOT MODEL RESULTS',ha='center',rotation=25,alpha=.45,color='red',fontsize=20)
    files={}
    for extension in ('png','svg','pdf'):
        path=args.output/('aime2024-accuracy-truncation.'+extension)
        fig.savefig(path,dpi=180);files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    csv_path=args.output/'measured-points.csv'
    with csv_path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    files[csv_path.name]=hashlib.sha256(csv_path.read_bytes()).hexdigest()
    report=dict(status='cpu_fixture' if args.cpu_fixture else 'completed',synthetic_fixture=args.cpu_fixture,
        actual_measured_points=0 if args.cpu_fixture else 21,source_curve=str(args.curve.resolve()),
        source_curve_sha256=hashlib.sha256(raw).hexdigest(),files_sha256=files,
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib_version=matplotlib.__version__,
        scope='Every point uses30 actual questions; line only connects measured checkpoints; no interpolated scores')
    (args.output/'plot-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
