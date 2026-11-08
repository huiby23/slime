"""Compare token IDs, prompt/output probabilities and measured client throughput."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def paired_metrics(folder, length, repeat, answers):
    prefix = f"len{length}-repeat{repeat}"
    # SSE timestamps are measured from each request's release from the same
    # thread barrier. Tail numbers exclude initial prefill and carry client jitter.
    events = []
    for row in answers:
        items = [json.loads(line) for line in (folder / f"{prefix}-r{row['rank']}.jsonl").read_text().splitlines()]
        events.append([(v["client_elapsed_seconds"], v["body"]["meta_info"]["completion_tokens"]) for v in items])
    start = max(row["first"] for row in answers)
    end = max(row["last"] for row in answers)
    count = sum(wave[-1][1] - max([c for t,c in wave if t<=start]+[0]) for wave in events)
    return dict(seconds=end-start, tokens=count, batch_decode_tokens_s=count/(end-start))


def compare(left, right):
    summaries = [json.loads((folder / "result.json").read_text()) for folder in (left,right)]
    if not all(r.get("completed") and "hbm_error" not in r for r in summaries):
        raise ValueError("Incomplete server measurement")
    l,r = summaries
    if l["policy"] != r["policy"] or l["implementation"] == r["implementation"]:
        raise ValueError("Expected identical numerical policy and distinct executable implementations")
    if {k:v for k,v in l["server"].items() if k!="port"} != {k:v for k,v in r["server"].items() if k!="port"}:
        raise ValueError("Serving configurations disagree")
    if [(s["length"],s["repeat"]) for s in l["samples"]] != [(s["length"],s["repeat"]) for s in r["samples"]]:
        raise ValueError("Unmatched request sets")
    result = dict(policy=l["policy"], baseline=l["implementation"], candidate=r["implementation"],
                  peak_sampled_hbm_mib=[l["peak_sampled_hbm_mib"], r["peak_sampled_hbm_mib"]], pairs=[],
                  optimizer_updates=0)
    exact = True
    for a,b in zip(l["samples"],r["samples"],strict=True):
        length,repeat=a["length"],a["repeat"]
        filename=f"len{length}-repeat{repeat}-outputs.json"
        aa,bb=[json.loads((folder/filename).read_text()) for folder in (left,right)]
        row=dict(length=length,repeat=repeat,e2e_tokens_s=[a["tokens_s"],b["tokens_s"]],token_mismatches=0,
                 checked_prompt=0,checked_output=0,max_abs_logprob_diff=0.,baseline=[],candidate=[])
        for x,y in zip(aa,bb,strict=True):
            for kind in ("prompt","output"):
                for (p,i,*_),(q,j,*_) in zip(x[kind],y[kind],strict=True):
                    row["checked_"+kind]+=1
                    row["token_mismatches"]+=int(i!=j)
                    if (p is None)!=(q is None):
                        raise ValueError("Missing logprob")
                    if p is not None:
                        row["max_abs_logprob_diff"]=max(row["max_abs_logprob_diff"],abs(p-q))
        exact &= row["token_mismatches"]==0 and row["max_abs_logprob_diff"]==0
        row["baseline"]=paired_metrics(left,length,repeat,aa)
        row["candidate"]=paired_metrics(right,length,repeat,bb)
        result["pairs"].append(row)
    result["exact"] = exact
    result["warm_median_by_prefix"] = {}
    for length in sorted({row["length"] for row in result["pairs"]}):
        rows=[row for row in result["pairs"] if row["length"]==length and row["repeat"]>0]
        result["warm_median_by_prefix"][length] = dict(
            decode_tokens_s=[statistics.median(x[v]["batch_decode_tokens_s"] for x in rows) for v in ("baseline","candidate")],
            e2e_tokens_s=[statistics.median(x["e2e_tokens_s"][i] for x in rows) for i in range(2)])
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline",type=Path)
    parser.add_argument("candidate",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    report=compare(args.baseline,args.candidate)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ("exact","warm_median_by_prefix","peak_sampled_hbm_mib")}))
    if not report["exact"]:
        raise SystemExit("Token/logprob mismatch: examine paired requests before choosing the optimization")
