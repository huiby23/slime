"""Compare token IDs, prompt/output probabilities and measured client throughput."""
import argparse
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


def memory_peak(folder):
    external = folder / "hbm-summary.json"
    if external.is_file():
        report = json.loads(external.read_text())
        if report["samples"] < 2:
            raise ValueError("Insufficient external HBM samples")
    else:
        report = json.loads((folder / "result.json").read_text())
        if report.get("hbm_error") or report.get("hbm_valid_samples", 0) < 2:
            raise ValueError(f"HBM sampler did not cover the run in {folder}")
    peak = report["peak_sampled_hbm_mib"]
    if len(peak) != 8 or min(peak) < 8000:
        raise ValueError("Eight resident model shards are missing from the HBM sample")
    return peak


def compare(left, right, memory_baseline=None):
    summaries = [json.loads((folder / "result.json").read_text()) for folder in (left,right)]
    if not all(r.get("completed") for r in summaries):
        raise ValueError("Incomplete server measurement")
    l,r = summaries
    if l["policy"] != r["policy"] or l["implementation"] == r["implementation"]:
        raise ValueError("Expected identical numerical policy and distinct executable implementations")
    if {k:v for k,v in l["server"].items() if k!="port"} != {k:v for k,v in r["server"].items() if k!="port"}:
        raise ValueError("Serving configurations disagree")
    if [(s["length"],s["repeat"]) for s in l["samples"]] != [(s["length"],s["repeat"]) for s in r["samples"]]:
        raise ValueError("Unmatched request sets")
    if memory_baseline is not None:
        m = json.loads((memory_baseline / "result.json").read_text())
        config = lambda r: {k:v for k,v in r["server"].items() if k != "port"}
        if (not m.get("completed") or config(m) != config(l) or m["implementation"] != l["implementation"]
                or {x["length"] for x in m["samples"]} != {x["length"] for x in l["samples"]}):
            raise ValueError("Memory rerun must use the paired baseline implementation and workload")
    result = dict(policy=l["policy"], baseline=l["implementation"], candidate=r["implementation"],
                  peak_sampled_hbm_mib=[memory_peak(memory_baseline or left),memory_peak(right)],
                  baseline_memory_run=str(memory_baseline or left), pairs=[],
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
    ref_variability = []
    for folder in (left, right):
        checks = {}
        for length in {x["length"] for x in result["pairs"]}:
            first, second = [json.loads((folder/f"len{length}-repeat{i}-outputs.json").read_text()) for i in (1,2)]
            checks[length] = {}
            for kind in ("prompt","output"):
                values = [(x,y) for a,b in zip(first,second,strict=True) for x,y in zip(a[kind],b[kind],strict=True)]
                checks[length][kind] = dict(max_abs_logprob_diff=max(abs(x[0]-y[0]) for x,y in values if x[0] is not None),
                                            token_mismatches=sum(x[1]!=y[1] for x,y in values))
        ref_variability.append(checks)
    result["within_implementation_repeat_variability"] = ref_variability
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
    parser.add_argument("--memory-baseline",type=Path)
    parser.add_argument("--paired-attention",type=Path,help="Full same-input 48-layer reference/candidate comparison")
    args=parser.parse_args()
    report=compare(args.baseline,args.candidate,args.memory_baseline)
    if args.paired_attention:
        checks=json.loads(args.paired_attention.read_text())
        if (checks.get("identical") is not True or checks.get("layers")!=48 or checks.get("ranks")!=8
                or checks["baseline_implementation"]!=report["baseline"] or checks["candidate_implementation"]!=report["candidate"]
                or checks["policy_sha256"]!=report["policy"]):
            raise ValueError("Paired real-model evidence does not cover these two implementations")
        report["same_input_attention_bitwise_equal"]=True
        report["paired_attention_evidence"]=str(args.paired_attention)
    report["independent_logprobs_and_ids_bitwise_equal"]=report["exact"]
    report["greedy_tokens_equal"]=all(x["token_mismatches"]==0 for x in report["pairs"])
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ("exact","warm_median_by_prefix","peak_sampled_hbm_mib")}))
    if not report["greedy_tokens_equal"]:
        raise SystemExit("Different greedy tokens")
    if not report["exact"] and not report.get("same_input_attention_bitwise_equal"):
        raise SystemExit("Independent logprobs differ; a paired same-input check is required")
