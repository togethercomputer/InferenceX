#!/usr/bin/env python3
"""together_runner result tooling.

Two subcommands:
  emit     map a raw sglang bench_serving output -> InferenceX-schema result JSON
  compare  diff a result JSON against the committed baseline; non-zero exit on
           throughput regression beyond the threshold.

Result schema (compatible-by-design with utils/process_result.py wrapper keys):
  hw, model, framework, precision, isl, osl, tp, ep, conc, profile, image,
  tuning, host, ts, metrics{...}, gpu{...}
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone


# --------------------------- shared helpers --------------------------------
def _load_bench(path):
    """sglang bench_serving --output-file writes JSON (sometimes JSONL).
    Return the last JSON object found."""
    with open(path) as f:
        txt = f.read().strip()
    try:
        obj = json.loads(txt)
        return obj[-1] if isinstance(obj, list) else obj
    except json.JSONDecodeError:
        last = None
        for line in txt.splitlines():
            line = line.strip()
            if line:
                last = json.loads(line)
        if last is None:
            raise
        return last


def _get(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _gpu_stats(csv_path):
    """Parse a host nvidia-smi CSV (power.draw column) -> approx power stats."""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    powers, idxs = [], set()
    with open(csv_path) as f:
        header = f.readline()
        cols = [c.strip() for c in header.split(",")]
        # locate columns
        pcol = next((i for i, c in enumerate(cols) if c.startswith("power.draw")), 2)
        icol = next((i for i, c in enumerate(cols) if c == "index"), 1)
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) <= pcol:
                continue
            try:
                powers.append(float(parts[pcol].split()[0]))
                idxs.add(parts[icol])
            except (ValueError, IndexError):
                continue
    if not powers:
        return {}
    n = max(len(idxs), 1)
    mean_per_gpu = sum(powers) / len(powers)
    return {
        "n_gpus": n,
        "mean_power_per_gpu_w": round(mean_per_gpu, 1),
        "total_avg_power_w": round(mean_per_gpu * n, 1),   # approx
        "peak_power_per_gpu_w": round(max(powers), 1),
    }


# --------------------------- emit ------------------------------------------
def cmd_emit(a):
    b = _load_bench(a.raw)
    metrics = {
        "successful_requests": _get(b, "completed", "successful_requests"),
        "total_token_throughput": _get(b, "total_token_throughput", "total_throughput"),
        "output_token_throughput": _get(b, "output_token_throughput", "output_throughput"),
        "request_throughput": _get(b, "request_throughput"),
        "median_ttft_ms": _get(b, "median_ttft_ms"),
        "p99_ttft_ms": _get(b, "p99_ttft_ms"),
        "median_tpot_ms": _get(b, "median_tpot_ms"),
        "p99_tpot_ms": _get(b, "p99_tpot_ms"),
        "median_itl_ms": _get(b, "median_itl_ms"),
        "median_e2el_ms": _get(b, "median_e2e_latency_ms", "median_e2el_ms"),
        "p99_e2el_ms": _get(b, "p99_e2e_latency_ms", "p99_e2el_ms"),
    }
    gpu = _gpu_stats(a.gpu_csv)
    out_tp = metrics.get("output_token_throughput")
    if gpu.get("total_avg_power_w") and out_tp:
        gpu["tokens_per_kw"] = round(out_tp / (gpu["total_avg_power_w"] / 1000.0), 1)

    result = {
        "hw": a.hw, "cluster": a.cluster, "model": a.model, "framework": a.framework,
        "precision": a.precision, "isl": a.isl, "osl": a.osl,
        "tp": a.tp, "ep": a.ep, "conc": a.conc, "profile": a.profile,
        "image": a.image, "tuning": int(a.tuning), "host": a.host,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics, "gpu": gpu,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


# --------------------------- compare ---------------------------------------
def _seqtag(isl, osl):
    f = lambda n: f"{n // 1024}k" if n % 1024 == 0 else str(n)
    return f(isl) + f(osl)


def _baseline_candidates(baselines_dir, r):
    # Tuning is part of the key (autotune alone moves throughput ~4-10%).
    # Try cluster-specific golden first, then fall back to the hw-wide golden.
    tune = "tuned" if int(r.get("tuning", 0)) == 1 else "untuned"
    seq, leaf = _seqtag(r["isl"], r["osl"]), f"conc{r['conc']}.json"
    cands = []
    if r.get("cluster"):
        cands.append(os.path.join(baselines_dir, r["hw"], r["cluster"],
                                  r["profile"], seq, tune, leaf))
    cands.append(os.path.join(baselines_dir, r["hw"], r["profile"], seq, tune, leaf))
    return cands


# (metric, higher_is_better)
_REPORT = [
    ("total_token_throughput", True),
    ("output_token_throughput", True),
    ("request_throughput", True),
    ("median_ttft_ms", False),
    ("median_tpot_ms", False),
    ("median_e2el_ms", False),
]


def cmd_compare(a):
    with open(a.result) as f:
        cur = json.load(f)
    if a.baseline:
        bpath = a.baseline if os.path.exists(a.baseline) else None
    else:
        bpath = next((c for c in _baseline_candidates(a.baselines_dir, cur)
                      if os.path.exists(c)), None)
    if not bpath:
        print("[compare] no baseline found. Looked at:")
        for c in ([a.baseline] if a.baseline else _baseline_candidates(a.baselines_dir, cur)):
            print(f"           {c}")
        print("[compare] record one with ENABLE_TUNING=1 and commit it. Exit OK.")
        return 0
    with open(bpath) as f:
        base = json.load(f)

    cm, bm = cur["metrics"], base["metrics"]
    print(f"\n=== delta vs baseline ({os.path.relpath(bpath)}) ===")
    print(f"{'metric':<26}{'baseline':>14}{'current':>14}{'delta%':>10}  note")
    regression = False
    for m, higher_better in _REPORT:
        bv, cv = bm.get(m), cm.get(m)
        if bv in (None, 0) or cv is None:
            print(f"{m:<26}{str(bv):>14}{str(cv):>14}{'n/a':>10}")
            continue
        d = (cv - bv) / bv * 100.0
        note = ""
        if m == "total_token_throughput":  # primary pass/fail
            if d < -a.threshold:
                note, regression = "REGRESSION ❌", True
            elif d > a.threshold:
                note = "IMPROVED ✅"
            else:
                note = "within ±%.0f%%" % a.threshold
        else:
            worse = (d < 0) if higher_better else (d > 0)
            note = "worse" if worse and abs(d) > a.threshold else ""
        print(f"{m:<26}{bv:>14.2f}{cv:>14.2f}{d:>+9.1f}%  {note}")

    if cur.get("gpu", {}).get("tokens_per_kw") and base.get("gpu", {}).get("tokens_per_kw"):
        b_e, c_e = base["gpu"]["tokens_per_kw"], cur["gpu"]["tokens_per_kw"]
        print(f"{'tokens_per_kw':<26}{b_e:>14.1f}{c_e:>14.1f}{(c_e-b_e)/b_e*100:>+9.1f}%")

    print("=" * 64)
    if regression:
        print(f"VERDICT: REGRESSION ❌ (throughput dropped >{a.threshold:.0f}% vs baseline)")
        return 1
    print(f"VERDICT: OK ✅ (throughput within / above baseline -{a.threshold:.0f}%)")
    return 0


# --------------------------- collect --------------------------------------
def _iter_results(results_dir):
    """Yield (path, dict) for schema result JSONs (skip raw *.bench.json)."""
    for root, _, files in os.walk(results_dir):
        for fn in files:
            if not fn.endswith(".json") or fn.endswith(".bench.json"):
                continue
            p = os.path.join(root, fn)
            try:
                d = json.load(open(p))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(d, dict) and "metrics" in d and "hw" in d:
                yield p, d


def cmd_collect(a):
    rows = []
    for _, d in _iter_results(a.results_dir):
        if a.hw and d.get("hw") != a.hw:
            continue
        m = d.get("metrics", {})
        # delta vs baseline (throughput), if a baseline exists
        delta = None
        bp = next((c for c in _baseline_candidates(a.baselines_dir, d) if os.path.exists(c)), None)
        if bp:
            bt = json.load(open(bp))["metrics"].get("total_token_throughput")
            ct = m.get("total_token_throughput")
            if bt and ct:
                delta = (ct - bt) / bt * 100.0
        rows.append((d.get("ts", ""), d.get("hw"), d.get("cluster", "?"), d.get("host", "?"),
                     d.get("profile"), f'{d.get("isl")}/{d.get("osl")}',
                     "T" if d.get("tuning") else "U", d.get("conc"),
                     m.get("total_token_throughput"), m.get("median_tpot_ms"),
                     d.get("gpu", {}).get("tokens_per_kw"), delta))
    rows.sort(key=lambda r: (r[1] or "", r[2] or "", r[3] or "", r[6], r[7] or 0, r[0]))
    print(f"{'hw':<6}{'cluster':<10}{'host':<22}{'profile':<12}{'seq':>8}{'t':>2}"
          f"{'conc':>6}{'tot_tok/s':>11}{'mTPOT':>8}{'tok/kW':>8}{'Δ%base':>9}")
    for r in rows:
        d = f"{r[11]:+.1f}" if r[11] is not None else "-"
        kw = f"{r[10]:.0f}" if r[10] else "-"
        print(f"{(r[1] or '?'):<6}{(r[2] or '?'):<10}{(r[3] or '?'):<22}{(r[4] or '?'):<12}"
              f"{r[5]:>8}{r[6]:>2}{(r[7] or 0):>6}{(r[8] or 0):>11.0f}{(r[9] or 0):>8.1f}{kw:>8}{d:>9}")
    print(f"\n{len(rows)} result(s) under {a.results_dir}")
    return 0


# --------------------------- argparse --------------------------------------
def main():
    p = argparse.ArgumentParser(description="together_runner result tooling")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit", help="raw bench output -> result JSON")
    e.add_argument("--raw", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--gpu-csv", default="")
    for k in ("hw", "cluster", "model", "framework", "precision", "profile", "image", "host"):
        e.add_argument(f"--{k}", required=(k in ("hw", "model")), default="")
    for k in ("isl", "osl", "tp", "ep", "conc", "tuning"):
        e.add_argument(f"--{k}", type=int, default=0)
    e.set_defaults(func=cmd_emit)

    c = sub.add_parser("compare", help="result JSON vs baseline")
    c.add_argument("--result", required=True)
    c.add_argument("--baseline", default="", help="explicit baseline path (else auto-resolved)")
    c.add_argument("--baselines-dir",
                   default=os.path.join(os.path.dirname(__file__), "baselines"))
    c.add_argument("--threshold", type=float, default=5.0, help="regression %% (default 5)")
    c.set_defaults(func=cmd_compare)

    co = sub.add_parser("collect", help="tabulate all results across the fleet")
    co.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    co.add_argument("--baselines-dir", default=os.path.join(os.path.dirname(__file__), "baselines"))
    co.add_argument("--hw", default="", help="filter by hardware tag")
    co.set_defaults(func=cmd_collect)

    a = p.parse_args()
    sys.exit(a.func(a) or 0)


if __name__ == "__main__":
    main()
