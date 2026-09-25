#!/usr/bin/env python3
"""Tabulate measured results as Markdown (and optionally CSV).

Reads every result under results/ through results_lib, so both schemas -- the
bespoke-profile runner's and the official-recipe runner's -- appear in one
table, on either vendor. A sweep is usually split across several timestamped
directories (different TP values, reruns after a failure); results are keyed on
their embedded fields, not their path, so the split does not matter.

    python3 report.py                      # every result, grouped
    python3 report.py --hw mi350x          # one hardware tag
    python3 report.py --csv out.csv        # also write CSV
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_lib import latest_per_config, seqtag  # noqa: E402

TR_ROOT = Path(__file__).resolve().parent.parent

COLUMNS = [
    ("TP", lambda d: f'{d["tp"]}'),
    ("conc", lambda d: f'{d["conc"]}'),
    ("tput/GPU (tok/s)", lambda d: _fmt(_per_gpu(d, "total_token_throughput"), ",.0f")),
    ("output/GPU", lambda d: _fmt(_per_gpu(d, "output_token_throughput"), ",.0f")),
    ("median TTFT (ms)", lambda d: _fmt(d["metrics"].get("median_ttft_ms"), ",.0f")),
    ("p99 TTFT (ms)", lambda d: _fmt(d["metrics"].get("p99_ttft_ms"), ",.0f")),
    ("median TPOT (ms)", lambda d: _fmt(d["metrics"].get("median_tpot_ms"), ".2f")),
    ("W/GPU", lambda d: _fmt(d.get("gpu", {}).get("mean_power_per_gpu_w"), ",.0f")),
    ("tok/kW", lambda d: _fmt(d.get("gpu", {}).get("tokens_per_kw"), ",.0f")),
]


def _fmt(v, spec):
    """Format a metric, or '-' when it is missing.

    Every cell goes through this: a bare conditional around a multi-part
    f-string silently replaces the WHOLE row, not the one absent field.
    """
    return format(v, spec) if isinstance(v, (int, float)) else "-"


def _per_gpu(d, key):
    """Metrics are cluster totals; the table reports per-GPU."""
    v = d["metrics"].get(key)
    tp = d.get("tp") or 1
    return v / tp if isinstance(v, (int, float)) else None


def group_key(d):
    return (d.get("hw"), d.get("framework"), d.get("profile"),
            d.get("model"), seqtag(d["isl"], d["osl"]), d.get("image"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(TR_ROOT / "results"))
    ap.add_argument("--hw", help="only this hardware tag")
    ap.add_argument("--framework", help="only this framework")
    ap.add_argument("--csv")
    args = ap.parse_args()

    rows = latest_per_config(args.results, hw=args.hw, framework=args.framework)
    if not rows:
        print(f"no results found under {args.results}")
        return 1

    groups = {}
    for d in rows:
        groups.setdefault(group_key(d), []).append(d)

    for (hw, fw, profile, model, seq, image), items in sorted(
            groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        print(f"**{profile} · {fw}** ({model}) — {hw}, {seq}, image `{image}`\n")
        print("| " + " | ".join(c[0] for c in COLUMNS) + " |")
        print("|" + "|".join("---:" for _ in COLUMNS) + "|")
        for d in sorted(items, key=lambda x: (x.get("tp") or 0, x.get("conc") or 0)):
            print("| " + " | ".join(fn(d) for _, fn in COLUMNS) + " |")
        print()

    if args.csv:
        keys = ["hw", "cluster", "host", "ts", "framework", "precision", "model",
                "isl", "osl", "tp", "ep", "conc"]
        metric_keys = ["total_token_throughput", "output_token_throughput",
                       "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms",
                       "median_e2el_ms"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys + metric_keys
                               + ["mean_power_per_gpu_w", "tokens_per_kw"])
            w.writeheader()
            for d in rows:
                r = {k: d.get(k) for k in keys}
                r.update({k: d["metrics"].get(k) for k in metric_keys})
                r["mean_power_per_gpu_w"] = d.get("gpu", {}).get("mean_power_per_gpu_w")
                r["tokens_per_kw"] = d.get("gpu", {}).get("tokens_per_kw")
                w.writerow(r)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
