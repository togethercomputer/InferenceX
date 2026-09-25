#!/usr/bin/env python3
"""Collect every agg_*.json under results/ into one table.

A sweep is often split across several invocations (different TP values, reruns
after a failure), so results live in several timestamped directories. This
reads the JSON fields rather than the paths, so the split does not matter.

    python3 report.py                 # markdown table
    python3 report.py --csv out.csv   # also write CSV
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TR_ROOT = HERE.parent


def load(results_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(results_dir.rglob("agg_*.json")):
        d = json.loads(f.read_text())
        # Later sweeps supersede earlier ones for the same config key.
        rows.append((f.stat().st_mtime, d))
    best: dict[tuple, dict] = {}
    for mtime, d in rows:
        key = (d["hw"], d["framework"], d["infmax_model_prefix"],
               d["isl"], d["osl"], d["tp"], d["ep"], d["conc"])
        if key not in best or mtime > best[key][0]:
            best[key] = (mtime, d)
    return [d for _, d in sorted(best.values(), key=lambda x: (x[1]["tp"], x[1]["conc"]))]


def tok_per_mw(d: dict) -> float | None:
    p = d.get("avg_power_w")
    if not p:
        return None
    # tok/s/GPU divided by W/GPU = tok/J; x1e6 W per MW.
    return d["tput_per_gpu"] / p * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(TR_ROOT / "results"))
    ap.add_argument("--csv")
    args = ap.parse_args()

    rows = load(Path(args.results))
    if not rows:
        print("no results found")
        return

    h = rows[0]
    print(f"**{h['infmax_model_prefix']} · {h['precision']} · {h['framework']}** "
          f"({h['model']}) — {h['hw']}, {h['isl']//1024}k{h['osl']//1024}k, "
          f"image `{h['image']}`\n")
    print("| TP | conc | tput/GPU (tok/s) | output/GPU | median TTFT (ms) | "
          "p99 TTFT (ms) | median TPOT (ms) | avg W/GPU | tok/s per MW |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in rows:
        tm = tok_per_mw(d)
        print(f"| {d['tp']} | {d['conc']} | {d['tput_per_gpu']:,.0f} | "
              f"{d['output_tput_per_gpu']:,.0f} | {d['median_ttft']*1000:,.0f} | "
              f"{d['p99_ttft']*1000:,.0f} | {d['median_tpot']*1000:.2f} | "
              f"{d.get('avg_power_w', float('nan')):,.0f} | "
              f"{tm:,.0f}" if tm else "n/a", end="")
        print(" |")

    if args.csv:
        import csv
        keys = ["hw", "framework", "precision", "model", "isl", "osl", "tp", "ep",
                "conc", "tput_per_gpu", "output_tput_per_gpu", "median_ttft",
                "p99_ttft", "median_tpot", "avg_power_w", "joules_per_output_token"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys + ["tok_per_mw"])
            w.writeheader()
            for d in rows:
                r = {k: d.get(k) for k in keys}
                r["tok_per_mw"] = tok_per_mw(d)
                w.writerow(r)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
