#!/usr/bin/env python3
"""Promote measured results into committed golden baselines.

Baselines are the regression reference for `compare.py compare` (>5% throughput
drop = non-zero exit). They are keyed by
    baselines/<hw>/[<cluster>/]<framework>/<profile>/<seqtag>/<tuned|untuned>/
        tp<N>_conc<M>.json
and stored in the nested schema, so results from either runner can be promoted:
flat official-recipe records are normalised by results_lib first.

    python3 promote_baseline.py --hw mi350x --dry-run
    python3 promote_baseline.py --hw mi350x
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from results_lib import iter_results, seqtag  # noqa: E402

TR_ROOT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(TR_ROOT / "results"))
    ap.add_argument("--baselines-dir", default=str(TR_ROOT / "baselines"))
    ap.add_argument("--hw", help="only promote this hardware tag")
    ap.add_argument("--framework", help="only promote this framework")
    ap.add_argument("--cluster-scoped", action="store_true",
                    help="write under <hw>/<cluster>/... instead of hw-wide")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing baseline")
    a = ap.parse_args()

    # Keep the newest result per (hw, cluster, fw, profile, seq, tuning, tp, conc).
    best = {}
    for path, d in iter_results(a.results_dir):
        if a.hw and d.get("hw") != a.hw:
            continue
        if a.framework and d.get("framework") != a.framework:
            continue
        key = (d["hw"], d.get("framework"), d.get("profile"),
               seqtag(d["isl"], d["osl"]),
               "tuned" if d.get("tuning") else "untuned",
               d.get("tp"), d.get("conc"))
        mt = os.path.getmtime(path)
        if key not in best or mt > best[key][0]:
            best[key] = (mt, d)

    if not best:
        print("no matching results"); return 1

    n_new = n_skip = 0
    for (hw, fw, profile, seq, tune, tp, conc), (_, d) in sorted(best.items(),
                                                                key=lambda kv: (kv[0][5] or 0, kv[0][6] or 0)):
        parts = [a.baselines_dir, hw]
        if a.cluster_scoped and d.get("cluster"):
            parts.append(d["cluster"])
        parts += [fw, profile, seq, tune, f"tp{tp}_conc{conc}.json"]
        dst = Path(os.path.join(*parts))
        if dst.exists() and not a.force:
            print(f"  skip (exists)  {dst.relative_to(a.baselines_dir)}")
            n_skip += 1
            continue
        print(f"  {'would write' if a.dry_run else 'write'}    "
              f"{dst.relative_to(a.baselines_dir)}  "
              f"({d['metrics'].get('total_token_throughput', 0):,.0f} tok/s)")
        if not a.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(d, indent=2) + "\n")
        n_new += 1

    print(f"\n{n_new} baseline(s) {'to write' if a.dry_run else 'written'}, {n_skip} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
