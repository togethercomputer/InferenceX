#!/usr/bin/env python3
"""Loading and normalising together_runner benchmark results.

Two result schemas coexist:

  nested  {hw, cluster, host, metrics{...}, gpu{...}}
          written by local/run_3_test_client.sh; this is what baselines/ hold.
  flat    the utils/process_result.py record -- no "metrics" key, times in
          seconds rather than milliseconds, throughput per GPU rather than
          per cluster. Written by the official-recipe runner.

Everything here speaks the nested schema; flat records are converted on read by
adapt_flat(). Keeping that in one module means compare.py, report.py and
promote_baseline.py cannot drift on how a result is interpreted, and none of
them has to reach into another's privates.
"""
import json
import os

# flat InferenceX key -> nested metrics key
_FLAT_METRIC_KEYS = {
    "total_token_throughput": "total_token_throughput",
    "output_throughput": "output_token_throughput",
    "request_throughput": "request_throughput",
    "median_ttft": "median_ttft_ms",
    "p99_ttft": "p99_ttft_ms",
    "median_tpot": "median_tpot_ms",
    "p99_tpot": "p99_tpot_ms",
    "median_itl": "median_itl_ms",
    "median_e2el": "median_e2el_ms",
    "p99_e2el": "p99_e2el_ms",
}


def seqtag(isl, osl):
    """1024/1024 -> '1k1k'; falls back to the raw numbers."""
    def one(n):
        n = int(n)
        return f"{n // 1024}k" if n % 1024 == 0 else str(n)
    return f"{one(isl)}{one(osl)}"


def is_nested(d):
    return isinstance(d, dict) and "metrics" in d and "hw" in d


def is_flat_inferencex(d):
    """A utils/process_result.py record: flat, has tput_per_gpu, no metrics."""
    return (isinstance(d, dict) and "metrics" not in d
            and "hw" in d and "tput_per_gpu" in d)


def adapt_flat(d):
    """Normalise a flat InferenceX record into the nested schema.

    process_result.py stores seconds (it divides every *_ms field by 1000) and
    per-GPU throughput; the nested schema wants milliseconds and cluster
    totals, so scale on the way in.
    """
    tp = int(d.get("tp") or 1)
    metrics = {}
    for flat, nested in _FLAT_METRIC_KEYS.items():
        v = d.get(flat)
        if v is None:
            continue
        metrics[nested] = v * 1000.0 if nested.endswith("_ms") else v
    if "tput_per_gpu" in d:
        metrics["total_token_throughput"] = d["tput_per_gpu"] * tp
    if "output_tput_per_gpu" in d:
        metrics["output_token_throughput"] = d["output_tput_per_gpu"] * tp

    gpu = {}
    per_gpu = d.get("avg_power_w")
    if per_gpu:
        gpu = {
            "n_gpus": tp,
            "mean_power_per_gpu_w": round(per_gpu, 1),
            "total_avg_power_w": round(per_gpu * tp, 1),
            "tokens_per_kw": round(
                metrics.get("total_token_throughput", 0) / (per_gpu * tp) * 1000.0, 1),
        }
    return {
        "hw": d.get("hw"),
        "cluster": d.get("cluster", os.environ.get("CLUSTER", "local")),
        "host": d.get("host", ""),
        "ts": d.get("ts", ""),
        "model": d.get("model"),
        "framework": d.get("framework"),
        "precision": d.get("precision"),
        "isl": d.get("isl"), "osl": d.get("osl"),
        "tp": tp, "ep": d.get("ep"), "conc": d.get("conc"),
        # The recipe runner has no PROFILE; synthesise the baseline key from the
        # model prefix + precision so it lines up with baselines/<hw>/<fw>/<profile>/.
        "profile": d.get("profile") or
                   f'{d.get("infmax_model_prefix", "?")}-{d.get("precision", "?")}',
        "image": d.get("image"),
        # tuning is 0/1 in this schema, not a word. The official recipes expose
        # no autotune switch, so recipe runs are recorded untuned.
        "tuning": int(d.get("tuning", 0) or 0),
        "source": "recipe",
        "metrics": metrics,
        "gpu": gpu,
    }


def normalise(d):
    """Return the nested form of a result, or None if it is not one."""
    if is_nested(d):
        return d
    if is_flat_inferencex(d):
        return adapt_flat(d)
    return None


def iter_results(results_dir):
    """Yield (path, nested-dict) for every result under results_dir.

    Skips raw bench output (*.bench.json) and, for the recipe runner, the
    per-config raw file that sits beside each agg_*.json.
    """
    for root, _, files in os.walk(results_dir):
        for fn in sorted(files):
            if not fn.endswith(".json") or fn.endswith(".bench.json"):
                continue
            path = os.path.join(root, fn)
            try:
                d = json.load(open(path))
            except (json.JSONDecodeError, OSError):
                continue
            if is_flat_inferencex(d) and not fn.startswith("agg_"):
                continue  # raw sibling of an agg_ file
            nd = normalise(d)
            if nd is not None:
                yield path, nd


def config_key(d):
    """Identity of a benchmark point, for dedup and baseline lookup."""
    return (d.get("hw"), d.get("cluster"), d.get("framework"), d.get("profile"),
            seqtag(d["isl"], d["osl"]), int(d.get("tuning", 0) or 0),
            d.get("tp"), d.get("conc"))


def latest_per_config(results_dir, **filters):
    """Newest result per config_key, optionally filtered on top-level fields."""
    best = {}
    for path, d in iter_results(results_dir):
        if any(v is not None and d.get(k) != v for k, v in filters.items()):
            continue
        mtime = os.path.getmtime(path)
        key = config_key(d)
        if key not in best or mtime > best[key][0]:
            best[key] = (mtime, d)
    return [d for _, d in sorted(best.values(),
                                 key=lambda md: (md[1].get("tp") or 0,
                                                 md[1].get("conc") or 0))]
