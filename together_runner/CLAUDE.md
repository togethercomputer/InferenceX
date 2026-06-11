# together_runner — Claude Code memory

Single-node (this 8×B200) SGLang benchmark harness for InferenceX. Standalone
helper scripts; does NOT touch the CI sweep. Built on patterns from
`../benchmarks/benchmark_lib.sh` and `../runners/launch_b200-dgxc.sh`.

## What it does (3 pillars)
1. **Two-tier smoke** — `run_sglang_0_preflight.sh` (gate-1: env/machine, seconds)
   + gate-2 minimal real completion inside `--full`.
2. **Weight externalization** — `prestage_weights.sh` downloads once to
   `$MODELS_ROOT/$PROFILE` (+`.ready` marker); launches then do ZERO download.
3. **Staged monitoring + baseline/delta** — `sglang_lib.sh` maps server-log
   markers to named stages with per-stage + next-stage + to-ready ETA;
   `compare.py` diffs results vs committed baselines (throughput ±5%).

## Run it
```bash
cd together_runner
hf auth login                 # or export HF_TOKEN=...   (only needed for download)
bash run_all.sh --smoke       # gate-1 preflight only (seconds)
bash run_all.sh --full        # preflight→prestage→start→launch→gate-2→bench→compare
bash run_all.sh --baseline    # --full with ENABLE_TUNING=1, saves+ a baseline
bash record_baselines.sh      # one warm server, sweep CONC_LIST="16 64 128"
python3 compare.py collect     # tabulate ALL results across the fleet (+Δ vs baseline)
```
All config in `config.env` (sourced by every script; override by exporting first).
Each step also runs standalone (`bash run_sglang_2_launch_server.sh`).

## Key config (config.env)
- `PROFILE` = `dsr1-fp4` (TP8/EP8, default) or `smoke` (Llama-8B TP1).
- **`ENABLE_TUNING`** default **0** → adds `--disable-flashinfer-autotune` (fast,
  ~10min cold start). `=1` keeps autotune (~24min, peak perf) — use for baselines.
- Paths on the 14TB array: `HF_CACHE`, `MODELS_ROOT`, `FLASHINFER_CACHE` under
  `/scratch/home/johnson/`. NEVER use the 34GB root disk for weights.
- `CONTAINER_NAME=sglang_run`, `PORT=8888`, `IMAGE=lmsysorg/sglang:dev-cu13`.
- `HW` + `CLUSTER` tag each run (same HW may live on several clusters; set CLUSTER per box).

## Directory layout (scales across hw × cluster × host × date)
- **results/** (gitignored, raw, provenance-first):
  `results/<hw>/<cluster>/<host>/<date>/<profile>_<seqtag>_<tuned|untuned>_conc<N>.json`
  + companion `*.bench.json` (raw) and `*.gpu.csv`.
- **baselines/** (committed golden): `baselines/<hw>/<profile>/<seqtag>/<tuned|untuned>/conc<N>.json`,
  optional per-cluster override at `baselines/<hw>/<cluster>/...` (compare prefers
  cluster-specific, falls back to hw-wide). `record_baselines.sh` saves hw-wide by
  default; `BASELINE_CLUSTER=1` scopes under the cluster.
- Query any axis with `compare.py collect` (reads embedded fields, not the path).

## Baselines & metrics
- Path key includes tuning (see Directory layout above).
  **Keep ENABLE_TUNING consistent** between reference and test machine — autotune
  alone moves throughput ~4–10% and would false-flag a regression otherwise.
- Primary pass/fail metric: `total_token_throughput`; regression if it drops
  **>5%** vs baseline (`compare.py compare` → non-zero exit). Result JSON keys
  mirror `../utils/process_result.py` (reuse `collect_results.py`/`summarize.py`).
- Reference baselines recorded on THIS node (`gpu-dp-96sjj-gkwph`), tuned, 1k1k:

  | conc | total tok/s | out tok/s | medTPOT | W/gpu | tok/kW |
  |---:|---:|---:|---:|---:|---:|
  | 16 | 2,152 | 1,068 | 13.3ms | 425 | 314 |
  | 64 | 5,482 | 2,712 | 20.2ms | 509 | 666 |
  | 128 | 9,670 | 4,780 | 24.5ms | 558 | 1,071 |
  | 256 | 13,490 | 6,727 | 35.3ms | 546 | 1,540 |

  Throughput scales ~6.3× over 16× concurrency; efficiency (tok/kW) keeps rising
  to conc256 (the `--max-running-requests` ceiling) at only 546W/1000W → still
  bandwidth/compute-bound, not power-bound.

## Startup stages (dsr1-fp4 cold start, this node)
engine-init (~1m) → weight-load (~1m from local) → [autotune ~12m only if
ENABLE_TUNING=1] → graph-capture (~2m) → ready. Tuned ≈15m, untuned ≈4–5m.
Live progress: `docker exec sglang_run tail -f /root/server.log`.

## Gotchas
- **docker-proxy holds host PORT for the container's whole life** — preflight's
  port check is container-aware (reusing `sglang_run` is OK, not a conflict).
- **Reusing the container to relaunch** a fresh server: `pkill -f
  sglang.launch_server` inside it first (record_baselines.sh does this).
- autotune-result persistence across restarts is unsolved; `ENABLE_TUNING=0` is
  the fast lever. FlashInfer JIT cache IS mounted (`FLASHINFER_CACHE`).
- See `../KLAUD_DEBUG.md` for B200/B300 OOM / DeepGemm / timeout failure modes.
- Teardown: `docker exec sglang_run pkill -f sglang.launch_server` (server) /
  `docker rm -f sglang_run` (container).
- Commit identity for this repo: `Johnsonms <lizhaofu@gmail.com>`, no Claude trailer.
```
