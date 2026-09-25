# together_runner/local/recipes — official InferenceX recipes, locally

Runs the **official InferenceX single-node recipes** on a local box, NVIDIA or
AMD. Deliberately re-implements nothing: it executes `benchmarks/single_node/**`
unmodified and feeds them the same env vars
`.github/workflows/benchmark-tmpl.yml` does, so numbers are comparable to
published InferenceX results rather than to a local re-invention.

Sibling of `../` (the bespoke-profile harness). GPU plumbing comes from
`../../lib/vendor.sh`; both front ends share `../../baselines/` and
`../../results/`.

## Why a new launcher

Every launcher in `runners/` assumes a scheduler — AMD's uses `salloc`/`srun`
(Slurm), `enroot import` + squashfs and NFS paths like `/it-share`; the NVIDIA
DGXC ones are similar. Our boxes have plain Docker. `launch_local.sh` provides
the same contract (set env → run recipe in the pinned image → leave
`$RESULT_FILENAME.json`) over `docker run`, on either vendor.

## MI350X vs MI355X

The master config and recipes are keyed on `mi355x`. MI350X is the **same
gfx950/CDNA4 silicon with the same 288 GB HBM3E**, so the recipes run
unmodified and the sweep is generated with `--runner-type mi355x`. Results are
recorded as `mi350x`.

They differ in **cooling and power envelope**: MI350X is air-cooled at 1000 W,
MI355X liquid-cooled at 1400 W. Expect MI350X to be at or below MI355X on
throughput, and **do not compare tokens/MW figures directly** — clocks are
allowed to boost further on MI355X.

## Layout

```
launch_local.sh   # docker equivalent of runners/launch_*.sh, both vendors
sweep.py          # generates the official matrix, loops it, aggregates
../../results/<hw>/<prefix>-<framework>-<UTC stamp>/   # gitignored
    agg_<RESULT_FILENAME>.json   # InferenceX schema (utils/process_result.py)
    <RESULT_FILENAME>.json       # raw bench_serving output
    <RESULT_FILENAME>.server.log
    <RESULT_FILENAME>.gpu_metrics.csv
    summary.json
```

## Requirements

`sweep.py` shells out to `utils/matrix_logic/generate_sweep_configs.py`, which
imports `pydantic` and `pyyaml`. It looks for an interpreter that has them
(current one, `<repo>/.venv`, `~/.venv-inferencex`) and prints how to fix it if
none do. Override with `GEN_PYTHON=/path/to/python`.

```bash
python3 -m venv .venv && .venv/bin/pip install pydantic pyyaml
```

## Usage

```bash
cd together_runner/local/recipes

# What CI would run for this config (no GPUs touched):
python3 sweep.py --model-prefix gptoss --framework vllm \
    --seq-lens 1k1k --dry-run

# One point, on one GPU:
python3 sweep.py --model-prefix gptoss --framework vllm \
    --seq-lens 1k1k --tp 1 --conc 4 --gpus 0

# The full official 1k1k matrix (TP1 conc 4-128, TP4 4-8, TP8 4-16):
python3 sweep.py --model-prefix gptoss --framework vllm --seq-lens 1k1k
```

`--tp` / `--conc` filter the generated matrix; `--gpus` sets
`ROCR_VISIBLE_DEVICES` (default: the first `$TP` GPUs). Default port is **8710**
— 8888 is already taken by another tenant on this box.

### Serving pre-staged weights (`--model-path`)

The recipes run `hf download "$MODEL"`, which fetches the **whole** repo. For
`openai/gpt-oss-120b` that is 196 GB, of which vLLM uses 65 GB — `metal/`
(Apple-silicon build) and `original/` (pre-MXFP4 checkpoint) are dead weight,
and on this node's link the difference is ~1.5 h versus ~5.5 h.

The recipes skip the download when `$MODEL` is an absolute path, so pass the
cached snapshot directly:

```bash
python3 sweep.py --model-prefix gptoss --framework vllm --seq-lens 1k1k \
    --model-path /hf-cache/models--openai--gpt-oss-120b/snapshots/<commit>
```

Point it at the snapshot **inside** the `/hf-cache` mount, not at a copy or a
symlink outside it: HF snapshot dirs are trees of relative symlinks into
`../../blobs/`, which only resolve if the whole repo dir is mounted.

`HF_HUB_OFFLINE=1` is **not** a substitute — it raises `IncompleteSnapshotError`
when any repo file is absent from the cache, including the ones we skipped on
purpose.

The driver restores the canonical HF id in the `model` field of the aggregated
JSON (`bench_serving` records whatever was passed to `--model`) and keeps the
path under `model_path`.

## Results — gpt-oss-120b, 1k1k (2026-09-25)

Full official matrix for `gptoss-fp4-mi355x-vllm`, 11/11 configs, run on
MI350X. Regenerate with `python3 ../../analysis/report.py`.

**gptoss · fp4 · vllm** (openai/gpt-oss-120b) — mi350x, 1k1k, image `vllm/vllm-openai-rocm:v0.22.0`

| TP | conc | tput/GPU (tok/s) | output/GPU | median TTFT (ms) | p99 TTFT (ms) | median TPOT (ms) | avg W/GPU | tok/s per MW |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 1,699 | 846 | 45 | 103 | 4.54 | 692 | 2,455,690 |
| 1 | 8 | 2,842 | 1,426 | 45 | 144 | 5.45 | 738 | 3,851,436 |
| 1 | 16 | 4,644 | 2,310 | 50 | 244 | 6.73 | 808 | 5,745,187 |
| 1 | 32 | 7,344 | 3,678 | 55 | 2,822 | 8.20 | 887 | 8,283,999 |
| 1 | 64 | 11,446 | 5,722 | 64 | 688 | 10.83 | 942 | 12,156,465 |
| 1 | 128 | 16,815 | 8,398 | 86 | 1,226 | 14.92 | 990 | 16,976,449 |
| 4 | 4 | 582 | 289 | 31 | 75 | 3.32 | 489 | 1,189,455 |
| 4 | 8 | 966 | 485 | 33 | 123 | 4.02 | 532 | 1,816,113 |
| 8 | 4 | 278 | 138 | 28 | 561 | 3.42 | 408 | 681,173 |
| 8 | 8 | 543 | 273 | 31 | 113 | 3.55 | 426 | 1,274,392 |
| 8 | 16 | 1,071 | 532 | 32 | 217 | 3.62 | 445 | 2,404,935 |

**Reading these.** TP1 is the headline: throughput/GPU scales near-linearly to
concurrency 32 and keeps climbing sublinearly to 16.8k tok/s/GPU at conc 128,
while median TPOT stays under 15 ms. Per-GPU throughput *falls* under tensor
parallelism (TP8 conc 4 is 278 tok/s/GPU) because gpt-oss-120b is ~65 GB and
fits one MI350X's 288 GB — splitting it only adds collective traffic and
starves each GPU. That is why the official matrix caps TP4 at conc 8 and TP8 at
conc 16 instead of sweeping them to 128; the interesting TP>1 numbers are
latency, not throughput (TP8 holds the lowest median TTFT, 28-32 ms).

**Caveats.**
- MI350X is air-cooled at 1000 W; MI355X is liquid-cooled at 1400 W. Expect
  these to sit at or below published MI355X numbers, and treat the tok/MW
  column as indicative — clocks boost further on MI355X.
- Power is sampled from `amd-smi` over the whole run including startup, so
  `avg_power_w` is a floor. It is correctly scoped to the GPUs under test only
  because device-node selection hides the others from `amd-smi`.
- p99 TTFT at TP1/conc32 (2.8 s) is an outlier against its neighbours and looks
  like a scheduling artefact rather than a real cliff; re-run before quoting it.
- Weights served from a local snapshot path, so `model_path` is recorded
  alongside the canonical `model` id.

## Node-specific notes

- **Shared box.** ~23 home dirs and a dozen long-lived containers. Check
  `rocm-smi --showmeminfo vram` before taking all 8 GPUs; pick free ones with
  `--gpus`. GPU 0 carries stale KFD entries from other users' crashed MPI runs
  (0 VRAM / 0 CU) — harmless.
- **ROCm container flags.** `--device=/dev/kfd --device=/dev/dri` plus the
  `render` and `video` groups. `--gpus all` does nothing on ROCm, and without
  the `render` group `rocm-smi` still lists all 8 GPUs while HIP reports none.
- **Weights** live in `/mnt/data/johnson/hf-cache` (`/mnt/data` is the writable
  array; `/mnt/data2` is not writable by this user, and the root disk is tight).
- **Egress is ~8-11 MB/s sustained**, authenticated or not (brief faster bursts
  at the start of a transfer are not representative). Pre-stage weights and
  images before a session; a 65 GB model is ~1.5 h. `HF_HUB_DISABLE_XET=1` is
  required — the Xet backend stalls at 0 bytes on this network, leaving every
  large shard at exactly 0 bytes, while plain HTTP works.
- **Benchmark output never touches the git checkout**: `/workspace` is a scratch
  dir on `/mnt/data` with `utils/` and `benchmarks/` bind-mounted read-only
  beneath it.
