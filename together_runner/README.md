# together_runner

Together's own InferenceX benchmark harnesses, for **NVIDIA and AMD**, on boxes
we control. Nothing here touches the CI sweep (`.github/workflows/run-sweep.yml`).

The upstream launchers in `../runners/` all assume a scheduler — Slurm
`salloc`/`srun` plus `enroot` squashfs imports, and site paths like `/it-share`.
Our boxes have plain Docker, so these run the same benchmarks without one.

## Layout

```
lib/          common.sh    logging, env validation, timers, preflight helpers
              vendor.sh    ALL NVIDIA-vs-AMD differences live here
              monitor.sh   staged startup detection + ETA

local/        single-node, both vendors
              config.env run_0..3 run_all.sh prestage_weights.sh
                           -> bespoke recipes per (ENGINE, PROFILE) + baseline gate
              recipes/     -> the OFFICIAL benchmarks/single_node/** recipes

multinode/    slurm-disagg/   2-node SGLang prefill/decode disaggregation

analysis/     compare.py            result <-> baseline diff, fleet collect
              report.py             sweep results -> markdown/CSV table
              promote_baseline.py   results -> committed golden baselines

baselines/<hw>/[<cluster>/]<framework>/<profile>/<seqtag>/<tuned|untuned>/
                  tp<N>_conc<M>.json      committed golden
results/<hw>/<prefix>-<framework>-<stamp>/   gitignored per-run output
```

`baselines/` and `results/` are keyed by `<hw>` (`b200`, `mi350x`, ...), so both
vendors live side by side with no schema change.

## Which front end?

| | runs | gives you |
|---|---|---|
| `local/run_all.sh` | bespoke recipes per (ENGINE, PROFILE) | staged startup ETA, weight prestaging, regression gate vs committed baselines |
| `local/recipes/sweep.py` | the official `benchmarks/single_node/**` recipes, unmodified, from the generated CI matrix | numbers directly comparable to published InferenceX results |

Both write the same `baselines/` and `results/` trees and are read by the same
`analysis/` tools.

## Quick start

```bash
# Official matrix, whatever GPU this box has (hw auto-detected):
python3 local/recipes/sweep.py --model-prefix gptoss --framework vllm --dry-run

# AMD MI350X using the mi355x recipes (same gfx950 silicon):
python3 local/recipes/sweep.py --model-prefix gptoss --framework vllm \
    --recipe-runner mi355x --hw mi350x --tp 1 --conc 4 8 16

# NVIDIA B200:
python3 local/recipes/sweep.py --model-prefix gptoss --framework vllm \
    --recipe-runner b200 --hw b200

# Bespoke-profile harness:
bash local/run_all.sh --smoke | --full | --baseline

# Analysis:
python3 analysis/report.py                      # table of everything measured
python3 analysis/compare.py collect             # fleet view + Δ vs baseline
python3 analysis/compare.py compare --result <agg.json>   # regression gate
python3 analysis/promote_baseline.py --hw mi350x --dry-run
```

## Vendor support

Everything vendor-specific is behind `lib/vendor.sh`:

| | NVIDIA | AMD / ROCm |
|---|---|---|
| detect | `nvidia-smi -L` | `/dev/kfd` + `rocm-smi` |
| expose GPUs to docker | `--gpus all` / `--gpus device=0,1` | `--device=/dev/kfd` + one `--device=/dev/dri/renderD*` per GPU, plus `render`/`video` groups |
| power sampling | `nvidia-smi --query-gpu=... -l N` | `amd-smi metric -w N --csv` (preamble + repeated headers filtered) |
| master config | `nvidia-master.yaml` | `amd-master.yaml` |

Two AMD traps worth knowing, both handled in `vendor.sh`:

- **Do not set `ROCR_VISIBLE_DEVICES` to pick GPUs.** The recipes mirror it into
  `HIP_VISIBLE_DEVICES`, but ROCr filters *first* — so after
  `ROCR_VISIBLE_DEVICES=N` the container holds one device numbered 0, and
  mirroring `N` selects a device that no longer exists. The engine then dies
  with `No HIP GPUs are available` for every `N` except 0. Selecting by device
  node avoids it, and additionally scopes `amd-smi` to our GPUs so power
  sampling excludes other tenants.
- **`ROCR_VISIBLE_DEVICES` indices are not `rocm-smi` indices.** `vendor.sh`
  maps them by sorting primary render nodes by PCI address, which reproduces
  `rocm-smi --showbus` ordering.

## Result schemas

Two coexist, and `analysis/compare.py` normalises between them:

- **nested** — `{hw, cluster, host, metrics{...}, gpu{...}}`, written by
  `local/run_3_test_client.sh`; this is what `baselines/` hold.
- **flat** — the `utils/process_result.py` record (no `metrics` key), written by
  the official-recipe runner.

`compare.adapt_flat()` converts flat -> nested on read (seconds -> ms, per-GPU
-> cluster totals), so recipe runs land in the same tables and the same
regression gate. The committed baselines were not rewritten.

Baseline leaf names are `tp<N>_conc<M>.json`. TP is part of the key because an
official sweep varies it — without TP, TP1/TP4/TP8 at the same concurrency all
collide. Older baselines written as `conc<M>.json` still resolve via a fallback.
