# together_runner — Claude Code memory

Together's InferenceX benchmark harnesses for **NVIDIA and AMD** boxes we
control. Standalone; does NOT touch the CI sweep. The upstream `../runners/`
launchers all need a scheduler (Slurm + enroot); these use plain Docker.

## Layout (reorganised 2026-09-25 — by capability, not by vendor)
```
lib/      common.sh (log/env/timers/preflight) · vendor.sh (ALL NVIDIA-vs-AMD
          differences) · monitor.sh (staged startup + ETA)
local/    single-node both vendors: config.env run_0..3 run_all.sh
          prestage_weights.sh record_baselines.sh   (bespoke profiles)
          recipes/  launch_local.sh + sweep.py      (official recipes)
multinode/slurm-disagg/
analysis/ compare.py · report.py · promote_baseline.py
baselines/<hw>/[<cluster>/]<fw>/<profile>/<seq>/<tuned|untuned>/tp<N>_conc<M>.json
results/<hw>/<prefix>-<fw>-<stamp>/     (gitignored)
```
`<hw>` (b200, mi350x, ...) keys both trees, so vendors coexist unchanged.

## Two front ends, one set of plumbing
- **bespoke** (`local/run_all.sh`): hand-written recipes per (ENGINE, PROFILE);
  gives staged startup ETA, weight prestaging, baseline regression gate.
- **official** (`local/recipes/sweep.py`): runs `benchmarks/single_node/**`
  UNMODIFIED from the generated CI matrix -> comparable to published numbers.
  Auto-routes `amd-master.yaml` vs `nvidia-master.yaml` off `--recipe-runner`.
  `--hw` defaults to the detected GPU; `--recipe-runner` defaults to `--hw`
  (MI350X uses `--recipe-runner mi355x --hw mi350x`, same gfx950 silicon).

## lib/vendor.sh — the whole point
`vendor_detect / vendor_smi / vendor_gpu_count / vendor_gpu_busy_count /
vendor_set_gpu_flags (array) / vendor_power_sample_cmd / vendor_amd_render_nodes`.

**AMD traps encoded there:**
- NEVER select GPUs with `ROCR_VISIBLE_DEVICES`. Recipes mirror it into
  `HIP_VISIBLE_DEVICES`, ROCr filters FIRST -> the container has one device
  numbered 0 -> mirroring N picks a device that does not exist ->
  `No HIP GPUs are available` for every N except 0. Select by **render node**
  (`--device=/dev/dri/renderD*`). Bonus: scopes `amd-smi` to our GPUs.
- ROCR index != rocm-smi index. Map by sorting PRIMARY render nodes
  (skip `amdgpu_xcp_*`) by PCI address.
- `amd-smi -w` emits a `CTRL + C` preamble and repeats headers — filter with the
  same awk as `benchmarks/benchmark_lib.sh:start_gpu_monitor`.
- Engine images ENTRYPOINT to `vllm`/`sglang` — need `--entrypoint bash`
  (= the official launchers' `--no-container-entrypoint`).

## Result schemas (two, bridged)
- **nested** `{hw, cluster, host, metrics{...}, gpu{...}}` — `local/run_3`, and
  what `baselines/` hold. `tuning` is **0/1 int**, not a word.
- **flat** — `utils/process_result.py` record, no `metrics` key; official runner.
`compare.adapt_flat()` converts flat->nested on read (seconds->ms, per-GPU->
cluster totals). Committed b200 baselines were NOT rewritten.
Baseline leaf is `tp<N>_conc<M>.json` — TP must be in the key because an official
sweep varies TP; bare `conc<M>.json` still resolves via fallback.

## Gotchas
- `docker-proxy` holds host PORT for the container's life; only one engine per
  port. Default port here is **8710** (8888 is taken by another tenant).
- Set `VLLM_CACHE_ROOT`/`TRITON_CACHE_DIR` to host dirs or every config
  recompiles cold.
- Shared box: check `vendor_gpu_busy_count` before taking GPUs.
- Commit identity for this repo: `Johnsonms <lizhaofu@gmail.com>`, no Claude trailer.

See `README.md` (structure + vendor matrix) and `local/recipes/README.md`
(weight prestaging, `--model-path`, network traps).
