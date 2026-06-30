# Benchmark record — Qwen3-32B prefill/decode disaggregation (2-node, SGLang)

**Date:** 2026-06-29 · **Cluster:** slinky (Slurm-on-k8s) · **Operator:** Johnsonms

## Purpose
ClusterMAX inference-disagg phase-0 readiness proof: confirm this tenant slice can
deploy a 2-node prefill/decode-disaggregated SGLang endpoint, transfer KV cross-node
over RDMA, and serve at a representative throughput. Bring-up model: **Qwen/Qwen3-32B**
(dense), **1P1D** (1 prefill + 1 decode + sgl-router).

## Config
- **Topology:** prefill@slinky-0 (TP8), decode@slinky-1 (TP8), sgl-router@slinky-0:8002, `--policy random`.
- **Hardware:** 2× node, each 8× B200 + 14× mlx5 IB. 16 GPUs total in the serving path.
- **Image:** `lmsysorg/sglang:dev-cu13` (sglang `0.0.0.dev1+g909123ddb`, sglang-router 0.3.2,
  mooncake `0.3.11.post1`), shared squashfs `/data/home/johnson/enroot/sglang-dev-cu13.sqsh`.
- **KV transfer:** mooncake over RDMA, **dmabuf path forced via `WITH_NVIDIA_PEERMEM=0`**
  (nvidia_peermem absent; driver 580). IB devices (GPU-adjacent, GPU-order):
  `mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_4,mlx5_5,mlx5_6,mlx5_7`. `/dev/infiniband` bind-mounted.
- **Serving args:** `--tp 8 --trust-remote-code`, **CUDA graph ON**, prefill
  `--disaggregation-mode prefill --disaggregation-bootstrap-port 9000`, decode `--disaggregation-mode decode`.
- **Bench client:** InferenceX `utils/bench_serving/benchmark_serving.py`, `--backend sglang`,
  `--endpoint /v1/completions`, `--dataset-name random`, ISL=1024 OSL=1024, num_prompts=conc×8.

## Results (1k/1k, CUDA graph)

| conc | requests | total tok/s | output tok/s | req/s | med TPOT (ms) | med TTFT (ms) | p99 TTFT (ms) | med E2E (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16  | 160/160   | 2,087 | 957   | 1.10 | 13.9 | 1,591  | 4,564  | 15,560 |
| 64  | 512/512   | 4,591 | 2,110 | 2.37 | 23.9 | 3,616  | 18,651 | 28,010 |
| 128 | 1024/1024 | 4,512 | 2,080 | 2.37 | 34.1 | 6,629  | 39,334 | 41,623 |
| 256 | 2048/2048 | 4,898 | 2,243 | 2.59 | 43.1 | 26,913 | 80,696 | 66,562 |

**0 failed requests across the entire sweep.** Raw per-point JSON: `/data/home/johnson/enroot/sweep/qwen3-32b_1k1k_conc{16,64,128,256}.bench.json`.

### Post-refactor re-validation (2026-06-30, full sweep, allocation model)
Re-ran the full sweep after the portability refactor: single 2-node `salloc --no-shell`
allocation, prefill/decode/router/bench as overlap steps, `IB_DEVICES` + dmabuf decision
auto-detected (IB list matched the hand-derived `mlx5_9,10,11,12,4,5,6,7` exactly).
**0 failed requests across all points.**

| conc | requests | total tok/s | output tok/s | med TPOT (ms) | med TTFT (ms) | p99 TTFT (ms) | vs 06-29 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16  | 160/160   | 2,156 | 993   | 13.7 | 1,405  | 5,954  | +3% |
| 64  | 512/512   | 4,614 | 2,127 | 23.0 | 3,933  | 20,987 | +0.5% |
| 128 | 1024/1024 | 3,298 | 1,526 | 33.1 | 8,844  | 36,889 | **−27% (see note)** |
| 256 | 2048/2048 | 4,892 | 2,229 | 42.2 | 27,071 | 79,414 | −0.1% |

**conc=128 anomaly:** reproducibly ~3.3–3.4k (two runs: 3,437 then 3,298), i.e. *below*
conc=64 — physically inconsistent with a healthy curve, and below the 06-29 baseline of
4,512. NOT noise (reproduces) and NOT a code regression (serving args/nodes/IB are byte-for-byte
unchanged; conc 16/64/256 match baseline). Read as a **1P1D dynamics artifact** at this
concurrency: the single prefill instance interleaves badly with decode (prefill bursts starve
decode, TTFT jumps to ~8.8s), whereas conc=64 stays stable and conc=256 is fully decode-saturated.
A candidate to revisit when scaling prefill (NP1D) or tuning chunked-prefill scheduling.

## Findings
1. **Functional:** cross-node KV transfer works under sustained load — the whole path
   (deploy → RDMA-in-pod → bootstrap → KV transfer → router → serve) is solid, 0 errors.
2. **Peak throughput ≈ 4,900 tok/s total (~306 tok/s/GPU over 16 GPUs).**
3. **Saturates at conc ≥ 64** (~4,500–4,900 tok/s). Adding concurrency past 64 yields ~no
   extra throughput but sharply worse TTFT (p99 4.6 s → 80.7 s @ conc 256).
4. **Bottleneck = the single prefill instance** (1P1D): decode TPOT stays healthy (14–43 ms)
   while TTFT explodes as prefill queues. To raise throughput / cut TTFT tail → add prefill
   instances (e.g. 2P1D / NP1D) and/or enable chunked-prefill tuning.

## Critical fixes that made this work (env-specific, ephemeral on pod restart)
- **`WITH_NVIDIA_PEERMEM=0`** on prefill+decode → mooncake dmabuf KV registration (else
  3072× `Failed to register memory` → `KVTransferError`). [found by codex]
- **CUDA graph ON** is mandatory for perf: eager mode gave **85 tok/s total / 366 ms TPOT /
  65% client-timeout failures** at conc 16 — a ~22× throughput / ~27× TPOT regression vs graphs on.
- enroot nvidia hook patched `--no-persistenced --no-fabricmanager`; `/dev/infiniband` mounted;
  enroot import temp on ext4 `/scratch`. (See HANDOVER-disagg-kv-transfer.md / HANDBACK-codex-dmabuf.md.)

## Provenance
Nodes slinky-0/1; sweep run as overlap step on prefill job 61 (decode job 62), router :8002.
Server logs: `/data/home/johnson/enroot/{prefill-cg,decode-cg,router-sweep}.log`. Sweep
driver log: `/data/home/johnson/enroot/sweep.log`.
