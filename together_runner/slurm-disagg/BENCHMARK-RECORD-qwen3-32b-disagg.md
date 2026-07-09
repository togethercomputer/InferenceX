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

---

## 2026-07-09 re-benchmark (newer stack) — +79% peak throughput, conc-128 anomaly gone

Re-ran the identical harness (`00_setup -> 01_preflight -> 10_launch -> 20_benchmark`) on the
same 2 nodes, same 1P1D/TP8 topology, same client + ISL/OSL 1024/1024, same sweep
`CONC_LIST="16 64 128 256"`. **Config and serving args unchanged** — only the platform underneath
moved (rolling `dev-cu13` image + GPU driver). **0 failed requests across all points.**

### Environment deltas (config was identical; only the stack changed)
| | 06-29 / 06-30 record | 2026-07-09 |
|---|---|---|
| GPU driver | 580 | **610.43.02** |
| sglang | `0.0.0.dev1+g909123ddb` | `0.0.0.dev1+g0ffed946f` |
| sgl-router | 0.3.2 | 0.3.2 (same) |
| torch / cuda | — | 2.11.0+cu130 / cuda 13.0 |
| attention backend | (unstated) | **`trtllm_mha`** |
| KV path | dmabuf (`WITH_NVIDIA_PEERMEM=0`) | dmabuf (same, auto-decided) |
| IB devices (auto) | `mlx5_9,10,11,12,4,5,6,7` | `mlx5_11,12,5,6,2,3,0,1` (re-detected) |
| image sqsh md5 | — | `3cdb6e0a9dec073c4a30b374ca5f8a02` (23.7 GB) |

### Results (1k/1k, CUDA graph, 07-09)
| conc | requests | total tok/s | output tok/s | med TPOT (ms) | med TTFT (ms) | p99 TTFT (ms) | med E2E (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16  | 160/160   | 2,482 | 1,133 | 12.1 | 335   | 5,261  | 12,703 |
| 64  | 512/512   | 6,070 | 2,807 | 19.5 | 959   | 14,595 | 20,571 |
| 128 | 1024/1024 | 7,786 | 3,594 | 30.9 | 1,758 | 28,405 | 33,001 |
| 256 | 2048/2048 | 8,755 | 4,018 | 48.9 | 5,214 | 58,008 | 56,243 |

### Comparison — total tok/s vs the prior record
| conc | 06-29 | 06-30 | **07-09** | Δ vs 06-30 |
|---:|---:|---:|---:|---:|
| 16  | 2,087 | 2,156 | **2,482** | +15% |
| 64  | 4,591 | 4,614 | **6,070** | +32% |
| 128 | 4,512 | 3,298 | **7,786** | +136% |
| 256 | 4,898 | 4,892 | **8,755** | +79% |

### Findings
1. **Peak total throughput 4,900 -> 8,755 tok/s (+79%)**, ~306 -> ~547 tok/s/GPU over the 16 GPUs.
2. **Curve now scales past conc 64.** The prior record's "saturates at conc >= 64" no longer holds
   for this stack — throughput rises monotonically 16 -> 256. The single-prefill bottleneck is still
   visible in the TTFT tail (p99 5.3s -> 58s) but no longer caps total throughput in this range.
3. **conc-128 anomaly resolved.** 06-30 reproducibly dipped to ~3.3k (below conc 64); 07-09 gives
   7,786 on a clean curve. Since only the stack changed (same scripts / CPU alloc / IB / args), this
   points at the dip being a **software interleaving artifact** in the older sglang build rather than
   the CPU-per-step cap hypothesized in `INVESTIGATE-conc128.md` — worth re-reading that doc against
   `g0ffed946f`.
4. **TTFT dramatically lower** at every level (median TTFT down 70-80%). Only regression: TPOT at
   conc 256 slightly worse (42 -> 49 ms) — a fair trade for +79% throughput.

### Caveat on rigor
`dev-cu13` is a rolling tag and the driver moved (580 -> 610), so this is "same recipe, newer
platform," not a controlled A/B. The gain is real and large but not attributable to a single cause
without pinning the image digest (md5 recorded above for this run).

### Provenance (07-09)
Nodes slinky-0/1, allocation 29; prefill@slinky-0(:9000 bootstrap), decode@slinky-1, router
`http://10.245.232.13:8002`. Raw per-point JSON `$HOME/enroot/sweep/qwen3-32b_1k1k_conc{16,64,128,256}.bench.json`
+ sweep driver log `$HOME/enroot/sweep.log`; server logs `$HOME/enroot/{prefill,decode,router}.log`.
These per-run outputs stay machine-local by repo convention (`together_runner/.gitignore`: `results/`, `*.log`);
the tables above are the committed record. Image sqsh md5 `3cdb6e0a9dec073c4a30b374ca5f8a02`.
mooncake version not pip-visible in this image (unresolved).
