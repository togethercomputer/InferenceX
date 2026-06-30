# slurm-disagg — 2-node SGLang prefill/decode-disaggregated benchmark

Reproducible harness for the ClusterMAX inference-disagg phase-0 readiness proof:
deploy a 2-node prefill/decode-disaggregated SGLang endpoint on a Slurm + enroot/pyxis
cluster, transfer KV cross-node over RDMA, and benchmark.

**Portable by design — runs unmodified on a new cluster.** The harness grabs ONE 2-node
allocation (`salloc --no-shell`) and runs prefill/decode/router/bench as overlap steps
into it, so **Slurm picks the nodes** (no node names hardcoded). Partition, IB device
list, the peermem-vs-dmabuf KV path, and the enroot temp dir are all **auto-detected**;
only GPUs/node (8) and TP (8) are pinned, on purpose, for B200.

Validated 2026-06-29 with **Qwen3-32B** (1P1D, TP8 each): peak ~4,900 tok/s total
(~306 tok/s/GPU), 0 failures. See `BENCHMARK-RECORD-qwen3-32b-disagg.md` (re-validated 2026-06-30 post-refactor).

## Run it
```bash
cd together_runner/slurm-disagg
bash 00_setup.sh       # one-time per pod boot: node fixes + image import + weights (idempotent)
bash 01_preflight.sh   # auto-detect IB devices + peermem path, verify RDMA — seconds; writes detected.env
bash 10_launch.sh      # prefill + decode + router; waits healthy; writes state file
bash 20_benchmark.sh   # concurrency sweep -> result JSONs + summary table
bash teardown.sh       # scancel the servers
# or: bash run_all.sh  # 00 -> 01 -> 10 -> 20 in sequence
```
All config in `config.env` (sourced everywhere; override by exporting first).
`IB_DEVICES` and `WITH_NVIDIA_PEERMEM` default to **empty = auto-detected by preflight**
— set them explicitly only to override. `PROBE_MOONCAKE=1 bash 01_preflight.sh` adds the
heavy mooncake `register_memory` GPU probe.

## Layout
- `config.env` — paths, model, ports, sweep params. Partition/nodes/IB/peermem/enroot-temp
  default to AUTO; GPUs/node + TP pinned to 8 (B200). `ENROOT_DIR`/`MODELS_ROOT` default to
  `$HOME` and must be on a cross-node-shared FS (preflight verifies).
- `disagg_lib.sh` — allocation/node resolution (`ensure_allocation`, `resolve_partition`),
  container-mounts, health-wait, cuda-graph helpers, `load_nodes`/`load_resolved`.
- `00_setup.sh` — grab/reuse the 2-node allocation (persists `disagg_nodes.env`), ephemeral
  node fixes, `enroot import` (auto-detects a node-local non-overlay temp), `hf download`. Idempotent.
- `01_preflight.sh` — detect GPU↔NIC topology + peermem/dmabuf decision + IB-port/RDMA
  checks on both nodes; writes `$LOG_DIR/disagg_detected.env` (the resolved RDMA truth source).
- `_detect_rdma.py` — `nvidia-smi topo -m` + `/sys` link_layer → GPU-ordered IB device list.
- `_check_container_rdma.py` — in-container ibv device count, dmabuf export, optional mooncake probe.
- `10_launch.sh` — sources detected.env; launch prefill/decode/router, wait healthy, write `disagg_state.env`.
- `20_benchmark.sh` — sweep via `utils/bench_serving/benchmark_serving.py`, summary table.
- `teardown.sh` — cancel jobs. `run_all.sh` — full pipeline.

## Environment gotchas (this is Slurm-on-k8s; nodes are nested pods)
Preflight (`01_preflight.sh`) now **detects or verifies** most of these so a new cluster
fails in seconds with a clear reason instead of mid-launch:
1. **enroot import temp must be ext4** (`ENROOT_TEMP_PATH=/scratch/...`) — overlay `/`
   can't `mknod` the overlayfs whiteouts → `aufs2ovlfs ... Operation not permitted`.
2. **enroot nvidia hook** patched with `--no-persistenced --no-fabricmanager` (00_setup.sh,
   needs sudo) — else `nvidia-container-cli` can't bind-mount those sockets in the pod.
   *This sed-patch is unavoidable on this stack:* enroot 4.0.1 runs system+user `hooks.d`
   with no basename dedup (a user hook can't replace the system one), and pyxis ignores a
   per-job `ENROOT_SYSCONF_PATH` redirect — both verified. The patch is idempotent + self-checking.
3. **RDMA-to-pod** = bind-mount `/dev/infiniband` (in `container_mounts`). Preflight verifies
   libibverbs sees the same device count as `/sys/class/infiniband`. (Image lacks `ibv_devinfo`/`rdma` CLIs — cosmetic.)
4. **KV-mem registration path auto-decided** by preflight: `nvidia_peermem` present → default
   path; absent + driver ≥535 → `WITH_NVIDIA_PEERMEM=0` (mooncake dmabuf). Without the right
   choice: thousands of `Failed to register memory` → `KVTransferError` on routed requests.
5. **IB device list auto-detected** (`--disaggregation-ib-device`): GPU↔NIC PCIe affinity from
   `nvidia-smi topo -m`, filtered to the majority fabric (IB drops the Ethernet storage NICs;
   RoCE keeps Ethernet). Override via `IB_DEVICES=...`.
6. **CUDA graph ON** (default) is mandatory for perf — eager mode is ~22x slower with
   65% client timeouts. (`DISABLE_CUDA_GRAPH=1` only for debugging.)
7. **Router** runs as an `srun --jobid=<prefill> --overlap` step (Slurm won't co-schedule a
   fresh job on a full node) and **mounts the model dir** (else tokenizer 404s to HF).

Fixes 1–2 are ephemeral and reset on pod restart — re-run `00_setup.sh`. Detection (3–5) is
re-run each `01_preflight.sh`, so it self-adjusts on a new cluster/HW.

## Known limitation (current config)
1P1D throughput saturates ~conc 64 (the single prefill instance is the bottleneck); TTFT
tail grows sharply with concurrency. For higher throughput / lower TTFT, scale prefill
instances (NP1D) — a future extension of `10_launch.sh`.
