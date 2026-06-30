# slurm-disagg — Claude Code memory (quick debug/ramp-up)

2-node SGLang **prefill/decode-disaggregated** benchmark harness on Slurm + enroot/pyxis.
Productized ClusterMAX disagg phase-0 proof. **Portable**: meant to run unmodified on any
such cluster. Sibling of `../CLAUDE.md` (single-node together_runner) but separate code.

## State (2026-06-30)
- Committed on branch `together-runner-slurm-disagg`, **PR #2** open vs main
  (togethercomputer/InferenceX). Author `Johnsonms <lizhaofu@gmail.com>`, no Claude trailer.
- Validated on slinky (Qwen3-32B 1P1D, TP8): conc 16/64/256 match 06-29 baseline, 0 failures.
- **OPEN issue**: conc=128 reproducible throughput dip (~3.3k, below conc64). NOT the refactor
  (CPU alloc identical old vs new — proven). See `INVESTIGATE-conc128.md`. Deferred.

## Architecture (how it runs)
ONE 2-node allocation, everything is an overlap step into it:
```
00_setup.sh   -> salloc -N2 --no-shell (ALLOC_JOB) ; patch enroot nvidia hook ; enroot import ; hf download
01_preflight  -> auto-detect IB_DEVICES + peermem/dmabuf ; verify IB ports + bind-mount + dmabuf
10_launch.sh  -> prefill@node0 + decode@node1 + router@node0  (all `srun --jobid=$ALLOC --overlap`)
20_benchmark  -> bench client (overlap step on node0) -> result JSONs + summary
teardown.sh   -> scancel $ALLOC_JOB ; rm nodes/state files
run_all.sh    -> 00 -> 01 -> 10 -> 20
```
Resolved state lives in `$LOG_DIR` (= `$ENROOT_DIR` = `$HOME/enroot`):
- `disagg_nodes.env` — ALLOC_JOB, PREFILL/DECODE_NODE+IP, partition (written by 00_setup).
- `disagg_detected.env` — IB_DEVICES, WITH_NVIDIA_PEERMEM (written by 01_preflight).
- `disagg_state.env` — ENDPOINT (written by 10_launch).
- `load_nodes()` (nodes only, for preflight) vs `load_resolved()` (nodes + detected, for launch/bench).

## What's AUTO vs PINNED
- AUTO: partition (first w/ ≥2 idle GPU nodes), nodes (Slurm-assigned), IB_DEVICES
  (`nvidia-smi topo -m` + `/sys` link_layer, majority-fabric), WITH_NVIDIA_PEERMEM
  (peermem present→default; absent+drv≥535→0/dmabuf), enroot temp (first non-overlay fs).
- PINNED on purpose: GPUS_PER_NODE=8, TP=8 (B200). Override anything via env / config.env.

## Gotchas / hard-won facts (don't relearn these)
- **enroot nvidia-hook sed-patch is unavoidable** (needs sudo, ephemeral). enroot 4.0.1 runs
  system+user hooks.d with NO basename dedup (user hook can't override system 98-nvidia.sh);
  pyxis ignores per-job ENROOT_SYSCONF_PATH. Both verified. Patch is idempotent + self-checking.
- **KV transfer needs the dmabuf path** here (`WITH_NVIDIA_PEERMEM=0`) — `nvidia_peermem` absent
  (driver 580, only gdrdrv+nvidia_fs). Without it: 3072× `Failed to register memory` → KVTransferError.
- **CUDA graph mandatory** (default ON) — eager ≈ 22× slower / 65% timeouts.
- **enroot import temp must be non-overlay fs** — pod `/` is overlayfs, can't mknod whiteouts.
- **Every step gets only 2 CPUs** on slinky (CR_CORE_MEMORY, no DefCpuPerGPU) — suspected
  throughput cap; see INVESTIGATE-conc128.md item 1 (`--exclusive`/`--cpus-per-task`).
- Router MUST mount the model dir (else tokenizer 404s to HF); runs on the prefill node.

## Debug entry points
- Server logs: `$HOME/enroot/{prefill,decode,router}.log`. Sweep: `$HOME/enroot/sweep.log`,
  results `$HOME/enroot/sweep/*.bench.json`.
- Liveness while running: `tail -f decode.log` (look for `Decode batch ... gen throughput`),
  `squeue --me` (ALLOC_JOB), endpoint `curl $ENDPOINT/health`.
- Re-detect RDMA only: `bash 01_preflight.sh` (rewrites detected.env from clean config).
- Re-run subset: `CONC_LIST="64" bash 20_benchmark.sh` (endpoint must be up).
- Cross-cluster: just `bash run_all.sh` — if it fails, preflight prints the exact reason.

## Related docs
`README.md` (usage), `PORTABILITY-ANALYSIS.md` (root-cause table + decisions),
`BENCHMARK-RECORD-qwen3-32b-disagg.md` (numbers), `INVESTIGATE-conc128.md` (open perf question).
Testbed/node fixes: see repo memory `disagg-enroot-node-setup`.
