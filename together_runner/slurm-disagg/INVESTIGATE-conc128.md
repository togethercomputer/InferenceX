# Investigation: conc=128 throughput dip (1P1D, Qwen3-32B, slinky)

Status: **OPEN — deferred**. Captured 2026-06-30 so we can resume without re-deriving.
Owner: johnson. Related: `BENCHMARK-RECORD-qwen3-32b-disagg.md`, `PORTABILITY-ANALYSIS.md`.

## The observation
1k/1k sweep, 1P1D, TP8 each. **0 failed requests at every point.** conc 16/64/256 match the
06-29 baseline; **conc=128 is reproducibly low**:

| conc | 06-29 baseline | 06-30 run1 | 06-30 run2 | note |
|---:|---:|---:|---:|---|
| 16  | 2,087 | 1,874 | 2,156 | run1 low, run2 on-baseline |
| 64  | 4,591 | 4,614 | —     | ✓ |
| 128 | 4,512 | 3,437 | 3,298 | **reproducibly ~3.3–3.4k, below conc=64** |
| 256 | 4,898 | 4,892 | —     | ✓ |

The dip is *physically odd*: conc=128 sits **below** conc=64, and TTFT jumps (med ~6.8–8.8 s,
p99 ~37 s). On a healthy curve 128 should sit between 64 and 256.

## Is it our refactor? — NO (proven 2026-06-30)
Hypothesis was that the allocation/overlap-step model (this PR) changed CPU/resource isolation
vs the old separate-`srun`-job model, starving prefill at the conc=128 operating point.

**Disproved by direct probe** — both models grant the SAME CPUs per node:

```
# NEW model (overlap step into 'salloc -N2 --gres=gpu:8' allocation):
srun --jobid=$ALLOC --overlap -N1 -w slinky-0 bash -c 'nproc; grep Cpus_allowed_list /proc/self/status'
  -> nproc=2   Cpus_allowed_list=0-1   SLURM_CPUS_ON_NODE=2

# OLD model (standalone job, what 06-29 used):
srun -p slinky -N1 -w slinky-0 --gres=gpu:8 bash -c 'nproc; grep Cpus_allowed_list /proc/self/status'
  -> nproc=2   Cpus_allowed_list=0-1   SLURM_CPUS_ON_NODE=2
```

Both = 2 CPUs/node (cluster is `SelectTypeParameters=CR_CORE_MEMORY` with **no `DefCpuPerGPU`**,
so a gpu:8 request gets the minimal 1–2 cores). Serving args / nodes / IB list are byte-for-byte
identical between the runs. **So the refactor is not the cause of the 128 dip.**

Most likely: **environmental / 1P1D dynamics variance** — at conc=128 the single prefill instance
interleaves badly with decode (prefill bursts starve decode), an unstable operating point; 06-29's
4,512 was a luckier sample. (conc=16 also swung 1,874↔2,156 between runs → this setup has real
run-to-run variance.)

## Separate finding worth chasing: both models are CPU-starved to 2 cores
Every server (prefill, decode, router, bench client) runs pinned to **2 CPUs**. CPU-bound work
(tokenization, scheduler, sampling, FastAPI/uvicorn, mooncake orchestration) is throttled. This
likely **caps throughput across ALL points**, and more CPU headroom may also **stabilize the 128
point**. This is the highest-value lever and is independent of the refactor.

## Plan to resume (ranked)
1. **Give the allocation real CPUs, re-sweep.** Easiest, highest upside. In `disagg_lib.sh`
   `ensure_allocation`, add to the `salloc`: `--cpus-per-task` (or `--exclusive`, or
   `--cpus-per-gpu=<n>`). Probe the node core count first (`sinfo -h -n <node> -o '%c'` showed
   the *allocated* 2, not physical — check `scontrol show node <n> | grep CPUTot`; the box has
   ~160). Try `--exclusive` (whole node) → re-run conc 128 (and full sweep). Expectation: all
   points rise; if 128 also normalizes, the dip was CPU contention after all.
2. **A/B old vs new launch at conc=128**, CPU held constant, to fully close the refactor question
   (should match — both 2 CPU). Old launch commands preserved below.
3. **Instrument the 128 point.** While conc=128 runs, watch:
   - prefill log: `Prefill batch ... #queue-req`, input throughput — is prefill the bottleneck?
   - decode log: `Decode batch ... #running-req, gen throughput` — is decode starved (running-req
     oscillating / low) at 128 but steady at 256?
   - `nvidia-smi dmon` / GPU util on both nodes; CPU util of the 2 allowed cores (likely pegged).
4. **Sweep around it**: conc 96 / 128 / 160 / 192 to map whether it's a single bad point or a
   trough between 64 and 256.

## Reproducing the ORIGINAL (06-29) conditions — so we don't lose the 4,512 baseline
The 06-29 run used the OLD launch (separate jobs, hardcoded nodes) + first 20_benchmark. That code
was refactored in place (not in git), but the exact commands are preserved in
`~/enroot/HANDOVER-disagg-kv-transfer.md` + `HANDBACK-codex-dmabuf.md`, and reproduced here:

```bash
IMG=$HOME/enroot/sglang-dev-cu13.sqsh
MODEL=$HOME/models/qwen3-32b
IB=mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_4,mlx5_5,mlx5_6,mlx5_7
MOUNTS=$MODEL:$MODEL,$HOME/enroot:$HOME/enroot,$HOME/.cache/huggingface:/root/.cache/huggingface,/dev/infiniband:/dev/infiniband

# PREFILL (slinky-0) — standalone job, CUDA graph ON (omit --disable-cuda-graph):
srun -p slinky -N1 -w slinky-0 --gres=gpu:8 --job-name=pd_prefill \
  --container-image=$IMG --container-mounts=$MOUNTS \
  bash -c 'WITH_NVIDIA_PEERMEM=0 python3 -m sglang.launch_server --model-path '$MODEL' \
    --served-model-name qwen3-32b --tp 8 --host 0.0.0.0 --port 30000 --trust-remote-code \
    --disaggregation-mode prefill --disaggregation-bootstrap-port 9000 --disaggregation-ib-device '$IB
# DECODE (slinky-1): same but --disaggregation-mode decode (no bootstrap port).
# ROUTER: srun --jobid=<prefill_jobid> --overlap -w slinky-0 ... launch_router --pd-disaggregation \
#   --prefill http://<pf_ip>:30000 9000 --decode http://<dec_ip>:30000 --port 8002 --policy random
```
Note: 06-29 also ran on **2 CPUs/node** (same default), so the 4,512 should be reproducible under
the same conditions — the gap is variance, not a lost config. For a *deterministic* baseline, pin
CPUs via `--exclusive` (item 1) in BOTH old and new and compare.

## Not a blocker
0 failures, correctness intact, peak (4,892) matches baseline. conc=128 is a perf-shape question,
documented for later — does not block the harness landing.
