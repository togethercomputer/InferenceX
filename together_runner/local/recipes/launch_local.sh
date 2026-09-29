#!/usr/bin/env bash
#
# Run an official InferenceX single-node recipe on a local box, via Docker.
#
# The shipped launchers (runners/launch_*.sh) assume a scheduler: Slurm
# salloc/srun plus enroot squashfs imports (AMD, NVIDIA DGXC) or cloud-specific
# mounts. This provides the same contract -- set the recipe's env vars, run the
# recipe in the pinned image, leave $RESULT_FILENAME.json behind -- using plain
# `docker run`, on either vendor. GPU plumbing comes from lib/vendor.sh.
#
# Required env (the names benchmark-tmpl.yml exports), for every scenario:
#   MODEL TP EP_SIZE DP_ATTENTION CONC RESULT_FILENAME IMAGE FRAMEWORK
#   PRECISION EXP_NAME MODEL_PREFIX RECIPE_RUNNER
# fixed-seq-len additionally:
#   ISL OSL MAX_MODEL_LEN RANDOM_RANGE_RATIO
# agentic-coding additionally:
#   KV_OFFLOADING TOTAL_CPU_DRAM_GB DURATION   (RESULT_DIR defaults below)
# Optional:
#   GPUS (comma-separated GPU indices), PORT, WORK_DIR, HF_HUB_CACHE,
#   SPEC_DECODING, DISAGG, RUN_EVAL, EVAL_ONLY, SCENARIO_TYPE
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TR_ROOT="$(cd "$HERE/../.." && pwd)"
source "$TR_ROOT/lib/common.sh"
source "$TR_ROOT/lib/vendor.sh"
source "$TR_ROOT/lib/paths.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$TR_ROOT/.." && pwd)}"
WORK_DIR="${WORK_DIR:-$TR_WORK_DIR}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$TR_HF_CACHE}"
VLLM_CACHE="${VLLM_CACHE:-$TR_VLLM_CACHE}"
TRITON_CACHE="${TRITON_CACHE:-$TR_TRITON_CACHE}"
# agentic 回放的 uv/venv 运行时。默认是容器内 /tmp/inferencex-agentic-$$，
# 每个配置都会重建 venv 并重下 uv+aiperf 依赖；固定到宿主机目录后整个 sweep
# 只装一次。注意这会让并发运行互相踩踏 —— sweep 是串行的，所以没问题。
#
# UV_PYTHON_INSTALL_DIR 也必须一起固定：uv 会下载一份 standalone CPython 到
# /root/.local/share/uv/python（容器内临时目录），venv/bin/python 是指向它的
# 符号链接。只持久化 venv 的话，换个容器就是断链，venv 等于没存。
AGENTIC_RUNTIME="${AGENTIC_RUNTIME:-$TR_AGENTIC_RUNTIME}"
PORT="${PORT:-8710}"
CONTAINER_NAME="${CONTAINER_NAME:-ix_${FRAMEWORK}_tr}"

# The two scenarios have different contracts: agentic-coding replays a trace
# corpus for a wall-clock DURATION and has no fixed sequence lengths, so
# demanding ISL/OSL/MAX_MODEL_LEN there would reject a valid config.
SCENARIO_TYPE="${SCENARIO_TYPE:-fixed-seq-len}"
case "$SCENARIO_TYPE" in
    agentic-coding) SCENARIO_SUBDIR="${SCENARIO_SUBDIR:-agentic/}" ;;
    fixed-seq-len)  SCENARIO_SUBDIR="${SCENARIO_SUBDIR:-fixed_seq_len/}" ;;
    *) echo "ERROR: unknown SCENARIO_TYPE=$SCENARIO_TYPE" >&2; exit 1 ;;
esac

check_env_vars MODEL TP EP_SIZE DP_ATTENTION CONC RESULT_FILENAME IMAGE \
               FRAMEWORK PRECISION EXP_NAME MODEL_PREFIX RECIPE_RUNNER || exit 1
if [[ "$SCENARIO_TYPE" == "agentic-coding" ]]; then
    check_env_vars KV_OFFLOADING TOTAL_CPU_DRAM_GB DURATION || exit 1
    # The recipes write every artefact here and the replay driver reads it back.
    export RESULT_DIR="${RESULT_DIR:-/workspace/results}"
else
    check_env_vars ISL OSL MAX_MODEL_LEN RANDOM_RANGE_RATIO || exit 1
fi

mkdir -p "$WORK_DIR" "$HF_HUB_CACHE" "$VLLM_CACHE" "$TRITON_CACHE" "$AGENTIC_RUNTIME"

# The recipe writes server.log, gpu_metrics.csv and $RESULT_FILENAME.json into
# /workspace. Mount a scratch dir there and the repo's code read-only beneath
# it, so benchmark output never dirties the git checkout.
rm -f "$WORK_DIR/server.log" "$WORK_DIR/gpu_metrics.csv" \
      "$WORK_DIR/${RESULT_FILENAME}.json" 2>/dev/null || true
# benchmark_lib.sh sources runners/srt-slurm/hooks/common.sh partway through;
# without that mount the source aborts there and every function defined below
# it (resolve_trace_source among them) is silently missing.
CODE_DIRS=(utils benchmarks runners infx configs)
CODE_MOUNTS=()
for d in "${CODE_DIRS[@]}"; do
    if [[ -d "$REPO_ROOT/$d" ]]; then
        mkdir -p "$WORK_DIR/$d"
        CODE_MOUNTS+=(-v "$REPO_ROOT/$d:/workspace/$d:ro")
    fi
done
# Agentic output is a tree under results/; a leftover from the previous config
# would be harvested into this one, so start empty -- and fail loudly if an
# older root-owned tree cannot be removed rather than mixing runs.
rm -rf "$WORK_DIR/results" || { trerr "cannot clear $WORK_DIR/results (root-owned leftover?)"; exit 1; }
mkdir -p "$WORK_DIR/results"

# Recipe filename, derived exactly as the official launchers derive it:
#   ${EXP_NAME%%_*}_${PRECISION}_${RECIPE_RUNNER}[_${FRAMEWORK}][_mtp].sh
SPEC_SUFFIX=$([[ "${SPEC_DECODING:-none}" == "mtp" ]] && printf '_mtp' || printf '')
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "atom" ]] && printf '_atom' || printf '')
SCRIPT_BASE="${EXP_NAME%%_*}_${PRECISION}_${RECIPE_RUNNER}"
SUBDIR="$SCENARIO_SUBDIR"
SCRIPT_FW="benchmarks/single_node/${SUBDIR}${SCRIPT_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
SCRIPT_FALLBACK="benchmarks/single_node/${SUBDIR}${SCRIPT_BASE}${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
if [[ -f "$REPO_ROOT/$SCRIPT_FW" ]]; then
    BENCHMARK_SCRIPT="$SCRIPT_FW"
elif [[ -f "$REPO_ROOT/$SCRIPT_FALLBACK" ]]; then
    BENCHMARK_SCRIPT="$SCRIPT_FALLBACK"
else
    trerr "no recipe at $SCRIPT_FW or $SCRIPT_FALLBACK"; exit 1
fi

vendor_set_gpu_flags "${GPUS:-}" || exit 1
if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -ra _g <<< "$GPUS"
    (( ${#_g[@]} == TP )) || { trerr "GPUS=$GPUS lists ${#_g[@]} GPU(s) but TP=$TP"; exit 1; }
fi
trlog "vendor=$(vendor_detect) recipe=$BENCHMARK_SCRIPT gpus=${GPUS:-all} port=$PORT"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# --entrypoint bash is the docker equivalent of the official launchers'
# `--no-container-entrypoint`: engine images ENTRYPOINT to `vllm` /
# `python3 -m sglang...`, so without it the recipe path is silently handed to
# the engine as CLI arguments instead of being executed.
docker run --rm --name "$CONTAINER_NAME" \
    --entrypoint bash \
    "${VENDOR_GPU_FLAGS[@]}" \
    --cap-add=SYS_PTRACE \
    --ipc=host --shm-size 64G \
    --network host \
    -v "$WORK_DIR:/workspace" \
    "${CODE_MOUNTS[@]}" \
    -v "$HF_HUB_CACHE:/hf-cache" \
    -v "$VLLM_CACHE:/root/.cache/vllm" \
    -v "$TRITON_CACHE:/root/.triton" \
    -v "$AGENTIC_RUNTIME:/agentic-runtime" \
    -w /workspace \
    -e VLLM_CACHE_ROOT=/root/.cache/vllm \
    -e TRITON_CACHE_DIR=/root/.triton \
    -e INFMAX_CONTAINER_WORKSPACE=/workspace \
    -e INFERENCEX_REPO_ROOT=/workspace \
    -e AIPERF_RUNTIME_DIR=/agentic-runtime \
    -e UV_PYTHON_INSTALL_DIR=/agentic-runtime/uv-python \
    -e UV_CACHE_DIR=/agentic-runtime/uv-cache \
    -e HF_HUB_CACHE=/hf-cache \
    -e HF_HOME=/hf-cache \
    -e HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}" \
    -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    -e PORT="$PORT" \
    -e MODEL -e TP -e EP_SIZE -e DP_ATTENTION -e CONC \
    -e ISL -e OSL -e MAX_MODEL_LEN -e RANDOM_RANGE_RATIO \
    -e RESULT_FILENAME -e FRAMEWORK -e PRECISION -e EXP_NAME -e MODEL_PREFIX \
    -e SCENARIO_TYPE -e RESULT_DIR \
    -e RUNNER_TYPE -e IMAGE \
    -e PP_SIZE -e DCP_SIZE -e PCP_SIZE \
    `# CI env-block vars that runtime_settings.sh does NOT cover` \
    -e RUNNER_NAME="${RUNNER_NAME:-$(hostname -s)}" \
    -e IS_MULTINODE="${IS_MULTINODE:-false}" \
    -e IS_AGENTIC="$([[ "$SCENARIO_TYPE" == "agentic-coding" ]] && echo 1 || echo 0)" \
    -e THINKING_MODE="${THINKING_MODE:-thinking_on}" \
    -e GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}" \
    -e GPU_METRICS_CSV="${GPU_METRICS_CSV:-gpu_metrics.csv}" \
    -e REQUIRE_POWER="${REQUIRE_POWER:-0}" \
    -e ENABLE_AGENTX_POWER="${ENABLE_AGENTX_POWER:-0}" \
    -e AIPERF_EXPERIMENTAL_FAST="${AIPERF_EXPERIMENTAL_FAST:-0}" \
    -e SWEBENCH_USE_MODAL="${SWEBENCH_USE_MODAL:-false}" \
    -e SWEBENCH_GEN_MODE="${SWEBENCH_GEN_MODE:-agentic}" \
    -e KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-}" \
    -e ROUTER_METADATA="${ROUTER_METADATA:-}" \
    -e EVAL_FRAMEWORK="${EVAL_FRAMEWORK:-}" \
    -e EVAL_SUITE="${EVAL_SUITE:-}" \
    -e EVAL_LIMIT="${EVAL_LIMIT:-}" \
    -e KV_OFFLOADING -e TOTAL_CPU_DRAM_GB -e DURATION \
    -e SPEC_DECODING="${SPEC_DECODING:-none}" \
    -e DISAGG="${DISAGG:-false}" \
    -e RUN_EVAL="${RUN_EVAL:-false}" \
    -e EVAL_ONLY="${EVAL_ONLY:-false}" \
    -e PYTHONUNBUFFERED=1 \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    "$IMAGE" \
    -c '
# CI sources these in the WORKFLOW before the recipe runs
# (benchmarks/runtime_settings.sh: "workflow-owned common settings, loaded
# before recipe-specific overrides"). This launcher plays the workflow role, so
# it must do the same -- without them the agentic path fails on
# AIPERF_PYTHON_VERSION. Guarded so older checkouts without the file still work.
set -a
[ -f benchmarks/runtime_settings.sh ] && . benchmarks/runtime_settings.sh
[ -f runners/runtime_settings.sh ] && . runners/runtime_settings.sh
set +a
bash "$0"; rc=$?
# The container runs as root, so everything it wrote is root-owned; a root
# subdirectory (aiperf_artifacts/) cannot even be emptied by the host user,
# which broke the cross-device move in sweep.py. Hand it back, pass or fail.
chown -R "$HOST_UID:$HOST_GID" /workspace/results 2>/dev/null
find /workspace -maxdepth 1 -type f -exec chown "$HOST_UID:$HOST_GID" {} + 2>/dev/null
exit $rc' "$BENCHMARK_SCRIPT"

if [[ ! -f "$WORK_DIR/${RESULT_FILENAME}.json" ]]; then
    trerr "benchmark produced no $WORK_DIR/${RESULT_FILENAME}.json"; exit 1
fi
trlog "OK -> $WORK_DIR/${RESULT_FILENAME}.json"
