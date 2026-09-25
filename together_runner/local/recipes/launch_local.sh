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
# Required env (the names .github/workflows/benchmark-tmpl.yml exports):
#   MODEL TP EP_SIZE DP_ATTENTION CONC ISL OSL MAX_MODEL_LEN
#   RANDOM_RANGE_RATIO RESULT_FILENAME IMAGE FRAMEWORK PRECISION EXP_NAME
# Plus:
#   RECIPE_RUNNER   token in the recipe filename (mi355x, b200, h200, ...)
# Optional:
#   GPUS (comma-separated GPU indices), PORT, WORK_DIR, HF_HUB_CACHE,
#   SPEC_DECODING, DISAGG, RUN_EVAL, EVAL_ONLY, SCENARIO_SUBDIR
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TR_ROOT="$(cd "$HERE/../.." && pwd)"
source "$TR_ROOT/lib/common.sh"
source "$TR_ROOT/lib/vendor.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$TR_ROOT/.." && pwd)}"
WORK_DIR="${WORK_DIR:-${TR_WORK_DIR:-/tmp/inferencex-work}}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface}"
VLLM_CACHE="${VLLM_CACHE:-$HOME/.cache/tr-vllm}"
TRITON_CACHE="${TRITON_CACHE:-$HOME/.cache/tr-triton}"
PORT="${PORT:-8710}"
CONTAINER_NAME="${CONTAINER_NAME:-ix_${FRAMEWORK}_tr}"

check_env_vars MODEL TP EP_SIZE DP_ATTENTION CONC ISL OSL MAX_MODEL_LEN \
               RANDOM_RANGE_RATIO RESULT_FILENAME IMAGE FRAMEWORK PRECISION \
               EXP_NAME RECIPE_RUNNER || exit 1

mkdir -p "$WORK_DIR" "$HF_HUB_CACHE" "$VLLM_CACHE" "$TRITON_CACHE"

# The recipe writes server.log, gpu_metrics.csv and $RESULT_FILENAME.json into
# /workspace. Mount a scratch dir there and the repo's code read-only beneath
# it, so benchmark output never dirties the git checkout.
rm -f "$WORK_DIR/server.log" "$WORK_DIR/gpu_metrics.csv" \
      "$WORK_DIR/${RESULT_FILENAME}.json" 2>/dev/null || true
mkdir -p "$WORK_DIR/utils" "$WORK_DIR/benchmarks"

# Recipe filename, derived exactly as the official launchers derive it:
#   ${EXP_NAME%%_*}_${PRECISION}_${RECIPE_RUNNER}[_${FRAMEWORK}][_mtp].sh
SPEC_SUFFIX=$([[ "${SPEC_DECODING:-none}" == "mtp" ]] && printf '_mtp' || printf '')
FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "atom" ]] && printf '_atom' || printf '')
SCRIPT_BASE="${EXP_NAME%%_*}_${PRECISION}_${RECIPE_RUNNER}"
SUBDIR="${SCENARIO_SUBDIR:-fixed_seq_len/}"
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
    -v "$REPO_ROOT/utils:/workspace/utils:ro" \
    -v "$REPO_ROOT/benchmarks:/workspace/benchmarks:ro" \
    -v "$HF_HUB_CACHE:/hf-cache" \
    -v "$VLLM_CACHE:/root/.cache/vllm" \
    -v "$TRITON_CACHE:/root/.triton" \
    -w /workspace \
    -e VLLM_CACHE_ROOT=/root/.cache/vllm \
    -e TRITON_CACHE_DIR=/root/.triton \
    -e HF_HUB_CACHE=/hf-cache \
    -e HF_HOME=/hf-cache \
    -e HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}" \
    -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    -e PORT="$PORT" \
    -e MODEL -e TP -e EP_SIZE -e DP_ATTENTION -e CONC -e ISL -e OSL \
    -e MAX_MODEL_LEN -e RANDOM_RANGE_RATIO -e RESULT_FILENAME \
    -e FRAMEWORK -e PRECISION -e EXP_NAME \
    -e SPEC_DECODING="${SPEC_DECODING:-none}" \
    -e DISAGG="${DISAGG:-false}" \
    -e RUN_EVAL="${RUN_EVAL:-false}" \
    -e EVAL_ONLY="${EVAL_ONLY:-false}" \
    -e PYTHONUNBUFFERED=1 \
    "$IMAGE" \
    "$BENCHMARK_SCRIPT"

if [[ ! -f "$WORK_DIR/${RESULT_FILENAME}.json" ]]; then
    trerr "benchmark produced no $WORK_DIR/${RESULT_FILENAME}.json"; exit 1
fi
trlog "OK -> $WORK_DIR/${RESULT_FILENAME}.json"
