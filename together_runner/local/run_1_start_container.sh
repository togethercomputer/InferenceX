#!/usr/bin/env bash
# Step 1/3 — Start (or reuse) the inference container ($ENGINE) on this node.
# Works on NVIDIA and AMD/ROCm; GPU plumbing comes from lib/vendor.sh.
#   bash run_1_start_container.sh
# Idempotent: reuses/restarts an existing container of the same name.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
TR_LIB="$(cd "$HERE/../lib" && pwd)"
source "$TR_LIB/common.sh"
source "$TR_LIB/vendor.sh"
source "$TR_LIB/monitor.sh"

check_env_vars ENGINE IMAGE CONTAINER_NAME PORT HF_CACHE MODELS_ROOT REPO_ROOT || exit 1

mkdir -p "$HF_CACHE" "$MODELS_ROOT" "$FLASHINFER_CACHE" \
         "${VLLM_CACHE:-$HOME/.cache/tr-vllm}" "${TRITON_CACHE:-$HOME/.cache/tr-triton}"

run_timer_start
stage_begin "container-start"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        trlog "container '$CONTAINER_NAME' already running — reusing."
    else
        trlog "container '$CONTAINER_NAME' exists but stopped — starting."
        docker start "$CONTAINER_NAME" >/dev/null
    fi
else
    trlog "launching container '$CONTAINER_NAME' (engine=$ENGINE) from '$IMAGE'..."
    vendor_set_gpu_flags "${GPUS:-}" || exit 1
    # NVIDIA-only knobs; harmless to omit on ROCm.
    VENDOR_ENV=()
    if [[ "$(vendor_detect)" == "nvidia" ]]; then
        VENDOR_ENV=(-e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}")
    fi
    docker run -d --name "$CONTAINER_NAME" \
        "${VENDOR_GPU_FLAGS[@]}" "${VENDOR_ENV[@]}" \
        --ipc=host --shm-size=32g \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -p "${PORT}:${PORT}" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -v "${MODELS_ROOT}:${MODELS_ROOT}" \
        -v "${FLASHINFER_CACHE}:/root/.cache/flashinfer" \
        -v "${VLLM_CACHE:-$HOME/.cache/tr-vllm}:/root/.cache/vllm" \
        -v "${TRITON_CACHE:-$HOME/.cache/tr-triton}:/root/.triton" \
        -v "${REPO_ROOT}:/inferencex:ro" \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e PORT="${PORT}" \
        --entrypoint bash \
        "$IMAGE" \
        -c "sleep infinity" >/dev/null
fi

stage_end
trlog "sanity check:"
# Engine-aware import check (vLLM containers have no sglang and vice-versa).
if [[ "$ENGINE" == "vllm" ]]; then
    docker exec "$CONTAINER_NAME" python3 -c "import vllm; print('  vllm', vllm.__version__)"
else
    docker exec "$CONTAINER_NAME" python3 -c "import sglang; print('  sglang', sglang.__version__)"
fi
trlog "bench client: $(docker exec "$CONTAINER_NAME" bash -c '[ -f /inferencex/utils/bench_serving/benchmark_serving.py ] && echo mounted || echo MISSING')"
trlog "vendor: $(vendor_detect) | GPUs visible in container: $(docker exec "$CONTAINER_NAME" bash -c "$(vendor_container_gpu_check)" 2>/dev/null || echo '?')"
trlog "next: bash run_2_launch_server.sh"
