#!/usr/bin/env bash
# Step 1/3 — Start (or reuse) the inference container ($ENGINE) on this B200 node.
#   bash run_1_start_container.sh
# Idempotent: reuses/restarts an existing container of the same name.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
source "$HERE/bench_lib.sh"

check_env_vars ENGINE IMAGE CONTAINER_NAME PORT HF_CACHE MODELS_ROOT REPO_ROOT || exit 1

mkdir -p "$HF_CACHE" "$MODELS_ROOT" "$FLASHINFER_CACHE"

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
    docker run -d --name "$CONTAINER_NAME" \
        --gpus all \
        --ipc=host --shm-size=32g \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -p "${PORT}:${PORT}" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -v "${MODELS_ROOT}:${MODELS_ROOT}" \
        -v "${FLASHINFER_CACHE}:/root/.cache/flashinfer" \
        -v "${REPO_ROOT}:/inferencex:ro" \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e PORT="${PORT}" \
        -e TORCH_CUDA_ARCH_LIST="10.0" \
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
trlog "GPUs visible in container: $(docker exec "$CONTAINER_NAME" nvidia-smi -L | wc -l)"
trlog "next: bash run_2_launch_server.sh"
