#!/usr/bin/env bash
# Step 2/3 — Launch the SGLang server inside the container, with staged
# progress monitoring, zero-download when weights are pre-staged, and a tuning
# switch.
#   bash run_sglang_2_launch_server.sh
#   SMOKE=1 bash run_sglang_2_launch_server.sh   # forces fast path (tuning off)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
source "$HERE/sglang_lib.sh"

check_env_vars CONTAINER_NAME PORT PROFILE MODEL TP || exit 1
SMOKE="${SMOKE:-0}"
SERVER_LOG="/root/server.log"

# SMOKE (gate-2) always takes the fast path regardless of config.
[[ "$SMOKE" == "1" ]] && ENABLE_TUNING=0

# --- Resolve model source: prefer pre-staged local weights (zero download) ---
STAGED="$MODELS_ROOT/$PROFILE"
if [[ -n "${MODEL_PATH:-}" && -d "$MODEL_PATH" ]]; then
    SRC="$MODEL_PATH"; trlog "using MODEL_PATH=$SRC (skip download)"
elif [[ -f "$STAGED/.ready" ]]; then
    SRC="$STAGED"; trlog "using pre-staged weights $SRC (.ready present, skip download)"
elif [[ "$MODEL" == /* ]]; then
    SRC="$MODEL"; trlog "MODEL is a local path $SRC (skip download)"
else
    SRC="$MODEL"
    trwarn "weights NOT pre-staged — server will download $MODEL on first start (slow)."
    trwarn "run prestage_weights.sh first to avoid this."
fi

# --- tuning flag ---
TUNE_ARGS=()
if [[ "$ENABLE_TUNING" == "1" ]]; then
    trlog "ENABLE_TUNING=1 — autotune KEPT (slower start, peak perf; use for baselines)"
else
    TUNE_ARGS+=(--disable-flashinfer-autotune)
    trlog "ENABLE_TUNING=0 — autotune SKIPPED (--disable-flashinfer-autotune)"
fi

# --- per-profile server args ---
case "$PROFILE" in
  smoke)
    SERVER_ARGS=(--model-path "$SRC" --host 0.0.0.0 --port "$PORT"
                 --tensor-parallel-size "$TP" --trust-remote-code "${TUNE_ARGS[@]}")
    ;;
  dsr1-fp4)
    check_env_vars EP_SIZE || exit 1
    SCHED_RECV=10; (( CONC >= 16 )) && SCHED_RECV=30
    SERVER_ARGS=(--model-path "$SRC" --host 0.0.0.0 --port "$PORT" --trust-remote-code
                 --tensor-parallel-size="$TP" --data-parallel-size=1
                 --cuda-graph-max-bs 256 --max-running-requests 256 --mem-fraction-static 0.85
                 --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 16384
                 --ep-size "$EP_SIZE" --quantization modelopt_fp4
                 --enable-flashinfer-allreduce-fusion --scheduler-recv-interval "$SCHED_RECV"
                 --enable-symm-mem --disable-radix-cache --attention-backend trtllm_mla
                 --moe-runner-backend flashinfer_trtllm --stream-interval 10 "${TUNE_ARGS[@]}")
    ;;
  *) trerr "unknown PROFILE='$PROFILE' (smoke | dsr1-fp4)"; exit 1 ;;
esac

trlog "profile=$PROFILE model_src=$SRC tp=$TP port=$PORT smoke=$SMOKE"

# Startup plan drives the monitor's "next stage / time to ready" hints.
if [[ "$ENABLE_TUNING" == "1" ]]; then
    STAGE_PLAN="engine-init weight-load autotune graph-capture"
else
    STAGE_PLAN="engine-init weight-load graph-capture"
fi

run_timer_start
# Launch detached inside the container.
docker exec -d "$CONTAINER_NAME" bash -c \
  "PYTHONNOUSERSITE=1 python3 -m sglang.launch_server $(printf '%q ' "${SERVER_ARGS[@]}") > $SERVER_LOG 2>&1"

# Staged startup monitor (handles weight-load / autotune / graph-capture / ready).
if monitor_server_until_ready "$CONTAINER_NAME" "$PORT" "$SERVER_LOG" 3600; then
    trlog "server READY in $(_fmt_dur "$(run_elapsed)") — next: bash run_sglang_3_test_client.sh"
else
    trerr "server failed to start (see log dump above)."
    exit 1
fi
