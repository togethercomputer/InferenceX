#!/usr/bin/env bash
# Launch the 1P1D disaggregated endpoint as overlap steps INTO the 2-node allocation:
# prefill@PREFILL_NODE + decode@DECODE_NODE + sgl-router@PREFILL_NODE. Waits for health
# and writes a state file (endpoint) for benchmark/teardown.
#
#   bash 10_launch.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"; source "$HERE/disagg_lib.sh"
load_resolved
alloc_alive "${ALLOC_JOB:-}" || { trerr "no live allocation — run 00_setup.sh first"; exit 1; }
STATE="$LOG_DIR/disagg_state.env"

# Resolved RDMA config (IB_DEVICES + WITH_NVIDIA_PEERMEM) comes from preflight.
DETECTED="$LOG_DIR/disagg_detected.env"
[[ -f "$DETECTED" ]] || { trlog "no $DETECTED — running 01_preflight.sh first ..."; bash "$HERE/01_preflight.sh"; }
source "$DETECTED"
[[ -n "${IB_DEVICES:-}" ]] || { trerr "IB_DEVICES unresolved after preflight"; exit 1; }

MOUNTS="$(container_mounts)"
CG="$(cuda_graph_arg)"
# peermem path: empty => default (peermem); set => prefix WITH_NVIDIA_PEERMEM=<val> (dmabuf when 0).
PEERMEM_PREFIX=""; [[ -n "${WITH_NVIDIA_PEERMEM:-}" ]] && PEERMEM_PREFIX="WITH_NVIDIA_PEERMEM=$WITH_NVIDIA_PEERMEM "
STEP="srun --jobid=$ALLOC_JOB --overlap --export=ALL"
trlog "RDMA: IB_DEVICES=$IB_DEVICES  peermem=${WITH_NVIDIA_PEERMEM:-<unset>}  alloc=$ALLOC_JOB"

# --- prefill (with KV bootstrap server) ---
trlog "launching prefill on $PREFILL_NODE (TP$TP, cuda_graph=$([[ -z $CG ]] && echo on || echo off)) ..."
: > "$LOG_DIR/prefill.log"
nohup $STEP -N1 -w "$PREFILL_NODE" --gres=gpu:$GPUS_PER_NODE \
  --container-image="$SQSH" --container-mounts="$MOUNTS" \
  bash -c "${PEERMEM_PREFIX}python3 -m sglang.launch_server \
    --model-path $MODEL_DIR --served-model-name $SERVED_NAME --tp $TP \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --disaggregation-mode prefill --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
    --disaggregation-ib-device $IB_DEVICES $CG > $LOG_DIR/prefill.log 2>&1" >/dev/null 2>&1 &

# --- decode ---
trlog "launching decode on $DECODE_NODE (TP$TP) ..."
: > "$LOG_DIR/decode.log"
nohup $STEP -N1 -w "$DECODE_NODE" --gres=gpu:$GPUS_PER_NODE \
  --container-image="$SQSH" --container-mounts="$MOUNTS" \
  bash -c "${PEERMEM_PREFIX}python3 -m sglang.launch_server \
    --model-path $MODEL_DIR --served-model-name $SERVED_NAME --tp $TP \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --disaggregation-mode decode \
    --disaggregation-ib-device $IB_DEVICES $CG > $LOG_DIR/decode.log 2>&1" >/dev/null 2>&1 &

trlog "waiting for both servers to be healthy (cuda-graph capture can take several minutes) ..."
wait_health "http://$PREFILL_IP:$PORT/health" 1800 "$ALLOC_JOB" || { trerr "prefill never healthy; see $LOG_DIR/prefill.log"; exit 1; }
trlog "prefill healthy."
wait_health "http://$DECODE_IP:$PORT/health" 1800 "$ALLOC_JOB" || { trerr "decode never healthy; see $LOG_DIR/decode.log"; exit 1; }
trlog "decode healthy."

# --- router: another overlap step on the prefill node. MUST mount the model dir so the
#     tokenizer loads locally (else 404s to HF). ---
trlog "launching sgl-router on $PREFILL_NODE:$ROUTER_PORT ..."
: > "$LOG_DIR/router.log"
nohup $STEP -N1 -w "$PREFILL_NODE" \
  --container-image="$SQSH" --container-mounts="$(container_mounts "$REPO_ROOT:/inferencex:ro")" \
  bash -c "python3 -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://$PREFILL_IP:$PORT $BOOTSTRAP_PORT --decode http://$DECODE_IP:$PORT \
    --host 0.0.0.0 --port $ROUTER_PORT --policy random > $LOG_DIR/router.log 2>&1" >/dev/null 2>&1 &

wait_health "http://$PREFILL_IP:$ROUTER_PORT/health" 120 "$ALLOC_JOB" || { trerr "router never healthy; see $LOG_DIR/router.log"; exit 1; }

cat > "$STATE" <<EOF
ALLOC_JOB=$ALLOC_JOB
ENDPOINT=http://$PREFILL_IP:$ROUTER_PORT
EOF
trlog "ENDPOINT READY: http://$PREFILL_IP:$ROUTER_PORT  (state: $STATE)"
trlog "next: bash 20_benchmark.sh   |   teardown: bash teardown.sh"
