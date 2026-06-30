#!/usr/bin/env bash
# Shared helpers for slurm-disagg. Source AFTER config.env.

trlog()  { echo "[disagg $(date +%H:%M:%S)] $*"; }
trerr()  { echo "[disagg $(date +%H:%M:%S)] ERROR: $*" >&2; }

# Source persisted node/allocation resolution (ALLOC_JOB + nodes + IPs).
load_nodes() { [[ -f "$LOG_DIR/disagg_nodes.env" ]] && source "$LOG_DIR/disagg_nodes.env"; return 0; }

# load_nodes + the detected RDMA config (IB_DEVICES + WITH_NVIDIA_PEERMEM). Used by
# launch/benchmark. NOT used by 01_preflight (which *produces* detected.env and must
# re-derive from clean config defaults each run).
load_resolved() {
    load_nodes
    [[ -f "$LOG_DIR/disagg_detected.env" ]] && source "$LOG_DIR/disagg_detected.env"
    return 0
}

# Routable IP for a Slurm node: hostname resolution first, then Slurm's NodeAddr.
ip_of() {
    local ip; ip="$(getent hosts "$1" 2>/dev/null | awk '{print $1; exit}')"
    [[ -z "$ip" ]] && ip="$(scontrol show node "$1" 2>/dev/null | grep -oE 'NodeAddr=[^ ]+' | cut -d= -f2)"
    echo "$ip"
}

# True if Slurm job $1 is currently allocated (pending/running/configuring).
alloc_alive() { [[ -n "${1:-}" ]] && squeue -h -j "$1" -o '%t' 2>/dev/null | grep -qE 'R|PD|CF'; }

# Pick PARTITION if unset: first partition with >=2 fully-idle GPU nodes.
resolve_partition() {
    [[ -n "$PARTITION" ]] && return 0
    declare -A cnt
    local node part gres st g
    while IFS='|' read -r node part gres st; do
        part="${part%\*}"; [[ "$st" == "idle" ]] || continue
        g="$(grep -oE 'gpu(:[^:,]+)*:[0-9]+' <<<"$gres" | grep -oE '[0-9]+$' | head -1)"
        [[ -n "$g" && "$g" -gt 0 ]] && cnt[$part]=$(( ${cnt[$part]:-0} + 1 ))
    done < <(sinfo -h -N -o '%N|%P|%G|%t' 2>/dev/null)
    for part in "${!cnt[@]}"; do [[ ${cnt[$part]} -ge 2 ]] && { PARTITION="$part"; break; }; done
    [[ -n "$PARTITION" ]] || { trerr "no partition with >=2 idle GPU nodes (set PARTITION)"; return 1; }
    trlog "auto-selected PARTITION=$PARTITION"
}

# Ensure a live 2-node allocation; set ALLOC_JOB/PREFILL_NODE/DECODE_NODE/IPs and persist
# to $LOG_DIR/disagg_nodes.env. Idempotent: reuses the persisted allocation if still alive.
ensure_allocation() {
    if alloc_alive "${ALLOC_JOB:-}"; then trlog "reusing allocation $ALLOC_JOB"; else
        resolve_partition || return 1
        local nlflag=""
        [[ -n "$PREFILL_NODE" && -n "$DECODE_NODE" ]] && nlflag="--nodelist=$PREFILL_NODE,$DECODE_NODE"
        trlog "requesting 2-node allocation on '$PARTITION' (gpu:$GPUS_PER_NODE/node, t=$ALLOC_TIME) ..."
        local out; out="$(salloc -N2 --gres=gpu:"$GPUS_PER_NODE" -p "$PARTITION" --no-shell \
            -J "$ALLOC_NAME" --time="$ALLOC_TIME" $nlflag 2>&1)" || { trerr "salloc failed: $out"; return 1; }
        ALLOC_JOB="$(grep -oE 'job allocation [0-9]+' <<<"$out" | grep -oE '[0-9]+' | head -1)"
        [[ -n "$ALLOC_JOB" ]] || { trerr "could not parse job id from: $out"; return 1; }
    fi
    # Discover the assigned nodes from the allocation (Slurm-ordered).
    local nl; nl="$(squeue -h -j "$ALLOC_JOB" -o '%N' 2>/dev/null)"
    mapfile -t NODES < <(scontrol show hostnames "$nl" 2>/dev/null)
    [[ ${#NODES[@]} -ge 2 ]] || { trerr "allocation $ALLOC_JOB has <2 nodes ($nl)"; return 1; }
    PREFILL_NODE="${PREFILL_NODE:-${NODES[0]}}"
    DECODE_NODE="${DECODE_NODE:-${NODES[1]}}"
    PREFILL_IP="$(ip_of "$PREFILL_NODE")"; DECODE_IP="$(ip_of "$DECODE_NODE")"
    [[ -n "$PREFILL_IP" && -n "$DECODE_IP" ]] || { trerr "could not resolve node IPs ($PREFILL_NODE/$DECODE_NODE)"; return 1; }

    mkdir -p "$LOG_DIR"
    cat > "$LOG_DIR/disagg_nodes.env" <<EOF
# Auto-resolved by 00_setup.sh (ensure_allocation). teardown.sh removes this.
ALLOC_JOB=$ALLOC_JOB
PARTITION=$PARTITION
PREFILL_NODE=$PREFILL_NODE
DECODE_NODE=$DECODE_NODE
GPUS_PER_NODE=$GPUS_PER_NODE
TP=$TP
PREFILL_IP=$PREFILL_IP
DECODE_IP=$DECODE_IP
EOF
    export ALLOC_JOB PARTITION PREFILL_NODE DECODE_NODE GPUS_PER_NODE TP PREFILL_IP DECODE_IP
    trlog "allocation $ALLOC_JOB: prefill=$PREFILL_NODE($PREFILL_IP) decode=$DECODE_NODE($DECODE_IP)"
}

# srun a step INTO the allocation (overlap). Usage: alloc_step <node> [extra srun args] -- <cmd...>
# (kept as a helper; scripts mostly inline srun --jobid=$ALLOC_JOB --overlap for clarity.)

# Common pyxis container-mounts. Pass extra mounts as $1 (comma-prefixed or empty).
# /dev/infiniband is REQUIRED for RDMA-to-pod (KV transfer). HF cache + model + logs.
container_mounts() {
    local extra="${1:-}"
    echo "${MODEL_DIR}:${MODEL_DIR},${LOG_DIR}:${LOG_DIR},${HF_CACHE}:/root/.cache/huggingface,/dev/infiniband:/dev/infiniband${extra:+,$extra}"
}

# Poll an HTTP /health until 200, a process/job dies, or timeout.
# wait_health <url> <timeout_s> [<jobid_to_watch>]
wait_health() {
    local url="$1" timeout="${2:-1800}" watch_job="${3:-}" t0 i=0
    t0=$(date +%s)
    while :; do
        if curl -sf -m4 "$url" >/dev/null 2>&1; then return 0; fi
        if [[ -n "$watch_job" ]] && ! squeue -h -j "$watch_job" >/dev/null 2>&1; then
            trerr "watched job $watch_job exited before $url became healthy"; return 1
        fi
        (( $(date +%s) - t0 > timeout )) && { trerr "timeout waiting for $url"; return 1; }
        sleep 5; (( i++ ))
    done
}

# cuda-graph flag: by default ON (empty); DISABLE_CUDA_GRAPH=1 adds the flag.
cuda_graph_arg() { [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]] && echo "--disable-cuda-graph" || true; }
