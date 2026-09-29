#!/usr/bin/env bash
# Preflight: auto-detect + verify the RDMA/KV path BEFORE launching servers, so a
# new cluster fails in seconds with a clear reason instead of after a multi-minute
# server startup. Resolves IB_DEVICES (GPU<->NIC topology) and the
# WITH_NVIDIA_PEERMEM decision (peermem vs dmabuf), checks IB port state on both
# nodes, and verifies the /dev/infiniband bind-mount + dmabuf inside the container.
# Writes the resolved values to $LOG_DIR/disagg_detected.env (sourced by 10_launch.sh).
#
#   bash 01_preflight.sh            # detect + check, write detected.env
#   PROBE_MOONCAKE=1 bash 01_preflight.sh   # + heavy mooncake register_memory probe
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"; source "$HERE/disagg_lib.sh"
load_nodes   # nodes/ALLOC_JOB only — re-derive IB/peermem from clean config each run
alloc_alive "${ALLOC_JOB:-}" || { trerr "no live allocation — run 00_setup.sh first"; exit 1; }
DETECTED="$LOG_DIR/disagg_detected.env"
SR="srun --jobid=$ALLOC_JOB --overlap --export=ALL -t 5"
fail=0

# --- 1. IB ports ACTIVE/LinkUp on BOTH nodes (host-side, fast) ---
trlog "checking IB port state on $PREFILL_NODE,$DECODE_NODE ..."
$SR -N2 --ntasks-per-node=1 --nodelist="$PREFILL_NODE,$DECODE_NODE" bash -c '
  n=0; act=0
  for d in /sys/class/infiniband/*; do
    [ -e "$d/ports/1/state" ] || continue; n=$((n+1))
    s=$(cat "$d/ports/1/state"); p=$(cat "$d/ports/1/phys_state")
    [[ "$s" == *ACTIVE* && "$p" == *LinkUp* ]] && act=$((act+1))
  done
  echo "[$(hostname)] IB HCAs: $act/$n ACTIVE+LinkUp"
  [ "$act" -gt 0 ] || { echo "[$(hostname)] ERROR: no ACTIVE IB port"; exit 1; }
' || { trerr "IB port check failed"; fail=1; }

# --- 2. Resolve IB_DEVICES (topology) + container RDMA check on prefill node ---
CHK_OUT="$LOG_DIR/.preflight_chk.out"
trlog "detecting GPU<->NIC topology + checking RDMA inside container on $PREFILL_NODE ..."
$SR -N1 --nodelist="$PREFILL_NODE" --gres=gpu:$GPUS_PER_NODE \
  --container-image="$SQSH" --container-mounts="$(container_mounts "$REPO_ROOT:/inferencex:ro")" \
  bash -c "
    echo '### DETECT ###'
    python3 /inferencex/together_runner/slurm-disagg/_detect_rdma.py
    echo '### CHECK ###'
    python3 /inferencex/together_runner/slurm-disagg/_check_container_rdma.py ${PROBE_MOONCAKE:+--mooncake}
  " 2>&1 | tee "$CHK_OUT" || { trerr "container RDMA check failed (see above)"; fail=1; }

# Pull machine-readable values out of the captured output.
DET_IB="$(grep -m1 '^IB_DEVICES=' "$CHK_OUT" | cut -d= -f2- || true)"
DET_LL="$(grep -m1 '^IB_LINK_LAYER=' "$CHK_OUT" | cut -d= -f2- || true)"
DMABUF="$(grep -m1 '^DMABUF_SUPPORTED=' "$CHK_OUT" | cut -d= -f2- || true)"

# Honor an explicit override; else use the detected list.
IB_FINAL="${IB_DEVICES:-$DET_IB}"
[[ -n "$IB_FINAL" ]] || { trerr "could not resolve IB_DEVICES (set it explicitly in config.env)"; fail=1; }
[[ -n "${IB_DEVICES:-}" && -n "$DET_IB" && "$IB_DEVICES" != "$DET_IB" ]] && \
  trlog "NOTE: explicit IB_DEVICES ($IB_DEVICES) differs from detected ($DET_IB) — using explicit."

# --- 3. peermem vs dmabuf decision (host-side: module presence + driver version) ---
PEERMEM_FINAL="${WITH_NVIDIA_PEERMEM:-}"
if [[ -z "$PEERMEM_FINAL" ]]; then
  trlog "deciding KV mem-registration path (nvidia_peermem vs dmabuf) ..."
  DEC="$($SR -N1 --nodelist="$PREFILL_NODE" bash -c '
    drv=$(cat /sys/module/nvidia/version 2>/dev/null); maj=${drv%%.*}
    if modinfo nvidia_peermem >/dev/null 2>&1; then echo "peermem $drv"; else echo "dmabuf $drv $maj"; fi')"
  read -r MODE DRV MAJ <<<"$DEC"
  if [[ "$MODE" == "peermem" ]]; then
    trlog "nvidia_peermem AVAILABLE (driver $DRV) — using default peermem path (WITH_NVIDIA_PEERMEM unset)."
  else
    if [[ "${MAJ:-0}" -ge 535 ]]; then
      PEERMEM_FINAL=0
      trlog "nvidia_peermem ABSENT (driver $DRV ≥535) — forcing dmabuf (WITH_NVIDIA_PEERMEM=0)."
      [[ "$DMABUF" == "1" ]] || { trerr "dmabuf chosen but libibverbs lacks ibv_reg_dmabuf_mr — KV transfer will fail"; fail=1; }
    else
      trerr "nvidia_peermem ABSENT and driver $DRV <535 (no dmabuf) — KV transfer cannot register GPU mem"; fail=1
    fi
  fi
else
  trlog "WITH_NVIDIA_PEERMEM explicitly set to '$PEERMEM_FINAL' — honoring it."
fi

# --- 4. write the resolved truth source ---
if [[ "$fail" == "0" ]]; then
  mkdir -p "$LOG_DIR"
  cat > "$DETECTED" <<EOF
# Auto-resolved by 01_preflight.sh — sourced by 10_launch.sh. Re-run preflight to refresh.
IB_DEVICES=$IB_FINAL
IB_LINK_LAYER=${DET_LL:-unknown}
WITH_NVIDIA_PEERMEM=$PEERMEM_FINAL
EOF
  rm -f "$CHK_OUT"
  trlog "PREFLIGHT OK → $DETECTED"
  trlog "  IB_DEVICES=$IB_FINAL"
  trlog "  WITH_NVIDIA_PEERMEM=${PEERMEM_FINAL:-<unset:peermem>}"
  trlog "next: bash 10_launch.sh"
else
  trerr "PREFLIGHT FAILED — fix the above before launching (detected.env NOT written)."; exit 1
fi
