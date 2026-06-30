#!/usr/bin/env bash
# One-time (per pod boot) setup: grab the 2-node allocation, apply ephemeral node fixes,
# import the image, prestage weights. Idempotent — safe to re-run (reuses a live
# allocation). Needs sudo for the enroot nvidia-hook patch.
#
#   bash 00_setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"; source "$HERE/disagg_lib.sh"
load_resolved                       # reuse a persisted live allocation if present
ensure_allocation || exit 1         # ALLOC_JOB + PREFILL_NODE/DECODE_NODE now set
STEP="srun --jobid=$ALLOC_JOB --overlap --export=ALL"

# 1) Patch the enroot nvidia hook on BOTH nodes (skip the persistenced/fabricmanager
#    sockets that can't be bind-mounted inside the nested pod). Ephemeral; redo per boot.
#    NOTE: this sed-patch of the *system* hook is unavoidable on this stack — a clean
#    user-level override is not possible here (verified on enroot 4.0.1): runtime.sh runs
#    the system AND user hooks.d with no basename dedup, so a user copy can't replace the
#    system 98-nvidia.sh; and pyxis ignores a per-job ENROOT_SYSCONF_PATH redirect.
trlog "patching enroot nvidia hook on $PREFILL_NODE,$DECODE_NODE ..."
$STEP -N2 --ntasks-per-node=1 -w "$PREFILL_NODE,$DECODE_NODE" bash -c '
  set -e; H=$(hostname); F=/etc/enroot/hooks.d/98-nvidia.sh
  if grep -q "no-persistenced" "$F"; then echo "[$H] hook already patched"; exit 0; fi
  sudo -n true 2>/dev/null || { echo "[$H] ERROR: sudo unavailable — cannot patch $F"; exit 1; }
  sudo cp "$F" "$F.bak"
  sudo sed -i "s|cli_args=(\"--no-cgroups\" |cli_args=(\"--no-cgroups\" \"--no-persistenced\" \"--no-fabricmanager\" |" "$F"
  if grep -q "no-persistenced" "$F"; then echo "[$H] hook patched"; else
    echo "[$H] ERROR: patch did not take (hook format changed?) — restoring backup"; sudo cp "$F.bak" "$F"; exit 1
  fi'

# 2) Import the SGLang image to the shared squashfs. enroot import needs a node-local
#    NON-overlay fs for its temp (overlay / can't mknod the overlayfs whiteouts) — the
#    step auto-detects one (honors $ENROOT_SCRATCH if set).
if [[ -f "$SQSH" ]]; then
    trlog "image already imported: $SQSH"
else
    trlog "importing $DOCKER_IMAGE -> $SQSH (auto-detect ext4 temp; multi-GB, ~minutes) ..."
    mkdir -p "$ENROOT_DIR"
    $STEP -N1 -w "$PREFILL_NODE" bash -c '
      set -e
      S=""
      for c in "$ENROOT_SCRATCH" /scratch /raid /mnt/local /mnt/resource /var/tmp /tmp; do
        [ -n "$c" ] || continue
        [ -d "$c" ] || mkdir -p "$c" 2>/dev/null || continue
        t=$(stat -f -c %T "$c" 2>/dev/null)
        case "$t" in overlayfs|overlay|tmpfs|"") continue;; esac
        mkdir -p "$c/enroot" 2>/dev/null || continue
        S="$c/enroot"; break
      done
      [ -n "$S" ] || { echo "[$(hostname)] ERROR: no node-local non-overlay scratch (set ENROOT_SCRATCH)"; exit 1; }
      export ENROOT_CACHE_PATH="$S/cache" ENROOT_TEMP_PATH="$S/tmp" TMPDIR="$S/tmp"
      mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_TEMP_PATH"
      echo "[$(hostname)] enroot temp on $S ($t)"
      enroot import -o "$SQSH" "$DOCKER_IMAGE"'
fi

# 3) Prestage weights to the shared FS (zero-download launches).
if [[ -f "$MODEL_DIR/config.json" ]]; then
    trlog "weights present: $MODEL_DIR"
else
    trlog "downloading $MODEL_HF_ID -> $MODEL_DIR ..."
    mkdir -p "$MODEL_DIR"
    $STEP -N1 -w "$DECODE_NODE" \
      --container-image="$SQSH" \
      --container-mounts="$HF_CACHE:/root/.cache/huggingface,$MODELS_ROOT:$MODELS_ROOT" \
      bash -c 'export HF_TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null)
               hf download "$MODEL_HF_ID" --local-dir "$MODEL_DIR"'
fi
trlog "setup complete. (allocation $ALLOC_JOB held; next: bash 01_preflight.sh)"
