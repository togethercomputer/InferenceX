#!/usr/bin/env bash
# Weight externalization: download model weights ONCE to a stable local dir so
# server startup does zero downloading. Idempotent — safe to re-run.
#
#   bash prestage_weights.sh
#
# On success writes a .ready marker and prints the MODEL_PATH to export. The
# launch script auto-detects $MODELS_ROOT/$PROFILE/.ready and skips download.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
source "$HERE/sglang_lib.sh"

check_env_vars MODEL PROFILE MODELS_ROOT || exit 1

if [[ "$MODEL" == /* ]]; then
    trlog "MODEL is already a local path ($MODEL) — nothing to download."
    exit 0
fi

DEST="$MODELS_ROOT/$PROFILE"
mkdir -p "$DEST"

if [[ -f "$DEST/.ready" ]]; then
    trlog "Already staged: $DEST (.ready present). Re-verifying with hf download (idempotent)..."
fi

# hf CLI: prefer host install, else run inside the container image.
run_hf() {
    if command -v hf >/dev/null 2>&1; then
        HF_TOKEN="${HF_TOKEN:-}" hf "$@"
    else
        docker run --rm \
            -e HF_TOKEN="${HF_TOKEN:-}" \
            -v "$MODELS_ROOT:$MODELS_ROOT" \
            -v "$HF_CACHE:/root/.cache/huggingface" \
            "$IMAGE" hf "$@"
    fi
}

run_timer_start
stage_begin "weight-download" 0
trlog "Downloading $MODEL@$MODEL_REVISION -> $DEST"
trlog "(this is the slow step; doing it here keeps every later launch fast)"

# --local-dir gives a stable, self-contained path; hf download is resumable.
run_hf download "$MODEL" --revision "$MODEL_REVISION" --local-dir "$DEST"

stage_end

# Stamp readiness with provenance.
{
    echo "model=$MODEL"
    echo "revision=$MODEL_REVISION"
    echo "staged_by=$_TR_HOST"
    echo "staged_seconds=$(run_elapsed)"
} > "$DEST/.ready"

trlog "DONE in $(_fmt_dur "$(run_elapsed)"). Staged at: $DEST"
echo
echo "To use the pre-staged weights (zero-download launch), export:"
echo "    export MODEL_PATH=\"$DEST\""
echo "run_all.sh / run_sglang_2 auto-detect this via the .ready marker."
