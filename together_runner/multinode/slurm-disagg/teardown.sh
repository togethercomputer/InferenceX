#!/usr/bin/env bash
# Tear down the disagg endpoint: cancel the 2-node allocation (its prefill/decode/router
# steps die with it). Removes the node + state files so the next run re-allocates fresh.
# (image, weights, detected RDMA config preserved.)
#
#   bash teardown.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"; source "$HERE/disagg_lib.sh"
load_resolved

JOB="${ALLOC_JOB:-}"
[[ -z "$JOB" ]] && JOB="$(squeue --me -h -n "$ALLOC_NAME" -o '%i' 2>/dev/null | head -1)"
if [[ -n "$JOB" ]]; then
    trlog "scancel allocation $JOB"; scancel "$JOB" 2>/dev/null || true
else
    trlog "no allocation found to cancel."
fi
rm -f "$LOG_DIR/disagg_state.env" "$LOG_DIR/disagg_nodes.env"
trlog "torn down. (image, weights, detected.env preserved)"
