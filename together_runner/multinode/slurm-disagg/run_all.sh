#!/usr/bin/env bash
# One-shot end-to-end: setup (idempotent) -> launch -> benchmark sweep.
# Leaves the endpoint running; run teardown.sh when done.
#
#   bash run_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$HERE/00_setup.sh"
bash "$HERE/01_preflight.sh"
bash "$HERE/10_launch.sh"
bash "$HERE/20_benchmark.sh"
echo
echo "End-to-end done. Endpoint still up — see $(. "$HERE/config.env"; echo "$LOG_DIR/disagg_state.env")."
echo "Teardown: bash $HERE/teardown.sh"
