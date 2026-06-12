#!/usr/bin/env bash
# One-click orchestrator for the together_runner SGLang harness.
#
#   bash run_all.sh --smoke      # gate-1 only: machine + env preflight (seconds)
#   bash run_all.sh --full       # end-to-end: preflight -> prestage -> start ->
#                                #   launch(+monitor) -> gate-2 -> bench -> compare
#   bash run_all.sh --baseline   # like --full but ENABLE_TUNING=1, then save the
#                                #   result as the committed baseline
#
# All config comes from config.env (edit it or export overrides first).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
source "$HERE/bench_lib.sh"

MODE="${1:-}"
case "$MODE" in
  --smoke|--full|--baseline) ;;
  *) echo "usage: $0 --smoke | --full | --baseline" >&2; exit 2 ;;
esac

step() { echo; echo "######## $* ########"; }

# ---- gate-1: preflight (all modes) ----
step "GATE-1 PREFLIGHT"
bash "$HERE/run_0_preflight.sh"

if [[ "$MODE" == "--smoke" ]]; then
    trlog "smoke (gate-1) complete — environment looks good."
    exit 0
fi

# --baseline forces tuning on for peak numbers.
if [[ "$MODE" == "--baseline" ]]; then
    export ENABLE_TUNING=1
    trlog "baseline mode: ENABLE_TUNING=1 (autotune kept for peak perf)"
fi

# ---- prestage weights if needed (HF id + not yet staged) ----
if [[ "$MODEL" != /* && ! -f "$MODELS_ROOT/$PROFILE/.ready" && -z "${MODEL_PATH:-}" ]]; then
    step "PRESTAGE WEIGHTS"
    bash "$HERE/prestage_weights.sh"
fi

# ---- start container ----
step "START CONTAINER"
bash "$HERE/run_1_start_container.sh"

# ---- launch server (staged monitor inside) ----
step "LAUNCH SERVER"
bash "$HERE/run_2_launch_server.sh"

# ---- gate-2: minimal real run (health + one completion) ----
step "GATE-2 MINIMAL REAL RUN"
bash "$HERE/run_3_test_client.sh"   # no BENCH -> health + chat only

# ---- full benchmark + emit result ----
step "BENCHMARK"
BENCH=1 bash "$HERE/run_3_test_client.sh"

# ---- locate the just-written result ----
DATE="$(date +%F)"
TUNE=$([[ "$ENABLE_TUNING" == "1" ]] && echo tuned || echo untuned)
RESULT="$HERE/results/$HW/$CLUSTER/$_TR_HOST/$DATE/${ENGINE}_${PROFILE}_${SEQ_TAG}_${TUNE}_conc${CONC}.json"

# ---- baseline mode: install result as the committed baseline ----
if [[ "$MODE" == "--baseline" ]]; then
    # hw-wide golden by default; BASELINE_CLUSTER=1 scopes under <hw>/<cluster>.
    if [[ "${BASELINE_CLUSTER:-0}" == "1" ]]; then
        BDIR="$HERE/baselines/$HW/$CLUSTER/$ENGINE/$PROFILE/$SEQ_TAG/$TUNE"
    else
        BDIR="$HERE/baselines/$HW/$ENGINE/$PROFILE/$SEQ_TAG/$TUNE"
    fi
    mkdir -p "$BDIR"
    cp "$RESULT" "$BDIR/conc${CONC}.json"
    trlog "baseline saved: $BDIR/conc${CONC}.json  (commit this to the repo)"
    exit 0
fi

# ---- compare vs baseline ----
step "COMPARE VS BASELINE"
python3 "$HERE/compare.py" compare --result "$RESULT" --threshold 5
