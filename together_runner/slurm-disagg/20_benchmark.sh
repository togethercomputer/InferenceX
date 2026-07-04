#!/usr/bin/env bash
# Concurrency sweep against the live disagg endpoint using the InferenceX unified
# client (utils/bench_serving/benchmark_serving.py). Reuses one warm router.
# Saves a result JSON per concurrency under RESULTS_DIR and prints a summary table.
#
#   bash 20_benchmark.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"; source "$HERE/disagg_lib.sh"
load_resolved
STATE="$LOG_DIR/disagg_state.env"
[[ -f "$STATE" ]] || { trerr "no state file ($STATE) — run 10_launch.sh first"; exit 1; }
source "$STATE"
alloc_alive "${ALLOC_JOB:-}" || { trerr "allocation $ALLOC_JOB gone — relaunch"; exit 1; }
mkdir -p "$RESULTS_DIR"

curl -sf -m4 "$ENDPOINT/health" >/dev/null || { trerr "endpoint $ENDPOINT not healthy"; exit 1; }
SEQTAG="$(( ISL/1024 ))k$(( OSL/1024 ))k"
trlog "sweep CONC_LIST='$CONC_LIST' at ISL=$ISL OSL=$OSL via $ENDPOINT"

# Run the bench client inside the image (mount repo at /inferencex + model for tokenizer),
# as an overlap step on the prefill allocation. One srun does the whole sweep.
nohup srun --jobid="$ALLOC_JOB" --overlap --nodelist="$PREFILL_NODE" \
  --container-image="$SQSH" --container-mounts="$(container_mounts "$REPO_ROOT:/inferencex:ro")" \
  bash -c "
    pip install -q datasets pandas >/dev/null 2>&1 || true
    for C in $CONC_LIST; do
      NP=\$(( C * $PROMPTS_PER_CONC )); [ \$NP -lt 160 ] && NP=160
      echo \"############ conc=\$C num_prompts=\$NP ############\"
      python3 /inferencex/utils/bench_serving/benchmark_serving.py \
        --backend sglang --model $SERVED_NAME --tokenizer $MODEL_DIR \
        --base-url $ENDPOINT --endpoint /v1/completions \
        --dataset-name random --random-input-len $ISL --random-output-len $OSL \
        --max-concurrency \$C --num-prompts \$NP --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir $RESULTS_DIR \
        --result-filename ${SERVED_NAME}_${SEQTAG}_conc\${C}.bench.json 2>&1 \
        | grep -E 'Successful requests|Total Token throughput|Output token throughput|Median TTFT|P99 TTFT|Median TPOT|Median E2EL'
      echo \"=== conc=\$C done ===\"
    done
    echo ALL_SWEEP_DONE" 2>&1 | tee "$LOG_DIR/sweep.log"

# Summary table from the result JSONs.
trlog "==== SWEEP SUMMARY ($SEQTAG) ===="
python3 - "$RESULTS_DIR" "$SERVED_NAME" "$SEQTAG" $CONC_LIST <<'PY'
import json, sys, os
rdir, name, seqtag = sys.argv[1], sys.argv[2], sys.argv[3]
concs = sys.argv[4:]
hdr = f"{'conc':>5} {'ok':>11} {'total tok/s':>12} {'out tok/s':>10} {'mTPOT ms':>9} {'mTTFT ms':>9} {'p99TTFT ms':>11}"
print(hdr); print('-'*len(hdr))
for C in concs:
    f = os.path.join(rdir, f"{name}_{seqtag}_conc{C}.bench.json")
    if not os.path.exists(f): print(f"{C:>5}  (missing)"); continue
    d = json.load(open(f))
    ok = f"{d.get('completed')}/{d.get('num_prompts','?')}"
    print(f"{C:>5} {ok:>11} {d.get('total_token_throughput',0):>12.0f} {d.get('output_throughput',0):>10.0f} "
          f"{d.get('median_tpot_ms',0):>9.1f} {d.get('median_ttft_ms',0):>9.0f} {d.get('p99_ttft_ms',0):>11.0f}")
PY
trlog "raw results: $RESULTS_DIR  | sweep log: $LOG_DIR/sweep.log"
