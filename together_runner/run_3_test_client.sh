#!/usr/bin/env bash
# Step 3/3 — Validate the running server and (optionally) benchmark it, then
# emit an InferenceX-schema result JSON under results/<hw>/<cluster>/<host>/<date>/.
#   bash run_3_test_client.sh            # health + one chat completion
#   BENCH=1 bash run_3_test_client.sh    # also run bench_serving + emit result
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
source "$HERE/bench_lib.sh"

check_env_vars ENGINE CONTAINER_NAME PORT MODEL ISL OSL CONC HW CLUSTER PROFILE || exit 1
BASE="http://localhost:${PORT}"

trlog "== Health =="
curl -sf "${BASE}/health" >/dev/null && pf_pass "healthy" || { trerr "server not healthy on ${BASE}"; exit 1; }

trlog "== Chat completion =="
curl -s "${BASE}/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what is a B200 GPU?\"}],\"max_tokens\":64,\"temperature\":0}"
echo

if [[ "${BENCH:-0}" != "1" ]]; then
    trlog "done (set BENCH=1 to run the serving benchmark + emit a result)."
    exit 0
fi

# ---------------------------------------------------------------------------
# Benchmark + result emission
# ---------------------------------------------------------------------------
DATE="$(date +%F)"
TUNE=$([[ "$ENABLE_TUNING" == "1" ]] && echo tuned || echo untuned)
OUTDIR="$HERE/results/$HW/$CLUSTER/$_TR_HOST/$DATE"
mkdir -p "$OUTDIR"
NAME="${ENGINE}_${PROFILE}_${SEQ_TAG}_${TUNE}_conc${CONC}"
RAW_IN="/root/${NAME}.bench.json"          # raw bench output (inside container)
RAW_HOST="$OUTDIR/${NAME}.bench.json"      # copied to host
RESULT="$OUTDIR/${NAME}.json"              # InferenceX-schema result
GPU_CSV="$OUTDIR/${NAME}.gpu.csv"

trlog "== Benchmark (engine=$ENGINE ISL=$ISL OSL=$OSL conc=$CONC) =="
# GPU power/clock sampling on the host (for energy metrics).
nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.current.sm,utilization.gpu \
    --format=csv -l 1 > "$GPU_CSV" 2>/dev/null &
GPU_MON_PID=$!

# Unified client: the InferenceX vendored benchmark_serving.py (mounted at
# /inferencex), run inside the container against the OpenAI-compatible server.
# Same tool for sglang & vllm => identical metric definitions => fair comparison.
docker exec "$CONTAINER_NAME" bash -lc "pip install -q datasets pandas >/dev/null 2>&1 || true
python3 /inferencex/utils/bench_serving/benchmark_serving.py \
    --backend $ENGINE --model '$MODEL' \
    --base-url http://127.0.0.1:$PORT --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len $ISL --random-output-len $OSL --random-range-ratio 0 \
    --max-concurrency $CONC --num-prompts $(( CONC * 10 )) \
    --num-warmups $(( 2 * CONC )) --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir /root --result-filename '${NAME}.bench.json'"

kill "$GPU_MON_PID" 2>/dev/null || true
docker cp "$CONTAINER_NAME:$RAW_IN" "$RAW_HOST" 2>/dev/null || true

# Map raw bench output -> InferenceX-schema result JSON (compare.py emit).
PRECISION=$([[ "$PROFILE" == *fp4* ]] && echo fp4 || echo fp8)
python3 "$HERE/compare.py" emit \
    --raw "$RAW_HOST" --out "$RESULT" --gpu-csv "$GPU_CSV" \
    --hw "$HW" --cluster "$CLUSTER" --model "$MODEL" --framework "$ENGINE" --precision "$PRECISION" \
    --profile "$PROFILE" \
    --isl "$ISL" --osl "$OSL" --tp "$TP" --ep "${EP_SIZE:-0}" --conc "$CONC" \
    --image "$IMAGE" --tuning "$ENABLE_TUNING" --host "$_TR_HOST"

trlog "result written: $RESULT"
trlog "stop server:  docker exec $CONTAINER_NAME pkill -f '$(server_proc_pat)'"
trlog "remove all:   docker rm -f $CONTAINER_NAME"
