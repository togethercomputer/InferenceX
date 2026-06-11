#!/usr/bin/env bash
# Step 3/3 — Validate the running server and (optionally) benchmark it, then
# emit an InferenceX-schema result JSON under results/<hw>/<cluster>/<host>/<date>/.
#   bash run_sglang_3_test_client.sh            # health + one chat completion
#   BENCH=1 bash run_sglang_3_test_client.sh    # also run bench_serving + emit result
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
source "$HERE/sglang_lib.sh"

check_env_vars CONTAINER_NAME PORT MODEL ISL OSL CONC HW CLUSTER PROFILE || exit 1
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
NAME="${PROFILE}_${SEQ_TAG}_${TUNE}_conc${CONC}"
RAW_IN="/root/${NAME}.bench.json"          # raw sglang output (inside container)
RAW_HOST="$OUTDIR/${NAME}.bench.json"      # copied to host
RESULT="$OUTDIR/${NAME}.json"              # InferenceX-schema result
GPU_CSV="$OUTDIR/${NAME}.gpu.csv"

trlog "== Benchmark (ISL=$ISL OSL=$OSL conc=$CONC) =="
# GPU power/clock sampling on the host (for energy metrics).
nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.current.sm,utilization.gpu \
    --format=csv -l 1 > "$GPU_CSV" 2>/dev/null &
GPU_MON_PID=$!

docker exec "$CONTAINER_NAME" python3 -m sglang.bench_serving \
    --backend sglang-oai --model "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --dataset-name random \
    --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 0 \
    --max-concurrency "$CONC" --num-prompts "$(( CONC * 10 ))" \
    --output-file "$RAW_IN"

kill "$GPU_MON_PID" 2>/dev/null || true
docker cp "$CONTAINER_NAME:$RAW_IN" "$RAW_HOST" 2>/dev/null || true

# Map raw bench output -> InferenceX-schema result JSON (compare.py emit).
PRECISION=$([[ "$PROFILE" == *fp4* ]] && echo fp4 || echo fp8)
python3 "$HERE/compare.py" emit \
    --raw "$RAW_HOST" --out "$RESULT" --gpu-csv "$GPU_CSV" \
    --hw "$HW" --cluster "$CLUSTER" --model "$MODEL" --framework sglang --precision "$PRECISION" \
    --profile "$PROFILE" \
    --isl "$ISL" --osl "$OSL" --tp "$TP" --ep "${EP_SIZE:-0}" --conc "$CONC" \
    --image "$IMAGE" --tuning "$ENABLE_TUNING" --host "$_TR_HOST"

trlog "result written: $RESULT"
trlog "stop server:  docker exec $CONTAINER_NAME pkill -f sglang.launch_server"
trlog "remove all:   docker rm -f $CONTAINER_NAME"
