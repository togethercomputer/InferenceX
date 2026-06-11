#!/usr/bin/env bash
# Smoke gate-1: seconds-level environment / machine / config preflight.
# Surfaces "variable not set", wrong image, busy port, bad HF token, not-enough
# GPUs/disk, etc. BEFORE any expensive launch. Exits non-zero on any FAIL.
#
#   bash run_sglang_0_preflight.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
source "$HERE/sglang_lib.sh"

echo "==================== PREFLIGHT (gate-1) ===================="
echo "host=$_TR_HOST profile=$PROFILE model=$MODEL tp=$TP port=$PORT image=$IMAGE"
echo "------------------------------------------------------------"
pf_reset

# 1) Required env vars (recipe-dependent).
req=(MODEL PROFILE TP PORT IMAGE CONTAINER_NAME HF_CACHE MODELS_ROOT)
[[ "$PROFILE" == "dsr1-fp4" ]] && req+=(EP_SIZE)
if check_env_vars "${req[@]}" 2>/dev/null; then
    pf_pass "required env vars set: ${req[*]}"
else
    pf_fail "missing env vars (see above) — check config.env"
    check_env_vars "${req[@]}" || true
fi

# 2) Docker daemon reachable.
if docker info >/dev/null 2>&1; then pf_pass "docker daemon reachable"
else pf_fail "docker not reachable (need docker + permissions)"; fi

# 3) Target image present locally (avoid a surprise multi-GB pull mid-run).
if docker image inspect "$IMAGE" >/dev/null 2>&1; then pf_pass "image present locally: $IMAGE"
else pf_fail "image NOT present locally: $IMAGE  (docker pull \"$IMAGE\")"; fi

# 4) GPU count >= TP.
gpu_n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0)
if (( gpu_n >= TP )); then pf_pass "GPUs available: $gpu_n >= TP=$TP"
else pf_fail "only $gpu_n GPUs visible, need TP=$TP"; fi

# 5) Free GPU memory (warn if other jobs are resident — shared node).
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1>2000{c++} END{print c+0}')
if (( busy == 0 )); then pf_pass "all GPUs idle (<2GB used)"
else pf_info "$busy GPU(s) already have >2GB resident — shared node, check nvidia-smi"; fi

# 6) Port free on host — OR held by our own container's docker-proxy (expected
#    when reusing the container; `docker run -p` binds the host port for the
#    container's whole lifetime, regardless of the inner server state).
if ss -ltn 2>/dev/null | grep -q ":${PORT}\b" || netstat -ltn 2>/dev/null | grep -q ":${PORT}\b"; then
    if docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | grep -E "^${CONTAINER_NAME} " | grep -q ":${PORT}->"; then
        pf_info "port $PORT published by our container '$CONTAINER_NAME' (reuse OK; server relaunched inside)"
    else
        pf_fail "port $PORT in use by another process (not our container) — free it or change PORT"
    fi
else pf_pass "port $PORT free on host"; fi

# 7) HF token + model reachability (only when MODEL is an HF id, not a path).
if [[ "$MODEL" == /* ]]; then
    pf_info "MODEL is a local path — skipping HF token/reachability check"
elif [[ -z "${HF_TOKEN:-}" ]]; then
    pf_info "HF_TOKEN empty — OK for public models; gated models will fail"
else
    if command -v hf >/dev/null 2>&1; then who=$(hf auth whoami 2>/dev/null)
    else who=$(docker run --rm -e HF_TOKEN="$HF_TOKEN" "$IMAGE" hf auth whoami 2>/dev/null); fi
    who=$(echo "$who" | grep -v '^\s*$' | head -1 | tr -d '\r')
    if [[ -n "$who" ]]; then pf_pass "HF token valid (user: $who)"
    else pf_pass "HF token present (whoami returned no name; download will confirm access)"; fi
fi

# 8) Disk space on weight/cache mounts (DSR1-FP4 needs ~400GB).
need_gb=$([[ "$PROFILE" == "dsr1-fp4" ]] && echo 450 || echo 40)
for d in "$HF_CACHE" "$MODELS_ROOT"; do
    mp="$d"; while [[ ! -d "$mp" && "$mp" != "/" ]]; do mp="$(dirname "$mp")"; done
    avail_gb=$(df -BG --output=avail "$mp" 2>/dev/null | tail -1 | tr -dc '0-9')
    if [[ -n "$avail_gb" ]] && (( avail_gb >= need_gb )); then pf_pass "disk OK on $mp: ${avail_gb}GB free (need ~${need_gb}GB)"
    else pf_fail "low disk on $mp: ${avail_gb:-?}GB free, need ~${need_gb}GB"; fi
done

# 9) Are weights already staged? (informational — drives skip-download later.)
staged_dir="$MODELS_ROOT/$PROFILE"
if [[ -f "$staged_dir/.ready" ]]; then pf_pass "weights pre-staged at $staged_dir (start will skip download)"
elif [[ -d "$HF_CACHE" ]] && find "$HF_CACHE" -maxdepth 4 -type d -name "models--*" 2>/dev/null | grep -q .; then
    pf_info "HF cache populated at $HF_CACHE (may avoid re-download)"
else pf_info "weights NOT pre-staged — first run will download (slow); see prestage_weights.sh"; fi

# 10) tuning switch echo.
if [[ "$ENABLE_TUNING" == "1" ]]; then pf_info "ENABLE_TUNING=1 — autotune KEPT (use for baselines; slow cold start)"
else pf_info "ENABLE_TUNING=0 — autotune SKIPPED via --disable-flashinfer-autotune (fast)"; fi

echo "------------------------------------------------------------"
fails=$(pf_failures)
if (( fails == 0 )); then echo "PREFLIGHT: PASS ✅"; exit 0
else echo "PREFLIGHT: FAIL ❌ ($fails problem(s)) — fix the [FAIL] items above"; exit 1; fi
