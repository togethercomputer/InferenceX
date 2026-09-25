#!/usr/bin/env bash
# together_runner shared library: logging, env validation, timers and preflight
# assertion helpers. Vendor- and engine-agnostic.
#
# Source this AFTER config.env. Side-effect free except for defining functions.

# Log line prefix — carries node + rank so the same format works unchanged for
# multi-node (rank defaults to 0 on a single box).
_TR_HOST="$(hostname -s 2>/dev/null || hostname)"
log_prefix() { echo "[node=${_TR_HOST} rank=${RANK:-0}]"; }
trlog()  { echo "$(log_prefix) $*"; }
trwarn() { echo "$(log_prefix) WARN: $*" >&2; }
trerr()  { echo "$(log_prefix) ERROR: $*" >&2; }

# ---------------------------------------------------------------------------
# Env validation (mirrors benchmarks/benchmark_lib.sh:check_env_vars)
# ---------------------------------------------------------------------------
check_env_vars() {
    local missing=()
    local v
    for v in "$@"; do
        if [[ -z "${!v:-}" ]]; then missing+=("$v"); fi
    done
    if (( ${#missing[@]} > 0 )); then
        trerr "missing required environment variables:"
        for v in "${missing[@]}"; do echo "  - $v" >&2; done
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------
_RUN_T0=0
_now() { date +%s; }
_fmt_dur() { local s=$1; printf '%dm%02ds' $(( s/60 )) $(( s%60 )); }
run_timer_start() { _RUN_T0=$(_now); }
run_elapsed()     { echo $(( $(_now) - _RUN_T0 )); }

# ---------------------------------------------------------------------------
# Preflight assertion helpers (used by local/run_0_preflight.sh).
# Each prints a PASS/FAIL line and returns 0/1. Never exit — caller tallies.
# ---------------------------------------------------------------------------
_PF_FAILS=0
pf_pass() { echo "  [PASS] $*"; }
pf_fail() { echo "  [FAIL] $*"; _PF_FAILS=$(( _PF_FAILS + 1 )); }
pf_info() { echo "  [info] $*"; }
pf_reset() { _PF_FAILS=0; }
pf_failures() { echo "$_PF_FAILS"; }
