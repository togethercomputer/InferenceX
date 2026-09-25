#!/usr/bin/env bash
# Storage locations, shared by both front ends.
#
# Defaults are deliberately generic so a fresh clone works anywhere. Real boxes
# differ (big arrays live in different places), so per-box values belong in
#   together_runner/paths.local.sh
# which is gitignored and sourced first. Example:
#
#   TR_HF_CACHE=/mnt/data/$USER/hf-cache
#   TR_WORK_DIR=/mnt/data/$USER/inferencex-work
#
# Never put a box-specific absolute path in a committed file.

_TR_PATHS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
[[ -f "$_TR_PATHS_ROOT/paths.local.sh" ]] && source "$_TR_PATHS_ROOT/paths.local.sh"

export TR_HF_CACHE="${TR_HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
export TR_MODELS_ROOT="${TR_MODELS_ROOT:-$HOME/models}"
export TR_VLLM_CACHE="${TR_VLLM_CACHE:-$HOME/.cache/tr-vllm}"
export TR_TRITON_CACHE="${TR_TRITON_CACHE:-$HOME/.cache/tr-triton}"
export TR_FLASHINFER_CACHE="${TR_FLASHINFER_CACHE:-$HOME/.cache/tr-flashinfer}"
export TR_WORK_DIR="${TR_WORK_DIR:-/tmp/inferencex-work}"
