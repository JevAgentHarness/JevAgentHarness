# shellcheck shell=bash
# Shared helpers for scripts/train/NN_*.sh -- sourced, never executed directly.
#
# Common variables (every script):
#   DRY_RUN=1      print each command (shell-quoted) instead of running it; input files are not required
#   GPUS=N         number of GPUs; N > 1 -> `accelerate launch --num_processes N` for multi-process trainers
#   ACCEL=1        use `accelerate launch` even on one GPU (configs/train/accelerate_1gpu.yaml)
#   ACCEL_CONFIG   override the accelerate config file
#   PY             python interpreter (default: `python` of the active venv, else .venv/)
#   MODEL          base model (HF id or local dir), default Qwen/Qwen3-4B
#   DATA           decision-data directory (default data/decisions)
#   CKPT           checkpoint root (default checkpoints)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$ROOT/.venv/bin/activate" ]]; then
  set +u
  # shellcheck disable=SC1091
  . "$ROOT/.venv/bin/activate"
  set -u
fi

: "${DRY_RUN:=0}"
: "${GPUS:=1}"
: "${ACCEL:=0}"
: "${PY:=python}"
: "${MODEL:=Qwen/Qwen3-4B}"
: "${DATA:=data/decisions}"
: "${CKPT:=checkpoints}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

log() { printf '[%s] %s\n' "$(basename "$0")" "$*" >&2; }

# run CMD...: echo the command, then execute it unless DRY_RUN=1
run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then
    "$@"
  fi
}

# launch MODULE ARGS...: `python -m MODULE` on one device, `accelerate launch -m MODULE` otherwise
launch() {
  local mod=$1
  shift
  if (( GPUS > 1 )); then
    run accelerate launch --config_file "${ACCEL_CONFIG:-$ROOT/configs/train/accelerate_8gpu.yaml}" \
      --num_processes "$GPUS" -m "$mod" "$@"
  elif [[ "$ACCEL" == 1 ]]; then
    run accelerate launch --config_file "${ACCEL_CONFIG:-$ROOT/configs/train/accelerate_1gpu.yaml}" \
      --num_processes 1 -m "$mod" "$@"
  else
    run "$PY" -m "$mod" "$@"
  fi
}

# need PATH...: fail early when an input is missing (skipped under DRY_RUN)
need() {
  if [[ "$DRY_RUN" == 1 ]]; then
    return 0
  fi
  local f
  for f in "$@"; do
    if [[ ! -e "$f" ]]; then
      log "missing input: $f (run the earlier stage first, or set the variable)"
      exit 1
    fi
  done
}

# grad_accum EFF_BATCH PER_DEVICE_BATCH GPUS -> gradient-accumulation steps (>= 1)
grad_accum() {
  local ga=$(( $1 / ($2 * $3) ))
  if (( ga < 1 )); then
    ga=1
  fi
  echo "$ga"
}

# split_list STRING SEP -> one item per line (for ';'-separated lists of JSON specs / overrides)
split_list() {
  local -a parts=()
  local item
  IFS="$2" read -r -d '' -a parts <<< "$1" || true   # no pathname expansion (JSON may contain [ ])
  for item in "${parts[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    if [[ -n "$item" ]]; then
      printf '%s\n' "$item"
    fi
  done
}
