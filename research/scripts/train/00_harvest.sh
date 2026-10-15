#!/usr/bin/env bash
# T0 / data -- build the decision-level training set (gold labels) from harness runs + generic replay data.
# Never includes BFCL or When2Call TEST items (our evaluation sets); splits are by task.
#
# Variables (defaults in brackets):
#   RUNS        space-separated run dirs / JSONL globs with oracle gold (JevGym)   [runs/train_jevgym]
#               Generate them on TRAINING seeds with arms that consult System-1 (so every decision is logged with its
#               options, state and oracle gold; Noul gates -- guard / stop -- are harvested too), e.g.
#                 jevbench run configs/experiments/jevgym_tool_select.yaml -o name=train_jevgym suite.seed0=10000 \
#                   suite.n=1000 's1={type: simulated, accuracy: 0.8}' 's1_backends={}' 'sweeps=[]' s1_cache=null \
#                   'arms=[{name: gen_replace, tool_select: {mode: replace}, tool_params: {mode: replace},
#                           stop: {mode: trigger, tau: 0.5}}, {name: gen_guard, tool_select: {mode: guard, tau: 0.5},
#                           stop: {mode: guard, tau: 0.5}}]'
#               (ONE -o followed by several KEY=VALUE: repeated -o flags override each other in jevbench.)
#   MIN_JEVGYM_SEED  JevGym tasks with smaller seeds are dropped (E2/pilot evaluation seeds are 0..199)  [1000]
#   LABEL       gold | s2 | final                                                [gold]
#   ONLY_SUCCESSFUL  1 = keep decisions from successful episodes only (s2/final)  [0]
#   WHEN2CALL   1 = add the When2Call TRAIN split (downloaded from GitHub)       [1]
#   XLAM        path to xlam_function_calling_60k.json ('' = skip; HF, gated)   ['']
#   XLAM_MAX    rows of xLAM to use                                               [20000]
#   OUT         output dir (train.jsonl, dev.jsonl)                               [$DATA = data/decisions]
#   DEV_FRAC    fraction of TASKS held out                                        [0.1]
#   SEED                                                                          [0]
#   DRY_RUN     1 = print commands only
# Example:
#   RUNS="runs/train_jevgym" XLAM=data/raw/xlam_function_calling_60k.json bash scripts/train/00_harvest.sh
source "$(dirname "$0")/_common.sh"

: "${RUNS:=runs/train_jevgym}"
: "${MIN_JEVGYM_SEED:=1000}"
: "${LABEL:=gold}"
: "${ONLY_SUCCESSFUL:=0}"
: "${WHEN2CALL:=1}"
: "${XLAM:=}"
: "${XLAM_MAX:=20000}"
: "${OUT:=$DATA}"
: "${DEV_FRAC:=0.1}"
: "${SEED:=0}"

read -r -a run_list <<< "$RUNS"
args=(--label "$LABEL" --out "$OUT" --dev-frac "$DEV_FRAC" --seed "$SEED" --min-jevgym-seed "$MIN_JEVGYM_SEED")
if (( ${#run_list[@]} > 0 )); then
  args+=(--runs "${run_list[@]}")
fi
if [[ "$ONLY_SUCCESSFUL" == 1 ]]; then
  args+=(--only-successful)
fi
if [[ "$WHEN2CALL" == 1 ]]; then
  args+=(--when2call-train)
fi
if [[ -n "$XLAM" ]]; then
  need "$XLAM"
  args+=(--xlam "$XLAM" --xlam-max "$XLAM_MAX")
fi

run "$PY" -m jevharness.training.harvest "${args[@]}"
run "$PY" -m jevharness.training.data stats "$OUT/train.jsonl"
run "$PY" -m jevharness.training.data stats "$OUT/dev.jsonl"
log "done -> $OUT/{train,dev}.jsonl"
