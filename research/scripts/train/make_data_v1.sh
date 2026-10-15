#!/usr/bin/env bash
# Build data/train_v1 -- decision states from the harness's own state distribution on the EVALUATION SETTING
# (JevGym v2 clarify + multi, 60 near distractors, 20 noise lines, strict) on TRAINING seeds, with Jev soft labels.
# Why and how: docs/TRAINING_RECIPES.md §18; module: src/jevharness/training/data_v1.py. Resumable / cached: a re-run
# over existing runs and the teacher cache costs $0.
#
#   1. collect   three behaviour arms with the real harness, executor Gemini 3.5 Flash-Lite (T=0), seeds SEED0.. :
#                  lite_shadow       Flash-Lite acts, System-1 asked in shadow (tool_select / tool_params / stop)
#                  nearmiss_replace  System-1 = oracle + near-miss errors (eps) in replace -> recovery states
#                  lite_guard        Flash-Lite proposes, System-1 guards tool calls and the stop
#   2. build     harvest (gold), dedupe across arms, hold-out checks (seeds, gate-dev tasks, E1 dev-prefix ids,
#                8-gram decontamination vs BFCL + When2Call test), balance (stop-trigger 'no', OTHER args), split by
#                task, + v0 train rows, + knockout-final variants, + decontaminated When2Call-train replay
#   3. label     Jev (OpenRouter) soft labels on train (q = 0.3 gold + 0.7 p_Jev; gold one-hot where Jev disagrees),
#                Jev reference metrics on dev (dev targets stay gold)                          -> Ledger $ in summaries
#   4. manifest  data/train_v1/MANIFEST.json + results/training/train_v1/{MANIFEST.md,manifest.json} (committable)
# Outputs: data/train_v1/{train,dev,dev_recovery,dev_knockout,replay_when2call}.jsonl.gz (~12 MB in all, gzip: every
# loader reads .jsonl.gz), summaries, build_info.json, MANIFEST.json. Bulky intermediates stay under data/cache/
# (gitignored): trajectories $RUNS_DIR/$NAME, gold rows $RUNS_DIR/_work, the teacher cache $CACHE.
#
# Variables (defaults in brackets):
#   OUT [data/train_v1]  RUNS_DIR [data/cache/train_v1]  NAME [train_v1_jevgym]  SEED0 [30000]  N [600]
#   N_SHADOW [250]  N_GUARD [150]  N_REPLACE [600]  WORKERS [8]  EPS_TOOL [0.2]  EPS_PARAM [0.15]  EPS_GATE [0.1]
#   TEACHER [jev=jev-openrouter]  ALPHA [0.3]  CACHE [data/cache/teacher_v1.sqlite]  KO_FRAC [0.5]  DEV_FRAC [0.15]
#   V0_TRAIN [data/train_v0/train.jsonl]  REPLAY [data/train_v0/replay_when2call.jsonl.gz]
#   RESULTS [results/training/train_v1]  SKIP_EPISODES=1 (reuse RUNS_DIR)  SKIP_LABEL=1  DRY_RUN=1
# Cost of the built v1: results/training/train_v1/MANIFEST.md "Cost" (episodes + teacher Ledgers).
source "$(dirname "$0")/_common.sh"

: "${OUT:=data/train_v1}"
: "${RUNS_DIR:=data/cache/train_v1}"
: "${NAME:=train_v1_jevgym}"
: "${SEED0:=30000}"
: "${N:=600}"
: "${N_SHADOW:=250}"
: "${N_GUARD:=150}"
: "${N_REPLACE:=600}"
: "${WORKERS:=8}"
: "${EPS_TOOL:=0.2}"
: "${EPS_PARAM:=0.15}"
: "${EPS_GATE:=0.1}"
: "${TEACHER:=jev=jev-openrouter}"
: "${ALPHA:=0.3}"
: "${CACHE:=data/cache/teacher_v1.sqlite}"
: "${KO_FRAC:=0.5}"
: "${DEV_FRAC:=0.15}"
: "${V0_TRAIN=data/train_v0/train.jsonl}"
: "${REPLAY=data/train_v0/replay_when2call.jsonl.gz}"
: "${RESULTS:=results/training/train_v1}"
: "${SKIP_EPISODES:=0}"
: "${SKIP_LABEL:=0}"

mod=("$PY" -m jevharness.training.data_v1)
RUN="$RUNS_DIR/$NAME"

# 1. episodes (resumable: finished tasks are skipped; the seed plan is checked before anything runs)
if [[ "$SKIP_EPISODES" != 1 ]]; then
  run "${mod[@]}" collect --out "$RUNS_DIR" --name "$NAME" --seed0 "$SEED0" --n "$N" --workers "$WORKERS" \
    --n-by-arm "{\"lite_shadow\": $N_SHADOW, \"lite_guard\": $N_GUARD, \"nearmiss_replace\": $N_REPLACE}" \
    --eps-tool "$EPS_TOOL" --eps-param "$EPS_PARAM" --eps-gate "$EPS_GATE"
fi
need "$RUN/experiment.yaml"

# 2. harvest + hold-out checks + balance + split (no API call)
run "${mod[@]}" build --runs "$RUN" --out "$OUT" --work "$RUNS_DIR/_work" --v0-train "$V0_TRAIN" --replay "$REPLAY" \
  --dev-frac "$DEV_FRAC" --ko-frac "$KO_FRAC"

# 3. Jev soft labels (train) + Jev reference on dev
if [[ "$SKIP_LABEL" != 1 ]]; then
  run "${mod[@]}" label --out "$OUT" --work "$RUNS_DIR/_work" --teacher "$TEACHER" --alpha "$ALPHA" --cache "$CACHE" \
    --workers "$WORKERS"
fi

# 4. manifests (the data files stay out of git; the manifest and the markdown summary are committable)
run "${mod[@]}" manifest --out "$OUT" --info "$OUT/build_info.json" --md "$RESULTS/MANIFEST.md"
run mkdir -p "$RESULTS"
run cp "$OUT/MANIFEST.json" "$RESULTS/manifest.json"
for f in train.jsonl.summary.json dev_teacher.summary.json; do
  if [[ -e "$OUT/$f" || "$DRY_RUN" == 1 ]]; then
    run cp "$OUT/$f" "$RESULTS/$f"
  fi
done
log "done -> $OUT (*.jsonl.gz ~12 MB + MANIFEST.json, committable like data/train_v0) and $RESULTS (manifest + summary)"
