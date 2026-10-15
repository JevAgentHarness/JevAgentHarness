#!/usr/bin/env bash
# T8 (distil part) -- distil the best decider into a small, fast student (recipe R7).
# Step 1  soft labels from the TEACHER on the exact training prompt (teacher_label, local in-process readout of
#         TEACHER_MODEL + TEACHER_ADAPTER [+ TEACHER_CAL temperatures]); or any other teacher via TEACHER_SPEC
#         (e.g. 'jev=jev-openrouter' to distil Jev directly).
# Step 2  SFT of STUDENT on  q = ALPHA * gold + (1 - ALPHA) * p_teacher^(1/T)  with soft CE + Brier (10_sft.sh).
# Then calibrate + serve the student with 90_calibrate_serve_eval.sh.
#
# Variables (defaults in brackets):
#   TEACHER_MODEL [Qwen/Qwen3-4B]   TEACHER_ADAPTER [$CKPT/r4]   TEACHER_CAL ('' = raw logits) ['']
#   TEACHER_SPEC  ';'-separated teacher list replacing the local teacher ('' = local)          ['']
#   STUDENT       [Qwen/Qwen3-0.6B]
#   IN            decisions to distil on (gold where available)                              [$DATA/train.jsonl]
#   EXTRA         extra JSONL (e.g. MC states, unlabeled harness states) mixed into IN ('' = none)  ['']
#   LABELED       [$DATA/train_distill.jsonl]   ALPHA [0.2]   T (teacher temperature) [1.0]
#   GOLD_FILTER   keep | gold | drop (teacher argmax != gold)                                [keep]
#   OUT [$CKPT/r7-0.6b]   LORA_R (0 = full fine-tuning) [0]   LR [2e-5]   EPOCHS [2]   BATCH [8]   EFF_BATCH [64]
#   MAX_LEN [4096]   TEACHER_BATCH [32]   GPUS / DRY_RUN / SEED
source "$(dirname "$0")/_common.sh"

: "${TEACHER_MODEL:=Qwen/Qwen3-4B}"
: "${TEACHER_ADAPTER:=$CKPT/r4}"
: "${TEACHER_CAL:=}"
: "${TEACHER_SPEC:=}"
: "${STUDENT:=Qwen/Qwen3-0.6B}"
: "${IN:=$DATA/train.jsonl}"
: "${EXTRA:=}"
: "${LABELED:=$DATA/train_distill.jsonl}"
: "${ALPHA:=0.2}"
: "${T:=1.0}"
: "${GOLD_FILTER:=keep}"
: "${OUT:=$CKPT/r7-0.6b}"
: "${LORA_R:=0}"
: "${LR:=2e-5}"
: "${EPOCHS:=2}"
: "${BATCH:=8}"
: "${EFF_BATCH:=64}"
: "${MAX_LEN:=4096}"
: "${TEACHER_BATCH:=32}"
: "${SEED:=0}"

need "$IN"
src="$IN"
if [[ -n "$EXTRA" ]]; then
  need "$EXTRA"
  src="$LABELED.src.jsonl"
  run "$PY" -m jevharness.training.data mix --in "$IN:1" "$EXTRA:1" --out "$src" --seed "$SEED"
fi

args=(--in "$src" --out "$LABELED" --alpha "$ALPHA" --temperature "$T" --gold-filter "$GOLD_FILTER"
      --on-disagree gold --min-agree 1 --seed "$SEED")
if [[ -n "$TEACHER_SPEC" ]]; then
  while IFS= read -r t; do
    args+=(--teacher "$t")
  done < <(split_list "$TEACHER_SPEC" ";")
else
  need "$TEACHER_ADAPTER"
  cal=null
  if [[ -n "$TEACHER_CAL" ]]; then
    cal="\"$TEACHER_CAL\""
  fi
  spec=$(printf '{"type": "local", "model": "%s", "adapter": "%s", "calibration": %s, "max_len": %s, "batch_size": %s}' \
    "$TEACHER_MODEL" "$TEACHER_ADAPTER" "$cal" "$MAX_LEN" "$TEACHER_BATCH")
  args+=(--teacher "teacher=$spec" --cache "")
fi
run "$PY" -m jevharness.training.teacher_label "${args[@]}"

MODEL="$STUDENT" TRAIN="$LABELED" OUT="$OUT" INIT="" RECIPE=r7 LOSS=soft_ce_brier PERMUTE=1 LR="$LR" \
  EPOCHS="$EPOCHS" BATCH="$BATCH" EFF_BATCH="$EFF_BATCH" LORA_R="$LORA_R" MAX_LEN="$MAX_LEN" SEED="$SEED" REPLAY="" \
  bash "$(dirname "$0")/10_sft.sh"
log "done -> $OUT (next: MODEL=$STUDENT ADAPTER=$OUT bash scripts/train/90_calibrate_serve_eval.sh)"
