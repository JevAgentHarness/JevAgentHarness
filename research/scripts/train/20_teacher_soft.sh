#!/usr/bin/env bash
# T3 -- teacher soft labels (Jev when reachable, Gemini verbalised distributions as the stand-in) + SFT (recipe R3).
#
# Step 1  training.teacher_label: every training decision is asked to each teacher; pooled distribution,
#         agreement filter, alpha-mix with the one-hot gold, SQLite cache, Ledger cost in <out>.summary.json.
# Step 2  (DO_TRAIN=1) training.train_decision --loss soft_ce_brier on the labelled set (+ gold replay).
# The DEV set stays gold-labelled (it is the evaluation target).
#
# Variables (defaults in brackets):
#   TEACHERS     ';'-separated [NAME=]SPEC list (alias or JSON make_backend spec)          [gemini-lite]
#                Jev (needs openrouter.ai allow-listed):  TEACHERS='jev=jev-openrouter;lite=gemini-lite'
#   IN           input JSONL                                                               [$DATA/train.jsonl]
#   LABELED      output JSONL                                                              [$DATA/train_teacher.jsonl]
#   ALPHA        weight of the gold one-hot in the target                                  [0.3]
#   MIN_AGREE    teachers whose argmax must equal the consensus ('' = all)                ['']
#   MIN_CONF     drop when the pooled max prob is below this                              [0.0]
#   POOL         mean | geo                                                                [mean]
#   GOLD_FILTER  consensus != gold: drop | gold | keep                                     [drop]
#   ON_DISAGREE  drop | gold                                                               [gold]
#   CACHE        SQLite decision cache ('' = no cache)                                     [cache/teacher.sqlite]
#   WORKERS / LIMIT (label only N examples; '' = all) / SHUFFLE (1 = shuffle before LIMIT)  [8 / '' / 0]
#   DO_TRAIN     1 = run the R3 SFT after labelling                                        [1]
#   OUT          R3 checkpoint                                                             [$CKPT/r3]
#   INIT         '' = fresh LoRA on MODEL (paper R3) | $CKPT/r2 to continue               ['']
#   LR / EPOCHS / BATCH / EFF_BATCH / LORA_R / LORA_ALPHA / MAX_LEN / MODEL / GPUS / DRY_RUN
# Examples:
#   LIMIT=200 DO_TRAIN=0 bash scripts/train/20_teacher_soft.sh             # budget probe: see the summary's cost
#   TEACHERS='jev=jev-openrouter;lite=gemini-lite' MIN_AGREE=2 GPUS=8 bash scripts/train/20_teacher_soft.sh
source "$(dirname "$0")/_common.sh"

: "${TEACHERS:=gemini-lite}"
: "${IN:=$DATA/train.jsonl}"
: "${LABELED:=$DATA/train_teacher.jsonl}"
: "${ALPHA:=0.3}"
: "${MIN_AGREE:=}"
: "${MIN_CONF:=0.0}"
: "${POOL:=mean}"
: "${GOLD_FILTER:=drop}"
: "${ON_DISAGREE:=gold}"
: "${CACHE=cache/teacher.sqlite}"   # CACHE='' disables the cache (unset -> default path)
: "${WORKERS:=8}"
: "${LIMIT:=}"
: "${SHUFFLE:=0}"
: "${DO_TRAIN:=1}"
: "${OUT:=$CKPT/r3}"
: "${INIT:=}"
: "${LR:=1e-4}"
: "${EPOCHS:=2}"
: "${BATCH:=4}"
: "${EFF_BATCH:=64}"
: "${LORA_R:=32}"
: "${LORA_ALPHA:=64}"
: "${MAX_LEN:=4096}"
: "${SEED:=0}"

need "$IN"
args=(--in "$IN" --out "$LABELED" --alpha "$ALPHA" --min-conf "$MIN_CONF" --pool "$POOL"
      --gold-filter "$GOLD_FILTER" --on-disagree "$ON_DISAGREE" --cache "$CACHE" --workers "$WORKERS" --seed "$SEED")
while IFS= read -r t; do
  args+=(--teacher "$t")
done < <(split_list "$TEACHERS" ";")
if [[ -n "$MIN_AGREE" ]]; then
  args+=(--min-agree "$MIN_AGREE")
fi
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ "$SHUFFLE" == 1 ]]; then
  args+=(--shuffle)
fi
run "$PY" -m jevharness.training.teacher_label "${args[@]}"

if [[ "$DO_TRAIN" == 1 ]]; then
  MODEL="$MODEL" TRAIN="$LABELED" OUT="$OUT" INIT="$INIT" RECIPE=r3 LOSS=soft_ce_brier PERMUTE=1 LR="$LR" \
    EPOCHS="$EPOCHS" BATCH="$BATCH" EFF_BATCH="$EFF_BATCH" LORA_R="$LORA_R" LORA_ALPHA="$LORA_ALPHA" \
    MAX_LEN="$MAX_LEN" SEED="$SEED" bash "$(dirname "$0")/10_sft.sh"
  log "done -> $LABELED (+ $LABELED.summary.json), $OUT"
else
  log "done -> $LABELED (+ $LABELED.summary.json)"
fi
