#!/usr/bin/env bash
# T1 / T2 -- supervised fine-tuning of the single-pass decider (recipes R1, R2).
#   R1: hard CE on gold labels, fixed option order        (RECIPE=r1)
#   R2: soft CE + 0.5 Brier, random option permutation    (RECIPE=r2, default)
#
# Variables (defaults in brackets):
#   RECIPE        r1 | r2 (sets LOSS / PERMUTE defaults)                      [r2]
#   MODEL         base model                                                 [Qwen/Qwen3-4B]
#   TRAIN / DEV   training / held-out JSONL                                  [$DATA/train.jsonl / $DATA/dev.jsonl]
#   REPLAY        optional generic-replay JSONL mixed in (e.g. xLAM/When2Call-only set)   ['']
#   REPLAY_FACTOR scale factor of REPLAY (data.py mix)                        [1.0]
#   OUT           output checkpoint                                          [$CKPT/$RECIPE]
#   INIT          adapter to continue from ('' = fresh LoRA)                 ['']
#   LOSS          soft_ce | soft_ce_brier                                    [r1: soft_ce, r2: soft_ce_brier]
#   PERMUTE       1 = random option order per example                        [r1: 0, r2: 1]
#   BRIER_WEIGHT                                                             [0.5]
#   LORA_R / LORA_ALPHA   (LORA_R=0 -> full fine-tuning)                     [32 / 64]
#   LR / EPOCHS / MAX_STEPS                                                  [1e-4 / 2 / -1]
#   BATCH         per-device batch                                           [4]
#   EFF_BATCH     effective batch (grad accumulation = EFF_BATCH / (BATCH * GPUS))       [64]
#   MAX_LEN       prompt tokens kept (tail)                                  [4096]
#   GPUS / ACCEL / DRY_RUN / REPORT_TO(none|wandb) / SEED
#   SFT_EXTRA     extra train_decision flags, e.g. '--group-by-length --no-gradient-checkpointing'   ['']
# Examples:
#   GPUS=8 bash scripts/train/10_sft.sh                    # R2 on 8xH100 (~1 h for 20k examples, 4B LoRA)
#   RECIPE=r1 GPUS=8 bash scripts/train/10_sft.sh
#   MODEL=Qwen/Qwen3-0.6B LORA_R=0 LR=2e-5 RECIPE=r2-0.6b bash scripts/train/10_sft.sh
source "$(dirname "$0")/_common.sh"

: "${RECIPE:=r2}"
case "$RECIPE" in
  r1*) : "${LOSS:=soft_ce}";       : "${PERMUTE:=0}" ;;
  *)   : "${LOSS:=soft_ce_brier}"; : "${PERMUTE:=1}" ;;
esac
: "${TRAIN:=$DATA/train.jsonl}"
: "${DEV=$DATA/dev.jsonl}"   # DEV= (empty) -> no held-out eval
: "${REPLAY:=}"
: "${REPLAY_FACTOR:=1.0}"
: "${OUT:=$CKPT/$RECIPE}"
: "${INIT:=}"
: "${BRIER_WEIGHT:=0.5}"
: "${LORA_R:=32}"
: "${LORA_ALPHA:=64}"
: "${LR:=1e-4}"
: "${EPOCHS:=2}"
: "${MAX_STEPS:=-1}"
: "${BATCH:=4}"
: "${EFF_BATCH:=64}"
: "${MAX_LEN:=4096}"
: "${REPORT_TO:=none}"
: "${SEED:=0}"

need "$TRAIN"
train_file="$TRAIN"
if [[ -n "$REPLAY" ]]; then
  need "$REPLAY"
  train_file="$OUT/train_mix.jsonl"
  run mkdir -p "$OUT"
  run "$PY" -m jevharness.training.data mix --in "$TRAIN:1" "$REPLAY:$REPLAY_FACTOR" --out "$train_file" --seed "$SEED"
fi

args=(--model "$MODEL" --train "$train_file" --out "$OUT" --loss "$LOSS" --brier-weight "$BRIER_WEIGHT"
      --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --lr "$LR" --epochs "$EPOCHS" --max-steps "$MAX_STEPS"
      --batch-size "$BATCH" --grad-accum "$(grad_accum "$EFF_BATCH" "$BATCH" "$GPUS")" --max-len "$MAX_LEN"
      --report-to "$REPORT_TO" --seed "$SEED")
if [[ -n "$DEV" && ( -e "$DEV" || "$DRY_RUN" == 1 ) ]]; then
  args+=(--dev "$DEV")
fi
if [[ "$PERMUTE" != 1 ]]; then
  args+=(--no-permute)
fi
if [[ -n "$INIT" ]]; then
  args+=(--init "$INIT")
fi
if [[ -n "${SFT_EXTRA:-}" ]]; then
  # shellcheck disable=SC2206  # SFT_EXTRA is a whitespace-separated flag list by contract
  args+=($SFT_EXTRA)
fi

launch jevharness.training.train_decision "${args[@]}"
log "done -> $OUT (next: 20_teacher_soft.sh or 90_calibrate_serve_eval.sh ADAPTER=$OUT)"
