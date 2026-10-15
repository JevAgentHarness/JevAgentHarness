#!/usr/bin/env bash
# T5 -- DPO on counterfactual MC pairs (recipe R5): winner = argmax Q̂, loser = argmin Q̂ per state,
#   L = -log sigmoid( beta * [ (log pi(a_w) - log pi_ref(a_w)) - (log pi(a_l) - log pi_ref(a_l)) ] )
# over the restricted option softmax; pi_ref = frozen copy of INIT (train_decision --ref auto).
# Only states whose Q̂ gap max-min >= MIN_Q_GAP are used (noisy MC pairs hurt DPO).
#
# Variables (defaults in brackets):
#   INIT [$CKPT/r3]   MODEL [Qwen/Qwen3-4B]   MC [$DATA/mc.jsonl]   OUT [$CKPT/r5]
#   BETA [0.1]   MIN_Q_GAP [0.3]   LR [5e-6]   EPOCHS [1]   MAX_STEPS (> 0 overrides EPOCHS) [-1]   BATCH [4]   EFF_BATCH [64]   MAX_LEN [4096]
#   REF (auto | init | base) [auto]   DEV [$DATA/dev.jsonl]   GPUS / DRY_RUN / SEED
source "$(dirname "$0")/_common.sh"

: "${INIT:=$CKPT/r3}"
: "${MC:=$DATA/mc.jsonl}"
: "${OUT:=$CKPT/r5}"
: "${BETA:=0.1}"
: "${MIN_Q_GAP:=0.3}"
: "${LR:=5e-6}"
: "${EPOCHS:=1}"
: "${MAX_STEPS:=-1}"
: "${BATCH:=4}"
: "${EFF_BATCH:=64}"
: "${MAX_LEN:=4096}"
: "${REF:=auto}"
: "${DEV=$DATA/dev.jsonl}"   # DEV= (empty) -> no held-out eval
: "${SEED:=0}"

need "$INIT" "$MC"
args=(--model "$MODEL" --init "$INIT" --train "$MC" --out "$OUT" --loss dpo --beta "$BETA" --ref "$REF"
      --min-q-gap "$MIN_Q_GAP" --lr "$LR" --epochs "$EPOCHS" --max-steps "$MAX_STEPS" --batch-size "$BATCH"
      --grad-accum "$(grad_accum "$EFF_BATCH" "$BATCH" "$GPUS")" --max-len "$MAX_LEN" --seed "$SEED")
if [[ -n "$DEV" && ( -e "$DEV" || "$DRY_RUN" == 1 ) ]]; then
  args+=(--dev "$DEV")
fi
launch jevharness.training.train_decision "${args[@]}"
log "done -> $OUT"
