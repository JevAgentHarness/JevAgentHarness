#!/usr/bin/env bash
# T6 -- exact-expectation policy gradient on MC Q̂ (recipe R6). Options are enumerable, so the expected reward is
# computed exactly instead of sampled:
#   L = -sum_a pi_theta(a|s) Q̂(s,a) + KL_WEIGHT * KL(pi_theta(.|s) || pi_ref(.|s)),   pi_ref = frozen INIT
# (options without a Q̂ estimate contribute 0 to the first term).
#
# Variables (defaults in brackets):
#   INIT [$CKPT/r3]   MODEL [Qwen/Qwen3-4B]   MC [$DATA/mc.jsonl]   OUT [$CKPT/r6]
#   KL_WEIGHT [0.02]   MIN_Q_GAP [0.0]   LR [1e-5]   EPOCHS [1]   MAX_STEPS (> 0 overrides EPOCHS) [-1]   BATCH [4]   EFF_BATCH [64]   MAX_LEN [4096]
#   REF (auto | init | base) [auto]   DEV [$DATA/dev.jsonl]   GPUS / DRY_RUN / SEED
source "$(dirname "$0")/_common.sh"

: "${INIT:=$CKPT/r3}"
: "${MC:=$DATA/mc.jsonl}"
: "${OUT:=$CKPT/r6}"
: "${KL_WEIGHT:=0.02}"
: "${MIN_Q_GAP:=0.0}"
: "${LR:=1e-5}"
: "${EPOCHS:=1}"
: "${MAX_STEPS:=-1}"
: "${BATCH:=4}"
: "${EFF_BATCH:=64}"
: "${MAX_LEN:=4096}"
: "${REF:=auto}"
: "${DEV=$DATA/dev.jsonl}"   # DEV= (empty) -> no held-out eval
: "${SEED:=0}"

need "$INIT" "$MC"
args=(--model "$MODEL" --init "$INIT" --train "$MC" --out "$OUT" --loss pg --kl-weight "$KL_WEIGHT" --ref "$REF"
      --min-q-gap "$MIN_Q_GAP" --lr "$LR" --epochs "$EPOCHS" --max-steps "$MAX_STEPS" --batch-size "$BATCH"
      --grad-accum "$(grad_accum "$EFF_BATCH" "$BATCH" "$GPUS")" --max-len "$MAX_LEN" --seed "$SEED")
if [[ -n "$DEV" && ( -e "$DEV" || "$DRY_RUN" == 1 ) ]]; then
  args+=(--dev "$DEV")
fi
launch jevharness.training.train_decision "${args[@]}"
log "done -> $OUT"
