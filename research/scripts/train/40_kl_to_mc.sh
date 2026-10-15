#!/usr/bin/env bash
# T4 -- KL-regularised optimum on counterfactual MC targets (recipe R4):
#   p*(a|s) ∝ pi_ref(a|s) exp(Q̂(s,a)/beta),   minimise  -sum_a p*(a|s) log pi_theta(a|s)   (soft CE)
# By default pi_ref is RE-COMPUTED from the init checkpoint (REF_FROM_INIT=1: teacher_label with a local teacher
# = INIT and --mc-beta), so p* is the optimum relative to the policy being fine-tuned; REF_FROM_INIT=0 keeps the
# targets written by 30_mc_rollouts.sh (pi_ref = the logged System-1 distribution, or uniform).
# Generic replay (gold SFT data) is mixed in to prevent forgetting.
#
# Variables (defaults in brackets):
#   INIT           adapter to start from (R3)                            [$CKPT/r3]
#   MODEL          base model of INIT                                    [Qwen/Qwen3-4B]
#   MC             MC-labelled JSONL                                     [$DATA/mc.jsonl]
#   BETA           target temperature                                    [0.2]
#   REF_FROM_INIT  1 = pi_ref from INIT (see above)                      [1]
#   REPLAY         replay JSONL ('' = none)                              [$DATA/train.jsonl]
#   REPLAY_FACTOR  replay size relative to its file                      [0.5]
#   OUT            [$CKPT/r4]    LR [3e-5]   EPOCHS [1]   BATCH [4]   EFF_BATCH [64]   MAX_LEN [4096]
#   DEV / GPUS / DRY_RUN / SEED
source "$(dirname "$0")/_common.sh"

: "${INIT:=$CKPT/r3}"
: "${MC:=$DATA/mc.jsonl}"
: "${BETA:=0.2}"
: "${REF_FROM_INIT:=1}"
: "${REPLAY=$DATA/train.jsonl}"   # REPLAY= (empty) disables replay
: "${REPLAY_FACTOR:=0.5}"
: "${OUT:=$CKPT/r4}"
: "${LR:=3e-5}"
: "${EPOCHS:=1}"
: "${BATCH:=4}"
: "${EFF_BATCH:=64}"
: "${MAX_LEN:=4096}"
: "${SEED:=0}"

need "$INIT" "$MC"
run mkdir -p "$OUT"
targets="$MC"
if [[ "$REF_FROM_INIT" == 1 ]]; then
  targets="$OUT/mc_ref_init.jsonl"
  spec=$(printf '{"type": "local", "model": "%s", "adapter": "%s", "max_len": %s}' "$MODEL" "$INIT" "$MAX_LEN")
  run "$PY" -m jevharness.training.teacher_label --in "$MC" --out "$targets" --teacher "init=$spec" \
    --mc-beta "$BETA" --gold-filter keep --on-disagree gold --alpha 0 --cache ""
fi
train_file="$targets"
if [[ -n "$REPLAY" ]]; then
  need "$REPLAY"
  train_file="$OUT/train_mix.jsonl"
  run "$PY" -m jevharness.training.data mix --in "$targets:1" "$REPLAY:$REPLAY_FACTOR" --out "$train_file" --seed "$SEED"
fi

MODEL="$MODEL" TRAIN="$train_file" OUT="$OUT" INIT="$INIT" RECIPE=r4 LOSS=soft_ce PERMUTE=1 LR="$LR" \
  EPOCHS="$EPOCHS" BATCH="$BATCH" EFF_BATCH="$EFF_BATCH" MAX_LEN="$MAX_LEN" SEED="$SEED" REPLAY="" \
  bash "$(dirname "$0")/10_sft.sh"
log "done -> $OUT"
