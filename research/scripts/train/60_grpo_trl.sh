#!/usr/bin/env bash
# T6b -- decision-level GRPO with verifiable rewards (recipe R6b), training/grpo_trl.py.
#   IMPL=trl    TRL GRPOTrainer: G sampled completions (1-2 tokens) from the full vocabulary per decision prompt;
#               rewards correct / format / soft / qvalue; LoRA; reference = frozen INIT adapter; multi-GPU.
#   IMPL=torch  the same objective on the restricted option softmax (no format failures, one prefill per prompt);
#               single process (GPUS must be 1).
# Verified against TRL 1.14.1 / transformers 5.18 / peft 0.21 (see the module docstring).
#
# Variables (defaults in brackets):
#   IMPL [trl]   INIT [$CKPT/r3]   MODEL [Qwen/Qwen3-4B]   TRAIN [$DATA/train.jsonl]   DEV [$DATA/dev.jsonl]
#   OUT [$CKPT/r6b]
#   REWARDS        comma list: correct,format (gold data) | soft (teacher/MC targets) | qvalue (MC Q̂)   [correct,format]
#   REWARD_WEIGHTS comma list aligned with REWARDS ('' = all 1)                                        ['']
#   NUM_GEN [8]   MAX_COMPLETION [1]   TEMPERATURE [1.0]   BETA (KL) [0.02]   LOSS_TYPE [dapo]
#   SCALE_REWARDS (group|batch|none) [group]   EPSILON [0.2]   EPSILON_HIGH [0.28]   NUM_ITER (mu) [1]
#   LR [1e-5]   BATCH (completions per device per step) [16]   GRAD_ACCUM [4]   EPOCHS [1]   MAX_STEPS [-1]
#   N_PERM (rows per example with fresh option orders) [1]   MAX_LEN [4096]   USE_VLLM [0]   GPUS / DRY_RUN / SEED
# Prompts per optimizer step (TRL) = BATCH * GRAD_ACCUM * GPUS / NUM_GEN.
source "$(dirname "$0")/_common.sh"

: "${IMPL:=trl}"
: "${INIT=$CKPT/r3}"   # INIT= (empty) -> fresh LoRA
: "${TRAIN:=$DATA/train.jsonl}"
: "${DEV=$DATA/dev.jsonl}"   # DEV= (empty) -> no held-out eval
: "${OUT:=$CKPT/r6b}"
: "${REWARDS:=correct,format}"
: "${REWARD_WEIGHTS:=}"
: "${NUM_GEN:=8}"
: "${MAX_COMPLETION:=1}"
: "${TEMPERATURE:=1.0}"
: "${BETA:=0.02}"
: "${LOSS_TYPE:=dapo}"
: "${SCALE_REWARDS:=group}"
: "${EPSILON:=0.2}"
: "${EPSILON_HIGH:=0.28}"
: "${NUM_ITER:=1}"
: "${LR:=1e-5}"
: "${BATCH:=16}"
: "${GRAD_ACCUM:=4}"
: "${EPOCHS:=1}"
: "${MAX_STEPS:=-1}"
: "${N_PERM:=1}"
: "${MAX_LEN:=4096}"
: "${USE_VLLM:=0}"
: "${SEED:=0}"

if [[ "$IMPL" == torch && "$GPUS" -gt 1 ]]; then
  log "IMPL=torch is single-process: set GPUS=1 (or run seeds/ablations one per GPU with CUDA_VISIBLE_DEVICES)"
  exit 2
fi
need "$TRAIN"
args=(--impl "$IMPL" --model "$MODEL" --train "$TRAIN" --out "$OUT" --rewards "$REWARDS"
      --num-generations "$NUM_GEN" --max-completion-length "$MAX_COMPLETION" --temperature "$TEMPERATURE"
      --beta "$BETA" --loss-type "$LOSS_TYPE" --scale-rewards "$SCALE_REWARDS" --epsilon "$EPSILON"
      --epsilon-high "$EPSILON_HIGH" --num-iterations "$NUM_ITER" --lr "$LR" --batch-size "$BATCH"
      --grad-accum "$GRAD_ACCUM" --epochs "$EPOCHS" --max-steps "$MAX_STEPS" --n-permutations "$N_PERM"
      --max-len "$MAX_LEN" --seed "$SEED")
if [[ -n "$INIT" ]]; then
  need "$INIT"
  args+=(--init "$INIT")
fi
if [[ -n "$REWARD_WEIGHTS" ]]; then
  args+=(--reward-weights "$REWARD_WEIGHTS")
fi
if [[ -n "$DEV" && ( -e "$DEV" || "$DRY_RUN" == 1 ) ]]; then
  args+=(--dev "$DEV")
fi
if [[ "$USE_VLLM" == 1 ]]; then
  args+=(--use-vllm)
fi
if [[ "$IMPL" == trl ]]; then
  launch jevharness.training.grpo_trl "${args[@]}"
else
  run "$PY" -m jevharness.training.grpo_trl "${args[@]}"
fi
log "done -> $OUT"
