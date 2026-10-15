#!/usr/bin/env bash
# Stage 65 / recipe R9 -- ON-POLICY DISTILLATION (OPD) of the decider INSIDE the harness, training/opd.py.
# Loop: the STUDENT acts as System-1 (replace on tool_select / tool_params by default) on JevGym training tasks
# (seeds >= 10000) with its own sampled policy -> the TEACHER (Jev / Intern-Decision / Gemini stand-in / a checkpoint
# read out in-process by serve_jev.Engine) answers the same questions at the same states -> KL(teacher || student)
# (or reverse KL / JSD) over the option codes on those rows (+ replay of the previous iteration). Fresh rollouts every
# iteration; the teacher spend is on a Ledger and BUDGET_USD stops the run. Single process (no accelerate): the
# student never generates; more GPUs run independent seeds / ablations side by side (CUDA_VISIBLE_DEVICES).
#
# Variables (defaults in brackets; '' = take the value from CONFIG):
#   CONFIG        YAML of opd.py flags                                              [configs/train/opd_jevgym.yaml]
#   MODEL         student base model (or a full-FT checkpoint dir)                [Qwen/Qwen3-4B]
#   INIT          student adapter to start from ('' = fresh LoRA / full FT)       [$CKPT/r3]
#   OUT           output dir (adapter + metrics.jsonl + labelled/ + ckpt-*/)      [$CKPT/r9-opd]
#   TEACHER       [NAME=]alias | JSON make_backend spec | {"type": "local", ...}   [jev-openrouter]
#                 e.g. 'intern={"type": "intern_decision", "model": "internlm/Intern-Decision-4B"}'
#                      'r4={"type": "local", "model": "Qwen/Qwen3-8B", "adapter": "checkpoints/r4-8b"}'
#   TEACHER_CACHE SQLite cache for real teachers ('' = none)                       [cache/opd_teacher.sqlite]
#   EXECUTOR      config | sim | gemini-lite | gemini-flash | vllm                 [config]
#                 sim = offline pipeline check (simulated System-2, never for reported numbers)
#                 vllm = OpenAI-compatible server at VLLM_BASE_URL serving VLLM_MODEL [Qwen/Qwen3-8B]
#   ITERATIONS / TASKS_PER_ITER / ROLLOUTS_PER_TASK / ROLLOUT_T (0 = greedy)      ['' / '' / '' / '']
#   SEED0 (>= 10000) / N_TASKS / SUITE (YAML dict) / ARMS (YAML list) / MAX_STEPS ['' ...]
#   LOSS          forward_kl | reverse_kl | jsd                                   ['']
#   REPLAY_FRAC / MAX_QUESTIONS / BUDGET_USD / GOLD_ALPHA / TEACHER_T             ['' ...]
#   LORA_R (0 = full FT) / LORA_ALPHA / LR / EPOCHS_PER_ITER / MINIBATCH / MICRO_BATCH / MAX_LEN / WORKERS
#   EVAL_EVERY / EVAL_N / EVAL_SEED0 / SAVE_EVERY / SEED / PRICE_MODEL            ['' ...]
#   RESUME        '' | auto (latest ckpt under OUT) | ckpt-XXXXX dir               ['']
#   EXTRA         extra flags passed verbatim (e.g. '--no-permute --gradient-checkpointing')   ['']
#   GPUS / DRY_RUN (prints the resolved config and the command)
# Examples:
#   EXECUTOR=sim ITERATIONS=1 TASKS_PER_ITER=2 INIT= MODEL=Qwen/Qwen3-0.6B TEACHER=sim bash scripts/train/65_opd.sh  # smoke
#   BUDGET_USD=5 INIT=checkpoints/r3 bash scripts/train/65_opd.sh                             # Jev teacher, Flash-Lite S2
#   TEACHER='intern={"type":"intern_decision","model":"internlm/Intern-Decision-4B"}' OUT=checkpoints/r9-opd-intern \
#     bash scripts/train/65_opd.sh
#   LOSS=reverse_kl OUT=checkpoints/r9-opd-rkl RESUME=auto bash scripts/train/65_opd.sh
source "$(dirname "$0")/_common.sh"

: "${CONFIG:=configs/train/opd_jevgym.yaml}"
: "${INIT=$CKPT/r3}"
: "${OUT:=$CKPT/r9-opd}"
: "${TEACHER:=jev-openrouter}"
: "${TEACHER_CACHE=cache/opd_teacher.sqlite}"
: "${EXECUTOR:=config}"
: "${VLLM_MODEL:=Qwen/Qwen3-8B}"
: "${VLLM_BASE_URL:=http://localhost:8000/v1}"
: "${RESUME:=}"
: "${EXTRA:=}"

need "$CONFIG"
args=(--config "$CONFIG" --student "$MODEL" --out "$OUT" --teacher "$TEACHER" --teacher-cache "$TEACHER_CACHE"
      --init "$INIT")
if [[ -n "$INIT" ]]; then
  need "$INIT"
fi
case "$EXECUTOR" in
  config) ;;
  sim) args+=(--executor sim) ;;
  gemini-lite) args+=(--executor jevgym --llm "{provider: gemini, model: gemini-3.5-flash-lite, temperature: 0.7, reasoning_effort: minimal}") ;;
  gemini-flash) args+=(--executor jevgym --llm "{provider: gemini, model: gemini-3.5-flash, temperature: 0.7, reasoning_effort: minimal}") ;;
  vllm) args+=(--executor jevgym --llm "{provider: vllm, model: $VLLM_MODEL, base_url: '$VLLM_BASE_URL', temperature: 0.7}") ;;
  *) log "unknown EXECUTOR=$EXECUTOR"; exit 2 ;;
esac
# optional overrides: VAR -> --flag only when the variable is set and non-empty (else the CONFIG value applies)
opt() {  # opt VAR FLAG
  local v="${!1:-}"
  if [[ -n "$v" ]]; then
    args+=("$2" "$v")
  fi
}
opt ITERATIONS --iterations;         opt TASKS_PER_ITER --tasks-per-iter;   opt ROLLOUTS_PER_TASK --rollouts-per-task
opt ROLLOUT_T --rollout-temperature; opt SEED0 --seeds;                     opt N_TASKS --n-tasks
opt SUITE --suite;                   opt ARMS --arms;                       opt MAX_STEPS --max-steps
opt LOSS --loss;                     opt REPLAY_FRAC --replay-frac;         opt MAX_QUESTIONS --max-questions
opt BUDGET_USD --budget-usd;         opt GOLD_ALPHA --gold-alpha;           opt TEACHER_T --teacher-temperature
opt LORA_R --lora-r;                 opt LORA_ALPHA --lora-alpha;           opt LR --lr
opt EPOCHS_PER_ITER --epochs-per-iter; opt MINIBATCH --minibatch;           opt MICRO_BATCH --micro-batch
opt MAX_LEN --max-len;               opt WORKERS --workers;                 opt EVAL_EVERY --eval-every
opt EVAL_N --eval-n;                 opt EVAL_SEED0 --eval-seed0;           opt SAVE_EVERY --save-every
opt SEED --seed;                     opt PRICE_MODEL --price-model
if [[ -n "$RESUME" ]]; then
  args+=(--resume "$RESUME")
fi
if [[ -n "$EXTRA" ]]; then
  # shellcheck disable=SC2206  # EXTRA is a whitespace-separated flag list by contract
  args+=($EXTRA)
fi
if [[ "$DRY_RUN" == 1 ]]; then
  "$PY" -m jevharness.training.opd "${args[@]}" --print-config
fi
run "$PY" -m jevharness.training.opd "${args[@]}"
log "done -> $OUT (metrics.jsonl, labelled/, summary.json; next: 90_calibrate_serve_eval.sh ADAPTER=$OUT)"
