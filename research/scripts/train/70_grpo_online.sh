#!/usr/bin/env bash
# T7 -- episode-level GRPO of the decider INSIDE the harness (recipe R8), training/grpo_online.py.
# The decider runs in-process and samples its options; G rollouts per JevGym task with the JevAgent harness;
# reward = success - cost terms (anti-cheapness gated by default); GRPO group advantages (+ GiGPO anchor states);
# PPO-clip on stored behaviour log-probs; exact KL to the frozen INIT adapter. Single process per run: more GPUs
# run independent SEEDS (or ablations via SET) side by side, one per GPU.
#
# Variables (defaults in brackets):
#   CONFIG      YAML                                                  [configs/train/grpo_online_jevgym.yaml]
#   MODEL [Qwen/Qwen3-4B]   INIT [$CKPT/r4] ('' = fresh LoRA; only for replace/assist/gate arms)
#   OUT         output dir; with several SEEDS -> $OUT-s<seed>       [$CKPT/r8]
#   EXECUTOR    config | sim | gemini-lite | gemini-flash | vllm      [config]
#               sim = offline pipeline check (simulated LLM, never for reported numbers)
#               vllm = OpenAI-compatible server at VLLM_BASE_URL serving VLLM_MODEL [Qwen/Qwen3-8B]
#   ITERATIONS  ('' = from CONFIG)                                    ['']
#   SEEDS       space-separated seeds, run in parallel                [0]
#   SET         ';'-separated dotted overrides, e.g. 'grpo.kl_beta=0.05;reward.kind=linear;reward.lambda_calls=0.02'
#   RESUME      ckpt-XXXXX dir to continue from ('' = fresh)          ['']
#   GPUS        GPUs available; seed i runs on GPU (i mod GPUS)       [1]
#   DRY_RUN     1 = print commands (also prints the resolved config of the first seed)
# Examples:
#   EXECUTOR=sim ITERATIONS=2 SET='grpo.tasks_per_iter=2;grpo.group_size=2;suite.n=4;eval_suite.n=2' \
#     INIT= MODEL=Qwen/Qwen3-0.6B bash scripts/train/70_grpo_online.sh                 # smoke
#   GPUS=8 SEEDS="0 1 2" bash scripts/train/70_grpo_online.sh                          # 3 seeds
#   GPUS=8 SEEDS="0" SET='grpo.gigpo.enabled=false' OUT=checkpoints/r8-grpo bash scripts/train/70_grpo_online.sh
source "$(dirname "$0")/_common.sh"

: "${CONFIG:=configs/train/grpo_online_jevgym.yaml}"
: "${INIT=$CKPT/r4}"
: "${OUT:=$CKPT/r8}"
: "${EXECUTOR:=config}"
: "${VLLM_MODEL:=Qwen/Qwen3-8B}"
: "${VLLM_BASE_URL:=http://localhost:8000/v1}"
: "${ITERATIONS:=}"
: "${SEEDS:=0}"
: "${SET:=}"
: "${RESUME:=}"

need "$CONFIG"
sets=()
while IFS= read -r kv; do
  sets+=("$kv")
done < <(split_list "$SET" ";")
case "$EXECUTOR" in
  config) ;;
  sim) sets+=("executor={provider: simulated, model: scripted, error_rate: 0.1, simulated_latency_ms: 0}") ;;
  gemini-lite) sets+=("executor={provider: gemini, model: gemini-3.5-flash-lite, temperature: 0.7, reasoning_effort: minimal}") ;;
  gemini-flash) sets+=("executor={provider: gemini, model: gemini-3.5-flash, temperature: 0.7, reasoning_effort: minimal}") ;;
  vllm) sets+=("executor={provider: vllm, model: $VLLM_MODEL, base_url: '$VLLM_BASE_URL', temperature: 0.7}") ;;
  *) log "unknown EXECUTOR=$EXECUTOR"; exit 2 ;;
esac

read -r -a seed_list <<< "$SEEDS"
if [[ -n "$INIT" ]]; then
  need "$INIT"
  sets+=("init=$INIT")
else
  sets+=("init=null" "ref=base")
fi
pids=()
i=0
for seed in "${seed_list[@]}"; do
  out="$OUT"
  if (( ${#seed_list[@]} > 1 )); then
    out="$OUT-s$seed"
  fi
  args=(--config "$CONFIG" --model "$MODEL" --out "$out" --set "${sets[@]}" "seed=$seed")
  if [[ -n "$ITERATIONS" ]]; then
    args+=(--iterations "$ITERATIONS")
  fi
  if [[ -n "$RESUME" ]]; then
    args+=(--resume "$RESUME")
  fi
  if [[ "$DRY_RUN" == 1 && $i == 0 ]]; then
    "$PY" -m jevharness.training.grpo_online "${args[@]}" --print-config
  fi
  gpu=$(( i % (GPUS > 0 ? GPUS : 1) ))
  if (( ${#seed_list[@]} > 1 )); then
    CUDA_VISIBLE_DEVICES="$gpu" run "$PY" -m jevharness.training.grpo_online "${args[@]}" &
    pids+=($!)
  else
    run "$PY" -m jevharness.training.grpo_online "${args[@]}"
  fi
  i=$(( i + 1 ))
done
for p in ${pids[@]+"${pids[@]}"}; do
  wait "$p"
done
log "done -> $OUT (metrics.jsonl per run; next: 90_calibrate_serve_eval.sh ADAPTER=$OUT)"
