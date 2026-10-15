#!/usr/bin/env bash
# T4 data -- counterfactual Monte-Carlo Q-values for logged tool-selection states (training/mc_rollouts.py).
# For each sampled state s_t and K_ALT alternative tools a (incl. the one taken), the action prefix is replayed,
# a is forced, and the executor finishes the episode M times:  Q(s, a) = (wins + 1) / (M + 2).
# Output rows carry q_values (for DPO / exact PG / GRPO qvalue rewards) and target = p* ∝ pi_ref exp(Q/BETA).
#
# Variables (defaults in brackets):
#   RUNS        space-separated trajectory JSONL files                     [runs/train_llm_only/llm_only.jsonl]
#               (MC states must come from TRAINING tasks: run the executor on seeds disjoint from E2, e.g.
#                `jevbench run configs/experiments/jevgym_tool_select.yaml -o suite.seed0=10000 suite.n=500 \
#                 name=train_llm_only 'arms=[{name: llm_only}]' 'sweeps=[]'`  -- ONE -o, several KEY=VALUE)
#   EXP         experiment YAML whose `suite` produced RUNS and whose `llm` is the executor
#               [the experiment.yaml saved next to the first RUNS file by `jevbench run`]
#   MAX_STATES  states sampled (before sharding)                           [2000]
#   K_ALT / M   alternatives per state / rollouts per alternative         [4 / 4]
#   BETA        temperature of the MC target p*                           [0.2]
#   WORKERS     concurrent states per shard (threads; API executors)       [8]
#   SHARDS      parallel processes (disjoint state shards)                 [1]
#   OUT         output JSONL                                                [$DATA/mc.jsonl]
#   SEED / DRY_RUN
# Cost: MAX_STATES * K_ALT * M episodes (2000*4*4 = 32k). Budget with the executor's per-episode cost
# (Gemini 3.5 Flash ~ $0.011/JevGym episode; a vLLM-served executor costs GPU time only).
source "$(dirname "$0")/_common.sh"

: "${RUNS:=runs/train_llm_only/llm_only.jsonl}"
read -r -a run_list <<< "$RUNS"
: "${EXP:=$(dirname "${run_list[0]}")/experiment.yaml}"
: "${MAX_STATES:=2000}"
: "${K_ALT:=4}"
: "${M:=4}"
: "${BETA:=0.2}"
: "${WORKERS:=8}"
: "${SHARDS:=1}"
: "${OUT:=$DATA/mc.jsonl}"
: "${SEED:=0}"

need "$EXP" "${run_list[@]}"
run mkdir -p "$(dirname "$OUT")"
common=(--exp "$EXP" --runs "${run_list[@]}" --max-states "$MAX_STATES" --k-alt "$K_ALT" --m "$M" --beta "$BETA"
        --workers "$WORKERS" --seed "$SEED")

if (( SHARDS <= 1 )); then
  run "$PY" -m jevharness.training.mc_rollouts "${common[@]}" --out "$OUT"
else
  pids=()
  for (( i = 0; i < SHARDS; i++ )); do
    run "$PY" -m jevharness.training.mc_rollouts "${common[@]}" --num-shards "$SHARDS" --shard-id "$i" \
      --out "$OUT.shard$i" &
    pids+=($!)
  done
  for p in ${pids[@]+"${pids[@]}"}; do
    wait "$p"
  done
  shards=()
  for (( i = 0; i < SHARDS; i++ )); do
    shards+=("$OUT.shard$i")
  done
  run "$PY" -m jevharness.training.data mix --in "${shards[@]}" --out "$OUT" --seed "$SEED"
fi
run "$PY" -m jevharness.training.data stats "$OUT"
log "done -> $OUT (next: 40_kl_to_mc.sh, 50_dpo.sh, 55_pg_exact.sh)"
