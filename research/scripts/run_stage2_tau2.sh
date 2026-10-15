#!/usr/bin/env bash
# Stage 2, S2-L3a (docs/PREREG_A7_STAGE2.md A7.2 (b), A7.3 (b), A7.15): the profile's tau^2 rendering vs LLM-only on the
# FRESH telecom domain (every task of configs/tau2/stage2/telecom_task_ids.json -- A7.15's random sample of n <= 800
# eligible tasks of the `full` pool -- x ONE attempt, the endpoint being single-attempt task success (A7.15 (e)); the
# A6.15 models; one seed for both arms so the user simulator is paired per task). The two arms run back to back with
# identical settings (tau2's own runner; outside the JevGym task-major scheduler, A7.15 (e) adaptation).
#
#   VALIDATION=1 bash scripts/run_stage2_tau2.sh    # A7.15 (b), (c): the 32 reserved validation tasks of
#                                                   #   configs/tau2/stage2/telecom_inventory.json -> runs/stage2_tau2_validation
#                                                   #   (adapter check + cost per simulation; no success number is read)
#   bash scripts/run_stage2_tau2.sh                 # both arms on the frozen sample; re-running resumes
#   DRY_RUN=1 bash scripts/run_stage2_tau2.sh       # print the commands only
#
# Preconditions (the script refuses otherwise):
#   * configs/harness/jevagent_best.yaml has `status: selected` and configs/tau2/stage2/jevagent_best.yaml exists (both written
#     by scripts/select_profile.py, A7.1 (h)); a rendering with every decision point off is not run (A7.1 (i));
#   * the inventory (scripts/telecom_stage2.py inventory) and, for the evaluation, the frozen sample
#     (scripts/telecom_stage2.py sample, after validation) exist.
# The ids come from the `full` pool, so tau2 is told `--task-split-name full` (its default split `base` holds 114 of
# them). Retries (A6.22 (e) 4): `suites.tau2 rerun-errored` re-runs an errored simulation up to twice; `convert` stamps
# attempts.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${PY:=python}"
: "${VALIDATION:=}"
: "${DOMAIN:=telecom}"
: "${NUM_TRIALS:=1}"          # A7.15 (e): one attempt per task (was 3 x majority pass before A7.15)
: "${TASK_SPLIT:=full}"
: "${MAX_CONCURRENCY:=4}"
: "${SEED:=300}"
: "${AGENT_LLM:=gemini/gemini-3.5-flash}"
: "${USER_LLM:=gemini/gemini-3.5-flash-lite}"
: "${DRY_RUN:=}"
GEMINI_ARGS='{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}'
INVENTORY="configs/tau2/stage2/${DOMAIN}_inventory.json"
if [ -n "$VALIDATION" ]; then
  : "${OUT:=runs/stage2_tau2_validation}"
  IDS_FILE="$INVENTORY"; IDS_KEY="validation"
else
  : "${OUT:=runs/stage2_tau2}"
  IDS_FILE="configs/tau2/stage2/${DOMAIN}_task_ids.json"; IDS_KEY=""
fi
PROFILE="configs/harness/jevagent_best.yaml"
RENDER="configs/tau2/stage2/jevagent_best.yaml"

[ -f "$IDS_FILE" ] || { echo "missing $IDS_FILE: write the $DOMAIN inventory / freeze the sample first (A7.15)" >&2; exit 2; }
grep -q '^status: selected' "$PROFILE" || { echo "$PROFILE is not selected yet (scripts/select_profile.py, A7.1 (h))" >&2; exit 2; }
[ -f "$RENDER" ] || { echo "$RENDER missing: the profile has no tau^2-applicable decision point; S2-L3a is not run (A7.1 (i))" >&2; exit 0; }

mapfile -t TASK_IDS < <($PY -c "
import json, sys
d = json.load(open(sys.argv[1]))
d = d[sys.argv[2]] if sys.argv[2] else d
ids = d['task_ids'] if isinstance(d, dict) else d
print('\n'.join(map(str, ids)))" "$IDS_FILE" "$IDS_KEY")
[ "${#TASK_IDS[@]}" -gt 0 ] || { echo "no task ids in $IDS_FILE" >&2; exit 2; }
OUT_ABS="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
TAU2_ARGS=(--agent jev_agent --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" --num-trials "$NUM_TRIALS"
  --max-concurrency "$MAX_CONCURRENCY" --seed "$SEED" --auto-resume --task-split-name "$TASK_SPLIT"
  --task-ids "${TASK_IDS[@]}" --agent-llm-args "$GEMINI_ARGS" --user-llm-args "$GEMINI_ARGS")

run() { if [ -n "$DRY_RUN" ]; then printf '%q ' "$@"; echo; else "$@"; fi; }

for arm in llm_only jevagent_best; do
  cfg="configs/tau2/${arm}.yaml"
  [ "$arm" = jevagent_best ] && cfg="$RENDER"
  RUN=(--jev-config "$cfg" --jev-name "$arm" --domain "$DOMAIN" --save-to "$OUT_ABS/raw/$DOMAIN/$arm" "${TAU2_ARGS[@]}")
  run "$PY" -m jevharness.bench.suites.tau2 run "${RUN[@]}"
  run "$PY" -m jevharness.bench.suites.tau2 rerun-errored "$OUT_ABS/raw/$DOMAIN/$arm" -- "${RUN[@]}"
  run "$PY" -m jevharness.bench.suites.tau2 convert "$OUT_ABS/raw/$DOMAIN/$arm/results.json" --arm "$arm" \
    --out "$OUT/$DOMAIN/$arm.jsonl"
done
