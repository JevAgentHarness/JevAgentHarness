#!/usr/bin/env bash
# Baseline vs System-1 (Jev) arms of mini-swe-agent on one fixed SWE-bench subset.
#
#   adapters/mini_swe/run_swebench.sh                      # default arms (see ARMS below)
#   adapters/mini_swe/run_swebench.sh shadow stop_guard    # only these arms
#   N=5 WORKERS=2 adapters/mini_swe/run_swebench.sh shadow # smoke test
#   EVAL=local EVAL_PY=.venv-eval/bin/python adapters/mini_swe/run_swebench.sh   # + local SWE-bench harness
#   EVAL=sbcli adapters/mini_swe/run_swebench.sh           # + hosted evaluation (sb-cli, SWEBENCH_API_KEY)
#   # Gemini agent + Gemini stand-in S1 (validated in the L3 smoke, docs/HARNESS_INTEGRATION.md):
#   CONFIG_OVERLAY="adapters/mini_swe/gemini.yaml adapters/mini_swe/s1_gemini_standin.yaml" N=2 WORKERS=2 \
#     adapters/mini_swe/run_swebench.sh baseline action_guard
#   # huggingface.co blocked? SUBSET=<dir with test.jsonl in the SWE-bench schema> (see README "Offline dataset")
#
# Every arm runs on the SAME instance-id file ($IDS, created once), so arms are paired per instance.
# Re-running resumes: instances already in <arm>/preds.json are skipped (RUN_ARGS=--redo-existing to redo).
# Pseudo-arm `cheap_only` = the baseline arm run with the cheap model (routing reference point).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
CONFIG="${CONFIG:-$REPO/adapters/mini_swe/swebench_jev.yaml}"
CONFIG_OVERLAY="${CONFIG_OVERLAY:-}"     # extra config specs merged after $CONFIG (space-separated; files or key=value)
ENV_FILE="${ENV_FILE:-$REPO/.env}"       # API keys (litellm reads GEMINI_API_KEY / OPENAI_API_KEY ...); "" = skip
SUBSET="${SUBSET:-verified}"            # verified | lite
SPLIT="${SPLIT:-test}"
N="${N:-100}"                            # first N instances after mini-swe's seeded shuffle
OUT="${OUT:-$REPO/runs/swe_$(basename "$SUBSET")_${N}}"   # SUBSET may be a local dataset dir
IDS="${IDS:-}"                           # optional existing id file (one instance id per line)
WORKERS="${WORKERS:-4}"                  # parallel instances (Docker containers) per arm
MODEL="${MODEL:-}"                       # strong model; default: model.model_name in $CONFIG
CHEAP_MODEL="${CHEAP_MODEL:-}"           # routing target; default: jev.cheap_model.model_name in $CONFIG
RUN_ARGS="${RUN_ARGS:-}"                 # extra args for `mini_swe run`, e.g. "-c jev.s1.model=jev-1.13.0"
EVAL="${EVAL:-none}"                     # none | local | sbcli
EVAL_PY="${EVAL_PY:-python}"             # python of a venv with swebench==4.1.0 (EVAL=local)
EVAL_WORKERS="${EVAL_WORKERS:-8}"

# The Python entry point does not read .env; export its keys (values are never printed).
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi
CFG_ARGS=(-c "$CONFIG")
for spec in $CONFIG_OVERLAY; do CFG_ARGS+=(-c "$spec"); done

if [ "$#" -gt 0 ]; then
  ARMS=("$@")
else
  # `shadow` doubles as the baseline (identical agent behaviour, S1 decisions logged); add `baseline`
  # if you also want a run without any S1 latency/cost in the logs.
  ARMS=(shadow action_guard stop_guard stop_trigger route reflect_always reflect action_hint)
fi

case "$SUBSET" in
  verified) HF_DATASET="princeton-nlp/SWE-bench_Verified"; SB_SUBSET="swe-bench_verified" ;;
  lite)     HF_DATASET="princeton-nlp/SWE-bench_Lite";     SB_SUBSET="swe-bench_lite" ;;
  *)        HF_DATASET="$SUBSET";                          SB_SUBSET="$SUBSET" ;;
esac
# Local dataset directory (offline copy with <split>.jsonl): the SWE-bench harness needs the file itself.
if [ -d "$SUBSET" ] && [ -f "$SUBSET/$SPLIT.jsonl" ]; then
  SUBSET="$(cd "$SUBSET" && pwd)"; HF_DATASET="$SUBSET/$SPLIT.jsonl"
fi

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
if [ -z "$IDS" ]; then
  IDS="$OUT/instance_ids.txt"
  if [ ! -s "$IDS" ]; then
    echo ">> selecting $N instances of $SUBSET/$SPLIT -> $IDS"
    "$PY" -m jevharness.adapters.mini_swe ids --subset "$SUBSET" --split "$SPLIT" --shuffle --slice "0:$N" \
      -o "$IDS" >/dev/null
  fi
fi
echo ">> $(wc -l < "$IDS") instances, arms: ${ARMS[*]}, output: $OUT"

cfg_value() {  # cfg_value <python expr on the merged config dict `c`> ($CONFIG + $CONFIG_OVERLAY)
  "$PY" -c "import sys; from jevharness.adapters.mini_swe import load_config; c=load_config(sys.argv[1:]); print($1)" \
    "$CONFIG" $CONFIG_OVERLAY 2>/dev/null | tail -1
}

eval_arm() {  # eval_arm <arm dir name> -> prints the report path (or nothing)
  local name="$1" d="$OUT/$1"
  [ -s "$d/preds.json" ] || return 0
  mkdir -p "$d/eval"
  case "$EVAL" in
    local)
      # swebench 4.1.0: pulls the same docker.io/swebench/sweb.eval.x86_64.* images the agent used
      (cd "$d/eval" && "$EVAL_PY" -m swebench.harness.run_evaluation --dataset_name "$HF_DATASET" \
        --split "$SPLIT" --predictions_path "$d/preds.json" --max_workers "$EVAL_WORKERS" \
        --run_id "jev_$name" --report_dir "$d/eval") >&2
      ls "$d"/eval/*."jev_$name".json 2>/dev/null | head -1 || true ;;   # no report: warn below, don't abort
    sbcli)
      sb-cli submit "$SB_SUBSET" "$SPLIT" --predictions_path "$d/preds.json" --run_id "jev_$name" \
        --output_dir "$d/eval" >&2
      ls "$d/eval/${SB_SUBSET}__${SPLIT}__jev_$name.json" 2>/dev/null | head -1 || true ;;
  esac
}

for arm in "${ARMS[@]}"; do
  name="$arm"
  args=(--arm "$arm")
  model_args=()
  [ -n "$MODEL" ] && model_args+=(-m "$MODEL")
  [ -n "$CHEAP_MODEL" ] && model_args+=(--cheap-model "$CHEAP_MODEL")
  if [ "$arm" = "cheap_only" ]; then
    cheap="${CHEAP_MODEL:-$(cfg_value "c['jev']['cheap_model']['model_name']")}"
    args=(--arm baseline --name cheap_only)
    model_args=(-m "$cheap")
  fi
  echo ">> [$name] running"
  # shellcheck disable=SC2086  # RUN_ARGS is intentionally word-split
  "$PY" -m jevharness.adapters.mini_swe run "${CFG_ARGS[@]}" "${args[@]}" ${model_args[@]+"${model_args[@]}"} \
    --subset "$SUBSET" --split "$SPLIT" --instance-ids "$IDS" -o "$OUT/$name" -w "$WORKERS" $RUN_ARGS

  report=""
  if [ "$EVAL" != "none" ]; then
    echo ">> [$name] evaluating ($EVAL)"
    report="$(eval_arm "$name")"
    [ -n "$report" ] || echo "!! [$name] no evaluation report found in $OUT/$name/eval" >&2
  fi
  echo ">> [$name] collecting -> $OUT/$name.jsonl"
  "$PY" -m jevharness.adapters.mini_swe collect "$OUT/$name" ${report:+--report "$report"} -o "$OUT/$name.jsonl"
done

echo ">> done. Per-arm trajectories: $OUT/<arm>.jsonl (jevharness Trajectory format; success = resolved)"
