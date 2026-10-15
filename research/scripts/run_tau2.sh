#!/usr/bin/env bash
# tau2-bench arms for the System-1 x harness study (see docs/TAU2.md).
#
# For every domain x arm: run tau2 with our registered `jev_agent` (configs/tau2/<arm>.yaml), then
# convert tau2's results.json + our side-car decision log into jevharness Trajectory JSONL and print
# `jevbench report` per domain (airline: also the A5.4 per-split summaries, see below).
#
#   scripts/run_tau2.sh                                   # everything below with the defaults
#   NUM_TRIALS=1 NUM_TASKS=5 DOMAINS=airline scripts/run_tau2.sh     # cheap pilot
#   ARMS="llm_only jev_guard" TAUS="0.8 0.9" scripts/run_tau2.sh
#   S1='{type: simulated, accuracy: 0.85}' scripts/run_tau2.sh       # no Jev key: simulated S1
#   DRY_RUN=1 scripts/run_tau2.sh                          # print the commands only
#
#   # PREREGISTRATION A5.4 (C7). The 50 airline tasks are split once (seed 0; configs/tau2/airline_task_ids.json;
#   # `python -m jevharness.bench.suites.tau2 split airline`) into 15 SHADOW tasks (tau-hat* refit input only) and
#   # the 35-task airline TEST set. SPLIT=shadow|test|all resolves to tau2's --task-ids; TASK_IDS="0 1 2" or
#   # TASK_IDS=<file> (JSON list / {"task_ids": [...]} / one id per line) overrides it.
#   #  1. shadow run: Jev logged, the LLM decides (shadow mode), on the 15 shadow tasks x 3 trials
#   DOMAINS=airline ARMS=jev_shadow SPLIT=shadow NUM_TRIALS=3 scripts/run_tau2.sh
#   #  2. refit tau-hat* with the H1 fit (docs/TAU2.md "A5.4 procedure"):
#   python -m jevharness.bench.suites.tau2 report runs/tau2/airline --split shadow --no-report   # -> runs/tau2/airline_shadow/
#   python -m jevharness.bench.threshold_theory runs/tau2/airline_shadow --arm jev_shadow --dp tool_select --format json
#   #  3. C7 (tau-hat* refit) + the E2-carried tau-hat* secondary row, every arm on ALL 50 airline tasks + retail:
#   ARMS="llm_only jev_ts_cascade" TAUS="<tau-hat* refit> <tau-hat* E2>" scripts/run_tau2.sh
#   #     fallback rows (A5.4 d; airline only, tested on the 35): ASSIST k=3 and the C1 arm if it differs from C7
#   DOMAINS=airline ARMS="jev_ts_assist_k3" scripts/run_tau2.sh
#   #  4. report: comparisons on the 35 test tasks (the script prints them for airline; by hand:)
#   python -m jevharness.bench.suites.tau2 report runs/tau2/airline --split test     # -> runs/tau2/airline_test/ + jevbench report
#   python -m jevharness.bench.suites.tau2 report runs/tau2/airline --split shadow   # the 15, descriptive only
#
#   # Gemini agent + user simulator (validated in the pilot, docs/PILOT_RESULTS.md); litellm maps
#   # reasoning_effort=minimal to thinkingLevel=low for gemini-3.5-*, so pass the thinking config itself:
#   AGENT_LLM=gemini/gemini-3.5-flash USER_LLM=gemini/gemini-3.5-flash-lite SMALL_LLM=gemini/gemini-3.5-flash-lite \
#   AGENT_LLM_ARGS='{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}' \
#   USER_LLM_ARGS='{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}' scripts/run_tau2.sh
#   # no OpenAI access: retail's NL-assertion judge (tau2 hard-codes gpt-4.1) -> any litellm model, fixed across arms
#   NL_JUDGE_LLM=gemini/gemini-3.5-flash ... scripts/run_tau2.sh
#
# The cascade sweep sets tool_select.tau and tool_params.tau to the same value.
# Needs OPENAI_API_KEY (agent / user simulator; retail NL-assertion judge = tau2's fixed gpt-4.1 unless
# NL_JUDGE_LLM is set) and
# TYPESAFE_API_KEY (or whatever the Jev backend reads, see docs/API_KEYS.md) for the Jev arms.
# Re-running resumes: tau2 skips finished (task, trial) pairs (--auto-resume), conversion is idempotent.
# Errored simulations (termination agent_error / user_error / infrastructure_error, or a JevTau2Agent exception) are
# re-run up to twice after each arm's run (`suites.tau2 rerun-errored`, PREREGISTRATION A6.22 (e) 4).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${AGENT_LLM:=gemini/gemini-3.5-flash}"   # System-2 model under test (litellm name); PREREGISTRATION A6.15: Flash agent, Flash-Lite user simulator
: "${USER_LLM:=gemini/gemini-3.5-flash-lite}"        # user simulator, fixed across arms
: "${SMALL_LLM:=gemini/gemini-3.5-flash-lite}"       # cheap model for jev_route
# Registered tau^2 System-2 (PREREGISTRATION A6.15): Gemini 3.5 Flash agent at temperature 0 with minimal thinking,
# Flash-Lite user simulator with the same settings; the args default to that whenever the model is a Gemini one.
GEMINI_ARGS='{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}'
case "$AGENT_LLM" in gemini/*) : "${AGENT_LLM_ARGS:=$GEMINI_ARGS}";; esac
case "$USER_LLM" in gemini/*) : "${USER_LLM_ARGS:=$GEMINI_ARGS}";; esac
: "${AGENT_LLM_ARGS:=}"              # JSON for tau2 --agent-llm-args (also used for SMALL_LLM), e.g. Gemini thinkingConfig
: "${USER_LLM_ARGS:=}"               # JSON for tau2 --user-llm-args
: "${NL_JUDGE_LLM:=}"                # retail NL-assertion judge (litellm name); empty = tau2's gpt-4.1-2025-04-14
: "${NL_JUDGE_LLM_ARGS:=}"           # JSON litellm args for it (default: temperature 0 + JSON response)
: "${DOMAINS:=airline retail}"
: "${ARMS:=llm_only jev_ts_cascade jev_ts_assist_k3 jev_guard jev_route}"
: "${TAUS:=0.5 0.7 0.8 0.9 0.95}"    # jev_ts_cascade sweep (tool_select.tau = tool_params.tau)
: "${NUM_TRIALS:=3}"                 # 1-3; pass^k is reported for k <= NUM_TRIALS
: "${NUM_TASKS:=}"                   # e.g. 5 for a pilot; empty = the whole split / task list
: "${TASK_SPLIT:=base}"              # tau2's split: base (airline 50 / retail 114) | test (20 / 40) | train
: "${SPLIT:=all}"                    # A5.4 airline split: all | shadow (15 tasks, tau-hat* refit) | test (35 tasks)
: "${TASK_IDS:=}"                    # explicit task ids ("0 1 2", "0,1,2" or a file); overrides SPLIT
: "${MAX_CONCURRENCY:=4}"            # simulations in flight (threads); --workers is not supported
: "${SEED:=300}"                     # tau2 default; same trial seeds for every arm (paired comparison)
: "${S1:=}"                          # optional S1 override for all Jev arms, e.g. '{type: oracle}'
: "${OUT:=runs/tau2}"
: "${PY:=python}"
: "${DRY_RUN:=}"
: "${MAX_SIM_ATTEMPTS:=3}"           # A6.22 (e) 4: an errored simulation runs at most 3 times (first + 2 re-runs)

case "$SPLIT" in all|shadow|test) ;; *) echo "SPLIT must be all|shadow|test (got '$SPLIT')" >&2; exit 2 ;; esac
OUT_ABS="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"

run() {
  echo "+ $*" >&2
  [[ -n "$DRY_RUN" ]] || "$@"
}

# task ids for <domain> on stdout (empty = every task of TASK_SPLIT); exit 1 = nothing to run in this domain
task_ids_for() {
  local domain=$1
  if [[ -n "$TASK_IDS" ]]; then
    if [[ -f "$TASK_IDS" ]]; then
      "$PY" -m jevharness.bench.suites.tau2 split --ids-file "$TASK_IDS" --format args
    else
      echo "$TASK_IDS" | tr ',' ' '
    fi
  elif [[ "$SPLIT" == all ]]; then
    echo ""
  elif [[ "$domain" == airline ]]; then
    "$PY" -m jevharness.bench.suites.tau2 split airline --which "$SPLIT" --format args
  elif [[ "$SPLIT" == test ]]; then
    echo ""   # no shadow split outside airline: every task is a test task (A5.4 c)
  else
    return 1  # SPLIT=shadow: no shadow tasks in this domain
  fi
}

run_arm() {  # run_arm <domain> <arm name> <config yaml> [extra --jev-* args...]
  local domain=$1 arm=$2 cfg=$3
  shift 3
  local save="$OUT_ABS/raw/$domain/$arm"
  local jev=(--jev-config "$cfg" --jev-name "$arm")
  [[ -n "$S1" && "$arm" != llm_only ]] && jev+=(--jev-set "s1=$S1")
  [[ -n "$NL_JUDGE_LLM" ]] && jev+=(--jev-nl-judge "$NL_JUDGE_LLM")
  [[ -n "$NL_JUDGE_LLM_ARGS" ]] && jev+=(--jev-nl-judge-args "$NL_JUDGE_LLM_ARGS")
  local tau=(--domain "$domain" --agent jev_agent --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM"
             --num-trials "$NUM_TRIALS" --task-split-name "$TASK_SPLIT" --max-concurrency "$MAX_CONCURRENCY"
             --seed "$SEED" --save-to "$save" --auto-resume)
  # shellcheck disable=SC2206  # ids are whitespace-separated by construction
  [[ -n "$IDS" ]] && tau+=(--task-ids $IDS)
  [[ -n "$NUM_TASKS" ]] && tau+=(--num-tasks "$NUM_TASKS")
  [[ -n "$AGENT_LLM_ARGS" ]] && tau+=(--agent-llm-args "$AGENT_LLM_ARGS")
  [[ -n "$USER_LLM_ARGS" ]] && tau+=(--user-llm-args "$USER_LLM_ARGS")
  local args=("${jev[@]}" ${1+"$@"} "${tau[@]}")
  run "$PY" -m jevharness.bench.suites.tau2 run "${args[@]}"
  # PREREGISTRATION A6.22 (e) 4: re-run every errored simulation (same task, trial and seed) up to twice, recording
  # the attempts in $save/jev_attempts.json (convert stamps them; scripts/primaries.py reads them)
  run "$PY" -m jevharness.bench.suites.tau2 rerun-errored "$save" --max-attempts "$MAX_SIM_ATTEMPTS" -- "${args[@]}"
  run "$PY" -m jevharness.bench.suites.tau2 convert "$save/results.json" --arm "$arm" \
      --out "$OUT_ABS/$domain/$arm.jsonl"
}

for domain in $DOMAINS; do
  if ! IDS="$(task_ids_for "$domain")"; then
    echo "=== $domain: SPLIT=$SPLIT selects no task in this domain (only airline has a shadow split); skipped" >&2
    continue
  fi
  [[ -n "$IDS" ]] && echo "=== $domain: task ids ($(echo $IDS | wc -w)): $IDS" >&2
  for arm in $ARMS; do
    case "$arm" in
      jev_ts_cascade)
        for tau in $TAUS; do
          run_arm "$domain" "jev_ts_cascade_tau$tau" configs/tau2/jev_ts_cascade.yaml \
            --jev-set "tool_select.tau=$tau" --jev-set "tool_params.tau=$tau"
        done ;;
      jev_route)
        run_arm "$domain" "$arm" configs/tau2/jev_route.yaml --jev-set "small_llm=$SMALL_LLM" ;;
      jev_shadow)  # A5.4 (c): LLM-only behaviour, Jev logged at tool_select / tool_params / stop (tau-hat* refit input)
        run_arm "$domain" "$arm" configs/tau2/jev_shadow.yaml ;;
      *)
        run_arm "$domain" "$arm" "configs/tau2/$arm.yaml" ;;
    esac
  done
  echo "=== $domain ===" >&2
  run "$PY" -m jevharness.bench.cli report "$OUT_ABS/$domain"
  if [[ "$domain" == airline ]]; then  # A5.4: comparisons on the 35 test tasks; the 15 shadow tasks descriptively
    run "$PY" -m jevharness.bench.suites.tau2 report "$OUT_ABS/$domain" --split test
    run "$PY" -m jevharness.bench.suites.tau2 report "$OUT_ABS/$domain" --split shadow --no-report
  fi
done
