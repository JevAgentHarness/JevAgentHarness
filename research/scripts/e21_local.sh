#!/usr/bin/env bash
# Exploratory row E21, DecisionGym-Shift (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E21: a shifted DecisionGym
# split"): this container's live runs (the L2 sources and the local L3 rows: LLM-only, Jev, the Decisions API), with
# the pause rule of registration (g) (start after two consecutive clean probe rounds; a watchdog pauses the runner on an
# overload error and caps the spend; resume after two clean rounds; errored episodes re-run once) and the gate of (e).
# Keys come from .env into this local process tree only (Jev with its Jev-only key, decision/jev.py); never printed.
#   bash scripts/e21_local.sh l2_smoke | l2 | live_smoke | live | oaidec_smoke | oaidec
# Caps (registration (h)): Gemini $38 on this side (CAP_USD), Jev $5, OpenAI $5 (minus their replay spend).
set -uo pipefail
STAGE=${1:?usage: e21_local.sh STAGE}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PYB=.venv/bin/python
EX=scripts/explore_e21_shift.py
OUT=results/explore/e21_shift
LOG=$OUT/local_runs.log
CAP_USD=${CAP_USD:-38}
JEV_CAP=${JEV_CAP:-5}
OAI_CAP=${OAI_CAP:-5}
STOP=runs/e21_STOP.json
CFGD=configs/explore/e21_shift
mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
set -a; . ./.env; set +a          # keys into this local process tree only (as load_dotenv); never printed
probe_ok() {
  local end=$(( $(date +%s) + ${PROBE_WAIT_MIN:-240} * 60 )) clean=0
  while (( $(date +%s) < end )); do
    if "$PYB" "$EX" probe --log "$OUT/api_probe_log.txt" > /dev/null 2>&1; then clean=$((clean + 1)); else clean=0; fi
    (( clean >= 2 )) && return 0
    sleep $(( clean > 0 ? 30 : 300 ))
  done
  return 1
}
s1cap() {  # s1cap PROVIDER: that provider's cap minus its spend outside live runs (the replays)
  "$PYB" "$EX" ledger | "$PYB" -c "import json,sys; d=json.load(sys.stdin); print(max(0.0, float(sys.argv[2]) - d[sys.argv[1] + '_replay']))" "$1" "$2"
}
run_live() {  # run_live CONFIG RUN_DIR S1ARM S1CAP [extra jevbench args]
  local cfg=$1 run=$2 s1arm=$3 s1cap=$4 pid wpid round; shift 4
  for round in 1 2 3 4 5 6 7 8; do
    [[ -f "$STOP" ]] && { say "a cap was reached earlier ($STOP); $cfg not run further"; return 3; }
    probe_ok || { say "probes not clean; $cfg paused (round $round)"; return 4; }
    say "$cfg: start (round $round) $*"
    setsid env JEV_EVAL_RUNS=1 .venv/bin/jevbench run "$cfg" --out runs "$@" >> "$OUT/local_runner.log" 2>&1 < /dev/null &
    pid=$!
    "$PYB" "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --every 30 --s1-arm "$s1arm" \
      --s1-cap "$s1cap" >> "$OUT/local_watch.log" 2>&1 &
    wpid=$!
    wait "$pid"
    kill "$wpid" 2>/dev/null
    if [[ -f "$run/PAUSED_overload.json" ]]; then
      say "$cfg paused on an overload error (round $round); waiting for clean probes"
      mv -f "$run/PAUSED_overload.json" "$run/paused_overload_round$round.json"
      continue
    fi
    break
  done
  [[ -f "$STOP" ]] && { say "cap reached during $cfg: $(cat "$STOP" | tr '\n' ' ')"; return 3; }
  local ne
  ne=$("$PYB" "$EX" errored --run "$run")
  if [[ "$ne" != 0 ]]; then
    say "$ne errored episode(s) in $run: one re-run after clean probes"
    probe_ok || { say "probes not clean; re-run skipped"; return 0; }
    setsid env JEV_EVAL_RUNS=1 .venv/bin/jevbench run "$cfg" --out runs "$@" >> "$OUT/local_runner.log" 2>&1 < /dev/null &
    pid=$!
    "$PYB" "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --every 30 --no-pause --s1-arm "$s1arm" \
      --s1-cap "$s1cap" >> "$OUT/local_watch.log" 2>&1 &
    wpid=$!
    wait "$pid"; kill "$wpid" 2>/dev/null
  fi
  say "$cfg done; errored now $("$PYB" "$EX" errored --run "$run"); $("$PYB" "$EX" spend --runs runs --short)"
  return 0
}
gated() {  # gated SMOKE_RUN N ARMS CONFIG RUN_DIR S1ARM S1CAP: the registration (e) gate, then the run (first k if cut)
  local smoke=$1 n=$2 arms=$3 cfg=$4 run=$5 s1arm=$6 s1cap=$7 g nfit
  g=$("$PYB" "$EX" gate --smoke "$smoke" --runs runs --n "$n" --arms "$arms" --cap "$CAP_USD") || { say "gate refuses $cfg: $g"; return 3; }
  echo "$g" >> "$OUT/gates.jsonl"
  nfit=$("$PYB" -c "import json,sys; print(json.loads(sys.argv[1])['n_fit'])" "$g")
  say "gate for $cfg: $g"
  if [[ "$nfit" -lt "$n" ]]; then run_live "$cfg" "$run" "$s1arm" "$s1cap" --max-tasks "$nfit"
  else run_live "$cfg" "$run" "$s1arm" "$s1cap"; fi
}
case "$STAGE" in
  l2_smoke)    run_live "$CFGD/l2_sources_smoke.yaml" runs/e21_l2_smoke jev_ts "$(s1cap jev "$JEV_CAP")" ;;
  l2)          gated runs/e21_l2_smoke 100 2 "$CFGD/l2_sources.yaml" runs/e21_l2 jev_ts "$(s1cap jev "$JEV_CAP")" ;;
  live_smoke)  run_live "$CFGD/live_local_smoke.yaml" runs/e21_live_local_smoke jev_ts "$(s1cap jev "$JEV_CAP")" ;;
  live)        gated runs/e21_live_local_smoke 200 2 "$CFGD/live_local.yaml" runs/e21_live_local jev_ts "$(s1cap jev "$JEV_CAP")" ;;
  oaidec_smoke) run_live "$CFGD/live_oaidec_smoke.yaml" runs/e21_live_oaidec_smoke oaidec "$(s1cap openai "$OAI_CAP")" ;;
  oaidec)      gated runs/e21_live_oaidec_smoke 200 1 "$CFGD/live_oaidec.yaml" runs/e21_live_oaidec oaidec "$(s1cap openai "$OAI_CAP")" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
