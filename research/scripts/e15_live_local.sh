#!/usr/bin/env bash
# Exploratory row E15 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E15: more decision-model baselines" (c) 4, (d)):
# the OpenAI Decisions API live row from the coordinating container (its key never goes to AWS). The same pause rule as
# the box's run_live (scripts/aws/e15_baselines.sh): start after two consecutive clean probe rounds; a watchdog pauses
# the runner on an overload error and caps the spend; resume after two clean rounds; errored episodes re-run once.
# Keys come from .env into this local process tree only; never printed.
#   bash scripts/e15_live_local.sh oaidec
set -uo pipefail
SLUG=${1:?usage: e15_live_local.sh SLUG}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PYB=.venv/bin/python
EX=scripts/explore_e15_baselines.py
OUT=results/explore/e15_baselines
LOG=$OUT/live_local_$SLUG.log
CAP_USD=${CAP_USD:-35}
STOP=runs/e15_STOP.json
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
run_live() {  # run_live CONFIG RUN_DIR
  local cfg=$1 run=$2 pid wpid round
  for round in 1 2 3 4 5 6 7 8; do
    [[ -f "$STOP" ]] && { say "spend cap reached earlier; $cfg not run further"; return 3; }
    probe_ok || { say "probes not clean; $cfg paused (round $round)"; return 4; }
    say "$cfg: start (round $round)"
    setsid env JEV_EVAL_RUNS=1 .venv/bin/jevbench run "$cfg" --out runs >> "$OUT/live_runner_$SLUG.log" 2>&1 < /dev/null &
    pid=$!
    "$PYB" "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --every 30 >> "$OUT/live_watch_$SLUG.log" 2>&1 &
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
  [[ -f "$STOP" ]] && return 3
  local ne
  ne=$("$PYB" "$EX" errored --run "$run")
  if [[ "$ne" != 0 ]]; then
    say "$ne errored episode(s) in $run: one re-run after clean probes"
    probe_ok || { say "probes not clean; re-run skipped"; return 0; }
    setsid env JEV_EVAL_RUNS=1 .venv/bin/jevbench run "$cfg" --out runs >> "$OUT/live_runner_$SLUG.log" 2>&1 < /dev/null &
    pid=$!
    "$PYB" "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --every 30 --no-pause >> "$OUT/live_watch_$SLUG.log" 2>&1 &
    wpid=$!
    wait "$pid"; kill "$wpid" 2>/dev/null
  fi
  say "$cfg done; errored now $("$PYB" "$EX" errored --run "$run"); $("$PYB" "$EX" spend --runs runs --short 2>/dev/null)"
  return 0
}
run_live "configs/explore/e15_baselines/live_smoke_$SLUG.yaml" "runs/e15_live_smoke_$SLUG" || exit $?
"$PYB" "$EX" gate --runs runs --smoke "runs/e15_live_smoke_$SLUG" --n 200 --cap "$CAP_USD" | tee -a "$LOG" || { say "gate refuses"; exit 3; }
run_live "configs/explore/e15_baselines/live_$SLUG.yaml" "runs/e15_live_$SLUG"
