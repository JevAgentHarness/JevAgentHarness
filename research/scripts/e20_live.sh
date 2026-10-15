#!/usr/bin/env bash
# E20 live driver, coordinating container only (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E20: decision-model
# ensembles and a hosted decision API behind the gate" (d), (g)). No GPU, no AWS: the OpenAI key and
# OPENROUTER_API_KEY_JEV never leave this container (jevbench loads .env; only decision/jev.py reads the Jev-only key).
#
#   bash scripts/e20_live.sh smoke oaidec|agree   # 2 development tasks (seeds 912-913 / 914-915), never reported
#   bash scripts/e20_live.sh gate  oaidec|agree   # the smoke's projection per provider must fit the caps
#   bash scripts/e20_live.sh live  oaidec|agree   # the 200 tasks (JEV_EVAL_RUNS=1)
#
# E14's pause-and-resume rule: start only after two consecutive clean probe rounds (six one-word Flash calls each,
# logged to results/explore/e20_ensembles/api_probe_log.txt); a watchdog stops the runner on an overload error (and at
# any provider's cap); probes every 5 minutes, resume unchanged after two clean rounds (the runner skips completed
# tasks); once every task is covered, errored episodes are re-run once after two clean rounds; nothing after that.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
S=scripts/explore_e20_ensembles.py
LOG=results/explore/e20_ensembles/live_driver.log
kind=${1:?smoke|gate|live}
row=${2:?oaidec|agree}
case "$row" in oaidec|agree) ;; *) echo "unknown row $row"; exit 2 ;; esac
mkdir -p results/explore/e20_ensembles
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

clean_twice() {
  local n=0 i
  for i in $(seq 1 72); do
    if $PY $S probe > /dev/null 2>&1; then n=$((n + 1)); [[ $n -ge 2 ]] && return 0
    else n=0; sleep 300; fi
  done
  return 1
}

run_once() {  # $1 config, $2 run name, $3 eval flag (1 = JEV_EVAL_RUNS=1)
  local cfg=$1 run=$2 ev=$3 pid wpid rc
  if [[ "$ev" == 1 ]]; then JEV_EVAL_RUNS=1 setsid .venv/bin/jevbench run "$cfg" --out runs >> "runs/${run}.runner.log" 2>&1 &
  else setsid .venv/bin/jevbench run "$cfg" --out runs >> "runs/${run}.runner.log" 2>&1 & fi
  pid=$!
  $PY $S watch --pid "$pid" --run "$run" --every 20 >> "runs/${run}.watch.log" 2>&1 &
  wpid=$!
  wait "$pid"; rc=$?
  wait "$wpid" 2>/dev/null
  return $rc
}

drive() {  # $1 config, $2 run name, $3 eval flag
  local cfg=$1 run=$2 ev=$3 round=1 err
  while :; do
    clean_twice || { say "$cfg: the API did not recover (no two clean probe rounds); stopping"; return 1; }
    say "$cfg: start (round $round)"
    run_once "$cfg" "$run" "$ev"
    if ls runs/e20_STOP_"$run".json > /dev/null 2>&1; then say "$cfg: a provider cap was reached; stopped"; return 1; fi
    if [[ -f "runs/$run/PAUSED_overload.json" ]]; then
      mv "runs/$run/PAUSED_overload.json" "runs/$run/paused_overload_round${round}.json"
      say "$cfg: paused on an overload error (round $round); waiting for clean probes"
      round=$((round + 1)); continue
    fi
    break
  done
  err=$($PY $S errored --run "runs/$run")
  if [[ "$err" -gt 0 ]]; then
    say "$cfg: $err errored episode(s); one re-run after two clean rounds"
    clean_twice && run_once "$cfg" "$run" "$ev"
    [[ -f "runs/$run/PAUSED_overload.json" ]] && mv "runs/$run/PAUSED_overload.json" "runs/$run/paused_overload_rerun.json"
  fi
  say "$cfg done; errored now $($PY $S errored --run "runs/$run"); $($PY $S spend --short)"
}

case "$kind" in
  smoke) drive "configs/explore/e20_ensembles/live_smoke_${row}.yaml" "e20_live_smoke_${row}" 0 ;;
  gate)  $PY $S gate --smoke "runs/e20_live_smoke_${row}" --n 200 | tee -a "$LOG" ;;
  live)  drive "configs/explore/e20_ensembles/live_${row}.yaml" "e20_live_${row}" 1 ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
