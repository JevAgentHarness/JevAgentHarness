#!/usr/bin/env bash
# Registered API main runs (docs/MAIN_RUN_PLAN.md steps 2-6), in order, three passes per evaluation config.
#
#   STEPS="E1 E2a C7 E2b E2c" WORKERS=8 bash scripts/run_main.sh        # default: all five
#
# Start record instead of a freeze tag (authors' decision 2026-10-02: no formal freeze before submission): refuses to
# start unless the analysis plan and every registered config are committed and unchanged in the working tree,
# prereg_check passes and PREREG_EFFECTIVE.md is current; it then writes the starting commit and time to
# results/main/START_<step>.txt (committed afterwards), so the paper can state which plan preceded each run. Only this
# script sets JEV_EVAL_RUNS=1, and only for the evaluation commands (MAIN_RUN_PLAN §0). GPU steps (1, 7) and the tau^2
# steps (8, 9) are run separately (MAIN_RUN_PLAN §2).
#
# During a run inspect operational fields only (errors, cost per episode, served model); no Delta, p-value or score card
# on a primary arm before its runs are complete (MAIN_RUN_PLAN §0). Logs: runs/main_logs/<step>.log.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
JB=${JB:-.venv/bin/jevbench}
WORKERS=${WORKERS:-8}
STEPS=${STEPS:-"E1 E2a C7 E2b E2c"}
LOG=runs/main_logs
mkdir -p "$LOG"

start_guard() {
  local dirty
  dirty=$(git status --porcelain -- docs/PREREGISTRATION.md docs/PREREG_EFFECTIVE.md configs/ scripts/primaries.py \
          src/jevharness/bench/metrics.py src/jevharness/bench/registry.py)
  [ -z "$dirty" ] || { echo "refusing: uncommitted changes in the plan or registered configs:"; echo "$dirty"; exit 2; }
  "$PY" scripts/prereg_check.py --quiet || { echo "refusing: prereg_check fails"; exit 2; }
  "$PY" scripts/prereg_effective.py --check >/dev/null || { echo "refusing: PREREG_EFFECTIVE.md is stale"; exit 2; }
}

record_start() {  # record_start <step>
  mkdir -p results/main
  printf 'step %s\nstarted %s\ncommit %s\nplan sha256 %s\n' "$1" "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" \
    "$(sha256sum docs/PREREGISTRATION.md | cut -c1-16)" > "results/main/START_$1.txt"
}

three_passes() {  # three_passes <config> <run dir> [extra jevbench args...]
  local cfg=$1 dir=$2; shift 2
  for pass in 1 2 3; do
    echo "== $(date -u +%FT%TZ) $cfg pass $pass"
    JEV_EVAL_RUNS=1 "$JB" run "configs/experiments/$cfg" --out runs -o "workers=$WORKERS" "$@"
  done
  "$JB" report "runs/$dir" > "runs/$dir/report.md"
  echo "== $(date -u +%FT%TZ) $cfg done; report: runs/$dir/report.md (check the errors column)"
}

start_guard
echo "start guard passed at $(git rev-parse --short HEAD)"

for step in $STEPS; do
  record_start "$step"
  case $step in
    E1)   # step 2: C8 (E1 logs no attempt count; a second invocation only re-asks errored items, cache hits are free)
      { "$JB" decide configs/decision/bfcl_when2call.yaml --out runs
        "$JB" decide configs/decision/bfcl_when2call.yaml --out runs; } 2>&1 | tee "$LOG/E1.log" ;;
    E2a)  # step 3: C1F C1L C2F C2L, C3's arm, the C7 candidates
      three_passes jevgym_tool_select.yaml e2a_jevgym_tool_decisions 2>&1 | tee "$LOG/E2a.log" ;;
    C7)   # step 4: the C7 configuration, fixed on E2 before any tau^2 run (A5.4 (a)); commit the JSON afterwards
      "$PY" scripts/primaries.py --runs runs --select-c7 --out results/primaries --strict 2>&1 | tee "$LOG/C7.log" ;;
    E2b)  # step 5: C4, C6F C6L, measurement arms
      three_passes jevgym_meta_control.yaml e2b_jevgym_meta_control 2>&1 | tee "$LOG/E2b.log" ;;
    E2c)  # step 6: pass 1 (all but the three secondary placeholders), rates, pass 2 (the placeholders)
      { three_passes baselines_jevgym.yaml e2c_jevgym_baselines
        read -r R1 R2 R3 < <("$PY" - <<'EOF'
from jevharness.decision.controls import trigger_rates as t
r = "runs/e2c_jevgym_baselines"
print(t(f"{r}/jev_stop_guard_s2lite.jsonl")["stop/guard"]["rate"], t(f"{r}/jev_route_tau0.5.jsonl")["routing/trigger"]["rate"],
      t(f"{r}/jev_memw.jsonl")["memory_write/trigger"]["rate"])
EOF
)
        echo "matched-rate secondary controls: ratem_stop_lite=$R1 ratem_route=$R2 ratem_memw=$R3 (A4, A6.18 (f))"
        three_passes baselines_jevgym.yaml e2c_jevgym_baselines \
          -o "s1_backends.ratem_stop_lite={type: rate, rate: $R1, seed: 0}" \
          -o "s1_backends.ratem_route={type: rate, rate: $R2, seed: 0}" \
          -o "s1_backends.ratem_memw={type: rate, rate: $R3, seed: 0}"; } 2>&1 | tee "$LOG/E2c.log" ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
done
echo "== $(date -u +%FT%TZ) steps done: $STEPS"
