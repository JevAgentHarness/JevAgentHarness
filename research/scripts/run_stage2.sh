#!/usr/bin/env bash
# Stage-2 JevGym evaluation runs (docs/PREREG_A7_STAGE2.md A7.6): launch gate, start record, three passes, report.
#
#   RUN=stage2_jevgym EXCLUDE=llm_only_s2lite nohup setsid bash scripts/run_stage2.sh > /dev/null 2>&1 &
#   RUN=stage2_jevgym TAG=stage2_jevgym_sl4b GATE_ARGS=--startlux WORKERS=4 bash scripts/run_stage2.sh \
#       -o 's1_backends.startlux_4b={type: jev, ...}'                  # extra args go to every `jevbench run`
#   PASSES="2 3" RUN=... bash scripts/run_stage2.sh                     # resume after a container restart
#
# - The gate (`scripts/stage2_spec.py --gate <GATE_RUN> <GATE_ARGS>`) must pass, else nothing starts (A7.17 (c)).
# - results/stage2/START_<TAG>.txt is written once, in the run_main.sh format (step, started, commit, plan sha256 of
#   docs/PREREG_A7_STAGE2.md); a resume never overwrites it (record RESUME_<TAG>.txt by hand).
# - EXCLUDE (space-separated arm names) drops arms that the registration does not run although the config lists them
#   (e.g. A7.1 (g) 5: the fresh arms of a test whose profile section is empty); the arm list is passed as one
#   `-o arms=[...]` override, so every pass and every later invocation must use the same EXCLUDE.
# - Only this script sets JEV_EVAL_RUNS=1, and only for the evaluation commands. Pass log:
#   runs/stage2_logs/<TAG>.status ("pass N start|end <time> exit <code>"); full log runs/stage2_logs/<TAG>.log.
# While a config runs read only operational fields (errors, cost per episode, served model, unpriced models): no
# Delta, p-value or score card before its three passes are complete (A7.6).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
JB=${JB:-.venv/bin/jevbench}
RUN=${RUN:?set RUN (the experiment name)}
CFG=${CFG:-configs/experiments/$RUN.yaml}
GATE_RUN=${GATE_RUN:-$RUN}
GATE_ARGS=${GATE_ARGS:-}
TAG=${TAG:-$RUN}
WORKERS=${WORKERS:-8}
PASSES=${PASSES:-"1 2 3"}
EXCLUDE=${EXCLUDE:-}
LOG=runs/stage2_logs
mkdir -p "$LOG" results/stage2
exec >> "$LOG/$TAG.log" 2>&1

echo "== $(date -u +%FT%TZ) $TAG: gate $GATE_RUN $GATE_ARGS"
# shellcheck disable=SC2086
"$PY" scripts/stage2_spec.py --gate "$GATE_RUN" $GATE_ARGS || { echo "gate FAILED: nothing started"; exit 2; }
if [ ! -f "results/stage2/START_$TAG.txt" ]; then
  printf 'step %s\nstarted %s\ncommit %s\nplan sha256 %s\nconfig %s\nworkers %s\nexcluded arms %s\nextra args %s\n' \
    "$TAG" "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" \
    "$CFG" "$WORKERS" "${EXCLUDE:-none}" "$(printf '%s ' "$@" | sed -E 's/(api_key|key)[^,}]*/\1: <redacted>/g')" \
    > "results/stage2/START_$TAG.txt"
fi
args=()
if [ -n "$EXCLUDE" ]; then
  arms=$("$PY" - "$CFG" "$EXCLUDE" <<'EOF'
import json, sys, yaml
exp = yaml.safe_load(open(sys.argv[1]))
drop = set(sys.argv[2].split())
names = {a["name"] for a in exp["arms"]}
assert drop <= names, f"unknown arm(s) {sorted(drop - names)}"
print(json.dumps([a for a in exp["arms"] if a["name"] not in drop]))
EOF
) || { echo "arm list FAILED"; exit 2; }
  args+=(-o "arms=$arms")
fi
for pass in $PASSES; do
  echo "pass $pass start $(date -u +%FT%TZ)" >> "$LOG/$TAG.status"
  echo "== $(date -u +%FT%TZ) $TAG pass $pass"
  JEV_EVAL_RUNS=1 "$JB" run "$CFG" --out runs -o "workers=$WORKERS" "${args[@]}" "$@"
  rc=$?
  echo "pass $pass end $(date -u +%FT%TZ) exit $rc" >> "$LOG/$TAG.status"
done
"$JB" report "runs/$RUN" > "runs/$RUN/report.md" 2>/dev/null
echo "done $(date -u +%FT%TZ)" >> "$LOG/$TAG.status"
echo "== $(date -u +%FT%TZ) $TAG done (runs/$RUN/report.md: check the errors column only)"
