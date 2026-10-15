#!/usr/bin/env bash
# Terminal-Bench-Science arms of pi (+ the Jev extensions) through Harbor: run_tb2.sh with TB-science defaults.
#
#   adapters/pi/run_tbsci.sh baseline jev_stop_reflect                    # published dataset from Harbor Hub
#   SUBSET=hard_pool adapters/pi/run_tbsci.sh baseline                     # one -i per task of tbsci_hard_pool.txt
#   TASKS_DIR=runs/tbsci/tasks adapters/pi/run_tbsci.sh baseline ...        # local flat dir from tbsci_tasks.sh
#   DRY_RUN=1 adapters/pi/run_tbsci.sh                                      # print the harbor commands only
#
# Arms, decision server and every other variable: see run_tb2.sh. Defaults that differ from it here:
#   DATASET   terminal-bench-science/terminal-bench-science@0.1.0 (Harbor Hub package; public, 70 tasks)
#   OUT       runs/tbsci
#   N_CONCURRENT 2         tasks ask for up to 4 CPUs / 16 GB each, and every image is built locally
#   AGENT_TIMEOUT_MULT 0.125  the published agent budget is 8 h per task; 0.125 caps it at 1 h. Set 1 (or empty)
#                          for the official budget. Any other value is a budget deviation: the instruction still says 8 h.
#   MAX_TURNS 100          pi turn cap (empty = none). pi does not compact within Gemini's 1M context, so input
#                          tokens per turn grow linearly and cost per trial roughly quadratically in turns
#                          (README: about $1.4-2.7 per trial at 100 turns, $5-10 at 200).
#   EXTRA_CA_VERIFIER 0    TB-science verifiers run in separate, no-network containers that do not get the mount
#   SUBSET    (unset)      all | hard_pool | sandbox | hard | <file>: restrict to the slugs of
#                          adapters/pi/tbsci_<SUBSET>.txt (or of <file>) via INCLUDE; `all` = no filter
# Task names: Hub tasks are named terminal-bench-science/<slug>, local ones <slug>; SUBSET handles both.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export DATASET="${DATASET:-terminal-bench-science/terminal-bench-science@0.1.0}"
export OUT="${OUT:-$REPO/runs/tbsci}"
export N_CONCURRENT="${N_CONCURRENT:-2}"
export AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT-0.125}"
export MAX_TURNS="${MAX_TURNS-100}"
export EXTRA_CA_VERIFIER="${EXTRA_CA_VERIFIER:-0}"
SUBSET="${SUBSET:-}"

if [ -n "$SUBSET" ] && [ "$SUBSET" != "all" ]; then
  f="$SUBSET"
  [ -f "$f" ] || f="$HERE/tbsci_${SUBSET}.txt"
  [ -f "$f" ] || { echo "!! unknown SUBSET $SUBSET (no $f)" >&2; exit 2; }
  prefix="terminal-bench-science/"
  [ -n "${TASKS_DIR:-}" ] && prefix=""
  inc=""
  while IFS= read -r s; do inc+="${inc:+ }$prefix$s"; done < <(sed 's/#.*//; s/[[:space:]]//g; /^$/d' "$f")
  [ -n "$inc" ] || { echo "!! SUBSET $SUBSET is empty" >&2; exit 2; }
  export INCLUDE="${INCLUDE:+$INCLUDE }$inc"
fi
exec bash "$HERE/run_tb2.sh" "$@"
