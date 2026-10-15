#!/usr/bin/env bash
# Run the JevAgentBench experiment registry (docs/ROADMAP.md §5) at full scale.
#
#   bash scripts/run_experiments.sh                 # E1 E2a E2b E3 E4 E6 E7 (API-only stages)
#   bash scripts/run_experiments.sh E1 E2a          # selected stages
#   N=20 WORKERS=4 bash scripts/run_experiments.sh E2a   # quick pilot of a stage
#
# Env knobs:
#   OUT       output root (default runs/)            WORKERS   parallel episodes (default 8)
#   N         override number of tasks (pilot)       DRY_RUN=1 print commands only
#   NEAR=1    also run E2a with near-duplicate distractors (JevGym v2)
# Stages needing other scripts: E5 tau2 -> scripts/run_tau2.sh ; E8 SWE-bench -> adapters/mini_swe/run_swebench.sh ;
# E9 Terminal-Bench -> adapters/pi/README.md (Harbor) ; T* training -> scripts/train/.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-runs}
WORKERS=${WORKERS:-8}
# Figures / numbers go into the paper only for full-scale runs; N=<pilot> runs keep them under $OUT.
if [ -n "${N:-}" ]; then GEN=${GEN:-$OUT/generated}; FIG=${FIG:-$OUT/figures}
else GEN=${GEN:-paper/generated}; FIG=${FIG:-paper/figures/generated}; fi
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(E1 E2a E2b E3 E4 E6 E7)

# PREREGISTRATION A6.13: the evaluation suite (JevGym seeds 200000-200499, frozen list configs/eval/jevgym_eval_tasks.json)
# runs only for the full-scale main runs after the freeze (the runner refuses it otherwise); a pilot (N set) runs on
# development seeds 0.. so no evaluation task is exposed again.
if [ -n "${N:-}" ]; then PILOT_SEEDS="-o suite.seed0=0"; else export JEV_EVAL_RUNS=1; PILOT_SEEDS=""; fi
run() { echo "+ $*"; [ "${DRY_RUN:-0}" = 1 ] || "$@"; }
ntask() { [ -n "${N:-}" ] && echo "--max-tasks $N $PILOT_SEEDS" || true; }
report() {  # markdown report + score card next to the runs
  local dir="$1"
  [ "${DRY_RUN:-0}" = 1 ] && return 0
  jevbench report "$dir" > "$dir/report.md" || true
  python -m jevharness.bench.scorecard "$dir" --baseline "${2:-llm_only}" --format md > "$dir/scorecard.md" 2>/dev/null || true
  echo "  -> $dir/report.md, $dir/scorecard.md"
}

fig() {  # paper figure from a run (never fails the stage); FIG defaults to paper/figures/generated
  local kind="$1"; shift
  [ "${DRY_RUN:-0}" = 1 ] && { echo "+ plots $kind $*"; return 0; }
  python -m jevharness.bench.plots "$kind" "$@" 2>&1 | tail -2 || true
}

for st in "${STAGES[@]}"; do
  echo "================ $st ================"
  case "$st" in
    E1)   # Level-1 static decisions
      run jevbench decide configs/decision/bfcl_when2call.yaml --out "$OUT" ${N:+--max-items $N}
      fig reliability "$OUT/e1_bfcl_when2call" --backends jev,gemini_lite_dist,gemma4_dist --out "$FIG/reliability_e1.pdf"
      ;;
    E2a)  # object-level decision points x modes x backends (+ tau sweeps)
      run jevbench run configs/experiments/jevgym_tool_select.yaml --out "$OUT" $(ntask) -o workers=$WORKERS
      report "$OUT/e2a_jevgym_tool_decisions"
      fig pareto "$OUT/e2a_jevgym_tool_decisions" --out "$FIG/pareto_e2a.pdf"
      fig sweep "$OUT/e2a_jevgym_tool_decisions" --prefix jev_ts_cascade_tau --out "$FIG/tau_sweep_e2a.pdf"
      if [ "${NEAR:-0}" = 1 ]; then
        run jevbench run configs/experiments/jevgym_tool_select.yaml --out "$OUT" $(ntask) \
          -o workers=$WORKERS name=e2a_jevgym_tool_decisions_near suite.distractor_style=near
        report "$OUT/e2a_jevgym_tool_decisions_near"
      fi
      ;;
    E2b)  # meta-level decision points
      run jevbench run configs/experiments/jevgym_meta_control.yaml --out "$OUT" $(ntask) -o workers=$WORKERS
      report "$OUT/e2b_jevgym_meta_control"
      fig pareto "$OUT/e2b_jevgym_meta_control" --out "$FIG/pareto_e2b.pdf"
      ;;
    E3)   # calibration: fit Jev temperatures on E2a decisions (gold from JevGym), rerun calibrated arms
      mkdir -p "$OUT/cal"
      run python -m jevharness.decision.calibration fit "$OUT"/e2a_jevgym_tool_decisions/jev_ts_replace.jsonl \
        "$OUT"/e2a_jevgym_tool_decisions/jev_ts_params_replace.jsonl --out "$OUT/cal/jev_jevgym.json"
      run jevbench run configs/experiments/jevgym_tool_select.yaml --out "$OUT" $(ntask) -o workers=$WORKERS \
        name=e3_jevgym_calibrated \
        "s1_backends={jev_ts: {type: calibrated, inner: {type: jev, provider: openrouter}, temperatures_file: $OUT/cal/jev_jevgym.json}}" \
        "arms=[{name: llm_only}, {name: jevts_ts_params_replace, s1: jev_ts, tool_select: {mode: replace}, tool_params: {mode: replace}}]" \
        "sweeps=[{base: {name: jevts_ts_cascade, s1: jev_ts, tool_select: {mode: cascade}, tool_params: {mode: cascade, tau: 0.9}}, set: tool_select.tau, values: [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]}]"
      report "$OUT/e3_jevgym_calibrated"
      ;;
    E4)   # scaling: tool-set size and observation noise
      SUB='[{name: llm_only}, {name: jev_ts_params_replace, tool_select: {mode: replace}, tool_params: {mode: replace}}, {name: jev_ts_assist_k3, tool_select: {mode: assist, k: 3}}, {name: jev_ts_cascade_tau0.9, tool_select: {mode: cascade, tau: 0.9}, tool_params: {mode: cascade, tau: 0.9}}]'
      for d in 10 30 60 120; do
        run jevbench run configs/experiments/jevgym_tool_select.yaml --out "$OUT" $(ntask) -o workers=$WORKERS \
          name=e4_scale_d$d suite.n_distractors=$d "arms=$SUB" "sweeps=[]"
        report "$OUT/e4_scale_d$d"
      done
      for z in 0 50; do
        run jevbench run configs/experiments/jevgym_tool_select.yaml --out "$OUT" $(ntask) -o workers=$WORKERS \
          name=e4_noise_$z suite.noise_lines=$z "arms=$SUB" "sweeps=[]"
        report "$OUT/e4_noise_$z"
      done
      ;;
    E6)   # ALFWorld action-only (needs ALFWORLD_DATA, see docs/ALFWORLD.md)
      run jevbench run configs/experiments/alfworld_action_only.yaml --out "$OUT" $(ntask) -o workers=$WORKERS
      report "$OUT/e6_alfworld_action_only" llm_react
      ;;
    E7)   # memory QA (LoCoMo; add configs/memory/longmemeval.yaml after downloading the data)
      run python -m jevharness.bench.suites.memory_qa --config configs/memory/locomo.yaml --out "$OUT/memory"
      ;;
    *) echo "unknown stage $st (see docs/ROADMAP.md §5)"; exit 2 ;;
  esac
done

# Refresh the paper's numbers from everything under $OUT (simulated runs are skipped) and list the
# \res{} cells that are still empty.
if [ "${DRY_RUN:-0}" != 1 ]; then
  python -m jevharness.bench.paper_numbers "$OUT" --out "$GEN/results.tex" --json "$GEN/results.json" \
    --check paper || true
fi
