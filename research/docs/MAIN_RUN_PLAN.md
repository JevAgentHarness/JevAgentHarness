# Main-run plan (after the `prereg-freeze` tag)

The run order, commands and budget for the registered main runs (the 11-test family of
[`PREREG_EFFECTIVE.md`](PREREG_EFFECTIVE.md) §1 and the registered secondary rows), and how they turn into the family-level
analysis (`scripts/primaries.py`). [`PREREGISTRATION.md`](PREREGISTRATION.md) is authoritative; [`HANDOFF.md`](HANDOFF.md)
covers keys, hosts and troubleshooting. Budget and wall-clock figures are **estimates from the development pilots**
(method in §6), never results.

## 0. Rules that hold for every step

- `JEV_EVAL_RUNS=1` appears **only** in the evaluation commands below (JevGym seeds 200000–200499, A6.13). Pilots, smoke
  tests and the dry run never set it (they use `-o suite.seed0=0`); `JEV_ALLOW_PENDING_TAU` is never set.
- No interim looks (§1): while a config runs, inspect **operational** fields only (errors, cost per episode, served model,
  unpriced models). No Δ, p-value, score card or `primaries.py` call on a primary arm before its runs are complete; the one
  registered exception is the C7 selection on E2a (step 4), which A5.4 (a) places before the τ² runs.
- Every config runs as the registered YAML (no `-o arms=[...]` subsets; `-o workers=...` is fine). The runner resumes:
  re-running the same command re-runs only tasks whose latest record is an infrastructure error. **Run every evaluation
  command three times in total** (first pass + the 2 registered extra attempts, §1); afterwards an errored task is final
  and the drop rule applies (`primaries.py` counts the attempts from the JSONL and flags tasks with retries left).
- After each config: `jevbench report runs/<exp> > runs/<exp>/report.md` (the `errors` column must be ≈ 0) and copy the
  cited summaries to `results/<exp>/` (HANDOFF §9).

## 1. Order and constraints

| # | Step | Host | Depends on | Registered constraint |
|---|---|---|---|---|
| 0 | Pre-flight: tag, checks, dry run | any | — | freeze checklist (HANDOFF §7.1) done |
| 1 | GPU round 3: Intern-Decision calibration arms + τ̂\* fits | GPU | 0 | A6.5 / A6.12 / A6.16 (b): fits written **before any Intern C1 arm runs** |
| 2 | E1 static decisions (C8) | API | 0 | population frozen (`e1_c8_population.json`, A6.16 (l)) |
| 3 | E2a (C1F C1L C2F C2L, C3's arm, the C7 candidates) | API | 0 | τ̂\* of the Jev C1 arms already fitted (A6.5; done 2026-10-01) |
| 4 | **C7 selection** on E2a | any | 3 | A5.4 (a): fixed before any τ² run; committed |
| 5 | E2b (C4, C6F C6L, measurement arms) | API | 0 | C4 τ = 0.4 fixed (A6.15); the C4 opportunity count is read before C4 is scored (A5.5; `primaries.py` does it) |
| 6 | E2c pass 1, then pass 2 (matched-rate secondary controls) | API | 0 | C5 control rate 0.1424 fixed in the YAML (A6.16 (i)); `ratem_stop_lite` / `ratem_route` / `ratem_memw` from this run's System-1 arms (A4, A6.18 (f)) |
| 7 | E2a' Intern-Decision arms | GPU (+ OpenRouter for its paired Jev arms) | 1 | Intern C1 arms only with fitted τ̂\* (`calibration: pending` removed) |
| 8 | τ² shadow (15 airline shadow tasks × 3) → τ̂\* refit | API | 4 | A5.4 (c): refit before the C7 arms; no τ² test task in the fit |
| 9 | τ² C7 arm, secondary rows, proxy-check arms | API | 8 | every arm on all 50 airline + 114 retail tasks × 3 |
| 10 | Analysis: size table at n′, `primaries.py`, score cards, H2, paper numbers | any | 1–9 | A6.17 (b): exact-size table re-run at the final n′ before the analysis is reported |

Steps 1 and 7 run on the GPU host in parallel with steps 2–6 on the API host. Steps 3, 5, 6 share Gemini keys: run them one
after the other (or at most two at once with `GEMINI_API_KEYS` rotation); they share `cache/s1.sqlite`.

## 2. Commands

Steps 2–6 (the API host) run unattended with `STEPS="E1 E2a C7 E2b E2c" WORKERS=8 bash scripts/run_main.sh`. The
authors decided on 2026-10-02 not to cut a formal freeze tag before submission; instead the script refuses to start
while the plan or a registered config has uncommitted changes or `prereg_check` / `prereg_effective --check` fail, and
writes the starting commit, time and plan hash to `results/main/START_<step>.txt` (committed). It runs each evaluation
config three times, writes the reports and logs to
`runs/main_logs/`, fills the three secondary matched-rate controls between the E2c passes, and runs the C7 selection
right after E2a (commit `results/primaries/c7_selection.json` before any τ² run). The commands below are what it runs.

### Step 0 — pre-flight (no evaluation seed touched)

```bash
git merge-base --is-ancestor prereg-freeze HEAD && echo "at or after the tag"   # FREEZE_COMMIT in PREREGISTRATION.md;
                                      # registered files unchanged since the tag: prereg_check below must pass
.venv/bin/python -m pytest -q && python scripts/prereg_check.py && python3 scripts/check_no_secrets.py
python scripts/prereg_effective.py --check
python scripts/dryrun_main_analysis.py --out /tmp/dryrun_main      # simulated: every registered arm + the whole analysis
                                                                   # chain on dev seeds; must exit 0 (~3 min at 30 tasks)
jevbench ping --s1 '{"type": "jev", "provider": "openrouter"}'     # served model must read typesafe/jev-1.13
OUT=runs/smoke_main N=10 WORKERS=4 bash scripts/run_experiments.sh E2a   # real-API smoke on DEV seeds (-o suite.seed0=0
                                      # is implied); OUT keeps it out of runs/e2a_*: dev episodes in a main run directory
                                      # would mix into its score cards (primaries.py flags them as an ERROR)
```

### Step 1 — GPU round 3 (calibration seeds 5000–5149; no `JEV_EVAL_RUNS`)

```bash
jevbench run configs/experiments/jevgym_shadow_cal_intern.yaml --out runs -o workers=4
for arm in intern_shadow_cal intern_shadow_cal_s2lite; do for dp in tool_select tool_params; do
  python -m jevharness.bench.threshold_theory runs/e2a_jevgym_shadow_cal_intern --arm $arm --dp $dp --calibration-arm \
    --task-value 1.0 --lam-per-hour 0 --format json --out results/calibration/tauhat_${arm}_${dp}.json
done; done
python scripts/measure_latency.py --s1 '{"type": "intern_decision", "model": "internlm/Intern-Decision-4B", "device": "auto"}' \
  --concurrency 1 4 --n 200 --state-tokens 1500 --questions 3 --gpu-hour-usd 1.32   # A5.1: replaces the assumed price
```

Then, before step 7: write each τ̂\* into `intern_ts_cascade_tauhat` / `intern_ts_cascade_tau0.9_s2lite` of
`jevgym_tool_select_intern.yaml` (a "never" fit → `always_defer: true`, A6.11 (c).4) and drop `calibration: pending`; move
the arms from `PENDING_TAUS` to `C1_TAUS` in `scripts/prereg_check.py`; add a dated Deviations line (as for the Jev fits);
re-run `python scripts/prereg_check.py` and `python scripts/prereg_effective.py`; commit.

### Step 2 — E1 (C8)

```bash
bash scripts/run_experiments.sh E1     # = jevbench decide configs/decision/bfcl_when2call.yaml --out runs
```
A second invocation re-runs everything (cache hits are free); when one backend shows errors (`n_errors` in
`summary.json`), re-run just that backend: `python -m jevharness.bench.decision_eval configs/decision/bfcl_when2call.yaml
--out runs --only jev` (keeps the other backends' records). E1 logs no attempt count: errors left after the re-runs are
final and drop their items from the C8 pair. C8 = `jev` vs `gemini_flash_typed` on the 1,114 BFCL tool_select items; `primaries.py` rebuilds the
population and checks its sha256.

### Step 3 — E2a (three passes)

```bash
for pass in 1 2 3; do JEV_EVAL_RUNS=1 jevbench run configs/experiments/jevgym_tool_select.yaml --out runs -o workers=8; done
jevbench report runs/e2a_jevgym_tool_decisions > runs/e2a_jevgym_tool_decisions/report.md
```

### Step 4 — C7 selection (before any τ² run)

```bash
python scripts/primaries.py --runs runs --select-c7 --out results/primaries --strict
git add results/primaries/c7_selection.json && git commit -m "C7 configuration selected on E2 (A5.4 (a))"
```
Selection rule (A6.9 #11, A6.18 (c)): the most cost-saving of LLM-only / the Flash Jev cascade at each swept τ and at τ̂\* /
assist k = 3 / the guard that is NI on E2 at the unadjusted level; within 2 pt of the best savings → tie; ties and the no-NI
case → the C1 arm. A cascade selection runs on τ² as the cascade at the **τ² refit** τ̂\* (A5.4 (c)); assist →
`jev_ts_assist_k3`; the guard → `jev_guard` (τ²'s policy guard; E2a's `jev_ts_guard` is read as its E2 counterpart — see §7).

### Step 5 — E2b

```bash
for pass in 1 2 3; do JEV_EVAL_RUNS=1 jevbench run configs/experiments/jevgym_meta_control.yaml --out runs -o workers=8; done
```
The measurement arms (`meta_shadow`, `meta_shadow_s2lite`) run counterfactual branches and probes after each episode
(≈ 3× the Flash-Lite episode spend, already in §5).

### Step 6 — E2c (pass 1: everything but the three secondary placeholders; pass 2: the placeholders)

```bash
for pass in 1 2 3; do JEV_EVAL_RUNS=1 jevbench run configs/experiments/baselines_jevgym.yaml --out runs -o workers=8; done
python - <<'EOF'
from jevharness.decision.controls import trigger_rates as t
r = "runs/e2c_jevgym_baselines"
print("ratem_stop_lite", t(f"{r}/jev_stop_guard_s2lite.jsonl")["stop/guard"]["rate"])        # ACCEPTANCE rate
print("ratem_route", t(f"{r}/jev_route_tau0.5.jsonl")["routing/trigger"]["rate"])
print("ratem_memw", t(f"{r}/jev_memw.jsonl")["memory_write/trigger"]["rate"])
EOF
for pass in 1 2 3; do JEV_EVAL_RUNS=1 jevbench run configs/experiments/baselines_jevgym.yaml --out runs -o workers=8 \
  -o 's1_backends.ratem_stop_lite={type: rate, rate: <r1>, seed: 0}' -o 's1_backends.ratem_route={type: rate, rate: <r2>, seed: 0}' \
  -o 's1_backends.ratem_memw={type: rate, rate: <r3>, seed: 0}'; done
```
`ratem_lite` (the C5 control, 0.1424) is never touched. Report the realised rates of both arms of every matched pair.

### Step 7 — E2a' Intern-Decision (GPU host; its paired Jev arms need `OPENROUTER_API_KEY`)

```bash
for pass in 1 2 3; do JEV_EVAL_RUNS=1 jevbench run configs/experiments/jevgym_tool_select_intern.yaml --out runs -o workers=4; done
```
Keep `workers ≤ 4` (forward passes are serialised; HANDOFF §8). Archive the run directory to the bucket, stop the instance.

### Step 8 — τ² shadow and refit (Python 3.12 env with tau2-bench, docs/TAU2.md)

```bash
export S1='{type: jev, provider: openrouter}'     # configs/tau2/*.yaml name the TypeSafe-direct `jev-1.13.0` (needs
                                                  # TYPESAFE_API_KEY); the OpenRouter key is the live one
DOMAINS=airline ARMS=jev_shadow SPLIT=shadow NUM_TRIALS=3 bash scripts/run_tau2.sh
python -m jevharness.bench.suites.tau2 report runs/tau2/airline --split shadow --no-report     # -> runs/tau2/airline_shadow/
for dp in tool_select tool_params; do
  python -m jevharness.bench.threshold_theory runs/tau2/airline_shadow --arm jev_shadow --dp $dp --calibration-arm \
    --task-value 1.0 --lam-per-hour 0 --format json --out results/calibration/tauhat_tau2_jev_shadow_${dp}.json
done
```

### Step 9 — τ² arms (all 50 airline + 114 retail tasks × 3 trials)

`scripts/run_tau2.sh` sets one τ for both decision points; the registered C7 arm uses the per-decision-point refit, so call
the two steps of its `run_arm` directly (same flags):

```bash
TAU2_ARGS=(--agent jev_agent --agent-llm gemini/gemini-3.5-flash --user-llm gemini/gemini-3.5-flash-lite --num-trials 3
  --task-split-name base --max-concurrency 4 --seed 300 --auto-resume
  --agent-llm-args '{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}'
  --user-llm-args '{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}')
tau2_arm() {  # tau2_arm <name> <tool_select tau> <tool_params tau>
  for d in airline retail; do
    RUN=(--jev-config configs/tau2/jev_ts_cascade.yaml --jev-name "$1" \
      --jev-set tool_select.tau=$2 --jev-set tool_params.tau=$3 --jev-set "s1=$S1" \
      --domain $d --save-to "$PWD/runs/tau2/raw/$d/$1" "${TAU2_ARGS[@]}")
    python -m jevharness.bench.suites.tau2 run "${RUN[@]}"
    python -m jevharness.bench.suites.tau2 rerun-errored runs/tau2/raw/$d/$1 -- "${RUN[@]}"   # A6.22 (e) 4
    python -m jevharness.bench.suites.tau2 convert runs/tau2/raw/$d/$1/results.json --arm "$1" --out runs/tau2/$d/$1.jsonl
  done; }
ARMS=llm_only bash scripts/run_tau2.sh                                 # the reference
tau2_arm jev_ts_cascade_tauhat_refit <τ̂*_ts refit> <τ̂*_tp refit>      # C7 (when the selection is a cascade)
tau2_arm jev_ts_cascade_tauhat_e2 1.0 0.99                             # secondary row: the E2-carried τ̂* (A5.4 (c))
DOMAINS=airline ARMS="jev_ts_cascade jev_ts_assist_k3 jev_guard" TAUS="0.5 0.7 0.8 0.9 0.95 0.99" bash scripts/run_tau2.sh
                                                                       # A5.4 (e) proxy check + (d) fallback rows (airline)
```
If the selection is assist or the guard, the C7 arm is `ARMS=jev_ts_assist_k3` / `ARMS=jev_guard` on both domains instead.
Errored simulations (termination `agent_error` / `user_error` / `infrastructure_error`, or a `JevTau2Agent` exception) are
re-run up to twice by `rerun-errored` (same task, trial and seed; A6.22 (e) 4), which records the attempts in
`jev_attempts.json`; `convert` stamps them (`meta.tau2.attempts`) and `primaries.py` flags an errored simulation with
attempts left as *retries outstanding*; a task with a trial still errored after three executions leaves the pair.

### Step 10 — analysis

```bash
python scripts/primaries.py --runs runs --c7-arm jev_ts_cascade_tauhat_refit --c7-selection results/primaries/c7_selection.json \
  --tau2-secondary jev_ts_cascade_tauhat_e2 --tau2-secondary jev_ts_assist_k3 --out results/primaries --strict
# A6.17 (b): add each NI pair's retained n' (primaries.json -> primaries.<label>.<jev|intern>.n_pairs) to SETTINGS in
# scripts/ni_exact_size.py, re-run it (results/reviews/...), report a residual above alpha, never act on it
python scripts/ni_exact_size.py
python -m jevharness.bench.scorecard runs/e2a_jevgym_tool_decisions --baseline llm_only --criterion noninferior --margin 0.03
python -m jevharness.bench.scorecard runs/e2c_jevgym_baselines --baseline llm_only --criterion noninferior --margin 0.03 \
  --pair jev_reflect_trigger_s2lite:ratem_reflect_trigger_s2lite --pair jev_ts_assist_k3:bm25_ts_assist_k3 \
  --pair jev_route_tau0.5:ratem_route --pair jev_memw:ratem_memw --pair jev_stop_guard_s2lite:ratem_stop_guard_s2lite \
  --measure-arm meta_shadow_s2lite --measure-arm meta_shadow --measure-run runs/e2b_jevgym_meta_control --task-value 0.1 1 10
python -m jevharness.bench.meta_gold h2 runs/e2c_jevgym_baselines --measure meta_shadow_s2lite --measure meta_shadow \
  --measure-run runs/e2b_jevgym_meta_control --pair jev_reflect_trigger_s2lite:ratem_reflect_trigger_s2lite \
  --pair jev_route_tau0.5:ratem_route --pair jev_memw:ratem_memw --pair jev_stop_guard_s2lite:ratem_stop_guard_s2lite
python -m jevharness.bench.paper_numbers runs --json runs/paper_numbers_full.json --check paper   # every key (~10^5)
python scripts/filter_res_keys.py runs/paper_numbers_full.json --paper paper --out paper/generated/results.tex
cp results/primaries/primaries.tex paper/generated/primaries.tex
```
The full key set overflows TeX's pool when `\input`; `scripts/filter_res_keys.py` keeps the keys the paper can use (its
literal `\res` / `\resm` / `\resdec` keys after macro expansion, the literal prefixes of templated keys, and the
templated suffixes on the Stage-1 main runs; `tests/test_filter_res_keys.py`).
`primaries.py` writes `primaries.{json,md,tex}`: the 11 decisive p-values and their rules, Holm (rank, α_adj, adjusted p,
decision), the registered labels (NI / inconclusive at this n / not NI with the loss size; cross-backend replication /
split / Jev-specific confirmation; insufficient observed opportunities; "transfer NI was not established"), the MDE at the
Holm level, the drop counts and n′ per arm, the full-set sensitivity row per primary and for the family, the C4
opportunity count, C5's realised firing rates and ceiling, C7's majority-pass and pass¹ rows, C8's cost / latency ratios,
and the provenance (path, sha256, record count of every file read). `\input{results/primaries/primaries.tex}` defines
`\res{primaries/<label>/...}`. τ² paper numbers come from the `tau2/airline_test` + `tau2/retail` directories (the 15
shadow tasks never enter a comparison), never from `tau2/airline`.

### Step 10b — secondary analyses (analysis of the closed Stage-1 logs only; no episode, no API call)

```bash
python scripts/h1_threshold_theory.py          # H1: the 72 registered threshold_theory cells (A6.22 (c)) -> results/secondary/h1.{json,md}, h1_cells/
python scripts/h3_delegation_classes.py        # H3: class credit at 5 %, NI / front, class ranking, tau-b (A6.22 (c)) -> h3.{json,md}
python scripts/h4_tost.py                      # H4: matching + TOST rule (A6.22 (c)); without an R4 checkpoint "not tested at matched accuracy"
python scripts/tau2_proxy_check.py             # C7 proxy check (A5.4 (e), A6.20 (f)): Kendall tau-b E2 vs tau^2 airline -> tau2_proxy_check.{json,md}
python scripts/secondary_tex.py                # \res{secondary/...}, \res{calibration/c1/...}, \res{tau2/refit/...} -> paper/generated/secondary.tex
```
The readings each script takes where the registration leaves a choice are printed in its Markdown output.

## 3. Arms × tasks × trials per config (from the YAMLs)

| Config (run dir) | Arms | n = 500 arms | Episodes | Trials | Primaries it feeds |
|---|---|---|---|---|---|
| `jevgym_tool_select.yaml` (`e2a_jevgym_tool_decisions`) | 27 | 8 | 7,800 | 1 | C1F C1L C2F C2L, C3 (arm), C7 selection |
| `jevgym_meta_control.yaml` (`e2b_jevgym_meta_control`) | 21 | 7 | 6,300 | 1 | C4, C6F C6L; C4 count, C5 ceiling, H2 |
| `baselines_jevgym.yaml` (`e2c_jevgym_baselines`) | 65 | 4 | 14,200 | 1 | C3 (control), C5; H2 pairs |
| `jevgym_tool_select_intern.yaml` (`e2a_jevgym_tool_decisions_intern`) | 30 | 10 | 9,000 | 1 | Intern constituents of C1 C2 C6 |
| `jevgym_shadow_cal_intern.yaml` (calibration seeds) | 2 | — (150) | 300 | 1 | Intern C1 τ̂\* |
| `bfcl_when2call.yaml` (`e1_bfcl_when2call`) | 8 deciders + FC | — | 3,991 items | 1 | C8 (1,114 BFCL tool_select items) |
| τ² tier 1 (`runs/tau2/{airline,retail}`) | 3 (+ shadow) | — | 164 tasks × 3 = 492 sims per arm (+45) | 3 | C7 (149 tested tasks) |
| τ² tier 2 (airline only) | 8 | — | 50 × 3 = 150 sims per arm | 3 | A5.4 (d) fallback, (e) proxy check |

Every other arm of the E2 configs runs n = 200 (A2); the Jev calibration arms (`jevgym_shadow_cal.yaml`) already ran
(2026-10-01, ≈ $8).

## 4. GPU-dependent arms

All 30 arms of `jevgym_tool_select_intern.yaml` except its four paired Jev arms load Intern-Decision in process (4B: the 10
co-primary arms and references at 500; the 0.8B / 2B size ladder and the uncalibrated 4B guard at 200), plus the two
Intern calibration arms. One L4 (`g6.4xlarge`, $1.32/h): ≈ 0.3 h calibration + ≈ 6.7 h E2a' (workers 4) + latency
measurement ≈ 8 h ≈ $11 of GPU (plus ≈ $200 of Gemini System-2 API, §5). The Intern constituents of C1 / C2 / C6 are
co-primary (`results/gpu_round2/gate_verdict.json`): the family is incomplete until they run (`primaries.py` enters the
conjunction with p = 1 and an ERROR flag while one is missing).

## 5. Budget (estimates)

| Item | Episodes / sims | API $ (est.) | Wall-clock (est.) |
|---|---|---|---|
| E2a | 7,800 | 163 | 2.7 h (workers 8) |
| E2b | 6,300 | 179 | 2.3 h |
| E2c (both passes) | 14,200 | 331 | 6.1 h |
| E2a' Intern (System-2 API; GPU separate) | 9,000 | 192 | 6.7 h (workers 4, GPU) |
| Intern calibration arms | 300 | 8 | 0.3 h (GPU) |
| E1 | 3,991 items × 8 deciders + FC | ≈ 10 | 1–2 h |
| τ² tier 1 (llm_only, C7 arm, E2-carried τ̂\*; + 45 shadow sims) | 1,521 | ≈ 140 | ≈ 4 h (concurrency 4) |
| τ² tier 2 (airline proxy check / fallback, 8 arms) | 1,200 | ≈ 100 | ≈ 3.5 h |
| **API total** | | **≈ 1,120** | |
| + 20 % contingency (retries, measurement spend above the pilot, price drift) | | ≈ 1,350 | |
| GPU (one L4) | | ≈ 11–15 | ≈ 8–10 h |

Within A2's signed-off $1.5–2k. The primary-critical subset alone (the 11 tests' arms, the C7 candidates, C4's
measurement arm, the Intern constituents and calibration, τ² tier 1, E1) is ≈ $430. Re-runs of errored tasks are included in the
contingency (the n = 50 pilots logged 1 infrastructure error in 3,119 episodes).

## 6. How the estimates were made

- **E2**: every registered arm × its registered n (`runner.arm_task_counts`) × the per-episode spend of the same arm in the
  retired-seed n = 50 pilots (`runs/pilot_jev_n50_split/e2a_jevgym_tool_decisions_old`, `.../e2b_jevgym_meta_control_new`;
  `usage.total_with_shadow`, i.e. including shadow / counterfactual / probe calls; Gemini prompt-cache discounts as billed).
  Arms without a pilot twin use the nearest arm: the C1F arm `jev_ts_cascade_tauhat` → `jev_ts_cascade_tau0.99`; the
  always-defer C1L arm → `llm_only_s2lite`; Intern arms → their Jev twin; E2c lexical / rule / rate / s2self arms → the Jev
  arm of the same mode (s2self assist × 1.5); Flash-Lite reflection controls → the Flash-Lite Jev reflection arm × 1.2
  (fired-step counterfactuals). Wall-clock = Σ (n × pilot episode wall time) / workers, arms run one after another. The
  same method reproduces the measured Jev calibration spend (2 × 150 episodes: $8.1 estimated vs ≈ $8 billed).
- **E1**: HANDOFF §4's measured projection (all Gemini arms ≈ $8.3 for the full pool) + Jev ($0.003 per 200 items in
  `results/pilot_jev/e1_ledger.json`) + native FC.
- **τ²**: HANDOFF §4's retail measurement ($0.074–0.082 per simulation for the agent + ≈ $0.001 System-1 + ≈ $0.007 user
  simulator ≈ $0.09); 20–60 s per simulation.

## 7. Known gaps (decide before the runs that need them)

- **τ² always-defer** (closed 2026-10-02): `JevTau2Agent` honours `--jev-set tool_select.always_defer=true` (and the same
  for tool_params): the key lands in the decision point's `extra`, System-1 is not queried and every decision is logged as
  deferred (`bench/suites/tau2.py`, `tests/test_tau2_adapter.py::test_cascade_always_defer_never_queries_system1`).
- **τ² per-decision-point τ.** `scripts/run_tau2.sh` sets tool_select τ = tool_params τ; the registered C7 arm and the
  E2-carried row use per-decision-point thresholds (step 9 calls `suites.tau2 run` directly).
- **τ² retries.** tau2's `--auto-resume` re-runs only `infrastructure_error` simulations; `suites.tau2 rerun-errored`
  (called by `scripts/run_tau2.sh` and in step 9) prunes every errored simulation with attempts left and re-runs it,
  up to twice (A6.22 (e) 4).
- **"Policy GUARD" on E2** (A6.9 #11) has no E2 twin; `primaries.py` reads E2a's `jev_ts_guard` as its counterpart and says
  so in `c7_selection.json` (an interpretation).
- **τ² System-1 spec.** `configs/tau2/*.yaml` name `{type: jev, model: jev-1.13.0}` (TypeSafe direct); with only the
  OpenRouter key live, every τ² arm runs with `S1='{type: jev, provider: openrouter}'` (logged per episode; state it in
  the paper).
- **Score card defaults.** `python -m jevharness.bench.scorecard` still defaults to `--criterion not_worse` (A6.9 #10 /
  P1-9, a pre-submission item): pass `--criterion noninferior --margin 0.03` (0.05 on τ²) for any paper table;
  `paper_numbers` already does.
