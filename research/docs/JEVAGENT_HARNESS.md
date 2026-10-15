# JevAgentHarness

JevAgentHarness is deliverable 3 of `docs/SCOPE_V2.md`: an agent harness in which a fast System-1 decision model and
the System-2 LLM share control *per decision point*, shipped with a best-practice profile and with hooks into real
harnesses. It is the reference loop of this repository (`src/jevharness/harness/agent.py`, `JevAgent`) run with a
**profile** (`configs/harness/*.yaml`). The loop is unchanged: every paradigm the paper studies is a configuration,
and a profile is a configuration that covers every decision point at once.

## 1. What the harness is

At every step of an episode the harness reaches up to eight decision points (`DP` in `harness/config.py`). At each one,
the profile says who decides and how:

| decision point | what is decided | modes (who decides) |
|---|---|---|
| `tool_select` | the next tool | `off` (the LLM), `cascade` (System-1 when its confidence ≥ τ, else the LLM), `assist` (System-1 keeps the top-k tools, the LLM chooses), `guard` (the LLM proposes, System-1 can veto once), `replace` (System-1) |
| `tool_params` | bounded and copy arguments of a System-1-chosen call | `off`, `cascade`, `replace`; `assist` (System-1 shortlists each argument's values, the LLM fills the call) and `guard` (System-1 can veto an argument of the LLM's own call once), both E6b; free text is always the LLM's |
| `stop` | whether the episode ends | `guard` (System-1 vets the LLM's `finish`), `trigger` (System-1 can end it early; the LLM writes the answer), `cascade` (both, when System-1 is confident; E6b) |
| `reflection` | whether the LLM reflects before acting | `trigger` (System-1 decides), `cascade` (System-1 when confident, else the LLM answers the same question; E6b), `always` (reference) |
| `routing` | strong or cheap System-2 for this step | `trigger`, `cascade` (System-1 when confident, else the cheap LLM decides; E6b) |
| `memory_write` / `memory_read` | what is stored / recalled | `trigger`, `cascade` (else the LLM gate), `guard` (System-1 vetoes the LLM gate's writes) / `assist`, `cascade` (else every recent memory), `trigger` (read or not); the cascades, guard and read trigger are E6b |
| `action` | the action in a discrete-action environment | `cascade`, `assist`, `replace`; `guard` (the LLM proposes, System-1 can veto once, as the tool guard; E6c) |

`shadow` (measurement only: System-1 is asked and logged, the LLM decides) exists at most decision points and is used
for calibration and training data, never in a profile.
The cells of the 8 × 5 (decision point × replace / cascade / assist / guard / trigger) grid that are not listed are
undefined or equivalent to a listed one, each with its reason in `harness/config.py` (`UNDEFINED_CELLS`) and
`results/explore/e6b_modes/cells.md`; the E6b modes and the E6c action guard are exploratory and enter no registered
profile.

Guarantees that hold for every profile:

- **One ledger.** Every System-1 and System-2 call goes through the episode's `Ledger` (cost at list prices, tokens,
  latency, decision point). The episode cost is `usage.total`; measurement calls are kept out of it
  (`usage_summary`). The development check below shows the per-call records and the ledger totals agree.
- **Gold labels never reach a backend.** Oracle labels are computed after System-1 answered and only logged.
- **Execution semantics are those of the registration** (PREREGISTRATION A6.22 (a), (e)): a deferred tool choice
  leaves the whole call to one native LLM call; an accepted one has System-1 fill the bounded / copy arguments and the
  LLM fill free text and open required arguments in one forced call; a live System-1 failure is an infrastructure
  error in the reference loop and fail-open in the real-harness adapters.

## 2. Why per decision point

The registered study (Stage 1) measures each decision point and mode separately; the closed-form delegation rules of
paper §3 say where delegation pays (a cascade at τ̂\* when System-1's certainty set is right more often than the LLM,
weighted by the stakes; a gate in proportion to its resolution J). A profile composes those findings: the cheapest
non-inferior setting at each decision point, with the LLM keeping every decision point where nothing qualifies.

## 3. Profiles

A profile is a YAML file; the schema is in `harness/profile.py`:

```yaml
kind: jevagent_profile
version: 1
name: jevagent_default
status: development            # development | pending | selected
system2:
  gemini-3.5-flash:            # one section per System-2 model
    decision_points:           # exactly as an experiment arm writes them; omitted = off
      tool_select: {mode: cascade, tau: 1.0}
      tool_params: {mode: cascade, tau: 0.99}
      stop: {mode: trigger, tau: 0.9}
    harness: {}                # optional HarnessConfig scalars (guard_max_retries, max_steps, ...)
    best_single: {...}         # optional: the strongest single-mode arm (Stage 2's S2-2 reference)
    backends: {...}            # optional: per-System-1 threshold overrides (e.g. JevAgentRL's own τ̂*)
    alfworld: {...}            # optional: the rendering for action environments
    skip: "reason"             # optional: arms using this section do not run
```

Shipped profiles:

| file | status | how it was made | Flash | Flash-Lite |
|---|---|---|---|---|
| `configs/harness/jevagent_default.yaml` | development | development evidence only: the calibration fits (seeds 5000–5149), the development rounds (seeds 0–999), the n = 50 pilots (`results/pilot_jev/JEV_PILOT_NOTES.md`) and the Stage-2 development check (seeds 900–999, `results/stage2_dev/NOTES.md`) | tool choice and its arguments in System-1's certainty set (cascade τ 1.00 / 0.99); everything else with the LLM (the stop trigger never fired in the development check and only added calls) | every decision with the LLM (the tool-selection fit accepts no level; the reflection trigger's evidence is mixed) |
| `configs/harness/jevagent_best.yaml` | pending → selected | written by `scripts/select_profile.py` from the Stage-1 main-run results by the registered rule (`docs/PREREG_A7_STAGE2.md` A7.1) | — | — |

The best profile is selected per decision-point group (tool, stop, reflection, routing, memory, action): among the
registered Jev arms that delegate there, those that are complete, non-inferior to LLM-only (the registered NI rule,
3 pt) and cheaper (paired t-test) qualify; the largest saving wins, a mode within 2 pt of it that leaves more to the
LLM wins the tie, and a group with no qualifying arm stays with the LLM. Thresholds are never tuned on evaluation data:
a cascade enters at its calibration fit τ̂\*, every other mode at its registered τ / k. Until it is selected the file
has `status: pending`, and the runner skips every arm that names it.

**Loading a profile.** In an experiment YAML an arm names the profile instead of setting decision points:

```yaml
arms:
  - {name: llm_only}
  - {name: jevagent, profile: configs/harness/jevagent_default.yaml}                 # section = the arm's LLM model
  - {name: jevagent_lite, llm: lite2, profile: configs/harness/jevagent_default.yaml}
  - {name: single, profile: configs/harness/jevagent_best.yaml, profile_part: best_single}
  - {name: rl, s1: jevagent_rl, profile: configs/harness/jevagent_best.yaml, profile_backend: jevagent_rl}
  - {name: alfworld, profile: configs/harness/jevagent_best.yaml, profile_env: alfworld}
```

`profile_s2` picks a section explicitly (default: the arm's System-2 model id). An arm that names a profile may not
set a decision point itself. The runner (`bench/runner.py`) writes the resolved profile of every such arm to
`<out>/profiles.json` and stamps each episode with `meta.profile` (path, sha256, status, section), and
`meta.config` holds the resolved decision points. On the command line:

```bash
jevbench profile configs/harness/jevagent_default.yaml --tau2 --harnesses     # how it resolves, per section and harness
jevbench run configs/dev/stage2_dev.yaml --out runs/stage2_dev -o workers=4   # a profile arm next to LLM-only
jevbench run configs/experiments/jevgym_tool_select.yaml --out runs/dev -o suite.seed0=0 \
  -o 'arms=[{name: llm_only}, {name: jevagent, profile: configs/harness/jevagent_default.yaml}]'   # any config
```

## 4. Plugging into real harnesses

The same profile renders onto every harness this repository integrates, through each harness's public extension
surface (no forks; `docs/HARNESS_INTEGRATION.md`). What a harness can deploy is limited by its affordances, and the
renderings say what they drop (`jevbench profile … --tau2 --harnesses`; `harness/profile.py`):

| harness | how the profile is applied | deployable | dropped |
|---|---|---|---|
| reference loop (JevGym; `bench/runner.py`) | profile arms (§3) | every decision point and mode | — |
| τ²-bench (`bench/suites/tau2.py`, `JevTau2Agent`) | `tau2_rendering` → a flat τ² config (`configs/tau2/stage2/jevagent_best.yaml` for the selected profile; run with `scripts/run_stage2_tau2.sh` or `python -m jevharness.bench.suites.tau2 run --jev-config …`) | tool_select cascade / assist / guard / replace, tool_params, stop guard (on `transfer_to_human_agents`), routing | stop trigger (the user simulator ends conversations), reflection, memory |
| ALFWorld (`bench/suites/alfworld.py`) | `profile_env: alfworld` (the action, reflection, routing and memory decision points) | action cascade / assist / replace, reflection, routing, memory | tool / stop decision points (no tool calls, no finish proposal) |
| pi on Terminal-Bench via Harbor (`adapters/pi/`) | `pi_rendering` → the extensions to load (`pi -e adapters/pi/<ext>.ts`, or `harbor_jev_pi.py`) and their `JEV_*` environment | tool guard (`jev-tool-gate.ts`), tool assist (`jev-active-tools.ts`), stop guard / trigger (`jev-stop-verify.ts`), routing (`jev-router.ts`), reflection (`jev-reflect.ts`), memory (`jev-memory.ts`) | tool cascade / replace and tool_params: pi cannot answer a model request without the LLM |
| mini-SWE-agent on SWE-bench (`adapters/mini_swe/`, `adapters/mini_swe.py`) | `mini_swe_rendering` → a `jev.arms.<arm>` block of `adapters/mini_swe/swebench_jev.yaml` | command guard (= tool_select guard), stop guard / trigger, routing, reflection | tool cascade / assist / replace and tool_params (one bash command per step), memory |
| Codex, DeepSeek Harness (`adapters/{codex,deepseek}/`, `adapters/hooks.py`) | the hook script's `JEV_*` settings, mapped by hand from the same table | guard (deny-only on Codex PreToolUse), stop, reflection / memory via additional context | cascade / replace, assist |

The consequence is a finding in itself: the profile that wins in the reference loop (a tool cascade on a strong
System-2) is not deployable on a harness that cannot answer a model request without the LLM; on those harnesses the
deployable part of the profile is its gates. Every adapter logs decisions in one JSONL schema and fails open.

## 5. Serving System-1

Every harness speaks the Jev wire format (`POST /v1/systemone`, `decision/jev.py`): hosted Jev, OpenJev, our trained
model and any other backend are a base-URL change.

- **Hosted Jev**: `{type: jev, provider: openrouter}` (the pinned `typesafe/jev-1.13`; priced from OpenRouter's
  reported `usage.cost`).
- **JevAgentRL** (deliverable 4; the checkpoint frozen under `docs/TRAINING_RECIPES.md` §18.4, recorded in
  `results/gpu_round5/`): `python -m jevharness.training.serve_jev --model <base> --adapter <checkpoint> --name
  jevagent-rl --port 8081`, then `{type: jev, name: jevagent-rl, base_url: http://<host>:8081, model: jevagent-rl,
  api_key: ""}`. Its price is the amortised GPU cost measured with `scripts/measure_latency.py` (A7.3 (e)).
- **Session cache** (`serve_jev`, `docs/TRAINING_RECIPES.md` §16): consecutive states of one episode share their
  prefix up to `CURRENT OBSERVATION`, so the server keeps an LRU of KV snapshots and prefills only the new suffix
  (exact: `option_logits` equal the stateless path's); `--session-cache`, `--cache-tokens`, `--cache-entries`.
- **Multi-question requests**: the questions of one request share one state; the server prefills the common prefix
  once and scores every question's suffix in one batch (`--batch-suffixes`). The reference loop asks one question per
  request where the registration fixes the call graph (A6.22 (a)); mini-SWE-agent batches all questions of a step
  (`batch_s1: true`) and the E1 static evaluation batches by state (`batch_by_state`).
- **Any other backend** behind the same wire: `python -m jevharness.decision.server --port 8090 --s1 '<spec>'`
  (`decision/server.py`), e.g. for the pi extensions inside Harbor containers.
- **Decision cache**: `s1_cache` (SQLite, keyed on the backend description, state and questions) replays an answer
  with its original cost and latency (`cached` on the call record); Stage 2 uses fresh cache files so no Stage-1 or
  development answer is replayed.

## 6. Development check (development seeds 900–999; not a paper number)

`configs/dev/stage2_dev.yaml`: LLM-only vs JevAgentHarness[`jevagent_default`, Jev] with Flash and with Flash-Lite as
System-2, plus the default profile's tool group alone, 100 development tasks each (the default as it was then: cascade
+ stop trigger on Flash, reflection trigger on Flash-Lite). Results and the reading are in `results/stage2_dev/`
(marked development): every episode ran without an error and its call records sum to the ledger; on Flash the
composed profile and the cascade alone kept 100 / 100 at −21 % cost but were 3–4 s slower per episode (≈ 11 Jev
requests at ≈ 0.2 s each); the stop trigger never fired; on Flash-Lite the reflection trigger read 83 vs 88. The
default profile was then reduced to the cascade (Flash) and to LLM-only (Flash-Lite). The check selects nothing and
enters no test.

## 7. Stage 2

The confirmation of the selected profile on fresh data (JevGym seeds 300000–300499, τ² telecom, ALFWorld valid_seen)
and of JevAgentRL is registered in `docs/PREREG_A7_STAGE2.md` (A7, amended by A7.9–A7.19); run order in A7.6.

| run (`runs/<name>/`) | config | arms | serves |
|---|---|---|---|
| `stage2_jevgym` | `configs/experiments/stage2_jevgym.yaml` | `llm_only`, `jevagent_best`, `best_single`, `jevagent_best_rl`, `jevagent_default`, `jevagent_best_sl4b` (StartLux-4B), `jevagent_minus_<group>` (n = 200), `llm_only_s2lite`, `jevagent_best_s2lite` | S2-1, S2-2, S2-4, S2-1T, S2-1L, S2-4S, LOGO rows |
| `stage2_rl_states` | `stage2_rl_states.yaml` | `jev_ts_replace_f`, `intern_ts_replace_f`, `rl_ts_replace_f` (n = 200) | S2-3's corpus sources |
| `stage2_x` | `stage2_x.yaml` (+ `jevgym_shadow_cal_x.yaml`, seeds 5150–5299) | `llm_only_x`, `jevagent_best_x`, `llm_only_xsmall` | S2-1X (DeepSeek-V4-Pro / V4-Flash, pinned) |
| `stage2_best_control` | `stage2_best_control.yaml` (written by `select_profile.py --control-out`) | `best_control` (none when `small_only` is selected: then `stage2_jevgym/llm_only_s2lite`) | A7.14 |
| `stage2_rl_attribution` | `stage2_rl_attribution.yaml` | `rl_base_ts_replace_f`, `rl_sft_v1_ts_replace_f`, `rl_sft_v1_opd_ts_replace_f` (n = 100) | A7.13 rows |
| `stage2_tau2/telecom` | `scripts/run_stage2_tau2.sh` (one attempt per task) | `llm_only`, `jevagent_best` | S2-L3a |
| `stage2_alfworld` | `stage2_alfworld.yaml` | `llm_react`, `jevagent_best` | S2-L3b |
| `e1_bfcl_when2call_rl` | `configs/decision/bfcl_when2call_rl.yaml` | `jevagent_rl` (one `s1_<backend>.jsonl` per evaluated row) | A7.3 (d), A7.13 |
| `stage2_startlux_states` | `stage2_startlux_states.yaml` | `sl4b_ts_replace_f`, `sl27b_ts_replace_f` (n = 200) | A7.18 (e) 3 |
| `e2a_jevgym_shadow_cal_startlux` | `jevgym_shadow_cal_startlux.yaml` (seeds 5000–5149) | `sl4b_shadow_cal` | A7.18 (e) 4 → `scripts/fill_backend_tau.py --backend startlux_4b --fit-arm sl4b_shadow_cal` |
| `e1_bfcl_when2call_startlux` | `configs/decision/bfcl_when2call_startlux.yaml` | `startlux_4b`, `startlux_27b`, `intern_4b` | A7.18 (e) 1, (h) |

**S2-3's corpus** (`scripts/stage2_corpus.py`, library `bench/stage2_corpus.py`): `build` writes
`results/stage2/s23/{occurrences,questions}.jsonl.gz` and `corpus.json` (eligible / excluded tasks, file hashes);
`replay --name {jev,intern,rl}` (and the A7.13 rows `base`, `sft_v1`, `sft_v1_opd`; the A7.18 rows `sl4b`, `sl27b`,
`sl0.8b`, `sl2b`, `sl9b`; A7.19's `v2`) writes `answers_<name>.jsonl` and `replay_<name>.json`.

**StartLux-Decision** (A7.18): `bash scripts/aws/serve_startlux.sh` on the GPU host installs the vendor package at the
pinned commit and the fast linear-attention kernels in their own venv, plans one size per GPU
(`scripts/startlux_plan.py`: original weights when the GPU is large enough, the 27B's Q8_0 GGUF on ≥ 40 GiB, else
not run), checks the kernels, starts the servers (4B :8090, 27B :8091; the Jev wire format, the harness's `{type:
jev, base_url: ...}` backend), smoke-tests them and writes `results/stage2/startlux_manifest.json` (weights
revisions, package commit, kernel versions, status per size). Commit the manifest before the first evaluation request.
S2-4S's tariff is `results/stage2/tariff_sl4b.json` (`scripts/stage2_tariff.py --arm jevagent_best_sl4b
--served-model StartLux-Decision-4B --out results/stage2/tariff_sl4b.json`).

**Files the analysis reads besides the runs.** `results/gpu_round5/FREEZE_RECORD.json` (JevAgentRL's freeze record,
A7.10 (b), (h); at least `{"frozen_at": "<ISO time>"}`, and `"evaluation_exposure_before_freeze": true` makes S2-3 /
S2-4 exploratory); `results/stage2/tariff.json` (A7.13 (iii): `{"q_usd_per_episode": q, "frozen_at": ..., "served_model":
..., "workload_matches": true, "sensitivity": {"busy_throughput": q_b, ...}}`, written once by `scripts/stage2_tariff.py`
from the committed serving window; the served model's per-token price must be an exact 0 entry so nothing is charged
twice); `<stage2_jevgym>/schedule.json` (A7.16 (a): `{"mode": "task_major", "seed": 732,
"concurrency": c, "arms": [...]}`; without it S2-1T enters with p = 1); `results/stage2/control_selection.json`;
`configs/tau2/stage2/telecom_task_ids.json`; `configs/harness/jevagent_best{,_x}.yaml`.

**Outputs.** `python scripts/stage2_analysis.py --runs runs --out results/stage2 --strict` writes
`results/stage2/stage2.{json,md,tex}` and `paper/generated/stage2.tex`:

- `stage2/<test>/{delta, delta_ci/lo|hi, cost_savings, cost_savings_ci/lo|hi, p_decisive, p_holm, alpha_adj,
  holm_rank, decision, n_pairs, n_expected, n_dropped}` for S2-1..S2-4, S2-1T, S2-1L, S2-L3a, S2-L3b, S2-1X, S2-4S.
  `delta` is
  the success difference arm − reference in points; for S2-3 it is JevAgentRL − Intern-Decision accuracy in points
  (`delta_jev` the Jev clause), for S2-1T the latency saved per episode in seconds (`latency_savings` in %).
- `stage2/<test>/<clause>/{est, est_ci/lo|hi, boundary, p, decision}` with clauses `ni` (points; boundary −3 or −5;
  `est_ci` the two-sided 95 % bootstrap interval as in Stage 1, `est_ci_newcombe` beside it), `cost` ($ saved per
  episode, boundary 0), `latency` (s saved per episode), `intern` / `jev` (accuracy points; `jev`'s boundary −5);
  `decision` is the clause's reading.
- `stage2_rl/<row>/{corpus_acc, corpus_acc_offplan, ece, live_success, own_acc, l1_acc}` for rows `jev`, `intern`,
  `base`, `sft_v1`, `sft_v1_opd`, `frozen` (A7.13; live rows on the first 100 Stage-2 tasks; `l1_acc` = E1 BFCL
  tool_select accuracy).
- `stage2_s1/<row>/{l1_acc, corpus_acc, corpus_acc_offplan, ece, live_success, own_acc, status,
  vs_jev/{e1_delta, e1_delta_ci, live_delta, live_delta_ci, corpus_delta, corpus_delta_ci}}` for rows `jev`, `intern`,
  `frozen`, `sl4b`, `sl27b` and `stage2_s1/kendall/{e1_vs_corpus, e1_vs_live}` (A7.18 (h), static vs in the loop; live
  rows on the first 200 tasks; differences to Jev with Newcombe intervals for E1 and live success, a task bootstrap for
  C).
- `stage2/scorecard/<arm>/{success, success_ci, cost_usd, latency_s, llm_calls, s1_calls, n, vs_baseline/{delta,
  delta_ci, cost_savings, cost_savings_ci}}` for `llm_only`, `jevagent_best`, `best_single`, `jevagent_best_rl`,
  `jevagent_default`, `jevagent_best_sl4b`, `llm_only_s2lite`, `jevagent_best_s2lite`, `best_control`, `llm_only_x`,
  `jevagent_best_x`,
  `llm_only_xsmall` (the reference is the LLM-only arm of the same System-2).
- `stage2/descriptive/{logo-<group>, best_control-llm_only, best_control-jevagent_best, llm_only_xsmall,
  jevagent_default}/...` (Newcombe and Tango intervals, savings, latency).

**Effective specification and launch gate** (A7.17 (c)): `python scripts/stage2_spec.py` renders
`docs/STAGE2_EFFECTIVE.md` (tests, runs, populations, failure rules, resource gates, prerequisites; a test keeps it in
sync); `python scripts/stage2_spec.py --gate <run> [--rl] [--startlux]` checks the spec, the Stage-2 fixture tests and the run's
prerequisites and exits 1 unless all pass. Telecom population: `scripts/telecom_stage2.py inventory | sample` (A7.15).
The X cost-gate record is `results/stage2/x_cost_gate.json` (`{"n": 500 | 300, "predicted_usd": ...}`), written by
`python scripts/x_eligibility.py --run runs/stage2_dev/x_eligibility --write` after the development smoke
`configs/dev/x_eligibility.yaml` (A7.12 (b), (c): malformed share, pin check from `served_provider`, errors; no success
number is read).

**JevGym statistics** (`scripts/jevgym_data_stats.py`, task generation only): `results/data_stats/jevgym_eval.json` and
`paper/generated/data_stats.tex` (`data_stats/jevgym_eval/{plan_steps_median, n_distinct_plans, ...}` on the Stage-2
task list).

## 8. The final harness (exploratory; `docs/PREREG_A7_STAGE2.md` Deviations 2026-10-05 "harness_final")

Stage 2 found the cheapest profile slower per episode than LLM-only: with JevAgentRL-4B the agent made 3.03 instead of
7.52 System-2 calls per episode, yet the episode took 10.2 s instead of 6.8 s. The diagnosis
(`python scripts/harness_final.py diagnose` → `results/harness_final/latency_diagnosis.md`) found the time in the
System-1 path: 24.6 one-question requests per episode (a reflection question that never fired, the tool choice, and one
request per argument question), served one at a time by one GPU server that four episodes shared (busy 97 % of the
run; two thirds of the System-1 latency was queueing), with the 69-option tool choice (a two-round knockout that
re-reads every option's description after the state at every step) the most expensive request.

The final harness changes how decisions are served, not what is decided:

| change | where | behaviour |
|---|---|---|
| reflection trigger off | `configs/harness/jevagent_final.yaml` | a post-hoc configuration choice (it never fired with JevAgentRL; with Jev it lost success on fresh seeds) |
| the argument questions of one call in ONE System-1 request | `tool_params.extra.batch_s1` (`harness/agent.py::_fill_params`) | every answer, threshold test, executed call and log equals the per-question path's (`tests/test_harness_final.py`); off by default (the registered call graph asks one question per request, A6.22 (a)) |
| fused multi-question pass | `serve_jev --fuse-multi` | the prefix delta and the longest question's suffix in one forward pass, the cache cropped back to the prefix (as the single-question fused path) |
| length-grouped suffix batches | `serve_jev --suffix-pad-slack 0.25` | a 52-option knockout chunk and a 17-option one are no longer padded to the same length |
| LM head at the read-out positions only | `serve_jev --readout-gather` | `logits_to_keep` as an index tensor instead of the whole window |
| serving counters on every System-1 record | `decision/jev.py`, `Usage.server` → `CallRecord.meta["server"]` | prefilled / prefix-cache-hit tokens, server time, time waiting for the engine vs computing |
| no System-1 decision cache | the live config's backend (`cache: null`) | the SQLite cache was a research convenience (one read, insert and commit per request) |

`serve_jev --fast` turns on the three serving paths; the option logits equal the Stage-2 server's and the stateless
path's within fp32 tolerance on a tiny model (`tests/test_harness_final.py`, `tests/test_serving.py`), and the replay
benchmark (`scripts/aws/harness_final_gpu.sh bench` → `results/harness_final/replay_bench.md`) compares every replayed
answer of the frozen JevAgentRL-4B with the logged one and across server versions. No independent System-1 → System-2
wait remains in the final call graph: the argument questions depend on the tool choice and the forced System-2 call on
the argument answers. Cross-episode batching of different states is not implemented: the dominant pass (the knockout
chunk, ~840 suffix tokens) is compute-bound, so batching it with another episode's would not raise throughput.

Run: `scripts/aws/harness_final_gpu.sh` (stages `rearm bench serve_fast tariff smoke live final`); analysis:
`python scripts/harness_final.py analyze` → `results/harness_final/summary.{json,md}`, keys `\res{hf/...}` in
`paper/generated/harness_final.tex`.
