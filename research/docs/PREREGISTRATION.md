# Pre-registered analysis plan (frozen before the full-scale runs)

Status: **draft for freeze on 2026-10-03** (before any full-scale Jev run). Frozen 2026-10-03 at commit `FREEZE_COMMIT`
(tag `prereg-freeze`; the placeholder is replaced by the hash at the tag, HANDOFF §7.1). Structure: §1–5, the
pre-freeze amendments A1–A3, the "Deviations" section, then amendments A4, A5 and A6 (A6.1–A6.22), each appended after
the previous one in the order written and never edited once the next exists (on 2026-10-02 the A6.17–A6.20 blocks,
first inserted above the Deviations section, were moved unchanged to follow A6.16). "Deviations" holds dated factual
entries only: before the freeze, data-derived settings written into configs (fitted thresholds, control rates,
population counts, validation results); after the freeze, every departure from the registered plan with a date and a
reason. The paper's §6 cites this file and reports every deviation.
Derived from `docs/REVIEW_SIMULATION.md` §7 (P0-5, P0-7) and `docs/BENCHMARK_SPEC.md` (metric definitions).

## 1. Units, samples, exclusions

| Suite | Unit | Size | Trials | Why this size |
|---|---|---|---|---|
| JevGym v2 (E2a/E2b), main variant | task (seeded) | **500** | 1 | 3-pt non-inferiority at ≤5% discordance needs ≈340 tasks; MDE ≈3.3 pt at 7% discordance |
| τ²-bench airline + retail (E5) | task | all (50 + 114) | 3 (pass^k) | 5-pt margin needs ≈125–250 tasks at 5–10% discordance |
| ALFWorld (E6) | game | 134 (valid-unseen) | 1 | whole split |
| BFCL v4 (E1) | item | ≤200 per category, 6 categories | 1 | whole categories where smaller |
| When2Call (E1) | item | 1,000 (camera-ready: 3,652) | 1 | — |

- **Resampling unit.** Every bootstrap resamples tasks. Trials of one task stay together.
- **Pairing.** All arms of an experiment run on identical task lists and seeds. Comparisons are paired.
- **Infrastructure errors.** Rate limits, 5xx and provider exceptions are retried on resume, up to 2 extra attempts.
  A task still errored in any arm after that is dropped from *every* pairwise comparison that involves that arm. The
  number of dropped tasks is reported per arm. A task failure caused by the agent (wrong action, step limit) is a
  failure, never an exclusion.
- **Frozen before runs.** Prices (`configs/prices/models.yaml`, verified on the run date), model ids (Jev pinned to
  `typesafe/jev-1.13`), prompts and harness defaults are fixed before the main runs.
- **No stopping rule.** Suites run to the sizes above. There are no interim looks at primary comparisons.
- **System-1 families.**
  - Every primary comparison uses a **prefill-only** decider: Jev, or else OpenJev, or else Qwen3-4B logprob (P0-1).
  - Generative deciders (Gemini Flash-Lite verbalised) are reported as labelled reference rows, never as the primary
    System-1.
  - Simulated backends never appear in results tables.

## 2. Primary comparisons (family-wise α = 0.05, Holm across the 8)

Here "NI" means non-inferiority. The test is a paired bootstrap (B = 10,000) giving a one-sided 95% lower bound on
Δsuccess = arm − reference, and NI holds if that lower bound is above −margin. Margins are **3 points on JevGym and
ALFWorld, 5 points on τ²** (pass^1). "Sup." means a one-sided superiority test at the same α.

| # | Experiment | Arm | Reference | Test | Secondary readout |
|---|---|---|---|---|---|
| C1 | E2a | Jev tool_select + tool_params cascade at τ̂* (plug-in, H1) | LLM-only | NI | cost savings with CI, cost-of-pass |
| C2 | E2a | Jev tool_select assist k=3 | LLM-only | NI | input-token savings |
| C3 | E2a | Jev tool_select assist k=3 | BM25 assist k=3 (same k) | Sup. | also vs embedding assist |
| C4 | E2b | Jev routing trigger | small-only | Sup. | NI vs LLM-only, cost savings |
| C5 | E2b | Jev reflection trigger | matched-rate random gate | Sup. (ΔU) | VoI with CI (H2) |
| C6 | E2b | Jev stop guard | LLM-only | NI | guard FAR/FRR on ≥50 wrong proposals |
| C7 | E5 τ² | best System-1 configuration selected **on E2** (fixed before τ² runs) | LLM-only | NI (5 pt) | pass^k curve, cost |
| C8 | E1 | Jev, typed question | System-2 answering the same typed question (structured output) | NI on accuracy (3 pt) | cost/1k and p50 latency ratio |

The C7 configuration is chosen by the rule "most cost savings among E2 configurations that are NI on E2", before any τ²
run. Every other arm-vs-reference comparison in the paper is **secondary**: Holm-adjusted within its table, and worded as
exploratory.

## 3. Hypotheses and how each is tested (from REVIEW_SIMULATION §8.5)

**H1: plug-in thresholds transfer (Cascade).**
- *Data.* Shadow arms (`mode: shadow`: System-1 is queried and logged, but the LLM decides) on a 30% task split.
- *Fit.* Estimate g_d (calibration map), q_L(c) (LLM accuracy as a function of System-1 confidence), ΔC and V.
- *Evaluate on the 70% split.*
  1. Compute regret of τ̂* against the best swept τ ∈ {0.5, 0.7, 0.8, 0.9, 0.95, 0.99}.
  2. Compute regret of the fixed default τ = 0.8 against the same best swept τ.
  3. Do this for task values W ∈ {$0.1, $1, $10}, with utility U = W·success − cost − λ·latency and λ ∈ {0, $0.5/h, $5/h}.
- *Tool.* `python -m jevharness.bench.threshold_theory`.
- *Falsified if* τ̂* does not beat τ = 0.8 in a majority of (decision point, backend, W) cells, or temperature scaling
  (E3) does not move over-confident backends back onto the front.

**H2: resolution, not accuracy, makes gates pay (Guard/Trigger).**
- *Cells.* About 40: 5 gates (reflection, stop guard, stop trigger, routing, memory write) × backends × System-2 families.
- *Test.* Regress ΔU(gate − matched random gate) on the gate's AUROC and accuracy, then run a likelihood-ratio test for
  adding accuracy. Run a TOST equivalence test (±2 pt) of the random gate against the System-1 gate.
- *Falsified if* the AUROC slope is ≤ 0, accuracy explains ΔU better, or System-1 gates are equivalent to matched random
  gates.

**H3: delegate bounded, recoverable, meta-level decisions.**
- *Method.* Classify every configuration by the class of decision it delegates:
  - meta;
  - bounded-recoverable object-level;
  - copy-dependent arguments;
  - irreversible actions.
- *Report.* The front of NI configurations per suite and per System-2 family. Kendall τ of the class ranking (median
  savings) across families and between L2 and L3.
- *Falsified if* a copy-dependent or irreversible delegation is NI **and** on the front, or the ranking reverses
  (Kendall τ < 0).

**H4 (secondary, training).** Counterfactual Monte-Carlo targets (R4) improve downstream success over teacher distillation
(R3) at matched decision accuracy.
- *Falsified if* the downstream difference is inside the ±3-pt band.

## 4. Reported for every arm (no selective reporting)

- Episodes:
  - success with a 95% task-bootstrap CI, pass^k (τ²);
  - cost per episode (list price, and OpenRouter-billed where available), cost-of-pass;
  - p50/p95 latency on the critical path;
  - LLM calls and input tokens, System-1 calls;
  - dropped (errored) tasks.
- Decisions:
  - System-1 accuracy, ECE (10 equal-width bins, the code default), adaptive ECE (10 equal-mass bins), Brier, AUROC, AURC;
  - delegation ratio, harmful-delegation rate, guard FAR/FRR.
- Run-level:
  - iso-quality cost savings;
  - Pareto hypervolume (log cost);
  - APGR / CPT(50%) / CPT(80%) per τ sweep, and AIQ against the zero router.
- No row may show $0 cost without a justification in its caption (unpriced models are flagged in `unpriced_models`).
- Prompt caching: report the main E2 table with and without provider caching (appendix).

## 5. What would change the plan (declared now)

- **Jev unreachable by 2026-10-03.** OpenJev (or Qwen3-4B logprob) becomes the primary System-1. Every claim is worded
  about "prefill-only decision models". Jev rows move to an appendix or are marked pending.
- **LLM-only above 90% on the JevGym v2 pilot (50 tasks).** Harden the generator (near distractors, longer horizons,
  noise) until LLM-only is 60–85%, *before* the main runs. The chosen generator settings are frozen and reported.
- **Only one System-2 family reachable.** H3's cross-family clause is reported as untested. It is not dropped silently.

## Pre-freeze amendments (proposed 2026-09-30; accept or reject before the 10/3 freeze)

- **A1 — JevGym suite.** Use the P4 calibration (`docs/PILOT_RESULTS.md` §P4): families v2, near distractors,
  60 distractors, noise 20, strict.
  - With Flash, LLM-only is at ceiling (100%). No affordable setting brings Flash into 60–85%.
  - So System-2 strength becomes a factor:
    - **Flash** is primary. C1, C2 and C6 stay non-inferiority claims.
    - **Flash-Lite** is secondary, with `*_s2lite` arms (LLM-only 77%). It shows where System-1 help can add success.
    - C5 (reflection vs matched random gate) is tested with **Flash-Lite**, because with Flash there is nothing
      left to fix.
- **A2 — Sample sizes vs budget.** On this suite a Flash LLM-only episode costs about $0.07 before the prompt-cache
  discount. At n = 500 the E2a + E2b + E2c grids cost roughly $1.5–2k.
  - Proposal: run **n = 500 only for the arms of C1–C6 and their references**, and n = 200 for every other arm.
    Those extra arms are reported as secondary, with their MDE.
- **A3 — Second System-2 family.** Gemma 4 through the Gemini API is quota-limited (16k TPM).
  - P0-4 therefore needs OpenRouter, a DeepSeek / Qwen model, or a self-hosted open model on vLLM.
  - Until then, H3's cross-family clause is reported as untested (§5).

## Deviations

- **2026-10-01** (A6.5 / A6.11): τ̂\* fitted from the calibration arms `jev_shadow_cal` / `jev_shadow_cal_s2lite`
  (seeds 5000–5149, run 2026-10-01 20:54–21:05 UTC, ≈ $8): Flash tool_select 1.00, tool_params 0.99; Flash-Lite
  tool_select never → 0.99 (A6.11 (b)), tool_params 0.13; V by the A6.11 (a) fallback (V = W) on all four, because
  the Flash log holds no wrong System-2 decision and the Flash-Lite counterfactual V is not positive. Values written
  into the two C1 arms of `jevgym_tool_select.yaml` (`results/calibration/*.json`).
- **2026-10-01** (A6.4 exposure): A6.4 states that no seed ≥ 100000 was run before 2026-10-03. That became untrue
  the same evening: the n = 50 pilot's E2b run (`runs/pilot_jev_n50`, started before A6.4 on seeds 0–49) was
  interrupted by a 2-hour job limit and resumed after the configs had moved to `seed0: 100000`, so its 21 E2b arms
  (`jevgym_meta_control.yaml`: references, gates, measurement arms, routing sweep; Flash and Flash-Lite; Jev) ran once on
  **evaluation seeds 100000–100049** (50 of the 500 tasks, 1,050 episodes; `runs/pilot_jev_n50_split/
  e2b_jevgym_meta_control_new`, results in `results/pilot_jev/JEV_PILOT_NOTES.md`). No E2a / E2c arm touched them.
  Consequences: (1) no prompt, label, arm, size or test was changed on this evidence (the only pre-freeze changes after
  it are A6.11 and the fitted thresholds, both from the calibration seeds); (2) the C6F / C6L arms and their references
  have therefore been observed once on 10 % of their evaluation tasks before the freeze; the registered tests still run
  on all 500 tasks, and the paper's exposure statement (§6 / App. G) says so; (3) the exposure manifest's line "no seed
  ≥ 100000 was run" is corrected to "100000–100049: one n = 50 E2b pilot pass on 2026-10-01, see Deviations".
  **Resolution (same night, A6.13):** the evaluation suite moved to 200000–200499 with a frozen, hashed task list;
  100000–100499 is retired.
- **2026-10-01** (A6.16 (f) validation, recorded as required): the exact-size table (`results/reviews/2026-10-01d/
  ni_exact_size.md`) finds the A6.14 rule at ≤ 1.011 α for n ≥ 300 and ≤ 1.023 α at n = 149 (two cells above the 2 %
  threshold: 0.05/11 at w = 0.25 and 0.05/2 at w = 0.15). What follows from it is registered in A6.17, not here.
- **2026-10-01** (A6.16 (i)): the C5 control's firing rate is set in `baselines_jevgym.yaml` (`s1_backends.ratem_lite`, rate 0.1424 =
  47 reflection firings / 330 opportunities of `jev_reflect_trigger_s2lite_e2b` in the retired-seed n = 50 pilot,
  seeds 100000–100049), replacing the 0.076 placeholder from the development seeds; locked by `scripts/prereg_check.py`.
  The frozen task list is re-written with the suite-block hash and the generator revision (A6.16 (l)); its 500 ids and
  their sha256 are unchanged.
- **2026-10-01** (A6.11 (c).4, deployed value): the Flash-Lite C1 arm `jev_ts_cascade_tau0.9_s2lite` carries
  `tool_select: {mode: cascade, tau: 0.99, always_defer: true}` (System-1 not queried) and `tool_params` τ 0.13; the
  first entry's "never → 0.99" was never run.
- **2026-10-01** (clarification of the second entry's "(1)"): true as of its time stamp; the retired-seed pilot was later
  used as development data by A6.15 (C4's τ = 0.4) and A6.16 (i) (C5's control rate) and (k) states so.
- **2026-10-01** (A6.16 (l)): `configs/eval/e1_c8_population.json` written by `scripts/c8_population.py`: 3,991 decision
  records from 2,042 source items after excluding the 299 dev-prefix records by parent item (BFCL tool_select 1,114
  items: irrelevance 185, live_irrelevance 185, live_multiple 186, live_simple 186, multiple 186, simple_python 186;
  BFCL tool_params 1,949 slots; When2Call 928 items); C8's primary n is therefore 1,114.
- **2026-10-01** (A5.1 competence gate, descriptive interval): `results/gpu_round2/gate_verdict.json` now carries a
  stratified binomial bootstrap 95 % interval of each accuracy ratio from the per-decision-point counts (4B: see the
  file); GPU round 2 kept per-decision-point counts, not per-item records, so A5.1's paired per-item bootstrap cannot be
  computed for that run (A6.18).
- **2026-10-02** (A6.5 / A6.11 / A6.19, Intern-Decision τ̂\*): the calibration arms `intern_shadow_cal` /
  `intern_shadow_cal_s2lite` (`jevgym_shadow_cal_intern.yaml`, seeds 5000–5149, 150 tasks each; GPU round 3 on one L4,
  02:53–04:07 UTC; System-2 API $8.15 from the episode ledgers) were fitted with the registered command
  (`threshold_theory --calibration-arm --task-value 1.0 --lam-per-hour 0 --v-fallback W`; `results/calibration/
  tauhat_intern_*.json`): Flash tool_select never → always-defer, tool_params 0.6309; Flash-Lite tool_select never →
  always-defer, tool_params 0.8518 (accept region not an upper tail: one isolated accepting bin below the tail, reported;
  the A6.19 tail edge and the gain-maximising edge coincide here). V = W on all four (no wrong System-2 decision on Flash;
  counterfactual V not positive on Flash-Lite tool_select). Written into `intern_ts_cascade_tauhat` /
  `intern_ts_cascade_tau0.9_s2lite` at full precision; `calibration: pending` removed; locked by `prereg_check`.
  Timing, stated for completeness: the GPU host's own automated fit, run with non-registered flags, reached the bucket
  at 04:07:33 UTC, one minute before A6.19 was written (04:08); its files were not opened before A6.19 was committed and
  are not used.
- **2026-10-03** (execution, A6.22 (e) 4; E2a and E2b resumed after container restarts): a restart of the API
  container killed the main run twice in pass 1 — E2a (`jevgym_tool_select.yaml`, started 2026-10-02 18:51:10 UTC at
  `256e953`) at about 20:32 UTC, resumed 20:34:23 UTC at `345e9e9`; E2b (`jevgym_meta_control.yaml`, started 21:38:21
  UTC at `84f468e`) at about 23:31 UTC, resumed 23:32:52 UTC at `016a5ba` (`results/main/RESUME_E2a.txt`,
  `results/main/RESUME_E2b.txt`). No record was truncated and the episodes in flight had written nothing, so they left
  no attempt: the runner resumes by task id (`bench/runner.py`), they ran as first attempts in the resumed pass 1, and
  passes 2–3 followed as registered. Between each start and resume commit nothing under `src/`, `configs/`,
  `scripts/primaries.py` or this file changed, except that `configs/eval/jevgym_stage2_tasks.json` (the Stage-2 task
  list of A7, read only by the Stage-2 configs) was added before the E2b resume; the plan hash `9a2182aa2a975e85` is
  the same in every start and resume record. Only the wall-clock order of the episodes changed.
- **2026-10-03** (execution; E2a′, MAIN_RUN_PLAN step 7, on the GPU host): E2a′ (`jevgym_tool_select_intern.yaml`,
  three passes, workers 4) started 2026-10-02 17:11:47 UTC at `8941f54` (`results/gpu_round4_main/START_E2a_intern.txt`)
  on one NVIDIA A10G, the stand-in `scripts/aws/gpu_round4_main.sh` starts when L4 capacity is short (MAIN_RUN_PLAN §4
  planned one L4); the A5.1 latency measurement (`results/gpu_round4_main/latency_intern4b.txt`) and every
  Intern-Decision latency of E2a′ are therefore A10G figures. The A10G is slower than the L4 the script's 600-min budget
  guard was sized for, and the guard stopped the instance in pass 1 at about 03:11 UTC on 2026-10-03. The run data were
  intact on the volume (no final line cut by the shutdown, nothing dropped). The run was resumed at 04:19:15 UTC with
  `scripts/aws/gpu_round4_resume.sh` on the unchanged checkout (`8941f54`, no bundle pulled) under a 720-min guard
  (`results/main/RESUME_E2a_intern.txt`), by task id: the episodes in flight at the stop had written nothing and ran as
  first attempts; pass 1 ended 05:48:51 UTC and passes 2 and 3 found no errored task to re-run (05:49:18, 05:49:46). All
  30 arms hold their registered number of records (9,000 in total) with 0 errored records
  (`results/gpu_round4_main/e2a_intern_errors.txt`).
- **2026-10-03** (A6.22 (d) 2 and A6.22 (e), H5, exploratory; `results/hard/NOTES.md`): **Terminal-Bench-Science
  subset smaller than registered.** The selection rule of A6.22 (d) 2 ran once on the calibration pass (`runs/tbsci_cal`:
  `oracle` and `nop` on the 37-task pool, then one capped `baseline` attempt per valid task; `tbsci.py select` at
  2026-10-02 22:44 UTC). Validity dropped 9 of the 37 tasks (oracle reward 0: 4; oracle agent time over 1,800 s: 5) and the
  signal rule 13 of the remaining 28 (verifier tests passed at or below the `nop` floor), leaving 15 eligible tasks, fewer
  than N = 20, so all 15 are chosen (earth 3, engineering 2, life 7, mathematical 1, physical 2). Frozen list
  `adapters/pi/tbsci_hard.txt`: sha256 `a935a41f47096de42019b0d4d993e3f8dd7b62fc49c97b352e6900aba8d7d7f5`, 846 bytes,
  **15 tasks (N = 15 of the registered 20)**, written by `tbsci.py select` on the calibration host, re-derived locally from
  the synced calibration results (identical list) and committed and pushed in `016a5ba` before the first evaluation
  trial. The evaluation then ran 3 arms × 15 tasks × k = 3 = 135 trials with 0 infrastructure errors. **H5 execution
  deviations of that round** (numbered as in NOTES.md): (1) SWE-bench Verified: the per-arm spend guard of the launcher
  (`ARM_COST_LIMIT=55`, mini-SWE's global cost limit) ended 5 baseline and 6 stop-guard instances as errored records; they
  were re-run once under A6.22 (e) 4, from scratch, later and on a less loaded host than their paired arms' attempts
  (time is a co-outcome); none errored again; the discarded attempts cost $7.10; (2) three SWE-bench evaluation timeouts
  (swebench's 1,800 s test timeout on the loaded host) are scored unresolved as run; an idle-host re-evaluation is a
  sensitivity view only (two of the three resolve; no H5 label changes); (3) a stale script on the Terminal-Bench host
  started the first calibration signal launch on all 37 pool tasks; it was stopped after 3 min (6 trials cancelled, no
  cost logged, moved aside and never used) and re-launched on the 28 valid tasks; (4) the floor stage's self-stop was
  cancelled so that the signal stage could follow at once, and the hosts' power-off guards were re-armed; (6) the
  calibration signal stage ran 6 concurrent trials (script default 4; it uses rewards and test counts only, never times);
  (7) the Terminal-Bench-Science evaluation ran as three attempt-major passes (Harbor's own `-k 3` order, cut into passes)
  so that a spend cap could stop between passes; all three ran, so k = 3 as registered, and between passes the arms
  waited for the slowest one; (8) the Docker Hub pre-pull skips images already present and waits while the anonymous
  pull budget is below 10 (no wait was needed); (9) both hosts ran the bundle `8941f54` plus an overlay of the
  hard-task scripts identical to the committed files; (10) the controlling session's console login expired for 21 min
  (monitoring only; nothing was interrupted). **Terminal-Bench 2.0 was not run**: the round's logged spend ($413.63 of
  its $440 cap) left no budget for it, and whether it runs before submission is an author decision still pending; until
  then H5 is reported on SWE-bench Verified (k = 1, 4 arms × 100 instances) and Terminal-Bench-Science (k = 3, 15 tasks)
  only, and the paper says so.
- **2026-10-03** (execution, A5.4 (c)–(e), A6.20, A6.22 (b), (e); τ², MAIN_RUN_PLAN steps 8–9): the τ² runs ran
  06:29:53–11:44:57 UTC from the main working tree, starting at `cdd7876` (`results/main/START_tau2.txt`, plan hash
  `9a2182aa2a975e85`); the commits made in the working tree during the runs (a parallel Stage-2 session's work and
  paper edits, and this step's own records) changed nothing under `src/`, `configs/prices/`, the registered arm configs
  `configs/tau2/*.yaml` or `scripts/run_tau2.sh`. Every command is the registered one
  (agent `gemini/gemini-3.5-flash`, user simulator `gemini/gemini-3.5-flash-lite`, both at temperature 0 with minimal
  thinking; seed 300; 3 trials; `--max-concurrency 4`; tau2's `base` split; `--auto-resume`, then `rerun-errored`,
  then `convert`); the commands were run by two resumable driver scripts under `nohup setsid`; no container restart
  occurred, so no resume record exists. Departures and facts the plan leaves open:
  (1) **Python 3.11, not a separate 3.12 environment.** MAIN_RUN_PLAN step 8 names a Python 3.12 environment; the runs
  used the main virtual environment (Python 3.11.15) with tau2-bench `5bfa7e3` (`v1.0.1-47-g5bfa7e3`, the pinned commit)
  installed editable with `--ignore-requires-python` and litellm 1.82.6, the setup `docs/TAU2.md` documents and verifies
  for text mode; the installed airline task file matches the sha256 in `configs/tau2/airline_task_ids.json`.
  (2) **Retail NL-assertion judge**: tau2's own default, `gpt-4.1-2025-04-14` (the official protocol; the OpenAI key was
  live, so no `--jev-nl-judge` replacement was used), fixed across arms. Its cost is environment cost and tau2 does not
  log it (≈ $3 estimated for the 360 retail simulations with judged assertions).
  (3) **System-1**: `{type: jev, provider: openrouter}` on every acting and shadow arm, as `configs/tau2/*.yaml` name it
  (A6.20 (c)); served model `typesafe/jev-1.13-20260917`.
  (4) **Refit (step 8).** The shadow arm ran the 15 airline shadow tasks × 3 trials (45 simulations, 0 errored); the
  registered command (`threshold_theory --calibration-arm --task-value 1.0 --lam-per-hour 0` on
  `runs/tau2/airline_shadow`) gives tool_select τ̂\* = 0.28 (488 paired decisions, so the scalar-threshold projection,
  A6.22 (b) 5; all ten bins accept, so τ̂\* is the lowest observed confidence and the shadow log's accepted share is 1.00;
  V = W because the counterfactual V is not positive) and tool_params τ̂\* = 1.00 (115 paired decisions, projection;
  V = 0.537 from the paired continuations; accepted share 0.40); neither decision point always defers
  (`results/calibration/tauhat_tau2_jev_shadow_tool_select.json`, `..._tool_params.json`, committed in `aabc46b`
  before the C7 arm started at 07:18 UTC). The C7 arm (the cascade selection of `results/primaries/c7_selection.json`)
  ran at tool_select τ 0.28 / tool_params τ 1.0 into `runs/tau2/{airline,retail}/jev_ts_cascade_tauhat_refit.jsonl`;
  the E2-carried secondary row at 1.00 / 0.99 into `runs/tau2/{airline,retail}/jev_ts_cascade_tauhat_e2.jsonl`.
  (5) **Errors and attempts**: 2,721 simulations (45 shadow; reference, C7 arm and E2-carried row on 50 airline + 114
  retail tasks × 3; eight airline sweep rows × 150); none ended in `agent_error`, `user_error` or
  `infrastructure_error`, so `rerun-errored` re-ran nothing (every `jev_attempts.json` is empty) and no task left a
  pair (C7 n′ = 149 of 149). Another agent's Stage-2 runs used the same Gemini keys from 06:57 UTC.
  (6) **Spend**: $235.20 logged (tau2's `agent_cost`, which includes System-1 for the acting arms and applies Gemini's
  implicit-cache discount, + the user simulator + the shadow arm's System-1), plus the unlogged judge; MAIN_RUN_PLAN §5
  estimated ≈ $240 (`results/tau2/runs.json`).
  (7) **Stated difference for the proxy check (A5.4 (e))**: the airline sweep rows were run with `scripts/run_tau2.sh`
  as registered in step 9, which sets tool_select τ = tool_params τ, whereas the E2a sweep arms fix the argument
  threshold at 0.90 (A6.16 (h)); the E2 and τ² sweep configurations therefore share their argument threshold only at
  τ = 0.9, and the proxy check is reported with that difference. Its Kendall-τ code (A6.20 (f)) has not landed; the
  check is not computed yet and lands before it is reported (else a dated deviation).
- **2026-10-05** (record of the freeze; found by the submission audit): no `prereg-freeze` tag was cut and the header's
  `FREEZE_COMMIT` placeholder was never replaced. By the authors' decision of commit `75bab5c` (2026-10-02 17:01:09
  UTC, "Main runs start without a freeze tag"), the main runs started without a tag, behind a start guard that writes
  each run's start commit and the plan hash into its START record (`results/main/START_*.txt`,
  `results/gpu_round4_main/START_*.txt`; plan sha256 `9a2182aa2a975e85` in every start and resume record). "The freeze"
  in this file and in the paper therefore means the registration as committed when the first main run started
  (`START_E1.txt`: 2026-10-02 17:01:32 UTC at `75bab5c`), not a separate event on 2026-10-03. Every pre-freeze
  amendment (A1–A6.22) was committed before that start (the last, A6.21–A6.22, in `de8634f` at 08:22 UTC); after it
  this file changed only by dated Deviations entries, and A7 lives in its own file. The external OpenTimestamps
  attestations of A7.17 cover these files from 2026-10-02 21:46 UTC. No arm, size, seed, margin, test or label changes.

## Pre-freeze amendment A4 — meta-decision golds and wrong-proposal probes (proposed 2026-10-01; accept or reject before the 10/3 freeze)

This amendment belongs with A1–A3 and is not a deviation. It is appended at the end to keep this file append-only.
The normative details are in [`META_GOLD_SPEC.md`](META_GOLD_SPEC.md). It answers dev round 2
(`results/dev_round2/E2_DEV_NOTES.md`): there was no gold for the reflection, routing and memory gates, guard FAR was
undefined, and System-1 filled free-text arguments.

- **Golds.**
  - Every JevGym gold is a pure function of the environment state.
    - It is computed on a copy after System-1 has answered.
    - It is written only to `DecisionLog.gold` / `gold_aux`, never to `Question.meta`, a prompt or a rendered state.
  - The shared primitive is verdict(s, a) ∈ {progress, detour, fatal, premature, wrong_answer}, computed by an
    early-exit oracle rollout. It needs two oracle fixes first:
    - tolerant argument matching in `_done_step`;
    - `get_orders(…, all)` satisfies a status-filtered read.
  - Primary golds:
    - **reflection**: the reflection-free action at step t is failing (fatal / premature / wrong_answer);
    - **routing**: the small model's action at step t is failing;
    - **stop guard**: finishing now with the proposed answer succeeds (`evaluate`). This replaces `oracle_done`;
    - **stop trigger**: only the answer remains;
    - **memory_write**: the offered text contains a value that the remaining oracle plan consumes;
    - **tool guard**: the proposal is in the acceptable-action set;
    - **memory_read**: descriptive only.
  - Tool-level accuracy, C3 recall@k and H1 use the information-feasible acceptable tool set. The canonical tool
    becomes a secondary label.
- **Wording.** These changes affect the C5 and C6 arms and are frozen with this amendment:
  - `Q_REFLECTION` becomes "Should the agent pause and re-check before its next action: is it stuck or off track, or
    likely to make a wrong action or give a wrong final answer next?";
  - `Q_STOP_GUARD` shows the proposed answer;
  - the stop-guard rejection note asks the agent to re-check the requests and recompute the answer, instead of
    "continue working".
- **Measurement arms.**
  - `meta_shadow` (Flash, n = 200) and `meta_shadow_s2lite` (Flash-Lite, n = 500) behave as LLM-only. Reflection,
    stop and memory_write run in shadow mode, and routing does too on Flash-Lite.
  - AUROC, J and accuracy for H2 and C5 come from these arms. Other backends are scored by replaying the logged
    states.
  - Deferred one-step counterfactual System-2 calls run after each episode, netted against a placebo resample. They
    give the efficacies e and h.
  - Like all shadow arms, these are excluded from Holm, iso-quality and Pareto.
- **Wrong-proposal probes.**
  - Probes are phantom proposals that the guard scores after the episode. They are never executed, run only in the
    measurement arms, and are stored outside `steps[]` in `Trajectory.probes`.
  - Primary sources are model-generated: natural proposals, forced premature finishes and weak-model transplants.
    Synthetic edits that use only state-visible values serve for robustness, and are the only source of fatal
    tool-guard probes.
  - The oracle labels and selects probes, but never writes their content.
  - Probes never enter success, cost, latency, the non-inferiority tests, Holm or the matched-rate protocol.
  - No live fault-injection arm is run: injected answers are not System-2's belief, so the measured rescue would be
    inflated.
- **C6 readout.** "Guard FAR/FRR on ≥ 50 wrong proposals" becomes:
  - FAR on ≥ 50 wrong finish proposals per System-2 model (natural, forced-premature and transplant, under the
    answer-aware gold), stratified by kind;
  - natural FAR on Flash-Lite, with an equivalence check of ±0.10 against the probe FAR;
  - FRR on natural right proposals;
  - rescue from natural wrong proposals only.

  The non-inferiority primary is unchanged.
- **H2.**
  - A cell needs ≥ 30 positive and ≥ 30 negative golds. Otherwise it is listed as "no opportunity".
  - Rate backends are matched controls, not cells.
  - ΔU is computed gross of System-1's own price; net ΔU is reported alongside.
  - The memory-write cells hold the read mode fixed (`memory_read: always`).
  - A secondary theory check tests ΔU = W·n̄·π(1−π)·(e+h)·J − κ·n̄.
- **C5.**
  - The primary is unchanged.
  - Secondary readouts are AUROC/J for the reflection gold, ē net of placebo, and predicted against observed ΔU.
  - If the CI of the one-step oracle-ceiling ΔU at W = 1 lies within ±3.3 pt, C5 is reported as "no headroom on
    this suite".
  - The Flash-Lite C5 control arms are added (`ratem`, `rate10/30/50`, `heuristic`, `s2self`).
- **Free-text parameters.**
  - Each parameter is classified as bounded, copy or free, from the schema only. The default for an unrecognised
    string is free.
  - Before the freeze, a copy parameter whose dev-seed candidate coverage is below 0.8 is reclassified as free.
  - System-1 never fills free parameters (`finish.answer`, `ask_user.question`, …). System-2 fills them, required or
    optional.
  - H3's "copy-dependent arguments" means the copy class only.
- **Cost.** Measurement costs about $55 extra and the new arms about $40. These are estimates from dev unit prices.
- **Slip rule.**
  - Must land before the freeze, because they change primary arms: the wording, the free-text policy and the arm
    definitions.
  - May land later: gold logging, the counterfactual and probe runners, and the metric code. JevGym golds are
    recomputed by deterministic replay of the logged actions, and the measurement arms run later on the same tasks.
  - Anything still missing when the analysis runs is filed as a dated deviation.

## Pre-freeze amendment A5 — review-round-4 P0 items (proposed 2026-10-01; accept or reject before the 10/3 freeze)

Appended after A4 to keep this file append-only. A1–A4 and the frozen text above are unchanged; where A5 restates
one of their numbers (the Holm family size in §2, A2's budget line, C7's airline size) A5 supersedes it from the
freeze on. Source: [`REVIEW_ROUND4.md`](REVIEW_ROUND4.md) §2 (P0-1 … P0-5); the options that review left open are
settled here. Everything below is written before any main run.

### A5.1 — Two System-1 families (P0-1)

- C1, C2 and C6 are run and reported for **two** prefill-only deciders on identical tasks and seeds: **Jev**
  (`typesafe/jev-1.13`, served id logged) and **Intern-Decision-4B** (`internlm/Intern-Decision-4B`, Apache-2.0,
  in-process; `Intern-Decision-2B` if the 4B does not fit the GPU budget, and the paper says which).
- A primary claim is **confirmed only if it holds for both families**. Each of C1, C2 and C6 is therefore a
  conjunction of two tests (the Jev arm and the Intern-Decision arm, each against the same reference, each at the
  Holm-adjusted level of that primary; a conjunction cannot raise the Type-I rate, so no further correction). A split
  result (one family passes, the other fails) is reported as *split*, with the decision-accuracy gap between the two
  families on the same logged items.
- **Competence gate** (so that a weak open model cannot sink the primaries by construction). Before the main runs
  both backends answer the 210 gold decisions of the training-side dev set `data/train_v0/dev.jsonl` (tool_select 62,
  tool_params 86, stop 62; JevGym seeds ≥ 1000, never used for evaluation) through the same harness backends
  (`decision.backends`, cache off). Intern-Decision is promoted to **co-primary** iff its decision accuracy on these
  items is at least **0.8 × Jev's** (point estimates; the paired-bootstrap CI of the ratio is reported). Otherwise it
  is reported as a **secondary** System-1: its rows stay in every headline table, labelled, the gate result is stated
  in the paper's §6, and C1/C2/C6 are confirmed on Jev alone. Jev's gate cost ≈ $0.01; the gate verdict is written
  to `runs/gate_intern_decision.json` before any C1/C2/C6 arm runs.
- H2 cells and the VoI readout get Intern-Decision by **replaying the logged measurement states** (A4 measurement
  arms; `META_GOLD_SPEC.md` §15 backend identity): no System-2 cost.
- C3, C4, C5, C7 and C8 stay Jev-only primaries; Intern-Decision rows there are secondary.
- Sizes follow A2: the Intern-Decision C1/C2/C6 arms and their references at n = 500, every other Intern-Decision arm
  at n = 200 (`configs/experiments/jevgym_tool_select_intern.yaml`, plus a stop-guard arm for C6; the config is
  updated to these sizes before the runs). Prices for `internlm/Intern-Decision-*` in `configs/prices/models.yaml`
  are an assumed amortised GPU cost until `scripts/measure_latency.py` has run on the real GPU.
- Cost: GPU ≈ 6–10 h (4B); System-2 API ≈ $40–60.

### A5.2 — H5 ("faster and better" on hard tasks) registered (P0-2)

- H5 is **exploratory**: outside the C1–C8 Holm family, never a stopping or selection rule for the primaries. Its
  arms share the unmodified harness's tasks, System-2 model and caps and are paired with it per task.
- **Attempts:** k = 3 per task on Terminal-Bench-Science and Terminal-Bench 2.0, k = 1 on SWE-bench Verified. Per
  task, success is the attempt mean and agent-run time is the attempt mean with timeouts at the cap.
- ***Better*** iff the paired Δsuccess (arm − unmodified, attempts averaged per task) has a **one-sided 90 %
  task-bootstrap lower bound (B = 10,000) above −5 pt** **and** **wins ≥ losses** (tasks with higher / lower per-task
  success; ties dropped; the exact sign test is reported, two-sided and one-sided; McNemar's test where k = 1).
- ***Faster*** iff, on the **tasks both arms solve** (at least one successful attempt in each arm; a task's time in
  an arm is the mean agent-run time of its solved attempts), the **median paired Δ agent-run time is < 0** **and** a
  **one-sided Wilcoxon signed-rank test gives p < 0.05** (zeros dropped, average ranks; exact conditional
  distribution up to n = 200, normal approximation with tie correction beyond). The all-tasks median Δ (timeouts at
  the cap included) is reported next to it, never as the criterion.
- ***Faster and better*** needs both. Solved-by-t curves at t ∈ {600, 1800, 3600} s, the fraction of verifier tests
  passed and the cost change are descriptive only.
- The rule was chosen so that one capped attempt moves Δsuccess by 1/(k·n) and each median by at most one order
  statistic; it cannot, by itself, change a verdict that is not already on the boundary (checked offline in
  `tests/test_metrics.py`).
- Readout: pure functions in `src/jevharness/bench/metrics.py` (`paired_success_lb`, `sign_test_exact`,
  `wilcoxon_one_sided`, `h5_verdict`); `adapters/pi/summarize_harbor.py --baseline <arm>` prints the verdict for
  the Harbor suites, and the mini-SWE summary uses the same functions.

### A5.3 — Flash-Lite as co-primary System-2 for C1, C2, C6 (P0-3)

- C1, C2 and C6 each become a **pair of tests**: **Flash** (n = 500; the "no loss at the ceiling, X % cheaper"
  reading) and **Flash-Lite** (n = 500; the "where System-1 adds success" reading). All eleven primaries form **one**
  Holm family at family-wise α = 0.05: **C1F, C1L, C2F, C2L, C3, C4, C5, C6F, C6L, C7, C8** (§2's "Holm across the 8"
  is superseded). One family, not two: Holm controls the FWER over any set of one-sided nulls, so NI tests
  (H0: Δ ≤ −m) and superiority tests (H0: Δ ≤ 0) share it (`REVIEW_ROUND4.md` §6).
- C5 stays Flash-Lite only (A1). C3, C4, C7 and C8 are unchanged (Flash).
- Arms raised to n = 500 (per-arm `n_tasks: 500` over the 500-task suite, the A2 mechanism):
  `configs/experiments/jevgym_tool_select.yaml`: `llm_only_s2lite`, `jev_shadow_s2lite`,
  `jev_ts_assist_k3_s2lite` (C2L) and `jev_ts_cascade_tau0.9_s2lite` (the C1L arm, whose τ is set to the Flash-Lite
  τ̂\* once the `jev_shadow_s2lite` fit is in); `configs/experiments/jevgym_meta_control.yaml`: `llm_only_s2lite`
  and `jev_stop_guard_s2lite` (C6L); `meta_shadow_s2lite` was already at 500 (A4). Every other arm keeps its A2 size.
- **A2 budget line, amended (A2's text stays as written):** n = 500 for the C1–C6 arms and their references on
  *both* System-2 models, n = 200 elsewhere; + ≈ $15–25 for the Flash-Lite pairs (≈ $0.01 per Flash-Lite episode,
  ≈ 4 × 300 extra episodes) and + ≈ $40–60 System-2 API plus 6–10 GPU-h for A5.1, on top of A2's $1.5–2k.

### A5.4 — C7: degenerate cases of the selection rule and the L2→L3 proxy check (P0-4)

- (a) **Disjoint data.** The configuration is selected on E2 (JevGym, Flash) and tested on τ²; no τ² number enters
  the selection, and no τ² test task enters the τ̂\* fit (c).
- (b) **Default and tie-break arm.** The tool+args CASCADE at τ̂\* (the C1 arm) is C7's arm if several E2
  configurations are within 2 pt of cost savings of the best, or if no E2 configuration is NI.
- (c) **τ̂\* on τ² is refit (primary).** The 50 airline tasks are split once, before any τ² run, by a seeded shuffle
  (seed 0 over the task ids in sorted order) into a **15-task shadow split** (30 %, as H1's calibration split) and the
  **35-task airline test set**. Jev shadows the LLM-only arm on the shadow split (`jev_shadow`, 3 trials) and τ̂\* is
  refit from that log with the same H1 fit the C1 arm uses. C7's comparisons use the 35 airline test tasks × 3 and all
  114 retail tasks × 3 (149 tasks, within the §1 planning size). The **E2-carried τ̂\*** (the Flash fit on JevGym) runs
  as a **secondary row** (`jev_ts_cascade_tau<τ̂*_E2>`), Holm-adjusted within the τ² table, and the two thresholds
  are reported side by side. Every τ² arm still runs on all 50 airline tasks so that arms share one task list; the
  15 shadow tasks are excluded from every C7 and fallback comparison and reported descriptively.
- (d) **Fallback.** If C7 fails NI, the C1 arm (if C7's configuration differs from it) and tool ASSIST k = 3
  (`jev_ts_assist_k3`) run on τ² airline only (50 × 3; tested on the 35 test tasks) as secondary rows, Holm-adjusted
  within the τ² table. The paper reports "transfer failed" for C7 either way.
- (e) **Proxy check.** Kendall τ between the E2 (Flash) and τ²-airline rankings of every configuration that ran on
  both (LLM-only, CASCADE at each swept τ, ASSIST k = 3, policy GUARD), ranked by paired Δsuccess vs LLM-only and,
  separately, by iso-quality cost savings; reported with n and a permutation p.
- Tooling: `scripts/run_tau2.sh` recognises `jev_shadow`; shadow mode in the τ² agent (`bench/suites/tau2.py`
  rejects it today) must land before the shadow run, which precedes the C7 runs, not the freeze.

### A5.5 — C4: no-opportunity rule (P0-5)

- C4 is tested **only if the Flash-Lite measurement arm `meta_shadow_s2lite` (n = 500) yields ≥ 30 routing
  positives** (routing gold = the small model's action at that step is failing, A4). The count is read from the
  measurement arm before any C4 live arm is scored. Otherwise C4 is reported as **"no opportunity on JevGym"**, leaves
  the Holm family (ten primaries remain; the removal does not depend on C4's outcome), and routing is read
  descriptively from the hard-task arms (H5). At the pilot base rate (3.5 % of steps) n = 500 should give 100+
  positives, so this is a safeguard, not a retreat.

### A5.6 — Slip rule and costs

- Must land before the freeze, because they define primary arms and tests: this text (A5.1–A5.5), the YAML sizes
  and the arm names.
- May land later, before the run it serves: the H5 readout code, the τ² shadow mode, the competence-gate scorer,
  the Intern-Decision C6 arm config. Anything still missing when the analysis runs is filed as a dated deviation.
- Cost: A5.1 ≈ $40–60 API + 6–10 GPU-h; A5.3 ≈ $15–25; A5.4 shadow ≈ $30, fallback ≈ $50–100 if triggered;
  A5.2 and A5.5 writing and code only.

### A2 budget sign-off (2026-10-01)

The budget in A2 (as amended by A5.3) was approved by the PI on 2026-10-01. Accordingly the Flash primary arms and
their references now carry `n_tasks: 500`: `llm_only`, `jev_shadow`, `jev_ts_assist_k3` (C2F) and the explicit C1F arm
`jev_ts_cascade_tauhat` (τ set to τ̂\* from `jev_shadow` before the run; the τ sweep stays at n = 200) in
`jevgym_tool_select.yaml`; `llm_only` and `jev_stop_guard` (C6F) in `jevgym_meta_control.yaml`. No test, margin or
arm definition changes; this records the sign-off A2 asked for.

## Pre-freeze amendment A6 — review-round-5 P0 items (proposed 2026-10-01; accept or reject before the 10/3 freeze)

Appended after the A2 sign-off to keep this file append-only. A1–A5 and the frozen text above are unchanged; where
A6 restates one of their numbers or rules (§2's NI test sentence, §3 H2's *Test* / *Falsified if*, A5.5's family-size
clause, the evaluation seed range of §1 and `BENCHMARK_SPEC.md` §8, A5.2's verdict names, the source of τ̂\* for the
C1 arms, A5.4(c)'s "E2-carried τ̂\*") A6 supersedes it from the freeze on. Source: [`REVIEW_ROUND5.md`](REVIEW_ROUND5.md)
§1 (P0-1 … P0-10), §5 (rulings a–e) and §6; every option that review left open is settled here. Everything below is
written before any main run. Code named here (`metrics.ni_pvalue`, `ni_exact_check`, `ni_power`, `h5_verdict`'s
`strictly_better`, `meta_gold.h2_regression_j`, `audit_implications`, `jevgym.oracle_verdict`'s `fatal_kind`) lands
under the A6.10 slip rule.

### A6.1 — The NI test as one procedure; power at the registered sizes (P0-1, P0-2)

- **Statistic.** For a primary NI test with margin m (3 pt on JevGym, ALFWorld and L1; 5 pt on τ² pass^1) the test
  statistic is the paired task-bootstrap distribution of Δ\* = arm − reference (B = 10,000; the task is the resampling
  unit; trials of one task stay together; `metrics.paired_bootstrap_diff`). Its p-value is **p_NI = P\*(Δ\* ≤ −m)**
  (`metrics.ni_pvalue`); superiority primaries (C3, C4, C5) use p_sup = P\*(Δ\* ≤ 0).
- **Holm.** p_NI and p_sup enter the one Holm family of 11 (A5.3) at family-wise α = 0.05. For C1, C2 and C6 the
  two-family conjunction of A5.1 enters Holm as **p = max(p_Jev, p_Intern)** (intersection–union test; no further
  correction); while Intern-Decision is a secondary System-1 (competence gate), p = p_Jev.
- **Decision rule.** A primary NI test rejects H0: Δ ≤ −m iff p_NI ≤ its Holm-adjusted level α_adj, equivalently iff
  the one-sided (1 − α_adj) bootstrap lower bound of Δ is above −m. §2's one-sided 95 % lower bound is still reported
  next to every primary, as a descriptive bound; it is no longer the decision rule.
- **Few discordant tasks.** When a comparison has fewer than 5 discordant tasks (tasks whose per-task success differs
  between the two arms), the bootstrap is anti-conservative at the Holm floor (simulated type-I 0.013 at d = 4 %,
  `results/reviews/2026-10-01b/ni_power.out`), so the bound is reported together with the exact one-sided binomial
  p-value of H0: P(loss) ≥ m over the n paired tasks, p = P[L ≤ l | n, m] (`metrics.ni_exact_check`), and the exact p
  is the one that enters Holm for that primary. With zero discordant tasks p = (1 − m)^n (0.97^500 = 2.4·10⁻⁷ for
  m = 3 pt, n = 500): NI holds, and the paper reports the one-sided 95 % upper bound on the arm's loss rate,
  1 − 0.05^{1/n} (0.6 pt at n = 500), instead of a zero-width interval.
- **Descriptive intervals.** Score-type intervals for paired proportions (Newcombe 1998; Tango 1998) are reported as
  descriptive CIs for single-trial binary endpoints; they are not the test. Replacing the bootstrap by Tango /
  Agresti–Min / Farrington–Manning (round 5, G) was considered and rejected: the bootstrap holds its size at these n
  (simulated type-I at the NI boundary 0.051–0.060 at α = 0.05 and ≤ 0.013 at the Holm floor 0.05/11; `ni_power.out`)
  and is the one estimator that also handles the τ² 3-trial clusters. A fixed-sequence (gatekeeping) procedure —
  Flash pairs first at full α, then Flash-Lite, then C7 — would raise power but changes A5.3 at the freeze; considered
  and rejected.
- **Sizes stay** n = 500 (JevGym) and 149 tested tasks (τ²; A5.4). §1's planning line "≈ 340 tasks" is the
  unadjusted-α figure; the same case needs ≈ 650 tasks at the Holm floor 0.05/11 (the first Holm step; later steps
  are tested at 0.05/10 … 0.05, so the family's achieved power lies between the two columns). Achieved power
  (one-sided, 80 %-power Wald/score approximation and a 1,000–1,500-replication simulation of the bootstrap rule;
  `results/reviews/2026-10-01b/ni_power.py` → `ni_power.out`; `metrics.ni_power` reproduces the planning column):

  | primary | n | m | discordance d (r2 pilot) | true Δ | power, α = 0.05 | power, Holm floor 0.05/11 |
  |---|---|---|---|---|---|---|
  | C1F, C2F, C6F (Flash; ceiling reference; pilot d = 0–10 %, losses only) | 500 | 3 pt | 2 % | −1 pt | 0.94 | 0.73 |
  | | | | 5 % | 0 | 0.91 | 0.66 |
  | | | | 10 % | 0 | 0.69 | 0.32 |
  | | | | 10 % | −1 pt | 0.41 | 0.12 |
  | C1L, C2L, C6L (Flash-Lite; pilot d = 10–50 %, wins and losses) | 500 | 3 pt | 20 % | 0 | 0.44 | 0.14 |
  | | | | 30 % | 0 | 0.34 | 0.08 |
  | | | | 20 % | +3 pt | 0.91 | 0.65 |
  | | | | 30 % | +5 pt | 0.95 | 0.75 |
  | C7 (τ² pass^1; 35 + 114 tested tasks) | 149 | 5 pt | 20 % | 0 | 0.39 | 0.11 |
  | | | | 30 % | 0 | 0.30 | 0.07 |
  | | | | 30 % | +3 pt | 0.56 | 0.21 |

  Reading: n = 500 is adequate for the Flash NI pairs (a true loss ≤ 1 pt at ≤ 5 % discordance) and for C3 / C8
  (superiority / NI at low discordance on ≥ 500 units; `REVIEW_ROUND5.md` §5a); it is **not** adequate for the
  Flash-Lite NI pairs at a true difference of zero (they are powered only if System-1 adds ≥ 3 pt, which is the
  A5.3 reading of those arms) nor for C7 unless the per-task Δ sd is ≤ 0.25. Margins are not widened to fit the sample.
- **MDE.** For every primary the paper prints, next to the result, the minimum detectable effect: the true Δ at
  which the registered test has 80 % power at the achieved discordance and the primary's Holm level (same
  approximation; `metrics.ni_power`). Planning values at n = 500, m = 3 pt, d = 5 %: a true loss of ≤ 0.5 pt is
  detectable as NI at α = 0.05, and a true gain of ≥ 0.45 pt is needed at the Holm floor.
- **Reading rule (registered).** A primary NI test that fails while the two-sided 95 % task-bootstrap CI of Δ covers
  0 is reported as ***inconclusive at this n***, never as inferiority; the word "harm" is not used for a failed NI
  test. A failed NI test whose 95 % CI lies entirely below 0 is reported as a loss of the stated size.

### A6.2 — H2: J at the operating point and the predicted gain of Eq. 2 (P0-3)

Replaces §3 H2's *Test* and *Falsified if* sentences; A4's H2 cell rule (≥ 30 positive and ≥ 30 negative golds in
the measurement arm; live and matched-rate arms present) is unchanged.

- **Primary regression.** For each estimable cell the dependent variable is the paired ΔU of the live gate over its
  matched-rate random gate, gross of System-1's price (net alongside), at the primary operating point **W = $1**
  (W ∈ {0.1, 10} are sensitivity rows). ΔU is regressed on **J at the operating point** — J_τ = TPR − FPR of the
  measurement-arm gate thresholded at the live arm's τ (`meta_gold.h2_regression_j`; `gate_stats` already computes
  J) — and on accuracy at the same τ; a likelihood-ratio test asks whether accuracy adds to J. *Falsified if* the J
  slope is ≤ 0, or accuracy explains ΔU better than J (LR test, α = 0.05).
- **Second primary readout (theory check).** Predicted ΔÛ = W·n̄·π(1−π)·(e+h)·J_τ − κ·n̄ (Eq. 2 of the paper; A4's
  "secondary theory check") against observed ΔU, cell by cell, with a task-level percentile bootstrap (B = 10,000)
  over the cells' shared tasks; the paper reports the predicted-vs-observed scatter with both intervals and the share
  of cells whose observed ΔU lies inside the predicted interval. This readout is reported, not a falsification
  condition; H2 is falsified only by the regression above.
- **AUROC** is the τ-free upper bound on attainable J (App. A: max_τ J ≤ 2·AUROC − 1); the AUROC regression of §3
  becomes an exploratory row. The ±2-pt TOST of the random gate against the System-1 gate (§3 H2) leaves the
  falsification list and is reported as a descriptive equivalence check per cell (underpowered at ≈ 40 cells;
  `REVIEW_ROUND5.md` §5c-14).
- **Guards' counterfactual.** For guards (stop guard, tool guard) e and h are measured against *executing the
  original proposal* — e_raw, h_raw: the live alternative to a rejection is the proposal going through
  (`meta_gold.voi_readout`'s `verdict_reject` branch for the stop guard, whose outcome is `finish_ok`; the proposal's
  own verdict for the tool guard) — and the placebo-netted values are reported alongside; the prediction ΔÛ uses
  e_raw / h_raw for guards. For pre-action gates (reflection, routing, memory write) the placebo-netted e, h stay
  primary because the default resample *is* the live alternative.
- **Assumptions of Prop. 2** (stated in the paper's §3.4): beyond additivity across opportunities, (i) effect
  homogeneity within need strata (e and h do not depend on the state beyond need y), and (ii) equal expected firing
  cost at the matched rate (the module's cost does not depend on y). Violations show up as a predicted-vs-observed
  gap and are reported, not corrected.
- **Intern-Decision** enters H2 by replaying the logged measurement states (A5.1): the replay yields H2 *scores*
  (AUROC, J_τ, accuracy) and the VoI *prediction*, not live cells (A5.1's "H2 cells" reads so from the freeze on;
  `REVIEW_ROUND5.md` §6).

### A6.3 — C4 stays in the Holm family (P0-4)

- A5.5's clause "leaves the Holm family (ten primaries remain; the removal does not depend on C4's outcome)" is
  **replaced** from the freeze on by: when `meta_shadow_s2lite` (n = 500) yields fewer than 30 routing positives,
  **C4 stays in the 11-test family and enters Holm with p = 1**, reported as ***insufficient observed
  opportunities*** (not as zero gain, not as "no opportunity on JevGym"). The rest of A5.5 — the ≥ 30-positive rule,
  the count read from the measurement arm before any C4 live arm is scored, the descriptive routing readout from the
  hard-task arms — is unchanged.
- Reason: the opportunity count comes from the same task population as the other tests, so a data-dependent family
  size is not what Holm licenses; the planned size was 11 and keeping C4 in costs nothing. App. F's "with predicted
  gain zero" becomes "insufficient labels; no gain is asserted".

### A6.4 — Evaluation seed namespace and the exposure manifest (P0-5)

- **Evaluation tasks are JevGym seeds 100000–100499**: `suite: {name: jevgym, n: 500, seed0: 100000, …}` in
  `configs/experiments/jevgym_tool_select.yaml`, `jevgym_meta_control.yaml`, `jevgym_tool_select_intern.yaml` and
  `baselines_jevgym.yaml` — every arm, measurement arms and controls included, so pairing across E2a / E2b / E2c is
  preserved. The suite takes the offset through `task_factories(n, seed0)` (`bench/suites/jevgym.py`; `runner.py`
  passes the `suite:` block through `load_suite`); task ids stay `jevgym-<family>-<seed>-d60-…`, and family
  round-robin and variant choice are functions of the seed, so the suite's family balance is unchanged. No seed
  ≥ 100000 was run before 2026-10-03. Gemini's "1000–1499" (round 5) was rejected: it collides with A5.1's dev range.
- **Exposure manifest** (every JevGym task id logged under `runs/` as of 2026-10-01, counted by script):
  - seeds **0–59** were run with real models: the Jev pilots n = 10 (seeds 0–9: `pilot_jev`, `pilot_jev_v2`,
    `pilot_jev_r2`) and n = 50 (0–49: `pilot_jev_n50`), the Gemini pilot (0–19), the v2 calibration pilot (0–29),
    dev rounds 1–3 (0–9; `e2b_dev_r3` 0–29), the CF validity pilot (0–29 and 30–59), the E2c check (0–4); prompts
    (A4's stop-guard wording) were changed on this evidence, so **0–999 are development seeds**;
  - seeds **0–149** were touched by simulated-backend runs only (`sim_threshold`, `sim_design_space`, `sim_train`);
  - **no seed in 1000–9999** has been run; 1000–4999 are the training-side dev defaults (`training/grpo_online.py`
    default suite `seed0: 1000`; A5.1's "seeds ≥ 1000" is the `harvest.py --min-jevgym-seed` guard), **5000–5999 are
    calibration** (`BENCHMARK_SPEC.md` §8; `training/opd.py --eval-seed0 5000`), of which **5000–5149** are A6.5's
    calibration shadow arms;
  - seeds **≥ 10000 are training**: `data/train_v0/MANIFEST.json` holds v2 10000–10029 and v1 12000–12059 (the
    competence-gate dev set `dev.jsonl` draws its 210 items from 10005–12055, A5.1), i.e. **10000–12059** so far.
- Code items (A6.10): `training/harvest.py --min-jevgym-seed` (default 1000) must also exclude the evaluation range
  100000–100499, and `grpo_online`'s default `eval_suite` (`seed0: 0`, n = 16) moves to the 5000 range; until then
  no training run may read a `runs/` directory produced by an evaluation config. Stale "seeds 0..199 / 0..499"
  comments in `configs/train/*`, `configs/dev/*`, `TRAINING_RECIPES.md` and `TRAINING_AGENT_BRIEF.md` are
  documentation only and are updated as touched.

### A6.5 — Where τ̂\* for the C1 arms comes from; the operating point; the threshold form (P0-6)

- **Calibration arms.** τ̂\* for C1F (`jev_ts_cascade_tauhat`) and C1L (`jev_ts_cascade_tau0.9_s2lite`, A5.3) is
  fitted from the new shadow arms **`jev_shadow_cal`** (Flash) and **`jev_shadow_cal_s2lite`** (Flash-Lite), and for
  the Intern-Decision family from **`intern_shadow_cal`** / **`intern_shadow_cal_s2lite`**: each shadows tool_select,
  tool_params and stop with the same System-2 as the arm it calibrates, on the **150 calibration seeds 5000–5149**
  (`suite: {name: jevgym, n: 150, seed0: 5000, …}` with the frozen A1 settings; n = 150, as the 30 % calibration
  share H1 uses, so the fit is as precise as the registered H1 fit). They never run on evaluation tasks. Because the
  runner builds one task list per experiment, the calibration arms live in their own experiment files,
  `configs/experiments/jevgym_shadow_cal.yaml` (Jev) and `jevgym_shadow_cal_intern.yaml` (Intern-Decision), not as
  arms of the evaluation YAMLs. The evaluation shadow arms (`jev_shadow`, `jev_shadow_s2lite`, `intern_shadow*`,
  n = 500) stay: they serve H1's regret analysis and the Prop. 1 tests, never the deployed τ.
- **Fit.** `python -m jevharness.bench.threshold_theory runs/e2a_jevgym_shadow_cal --arm jev_shadow_cal --dp
  tool_select --rule general --task-value 1.0 --lam-per-hour 0 --train-frac 1` (the tool fits g_d, q_L(c), ΔC and V
  on every logged decision; its held-out regret block is read only on the evaluation shadow log, H1), likewise
  `--dp tool_params`, and `--arm jev_shadow_cal_s2lite` for the Flash-Lite arm: **one τ̂\* per decision point and
  System-2 model**, written into the `tau:` of the C1 arm's `tool_select` and `tool_params` before it runs (rounded
  to two decimals; the fitted values are recorded in the Deviations section with their date).
- **Operating point.** Primary: **W = $1 per task success, λ = 0** (`--task-value 1.0 --lam-per-hour 0`), for C1, C5
  and the H2 regression alike; W ∈ {0.1, 10} and λ ∈ {0.5, 5} $/h are the sensitivity rows of the H1 grid.
- **Threshold form.** The deployed rule is the **scalar-threshold projection of Eq. 3**: τ̂\* = the smallest c at
  which the fitted advantage a(c) = g(c) − q_L(c) + ΔC/V is ≥ 0 (`general_crossing`); when {a ≥ 0} is not a threshold
  region (`general_interval` false) the gain-maximising threshold over the observed confidences (`general`) is
  deployed and the disagreement fraction between threshold and pointwise rule is reported. The pointwise rule is a
  secondary row (simulated on the shadow log), never a live arm.
- **Bins and fallback.** g(c) and q_L(c) are the isotonic (monotone) fits over all paired calibration decisions of
  the decision point; the 10 equal-width bins of `--n-bins 10` are descriptive. If a decision point has fewer than
  30 paired calibration decisions, its τ̂\* uses the marginal rule `prop1` (q_L − ΔC/V with the pooled q_L); if fewer
  than 10, the arm keeps τ = 0.9 and the paper says so.
- **C1 is tested on all 500 evaluation tasks** (no held-out subset: the 350-task alternative would cut the floor
  power from 0.66 to ≈ 0.5). H1's regret analysis keeps its registered 30/70 split of the evaluation shadow log
  (`--train-frac 0.3` on `jev_shadow`); A5.4(c)'s "E2-carried τ̂\*" for the secondary τ² row is now the Flash
  calibration-arm τ̂\*.
- **Cost** ≈ $8 per System-1 family (Flash ≈ $6, Flash-Lite ≈ $1.5, Jev ≈ $0.3); the Intern arms add ≈ $8 of
  System-2 API and ≈ 1 GPU-h.

### A6.6 — H5 verdict names (P0-7)

- A5.2's registered rule is unchanged but **renamed**: an arm whose paired Δsuccess has a one-sided 90 %
  task-bootstrap lower bound above −5 pt with wins ≥ losses, and that is *faster* on the co-solved tasks (Wilcoxon
  p < 0.05), is ***success-preserving and faster***. ***Faster and better*** is reserved for the stricter tier:
  one-sided 90 % lower bound of Δsuccess **> 0** **and** exact one-sided sign test p_wins < 0.05, together with
  *faster*; both tiers are reported for every arm (`metrics.h5_verdict`, field `strictly_better`). The all-task
  capped-time median stays reported next to the co-solved median, never as the criterion.
- A5.2's sentence "it cannot, by itself, change a verdict that is not already on the boundary" is replaced by: "one
  capped attempt moves Δsuccess by 1/(k·n) and each median by at most one order statistic
  (`tests/test_metrics.py::test_h5_one_timeout_cannot_flip_the_verdict`)".
- Paper: intro H5 "raises success" → "preserves success"; §6, Table 2's footnote and App. I carry the two tiers; the
  Limitations' robustness sentence is softened accordingly.

### A6.7 — Oracle labels: the kind of `fatal`, the implication audit, the retracted "never" (P0-8)

- `oracle_verdict` logs **`gold_aux.fatal_kind ∈ {terminal, budget}`** with every fatal verdict: *terminal* when the
  step ended the episode unsuccessfully (proven in the strict world); *budget* when the canonical plan's cost-to-go
  after the step, D(s′), is ∞ because the plan no longer fits the step budget (oracle-relative: D is an upper bound on
  the optimum, so a budget-kind fatal could in principle be recoverable by a non-canonical plan).
- **Reporting.** The share of `need_fatal` positives (and of fatal verdicts in every gold built on F = {fatal,
  premature, wrong_answer}) whose kind is `budget` is reported per arm, and a **sensitivity row excludes budget-kind
  positives** from the H2 cells and from C5.
- **Implication audit.** `meta_gold audit` checks every logged trajectory (`meta_gold.audit_implications`): an
  executed action with verdict ∈ {fatal, premature, wrong_answer} ⇒ the episode failed; a finish with `finish_ok` ⇒
  the episode succeeded. The violation count **must be 0**; it is printed per arm and reported in App. F. A non-zero
  count is an oracle bug that stops the analysis of the affected decision point until fixed (filed as a deviation).
- `BENCHMARK_SPEC.md` §8 and `META_GOLD_SPEC.md` §11.1 are reworded: residual oracle errors never touch `finish_ok` or
  `world_complete` (both rest on `evaluate`); they can reach `need_fatal` only through the budget route, whose share is
  reported. The exhaustive-planner certification of budget-kind fatals is P1-4 (before submission), not a freeze item.

### A6.8 — The counterfactual-validity run (P0-9)

- **Registered run.** The `meta_shadow_s2lite` measurement arm (Flash-Lite, the evaluation suite) through
  `scripts/cf_validity.py --mode live` (the continuation needs the in-process snapshots, so the arm and the rollouts
  share a process; the H2 measurement arm itself is run separately, as A4 specifies): **N = 150 opportunities**,
  stratified — every `helps` and every `harms` label up to 50 each, ≥ 60 non-final `neutral`, the rest seeded-random
  (`--min-harms`) — and **R = 3** replicates per branch (≈ 900 rollouts); task-cluster Wilson / bootstrap intervals.
- **Criterion.** e and h are reported as ***validated*** iff the **binary agreement** (one-step `helps` ↔ full-episode
  Δ > 0) is ≥ 0.8 with a Wilson 95 % lower bound ≥ 0.7, **and** the non-final-`neutral` stratum's mean Δ has a 95 %
  task-bootstrap CI within ±0.05. Otherwise e and h are reported as *unvalidated one-step estimates*, and the H2 theory
  readout (A6.2) and the gate τ\* analogue are labelled accordingly. The 3 × 3 table and κ (binary and three-class)
  are descriptive; the same protocol runs on ALFWorld (mid-episode positives) if the budget allows (secondary).
- **Wording.** The dev pilot's 38/42 is the *binary* (helps vs not) agreement and 36/42 the *three-class* sign
  agreement (`results/dev_round2/CF_VALIDITY_NOTES.md` §5–6); App. F says the run is registered here and reports no
  pilot number.
- Cost ≈ $5–10 (Flash-Lite) plus the measurement arm.

### A6.9 — Consolidated manifest: the PREREG-vs-paper discrepancies (P0-10)

Numbered as in `REVIEW_ROUND5.md` §5c; each resolution is binding from the freeze on.

1. **τ² size.** Run size 164 × 3 (50 airline + 114 retail); test size 149 (35 airline test + 114 retail); the 15
   airline shadow tasks are descriptive. Paper Table 2: "164×3 run (149 tested; 15 airline shadow)"; Table 5:
   "airline (35 test tasks × 3; 15 shadow tasks descriptive)".
2. **NI level vs Holm.** A6.1 (p_NI; the Holm-adjusted bound decides; the 95 % bound is descriptive).
3. **C4 and the family size.** A6.3.
4. **Where τ̂\* is fitted.** A6.5 (calibration arms on seeds 5000–5149; H1's 30/70 split is the regret analysis only).
5. **Evaluation seeds.** A6.4.
6. **Confirmation labels.** A C1 / C2 / C6 result is labelled ***cross-backend replication*** when both System-1
   families pass (A5.1's confirmation), ***Jev-specific confirmation*** when the competence gate demotes
   Intern-Decision and the Jev test passes, and *split* when both are co-primary and one fails (A5.1). Table 3's
   caption says so.
7. **CF validity run.** A6.8.
8. **H5 wording.** A6.6.
9. **"No opportunity" = "predicted gain zero".** A6.3 (insufficient labels; no gain asserted).
10. **Iso-quality qualification.** Every paper table qualifies arms with `noninferior` (LCB at the registered margin);
    `not_worse` is not used for a paper number and `BENCHMARK_SPEC.md` §5.2's "conservative" is retracted (the code
    default changes under P1-9 before submission).
11. **C7 eligible set and tie-break.** Eligible = the E2 (Flash, Jev) configurations that also run on τ²: LLM-only,
    tool+args CASCADE at each swept τ and at τ̂\*, tool ASSIST k = 3, policy GUARD. Selected = the most cost-saving
    configuration among those that are NI on E2 by the unadjusted one-sided 95 % bound (a selection rule, not a
    test); configurations within 2 pt of cost savings of the best tie, and ties and the no-NI case resolve to the C1
    arm (A5.4(b)); if the C1 arm itself is not NI on E2, C7 runs it anyway and A5.4(d)'s fallback clause applies
    ("transfer failed" is reported either way). Table 3's footnote ‡ carries the fallback sizes "airline 50 × 3,
    tested on the 35 test tasks".
12. **Freeze tense and hash.** At the freeze the header of this file gains the line "Frozen 2026-10-03 at commit
    `FREEZE_COMMIT`" (placeholder until the freeze tag is cut; the paper's §6 cites it as `\tbd`); the paper's "were
    frozen" is true only once that line exists.
13. **Dev-set seed range.** A6.4's manifest: A5.1's "seeds ≥ 1000" is the harvest guard; the actual dev items are at
    10005–12055, inside the training range and never evaluation seeds.
14. **H2's ±2-pt TOST.** A6.2 (leaves the falsification list; descriptive per cell).
15. **Second System-2 family.** Gemma 4 is removed from the paper's System-2 list until it runs (A3); the
    `\todo{A non-Google family (P1-3)}` stays. It remains a labelled generative-decider reference row if it runs.
16. **"Every headline table".** Placeholder rows for Intern-Decision (C1, C2, C6) are added to the paper's L2
    results now and filled with the results.

Further rulings from P0-10 and §6:

- **C8's endpoint.** Item-weighted accuracy over the BFCL tool-selection items of the six categories (3-pt NI of Jev
  against the System-2 model answering the same typed question through structured output); argument-slot accuracy
  and When2Call are secondary rows, Holm-adjusted within the L1 table.
- **C5's W = 1** (the primary operating point of A6.5; the no-headroom ceiling check of A4 already uses it).
- **Drop rule (paired complete-case).** A task that still errored in either arm after the two extra attempts (§1) is
  out of that pairwise comparison; the per-arm dropped count and the resulting denominators n′ ≤ n are printed with
  every primary, and a full-set sensitivity row counts an errored task as a failure in the arm where it errored.
- **Prices wording.** "List prices" → "tariff-estimated with measured cached tokens (Gemini); OpenRouter-billed for
  Jev" in the paper's §4 and App. H (the ledger fields of P1-3 follow before submission).
- **A5.1's H2 wording.** "H2 cells and the VoI readout get Intern-Decision by replaying" reads "H2 scores and the VoI
  prediction" (A6.2).

### A6.10 — Slip rule and costs

- Must land before the freeze, because they define tests, labels, sizes or samples: this text (A6.1–A6.9), the seed
  namespace in the four E2 configs, the two calibration-arm configs, and the paper wording changes listed above.
- May land later, before the run or analysis it serves: `metrics.ni_pvalue` / `ni_exact_check` / `ni_power`,
  `h5_verdict`'s `strictly_better`, `meta_gold.h2_regression_j`, `voi_readout`'s e_raw / h_raw,
  `audit_implications`, `jevgym.oracle_verdict`'s `fatal_kind`, the harvest-guard and GRPO eval-seed defaults, the
  score card's iso-quality default. Anything still missing when the analysis runs is filed as a dated deviation.
- Cost: A6.5 ≈ $8 per System-1 family (+ ≈ 1 GPU-h for Intern-Decision); A6.8 ≈ $5–10 plus the measurement arm
  (Flash-Lite); A6.1–A6.4, A6.6, A6.7 and A6.9 are writing and code only.

### A6.11 — V when the calibration log holds no wrong System-2 decision (found 2026-10-01, before the freeze)

- **What was found.** The first registered calibration arm (`jev_shadow_cal`, Flash, 150 tasks, seeds 5000–5149,
  run 2026-10-01 21:00 UTC) has **no episode in which System-2 decided wrongly** (150/150 right on tool_select and
  tool_params), so the counterfactual stakes V = W (S_right − S_wrong) of A6.5 / H1 are undefined and the registered
  fit returns no τ̂\*. A6.5's fallbacks cover too few paired decisions, not an undefined V.
- **Rule (applies to every calibration arm, both families, both System-2 models).** When the counterfactual V of a
  calibration arm is undefined or not positive, V is set to the **conservative bound V = W** (at the primary operating
  point W = $1: every accepted wrong decision loses the task; on strict JevGym tasks an accepted wrong state-changing
  tool is exactly that, and the n = 50 pilot's cascade arms at τ ≤ 0.95 lost every episode in which a wrong tool was
  accepted). The sensitivity rows keep their own W (V = W each). The fit records `V_source` ("counterfactual" or
  "fallback W") and the fitted τ̂\* is used exactly as A6.5 says. When the counterfactual V exists it is used
  unchanged (the Flash-Lite arms are expected to have wrong System-2 decisions; the pilot reference is at 80 %).
- **Why the bound, not a smaller V.** V < W lowers τ̂\* (more delegation); with no wrong-decision episode to
  estimate the loss from, the bound is the choice that cannot overstate System-1's benefit, and it coincides with the
  JevGym design (irreversible wrong state changes). A V estimated from the share of state-changing tools among
  System-1's wrong calibration choices is a reported sensitivity row, never the deployed value.
- **Code.** `threshold_theory.fit_calibration_arm(v_fallback="W")` (CLI `--v-fallback W|none|<number>`), test in
  `tests/test_a6_metrics.py`. Lands under the A6.10 slip rule (before any C1 arm runs).
- Cost: none.
- **(b) No accepting confidence level ("never").** When the projection accepts no observed confidence level at the
  operating point (every bin's advantage a(c) = g(c) − q_L(c) + ΔC/V is negative), Prop. 1 predicts that no cascade
  is profitable for that System-2 model and decision point. The C1 arm of that model is then run with that decision
  point at **τ = 0.99**, the top of the registered sweep (A2), so that the prediction is testable, and C1 for that
  model is reported as *theory predicts no profitable threshold*: the registered NI test still runs; a pass would
  falsify the V = W bound of (a), and the arm's own accepted-but-wrong decisions then give the realised V, reported
  next to the bound. A fitted τ̂\* of exactly 1.00 is not this case: it accepts the decisions at confidence 1.0
  (System-1 certainty), which the harness compares with `>=`.
- **Fitted values (2026-10-01 21:05 UTC, `results/calibration/`, commit of this entry):** Flash — tool_select
  **1.00** (accepts the 336 / 1125 calibration decisions at confidence 1.0, 336 correct), tool_params **0.99**
  (940 / 1324, all correct); Flash-Lite — tool_select **never** → 0.99 by (b), tool_params **0.13** (all 1480 paired
  argument decisions: Jev and Flash-Lite both correct). V = W = $1 by (a) on all four fits (Flash: 150 / 150
  episodes right; Flash-Lite tool_select: the 13 wrong-decision episodes succeed as often as the rest; Flash-Lite
  tool_params: no wrong decision). ΔC: $0.0057 / $0.0056 (Flash), $0.0013 / $0.0013 (Flash-Lite). Written into
  `jev_ts_cascade_tauhat` (tool_select 1.00, tool_params 0.99) and `jev_ts_cascade_tau0.9_s2lite` (tool_select
  0.99, tool_params 0.13) in `configs/experiments/jevgym_tool_select.yaml`; the Intern-Decision C1 arms get theirs
  from `jevgym_shadow_cal_intern.yaml` on the GPU host (same rules).
- **(c) Revision after review round 6 (2026-10-01 late; `docs/REVIEW_ROUND6.md`), supersedes (a) and (b) above.**
  1. *What V = W is.* "When paired counterfactual continuations do not identify a positive stake for a decision
     point, the deployed fit uses the prespecified conservative assumption V = W. This is an upper bound on the
     success-value loss from one decision, not an estimate of that loss. Episode success conditional on whether a
     logged decision was wrong is descriptive and is not a counterfactual estimator; the fit records the estimator,
     the branch counts and the fallback reason (`V_source`, `n_wrong_episodes`)." The sentence "the bound is the
     choice that cannot overstate System-1's benefit" is withdrawn: holding the fitted accuracy functions and the
     avoidable call cost fixed, V = W weakly restricts acceptance relative to smaller positive stakes; it does not
     protect against estimation error in accuracy, cost attribution or trajectory distribution shift.
  2. *Diagnostics, not replacements.* Two stake diagnostics are reported next to every fit and never deployed: the
     share of System-1's wrong, executable calibration answers that act through an irreversible or final tool
     (`fatal_share`; strict JevGym: cancel / refund / ship / finish; Flash 56 / 67 = 0.84 on tool_select, 25 / 56 =
     0.45 on tool_params; Flash-Lite 68 / 81 = 0.84 on tool_select) and the resulting τ̂\* under V = W × that share
     (unchanged for all four fits: 1.00 / 0.99 / never / 0.13); and, from the n = 500 evaluation shadow logs, the
     oracle verdict (progress / detour / terminal-fatal / budget-fatal, A6.7's `fatal_kind`) of System-1's actual
     proposals where their arguments can be instantiated without gold, with the unidentified share stated. Paired
     counterfactual continuations of System-1 proposals are a registered sensitivity analysis under the A6.8 budget
     rule, never a replacement chosen after seeing which estimator delegates more.
  3. *The "reported-one set".* Audit (2026-10-01, `cache/s1.sqlite`, 22,740 Jev responses): the Jev wire reports
     `confidence` and `probabilities` at **two decimals** (4,950 of 9,809 choice answers carry confidence exactly
     1.0; none lies in (0.995, 1)); our full-precision `s1_dist` adds an ε = 10⁻⁶ floor over the non-finalists of
     the option tournament and is not the compared quantity. The harness compares the wire's two-decimal confidence
     with `tau` using `>=`; no clipping or rounding beyond the provider interface is introduced. The threshold 1.00
     therefore accepts the **reported-one set**: the decisions whose wire confidence rounds to 1.00 (≥ 0.995). It is
     not a set of certain predictions; its calibration accuracy (336 / 336 within 150 tasks), task-cluster
     uncertainty, coverage (30 % of calibration decisions) and the two-decimal score precision are reported. A6.5's
     two-decimal rounding of τ̂\* is withdrawn: thresholds are stored at full precision in `results/calibration/`
     and in the arm's `tau:` (for Jev the score itself has two decimals, so nothing changes; for Intern-Decision,
     an in-process float32 score, the full-precision value is deployed).
  4. *No accepting confidence level ("never").* (b) is withdrawn. If no observed confidence level has a non-negative
     fitted acceptance advantage, **the plug-in policy always defers at that decision point**, implemented by the
     explicit `always_defer: true` sentinel of the cascade mode (`harness/agent.py`: System-1 is not queried there,
     so it incurs no cost or latency; the decision is logged as deferred with no System-1 answer). The remaining
     decision points keep their own fitted policies. C1 evaluates this complete plug-in configuration under the
     registered NI procedure and reports its utility, savings and delegation coverage. A forced τ = 0.99
     configuration is a secondary diagnostic (the existing n = 200 sweep arms), never the C1 arm. Neither an NI pass
     nor a utility improvement alone falsifies V ≤ W. Wording: "the fitted model predicts no beneficial
     tool-selection acceptance on the observed calibration support at this operating point" (not "Proposition 1
     predicts that no cascade is profitable"). **C1L runs as registered** with tool_select always deferred and
     tool_params at 0.13; because the harness delegates arguments only inside System-1-chosen calls, this policy
     delegates nothing on Flash-Lite and the arm is behaviourally the Flash-Lite reference with a second seed of
     LLM noise: the paired comparison then estimates that noise, and an NI result with zero delegation is reported
     as *policy non-inferiority with zero coverage*, never as successful delegation.
  5. *ΔC for arguments.* Several argument decisions of one call share one avoidable System-2 call; the avoided cost
     is credited once per call and split over that call's argument decisions (`cost_terms(couple_by_step=True)`;
     1.67 slots per call on Flash, 1.65 on Flash-Lite: ΔC 0.0034 / 0.0008 instead of 0.0056 / 0.0013; τ̂\*
     unchanged). The deployed configuration's ledger is authoritative; this per-decision approximation is stated.
  7. *Deployed estimator, stated once.* The deployed τ̂\* comes from `threshold_theory.scalar_threshold_projection`:
     g(c) and q_L(c) are the empirical System-1 / System-2 accuracies per **equal-mass confidence bin** (10 bins of
     at least 20 paired decisions, smaller bins merged), the advantage a(c) = g(c) − q_L(c) + ΔC/V is evaluated
     per bin, the accept region is the set of bins with a ≥ 0, and the scalar projection is the smallest accepted
     bin's lower edge when the region is an upper tail (else the gain-maximising threshold, with the disagreement
     reported). A6.5's "isotonic (monotone) fits" are descriptive and are not what is deployed; no temperature
     scaling enters the deployed threshold (temperature scaling is a backend's own calibration, reported in E1 /
     H1). The calibration artefact (`results/calibration/*.json`) holds the backend and served model id, the
     calibration task ids, the score representation and precision, the full-precision thresholds, the tie rule
     (`>=`), the per-bin fitted maps, `V_source`, the cost estimator and the acceptance counts.
  6. *Operating point.* W = $1 per task success is a declared preference, not an economic truth: it values one
     success point at $0.01 per task, about a quarter of a Flash episode and a whole Flash-Lite episode. The paper
     shows the registered W ∈ {0.1, 1, 10}, λ ∈ {0, 0.5, 5} $/h grid with success, cost, critical-path latency, net
     utility, acceptance coverage per decision point, V = W versus any credible counterfactual estimate, and the
     no-delegation endpoint.

### A6.12 — Reporting a System-1 family that passes the static gate but is weaker in the harness (pre-committed 2026-10-01)

- **What was found** (GPU round 2, `results/gpu_round2/NOTES.md`, n = 10, pilot): Intern-Decision-4B passes the A5.1
  competence gate (0.862 vs Jev 0.971 on the static dev set) and matches Jev on BFCL tool selection (0.953 vs 0.965),
  but on the states the harness visits it is a weaker decider (shadow tool-selection accuracy 0.76 vs 0.93; replace
  episodes 1/10 vs 9/10; its errors are sequential: skipping a lookup step, acting where the policy says to ask) while
  being as good a guard as Jev (0.96 vs 0.97). A5.1's gate selects static accuracy, which is not the property the
  cascade / replace primaries need.
- **Rule.** The gate stays as registered (A5.1 is not edited). If a family passes the gate and then fails, or splits,
  an in-harness primary (C1, C2, C6), the paper reports the **static-vs-interactive gap** as a primary finding: the
  family's static accuracy (E1, BFCL / When2Call), its in-harness shadow accuracy on the same evaluation states as
  the other family (the n = 500 shadow arms), and the primary's outcome per family, in one table — the claim being
  that static tool-use benchmarks do not predict cascade / replace viability because in-harness errors compound. The
  gap is **not** a reason to demote the family after the fact: its C1 / C2 / C6 arms run as registered, its C1 arm at
  its own fitted τ̂\* from `jevgym_shadow_cal_intern.yaml` under the A6.5 / A6.11 rules (including "never" → 0.99),
  and the conjunction rule of A5.1 applies unchanged ("split" when one family passes).
- **Eligibility, not role competence (review round 6).** "The competence gate establishes eligibility on the
  specified static mixture only (tool selection, arguments and stopping pooled; strong stopping can offset weaker
  tool selection). Passing it does not establish in-harness competence at each decision point. Once promoted,
  Intern-Decision-4B remains co-primary regardless of subsequent pilot or evaluation performance; no role-specific
  demotion or checkpoint substitution is made using those outcomes."
- **Reporting a split.** "For C1 / C2 / C6, report each backend's effect estimate and interval, the conjunction
  p-value, and both same-state shadow accuracy and live-policy decision accuracy. A split result means that
  cross-backend replication was not established; distinguish evidence of loss from an inconclusive constituent NI
  test. Similar static accuracy with lower same-state in-harness accuracy is reported as a static-to-sequential
  competence gap, not as evidence that the weaker backend is representative of all open decision models." The
  same-state shadow comparison is the competence comparison; replace-mode accuracy mixes competence with the
  state distribution its own earlier errors induce.
- **Coverage.** "Each backend uses its own calibration fit under the same algorithm. An always-defer outcome does
  not demote that backend. An NI result with zero or negligible delegation is reported as policy non-inferiority,
  not evidence of successful delegation; coverage and net savings accompany every replication claim."
- **The pilot's guard rows** say that Intern-Decision's guard *accuracy* on mostly-correct proposals matches Jev's;
  they do not show that it detects wrong proposals: the registered natural / probe FAR and FRR do. The pilot's
  assist / cascade rows with OOM tasks are a systems failure, not a model-quality estimate (re-run in GPU round 3
  with the shared-weights fix; checkpoint hashes, temperatures and tournament behaviour frozen in the run's
  `experiment.yaml`).
- **Paper.** §6 states the rule; §7 carries the table when the case occurs; App. I (H5) notes that a local 4B model at
  batch 1 on one L4 (≈ 330 ms per decision measured in round 2) is slower than Jev over the network (≈ 200 ms), so the
  "faster" tier of H5 can fail for Intern-Decision for latency reasons alone, which is reported as such.
- Cost: writing only (the arms are registered already).

### A6.13 — Evaluation seeds moved to 200000–200499; frozen task list; reserved namespaces; C8 population (2026-10-01 late)

- **Why.** The Deviations entry of 2026-10-01 records that the n = 50 E2b pilot resume ran once on 100000–100049,
  which A6.4 had declared unexposed. Review round 6 (gpt-6-astra C1 / E1) calls this the highest-priority freeze
  defect: a partially exposed evaluation list is not repaired by discarding the exposed prefix. Audit of every
  JevGym task id logged under `runs/` on 2026-10-01 (script in A6.4's manner): seeds 0–149 (pilots, dev, simulated),
  5000–5149 (calibration), 10000–10029 and 12000–12059 (training), 100000–100049 (the exposed pilot pass); no seed
  ≥ 200000 has ever been run.
- **Rule.** The evaluation suite is **seeds 200000–200499** (`suite: {name: jevgym, n: 500, seed0: 200000, …}` in
  `jevgym_tool_select.yaml`, `jevgym_meta_control.yaml`, `jevgym_tool_select_intern.yaml`, `baselines_jevgym.yaml`).
  The complete list of its 500 task ids is frozen in `configs/eval/jevgym_eval_tasks.json` (250 `clarify` + 250
  `multi`; sha256 `81bc6c3124531a6c5f7c21feb1335d4545c236915725706ad33919bb62dcec44`); `scripts/prereg_check.py`
  (run by the test suite) fails when the configs stop reproducing it. **100000–100499 is retired**: reserved,
  never evaluated, never trained on, never run again. "Evaluation ids are reserved explicitly and excluded from
  every training, calibration and development loader, regardless of numeric range": `bench/registry.py` holds the
  ranges; `training/harvest.py` excludes `200000-200499,100000-100499` by default; the runner **refuses** any
  jevgym suite touching an evaluation or retired range unless the environment carries `JEV_EVAL_RUNS=1`, which
  the main runs set after the freeze and nothing else does (no pilot, smoke test or resume can expose the suite by
  accident again; `tests/test_registry.py`). A6.4's sentence "no seed ≥ 100000 was run before 2026-10-03" is
  replaced by this entry; the exposure manifest above gains "100000–100049: one n = 50 E2b pilot pass, 2026-10-01,
  retired with its range".
- **C8 population.** The 299 E1 items of the dev prefix (`configs/dev/e1_dev.yaml`, `--max-items 300`: BFCL and
  When2Call items used in dev rounds 2–3 and GPU round 2 for the Jev / Intern-Decision comparison) are listed in
  `configs/eval/e1_dev_prefix_ids.json` (sha256 `0276eacb50debf27…`) and **excluded from the C8 confirmatory
  population** (`exclude_ids_file` in `configs/decision/bfcl_when2call.yaml`); C8's sizes are the remaining items
  of the registered E1 pools (the gate's 210 dev decisions are training-side JevGym seeds, disjoint by range).
- Cost: writing and configuration; the n = 50 pilot results on 100000–100049 stay in the notes as a retired pilot.

### A6.14 — The NI decision rule with demonstrated size (review round 6, gpt-6-astra C4 / E2; 2026-10-01 late)

- **What was found.** A6.1's own simulation reported a type-I error of 0.013 at the Holm floor 0.05/11 ≈ 0.0045 and
  called the rule nominally controlled; a reviewer pointed out that this is a threefold inflation and that the
  "< 5 discordant tasks" fallback does not reach the problematic region (losses-only data with ≥ 5 losses). A
  dedicated simulation (`scripts/ni_size_sim.py` → `results/reviews/2026-10-01c/ni_size_sim.md`; 50,000 replicates per
  size cell, 20,000 per power cell, B = 2,000 with a B = 10,000 check, seed 20261001; the bootstrap rule validated
  against `metrics.ni_pvalue` on per-task records, decision agreement ≥ 99.5 %) confirms it: at the boundary Δ = −m the
  registered bootstrap rule rejects 0.066 / 0.016 / 0.0105 at α = 0.05 / 0.05/6 / 0.05/11 on losses-only data
  (1.3× / 1.9× / 2.3× α), 0.054 / 0.011 / 0.0067 at p_ref 0.98 with 5 % discordance, and 0.054 / 0.0095 / 0.0058 on the
  τ² case (n = 149, m = 5 pt, 20 % discordance); it holds its size only at 20–30 % discordance (p_ref 0.80). The
  fallback fired in 0.07 % of the losses-only replicates.
- **Rule (replaces A6.1's decision rule from the freeze on; the statistic of §2 and A6.1 becomes descriptive).** For
  every NI primary the decision statistic is **Tango's (1998) one-sided score test** of H0: Δ ≤ −m on the paired
  counts (wins, losses, n): p_T = 1 − Φ(Z(−m)), Z the score statistic with the restricted MLE of the discordant
  probability under H0 (`metrics._tango_z`); the test rejects iff p_T ≤ α_adj of its Holm step, equivalently iff the
  Tango one-sided (1 − α_adj) lower bound (`metrics.tango_interval(..., one_sided=True)`) lies above −m. **Switch:**
  whenever the arm has **zero wins** or fewer than 5 discordant tasks, the exact one-sided binomial test of
  H0: P(loss) ≥ m, p = P[L ≤ l | n, m] (`metrics.ni_exact_check`), replaces p_T and decides. The p that enters Holm is
  this decisive p (`ni_pvalue()["p_decisive"]`, `["rule"]` ∈ {tango, exact}). Zero discordance is handled as before
  (p = (1 − m)^n; the one-sided (1 − α) upper bound on the loss rate replaces a zero-width interval). The paired task
  bootstrap (p_NI, the (1 − α_adj) lower bound, the 95 % CI) is **reported as descriptive** next to every primary and
  no longer decides; superiority primaries (C3, C4, C5) keep p_sup from the bootstrap (not affected by the finding,
  which concerns the margin test near a ceiling). The two-family conjunction (A6.1, max of the two decisive p) and
  the "inconclusive at this n" rule (A6.1) are unchanged.
- **Why this rule.** In the simulation the exact loss-count test everywhere (reviewer option 2) holds its size but has
  no power once wins occur (0.000 at 20 % discordance); the Tango test alone holds its size in every cell (worst
  1.03× α, within Monte-Carlo error) and matches the bootstrap's power where the bootstrap is valid (20–30 %
  discordance: 0.545 / 0.293 / 0.238 vs 0.556 / 0.299 / 0.236 at the three levels, mean over Δ ∈ {0, +2 pt}); the
  switch to the exact test on losses-only data is more powerful than Tango there (0.125 vs 0.064 at 0.05/6) and
  reproduces A6.1's zero-discordance text. Where the bootstrap was inflated the new rule is less powerful (e.g.
  0.536 vs 0.621 at p_ref 0.98, 5 % discordance, Δ = 0, Holm floor): that is the removed inflation, not a loss.
- **τ² (C7).** The rule applies to a per-task binary outcome. C7's per-task outcome for the decision is **majority
  pass** (≥ 2 of the 3 trials pass^1) for the arm and the reference; the 3-trial pass^1 mean and its bootstrap CI stay
  the descriptive effect. Margin 5 pt as registered (A5.4 sizes).
- **Power at the registered sizes** is restated for this rule (`scripts/ni_power_table.py` →
  `results/reviews/2026-10-01c/ni_power_table.md`; the paper's App. G table is regenerated from it) and the blanket
  "well powered for the Flash pairs" wording is withdrawn: at n = 500 a persistent 2-point loss against a 3-point
  margin is not reliably certified at the Holm floor; such primaries are expected to end *inconclusive at this n*.
- **Power restated for this rule** (20,000 replicates per cell, `results/reviews/2026-10-01c/ni_power_table.md`; the
  A6.1 table is superseded):

  | primary (reference success) | n | m | d | true Δ | α = 0.05 | Holm floor 0.05/11 |
  |---|---|---|---|---|---|---|
  | C1F / C2F / C6F (Flash, 0.98) | 500 | 3 pt | 2 % | −1 pt | 0.86 | 0.39 |
  | | | | 5 % | 0 | 0.89 | 0.54 |
  | | | | 10 % | 0 | 0.66 | 0.28 |
  | | | | 10 % | −1 pt | 0.40 | 0.10 |
  | ceiling reference, losses only (1.00) | 500 | 3 pt | 2 % | −2 pt | 0.33 | 0.07 |
  | | | | 1 % | −1 pt | 0.94 | 0.62 |
  | C1L / C2L / C6L (Flash-Lite, 0.80) | 500 | 3 pt | 20 % | 0 | 0.44 | 0.13 |
  | | | | 30 % | 0 | 0.34 | 0.08 |
  | | | | 20 % | +3 pt | 0.91 | 0.64 |
  | | | | 30 % | +5 pt | 0.95 | 0.74 |
  | C7 (τ², majority pass, 0.80) | 149 | 5 pt | 20 % | 0 | 0.37 | 0.09 |
  | | | | 30 % | +3 pt | 0.56 | 0.19 |

  Reading: the Flash pairs are well powered only for a true loss ≤ 1 pt at ≤ 5 % discordance; a persistent 2-pt loss
  (ceiling reference) is certified with power 0.07 at the Holm floor, i.e. such a primary ends *inconclusive at
  this n*; the Flash-Lite and τ² pairs at Δ = 0 are underpowered as A6.1 already said.
- **Code** (lands under A6.10): `metrics.ni_pvalue` (p_tango, tango_lcb_adj, rule, p_decisive; ni_boot_adj kept as
  the descriptive pre-A6.14 decision), the score card's NI table (p_T (rule), Tango LB, p_boot), tests.
- Cost: none (CPU simulation, ≈ 20 min).

### A6.15 — Tidy-up from the generated effective manifest (2026-10-01 late; `docs/PREREG_EFFECTIVE.md`)

`scripts/prereg_effective.py` renders the effective registration from the configs, registries and fits (review round 6,
E7.6) and reported the following gaps between the text and the machine-readable sources; each is settled here.

- **C3 size and pairing.** C3 (Jev tool_select assist k = 3 vs BM25 assist k = 3, superiority) is tested on the
  **E2a arm `jev_ts_assist_k3` (n = 500, A2 sign-off)** paired with the **E2c control `bm25_ts_assist_k3`, now at
  n = 500** (`baselines_jevgym.yaml`; both run on the frozen task list, so the pairing crosses the two experiment
  files); E2c's own `jev_ts_assist_k3` (n = 200) is the run-local reference for the other E2c controls. Cost ≈ $10.
- **C4 live arm.** The registration named no routing τ. C4 is **`jev_route_tau0.4` vs `small_only`, both at
  n = 500** in `jevgym_meta_control.yaml` (the A5.5 opportunity rule and A6.3 unchanged). τ = 0.4 is fixed here,
  before the main runs, from the retired-seed n = 50 pilot (`results/pilot_jev/JEV_PILOT_NOTES.md`: 0.90 success at
  −70 % cost vs 0.94 at −33 % for 0.2), never from evaluation data; the sweep keeps 0.2 / 0.6 / 0.8 at n = 200.
  Cost ≈ $12 (the two arms from 200 to 500).
- **C5 lock.** The C5 pair `jev_reflect_trigger_s2lite` / `ratem_reflect_trigger_s2lite` (A4; n = 500 in
  `baselines_jevgym.yaml`) is added to `scripts/prereg_check.py`'s registry so a drift fails the suite.
- **τ² System-2 (C7).** Not named before. Registered: **agent = Gemini 3.5 Flash** (`gemini/gemini-3.5-flash`,
  temperature 0, minimal thinking), **user simulator = Gemini 3.5 Flash-Lite** (same settings), the small model of
  the routing arm = Flash-Lite; `scripts/run_tau2.sh` now defaults to exactly this (the earlier default `gpt-4.1` was
  tau2-bench's own and was never a registered choice).
- **Forced-threshold diagnostic (A6.11 (c).4).** Exists for Flash only (the n = 200 cascade τ sweep); no Flash-Lite
  Jev τ-sweep arm exists and none is added, so the C1L diagnostic is "no diagnostic arm".
- **Superseded references.** Wherever earlier text (A6.12, the first Deviations entry) says "never → 0.99 (A6.11
  (b))", the effective reading is A6.11 (c).4: always-defer. The manifest states the effective reading.
- **Slip-rule code still outstanding** (A6.10: before the analysis that uses it): `h5_verdict`'s two-tier naming
  (`strictly_better`, A6.6) and `meta_gold.audit_implications` (A6.7; `implication_audit` exists under that name and
  is the implementation). Listed in the manifest's "Open before the freeze".
- **Not changed:** the fitted `tool_select` calibration artefacts were regenerated with the current code so they
  carry `fatal_share` (values as quoted in A6.11 (c).2); the Intern-Decision fits and the gate verdict file land with
  GPU round 3.
- Cost: ≈ $22 of additional System-2 API (C3, C4 sizes); writing otherwise.

### A6.16 — Review round 7 (re-score of the plan as frozen; `docs/REVIEW_ROUND7.md`), 2026-10-01 night

Round 7 (gpt-6-astra, gemini-3.1-pro) re-scored the plan after A6.11(c)–A6.15. Gemini found no blocker beyond the
pending Intern-Decision fits; astra listed five (B1–B5). Each is settled here; nothing widens the experimental grid.

- **(a) One current protocol.** The paper (§6, App. A, App. G, Tables 3 / 5) and `docs/PREREG_EFFECTIVE.md` describe
  the effective rules of A6.11(c)–A6.16, not superseded ones: the decisive NI p-value is A6.14's (bootstrap
  descriptive); no-acceptance means always-defer, never τ = 0.99; the deployed estimator is the tie-preserving binned
  one of (c) below; C7's endpoint is named as in (e). `scripts/prereg_effective.py` is regenerated by the test suite.
- **(b) No silent placeholders.** A decision point whose τ̂\* is not fitted carries `calibration: pending` in its
  config (the two Intern-Decision C1 arms) and the runner refuses to run it (`bench/runner.py`;
  `JEV_ALLOW_PENDING_TAU=1` is a smoke-test override, never used for a main run); `scripts/prereg_check.py` fails if
  such an arm carries a bare number. The competence-gate verdict is committed as
  `results/gpu_round2/gate_verdict.json` (4B: 0.862 vs Jev 0.971, ratio 0.888 ≥ 0.8 → co-primary; 2B 0.657, 0.8B
  0.588 fail). GPU prices for in-process backends stay an assumed amortised figure, labelled as such, until
  `scripts/measure_latency.py` runs on the training host; measured and assumed prices are never mixed in one column.
- **(c) Confidence ties and the deployed estimator** (A6.11 (c).7 refined). "Identical deployed confidence values
  are never split across bins. Approximate equal-mass bins are formed from whole tie groups (every bin edge is an
  observed value); the number of bins may therefore be below ten. Reported acceptance counts are recomputed with the
  deployed `>=` comparator." Implemented in `threshold_theory.equal_mass_bins`; the four Jev fits refitted with it
  are unchanged (tool_select 1.00 / never, tool_params 0.99 / 0.13; 8 / 7 / 5 / 4 bins instead of 10).
- **(d) Cost coupling, stated as an approximation.** "Per-decision cost allocations are approximations to the joint
  tool-and-argument policy. We do not sum their predicted savings as independently avoidable calls; deployment
  savings are measured from the complete configuration's ledger."
- **(e) C7's endpoint, named.** "C7 tests non-inferiority, with a five-percentage-point margin, in the probability
  that at least two of three registered attempts succeed (*majority pass*). Mean per-attempt success (pass¹) and its
  paired task-bootstrap interval are separate descriptive endpoints. Majority-pass NI does not establish pass¹ NI."
  The margin is kept at 5 pt as the tolerance on majority pass (A5.4's sizes were planned for a 5-pt margin on a
  per-task binary outcome, which majority pass is). After a failed NI test the wording is "transfer NI was not
  established; the effect estimate and interval distinguish an inconclusive comparison from evidence of a success
  loss", never "transfer failed".
- **(f) A6.14's validity claim, scoped.** "The switching rule showed no detected size inflation in the simulated
  scenarios and levels. It is an approximate score-based procedure, not a finite-sample exact NI test; its exact
  component tests the stronger loss-probability condition P(loss) ≥ m." Before the tag its rejection region is
  evaluated by exact trinomial summation over a dense grid of win probabilities (including w → 0) at all eleven
  Holm levels for n = 500 / m = 3 pt, n = 149 / m = 5 pt and the C8 sizes (`scripts/ni_exact_size.py` →
  `results/reviews/2026-10-01d/ni_exact_size.md`); the result is recorded in the Deviations section with its date,
  and if the exact size exceeds the level anywhere by more than 2 % relative, the exact loss-count test is used at
  those (n, level) cells (a registered contingency, not a new rule).
- **(g) Superiority primaries.** C3 and C4 (binary endpoints) are decided by the exact conditional test on discordant
  pairs, p = P[W ≥ w | W + L = d, ½] (`metrics.sign_test_exact`, `p_wins`); C5 (continuous, paired ΔU) by the
  one-sided sign-flip permutation test of the mean difference (`metrics.sign_flip_test`: exact under the sharp null
  whatever the skew or the share of zero differences; 20,000 sign draws, seed 0, exact enumeration for n ≤ 16); the
  task-bootstrap p_sup and CI stay descriptive. A6.1's "superiority primaries use p_sup" is superseded.
- **(h) H1's estimand.** "The shadow-log analysis estimates priced one-decision surrogate regret on states visited
  by LLM-only. It does not estimate counterfactual episode utility. C1 separately evaluates the deployed policy. Live
  sweep comparisons use common tasks and explicitly stated argument policies: the τ sweep arms fix the argument
  threshold at 0.90, whereas the C1 arms use the fitted argument thresholds, so they are different complete
  policies, not merely different tool-selection thresholds."
- **(i) H2's inference.** The theory check is the equation-based predicted-vs-observed comparison (A6.2's Eq. 2
  readout) with shared task ids resampled jointly across cells (the derived predictors and effects recomputed in
  every resample); the pooled regression of ΔU on J and accuracy is exploratory. Minimum requirements: ≥ 4 estimable
  cells and full column rank for any regression; otherwise H2 is reported as *not estimable*, never falsified. Gross
  ΔU (without System-1 query cost) is what Eq. 2 predicts; net ΔU (with it) is reported next to it. The matched-rate
  control's firing rate for C5 is **fixed before the main runs from the retired-seed n = 50 pilot's live arm**
  (`jev_reflect_trigger_s2lite_e2b`, `results/pilot_jev/JEV_PILOT_NOTES.md`; development data), the realised rates
  of both arms are reported, and inference treats the rate as fixed; the baselines config's `ratem_*` rates are set
  from that pilot, not from the evaluation run (supersedes the A4 header text that reads them from the live arm).
- **(j) H5, bounded.** H5's "success-preserving" condition is an **exploratory screen** (the 90 % task-bootstrap
  lower bound with wins ≥ losses on ≈ 20–37 tasks is not a size-controlled NI test: with a ceiling reference and
  true loss probability 0.05, zero observed losses occurs with probability ≈ 0.36 at n = 20), labelled "passes the
  exploratory success-preservation screen", with no formal NI interpretation. "Faster" is "faster on co-solved
  tasks", decided by the exact sign test on the per-task time differences (A5.2's Wilcoxon is dropped: it is not a
  distribution-free test of the median without symmetry); all-task capped time and solved-by-time curves are shown
  next to it. What is frozen now: the criterion, the 37-task hard pool and its deterministic selection rule, the
  model ids (Gemini 3.5 Flash agent), the time caps of the adapters; what is logged per run: backend hardware,
  concurrency, queueing, retries, tool-execution time; arm execution order is interleaved per task. Only the
  criterion and the pool are registered at this tag; the execution details above are logged, not pre-set.
- **(k) C4 and the pilot.** "C4's threshold was selected using the disclosed retired development pilot and is fixed
  before fresh evaluation. C4 tests the resulting selected policy, not a threshold chosen without development
  evidence." The first Deviations entry's sentence "no prompt, label, arm, size or test was changed on this evidence"
  holds as of its time stamp; A6.15 later used that pilot (as development data) to fix C4's τ and A6.16 (i) to fix
  C5's control rate.
- **(l) Evaluation-list integrity.** "The runner rejects reserved evaluation ids unless explicitly authorised for a
  registered main run; the authorisation, the configuration hash, the task-list hash and the run identity are
  logged." The frozen artefact hashes the task-generating configuration and the generator's code revision
  (`configs/eval/jevgym_eval_tasks.json`: suite block + `git` commit of `bench/suites/jevgym.py` at the tag), not
  only the ids. C8's population is frozen as the unique source items (`configs/eval/e1_dev_prefix_ids.json` lists
  the excluded decision records; exclusion is applied by parent item, and the remaining per-category item counts are
  written to `configs/eval/e1_c8_population.json` before the C8 run).
- **(m) A6.12, one sentence softened.** "Similar static accuracy need not imply similar in-harness competence or
  deployment performance. We report same-state interactive accuracy separately from live-policy outcomes; these
  comparisons do not by themselves identify error compounding as the cause." Replication claims with negligible
  coverage in either constituent are "cross-backend replication of policy NI".
- **(n) Scope of the two-family claim.** The two System-2 models are two strengths of one family (Gemini 3.5 Flash,
  Flash-Lite); cross-family generalisation is untested unless a non-Google family runs (A3, A6.9 #15); the paper's
  §1 says so.
- Cost: writing and CPU only; the C5 control rate change saves the placeholder-rate re-run.

### A6.17 — Exact size of the NI rule and the decisive p-value of its score branch (2026-10-01, night; before the tag)

- **(a) Result of A6.16 (f)'s validation** (`scripts/ni_exact_size.py` → `results/reviews/2026-10-01d/ni_exact_size.md`;
  exact summation over the paired trinomial counts, 20 win probabilities × the eleven Holm levels (+ 0.10) ×
  n ∈ {500, 149, 300, 1000}): the A6.14 rule's size at the boundary Δ = −m is ≤ 1.009 α at n = 500, ≤ 1.011 α at 300 and
  ≤ 1.004 α at 1000 at every Holm level, and ≤ 1.023 α at n = 149, where two cells exceed A6.16 (f)'s 2 % threshold
  (0.05/11 at w = 0.25: 1.020 α; 0.05/2 at w = 0.15: 1.023 α). The excess is the score component's finite-sample
  discreteness at 30–40 % discordance; the exact fallback never adds size and repairs the score test's genuine
  over-rejection on losses-only data (up to 1.13 α at α = 0.05).
- **(b) The A6.16 (f) contingency is withdrawn before use** and replaced by an exact p-value. Applied literally through
  Holm, "the exact loss-count test at those (n, level) cells" would make C7's rejection at the 0.05/2 step depend on a
  test whose critical loss count is 2 of 149, a power collapse unrelated to a 2.3 % relative inflation. Instead the
  decisive p-value of the score branch is its **exact unconditional p-value with the Berger–Boos confidence-set
  restriction** (Suissa & Shuster 1985; Berger & Boos 1994):

      p_U = max_{w ∈ C_γ} P_w[ Z(−m) ≥ z_obs, W ≥ 1, W + L ≥ 5 ] + γ,

  where w is the per-task win probability at the boundary (P(loss) = w + m, P(tie) = 1 − 2w − m), C_γ the (1 − γ)
  Clopper–Pearson set for w from the observed discordant count W + L, γ = 10⁻⁴, the maximum over the grid
  {0, 0.0005, 0.001, 0.0025, 0.005, …, (1 − m)/2} ∩ C_γ plus the set's end points and the point estimate, and the tail
  restricted to the region of the sample space on which the score branch decides (`metrics.ni_unconditional_p`;
  `ni_pvalue` reports `p_uncond` and `rule = tango_uncond`; the asymptotic p_T stays reported). The exact loss-count
  branch (W = 0 or W + L < 5) is unchanged. p_U is a valid p-value at every n and level (Berger & Boos 1994, Thm. 1), so
  the two-branch rule needs no cell-wise contingency; by the same exact summation (`registered_u` in
  `ni_exact_size.md`) its size is ≤ α at every (n, w, level) of the table (maximum 0.998 α) and its rejection
  probability near the boundary is ≥ 0.96 of the A6.14 rule's at the Holm floor and ≥ 0.97 at α = 0.05 (the price of
  exactness). The table is re-run at the final retained n′ before analysis (one tuple added to `SETTINGS`); a residual
  above α at an off-grid w would be reported, not acted upon.
- **(c) Wording.** "The switching rule showed no detected size inflation in the simulated scenarios and levels" (A6.16 (f))
  is replaced by: "The decisive p-value of the score branch is the Berger–Boos exact unconditional p-value, valid at
  every sample size and level; by exact summation over the paired trinomial counts the two-branch rule's size is at most
  0.998 α on a dense nuisance grid at the registered sizes and all eleven Holm levels, and the asymptotic score p-value
  it replaces reached 1.023 α." "Its exact component tests the stronger loss-probability condition" stays. The power
  table (App. G) is recomputed exactly for the A6.17 rule (`scripts/ni_power_exact.py`) before the tag.
- Cost: CPU only.

### A6.18 — Pre-tag consistency pass (2026-10-01, late night; names what it supersedes)

An independent read of A1–A6.17 against the manifest, the configs and the paper (`docs/REVIEW_ROUND7.md` §3 lists
the reader) found stale sentences that later amendments had overridden without saying so. Each is settled here; no
test, size, margin or arm changes.

- **(a) A6.14's size sentence.** "the Tango test alone holds its size in every cell (worst 1.03× α, within
  Monte-Carlo error)" is superseded by the exact table (A6.17 (a)): Tango alone reaches 1.133 α at n = 149, α = 0.05
  on losses-only data and 1.023 α at a Holm level; the registered switch repairs the former and A6.17's p_U the latter.
  A6.14's simulated power table is superseded by the exact one (`results/reviews/2026-10-01d/ni_power_exact.md`,
  the paper's App. G); A6.14's "superiority primaries keep p_sup from the bootstrap" is superseded by A6.16 (g).
- **(b) "Transfer failed".** A5.4 (d) and A6.9 #11's "the paper reports transfer failed" read "the paper reports that
  transfer NI was not established" (A6.16 (e)).
- **(c) C7's selection rule and the `noninferior` label.** A6.9 #11's "NI on E2 by the unadjusted one-sided 95 % bound"
  and A6.9 #10's `noninferior` (iso-quality) both mean: the registered decision statistic at the unadjusted level, i.e.
  the A6.17 rule's p (p_U, or the exact loss-count p when it applies) ≤ 0.05 at the registered margin
  (`metrics.ni_pvalue(...)["ni_95"]`; `metrics.iso_quality_savings(criterion="noninferior")` is implemented so). The
  bootstrap and Tango lower bounds printed next to it are descriptive.
- **(d) C8's size.** The C8 primary population is the 1,114 BFCL tool_select items of
  `configs/eval/e1_c8_population.json` (secondary rows: 1,949 argument slots, 928 When2Call items); A6.17's exact-size
  table gains the row n = 1,114, m = 3 pt (`SETTINGS`), re-run at the final retained count as A6.17 (b) says.
- **(e) Gate verdict file and numbers.** A5.1's `runs/gate_intern_decision.json` is superseded by the committed
  `results/gpu_round2/gate_verdict.json` (A6.16 (b)). Its numbers, labelled: accuracies on the 210 dev decisions Jev
  0.971, Intern-Decision-4B 0.862, 2B 0.638, 0.8B 0.571; ratios to Jev 0.887, 0.657, 0.588 (gate at 0.8); the paper
  and A6.16 (b) use these labels. A5.1's paired per-item bootstrap of the ratio is replaced by the stratified binomial
  bootstrap interval from the per-decision-point counts (descriptive; Deviations), the per-item records of that run not
  having been kept; the gate decides on the point estimate as A5.1 says.
- **(f) Scope of A6.16 (i).** Only the C5 control (`s1_backends.ratem_lite`) is fixed from the retired pilot. The other
  matched-rate controls (`s1_backends.ratem_stop_lite`, `.ratem_route`, `.ratem_memw`) are secondary rows and keep A4's rule (rate
  set from this run's System-1 arm's realised rate before the control arm runs, realised rates reported); `jev_memw`
  never ran in the pilot, so no pilot rate exists for it.
- **(g) H5's Wilcoxon.** A6.16 (j)'s "dropped" means dropped as the criterion; the Wilcoxon p is printed as
  descriptive next to the sign test (as the paper, the manifest and `h5_verdict` already do).
- **(h) Labels and tags.** C4's System-2 tag in the paper's Table 3 is "L→F" (routing from Flash-Lite to Flash;
  reference small-only = Flash-Lite), not "F" (A5.3's "C4 ... unchanged (Flash)" refers to the large model it routes
  to). The confirmation label is *Jev-specific confirmation* (A6.9 #6) everywhere. A6.3's label is *insufficient
  observed opportunities*, never "no opportunity". A6.11 (c)'s items are numbered 1–5, 7, 6 in the order written;
  cross-references use those numbers. A5.1's `decision.backends` means `jevharness.decision.make_backend`; A6.7's
  `audit_implications` is implemented as `meta_gold.implication_audit` (A6.15). A5.5's "pilot base rate (3.5 % of
  steps)" is the dev round-3 routing-positive rate 8 / 220 = 0.035 of `results/dev_round2/R3_DEV_NOTES.md`.
- **(i) H3 / C7 cross-family wording.** The paper's "across System-2 families" reads "across the two System-2 strengths
  (cross-family generalisation untested unless a non-Google family runs, A6.16 (n))".
- **(j) Deviations.** The section sits after §5 and before A4 (structure stated in the header); its entries are facts
  (A6.11 (c).4's deployed value, A6.16 (i)/(k)'s use of the pilot, the C8 counts and the gate interval are added as
  dated lines).
- Cost: writing and CPU only.

### A6.19 — The deployed threshold when the accept region is not an upper tail (2026-10-02, before the Intern-Decision fits)

The theory pass of 2026-10-02 (App. A proofs, `scripts/check_propositions.py`) found that A6.11 (c).7 and the code
disagree in one case. Settled here, **before the Intern-Decision calibration logs (seeds 5000–5149, then running on the
GPU host) were fitted or inspected**; the four Jev fits are unaffected.

- **(a) Rule.** A6.11 (c).7's parenthetical "(else the gain-maximising threshold, with the disagreement reported)" and
  A6.5's "Threshold form" sentence are replaced by the rule `threshold_theory.scalar_threshold_projection` implements
  and that produced all four registered Jev fits: τ̂\* is the lower edge of the **accepting upper tail** (the lowest
  confidence bin from which every higher bin has a(c) ≥ 0); bins below that tail which accept in isolation are
  reported as *islands*, with the pointwise rule over them as the secondary row; `tau_gain` (the gain-maximising bin
  edge) and the threshold-vs-pointwise disagreement fraction are reported next to it; no accepting tail means
  always-defer (A6.11 (c).4). When the accept region is an upper tail the two rules coincide (App. A), which is the
  case for all four Jev fits.
- **(b) Why.** The gain-maximising edge over noisy equal-mass bins can deploy a threshold inside an island, i.e. accept
  confidence levels whose higher neighbours all reject; the tail edge never accepts a bin above which System-1 is
  worse than the plug-in rule requires.
- **(c) Two stated approximations** of the deployed estimator: per bin, g is computed over System-1's executable
  (non-*Other*) answers while q_L is computed over all paired decisions, including *Other* states the cascade defers
  anyway (exact at tool_select; immaterial where q_L = 1, as in every Jev tool_params bin); and V is the descriptive
  episode contrast W·(S_right − S_wrong), not a counterfactual, with V = W when that contrast is undefined or not
  positive (A6.11 (c).1).
- **(d) H1 grid.** A6.5's swept grid {0.5, 0.7, 0.8, 0.9, 0.95, 0.99} is the CLI's `--taus h1`; the held-out regret
  table also reports the deployed estimator's own row (`tau_hat_projection`).
- Cost: none.

### A6.20 — τ² and C7 execution rulings from the analysis dry run (2026-10-02, before any τ² main-run data)

A simulated end-to-end dry run of the registered analysis (`scripts/dryrun_main_analysis.py`, all 147 registered
arms; `scripts/primaries.py`, the family analysis) exposed execution gaps and readings the registration leaves open.
They are settled here before any τ² main run exists; no test, size, margin or arm changes.

- **(a) τ² always-defer.** A6.11 (c).4 applies on τ²: a decision point whose airline refit accepts no confidence level
  runs as always-defer (System-1 not queried, deferred records kept). `JevTau2Agent` now implements it for tool_select
  and tool_params (`tests/test_tau2_adapter.py`).
- **(b) τ² thresholds per decision point.** τ̂\* is refit separately for tool_select and tool_params from the 15-task
  airline shadow log under the A6.11 (c) / A6.19 rules and passed per decision point (`-o tool_select.tau=…
  -o tool_params.tau=…`, `always_defer: true` where the fit says never); `scripts/run_tau2.sh`'s single-τ shortcut is
  not used for the C7 arms (`docs/MAIN_RUN_PLAN.md` gives the commands).
- **(c) τ² System-1.** Every τ² arm uses the same Jev endpoint as E2 (`{type: jev, provider: openrouter}`, the model
  pinned in §1); `configs/tau2/*.yaml` are set so.
- **(d) C7's eligible set (A6.9 #11).** "Policy GUARD" is E2a's `jev_ts_guard` (System-1 vets the System-2's proposed
  tool call before it executes; on τ² this is the policy-guard arm, configs/tau2/jev_guard.yaml). LLM-only is the reference and is never
  the selected configuration: if no System-1 configuration is non-inferior on E2, or LLM-only would otherwise be the
  most cost-saving candidate, the C1 arm is tested (A6.9 #11's no-NI rule). The selection is computed by
  `scripts/primaries.py --select-c7` and written to `c7_selection.json` before the first τ² main run.
- **(e) "Negligible coverage" (A6.16 (m)).** Coverage = the share of a decision point's opportunities on which
  System-1's answer was executed; it is negligible when below 5 % in an arm. The label then reads "cross-backend
  replication of policy NI", and the coverage of both constituents is printed with it.
- **(f) Slip-rule code (A5.6 / A6.10).** Still to land before the run or analysis it serves, never after: τ² and E1
  attempt counts and the automatic re-run of infrastructure-errored τ² simulations (the drop rule's "two extra
  attempts"); the C7 proxy check of A5.4 (e) (Kendall τ with a permutation p); C8's secondary rows (argument slots,
  When2Call, Holm within the L1 table).
- **(g) Analysis entry point.** `scripts/primaries.py` is the registered family analysis (arms, references and margins
  read from the effective manifest, never retyped); paper tables use the score card's `noninferior` criterion
  (A6.18 (c)); smoke runs write to a separate output directory, and the family analysis refuses episodes on any seed
  outside a main run's registered range.
- Cost: none.

### A6.21 — A certified NI p-value, C5's null, and the power table (review round 8; 2026-10-02, before the tag)

Review round 8 (`results/reviews/2026-10-02/gpt-6-astra.md`, W1, W2, W8) found that A6.17 does not establish what it
claims: a maximum over grid points is a lower approximation of the Berger–Boos supremum, not an upper bound; the
switch to the loss-count test makes the rule a data-dependent union of two tests; and only the boundary of the null
was evaluated. It also found that A6.16 (g) calls C5's sign-flip test exact for a null it does not test, and that the
power table contains scenarios that are impossible at the stated reference success while the text calls n = 500
adequate. Each is settled here. The decisive test of the eight NI primaries changes, and so does C5's (b); no arm, size, margin, seed, level or member of the Holm family changes.

- **(a) The NI decision rule (replaces the decisive p-values of A6.14 and A6.17).** For an NI primary with margin m on
  n paired binary outcomes, W wins (arm succeeds, reference fails), L losses and D = W + L discordant tasks:
  1. *One ordering of every outcome.* s(W, L) is the log of the nominal p-value of the A6.14 switching rule:
     log(1 − Φ(Z(−m))), Tango's one-sided score statistic, when W ≥ 1 and D ≥ 5, and log P[Bin(n, m) ≤ L] otherwise.
     The ordering statistic is its monotone hull s̃(W, L) = min{s(W′, L′) : W′ ≤ W, L′ ≥ L, W′ + L′ ≤ n}
     (`metrics.ni_order_table`). It ranks the outcomes the A6.14 switch sent to the loss-count test together with all
     others, so there is one test, not a union of two.
  2. *p-value.* p = min{1, γ + U}, where U is an upper bound on sup over w ∈ C_γ(D) of P_w[s̃ ≤ s̃_obs + 10⁻⁹]. P_w is
     the boundary trinomial (P(win) = w, P(loss) = w + m, P(tie) = 1 − 2w − m). C_γ(D) is the two-sided (1 − γ)
     Clopper–Pearson interval for the discordance rate, mapped by w = (ρ − m)/2, clipped to [0, (1 − m)/2] and
     widened by 10⁻⁹; γ = 10⁻⁴ (`metrics.ni_bb_nuisance_set`). The primary rejects iff p ≤ its Holm level.
     `metrics.ni_pvalue` returns p as `p_decisive` with `rule = bb_ordered` (`metrics.ni_certified_p`). The two-family
     conjunction max(p_Jev, p_Intern), the Holm family and the reading rules of A6.1 are unchanged.
  3. *Certified supremum.* U is computed by branch and bound over C_γ(D) (`metrics.ni_tail_sup`), and the upper bound is
     what is returned. Each evaluated w carries f_hi(w) ≥ P_w(R): the windowed sum × (1 + 10⁻⁹) + the exact binomial
     mass outside the window × (1 + 10⁻⁶) + 10⁻¹⁵. The slack exceeds the floating-point error of the log-gamma pmf
     (< 10⁻¹¹ relative). On a subinterval [a, b] with midpoint c the bound is min(max{B(a → c), B(b → c)}, B(a → b),
     B(b → a)) × (1 + 10⁻⁹), with B(a → x) = min over q of f_hi(a)^(1−1/q) · M_q(a, x)^(n/q), 181 exponents
     q ∈ (1.01, 10⁷], and M_q(a, x) = Σ_i p_i(x)^q p_i(a)^(1−q) in closed form. Subintervals whose bound exceeds
     L + tol are bisected, where L is the largest evaluated value and tol = max(10⁻⁷, 10⁻³ L). The search stops when
     U ≤ L + tol; 2n · 10⁻¹⁵ is then added to cover floating-point end points. **Tolerance:** U − sup ≤
     max(10⁻⁷, 10⁻³ sup) + 2n · 10⁻¹⁵, so p exceeds the exact Berger–Boos p-value of this statistic by at most 0.1 %
     of the supremum. If 20,000 evaluations do not reach the tolerance, U is still an upper bound and is used;
     `converged` is reported.
  4. *Validity: P_θ(p ≤ α) ≤ α for every n, every α and every θ in H0: P(loss) − P(win) ≥ m.*
     (i) *Monotone tails.* s̃ is non-increasing in W and non-decreasing in L, because the sets it minimises over are
     nested. So every tail R_t = {s̃ ≤ t} is an upper set of the order "more wins, fewer losses".
     (ii) *Least favourable point.* Draw each task's outcome from one uniform U_i: a loss if U_i < P(loss), a win if
     U_i > 1 − P(win), a tie otherwise. Let ρ = P(win) + P(loss) be the discordance of θ and θ* the boundary point
     with the same discordance, w* = (ρ − m)/2. Then P_θ*(win) ≥ P_θ(win) and P_θ*(loss) ≤ P_θ(loss). The coupling
     never lowers a task's outcome, so (W, L) moves up in the order and P_θ(R_t) ≤ P_θ*(R_t) for every t.
     (iii) *Coverage.* D ~ Bin(n, ρ) under every θ. The Clopper–Pearson interval therefore covers ρ, and C_γ(D) covers
     w*, with probability ≥ 1 − γ.
     (iv) *Berger–Boos.* On that event, U ≥ P_θ*(R) ≥ P_θ(R) ≥ G_θ(s̃(X)) for the observed tail R (which contains
     {s̃ ≤ s̃(X)}), where G_θ is the
     distribution function of s̃(X) under θ; and P(G_θ(s̃(X)) ≤ u) ≤ u for every u. Hence
     P_θ(p ≤ α) ≤ γ + P_θ(G_θ(s̃(X)) ≤ α − γ) ≤ α (Berger & Boos 1994, applied to one statistic and to a set that
     covers the least favourable point of θ).
     (v) *The certificate is an upper bound.* Hölder's inequality under P_a gives P_x(R) ≤ P_a(R)^(1−1/q) ·
     E_a[(dP_x/dP_a)^q]^(1/q) = P_a(R)^(1−1/q) · M_q(a, x)^(n/q) for every event R and every q > 1. M_q(a, ·) is
     convex along the boundary path with minimum 1 at a. The bound at x therefore holds on the whole segment from a to
     x, and the subintervals cover C_γ(D).
     The argument uses no grid, restricts no tail to a branch, and covers the interior of the null through (ii).
  5. *Exact size (a check of the conclusion, not part of the argument).* `scripts/ni_exact_size.py` →
     `results/reviews/2026-10-02/ni_exact_size_r8.md` computes the deployed rule's rejection region exactly.
     Bracketing critical values on a nuisance grid decide every outcome except 749 "gray" ones, which
     `metrics.ni_certified_p` decides; 358 outcomes next to the brackets were re-decided by `ni_certified_p`
     with no disagreement. Exact summation of the trinomial pmf covers n ∈ {500, 149, 300, 1,000, 1,114}, 169 boundary
     values w ∈ [0, 0.4], 18 interior null points, the eleven Holm levels and 0.10. The size is at most 0.997 α at
     every Holm level (0.9970 α at n = 1,114, 0.9964 α at 1,000, 0.9959 α at 500, 0.9958 α at 300, 0.9903 α at 149)
     and 0.998 α at 0.10; no cell exceeds its level. Inside the null it is at most 0.66 α at the Holm levels. On the
     round-7 grid the A6.14 rule reached 1.023 α. Over every power scenario of (c) the A6.21 and A6.17 rules' power
     differs by at most 0.002, so certification costs no measurable power.
  6. *Runtime.* On one CPU, outcomes near the 0.05/11 and 0.05 boundaries take at most 2.3 s per p-value at
     n = 1,114 (median 1.8 s) and 0.9 s at n = 500; the hardest outcome of the size table needed 10,973 evaluations
     (about 3.5 s). Results are memoised per (W, L, n, m).
  7. *Reported, descriptive.* The asymptotic Tango p, the A6.17 p_U (`ni_pvalue(..., rule="tango_uncond")`), the
     exact loss-count p, the loss-rate upper bound 1 − α^(1/n) at zero discordance, and the paired task bootstrap.
     At zero discordance the decisive p is now (1 − m)^n + γ (n = 500: 1.0024 · 10⁻⁴), not (1 − m)^n.
  8. *Final n′.* As A6.17 (b) said, each final retained n′ is added to `SETTINGS` in `scripts/ni_exact_size.py` and the
     table is re-run before analysis. Validity is proven, so a size above α would indicate an implementation error; the
     analysis stops until it is found. The sentence "reported, not acted upon" is withdrawn.
  9. *Code* (lands under A6.10): `metrics.ni_order_table`, `ni_holder_logbound`, `ni_tail_sup`,
     `ni_bb_nuisance_set` and `ni_certified_p`; `ni_pvalue`'s default rule is `bb_ordered`, with `rule="tango_uncond"`
     (A6.17) and `rule="tango"` (A6.14) kept as comparison rules; `tests/test_ni_certified.py`.

- **(b) C5: the null, the validity statement, and the decisive test (replaces A6.16 (g)'s C5 clause).** C5 tests
  H0: E[ΔU] ≤ 0 against E[ΔU] > 0, where ΔU is the paired per-task utility difference (gross of System-1's price,
  W = $1, λ = 0) between the trigger arm and its matched-rate random control.
  1. *What holds for the registered sign-flip test.* Σ d_i² is invariant under sign changes, so the sign-flip test of
     the mean coincides with the studentized (paired-t) sign-flip test. It is exact when the per-task difference is
     symmetric about zero, which holds under within-task exchangeability of the two arms. For the weak null
     E[ΔU] = 0 with finite variance it is asymptotically of level α: the sign-change distribution of n^(−1/2) Σ ε_i d_i
     tends to N(0, E d²), and E d² = Var d under the null (Romano 1990, JASA 85:686–692; Lehmann & Romano 2022, TSH
     4th ed., Example 17.2.4; Ritzwoller, Romano & Shaikh 2025, arXiv:2406.09521, §3.2). It is not exact there.
     Janssen (1997, Statist. Probab. Lett. 36:9–21) and Chung & Romano (2013, Ann. Statist. 41:484–507) prove the
     corresponding result for two-sample studentized permutation tests; Janssen (1999, Math. Methods Statist.
     8:320–343) treats symmetry tests of statistical functionals. Romano (1990) appeared in JASA; Romano's Annals of
     Statistics paper on randomization tests is 1989 (17:141–159).
  2. *Finite-sample size* (`scripts/c5_signflip_size.py` → `results/reviews/2026-10-02/c5_signflip_size.md`; the
     registered procedure itself, its 20,000 seed-0 sign vectors, n = 500, 100,000 replicates per shape, every shape
     with mean exactly 0). With balanced success flips (±$1 on 5–20 % of tasks, the dev pair's cost differences
     recentred) the size is at most 0.99 α at 0.05 and 1.00 α at 0.05/11. When a small systematic cost difference
     offsets a success deficit, the size reaches 1.54 α at 0.05 and 1.96 α at 0.05/11. That holds already at the dev
     pair's own mean cost difference (+4·10⁻⁵ dollars per task, flips unbalanced by the same amount). The success part
     of the permutation distribution lies on a lattice of step 2/n. A systematic cost part, however small, moves the
     observed mean off its lattice point in one direction. The observed sign pattern then beats every pattern with the
     same success sum, and p falls by a whole lattice step. The registered test is therefore materially
     anti-conservative for the weak null on shapes C5 can produce.
  3. *Decisive test (changed).* C5 is decided by the one-sided paired t-test on the same
     per-task ΔU (Student t with n − 1 degrees of freedom; `metrics.paired_t_test`, `p_greater`); the sign-flip p is
     reported as descriptive. On the same replicates its size is at most 1.01 α at 0.05 and 1.09 α at 0.05/11, within
     two Monte-Carlo standard errors of α on every realistic shape. Registered wording: "C5 is decided by the
     one-sided paired t-test of the mean per-task utility difference. It is asymptotically of level α for H0:
     E[ΔU] ≤ 0; by simulation at n = 500 its size is within Monte-Carlo error of α on utility differences built from
     the development pair, including a systematic cost difference that offsets a success deficit." Code: one line in
     `scripts/primaries.du_block` (`p_decisive` from `M.paired_t_test(g)["p_greater"]` instead of
     `M.sign_flip_test(...)["p_greater"]`), with `tests/test_primaries.py`'s C5 assertion updated; the sign-flip p
     stays in the block.
  4. *Shapes outside the guarantee.* Rare large losses offset by small gains on most tasks inflate both tests (up to
     29 α at the floor in the stress rows). They cannot arise in C5: with W = $1 the only large values of ΔU are the
     success flips ±1, and the System-2 cost differences are below 10⁻³ dollars per task in the dev pair. Under
     E[ΔU] = 0 the flips can then be unbalanced by at most about 10⁻³, which the realistic rows cover.
- **(c) The power table (replaces the power tables of A6.1 and A6.14, their "Reading" sentences, and A6.17 (c)'s
  pointer to `ni_power_exact.md`).** `scripts/ni_power_exact.py` → `results/reviews/2026-10-02/ni_power_exact_r8.md`
  computes exact power for the A6.21 rule, with A6.17 and A6.14 next to it.
  1. *Feasibility.* A scenario (discordance d, true difference Δ) is possible against reference success p_ref only if
     P(win) = (d + Δ)/2 ≤ 1 − p_ref and P(loss) = (d − Δ)/2 ≤ p_ref. At p_ref = 0.98 this means d + Δ ≤ 4 pt. The
     round-7 rows d = 5 % / Δ = 0, d = 10 % / Δ = 0 and d = 10 % / Δ = −1 pt are impossible there. They are kept only
     as generic scenarios (feasible for p_ref ≤ 0.975, 0.95 and 0.955).
  2. *Power (A6.21 rule, exact).* One family at α = 0.05 and at the Holm floor 0.05/11; the conjunction of the two
     System-1 families (C1, C2, C6) under item 3's independence assumption. Every row is feasible at its pilot
     reference success; the A6.17 rule's power differs by at most 0.002 in every row.

     | primary (pilot reference success) | n | m | d | true Δ | one family, α = 0.05 | one family, 0.05/11 | conjunction, 0.05 | conjunction, 0.05/11 |
     |---|---|---|---|---|---|---|---|---|
     | C1F / C2F / C6F (Flash, 0.98) | 500 | 3 pt | 2 % | 0 | 0.99 | 0.91 | 0.99 | 0.82 |
     | | | | 2 % | −1 pt | 0.85 | 0.45 | 0.72 | 0.21 |
     | | | | 4 % | 0 | 0.94 | 0.67 | 0.88 | 0.45 |
     | | | | 4 % | −1 pt | 0.67 | 0.27 | 0.45 | 0.07 |
     | | | | 5 % | −1 pt | 0.60 | 0.22 | 0.37 | 0.05 |
     | | | | 6 % | −2 pt | 0.22 | 0.04 | 0.05 | < 0.01 |
     | ceiling reference, losses only (1.00) | 500 | 3 pt | 1 % | −1 pt | 0.93 | 0.62 | 0.87 | 0.38 |
     | | | | 2 % | −2 pt | 0.33 | 0.07 | 0.11 | < 0.01 |
     | C1L / C2L / C6L (Flash-Lite, 0.80) | 500 | 3 pt | 20 % | 0 | 0.44 | 0.13 | 0.19 | 0.02 |
     | | | | 30 % | 0 | 0.33 | 0.08 | 0.11 | 0.01 |
     | | | | 20 % | +3 pt | 0.91 | 0.64 | 0.83 | 0.41 |
     | | | | 30 % | +5 pt | 0.95 | 0.74 | 0.90 | 0.55 |
     | C7 (τ², majority pass, 0.80) | 149 | 5 pt | 20 % | 0 | 0.37 | 0.09 | – | – |
     | | | | 30 % | 0 | 0.29 | 0.06 | – | – |
     | | | | 30 % | +3 pt | 0.55 | 0.19 | – | – |
     | C8 (E1, 0.93: pilot, n = 56) | 1,114 | 3 pt | 6 % | 0 | 0.99 | 0.91 | – | – |
     | | | | 10 % | 0 | 0.93 | 0.69 | – | – |
     | | | | 10 % | +3 pt | > 0.99 | > 0.99 | – | – |
     | generic, no reference (round-7 rows) | 500 | 3 pt | 5 % | 0 | 0.89 | 0.57 | – | – |
     | | | | 10 % | 0 | 0.67 | 0.29 | – | – |
     | | | | 10 % | −1 pt | 0.40 | 0.11 | – | – |

  3. *Conjunction (C1, C2, C6).* The conjunction passes iff both families pass. With the two families facing the same
     scenario and independent tests, its power is the square of one family's power. The Fréchet bound
     min(π_Jev, π_Intern) is the upper limit under any dependence. Both families are compared with the same reference
     runs, which makes the tests positively dependent, so the true value lies between the two. With different scenarios
     per family the product of the two single-family entries applies.
  4. *Wording.* "n = 500 is adequate for the Flash NI pairs …" (A6.1) and "the Flash pairs are well powered only for …"
     (A6.14) are withdrawn. Registered wording: "The exact power table of A6.21 (c) is the planning statement. At the
     Holm floor, three of its feasible scenarios reach 80 % power for one family (Flash at d = 2 %, Δ = 0; C8 at
     d = 6 %, Δ = 0, and at d = 10 %, Δ = +3 pt), and one two-family conjunction does (Flash at d = 2 %, Δ = 0). A
     primary that fails elsewhere is expected to read *inconclusive at this n*. The two-family conjunctions have lower
     power than either family alone."
- **(d) Superseded text.** A6.21 supersedes, without editing them: A6.1's zero-discordance p and its power table
  and "adequate" reading; A6.14's *Rule* and *Switch* as the decisive p, and its power reading; A6.16 (g)'s C5 clause
  (the C3 / C4 sign-test clause stays); A6.17 (b)'s decisive p_U, its grid maximum, its validity sentence and "a
  residual above α at an off-grid w would be reported, not acted upon"; A6.17 (c)'s replacement wording and power
  pointer; A6.18 (a)'s pointer to `results/reviews/2026-10-01d/ni_power_exact.md`; and A6.18 (c)'s "the A6.17 rule's p",
  which reads "the A6.21 rule's p (`ni_pvalue(...)["p_decisive"]`, rule `bb_ordered`)".
- Cost: CPU only (exact size table 9 min, power table 1 min, C5 simulation 9 min on four CPU threads).

### A6.22 — Design specifications from review round 8 (2026-10-02, before the tag; `results/reviews/2026-10-02/`)

Review round 8 (gpt-6-astra W5, W6, W7, W11, W13, D.22–25, D.32–36; gemini-3.1-pro W3) found six places where the
registration leaves an execution or verdict rule to be read off the code or left open. Each is settled here, by
specifying what the code does where it does it, and by a minimal explicit rule where the code leaves a choice. No arm
name, size, seed, margin, threshold or primary test changes; H1, H3, H4 and H5 stay secondary / exploratory.

- **(a) The C1 call graph and three coverage quantities (W5).** Specification of `harness/agent.py` as implemented
  (`_tool_decision` → `_delegated_call` → `_fill_params`); no behaviour changes. Per agent step:
  1. *Tool deferred.* With `tool_select.always_defer` (A6.11 (c).4) System-1 is not queried; otherwise it answers one
     tool Choice — one request when the K options fit one read (`max_options`: Jev 255, so the K = 69 options of the
     registered suite take one request; Intern-Decision 62, so the harness tournament `_choice_tournament` takes two
     batched requests) — and the decision defers when its wire confidence is below τ (`>=`, A6.11 (c).3). In both
     cases ONE native System-2 call (every tool offered, no `tool_choice`) writes the tool and all its arguments, and
     **`tool_params` is never consulted**: no argument question is asked or logged, whatever its τ, and the executed
     arguments are the LLM's. Nothing System-1 answered is reused (its tool answer is logged only). System-1 requests
     S1 = 0 (always-defer) or r ≥ 1; System-2 calls S2 = 1.
  2. *Tool accepted.* `_fill_params` walks the chosen tool's parameters with the frozen fill kinds (A4): a bounded or
     copy argument with ≥ 2 options gets its own System-1 Choice (one unbatched request each; copy options are state
     spans, instruction values first, plus *Other*); a bounded argument with one admissible value is filled by rule
     (no request); a copy argument without a candidate in the state, and every free-text argument, go to the LLM and
     are never asked. An asked argument is accepted iff its answer is not *Other* and its confidence ≥ the tool_params
     τ. If a required argument is still open, or the tool has a free-text argument (required or optional), exactly ONE
     forced System-2 call follows (`tool_choice` = the chosen tool, only its schema offered, the accepted values shown
     as already determined); System-1's accepted values overwrite the LLM's on conflict, deferred arguments take the
     LLM's value (logged as `s2_value` = `final_value`), and a reply without a tool call leaves the open arguments
     empty (`finish`: its text becomes the answer). Otherwise no System-2 call is made, and a deferred *optional*
     argument is omitted from the executed call. S1 = r + a (a = asked arguments), S2 ∈ {0, 1}. An accepted `finish`
     always costs the forced call (its `answer` is free text).
  3. *Calls avoided.* LLM-only makes one native call per tool step. The cascade avoids it exactly on the steps where
     the tool and every required argument were accepted and the tool has no free-text argument; it never adds a
     System-2 call to a step (the forced call replaces the native one) and adds r + a System-1 requests. The fit's
     tool_select ΔC credits the whole native call to an accepted tool although the call is avoided only on those
     steps; A6.16 (d) applies (the deployed ledger is authoritative).
  4. *Consequence.* Where tool selection always defers — Jev on Flash-Lite, Intern-Decision on Flash and on Flash-Lite
     (Deviations, 2026-10-01 and 2026-10-02) — the fitted argument thresholds (0.13; 0.6309; 0.8518) are unreachable:
     the complete policy behaves as LLM-only with a deferred record per step, its coverage is zero by construction, and
     an NI result is *policy non-inferiority with zero coverage* (A6.11 (c).4). Jev on Flash is the only C1 constituent
     whose argument threshold (0.99) can act, and only on steps whose tool choice is in the reported-one set (τ 1.00).
  5. *Coverage, three registered quantities*, reported next to every C1, C2, C6 and C7 result and in the per-family
     table (`bench/scorecard.py`: `decisions.coverage`, `vs_baseline`):
     - **tool-decision coverage** = System-1 tool choices executed / tool-selection opportunities (`tool_coverage`;
       = the tool_select entry of `delegation_by_dp`);
     - **argument coverage** = argument values chosen by System-1 / argument slots of all executed tool calls
       (`arg_coverage`; the slots of LLM-written calls count in the denominator), with its conditional companion
       inside System-1-chosen calls (`arg_coverage_s1_calls`, undefined when System-1 chose no call);
     - **native LLM calls avoided per episode versus LLM-only** = paired per-task mean of the reference's minus the
       arm's System-2 calls (`llm_calls_avoided`, `llm_calls_avoided_frac`), printed with the within-arm share of
       LLM-free steps (`llm_free_steps`), because the paired difference also moves with trajectory length.
     A6.20 (e)'s 5 % rule applies per decision point: an arm's delegation is *negligible* iff its tool-decision and its
     argument coverage are both below 5 %, and *zero* iff both are 0. A coverage without any opportunity (undefined)
     does not enter the rule; with both undefined the arm carries no delegation label. In assist, guard and trigger modes System-1's
     answer acts at every opportunity (coverage 1 by A6.20 (e)); only their calls-avoided figure is informative.
  6. *Request-trace tests* (`tests/test_cascade_call_graph.py`): one per branch — tool deferred with tool_params τ = 0;
     always-defer; tool and every argument accepted (LLM-free step, then an accepted finish with its forced call); one
     required argument deferred (forced `tool_choice`, only that tool offered, System-1 winning a conflict); free text;
     a deferred optional argument omitted and a single-option argument filled by rule; a tournament — asserting the
     System-1 requests and System-2 calls per step and the score-card quantities above on the same traces.
  7. *Code item.* `scripts/primaries.py` labels the C1 constituents from the pooled `delegation_ratio` and tests
     `== 0` for both the zero-coverage and the replication label; it reads the two coverages of (5) and applies the
     5 % rule of A6.20 (e) to "cross-backend replication of policy NI".

- **(b) The τ² refit: labels, grouping and non-cascade selections (W6).** Specification of `bench/suites/tau2.py`
  (`_Reference`, `_select`, `_shadow_params`) and of `threshold_theory --calibration-arm` as run in
  `docs/MAIN_RUN_PLAN.md` step 8 on `runs/tau2/airline_shadow` (the 15 shadow tasks × 3 trials). The refit is the
  **deployed C1 estimator** (`scalar_threshold_projection`, A6.11 (c).7, A6.19; V rule of A6.11 (c).1) applied to
  **reference-agreement labels**, not to an oracle: A5.4 (c)'s "the same H1 fit the C1 arm uses" reads "the same
  deployed estimator", and the paper calls the τ² labels reference-agreement labels.
  1. *Acceptable tool set.* τ² logs none. Each tool_select record's gold is one *reference next step*
     (`_Reference.next_step`): the first pending reference action (tau2 `evaluation_criteria.actions` of the
     assistant not yet executed successfully, by tau2's `compare_with_tool_call`; a pending READ that precedes an
     executed action is moot) whose scalar argument values have all appeared in the conversation and, for a WRITE tool
     under a confirmation policy, only when the last user message is affirmative; `respond_to_user` when no such action
     exists. The acceptable set is the singleton {gold}; `respond_to_user` is one of the options.
  2. *Argument correctness.* Only required arguments are asked (`_fill_params` / `_shadow_params`; tau2's own
     free-text rule `_FREE_TEXT_PARAM` and `ParamInfo.is_copyable`), and a record has an argument gold only when the
     executed tool is the reference next step. The gold is the reference action's value (every argument of the
     action, also those tau2's `compare_args` would skip); a copy option list without it makes the gold *Other*. There
     are no alternative valid values: any other value is wrong. System-1 is right iff its option label equals the gold
     label (`metrics.is_correct`), the LLM iff `_same_value(value, gold)`.
  3. *Paired System-2 label.* The LLM's executed action on the same turn of the same simulation (the shadow arm
     executes it): tool_select `s2_correct` = (the LLM's action equals the gold; a text reply is `respond_to_user`);
     tool_params `s2_correct` = `_same_value`.
  4. *Episode grouping for V.* `counterfactual_v` groups by (arm, task id) and the τ² task id carries the trial
     (`airline/<id>#<trial>`): each of the 45 simulations is one episode, success is reward = 1, and V = W (S_right −
     S_wrong) over them, with V = W when undefined or not positive (A6.11 (c).1); its bootstrap resamples
     simulations (descriptive only).
  5. *Unlabelled decisions and empty bins.* Records without a gold (no reference; arguments of a non-reference tool)
     leave g, q_L, ΔC and V (`metrics.has_gold`). A bin without an executable (non-*Other*) System-1 answer takes the
     global g (`scalar_threshold_projection`'s per-bin fallback; A6.19 (c)); with no executable answer at all every
     bin rejects and the decision point always defers (A6.20 (a)). A6.5's sample-size rule, never withdrawn and not
     binding on E2, applies to each τ² decision point: with fewer than 30 paired decisions τ̂\* is the marginal
     `prop1` value the fit reports (`primary.plug_in.prop1`), with fewer than 10 τ = 0.9; the fit's `n_paired`
     decides, the choice is written with the threshold, and `fit_calibration_arm` gains the switch (code item).
     A6.5's count rule is kept because it is registered text.
  6. *Stated approximation.* ΔC is the turn's logged LLM cost (tool_select) and that cost split over the turn's
     argument records (tool_params, `couple_by_step`); accepting `respond_to_user` still costs the message call
     (`_respond`), so ΔC overstates the avoidable cost of that answer; A6.16 (d) applies.
  7. *Non-cascade selections.* When `c7_selection.json` selects ASSIST or GUARD, nothing of C7's arm is refit: ASSIST
     runs `configs/tau2/jev_ts_assist_k3.yaml` (k = 3), GUARD runs `configs/tau2/jev_guard.yaml` (policy guard τ 0.5
     on WRITE calls, transfer guard τ 0.5, `guard_max_retries` 1; `jev_shadow` does not shadow the policy guard, so no
     τ² data exist to refit it). The shadow run and the refit still precede the C7 runs (they set the A5.4 (d)
     fallback C1-arm row and are reported descriptively), and A5.4 (c)'s E2-carried cascade row still runs.

- **(c) Verdict rules of H1, H3 and H4 (W7, D.24–25).** These hypotheses stay secondary / exploratory; each verdict is
  computed mechanically as follows and labelled as such.
  - **H1** (replaces §3's *Falsified if*; A6.5, A6.16 (h), A6.19 (d) unchanged).
    - *Cells:* {tool_select, tool_params} × {Jev, Intern-Decision-4B} × {Flash, Flash-Lite} × W ∈ {0.1, 1, 10} ×
      λ ∈ {0, 0.5, 5} $/h = 72 cells, from the n = 500 evaluation shadow arms (`jev_shadow`, `jev_shadow_s2lite`,
      `intern_shadow`, `intern_shadow_s2lite`), each `threshold_theory RUN --arm <arm> --dp <dp> --train-frac 0.3
      --seed 0 --taus h1 --rule general --task-value W --lam-per-hour λ`. When the counterfactual V of the training
      split or of the cell is undefined or not positive, the cell is run with `--V W` (A6.11 (c).1) and says so.
    - *Statistic:* d = held-out loss of τ = 0.8 minus that of τ̂\* (`regret.default_minus_hat.diff`; τ̂\* =
      `candidates.tau_hat`, the `general` plug-in); the deployed projection (`tau_hat_projection`) gets the same count
      as a secondary line. Win d > 0; tie |d| ≤ 10⁻¹² (identical held-out loss, i.e. identical accept sets on the
      test decisions), counted as a non-win; loss d < 0. One vote per cell (the decision-weighted share is
      descriptive). Absent cells (arm missing, no paired decision on a split, τ̂\* undefined) are listed and leave the
      denominator; below 36 estimable cells the clause is *not estimable*.
    - *Clause 1* holds iff wins > half of the estimable cells, on point estimates (§3's "beat"); every cell prints
      the 95 % paired task-bootstrap CI of d (`default_minus_hat.ci`) and the numbers of cells whose CI lies above and
      below 0.
    - *Clause 2* (recalibration), at the primary operating point W = $1, λ = 0 (A6.5). A (backend, decision point,
      System-2) cell is **over-confident** iff, on its 30 % training split, the mean top-label confidence minus the
      accuracy of System-1's paired Choice decisions has a one-sided 95 % task-bootstrap lower bound (the 5 %
      quantile) > 0. It **returns to the front** iff the held-out regret of the fixed τ = 0.8 against the best swept
      τ has the lower end of its two-sided 95 % task-bootstrap interval > 0 on raw confidences (the logged wire
      confidence) and ≤ 0 on recalibrated ones; a T = 1 control row (the logged distribution rescaled without a
      temperature) is printed next to them (temperature fitted on the training split with
      `decision.calibration`, per question kind, applied to the test split's logged distributions). Clause 2 holds
      iff at least half of the over-confident cells return; with none it is *not applicable*.
    - *Binary-score caveat.* A strictly monotone recalibration leaves the set of threshold policies on a log
      unchanged, and τ̂\* (isotonic fits) and the deployed projection (equal-mass bins of whole tie groups) are
      rank-based, so recalibration cannot move them or the attainable front; temperature scaling is monotone in the
      score for yes/no and two-option questions and within each option count otherwise. Clause 2 therefore measures
      the cost of a mis-set fixed threshold (a reparameterisation of the default), never a change of the attainable
      frontier, and the paper says so.
    - *Verdict:* supported iff clause 1 holds and clause 2 holds or is not applicable; otherwise falsified (or *not
      estimable*). Code item: the over-confidence gap with its task bootstrap and the recalibrated-default regret row
      (`threshold_theory`); missing at analysis time, clause 2 is reported as not evaluated (dated deviation).
  - **H3** (§3's method made mechanical).
    - *Universe:* the acting arms whose System-1 is Jev or Intern-Decision-4B, per suite × System-2 model — L2: the
      four JevGym evaluation configs; L3: the τ² airline test tasks (the A5.4 (e) configurations) and ALFWorld (E6).
      Shadow / measurement arms, references, controls (BM25, TF-IDF, embedding, rate, random-score, heuristic, prior,
      s2self) and generative stand-in rows are outside it.
    - *Class of an executed System-1 decision:* **meta** — reflection, routing, stop, memory_write, memory_read in
      trigger / guard / assist mode; **bounded-recoverable** — tool_select or action in assist or guard mode (the
      LLM's choice executes; τ²'s policy guard included), System-1 tool or action choices executed in replace or
      cascade mode that are not irreversible, executed bounded arguments (options without *Other*); **copy-dependent**
      — executed copy arguments (options ending in *Other*, A4's copy class); **irreversible** — System-1 tool choices
      executed in replace or cascade mode of an irreversible or final tool: strict JevGym
      `threshold_theory.JEVGYM_FATAL_TOOLS` (cancel_order, refund, ship_order, finish), τ² WRITE tools
      (`_tool_kind == "write"`) and `transfer_to_human_agents`, none on ALFWorld.
    - *Credit (non-negligible delegation):* an arm is credited with class K iff its executed System-1 decisions of
      class K are ≥ 5 % of the opportunities of the decision point at which they are taken (A6.20 (e); tool-selection
      opportunities for tool choices, argument slots for argument values, as in (a)); assist, guard and trigger modes
      act at every opportunity and are credited with their class whenever the arm has an opportunity. An arm credited
      with no class (always-defer, a cascade below 5 %) is *negligible delegation* and leaves the front analysis and
      the ranking. This maps every registered configuration: `*_replace` tool arms → irreversible (+ copy-dependent
      for `*_params_replace`); cascades → by their executed decisions; tool `*_assist*` / `*_guard` → bounded-
      recoverable; stop, reflection, routing and memory arms → meta; ALFWorld action arms → bounded-recoverable
      (+ meta for the reflection-trigger arm of `configs/experiments/alfworld_action_only.yaml`); τ² cascade → by executed decisions, `jev_ts_assist_k3` →
      bounded-recoverable, `configs/tau2/jev_guard.yaml` → bounded-recoverable + meta, `configs/tau2/jev_route.yaml` → meta. A multi-class arm is ranked
      under its riskiest class (irreversible > copy-dependent > bounded-recoverable > meta) and is a counterexample
      candidate whenever it is credited with copy-dependent or irreversible.
    - *NI and the front (descriptive):* the arm passes the `noninferior` qualifier (A6.18 (c)) against its suite ×
      System-2 LLM-only reference and no other qualifying arm or the reference dominates it in (cost, success)
      (`metrics.pareto_front`). No multiplicity or selection adjustment; arms at n = 200 and n = 500 differ in power,
      so n is printed with every membership.
    - *Ranking statistic:* per suite × System-2 model and class, the median net cost savings (`vs_baseline.
      cost_savings`) over the class's qualifying arms, 0 (the reference's) for a class without one; Kendall τ-b over
      the four classes for Flash vs Flash-Lite (L2) and L2 vs L3 (Flash), with the exact permutation p over the 24
      orderings.
    - *Verdict (unchanged):* falsified iff a credited copy-dependent or irreversible arm is NI and on the front, or a
      τ-b is < 0; otherwise "no registered counterexample", which does not establish the ordering.
  - **H4** (replaces §3's *Falsified if*; the R3 / R4 recipes are unchanged).
    - *Matching on development data:* decision accuracy of R3 and of every saved R4 checkpoint on the v1 development
      set (`training.eval_decisions`, `data/train_v1/dev.jsonl.gz`, 2,377 records from training seeds; never E1 / E2
      evaluation data); matched iff |acc(R4) − acc(R3)| ≤ 1 pt; the R4 checkpoint used is the matched one closest to
      R3 (ties → the later step). With none within 1 pt, H4 is *not tested at matched accuracy* and the comparison
      is descriptive, with the gap.
    - *Downstream comparison:* R3 and the matched R4, each served by `training/serve_jev.py`, as System-1 of the
      `jev_ts_params_replace` configuration (Flash) on the first 200 tasks of the frozen evaluation list, on identical
      tasks, after the freeze. Replace mode is chosen because there decision quality reaches the outcome without a
      threshold.
    - *Decision on Δ = success(R4) − success(R3), paired task bootstrap (B = 10,000):* supported iff the one-sided
      95 % lower bound is > 0; **falsified iff equivalence within ±3 pt is shown** (TOST at α = 0.05: the 90 % CI
      inside (−3, +3) pt, `metrics.tost_paired(..., 0.03)`); otherwise *inconclusive at this n*, with the estimate and
      both intervals. "Falsified inside ±3 points" never means a point estimate inside the band.

- **(d) H5: displayed verdict, the Terminal-Bench-Science subset, the withdrawn order-statistic sentence (W11; gemini
  W3).**
  1. *Label only.* The registered verdict (A5.2 / A6.6 / A6.16 (j): the exploratory success screen and faster on
     co-solved tasks; `metrics.h5_verdict`'s `success_preserving_and_faster`) is displayed as **"passes the exploratory
     success screen; faster among jointly solved tasks"**; the stricter tier is displayed as "faster and better
     (faster among jointly solved tasks)". Only the label changes: rule, thresholds, tests and field names are as
     before. Code item: `metrics.h5_verdict` (`screen_label`) and `adapters/pi/summarize_harbor.py` (`h5_verdict`).
     Wherever a verdict is printed (Table "hard", App. I) the same row shows the all-task capped-time median Δ and
     solved-by-t at t ∈ {600, 1800, 3600} s.
  2. *The subset (A6.16 (j)'s "deterministic selection rule", now exact).* Input: the 37-task pool
     `adapters/pi/tbsci_hard_pool.txt` (sha256 `f131ee5289003d95facdb055bb308a18a9d68280ddabb2aeb1c9fd122c9261fe`,
     1,389 bytes; terminal-bench-science@`cfff307`; life 11, physical 11, earth 5, engineering 5, mathematical 5) and
     one calibration pass in its own output directory at the registered caps of (e):
     - *validity:* drop a task whose `oracle` trial does not reach reward 1, or whose oracle agent-run time exceeds
       1,800 s (half the 1 h cap; the cap is not raised), or whose build / verifier fails twice (an infrastructure
       failure is retried once);
     - *floor:* `nop` gives each task's floor f, the verifier tests passed with no agent output;
     - *signal:* one `baseline` attempt; keep a task iff it has reward 1 or passes more tests than f (an
       infrastructure-errored attempt, any Harbor exception except `AgentTimeoutError`, is retried once, then the task
       is dropped; a timed-out attempt is scored as it stands);
     - *fill:* N = 20 — 4 slots per domain (the task directory's domain: earth, engineering, life, mathematical,
       physical); within a domain, eligible tasks by `expert_time_estimate_hours` ascending, ties by slug; slots a
       domain cannot fill go one at a time, round-robin in that domain order, to the domains with eligible tasks left,
       until 20 tasks are chosen or none is left (a smaller subset is reported as such);
     - *freeze:* `adapters/pi/tbsci_hard.txt` (one slug per line; header with the pool hash, the calibration run id
       and the commit), its sha256 and size recorded in the Deviations section before any evaluation trial;
       calibration attempts are never reused. Code item: a `tbsci.py select` subcommand implementing these steps
       (else they are applied by hand and their inputs committed). `tbsci_hard.txt` does not exist at this writing.
  3. *Withdrawn.* A6.6's sentence "one capped attempt moves Δsuccess by 1/(k·n) and each median by at most one order
     statistic (`tests/test_metrics.py::test_h5_one_timeout_cannot_flip_the_verdict`)" is withdrawn. What that test
     checks: in one fixed scenario (20 tasks × 3 attempts; baseline solves 14 tasks, the arm 18 in every attempt and
     2 in two of three), turning any single solved arm attempt into a timeout lowers Δsuccess by exactly 1/60 and
     leaves the registered verdict standing. The co-solved set never changes in that scenario and the test asserts
     nothing about medians. Replacement: "changing one binary trial outcome changes the all-task mean success by
     1/(k·n); co-solved timing summaries can also change through their inclusion set."

- **(e) Execution semantics (W13).** Specification of `harness/agent.py::JevAgent.run`, `bench/runner.py`,
  `bench/suites/tau2.py::convert_results`, `scripts/primaries.py`, the LLM / System-1 clients and the adapters.
  1. *Infrastructure error* = an episode record with `error` set: in the reference loop (JevGym, ALFWorld) any
     exception that escapes the episode — System-2 transport failures after the client's retries
     (`llm/openai_compat.py`: up to 6 with backoff on 408, 409, 429, 500, 502–504, 529, connection errors and
     timeouts; any other status at once; τ²: litellm's own retries), System-1 transport failures after the Jev
     client's 2 retries (408, 429, 500, 502–504, 529, timeouts) or a malformed System-1 response (a missing answer,
     `JevAPIError`), any exception of a live System-1 call (live modes are not fail-open in the reference loop),
     harness exceptions; on τ²: termination `agent_error`, `user_error` or `infrastructure_error`, or an exception in
     `JevTau2Agent` (side-car `error`); on the hard tasks: a Harbor trial exception other than `AgentTimeoutError`.
  2. *Agent failure* (a failure, never an exclusion): an episode that ran to its end and failed — a wrong or
     irreversible action, a wrong answer, the step cap (`max_steps` 30 on JevGym, 50 on ALFWorld), an empty or
     text-only LLM reply (read as a final answer), an unknown tool name, unparseable tool-call JSON (executed as
     `__raw__`), the LLM ignoring a forced call, a guard rejecting until the cap; τ²: any other termination with
     reward < 1; hard tasks: the agent timeout or turn cap (time counted at the cap), a missing artifact.
  3. *Measurement failures* (fail-open, never an error and never a change of the episode): shadow System-1 errors
     (`meta.shadow_s1_errors`), oracle / gold, counterfactual and probe errors; they only reduce labelled
     opportunities and are counted per arm.
  4. *Retries.* Transport retries happen inside a call (attempts and backoff logged per call; backoff is excluded from
     latency). Episode retries: the runner re-runs a task whose latest record is an error on every invocation, and the
     main runs invoke each config three times (`docs/MAIN_RUN_PLAN.md`), so a task has at most three attempts (§1); a
     retry is a fresh execution from the start; the latest record decides; `scripts/primaries.py` counts attempts
     (`MAX_ATTEMPTS = 3`) and refuses to treat an errored task with attempts left as final. τ²: an errored simulation
     is re-run (same task, same trial) up to twice before the analysis (code item, A6.20 (f)); a task with any trial
     still errored leaves the pair with all three trials. Hard tasks: an errored trial is re-run up to twice (a rule;
     nothing automates it).
  5. *Drop rule and its reporting.* The paired complete case of §1 / A6.9 is unchanged and assumes that the retained
     tasks stay representative of the pair. Next to every primary's full-set (errors-as-failures) sensitivity row the
     dropped tasks are listed — task id, suite family (JevGym `clarify` / `multi`; τ² domain and split), the arm(s) in
     which they errored, the error class (exception type and first line; τ² termination reason), the attempts made and
     the other arm's outcome (code item: `scripts/primaries.py`). The same rule applies to H5 pairs;
     `adapters/pi/summarize_harbor.py` today counts errored trials as failures, which is the full-set view, and gains
     the complete-case view (code item).
  6. *Guard exhaustion.* Reference loop (`guard_max_retries` = 1 in every registered config): a rejected tool proposal
     is regenerated once with the rejection note and the regeneration executes unchecked; a rejected finish adds the
     stop note and the episode continues, with no cap on rejections other than `max_steps`. τ²: one check and at most
     one regeneration per turn shared by all guards (`guard_max_retries` 1 in `configs/tau2/jev_guard.yaml`), the last
     regeneration executes unchecked. pi: at most 2 stop nudges per prompt, then the stop is accepted; after 1 blocked
     tool call the next one runs unverified. mini-SWE-agent: a re-issued identical command runs; the submit guard
     defers at most `max_nudges` times. Exhaustion is deployed behaviour, never an error.
  7. *Fail-open events* (pi extensions, the Codex / DeepSeek Harness hooks, mini-SWE-agent): a System-1 error or
     timeout makes that decision behave as the unmodified harness (tool allowed, stop accepted, router → its fallback
     model, default compaction). The episode is scored as run — fail-open is part of the deployed policy, not an
     infrastructure error — and the count and rate of fail-open decisions per arm (`jev-decisions.jsonl` records with
     `error`; mini-SWE mode `error`) are reported with the arm.
  8. *Cached and fresh responses.* System-1 answers go through the SQLite cache (`s1_cache`, keyed on the backend
     description, state and questions): a hit replays the stored answer with its original cost and latency (`cached`
     on the call record), so identical (state, question) pairs in another arm, pass or retry get the identical answer;
     the share of cache-served System-1 calls per arm is reported (`s1_cached_frac`). System-2 calls are never served
     from a response cache — every attempt, trial and retry is a fresh request (temperature 0; provider
     nondeterminism is part of the measured variance) — and no litellm / tau2 response cache may be enabled in a main
     run. Provider prompt caching is billed as reported and the no-cache price is a sensitivity column (§4, A6.9).
     τ² trials (`--num-trials 3`) and hard-task attempts (k = 3) are independent executions.
  9. *Hard-task caps*, as found in the adapters: **SWE-bench Verified** (mini-SWE-agent,
     `adapters/mini_swe/swebench_jev.yaml`): `step_limit` 250, `cost_limit` $3 per instance, 60 s per command, k = 1;
     **Terminal-Bench 2.0** (`adapters/pi/run_tb2.sh` defaults): each task's own agent timeout, no turn cap, k = 3
     (`N_ATTEMPTS=3`; the script's default is 1); **Terminal-Bench-Science** (`adapters/pi/run_tbsci.sh` defaults):
     1 h of agent time (`AGENT_TIMEOUT_MULT=0.125` of the stated 8 h) and 100 turns (`MAX_TURNS=100`), k = 3. Every
     arm of a suite runs under the same caps; a cap is a budget deviation stated in the paper. The registered
     Terminal-Bench-Science turn cap is the script's 100 (the paper's earlier "150 turns" is corrected); Terminal-Bench
     2.0 has no turn cap.

- **(f) Generated view.** `docs/PREREG_EFFECTIVE.md` (`scripts/prereg_effective.py`): C7's eligible set no longer
  lists LLM-only (A6.20 (d)); the C1 rows say why an always-defer policy delegates nothing (the argument thresholds
  are unreachable after tool deferral, (a)); C7's refit is named as the deployed estimator on reference-agreement
  labels ((b)); the power sentence follows A6.14 / A6.17 (no "adequate"); the H1, H3, H4 and H5 entries carry (c) and
  (d), and the drop-rule entry (e).

- **(g) Slip rule.** Must land before the tag: this text and the paper wording it implies. May land later, before the
  run or analysis it serves (else a dated deviation): the code items of (a) 7, (b) 5, (c) H1 clause 2, (d) 1–2 and
  (e) 4–5; `tests/test_cascade_call_graph.py` and the score card's coverage block (`decisions.coverage`,
  `vs_baseline.llm_calls_avoided`) and `s1_cached_frac` land with this amendment.

- Cost: none (writing and CPU); H4's downstream runs are within the training budget (2 × 200 Flash episodes).

## Amendment A7 — Stage 2 (registered 2026-10-02; lives in `docs/PREREG_A7_STAGE2.md`)

Amendment A7 (Stage 2) lives in `docs/PREREG_A7_STAGE2.md`. It was registered on 2026-10-02 (first committed in
`eab3d1f`, 20:20 UTC; on this branch since `034b4f4`), before any Stage-1 main-run result was read, and was kept in its
own file because nothing in this file could change while the Stage-1 main run was in flight (`docs/SCOPE_V2.md`). It
continues this registration's amendment sequence (A7, A7.1–A7.19 so far, and its own dated Deviations section) under
the same append-only rule. Stage 2 is a separate confirmatory family on fresh data with its own family-wise α; it shares
no α with the eleven Stage-1 primaries and changes no arm, size, seed, margin, threshold, test or Holm family of A1–A6.22.
Its generated effective view is `docs/STAGE2_EFFECTIVE.md` (`scripts/stage2_spec.py`); `docs/PREREG_EFFECTIVE.md`
covers Stage 1 only. Later Stage-2 changes go into `docs/PREREG_A7_STAGE2.md`, not here.
