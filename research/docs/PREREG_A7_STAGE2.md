# Amendment A7 — Stage 2: the best-practice profile and its confirmation on fresh data

Status: **registered 2026-10-02, before any Stage-1 main-run result was looked at** (written and committed on
2026-10-02 while the API main run, started 17:01 UTC, was in flight; the commit that adds this file is the time stamp).
A7 continues the amendment sequence of `docs/PREREGISTRATION.md`
(A1–A6.22). It lives in its own file because nothing in `docs/PREREGISTRATION.md` may change in the main working tree
while the Stage-1 main run is in flight (`docs/SCOPE_V2.md`); after that run it is referenced from the registration's
amendment list, and this file stays append-only in the same way: later changes are dated sub-amendments (A7.x) or
dated entries in the Deviations section at the end, never edits.

**Blindness.** A7 was written from the registration, the configs, the code, the calibration fits
(`results/calibration/`), the development rounds on seeds 0–999 and the retired-seed pilots (`results/pilot_jev/`).
No Stage-1 main-run output (E1, E2a, E2a′, E2b, E2c, τ², E6 or hard-task runs) was opened, read or summarised
before this commit; the only contact was a listing of the top-level directory names under `runs/` while locating the
development pilots (`runs/pilot_jev_n50_split`, read for the cost variances of A7.5). The selection rule below reads those outputs only through `scripts/select_profile.py`,
after the Stage-1 runs it names are complete.

**What A7 does not change.** No arm, size, seed, margin, threshold, test or Holm family of Stage 1 (A1–A6.22) changes.
The eleven Stage-1 primaries are analysed and reported exactly as registered, whatever Stage 2 shows. Stage 2 is a
separate confirmatory family with its own family-wise α; it shares no α with Stage 1.

**What A7 adds.** (A7.1) a mechanical rule that turns Stage-1 results into the best-practice profile of
JevAgentHarness; (A7.2) fresh confirmatory data; (A7.3) the arms on it; (A7.4) four confirmatory hypotheses with decision
rules, plus secondary rows; (A7.5) power; (A7.6) execution; (A7.7) budget; (A7.8) slip rule and open items.

Terms. *JevAgentHarness* is `harness/agent.py` (`JevAgent`) run with a *profile*: a per-decision-point mapping
(mode, τ, k, mode-specific keys) loaded from `configs/harness/*.yaml` (`docs/JEVAGENT_HARNESS.md`). *JevAgentRL* is our
post-trained decision model: **the checkpoint frozen under `docs/TRAINING_RECIPES.md` §18.4, recorded in
`results/gpu_round5/`**, served in the Jev wire format by `training/serve_jev.py`; it is frozen by development gates
only (development and training seeds; no evaluation seed and no Stage-2 seed). *Jev* is the pinned
`typesafe/jev-1.13` endpoint of §1 (`{type: jev, provider: openrouter}`, A6.20 (c)).

---

## A7.1 — The selection rule for the best-practice profile

The profile is chosen per decision point from the Stage-1 main-run results, by the rule below, once, and written to
`configs/harness/jevagent_best.yaml` by `scripts/select_profile.py` before any Stage-2 run. Nothing else in Stage 1
feeds it, and no Stage-2 number ever feeds back into it.

- **(a) Decision-point groups.** The rule chooses one setting per *group* (a set of `DP`s of `harness/config.py`
  whose modes the registered arms set together):

  | group | decision points (`DP`) | why grouped |
  |---|---|---|
  | **tool** | `tool_select`, `tool_params` | argument questions are asked only inside System-1-chosen calls (A6.22 (a)); the registered arms set the two together |
  | **stop** | `stop` | |
  | **reflection** | `reflection` | |
  | **routing** | `routing` | |
  | **memory** | `memory_write`, `memory_read` | the registered System-1 memory arm sets both |
  | **action** | `action` | discrete-action environments (ALFWorld) only |

  The rule runs separately for each System-2 model with a Stage-1 reference: **Flash** (`gemini-3.5-flash`, the
  primary section of the profile) and **Flash-Lite** (`gemini-3.5-flash-lite`, the secondary section). System-1 is Jev
  throughout; JevAgentRL inherits the Jev profile with the threshold rule of (f).

- **(b) Candidate arms (exhaustive).** Exactly these Stage-1 arms are read. Each is paired with the LLM-only reference
  *of its own run directory* on the first min(n_arm, n_ref) tasks of the frozen evaluation list (A6.13; E6: the 134
  games of its own suite), and each carries exactly one registered (mode, τ, k) per decision point, fixed before any
  evaluation data existed.

  Flash (System-2 `gemini-3.5-flash`):

  | group | candidate arm | config (run) | reference | n | registered setting | authority rank |
  |---|---|---|---|---|---|---|
  | tool | `jev_ts_cascade_tauhat` | `jevgym_tool_select.yaml` (E2a) | `llm_only` | 500 | tool_select cascade τ 1.00, tool_params cascade τ 0.99 (τ̂\*, A6.5 / A6.11 / A6.19) | 4 |
  | tool | `jev_ts_assist_k5` | E2a | `llm_only` | 200 | tool_select assist k 5 | 2 |
  | tool | `jev_ts_assist_k3` | E2a | `llm_only` | 500 | tool_select assist k 3 | 2.5 |
  | tool | `jev_ts_guard` | E2a | `llm_only` | 200 | tool_select guard τ 0.5 | 1 |
  | tool | `jev_ts_replace` | E2a | `llm_only` | 200 | tool_select replace | 5 |
  | tool | `jev_ts_params_replace` | E2a | `llm_only` | 200 | tool_select replace, tool_params replace | 6 |
  | stop | `jev_stop_guard` | `jevgym_meta_control.yaml` (E2b) | `llm_only` | 500 | stop guard τ 0.5 | 1 |
  | stop | `jev_stop_trigger` | E2b | `llm_only` | 200 | stop trigger τ 0.9 | 3 |
  | reflection | `jev_reflect_trigger` | E2b | `llm_only` | 200 | reflection trigger τ 0.5 | 3 |
  | routing | `jev_route_tau0.4` | E2b | `llm_only` | 500 | routing trigger τ 0.4 (A6.15), small model Flash-Lite | 3 |
  | memory | `jev_mem` | E2b | `llm_only` | 200 | memory_write trigger τ 0.5, memory_read assist k 3 | 3 |
  | action | `jev_action_assist_k3` | `alfworld_action_only.yaml` (E6) | `llm_react` | 134 | action assist k 3 | 2.5 |
  | action | `jev_action_replace` | E6 | `llm_react` | 134 | action replace | 5 |

  Flash-Lite (System-2 `gemini-3.5-flash-lite`):

  | group | candidate arm | config (run) | reference | n | registered setting | authority rank |
  |---|---|---|---|---|---|---|
  | tool | `jev_ts_cascade_tau0.9_s2lite` | E2a | `llm_only_s2lite` | 500 | tool_select cascade always-defer, tool_params τ 0.13 (C1L τ̂\*); delegates nothing by construction (A6.22 (a) 4), so it never passes (d) 2 | 4 |
  | tool | `jev_ts_assist_k3_s2lite` | E2a | `llm_only_s2lite` | 500 | tool_select assist k 3 | 2.5 |
  | tool | `jev_ts_guard_s2lite` | E2a | `llm_only_s2lite` | 200 | tool_select guard τ 0.5 | 1 |
  | stop | `jev_stop_guard_s2lite` | E2b | `llm_only_s2lite` | 500 | stop guard τ 0.5 | 1 |
  | reflection | `jev_reflect_trigger_s2lite` | `baselines_jevgym.yaml` (E2c) | `llm_only_s2lite` (E2c) | 500 | reflection trigger τ 0.5 | 3 |

  Flash-Lite has no routing, memory or action candidate (no registered Jev arm with a Flash-Lite reference): those
  groups stay with the LLM in the Flash-Lite section.

  Authority rank orders the modes by how much a decision is left to System-1 (lower = more conservative): guard
  (System-1 can only veto) < assist (System-1 prunes, the LLM decides; a larger k is more conservative) < trigger
  (System-1 decides whether a module runs) < cascade (System-1 decides when confident) < replace (System-1 always
  decides) < tool-and-argument replace. LLM-only (the group stays with the LLM, mode `off`) is more conservative than
  every candidate.

- **(c) Not read, and why.** Every τ-sweep arm (`jev_ts_cascade_tau*`, `jev_route_tau{0.2, 0.6, 0.8}`, E2c's
  `jev_route_tau*`, E6's `jev_action_cascade_tau*`): choosing among swept thresholds by their evaluation results would
  set a threshold from evaluation data, which (f) forbids; a cascade enters only at its calibration fit τ̂\*. Arms that
  repeat a candidate's setting in another config at a smaller n (E2c's Flash `jev_ts_assist_k3`, `jev_ts_replace`,
  `jev_ts_assist_k5`, `jev_reflect_trigger`, `jev_stop_trigger`, `jev_stop_guard`, `jev_stop_guard_s2lite`; E2b's
  `jev_reflect_trigger_s2lite_e2b`): each setting is read once, from the run with the larger registered n (E2a, then
  E2b, then E2c on a tie of n). Shadow / measurement arms, references and the non-System-1 modes (`always`, `llm`),
  controls and non-Jev System-1s (`lite_*`, `embed_*`, `bm25*`, `tfidf_*`, `heuristic_*`, `prior_*`, `random_*`,
  `rate*`, `randscore_*`, `s2self_*`), Intern-Decision arms (the profile is Jev's; (f) covers JevAgentRL), arms that set
  more than one group (`jev_action_replace_reflect`), the H3 ablation `jev_ts_params_replace_answercopy`, the H2 pair
  `jev_memw` (its read side is the non-System-1 `always` reference), and every τ², hard-task, E1 and development run.

- **(d) Qualification.** A candidate c of group g qualifies iff all four hold, computed on its pair exactly as
  `scripts/primaries.py` computes a JevGym pair (`load_arm`, `task_table`, `pair_tables`: the latest record per task
  decides, paired complete case, the A6.22 (e) error taxonomy):
  1. *Complete.* Neither side of the pair has a missing task or an errored task with retries outstanding (fewer than
     three attempts, §1 / A6.22 (e) 4); tasks errored after three attempts leave the pair by the drop rule.
  2. *Delegating.* Replace and cascade candidates: the arm's delegation is non-negligible by A6.20 (e) / A6.22 (a) 5
     (`primaries.delegation_class(tool_coverage, arg_coverage) == "non-negligible"`; ALFWorld: the action entry of
     `metrics.delegation_ratio(...)["by_dp"]` ≥ 5 %). Assist, guard and trigger candidates: at least one executed
     (non-shadow) System-1 decision at the group's decision point (A6.20 (e): they act at every opportunity).
  3. *Non-inferior.* `metrics.ni_pvalue(ref_y, arm_y, m)["p_decisive"] ≤ 0.05` (rule `bb_ordered`, A6.21 (a); the
     `noninferior` qualifier of A6.18 (c) at the unadjusted level), m = 3 pt (JevGym and ALFWorld, §2).
  4. *Cheaper.* The one-sided paired t-test of the per-task cost difference d_t = c_ref(t) − c_arm(t) over the paired
     tasks, `metrics.paired_t_test(d)["p_greater"] ≤ 0.05`. Cost c is the episode ledger's `usage.total.cost_usd` at
     the frozen list prices (`configs/prices/models.yaml`; Jev at OpenRouter's reported `usage.cost`), System-1 calls
     included, shadow / counterfactual / probe calls excluded (`harness/agent.py::usage_summary`).
- **(e) Choice per group.** Let Q_g be the qualifying candidates of group g and s(c) = 1 − mean c_arm / mean c_ref
  over c's pairs (`primaries.cost_block`'s `cost_savings`).
  1. Q_g empty → **the group stays with the LLM** (every decision point of g `off`).
  2. Otherwise let b maximise s over Q_g, and T = {c ∈ Q_g : s(c) ≥ s(b) − 0.02} (the 2-pt tie width of C7, A5.4 (b)).
     The selected setting is the candidate in T with the lowest authority rank (ties: the larger n, then the arm name in
     byte order). A more aggressive mode is therefore selected only when it saves more than 2 pt more than every more
     conservative qualifying mode.
  3. Calls avoided (`vs_baseline.llm_calls_avoided`, A6.22 (a) 5), latency savings and Δsuccess are printed for every
     candidate; they do not enter the choice.
- **(f) Thresholds come from fits or from the registration, never from evaluation data.** The selected setting is
  copied verbatim from the candidate's registered decision-point blocks (mode, τ, k and mode-specific keys such as
  `always_defer`; arm-level keys — `name`, `n_tasks`, `llm`, `s1`, `measure` — are not part of a setting): for a cascade that is the calibration fit τ̂\* (A6.5, A6.11 (c), A6.19; seeds 5000–5149); for
  guard / trigger / assist the τ or k fixed before the main runs. The rule picks among these fixed settings and
  never tunes one. **JevAgentRL:** when the profile's tool group is a cascade, JevAgentRL's τ̂\* is fitted per
  decision point and System-2 model on its own calibration shadow arms (`configs/experiments/jevgym_shadow_cal_rl.yaml`,
  seeds 5000–5149, n = 150, identical to `jev_shadow_cal` / `jev_shadow_cal_s2lite` except for the System-1) with
  the registered command (`threshold_theory --calibration-arm --task-value 1.0 --lam-per-hour 0 --v-fallback W`,
  A6.11 (c), A6.19; a fit that accepts no level → `always_defer: true`), and written into the profile's
  `backends.jevagent_rl` block before any Stage-2 JevAgentRL arm runs; the same fits set the τ of the JevAgentRL
  cascade arms of the E2a mirror (A7.3 (c)). Every other mode keeps the profile's τ / k for JevAgentRL.
- **(g) The profile, the strongest single mode, degenerate cases.**
  1. *Profile.* For each System-2 model the profile is the union of the selected groups' settings; every other
     decision point is `off`. Harness defaults are those of the source configs (HarnessConfig defaults: `max_steps`
     30, `history_window` 8, `guard_max_retries` 1; on ALFWorld E6's `max_steps` 50, `history_window` 12).
  2. *JevGym-applicable groups* are tool, stop, reflection, routing and memory; the action group applies only to
     action environments (A7.1 (i)).
  3. *Strongest single-mode arm* (S2-2's reference): the Flash candidate in the union of the JevGym-applicable Q_g
     with the largest s (exact ties: lower authority rank, larger n, arm name). It is recorded in the profile
     (`best_single`) and run unchanged on the fresh seeds.
  4. *Single-group profile.* If the Flash profile has fewer than two JevGym-applicable groups not `off`, the composed
     harness *is* a single mode: S2-2 enters the Stage-2 family with p = 1 and the label *composition not tested (the
     selected profile delegates at one decision-point group)*, and the `best_single` arm is not run.
  5. *Empty profile.* If no JevGym-applicable group of the Flash section qualifies, JevAgentHarness[profile] is the
     LLM-only harness: S2-1, S2-2 and S2-4 enter with p = 1 and the label *not tested (no decision point qualified in
     Stage 1; the best-practice profile is LLM-only)*, their fresh arms are not run, and S2-3 is tested as registered.
     The same holds for the Flash-Lite section and S2-1L.
- **(h) Code and timing.** `scripts/select_profile.py` implements (a)–(g) with the `scripts/primaries.py` loaders and
  blocks, and writes `configs/harness/jevagent_best.yaml` (the profile: both System-2 sections, `best_single`, the
  τ² and ALFWorld renderings of (i), `status: selected`) and `results/stage2/profile_selection.json` (every candidate
  with its pair size, drop counts, coverage, NI p, cost p, savings, calls avoided and the reason for its fate; the
  provenance of every file read: path, sha256, record count; the commit). It is tested on fixture runs
  (`tests/test_select_profile.py`). It runs **once**, after the three passes of E2a, E2b and E2c are complete (and E6's,
  if E6 is complete by then; otherwise the action group stays `off` and S2-L3b is not run), and before any Stage-2
  run; both outputs are committed before the first Stage-2 episode. Running it is the registered look at those Stage-1
  results that the selection needs (like the C7 selection, A5.4 (a)); it prints no Stage-1 test result.
  `--allow-simulated` / `--allow-dev-tasks` exist for dry runs only and mark the outputs `SIMULATED` / `DEV`.
- **(i) Rendering the profile on the L3 harnesses** (a fixed translation, written by `select_profile.py`; no threshold
  is refit on L3 data: the profile ships as the default of a new deployment would).

  | Flash profile group (JevGym) | τ² (`configs/tau2/stage2/jevagent_best.yaml`, `JevTau2Agent`) | ALFWorld (`configs/experiments/stage2_alfworld.yaml`) |
  |---|---|---|
  | tool: cascade (τ_ts, τ_tp, `always_defer`) | tool_select / tool_params cascade at the same τ (E2-carried; A5.4 (c)'s refit is not used) | n/a (no tool calls) |
  | tool: assist k / guard τ / replace / tool-and-argument replace | tool_select assist k / guard τ / replace (+ tool_params replace) | n/a |
  | stop: guard τ | stop guard τ (verifies `transfer_to_human_agents`) | n/a (no finish proposal) |
  | stop: trigger | `off` (the user simulator ends τ² conversations; `JevTau2Agent` supports stop off / guard / shadow only) | n/a |
  | routing: trigger τ | routing trigger τ, `small_llm: gemini/gemini-3.5-flash-lite` | routing trigger τ (small model Flash-Lite) |
  | reflection: trigger τ | `off` (not implemented on τ²) | reflection trigger τ |
  | memory: settings | `off` (not implemented on τ²) | memory settings |
  | action: assist k / replace | n/a | action assist k / replace |
  | — | `policy_guard` `off` (a τ²-only decision point, outside the JevGym-selected profile) | — |

  The τ² rendering lives in `configs/tau2/stage2/` so that it never joins the Stage-1 τ² arm configs
  (`configs/tau2/*.yaml`, which the effective registration lists for C7).
  τ²: System-1 `{type: jev, provider: openrouter}`, cache `cache/s1_stage2_tau2.sqlite`, `guard_max_retries` 1. A
  rendering with every decision point `off` is not run, and its secondary row reads *not run (the profile has no
  decision point this harness exposes)*.

---

## A7.2 — Fresh confirmatory data

- **(a) JevGym seeds 300000–300499.** 500 tasks of the frozen Stage-1 suite settings (A1 / P4: `families: v2`,
  `distractor_style: near`, `n_distractors: 60`, `noise_lines: 20`, `strict: true`), `seed0: 300000`
  (`configs/experiments/stage2_jevgym.yaml`). The task list is frozen in `configs/eval/jevgym_stage2_tasks.json`
  (written by `scripts/freeze_eval_tasks.py --stage2`; task generation only, no episode run; 500 ids, clarify 250 /
  multi 250, sha256 `c167829adccbf843052d6ea8eb207a37eb13e7e1a8281cbddbbac9af976e704e`, with the suite-block and
  generator hashes). Reservation (from 2026-10-02): the block **300000–300999** is never trained on and never piloted;
  300500–300999 is a held-back spare that only a dated A7 deviation can open. `bench/registry.py`
  (`STAGE2_EVAL_RANGES`, `STAGE2_RESERVED_RANGES`, `training_exclusions()`) makes the runner refuse any JevGym suite
  touching the block without `JEV_EVAL_RUNS=1`, and the training guards (`training/harvest.py`'s default
  `--exclude-jevgym-seeds`, `training/data_v1.py`'s forbidden ranges, `training/opd.py`'s seed checks) exclude it
  (`tests/test_registry.py`). Every Stage-2 JevGym arm runs all 500 tasks (one task list, so every comparison pairs).
- **(b) L3: τ² telecom (the fresh L3 set).** Stage 1 runs τ² airline and retail (C7, A5.4) and ALFWorld valid_unseen
  (E6, the split of every ALFWorld pilot), so neither is fresh. The **telecom** domain of the pinned tau2-bench release
  (`v1.0.1-47-g5bfa7e3`, as for airline, A5.4 (c)) has never been run by this project. Its default task set (114 tasks
  expected) is listed from the dataset, without running a simulation, into `configs/tau2/stage2/telecom_task_ids.json`
  (sha256 recorded in Deviations) before the first telecom simulation; every task × 3 trials (pass^k, A6.16 (e)), the
  A6.15 models (agent `gemini/gemini-3.5-flash`, temperature 0, minimal thinking; user simulator
  `gemini/gemini-3.5-flash-lite`), seed 300. Adapter status: `JevTau2Agent` makes no airline / retail assumption, but
  telecom is untested (`docs/TAU2.md`, "Known limitations"; its user simulator calls tools on the user's device).
  Adapter validation uses tau2's `mock` domain and, if the release ships telecom tasks outside the default set, those
  tasks; a telecom task of the frozen list that is touched before the Stage-2 run leaves the list and is reported as a
  deviation. Cost ≈ $0.12–0.15 per simulation (agent ≈ $0.08–0.12 on Flash, as measured on retail, plus the user
  simulator), ≈ $40–50 per arm.
- **(c) L3, second row: ALFWorld valid_seen.** `eval_in_distribution` (`json_2.1.1/valid_seen`, 140 games after
  ALFWorld's own filters, `docs/ALFWORLD.md`) has never been run by this project (no config names it; Stage 1 and the
  pilots use `eval_out_of_distribution`). It is run only if the profile's ALFWorld rendering is not all `off`
  (A7.1 (i)), with E6's System-2 (`gemini-3.5-flash`) and harness defaults (`configs/experiments/stage2_alfworld.yaml`).
  Adapter checks use train-split games only. Cost ≈ $0.03 per game per arm (E6 estimate: ≈ $20 for 6 × 134 games).

## A7.3 — Arms

- **(a) Fresh JevGym (`configs/experiments/stage2_jevgym.yaml`, seeds 300000–300499, n = 500 every arm).**

  | arm | System-2 | System-1 | setting | role |
  |---|---|---|---|---|
  | `llm_only` | Flash | — | every decision point off | reference of S2-1, S2-4 |
  | `jevagent_best` | Flash | Jev | the profile's Flash section (A7.1) | S2-1, S2-2 |
  | `best_single` | Flash | Jev | the strongest single-mode Stage-1 arm (A7.1 (g) 3), unchanged | reference of S2-2 (not run when A7.1 (g) 4 applies) |
  | `jevagent_best_rl` | Flash | JevAgentRL | the profile's Flash section with the `backends.jevagent_rl` thresholds (A7.1 (f)) | S2-4 |
  | `jevagent_default` | Flash | Jev | `configs/harness/jevagent_default.yaml` (development evidence only) | descriptive |
  | `llm_only_s2lite` | Flash-Lite | — | off | reference of S2-1L |
  | `jevagent_best_s2lite` | Flash-Lite | Jev | the profile's Flash-Lite section | S2-1L (secondary) |

  `jevagent_best_rl` is bound to the placeholder backend `jevagent_rl` (`placeholder: true`): the runner skips it until
  the served endpoint is set (`-o 's1_backends.jevagent_rl={type: jev, name: jevagent-rl, base_url: …, model: …}'`), so
  it can never run against a wrong endpoint by default.
- **(b) Fresh L3.** τ² telecom: `llm_only` and `jevagent_best` (the τ² rendering), each 114 tasks × 3 trials, through
  `scripts/run_stage2_tau2.sh`. ALFWorld valid_seen: `llm_react` and `jevagent_best` (the ALFWorld rendering), 140 games
  each. JevAgentRL does not run on L3 (GPU host time; stated as a limitation).
- **(c) JevAgentRL on the Stage-1 data (`configs/experiments/jevgym_tool_select_rl.yaml`, evaluation seeds
  200000–200499, `JEV_EVAL_RUNS=1`).** The Intern-Decision family arms of `jevgym_tool_select_intern.yaml` with
  System-1 = JevAgentRL, same names with `intern` → `rl`, same modes, τ, k, System-2 and n, so every row pairs task by
  task with the Intern-Decision and Jev rows of Stage 1: `rl_shadow` (500), `rl_ts_replace` (200),
  `rl_ts_params_replace` (200), `rl_ts_assist_k3` (500), `rl_ts_cascade_tauhat` (500; τ from the JevAgentRL Flash fit
  of A7.1 (f); `calibration: pending` until then, so the runner skips it, A6.16 (b)), `rl_stop_guard` (500),
  `rl_ts_assist_k5` (200), `rl_ts_guard` (200), the cascade sweep `rl_ts_cascade_tau{0.5, 0.7, 0.8, 0.9, 0.95, 0.99}`
  (200; tool_params τ 0.9), and on Flash-Lite `rl_shadow_s2lite` (500), `rl_ts_assist_k3_s2lite` (500),
  `rl_ts_guard_s2lite` (200), `rl_ts_cascade_tau0.9_s2lite` (500; JevAgentRL Flash-Lite fit) and
  `rl_stop_guard_s2lite` (500). The references (`llm_only`, `llm_only_s2lite`) and the Intern / Jev rows are Stage-1's
  own runs, read in place (pairing across configs on the frozen list, as A6.15 does for C3); the size ladder, the
  uncalibrated ablation and the paired Jev arms of the Intern config are not mirrored.
- **(d) JevAgentRL, E1 static (descriptive).** `configs/decision/bfcl_when2call_rl.yaml`: the E1 config with the one
  backend `jevagent_rl` (served), on the C8 population (`exclude_ids_file` as registered, `batch_by_state: true`);
  compared with the Stage-1 `jev` and Intern-Decision E1 rows item by item (accuracy, AUROC, ECE, McNemar), descriptive.
- **(e) Prices.** JevAgentRL's System-1 price is its amortised GPU cost, measured with `scripts/measure_latency.py`
  (`--gpu-hour-usd` = the host's on-demand price) on the serving host and written to `configs/prices/models.yaml`
  under the served model id before any priced JevAgentRL episode; until then the arm's cost is flagged by
  `unpriced_models` and S2-4 is not analysed. Every other price is the frozen Stage-1 table.

## A7.4 — Hypotheses and decision rules

**The Stage-2 family: four one-sided tests, Holm at FWER α = 0.05** (levels 0.0125, 0.0167, 0.025, 0.05). Each test is
an intersection–union test: it rejects iff every clause rejects, and its decisive p is the maximum of the clause p
values (valid at level α for the union null without adjustment; Berger 1982). A missing or not-tested member enters
with p = 1 and stays in the family.

| test | claim | data | arm → reference | clauses (each one-sided, decisive p = max) |
|---|---|---|---|---|
| **S2-1** | JevAgentHarness[profile, Jev] is non-inferior to LLM-only on success **and** cheaper per episode in cost **and** in latency | fresh JevGym, Flash, n = 500 | `jevagent_best` → `llm_only` | (i) NI on success, m = 3 pt, `ni_pvalue(...)["p_decisive"]` (`bb_ordered`); (ii) cost: `paired_t_test(c_ref − c_arm)["p_greater"]`; (iii) latency: the same on `usage.total.latency_ms` (the critical path of the loop, BENCHMARK_SPEC §4) |
| **S2-2** | composing beats the best single mode: the composed harness is non-inferior to the strongest single-mode arm on success **and** cheaper | fresh JevGym, Flash, n = 500 | `jevagent_best` → `best_single` | (i) NI on success, m = 3 pt (`bb_ordered`); (ii) cost (paired t-test); A7.1 (g) 4–5: p = 1 when not tested |
| **S2-3** | JevAgentRL beats Intern-Decision-4B on live-policy decision accuracy **and** is non-inferior to Jev, m = 5 pt | Stage-1 evaluation seeds, first 200 frozen tasks, E2a, Flash | `rl_ts_replace` → `intern_ts_replace`; `rl_ts_replace` → `jev_ts_replace` (the Intern config's paired Jev arm) | (i) `paired_t_test(acc_RL − acc_Intern)["p_greater"]`; (ii) `paired_t_test(acc_RL − acc_Jev + 0.05)["p_greater"]` |
| **S2-4** | JevAgentHarness[profile, JevAgentRL] is non-inferior to LLM-only on success **and** cheaper | fresh JevGym, Flash, n = 500 | `jevagent_best_rl` → `llm_only` | (i) NI on success, m = 3 pt (`bb_ordered`); (ii) cost (paired t-test) |

Definitions and rules shared by the four:

- **Pairs and drops.** Paired complete case over the arm's expected tasks (§1, A6.9, A6.22 (e)): a task whose latest
  record is an infrastructure error after three attempts leaves the pair; per-arm drop counts, n′ and the dropped-task
  listing are printed, with the full-set sensitivity row (an errored task counts as a failure, and at its logged cost,
  in the arm where it errored). Agent failures are failures.
- **Cost and latency clauses.** Per-task differences reference − arm of `usage.total.cost_usd` /
  `usage.total.latency_ms` (System-1 included, measurement calls excluded); H0: E[d] ≤ 0; Student t with n′ − 1
  degrees of freedom (`metrics.paired_t_test`, `p_greater`; A6.21 (b) for why the t-test and not the sign-flip test);
  savings 1 − C_arm / C_ref with its paired task-bootstrap 95 % CI (B = 10,000, seed 0) are reported with it. Provider
  prompt caching is billed as reported (A6.22 (e) 8); the no-cache price is a sensitivity column. Latency excludes
  tool execution and client back-off (A6.22 (e) 4); JevAgentRL's latency depends on the serving host and is reported
  with its hardware, concurrency and queueing (A6.16 (j)), which is why S2-4 has no latency clause.
- **S2-3's accuracy.** acc_X(t) = the share of task t's executed System-1 `tool_select` decisions (mode `replace`, on
  the arm's own states) that carry an oracle gold and are correct by `metrics.is_correct` against the logged gold (the
  A4 acceptable tool set; `metrics.decision_records`). Pairs: tasks with no infrastructure error and ≥ 1 labelled
  decision in both arms of the comparison (the two comparisons may keep different task sets; each n′ is printed).
  "Live-policy" means every decision is taken on a state the arm's own policy reached; a same-state comparison is the
  shadow arms' (`rl_shadow` vs `intern_shadow`, descriptive). Pooled decision-level accuracy, AUROC, ECE and the
  `tool_params` accuracies of the `*_ts_params_replace` arms are descriptive.
- **Margins.** 3 pt on JevGym success (§2) for S2-1, S2-2, S2-4; 5 pt on live-policy tool-selection accuracy for S2-3's
  NI clause (stated now: within 5 pt of the teacher on the states the policy itself reaches is the claim "JevAgentRL
  matches Jev in the loop"; the development gate G4 of §18.4 used 0.8 × Jev, a much weaker bar).
- **Reading.** A rejected test reads as its claim. A failed test is read clause by clause, using the A6.1 rule for NI
  clauses (two-sided 95 % CI covering 0 → *inconclusive at this n*; entirely below 0 → a loss of the stated size) and,
  for a t-test clause, *no saving established at this n* when the two-sided 95 % CI of the mean difference covers 0
  and *more expensive (slower) by X* when it lies entirely on the wrong side. The MDE at the achieved discordance and
  the test's Holm level is printed for every NI clause (`metrics.ni_mde`) and the normal-approximation MDE for every
  t-test clause. S2-3's NI-clause failure reads "JevAgentRL was not shown to be within 5 pt of Jev at this n".
- **Not tested.** A7.1 (g) 4–5 and a missing input (p = 1, label, ERROR flag); a JevAgentRL that is not frozen and
  recorded in `results/gpu_round5/` by the time the GPU host starts the Stage-2 arms leaves S2-3 and S2-4 *not tested
  (no frozen JevAgentRL checkpoint)*, never run on an unfrozen checkpoint.
- **Iso-quality wording.** "Cheaper at non-inferior success" in the paper means a rejected S2-1 / S2-4; S2-2's claim
  is "composition adds savings over the best single mode at non-inferior success".

**Secondary (registered, outside the family; Holm within their own table of three, worded as secondary):**

| row | claim | data | arm → reference | decision |
|---|---|---|---|---|
| **S2-1L** | the Flash-Lite profile is NI to LLM-only and cheaper | fresh JevGym, Flash-Lite, n = 500 | `jevagent_best_s2lite` → `llm_only_s2lite` | NI (m = 3 pt, `bb_ordered`) ∧ cost (paired t-test) |
| **S2-L3a** | the profile transfers to a fresh domain: NI to LLM-only on majority pass **and** cheaper | τ² telecom, 114 tasks × 3 trials | `jevagent_best` (τ² rendering) → `llm_only` | NI on majority pass (≥ 2 of 3 trials, A6.16 (e)), m = 5 pt (`bb_ordered`) ∧ cost per task (mean over its trials; paired t-test); pass^1 and pass^k descriptive; a failure reads "transfer NI was not established" (A6.16 (e)) |
| **S2-L3b** | the profile's ALFWorld rendering is NI to the ReAct LLM and cheaper | ALFWorld valid_seen, 140 games | `jevagent_best` (ALFWorld rendering) → `llm_react` | NI (m = 3 pt, `bb_ordered`) ∧ cost (paired t-test) |

**Descriptive (no test):** `jevagent_default` vs `llm_only` on the fresh seeds; every arm's §4 report (success with
CI, cost, latency, cost-of-pass, LLM calls, System-1 calls, coverage, calls avoided, the per-task utility ΔU at
W = $1, λ = 0 and its sensitivity rows); the JevAgentRL E2a mirror with the Stage-1 score card (C1–C3 / C6-style
comparisons against `llm_only`, side by side with the Jev and Intern rows; outside every Holm family); E1 static
(A7.3 (d)). The per-task utility U is not a decision statistic of S2-2: one decisive endpoint per test, fixed now.

## A7.5 — Power (planning statements; exact where the NI rule is involved)

- **NI clauses** (`scripts/stage2_power.py` → `results/reviews/2026-10-02/stage2_power.{md,json}`; exact summation with
  the certified rejection regions of the A6.21 rule, the method of `scripts/ni_power_exact.py`). n = 500, m = 3 pt:

  | scenario (reference success) | d | true Δ | floor of its table | 0.05 |
  |---|---|---|---|---|
  | family NI clause (S2-1, S2-2, S2-4), Flash (0.98) | 2 % | 0 | 0.96 (0.0125) | 0.99 |
  | | 2 % | −1 pt | 0.64 | 0.85 |
  | | 4 % | 0 | 0.80 | 0.94 |
  | | 4 % | −1 pt | 0.41 | 0.67 |
  | | 4 % | −2 pt | 0.10 | 0.27 |
  | family NI clause, ceiling reference (1.00), losses only | 1 % | −1 pt | 0.76 (0.0125) | 0.93 |
  | | 2 % | −2 pt | 0.13 | 0.33 |
  | S2-1L (secondary), Flash-Lite (0.80) | 20 % | 0 | 0.26 (0.05/3) | 0.44 |
  | | 20 % | +3 pt | 0.80 | 0.91 |
  | S2-L3a (secondary), τ² telecom majority pass, n = 114, m = 5 pt (0.60) | 15 % | 0 | 0.20 (0.05/3) | 0.37 |
  | | 25 % | 0 | 0.14 | 0.27 |
  | | 25 % | +3 pt | 0.32 | 0.51 |
  | S2-L3b (secondary), ALFWorld valid_seen, n = 140 (0.80) | 10 % | 0 | 0.14 (0.05/3) | 0.29 |
  | | 20 % | 0 | 0.08 | 0.19 |
  | | 20 % | +5 pt | 0.48 | 0.67 |

  The family's intermediate Holm levels (0.05/3, 0.05/2) are in the results file.   Reading: at the Holm floor the Flash NI clause has ≥ 80 % power when the fresh discordance is ≤ 4 % and the true
  difference is 0 (the C1-pilot regime: τ = 0.99 cascade at 2 % discordance), and loses power fast with a true deficit
  (≈ 0.4 at Δ = −1 pt, d = 4 %). The secondary rows are expected to read *inconclusive at this n* unless System-1 adds
  success; they are confirmatory only in direction.
- **Cost and latency clauses** (one-sided paired t-test, normal approximation): 80 % power at level α needs a mean
  per-task difference of (z_{1−α} + z_{0.8}) · σ_d / √n′; at n′ = 500 and the floor 0.0125 that is 0.138 σ_d. On the
  retired-seed / development pilots (`runs/pilot_jev_n50_split`, seeds 0–49, Flash), σ_d of the cost difference was
  $0.013 for the τ = 0.99 cascade (mean saving $0.017 of $0.043, so a 4 % saving, $0.0018, is detectable), $0.0023
  for the stop trigger (a $0.0003, 0.8 %, saving is detectable) and $0.022 for the stop guard; latency σ_d was 10–21 s
  per episode (mean 14–20 s), so a latency saving needs ≥ 1.5–3 s per episode (the τ = 0.99 cascade saved 5.2 s).
  S2-2's cost clause has 80 % power if composition saves ≥ 0.138 σ_d beyond the best single mode (≈ 0.8 % of the
  LLM-only cost for a stop-trigger-sized σ_d).
- **S2-3** (normal approximation on per-task accuracy differences, n′ ≈ 200, σ_d = 0.20 assumed as a planning value
  — a task carries 5–10 tool decisions): the MDE at 0.0125 is 3.08 · 0.20 / √200 = 4.4 pt. Clause (i) has ≥ 80 % power
  when JevAgentRL beats Intern-Decision by ≥ 4.4 pt (development evidence: Intern's own-state tool_select accuracy
  0.41 against Jev's 0.97 on dev seeds 0–9, §18.1); clause (ii) has ≥ 80 % power when JevAgentRL's true deficit to Jev
  is ≤ 0.6 pt, ≈ 45 % at a 2-pt deficit (σ_d = 0.15: ≥ 80 % up to a 1.7-pt deficit). A failure of (ii) at a deficit of
  1–4 pt is expected to read *inconclusive at this n*.
- **Conjunctions.** An intersection–union test's power is at most its weakest clause's; with independent clauses it
  is their product. For S2-1 the cost and latency clauses are near 1 for a cascade-type profile on the pilot effect
  sizes, so S2-1's power is that of its NI clause.

## A7.6 — Execution

- **Order.** (1) Stage 1 complete for E2a, E2b, E2c (three passes each; E6 if run) → (2) `python scripts/select_profile.py
  --runs runs --out results/stage2` → commit `configs/harness/jevagent_best.yaml`, `configs/tau2/stage2/jevagent_best.yaml`
  and `results/stage2/profile_selection.json` → (3) API host: `configs/experiments/stage2_jevgym.yaml` (every arm but
  the JevAgentRL one) → (4) τ² telecom (freeze `configs/tau2/stage2/telecom_task_ids.json` first) and ALFWorld valid_seen
  if run → (5) GPU host, after the JevAgentRL freeze record exists in `results/gpu_round5/`: price measurement (A7.3
  (e)); `jevgym_shadow_cal_rl.yaml` and the τ̂\* fits (A7.1 (f)) written into the profile's `backends.jevagent_rl`
  block and into the RL cascade arms of `jevgym_tool_select_rl.yaml` (commit); then `jevagent_best_rl` on the fresh
  seeds, `jevgym_tool_select_rl.yaml` on the evaluation seeds and `bfcl_when2call_rl.yaml` (steps 3 and 5 can run in
  parallel) → (6) analysis.
- **Seeds and guard.** Every JevGym command on 300000–300499 or 200000–200499 sets `JEV_EVAL_RUNS=1`; nothing else
  does. Development, smoke tests and dry runs use seeds ≤ 999 (`-o suite.seed0=0`); the runner refuses the Stage-2
  block otherwise (A7.2 (a)).
- **Three passes and retries.** Every evaluation command runs three times in total (the first pass and the two
  registered extra attempts, §1 / A6.22 (e) 4); the runner re-runs only tasks whose latest record is an
  infrastructure error. τ²: `suites.tau2 rerun-errored` re-runs an errored simulation up to twice (A6.22 (e) 4) and
  `convert` stamps the attempts. Afterwards an errored task is final and the drop rule applies.
- **Caches.** System-1 answers go through `cache/s1_stage2.sqlite` / `cache/s1_stage2_tau2.sqlite` (new files: no
  Stage-1 or development answer is replayed); System-2 calls are never served from a cache (A6.22 (e) 8).
- **No interim looks.** While a Stage-2 config runs only operational fields are read (errors, cost per episode,
  served model, unpriced models). No Δ, p-value or score card of a Stage-2 test arm before its three passes are
  complete.
- **Analysis.** `scripts/stage2_analysis.py` (slip-rule code item, A7.8) computes the family and the secondary table
  with the `scripts/primaries.py` blocks (`task_table`, `pair_tables`, `ni_block`, `cost_block`,
  `metrics.paired_t_test`, `metrics.ni_holm_levels`), prints provenance (path, sha256, record count of every file),
  refuses simulated inputs and episodes outside the frozen task lists, and writes `results/stage2/stage2.{json,md,tex}`.
  The final retained n′ of every NI clause is added to `scripts/ni_exact_size.py`'s `SETTINGS` and the size table is
  re-run before the analysis is reported (A6.21 (a) 8).

## A7.7 — Budget (estimates from development pilots, never results)

| item | episodes / simulations | API $ (est.) | wall clock (est.) |
|---|---|---|---|
| fresh JevGym, Flash: `llm_only` ($0.043 / ep.), `jevagent_best`, `best_single`, `jevagent_default`, `jevagent_best_rl` (≈ $0.03 / ep. each) | 2,500 | ≈ 82 | ≈ 1.2 h (workers 8); RL arm ≈ 0.6 h (GPU, workers 4) |
| fresh JevGym, Flash-Lite: `llm_only_s2lite`, `jevagent_best_s2lite` (≈ $0.01 / ep.) | 1,000 | ≈ 10 | ≈ 0.3 h |
| JevAgentRL calibration arms (`jevgym_shadow_cal_rl.yaml`, 2 × 150) | 300 | ≈ 8 | ≈ 0.3 h (GPU) |
| JevAgentRL E2a mirror (`jevgym_tool_select_rl.yaml`: 4,000 Flash + 2,200 Flash-Lite episodes) | 6,200 | ≈ 160 | ≈ 5–7 h (GPU, workers 4) |
| JevAgentRL E1 static (3,991 records) | — | 0 (GPU only) | ≈ 0.5 h |
| τ² telecom: 2 arms × 114 × 3 | 684 | ≈ 85–100 | ≈ 3 h (concurrency 4) |
| ALFWorld valid_seen: 2 arms × 140 (if run) | 280 | ≈ 8 | ≈ 1 h |
| **API total** | | **≈ 355–370** | |
| + 20 % contingency (retries, price drift, longer telecom conversations) | | **≈ 430–445** | |
| GPU (one L4, $1.32/h): serving, calibration, E2a mirror, E1, latency measurement | | ≈ 12–15 | ≈ 9–11 h |

Hours: the API part (fresh JevGym, τ², ALFWorld) is ≈ 5–6 h of wall clock; the GPU part ≈ 9–11 h in parallel. Stage 2
fits in about one day after the profile is selected.

## A7.8 — Slip rule and open items

- **Must land before any Stage-2 run** (else the run does not start): this text; `configs/experiments/stage2_jevgym.yaml`
  and `configs/eval/jevgym_stage2_tasks.json` (with this commit); the registry reservation and its tests; the profile
  mechanism (`profile` / `profile_s2` / `profile_part` / `profile_backend` / `profile_env` arm keys, `status: pending` profiles skipped
  by the runner), `configs/harness/jevagent_default.yaml`; `scripts/select_profile.py` with its fixture test;
  `configs/experiments/stage2_alfworld.yaml`, `configs/experiments/jevgym_tool_select_rl.yaml`,
  `configs/experiments/jevgym_shadow_cal_rl.yaml`, `configs/decision/bfcl_when2call_rl.yaml`,
  `scripts/run_stage2_tau2.sh`.
- **May land later, before the analysis it serves** (else a dated deviation): `scripts/stage2_analysis.py` and its
  tests; `configs/tau2/stage2/telecom_task_ids.json` (before the first telecom simulation).
- **Decided by A7 and not reopened by results:** the candidate table, the qualification and choice rules, the
  translations of A7.1 (i), the family, the margins and the clause tests.
- Cost of A7 itself: CPU only (the power table took 10 s).

## Deviations (dated entries only)

- 2026-10-02, recorded about 22:20 UTC. **JevAgentRL operator exposure before the barrier (A7.10 (a)).** At about
  20:40 UTC, before the A7.10 barrier text existed and before it reached the operator, the JevAgentRL operator session
  printed the schema of the first record of one Stage-1 trajectory file
  (`runs/e2a_jevgym_tool_decisions/jev_ts_replace*.jsonl`) to learn the log format. It saw the key names, one episode's
  usage block and the first decision-log entry (option list, s1_value, part of s1_dist). It opened no aggregate, score
  card or other record. Ruling (authors): one record's format and one decision cannot inform a training choice. The
  chain's stage order, configs, seeds and gates were fixed in `results/gpu_round5/TRAINING_MANIFEST.md` (commit
  9945d04), which records the exposure, and they were not changed on it. S2-3 and S2-4 therefore stay confirmatory,
  and the paper's appendix reports this entry verbatim. If a reader judges otherwise, S2-3/S2-4 read as exploratory
  under A7.10 (a); both readings are printed.
- 2026-10-02, recorded about 22:20 UTC. **A7.17 (a) snapshot list.** The first stamp (manifest
  `results/stage2/timestamps/manifest_2026-10-02T2146Z.txt`, receipt in commit 8a94c37) was made before the training
  manifest existed, so the stamped text is left as it is: `results/gpu_round5/TRAINING_MANIFEST.md` at commit 9945d04
  is added to the next stamp (made right after this entry is committed, in `results/stage2/timestamps/`, with this
  file) and to snapshot C's list.
- 2026-10-03, recorded about 06:55 UTC, before any Stage-2 episode. **A7.1 (h) applied: S2-L3b is not run.** E6
  (`alfworld_action_only.yaml`) was not run in Stage 1, so it was not complete when `scripts/select_profile.py` ran
  (2026-10-03 06:28 UTC; outputs committed in 69eff5a). By A7.1 (h) the action group stays `off` *and S2-L3b is not
  run*. The profile's ALFWorld block (the reflection trigger, rendered by A7.1 (i)) is written by the script but not run:
  A7.2 (c)'s "run only if the profile's ALFWorld rendering is not all `off`" is a necessary condition, not a sufficient
  one, and A7.1 (h) is the specific rule for a missing E6. S2-L3b enters the secondary table with p = 1 and the label
  *not run (A7.1 (h): E6 not run in Stage 1)* (`scripts/stage2_analysis.py` reads it from
  `results/stage2/profile_selection.json`). No ALFWorld valid_seen game has been run.
- 2026-10-03, recorded about 06:55 UTC, before any Stage-2 episode. **Arms the config lists but the registration does
  not run.** (i) The Flash-Lite section qualified nothing, so S2-1L enters with p = 1 and its fresh arms are not run
  (A7.1 (g) 5): `jevagent_best_s2lite` is skipped by the profile and `llm_only_s2lite` is dropped from the arm list of
  every `stage2_jevgym` invocation (`scripts/run_stage2.sh`, `EXCLUDE=llm_only_s2lite`; recorded in the START file and
  in `schedule.json`). (ii) No listed control qualified (A7.14 (b), `results/stage2/control_selection.json`), so there is
  no `best_control` run. (iii) The LOGO plan written with the profile (A7.11 (c)): |G| = 2 (tool, reflection), so
  `jevagent_minus_tool` (the reflection trigger alone) runs on the first 200 tasks and `jevagent_minus_reflection` is
  configuration-identical to `best_single`, whose first 200 tasks are reused. These follow the registration; they are
  recorded here because the config lists the arms.
- 2026-10-03, recorded about 07:20 UTC. **S2-1X executed from a side worktree.** S2-1X (the X eligibility smoke, the
  calibration on seeds 5150–5299, the X profile rendering and `stage2_x`) runs from the git worktree
  `<local-path>`, branch `stage2-pricefix` = main + the `PriceTable.get` suffix fix (2526853: a
  price row matches only on a whole path segment, so a bare "flash" can never resolve to `deepseek/deepseek-v4-flash`;
  every model id logged so far resolves as before) + the DeepSeek V4 list prices (re-read on OpenRouter 2026-10-03 06:46
  UTC, unchanged from A7.12 (a)), while the Stage-1 τ² main run occupied the main working tree (SCOPE_V2: no change under
  `src/` or to existing configs there while it runs). The exact commits are those in `results/stage2/START_*_x*.txt`
  (the smoke started at bf70f85). Its run directories and System-1 cache are the registered relative paths inside the
  worktree (`runs/stage2_dev/x_eligibility`, `runs/e2a_jevgym_shadow_cal_x`, `runs/stage2_x`, `cache/s1_stage2.sqlite`,
  a file separate from the main tree's; replays are charged at their original cost and latency either way, A7.16 (b)),
  and the same launch gate (`scripts/stage2_spec.py --gate stage2_x`). The branch is merged into the main branch after
  the τ² main run ends and the run directories are copied to the main tree for the analysis. No change to arms, seeds,
  n or rules.
- 2026-10-03, recorded about 07:20 UTC. **Telecom sample frozen (A7.15 (c), (d)).** Adapter validation (the 32 reserved
  tasks, both arms, one attempt; `runs/stage2_tau2_validation`) completed with no infrastructure error, side-car logs for
  all 64 simulations and System-1 calls in the rendering arm. Validation spend V = $10.928951 (agent and System-1 ledger
  cost + user simulator), ĉ = $0.170765 per simulation, so n = min(800, 2285 − 32, ⌊(250 − V) / (2 × 1.25 × ĉ)⌋) = 560;
  `configs/tau2/stage2/telecom_task_ids.json`, sha256 `7c352a9b556d46d670b531af5d05a5afa01e2c1c6df31994b05cc7efcf5f3702`;
  the exact power at n = 560 is added to `results/reviews/2026-10-02/stage2_power_amended.md` (`--telecom-n 560`).
  Disclosure: while checking the validation run's progress, the tail of tau2's console log showed one validation
  simulation's evaluation panel (its partial action reward) on screen; no success number was computed or used, n
  follows from V and ĉ only, and validation tasks are never evaluated. Later progress checks count lines only.
- 2026-10-03, recorded about 07:25 UTC. **S2-1X capability manifest written after the eligibility smoke (A7.12 (a)).**
  A7.12 (a) asks for a committed capability manifest per pair before selection; none existed when the smoke ran
  (07:11–07:19 UTC; the facts verified for pair (1) on 2026-10-02 are in A7.12 (a) itself). It was written right after
  the smoke from the OpenRouter listings read at 06:46 UTC and the Hugging Face model records read at 07:20 UTC
  (`results/stage2/x_capability_manifest.json`: ids, snapshot slugs `-20260423`, weights revisions, licence MIT, the
  pinned StreamLake fp8 endpoint, prices, context, request settings) and committed with the cost gate before the first
  calibration episode. Pair (1) is eligible (0 malformed responses, no error left, every call served by StreamLake) and
  the gate gives n = 500 (predicted $5.71); pair (2) was not needed. Nothing in the manifest bears on eligibility.
- 2026-10-03, recorded about 17:45 UTC, before any Stage-2 JevAgentRL episode and before any S2-3 source episode.
  **JevAgentRL rows: calibration as run, the serving host and the attribution rows.** (a) *Calibration (A7.1 (f)).* The
  JevAgentRL τ̂\* fits were made by the operator after the freeze (freeze 2026-10-03 10:17:50 UTC,
  `results/gpu_round5/FREEZE_RECORD.json`) on seeds 5000–5149 with the registered command and arms, from
  `configs/experiments/jevgym_shadow_cal_rl.yaml` as run (merged in 487e212): System-1 under the served name
  `jevagentrl-4b`, workers 8 and no System-1 cache, where the registered file had `jevagent-rl`, workers 4 and
  `cache/s1_stage2.sqlite`. Workers and cache are operational only (shadow arms; the LLM decides every step); arms,
  seeds, n, models and the fit command are unchanged. The fits (`results/gpu_round5/cal/`; Flash tool_select τ̂\* 0.5,
  i.e. every choice accepted, tool_params 0.99999988) are written into the profile's `backends.jevagent_rl` block by
  `scripts/fill_backend_tau.py --fits-dir results/gpu_round5/cal` and committed before any Stage-2 JevAgentRL arm runs;
  they are not refitted. (b) *Host.* A7.7 and A7.10 (c), (h) planned JevAgentRL's serving and the in-process
  Intern-Decision-4B on one L4; the remaining GPU chain runs on the persistent Stage-2 host instead (the g6e.2xlarge, one
  L40S 48 GB, that served StartLux; `scripts/aws/stage2_gpu.sh`), still one model on the GPU at a time. The served
  model is the freeze record's (base `Qwen/Qwen3-4B` at its recorded snapshot `1cfa9a72…`, the adapter and temperature
  file sha256-checked against the record before any request, `serve_jev` code identical to the serving commit c6c1cc1).
  S2-4's tariff q is measured on this host by the committed plan `results/stage2/tariff_rl_plan.json` (dev seeds
  600–699, as the S2-4S tariff); its hardware is reported with it. The host waits for this session to commit
  `results/stage2/tariff.json` before S2-4 and the frozen corpus before any replay (a timeout would be recorded here).
  (c) *Rows of A7.13 (i), from the freeze record.* The untrained base (with its v1-dev temperature fit,
  `results/gpu_round5/base_calibration.json`), SFT v1 and the frozen model (SFT v1 → GRPO iteration 15). SFT v1 + OPD is
  not a row: OPD does not precede the frozen checkpoint (dropped by the chain), so its row reads *not run (dropped by
  the chain)*. The freeze record lists no temperature file for SFT v1, so it is served without one (replace-arm, corpus
  and E1 accuracies are argmax measures and do not depend on it; its ECE is uncalibrated, stated). Served names
  `jevagentrl-4b`, `rl-base`, `rl-sft-v1` carry exact-0 price rows (A7.13 (iii): the GPU cost enters only through q;
  the attribution rows report API cost only). (d) *Order.* `stage2_rl_states` runs its RL arm first while the server is
  up (arm list restricted to it), then the Jev arm (workers 8) and the Intern-Decision arm (workers 4, the RL server
  stopped) as in the config header; every source arm has its three passes before the corpus is built. (e) *StartLux
  replays (A7.18 (e) 2).* The 4B and 27B (Q8_0) serve the revisions of the committed manifest (`PIN_MANIFEST`); the
  ladder sizes (0.8B, 2B, 9B; descriptive, GPU time only) record their revisions on the host
  (`results/stage2/gpu/manifest_replay_<size>.json`, synced to S3 before their replay) and are committed afterwards.
- 2026-10-03, recorded about 19:05 UTC, before any attribution episode and before any replay answer. **SFT v1's
  temperatures (A7.13 (i)) and the frozen corpus.** (a) The freeze record lists temperature files for the base and the
  frozen model only. So that the attribution rows are comparable, SFT v1's temperatures were fitted by the evaluation
  session after the freeze, on v1 dev only (`data/train_v1/dev.jsonl.gz`, 2,377 decisions, sha256 `9301ae53…` as in
  `data/train_v1/MANIFEST.json`), with the command and settings that produced `final_calibration.json` and
  `base_calibration.json` (`training/calibrate.py --min-bucket 200`, the base at the recorded snapshot, the training code
  identical to c6c1cc1), on the Stage-2 host while its GPU was otherwise idle. The same command refitted on the frozen
  model first and reproduced `results/gpu_round5/final_calibration.json` exactly (relative difference 0,
  `results/stage2/rl_attr/reproduction_check.json`). The fit is `results/stage2/rl_attr/sft_v1_calibration.json`
  (sha256 `b73c7b8b…`), and SFT v1 is served with it; this replaces the "served without one" of the previous entry. No
  evaluation output was read for it. (b) The common-state corpus C was built on the host from the three passes of the
  three `stage2_rl_states` source arms and `stage2_jevgym`'s `llm_only` (first 200 tasks): 200 eligible tasks, none
  excluded, 5,314 occurrences, 3,939 distinct questions. It is committed (`results/stage2/s23/`) before any backend
  answers a replay question.
- 2026-10-04, recorded about 15:30 UTC, after the C7 analysis and the freeze, before any A7.13 (iv) replay answer.
  **The τ² reference-agreement corpus (A7.13 (iv)) as executed** (`scripts/tau2_reference_replay.py`, corpus
  `results/stage2/t2ref/`). (a) *Set.* "The System-1 questions logged by the Stage-1 τ² arms" is read as every
  tool_select / tool_params decision record that System-1 was asked (mode not `off`; no always-defer record exists) in
  the final trajectory of each task and trial of every arm file under `runs/tau2/{airline_test,retail}/`; the `off`
  records of `llm_only` and `jev_guard` log the state for measurement and were not asked
  (2,644 airline, 3,979 retail tool_select records). Jev's live answer is logged on every asked record.
  (b) *Questions.* Rebuilt as the adapter asked them: the logged state, question text and options; tool_select's
  descriptions from the τ² tool schemas in the adapter's order (checked against every logged option list: 0
  mismatches); no descriptions on tool_params. Keyed by a canonical hash; only labelled questions are replayed:
  22,704 distinct questions (sha256 `30601beb…`, on S3 because of its size; the labels stay in
  `occurrences.jsonl.gz`, sha256 `9fefe8fe…`, and never reach the GPU host). (c) *Coverage* (labelled / asked
  occurrences): tool_select 11,992 / 11,992 (airline test) and 7,942 / 7,942 (retail); tool_params
  1,709 / 3,363 and 2,314 / 3,973 (an argument label exists only when the executed tool is the reference next
  step, A6.22 (b) 2). (d) *Replay.* A7.10 (c)'s path (the harness tournament, one question per request, its own
  cache, two retries, then a disagreement); the three rows in parallel on one host, one GPU each (Intern-Decision-4B
  in two hash shards). (e) *Readout (descriptive).* Agreement per row, domain and decision point as the mean over τ²
  tasks (trials and arms pooled) of the mean over the task's labelled occurrences, with a task-bootstrap 95% interval
  (10,000 resamples); paired differences frozen − Intern and frozen − base; Jev's logged live answer on the same
  occurrences; the most frequent label's rate (tool_select) as a floor. Not oracle accuracy and not evidence of τ²
  success or savings.
- 2026-10-04, recorded about 16:10 UTC, before any JevAgentRL-v2 calibration, tariff or evaluation episode and before
  any v2 replay answer. **The JevAgentRL-v2 rows of A7.19 (f) as they will be executed** (prepared without reading any
  Stage-1 or Stage-2 result table; `results/stage2/v2_eval_plan.json`, `scripts/aws/stage2_gpu.sh` stages `v2_*`).
  (a) *Model and serving.* The frozen checkpoint of `results/gpu_round6_miles/final_FROZEN.txt` (2026-10-04 00:53:20
  UTC, before the A7.19 (e) deadline; SFT v2 on StartLux-Decision-4B, a full fine-tune in Hugging Face format), every
  sha256 line of the record checked on the host before any request, served by `training/serve_jev.py` (code identical
  to the freeze's serving commit 4c0bdf8) with the record's temperature file under the name `jevagentrl-v2` (an exact-0
  price row: its GPU cost enters only through its tariff). StartLux-Decision-4B is a Qwen3.5 hybrid, so serve_jev runs
  in its own venv on the Stage-2 host: the harness venv's stack plus flash-linear-attention 0.5.2 and causal-conv1d
  1.7.0 (the A7.18 (d) StartLux environment's versions); the harness venv and the vendor environment are not changed.
  Before the first request the host must reproduce the freeze-time option probabilities of the Miles image on the
  freeze's 64-row regression corpus within the freeze's tolerance (max |Δp| ≤ 0.02), else nothing is served and the
  rows read *not run* with the reason. (b) *Calibration (as A7.18 (e) 4).* `v2_shadow_cal`
  (`configs/experiments/jevgym_shadow_cal_v2.yaml`: Flash; shadow at tool_select, tool_params and stop; seeds
  5000–5149; three passes), fitted per decision point with the registered command; the fits fill a
  `backends.jevagentrl_v2` block of the selected profile (`scripts/fill_backend_tau.py`; cascade τ only). The block was
  added now with `calibration: pending` by select_profile.py's own rule (`fill_backend_tau.py --add-pending`), so no v2
  profile arm can run before the fit (an arm without a block would take Jev's thresholds); nothing else in the profile
  changed. (c) *Tariff (A7.13 (iii)).* `results/stage2/tariff_v2_plan.json`, committed before the measurement: dev seeds
  590–599 warm-up and 600–699 workload, as the S2-4 and S2-4S tariffs; the host is priced by its instance type at the
  measurement (the plan lists g6e.2xlarge $2.24208 / h and g6e.16xlarge $7.57719 / h, each one L40S 48 GB; another type
  is refused); on a type other than g6e.2xlarge the same window priced at g6e.2xlarge is a sensitivity row. (d) *Rows.*
  The live replace arm `v2_ts_replace_f` (`configs/experiments/stage2_v2_states.yaml`, 300000–300199, three passes),
  the profile row `jevagent_best_v2` (`configs/experiments/stage2_v2.yaml`, 300000–300499, three passes) and the corpus
  replay `--name v2` (its own replay cache). The profile row has its own config and run directory (`runs/stage2_v2`)
  instead of an arm added to `stage2_jevgym.yaml`, whose registered arm list is not edited; every setting other than
  System-1 is `stage2_jevgym.yaml`'s, workers 4 as S2-4 / S2-4S, and it is read beside `stage2_jevgym`'s `llm_only`.
  System-1 timeout 60 s as the StartLux backends; new System-1 caches. (e) *Order and gates.* Calibration; the host
  waits for this session to commit the fits byte-identical and the filled block; the tariff; the host waits for the
  committed `results/stage2/tariff_v2.json`; then the profile row, the live arm and the replay. A gate timeout stops
  the v2 chain (no uncommitted threshold or tariff is used). (f) *Budget (A7.19 (g)).* Before each priced stage the host
  sums the rows' Flash spend (every record, errored attempts included) and does not start a stage whose reserve would
  pass $45 (A7.19 (g)'s $100 less the training track's spend: its ledger read $42.43 before its final E2 development
  slice, which is not in it); the cap may be lowered, never raised. (g) *Reading.* `scripts/stage2_analysis.py`
  reports the rows in their own section beside JevAgentRL's, labelled "exploratory (A7.19)" (`stage2_v2/...`), with no
  test and outside every Holm table; S2-3's further-replay rows leave out `answers_v2`. Nothing of v2 is reported under
  the JevAgentRL name.
- 2026-10-05, recorded about 07:10 UTC, before the Sol eligibility smoke and before any Sol episode. **The optional
  `gpt-6.1-sol` row of A7.12 (i) is run now.** (a) *Budget check.* Every registered Stage-2 row is complete
  (`results/stage2/stage2.md`); the one open registered item, the A7.13 (iv) τ² replay, answers on the GPU host and
  needs no API spend. The agent ledgers of every Stage-2 run directory sum to $296.6 (shadow System-1 calls and errored
  attempts included); with the τ² user simulator (outside those ledgers; ≈ 25 % on top of the telecom agent cost by the
  validation's ratio, ≈ $39) Stage-2 API spend is ≈ $336 against the ≈ $550–625 planning budget (A7.10–A7.17 tables),
  so the row's ≈ $70–80 fits after every registered row. (b) *Pair and request settings.* X = `gpt-6.1-sol`, X-small =
  `gpt-6-luna`, OpenAI Responses API called directly (no router, so no provider fallback exists). /v1/models lists no
  dated snapshot of either id, so the id is the pin; every call now logs the API's own `model` and `service_tier`
  (ledger meta `served_model`, `service_tier`: an addition to `src/jevharness/llm/openai_responses.py` made for this
  row, no other change), and a served model other than the id is a deviation. Sol accepts reasoning effort low to max
  only (none and minimal answer 400), so `low`, its lowest, stands in for A7.12 (a)'s "reasoning off or minimal", and
  no temperature is sent (the API rejects sampling parameters above effort none); Luna runs at effort none and
  temperature 0. Stateless (store false; `previous_response_id` chaining works but is not used). The capability
  manifest was written before the smoke from ten tiny calls ($0.0016): `results/stage2/sol_capability_manifest.json`
  (tool calling with tool choice auto and a forced named tool, the call round trip, structured output, chaining,
  accepted efforts, rate limits, prices). (c) *Procedure, as A7.12 (b)–(f).* Eligibility smoke
  `configs/dev/sol_eligibility.yaml` (both members LLM-only on development seeds 0–49; A7.12 (b)'s criteria with the
  served-model pin; `scripts/sol_eligibility.py gate`); an ineligible pair means the row is not run (no other closed
  pair is registered). n by A7.12 (c)'s rule: n = 500 if the predicted total (probe, both smokes, calibration, the
  three arms, Jev's calls, × 1.2) fits the envelope, else 300 if it fits, else not run (resource gate). The envelope is
  **$80**, the upper end of A7.12 (i)'s "≈ $70–80", fixed here before any smoke token is read. Prices: the frozen Sol
  price and Luna's verified price, plus the cache-write surcharge the pricing page lists (re-read 2026-10-05: cache
  writes $2.50 / $0.125 per M, i.e. input × 1.25 on `cache_write_tokens`); the surcharge is not part of the frozen
  price, so it enters the gate and the spend, and the analysis reports it as a cost sensitivity beside the frozen-price
  cost. The gate record `results/stage2/sol_cost_gate.json` is written before the first calibration episode.
  Calibration: `configs/experiments/jevgym_shadow_cal_sol.yaml` (`jev_shadow_cal_sol`, seeds 5150–5299, which A7.12 (d)
  reserves for "any later backend calibration"), fitted with the registered command; derived profile
  `configs/harness/jevagent_best_sol.yaml` by `scripts/render_profile_x.py --s2 gpt-6.1-sol --fit-arm
  jev_shadow_cal_sol` (A7.12 (e)). Arms (`configs/experiments/stage2_sol.yaml`): `llm_only_sol`, `jevagent_best_sol`,
  `llm_only_solsmall` on the first n tasks of the frozen Stage-2 list (seeds 300000 to 300000 + n − 1, the tasks S2-1X
  used), Jev via OpenRouter as System-1, three passes, the task-major scheduler (seed 732), launched only after
  `scripts/stage2_spec.py --gate stage2_sol` passes. (d) *Operational.* Workers 4 for every Sol run (X: 8; the account
  allows 500k tokens / min on Sol and 200k on Luna); a new System-1 cache file `cache/s1_stage2_sol.sqlite` shared by
  the Sol calibration and arms, as X's worktree cache was (replays charged at their original cost and latency, A7.16
  (b)); the operator's hard cap of $85 for the whole row, probe, smoke and calibration included, is enforced from the
  ledgers (`scripts/sol_eligibility.py watch`); a forced stop yields an incomplete row, labelled as such with its
  achieved n. (e) *Reading.* Descriptive only: no test, outside every Holm table (A7.12 (i)).
  `scripts/stage2_sol.py` reports the quantities of the X rows (success, Δ vs `llm_only_sol` with the paired task
  bootstrap, cost per episode and savings, coverage, trigger activity, latency) under `stage2_sol/...`. It changes no
  test, arm, seed or rule of Stage 1 or Stage 2.
- 2026-10-05, recorded about 07:15 UTC, before any E1 episode (validation included). **Exploratory row E1: live τ²
  telecom backend substitution with the frozen JevAgentRL-4B** (`docs/REVIEW_ROUND12.md` §3 #3, P1-3). Outside both
  registered stages; descriptive, labelled "exploratory (E1)". (a) *Purpose.* JevAgentRL's live effectiveness outside
  its training generator is unmeasured (A7.3 (b): "JevAgentRL does not run on L3"); E1 measures it once, on fresh τ²
  telecom tasks. (b) *Arms.* `llm_only` (`configs/tau2/llm_only.yaml`) and `jevagent_best_rl`: A7.1 (i)'s τ² rendering
  of the selected profile's Flash section with its `backends.jevagent_rl` thresholds (A7.1 (f): tool_select cascade
  τ 0.5, tool_params cascade τ 0.99999988; the reflection trigger is off on τ²; nothing refit on L3 data), written by
  `scripts/explore_telecom_rl.py render` to `configs/tau2/explore/jevagent_best_rl.yaml`. System-1 is the frozen
  checkpoint of `results/gpu_round5/FREEZE_RECORD.json` behind `training/serve_jev.py` on the Stage-2 GPU host (served
  name `jevagentrl-4b`, the record's temperature file, files sha256-checked against `results/stage2/rl_eval_plan.json`
  before any request), with a new System-1 cache `cache/s1_explore_telecom_rl.sqlite`. Otherwise A7.15 (e)'s settings:
  agent `gemini/gemini-3.5-flash` and user simulator `gemini/gemini-3.5-flash-lite` (temperature 0, minimal thinking),
  `--task-split-name full`, seed 300 for both arms (user simulator paired per task), concurrency 8, A6.22 (e) retries
  for infrastructure errors only. τ² runs on the GPU host (tau2-bench at the pinned 5bfa7e3 in its own venv; driver
  `scripts/aws/explore_gpu.sh`), the two arms back to back. (c) *Tasks.* n = 150 never-run telecom ids. The pool is the
  committed inventory's `full` pool; excluded is every id that appears in any `*.json` / `*.jsonl` under `runs/` (592
  ids: exactly the S2-L3a sample and the 32 validation tasks). The remaining 1,693 ids are ordered by
  sha256("jev-explore-E1-telecom-rl-2026-10-05" + "\n" + id), and the first 150 are taken
  (`configs/tau2/explore/telecom_rl_task_ids.json`, id-list sha256
  `ab411862eefd5f7ef3b584796851a0963719af848cdf1b0c802644e652d65b50`; 136 mms, 13 mobile-data, 1 service issue). One
  attempt per task: A7.15 (e) replaced the three-trial majority by single-attempt success for telecom, and three trials
  (≈ 900 simulations at S2-L3a's ≈ $0.165–0.17 each, ≈ $150) would exceed the cap. (d) *Day-1 go/no-go* on the first 5
  of the 32 reserved validation tasks (A7.15 (b); never evaluated), RL arm only (`scripts/explore_telecom_rl.py
  gonogo`). Go requires all of: no infrastructure error; System-1 answers on tool_select whose options are the telecom
  tools + `respond_to_user`, with confidences in [0, 1] and distributions summing to 1; tool_params questions answered or
  deferred without a malformed call; the ledger charging System-1 calls under `jevagentrl-4b` at its exact-0 price row
  (no unpriced model) and the LLM at the frozen prices; no crash and no loop (every conversation ended by the user
  simulator or the agent, never more than 3 identical consecutive tool calls); projected spend within the cap. On
  no-go, E1 is not run, the reason is recorded, and the paper keeps "live effectiveness outside the generator is
  untested". No success number of a validation task is read. (e) *Cost cap: $75 API* (agent ledger + user simulator,
  validation included; JevAgentRL's System-1 is GPU-only). The 150 tasks run in two chunks of 75, both arms in each.
  Before the second chunk the host checks that spend so far + the first chunk's cost ≤ $75, else it stops (an
  incomplete row, labelled with its n). Expected ≈ $52. (f) *Host.* No g6e.2xlarge, g6e.4xlarge or g6e.8xlarge
  capacity at 06:58 UTC, so the host runs as g6e.16xlarge (one L40S 48 GB, $7.57719 / h as priced in
  `results/stage2/v2_eval_plan.json`) and is resized back to g6e.2xlarge afterwards. (g) *Analysis*
  (`scripts/explore_telecom_rl.py analyze` → `results/explore/telecom_rl/summary.{json,md}`, keys
  `explore/telecom_rl/...` in `paper/generated/explore.tex`). Paired complete case over the 150 ids. Reported: success
  per arm and Δ with the paired task bootstrap (B = 10,000, seed 0) and the Newcombe interval; agent cost per task and
  savings with a bootstrap interval, where the RL arm's System-1 is charged by S2-4's provisioned-tariff convention
  (A7.13 (iii): host $ / h × the RL arm's own serving windows / tasks, measured on this workload), with busy-throughput,
  Jev-API-price, zero and g6e.2xlarge-reference sensitivity rows; the user simulator's cost reported apart; tool and
  argument coverage; LLM calls avoided per task; model-call latency and wall clock (descriptive; they depend on the
  serving host and queueing). No NI or other test; outside every Holm table. E1 revises no registered result: S2-L3a,
  S2-4 and every JevAgentRL sentence stand as registered, and E1 is reported as one exploratory external row whatever
  its outcome.
- 2026-10-05, recorded about 07:15 UTC, before any E2 episode. **Exploratory row E2: the StartLux-Decision size ladder,
  live** (`docs/REVIEW_ROUND12.md` P2-3). Outside both registered stages; descriptive, labelled "exploratory (E2)".
  (a) *Purpose.* A7.18 (e) 2 ran the ladder sizes (0.8B, 2B, 9B) on the common-state corpus C only, and only the 4B and
  the 27B (Q8_0) have live and static rows (A7.18 (e) 1, 3). E2 adds the ladder's live and static rows, so that model
  size can be shown against static accuracy, common-state accuracy and live success. The dense sizes are 0.8B, 2B, 4B,
  9B and 27B (A7.18 (b)), so these three are all that remain; the 35B-A3B mixture of experts is not part of the ladder
  and is not run. (b) *Live arms.* `sl0.8b_ts_replace_f`, `sl2b_ts_replace_f` and `sl9b_ts_replace_f`
  (`configs/experiments/explore_startlux_live.yaml`), each exactly `sl4b_ts_replace_f` of
  `configs/experiments/stage2_startlux_states.yaml`: tool_select replace, Flash, the same suite block, the first 200
  fresh Stage-2 tasks (seeds 300000–300199, `JEV_EVAL_RUNS=1`), workers 4, three passes. The seeds are those of
  A7.18's live rows; no Stage-2 test reads these arms, and their trajectories never enter C. (c) *Static rows (GPU only,
  no API cost).* `configs/decision/explore_bfcl_when2call_startlux_ladder.yaml`: A7.18 (e) 1's E1 config (C8
  population, `batch_by_state`) with the three ladder backends, run while each size is served. (d) *Serving.* The
  vendor's server, one size on the GPU at a time (`scripts/aws/serve_startlux.sh`), package commit a9101e9, at the
  weights revision of the ladder's replay manifest (`results/stage2/gpu/manifest_replay_<size>.json`, `PIN_MANIFEST`).
  If a size serves another revision, that size is not run. New System-1 caches (`cache/s1_explore_startlux_live.sqlite`,
  `cache/s1_explore_startlux_ladder_e1.sqlite`); exact-0 price rows are added for the three sizes in
  `configs/prices/models.yaml` (API cost only, as for the other StartLux rows). Host as E1 (f). (e) *Caps.* API $15
  (expected ≈ $7.5 at the 4B row's ≈ $0.012 per episode); before each size the host requires spend so far + $3.50 ≤
  $15, else it stops. GPU ≤ 12 h (expected ≈ 2 h). (f) *Analysis* (`scripts/explore_startlux_live.py` →
  `results/explore/startlux_live/summary.{json,md}`, keys `explore/startlux_live/...`): per size, live success,
  own-state accuracy, steps and API cost per episode, beside the existing 4B, 27B, Jev and Intern-Decision live rows on
  the same 200 tasks (paired Newcombe intervals vs Jev); static E1 accuracy (C8 population) and common-state accuracy
  (C, A7.18 (e) 2 as reported); Kendall's τ-b of size against each measure, descriptive. No test; outside every Holm
  table; it revises no registered result.
- 2026-10-05, recorded about 08:00 UTC, after the E1 go/no-go round 1 and before any E1 main episode. **E1 day-1
  go/no-go: round 1 no-go (an adapter defect), the fix, and round 2 on validation tasks 6–10.** (a) *What failed.*
  Round 1 (the RL arm on the first 5 reserved validation tasks, 07:17–07:18 UTC) failed one criterion of the E1 entry
  (d), "no crash and no loop": 3 of the 5 conversations were ended by τ²'s `too_many_errors` (10 tool errors), with 37
  agent tool errors, 36 of them executed calls to tools that do not exist (`get_line`, `get_bills`, …). Every other
  criterion held: 5 of 5 records, no infrastructure error; 70 tool_select decisions, all with a System-1 answer over the
  14 telecom tools + `respond_to_user`, none malformed; 10 tool_params decisions answered (all deferred); 80 System-1
  calls charged under `jevagentrl-4b` at its exact-0 row, no unpriced model; projected spend $41.02 ≤ $75. The first
  `gonogo.json` (about 07:45 UTC) showed 0 of 5 records for three checks: a key mismatch in the script (τ² records
  carry the trial suffix `#0`), fixed in `scripts/explore_common.py` (`load_latest`); round 1 was re-decided with the
  fix (`results/explore/telecom_rl/gonogo_round1.{json,md}`). (b) *Cause* (operational fields only: decisions, tool
  names, termination reasons, costs, models). JevAgentRL-4B answers 61 of the 70 tool_select decisions with `respond_to_user`
  at confidence 0.51–0.9997, above the cascade's τ = 0.5. The adapter (`_respond` in
  `src/jevharness/bench/suites/tau2.py`) then asks Flash for the message under `tool_choice="none"`; Gemini 3.5 Flash
  often answers that with a function call and no text (documented in the adapter), and without the tool declarations
  the call names an invented tool; the registered fallback executes that call as the LLM's override. In S2-L3a's
  `jevagent_best` arm (τ = 1.0) the same path ran 23 times in 560 simulations, 19 of them an undeclared tool, against
  743 accepted `respond_to_user` answered in text; S2-L3a is not re-run and stands as registered. (c) *Fix.* A new adapter setting
  `respond_fallback`: `execute` (the default: the registered behaviour, unchanged for every registered config) or
  `defer` (when the respond call returns a tool call and no text, the LLM is asked again with its tools and the
  default tool choice, exactly the cascade's fall-through; logged as a deferral with `s2_value` = its action; the extra
  LLM call is charged to the arm). E1's rendering (`scripts/explore_telecom_rl.py render` →
  `configs/tau2/explore/jevagent_best_rl.yaml`) sets `respond_fallback: defer`; arms, thresholds, tasks, settings and
  the $75 cap are unchanged (round 1's spend counts against it). Test:
  `tests/test_tau2_adapter.py::test_respond_fallback_defer_reasks_the_llm_with_its_tools`. The host runs the fixed
  adapter from the session's overlay (`scripts/aws/explore_gpu.sh`, stage `e1_val2`). (d) *Round 2.* No validation
  success number was read in round 1 or in this diagnosis. A fresh go/no-go runs the RL arm on the next 5 reserved
  validation tasks (6–10 of the 32; `validation_gonogo_2` of `configs/tau2/explore/telecom_rl_task_ids.json`, whose
  `task_ids` and their sha256 are unchanged) under the same criteria of the E1 entry (d) (`scripts/explore_telecom_rl.py
  gonogo --round 2`). On go, E1 runs as registered; on no-go, E1 is not run, the reason is recorded, and the paper keeps
  "live effectiveness outside the generator is untested". The validation tasks stay outside every evaluation.
- 2026-10-05, recorded about 08:00 UTC, before any E1 main episode. **E1 go/no-go round 2: no-go; E1 is not run.**
  Round 2 (validation tasks 6–10, the fixed adapter, 07:52–07:54 UTC) met every criterion of the E1 entry (d) except
  "no loop": every conversation was ended by the user simulator, with no tool error and no call to an undeclared tool;
  118 tool_select and 49 tool_params decisions, all with a System-1 answer, none malformed; 167 System-1 calls under
  `jevagentrl-4b` at its exact-0 row; projected spend $64.27 ≤ $75. On one of the 5 tasks the agent made 23 identical
  `get_data_usage` calls (22 of them back to back). System-1 chose that call above τ at 10 of those turns although
  its state shows the preceding identical calls and their results, and Flash, on deferral, continued the pattern; this
  is the frozen checkpoint's behaviour on τ² telecom, not an adapter defect (S2-L3a's two arms had 0 of 1,120
  simulations with more than 3 identical consecutive calls). No third round is run. By the E1 entry (d) E1 is not run;
  the paper keeps "live effectiveness outside the generator is untested", and may report, labelled as a go/no-go
  outcome on 10 validation tasks and not as an effectiveness measurement, that the frozen checkpoint failed the
  registered day-1 check on τ² telecom. E1's API spend: $1.79 (both validation rounds; agent ledger + user
  simulator). Records: `results/explore/telecom_rl/gonogo_round{1,2}.{json,md}` (`gonogo.{json,md}` = round 2) and
  `results/explore/gpu/`. E2 is unaffected and runs under its own entry.
- 2026-10-05, recorded about 11:30 UTC (presentation only; every Stage-2 result was already known). **A7.16 (e)
  sentences without their intervals.** By the author's decision, the abstract's and introduction's finding sentences
  give the registered label, the point estimate and the cost difference, but not the 95 % interval that the A7.16 (e)
  templates place after them, and the main-text scorecard (paper Table 1) lists point estimates and outcomes only. Every
  interval stays in the paper's appendix table of all registered tests, in `results/primaries/primaries.md` and in
  `results/stage2/stage2.md`. No estimate, interval, label, test or decision changes.
- 2026-10-05, recorded about 17:50 UTC, after the Stage-2 latency diagnosis and while the offline replay benchmark of
  development episodes runs (no LLM call), before any episode on the seeds below. **harness_final: the final
  JevAgentHarness, live (exploratory, descriptive; no test).** (a) *What.* S2-4's profile (`jevagent_best_rl`) was
  cheaper but slower per episode than LLM-only (10.2 s vs 6.8 s, `results/stage2/economics.md`). The diagnosis
  (`scripts/harness_final.py diagnose` → `results/harness_final/latency_diagnosis.md`) places the time in the System-1
  path: 24.6 one-question requests per episode on one GPU server shared by four episodes (busy 97 %, two thirds of the
  System-1 latency queueing; a closed-loop model R = N·S − Z reproduces the logged System-1 time per step), the
  reflection question asked 3,250 times and never fired. The final harness is a post-hoc configuration choice plus
  serving engineering meant to leave every decision unchanged, chosen after Stage 2 and before any of its evaluation
  episodes: the profile `configs/harness/jevagent_final.yaml` (status `development`) = the selected profile's tool +
  argument cascade with JevAgentRL-4B at its frozen thresholds (`backends.jevagent_rl` of
  `configs/harness/jevagent_best.yaml`, unchanged), reflection trigger off, the argument questions of one call in one
  System-1 request (`tool_params.extra.batch_s1`); served by `serve_jev --fast` (fused multi-question pass,
  length-grouped suffix batches, LM head at the read-out positions, prompts tokenized outside the engine lock); no
  System-1 decision cache. Equivalence: offline tests (`tests/test_harness_final.py`) and the replay benchmark
  (`results/harness_final/replay_bench.md`: every replayed answer vs the logged one and across server versions).
  (b) *Arms.* `llm_only` and `harness_final` (`configs/experiments/harness_final_live.yaml`). (c) *Data.* n = 200, the
  spare fresh Stage-2 seeds 300500–300699 (A7.2 (a): reserved, never run, never trained on), the frozen suite settings.
  (d) *System-2.* Gemini 3.5 Flash exactly as in Stage 2 (direct Gemini API, temperature 0, minimal reasoning). A
  5-episode development smoke (seeds 900–904, `configs/dev/harness_final_route_smoke.yaml`,
  `results/harness_final/route_smoke.json`) showed the OpenRouter route with the same successes and call counts but
  about half the reported input tokens, so it is not identical in cost accounting and is not used. (e) *Host and
  execution.* The Stage-2 GPU host <instance-id>, now a g5.4xlarge (one NVIDIA A10G 24 GB; no g6e or g6 capacity
  in us-east-1d on 2026-10-05, the author approved any available type): its latencies are A10G figures, while S2-4's
  JevAgentRL rows ran on an L40S. The frozen checkpoint is sha256-checked against `results/stage2/rl_eval_plan.json`
  before serving (`scripts/aws/harness_final_gpu.sh`). One invocation, both arms task-major interleaved (arm order seeded
  per task, seed 732), workers = 8 (Stage 2's LLM-only concurrency; about four harness_final episodes share the GPU, S2-4's
  load); three passes, passes 2–3 re-running infrastructure errors only; while it runs only operational fields are read.
  The serving setup -- one fast server, or two fast replicas on the one GPU with every request of an episode routed to
  the same replica (`JevBackend(replicas=...)`) -- is the one with the lower mean System-1 time per step at c = 4 on the
  development replay, provided neither changes a decision against the logged answers; it is written into the live config
  and recorded in `results/harness_final/live_choice.json` before the first live episode. A 5-episode development smoke of
  both arms (seeds 900–904) checks errors only. (f) *Tariff.* As S2-4's (A7.13 (iii)): the harness_final configuration on
  S2-4's tariff tasks (development seeds 600–699) at concurrency 4 inside a recorded serving window, q = $1.624 / h
  (g5.4xlarge on demand, us-east-1, AWS price list 2026-10-05) × window hours / completed episodes
  (`results/harness_final/tariff.json`); sensitivities q = 0 and S2-4's q. (g) *Analysis* (`scripts/harness_final.py
  analyze` → `results/harness_final/summary.{json,md}`, keys `\res{hf/...}` in `paper/generated/harness_final.tex`): on
  the tasks whose latest record is complete in both arms, success per arm and the paired difference (point estimate in
  the text, the 95 % paired task-bootstrap interval in the appendix only), cost per episode (harness_final API cost + q)
  and the relative saving, wall-clock and model-call time per episode (means, medians, paired differences; B = 10,000,
  seed 0), and the final harness's decomposition (System-1 compute vs waiting for the engine, prefix-cache hit share,
  tool and argument coverage). Descriptive; no test; outside every Holm table; it revises no registered result. If the
  final harness is not faster per episode, the paper says so with the diagnosis; nothing is tuned or re-run on these
  seeds after any result is read. (h) *Caps.* API $30 (expected about $11: 200 × $0.041 + 200 × $0.006, the tariff
  window and the smokes); GPU within the box's 10-hour guard, the box stopped afterwards.
- 2026-10-05, recorded about 19:20 UTC, while the registered harness_final live row runs (no harness_final or
  dedicated-row result read; only the offline replay benchmark of development episodes and the live row's smoke,
  operational fields only) and before any episode of the row below. **harness_final_dedicated: the final harness with a
  dedicated GPU (exploratory provisioning row, descriptive; no test).** (a) *Why.* The development replay
  (`results/harness_final/replay_bench.md`, A10G) shows the fast System-1 path at 455 ms per step with one episode on the
  GPU and 1,623 ms with four (the server saturates; `latency_diagnosis.md`, closed-loop model), so the per-episode
  wall-clock of the final harness depends on how many episodes share one GPU. How many is a provisioning choice traded
  against the GPU tariff; this row measures the other end of it. (b) *Arms and data.* As the harness_final entry above
  (profile `configs/harness/jevagent_final.yaml`, `serve_jev --fast`, one server, the frozen JevAgentRL-4B; Gemini 3.5
  Flash on the direct API as in Stage 2; n = 200 on the same spare seeds 300500–300699; the frozen suite), in a separate
  run `runs/harness_final_dedicated` (`configs/experiments/harness_final_dedicated.yaml`) with its own same-session
  `llm_only` arm on the same seeds. (c) *Execution.* ONE worker, so at most one harness episode is ever in flight on the
  GPU and both arms run under the same worker setting; task-major interleaved (arm order seeded per task, seed 732); three
  passes, passes 2–3 re-running infrastructure errors only; same host (g5.4xlarge, one A10G 24 GB); operational fields
  only while it runs. (d) *Tariff.* Its own window: the harness_final configuration on S2-4's tariff tasks (development
  seeds 600–699) with one worker (`configs/dev/harness_final_tariff_dedicated.yaml`), q = $1.624 / h × window hours /
  completed episodes (`results/harness_final/tariff_dedicated.json`); the GPU cost per episode is higher than in the
  shared row and is reported as such. (e) *Analysis.* Exactly the harness_final analysis (`scripts/harness_final.py
  analyze --name summary_dedicated` → `results/harness_final/summary_dedicated.{json,md}`, keys `\res{hf_ded/...}`);
  both rows side by side in `results/harness_final/summary_rows.md`. The registered harness_final row stays the primary
  harness_final result; this row is reported beside it as a provisioning sensitivity, never instead of it. Running
  `llm_only` again on the same seeds yields a second LLM-only measurement of the same tasks; each row is compared only
  with its own same-session reference. Nothing is tuned or re-run on these seeds after any result is read. (f) *Caps.*
  API: the harness_final total including this row stays within $30 (estimate before launch: about $10 for this row,
  200 × $0.041 + 200 × $0.0045 + the tariff window, on top of about $10 for the shared row and $0.29 of smokes); GPU
  about 1.5 h more within the box's guard; the box stopped afterwards.
- 2026-10-05, recorded about 20:45 UTC (presentation only; every Stage-2 result was already known). **A7.16 (e)
  sentences in plain words.** By the author's request (2026-10-05, "easier to understand, no obscure words"), the
  abstract's and introduction's finding sentences paraphrase the A7.16 (e) templates in plain words with the same
  outcome, point estimate and cost difference (for example, "non-inferior to LLM-only within 3 points of success" is
  written as "no worse than the LLM alone by more than 3 points of task success", and "the registered joint
  cost-and-success claim was not established" as "our pre-registered claim (cheaper and no worse) did not hold"); none
  is strengthened, and each keeps its scope (S2-3 and S2-4 inside the training generator). The template sentences
  themselves stay in the paper's Results section (§7) and appendix. No estimate, interval, label, test or decision
  changes.
- 2026-10-05, recorded about 21:23 UTC, before any ablation training step and before any ablation checkpoint answers
  a question. **Exploratory row E3: an SFT training ablation of the decision model** (`results/training/ablation_plan.md`,
  sections 1-10 and the addendum of section 11). Outside both registered stages; descriptive, labelled "exploratory
  (E3)"; it revises no registered result, freezes no checkpoint, serves no ablation checkpoint in any other experiment
  and changes nothing in JevAgentRL (A7.10) or JevAgentRL-v2 (A7.19). (a) *Purpose.* The freeze records show that SFT on
  data v1 carries JevAgentRL's development accuracy, but data v1 differs from data v0 in states, size, task setting and
  labels at once. E3 varies one factor at a time with everything else at the SFT v1 recipe: data size (nested task
  subsets of 10, 25, 50 and 100 %), label source (gold one-hot, Jev alone, the v1 mixture; at 25 and 100 %) and state
  distribution (rows without an earlier off-plan step vs a size-matched random subset; inside D25's rows and on all
  rows), with two seed replicates (D10, D25). (b) *Data and seeds.* Every training row is a row of
  `data/train_v1/train.jsonl.gz` (sha256 `9c0450c2...`), selected and relabelled by `scripts/train/ablation_subsets.py`
  (seed 0); no episode is run and no JevGym seed is drawn, so every training seed stays a data-v1 seed (>= 10000,
  outside 100000-100499, 200000-200499, 300000-300999 and 5000-5999; `training/seed_guard.TRAIN_FORBIDDEN`, re-checked by
  the builder). The Jev-only targets were rebuilt from the cached Jev answers of the data-v1 build with no key in the
  environment (15,694 rows, 5,175 cached calls, 0 fresh calls, $0). The hash of every arm file and of every arm's
  training file (rows + When2Call replay) is fixed in `results/training/ablation_inputs.json` and checked on the GPU host;
  D100's training file is byte-identical to SFT v1's. (c) *Training.* The SFT v1 recipe unchanged (round-5 manifest §3
  stage 1: Qwen3-4B at snapshot `1cfa9a72...`, LoRA r 32 / alpha 64 on all linear layers, soft CE + 0.5 Brier, option
  permutation, LR 1e-4 cosine, warm-up 3 %, 2 epochs, 64 rows per optimizer step, 3,072-token micro-batches, bf16, gate
  set `data/train_v0/dev.jsonl`), replay factor 0.45 x rows / 15,694, trainer seed 0 (the replicates 1). Only on a CUDA
  out-of-memory error, and in this fixed order: activation checkpointing (same maths), then checkpointing with 2,048,
  1,536 or 1,024-token micro-batches and the same 64 rows per step; LR, epochs, effective batch, loss, LoRA rank and data
  never change, and each arm records the setting it used. (d) *Evaluation, GPU only, no API.* `eval_decisions` on v1
  dev, dev_recovery and dev_knockout (training-side development data); temperatures fitted on v1 dev
  (`training/calibrate.py --min-bucket 200`); the common-state corpus C (`results/stage2/s23/`, built from Stage-2 seeds
  300000-300199 and already analysed) replayed through `serve_jev` on the box (`--fast`; the plain Stage-2 server if the
  anchor's C answers fail under `--fast` alone) into a copy of the corpus with per-arm caches. No live episode, no tau2
  simulation, no LLM call. C is used post hoc and only descriptively: no checkpoint, threshold or recipe is selected on
  it. (e) *Host, order and caps.* The Stage-2 GPU host <instance-id> as a g5.8xlarge (one A10G 24 GB, $2.448 / h; at launch there was no capacity for g6e.2xlarge, g6e.xlarge, g6e.4xlarge, g6e.8xlarge, g5.4xlarge or g5.2xlarge), driven by `scripts/aws/train_ablation.sh`;
  no key is loaded on the box and any priced replay record stops the run. The existing SFT v1 checkpoint is first
  re-evaluated through the same pipeline; if it does not reproduce its recorded v1-dev accuracy (0.99706 within 0.001),
  its temperatures (relative 1e-3 on an L40S, 2e-2 on another GPU type) and >= 99 % of its published C answers, E3 stops
  before any training. Then the core block (D25, L_gold_25, L_teach_25, S_canon_25 + S_match_25, D10, D50), then the
  extended block (D25_s1, D10_s1, D100, L_gold_100 + L_teach_100, S_canon + S_match), under a cap of 24 GPU-hours:
  before each arm (each pair as one unit) the used plus projected hours must fit, else that unit is not run and is
  reported as not run. API $0. (f) *Analysis.* `scripts/train/ablation_score.py` ->
  `results/training/ablation/ablation.{json,md}`, keys `recipe_abl/...`: per arm v1-dev accuracy by decision type, after
  an off-plan step and on-plan, raw ECE; C accuracy (all, after an off-plan step) and ECE with the arm's temperatures;
  the contrasts of the plan (size vs D100, vs D50 and vs the anchor; gold / teacher vs mixed at 25 and 100 %; canon vs
  match at 25 and 100 %; seed; retrain vs anchor) with paired task-bootstrap intervals on C (B = 2,000, seed 0) and the
  reading rules of plan §7 and §11.10 (a factor is described as making a difference on C only when its interval
  excludes 0 and its |delta| exceeds the largest seed |delta| at a comparable size; without the seed arms, the text says
  that seed noise was not measured). Development and descriptive numbers only; outside every Holm table.
- 2026-10-05, recorded about 21:35 UTC, before any E4 agent episode (so far only API-level verification calls of a few
  tokens each). **Exploratory row E4: JevAgentHarness on an open-source harness (mini-SWE-agent), per-step System-1 /
  System-2 routing on SWE-bench Verified.** Outside both registered stages; descriptive, labelled "exploratory (E4)".
  (a) *Purpose.* The author's brief of 2026-10-05: the harness should build on a well-known open-source harness and save
  much time and cost without losing success; E4 evaluates that once. In mini-SWE-agent every step is a free-form bash
  command, so the decision model cannot pick the tool; its lever is per-step routing: does this step need slow,
  deliberate reasoning (System 2), or can a fast path take it? (b) *Harness.* mini-SWE-agent 2.4.6 (the version pinned
  in H5), unmodified, through the existing adapter `jevharness.adapters.mini_swe` and H5's routing hook (the `route`
  arm machinery: before each step one Jev question, "does the next step require careful multi-step reasoning,
  planning, debugging, or recovery from an error … rather than a routine action?"; p(hard) ≥ τ → the deep model, else
  the fast model; the two models share one message history; a System-1 error fails open to the deep model). Config:
  mini-SWE-agent's own SWE-bench config (`adapters/mini_swe/swebench_jev.yaml`) with the overlays
  `s1_jev_openrouter.yaml`, `e4_deepseek.yaml` and, for `always_fast`, `e4_fast.yaml`. The only adapter change is
  `OpenRouterTimeoutModel`: mini-SWE-agent's own `OpenRouterModel` with a configurable request timeout (600 s; upstream
  hard-codes 60 s, which a thinking step can exceed) and a 200 response without `choices` raised as a retryable API
  error; payload, `usage.cost` accounting and parsing are upstream's. (c) *Models and the reasoning toggle.* OpenRouter
  only (no direct Gemini call). Deep path (System 2): `deepseek/deepseek-v4-pro` (canonical slug
  `deepseek/deepseek-v4-pro-20260423`) with OpenRouter's `reasoning: {enabled: true}`; fast path:
  `deepseek/deepseek-v4-flash` (`…-20260423`) with `reasoning: {enabled: false}`; both pinned to StreamLake (fp8, the
  S2-1X pin of A7.12) with `allow_fallbacks: false`; provider default temperature (mini-SWE-agent's config sets none).
  Verified 2026-10-05, 21:12–21:20 UTC, with tiny calls from the project key: without a `reasoning` field both models
  think by default (85 and 39 reasoning tokens on one probe); `enabled: false` gives 0 reasoning tokens and
  `enabled: true` thinks; tool calls and text output work with both; a shared history Pro (thinking) → Flash (no
  thinking) → Pro runs through mini-SWE-agent's model class; on a 9k-token prompt the deep path took 7.7–12.8 s per
  call (173–484 reasoning tokens) and the fast path 5.0–8.4 s. Endpoint prices at the pin (OpenRouter endpoint list,
  read 21:12 UTC): Pro $0.2088 / $0.4176 per M input / output tokens, cache read $0.0174 (as in A7.12 (a)); Flash
  $0.042 / $0.084, cache read $0.0084 (StreamLake's Flash price has risen since A7.12 (a)'s $0.028 / $0.056; the
  `models.yaml` row documents S2-1X's run date and is not edited). E4's agent cost is OpenRouter's `usage.cost` per
  call (mini-SWE-agent's OpenRouter class raises on a call without a cost). Decision model: Jev via OpenRouter
  (`{"type": "jev", "provider": "openrouter"}`, `typesafe/jev-1.13`), every call in the Ledger, a new System-1 cache per
  arm and phase; no gold label is sent (the routing hook has none). (d) *Arms.* `always_deep` (every step on the deep
  path; the baseline), `jevagent` (Jev routes each step: deep if p(hard) ≥ τ or no answer, fast otherwise),
  `always_fast` (every step on the fast path; the lower bound, what deliberation buys). Same templates, caps and pin;
  the arms differ only in which model answers a step. (e) *Instances.* SWE-bench Verified (test) only. Pool: the 500
  instances minus H5's 100 evaluation instances (`results/hard/swe/instance_ids.txt`) and the H5 smoke pair
  (`django__django-16502`, `scikit-learn__scikit-learn-13496`), which is E4's smoke (development only, never reported).
  The remaining 398 are ordered by sha256("jev-explore-E4-swe-2026-10-05" + newline + id); dev = the first 45
  (`configs/explore/e4_swe/dev_ids.txt`, list sha256 `c5d0a510cab2b4094c5ea71deed85e84adafdd7682e4c0c21fbe3c077b742d24`),
  eval = the next 100 (`eval_ids.txt`, sha256 `8d67497edcc71e78e2c0d49496f0bb9333dc87959addf7a3cf20eb440b89a1ee`);
  `selection.json` records the rule and the pool hash. Committed before any agent episode. (f) *Threshold rule (dev
  only).* Dev arms: `always_deep`, `always_fast` and `jevagent` at τ ∈ {0.5, 0.6, 0.7} (0.5 is H5's registered route τ;
  on H5's Gemini trajectories Jev's p(hard) lay mostly between 0.5 and 0.75, which gives about 16 %, 34 % and 72 % of
  steps on the fast path at these τ). The fast-share of an arm is the pooled share of its agent LLM calls answered by
  the fast path. Rule, applied mechanically by `scripts/explore_e4_swe.py threshold`: (1) if `always_fast`'s dev
  resolved count is itself within 2 of `always_deep`'s, dev cannot separate the settings and the eval arm uses the
  registered route τ = 0.5; (2) otherwise, among the candidates whose dev resolved count is ≥ `always_deep`'s − 2, the
  one with the largest dev fast-share (ties to 3 decimals → the lower, more conservative τ); (3) if none qualifies,
  τ = 0.5. A second entry, appended before any eval episode, records the τ chosen and the eval list hash; the driver
  refuses the eval phase without the committed `threshold.json` whose eval hash matches. (g) *Caps and execution.*
  k = 1; 250 steps (H5) and $0.42 per instance (H5's $3 scaled by the deep path's input price over H5's Gemini 3.5
  Flash input price, 0.2088 / 1.50 × $3), the same cap for every arm; 60 s per command. Box: H5's SWE box
  `<instance-id>` (m7i.8xlarge), driver `scripts/aws/e4_swe.sh` (unattended, status and outputs synced to S3,
  power-off and idle-stop guards). The arms run interleaved on the same box in one shared task-major pool: instance by
  instance, the arms of an instance in a seeded order (seed 732), so each instance's runs start together (dev: 25
  workers, about 5 per arm; eval: 24, about 8 per arm). An instance whose record carries an infrastructure error (an
  exception escaped the agent) is re-run once and recorded; nothing else is re-run, and nothing is tuned on eval
  instances. Local swebench 4.1.0 evaluation. API cap $150 in total (OpenRouter DeepSeek + Jev; the key's usage was
  $33.36 at 21:10 UTC, and the driver's watchdog stops the runner at $183.36 of key usage, which counts any other spend
  on the shared key); a smoke (the smoke pair × 3 arms) projects the cost before the dev runs, and E4 stops if a
  projection exceeds the cap. (h) *Readout* (`scripts/explore_e4_swe.py summarize` →
  `results/e4_swe/{summary.md, summary.json, per_instance.csv, NOTES.md}`, keys `e4/...` in
  `paper/generated/e4_swe.tex`): per arm resolved / 100, Δ vs `always_deep` with wins / losses, mean $ per instance
  (DeepSeek `usage.cost` + Jev) and the saving in %, median and mean agent-run time, the paired median time difference
  on instances both arms resolve, and the fast-share; p values are descriptive. Descriptive, outside every Holm table,
  revises no registered result; reported whatever the outcome.
- 2026-10-05, recorded about 21:40 UTC (presentation only; every Stage-2 result was already known). **Addendum to the
  20:45 entry (A7.16 (e) sentences in plain words): the final wording.** After two rounds of external reading, the
  plain paraphrases keep the registered term "non-inferior" and gloss it once as the test it is ("a test rules out
  losing over 3 points of (task) success", as in the paper's statistics paragraph), because "no worse than ... by more
  than 3 points" made readers ask why a −2.2-point estimate failed. Final forms: S2-1, "the pre-registered claim
  (cheaper and non-inferior) did not hold: the estimated success difference was −2.2 points, at 40.7 % lower cost";
  S2-4, "[with it, the agent / its profile] was non-inferior to [the LLM alone / LLM-only] and 81.1 % cheaper per
  episode", on fresh tasks from its training generator; S2-3, "on the same recorded states, [it / JevAgentRL-4B] was
  more accurate than an off-the-shelf 4B model and no more than five points less accurate than [the hosted one / Jev]".
  Same outcomes, point estimates, cost differences and scope; no estimate, interval, label, test or decision changes.
- 2026-10-05, recorded about 23:15 UTC, after the E4 dev runs and before any E4 eval episode. **E4: the routing
  threshold fixed on dev, and the eval list.** (a) *Dev* (45 instances, list sha256 `c5d0a510…`, k = 1, all five arms
  complete, 0 infrastructure errors left): resolved `always_deep` 36, `always_fast` 34; `jevagent` at τ 0.5: 38
  (fast-share 21.4 %), τ 0.6: 38 (39.0 %), τ 0.7: 33 (72.8 %, not within 2). (b) *Threshold.* The registered rule of the
  E4 entry (f) gives case (1): `always_fast` (34) is within 2 of `always_deep` (36), so dev cannot separate the settings
  and **the eval arm `jevagent` runs at the registered route τ = 0.5** (`configs/explore/e4_swe/threshold.json`, the dev
  output as written by `scripts/explore_e4_swe.py threshold` at 23:11 UTC). Without case (1), case (2) would have
  chosen τ 0.6; the rule was fixed before the dev runs and is applied as written. (c) *Eval list* unchanged:
  `configs/explore/e4_swe/eval_ids.txt`, 100 instances, sha256
  `8d67497edcc71e78e2c0d49496f0bb9333dc87959addf7a3cf20eb440b89a1ee`; the driver refuses the eval phase unless
  `threshold.json` carries this hash. (d) *Execution notes from dev* (development only; nothing about the arms, caps or
  rules changes). (i) The pinned endpoint sometimes returns a cut-off generation (`finish_reason` "error", a few tokens,
  no cache hit) that OpenRouter does not bill (`usage.cost` 0); mini-SWE-agent's OpenRouter class then fails the whole
  instance in its cost accounting. Six dev instance-arms failed this way in the first 54 of 225 jobs. The adapter's
  `OpenRouterTimeoutModel` now treats an unbilled response as a failed generation and retries it through
  mini-SWE-agent's own retry (commit `11b3fcd`, test in `tests/test_explore_e4_swe.py`); the dev runner was stopped at
  22:03 UTC, the fix deployed, and the run resumed (the in-flight jobs were run again from scratch; the six errored
  instance-arms got their one re-run, all completed). After the fix, 3 such responses were retried and none failed an
  instance. (ii) A background pull of the eval images beside the dev container starts saturated the box's disk and
  delayed every container start by about 10 min (container start is outside the agent-run time); the driver no longer
  pulls in the background. Dev wall-clock numbers are therefore not comparable across arms and are not used.
- 2026-10-05, recorded about 23:55 UTC, before any E4b episode, and before any E4 eval result was read (while E4's eval
  runs, only its operational fields were read: jobs done, infrastructure errors, retries). **Exploratory row E4b: the same
  routing with one model on both paths (Gemini 3.8 Flash, thinking high vs low).** Outside both registered stages;
  descriptive, labelled "exploratory (E4b)"; approved by the author on 2026-10-05 with a $400 cap on the direct Gemini
  API. (a) *Purpose.* E4 (above) can hardly save: DeepSeek V4 is very cheap, and its deep and fast paths are two models,
  so every switch breaks the prompt cache. E4b tests the case built for delegation to pay: an expensive, slow deep step,
  and the same model on both paths, so the cache survives a switch. (b) *Harness, decision model, arms.* As E4 (b), (d):
  mini-SWE-agent 2.4.6, the adapter's routing hook, Jev via OpenRouter asking the same routing question per step;
  `always_deep`, `jevagent`, `always_fast` (`scripts/explore_e4_swe.py --exp e4b`, overlays
  `adapters/mini_swe/e4b_gemini.yaml`, `e4b_fast.yaml`, `s1_jev_openrouter.yaml`). (c) *Agent LLM.* `gemini/gemini-3.8-flash`
  through the direct Gemini API (litellm; keys from SSM as in H5). Deep path: `thinkingConfig.thinkingLevel: high`; fast
  path: the same model with `thinkingLevel: low`. The model rejects `minimal` (HTTP 400, "Thinking level MINIMAL is not
  supported for this model"), so `low` is the closest fast setting. Verified 23:46 UTC with one tiny call per level:
  high → 1,011 reasoning tokens, low → 281 on the same prompt, a bash tool call at both. Price, read 23:50 UTC on
  Google's pricing page (paid standard tier): $0.75 / $3.75 per M input / output tokens (output includes thinking),
  context caching $0.075, an introductory price through 2026-12-31 ($1.50 / $7.50 / $0.15 from 2027-01-01); litellm
  1.82.6 prices the model with the same numbers, so mini-SWE-agent's cost accounting and `agent.cost_limit` bind on it;
  `configs/prices/models.yaml` marks the row verified. (d) *Instances (shared with E4).* Smoke: E4's smoke pair. Dev: the
  first 30 of E4's dev list (`configs/explore/e4b_swe/dev_ids.txt`, sha256 `37cf9ace…`), so no new instance is exposed.
  Eval: **E4's 100-instance eval list, unchanged** (sha256 `8d67497edcc71e78e2c0d49496f0bb9333dc87959addf7a3cf20eb440b89a1ee`),
  so the two rows are comparable instance by instance. (e) *Threshold rule (dev, fixed now in code,
  `scripts/explore_e4_swe.py --exp e4b threshold`).* Dev arms `always_deep`, `always_fast`, `jevagent` at τ ∈ {0.5, 0.6,
  0.7}; exactly E4 (f)'s rule: (1) `always_fast` within 2 of `always_deep` → the registered route τ 0.5; (2) else the
  largest dev fast-share among the candidates within 2 of `always_deep` (ties → the lower τ); (3) none → 0.5. The fast
  path is read from the routing hook's role (both paths are one model). The box applies the rule and launches eval with
  no human step between dev and eval (`scripts/aws/e4b_chain.sh`); the τ chosen is recorded in
  `runs/e4b_swe/dev/threshold.json` and in the readout. (f) *Caps and execution.* k = 1; 250 steps and $1.50 per
  instance (H5's $3 scaled by the input-price ratio 0.75 / 1.50, as in E4), the same for every arm; 60 s per command.
  Same box and driver as E4; arms interleaved task-major in one shared pool (dev 25 workers, eval 24), seed 732;
  infrastructure errors re-run once. The smoke may overlap the last minutes of E4's eval (6 extra containers); dev and
  eval start only after E4's eval driver has exited. (g) *Budget ($400 of Gemini API, logged per call).* Smoke first;
  projection = the smoke's mean Gemini $ per instance per arm × the dev + eval jobs × 1.5. If spend + projection exceeds
  $400, E4b stops after the smoke; if the projection exceeds $300, **`always_fast` runs on the first 50 eval
  instances only** (the other arms on all 100). Before eval, the same gate on the dev means. Each phase process carries
  mini-SWE-agent's global cost limit set to what is left of the $400, so the cap binds on the real spend in flight. Jev
  (OpenRouter) is logged apart (cents). (h) *Readout* (`scripts/explore_e4_swe.py --exp e4b summarize` →
  `results/e4b_swe/{summary.md, summary.json, per_instance.csv, NOTES.md}`, keys `e4b/...` in
  `paper/generated/e4b_swe.tex`): as E4 (h). Descriptive, outside every Holm table, revises no registered result;
  reported whatever the outcome, beside E4.
- 2026-10-06, recorded about 00:06 UTC, after the E4b smoke and before any E4b dev or eval episode (only the smoke's
  costs, step counts and exit statuses were read; no smoke instance was evaluated). **E4b: scope amendment for the
  budget.** (a) *Smoke cost.* Five of the six smoke runs had finished: Gemini $ per instance `always_deep` 0.40 and 1.16
  (50 and 101 steps), `always_fast` 0.30 and 0.61, `jevagent` (τ 0.5) 0.67. The Gemini API answered many calls with
  HTTP 503 ("high demand") and some with HTTP 429 (the project's tier-3 limit of 20 M input tokens per minute for
  gemini-3.8-flash); mini-SWE-agent's retry absorbed both, and no smoke run errored. (b) *What the registered rule did.*
  The chain projected dev + eval at $288 (× 1.5 = $432) before applying the registered `always_fast` cut, so it would
  have stopped E4b although the cut was due (projection > $300). The E4b entry (g) reads "if the projection exceeds
  $300, `always_fast` runs on the first 50 eval instances"; the gate must judge the run as it will be run, so the chain
  now applies the cut first and then the $400 gate (`scripts/aws/e4b_chain.sh`). The chain was stopped by hand before
  its gate acted (00:03 UTC); the smoke finished on its own. (c) *Scope.* Even with the cut, the projection at the
  smoke's `jevagent` cost is $398 + $3.5 of smoke, at the $400 line. **The dev candidates are reduced to τ ∈ {0.5, 0.6}**
  (τ 0.7 dropped: the most aggressive candidate, which lost success on E4's DeepSeek dev; τ 0.5 stays as the registered
  fallback), and **`always_fast` runs on the first 50 eval instances** (the registered cut). Projection with both: about
  $245 × 1.5 = $368. The rule of E4b (e), the arms on eval, the eval list, the caps and the $400 cap are unchanged; the
  readout pairs `always_fast` with `always_deep` on its 50 instances. (d) *Execution.* The chain restarts with the smoke
  as run (`SKIP_SMOKE=1`), waits for E4's eval driver to exit, then runs dev, the threshold rule and eval unattended.
- 2026-10-06, recorded about 00:08 UTC, after the E4b smoke and before any E4b eval episode (no E4b dev episode was run;
  only the smoke's costs, step counts and exit statuses were read, no smoke instance evaluated; no E4 eval result read).
  **E4b: dev skipped for cost; eval at the registered route τ.** (a) *Why.* With the amendment above, the chain's
  registered gate projected dev + eval at $303 from the smoke's means, $455 at the registered 1.5× safety factor, over
  the $400 cap, and stopped E4b after the smoke (00:05 UTC, as registered). The smoke's Gemini $ per instance (n = 2 per
  arm): `always_deep` 0.40 and 1.16 (mean 0.78), `always_fast` 0.30 and 0.61 (mean 0.46), `jevagent` (τ 0.5) 0.67 and
  1.40 (mean 1.04). **In the smoke, `jevagent` cost more than `always_deep`**; with n = 2 that is no evidence either way,
  and it is recorded so that it stays visible. (b) *Decision* (the coordinator for the author; the $400 cap is not
  raised): dev is skipped, and **the eval arm `jevagent` runs at the registered route τ = 0.5** (H5's `route` τ, the
  value E4's rule gave after its dev); `configs/explore/e4b_swe/threshold.json` is written on the box with case
  `no_dev`. **`always_fast` stays on the first 50 eval instances** (the amendment above); `always_deep` and `jevagent`
  run on all 100 of E4's eval list. (c) *Unchanged:* arms, models and thinking levels, Jev routing question, caps
  ($1.50, 250 steps, 60 s), the task-major shared pool (24 workers, seed 732), one re-run of infrastructure errors, and
  the $400 watchdog: the chain's gate on the eval projection (smoke means × eval jobs × 1.5, after the cut; about $306
  plus the smoke's spend) and mini-SWE-agent's global cost limit set to what is left of the $400. (d) *Readout.* The two
  smoke instances are excluded from the readout (development only). The E4b row is reported beside E4 with "no dev; τ
  fixed in advance at the registered route value".
- 2026-10-06, recorded about 03:30 UTC (presentation only; every Stage-2 result was already known). **Abstract in the
  form of a findings summary.** By the author's request (2026-10-06: the abstract should summarise and tell the story
  as strong abstracts do, without statistical definitions or test bookkeeping), the abstract states the S2-1 outcome
  as a finding ("two components that each passed screening lost success when composed", with its point estimate where
  it fits), and states S2-4 and S2-3 in plain words without test definitions ("non-inferior" may appear once without
  its gloss; the paper defines it in Section 6), keeping the in-generator scope ("on fresh tasks from its training
  generator"). It replaces, in the abstract only, the plain paraphrases fixed by the 20:45 and 21:40 entries of
  2026-10-05; the introduction may use the same short forms with a status word. The A7.16 (e) template sentences
  remain in Section 7 and the appendix. No outcome, estimate, interval, scope, label, test or decision changes; no
  claim is strengthened (S2-1 stays not established, S2-3 and S2-4 stay held and in-generator, the trigger never
  fired, E4 and the final harness stay exploratory).
- 2026-10-06, recorded about 03:50 UTC (presentation only). **Addendum to the 03:30 entry: test bookkeeping out of
  the main-text prose.** By the author's further request (2026-10-06: "fix all similar unprofessional writing
  throughout the main text"), the main text (abstract, Sections 1-9 and the captions of pp. 1-8) states each
  registered outcome with a short status word ("held" / "did not hold"; "did not hold" means only that the test did
  not reject, as Section 6 says), and test IDs (S2-x, Cx, H1) appear only in Table 1 and the appendix. The A7.16 (e)
  template sentences therefore move from Section 7 to the appendix (paragraph "Claim labels" of the Stage-2
  registration appendix), which quotes them for all four primaries. No outcome, estimate, interval, scope, label,
  test or decision changes.
- 2026-10-06, recorded about 05:40 UTC, before any E5 training step, before any E5 checkpoint answers a question and
  before any episode on seeds 300700–300799 (no E5 model has been downloaded or run; the inputs below are committed
  files, a CPU-only tokenizer and template preflight and public model cards). **Exploratory row E5: small decision
  models** (`results/explore/small_models/{survey,e5_plan}.md`; where this entry and the plan differ, this entry holds).
  Outside both registered stages; descriptive, labelled "exploratory (E5)"; it revises no registered result, freezes no
  checkpoint, serves no E5 checkpoint in any other experiment and changes nothing in JevAgentRL (A7.10) or
  JevAgentRL-v2 (A7.19). (a) *Purpose.* JevAgentRL is Qwen3-4B. E5 asks whether the frozen SFT v1 recipe keeps the 4B's
  in-generator accuracy at 1.7B, about 1B and 0.6B parameters, and what the smaller size buys in System-1 latency and
  GPU memory on the same GPU. (b) *Models* (Hugging Face, ungated, downloaded at the pinned revision into the box's HF
  cache, no token): `Qwen/Qwen3-1.7B` (revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`), `Qwen/Qwen3-0.6B`
  (`c1899de289a04d12100db370d81485cdf75e47ca`) and `openbmb/MiniCPM5-1B` (`87179e5c1f455ef22e6223592d2d61351b525bfc`),
  run in this order. `ibm-granite/granite-4.0-1b` (`6a7381ba1f54d684ff508d991aeb7dc580157103`) replaces MiniCPM5-1B
  only if MiniCPM5-1B fails its smoke (it does not load, an option code is not a single token, every memory setting
  runs out of memory, or the smoke's loss is not finite), and the swap is recorded. Chosen from public model cards and
  the CPU-only preflight (`results/explore/small_models/preflight/`) before any E5 number existed. (c) *Data and
  training.* E3's D100 file (the 15,694 SFT v1 rows, rebuilt on the box by `scripts/train/ablation_subsets.py build
  --arms D100` from the committed data v1 and checked against sha256 `65fa28a1…` in
  `results/training/ablation_inputs.json`) plus the When2Call replay at factor 0.45; each model's training file is
  checked against SFT v1's (`9fe1088d…`; the mix does not depend on the tokenizer). No episode is run and no JevGym seed
  is drawn for training. The SFT v1 recipe unchanged except the base model: LoRA r 32 / alpha 64 on all linear layers,
  soft CE + 0.5 Brier, fresh option permutation per row, LR 1e-4 cosine, warm-up 3 %, 2 epochs, 64 rows per optimizer
  step, 3,072-token micro-batches without activation checkpointing, bf16, `MAX_LEN` 4,096 (tail kept), gate set
  `data/train_v0/dev.jsonl`, trainer seed 0, one seed per model. Memory fallback exactly as E3 (only on a CUDA
  out-of-memory error, in this order: checkpointing with the same 3,072-token budget, then 2,048, 1,536 and 1,024-token
  budgets with the same 64 rows per step); each model's 4-step smoke (the 48 longest D100 prompts plus 48 random rows)
  fixes its starting setting, and each model records the setting it trained with. LR, epochs, effective batch, loss,
  LoRA rank and data never change; no LR sweep, no GRPO. (d) *Evaluation, GPU only, no API.* As E3 (d):
  `eval_decisions` (raw) on v1 dev, dev_recovery and dev_knockout; temperatures fitted on v1 dev
  (`training/calibrate.py --min-bucket 200`); the common-state corpus C (`results/stage2/s23/`) replayed through
  `serve_jev --fast` (E3's anchor serving mode) with the model's temperatures into a copy of the corpus with a per-model
  cache. Read-outs: v1 accuracy by decision type, after an off-plan step, on-plan and knockout, raw ECE; C task-mean
  accuracy over all states, after an off-plan step (386 occurrences, 57 tasks) and on the ask-vs-act stratum (the 582
  occurrences whose oracle answer is `ask_user`, 100 tasks), and C ECE with the model's temperatures. The 4B references
  are committed rows of the same pipeline: SFT v1 (the same recipe at 4B; `results/gpu_round5/eval/eval_sft_v1_v1.json`,
  `results/stage2/s23/answers_sft_v1.jsonl`) and the frozen JevAgentRL-4B (`answers_rl.jsonl`). C is used post hoc and
  only descriptively: no checkpoint, threshold or recipe is selected on it, except the live-row rule in (g).
  (e) *Latency and memory* (one session, one GPU). Each small model with its adapter merged and its temperatures, and
  the frozen JevAgentRL-4B (sha256-checked against `results/stage2/rl_eval_plan.json` before it is served), each served
  alone by `serve_jev --fast`: (i) the final harness's request stream, `scripts/harness_final.py bench` on the committed
  `results/harness_final/replay_workload.jsonl.gz` (new call graph, logged System-2 time emulated), c = 1 on 25 episodes
  and c = 4 on 100 episodes; read-out p50 and p90 ms per request (client side; all requests and by decision point) and
  System-1 ms per step; (ii) `scripts/measure_latency.py` (c = 1 and 4, n = 200, 1,500-token states, 3 questions;
  p50 / p90); (iii) peak GPU memory (`nvidia-smi` sampled during (i), one model on the GPU; and during training) and the
  server's cache facts (croppable, KV bytes per token). Speed-up = the 4B's p50 (p90) over the model's at the same c.
  Latency is never compared across sessions or GPU types. (f) *Host and caps.* One new A10G instance launched from an
  image of the E3 host (its id and type are recorded in the run), driven by `scripts/aws/e5_small.sh`; records go to
  `s3://<bucket>/e5_small/` only. No key is on the box during the offline block and any priced
  record stops the run (API $0). GPU ≤ 14 host hours for (c)–(e): before each model the used plus projected hours
  (with the latency stage reserved) must fit, else that model is not run and is reported as not run. (g) *Live row*
  (run only if at least one small model reaches ≥ 95.0 % C accuracy, task mean over all states). The model: the highest
  C accuracy among those ≥ 95 %; when two are within 1.0 point, the one with the lower c = 1 p50 per request in (e)(i);
  `scripts/e5_analysis.py` applies the rule to the offline records before the first live episode. *Thresholds, offline
  at $0:* in shadow mode System-1 never acts, so the trajectories of a calibration shadow arm do not depend on it. Every
  tool_select and tool_params question of the frozen 4B's own calibration arm (`rl_shadow_cal` of
  `runs/e2a_jevgym_shadow_cal_rl`: Flash, calibration seeds 5000–5149, n = 150; the arm that fitted the frozen
  thresholds) is rebuilt in the Jev wire format (tool descriptions regenerated from the task seed, as
  `scripts/harness_final.py workload`) and answered by the served model on the box (`scripts/e5_cal_replay.py`); a copy
  of the arm with the model's answers in the System-1 fields goes through the registered command
  (`threshold_theory --calibration-arm --task-value 1.0 --lam-per-hour 0 --v-fallback W`, per decision point), and the
  fits are written into `backends.e5_small` of `configs/harness/explore_e5_final.yaml` (a copy of
  `configs/harness/jevagent_final.yaml`; the frozen 4B's block unchanged) by `fill_backend_tau.fill`. The same replay
  with the frozen 4B is the check: its replayed argmax must agree with the logged answers on ≥ 99 % of the questions at
  each decision point; if not, the thresholds come instead from the registered calibration command on seeds 5000–5149
  (a shadow arm with the E5 model, Flash, n = 150). A model whose fit accepts no level at either decision point is not
  run live ("always defers; equals LLM-only by construction"). *Arms* (`configs/experiments/e5_small_live.yaml`):
  `llm_only`, `harness_final` (the frozen JevAgentRL-4B at its frozen thresholds) and `e5_final` (the E5 model at its
  fitted thresholds), the final-harness profile (tool + argument cascade, reflection off, `batch_s1`), both System-1
  models served by `serve_jev --fast` on the same GPU. *Data:* seeds **300700–300799** (n = 100) of the spare block of
  A7.2 (a), opened by this entry (300500–300699 were used by harness_final and its dedicated row; nothing has run on
  300700 or above, checked in the configs, the results and the S3 run records); the frozen suite settings (`families:
  v2`, `distractor_style: near`, `n_distractors: 60`, `noise_lines: 20`, `strict: true`); `JEV_EVAL_RUNS=1`.
  *Execution:* Gemini 3.5 Flash on the direct Gemini API (temperature 0, minimal reasoning; as the final harness; the
  key from SSM, never on disk or printed); ONE worker, so at most one episode is in flight on the GPU and every arm runs
  under the same worker setting in the same session; task-major interleaved (arm order seeded per task, seed 732);
  three passes, passes 2–3 re-running infrastructure errors only; a 5-episode development smoke (seeds 900–904) of the
  three arms first, operational fields only; only operational fields are read while it runs. *GPU tariff* q = host
  $ / h × the arm's mean episode wall-clock (one episode on a dedicated GPU), sensitivity q = 0. *Caps:* API about $15
  expected, $60 cap enforced by a watchdog on the logged spend (smoke included); GPU ≤ 3 h for the live row; the box is
  stopped afterwards. Nothing is tuned or re-run on 300700–300799 after any result is read. (h) *Analysis.*
  `scripts/e5_analysis.py` → `results/explore/e5_small/{summary.md, summary.json}` and `paper/generated/e5_small.tex`
  (keys `e5/...`): per model the v1 read-outs (accuracy by decision type, recovery, knockout, raw ECE), C accuracy
  (all, after an off-plan step, ask-vs-act) and C ECE, and paired task-bootstrap contrasts against the 4B references
  (B = 2,000, seed 0) with the reading rules of the plan §8 (keeps the 4B's accuracy if the interval's lower end is
  ≥ −1.0 point, loses it if the upper end is < 0, else inconclusive; one seed per model, so seed noise is not measured);
  latency p50 / p90 at c = 1 and 4 and the speed-up against the 4B; GPU memory; if run, the live row's success per arm,
  the paired differences against `llm_only` and against the 4B arm (intervals in the appendix only), cost per episode
  (API + q) and wall-clock and System-1 time per episode. Development and descriptive numbers only; outside every Holm
  table; reported whatever the outcome.
- 2026-10-06, recorded about 05:35 UTC, before any E6a episode (no ALFWorld train or valid_seen game has been run for
  E6a, and no valid_seen game has been run by this project at all: 2026-10-03 06:55 entry). **Exploratory row E6a:
  action-only control on ALFWorld valid_seen.** Outside both registered stages; labelled "exploratory (E6a)":
  **descriptive, revises no registered result, not the registered S2-L3b** (S2-L3b stays *not run (A7.1 (h): E6 not
  run in Stage 1)*; E6a selects nothing and changes no profile). Approved by the author on 2026-10-06 (proportionate
  experiments without confirmation; fill the module map's "not run" action row). (a) *Purpose.* The module map's action
  row (REPLACE, CASCADE, ASSIST) reads "not run" because E6 (`configs/experiments/alfworld_action_only.yaml`) never
  ran. E6a measures the three cells once, on the fresh split. (b) *Games.* ALFWorld text (alfworld 0.4.2, textworld
  1.7.0) through `bench/suites/alfworld.py` unchanged (oracle `auto`, `help` / `inventory` excluded, TextWorld budget 50
  steps). Evaluation: **all 140 valid_seen games** (`eval_in_distribution`) in the stratified order
  (`load_alfworld(..., stratify=True)`), frozen in `configs/explore/e6a_alfworld/eval_games.json` (id-list sha256
  `5a914a32fa171b1d…`; the 140 game files on disk are byte-identical to release 0.4.2's tw-pddl archive). Calibration:
  **48 ALFWorld train games**, 8 per task type, by ALFWorld's own train-split filters (no `movable` / `Sliced`, the six
  task types, `solvable`) in the order of sha256("jev-explore-E6a-alfworld-train-2026-10-06" + newline + "<task
  dir>/<trial>"), frozen in `configs/explore/e6a_alfworld/train_games.json` (list sha256 `9304ea5bb09248e8…`, with the
  release archives' URLs and sha256) and extracted alone to `data/raw/alfworld_e6a` (`scripts/explore_e6a_alfworld.py
  select-train`). (c) *Arms* (`configs/explore/e6a_alfworld/eval.yaml`): `llm_only` (the reference loop's action
  prompt), `jev_action_replace`, `jev_action_cascade_tauhat` (τ by (e)) and `jev_action_assist_k3` (k = 3, the
  registered tool-ASSIST k of C2F). (d) *Model settings and seeds.* Agent LLM `gemini-3.5-flash` on the direct Gemini
  API (allowed by the author on 2026-10-06), temperature 0, `reasoning_effort: minimal`, as in Stage 1; decision model
  Jev via OpenRouter (`{type: jev, provider: openrouter}`, pinned `typesafe/jev-1.13`), no System-1 cache (fresh
  answers, so wall time is not shortened by replays); `max_steps` 50, `history_window` 12 (E6 / `stage2_alfworld.yaml`).
  The games have no randomness (one single-game env per game, no domain randomisation; `docs/ALFWORLD.md` §4) and the
  LLM runs at temperature 0, so the only seed is the task-major schedule's (seed 732, as A7.16 (a)); one episode per
  game and arm. (e) *CASCADE threshold rule (fixed now).* Before any valid_seen episode the calibration shadow arm
  `jev_action_shadow_cal` (`configs/explore/e6a_alfworld/cal.yaml`: action `shadow`, the LLM acts, Jev is asked the
  same action question at every step and logged) runs on the 48 train games, and τ̂ is fitted with the repo's
  calibration-arm estimator at the action decision point, the one the tool cascade used
  (`threshold_theory.calibration_arm_run`: V = W (S_right − S_wrong) at W = $1 with the A6.11 fallback V = W, dC from
  the ledger, the scalar-threshold projection of Eq. 3, A6.5's sample-size rule; the gold is the environment's single
  gold action), by `scripts/explore_e6a_alfworld.py fit` → `configs/explore/e6a_alfworld/tauhat_action.json`. The
  fitted τ̂ is recorded in a second dated entry and written into `eval.yaml` before the smoke and before any valid_seen
  episode. If the fit accepts no confidence level, the cascade always defers (A6.11 (c) 4), i.e. it is the LLM-only
  harness: that arm is not run and its cell reads "always defers". Nothing is tuned or chosen on valid_seen. (f) *Smoke
  and caps.* Smoke: the four arms on the first 2 calibration train games (`smoke.yaml`; development, never reported as
  a result). **Cap: $80 of API** (Gemini + Jev as logged per episode at list prices, measurement calls included) for
  calibration, smoke and evaluation together. Projection = the smoke's mean $ per episode per arm, summed over the arms,
  × 140 × 1.5, plus the spend so far; if it exceeds $80 the evaluation runs the first 100 games of the list (same check),
  and if that exceeds too, E6a stops after the smoke. A watchdog (`explore_e6a_alfworld.py watch`) stops the runner
  when the logged spend reaches $80; an incomplete row is reported with its n. (g) *Execution.* In the development
  container (8 workers): one runner invocation for the four arms, task-major interleaved (seed 732); a second
  invocation re-runs episodes that ended in an infrastructure error (`error` set) once; nothing is re-run after that.
  (h) *Readout* (`scripts/explore_e6a_alfworld.py summarize` → `results/explore/e6a_alfworld/{summary.md,
  summary.json}`, keys `e6a/...` in `paper/generated/e6a.tex`): paired complete case over the evaluation list; success
  per arm and Δ vs `llm_only` in points (paired task bootstrap, B = 10,000, seed 0; the Newcombe interval beside it),
  paired wins and losses, cost per episode and saving % (task bootstrap), mean and median wall time, steps and calls per
  episode, the decision model's coverage (share of action decisions executed from its answer; ASSIST 0 by definition)
  and its accuracy against the single gold action (a lower bound). `scripts/module_map.py` shows the three cells marked
  exploratory, with its fixed descriptive verdict rule. No test; outside every Holm table; reported whatever the
  outcome.
- 2026-10-06, recorded about 05:45 UTC, after the E6a calibration run and before the E6a smoke and before any
  valid_seen episode (no valid_seen game has been run). **E6a: the fitted CASCADE threshold, and the execution host.**
  (a) *Host.* The first calibration attempt ran in the development container as registered (E6a entry (g)) and was
  aborted when that disk filled (other jobs' files): 22 of its 47 readable records ended at the TextWorld reset with
  `OSError: [Errno 28] No space left on device` and its last record was truncated. That attempt is discarded whole
  (none of its episodes enters the fit); its logged spend, $0.516 plus an allowance of $0.05 for the truncated record,
  counts toward the $80 cap (`PRIOR_SPEND` in `scripts/explore_e6a_alfworld.py`). Calibration, smoke and evaluation run
  instead on the CPU box `<instance-id>` (jev-hard-tb, m7i.8xlarge; SSM only, keys from Parameter Store) with
  `scripts/aws/e6a_alfworld.sh`: the same committed configs and code (the branch bundle), 8 workers, the same $80
  watchdog over every E6a run, outputs synced to S3. Other sessions edit `src/` on this branch, so the harness source
  (`src/`) of the smoke and the evaluation is pinned at the calibration's commit 4bace1e (`SRC_PIN`): every E6a episode
  runs the same harness code. Time label: the E6a entry above says "recorded about 05:35 UTC";
  it was committed at 05:29:13 UTC (d81a284), before the first calibration episode (05:29:38 UTC). Nothing else in it
  changes. (b) *Calibration.* `runs/e6a_alfworld_cal` (box): 48 of 48 train episodes, no infrastructure error, 707
  paired action decisions (JSONL sha256 `7c2807271f0a06ab…`). The fit (`configs/explore/e6a_alfworld/tauhat_action.json`,
  sha256 `5fcf8b406c4733ee…`): A6.5 rule `projection` (n_paired ≥ 30); V = $0.087 (counterfactual, 46 episodes with a
  wrong executed decision against the single gold action), dC = $0.00148 per decision; the accepting tail is the two
  top confidence bins ([0.95, 0.99) and ≥ 0.99, where Jev's and the LLM's accuracy against the single gold action are
  equal, 0.78 and 0.82), so **τ̂ = 0.95** (scalar accept rate 20 % of the calibration decisions). The advantage is not
  monotone in c (bins below 0.49 and at 0.69–0.88 also accept in isolation; scalar vs pointwise disagreement 0.49), as
  the estimator reports; the scalar projection is the registered rule and is deployed as fitted. Sensitivity rows: W =
  $0.1 → 0.09, W = $10 → 0.95, λ = 0.5 or 5 $/h → 0.95 (reported, not used). (c) *Deployed.* `jev_action_cascade_tauhat`
  runs at τ = 0.95 in `smoke.yaml` and `eval.yaml` (the `calibration: pending` placeholder removed). Arms, games, model
  settings, caps, rules and readout of the E6a entry are unchanged.
- 2026-10-06, recorded about 05:50 UTC, before any E6b episode (smoke included; no E6b configuration has run on any
  seed, and nothing has run on JevGym seeds 300800 or above: checked in the configs, `runs/` and the results).
  **Exploratory row E6b: completing the module map.** Outside both registered stages; labelled "exploratory (E6b)":
  **descriptive, no test, outside every Holm table, revises no registered result, selects nothing and changes no
  profile.** Approved by the author on 2026-10-06 (proportionate experiments without confirmation). (a) *Purpose.* The
  module map (paper Figure `fig:modulemap`, `scripts/module_map.py`) leaves most of its 8 × 5 (decision point, mode)
  cells empty. E6b classifies all 40 cells as implemented, newly meaningful, undefined or equivalent to another cell,
  with a one-line reason each (`results/explore/e6b_modes/cells.md`; machine-readable in `harness/config.py`:
  `VALID_MODES`, `E6B_CELLS`, `UNDEFINED_CELLS`, `ACTION_PENDING`). It implements the nine newly meaningful cells and runs
  each once. The action row stays with E6a, and action GUARD is meaningful but not implemented. (b) *The nine cells and
  their semantics* (`harness/agent.py`, `harness/memory.py`). **Tool arguments × ASSIST:** behind the registered C1
  tool cascade, when the model's tool is accepted, the model answers each bounded / copy argument's Choice. Its top k
  values form a shortlist, and the LLM fills the whole call in ONE forced call: string arguments are restricted by an
  `enum`, other types by the instruction, and a shortlist holding "other value" or every option restricts nothing.
  **Tool arguments × GUARD:** the LLM writes the call; the model answers each bounded / copy argument's Choice and
  vetoes the call once when it is confident (≥ τ) of another value than the LLM's (never on "other value"). The veto
  names the argument, no value. The LLM rewrites the call once, and the rewrite is executed unchecked, as the tool
  guard. **Memory write × CASCADE:** the model's store / skip at confidence ≥ τ, else the LLM gate (`memory_write: llm`,
  same prompt). **Memory write × GUARD:** the LLM gate proposes; a proposed write the model rejects (p_yes < τ) is not
  stored. **Memory read × CASCADE:** the model's top-k selection (the read assist, with its "none" option) when its
  top memory's confidence is ≥ τ, else every recent memory (`always`). **Memory read × TRIGGER:** the model decides
  before each step whether the agent needs information it stored earlier (yes: the recent memories, as `always`).
  **Reflection × CASCADE:** the model's reflect / skip when confident, else the agent LLM answers the same question in
  one yes / no call on the same state. **Stop × CASCADE:** the model is asked every step (the guard question at a
  finish proposal, the trigger question otherwise), and its stop / continue is executed when it is confident, else the
  LLM's own decision stands; this is the trigger at τ and the guard at 1 − τ in one arm. **Routing × CASCADE:** with
  Flash-Lite as the agent, the model picks Flash or Flash-Lite for the step when confident, else Flash-Lite answers
  the routing question. Every System-1 and System-2 call goes through the episode's ledger at its decision point (LLM
  gates as System-2). Gold labels are logged wherever JevGym exposes an oracle and never reach a backend (canary test
  `tests/test_e6b_modes.py`). The same file's fingerprint test shows that every pre-E6b mode makes the same decisions
  and calls as before. (c) *Thresholds, fixed now, nothing tuned on evaluation data.* The registered τ of the same
  decision point's trigger or guard where one exists, else τ = 0.9. A cascade's τ is a confidence threshold, and for a
  yes / no decision a cascade at 0.5 is the trigger itself, so the triggers' 0.5 cannot carry over: every cascade runs
  at **0.9** (for stop also its registered trigger τ). Memory-write guard **0.5** (the registered write trigger);
  argument guard and read trigger **0.9** (no registered τ there); shortlists **k = 3** (as tool ASSIST; also the
  registered read k). The argument assist sits behind the C1 tool cascade at Flash's registered τ̂\* = **1.00** (A6.5 /
  A6.11). (d) *Arms* (`configs/explore/e6b_modes/eval.yaml`): `llm_only` (the reference of every cell but routing),
  `small_only` (Flash-Lite alone, the routing reference as in C4; its comparison with Flash is reported beside it),
  and one arm per cell: `tp_assist`, `tp_guard` (tool choice with the LLM), `mw_cascade`, `mw_guard` (reads `always`, as
  the Stage-1 write rows), `mr_cascade`, `mr_trigger` (writes `always`), `rf_cascade`, `st_cascade` and `rt_cascade`.
  (e) *Tasks.* Fresh JevGym seeds **300800–300999 (n = 200)** of the held-back spare block of A7.2 (a), opened by this
  entry (300500–300699: the final harness and its dedicated row; 300700–300799: E5). Frozen suite settings (`families:
  v2`, `distractor_style: near`, `n_distractors: 60`, `noise_lines: 20`, `strict: true`); `JEV_EVAL_RUNS=1`. (f) *Models.*
  Agent LLM Gemini 3.5 Flash on the direct Gemini API (allowed by the author on 2026-10-06), temperature 0, minimal
  reasoning, as in Stage 1. The cheap LLM of the routing arm and of `small_only` is Gemini 3.5 Flash-Lite with the same
  settings. Decision model Jev via OpenRouter (`{type: jev, provider: openrouter}`), with no System-1 cache.
  (g) *Execution.* On the CPU box `<instance-id>` (jev-hard-swe, m7i.8xlarge; SSM only, keys from Parameter Store
  into the process environment; `scripts/aws/e6b_modes.sh`, the committed branch bundle; outputs to S3; a power-off guard
  and the box stopped when done), because the development container's disk is nearly full. One runner invocation runs
  all eleven arms task-major interleaved (arm order seeded per task, seed 732) with 8 workers. Passes 2–3 re-run only
  episodes that ended in an infrastructure error; nothing is re-run after that. (h) *Smoke and caps.* Smoke: every arm
  on 2 development tasks (seeds 900–901, `configs/explore/e6b_modes/smoke.yaml`; development, never reported as a
  result). **Cap: $150 of API** (Gemini + Jev as logged per episode at list prices) for smoke and evaluation together.
  Projection (`scripts/explore_e6b_modes.py project`) = the smoke's mean $ per episode per arm, summed over the arms,
  × 200 × 1.5, plus the spend so far. If it exceeds $150 the evaluation runs the first 100 tasks (300800–300899; same
  check); if that exceeds too, E6b stops after the smoke. A watchdog (`explore_e6b_modes.py watch`) stops the runner
  when the logged spend reaches $150, and an incomplete row is reported with its n. (i) *Readout* (`scripts/
  explore_e6b_modes.py summarize` → `results/explore/e6b_modes/{summary.md, summary.json}`, keys `e6b/...` in
  `paper/generated/e6b.tex`). The readout uses the paired complete case per cell against its reference. It reports
  success and Δ in points (paired task bootstrap, B = 10,000, seed 0; the Newcombe interval beside it), paired wins and
  losses, cost per episode and saving % (task bootstrap), coverage (the share of the point's decisions executed from the
  model's answer: cascade = answered at confidence ≥ τ, trigger 100 %, assist and guard 0 by definition), activity (as
  the module map: cascade = coverage, assist = 100 % with the binding share reported, guard = proposals vetoed, trigger =
  fired), the model's accuracy against the gold label, calls and wall time per episode, and the module map's fixed
  descriptive verdict. `scripts/module_map.py` shows the nine cells as exploratory tiles (never mixed into a Stage-1
  cell, count or module verdict) and the undefined and equivalent cells explicitly. Reported whatever the outcome.
- 2026-10-06, recorded about 06:15 UTC, during the E6b evaluation (only operational fields read: record counts per
  arm; no success, cost or decision of an E6b evaluation episode). **E6b addendum: concurrency 8 → 24.** At the
  author's request to finish the module-map cells sooner (relayed by the coordinating session), the E6b evaluation
  runner was stopped at about 06:12 UTC, after 16–18 tasks per arm at 8 workers. It resumes on the same box with **24
  workers**, in the same single task-major pool (seed 732), which runs all eleven arms in parallel. The runner skips
  every task whose latest record is complete, and the episodes in flight at the stop were not logged (their few cents
  count toward the cap only if logged). Everything else in the E6b entry is unchanged: arms, tasks, thresholds, models,
  the $150 cap and watchdog, passes 2–3 for infrastructure errors only, and the readout. Concurrency changes only
  waiting time. Every arm runs under the same pool, so the paired wall-time comparisons stay within one load level per
  task, but the reported wall time per episode mixes the two levels (about 9 % of the tasks at 8 workers). The readout
  says so.
- 2026-10-06, recorded about 06:25 UTC (no result read or changed). **Serving code after the A7.19 rows.** The v2 plan
  (`results/stage2/v2_eval_plan.json`) pins the serving code (`training/serve_jev.py`, `data.py`, `decision_model.py`)
  to the v2 freeze commit `4c0bdf8`. It was unchanged through `7a44310` (2026-10-05 04:02 UTC), the commit that
  completed every A7.19 (f) row, so all v2 rows ran on the freeze's serving code. Two later commits changed
  `serve_jev.py` for other work: `b81ed20` (2026-10-05 17:15 UTC; per-step batched multi-question requests and prefix-
  cache fixes for the final harness, behaviour-preserving and replay-checked) and `59a1227` (2026-10-06; the preencode
  flag read with `getattr` so test stubs work; no behaviour change). No v2 row is re-run or re-read. The check in
  `tests/test_stage2_v2.py` now verifies the pin over the span that produced the rows (`4c0bdf8` to `7a44310`) instead
  of the working tree.
- 2026-10-06, recorded at the commit that adds it (execution record, written after the E6a evaluation finished; it
  changes no rule, threshold, arm or game). **E6a: execution as run.** (a) *Times.* The "recorded about" labels of the
  two E6a entries above were estimates written before their commits and are later than the commits; the commits are
  authoritative: d81a284 at 05:29:13 UTC (first calibration episode, in the container, 05:29:38) and 749cecf at 05:41:29
  UTC (smoke start 05:41:52, first valid_seen episode 05:46:14). (b) *Execution.* The smoke projection was $13.0 for all
  140 games (`results/explore/e6a_alfworld/projection.json`), so all 140 ran. Evaluation: one invocation of 560
  episodes on the box, 05:46–06:22 UTC, task-major (seed 732, 8 workers), harness source pinned at 4bace1e; one
  infrastructure error (Jev API HTTP 520 in one `jev_action_replace` episode), re-run once by a second invocation
  (06:23 UTC) without error; nothing else was re-run, and no game left the paired comparison. Logged spend of E6a in
  total: $14.09 of the $80 cap (calibration $1.07, smoke $0.12, evaluation $12.33, the aborted container attempt
  $0.57). The box was stopped afterwards. Readout: `results/explore/e6a_alfworld/`.
- 2026-10-06, recorded about 06:30 UTC, after E5's smoke and while Qwen3-1.7B trains on the first host, before any
  E5 model has been evaluated and before any training step on the hosts below. **E5: three hosts instead of one.**
  For wall time (the author's request), E5 (f) runs on three g5.8xlarge instances (one A10G 24 GB each, all launched
  from the same image of the E3 host, us-east-1d): host A <instance-id> trains Qwen/Qwen3-1.7B and then runs the
  whole latency session (every model and the frozen JevAgentRL-4B in one session on its one GPU, as (e)) and the 4B's
  calibration-state replay; host B <instance-id> trains Qwen/Qwen3-0.6B; host C <instance-id> trains
  openbmb/MiniCPM5-1B (or its registered backup). Each host runs its own inputs check, smoke, training, evaluation and
  replays with `scripts/aws/e5_small.sh` (B and C with `E5_PART`, records under `s3://…/e5_small/part_<b|c>/`; adapters
  as before under `checkpoints/e5_small/<slot>`) and powers off when done; host A takes the B and C adapters and
  temperatures from S3 for the latency session and skips their training (marked done there). Models, revisions, data,
  recipe, smoke and memory rules, evaluation, latency protocol, live-row rule and analysis are unchanged; the GPU cap of
  (f) stays 14 host hours in total over the three hosts (caps B 3 h, C 4 h; host A's own use projected at about 6 h,
  its latency and 4B replay skipped if they would take the total over 14 h).
- 2026-10-06, recorded about 07:30 UTC, before any E6c episode (smoke included: the action guard has never run on any
  ALFWorld game, and E6c reads no E6a episode). **Exploratory row E6c: the action guard on ALFWorld valid_seen.**
  Outside both registered stages; labelled "exploratory (E6c)": **descriptive, no test, outside every Holm table,
  revises no registered result, not the registered S2-L3b, selects nothing and changes no profile.** Approved by the
  author on 2026-10-06 (fill every meaningful cell of the module map; proportionate experiments without confirmation).
  (a) *Purpose.* After E6a and E6b one meaningful cell of the module map has no measurement: action × GUARD (E6b entry
  (a): meaningful but not implemented; `ACTION_PENDING`). E6c implements it and runs it once, on E6a's games and
  settings. (b) *Semantics* (`harness/agent.py` `_guard_action`; `harness/config.py` `E6C_CELLS`, `ACTION_PENDING` now
  empty). It is the tool guard at the action point. The LLM proposes an admissible action with the reference loop's
  action prompt, exactly as in `llm_only`. The model is asked the tool guard's yes / no question about the proposal on
  the same System-1 state (`prompts.q_action_guard`: "The agent proposes to take the action `<a>`. Is this the right
  next step toward completing the task?") and vetoes it when p_yes < τ. A vetoed proposal is regenerated once
  (`guard_max_retries` 1): the same prompt with the tool guard's rejection note after the admissible list
  (`prompts.action_reject_note`; it names the vetoed action and gives no hint). The regenerated action is executed
  unchecked, as the tool guard does. Every System-1 and System-2 call goes through the episode's ledger at the action
  point. Gold (in the log only, never on the wire; canary test in `tests/test_e6c_action_guard.py`): the proposal equals
  the environment's single gold action, a lower bound of "right". The fingerprint test of `tests/test_e6b_modes.py`
  (fixture unchanged) shows that every other mode, the four other action modes included, makes the same decisions and
  calls as before. (c) *Threshold, fixed now.* **τ = 0.5**, the fixed threshold of the Stage-1 tool guard on Flash
  (`jev_ts_guard`, E2a, `configs/experiments/jevgym_tool_select.yaml`; the same τ as C7's "policy GUARD" and the τ²
  policy guard). Nothing is fitted or tuned. (d) *Arms* (`configs/explore/e6c_alfworld/eval.yaml`): `llm_only` (run
  again in this pool as the paired reference) and `jev_action_guard` (action guard, τ = 0.5). (e) *Games, models,
  seeds* as E6a (b), (d): all 140 valid_seen games of `configs/explore/e6a_alfworld/eval_games.json` (id-list sha256
  `5a914a32fa171b1d…`) in the same stratified order; ALFWorld through `bench/suites/alfworld.py` unchanged (oracle
  `auto`, `help` / `inventory` excluded, 50 steps); agent LLM `gemini-3.5-flash` on the direct Gemini API, temperature 0,
  `reasoning_effort: minimal`; decision model Jev via OpenRouter (`{type: jev, provider: openrouter}`), no System-1
  cache; `max_steps` 50, `history_window` 12; one episode per game and arm, task-major interleaved with seed 732. E6a's
  `llm_only` episodes are not reused, so the paired comparison shares one pool, load level and date; E6a's `llm_only` is
  reported beside it, descriptively and unpaired. (f) *Execution.* On the E6a CPU box `<instance-id>` (SSM only,
  keys from Parameter Store into the process environment; `scripts/aws/e6a_alfworld.sh` with `EXP=e6c`; the committed
  branch bundle, with the harness source `src/` pinned at the commit of this entry; outputs to S3; the box stopped when
  done). One runner invocation with **16 workers** (E6a ran 8, E6b 24 against the same API); a second invocation
  re-runs episodes that ended in an infrastructure error once; nothing is re-run after that. (g) *Smoke and cap.* Smoke:
  both arms on the first 2 of E6a's calibration train games (`configs/explore/e6c_alfworld/smoke.yaml`; development,
  never reported as a result). **Cap: $25 of API** (Gemini + Jev as logged per episode at list prices) for smoke and
  evaluation together. Projection (`scripts/explore_e6c_alfworld.py project`) = the smoke's mean $ per episode per arm,
  summed over the arms, × 140 × 1.5, plus the spend so far. If it exceeds $25 the evaluation runs the first 100 games of
  the list (same check); if that exceeds too, E6c stops after the smoke. A watchdog (`explore_e6c_alfworld.py watch`)
  stops the runner when the logged spend reaches $25, and an incomplete row is reported with its n. (h) *Readout*
  (`scripts/explore_e6c_alfworld.py summarize` → `results/explore/e6c_alfworld/{summary.md, summary.json}`, keys
  `e6c/...` in `paper/generated/e6c.tex`): paired complete case over the evaluation list; success per arm and Δ vs
  `llm_only` in points (paired task bootstrap, B = 10,000, seed 0; the Newcombe interval beside it), paired wins and
  losses, cost per episode and saving % (task bootstrap), mean and median wall time, steps and calls per episode. For a
  guard, coverage is the module map's activity: the share of proposals vetoed (the share of decisions executed from the
  model's answer is 0 by definition). Against the single gold action (lower bounds): the model's verdict accuracy, the
  LLM's proposal accuracy and the vetoed proposals' rewrite accuracy. `scripts/module_map.py` shows the cell as an
  exploratory tile of the action row (`mm/xcell/flash/action/guard/...`) with its fixed descriptive verdict rule (costs
  success if Δ ≤ −3 pt; no saving if saving ≤ 0; takes over safely if Δ ≥ −1 pt, saving > 0 and ≥ 5 % of proposals
  vetoed; else mixed), never mixed into a Stage-1 cell, count or module verdict. Reported whatever the outcome.
- 2026-10-06, recorded at the commit that adds it (execution record, written after the E6c evaluation finished; it
  changes no rule, threshold, arm or game). **E6c: execution as run.** (a) *Times.* The E6c entry above says "recorded
  about 07:30 UTC"; it was committed at 07:39:11 UTC (715fa33), before the first E6c episode (smoke start 07:40:21 UTC).
  The commit is authoritative. (b) *Execution.* On the box `<instance-id>` with `scripts/aws/e6a_alfworld.sh`
  (`EXP=e6c`), harness source pinned at 715fa33 (the bundle's head; src tree `b74e454e00a1` on the box). Smoke 07:40:21–
  07:40:49 UTC (4 episodes on 2 train games, no error). Projection (`results/explore/e6c_alfworld/projection.json`):
  $3.50 for all 140 games plus $0.03 spent, so all 140 ran. The smoke's train games were cheaper than the valid_seen
  episodes turned out to be ($0.008 against $0.027 per episode over both arms), which the 1.5× safety did not cover;
  the cap was never near. Evaluation: one invocation of 280 episodes, 07:41:33–07:57:49 UTC, task-major (seed 732, 16
  workers), no infrastructure error, so the registered second pass had nothing to re-run and did not run; no game left
  the paired comparison. Run files: `runs/e6c_alfworld_eval/llm_only.jsonl` (sha256 `1df5dc0fc45e338a…`) and
  `jev_action_guard.jsonl` (`a3eb7c9e5838f28b…`), synced to S3 under `e6c/runs/`; no key in any run or status file.
  Logged spend of E6c in total: $7.70 of the $25 cap (smoke $0.03, evaluation $7.67; Gemini $7.65, Jev $0.056). The box
  was stopped afterwards. Readout: `results/explore/e6c_alfworld/`; the module map's action row now has four
  exploratory cells (`mm/xcell/flash/action/guard/...`).
- 2026-10-06, recorded about 15:10 UTC, before any E6d episode (smoke included: no E6d mode has run on any seed, and
  nothing has run on JevGym seeds 301000–301199: checked in `runs/`, `configs/`, `scripts/`, the docs and
  `results/training/leakage_audit.json`, whose training seeds end at 72999). **Exploratory row E6d: defining the
  module map's remaining cells.** Outside both registered stages; labelled "exploratory (E6d)": **descriptive, no test,
  outside every Holm table, revises no registered result, selects nothing and changes no profile.** Requested by the
  author on 2026-10-06 (relayed by the coordinating session): define as many of E6b's undefined cells as is
  meaningful, and run them. (a) *Cells* (`results/explore/e6d_modes/cells.md`; `harness/config.py` `E6D_CELLS`;
  `harness/agent.py`, `memory.py`, `prompts.py`). **ASSIST at the four yes / no points, as "advise":** the binary
  analogue of a shortlist. The decision model answers the question, and its answer and confidence reach the LLM as a
  hint; the LLM decides. *Memory write:* the LLM write gate (same prompt) gets the advice; reads `always`.
  *Reflection:* E6b's LLM gate (the same question on the same state) gets the advice and decides whether the LLM
  reflects. *Stop:* before each step the model answers the stop-trigger question, and its answer and confidence go into
  the LLM's turn as a note; the LLM decides by its own call (`finish` or another tool). *Routing:* the cheap LLM
  (Flash-Lite, the default agent) answers the routing question with the advice and decides on escalation to Flash.
  **GUARD at reflection and routing:** E6b's LLM gate proposes reflect / escalate, and the model vetoes a proposal (not
  executed) when p_yes < τ; a "no" needs no check. **GUARD at memory read:** every observation is stored (`always`).
  Before each step the LLM selects the stored memories its next step needs (one call over the recent memories, ids or
  "none"); this proposer is new, the read analogue of the LLM write gate. The model is asked the tool-guard-style
  question about the selection and vetoes it when p_yes < τ. The LLM selects once more with a rejection note, and that
  selection is injected unchecked. **Stay undefined:** TRIGGER at tool choice, tool arguments and action (there is no
  optional module to skip). The five equivalent cells (REPLACE at the yes / no points and at memory read) are
  unchanged. Every call goes through the episode ledger. Golds are logged where JevGym has an oracle and never reach a
  backend (canary test `tests/test_e6d_modes.py`). Every pre-existing mode makes the same decisions and calls: the
  pre-E6b fingerprint fixture is unchanged, and a second fixture of the E6b / E6c modes was generated on the pre-E6d
  code. (b) *Thresholds, fixed now; nothing tuned on evaluation data.* Every guard at **τ = 0.5** (the Stage-1 guards'
  τ). The assists have no threshold. (c) *Arms* (`configs/explore/e6d_modes/eval.yaml`): `llm_only` (reference of every
  cell but routing) and `small_only` (Flash-Lite alone, the routing reference as in C4; the comparison with Flash is
  reported beside it), then one arm per cell: `mw_assist`, `rf_assist`, `st_assist`, `rt_assist`, `rf_guard`,
  `rt_guard`, `mr_guard`. (d) *Tasks.* **Fresh JevGym seeds 301000–301199 (n = 200)**, outside the reserved Stage-2
  block 300000–300999, which E6b exhausted. They are named here so that no other row uses them. The runner does not
  guard them, so no pilot or training run may touch them. The suite settings are E6b's (`families: v2`,
  `distractor_style: near`, `n_distractors: 60`, `noise_lines: 20`, `strict: true`). (e) *Models.* As E6b: Gemini 3.5
  Flash on the direct Gemini API (temperature 0, minimal reasoning) as the agent; Gemini 3.5 Flash-Lite with the same
  settings as the cheap LLM of the routing arms and of `small_only`; Jev via OpenRouter, with no System-1 cache.
  (f) *Execution.* On the CPU box `<instance-id>` (SSM only, keys from Parameter Store into the process
  environment; `scripts/aws/e6d_modes.sh`, the committed branch bundle; outputs to S3; a power-off guard, and the box
  stopped when done). One runner invocation runs all nine arms task-major interleaved (seed 732) with **24 workers**
  (E6b's resumed setting). Passes 2–3 re-run only episodes that ended in an infrastructure error; while it runs, only
  operational fields are read. (g) *Smoke and caps.* Smoke: every arm on 2 development tasks (seeds 900–901,
  `configs/explore/e6d_modes/smoke.yaml`; development, never reported). **Cap: $120 of API** (Gemini + Jev as logged)
  for smoke and evaluation together. Projection (`scripts/explore_e6d_modes.py project`) = the smoke's mean $ per
  episode per arm, summed over the arms, × 200 × 1.5, plus the spend so far. If it exceeds $120 the evaluation runs
  the first 100 tasks (same check); if that exceeds too, E6d stops after the smoke. A watchdog stops the runner at
  $120, and an incomplete row is reported with its n. (h) *Readout* (`scripts/explore_e6d_modes.py summarize` →
  `results/explore/e6d_modes/{summary.md, summary.json}`, keys `e6d/...` in `paper/generated/e6d.tex`). E6b's readout:
  paired complete case per cell against its reference; success and Δ in points (paired task bootstrap, B = 10,000,
  seed 0; Newcombe beside it); wins and losses; cost per episode and saving %; coverage (the share of the point's
  decisions executed from the model's answer: 0 for assist and guard by definition); activity as in the module map
  (assist 100 %, with the share of decisions where the LLM followed the advice reported; guard = proposals vetoed); the
  model's accuracy against the gold; calls and wall time per episode; and the module map's fixed descriptive verdict.
  `scripts/module_map.py` shows the seven cells as exploratory tiles, never mixed into a Stage-1 cell, count or module
  verdict. Reported whatever the outcome.
- 2026-10-06, recorded about 15:30 UTC, before any E7 evaluation episode (smoke included). No configuration has run on
  JevGym seeds 310000–310199: checked in `configs/`, `runs/`, the S3 run manifests and this registration. **Exploratory
  row E7 (JevGym-Hard).** Outside both registered stages; labelled "exploratory (E7)": **descriptive, no test, outside
  every Holm table, revises no registered result, selects nothing and changes no profile.** Approved by the author on
  2026-10-06 (relayed by the coordinating session; proportionate experiments without confirmation). (a) *Purpose.* On
  the frozen Stage-2 setting, Flash alone succeeds 99.4–99.5 %, which leaves no headroom. E7 builds a harder JevGym
  setting, tuned on development seeds until Flash alone succeeds 60–80 %, and runs the paper's main configurations on
  it once. (b) *Generator knob* (`multi_rule`, in `bench/suites/jevgym_hard.py`: a subclass of the JevGym env, used
  by `load_suite("jevgym", ...)` only when the suite block carries `multi_rule`. The frozen generator `jevgym.py` stays
  byte-identical, so its sha256 lock in `scripts/prereg_check.py` holds). The default `"given"` reproduces every
  task unchanged (fingerprints in `tests/test_explore_e7.py`). Under `"amount"` or `"tier"`, a `multi` task names only the
  customers and states an order policy, and the agent must work out each customer's operation from what the episode
  reveals. `amount`: cancel a pending order above $450.00, else ship it. `tier`: cancel it if the customer is basic
  tier and the amount is above $300.00, or silver tier and above $600.00; never cancel a gold-tier customer's order;
  otherwise ship it (express for gold tier, standard otherwise). Then finish with the total amount of the processed
  orders. The gold plan keeps its shape (find_user, get_orders, cancel or ship per customer, finish). Every oracle and
  gold is computed as before, and a seed's world, customers and orders are those of its `"given"` task. (c) *Pilot*
  (`configs/explore/e7_hard/pilot.yaml`; record `results/explore/e7_hard/pilot.{json,md}`). It ran on development seeds
  7000–7299, which no earlier config, run or S3 manifest had used. Flash LLM-only, temperature 0, minimal reasoning,
  30 tasks per candidate; base settings as Stage 2 (60 near distractors, 20 noise lines, strict). A first attempt on
  the direct Gemini API, at about 15:00 UTC, was aborted. The API answered HTTP 503 "high demand", trivial calls took
  9–60 s, and both of the attempt's records errored ($0). Every later pilot episode went through OpenRouter (see (g));
  none errored, and 4 of their calls needed one retry. Success of Flash alone: v2 families, 6 customers, given rule:
  100 % (clarify 15/15, multi 15/15); v2, 10 customers, given: 100 % (15/15, 15/15). Multi only: amount rule 8
  customers 100 %, 12 customers 66.7 %; tier rule 8 customers 90.0 %, 10 customers 76.7 %, 11 customers 80.0 % and
  76.7 % (two batches, 78.3 % pooled, n = 60), 12 customers 56.7 % and 36.7 % (46.7 % pooled, n = 60). Every failed
  pilot episode had all of its state-changing calls right and a wrong final total. Flash made no wrong cancel / ship
  call under either rule. A 4-episode route check (seeds 7290–7291, `llm_only` and `jevagent_best`, 6 customers) ran
  the cascade profile through OpenRouter without error. Pilot spend $17.64, route check $0.10: **$17.75 in total**.
  (d) *Chosen setting* (`configs/explore/e7_hard/eval.yaml`): **multi tasks only, 10 customers per task, `tier` rule**,
  60 near distractors, 20 noise lines, strict, `max_steps` 45 (gold plan 31 steps). Among the in-band candidates, the
  tier rule is preferred to the amount rule because its cancel-vs-ship decision depends on two observed attributes
  (tier and amount), and that decision is the tool choice the decision models make. Among the tier candidates in the
  60–80 % band, 10 customers (76.7 %) is the one closest to 70 % (11 customers: 78.3 %). (e) *Tasks.* Fresh JevGym
  seeds **310000–310199 (n = 200)**, outside every reserved, calibration, training and development range, and not E6d's
  301000–301199. The runner refuses them without `JEV_EVAL_RUNS=1` (`bench/registry.py` `EXPLORE_EVAL_RANGES`). (f)
  *Arms*, in one pool. (a) `llm_only`. (b) `jevagent_best`: the Stage-2 Jev profile exactly as frozen
  (`configs/harness/jevagent_best.yaml`, sha256 `85289f656c5610a6…`; tool cascade at Flash's Stage-1 fitted τ̂\* 1.00 /
  0.99, reflection trigger τ = 0.5). We run the whole profile, not the tool cascade alone, because it is the paper's
  main Stage-2 configuration and has the same structure as (c), so (b) and (c) differ only in the decision model and
  its thresholds. (c) `jevagent_best_rl`: the same profile on the frozen JevAgentRL-4B at its frozen thresholds (the
  profile's `backends.jevagent_rl` block, τ 0.5 / 0.9999999). Neither decision model was trained or tuned on the tier
  rule. (d) E5's small model is **not run**: E5 has not finished (its live row is pending). (g) *Models and route.* One
  deviation from the Stage-2 route: System-2 is **Gemini 3.5 Flash via OpenRouter** (`google/gemini-3.5-flash`,
  provider Google; temperature 0, minimal reasoning) in every arm and in the pilot, because the direct API answered 503
  at 15:00 UTC. The development route smoke of 2026-10-05 (`results/harness_final/route_smoke.json`) showed the same
  successes and call counts with about half the reported input tokens. E7's dollar figures therefore use OpenRouter's
  accounting and compare within E7 only. Jev via OpenRouter (`{type: jev, provider: openrouter}`), no System-1 cache.
  JevAgentRL-4B: the checkpoint of `results/stage2/rl_eval_plan.json` row `frozen`, sha256-checked before serving and
  fetched from S3 if absent. It is served by `training/serve_jev.py` with the plan's `serve_args` (none: S2-4's server)
  on the E7 GPU host, with no System-1 cache. (h) *Execution* (`scripts/aws/e7_hard.sh`). One new GPU host
  `<instance-id>` (Name=jev-e7-hard): a g5.8xlarge with one NVIDIA A10G 24 GB in us-east-1d, launched from
  image `<aws-resource-id>` because us-east-1d had no g5.4xlarge capacity. SSM only, with the System-2 key read
  from Parameter Store into each stage's process only. It runs the committed branch bundle. Smoke: the three arms on 2
  development tasks (seeds 7298–7299, `configs/explore/e7_hard/smoke.yaml`; development, never reported as a result).
  Evaluation: one runner invocation, all three arms task-major interleaved (arm order seeded per task, seed 732), **12
  workers**, so about four JevAgentRL episodes are in flight (S2-4's tariff concurrency). Passes 2–3 re-run only
  episodes that ended in an infrastructure error. If a pass leaves errored episodes, the next pass first waits (up to
  30 minutes) until the System-2 route answers a probe call again. Episodes still errored after pass 3 leave the paired
  comparison and are reported.
  (i) *Caps.* API: **$150 for E7 in total**, the pilot's $17.75 included. A watchdog (`scripts/explore_e7_hard.py
  watch`) stops the runner when the logged spend plus $17.75 reaches $150. Projection gate (`project`): the smoke's mean
  $ per episode per arm, summed over the arms, × 200 × 1.5, plus the spend so far. If that exceeds $150 the evaluation
  runs the first 100 tasks (same check); if that also exceeds, E7 stops after the smoke. GPU: **≤ 6 GPU-hours**. A
  power-off guard was armed at about 15:17 UTC for 20:47 UTC, 5.6 h after launch at 15:13 UTC. The driver re-arms it
  for the same end time or earlier, with a 30-minute idle stop, and the host powers off when the run ends. (j) *Readout* (`scripts/explore_e7_hard.py
  summarize` → `results/explore/e7_hard/{summary.md, summary.json}`, keys `e7/...` in
  `paper/generated/e7_hard.tex`). Paired complete case per arm against `llm_only` over the 200 tasks. Success and Δ in
  points (paired task bootstrap, B = 10,000, seed 0; the Newcombe interval beside it), and paired wins and losses. Cost
  per episode: API as logged, and for (c) also the GPU tariff q per episode under S2-4's provisioned convention (A7.13
  (iii)): q = $2.448 / h (g5.8xlarge on demand, us-east-1, AWS price list 2026-10-06) × the evaluation window (pass 1
  start to the last pass's end) / completed (c) episodes. Sensitivity rows: S2-4's frozen q and q = 0. Saving
  1 − C_arm / C_ref with a task-bootstrap interval. Coverage: the share of tool decisions (tool choice and arguments)
  executed from the decision model's answer, per decision point and pooled, and the reflection trigger's firing rate.
  Mean and median wall time per episode. The decision model's accuracy against the gold label on the run's own states:
  all of its tool answers, and the accepted ones. Its accuracy on the tier rule's **cancel-vs-ship decisions** is
  reported separately: tool choice where the gold next tool is a write, with coverage, accuracy, accepted-wrong count
  and the LLM's accuracy on the deferred ones. Each arm's failures are broken down by type, from the logged calls
  replayed in the deterministic environment: wrong final total; wrong state-changing call (strict stop or harmful
  side effect); ask / finish error (finished with writes missing, no final answer, or an ask in a multi task). Reported
  whatever the outcome.
- 2026-10-06, recorded about 16:30 UTC, after the E6d smoke and before any E6d evaluation episode (only the smoke's
  operational fields and API health probes were read). **E6d addendum: the OpenRouter route, because the direct
  Gemini API was overloaded.** (a) *Why.* The E6d smoke (15:24–15:30 UTC, development seeds 900–901, direct Gemini API
  as registered) met the overload of the direct Gemini 3.5 Flash API that the E5 row reported from about 15:00 UTC.
  Flash calls were retried in every Flash arm (5–88 s of retry waits per arm), the median call latency was about
  10 s instead of about 1.5 s, episodes took 90–330 s, and one smoke episode ended in HTTP 503 "This model is currently
  experiencing high demand"; Flash-Lite calls were normal. Running the evaluation on an overloaded API would have
  contaminated the comparison with errored episodes and inflated wall times, so it was paused. Health probes (six
  one-word Flash calls every 5 minutes from 15:33 UTC; healthy = all six succeed on the first attempt with a median
  under 2.5 s, twice in a row) did not pass by 16:30 UTC (log: `results/explore/e6d_modes/api_probe_log.txt`).
  (b) *Change.* **Every E6d arm** (`llm_only`, `small_only` and the seven cells) uses the OpenRouter route of the same
  models (provider Google): `{provider: openrouter, model: google/gemini-3.5-flash}` as the agent and
  `google/gemini-3.5-flash-lite` as the cheap LLM, temperature 0, minimal reasoning, priced at the Gemini list prices
  (`configs/prices/models.yaml`). Every cell is still compared with its own reference arm in the same task-major pool
  on the same tasks, so the paired comparison stays valid. Absolute costs are not comparable with E6b's direct-API
  costs: on this route the reported input tokens were about half the direct API's in the harness_final route smoke
  (Deviations 2026-10-05 "harness_final" (d)), and the E6d readout says so. (c) *Smoke.* A new smoke of every arm on
  the same two development tasks on the OpenRouter route (`configs/explore/e6d_modes/smoke.yaml`, run
  `e6d_modes_smoke_or`) gives the projection. The direct-API smoke (`runs/e6d_modes_smoke`, $0.63) stays counted toward
  the $120 cap. Seeds, arms, thresholds, caps, workers, passes and readout of the E6d entry are unchanged.
- 2026-10-06, recorded at the commit that adds it (execution record, written after the E7 evaluation finished; it
  changes no rule, setting, arm or task). **E7: execution as run.** (a) *Times.* The E7 entry above says "recorded
  about 15:30 UTC". It was committed at 15:34:28 UTC (2757bbf), before the first E7 episode on the host (smoke start
  15:50:33 UTC); the commit is authoritative. (b) *Setup.* On the host `<instance-id>` (g5.8xlarge, launched
  15:12:58 UTC), the setup stage of `scripts/aws/e7_hard.sh` stopped at 15:35 UTC on its own registration check. That
  check searched for the entry's title on one line, and the title wraps across two. Every earlier check had passed:
  the checkout was at 2757bbf and the venv imported it. The entry was then confirmed by hand in the host's checkout
  (one line of the title, found once), and the same committed driver ran the serve and smoke stages. The one-line
  check was committed afterwards (2a6f7e8); no arm, rule or setting changed. The frozen JevAgentRL-4B matched every
  sha256 of `results/stage2/rl_eval_plan.json` row `frozen` (`results/explore/e7_hard/box/frozen_verify.json`). It was
  served by `serve_jev` with no extra arguments (the plan's `serve_args` are empty; `box/serve.cmd`). (c) *Smoke.* 6
  episodes on seeds 7298–7299, 15:50:33–15:51:50 UTC, no infrastructure error (development, never a result).
  Projection (`results/explore/e7_hard/box/projection.json`): $0.123 per task over the three arms × 200 × 1.5 = $36.78,
  plus $18.00 spent, against the $150 cap: run all 200. (d) *Evaluation.* One runner invocation of 600 episodes,
  15:52:52–16:33:07 UTC, task-major (seed 732), 12 workers, harness at 2757bbf. There was no infrastructure error, so
  passes 2–3 had nothing to re-run and did not run, and no task left the paired comparison. Run files
  `runs/e7_hard/llm_only.jsonl` (sha256 `85e864899a653c54…`), `jevagent_best.jsonl` (`41eb742a40341854…`) and
  `jevagent_best_rl.jsonl` (`95df248654cce2a2…`), synced to S3 under `e7_hard/runs/`. No key appears in any run or
  status file (the driver's check). (e) *Spend and GPU.* E7's logged API spend in total is $43.01 of the $150 cap:
  pilot and route check $17.75, smoke $0.25, evaluation $25.02 (System-2 $23.06, Jev $1.95). The host ran from 15:13
  to 16:35 UTC, 1.37 GPU-hours of the 6-hour cap. Its serving window for the tariff was 15:52:52–16:33:07 UTC
  (0.671 h). The host powered itself off and was also stopped explicitly at 16:35 UTC. Readout:
  `results/explore/e7_hard/`.
- 2026-10-06, recorded about 16:45 UTC, before any E7b episode (smoke included). Nothing has run on JevGym seeds
  310200–310399 (checked in `configs/`, `runs/`, the S3 run manifests and this registration), and the E7b smoke seeds
  7296–7297 are unused. **E7b (JevGym-Hard, 12 customers): an addendum to exploratory row E7.** Labels as E7:
  **descriptive, no test, outside every Holm table, revises no registered result, selects nothing and changes no
  profile.** Requested by the author and relayed by the coordinating session after the E7 result had been read. (a)
  *Why.* The author asked for a setting where Flash alone succeeds 60–80 %. E7's evaluation gave 88.0 % (n = 200) with
  10 customers, about 11 points above its 30-task pilot (76.7 %). **The number of customers for E7b was chosen after
  seeing that E7's evaluation exceeded the band.** E7 stands as registered and is reported as run. E7b is reported
  beside it, and **both are reported whatever the outcome**. No part of E7 is re-run, replaced, or used to select
  anything. (b) *Setting.* As E7: multi tasks only, `tier` rule, 60 near distractors, 20 noise lines, strict,
  `max_steps` 45. The one change is **12 customers per task**, so the gold plan has 37 steps
  (`configs/explore/e7_hard/eval_b.yaml`). In E7's pilot (development seeds, E7 (c)), Flash alone succeeded on this
  setting 56.7 % and 36.7 % of the time in two 30-task batches (46.7 % pooled, n = 60), and every failure there was a
  wrong final total. (c) *Tasks.* Fresh seeds **310200–310399 (n = 200)**. The runner's guard now covers 310000–310399
  (`bench/registry.py` `EXPLORE_EVAL_RANGES`). (d) *Arms, models, route and readout* are exactly as in E7 (f), (g) and
  (j). Arms: `llm_only`, plus `jevagent_best` and `jevagent_best_rl` at their frozen thresholds. Route: Flash and Jev via
  OpenRouter, as in E7 and for the same reason. Model: the frozen JevAgentRL-4B, sha256-checked, served by `serve_jev`
  with the plan's `serve_args`. The readout is the same, including the failure types and the cancel-vs-ship split
  (`scripts/explore_e7_hard.py --exp e7b summarize` → `results/explore/e7_hard/summary_e7b.{md,json}`; keys `e7b/...` in
  `paper/generated/e7b_hard.tex`). The GPU tariff follows E7's convention over E7b's own evaluation window. (e)
  *Execution.* On E7's host `<instance-id>` (started again, stopped when done), with `scripts/aws/e7_hard.sh` and
  `EXP=e7b`. The first restart, at 16:44 UTC, found no g5.8xlarge capacity in us-east-1d. The host started as a
  g5.8xlarge at 16:45 UTC; it idles until this entry is committed. Records go to `results/explore/e7_hard/box_e7b` and `s3://…/e7_hard/e7b/`. First a smoke on development
  seeds 7296–7297 (`configs/explore/e7_hard/smoke_b.yaml`; never a result), then the projection gate, then one
  task-major runner invocation of the three arms with 12 workers. Passes 2–3 re-run infrastructure errors only, after
  the route probe of E7 (h). (f) *Caps.* **$60 of API for E7b**, smoke and evaluation together. E7 has spent $43.01 of
  its $150, so E7 and E7b together stay within $150. A watchdog stops the runner when E7b's own logged spend reaches
  $60. The projection gate is E7's with $60 as the cap: 200 tasks, else the first 100, else stop after the smoke. GPU:
  the host's hours count against E7's 6-hour cap (1.37 h used so far), with a power-off guard at 20:45 UTC.
- 2026-10-06, recorded about 16:55 UTC, after the E6d evaluation on the OpenRouter route and before any episode of
  the run below (only operational fields of the routing arms were read to find the fault). **E6d addendum: the
  routing cells are repaired on their own run.** (a) *Fault.* On the OpenRouter route, Gemini 3.5 Flash-Lite returned
  empty replies (neither text nor a tool call), which the loop reads as an empty `finish`. Such replies occurred in
  147 of the 200 `small_only` episodes and in 141 and 146 of the `rt_assist` and `rt_guard` episodes; 140 of
  `small_only`'s 149 failures are such episodes. On the direct API in E6b, the same reference succeeded at 85.0 %. The
  Flash arms of the same run had 0–1 such episodes. Both routing cells and their reference are therefore
  infrastructure-invalid. They are reported apart as excluded, with these counts, and never as results; the five
  Flash cells of the run stand. (b) *Repair.* The routing trio (`small_only`, `rt_assist`, `rt_guard`; τ, semantics
  and readout unchanged) runs again in `runs/e6d_modes_routing` (`configs/explore/e6d_modes/eval_routing.yaml`). Every
  Flash-Lite call goes to the direct Gemini API, whose Flash-Lite calls were healthy again at 16:49 UTC (six probes at
  0.5–1.2 s, first attempt), as in E6b. The Flash calls of escalated steps go through OpenRouter, because the direct
  Flash API was still overloaded at 16:49 UTC (5–21 s, one retry) and OpenRouter Flash showed no empty replies. The run
  uses **fresh JevGym seeds 301200–301399** (n = 200; unused, checked against `runs/`, the configs and the docs), the
  same suite settings, task-major (seed 732), 24 workers, three passes for infrastructure errors only, and the same
  $120 cap and watchdog over every E6d run. A smoke of the trio on development seeds 900–901
  (`smoke_routing.yaml`) checks for errors and empty replies first. The readout reads the two routing cells and their
  reference from this run. Nothing else changes.
- 2026-10-06, recorded at the commit that adds it (execution record, written after the E6d runs finished; it changes
  nothing registered). **E6d: what was run.** (1) *Direct-API smoke,* 15:24–15:30 UTC, development seeds 900–901,
  $0.63: Flash calls retried and slow, one episode ended in HTTP 503; the evaluation was paused and the box stopped.
  Health probes ran every 5–13 minutes from 15:33 to 16:27 UTC and never passed
  (`results/explore/e6d_modes/api_probe_log.txt`). (2) *OpenRouter route* (addendum committed at 7d59c34): the smoke
  ran at 16:31 UTC ($0.33; projection $48.8 at 1.5×, so all 200 tasks) and the evaluation from 16:32 to 16:47 UTC (24
  workers, 9 arms × 200 tasks, no infrastructure error, passes 2–3 empty). (3) *Routing fault and repair.* The fault is
  described in the addendum that follows the OpenRouter one. That addendum was committed at 16:51:04 UTC (3bed3c7),
  before the repair smoke (16:52 UTC) and run; its time label "about 16:55 UTC" is late by four minutes, and nothing
  else in it changes. Repair smoke: 6 of 6 episodes, no empty reply. Repair run: 16:54–16:58 UTC (3 arms × 200 tasks
  on seeds 301200–301399, no infrastructure error). The driver's first version did not sync that run directory, so the
  box was started once more at about 17:00 UTC only to copy it to S3, and stopped. (4) *Spend:* $40.57 for every E6d
  record (Gemini $40.12, Jev $0.45) of the $120 cap; the watchdog never fired. (5) *Readout*
  (`results/explore/e6d_modes/summary.md`): the five Flash cells come from `runs/e6d_modes`, and the two routing
  cells and their Flash-Lite reference from `runs/e6d_modes_routing`. The excluded OpenRouter routing records are
  listed with their empty-reply counts. Every cost is the route's own list-price cost (OpenRouter for the Flash calls),
  so E6d's absolute costs are not comparable with E6b's direct-API costs; each saving is paired within E6d.
- 2026-10-06, recorded at the commit that adds it (execution record, written after the E7b evaluation finished; it
  changes no rule, setting, arm or task). **E7b: execution as run.** (a) *Times.* The E7b entry above says "recorded
  about 16:45 UTC". Its text was committed at 16:51:04 UTC inside 3bed3c7, an E6d commit that staged the shared
  working copy of this file. The E7b configs, guard and driver followed in 7b53fae (16:52:01 UTC). Both commits come
  before the first E7b episode (smoke start 16:54:31 UTC). (b) *Host and setup.* On `<instance-id>`
  (g5.8xlarge), restarted at 16:45:33 UTC, with `scripts/aws/e7_hard.sh` and `EXP=e7b` at 7b53fae. The setup stage
  passed every check, including the registration check. The frozen JevAgentRL-4B matched every sha256 of the plan's
  frozen row (`results/explore/e7_hard/box_e7b/frozen_verify.json`), served with no extra arguments. (c) *Smoke.* 6
  episodes on seeds 7296–7297, 16:54:31–16:55:55 UTC, no infrastructure error (development, never a result).
  Projection (`box_e7b/projection.json`): $0.171 per task over the three arms × 200 × 1.5 = $51.26, plus $0.34
  spent, against the $60 cap: run all 200. (d) *Evaluation.* One runner invocation of 600 episodes, 16:56:26–17:45:03
  UTC, task-major (seed 732), 12 workers. There was no infrastructure error, so passes 2–3 did not run, and no task
  left the paired comparison. Run files `runs/e7b_hard/llm_only.jsonl` (sha256 `2fda75f02c83d5b5…`),
  `jevagent_best.jsonl` (`d9bcc240ae741ddf…`) and `jevagent_best_rl.jsonl` (`b96084ba12ec4552…`), synced to S3 under
  `e7_hard/e7b/runs/`. No key appears in any run or status file. (e) *Spend and GPU.* E7b's logged API spend is
  $31.47 of its $60 cap: smoke $0.34, evaluation $31.13 (System-2 $28.76, Jev $2.36). E7 and E7b together come to
  $74.48 of $150. The host ran from 16:45:33 to 17:46 UTC (1.01 h), 2.38 GPU-hours with E7's 1.37, against the 6-hour
  cap. The serving window for the tariff was 16:56:26–17:45:03 UTC (0.810 h). The host powered itself off and was
  also stopped explicitly. Readout: `results/explore/e7_hard/summary_e7b.{md,json}`, `paper/generated/e7b_hard.tex`.
- 2026-10-06, recorded about 18:00 UTC, after E5's offline results and before any episode on seeds 300700–300799 (only
  the first live smoke ran: 3 development episodes on seeds 900–904, operational fields only). **E5 live row: System-2
  through OpenRouter.** From 15:00 UTC the direct Gemini API answered Gemini 3.5 Flash with HTTP 503 ("high demand"):
  smoke episodes took 90–350 s instead of about 8 s, one errored, and one-token probes every 10 min from 15:32 to 17:46
  found no healthy round, while the OpenRouter route to the same model answered 5 of 5 probes in 0.7–1.5 s (17:47). E5
  (g)'s System-2 therefore changes for every arm (`llm_only`, `harness_final`, `e5_final`) from the direct API to
  OpenRouter (`google/gemini-3.5-flash`, temperature 0, minimal reasoning; the key from SSM on the host, never printed);
  everything else in (g) is unchanged (seeds, arms, profile, thresholds, one worker, interleaving, passes, smoke, caps,
  analysis). Consequences fixed now: success and time comparisons stay within E5 (one route for every arm); dollar
  figures are reported as within-E5 only, because this route reports about half the input tokens of the direct API
  (`results/harness_final/route_smoke.json`), so they are not comparable with harness_final's direct-API rows. If the
  route degrades during the row (an errored episode, or repeated retried LLM calls), the row is paused and reported;
  routes are never mixed within the row.
- 2026-10-06, recorded about 18:10 UTC, before any E7c episode (calibration included). Nothing has run on JevGym seeds
  310400–310599, the calibration seeds 5300–5399 or the smoke seeds 7294–7295 (checked in `configs/`, `runs/`, the S3
  run manifests and this registration). **E7c (JevGym-Hard, refitted thresholds): a second addendum to exploratory row
  E7.** Labels as E7: **descriptive, no test, outside every Holm table, revises no registered result, selects nothing
  and changes no profile.** Requested by the author and relayed by the coordinating session after E7 and E7b had been
  read. (a) *Why.* E7 and E7b ran both decision-model profiles at cascade thresholds fitted on the Stage-1 setting. On
  the new tier rule the frozen JevAgentRL-4B collapsed (E7b: 0.5 % success). At its frozen τ̂ 0.5 it takes every tool
  choice, and it is right on only 81.5 % of the cancel-vs-ship decisions. The paper's procedure (Sec. 2.1; PREREGISTRATION
  A6.5) fits one threshold per decision point and agent LLM, in the deployment setting, from a shadow calibration arm.
  **E7c was designed after E7 and E7b showed this failure, as a test of that calibrate-then-deploy procedure.** E7,
  E7b and E7c are each reported whatever the outcome, and no part of E7 or E7b is re-run or replaced. (b) *Setting.*
  E7b's: multi tasks only, 12 customers, `tier` rule, 60 near distractors, 20 noise lines, strict, `max_steps` 45.
  (c) *Calibration* (`configs/explore/e7_hard/cal_c.yaml`). Seeds 5300–5399 (n = 100) of the reserved calibration
  range 5000–5999, which is never trained on; earlier calibrations used 5000–5299. Flash LLM-only acts. Arm
  `jev_shadow_cal` asks Jev, and arm `rl_shadow_cal` asks the frozen 4B (served as in E7), every tool-choice and
  argument question in shadow mode: answers are logged and never executed, as in Stage 1's shadow arms. Stop and
  reflection are not shadowed; reflection keeps the profile's fixed trigger τ = 0.5 and is not a cascade. One pool,
  task-major (seed 732), 12 workers on the E7 host. *Fit:* the registered calibration-arm estimator per arm and
  decision point (`threshold_theory --calibration-arm --task-value 1.0 --lam-per-hour 0 --v-fallback W`, with A6.5's
  sample-size rule), run by `scripts/explore_e7_hard.py fit` → `results/explore/e7_hard/cal_e7c/`. Nothing is tuned
  beyond this rule. τ̂ = +∞ means the fit accepts no confidence level, and the cascade always defers there (A6.11 (c)
  4). At tool_select that is deployed as `always_defer` (System-1 is not asked, A6.11 (c)); at tool_params as τ = +∞
  (asked, never accepted). The fitted τ̂ go into a second dated entry and into `eval_c.yaml` and `smoke_c.yaml`,
  replacing `calibration: pending`, before the smoke and before any evaluation episode. (d) *Tasks.* Fresh seeds
  **310400–310599 (n = 200)**; the runner's guard now covers 310000–310599. (e) *Arms.* `llm_only`. `jev_refit`: the
  Stage-2 Jev profile's decision points (tool_select and tool_params cascades, reflection trigger τ = 0.5) at Jev's
  refitted τ̂. `rl_refit`: the same on the frozen 4B at its refitted τ̂. Everything else in these two arms equals the
  frozen profile arms (`tests/test_explore_e7.py`). The optional 4B tool cascade without the reflection trigger is
  not run, to stay within the cap. (f) *Route and readout.* As E7b: Flash and Jev via OpenRouter; the readout includes
  the failure types and the cancel-vs-ship split; keys `e7c/...` go in `paper/generated/e7c_hard.tex`. It adds the
  calibration fits, with each model's coverage and accuracy at its τ̂ on the calibration states. (g) *Execution.* On
  the E7 host (g5.8xlarge), with `scripts/aws/e7_hard.sh` and `EXP=e7c`. Stages: rearm, setup, serve, then the
  calibration (`cal`). The fit runs locally, then the second entry. Then the smoke on seeds 7294–7295 (development,
  never a result), the projection, the evaluation (one task-major invocation with 12 workers; passes 2–3 re-run
  infrastructure errors only), and the final stage. (h) *Caps.* **$70 of API for E7c, the calibration included.** A
  watchdog stops any E7c runner when E7c's logged spend reaches $70. The projection gate is E7's with a 1.2 safety
  factor instead of 1.5, because E7's and E7b's smokes predicted their evaluations' cost to within 2 % and 9 % (ratios
  1.02 and 0.91). If 200 tasks do not fit, the evaluation runs the first 100; if those do not fit either, E7c stops
  after the smoke. **GPU: at most 3 hours for E7c**, with a power-off guard 3 hours after the host starts.
- 2026-10-06, recorded about 18:10 UTC, during E5's live row (5 of 300 episodes run). **E5 live row: serving memory,
  watchdog and an inadvertent read.** (a) At 18:05 the watchdog paused the row after one `e5_final` episode errored:
  not the OpenRouter route but the MiniCPM5-1B server, which ran out of GPU memory because the frozen 4B's server on the
  same 24 GB A10G had grown its session cache to about 18 GB (each server's default cache budget is 65,536 tokens). For
  the rest of the row both servers run with `--cache-tokens 16384` (the session cache only reuses computation; read-outs
  are exact, and with one episode in flight 16,384 tokens hold the episode's prefixes). The errored episode is an
  infrastructure error and is re-run by the row's registered passes. (b) The watchdog now pauses on new errored
  episodes counted by source (route, System-1 server, other) and on more than 4 new retried LLM calls, relative to the
  state when the stage starts. (c) While diagnosing (a), a script printed the success field of the first five
  evaluation episodes (three tasks) next to their errors; nothing is changed, tuned or re-run on that basis, and the
  row continues exactly as registered (seeds, arms, thresholds, passes, analysis).
- 2026-10-06, recorded about 18:30 UTC, before any E6e episode (smoke included: no replace arm at memory write, memory
  read, reflection, stop or routing has run on any seed, and nothing has run on JevGym seeds 301400–301599: checked in
  every run's `experiment.yaml` under `runs/`, in `configs/`, `scripts/`, the docs and `results/`). **Exploratory row
  E6e: the five REPLACE cells, run as their own arms.** Outside both registered stages; labelled "exploratory (E6e)":
  **descriptive, no test, outside every Holm table, revises no registered result, selects nothing and changes no
  profile.** Requested by the author on 2026-10-06 (relayed by the coordinating session). Figure 5 drew five REPLACE
  cells as "= mode", because E6b had declared them equivalent to an implemented cell; the author wants them run, not
  declared. (a) *Cells* (`results/explore/e6e_modes/cells.md`; `harness/config.py` `E6E_CELLS` and
  `EQUIVALENT_NOTES`; `harness/agent.py`, `memory.py`). In each cell System-1's argmax decides alone, the LLM is never
  asked at that point, and the arm's τ is ignored. *Memory write:* store iff p_yes ≥ 0.5; reads `always`.
  *Reflection:* reflect iff p_yes ≥ 0.5. *Routing:* the step goes to Flash iff p_yes ≥ 0.5, else to Flash-Lite.
  *Stop:* at every step the model decides, with the guard question at a finish proposal (a "no" rejects it) and the
  trigger question at any other step (a "yes" ends the episode early). *Memory read:* every observation is stored
  (`always`), and the model's single top memory (or none) is injected, as in the read assist with k = 1. Each cell
  behaves exactly as the cell E6b named: the trigger at τ 0.5; for stop, the cascade at τ 0.5; for memory read, the
  assist with k = 1. The offline test `tests/test_e6e_modes.py` finds the same decisions, calls, outcomes and ledger
  records on simulated JevGym tasks once the mode label is set aside. The cells are run anyway, each as its own arm,
  so that each has its own paired numbers; their logs carry `mode: replace`. Every call goes through the episode
  ledger. Golds are logged where JevGym has an oracle and never reach a backend (canary test). Every pre-existing mode
  makes the same decisions and calls: the two earlier fingerprint fixtures are unchanged, and a third, of the E6d
  modes, was generated on the pre-E6e code. The TRIGGER cells at tool choice, tool arguments and action stay
  undefined. (b) *Thresholds.* None to set: the argmax (p_yes ≥ 0.5) and, for memory read, k = 1. (c) *Arms*
  (`configs/explore/e6e_modes/eval.yaml`), in one pool: `llm_only` (reference of every cell but routing), `small_only`
  (Flash-Lite alone, the routing reference as in C4; the comparison with Flash is reported beside it), and one arm per
  cell: `mw_replace`, `mr_replace`, `rf_replace`, `st_replace`, `rt_replace`. (d) *Tasks.* **Fresh JevGym seeds
  301400–301599 (n = 200)**, named here so that no other row uses them. The runner does not guard them, so no pilot or
  training run may touch them. The suite settings are E6b's. (e) *Models and route.* Gemini 3.5 Flash (temperature 0,
  minimal reasoning) is the agent; Flash-Lite, with the same settings, is the routing arms' cheap LLM and
  `small_only`'s agent; Jev runs via OpenRouter with no System-1 cache. The route follows a probe at 18:11 UTC (six
  one-token calls per route). The direct Flash API still answered slowly: 4.6–18.3 s per call, one call needing 3
  attempts. The OpenRouter route to Flash answered in 0.6–1.4 s and the direct Flash-Lite API in 0.5–1.2 s. E6e
  therefore uses the route of E6d's routing repair: **every Flash call on OpenRouter, every Flash-Lite call on the
  direct Gemini API** (OpenRouter's Flash-Lite returned empty replies in E6d). Each model has one route in every arm.
  Dollar figures are compared within E6e only, because the OpenRouter route reports about half the input tokens of the
  direct API. If the route degrades during the row (episodes with an empty reply, or repeated infrastructure errors),
  the row is paused and the pause is recorded; routes are never mixed within the row. (f) *Execution.* On the CPU box
  `<instance-id>` (`scripts/aws/e6e_modes.sh`, the committed branch bundle; keys from Parameter Store into the
  process environment; outputs to S3; a power-off guard, and the box stopped when done). One runner invocation runs
  all seven arms task-major interleaved (seed 732) with 24 workers. Passes 2–3 re-run only episodes that ended in an
  infrastructure error, and while the row runs only operational fields are read. (g) *Smoke and cap.* Smoke: every arm
  on 2 development tasks (seeds 900–901, `configs/explore/e6e_modes/smoke.yaml`; never reported). **Cap: $60 of API**
  (Gemini + Jev as logged) for smoke and evaluation together. Projection (`scripts/explore_e6e_modes.py project`) =
  the smoke's mean $ per episode per arm, summed over the arms, × 200 × 1.5, plus the spend so far. If it exceeds $60
  the evaluation runs the first 100 tasks (same check); if that exceeds too, E6e stops after the smoke. A watchdog stops
  the runner at $60, and an incomplete row is reported with its n. (h) *Readout* (`scripts/explore_e6e_modes.py
  summarize` → `results/explore/e6e_modes/{summary.md, summary.json}`, keys `e6e/...` in `paper/generated/e6e.tex`).
  E6b's readout: the paired complete case per cell against its reference; success and Δ in points (paired task
  bootstrap, B = 10,000, seed 0; Newcombe beside it); wins and losses; cost per episode and saving %; coverage and
  activity (both 100 % for replace, since the model decides every question); the model's accuracy against the gold;
  for stop, the rejected finishes and early stops; for memory read, the memories injected; calls and wall time per
  episode; and the module map's fixed descriptive verdict. `scripts/module_map.py` shows the five cells as exploratory
  tiles that note the cell they equal ("= trig.", "= casc.", "= assist"), never mixed into a Stage-1 cell, count or
  module verdict. Reported whatever the outcome.
- 2026-10-06, recorded at the commit that adds it (execution record, written after the E6e evaluation finished; it
  changes no rule, setting, arm or task). **E6e: execution as run.** (a) *Code.* The entry above and every E6e file
  were committed in aad6883 (18:32:06 UTC) and pushed before the first E6e episode; the box ran exactly that commit
  (its status line). (b) *Smoke.* 14 episodes on seeds 900–901, 18:33:23–18:33:50 UTC, on the CPU box
  `<instance-id>` (started at about 18:32 UTC): no infrastructure error, no empty reply, no retried call (development,
  never a result). Projection (`results/explore/e6e_modes/projection.json`): $0.112 per task over the seven arms × 200
  × 1.5 = $33.73, plus $0.22 spent, against the $60 cap: run all 200. (c) *Evaluation.* One runner invocation of 1,400
  episodes, 18:34:30–18:42:51 UTC, task-major (seed 732), 24 workers. No infrastructure error, so passes 2–3 had
  nothing to re-run, and no task left a paired comparison. Run files under `runs/e6e_modes/` (sha256 prefixes:
  `llm_only` `919d657c5c561cf7`, `small_only` `14a32954ffcb4fb3`, `mw_replace` `8f5dc46a79c1606e`, `mr_replace`
  `754913e244bf46ae`, `rf_replace` `4b35e8b97d844a68`, `st_replace` `ccc70d535a58d7c3`, `rt_replace`
  `79ef7e554ae4ac87`), synced to S3 under `e6e/runs/`; no key in any run or status file. (d) *Empty replies.* Entry
  (e) names episodes with an empty reply as a sign that the route degrades. A check at about 18:41 UTC (operational
  fields only) found 3 such episodes among the first 846 records. That is close to the rate E6d saw on the same
  OpenRouter route to Flash (2 of about 1,800 episodes) and nothing like E6d's Flash-Lite failure (147 of 200), so it
  was not read as the route degrading and the row was not paused. In total 5 of the 1,400 episodes had an empty reply
  (`llm_only` 1, `rf_replace` 2, `st_replace` 2; none on Flash-Lite). They stay in the paired comparison, as in E6d,
  and the readout lists them per cell. (e) *Spend.* $22.65 of the $60 cap (smoke $0.22, evaluation $22.42; Gemini
  $22.20, Jev $0.45). The box powered itself off at 18:43 UTC and was confirmed stopped. Readout:
  `results/explore/e6e_modes/summary.{md,json}`, `paper/generated/e6e.tex`; the module map draws the five cells as
  exploratory tiles noting the cell each one equals.
- 2026-10-06, recorded at the commit that adds it, after the E7c calibration and before any E7c smoke or evaluation
  episode (nothing has run on seeds 310400–310599 or 7294–7295). **E7c: the fitted thresholds, and a pause.** (a)
  *Times.* The E7c entry above says "recorded about 18:10 UTC". It was committed at 18:08:41 UTC inside 847fb86, an E5
  commit made in the shared working tree that included the staged E7c files, before the first E7c episode; the commit
  is authoritative. (b) *Calibration as run.* On `<instance-id>` (g5.8xlarge, started 18:07:30 UTC), with
  `scripts/aws/e7_hard.sh` and `EXP=e7c` at 797cd9a. The frozen 4B's sha256 all matched (`box_e7c/frozen_verify.json`).
  `runs/e7c_cal` ran 18:17:12–19:09:30 UTC: 100 tasks per arm, task-major, 12 workers. `jev_shadow_cal` had no
  error. In `rl_shadow_cal`, the last 9 tasks (seeds 5391–5399) ended in an infrastructure error. That arm's first
  agent call was answered HTTP 403 "Org member budget limit exceeded (monthly limit)": the OpenRouter key had reached
  its org's monthly budget. Logged spend $14.97 of E7c's $70 cap (System-2 $13.92, Jev's shadow answers $1.05). The
  box watchdog counts the shadow System-1 calls apart from the total, so it showed $13.92; the readout's `episode_spend`
  now adds them. (c) *Fits* (`scripts/explore_e7_hard.py fit` → `results/explore/e7_hard/cal_e7c/`; the registered
  estimator, W = $1, λ = 0, V = W by the A6.11 fallback because no System-2 decision was wrong in the log; A6.5's
  projection rule, since n_paired ≥ 30 throughout). The fits use every logged decision; refitting the 4B without the
  9 errored episodes gives the same τ̂. Results:
  - Jev, tool_select: **τ̂ = 0.91** (3,699 paired decisions). At τ̂ it takes 90.1 % of the decisions, with model and
    LLM both 100 % accurate on those.
  - Jev, tool_params: **τ̂ = 0.87** (5,998). It takes 80.2 %, both 100 % accurate on those.
  - JevAgentRL-4B, tool_select: **τ̂ = +∞: the fit accepts no confidence level, so the cascade always defers**
    (3,561). The model is 86.5 % accurate on all its tool choices, against the LLM's 100 %.
  - JevAgentRL-4B, tool_params: **τ̂ = 1.0** (5,781). It would take 62.9 %, at 99.97 % accuracy.
  - On the tier rule's cancel-vs-ship decisions (tool choice where the gold next tool is a write), Jev is 100 %
    accurate on 1,200 and takes 99.5 % of them at its τ̂. The 4B is 81.1 % accurate on 1,153, against the LLM's 100 %.
  (d) *Deployed* (`configs/explore/e7_hard/eval_c.yaml` and `smoke_c.yaml`, `calibration: pending` removed).
  `jev_refit`: tool_select cascade τ = 0.91, tool_params cascade τ = 0.87, reflection trigger τ = 0.5. `rl_refit`:
  tool_select cascade `always_defer` (System-1 not asked, A6.11 (c)); tool_params cascade τ = 1.0, which is reached only
  inside System-1-chosen calls and so never fires (as A6.11 (c) states for C1L); reflection trigger τ = 0.5 on the 4B.
  The refitted 4B profile is therefore the LLM's tool control plus the 4B's reflection trigger. (e) *Pause.* Since
  about 19:05 UTC every OpenRouter call answers 403 for the budget limit (confirmed by a probe at 19:12 UTC). Flash and
  Jev both run on that route, and Jev is reachable only there, so E7c's smoke and evaluation cannot run. The host was
  stopped at 19:11 UTC (1.06 GPU-hours so far of the 3-hour cap). E7c resumes unchanged on the registered route once
  the org admin lifts the limit; the thresholds above are final. Arms, tasks, caps and readout are unchanged. If E7c
  cannot resume, it is reported as calibration only.
- 2026-10-06, recorded about 19:30 UTC, after E5's live row stopped and before any of its outcomes was read (only
  operational fields: errors, spend, progress). **E5 live row: 90 of 100 tasks.** At 19:09 UTC OpenRouter refused every
  call with HTTP 403 ("Org member budget limit exceeded (monthly limit)"): the organisation's monthly OpenRouter budget
  was exhausted. Tasks 300700–300789 had completed in all three arms (no route error, no System-1 error, no retried LLM
  call in 1,517 calls; logged spend $3.18); tasks 300790–300799 (30 episodes) were not run, and the 4 episodes that
  errored on the 403 (tasks 300790–300791) are excluded as not run. No other route is used for them (routes are never
  mixed within the row). The registered rule applies unchanged: the analysis uses the tasks complete in every arm
  (90 tasks); the 10 tasks are reported as not run because of the budget.
- 2026-10-06, recorded about 19:35 UTC, before any episode of this row. Nothing has run on seeds 310400–310599 or on the
  smoke seeds 7294–7295. **E7c direct-API row: the part of E7c that does not need OpenRouter.** Labels as E7.
  Requested by the author and relayed by the coordinating session after the calibration and the pause above. (a)
  *Why.* The org's OpenRouter budget is exhausted (HTTP 403 since about 19:05 UTC), and Jev is reachable only through
  OpenRouter. This row runs the two E7c arms that need no OpenRouter call: `llm_only` and `rl_refit`. **`jev_refit` is
  not run.** (b) *Route.* Flash on the **direct Gemini API** (`provider: gemini`, `gemini-3.5-flash`, temperature 0,
  minimal reasoning) in both arms. That is the model and settings of the E7 / E7b rows, on Stage 2's original route.
  A probe at 19:30 UTC returned 16 of 16 calls with HTTP 200, but slowly (6–8 s each). Both arms use the same route,
  so success and cost compare **within this row only**: not with E7 / E7b, whose dollars use OpenRouter's accounting,
  and not across routes. (c) *Thresholds.* As fitted in the entry above, on E7c's calibration arm, which ran on the
  OpenRouter route; they are unchanged. `rl_refit`: tool_select cascade `always_defer`; tool_params cascade τ = 1.0,
  never reached; reflection trigger τ = 0.5 asking the frozen 4B, served as in E7 (sha256-checked)
  (`configs/explore/e7_hard/eval_cd.yaml`). (d) *Tasks.* As registered for E7c: seeds 310400–310599, n = 200 per arm,
  12 customers, tier rule, `max_steps` 45. (e) *Execution.* On the E7 host with `scripts/aws/e7_hard.sh` and
  `EXP=e7cd`; System-2 keys come from Parameter Store into each stage's process only. First a smoke of both arms on
  seeds 7294–7295 (`configs/explore/e7_hard/smoke_cd.yaml`; development, never a result). If the smoke meets HTTP 503
  or overload errors, the row stops and is reported; it does not retry in a loop. Then E7c's projection gate (safety
  1.2), then one task-major runner invocation (seed 732) with **24 workers**, so the slow route finishes within the GPU
  cap. Passes 2–3 re-run infrastructure errors only, after a route probe. (f) *Caps.* E7c's $70 in total. $14.97 is
  already spent, so the watchdog stops the runner when this row's logged spend plus $14.97 reaches $70. GPU: E7c's 3
  hours, of which 1.06 are used, so the power-off guard sits 1.9 hours after the host starts. (g) *Readout.* As E7b:
  success, Δ against `llm_only` with its interval, wins and losses, cost per 1,000 episodes, the tariff, coverage,
  cancel-vs-ship accuracy, failure types, and wall time. It is written by `scripts/explore_e7_hard.py --exp e7cd
  summarize` to `results/explore/e7_hard/summary_e7c.{md,json}`, with keys `e7c/...` in `paper/generated/e7c_hard.tex`
  (`e7c/jev_refit/state` = not run). The calibration fits are reported beside the row, and the route note appears in
  both the summary and the keys. Reported whatever the outcome.
- 2026-10-06, recorded about 19:40 UTC, after E5's 90-task live analysis and before any episode of the run below.
  **E5 live route diagnostic.** Motivation, read after the 90-task analysis: in the OpenRouter row the harness arms'
  extra System-2 time sits almost entirely in the LLM argument calls (decision point `tool_params`, one tool offered,
  about 400 input tokens; in `harness_final` median 1.36 s but p75 8.4 s, 93 of 269 calls above 5 s accounting for 920
  of 1,173 s; every call served by "Google AI Studio" took about 11.6 s, a timeout-and-fallback signature), while the
  69-tool `tool_select` calls in every arm show no such tail (median about 1.0 s); on the direct API the final harness
  was faster than LLM-only (harness_final dedicated row). The OpenRouter row stays the E5 live row as reported. The
  diagnostic re-runs the same 90 tasks (seeds 300700–300789) with the same three arms, profiles, thresholds, servers
  (`serve_jev --fast`, 16,384 session-cache tokens each), one worker and the same task-major order (seed 732), in its
  own run (`runs/e5_small_live_direct`, `configs/experiments/e5_small_live_direct.yaml`); the only change is System-2 on
  the direct Gemini API (provider `gemini`, `gemini-3.5-flash`, temperature 0, minimal reasoning; key from SSM on the
  host). Before it: a few one-token probes of the direct API and a 1-task smoke of the three arms on a development
  seed (900); if the API answers 503 or is overloaded, the diagnostic is not started. Three passes (passes 2–3 re-run
  infrastructure errors only); the watchdog pauses it on a new errored episode or more than 4 new retried LLM calls.
  Readouts: as the live row (success per arm, paired differences, cost per episode incl. q, wall-clock and System-1 time,
  coverage), on the tasks complete in every arm, plus the LLM latency per decision point (median, p90, mean, calls above
  5 s and their share of LLM time) and per served provider, for both routes. Dollar figures stay within each route.
  Descriptive; outside every Holm table. Caps: API $15, 3 instance-hours; host A stopped afterwards.
- 2026-10-06, recorded at the commit that adds it (no E7c evaluation episode has run). **E7c: the direct-API row
  stopped at its gate; E7c is on hold.** (a) *Direct-API row as run.* On the E7 host (started 19:32:26 UTC, at
  4784956), the smoke of both arms ran on seeds 7294–7295 from 19:42:45 to 19:44:36 UTC with no error and no 503. It
  cost $0.265 per episode for `llm_only` and $0.275 for `rl_refit`, with wall times of 78 s and 96 s per episode: about
  3.6 times OpenRouter's $0.073 per episode in E7b. The registered gate projected $0.539 per task × 1.2: $129 for 200
  tasks and $65 for 100, against the $70 cap with $16.05 already spent (calibration $14.97, this smoke $1.08). It
  therefore decided **"stop after the smoke"**. The figure the host printed, $31.01, counted the calibration twice:
  once as the prior and once from `runs/e7c_cal` on the host disk. The readout no longer counts it twice (`e7cd` has
  no extra run; the prior carries the calibration), and the decision is the same either way
  (`results/explore/e7_hard/box_e7cd/projection.json`). The host was stopped at 19:45 UTC. E7c's GPU time so far is
  1.27 hours of its 3. (b) *Hold.* The author does not raise the cap for the direct API. **E7c's evaluation waits until
  the OpenRouter budget is restored, then runs as registered: all three arms on OpenRouter** (`configs/explore/e7_hard/
  eval_c.yaml` with the thresholds of the fitted-thresholds entry, `EXP=e7c`). The direct-API row is not run. Its
  smoke is development only and never a result. E7c's spend so far is $16.05 of $70.
- 2026-10-06, recorded about 20:00 UTC, before any episode of this row. Nothing has run on seeds 310400–310499 or on
  the smoke seeds 7270–7271. **E7c Jev row, local: the refitted Jev cascade on JevGym-Hard.** Labels as E7.
  Requested by the author and relayed by the coordinating session after the E7c hold above. The author provided an
  OpenRouter key reserved for Jev calls (`OPENROUTER_API_KEY_JEV`, read only by `decision/jev.py`'s openrouter
  preset; never on AWS, never printed). (a) *Arms.* `llm_only` and `jev_refit`: the Stage-2 Jev profile's decision
  points at the thresholds fitted on E7c's calibration arm, unchanged (tool_select cascade τ = 0.91, tool_params
  cascade τ = 0.87, reflection trigger τ = 0.5; `configs/explore/e7_hard/eval_cj.yaml`). **`rl_refit` is not run
  live:** its calibration already turns its tool choice off, which is the reported result (the fitted-thresholds entry
  above). (b) *Tasks.* **n = 100**: E7c's first 100 registered seeds, 310400–310499, with E7c's setting (12
  customers, tier rule, `max_steps` 45). The arms run task-major in one pool (seed 732). (c) *Route.* Flash on the
  **direct Gemini API** (`gemini-3.5-flash`, temperature 0, minimal reasoning: the model and effort of the
  direct-API smoke). Jev via OpenRouter with the author's Jev-only key, with no System-1 cache. Both arms use the same
  System-2 route, so **dollar figures compare within this row only.** (d) *Execution.* Locally in the development
  container (no GPU), with 16 workers. Keys come from the container's `.env` and never leave it. First a smoke of both
  arms on development seeds 7270–7271 (`configs/explore/e7_hard/smoke_cj.yaml`; development, never a result), then the
  gate, then one evaluation invocation with `JEV_EVAL_RUNS=1`. A second invocation re-runs infrastructure-error
  episodes once, after a route probe; nothing is re-run after that. (e) *Caps.* **E7c's total cap becomes $90** (the
  author's decision; $16.05 spent before this row). The gate (`scripts/explore_e7_hard.py --exp e7cj project --prior
  16.05`) is the smoke's mean $ per episode per arm, summed over the arms, × 100 × 1.2, plus the spend so far. If that
  exceeds $90 the row stops after the smoke. A watchdog stops the runner when the row's logged spend plus $16.05
  reaches $90. (f) *Readout* (`scripts/explore_e7_hard.py --exp e7cj summarize` →
  `results/explore/e7_hard/summary_e7c.{md,json}`, keys `e7c/...` in `paper/generated/e7c_hard.tex`). As E7b:
  success, Δ against `llm_only` (paired task bootstrap, B = 10,000, seed 0, with the Newcombe interval), wins and
  losses, cost per 1,000 episodes and saving, coverage per decision point, cancel-vs-ship accuracy, failure types, mean
  and median wall time, and the latency breakdown by component, decision point and served provider. The calibration
  fits sit beside it (`e7c_cal.tex`). Reported whatever the outcome.
- 2026-10-06, recorded about 21:35 UTC, before any E7d step (no harvest, training or scoring has run). **E7d: re-run
  the recipe on the new rule.** Exploratory, labelled "exploratory (E7d)": **descriptive, no test, outside every Holm
  table, revises no registered result, selects nothing, changes no profile, and never replaces the frozen
  checkpoint.** Requested by the author and relayed by the coordinating session after E7–E7c. (a) *Why.*
  AgentDecision-4B (the paper's name for the frozen JevAgentRL-4B checkpoint of the records) is right on only 81 % of
  the tier rule's cancel-vs-ship decisions (E7b, E7c), and E7c's calibration turns its tool choice off. **E7d was
  designed after those results.** It asks whether re-running the training recipe on states of the new rule repairs
  the decision. The continued model is a **new checkpoint, "AgentDecision-4B (tier)"**. (b) *States, $0 API.*
  `scripts/explore_e7d_tier.py harvest`: the reference loop on the E7b / E7c setting (multi, 12 customers, tier rule,
  60 near distractors, 20 noise lines, strict, `max_steps` 45). The executor is the harness's simulated oracle LLM
  (`llm/simulated.py`, error rate 0.05) and System-1 is `data_v1.NearMissBackend` (eps 0.1 for tool, argument and
  gate). Both are simulated components that only choose which states are visited; every label is the JevGym oracle's
  gold, and no state or label comes from an LLM. **Training seeds 320000–320599** run under data_v1's behaviour
  policies `lite_shadow` and `nearmiss_replace`. **Held-out seeds 321000–321199** run under `lite_shadow` only. The
  training guards (`bench/registry.training_exclusions`, `training/seed_guard.TRAIN_FORBIDDEN`) now also exclude
  310000–310999 (the JevGym-Hard evaluation block) and 321000–321999 (E7d's held-out block). The frozen hashes
  (`jevgym.py`, the serving code) are untouched. (c) *Rows* (`build`, seed 0; quotas fixed now). The rows are
  data_v1's gold rows at tool_select and tool_params, deduplicated. 10 % of the training tasks form the tier dev split.
  From the rest: 900 tool-choice rows at write states (gold cancel_order or ship_order), 450 other tool-choice rows
  and 450 argument rows. Replay of the original mix: 1,200 rows sampled from E3's D100 file (the SFT v1 recipe's
  states, which AgentDecision-4B's SFT stage saw). The held-out set is 600 write-state, 450 other tool-choice and 450
  argument rows, all from the held-out seeds. (d) *Training.* **Continue the frozen adapter**
  (`training/train_decision.py --init <frozen adapter>`; LoRA r 32 / α 64, as frozen). The recipe's loss
  (soft_ce_brier, Brier weight 0.5), lr 5e-5 (half the SFT v1 rate, for a continuation), 1 epoch, effective batch 64,
  max length 4,096. Memory setting gc_3072, or gc_2048 on out-of-memory; capped at 100 minutes. Why continue rather
  than redo SFT v1 from base: it costs about an hour instead of E3's 2+ hours, it changes only what the new states
  teach the paper's model, and the replay guards the original states (checked on corpus C). Temperatures are refitted
  on v1 dev plus the tier dev split (`training/calibrate.py`). (e) *Offline scores (no live run).*
  - Held-out tier states (`training/eval_decisions.py`, with the served temperatures): accuracy per stratum
    (write-state tool choice, i.e. cancel vs ship; other tool choice; arguments) for both checkpoints.
  - **The E7c calibration states**, an independent read on seeds 5300–5399. The frozen 4B's logged questions are
    re-asked to both checkpoints through `serve_jev` (`scripts/e5_cal_replay.py`). The frozen checkpoint's replay must
    agree with its logged answers on ≥ 99 % (E5's check). The registered estimator then fits τ̂ per decision point for
    each checkpoint (W = $1, λ = 0, V fallback W). Reported per checkpoint: τ̂, coverage at τ̂, accuracy on all and on
    the accepted decisions, and the cancel-vs-ship subset. Jev's numbers come from its own E7c calibration arm (the same
    seeds, its own trajectories), since asking Jev on these states would cost API.
  - **Corpus C** (the original JevGym common states): the new checkpoint's accuracy against the frozen one's
    (`results/stage2/s23/answers_rl.jsonl`), as a paired task bootstrap of the delta. E5's reading rule applies:
    accuracy is kept if the interval's lower end is ≥ −1 point.
  - v1 dev accuracy.
  (f) *Execution.* The E7 host (g5.8xlarge), with `scripts/aws/e7_hard.sh` and `EXP=e7d`. Stages: `rearm setup
  d_inputs d_harvest d_train d_eval d_replay final`. No API key is on the host for this row. The adapter goes to
  `s3://…/checkpoints/e7d_tier/`, and the records to `s3://…/e7_hard/e7d/`. (g) *Caps.* **4 GPU-hours and $0 API**,
  with a power-off guard 3.9 hours after the host starts; the host powers off when done. (h) *Readout*
  (`scripts/explore_e7d_tier.py score` → `results/explore/e7d_tier/summary.json`, keys `e7d/...` in
  `paper/generated/e7d_tier.tex`). Reported whatever the outcome. A live evaluation of the new checkpoint needs a
  separate addendum.
- 2026-10-06, recorded about 22:35 UTC, while E7d's training runs and before any E7d scoring step (no temperature
  fit, held-out pass or replay has run; no E7d number has been read). **E7d: scoring within the GPU cap.** Changes
  only the size and order of E7d's offline scoring; harvest, rows and training are as registered. (a) *Why.* The
  measured rates do not fit the registered scoring into the cap. The $0 harvest took 35 minutes, and training runs 47
  optimizer steps at about 102 s each (about 80 minutes; the host started at 21:33 UTC and the guard is at 01:25).
  Forward passes on this host run at about 0.35 s per state. The registered scoring is about 35,000 such passes:
  temperatures on 7,713 rows, 3,000 held-out rows, the v1 dev files, corpus C's 3,939 questions, and the 8,827 E7c
  calibration-state questions for each checkpoint. That is about 3 GPU-hours, past the 4-hour cap. The cuts below
  are fixed now from time alone (seed 0), not from any result. (b) *Temperatures* are refitted on seeded samples of
  1,000 v1 dev and 1,000 tier dev rows (`explore_e7d_tier.py trim`) instead of all 7,713 rows. (c) *Held-out tier
  states.* The write-state stratum stays whole (600 rows). The other tool-choice and argument strata are sampled to
  225 rows each. Both checkpoints are scored on the same rows. (d) The v1 dev pass is dropped; corpus C stays the
  forgetting check, in full. (e) *E7c calibration states.* The serving-path check re-asks the registered checkpoint
  only the questions of the first 5 tasks (sorted task ids); the ≥ 99 % agreement rule is unchanged. The new checkpoint
  then answers the questions in file order (sorted task ids) until 01:00 UTC. Both fits use the tasks it answered in
  full (`complete`; at least 10, else no fit). The registered checkpoint's fit on those tasks reads its logged E7c
  answers, which are the answers its E7c fit used (the check confirms the replay reproduces them). Its 100-task E7c fit
  (tool choice always defer, arguments τ̂ 1.0) is reported beside it. (f) *Order.* Temperatures, then the held-out
  passes, then the check, corpus C, the calibration states, the fits and the score. (g) *Mechanics.* The first
  chain's top-level process is stopped while its `d_train` subshell keeps training. A second chain (`rearm setup d_wait
  d_eval d_replay final`, `scripts/aws/e7_hard.sh`) waits for the adapter's hash file and continues. The guard (01:25
  UTC), the 4 GPU-hour cap and $0 API are unchanged.
- 2026-10-07, recorded about 00:20 UTC, after E7d's held-out passes and before any corpus C or calibration-state
  answer of the new checkpoint (none had been produced). **E7d: one server at a time.** The replay stage served both
  checkpoints at once on the host's one 22 GB GPU. The registered checkpoint's server ran out of GPU memory once the
  states grew long, so 76 of the first 104 serving-path check answers were server errors. The stage is stopped and
  re-run with one server at a time. First the registered checkpoint alone answers the 5-task check (the errored
  answers are kept aside as `answers_frozen_check_oom.jsonl` and not used). Then the new checkpoint alone answers
  corpus C, then the calibration states. The calibration-state deadline moves from 01:00 to 01:14 UTC. To stay inside
  the 01:25 guard, `complete` stays on the host while the two fits and the score run off the host (same commands,
  CPU, $0). Nothing else changes, and the guard, the cap and $0 API are unchanged.
- 2026-10-07, recorded about 01:20 UTC, after the fits and the score (main session; the E7 agent's run ended with
  the container restart at about 00:55 UTC). **E7d readout (exploratory, descriptive).** On-host stages all exited 0:
  training (76 min, gc_3072), temperatures, held-out tier passes, the serving-path check (agreement 1.0 on tool choice,
  185 questions, and arguments, 300), corpus C (3,939 answers) and the calibration-state replay until 01:14 UTC, which
  re-answered the first 20 of the 100 E7c calibration tasks in full (2,017 of 8,827 questions; `cal_tasks.json`
  `enough: true`, above the 10-task minimum). `final` synced to S3 at 01:14 and the host powered off (stopping at
  01:16; guard and cap not extended; $0 API). Off the host, with the registered commands: the two fits on those 20
  tasks and the score (`results/explore/e7d_tier/summary.md`, keys `paper/generated/e7d_tier.tex`). Held-out tier
  states, new vs registered checkpoint: cancel/ship tool choice 95.3 vs 76.5 %, other tool choices 100.0 vs 91.6 %,
  arguments 98.2 vs 95.1 %. E7c calibration states: the tool-choice fit no longer always defers (tau-hat 0.9999,
  coverage 50.0 %, accuracy on cancel/ship 96.7 % vs 79.2 %); arguments tau-hat 0.9929 at 90.0 % coverage (registered
  checkpoint: 1 at 70.1 %). Corpus C 99.1 vs 99.2 % (-0.1 points, paired task bootstrap [-0.4, +0.1]). Cut by the cap:
  the other 80 calibration tasks, and the v1 dev pass (dropped earlier by "scoring within the GPU cap"). No live
  JevGym-Hard episode was run with the new checkpoint; nothing here is a test or revises a registered result.
- 2026-10-07, recorded about 04:30 UTC, before any E8 call (no gate question has been sent to any model; of the source
  log only the quantities of the registered C1 fit were read: counts, correctness, confidence bins, ΔC and κ).
  **Exploratory row E8: what decides whether the decision model's answer is accepted.** Offline; labelled "exploratory
  (E8)": **descriptive, no test, outside every Holm table, revises no registered result, selects nothing and changes no
  profile or threshold.** Requested by the author on 2026-10-07 (relayed by the coordinating session): the gate is a
  threshold on the decision model's own confidence; what if the decision model or an LLM judged acceptance instead?
  (a) *Data.* The registered calibration shadow arm `runs/e2a_jevgym_shadow_cal/jev_shadow_cal.jsonl` (sha256
  `fff21516e0aad243…`): JevGym calibration seeds 5000–5149 (150 tasks; outside every evaluation block, 100000–100499,
  200000–200499, 300000–300999, 310000–310599), Gemini 3.5 Flash as the agent, Jev answering in shadow at every state.
  It logs per decision the model's input state, options, answer, confidence field and distribution, the LLM's own
  choice and the answer-key label. Points: tool choice (1,125 decisions) and tool arguments (1,295 with an executable
  model answer). (b) *Sample.* 500 decisions per point (1,000 in all), drawn with seed 0 within each point and
  stratified in proportion by whether the model's answer is right (tool choice 30 wrong / 470 right, arguments 22 /
  478); listed in `results/explore/e8_gates/sample.json`. (c) *Gates* (a score for "accept the model's answer", on the
  same decisions). G1 the model's confidence field (the registered gate; free). G2 the margin between its top two
  options in the logged distribution (free). G3 a second decision-model call: Jev is asked on the same state whether
  its proposed answer is the right next step (the tool-guard wording; for arguments, the proposed value), score
  P(yes); one Jev call per decision via OpenRouter with the author's Jev-only key, run locally, no cache. G4 an LLM
  verifier: Gemini 3.5 Flash on the direct API (temperature 0, minimal reasoning), shown the state the model saw (for
  tool choice also the 69 tools with their one-line descriptions) and the proposal, answers yes / no and a 0–100
  confidence that the proposal is right; score = confidence / 100 (the verdict when no number comes back); one call
  per decision. G5 agreement with the LLM's logged choice (no new call here; deployed, it needs the agent's own LLM
  call and is charged so). Two-stage rows from the same scores, no new call: G1 accepts at the registered τ̂ (1.00 tool
  choice, 0.99 arguments) and G3 or G4 is asked only below it. No prompt holds the answer key or any gold field
  (`tests/test_explore_e8.py`). (d) *Metrics per gate and point.* AUROC of the score for "the model's answer is right"
  (ties ½; 95 % task bootstrap, B = 2,000, seed 0, and the paired difference to G1); selective accuracy at 25, 50, 75
  and 90 % coverage (ties in expectation, `metrics.selective_accuracy`); net economics under the paper's rule: the
  expected saving per decision against the LLM alone, Σ over accepted decisions of [ΔC − V (q − g)] / n, minus the
  model's own call κ and the gate's call, maximised over the gate's thresholds on these decisions (an in-sample upper
  bound). V = $1 per wrong decision (the registered fit's W, A6.11 fallback; $0.1 and $10 beside it); ΔC and κ are
  the logged per-decision LLM and model costs as in the registered fit (argument decisions share their step's call);
  the gate's cost and latency are measured in E8's own ledger (G5: the agent's call, so accepting saves nothing). The
  latency change at that threshold is reported beside it, and the same optimum with the gate free (its ranking alone)
  and for a perfect gate (the answer key; reference only). (e) *Execution and cap.* Local container,
  `scripts/explore_e8_gates.py` (`--dry-run` makes no call); every call through a Ledger, records in
  `results/explore/e8_gates/`. A 10-decision smoke per gate first (kept: part of the sample); the run continues only
  if the smoke's mean cost per call × 1,000 × 1.3 plus the spend so far stays under the cap, and a guard stops new
  calls at it. **Cap: $40 of API in total** (Gemini + Jev, as logged). Failed calls are retried in up to two passes; a
  decision still without a score is reported as missing, with its n. (f) *Readout*
  (`results/explore/e8_gates/summary.{md,json}`, keys `e8/...` in `paper/generated/e8_gates.tex`), whatever the
  outcome.
- 2026-10-07, recorded about 04:35 UTC, after all E8 calls and a first scoring (analysis only; no new call). **E8: the
  verifier's verdict as its score.** Execution as registered: smoke projection $3.12 (go), then every sampled
  decision scored by G3 and G4 without an error; logged spend $2.49. Every one of the 1,000 G4 replies gave
  CONFIDENCE 100, whatever its verdict (68 "no" replies included): the number is the verifier's confidence in its own
  verdict, not the probability that the proposal is right. The registered G4 score (confidence / 100) is therefore
  constant (AUROC 0.5; the best threshold accepts nothing). Added, post hoc and labelled so: **G4v**, the verdict as
  the score (yes 1, no 0; the registered fallback when no number comes back) from the same replies, and its two-stage
  row G1>G4v. The registered G4 rows stay in the readout beside them. Also added, post hoc: a **held-out saving**
  beside the in-sample optimum (the threshold chosen on half of the tasks and scored on the other half, both ways;
  mean of 10 random halvings, `random.Random("e8:cv:<r>")`), because at V = $1 the in-sample optimum rewards gates
  with many distinct scores (G2, G3) for finding a cutoff that happens to exclude every wrong answer; and a row for
  G1 at the registered τ̂ with no threshold chosen on these decisions (the deployed rule; the baseline of the
  two-stage rows, whose first stage is that fixed τ̂). Nothing else changes (sample, gates, metrics, costs, V,
  thresholds).

---

## A7.9 — S2-1's latency clause becomes a secondary row (2026-10-02, about 20:45 UTC; before any Stage-1 main-run result was read)

Written after review of A7 by the main session, with the same blindness as A7: no Stage-1 main-run output had been
opened, read or summarised. It changes one test of the Stage-2 family and the secondary table; the earlier text of A7
is not edited, and where it disagrees with this sub-amendment, A7.9 governs.

- **(a) S2-1.** S2-1 is the intersection–union test of two clauses: (i) non-inferiority on success, m = 3 pt
  (`ni_pvalue(...)["p_decisive"]`, `bb_ordered`), and (ii) cost (one-sided paired t-test of c_ref − c_arm). Its
  decisive p is max(p_(i), p_(ii)). The latency clause (iii) of A7.4 leaves the family. Claim wording: "JevAgentHarness
  [profile, Jev] is non-inferior to LLM-only on success and cheaper per episode".
- **(b) New secondary row S2-1T.** "JevAgentHarness[profile, Jev] is faster per episode than LLM-only": fresh JevGym
  seeds 300000–300499, Flash, n = 500, `jevagent_best` → `llm_only` (the same pair as S2-1); decided by the one-sided
  paired t-test of the per-task difference of `usage.total.latency_ms` (reference − arm; H0: E[d] ≤ 0; Student t with
  n′ − 1 degrees of freedom, `metrics.paired_t_test`, `p_greater`). Latency is as defined in A7.4 (the critical path of
  the loop: the sum of call latencies, excluding tool execution and client back-off). Reading: rejected → "faster per
  episode"; otherwise "no latency saving established at this n" when the two-sided 95 % CI of the mean difference
  covers 0, "slower by X s" when it lies entirely below 0. It is *not tested* (p = 1) exactly when S2-1 is (A7.1 (g) 5).
- **(c) The secondary table** now has four rows — S2-1T, S2-1L, S2-L3a, S2-L3b — with Holm within it at α = 0.05
  (levels 0.0125, 0.0167, 0.025, 0.05). The family of A7.4 keeps its four tests (S2-1, S2-2, S2-3, S2-4) and its Holm
  levels.
- **(d) Rationale.** The selection rule of A7.1 qualifies and chooses settings on success (non-inferiority) and cost
  only; latency never enters it (A7.1 (e) 3 prints it). A confirmatory test should not hinge on an endpoint the
  selection never looked at: an assist or guard profile can save cost (fewer System-2 input tokens, fewer LLM calls)
  while adding System-1 latency on every step, so the profile the rule is designed to pick could fail the family test
  for a reason the rule cannot see. Latency stays a registered, Holm-controlled secondary claim.
- **(e) Power (replaces the S2-1 sentence of A7.5's "Conjunctions" item and the levels of its secondary rows).**
  S2-1's power is at most the smaller of its two clauses'; on the pilot effect sizes the cost clause is near 1 for a
  cascade-type profile, so S2-1's power is that of its NI clause (the A7.5 table, unchanged). S2-1T: 80 % power at the
  secondary floor 0.0125 needs a mean per-task latency saving of 0.138 σ_d at n′ = 500; with the pilot σ_d of 10–21 s
  per episode (A7.5) that is 1.4–2.9 s per episode (the τ = 0.99 cascade saved 5.2 s in the pilot, the stop trigger
  none). With four secondary rows their floor is 0.05/4 = 0.0125 instead of 0.05/3; the exact NI powers at 0.0125 are
  in `results/reviews/2026-10-02/stage2_power.md` (column 0.05/4): S2-1L 0.22 (d = 20 %, Δ = 0) and 0.77 (Δ = +3 pt);
  S2-L3a 0.16 (d = 15 %, Δ = 0), 0.11 (d = 25 %, Δ = 0), 0.27 (d = 25 %, Δ = +3 pt); S2-L3b 0.11 (d = 10 %, Δ = 0), 0.07
  (d = 20 %, Δ = 0), 0.44 (d = 20 %, Δ = +5 pt). The expected reading of the L3 rows remains *inconclusive at this n*.
- **(f) Code** (lands in a separate commit, before any Stage-2 run): `scripts/stage2_analysis.py` (S2-1 without the
  latency clause; S2-1T in the secondary table) and its tests (`tests/test_stage2_analysis.py`); the power script's
  header text names the four-row secondary table.
- Cost: none.

---

## A7.10–A7.17 — Review round 9 (written 2026-10-02 late evening UTC, committed before 2026-10-03 03:30 UTC; before any Stage-1 main-run result was read)

Source: `docs/REVIEW_ROUND9.md` §1 (the round-1 proposals after both external models' round-2 red-team). The main
session adopted A7.10–A7.17 and **did not adopt A7.18** (the non-strict robustness pair: telecom carries the
external-validity question). Choices on the open points: the second System-2 is an open-weights, non-Google family via
OpenRouter (A7.12, verified below), `gpt-6.1-sol` only as an optional, budget-permitting descriptive row; telecom is
enlarged (A7.15); the Stage-1-seed mirror is removed (A7.13 (v)); the JevAgentRL freeze deadline is 2026-10-07
12:00 UTC with a target of 2026-10-04 (A7.10 (h)). The texts below are the drafts as adopted; where the code or a
verified fact says a draft was wrong, the text is adapted and an *Adaptation* line says so. Blindness as in A7: no
Stage-1 main-run output was opened, read or summarised. Where these sub-amendments disagree with earlier A7 text, the
later one governs.

### A7.10 — JevAgentRL is developed behind an actor-level barrier and confirmed on fresh, common states

Where it disagrees with A7.3 (c), A7.4 (S2-3) or A7.5 (S2-3's power), A7.10 governs.

- **(a) Development inputs, fixed now.** Before evaluation outputs are released to anyone who develops the model,
  `results/gpu_round5/TRAINING_MANIFEST.md` is committed: the training-input whitelist (`data/train_v0`, `data/train_v1`
  and its dev files, JevGym training seeds ≥ 10000 outside every reserved range (`registry.training_exclusions()`),
  development seeds ≥ 1000 for greedy development rollouts, and the 210-decision gate set); the complete recipe with its
  allowed ranges, maximum budgets (GPU hours, teacher $), the stage order, the checkpoint-selection order and every
  deterministic fallback (G3 uses the actual Intern-Decision v1-dev reference, never its round-2 proxy; the retention
  rule among several passing continuations is fixed, e.g. the highest v1-dev accuracy, then the later step). The gonogo
  script, not a person, keeps or drops each stage and names the frozen checkpoint.
  *Adaptation:* the draft moved GRPO's `eval_suite.seed0` default from 5000 to 1000; that code default is registered by
  A6.4 and checked by `scripts/prereg_effective.py`, so it stays, and the manifest sets the GRPO development rollouts to
  seeds ≥ 1000 in the run configuration instead (as OPD's `--eval-seed0 1000`).
- **(b) Actor-level separation.** Until the freeze, model development runs in an isolated workspace (the GPU host and a
  named development session or person) whose users and processes cannot access Stage-1 or Stage-2 evaluation outputs,
  profile-selection outputs, or summaries of them from other team members; the responsible developers, the workspace
  input manifest and the access controls are recorded. Anyone who has inspected evaluation results takes no
  model-development decision before the freeze. The freeze covers weights, adapter, tokenizer, prompts, option ordering
  and knockout logic, temperatures, serving code and the checkpoint-selection record (sha256 each, with the freeze
  time); after it, a separate evaluation workspace receives these artefacts. **A post-exposure training change is not
  authorised by documenting it:** if evaluation information reaches model development before the freeze, every
  comparison involving that checkpoint is exploratory and S2-3 and S2-4 enter the family with p = 1. Purely operational
  repairs after the freeze must reproduce the frozen model's predictions on a committed regression corpus and are
  documented before evaluation resumes.
- **(c) S2-3 replaced: fresh seeds, pooled common states.** Source arms on fresh seeds 300000–300199 (the first 200
  tasks of the Stage-2 list), Flash, three passes: `jev_ts_replace_f`, `intern_ts_replace_f`, `rl_ts_replace_f`
  (tool_select replace, n = 200 each; Intern-Decision-4B in-process and JevAgentRL served, loaded sequentially on the L4
  unless a development memory test commits a concurrent configuration) and Stage 2's own `llm_only` (its first 200
  tasks). All three backends are frozen before the corpus is collected. **Corpus.** For each eligible task t, C_t is the
  multiset of labelled pre-action tool-selection question occurrences in the final valid trajectory of each of the four
  source arms; every occurrence has equal weight (repeated questions and longer trajectories weigh more), and every task
  has equal weight in the mean. Each occurrence keeps task id, source arm, step, a canonical question hash (the exact
  System-1 state string, option list with descriptions and question text given to System-1; no oracle label, no source
  answer) and the label's provenance. **Scoring.** Every backend answers every distinct canonical question through one
  replay path that calls the live harness's own option handling with that backend — `JevAgent._choice_tournament` over
  the backend's `max_options` (Jev 255, so the 69 options of the suite are one read; Intern-Decision 62, so the harness
  tournament takes two batched reads; a served checkpoint by its declared `max_options` and the server's own 52-option
  knockout) — never reusing the answers that generated the source trajectories, through a new replay cache separate from
  rollout caches; replay order, batching, cache keys and seeds are committed before scoring; scores are expanded back to
  the occurrence multiset. acc_X(t) = mean correctness over C_t (`metrics.is_correct` against the A4 acceptable set).
  **Clauses** (one common task set): (i) `paired_t_test(acc_RL − acc_Intern)["p_greater"]`; (ii)
  `paired_t_test(acc_RL − acc_Jev + 0.05)["p_greater"]`; decisive p = max; S2-3 keeps its place in the family.
  **Claim:** "On the registered common-state distribution, JevAgentRL was more accurate than Intern-Decision-4B and no
  more than five percentage points less accurate than Jev" — about the occurrence-weighted distribution generated by
  these four frozen policies, not every policy's states or arbitrary deployment states. **Reported with it:** corpus
  size, unique questions, source-arm weights, labelled-state coverage, on-plan / after-an-off-plan-step strata, the
  LLM-only stratum; live success, own-state accuracy and trajectory length per replace arm (descriptive); task-level
  bootstrap only. A7.3 (c)'s Stage-1-seed comparison is withdrawn (A7.13 (v)).
  *Adaptation:* the draft's "the live harness's knockout beyond 52 options" describes `serve_jev` only; in the harness
  the split depends on each backend's `max_options` (code: `harness/agent.py::_choice_tournament`,
  `decision/intern_decision.py`), so the replay goes through that path per backend.
- **(d) Failures.** Source-trajectory infrastructure errors get the registered retries; a task without four valid source
  trajectories leaves the primary corpus (listed with the failing source), with a worst-case sensitivity row (excluded
  tasks at accuracy 0 for JevAgentRL and 1 for each comparator). Once C is frozen, a backend scoring failure is retried
  twice and then counted **incorrect** for that backend; invalid option outputs are model failures; no question leaves
  C. No frozen checkpoint → S2-3 and S2-4 *not tested* (p = 1).
- **(e) Power.** A7.5's S2-3 paragraph is withdrawn (it assumed the live-accuracy endpoint); the power of (c) is
  recomputed from a development corpus of `stage2_dev`-style runs on development seeds before the first S2-3 source
  episode and committed.
- **(f) GRPO validity gate.** No GRPO checkpoint is frozen unless the sampled action space of every trained decision
  point and mode is written down (categorical over options for replace; a stated stochastic accept / defer rule for
  cascade; a stated sampler for any top-k set or gate outcome) and tests show that, before any update, recomputing
  log-probabilities under the frozen behaviour-policy snapshot reproduces the stored behaviour log-probabilities within a
  stated tolerance; that after updates, probability ratios use current-policy log-probabilities against the stored
  behaviour ones; and that masking, option permutations, multi-round knockout decisions, normalisation and the declared
  KL estimator act on the same sampled action space. Deterministic thresholds or top-k operators are never given an
  invented probability; a mode without a tested sampler is excluded from GRPO training. If no audited GRPO checkpoint
  passes the gates by the freeze, the frozen model is the last gated SFT or OPD checkpoint and the paper makes no RL
  claim.
- **(g) Reward wording.** The reward used is recorded verbatim (form, λ, C₀, unit, irreversible penalty); with cost
  counted in System-2 calls the paper calls it "call-count-penalised success".
- **(h) Freeze deadline (AD-4).** The freeze record must exist by **2026-10-07 12:00 UTC** (target 2026-10-04). Without
  it, S2-3 and S2-4 are *not tested* (p = 1) and JevAgentRL is reported as a training study (a short paragraph and an
  appendix). Up to 48 L4 hours are reserved for training (SFT v1, OPD, GRPO; ≈ $63) on top of ≈ 24 L4 hours of serving
  (≈ $32).
- Cost: 600 source episodes × ≤ $0.043 ≈ $26 + replay of ≈ 6–8k questions (Jev ≈ $1–5) ≈ **$31**; GPU ≈ 2 h;
  engineering: the corpus builder, the Choice replay (`meta_gold.replay_backend` replays yes / no questions only), the
  analysis clause, the sampler tests.

### A7.11 — S2-2's reference and leave-one-group-out arms

Where it disagrees with A7.1 (g) 3, A7.3 (a) or A7.4 (S2-2), A7.11 governs.

- **(a) Reference.** For each System-2 section, `best_single` is the profile restricted to the selected
  JevGym-applicable group g\* whose selected candidate has the largest Stage-1 saving s (ties: lower authority rank,
  larger n, arm name): both arms are built from the same frozen profile and harness defaults, and the reference differs
  only by setting every group but g\* to `off` (a fixture test checks that no model, prompt, history, step budget or
  retry setting changes with the restriction). The candidate A7.1 (g) 3 as first written would have chosen is recorded
  in the profile with its Stage-1 statistics and **not run**. A7.1 (g) 4–5 are unchanged.
- **(b) Claim (replaces S2-2's).** "The composed profile is non-inferior to, and cheaper than, the selected component
  with the largest standalone Stage-1 saving." It is not a comparison with the strongest qualifying candidate overall or
  with the strongest component on fresh data. Clauses, margin and Holm level are those of A7.4.
- **(c) Leave-one-group-out (descriptive).** Let G be the active JevGym-applicable groups of the Flash section,
  |G| ≥ 2. For every g ∈ G the profile with g `off` runs on seeds 300000–300199 (n = 200, Flash, Jev, three passes) if
  the outcome-blind development cost estimate of all of them fits $45; otherwise only the profile with g\* removed runs.
  This is decided when the profile is committed, before any Stage-2 outcome exists. Configuration-identical rows reuse
  the same trajectories (with |G| = 2, "g\* removed" is the other group alone and "g′ removed" is `best_single`). Each is
  compared with `jevagent_best` on the identical tasks: success with a paired binary interval that stays non-degenerate
  at zero discordance (Tango / Newcombe score interval, `metrics.tango_interval` / `metrics.newcombe_interval`; a
  [0, 0] bootstrap is never shown), cost with the paired task bootstrap, latency, System-1 and System-2 calls, coverage
  per decision point. Removal effects are context-dependent and need not sum to the profile's effect; no row repairs
  or changes the profile.
- **(d) Attribution wording.** Per-decision-point ledger allocations describe accounting conventions, not causal
  effects; where avoided calls cannot be read directly from the execution trace they are reported as NA.
- **(e) Code** (before any Stage-2 run): `select_profile.py` (`best_single` per (a), the recorded original candidate),
  `harness/profile.py` (`profile_part: only_<group>` / `minus_<group>`), the fixture test of (a).
- Cost: |G| = 2 → one extra arm ≈ $9; |G| = 5 → five arms ≈ $43.

### A7.12 — Secondary row S2-1X: hosted cross-family transfer of the procedure

- **(a) Candidate pairs, in order** (strong X / same-family small X-small, via OpenRouter): (1)
  `deepseek/deepseek-v4-pro` / `deepseek/deepseek-v4-flash`; (2) `qwen/qwen3.5-122b-a10b` / `qwen/qwen3.5-9b`. Before
  selection a capability manifest per pair is committed: exact provider and model ids, source licence URLs and
  retrieval dates, weight / version ids where available, prices including cache and reasoning charges, context limits,
  request settings (temperature 0; reasoning off or minimal).
  *Verified for pair (1) on 2026-10-02 at 21:43 UTC* (OpenRouter's model and endpoint lists and one tiny call each from
  the project's key): `deepseek/deepseek-v4-pro` is listed as "DeepSeek V4 Pro 0423" (weights
  `deepseek-ai/DeepSeek-V4-Pro`, context 1,048,576), list price $0.2088 / $0.4176 per M input / output tokens (cache
  read $0.0174); `deepseek/deepseek-v4-flash` as "DeepSeek V4 Flash 0423" (`deepseek-ai/DeepSeek-V4-Flash`), $0.028 /
  $0.056 per M (cache read $0.0056); both calls returned 200 (V4-Pro served by StreamLake at $1.6 · 10⁻⁶; V4-Flash, not
  pinned, by OpenInference). Each model is served by more than fifteen providers at prices up to ≈ 9× the list price
  and at different quantisations, so the pin is load-bearing: the planned pin is the provider that serves both members
  at the list price (StreamLake, fp8) with `allow_fallbacks: false`, confirmed by the eligibility smoke of (b). Later
  snapshots exist under other ids (`deepseek/deepseek-v4-pro-0813`, `deepseek/deepseek-v4.1-flash`); the registered ids
  are the two above, and a change of the snapshot they resolve to before the X runs end is a deviation.
- **(b) Eligibility (development only).** Both members run LLM-only on the same development seeds 0–49
  (`-o suite.seed0=0`). A pair is eligible iff each member's malformed responses (those violating the harness's
  permitted response schema; a legitimate final-text answer is not malformed) are ≤ 5 % of steps, no infrastructure
  error remains after the A6.22 (e) retries, the context fits, and the served provider and model are identifiable and
  **pinned with fallbacks disabled** (if pinning cannot be enforced, the pair is ineligible). No success threshold
  selects a pair. The first eligible pair is X; no model switch after calibration or evaluation begins. No closed-model
  fallback is authorised by this amendment.
- **(c) Envelope and n.** From the development token accounting, n = 500 if the predicted total (both smokes,
  calibration, `llm_only_x`, `jevagent_best_x`, `llm_only_xsmall`, Jev calls, 20 % contingency) fits **$25**; else
  n = 300 if it fits; else S2-1X is *not tested (resource gate)*. Decided before the first calibration episode.
- **(d) Calibration.** Seeds 5150–5299 (inside the calibration range 5000–5999, never used before, excluded from
  training) are reserved for calibration from now on (this amendment and any later backend calibration).
  `jev_shadow_cal_x` (n = 150, X as System-2, Jev, otherwise identical to `jev_shadow_cal`) is fitted per decision point
  with the registered command (no accepted level → `always_defer: true`). Only cascade parameters are refit; nothing
  else is optimised on X.
- **(e) Profile.** A separate deterministic rendering step reads the committed selection artefact (it does not re-run
  the selection or re-read Stage-1 outputs) and writes a derived profile with its parents' hashes: the **Flash** section,
  every cascade τ replaced by X's fit, guard / trigger τ and assist k transferred zero-shot, a selected routing group
  routing between X-small (default) and X.
- **(f) Arms** (fresh seeds 300000–300000+n−1, Jev, three passes, under A7.16's scheduler): `llm_only_x`,
  `jevagent_best_x`, and the descriptive cheap-model reference `llm_only_xsmall`.
- **(g) Test.** S2-1X joins the secondary table of A7.9 (c), which then has five rows (S2-1T, S2-1L, S2-L3a, S2-L3b,
  S2-1X; Holm within it at α = 0.05, floor 0.01): NI on success (m = 3 pt, `bb_ordered`) ∧ cost (paired t-test). If no
  decision point of `jevagent_best_x` can intervene (every cascade always defers and no other group is active — all
  applicable decision points checked), S2-1X is *not tested (the fitted profile delegates nothing on X)*, reported as the
  procedure's prediction. Wording: "hosted cross-family transfer of the procedure", not a reproducible local deployment.
- **(h) Prices.** The pair's list prices (as verified in (a), re-read on the run date) enter
  `configs/prices/models.yaml` after the API main run ends (SCOPE_V2), before any X episode; unpriced → not analysed.
  Where OpenRouter reports `usage.cost` for the pinned endpoint, it is logged next to the list-price cost.
- **(i) Optional closed-family row (descriptive, budget-permitting).** `gpt-6.1-sol` (OpenAI Responses API; verified
  available with one tiny call on 2026-10-02, 21:43 UTC; frozen price $2.00 / $10.00 per M, cached input $0.10) with
  `gpt-6-luna` as its small model may run the same three arms under the same procedure (calibration on seeds 5150–5299,
  derived profile, n by the same rule against its own ≈ $70–80 envelope) only if Stage 2's remaining budget allows after
  every registered row. It is descriptive (no test, outside every Holm table), so whether it runs changes no test.
- Planning cost: ≈ 25k input and 1–3k output tokens per episode (Flash's ledgers) → ≈ $0.006–0.008 per X episode →
  at n = 500, 1,150 X episodes ≈ $8–10 + 500 X-small ≈ $0.5 + Jev ≈ $1.5 + smokes ≈ $1; inside $25 with the contingency.

### A7.13 — What the post-training buys, and at what price

Descriptive except (iii). Every row runs once, after the freeze record of A7.10 exists, in the evaluation workspace;
none selects or changes the frozen model.

- **(i) Rows.** The freeze record enumerates every evaluated checkpoint row with its training stages, hashes,
  temperature-fitting procedure and inference configuration: the **untrained base** (Qwen3-4B, no adapter, served by
  `training/serve_jev.py` with JevAgentRL's prompt, option-code readout and a v1-dev temperature fit done exactly as for
  JevAgentRL), **SFT v1**, **SFT v1 + OPD** (when it precedes the frozen checkpoint) and the **frozen model**. A stage
  never trained is *not run*, stated; identical artefacts share a row.
- **(ii) Evaluation of the rows:** A7.10's common-state corpus C (replay; no API); E1 on the C8 population; live
  tool-selection replace on seeds 300000–300099 (n = 100, Flash) for the base, SFT v1 and SFT v1 + OPD rows (the frozen
  model's row reuses its first 100 A7.10 tasks). Readout: accuracy on C by stratum, ECE, AUROC, live success and
  own-state accuracy, with task-bootstrap intervals. Stage comparisons describe the retained continuation packages
  (data v1; OPD continuation), not the causal benefit of refreshing the behaviour policy; no isolated OPD-mechanism claim
  is made.
- **(iii) Provisioned tariff for S2-4 (supersedes the open part of A7.3 (e)).** After the checkpoint and profile freeze,
  a dedicated development workload is measured: `jevagent_best_rl`'s configuration (final profile, serving stack,
  session cache, workers 4, Flash executor) on a committed list of development tasks, with the serving window's
  boundaries committed before measurement; the window includes executor waiting and provisioned idle time and no
  unrelated work. The tariff q = host $ / h × measured serving hours ÷ completed development episodes is frozen before
  any priced evaluation and charged to each evaluated JevAgentRL episode on top of its API cost; S2-4 is conditional on
  this declared deployment tariff. Reported beside it: hardware, precision, batching, cache policy, concurrency,
  start-up allocation, retries, the realised evaluation-host bill, the busy-throughput tariff, Jev's API price and $0
  as sensitivity rows. If the measured workload does not match the evaluated deployment, S2-4 is not analysed under q.
  *Adaptation:* the ledger prices System-1 per input token (`configs/prices/models.yaml`); the per-episode tariff q is
  charged by the Stage-2 analysis (`scripts/stage2_analysis.py`) on top of the episode ledger, with the per-token entry
  of the served model id set to 0 so that nothing is charged twice.
- **(iv) External reference agreement.** After the freeze and after the C7 analysis is complete, the frozen model,
  Intern-Decision-4B and the base answer the common set of System-1 questions logged by the Stage-1 τ² arms (airline
  test, retail), scored as **agreement with the A6.22 (b) reference labels** (not oracle accuracy), with the label
  construction and missing-label coverage stated: an offline distribution-shift diagnostic, not evidence of τ² success
  or savings.
- **(v) Mirror removed (supersedes A7.3 (c)'s arm list and A7.7's mirror line).** The Stage-1-seed JevAgentRL mirror
  (`configs/experiments/jevgym_tool_select_rl.yaml`, 6,200 episodes) is not run. A7.10's fresh rows, the common-state
  replay, E1, the calibration arms (`jevgym_shadow_cal_rl.yaml`) and S2-4 replace it. No Stage-1 confirmatory
  comparison changes.
- **(vi) Wording.** JevGym comparisons with Intern-Decision-4B are "generator-adapted vs off-the-shelf"; E1 and the τ²
  replay are labelled by their actual populations; no "state of the art" or "best open decision model" claim.
- Cost: (ii) 300 live episodes ≤ $13 + (iii) the development workload ≈ $2–5 ≈ **$20**; mirror removal **−$160** and
  −5–7 GPU h; GPU for (ii)–(iv) ≈ 3 h, models loaded sequentially.

### A7.14 — `best_control`: the strongest listed non-decision-model control (descriptive)

- **(a) Candidate table, exhaustive** (`configs/harness/control_candidates.yaml`, committed with this amendment's
  configs, before any Stage-1 main-run result is read): one row per registered **Flash** arm whose System-1 is neither
  Jev nor Intern-Decision and that acts at a JevGym-applicable decision point — the retrieval, heuristic, prior, random,
  rate-matched, random-score, System-2-self and Flash-Lite-as-System-1 arms of E2a, E2b and E2c, threshold-sweep arms
  included — plus `small_only` (Flash-Lite alone, E2b). Each row names its config path, arm, reference arm, expected
  task count, intervention type, its activation statistic (read from its own intervention mechanism; the Jev-specific
  delegation function is not applied to controls without those records; void for `small_only`) and duplicate-setting
  precedence. No runtime discovery adds candidates.
  *Adaptation:* "committed now" in the draft reads "committed with the configs of this amendment", a separate commit
  after this text and still before any Stage-1 main-run result is read.
- **(b) Selection.** In the same single run as A7.1, `select_profile.py` applies A7.1 (d)'s complete-case, NI and cost
  screens and the row's activation screen; the qualifying control with the largest observed saving (ties: larger n, arm
  name) is `best_control`, run unchanged on the fresh seeds (n = 500, three passes), reusing an existing fresh arm when
  the configuration is identical. Stage-1 savings are selection-biased; the fresh row is the estimate.
  `results/stage2/control_selection.json` keeps every row's statistics and fate. If none qualifies: "no listed control
  satisfied the prespecified Stage-1 screen", with the table.
- **(c) Reporting.** In the Stage-2 score card and cost–success plot beside `llm_only`, `llm_only_s2lite`,
  `jevagent_best` and `best_single`, with differences to `llm_only` and to `jevagent_best`; no test. It is a selected
  baseline, not the best possible non-decision-model system.
- Cost: ≤ 500 × $0.043 ≈ $22 (a routing or assist control ≈ $8–12).

### A7.15 — S2-L3a on a larger random sample of telecom tasks

Replaces the telecom population of A7.2 (b) and S2-L3a's endpoint in A7.4; Stage-1 C7 is unchanged.

- **(a) Inventory and inference unit.** Before any telecom simulation, an inventory of the pinned tau2-bench telecom
  `full` pool is committed (`configs/tau2/stage2/telecom_inventory.json`: every id, its split memberships, issue type,
  persona, fault composition and a duplicate audit; as read by the review on 2026-10-02: 2,285 ids, 1,984 `mms_issue`,
  254 `mobile_data_issue`, 47 `service_issue`, all distinct fault-composition × persona strings, no parent-scenario
  field — the committed inventory, not these counts, decides). The inference unit is the task id; the eligible
  population is the pool minus any exact duplicates. Under the registered superpopulation assumption (tasks as
  independent draws from the eligible population), the paired NI procedure is used as in Stage 1; its iid-trinomial
  size certification is not claimed as a finite-population or cluster-sampling guarantee.
- **(b) Validation tasks.** 32 eligible tasks are reserved by seeded random order (seed 731) for adapter validation and
  cost measurement, and never evaluated.
- **(c) Size by an outcome-blind cost rule.** Let V be the actual validation spend and ĉ the mean cost per simulation
  across both validation arms (LLM-only and the rendering; failed attempts included). n = min(800, N_eligible,
  ⌊(250 − V) / (2 × 1.25 × ĉ)⌋), frozen with the evaluation list before evaluation; no success number of the validation
  tasks is read. The 25 % factor is a planning reserve, not a spend cap; evaluation never stops on observed success or
  effects, and a forced resource stop yields an incomplete, explicitly non-confirmatory row.
- **(d) Sample.** A simple random sample without replacement (seed 731) of n eligible, non-validation tasks, frozen with
  its sha256 in `configs/tau2/stage2/telecom_task_ids.json`. Issue-type and `base`-split results are descriptive; no
  quota enters the test.
- **(e) Arms and endpoint.** `llm_only` and `jevagent_best` (the τ² rendering of A7.1 (i)), one attempt per task each,
  user-simulator seed paired per task; A6.22 (e) retries for infrastructure errors only. S2-L3a: NI on **single-attempt
  task success**, m = 5 pt (`bb_ordered`) ∧ cost per task (paired t-test), worded as single-attempt success; never
  compared directly with majority-pass or pass^k estimates.
  *Adaptation:* the draft put the telecom pair under A7.16 (a)'s scheduler; τ² simulations run through tau2's own
  runner (`suites.tau2 run`, one arm at a time, its own concurrency), which the JevGym scheduler does not drive. The two
  arms therefore run back to back with identical settings and per-task seeds; S2-L3a has no latency endpoint, so the
  scheduler's purpose (a fair latency comparison) does not arise, and τ² latency is descriptive.
- **(f) Degenerate cases.** An all-`off` rendering → not run (A7.1 (i)). If adapter validation cannot be completed
  without changing evaluation semantics, S2-L3a is *not tested*, never moved to another domain.
- Power is recomputed exactly (`scripts/stage2_power.py`) at the frozen n before evaluation. Planning: at ĉ = $0.15 and
  V ≈ $10, n ≈ 640; at ĉ = $0.12, n = 800; at n = 640–800, discordance 15–25 %, Δ = 0 and the Holm floor 0.01, the
  normal approximation gives ≈ 0.6–0.9 (registered design: 0.11–0.27). Cost ≤ $250 including validation (registered:
  $85–100); wall clock ≈ 8–12 h at concurrency 8.

### A7.16 — Execution, accounting, reading rules and claims

- **(a) Scheduler.** Comparable JevGym API arms (S2-1, S2-2, S2-1X, S2-1L, A7.11's rows) run through one committed
  task-major scheduler with the arm order randomised within each task (seed 732), fixed concurrency and a recorded
  schedule; shared reference arms run once; request settings frozen. If the scheduler is not available before launch,
  **S2-1T enters the secondary table with p = 1** and latency is descriptive. Episode wall-clock time is reported
  separately from the registered model-call latency endpoint.
  *Adaptation:* the telecom pair is outside the scheduler (A7.15 (e)); `bench/runner.py` runs arm-major today, so the
  scheduler is new code (a task-major mode of the runner), tested before launch.
- **(b) Cache accounting.** One committed convention for all arms: System-1 cache hits are charged their replayed cost
  and latency (the A6.22 (e) 8 convention) and counted separately as physical calls vs ledger charges; cache order may
  not give any arm an artificial cost or latency advantage (each arm's System-1 answers come from the fresh Stage-2
  cache files in scheduler order).
- **(c) Reading rules (added to A7.4).** Every test reports the joint adjusted decision, each clause's raw p-value, its
  null boundary (−m for NI) and its effect interval. A favourable nominal interval with a non-rejected family test
  reads "favourable estimate; the registered joint claim was not established"; failure is attributed to multiplicity
  only when the clause and family decisions support it. "Within five points" is never written for S2-3 (it is one-sided
  NI: "no more than five points less accurate").
- **(d) Intervention reporting.** For every profile arm: query coverage; accepted System-1 choices (replace, cascade);
  strict candidate reductions (assist); vetoes (guard); module-state changes relative to the default policy (trigger);
  and directly observed LLM calls avoided. Query coverage alone is never called intervention; a "negligible delegation"
  label names its numerator and denominator (5 % of opportunities, A6.20 (e)).
- **(e) Outcome-to-claim map.** The abstract's and introduction's finding sentences are chosen here and filled only with
  registered wording:

  | outcome | sentence (template) |
  |---|---|
  | S2-1 / S2-4 rejected | "On fresh tasks, the selected profile (with JevAgentRL as System-1) was non-inferior to LLM-only within 3 points of success and X % [CI] cheaper per episode." |
  | S2-1 / S2-4 not rejected | "The registered joint cost-and-success claim was not established; the estimated success difference was Δ [CI] and the cost difference D [CI]." |
  | empty profile (A7.1 (g) 5) | "No decision point qualified in Stage 1: for these suites the procedure keeps every decision with the LLM." |
  | S2-2 rejected | "Composition added X % [CI] savings over the selected standalone component at success non-inferior within three points." |
  | S2-2 not rejected / not tested | both clause estimates and "the joint claim was not established" / "the selected profile delegates at one group; composition was not tested." |
  | S2-3 rejected | "On the registered common-state distribution, JevAgentRL was more accurate than Intern-Decision-4B and no more than five points less accurate than Jev." |
  | S2-3 not rejected | both clause estimates, without converting either clause into a confirmed claim. |
  | Stage 1 | each primary in its registered label; the abstract names at most C1F, C8 and C6F and counts the others by label. |

  "Best practice" names the procedure (shadow-log, fit, qualify per decision-point group by non-inferiority and cost,
  compose, default to the LLM), never a claim that delegation pays; "faster" only for a rejected S2-1T.
- **(f) A7.9 (d)'s rationale restated.** Latency is secondary because the study's primary objective is cost at
  non-inferior success, not because a confirmatory test may only test endpoints its selection rule used.
- **(g) Total spend and release.** The paper reports total experiment spend (calibration, failed attempts, teacher
  calls, training, GPU hours) apart from per-episode costs, and the deployment volume at which JevAgentRL's training and
  calibration cost is amortised. JevAgentRL's weights and Jev- or Gemini-labelled data are released only with a dated
  terms matrix (source, version, licence or terms; permission to train on outputs, to redistribute outputs and derived
  weights) showing permission; otherwise the release is code, recipes and seeds, and the paper says so.

### A7.17 — Timestamp, effective specification, launch gate

- **(a) Snapshot and anchor.** The registration documents (PREREGISTRATION.md, PREREG_A7_STAGE2.md with A7.10–A7.17,
  PREREG_EFFECTIVE.md, the generated effective Stage-2 specification of (c), the A7.14 candidate table, the frozen task
  lists `configs/eval/jevgym_eval_tasks.json`, `configs/eval/jevgym_stage2_tasks.json`,
  `configs/eval/e1_c8_population.json`, `configs/tau2/airline_task_ids.json`) are committed as snapshot commit C. A
  manifest then lists C and the sha256 of those files (it contains neither its own hash nor that of a later commit), is
  stamped with OpenTimestamps (only the hash leaves the machine) and its receipt committed separately. The initial
  calendar receipt and the later upgraded proof (`ots upgrade`, `ots verify`) are both kept and ship with the
  supplementary material. The verified anchoring time and the time evaluation outputs are released to anyone are
  recorded; if the verified anchor postdates that release, the paper says so. Later amendments are stamped the same
  way. A first stamp of this text and of the documents that exist now is made at once (`results/stage2/timestamps/`);
  the snapshot C with the generated specification and the candidate table is stamped when they land.
- **(b) What it proves.** The stamp shows that this content existed by the anchoring time; blindness and training
  independence rest on the access and freeze records (A7.10), not on the stamp.
- **(c) Effective specification and launch gate.** `scripts/prereg_effective.py` (or a sibling) generates one executable
  Stage-2 specification — every arm, population, endpoint, family, estimator, failure rule, resource gate and
  prerequisite — from A7 as amended, kept in sync by a test like Stage 1's. Fixture tests cover an empty profile, a
  single-group profile, zero discordance, scoring failures and an unavailable backend. The statistical code of each test
  is frozen (committed and hashed) before its evaluation outputs become accessible. No Stage-2 run launches before its
  fixtures, the effective specification, A7.10's access boundary and its cost gate pass.

### A7.10–A7.17: tables and budget as amended

- **The Stage-2 family** (unchanged in membership; Holm at FWER 0.05): S2-1 (A7.9: NI ∧ cost), S2-2 (A7.11's reference
  and claim), S2-3 (A7.10 (c): fresh common states), S2-4 (A7.13 (iii)'s tariff).
- **The secondary table** (Holm within, α = 0.05, floor 0.01): S2-1T, S2-1L, S2-L3a (A7.15's endpoint and population),
  S2-L3b, S2-1X (A7.12).
- **Descriptive:** `jevagent_default`, `best_control` (A7.14), the leave-one-group-out rows (A7.11 (c)), the attribution
  rows and the τ² reference-agreement replay (A7.13), `llm_only_xsmall`, the optional `gpt-6.1-sol` rows (A7.12 (i)), E1
  for JevAgentRL.
- **Budget (planning, not caps).** API ≈ $550–625 for Stage 2 in all (A7.7's ≈ $430–445 + A7.10 ≈ $31 + A7.11 $9–43 +
  A7.12 ≤ $25 + A7.13 ≈ $20 − $160 + A7.14 ≤ $22 + A7.15 + $150–165, with 20 % contingency on the additions); the
  optional `gpt-6.1-sol` rows ≈ $70–80 on top, only if the budget allows. GPU: ≤ 48 L4 hours for training (≈ $63) and
  ≈ 24 L4 hours for serving and evaluation (≈ $32).
- Cost of A7.10–A7.17 as text: none; the model verification of A7.12 (a) cost < $0.001.

---

## A7.18–A7.19 — A third System-1 family and an exploratory second post-training track (2026-10-03, about 02:30 UTC; before any Stage-1 main-run result was read)

Written at the main session's request while the Stage-1 main run was in E2c, with the same blindness as A7: no Stage-1
main-run output had been opened, read or summarised. Neither amendment touches the Stage-2 family (S2-1..S2-4), its
margins, its arms or its tests. *Numbering:* the round-9 proposal labelled A7.18 in `docs/REVIEW_ROUND9.md` (the
non-strict robustness pair) was not adopted (A7.10–A7.17 preamble); the main session reuses the number for this new text.

### A7.18 — StartLux-Decision, a third, open-weights System-1 family (secondary and descriptive)

- **(a) The question.** Does a decision model that leads the static leaderboards also lead in the loop? This is the
  paper's static-vs-in-harness thesis, asked of a model released after the Stage-1 registration whose vendor reports it
  ahead of Jev on static benchmarks. Every A7.18 row is secondary or descriptive; none enters, replaces or reweights a
  Stage-2 family test.
- **(b) The model, as read on 2026-10-03.** StartLux-Decision (StartLux Labs; release 2026-09-30 per the main session;
  repository `github.com/StartLuxLabs/startlux-decision`, HEAD `a9101e9` of 2026-10-01 04:18 UTC, read 2026-10-03 at
  about 02:15 UTC): dense 0.8B, 2B, 4B, 9B and 27B and a 35B-A3B mixture of experts; Hugging Face organisation
  `startlux-models`, collection StartLux-Decision (original weights 1.8 / 4.6 / 9.3 / 19.4 / 55.6 GB for the dense
  sizes; GGUF BF16, Q8_0 and Q4_K_M per dense size). Code Apache-2.0; weights CC BY-NC 4.0 (research and other
  non-commercial use with attribution), entered in the A7.16 (g) terms matrix. The server (`python -m
  startlux_decision.server --model <local dir> --port 8090`) speaks the TypeSafe `/v1/systemone` format and returns a
  probability for every option, so the harness's Jev wire backend (`{type: jev, base_url: ..., model: ...}`) serves it
  unchanged. It answers choice lists of up to 26 options in one pass and wider lists with its own group knockout
  (`model.py::_wide`), and it serves one request at a time per GPU. The fast linear-attention kernels
  (`flash-linear-attention` ≥ 0.5.2, `causal-conv1d` ≥ 1.7; torch ≥ 2.8, transformers ≥ 5.8) are required:
  `python -m startlux_decision.check <dir>` must print "fast kernels: active", and the server refuses to start without
  them.
- **(c) Vendor numbers are cited as reported, never reproduced, never tested.** README: Decision Index 0.2.1 63.88 for
  the 27B against 57.91 for Jev 1.13 (the 27B ahead on 31 of 38 benchmarks; the 4B at 52.75); BFCL 96.7 and When2Call
  85.7 for the 27B (4B: 94.7 and 80.3; Jev 1.13: 94.3 and 74.6); latency for three questions in one pass on one H200
  26.0 ms (4B) and 102.3 ms (27B). The README states that the training data include the public train splits of 14 of
  the 38 Decision Index benchmarks (marked †; BFCL and When2Call are not marked), with their test items filtered out. We
  run no Decision Index benchmark. JevGym is our generator and its Stage-2 seeds were never released, so no JevGym item
  can be in StartLux's training data; E1's BFCL / When2Call items are public, so the E1 rows carry the note that overlap
  with a vendor's undisclosed data cannot be excluded.
- **(d) Pinning.** Before the first StartLux request the serving host records, per size, the Hugging Face repository
  revision (commit sha) of the weights, the inference package commit (`a9101e9` unless a later one is needed to start,
  stated) and the kernel versions (`results/stage2/startlux_manifest.json`, written by `scripts/aws/serve_startlux.sh`).
  A different revision before the A7.18 rows end is a deviation. Temperature, prompts and readout are the vendor's
  defaults; nothing is tuned on our data.
- **(e) Rows** (System-2 Flash; a new System-1 cache `cache/s1_stage2_startlux.sqlite`; `JEV_EVAL_RUNS=1` on the fresh
  seeds). The 4B is the primary StartLux row and the 27B the largest; 0.8B, 2B and 9B are a size ladder on the corpus
  replay only, run if the GPU budget allows after every other A7.18 row.
  1. **E1 static** (descriptive): `configs/decision/bfcl_when2call_startlux.yaml`, the E1 config with backends
     `startlux_4b` and `startlux_27b` on the C8 population (`exclude_ids_file` as registered, `batch_by_state: true`),
     compared item by item with the Stage-1 `jev` and Intern-Decision E1 rows (accuracy, AUROC, ECE).
  2. **Common-state corpus** (descriptive): A7.10's corpus C, frozen before any StartLux answer, replayed to the 4B and
     27B (and the ladder) by the A7.10 (c) path (`scripts/stage2_corpus.py replay --name sl4b | sl27b | sl0.8b | sl2b |
     sl9b`; the server's own knockout beyond 26 options), scored as acc_X(t) and shown beside Jev, Intern-Decision-4B
     and JevAgentRL, by stratum. StartLux trajectories never enter C, and S2-3 is unchanged.
  3. **Live replace** (descriptive): `sl4b_ts_replace_f` and `sl27b_ts_replace_f` (tool_select replace, Flash, seeds
     300000–300199, n = 200 each, three passes; `configs/experiments/stage2_startlux_states.yaml`), paired task by task
     with `jev_ts_replace_f` and `intern_ts_replace_f` of A7.10 (c): live success, own-state accuracy, trajectory length.
  4. **Calibration** (4B only, the size with a profile row): `sl4b_shadow_cal` (Flash; shadow at tool_select,
     tool_params and stop, as Intern's calibration arm) on calibration seeds 5000–5149 (n = 150;
     `configs/experiments/jevgym_shadow_cal_startlux.yaml`), fitted per decision point with the registered command (no
     accepted level → `always_defer: true`). The fits fill the profile's `backends.startlux_4b` block as A7.1 (f) does for
     JevAgentRL (cascade τ only; guard / trigger τ and assist k carried over unchanged).
  5. **S2-4S** (secondary test): JevAgentHarness[profile, StartLux-4B] (`jevagent_best_sl4b` in
     `configs/experiments/stage2_jevgym.yaml`, `profile_backend: startlux_4b`, n = 500, three passes, under the A7.16 (a)
     scheduler) vs `llm_only` of the same run: NI on success (m = 3 pt, `bb_ordered`) ∧ cost (paired t-test). The cost
     includes a provisioned-GPU tariff q_S per episode, measured and frozen exactly as A7.13 (iii) measures q for
     JevAgentRL (its own development workload and serving window, `results/stage2/tariff_sl4b.json`; the served model's
     per-token price is an exact 0 entry). Latency is descriptive (it depends on the serving host and queueing).
     *Adaptation:* the request named "A7.13 (vii)"; the tariff procedure is A7.13 (iii).
- **(f) Where S2-4S sits.** S2-4S joins the secondary table, which then has six rows (S2-1T, S2-1L, S2-L3a, S2-L3b,
  S2-1X, S2-4S; Holm within it at α = 0.05, floor 0.05/6 ≈ 0.0083). This lowers the other secondary rows' Holm levels;
  it is decided now, before any Stage-1 or Stage-2 outcome exists. Not tested (p = 1, stated): the profile's Flash
  section is skipped; the StartLux server cannot start on the GPU host (A7.18 (g)); the fitted profile delegates
  nothing with StartLux-4B (every cascade always defers and no other System-1 group is active); no frozen tariff or a
  mismatched workload; unpriced. S2-4S does not depend on the JevAgentRL freeze.
- **(g) GPU host and "not run".** The 4B (9.3 GB) runs on the Stage-2 L4 host. The 27B at original precision (55.6 GB)
  needs one GPU with ≥ 64 GB. If none is available within the GPU budget, its rows run from the vendor's Q8_0 GGUF
  (28.6 GB; the vendor reports the same decision as the original weights on 100.0 % of its 231 JevBench public items)
  through `startlux_decision.gguf_server` on a GPU with ≥ 40 GB, labelled "27B (Q8_0)"; otherwise the 27B rows are
  *not run*. Q4_K_M is never substituted. The choice is made by availability before the first 27B request, never by an
  outcome. If the server cannot start on the host (kernels missing, out of memory, a format mismatch found by the
  smoke test), the affected rows are *not run*, stated in the paper with the reason; no other model replaces them.
- **(h) Reading (descriptive, fixed now).** For every System-1 row with all three measures (Jev, Intern-Decision-4B,
  StartLux-4B, StartLux-27B, and JevAgentRL if frozen) the paper reports side by side: static accuracy (E1, C8
  population, item-weighted, Newcombe intervals for paired differences), common-state accuracy (C, task bootstrap) and
  in-the-loop live success and own-state accuracy (n = 200, paired Newcombe intervals), plus the profile row's cost and
  success where it exists. "Leads statically" means a higher E1 accuracy with a paired interval that excludes 0; "leads
  in the loop" means the same for live success or C accuracy. The thesis is supported by these rows only if a static
  lead is absent or reversed in the loop with interval evidence; a model that leads on all three measures is reported
  as doing so. No test and no Holm adjustment; the rank orders on the three measures are printed with Kendall's τ as a
  description.
- **(i) Budget (planning).** API (Flash) ≈ $40–45: live replace 2 × 200 × ≈ $0.03–0.04 ≈ $12–16; calibration 150 ×
  ≈ $0.045 ≈ $7; S2-4S 500 × ≈ $0.035 ≈ $17; the tariff workload ≈ $3; E1 and the replay $0; ≈ $50–55 with the 20 %
  contingency. GPU: ≈ 5 L4 hours for the 4B rows and the ladder (≈ $7) and ≈ 4 hours on one ≥ 48 GB or 80 GB GPU for
  the 27B rows (≈ $10–30 by instance), inside the A7.10 (h) serving reservation where it fits, else reported.
- **(j) Code** (before any A7.18 run): the StartLux backends and the configs of (e); `scripts/aws/serve_startlux.sh`
  (installs the pinned package and kernels in its own environment, checks the kernels, starts the 4B and 27B servers and
  writes the manifest of (d)); S2-4S in `scripts/stage2_analysis.py` with its tariff; the effective specification.

### A7.19 — JevAgentRL-v2 on Miles: an exploratory second post-training track

- **(a) Status.** Exploratory only. It never replaces the frozen JevAgentRL of A7.10: S2-3, S2-4 and every JevAgentRL
  sentence of the paper refer to A7.10's checkpoint. No choice between the two models is made on evaluation results,
  and v2 is never reported under the JevAgentRL name.
- **(b) Framework.** Miles (`github.com/radixark/miles`, Apache-2.0; HEAD `49576fd` of 2026-10-02 23:09 UTC, read
  2026-10-03): SGLang rollout with Megatron-LM training; SFT, on-policy distillation, GRPO, GSPO, PPO and REINFORCE++.
  Its supported hardware is A100 / H100 / H200 and newer (the L4 is not a target), so v2 trains on a separate GPU host.
- **(c) Barrier.** The A7.10 (a), (b) barrier applies unchanged, with its own operator session and its own manifest
  (`results/gpu_round6/TRAINING_MANIFEST_V2.md`: the input whitelist of A7.10 (a), the recipe, ranges, budgets, stage
  order, checkpoint-selection order and deterministic fallbacks, and the sha256 of every input), committed before the
  first training step. The v2 operator sees no Stage-1 or Stage-2 evaluation output, no profile-selection output and
  none of the A7.18 rows before the v2 freeze; nothing of v2 reaches A7.10's development before JevAgentRL's freeze.
- **(d) Recipe space.** Base: StartLux-Decision-4B if it loads and trains in Miles (checked on development data only),
  else Qwen3-4B; the base is recorded in the manifest before training. On-policy distillation from a local teacher
  served by SGLang (StartLux-Decision-27B; the vendor server if SGLang cannot serve its readout) and / or Jev's option
  probabilities; then GRPO in the harness on training seeds. The A7.10 (f) GRPO validity gate and the A7.10 (g) reward
  wording apply. A StartLux-based v2 inherits CC BY-NC 4.0: its weights are released, if at all, only under those terms
  (A7.16 (g)).
- **(e) Freeze.** The v2 freeze record (as A7.10 (b)) must exist by 2026-10-07 12:00 UTC; without it v2 has no
  evaluation row, and the paper describes the track in the appendix as unfinished.
- **(f) Rows, if frozen (descriptive, outside every Holm table).** As A7.18 (e) 2, 3 and 5: the corpus replay of C
  (`--name v2`), a live replace arm `v2_ts_replace_f` on 300000–300199 (n = 200), and the profile row
  `jevagent_best_v2` vs `llm_only` on 300000–300499 (n = 500), with its own calibration on 5000–5149 and its own tariff
  measured as A7.13 (iii). No test; reported beside JevAgentRL with the label "exploratory (A7.19)".
- **(g) Budget cap.** ≈ $400 GPU (≈ 10–12 node-hours of one 8 × A100 node on demand, fewer on H100) and ≈ $100 API (Jev
  teacher calls and the Flash cost of the rows of (f), ≈ $35). The cap is a ceiling: the track stops at it, frozen or
  not.

### A7.18–A7.19: tables as amended

- **The Stage-2 family** is unchanged (S2-1..S2-4).
- **The secondary table** (Holm within, α = 0.05, floor 0.05/6): S2-1T, S2-1L, S2-L3a, S2-L3b, S2-1X, S2-4S.
- **Descriptive, added:** the StartLux E1, corpus-replay, live and calibration rows (A7.18 (e) 1–4), the size ladder,
  the static-vs-loop table of A7.18 (h), and the A7.19 rows if v2 is frozen.
- **Budget, added (planning):** A7.18 ≈ $50–55 API with contingency and ≈ $17–37 GPU; A7.19 ≤ $400 GPU and ≤ $100 API.
- Cost of A7.18–A7.19 as text: none (two public repositories read, no model called).

### Deviations 2026-10-06 ~21:55 UTC: "E7e: itemized final answer" (exploratory, descriptive; registered before any E7e episode)

Exploratory row, outside both registered stages and every Holm table; no test; nothing here revises a registered or
reported result.

(a) Motivation (read after the E7c Jev row's outcomes, so E7e is a follow-up, not a confirmation). In the E7c Jev row
(100 tasks, seeds 310400-310499; summary_e7c.md) every failure in both arms was a wrong final total; no state-changing
call was wrong. Grouped by the form of the final answer, success was the same in both arms: answers that list the
customers before the total succeeded in 60.9 % (llm_only, 69 answers) and 61.5 % (jev_refit, 26), bare-number answers in
0 of 31 and 3 of 74. The Jev arm answered with a bare number far more often (74 % vs 31 %). The task's last sentence
asked only for "the total amount".

(b) Change. A new suite option `answer_style` in `bench/suites/jevgym_hard.py` (default `total`: the E7-E7c wording,
byte-identical); `itemized` replaces only the instruction's last sentence with "When all are done, answer with one line
per customer giving the action you took and the order amount, then the total amount of the orders you processed, rounded
to 2 decimals." World, gold plan, total and success check (the total's string in the answer) are unchanged
(tests/test_e7e_answer_style.py). The frozen `jevgym.py` is untouched.

(c) Design. E7c's Jev row unchanged except the option, on E7c's OTHER 100 registered tasks: seeds 310500-310599 (never run;
12 customers, tier rule, 60 near distractors, strict), arms `llm_only` and `jev_refit` (tool_select cascade tau 0.91,
tool_params cascade tau 0.87, reflection trigger tau 0.5: E7c's fitted thresholds, unchanged), task-major order (seed
732), 16 workers; Flash (gemini-3.5-flash, reasoning effort minimal, temperature 0) on the direct Gemini API; Jev via
OpenRouter with the author's Jev-only key (`OPENROUTER_API_KEY_JEV`, read only by `decision/jev.py`). Configs
`configs/explore/e7_hard/{smoke,eval}_e7e.yaml`; driver `scripts/explore_e7_hard.py --exp e7e` (keys `e7e/...`,
`paper/generated/e7e_hard.tex`).

(d) Gate and cap. Smoke on development seeds 7272-7273 (both arms); the registered projection (x 1.2) must fit the cap of
$60 for E7e (separate from E7c's), else stop after the smoke. A watchdog stops the run at the cap.

(e) Readout: as E7c (success, Delta with the paired task bootstrap and Newcombe intervals, wins / losses, cost per 1k
episodes and saving, coverage, accuracy incl. cancel-vs-ship, reflection firing, wall time, latency breakdown), plus the
share of itemized vs bare-number final answers per arm and success by answer form. Dollars compare within this row only.
Reported as exploratory beside the E7c Jev row, never instead of it.

### Deviations 2026-10-06 ~22:45 UTC: "E7e: answer-form readout" (exploratory, descriptive; analysis only, after both rows were read)

Implements E7e's registered readout (e) "share of itemized vs bare-number final answers per arm and success by answer
form" in `scripts/explore_e7_hard.py` (`answer_form`, `answer_forms`; tests in `tests/test_explore_e7.py`) for the E7c
Jev row and E7e (keys `e7c/<arm>/form/...`, `e7e/<arm>/form/...`; summary_e7c.md / summary_e7e.md, section "Final-answer
form"). Three forms are separated: the amount alone (`number`), prose with one or two amounts, e.g. only the total
(`sentence`), and per-customer lines plus the total (`itemized`, three or more amounts; every answer on both rows holds
either 1 or at least 13 amounts). Correction of E7e's motivation (a): its "answers that list the customers before the
total" pooled `itemized` with `sentence`. Separated, the E7c Jev row reads: llm_only 51 itemized (82.4 % success), 18
total-only sentences (0 %), 31 bare numbers (0 %); jev_refit 26 itemized (61.5 %), 74 bare numbers (4.1 %). So the Jev
arm's itemized answers did not succeed as often as llm_only's (26 answers; descriptive). E7e: every answer in both arms
itemized (success 87.0 % llm_only, 89.0 % jev_refit). E7e's design, configs and results are unchanged; nothing else is
revised.

### Deviations 2026-10-07 ~04:45 UTC: "E7f: AgentDecision-4B (tier) live" (exploratory, descriptive; registered before any E7f episode)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E7f)". It has no test,
revises no registered or reported result, selects nothing, changes no profile and never replaces the frozen
checkpoint. Nothing has run on seeds 310600–310799 or on the smoke seeds 7274–7275. This was checked in `configs/`,
`runs/`, the 87 run manifests in the S3 bucket and this registration.

(a) *Why.* The author asked, relayed by the coordinating session, to fix E7b's 0.5 %. **E7f was designed after E7–E7e
and E7d had been read.** First, an offline diagnosis ($0; `results/explore/e7f/DIAGNOSIS.md`, `scripts/explore_e7_hard.py
diagnose`) went through the frozen 4B arm's logs of E7 and E7b. It found no pipeline bug. The arm ran at the
registered thresholds (0 of 2,881 tool choices and 0 of 4,751 argument decisions disagree with them). It was served
from the sha256-verified frozen checkpoint (every episode and call names `jevagentrl-4b` at :8081). At every
cancel-or-ship state, both writes were offered, and the state held the policy sentence, the customer's tier and the
order amount. The model itself fails on the rule: it cancels about half of the basic and silver orders that should
be shipped, and it finishes long tasks early. It made the tool choice at every failing step. E7d's continued
checkpoint addresses these decisions offline but was never run live. E7f runs it live. E7–E7e stand as reported.

(b) *Model and serving.* AgentDecision-4B (tier) is E7d's continued adapter on Qwen3-4B
(`s3://…/checkpoints/e7d_tier/`; `adapter_model.safetensors` sha256 `6825d2b79d4173fa…`, `adapter_config.json`
`7322fc6a5ed9fa8d…`, as in `results/explore/e7_hard/box_e7d/tier_sha256.txt`). It is served with E7d's refitted
temperatures (`results/explore/e7_hard/box_e7d/cal/tier_calibration.json`, sha256 `8d7e0b780e75c692…`). The server is
`training/serve_jev.py`, as in E7d's replay, with no extra arguments, under the name `agentdecision-4b-tier` on :8081.
Adapter and temperatures are sha256-checked before serving. No System-1 cache.

(c) *Thresholds.* E7d's fit is reused unchanged and nothing is refitted. It is the registered calibration-arm
estimator (`threshold_theory --calibration-arm`, W = $1, λ = 0, V fallback W; A6.5's projection rule, 740 / 1,200
paired decisions) on calibration seeds 5300–5319. Those are the first 20 of E7c's calibration tasks, cut by E7d's
GPU-cap entry. Result: **tool choice τ̂ 0.9999322567940803, arguments τ̂ 0.9929112792015076**
(`results/explore/e7_hard/box_e7d/cal/tauhat_tier_*.json`). The reflection trigger keeps the profile's fixed τ = 0.5
and asks the same model. The calibration states carry E7c's wording of the task's last sentence (the total only),
while E7f asks for the itemized answer. E7e had the same difference with E7c's thresholds.

(d) *Setting.* E7e's: multi tasks only, 12 customers, `tier` rule, `answer_style: itemized`, 60 near distractors, 20
noise lines, strict, `max_steps` 45 (`configs/explore/e7_hard/{eval,smoke}_e7f.yaml`).

(e) *Tasks.* **Fresh seeds 310600–310799 are reserved for E7f.** The runner's guard now covers 310000–310799
(`bench/registry.py` `EXPLORE_EVAL_RANGES`, tested), and the training guards already exclude 310000–310999. **n = 100
(seeds 310600–310699), decided now from cost.** In E7e (the same setting and route), an `llm_only` episode cost $0.265
as logged. The 4B arm is estimated at about $0.18 (about 25 LLM calls at E7e's about $0.007 per call). 200 tasks would
therefore cost about $90, far past the cap, and 100 tasks about $45. Seeds 310700–310799 stay unused. Smoke:
development seeds 7274–7275 (never a result).

(f) *Arms*, in one pool, task-major (seed 732), 12 workers. `llm_only`. `rl_tier`: the Stage-2 profile's decision points
(tool-choice and argument cascades, reflection trigger) on AgentDecision-4B (tier) at the thresholds of (c), and
otherwise equal to the frozen profile's 4B arm (`tests/test_explore_e7.py`). **The frozen 4B with refitted thresholds
is not run.** Its registered refit (E7c, all 100 calibration tasks) always defers at tool choice. The arm would cost
about as much as `llm_only` (E7c's direct-API smoke: $0.275 vs $0.265 per episode) and would show nothing E7c's fit has
not shown.

(g) *Route.* Flash on the direct Gemini API (`gemini-3.5-flash`, temperature 0, minimal reasoning) in both arms. The
key comes from Parameter Store into the stage's process only and is never printed. Dollars compare within this row
only, as in E7e. If an episode of the smoke errors (for example HTTP 503 or overload), the gate stops the row and the
stop is reported.

(h) *Execution.* The E7 host `<instance-id>` (g5.8xlarge, one A10G; started again and stopped when done), with
`scripts/aws/e7_hard.sh` and `EXP=e7f` on the committed branch bundle. Stages: `rearm` (power-off guard 3 hours after
the start, 30-minute idle stop), `setup` (the adapter and temperatures sha256-checked, fetched from S3 if absent),
`serve`, `smoke`, `eval` and `final` (sync to `s3://…/e7_hard/e7f/`, power-off). `eval` sets `JEV_EVAL_RUNS=1`; passes
2–3 re-run only episodes that ended in an infrastructure error, after a route probe.

(i) *Caps and gate.* **$45 of Gemini spend for E7f in total, smoke included** (the requested "about $40", with the
estimate of (e)). A watchdog stops the runner at $45. The gate after the smoke takes the smoke's mean $ per episode
per arm, summed over the arms, × 100 × 1.05, plus the spend so far. If that fits $45, the evaluation runs all 100
tasks; else the first 80 if they fit (same check); else the row stops after the smoke. The factor 1.05 is used
because E7e's smoke predicted its evaluation to within 1 % ($0.451 projected vs $0.448 per task). GPU: at most 6
GPU-hours as requested, with the guard at 3 hours. `scripts/stage2_spec.py --gate` has no entry for the E7 family
(none of E7–E7e used it), so it does not apply here.

(j) *Readout* (`scripts/explore_e7_hard.py --exp e7f summarize` → `results/explore/e7f/summary.{md,json}`, keys
`e7f/...` in `paper/generated/e7f_hard.tex`). As E7e: success; Δ against `llm_only` with the paired task bootstrap
(B = 10,000, seed 0) and the Newcombe interval; wins and losses; cost per 1,000 episodes and saving. Costs are API as
logged; the 4B arm also carries the GPU tariff q = $2.448/h × the evaluation window / completed `rl_tier` episodes,
with S2-4's q and q = 0 as sensitivity rows. Also: coverage per decision point; accuracy on all and on accepted
decisions; the cancel-vs-ship split (coverage, accuracy on accepted ones, accepted-wrong count); reflection firing;
mean and median wall time with the latency breakdown; failure types; answer form. Added for `rl_tier`: the diagnosis
view of (a), with the cancel rate by tier and amount band and the decision behind each failure. Reported whatever the
outcome.

### Deviations 2026-10-07 ~05:40 UTC: "E11: the training recipe on ALFWorld" (exploratory, descriptive; registered before any E11 episode)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E11)". It has no test,
revises no registered or reported result, selects nothing and changes no profile. The model it trains is a NEW
checkpoint (records and serving name `e11-alfworld-4b`); it is not AgentDecision-4B, which stays frozen and untouched.
Nothing has run for E11. In `configs/` and `runs/` of this checkout, the only ALFWorld train games ever run are E6a's 48
calibration games (E6a / E6c calibration and smoke); valid_seen ran only in E6a / E6c.

(a) *Why.* The author asked on 2026-10-07 to show that the training recipe of DecisionAgentRL (Sec. 5: train a small
decision model on the states the agent itself visits) carries over from DecisionGym to a real benchmark. E11 does this
at ALFWorld's action decision point ("in environments without tools, the next action", Sec. 2). E11 was designed after
E6a and E6c had been read. In E6a, Jev's action cascade kept success (+0.0 points) at 14.0 % lower cost, Jev alone
(REPLACE) lost 64.3 points, and Jev's answers matched the single gold action at 17.7 % of its states.

(b) *Data: ALFWorld TRAIN games only.* Release archives as recorded in `configs/explore/e6a_alfworld/train_games.json`
(URLs and sha256; checked before use). Pool: the train games that pass ALFWorld's own filters (no `movable` / `Sliced`,
the six task types, `solvable`), minus E6a's 48 calibration games. Order: sha256("jev-explore-E11-alfworld-harvest-
2026-10-07" + newline + "<task dir>/<trial>"). Per task type the first 100 games are the harvest games (600). The first
60 per type of those (360) also get a Flash episode. In this order, every 10th game per type (positions 10, 20, ..., 100)
is a development game (60 games, 36 of them with a Flash episode). Its rows are never trained on; they fit the
temperatures and give the offline readout. The list is written by `scripts/explore_e11_alfworld.py select` to
`configs/explore/e11_alfworld/harvest_games.json`. **valid_seen and valid_unseen are never used for harvesting,
training, temperatures or thresholds.** The build refuses any row whose task id is not `alfworld/train/...`, is one of
the 48 calibration games or names a valid_seen trial.

(c) *Harvest: the states the agent visits.* (1) *The agent's own trajectories.* LLM-only on the 360 Flash games,
one episode each, with E6a's settings: the reference loop, Flash on the direct Gemini API (`gemini-3.5-flash`,
temperature 0, minimal reasoning), `max_steps` 50, `history_window` 12, oracle `auto`, `help` / `inventory` excluded
(`configs/explore/e11_alfworld/harvest_flash.yaml`). (2) *Scripted error injection* ($0), as in the paper's recipe. The
same loop runs with a scripted executor in place of the LLM (`ExpertNoiseLLM` in the script). At each state it takes
the gold next action of (d); if that action is not admissible, it takes the hand-coded expert's first planned action
when admissible. With probability ε = 0.2 per step it makes a scripted mistake instead: a near miss w.p. 0.5 (another
admissible action with the gold's verb, e.g. another receptacle or object), a uniformly random admissible action w.p.
0.3, or a no-op w.p. 0.2 (repeat the previous action, or `look` / `examine`). A kind without candidates falls back to a
random action. At a state without a gold action it acts at random. One episode per harvest game (600), seed 0.

(d) *Labels.* The environment's gold next action at every logged state: the planner walkthrough while the agent is on
it, else ALFWorld's hand-coded expert, which re-plans from the current state (`bench/suites/alfworld.py`, oracle
`auto`). Many states admit several good actions, so this single action is a lower bound of correct. Rows whose gold is
missing or not admissible are dropped. **Target: the gold action alone (one-hot), no teacher distribution.** Three
reasons: E3 found that gold-only labels matched the mixed labels (App. app:e3); Jev matched ALFWorld's gold action at
only 17.7 % of states in E6a; and Jev is not called from AWS (the organisation's OpenRouter route is exhausted). Labels
never reach the model at test time (`Question.meta` is never sent on the wire).

(e) *Rows.* One row per logged action decision: the state as the harness renders it for System-1, the question "Which
action should the agent take next?", and the admissible actions as options. With more than 52 options, the row keeps
the gold plus a seeded sample of 51 others (`training/data.py` `cap_options`). Train rows are deduplicated on (state,
option set), keeping the first. If more than 10,000 train rows remain, a seeded (seed 0) uniform sample of 10,000 is
kept. Reported: rows by source, the share of rows after an off-plan step (after the episode's first executed action that
differed from that step's gold), task types and option counts.

(f) *Model and training: SFT only.* Base Qwen3-4B (HF snapshot `1cfa9a72…`, as AgentDecision-4B) with a fresh LoRA (r 32,
α 64, all linear layers). The SFT v1 recipe: soft cross-entropy + 0.5 Brier (Eq. 1, μ = 0.5), random option
permutation, lr 1e-4, 2 epochs, 64 rows per optimizer step, `max_len` 4096, seed 0. Memory setting `nogc_3072` (SFT
v1's), and `gc_3072` then `gc_2048` only after an out-of-memory error (the same maths). **Why SFT only:** in the paper
the SFT model alone matched AgentDecision-4B on the common states and in live success, so SFT carried the gain. GRPO is
not registered here; it would need its own entry before any step. Temperatures: `training/calibrate.py` (golden-section
NLL fit per kind × option-count bucket, minimum bucket 200, plus a global one) on the development rows; served with
`--calibration`. Adapter and temperatures are sha256-recorded and go to S3 (`checkpoints/e11_alfworld/`).

(g) *Threshold.* A calibration shadow arm runs on E6a's 48 calibration train games (`configs/explore/e11_alfworld/
cal.yaml`): Flash acts at every step, and the served model is asked the same action question and logged. τ̂ is then
fitted with the registered calibration-arm estimator at the action decision point, exactly as in E6a
(`threshold_theory.calibration_arm_run` through `explore_e6a_alfworld.fit_tau`: W = $1, λ = 0, V fallback W, dC from the
ledger, A6.5's projection rule; the self-hosted model costs 0 in the ledger). `scripts/explore_e11_alfworld.py fit`
writes `configs/explore/e11_alfworld/tauhat_action.json`, and `set-tau` writes τ̂ into `eval.yaml` and `smoke.yaml` on
the box before the smoke. That is the only change, checked by the script, and the configs' sha256 are recorded. Both
are committed afterwards. If the fit accepts no confidence level, the cascade always defers, i.e. it is the LLM-only
harness (A6.11 (c) 4): that arm is not run and its cell reads "always defers".

(h) *Evaluation.* The same 140 valid_seen games as E6a (`configs/explore/e6a_alfworld/eval_games.json`, list sha256
`5a914a32…`), in E6a's order and settings (Flash on the direct Gemini API, temperature 0, minimal reasoning; `max_steps`
50, `history_window` 12; task-major, seed 732; 8 workers), in one runner invocation. Passes 2–3 re-run only episodes
that ended in an infrastructure error. Arms (`configs/explore/e11_alfworld/eval.yaml`):
- `llm_only`, **re-run in the same pool rather than reused from E6a**: the host, the day and the harness commit differ,
  and the wall-clock must be paired. E6a's row is quoted beside it.
- `e11_action_cascade_tauhat`: the trained model at the action point, CASCADE at τ̂.
- `e11_action_replace`: the trained model alone, with no LLM call (secondary, descriptive; E6a's REPLACE analogue).

The trained model is served by `training/serve_jev.py` on the same host (:8081, name `e11-alfworld-4b`). There is no
System-1 cache.

(i) *Readout* (`scripts/explore_e11_alfworld.py summarize` → `results/explore/e11_alfworld/summary.{md,json}`, keys
`e11/...` in `paper/generated/e11_alfworld.tex`). As E6a: paired complete case against `llm_only`; success, Δ (paired
task bootstrap, B = 10,000, seed 0; Newcombe), wins / losses. Cost per episode is Gemini as logged plus the GPU tariff
q = host $/h (on demand, us-east-1) × evaluation window / completed episodes of the two GPU arms (E7's convention after
A7.13 (iii)). q = 0 and S2-4's frozen q are sensitivity rows. Saving = 1 − C_arm / C_ref. Also reported: the share of
actions taken by the trained model (coverage); its accuracy on the actions it took, against the single gold action (a
lower bound); the deferred LLM's accuracy; mean / median wall-clock per episode; steps and LLM calls. Further: offline
accuracy and ECE on the development rows for the trained model and for base Qwen3-4B without the adapter; on the
calibration states, the trained model's and Flash's accuracy against the gold; success by task type; data and training
statistics; spend. Reported whatever the outcome.

(j) *Execution.* One new GPU host launched by this session with `scripts/aws/e11_alfworld.sh` (Name=jev-e11-alfworld;
image `<aws-resource-id>`, E3's clone). It is a g6e.2xlarge (one L40S 48 GB) if capacity exists, else a g5.8xlarge
or g5.4xlarge (one A10G 24 GB). Access is SSM only. The Gemini key comes from Parameter Store into the stage's process
only; it is never printed and never written to disk. The driver refuses every other project host (E3's, E5's,
Stage 2's, E7's `<instance-id>`). A power-off guard fires 10 hours after launch, with a 30-minute idle stop, and
the host is stopped as soon as its outputs are in `s3://…/e11_alfworld/`. Harvest data, rows and the adapter stay in S3;
the repository keeps the lists, configs, fits, summaries and keys. `scripts/stage2_spec.py --gate` has no entry for
exploratory rows and does not apply.

(k) *Caps and gate.* **Gemini: $60 for E11 in total** (Flash harvest, calibration, smoke, evaluation), enforced by a
watchdog on the logged spend, as in E6a. Expected about $15: the Flash harvest about $8 (360 games at E6a calibration's
$0.022 per train game), calibration about $1, evaluation about $5. Smoke: the three arms on the first 2 of the 48
calibration games (development, never reported). Gate: the smoke's mean $ per episode per arm, summed over the arms,
× 140 × 1.5, plus the spend so far, must fit $60 for all 140 games; else the first 100 if they fit; else the row stops
after the smoke. **GPU: at most 10 GPU-hours** (host hours of the one-GPU host). If the row cannot finish by
2026-10-10 18:00 UTC, it stops cleanly and what exists is reported.

### Deviations 2026-10-07 ~05:45 UTC: "E9: cost-accuracy frontier of multiple-choice decisions" (exploratory, descriptive; registered before any E9 call)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E9)" (the numbering of
the exploratory rows in these Deviations; unrelated to ROADMAP's E9). It has no test, revises no registered or
reported result, selects nothing and changes no profile or threshold. No E9 model call has been made.

(a) *Why.* The author asked on 2026-10-07 (relayed by the coordinating session) for one figure: do decision models sit
on the cost-accuracy frontier of an agent's multiple-choice decisions, against LLMs answering the same questions?
Before registering, the inventory `results/explore/e9_frontier/DATA.md` read the per-item records of E1 and the
calibration shadow logs (accuracy, logged cost, latency). The E1 accuracies are already in the paper
(`paper/generated/difficulty.tex`). **The main readout is therefore post hoc and descriptive.**

(b) *Main readout, no call: the L1 static items.* E1's 3,991 decisions (the C8 population: BFCL v4 tool choice 1,114,
BFCL arguments 1,949, When2Call 928; the development prefix excluded), every answer already logged in
`runs/e1_bfcl_when2call*/`. Decision models: Jev, Intern-Decision-4B, StartLux-Decision 0.8B / 2B / 4B / 9B / 27B
(Q8_0), AgentDecision-4B (the frozen checkpoint). LLM deciders on the same multiple-choice questions: Gemini 3.5 Flash
(`s2_self`, verbalised distribution) and Flash-Lite (verbalised distribution; the label variant beside it). Also
shown: Flash's own native function call on the same items (not a multiple-choice answer). Listed, not drawn: Gemma 4
(no verified price), the untrained base and SFT-only checkpoints, the embedding / BM25 / TF-IDF controls. Accuracy:
pooled over the 3,991 decisions (the records' `correct`), with each family beside it. Cost per decision: API models
the records' logged `cost_usd` (list price on measured tokens, reasoning tokens billed as output); open-weights models
the paper's GPU tariff, $2.24208 per GPU-hour (the Stage-2 host, A7.13 (iii)), times the logged serving time per
decision (request latency / questions in the request; an upper bound, as requests went four at a time). Sensitivity:
the provisioned per-token tariff of the registered windows (`results/stage2/tariff.json`, `tariff_sl4b.json`; the
StartLux sizes take StartLux-4B's, Intern-Decision its registered $0.010 / M), and zero. Latency: the median request
latency per decision, as logged on each model's own host. *Frontier*: the models that no other drawn model matches or
beats on both cost and accuracy (one strictly). Per LLM decider: the cheapest decision model at least as accurate,
with the cost ratio; Jev against each LLM decider: the accuracy gap with a paired item bootstrap (B = 2,000, seed 0).
Wilson 95 % intervals for single accuracies. If the LLMs dominate, the readout and the caption say so.

(c) *The check, new calls: the cheapest LLM tier.* No record holds GPT-6-Luna or GPT-6.1-Sol as a multiple-choice
decider, and the calibration logs sit at the ceiling (DATA.md). The check scores **E1's development prefix**: the 299
items of `configs/eval/e1_dev_prefix_ids.json` (A6.13: exposed in development rounds, excluded from C8; no seed of any
evaluation block is involved, and no C8 item is sent to a new model). Backends through `jevharness.bench.decision_eval`
with E1's settings (batch by state, the prompts of `decision/llm_generative.py`), config
`configs/explore/e9_frontier/e1_dev_check.yaml`: GPT-6-Luna (reasoning effort none, temperature 0) and GPT-6.1-Sol
(effort low, its lowest; no temperature sent), verbalised distribution, through the OpenAI Responses API; Gemini 3.5
Flash (`s2_self`) and Flash-Lite (distribution), minimal reasoning, temperature 0, on the direct Gemini API; Jev via
OpenRouter with the author's Jev-only key (`OPENROUTER_API_KEY_JEV`, read only by `decision/jev.py`), run locally. A
new answer cache (`cache/s1_e9.sqlite`), so every row is measured in this session. Four LLM deciders in all. No gold
field is ever sent (`Question.meta` stays off the wire). Every call goes through a Ledger. The check is reported in the
text beside the figure, never drawn into it (different items, no open model).

(d) *Cap and gate.* **$30 of API for E9 in total.** Smoke: the first 30 items of the same stratified order, every
backend. Go if the smoke's fresh cost × 299 / 30 × 1.3 plus the spend so far fits the cap; else stop after the smoke
and report it. Expected about $2.

(e) *Readout* (`scripts/explore_e9_frontier.py` → `results/explore/e9_frontier/summary.{md,json}`, keys `e9/...` in
`paper/generated/e9_frontier.tex`, figure `paper/figures/fig_frontier.tex`), whatever the outcome.

### Deviations 2026-10-07 05:56 UTC: "E7f: execution as run" (exploratory, descriptive; written after the run, before any paper text used it)

(a) *Timing label.* The registration above says "~04:45 UTC"; it was committed at 04:37:23 UTC (`60c16ef`), before
the first E7f episode (smoke 04:38 UTC, evaluation from 04:42:32 UTC). The commit time is authoritative.

(b) *As run.* The registered cost gate chose "the first 80": seeds 310600-310679, 12 workers, both arms, Flash on the
direct Gemini API, the retrained checkpoint (`tier_verify ok`) at E7d's fitted thresholds. Spend $40.53 (cap $45). The
coordinating session lost AWS access at about 05:20 UTC (expired login) while the run was finishing; after the author
re-authenticated, the box `<instance-id>` was found **stopped** (its own power-off after the S3 sync), and the
outputs were fetched from S3 unchanged (`runs/e7f_hard`, gitignored; host records in
`results/explore/e7_hard/box_e7f/`). Nothing was re-run.

(c) *Readout* (`scripts/explore_e7_hard.py --exp e7f summarize` -> `results/explore/e7f/summary.{md,json}`, keys
`e7f/...`), as registered, whatever the outcome: success 76.2 % against 77.5 % for LLM-only on the same 80 tasks
(-1.2 points, paired bootstrap [-12.5, +8.8]); cost 3.4 % lower with the GPU tariff (12.2 % API only); the model took
49.0 % of the rule's cancel-or-ship decisions with no wrong answer accepted (96.7 % accuracy on all of them); time per
episode 336.0 s against 58.4 s (System-1 calls queued on one GPU for 12 workers). Exploratory; no test.

### Deviations 2026-10-07 ~06:00 UTC: "E11: harvest labels" (exploratory; amends E11 (c)–(d) before any E11 harvest episode)

Registered before any E11 harvest, calibration or evaluation episode. The E11 host (`<instance-id>`, a g5.4xlarge:
us-east-1d had no g6e.2xlarge or g5.8xlarge capacity) has so far only installed software and data; it selects the
harvest list next, which runs no episode.

(a) *What was found.* Before the harvest, a local $0 code check ran the scripted executor of (c) 2 (ε = 0.2) on E6a's
48 calibration train games. No model and no fit were involved, and nothing from it is kept. With the registered oracle,
37 of 48 episodes reached the goal (mean 18.6 steps), only 2 of 8 `pick_two` games did, and 17.3 % of the logged gold
actions were `look`. `look` is not a label: ALFWorld's `AlfredExpert` wrapper returns it whenever the hand-coded expert
has no admissible next command, and no walkthrough or hand-coded expert command is `look`. The cause is a tracking bug
in ALFWorld 0.4.2's hand-coded expert. Its inventory tracker recognises `put`, but 0.4.2 places objects with
`move X to Y`. After any `move`, its inventory goes stale and its next actions are wrong or not admissible. In E6a's Flash
runs, 3.2–4.4 % of gold actions were this fallback. ALFWorld's PDDL planner was tried as the labeler instead and was no
better (37 / 48, 24.7 steps, 18.2 % fallback). With the tracker reading `move X to Y` as `put X to Y`, the same check
reached the goal in 45 / 48 (6 of 8 `pick_two`), 12.7 steps, 2.0 % fallback.

(b) *Amendment, harvest labels only.*
1. Both harvests run in the script's own processes: `explore_e11_alfworld.py harvest-flash`, which runs
   `harvest_flash.yaml` through the same runner function, and `harvest-scripted`. In these processes the hand-coded
   expert's tracker reads `move X to Y` as `put X to Y` (`patch_handcoded_move`). The oracle is otherwise unchanged:
   the walkthrough while the agent is on it, else the hand-coded expert.
2. The fallback `look` is no label. The scripted executor treats it as a state without a gold action and acts at random,
   as registered. The build drops every row from an episode's first fallback on, because the expert has lost track by
   then (`lost_from`; the counts are reported).
3. The calibration shadow arm, the smoke and the evaluation keep the unpatched oracle, run by
   `python -m jevharness.bench.runner`, as in E6a. Their gold actions only score decisions, so accuracy there stays
   comparable with E6a's and remains a lower bound.

Everything else in E11 is unchanged.

### Deviations 2026-10-07 ~06:05 UTC: "E7f extension" (exploratory, descriptive; registered before any episode of the extension)

Addendum to E7f. It is exploratory, has no test, sits outside every Holm table and revises no registered or reported
result. It was requested by the author after E7f's 80-task result had been read: the author wants each experiment
complete. The first part stands as reported ("E7f: execution as run"). Nothing in it is re-run or replaced. Nothing
has run on seeds 310680–310799: the first part stopped at 310679 by its gate, and no other config, run or S3 manifest
uses these seeds.

(a) *Tasks.* **The remaining registered E7f seeds 310680–310799 (n = 120)**, so that E7f covers its whole reserved
block, 310600–310799 (200 tasks). The runner's guard already covers them.

(b) *Unchanged.* Everything else is as in the first part: the two arms (`llm_only`, `rl_tier`); the checkpoint
(AgentDecision-4B (tier), adapter and temperatures sha256-checked, served as `agentdecision-4b-tier` on :8081 with no
extra arguments); the thresholds (tool choice 0.9999322567940803, arguments 0.9929112792015076, reflection trigger 0.5);
the setting (12 customers, tier rule, itemized answer, 60 near distractors, 20 noise lines, strict, `max_steps` 45);
Flash on the direct Gemini API; 12 workers; the task-major schedule (seed 732); and passes 2–3 for infrastructure errors
only. The config is `configs/explore/e7_hard/eval_e7fx.yaml`, which differs from `eval_e7f.yaml` only in name, seed0
and n (tested). The run is `runs/e7fx_hard`, on the same host `<instance-id>`
(`scripts/aws/e7_hard.sh`, `EXP=e7fx`; records to `s3://…/e7_hard/e7fx/` and `results/explore/e7_hard/box_e7fx/`).

(c) *Cap and gate.* **$70 more Gemini spend for the extension**, with a watchdog at $70. There is no smoke. The gate
takes the first part's logged mean $ per episode per arm (`llm_only` $0.264, `rl_tier` $0.232 API), sums it over the
arms, multiplies by 120 × 1.05 and adds the extension's own spend so far. If the result is at most $70 (expected
about $62), the evaluation runs all 120 tasks; else it stops before any episode. GPU: the extension's evaluation is
expected to take about 1.2 hours, which with the first part's 0.9 stays within the 6 GPU-hours. A power-off guard
fires 2 hours after the start, and the host powers off when the run ends.

(d) *Readout.* `scripts/explore_e7_hard.py --exp e7f summarize` pools both parts: 200 tasks, paired per task, and
`e7f/setting/n` becomes 200. The registered metrics are E7f's (j). The GPU tariff is the host price × the two
evaluation windows / the completed `rl_tier` episodes of both parts. The readout also reports each part's own paired
comparison: the first 80 tasks and the new 120, with keys `e7f/first/...` and `e7f/ext/...`, and a "By part" table in
`results/explore/e7f/summary.md`. The pooled 200 is the row's result, reported whatever the outcome.

### Deviations 2026-10-07 07:18 UTC: "E7f extension: execution as run" (exploratory, descriptive; written after the run, before any paper text used it)

(a) *As run.* Registered at 06:04:04 UTC (`5bec1db`), before any extension episode. On host `<instance-id>` the
gate projected from the first part's logged costs and chose "run all 120" (seeds 310680-310799); the evaluation ran
06:06:55-07:15:54 UTC with 12 workers, both arms, Flash on the direct Gemini API and the same checkpoint and thresholds.
No episode errored, so no second pass ran. Spend $59.14 (cap $70); E7f in all $99.67 (caps $45 + $70). The host synced
to `s3://…/e7_hard/e7fx/` and powered itself off; outputs were fetched unchanged (`runs/e7fx_hard`, gitignored; host
records in `results/explore/e7_hard/box_e7fx/`). Nothing was re-run.

(b) *Readout* (`scripts/explore_e7_hard.py --exp e7f summarize`, both parts pooled as registered; keys `e7f/...`,
`e7f/first/...`, `e7f/ext/...`), whatever the outcome. **Pooled, 200 tasks:** success 81.5 % against 83.5 % for
LLM-only on the same tasks (-2.0 points, paired bootstrap [-8.5, +4.5]; 21 wins, 25 losses); cost 4.3 % lower with the
GPU tariff (13.1 % API only); the model took 51.6 % of the rule's cancel-or-ship decisions with no wrong answer accepted
(96.3 % accuracy on all of them); time per episode 327.2 s against 74.2 s (System-1 calls queued on one GPU for 12
workers). By part: the first 80 as reported (76.2 vs 77.5 %); the new 120, 85.0 vs 87.5 % (-2.5 points), 4.9 % lower
cost. The descriptive paired non-inferiority test of `scripts/hard_ni.py` (3-point margin, the A6.21 rule) does not
establish non-inferiority on the 200 (p = 0.402). LLM-only solved 83.5 % on the 200, above the 60-80 % band the
development pilot aimed for (the extension's tasks were easier for it, 87.5 %); the setting was fixed before E7f and is
not changed. Exploratory; no test.

### Deviations 2026-10-07 ~13:50 UTC: "E12: more training on public tool-use data" (exploratory, descriptive; registered before any E12 training step or evaluation episode)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E12)". It has no test,
revises no registered or reported result, selects nothing and changes no profile or threshold. The model it trains is
a NEW checkpoint, **"AgentDecision-4B (mixed)"** (served as `agentdecision-4b-mixed`). AgentDecision-4B (the frozen
JevAgentRL-4B checkpoint of the records) stays frozen and untouched. No model has been trained for E12, and no model
has answered any E12 item. Only the training items were built and audited, locally, at $0 and without any model call
(see (b)–(d)). The quotas, the audit rule and the mix were fixed in `scripts/explore_e12_train.py` before that build
ran.

(a) *Why.* The author asked on 2026-10-07: "can more training improve the results?" In Figure 4a (E9)
AgentDecision-4B is dominated on fixed questions: 78.8 % on E1's 3,991 BFCL and When2Call decisions, and 54.0 % on
When2Call. Inside the agent it ranks first (99.2 % on the common states, 100 % live; Figure 4b). E12 was designed
after those results had been read. A look at E1's logged records showed the cause: on When2Call, AgentDecision-4B
never answers "call one of the available tools" (0 of 329 such items), while its untrained base gets 92.7 % of them.
Its SFT stage saw a When2Call-train replay, but that file holds only "ask" and "cannot answer" items. E12 asks:
**does adding public tool-use training data put the trained model on the fixed-question frontier while keeping its
agent-state accuracy?**

(b) *Items, in E1's exact format.* E1's converter is reused unchanged (`bench/suites/static.py`). Its per-row bodies
are now the functions `bfcl_row_items` and `when2call_item`, and E1's 3,991 items come out byte-identical (fingerprint
test). Each item is rendered for training exactly as `serve_jev` renders the wire question
(`grpo_online.question_to_example`), with a one-hot target on the gold option. Sources: public training sets only.
1. When2Call TRAIN (`train/when2call_train_sft.jsonl` at E1's pinned commit; CC-BY-4.0; 15,000 rows with no label
   column). Label rule: a reply that asks for something (a "?" or a request phrase such as "please provide") is
   `request_for_info`; else a refusal ("unable", "cannot", "sorry", ...) is `cannot_answer`; else the row is dropped
   (none was). These rows give When2Call-format "ask" and "cannot" items. A "cannot" row with at least one tool also
   gives a BFCL-format tool-choice item whose gold is "none" (no available tool fits the request).
2. ToolACE (`Team-ACE/ToolACE`, revision `6bda777c…`, `data.json` sha256 `ba12c083…`; Apache-2.0; public, not
   gated). Usable rows: English, the first assistant turn ONE call of a listed tool (5,036 of 11,300). They give
   When2Call-format "call" items, BFCL-format tool-choice items (gold = the called tool) and BFCL-format argument items
   (one per scalar argument; candidates copied from the request plus OTHER, exactly as E1 builds them).
   xLAM-60k is gated and is not used. BFCL itself and When2Call TEST are never used.

(c) *Leakage audit, before any split or sample.* Each source row (unit) is matched by its user request against three
targets: E1's 3,991 decisions, E1's 299-item development prefix, and the full public test pools (every BFCL v4
tool-category row and every When2Call test question, `data_v1.eval_texts`). Two kinds of match count: exact (an
identical token sequence) and near-duplicate (a shared 8-gram of lowercase word tokens; a shorter request matches only
an identical one). A unit with any hit is dropped with all its items. As built: 19,909 units after removing 127
duplicate source rows; 207 units with a hit (485 items) dropped. Of those, 175 hit E1 (0 exact, 175 by 8-gram), 14 hit
the development prefix (0 exact, 14 by 8-gram) and 207 the full pools (2 exact). Counts are in
`results/explore/e12/build_info.json` and the dropped unit ids in `audit_hits.json`.

(d) *Split and mix* (seed 0; fixed before the build). A unit is a public development unit iff sha256("jev-explore-E12-dev"
newline unit) mod 10 = 0; those units are never trained on. From the training units, new items per family (each
tool-choice or "call" item from a different ToolACE row, each "none" or "cannot" item from a different When2Call row):
When2Call-format call 1,000, ask 1,000, cannot 1,000; BFCL-format tool choice 1,500 (gold a tool) and 500 (gold
"none"); BFCL-format arguments 2,000 (888 with gold OTHER). In all **7,000 new items**. **Replay: 7,000 rows**, a seeded
sample of AgentDecision-4B's SFT v1 training rows (`data/train_v1/train.jsonl.gz`, sha256 `9c0450c2…`, the 15,694
JevGym rows; E3's D100 file is byte-identical). Every replay row's JevGym seed is re-checked against the training guards
(`bench/registry.training_exclusions`, `training/seed_guard.TRAIN_FORBIDDEN`). No evaluation seed enters any training
file. **New : replay = 1 : 1, 14,000 rows.** The public development split holds 1,000 items in the same family
proportions (150 / 150 / 150 / 200 / 100 / 250). The temperature set is a seeded sample of 1,000 v1 dev rows plus that
split. All files are sha256-recorded in `build_info.json`. They go to `s3://…/e12/data/` and are re-checked on the host
(`explore_e12_train.py verify`).

(e) *Training.* **Continue AgentDecision-4B's adapter, as E7d did, rather than train from base.** The question is
whether more training of the paper's model repairs its fixed-question weakness. A continuation keeps everything that
model learnt (its SFT and RL stages), changes only what the new data teach, and costs about an hour of GPU instead of
the 2+ hours of SFT v1 from base (E3). Recipe (E7d's continuation): `training/train_decision.py --init <frozen
adapter>` on base Qwen3-4B (snapshot `1cfa9a72…`); the frozen adapter and its files are sha256-checked against
`results/stage2/rl_eval_plan.json`. LoRA r 32 / α 64 as frozen; soft cross-entropy + 0.5 Brier; option permutation;
lr 5e-5; 1 epoch; 64 rows per optimizer step; max length 4,096; trainer seed 0; one run. Memory setting `nogc_3072`,
then `gc_3072`, then `gc_2048`, the next only after an out-of-memory error (same maths); capped at 180 minutes.
Temperatures: `training/calibrate.py --min-bucket 200` on the temperature set of (d). Served by `training/serve_jev.py`
with the adapter and temperatures and no extra arguments, as the frozen row (the plan's `serve_args` are empty).

(f) *Offline evaluation* (the E12 host, no API).
1. **E1's 3,991 decisions** (`configs/explore/e12_train/e1_e12.yaml`: E1's pools, C8 exclusions, batching and 4
   workers, a new answer cache). Accuracy pooled and per family (BFCL tool choice, BFCL arguments, When2Call), the
   When2Call accuracy by gold label, and the paired item bootstrap of E12 minus AgentDecision-4B's logged E1 answers
   (E9's records and `paired_gap`: B = 2,000, seed 0). Cost per decision by **E9's rule**: $2.24208 per GPU-hour times
   the logged serving time per decision.
   - *Serving-path check and same-host reference.* Before E12 is served, the frozen AgentDecision-4B is served on the same
   host and answers the same 3,991 decisions. Its answers must agree with its logged E1 answers on at least 99 % (E3 /
   E5's rule). The readout reports the result either way. If the check fails, E12 is also compared with the re-hosted
   answers. The re-hosted serving time gives a host adjustment: E12's cost times (frozen logged cost / frozen re-hosted
   cost), reported beside E9's rule.
2. **The common states C** (5,314 occurrences, `results/stage2/s23/`): replayed through `scripts/stage2_corpus.py`
   with a new cache. AgentDecision-4B's answers are its registered ones (`answers_rl.jsonl`). Accuracy as in Stage 2
   and the paired task bootstrap of the delta (`ablation_score.paired_boot`, B = 2,000, seed 0).
3. **Held-out JevGym dev rows** (`data/train_v1/dev`, `dev_recovery`, `dev_knockout`; never trained on) and **the
   public development split** per family: `training/eval_decisions.py` for both checkpoints on the E12 host.

(g) *Live evaluation.* Replace mode at the tool choice (`configs/explore/e12_train/live.yaml`), as the frozen row
`rl_ts_replace_f` and the E2 ladder. Same suite block; the same first 200 fresh Stage-2 tasks (seeds 300000–300199;
evaluation seeds, run with `JEV_EVAL_RUNS=1`, never trained on); Flash on the direct Gemini API (temperature 0,
minimal reasoning); 4 workers; a new System-1 cache. Passes 2–3 re-run infrastructure errors only. First a smoke on 2
JevGym development tasks (seeds 900–901; never reported), then the gate in (i). Reported: success (latest record per
task, errored tasks left out), own-state accuracy, steps, calls, API $ per episode and wall-clock. The frozen and Jev
rows of `runs/stage2_rl_states` (same tasks, run 2026-10-03) are quoted beside it, with paired Newcombe intervals.

(h) *Pre-registered outcomes, reported whatever they turn out to be.*
- (i) Fixed-question accuracy (E1, 3,991 decisions) against AgentDecision-4B: the delta and its interval, pooled and
  per family.
- (ii) Common-state accuracy (C): **non-inferior to AgentDecision-4B at a 1-point margin** iff the lower end of the
  delta's 95 % interval is at least -1 point; otherwise "not shown".
- (iii) Whether the new point is on E9's frontier (E9's drawn models plus this point; E9's cost rule; no other drawn
  point matches or beats it on both cost and accuracy), and whether the frontier changes. The host-adjusted cost is
  reported beside it.
- (iv) Live success on the 200 tasks, beside the frozen row.

(i) *Execution, caps and gate.* One new GPU host launched by this session with `scripts/aws/e12_train.sh`
(Name=jev-e12-train; image `<aws-resource-id>`, E3's clone). Types in order g6e.2xlarge, g6e.xlarge, g5.8xlarge,
g5.4xlarge, each tried in us-east-1d, then the other us-east-1 zones. Access is SSM only. The Gemini key comes from
Parameter Store into the live stages' process only, is never printed and never written to disk; no other key is
loaded. The driver refuses every host it did not create. Stages: `rearm setup data train temps offline e1_frozen
e1_e12 c_replay live_smoke live final`. Records go to `s3://…/e12/box/`, the adapter to `s3://…/checkpoints/e12_mixed/`
(no optimizer state). **GPU: at most 12 GPU-hours** (host hours): a power-off guard 690 minutes after launch, an idle
stop after 30 minutes, and the host stops itself when `final` has synced. **API: $30 of Gemini for E12 in total**
(live smoke and live), enforced by a watchdog on the logged spend. Gate: the smoke's mean $ per episode × 200 × 1.5,
plus the spend so far, must be at most $30, else no live episode runs. No OpenRouter or Jev call. Downloads: the two
public datasets above and E1's pinned public files. `scripts/stage2_spec.py --gate` has no entry for exploratory rows
and does not apply. If the row cannot finish by 2026-10-09 18:00 UTC, it stops cleanly and what exists is reported.

(j) *Readout.* `scripts/explore_e12_train.py score` writes `results/explore/e12/summary.{md,json}` and the keys
`e12/...` in `paper/generated/e12_train.tex`. E9's `summarize` gains the E12 point: E9's own frontier and keys stay as
registered, and the recomputed frontier with E12 and the E12 point get their own keys (`e9/front12/...`,
`e9/e12/...`). An "execution as run" entry follows the run.

### Deviations 2026-10-07 ~14:00 UTC: "E13: other agent LLMs on DecisionGym-Hard" (exploratory, descriptive; registered before any E13 episode)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E13)". It has no test,
revises no registered or reported result, selects nothing and changes no profile. Nothing has run on the evaluation
seeds 311000–311299, the calibration seeds 5400–5699 or the smoke seeds 7276–7281. This was checked in `configs/`,
`runs/` of this checkout, `results/` and this registration. Before this entry one route probe was sent: a single
one-turn request to `gpt-6-luna` with one tool and no task (89 tokens, under $0.001). It confirmed that the model
accepts reasoning effort `none` with temperature 0 and calls the tool, and it read this key's rate limit (2,000,000
tokens per minute). The $0 dry runs of the pipeline used simulated models on development seeds 0–49.

(a) *Why.* The author asked, relayed by the coordinating session, whether the delegation result on DecisionGym-Hard
holds when the agent LLM is not Gemini Flash. **E13 was designed after E7–E7f had been read**, in particular E7e (Flash
with the refitted Jev cascade: 89.0 vs 87.0 % success, 30.5 % cheaper). Every E13 row is reported whatever the
outcome. E7e is not re-run.

(b) *Setting.* E7e's, unchanged: multi tasks only, 12 customers, `tier` rule, `answer_style: itemized`, 60 near
distractors, 20 noise lines, strict, `max_steps` 45, task-major schedule (seed 732). The configs are generated by
`scripts/explore_e13_llms.py` (`config`); a test checks that their suite block equals
`configs/explore/e7_hard/eval_e7e.yaml`'s apart from seed0 and n.

(c) *Agent LLMs and routes.*
1. **GPT-6-Luna** (`gpt-6-luna`) through the OpenAI Responses API directly (`provider: openai_responses`; reasoning
   effort `none`, its lowest, since `minimal` is rejected; temperature 0; stateless, no `previous_response_id`), as
   Stage 2's Luna rows. Run locally in the development container. Jev runs via OpenRouter on the author's Jev-only key
   (`OPENROUTER_API_KEY_JEV`, read only by `decision/jev.py`; never on AWS, never printed).
2. **DeepSeek-V4-Flash** (`deepseek/deepseek-v4-flash`) through OpenRouter, pinned to StreamLake (fp8) with fallbacks
   off and reasoning off, as S2-1X's X-small.
3. *[Public copy: this optional third agent-LLM row (a model from a third provider through OpenRouter) was never run; its description is omitted.]*

Rows 2 and 3 are **run by a collaborator on their own keys; date recorded in an execution-as-run record.** They use one
OpenRouter inference key, which also pays for Jev, through the handoff package `handoff/experiments/e13/` (smoke, gate,
calibration, fit, evaluation and readout in one resumable job; the results are pushed to a new branch under
`results/handoff/e13/`).

*Prices* (`configs/prices/models.yaml`; OpenRouter's model and endpoint listings, read 2026-10-07 13:23 UTC). Luna:
$0.10 / $0.50, cache reads $0.01 (unchanged, and the same on OpenRouter). DeepSeek-V4-Flash on StreamLake: $0.042 /
$0.084, cache reads $0.0084. That price changed since S2-1X (2026-10-03: $0.028 / $0.056), so E13 charges a dated row
through a new `price_model` key of the llm block, which is never sent to the provider; S2-1X's row stays as charged. Each LLM's dollars are its own route's list prices.
The saving is paired within an LLM's row; across rows, costs compare as list prices only.

(d) *Tasks.* A fresh block of 100 evaluation tasks per LLM: Luna **311000–311099**, DeepSeek **311100–311199**, row 3
**311200–311299**. The runner refuses 311000–311299 without `JEV_EVAL_RUNS=1` (`bench/registry.py`
`EXPLORE_EVAL_RANGES`). Every training guard excludes 311000–311999 (`EXPLORE_TRAIN_EXCLUDED`, through
`training_exclusions()` and `training/seed_guard.TRAIN_FORBIDDEN`). Both are tested (`tests/test_explore_e13.py`).
Calibration seeds, from the reserved calibration range and never run before: Luna 5400–5499, DeepSeek 5500–5599,
row 3 5600–5699. Smoke seeds, development and never a result: Luna 7276–7277, DeepSeek 7278–7279, row 3 7280–7281.

(e) *Arms* (per LLM, one task-major pool). `llm_only`. `jev_refit`: E7e's `jev_refit` with that LLM's refitted
thresholds, that is Jev's tool-choice cascade and argument cascade at the τ̂ of (f), plus the profile's fixed reflection
trigger (τ = 0.5; not a cascade, not refitted). Each LLM's row therefore has the structure of E7e's Flash row. Jev is the
pinned OpenRouter endpoint (`{type: jev, provider: openrouter}`), with no System-1 cache.

(f) *Refit: the registered calibrate-then-deploy procedure, as E7c for Flash.* Calibration arm `jev_shadow_cal`: the
LLM acts alone, and Jev answers every tool-choice and argument question in shadow mode (logged, never executed). Fit:
the registered estimator per decision point (`threshold_theory --calibration-arm --task-value 1.0 --lam-per-hour 0
--v-fallback W`, with A6.5's sample-size rule), run by `explore_e7_hard.fit_calibration` on the LLM's calibration run.
Nothing is tuned beyond this rule. τ̂ = +∞ at tool choice is deployed as `always_defer` (System-1 not asked; A6.11 (c));
at the arguments as τ = +∞ (asked, never accepted), as E7c registered. A fit without a τ̂ stops that LLM's row before
its evaluation. The fit is mechanical. It is written to `cal_<llm>/fits.json` and into the generated evaluation config
before that LLM's first evaluation episode, and the readout checks this order from the status record. For Luna the
fitted values are also committed before its evaluation. The collaborator's rows run unattended in one job; their fitted
values are reported in the execution record.

(g) *Order, smoke and gate, per LLM* (`scripts/explore_e13_llms.py --llm <llm> run`; every stage is recorded in
`<out>/<llm>/status.json`, and a rerun skips finished stages and episodes).
1. *Smoke* on the 2 smoke seeds: `llm_only`, `jev_shadow_cal` and `jev_cascade_dev` (the cascade at E7c's Flash τ̂
   0.91 / 0.87 plus the trigger). The last arm exercises the delegated-tool path (the LLM's forced single-tool argument
   call); it is development only and never a result. If, after up to three passes, a smoke episode is still errored or
   has no tool call at all, the row stops and the cause is reported, with no patch. The same holds for an OpenRouter
   call served by a provider other than the pin. For Luna this is the check that OpenAI tool calling works on this task.
2. *Gate.* The smoke's mean $ per episode per arm, LLM and Jev dollars apart. Projection = spend so far + 1.2 × [n_cal ×
   c(calibration arm) + n_eval × (c(`llm_only`) + c(cascade arm))]. Both parts must fit their caps of (h); on the
   collaborator's key the new spend must also fit what the key may still spend under its cap. (n_cal, n_eval) = (100,
   100) if it fits, else (50, 50) (the first 50 seeds of each block), else the row stops after the smoke.
3. *Calibration* on n_cal tasks.
4. *Fit*, as in (f).
5. *Gate again*, with the calibration's actual spend: the evaluation tier of step 2 or a smaller one; the evaluation
   never grows.
6. *Evaluation* on n_eval tasks, with `JEV_EVAL_RUNS=1`.

Every stage runs up to three passes; passes 2–3 re-run only episodes that ended in an infrastructure error. Workers:
Luna 6 (smoke), 4 (calibration) and 6 (evaluation), below this key's 2,000,000 tokens per minute; DeepSeek and row 3
6 / 8 / 8. A watchdog stops the runner when the LLM's logged LLM or Jev spend, plus a margin for the episodes in flight
(workers × the smoke's largest cost per episode), reaches its cap. The collaborator's job adds a key-usage watchdog at
the key's usage at E13's start + $150.

(h) *Caps.* Luna: OpenAI **$60**, Jev **$5**. DeepSeek: OpenRouter **$15** (LLM) + **$5** (Jev). Row 3: OpenRouter
**$125** (LLM) + **$5** (Jev). The collaborator's job in total: at most **$150** on their key. Expected, from E7e's
tokens per episode (about 349,000 input tokens per `llm_only` episode) at these prices: Luna about $8 + $3 Jev,
DeepSeek about $4 + $3, row 3 about $100 + $3.

(i) *Readout* (`scripts/explore_e13_llms.py summarize` → `results/explore/e13/summary.{md,json}`, keys `e13/<llm>/...`
in `paper/generated/e13_llms.tex`; the collaborator's `results/handoff/e13/` is merged when present). Per LLM, the
paired complete case of `jev_refit` against `llm_only`: success; Δ in points (paired task bootstrap, B = 10,000, seed 0;
Newcombe beside it); wins and losses; cost per 1,000 episodes and the saving (task bootstrap); mean and median wall
time; the share of tool decisions delegated (per decision point and pooled); Jev's accuracy against the gold label on
the accepted decisions, on all of them, and on the tier rule's cancel-vs-ship decisions; the trigger's firing rate;
failure types; final-answer forms; the calibration fits with coverage at τ̂; spend against the caps. These are computed
by `explore_e7_hard.summarize` (E7e's readout) on the records. The descriptive paired NI outcome of `scripts/hard_ni.py`
(3-point margin, the A6.21 rule) is added for each LLM's row, as for Table 1's other rows. E7e's Flash row is shown
beside the E13 rows for context only (other seeds; thresholds fitted for Flash in E7c). Records are kept slimmed and
gzipped (`records/<llm>/`): step observations, decision states and questions are dropped, and System-1 distributions
over tools are cut to their top 5. The readout and the fit read nothing that is dropped; tests check that the readout
and the fit are identical on raw and slimmed records. Reported whatever the outcome; a row that does not run is
reported as not run, with the reason.

### Deviations 2026-10-07 ~14:12 UTC: "E4r: replicate and extension of E4" (exploratory, descriptive; registered before any E4r episode)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E4r)". It has no test of
its own, revises no registered or reported result (E4 and E4b stand as reported), selects nothing and changes no
profile. No E4r episode has run, and no model call has been billed for it. The only calls so far read OpenRouter's
public endpoint list and key metadata, or were rejected for an invalid model id (the $0 preflight). None of the 200
extension instances has been run by any earlier row. This was checked against `configs/`, `results/`, `runs/`, the S3
run prefixes (`hard/swe`, `e4`, `e4b`) and this registration.

(a) *Why.* E4 ran once on 100 held-out instances. Routing resolved 81, like the deep path alone, at 21.3 % lower cost
and 11.3 % less mean agent time. It saved no cost on E4's 45 development instances. At n = 100 the paper's paired
non-inferiority test at 3 points does not hold (3 wins, 3 losses; p = 0.135). The author asked on 2026-10-07 for
complete, publishable experiments. E4r asks whether E4's result replicates on the same instances and whether it holds
on more instances. **E4r was designed after Stage 1 and after E4, E4b and their readouts had been read. It is
registered here before any E4r episode.**

(b) *Harness, models, decision model and arms: E4's evaluation, unchanged.* mini-SWE-agent 2.4.6 runs unmodified
through `jevharness.adapters.mini_swe` with the overlays `swebench_jev.yaml`, `s1_jev_openrouter.yaml` and
`e4_deepseek.yaml` (+ `e4_fast.yaml` for `always_fast`). Calls go through `OpenRouterTimeoutModel` (600 s timeout;
unbilled cut-off generations are retried, as in E4 since `11b3fcd`).
- The deep path is `deepseek/deepseek-v4-pro` with reasoning on. The fast path is `deepseek/deepseek-v4-flash` with
  reasoning off. Both are pinned to StreamLake (fp8) with `allow_fallbacks: false`.
- Endpoint prices, read again on 2026-10-07, are unchanged from E4: Pro $0.2088 / $0.4176 per M input / output
  tokens, cache read $0.0174; Flash $0.042 / $0.084, cache read $0.0084.
- Jev is called through OpenRouter (`typesafe/jev-1.13`) with E4's routing question before each step: p(hard) ≥ τ or
  no answer → deep, else fast.
- Arms: `always_deep` (the reference), `jevagent` at **τ = 0.5** and `always_fast` (the lower bound; it cost about
  $0.011 per instance in E4).
- τ = 0.5 is E4's registered evaluation τ. `configs/explore/e4r_swe/threshold.json` copies E4's (source sha256
  `92bf0247…`). There is no dev phase and no tuning.
- Each arm and phase gets a new System-1 cache, so no Jev answer is reused from E4.
- Cost is OpenRouter's `usage.cost` per call (DeepSeek + Jev). Every call, Jev's included, uses one key, so the key's
  usage counts all of E4r's spend.

(c) *Instances* (`scripts/explore_e4_swe.py --exp e4r select`, `configs/explore/e4r_swe/selection.json`, committed
in `b491d3f` before any episode). Pool: SWE-bench Verified test, 500 instances; the pool hash `a6b0fd7c…` is E4's.
- **Replicate:** E4's eval list, unchanged: 100 instances, sha256
  `8d67497edcc71e78e2c0d49496f0bb9333dc87959addf7a3cf20eb440b89a1ee`.
- **Exclusion list for the extension** (250 distinct instances, every SWE-bench instance any earlier run touched):
  1. H5's 100 evaluation instances (`results/hard/swe/instance_ids.txt`).
  2. The H5 smoke pair, which was also E4's and E4b's smoke.
  3. E4's 45 dev instances (E4b's 30 dev instances are their first 30).
  4. E4's 100 eval instances (also E4b's eval list).
  5. Three development instances of the adapter: the L3 smoke pair `django__django-11099` and
     `pytest-dev__pytest-7432`, and the README's debug example `sympy__sympy-20590`.
- **Extension:** the remaining 250 instances are ordered by sha256("jev-explore-E4r-swe-2026-10-07" + newline + id).
  The first 200 form the extension (`extension_ids.txt`, sha256
  `46fe87bf1d22ca302f5da0081d0f1dd11fd02e8e0bb0706accbf77c5fac0bea9`). The last 50 are a reserve that is never run.
- **Run list:** `eval_ids.txt` (300 instances, sha256
  `99831a6c3dc0babb9288369a26478af4aa5392432894f7f4d2d8406b71cf652f`) interleaves the two lists as R E E R E E …, so
  the replicate and the extension run under the same machine and API conditions throughout.
- Smoke: the H5 smoke pair × 3 arms (development only, never reported).

(d) *Caps and execution.* As E4's evaluation: k = 1; 250 steps, $0.42 and 60 s per command per instance, the same
for every arm. One task-major shared pool, the arms of each instance in a seeded order (seed 732). Infrastructure
errors are re-run once. Evaluation is local swebench 4.1.0.
- **Where it runs.** On a collaborator's Linux machine with Docker, with an OpenRouter inference key of the
  collaborator's own account. The key's value never reaches the project.
- **Package.** `handoff/` runs it with one command: `handoff/run.sh e4r`. It does a $0 preflight, then the smoke,
  gate, eval and readout stages; then it publishes `results/handoff/e4r/`. The phase logic is E4's own driver
  (`scripts/aws/e4_swe.sh`, in its local mode) and `scripts/explore_e4_swe.py --exp e4r`.
- **Workers.** 24 by default, as in E4. A smaller machine may use fewer; the number is recorded in the published
  `host.txt`. Arms are compared instance by instance within E4r's run. E4r's agent times are not compared with
  E4's, which ran on another machine.
- **Fallback.** The project's own AWS host remains a fallback that runs the same phases (`scripts/aws/e4r_swe.sh`).
- **Execution record.** Where E4r actually ran, its run date, its worker count and its spend are recorded afterwards
  in a separate "E4r: execution as run" entry.

(e) *Budget, watchdog and gate.* **$150 of OpenRouter spend (DeepSeek + Jev) for E4r in total, smoke included.**
- The key's usage at the first smoke is E4r's start.
- A watchdog stops the agent runs when the key's usage reaches the start usage plus $150 (minus $1). The local driver
  also checks the same bound before each paid stage. Any other spend on the key counts too.
- mini-SWE-agent's global cost limit is $150 minus the logged smoke spend.
- After the smoke, `scripts/explore_e4_swe.py --exp e4r gate` checks five conditions:
  1. Every (instance, arm) record is present without an infrastructure error.
  2. Every agent call was served by StreamLake.
  3. Every fast-path call reported 0 reasoning tokens, and at least 95 % of deep-path calls reported reasoning
     tokens.
  4. Jev answered in the routing arm: every routing record has a Jev call, and at most 5 % of Jev calls failed. A
     failure fails open to the deep path, so this check is what shows the route actually ran.
  5. The projection fits: 1.5 × (sum over arms of the larger of the smoke's and E4's evaluation mean $ per
     instance) × 300 + the logged spend ≤ $150.
- If (5) fails, only the first K positions of the run list are run: the largest multiple of 3 that fits. If K < 150,
  E4r stops after the smoke and reports that.
- If any of (1)–(4) fails, E4r stops after the smoke and reports that.
- Expected spend: about $25 (E4's means). The projection is about $38.
- If the watchdog stops the run, the readout covers the instances complete in all three arms and says so.

(f) *Readout* (`scripts/explore_e4_swe.py --exp e4r readout` → `summary.md`, `summary.json`, `per_instance.csv` and
the keys `e4r/...` (`e4r_swe.tex`) in `results/handoff/e4r/`; moved to `results/e4r_swe/` and
`paper/generated/e4r_swe.tex` on the project's side unchanged), reported whatever the outcome.
- **Primary:** all held-out instances of E4r's run pooled (replicate + extension, 300), paired by instance. Per arm:
  the resolved rate, the mean $ per instance, and the mean and median agent-run wall-clock.
- Against `always_deep`:
  - Δ success in points, with wins and losses, the paired task-bootstrap 95 % interval (B = 10,000, seed 0) and
    Tango's paired 95 % score interval;
  - the one-sided non-inferiority test at the paper's 3-point margin (`metrics.ni_certified_p`, rule `bb_ordered`;
    "holds" when p ≤ 0.05), reported descriptively;
  - the cost saving and the mean-time saving (ratios of means), with paired bootstrap 95 % intervals;
  - the median paired time difference on co-solved instances, and the fast-share.
- The same for the replicate (100) and the extension (200) on their own.
- **Replicate agreement** with E4 on its 100 instances:
  - per arm, the resolved count in each run, the share of instances with the same outcome and Cohen's κ;
  - the routing effect (Δ, cost and time saving, NI) in E4 and in E4r.
- Sensitivity: E4's original run in place of the replicate (E4 + extension).
- An instance whose record still carries an infrastructure error after its one re-run, or that has no SWE-bench
  report, counts as unresolved (as in E4). Counts are reported.
- Descriptive, outside every Holm table, revises no registered result. E4 and E4r are both reported.
- **Published files.** Small gzipped records (per-call cost, latency and model; per-instance outcomes; the submitted
  patches; the SWE-bench reports), with no trajectory and no log. Nothing in them identifies the machine or the
  collaborator.

### Deviations 2026-10-07 ~14:12 UTC: "E12: training cap on the A10G host" (exploratory; amends E12 (e) and (i) before any E12 training step)

Registered before any E12 training step, temperature fit or evaluation episode. (a) *What happened.* At 13:54 UTC no
g6e.2xlarge or g6e.xlarge capacity existed in any us-east-1 zone, so the registered order of types gave a g5.8xlarge
(one A10G 24 GB, us-east-1d; `<instance-id>`). The AWS login of the coordinating session expired about one
minute after the launch. Until the author renews it, the host sits idle with only its power-off guard armed.
That idle time counts toward the 12 GPU-hours. (b) *Training cap.* On this GPU type E3 measured about 1.05 s per JevGym
SFT row per epoch with `gc_3072` (D25: 136 min, D50: 266 min for two epochs). That predicts about 125 minutes for the
7,000 replay rows plus about 30 minutes for the 7,000 shorter public items, close to the registered 180-minute cap. The
cap becomes **240 minutes**. This is fixed from time alone, before any E12 training step. (c) *Unchanged.* Everything
else in E12, including the 12 GPU-hour cap and the power-off guard at 690 minutes after boot (about 01:24 UTC on
2026-10-08). If the guard leaves too little time for every stage, the stages run in the registered order and the
readout reports which did not run.

### Deviations 2026-10-07 ~14:12 UTC: "E12b: reading the order rule" (exploratory, secondary, descriptive; registered before any E12b harvest, training or scoring step)

Secondary row of E12, exploratory, labelled "exploratory (E12b)". It has no test, sits outside every Holm table,
revises nothing and selects nothing. The model it trains is a NEW checkpoint, "AgentDecision-4B (random rules)", which
is never served live. AgentDecision-4B (frozen) and E7d's AgentDecision-4B (tier) are untouched. Nothing has run for
E12b.

(a) *Why.* The author asked, in E12's brief, whether the model learns to read the order rule rather than memorise it.
E7d trained on `tier`-rule states and lifted held-out cancel-vs-ship accuracy from 76.5 % to 95.3 %. That cannot tell
reading from memorising, because the training rule and the test rule were the same. E12b trains on rules that are
never `tier` and tests zero-shot on E7d's held-out `tier` states. It was designed after E7d's results had been read.

(b) *Generator: a small, tested extension.* `bench/suites/jevgym_hard.py` gains `multi_rule: random`: one rule per seed
from its own random stream (`random_rule`; the parent's draws, and so the world, customers and orders, are
unchanged). With equal probability the rule is "cancel above $T", "cancel below $T", or a per-tier rule in which each
of basic / silver / gold gets either a threshold or "never" (never with probability 0.3; at least one threshold).
Thresholds come from {150, 200, 250, 350, 400, 450, 500, 550, 650, 700, 750, 800}: never `tier`'s $300 or $600, so
`tier` itself is never drawn (a per-tier draw equal to it would be redrawn). The policy sentence follows `tier`'s
template. `given`, `amount` and `tier` are unchanged; the generator fingerprints and policy texts are tested, and so is
the oracle solving random-rule tasks (`tests/test_explore_e12.py`). The frozen `jevgym.py` is untouched.

(c) *Harvest ($0).* E7d's harvest (`explore_e12_train.py harvest-b`) on E7d's setting (multi, 12 customers, 60 near
distractors, 20 noise lines, strict, `max_steps` 45) with `multi_rule: random`: the simulated oracle executor (error
rate 0.05), NearMiss System-1 (eps 0.1), behaviour policies `lite_shadow` and `nearmiss_replace`. **Training seeds
322000–322599**, a new block (≥ 10000 and outside every guarded range; the training guards check it). Every label is
the oracle's gold.

(d) *Rows* (`build-b`; E7d's quotas, seed 0). 10 % of the tasks form the dev split. The training rows are 900
write-state rows (gold cancel or ship), 450 other tool-choice rows and 450 argument rows, plus a replay of 1,200 rows of
E3's D100 file (rebuilt on the host by `ablation_subsets.py build`). Temperatures are fitted on 1,000 v1 dev rows and
1,000 E12b dev rows (seeded samples).

(e) *Training.* E7d's continuation recipe exactly: the frozen adapter is continued with lr 5e-5, 1 epoch, 64 rows per
step, max length 4,096 and seed 0, using `gc_3072` and then `gc_2048` on out-of-memory only; capped at 120 minutes.

(f) *Scoring, offline only.* E7d's held-out `tier` states (seeds 321000–321199): `heldout_op` (600, whole), and
`heldout_ts_other` / `heldout_tp` trimmed to 225 rows each with E7d's seeded trim (the same rows as E7d's readout). The
files come from `s3://…/e7_hard/e7d/box/data/` and are sha256-checked against E7d's build record. `eval_decisions`
(n-perm 0) runs for the E12b checkpoint with its temperatures and for the frozen checkpoint on the same host.
Reported per stratum beside E7d's recorded numbers: frozen 76.5 / 91.6 / 95.1 %, tier-trained 95.3 / 100.0 / 98.2 %
(cancel/ship, other tool choices, arguments). Descriptive, reported whatever the outcome. No live run and no corpus C
(stated).

(g) *Execution.* The E12 host (`jev-e12-train`), with `scripts/aws/e12_train.sh` stages `b_inputs b_harvest` (CPU; in a
second chain beside E12's training) and `b_wait b_train b_eval` after E12's live stage and before `final`. Records go
to `s3://…/e12/box/b/`, the adapter to `s3://…/checkpoints/e12b_random/`. E12b uses $0 API and its GPU time comes out of
E12's 12 GPU-hours (expected about 2 hours). If too little time is left before the guard, E12b's training is not run
and is reported as such. Readout: E12's summary (section E12b), keys `e12b/...`.

### Deviations 2026-10-07 ~15:05 UTC: "E12c: the recipe on a stronger base, and same-engine serving cost" (exploratory, descriptive; registered before any E12c step)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E12c)". It has no test,
revises no registered or reported result and selects nothing outside its own development data. The model it trains is
a NEW checkpoint, **"StartLux-4B (mixed)"** (served as `startlux-4b-mixed`). Nothing has run for E12c.

(a) *Why.* Requested by the author on 2026-10-07 at about 14:55 UTC (relayed by the coordinating session), after E12's
registration and before any E12 result. In Figure 4a, AgentDecision-4B (78.8 %) sits below StartLux-4B (83.1 %) and Jev
(86.4 %) on E1. The author wants a trained model that beats StartLux-4B on the fixed questions, ideally comes close to
Jev, and stays first inside the agent. If its cost cannot beat Jev's, its accuracy should.

(b) *Base.* StartLux-Decision-4B: the Stage-2 manifest's weights (`startlux-models/StartLux-Decision-4B`, revision
`3890ae0c…`, A7.18 (d)), already on the E3 image at `/opt/startlux/models/StartLux-Decision-4B`. The sha256 of its weight
files is recorded on the host. **License: CC BY-NC 4.0.** The trained adapter and any merged weights are derivatives
under the same non-commercial terms: research use only, never released for commercial use. Environment:
`/opt/jev/v2venv` (A7.19 (f)'s serving venv for this Qwen3.5-hybrid base: transformers 5.18.0, flash-linear-attention
0.5.2, causal-conv1d 1.7.0) plus peft 0.21.2 (the harness venv's version), installed at setup. **Fallback:** a 4-step
training smoke runs first. If it fails, or its step time projects the two epochs of (d) beyond the 360-minute cap, the
base becomes **Qwen3-8B** (`Qwen/Qwen3-8B`, public) with the same data and recipe, in the harness venv, and the switch
is recorded. Cost implication: about twice the parameters of a 4B model, so on the same stack about twice the GPU time
per decision (E9's rule, about $0.19 per 1,000 decisions against AgentDecision-4B's $0.093). It could then beat Jev on
accuracy only, not on cost.

(c) *Data.* E12's files unchanged: `train_mix.jsonl.gz` (7,000 public tool-use items in E1's format plus 7,000 SFT v1
replay rows; sha256 in `results/explore/e12/build_info.json`, re-checked on the host) and E12's temperature set
`cal_dev.jsonl.gz`.

(d) *Recipe: SFT only, from a fresh LoRA,* as SFT v1 and E11. LoRA r 32 / α 64 on all linear layers; soft cross-entropy
+ 0.5 Brier; option permutation; lr 1e-4, cosine, 3 % warm-up; 2 epochs; 64 rows per optimizer step; max length 4,096;
seed 0. Memory setting `nogc_3072`, then `gc_3072`, then `gc_2048`, each only after an out-of-memory error. Capped at
360 minutes. The trainer saves the adapter at the end of each epoch.

(e) *Checkpoint selection, development data only.* The two epoch-end adapters are scored offline (`eval_decisions`,
accuracy, no temperatures) on two sets. (1) **E1's 299-item development prefix** (`configs/eval/e1_dev_prefix_ids.json`,
excluded from E1's 3,991 decisions), built by E1's converter and E12's renderer (`explore_e12_train.py devprefix`).
E12's audit already dropped every training unit that overlaps it. (2) **The held-out JevGym dev rows**
(`data/train_v1/dev.jsonl.gz`, 2,377 rows). The adapter with the higher mean of the two accuracies is selected; a tie
selects epoch 2, the recipe's endpoint. If only epoch 1 finishes within the cap, it is selected. **E1's 3,991
evaluation items are never used for selection.** Temperatures for the selected adapter: `training/calibrate.py
--min-bucket 200` on E12's temperature set.

(f) *Evaluation, each run once.* Served by `training/serve_jev.py` in the v2 venv, as JevAgentRL-v2 was, with no
extra arguments. (1) **E1's 3,991 decisions** (`configs/explore/e12_train/e1_e12c.yaml`: E1's pools, exclusions,
batching and 4 workers, a new cache), cost by E9's rule. (2) **The common states C**: `stage2_corpus.py` replay with a
new cache. (3) **Live**: replace mode at the tool choice on the same first 200 fresh Stage-2 tasks as Figure 4b (seeds
300000–300199, evaluation seeds, `JEV_EVAL_RUNS=1`, never trained on), Flash on the direct Gemini API, 4 workers
(`configs/explore/e12_train/live_e12c.yaml`). It is preceded by a smoke on development seeds 902–903 and E12's gate
rule, here with a $15 cap. Also, descriptive: the held-out JevGym dev files and E12's public development split.

(g) *Pre-registered outcomes, reported whatever they turn out to be.*
- (i) E1 accuracy against StartLux-4B (E9's record `e1_bfcl_when2call_startlux/s1_startlux_4b.jsonl`): the paired item
  bootstrap of the delta (B = 2,000, seed 0); "beats StartLux-4B" iff the interval's lower end is above 0; by how much.
- (ii) The gap to Jev on E1, with the same interval.
- (iii) Common-state accuracy **not worse than AgentDecision-4B's at a 1-point margin**: the lower end of the paired
  task bootstrap of the delta is at least -1 point.
- (iv) Live success on the 200 tasks, beside AgentDecision-4B's and Jev's rows.
- Also: whether the point is on E9's frontier (E9's rule) and its cost against Jev's.

(h) *Added readout, same-engine serving cost* (E12, E12c, AgentDecision-4B, StartLux-4B). The E9 costs of the open
models mix model size with the speed of each serving stack (AgentDecision-4B 268 ms median under `serve_jev`, StartLux-4B
111 ms under its vendor's server). All four models are therefore also timed with one fast engine: **vLLM** (version
recorded) with each LoRA merged into its base (peft `merge_and_unload`, bf16), automatic prefix caching on, on one GPU of
the E12c host. Items: E1's 3,991 decisions. Every model gets the prompt exactly as `serve_jev` renders it, so StartLux-4B
is timed on the same prompts as the others; its vendor prompt is not used, and its accuracy under this prompt is not its
E1 accuracy. One request per question; the requests of a state go together, as in E1's `batch_by_state`; 4 states are in
flight, matching E1's 4 workers. The answer is the most likely option code (restricted log-probabilities); its agreement
with the HF-served answers is reported as a check. After a warm-up pass over the first 100 states, each model is timed
once. Reported per model: wall-clock GPU seconds per decision (wall time / 3,991) and E9's rule (request latency /
questions), each × $2.24208 per GPU-hour, beside the E9-rule cost from its own serving stack. If vLLM cannot load a model,
that row says so and nothing replaces it.

(i) *Execution and caps.* A second new GPU host, launched by this session with `scripts/aws/e12c_train.sh`
(Name=jev-e12c-train; image `<aws-resource-id>`). Types in order g6e.2xlarge, g6e.xlarge, g6e.4xlarge, g5.8xlarge,
g5.4xlarge, each in every us-east-1 zone. SSM only. The Gemini key comes from Parameter Store into the live stages only.
**GPU: at most 10 GPU-hours more** (a power-off guard 590 minutes after launch, a 30-minute idle stop, and the host stops
when done). **Gemini: $15** for E12c's live smoke and live run (a watchdog). No OpenRouter or Jev call. Records go to
`s3://…/e12c/box/` and `results/explore/e12/box_c/`; the adapter goes to `s3://…/checkpoints/e12c_sl4b/`. If the cap or
the deadline (2026-10-09 18:00 UTC) cuts a stage, the readout says which. Readout: `scripts/explore_e12_train.py score`
(keys `e12c/...` and `e12/fast/...`) and E9's summarize (keys `e9/e12c/...`). An execution-as-run entry follows.

### Deviations 2026-10-07 ~15:15 UTC: "Exploratory row E6f: the trigger cells at tool choice, arguments and action" (exploratory, descriptive; registered before any E6f episode)

Outside both registered stages; labelled "exploratory (E6f)": **descriptive, no test, outside every Holm table, revises
no registered result, selects nothing and changes no profile; not the registered S2-L3b.** Requested by the author on
2026-10-07 (relayed by the coordinating session): define the module map's three undefined cells and run them, so that
the map, which goes back into the main text, is complete. Nothing has run for E6f: no trigger or always mode at tool
choice, tool arguments or action has run on any seed, and nothing has run on JevGym seeds 301600–301799 (checked in
`configs/`, every run's `experiment.yaml` under `runs/`, the docs and `results/`).

(a) *Why the cells were undefined, and the definition.* E6b, E6d and E6e left TRIGGER undefined at tool choice, tool
arguments and action: a trigger decides whether an optional module runs, and these points have no optional module (the
tool, its arguments and the action are needed at every step). E6f gives each of them one: an **LLM verification pass**
on the LLM's own proposal. At these points, TRIGGER = the decision model decides whether that pass runs:
(1) the LLM proposes the tool call (tool choice, tool arguments) or the admissible action (action) as in `llm_only`;
(2) the decision model is asked a fixed question, in the harness's prompt style, with the proposal filled in, on the
same System-1 state as the other cells: *tool choice* "The agent proposes to call `<tool>` with arguments <arguments>.
Should the agent re-check this choice of tool before the call runs: could another tool be the right next step?";
*tool arguments* "... Should the agent re-check these argument values before the call runs: could a value be wrong or
missing?"; *action* "The agent proposes to take the action `<action>`. Should the agent re-check this action before
it runs: could another action be the right next step?" (`harness/prompts.py` `q_verify_tool` / `q_verify_args` /
`q_verify_action`); (3) if p_yes ≥ τ, the LLM runs **one** verification call: its context plus a check note that names
the proposal (`verify_tool_note` / `verify_args_note` / `verify_action_note`: "[Check] Before your proposed call ...
runs, re-read the task and the observations so far and check ..."). It confirms the proposal or revises it once. At
tool choice every tool is offered; at tool arguments the call is forced to the proposed tool, so only the arguments can
change; at the action point the admissible list is shown again. A reply without a call (or an empty action reply)
leaves the proposal standing and is never read as a finish; (4) otherwise the proposal runs unchecked. The final
proposal is executed. **This is not a guard:** in a guard the decision model judges the proposal and vetoes it, which
forces a regeneration; here the decision model never judges or vetoes, it only decides when the LLM checks itself. The
argument trigger is asked about every call of a known tool that has parameters (`finish` included, whose argument is
the answer); the tool-choice trigger about every proposed call. A reference mode `always` (no decision model) runs the
verification on every proposal. Code: `harness/config.py` (`E6F_CELLS`; `trigger` and `always` added to `VALID_MODES`
at the three points; `UNDEFINED_CELLS` is now empty, so all 40 cells are defined; `tool_params` trigger / always
cannot be combined with a System-1-chosen call), `harness/agent.py` (`_verify_call`, `_verify_action`), and the
three questions and three notes in `harness/prompts.py` and in the prompt catalogue (`scripts/paper_prompts.py`).
Semantics: `results/explore/e6f_modes/cells.md` (generated from the code). Every System-1 and System-2 call goes
through the episode ledger at its decision point.

(b) *Golds (log only; never on the wire, canary tests in `tests/test_e6f_modes.py`).* Tool choice: the proposed tool
is in JevGym's information-feasible acceptable set AT(s); whether the whole call is progress (the tool guard's gold)
is logged beside it. Tool arguments: the proposed call is progress (`oracle_verdict`). Action: the proposal equals
ALFWorld's single gold action, a lower bound (as E6c). Each decision logs `gold` = the proposal needed a second look
(it was wrong) and, in `gold_aux`, whether the verification ran, whether it changed the proposal, the correctness of
the proposal and of the executed value, and the outcome: `fixed` (wrong → right), `broke` (right → wrong),
`confirmed_right` / `confirmed_wrong`, `revised_right` / `revised_wrong`. Every mode that existed before E6f makes the
same decisions and calls: the three earlier fingerprint fixtures are unchanged, and a fourth, of the E6e modes, was
generated on the pre-E6f code (`tests/fixtures/e6f_prior_modes_fingerprint.json`, at d04fea0).

(c) *Thresholds, fixed now; nothing fitted or tuned on evaluation data.* E6b's rule (E6b entry (c), kept by E6d and
E6e): the registered τ of the same decision point's trigger or guard where one exists, else 0.9. **Tool choice τ =
0.5** (the Stage-1 tool guard on Flash, `jev_ts_guard`); **tool arguments τ = 0.9** (E6b's argument guard; no Stage-1
τ at that point, and E6b's else-clause gives the same 0.9); **action τ = 0.5** (E6c's action guard).

(d) *Arms.* JevGym (`configs/explore/e6f_modes/eval.yaml`): `llm_only` (the reference of both cells), `ts_trigger`
(tool choice, τ 0.5), `ts_verify_all` (tool choice, always), `tp_trigger` (tool arguments, τ 0.9), `tp_verify_all`
(tool arguments, always). ALFWorld (`configs/explore/e6f_alfworld/eval.yaml`): `llm_only`, `jev_action_trigger`
(τ 0.5), `action_verify_all`. The module-map cell compares each trigger with the `llm_only` of its own run, as every
other cell; the always-verify arm is a reference that says how much of verification's effect the trigger keeps and
at what share of its cost.

(e) *Tasks.* **JevGym: fresh seeds 301600–301799 (n = 200)**, named here so that no other row uses them; the runner
does not guard them, so no pilot or training run may touch them. Suite settings as E6b / E6d / E6e (`families: v2`,
`distractor_style: near`, `n_distractors: 60`, `noise_lines: 20`, `strict: true`). E6d's block (301000–301199) is
**not** reused: its `llm_only` ran on the OpenRouter route to Flash, which reports about half the direct API's input
tokens (E6d addendum), so it cannot be paired with arms on the direct API. E6f therefore runs a fresh block with its
own `llm_only`. **ALFWorld: the 140 valid_seen games of E6a / E6c** (`configs/explore/e6a_alfworld/eval_games.json`,
same stratified order, the same local data; the id list matches) with E6a / E6c's settings (oracle `auto`, `help` /
`inventory` excluded, 50 steps, history 12).

(f) *Models and route.* Gemini 3.5 Flash on the **direct Gemini API** (temperature 0, minimal reasoning) in every arm.
Jev via OpenRouter (`{type: jev, provider: openrouter}`) on the author's Jev-only key `OPENROUTER_API_KEY_JEV` (read
only by `decision/jev.py`), with no System-1 cache. If the direct API is overloaded in the smoke (HTTP 503 or repeated
infrastructure errors), the row is paused and the pause recorded; routes are never mixed within the row.

(g) *Execution.* Locally, in this container (the Jev-only key never goes to AWS). JevGym: one runner invocation, all
five arms task-major interleaved (seed 732), **12 workers**; passes 2–3 re-run only episodes that ended in an
infrastructure error. ALFWorld: task-major (seed 732), **8 workers**, in runner invocations over the first 20, 40,
..., 140 games of the list (`--max-tasks`; each resumes and skips completed games), with `TMPDIR` on `/dev/shm`,
because TextWorld keeps a 32 MB copy of its planner library per game until its process exits and the container's disk
is nearly full; a last invocation re-runs infrastructure errors once. The two evaluations may run at the same time
as separate processes; each cell is paired within its own run. While the row runs only operational fields are read.
Records are gzipped after the readout. No existing env code path changes (another row runs JevGym-Hard episodes from
the same tree).

(h) *Smoke, gate and caps.* Smoke (development, never reported): the five JevGym arms on seeds 900–901
(`configs/explore/e6f_modes/smoke.yaml`) and the three ALFWorld arms on the first 2 of E6a's calibration train games
(`configs/explore/e6f_alfworld/smoke.yaml`, E6c's smoke games). **Caps for smoke and evaluation together: Gemini $60,
Jev $2** (as logged per episode at list prices). Projection (`scripts/explore_e6f_modes.py project`): per setting, the
smoke's mean $ per episode per arm (Gemini and Jev apart), summed over the arms, × the evaluation size × 1.5, plus the
spend so far, against both caps. In order: JevGym 200 + ALFWorld 140; else JevGym 100 (301600–301699) + ALFWorld 140;
else E6f stops after the smoke. A watchdog (`explore_e6f_modes.py watch`) stops every E6f runner when the logged
Gemini spend reaches $60 or Jev's reaches $2; an incomplete row is reported with its n.

(i) *Readout* (`scripts/explore_e6f_modes.py summarize` → `results/explore/e6f_modes/{summary.md, summary.json,
cells.md}`, keys `e6f/...` in `paper/generated/e6f.tex`). Per cell the module map's paired comparison with its run's
`llm_only`: paired complete case; Δ success in points (paired task bootstrap, B = 10,000, seed 0; Newcombe beside it);
wins and losses; cost per episode and saving (task bootstrap); the fixed descriptive verdict of `scripts/module_map.py`
(costs success if Δ ≤ −3 pt; no saving if saving ≤ 0; takes over safely if Δ ≥ −1 pt, saving > 0 and acted ≥ 5 %; else
mixed), with acted = the share of proposals on which the trigger fired; the trigger's accuracy against the gold "the
proposal is wrong"; calls and wall time per episode. Then the verification itself, for the trigger arm and its
always-verify reference: proposals, the share verified, the share of wrong proposals verified, and the outcomes
(fixed, broke, confirmed, revised; replies without a call). The always-verify comparison: its own Δ and saving against
`llm_only`; the trigger's share of its extra cost, (C_trigger − C_ref) / (C_always − C_ref); its share of its Δ (only
when that Δ ≠ 0) and of its fixes; the paired trigger − always Δ. `scripts/module_map.py` reads the summary like E6b's
into the exploratory tiles `mm/xcell/flash/{tool_select, tool_params, action}/trigger/...`, never mixed into a
Stage-1 cell, count or module verdict. Expectation, stated before any episode: on JevGym the LLM alone is near the
ceiling (the `llm_only` references of E6b / E6d / E6e succeeded 99.5 %, 100.0 % and 99.5 %), so these cells will most likely show no gain and extra cost. Reported
whatever the outcome.

### Deviations 2026-10-07 ~15:19 UTC: "E4r: route and host" (exploratory; amends E4r (d) before any E4r episode)

Registered before any E4r episode. No E4r smoke, gate or evaluation call has been made. The E4r entry above (d) says
the run uses a collaborator's OpenRouter key on the collaborator's machine. That plan changes as follows.

(a) *Host.* E4r runs on the authors' own AWS host `<instance-id>`, as originally prepared: an m7i.8xlarge (32
vCPU, the type of E4's host), Ubuntu 24.04, 500 GB gp3, SSM only. The box is already set up, and all 302 instance
images were pulled before this entry with 0 failures. It is driven by `scripts/aws/e4r_swe.sh` (stages `rearm update
smoke gate eval final`), and the phases run through E4's own driver `scripts/aws/e4_swe.sh` in its AWS mode. The
host runs 24 agent workers and 5 evaluation workers per arm, as E4 did. The driver's power-off guard (10 hours for the
evaluation) and its 30-minute idle stop stay in place.

(b) *Key.* The run uses one OpenRouter inference key of the authors' account, named "jev-harness-e4r", with a hard
credit limit of $150. It was created on 2026-10-07 at 15:19 UTC. It sits only in SSM Parameter Store
(`<ssm-prefix>/OPENROUTER_API_KEY`, SecureString) and is read into the stage processes only. DeepSeek and Jev both use
it, and its value is never printed or written anywhere. A $0 probe (an invalid model id) was rejected with HTTP 400,
which shows it is an inference key. The smoke stage checks the key type again before any paid call.

(c) *Unchanged.* Everything else in the E4r entry stands as registered:
- arms, models, provider pin, τ = 0.5, the caps per instance;
- the instance lists and their hashes, the interleaved run list, the smoke pair;
- the $150 cap, now also the key's own hard limit;
- the key-usage watchdog and the five gate conditions;
- the readout, the replicate agreement and the sensitivity view.

The collaborator package (`handoff/`) stays in the repository unused. The execution-as-run entry follows after the
run.

### Deviations 2026-10-07 ~15:30 UTC: "E13: collaborator rows optional" (configuration defaults only; recorded before any collaborator episode)

Because OpenRouter spend matters to the author, E13's collaborator rows (DeepSeek-V4-Flash and row 3) are
optional and are not planned to run for now. The handoff package's defaults become the DeepSeek row only, under a $60
cap (`handoff/experiments/e13/caps.env`); row 3 runs only on request (not run; cap up to the
registered $150). Every other E13 rule, cap per row, seed and readout is unchanged; a row that does not run is
reported as not run. The Luna row is unaffected.

### Deviations 2026-10-07 ~15:33 UTC: "E4r: minimum spend" (exploratory; amends E4r (c), (e) and (f) before any E4r episode)

Registered before any E4r episode. No E4r call has been made. The host was started at 15:20 UTC and only its guards
were armed. The author asked to keep OpenRouter spend to a minimum.

(a) *Run list.* The 100-instance replicate is not run again. E4r runs only the **200 extension instances**
(`extension_ids.txt`, sha256 `46fe87bf1d22ca302f5da0081d0f1dd11fd02e8e0bb0706accbf77c5fac0bea9`, in that order;
`eval_ids.txt` is now this list). The three arms, τ = 0.5, the models, the provider pin and the per-instance caps are
unchanged. The interleaved 300-instance list of the E4r entry (sha256 `99831a6c…`) is superseded and never run;
`configs/explore/e4r_swe/selection.json` records it.

(b) *Smoke: one instance × 3 arms*, `scikit-learn__scikit-learn-13496`, development only. It is one of the H5 smoke
pair: the cheaper one in E4's smoke, at $0.022 for its three arms against $0.073. One instance per arm is the minimum
the gate needs, because each arm's own calls are checked: the provider, the reasoning toggle on both paths, and Jev.

(c) *Cap: $40 in total for E4r, smoke included.* The key's own hard limit stays at $150.
- The key-usage watchdog stops the agent runs at the start usage plus $40 (minus $1).
- mini-SWE-agent's global cost limit is $40 minus the logged smoke spend.
- Gate conditions (1)–(4) are unchanged. Condition (5) becomes: the projection for the 200 instances (1.5 × the larger
  of the smoke's and E4's per-instance means, summed over arms, × 200) plus the logged spend must be ≤ $40.
- If it is, all 200 run. Otherwise the first 100 extension instances run, if their projection fits; this would be
  reported. Otherwise E4r stops after the smoke.
- Expected spend: about $17 (E4's means × 600 agent runs). The projection is about $25.

(d) *Readout.* The primary view is now the pool of E4's own run on its 100 evaluation instances and E4r's run on the
200 extension instances: **300 distinct held-out instances**, paired by instance.
- All three arms of an instance come from the same run, on the same host type (m7i.8xlarge, 24 workers).
- It is reported as in (f): per arm the resolved rate, cost and time; Δ with the paired bootstrap and Tango intervals;
  the 3-point NI test, descriptively; the savings with their intervals.
- Also reported: the extension alone (200) and E4 alone (100, as published).
- Dropped: the replicate view, E4r's own 300-instance pool and the replicate agreement.

(e) *No other OpenRouter call.* Besides the run itself, nothing calls OpenRouter except $0 metadata reads (the key's
type and usage).

Everything else in the E4r entry and the "E4r: route and host" addendum stands.

### Deviations 2026-10-07 ~15:41 UTC: "E12c: same-engine timing on the E12 host" (exploratory; amends E12c (h) and (i) before any E12c timing step)

Registered before any vLLM install, merge or timing. (a) *What was found.* E12c's host is a g5.8xlarge (one A10G; no
L40S capacity in any us-east-1 zone at 15:12 UTC). Its training smoke measured 47.8 s per step with `gc_3072`
(`nogc_3072` ran out of memory), which projects 341 of the 360 training minutes. The stages after training then leave
no slack before E12c's power-off guard (01:00 UTC). (b) *Amendment.* The same-engine timing of E12c (h) runs on **the
E12 host** `<instance-id>`, the same instance type (g5.8xlarge, one A10G), after E12's own chain (stage `fast` of
`scripts/aws/e12_train.sh`). All four models are timed there, one at a time, with the protocol of E12c (h) unchanged.
E12c is timed once its selected adapter is in S3 (the stage waits up to 120 minutes). If E12's host stopped at the end
of its chain, it is started again for this stage only. Its GPU time counts toward **E12's 12 GPU-hours** (about 8 used
by the end of E12's chain), so E12c's 10 GPU-hours are not touched by it. E12c's own `fast` stage is not run: its
chain is stopped after `live` and its `final` stage is sent alone. (c) Nothing else changes.

### Deviations 2026-10-07 ~15:55 UTC: "E13: GPT-6-Luna execution as run" (exploratory, descriptive; written after the run, before any paper text used it)

(a) *Times.* The E13 entry above says "~14:00 UTC"; it was committed at 14:11:52 UTC (`5462386`), before the first
E13 episode (smoke from 14:12:21 UTC). The commit time is authoritative. The collaborator rows did not run (the note
"E13: collaborator rows optional"); they are reported as not run.

(b) *As run* (`scripts/explore_e13_llms.py --llm luna run`, locally; status `results/explore/e13/luna/status.json`).
Smoke 14:12:21–14:14:01 UTC: 6 episodes, no error, every episode with tool calls; both LLM-only smoke episodes ended at a
wrong `ship_order` under strict mode (a rule error, not a tool-calling failure), and the cascade arm's forced
single-tool argument calls worked. Gate: calibration 100, evaluation 100. Calibration on seeds 5400–5499 from 14:14:27
UTC, 4 workers. At about 14:29 UTC the shared disk filled; one record was cut and the runner could no longer write
records, so the driver and runner were stopped at 14:37 UTC and the stage resumed at 14:37:56 UTC (the 56 tasks without
a complete record; finished 14:55:48 UTC, 100 tasks, no error). The driver now leaves a stage unfinished when its runner
dies and stops the runner when the disk is nearly full. Fit 14:56:05 UTC, by the registered estimator: **tool choice
τ̂ = 0.49, arguments τ̂ = 0.61** (2,790 and 4,469 paired decisions; at τ̂ Jev takes 100 % and 90.2 % of them). On the
calibration states Jev is 99.9 % accurate on tool choice against Luna's 98.5 %, and 100 % on the tier rule's 911
cancel-or-ship choices against Luna's 95.3 %. The fit was committed at 15:02:03 UTC (inside `e055c5c`, a snapshot
commit of the coordinating session), before any evaluation episode. Gate before the evaluation: evaluation 100.
Evaluation on seeds 311000–311099 from 15:11:26 UTC, 6 workers, `JEV_EVAL_RUNS=1`. The background job that ran it hit its
30-minute limit at 174 of 200 episodes and was stopped with the runner; the evaluation resumed at 15:43:28 UTC (the 26
episodes without a record) and finished at 15:48:24 UTC. No episode errored; no task left the paired comparison. Raw
runs were slimmed and gzipped into `results/explore/e13/records/luna/` (sha256 of the raw files in each `slim.json`)
and deleted.

(c) *Spend.* Logged: OpenAI $1.46 and Jev $2.19 ($3.65; caps $60 and $5). Not logged: the route probe (under $0.001),
the calibration episodes that finished after the disk filled and before the stop (about 24, about $0.3 at the
calibration's $0.011 per episode), and at most 6 evaluation episodes in flight at the stop (under $0.1). The tasks of
lost records were run again on resume.

(d) *Readout* (`scripts/explore_e13_llms.py summarize` → `results/explore/e13/summary.{md,json}`, keys `e13/luna/...` in
`paper/generated/e13_llms.tex`), as registered, whatever the outcome. On the same 100 tasks: **LLM-only 45.0 %, Jev
refit 83.0 % (+38.0 points, paired bootstrap [+26.0, +50.0]; 44 wins, 6 losses).** 44 of LLM-only's 55 failures are a
wrong cancel-or-ship call that strict mode stops; the Jev arm has one such failure. Jev took 82.0 % of the tool
decisions (99.9 % of tool choices, 70.9 % of arguments), 99.1 % of the accepted ones right, and every one of the 1,182
cancel-or-ship choices right. **Cost: $22.8 against $2.7 per 1,000 episodes**, so the Jev arm costs about 8 times as
much. Jev is $13.7 of it. LLM-only episodes are shorter (29.1 against 37.5 steps; many stop at a wrong write), and 95.5 %
of LLM-only's input was billed as cache hits against 0.9 % in the Jev arm, whose delegated steps break the cached
prompt prefix (a descriptive readout line, `summary.md`). Time: 67.4 against 53.9 s per episode. The descriptive paired NI test of `scripts/hard_ni.py`
(3-point margin) shows non-inferiority (p < 0.001). Exploratory; no test.

### Deviations 2026-10-07 ~17:10 UTC: "E14: a sub-1B decision model with a scoring head" (exploratory, descriptive; registered before any E14 training step or evaluation episode)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E14)" (post-Stage-1;
marked with the dagger in the paper). It has no test, revises no registered or reported result and changes no profile
or threshold of any other row. The models it trains are NEW checkpoints on a new base; AgentDecision-4B (the frozen
JevAgentRL-4B checkpoint of the records) stays frozen and untouched. Nothing has run for E14: no E14 model has been
downloaded to a GPU, trained or asked a question, and no E14 episode exists. Read before registering, at $0: the code
of the `unsloth` 2026.10.2 wheel (`unsloth/models/decision*.py`, `clef.py`, the vendored Cloudflare head
`unsloth/_vendor/clef/joint_schema_model.py`, `studio/backend/core/training/decision_trainer.py`, the model-default
YAMLs), Unsloth's `scripts/train_decision_from_lm.py` (GitHub main, fetched 2026-10-07), the guide
(https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth), public model cards, and SFT v1's training
input rebuilt locally on CPU (its sha256 matches, see (c)).

(a) *Why.* The author asked on 2026-10-07 (relayed by the coordinating session) for a much smaller decision model
(at most 1B) that does well at low cost, and suggested Qwen3.5 as a starting point. Unsloth has just released
decision-model training: a plain language model plus a fresh Clef "joint schema head" (Cloudflare's design: one logit
per allowed option of every question, all questions of a request scored jointly), trained with LoRA. Its guide reports
Qwen3.5-0.8B rising from 20.7 % to 74.3 % mean accuracy over typed-decisions, BANKING77 and CLINC150 after training on
a generic 12-source classification mix. E5 (Deviations 2026-10-06 "Exploratory row E5") found that generative SFT v1
students of 0.6B-1.7B came within 1 point of the 4B on the common states C but lost 1.5-6 points after off-plan steps,
that their option-token confidences were so peaked that the fitted cascade thresholds sat near 1, and that the live 1B
harness saved nothing once GPU time was priced. **E14 asks whether a Clef-style scoring head on Qwen3.5-0.8B, trained
on the agent's own states, gives (i) usable, calibrated confidences, so the cascade takes most decisions, and (ii)
near-4B accuracy at a fraction of the serving cost.** It is also the training-side test of RQ1 (Figure `fig:frontier`:
fixed-question scores do not rank decision models for agents): the same backbone and head trained on generic decision
data (Unsloth's recipe) against the same trained on the agent's states.

(b) *Model and recipe, every arm.* Backbone `Qwen/Qwen3.5-0.8B` (revision `2fc06364715b967f1860aea9cf38778875588b17`,
Apache-2.0; hidden size 1,024; only the language model is used). Unsloth 2026.10.2
(`FastDecisionModel.from_pretrained(..., decision_head="clef")`) adds a fresh joint schema head with Unsloth's default
size for small backbones (width 512, 2 routing + 4 decoder layers, 8 heads, feed-forward 2,048; seed 3407). The recipe
is Unsloth's from-LM script with its defaults, which equal `llm_decision_defaults.yaml` in rank, learning rate, batch,
accumulation, epochs and length: LoRA r 64 / alpha 64 / dropout 0 on all linear layers of the language model
(Unsloth's Clef LoRA; the vision tower and the output embeddings frozen), the head trained in full at lr 3e-4, LoRA
lr 1e-4, cosine schedule with 5 % warm-up, weight decay 0.01, 8 records per device x 2 accumulation steps, maximum
length 4,096 tokens (Unsloth cuts the end of a longer state, never a question or option; the build reports how many
training records are cut), bf16, Unsloth gradient checkpointing, no head warm-up, no 4-bit loading, trainer seed
3407, Unsloth's Clef loss (soft-target cross-entropy; no label smoothing, Brier or field permutation, its
`CLEF_RECIPE` defaults). Temperatures: Unsloth's `calibrate` (a head temperature fitted to the gold label plus per-type
temperatures) on the arm's development records (A: (c) 2; G: its held-out split). One seed per arm. Unsloth's
decision-model code is AGPL-3.0: it is run, not redistributed; the Clef head code it vendors is Apache-2.0.

(c) *Data.*
1. **Arm A, the agent's states (the main arm).** SFT v1's training input exactly: `train_mix.jsonl` rebuilt by
   `python -m jevharness.training.data mix --in data/train_v1/train.jsonl.gz:1 data/train_v1/replay_when2call.jsonl.gz:0.45
   --seed 0` (15,694 JevGym rows of SFT v1 + 1,208 When2Call-train replay rows = 16,902 rows); its sha256
   `9fe1088d8e5be533…` equals SFT v1's (`results/training/ablation_inputs.json`), checked again on the host. Converted
   to Clef records by `jevharness.training.clef_data` (committed with tests before the first training step): every
   question in the wire form the harness sends (`decision/jev.question_to_wire`): a Choice's criteria are its option
   labels with their descriptions (tool choice: "name -- description" split at the first " -- "; arguments and When2Call:
   no description); a Noul without criteria. One record per decision, except that the argument questions of one agent
   step form one record with questions `q0`, `q1`, ... in their logged order, as the final harness batches them per
   step. Gold: the row's target distribution (soft where the row is soft) with its argmax as label. **Gold enters only
   the training targets; no served request ever carries it** (`Question.meta` stays off the wire). The JevGym seeds of
   every training row are re-checked against the training guards (`bench/registry.training_exclusions`,
   `training/seed_guard.TRAIN_FORBIDDEN`); no evaluation seed is in any training file.
2. **A's development split.** SFT v1's held-out rows `data/train_v1/dev.jsonl.gz` (2,377 rows, never trained on),
   converted the same way; `dev_recovery` and `dev_knockout` for the read-outs only.
3. **Arm G, generic.** Unsloth's published mix, built on the host exactly as its from-LM script builds it with its
   defaults: 12,000 rows from its default sources (ag_news, arc, banking77, boolq, clinc150, commonsense_qa, mmlu
   auxiliary_train, mnli, prompt_injections, snli, sst5, wanli; xLAM is gated and is skipped as the script skips it,
   which leaves the guide's 12 sources), each row with Unsloth's schema augmentation, decontaminated (13-word shingles)
   against the evaluation sets, plus typed-decisions train (config `all`, augmented and decontaminated); `split_holdout`
   (seed 3407, 10 %, at most 400 decisions) gives G's held-out split. Evaluation sets as the script's: typed-decisions
   test, BANKING77 and CLINC150 test (500 each, seed 0).
4. **Arm AG** (only under (d)): A's records together with G's training rows, the same recipe; temperatures on A's
   development split.

(d) *Epochs and selection, development data only.* **A**: epoch 1 is Unsloth's default. A second pass over the same
records with the same schedule restarted runs only if the used GPU-hours plus epoch 1's duration plus 3.5 hours
reserved for evaluation, serving and live stay within the cap of (i). The epoch with the higher accuracy on A's
development split is used; a tie keeps epoch 1. **G**: one epoch, the published recipe; nothing is selected. G's
reproduction check (before G is used anywhere): base (the fresh head before training) and trained accuracy on the three
Unsloth sets; G counts as reproduced iff its trained mean over typed-decisions, BANKING77 and CLINC150 is at least
69.3 % (the guide's 74.3 % minus 5 points). If not, G's L1 and L2 numbers are still reported, labelled "not
reproduced", and G gets no live row. **AG**: A's selected epoch count; trained only if, after A, G and A's evaluation,
the used GPU-hours plus AG's projection (A's and G's measured epoch times) plus the remaining stages stay within the
cap. **The common states C, E9's items (E1's 3,991 decisions and E1's development prefix) and every evaluation task are
never used for any choice.**

(e) *Serving.* `training/serve_jev.py --backend clef` (committed before the first served request): the arm's checkpoint
(Unsloth's `save_pretrained_merged` folder, its temperatures in its decision config) loaded by Unsloth's loader; each
wire request is encoded with Cloudflare's record encoding (all its questions in one pass) at up to 16,384 tokens
(Unsloth's serving length; nothing is cut at serving); answers in the Jev wire format with unrounded probabilities;
one request at a time on the GPU. The harness asks it through the `jev` backend, so every System-1 call goes through the
Ledger. An offline test with a stand-in model (and Unsloth's own head when Unsloth is installed) checks the wire mapping.

(f) *Offline evaluation, each run once* (the E14 host, no API).
1. **L1 static.** E1's 3,991 decisions with E1's config (pools, C8 exclusions, batching and 4 workers; a new answer
   cache; `configs/explore/e14_small/e1_e14.yaml`): accuracy pooled and per family; cost per decision by **E9's rule**
   ($2.24208 per GPU-hour times the logged serving time per decision). The BFCL tool-choice items as Figure 4b's static
   column. The E14 points join E9's drawn set in their own keys (`e9/e14a/...`, `e9/e14g/...`, `e9/e14ag/...`); E9's
   registered frontier and keys stay unchanged. Paired item bootstrap (E9's `paired_gap`, B = 2,000, seed 0) against
   AgentDecision-4B and StartLux-0.8B.
2. **L2 common states C** (`results/stage2/s23/`; `stage2_corpus.py replay`, a new cache): task-mean accuracy over all
   states, after an off-plan step and on the ask-vs-act stratum, and ECE, as E5. Paired task bootstrap (B = 2,000,
   seed 0) against AgentDecision-4B (`answers_rl.jsonl`), SFT v1 at 4B and E5's Qwen3-0.6B, with E5's reading rule
   (keeps the reference's accuracy if the lower end is at least -1.0 point; loses it if the upper end is below 0; else
   inconclusive).
3. **Calibration** (E5's offline procedure, entry (g)): the 1,125 tool-choice and 1,328 argument questions of the
   frozen 4B's calibration shadow arm (`rl_shadow_cal`, Flash, calibration seeds 5000-5149), answered by the served
   model (`scripts/e5_cal_replay.py`); tau-hat per decision point by the registered command (`threshold_theory
   --calibration-arm --task-value 1.0 --lam-per-hour 0 --v-fallback W`). Reported: tau-hat, the share of those
   decisions the cascade takes at tau-hat (coverage), and the accuracy on them of System-1, the LLM and the cascade, as
   E5's table. A point whose fit accepts no level always defers.
4. **Serving** (one session on the E14 host's A10G): the final harness's request stream (`scripts/harness_final.py
   bench` on `results/harness_final/replay_workload.jsonl.gz`; c = 1 on 25 episodes, c = 4 on 100), p50 / p90 ms per
   request and System-1 ms per step; peak GPU memory (`nvidia-smi`) while serving and while training. The frozen
   AgentDecision-4B (sha256-checked) is served by `serve_jev --fast` on the same host in the same session as the
   reference; speed-up = its p50 over ours at the same c.
5. Also: A's development read-outs (accuracy per decision point, after an off-plan step, knockout, ECE), and every
   arm's accuracy on Unsloth's three evaluation sets (base and trained).

(g) *Live (L3).* Gemini 3.5 Flash on the direct Gemini API (temperature 0, minimal reasoning); evaluation seeds run with
`JEV_EVAL_RUNS=1`; no System-1 decision cache; passes 2-3 re-run infrastructure errors only.
- **(a) REPLACE at the tool choice** (`configs/explore/e14_small/live_a.yaml`), exactly as `rl_ts_replace_f`, the E2
  ladder and E12: the first 200 fresh Stage-2 tasks (seeds 300000-300199), 4 workers. Arm A; arm G too if G is
  reproduced and the caps allow. Reported: success (latest record per task, errored tasks left out), own-state
  accuracy, steps, calls, API $ per episode, wall-clock; beside the logged frozen, Jev and StartLux-0.8B rows of
  Figure 4b (paired Newcombe intervals).
- **(b) The final harness with arm A** (`configs/explore/e14_small/live_b.yaml`; profile
  `configs/harness/explore_e14_final.yaml` = `jevagent_final.yaml` (tool + argument cascade, reflection off, argument
  questions batched per step) with a backend block for arm A at the tau-hat of (f) 3, committed before any (b) episode).
  Tasks: the 200 tasks of the logged dedicated final-harness rows (seeds 300500-300699; `runs/harness_final_dedicated`:
  `llm_only` and `harness_final` with the frozen 4B, 1 worker, g5.4xlarge, 2026-10-05), under the same settings: 1
  worker, task-major schedule (seed 732), the model alone on its own GPU. **Only the new arm runs.** Reported: success,
  API $ per episode, GPU tariff q (host $ per hour x mean episode wall-clock; the logged rows' tariff file, and E9's
  $2.24208 per GPU-hour as a sensitivity), wall-clock and System-1 / System-2 time per episode, the share of tool and
  argument decisions System-1 took; paired against both logged rows (Newcombe, paired task bootstrap). The logged rows
  ran two days earlier on another host type, so wall-clock differences are read descriptively. If arm A's fit always
  defers at both points, (b) is not run ("equals LLM-only by construction").
- Smoke before each live configuration: 2 JevGym development tasks (seeds 906-907; never reported). Gate: the smoke's
  mean $ per episode x the episodes to run x 1.5, plus the spend so far, must fit the Gemini cap of (i); else that
  configuration does not run.

(h) *Optional, lowest priority.* Cloudflare `clef-flash` (9B; revision `fde727a287004204b7518dcc983fe64379776712`,
Apache-2.0) scored offline on E1's 3,991 decisions and on C through the same server, only if GPU time remains after
(a)-(g) and it loads on the A10G; no live row, no threshold.

(i) *Execution and caps.* One new GPU host launched by this session with `scripts/aws/e14_small.sh` (Name=jev-e14-small;
image `<aws-resource-id>`, E3's clone; g5.2xlarge with one A10G 24 GB, else g5.4xlarge, else g5.8xlarge), SSM only.
Unsloth 2026.10.2 in its own venv on the host. **GPU: at most 10 GPU-hours** (host hours: a power-off guard 590 minutes
after launch, a 30-minute idle stop, the host stopped when done). **Gemini: at most $40** (live smokes and live runs;
a watchdog on the logged spend). **No OpenRouter and no Jev call.** The Gemini key comes from Parameter Store into the
live stages' process only, never printed, never on disk; `OPENROUTER_API_KEY_JEV` never goes to AWS. Downloads: the
backbone, Unsloth's public datasets, optionally clef-flash. Records to `s3://…/e14_small/`, merged checkpoints to
`s3://…/checkpoints/e14_small/<arm>/`. `scripts/stage2_spec.py --gate` has no entry for exploratory rows and does not
apply. If a cap or the deadline (2026-10-09 18:00 UTC) cuts a stage, the readout says which.

(j) *Reading rules* (reported whatever the outcome; descriptive; one seed per arm).
- (i) *Usable confidences*: holds for an arm iff at both decision points its fitted tau-hat is at most 0.99 and the
  cascade takes at least 50 % of the calibration-state decisions at tau-hat; C ECE is reported beside it.
- (ii) *Near-4B accuracy at a fraction of the cost*: C accuracy keeps AgentDecision-4B's within 1 point (the lower end
  of the paired task bootstrap at least -1.0 point), and both the c = 1 p50 per request in the same session and the
  E9-rule cost per decision are below the 4B's.
- (iii) *Training-side RQ1*: A against G at each level (L1 accuracy, L2 accuracy, L3 REPLACE success if G runs live).
  "Fixed questions mis-rank the two" iff G is ahead of A on L1 while A is ahead of G on L2, each with a paired interval
  excluding 0.
- Live (b): success, cost incl. GPU and wall-clock against both logged rows, with intervals; no non-inferiority claim.

(k) *Readout.* `scripts/explore_e14_small.py summarize` → `results/explore/e14_small/summary.{md,json}`, keys `e14/...`
and `e9/e14a/...` (and `e9/e14g/...`, `e9/e14ag/...`) in `paper/generated/e14_small.tex`; appendix `app:e14`. An
"execution as run" entry follows the run.

### Deviations 2026-10-07 ~17:30 UTC: "E14: pause and resume on API overload" (exploratory; amends E14 (g) before any E14 live episode)

The direct Gemini API has answered HTTP 503 "high demand" since about 16:45 UTC (E6f's probe log,
`results/explore/e6f_modes/api_probe_log.txt`). Nothing of E14 has run except code; no E14 model has been trained. (a)
*Order.* Training and the offline levels (L1, L2, calibration, serving) run first; the live configurations of (g)
start only once the API answers cleanly. (b) *Probe.* One round = six one-word Gemini 3.5 Flash calls on the direct
API through the harness client (its own retries included), logged with time, count of successes and latencies
(`scripts/explore_e14_small.py probe`, appended to `results/explore/e14_small/api_probe_log.txt`). A live
configuration (smoke included) starts only after two consecutive rounds with 6 of 6 calls answered. (c) *Pause.* While
a live configuration runs, a watchdog pauses it (stops the runner, keeps every record) when an episode ends in an
infrastructure error that names HTTP 503 or overload; probes then run every 5 minutes, and the configuration resumes
unchanged after two consecutive clean rounds (the runner skips completed tasks). (d) *Re-run.* After the configuration
has covered every task, the episodes whose latest record is an infrastructure error are re-run once, after two clean
probe rounds; nothing is re-run after that, and tasks still errored are left out and counted. This replaces the
"passes 2-3" sentence of (g). (e) *Never another route:* no OpenRouter fallback; if the API does not recover in time
for the deadline of (i), the live rows are reported as not run (or partial, with the tasks covered).

### Deviations 2026-10-07 ~17:45 UTC: "E14: the optional Clef row is dropped" (exploratory; amends E14 (h) before any E14 evaluation step)

At the coordinating session's request (scope change, 2026-10-07), E14 (h) is withdrawn: E14 does not score Cloudflare
`clef-flash` or `clef` on any item. Cloudflare's released Clef models belong to E15 ("more decision-model baselines"),
which registers them in its own entry. Nothing else in E14 changes: arms A, G and AG on Qwen3.5-0.8B with a fresh
Clef-style head. No E14 model has answered any evaluation item yet (the host is installing; no training step has run).

### Deviations 2026-10-07 ~17:47 UTC: "E15: more decision-model baselines" (exploratory, descriptive; registered before any E15 model call)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E15)" (post-Stage-1;
marked with the dagger in the paper). It has no test, trains nothing, revises no registered or reported result and
changes no profile, threshold, key or record of any other row. Nothing has run for E15: no E15 model has been
downloaded to a GPU or asked a question, and no E15 episode exists. Read before registering, at $0: the public model
cards, repositories and image tags named below (fetched 2026-10-07), the OpenAI Decisions API guide
(https://developers.openai.com/api/docs/guides/decisions), and one probe call the coordinating session made to that API
(one 128-token request, about $0.00001, not an E15 record).

(a) *Why.* The author asked why other decision models are missing as baselines. RQ1 (Figure `fig:frontier`, panel (b)
`fig_levels`) rests on Jev, Intern-Decision-4B, StartLux-Decision 0.8B-27B and AgentDecision-4B (the frozen
JevAgentRL-4B of the records), against Gemini deciders and controls. **E15 adds every public decision model we can run,
and OpenAI's hosted decision API, at the three levels, and asks whether RQ1's reading ("fixed-question scores do not
rank decision models for agents") holds on the larger set.**

(b) *Candidates* (revisions and image digests are recorded at download; a candidate that cannot be downloaded, does not
load on the host or does not answer the wire is dropped and reported with the reason).
1. **OpenAI Decisions API** (`POST https://api.openai.com/v1/decisions`, model `gpt-6-luna`, public beta; proprietary
   API; the key from this container's `.env`, never on AWS). A new backend `openai_decisions` maps the Jev wire
   questions to the API's schema: `choice` with `choices: [{value, description}]` (description omitted when the option
   has none, unless the dev-prefix check of (c) 1 shows the API needs it), `predicate` for a yes / no question and
   `score` with `levels: [{label, description}]`; each question carries `name` = q0, q1, ... and `instructions`;
   answers are matched by `name`. All questions about one state go in one request (as Jev), unless the dev-prefix check
   shows that a multi-question request fails; then one request per question, recorded. Price: $0.10 per 1M input
   tokens, no output, cache-read or cache-write charge (the guide; default region; added to `configs/prices/models.yaml`
   with its source). **A refusal** (an answer of type `refusal`) counts as no answer: never correct, its confidence 0,
   and counted per row; in the live REPLACE arm a refused tool choice falls through to the agent LLM (a one-line harness
   rule that applies only to answers flagged as refusals; no logged backend produces them, so no logged row changes).
   GPT-6-Luna, GPT-6.1-Sol and GPT-6-Astra as Responses-API multiple-choice deciders are **not** E15 rows (the author:
   not decision models); E9's 299-item check rows stay as they are.
2. **OpenJev** (razorback16/openjev, Apache-2.0): image `razorback16/openjev:0.5.0`, model `openjev-latest`
   (DiffusionGemma-26B-A4B, NVFP4 weights `nvidia/diffusiongemma-26B-A4B-it-NVFP4`, Apache-2.0), the image's defaults.
   The image is tested on Blackwell; if the NVFP4 weights do not load or answer on the L40S (Ada), the image's
   documented `OPENJEV_MODEL` switch serves the same model's FP8 weights (`RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic`),
   labelled so; if neither runs, the row is dropped. The free hosted Codiv endpoint needs a sign-up and is not used.
3. **OpenJev (HF)** (`openjev/openjev`, revision `ac97900f`, weights CC BY-NC 4.0: research use; helper Apache-2.0):
   its documented FP8 format `openjev/openjev-FP8` (the 16-bit weights, about 54 GB, do not fit 48 GB) served by vLLM
   0.29.0 behind the model card's helper `helper/shim.py` with the card's fixed readout settings (`READOUT_T=0.85`,
   `READOUT_NOUL_T=1.829074`, `READOUT_NOUL_BIAS=0`, `READOUT_TARGETED=1`, `READOUT_INSTR_STYLE=pyrepr`,
   `SHIM_STAGGER=1`; vLLM flags as the card's quick start except `--quantization fp8`, which the FP8 checkpoint carries).
4. **Cloudflare Clef** (Apache-2.0): `Cloudflare/clef-flash` (9B, Qwen3.5-9B backbone, revision `fde727a2`) in bf16,
   answered by the model card's `systemone()` (`joint_schema_model.py`, the card's loader and defaults: one request =
   one record, `max_length` 16,384) behind a thin HTTP wrapper of ours on the host. `Cloudflare/clef` (27B, revision
   `ed3eed33`) only if, after every other candidate's L1 and L2, at least 3 GPU-hours remain beyond the live reserve of
   (e): the backbone loaded in 8-bit (bitsandbytes LLM.int8 through the card loader's `from_pretrained` keyword
   arguments; the joint head in bf16), because bf16 does not fit 48 GB; else not run, reported.
5. **Community Jev-likes** (`docs/TRAINING_RECIPES.md` §1), each with downloadable weights and a licence that permits
   research evaluation; one size per family, the one named in our notes:
   - JevK5 v0.2 (`alibiserikbay/JevK5`, Qwen3.5-4B, Apache-2.0) through `razorback16/openjev-jevk5:0.5.1` (it builds
     prompts and combines passes with JevK5's own `jevk5` package 0.2.2; the OpenJev README reports the same top answer
     as JevK5's own run on all 231 public JevBench items);
   - CLM v0.1 (`Contrastive-LM/CLM-v0.1-8B` heads over Qwen3-8B-FP8, Apache-2.0) through `razorback16/openjev-clm:0.5.1`;
   - Laya (`convaiinnovations/laya-typed-decisions`, 421M encoder, Apache-2.0) through `razorback16/openjev-laya:0.5.1`
     (it reads 1,024 tokens; a longer state is cut, as served);
   - Verdict 1.4 (`heman10x/rlcd-modernbert-151m`, 151M encoder, Apache-2.0) through `razorback16/openjev-verdict:0.5.1`
     (512 tokens; at most 24 options per read, so the harness's own knockout answers wider Choices, as it does for
     every backend with a smaller read);
   - Imajev-4B (`mohit67890/imajev-4b`, a LoRA on Qwen3.5-4B, Apache-2.0) through its own server (github
     mohit67890/imajev at the commit recorded on download) with the model card's Linux / CUDA command
     (`--backend torch --rotations 4 --calibration calibration-rot4.json`);
   - decider-4b v2.1 (`Mapika/decider-4b`, Qwen3.5-4B, Apache-2.0) through its own server (`uvicorn decider.serve:app`,
     the repository's `decider/` package, its defaults);
   - Bosun 3.1 1.7B (`Hanno-Labs/bosun-v3.1-1.7b`, Qwen3-1.7B LoRA, Apache-2.0) through the card's `model.predict`
     (one call per question; a Choice's options as candidates `{id, label, description}`) behind a thin HTTP wrapper of
     ours.
   Every model is asked through the harness's `jev` backend (or `openai_decisions`), so every call goes through the
   Ledger; Jev wire requests carry no gold label (`Question.meta` stays off the wire). The other sizes of these families
   (JevK5 2B / 9B / Lite, decider 0.8B / 2B / 35B, Imajev 2B / 9B, Bosun 0.6B and the June Bosun releases) are not run.

(c) *Levels and items* (each run once; no setting is chosen on C or on an evaluation task; vendors' documented defaults).
1. **Wire check on E1's 299-item development prefix** (`configs/eval/e1_dev_prefix_ids.json`, never in C8) for every
   model before its L1 run: errors, refusals, accuracy, latency; reported in the summary, never drawn. For the OpenAI
   Decisions API it also settles the two format points of (b) 1 (multi-question requests; choices without a
   description). It is the only place where an adaptation may be chosen.
2. **L1 static**: E1's 3,991 decisions (BFCL v4 tool choice 1,114, arguments 1,949, When2Call 928) with E1's config
   (pools, C8 exclusions, batching by state, 4 workers) and a new answer cache (`configs/explore/e15_baselines/e1_e15.yaml`).
   Accuracy pooled and per family (an error or a refusal is wrong); cost per decision by **E9's rule**: the API at its
   logged list price, open weights at $2.24208 per GPU-hour times the logged serving time per decision (request latency
   / questions in the request); E9's sensitivity variants (zero; the provisioned per-token tariff, using the `sl4b`
   window's rate for every new open-weights row). Keys `e9/<slug>/{acc,lg,usdk,...}` beside E9's drawn set; E9's
   frontier and keys stay unchanged; the recomputed frontier gets its own keys. The BFCL tool-choice accuracy as
   Figure 4b's static column (`e15/<slug>/e1_acc`, computed as `explore/startlux_live/<row>/e1_acc`). Paired item
   bootstrap (E9's `paired_gap`, B = 2,000, seed 0) against Jev and AgentDecision-4B.
3. **L2 common states C** (`results/stage2/s23/`: 5,314 tool-choice occurrences, 3,939 distinct questions, 200 tasks;
   `stage2_corpus.py replay`, one question per request, a new cache): task-mean accuracy over all states, after an
   off-plan step and on the ask-vs-act stratum (answer key `ask_user`), and ECE, exactly as `stage2_ladder` / E5.
   Paired task bootstrap (B = 2,000, seed 0) against Jev and AgentDecision-4B.
4. **L3 live**: REPLACE at the tool choice (the model picks every tool), Gemini 3.5 Flash on the direct Gemini API
   (temperature 0, `reasoning_effort: minimal`), the first 200 fresh Stage-2 tasks (seeds 300000-300199; evaluation
   seeds, `JEV_EVAL_RUNS=1`), 4 workers, a new System-1 cache: the protocol and tasks of `live_success` (`jev_ts_replace_f`,
   the E2 ladder, E12). **Rows: the OpenAI Decisions API, whatever its L2 rank (the hosted rival to Jev), plus the two
   other new models with the highest L2 accuracy** (ties: higher L1 accuracy, then lower E9-rule cost). The OpenAI row
   runs from this container (its key never goes to AWS); the two GPU rows on the E15 host (the Gemini key from SSM into
   the live process only). Reported: success (latest record per task, errored tasks left out and counted), own-state
   accuracy, steps, calls, API $ per episode, wall-clock, refusals; paired Newcombe intervals against Jev's logged row.
   Smoke before each live configuration: 2 JevGym development tasks (seeds 906-907, never reported); gate: the smoke's
   $ per episode x 200 x 1.5 plus the spend so far must fit the Gemini cap of (e).
5. **RQ1 ranking**: Kendall's tau-b between the static (BFCL tool-choice `e1_acc`), common (`corpus_acc`) and live
   (`live_success`) columns over every row that has all three, old (the eight rows of `explore/startlux_live`) plus new,
   computed as `explore/startlux_live/kendall/{e1,corpus}_vs_live_all_rows`; also static vs common over every row with
   L1 and L2. New keys `e15/kendall/...`; the old keys stay.

(d) *Pause and resume on API overload* (as E14's amendment of 17:30 UTC). Offline levels run first. One probe round =
six one-word Gemini 3.5 Flash calls on the direct API through the harness client, logged with time, successes and
latencies (`results/explore/e15_baselines/api_probe_log.txt`). A live configuration (smoke included) starts only after
two consecutive rounds with 6 of 6 answered. While it runs, a watchdog pauses it (stops the runner, keeps every record)
when an episode ends in an infrastructure error that names HTTP 503 or overload; probes then run every 5 minutes and it
resumes unchanged after two consecutive clean rounds (the runner skips completed tasks). After every task is covered,
episodes whose latest record is an infrastructure error are re-run once after two clean rounds; nothing is re-run after
that, and tasks still errored are left out and counted. **Never another route**: no OpenRouter fallback; if the API does
not recover by the deadline of (e), the live rows are reported as not run or partial.

(e) *Execution and caps.* One new GPU host launched by this session with `scripts/aws/e15_baselines.sh`
(Name=jev-e15-baselines; image `<aws-resource-id>`, E3's clone; g6e.2xlarge with one L40S 48 GB, else g6e.4xlarge;
a 400 GB root volume), SSM only; downloaded code and weights run there only, each in its own directory or container,
never in the coordinating container; weights deleted after each model. Order on the GPU: Clef-Flash, OpenJev, JevK5,
decider-4b, Imajev-4B, Bosun, CLM, Laya, Verdict, OpenJev (HF), then Clef 27B under (b) 4; live rows after every L2.
**Caps: OpenAI at most $40** (wire checks, L1, L2, live System-1 calls); **direct Gemini at most $35** (live only; a
watchdog on the logged spend); **OpenRouter $0**; **Jev $0** (Jev's logged answers only; no Jev call); **GPU at most 12
GPU-hours** (a power-off guard 710 minutes after launch, a 30-minute idle stop, about 2.5 GPU-hours reserved for the
two GPU live rows; the host stopped when done). `OPENROUTER_API_KEY_JEV` and the OpenAI key never go to AWS. Records to
`s3://<bucket>/e15_baselines/`. Deadline for E15 runs: 2026-10-09 18:00 UTC; a stage cut by a cap
or the deadline is reported as such.

(f) *Reading rules* (descriptive; reported whatever the outcome).
- Per model: L1 accuracy and E9-rule cost, its place against E9's drawn set (on or off the recomputed frontier); L2
  accuracy, after-off-plan, ask-vs-act and ECE, with the paired difference to Jev; L3 success with the Newcombe interval
  to Jev's logged row where run.
- RQ1's reading **holds on the larger set** iff, over every row with all three levels, tau-b(static, live) is below
  tau-b(common, live) and below 0.5; otherwise it is reported as not holding. The old rows' keys are unchanged.
- Public BFCL / When2Call items may be in a vendor's training data (as for StartLux, A7.18 (c)); C never is. A model
  served through another project's image (the four `openjev-*` community images) is labelled so.
- Refusals (OpenAI) and errors are counted per level; a level with more than 5 % errors is flagged.

(g) *Readout.* `scripts/explore_e15_baselines.py summarize` -> `results/explore/e15_baselines/summary.{md,json}` and
`paper/generated/e15_baselines.tex` (`\res` keys only: `e9/<slug>/...`, `e15/<slug>/...`, `e15/kendall/...`); appendix
`app:e15`. An "execution as run" entry follows the run.

### Deviations 2026-10-07 ~18:45 UTC: "E6f: execution as run" (exploratory, descriptive; written after the run, before any paper text used it)

Execution record of exploratory row E6f; it changes no rule, threshold, arm, task or readout. (a) *Code and times.* The
E6f entry was committed at 15:12:52 UTC (6768b82), before the first E6f episode (smoke start 15:13 UTC). The harness
code, configs, readout and tests it names were committed before it, in the snapshots 1e74ff6 (14:41 UTC) and e055c5c
(15:02 UTC). Two commits of other rows changed harness files while E6f ran: 2f2a98c (16:28 UTC: the off-by-default
key `tool_params.extra.full_tools` of the delegated argument call) and 4d2e61d (18:23 UTC: a refusal flag that only
the OpenAI decision backend sets, and an optional read width of the Jev wire backend, unset by default). No E6f arm
sets either key or uses that backend, and no E6f arm makes a delegated call, so every E6f episode ran the E6f code as
registered. (b) *Smoke* (development, never a result): 15:13–15:20 UTC, 16 episodes (JevGym seeds 900–901, five arms;
two ALFWorld train games, three arms), no infrastructure error and no empty reply. The direct API answered without
error but slowly (median 10.0 s per LLM call, p90 20 s), which is not the registered pause condition. Projection
(`results/explore/e6f_modes/projection.json`): JevGym 200 + ALFWorld 140 projected $93.93 of Gemini (over the $60
cap); JevGym 100 + ALFWorld 140 projected $51.41 of Gemini and $0.14 of Jev. So, as registered, **JevGym ran its first
100 tasks (seeds 301600–301699)** and ALFWorld all 140 games. (c) *JevGym evaluation:* one runner invocation of 500
episodes (five arms × 100 tasks, task-major, seed 732, 12 workers), 15:24:55–16:29:05 UTC, with no infrastructure
error and no empty reply; passes 2–3 had nothing to re-run. (d) *ALFWorld evaluation* (task-major, seed 732, 8
workers, `TMPDIR` on `/dev/shm`), from 15:24:56 UTC in invocations over the first 20, 40, ... games, at the same time
as the JevGym run. From about 16:39 UTC the direct Gemini API answered HTTP 503 "This model is currently experiencing
high demand"; 84 episodes ended in that error (27 `llm_only`, 29 `jev_action_trigger`, 28 `action_verify_all`; last
calls between 16:39 and 17:03 UTC). The row was **paused at 17:12:22 UTC** (registered entry (f); routes were never
mixed). Episodes in flight at the pause were not logged, so their few calls are missing from the spend. Health
probes (six one-word Flash calls on the same client; `results/explore/e6f_modes/api_probe_log.txt`): 17:15 UTC 5/6
(one 503), then 17:23 and 17:29 UTC 6/6, and the driver **resumed at 17:29:55 UTC** after two clean rounds. The
remaining invocations ran to 18:21 UTC. The registered error pass re-ran the 84 errored episodes once (18:21:47–18:31:57
UTC) without error. Every game is complete in all three arms, and no game left a paired comparison. (e) *Run
files* (gzipped after the run; sha256 of the raw files): `runs/e6f_modes/` `llm_only` `26cb13faed194dc6`, `ts_trigger`
`58578e8a508f2672`, `ts_verify_all` `911b38ca0ec8944d`, `tp_trigger` `708bb3c5e3f5d9aa`, `tp_verify_all`
`3b9a92971bc08137`; `runs/e6f_alfworld_eval/` `llm_only` `f7dd1fe8c5249481`, `jev_action_trigger` `86d958db8fff08ff`,
`action_verify_all` `a7f9d912c3878f0b`. Kept locally (gitignored); no key in any record or status file. (f) *Spend:*
**Gemini $43.55 of the $60 cap and Jev $0.125 of the $2 cap**, every E6f record included (smokes $0.645 Gemini and
$0.0017 Jev; JevGym evaluation $27.19 and $0.073; ALFWorld evaluation, the 84 errored attempts included, $15.72 and
$0.050). The probe calls are not in the run records. The watchdog never fired. (g) *Wall time.* The direct API was
slower than in E6b throughout (the JevGym `llm_only` took 48.8 s per episode here against 13.0 s in E6b), and 453 of
the 14,655 logged LLM calls needed a retry. Wall times are compared only within each run. (h) *Readout:*
`results/explore/e6f_modes/summary.{md,json}`, keys `e6f/...` in `paper/generated/e6f.tex`. `scripts/module_map.py`
now draws the three cells as exploratory tiles (`mm/xcell/flash/{tool_select,tool_params,action}/trigger/...`), and
`mm/count/flash/undefined` is 0.

### Deviations 2026-10-07 19:54 UTC: "E4r: execution as run" (exploratory, descriptive; written after the run, before any paper text used it)

(a) *As run.*
- Host `<instance-id>` (m7i.8xlarge), code at `50e59d0` (the minimum-spend commit). The stages `rearm update smoke
  gate eval final` were sent at 15:34 UTC.
- The key check found an inference key with a $150 limit and $0 of usage.
- Smoke: `scikit-learn__scikit-learn-13496` × 3 arms, 15:34–15:39 UTC, 0 errors, $0.0196.
- Gate: all five conditions held. Decision "full", k = 200, projection $25.29 against the $40 cap.
- Agent runs: 15:39–19:17 UTC, 24 workers, 600 jobs. There were 0 infrastructure errors, so nothing was re-run. One
  instance hit its limits in each of `always_deep` and `jevagent`.
- Local SWE-bench evaluation: 19:17–19:50 UTC, no unevaluated instance. The leak check found the key in 0 files.
- The host powered itself off at 19:50 UTC, after the S3 sync.
- **Spend: $17.61 logged (smoke $0.02, evaluation $17.59), equal to the key's usage.** The watchdog never fired.
- Nothing was tuned or re-run.

(b) *Readout* (`scripts/explore_e4_swe.py --exp e4r readout` → `results/e4r_swe/{summary.md, summary.json,
per_instance.csv, NOTES.md}`, keys `e4r/...` in `paper/generated/e4r_swe.tex`), as registered, whatever the outcome.
- **Pooled, 300 distinct held-out instances** (E4's 100 + E4r's 200):
  - `always_deep` 77.3 %;
  - `jevagent` 76.3 %: Δ −1.0 points, 9 wins / 12 losses, paired bootstrap [−4.0, +2.0], Tango [−4.3, +2.2]. The
    3-point non-inferiority test gives p = 0.108, so non-inferiority is not shown. Cost 10.5 % lower [0.4, 19.5];
    mean agent time 4.7 % lower [−2.4, +11.1] (436 vs 457 s);
  - `always_fast` 75.7 % (Δ −1.7 points), 70.7 % lower cost, 27.9 % less time.
- **The 200 new instances alone:**
  - `always_deep` 75.5 %;
  - `jevagent` 74.0 % (Δ −1.5 points; 6 / 9), cost 4.8 % lower [−8.5, +16.9], time 2.0 % lower;
  - `always_fast` 74.5 %, 69.1 % lower cost.
- Fast-share on the new 200: 21.7 % (E4: 23.5 %).
- E4's own 100 instances are as reported.
- Descriptive; outside every Holm table.

### Deviations 2026-10-07 ~20:35 UTC: "E14: serving-engine fix" (exploratory; implementation only; before any re-timing and before any E14 live episode)

(a) *Why.* Requested by the coordinating session after arm A's first readout (dev, C, L1). By E9's cost rule arm A's
E1 point costs about $0.12 per 1,000 decisions, above AgentDecision-4B's $0.093, although a 0.8B model scores all
options of a request in one forward pass. Three properties of the first engine (registration (e)), not of the model,
explain it: (1) one request per forward pass, so under E1's 4 workers a request waits for up to three others and its
logged latency, which E9's rule counts as GPU time, is about four times its compute; (2) a cold start: the first
requests waited about 40 s while GPU kernels compiled (7 of the 3,991 decisions above 5 s); (3) the gated-convolution
layers ran the reference PyTorch path (the `causal_conv1d` kernel was not installed).
(b) *The fix (engine v2), no change to the model.* Same merged checkpoint, served temperatures, record encoding and
answer format. (1) Requests that arrive while a forward pass runs are queued and scored together in the next padded
pass (at most 16 records and 32,768 padded tokens; a lone request is never delayed). (2) Warm-up forward passes at load,
before the server reports healthy. (3) The `causal_conv1d` kernel, only if it installs from a published wheel within 20
minutes; else the reference path stays and this is reported.
(c) *Check, development items only, before any re-timing.* SFT v1's dev records (2,065 records, 2,377 decisions) and E1's
299-item development prefix (`configs/eval/e1_dev_prefix_ids.json`, never part of E1's 3,991), answered by engine v1
and engine v2 on the E14 host: v2 must return the same argmax on every decision except exact near-ties (top-two gap
below 1e-3), with probabilities within 1e-3; the agreement is reported either way. Nothing is tuned on E1's 3,991
decisions or on C. If the check fails, v2 is not used and only the original timings are reported.
(d) *Re-timing, reported beside the original* (each engine and its settings named in the readout): E1's 3,991 decisions
once more through v2 (a new answer cache; its answers compared with v1's, which stay the L1 accuracy of record), cost by
E9's rule and by wall-clock GPU time (wall time / 3,991 at E9's tariff, as E12c (h)); the latency bench (c = 1 and
c = 4) with v2 and, as a drift check, AgentDecision-4B at c = 1 again. The live rows of (g) use v2 if the check passed,
else v1. GPU: within E14's 10 GPU-hours.

### Deviations 2026-10-07 ~22:30 UTC: "E12, E12b, E12c: execution as run" (exploratory, descriptive; written after the runs, before any paper text used them)

(a) *Timing labels* (commit times are authoritative). E12's scripts and configs first entered the branch in the
coordinating session's snapshot `427e7e2` (13:43:46 UTC). E12's registration is `83a7d25` (13:52:43). The training cap
and E12b's registration are `ef99033` (14:32:42), before E12's training (14:38) and E12b's harvest (14:39). E12c's
registration is `653e98a` (15:06:01), before its host was launched (15:12:44). The timing-host amendment is `20b3ca8`
(15:41:33), before any vLLM step. No training or evaluation step of any of the three ran before its entry.

(b) *E12 as run* (host `<instance-id>`, g5.8xlarge, us-east-1d). The AWS login expired a minute after the launch,
and the host sat idle from 13:55 to 14:33. The first model load read the new volume lazily from its snapshot (about 4
MB/s); a parallel read of the files (read-only) sped it up. `nogc_3072` ran out of memory, and `gc_3072` trained in 146
minutes. Temperatures, the offline files, E1 for the frozen and the new checkpoint, corpus C, the live smoke and the
live run followed as registered. The live smoke's gate said "run all 200"; one episode with an infrastructure error was
re-run in pass 2. On the host the serving-path check command failed, because the frozen model's logged E1 records are
not on the host. It ran off the host with the same function: 3,983 of 3,991 answers agree (99.8 %, rule ≥ 99 %). Gemini
spend was $2.93 (cap $30).

(c) *E12b as run.* The $0 harvest ran on the CPU beside E12's training (28 minutes; 600 tasks, 540 for training and 60
for dev; rule kinds above 204, below 184, tiered 212), and the rows matched the registered quotas. Training took 74
minutes (`gc_3072`). On the same host the frozen checkpoint reproduced E7d's recorded held-out numbers exactly.

(d) *E12c as run* (host `<instance-id>`, g5.8xlarge, us-east-1c; no g6e capacity in any zone). StartLux-4B with
a fresh LoRA trained in our stack. The smoke found `nogc_3072` out of memory and `gc_3072` at 47.8 s per step
(projected 341 of 360 minutes), so the StartLux base was kept and the Qwen3-8B fallback was not used. Training took
270 minutes for both epochs. Selection first failed: the trainer's epoch checkpoints hold no tokenizer files. The final
adapter's tokenizer files (the base's) were copied into both checkpoint directories on the host, and the chain was
sent again from `select`; the driver now makes this copy. The first resend (20:11) fetched E12's driver, because the
two drivers had shared one S3 key. It ran E12's `temps` on the E12c host (and `select`, unknown to that driver), which
failed within a second, changed nothing and wrote two status lines into E12's status file. The E12c driver got its own
key (`d7f0626`) and was resent at 20:12. Selection chose epoch 1: development-prefix accuracy 89.6 % against 87.9 %
for epoch 2, and v1 dev 99.5 % for both. Temperatures, the offline files, E1, corpus C, the live smoke (gate "run all
200") and the live run followed, with no errored episode. Gemini spend was $2.92 (cap $15).

(e) *Same-engine timing as run* (on the E12 host, started again at 21:40 for this stage only). The first attempt failed
at engine start: vLLM 0.31.0's FlashInfer kernels compile with nvcc, which the host lacks. The engine then used
FlashAttention and the plain sampler (`9c85635`). At the coordinating session's request, and before any timing ran, the
script also got a second pass with one request in flight, per-request p50 / p90 latency and each model's weight size.
The registered pass, four requests in flight, is the primary. All four models were timed. Answers agree with each
model's served answers on 99.8 % (AgentDecision-4B), 99.6 % (E12) and 99.8 % (E12c). For StartLux-4B the figure is
96.3 %, against its vendor-served answers, whose prompt differs. StartLux-4B's checkpoint is a multimodal one
(`Qwen3_5ForConditionalGeneration`). The merged E12c checkpoint is text only (`Qwen3_5ForCausalLM`).

(f) *Caps and hosts.* GPU: the E12 host ran 8.4 hours (cap 12), the E12c host 7.1 hours (cap 10). Gemini: $5.85 in
all. No OpenRouter or Jev call. Downloads: the two public datasets, E1's pinned files, and the vLLM and peft packages.
**Both hosts are stopped.**

(g) *Readout* (`scripts/explore_e12_train.py score`, `scripts/explore_e9_frontier.py summarize`; keys `e12/...`,
`e12b/...`, `e12c/...`, `e9/e12/...`, `e9/e12c/...`). Reported whatever the outcome.
- **E12.** (i) E1 85.1 % against 78.8 % (+6.3 points, paired bootstrap [+5.1, +7.5]; When2Call 68.5 against 54.0 %).
  (ii) Common states 99.1 % against 99.2 % (-0.1 [-0.3, +0.1]): non-inferior at 1 point. After an off-plan step,
  91.7 % against 94.7 % (-3.1 [-7.4, -0.1]). (iii) Not on E9's frontier, which is unchanged: Jev is 1.3 points more
  accurate [0.3, 2.4] and cheaper. The raw A10G cost was $0.140 per 1,000, $0.091 host-adjusted. (iv) Live 100 % (200
  of 200), as the frozen model.
- **E12b.** Zero-shot on held-out tier states: cancel / ship 91.0 % (frozen 76.5 %, E7d's tier-trained model 95.3 %),
  other tool choices 100 %, arguments 98.7 %.
- **E12c.** (i) E1 85.5 % against StartLux-4B's 83.1 % (+2.3 [+1.1, +3.6]): beats StartLux-4B. (ii) Jev -0.9
  [-2.1, +0.2]. (iii) Common states 99.6 % against AgentDecision-4B's 99.2 % (+0.4 [+0.0, +1.0]): non-inferior
  (StartLux-4B itself 89.1 %). (iv) Live 100 % (StartLux-4B 60 %, Jev 83 %). Not on the frontier: $0.295 per 1,000 raw,
  $0.193 host-adjusted.
- **Same engine** ($ per 1,000, wall clock, four in flight): AgentDecision-4B 0.0147, E12 0.0148, E12c 0.0246,
  StartLux-4B 0.0248; Jev's API price is 0.0163.

### Deviations 2026-10-08 ~00:30 UTC: "E14: execution as run" (exploratory, descriptive; written after the run; A's and G's rows were already in the paper at the coordinating session's request, AG's numbers enter with this entry)

(a) *Times and order.* The registration was committed at 17:12:31 UTC (`0d9683d`) and the pause rule at 17:28:09
(`92416c7`), both before the host was launched (17:32:28 UTC; g5.2xlarge <instance-id>, us-east-1d). The code
that converts rows and serves the model followed at 17:38:35 (`92d569e`), before the first training stage (17:45). The
optional Clef row was withdrawn at 17:45:28 (`2355e45`), before the first optimizer step (after 17:51) and before any
evaluation step. Arm A trained 17:45-20:01 UTC. Its development records, the common states C, E1 and the
calibration-state answers were scored 20:02-20:19. G's mix was built at 17:45 beside A's training stage; G trained
20:24-20:43 and was scored 20:43-20:59. The latency session ran 21:02-21:32. The serving-engine entry (`bf86d54`,
20:23:36) came after arm A's L1 readout and before any re-timing; its development-item check ran 21:33-21:40 and
failed (b). Arm A's fitted thresholds were committed in the E14 profile at 20:37 (`28be336`), before the first live
episode (21:45). The live rows ran 21:45-22:12, each after clean probes. AG trained 22:13-23:36 and was scored
23:40-23:59; the host's last stage then powered it off (about 00:02 UTC on 2026-10-08). The coordinating session's AWS
login expired at about 22:52 UTC while AG trained; the queued stages ran on, and the records were read from S3 after
the login came back, after midnight. A's and G's rows entered the appendix (`02dafe4`, 22:24) and the main text
(`a007753`, 22:48) before this entry, at the coordinating session's request.

(b) *As run, against the registration.*
- No memory fallback was needed (training peak 6.3 GB). No training record was cut at 4,096 tokens.
- Records: A 14,787. G 12,350 built, of which it trained on 12,078; `split_holdout` kept 272 for G's temperatures. AG
  27,137: A's 14,787 and all 12,350 of G's. AG fits its temperatures on A's development split, so nothing of G's mix
  was held out for it.
- A's second epoch ran under the rule of (d) (projection 5.8 of 10 GPU-hours) and was not kept: development accuracy
  99.66 % against epoch 1's 99.71 %.
- G reproduced Unsloth's gain by the registered criterion (mean 69.6 % against the 69.3 % threshold; the fresh head
  20.8 %), so it was used, including its live row.
- The engine-v2 check failed. On SFT v1's development records v2 kept every argmax but moved probabilities by up to
  0.044 (tolerance 0.001), and it changed 1 of E1's 299 development-prefix answers. No `causal_conv1d` wheel was
  published, so the kernel was not installed. As the entry says, v2 was not used: every timing and every live row is
  engine v1, and E1 was not re-timed.
- The live order was kept. The API probes were clean before every configuration (12 rounds, each 6 of 6; the log is
  `results/explore/e14_small/records/api_probe_log.txt`, not the path the pause rule named). No episode errored, and
  nothing was paused or re-run.
- AG ran under (d). At its start 4.67 GPU-hours were used, and its projection (A's and G's measured epochs plus its
  evaluation, about 1.8 hours) left the total near 6.5 of 10. It trained for A's selected count, one epoch
  (71 minutes); its calibrated development accuracy was 99.66 %. AG was not timed (A and G, the same architecture on
  the same engine, timed alike) and has no live row (the live rows of (g) name A and G only).
- Optional item (h) was withdrawn and not run.

(c) *Caps and host.* GPU: about 6.5 host hours of the 10-hour cap (g5.2xlarge, launch 17:32:28 to the host's own
power-off about 00:02 UTC; the driver counted 6.45 hours at its last stage and powered off two minutes later). Gemini
(direct API): $5.82 logged in the live smokes and the three live rows, plus $0.002 of probes ($40 cap). No OpenRouter
or Jev call. **The host is stopped.**

(d) *Readout* (`scripts/explore_e14_small.py summarize` -> `results/explore/e14_small/summary.{md,json}`; keys
`e14/...`, `e9/e14a/...`, `e9/e14g/...` and `e9/e14ag/...` in `paper/generated/e14_small.tex`; appendix `app:e14`),
reported whatever the outcome.
- **A** (agent states). C 99.2 % against AgentDecision-4B's 99.2 % (+0.0 [-0.2, +0.2]); after an off-plan step 94.1 %
  against 94.7 % (-0.6 [-2.2, +0.7], inconclusive). E1 67.8 % (BFCL tool choice 80.5 %). REPLACE live 100 % (200 of
  200). In the final harness 100 % at $15.42 per 1,000 episodes with the GPU at $1.624 per hour: 64.2 % below Flash
  alone ($43.12), above the 4B harness ($10.15).
- **G** (generic mix). Unsloth's sets 69.6 %; C 23.9 %; E1 52.3 % (tool choice 81.8 %); REPLACE live 0 %. Its threshold
  fit accepts no level at either point.
- **AG** (both). C 99.1 % (-0.1 [-0.3, +0.1]); after an off-plan step 92.7 % (-2.0 [-4.0, -0.4], loses by E5's
  rule). E1 68.8 % (tool choice 84.8 %); Unsloth's sets 69.4 %. Tau-hat 0.99984 at the tool choice and within 1e-11 of
  1 at the arguments; coverage 90.0 % and 70.0 %.
- Serving (engine v1, one session, one A10G). Median per request: A 91 ms at c = 1 and 203 ms at c = 4, against the
  4B's 319 and 879 ms. Memory: 2.4 GB against 22.0 GB.
- Readings of (j). (i) did not hold for A: tau-hat within 1e-9 of 1 at both points, although the cascade takes
  80.0 % and 78.8 % of the calibration-state decisions. It did not hold for G (always defers) or AG (above) either.
  (ii) did not hold for A. C within 1 point and the c = 1 p50 both held, but E9's cost rule puts A at $0.12 per
  1,000 decisions against the 4B's $0.093 ($0.032 by wall clock). AG's E9-rule cost is also above the 4B's ($0.100),
  and AG was not timed. (iii) did not hold: A is ahead of G on E1 pooled, +15.5 points [+13.7, +17.2], and on C,
  +75.3 points [+73.9, +76.5]. On BFCL tool choice alone the two are level, -1.3 [-4.0, +1.3].

### Deviations 2026-10-08 ~03:40 UTC: "E15: execution as run" (exploratory, descriptive; written after the runs, before any paper text used them)

What ran, against the registration of 2026-10-07 17:47 UTC and the coordinating session's scope note of the same
evening (the OpenAI candidate is the Decisions API only; GPT-6-Luna, GPT-6.1-Sol and GPT-6-Astra as Responses-API
deciders were never run for E15). Records: `results/explore/e15_baselines/` (slim copies: per-decision records of the
wire check, L1 and C, slim per-task live records, host records); the raw run directories are on S3 under
`e15_baselines/`. Readout: `scripts/explore_e15_baselines.py summarize` -> `summary.{md,json}`,
`paper/generated/e15_baselines.tex`.

(a) *Host.* No g6e.2xlarge capacity in any of the six subnets; the driver's second type, a g6e.4xlarge (one L40S
48 GB), ran from 18:23 UTC on 2026-10-07 (<instance-id>, three chains; the Docker images and the vendors'
servers as registered). GPU time 8.8 host-hours (cap 12; the host was never stopped between chains, so host
hours = GPU hours). The OpenAI rows ran from the coordinating container (its key never left it).
(b) *Fixes before a model produced a recorded answer* (each committed before the re-run; nothing was chosen on an
answer's value): Clef-Flash first failed to load (its processor imports torchvision, absent from the host venv) and
then never passed the readiness check (our wrapper's request handler was rejected with HTTP 422: an unresolved
annotation); Imajev-4B and Bosun first failed to load their base models (a container had written into the shared
Hugging Face cache as root); OpenJev-27B (HF) first failed to start twice (vLLM refused the card's 256 sequences on
48 GB, then its FlashInfer sampler needed nvcc, which the image lacks). All four were re-queued and ran from scratch.
(c) *Vendor limits met as served* (limits, not choices): Imajev-4B's server takes at most 8 questions per request; its
first wire check and L1 sent up to 64 and 9 / 118 decisions came back as errors; the backend was then capped at 8
per request and those requests re-run in the final chain (the answer cache kept every other answer; the re-run left no error and lifted its L1 accuracy from 76.6 % with the 118 errors counted wrong to 78.9 %).
Verdict reads at most 24 options: 11 L1 decisions with wider Choices came back as errors (scored wrong); on C the
harness's knockout answered wider Choices (registration (b) 5). OpenJev-27B (HF) served with `--max-num-seqs 128`
(capacity) and vLLM's own sampler (the helper reads the option letters' log-probabilities, computed before sampling).
(d) *Variants.* OpenJev's NVFP4 weights loaded and answered on the L40S (vLLM's MARLIN NVFP4 path), so the FP8
switch of (b) 2 was not used. The OpenAI Decisions API's wire check (299 items, no error, one refusal) settled both
format points of (b) 1: several questions per request, and options without a description sent without the field.
Clef 27B (8-bit) ran under (b) 4 (3.5 GPU-hours remained beyond the live reserve when it started): its wire check and
L1 completed; its C run (about 1.6 s per request, about 2 GPU-hours) was stopped by the coordinating session after 15
answers so that the OpenJev-27B retry and the two GPU live rows fit the cap; those 15 answers were set aside unused.
The author then withdrew the 27B's remaining levels (relayed 2026-10-08 ~02:35 UTC): no C or live row for it; it
stays an L1-only row.
(e) *Live.* The OpenAI row ran from the coordinating container on 2026-10-07 18:28-18:45 UTC, after its own L1 and
L2, while the GPU models' offline levels were still running; it paused once on an overload error (18:33 UTC) and
resumed after two clean probe rounds; no task was left errored. The two GPU rows were selected by the registered rule
from every complete L2 row on the host (`box/live_selection.json`): OpenJev-27B (HF) (C 92.4 %) and OpenJev (91.1 %); Imajev-4B (76.9 %) was third and has no live row. Both GPU rows ran on the host (02:24-03:10 UTC, 200 tasks each, no task errored, no pause); the OpenJev-27B
weights were kept on the host between its offline and live stages instead of being deleted and fetched again. The coordinating container
restarted at about 01:00 UTC on 2026-10-08; the host chain was unaffected (its own process); no local live row was
running then.
(f) *Spend.* OpenAI $1.13 (wire check $0.01, L1 $0.11, C $0.74, the live row's System-1 calls $0.27; cap 40); direct
Gemini $10.34 (live rows only, every record of every attempt: $3.42 for the OpenAI row from the container, $6.91 for
the two GPU rows and their smokes on the host; $9.93 over the deciding records; cap 35); OpenRouter $0; Jev $0 (logged answers only).
(g) *Keys added after the registration, from logged rows only* (a reviewer's request relayed on 2026-10-07 ~22:00 UTC):
Kendall's tau-b and the live-success spread over the off-the-shelf rows (without AgentDecision-4B; and without any
trained model, which is the same set here), `e15/kendall/offtheshelf/...`, `e15/kendall/notrained/...`,
`e15/spread/...`; and the mean confidence beside the occurrence-level accuracy on C (`e15/<slug>/meanconf`,
`e15/<slug>/occacc`) to say in which direction a model's calibration error lies. Nothing else changed.
(h) *Host stop.* The host stopped itself one minute after its final stage (2026-10-08 03:12 UTC; instance state
`stopped`, checked from the coordinating container).

### Deviations 2026-10-08 ~04:20 UTC: "E16: the best open-weights decision model on all three levels" (exploratory, descriptive; registered before any E16 training step, model call or evaluation episode)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E16)" (post-Stage-1;
marked in the paper like the other post-Stage-1 rows). It has no test, revises no registered or reported result,
selects nothing outside its own development data and changes no profile, threshold, key or record of any other row.
AgentDecision-4B (the frozen JevAgentRL-4B checkpoint of the records), E12's and E12c's checkpoints stay frozen and
untouched. The models it trains are two NEW checkpoints: **"AgentDecision-4B (mixed+)"** (arm A, served as
`agentdecision-4b-mixedplus`) and **"StartLux-9B (mixed)"** (arm B, served as `startlux-9b-mixed`). Nothing has run for
E16: no model has been trained, no E16 model has answered any question, no host has been launched. Only the training
items were built and audited, locally, at $0 and without any model call (see (c)); the quotas, the audit rule, the mix
and the gates below were fixed in `scripts/explore_e16_best_open.py` before that build ran, and the build record is
committed with this entry (`results/explore/e16_best_open/build_info.json`, `audit_hits.json`).

(a) *Why.* The author's goal (2026-10-08): our trained model should be the best open-weights decision model on our own
benchmark at all three levels. Today AgentDecision-4B is first on the common states (L2, 99.2 %) and live (L3, 100 %),
but not on the fixed questions (L1, E1's 3,991 BFCL + When2Call decisions): E12 (the 4B continued on 7,000 public
tool-use items mixed 1 : 1 with 7,000 of its own states) reached 85.1 % and E12c (the recipe on StartLux-Decision-4B)
85.5 %, while StartLux-27B scores 87.7 % and Jev 86.4 %. The When2Call part is the gap (E12 68.5 %, StartLux-27B
81.2 %; on the BFCL tool choices 89.3 against 92.2 for StartLux-9B). E12's logged When2Call answers show the shape of
it: E12 over-predicts "ask the user" (gold "call" answered "ask" 110 of 329 times, gold "cannot" answered "ask" 64 of
333). **E16 asks whether more public data (arm A) and / or a larger open base (arm B) close the L1 gap to the best
open-weights decision model without losing L2 or L3.** E16 was designed after E12, E12c, E14 and E15 had been read.

(b) *Arms* (one seed each; every choice on development data only).
- **A, more public data.** E12's recipe, continued from AgentDecision-4B's frozen adapter (sha256-checked against
  `results/stage2/rl_eval_plan.json`), on twice E12's public items (c) mixed 1 : 1 with a replay of its own SFT v1
  states: `training/train_decision.py --init <frozen adapter>` on base Qwen3-4B (snapshot `1cfa9a72…`); LoRA r 32 /
  α 64 as frozen; soft cross-entropy + 0.5 Brier; option permutation; lr 5e-5; **2 epochs** (E12 ran 1), 64 rows per
  optimizer step; max length 4,096; trainer seed 0; one run. Memory setting `nogc_3072`, then `gc_3072`, then `gc_2048`,
  the next only after an out-of-memory error (same maths); capped at 330 minutes. The trainer saves the adapter at the
  end of each epoch.
- **B, a larger open base.** E12c's recipe (SFT only, from a fresh LoRA) on **StartLux-Decision-9B**
  (`startlux-models/StartLux-Decision-9B`, revision `342aad64…`, the Hub's current revision, read 2026-10-08; public, not
  gated; 19.3 GB; `Qwen3_5ForConditionalGeneration`, loaded text-only as the 4B was; **licence CC BY-NC 4.0**, the same
  family as StartLux-4B: the trained adapter and any merged weights are derivatives under the same non-commercial terms,
  research use only). The weights are downloaded on the host at that revision; their sha256 and the licence file are
  recorded. Recipe: LoRA r 32 / α 64 on all linear layers; soft cross-entropy + 0.5 Brier; option permutation; lr 1e-4,
  cosine, 3 % warm-up; 64 rows per optimizer step; max length 4,096; seed 0; the same mix as A; memory settings as A,
  chosen by a 4-step smoke (as E12c). Environment: `/opt/jev/v2venv` (A7.19 (f): transformers 5.18, flash-linear-
  attention, causal-conv1d) plus peft 0.21.2, as E12c. **Epochs: 2 when the smoke's step time projects two epochs
  within B's training budget (the minutes left before the host's power-off guard minus a 170-minute reserve for B's
  evaluation and the timing, at most 420 minutes), else 1 (recorded), else B is not run (recorded, with the reason).**
  bf16 LoRA needs a 48 GB GPU; on a 24 GB host (g5.8xlarge, A10G) B is not run (the trainer has no 4-bit path, and A
  alone takes most of the cap there), and the readout says so. No GRPO.
- **Checkpoint selection (both arms), development data only, as E12c (e):** the epoch-end adapters are scored offline
  (`eval_decisions`, accuracy, no temperatures) on E1's 299-item development prefix (`configs/eval/e1_dev_prefix_ids.json`,
  excluded from E1's 3,991 decisions; built by `explore_e12_train.py devprefix`) and the held-out JevGym dev rows
  (`data/train_v1/dev.jsonl.gz`, 2,377 rows). The adapter with the higher mean of the two accuracies is selected; a tie
  selects the later epoch; if only epoch 1 finishes within the cap, it is selected. E1's 3,991 evaluation items, the
  common states and the live tasks are never used for selection. Temperatures of the selected adapter:
  `training/calibrate.py --min-bucket 200` on E12's temperature set.

(c) *Data: E12's builders, audit and development rule, unchanged; only the quotas differ.* Items are built by E12's
code paths (`explore_e12_train`: E1's converter `bench/suites/static.py`, E12's label rule for the When2Call training
split, E12's ToolACE parser, E12's renderer), from the same two public sources at the same pinned revisions (When2Call
TRAIN `train/when2call_train_sft.jsonl` at E1's commit, CC-BY-4.0; ToolACE `Team-ACE/ToolACE` revision `6bda777c…`,
`data.json` sha256 `ba12c083…`, Apache-2.0). BFCL, When2Call TEST and E1's items are never used.
1. *Leakage audit, before any split or sample, E12's rule unchanged:* every unit's user request against E1's 3,991
   decisions, the 299-item development prefix and the full public test pools (6,687 texts), exact token sequence or a
   shared 8-gram (a shorter request matches only an identical one); a unit with any hit is dropped with all its items.
   As built: 19,909 units (127 duplicate source rows removed), **the same 207 units (485 items) dropped as in E12**
   (175 hit E1, 14 the development prefix, 207 the full pools, 2 of them exact); `same_hits_as_e12: true` in the record.
2. *Development units:* E12's rule (sha256("jev-explore-E12-dev" + newline + unit) mod 10 = 0), so E12's 1,872
   development units are E16's; they are never trained on. **The public development split (1,000 items), its three
   per-family files and the temperature set (1,000 v1 dev rows + the split) are E12's files, byte-identical
   (sha256-checked in the build and on the host).**
3. *Training quotas (fixed before the build; twice E12's 7,000, tilted to the When2Call-format families):* When2Call-
   format call 2,500 (ToolACE; E12 1,000), ask 3,000 and cannot 3,000 (When2Call-train; E12 1,000 each); BFCL-format
   tool choice 1,900 with a tool as gold (ToolACE; E12 1,500) and 1,000 with gold "none" (When2Call-train cannot rows
   with tools; E12 500); BFCL-format arguments 2,600 (ToolACE; E12 2,000). Within a source, families draw from disjoint
   units as E12 (ToolACE: "call" first, then tool choice; When2Call-train: "none", then "cannot", then "ask"; argument
   items from any ToolACE unit). The ToolACE training pool holds 4,392 units, so tool choice got **1,892**: as built,
   **13,992 public items** from 11,392 units (ToolACE 6,992, When2Call-train 7,000; 1,136 argument items with gold
   OTHER). The training pools were 13,438 When2Call-train units (6,670 ask, 6,768 cannot, 4,746 of them with tools) and
   4,392 ToolACE units; the author's "full When2Call training split" is this pool, sampled to the quota because the 1 : 1
   replay of the next point and the 14-hour cap of (h) bound the mix.
4. *Replay, 1 : 1:* **13,992 rows**, a seeded uniform sample (seed 0) of AgentDecision-4B's SFT v1 training rows
   (`data/train_v1/train.jsonl.gz`, sha256 `9c0450c2…`, 15,694 rows), every JevGym seed re-checked against the training
   guards; no evaluation seed enters any training file. **Mix: 27,984 rows**, shuffled (seed 0), sha256
   `f4db56bc…`. E12's items were a smaller seeded sample from the same pools; E16's sample is drawn afresh, so the two
   mixes overlap but are not nested.
5. Files to `s3://…/e16/data/`, re-checked on the host (`explore_e16_best_open.py verify`).

(d) *Evaluation, each run once, reusing E12's / E12c's scripts and conventions; no finished row is re-run.*
1. **L1, E1's 3,991 decisions** (`configs/explore/e16_best_open/e1_e16.yaml`: E1's pools, C8 exclusions, batching and 4
   workers, a new answer cache), the arm served by `training/serve_jev.py` with its adapter and temperatures and no extra
   arguments. Accuracy pooled and per family (BFCL tool choice, arguments, When2Call), the When2Call accuracy by gold
   label, the paired item bootstrap (E9's `paired_gap`, B = 2,000, seed 0) of the arm minus AgentDecision-4B, Jev,
   StartLux-4B, StartLux-9B and StartLux-27B (their logged E1 answers), $ per 1,000 decisions by E9's rule ($2.24208 per
   GPU-hour × logged serving time per decision) and the same-engine vLLM timing of E12c (h) (`scripts/e12_fast_engine.py`,
   merged LoRA, four states in flight, then one; E1's prompts as `serve_jev` renders them) for both arms and the frozen
   model on the same host. *Serving-path check:* before the arms, the frozen AgentDecision-4B is served on the E16 host
   and answers the same 3,991 decisions; agreement with its logged E1 answers (rule ≥ 99 %, reported either way) and the
   host factor (logged / re-hosted cost), as E12 (f) 1; the agreement is computed off the host from the synced records.
   Keys `e9/e16a/{acc,lg,usdk,…}`, `e9/e16b/{…}` through E9's extra-point mechanism (E9's own frontier and keys and
   `e9/front12/…` stay as written; `e9/front16/…` is the frontier with E12's, E12c's and E16's points), plus
   `e16/<arm>/e1/acc/{toolchoice,args,w2c}` and `e16/<arm>/e1/acc`.
2. **L2, the common states C** (5,314 occurrences, `results/stage2/s23/`): replayed through `scripts/stage2_corpus.py`
   with a new cache; accuracy (task means, as Stage 2), after an off-plan step and on the ask-vs-act stratum; the paired
   task bootstrap of the delta against AgentDecision-4B's registered answers (`ablation_score.paired_boot`, B = 2,000,
   seed 0); keys `e16/<arm>/c/{acc,ask,offplan}`, `e16/<arm>/c/delta{,/lo,/hi}`, `e16/<arm>/c/ni` (non-inferior at the
   1-point margin iff the interval's lower end ≥ −1 point).
3. **L3, live** (`configs/explore/e16_best_open/live_e16a.yaml`, `live_e16b.yaml`): replace mode at the tool choice,
   as the frozen row `rl_ts_replace_f`, E12 and E12c; the same first 200 fresh Stage-2 tasks (seeds 300000–300199,
   `JEV_EVAL_RUNS=1`, never trained on); Flash on the direct Gemini API (temperature 0, minimal reasoning); 4 workers;
   a new System-1 cache; passes 2–3 re-run infrastructure errors only. **Registered gate: an arm runs live only when
   its C accuracy is within 1 point of AgentDecision-4B's (`explore_e16_best_open.py c-gate`: task-mean accuracy minus
   99.2 % ≥ −1.0 point, every question answered).** Before an arm's live run, a smoke on 2 development tasks (A: seeds
   908–909, B: 910–911; never reported) feeds the cost gate of (h). Reported: success (latest record per task, errored
   tasks left out), own-state accuracy, steps, API $ per episode, wall-clock, Newcombe intervals against the frozen and
   Jev rows (`runs/stage2_rl_states`); key `e16/<arm>/live/success`.
4. **The best open-weights model per level, reading rule.** The comparison set is every logged row of a decision model
   with downloadable weights that we did not train: StartLux-Decision 0.8B / 2B / 4B / 9B / 27B (Q8_0),
   Intern-Decision-4B, and E15's open models (OpenJev, OpenJev-27B, Clef-Flash, Clef-27B (L1 only), JevK5, CLM, Laya,
   Verdict, decider-4b, Imajev-4B, Bosun), read from the logged records and keys of E9, the size ladder
   (`explore/startlux_live/…`), Stage 2's corpus answers and E15 (`paper/generated/{e9_frontier,explore,e15_baselines}.tex`
   and the records behind them). Jev, the OpenAI Decisions API and the Gemini deciders are not open weights. Per level,
   the best open model is the row with the highest value: L1 = accuracy on the 3,991 decisions; L2 = common-state
   accuracy (task means); L3 = live success on the 200 tasks among the rows that ran live. An E16 arm "matches or beats"
   it when the arm's point estimate is at least that value (`e16/<arm>/best_open/{l1,l2,l3}` = yes / no, with the gap
   in points `…/gap` and the model's name `…/model`); the paired bootstrap interval of the L1 gap against that model
   (its answers are logged) and the Newcombe interval of the L3 gap (when it ran live) are reported beside. Our own
   earlier open models (AgentDecision-4B, E12, E12c, E14's A) are reported beside the arms, never as "the best open
   model" in this rule. Both arms are reported whatever the outcome; an arm that is not run or does not finish is
   reported as such.
5. Descriptive, also on the host: the held-out JevGym dev files and E12's public development split per family
   (`eval_decisions`) for each selected adapter, when the budget allows (after every registered stage).

(e) *Pre-registered outcomes, reported whatever they turn out to be.* Per arm: (i) L1 accuracy and its paired deltas
against AgentDecision-4B, E12 (A) / E12c (B), Jev and the best open model; (ii) common-state accuracy against
AgentDecision-4B at the 1-point margin; (iii) live success, when the gate let the arm run; (iv) whether the arm matches
or beats the best open model at each level ((d) 4); (v) the same-engine cost beside E9's rule.

(f) *Caps and infrastructure.* One new GPU host launched by this session with `scripts/aws/e16_train.sh`
(Name=jev-e16-train; image `<aws-resource-id>`, E3's clone; 200 GB). Types in order g6e.2xlarge (one L40S 48 GB),
g6e.4xlarge, g5.8xlarge (one A10G 24 GB: arm A only), each tried in every us-east-1 zone. Access is SSM only; the Gemini
key comes from Parameter Store into the live stages' process only, is never printed and never written to disk; no
other key is loaded; `OPENROUTER_API_KEY_JEV` never goes to AWS. The driver refuses every host it did not create.
Stages, in this order: `rearm setup data a_train a_select a_temps rehost a_e1 a_c a_live_smoke a_live b_smoke b_train
b_select b_temps b_e1 b_c b_live_smoke b_live fast final` (a failed stage stops the chain; `final` always runs).
**GPU: at most 14 host-hours**: a power-off guard 830 minutes after launch, an idle stop after 30 minutes, and the
host stops itself when `final` has synced; B's epochs and training cap follow the guard as in (b). **API: $8 of Gemini
for E16 in total** (both smokes and both live runs), enforced by a watchdog on the logged spend; per arm the gate is:
spend so far + 1.2 × (the larger of the arm's smoke mean $ per episode and the other arm's finished live mean) × 200
≤ $8, else no live episode for that arm. No OpenRouter or Jev call. Downloads: the StartLux-9B weights at the pinned
revision, the vLLM and peft packages. Records go to `s3://…/e16/box/` and `results/explore/e16_best_open/`, runs to
`s3://…/e16/runs/`, the adapters to `s3://…/checkpoints/e16_a/` and `e16_b/` (no optimizer state). If a stage cannot
finish before the guard, the readout says which. `scripts/stage2_spec.py --gate` has no entry for exploratory rows and
does not apply.

(g) *Readout.* `scripts/explore_e16_best_open.py summarize` writes `results/explore/e16_best_open/summary.{md,json}` and
the keys `e16/…` in `paper/generated/e16_best_open.tex`; `scripts/explore_e9_frontier.py summarize` adds the two
points (`e9/e16a/…`, `e9/e16b/…`, `e9/front16/…`). An appendix subsection (`paper/sections/app_e16_best_open.tex`,
`app:e16`) reports the table arm × level with the best open model and the gap; no main-text section or figure file is
changed by E16's agent. An "execution as run" entry follows the run.

### Deviations 2026-10-08 ~05:05 UTC: "E19: the rule-randomised model on DecisionGym-Hard" (exploratory, descriptive; registered before any E19 model answer, smoke or evaluation episode)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E19)" (post-Stage-1;
marked in the paper like the other post-Stage-1 rows). It has no test, revises no registered or reported result,
selects nothing and changes no profile, threshold, key or record of any other row. It trains nothing: its model is
E12b's checkpoint "AgentDecision-4B (random rules)", unchanged; AgentDecision-4B (frozen) and E7d's AgentDecision-4B
(tier) are untouched. Nothing has run for E19: no E19 host exists, the checkpoint has answered no E19 question, and
nothing has run on the smoke seeds 7284–7285 (checked in `configs/`, `runs/`, the 115 run manifests in the S3 bucket and
this registration; E7f's seeds 310600–310799 appear only in E7f's two run manifests). E12b's entry said its checkpoint
"is never served live" within E12b; E19 is a separate row that serves it, and E12b's offline results stand as reported.

(a) *Why.* The author's question (2026-10-08): on DecisionGym-Hard's new cancel-or-ship rule (`tier`), AgentDecision-4B
at its original thresholds took nearly all of the rule's decisions and the agent solved 0.5 % of E7b's tasks;
retraining on the rule itself (E7d) and running that checkpoint live (E7f) recovered most of the loss (81.5 % against
83.5 % for LLM-only on 200 tasks; non-inferiority at 3 points not shown). E12b trained the 4B on randomised order rules
that are never `tier` and reached 91.0 % zero-shot on E7d's held-out tier cancel-or-ship states (frozen 76.5 %,
tier-trained 95.3 %), offline only. E19 asks the live question: does rule randomisation in the training environment
repair the collapse on the hard tasks without the model ever seeing the new rule? If the row is non-inferior to
LLM-only, "randomise the task rules in the training environment" becomes a recommendation in the training section; if
not, it is reported as such. **E19 was designed after E7–E7f, E12 and E12b had been read.** No part of those rows is
re-run or replaced.

(b) *Model and serving.* AgentDecision-4B (random rules) is E12b's continued adapter on Qwen3-4B
(`s3://…/checkpoints/e12b_random/`; `adapter_model.safetensors` sha256 `06d5ca9c27159a20…`, `adapter_config.json`
`0beea3e65a1c7fc5…`, as in `results/explore/e12/box/b/e12b_sha256.txt`), served with E12b's fitted temperatures
(`results/explore/e12/box/b/cal/e12b_calibration.json`, sha256 `aa035e01c4ef7037…`) by `training/serve_jev.py` with no
extra arguments (as E7f and E12), under the name `agentdecision-4b-random` on :8081. Adapter and temperatures are
sha256-checked before serving (the adapter is fetched from S3 if absent). Base model: the image's Qwen3-4B snapshot
(`base_model_path.txt`, as E7f). No System-1 cache.

(c) *Thresholds: refitted on the hard calibration states, as E7d / E7f.* The paired row E7f ran its checkpoint at
thresholds fitted by the registered calibration-arm estimator (`threshold_theory --calibration-arm`, W = $1, λ = 0,
V fallback W, A6.5's projection rule) on E7c's calibration states re-answered by that checkpoint, on the first 20 of
E7c's 100 calibration tasks (seeds 5300–5319, the 20 tasks of `results/explore/e7_hard/box_e7d/cal/cal_tasks.json`;
2,017 questions). E19 does the same for its model: the served model re-answers every tool-choice and argument question
of those 20 tasks through the harness's own asking path (`scripts/e5_cal_replay.py build` from `runs/e7c_cal`'s
`rl_shadow_cal` arm, fetched from S3, its arm file sha256-checked against E7d's record `6fd651974c30cfd2…` and the
rebuilt request counts against E7d's `cal_requests.json` (100 tasks; 3,367 tool-choice and 5,460 argument questions);
`explore_e7d_tier.py head-tasks --tasks 20`, `e5_cal_replay.py answer`, `explore_e7d_tier.py complete` and `cal`), at
$0 API, and the estimator gives τ̂ per decision point (`results/explore/e19_hard_random/box/cal/fits_random.json`,
`tauhat_random_*.json`) with the coverage and accuracy at τ̂ on the calibration states, for all tool choices and the
cancel-vs-ship subset. Why not the other two options: the frozen thresholds (τ = 0.5 at the tool choice) are those at
which the frozen model collapsed and at which no retrained checkpoint has run, so they would not pair with E7f; E7f's
own values (0.9999322567940803 / 0.9929112792015076) were fitted on another checkpoint's confidences and mean nothing
for this one. The reflection trigger keeps the profile's fixed τ = 0.5 and asks the same model, as E7f. **Fallback,
fixed now:** if the fit returns "always defer" at the tool choice (as E7c's fit for the frozen model did), the row runs
at E7f's tool-choice threshold instead and the readout says so; the arguments threshold is the fit's in either case
(E7f's if that fit also always defers). The fitted values are appended as an amendment before the smoke, as E7c did,
and copied into the configs; nothing about them is chosen by hand. No serving-path check of the frozen model is run on
this host (E7d's and E12's checks of the same path agreed with the logged answers on 100 % and 99.8 %).

(d) *Setting.* E7f's: multi tasks only, 12 customers, `tier` rule, `answer_style: itemized`, 60 near distractors, 20
noise lines, strict, `max_steps` 45 (`configs/explore/e7_hard/{eval,smoke}_e19.yaml`, added with the amendment of (c);
the evaluation config differs from `eval_e7fx.yaml` only in its name, seeds and n, the backend and the thresholds,
tested).

(e) *Tasks.* **E7f's 200 tasks, seeds 310600–310799**, in one run `runs/e19_hard` (n = 200, seed0 310600; the runner's
guard covers them and the stage sets `JEV_EVAL_RUNS=1`). Task ids are seed-based (`jevgym-multi-<seed>-d60-near-strict`),
so each E19 episode pairs with E7f's logged `llm_only` and `rl_tier` episodes of the same task (`runs/e7f_hard` and
`runs/e7fx_hard`, synced from `s3://…/e7_hard/e7f/` and `e7fx/`; their sha256 are recorded in the summary). Smoke:
development seeds 7284–7285 (the one arm on 2 tasks), never a result.

(f) *Arm.* One arm, `rl_random`: the Stage-2 profile's decision points (tool-choice and argument cascades, reflection
trigger) on AgentDecision-4B (random rules) at the thresholds of (c), otherwise equal to E7f's `rl_tier` arm; 12
workers, task-major (seed 732), as E7f. **LLM-only is not re-run: the reference is E7f's logged `llm_only` arm** (the
same tasks, setting, agent LLM, route, prices and worker count). With one arm in the pool all 12 workers queue their
decision-model calls on the one GPU (in E7f about ten did), which the time readout notes. E7f's `rl_tier` is a second,
descriptive comparison on the same tasks; E7b's frozen-threshold row (0.5 %; other seeds, bare-total answers) is quoted
beside, not paired.

(g) *Route.* Flash on the direct Gemini API (`gemini-3.5-flash`, temperature 0, minimal reasoning), the key from
Parameter Store into the stage's process only (never printed, never on disk); **no OpenRouter call and no Jev call**
($0 each). Dollars compare with E7f's arms, which ran on the same route at the same list prices.

(h) *Pause on API overload* (E14's rule). A live stage (the smoke included) starts only after two consecutive clean
probe rounds (one round = six one-word Flash calls through the harness client, `scripts/explore_e19_hard_random.py
probe`, logged to `results/explore/e19_hard_random/api_probe_log.txt`). While a live stage runs, a watchdog pauses the
runner (records kept) when an episode ends in an infrastructure error naming HTTP 503, overload, high demand,
unavailable or 429; probes then run every 5 minutes, and the stage resumes unchanged after two consecutive clean
rounds (the runner skips completed tasks). After the run has covered every task, the episodes whose latest record is an
infrastructure error are re-run once, after two clean rounds; nothing is re-run after that, and tasks still errored are
left out and counted. Never another route. If the API does not recover before the GPU cap, the row is reported as
partial, with the tasks covered.

(i) *Caps and gate.* **Gemini: $56 for E19 in total, smoke included**, set by the coordinating session after this
session reported, before this entry, that its brief's estimate ("about $9 per arm" for E7f's 200 tasks) was wrong:
E7f's logged cost was $0.264 per LLM-only episode and $0.232 per 4B-arm episode, so the one arm on 200 tasks plus the
smoke projects to about $50 at the gate's factor. A watchdog stops the runner at $56 of logged spend. The gate after
the smoke: the smoke's mean $ per episode × n × 1.05 + the spend so far must fit the cap, for n = 200, else for the
largest multiple of 20 tasks that fits (the first n tasks in seed order, `--max-tasks`; then the row is descriptive and
the record says so), else the row stops after the smoke; the factor is E7f's. OpenRouter $0, Jev $0. **GPU: at most 4 host-hours on one g5.2xlarge** (one A10G 24 GB; on-demand $1.212/h,
us-east-1; tag `jev-e19-hard`; launched by `scripts/aws/e7_hard.sh launch` with `EXP=e19`, with E12's zone fallback,
from E3's clone image as E7f), with a power-off guard 230 minutes after launch, a 30-minute idle stop and a power-off
when `final` has synced. Access is SSM only; the driver refuses every other host. If the calibration replay of (c) is
not complete when 60 minutes remain before the guard, the row is not run and reported as such.

(j) *Execution.* `scripts/aws/e7_hard.sh` with `EXP=e19` on the committed branch bundle: stages `rearm setup serve
refit` (the fit of (c); then its amendment is committed and the bundle pushed again) and `setup smoke eval final`;
`eval` sets `JEV_EVAL_RUNS=1` and follows (h) and (i). Records go to `s3://…/e7_hard/e19/` and
`results/explore/e19_hard_random/box/`, the run to `runs/e19_hard` (gitignored; synced from S3).

(k) *Readout* (`scripts/explore_e19_hard_random.py summarize` → `results/explore/e19_hard_random/summary.{md,json}`,
keys `e19/...` in `paper/generated/e19_hard_random.tex`; `scripts/hard_ni.py` row `e19`). E7f's (j) metrics, against
E7f's `llm_only` on the paired tasks: success; Δ with the paired task bootstrap (B = 10,000, seed 0) and the Newcombe
interval; wins and losses; cost per 1,000 episodes and saving (API as logged; the arm plus the GPU tariff q = $1.212/h ×
the evaluation window / completed `rl_random` episodes, with S2-4's q and q = 0 as sensitivity rows); coverage per
decision point; accuracy on all and on accepted decisions; the cancel-vs-ship split (coverage, accuracy, accepted-wrong
count); reflection firing; mean and median wall time with the latency breakdown; failure types; answer form; E7f's
diagnosis view. Beside it, the same paired comparison against E7f's `rl_tier` (keys `e19/rl_random/vs_tier/...`), and
E7b's frozen row quoted from its keys. **Registered reading:** "non-inferior to LLM-only at 3 points on the 200 tasks"
if and only if `hard_ni.py`'s row `e19` (the A6.21 rule `bb_ordered`, p ≤ 0.05) says so on all 200 paired tasks; with
fewer tasks (the gate, a pause or errors) the row is descriptive and says so. Reported whatever the outcome. An
appendix paragraph in `paper/sections/app_e7_hard.tex` and a row in the NI table of `app_ceiling.tex`; no main-text
section or figure file is changed by E19's agent. An "execution as run" entry follows the run.

### Deviations 2026-10-08 ~05:15 UTC: "E17 and E18: read-out ablation and a deferral-aware objective" (exploratory, descriptive; registered before any E17 or E18 training step, model answer or evaluation episode)

Two exploratory rows, outside both registered stages and every Holm table, labelled "exploratory (E17)" and
"exploratory (E18)" (post-Stage-1; marked in the paper like the other post-Stage-1 rows). They have no test, revise no
registered or reported result, select nothing outside the development data named below and change no profile,
threshold, key or record of any other row. AgentDecision-4B (the frozen JevAgentRL-4B checkpoint), E14's arm A and
every other checkpoint stay frozen and untouched; E17 trains ONE new checkpoint (the generative read-out of (b) (ii))
and E18's optional arm F ONE more. Nothing has run for E17 or E18: no host has been launched, no model has been trained
and no model has answered an E17 or E18 question. Read before registering, at $0 and without any model call: E5's and
E14's records and registrations, E14's logged answers of arm A on the common states and on the calibration states
(`results/explore/e14_small/records/`), the frozen 4B's calibration arm (`runs/e2a_jevgym_shadow_cal_rl`) and the
threshold-fit records (`results/gpu_round5/cal/`, `results/explore/e14_small/cal/`), from which the counts in (c) were
taken.

(a) *Why.* The paper's training story so far says what a decision model is trained on and with which loss. Two facts
ask for more. E5 (generative students, 0.6B-1.7B) and E14 (a 0.8B scoring head) both produced confidences so peaked
that the fitted cascade thresholds sat within 10^-9 of 1; coverage at those thresholds (E14 arm A: 80.0 % of the tool
choices and 78.8 % of the arguments; the 4B: 77.5 % of the arguments) is then decided by rounding luck, and the harness
cannot tell the decisions the model gets wrong from the ones it gets right by confidence. And the harness's rule
(Section 3, Proposition 1: accept a decision iff the model's chance of being right beats the LLM's by the cost margin)
never appears in the objectives (cross-entropy and Brier on gold or teacher labels). **E17 asks whether the read-out
(scoring head against one-token generative read-out), with the backbone and the rows held fixed, changes the usable
coverage and the latency. E18 asks whether a model can learn the rule itself: a deferral option, trained from states on
which the LLM's answer and the gold are both known.** Both were designed after E5 and E14 had been read.

(b) *E17: read-out ablation (same backbone, same rows, two read-outs; one seed).*
1. **(i) Scoring head** = E14 arm A exactly: the merged checkpoint `s3://…/checkpoints/e14_small/A/final/` (Unsloth
   2026.10.2, Qwen3.5-0.8B revision `2fc06364…` plus the fresh Clef joint schema head, A's recipe and its selected
   epoch 1; the sha256 of `joint_head.safetensors` and `unsloth_decision_config.json` are recorded on the host). It is
   not retrained. Its logged development, common-state and calibration-state answers (E14's records) are the (i) rows
   of every table; only its latency is measured again, in the same session as (ii).
2. **(ii) Generative one-token read-out** = SFT v1's recipe (E5's `sft_run`, `scripts/train/10_sft.sh`, E12c's flags:
   LoRA r 32 / alpha 64 on all linear layers, soft cross-entropy + 0.5 Brier, a fresh option permutation per row, lr
   1e-4 cosine with 3 % warm-up, 64 rows per optimizer step, 3,072-token micro-batches without activation
   checkpointing, bf16, maximum length 4,096 (tail kept), gate set `data/train_v0/dev.jsonl`, trainer seed 0) on the
   SAME backbone (Qwen3.5-0.8B at the same revision, loaded through the text-only class in `/opt/jev/v2venv`, A7.19
   (f) / E12c) and the SAME rows: SFT v1's `train_mix.jsonl` (sha256 `9fe1088d…`, 16,902 rows; the file A's 14,787
   records were built from; checked on the host). Memory setting as E5: a 4-step smoke on the 48 longest prompts plus
   48 random rows fixes it (`nogc_3072`, then `gc_3072`, `gc_2048`, `gc_1536`, `gc_1024`, the next only after an
   out-of-memory error). **Epochs: 2 (the recipe) when the smoke's step time projects two epochs within 170 minutes,
   else 1 (recorded), else the arm is not run (recorded).** The final adapter is used (no selection); temperatures by
   `training/calibrate.py --min-bucket 200` on SFT v1's dev rows; served by `serve_jev --fast` with the adapter and the
   temperatures (E5's serving). A single-token check of the option codes (E5's preflight, `e5_box.py readout`) runs
   before the smoke; if it fails the arm is not run.
3. *Comparison, each once, keys `e17/head/…` and `e17/gen/…`.* (1) SFT v1's dev files (`eval_decisions` with the
   temperatures: accuracy by decision point, after an off-plan step, knockout, ECE; (i) from E14's `e14a_v1.json`).
   (2) The common states C (`results/stage2/s23/`, `stage2_corpus.py replay`, a new cache): task-mean accuracy over all
   states, after an off-plan step and on the ask-vs-act stratum, ECE; the paired task bootstrap (ii) minus (i) and
   against AgentDecision-4B (B = 2,000, seed 0; E5's reading rule). (3) The calibration fit, E5's entry (g): the 1,125
   tool-choice and 1,328 argument questions of `rl_shadow_cal` (Flash, seeds 5000-5149) answered by the served (ii)
   (`scripts/e5_cal_replay.py answer`), written into a copy of the arm, the registered command per decision point
   (`threshold_theory --calibration-arm --task-value 1.0 --lam-per-hour 0 --v-fallback W`); reported per point: tau-hat,
   coverage at tau-hat, the accuracy of the accepted decisions and the cascade's accuracy ((i) from E14's fit). (4)
   Latency, one session on the host's A10G, one model on the GPU at a time: the frozen AgentDecision-4B (sha256-checked,
   `serve_jev --fast`) first as the anchor, then (i) (`serve_jev --backend clef`, E14's engine v1), then (ii); the final
   harness's request stream `scripts/harness_final.py bench --path new` on `results/harness_final/replay_workload.jsonl.gz`,
   c = 1 on 25 episodes and c = 4 on 100; p50 / p90 ms per request and System-1 ms per step; peak GPU memory
   (`nvidia-smi`) while serving each model and while (ii) trains. **Registered reading** (descriptive): at each decision
   point the read-out with the higher coverage at its fitted tau-hat *among the read-outs whose accepted
   calibration-state decisions are all right* "yields the higher usable coverage"; when neither is error-free, no
   read-out does; the c = 1 median latency per request is read beside it.

(c) *E18: a deferral-aware objective (the model learns when to defer).*
1. **Labelled states.** States with a gold label AND the live LLM's own answer: the calibration states of the frozen
   4B's shadow arm (`rl_shadow_cal`: Flash, the live LLM, seeds 5000-5149; 1,125 tool choices and 1,328 arguments with
   a gold label and Flash's logged answer), with the reference model's answers = E14 arm A's logged answers on them
   (`records/cal/answers_e14a.jsonl.gz`). Counts read from the logs before this entry: Flash is right on 1,125 of the
   tool choices and 1,244 of the arguments; A on 1,122 and 1,315. Other states with a logged LLM answer were
   considered and are NOT used: SFT v1's shadow-mode rows carry Flash-Lite's answers (the data v1 executor), but A was
   trained on those rows, so the reference model's correctness is uninformative there; the GRPO logs of
   `results/gpu_round5/` hold no per-state LLM answers. **Never used: the common states C, the Stage-2 block
   300000-300999, the evaluation seeds 200000-200499 and the E7 / E7d / E13 seeds** (`bench/registry.training_exclusions`).
2. **Target (an expected-loss target in the style of Mozannar & Sontag's learning to defer, `mozannar2020consistent`).**
   For a labelled question x with gold y, the LLM's answer m and the reference head's answer h with top probability
   p1 and second p2: d(x) = 1 ("defer") iff m is right and (h is wrong or p1 - p2 < k_dp); else d(x) = 0 ("answer",
   with the gold as the answer). "Right" is the registered correctness (`bench/metrics.is_correct`: the acceptable tool
   set at the tool choice; normalised equality at the arguments). k_dp = the cost margin of the frozen 4B's calibration
   fits (Delta c / V with V = W = $1, `results/gpu_round5/cal/tauhat_rl_shadow_cal_*.json`): 0.0058322 at the tool
   choice, 0.0034651 at the arguments. **Read from the logs before this entry: d = 1 on 3 tool choices and 12
   arguments of the 2,453 labelled questions.** With 15 positive states the design's fallback applies: the defer option
   is NOT trained into the backbone; it is fitted as a small head on top of the frozen scoring head (A).
3. **The defer head (E18, the row).** Logistic regression P(d = 1 | x) = sigmoid(w . f(x)) with f(x) = [1,
   log10(1 - p1) clipped below at -15, 1[tool choice], ln(number of options)], fitted on the 2,453 labelled questions by
   Newton's method on the L2-penalised negative log-likelihood L(w) = -sum_x [d log P + (1 - d) log(1 - P)] + (lambda / 2)
   ||w[1:]||^2 with lambda = 1 (fixed now; the intercept is not penalised). No split and no selection: 15 positives
   leave nothing to select on; the coefficients, the fitted positives and the in-sample log-loss are reported.
   **Decision rule** (Proposition 1 with the plug-ins P(LLM right and model wrong | x) = P(d = 1 | x) and
   P(model right and LLM wrong | x) = 0: Flash was right on every labelled tool choice and 93.7 % of the arguments):
   defer iff P(d = 1 | x) > k_dp. It is served as confidence = 1 - P(d = 1 | x) by the clef engine with the head
   (`serve_jev --backend clef --clef-defer-head`), with tau = 1 - k_dp per decision point in
   `configs/harness/explore_e18_final.yaml` (a copy of `jevagent_final.yaml` plus `backends.e18`; written and
   committed with the head before any live episode); the option probabilities are A's, unchanged. No threshold is
   searched anywhere. The head file is `results/explore/e17_e18/e18_head.json`.
4. **Optional arm E18-F (lowest priority; offline only).** The same target as a (K + 1)-th option `__defer__` of every
   choice question, trained into a NEW checkpoint with A's recipe (Unsloth's from-LM defaults, one epoch = A's
   selected count, no second epoch, temperatures by Unsloth's `calibrate` on A's development records carrying the
   option) on A's 14,787 records (target: the gold distribution, the option at 0; the reference head answers 99.7 % of
   its held-out rows, so the plug-in target there is the answer) plus the 2,453 labelled calibration-state questions
   as records (the arguments of one step in one record; the harvest's option cap of 52: gold plus a seeded sample;
   target `__defer__` where d = 1, else one-hot on the gold). Loss: Unsloth's soft-target cross-entropy over the K + 1
   options, L = -sum_q sum_o t_{q,o} log p_{q,o}; rule: defer iff p_defer > k_dp (served confidence = 1 - p_defer, the
   other probabilities renormalised). Read out on SFT v1's dev files (accuracy with and without the option; coverage
   and accepted accuracy with it) and on C as in 5; no live row, no threshold fit. It runs only after every other
   stage of (d) has finished and only if the projected minutes fit the cap; otherwise it is reported as not run.
5. **Evaluation on C (descriptive; the tool choices of C: 5,314 occurrences, 3,939 questions).** Coverage = the share
   of occurrences the rule accepts; the accuracy of the accepted decisions; the accuracy of the deferred ones; against
   E14 arm A at its fitted tau-hat (its logged C answers) and AgentDecision-4B at its frozen thresholds (`answers_rl.jsonl`,
   tau 0.49999999998 at the tool choice: it takes every decision). E18's C numbers are computed from A's logged C
   answers (the head reads only the served top probability, the question type and the option count), at $0. The share
   of the deferred occurrences on which the LLM was right is read on the 1,498 occurrences from the `llm_only` source,
   where the LLM's executed tool is logged, with its count.
6. **Live, ONE row** (`configs/explore/e17_e18/live_e18.yaml`): the final harness with E18 (arm A served with the defer
   head; tool + argument cascade at tau = 1 - k, reflection off, argument questions batched per step), the 200 tasks of
   the logged dedicated final-harness rows (seeds 300500-300699, `runs/harness_final_dedicated`; the same 200 tasks as
   E14's `e14_final`), 1 worker, task-major schedule seed 732, the model alone on its own GPU, no System-1 cache; Gemini
   3.5 Flash on the direct Gemini API (temperature 0, minimal reasoning), `JEV_EVAL_RUNS=1`; a smoke on 2 development
   tasks (seeds 906-907, never reported) first; E14's probe, pause and one re-run rules ("E14: pause and resume on API
   overload"); the cost gate: the smoke's mean $ per episode x 200 x 1.5 + the spend so far must fit the Gemini cap of
   (d). Reported: success (latest record per task, errored tasks left out and counted), API $ per episode, the GPU
   tariff q = $1.624 per hour x the episode's wall-clock (the logged rows' tariff host; E9's $2.24208 as sensitivity),
   wall-clock, System-1 / System-2 time, the share of tool and argument decisions System-1 took; paired against the
   logged `llm_only` and `harness_final` rows and E14's `e14_final` row on the common tasks (Newcombe intervals, paired
   task bootstrap). Keys `e18/final/{n,success,saving_vs_llm,saving_vs_e14,toolcov,argcov,api,q,wall}`.
7. **Registered reading** (descriptive): "coverage higher than E14 arm A's with success within 3 points of LLM-only"
   holds iff the live row's tool-choice coverage and argument coverage both exceed E14's `e14_final` row's (85.5 % and
   86.4 %) and its success minus the logged `llm_only` success is at least -3.0 points (point estimate; the Newcombe
   interval beside). Also reported: whether E18's C coverage exceeds A's at tau-hat with no accepted error more than A's.

(d) *Caps and infrastructure.* One new GPU host launched by this session with `scripts/aws/e17_e18.sh`
(Name=jev-e17-e18; image `<aws-resource-id>`, E3's clone; g5.2xlarge with one A10G 24 GB, else g5.4xlarge, else
g5.8xlarge; every us-east-1 subnet tried). Access is SSM only; the Gemini key comes from Parameter Store into the live
stage's process only, is never printed and never written to disk; no other key is loaded; `OPENROUTER_API_KEY_JEV`
never goes to AWS. The driver refuses every host it did not create and the instances listed as never-touch. Stages,
in this order (a failed stage stops the chain; `final` always runs): `rearm setup data live_e18 smoke_gen train_gen
eval_gen latency e18f_train e18f_eval final`. **GPU: at most 8 host-hours**: a power-off guard 470 minutes after launch,
an idle stop after 30 minutes, the host stopped by its last stage. **API: at most $8 of Gemini** (the smoke and the live
row; a watchdog on the logged spend). **No OpenRouter and no Jev call.** Downloads: Qwen3.5-0.8B at the pinned
revision; A's checkpoint, E14's calibration requests and the frozen 4B from the bucket. Records go to
`s3://…/e17_e18/box/` and (slimmed) `results/explore/e17_e18/`; runs to `s3://…/e17_e18/runs/`; the generative adapter
to `s3://…/checkpoints/e17_gen/`, E18-F's checkpoint to `s3://…/checkpoints/e18f/`. If a cap or the deadline cuts a
stage, the readout says which. `scripts/stage2_spec.py --gate` has no entry for exploratory rows and does not apply.

(e) *Readout.* `scripts/explore_e17_e18.py summarize` writes `results/explore/e17_e18/summary.{md,json}` and the keys
`e17/…`, `e18/…` in `paper/generated/e17_e18.tex`; an appendix subsection (`paper/sections/app_e17_e18.tex`, labels
`app:e17` and `app:e18`) states the loss, the target and the rule and reports every number above; no main-text section
or figure file is changed by this agent. An "execution as run" entry follows the run.

### Deviations 2026-10-08 ~06:00 UTC: "E19: the fitted thresholds" (exploratory; written after the refit of E19 (c) and before any E19 smoke or evaluation episode)

(a) *Times and host.* E19's registration was committed at 05:06:50 UTC (`c8560a8`); its driver, readout and tests at
05:36:52 UTC (`b6628eb`). The host `<instance-id>` (g5.2xlarge, one A10G, us-east-1d; tag `jev-e19-hard`) was
launched at 05:11:09 UTC, between the two commits, and sat idle until the chain `rearm setup serve refit` was sent at
05:37:28 UTC on the code commit. No E19 model answer, smoke or evaluation episode preceded this entry. The launch's
user-data armed the power-off guard 230 minutes after launch (09:01 UTC); every send carries `GUARD_END` 09:00 UTC.

(b) *The refit as run* ($0 API; `results/explore/e19_hard_random/box/`). `setup` (5 min) fetched the checkpoint from S3
and verified it: `adapter_model.safetensors`, `adapter_config.json` and E12b's temperatures all match their recorded
sha256 (`random_verify.json`). `serve` brought up `agentdecision-4b-random` on :8081 at 05:44:06 UTC with no extra
argument (`serve.cmd`). `refit` (05:44–05:55 UTC): the source arm file `runs/e7c_cal/rl_shadow_cal.jsonl` matched E7d's
record, the rebuilt request file had E7d's counts (100 tasks; 3,367 tool-choice and 5,460 argument questions) and
E7d's sha256 (`9130f9e5…`), and the first 20 tasks (seeds 5300–5319, the tasks of E7d's `cal_tasks.json`) hold
**1,940 questions, all answered** (replay 05:44:14–05:55:09 UTC). The registration's "2,017 questions" was E7d's count of
answered questions including partial tasks; E7d's own fit used these same 1,940 (`derive: replaced 1940`).

(c) *Fitted thresholds* (the registered calibration-arm estimator, W = $1, λ = 0, V fallback W, projection rule,
unchanged; `cal/fits_random.json`, `cal/tauhat_random_*.json`): **tool choice τ̂ = 0.9999538704122877** (740 paired
decisions), **arguments τ̂ = 0.9862272143363953** (1,200 paired). Neither is "always defer", so the fallback of E19 (c)
does not apply; the reflection trigger keeps the profile's τ = 0.5. At τ̂ on the calibration states the model takes
20.0 % of the tool choices and 12.9 % of the 240 cancel-vs-ship decisions, right on every one it takes; its accuracy on
all 240 cancel-vs-ship decisions is 87.1 % (E7d's record on the same states: the tier-trained checkpoint 96.7 % and
42.1 % taken at its τ̂; the frozen checkpoint 79.2 % and always-defer). At the arguments it takes 90.0 % (99.1 %
accurate on all). The values are copied unchanged into `configs/explore/e7_hard/{eval,smoke}_e19.yaml` (tested against
`tauhat_random_*.json`). The three answer files (11 MB) stay in `s3://…/e7_hard/e19/box/cal/`; `answers_random_cal.jsonl`
sha256 `690235e274a13f20…` is in `fits_random.json`.

(d) *Stated before the live run.* At this fit the model takes far fewer of the rule's decisions than E7f's checkpoint
did on the same states (12.9 % against 42.1 %), so the live row tests the paper's calibrate-then-deploy procedure on
the rule-randomised model, not the model's own rule reading alone; the live coverage and accuracy on the cancel-or-ship
decisions are reported as registered, beside the offline 91.0 % of E12b.

(e) *Next.* After this entry's commit the bundle is pushed again and `setup smoke eval final` is sent; the smoke seeds,
the $56 cap, the gate and the pause rule are as registered ((e), (h), (i)).

### Deviations 2026-10-08 ~06:10 UTC: "E17: the generative arm's epoch and micro-batch under the measured step time" (exploratory; amends E17 (b) 2 before any E17 training step)

The 4-step smoke at `nogc_3072` took 60.3 s per optimizer step on the host's A10G (the first step includes kernel
compilation; 37.7 s at the last step): the Qwen3.5 architecture runs its convolution on the reference PyTorch path
(no `causal_conv1d` wheel for the image's torch), and at the 3,072-token budget each 64-row step is 47 forward passes
of 1.35 rows. The registered rule (2 epochs within 170 minutes, else 1, else not run) projects 266 minutes per epoch
and no epoch; the driver recorded the arm as "not run" at 05:59 UTC, before this entry. Amendment: (1) the micro-batch
token budget starts at 6,144 (on an out-of-memory error the registered ladder from 3,072 follows), which changes the
number of rows per forward pass and nothing else (64 rows per optimizer step, the loss, learning rate, LoRA rank,
sequence length and data unchanged; E3's addendum on memory settings); (2) the arm trains ONE epoch when the new
smoke projects it within 280 minutes, else it is not run; the second epoch of SFT v1's recipe is dropped and recorded
(E14's head also saw the rows once); (3) the chain is restarted in the order `code smoke_gen train_gen eval_gen latency
e18f_train e18f_eval final`: the latency stage that started at 05:59 UTC is stopped and re-run later in one session
with all three models, and the optional arm F runs after the generative arm only if the cap allows; (4) peft 0.21.2
was installed into the image's v2 venv by hand at 05:49 UTC (it was missing and the generative arm's LoRA needs it; the
driver's setup stage now installs it). No E17 model has been trained and no E17 evaluation has run; E18's live row
(finished 05:55 UTC, 200 tasks) is unaffected.

### Deviations 2026-10-08 ~06:10 UTC: "E19 extension: the remaining 20 tasks" (exploratory, descriptive; registered before any episode of the extension)

Addendum to E19. It is exploratory, has no test, sits outside every Holm table and revises no registered or reported
result. Requested by the coordinating session after E19's smoke projection had been read and while E19's evaluation of
the first 180 tasks runs (nothing of that run has been read: no record of `runs/e19_hard` has been opened, and its
readout has not been produced). Nothing has run on seeds 310780–310799 in E19 (E7f's two arms ran them; no E19 record
or manifest holds them).

(a) *Why.* E19's registered gate ran at the smoke: the 2-task smoke cost $0.57 ($0.285 per episode, more than E7f's
4B arm's $0.232 because this model defers more tool choices), so 200 tasks projected to $60.4 against the $56 cap and
the gate chose "run the first 180" (seeds 310600–310779; `box/projection.json`). The registered reading of E19 (k),
non-inferiority to LLM-only at 3 points by the paper's rule, needs all 200 paired tasks; with 180 the row is descriptive
only. The coordinating session therefore raised E19's total direct-Gemini cap **from $56 to $62** (about $6 more: the
20 tasks at the first part's cost) so that the row covers E7f's whole block and pairs fully with E7f's logged arms.

(b) *Tasks.* **The remaining registered E19 seeds 310780–310799 (n = 20)**, the last 20 of E7f's block in seed order,
in one run `runs/e19x_hard` (`configs/explore/e7_hard/eval_e19x.yaml`, which differs from `eval_e19.yaml` only in its
name, seed0 and n; tested). No smoke.

(c) *Unchanged.* Everything else is E19's: the one arm `rl_random`, the checkpoint (sha256-checked and served as
`agentdecision-4b-random` on :8081 with E12b's temperatures, no extra argument), the thresholds of the amendment "E19:
the fitted thresholds" (tool choice 0.9999538704122877, arguments 0.9862272143363953, reflection trigger 0.5), the
setting (12 customers, tier rule, itemized answer, 60 near distractors, 20 noise lines, strict, `max_steps` 45), Flash on
the direct Gemini API (never OpenRouter), 12 workers, task-major (seed 732), `JEV_EVAL_RUNS=1`, the probe gate and the
pause-on-overload rule of E19 (h) with one re-run of errored episodes, and the same host `<instance-id>` (started
again if it has stopped itself; its power-off guard at 09:00 UTC stands, so the extension runs only if it fits before
the guard; `scripts/aws/e7_hard.sh` with `EXP=e19x`; records to `s3://…/e7_hard/e19x/` and
`results/explore/e19_hard_random/box_e19x/`).

(d) *Cap and gate.* **$62 of direct-Gemini spend for E19 in total** (smoke, first part and extension together),
enforced by the watchdog on the logged spend of all three runs. There is no smoke: the gate takes the first part's
logged mean $ per `rl_random` episode, × 20 × 1.05, plus the spend so far (smoke and first part); if the result is at
most $62 the extension runs all 20 tasks, else it does not run and is reported as such (`box_e19x/projection.json`).
GPU: within E19's 4 host-hours; the extension is expected to take about 15 minutes.

(e) *Readout.* `scripts/explore_e19_hard_random.py summarize` pools both runs: 200 tasks, paired per task with E7f's
logged arms, and `e19/setting/n` becomes 200; the registered metrics and reading of E19 (k) apply to the pooled 200
(`hard_ni.py` row `e19`). The readout also reports each part's own paired comparison (keys `e19/first/...` for the 180
and `e19/ext/...` for the 20, with a "By part" table in `summary.md`). The GPU tariff is the host price × the two
evaluation windows / the completed `rl_random` episodes of both parts. The pooled 200 is the row's result, reported
whatever the outcome; the first part's 180 are never re-run.

### Deviations 2026-10-08 ~08:05 UTC: "E19: execution as run" (exploratory, descriptive; written after the runs, before any paper text used them)

(a) *Timing labels* (commit times are authoritative; all UTC). E19's registration `c8560a8` 05:06:50; its driver,
readout and tests `b6628eb` 05:36:52; the thresholds amendment with the configs `3986fc5` 05:57:43; the extension's
registration `c9bcb9a` 06:03:37; the extension's driver, readout and config `af66aa9` 07:27:42. The host
`<instance-id>` (g5.2xlarge, us-east-1d) was launched at 05:11:09 and sat idle until the first chain (`rearm setup
serve refit`) was sent at 05:37:28 on `b6628eb`. The refit ran 05:44–05:55 as recorded in the amendment. The second
chain (`setup smoke eval final`, on `3986fc5`) was sent at 05:58:07: two clean probe rounds, the smoke 05:59:06–06:00:39
(2 development tasks, $0.57, no error), the registered gate "run the first 180" (the smoke's $0.285 per episode
projected 200 tasks to $60.4 against the $56 cap), two clean probe rounds, and the evaluation of seeds 310600–310779
from 06:00:44 to 07:41:49 (12 workers, Flash on the direct Gemini API, no pause and no errored episode: `box/watch.log`,
`box/api_probe_log.txt`, `box/status.txt`). Every stage of the first chain, the refit and the first 180 episodes
preceded the extension's registration; no record of the first part was read before it (the readout of the first part was
first produced at 07:43, after the extension's registration and after its chain had been sent).

(b) *The extension as run.* The host's own `final` announced its power-off at 07:41:58; this session cancelled it over
SSM at 07:42:03 (`sudo shutdown -c`, status Success) so that the extension could run on the same host, as its entry
allows, and sent `rearm setup serve xproj eval final` at 07:42:39 on `af66aa9`. The server came up again at 07:43:38
(the same checkpoint and temperatures, sha256-verified again: `box_e19x/random_verify.json`); the gate projected from
the first part's logged $0.266 per episode: $48.37 so far + $5.58 for 20 tasks ≤ $62, "run all 20"
(`box_e19x/projection.json`, basis sha256 in `basis.sha256`); two clean probe rounds; the evaluation of seeds
310780–310799 ran 07:43:46–07:55:56 with no pause and no errored episode. The host powered itself off at 07:58
(state `stopping` at 07:59, `stopped` at 08:03:34, verified; no other instance was touched).

(c) *Caps.* Direct Gemini **$53.79 of $62** (smoke $0.57, first part $47.80, extension $5.42; `e19/spend`); OpenRouter
$0; Jev $0; no other key was loaded on the host. GPU: the host ran from 05:11:09 to 07:58, **2.78 host-hours of 4**;
the tariff window (both evaluations) is 1.887 h, q = $0.01144 per episode at $1.212/h over 200 episodes.

(d) *Readout* (`scripts/explore_e19_hard_random.py summarize` → `results/explore/e19_hard_random/summary.{md,json}`,
keys `e19/...` in `paper/generated/e19_hard_random.tex`, `hardni/e19/...` in `paper/generated/hard_ni.tex`), reported
whatever the outcome. **Pooled, 200 tasks, paired with E7f's logged `llm_only`:** success 84.5 % against 83.5 %
(+1.0 points, paired task bootstrap [−4.5, +7.0]; Newcombe [−4.9, +6.9]; 18 wins, 16 losses). **Registered reading:
the certified ordered p at the 3-point margin is 0.091, so non-inferiority to LLM-only is not shown** (the same
outcome as E7f's retrained model, p = 0.402). Against E7f's `rl_tier` on the same tasks: 84.5 % against 81.5 % (+3.0
[−3.5, +9.5]; 26 wins, 20 losses; descriptive). Cost: API $266.1 against $264.3 per 1,000 episodes (−0.7 % saving,
[−2.7, +1.5]); with the GPU tariff $277.5 (−5.0 % [−7.1, −2.8]): no saving. Live, the model took 43.4 % of the 11,275
tool decisions (tool choice 32.0 %, arguments 65.2 %) and 33.4 % of the 2,400 cancel-or-ship decisions, was right on
96.6 % of all of those and accepted no wrong one; the reflection trigger fired on 14.2 % of its questions. Time per
episode 389.4 s against 74.2 s (twelve workers queued their decision-model calls on one GPU; 92.4 System-1 calls per
episode). Failures: 31 wrong final totals against 33 for LLM-only; no wrong state-changing call, no early finish. By
part: the first 180, 85.0 % against 83.9 % (+1.1); the 20 of the extension, 80.0 % against 80.0 %. The live share of
cancel-or-ship decisions taken (33.4 %) is higher than on the calibration states (12.9 %): the calibration tasks ask for
the total only and the agent visits other states there; both are reported.

(e) *Code after the registration* (presentation and the extension only): the per-experiment label in the summary
header; the extension's driver stage `xproj` for `EXP=e19x`, the pooled readout with each part, and the calibration-fit
keys (`e19/cal/...`) written beside the readout's keys. No threshold, setting, arm or record was changed after its
entry. The paper's appendix paragraph (`app_e7_hard.tex`), the NI-table row (`app_ceiling.tex`) and the generated-keys
input line (`main.tex`) were added after this readout; no main-text section or figure file was changed by E19's agent.

### Deviations 2026-10-08 ~14:10 UTC: "E17 and E18: execution as run" (exploratory, descriptive; written after the runs; E18's live row was already in the paper at the coordinating session's request, E17's and arm F's numbers enter with this entry)

(a) *Timing labels* (commit times are authoritative; all UTC). The registration `b7ee800` 05:33:03; the code (deferral
option, defer head, driver, readout) `93e60c7` 05:33:42; the in-progress snapshot with E18's live records `aa43268`
05:56:59; the amendment "E17: the generative arm's epoch and micro-batch" `ad355d2` and its driver `2e69428` 06:03:12;
the driver fix of the restart stage `f65f8bb` 07:24:53. The host `<instance-id>` (g5.2xlarge, one A10G,
us-east-1d; tag `jev-e17-e18`) was launched at 05:34:18 (`RunInstances`); its launch armed the power-off guard 470
minutes after the first boot. **First chain** (on `93e60c7`): `rearm` 05:35; `setup` 05:35:38–05:41:35 (the frozen 4B
sha256-checked); `data` 05:41:36–05:41:58 (the built records sha256-matched `data_sha256.json`, arm A's checkpoint and
its hashes in `arm_a_sha256.txt`, the calibration requests matched their sha256, Qwen3.5-0.8B at the pinned revision);
`live_e18` 05:41:59–05:55:00 (arm A with the defer head up on :8083 at 05:42:27; four clean probe rounds; the smoke on
the 2 development tasks 05:43:17–05:44:13, $0.006, no error; the gate projected $0.94 for 200 tasks against the $8 cap,
"go"; the 200 tasks 05:44:55–05:54:56 with 1 worker, no pause, no errored episode); `smoke_gen` 05:55:01–05:59:27 at
`nogc_3072` (the read-out check passed; 60.3 s per step, so the registered rule projected no epoch and the arm was
recorded as not run at 05:59, the fact that led to the amendment); `train_gen` and `eval_gen` skipped; `latency` started
at 05:59:33 and was stopped by the restart before any model was served or timed (no bench record exists from it). peft
0.21.2 was installed into the v2 venv by hand at 05:49 (amendment (4)). The restart chain `kill_prev code smoke_gen ...`
was sent at 06:03:56 on `2e69428`; its `kill_prev` stage killed its own forked subshell (exit 143) and the chain stopped
at 06:04:04 without running another stage. The host stopped itself on the idle rule at about 06:30 (CloudWatch: no
datapoint after the 06:25 bin). **Second chain**: the host was started again at 07:25:39 (`StartInstances`) and
`rearm code smoke_gen train_gen eval_gen latency e18f_train e18f_eval final` was sent on `f65f8bb`. `rearm` 07:26:14
re-armed the guard for the remaining 358 minutes (13:24 at the latest); `smoke_gen` 07:26:38–07:28:47 at `nogc_6144`
with no out-of-memory error: 22.7 s per step, 100.2 minutes projected for one epoch within the amended 280, so one
epoch; `train_gen` 07:28:48–08:21:41 (SFT v1's recipe on `train_mix.jsonl`, sha256 `9fe1088d…` checked again; 265
optimizer steps, 3,111 s of training, final training loss 0.377, peak GPU memory 17.4 GB; the adapter and the
temperatures in `s3://…/checkpoints/e17_gen/`); `eval_gen` 08:21:42–09:04:17 (SFT v1's dev files raw and with the
temperatures fitted by `calibrate --min-bucket 200`; served by `serve_jev --fast` from 08:35:08; C replayed, 3,939
answers, none errored; the calibration states replayed from E14's request file, 2,719 answers, none missing);
`latency` 09:04:18–09:33:57 in one session, one model on the GPU at a time: the frozen 4B (sha256-checked) up at
09:09:01, arm A on E14's engine v1 at 09:20:12, the generative arm at 09:25:44, each at c = 1 on 25 episodes and c = 4 on
100; `e18f_train` 09:33:58–11:02:12 (the cap check: 3.99 h counted + 2.2 h needed ≤ 7.75 h; one epoch, 1,045 steps on
16,707 records = A's 14,787 plus 1,920 records holding the 2,453 labelled questions; 84.2 minutes with the temperature
fit and the merge; the checkpoint in `s3://…/checkpoints/e18f/final/`); `e18f_eval` 11:02:13–11:16:46 (the with-option
read-out on the dev and labelled records, served with the option from 11:07:48, C replayed, 3,939 answers); `final`
11:16:47 announced the power-off in 2 minutes. The host powered itself off at about 11:19 and was found `stopped` at
13:49 by this session (AWS credentials were unavailable from the end of the previous session until 13:49); no other
instance was touched. Every registered stage ran, including the optional arm F.

(b) *Analysis as run* ($0, local, after the sync of `s3://…/e17_e18/{box,runs}` into `runs/`): the generative arm's
calibration fit by the registered command (`scripts/explore_e17_e18.py fit-gen`: E5's derive replaced 2,719 answers,
none missing; projection rule at both points, 1,125 and 1,328 paired decisions; `results/explore/e17_e18/cal/`), then
`collect` and `summarize`. Code changed after the registration, presentation only: the summary reads the two JSON lines
of `live_window.json`; keys for arm F's with-option read-out (registered in (c) 4 but without keys), the training rows of
both read-outs and the head's latency relative to the one-token read-out; the generated file's header names E17 and E18.
No threshold, setting, arm or record was changed.

(c) *Caps.* Direct Gemini **$0.66 of $8** (smoke $0.006, live row $0.657, plus about $0.001 of probes); OpenRouter $0;
Jev $0; no other key was loaded on the host. GPU: the host ran 05:34:18–~06:30 and 07:25:39–~11:19, **about 4.8
host-hours of 8**; the driver's own count runs from the first boot and includes the stopped gap (5.74 h, `e17/gpuhours`),
so every cap check on the host was conservative.

(d) *Readout* (`results/explore/e17_e18/summary.{md,json}`, keys `e17/...`, `e18/...` in `paper/generated/e17_e18.tex`),
reported whatever the outcome. **E17.** Held-out rows (SFT v1's dev file): 99.7 % for both read-outs; ECE 0.001 for
both with temperatures (the one-token read-out 0.032 raw). Common states C: 99.0 % against the head's 99.2 % (−0.2
points, paired task bootstrap [−0.4, −0.0]); after an off-plan step 92.3 % against 94.1 % (−1.8 [−3.6, −0.2]); ask-vs-act
100 % for both. Calibration states (E5's procedure): fitted τ̂ 0.99962 at the tool choice and 0.99967 at the arguments
(the head's: within 10^-9 of 1); coverage at τ̂ 100.0 % of the 1,125 tool choices and 88.5 % of the 1,328 arguments
against the head's 80.0 % and 78.8 %, with no wrong decision accepted by either. **Registered reading: the one-token
read-out yields the higher usable coverage at both decision points.** Beside it: on the calibration states the one-token
read-out was right on every tool choice (the head missed 3), so the fit saw no error to guard against there; on C its
tool-choice threshold accepted 98.4 % of the 5,314 occurrences with 66 wrong decisions accepted, where the head at its
threshold accepted 78.0 % and no wrong one (descriptive). Serving, one session on the A10G, median per request at c = 1:
head 92 ms, one-token read-out 247 ms, the 4B 318 ms; at c = 4: 204, 796 and 875 ms; GPU memory while serving 2.5, 4.5
and 22.0 GB. **E18-F.** One epoch; 99.7 % on the held-out rows without the option; with the option the rule deferred none
of the 668 held-out tool choices and 824 held-out argument questions, no tool choice and 11 of the 1,328 arguments of its
own labelled training states, and on C it
accepted every tool choice (63 wrong ones accepted, as arm A taking every decision); C accuracy 99.1 %, after an off-plan
step 86.2 % (−8.0 points against the head [−14.0, −2.6]). **E18** (unchanged from the readout used by the paper): live,
100.0 % against 99.5 % for LLM-only on the 200 common tasks, 89.3 % cheaper with the GPU charged, coverage 100.0 % of
the tool choices and 98.7 % of the arguments; the registered live reading holds; on C the defer head accepts 98.5 % with
46 wrong decisions accepted against arm A's 78.0 % and none, so the C reading does not hold.

(e) *Paper.* The E17 result paragraph and arm F's paragraph of `app_e17_e18.tex` are written after this entry from the
keys above; E18's live paragraph, the E18 table and the main-text row (`3263859`) are unchanged. The registered
one-epoch amendment applies: the one-token read-out saw the rows once, as the head did.

### Deviations 2026-10-08 ~14:50 UTC: "E16: execution as run" (exploratory, descriptive; written after the runs; arm A's readout and appendix paragraph were already committed by the coordinating session in `f9ab7b0`, arm B's numbers enter with this entry)

(a) *Timing labels* (commit times are authoritative; all UTC). E16's registration is `1e4967f` (04:24:13). The host
`<instance-id>` (g6e.2xlarge, one L40S 48 GB, us-east-1c) was launched at 04:27:30, after a first round at 04:25
found no capacity for any of the three types in any zone. The whole chain (`rearm setup data a_train a_select a_temps
rehost a_e1 a_c a_live_smoke a_live b_smoke b_train b_select b_temps b_e1 b_c b_live_smoke b_live fast final`) was sent
once, on `1e4967f` (first status line 04:28:41); no other chain or stage was sent. The power-off guard was armed for
18:16:42. The driver, `explore_e16_best_open.py` and the E16 configs are unchanged since the registration; the readout
(`explore_e16_summary.py`, last changed in `1998413`, 04:32:14) and the paper wiring (`21ddcd4`, 05:05:55: the guarded
input lines in `main.tex` and `appendix.tex` and the section draft) never ran on the host. The E16 agent stopped (usage
limit) after arm A's records had been synced; the coordinating session then committed arm A's readout, keys and appendix
paragraph (`f9ab7b0`, 13:38:27), so arm A's numbers entered the appendix before this entry. This container's AWS login
expired while the chain ran and was renewed at 13:49; the host ran unattended meanwhile and nothing in the chain needed
it.

(b) *Arm A as run.* Training `nogc_3072` with no out-of-memory error, 145 minutes (04:36–07:02), both epoch adapters
saved. Selection: the two epochs scored exactly the same on both development files (E1 prefix 87.248 %, v1 dev
99.748 %), so the registered tie rule took epoch 2. Temperatures (2 min), the serving-path check (the frozen model
re-served on this host on E1's 3,991 decisions; agreement with its logged answers computed off the host: 99.8 %, rule
≥ 99 %; host factor 1.01), E1 (3 min), corpus C (16 min; 3,939 answers, none missing; gate: 99.14 % against 99.20 %,
live allowed), the smoke on development seeds 908–909 (gate "run all 200") and the live run of seeds 300000–300199
(07:36:53–07:46:03, one pass, no errored episode) followed as registered.

(c) *Arm B as run.* The 9B weights were downloaded at the pinned revision during setup (sha256 in
`box/startlux_9b_sha256.txt`, licence in `box/startlux_9b_LICENSE.txt`). The smoke (07:46–07:50) found `nogc_3072` out
of memory and `gc_3072` at 28.4 s per step, one epoch projected at 205 minutes; with 626 minutes left before the guard
the training cap was 420 minutes, so the registered rule gave **two epochs**. Training took 303 minutes (07:50–12:54),
both epoch adapters saved. Selection chose epoch 2 (E1 prefix 87.9 % against 86.9 %, v1 dev 99.8 % against 99.7 %).
Temperatures (4 min), E1 (7 min), corpus C (31 min; 3,939 answers, none missing; gate 99.31 % against 99.20 %, live
allowed), the smoke on development seeds 910–911 (projection: $2.95 spent + 1.2 × $0.0145 per episode, the larger of
the smoke's mean and arm A's live mean, × 200 = $6.43 ≤ $8, "run all 200") and the live run (13:52:49–14:11:01, one
pass, no errored episode) followed as registered.

(d) *Same-engine timing as run* (vLLM 0.31.0 with FlashAttention and the plain sampler, as in E12's fix; merged LoRA;
E1's prompts; four requests in flight as the primary, then one). All three models were timed on this host's L40S:
AgentDecision-4B (mixed+) $0.0047, StartLux-9B (mixed) $0.0140 and the frozen AgentDecision-4B $0.0047 per 1,000
decisions (wall clock at E9's $2.24208 per GPU-hour). Agreement with each model's served E1 answers: 99.6 %, 99.9 % and
99.8 %. E12c's timings were made on an A10G, so the two sets compare only within each host.

(e) *Not run.* The descriptive offline evaluations of (d) 5 (the held-out JevGym dev files and E12's public development
split per family for each selected adapter) had no stage in the registered stage list of (f), and the chain ended at
`final`; they were not run.

(f) *Caps and host.* GPU: the host ran from 04:27:30 until its own `final` stage powered it off (scheduled at 14:35:33
for 14:37:33), **10.2 host-hours of 14**; neither the guard nor the idle stop fired. State `stopped`
(`Client.InstanceInitiatedShutdown`) was verified from the coordinating container at 14:44; no other instance was
touched. Gemini on the direct API: **$5.85 of $8** (arm A smoke $0.03 and live $2.90; arm B smoke $0.03 and live
$2.90); the watchdog never stopped a run; the key-leak check passed after every live stage. OpenRouter $0; Jev $0.
Downloads: the StartLux-9B weights, the vLLM and peft packages. The selected adapters with their temperatures are in
`s3://…/checkpoints/e16_a/selected/` and `e16_b/selected/` (epoch adapters under `train/`, no optimizer state).

(g) *Readout* (`scripts/explore_e16_best_open.py summarize` → `results/explore/e16_best_open/summary.{md,json}`, keys
`e16/…` in `paper/generated/e16_best_open.tex`; `scripts/explore_e9_frontier.py summarize` → `e9/e16a/…`, `e9/e16b/…`,
`e9/front16/…`), reported whatever the outcome. The best open-weights decision model we did not train is StartLux-27B
(Q8_0) on the fixed questions (87.7 %) and on the common states (93.9 %), and StartLux-9B live (74.0 %).
- **Arm A, AgentDecision-4B (mixed+).** (i) E1 84.3 % [83.2, 85.4]: against AgentDecision-4B +5.6 points [+4.4,
  +6.8], against E12 −0.7 [−1.6, +0.1], against Jev −2.1 [−3.2, −1.0], against StartLux-27B −3.4 [−4.5, −2.2]
  (When2Call 66.9 %, the weak part). (ii) Common states 99.1 % against 99.2 % (−0.1 [−0.3, +0.2]): non-inferior at
  1 point. (iii) Live 100 % (200 of 200; against AgentDecision-4B +0.0 [−1.9, +1.9], against StartLux-9B +26.0 [+20.1,
  +32.5]).
  (iv) Matches or beats the best open model at L1 / L2 / L3: no (−3.4) / yes (+5.2) / yes (+26.0). (v) $0.0909 per
  1,000 by E9's rule on this host; $0.0047 on the same engine.
- **Arm B, StartLux-9B (mixed).** (i) E1 84.8 % [83.7, 85.9]: against AgentDecision-4B +6.1 [+4.6, +7.5], against the
  untrained StartLux-9B +1.4 [+0.2, +2.6], against E12c −0.7 [−1.7, +0.4], against Jev −1.6 [−2.8, −0.5], against
  StartLux-27B −2.9 [−4.0, −1.7]. When2Call 63.6 %: it answered "call" on most items whose gold reply is to ask the
  user (14.3 % right on those), the reverse of E12's dominant error (answering "ask"). (ii) Common states 99.3 %
  against 99.2 % (+0.1 [−0.1, +0.4]): non-inferior at 1 point. (iii) Live 100 % (200 of 200; against StartLux-9B
  +26.0 [+20.1, +32.5]). (iv) No (−2.9) / yes (+5.4) / yes (+26.0). (v) $0.2404 per 1,000 by E9's rule on this host;
  $0.0140 on the same engine, three times the 4B arms'.
- **Registered reading.** Neither arm matches or beats the best open-weights decision model at every level: both lead
  on the common states and live, and both stay behind StartLux-27B on the fixed questions. Neither E16 point is on E9's
  frontier, which is unchanged (StartLux-0.8B → Jev → Gemini 3.5 Flash).

(h) *Paper.* Arm B's appendix paragraph (`app_e16_best_open.tex`) is written after this entry from the keys above; arm
A's paragraph and the guarded table row stay as committed. No main-text section or figure file was changed for E16
(`main.tex` and `appendix.tex` hold only the guarded input lines of `21ddcd4`).

### Deviations 2026-10-08 ~15:30 UTC: "E20: decision-model ensembles and a hosted decision API behind the gate" (exploratory, descriptive; registered before any E20 model call or episode)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E20)" (post-Stage-1;
marked in the paper like the other post-Stage-1 rows). It has no test, trains nothing, revises no registered or
reported result and changes no profile, threshold, key or record of any other row. Nothing has run for E20: no model
has answered an E20 question and no E20 episode exists. Read before registering, at $0 and without any model call:
the stored answers of every decision model on E1's 3,991 fixed decisions and on the common states C (the inputs of
`scripts/explore_drift.py`), E14's, E15's, E17/E18's and E19's registrations and records, and Part A below up to its
development half.

(a) *Why.* The author's question (2026-10-08): can several decision models voting together, or a hosted decision API
behind our confidence gate, replace training on the agent's states? **No LLM is ever a voter**: an ensemble consists of
decision models only, and the agent LLM receives only the deferred decisions, as in every cascade. **Part A** (offline,
post hoc, descriptive, $0; no entry of its own by the convention of `scripts/explore_drift.py`; labelled so in every
output) combines the stored answers into majority votes, confidence-weighted votes and "m of k" agreement gates, for
three families (off the shelf; off the shelf plus our trained models; hosted only = Jev and the OpenAI Decisions API).
Members and m are chosen on a development half (C split by task, L1 by item, a fixed hash written to
`results/explore/e20_ensembles/split.json` at 15:22 UTC before any accuracy was computed; `choice.json`) and reported on
the other half. **Part B** (this entry) asks the live question for the hosted candidates, the only decision models
callable from this container without a GPU. Part A's development half decided Part B's second row by the rule written
into `scripts/explore_e20_ensembles.py` before `choose` ran: on C's development half (92 tasks, 2,493 occurrences) the
hosted agreement gate (Jev and the Decisions API give the same tool) delegates 92.6 % with 113 accepted errors, and the
Decisions API alone, accepting its most confident 92.6 %, accepts 133.1 (expected count under random tie-breaking); so
row (ii) runs. (Jev alone at the same coverage accepts 102.3, fewer than the gate; Jev alone behind the gate is not an
E20 row.) Part A's held-out half had not been computed when this entry was written.

(b) *Rows* (the final harness: tool + argument cascade, reflection off, argument questions batched per step; the
profile `configs/harness/explore_e20_final.yaml` = `jevagent_final.yaml` plus one `backends.<name>` block per row with
the fitted thresholds of (c)). At most two rows, both hosted, no GPU:
1. **(i) `oaidec_final`**: OpenAI's Decisions API (`openai_decisions`, `gpt-6-luna`, the E15 backend unchanged: all
   questions about one state in one request, a refusal is no answer and falls through to the LLM).
2. **(ii) `agree_final`**: an agreement gate of two decision models, Jev (`jev`, OpenRouter preset, `typesafe/jev-1.13`,
   called with the author's Jev-only key `OPENROUTER_API_KEY_JEV`, which only `decision/jev.py` reads) and the
   Decisions API (as in (i)). A new backend `decision/ensemble.py` (`{type: ensemble, rule: agree, members: [...]}`)
   sends each question set to both members in parallel (each member's call goes through the Ledger under its own
   backend name and price; the two calls of one decision share a group tag so that parallel System-1 time can be read
   as the slower member's), and returns: when both members give the same answer, that answer with **confidence = the
   smaller of the two members' confidences**; otherwise (different answers, a refusal or an error of either member)
   the answer with the larger summed confidence and **confidence 0**, so that the cascade delegates only when the two
   agree and the combined confidence reaches the fitted threshold. The logged distribution is the members' mean; each
   member's answer and confidence are kept in the answer's `raw`. No LLM call happens inside the backend.

(c) *Thresholds: the registered estimator on the labelled calibration states E18 used.* The 2,453 labelled questions of
`rl_shadow_cal` (`runs/e2a_jevgym_shadow_cal_rl`: Flash as the live LLM, calibration seeds 5000–5149; 1,125 tool choices
and 1,328 arguments; E18's `e18_labelled.jsonl.gz` lists them) are rebuilt exactly as the agent asks them
(`scripts/e5_cal_replay.py build`) and answered once by each member through the harness's asking path (the Decisions
API and Jev; a local answer cache so that an interrupted pass resumes without paying twice). The ensemble's answers on
these states are computed from the two members' answers by the same function the live backend uses (no further call).
Each row's answers are written into a copy of the arm (`e5_cal_replay derive`) and the registered command per decision
point fits τ̂ (`threshold_theory --calibration-arm --task-value 1.0 --lam-per-hour 0 --v-fallback W`, as E5 (g), E17 and
E19). A fit that accepts no confidence level ("always defer") makes that decision point always defer for the row
(System-1 not queried there, A6.11 (c)); a row whose two decision points both always defer equals LLM-only and is not
run live (reported as such). Reported per row and point: τ̂, coverage at τ̂ on the calibration states, the accuracy of
the accepted answers and the accepted-error count; for (ii) also the members' agreement rate. The fitted thresholds are
written into the profile and committed with an amendment "E20: the fitted thresholds" before any smoke episode;
nothing is searched beyond the registered command.

(d) *Live.* The 200 tasks of the logged dedicated final-harness rows (seeds 300500–300699, `runs/harness_final_dedicated`:
the suite block, 1 worker, task-major schedule seed 732; the same tasks as E14's `e14_final` and E18's `e18_final`),
Gemini 3.5 Flash on the direct Gemini API (temperature 0, minimal reasoning), `JEV_EVAL_RUNS=1`, no System-1 cache.
Configurations `configs/explore/e20_ensembles/live_{oaidec,agree}.yaml` and smokes `live_smoke_{oaidec,agree}.yaml`;
the two rows may run at the same time (separate processes, one worker each). A smoke of 2 development tasks per row
(seeds 912–913 for (i), 914–915 for (ii); never reported) first; the gate per row: the smoke's mean $ per episode × 200
× 1.5 plus the spend so far must fit each provider's cap of (g). E14's pause-and-resume rule ("E14: pause and resume on
API overload": probe rounds of six one-word Flash calls logged to `results/explore/e20_ensembles/api_probe_log.txt`,
start and resume after two consecutive clean rounds, a pause on an overload error, one re-run of errored episodes,
never another route). **Reference: the logged `llm_only` row of `runs/harness_final_dedicated` (2026-10-05), not
re-run** (E18's reference; nothing in this entry needs a same-session reference; its date is stated beside every time
comparison). Descriptive comparisons also against the logged `harness_final` (AgentDecision-4B) and `e18_final` rows on
the common tasks.

(e) *Reported* (latest record per task; errored tasks left out and counted). Success; the paired difference to
`llm_only` (Newcombe interval; paired task bootstrap, B = 2,000, seed 0); API $ per episode split by provider (Gemini,
OpenAI, Jev); the saving against `llm_only` (1 − mean cost ratio, paired task bootstrap; no GPU tariff: no GPU is used);
mean and median wall-clock per episode and its paired difference; System-2 time and the parallel System-1 time per
episode; the share of tool choices and of arguments System-1 decided (coverage); the accepted errors (accepted System-1
decisions whose answer is wrong by the registered rule, `bench/metrics.is_correct`) per decision point; refusals; for
(ii) the share of questions on which the members disagreed. Keys `e20/live/{oaidec,agree}/…`, `e20/cal/…`.

(f) *Registered reading* (descriptive, per row): "the hosted gate keeps success and saves cost" holds iff the row's
success minus `llm_only`'s is at least −3.0 points (point estimate; the Newcombe interval beside) and its saving point
estimate is above 0. Also read, without a pass mark: (ii) against (i) on coverage and accepted errors at each decision
point, and both rows against AgentDecision-4B's logged `harness_final` row.

(g) *Caps and infrastructure.* **OpenAI at most $10** (calibration answers, smokes, live), **direct Gemini at most $25**
(smokes and live; a watchdog on the logged spend stops a runner at the cap), **Jev at most $3** (calibration answers,
smoke, live), OpenRouter other than the Jev-only key $0, **no GPU host and no AWS resource**. Every call runs from the
coordinating container; the OpenAI key and `OPENROUTER_API_KEY_JEV` never leave it. Records to `runs/e20_*` (live) and
`results/explore/e20_ensembles/` (calibration answers, fits, slim copies, probe log); readout
`scripts/explore_e20_ensembles.py summarize` → `results/explore/e20_ensembles/summary.{md,json}` and
`paper/generated/e20_ensembles.tex` (keys `e20/…`), an appendix section `paper/sections/app_e20_ensembles.tex` (label
`app:e20`) behind guarded input lines; no main-text section or figure file is changed by this agent. Deadline:
2026-10-09 12:00 UTC; a stage cut by a cap or the deadline is reported as such. An "execution as run" entry follows the
runs. `scripts/stage2_spec.py --gate` has no entry for exploratory rows and does not apply.

### Deviations 2026-10-08 ~15:50 UTC: "E21: a shifted DecisionGym split (DecisionGym-Shift)" (exploratory, descriptive; registered before any E21 model call, episode or evaluation on the new tasks)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E21)" (post-Stage-1;
marked in the paper like the other post-Stage-1 rows). It has no test, trains nothing, revises no registered or
reported result and changes no profile, threshold, key or record of any other row. Nothing has run for E21: no agent
LLM or decision model has seen an E21 task or answered an E21 question, no E21 episode exists and no host has been
launched. Read before registering, at $0 and without any model call: the generator and its oracle, the scripted oracle
policy on the new suite's development seeds (the implementation checks of (b)), and the logged in-domain rows named
in (f).

(a) *Why.* The author (2026-10-08): the trained decision models score about 99 % on the common states (L2) and 100 %
live (L3), and the author reads this as the in-domain evaluation being too easy: training and test come from the same
generator, with the same tool names, request templates and 11 plan shapes. **E21 builds a harder, shifted DecisionGym
split and re-measures the decision models on it at L2 and L3, so that the scores spread and the paper can show where
training on the agent's states stops carrying over.** E21 was designed after E12, E12c, E14, E15, E16, E17/E18 and the
drift analysis had been read. Whatever the outcome, it is reported; if the trained models stay near 100 % here too,
that is the finding.

(b) *Suite "DecisionGym-Shift"* (`src/jevharness/bench/suites/jevgym_shift.py`, a new module; the frozen
`jevgym.py` and `jevgym_hard.py` are not edited; built by `load_suite("jevgym", shift=...)` from the Stage-2 suite
block: families v2, near distractors, 60 distractors, 20 noise lines, strict). A canonical core holds the world, the
tasks and the oracle under the generator's tool names; the environment the harness sees translates tool names at the
boundary. Three axes, each switchable (`shift: {surface, distraction, structure}`), **all on in every E21 run**:
1. *Surface.* Every core tool and the ask tool get one consistent alias and a paraphrased description: find_user →
   lookup_customer, get_orders → list_customer_orders, cancel_order → cancel_purchase, refund → reimburse_order,
   check_stock → inventory_count, ship_order → dispatch_order, send_message → message_customer, ask_user →
   request_clarification (`ALIASES`). Every request text, the policy line of the first observation (which names the
   ask tool) and every customer reply use new paraphrased templates with the same content. The environment maps each
   alias to its canonical tool, so the answer key is the generator's up to renaming and is **expressed in the alias
   names the agent sees** (the oracle's tool, the acceptable sets); a canonical core name is not offered and is an
   unknown tool to the oracle, as to the harness. Implementation fact: parameter names, enum values and observation
   formats are unchanged (renaming them would change the argument decisions, which E21 does not score).
2. *Distraction.* Twice the distractors: the Stage-2 60 (31 near-duplicates of the core tools and 29 unrelated tools)
   plus 22 near-duplicates of the alias names (`SHIFT_NEAR`, functional as in the generator: the read-only ones answer,
   the 10 state-changing ones are side effects that end the episode in strict mode) and 38 new unrelated tools, so a
   tool choice has 129 options (128 tools and finish) instead of 69; and twice the heartbeat noise lines (40) per
   observation.
3. *Structure.* Task kinds by seed mod 10 (`KINDS`): the generator's clarify variants order / method / speed / name
   (40 %), multi with 3 customers (30 %) and three kinds the training generator never produces (30 %):
   **ask twice** (who the customer is and the shipping speed are both missing; plan ask, look up, list pending orders,
   ask, ship, finish), **ask after a lookup** (refund one of the customer's 2-3 delivered orders as store credit: the
   request becomes ambiguous only once the orders are listed; plan look up, list delivered orders, ask, refund,
   finish) and **notify** (ship, then confirm in the support chat; the message must follow the shipment, and the
   instruction rules out e-mail, so the e-mail look-alike is a detour; plan look up, list pending orders, ship,
   message, finish). Only what the answer key labels exactly was added; labels come from the generator's oracle
   (acceptable sets by the cost-to-go rollout, information feasibility by the evidence atoms, with the atoms of the new
   steps added: the customer's id for the message, the replies for the asks).
*Implementation checks (tests in `tests/test_explore_e21.py`, all passing before this entry):* the scripted oracle
policy solves 100 % of the development tasks under each of the 8 axis combinations; aliases round-trip; at every state
the oracle's tool and the acceptable sets are offered names and equal the canonical core's labels after renaming; with
every axis off a task is the generator's own Stage-2 task of that seed (instruction, plan, tools, first observation);
the three new plan shapes never occur in the training generator (families all and v2, 2-4 customers, 400 seeds each);
strict failures, detours and must-ask states of the new kinds behave as described; a snapshot never changes the
episode; the wire canary (`tests/test_wire_canary.py`'s harness test) on the shifted environment with REPLACE at the
tool choice finds no oracle value, label or hidden name in any request to System-1 or System-2. Under the oracle
policy on the 100 development tasks, 7.6 % of the states are must-ask (the acceptable set is the ask tool alone),
against 3.3 % for the Stage-2 generator on the same seeds.

(c) *Seeds.* **Evaluation block 330000-330999** (`registry.E21_EVAL_RANGES`; the runner refuses it without
`JEV_EVAL_RUNS=1`): L2 source tasks **330000-330099** (n = 100, 10 per kind), L3 tasks **330100-330299** (n = 200, 20
per kind), the rest a reserve. **Development seeds 339000-339099** (smokes and debugging only, never reported). The
whole block 330000-339999 is excluded by every training guard (`registry.training_exclusions()`,
`training/seed_guard.TRAIN_FORBIDDEN`; test). Implementation fact: the brief's example block 320000-320999 is not free
(E7d trained on 320000-320599, E7d's held-out block is 321000-321999, E12b trained on 322000-322599), so E21 uses
330000-339999, which no record, configuration or guard used before.

(d) *L2 on the shift.*
1. *Source runs* (`configs/explore/e21_shift/l2_sources.yaml`, run `runs/e21_l2`): `llm_only` and `jev_ts_replace_f`
   (Jev REPLACE at the tool choice) on the 100 L2 tasks; Gemini 3.5 Flash on the direct Gemini API (temperature 0,
   minimal reasoning), 4 workers, task-major schedule (seed 732), a new System-1 cache; further passes re-run
   infrastructure errors only. Jev = `{type: jev, provider: openrouter}` (`typesafe/jev-1.13`), run from this container
   only with the Jev-only key, which only `decision/jev.py` reads and which never goes to AWS.
2. *Corpus* exactly as C (A7.10 (c), `bench/stage2_corpus.build_corpus`): per task and source, the pre-action
   tool-selection occurrences of the final valid trajectory with the logged acceptable set as the label and the
   on-plan / after-an-off-plan-step stratum; question hash over state, options, descriptions and question text; a task
   without both valid sources leaves the corpus and is listed. `results/explore/e21_shift/c/`.
3. *Answers:* every model answers every distinct question through `stage2_corpus.replay` (the live harness's option
   handling: one question per request, the harness knockout above the backend's read width, two retries, a failure
   counted wrong), with a new cache per model; no source answer is reused. Models and serving (as in their logged
   rows): Jev (as in 1, from this container); the OpenAI Decisions API (`{type: openai_decisions, model: gpt-6-luna}`,
   E15's backend, from this container, its key never on AWS); and on ONE GPU host: AgentDecision-4B (the frozen
   JevAgentRL-4B checkpoint, sha256-checked against `results/stage2/rl_eval_plan.json`, `training/serve_jev.py` with
   its temperatures, as its C row), E12 (`agentdecision-4b-mixed`, `checkpoints/e12_mixed/`), E12c
   (`startlux-4b-mixed`, `checkpoints/e12c_sl4b/selected/`, base StartLux-Decision-4B, in the v2 venv), the 0.8B
   scoring head (E14 arm A, `checkpoints/e14_small/A/final/`, `serve_jev --backend clef`, engine v1 as its C and live
   rows), StartLux-Decision-4B / 9B (the vendor's server at package commit `a9101e9`, weights `3890ae0c` / `342aad64`)
   and 27B (the Q8_0 GGUF `a54b311e` via llama.cpp's CUDA server image `d87b02aa` and the vendor's GGUF server, A7.18
   (g)), OpenJev-27B (HF, FP8; E15's vLLM 0.29.0 + the card's helper, E15's flags) and Intern-Decision-4B (in
   process, the Stage-2 replay's spec). A model that does not load or answer on the host is reported as not run, with
   the reason. "Trained" rows: AgentDecision-4B, E12, E12c, the 0.8B head (all trained on the generator's states);
   "off-the-shelf" rows: Jev, the Decisions API, StartLux 4B / 9B / 27B, OpenJev-27B, Intern-Decision-4B.
4. *Metrics* (task means, A7.10 (d)): accuracy over all occurrences and on subsets fixed by rule from the state's
   label, stratum and task before any answer is read: **must-ask** (the acceptable set is the ask tool alone),
   **after an off-plan step** (the stratum), **renamed-tool states** (every acceptable tool is a renamed core tool, i.e.
   finish is not acceptable), **new-kind tasks** (the three new kinds) and **generator-kind tasks**. A subset with fewer
   than 30 occurrences is flagged. Paired task bootstrap (B = 2,000, seed 0) of each model minus AgentDecision-4B and
   minus Jev.
5. *Axis attribution (secondary, descriptive, offline):* every L2 question is also asked in two re-renderings,
   answered by every model in the same serving window after its main replay: **surface undone** (in the options, the
   descriptions and the state text, every alias goes back to its canonical name and description, and the instruction,
   the policy line and the customer's replies go back to the generator's own wording of that task) and **distraction
   undone** (the 60 added distractors are removed from the options; the state text is unchanged, so the doubled noise
   lines stay). Answers are mapped back to the alias names and scored against the same labels. The cost of an axis for
   a model = its accuracy with the axis undone minus its accuracy on the recorded states. These are re-renderings of
   recorded states, not new episodes; the structure axis is read from the new-kind subset. If the GPU guard of (h)
   approaches, the re-renderings of the last models are skipped first (reported).

(e) *L3 on the shift.* REPLACE at the tool choice (the decision model picks every tool, the agent LLM writes the
arguments), the 200 L3 tasks, Gemini 3.5 Flash on the direct Gemini API (temperature 0, minimal reasoning), 4 workers,
a new System-1 cache per row; one re-run of infrastructure errors. Rows: **LLM-only** (the reference), **Jev**,
**AgentDecision-4B**, **the 0.8B head**, **E12** and **StartLux-9B**, and **the OpenAI Decisions API** if its cap
allows. LLM-only, Jev and the Decisions API run from this container (`configs/explore/e21_shift/live_local.yaml`), the
four GPU rows on the host (`live_<row>.yaml`), each right after its L2 window while the model is served. Before each
configuration a smoke on 2 development tasks (never reported) and a gate: spend so far + 1.2 × the smoke's $ per
episode × the tasks still to run ≤ that side's Gemini cap of (h); a row that does not fit runs its first k tasks that
fit (reported with its n). Reported per row: success (latest record per task; errored tasks left out and counted),
own-state tool accuracy (A4 sets), steps, System-1 / System-2 calls, API $ per episode, wall time, refusals (Decisions
API: a refused tool choice falls through to the agent LLM, as E15), success per kind, and paired Newcombe intervals
against LLM-only and against Jev on the common tasks.

(f) *Reading rules* (descriptive; reported whatever the outcome).
- Per model, beside the logged in-domain numbers (C accuracy: Stage 2's corpus replays, E12, E12c, E14, E15; live
  success on the first 200 Stage-2 tasks: `explore/startlux_live`, E12, E12c, E14, E15): L2 on the shift (all,
  must-ask, after an off-plan step, renamed-tool states, new-kind and generator-kind tasks), the axis costs of (d) 5
  and L3 on the shift.
- *Lead.* The trained models keep their lead on a measure iff the best trained row's point estimate is at least the
  best off-the-shelf row's (the gap in points and the paired interval of the two rows are reported). Per axis: the
  lead with all axes on (L2 and L3), on the new-kind tasks (structure) and on must-ask states, and whether an axis
  costs the trained rows more than the off-the-shelf rows (mean axis cost over the trained rows minus the mean over the
  off-the-shelf rows, in points).
- *Ranks.* Kendall's tau-b between in-domain and shift accuracy over every row with both: C vs shift L2 (all rows
  with an in-domain C), in-domain live vs shift L3 (the decision-model rows that ran live in both), and shift L2 vs
  shift L3.
- Public decision models may have seen public tool-use data; nothing in E21 is public. The scores here are of the
  same answer key as C, on a different generator setting, not a measure of real-world transfer.

(g) *Pause and resume on API overload* (E14's / E15's rule): probe rounds of six one-word Flash calls on the direct API;
a live configuration (smokes included) starts after two consecutive clean rounds; a watchdog pauses it on an episode
that errored with HTTP 503 / overload and resumes after two clean rounds; after every task is covered, errored episodes
are re-run once; never another route (no OpenRouter fallback for Flash).

(h) *Caps and infrastructure.* **Direct Gemini: $70 for E21 in total, smokes and probes included**, split into $38
for this container's runs (L2 sources, the local L3 rows) and $32 for the host's live rows, each enforced by a watchdog
on the logged spend of E21's runs; **OpenAI at most $5** (L2 replays and the live row; watchdog on the logged cost);
**Jev at most $5** (L2 source run, replays, live row); **OpenRouter other than the Jev key $0**. **GPU: at most 12
host-hours on one g6e.2xlarge** (one L40S 48 GB; else g6e.4xlarge, else g5.8xlarge, each tried in every us-east-1
zone; on a 24 GB GPU the two 27B rows are not run and the readout says so), launched by `scripts/aws/e21_shift.sh`
(Name=jev-e21-shift; image `<aws-resource-id>`, E3's clone; 400 GB), SSM only, a power-off guard 710 minutes after
launch, a 30-minute idle stop, the host stopped when done; the Gemini key from Parameter Store into the live stages'
process only, never printed or written; no other key on the host; downloaded code and weights run only on the host;
every other instance is left alone. Records to `s3://…/e21_shift/` (raw logs stay there) and slim copies to
`results/explore/e21_shift/`. Deadline: 2026-10-09 18:00 UTC; a stage cut by a cap or the deadline is reported with
the rows that exist and their n. `scripts/stage2_spec.py --gate` has no entry for exploratory rows and does not apply.

(i) *Readout.* `scripts/explore_e21_shift.py summarize` → `results/explore/e21_shift/summary.{md,json}` and
`paper/generated/e21_shift.tex` (keys `e21/…`); an appendix section `paper/sections/app_e21_shift.tex` (label
`app:e21`; one table: model × {L2 shift all, must-ask, renamed-tool states, L3 shift live} beside the in-domain C and
live numbers) behind guarded input lines in `paper/main.tex` and `paper/sections/appendix.tex`; no main-text section
or figure file is changed by E21's agent. An "execution as run" entry follows the runs.

### Deviations 2026-10-08 ~15:50 UTC: "E20: the fitted thresholds" (exploratory; written after the fits of E20 (c) and before any E20 smoke or live episode)

(a) *What ran.* The registration was committed at `ab473d7`. The 2,719 questions of `rl_shadow_cal` were rebuilt
locally (`scripts/explore_e20_ensembles.py cal-build`, sha256 `d46701e8…`), and the 2,453 labelled ones (E18's keys:
1,125 tool choices, 1,328 arguments) were answered once by each member through the harness's asking path, four
questions in flight: the Decisions API (no error, no refusal, $0.33) and Jev with the Jev-only key (no error, $0.21).
The ensemble's answers were computed from the two members' answers by `decision.ensemble.combine` (no further call).
The registered command fitted τ̂ per row and point (`cal-fit` → `results/explore/e20_ensembles/cal/`). No E20
episode has run.

(b) *Fits* (coverage and accepted errors at τ̂ on the calibration states; Flash, the live LLM, is right on all 1,125
tool choices and on 1,244 of the 1,328 arguments).
- **(i) Decisions API: "always defer" at the tool choice and at the arguments.** It is right on 1,054 of the tool
  choices (93.7 %) and 1,153 of the arguments (86.8 %); even its most confident answers are wrong more often than the
  cost margin allows (confidence 1.0: 3 of 34 tool choices and 12 of 626 arguments wrong).
- **(ii) Agreement gate: "always defer" at the tool choice; τ̂ = 0.99 at the arguments** (coverage 50.2 %, 666
  arguments, no accepted error). The members give the same tool on 89.5 % of the tool choices (963 of those 1,007
  right) and the same argument on 75.7 % of the arguments; with both confidences at least 0.99 they agree on only 49
  tool choices, 2 of them wrong.
- Descriptive, no fit: Jev alone is right on 1,060 tool choices and 1,272 arguments; at confidence 1.0 on 352 of 353
  tool choices and all 870 arguments. The Decisions API as a co-voter lowers the combined confidence below Jev's own.

(c) *Consequence for the live rows* (a reading of registration (c), fixed before any episode). Row (i) always defers
at both points: by (c) it equals LLM-only and is **not run live**. Row (ii) always defers at the tool choice. In the
final harness System-1 fills arguments only for a tool that System-1 itself chose (`JevAgent._delegated_call`; when the
LLM chooses the tool, its own function call carries the arguments), so with the tool choice always deferring the
fitted argument threshold is never reached and the row makes no System-1 call at all (shown on a stub in
`tests/test_explore_e20.py`). It therefore also equals LLM-only and is **not run live**; the literal text of (c) names
only rows whose two points both always defer, and this entry extends it to a row whose only non-deferring point is
unreachable. Both backend blocks of `configs/harness/explore_e20_final.yaml` carry `skip` with the reason (written by
`prepare-live`, committed with this entry). Nothing else changes: no threshold is searched, no other row is added,
no Gemini call is made for E20, and the reference row stays the logged `llm_only`. Part B's result is the fits above;
the readout reports them beside Part A (keys `e20/cal/…`), and the "execution as run" entry follows.

(d) *Spend so far.* OpenAI $0.33 and Jev $0.21 (each pass started with a four-question wire check whose answers were
kept; the full pass resumed after them), direct Gemini $0 (no probe has run), OpenRouter other than the Jev key $0,
no GPU.

### Deviations 2026-10-08 ~16:00 UTC: "E20: execution as run" (exploratory, descriptive; written after the calibration answers and the readout, before any paper text used them)

(a) *Timing labels* (commit times are authoritative; all UTC). Part A's split `split.json` (v1) 15:22:14; the
development-half choice (`choice.json`) right after it; the registration text written about 15:26 and committed
unchanged at 15:36:09 (`ab473d7`). Between writing and committing, Part A's held-out half was first computed (about
15:28) and showed that v1 keyed every When2Call decision by its shared prefix `w2c/`, which put all 928 into the
development half of the fixed questions; the split was rewritten with When2Call items keyed by their whole id (v2,
15:29:44, its `revision` field says so) and the choice re-run. The common-state split, the common-state choices and
the hosted development check that decided Part B's rows are identical under v1 and v2; nothing the registration fixes
depends on the fixed questions or on the held-out half. On the development half only, the gate rule's fallback was
refined before any held-out number was computed (the first fallback, the lowest error rate over every gate, picked a
gate that delegated nothing; the second tier requires delegating at least half). Calibration: the questions rebuilt at
15:38, both members' passes 15:38–15:42 (four questions in flight each), the fits and the profile before the amendment
"E20: the fitted thresholds" (`92fc965`, 15:45:54); code, configurations and readout `d16ed45`.

(b) *Part B as run.* Exactly what the amendment records: the 2,453 labelled calibration questions answered once by the
Decisions API and by Jev (no error, no refusal), the ensemble's answers computed from them, the registered command per
row and decision point; row (i) always defers at both points, row (ii) at the tool choice (its argument threshold 0.99
is unreachable in the final harness). Neither row ran live: no probe round, no smoke and no episode; the live and smoke
configurations and the driver `scripts/e20_live.sh` were written before the fit and never run. The logged `llm_only`
reference was not needed.

(c) *Part A as run* (post hoc, descriptive; `scripts/explore_e20_ensembles.py split | choose | offline`). Every stored
answer was read locally: E15's slim per-decision copies in `results/explore/e15_baselines/l1` and `/c` are complete
(3,991 and 3,939 answers per model), so nothing was downloaded from S3 and no AWS call was made. The single-model
accuracies reproduce the published keys (for example Jev 86.4 / 91.5 %, StartLux-27B 87.7 / 93.9 %, AgentDecision-4B
78.8 / 99.2 % on the fixed questions / common states). Families: 18 off-the-shelf models on the common states (Clef
27B, fixed questions only, joins the fixed-question pool), plus AgentDecision-4B, E12, E12c, E14 arm A and E16's two
arms; the hosted pair. DecisionGym-Hard slice: E7c's two shadow calibration arms (`runs/e7c_cal`, seeds 5300–5399)
share 392 labelled tool-choice states, 97 of them the tier rule's cancel-or-ship decisions.

(d) *Spend.* OpenAI $0.33 of $10; Jev $0.21 of $3 (the Jev-only key, from this container only); direct Gemini $0 of
$25; OpenRouter other than the Jev key $0; no GPU host, no AWS resource.

(e) *Readout* (`summarize` → `results/explore/e20_ensembles/summary.{md,json}`, keys `e20/…` in
`paper/generated/e20_ensembles.tex`), reported whatever the outcome. Held-out halves (108 tasks and 2,821 common-state
occurrences; 2,069 fixed decisions):
- **Votes do not reach the trained models.** The best off-the-shelf vote (StartLux-27B with OpenJev; majority and
  weighted agree) is right on 94.9 % of the common states against 93.9 % for StartLux-27B alone and 99.7 % for
  AgentDecision-4B; on the must-ask states 47.3 % against 100 %; predicted live success 81.0 % against 100 %. On the
  fixed questions no vote beats the best single model (86.4 against 87.7 % off the shelf; 88.0 % with our models in
  the vote). A vote that includes our models is no better than the best trained model in it (99.6 against 99.9 % for
  E12c alone on the common states).
- **Errors are correlated.** Over all of C, the six strongest off-the-shelf models' error sets overlap with a median
  Jaccard index of 43.8 % (3.3 % if independent at the same rates), and 86 % of joint errors are the same wrong tool;
  39 % of the 267 must-ask occurrences are failed by all 18 off-the-shelf models; StartLux-27B and OpenJev-27B fail
  165 and 164, 162 of them with the same wrong tool.
- **Agreement is not a better confidence signal than the strongest member's own confidence.** At each gate's coverage
  the strongest member's own confidence accepts fewer errors: 50 against 63 (off the shelf, 4 of 4 agree, 88.8 %
  delegated), 101 against 103 (hosted, 88.2 %), 1 against 3 (with our models, 99.0 %); agreement beats only the
  weaker member (the Decisions API alone: 158). With both hosted confidences at least 0.9 the pair delegates 25.6 % of
  the common states with 21 accepted errors, Jev's own confidence 1 at the same coverage.
- **DecisionGym-Hard.** On the 392 shared states Jev is right on all 97 tier decisions, AgentDecision-4B on 79.4 %;
  the confidence-weighted pair vote 83.5 % (the 4B's wrong answers are the confident ones); the agreement gate
  accepts no error while delegating 79.4 % of the tier decisions and 99.0 % of the others (on the main setting's
  common states the same pair delegates 88.4 % with 1 accepted error). The gate, not the vote, keeps both strengths.
- **Part B.** Neither hosted row earns coverage behind the gate (fits in the amendment); the trained 0.8B head takes
  80.0 % of the same calibration tool choices and 78.8 % of the arguments with no accepted error (E17's keys).

(f) *Paper.* The appendix section `paper/sections/app_e20_ensembles.tex` (label `app:e20`, two tables) is written after
this entry from the keys above, behind guarded input lines in `main.tex` (the generated keys) and `appendix.tex`
(after the drift section); no main-text section or figure file is changed by this agent.

### Deviations 2026-10-08 ~16:10 UTC: "E22: the training recipe on τ²-bench" (exploratory, descriptive; registered before any E22 episode, training step or model answer)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E22)" (post-Stage-1;
marked in the paper like the other post-Stage-1 rows). It has no confirmatory test, revises no registered or reported
result and changes no profile, threshold, key or record of any other row. The model it trains is a NEW checkpoint
(records and serving name `e22-tau2-0.8b`); E14's arm A, AgentDecision-4B (the frozen JevAgentRL-4B checkpoint) and
every other checkpoint stay frozen and untouched. Nothing has run for E22: no E22 simulation exists, no host has been
launched, no model has been trained or asked an E22 question. Read before registering, at $0 and without any model
call: the τ² adapter, the Stage-1 τ² records (`results/tau2/`), E11's, E14's and E18's registrations and records, and
tau2-bench's task-split files.

(a) *Why.* The author (2026-10-08): the open-source contribution should show that the harness plus a small trained
decision model reaches the success of the LLM alone on well-known public agent benchmarks at much lower cost and time.
ALFWorld shows this (E11); τ²-bench does not yet. In the Stage-1 τ² runs (Flash, the hosted Jev) the tool cascade at
its refitted threshold lost success and cost more: retail −4.1 points at −17.7 % saving, airline test set −7.6 points
at −19.5 % (`results/tau2/scorecard_retail.md`, `scorecard_airline_test.md`, pass^1). **E22 applies the paper's recipe
(train a small decision model on the states the agent itself visits; gate it by confidence) to τ²-bench retail and
airline, with the benchmark's official train / test task splits, and measures success, cost and time of the final
harness against the LLM alone on the test tasks.** E22 was designed after the Stage-1 τ² runs, E11, E14, E17 and E18
had been read.

(b) *Tasks.* τ²-bench at the pinned commit `5bfa7e3`, domains retail and airline, the official
`data/tau2/domains/<domain>/split_tasks.json` (`train` / `test`; retail 74 / 40, airline 30 / 20; sha256 `ed0580ec…`
and `b22ced4d…`). **Development tasks:** per domain, the train ids ordered by sha256("jev-explore-E22-tau2-dev-2026-10-08"
+ newline + "<domain>/<id>"); the first round(0.2 × n) are the dev tasks (retail 15: 4 14 21 28 34 44 46 48 63 66 72 76
82 96 107; airline 6: 9 20 27 38 41 43), the other train ids are the fit tasks (59 + 24). The lists are written by
`scripts/explore_e22_tau2.py split` to `configs/explore/e22_tau2/split.json` (committed with this entry). **Test tasks
are never used for harvesting, training, temperatures, epoch choice, thresholds or smokes.** Dev tasks are never
trained on. (A5.4's airline shadow / test split is unrelated and not used.)

(c) *Harvest: the states the agent visits (train tasks only, run in this container).* The τ² adapter with every
decision point off (`configs/explore/e22_tau2/harvest.yaml`; `log_state: true`) and one new logging key, `log_params`:
at every tool call the LLM makes, the argument questions the cascade would ask for its required arguments (options
exactly as the cascade builds them: bounded values, or values copied from the conversation plus `<OTHER>`) are logged
with the LLM's own value, without any System-1 call (behaviour unchanged; offline tests). Agent Gemini 3.5 Flash on
the direct API (`gemini/gemini-3.5-flash`, temperature 0, `thinkingLevel` minimal); user simulator Gemini 3.5
Flash-Lite with the same arguments (the user simulator of the Stage-1 τ² runs, `results/tau2/runs.json`); retail's
NL-assertion judge Gemini 3.5 Flash (tau2's fixed judge is gpt-4.1 and OpenAI is at $0 for E22: a deviation from the
official protocol, identical in every E22 run and reported). K = 8 trials per train task (fit and dev), seed 2201,
`--max-concurrency` 10, retail and airline: 104 × 8 = 832 simulations. A simulation that ends in an infrastructure or
agent error is re-run once (`rerun-errored`), then dropped and counted. Raw outputs are gzipped to
`s3://…/e22_tau2/harvest/`.

(d) *Labels (no step oracle in τ²; the agent's own choices in successful episodes).* Per logged tool-choice state
(options: the domain's tools plus `respond_to_user`; the LLM's executed choice):
1. **Successful episode** (reward 1): target = the LLM's choice, one-hot. Dropped: states whose executed call returned
   a tool error, and states whose executed call is a WRITE tool (tau2 `ToolType`) that matches no reference action of
   the task (tau2's `Action.compare_with_tool_call`).
2. **Failed episode** (reward < 1): a state is kept only if the LLM's executed call matches a then-pending reference
   action exactly (the adapter's `_Reference.pending`, `compare_with_tool_call`) and returned no error; target = that
   tool. Every other state of a failed episode (messages included) is dropped.
3. **Arguments** (from `log_params`) of a kept state's executed call: target = the LLM's value when it is an option,
   else `<OTHER>`.
4. Errored episodes are dropped entirely.
Cross-check, reported: the agreement of the targets with the adapter's reference next step where that names a tool,
and the counts each rule drops. **Records:** one per question in the wire form the harness sends (the tool-choice
`Choice` with the tool descriptions, an argument `Choice` without; `decision/jev.question_to_wire`), one question per
record because the τ² adapter asks one question per request; deduplicated on (state, question) keeping the first;
fit-task records train, dev-task records are held out. Gold enters only training targets; no served request carries it.

(e) *Model and training.* Start: E14 arm A's merged checkpoint (Qwen3.5-0.8B revision `2fc06364…` plus its Clef joint
schema head, `s3://…/checkpoints/e14_small/A/final/`, sha256 of the head and its decision config recorded), continued
with a fresh LoRA under E14's recipe (Unsloth 2026.10.2; LoRA r 64 / α 64 / dropout 0 on the language model, the head
trained in full at 3e-4, LoRA lr 1e-4, cosine with 5 % warm-up, weight decay 0.01, 8 records × 2 accumulation, bf16,
seed 3407, Unsloth's Clef loss) on the fit-task records. **Maximum length 8,192 tokens** instead of E14's 4,096 (an
implementation fact: a τ² state is up to 12,000 characters plus up to 16 tool options with descriptions, and Unsloth
cuts the end of a longer state, which on τ² holds the latest message; the count cut is reported); memory fallback 4 × 4,
then 2 × 8 on an out-of-memory error only. Epochs: one, then a second pass with the schedule restarted only if the
host hours used plus epoch 1's duration plus a 3.0-hour reserve stay within 9.5 hours; the epoch with the higher dev
accuracy (all dev questions) is used, a tie keeps epoch 1. Temperatures: Unsloth's `calibrate` on the dev records. If
arm A's checkpoint cannot be continued (an error before the first optimizer step), the run falls back to E14's recipe
from the fresh backbone and head (recorded).

(f) *Thresholds (the registered estimator, on held-out training tasks).* The served model answers every labelled
dev-task question once, through the `jev` backend and a Ledger exactly as the harness asks it (`explore_e22_tau2.py
replay`), and the answers are written into a shadow-arm file beside the LLM's logged choice (the label of (d), so the
LLM is right on every labelled dev decision) and the logged cost of the LLM call that made it (System-1 costs 0 in the
ledger: self-hosted). τ̂ per decision point by the registered command (`threshold_theory --calibration-arm
--task-value 1.0 --lam-per-hour 0 --v-fallback W`; A6.5's sample-size rule). One implementation fact: in E22's harness
(g) a `respond_to_user` answer of System-1 is never acted on, so the tool-choice fit uses the dev decisions on which
System-1 answers a tool. A point whose fit accepts no confidence level always defers; if either point always defers,
the harness equals LLM-only by construction and is not run (its cell reads "always defers").

(g) *The harness* (`configs/explore/e22_tau2/final.yaml`, written by `fit` with τ̂): the final harness's two decision
points on τ², tool choice and arguments in CASCADE at τ̂, every other point off, the model alone on the host's GPU
(`serve_jev --backend clef`, one request at a time, no System-1 cache). One new adapter key, `s1_act: complete`, for
two implementation facts of the Stage-1 τ² runs: the cascade's LLM calls that were forced to one tool (`tool_choice`
with only that tool's schema) or to text (`tool_choice="none"`) lost Gemini's implicit prompt cache (cached input
9.5 % against 48.0 % for LLM-only on retail, 9.6 % against 48.6 % on the airline test set, which cost more than the
calls the cascade saved), and Gemini 3.5 Flash ignores `tool_choice="none"` in about half of the turns
(`docs/TAU2.md`). Under `s1_act: complete` System-1's tool choice is acted on only when it names a tool and System-1
also decides every required argument at τ̂ (a single-option argument is filled directly): the call is emitted with no
LLM call, the final harness's LLM-free step. Every other turn (below τ̂, `respond_to_user`, an argument left open) is
the LLM-only turn: the same request `llm_only` sends (full tool list, `tool_choice` auto).

(h) *Evaluation (test tasks only).* Retail 40 + airline 20 = 60 tasks, k = 4 trials, seed 2202 (every arm the same
per-trial seeds, so arms pair by task and trial), the settings of (c) (Flash agent, Flash-Lite user, Flash retail
judge), on the GPU host. Arms: `llm_only` (re-run in the same window, not reused) and `e22_final` ((g) at τ̂); optional,
lowest priority, if the cap allows: `e22_tau05` ((g) with τ = 0.5 at both points). The arms run concurrently, one tau2
process per arm, `--max-concurrency` 6 each, `log_state: false`. A smoke first: `e22_final` on two dev tasks per
domain, one trial (never reported). E14's pause-and-resume rule on API overload applies (probe rounds of six one-word
Flash calls; start after two clean rounds; a run whose simulations end in errors naming 503 or overload is paused and
resumed after two clean rounds); after a full pass, errored simulations are re-run once (`rerun-errored`), then left out
of the pair and counted.

(i) *Readout* (`scripts/explore_e22_tau2.py summarize` → `results/explore/e22_tau2/summary.{md,json}`, keys `e22/...`
in `paper/generated/e22_tau2.tex`, appendix `app:e22`), whatever the outcome, per domain and pooled:
- Success = pass^1 (mean over task × trial of reward = 1) and pass^k for k = 1..4; Δ = arm − `llm_only` on the
  (task, trial) pairs; Newcombe 95 % interval on the pairs and a paired task bootstrap (tasks resampled with their trials;
  B = 10,000, seed 0).
- **Non-inferiority by the paper's rule at the τ² margin (5 points):** `metrics.ni_certified_p(wins, losses, n, 0.05)`
  (A6.21, rule `bb_ordered`) on the pooled (task, trial) pairs, α = 0.05: "non-inferior" iff p ≤ 0.05, else "not shown";
  per domain the same p, descriptive. Sensitivity row: per-task majority pass (> half of a task's trials), as A6.14 /
  A6.16 (e).
- Cost per episode = the agent's ledger cost (Flash calls; System-1 at 0) plus the GPU tariff q = the host's on-demand
  $/h (g5.2xlarge $1.212; g6e.2xlarge $2.24208) × the evaluation window of the GPU arms / their completed episodes;
  sensitivity rows q = 0 and E9's $2.24208 per GPU-hour. Saving = 1 − C_arm / C_llm_only. The user simulator and the
  judge are environment cost, reported apart.
- Time: mean and median wall-clock per simulation (tau2's duration). Coverage: the share of agent turns with no LLM call
  (System-1 emitted the complete call); System-1's accepted tool choices and argument decisions; LLM calls per episode.
  Agreement of System-1's emitted calls with the reference next step (descriptive).
- Offline (dev questions): accuracy, by decision point, of the trained model and of arm A before training on τ²; ECE;
  data statistics (states, records by source and rule, options, records cut).
- Ready-to-place rows for Table 1's τ²-bench band.

(j) *Execution and caps.* One new GPU host launched by this session with `scripts/aws/e22_tau2.sh` (Name=jev-e22-tau2;
image `<aws-resource-id>`, E3's clone; g5.2xlarge with one A10G 24 GB, else g6e.2xlarge), SSM only; the Gemini key
comes from Parameter Store into the evaluation stage's process only, never printed, never on disk. The driver refuses
every host it did not create and the never-touch instances (among them `<instance-id>`, `<instance-id>`
and E21's host). Stages: `rearm setup data train replay fit smoke eval final`; a power-off guard 590 minutes after
launch, a 30-minute idle stop, the host stopped by its last stage. **Caps: direct Gemini at most $200 for E22 in total**
(harvest, smokes, evaluation; a watchdog on the logged spend: the agent's ledger plus tau2's user cost plus $0.003 per
retail simulation for the unlogged judge); **GPU at most 10 host-hours**; OpenRouter, Jev and OpenAI $0. *Gate after the
harvest smoke* (two fit tasks per domain, one trial, seed 2299, never trained on): projection = the smoke's mean $ per
simulation × (832 + 480 + 240) × 1.3 plus the spend so far; if above $200, harvest trials are cut (K = 6, then 4; never
tasks), then the optional arm is dropped, then k = 2. *Gate before the evaluation:* the evaluation smoke's mean $ per
simulation × the evaluation simulations × 1.3 plus the spend so far must fit the cap; else the optional arm is dropped,
then k = 2. Deadline: results by 2026-10-10 06:00 UTC; a stage cut by a cap or the deadline is reported with its n.
`scripts/stage2_spec.py --gate` has no entry for exploratory rows and does not apply. An "execution as run" entry
follows the run.

### Deviations 2026-10-08 ~18:40 UTC: "E23: the E11 checkpoint in ALFWorld's unseen rooms (valid_unseen)" (exploratory, descriptive; registered before any E23 episode)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E23)" (post-Stage-1;
marked in the paper like the other post-Stage-1 rows). It has no confirmatory test, trains nothing, fits nothing,
revises no registered or reported result (E11's rows stay as reported) and changes no profile, threshold, key or record
of any other row. It is not Stage 1's registered E6 (`configs/experiments/alfworld_action_only.yaml`: the same split
with Jev, never run, A7.1 (h)) and does not stand in for it. Nothing has run for E23: no E23 episode exists, no E23
host has been launched, and E11's checkpoint has never acted in a valid_unseen game.

Read before registering, at $0 and without any episode: E11's registration, records and summary
(`results/explore/e11_alfworld/`), the valid_unseen game list (task ids and their order) and the scene numbers in the
game names. Prior contact with valid_unseen in this project: the pre-Stage-1 development pilot P3b
(`docs/PILOT_RESULTS.md`) played 5 of its games (one of each task type except `pick_two_obj_and_place`) with Flash as
the agent LLM and Flash-Lite as System-1, and one development smoke (`runs/dev/l3_smoke`) played one of the same 5.
Both are development runs, never reported as results; neither involved E11's model, its data or its threshold. These 5
games stay in the evaluation (the split is run whole) and are listed in the readout. valid_unseen was never used for
E11's harvest, training, temperatures or threshold (E11 (b); its build refused every task id outside `alfworld/train/`).

(a) *Why.* The author (2026-10-08): reviewers will ask whether the trained models' 95–100 % live numbers only show that
a model learned its training generator. E11 evaluated its checkpoint on valid_seen, whose rooms are those of the
training games (74 of its 76 scenes occur among E11's 600 harvest games) with new object placements: 95.0 % success
against 90.7 % for Flash alone, 89 % cheaper. **E23 asks the harder question on the same benchmark: ALFWorld's
valid_unseen split, 134 games in four rooms (AI2-THOR floor plans 10, 219, 308 and 424) that occur in none of E11's 600
harvest games (development games included) and in none of E6a's 48 calibration games, with the same frozen checkpoint
and the same threshold.** E23 was designed after E11's result had been read. Whatever the outcome, it is reported.

(b) *Model and threshold, frozen.* E11's checkpoint `e11-alfworld-4b`: base Qwen3-4B (HF snapshot `1cfa9a72…`) with
E11's LoRA adapter and fitted temperatures from `s3://…/checkpoints/e11_alfworld/` (`adapter_model.safetensors` sha256
`53c5893b…`, `adapter_config.json` `06d8e3d5…`, `calibration.json` `d089d0e7…`; E11's record
`results/explore/e11_alfworld/box/adapter_sha256.txt`). The host downloads these files into a new directory and checks
the three sha256 against E11's record before serving; any mismatch stops the row. Served as in E11:
`training/serve_jev.py --adapter --calibration` on :8081, name `e11-alfworld-4b`, one GPU, no System-1 cache. The
CASCADE threshold is E11's τ̂ = 0.120764 (`configs/explore/e11_alfworld/tauhat_action.json`, the value E11's eval.yaml
ran). No refit, no new calibration run, no training step.

(c) *Games.* All 134 valid_unseen games that pass ALFWorld's own filters (no `movable` / `Sliced`, the six task types,
`solvable`): pick_and_place_simple 24, look_at_obj_in_light 18, pick_clean_then_place_in_recep 31,
pick_heat_then_place_in_recep 23, pick_cool_then_place_in_recep 21, pick_two_obj_and_place 17. Order:
`load_alfworld("valid_unseen", stratify=True)` (task types round-robin over the sorted game files, as E6a and E11),
written by `scripts/explore_e23_alfworld_unseen.py eval-list` to `configs/explore/e23_alfworld_unseen/eval_games.json`,
list sha256 `3c46e12b31bfcb62…`. On the host the games are extracted from the ALFWorld release archives (sha256 as in
E6a's record), and the host's list must reproduce this sha256, else the row stops.

(d) *Arms and settings* (`configs/explore/e23_alfworld_unseen/eval.yaml`): E11 (h)'s, with only the split changed
(`suite.split: eval_out_of_distribution`, n 134). Flash on the direct Gemini API (`gemini-3.5-flash`, temperature 0,
minimal reasoning), `max_steps` 50, `history_window` 12, oracle `auto` unpatched (it only scores decisions, as in E11's
evaluation), `help` / `inventory` excluded, task-major schedule seed 732, 8 workers, one runner invocation. Arms, with
E11's names:
- `llm_only`, re-run in the same pool (paired wall-clock), never reused from E11 or the pilot.
- `e11_action_cascade_tauhat`: the trained model at the action point, CASCADE at τ̂ = 0.120764 (E23's primary arm).
- `e11_action_replace`: the trained model alone, no LLM call (secondary).

(e) *Smoke and gate.* Smoke (`configs/explore/e23_alfworld_unseen/smoke.yaml`): the three arms on the first 2 of E6a's
48 calibration train games, as E11's smoke (development, never reported; no valid_unseen game). Gate: the smoke's mean
$ per episode per arm, summed over the arms, × 134 × 1.5, plus E23's spend so far, must fit $15 for all 134 games;
else the first 100 games of the list if they fit; else the row stops after the smoke.

(f) *API overload: E14's pause-and-resume rule* (Deviations 2026-10-07 "E14: pause and resume on API overload"
(b)–(e)), in place of E11's passes 2–3. A probe round is six one-word Flash calls on the direct API through the harness
client (`explore_e23_alfworld_unseen.py probe`, appended to `results/explore/e23_alfworld_unseen/api_probe_log.txt`).
The smoke and the evaluation start only after two consecutive clean rounds (6 of 6). A watchdog pauses the runner
(records kept) when an episode ends in an infrastructure error naming 503, overload, high demand, unavailable, resource
exhausted or 429; the runner resumes unchanged after two consecutive clean rounds (completed tasks are skipped). After
every task of every arm has a record, the episodes whose latest record is an infrastructure error are re-run once,
after two clean rounds; tasks still errored are left out of the pairs and counted. No other route (no OpenRouter
fallback).

(g) *Readout* (`scripts/explore_e23_alfworld_unseen.py summarize` → `results/explore/e23_alfworld_unseen/
summary.{md,json}`, keys `e23/...` in `paper/generated/e23_alfworld_unseen.tex`, a short appendix paragraph App.
`app:e23` right after App. `app:e11`), whatever the outcome, as E11 (i). Paired complete case against `llm_only`. Per
arm: success; Δ with the paired task bootstrap (B = 10,000, seed 0) and the Newcombe interval; wins / losses.
**Non-inferiority by the rule Table 1 applies to E11's row (`scripts/hard_ni.py`):
`metrics.ni_certified_p(wins, losses, n, 0.03)` (A6.21, rule `bb_ordered`), "non-inferior" iff p ≤ 0.05, else "not
shown", with the Tango 95 % interval**, for the cascade (primary) and, descriptively, the replace arm. Cost per
episode: Gemini as logged plus the GPU tariff q = host $/h (on demand, us-east-1) × evaluation window / completed
episodes of the two GPU arms (E11's convention); q = 0 and S2-4's q as sensitivity rows; saving = 1 − C_arm / C_ref
with its bootstrap interval. Wall-clock mean and median per episode; steps; LLM calls per episode; coverage (the share
of actions taken from the trained model); among the actions it took where the gold action is known, the share that
matched the single gold action and the number and share that did not ("accepted actions that differ from the gold", an
upper bound of its errors, since many states admit several good actions); the deferred LLM's accuracy; the model's
mean confidence. Success by task type (descriptive, small n). E11's valid_seen numbers are quoted beside each row
(read from `results/explore/e11_alfworld/summary.json`, not recomputed), with the seen-to-unseen change per arm
(unpaired: different games; descriptive). A row `e23` (the cascade) is added to `scripts/hard_ni.py`, and a ready row
for Table 1's ALFWorld band is produced; placing it in the main text is the author's decision.

(h) *Execution and caps.* One new GPU host launched by this session with `scripts/aws/e23_alfworld_unseen.sh`
(Name=jev-e23-alfworld-unseen; image `<aws-resource-id>`, E3's clone, as E11), SSM only: a g5.4xlarge (E11's type,
one A10G 24 GB) if capacity exists, else a g5.2xlarge, else a g6e.2xlarge. The driver refuses every host it did not
create (E11's stopped host `<instance-id>` is not restarted) and the never-touch instances
(`<instance-id>`, `<instance-id>`, E21's and E22's hosts). The Gemini key comes from Parameter Store into
the stage's process only; it is never printed and never written to disk. Stages: `rearm setup serve smoke eval final`;
a power-off guard 230 minutes after launch, a 30-minute idle stop, and the host stopped by its last stage once the
outputs are in `s3://…/e23_alfworld_unseen/`. **Caps: direct Gemini at most $15 for E23 in total** (probes, smoke,
evaluation; a watchdog on the logged spend stops the runner at the cap; expected about $3, since LLM-only cost $0.019
per valid_seen episode in E11); **GPU at most 4 host-hours**; OpenRouter, Jev and OpenAI $0. If the row cannot finish
by 2026-10-09 18:00 UTC, it stops cleanly and what exists is reported with its n. `scripts/stage2_spec.py --gate` has no
entry for exploratory rows and does not apply. An "execution as run" entry follows the run.

### Deviations 2026-10-08 ~19:55 UTC: "E23: execution as run" (exploratory, descriptive; written after the run, before any paper text used it)

(a) *Timing and code.* The registration above says "~18:40 UTC"; it was committed at 18:46:17 UTC (`6a1de50`), before
any E23 episode; the code followed in `f329678` (the commit the host ran). After the launch, `c541ab6` let the local
launcher try every default-VPC subnet (as E21's and E22's drivers do); the box side is unchanged. Between E11's
commit (`08abc08`) and `f329678`, the serving and harness code changed only by additions that are off by default
(E6f's trigger cells, E13's `full_tools` key, E15's read width, the Clef engine); the `lm` serving path is unchanged,
and the host's torch / transformers / peft versions are E11's (2.14.1 / 5.18.0 / 0.21.2).

(b) *As run.* No registered type had capacity at first: us-east-1d had no g5.4xlarge in any subnet tried, and the
host is a **g5.2xlarge** (`<instance-id>`, us-east-1d, one A10G 24 GB, 8 vCPU against E11's 16), a registered
fallback, launched 18:57:29 UTC. Setup: the 134 valid_unseen games extracted from the release archives reproduced the
registered list sha256 (`3c46e12b…`); the three checkpoint files from `s3://…/checkpoints/e11_alfworld/` matched E11's
sha256 record; the base snapshot is E11's. The server took 12 minutes to come up (first read of the model from a new
disk). Two clean probe rounds (19:16 and 19:18 UTC), then the smoke (2 calibration train games × 3 arms, $0.010, no
error); the gate projected $1.01 + $0.01 against the $15 cap: "run all 134". Evaluation window 19:18:25–19:40:01 UTC,
one round: no pause, no overload error, no errored episode, so no re-run. The last stage stopped the server, synced
the outputs to `s3://…/e23_alfworld_unseen/` and powered the host off at about 19:42 UTC; its state "stopped" was
verified. Host time about 0.75 h (cap 4); Gemini $2.90 in 408 records plus $0.0006 of probes (cap $15); OpenRouter,
Jev and OpenAI $0. The GPU tariff uses the g5.2xlarge's on-demand $1.212 per hour (E11's g5.4xlarge: $1.624).
Records: `results/explore/e23_alfworld_unseen/` (box records, probe log, summary); the runs are in S3 and in the
gitignored `runs/e23_alfworld_unseen_eval`.

(c) *Readout* (`scripts/explore_e23_alfworld_unseen.py summarize` → `results/explore/e23_alfworld_unseen/
summary.{md,json}`, keys `e23/...`; `scripts/hard_ni.py` row `e23`), as registered, whatever the outcome. Flash alone
solved 88.8 % of the 134 unseen games (90.7 % of valid_seen in E11). **The cascade at E11's τ̂ solved 89.6 % (+0.7
points, paired bootstrap [-6.7, +8.2]; 14 wins, 13 losses); non-inferiority at the 3-point margin was not shown
(p = 0.175, Tango interval [-7.1, +8.7]; valid_seen: p = 0.014, shown).** It cost 92.0 % less with the GPU tariff
(99.6 % API only), took 18.2 s per game against 35.1 s (median 12.5 against 22.7 s) and 16.4 steps against 14.8, and
took 99.4 % of the actions (13 of 2,191 deferred); 37.0 % of the actions it took differed from the single gold action
(805 of 2,178; valid_seen: 35.4 %). The model alone (REPLACE) solved 88.1 % (-0.7 points; p = 0.301, not shown;
92.4 % cheaper). From valid_seen to valid_unseen, success changed by -1.9 points for Flash alone, -5.4 for the cascade
and -7.7 for the model alone (different games; unpaired). By task type (small n) the model lost on placing an object
(88 % against 100 %) and on examining one under a lamp (78 % against 100 %) and won on cleaning (94 % against 68 %) and
heating (96 % against 87 %). Of the 5 games the pilot had played, the cascade lost one that Flash alone solved.
Exploratory; no test.

### Deviations 2026-10-08 ~20:55 UTC: "E24: agent-state accuracy (L2) on public benchmarks" (exploratory, descriptive; registered before any E24 model answer)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E24)" (post-Stage-1;
marked in the paper like the other post-Stage-1 rows). It has no test, trains nothing, fits nothing, plays no episode,
revises no registered or reported result and changes no profile, threshold, key or record of any other row. Nothing
has been answered for E24: no decision model has seen an E24 question and no E24 host has been launched. Read before
registering, at $0 and without any model call: E11's and E23's registrations, records and summaries, E6a's records,
the ALFWorld adapter (`bench/suites/alfworld.py`), ALFWorld 0.4.2's hand-coded expert and its `AlfredExpert` wrapper,
E15's and E21's replay code and host drivers, `bench/stage2_corpus.py` and the stored Flash LLM-only logs named in (b)
(their options, states and logged gold labels). Implementation check, no model involved: replaying 20 of E11's logged
LLM-only episodes action by action in fresh TextWorld games reproduced every logged option list, observation and gold
action; with E11's harvest labeler (below) 18 of the 296 labels of 40 E23 episodes changed.

(a) *Why.* The author (2026-10-08): DecisionGym's in-domain agent-state level (L2, the common states) and live level
(L3) saturate for the models trained on its generator (about 99 and 100 %), which reviewers can attack. **E24 builds
the agent-state level on PUBLIC benchmarks, where the label comes from the benchmark itself and no decision model chose
the states, and scores every decision model there, so that L2 no longer rests on our own generator.** Part 1 (this
entry): L2-ALFWorld. Part 2 (L2-τ²; τ²-bench's official test split from E22's LLM-only episodes) gets its own entry
before any τ² answer, and only if E22 has committed those episodes by 2026-10-09 18:00 UTC; otherwise E24 reports
Part 1 only. E24 was designed after E11, E23, E6a, E15 and E21's L2 replays had been read.

(b) *Corpus L2-ALFWorld* (`scripts/explore_e24_public_l2.py build-alfworld` → `results/explore/e24_public_l2/alfworld/`,
committed before any answer: `questions.jsonl.gz` holds only the state and the options; labels are in
`occurrences.jsonl.gz`, never sent to a backend).
1. *Sources:* the stored logs of Gemini 3.5 Flash alone (`llm_only`, no decision model in the loop) on ALFWorld
   valid_seen (E11's evaluation run, `runs/e11_alfworld_eval/llm_only.jsonl`, 140 games) and valid_unseen (E23's,
   `runs/e23_alfworld_unseen_eval/llm_only.jsonl`, 134 games in four rooms unseen in training); the latest
   non-errored record per game; sha256 of both files recorded in `corpus.json`.
2. *States and options:* one state per logged action decision: the logged System-1 state string (the harness's
   rendering, exactly what a decision model is sent at the action point), the logged admissible actions in their
   logged order as options, and the live harness's action question "Which action should the agent take next?"; no
   option descriptions. Question hash as `stage2_corpus.question_hash`; identical questions are answered once.
3. *Replay check:* every episode is replayed in a fresh single-game TextWorld env (the release files of E6a / E23,
   `data/raw/alfworld`), executing the logged actions; the env must reproduce every logged option set and observation.
   From the first mismatch on, the episode gives no state (counted).
4. *Strict label (primary): ALFWorld's own expert, one action.* The adapter's `auto` oracle: the game's planner
   walkthrough while the executed actions are still its prefix, else ALFWorld's hand-coded expert, which re-plans from
   the current state, run as E11's harvest labeler (Deviations 2026-10-07 "E11: harvest labels": its tracker reads
   `move X to Y` as `put X to Y`, a bug of ALFWorld 0.4.2 after which its labels go stale). This is the label function
   E11's model was trained on. ALFWorld's experts name a single action, so the strict score is a lower bound. Its
   random instance draws (below) are seeded per game (sha256 of "jev-explore-E24-alfworld-replay-2026-10-08" + newline
   + task id). A state has no label, and is dropped (counted by reason), when off the walkthrough the wrapper returns its
   fallback `look` (the expert has lost track: that state and every later state of the episode, E11's rule), when the
   hand-coded expert returns its random fallback (its policy proposed the agent's last action again, and ALFWorld then
   returns a uniformly random admissible action), or when the label is not among the logged options. The logged gold
   of the live runs (E11 / E23 scored decisions against it; unpatched tracker) is kept beside it.
5. *Lenient set (secondary), justified by the expert's own code:* the strict label plus everything ALFWorld's experts
   would equally have done at that state: (i) the hand-coded expert's own next action, also while the agent is on the
   walkthrough (two experts of the benchmark); (ii) for `take O from R` and `use O`, every offered action of the same
   object class at the same receptacle (the expert draws the instance with `random.choice` among the visible objects
   of the class, and every ALFWorld goal names a class); (iii) when the hand-coded expert's action is a search
   `go to R` (FIND subgoal, location unknown), every offered `go to` of a receptacle in the same priority tier of its
   remaining search list (its list holds the class's prior receptacles, checked first, then the others; the order
   inside a tier is the room's listing order), and for a GOTO subgoal (fridge, microwave, sinkbasin) every receptacle
   of that type (it takes the first listed). The set is read from the expert by read-only instrumentation that changes
   none of its actions (tested).
6. *Strata (fixed by rule before any answer):* split (seen / unseen), task type (six), and after an off-plan step (an
   earlier executed action of the episode differed from that step's strict label, E11's definition) vs on plan.

(c) *Models and answers.* Every model answers every distinct question once, through the live harness's option handling
(`JevAgent._choice_tournament` at the action decision point: one question per request, the harness knockout above the
backend's read width, a served checkpoint's own knockout otherwise), two retries, a new replay cache per model; a
failed, refused or invalid answer is wrong (A7.10 (d)). Gold never reaches a backend (`gold=None`).
- *Hosted, from this container only:* **Jev** (`{type: jev, provider: openrouter}`, `typesafe/jev-1.13`, the Jev-only
  key, which only `decision/jev.py` reads and which never goes to AWS) and the **OpenAI Decisions API**
  (`{type: openai_decisions, model: gpt-6-luna}`, E15's backend; its key never on AWS).
- *On ONE GPU host, one model at a time, served as in its logged rows:* **E11's ALFWorld-trained 4B** (`e11-alfworld-4b`,
  the three files from `s3://…/checkpoints/e11_alfworld/` sha256-checked against E11's record, any mismatch stops that
  row; `serve_jev --adapter --calibration`); **Qwen3-4B untrained** (the same base snapshot `1cfa9a72…`, `serve_jev`
  without adapter or temperatures, as E11's offline base row); **AgentDecision-4B** as released = E12's checkpoint
  (`agentdecision-4b-mixed`, adapter sha256 `5c9542d5…`, its temperatures; trained on DecisionGym and public tool-use
  data, never on ALFWorld: transfer); **the 0.8B scoring head** (E14 arm A, joint head sha256 `a7ce7a7e…`, `serve_jev
  --backend clef` in Unsloth 2026.10.2's venv; transfer); **StartLux-Decision-4B and -9B** (the vendor's server at
  package commit `a9101e9`, weights `3890ae0c` / `342aad64`, as E21) and, if time and the caps allow and the GPU has at
  least 40 GiB, **StartLux-27B** (Q8_0 GGUF `a54b311e` via llama.cpp's CUDA image `d87b02aa`, A7.18 (g)). Order on the
  host: E11's 4B, AgentDecision-4B, the 0.8B head, Qwen3-4B, StartLux-4B, -9B, -27B. A model that does not load or
  answer is reported as not run, with the reason.
- *Reference row (not a decision model, never ranked):* Flash's own executed action on its own states (in sample).

(d) *Score and readout* (`explore_e24_public_l2.py summarize` → `results/explore/e24_public_l2/summary.{md,json}`, keys
`e24/...` in `paper/generated/e24_public_l2.tex`, appendix `paper/sections/app_e24_public_l2.tex`, label `app:e24`,
behind guarded input lines; no main-text file is edited by E24's agent; ready-to-place Table 3 cells and two candidate
sentences are handed to the coordinating session).
1. **Primary: strict accuracy, game means** (per game, the mean correctness over its labelled states; every game
   weighs the same), for valid_seen and valid_unseen separately and pooled; task bootstrap 95 % interval (B = 2,000,
   seed 0; descriptive).
2. Secondary: lenient accuracy; accuracy against the logged gold of the live runs (sensitivity; states where it is the
   fallback `look`, or after it, are left out); occurrence-weighted accuracy; by task type; after an off-plan step vs
   on plan; the mean confidence of the chosen option where the backend returns one.
3. *Live, where a row exists (quoted, not recomputed):* the model alone at the action point (REPLACE): E11's model on
   valid_seen (E11) and valid_unseen (E23), Jev on valid_seen (E6a); the cascades beside them (E11, E23, E6a) and Flash
   alone (E11, E23).
4. *Kendall's tau-b, L2 vs live (descriptive):* (i) across the (model, split) cells with a live REPLACE row (Jev seen,
   E11's model seen and unseen: n = 3, reported with its n, no reading beyond the sign); (ii) within a model, strict L2
   accuracy against live REPLACE success over its (split × task type) cells: E11's model (12 cells), Jev (6 cells), and
   the two pooled (18 cells).
5. *Reading rules* (reported whatever the outcome): the spread of L2-ALFWorld across models (max − min, strict, pooled);
   whether the ALFWorld-trained model leads, and by how much over the best off-the-shelf and the best transfer model;
   the seen-to-unseen change per model; whether the in-domain-trained models (AgentDecision-4B, the 0.8B head) keep
   their DecisionGym lead here. The label is a lower bound of a correct choice and Flash chose the states; no claim of
   live success is made for a model without a live row.

(e) *Execution and caps.* Hosted replays from this container (`--cap-usd` stops a backend at its cap): **Jev at most $3,
OpenAI at most $3**; direct Gemini $0 (no LLM call, no episode); OpenRouter other than the Jev key $0. **GPU: at most 8
host-hours on one new host** launched by `scripts/aws/e24_public_l2.sh` (Name=jev-e24-public-l2; image
`<aws-resource-id>`, E3's clone), a g6e.2xlarge (one L40S 48 GB) in any us-east-1 subnet, else a g5.2xlarge (one
A10G 24 GB: StartLux-9B, which needs 32 GiB, and the 27B are then not run, and the readout says so); SSM only; no API
key of any kind on the host; a power-off guard at 470 minutes of host time, a 30-minute idle stop, the host stopped
when done (verified); the driver refuses every host it did not create and the never-touch instances
(`<instance-id>`, `<instance-id>`, E21's and E22's hosts). The same host, stopped in between, serves Part 2
if it is registered (the 8 host-hours cover both parts). Answers to `s3://…/e24_public_l2/` and the repository.
Deadline: 2026-10-09 18:00 UTC for Part 1; a stage cut by a cap or the deadline is reported with what exists.
`scripts/stage2_spec.py --gate` has no entry for exploratory rows and does not apply. An "execution as run" entry
follows.

### Deviations 2026-10-08 ~21:15 UTC: "E22: the task as the unit of analysis, the label wording and a label audit" (exploratory; amends E22 (i) before any E22 test-task episode; review round 23, P0-4)

Written at the coordinating session's request after review round 23 (`docs/REVIEW_ROUND23_EVAL.md`, P0-4), while E22's
model trains; no E22 test-task episode has run and no E22 model has answered a test-task question. Nothing else in
E22 changes (tasks, labels, model, thresholds, harness, arms, k, caps).

(a) *Unit of analysis: the test task.* The 60 test tasks are the units; the k trials of a task are clustered within
it, never counted as independent episodes. Amends E22 (i):
1. **Success** per arm = the mean over tasks of the per-task mean success over its k trials (equal to pass^1 when
   every task has k completed trials). Δ = the mean over tasks of the paired per-task difference; its 95 % interval
   from the paired task bootstrap (tasks resampled with their trials; B = 10,000, seed 0).
2. **The non-inferiority verdict** is the paper's rule on the tasks, as the registered τ² test C7 applied it with
   several trials (A6.14 / A6.16 (e)): a task's outcome is majority pass (more than half of its k trials succeed), and
   `metrics.ni_certified_p(wins, losses, n_tasks, 0.05)` (A6.21, rule `bb_ordered`) on the paired task outcomes decides
   at α = 0.05 ("non-inferior" iff p ≤ 0.05, else "not shown"); pooled over retail and airline as the verdict, per
   domain descriptive. Beside it, descriptive: the one-sided 95 % lower bound of the paired task bootstrap of the
   per-task mean difference against −5 points, and the certified p on the (task, trial) pairs that E22 (i) named
   (which treats trials as independent and is no longer the verdict).
3. Cost and time per arm are per-task means of the per-episode values (trials averaged within a task); the saving is
   1 − (mean arm cost) / (mean LLM-only cost) on the paired tasks, with a paired task-bootstrap interval.
A task with an errored trial in either arm keeps its other trials (both arms use the trials completed in both); a task
without any such trial leaves the pair and is counted.

(b) *Wording.* The offline accuracy on held-out training tasks is reported as **agreement with Flash's choices in
successful episodes** (and, for the failed-episode states, with the matched reference call), never as accuracy
against gold: E22's labels are the agent's own choices filtered by outcome.

(c) *Label audit, $0, from the stored harvest logs.* 200 labelled tool-choice states, stratified by label source:
100 drawn from successful episodes and 100 from failed episodes (exact reference matches), by a seeded sample
(`random.Random(0)` over the states in record order, per stratum), from fit and dev tasks. For each: whether the
labelled call matches an action of the task's expected-action list (name and compared arguments, tau2's
`compare_with_tool_call`; only for labels that are tool calls), whether the labelled tool appears in that list at all,
and agreement with the adapter's reference next step where that names a tool (a heuristic, `_Reference.next_step`).
Reported per stratum with counts, plus label coverage: the share of logged tool-choice states that received a label
(with the drops by rule of E22 (d)). The audit uses no model call and changes no label.

(d) *The retail judge.* E22 grades retail's natural-language assertions with Gemini 3.5 Flash
(`--jev-nl-judge gemini/gemini-3.5-flash`, temperature 0, JSON output, minimal thinking) instead of τ²-bench's fixed
gpt-4.1-2025-04-14, because E22's OpenAI cap is $0 (E22 (c)). The same judge grades every E22 retail simulation; the
deviation from the official protocol is stated in the execution-as-run entry, in the appendix and in Table 1's note.

### Deviations 2026-10-08 ~21:30 UTC: "E21b: the deployed confidence-gated cascades on DecisionGym-Shift" (exploratory, descriptive; registered before any E21b episode, model answer or host launch; review round 23, P0-3)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E21b)" (post-Stage-1;
marked in the paper like the other post-Stage-1 rows). It has no confirmatory test, trains nothing, revises no
registered or reported result and changes no profile, threshold, key or record of any other row (E21's included).
Nothing has run for E21b: no E21b host has been launched, no decision model has answered an E21b question and no
E21b episode exists. Read before registering, at $0: E21's registration and committed code (`ac3bdad`, `3ec1295`,
`e5d9de0`), E21's working-tree summary and its live logs of the 200 L3 tasks (`runs/e21_live_local`: LLM-only solved
200 of 200 at $0.0535 and 8.0 s per episode; REPLACE at the tool choice: the frozen 4B 96.0 %, E12 91.5 %, E14 arm A
17.5 %), review round 23 (`docs/REVIEW_ROUND23_EVAL.md`, P0-3), E17/E18's and E20's registrations and records, and
the logged in-domain final-harness rows (`runs/harness_final_dedicated`, `runs/e18_live`). E21b was designed after
E21's REPLACE results had been read; whatever it shows is reported.

(a) *Why.* The deployed system of the paper is a cascade gated by confidence (the final harness: tool + argument
cascade at the fitted thresholds); E21 measured the decision models on its shifted split only with REPLACE at the
tool choice (every tool choice taken from the model). **E21b asks whether the confidence gate protects success under
shift: at their in-domain thresholds, with no refit, do the deployed cascades keep the agent's success on E21's 200
shifted L3 tasks, and how much do they still take and save?**

(b) *Arms* (the final harness: tool + argument cascade, reflection off, argument questions batched per step; each
model alone on the host's GPU, no System-1 cache):
1. **`rl_final`**: the frozen JevAgentRL-4B (the Stage-2 / final-harness checkpoint; sha256-checked against
   `results/stage2/rl_eval_plan.json` row `frozen` before serving), `training/serve_jev.py --fast` with its adapter
   and temperatures, profile `configs/harness/jevagent_final.yaml` (sha256 `d8ad6549…`) unchanged, backend
   `jevagent_rl`: τ = 0.4999999999978404 at the tool choice and 0.9999998807907104 at the arguments.
2. **`e18_final`**: E18 = E14 arm A (`s3://…/checkpoints/e14_small/A/final/`, `joint_head.safetensors` sha256
   `a7ce7a7e…`) served with the logistic defer head (`serve_jev --backend clef --clef-defer-head
   results/explore/e17_e18/e18_head.json`, sha256 `23dbbbee…`) as in E18's live row, profile
   `configs/harness/explore_e18_final.yaml` (sha256 `8360c9e5…`) unchanged, backend `e18`: τ = 0.9941678 at the tool
   choice and 0.9965348931475904 at the arguments (1 − k_dp; the head's feature ln(number of options) reads the
   shifted option count, 129 at a tool choice instead of 69, as the deployed head would).
3. **`e12_final`**: E12 = the released AgentDecision-4B (`s3://…/checkpoints/e12_mixed/`, adapter sha256
   `5c9542d5…`, temperatures `calibration.json` sha256 `5b027abf…` = E12's record), `serve_jev --fast` with its
   adapter and temperatures, profile `configs/harness/explore_e21b_e12_final.yaml` = a copy of `jevagent_final.yaml`
   plus `backends.e12`, written with `status: pending` (the runner skips a pending profile). **E12 has no fitted
   cascade threshold** (its live rows were REPLACE). Its in-domain thresholds are fitted now, before any E21b episode
   of this arm, by the procedure that fitted the frozen 4B's (and E17's, E19's and E20's): the 2,719 System-1
   questions of the frozen 4B's calibration arm `rl_shadow_cal` (`runs/e2a_jevgym_shadow_cal_rl`: Flash, the in-domain
   calibration seeds 5000–5149; rebuilt by `scripts/e5_cal_replay.py build`, sha256 `d46701e8…` = E14's and E20's
   file) answered once by the served E12 through the harness's asking path (`e5_cal_replay.py answer`), written into a
   copy of the arm (`derive`) and fitted per decision point by the registered command (`threshold_theory
   --calibration-arm --task-value 1.0 --lam-per-hour 0 --v-fallback W`; `e5_cal_replay.py fit`). No shifted state and
   no evaluation seed is used. The fitted values are written into the profile (`status: development`) and committed
   with an amendment "E21b: E12's fitted thresholds" before any `e12_final` smoke. A point whose fit is "always defer"
   is not delegated (System-1 is not asked there, A6.11 (c)); if the tool choice always defers, the row would equal
   LLM-only (E20 (c)), so it runs instead at the frozen 4B's thresholds of `jevagent_final.yaml` and is labelled so.

(c) *Tasks and settings.* E21's 200 L3 tasks (seeds 330100–330299; the suite block of
`configs/explore/e21_shift/live_*.yaml`, `shift: true`: surface, distraction and structure all on), the same task ids
as E21's logged `llm_only` row. Gemini 3.5 Flash on the direct Gemini API (temperature 0, minimal reasoning),
`JEV_EVAL_RUNS=1`, **1 worker** and task-major schedule seed 732 (the dedicated final-harness rows and E18's row), one
run per arm (`runs/e21b_<arm>`), configurations `configs/explore/e21b_shift_cascade/live_<arm>.yaml`. Arm order on
the host: `rl_final`, `e18_final`, `e12_final`. Before each arm a smoke on 2 development tasks of E21's development
block (never reported): 339020–339021 (`rl_final`), 339022–339023 (`e18_final`), 339024–339025 (`e12_final`).
**Reference: E21's logged `llm_only` row, not re-run** (`runs/e21_live_local/llm_only.jsonl`, 2026-10-08
~16:00–16:19 UTC, from this container with 4 workers; its slim copy `results/explore/e21_shift/live/
e21_live_local.json` once E21 commits it); the same tasks, paired per task. E21's REPLACE rows of the same models
(`rl_ts_replace_f`, `e12_ts_replace_f`, and `e14a_ts_replace_f` = arm A without the head) are read beside, paired on
the same tasks (descriptive: what the gate changed against taking every tool choice).

(d) *Secondary row, lowest priority:* **`e12_final_in`**, E12's cascade at the same fitted thresholds on the 200
in-domain tasks of the dedicated final-harness rows (seeds 300500–300699, `runs/harness_final_dedicated`'s suite block,
1 worker, task-major seed 732; smoke on development seeds 918–919), paired with that run's logged `llm_only` row, so
that arm 3 has an in-domain live coverage beside its shifted one (arms 1 and 2 have theirs: `harness_final_dedicated`
and `e18_live`). It runs only after the three shifted arms, and only if the Gemini gate of (e) and the GPU cap of (g)
leave room for all 200 tasks; otherwise it is reported as not run.

(e) *Smoke, gate and overload.* Per arm, after its smoke: spend so far + 1.2 × the smoke's Gemini $ per episode ×
200 ≤ the $35 cap of (g), else the arm runs its first k tasks that fit (reported with its n), else it is not run.
E14's pause-and-resume rule (E21 (g)): probe rounds of six one-word Flash calls on the direct API (logged to
`results/explore/e21b_shift_cascade/box/api_probe_log.txt`); a configuration (smokes included) starts after two
consecutive clean rounds; a watchdog pauses the runner on an episode that errored with HTTP 503 / overload / 429 and
resumes after two clean rounds (completed tasks are skipped); after every task has a record, errored episodes are
re-run once; no other route.

(f) *Readout* (`scripts/explore_e21b_shift_cascade.py summarize` → `results/explore/e21b_shift_cascade/
summary.{md,json}`, keys `e21b/…` in `paper/generated/e21b_shift_cascade.tex`), whatever the outcome; latest record per
task, errored tasks left out of the pairs and counted. Per arm against E21's `llm_only` on the common tasks:
1. **Success**, the paired Δ (Newcombe interval; paired task bootstrap, B = 10,000, seed 0), wins / losses, and
   **non-inferiority at 3 points by the paper's rule** (`metrics.ni_certified_p(wins, losses, n, 0.03)`, A6.21 rule
   `bb_ordered`; "non-inferior" iff p ≤ 0.05, else "not shown"; Tango 95 % interval), as Table 1 applies it.
2. **Cost per episode**: Gemini as logged plus the GPU tariff q = $1.624 per hour × the episode's wall-clock (the
   convention of the in-domain final-harness and E18 rows: one A10G, g5.4xlarge on demand); sensitivity rows with the
   E21b host's own on-demand price and with q = 0. Saving = 1 − mean arm cost / mean `llm_only` cost (paired task
   bootstrap).
3. **Wall-clock** per episode, mean and median, and its paired difference (descriptive: the reference ran from this
   container with 4 workers, the arms on the host with 1 worker; the in-domain rows' times are quoted beside).
4. **Coverage**: the share of tool choices and of argument decisions System-1 took (`harness_final._episode_live`),
   beside the same arm's in-domain coverage (`rl_final`: `harness_final_dedicated`; `e18_final`: `e18_live`;
   `e12_final`: `e12_final_in` if run, else its calibration-state coverage at τ̂) and E21's REPLACE row; "handed back"
   = the share of tool choices deferred to the LLM.
5. **Accepted wrong decisions**: accepted (not deferred) System-1 tool choices outside the answer key's acceptable set
   and accepted System-1 arguments that differ from the oracle's (`bench/metrics.is_correct`, only where the answer key
   labels the decision), counts and shares, at must-ask states separately (the acceptable set is the ask tool alone);
   the LLM's accuracy on the deferred decisions.
6. Steps, System-2 calls and System-1 requests per episode; success by task kind (n = 20 per kind; descriptive).
*Registered reading* (descriptive, per arm): "the gate protects success under shift" iff non-inferiority at 3 points
is shown by 1; "it still saves" iff the saving point estimate of 2 (GPU tariff included) is above 0; an arm that takes
fewer than 10 % of the tool choices is described as deferring almost everything (success kept by deferral is not a
deployment result). The paired difference cascade − REPLACE on the same tasks is read beside, descriptively.

(g) *Caps and infrastructure.* **Direct Gemini at most $35 for E21b in total** (probes, smokes, live rows; a watchdog
on the logged spend of E21b's runs plus the probes stops the runner at the cap); **GPU at most 4 host-hours on one new
host**; OpenRouter, Jev and OpenAI $0 (no call). The host is launched by `scripts/aws/e21b_shift_cascade.sh`
(Name=jev-e21b-shift-cascade; image `<aws-resource-id>`, E3's clone; a g5.2xlarge with one A10G 24 GB, the GPU of
the in-domain rows, else a g6e.2xlarge; every us-east-1 default subnet tried), SSM only, a power-off guard 230 minutes
after launch, a 30-minute idle stop, the host stopped by its last stage and its state verified. The driver refuses
every host it did not create and never touches `<instance-id>`, `<instance-id>` or E21's, E22's, E23's
and E24's hosts. The Gemini key comes from Parameter Store into the live stages' process only, is never printed or
written; no other key is on the host. Records to `s3://…/e21b_shift_cascade/` and slim copies to
`results/explore/e21b_shift_cascade/`. Deadline 2026-10-09 12:00 UTC; an arm cut by a cap or the deadline is reported
with its n. `scripts/stage2_spec.py --gate` has no entry for exploratory rows and does not apply.

(h) *Paper.* A short appendix paragraph with one small table (`paper/sections/app_e21b_shift_cascade.tex`, label
`app:e21b`, every number a `\res{}` key) behind guarded input lines in `paper/main.tex` and
`paper/sections/appendix.tex`; no main-text section is edited by this agent (candidate sentences go to the
coordinating session). An "execution as run" entry follows the runs.

### Deviations 2026-10-08 ~21:25 UTC: "E24: consequence labels lead" (exploratory; amends E24 (b) and (d) before any E24 model answer)

Registered before any E24 answer: no decision model has answered an E24 question, no E24 host stage has run (the host
`<instance-id>`, a g5.2xlarge since us-east-1 had no g6e.2xlarge, was launched at 20:50 and stopped at 21:01
UTC without any stage). Facts: the E24 entry above was committed at 20:42:31 UTC (`a7981f8`), not ~20:55 as its header
says. The corpus was then built by the registered rule ($0, no model): 274 games, 3,824 logged states, every option
list and observation reproduced by the replay (0 mismatches); 3,355 labelled states (valid_seen 1,660, valid_unseen
1,695; 3,263 distinct questions); dropped 327 after the expert's fallback and 142 at its random fallback; the strict
label equals the logged gold at 86.7 % (seen) and 82.2 % (unseen) of the states; Flash's own action equals the strict
label at 48.3 % and 41.7 %. Review round 23 (`docs/REVIEW_ROUND23_EVAL.md`, P1-2) and the coordinating session asked to
lead with an acceptable-set label from a consequence check, with the single expert action reported only as a lower
bound (Flash agrees with it on under half of its states while solving about 90 % of the games).

(a) *Acceptable set (the lead label from now on).* A(s) = the lenient set of (b) 5 plus every offered action a after
which ALFWorld's hand-coded expert needs fewer steps to win: H(s + a) < H(s). H(s) is measured by replaying the
episode's logged actions up to s in a fresh TextWorld game (the corpus's replay), then letting the hand-coded expert
(E11's labeler with E24's read-only instrumentation) act until the game is won, at most 60 steps; at the first state
the expert's opening `look` (it only re-reads the room) is not counted. H(s + a) is measured the same way after
executing a, with at most H(s) − 1 expert steps; an action that wins the game is accepted. A rollout in which the
expert falls back (`look`, its random fallback) or raises counts as not winning. Where H(s) is undefined (the expert
does not win from s within 60 steps), A(s) is the lenient set. The expert's random draws are seeded per (game, state,
action) from E24's replay salt. This is the review's "the expert's remaining plan shrinks after the action".
**ALFWorld's PDDL planner is not used:** it plans with the hidden object locations (privileged), and its search took up
to 10 s for one state in the implementation check, which the time and the hosts do not allow for every pair.
(b) *Evaluation.* Membership is a function of the state and the action alone, so it is evaluated for every
(state, action) pair that a scored row chose and that is not already in the lenient set (all models' answers and
Flash's own action), after all answers exist, blind to which row chose it (`explore_e24_public_l2.py consequence`, on
the E24 host's CPUs with alfworld 0.4.2 / textworld 1.7.0 as the corpus build, or in this container). Set sizes are
reported, descriptively, on a seeded sample of 40 states per split whose every option is evaluated (sha256 of
"jev-explore-E24-alfworld-setsize-2026-10-08" + newline + task id + newline + step, the first 40 per split).
(c) *Readout order.* L2-ALFWorld accuracy against A(s) is the lead number (game means as registered, seen / unseen /
pooled, with its bootstrap interval); the lenient set of (b) 5 and the strict single action follow, the strict one
labelled "agreement with the expert's action (a lower bound)". Everything else in E24 (corpus, models, Kendall, caps,
reading rules, now read on A(s)) is unchanged; the Kendall statistics are also reported on the strict label.

### Deviations 2026-10-08 ~23:55 UTC: "E22: execution as run" (exploratory, descriptive; written after the runs, before any paper text used them)

(a) *Times and order.* Registration `8c8490d` (16:18:53 UTC; its header says ~16:10, the commit time is
authoritative) before any E22 simulation; the adapter keys and scripts `ec1be5c` (16:21:10); the harvest smoke 16:21-16:22
(two fit tasks per domain, one trial; 4 of 4 reward 1); the gate record `ecdaa46` (16:30:09: 0.07 USD per simulation,
projection 141.5 of 200 USD, so K = 8, k = 4 and the optional arm); the harvest 16:24-17:07 in this container; the
records `3570c81` (17:08:26). The GPU host `<instance-id>` (g5.2xlarge, us-east-1c; no g6e.2xlarge or g5.2xlarge
capacity in the other subnets) was launched at 16:31:41 and installed its software 16:33-16:36; the chain sent at
17:08:54 ran data, train (17:09-21:27), replay (21:27-21:34), fit (21:34), smoke (21:34-21:36), eval (21:36-21:58:
retail 21:36-21:50, airline 21:50-21:58, the three arms concurrently) and final, and the host powered itself off at
about 22:00. The amendment on the unit of analysis `bc70694` (21:09:43) preceded every test-task episode. The
coordinating session's AWS login expired at about 21:30 and came back at 23:41; the host ran its chain alone in
between, and its records were fetched from S3 afterwards.

(b) *As run, against the registration.*
- Harvest: 832 simulations (retail 74 × 8, airline 30 × 8, seed 2201), none ended in an error, so `rerun-errored`
  re-ran nothing (tau2's own retry re-ran one airline simulation, whose side-car keeps both records). Success on the
  train tasks: retail 395 / 592, airline 175 / 240. **Correction:** the commit message of `3570c81` misstates two counts
  and one fact: 7,116 of 9,959 tool-choice states were labelled (6,457 from successful episodes, 659 exact reference
  matches from failed ones), not "9,986" / "8,231", and no simulation was re-run for an error (not "2"). The records
  and their sha256 in that commit are right: fit 11,096 (5,426 tool choices, 5,670 arguments), dev 2,940
  (1,504 / 1,436), 11.9 % of the argument targets `<OTHER>`.
- Training: arm A's checkpoint was continued as registered (no fallback); no record was cut at 8,192 tokens; peak
  8.4 GB reserved; epoch 1 took 109.4 minutes, and the second epoch ran under the rule (2.73 host hours used, projection
  7.56 ≤ 9.5) and took 108.1; dev agreement with Flash's choices 88.6 % after epoch 1 and 89.9 % after epoch 2, so epoch
  2 was used; 36.1 % for arm A before E22's training; after Unsloth's `calibrate` ECE 0.021. The final checkpoint is
  in `s3://…/checkpoints/e22_tau2/final/` (head sha256 `fbc4278c…`).
- Thresholds (`results/explore/e22_tau2/box/cal/`): the served model answered the 2,940 dev questions in 6.3 minutes
  (agreement with Flash: tool choices 84.6 %, arguments 95.5 %); the registered command gave τ̂ = 0.99446 at the tool
  choice (997 paired decisions on which the model answered a tool; ΔC 0.00589 USD, V = W = 1 USD by the fallback) and
  0.99801 at the arguments (1,436; ΔC 0.00386), both by the projection rule; at τ̂ the model would have taken 13.3 % of
  the dev tool choices and 49.8 % of the arguments with no disagreement. `configs/explore/e22_tau2/final.yaml` was
  written by the fit on the host at 21:34 (its copy and sha256 in the records) and committed with this entry.
- Evaluation: the smoke (two dev tasks per domain, one trial, `e22_final`) cost 0.0788 USD per simulation; the gate's
  projection 143.0 ≤ 200 USD kept all three arms at k = 4. Every arm completed all 240 simulations (60 tasks × 4) with
  no error; nothing was paused, no probe failed, nothing was re-run. Retail's NL assertions were judged by Gemini 3.5
  Flash in every E22 run (harvest, smokes, evaluation), not by τ²-bench's gpt-4.1-2025-04-14 (the deviation of
  E22 (c), stated in the appendix and in Table 1's note); the user simulator was Gemini 3.5 Flash-Lite.

(c) *Caps and host.* Direct Gemini: 120.69 USD of 200 (harvest 68.68, harvest smoke 0.28, evaluation smoke 0.32,
evaluation 51.42: `llm_only` 19.56, `e22_final` 18.41, `e22_tau05` 13.46), from the logged ledgers plus the
user-simulator tokens priced at Flash-Lite's list price and 0.003 USD per retail simulation for the judge. GPU: about
5.5 host hours of 10 (16:31:41 to the power-off at about 22:00; the driver counted 5.44 at its final stage). OpenRouter,
Jev and OpenAI: 0. **The host is stopped** (state `stopped`, read at 23:42). The label audit of the amendment cost 0.

(d) *Readout* (`scripts/explore_e22_tau2.py summarize` → `results/explore/e22_tau2/summary.{md,json}`, keys `e22/...`
in `paper/generated/e22_tau2.tex`), the task as the unit (60 tasks, trials averaged within a task):
- **Fitted thresholds (`e22_final`):** success 54.2 % against 66.2 % for LLM-only in the same window, Δ −12.1 points
  (paired task bootstrap [−19.2, −5.4]); retail 52.5 vs 64.4, airline 57.5 vs 70.0. **Non-inferiority not shown**
  (certified p on per-task majority pass 0.979: 2 wins, 12 losses; the (task, trial) pair p 0.986, descriptive). The
  model completed 3.9 % of the agent's turns without the LLM (111 calls: 93 order lookups, 9 reservation lookups,
  6 user lookups by e-mail, 3 writes), took 21.0 % of the argument questions it was asked; cost 4.6 % lower with the
  GPU charged (q = 0.09 cents per episode; 5.8 % API only), time per episode 26.9 s against 19.8 s (7.9 s of model time
  per episode: 14 requests queued on one GPU for twelve concurrent episodes).
- **τ = 0.5 (`e22_tau05`, optional arm):** 59.2 %, Δ −7.1 [−13.8, −0.4], NI not shown (p 0.654); 39.0 % of the turns
  completed by the model (1,141 calls, 70 of them writes), 31.3 % cheaper, 28.0 s per episode.
- **Post hoc diagnostics (descriptive, 0 USD):** of the 22 test tasks on which `e22_final` solved fewer trials than
  LLM-only, 13 (22 of the 36 lost trial successes) had no call from the model at all, i.e. the harness sent exactly
  LLM-only's requests there; the arms' conversations already differ at the simulated user's first message, so they pair
  by task and trial seed only. On the harvest, splitting each training task's eight LLM-only trials into two random
  halves moves the mean success over 60 tasks with a standard deviation of 3.6 points (95 % range ±7.1). The gap is
  larger than that and arose mostly where the model never acted; its cause was not identified. A repeat of the two arms
  (about 35 USD of Gemini and one GPU hour, within the caps) would be needed to separate it from run-to-run variation;
  it is not part of this registration and was not run.
- Reading against the goal (the author's, 2026-10-08): on τ²-bench the trained 0.8B model did **not** reach Flash's
  success and saved little at its fitted thresholds; the lower threshold saved a third of the cost but lost success.

### Deviations 2026-10-09 ~00:10 UTC: "E21b: E12's fitted thresholds" (exploratory; written after E12's calibration answers and fit of E21b (b) 3, before any `e12_final` smoke or episode)

(a) *Times and host.* The registration was committed at 21:22:26 UTC (`53b1de8`; its heading says ~21:30), the code at
`e8bb78d`. The host `<instance-id>` (g5.2xlarge, one A10G 24 GB, us-east-1c; tag `jev-e21b-shift-cascade`) was
launched at 21:29:37 UTC; AWS credentials expired at about 21:34 before any chain was sent, and the host, idle, shut
itself down at about 22:30 (instance-initiated; the image's own idle stop; CloudWatch has no datapoint after the 22:25
bin). Credentials returned at 23:41; the host was started again at 23:42:11 and, as the coordinating session confirmed,
the idle hour counts against the 4 host-hour cap: the guard was re-armed for 172 minutes from the restart (power-off by
about 02:35 UTC, about 3.9 host-hours in all). The chain `rearm setup cal_e12 live_rl live_e18` was sent on `f97f2c4`
(the branch head; E21b's code unchanged since `e8bb78d`): setup 23:42–23:49 (the frozen 4B, E12's adapter and
temperatures, E14 arm A's head, E18's defer head and the calibration questions all matched their sha256).

(b) *E12's answers and fit* ($0; no LLM call). E12 served by `serve_jev --fast` (adapter + temperatures) answered all
2,719 calibration questions (`d46701e8…`) through the harness's asking path, 23:50:05–23:59:55 UTC, no error
(`results/explore/e21b_shift_cascade/cal/answers_e12_cal.jsonl.gz`). `derive` and the registered command
(`e5_cal_replay.py fit`, projection rule) gave **tool choice τ̂ = 0.44503032941063925** (1,125 paired decisions) and
**arguments τ̂ = 0.9958731532096863** (1,328 paired). Neither is "always defer", so the fallback of (b) 3 does not
apply. At τ̂ on the calibration states E12 takes 100.0 % of the tool choices and 83.7 % of the arguments (1,112 of
1,328), with no accepted error (the frozen 4B's own fit: 100 % and 77.5 %). The values are in
`configs/harness/explore_e21b_e12_final.yaml` (`status: development`, `backends.e12`, with the fit files' sha256) and
`results/explore/e21b_shift_cascade/cal/fits.json`, committed right after this entry. No shifted state, evaluation seed
or E21b episode entered the fit; nothing else changes.

### Deviations 2026-10-09 ~00:10 UTC: "E21 addendum: shift live rows for every model" (exploratory, descriptive; registered before any episode of these rows)

Addendum to E21 ("E21: a shifted DecisionGym split (DecisionGym-Shift)"). It is exploratory, has no test, sits outside
every Holm table and revises no registered or reported row; the rows of E21 (e) stand as run. Requested by the
coordinating session for the author (2026-10-09 ~00:00 UTC): E21's shift L2 and shift L3 become the L2 and L3 columns of
the main-text Figure 4b, so every model in it needs a shift live row. **Read before this entry:** every E21 row that has
run (the L2 answers of every model but StartLux-27B, whose main replay is still running on the host; the seven live rows
of E21 (e)). No episode of the rows below exists.

(a) *Rows.* REPLACE at the tool choice for the models that have no E21 live row, **in the coordinating session's priority
order: OpenJev-27B (HF, FP8), StartLux-27B (Q8_0), StartLux-4B, Intern-Decision-4B, StartLux-4B + recipe (E12c)**. Each
row is E21 (e) unchanged: the same 200 L3 tasks (seeds 330100-330299), Gemini 3.5 Flash on the direct API (temperature
0, minimal reasoning), 4 workers, a new System-1 cache, the model served exactly as for its E21 L2 replay
(`configs/explore/e21_shift/backends.yaml`), a smoke on 2 development tasks first (seeds 339014-339023, two per row,
never reported), E21 (e)'s gate, (g)'s pause rule and one re-run of infrastructure errors; no row is chosen by an
outcome. Configs `configs/explore/e21_shift/live_{openjevhf,sl27b,sl4b,intern,e12c}.yaml` and their smokes; host stage
`l_<slug>` of `scripts/aws/e21_shift.sh` (serve, smoke, gate, the row, stop). Keys `e21/<slug>/l3/...` as E21's.

(b) *Order on the host and time box.* StartLux-27B is served when its L2 main replay ends, so its row runs first (no
reload), then OpenJev-27B, StartLux-4B, Intern-Decision-4B and E12c. StartLux-27B answers one request at a time through
llama.cpp (about 5.5 s per 129-option question on the replay), so its row is **time-boxed**: the runner is stopped
when the minutes left before the power-off guard fall to 140 (the reserve for the four other rows and `final`); the
tasks it completed are its row, reported with their n (the runner works task by task in seed order, so they are the
first n tasks). If a later row would not fit before the guard, the rows are cut from the end of the priority list
(E12c first), and the readout says so. StartLux-27B's two re-renderings stay skipped (E21 (d) 5).

(c) *Caps.* Direct Gemini: **at most $25 for these five rows** (smokes included; observed $0.016-0.023 per REPLACE
episode), so E21 stays within its $70; the host's watchdog cap rises from $32 to $43 to cover them. GPU: **within E21's
12 host-hours** on the same g6e.2xlarge `<instance-id>` (its power-off guard, armed for 710 minutes after launch,
is unchanged); the host is stopped when the rows are done. No other key, provider or host. Everything else is E21's.

(d) *Readout.* As E21 (i): `scripts/explore_e21_shift.py summarize` adds the rows to `summary.{md,json}` and the keys
(`e21/<slug>/l3/success`, `.../l3/n`, per kind, Newcombe intervals against LLM-only and Jev); E21's Kendall rows over
in-domain and shift live success then include every model with both. The "execution as run" entry covers E21 and this
addendum.

### Deviations 2026-10-09 ~01:35 UTC: "E25: the decision model in two more open-source agent harnesses (pi on Terminal-Bench 2.0, SWE-agent on SWE-bench Verified)" (exploratory, descriptive; registered before any E25 episode)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E25)" (post-Stage-1; marked
in the paper like the other post-Stage-1 rows). It trains nothing, fits nothing, tunes no threshold, revises no
registered or reported result (H5, E4, E4b and E4r stand as reported) and changes no profile, key or record of any other
row. **Nothing has run for E25:** no E25 host exists, no smoke or evaluation trial has started and no model call has been
billed. Read before registering, at $0 and without any model call: the registrations and readouts of H5 (its TB2 smoke and
its SWE-bench `route` arm), E4, E4b and E4r; the pi adapter (`adapters/pi/`, `jev-router.ts`) and pi-ai 0.99.2's model
catalog (Gemini 3.5 Flash $1.50 / $9.00 per M input / output, cache read $0.15; Gemini 3.5 Flash-Lite $0.30 / $2.50, cache
read $0.03; the same as `configs/prices/models.yaml`); Harbor at git `4f848e3` (its `pi` and `swe-agent` agents, the model
connection code and the registry entries `terminal-bench@2.0`, 89 tasks, and `swebench-verified@1.0`, 500 tasks);
SWE-agent v1.1.0's source (model classes, default config).

(a) *Why.* The author (2026-10-09): "DecisionAgentHarness's contribution should be that strong open-source agent
harnesses, once our mode is added, save cost and time stably, with success kept. Verify this, so the experiments support
that conclusion." So far one open-source harness (mini-SWE-agent; E4, E4b, E4r) carries that claim. **E25 adds the same
mode, per-step routing where the decision model decides whether each LLM step needs the strong LLM, to two more
open-source harnesses on their own benchmarks, each against the unmodified harness on the same tasks, and reports what
it finds whatever the outcome.** E25 was designed after H5, E4, E4b and E4r had been read.

(b) *Shared design (both harnesses).*
1. *LLMs (direct Gemini API, key from SSM `<ssm-prefix>/GEMINI_API_KEY`).* Strong: Gemini 3.5 Flash, minimal thinking (the
   paper's agent LLM). Fast: Gemini 3.5 Flash-Lite, minimal thinking (H5's routing target).
2. *Decision model.* Jev through OpenRouter (`typesafe/jev-1.13`) behind the host decision server
   (`python -m jevharness.decision.server --name jev-1.13 --s1 '{"type":"jev","provider":"openrouter"}'`, bound to
   0.0.0.0:8090, **no System-1 cache**), key from SSM `<ssm-prefix>/OPENROUTER_API_KEY`; task containers reach it as
   `host.docker.internal:8090` (`adapters/pi/harbor_host_gateway.yaml`). Before each LLM request of the routed arm it
   answers one yes/no question (below); p(hard) ≥ τ, or no answer (fail-open), sends the request to the strong LLM, else
   to the fast one. **τ = 0.5**, the registered τ of every earlier routing row (H5 `route`, E4, E4r); no development
   phase and no tuning.
3. *Arms (the same tasks, paired by task).* `baseline`: the unmodified harness with the strong LLM on every step.
   `jev_router`: the same harness plus the routing mode (DecisionAgentHarness). `always_cheap`: the unmodified harness
   with the fast LLM on every step, a descriptive lower bound that runs only when the gate's projection (e) fits.
4. *Execution.* Harbor at git `4f848e3` (`scripts/aws/hard_setup.sh`'s install) on **one new CPU host per harness**
   (m7i.8xlarge, fallbacks m6i.8xlarge / c7i.8xlarge; Ubuntu 24.04, 500 GB gp3, profile `<instance-profile>`, SG
   `jev-train-noingress`, SSM only, no inbound; tags `Name=jev-e25-tb` and `Name=jev-e25-swe`; a power-off guard and a
   30-minute idle stop armed from boot; driver `scripts/aws/e25_harnesses.sh`). The arms of a harness run as parallel
   Harbor jobs over the same task list in the same period, 5 concurrent trials per arm, k = 1. Harbor re-runs an
   infrastructure error up to twice (`--max-retries 2 --retry-exclude AgentTimeoutError`, as H5); a trial that still
   errors counts as a failure in the full set. Keys reach the harness only through Harbor's own environment resolution
   (never `--ae` or an env file); every phase ends with a leak check (files that contain a key value; must be 0).

(c) *Harness A: pi on Terminal-Bench 2.0.* pi 0.99.2 (`harbor_jev_pi:JevPi`, `--thinking minimal`) on
`terminal-bench@2.0` (89 tasks, task repository `69671fb`), each task's own agent timeout and **a 100-turn cap for every
arm** (`--ak max_turns=100`, as H5's TB-science runs; a budget cap, stated in the paper). `baseline` =
`google/gemini-3.5-flash` with no extension (plain pi through the same wrapper). `jev_router` = pi's virtual model
`s1/router` from `adapters/pi/jev-router.ts`, unchanged (strong `google/gemini-3.5-flash`, fast
`google/gemini-3.5-flash-lite`, its question "Does the next step require careful multi-step reasoning, planning, or
recovery from an error (i.e., a strong model) rather than a routine action?", asked before every request; its other
settings at their defaults: two failed tool results among the last four escalate to the strong LLM without asking,
a retry after a failed fast request escalates, `direct` requests go to the fast LLM). The only adapter change before
E25: Harbor resolves no credentials for the virtual `s1` provider, so `JevPi` resolves the Gemini key for it exactly as
Harbor does for `google` (no key in any argument or file). `always_cheap` = `google/gemini-3.5-flash-lite`, no extension.
Smoke: `prove-plus-comm` and `overfull-hbox` × 3 arms (H5's TB2 smoke tasks; development only, never reported; the
evaluation runs every task afresh).

(d) *Harness B: SWE-agent on SWE-bench Verified.* SWE-agent v1.1.0 through Harbor's `swe-agent` agent (installed in each
task container; `--env.deployment.type=local`) on `swebench-verified@1.0` (Harbor's verifier decides "resolved"), via the
subclass `adapters/swe_agent/harbor_jev_swe_agent.py:JevSweAgent` in **every** arm, so the arms differ only in the model:
1. *Config* (`adapters/swe_agent/e25_default.yaml`): SWE-agent's `config/default.yaml` at v1.1.0 with two changes for all
   arms: its `cache_control` history processor is removed (a provider prompt-cache marker that litellm would turn into
   explicit Gemini context caches), and `completion_kwargs: {thinkingConfig: {thinkingLevel: minimal}}`. litellm is
   pinned to 1.82.6 inside the container (the version of H5's, E4b's and E22's Gemini runs).
2. *Caps (every arm):* 50 LLM calls per instance (`per_instance_call_limit`) and SWE-agent's default $3 per instance as
   a safety net, the task's agent timeout. In `jev_router` both caps apply to the two LLMs together.
3. *Routing* (`adapters/swe_agent/jev_sweagent.py`, loaded by the container's `sweagent` wrapper; with routing off it
   only starts SWE-agent unchanged): a routed model wraps two SWE-agent `LiteLLMModel`s and, before each query, asks
   E4's routing question (`Q_ROUTING` of `jevharness.adapters.mini_swe`) on E4's state rendering (task, the last 8
   actions with their outputs, the current observation; same truncations). Every decision is logged to
   `/logs/agent/jev-decisions.jsonl`.
4. *Instances* (`scripts/explore_e25_harnesses.py select-swe` → `configs/explore/e25_harnesses/`, committed before any
   episode): the 500 Verified instances ordered by sha256("jev-explore-E25-swe-agent-2026-10-09" + newline + id).
   Evaluation: the first 100 (or the first N the gate (e) allows). Smoke: positions 100 and 101 (never evaluated).
   Reused instances are not excluded: no E25 arm is tuned or trained on any instance.

(e) *Budget, gate and watchdog.* **Direct Gemini at most $250 for E25 in total: harness A at most $140, harness B at most
$100 (smokes included), $10 reserve. OpenRouter (SSM key) at most $25. AWS at most 40 host-hours.** Each host runs its
smoke, then the gate (`scripts/explore_e25_harnesses.py gate`):
1. every smoke trial is present with no infrastructure error after Harbor's retries;
2. `jev_router`: at least 95 % of its routing decisions carry a Jev answer;
3. every trial reports a cost above 0;
4. the leak check finds 0 files;
5. projection: n is the largest admissible size with 1.2 × Σ_arms max(smoke mean $ per trial of the arm, prior) × n +
   smoke spend ≤ the harness cap; priors $0.60 per trial for `baseline` and `jev_router`, $0.15 for `always_cheap`.
   Harness A: n = 89 if the two main arms fit at 89, else the first K tasks (K a multiple of 10, at least 40) of
   the 89 ordered by sha256("jev-explore-E25-tb2-2026-10-09" + newline + name). Harness B: n = 100 if it fits, else
   the largest multiple of 10 ≥ 40 that fits. `always_cheap` runs only if it fits at that n next to the two main arms.
If (1)–(4) fail, or n < 40, that harness stops after its smoke and E25 reports the no-go; one re-smoke after fixing an
implementation defect is allowed and recorded. A watchdog on each host sums the logged LLM spend of finished trials every
5 minutes and stops the runs at the harness cap minus $15; the readout then covers the tasks complete in every arm and
says so.

(f) *Readout* (`scripts/explore_e25_harnesses.py readout` → `results/explore/e25_harnesses/summary.{md,json}` and
`paper/generated/e25_harnesses.tex`, keys `e25/<harness>/<arm>/...`), reported whatever the outcome, per harness and arm:
1. *Success:* Harbor reward 1 (TB2: every test passes; SWE-bench: the instance is resolved).
2. *Cost per task:* the harness's own LLM cost (pi: the sum of its per-message costs; SWE-agent: the trajectory's
   `model_stats.instance_cost`) plus the decision model's (input tokens × $0.042 per M, Jev's price; OpenRouter's
   reported cost logged beside it); the mean over tasks.
3. *Wall-clock per task:* Harbor's agent-execution time (it includes the decision model's calls; environment set-up and
   the verifier are excluded), mean and median.
4. *Taken share:* the share of `jev_router`'s LLM requests that went to the fast LLM (the steps the decision model took
   away from the strong LLM), with the share decided by Jev versus by the escalation rules.
5. *Paired comparison with `baseline`* (the full set is primary; complete case beside it): Δ success in points with a
   95 % task-bootstrap interval (10,000 resamples, seed 25) and the Tango interval; the certified non-inferiority p at a
   3-point margin (`metrics.ni_certified_p`, as `scripts/hard_ni.py`), "shown" iff p ≤ 0.05; the cost saving
   1 − mean cost(arm) / mean cost(baseline) and the time saving 1 − mean time(arm) / mean time(baseline), each with its
   95 % task-bootstrap interval, the median-time saving, and the shares of tasks on which the arm was cheaper and faster.
6. The spend (logged, and the OpenRouter key's usage at start and end), host-hours and the code commit.
Descriptive; no claim beyond the measured n. A no-go of one harness is reported, and E25 then reports the other alone.
An "E25: execution as run" entry records what ran before any paper text uses it.

### Deviations 2026-10-09 ~01:35 UTC: "E25: scale after the smokes, and harness B's wrapper fix" (exploratory; amends E25 (d) and (e) before any E25 evaluation result was read and before any harness-B evaluation episode)

**What had run and been read when this entry was written.** Both smokes (development only). Harness A's evaluation
(`baseline`, `jev_router`, 89 tasks) started at 01:22 UTC under E25 (e)'s gate; of it only its operational status lines
were read (no trial outcome, cost or time). Harness B: no evaluation episode.
1. *Harness B's wrapper fix (implementation defect, one re-smoke as E25 (e) allows).* The first SWE smoke showed that the
   container's `sweagent` wrapper, copied from Harbor's, sources SWE-agent's venv before starting it, so the agent's
   shell resolved `python` to SWE-agent's own Python 3.12 instead of the task's testbed environment (`which python` ->
   `/opt/sweagent-venv/bin/python`; a repository `setup.py` failed on `distutils`), against Harbor's own stated intent.
   From `144a45f` the wrapper runs SWE-agent's interpreter by absolute path without activating the venv, in every arm
   (SWE-agent's tools are written for the task's Python). The first smoke is kept as `runs/e25/swe/smoke_1`
   (development only; its spend counts); harness B is smoked again before its gate.
2. *The projection's per-trial costs.* E25 (e) 5 used priors of $0.60 ($0.15 for `always_cheap`) per trial, set before
   any measurement; the smokes cost far less (mean $ per trial: harness A 0.12 / 0.10 / 0.06, harness B 0.12 / 0.10 /
   0.02 for `baseline` / `jev_router` / `always_cheap`). Under the old priors harness A's gate dropped `always_cheap` and
   harness B's first gate cut n to 60. **From this entry the projection uses, per arm, max(3 × the smoke's mean $ per
   trial, $0.30) for `baseline` and `jev_router` and max(3 × the smoke's mean, $0.08) for `always_cheap`**; the factor
   1.2, the caps ($140 / $100), the watchdog (cap − $15) and every other rule stand. At the smokes' means this gives
   harness B n = 100 with all three arms, and harness A all three arms on the 89 tasks.
3. *Harness A's `always_cheap`.* It joins the running evaluation as a third Harbor job on the same host, the same 89
   tasks, 5 concurrent trials, started minutes after the other two (stage `evalcheap` of `scripts/aws/e25_harnesses.sh`);
   it stays a descriptive lower bound. Its trials enter the same spend watchdog, and the host powers off only after it
   ends.
The analysis (E25 (f)) is unchanged. The reasons are costs measured in development runs, not outcomes; the change makes
E25 larger and adds the lower bound the registration wanted where it fits.

### Deviations 2026-10-09 ~01:45 UTC: "E25: harness B's pager setting" (exploratory; amends E25 (d) 1 before any harness-B evaluation episode)

Harness B's second smoke (after `144a45f`) ran with the task's Python as intended, but four of its six trials, in every
arm, ended early on SWE-agent's rule "exit after multiple consecutive command timeouts": `git log` and `git diff` opened
the interactive pager in SWE-agent's terminal session and hung until the 30-second command timeout (the first smoke shows
the same hangs). SWE-agent's default config sets no pager variables; mini-SWE-agent's SWE-bench config, which E4/E4r
ran, sets them. **From this entry harness B's config (`adapters/swe_agent/e25_default.yaml`, every arm) adds
`tools.env_variables` with mini-SWE-agent's values: `PAGER: cat`, `MANPAGER: cat`, `LESS: -R`, `PIP_PROGRESS_BAR: "off"`,
`TQDM_DISABLE: "1"`.** Harness B is smoked a third time (the two earlier smokes stay as `smoke_1`, `smoke_2`, development
only, their spend counted) and then gated as amended. Nothing else changes; no harness-B evaluation episode has run and
no E25 evaluation result has been read (harness A's evaluation is running; only its status lines were read).

### Deviations 2026-10-09 ~01:55 UTC: "E24 Part 2: L2-τ² from E22's test episodes" (exploratory, descriptive; registered before any E24 τ² answer)

Part 2 of E24 (Deviations 2026-10-08 "E24: agent-state accuracy (L2) on public benchmarks" (a)). Nothing has been
answered for it: no decision model has seen an E24 τ² question. E22 committed its test-split LLM-only episodes in
`dbf0c8d` (`results/explore/e22_tau2/test_episodes/{retail,airline}/llm_only/`: tau2's results with every message,
and the harness side-car without state strings, `log_state: false`); E24 reads them only. Read before registering,
at $0 and without any model call: those files, the τ² adapter (`bench/suites/tau2.py`), E22's registration and build
code, and one Stage-1 τ² run with logged states. Implementation checks, no model involved: rebuilding the System-1
state with the adapter's own rendering (`JevTau2Agent._view` under the run's logged configuration) from the messages
before each turn reproduced all 692 logged states of 60 Stage-1 retail LLM-only simulations exactly; on E22's test
episodes the rebuilt reference next step equals the logged one at every turn (retail 1,846 of 1,846, airline 1,023 of
1,023), so the message prefixes align with the logged turns.

(a) *Corpus L2-τ²* (`explore_e24_public_l2.py build-tau2` → `results/explore/e24_public_l2/tau2/`, committed before
any answer; the question file holds state, question, options and descriptions only).
1. *Sources:* the `llm_only` simulations of E22's test evaluation (τ²-bench's official test split: retail 40 tasks,
   airline 20 tasks, 4 trials each; Gemini 3.5 Flash alone, no decision model); a simulation whose termination is an
   agent, user or infrastructure error is left out (counted). sha256 of the four files recorded.
2. *States:* every agent turn with a logged tool-choice decision; the state rebuilt as above (the string the harness
   sends to System-1), the options = the domain's tools and `respond_to_user` in the logged order, with the
   descriptions the adapter sends, and the adapter's tool-choice question.
3. *Label: the reference write action that is due.* A state is labelled iff the adapter's reference next step
   (`_Reference.next_step`: the first pending reference action of the task whose argument values have all appeared in
   the conversation and, for a write tool under the confirmation policy, after an affirmative last user message) is a
   WRITE tool (tau2's ToolType); the tool-choice label is that tool and the call's label is that action with its
   arguments. States whose reference next step is a read tool or `respond_to_user` are not labelled: there the
   reference is a heuristic with many valid actions. (Write-due states found by the checks: retail 323, airline 99.)
4. *Argument questions (secondary):* at a labelled state, for every required argument of the label tool the adapter's
   own argument question (`_param_options` / `_param_question`: the bounded values, or values copied from the
   conversation plus `<OTHER>`, at most 51; built without the label); arguments it would leave to the LLM are not
   asked. Label: the reference value as the adapter writes it, or `<OTHER>` when the value is not offered.

(b) *Models and answers* as Part 1 ((c) of the entry above: one question per request through the harness's option
handling, two retries, a failure / refusal / invalid answer wrong, new caches), with **E22's τ²-trained 0.8B head**
(`e22-tau2-0.8b`, `s3://…/checkpoints/e22_tau2/final/`, sha256 of `joint_head.safetensors` `fbc4278c…` and
`unsloth_decision_config.json` `a196c06a…` checked against E22's record; `serve_jev --backend clef`) in place of E11's
ALFWorld model. Jev and the OpenAI Decisions API from this container (their caps of Part 1 cover both parts: $3 each in
total); the GPU rows on E24's host within its 8 host-hours in total (StartLux-9B / 27B only on a GPU with enough
memory, as in Part 1). Flash's own executed tool and arguments are the in-sample reference row.

(c) *Score* (task as the unit, as E22's amendment): per (domain, task) the mean over its labelled states of all its
trials; every task weighs the same; retail, airline and pooled; task bootstrap 95 % (B = 2,000, seed 0). **Primary:
tool-choice accuracy at the write-due states.** Secondary: the complete call (the right tool and every asked argument
right); the wrong-write rate (another write tool chosen, the costly error); the argument accuracy. Keys `e24/…/tau2/…`;
the L2-τ² column of App. `app:e24`; the readout adds Kendall's tau-b between L2-ALFWorld and L2-τ² over the models
with both (descriptive). Deadline 2026-10-09 18:00 UTC; a row cut by it or a cap is reported as not run.

### Deviations 2026-10-09 ~02:15 UTC: "E21b: execution as run" (exploratory, descriptive; written after the runs, before any paper text used them)

(a) *Timing and code* (commit times authoritative; UTC). Registration `53b1de8` 2026-10-08 21:22:26 (its heading says
~21:30); code `e8bb78d` 21:34:12; amendment "E21b: E12's fitted thresholds" `c4e2931` 00:01:10 and E12's profile and
fit records `001c781` 00:01:15, both before any `e12_final` smoke or episode. The host `<instance-id>`
(g5.2xlarge, one A10G 24 GB, us-east-1c; tag `jev-e21b-shift-cascade`) was launched at 21:29:37; AWS credentials
expired at about 21:34, before any chain was sent, and the idle host shut itself down at about 22:30 (as the amendment
records). It was started again at 23:42:11 with the guard re-armed for 172 minutes so that the idle hour counts against
the cap (the coordinating session's instruction). **Chain 1** (`rearm setup cal_e12 live_rl live_e18`, on `f97f2c4`,
E21b's code unchanged since `e8bb78d`): setup 23:42–23:49 (every checkpoint, the defer head and the calibration
questions matched their sha256; the frozen 4B by `rl_eval_plan.json`); E12's calibration answers 23:50–00:00 (all
2,719, no error), run before the live rows so that the fit could be committed while they ran (the live arms kept the
registered order); `rl_final` smoke 00:01:26–00:01:49 ($0.017 per episode; gate: $4.10 projected, run all 200), row
00:01:52–00:34:22; `e18_final` smoke 00:35:43–00:36:59 ($0.063 per episode; gate $18.12, run all 200), row
00:37:05–01:01:15. **Chain 2** (`code live_e12 live_e12_in final`, on `001c781`): `e12_final` smoke 01:03:25–01:03:44
($0.0066 per episode; gate $12.08, run all 200), row 01:03:51–01:32:53; the secondary in-domain row `e12_final_in`
(registration (d): 62 minutes of the guard left against the 45 required, and the gate fitted all 200) smoke
01:34:29–01:34:49, row 01:34:59–01:53:57; `final` 01:54:24 scheduled the power-off for 01:56:24, and the host's state
`stopped` was verified at 02:01. 4B rows served by `serve_jev --fast` (adapter + temperatures), E18 by `serve_jev
--backend clef --clef-defer-head`, each alone on the GPU, 1 worker. Every probe round was clean, no runner was paused,
no episode errored, so nothing was re-run. No other instance was touched.

(b) *Caps.* Direct Gemini **$13.70 of $35** (four smokes and four rows, 808 records, plus $0.002 of probes);
OpenRouter, Jev and OpenAI $0 (no call); E12's calibration answers $0. GPU: 21:29:37–~22:30 (idle) and
23:42:11–~01:57, **about 3.3 host-hours of 4**. Records: `results/explore/e21b_shift_cascade/` (box records, probe log,
gates, windows, slim per-task copies in `live/`, the summary); raw runs in `s3://…/e21b_shift_cascade/runs/` and the
gitignored `runs/e21b_*`.

(c) *Readout* (`scripts/explore_e21b_shift_cascade.py summarize`; keys `e21b/…`), as registered, whatever the outcome;
200 of 200 tasks per arm, paired with E21's logged `llm_only` row (200 of 200 solved, $0.0535 and 8.0 s per episode).
- **Frozen 4B** (final harness, τ 0.5 / 0.99999988): **94.0 %** (−6.0 points, Newcombe [−10.2, −2.8]; 0 wins, 12
  losses); non-inferiority at 3 points **not shown** (p = 0.993; Tango [−10.2, −3.5]); 64.8 % cheaper with the GPU
  tariff ($0.0189 per episode; API only 72.8 %); 9.5 s per episode (median 7.0) against 8.0 s; took 99.6 % of the tool
  choices (in-domain 100.0 %) and 58.9 % of the arguments (85.5 %); 62 of 1,364 labelled tool choices it took were
  outside the acceptable set (7 of them at the 113 must-ask states, all of which it took); 17 of 959 arguments wrong.
  Against E21's REPLACE row on the same tasks (96.0 %): −2.0 points [−5.8, +1.5].
- **E18** (the 0.8B head with the defer head, τ = 1 − k): **64.5 %** (−35.5 [−42.3, −28.9]; 0 / 71); not shown
  (p = 1.000); 24.8 % cheaper ($0.0403; API only 30.7 %); 7.0 s; it **handed back 32.8 % of the tool choices** (none
  in-domain) and 43.4 % of the arguments (1.3 % in-domain); the LLM was right on 97.9 % of the deferred tool choices,
  but 383 of the 984 tool choices the head took were wrong (33 at must-ask states). Against arm A's REPLACE row
  (17.5 %): +47.0 points [+39.6, +53.5].
- **E12** (released AgentDecision-4B, τ̂ 0.445 / 0.99587): **92.5 %** (−7.5 [−12.0, −4.0]; 0 / 15); not shown
  (p = 1.000); 69.7 % cheaper ($0.0162; API only 76.8 %); 8.4 s; took 100.0 % of the tool choices and 62.2 % of the
  arguments (in-domain row: 100.0 % and 92.3 %); 22 of 1,272 tool choices wrong (2 at must-ask states), no argument
  wrong; its losses fell mostly on the new kind "ask after a lookup" (35 % solved). Against its REPLACE row (91.5 %):
  +1.0 [−2.2, +4.4].
- **E12 in-domain** (secondary, seeds 300500–300699, paired with the logged dedicated `llm_only`, 99.5 %):
  100.0 % (+0.5; 1 win, 0 losses), **non-inferior** (p = 0.002), 86.3 % cheaper, 5.4 s against 13.6 s (that row ran
  on 2026-10-05), 100.0 % of the tool choices and 92.3 % of the arguments taken, no accepted error.
- *Registered reading:* for all three shifted arms "the gate protects success under shift" does **not** hold, "it still
  saves" holds, and none defers almost everything. The two 4B models' confidences stayed above their in-domain
  thresholds, so the gate handed back almost no tool choice (more arguments were handed back); E18's learned deferral
  reacted but too little. Exploratory; no test.

### Deviations 2026-10-09 ~02:30 UTC: "E25: one re-run of infrastructure errors" (exploratory; amends E25 (b) 4 while the evaluations run, before any E25 evaluation outcome was read)

Read so far from the running evaluations: counts of finished trials and their exception types only. Harness A has four
trials (two per main arm) that ended in a RuntimeError after Harbor's retries: the environment did not start within
Harbor's 600-second limit (`EnvironmentStartTimeoutError`, under the host's load), and Harbor's own retry then failed on
a container-name conflict with the first attempt's leftover container, so Harbor's retries cannot recover these trials.
**From this entry, after a harness's evaluation has ended (all its arms), every trial whose final result carries an
exception other than `AgentTimeoutError` is run once more from scratch, in the same arm with the same settings on the
same host (one Harbor job per arm over those tasks, 5 concurrent, the spend watchdog unchanged; stage `rerun` of
`scripts/aws/e25_harnesses.sh`, output `runs/e25/<H>/eval_rerun`).** The re-run's result replaces the errored one; a
trial that errors again stays a failure in the full set and leaves its pair in the complete case. This is E4r's rule
("infrastructure errors are re-run once"). Both views of E25 (f) are reported, and the readout lists the re-run trials.

### Deviations 2026-10-09 ~03:05 UTC: "E25: harness A's router defect, its fix and a new router run" (exploratory; amends E25 (c) during harness A's evaluation)

**How it was found.** At about 02:55 UTC an interim readout of harness A's running evaluation was read (the first
outcome read of any E25 evaluation): with only 3.3 % of its requests on the fast LLM, `jev_router` solved 20 % of 50
tasks against 48 % for `baseline`, at about a third of the cost, and many of its runs ended after a few requests.
Inspection of pi's session logs showed the cause: **an implementation defect, not routing.** Gemini 3.x rejects a request
whose current turn holds a function call without a thought signature (HTTP 400 "Function call is missing a
thought_signature"), and pi-ai 0.99.2 keeps signatures only on function calls the same model produced, so the first
switch between Gemini 3.5 Flash and Flash-Lite after a tool call ended the run (pi does not retry a 400). 51 of the 74
`jev_router` trials that had finished, and none of `baseline`'s, contain this error. The smoke gate did not check for
agent errors, and its two tasks had hidden it (one of them lost on it).
1. *Fix* (`adapters/pi/jev-router.ts`, committed before the new run): a `before_provider_request` handler gives every
   function-call part without a signature Google's documented placeholder `skip_thought_signature_validator` (the
   placeholder `jevharness.llm.openai_compat` already uses for calls the decision model made). Requests whose function
   calls carry their own signatures, as in every unrouted run, are unchanged. Checked locally at about $0.01: a pi
   session whose decision server forces a switch on every request failed at the first switch without the handler and
   completed four switches and the task with it. Harness B is not affected (litellm keeps the signatures; no such error
   in its logs).
2. *The defective run* was stopped at about 03:00 UTC (74 trials finished). It is not E25's router arm: its trials are
   kept as `runs/e25/tb/eval_defect/jev_router` and the readout reports them only as the defect run (n, success, the
   count of trials that ended on the error). Its spend counts against harness A's cap.
3. *New router run.* `jev_router` runs again on all 89 tasks with the fixed extension and otherwise E25 (c)'s settings
   (5 concurrent, same host, new decision-model calls), as stage `evalrouter` of `scripts/aws/e25_harnesses.sh`, into
   `runs/e25/tb/eval/jev_router`. It starts while `baseline` and `always_cheap` (unchanged, still running) finish, so
   its later trials run on a less loaded host than `baseline`'s; the readout states this next to the time comparison,
   which is weaker for harness A than the cost and success comparisons. The re-run rule for infrastructure errors
   (amendment ~02:30 UTC) runs after all three arms have ended.
4. *Budget.* Harness A's cap ($140 of direct Gemini, the watchdog at $125) is unchanged; at the interim means the new run
   adds about $35. If the watchdog stops it, the readout covers the tasks complete in every arm, as registered.
Nothing else changes. This entry is written after harness A's interim outcomes were read; the fix is the only change and
it is the one the defect requires.

### Deviations 2026-10-09 ~04:15 UTC: "E21: execution as run" (exploratory, descriptive; written after the runs of E21 and its addendum; the coordinating session wired E21's keys into the paper while the addendum's rows ran)

Execution record of exploratory row E21 and of "E21 addendum: shift live rows for every model"; it changes no rule,
subset, threshold, row or record. (a) *Timing labels* (commit times are authoritative; UTC, 2026-10-08 unless stated).
The registration `ac3bdad` 15:44:02; the suite, configs, tests, drivers, readout and the L2 corpus `3ec1295` 16:07:22;
host-driver fixes `16e30b8` 17:09:33 and `e5d9de0` 17:47:14 (below); an interim checkpoint `245c84b` (2026-10-09
~22:10 on 10-08) and the interim rows `94486e3`, `b4cd239`, `6f14f4d`, `3bfc02f`, `36b981b`, `816f241`; the addendum's
registration `edd4c28` (10-09 00:03:55) and code `293362a` (10-09 ~00:05). **This container** (the code in the
working tree, committed unchanged in `3ec1295`): the L2 source smoke on development seeds 339000-339001
15:52:52-15:53:39 ($0.16, no error) and its gate ("run all 100", projection $9.80 against the $38 share); the L2 source
runs 15:55:35-16:01:44 (100 tasks each for `llm_only` and `jev_ts_replace_f`, one pass, no errored episode); the corpus
built at 16:02 (100 tasks, none excluded; 1,250 occurrences, 1,093 questions of 129 options; must-ask 100, after an
off-plan step 81, renamed-tool 1,069, new-kind 307 occurrences) with both re-renderings (no surface render failure; 69
options without the added distractors); Jev's three replays 16:02-16:06 and the Decisions API's 16:02-16:11 (1,093
answers each, no error, no refusal); the local live smoke (339002-339003) 16:03:41-16:04:01, its gate ("run all 200",
projection $26.39) and the LLM-only and Jev rows 16:04:45-16:17:15 (400 episodes, none errored); the Decisions API
smoke (339004-339005) 16:20:41 and its row 16:21:45-16:30:18 (200 episodes, none errored, no refusal). Every probe
round was clean; no run paused.

(b) *The host.* A first launch round (16:08-16:10) found no capacity for g6e.2xlarge, g6e.4xlarge or g5.8xlarge in any
zone. A second round launched a g5.8xlarge (`<instance-id>`, us-east-1c) at 16:20:52; this session terminated it
at 16:21:16 while it was still pending, because on its 24 GB GPU the vendor's serving rule (A7.18 (g)) serves neither
StartLux-9B, a registered live row, nor the two 27B models; retries of the registered first type found a g6e.2xlarge
(`<instance-id>`, one L40S 48 GB, us-east-1c) at 16:52:22. Its image already held the StartLux venv, package and
weights; the 4B, 9B and 27B snapshots were re-checked at the pinned revisions, the OpenJev-27B FP8 snapshot resolved
to E15's revision `02f1755f` with E15's helper (same sha256), and every checkpoint matched its logged sha256
(`box/ckpt_sha256.txt`, `box/frozen_verify.json`, `box/startlux_4b_sha256.txt` = E12c's). Chains: (1) `rearm setup`
16:53:36-17:06:16. (2) The model chain sent at 17:07:09 started outside the checkout, so its first replay could not find
the readout script; this session stopped it at 17:09 during AgentDecision-4B's live smoke on development seeds (no
answer and no evaluation episode existed), fixed the driver (`16e30b8`) and resent it. (3) `code m_rl m_e14a ...` from
17:09:53: AgentDecision-4B 17:10-17:35 (main replay; live 17:18:09-17:25:08; both re-renderings); the 0.8B head from
17:35: its replays failed at once because its backend spec carries the live configs' `cache: null`, which the replay
passed to the backend (no answer was recorded), while its live row ran as registered 17:40:51-17:50:01; the session
stopped the chain after that row, fixed the readout script (`e5d9de0`) and resent it. (4) `code r_e14a m_e12 m_sl9b
m_e12c m_sl4b m_intern m_openjevhf m_sl27b final` from 17:50:24: the 0.8B head's three replays 17:51-17:55; E12
17:55-18:21 (live 18:03:54-18:11:14); StartLux-9B 18:21-19:47 (the server needed 14 of its 15 start-up minutes to load
the image's lazily restored weights; at 18:33 this session started a read-only prewarm of the 9B and 27B weight files
over SSM; live 18:57:42-19:18:26); E12c 19:47-20:12; StartLux-4B 20:12-20:43; Intern-Decision-4B 20:43-21:15 (in
process, the documented fallback kernels: no `flash-linear-attention` in the harness venv); OpenJev-27B 21:15-23:02;
StartLux-27B from 23:02, its main replay ending at 10-09 00:45:50. This container's AWS login expired at about 21:40 and
was renewed at 23:41; the host ran unattended meanwhile and nothing in the chain needed it. The coordinating session's
request for every model's live row arrived at about 00:00 on 10-09; the addendum was registered before any of its
episodes, and this session stopped chain (4) at 00:46:00, right after StartLux-27B's main replay and before its
re-renderings (skipped as both entries allow). (5) `code l_sl27b l_openjevhf l_sl4b l_intern l_e12c final` from 10-09
00:46:12: StartLux-27B live 00:49:57-02:22:51, stopped by its time box with 140 minutes left before the guard (164 of
the 200 tasks, the first 164 in seed order, no errored episode); OpenJev-27B 02:29:09-03:09:32; StartLux-4B
03:13:38-03:25:32; Intern-Decision-4B 03:28:34-03:41:04; E12c 03:44:29-03:56:01; `final` 03:56:58. Every live row
passed its gate with "run all 200"; no live run paused; no episode errored.

(c) *Caps.* Direct Gemini **$57.92 of $70**: this container $25.15 of its $38 share (L2 sources $6.99 with the smoke,
LLM-only and Jev $13.98 with the smoke, the Decisions API $4.18 with the smoke) and the host $32.76 (E21's four rows
$17.95 with smokes and the first smoke attempt; the addendum's five rows $14.81 of its $25), probes $0.007. OpenAI
**$1.23 of $5** (replays $0.82, live $0.41); Jev **$0.84 of $5** (replays $0.52, the L2 source run and the live row
$0.32); OpenRouter other than the Jev key $0. GPU: the g5 host was terminated within one minute of its launch; the g6e
host ran 16:52:22 to 10-09 03:57:59, **11.09 host-hours of 12**; neither the guard nor the idle stop fired; the host
powered itself off after `final` (state `stopped`, `Client.InstanceInitiatedShutdown`, verified from this container at
10-09 04:04). The Jev-only key and the OpenAI key never left this container; the host loaded only the Gemini key, into
its live stages, and the key-leak check passed after every live row. Code changed after the registration (driver and
readout only): the two fixes above, the replay-only stage `r_<slug>`, the addendum's live-only stage `l_<slug>` with
the time box, the summary's post hoc breakdown of wrong answers, spend per provider and gzip-aware answer files.

(d) *Readout* (`scripts/explore_e21_shift.py summarize` → `results/explore/e21_shift/summary.{md,json}`, keys
`e21/...` in `paper/generated/e21_shift.tex`), reported whatever the outcome. Shift L2 (task means; in-domain C in
brackets): AgentDecision-4B 98.6 % (99.2), E12c 98.5 (99.6), E12 94.9 (99.1), the 0.8B head 73.2 (99.2); StartLux-27B
94.7 (93.9), OpenJev-27B 92.4 (92.4), Jev 91.8 (91.5), StartLux-9B 90.9 (91.8), StartLux-4B 86.5 (89.1), the Decisions
API 86.1 (90.9), Intern-Decision-4B 75.9 (75.2). Must-ask states: E12c 100.0, AgentDecision-4B 90.8, the 0.8B head
77.2, E12 67.4, StartLux-27B 65.0, Jev and OpenJev-27B 58.3, StartLux-9B 55.0, the Decisions API 41.7, StartLux-4B
36.7, Intern 33.9. Shift L3 (success on the 200 L3 tasks; in-domain live in brackets): LLM-only 100.0 (99.5);
E12c 98.0 (100), AgentDecision-4B 96.0 (100), E12 91.5 (100), the 0.8B head 17.5 (100); Jev 82.0 (83.0),
StartLux-27B 79.9 on 164 (63.0), StartLux-9B 69.5 (74.0), OpenJev-27B 69.5 (63.0), the Decisions API 66.0 (78.5),
StartLux-4B 49.0 (60.0), Intern-Decision-4B 34.0 (26.5). Axis costs (accuracy with the axis undone minus on the
recorded states): surface, trained rows +7.0 points on average (the 0.8B head +21.9, E12 +3.7, E12c +1.4,
AgentDecision-4B +1.1), off-the-shelf rows -0.2; distraction -0.9 and +0.9. **Registered reading:** the trained rows
keep their lead with all axes on at L2 (+3.9 points, AgentDecision-4B over StartLux-27B) and L3 (+16.0, E12c over Jev),
on must-ask states (+35.0) and on the new-kind tasks at L2 (+3.6), not at L3 on the new kinds (-3.3, AgentDecision-4B
95.0 against Jev 98.3); the surface axis costs the trained rows more than the off-the-shelf rows (+7.2 points), the
distraction axis does not (-1.8). Kendall's tau-b over the 11 decision models with both: in-domain C vs shift L2 0.56,
in-domain live vs shift L3 0.49, shift L2 vs shift L3 0.84.

(e) *Paper.* The coordinating session wired the keys into `main.tex`, `appendix.tex` and Figure 4b and wrote the
main-text sentences while the addendum ran; this session wrote `paper/sections/app_e21_shift.tex` and changed no
main-text section or figure file.

### Deviations 2026-10-09 ~05:15 UTC: "E25: harness A's watchdog stop and the re-run" (exploratory; written while harness A's router arm runs, before its watchdog fired; no rule of E25 (f) changes)

**Read when this entry was written.** Harness B's evaluation has ended (no trial with an infrastructure error, so nothing
was re-run; the host stopped itself at about 04:53 UTC) and its readout was read. Harness A: `baseline` and
`always_cheap` have ended; the fixed router run (amendment ~03:05 UTC) has finished 68 of 89 tasks; harness A's logged
spend, the defect run included, is $118.3 against the watchdog's stop line of $125 (cap $140 − $15), so the watchdog
will probably stop the router run before its last tasks. Harness A's driver chain would then start the registered re-run
of infrastructure errors (amendment ~02:30 UTC) at once, under the same watchdog, which would stop those re-runs at its
first check, up to five minutes later: spend without a result, and a re-run that happens to end within those minutes
would replace an errored trial by chance.
1. *No re-run after the stop line.* The watchdog rule of E25 (e) stops the runs at cap − $15 and the re-run amendment
   keeps that watchdog, so **once harness A's watchdog has fired, no trial is re-run**: the driver chain is ended right
   after the watchdog's stop (before its `rerun` stage starts) and `final` (spend, leak check, sync, power-off) is sent
   alone. The trials that carried an infrastructure error stay failures in the full set and leave the complete case,
   as the re-run amendment states for a trial that errors again. If the router run ends below the stop line, the chain
   runs as sent (`rerun`, then `final`) and the watchdog applies to the re-runs as registered.
2. *What the readout covers after a stop.* As E25 (e) registered, the readout then covers the tasks complete in every
   arm and says so: a trial the watchdog stopped (Harbor records it as `CancelledError`) or never started is not
   complete, and its task leaves **both** views (it is neither a failure nor a pair); within the tasks complete in every
   arm, the full set counts a trial with an infrastructure error as a failure and the complete case drops its pair.
   The readout lists the tasks left out and the arm in which each was stopped. Arm means are taken over the same
   tasks as the paired comparison.
Nothing else changes; harness B is unaffected.

### Deviations 2026-10-09 ~05:15 UTC: "E21 addendum 2: the untrained and SFT rows on the shift" (exploratory, descriptive; registered before any model call of these rows; written after E21's readout was read)

Second addendum to E21 ("E21: a shifted DecisionGym split (DecisionGym-Shift)"; first addendum "E21 addendum: shift live
rows for every model"). It is exploratory, has no test, sits outside every Holm table and revises no registered or
reported row: E21's rows, readings and keys stand as run and read. **Written after E21's readout was read** (`1294dda`,
"E21: execution as run" `8de8800`: every shift L2 and L3 number of E21's 11 decision models and of LLM-only). Nothing
of the two rows below has run: neither model has answered an E21 question or controlled an E21 episode.

(a) *Why.* Requested by the coordinating session for the author (2026-10-09 ~05:00 UTC): the training-ladder table
(`tab:train`, RQ4) reads L2 and L3 on the shifted split instead of the saturated in-domain numbers, and its rows
"Untrained" and "+ SFT" (keys `stage2_rl/base/*`, `stage2_rl/sft_v1/*`) are the only ladder rows without an E21
measurement. The two rows are added so that the training table can read the shifted split; no row is chosen by an
outcome, and whatever they show is reported.

(b) *Rows and serving* (as their logged Stage-2 rows: `results/stage2/rl_eval_plan.json` rows `base` / `sft_v1`,
`scripts/aws/stage2_gpu.sh` `serve_ck`). Slug **`base`**: the untrained `Qwen/Qwen3-4B` at revision `1cfa9a72` (the
pinned snapshot E21's 4B rows were served on), no adapter, its temperatures `results/gpu_round5/base_calibration.json`
(sha256 `d75673a4…`), served name `rl-base`. Slug **`sft_v1`**: the same base plus the SFT v1 LoRA (R3-v1, the step
before GRPO; `s3://…/checkpoints/round5/sft_v1/`, `adapter_model.safetensors` and `adapter_config.json` checked against
the plan's sha256 `6736730a…` / `fbaae782…` before serving), its temperatures
`results/stage2/rl_attr/sft_v1_calibration.json` (sha256 `b73c7b8b…`), served name `rl-sft-v1`. Both behind
`training/serve_jev` on :8081 in E21's harness venv, exactly as E21's `rl` row (backend specs in
`configs/explore/e21_shift/backends.yaml`: Jev wire format, timeout 60 s, price model `jevagentrl-4b`). A model that
does not load or answer is reported as not run, with the reason.

(c) *L2.* E21 (d) 3-4 unchanged on E21's corpus (`results/explore/e21_shift/c`: 100 tasks, 1,250 occurrences, 1,093
questions of 129 options): every distinct question through `stage2_corpus.replay` (one question per request, the
harness knockout above the read width, two retries, a failure counted wrong), a new cache per model, 4 workers; the
same metrics, subsets (task means: all, must-ask, after an off-plan step, renamed-tool states, new-kind and
generator-kind tasks) and paired task bootstrap against AgentDecision-4B and Jev. **Main rendering only**: E21 (d) 5's
two axis re-renderings are not run for these rows (to save GPU), so the rows have no axis cost.

(d) *L3.* E21 (e) unchanged: REPLACE at the tool choice (the model picks every tool, the agent LLM writes the
arguments), the same 200 L3 tasks (seeds 330100-330299), Gemini 3.5 Flash on the direct Gemini API (temperature 0,
minimal reasoning), 4 workers, a new System-1 cache per row, E21 (g)'s pause rule and one re-run of infrastructure
errors; the same scoring (success = latest record per task, errored tasks left out and counted; own-state tool
accuracy, steps, calls, API $, wall time, success per kind, Newcombe intervals against LLM-only and Jev on the common
tasks). Before each row a smoke on 2 development tasks (seeds 339026-339027 for `base`, 339028-339029 for `sft_v1`;
never reported) and E21 (e)'s gate against this addendum's Gemini cap (the host's watchdog and gate cap = the host's
logged E21 Gemini spend before this addendum, $32.76, plus the $15 of (f)); a row that does not fit at 200 runs its
first k tasks that fit (reported with its n), as E21 (e).

(e) *Host and order.* E21's stopped host `<instance-id>` (g6e.2xlarge, one L40S 48 GB), restarted; its idle stop
(30 minutes) and a power-off guard 170 minutes after the restart are re-armed first. Chain 1 `rearm code r_base
r_sft_v1` (L2 for both rows; scored from the answer files), then chain 2 `l_base l_sft_v1 final` (L3 for both rows,
then the host powers itself off; the stopped state is verified from this container). Configs
`configs/explore/e21_shift/live_{base,sft_v1}.yaml` and their smokes; the host driver gains the two slugs (serving as
(b), main rendering only) before any run. No other host is touched.

(f) *Caps and deadline.* **Direct Gemini at most $15 for both rows** (smokes and probes included; E21's REPLACE
rows logged $0.013-0.041 per episode); **GPU at most 3 host-hours** on that host; no OpenAI, Jev or OpenRouter spend; no
key other than the Gemini key (from Parameter Store into the live stages only) on the host. **Deadline 2026-10-09
12:00 UTC.** A row stopped part-way by a cap, the guard or the deadline is reported as not run.

(g) *Readout.* `scripts/explore_e21_shift.py summarize` adds the keys `e21/base/...` and `e21/sft_v1/...` (`l2/{all,
mustask, offplan, renamed, newkind, genkind, askany}`, `l2/vs_rl`, `l2/vs_jev`, `l3/{success, n, own, newkind, usd}`,
per kind, `l3/vs_llm_only`, `l3/vs_jev`) to `summary.{md,json}` and `paper/generated/e21_shift.tex`, beside the logged
in-domain numbers of the same models (`results/stage2/stage2.json` `descriptive.stage2_rl.{base,sft_v1}`: C accuracy on
200 tasks and live success on the first 100 Stage-2 tasks) as `e21/<slug>/in/{c,live}`. The two rows enter **none of
E21's registered readings**: the lead of the trained rows and the three Kendall rows stay over E21's 11 decision models
as read (the untrained base is no decision model; SFT v1 is a stage of AgentDecision-4B's training). Descriptive only:
shift L2 vs shift L3 Kendall's tau-b with the two rows added (`e21/kendall/l2_vs_l3_ladder`). The appendix table
`tab:e21` gains the two rows at the top of its trained block, and no other prose changes; no main-text file is changed
by this addendum's agent. An "execution as run" entry follows the runs.

### Deviations 2026-10-09 ~06:10 UTC: "E27: a 0.8B decision model that holds under shift" (exploratory, descriptive; registered before any E27 training step, model answer, episode or host launch; written after E21's and E21b's readouts were read)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E27)" (post-Stage-1;
marked in the paper like the other post-Stage-1 rows). It has no confirmatory test, revises no registered or reported
result and changes no profile, threshold, key or record of any other row (E14's, E18's, E21's and E21b's included).
The models it trains are NEW checkpoints; E14 arm A, E18 and AgentDecision-4B stay as they are. Nothing has run for
E27: no E27 data file has been built, no E27 model trained or asked a question, no E27 episode exists and no E27 host
has been launched. **Written after E21's readout and E21b's readout were read** (`1294dda`, "E21: execution as run"
`8de8800`; "E21b: execution as run"): E27 targets the weakness they showed. Read before registering, at $0 and without
any model call: E14's, E17/E18's, E21's, E21b's and E12's registrations, execution entries, drivers and readouts.

(a) *Why.* Requested by the coordinating session for the author (2026-10-09 ~05:15 UTC): the author would put a 0.8B
decision model forward as a headline if it performed well. In-domain the 0.8B scoring head (E14 arm A: Qwen3.5-0.8B +
a Clef-style head, SFT on the agent's own states) equals the 4B (common states C 99.2 %, live REPLACE 100 %), is
3.5-4.3x faster and needs 2.4 vs 22.0 GB of GPU memory (E14's execution entry); E18 (the same head + a learned defer head) was 89.3 % cheaper and 4.5x faster
than Flash alone in the final harness, in-domain. On E21's DecisionGym-Shift it fell to 73.2 % open-loop (shift L2) and
17.5 % closed-loop (shift L3, REPLACE at the tool choice); with the generator's names and wording restored (E21's
surface-undone re-rendering) it scored 95.1 %, i.e. the surface axis cost it +21.9 points (the trained 4B rows +1.1 to
+3.7); E21b's gated cascade with it (E18: the head plus its learned defer head) reached 64.5 % on the shift. The 4B models carried over (AgentDecision-4B = E12:
94.9 % / 91.5 %; the checkpoint before public data 98.6 % / 96.0 %). **E27 asks whether training the same 0.8B
recipe with surface augmentation (renamed tools, reworded descriptions and requests, more near-copy distractors) and,
in a second arm, E12's public tool-use data makes the 0.8B head hold under shift while keeping its in-domain accuracy,
speed and saving.** Whatever it shows is reported.

(b) *Model and recipe, both arms: E14 arm A's, unchanged* (E14 (b)): `Qwen/Qwen3.5-0.8B` at revision
`2fc06364715b967f1860aea9cf38778875588b17`, a fresh Clef joint schema head (Unsloth 2026.10.2,
`FastDecisionModel.from_pretrained(..., decision_head="clef")`, Unsloth's default size), LoRA r 64 / alpha 64 / dropout
0, head lr 3e-4, LoRA lr 1e-4, cosine with 5 % warm-up, weight decay 0.01, 8 records x 2 accumulation, maximum length
4,096, bf16, seed 3407, Unsloth's Clef loss; the memory fallback only on an out-of-memory error. **One epoch per arm**
(arm A's selected epoch count, as E14's AG; E14 (d)'s second-epoch rule is not used). Temperatures: Unsloth's
`calibrate` on arm A's development split (SFT v1's `data/train_v1/dev.jsonl.gz`, unaugmented, never trained on). One
seed per arm. Code: `scripts/train/e27_clef.py` (reuses `scripts/train/e14_clef.py`'s model, trainer and epoch
functions unchanged).

(c) *Data.*
1. **Arm S (surface augmentation).** Arm A's training input exactly (SFT v1's `train_mix.jsonl` rebuilt by E14's
   `build-a`, sha256 `9fe1088d…`, 16,902 rows: 15,694 JevGym rows + 1,208 When2Call-train replay rows; every JevGym seed
   re-checked against `registry.training_exclusions()` and `seed_guard.TRAIN_FORBIDDEN`, which include E21's whole
   block 330000-339999). A seeded half of the JevGym **states** (sha256("jev-e27-aug" newline state) even) is
   augmented; every row on an augmented state (tool choice, its knockout copy, the stop question, the argument
   questions of the step) gets the same mapping, so a step's argument questions stay one record. When2Call rows are not
   agent states and stay unchanged. Then `clef_data.rows_to_records` as E14.
2. **The augmentation** (`src/jevharness/training/surface_aug.py`, pure Python, committed with tests before any
   training step), per augmented state with a seeded generator: (i) **rename** each of the 8 core tools (`find_user`,
   `get_orders`, `cancel_order`, `refund`, `check_stock`, `ship_order`, `send_message`, `ask_user`) to a procedural
   `verb_noun` name drawn from per-role word lists (8 distinct names, none equal to another option of the state);
   `finish` and every distractor name stay; every whole-identifier occurrence in the state text (history calls, policy
   line), the option labels, the question text and the target is renamed consistently; (ii) **reword** the renamed
   tools' descriptions with per-role template paraphrases; the request (the `TASK:` line) with templates when it is one
   of the generator's request forms (cancel, refund, stock, ship, total, the four clarify variants, multi; names, ids,
   items and enum values kept), otherwise it stays and is counted; the opening and policy lines of the first
   observation and the customer's replies with templates; (iii) **add near-copy distractors**: 2-6 per augmented tool
   choice, each a renamed core name plus a qualifier word, with the role's paraphrase plus a clause that changes what
   it does, target 0, at seeded positions. Parameter names, enum values, observation formats and argument values stay
   (as E21's surface axis). The answer key is only moved to the renamed option; no label changes.
3. **No leakage: a disjointness check before training.** V_aug = the word tokens (lowercase, split at every non-letter
   and at underscores) of every word list and every template literal of the augmentation (placeholders, which take the
   state's own names, ids, items, enum values and field names, excluded), minus a fixed function-word list committed
   with the code. V_E21 = the word tokens of every string literal in `bench/suites/jevgym_shift.py` and
   `bench/suites/jevgym.py` (E21's alias and distractor names, descriptions, request / reply / policy templates and the
   generator's own wording, which is E21's surface-undone re-rendering) plus every state, option, description and
   question text of E21's L2 corpus in all three renderings (`results/explore/e21_shift/c/questions*.jsonl.gz`).
   **Required: |V_aug ∩ V_E21| = 0, on tokens and on light stems (common suffixes stripped).** The overlap count and the
   sha256 of both sorted vocabularies are written to `results/explore/e27_small_shift/vocab_check.json` and recorded in
   the execution entry; the build refuses to write a training file unless the overlap is 0. Reading E21's corpus for
   this check is its only use in training: no E21 seed, state, label or answer enters any training file, and E21's
   live and test seeds are never run before the evaluation stages.
4. **Arm SP (S + public tool-use data).** S's records plus E12's public training items (E12's `train_mix.jsonl.gz`,
   sha256 `fcdb3b9a…`, its 7,000 rows whose `meta.e12` is not `replay_v1`; E12's leakage audit against E1, its
   development prefix and the public test pools already applied), converted by `rows_to_records`, not augmented, **at
   E12's ratio new : agent-state rows = 1 : 1**: 16,902 public rows, two full passes of the 7,000 plus a seeded sample
   of 2,902 without replacement (E12 had 7,000 : 7,000; S keeps all of arm A's 16,902 rows, so the public items are
   repeated, about 2.4 times each, as E12c saw its items twice); shuffled with seed 3407. A public row that does not
   convert (duplicate option labels) is dropped and counted.
5. Build: `scripts/train/e27_clef.py build` on the host (CPU) after E14's `build-a`; all files sha256-recorded.

(d) *Serving.* `training/serve_jev.py --backend clef` (engine v1, as E14's and E21's rows of the 0.8B head), one arm at a
time on :8083 of the E27 host.

(e) *Evaluation, each once, both arms, exactly as the existing rows were measured.*
1. **L1:** E1's 3,991 decisions (`configs/explore/e27_small_shift/e1_e27.yaml` = E14's config with the two arms; a new
   cache): pooled accuracy, BFCL tool choice; E9's cost rule; paired item bootstrap against E14 arm A.
2. **L2 in-domain:** the common states C (`results/stage2/s23`, `stage2_corpus.py replay`, a new cache): task-mean
   accuracy all / after an off-plan step / ask stratum and ECE (E5's `c_rows`), paired task bootstrap (B = 2,000, seed 0)
   against E14 arm A and AgentDecision-4B with E5's reading rule.
3. **L2 shift:** E21's corpus, main rendering (a copy of `results/explore/e21_shift/c`, questions sha256 `827a7678…`),
   `explore_e21_shift.py replay` (one question per request, the harness knockout, two retries, a failure counted wrong);
   E21's task-mean metrics and subsets (all, must-ask, after an off-plan step, renamed-tool states, new-kind and
   generator-kind tasks) and paired task bootstrap against E14 arm A's and the 4B rows' E21 answers. Secondary,
   descriptive, after the main replay: E21's surface-undone re-rendering (the surface axis cost, as E21 (d) 5).
4. **L3 shift:** REPLACE at the tool choice (the model picks every tool, Flash writes the arguments) on E21's 200 L3
   tasks (seeds 330100-330299), 4 workers, no System-1 cache, as E21's `e14a` row; success (latest record per task,
   errored tasks left out and counted), own-state accuracy, success per kind, API $ and wall time per episode, Newcombe
   intervals against E21's LLM-only, E14 arm A and AgentDecision-4B rows on the same tasks.

(f) *Selection rule* (registered now; development of the rule uses no result): **the arm with the higher L2-shift
accuracy ((e) 3, task mean over all occurrences, main rendering) is selected; an exact tie goes to the higher L1 pooled
accuracy, then to S.** The selected arm is named **AgentDecision-0.8B** and additionally gets:
5. **L3 in-domain:** REPLACE at the tool choice on the first 200 fresh Stage-2 tasks (seeds 300000-300199), 4 workers,
   as E14's `live_a`.
6. **In-domain thresholds**, E14 (f) 3 / E21b (b) 3's procedure: the 2,719 calibration questions of the frozen 4B's
   calibration arm (`rl_shadow_cal`, seeds 5000-5149; `cal_requests.jsonl.gz` sha256 `d46701e8…`) answered once by the
   served model (`e5_cal_replay.py answer`), `derive` and the registered command (`e5_cal_replay.py fit`); the values
   go into `configs/harness/explore_e27_final.yaml` (= `jevagent_final.yaml` + `backends.e27`, written `status:
   pending`) and are committed with an amendment "E27: the fitted thresholds" before any final-harness smoke. A point
   whose fit is "always defer" is not delegated; if the tool choice always defers, the final-harness rows equal
   LLM-only by construction and are not run (reported so). No shifted state and no evaluation seed enters the fit.
7. **The gated final harness in-domain**, as E14 (g) (b) and E21b's `e12_final_in`: tool + argument cascade at those
   thresholds, reflection off, argument questions batched per step; the 200 tasks of the dedicated final-harness rows
   (seeds 300500-300699), 1 worker, task-major seed 732, the model alone on the host's GPU, no System-1 cache; paired
   with the logged `llm_only` row of `runs/harness_final_dedicated`: success, **non-inferiority at 3 points by the
   paper's rule** (`metrics.ni_certified_p`, A6.21 `bb_ordered`; Tango interval), cost per episode with the GPU tariff
   q = $1.624 per hour x wall-clock (the convention of the in-domain final-harness rows) and the saving against
   LLM-only (paired task bootstrap), wall time per episode, coverage and accepted wrong decisions.
8. **If the GPU and Gemini caps allow (lowest live priority):** the gated final harness on E21's 200 shifted L3 tasks
   at the same in-domain thresholds (no refit), paired with E21's logged `llm_only` row, as E21b.
9. **Serving, lowest priority:** if host time remains, the latency bench of E14 (f) 4 (`harness_final.py bench`, c = 1
   on 25 episodes, c = 4 on 100) for AgentDecision-0.8B and the frozen 4B (sha256-checked, `serve_jev --fast`) in one
   session on the E27 host, with peak GPU memory; if it does not run, E14's same-architecture numbers are quoted,
   labelled as E14's.

(g) *Live operations.* Gemini 3.5 Flash on the direct Gemini API (temperature 0, minimal reasoning), `JEV_EVAL_RUNS=1`.
E21's pause rule: probe rounds of six one-word Flash calls; a configuration (smokes included) starts after two
consecutive clean rounds; a watchdog pauses it on an episode that errored with HTTP 503 / overload / 429 and resumes
after two clean rounds; after every task is covered, errored episodes are re-run once; no other route. Smokes on
development seeds (never reported): shifted rows on E21's development seeds 339040-339047, in-domain rows on JevGym
development seeds 940-943. Gate per row: spend so far + 1.2 x the smoke's $ per episode x 200 <= the Gemini cap of
(h), else the row runs its first k tasks that fit (reported with its n), else not at all. Live order: L3 shift of the
selected arm, of the other arm, L3 in-domain, final harness in-domain, final harness on the shift.

(h) *Caps and infrastructure.* **One NEW GPU host**, launched by `scripts/aws/e27_small_shift.sh` (tag Name=jev-e27-08b;
image `<aws-resource-id>`, E3's clone; g6e.2xlarge, one L40S 48 GB, else g6e.4xlarge, else g5.2xlarge, each tried
in every us-east-1 default subnet), SSM only, **at most 10 host-hours** (a power-off guard 590 minutes after launch, a
30-minute idle stop, the host stopped by its last stage and its state verified). **Direct Gemini at most $30** (smokes,
probes and live rows; a watchdog on the logged spend). **No OpenRouter, Jev or OpenAI call.** The Gemini key comes from
Parameter Store into the live stages' process only, never printed or written; no other key on the host;
`OPENROUTER_API_KEY_JEV` never goes to AWS. The driver refuses every host it did not create and never touches another
instance (its forbidden list names every instance id of the other drivers plus `<instance-id>`,
`<instance-id>`, `<instance-id>`, `<instance-id>`). Records to `s3://…/e27_small_shift/`, merged
checkpoints to `s3://…/checkpoints/e27_small_shift/<arm>/`, slim copies to `results/explore/e27_small_shift/`.
**Deadline 2026-10-09 20:00 UTC.** Order: training and the offline levels first, then the live rows; a step cut by a
cap, the guard or the deadline is reported as not run (a row cut part-way is reported with its n). `scripts/stage2_spec.py
--gate` has no entry for exploratory rows and does not apply.

(i) *Readings* (descriptive; reported whatever the outcome; one seed per arm).
- Per arm: L1, L2 in-domain, L2 shift (all, must-ask and the other subsets), the surface cost, L3 shift (n), beside
  E14 arm A's and the 4B rows' logged numbers.
- **"Holds under shift"** for AgentDecision-0.8B iff both (1) its L2-shift accuracy is at least AgentDecision-4B's (E12,
  94.9 %) minus 3 points, and (2) on L3 shift the lower end of the paired Newcombe 95 % interval of its success minus
  AgentDecision-4B's REPLACE row (E21's `e12` row, the same 200 tasks) is at least -5 points. "Keeps its in-domain
  level" iff its C accuracy keeps E14 arm A's by E5's rule (lower end of the paired task bootstrap at least -1.0 point)
  and, if run, its in-domain final harness is non-inferior to LLM-only at 3 points by (f) 7. The final-harness rows
  are reported with success, non-inferiority, saving and wall time, as E21b.

(j) *Readout.* `scripts/explore_e27_small_shift.py summarize` → `results/explore/e27_small_shift/summary.{md,json}` and
`paper/generated/e27_small_shift.tex` (keys `e27/…`: per arm `e27/<s|sp>/…`, the selected arm also as `e27/sel/…`); an
appendix section `paper/sections/app_e27_small_shift.tex` (label `app:e27`, every number a `\res{}` key); no main-text
file is changed by E27's agent (the coordinating session integrates). An "execution as run" entry follows the runs.

### Deviations 2026-10-09 ~06:05 UTC: "E25: execution as run" (exploratory, descriptive; written after both harnesses' runs and the readout, before any paper text used them)

Execution record of exploratory row E25 and its amendments (~01:35, ~01:45, ~02:30, ~03:05 and ~05:15 UTC); it changes
no rule, task list, arm or record. (a) *Timing* (commit times are authoritative; UTC, 2026-10-09). Registration
`9b6048f` 01:04:03; code and frozen task lists `c968546` 01:15:42. Two new m7i.8xlarge hosts in us-east-1d, SSM only:
harness A (pi, Terminal-Bench 2.0) `<instance-id>` launched 01:14:39, harness B (SWE-agent, SWE-bench Verified)
`<instance-id>` launched 01:14:44; set-up 01:16-01:17. **Harness A:** smoke 01:18:21-01:21:34, gate go at 01:21:38
(n 89; `baseline` and `jev_router` under the first priors); evaluation of those two arms from 01:22:12; the amended
projection (`c98e801` 01:29:23) admitted `always_cheap` (projected 88.24 USD), which ran from 01:30:00; the router job
was stopped by hand at 02:59:21 for the implementation defect (amendment `1038970` 03:00:32, fix `cc8d896` 03:01:29;
74 finished trials kept as the defect run); the fixed router run 03:02:03-05:43:51; `baseline` ended 04:33:17 and
`always_cheap` 04:35:18 (their chains' power-offs were held by the router stage, as designed). **Harness B:** smoke
01:18:25-01:27:06 (kept as `smoke_1`), the wrapper fix `144a45f` and a re-smoke 01:29:55-01:38:37 (kept as `smoke_2`),
the pager fix `5c68ea2` and the third smoke 01:41:19-01:49:37 (gate go: n 100, all three arms); evaluation
01:50:06-04:51:34 (the build-cache pruner started by hand at 02:03, as recorded in `656ad82`). The session that ran
E25 ended at about 04:05 (container restart); a new session took over at 04:08 with both evaluations still running.
Driver change `7ab55b7` (04:11:56): the `rerun` stage holds a power-off that another chain's `final` schedules, so
harness B's `update rerun final` chain (sent 04:12:23 next to its running `eval final`) could wait for every arm.
**Re-runs.** Harness B: no trial carried an infrastructure error, so nothing was re-run (04:51:36); `final` ran and the
host powered itself off at about 04:53. Harness A: the amendment "E25: harness A's watchdog stop and the re-run" was
committed as `dfb7c5c` at 05:09:00 (its header says ~05:15; the commit time is authoritative), before the watchdog
fired and before any action on it. A helper on the host, armed at 05:09 and replaced at 05:26 by one that also acts
when the router run ends at or above the stop line, ended the driver chain before its `rerun` stage. The watchdog
never fired (its last check, 05:42:11, read 124.81 USD), but the router run ended at 05:43:51 with 125.46 USD logged,
above the 125 USD stop line, so under that amendment's point 1 **no trial of harness A was re-run**: its 12 trials
with an infrastructure error stay failures in the full set and leave the complete case (`baseline` 5: two Docker
compose start failures and three non-zero agent exits; `jev_router` 4 and `always_cheap` 3, non-zero agent exits; the
two qemu tasks failed at the agent's set-up in every arm). `final` was sent alone at 05:44:33 and the host powered
itself off at about 05:46. **Both hosts were read `stopped` at 05:46:36.** The idle stop and the power-off guards were
armed throughout (guard 720 min from each evaluation start, moved by the hold loops, never reached).

(b) *As run, against the registration.* Every arm of both harnesses completed every task (89 / 89 / 89 and
100 / 100 / 100), so no task left either view for a stop. Routing decisions: harness A 3,272 (Jev answered 88.7 %,
the error-escalation rule 11.1 %, decision-model errors routed to the strong LLM 0.2 %); harness B 3,342 (Jev 100 %).
The defect run: 74 trials, 51 of them ended on the missing-thought-signature error, 18.9 % solved; reported apart.

(c) *Caps and hosts.* Direct Gemini (logged): harness A 125.11 USD of 140 (smoke 0.55, evaluation 99.44, defect run
25.13), harness B 59.16 of 100 (three smokes 1.26, evaluation 57.90); E25 184.27 of 250. Jev through OpenRouter
(logged estimate) 0.35 + 0.32 USD; the SSM key's usage went from 17.61 (01:18) to 18.29 USD (05:44), 0.68 of the
25 USD cap (an upper bound: the key is shared). AWS: about 4.5 (A) + 3.6 (B) = 8.2 of 40 host-hours.

(d) *Leak check (registered "must be 0"): harness B 0 files; harness A not met.* Harness A's `final` stages counted 32,
33 and 39 files. Cause: in 13 trials (`baseline` 3, `jev_router` 5, the defect run 2, `always_cheap` 3) the agent ran
`env`, `export` or a print of the process environment inside the task container, and pi's environment holds the Gemini
key (Harbor's pi agent passes it there), so the key went into that trial's pi session log, Harbor trajectory and pi
stdout log (39 files). The smokes' check had found 0. Remediation at 05:49-05:53: the key was replaced by a
placeholder in the 39 S3 objects (each checked clean afterwards) and in the 26 local copies; the copies on the stopped
host's disk are unredacted (the volume is deleted with the instance). In the 7 routed trials the environment listing
was the latest tool result in the state that the router sends to the decision server, so the key very likely reached
OpenRouter and Jev's provider (the state is not logged). **Rotating the Gemini key is the author's step.** Harness B's
checks found 0 files.

(e) *Readout* (`scripts/explore_e25_harnesses.py readout` → `results/explore/e25_harnesses/summary.{md,json}`,
`per_task.csv`, keys `e25/…` in `paper/generated/e25_harnesses.tex`). Full set (complete case in brackets):
- **pi on Terminal-Bench 2.0 (89 tasks):** strong LLM alone 42.7 %, 0.5103 USD and 257 s mean (121 median) per task;
  routing 50.6 %, Δ +7.9 points (bootstrap [−1.1, +16.9]; 12 wins, 5 losses), **non-inferior** (certified p 0.012;
  complete case 83 tasks, +8.4, p 0.013), 12.4 % of the requests on the fast LLM, cost −9.6 % (saving interval
  [−13.6, 27.5] %), mean time +1.0 % (median 0.6 % faster; the router's later start on a less loaded host weakens this
  comparison, as registered); fast LLM alone 36.0 %, −70.9 % cost, +30.5 % time.
- **SWE-agent on SWE-bench Verified (100 instances):** strong LLM alone 63.0 %, 0.2605 USD, 110 s (98); routing
  57.0 %, Δ −6.0 ([−12.0, 0.0]; 2 wins, 8 losses), **non-inferiority not shown** (p 0.848; no infrastructure error, so
  the complete case is the same), 23.2 % on the fast LLM, cost −7.7 % (saving interval [−1.9, 16.3]), mean time
  +12.7 % (33.4 against 29.3 agent steps per instance, 7.5 s of decision-model calls per instance); fast LLM alone
  28.0 %, −68.8 % cost, −7.5 % time.
- *Reading against the goal (the author's, 2026-10-09):* "an open-source harness plus the routing mode saves cost and
  time with success kept" does **not** hold in either harness at this n: pi kept success (non-inferior) but its saving
  is small with an interval that includes zero, and it saved no time; SWE-agent lost 6 points (not non-inferior) and
  was slower. In both, the decision model sent most requests to the strong LLM.

(f) *Paper.* This session wrote `paper/sections/app_e25_harnesses.tex` (label `app:e25`, every number a `\res{}` key,
rendered only once the keys exist) and changed no main-text section, figure, `main.tex` or `appendix.tex`.

### Deviations 2026-10-09 ~06:00 UTC: "E21 addendum 2: execution as run" (exploratory, descriptive; written after the runs of "E21 addendum 2: the untrained and SFT rows on the shift" and their readout)

Execution record of "E21 addendum 2: the untrained and SFT rows on the shift"; it changes no rule, subset, threshold,
row or record. (a) *Timing* (UTC, 2026-10-09; commit times are authoritative). The registration `3a3face` 05:12:35;
the code (host driver, backend specs, live and smoke configs, readout) `e2a0fd5` 05:16:24, before any model call; no
code changed after it. The host `<instance-id>` (g6e.2xlarge, us-east-1c): the start request at 05:16:51 and a
retry at 05:17:04 failed with InsufficientInstanceCapacity; it started at 05:18:08. Chain 1 `rearm code r_base
r_sft_v1` from 05:18:45: the guard re-armed at 170 minutes and the idle stop at 30; code at `e2a0fd5`; `base` served
at 05:19:48 (the plan's temperature file and the base snapshot at revision `1cfa9a72` checked,
`box/ladder_verify_base.json`), its main replay done at 05:25:10 (1,093 answers, no error, no refusal); `sft_v1`: the
adapter already on the host's image matched the plan's sha256 (`box/ladder_verify_sft_v1.json`; no download), served
at 05:26:25, its main replay done at 05:32:04 (1,093 answers, no error, no refusal); chain end 05:32:42. Both rows'
L2 was scored from the answer files before chain 2 was sent. Chain 2 `l_base l_sft_v1 final` from 05:33:24: `base`
smoke on 339026-339027 and the gate at 05:35:00 ("run all 200", projection $33.17 against the host cap $47.75), the
row 05:35:38-05:38:36; `sft_v1` served at 05:39:52, smoke on 339028-339029 05:40:32-05:40:57, the gate at 05:41:16
("run all 200", projection $34.96), the row 05:41:54-05:49:14; `final` at 05:50:20, the host's shutdown at 05:51:21.
All 8 probe rounds were clean; no run paused; no episode errored, so nothing was re-run; the key-leak check found 0
files after each live row.

(b) *Caps.* Direct Gemini **$3.95 of $15** (the `base` row $0.66 and its smoke $0.003, the `sft_v1` row $3.28 and its
smoke $0.013, probes $0.001); OpenAI, Jev and OpenRouter $0. E21's direct Gemini total with both addenda is now $61.87
(the host side $36.71; key `e21/spend/gemini`). GPU **0.55 host-hours of 3** (start 05:18:08 to the shutdown at
05:51:21; the driver's own count 0.526 h from the re-arm); the host powered itself off after `final` (state `stopped`,
`Client.InstanceInitiatedShutdown`, verified from this container at 05:57:22; it reported `stopping` until then). The
deadline (12:00) was not reached. The host loaded only the Gemini key, into its live stages; no other host was touched.

(c) *Readout* (`scripts/explore_e21_shift.py summarize`; keys `e21/base/...`, `e21/sft_v1/...` in
`paper/generated/e21_shift.tex`), reported whatever the outcome; in-domain numbers (the logged Stage-2 rows: C on 200
tasks, live on the first 100) in brackets. Untrained Qwen3-4B (`base`): shift L2 52.9 % of the common states (50.1),
must-ask 27.6, renamed-tool states 44.7; shift L3 13.5 % of the 200 live tasks (13.0). Qwen3-4B + SFT v1 (`sft_v1`):
shift L2 98.4 % (99.2), must-ask 95.0, renamed-tool states 98.1; shift L3 96.0 % (100.0), no errored episode in either
row. E21's registered readings are unchanged (the lead of the trained rows and the three Kendall rows over its 11
decision models); descriptive only, shift L2 vs shift L3 Kendall's tau-b over the 13 models with the two rows added
is 0.88 (`e21/kendall/l2_vs_l3_ladder`). Records: answers `results/explore/e21_shift/c/answers_{base,sft_v1}.jsonl.gz`,
slim live copies `results/explore/e21_shift/live/e21_live_{base,sft_v1}.json` (and their smokes), host records in
`results/explore/e21_shift/box/` (raw logs on S3), host hours in `results/explore/e21_shift/gpu_hours.json`
(`addendum_2`).

(d) *Paper.* The two rows were added at the top of the trained block of `tab:e21` (App. `app:e21`); no prose and no
main-text file was changed by this addendum's agent.

### Deviations 2026-10-09 ~06:55 UTC: "E24 Part 2: execution as run" (exploratory, descriptive; written after the runs of both parts; also the execution record of Part 1, which had no entry of its own)

Execution record of exploratory row E24 (Deviations 2026-10-08 "E24: agent-state accuracy (L2) on public benchmarks",
its amendment "E24: consequence labels lead" and "E24 Part 2: L2-τ² from E22's test episodes"); it changes no rule,
label, model, row or record. Part 1's readout (`05a1a8e`, `791778a`), the appendix section (`f106e05`) and Part 2's
interim rows (`4d23f03`, `ba3b839`) were committed before this entry at the coordinating session's request; Part 2's last
row (StartLux-27B) and the final readout were committed just before it (`e4a0b0c`).

(a) *Timing and code* (commit times are authoritative; UTC). The E24 entry `a7981f8` 2026-10-08 20:42:31 (its header
says ~20:55), the amendment `6604cf2` 21:24:15, the Part 2 entry `e114e9b` 2026-10-09 01:51:04 (header ~01:55), each
before any answer of the part it governs. The L2-ALFWorld corpus and the code `a35fc02` 21:44:36; the L2-τ² corpus
`b563dac` 01:54:11 and its code `4d23f03` 02:05:08 (from E22's test episodes committed in `dbf0c8d`, 2026-10-08
23:58:42); each corpus before any answer of its part. Later code: `f97f2c4` 23:39:06 (the consequence check forks the
replayed game at each state instead of replaying the prefix for every branch; same rule), `1f127f0` 23:49:38 (the
driver syncs E11's checkpoint directory whole, as E23 did), and `3308053` 2026-10-09 04:04:57, written after the GPU
rows' τ² answers had been read: the in-sample reference row's *argument* accuracy had counted Flash's missing argument
values as wrong at every state where Flash called another tool (46.6 %); Flash's argument values exist only where it
called the label tool, so the row now averages over those states (98.1 %). No decision-model score, and none of the
reference row's other scores, changed.

(b) *Corpora as built.* L2-ALFWorld: 274 games (valid_seen 140, valid_unseen 134), 3,824 logged states replayed with no
option or observation mismatch; 3,355 labelled states (1,660 / 1,695; 3,263 distinct questions); dropped 327 from the
expert's fallback on and 142 at its random fallback. L2-τ²: 240 simulations (retail 160, airline 80; none dropped),
2,869 agent turns rebuilt (the rebuilt reference next step equalled the logged one at every turn), 422 write-due states
of 46 of the 60 test tasks (retail 323 states of 35 tasks, airline 99 of 11), 422 tool-choice and 1,445 argument
questions (18 argument labels are `<OTHER>`; 540 required arguments are left to the LLM by the adapter and not asked).
Neither question file holds a label.

(c) *Consequence labels* (Part 1). Computed in this container (the amendment allows it), blind to the row that chose a
pair. A first implementation replayed the prefix for every branch; its partial output (590 pairs) was replaced, before
any score was read, by the fork-based code of `f97f2c4`, which agreed with it on all 590 pairs. In all 9,120 (state,
action) pairs were evaluated and 1,411 accepted; the expert's baseline H(s) exists at 1,997 of the 2,843 evaluated
states; at the others the expert does not win within 60 steps (mostly at co-located receptacles), and the acceptable
set is the lenient set, as registered. On the 80 sampled states the acceptable set holds 8.3 of 33.2 offered actions
(seen) and 3.9 of 25.9 (unseen). No game errored.

(d) *Host.* One host, `<instance-id>` (Name=jev-e24-public-l2, us-east-1d), launched at about 20:45 as a
g5.2xlarge (no g6e.2xlarge capacity in any subnet; the amendment's "20:50" is approximate) and stopped at 21:01 while
the code was tested; the AWS session then
expired (first failed call 21:45) until 23:41. The stopped host's type was switched to g6e.2xlarge whenever it was to
start (one host, both registered types); its start was refused for capacity at 23:42 and 00:29 (it was switched back
and started as a g5.2xlarge, one A10G 24 GB) and at 00:48 and 01:00, and succeeded at 01:01 (one L40S 48 GB). Chain 1
(23:42-00:24, g5): E11's model did not serve (the driver had copied only its three sha256-checked files and
`serve_jev` reads the tokenizer from the adapter directory; fixed in `1f127f0`), then AgentDecision-4B, the 0.8B head,
Qwen3-4B and StartLux-4B. Chain 2 (00:31-00:43, g5): E11's model. Chain 3 (01:02-02:24, g6e): StartLux-9B and -27B
(Q8_0) on L2-ALFWorld. Chain 4 (Part 2, `CORPUS=tau2`, sent at 02:23 before chain 3's power-off took effect; its first
stage cancelled that power-off): E22's head 02:25-02:30, AgentDecision-4B 02:30-02:38, the 0.8B head 02:38-02:41,
Qwen3-4B 02:41-02:48, StartLux-4B 02:48-03:15, StartLux-9B 03:15-03:55, StartLux-27B 03:55-06:40 (serving from 03:56;
165 minutes, about 11 answers a minute); `final` at 06:40:07 scheduled the power-off for 06:41:07. Every checkpoint
matched its record (E11's three files, E12's adapter, the 0.8B head, E22's head and decision config); StartLux at the
pinned revisions (4B and 9B original weights, 27B the Q8_0 GGUF through llama.cpp's image at the logged digest). Every
row answered every question of its corpus with no error left after the registered retries (a second attempt was needed
by Jev for 4 ALFWorld questions and by OpenAI Decisions for 7 τ² questions); OpenAI Decisions refused 3 of the 1,867 τ²
questions (scored wrong); no answer was invalid. No API key of any kind was on the host; it made no LLM call and played no
episode. *A record defect, no answer affected:* each chain's `code` stage resets the host's checkout to the pushed
commit, and this also reset the tracked box records the host had appended after that commit; chain 4 thereby dropped
the lines of StartLux-9B and -27B (L2-ALFWorld) from `box/model_runs.jsonl` and chain 3's line from
`box/host_hours.jsonl`, and replaced the Part 1 server logs of AgentDecision-4B, the 0.8B head and Qwen3-4B and the
StartLux manifests with Part 2's. The dropped lines are restored from `791778a`, the commit that recorded them; Part 1's
server logs and manifests stay in `791778a`. Answer files are not tracked on the host and were not affected. The
first E24 session's container restarted at about 04:05 while StartLux-27B answered; the session that resumed the row
only read the host's status and answer counts (S3, read-only SSM commands), sent it no stage, and fetched the answers
and box records from S3 after `final`.

(e) *Caps.* GPU: **6.70 host-hours of 8** by the driver's clock at `final` (it excludes the stopped periods; about 6.8 h
from the instance's state changes, the boots and power-offs included); the host stopped itself after `final`, and its
state `stopped` was verified at 06:47:36. Jev **$0.44** (L2-ALFWorld 0.12, L2-τ² 0.32; cap $3) and OpenAI Decisions
**$0.82** (0.21 and 0.61; cap $3), both from this container; direct Gemini $0; OpenRouter other than the Jev-only key
$0. The deadline (2026-10-09 18:00 UTC) cut no row. Records: `results/explore/e24_public_l2/` (both corpora, every
answer file, the box records, the summary); answers and box records also in `s3://…/e24_public_l2/`.

(f) *Readout* (`explore_e24_public_l2.py summarize` → `results/explore/e24_public_l2/summary.{md,json}`, keys `e24/…`;
App. `app:e24`), as registered, whatever the outcome; game means (Part 1) and task means (Part 2), bootstrap 95 %.
- *Part 1, L2-ALFWorld, acceptable set, pooled [95 %]* (strict expert agreement, a lower bound, in brackets after the
  slash): ALFWorld-trained 4B 92.8 [91.3, 94.1] / 84.5; StartLux-9B 72.8 [70.1, 75.5] / 55.0; StartLux-27B 70.7
  [68.0, 73.3] / 55.2; OpenAI Decisions 68.2 [65.2, 71.1] / 48.1; Jev 67.9 [64.8, 70.9] / 49.0; StartLux-4B 66.5
  [63.5, 69.6] / 47.6; AgentDecision-4B 64.7 [61.6, 67.7] / 46.9; Qwen3-4B 61.1 [58.3, 63.8] / 42.5; the 0.8B head 48.8
  [46.1, 51.4] / 30.2; Flash's own action (in sample) 80.9 [78.2, 83.8] / 58.7. *Reading rules ((d) 5):* spread 44.0
  points (the ALFWorld-trained 4B against the 0.8B head); the ALFWorld-trained model leads, by 28.1 points over the best
  transfer row (AgentDecision-4B) and by 20.0 over the best off-the-shelf row (StartLux-9B); seen to unseen, every row
  lost 0.8 to 7.8 points (the ALFWorld-trained 4B −2.4); the DecisionGym-trained models do not keep their DecisionGym
  lead (AgentDecision-4B is above only its untrained base and the 0.8B head; the 0.8B head is last). Kendall's tau-b of
  L2 against live REPLACE success (descriptive): across the three (model, split) cells with a live row 1.00 (strict
  0.33); within the ALFWorld-trained 4B over 12 split × task-type cells 0.08 (strict −0.38), within Jev over 6 cells 0.07
  (0.07), both pooled over 18 cells 0.49 (0.31).
- *Part 2, L2-τ², pooled over 46 tasks* (tool choice at the due write [95 %]; complete call; wrong write; arguments):
  AgentDecision-4B 62.7 [53.7, 71.0], 53.4, 9.7, 89.1; the 0.8B head 61.8 [50.7, 71.6], 24.0, 16.1, 64.5; StartLux-9B
  59.3 [51.1, 67.1], 55.2, 7.6, 93.0; StartLux-4B 59.2 [50.9, 67.3], 48.0, 8.9, 79.8; Qwen3-4B 59.2 [49.9, 68.1], 48.1,
  9.0, 86.7; Jev 52.0 [43.9, 59.9], 46.1, 5.2, 91.3; StartLux-27B 51.6 [43.8, 59.3], 48.9, 5.4, 88.8; E22's τ²-trained
  head 46.6 [39.2, 54.0], 42.7, 4.6, 92.1; OpenAI Decisions 37.7 [30.4, 44.9], 34.8, 3.3, 90.2; Flash's own action (in
  sample) 47.7 [40.1, 55.1], 46.0, 4.0, 98.1 (arguments where it called the label tool). E11's ALFWorld model is not a
  Part 2 row (E22's head took its place, as registered). Retail / airline tool choice: AgentDecision-4B 68.2 / 45.3,
  the 0.8B head 68.9 / 39.2, StartLux-27B 54.2 / 43.5. Kendall's tau-b between L2-ALFWorld (acceptable, pooled) and
  L2-τ² (tool choice, pooled) over the 8 rows with both: −0.43. Read descriptively: AgentDecision-4B chose the due
  write most often, but its interval overlaps every other row's except OpenAI Decisions', it chose another write more
  often than the hosted rows, and StartLux-9B made the most complete calls; E22's head, trained to
  agree with Flash, scored close to Flash.
Exploratory; no test.

### Deviations 2026-10-09 ~13:50 UTC: "E27: the fitted thresholds" (exploratory; written after AgentDecision-0.8B's calibration answers and the fit of E27 (f) 6, before any E27 final-harness smoke or episode)

(a) *Times and host.* The registration "E27: a 0.8B decision model that holds under shift" was committed at 05:29:53
UTC (`c97cd70`; its heading says ~06:10, the commit time is authoritative), the code at 05:47:03 (`50debcf`). The host
`<instance-id>` (g6e.2xlarge, one L40S 48 GB; tag `jev-e27-08b`) was launched at 05:47:18 UTC. Chain 1 (`rearm
setup data train_s train_sp eval_s eval_sp select cal_sel shift_sel shift_other live_in`) ran 05:48:48-08:16:40 with
every stage exit 0; the registered selection rule picked **SP** (higher L2-shift task mean), which is therefore
AgentDecision-0.8B. The host then stopped itself by its idle stop (instance-initiated). The AWS login expired while it
was stopped and was renewed at about 13:37 UTC; the full timing, with the gap, goes into the execution entry.

(b) *What was read before this entry.* The chain's status lines (stage exits, the selection output, the live rows'
gates and spend lines), the calibration answers and the fit outputs. The live rows' records were copied to the local
disk with the box records (`fetch`) but not read: no success, own-state accuracy or episode outcome of any E27 live
row was looked at before the thresholds were fixed.

(c) *AgentDecision-0.8B's answers and fit* ($0; no LLM call). Arm SP behind `serve_jev --backend clef` (engine v1) on
:8083 answered all 2,719 calibration questions (`d46701e8…`) through the harness's asking path, 07:48:53-07:51:43 UTC,
no error (`results/explore/e27_small_shift/cal/answers_sel_cal.jsonl.gz`). `derive` and the registered command
(`e5_cal_replay.py fit`, projection rule) gave **tool choice τ̂ = 0.9999751860066713** (1,125 paired decisions) and
**arguments τ̂ = 0.9999751693727809** (1,328 paired). Neither is "always defer", so the final-harness rows of (f) 7-8
run as registered. At τ̂ on the calibration states AgentDecision-0.8B takes 50.0 % of the tool choices (563 of 1,125)
and 77.8 % of the arguments (1,033 of 1,328), with no accepted error; on all paired states it is right on 99.7 % of
the tool choices and 98.0 % of the arguments. (For comparison, from their records: E14 arm A's fit sat within 10^-9
of 1 and took 80.0 % / 78.8 %; the frozen 4B takes 100 % / 77.5 %.) The values are in
`configs/harness/explore_e27_final.yaml` (`status: development`, `backends.e27`, with the fit files' sha256) and
`results/explore/e27_small_shift/cal/fits.json`, committed with this entry. No shifted state, evaluation seed or E27
final-harness episode entered the fit; the shift row of (f) 8 uses the same thresholds (no refit). Nothing else
changes.

### Deviations 2026-10-09 ~15:50 UTC: "E28: our recipe on StartLux-Decision-0.8B" (exploratory, descriptive; registered before any E28 data build, training step, model answer, episode or host launch; written after E27's interim and E21's readouts were read)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E28)" (post-Stage-1;
marked in the paper like the other post-Stage-1 rows). It has no confirmatory test, revises no registered or reported
result and changes no profile, threshold, key or record of any other row (E12c's, E21's, E21b's and E27's included).
The model it trains is a NEW checkpoint, **"StartLux-0.8B (mixed)"** in the records (served as `startlux-0.8b-mixed`,
slug `e28`). Nothing has run for E28: no E28 data file has been built, no model trained or asked a question, no E28
episode exists and no E28 host has been launched. **Written after E27's interim and readout and E21's readouts were
read:** E21's summary (`results/explore/e21_shift/summary.md`) and E27's working-tree readout of 14:57 UTC (arm SP
selected; L1 77.2 %, C 99.2 %, shift L2 96.3 % with must-ask 93.9 %, shift L3 80.0 %, in-domain L3 100 %, the
in-domain final harness 99.0 vs 99.5 % with non-inferiority not shown). E28 is designed in the light of both. Read
before registering, at $0 and without any model call: E9's, E12c's, E14's, E16's, E21's and E27's registrations,
execution entries, drivers and readouts, and the E2 ladder's StartLux-0.8B manifest.

(a) *Why.* Requested by the author on 2026-10-09 (relayed by the coordinating session): a 0.8B decision model
post-trained by our recipe on the StartLux-0.8B base; if it is strong it may be presented as AgentDecision-0.8B with its
base model named (the naming is the author's decision, not a rule of this entry). E12c (our recipe on StartLux-4B) is
our strongest shifted row (shift L2 98.5 %, must-ask 100 %, shift L3 98.0 %; StartLux-4B untrained 86.5 / 49.0), while
E27's 0.8B model starts from a general base (Qwen3.5-0.8B) and fell to 80.0 % closed-loop on the shift. **E28 asks
whether E12c's recipe, on E27 arm SP's kind of data mix (surface-augmented agent states plus E12's public items), gives
a 0.8B decision model from a decision-tuned 0.8B base that holds under shift while keeping its in-domain level.**
Whatever it shows is reported.

(b) *Base.* StartLux-Decision-0.8B: `startlux-models/StartLux-Decision-0.8B` at revision
`bd4f76a600e23227547fee7bfc1825e12a32764c`, the weights of the E2 ladder's and E9's logged rows
(`results/explore/gpu/manifest_e2_0.8B.json`), downloaded on the host at that revision into
`/opt/startlux/models/StartLux-Decision-0.8B` (the directory the vendor's server reads); the sha256 of its weight files
and its licence file are recorded on the host. **Licence: CC BY-NC 4.0.** The trained adapter and any merged weights are
derivatives under the same non-commercial terms: research use only, never released for commercial use. Environment:
`/opt/jev/v2venv` (transformers 5.18, flash-linear-attention, causal-conv1d) plus peft 0.21.2, as E12c. There is no
fallback base: if the base does not load or train, E28 is reported as not run.

(c) *Recipe: E12c (d) unchanged* (SFT only, from a fresh LoRA): LoRA r 32 / α 64 on all linear layers; soft
cross-entropy + 0.5 Brier; option permutation; lr 1e-4, cosine, 3 % warm-up; **2 epochs**; 64 rows per optimizer
step; max length 4,096; seed 0; memory setting `nogc_3072`, then `gc_3072`, then `gc_2048`, each only after an
out-of-memory error. The only change for the smaller model is the training cap, **150 minutes** (E12c: 360 for the 4B
on an A10G); E12c's 4-step smoke existed to choose its fallback base and is not run. The trainer saves the adapter at
the end of each epoch; if the cap cuts epoch 2, epoch 1 is selected. *Checkpoint selection, development data only, E12c
(e) unchanged:* the epoch-end adapters scored offline (`eval_decisions`, accuracy, no temperatures) on E1's 299-item
development prefix (`configs/eval/e1_dev_prefix_ids.json`, built by `explore_e12_train.py devprefix`) and the held-out
JevGym dev rows (`data/train_v1/dev.jsonl.gz`, 2,377 rows); the higher mean of the two accuracies is selected, a tie
selects epoch 2. E1's 3,991 evaluation items, the common states, E21's corpus and every live task are never used for
selection. Temperatures: `training/calibrate.py --min-bucket 200` on E12's temperature set (`cal_dev.jsonl.gz`).

(d) *Data: E12c's mix with E27's surface augmentation* (built on the host, CPU, by `scripts/explore_e28_sl08b.py
build`, before any training step).
1. E12's `train_mix.jsonl.gz` (sha256 `fcdb3b9a…`, re-checked; 14,000 rows) split by `meta.e12`: its **7,000 SFT v1
   replay rows (E12c's agent states)** and its **7,000 public tool-use items** (E12's leakage audit against E1, its
   development prefix and the public test pools already applied).
2. The agent states get **E27's surface augmentation, imported unchanged** (`jevharness.training.surface_aug.augment_rows`:
   a seeded half of the states, sha256("jev-e27-aug" newline state) even; per state one consistent mapping of the 8 core
   tools to procedural `verb_noun` names, reworded descriptions, requests, opening / policy lines and replies, 2-6
   near-copy distractors on the tool choice with target 0; the answer key only moved to the renamed option, no label
   changed). The public items are not augmented.
3. **Mix: the 7,000 (half augmented) agent-state rows + the 7,000 public items, E12's ratio 1 : 1**, shuffled with seed 0;
   14,000 rows, the size of E12c's mix (E27's SP repeated the public items because its agent part was larger; here no
   item is repeated).
4. **No leakage.** (i) E27's vocabulary disjointness check, imported unchanged (`scripts/train/e27_clef.py`'s
   `vocab_check`: the augmentation's word lists and template literals against every string literal of
   `bench/suites/jevgym_shift.py` and `bench/suites/jevgym.py` plus every text of E21's L2 corpus in all three
   renderings, on tokens and light stems): **the overlap must be 0**, else no training file is written; the record goes to
   `results/explore/e28_sl08b/vocab_check.json`. (ii) E27's seed guard (`e27_clef.seed_guard`) on every row before and
   after augmentation: every JevGym seed is a training seed (outside `registry.training_exclusions()`, which include
   E21's block 330000-339999, the calibration range and `seed_guard.TRAIN_FORBIDDEN`), so no evaluation seed of any
   namespace (100000-100499, 200000-200499, 300000-300999, 310000-310599, 330000-339999, the calibration seeds 5000-5149)
   enters training. (iii) Reading E21's corpus for the vocabulary check is its only use; no E21 seed, state, label or
   answer enters any training file. (iv) Answer keys are never sent to a served model (`Question.meta` is never on the
   wire). All files are sha256-recorded (`build_e28.json`).

(e) *Serving.* The selected adapter + temperatures behind `training/serve_jev.py` in the v2 venv with no extra arguments
(as E12c), on :8081 of the E28 host, served as `startlux-0.8b-mixed`. **The untrained reference** StartLux-0.8B is
served as in its logged rows: the vendor's server (`scripts/aws/serve_startlux.sh`, the package at the pinned commit
`a9101e9…`, revision `bd4f76a6…` through a pin file, port 8092, served as `StartLux-Decision-0.8B`).

(f) *Evaluation, each run once, exactly as the existing rows were measured.*
1. **L1:** E1's 3,991 decisions (`configs/explore/e28_sl08b/e1_e28.yaml` = E12c's `e1_e12c.yaml` with this backend and a
   new cache): pooled accuracy, BFCL tool choice, BFCL arguments, When2Call; $ per 1,000 decisions by E9's rule
   ($2.24208 per GPU-hour × logged serving time per decision) and the median latency per decision; paired item bootstrap
   (E9's `paired_gap`) against the untrained StartLux-0.8B (E9's logged answers), E27's SP and AgentDecision-4B (E12).
2. **L2 in-domain:** the common states C (`results/stage2/s23`, `stage2_corpus.py replay`, a new cache): task-mean
   accuracy all / after an off-plan step / ask stratum (E5's `c_rows`); paired task bootstrap (B = 2,000, seed 0)
   against AgentDecision-4B's registered answers (the frozen 4B, 99.2 %) with E5's reading rule.
3. **L2 shift:** E21's corpus, main rendering (questions sha256 `827a7678…`), `explore_e21_shift.py replay` (one question
   per request, the harness knockout, two retries, a failure counted wrong); E21's task-mean metrics and subsets (all,
   must-ask, after an off-plan step, renamed-tool states, new-kind and generator-kind tasks); paired task bootstrap
   against AgentDecision-4B (E12), E12c and E27's SP. Secondary, after the main replay: E21's surface-undone rendering
   (the surface cost, as E21 (d) 5).
4. **L3 shift:** REPLACE at the tool choice (the model picks every tool, Flash writes the arguments) on E21's 200 L3
   tasks (seeds 330100-330299), 4 workers, no System-1 cache, as E21's and E27's rows: success (latest record per task,
   errored tasks left out and counted), own-state accuracy, success per kind, API $ and wall time per episode, Newcombe
   intervals against E21's LLM-only, AgentDecision-4B (E12), E12c and E14 arm A rows and E27's SP row on the same tasks.
5. **L3 in-domain:** REPLACE at the tool choice on the first 200 fresh Stage-2 tasks (seeds 300000-300199), 4 workers,
   as E12c's live row.
6. **The untrained StartLux-0.8B on the shift** (E21 has no StartLux-0.8B row): L2 shift (main, then surface-undone)
   and L3 shift (REPLACE, the 200 tasks, a new System-1 cache as E21's StartLux rows), served as in (e). Its L1, C and
   in-domain live rows are the logged ones (E9; Stage 2's corpus; the E2 ladder).
7. **In-domain thresholds**, E27 (f) 6's procedure: the 2,719 calibration questions of the frozen 4B's calibration arm
   (`rl_shadow_cal`, seeds 5000-5149; `cal_requests.jsonl.gz` sha256 `d46701e8…`) answered once by the served model
   (`e5_cal_replay.py answer`, $0), then `derive` and the registered command (`e5_cal_replay.py fit`) locally; the values
   go into `configs/harness/explore_e28_final.yaml` (= `jevagent_final.yaml` + `backends.e28`, written `status:
   pending`) and are committed with an amendment "E28: the fitted thresholds" before any final-harness smoke. A point
   whose fit is "always defer" is not delegated; if the tool choice always defers, the final-harness rows equal LLM-only
   by construction and are not run (reported so). No shifted state and no evaluation seed enters the fit.
8. **The gated final harness in-domain**, as E27 (f) 7: tool + argument cascade at those thresholds, reflection off,
   argument questions batched per step; the 200 tasks of the dedicated final-harness rows (seeds 300500-300699), 1
   worker, task-major seed 732, the model alone on the host's GPU (the vendor's server stopped), no System-1 cache;
   paired with the logged `llm_only` row of `runs/harness_final_dedicated`: success, **non-inferiority at 3 points by the
   paper's rule** (`metrics.ni_certified_p`, A6.21 `bb_ordered`), cost per episode with the GPU tariff q = $1.624 per
   hour × wall-clock and the saving against LLM-only (paired task bootstrap), wall time per episode, coverage and
   accepted wrong decisions.
9. **If the caps allow (lowest live priority):** the gated final harness on E21's 200 shifted L3 tasks at the same
   in-domain thresholds (no refit), paired with E21's logged `llm_only` row, as E21b and E27 (f) 8.

(g) *Live operations: E27 (g) unchanged.* Gemini 3.5 Flash on the direct Gemini API (temperature 0, minimal
reasoning), `JEV_EVAL_RUNS=1`; E21's pause rule (probe rounds of six one-word Flash calls; start after two consecutive
clean rounds; a watchdog pauses on an episode errored with HTTP 503 / overload / 429 and resumes after two clean
rounds; after every task is covered, errored episodes are re-run once). Smokes on development seeds (never reported):
the shifted rows on E21's development seeds 339048-339053 (two per row), the in-domain rows on JevGym development seeds
944-947. Gate per row: spend so far + 1.2 × the smoke's $ per episode × 200 ≤ the Gemini cap of (h), else the row runs
its first k tasks that fit (reported with its n), else not at all. **Live order:** L3 shift (E28), L3 shift (untrained
StartLux-0.8B), L3 in-domain, final harness in-domain, final harness on the shift.

(h) *Caps and infrastructure.* **One NEW GPU host**, launched by `scripts/aws/e28_sl08b.sh` (tag Name=jev-e28-sl08b;
image `<aws-resource-id>`, E3's clone; g6e.2xlarge, one L40S 48 GB, else g5.2xlarge, one A10G 24 GB, each tried
in every us-east-1 default subnet), SSM only, **at most 8 host-hours** (a power-off guard 470 minutes after the first
boot, a 30-minute idle stop, the host stopped by its last stage and its state verified). **Direct Gemini at most
$30** (smokes, probes and live rows; a watchdog on the logged spend). **No OpenRouter, Jev or OpenAI call.** The Gemini
key comes from Parameter Store into the live stages' process only, never printed or written; no other key on the host;
`OPENROUTER_API_KEY_JEV` never goes to AWS. The driver refuses every host it did not create and never touches another
instance (its forbidden list, kept in `configs/explore/e28_sl08b/forbidden_iids.txt`, names every instance id of the
other drivers plus `<instance-id>`, `<instance-id>`, `<instance-id>`, `<instance-id>`,
`<instance-id>`). Records to `s3://…/e28_sl08b/`, the adapter to `s3://…/checkpoints/e28_sl08b/` (no optimizer
state), slim copies to `results/explore/e28_sl08b/`. **Deadline 2026-10-10 02:00 UTC.** Order: training and the offline
levels first, then the live rows; a step cut by a cap, the guard or the deadline is reported as not run (a row cut
part-way is reported with its n). `scripts/stage2_spec.py --gate` has no entry for exploratory rows and does not apply.

(i) *Readings* (descriptive; reported whatever the outcome; one seed).
- Every level of (f) beside the untrained StartLux-0.8B, E27's SP (E27's selected AgentDecision-0.8B), E12c and
  AgentDecision-4B (E12; the frozen 4B for C).
- **"Holds under shift"** by E27 (i)'s rule: iff (1) its L2-shift accuracy is at least AgentDecision-4B's (E12, 94.9 %)
  minus 3 points, and (2) on L3 shift the lower end of the paired Newcombe 95 % interval of its success minus
  AgentDecision-4B's REPLACE row (E21's `e12` row, the same 200 tasks) is at least -5 points. **"Keeps the in-domain
  level"** iff its C accuracy keeps the frozen 4B's by E5's rule (lower end of the paired task bootstrap at least -1.0
  point) and, if run, its in-domain final harness is non-inferior to LLM-only at 3 points by (f) 8.
- The comparison with E27's SP is descriptive, level by level (L1 paired item bootstrap, L2 paired task bootstrap, L3
  Newcombe on the same tasks); no winner is declared by this entry.

(j) *Readout.* `scripts/explore_e28_sl08b.py summarize` → `results/explore/e28_sl08b/summary.{md,json}` and
`paper/generated/e28_sl08b.tex` (keys `e28/…`, the untrained reference as `e28/base/…`, and E9-style L1 points
`e9/e28/…`); an appendix section `paper/sections/app_e28_sl08b.tex` (label `app:e28`, every number a `\res{}` key); no
main-text file is changed by E28's agent (the coordinating session integrates). An "execution as run" entry follows
the runs.

### Deviations 2026-10-09 ~15:25 UTC: "E27: execution as run" (exploratory, descriptive; written after every run of E27 and its readout)

Execution record of "E27: a 0.8B decision model that holds under shift" and its amendment "E27: the fitted thresholds";
it changes no rule, arm, threshold, task list or record. (a) *Timing* (UTC, 2026-10-09; commit times are authoritative).
The registration `c97cd70` 05:29:53 (its heading says ~06:10); the code `50debcf` 05:47:03, before any data build,
training step, model answer or episode; two readout-only commits during chain 1 (`2f9f6a7` 07:46:21, the reference rows'
logged numbers; `0d700c1` 07:47:43, an interim scaffold generated at 05:50:41 from the reference rows only, no E27
result in it); the amendment "E27: the fitted thresholds" `8d0c2cf` 13:55:03; the readout commit adds keys only (L1
cost per 1,000 decisions and the E9-style points `e9/e27s|e27sp|e27sel/…`, latency and memory, per-kind and new- vs
generator-kind L3 success, the final rows' hand-back, wrong-accept and win / loss counts, unsigned renaming costs, yes /
no words of the registered readings); no rule or computed number changed. **Host** `<instance-id>`
(g6e.2xlarge, one L40S 48 GB, us-east-1c; the first instance type of (h)), launched 05:47:18. **Chain 1** (`rearm setup data
train_s train_sp eval_s eval_sp select cal_sel shift_sel shift_other live_in`, sent on `50debcf`) 05:48:48-08:16:40,
every stage exit 0: setup to 05:53:31 (Unsloth 2026.10.2 venv; the calibration questions, E12's training mix and the
frozen 4B matched their sha256); data 05:53:48 (vocabulary check: 193 augmentation words against 1,011 E21 words,
overlap 0 on tokens and on light stems; S 14,787 training records from arm A's 16,902 rows, 2,552 of 5,168 agent
states augmented; SP 31,689 records = S plus 16,902 public rows, two passes of E12's 7,000 items plus 2,902 sampled,
none dropped); train_s 05:53:49-06:35:43 (one epoch, 925 steps, 32.6 min), train_sp 06:35:43-07:25:33 (1,981 steps,
40.4 min), no out-of-memory, peak reserved 6.6 GB; eval_s 07:25:33-07:37:05 and eval_sp 07:37:05-07:48:28 (E1: 3,991
decisions each, no error; C: 3,939 answers each; shift L2: 1,093 answers per rendering, no error, no refusal); select
07:48:30 (SP, by L2 shift); cal_sel 07:48:53-07:51:43 (2,719 answers, no error); L3 shift of SP (smoke 07:53:00-07:53:45
on 339042-339043, gate 07:53:48 "all 200", row 07:54:27-08:00:14), of S (smoke 08:01:54-08:02:27 on 339040-339041, gate 08:02:31, row
08:03:17-08:08:23), L3 in-domain of SP (smoke 08:09:33-08:09:52 on 940-941, gate 08:09:58, row 08:10:36-08:16:20).
The host's idle stop then shut it down at 08:45:09 (the end of that boot in the host journal; instance-initiated).
**Gap.** This container's AWS login expired after chain 1 and was renewed at about 13:37; nothing ran for E27 between
08:16:40 and 13:38. 13:38-13:55: the records were fetched, the fit made and the amendment committed (it states what was
read before it). **Chain 2.** Three start requests (13:57:40 to about 13:59:20) failed with InsufficientInstanceCapacity;
the fourth started the host at 14:00:12. The guard counts minutes from the first boot (05:47:33), so the chain `rearm
code final_in final_shift latency final` was sent at 14:01:09 with `GUARD_MIN=850`: a power-off by 19:57:33, before the
20:00 deadline; with the 2.97 host-hours already used, that bounded the total at about 8.9 host-hours (the 590 running
minutes of (h) would have allowed 411 more, to about 20:51, so the deadline was the tighter bound). Rearm 14:01:16
(357 minutes left), code at `8d0c2cf`; final_in (smoke 14:03:01-14:03:37 on 942-943, gate 14:03:45 "all 200", projection
$15.88, row 14:04:32-14:26:21); final_shift (smoke 14:28:23-14:28:49 on 339044-339045, gate 14:28:59 "all 200",
projection $22.73, row 14:29:38-14:56:55); latency 14:57:15-15:07:27 (AgentDecision-0.8B, then the frozen 4B, whose
seven files matched their sha256); `final` 15:07:28, chain end 15:07:49, the host's shutdown at 15:09:39. All 20 probe
rounds were clean; no run paused; no episode errored in any row, so nothing was re-run; the key-leak check found 0 files
after each of the five live rows.

(b) *Caps.* Direct Gemini **$24.07 of $30** (L3 shift SP $3.21, S $2.97, L3 in-domain $2.90, final_in $5.42,
final_shift $9.33, five smokes $0.24, probes $0.003); OpenRouter, Jev and OpenAI $0. GPU **about 4.1 host-hours of 10**:
05:47:18-08:45:09 (2.96 h) and 14:00:12-15:09:39 (1.16 h); the driver's own `hours_used` (9.33 h in `host_final.json`)
counts wall time since the first boot and so includes the 5.3 hours the host was stopped. The host stopped itself
after `final` (state `stopped`, `Client.InstanceInitiatedShutdown`, verified from this container at 15:15:59). Every
registered row ran, all of them with n 200; the deadline was not reached. The host loaded only the Gemini key, into its
live stages; no other instance was touched.

(c) *Readout* (`scripts/explore_e27_small_shift.py summarize`; keys `e27/…` and `e9/e27…` in
`paper/generated/e27_small_shift.tex`; `results/explore/e27_small_shift/summary.{md,json}`), reported whatever the
outcome. Per arm, S / SP: L1 pooled 67.7 / 77.2 % (E14 arm A 67.8); C 98.8 / 99.2 % (arm A 99.2; E5's rule against
arm A: both keep its accuracy; after an off-plan step S loses, SP keeps); shift L2 93.7 / 96.3 % (arm A 73.2,
AgentDecision-4B 94.9), must-ask 77.2 / 93.9 % (AgentDecision-4B 67.4); the surface cost +1.6 / -1.7 points (arm A
+21.9); shift L3 72.0 / 80.0 % of 200 tasks (arm A 17.5, AgentDecision-4B 91.5, the checkpoint before public data 96.0,
Flash alone 100.0). The registered rule selected SP (AgentDecision-0.8B). Its shift L3 losses are all in two of E21's
three new kinds (asking twice 0 of 20, notify 0 of 20; the generator's kinds 140 of 140). L3 in-domain 100.0 % of 200.
Final harness in-domain (fitted thresholds): 99.0 vs 99.5 % for the logged LLM-only row (n 200; 1 task won, 2 lost),
non-inferiority not shown (p 0.059); 30.3 % cheaper with the q = $1.624 / h tariff (37.1 % on API cost alone); 6.5 s per
episode vs 13.6 s in the logged reference; 48.5 % of the tool choices and 89.0 % of the arguments taken, no wrong one
accepted. Final harness on the shift (same thresholds, no refit): 100.0 vs 100.0 % (n 200), non-inferior (p 0.002);
6.0 % cheaper (12.9 % on API cost); 8.1 vs 8.0 s; 27.7 % of the tool choices taken (72.3 % handed back, every must-ask
state among them) and 71.3 % of the arguments, no wrong one accepted. Latency (one L40S, the bench of E14 (f) 4):
median request 69 vs 155 ms for the frozen 4B at c = 1 (2.2x) and 120 vs 318 ms at c = 4 (2.6x); peak GPU memory
2.6 vs 22.6 GB. **Registered readings (i):** "holds under shift" **did not hold** (L2 condition held: 96.3 ≥ 94.9 - 3;
L3 condition did not: the paired Newcombe interval of the difference to AgentDecision-4B's REPLACE row is -11.5
[-18.6, -4.3] points, lower end below -5); "keeps its in-domain level" **did not hold** (C keeps E14 arm A's, but the
in-domain final harness was not shown non-inferior).

(d) *Records and paper.* Host records `results/explore/e27_small_shift/box/` (raw latency bench files and the raw
calibration answers stay on S3; the latter's gzip copy is `cal/answers_sel_cal.jsonl.gz`), slim live copies
`live/`, the L1 answers `records/e1/` (gzip), the full runs on `s3://…/e27_small_shift/runs/`. The appendix section
`paper/sections/app_e27_small_shift.tex` (label `app:e27`, every number a key) is written; it is not yet input by
`sections/appendix.tex`, and no main-text file was changed by E27's agent (the coordinating session integrates).

### Deviations 2026-10-09 ~16:05 UTC: "E29: the full harness in open-source harnesses" (exploratory, descriptive; registered before any E29 code, model call, episode or host launch)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E29)" (post-Stage-1;
marked in the paper like the other post-Stage-1 rows). It has no confirmatory test, revises no registered or reported
result (E4, E4b, E4r and E25 stand as reported) and changes no profile, threshold, key or record of any other row.
**Nothing has run for E29:** no E29 code exists, no model has been called, no episode has run and no host has been
launched. Read before registering, at $0 and without any model call: the registrations, amendments, execution entries
and readouts of E4, E4b, E4r and E25 (`results/explore/e25_harnesses/summary.md`, `results/e4r_swe/summary.md`), the
E25 driver and readout (`scripts/aws/e25_harnesses.sh`, `scripts/explore_e25_harnesses.py`), the pi router
(`adapters/pi/jev-router.ts`, with its thought-signature fix), the SWE-agent and mini-SWE-agent routing adapters, the
source of pi-ai / pi-coding-agent 0.99.2 (the google provider takes a `baseUrl` from `models.json`), of Harbor at git
`4f848e3` (its `pi` and `swe-agent` agents, model-connection code and `registry.json`) and of litellm 1.82.6 (a
`gemini/` model with `api_base` posts to `{api_base}/models/{model}:{method}`), and Sections 3 and 5 of the paper (the
gate rule: delegate when c ≥ q_L − ΔC/V).

(a) *Why.* The earlier ports (E4/E4b/E4r on mini-SWE-agent; E25 on pi and SWE-agent) used one mode only, per-step
routing at a FIXED threshold (0.5), and saved 8-10 % of cost with mixed success (E25: pi +7.9 points, 9.6 % cheaper,
same time; SWE-agent −6.0 points, 7.7 % cheaper, slower). Their records point to three causes: (1) no fitted gate, so
only 12-23 % of the requests went to the fast LLM; (2) switching models breaks the prompt cache, and input tokens
dominate the cost (pi: about 1.2 M input tokens per task); (3) the decision-model call sat on the critical path (about
6-8 s per task). **E29 ports the harness as the paper defines it (a typed question, a gate fitted on shadow logs, cache-
aware routing, the decision model off the critical path, fail-open) into open-source harnesses and tests it against the
unchanged harness on the same tasks, whatever the outcome.** It is the paper's portability claim and the code to be
released.

(b) *The decision layer: one shared module plus thin adapters.* `src/jevharness/adapters/route_proxy.py` (server,
routing policy, logging, spend cap) and `src/jevharness/adapters/route_formats.py` (the two wire formats, state
rendering, action keys): a **host-side model proxy** that every LLM request of every arm passes through. It holds the
API keys; harness processes and task containers get only the proxy's URL and a dummy key (**no API key ever enters a
task container or a harness process**: E25's leak). It speaks Gemini's native `generateContent` /
`streamGenerateContent` (pi through `models.json`'s `baseUrl`; SWE-agent through litellm's `api_base`) and OpenAI chat
completions (mini-SWE-agent through a configurable URL, forwarded to OpenRouter). The path of each request names the
arm and the episode (`/e29/<arm>/<episode>/...`; the episode is the Harbor trial name or the SWE-bench instance id), so
arms and concurrent episodes never mix. The thin adapters only point each harness at the proxy:
`adapters/pi/harbor_jev_pi.py` (agent option `llm_proxy`: writes pi's `models.json` with the google provider's
`baseUrl`; no pi extension is loaded in any arm), `adapters/swe_agent/harbor_jev_swe_agent.py` + `jev_sweagent.py`
(option `llm_proxy`: `--agent.model.api_base`; routing inside the container stays off), `adapters/mini_swe/e29_proxy.yaml`
(+ an `api_url` setting of `OpenRouterTimeoutModel` with the instance id in the path). Modes (one per arm):
1. `strong` (arm `baseline`): every request goes to the strong LLM, unchanged; no decision-model call. This is the
   unchanged harness (only the transport differs from E25 / E4r, the same for both arms).
2. `gated` (arm `gated`, the full harness):
   - *Typed question* (one multiple-choice question per step, Jev's `choice` type, options `fast` / `strong` with one-line
     descriptions): "Which LLM should write the agent's next step, after the result of the action in progress comes
     back?" (`fast`: routine, a smaller and faster LLM will choose the same action: read or search output, run or re-run
     a command, an edit that follows from the plan; `strong`: careful multi-step reasoning, planning, debugging or
     recovery from an error). The confidence c is Jev's probability of `fast`.
   - *Off the critical path (decide-ahead):* the question is asked as soon as the proxy has returned the response of
     step t, on the state up to and including step t's action (task, the last 8 steps with truncated outputs, the action
     in progress; E4's truncations), while the harness executes that action; at step t + 1 the proxy waits for the
     answer at most 10 s, else fails open. The first request of an episode goes to the strong LLM. A retried request
     with the same step index reuses its decision.
   - *Fitted gate:* step t + 1 goes to the fast LLM iff c ≥ τ̂ (the harness's fitted threshold, (c)).
   - *Cache-aware ("sticky") routing:* once a step goes to the fast LLM, the following steps stay on the fast LLM while
     c ≥ τ̂, for at most **k = 4** consecutive fast steps; then the strong LLM takes the next step, and a new fast segment
     may start at the step after it. The strong LLM's cache keeps the shared prefix in between; the proxy never alters
     the prefix (no per-step rewrite of the history).
   - *Fail open:* a decision-model error, timeout or missing answer sends the step to the strong LLM; a fast-LLM request
     that fails (an HTTP error after the proxy's retries) is sent to the strong LLM once (E25's escalation rule).
   - *Thought signatures (Gemini 3.x):* a function-call part without a signature gets Google's documented placeholder
     (E25's fix, `skip_thought_signature_validator`); an upstream HTTP 400 that names thought signatures is retried once
     with every function-call signature replaced by the placeholder.
3. `shadow` (calibration only, never on an evaluation task): the strong LLM acts at every step; after each response the
   proxy sends the identical request once to the fast LLM and once more to the strong LLM (non-streaming, never
   executed; their actions are only logged) and asks the decide-ahead question.
Logs (JSONL, never a key or a request header): one record per LLM call (arm, episode, step, tier, role
primary / shadow_fast / shadow_strong, model, input / cached / output tokens, cost, latency, HTTP status), one per
decision (c, distribution, latency, time the request waited for it, cost, fail-open reason) and, in shadow mode, the
parsed first action of each of the three responses. Cost: OpenRouter's reported `usage.cost` (DeepSeek, Jev); Gemini at
`configs/prices/models.yaml` (Flash $1.50 / $9.00 per M input / output, cached input $0.15; Flash-Lite $0.30 / $2.50,
$0.03; thinking tokens as output). **Spend cap in the path:** the proxy sums its logged spend (all its logs since the
first stage on that host) and refuses new LLM requests (HTTP 503) at the stage's cap; a host watchdog also stops the
harness runs at the cap.

(c) *Calibration and the fitted gate.*
1. *Calibration splits* (`scripts/explore_e29_port.py select` → `configs/explore/e29_port/`, committed before any
   episode), disjoint from every E29 evaluation list: pi: Terminal-Bench Pro (Harbor `terminal-bench-pro@1.0`, 200
   public tasks, alibaba/terminal-bench-pro at `82bc46d`; no task name in common with Terminal-Bench 2.0), the first
   **10** tasks in sha256("jev-explore-E29-tbpro-2026-10-09" + newline + name) order (the 11th is pi's smoke task);
   SWE-bench Verified for both SWE harnesses: the 500 instances minus E4's 100 evaluation instances, E4r's 200 extension
   instances, E25's 100 SWE-agent instances and every smoke or development instance of E4/E4r/E25/H5, in
   sha256("jev-explore-E29-swecal-2026-10-09" + newline + id) order: mini-SWE-agent the first **20**, SWE-agent the
   first **15**.
2. *Labels* (per step t of the shadow runs, the first step excluded): y_f = 1 iff the fast LLM's action equals the strong
   LLM's, y_s = 1 iff the second strong sample's action equals the first. *Action equivalence* (the same rule for all
   three harnesses, through their tool sets): the first tool call of a response, keyed by tool name and its main
   argument after whitespace normalisation (strip, runs of whitespace collapsed): `bash` → the command; SWE-agent's
   `str_replace_editor` → (sub-command, path); any tool with a `path` argument (pi's read / edit / write) → the path;
   `submit` and other tools → the name and canonical JSON of the arguments; a response with no tool call → "no tool
   call" (equal to another response with none). Steps whose shadow call failed are dropped.
3. *Fit* (the paper's rule with q_L = the strong LLM's own consistency at the same state): τ̂ = argmax over τ ∈ {every
   observed c} ∪ {never} of G(τ) = Σ_{t: c_t ≥ τ} [y_f,t − y_s,t + ΔC/V], with V = $1 (the paper's task value at its
   primary operating point; every excess disagreement is charged a whole task, a conservative choice) and ΔC the mean
   logged price difference per step (the strong call's cost minus the fast shadow call's) in that calibration run; ties
   go to the higher τ. If max G ≤ 0, no step pays and the gate routes nothing: that harness's gated arm is not run and
   reported as "the gate kept every step on the strong LLM" (no saving). The fit, its counts and the simulated fast
   share are written to `configs/explore/e29_port/gate_<harness>.json` on the host before any evaluation episode of
   that harness.

(d) *Harnesses, LLM pairs and evaluation tasks* (the pairs of the earlier rows, so references can be compared).
1. `tb`: pi 0.99.2 through Harbor (`harbor_jev_pi:JevPi`, `--thinking minimal`, the 100-turn cap of E25) on
   Terminal-Bench 2.0 (89 tasks); strong Gemini 3.5 Flash, fast Gemini 3.5 Flash-Lite (direct Gemini API, minimal
   thinking). Reference row: E25 harness A.
2. `mswe`: mini-SWE-agent 2.4.6 with E4's configuration on SWE-bench Verified; strong DeepSeek-V4-Pro (reasoning on),
   fast DeepSeek-V4-Flash (reasoning off), both pinned to StreamLake through OpenRouter (the SSM inference key); E4r's
   200 extension instances (`configs/explore/e4r_swe/extension_ids.txt`); E4's caps (250 steps, $0.42 per instance);
   local SWE-bench evaluation (swebench 4.1.0). Reference row: E4r (pooled: −1.0 points, 10.5 % cheaper).
3. `swea`: SWE-agent v1.1.0 through Harbor with E25's config (`e25_default.yaml`) on E25's 100 instances; Gemini 3.5
   Flash / Flash-Lite. Reference row: E25 harness B.
4. Codex CLI: **not run** (it speaks OpenAI's Responses API; a strong/fast pair and a Responses-API path in the proxy
   do not fit the deadline).
The decision model is Jev through OpenRouter (`typesafe/jev-1.13`, the SSM inference key, in-process in the proxy, no
cache). Smoke (development only, never reported): pi on Terminal-Bench Pro's 11th task in each mode; mini-SWE-agent on
`scikit-learn__scikit-learn-13496` (E4r's smoke); SWE-agent on E25's two smoke instances; the gated smoke runs at τ = 0
so that it switches; it must show no infrastructure error, no LLM error passed to the agent, at least one fast and one
strong call and one switch in the gated trial, parsed actions for all three roles in the shadow trial, a Jev answer for
≥ 95 % of the decisions, a cost above 0 and a leak check of 0 files, else that harness stops and the failure is
reported (one re-smoke after an implementation fix, recorded).

(e) *Pilot* (≤ $45 in total: `tb` $38, `mswe` $7). pi and mini-SWE-agent: calibration, fit, then the **first 30
evaluation tasks** of each harness (pi: the first 30 of `configs/explore/e25_harnesses/tb2_order.txt`; mini-SWE-agent:
the first 30 of E4r's extension list) in the gated arm and a fresh `baseline`, run concurrently on the same host (pi: 5
trials per arm; mini-SWE-agent: 8 workers per arm). **Go rule** (mechanical, `scripts/explore_e29_port.py go`, on the
pilot alone): for each harness, saving = 1 − mean cost(gated) / mean cost(baseline) over its 30 tasks (LLM + decision
model, as logged by the proxy) and drop = successes(baseline) − successes(gated); pass(h) iff saving ≥ 20 % and drop ≤ 3;
not_worse(h) iff its saving ≥ its reference row's (pi 9.6 %, mini-SWE-agent 10.5 %) and drop ≤ 3. **GO iff pass(tb) and
pass(mswe), or pass of one harness and not_worse of the other.** A harness whose gate routes nothing has saving 0. The
decision, its inputs and the time are written to S3 and recorded in a Deviations note; on NO-GO, E29 stops and reports
the pilot.

(f) *Full run* (only on GO, automatically): every listed harness, all its evaluation tasks (pi 89, mini-SWE-agent 200,
SWE-agent 100), the gated arm and a fresh `baseline` afresh and concurrently on the same host (SWE-agent's calibration
and fit first). The pilot is reported apart and **not pooled** (it decided the go). One re-run of trials with an
infrastructure error (E25's / E4r's rule), within the caps. Caps: pi $100, mini-SWE-agent $25, SWE-agent $80 (its
calibration included). **E29 in total (direct Gemini + OpenRouter + any OpenAI) ≤ $300**; OpenRouter part ≤ $40 and the
SSM key's own hard limit; AWS: new hosts tagged `jev-e29-tb`, `jev-e29-mswe`, `jev-e29-swea` (m7i.8xlarge, fallbacks
m6i.8xlarge / c7i.8xlarge; SSM only; power-off guard and 30-minute idle stop armed from boot), ≤ 50 host-hours; every
host stops when done (verified). If the watchdog stops a run, the readout covers the tasks complete in both arms and
says so. The leak check (files under the run directories and logs that contain a key value) runs after every stage;
it must find 0 files, else the host's chain stops.

(g) *Readout* (`scripts/explore_e29_port.py readout` → `results/explore/e29_port/summary.{md,json}`,
`paper/generated/e29_port.tex`, keys `e29/<harness>/...`; appendix `paper/sections/app_e29_port.tex`, label `app:e29`),
whatever the outcome, per harness (full set primary: an errored trial is a failure; complete case beside it): n;
success of `baseline` and `gated`; Δ in points with wins / losses and the 95 % task-bootstrap interval (10,000
resamples, seed 29); non-inferiority at 3 points by the repository's rule (`metrics.ni_pvalue`, `bb_ordered`; "shown"
iff p ≤ 0.05); cost saving 1 − mean cost(gated) / mean cost(baseline) with its 95 % bootstrap interval; mean- and
median-time saving (Harbor's agent-execution seconds; mini-SWE-agent's agent wall-clock); the fast-LLM share of the LLM
requests; sticky-segment statistics (segments per task, mean length, share ended by the cap k); decision-model calls,
cost and latency per task and the time requests waited for an answer; fail-open counts; the gate (τ̂, calibration
counts, simulated and realised fast share); spend against the caps; the leak checks. Descriptive; no claim beyond the
measured n. An "E29: execution as run" entry records what ran before any paper text uses it.

### Deviations 2026-10-09 ~16:25 UTC: "E28: L1 cost on the untrained model's serving engine" (exploratory; amends E28 (f) 1 before any E28 vendor-engine timing; requested by the coordinating session for the author)

(a) *What was read before this entry.* E28's chain 1 status lines up to the end of its L1 stage (setup, data, training,
checkpoint selection on development data: epoch 2, temperatures, and the exit of E1's two invocations); no E28 answer,
accuracy or latency of E1, C, the shift corpus or any live row was read.

(b) *Why.* Figure 4a draws E28's L1 point at (cost per 1,000 decisions, L1 accuracy) beside the untrained
StartLux-0.8B (E9's `e9/sl0.8b`: 70.0 %, measured with the vendor's server on an L40S). E28 (f) 1 prices E28 by E9's
rule on its own serving stack (`serve_jev` in the v2 venv), so the two points would mix model and engine (E12c (h)'s
point). The author asks for the cost on the same basis as `e9/sl0.8b/usdk`.

(c) *Amendment.* E28 (f) 1 gains one measurement, **timing only**. After chain 1, on the same host (one L40S, as E9's
StartLux-0.8B row), with nothing else on the GPU: (1) the selected LoRA is merged into StartLux-0.8B's original checkpoint
layout, so the vendor's server loads it exactly as the base: every tensor of the base's safetensors is kept and each
LoRA-targeted weight `W` becomes `W + (alpha / r) B A` (rsLoRA scaling if the adapter's config says so), computed in
float32 and stored in the base's dtype; the adapter's module names are mapped to the base's (`model.layers.*` ->
`model.language_model.layers.*`), every mapped key must exist and every adapter module must be used, else the stage
refuses; the base's config, `decision_config.json` (the vendor's temperatures for the base), tokenizer and processor
files are copied unchanged; sha256 of the merged weights recorded (`scripts/explore_e28_sl08b.py merge-vendor`). (2) The
vendor's server at the pinned package commit (`scripts/aws/serve_startlux.sh`'s command line and kernel check, port
8092, served as `StartLux-Decision-0.8B-e28`) answers E1's 3,991 decisions with E9's StartLux-ladder settings (pools,
exclusions, batching, 4 workers; `configs/explore/e28_sl08b/e1_e28_vendor.yaml`, a new cache). (3) **Keys:**
`e28/l1/usdk` and `e9/e28/usdk` (and `e9/e28/lg` = log10 of the dollars per decision, `e9/e28/lat`) are E9's rule
($2.24208 per GPU-hour × the logged serving time per decision) on this vendor-engine run, the basis of
`e9/sl0.8b/usdk`; `e28/l1/usdk_own` / `e9/e28/usdk_own` keep E9's rule on E28's own serving stack (`serve_jev`), as
registered. **`e28/l1/acc` and `e9/e28/acc` stay the pooled accuracy of E9's 3,991 questions from E28's own serving
stack** (the model's registered read-out). The vendor engine renders its own prompt, which the model was not trained on,
so the accuracy of the timing run is reported only as a check (`e28/l1/acc_vendor`), never as E28's L1 accuracy. If the
vendor's server does not load the merged checkpoint or its kernel check fails, the keys `e28/l1/usdk` and `e9/e28/usdk`
are not written, the summary says so and the own-stack cost stays the only one. Host time: about 20 minutes within
E28's 8 host-hours; no API call. Nothing else changes.

### Deviations 2026-10-09 ~16:35 UTC: "E30: every open decision model at its cheapest common deployment" (exploratory, descriptive; post hoc re-measurement; registered before any E30 code, model load, answer or host launch)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E30)" (post-Stage-1;
marked in the paper like the other post-Stage-1 rows). It has no confirmatory test and **revises no registered or
reported number**: E9's keys (`e9/*`), E12c's and E16's same-engine timings (`e12/fast/*`, `e16/fast/*`) and every
other row's records and keys stay as they are; E30 writes its own keys `e30/*`. **Post hoc**: written after every L1
accuracy and cost it re-measures was read (E9's readout, E12c's and E16's same-engine timings, E15's, E27's readouts).
Nothing has run for E30: no E30 code exists, no model has been loaded or asked a question for it and no E30 host has
been launched.

(a) *Why.* Figure 4a draws each open decision model at E9's cost rule ($2.24208 per GPU-hour × the logged serving time
per decision, four requests in flight) on its own host and engine: StartLux through the vendor's server (transformers
with the fla kernels; the 27B as a Q8_0 GGUF on llama.cpp), our trained models through `serve_jev`, Intern-Decision in
process; the hollow point of AgentDecision-4B is vLLM on one A10G at four states in flight (E12c (h)). The bases differ,
so the plot mixes model and serving effects. The author (relayed by the coordinating session): "0.8B and our 4B should
be costed at the cheapest way they can be deployed". **E30 asks what one decision costs for each open decision model
of Figure 4a when every model is served the same, cheapest way: one engine, one instance type, saturated.**

(b) *Models* (slug = E9's where E9 has one). AgentDecision-4B (`e12`: E12's checkpoint, `s3://…/checkpoints/e12_mixed/`,
merged into Qwen3-4B); the frozen GRPO 4B (`rl`: the freeze record's checkpoint, sha256-verified, merged); StartLux-
Decision 0.8B / 2B / 4B / 9B (`sl0.8b`, `sl2b`, `sl4b`, `sl9b`, at the revisions of E9's rows: the E2 ladder's
manifests and Stage 2's manifest); StartLux-Decision-27B (`sl27b`: the vendor's Q4_K_M GGUF,
`startlux-models/StartLux-Decision-27B-Q4_K_M-GGUF` at `ce98e33e03fe238638b9ec9df3071231bc933a9f`); StartLux-4B + our
recipe (`e12c`: E12c's selected adapter, merged); Intern-Decision-4B (`intern`, `internlm/Intern-Decision-4B` at
`0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd`); E27's AgentDecision-0.8B candidate (`e27sp`, arm SP's checkpoint); E28's
StartLux-0.8B (mixed) (`e28`) if its selected adapter is on S3 when its stage starts (read from
`s3://…/checkpoints/e28_sl08b/selected/` only; E28's host and files are never touched); OpenJev-27B (`openjevhf`) as
the last, optional stage ((d) 5). **Not run:** OpenJev (DiffusionGemma-26B-A4B): its read-out (a masked-slot read of a
diffusion LM) exists only in its own server image, which no common engine replaces, and its 26B weights do not fit
24 GB at FP8. E15's other baselines and E14's heads are outside this entry (their points keep E9's rule; the summary
lists them as not re-measured).

(c) *Deployment.* **Instance**: one g6.xlarge (1× NVIDIA L4 24 GB, 4 vCPU, 16 GiB), us-east-1, on-demand **$0.8048 per
hour** (AWS price list: Linux, shared tenancy, effective 2026-10-01), the cheapest on-demand instance type whose GPU
holds every model at some precision. If no us-east-1 zone has g6.xlarge capacity, a g6.2xlarge (the same GPU); the
cost then stays at the g6.xlarge price (the measurement is GPU-bound), and the instance used and its own price are
reported. **Engine**: vLLM 0.31.0 (the version of E12c's and E16's same-engine timings), in-process `AsyncLLM` fed with
token ids (prompts tokenized before the timed passes; no HTTP), prefix caching on, continuous batching, CUDA graphs
(vLLM's default compile), maximum model length 4,096 tokens (8,192 if a prompt is longer), GPU memory utilization 0.92,
the multimodal checkpoints loaded language-only. LoRA adapters are merged into their base in bf16 before serving
(E12c (h)'s `merge`). **Exceptions, each marked in every table:** StartLux-27B on llama.cpp (the pinned CUDA server image
of the logged 27B row), because only a GGUF int4 file of it fits 24 GB; E27's scoring-head model on our stack
(`serve_jev --backend clef`, coalescing engine, warmed up) driven over the Jev wire format by a client on the same host,
because vLLM does not serve its head.

(d) *Read-out* (each model's own; one forward pass per read; nothing generated). 1. **Our trained code models**
(`e12`, `rl`, `e12c`, `e28`): `serve_jev`'s prompt token for token (`training.data.render`, the tail kept at 4,096
tokens), one request per question, the next-token logits of the option codes at the last position (allowed-token mask,
processed logits); the most likely code is the answer (the served temperatures never change it). 2. **StartLux** (vLLM
and llama.cpp): the vendor's rendering and read-out at the pinned package commit `a9101e9` (`startlux_decision.jevfmt`
and `model.py`): one prompt per question, the option letters' logits at the last position, softmax at the vendor's
per-type temperature; a choice wider than 26 options (7 of the 3,991 decisions) by the vendor's own wide-choice code
(groups of 25, the top 3 of each to a final read). On llama.cpp the letters' log-probabilities come from llama-server's
full-vocabulary log-probabilities at the answer position, as the vendor's `gguf_server` reads them. 3. **Intern-
Decision-4B**: the vendored prompt compiler and encoder of `decision/intern_decision.py` on E1's (state, decision point)
groups (at most 16 fields per prompt, as the backend); each field is one read with the prompt cut right after the
position before its placeholder, which gives the reference's in-sequence logits; softmax over the field's symbols,
the checkpoint's temperature, the reference's tie-break. 4. **E27 SP**: its served head, E1's groups as requests.
5. **OpenJev-27B (optional, last, only with at least 60 host-minutes left)**: vLLM 0.31.0's in-flight 4-bit
(bitsandbytes) quantization of the original weights (`openjev/openjev` at `ac97900f`), the card's helper (`helper/shim.py`)
unchanged with E15's settings; if vLLM does not load it or the helper fails on it, not run (stated). **Answers become
predictions by the harness's own parsing** (`answer_from_wire`, `answer_from_intern`, `norm`; for the code models the
option label of the most likely code, as E12c (h)); a decision is correct iff its prediction is in its gold set, as in
E1. The agreement of each row's answers with its reference record file is reported.

(e) *Precision.* BF16 for every vLLM model where it fits (a model that does not load in 24 GB is reported as such);
FP8 for every vLLM model (vLLM's dynamic FP8 weights and activations from the BF16 checkpoint); StartLux-27B at Q4_K_M
only (BF16, FP8 and Q8_0 do not fit 24 GB); OpenJev-27B at in-flight 4-bit only. **Every precision measured re-answers
all 3,991 decisions** and its accuracy is reported; no accuracy is carried over from another precision or engine. The
**plotted point** of a model is its cheapest precision whose L1 accuracy is at least its reference minus 1 point
(reference: `e9/<slug>/acc`; `e28/l1/acc` for `e28`, reported as pending if it does not exist at the readout); if no
precision qualifies, the most accurate precision measured, flagged.

(f) *Workload and quantities.* E9's 3,991 L1 decisions in one fixed shuffled order (`random.Random(0).shuffle` of the
sorted item ids); E1's groups (Intern, E27 SP) in the order of their first decision. Warm-up: the first 200 decisions,
untimed. **Saturation**: every decision's read(s) submitted with C in flight, C = 64 and C = 256 (one timed pass each, the
prefix cache reset before each); decisions per second = 3,991 / the pass's wall seconds; the saturated rate is the higher
of the two and its C is reported. Answers and accuracy come from the C = 64 pass (the two passes' agreement is reported).
llama.cpp: P parallel slots, P = 8 and P = 16 (context P × 4,096, continuous batching), C = P. E27 SP: C = 16 and C = 64
requests in flight. **Cost per 1,000 decisions** = the hourly price / (3,600 × decisions per second) × 1,000; Figure 4a's
coordinate `lg` = log10(cost per 1,000 / 1,000). **Latency**: C = 1 over the first 1,000 decisions of the order after a
cache reset, median and p90 per decision in ms. Agent-state prompts (E21's L2 corpus) are not measured (time).

(g) *Outputs.* `scripts/explore_e30_cost.py` (readout; host side `scripts/e30_bench.py`, driver `scripts/aws/e30_cost.sh`)
→ `results/explore/e30_cost/summary.{md,json}` and `paper/generated/e30_cost.tex` with keys
`e30/<slug>/{usdk,lg,acc,prec,dps,lat_p50,lat_p90,conc}` (the plotted point) plus per-precision keys and
`e30/meta/{instance,price_h,engine,version}`; a plain-language appendix paragraph with one table
(`paper/sections/app_e30_cost.tex`, label `app:e30`). No main-text file is changed by E30's agent (the coordinating
session re-plots Figure 4a from these keys).

(h) *Caps.* One new host, g6.xlarge (tag `jev-e30-cost`, E3's clone image, SSM only), idle stop after 30 idle minutes and
a power-off guard 6 hours after its first boot (6 host-hours); no LLM API call; weights from Hugging Face and S3 through the
host's role. Deadline 2026-10-10 12:00 UTC. An "E30: execution as run" entry records what ran before any paper text uses
it.

### Deviations 2026-10-09 ~17:05 UTC: "E30: OpenJev-27B through its official 4-bit GGUF" (exploratory; amends E30 (d) 5 before any OpenJev-27B download, load or answer)

(a) *What was read before this entry.* E30's host status lines up to its setup stage (host launched, code checked out);
no E30 answer, accuracy, rate or latency of any model.

(b) *Why.* E30 (d) 5 planned OpenJev-27B through vLLM's in-flight bitsandbytes 4-bit quantization of the original
weights with the card's helper unchanged. The publisher ships an official 4-bit file instead
(`openjev/openjev-GGUF` at `24a4776974ef7b6f906065fc33e5ad73275d3d1b`: `OpenJev-Q4_K_M.gguf`, 16.5 GB, converted from
`openjev/openjev` by llama.cpp b11147 and validated by the publisher), which fits 24 GB and runs on the same llama.cpp
server as StartLux-27B's row; the in-flight path would load 55 GB of original weights on a 16 GiB host through a
quantizer whose support for this hybrid architecture is unverified, and the helper needs vLLM's per-token-id
log-probabilities, which llama.cpp does not serve.

(c) *Amendment.* OpenJev-27B (`openjevhf`) runs from `OpenJev-Q4_K_M.gguf` (its sha256 checked against the publisher's
`MANIFEST.json`) on llama.cpp exactly as E30 (f)'s llama.cpp rule (P = 8 slots with the C = 1 latency, then P = 16; the
same image as StartLux-27B's row). Its read-out is the card's helper's text lane (`helper/shim.py` at `ac97900f`, E15's
settings: `READOUT_TARGETED=1`, `READOUT_T=0.85`, one lettering, no padding, no compaction) re-implemented against
llama-server: the prompt `State:\n{state}\n\nQuestion: {instructions}\nOptions:\n[A] {option}: {description}\n…\n\nAnswer
with the letter of the best option only.` as one user message through the checkpoint's chat template with thinking
off (the tokenizer and template of `openjev/openjev` at `ac97900f`, tokens without added special tokens, as vLLM's chat
endpoint tokenizes), the option letters' logits at the answer position (A–Z then a–z, read from llama-server's
full-vocabulary log-probabilities), softmax at 0.85, the answer's probabilities rounded to 4 decimals as the helper
returns them, then the harness's parse (`answer_from_wire`, `norm`). Precision: Q4_K_M only. Reference accuracy: E15's
`e9/openjevhf/acc` (the FP8 weights on vLLM through the helper); the agreement with E15's records
(`results/explore/e15_baselines/l1/s1_openjevhf.jsonl.gz`) is reported. The stage runs last, only with at least 45 guard
minutes left; if the server does not load the file, not run (stated). Nothing else changes.

### Deviations 2026-10-09 ~16:55 UTC: "E28: the fitted thresholds" (exploratory; written after the calibration answers and the fit of E28 (f) 7, before any E28 final-harness smoke or episode)

(a) *Times and host.* The registration was committed at 15:19:59 UTC (`23c685d`; its heading says ~15:50, the commit
time is authoritative), the code at 15:32:32 (`f269fce`), before the host launch. The host `<instance-id>` (g6e.2xlarge, one L40S 48 GB, us-east-1c; tag `jev-e28-sl08b`) was launched
at about 15:32 UTC; chain 1 (`rearm setup data train select temps e1 c_replay shift_l2 cal base_l2 shift shift_base
live_in`) started at 15:34. The amendment "E28: L1 cost on the untrained model's serving engine" was committed while
chain 1 ran its L2 stages (`379dbfc`, code `5321f7d`).

(b) *What was read before this entry.* The chain's status lines up to the calibration stage: the build (vocabulary
overlap 0; 14,000 rows; 2,014 of 4,063 agent states augmented), training (two epochs, 28 minutes, setting `nogc_3072`),
checkpoint selection on development data (epoch 2: E1 dev prefix 82.6 %, v1 dev 99.5 %; epoch 1: 81.2 / 98.4 %), the
exits of the L1 and C stages, the C replay's answer count, **the L2-shift quick lines (main rendering 95.6 %, surface-undone
96.9 %, task means over the 100 tasks)**, and the calibration answers and fit outputs. No E1 or C accuracy, no
vendor-engine timing and no live row of E28 had been run or read.

(c) *The answers and the fit* ($0; no LLM call). StartLux-0.8B (mixed) behind `serve_jev` (v2 venv) on :8081 answered all
2,719 calibration questions (`d46701e8…`) through the harness's asking path, 16:46-16:54 UTC, no error
(`results/explore/e28_sl08b/cal/answers_cal.jsonl.gz`). `derive` and the registered command (`e5_cal_replay.py fit`,
projection rule) gave **tool choice τ̂ = 0.9516002311101601** (1,125 paired decisions) and **arguments τ̂ =
0.9989569187164307** (1,328 paired). Neither is "always defer", so the final-harness rows of (f) 8-9 run as registered.
At τ̂ on the calibration states the model takes 40.0 % of the tool choices (450 of 1,125) and 47.1 % of the arguments
(625 of 1,328), with no accepted error; on all paired states it is right on 98.8 % of the tool choices and 97.8 % of the
arguments. The values are in `configs/harness/explore_e28_final.yaml` (`status: development`, `backends.e28`, with the
fit files' sha256) and `results/explore/e28_sl08b/cal/fits.json`, committed with this entry. No shifted state,
evaluation seed or E28 final-harness episode entered the fit; the shift row of (f) 9 uses the same thresholds (no
refit). Chain 2 (`wait_prev code vendor_e1 final_in final_shift final`) is sent after this commit. Nothing else changes.

### Deviations 2026-10-09 ~19:15 UTC: "E29: the go decision" (exploratory; the registered rule of E29 (e) applied mechanically to the pilot; written after the pilot, before any full-run episode, none of which will run)

The registered go rule (`scripts/explore_e29_port.py go`, run on each host from the two pilot summaries in S3) gave
**NO-GO** at 18:51:56 UTC (mini-SWE-agent's host; the pi host and the SWE-agent host computed the same decision from the
same files at 18:53:39 and 18:53:18). Inputs: **pi (`tb`)**: the fitted gate routes nothing (E29 (c) 3: max G ≤ 0), so
its gated arm was not run and its saving is 0 by the rule; pass false, not_worse false. **mini-SWE-agent (`mswe`)**: 30
pilot instances, 21 resolved in both arms (drop 0), saving 10.7 % (≥ E4r's 10.5 %, < 20 %); pass false, not_worse
true. GO needs pass in at least one harness, so E29 stops after the pilot as registered: no full run of any harness,
SWE-agent ran its smoke only. The decision file is `results/explore/e29_port/box/go.json` (S3 `e29/go.json`).

### Deviations 2026-10-09 ~19:15 UTC: "E29: execution as run" (exploratory, descriptive; written after the pilot, the go decision and the readout, before any paper text used them)

Execution record of exploratory row E29; it changes no rule, task list or record. (a) *Timing* (commit times are
authoritative; UTC, 2026-10-09). Registration `49f6344` 16:05:18; code `ab09dbf` 16:26:27 (the proxy, the adapters, the
task lists, the driver, the readout, offline tests); before any host episode `4204350` 16:32:17 (at the cap the proxy
refuses new episodes and lets running ones go on to a hard limit of cap + 2 USD while the host watchdog, now every 30 s,
stops the runs, so no trial fails on a refusal; the guard re-armed per paid stage; SWE-agent's build-cache pruner;
mini-SWE-agent's image pull split); readout code `af0c492` 16:36:27 (a task's Harbor trial is its latest attempt) and
`1c90037` 16:53:47 (post hoc calibration descriptives, not part of the fit). **Development smoke, local** (after
`ab09dbf`, 16:27-16:31): the proxy on this workstation with pi 0.99.2 (smoke_gated and shadow) and litellm's `gemini/`
path, Gemini direct and Jev through the author's Jev-only key (`decision/jev.py`; never on AWS); 0.031 USD; it showed
routing Flash -> Flash-Lite -> Flash with no thought-signature error. Three new m7i.8xlarge hosts in us-east-1d (SSM
only, no inbound; power-off guard and 30-minute idle stop armed): `tb` `<instance-id>`, `mswe`
`<instance-id>`, `swea` `<instance-id>`, launched 16:26:55-16:27:05; set-up 16:29-16:30. **pi (`tb`):**
smoke 16:32:55-16:36:13 (every check passed, 1.42 USD); calibration on the 10 Terminal-Bench Pro tasks 16:37:21-16:47:41
(no trial error; 5 of 10 solved); fit 16:48:01: **no step pays** (187 steps from 10 tasks; Flash-Lite chose Flash's
action in 36.9 % of the steps, a second Flash sample in 60.4 %; G is negative at every threshold, from the most
confident step down), so the pilot's gated arm was not run (16:48:05) and no pilot baseline either; the host waited for
the other pilot, computed NO-GO at 18:53:39, ran `final` and powered off. **mini-SWE-agent (`mswe`):** images of the
smoke, calibration and pilot lists 16:32:52-16:44:44 (51 pulled, 0 failures); smoke 16:44:53-16:50:53 (every check
passed, 0.045 USD); calibration on 20 SWE-bench Verified instances 16:51:43-18:02:27 (the last instance ran ~70 minutes
under repeated StreamLake HTTP 520 / provider-unavailable answers, which mini-SWE-agent retried); fit 18:02:27:
**τ̂ = 0.99** (735 steps from 20 instances; exact-command agreement 7.8 % for the fast LLM, 15.1 % for a second strong
sample; the 18 steps at c ≥ 0.99, 2.4 %, agreed 44.4 % / 33.3 %); pilot (30 instances × {baseline, gated}, 8 workers
per arm, concurrently) 18:02:36-18:51:51; go 18:51:56 (NO-GO). While it waited for the go, the host pulled the
evaluation list's images (stopped by hand at 19:03 after the NO-GO, 105 of the 200 handled: 14 present, 91 pulled; no run used them); `final`
19:03:38. **SWE-agent (`swea`):** smoke 16:33:03-16:38:53 (every check passed, 0.99 USD), then waited for the go,
computed NO-GO at 18:53:18, `final` 18:53:25, powered off. **All three hosts were read `stopped` at 19:08:01.** About
2.5 + 2.6 + 2.5 = 7.5 of 50 host-hours.

(b) *As run, against the registration.* Everything ran as registered, with these records: (1) the proxy's shutdown at
the end of pi's calibration wrote one late shadow record (16:48:04) after the fit had read the log (16:48:01); the fit
used 187 steps, the final log holds 188, and the fit on the final log gives the same result (no step pays); the readout's
descriptives use the 187 records the fit read. (2) mini-SWE-agent's proxy logged 13 HTTP-200 answers without
`usage.cost` (OpenRouter error bodies, not billed; cost 0), which mini-SWE-agent retried. (3) In mini-SWE-agent the
decision model was asked at every step of the gated arm although the gate routes only c ≥ 0.99 (as registered: the
question is asked ahead for every step). (4) After the runs, `<instance-id>` was taken out of the driver's list of
other rows' hosts (E27's offline test requires E27's driver to list every host id another driver names); the driver acts
only on the one host tagged `jev-e29-<H>`, so it never touched that host.

(c) *Caps.* Logged spend (proxy logs: LLM + decision model): pi 7.02 USD (direct Gemini 7.00, Jev 0.026),
mini-SWE-agent 5.34 USD (OpenRouter: DeepSeek 5.06, Jev 0.27), SWE-agent 0.99 USD (smoke), local smoke 0.03 USD:
**13.38 USD of the 45 USD pilot cap and of the 300 USD E29 cap**. The SSM OpenRouter key's usage went from 18.29 to 23.65
USD (5.37; it also carried every harness's Jev calls). **Leak check: 0 files in all 9 checks** (after every stage and at
`final` on each host); no API key entered a task container or a harness process (they got a dummy key and the proxy URL).

(d) *Readout* (`scripts/explore_e29_port.py readout` → `results/explore/e29_port/summary.{md,json}`, keys `e29/...` in
`paper/generated/e29_port.tex`), as registered: **pi on Terminal-Bench 2.0**: the gate kept every step on the strong LLM
(0 % fast; no saving by construction). **mini-SWE-agent on SWE-bench Verified (pilot, 30 instances)**: 70.0 % resolved
in both arms (2 wins, 2 losses; Δ 0.0 points [−13.3, +13.3]; non-inferiority not shown, p 0.577); 0.7 % of the requests
on the fast LLM (0.2 fast segments per instance, mean length 1.5, none ended by the cap); cost 0.0459 against 0.0514 USD
per instance, a 10.7 % saving whose interval [−62.9, 42.2] % shows it is trajectory noise, not routing (the decision
model alone cost 0.0057 USD per instance, 11 % of the baseline's cost); mean time 3.3 % longer, median 11.2 % longer; the
decision model took 8.1 s per instance, of which requests waited 1.06 s (asked ahead, the rest overlapped the actions).
Post hoc, descriptive (the calibration records; not part of the fit): Jev's confidence hardly separated the steps where
Flash-Lite chose Flash's action in pi (AUROC 0.52; at c ≥ 0.5, 65.8 % of the steps, 40.7 % against 65.0 %); in
mini-SWE-agent it separated them somewhat (AUROC 0.70) but exact commands rarely repeat (7.8 % / 15.1 %). **Reading:
"porting the full harness to open-source harnesses cut cost substantially without losing success" does not hold on
this pilot:** with a gate fitted on the fast LLM's agreement with the strong LLM, pi routed nothing and mini-SWE-agent
under 1 % of its steps, so the full harness saved nothing beyond noise. This session wrote
`paper/sections/app_e29_port.tex` (label `app:e29`) and changed no main-text section, figure, `main.tex` or
`appendix.tex`.

### Deviations 2026-10-09 ~19:55 UTC: "E29b: step-type routing with a write guard" (exploratory, descriptive; a registered follow-up to E29, written after E29's pilot, go decision and readout were read; registered before any E29b model call, episode or host start)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E29b)" and marked in the
paper like the other post-Stage-1 rows. It has no confirmatory test, revises no registered or reported result (E4, E4b,
E4r, E25 and E29 stand as reported) and changes no profile, threshold, key or record of any other row. **Written after
E29's pilot was read:** E29's registration, its go note and its execution entry, `results/explore/e29_port/summary.md`
and E29's calibration records (`results/explore/e29_port/box/{tb,mswe}/proxy/cal`). **Nothing has run for E29b:** no
E29b model call, episode or host start; the only E29b code so far is the read-only classifier of (c), written before
this entry and applied, at $0 and without any model call, to E29's calibration records (below).

(a) *Why.* E29's gate was labelled by exact agreement between the fast LLM's next action and the strong LLM's. Actions
rarely repeat exactly (a second strong sample agreed with the first in 60.4 % of pi's steps and 15.1 % of
mini-SWE-agent's), so the fitted gate routed nothing (pi) or 0.7 % of the requests (mini-SWE-agent): no saving, no loss.
E29b changes the label from "the fast LLM would do the same" to "the next step only reads", which a mechanical guard can
enforce at run time: the decision model predicts the KIND of the next step, the fast LLM writes predicted inspection
steps, and the harness executes a fast-LLM action only if it is read-only. *Feasibility, computed before this entry from
E29's calibration records with the classifier of (c) (whitespace-normalised first-action keys; descriptive, not a
result of E29b):* the strong LLM's action was read-only in 39.9 % of pi's 188 calibration steps, which carried 16.8 % of
the strong LLM's cost (inspection comes early, when the context is short), and in 53.0 % of mini-SWE-agent's 745 steps
(50.0 % of the cost). Sending every read-only step to the fast LLM at E29's logged prices would have cut the strong
LLM's cost by at most 13.3 % (pi) and 32.4 % (mini-SWE-agent), before the decision model (0.7 % and 10.1 % of the
strong cost in E29's calibration), the precision rule of (d) and the guard's discarded calls. **The registered
expectation is therefore that pi cannot reach the 20 % saving of the go rule and mini-SWE-agent only if most of its
read-only steps are routed;** E29b runs the pilot as registered whatever the expectation.

(b) *The decision layer* (extends E29's `src/jevharness/adapters/route_proxy.py` / `route_formats.py`; everything E29
registered in (b) holds unless changed here: one host-side proxy per harness that holds the keys, harness processes and
task containers get a dummy key and the proxy URL, decide-ahead while the previous action runs with a 10 s wait and
fail-open, the first request of an episode on the strong LLM, a retried request reuses its decision, the
thought-signature fix, a failed fast request is sent to the strong LLM once, JSONL logs without keys or bodies, spend
caps inside the proxy, E25's leak check after every stage). New arm settings (`question`, `guard`, `shadow_roles`; E29's
arms keep their defaults):
1. *Typed question* (Jev's `choice` type; the state is E29's: the task, the last 8 steps with truncated outputs and the
   action in progress): "Which kind of step comes next, after the result of the action in progress comes back?" with
   options `inspect` (read files or run a read-only command: list, view, search, git status / diff / log), `edit` (edit,
   create or delete a file), `run` (run tests, a build or other code) and `finish` (finish or submit the task). The
   answer is Jev's most probable option; the confidence c is Jev's probability of `inspect`.
2. *Routing:* step t + 1 goes to the FAST LLM iff Jev's answer is `inspect` and c ≥ τ̂ (the harness's fitted gate, (d));
   otherwise the STRONG LLM writes it. *Sticky:* after a fast step the next step's question is still asked and decides
   the next step; a run of consecutive fast steps is capped at **k = 6**, after which the strong LLM takes the next step.
3. *Write guard* (mechanical, harness side): a fast-LLM response is buffered in the proxy (a streamed call is read to
   its end before anything reaches the harness) and returned only if it is read-only by the classifier of (c); otherwise
   it is discarded (its cost counts and is logged with `guard = reject`), the segment ends and the strong LLM writes the
   step from the same request. The strong LLM's responses are never guarded.
4. *Typed shadow mode* (calibration only, never on an evaluation task): the strong LLM acts at every step; after each
   response the identical request goes once to the fast LLM (non-streaming, never executed; `shadow_roles = [fast]`, no
   second strong sample) and the typed question is asked ahead. Every primary and shadow call record carries the
   classifier's verdict (`ro`) on all tool calls of its response.
Arms: `baseline` (mode strong: the unchanged harness), `gated` (question step_type, guard on, k = 6, τ = τ̂), `shadow`
(typed shadow mode), `smoke_gated` (τ = 0, development only).

(c) *The read-only classifier* (`route_formats.read_only_calls`; the same function gives the calibration label and the
guard's verdict). A response is read-only iff it has at least one tool call and every tool call is read-only. Tool
level: the harnesses' viewer / search tools (pi `read`, `grep`, `find`, `ls`; SWE-agent `str_replace_editor` with
`command = view`, and the windowed tools `open`, `goto`, `scroll_up`, `scroll_down`, `find_file`, `search_file`,
`search_dir`) are read-only; a shell tool (`bash`) is read-only iff its command is; every other tool (pi `edit`,
`write`; SWE-agent `str_replace_editor` create / str_replace / insert / undo_edit, `submit`; anything else) and a response
without a tool call are not. Command level (shlex with operators split, comments not stripped, newlines as separators):
no command substitution (`$(`, backticks), no process substitution, no redirection except to `/dev/null` and fd
duplication (`2>&1`), no here-document, no background `&`, no subshell or brace group, no `VAR=value` prefix; segments
joined by `&&`, `||`, `;`, newlines or pipes, and every segment's command on the allow-list: `ls`, `cat`, `head`,
`tail`, `less`, `more`, `wc`, `pwd`, `grep` / `egrep` / `fgrep`, `rg` (no `--pre`), `find` (no -exec, -execdir, -ok,
-okdir, -delete, -fprint, -fprint0, -fprintf, -fls), `sed` with -n and only print commands (`[addr[,addr]]p` or `=`, no
-i, no -f), `awk` (no -f / -i; the program without `>`, `|`, `system`, `getline`, `close(`, `fflush`), `git status /
diff / log / show` (no `-c`, no `--output`), `tree` (no -o), plus `cd` (navigation), `nl` (mini-SWE-agent's viewer) and
the pure filters `sort` (no -o), `uniq` (at most one operand), `cut`, `tr`, `diff`, `file`, `stat`, `du`, `which`.
Anything the classifier cannot parse is not read-only. Not read-only by construction: rm, mv, cp, touch, mkdir, chmod,
tee, xargs, patch, apply_patch, python, pytest, make, pip, npm, echo / printf (mini-SWE-agent submits with `echo
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`), sudo and every other command. Offline unit tests fix the classifier, the guard
and the routing state machine before any E29b call.

(d) *Calibration and the fitted gate.* E29's calibration logs hold the strong LLM's first actions (keys with
whitespace collapsed) but no step-type answer, so E29b collects new ones on **the same calibration splits** in typed
shadow mode: pi on E29's 10 Terminal-Bench Pro tasks (`configs/explore/e29_port/tb_cal.txt`), mini-SWE-agent on E29's
20 SWE-bench Verified instances (`mswe_cal.txt`), SWE-agent (only on GO) on E29's 15 (`swea_cal.txt`); disjoint from
every evaluation list. *Label* of step t ≥ 1 (steps with a Jev answer and a parsed strong response): y_t = 1 iff the
strong LLM's response at step t is read-only by (c). *Fit* (`scripts/explore_e29b_route.py fit`): the eligible steps are
those where Jev's answer is `inspect`; candidate thresholds are their observed c; precision P(τ) = mean y over eligible
steps with c ≥ τ; **τ̂ = the lowest candidate τ with P(τ) ≥ 0.95**; if none, the gate routes nothing (that harness's
gated arm is not run; saving 0 and drop 0 by construction). The fit reports τ̂, P(τ̂), the number of steps it routes
and the **routed share it implies** (routed steps / all calibration steps), the recall of read-only steps, and,
descriptively, the share of the fast LLM's shadow responses at the routed steps that the guard would accept. Written to
`gates/gate_<harness>.json` (S3 `e29b/gates/`) before any evaluation episode of that harness; no refit later.

(e) *Smoke* (development only, never reported): pi on E29's Terminal-Bench Pro smoke task and mini-SWE-agent on E4r's
smoke instance, each in `smoke_gated` (τ = 0) and `shadow`; SWE-agent (only on GO, before its calibration) on E25's two
smoke instances likewise. Checks: no infrastructure error; no LLM error passed to the agent; Jev answered ≥ 95 % of the
typed questions; at least one fast call that the guard judged (accept or reject) in `smoke_gated`; the shadow trial
logged a verdict for the primary and the fast shadow call; cost > 0; leak check 0 files. A failed smoke stops that
harness (one re-smoke after an implementation fix, recorded).

(f) *Pilot* (≤ $40 including smoke and calibration: pi $35, mini-SWE-agent $5): E29's pilot tasks, pi the first 30 of
Terminal-Bench 2.0 (`tb_pilot.txt`; Gemini 3.5 Flash strong / Flash-Lite fast, direct Gemini API, minimal thinking,
pi 0.99.2 through Harbor as in E29) and mini-SWE-agent 2.4.6 the first 30 of E4r's extension list (`mswe_pilot.txt`;
DeepSeek-V4-Pro reasoning on / V4-Flash reasoning off, StreamLake through OpenRouter with the SSM inference key; E4's
caps), each in `gated` and a fresh `baseline`, concurrently on the same host (pi 6 trials per arm, mini-SWE-agent 8
workers per arm, as E29). Saving = 1 − mean cost(gated) / mean cost(baseline) over the 30 tasks (every proxy-logged LLM
call including discarded fast calls, plus the decision model); drop = successes(baseline) − successes(gated). **Go rule**
(mechanical, `scripts/explore_e29b_route.py go`): pass(h) iff saving ≥ 20 % and drop ≤ 3; ok(h) iff drop ≤ 3; **GO iff
(pass(pi) and ok(mini-SWE-agent)) or (pass(mini-SWE-agent) and ok(pi))**. A harness whose gate routes nothing has saving
0 and drop 0; a harness without a pilot summary (smoke, calibration or pilot failed) has pass and ok false. The decision
is written to S3 (`e29b/go.json`) and recorded in a Deviations note; on NO-GO E29b stops and reports the pilot.

(g) *Full run* (only on GO, automatically, the pilot not pooled): pi on all **89** Terminal-Bench 2.0 tasks, mini-SWE-agent
on a registered **100-instance subset** (the first 100 of E4r's extension list, `configs/explore/e29b_route/mswe_full.txt`:
the pilot's 30 afresh plus the next 70), and SWE-agent v1.1.0 (Harbor, E25's config) on E25's **100** instances (Gemini
3.5 Flash / Flash-Lite) after its smoke, calibration and fit; each harness whose gate routes something runs `gated` and a
fresh `baseline` concurrently on the same host (pi 6, mini-SWE-agent 10, SWE-agent 6 per arm), with τ̂ from its own
calibration. One re-run of trials with an infrastructure error (E25's / E4r's rule) within the caps. **Caps** (proxy-
enforced, cumulative per host, stage caps after the pilot's): pi $95, mini-SWE-agent $14, SWE-agent $60 (its smoke and
calibration included); **E29b in total ≤ $250 and E29 + E29b ≤ $300** (E29 spent $13.38); OpenRouter part ≤ $25 and the
SSM key's own hard limit. Hosts: E29's three stopped hosts (`jev-e29-tb`, `jev-e29-mswe`, `jev-e29-swea`, restarted;
SSM only; the chain re-arms the power-off guard and the 30-minute idle stop first), ≤ 40 host-hours; a host watchdog per
harness stops its runs at the cap; every host stops when done (verified). If a watchdog stops a run, the readout covers
the tasks complete in both arms and says so. The leak check (files under the run directories and logs that contain a
key value) runs after every stage and must find 0 files, else that host's chain stops.

(h) *Readout* (`scripts/explore_e29b_route.py readout` → `results/explore/e29b_route/summary.{md,json}`,
`paper/generated/e29b_route.tex`, keys `e29b/...`; appendix section label `app:e29b` in
`paper/sections/app_e29_port.tex`), whatever the outcome, per harness and stage (full set primary: an errored trial is a
failure; complete case beside it): n; success of `baseline` and `gated`; Δ in points with wins / losses and the 95 %
task-bootstrap interval (10,000 resamples, seed 29); non-inferiority at 3 points (`metrics.ni_pvalue`, `bb_ordered`;
shown iff p ≤ 0.05); cost saving with its 95 % bootstrap interval; mean- and median-time saving; the fast-LLM share of the
executed steps; guard rejections (count, share of the fast calls, their cost); sticky segments (per task, mean length,
share ended by k); decision-model calls, cost, latency and the time requests waited; fail-open counts; the gate (τ̂,
precision, implied and realised routed share); spend against the caps; the leak checks. Descriptive; no claim beyond the
measured n. An "E29b: execution as run" entry records what ran before any paper text uses it.

### Deviations 2026-10-09 ~20:10 UTC: "E28: execution as run" (exploratory, descriptive; written after the runs and the readout, before any main-text file used them)

(a) *Commits and times.* Registration `23c685d` (15:19:59 UTC; its heading says ~15:50, the commit time is
authoritative), code `f269fce` (15:32:32), the amendment "E28: L1 cost on the untrained model's serving engine"
`379dbfc` (16:25:13) and its code `5321f7d` (16:29:28), "E28: the fitted thresholds" `be525c7` (16:57:25), the interim
readout `1ac1c62` (written while the last row ran), this entry and the final readout `(the commit that adds this entry)`.

(b) *Host.* One new host, `<instance-id>` (g6e.2xlarge, one L40S 48 GB, us-east-1c; tag `jev-e28-sl08b`;
us-east-1d had no g6e.2xlarge capacity), launched at about 15:32:50 UTC from E3's clone image with a 300 GB volume;
SSM only; power-off guard 470 minutes after the first boot, 30-minute idle stop. Chain 1 (`rearm setup data train
select temps e1 c_replay shift_l2 cal base_l2 shift shift_base live_in`) ran 15:34-17:35 with every stage exit 0;
chain 2 (`wait_prev code vendor_e1 final_in final_shift final`) waited for it and ran 17:35-19:55. The host powered itself off after `final` (instance-initiated shutdown, 19:57:33 UTC); its state was read `stopped` at 20:03:57 UTC.
Host time: about 4.4 (4.37 at the final stage) of 8 host-hours. No API key other than the Gemini key (from SSM into the live stages' process
only) was on the host; every live stage's leak check found 0 files. `OPENROUTER_API_KEY_JEV` never went to AWS.

(c) *Base and data, as registered.* StartLux-Decision-0.8B at revision `bd4f76a600e23227547fee7bfc1825e12a32764c`
(downloaded and confirmed on the host, licence `cc-by-nc-4.0`; weights sha256 `4305599c…`, recorded in
`box/startlux_08b_sha256.txt` with the licence file). E12's mix re-checked (`fcdb3b9a…`, 14,000 rows), split into its
7,000 agent states and 7,000 public items; E27's augmentation changed 3,463 rows on 2,014 of the 4,063 agent states; the
built mix (`2e517052…`, 14,000 rows) is byte-identical to the build run locally before the host launch. **E27's
vocabulary check: overlap 0** (augmentation 193 words, sha256 `1e86eb00…`; E21 1,011 words, `0012a84b…`;
`results/explore/e28_sl08b/vocab_check.json`); E27's seed guard passed on every row before and after augmentation.

(d) *Training and selection, as registered.* Setting `nogc_3072` (no out-of-memory error), two epochs, 398 steps, 28
minutes. Selection on development data only: epoch 2 (E1 dev prefix 82.6 %, v1 dev 99.5 %; epoch 1 81.2 / 98.4 %).
Temperatures on E12's cal_dev (`box/e28_calibration.json`). Adapter sha256 in `box/selected_sha256.txt`; adapters in
`s3://…/checkpoints/e28_sl08b/` (no optimizer state). Derived weights: CC BY-NC 4.0, research use only.

(e) *Evaluation, each once, as registered and amended* (`scripts/explore_e28_sl08b.py summarize` →
`results/explore/e28_sl08b/summary.{md,json}`, keys `e28/…` and `e9/e28/…` in `paper/generated/e28_sl08b.tex`).
- L1 (E1's 3,991 decisions, served by `serve_jev`, no error): **80.5 %** (untrained StartLux-0.8B 70.0 %, paired
  +10.5 [+9.1, +11.9]; E27's SP 77.2 %, +3.2 [+2.1, +4.3]; AgentDecision-4B (E12) 85.1 %, -4.6 [-5.8, -3.5]; E12c 85.5 %,
  -5.0 [-6.1, -4.0]); BFCL tool choice 87.1 % (untrained 88.2), arguments 87.2, When2Call 58.3. **Cost by E9's rule:
  on the base model's engine (the amendment) $0.0101 per 1,000 decisions, median 18 ms per request** (untrained, E9's
  row: $0.0127, 17 ms); the merge used all 186 adapter modules (scale 2) and the vendor's kernel check was active
  (package `a9101e9…`); that run answered all 3,991 with 80.8 % accuracy (a check only, the vendor's prompt). On E28's
  own stack (`serve_jev`): $0.179 per 1,000, median 456 ms (AgentDecision-4B on its stack $0.140, 344 ms).
- L2 in-domain (C, 200 tasks, no missing answer): **98.5 %** (the 4B checkpoint before public data 99.2 %; paired
  lower end -1.2 points, so by E5's rule not kept at the 1-point margin; untrained 48.3 %; E27's SP 99.2 %; E12c 99.6 %).
- L2 shift (main rendering, 1,093 questions, no error): **95.6 %** (untrained 51.7 %; AgentDecision-4B 94.9; E27's SP
  96.3; E12c 98.5), must-ask 93.5 % (untrained 27.2; AgentDecision-4B 67.4), surface cost +1.2 points [-0.3, +2.8]
  (untrained +5.7).
- L3 shift (REPLACE, 200 tasks, no errored episode): **82.0 %** (untrained StartLux-0.8B 13.0 %, n 200, no errored
  episode; E27's SP 80.0, paired +2.0 [-5.3, +9.3]; AgentDecision-4B 91.5, -9.5 [-14.5, -5.0]; E12c 98.0); generator
  kinds 96.4 % (140 tasks), new kinds 48.3 % (60).
- L3 in-domain (REPLACE, 200 tasks): **100.0 %** (untrained 17.5 %, the E2 ladder's row).
- Thresholds: as committed in `be525c7` (tool choice 0.9516, arguments 0.99896).
- Final harness in-domain (200 tasks, 1 worker, no errored episode): **100.0 % vs 99.5 %** for the logged LLM-only row
  (1 task gained, 0 lost; Newcombe +0.5 [-1.4, +2.8]; non-inferior at 3 points, p 0.002); tool choices taken 35.9 %, arguments 66.9 %, no accepted
  wrong decision; API cost 9.6 % lower, but **6.0 % [1.6, 10.7] higher with the GPU tariff** (q = $1.624 per hour × wall); 14.9 s
  per episode against 13.6 s. Descriptive split: the model's answers took 2.4 s per episode, the LLM 12.4 s (6.4 calls
  at 1.93 s each; the LLM-only reference's calls took 1.55 s each), so the wall-time and GPU-cost comparison carries
  the API's latency at run time.
- Final harness on the shift: **98.0 % vs 100.0 %** for E21's logged LLM-only row (0 gained, 4 lost; Newcombe -2.0 [-5.0, +0.2]; non-inferiority not shown, p 0.306); tool choices taken 61.7 % (6 accepted wrong; at the must-ask states 43 of 123 taken, none wrong), arguments 31.9 % (2 accepted wrong); 33.2 % lower API cost and 13.2 % [7.6, 18.2] lower with the GPU tariff; 23.8 s per episode against 8.0 s, of which the model 2.2 s and the LLM 20.9 s (6.0 calls at 3.51 s each against 1.20 s in the reference): the API was slow during this run, so its wall time, and the GPU share of its cost, compare with the reference's only descriptively.
- Not run: nothing registered was left out.

(f) *Readings (registration (i)).* Holds under shift: L2 yes (95.6 ≥ 94.9 - 3), L3 no (lower end -14.5 < -5), so **no**.
Keeps its in-domain level: C no (lower end -1.2 < -1.0), the in-domain final harness non-inferior; so **no** by the
registered conjunction. Descriptively against E27's SP: higher at L1 (+3.2) and L3 shift (+2.0, interval includes 0),
about the same at L2 (C -0.7, shift -0.7), and in the final harness in-domain non-inferior where E27's SP was not, but
without its saving (E27's SP: 30.3 % with the GPU, served by the faster scoring-head engine); on the shift E27's SP
handed more tool choices back (27.7 % taken) and stayed non-inferior, E28 took 61.7 % and was not shown non-inferior.

(g) *Caps.* Direct Gemini $26.65 (runs $26.64, probes $0.003) of $30 (watchdog never fired); no OpenRouter, Jev or OpenAI call; host time as
in (b); deadline 2026-10-10 02:00 UTC met.

(h) *Records.* Box records in `results/explore/e28_sl08b/box/` (the raw calibration answers stay in S3; their gzip copy
is in `results/explore/e28_sl08b/cal/`), slim live rows in `results/explore/e28_sl08b/live/`, full runs in
`s3://…/e28_sl08b/runs/`. Nothing else
deviates from the registration and its two amendments.

### Deviations 2026-10-09 ~20:50 UTC: "E30: E15's open baselines on the same basis" (exploratory; extends E30 (b) at the coordinating session's request before any measurement of these models; written while E30's StartLux-27B stage runs)

(a) *What was read before this entry.* E30's results up to Intern-Decision-4B (every vLLM row of E30 (b): rates, costs,
accuracies and agreements) and the StartLux-27B stage's progress lines; nothing of the models below.

(b) *Why.* The coordinating session, for the author: Figure 4a should draw every open decision model at or above 70 %
on one basis, so E15's open baselines it draws join E30: OpenJev (`openjev`), JevK5 (`jevk5`), decider-4b (`decider`),
Clef-Flash (`clefflash`) and Imajev-4B (`imajev`) (OpenJev-27B is already in E30). Models under 70 % (Bosun, CLM, Laya,
Verdict, E14's heads) stay out. The host-hour cap becomes 10 h on the same g6.xlarge (guard re-armed 600 minutes after
the first boot); a g6e.xlarge (one L40S 48 GB, $1.861 per hour on demand, the same price list) only for a model that
cannot fit 24 GB even at 4 bits. Deadline unchanged. The coordinating session also asks for the frontier of Figure 4a's
drawn points on the new basis as keys (`e30/front/{first,rest,n}`, E9's rule over the open models' E30 points and the
API and LLM points at their E9 keys); this is a readout addition.

(c) *Amendment.* Each model at the weights E15 used, with E30's workload, quantities, precision rule and keys
(`e30/<slug>/...`, E9's slugs); reference accuracy `e9/<slug>/acc`, agreement against E15's records
(`results/explore/e15_baselines/l1/s1_<slug>.jsonl.gz`).
1. **decider-4b** (`Mapika/decider-4b` at `eb5fbdfc`, Qwen3.5-4B): vLLM, BF16 and FP8, its own row construction
   imported from the checkpoint's `decider/` package (the plain state-first layout of its `decider_config.json`, every
   question its own row as its server's default), read at the row's last position over the option labels its engine reads
   from the LM head, softmax at the choice temperature (1.11), probabilities rounded to 4 decimals as its answer
   formatter returns them; then the harness's parse.
2. **JevK5 v0.2** (`alibiserikbay/JevK5` at the Hub tag `v0.2`, commit `ea4804e9`): vLLM, BF16 and FP8, the `jevk5`
   package 0.2.2 (github allebee/jevk5, tag `v0.2.2`): SemIf's JSON payload through the chat template with thinking off,
   the answer letters' logits at the last position, softmax at the checkpoint's temperature (1.532); a choice of more than
   16 options by the package's own knockout (`jevk5.prompt.spread`) with only the forward pass replaced; the package's
   answer, then the harness's parse.
3. **Clef-Flash 9B** (`Cloudflare/clef-flash` at `fde727a2`): its scoring head is not served by vLLM, so, as E30's
   scoring-head row, the model card's `systemone()` behind E15's wrapper (`scripts/aws/e15_wrap_server.py --kind clef`,
   BF16, one request on the GPU at a time as the card's function runs) on the same host, E1's groups as requests at
   C = 16 and 64 and one question per request at C = 1 (E30 (f)'s scoring-head rule); marked.
4. **Imajev-4B** (`mohit67890/imajev-4b` at `f8d8234c`): its read-out is a trained 255-row code readout over four
   rotated presentations of the options; it runs only if a vLLM port is ready and checked before 00:30 UTC: the LoRA merged
   into Qwen3.5-4B, the readout rows written into an untied LM head at the code tokens, the repository's prompt code
   (github mohit67890/imajev at `ccf586d4`), four reads per decision combined by its own `combine_rotations`; otherwise not
   measured (its own server needs about 2.9 s per decision as E15 logged it, beyond the cap).
5. **OpenJev** (DiffusionGemma-26B-A4B, image `razorback16/openjev:0.5.0` with `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic`,
   27.2 GB): no int4 file that its read-out (a masked-slot read of a diffusion language model, served only by its own image)
   loads exists, so it does not fit 24 GB; it runs, if host time allows after every row above, on one g6e.xlarge (tag
   `jev-e30-cost-l40s`, at most 2 host-hours, idle stop and power-off guard) through its image as E15 ran it, E1's groups
   as requests, priced at the g6e.xlarge's $1.861 per hour; marked; otherwise not measured.
Answers go through the harness's own parse as in E30 (d). An "E30: execution as run" entry records what ran.

### Deviations 2026-10-09 ~22:00 UTC: "E29b: the go decision" (exploratory; the registered rule of E29b (f) applied mechanically to the pilot; written after the pilot, before any full-run episode, none of which will run)

The registered go rule (`scripts/explore_e29b_route.py go`, run on each host from the two pilot summaries in S3) gave
**NO-GO** at 21:51:04 UTC (mini-SWE-agent's host; the pi host and the SWE-agent host computed the same decision from the
same files at 21:51:09 and 21:52:24; `e29b/go.json` written 21:51:07). Inputs: **pi (`tb`)**: 28 pilot tasks complete in
both arms (the pilot stopped at its registered cap, see the execution entry), 15 solved by `baseline` and 13 by `gated`
(drop 2), saving 15.5 %: pass false (< 20 %), ok true (drop ≤ 3). **mini-SWE-agent (`mswe`)**: 30 instances, 20
resolved in both arms (drop 0), saving −21.1 % (the gated arm cost more): pass false, ok true. GO needs pass in one
harness, so E29b stops after the pilot as registered: no full run of any harness; SWE-agent made no API call. The
decision file is `results/explore/e29b_route/box/go.json` (S3 `e29b/go.json`).

### Deviations 2026-10-09 ~22:00 UTC: "E29b: execution as run" (exploratory, descriptive; written after the pilot, the go decision and the readout, before any paper text used them)

Execution record of exploratory row E29b; it changes no rule, task list or record. (a) *Timing* (commit times are
authoritative; UTC, 2026-10-09). Registration `b697f55` 19:59:05; code `b0c415d` 20:10:47 (the typed question, the write
guard and the typed shadow mode in the proxy, the read-only classifier, `scripts/explore_e29b_route.py`, E29's driver with
`ROW=e29b`, the registered 100-instance subset, offline tests); interim readout `5df328a` 21:40:42 (below). E29's three
stopped m7i.8xlarge hosts were started at 20:11:33 (`tb` `<instance-id>`, `mswe` `<instance-id>`, `swea`
`<instance-id>`; SSM online 20:11:58) and the chains sent at 20:12:37; each chain re-armed the power-off guard and
the 30-minute idle stop first, updated to `b0c415d` and ran the E29 and E29b offline tests (passed on all three hosts by
20:12:55). **pi (`tb`):** smoke 20:12:48-20:14:54 (every check passed; the decision model answered inspect 3, edit 8,
run 14, finish 8 times), calibration on E29's 10 Terminal-Bench Pro tasks 20:15:04-20:24:23 (205 steps), fit **τ̂ =
0.92** (4 routed steps, 2.0 % of the steps, precision 100 %), pilot 20:24:32-21:04:30, go 21:51:09 (NO-GO), `final`
21:51:17, powered off 21:51:19. **mini-SWE-agent (`mswe`):** the smoke, calibration and pilot images were all present
(51 cached); smoke 20:12:54-20:16:52 (every check passed; inspect 14, edit 9, finish 4), calibration on E29's 20
SWE-bench Verified instances 20:17:01-21:04:30 (709 steps), fit **τ̂ = 0.980** (6 routed steps, 0.8 %, precision 100
%), pilot 21:04:39-21:51:01, go 21:51:04 (NO-GO); while it computed the go, the host pulled the full subset's images as
the driver does (159 s; 70 present, 30 pulled; no run used them); `final` 21:53:50, powered off 21:53:52. **SWE-agent
(`swea`):** waited for the go (no smoke, no calibration, no API call), NO-GO at 21:52:24, `final` 21:52:32, powered off
21:52:34. **All three hosts were read `stopped` at 21:55:33.** About 1.7 host-hours each, 5.2 in all (of 40).
**Credential gap:** the local AWS login expired at about 21:36 UTC and was renewed by the author at about 21:55 UTC
(this session never ran a login). The hosts ran their chains on their own throughout, with no change sent; before the gap
the session had fetched pi's records up to its pilot (21:05) and both gate files, and committed an interim readout of
those (`5df328a`, marked interim in a note that is removed with this entry); after the renewal it verified the hosts,
fetched every host's records (21:55) and wrote the readout below.

(b) *As run, against the registration.* Everything ran as registered, with these records: (1) **pi's pilot stopped at
its registered cap** ($35 for smoke, calibration and pilot): the host watchdog fired at 21:04:04 UTC, when 28 of the 30
tasks had finished in both arms; `rstan-to-pystan` and `mcmc-sampling-stan` were cancelled in both arms; no trial failed
on a refusal (0 infrastructure errors in either arm). The go rule and the readout use the 28 tasks complete in both arms,
so pi's drop is counted of 28, not 30. (2) mini-SWE-agent's proxy logged 13 HTTP-200 answers without `usage.cost` in
the pilot (6 baseline, 7 gated, all strong-tier; OpenRouter error bodies, not billed; cost 0), which mini-SWE-agent
retried, as in E29. (3) The smoke and calibration ran in the typed shadow mode with one fast shadow call per step, as
registered; the gated arm's question was asked at every step. (4) The paper's appendix section `app:e29b` was appended
to `paper/sections/app_e29_port.tex` in the interim commit, guarded so that it renders only once the readout holds the
go decision (this entry's commit). (5) Two action keys in pi's pilot proxy log (task `sanitize-git-repo`, step 8 of the
baseline trial and step 11 of the gated trial) held a Hugging-Face-token-shaped string from the task's own data (the task
plants fake secrets for the agent to remove; it matched no `.env` value, and the host's leak check, which scans for the
real keys, found 0 files); the repository's export scan refuses token shapes, so the two strings were redacted in the
committed log (marked in the records; no number depends on them). The interim commit `5df328a` carries the unredacted
log in history.

(c) *Caps.* Logged spend (proxy logs: LLM + decision model): pi 34.64 USD (direct Gemini 34.52, Jev 0.12) of its 35 USD
pilot cap, mini-SWE-agent 3.95 USD (OpenRouter: DeepSeek 3.69, Jev 0.26) of 5 USD, SWE-agent 0: **38.59 USD of the 40
USD pilot cap and of the 250 USD E29b cap; E29 + E29b 51.97 USD of 300 USD.** The SSM OpenRouter key's usage went from
23.65 to 27.72 USD (4.07; it also carried pi's Jev calls). **Leak check: 0 files in all 9 checks** (pi 4, mini-SWE-agent 4,
SWE-agent 1); no API key entered a task container or a harness process.

(d) *Readout* (`scripts/explore_e29b_route.py readout` → `results/explore/e29b_route/summary.{md,json}`, keys `e29b/...`
in `paper/generated/e29b_route.tex`), as registered. **Calibration:** in pi 24.9 % of the strong LLM's next steps were
read-only; the decision model answered inspect at 98 of 205 steps, and those were read-only in 25.5 % (25.8 % after an
edit answer); in mini-SWE-agent 48.4 % were read-only, inspect was answered at 531 of 709 steps, read-only in 53.7 %; the
95 % rule left 2.0 % and 0.8 % of the steps. **pi on Terminal-Bench 2.0 (pilot, 28 tasks):** 53.6 % vs 46.4 % solved
(2 wins, 4 losses; Δ −7.1 points [−25.0, +10.7]; non-inferiority not shown, p 0.702); the fast LLM wrote 6.2 % of the
executed steps (63 of 1,017); the guard stopped 46 of 109 fast-LLM actions (42.2 %; 0.0051 USD discarded per task); cost
0.4668 vs 0.5524 USD per task, 15.5 % lower [−2.4, 34.9]; mean time 8.8 % lower, median 24.0 % lower; the decision model
cost 0.0033 USD and 7.2 s per task, of which requests waited 4.53 s. **mini-SWE-agent on SWE-bench Verified (pilot, 30
instances):** 66.7 % resolved in both arms (1 win, 1 loss; Δ 0.0 [−10.0, +10.0]; non-inferiority not shown, p 0.774);
1.2 % of the executed steps on the fast LLM (15 of 1,246), none stopped by the guard; cost 0.0470 vs 0.0388 USD per
instance (21.1 % more, [−76.1, 16.5] % saving), the decision model alone 0.0055 USD and 7.8 s per instance (1.24 s
waited); mean time 6.0 % longer, median 8.3 % longer. Post hoc, descriptive (the calibration records; not part of the
fit): AUROC of p(inspect) for a read-only next step 0.56 (pi) and 0.70 (mini-SWE-agent); read-only steps carried 14.1 %
(pi) and 41.9 % (mini-SWE-agent) of the strong LLM's calibration cost. **Reading:** step-type routing with a write guard
does not cut cost substantially without losing success in these two open-source harnesses on this pilot. The guard
stopped every writing action of the fast LLM, but the decision model's step-kind answer hardly predicted read-only steps,
so the registered 95 % precision rule routed 1-6 % of the executed steps; pi's lower cost comes with two fewer solved
tasks and an interval that includes zero, and in mini-SWE-agent the decision model's own calls cost more than the routing
saved. This session changed no main-text section, figure, `main.tex` or `appendix.tex`.

### Deviations 2026-10-09 ~23:25 UTC: "E30: JevK5 at the weights E15 loaded" (exploratory; corrects E30's amendment "E15's open baselines" (c) 2 before any answer of these weights; written while E30's OpenJev-27B stage runs)

(a) *What was read before this entry.* Every E30 row read out so far (`8b4a581`: the vLLM rows, the Qwen3.5-0.8B variant,
OpenJev on the g6e.xlarge, StartLux-27B at P = 8), including JevK5 v0.2's two rows: 73.1 % (BF16) and 72.7 % (FP8) against
E15's 70.3 %, answers agreeing with E15's record on 86.7 % and 87.3 % of the 3,991 decisions, where every other model
agrees on at least 96.1 %. Nothing of the weights below.

(b) *Why.* The amendment named the Hub tag `v0.2` because E15's row is labelled "JevK5 v0.2" (the tag's object `ea4804e9`
resolves to commit `27d2d6b8`, the weights E30 measured). E15's own server log (`s3://…/e15_baselines/box/logs/docker_jevk5.log`,
2026-10-07 18:51 UTC, read after the readout flagged the agreement) shows that the image `razorback16/openjev-jevk5:0.5.1`
loaded `alibiserikbay/JevK5` at `revision=main`; main has been commit `c4f7fdb3` (the v0.3 release, tag `v0.3` →
`41a82c11` → `c4f7fdb3`, 2026-09-25) since before E15 ran. The image installs the `jevk5` package at `0571ef37` (the
`v0.2.2` tag E30 uses) and carries no weights. So Figure 4a's JevK5 point is v0.3's, and E30's amendment intent ("each
model at the weights E15 used") is v0.3. v0.3's `jevk5_config.json` holds `temperature` 1.22 (and a
`knockout_temperature` 0.93 the 0.2.2 package does not read); the package reads only `temperature`, as E15's image did.

(c) *Amendment.* JevK5 is measured again at commit `c4f7fdb3aeab5582336406e78d3bef11bf98833d` with everything else of
(c) 2 unchanged (vLLM, BF16 and FP8, the `jevk5` package 0.2.2's prompt and knockout, the temperature the package reads
from the checkpoint, E30's workload, quantities and precision rule; stage `jevk5m`, records `runs/jevk5m_{bf16,fp8}`).
Key slug `jevk5` (the one Figure 4a reads) takes these rows; the v0.2 rows stay in the readout as a separate descriptive
row `jevk5v02` ("JevK5 v0.2, the registered tag, not E15's weights"), never drawn in Figure 4a and outside the frontier
keys. Order on the main host: the running stage (OpenJev-27B at P = 8) finishes; the chain's loop is ended by its process
id only, and a new chain runs `wait_prev code jevk5m clefflash sl27b16 openjevhf16 final` with every guard unchanged
(power-off guard 595 minutes after the first boot, 02:45 UTC; idle stop 30 minutes; host-hour cap 10 h) and `MIN_LEFT`
55 minutes, so a llama.cpp P = 16 pass starts only when it can end before the guard. Answers go through the harness's
own parse as in E30 (d). The "execution as run" entry records what ran.

### Deviations 2026-10-10 ~01:00 UTC: "E30: execution as run" (exploratory, descriptive; written after the last stage and the readout, before any main-text file used them)

Execution record of exploratory row E30 and its amendments. It changes no rule, model list or record.

(a) *Commits and times* (UTC; commit times are authoritative where a heading's approximate time differs). 2026-10-09:
registration `eb7f94c` (16:31:41; heading ~16:35), code `0ea92f9` (16:48:46), the amendment "OpenJev-27B through its
official 4-bit GGUF" `989d243` (16:56:11; heading ~17:05) and its code `1b42163` (16:58:26); driver and bench fixes
before the first measured row: `180346d` (16:59:51, the download record no longer fails on a directory without a GGUF
file) and `5e483d9` (17:05:32, vLLM's capacity set before any measurement); `77bd41d` and `aeb60ef` (20:39:53 and
20:40:50, stage `wait_prev`); the amendment "E15's open baselines on the same basis" `a6acb7c` (20:48:19; heading ~20:50)
and its code `4771236` (20:50:15); `fb36e99` (21:13:07, one llama.cpp stage per slot count), `10b2b01` (21:14:37, the
g6e.xlarge), `100f128` (21:57:26, Clef-Flash's venv), `86a9589` (22:00:41, `wait_prev`'s pipeline-status bug); interim
readouts `7f653ee` (21:43:22) and `8b4a581` (23:16:43); the amendment "JevK5 at the weights E15 loaded" `c2e4470`
(23:21:11) and its code `34cb64a` (23:23:48); the interim readout `fd3489c` (23:52:44, written while the last stages
ran); this entry and the final readout (the commit that adds this entry).

(b) *Main host.* `<instance-id>` (g6.xlarge, one NVIDIA L4 24 GB, 4 vCPU, 15 GiB, us-east-1c; tag
`jev-e30-cost`), launched 16:50:10 from E3's clone image (300 GB volume), first boot 16:50:23, SSM only, no API key.
Power-off guard 355 minutes after the first boot, re-armed at 20:38 to 595 minutes (02:45:23 on 2026-10-10) for the
amendment's 10-hour cap; idle stop 30 minutes throughout. Chains as run:
1. 16:51 `rearm setup smoke`: the smoke stopped at the download record (a directory without a GGUF file, before any
   prompt or model load; fixed in `180346d`); 17:00 `code smoke`: the engine path passed on 300 decisions (never a
   result).
2. 17:05 `code e12 sl08b e28 rl sl4b e12c sl2b sl9b intern sl27b e27sp openjevhf final`: every vLLM row of the
   registration ran 17:06-20:05 with exit 0. StartLux-27B's stage (then P = 8 and P = 16 in one stage) began at 20:05.
3. To use the raised cap, the loop of chain 2 and its 27B stage shell were stopped at about 20:40 (the P = 8 bench went
   on, orphaned), the guard was re-armed (20:38) and a chain `code wait_prev sl27b e27sp openjevhf` (20:41) waited for
   the bench. At 21:10:57 that chain's `wait_prev` passed while the bench still ran, and at about 21:13 every E30 process on the
   host was stopped (both chains, the bench and llama-server) so that E15's baselines and the 27B rows could run one slot
   count per stage (`fb36e99`). **The first StartLux-27B P = 8 pass was lost then**: its file was never
   written and none of its answers was read.
4. 21:13 `code decider jevk5 clefflash e27sp sl27b openjevhf sl27b16 openjevhf16 final`: decider-4b 21:13-21:31,
   JevK5 at the `v0.2` tag 21:31-21:53, Clef-Flash's first attempt stopped at load at 21:56 (the v2 venv has no image
   processor class for the checkpoint; no answer), E27 SP 21:56-22:04.
5. A chain sent at 21:58:34 (after `100f128`) passed its `wait_prev` at once (the pipeline-status bug), stopped every GPU
   logger at 21:58:37 (E27 SP's GPU log ends there) and updated the code; it was stopped by its script id within two
   minutes, before any model load (`86a9589`). E27 SP's timed passes were then running on the GPU alone.
6. 22:01 `wait_prev code sl27b openjevhf clefflash sl27b16 openjevhf16 final`: waited for E27 SP (22:05), StartLux-27B
   at P = 8 22:05-23:12 (67 minutes), OpenJev-27B at P = 8 from 23:12:55 (the publisher's GGUF checked against its
   manifest, the bench from 23:17) to 00:26:26 (73 minutes, the bench's rc 0; the stage's own exit line was not written because its chain's loop had been ended, step 7).
7. Per the JevK5 amendment the loop of chain 6 was ended by its process id before 23:25 (its OpenJev-27B stage went
   on), and a chain `wait_prev code jevk5m clefflash sl27b16 openjevhf16 final` (23:25:36, `MIN_LEFT` 55) waited behind
   it.
8. After the second container restart (below), this session ended the waiting chain of step 7 by its script id at
   23:47:35 (only that chain's two shell processes; the OpenJev-27B stage and its llama-server were not touched) and
   sent `wait_prev code jevk5m final` at 23:48:00 with every guard unchanged (`GUARD_MIN` 595, `MIN_LEFT` 55, idle 30
   minutes). Reason: the coordinating session set about 1.5 hours for E30's end, and Clef-Flash's corrected stage (a new
   torch venv, then about an hour of one-at-a-time 9B passes) and the two P = 16 passes would have ended after it. JevK5
   at commit `c4f7fdb3` ran 00:26:46-00:48:54 (exit 0); `final` at 00:48:55 (power-off scheduled for 00:50:57); the host powered itself off (instance-initiated
   shutdown) and was read `stopped` at 00:57:36. Host time 7.98 (7.976 at the final stage) of 10 host-hours.

(c) *L40S host.* `<instance-id>` (g6e.xlarge, one NVIDIA L40S 48 GB, us-east-1c; tag `jev-e30-cost-l40s`),
launched 21:22:59 for OpenJev only (amendment "E15's open baselines" (c) 5), a power-off guard about two hours after its
first boot (112 minutes left at its rearm) and the 30-minute idle stop; chain `rearm setup_min openjev final` 21:26-21:42 (OpenJev through its image at
E15's digest with the FP8 weights, 21:26-21:41, exit 0), `final` at 21:41:57, powered off at 21:43:59; read `stopped`
at 23:43 and again at 00:57:36. 0.31 of 2 host-hours.

(d) *Session gaps.* The local AWS login lapsed from about 21:36 to 21:55 UTC and was renewed by the author (no login
by any session). The coordinating session's E30 agent was ended by two container restarts, at about 22:25 and about
23:35 UTC; this session took over at 23:43. In each gap the hosts ran their chains on their own and no command was
sent; records were fetched from S3 after each gap.

(e) *As run, against the registration and its amendments.*
1. **Measured** on the main host, each precision re-answering all 3,991 decisions: AgentDecision-4B, the frozen GRPO 4B,
   StartLux-4B + our recipe, AgentDecision-0.8B (E28), StartLux 0.8B / 2B / 4B / 9B, Intern-Decision-4B, decider-4b
   and JevK5 at the `v0.2` tag (BF16 and FP8 on vLLM 0.31.0); JevK5 at E15's commit `c4f7fdb3` (BF16 and FP8, exit 0);
   StartLux-27B and OpenJev-27B at Q4_K_M on llama.cpp with P = 8 slots (exit 0 for OpenJev-27B); the Qwen3.5-0.8B
   variant (E27 SP) on our stack. On the L40S host: OpenJev through its own image (FP8).
2. **Not measured:** Clef-Flash 9B (its first attempt stopped at load before any answer; the corrected stage did not run,
   for time, step 8); Imajev-4B (no vLLM port was ready and checked by the amendment's 00:30 UTC limit); the P = 16
   passes of both 27B models (registration (f)'s llama.cpp saturation check, for time), so their rate, cost and plotted
   point come from P = 8 alone.
3. The `v0.2` rows of JevK5 stay in the readout as the descriptive row `jevk5v02`, outside the summary counts, the
   frontier keys and Figure 4a, as the JevK5 amendment says.
4. The `sl4b` stage printed "sha256 differs from E12c's record": the two safetensors files and `config.json` have the
   same sha256 as E12c's record; the record also lists `tokenizer.json`, which E30's stage does not hash.
5. StartLux-9B at BF16 ran with 64 sequences, the bench's registered fallback after the engine refused 256 (`5e483d9`);
   every other vLLM row ran with 256.
6. AgentDecision-0.8B's reference is `e28/l1/acc` (registration (e)); its "x below E9's basis" uses `e28/l1/usdk`, the
   vendor engine's cost.

(f) *Readout* (`scripts/explore_e30_cost.py summarize` -> `results/explore/e30_cost/summary.{md,json}`,
`preds/*.jsonl.gz`, keys `e30/*` in `paper/generated/e30_cost.tex`). Fifteen models have a plotted point (JevK5's `v0.2` rows aside), twelve at FP8. All 11 models measured at both BF16 and FP8 on vLLM kept FP8 under the 1-point rule, and the largest BF16-FP8 accuracy difference among them is 0.6 points. Every plotted accuracy is within 0.7 points of its reference, and every plotted row agrees with its reference record on at least 96.1 % of the 3,991 decisions, so no plotted row is flagged. **Plotted points** (precision; L1 accuracy % (reference); decisions/s at C; USD per 1,000 decisions; C = 1 p50 ms): AgentDecision-4B FP8 84.8 (85.1), 50.2 at 256, 0.0045, 31; the frozen GRPO 4B FP8 78.6 (78.8), 50.2 at 256, 0.0045, 31; StartLux-4B + our recipe FP8 85.5 (85.5), 30.3 at 64, 0.0074, 45; AgentDecision-0.8B (E28) FP8 80.5 (80.5), 122.5 at 64, 0.0018, 27; StartLux 0.8B / 2B / 4B / 9B FP8 69.3 (70.0) / 76.0 (75.7) / 83.0 (83.1) / 83.3 (83.4), 0.0017 / 0.0027 / 0.0068 / 0.0114; StartLux-27B Q4_K_M 87.5 (87.7), 1.35 at P = 8, 0.166, 649; Intern-Decision-4B FP8 73.5 (73.4), 18.0 at 64, 0.0124, 61; the Qwen3.5-0.8B variant (E27 SP) BF16 77.3 (77.2), 26.7 at C = 16, 0.0084, 60; OpenJev-27B Q4_K_M 83.0 (82.7), 1.26 at P = 8, 0.178, 744 (agreement with E15's record 98.4 %); decider-4b FP8 79.4 (79.3), 37.1 at 64, 0.0060, 39; **JevK5 at E15's weights** FP8 70.1 (70.3), 25.1 at 64, 0.0089, 46 (agreement with E15's record 96.4 % at FP8 and 99.4 % at BF16, against 86.7 / 87.3 % for the `v0.2` rows, which confirms the amendment's reading of E15's server log; at BF16 70.3 % with E15's three family accuracies to the decimal); OpenJev (L40S, its own image, priced at 1.861 USD per hour) FP8 79.1 (78.8), 45.0 at C = 64, 0.0115, 144. JevK5 `v0.2` (descriptive, not drawn): FP8 72.7 %, 0.0091. **Frontier of Figure 4a's drawn points on this basis** (E9's rule; the open models at their E30 point and at least 68 %, Jev, OpenAI Decisions, Flash, Flash-Lite and Flash's own tool call at their E9 keys): `sl08b` -> `ours08` -> `ours` -> `jev` -> `sl27b` -> `flash` (`e30/front/{first,rest,n}`, n = 6), against `sl08b` -> `jev` -> `flash` on E9's basis. This session changed no main-text section, figure, `main.tex` or `appendix.tex`. The coordinating session re-plots Figure 4a from these keys.

### Deviations 2026-10-10 ~04:40 UTC: "E31: GRPO in the agent loop without SFT (RL only, from the untrained base)" (exploratory, descriptive; registered before any E31 code, training step, model answer, episode or host launch)

Exploratory row, outside both registered stages and every Holm table, labelled "exploratory (E31)" (post-Stage-1;
marked in the paper like the other post-Stage-1 rows). It has no confirmatory test, revises no registered or reported
result and changes no profile, threshold, key or record of any other row. The model it trains is a NEW checkpoint,
**"Qwen3-4B + GRPO only"** in the records (served as `rl-only-e31`, slug `e31`, keys `e31/rlonly/…`). Nothing has run
for E31: no E31 code exists, no model has been trained or asked a question for it, no E31 episode exists and no E31
host has been launched. **Written after these were read:** AgentDecision-4B v1's training records
(`results/gpu_round5/`: freeze record, chain decision, `grpo_metrics.jsonl`, spend), the Stage-2 attribution rows
(`results/stage2/stage2.json` `descriptive.stage2_rl`: C accuracy untrained 50.1 %, + SFT v1 99.2 %, v1 99.2 %), E9's L1
rows (71.4 / 78.4 / 78.8 %) and E21 with its addendum 2 (shift L2 52.9 / 98.4 / 98.6 %, must-ask 27.6 / 95.0 / 90.8 %,
shift L3 13.5 / 96.0 / 96.0 % for untrained / + SFT v1 / v1).

(a) *Why.* A reviewer's question, relayed by the coordinating session (2026-10-10): the paper trains AgentDecision-4B
v1 by SFT on the agent's own states and then GRPO in the agent loop, and finds that SFT gives nearly all of the gain
and GRPO adds no measurable gain; SFT labels may be biased, so would RL alone, without SFT, do as well or better?
**E31 runs v1's GRPO stage from the untrained base instead of from the SFT model** and measures the result as the
ladder's other rows were measured. Whatever it shows is reported; a run without learning signal is itself the answer.

(b) *Arm (one run, seed 0).* `configs/train/grpo_online_e31_rlonly.yaml` = `configs/train/grpo_online_round5.yaml`
with exactly these changes: **no init** (a fresh LoRA, r 32 / α 64 / dropout 0 on all linear layers, the shape of
v1's adapter); the **KL reference is the base model** (the LoRA disabled; v1's was its init, SFT v1) at v1's β 0.02;
**15 iterations** (v1's frozen checkpoint is iteration 15 of its 30; the learning rate is constant after 3 warm-up
steps, so a 15-iteration run has the schedule of v1's first 15 iterations); the run name and output directory.
Unchanged: the training tasks (JevGym v2, seeds 50000-52999, the task order of seed 0, so each iteration draws the 8
tasks v1 drew in the same iteration), G = 8 episodes per task, the four arms round-robin over task groups (REPLACE at
the tool and argument choices with the stop guard at 0.5; ASSIST top-3; CASCADE at τ 0.9; Flash with the model as tool
and stop guard), the agent LLMs (Gemini 3.5 Flash-Lite at temperature 0.7, Flash at 0.3 for the guard arm, minimal
reasoning, direct Gemini API), the reward of Eq. 2 in its gated form (λ 0.5, N0 = 10 LLM calls, irreversible-error
penalty 0.2), PPO clip 0.2 / 0.28, GiGPO anchors, dropping zero-variance groups, the decision-level loss, lr 1e-5, 16
workers, the decider's temperature 1.0 over the restricted read-out (the softmax over the option codes, so every
sampled answer is a valid option and the model decides from its first step), and the greedy development evaluations
(seeds 1000-1047, two arms) at iterations 0, 10 and 15. Base: `Qwen/Qwen3-4B` at revision `1cfa9a72` (v1's base, the
image's pinned snapshot). Environment: the harness venv of E3's clone image (torch 2.14.1, transformers 5.18.0; peft
and every version recorded on the host); `training/grpo_online.py` is unchanged since v1's training commit `cf767b1`.
The cold-start pitfall of `docs/TRAINING_AGENT_BRIEF.md` (a fresh LoRA in a cascade arm defers whenever no option
reaches τ, which carries no gradient) can affect only the cascade arm (one task group in four); it is recorded, not
avoided, since changing the arms would depart from v1. Before training, a pipeline smoke with the simulated executor
(1 iteration, 2 tasks × 2 episodes, no API call, a scratch directory, never reported).

(c) *Training diagnostics* (per iteration, from `metrics.jsonl` and `rollouts.jsonl`; reported whatever they show):
reward, success, LLM calls, the share of zero-variance task groups, decisions in the rollouts, **decisions the model
made** (all decisions minus cascade deferrals, from the per-arm decision counts and the cascade deferral rate),
updated decisions, the cascade deferral and forced-deferral shares, decision accuracy against the answer key (logged,
never rewarded), KL to the base, entropy, and greedy development success. **Degeneracy rule:** if three consecutive
iterations update no decision (no learning signal), training stops and E31 reports a degenerate run with the iteration
reached; the evaluation of (d) then runs on the last saved adapter (saved at iterations 5 and 10), and if none exists
it is not run.

(d) *Evaluation of the iteration-15 adapter, each run once, as the ladder's rows were measured.*
0. Temperatures by `training/calibrate.py --min-bucket 200` on `data/train_v1/dev.jsonl.gz` (as the untrained, SFT v1
   and v1 rows); diagnostic only: `eval_decisions` on the three v1 dev files (development data). Served by
   `training/serve_jev.py` with the adapter and temperatures on :8081 as `rl-only-e31` (price model `jevagentrl-4b`).
1. **L1:** E1's 3,991 decisions (`configs/explore/e31_rl_only/e1_e31.yaml` = E12's `e1_e12.yaml` with this backend and
   a new cache): pooled accuracy by E9's rule.
2. **C (in-domain common states):** `results/stage2/s23` (200 tasks) through `scripts/stage2_corpus.py replay`, a new
   cache; task-mean accuracy all and after an off-plan step (E5's `c_rows`); paired task bootstrap (B = 2,000, seed 0)
   against the logged answers of the untrained, SFT v1 and v1 rows.
3. **L2 shift:** E21's corpus, main rendering, `explore_e21_shift.py replay` with the answers in E31's own directory
   (E21's directory and keys are not touched); E21's task-mean metrics and subsets (all, must-ask, after an off-plan
   step, renamed-tool states, new- and generator-kind tasks); paired task bootstrap against E21's logged answers of the
   untrained, SFT v1 and v1 rows. No axis re-renderings.
4. **L3 shift:** REPLACE at the tool choice (the model picks every tool, Flash writes the arguments) on E21's 200 L3
   tasks (seeds 330100-330299), Flash on the direct Gemini API (temperature 0, minimal reasoning), 4 workers, a new
   System-1 cache, as E21 addendum 2 (d); E21 (g)'s pause rule (probe rounds; pause on an overload error), one re-run of
   errored episodes; a smoke on E21 development seeds 339070-339071 (never reported) and the gate: spend so far + 1.2 ×
   the smoke's $ per episode × 200 ≤ the Gemini cap of (f), else the first k tasks that fit (reported with its n).
   Success (latest record per task, errored tasks left out and counted), own-state accuracy, success per kind, $ per
   episode; Newcombe intervals against the untrained, SFT v1 and v1 rows on the same tasks.

(e) *Readings* (descriptive, one seed, one setting). A table of untrained / RL only (E31) / + SFT v1 / SFT + GRPO (v1)
on L1, C, L2 shift (all, must-ask) and L3 shift, with each training's GPU-hours (the GRPO stage's minutes / 60, as
`recipe/stage/grpo/gpu_h`) and API $ (rollouts + greedy evaluations, `scripts/train/r5_spend.py`'s rule). Per metric,
the share of SFT v1's gain over the untrained model that RL alone reaches, (E31 − untrained) / (SFT v1 − untrained).
**"RL alone matches SFT"** iff its L2-shift accuracy is at least SFT v1's minus 3 points and, if L3 ran, the lower end
of the paired Newcombe 95 % interval of its L3-shift success minus SFT v1's is at least −5 points (E27's and E28's
margins); otherwise it is reported as not matching. One seed and v1's budget and hyper-parameters only: a longer or
retuned RL run is not tested and the readout says so.

(f) *Caps and infrastructure.* **One NEW GPU host**, launched by `scripts/aws/e31_rl_only.sh` (tag
Name=jev-e31-rlonly; image `<aws-resource-id>`, E3's clone; g6e.2xlarge, else g6e.4xlarge, else g6e.xlarge: one
L40S 48 GB, the GPU of v1's GRPO stage, each tried in every us-east-1 default subnet), SSM only, **at most 10
host-hours** (a power-off guard 590 minutes after the first boot, a 30-minute idle stop, the host stopped by its last
stage and its stopped state verified from the container). Training wall cap 390 minutes; a run cut by it is evaluated
at its last saved adapter and reported with that iteration. **Direct Gemini at most $45 in total**: training
(rollouts and greedy evaluations) at most $30 by a watchdog on the logged spend, which stops the trainer; the live
watchdog and gate count the training spend. **No OpenRouter, Jev or OpenAI call.** The Gemini key comes from Parameter
Store into the training and live stages' processes only, is never printed or written, and a key-leak check runs over
the records after each such stage; no other key is on the host and none enters a task container. The driver refuses
every host it did not create; its forbidden list (`configs/explore/e31_rl_only/forbidden_iids.txt`) names every other
instance of the account, `<instance-id>` and `<instance-id>` included. Records to `s3://…/e31_rl_only/`,
the adapter to `s3://…/checkpoints/e31_rl_only/` (no optimizer state), slim copies to `results/explore/e31_rl_only/`.
**Deadline: results committed by 2026-10-11 12:00 UTC.** Order: training, temperatures, L1, C, L2 shift, then L3
shift; if L3 would exceed a cap or the deadline it is not run (L2 and C only, reported so). `scripts/stage2_spec.py
--gate` has no entry for exploratory rows and does not apply.

(g) *Readout.* `scripts/explore_e31.py summarize` → `results/explore/e31_rl_only/summary.{md,json}` and
`paper/generated/e31_rl_only.tex` (keys `e31/rlonly/…`: `gpu_h`, `api_usd`, `l1/acc`, `c/acc`, `l2/all`,
`l2/mustask`, `l3/success`, `zero_std_share`, `decisions_made` and the diagnostics of (c); a key that was not measured
is not defined); an appendix subsection `paper/sections/app_e31_rl_only.tex` (label `app:e31`, every number a `\res{}`
key), input by `appendix.tex` and `main.tex` behind file guards; no main-text file is changed by E31's agent (the
coordinating session integrates). An "execution as run" entry follows the runs.

### Deviations 2026-10-10 ~15:30 UTC: "E31: execution as run" (exploratory, descriptive; written after the runs and the readout, before any main-text file used them)

Execution record of "E31: GRPO in the agent loop without SFT (RL only, from the untrained base)"; it changes no rule,
row, threshold or record. (a) *Timing* (UTC, 2026-10-10; commit times are authoritative). The registration `82213b0`
04:29:14 (its heading's "~04:40" is approximate; the commit time stands); the code `a4eedf3` 04:49:51, before any
training step, model answer or host launch. The host `<instance-id>` (g6e.2xlarge, one L40S 48 GB, us-east-1d,
E3's clone image) was launched at 04:50:01 (first boot 04:50:16) with the code bundle pushed at 04:50:05. Chain 1
`rearm setup smoke`: guard re-armed at 588 minutes and the idle stop at 30 (04:52:40); setup 04:55:42 (torch 2.14.1,
transformers 5.18.0, peft 0.21.2, accelerate 1.15.0, the base snapshot at revision `1cfa9a72`); the simulated-executor
smoke ran end to end (1 iteration, 4 episodes, no update since no group had reward spread; never reported) and ended
05:08:02. The new volume loaded its blocks lazily (about 8 MB/s), so at about 05:06 one read-only parallel read of the
base model's and the harness venv's files was run through SSM to warm the volume; it changed no file. Chain 2 `train
select temps v1dev e1 c l2 l3 final` from 05:08:32. Training 05:08:37-06:44:13 (95.6 minutes), all 15 iterations, exit
0, no guard fired (spend, wall cap and degeneracy rule), peak GPU memory 31,125 MiB, key-leak check 0 files; the
iteration-15 adapter was selected (sha256 `814cfe02…`). Temperatures on v1 dev (global 6.47) 06:48:46; v1 dev diagnostic
06:54:53; L1 (3,991 answers) 06:58:01; C (3,939 answers, no error) 07:14:40; L2 shift (1,093 answers, no error, no
refusal) 07:20:23; L3 smoke on 339070-339071 07:21:08-07:21:18, the gate at 07:21:21 ("run all 200", projection $22.56
of $45), the row 07:22:02-07:25:26 (200 tasks, none errored, so nothing was re-run), every probe round clean, key-leak
check 0 files; `final` at 07:25:43 and the host's own shutdown at 07:26:43. **The AWS login of this container lapsed
from about 06:05 to 15:15 UTC**; the host ran the rest of chain 2 unattended under its guards, wrote to S3 with its
instance role, and powered itself off. Its stopped state (`Client.InstanceInitiatedShutdown`) was read from this
container at 15:17 and 15:21 UTC. No other host was touched.

(b) *Caps.* Direct Gemini **$22.24 of $45**: training $21.34 of its $30 (rollouts $20.28, greedy evaluations $1.06, by
`r5_spend.py`'s rule, which also prices the decider's tokens), the live smoke and row $0.89, probes below $0.001.
OpenRouter, Jev and OpenAI $0. GPU **2.6 of 10 host-hours** (launch to shutdown 2.61 h; the driver's count from the
first boot 2.59 h); the GRPO stage itself 1.6 h. Only the Gemini key was on the host, in the training and live stages'
processes. The deadline (2026-10-11 12:00) was not reached.

(c) *Readout* (`scripts/explore_e31.py summarize`; keys `e31/...` in `paper/generated/e31_rl_only.tex`), reported
whatever the outcome; untrained / + SFT v1 / v1 in brackets, recomputed from their logged answers by the same code and
equal to their logged keys. **RL only:** L1 71.4 % (71.4 / 78.4 / 78.8; paired gap to the untrained model +0.0
[-0.4, +0.4]); C 48.5 % (50.1 / 99.2 / 99.2; -1.6 [-2.4, -0.9] against the untrained model); shift L2 50.2 % (52.9 /
98.4 / 98.6; -2.7 [-4.1, -1.2]), must-ask 27.6 % (27.6 / 95.0 / 90.8); shift L3 11.5 % of 200 tasks (13.5 / 96.0 /
96.0; Newcombe -2.0 [-5.8, +1.7] against the untrained model, -84.5 [-89.0, -77.7] against SFT v1). Share of SFT v1's
gain over the untrained model that RL alone reaches: L1 0 %, C -3 %, L2 -6 %, must-ask 0 %, L3 -2 %. **Registered
reading: RL alone does not match SFT** (L2 48.3 points below SFT v1; the L3 lower end -89.0 < -5). Training
diagnostics: the model made 98.1 % of the 9,608 rollout decisions itself (the cascade arm deferred rarely); every
iteration updated some decisions (3,274 in all, against 5,882 in v1's first 15 iterations); 75.0 % of the task groups
had no reward spread (v1's first 15 iterations: 60.0 %); rollout success 38.0 % in the first three and 39.6 % in the
last three iterations; greedy development success of the REPLACE arm 12.5 % at iterations 0, 10 and 15; KL to the base
at most 0.88. The run is not degenerate by the registered rule; it learned little in v1's budget. Added to the readout
after the run (descriptive, from the same records): v1 dev accuracy (65.8 % against the untrained model's 70.1 % in
round 5's file), the reference rows as `e31/ref/...` keys, and the kinds of wrong L2 answers (premature `finish` is
47 % of the RL-only model's wrong answers against 33 % for the untrained model). `explore_e31.py fetch` now also unsets
`AWS_SESSION_TOKEN` (an operational fix, no effect on any number). Records: `results/explore/e31_rl_only/` (box records,
the training metrics and rollouts, the L1 / C / L2 answers gzipped, slim live copies; raw logs and the adapters on S3,
`s3://…/checkpoints/e31_rl_only/`).

(d) *Paper.* Appendix subsection `app:e31` (`paper/sections/app_e31_rl_only.tex`, input by `appendix.tex` after the E3
ablation behind a key guard) and `main.tex` loading `generated/e31_rl_only`; the coordinating session's guarded row of
the paradigms table reads the keys. No main-text section was changed by E31's agent.
