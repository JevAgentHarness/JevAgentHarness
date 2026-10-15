# Amendment A6 — code half (review round 5: P0-1, P0-3, P0-6, P0-8; rulings a, d, e)

Code companion of the A6 block in `docs/PREREGISTRATION.md` (owned by the writing side). Everything here is offline-tested
in `tests/test_a6_metrics.py`; the numbers quoted are hand-computed there and reproduce
`results/reviews/2026-10-01b/ni_power.out`. Nothing in this file is a result.

## 1. The NI test as one procedure (P0-1, ruling a) — `bench/metrics.py`

> **A6.14 (2026-10-01, review round 6):** the decision statistic is no longer the bootstrap p_NI. `ni_pvalue` now returns
> `p_tango` = 1 − Φ(Z(−m)) (Tango's one-sided score test on the paired wins / losses / n counts; `_tango_z`), the matching
> `tango_lcb_adj`, and `rule` ∈ {tango, exact}: the exact binomial on losses decides when `wins == 0` or fewer than 5 tasks
> are discordant. `p_decisive` (what enters Holm) follows that rule; `p_ni` / `lcb_adj` / `ni_boot_adj` are kept as
> descriptive values. Evidence: `results/reviews/2026-10-01c/ni_size_sim.md` (the bootstrap rule's size at the Holm floor is
> up to 2.3× α on losses-only data; the registered rule's is within Monte-Carlo error everywhere). The text below describes
> the pre-A6.14 procedure where it says the bootstrap decides.
>
> **A6.17 (2026-10-01, after the exact-size validation):** the score branch's decisive p is now `p_uncond`, the Berger–Boos
> exact unconditional p-value of the same Tango statistic (`ni_unconditional_p`: the tail P_w[Z ≥ z_obs, W ≥ 1, W + L ≥ 5]
> maximised over the win probabilities w in the (1 − γ) Clopper–Pearson set from the discordant count, plus γ = 1e-4);
> `rule` reads `tango_uncond` and `p_decisive = p_uncond`; `p_tango` (the asymptotic p_T) stays reported;
> `ni_pvalue(..., unconditional=False)` reproduces the A6.14 decision. Exact size and power tables:
> `scripts/ni_exact_size.py`, `scripts/ni_power_exact.py` → `results/reviews/2026-10-01d/`. The score card's NI table
> column is `p_U (rule)`.
>
> **A6.16 (i) (H2):** the primary H2 readout is `meta_gold.h2_theory_check` (`h2_report` key `h2_theory`; wording by
> `h2_theory_wording`): predicted vs observed ΔU per estimable cell with shared task ids resampled jointly across cells,
> a through-origin slope with its CI when ≥ 4 cells are estimable and the design has full rank, else *not estimable*;
> `h2_regression_j` is kept as `h2_regression_exploratory` (= the old `regression_j`). Where the text below calls the J
> regression "primary", read A6.16 (i).

| function | what it is |
|---|---|
| `ni_pvalue(base, arm, margin, n_boot=10000, seed=0, alpha=0.05, alpha_adj=None)` | **The registered test.** Statistic: the paired task-bootstrap distribution of Δ\* = mean_t(arm_t − base_t) (B = `n_boot`, task = resampling unit, trials of a task stay together because the inputs are per-task means) — the *same resampling* as the score card's Δ rows (`_paired_boot`, shared with `paired_bootstrap_diff`). Returns `p_ni` = P\*(Δ\* ≤ −m), `p_sup` = P\*(Δ\* ≤ 0) (the superiority p of the other primaries), `lcb95` (descriptive), `lcb_adj` = the (1 − α_adj) one-sided bound, `ci95`, `ni_adj` (p_ni ≤ α_adj ⇔ lcb_adj > −m), `ni_95`, `wins` / `losses` / `n_discordant`, `exact` (below 5 discordant tasks), `p_decisive` (the exact p then, else p_ni — what enters Holm), `mde`, `inconclusive`. |
| `ni_exact_check(wins, losses, n, margin, alpha)` | Exact one-sided binomial on the losses: p = P[L ≤ l \| n, m] (0.97^500 = 2.4·10⁻⁷ at zero discordance, m = 3 pt, n = 500) and the one-sided (1 − α) Clopper–Pearson upper bound on the loss rate, 1 − α^{1/n} at zero losses (0.6 pt at n = 500) — reported instead of a zero-width interval. |
| `ni_sample_size(d, m, alpha, delta=0, power=0.8)` | n = (z_{1−α}σ₀ + z_powerσ₁)² / (m + Δ)², σ₀² = d − m², σ₁² = d − Δ². 3 pt, d = 5 %, Δ = 0: **339** (α = .05), **611** (.05/8), **652** (.05/11). |
| `ni_power(n, d, m, alpha, delta=0, sims=0, n_boot, seed)` | Analytic power Φ(((m + Δ)√n − z_{1−α}σ₀)/σ₁): 0.91 / 0.70 / 0.66 at n = 500 and 0.50 / 0.20 / 0.17 at n = 149 for the three levels. With `sims > 0` the bootstrap rule itself is simulated (type-I at Δ = −m, power at Δ; the score test next to it). |
| `ni_mde(n, d, m, alpha, power=0.8)` | `delta_mde`: the smallest true Δ shown NI with the given power at the achieved discordance (n = 500, d = 5 %: a 0.53-pt true loss); `margin_mde`: the smallest margin the sample supports at Δ = 0. Printed next to every primary. |
| `holm_conjunction(p_jev, p_intern)` | = max (intersection–union test): one Holm entry for a claim confirmed on both System-1 families. |
| `ni_holm_levels(pvals, alpha)` | Holm step-down levels: for each test its adjusted p, α_adj = α/(m − rank + 1) at its step, and whether it rejects. |
| `newcombe_interval(both, only_a, only_b, neither, alpha)` | Newcombe (1998) method 10 (score-type, descriptive): Wilson limits of both marginals combined with the corrected φ (formula in the docstring). At zero discordance ±z²/(n + z²), not zero-width. |
| `tango_interval(..., one_sided=False)` | Tango (1998) score interval: {δ : \|Z(δ)\| ≤ z}, Z(δ) = (b − c − nδ)/√(n(2p̃₂₁ + δ(1 − δ))) with p̃₂₁ the restricted MLE (root of 2n p² − (b + c + δ(b − c − 2n)) p − cδ(1 − δ) = 0). Zero discordance: one-sided LB = −z²/(n + z²) by hand (0.54 pt at n = 500). |

**Score card** (`bench/scorecard.py`): `compute_scorecard(..., ni_margin=None)` / `--ni-margin` (default: `--margin` if
positive, else 0.03; τ² passes 0.05). Each arm's `vs_baseline` gains `ni` (the `ni_pvalue` block plus `p_ni_holm`,
`alpha_adj`, `lcb_adj`, `ni_holm`, `holm_rank`, `holm_m`) and `score_intervals` (Newcombe / Tango). Holm runs per reference
over `p_decisive` of every acting arm. Markdown gets a new section **"Non-inferiority (A6 procedure; margin x pp)"**
with columns `arm | reference | n | Δ pp | 95% LCB pp | p_NI | Holm p | α_adj | LCB(1−α_adj) pp | discordant (w/l) | exact p
(loss UB pp) | MDE Δ pp (80%) | Tango 95% LB pp | verdict` (verdict ∈ NI / *inconclusive at this n* / not NI). No
existing column changed; `delta`, `lcb`, `p_worse`, `p_worse_holm` are byte-identical.

## 2. H2 on J at the operating point (P0-3) — `bench/meta_gold.py`

| function | what it is |
|---|---|
| `h2_regression_j(cells, W=1, n_boot, seed, alpha)` | **Primary.** ΔU_gross on J(τ_op); `slope_J`, `slope_J_ci` (task-level bootstrap: the cells' shared tasks are resampled, each cell's ΔU and J recomputed from `task_data`), `slope_J_p_le_0`; models J-only / accuracy-only / J + accuracy with `lr_add_accuracy_p`, `lr_add_J_p`, `accuracy_explains_better` (smaller RSS of the one-regressor model); `falsified` = slope_J ≤ 0 or (accuracy better ∧ adding accuracy significant ∧ adding J not). |
| `predicted_gain(cell, W)` | Eq. 2 / Prop. 2: ΔÛ = W·n̄·π(1−π)·(e + h)·J − κ·n̄ (`pred_net`, `pred_gross`) with the observed ΔU next to it; computed twice: `net` (placebo-netted e, h) and `raw` (`e_raw` / `h_raw`: the branch rates against *executing the original proposal*). `efficacy_primary` = `raw` for guards (stop guard: the live alternative to a rejection is the proposal), `net` for pre-action gates. |
| `h2_regression_auroc(cells, W)` (alias `h2_regression`) | The former test, now secondary: AUROC is the τ-free upper bound on attainable J (max_τ J ≤ 2·AUROC − 1); keeps the LR test and the slope-1 / intercept-0 theory check. |
| `h2_cells(..., task_data=True)` | now also stores per cell `kappa`, `efficacy_primary`, `stats_no_budget` (P0-8 sensitivity) and the `task_data` the bootstrap needs. `voi_readout` adds `predicted_raw` and `efficacy_primary`. |

CLI: `python -m jevharness.bench.meta_gold h2 RUN --measure ... --pair LIVE:CONTROL` prints `regression_j`,
`predicted_gain` (per cell and W), `regression_auroc` (and the old `regression` key = the AUROC one).

## 3. τ̂\* calibration arms and the scalar projection (P0-6) — `bench/threshold_theory.py`

`python -m jevharness.bench.threshold_theory RUN --calibration-arm --arm jev_shadow_cal --dp tool_select
[--task-value 1.0 --lam-per-hour 0] [--w-grid 0.1 10] [--lam-grid 0.5 5] [--n-bins 10 --min-bin 20] [--out tauhat.json]`

* `fit_calibration_arm(recs, task_value=1, lam_per_hour=0, w_grid, lam_grid, n_bins=10, min_count=20, ...)`: the H1 fit on
  the calibration shadow arm's records (`decision_pairs`): V = W·(S_right − S_wrong) (`counterfactual_v`), dC from the
  ledger (`cost_terms`), then the projection. `primary` is the operating point **W = $1, λ = 0**; `sensitivity` holds
  W ∈ {0.1, 10} at λ = 0 and λ ∈ {0.5, 5} $/h at W = 1; `tau_hat` of `primary` is the C1 arm's `tau`. Task seeds are
  checked against 5000–5149 (`CAL_SEEDS`; a note names seeds outside it; `task_seeds` parses
  `jevgym-<family>-<seed>-…`).
* `scalar_threshold_projection(recs, V, delta_c, n_bins=10, min_count=20)`: Eq. 3's pointwise rule a(c) = g(c) − q_L(c) +
  dC/V ≥ 0 estimated in 10 **equal-mass bins of c with ≥ 20 decisions** (`equal_mass_bins`: smaller bins merged with the
  upper neighbour, the last with the lower; a bin without q_L or g falls back to the global value), projected onto one
  scalar: **τ̂\* = the smallest c at which Eq. 3 accepts and keeps accepting** — the lowest observed confidence of the
  lowest bin from which every higher bin has a(c) ≥ 0 (the accepting tail); +∞ (defer everything) when no bin accepts.
  When a(c) is monotone this is exactly the crossing of Eq. 3 (`monotone = True`, `crossing == tau_hat`); otherwise the
  isolated accepting bins below the tail are listed as `islands`, the pointwise rule is the secondary row
  (`pointwise.accept_rate`, `disagreement` = share of decisions on which the two rules differ), and `tau_gain` (the
  bin edge maximising the plug-in gain, the pre-A6 estimator of `plug_in_thresholds`) is reported next to it.
* H1's regret analysis is untouched (still the 30/70 split of the *evaluation* shadow log, `analyze`).

## 4. Oracle labels: `fatal_kind` and the implication audit (P0-8, ruling d)

* `JevGymEnv.oracle_verdict` (`bench/suites/jevgym.py`): a `fatal` verdict now carries `fatal_kind ∈ {terminal, budget}`
  — `terminal` when the proposed step itself ends the episode unsuccessfully (`r.done and not success`: proven in the
  strict world), `budget` when D(s′) = ∞ only through the canonical rollout (oracle-relative: the canonical plan is an
  upper bound on the optimum) — and `fatal_detail ∈ {terminal, budget_steps, plan_stuck, plan_fails}`
  (`_rollout_reason`). A read-only detour at budget = D is `fatal`/`budget`/`budget_steps`; at budget = D + 1 a `detour`;
  at budget = D − 1 the state is doomed.
* Propagation: `harness/agent.py::_need_gold` (reflection / routing, `verdict_src = executed`) and
  `_proposal_verdict` (tool guard), `harness/measure.py` (fired-step CF labels), `meta_gold.replay_golds`
  (`fatal_kind` per replayed step) — all as `gold_aux.fatal_kind`.
* `meta_gold.implication_audit(trajs)`: over every logged trajectory, executed action with verdict ∈ {fatal, premature,
  wrong_answer} ⇒ episode failed (`exec_fail`); executed finish with `finish_ok` ⇒ success, with `finish_ok = False` ⇒
  failure (`finish_ok` / `finish_not_ok`). Returns per-implication checked / violation counts (**`n_violations` must be
  0**), the violations, and `fatal_kind_share` (positives: `terminal`, `budget`, `finish`, `unknown`, `budget_share`).
* `exclude_budget_fatal(records)` and `gate_stats` on it (`h2_cells` → `stats_no_budget`), `ceiling(...,
  exclude_budget=True)`: the sensitivity row without budget-kind positives.
* CLI: `python -m jevharness.bench.meta_gold audit RUN --arm ARM --implications [--no-replay]` prints the replay audit,
  the implication audit, the budget-kind share and the `sensitivity_no_budget` gate stats; `ceiling --exclude-budget`.
* Real trajectories (runs/dev/e2b_dev_r3, 2026-10-01): `meta_shadow_s2lite` 46 checks / 0 violations (16 need_fatal
  positives, all finish-kind); `jev_reflect_trigger_s2lite` 7 / 0. The test `test_dev_round3_trajectories_pass_the_
  implication_audit` re-checks this whenever runs/dev is present.

## 5. Ledger fields (ruling e, additive) — `accounting.py`, clients, `bench/report.py`

`CallRecord` gains `attempts`, `retry_wait_ms`, `request_id`, `ts` (ISO-8601 UTC, set by `Ledger.add`) and
`tournament_round`; all default `None`, every existing field and `Ledger.summary()` unchanged. Fillers:
`llm/openai_compat.py` and `llm/openai_responses.py` (`ChatResponse.attempts / retry_wait_ms / request_id` from their
retry loops; latency still excludes backoff), `decision/jev.py` (attempts and backoff summed over the chunk requests
of a decision, `x-request-id`; carried by `Usage.attempts / retry_wait_ms / request_id`, additive, into
`decision/base.py`), `harness/agent.py::_choice_tournament` (round index on every chunk call, final round included).
`bench/report.py`: one new episode-table column **`retry_rate`** (retried calls / calls with a known attempt count;
`episode_retry_columns` also gives `retry_known_frac`, `retry_wait_s`); `errors` stays the error-rate column.

## What the writing side should reference

* PREREG A6 / BENCHMARK_SPEC §5.2, §8: `metrics.ni_pvalue` (p_NI, α_adj, `lcb_adj`, `p_decisive`), `ni_exact_check`,
  `ni_mde`, `holm_conjunction`; the score card's "Non-inferiority (A6 procedure)" section and `--ni-margin`.
* PREREG A6 H2: `meta_gold.h2_regression_j` (primary), `predicted_gain` (Eq. 2 check, `efficacy_primary` raw for guards),
  `h2_regression_auroc` (secondary bound); `meta_gold h2` CLI keys `regression_j` / `predicted_gain` / `regression_auroc`.
* PREREG A6 C1 / BENCHMARK_SPEC §8: `threshold_theory --calibration-arm` on `jev_shadow_cal{,_s2lite}` (seeds
  5000–5149), the projection definition above (tail of Eq. 3; bins 10 equal-mass ≥ 20, global-q_L fallback), the
  primary operating point W = $1, λ = 0 and the sensitivity rows.
* META_GOLD_SPEC §2.3 / §11.1, BENCHMARK_SPEC §8: `gold_aux.fatal_kind`, `meta_gold audit --implications`
  (violation count, must be 0), `fatal_kind_share.budget_share`, `stats_no_budget` / `ceiling --exclude-budget`.
* App. H / §4 economics: `CallRecord.attempts / retry_wait_ms / request_id / ts / tournament_round`, the report's
  `retry_rate` column.
