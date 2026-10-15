## Score card (baseline: `llm_only`)

- **Iso-quality cost savings**: no arm qualifies (non-inferior at margin 5.0 pp, α=0.05)
- **Quality at iso-cost** (≤ 0.0749 $/episode): `llm_only` 64.0% vs. reference 64.0% (Δ +0.0 pp, p_better=–); hull 64.0%
- **Frontier cost-of-pass**: ∞ $/success over 3 arms (best single arm: `llm_only`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.633, baseline alone 0.626; front: `jev_ts_cascade_tauhat_e2`, `llm_only`
- note: several trials per task: the NI rows, Holm and the `noninferior` iso-quality qualifier use majority pass (> half of a task's trials succeed; PREREGISTRATION A6.14 / A6.16 (e)); Delta and its CI are per-attempt (pass^1) means

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | McNemar p | lat save % | LoP s | LLM-call ratio | strong-call ratio | PGR | EuB | HV contrib | LLM-free steps % | cache-mod $ | no-cache $ | cached % | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 62.6 [56.4, 68.7] | – | – | 0.0749 | – | 0.120 | 14.7 | 0.0 | – | – | 23.4 | 1.00 | 1.00 | – | 0.310 | 0.137 | 0.0 | 0.0749 | 0.127 | 48.0 | front,baseline |
| jev_ts_cascade_tauhat_e2 | 63.5 [56.7, 69.6] | +0.9 | 0.634 | 0.0816 | -9.0 | 0.129 | 18.5 | 23.5 | 0.849 | -26.6 | 29.2 | 0.99 | 0.99 | – | 0.291 | 0.007 | 1.0 | 0.0816 | 0.126 | 42.6 | front |
| jev_ts_cascade_tauhat_refit | 58.5 [52.0, 64.9] | -4.1 | 0.094 | 0.0882 | -17.7 | 0.151 | 18.8 | 65.6 | 0.239 | -28.5 | 32.2 | 0.92 | 0.92 | – | 0.238 | 0.000 | 7.0 | 0.0882 | 0.0981 | 9.5 |  |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

_cost $: the billed (ledger) cost, the primary figure; cache-mod $: System-2 calls re-priced from their logged tokens with the measured cached tokens at the model's cached rate; no-cache $: list price with no cache; cached %: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md)._

### Non-inferiority (A6 procedure; margin 5.0 pp)

| arm | reference | n | Δ pp | 95% LCB pp | p (rule) | Holm p | α_adj | Tango LB(1−α_adj) pp | discordant (w/l) | exact p (loss UB pp) | MDE Δ pp (80%) | p_boot | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_ts_cascade_tauhat_e2 | llm_only | 114 | -2.6 | -11.4 | 0.347 (bb_ordered) | 0.695 | 0.0250 | -13.5 | 39 (18/21) | – | +10.2 | 0.353 | inconclusive at this n |
| jev_ts_cascade_tauhat_refit | llm_only | 114 | -6.1 | -14.0 | 0.610 (bb_ordered) | 0.695 | 0.0500 | -14.5 | 33 (13/20) | – | +7.5 | 0.604 | inconclusive at this n |

_A6.14: p_T = 1 − Φ(Z(−m)), Tango's one-sided score test of H0: Δ ≤ −m on the paired (wins, losses, n) counts, enters Holm with the family (every acting arm vs. the same reference); NI holds iff p_T ≤ α_adj of its Holm step, equivalently iff the Tango (1 − α_adj) lower bound is above −m. With zero wins or < 5 discordant tasks the exact binomial on losses (p = P[L ≤ l | n, m]) decides instead (`rule` = exact; it is the p that enters Holm, `Holm p`) and the one-sided 95% upper bound on the loss rate (1 − 0.05^(1/n) at zero losses) replaces a zero-width interval. A primary that fails while its two-sided 95% CI covers 0 is *inconclusive at this n*, not inferior; MDE: the smallest true Δ shown NI with 80% power at the achieved discordance and α_adj. p: the certified Berger-Boos p-value of one ordering of all (wins, losses) outcomes (A6.21; rule bb_ordered). Tango LB: descriptive score interval._

### Reliability (pass^k)

| arm | pass^1 | pass^2 | pass^3 |
|---|---|---|---|
| llm_only | 0.626 | 0.447 | 0.351 |
| jev_ts_cascade_tauhat_e2 | 0.635 | 0.474 | 0.404 |
| jev_ts_cascade_tauhat_refit | 0.585 | 0.398 | 0.307 |

### System-1 decisions

| arm | n scored | S1 acc | ECE | ACE | Brier | AUROC | AURC | E-AURC | acc@80%cov | harmful % | guard FAR | guard FRR | rescue % | yes/no n | yes/no ECE | yes/no Brier | yes/no AUROC | mem stored % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_ts_cascade_tauhat_e2 | 4503 | 0.635 | 0.175 | 0.175 | 0.256 | 0.640 | 0.269 | 0.192 | 0.677 | 9.5 | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tauhat_refit | 5753 | 0.675 | 0.147 | 0.147 | 0.225 | 0.693 | 0.199 | 0.139 | 0.725 | 14.1 | – | – | – | 0 | – | – | – | – |

_Calibration over System-1 Choice decisions with gold labels (top-label confidence); yes/no: guard / trigger decisions calibrated as forecasts P(yes) of the binary gold; harmful: delegated decisions that were wrong in failed episodes (per delegated decision); FAR/FRR: guard false accept / false reject rates; rescue: wrong LLM proposals caught by the guard in episodes that succeeded._

### Delegation coverage and native LLM calls (A6.22 (a))

| arm | tool cov. % | arg cov. % (S1 / slots) | arg cov. in S1 calls % | LLM-free steps % | LLM calls / ep. | calls avoided / ep. | avoided % | reference |
|---|---|---|---|---|---|---|---|---|
| jev_ts_cascade_tauhat_e2 | 13.4 | 13.6 (600/4399) | 71.9 | 1.0 | 11.56 | +0.08 | 0.7 | llm_only |
| jev_ts_cascade_tauhat_refit | 80.2 | 33.3 (1478/4439) | 47.1 | 7.0 | 10.75 | +0.89 | 7.6 | llm_only |

_tool cov.: tool choices executed from System-1 per tool-selection opportunity; arg cov.: argument values System-1 chose per argument slot of the executed calls (a deferred tool choice leaves the whole call, and its slots, to the LLM); in S1 calls: the same share inside System-1-chosen calls (– when it chose none: the argument thresholds were never reached); calls avoided: paired per-task mean of the reference's minus the arm's System-2 calls (trajectory length included)._

### Tool arguments by fill kind (H3; §7.4)

| arm | kind | n | delegated | delegation | harmful |
|---|---|---|---|---|---|
| jev_ts_cascade_tauhat_e2 | unknown | 834 | 600 | 71.9% | 0.5% |
| jev_ts_cascade_tauhat_refit | unknown | 3139 | 1478 | 47.1% | 1.3% |

