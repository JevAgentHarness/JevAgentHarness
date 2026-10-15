## Score card (baseline: `llm_only`)

- **Iso-quality cost savings**: no arm qualifies (non-inferior at margin 5.0 pp, α=0.05)
- **Quality at iso-cost** (≤ 0.0805 $/episode): `jev_ts_cascade_tauhat_e2` 77.1% vs. reference 71.4% (Δ +5.7 pp, p_better=0.249); hull 77.1%
- **Frontier cost-of-pass**: ∞ $/success over 11 arms (best single arm: `jev_ts_cascade_tauhat_e2`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.749, baseline alone 0.606; front: `jev_guard`, `jev_ts_cascade_tauhat_e2`
- **Sweep `jev_ts_cascade_tau*`** (6 arms): AIQ 0.555
- note: several trials per task: the NI rows, Holm and the `noninferior` iso-quality qualifier use majority pass (> half of a task's trials succeed; PREREGISTRATION A6.14 / A6.16 (e)); Delta and its CI are per-attempt (pass^1) means

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | McNemar p | lat save % | LoP s | LLM-call ratio | strong-call ratio | PGR | EuB | HV contrib | LLM-free steps % | cache-mod $ | no-cache $ | cached % | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 68.6 [54.3, 81.9] | – | – | 0.0805 | – | 0.117 | 14.9 | 0.0 | – | – | 21.7 | 1.00 | 1.00 | – | 0.390 | 0.000 | 0.0 | 0.0805 | 0.143 | 48.6 | baseline |
| jev_guard | 75.2 [63.8, 85.7] | +6.7 | 0.906 | 0.0836 | -3.9 | 0.111 | 19.3 | 0.0 | 0.189 | -29.2 | 25.6 | 1.02 | 1.02 | – | 0.413 | 0.015 | 0.0 | 0.0836 | 0.150 | 48.4 | front |
| jev_ts_cascade_tauhat_e2 | 73.3 [61.9, 84.8] | +4.8 | 0.854 | 0.0759 | 5.7 | 0.103 | 16.9 | 8.8 | 0.383 | -13.6 | 23.1 | 0.91 | 0.91 | – | 0.430 | 0.089 | 1.3 | 0.0759 | 0.129 | 46.3 | front |
| jev_ts_cascade_tau0.95 | 68.6 [56.2, 80.0] | +0.0 | 0.494 | 0.0959 | -19.2 | 0.140 | 25.0 | 27.9 | 1.000 | -67.6 | 36.4 | 0.95 | 0.95 | – | 0.352 | 0.000 | 4.0 | 0.0959 | 0.148 | 39.5 |  |
| jev_ts_cascade_tau0.99 | 65.7 [51.4, 79.0] | -2.9 | 0.277 | 0.0849 | -5.5 | 0.129 | 22.0 | 15.6 | 0.689 | -47.5 | 33.5 | 0.98 | 0.98 | – | 0.367 | 0.000 | 2.4 | 0.0849 | 0.143 | 43.6 |  |
| jev_ts_cascade_tau0.9 | 65.7 [53.3, 78.1] | -2.9 | 0.263 | 0.0950 | -18.1 | 0.145 | 22.5 | 40.2 | 0.664 | -50.9 | 34.2 | 0.90 | 0.90 | – | 0.351 | 0.000 | 7.2 | 0.0950 | 0.140 | 35.3 |  |
| jev_ts_assist_k3 | 64.8 [50.5, 78.1] | -3.8 | 0.236 | 0.0878 | -9.2 | 0.136 | 31.6 | 0.0 | 0.523 | -112.2 | 48.8 | 1.02 | 1.02 | – | 0.363 | 0.000 | 0.0 | 0.0878 | 0.0987 | 6.9 |  |
| jev_ts_cascade_tauhat_refit | 61.0 [47.6, 74.3] | -7.6 | 0.107 | 0.0962 | -19.5 | 0.158 | 19.7 | 53.6 | 0.201 | -31.9 | 32.3 | 0.92 | 0.92 | – | 0.313 | 0.000 | 9.8 | 0.0962 | 0.112 | 9.6 |  |
| jev_ts_cascade_tau0.7 | 60.0 [46.7, 72.4] | -8.6 | 0.063 | 0.0852 | -5.8 | 0.142 | 21.9 | 63.1 | 0.078 | -46.6 | 36.4 | 0.81 | 0.81 | – | 0.334 | 0.000 | 17.0 | 0.0852 | 0.113 | 23.7 |  |
| jev_ts_cascade_tau0.8 | 60.0 [46.7, 72.4] | -8.6 | 0.044 | 0.0863 | -7.2 | 0.144 | 19.2 | 53.2 | 0.137 | -28.9 | 32.0 | 0.84 | 0.84 | – | 0.321 | 0.000 | 12.7 | 0.0863 | 0.121 | 28.5 |  |
| jev_ts_cascade_tau0.5 | 56.2 [41.9, 69.5] | -12.4 | 0.013 | 0.100 | -24.4 | 0.178 | 19.9 | 69.5 | 0.021 | -33.7 | 35.5 | 0.88 | 0.88 | – | 0.315 | 0.000 | 27.0 | 0.100 | 0.136 | 18.3 |  |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

_cost $: the billed (ledger) cost, the primary figure; cache-mod $: System-2 calls re-priced from their logged tokens with the measured cached tokens at the model's cached rate; no-cache $: list price with no cache; cached %: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md)._

### Non-inferiority (A6 procedure; margin 5.0 pp)

| arm | reference | n | Δ pp | 95% LCB pp | p (rule) | Holm p | α_adj | Tango LB(1−α_adj) pp | discordant (w/l) | exact p (loss UB pp) | MDE Δ pp (80%) | p_boot | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_guard | llm_only | 35 | +0.0 | -8.6 | 0.374 (bb_ordered) | 1.000 | 0.0056 | -19.8 | 4 (2/2) | 0.746 (23.60) | +13.7 | 0.214 | inconclusive at this n |
| jev_ts_cascade_tauhat_e2 | llm_only | 35 | +5.7 | -5.7 | 0.075 (bb_ordered) | 0.754 | 0.0050 | -15.9 | 6 (4/2) | – | +18.2 | 0.063 | inconclusive at this n |
| jev_ts_cascade_tau0.95 | llm_only | 35 | -5.7 | -14.3 | 0.736 (bb_ordered) | 1.000 | 0.0125 | -22.8 | 4 (1/3) | 0.904 (25.26) | +12.2 | 0.584 | inconclusive at this n |
| jev_ts_cascade_tau0.99 | llm_only | 35 | -2.9 | -14.3 | 0.378 (bb_ordered) | 1.000 | 0.0063 | -23.3 | 5 (2/3) | – | +15.7 | 0.406 | inconclusive at this n |
| jev_ts_cascade_tau0.9 | llm_only | 35 | -2.9 | -11.4 | 0.736 (bb_ordered) | 1.000 | 0.0100 | -19.8 | 3 (1/2) | 0.746 (21.92) | +10.2 | 0.374 | inconclusive at this n |
| jev_ts_assist_k3 | llm_only | 35 | -8.6 | -20.0 | 0.701 (bb_ordered) | 1.000 | 0.0083 | -29.4 | 7 (2/5) | – | +18.8 | 0.721 | inconclusive at this n |
| jev_ts_cascade_tauhat_refit | llm_only | 35 | -14.3 | -28.6 | 0.875 (bb_ordered) | 1.000 | 0.0250 | -31.8 | 9 (2/7) | – | +18.4 | 0.888 | inconclusive at this n |
| jev_ts_cascade_tau0.7 | llm_only | 35 | -14.3 | -28.6 | 0.875 (bb_ordered) | 1.000 | 0.0167 | -33.4 | 9 (2/7) | – | +19.8 | 0.896 | inconclusive at this n |
| jev_ts_cascade_tau0.8 | llm_only | 35 | -8.6 | -22.9 | 0.687 (bb_ordered) | 1.000 | 0.0071 | -31.3 | 9 (3/6) | – | +22.4 | 0.687 | inconclusive at this n |
| jev_ts_cascade_tau0.5 | llm_only | 35 | -17.1 | -28.6 | 0.950 (bb_ordered) | 1.000 | 0.0500 | -31.0 | 8 (1/7) | – | +14.7 | 0.963 | not NI |

_A6.14: p_T = 1 − Φ(Z(−m)), Tango's one-sided score test of H0: Δ ≤ −m on the paired (wins, losses, n) counts, enters Holm with the family (every acting arm vs. the same reference); NI holds iff p_T ≤ α_adj of its Holm step, equivalently iff the Tango (1 − α_adj) lower bound is above −m. With zero wins or < 5 discordant tasks the exact binomial on losses (p = P[L ≤ l | n, m]) decides instead (`rule` = exact; it is the p that enters Holm, `Holm p`) and the one-sided 95% upper bound on the loss rate (1 − 0.05^(1/n) at zero losses) replaces a zero-width interval. A primary that fails while its two-sided 95% CI covers 0 is *inconclusive at this n*, not inferior; MDE: the smallest true Δ shown NI with 80% power at the achieved discordance and α_adj. p: the certified Berger-Boos p-value of one ordering of all (wins, losses) outcomes (A6.21; rule bb_ordered). Tango LB: descriptive score interval._

### Reliability (pass^k)

| arm | pass^1 | pass^2 | pass^3 |
|---|---|---|---|
| llm_only | 0.686 | 0.600 | 0.543 |
| jev_guard | 0.752 | 0.638 | 0.600 |
| jev_ts_cascade_tauhat_e2 | 0.733 | 0.619 | 0.543 |
| jev_ts_cascade_tau0.95 | 0.686 | 0.562 | 0.514 |
| jev_ts_cascade_tau0.99 | 0.657 | 0.552 | 0.486 |
| jev_ts_cascade_tau0.9 | 0.657 | 0.533 | 0.457 |
| jev_ts_assist_k3 | 0.648 | 0.571 | 0.543 |
| jev_ts_cascade_tauhat_refit | 0.610 | 0.495 | 0.457 |
| jev_ts_cascade_tau0.7 | 0.600 | 0.495 | 0.457 |
| jev_ts_cascade_tau0.8 | 0.600 | 0.457 | 0.371 |
| jev_ts_cascade_tau0.5 | 0.562 | 0.448 | 0.400 |

### System-1 decisions

| arm | n scored | S1 acc | ECE | ACE | Brier | AUROC | AURC | E-AURC | acc@80%cov | harmful % | guard FAR | guard FRR | rescue % | yes/no n | yes/no ECE | yes/no Brier | yes/no AUROC | mem stored % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_guard | 0 | – | – | – | – | – | – | – | – | – | 0.667 | 0.111 | 30.0 | 0 | – | – | – | – |
| jev_ts_cascade_tauhat_e2 | 1269 | 0.641 | 0.160 | 0.153 | 0.245 | 0.645 | 0.245 | 0.171 | 0.680 | 1.9 | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tau0.95 | 1452 | 0.585 | 0.209 | 0.208 | 0.266 | 0.689 | 0.258 | 0.156 | 0.633 | 7.7 | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tau0.99 | 1411 | 0.605 | 0.182 | 0.182 | 0.259 | 0.662 | 0.266 | 0.175 | 0.649 | 4.3 | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tau0.9 | 1504 | 0.652 | 0.142 | 0.141 | 0.228 | 0.693 | 0.209 | 0.140 | 0.705 | 8.6 | – | – | – | 0 | – | – | – | – |
| jev_ts_assist_k3 | 1344 | 0.624 | 0.168 | 0.167 | 0.260 | 0.609 | 0.305 | 0.223 | 0.660 | – | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tauhat_refit | 1672 | 0.664 | 0.125 | 0.125 | 0.224 | 0.671 | 0.221 | 0.157 | 0.709 | 17.5 | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tau0.7 | 1641 | 0.673 | 0.131 | 0.131 | 0.223 | 0.676 | 0.201 | 0.140 | 0.710 | 14.3 | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tau0.8 | 1530 | 0.628 | 0.171 | 0.171 | 0.240 | 0.695 | 0.226 | 0.146 | 0.684 | 13.8 | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tau0.5 | 1878 | 0.572 | 0.191 | 0.192 | 0.270 | 0.639 | 0.300 | 0.192 | 0.600 | 24.3 | – | – | – | 0 | – | – | – | – |

_Calibration over System-1 Choice decisions with gold labels (top-label confidence); yes/no: guard / trigger decisions calibrated as forecasts P(yes) of the binary gold; harmful: delegated decisions that were wrong in failed episodes (per delegated decision); FAR/FRR: guard false accept / false reject rates; rescue: wrong LLM proposals caught by the guard in episodes that succeeded._

### Delegation coverage and native LLM calls (A6.22 (a))

| arm | tool cov. % | arg cov. % (S1 / slots) | arg cov. in S1 calls % | LLM-free steps % | LLM calls / ep. | calls avoided / ep. | avoided % | reference |
|---|---|---|---|---|---|---|---|---|
| jev_ts_cascade_tauhat_e2 | 4.5 | 4.0 (58/1467) | 96.7 | 1.3 | 11.45 | +1.12 | 8.9 | llm_only |
| jev_ts_cascade_tau0.95 | 18.3 | 10.5 (181/1716) | 91.0 | 4.0 | 11.98 | +0.59 | 4.7 | llm_only |
| jev_ts_cascade_tau0.99 | 9.6 | 5.8 (96/1643) | 97.0 | 2.4 | 12.33 | +0.24 | 1.9 | llm_only |
| jev_ts_cascade_tau0.9 | 29.3 | 15.8 (266/1680) | 84.4 | 7.2 | 11.33 | +1.24 | 9.8 | llm_only |
| jev_ts_cascade_tauhat_refit | 65.7 | 14.5 (245/1685) | 32.2 | 9.8 | 11.60 | +0.97 | 7.7 | llm_only |
| jev_ts_cascade_tau0.7 | 54.8 | 30.9 (494/1598) | 80.6 | 17.0 | 10.22 | +2.35 | 18.7 | llm_only |
| jev_ts_cascade_tau0.8 | 43.8 | 22.7 (372/1637) | 78.3 | 12.7 | 10.53 | +2.04 | 16.2 | llm_only |
| jev_ts_cascade_tau0.5 | 57.7 | 34.8 (774/2225) | 91.9 | 27.0 | 11.11 | +1.46 | 11.6 | llm_only |

_tool cov.: tool choices executed from System-1 per tool-selection opportunity; arg cov.: argument values System-1 chose per argument slot of the executed calls (a deferred tool choice leaves the whole call, and its slots, to the LLM); in S1 calls: the same share inside System-1-chosen calls (– when it chose none: the argument thresholds were never reached); calls avoided: paired per-task mean of the reference's minus the arm's System-2 calls (trajectory length included)._

### Tool arguments by fill kind (H3; §7.4)

| arm | kind | n | delegated | delegation | harmful |
|---|---|---|---|---|---|
| jev_ts_cascade_tauhat_e2 | unknown | 60 | 58 | 96.7% | 0.0% |
| jev_ts_cascade_tau0.95 | unknown | 199 | 181 | 91.0% | 0.0% |
| jev_ts_cascade_tau0.99 | unknown | 99 | 96 | 97.0% | 0.0% |
| jev_ts_cascade_tau0.9 | unknown | 315 | 266 | 84.4% | 0.5% |
| jev_ts_cascade_tauhat_refit | unknown | 760 | 245 | 32.2% | 0.8% |
| jev_ts_cascade_tau0.7 | unknown | 613 | 494 | 80.6% | 3.8% |
| jev_ts_cascade_tau0.8 | unknown | 475 | 372 | 78.3% | 3.3% |
| jev_ts_cascade_tau0.5 | unknown | 842 | 774 | 91.9% | 15.3% |

### Sweeps

| sweep | arms | values | APGR | CPT(50%) | CPT(80%) | AIQ | AIQ zero-router | ΔAIQ |
|---|---|---|---|---|---|---|---|---|
| jev_ts_cascade_tau | 6 | 0.5, 0.7, 0.8, 0.9, 0.95, 0.99 | – | – | – | 0.555 | – | – |

_APGR / CPT: x = strong-model calls relative to the baseline, anchors weak (PGR 0) and baseline (PGR 1); AIQ: mean success of the non-decreasing convex hull over the linear cost range between the two references (sweep + references); zero-router: hull of the references alone._

