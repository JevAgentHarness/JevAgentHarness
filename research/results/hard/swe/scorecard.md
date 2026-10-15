## Score card (baseline: `baseline`)

- **Iso-quality cost savings**: 17.6% [-2.5, 34.7] with `route` (Δsuccess +2.0 pp [-5.0, 9.0], p_worse=0.758; not significantly worse, one-sided paired bootstrap, α=0.05; 3 arm(s) qualify)
- **Quality at iso-cost** (≤ 0.563 $/episode): `route` 77.0% vs. reference 75.0% (Δ +2.0 pp, p_better=0.348); hull 77.0%
- **Frontier cost-of-pass**: ∞ $/success over 4 arms (best single arm: `route`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.77, baseline alone 0.47; front: `route`

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | McNemar p | lat save % | LoP s | LLM-call ratio | strong-call ratio | PGR | EuB | HV contrib | LLM-free steps % | cache-mod $ | no-cache $ | cached % | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 75.0 [67.0, 83.0] | – | – | 0.563 | – | 0.750 | 145. | – | – | – | 193. | 1.00 | 1.00 | – | 0.536 | 0.000 | 0.0 | 0.563 | 2.01 | 63.0 | baseline |
| route | 77.0 [69.0, 85.0] | +2.0 | 0.758 | 0.464 | 17.6 | 0.602 | 129. | – | 0.791 | 11.3 | 167. | 0.89 | 0.75 | – | 0.544 | 0.270 | 0.0 | 0.464 | 1.60 | 62.1 | front,iso-best |
| action_guard | 76.0 [68.0, 84.0] | +1.0 | 0.652 | 0.554 | 1.6 | 0.729 | 154. | 0.0 | 1.000 | -6.1 | 202. | 0.97 | 0.97 | – | 0.508 | 0.000 | 0.0 | 0.554 | 1.95 | 63.4 | iso-q |
| stop_guard | 74.0 [66.0, 82.0] | -1.0 | 0.449 | 0.619 | -10.0 | 0.837 | 155. | – | 1.000 | -6.7 | 209. | 1.05 | 1.05 | – | 0.508 | 0.000 | 0.0 | 0.619 | 2.34 | 64.5 | iso-q |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

_cost $: the billed (ledger) cost, the primary figure; cache-mod $: System-2 calls re-priced from their logged tokens with the measured cached tokens at the model's cached rate; no-cache $: list price with no cache; cached %: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md)._

### Non-inferiority (A6 procedure; margin 3.0 pp)

| arm | reference | n | Δ pp | 95% LCB pp | p (rule) | Holm p | α_adj | Tango LB(1−α_adj) pp | discordant (w/l) | exact p (loss UB pp) | MDE Δ pp (80%) | p_boot | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| route | baseline | 100 | +2.0 | -4.0 | 0.101 (bb_ordered) | 0.304 | 0.0167 | -6.6 | 14 (8/6) | – | +8.0 | 0.104 | inconclusive at this n |
| action_guard | baseline | 100 | +1.0 | -5.0 | 0.149 (bb_ordered) | 0.304 | 0.0250 | -6.7 | 13 (7/6) | – | +7.0 | 0.155 | inconclusive at this n |
| stop_guard | baseline | 100 | -1.0 | -7.0 | 0.314 (bb_ordered) | 0.314 | 0.0500 | -7.4 | 13 (6/7) | – | +5.9 | 0.336 | inconclusive at this n |

_A6.14: p_T = 1 − Φ(Z(−m)), Tango's one-sided score test of H0: Δ ≤ −m on the paired (wins, losses, n) counts, enters Holm with the family (every acting arm vs. the same reference); NI holds iff p_T ≤ α_adj of its Holm step, equivalently iff the Tango (1 − α_adj) lower bound is above −m. With zero wins or < 5 discordant tasks the exact binomial on losses (p = P[L ≤ l | n, m]) decides instead (`rule` = exact; it is the p that enters Holm, `Holm p`) and the one-sided 95% upper bound on the loss rate (1 − 0.05^(1/n) at zero losses) replaces a zero-width interval. A primary that fails while its two-sided 95% CI covers 0 is *inconclusive at this n*, not inferior; MDE: the smallest true Δ shown NI with 80% power at the achieved discordance and α_adj. p: the certified Berger-Boos p-value of one ordering of all (wins, losses) outcomes (A6.21; rule bb_ordered). Tango LB: descriptive score interval._

### Live gates vs. their gold (consistency check; not H2 inputs)

| arm | gate/mode | opportunities (retries) | labelled share | n+ / n− labelled | AUROC (observed labels) | J(τ) | realised fire rate |
|---|---|---|---|---|---|---|---|
| route | routing/trigger | 5412 (0) | 100.0% | 0 / 0 | – | selective | 0.840 |
| action_guard | tool_select/guard | 5751 (0) | 100.0% | 0 / 0 | – | – | 0.012 |

_Live trigger arms label a reflection / routing opportunity only where the gate did not fire (selective labels): AUROC is computed on the labels observed at executed actions, J / accuracy / estimability are not reported, and these rows never enter H2 (META_GOLD_SPEC §9.2, §11.5–6). The realised fire rate counts every opportunity once (a stop-guard resubmission of a rejected answer is a retry)._

