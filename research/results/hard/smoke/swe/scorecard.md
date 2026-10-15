## Score card (baseline: `baseline`)

- **Iso-quality cost savings**: -9.4% [-85.5, 23.3] with `action_guard` (Δsuccess +0.0 pp [0.0, 0.0], p_worse=1.000; not significantly worse, one-sided paired bootstrap, α=0.05; 3 arm(s) qualify)
- **Quality at iso-cost** (≤ 0.249 $/episode): `baseline` 50.0% vs. reference 50.0% (Δ +0.0 pp, p_better=–); hull 50.0%
- **Frontier cost-of-pass**: ∞ $/success over 4 arms (best single arm: `baseline`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.5, baseline alone 0.5; front: `baseline`

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | McNemar p | lat save % | LoP s | LLM-call ratio | strong-call ratio | PGR | EuB | HV contrib | LLM-free steps % | cache-mod $ | no-cache $ | cached % | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 50.0 [0.0, 100.0] | – | – | 0.249 | – | 0.498 | 129. | – | – | – | 258. | 1.00 | 1.00 | – | 0.349 | 0.059 | 0.0 | 0.249 | 0.517 | 52.7 | front,baseline |
| action_guard | 50.0 [0.0, 100.0] | +0.0 | 1.000 | 0.272 | -9.4 | 0.545 | 117. | 0.0 | 1.000 | 9.1 | 234. | 1.01 | 1.01 | – | 0.221 | 0.000 | 0.0 | 0.272 | 0.699 | 67.8 | iso-best |
| route | 50.0 [0.0, 100.0] | +0.0 | 1.000 | 0.286 | -14.9 | 0.572 | 113. | – | 1.000 | 12.2 | 226. | 1.22 | 0.67 | – | 0.199 | 0.000 | 0.0 | 0.286 | 0.679 | 72.4 | iso-q |
| stop_guard | 50.0 [0.0, 100.0] | +0.0 | 1.000 | 0.425 | -70.8 | 0.850 | 179. | – | 1.000 | -38.8 | 358. | 1.37 | 1.37 | – | 0.314 | 0.000 | 0.0 | 0.425 | 1.24 | 65.1 | iso-q |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

_cost $: the billed (ledger) cost, the primary figure; cache-mod $: System-2 calls re-priced from their logged tokens with the measured cached tokens at the model's cached rate; no-cache $: list price with no cache; cached %: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md)._

### Non-inferiority (A6 procedure; margin 3.0 pp)

| arm | reference | n | Δ pp | 95% LCB pp | p (rule) | Holm p | α_adj | Tango LB(1−α_adj) pp | discordant (w/l) | exact p (loss UB pp) | MDE Δ pp (80%) | p_boot | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| action_guard | baseline | 2 | +0.0 | +0.0 | 0.942 (bb_ordered) | 1.000 | 0.0167 | -69.4 | 0 (0/0) | 0.941 (87.09) | – | <0.001 | inconclusive at this n |
| route | baseline | 2 | +0.0 | +0.0 | 0.942 (bb_ordered) | 1.000 | 0.0250 | -65.8 | 0 (0/0) | 0.941 (84.19) | – | <0.001 | inconclusive at this n |
| stop_guard | baseline | 2 | +0.0 | +0.0 | 0.942 (bb_ordered) | 1.000 | 0.0500 | -57.5 | 0 (0/0) | 0.941 (77.64) | – | <0.001 | inconclusive at this n |

_A6.14: p_T = 1 − Φ(Z(−m)), Tango's one-sided score test of H0: Δ ≤ −m on the paired (wins, losses, n) counts, enters Holm with the family (every acting arm vs. the same reference); NI holds iff p_T ≤ α_adj of its Holm step, equivalently iff the Tango (1 − α_adj) lower bound is above −m. With zero wins or < 5 discordant tasks the exact binomial on losses (p = P[L ≤ l | n, m]) decides instead (`rule` = exact; it is the p that enters Holm, `Holm p`) and the one-sided 95% upper bound on the loss rate (1 − 0.05^(1/n) at zero losses) replaces a zero-width interval. A primary that fails while its two-sided 95% CI covers 0 is *inconclusive at this n*, not inferior; MDE: the smallest true Δ shown NI with 80% power at the achieved discordance and α_adj. p: the certified Berger-Boos p-value of one ordering of all (wins, losses) outcomes (A6.21; rule bb_ordered). Tango LB: descriptive score interval._

### Live gates vs. their gold (consistency check; not H2 inputs)

| arm | gate/mode | opportunities (retries) | labelled share | n+ / n− labelled | AUROC (observed labels) | J(τ) | realised fire rate |
|---|---|---|---|---|---|---|---|
| action_guard | tool_select/guard | 66 (0) | 100.0% | 0 / 0 | – | – | 0.061 |
| route | routing/trigger | 82 (0) | 100.0% | 0 / 0 | – | selective | 0.549 |

_Live trigger arms label a reflection / routing opportunity only where the gate did not fire (selective labels): AUROC is computed on the labels observed at executed actions, J / accuracy / estimability are not reported, and these rows never enter H2 (META_GOLD_SPEC §9.2, §11.5–6). The realised fire rate counts every opportunity once (a stop-guard resubmission of a rejected answer is a retry)._

