## Score card (baseline: `llm_only`)

- **Iso-quality cost savings**: 65.7% [58.9, 76.5] with `jev_ts_replace` (Δsuccess -10.0 pp [-30.0, 0.0], p_worse=0.349; not significantly worse, one-sided paired bootstrap, α=0.05; 10 arm(s) qualify)
- **Quality at iso-cost** (≤ 0.0422 $/episode): `intern_ts_assist_k5` 100.0% vs. reference 100.0% (Δ +0.0 pp, p_better=1.000); hull 100.0%
- **Iso-quality cost savings** (vs. `llm_only_s2lite`): 58.3% [52.1, 65.2] with `intern_ts_assist_k3_s2lite` (Δsuccess -10.0 pp [-30.0, 0.0], p_worse=0.352; not significantly worse, one-sided paired bootstrap, α=0.05; 3 arm(s) qualify)
- **Quality at iso-cost** (vs. `llm_only_s2lite`) (≤ 0.00963 $/episode): `llm_only_s2lite` 80.0% vs. reference 80.0% (Δ +0.0 pp, p_better=–); hull 80.0%
- **Frontier cost-of-pass**: 0.00855 $/success over 25 arms (3 shadow arm(s) left out of the frontier) (best single arm: `intern_ts_assist_k3_s2lite`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.862, baseline alone 0.181; front: `intern_ts_assist_k3_s2lite`, `intern_ts_assist_k5`, `jev_ts_replace`, `llm_only_s2lite`
- **Sweep `intern_ts_cascade_tau*`** (6 arms): AIQ 0.704
- note: per-System-2 references: `llm_only_s2lite` (gemini-3.5-flash-lite) for 4 arm(s); every other arm vs. `llm_only`

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | McNemar p | lat save % | LoP s | LLM-call ratio | strong-call ratio | PGR | EuB | HV contrib | LLM-free steps % | cache-mod $ | no-cache $ | cached % | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 100.0 [100.0, 100.0] | – | – | 0.0422 | – | 0.0422 | 6.76 | 0.0 | – | – | 6.76 | 1.00 | 1.00 | – | 0.500 | 0.000 | 0.0 | 0.0422 | 0.0704 | 44.5 | baseline |
| intern_ts_assist_k5 | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0242 | 42.6 | 0.0242 | 19.4 | 0.0 | 1.000 | -186.5 | 19.4 | 1.25 | 1.25 | – | 0.713 | 0.004 | 0.0 | 0.0242 | 0.0242 | 0.0 | front,iso-q |
| intern_ts_assist_k3 | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0271 | 35.7 | 0.0271 | 22.7 | 0.0 | 1.000 | -236.5 | 22.7 | 1.39 | 1.39 | – | 0.678 | 0.000 | 0.0 | 0.0271 | 0.0271 | 0.0 | iso-q |
| intern_ts_cascade_tau0.99 | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0392 | 7.1 | 0.0392 | 15.9 | 3.9 | 1.000 | -135.8 | 15.9 | 0.99 | 0.99 | – | 0.535 | 0.000 | 1.3 | 0.0392 | 0.0698 | 49.3 | iso-q |
| intern_shadow | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0396 | 6.2 | 0.0396 | 9.64 | 0.0 | 1.000 | -42.6 | 9.64 | 1.00 | 1.00 | – | 0.531 | – | 0.0 | 0.0396 | 0.0704 | 49.1 |  |
| jev_shadow | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0396 | 6.2 | 0.0396 | 7.15 | 0.0 | 1.000 | -5.8 | 7.15 | 1.00 | 1.00 | – | 0.531 | – | 0.0 | 0.0396 | 0.0704 | 49.1 |  |
| jev_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0406 | 3.8 | 0.0406 | 8.01 | 0.0 | 1.000 | -18.5 | 8.01 | 1.01 | 1.01 | – | 0.519 | 0.000 | 0.0 | 0.0406 | 0.0717 | 49.0 | iso-q |
| intern_stop_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0420 | 0.5 | 0.0420 | 9.20 | 0.0 | 1.000 | -36.1 | 9.20 | 1.04 | 1.04 | – | 0.502 | 0.000 | 0.0 | 0.0420 | 0.0741 | 48.9 | iso-q |
| intern_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0420 | 0.4 | 0.0420 | 9.88 | 0.0 | 1.000 | -46.2 | 9.88 | 1.07 | 1.07 | – | 0.502 | 0.000 | 0.0 | 0.0420 | 0.0752 | 50.2 | iso-q |
| intern4b_uncal_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0423 | -0.3 | 0.0423 | 10.6 | 0.0 | 1.000 | -56.6 | 10.6 | 1.07 | 1.07 | – | 0.498 | 0.000 | 0.0 | 0.0423 | 0.0753 | 49.8 | iso-q |
| jev_ts_replace | 90.0 [70.0, 100.0] | -10.0 | 0.349 | 0.0145 | 65.7 | 0.0161 | 6.86 | 100.0 | 1.000 | -1.4 | 7.62 | 0.95 | 0.95 | – | 0.731 | 0.018 | 0.0 | 0.0145 | 0.0145 | 0.0 | front,iso-best |
| intern_ts_cascade_tau0.95 | 90.0 [70.0, 100.0] | -10.0 | 0.349 | 0.0339 | 19.7 | 0.0376 | 16.8 | 27.5 | 1.000 | -148.1 | 18.6 | 0.95 | 0.95 | – | 0.513 | 0.000 | 2.7 | 0.0339 | 0.0577 | 45.1 | iso-q |
| llm_only_s2lite | 80.0 [50.0, 100.0] | -20.0 | 0.101 | 0.00963 | 77.2 | 0.0120 | 4.99 | 0.0 | 0.500 | 26.2 | 6.24 | 1.01 | 1.01 | – | 0.715 | 0.014 | 0.0 | 0.00963 | 0.0143 | 35.5 | front |
| jev_ts_assist_k3 | 80.0 [50.0, 100.0] | -20.0 | 0.107 | 0.0216 | 48.8 | 0.0270 | 9.50 | 0.0 | 0.500 | -40.5 | 11.9 | 1.17 | 1.17 | – | 0.591 | 0.000 | 0.0 | 0.0216 | 0.0216 | 0.0 | iso-q |
| intern_ts_assist_k3_s2lite | 70.0 [40.0, 100.0] | -10.0 | 0.352 | 0.00401 | 58.3 | 0.00574 | 13.6 | 0.0 | 1.000 | -172.7 | 19.4 | 1.08 | 1.08 | – | 0.670 | 0.188 | 0.0 | 0.00401 | 0.00401 | 0.0 | front,iso-best,vs llm_only_s2lite |
| intern_shadow_s2lite | 70.0 [40.0, 90.0] | -10.0 | 0.344 | 0.00973 | -1.1 | 0.0139 | 4.74 | 0.0 | 1.000 | 4.9 | 6.78 | 1.00 | 1.00 | – | 0.629 | – | 0.0 | 0.00973 | 0.0143 | 34.1 | vs llm_only_s2lite |
| intern_ts_guard_s2lite | 70.0 [40.0, 100.0] | -10.0 | 0.344 | 0.00975 | -1.3 | 0.0139 | 7.40 | 0.0 | 1.000 | -48.2 | 10.6 | 1.00 | 1.00 | – | 0.628 | 0.000 | 0.0 | 0.00975 | 0.0144 | 34.0 | iso-q,vs llm_only_s2lite |
| intern_stop_guard_s2lite | 70.0 [40.0, 100.0] | -10.0 | 0.344 | 0.0130 | -34.9 | 0.0186 | 7.70 | 0.0 | 1.000 | -54.3 | 11.0 | 1.33 | 1.33 | – | 0.630 | 0.000 | 0.0 | 0.0130 | 0.0204 | 37.0 | iso-q,vs llm_only_s2lite |
| intern_ts_cascade_tau0.9 | 70.0 [40.0, 100.0] | -30.0 | 0.032 | 0.0290 | 31.4 | 0.0414 | 16.8 | 35.8 | 0.250 | -148.4 | 24.0 | 0.91 | 0.91 | – | 0.395 | 0.000 | 2.9 | 0.0290 | 0.0452 | 40.4 |  |
| intern08_ts_assist_k3 | 30.0 [0.0, 60.0] | -70.0 | <0.001 | 0.0355 | 15.8 | 0.118 | 16.2 | 0.0 | 0.016 | -140.1 | 54.1 | 1.69 | 1.69 | – | 0.218 | 0.000 | 0.0 | 0.0355 | 0.0416 | 3.3 | worse |
| intern_ts_cascade_tau0.7 | 20.0 [0.0, 50.0] | -80.0 | <0.001 | 0.00971 | 77.0 | 0.0486 | 8.11 | 48.1 | 0.008 | -20.0 | 40.5 | 0.47 | 0.47 | – | 0.152 | 0.000 | 2.8 | 0.00971 | 0.0131 | 29.4 | worse |
| intern_ts_cascade_tau0.8 | 20.0 [0.0, 50.0] | -80.0 | <0.001 | 0.0143 | 66.2 | 0.0714 | 9.85 | 42.4 | 0.008 | -45.8 | 49.3 | 0.55 | 0.55 | – | 0.135 | 0.000 | 2.4 | 0.0143 | 0.0205 | 30.9 | worse |
| intern_ts_cascade_tau0.5 | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.00466 | 88.9 | 0.0466 | 6.81 | 53.8 | 0.004 | -0.8 | 68.1 | 0.41 | 0.41 | – | 0.093 | 0.000 | 3.1 | 0.00466 | 0.00571 | 6.9 | worse |
| intern_ts_replace | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0109 | 74.2 | 0.109 | 9.86 | 100.0 | 0.004 | -45.9 | 98.6 | 0.69 | 0.69 | – | 0.088 | 0.000 | 0.0 | 0.0109 | 0.0109 | 0.0 | worse |
| intern_ts_params_replace | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0134 | 68.3 | 0.134 | 13.7 | 76.3 | 0.004 | -102.5 | 137. | 0.63 | 0.63 | – | 0.097 | 0.000 | 34.7 | 0.0134 | 0.0134 | 0.0 | worse |
| intern2b_ts_assist_k3 | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0165 | 61.0 | 0.165 | 12.6 | 0.0 | 0.004 | -86.3 | 126. | 1.27 | 1.27 | – | 0.094 | 0.000 | 0.0 | 0.0165 | 0.0165 | 0.0 | worse |
| intern2b_ts_replace | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0304 | 28.0 | 0.304 | 14.0 | 100.0 | 0.004 | -106.8 | 140. | 1.12 | 1.12 | – | 0.097 | 0.000 | 0.0 | 0.0304 | 0.0385 | 5.2 | worse |
| intern08_ts_replace | 0.0 [0.0, 0.0] | -100.0 | <0.001 | 0.0564 | -33.6 | ∞ | 26.3 | 100.0 | 0.002 | -289.5 | ∞ | 2.49 | 2.49 | – | 0.000 | 0.000 | 0.0 | 0.0564 | 0.0686 | 6.8 | worse |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

_cost $: the billed (ledger) cost, the primary figure; cache-mod $: System-2 calls re-priced from their logged tokens with the measured cached tokens at the model's cached rate; no-cache $: list price with no cache; cached %: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md)._

### Non-inferiority (A6 procedure; margin 3.0 pp)

| arm | reference | n | Δ pp | 95% LCB pp | p_U (rule) | Holm p | α_adj | Tango LB(1−α_adj) pp | discordant (w/l) | exact p (loss UB pp) | MDE Δ pp (80%) | p_boot | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intern_ts_assist_k5 | llm_only | 10 | +0.0 | +0.0 | 0.737 (exact) | 1.000 | 0.0028 | -43.5 | 0 (0/0) | 0.737 (44.49) | – | <0.001 | inconclusive at this n |
| intern_ts_assist_k3 | llm_only | 10 | +0.0 | +0.0 | 0.737 (exact) | 1.000 | 0.0026 | -43.8 | 0 (0/0) | 0.737 (44.79) | – | <0.001 | inconclusive at this n |
| intern_ts_cascade_tau0.99 | llm_only | 10 | +0.0 | +0.0 | 0.737 (exact) | 1.000 | 0.0029 | -43.1 | 0 (0/0) | 0.737 (44.17) | – | <0.001 | inconclusive at this n |
| jev_ts_guard | llm_only | 10 | +0.0 | +0.0 | 0.737 (exact) | 1.000 | 0.0033 | -42.4 | 0 (0/0) | 0.737 (43.47) | – | <0.001 | inconclusive at this n |
| intern_stop_guard | llm_only | 10 | +0.0 | +0.0 | 0.737 (exact) | 1.000 | 0.0025 | -44.1 | 0 (0/0) | 0.737 (45.07) | – | <0.001 | inconclusive at this n |
| intern_ts_guard | llm_only | 10 | +0.0 | +0.0 | 0.737 (exact) | 1.000 | 0.0031 | -42.8 | 0 (0/0) | 0.737 (43.83) | – | <0.001 | inconclusive at this n |
| intern4b_uncal_ts_guard | llm_only | 10 | +0.0 | +0.0 | 0.737 (exact) | 1.000 | 0.0024 | -44.3 | 0 (0/0) | 0.737 (45.34) | – | <0.001 | inconclusive at this n |
| jev_ts_replace | llm_only | 10 | -10.0 | -30.0 | 0.965 (exact) | 1.000 | 0.0038 | -52.1 | 1 (0/1) | 0.965 (55.84) | +27.6 | 0.650 | inconclusive at this n |
| intern_ts_cascade_tau0.95 | llm_only | 10 | -10.0 | -30.0 | 0.965 (exact) | 1.000 | 0.0036 | -52.5 | 1 (0/1) | 0.965 (56.23) | +27.8 | 0.650 | inconclusive at this n |
| llm_only_s2lite | llm_only | 10 | -20.0 | -40.0 | 0.997 (exact) | 1.000 | 0.0045 | -60.4 | 2 (0/2) | 0.997 (65.29) | +39.4 | 0.899 | inconclusive at this n |
| jev_ts_assist_k3 | llm_only | 10 | -20.0 | -40.0 | 0.997 (exact) | 1.000 | 0.0042 | -60.7 | 2 (0/2) | 0.997 (65.71) | +39.7 | 0.892 | inconclusive at this n |
| intern_ts_assist_k3_s2lite | llm_only_s2lite | 10 | -10.0 | -30.0 | 0.965 (exact) | 1.000 | 0.0250 | -40.4 | 1 (0/1) | 0.965 (44.50) | +22.4 | 0.647 | inconclusive at this n |
| intern_ts_guard_s2lite | llm_only_s2lite | 10 | -10.0 | -30.0 | 0.965 (exact) | 1.000 | 0.0500 | -34.8 | 1 (0/1) | 0.965 (39.42) | +19.9 | 0.656 | inconclusive at this n |
| intern_stop_guard_s2lite | llm_only_s2lite | 10 | -10.0 | -30.0 | 0.965 (exact) | 1.000 | 0.0167 | -43.3 | 1 (0/1) | 0.965 (47.23) | +23.7 | 0.656 | inconclusive at this n |
| intern_ts_cascade_tau0.9 | llm_only | 10 | -30.0 | -50.0 | 1.000 (exact) | 1.000 | 0.0050 | -68.0 | 3 (0/3) | 1.000 (73.51) | +48.4 | 0.968 | inconclusive at this n |
| intern08_ts_assist_k3 | llm_only | 10 | -70.0 | -90.0 | 1.000 (exact) | 1.000 | 0.0056 | -91.9 | 7 (0/7) | 1.000 (96.16) | +74.3 | 1.000 | not NI |
| intern_ts_cascade_tau0.7 | llm_only | 10 | -80.0 | -100.0 | 1.000 (exact) | 1.000 | 0.0063 | -95.8 | 8 (0/8) | 1.000 (98.78) | +78.8 | 1.000 | not NI |
| intern_ts_cascade_tau0.8 | llm_only | 10 | -80.0 | -100.0 | 1.000 (exact) | 1.000 | 0.0071 | -95.7 | 8 (0/8) | 1.000 (98.70) | +77.9 | 1.000 | not NI |
| intern_ts_cascade_tau0.5 | llm_only | 10 | -90.0 | -100.0 | 1.000 (exact) | 1.000 | 0.0125 | -98.5 | 9 (0/9) | 1.000 (99.87) | +78.4 | 1.000 | not NI |
| intern_ts_replace | llm_only | 10 | -90.0 | -100.0 | 1.000 (exact) | 1.000 | 0.0250 | -98.2 | 9 (0/9) | 1.000 (99.75) | +72.2 | 1.000 | not NI |
| intern_ts_params_replace | llm_only | 10 | -90.0 | -100.0 | 1.000 (exact) | 1.000 | 0.0167 | -98.4 | 9 (0/9) | 1.000 (99.83) | +75.9 | 1.000 | not NI |
| intern2b_ts_assist_k3 | llm_only | 10 | -90.0 | -100.0 | 1.000 (exact) | 1.000 | 0.0083 | -98.7 | 9 (0/9) | 1.000 (99.92) | +81.6 | 1.000 | not NI |
| intern2b_ts_replace | llm_only | 10 | -90.0 | -100.0 | 1.000 (exact) | 1.000 | 0.0100 | -98.6 | 9 (0/9) | 1.000 (99.90) | +80.2 | 1.000 | not NI |
| intern08_ts_replace | llm_only | 10 | -100.0 | -100.0 | 1.000 (exact) | 1.000 | 0.0500 | -100.0 | 10 (0/10) | 1.000 (100.00) | +68.4 | 1.000 | not NI |

_A6.14: p_T = 1 − Φ(Z(−m)), Tango's one-sided score test of H0: Δ ≤ −m on the paired (wins, losses, n) counts, enters Holm with the family (every acting arm vs. the same reference); NI holds iff p_T ≤ α_adj of its Holm step, equivalently iff the Tango (1 − α_adj) lower bound is above −m. With zero wins or < 5 discordant tasks the exact binomial on losses (p = P[L ≤ l | n, m]) decides instead (`rule` = exact; it is the p that enters Holm, `Holm p`) and the one-sided 95% upper bound on the loss rate (1 − 0.05^(1/n) at zero losses) replaces a zero-width interval. A primary that fails while its two-sided 95% CI covers 0 is *inconclusive at this n*, not inferior; MDE: the smallest true Δ shown NI with 80% power at the achieved discordance and α_adj. p_U: the Berger-Boos exact unconditional p of Tango's score statistic (A6.17; rule tango_uncond) or the exact loss-count p (rule exact). Tango LB: descriptive score interval._

### System-1 decisions

| arm | n scored | S1 acc | ECE | ACE | Brier | AUROC | AURC | E-AURC | acc@80%cov | harmful % | guard FAR | guard FRR | rescue % | yes/no n | yes/no ECE | yes/no Brier | yes/no AUROC | mem stored % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intern_ts_assist_k5 | 94 | 0.755 | 0.098 | 0.113 | 0.166 | 0.718 | 0.127 | 0.093 | 0.816 | – | – | – | – | 0 | – | – | – | – |
| intern_ts_assist_k3 | 104 | 0.760 | 0.078 | 0.090 | 0.178 | 0.662 | 0.149 | 0.116 | 0.786 | – | – | – | – | 0 | – | – | – | – |
| intern_ts_cascade_tau0.99 | 77 | 0.753 | 0.070 | 0.085 | 0.162 | 0.750 | 0.118 | 0.083 | 0.823 | 0.0 | – | – | – | 0 | – | – | – | – |
| intern_shadow | 169 | 0.822 | 0.069 | 0.080 | 0.133 | 0.757 | 0.078 | 0.061 | 0.875 | – | – | – | – | 75 | 0.196 | 0.072 | 0.979 | – |
| jev_shadow | 169 | 0.953 | 0.028 | 0.021 | 0.041 | 0.885 | 0.007 | 0.006 | 0.985 | – | – | – | – | 75 | 0.045 | 0.007 | 1.000 | – |
| jev_ts_guard | 0 | – | – | – | – | – | – | – | – | – | – | 0.013 | – | 75 | 0.120 | 0.026 | – | – |
| intern_stop_guard | 0 | – | – | – | – | – | – | – | – | – | – | 0.273 | – | 13 | 0.312 | 0.144 | – | – |
| intern_ts_guard | 0 | – | – | – | – | – | – | – | – | – | 0.000 | 0.040 | 100.0 | 76 | 0.142 | 0.045 | 0.960 | – |
| intern4b_uncal_ts_guard | 0 | – | – | – | – | – | – | – | – | – | 0.000 | 0.040 | 100.0 | 76 | 0.070 | 0.039 | 0.960 | – |
| jev_ts_replace | 71 | 0.944 | 0.072 | 0.046 | 0.040 | 0.948 | 0.005 | 0.003 | 1.000 | 1.4 | – | – | – | 0 | – | – | – | – |
| intern_ts_cascade_tau0.95 | 96 | 0.823 | 0.054 | 0.073 | 0.137 | 0.701 | 0.090 | 0.073 | 0.857 | 3.8 | – | – | – | 0 | – | – | – | – |
| jev_ts_assist_k3 | 88 | 0.830 | 0.073 | 0.069 | 0.119 | 0.818 | 0.053 | 0.036 | 0.887 | – | – | – | – | 0 | – | – | – | – |
| intern_ts_assist_k3_s2lite | 82 | 0.720 | 0.049 | 0.091 | 0.153 | 0.808 | 0.118 | 0.073 | 0.803 | – | – | – | – | 0 | – | – | – | – |
| intern_shadow_s2lite | 176 | 0.835 | 0.092 | 0.112 | 0.143 | 0.702 | 0.087 | 0.072 | 0.865 | – | – | – | – | 76 | 0.230 | 0.072 | 1.000 | – |
| intern_ts_guard_s2lite | 0 | – | – | – | – | – | – | – | – | – | 0.667 | 0.014 | 33.3 | 74 | 0.128 | 0.035 | 0.986 | – |
| intern_stop_guard_s2lite | 0 | – | – | – | – | – | – | – | – | – | 0.250 | 0.125 | 0.0 | 32 | 0.192 | 0.122 | 0.924 | – |
| intern_ts_cascade_tau0.9 | 108 | 0.806 | 0.110 | 0.129 | 0.164 | 0.644 | 0.115 | 0.093 | 0.828 | 7.3 | – | – | – | 0 | – | – | – | – |
| intern08_ts_assist_k3 | 109 | 0.092 | 0.308 | 0.305 | 0.172 | 0.715 | 0.846 | 0.153 | 0.114 | – | – | – | – | 0 | – | – | – | – |
| intern_ts_cascade_tau0.7 | 55 | 0.709 | 0.130 | 0.093 | 0.197 | 0.639 | 0.212 | 0.163 | 0.773 | 27.6 | – | – | – | 0 | – | – | – | – |
| intern_ts_cascade_tau0.8 | 60 | 0.683 | 0.191 | 0.205 | 0.252 | 0.542 | 0.265 | 0.206 | 0.667 | 33.3 | – | – | – | 0 | – | – | – | – |
| intern_ts_cascade_tau0.5 | 48 | 0.646 | 0.197 | 0.176 | 0.225 | 0.636 | 0.289 | 0.214 | 0.718 | 41.2 | – | – | – | 0 | – | – | – | – |
| intern_ts_replace | 52 | 0.423 | 0.264 | 0.303 | 0.283 | 0.712 | 0.417 | 0.199 | 0.524 | 53.8 | – | – | – | 0 | – | – | – | – |
| intern_ts_params_replace | 122 | 0.549 | 0.291 | 0.298 | 0.332 | 0.396 | 0.552 | 0.428 | 0.582 | 45.1 | – | – | – | 0 | – | – | – | – |
| intern2b_ts_assist_k3 | 92 | 0.130 | 0.431 | 0.417 | 0.272 | 0.716 | 0.717 | 0.109 | 0.135 | – | – | – | – | 0 | – | – | – | – |
| intern2b_ts_replace | 74 | 0.162 | 0.405 | 0.402 | 0.263 | 0.766 | 0.643 | 0.095 | 0.167 | 83.8 | – | – | – | 0 | – | – | – | – |
| intern08_ts_replace | 157 | 0.038 | 0.524 | 0.522 | 0.312 | 0.713 | 0.923 | 0.083 | 0.048 | 96.2 | – | – | – | 0 | – | – | – | – |

_Calibration over System-1 Choice decisions with gold labels (top-label confidence); yes/no: guard / trigger decisions calibrated as forecasts P(yes) of the binary gold; harmful: delegated decisions that were wrong in failed episodes (per delegated decision); FAR/FRR: guard false accept / false reject rates; rescue: wrong LLM proposals caught by the guard in episodes that succeeded._

### Meta-decision golds: coverage

| arm | dp/mode | n | labelled | coverage | doomed | unobserved | no_oracle | cf_error |
|---|---|---|---|---|---|---|---|---|
| llm_only | tool_select/off | 75 | 75 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_assist_k5 | tool_select/assist | 94 | 94 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_assist_k3 | tool_select/assist | 104 | 104 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.99 | tool_select/cascade | 75 | 75 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.99 | tool_params/cascade | 2 | 2 | 100.0% | 0 | 0 | 0 | 0 |
| intern_shadow | tool_select/shadow | 75 | 75 | 100.0% | 0 | 0 | 0 | 0 |
| intern_shadow | tool_params/shadow | 94 | 94 | 100.0% | 0 | 0 | 0 | 0 |
| intern_shadow | stop/shadow | 75 | 75 | 100.0% | 0 | 0 | 0 | 0 |
| jev_shadow | tool_select/shadow | 75 | 75 | 100.0% | 0 | 0 | 0 | 0 |
| jev_shadow | tool_params/shadow | 94 | 94 | 100.0% | 0 | 0 | 0 | 0 |
| jev_shadow | stop/shadow | 75 | 75 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_guard | tool_select/guard | 75 | 75 | 100.0% | 0 | 0 | 0 | 0 |
| intern_stop_guard | tool_select/off | 78 | 78 | 100.0% | 0 | 0 | 0 | 0 |
| intern_stop_guard | stop/guard | 13 | 13 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_guard | tool_select/guard | 76 | 76 | 100.0% | 0 | 0 | 0 | 0 |
| intern4b_uncal_ts_guard | tool_select/guard | 76 | 76 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_replace | tool_select/replace | 71 | 71 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.95 | tool_select/cascade | 73 | 73 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.95 | tool_params/cascade | 23 | 23 | 100.0% | 0 | 0 | 0 | 0 |
| llm_only_s2lite | tool_select/off | 76 | 76 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_assist_k3 | tool_select/assist | 88 | 88 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_assist_k3_s2lite | tool_select/assist | 82 | 82 | 100.0% | 0 | 0 | 0 | 0 |
| intern_shadow_s2lite | tool_select/shadow | 76 | 76 | 100.0% | 0 | 0 | 0 | 0 |
| intern_shadow_s2lite | tool_params/shadow | 100 | 100 | 100.0% | 0 | 0 | 0 | 0 |
| intern_shadow_s2lite | stop/shadow | 76 | 76 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_guard_s2lite | tool_select/guard | 74 | 74 | 100.0% | 0 | 0 | 0 | 0 |
| intern_stop_guard_s2lite | tool_select/off | 101 | 101 | 100.0% | 0 | 0 | 0 | 0 |
| intern_stop_guard_s2lite | stop/guard | 32 | 32 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.9 | tool_select/cascade | 70 | 70 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.9 | tool_params/cascade | 38 | 38 | 100.0% | 0 | 0 | 0 | 0 |
| intern08_ts_assist_k3 | tool_select/assist | 127 | 109 | 85.8% | 18 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.7 | tool_select/cascade | 36 | 36 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.7 | tool_params/cascade | 19 | 19 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.8 | tool_select/cascade | 42 | 42 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.8 | tool_params/cascade | 18 | 18 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.5 | tool_select/cascade | 32 | 32 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.5 | tool_params/cascade | 16 | 16 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_replace | tool_select/replace | 52 | 52 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_params_replace | tool_select/replace | 72 | 68 | 94.4% | 4 | 0 | 0 | 0 |
| intern_ts_params_replace | tool_params/replace | 54 | 54 | 100.0% | 0 | 0 | 0 | 0 |
| intern2b_ts_assist_k3 | tool_select/assist | 95 | 92 | 96.8% | 3 | 0 | 0 | 0 |
| intern2b_ts_replace | tool_select/replace | 84 | 74 | 88.1% | 10 | 0 | 0 | 0 |
| intern08_ts_replace | tool_select/replace | 187 | 157 | 84.0% | 30 | 0 | 0 | 0 |

### Gates vs. their primary gold (H2 inputs: measurement arms)

| arm | gate/mode | n | n+ / n− | π | AUROC | τ | J(τ) | acc(τ) | fire rate | ECE | H2 cell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| intern_shadow | stop_guard/shadow | 10 | 0 / 10 | 0.000 | – | 0.50 | – | 0.800 | 0.200 | 0.256 | no opportunity |
| intern_shadow | stop_trigger/shadow | 65 | 0 / 65 | 0.000 | – | 0.50 | – | 0.954 | 0.046 | 0.237 | no opportunity |
| jev_shadow | stop_guard/shadow | 10 | 0 / 10 | 0.000 | – | 0.50 | – | 0.900 | 0.100 | 0.159 | no opportunity |
| jev_shadow | stop_trigger/shadow | 65 | 0 / 65 | 0.000 | – | 0.50 | – | 1.000 | 0.000 | 0.027 | no opportunity |
| intern_shadow_s2lite | stop_guard/shadow | 10 | 3 / 7 | 0.300 | 1.000 | 0.50 | 1.000 | 1.000 | 0.300 | 0.248 | no opportunity |
| intern_shadow_s2lite | stop_trigger/shadow | 66 | 0 / 66 | 0.000 | – | 0.50 | – | 0.970 | 0.030 | 0.240 | no opportunity |

_Shadow records of measurement arms (every opportunity labelled), the arm's own System-1; other backends: `meta_gold replay` + `meta_gold h2`. Primary golds: reflection / routing = the System-2 action at the state is failing (need_fatal); stop guard = finishing now with the proposed answer succeeds; stop trigger = only the answer remains; memory_write = the offered text holds a value the remaining plan consumes; tool guard = the proposal is progress. A guard fires (rejects) on p_yes < τ, so J = 1 − FAR − FRR. H2 cell: ≥ 30 positive and ≥ 30 negative golds._

### Live gates vs. their gold (consistency check; not H2 inputs)

| arm | gate/mode | opportunities (retries) | labelled share | n+ / n− labelled | AUROC (observed labels) | J(τ) | realised fire rate |
|---|---|---|---|---|---|---|---|
| jev_ts_guard | tool_select/guard | 75 (0) | 100.0% | 0 / 75 | – | – | 0.013 |
| intern_stop_guard | stop_guard/guard | 11 (2) | 100.0% | 0 / 11 | – | – | 0.273 |
| intern_ts_guard | tool_select/guard | 76 (0) | 100.0% | 1 / 75 | 0.960 | 0.960 | 0.053 |
| intern4b_uncal_ts_guard | tool_select/guard | 76 (0) | 100.0% | 1 / 75 | 0.960 | 0.960 | 0.053 |
| intern_ts_guard_s2lite | tool_select/guard | 74 (0) | 100.0% | 3 / 71 | 0.986 | 0.319 | 0.027 |
| intern_stop_guard_s2lite | stop_guard/guard | 16 (16) | 100.0% | 8 / 8 | 0.836 | 0.625 | 0.438 |

_Live trigger arms label a reflection / routing opportunity only where the gate did not fire (selective labels): AUROC is computed on the labels observed at executed actions, J / accuracy / estimability are not reported, and these rows never enter H2 (META_GOLD_SPEC §9.2, §11.5–6). The realised fire rate counts every opportunity once (a stop-guard resubmission of a rejected answer is a retry)._

### Tool-selection recall@k vs. the acceptable tool set (C3)

| arm | n | recall@1 | recall@3 | recall@5 (undefined) | recall@3 (canonical) | offered recall (offered k) |
|---|---|---|---|---|---|---|
| intern_ts_assist_k5 | 94 | 0.755 | 0.894 | 0.894 (0) | 0.872 | 0.894 (5.4) |
| intern_ts_assist_k3 | 104 | 0.750 | 0.856 | 0.856 (0) | 0.827 | 0.856 (3.5) |
| intern_ts_cascade_tau0.99 | 75 | 0.747 | 0.840 | 0.853 (0) | 0.787 | – (–) |
| intern_shadow | 75 | 0.760 | 0.840 | 0.853 (0) | 0.787 | – (–) |
| jev_shadow | 75 | 0.933 | 0.973 | 0.973 (0) | 0.947 | – (–) |
| jev_ts_replace | 71 | 0.944 | 1.000 | 1.000 (0) | 0.944 | – (–) |
| intern_ts_cascade_tau0.95 | 73 | 0.767 | 0.849 | 0.863 (0) | 0.795 | – (–) |
| jev_ts_assist_k3 | 88 | 0.830 | 0.886 | 0.955 (0) | 0.864 | 0.920 (3.8) |
| intern_ts_assist_k3_s2lite | 82 | 0.720 | 0.829 | 0.829 (0) | 0.805 | 0.829 (3.6) |
| intern_shadow_s2lite | 76 | 0.737 | 0.829 | 0.842 (0) | 0.776 | – (–) |
| intern_ts_cascade_tau0.9 | 70 | 0.714 | 0.786 | 0.814 (0) | 0.729 | – (–) |
| intern08_ts_assist_k3 | 109 | 0.092 | 0.119 | 0.138 (0) | 0.101 | 0.147 (3.9) |
| intern_ts_cascade_tau0.7 | 36 | 0.611 | 0.806 | 0.833 (0) | 0.694 | – (–) |
| intern_ts_cascade_tau0.8 | 42 | 0.571 | 0.690 | 0.738 (0) | 0.595 | – (–) |
| intern_ts_cascade_tau0.5 | 32 | 0.469 | 0.625 | 0.625 (0) | 0.531 | – (–) |
| intern_ts_replace | 52 | 0.423 | 0.538 | 0.538 (0) | 0.481 | – (–) |
| intern_ts_params_replace | 68 | 0.750 | 0.838 | 0.838 (0) | 0.809 | – (–) |
| intern2b_ts_assist_k3 | 92 | 0.141 | 0.163 | 0.174 (0) | 0.098 | 0.163 (3.5) |
| intern2b_ts_replace | 74 | 0.162 | 0.297 | 0.338 (0) | 0.095 | – (–) |
| intern08_ts_replace | 157 | 0.038 | 0.045 | 0.045 (0) | 0.032 | – (–) |

_recall@k: the top-k of System-1's full ranking (tournaments log it as `s1_rank`: finalists, then each round's eliminated options by score); records whose ranking is incomplete at k (older tournament logs with only their finalists) are undefined and left out (count in brackets). Offered k = mean number of tools an assist arm offered System-2 (top-k + finish)._

### Tool arguments by fill kind (H3; §7.4)

| arm | kind | n | delegated | delegation | harmful |
|---|---|---|---|---|---|
| intern_ts_cascade_tau0.99 | bounded | 1 | 1 | 100.0% | 0.0% |
| intern_ts_cascade_tau0.99 | copy | 1 | 1 | 100.0% | 0.0% |
| intern_shadow | bounded | 45 | 0 | 0.0% | – |
| intern_shadow | copy | 61 | 0 | 0.0% | – |
| jev_shadow | bounded | 45 | 0 | 0.0% | – |
| jev_shadow | copy | 61 | 0 | 0.0% | – |
| intern_ts_cascade_tau0.95 | bounded | 13 | 11 | 84.6% | 0.0% |
| intern_ts_cascade_tau0.95 | copy | 14 | 2 | 14.3% | 0.0% |
| intern_ts_cascade_tau0.95 | free | 2 | 0 | 0.0% | – |
| intern_shadow_s2lite | bounded | 45 | 0 | 0.0% | – |
| intern_shadow_s2lite | copy | 62 | 0 | 0.0% | – |
| intern_ts_cascade_tau0.9 | bounded | 24 | 14 | 58.3% | 0.0% |
| intern_ts_cascade_tau0.9 | copy | 25 | 2 | 8.0% | 0.0% |
| intern_ts_cascade_tau0.9 | free | 4 | 0 | 0.0% | – |
| intern_ts_cascade_tau0.7 | bounded | 19 | 11 | 57.9% | 0.0% |
| intern_ts_cascade_tau0.7 | copy | 23 | 3 | 13.0% | 0.0% |
| intern_ts_cascade_tau0.7 | free | 3 | 0 | 0.0% | – |
| intern_ts_cascade_tau0.8 | bounded | 19 | 11 | 57.9% | 0.0% |
| intern_ts_cascade_tau0.8 | copy | 22 | 2 | 9.1% | 0.0% |
| intern_ts_cascade_tau0.8 | free | 2 | 0 | 0.0% | – |
| intern_ts_cascade_tau0.5 | bounded | 18 | 10 | 55.6% | 0.0% |
| intern_ts_cascade_tau0.5 | copy | 26 | 4 | 15.4% | 0.0% |
| intern_ts_cascade_tau0.5 | free | 4 | 0 | 0.0% | – |
| intern_ts_params_replace | bounded | 55 | 55 | 100.0% | 88.4% |
| intern_ts_params_replace | copy | 24 | 24 | 100.0% | 0.0% |
| intern_ts_params_replace | free | 47 | 0 | 0.0% | – |

### Sweeps

| sweep | arms | values | APGR | CPT(50%) | CPT(80%) | AIQ | AIQ zero-router | ΔAIQ |
|---|---|---|---|---|---|---|---|---|
| intern_ts_cascade_tau | 6 | 0.5, 0.7, 0.8, 0.9, 0.95, 0.99 | – | – | – | 0.704 | – | – |

_APGR / CPT: x = strong-model calls relative to the baseline, anchors weak (PGR 0) and baseline (PGR 1); AIQ: mean success of the non-decreasing convex hull over the linear cost range between the two references (sweep + references); zero-router: hull of the references alone._

