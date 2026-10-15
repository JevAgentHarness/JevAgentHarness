## Score card (baseline: `llm_only`)

- **Iso-quality cost savings**: 63.9% [56.8, 75.3] with `jev_ts_replace` (Δsuccess -10.0 pp [-30.0, 0.0], p_worse=0.349; not significantly worse, one-sided paired bootstrap, α=0.05; 6 arm(s) qualify)
- **Quality at iso-cost** (≤ 0.0393 $/episode): `llm_only` 100.0% vs. reference 100.0% (Δ +0.0 pp, p_better=–); hull 100.0%
- **Iso-quality cost savings** (vs. `llm_only_s2lite`): 69.1% [64.1, 73.0] with `intern_ts_assist_k3_s2lite` (Δsuccess -20.0 pp [-50.0, 0.0], p_worse=0.103; not significantly worse, one-sided paired bootstrap, α=0.05; 3 arm(s) qualify)
- **Quality at iso-cost** (vs. `llm_only_s2lite`) (≤ 0.00957 $/episode): `intern_ts_guard_s2lite` 70.0% vs. reference 60.0% (Δ +10.0 pp, p_better=0.352); hull 70.0%
- **Frontier cost-of-pass**: 0.0111 $/success over 27 arms (3 shadow arm(s) left out of the frontier) (best single arm: `intern_ts_assist_k3_s2lite`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.697, baseline alone 0.144; front: `intern_ts_assist_k3_s2lite`, `intern_ts_guard_s2lite`, `jev_ts_replace`, `llm_only`
- **Sweep `intern_ts_cascade_tau*`** (6 arms): AIQ 0.617
- note: per-System-2 references: `llm_only_s2lite` (gemini-3.5-flash-lite) for 5 arm(s); every other arm vs. `llm_only`

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | McNemar p | lat save % | LoP s | LLM-call ratio | strong-call ratio | PGR | EuB | HV contrib | LLM-free steps % | cache-mod $ | no-cache $ | cached % | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 100.0 [100.0, 100.0] | – | – | 0.0393 | – | 0.0393 | 11.4 | 0.0 | – | – | 11.4 | 1.00 | 1.00 | – | 0.500 | 0.002 | 0.0 | 0.0393 | 0.0704 | 49.8 | front,baseline |
| intern_shadow | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0393 | -0.0 | 0.0393 | 20.8 | 0.0 | 1.000 | -81.6 | 20.8 | 1.00 | 1.00 | – | 0.500 | – | 0.0 | 0.0393 | 0.0705 | 49.8 |  |
| jev_shadow | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0396 | -0.6 | 0.0396 | 13.0 | 0.0 | 1.000 | -13.3 | 13.0 | 1.00 | 1.00 | – | 0.497 | – | 0.0 | 0.0396 | 0.0704 | 49.1 |  |
| jev_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0413 | -5.0 | 0.0413 | 11.0 | 0.0 | 1.000 | 3.9 | 11.0 | 1.03 | 1.03 | – | 0.475 | 0.000 | 0.0 | 0.0413 | 0.0729 | 49.0 | iso-q |
| intern_stop_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0414 | -5.2 | 0.0414 | 14.3 | 0.0 | 1.000 | -24.6 | 14.3 | 1.03 | 1.03 | – | 0.474 | 0.000 | 0.0 | 0.0414 | 0.0728 | 47.8 | iso-q |
| intern_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0420 | -6.9 | 0.0420 | 17.8 | 0.0 | 1.000 | -55.7 | 17.8 | 1.07 | 1.07 | – | 0.466 | 0.000 | 0.0 | 0.0420 | 0.0752 | 50.2 | iso-q |
| intern4b_uncal_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0423 | -7.6 | 0.0423 | 15.7 | 0.0 | 1.000 | -37.2 | 15.7 | 1.07 | 1.07 | – | 0.462 | 0.000 | 0.0 | 0.0423 | 0.0753 | 49.5 | iso-q |
| jev_ts_replace | 90.0 [70.0, 100.0] | -10.0 | 0.349 | 0.0142 | 63.9 | 0.0158 | 15.4 | 100.0 | 1.000 | -34.8 | 17.1 | 0.92 | 0.92 | – | 0.722 | 0.048 | 0.0 | 0.0142 | 0.0142 | 0.0 | front,iso-best |
| jev_ts_assist_k3 | 80.0 [50.0, 100.0] | -20.0 | 0.107 | 0.0219 | 44.3 | 0.0274 | 10.3 | 0.0 | 0.500 | 10.0 | 12.9 | 1.19 | 1.19 | – | 0.575 | 0.000 | 0.0 | 0.0219 | 0.0219 | 0.0 | iso-q |
| intern_ts_guard_s2lite | 70.0 [40.0, 100.0] | +10.0 | 1.000 | 0.00929 | 3.0 | 0.0133 | 7.47 | 0.0 | 1.000 | -36.0 | 10.7 | 0.95 | 0.95 | – | 0.623 | 0.016 | 0.0 | 0.00929 | 0.0135 | 32.9 | front,iso-q,vs llm_only_s2lite |
| intern_shadow_s2lite | 70.0 [40.0, 100.0] | +10.0 | 1.000 | 0.00957 | 0.0 | 0.0137 | 5.71 | 0.0 | 1.000 | -4.0 | 8.16 | 1.00 | 1.00 | – | 0.624 | – | 0.0 | 0.00957 | 0.0143 | 34.8 | vs llm_only_s2lite |
| llm_only_s2lite | 60.0 [30.0, 90.0] | -40.0 | 0.008 | 0.00957 | 75.7 | 0.0160 | 5.49 | 0.0 | 0.125 | 52.0 | 9.16 | 1.01 | 1.01 | – | 0.539 | 0.000 | 0.0 | 0.00957 | 0.0143 | 34.8 |  |
| intern_stop_guard_s2lite | 60.0 [30.0, 90.0] | +0.0 | 1.000 | 0.0113 | -17.6 | 0.0188 | 7.48 | 0.0 | 1.000 | -36.1 | 12.5 | 1.18 | 1.18 | – | 0.529 | 0.000 | 0.0 | 0.0113 | 0.0175 | 37.0 | iso-q,vs llm_only_s2lite |
| intern_ts_assist_k3 | 60.0 [30.0, 90.0] | -40.0 | 0.006 | 0.0193 | 51.0 | 0.0321 | 20.7 | 0.0 | 0.125 | -80.9 | 34.5 | 1.13 | 1.13 | – | 0.425 | 0.000 | 4.5 | 0.0193 | 0.0193 | 0.0 | worse |
| intern_ts_assist_k5 | 50.0 [20.0, 80.0] | -50.0 | 0.002 | 0.0176 | 55.3 | 0.0351 | 22.0 | 0.0 | 0.062 | -92.7 | 44.1 | 1.03 | 1.03 | – | 0.375 | 0.000 | 6.1 | 0.0176 | 0.0176 | 0.0 | worse |
| intern_ts_cascade_tau0.99 | 50.0 [20.0, 80.0] | -50.0 | 0.002 | 0.0296 | 24.7 | 0.0593 | 15.0 | 4.9 | 0.062 | -30.9 | 30.0 | 0.77 | 0.77 | – | 0.341 | 0.000 | 9.4 | 0.0296 | 0.0518 | 48.6 | worse |
| intern_ts_assist_k3_s2lite | 40.0 [10.0, 70.0] | -20.0 | 0.103 | 0.00296 | 69.1 | 0.00740 | 11.4 | 0.0 | 0.500 | -107.6 | 28.5 | 0.91 | 0.91 | – | 0.389 | 0.113 | 6.8 | 0.00296 | 0.00296 | 0.0 | front,iso-best,vs llm_only_s2lite |
| intern_ts_cascade_tau0.95 | 40.0 [10.0, 70.0] | -60.0 | <0.001 | 0.0256 | 34.9 | 0.0641 | 13.0 | 25.3 | 0.031 | -13.8 | 32.5 | 0.75 | 0.75 | – | 0.301 | 0.000 | 11.1 | 0.0256 | 0.0436 | 45.3 | worse |
| intern08_ts_assist_k3 | 40.0 [10.0, 70.0] | -60.0 | <0.001 | 0.0393 | 0.2 | 0.0981 | 26.9 | 0.0 | 0.031 | -134.9 | 67.2 | 1.88 | 1.88 | – | 0.281 | 0.000 | 0.0 | 0.0393 | 0.0454 | 3.3 | worse |
| intern_ts_cascade_tau0.9_s2lite | 20.0 [0.0, 50.0] | -40.0 | 0.003 | 0.00565 | 41.0 | 0.0282 | 10.8 | 31.0 | 0.125 | -95.9 | 53.8 | 0.70 | 0.70 | – | 0.185 | 0.000 | 10.2 | 0.00565 | 0.00753 | 23.4 | worse,vs llm_only_s2lite |
| intern_ts_cascade_tau0.9 | 20.0 [0.0, 50.0] | -80.0 | <0.001 | 0.0215 | 45.3 | 0.108 | 13.3 | 31.8 | 0.008 | -16.6 | 66.7 | 0.69 | 0.69 | – | 0.149 | 0.000 | 11.9 | 0.0215 | 0.0356 | 43.3 | worse |
| intern_ts_cascade_tauhat | 20.0 [0.0, 50.0] | -80.0 | <0.001 | 0.0216 | 45.2 | 0.108 | 14.7 | 30.7 | 0.008 | -28.7 | 73.6 | 0.71 | 0.71 | – | 0.148 | 0.000 | 10.2 | 0.0216 | 0.0357 | 43.2 | worse |
| intern_ts_cascade_tau0.5 | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.00473 | 88.0 | 0.0473 | 7.73 | 54.5 | 0.004 | 32.5 | 77.3 | 0.40 | 0.40 | – | 0.093 | 0.000 | 3.2 | 0.00473 | 0.00551 | 5.8 | worse |
| intern_ts_cascade_tau0.7 | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.00820 | 79.2 | 0.0820 | 8.64 | 48.6 | 0.004 | 24.5 | 86.4 | 0.39 | 0.39 | – | 0.087 | 0.000 | 6.5 | 0.00820 | 0.0108 | 27.0 | worse |
| intern_ts_replace | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0121 | 69.2 | 0.121 | 16.8 | 100.0 | 0.004 | -46.7 | 168. | 0.72 | 0.72 | – | 0.090 | 0.000 | 0.0 | 0.0121 | 0.0121 | 0.0 | worse |
| intern_ts_cascade_tau0.8 | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0123 | 68.8 | 0.123 | 9.08 | 43.1 | 0.004 | 20.6 | 90.8 | 0.47 | 0.47 | – | 0.081 | 0.000 | 10.3 | 0.0123 | 0.0180 | 31.0 | worse |
| intern2b_ts_assist_k3 | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0151 | 61.7 | 0.151 | 13.3 | 0.0 | 0.004 | -16.2 | 133. | 1.16 | 1.16 | – | 0.094 | 0.000 | 0.0 | 0.0151 | 0.0151 | 0.0 | worse |
| intern_ts_params_replace | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0210 | 46.5 | 0.210 | 32.3 | 74.6 | 0.004 | -182.5 | 323. | 0.80 | 0.80 | – | 0.096 | 0.000 | 28.6 | 0.0210 | 0.0210 | 0.0 | worse |
| intern2b_ts_replace | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0340 | 13.6 | 0.340 | 30.4 | 100.0 | 0.004 | -166.2 | 304. | 1.16 | 1.16 | – | 0.097 | 0.000 | 0.0 | 0.0340 | 0.0451 | 6.3 | worse |
| intern08_ts_replace | 0.0 [0.0, 0.0] | -100.0 | <0.001 | 0.0483 | -22.7 | ∞ | 36.3 | 100.0 | 0.002 | -217.7 | ∞ | 2.13 | 2.13 | – | 0.000 | 0.000 | 0.0 | 0.0483 | 0.0600 | 6.5 | worse |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

_cost $: the billed (ledger) cost, the primary figure; cache-mod $: System-2 calls re-priced from their logged tokens with the measured cached tokens at the model's cached rate; no-cache $: list price with no cache; cached %: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md)._

### Non-inferiority (A6 procedure; margin 3.0 pp)

| arm | reference | n | Δ pp | 95% LCB pp | p_NI | Holm p | α_adj | LCB(1−α_adj) pp | discordant (w/l) | exact p (loss UB pp) | MDE Δ pp (80%) | Tango 95% LB pp | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_ts_guard | llm_only | 10 | +0.0 | +0.0 | <0.001 | 1.000 | 0.0026 | +0.0 | 0 (0/0) | 0.737 (44.79) | – | -21.3 | inconclusive at this n |
| intern_stop_guard | llm_only | 10 | +0.0 | +0.0 | <0.001 | 1.000 | 0.0024 | +0.0 | 0 (0/0) | 0.737 (45.34) | – | -21.3 | inconclusive at this n |
| intern_ts_guard | llm_only | 10 | +0.0 | +0.0 | <0.001 | 1.000 | 0.0025 | +0.0 | 0 (0/0) | 0.737 (45.07) | – | -21.3 | inconclusive at this n |
| intern4b_uncal_ts_guard | llm_only | 10 | +0.0 | +0.0 | <0.001 | 1.000 | 0.0023 | +0.0 | 0 (0/0) | 0.737 (45.59) | – | -21.3 | inconclusive at this n |
| jev_ts_replace | llm_only | 10 | -10.0 | -30.0 | 0.650 | 1.000 | 0.0028 | -40.0 | 1 (0/1) | 0.965 (57.53) | +28.3 | -34.8 | inconclusive at this n |
| jev_ts_assist_k3 | llm_only | 10 | -20.0 | -40.0 | 0.892 | 1.000 | 0.0029 | -60.0 | 2 (0/2) | 0.997 (67.34) | +40.8 | -45.9 | inconclusive at this n |
| intern_ts_guard_s2lite | llm_only_s2lite | 10 | +10.0 | +0.0 | <0.001 | 1.000 | 0.0167 | +0.0 | 1 (1/0) | 0.737 (33.60) | +23.7 | -13.4 | inconclusive at this n |
| llm_only_s2lite | llm_only | 10 | -40.0 | -60.0 | 0.992 | 1.000 | 0.0038 | -80.0 | 4 (0/4) | 1.000 (81.84) | +57.3 | -64.8 | not NI |
| intern_stop_guard_s2lite | llm_only_s2lite | 10 | +0.0 | +0.0 | <0.001 | 1.000 | 0.0125 | +0.0 | 0 (0/0) | 0.737 (35.48) | – | -21.3 | inconclusive at this n |
| intern_ts_assist_k3 | llm_only | 10 | -40.0 | -70.0 | 0.994 | 1.000 | 0.0036 | -80.0 | 4 (0/4) | 1.000 (82.09) | +57.7 | -64.8 | not NI |
| intern_ts_assist_k5 | llm_only | 10 | -50.0 | -80.0 | 0.998 | 1.000 | 0.0031 | -90.0 | 5 (0/5) | – | +65.3 | -73.1 | not NI |
| intern_ts_cascade_tau0.99 | llm_only | 10 | -50.0 | -80.0 | 0.998 | 1.000 | 0.0033 | -90.0 | 5 (0/5) | – | +65.0 | -73.1 | not NI |
| intern_ts_assist_k3_s2lite | llm_only_s2lite | 10 | -20.0 | -40.0 | 0.896 | 1.000 | 0.0250 | -50.0 | 2 (0/2) | 0.997 (55.61) | +32.8 | -45.9 | inconclusive at this n |
| intern_ts_cascade_tau0.95 | llm_only | 10 | -60.0 | -80.0 | 1.000 | 1.000 | 0.0125 | -90.0 | 6 (0/6) | – | +63.6 | -80.6 | not NI |
| intern08_ts_assist_k3 | llm_only | 10 | -60.0 | -80.0 | 1.000 | 1.000 | 0.0042 | -96.7 | 6 (0/6) | – | +70.3 | -80.6 | not NI |
| intern_ts_cascade_tau0.9_s2lite | llm_only_s2lite | 10 | -40.0 | -70.0 | 0.998 | 1.000 | 0.0500 | -70.0 | 4 (0/4) | 1.000 (69.65) | +42.4 | -64.8 | not NI |
| intern_ts_cascade_tau0.9 | llm_only | 10 | -80.0 | -100.0 | 1.000 | 1.000 | 0.0100 | -100.0 | 8 (0/8) | – | +75.5 | -93.1 | not NI |
| intern_ts_cascade_tauhat | llm_only | 10 | -80.0 | -100.0 | 1.000 | 1.000 | 0.0167 | -100.0 | 8 (0/8) | – | +71.5 | -93.1 | not NI |
| intern_ts_cascade_tau0.5 | llm_only | 10 | -90.0 | -100.0 | 1.000 | 1.000 | 0.0063 | -100.0 | 9 (0/9) | – | +83.8 | -97.7 | not NI |
| intern_ts_cascade_tau0.7 | llm_only | 10 | -90.0 | -100.0 | 1.000 | 1.000 | 0.0071 | -100.0 | 9 (0/9) | – | +82.8 | -97.7 | not NI |
| intern_ts_replace | llm_only | 10 | -90.0 | -100.0 | 1.000 | 1.000 | 0.0500 | -100.0 | 9 (0/9) | – | +64.8 | -97.7 | not NI |
| intern_ts_cascade_tau0.8 | llm_only | 10 | -90.0 | -100.0 | 1.000 | 1.000 | 0.0083 | -100.0 | 9 (0/9) | – | +81.6 | -97.7 | not NI |
| intern2b_ts_assist_k3 | llm_only | 10 | -90.0 | -100.0 | 1.000 | 1.000 | 0.0050 | -100.0 | 9 (0/9) | – | +85.3 | -97.7 | not NI |
| intern_ts_params_replace | llm_only | 10 | -90.0 | -100.0 | 1.000 | 1.000 | 0.0250 | -100.0 | 9 (0/9) | – | +72.2 | -97.7 | not NI |
| intern2b_ts_replace | llm_only | 10 | -90.0 | -100.0 | 1.000 | 1.000 | 0.0056 | -100.0 | 9 (0/9) | – | +84.6 | -97.7 | not NI |
| intern08_ts_replace | llm_only | 10 | -100.0 | -100.0 | 1.000 | 1.000 | 0.0045 | -100.0 | 10 (0/10) | – | +90.7 | -100.0 | not NI |

_A6 / P0-1: p_NI = P*(Δ* ≤ −m) from the paired task bootstrap (B = n_boot, task = resampling unit); it enters Holm with the family (every acting arm vs. the same reference); NI holds iff p_NI ≤ α_adj of its Holm step, equivalently iff the one-sided (1 − α_adj) bound is above −m; the 95% LCB is descriptive. With < 5 discordant tasks the exact binomial on losses (p = P[L ≤ l | n, m]) decides (it is the p that enters Holm, `Holm p`) and the one-sided 95% upper bound on the loss rate (1 − 0.05^(1/n) at zero losses) replaces a zero-width interval. A primary that fails while its two-sided 95% CI covers 0 is *inconclusive at this n*, not inferior; MDE: the smallest true Δ shown NI with 80% power at the achieved discordance and α_adj. Tango LB: descriptive score interval._

### System-1 decisions

| arm | n scored | S1 acc | ECE | ACE | Brier | AUROC | AURC | E-AURC | acc@80%cov | harmful % | guard FAR | guard FRR | rescue % | yes/no n | yes/no ECE | yes/no Brier | yes/no AUROC | mem stored % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intern_shadow | 169 | 0.822 | 0.068 | 0.078 | 0.131 | 0.766 | 0.076 | 0.059 | 0.882 | – | – | – | – | 75 | 0.205 | 0.072 | 0.979 | – |
| jev_shadow | 169 | 0.953 | 0.021 | 0.022 | 0.040 | 0.877 | 0.008 | 0.007 | 0.985 | – | – | – | – | 75 | 0.047 | 0.008 | 1.000 | – |
| jev_ts_guard | 0 | – | – | – | – | – | – | – | – | – | – | 0.027 | – | 75 | 0.121 | 0.026 | – | – |
| intern_stop_guard | 0 | – | – | – | – | – | – | – | – | – | – | 0.167 | – | 12 | 0.292 | 0.126 | – | – |
| intern_ts_guard | 0 | – | – | – | – | – | – | – | – | – | 0.000 | 0.040 | 100.0 | 76 | 0.141 | 0.044 | 0.960 | – |
| intern4b_uncal_ts_guard | 0 | – | – | – | – | – | – | – | – | – | 0.000 | 0.040 | 100.0 | 76 | 0.071 | 0.038 | 0.960 | – |
| jev_ts_replace | 69 | 0.971 | 0.070 | 0.047 | 0.037 | 0.899 | 0.004 | 0.003 | 1.000 | 1.4 | – | – | – | 0 | – | – | – | – |
| jev_ts_assist_k3 | 89 | 0.798 | 0.106 | 0.100 | 0.115 | 0.876 | 0.052 | 0.028 | 0.889 | – | – | – | – | 0 | – | – | – | – |
| intern_ts_guard_s2lite | 0 | – | – | – | – | – | – | – | – | – | 0.500 | 0.015 | 50.0 | 70 | 0.117 | 0.037 | 0.978 | – |
| intern_shadow_s2lite | 166 | 0.837 | 0.103 | 0.120 | 0.153 | 0.658 | 0.100 | 0.085 | 0.865 | – | – | – | – | 76 | 0.219 | 0.079 | 0.973 | – |
| intern_stop_guard_s2lite | 0 | – | – | – | – | – | – | – | – | – | 0.571 | 0.250 | 0.0 | 24 | 0.376 | 0.297 | 0.517 | – |
| intern_ts_assist_k3 | 85 | 0.718 | 0.081 | 0.115 | 0.201 | 0.630 | 0.190 | 0.144 | 0.735 | – | – | – | – | 0 | – | – | – | – |
| intern_ts_assist_k5 | 77 | 0.701 | 0.110 | 0.124 | 0.199 | 0.663 | 0.190 | 0.138 | 0.742 | – | – | – | – | 0 | – | – | – | – |
| intern_ts_cascade_tau0.99 | 61 | 0.705 | 0.123 | 0.163 | 0.201 | 0.699 | 0.167 | 0.116 | 0.755 | 0.0 | – | – | – | 0 | – | – | – | – |
| intern_ts_assist_k3_s2lite | 69 | 0.681 | 0.107 | 0.118 | 0.189 | 0.731 | 0.175 | 0.115 | 0.750 | – | – | – | – | 0 | – | – | – | – |
| intern_ts_cascade_tau0.95 | 73 | 0.767 | 0.093 | 0.135 | 0.171 | 0.672 | 0.134 | 0.103 | 0.797 | 5.6 | – | – | – | 0 | – | – | – | – |
| intern08_ts_assist_k3 | 119 | 0.092 | 0.308 | 0.305 | 0.170 | 0.729 | 0.834 | 0.143 | 0.115 | – | – | – | – | 0 | – | – | – | – |
| intern_ts_cascade_tau0.9_s2lite | 77 | 0.779 | 0.133 | 0.124 | 0.187 | 0.606 | 0.151 | 0.123 | 0.790 | 12.5 | – | – | – | 0 | – | – | – | – |
| intern_ts_cascade_tau0.9 | 76 | 0.737 | 0.126 | 0.142 | 0.200 | 0.623 | 0.171 | 0.131 | 0.754 | 16.0 | – | – | – | 0 | – | – | – | – |
| intern_ts_cascade_tauhat | 76 | 0.711 | 0.129 | 0.176 | 0.208 | 0.635 | 0.184 | 0.135 | 0.721 | 16.7 | – | – | – | 0 | – | – | – | – |
| intern_ts_cascade_tau0.5 | 47 | 0.681 | 0.197 | 0.194 | 0.229 | 0.569 | 0.291 | 0.230 | 0.737 | 39.4 | – | – | – | 0 | – | – | – | – |
| intern_ts_cascade_tau0.7 | 44 | 0.705 | 0.168 | 0.213 | 0.234 | 0.516 | 0.274 | 0.222 | 0.722 | 36.0 | – | – | – | 0 | – | – | – | – |
| intern_ts_replace | 54 | 0.407 | 0.278 | 0.286 | 0.302 | 0.668 | 0.468 | 0.235 | 0.477 | 57.4 | – | – | – | 0 | – | – | – | – |
| intern_ts_cascade_tau0.8 | 49 | 0.673 | 0.194 | 0.192 | 0.260 | 0.504 | 0.296 | 0.232 | 0.650 | 34.8 | – | – | – | 0 | – | – | – | – |
| intern2b_ts_assist_k3 | 84 | 0.143 | 0.416 | 0.400 | 0.263 | 0.736 | 0.688 | 0.104 | 0.147 | – | – | – | – | 0 | – | – | – | – |
| intern_ts_params_replace | 145 | 0.531 | 0.356 | 0.353 | 0.370 | 0.298 | 0.626 | 0.491 | 0.517 | 46.9 | – | – | – | 0 | – | – | – | – |
| intern2b_ts_replace | 77 | 0.156 | 0.398 | 0.396 | 0.253 | 0.769 | 0.654 | 0.094 | 0.161 | 84.4 | – | – | – | 0 | – | – | – | – |
| intern08_ts_replace | 133 | 0.045 | 0.527 | 0.527 | 0.325 | 0.689 | 0.917 | 0.099 | 0.056 | 95.5 | – | – | – | 0 | – | – | – | – |

_Calibration over System-1 Choice decisions with gold labels (top-label confidence); yes/no: guard / trigger decisions calibrated as forecasts P(yes) of the binary gold; harmful: delegated decisions that were wrong in failed episodes (per delegated decision); FAR/FRR: guard false accept / false reject rates; rescue: wrong LLM proposals caught by the guard in episodes that succeeded._

### Meta-decision golds: coverage

| arm | dp/mode | n | labelled | coverage | doomed | unobserved | no_oracle | cf_error |
|---|---|---|---|---|---|---|---|---|
| llm_only | tool_select/off | 75 | 75 | 100.0% | 0 | 0 | 0 | 0 |
| intern_shadow | tool_select/shadow | 75 | 75 | 100.0% | 0 | 0 | 0 | 0 |
| intern_shadow | tool_params/shadow | 94 | 94 | 100.0% | 0 | 0 | 0 | 0 |
| intern_shadow | stop/shadow | 75 | 75 | 100.0% | 0 | 0 | 0 | 0 |
| jev_shadow | tool_select/shadow | 75 | 75 | 100.0% | 0 | 0 | 0 | 0 |
| jev_shadow | tool_params/shadow | 94 | 94 | 100.0% | 0 | 0 | 0 | 0 |
| jev_shadow | stop/shadow | 75 | 75 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_guard | tool_select/guard | 75 | 75 | 100.0% | 0 | 0 | 0 | 0 |
| intern_stop_guard | tool_select/off | 77 | 77 | 100.0% | 0 | 0 | 0 | 0 |
| intern_stop_guard | stop/guard | 12 | 12 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_guard | tool_select/guard | 76 | 76 | 100.0% | 0 | 0 | 0 | 0 |
| intern4b_uncal_ts_guard | tool_select/guard | 76 | 76 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_replace | tool_select/replace | 69 | 69 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_assist_k3 | tool_select/assist | 89 | 89 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_guard_s2lite | tool_select/guard | 70 | 70 | 100.0% | 0 | 0 | 0 | 0 |
| intern_shadow_s2lite | tool_select/shadow | 76 | 76 | 100.0% | 0 | 0 | 0 | 0 |
| intern_shadow_s2lite | tool_params/shadow | 100 | 100 | 100.0% | 0 | 0 | 0 | 0 |
| intern_shadow_s2lite | stop/shadow | 76 | 76 | 100.0% | 0 | 0 | 0 | 0 |
| llm_only_s2lite | tool_select/off | 76 | 76 | 100.0% | 0 | 0 | 0 | 0 |
| intern_stop_guard_s2lite | tool_select/off | 90 | 90 | 100.0% | 0 | 0 | 0 | 0 |
| intern_stop_guard_s2lite | stop/guard | 24 | 24 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_assist_k3 | tool_select/assist | 85 | 85 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_assist_k5 | tool_select/assist | 77 | 77 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.99 | tool_select/cascade | 59 | 59 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.99 | tool_params/cascade | 2 | 2 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_assist_k3_s2lite | tool_select/assist | 69 | 69 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.95 | tool_select/cascade | 58 | 58 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.95 | tool_params/cascade | 15 | 15 | 100.0% | 0 | 0 | 0 | 0 |
| intern08_ts_assist_k3 | tool_select/assist | 141 | 119 | 84.4% | 22 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.9_s2lite | tool_select/cascade | 54 | 54 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.9_s2lite | tool_params/cascade | 23 | 23 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.9 | tool_select/cascade | 54 | 54 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.9 | tool_params/cascade | 22 | 22 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tauhat | tool_select/cascade | 54 | 54 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tauhat | tool_params/cascade | 22 | 22 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.5 | tool_select/cascade | 31 | 31 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.5 | tool_params/cascade | 16 | 16 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.7 | tool_select/cascade | 31 | 31 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.7 | tool_params/cascade | 13 | 13 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_replace | tool_select/replace | 54 | 54 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.8 | tool_select/cascade | 37 | 37 | 100.0% | 0 | 0 | 0 | 0 |
| intern_ts_cascade_tau0.8 | tool_params/cascade | 12 | 12 | 100.0% | 0 | 0 | 0 | 0 |
| intern2b_ts_assist_k3 | tool_select/assist | 87 | 84 | 96.6% | 3 | 0 | 0 | 0 |
| intern_ts_params_replace | tool_select/replace | 84 | 76 | 90.5% | 8 | 0 | 0 | 0 |
| intern_ts_params_replace | tool_params/replace | 69 | 69 | 100.0% | 0 | 0 | 0 | 0 |
| intern2b_ts_replace | tool_select/replace | 87 | 77 | 88.5% | 10 | 0 | 0 | 0 |
| intern08_ts_replace | tool_select/replace | 160 | 133 | 83.1% | 27 | 0 | 0 | 0 |

### Gates vs. their primary gold (H2 inputs: measurement arms)

| arm | gate/mode | n | n+ / n− | π | AUROC | τ | J(τ) | acc(τ) | fire rate | ECE | H2 cell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| intern_shadow | stop_guard/shadow | 10 | 0 / 10 | 0.000 | – | 0.50 | – | 0.800 | 0.200 | 0.250 | no opportunity |
| intern_shadow | stop_trigger/shadow | 65 | 0 / 65 | 0.000 | – | 0.50 | – | 0.954 | 0.046 | 0.235 | no opportunity |
| jev_shadow | stop_guard/shadow | 10 | 0 / 10 | 0.000 | – | 0.50 | – | 0.900 | 0.100 | 0.173 | no opportunity |
| jev_shadow | stop_trigger/shadow | 65 | 0 / 65 | 0.000 | – | 0.50 | – | 1.000 | 0.000 | 0.028 | no opportunity |
| intern_shadow_s2lite | stop_guard/shadow | 10 | 3 / 7 | 0.300 | 0.881 | 0.50 | 0.857 | 0.900 | 0.400 | 0.208 | no opportunity |
| intern_shadow_s2lite | stop_trigger/shadow | 66 | 0 / 66 | 0.000 | – | 0.50 | – | 0.924 | 0.076 | 0.242 | no opportunity |

_Shadow records of measurement arms (every opportunity labelled), the arm's own System-1; other backends: `meta_gold replay` + `meta_gold h2`. Primary golds: reflection / routing = the System-2 action at the state is failing (need_fatal); stop guard = finishing now with the proposed answer succeeds; stop trigger = only the answer remains; memory_write = the offered text holds a value the remaining plan consumes; tool guard = the proposal is progress. A guard fires (rejects) on p_yes < τ, so J = 1 − FAR − FRR. H2 cell: ≥ 30 positive and ≥ 30 negative golds._

### Live gates vs. their gold (consistency check; not H2 inputs)

| arm | gate/mode | opportunities (retries) | labelled share | n+ / n− labelled | AUROC (observed labels) | J(τ) | realised fire rate |
|---|---|---|---|---|---|---|---|
| jev_ts_guard | tool_select/guard | 75 (0) | 100.0% | 0 / 75 | – | – | 0.027 |
| intern_stop_guard | stop_guard/guard | 12 (0) | 100.0% | 0 / 12 | – | – | 0.167 |
| intern_ts_guard | tool_select/guard | 76 (0) | 100.0% | 1 / 75 | 0.960 | 0.960 | 0.053 |
| intern4b_uncal_ts_guard | tool_select/guard | 76 (0) | 100.0% | 1 / 75 | 0.960 | 0.960 | 0.053 |
| intern_ts_guard_s2lite | tool_select/guard | 70 (0) | 100.0% | 2 / 68 | 0.978 | 0.485 | 0.029 |
| intern_stop_guard_s2lite | stop_guard/guard | 15 (9) | 100.0% | 7 / 8 | 0.804 | 0.179 | 0.333 |

_Live trigger arms label a reflection / routing opportunity only where the gate did not fire (selective labels): AUROC is computed on the labels observed at executed actions, J / accuracy / estimability are not reported, and these rows never enter H2 (META_GOLD_SPEC §9.2, §11.5–6). The realised fire rate counts every opportunity once (a stop-guard resubmission of a rejected answer is a retry)._

### Tool-selection recall@k vs. the acceptable tool set (C3)

| arm | n | recall@1 | recall@3 | recall@5 (undefined) | recall@3 (canonical) | offered recall (offered k) |
|---|---|---|---|---|---|---|
| intern_shadow | 75 | 0.760 | 0.853 | 0.867 (0) | 0.800 | – (–) |
| jev_shadow | 75 | 0.933 | 0.973 | 0.973 (0) | 0.947 | – (–) |
| jev_ts_replace | 69 | 0.971 | 1.000 | 1.000 (0) | 0.942 | – (–) |
| jev_ts_assist_k3 | 89 | 0.798 | 0.876 | 0.966 (0) | 0.854 | 0.921 (3.8) |
| intern_shadow_s2lite | 66 | 0.712 | 0.803 | 0.818 (0) | 0.742 | – (–) |
| intern_ts_assist_k3 | 85 | 0.706 | 0.835 | 0.835 (0) | 0.800 | 0.835 (3.5) |
| intern_ts_assist_k5 | 77 | 0.701 | 0.870 | 0.870 (0) | 0.844 | 0.870 (5.5) |
| intern_ts_cascade_tau0.99 | 59 | 0.678 | 0.797 | 0.814 (0) | 0.729 | – (–) |
| intern_ts_assist_k3_s2lite | 69 | 0.667 | 0.754 | 0.754 (0) | 0.725 | 0.754 (3.8) |
| intern_ts_cascade_tau0.95 | 58 | 0.707 | 0.810 | 0.828 (0) | 0.741 | – (–) |
| intern08_ts_assist_k3 | 119 | 0.092 | 0.126 | 0.143 (0) | 0.109 | 0.151 (3.9) |
| intern_ts_cascade_tau0.9_s2lite | 54 | 0.704 | 0.796 | 0.815 (0) | 0.722 | – (–) |
| intern_ts_cascade_tau0.9 | 54 | 0.648 | 0.741 | 0.759 (0) | 0.667 | – (–) |
| intern_ts_cascade_tauhat | 54 | 0.611 | 0.722 | 0.759 (0) | 0.648 | – (–) |
| intern_ts_cascade_tau0.5 | 31 | 0.516 | 0.677 | 0.710 (0) | 0.581 | – (–) |
| intern_ts_cascade_tau0.7 | 31 | 0.613 | 0.742 | 0.774 (0) | 0.613 | – (–) |
| intern_ts_replace | 54 | 0.407 | 0.500 | 0.500 (0) | 0.444 | – (–) |
| intern_ts_cascade_tau0.8 | 37 | 0.568 | 0.703 | 0.730 (0) | 0.595 | – (–) |
| intern2b_ts_assist_k3 | 84 | 0.155 | 0.179 | 0.190 (0) | 0.107 | 0.179 (3.5) |
| intern_ts_params_replace | 76 | 0.803 | 0.855 | 0.855 (0) | 0.829 | – (–) |
| intern2b_ts_replace | 77 | 0.156 | 0.286 | 0.364 (0) | 0.091 | – (–) |
| intern08_ts_replace | 133 | 0.045 | 0.053 | 0.053 (0) | 0.038 | – (–) |

_recall@k: the top-k of System-1's full ranking (tournaments log it as `s1_rank`: finalists, then each round's eliminated options by score); records whose ranking is incomplete at k (older tournament logs with only their finalists) are undefined and left out (count in brackets). Offered k = mean number of tools an assist arm offered System-2 (top-k + finish)._

### Tool arguments by fill kind (H3; §7.4)

| arm | kind | n | delegated | delegation | harmful |
|---|---|---|---|---|---|
| intern_shadow | bounded | 45 | 0 | 0.0% | – |
| intern_shadow | copy | 61 | 0 | 0.0% | – |
| jev_shadow | bounded | 45 | 0 | 0.0% | – |
| jev_shadow | copy | 61 | 0 | 0.0% | – |
| intern_shadow_s2lite | bounded | 45 | 0 | 0.0% | – |
| intern_shadow_s2lite | copy | 62 | 0 | 0.0% | – |
| intern_ts_cascade_tau0.99 | bounded | 1 | 1 | 100.0% | 0.0% |
| intern_ts_cascade_tau0.99 | copy | 1 | 1 | 100.0% | 0.0% |
| intern_ts_cascade_tau0.95 | bounded | 9 | 7 | 77.8% | 0.0% |
| intern_ts_cascade_tau0.95 | copy | 10 | 2 | 20.0% | 0.0% |
| intern_ts_cascade_tau0.95 | free | 2 | 0 | 0.0% | – |
| intern_ts_cascade_tau0.9_s2lite | bounded | 16 | 10 | 62.5% | 0.0% |
| intern_ts_cascade_tau0.9_s2lite | copy | 16 | 1 | 6.2% | – |
| intern_ts_cascade_tau0.9_s2lite | free | 1 | 0 | 0.0% | – |
| intern_ts_cascade_tau0.9 | bounded | 16 | 9 | 56.2% | 0.0% |
| intern_ts_cascade_tau0.9 | copy | 17 | 2 | 11.8% | 0.0% |
| intern_ts_cascade_tau0.9 | free | 1 | 0 | 0.0% | – |
| intern_ts_cascade_tauhat | bounded | 16 | 9 | 56.2% | 0.0% |
| intern_ts_cascade_tauhat | copy | 17 | 1 | 5.9% | – |
| intern_ts_cascade_tauhat | free | 1 | 0 | 0.0% | – |
| intern_ts_cascade_tau0.5 | bounded | 17 | 10 | 58.8% | 0.0% |
| intern_ts_cascade_tau0.5 | copy | 25 | 4 | 16.0% | 0.0% |
| intern_ts_cascade_tau0.5 | free | 4 | 0 | 0.0% | – |
| intern_ts_cascade_tau0.7 | bounded | 16 | 9 | 56.2% | 0.0% |
| intern_ts_cascade_tau0.7 | copy | 21 | 4 | 19.0% | 0.0% |
| intern_ts_cascade_tau0.7 | free | 2 | 0 | 0.0% | – |
| intern_ts_cascade_tau0.8 | bounded | 15 | 9 | 60.0% | 0.0% |
| intern_ts_cascade_tau0.8 | copy | 18 | 3 | 16.7% | 0.0% |
| intern_ts_cascade_tau0.8 | free | 2 | 0 | 0.0% | – |
| intern_ts_params_replace | bounded | 68 | 68 | 100.0% | 91.4% |
| intern_ts_params_replace | copy | 24 | 24 | 100.0% | 0.0% |
| intern_ts_params_replace | free | 60 | 0 | 0.0% | – |

### Sweeps

| sweep | arms | values | APGR | CPT(50%) | CPT(80%) | AIQ | AIQ zero-router | ΔAIQ |
|---|---|---|---|---|---|---|---|---|
| intern_ts_cascade_tau | 6 | 0.5, 0.7, 0.8, 0.9, 0.95, 0.99 | – | – | – | 0.617 | – | – |

_APGR / CPT: x = strong-model calls relative to the baseline, anchors weak (PGR 0) and baseline (PGR 1); AIQ: mean success of the non-decreasing convex hull over the linear cost range between the two references (sweep + references); zero-router: hull of the references alone._

