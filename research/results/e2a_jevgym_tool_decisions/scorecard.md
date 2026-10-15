## Score card (baseline: `llm_only`)

- **Iso-quality cost savings**: 14.2% [12.7, 15.6] with `jev_ts_cascade_tauhat` (Δsuccess +0.0 pp [0.0, 0.0], p_worse=1.000; non-inferior at margin 3.0 pp, α=0.05; 3 arm(s) qualify)
- **Quality at iso-cost** (≤ 0.0424 $/episode): `jev_ts_cascade_tauhat` 99.8% vs. reference 99.8% (Δ +0.0 pp, p_better=1.000); hull 100.0%
- **Iso-quality cost savings** (vs. `llm_only_s2lite`): 0.2% [-0.2, 0.7] with `jev_ts_cascade_tau0.9_s2lite` (Δsuccess +1.0 pp [-1.0, 3.2], p_worse=0.857; non-inferior at margin 3.0 pp, α=0.05; 2 arm(s) qualify)
- **Quality at iso-cost** (vs. `llm_only_s2lite`) (≤ 0.0098 $/episode): `jev_ts_cascade_tau0.9_s2lite` 81.2% vs. reference 80.2% (Δ +1.0 pp, p_better=0.195); hull 85.4%
- **Frontier cost-of-pass**: 0.00626 $/success over 25 arms (2 shadow arm(s) left out of the frontier) (best single arm: `jev_ts_assist_k3_s2lite`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.707, baseline alone 0.133; front: `bm25_ts_assist_k3_s2lite`, `jev_ts_assist_k3`, `jev_ts_assist_k3_s2lite`, `jev_ts_cascade_tau0.9_s2lite`, `jev_ts_cascade_tauhat`, `jev_ts_guard`, `jev_ts_guard_s2lite`, `jev_ts_params_replace`
- **Sweep `jev_ts_cascade_tau*`** (6 arms): AIQ 0.842
- **Sweep `lite_ts_cascade_tau*`** (3 arms): AIQ 0.441
- note: per-System-2 references: `llm_only_s2lite` (gemini-3.5-flash-lite) for 5 arm(s); every other arm vs. `llm_only`

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | McNemar p | lat save % | LoP s | LLM-call ratio | strong-call ratio | PGR | EuB | HV contrib | LLM-free steps % | cache-mod $ | no-cache $ | cached % | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 99.8 [99.4, 100.0] | – | – | 0.0424 | – | 0.0425 | 8.93 | 0.0 | – | – | 8.95 | 1.00 | 1.00 | – | 0.499 | 0.000 | 0.0 | 0.0424 | 0.0708 | 43.9 | baseline |
| jev_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0430 | -0.2 | 0.0430 | 11.0 | 0.0 | 1.000 | -17.9 | 11.0 | 1.02 | 1.02 | – | 0.493 | 0.000 | 0.0 | 0.0430 | 0.0727 | 45.5 | front,iso-q |
| lite_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0536 | -24.9 | 0.0536 | 15.3 | 0.0 | 1.000 | -63.9 | 15.3 | 1.18 | 1.18 | – | 0.372 | 0.000 | 0.0 | 0.0536 | 0.0868 | 43.6 | iso-q |
| jev_ts_cascade_tauhat | 99.8 [99.4, 100.0] | +0.0 | 1.000 | 0.0364 | 14.2 | 0.0365 | 9.98 | 40.6 | 1.000 | -11.7 | 10.0 | 0.85 | 0.85 | – | 0.570 | 0.003 | 14.6 | 0.0364 | 0.0542 | 37.3 | front,iso-best |
| jev_shadow | 99.8 [99.4, 100.0] | +0.0 | 1.000 | 0.0408 | 3.7 | 0.0409 | 10.7 | 0.0 | 1.000 | -19.8 | 10.7 | 1.00 | 1.00 | – | 0.518 | – | 0.0 | 0.0408 | 0.0708 | 47.1 |  |
| jev_ts_assist_k3 | 92.0 [89.4, 94.2] | -7.8 | <0.001 | 0.0196 | 53.7 | 0.0213 | 10.3 | 0.0 | <0.001 | -15.3 | 11.2 | 1.08 | 1.08 | – | 0.708 | 0.003 | 0.0 | 0.0196 | 0.0196 | 0.0 | front,worse |
| jev_ts_cascade_tau0.99 | 92.0 [88.0, 95.5] | -8.0 | <0.001 | 0.0261 | 39.3 | 0.0283 | 11.1 | 61.9 | <0.001 | -18.9 | 12.0 | 0.65 | 0.65 | – | 0.619 | 0.000 | 33.4 | 0.0261 | 0.0403 | 40.0 | worse |
| jev_ts_assist_k5 | 90.0 [85.5, 94.0] | -10.0 | <0.001 | 0.0222 | 48.3 | 0.0247 | 9.47 | 0.0 | <0.001 | -1.8 | 10.5 | 1.11 | 1.11 | – | 0.669 | 0.000 | 0.0 | 0.0222 | 0.0222 | 0.0 | worse |
| lite_ts_assist_k3 | 87.5 [83.0, 92.0] | -12.5 | <0.001 | 0.0367 | 14.6 | 0.0419 | 27.0 | 0.0 | <0.001 | -190.0 | 30.9 | 1.00 | 1.00 | – | 0.494 | 0.000 | 0.0 | 0.0367 | 0.0367 | 0.0 | worse |
| jev_ts_guard_s2lite | 86.5 [81.5, 91.0] | +4.5 | 0.978 | 0.0104 | -6.7 | 0.0120 | 8.18 | 0.0 | 0.078 | -54.0 | 9.46 | 1.03 | 1.03 | – | 0.764 | 0.010 | 0.0 | 0.0104 | 0.0151 | 32.2 | front,iso-q,vs llm_only_s2lite |
| jev_shadow_s2lite | 81.2 [78.0, 84.4] | +1.0 | 0.839 | 0.00976 | 0.4 | 0.0120 | 5.93 | 0.0 | 0.472 | -10.3 | 7.31 | 1.00 | 1.00 | – | 0.724 | – | 0.0 | 0.00976 | 0.0143 | 32.5 | vs llm_only_s2lite |
| jev_ts_cascade_tau0.9_s2lite | 81.2 [77.8, 84.6] | +1.0 | 0.857 | 0.00978 | 0.2 | 0.0120 | 6.00 | 0.0 | 0.458 | -11.6 | 7.39 | 1.00 | 1.00 | – | 0.724 | 0.000 | 0.0 | 0.00978 | 0.0142 | 32.1 | front,iso-best,vs llm_only_s2lite |
| jev_ts_replace | 81.0 [75.5, 86.0] | -19.0 | <0.001 | 0.0142 | 67.0 | 0.0175 | 8.34 | 100.0 | <0.001 | 10.4 | 10.3 | 0.91 | 0.91 | – | 0.648 | 0.000 | 0.0 | 0.0142 | 0.0142 | 0.0 | worse |
| llm_only_s2lite | 80.2 [76.6, 83.8] | -19.6 | <0.001 | 0.00980 | 76.9 | 0.0122 | 5.38 | 0.0 | <0.001 | 39.8 | 6.71 | 1.01 | 1.01 | – | 0.715 | 0.000 | 0.0 | 0.00980 | 0.0143 | 32.2 | worse |
| jev_ts_params_replace | 80.0 [74.5, 85.5] | -20.0 | <0.001 | 0.00509 | 88.1 | 0.00637 | 6.74 | 91.1 | <0.001 | 27.6 | 8.43 | 0.24 | 0.24 | – | 0.749 | 0.004 | 76.3 | 0.00509 | 0.00509 | 0.0 | front,worse |
| jev_ts_cascade_tau0.9 | 79.5 [73.5, 85.0] | -20.5 | <0.001 | 0.0120 | 72.0 | 0.0151 | 7.83 | 83.3 | <0.001 | 15.9 | 9.84 | 0.36 | 0.36 | – | 0.668 | 0.000 | 61.9 | 0.0120 | 0.0153 | 26.9 | worse |
| jev_ts_cascade_tau0.95 | 79.5 [74.0, 85.0] | -20.5 | <0.001 | 0.0143 | 66.8 | 0.0179 | 7.57 | 81.0 | <0.001 | 18.6 | 9.53 | 0.39 | 0.39 | – | 0.642 | 0.000 | 58.3 | 0.0143 | 0.0197 | 32.0 | worse |
| jev_ts_cascade_tau0.5 | 78.5 [73.0, 84.0] | -21.5 | <0.001 | 0.00664 | 84.5 | 0.00845 | 6.51 | 86.5 | <0.001 | 30.1 | 8.29 | 0.31 | 0.31 | – | 0.721 | 0.000 | 67.5 | 0.00664 | 0.00718 | 4.3 | worse |
| jev_ts_assist_k3_s2lite | 76.8 [73.2, 80.6] | -3.4 | 0.056 | 0.00470 | 52.0 | 0.00613 | 8.94 | 0.0 | 0.115 | -66.2 | 11.6 | 1.06 | 1.06 | – | 0.728 | 0.015 | 0.0 | 0.00470 | 0.00470 | 0.0 | front,vs llm_only_s2lite |
| jev_ts_cascade_tau0.8 | 76.5 [70.5, 82.0] | -23.5 | <0.001 | 0.0101 | 76.5 | 0.0132 | 5.95 | 84.0 | <0.001 | 36.1 | 7.77 | 0.35 | 0.35 | – | 0.667 | 0.000 | 62.6 | 0.0101 | 0.0124 | 21.2 | worse |
| jev_ts_cascade_tau0.7 | 76.0 [70.0, 81.5] | -24.0 | <0.001 | 0.00818 | 81.0 | 0.0108 | 7.70 | 85.6 | <0.001 | 17.3 | 10.1 | 0.31 | 0.31 | – | 0.682 | 0.000 | 65.9 | 0.00818 | 0.00967 | 15.7 | worse |
| lite_ts_cascade_tau0.99 | 75.0 [69.0, 81.0] | -25.0 | <0.001 | 0.0432 | -0.7 | 0.0577 | 28.3 | 65.5 | <0.001 | -204.3 | 37.8 | 0.57 | 0.57 | – | 0.326 | 0.000 | 38.1 | 0.0432 | 0.0570 | 40.8 | worse |
| lite_ts_cascade_tau0.9 | 43.5 [37.0, 50.5] | -56.5 | <0.001 | 0.0263 | 38.8 | 0.0604 | 25.6 | 90.0 | <0.001 | -175.1 | 58.9 | 0.19 | 0.19 | – | 0.262 | 0.000 | 76.5 | 0.0263 | 0.0266 | 3.1 | worse |
| lite_ts_cascade_tau0.7 | 42.5 [36.0, 49.0] | -57.5 | <0.001 | 0.0267 | 37.9 | 0.0627 | 26.9 | 90.1 | <0.001 | -188.8 | 63.3 | 0.19 | 0.19 | – | 0.257 | 0.000 | 76.5 | 0.0267 | 0.0267 | 0.1 | worse |
| lite_ts_params_replace | 39.5 [33.0, 46.5] | -60.5 | <0.001 | 0.0271 | 36.9 | 0.0686 | 26.4 | 91.3 | <0.001 | -183.3 | 66.8 | 0.19 | 0.19 | – | 0.230 | 0.000 | 77.1 | 0.0271 | 0.0271 | 0.0 | worse |
| embed_ts_assist_k3 | 28.0 [21.5, 34.0] | -72.0 | <0.001 | 0.00924 | 78.5 | 0.0330 | 6.00 | 0.0 | <0.001 | 35.5 | 21.4 | 0.64 | 0.64 | – | 0.216 | 0.000 | 0.0 | 0.00924 | 0.00924 | 0.0 | worse |
| bm25_ts_assist_k3_s2lite | 12.5 [8.0, 17.0] | -69.5 | <0.001 | 0.00205 | 79.0 | 0.0164 | 4.26 | 0.0 | <0.001 | 19.8 | 34.1 | 0.76 | 0.76 | – | 0.124 | 0.030 | 0.0 | 0.00205 | 0.00205 | 0.0 | front,worse,vs llm_only_s2lite |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

_cost $: the billed (ledger) cost, the primary figure; cache-mod $: System-2 calls re-priced from their logged tokens with the measured cached tokens at the model's cached rate; no-cache $: list price with no cache; cached %: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md)._

### Non-inferiority (A6 procedure; margin 3.0 pp)

| arm | reference | n | Δ pp | 95% LCB pp | p (rule) | Holm p | α_adj | Tango LB(1−α_adj) pp | discordant (w/l) | exact p (loss UB pp) | MDE Δ pp (80%) | p_boot | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_ts_guard | llm_only | 200 | +0.0 | +0.0 | 0.002 (bb_ordered) | 0.045 | 0.0026 | -3.7 | 0 (0/0) | 0.002 (2.93) | – | <0.001 | NI |
| lite_ts_guard | llm_only | 200 | +0.0 | +0.0 | 0.002 (bb_ordered) | 0.045 | 0.0028 | -3.7 | 0 (0/0) | 0.002 (2.90) | – | <0.001 | NI |
| jev_ts_cascade_tauhat | llm_only | 500 | +0.0 | +0.0 | <0.001 (bb_ordered) | 0.002 | 0.0025 | -1.6 | 0 (0/0) | <0.001 (1.19) | – | <0.001 | NI |
| jev_ts_assist_k3 | llm_only | 500 | -7.8 | -9.8 | 1.000 (bb_ordered) | 1.000 | 0.0031 | -11.8 | 41 (1/40) | – | +1.6 | 1.000 | not NI |
| jev_ts_cascade_tau0.99 | llm_only | 200 | -8.0 | -11.5 | 1.000 (bb_ordered) | 1.000 | 0.0056 | -14.3 | 16 (0/16) | 1.000 (14.15) | +3.7 | 1.000 | not NI |
| jev_ts_assist_k5 | llm_only | 200 | -10.0 | -13.5 | 1.000 (bb_ordered) | 1.000 | 0.0033 | -17.2 | 20 (0/20) | 1.000 (17.08) | +4.9 | 1.000 | not NI |
| lite_ts_assist_k3 | llm_only | 200 | -12.5 | -16.5 | 1.000 (bb_ordered) | 1.000 | 0.0083 | -19.2 | 25 (0/25) | 1.000 (19.14) | +5.0 | 1.000 | not NI |
| jev_ts_guard_s2lite | llm_only_s2lite | 200 | +4.5 | +1.0 | 0.001 (bb_ordered) | 0.003 | 0.0167 | -0.4 | 21 (15/6) | – | +3.8 | <0.001 | NI |
| jev_ts_cascade_tau0.9_s2lite | llm_only_s2lite | 500 | +1.0 | -0.8 | <0.001 (bb_ordered) | 0.002 | 0.0125 | -1.5 | 29 (17/12) | – | +0.3 | <0.001 | NI |
| jev_ts_replace | llm_only | 200 | -19.0 | -23.5 | 1.000 (bb_ordered) | 1.000 | 0.0071 | -26.7 | 38 (0/38) | 1.000 (26.70) | +7.1 | 1.000 | not NI |
| llm_only_s2lite | llm_only | 500 | -19.6 | -22.8 | 1.000 (bb_ordered) | 1.000 | 0.0500 | -22.7 | 100 (1/99) | – | +2.0 | 1.000 | not NI |
| jev_ts_params_replace | llm_only | 200 | -20.0 | -25.0 | 1.000 (bb_ordered) | 1.000 | 0.0063 | -27.9 | 40 (0/40) | 1.000 (27.96) | +7.5 | 1.000 | not NI |
| jev_ts_cascade_tau0.9 | llm_only | 200 | -20.5 | -25.5 | 1.000 (bb_ordered) | 1.000 | 0.0045 | -28.9 | 41 (0/41) | 1.000 (28.88) | +8.0 | 1.000 | not NI |
| jev_ts_cascade_tau0.95 | llm_only | 200 | -20.5 | -25.5 | 1.000 (bb_ordered) | 1.000 | 0.0050 | -28.7 | 41 (0/41) | 1.000 (28.77) | +7.9 | 1.000 | not NI |
| jev_ts_cascade_tau0.5 | llm_only | 200 | -21.5 | -26.5 | 1.000 (bb_ordered) | 1.000 | 0.0036 | -30.2 | 43 (0/43) | 1.000 (30.25) | +8.5 | 1.000 | not NI |
| jev_ts_assist_k3_s2lite | llm_only_s2lite | 500 | -3.4 | -6.6 | 0.591 (bb_ordered) | 1.000 | 0.0250 | -7.4 | 103 (43/60) | – | +2.7 | 0.603 | inconclusive at this n |
| jev_ts_cascade_tau0.8 | llm_only | 200 | -23.5 | -29.0 | 1.000 (bb_ordered) | 1.000 | 0.0042 | -32.2 | 47 (0/47) | 1.000 (32.27) | +8.9 | 1.000 | not NI |
| jev_ts_cascade_tau0.7 | llm_only | 200 | -24.0 | -29.5 | 1.000 (bb_ordered) | 1.000 | 0.0038 | -32.9 | 48 (0/48) | 1.000 (32.90) | +9.1 | 1.000 | not NI |
| lite_ts_cascade_tau0.99 | llm_only | 200 | -25.0 | -30.0 | 1.000 (bb_ordered) | 1.000 | 0.0167 | -32.0 | 50 (0/50) | 1.000 (32.16) | +7.5 | 1.000 | not NI |
| lite_ts_cascade_tau0.9 | llm_only | 200 | -56.5 | -62.5 | 1.000 (bb_ordered) | 1.000 | 0.0125 | -64.1 | 113 (0/113) | 1.000 (64.42) | +13.3 | 1.000 | not NI |
| lite_ts_cascade_tau0.7 | llm_only | 200 | -57.5 | -63.5 | 1.000 (bb_ordered) | 1.000 | 0.0100 | -65.3 | 115 (0/115) | 1.000 (65.66) | +13.9 | 1.000 | not NI |
| lite_ts_params_replace | llm_only | 200 | -60.5 | -66.5 | 1.000 (bb_ordered) | 1.000 | 0.0250 | -67.0 | 121 (0/121) | 1.000 (67.32) | +12.3 | 1.000 | not NI |
| embed_ts_assist_k3 | llm_only | 200 | -72.0 | -77.0 | 1.000 (bb_ordered) | 1.000 | 0.0029 | -79.8 | 144 (0/144) | 1.000 (80.30) | +18.4 | 1.000 | not NI |
| bm25_ts_assist_k3_s2lite | llm_only_s2lite | 200 | -69.5 | -74.5 | 1.000 (bb_ordered) | 1.000 | 0.0500 | -74.6 | 139 (0/139) | 1.000 (74.86) | +11.6 | 1.000 | not NI |

_A6.14: p_T = 1 − Φ(Z(−m)), Tango's one-sided score test of H0: Δ ≤ −m on the paired (wins, losses, n) counts, enters Holm with the family (every acting arm vs. the same reference); NI holds iff p_T ≤ α_adj of its Holm step, equivalently iff the Tango (1 − α_adj) lower bound is above −m. With zero wins or < 5 discordant tasks the exact binomial on losses (p = P[L ≤ l | n, m]) decides instead (`rule` = exact; it is the p that enters Holm, `Holm p`) and the one-sided 95% upper bound on the loss rate (1 − 0.05^(1/n) at zero losses) replaces a zero-width interval. A primary that fails while its two-sided 95% CI covers 0 is *inconclusive at this n*, not inferior; MDE: the smallest true Δ shown NI with 80% power at the achieved discordance and α_adj. p: the certified Berger-Boos p-value of one ordering of all (wins, losses) outcomes (A6.21; rule bb_ordered). Tango LB: descriptive score interval._

### System-1 decisions

| arm | n scored | S1 acc | ECE | ACE | Brier | AUROC | AURC | E-AURC | acc@80%cov | harmful % | guard FAR | guard FRR | rescue % | yes/no n | yes/no ECE | yes/no Brier | yes/no AUROC | mem stored % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_ts_guard | 0 | – | – | – | – | – | – | – | – | – | 0.000 | 0.020 | 100.0 | 1501 | 0.129 | 0.029 | 0.999 | – |
| lite_ts_guard | 0 | – | – | – | – | – | – | – | – | – | 0.211 | 0.102 | 78.9 | 1603 | 0.105 | 0.102 | 0.840 | – |
| jev_ts_cascade_tauhat | 5530 | 0.948 | 0.054 | 0.057 | 0.056 | 0.768 | 0.017 | 0.015 | 0.969 | 0.0 | – | – | – | 0 | – | – | – | – |
| jev_shadow | 7952 | 0.935 | 0.049 | 0.048 | 0.063 | 0.795 | 0.018 | 0.016 | 0.957 | – | – | – | – | 3756 | 0.048 | 0.009 | 1.000 | – |
| jev_ts_assist_k3 | 4052 | 0.896 | 0.051 | 0.047 | 0.079 | 0.766 | 0.046 | 0.040 | 0.947 | – | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tau0.99 | 2507 | 0.956 | 0.045 | 0.050 | 0.047 | 0.792 | 0.012 | 0.011 | 0.973 | 1.0 | – | – | – | 0 | – | – | – | – |
| jev_ts_assist_k5 | 1674 | 0.876 | 0.044 | 0.040 | 0.086 | 0.808 | 0.045 | 0.037 | 0.945 | – | – | – | – | 0 | – | – | – | – |
| lite_ts_assist_k3 | 1508 | 0.861 | 0.126 | 0.126 | 0.132 | 0.674 | 0.086 | 0.075 | 0.887 | – | – | – | – | 0 | – | – | – | – |
| jev_ts_guard_s2lite | 0 | – | – | – | – | – | – | – | – | – | 0.357 | 0.013 | 23.8 | 1514 | 0.094 | 0.027 | 0.974 | – |
| jev_shadow_s2lite | 8719 | 0.972 | 0.029 | 0.031 | 0.029 | 0.789 | 0.009 | 0.009 | 0.983 | – | – | – | – | 3786 | 0.034 | 0.009 | 0.999 | – |
| jev_ts_replace | 1373 | 0.954 | 0.030 | 0.039 | 0.041 | 0.872 | 0.008 | 0.007 | 0.988 | 2.8 | – | – | – | 0 | – | – | – | – |
| jev_ts_params_replace | 3235 | 0.943 | 0.024 | 0.020 | 0.045 | 0.892 | 0.009 | 0.007 | 0.983 | 3.5 | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tau0.9 | 3018 | 0.972 | 0.027 | 0.028 | 0.029 | 0.848 | 0.005 | 0.005 | 0.986 | 1.6 | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tau0.95 | 2952 | 0.971 | 0.029 | 0.031 | 0.030 | 0.847 | 0.006 | 0.005 | 0.984 | 1.7 | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tau0.5 | 3106 | 0.958 | 0.025 | 0.020 | 0.037 | 0.887 | 0.006 | 0.005 | 0.987 | 2.9 | – | – | – | 0 | – | – | – | – |
| jev_ts_assist_k3_s2lite | 4021 | 0.913 | 0.065 | 0.064 | 0.082 | 0.659 | 0.054 | 0.050 | 0.929 | – | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tau0.8 | 3007 | 0.967 | 0.033 | 0.029 | 0.034 | 0.849 | 0.006 | 0.006 | 0.985 | 2.2 | – | – | – | 0 | – | – | – | – |
| jev_ts_cascade_tau0.7 | 3016 | 0.968 | 0.034 | 0.030 | 0.035 | 0.838 | 0.006 | 0.006 | 0.985 | 2.3 | – | – | – | 0 | – | – | – | – |
| lite_ts_cascade_tau0.99 | 2192 | 0.873 | 0.113 | 0.113 | 0.116 | 0.661 | 0.076 | 0.067 | 0.899 | 4.9 | – | – | – | 0 | – | – | – | – |
| lite_ts_cascade_tau0.9 | 2465 | 0.890 | 0.095 | 0.095 | 0.098 | 0.658 | 0.070 | 0.064 | 0.923 | 8.5 | – | – | – | 0 | – | – | – | – |
| lite_ts_cascade_tau0.7 | 2492 | 0.874 | 0.110 | 0.110 | 0.113 | 0.651 | 0.083 | 0.074 | 0.911 | 10.5 | – | – | – | 0 | – | – | – | – |
| lite_ts_params_replace | 2541 | 0.859 | 0.127 | 0.121 | 0.128 | 0.616 | 0.104 | 0.093 | 0.893 | 12.0 | – | – | – | 0 | – | – | – | – |
| embed_ts_assist_k3 | 964 | 0.128 | 0.153 | 0.133 | 0.121 | 0.659 | 0.776 | 0.166 | 0.140 | – | – | – | – | 0 | – | – | – | – |
| bm25_ts_assist_k3_s2lite | 1132 | 0.169 | 0.780 | 0.778 | 0.760 | 0.328 | 0.920 | 0.389 | 0.181 | – | – | – | – | 0 | – | – | – | – |

_Calibration over System-1 Choice decisions with gold labels (top-label confidence); yes/no: guard / trigger decisions calibrated as forecasts P(yes) of the binary gold; harmful: delegated decisions that were wrong in failed episodes (per delegated decision); FAR/FRR: guard false accept / false reject rates; rescue: wrong LLM proposals caught by the guard in episodes that succeeded._

### Delegation coverage and native LLM calls (A6.22 (a))

| arm | tool cov. % | arg cov. % (S1 / slots) | arg cov. in S1 calls % | LLM-free steps % | LLM calls / ep. | calls avoided / ep. | avoided % | reference |
|---|---|---|---|---|---|---|---|---|
| jev_ts_cascade_tauhat | 27.2 | 20.8 (1264/6072) | 67.7 | 14.6 | 6.42 | +1.09 | 14.6 | llm_only |
| jev_ts_cascade_tau0.99 | 45.1 | 41.1 (979/2381) | 82.8 | 33.4 | 4.89 | +2.61 | 34.8 | llm_only |
| jev_ts_cascade_tau0.9_s2lite | 0.0 | 0.0 (0/6103) | – | 0.0 | 7.57 | +0.01 | 0.2 | llm_only_s2lite |
| jev_ts_replace | 100.0 | 0.0 (0/2258) | – | 0.0 | 6.87 | +0.64 | 8.5 | llm_only |
| jev_ts_params_replace | 100.0 | 85.7 (2166/2527) | 85.7 | 76.3 | 1.80 | +5.71 | 76.0 | llm_only |
| jev_ts_cascade_tau0.9 | 81.3 | 73.1 (1668/2281) | 84.6 | 61.9 | 2.67 | +4.83 | 64.4 | llm_only |
| jev_ts_cascade_tau0.95 | 74.3 | 68.7 (1581/2302) | 86.1 | 58.3 | 2.96 | +4.54 | 60.6 | llm_only |
| jev_ts_cascade_tau0.5 | 97.3 | 78.1 (1835/2349) | 79.8 | 67.5 | 2.31 | +5.20 | 69.2 | llm_only |
| jev_ts_cascade_tau0.8 | 86.5 | 74.6 (1697/2274) | 82.3 | 62.6 | 2.61 | +4.89 | 65.2 | llm_only |
| jev_ts_cascade_tau0.7 | 91.3 | 77.3 (1742/2255) | 82.0 | 65.9 | 2.35 | +5.15 | 68.7 | llm_only |
| lite_ts_cascade_tau0.99 | 50.0 | 43.7 (984/2251) | 84.0 | 38.1 | 4.31 | +3.20 | 42.6 | llm_only |
| lite_ts_cascade_tau0.9 | 98.3 | 83.7 (1639/1958) | 84.9 | 76.5 | 1.42 | +6.09 | 81.1 | llm_only |
| lite_ts_cascade_tau0.7 | 99.9 | 84.0 (1682/2003) | 84.1 | 76.5 | 1.45 | +6.05 | 80.7 | llm_only |
| lite_ts_params_replace | 100.0 | 85.9 (1791/2084) | 85.9 | 77.1 | 1.46 | +6.04 | 80.5 | llm_only |

_tool cov.: tool choices executed from System-1 per tool-selection opportunity; arg cov.: argument values System-1 chose per argument slot of the executed calls (a deferred tool choice leaves the whole call, and its slots, to the LLM); in S1 calls: the same share inside System-1-chosen calls (– when it chose none: the argument thresholds were never reached); calls avoided: paired per-task mean of the reference's minus the arm's System-2 calls (trajectory length included)._

### Meta-decision golds: coverage

| arm | dp/mode | n | labelled | coverage | doomed | unobserved | no_oracle | cf_error |
|---|---|---|---|---|---|---|---|---|
| llm_only | tool_select/off | 3756 | 3756 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_guard | tool_select/guard | 1501 | 1501 | 100.0% | 0 | 0 | 0 | 0 |
| lite_ts_guard | tool_select/guard | 1603 | 1603 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tauhat | tool_select/cascade | 3756 | 3756 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tauhat | tool_params/cascade | 1774 | 1774 | 100.0% | 0 | 0 | 0 | 0 |
| jev_shadow | tool_select/shadow | 3756 | 3756 | 100.0% | 0 | 0 | 0 | 0 |
| jev_shadow | stop/shadow | 3756 | 3756 | 100.0% | 0 | 0 | 0 | 0 |
| jev_shadow | tool_params/shadow | 4196 | 4196 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_assist_k3 | tool_select/assist | 4052 | 4052 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tau0.99 | tool_select/cascade | 1471 | 1471 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tau0.99 | tool_params/cascade | 1036 | 1036 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_assist_k5 | tool_select/assist | 1674 | 1674 | 100.0% | 0 | 0 | 0 | 0 |
| lite_ts_assist_k3 | tool_select/assist | 1508 | 1508 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_guard_s2lite | tool_select/guard | 1514 | 1514 | 100.0% | 0 | 0 | 0 | 0 |
| jev_shadow_s2lite | tool_select/shadow | 3787 | 3787 | 100.0% | 0 | 0 | 0 | 0 |
| jev_shadow_s2lite | stop/shadow | 3787 | 3787 | 100.0% | 0 | 0 | 0 | 0 |
| jev_shadow_s2lite | tool_params/shadow | 4932 | 4932 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tau0.9_s2lite | tool_select/cascade | 3783 | 3783 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_replace | tool_select/replace | 1373 | 1373 | 100.0% | 0 | 0 | 0 | 0 |
| llm_only_s2lite | tool_select/off | 3789 | 3789 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_params_replace | tool_select/replace | 1516 | 1498 | 98.8% | 18 | 0 | 0 | 0 |
| jev_ts_params_replace | tool_params/replace | 1737 | 1737 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tau0.9 | tool_select/cascade | 1404 | 1404 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tau0.9 | tool_params/cascade | 1614 | 1614 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tau0.95 | tool_select/cascade | 1418 | 1418 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tau0.95 | tool_params/cascade | 1534 | 1534 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tau0.5 | tool_select/cascade | 1422 | 1413 | 99.4% | 9 | 0 | 0 | 0 |
| jev_ts_cascade_tau0.5 | tool_params/cascade | 1693 | 1693 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_assist_k3_s2lite | tool_select/assist | 4021 | 4021 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tau0.8 | tool_select/cascade | 1396 | 1396 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tau0.8 | tool_params/cascade | 1611 | 1611 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tau0.7 | tool_select/cascade | 1379 | 1379 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_cascade_tau0.7 | tool_params/cascade | 1637 | 1637 | 100.0% | 0 | 0 | 0 | 0 |
| lite_ts_cascade_tau0.99 | tool_select/cascade | 1393 | 1393 | 100.0% | 0 | 0 | 0 | 0 |
| lite_ts_cascade_tau0.99 | tool_params/cascade | 799 | 799 | 100.0% | 0 | 0 | 0 | 0 |
| lite_ts_cascade_tau0.9 | tool_select/cascade | 1202 | 1202 | 100.0% | 0 | 0 | 0 | 0 |
| lite_ts_cascade_tau0.9 | tool_params/cascade | 1263 | 1263 | 100.0% | 0 | 0 | 0 | 0 |
| lite_ts_cascade_tau0.7 | tool_select/cascade | 1236 | 1227 | 99.3% | 9 | 0 | 0 | 0 |
| lite_ts_cascade_tau0.7 | tool_params/cascade | 1265 | 1265 | 100.0% | 0 | 0 | 0 | 0 |
| lite_ts_params_replace | tool_select/replace | 1274 | 1261 | 99.0% | 13 | 0 | 0 | 0 |
| lite_ts_params_replace | tool_params/replace | 1280 | 1280 | 100.0% | 0 | 0 | 0 | 0 |
| embed_ts_assist_k3 | tool_select/assist | 964 | 964 | 100.0% | 0 | 0 | 0 | 0 |
| bm25_ts_assist_k3_s2lite | tool_select/assist | 1149 | 1132 | 98.5% | 17 | 0 | 0 | 0 |

### Gates vs. their primary gold (H2 inputs: measurement arms)

| arm | gate/mode | n | n+ / n− | π | AUROC | τ | J(τ) | acc(τ) | fire rate | ECE | H2 cell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_shadow | stop_guard/shadow | 499 | 0 / 499 | 0.000 | – | 0.50 | – | 0.926 | 0.074 | 0.171 | no opportunity |
| jev_shadow | stop_trigger/shadow | 3257 | 1 / 3256 | 0.000 | 1.000 | 0.50 | 1.000 | 1.000 | 0.000 | 0.029 | no opportunity |
| jev_shadow_s2lite | stop_guard/shadow | 499 | 94 / 405 | 0.188 | 0.969 | 0.50 | 0.590 | 0.906 | 0.138 | 0.077 | estimable |
| jev_shadow_s2lite | stop_trigger/shadow | 3287 | 0 / 3287 | 0.000 | – | 0.50 | – | 1.000 | 0.000 | 0.028 | no opportunity |

_Shadow records of measurement arms (every opportunity labelled), the arm's own System-1; other backends: `meta_gold replay` + `meta_gold h2`. Primary golds: reflection / routing = the System-2 action at the state is failing (need_fatal); stop guard = finishing now with the proposed answer succeeds; stop trigger = only the answer remains; memory_write = the offered text holds a value the remaining plan consumes; tool guard = the proposal is progress. A guard fires (rejects) on p_yes < τ, so J = 1 − FAR − FRR. H2 cell: ≥ 30 positive and ≥ 30 negative golds._

### Live gates vs. their gold (consistency check; not H2 inputs)

| arm | gate/mode | opportunities (retries) | labelled share | n+ / n− labelled | AUROC (observed labels) | J(τ) | realised fire rate |
|---|---|---|---|---|---|---|---|
| jev_ts_guard | tool_select/guard | 1501 (0) | 100.0% | 1 / 1500 | 0.999 | 0.980 | 0.021 |
| lite_ts_guard | tool_select/guard | 1603 (0) | 100.0% | 19 / 1584 | 0.840 | 0.687 | 0.110 |
| jev_ts_guard_s2lite | tool_select/guard | 1514 (0) | 100.0% | 42 / 1472 | 0.974 | 0.630 | 0.030 |

_Live trigger arms label a reflection / routing opportunity only where the gate did not fire (selective labels): AUROC is computed on the labels observed at executed actions, J / accuracy / estimability are not reported, and these rows never enter H2 (META_GOLD_SPEC §9.2, §11.5–6). The realised fire rate counts every opportunity once (a stop-guard resubmission of a rejected answer is a retry)._

### Tool-selection recall@k vs. the acceptable tool set (C3)

| arm | n | recall@1 | recall@3 | recall@5 (undefined) | recall@3 (canonical) | offered recall (offered k) |
|---|---|---|---|---|---|---|
| jev_ts_cascade_tauhat | 3756 | 0.942 | 0.971 | 0.973 (0) | 0.942 | – (–) |
| jev_shadow | 3756 | 0.941 | 0.972 | 0.973 (0) | 0.942 | – (–) |
| jev_ts_assist_k3 | 4052 | 0.896 | 0.946 | 0.965 (0) | 0.919 | 0.964 (3.8) |
| jev_ts_cascade_tau0.99 | 1471 | 0.942 | 0.973 | 0.975 (0) | 0.943 | – (–) |
| jev_ts_assist_k5 | 1674 | 0.876 | 0.938 | 0.962 (0) | 0.912 | 0.968 (5.8) |
| lite_ts_assist_k3 | 1508 | 0.861 | 0.893 | 0.896 (0) | 0.804 | 0.894 (3.7) |
| jev_shadow_s2lite | 3787 | 0.936 | 0.965 | 0.967 (0) | 0.936 | – (–) |
| jev_ts_replace | 1373 | 0.954 | 0.991 | 0.993 (0) | 0.946 | – (–) |
| jev_ts_params_replace | 1498 | 0.899 | 0.988 | 0.989 (0) | 0.947 | – (–) |
| jev_ts_cascade_tau0.9 | 1404 | 0.945 | 0.974 | 0.976 (0) | 0.934 | – (–) |
| jev_ts_cascade_tau0.95 | 1418 | 0.941 | 0.972 | 0.975 (0) | 0.941 | – (–) |
| jev_ts_cascade_tau0.5 | 1413 | 0.926 | 0.986 | 0.987 (0) | 0.940 | – (–) |
| jev_ts_assist_k3_s2lite | 4021 | 0.913 | 0.948 | 0.953 (0) | 0.921 | 0.948 (3.8) |
| jev_ts_cascade_tau0.8 | 1396 | 0.944 | 0.974 | 0.977 (0) | 0.914 | – (–) |
| jev_ts_cascade_tau0.7 | 1379 | 0.949 | 0.978 | 0.980 (0) | 0.922 | – (–) |
| lite_ts_cascade_tau0.99 | 1393 | 0.848 | 0.881 | 0.884 (0) | 0.786 | – (–) |
| lite_ts_cascade_tau0.9 | 1202 | 0.820 | 0.861 | 0.863 (0) | 0.775 | – (–) |
| lite_ts_cascade_tau0.7 | 1227 | 0.805 | 0.843 | 0.846 (0) | 0.759 | – (–) |
| lite_ts_params_replace | 1261 | 0.799 | 0.835 | 0.837 (0) | 0.754 | – (–) |
| embed_ts_assist_k3 | 964 | 0.128 | 0.254 | 0.344 (0) | 0.212 | 0.309 (4.0) |
| bm25_ts_assist_k3_s2lite | 1132 | 0.169 | 0.188 | 0.223 (0) | 0.138 | 0.210 (4.0) |

_recall@k: the top-k of System-1's full ranking (tournaments log it as `s1_rank`: finalists, then each round's eliminated options by score); records whose ranking is incomplete at k (older tournament logs with only their finalists) are undefined and left out (count in brackets). Offered k = mean number of tools an assist arm offered System-2 (top-k + finish)._

### Tool arguments by fill kind (H3; §7.4)

| arm | kind | n | delegated | delegation | harmful |
|---|---|---|---|---|---|
| jev_ts_cascade_tauhat | bounded | 820 | 673 | 82.1% | 0.0% |
| jev_ts_cascade_tauhat | copy | 982 | 591 | 60.2% | 0.0% |
| jev_ts_cascade_tauhat | free | 64 | 0 | 0.0% | – |
| jev_shadow | bounded | 2252 | 0 | 0.0% | – |
| jev_shadow | copy | 3066 | 0 | 0.0% | – |
| jev_ts_cascade_tau0.99 | bounded | 499 | 461 | 92.4% | 0.0% |
| jev_ts_cascade_tau0.99 | copy | 639 | 518 | 81.1% | 0.0% |
| jev_ts_cascade_tau0.99 | free | 45 | 0 | 0.0% | – |
| jev_shadow_s2lite | bounded | 2259 | 0 | 0.0% | – |
| jev_shadow_s2lite | copy | 3096 | 0 | 0.0% | – |
| jev_ts_params_replace | bounded | 986 | 986 | 100.0% | 4.4% |
| jev_ts_params_replace | copy | 1183 | 1180 | 99.7% | 0.0% |
| jev_ts_params_replace | free | 358 | 0 | 0.0% | – |
| jev_ts_cascade_tau0.9 | bounded | 804 | 772 | 96.0% | 0.1% |
| jev_ts_cascade_tau0.9 | copy | 1015 | 896 | 88.3% | 0.0% |
| jev_ts_cascade_tau0.9 | free | 152 | 0 | 0.0% | – |
| jev_ts_cascade_tau0.95 | bounded | 757 | 727 | 96.0% | 0.0% |
| jev_ts_cascade_tau0.95 | copy | 974 | 854 | 87.7% | 0.0% |
| jev_ts_cascade_tau0.95 | free | 105 | 0 | 0.0% | – |
| jev_ts_cascade_tau0.5 | bounded | 892 | 848 | 95.1% | 2.5% |
| jev_ts_cascade_tau0.5 | copy | 1113 | 987 | 88.7% | 0.0% |
| jev_ts_cascade_tau0.5 | free | 295 | 0 | 0.0% | – |
| jev_ts_cascade_tau0.8 | bounded | 831 | 796 | 95.8% | 1.9% |
| jev_ts_cascade_tau0.8 | copy | 1023 | 901 | 88.1% | 0.0% |
| jev_ts_cascade_tau0.8 | free | 209 | 0 | 0.0% | – |
| jev_ts_cascade_tau0.7 | bounded | 841 | 806 | 95.8% | 2.5% |
| jev_ts_cascade_tau0.7 | copy | 1060 | 936 | 88.3% | 0.0% |
| jev_ts_cascade_tau0.7 | free | 224 | 0 | 0.0% | – |
| lite_ts_cascade_tau0.99 | bounded | 460 | 422 | 91.7% | 0.3% |
| lite_ts_cascade_tau0.99 | copy | 576 | 562 | 97.6% | 5.7% |
| lite_ts_cascade_tau0.99 | free | 136 | 0 | 0.0% | – |
| lite_ts_cascade_tau0.9 | bounded | 720 | 650 | 90.3% | 1.8% |
| lite_ts_cascade_tau0.9 | copy | 1021 | 989 | 96.9% | 1.8% |
| lite_ts_cascade_tau0.9 | free | 190 | 0 | 0.0% | – |
| lite_ts_cascade_tau0.7 | bounded | 737 | 669 | 90.8% | 5.2% |
| lite_ts_cascade_tau0.7 | copy | 1047 | 1013 | 96.8% | 1.9% |
| lite_ts_cascade_tau0.7 | free | 217 | 0 | 0.0% | – |
| lite_ts_params_replace | bounded | 783 | 783 | 100.0% | 10.2% |
| lite_ts_params_replace | copy | 1036 | 1008 | 97.3% | 2.0% |
| lite_ts_params_replace | free | 265 | 0 | 0.0% | – |

### Sweeps

| sweep | arms | values | APGR | CPT(50%) | CPT(80%) | AIQ | AIQ zero-router | ΔAIQ |
|---|---|---|---|---|---|---|---|---|
| jev_ts_cascade_tau | 6 | 0.5, 0.7, 0.8, 0.9, 0.95, 0.99 | – | – | – | 0.842 | – | – |
| lite_ts_cascade_tau | 3 | 0.7, 0.9, 0.99 | – | – | – | 0.441 | – | – |

_APGR / CPT: x = strong-model calls relative to the baseline, anchors weak (PGR 0) and baseline (PGR 1); AIQ: mean success of the non-decreasing convex hull over the linear cost range between the two references (sweep + references); zero-router: hull of the references alone._

