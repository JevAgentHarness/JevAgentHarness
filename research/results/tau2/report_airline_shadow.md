## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_guard | 45 | 0.6889 | 0.5333 | 0.8222 | 0.0672 | 0.0001 | 21.1 | 11.0 | 0.7778 | 71371.1 | 10.9 | 0 | 0.0000 | 0.0672 | 0.1153 | 0.4646 |
| jev_shadow | 45 | 0.6444 | 0.5111 | 0.7778 | 0.0664 | 0.0000 | 12.0 | 10.8 | 0.0000 | 69928.4 | 10.8 | 0 | 0.0000 | 0.0664 | 0.1127 | 0.4635 |
| jev_ts_assist_k3 * | 45 | 0.7556 | 0.6222 | 0.8667 | 0.0855 | 0.0017 | 31.0 | 12.9 | 12.9 | 59423.4 | 12.9 | 0 | 0.0000 | 0.0855 | 0.0991 | 0.0638 |
| jev_ts_cascade_tau0.5 | 45 | 0.5111 | 0.3556 | 0.6444 | 0.1568 | 0.0040 | 26.8 | 16.6 | 26.8 | 217871.2 | 20.1 | 0 | 0.0000 | 0.1568 | 0.3395 | 0.2123 |
| jev_ts_cascade_tau0.7 | 45 | 0.6444 | 0.5111 | 0.7778 | 0.0730 | 0.0018 | 15.1 | 8.9778 | 14.9 | 55721.1 | 10.5 | 0 | 0.0000 | 0.0730 | 0.0921 | 0.2090 |
| jev_ts_cascade_tau0.8 | 45 | 0.6444 | 0.5111 | 0.7778 | 0.0738 | 0.0017 | 16.3 | 9.6222 | 14.0 | 60856.2 | 10.9 | 0 | 0.0000 | 0.0738 | 0.1001 | 0.2934 |
| jev_ts_cascade_tau0.9 | 45 | 0.6444 | 0.5111 | 0.7778 | 0.0731 | 0.0016 | 16.8 | 10.0 | 13.1 | 65508.5 | 10.9 | 0 | 0.0000 | 0.0731 | 0.1071 | 0.3439 |
| jev_ts_cascade_tau0.95 | 45 | 0.7111 | 0.5778 | 0.8444 | 0.0955 | 0.0016 | 24.3 | 10.5 | 12.7 | 91266.9 | 10.9 | 0 | 0.0000 | 0.0955 | 0.1591 | 0.4007 |
| jev_ts_cascade_tau0.99 * | 45 | 0.6889 | 0.5556 | 0.8222 | 0.0612 | 0.0012 | 15.3 | 9.6444 | 10.5 | 59509.0 | 9.8444 | 0 | 0.0000 | 0.0612 | 0.0979 | 0.4265 |
| jev_ts_cascade_tauhat_e2 | 45 | 0.6444 | 0.5111 | 0.7778 | 0.0658 | 0.0014 | 14.8 | 10.3 | 11.3 | 63959.4 | 10.4 | 0 | 0.0000 | 0.0658 | 0.1046 | 0.4227 |
| jev_ts_cascade_tauhat_refit | 45 | 0.6000 | 0.4444 | 0.7333 | 0.1007 | 0.0024 | 19.7 | 12.1 | 19.7 | 74204.6 | 13.3 | 0 | 0.0000 | 0.1007 | 0.1213 | 0.0877 |
| llm_only | 45 | 0.6667 | 0.5333 | 0.8000 | 0.0686 | 0.0000 | 12.3 | 11.3 | 0.0000 | 73875.8 | 11.3 | 0 | nan | 0.0686 | 0.1190 | 0.4734 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### jev_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| policy/guard | 27 | 27 | 0.7037 | nan | 0.7037 | 0.0000 | nan | nan | nan | nan | nan |
| stop/guard | 8 | 8 | 0.0000 | nan | 0.0000 | 0.0000 | nan | nan | nan | nan | nan |
| tool_select/off | 489 | 0 | nan | nan | nan | 0.0000 | 0.4990 | 0.4990 | nan | nan | nan |

### jev_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 12 | 12 | 0.1667 | nan | nan | 1.0000 | nan | nan | 0.6733 | 0.4831 | nan |
| tool_params/shadow | 422 | 115 | 0.6783 | nan | nan | 1.0000 | 0.8348 | 0.8348 | 0.1790 | 0.1937 | 0.8444 |
| tool_select/shadow | 488 | 488 | 0.6598 | nan | nan | 1.0000 | 0.5082 | 0.5082 | 0.1408 | 0.2400 | 0.6046 |

### jev_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 579 | 579 | 0.7219 | nan | 0.7219 | 0.0000 | 0.4041 | 0.4041 | 0.1128 | 0.2016 | 0.6080 |

### jev_ts_cascade_tau0.5

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 302 | 105 | 0.6667 | nan | 0.7263 | 0.1391 | nan | 0.7263 | 0.1864 | 0.1797 | 0.8910 |
| tool_select/cascade | 905 | 905 | 0.6530 | nan | 0.5684 | 0.5878 | 0.2086 | 0.3525 | 0.1758 | 0.2415 | 0.5405 |

### jev_ts_cascade_tau0.7

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 196 | 85 | 0.8000 | nan | 0.9254 | 0.2959 | nan | 0.9254 | 0.1068 | 0.0993 | 0.9048 |
| tool_select/cascade | 474 | 474 | 0.6477 | nan | 0.6934 | 0.3945 | 0.3194 | 0.5401 | 0.1405 | 0.2401 | 0.6245 |

### jev_ts_cascade_tau0.8

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 138 | 65 | 0.8462 | nan | 0.9216 | 0.2391 | nan | 0.9216 | 0.1411 | 0.1172 | 0.8655 |
| tool_select/cascade | 490 | 490 | 0.6510 | nan | 0.7244 | 0.5408 | 0.3820 | 0.5367 | 0.1338 | 0.2401 | 0.6097 |

### jev_ts_cascade_tau0.9

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 100 | 49 | 0.9184 | nan | 0.9778 | 0.1100 | nan | 0.9778 | 0.0431 | 0.0480 | 0.8500 |
| tool_select/cascade | 489 | 489 | 0.6380 | nan | 0.7338 | 0.6851 | 0.3839 | 0.4928 | 0.1561 | 0.2496 | 0.6091 |

### jev_ts_cascade_tau0.95

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 80 | 46 | 0.9130 | nan | 0.9744 | 0.1375 | nan | 0.9744 | 0.0585 | 0.0619 | 0.7857 |
| tool_select/cascade | 492 | 492 | 0.6443 | nan | 0.7714 | 0.7866 | 0.4496 | 0.5183 | 0.1602 | 0.2445 | 0.6338 |

### jev_ts_cascade_tau0.99

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 30 | 8 | 0.8750 | nan | 1.0000 | 0.0667 | nan | 1.0000 | 0.1213 | 0.1201 | 1.0000 |
| tool_select/cascade | 443 | 443 | 0.6591 | nan | 0.6757 | 0.9165 | 0.5123 | 0.5260 | 0.1424 | 0.2431 | 0.5871 |

### jev_ts_cascade_tauhat_e2

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 38 | 5 | 1.0000 | nan | 1.0000 | 0.0263 | nan | 1.0000 | 0.0040 | 0.0000 | nan |
| tool_select/cascade | 470 | 470 | 0.6596 | nan | 0.6486 | 0.9213 | 0.5081 | 0.5191 | 0.1440 | 0.2376 | 0.6063 |

### jev_ts_cascade_tauhat_refit

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 289 | 89 | 0.7753 | nan | 1.0000 | 0.6540 | nan | 1.0000 | 0.1106 | 0.1392 | 0.8598 |
| tool_select/cascade | 599 | 599 | 0.6694 | nan | 0.5873 | 0.3689 | 0.0268 | 0.3756 | 0.1013 | 0.2223 | 0.6222 |

### llm_only

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/off | 508 | 0 | nan | nan | nan | 0.0000 | 0.4980 | 0.4980 | nan | nan | nan |


## Score card (baseline: `llm_only`)

- **Iso-quality cost savings**: 10.8% [0.6, 19.1] with `jev_ts_cascade_tau0.99` (Δsuccess +2.2 pp [-4.4, 8.9], p_worse=0.702; not significantly worse, one-sided paired bootstrap, α=0.05; 10 arm(s) qualify)
- **Quality at iso-cost** (≤ 0.0686 $/episode): `jev_ts_cascade_tau0.99` 68.9% vs. reference 66.7% (Δ +2.2 pp, p_better=0.412); hull 70.9%
- **Frontier cost-of-pass**: ∞ $/success over 12 arms (best single arm: `jev_ts_cascade_tau0.99`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.737, baseline alone 0.602; front: `jev_ts_assist_k3`, `jev_ts_cascade_tau0.99`
- **Sweep `jev_ts_cascade_tau*`** (6 arms): AIQ 0.707
- note: several trials per task: the NI rows, Holm and the `noninferior` iso-quality qualifier use majority pass (> half of a task's trials succeed; PREREGISTRATION A6.14 / A6.16 (e)); Delta and its CI are per-attempt (pass^1) means

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| llm_only | 66.7 [44.4, 86.7] | – | – | 0.0686 | – | 0.103 | 12.3 | 0.0 | baseline |
| jev_ts_assist_k3 | 75.6 [55.6, 93.3] | +8.9 | 0.966 | 0.0855 | -24.5 | 0.113 | 31.0 | 0.0 | front,iso-q |
| jev_ts_cascade_tau0.95 | 71.1 [51.1, 88.9] | +4.4 | 1.000 | 0.0955 | -39.2 | 0.134 | 24.3 | 30.4 | iso-q |
| jev_ts_cascade_tau0.99 | 68.9 [48.9, 86.7] | +2.2 | 0.702 | 0.0612 | 10.8 | 0.0889 | 15.3 | 13.7 | front,iso-best |
| jev_guard | 68.9 [46.7, 88.9] | +2.2 | 0.742 | 0.0672 | 2.0 | 0.0976 | 21.1 | 0.0 | iso-q |
| jev_ts_cascade_tauhat_e2 | 64.4 [42.2, 86.7] | -2.2 | 0.393 | 0.0658 | 4.2 | 0.102 | 14.8 | 14.6 | iso-q |
| jev_shadow | 64.4 [40.0, 86.7] | -2.2 | 0.388 | 0.0664 | 3.2 | 0.103 | 12.0 | 0.0 | iso-q |
| jev_ts_cascade_tau0.7 | 64.4 [42.2, 86.7] | -2.2 | 0.391 | 0.0730 | -6.4 | 0.113 | 15.1 | 63.4 | iso-q |
| jev_ts_cascade_tau0.9 | 64.4 [40.0, 86.7] | -2.2 | 0.384 | 0.0731 | -6.5 | 0.113 | 16.8 | 41.3 | iso-q |
| jev_ts_cascade_tau0.8 | 64.4 [44.4, 84.4] | -2.2 | 0.389 | 0.0738 | -7.6 | 0.115 | 16.3 | 52.5 | iso-q |
| jev_ts_cascade_tauhat_refit | 60.0 [37.8, 80.0] | -6.7 | 0.236 | 0.101 | -46.8 | 0.168 | 19.7 | 53.8 | iso-q |
| jev_ts_cascade_tau0.5 | 51.1 [31.1, 71.1] | -15.6 | 0.004 | 0.157 | -128.5 | 0.307 | 26.8 | 52.4 | worse |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

