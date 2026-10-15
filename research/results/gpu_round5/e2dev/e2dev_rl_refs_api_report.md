## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_shadow * | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0404 | 0.0000 | 5.9488 | 7.5000 | 0.0000 | 45493.2 | 7.5000 | 0 | 0.0000 | 0.0404 | 0.0707 | 0.4786 |
| jev_stop_guard | 50 | 0.9800 | 0.9400 | 1.0000 | 0.0441 | 0.0001 | 7.6359 | 7.9800 | 1.4800 | 49117.9 | 7.9800 | 0 | 0.0000 | 0.0441 | 0.0768 | 0.4758 |
| jev_stop_guard_s2lite | 50 | 0.8800 | 0.7800 | 0.9600 | 0.0116 | 0.0001 | 5.8140 | 8.7200 | 2.1000 | 54930.9 | 8.7200 | 0 | 0.0000 | 0.0116 | 0.0172 | 0.3189 |
| jev_ts_assist_k3 * | 50 | 0.9200 | 0.8400 | 0.9800 | 0.0190 | 0.0009 | 6.2034 | 7.9400 | 7.9400 | 10627.4 | 7.9400 | 0 | 0.0000 | 0.0190 | 0.0190 | 0.0000 |
| jev_ts_assist_k3_s2lite * | 50 | 0.8200 | 0.7000 | 0.9200 | 0.0047 | 0.0009 | 5.7427 | 8.0400 | 8.0400 | 10705.2 | 8.0400 | 0 | 0.0000 | 0.0047 | 0.0047 | 0.0000 |
| jev_ts_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0417 | 0.0004 | 7.1969 | 7.6600 | 7.5000 | 46510.5 | 7.5000 | 0 | 0.0000 | 0.0417 | 0.0727 | 0.4788 |
| jev_ts_params_replace * | 50 | 0.8000 | 0.6800 | 0.9000 | 0.0041 | 0.0014 | 3.7799 | 1.3200 | 17.4 | 1533.9 | 7.1000 | 0 | 0.0000 | 0.0041 | 0.0041 | 0.0000 |
| jev_ts_replace | 50 | 0.8000 | 0.6800 | 0.9000 | 0.0139 | 0.0008 | 5.1773 | 6.7200 | 6.7200 | 7639.3 | 6.7200 | 0 | 0.0000 | 0.0139 | 0.0139 | 0.0000 |
| llm_only | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0429 | 0.0000 | 5.7168 | 7.5000 | 0.0000 | 45463.9 | 7.5000 | 0 | 0.0000 | 0.0429 | 0.0707 | 0.4294 |
| llm_only_s2lite * | 50 | 0.8800 | 0.8000 | 0.9600 | 0.0098 | 0.0000 | 4.9000 | 7.5200 | 0.0000 | 45284.3 | 7.5200 | 0 | 0.0000 | 0.0098 | 0.0142 | 0.3134 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### jev_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 375 | 375 | 0.9893 | nan | nan | 1.0000 | nan | nan | 0.0487 | 0.0093 | 1.0000 |
| tool_params/shadow | 531 | 418 | 0.9163 | nan | nan | 1.0000 | 0.8923 | 0.8923 | 0.0430 | 0.0681 | 0.8872 |
| tool_select/shadow | 375 | 375 | 0.9493 | 0.8773 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0867 | 0.0665 | 0.6130 |

### jev_stop_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 74 | 74 | 0.6622 | nan | 0.6622 | 0.0000 | nan | nan | 0.3484 | 0.2129 | nan |
| tool_select/off | 399 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### jev_stop_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 105 | 105 | 0.8381 | nan | 0.8381 | 0.0000 | nan | nan | 0.1258 | 0.1323 | 0.8929 |
| tool_select/off | 436 | 0 | nan | nan | nan | 0.0000 | 0.9862 | 0.9862 | nan | nan | nan |

### jev_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 397 | 397 | 0.9093 | 0.8463 | 0.9093 | 0.0000 | 0.9194 | 0.9194 | 0.0707 | 0.0775 | 0.7256 |

### jev_ts_assist_k3_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 402 | 402 | 0.9179 | 0.8607 | 0.9179 | 0.0000 | 0.9303 | 0.9303 | 0.0716 | 0.0834 | 0.6553 |

### jev_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 375 | 375 | 0.9787 | nan | 0.9787 | 0.0000 | nan | nan | 0.1295 | 0.0283 | nan |

### jev_ts_params_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/generate | 66 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_params/replace | 516 | 443 | 0.9752 | nan | 0.9752 | 0.0000 | nan | 0.9752 | 0.0283 | 0.0278 | 0.7391 |
| tool_select/replace | 355 | 355 | 0.9690 | 0.8620 | 0.9690 | 0.0000 | nan | 0.9690 | 0.0848 | 0.0549 | 0.6605 |

### jev_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 336 | 336 | 0.9702 | 0.8988 | 0.9702 | 0.0000 | nan | 0.9702 | 0.0387 | 0.0376 | 0.8035 |

### llm_only

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/off | 375 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### llm_only_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/off | 376 | 0 | nan | nan | nan | 0.0000 | 0.9973 | 0.9973 | nan | nan | nan |


## Score card (baseline: `llm_only`)

- **Iso-quality cost savings**: 2.8% [1.1, 4.4] with `jev_ts_guard` (Δsuccess +0.0 pp [0.0, 0.0], p_worse=1.000; not significantly worse, one-sided paired bootstrap, α=0.05; 2 arm(s) qualify)
- **Quality at iso-cost** (≤ 0.0429 $/episode): `jev_ts_guard` 100.0% vs. reference 100.0% (Δ +0.0 pp, p_better=1.000); hull 100.0%
- **Iso-quality cost savings** (vs. `llm_only_s2lite`): 51.9% [49.3, 54.6] with `jev_ts_assist_k3_s2lite` (Δsuccess -6.0 pp [-18.0, 6.0], p_worse=0.180; not significantly worse, one-sided paired bootstrap, α=0.05; 2 arm(s) qualify)
- **Quality at iso-cost** (vs. `llm_only_s2lite`) (≤ 0.00977 $/episode): `llm_only_s2lite` 88.0% vs. reference 88.0% (Δ +0.0 pp, p_better=–); hull 88.0%
- **Frontier cost-of-pass**: 0.00423 $/success over 9 arms (1 shadow arm(s) left out of the frontier) (best single arm: `jev_ts_params_replace`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.884, baseline alone 0.099; front: `jev_ts_assist_k3`, `jev_ts_assist_k3_s2lite`, `jev_ts_guard`, `jev_ts_params_replace`, `llm_only_s2lite`
- note: per-System-2 references: `llm_only_s2lite` (gemini-3.5-flash-lite) for 2 arm(s); every other arm vs. `llm_only`

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| llm_only | 100.0 [100.0, 100.0] | – | – | 0.0429 | – | 0.0429 | 5.72 | 0.0 | baseline |
| jev_shadow | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0404 | 5.9 | 0.0404 | 5.95 | 0.0 |  |
| jev_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0417 | 2.8 | 0.0417 | 7.20 | 0.0 | front,iso-best |
| jev_stop_guard | 98.0 [94.0, 100.0] | -2.0 | 0.390 | 0.0441 | -2.8 | 0.0450 | 7.64 | 0.0 | iso-q |
| jev_ts_assist_k3 | 92.0 [84.0, 98.0] | -8.0 | 0.014 | 0.0190 | 55.8 | 0.0206 | 6.20 | 0.0 | front,worse |
| llm_only_s2lite | 88.0 [80.0, 96.0] | -12.0 | 0.002 | 0.00977 | 77.2 | 0.0111 | 4.90 | 0.0 | front,worse |
| jev_stop_guard_s2lite | 88.0 [78.0, 96.0] | +0.0 | 0.601 | 0.0116 | -18.3 | 0.0131 | 5.81 | 0.0 | iso-q,vs llm_only_s2lite |
| jev_ts_assist_k3_s2lite | 82.0 [70.0, 92.0] | -6.0 | 0.180 | 0.00470 | 51.9 | 0.00573 | 5.74 | 0.0 | front,iso-best,vs llm_only_s2lite |
| jev_ts_params_replace | 80.0 [68.0, 90.0] | -20.0 | <0.001 | 0.00412 | 90.4 | 0.00515 | 3.78 | 93.0 | front,worse |
| jev_ts_replace | 80.0 [68.0, 90.0] | -20.0 | <0.001 | 0.0139 | 67.7 | 0.0173 | 5.18 | 100.0 | worse |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

