## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_shadow | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0405 | 0.0000 | 7.6105 | 7.5000 | 0.0000 | 45492.9 | 7.5000 | 0 | 0.0000 | 0.0405 | 0.0707 | 0.4744 |
| ours_stop_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0411 | 0.0000 | 8.1483 | 7.5000 | 1.0000 | 45493.5 | 7.5000 | 0 | 0.0000 | 0.0411 | 0.0707 | 0.4653 |
| ours_stop_guard_s2lite | 50 | 0.9000 | 0.8200 | 0.9800 | 0.0098 | 0.0000 | 4.5989 | 7.5400 | 1.0000 | 45412.0 | 7.5400 | 0 | 0.0000 | 0.0098 | 0.0142 | 0.3153 |
| ours_ts_assist_k3 | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0178 | 0.0000 | 13.7 | 7.8000 | 7.8000 | 10447.6 | 7.8000 | 0 | 0.0013 | 0.0178 | 0.0178 | 0.0000 |
| ours_ts_assist_k3_s2lite | 50 | 0.8000 | 0.7000 | 0.9000 | 0.0038 | 0.0000 | 14.0 | 8.1400 | 8.1400 | 10812.9 | 8.1400 | 0 | 0.0025 | 0.0038 | 0.0038 | 0.0000 |
| ours_ts_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0457 | 0.0000 | 9.5755 | 8.4600 | 7.7000 | 50847.4 | 7.7000 | 0 | 0.0000 | 0.0457 | 0.0789 | 0.4740 |
| ours_ts_params_replace * | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0033 | 0.0000 | 16.8 | 1.5000 | 18.1 | 1839.3 | 7.5000 | 0 | 0.0000 | 0.0033 | 0.0033 | 0.0000 |
| ours_ts_replace | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0145 | 0.0000 | 13.0 | 7.5000 | 7.5000 | 8394.4 | 7.5000 | 0 | 0.0000 | 0.0145 | 0.0145 | 0.0000 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### ours_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 375 | 375 | 1.0000 | nan | nan | 1.0000 | nan | nan | 0.0244 | 0.0010 | 1.0000 |
| tool_params/shadow | 531 | 418 | 1.0000 | nan | nan | 1.0000 | 0.8923 | 0.8923 | 0.0206 | 0.0027 | nan |
| tool_select/shadow | 375 | 375 | 1.0000 | 1.0000 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0513 | 0.0079 | nan |

### ours_stop_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 50 | 50 | 1.0000 | nan | 1.0000 | 0.0000 | nan | nan | 0.0598 | 0.0049 | nan |
| tool_select/off | 375 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### ours_stop_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 50 | 50 | 0.9000 | nan | 0.9000 | 0.0000 | nan | nan | 0.0567 | 0.0903 | 0.7867 |
| tool_select/off | 377 | 0 | nan | nan | nan | 0.0000 | 0.9947 | 0.9947 | nan | nan | nan |

### ours_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 390 | 390 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.9615 | 0.9615 | 0.0480 | 0.0081 | nan |

### ours_ts_assist_k3_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 407 | 407 | 0.9926 | 0.9926 | 0.9926 | 0.0000 | 0.9263 | 0.9263 | 0.0400 | 0.0117 | 0.9216 |

### ours_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 385 | 385 | 0.9013 | nan | 0.9013 | 0.0000 | nan | nan | 0.1766 | 0.1039 | nan |

### ours_ts_params_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/generate | 75 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_params/replace | 531 | 531 | 1.0000 | nan | 1.0000 | 0.0000 | nan | 1.0000 | 0.0147 | 0.0015 | nan |
| tool_select/replace | 375 | 375 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 1.0000 | 0.0474 | 0.0083 | nan |

### ours_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 375 | 375 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 1.0000 | 0.0473 | 0.0078 | nan |


## Score card

- **Frontier cost-of-pass**: 0.00284 $/success over 7 arms (1 shadow arm(s) left out of the frontier) (best single arm: `ours_ts_params_replace`)
- **Hypervolume** (log-cost × success, normalised): all arms 1, baseline alone –; front: `ours_ts_params_replace`
- note: baseline 'llm_only' not found: baseline-relative metrics omitted
- note: unpriced models (cost counted as 0): jevagentrl-opd

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| ours_ts_params_replace | 100.0 [100.0, 100.0] | – | – | 0.00326 | – | 0.00326 | 16.8 | 92.4 | front |
| ours_ts_replace | 100.0 [100.0, 100.0] | – | – | 0.0145 | – | 0.0145 | 13.0 | 100.0 |  |
| ours_ts_assist_k3 | 100.0 [100.0, 100.0] | – | – | 0.0178 | – | 0.0178 | 13.7 | 0.0 |  |
| ours_shadow | 100.0 [100.0, 100.0] | – | – | 0.0405 | – | 0.0405 | 7.61 | 0.0 |  |
| ours_stop_guard | 100.0 [100.0, 100.0] | – | – | 0.0411 | – | 0.0411 | 8.15 | 0.0 |  |
| ours_ts_guard | 100.0 [100.0, 100.0] | – | – | 0.0457 | – | 0.0457 | 9.58 | 0.0 |  |
| ours_stop_guard_s2lite | 90.0 [82.0, 98.0] | – | – | 0.00979 | – | 0.0109 | 4.60 | 0.0 |  |
| ours_ts_assist_k3_s2lite | 80.0 [70.0, 90.0] | – | – | 0.00384 | – | 0.00480 | 14.0 | 0.0 |  |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

