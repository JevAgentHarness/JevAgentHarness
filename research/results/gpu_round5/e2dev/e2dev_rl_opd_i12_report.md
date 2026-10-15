## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_shadow | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0439 | 0.0000 | 7.7564 | 7.5000 | 0.0000 | 45522.3 | 7.5000 | 0 | 0.0000 | 0.0439 | 0.0708 | 0.4148 |
| ours_stop_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0416 | 0.0000 | 11.3 | 7.5000 | 1.0000 | 45493.0 | 7.5000 | 0 | 0.0024 | 0.0416 | 0.0707 | 0.4522 |
| ours_stop_guard_s2lite | 50 | 0.9400 | 0.8600 | 1.0000 | 0.0098 | 0.0000 | 4.5099 | 7.5400 | 1.0000 | 45412.2 | 7.5400 | 0 | 0.0000 | 0.0098 | 0.0142 | 0.3179 |
| ours_ts_assist_k3 | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0177 | 0.0000 | 13.6 | 7.8400 | 7.8400 | 10432.1 | 7.8400 | 0 | 0.0000 | 0.0177 | 0.0177 | 0.0000 |
| ours_ts_assist_k3_s2lite | 50 | 0.7800 | 0.6600 | 0.8800 | 0.0037 | 0.0000 | 13.4 | 7.8400 | 7.8400 | 10296.0 | 7.8400 | 0 | 0.0000 | 0.0037 | 0.0037 | 0.0000 |
| ours_ts_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0455 | 0.0000 | 8.4170 | 8.4400 | 7.7000 | 50751.0 | 7.7000 | 0 | 0.0000 | 0.0455 | 0.0788 | 0.4786 |
| ours_ts_params_replace * | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0033 | 0.0000 | 16.6 | 1.5000 | 18.1 | 1839.3 | 7.5000 | 0 | 0.0000 | 0.0033 | 0.0033 | 0.0000 |
| ours_ts_replace | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0145 | 0.0000 | 13.1 | 7.5000 | 7.5000 | 8394.8 | 7.5000 | 0 | 0.0000 | 0.0145 | 0.0145 | 0.0000 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### ours_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 375 | 375 | 1.0000 | nan | nan | 1.0000 | nan | nan | 0.0255 | 0.0016 | 1.0000 |
| tool_params/shadow | 531 | 412 | 1.0000 | nan | nan | 1.0000 | 0.8835 | 0.8835 | 0.0243 | 0.0034 | nan |
| tool_select/shadow | 375 | 375 | 1.0000 | 1.0000 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0585 | 0.0084 | nan |

### ours_stop_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 50 | 50 | 1.0000 | nan | 1.0000 | 0.0000 | nan | nan | 0.0810 | 0.0091 | nan |
| tool_select/off | 375 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### ours_stop_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 50 | 50 | 0.9400 | nan | 0.9400 | 0.0000 | nan | nan | 0.0063 | 0.0535 | 0.7163 |
| tool_select/off | 377 | 0 | nan | nan | nan | 0.0000 | 0.9947 | 0.9947 | nan | nan | nan |

### ours_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 392 | 392 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.9566 | 0.9566 | 0.0465 | 0.0071 | nan |

### ours_ts_assist_k3_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 392 | 392 | 1.0000 | 0.9974 | 1.0000 | 0.0000 | 0.9566 | 0.9566 | 0.0386 | 0.0054 | nan |

### ours_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 385 | 385 | 0.9039 | nan | 0.9039 | 0.0000 | nan | nan | 0.1859 | 0.1013 | nan |

### ours_ts_params_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/generate | 75 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_params/replace | 531 | 531 | 1.0000 | nan | 1.0000 | 0.0000 | nan | 1.0000 | 0.0145 | 0.0019 | nan |
| tool_select/replace | 375 | 375 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 1.0000 | 0.0439 | 0.0064 | nan |

### ours_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 375 | 375 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 1.0000 | 0.0443 | 0.0064 | nan |


## Score card

- **Frontier cost-of-pass**: 0.00284 $/success over 7 arms (1 shadow arm(s) left out of the frontier) (best single arm: `ours_ts_params_replace`)
- **Hypervolume** (log-cost × success, normalised): all arms 1, baseline alone –; front: `ours_ts_params_replace`
- note: baseline 'llm_only' not found: baseline-relative metrics omitted
- note: unpriced models (cost counted as 0): jevagentrl-opd_i12

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| ours_ts_params_replace | 100.0 [100.0, 100.0] | – | – | 0.00326 | – | 0.00326 | 16.6 | 92.4 | front |
| ours_ts_replace | 100.0 [100.0, 100.0] | – | – | 0.0145 | – | 0.0145 | 13.1 | 100.0 |  |
| ours_ts_assist_k3 | 100.0 [100.0, 100.0] | – | – | 0.0177 | – | 0.0177 | 13.6 | 0.0 |  |
| ours_stop_guard | 100.0 [100.0, 100.0] | – | – | 0.0416 | – | 0.0416 | 11.3 | 0.0 |  |
| ours_shadow | 100.0 [100.0, 100.0] | – | – | 0.0439 | – | 0.0439 | 7.76 | 0.0 |  |
| ours_ts_guard | 100.0 [100.0, 100.0] | – | – | 0.0455 | – | 0.0455 | 8.42 | 0.0 |  |
| ours_stop_guard_s2lite | 94.0 [86.0, 100.0] | – | – | 0.00976 | – | 0.0104 | 4.51 | 0.0 |  |
| ours_ts_assist_k3_s2lite | 78.0 [66.0, 88.0] | – | – | 0.00366 | – | 0.00470 | 13.4 | 0.0 |  |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

