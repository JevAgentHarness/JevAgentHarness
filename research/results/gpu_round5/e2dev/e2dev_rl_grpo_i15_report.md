## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_shadow | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0435 | 0.0000 | 7.1139 | 7.5000 | 0.0000 | 45464.4 | 7.5000 | 0 | 0.0000 | 0.0435 | 0.0707 | 0.4205 |
| ours_stop_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0425 | 0.0000 | 12.5 | 7.5000 | 1.0000 | 45464.5 | 7.5000 | 0 | 0.0000 | 0.0425 | 0.0707 | 0.4367 |
| ours_stop_guard_s2lite | 50 | 0.9200 | 0.8400 | 0.9800 | 0.0098 | 0.0000 | 4.5784 | 7.5400 | 1.0200 | 45427.8 | 7.5400 | 0 | 0.0000 | 0.0098 | 0.0142 | 0.3136 |
| ours_ts_assist_k3 | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0177 | 0.0000 | 15.1 | 7.7800 | 7.7800 | 10401.6 | 7.7800 | 0 | 0.0013 | 0.0177 | 0.0177 | 0.0000 |
| ours_ts_assist_k3_s2lite | 50 | 0.8400 | 0.7400 | 0.9400 | 0.0037 | 0.0000 | 13.4 | 7.8400 | 7.8400 | 10242.0 | 7.8400 | 0 | 0.0000 | 0.0037 | 0.0037 | 0.0000 |
| ours_ts_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0408 | 0.0000 | 7.5790 | 7.5000 | 7.5000 | 45464.7 | 7.5000 | 0 | 0.0000 | 0.0408 | 0.0707 | 0.4653 |
| ours_ts_params_replace * | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0033 | 0.0000 | 16.6 | 1.5000 | 18.1 | 1875.7 | 7.5000 | 0 | 0.0000 | 0.0033 | 0.0033 | 0.0000 |
| ours_ts_replace | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0145 | 0.0000 | 12.7 | 7.5000 | 7.5000 | 8424.1 | 7.5000 | 0 | 0.0067 | 0.0145 | 0.0145 | 0.0000 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### ours_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 375 | 375 | 1.0000 | nan | nan | 1.0000 | nan | nan | 0.0025 | 0.0000 | 1.0000 |
| tool_params/shadow | 531 | 424 | 0.9976 | nan | nan | 1.0000 | 0.9009 | 0.9009 | 0.0073 | 0.0042 | 0.9645 |
| tool_select/shadow | 375 | 375 | 1.0000 | 0.9760 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0146 | 0.0034 | nan |

### ours_stop_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 50 | 50 | 1.0000 | nan | 1.0000 | 0.0000 | nan | nan | 0.0048 | 0.0000 | nan |
| tool_select/off | 375 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### ours_stop_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 51 | 51 | 0.9216 | nan | 0.9216 | 0.0000 | nan | nan | 0.0762 | 0.0710 | 0.9152 |
| tool_select/off | 377 | 0 | nan | nan | nan | 0.0000 | 0.9973 | 0.9973 | nan | nan | nan |

### ours_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 389 | 389 | 1.0000 | 0.9769 | 1.0000 | 0.0000 | 0.9640 | 0.9640 | 0.0127 | 0.0032 | nan |

### ours_ts_assist_k3_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 392 | 392 | 1.0000 | 0.9796 | 1.0000 | 0.0000 | 0.9337 | 0.9337 | 0.0119 | 0.0032 | nan |

### ours_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 375 | 375 | 1.0000 | nan | 1.0000 | 0.0000 | nan | nan | 0.0034 | 0.0000 | nan |

### ours_ts_params_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/generate | 75 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_params/replace | 531 | 523 | 1.0000 | nan | 1.0000 | 0.0000 | nan | 1.0000 | 0.0006 | 0.0000 | nan |
| tool_select/replace | 375 | 375 | 1.0000 | 0.9840 | 1.0000 | 0.0000 | nan | 1.0000 | 0.0111 | 0.0028 | nan |

### ours_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 375 | 375 | 1.0000 | 0.9840 | 1.0000 | 0.0000 | nan | 1.0000 | 0.0112 | 0.0028 | nan |


## Score card

- **Frontier cost-of-pass**: 0.00292 $/success over 7 arms (1 shadow arm(s) left out of the frontier) (best single arm: `ours_ts_params_replace`)
- **Hypervolume** (log-cost × success, normalised): all arms 1, baseline alone –; front: `ours_ts_params_replace`
- note: baseline 'llm_only' not found: baseline-relative metrics omitted
- note: unpriced models (cost counted as 0): jevagentrl-grpo_i15

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| ours_ts_params_replace | 100.0 [100.0, 100.0] | – | – | 0.00331 | – | 0.00331 | 16.6 | 92.4 | front |
| ours_ts_replace | 100.0 [100.0, 100.0] | – | – | 0.0145 | – | 0.0145 | 12.7 | 100.0 |  |
| ours_ts_assist_k3 | 100.0 [100.0, 100.0] | – | – | 0.0177 | – | 0.0177 | 15.1 | 0.0 |  |
| ours_ts_guard | 100.0 [100.0, 100.0] | – | – | 0.0408 | – | 0.0408 | 7.58 | 0.0 |  |
| ours_stop_guard | 100.0 [100.0, 100.0] | – | – | 0.0425 | – | 0.0425 | 12.5 | 0.0 |  |
| ours_shadow | 100.0 [100.0, 100.0] | – | – | 0.0435 | – | 0.0435 | 7.11 | 0.0 |  |
| ours_stop_guard_s2lite | 92.0 [84.0, 98.0] | – | – | 0.00981 | – | 0.0107 | 4.58 | 0.0 |  |
| ours_ts_assist_k3_s2lite | 84.0 [74.0, 94.0] | – | – | 0.00366 | – | 0.00436 | 13.4 | 0.0 |  |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

