## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_shadow | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0407 | 0.0000 | 9.0782 | 7.5000 | 0.0000 | 45493.5 | 7.5000 | 0 | 0.0000 | 0.0407 | 0.0707 | 0.4688 |
| ours_stop_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0407 | 0.0000 | 7.8639 | 7.5000 | 1.0000 | 45463.6 | 7.5000 | 0 | 0.0071 | 0.0407 | 0.0707 | 0.4690 |
| ours_stop_guard_s2lite | 50 | 0.9200 | 0.8400 | 0.9800 | 0.0098 | 0.0000 | 5.1118 | 7.5400 | 1.0000 | 45415.2 | 7.5400 | 0 | 0.0000 | 0.0098 | 0.0142 | 0.3133 |
| ours_ts_assist_k3 | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0174 | 0.0000 | 13.3 | 7.5400 | 7.5400 | 10308.8 | 7.5400 | 0 | 0.0000 | 0.0174 | 0.0174 | 0.0000 |
| ours_ts_assist_k3_s2lite | 50 | 0.8200 | 0.7000 | 0.9200 | 0.0034 | 0.0000 | 12.6 | 7.3400 | 7.3400 | 9609.4 | 7.3400 | 0 | 0.0014 | 0.0034 | 0.0034 | 0.0000 |
| ours_ts_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0418 | 0.0000 | 8.5021 | 7.5000 | 7.5000 | 45492.6 | 7.5000 | 0 | 0.0000 | 0.0418 | 0.0707 | 0.4505 |
| ours_ts_params_replace * | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0033 | 0.0000 | 16.7 | 1.5000 | 18.1 | 1889.1 | 7.5000 | 0 | 0.0020 | 0.0033 | 0.0033 | 0.0000 |
| ours_ts_replace | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0145 | 0.0000 | 13.2 | 7.5000 | 7.5000 | 8435.0 | 7.5000 | 0 | 0.0080 | 0.0145 | 0.0145 | 0.0000 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### ours_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 375 | 375 | 1.0000 | nan | nan | 1.0000 | nan | nan | 0.0002 | 0.0000 | 1.0000 |
| tool_params/shadow | 531 | 418 | 0.9904 | nan | nan | 1.0000 | 0.8923 | 0.8923 | 0.0089 | 0.0074 | 0.9758 |
| tool_select/shadow | 375 | 375 | 1.0000 | 0.9653 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0083 | 0.0019 | nan |

### ours_stop_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 50 | 50 | 1.0000 | nan | 1.0000 | 0.0000 | nan | nan | 0.0005 | 0.0000 | nan |
| tool_select/off | 375 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### ours_stop_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 50 | 50 | 0.9200 | nan | 0.9200 | 0.0000 | nan | nan | 0.0779 | 0.0770 | 0.8750 |
| tool_select/off | 377 | 0 | nan | nan | nan | 0.0000 | 0.9947 | 0.9947 | nan | nan | nan |

### ours_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 377 | 377 | 1.0000 | 0.9655 | 1.0000 | 0.0000 | 0.9947 | 0.9947 | 0.0074 | 0.0018 | nan |

### ours_ts_assist_k3_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 367 | 367 | 1.0000 | 0.9700 | 1.0000 | 0.0000 | 0.9755 | 0.9755 | 0.0077 | 0.0020 | nan |

### ours_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 375 | 375 | 1.0000 | nan | 1.0000 | 0.0000 | nan | nan | 0.0005 | 0.0000 | nan |

### ours_ts_params_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/generate | 75 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_params/replace | 531 | 519 | 1.0000 | nan | 1.0000 | 0.0000 | nan | 1.0000 | 0.0001 | 0.0000 | nan |
| tool_select/replace | 375 | 375 | 1.0000 | 0.9787 | 1.0000 | 0.0000 | nan | 1.0000 | 0.0051 | 0.0013 | nan |

### ours_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 375 | 375 | 1.0000 | 0.9787 | 1.0000 | 0.0000 | nan | 1.0000 | 0.0052 | 0.0014 | nan |


## Score card

- **Frontier cost-of-pass**: 0.00275 $/success over 7 arms (1 shadow arm(s) left out of the frontier) (best single arm: `ours_ts_params_replace`)
- **Hypervolume** (log-cost × success, normalised): all arms 1, baseline alone –; front: `ours_ts_params_replace`
- note: baseline 'llm_only' not found: baseline-relative metrics omitted
- note: unpriced models (cost counted as 0): jevagentrl-grpo

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| ours_ts_params_replace | 100.0 [100.0, 100.0] | – | – | 0.00333 | – | 0.00333 | 16.7 | 92.4 | front |
| ours_ts_replace | 100.0 [100.0, 100.0] | – | – | 0.0145 | – | 0.0145 | 13.2 | 100.0 |  |
| ours_ts_assist_k3 | 100.0 [100.0, 100.0] | – | – | 0.0174 | – | 0.0174 | 13.3 | 0.0 |  |
| ours_stop_guard | 100.0 [100.0, 100.0] | – | – | 0.0407 | – | 0.0407 | 7.86 | 0.0 |  |
| ours_shadow | 100.0 [100.0, 100.0] | – | – | 0.0407 | – | 0.0407 | 9.08 | 0.0 |  |
| ours_ts_guard | 100.0 [100.0, 100.0] | – | – | 0.0418 | – | 0.0418 | 8.50 | 0.0 |  |
| ours_stop_guard_s2lite | 92.0 [84.0, 98.0] | – | – | 0.00981 | – | 0.0107 | 5.11 | 0.0 |  |
| ours_ts_assist_k3_s2lite | 82.0 [70.0, 92.0] | – | – | 0.00340 | – | 0.00415 | 12.6 | 0.0 |  |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

