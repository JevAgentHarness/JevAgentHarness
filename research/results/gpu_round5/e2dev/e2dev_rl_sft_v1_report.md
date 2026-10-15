## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_shadow | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0432 | 0.0000 | 9.6964 | 7.5000 | 0.0000 | 45463.8 | 7.5000 | 0 | 0.0018 | 0.0432 | 0.0707 | 0.4241 |
| ours_stop_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0406 | 0.0000 | 14.2 | 7.5000 | 1.0000 | 45521.7 | 7.5000 | 0 | 0.0000 | 0.0406 | 0.0708 | 0.4728 |
| ours_stop_guard_s2lite | 50 | 0.9200 | 0.8400 | 0.9800 | 0.0100 | 0.0000 | 4.7749 | 7.7000 | 1.1000 | 46614.6 | 7.7000 | 0 | 0.0000 | 0.0100 | 0.0146 | 0.3133 |
| ours_ts_assist_k3 | 50 | 0.9800 | 0.9400 | 1.0000 | 0.0179 | 0.0000 | 13.6 | 7.8400 | 7.8400 | 10513.9 | 7.8400 | 0 | 0.0000 | 0.0179 | 0.0179 | 0.0000 |
| ours_ts_assist_k3_s2lite | 50 | 0.8400 | 0.7400 | 0.9400 | 0.0037 | 0.0000 | 13.6 | 7.9000 | 7.9000 | 10436.6 | 7.9000 | 0 | 0.0000 | 0.0037 | 0.0037 | 0.0000 |
| ours_ts_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0409 | 0.0000 | 13.9 | 7.5000 | 7.5000 | 45494.0 | 7.5000 | 0 | 0.0027 | 0.0409 | 0.0707 | 0.4662 |
| ours_ts_params_replace * | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0033 | 0.0000 | 16.6 | 1.5000 | 18.1 | 1839.2 | 7.5000 | 0 | 0.0000 | 0.0033 | 0.0033 | 0.0000 |
| ours_ts_replace | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0145 | 0.0000 | 12.9 | 7.5000 | 7.5000 | 8394.4 | 7.5000 | 0 | 0.0013 | 0.0145 | 0.0145 | 0.0000 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### ours_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 375 | 375 | 1.0000 | nan | nan | 1.0000 | nan | nan | 0.0258 | 0.0020 | 1.0000 |
| tool_params/shadow | 531 | 424 | 0.9953 | nan | nan | 1.0000 | 0.9009 | 0.9009 | 0.0256 | 0.0054 | 0.9976 |
| tool_select/shadow | 375 | 375 | 1.0000 | 1.0000 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0442 | 0.0054 | nan |

### ours_stop_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 50 | 50 | 1.0000 | nan | 1.0000 | 0.0000 | nan | nan | 0.0928 | 0.0134 | nan |
| tool_select/off | 375 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### ours_stop_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 55 | 55 | 0.9091 | nan | 0.9091 | 0.0000 | nan | nan | 0.1034 | 0.0827 | 0.8936 |
| tool_select/off | 385 | 0 | nan | nan | nan | 0.0000 | 0.9870 | 0.9870 | nan | nan | nan |

### ours_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 392 | 392 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.9541 | 0.9541 | 0.0386 | 0.0047 | nan |

### ours_ts_assist_k3_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 395 | 395 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.9494 | 0.9494 | 0.0345 | 0.0042 | nan |

### ours_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 375 | 375 | 1.0000 | nan | 1.0000 | 0.0000 | nan | nan | 0.0882 | 0.0113 | nan |

### ours_ts_params_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/generate | 75 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_params/replace | 531 | 531 | 1.0000 | nan | 1.0000 | 0.0000 | nan | 1.0000 | 0.0147 | 0.0017 | nan |
| tool_select/replace | 375 | 375 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 1.0000 | 0.0384 | 0.0050 | nan |

### ours_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 375 | 375 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | nan | 1.0000 | 0.0385 | 0.0050 | nan |


## Score card

- **Frontier cost-of-pass**: 0.00288 $/success over 7 arms (1 shadow arm(s) left out of the frontier) (best single arm: `ours_ts_params_replace`)
- **Hypervolume** (log-cost × success, normalised): all arms 1, baseline alone –; front: `ours_ts_params_replace`
- note: baseline 'llm_only' not found: baseline-relative metrics omitted
- note: unpriced models (cost counted as 0): jevagentrl-sft_v1

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| ours_ts_params_replace | 100.0 [100.0, 100.0] | – | – | 0.00326 | – | 0.00326 | 16.6 | 92.4 | front |
| ours_ts_replace | 100.0 [100.0, 100.0] | – | – | 0.0145 | – | 0.0145 | 12.9 | 100.0 |  |
| ours_stop_guard | 100.0 [100.0, 100.0] | – | – | 0.0406 | – | 0.0406 | 14.2 | 0.0 |  |
| ours_ts_guard | 100.0 [100.0, 100.0] | – | – | 0.0409 | – | 0.0409 | 13.9 | 0.0 |  |
| ours_shadow | 100.0 [100.0, 100.0] | – | – | 0.0432 | – | 0.0432 | 9.70 | 0.0 |  |
| ours_ts_assist_k3 | 98.0 [94.0, 100.0] | – | – | 0.0179 | – | 0.0182 | 13.6 | 0.0 |  |
| ours_stop_guard_s2lite | 92.0 [84.0, 98.0] | – | – | 0.0100 | – | 0.0109 | 4.77 | 0.0 |  |
| ours_ts_assist_k3_s2lite | 84.0 [74.0, 94.0] | – | – | 0.00372 | – | 0.00442 | 13.6 | 0.0 |  |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

