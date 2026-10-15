## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_shadow | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0447 | 0.0000 | 11.0 | 7.5000 | 0.0000 | 45492.7 | 7.5000 | 0 | 0.0006 | 0.0447 | 0.0707 | 0.3990 |
| ours_stop_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0407 | 0.0000 | 11.6 | 7.5000 | 1.0000 | 45464.8 | 7.5000 | 0 | 0.0024 | 0.0407 | 0.0707 | 0.4712 |
| ours_stop_guard_s2lite | 50 | 0.9400 | 0.8600 | 1.0000 | 0.0099 | 0.0000 | 4.5450 | 7.5600 | 1.0400 | 45573.7 | 7.5600 | 0 | 0.0000 | 0.0099 | 0.0143 | 0.3084 |
| ours_ts_assist_k3 | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0173 | 0.0000 | 15.6 | 7.6800 | 7.6800 | 10217.9 | 7.6800 | 0 | 0.0013 | 0.0173 | 0.0173 | 0.0000 |
| ours_ts_assist_k3_s2lite | 50 | 0.7600 | 0.6400 | 0.8600 | 0.0035 | 0.0000 | 13.1 | 7.7000 | 7.7000 | 9955.2 | 7.7000 | 0 | 0.0000 | 0.0035 | 0.0035 | 0.0000 |
| ours_ts_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0403 | 0.0000 | 9.2615 | 7.5000 | 7.5000 | 45464.0 | 7.5000 | 0 | 0.0000 | 0.0403 | 0.0707 | 0.4780 |
| ours_ts_params_replace * | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0033 | 0.0000 | 16.4 | 1.5000 | 18.1 | 1875.7 | 7.5000 | 0 | 0.0020 | 0.0033 | 0.0033 | 0.0000 |
| ours_ts_replace | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0145 | 0.0000 | 15.0 | 7.5000 | 7.5000 | 8423.9 | 7.5000 | 0 | 0.0080 | 0.0145 | 0.0145 | 0.0000 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### ours_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 375 | 375 | 1.0000 | nan | nan | 1.0000 | nan | nan | 0.0017 | 0.0000 | 1.0000 |
| tool_params/shadow | 531 | 418 | 0.9904 | nan | nan | 1.0000 | 0.8923 | 0.8923 | 0.0079 | 0.0081 | 0.9632 |
| tool_select/shadow | 375 | 375 | 1.0000 | 0.9760 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0015 | 0.0003 | nan |

### ours_stop_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 50 | 50 | 1.0000 | nan | 1.0000 | 0.0000 | nan | nan | 0.0033 | 0.0000 | nan |
| tool_select/off | 375 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### ours_stop_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 52 | 52 | 0.9231 | nan | 0.9231 | 0.0000 | nan | nan | 0.0670 | 0.0614 | 0.8724 |
| tool_select/off | 378 | 0 | nan | nan | nan | 0.0000 | 0.9974 | 0.9974 | nan | nan | nan |

### ours_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 384 | 384 | 1.0000 | 0.9818 | 1.0000 | 0.0000 | 0.9818 | 0.9818 | 0.0016 | 0.0003 | nan |

### ours_ts_assist_k3_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 385 | 385 | 1.0000 | 0.9818 | 1.0000 | 0.0000 | 0.9896 | 0.9896 | 0.0014 | 0.0002 | nan |

### ours_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 375 | 375 | 1.0000 | nan | 1.0000 | 0.0000 | nan | nan | 0.0024 | 0.0000 | nan |

### ours_ts_params_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/generate | 75 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_params/replace | 531 | 523 | 1.0000 | nan | 1.0000 | 0.0000 | nan | 1.0000 | 0.0000 | 0.0000 | nan |
| tool_select/replace | 375 | 375 | 1.0000 | 0.9840 | 1.0000 | 0.0000 | nan | 1.0000 | 0.0014 | 0.0003 | nan |

### ours_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 375 | 375 | 1.0000 | 0.9840 | 1.0000 | 0.0000 | nan | 1.0000 | 0.0014 | 0.0003 | nan |


## Score card

- **Frontier cost-of-pass**: 0.00268 $/success over 7 arms (1 shadow arm(s) left out of the frontier) (best single arm: `ours_ts_params_replace`)
- **Hypervolume** (log-cost × success, normalised): all arms 1, baseline alone –; front: `ours_ts_params_replace`
- note: baseline 'llm_only' not found: baseline-relative metrics omitted
- note: unpriced models (cost counted as 0): jevagentrl-final

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| ours_ts_params_replace | 100.0 [100.0, 100.0] | – | – | 0.00331 | – | 0.00331 | 16.4 | 92.4 | front |
| ours_ts_replace | 100.0 [100.0, 100.0] | – | – | 0.0145 | – | 0.0145 | 15.0 | 100.0 |  |
| ours_ts_assist_k3 | 100.0 [100.0, 100.0] | – | – | 0.0173 | – | 0.0173 | 15.6 | 0.0 |  |
| ours_ts_guard | 100.0 [100.0, 100.0] | – | – | 0.0403 | – | 0.0403 | 9.26 | 0.0 |  |
| ours_stop_guard | 100.0 [100.0, 100.0] | – | – | 0.0407 | – | 0.0407 | 11.6 | 0.0 |  |
| ours_shadow | 100.0 [100.0, 100.0] | – | – | 0.0447 | – | 0.0447 | 11.0 | 0.0 |  |
| ours_stop_guard_s2lite | 94.0 [86.0, 100.0] | – | – | 0.00989 | – | 0.0105 | 4.55 | 0.0 |  |
| ours_ts_assist_k3_s2lite | 76.0 [64.0, 86.0] | – | – | 0.00352 | – | 0.00463 | 13.1 | 0.0 |  |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

