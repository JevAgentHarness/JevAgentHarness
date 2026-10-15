## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intern_shadow * | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0410 | 0.0000 | 12.5 | 7.5000 | 0.0000 | 45464.0 | 7.5000 | 0 | 0.0000 | 0.0410 | 0.0707 | 0.4668 |
| intern_stop_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0427 | 0.0000 | 11.6 | 7.6200 | 1.1200 | 46360.8 | 7.6200 | 0 | 0.0000 | 0.0427 | 0.0722 | 0.4510 |
| intern_stop_guard_s2lite * | 50 | 0.9200 | 0.8400 | 0.9800 | 0.0120 | 0.0000 | 5.9105 | 9.1000 | 2.4800 | 57283.0 | 9.1000 | 0 | 0.0000 | 0.0120 | 0.0179 | 0.3124 |
| intern_ts_assist_k3 * | 50 | 0.9600 | 0.9000 | 1.0000 | 0.0221 | 0.0003 | 9.8078 | 9.1000 | 18.2 | 13067.7 | 9.1000 | 0 | 0.0000 | 0.0221 | 0.0221 | 0.0000 |
| intern_ts_assist_k3_s2lite * | 50 | 0.6000 | 0.4600 | 0.7400 | 0.0044 | 0.0003 | 8.1473 | 8.5400 | 17.1 | 11751.1 | 8.5400 | 0 | 0.0000 | 0.0044 | 0.0044 | 0.0000 |
| intern_ts_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0435 | 0.0001 | 10.8 | 8.0000 | 7.6400 | 48393.6 | 7.6400 | 0 | 0.0000 | 0.0435 | 0.0753 | 0.4755 |
| intern_ts_params_replace | 50 | 0.1800 | 0.0800 | 0.3000 | 0.0078 | 0.0002 | 5.6207 | 3.2000 | 20.4 | 4081.0 | 6.2800 | 0 | 0.0000 | 0.0078 | 0.0078 | 0.0000 |
| intern_ts_replace | 50 | 0.2600 | 0.1400 | 0.3800 | 0.0080 | 0.0001 | 5.3074 | 4.8200 | 9.6400 | 4345.3 | 4.8200 | 0 | 0.0083 | 0.0080 | 0.0080 | 0.0000 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### intern_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 375 | 375 | 0.9307 | nan | nan | 1.0000 | nan | nan | 0.2083 | 0.0754 | 0.9897 |
| tool_params/shadow | 531 | 424 | 0.8231 | nan | nan | 1.0000 | 0.9009 | 0.9009 | 0.1426 | 0.1134 | 0.9144 |
| tool_select/shadow | 375 | 375 | 0.7387 | 0.6800 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0821 | 0.1814 | 0.6985 |

### intern_stop_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 56 | 56 | 0.8929 | nan | 0.8929 | 0.0000 | nan | nan | 0.2370 | 0.0834 | nan |
| tool_select/off | 381 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### intern_stop_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 124 | 124 | 0.5887 | nan | 0.5887 | 0.0000 | nan | nan | 0.3264 | 0.2719 | 0.8115 |
| tool_select/off | 455 | 0 | nan | nan | nan | 0.0000 | 0.9868 | 0.9868 | nan | nan | nan |

### intern_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 455 | 453 | 0.6578 | 0.5850 | 0.6578 | 0.0000 | 0.8102 | 0.8102 | 0.1092 | 0.1896 | 0.7853 |

### intern_ts_assist_k3_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 427 | 427 | 0.6464 | 0.5621 | 0.6464 | 0.0000 | 0.7471 | 0.7471 | 0.1290 | 0.2063 | 0.7442 |

### intern_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 382 | 382 | 0.9555 | nan | 0.9555 | 0.0000 | nan | nan | 0.1404 | 0.0406 | 0.9685 |

### intern_ts_params_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/generate | 154 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_params/replace | 394 | 200 | 0.5450 | nan | 0.5450 | 0.0178 | nan | 0.5450 | 0.3400 | 0.3568 | 0.4149 |
| tool_select/replace | 314 | 297 | 0.6296 | 0.5286 | 0.6296 | 0.0000 | nan | 0.6296 | 0.1059 | 0.2334 | 0.6381 |

### intern_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 241 | 241 | 0.5436 | 0.4274 | 0.5436 | 0.0000 | nan | 0.5436 | 0.2064 | 0.2505 | 0.7331 |


## Score card

- **Frontier cost-of-pass**: 0.00884 $/success over 7 arms (1 shadow arm(s) left out of the frontier) (best single arm: `intern_ts_assist_k3_s2lite`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.811, baseline alone –; front: `intern_stop_guard`, `intern_stop_guard_s2lite`, `intern_ts_assist_k3`, `intern_ts_assist_k3_s2lite`
- note: baseline 'llm_only' not found: baseline-relative metrics omitted

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| intern_shadow | 100.0 [100.0, 100.0] | – | – | 0.0410 | – | 0.0410 | 12.5 | 0.0 |  |
| intern_stop_guard | 100.0 [100.0, 100.0] | – | – | 0.0427 | – | 0.0427 | 11.6 | 0.0 | front |
| intern_ts_guard | 100.0 [100.0, 100.0] | – | – | 0.0435 | – | 0.0435 | 10.8 | 0.0 |  |
| intern_ts_assist_k3 | 96.0 [90.0, 100.0] | – | – | 0.0221 | – | 0.0231 | 9.81 | 0.0 | front |
| intern_stop_guard_s2lite | 92.0 [84.0, 98.0] | – | – | 0.0120 | – | 0.0130 | 5.91 | 0.0 | front |
| intern_ts_assist_k3_s2lite | 60.0 [46.0, 74.0] | – | – | 0.00441 | – | 0.00735 | 8.15 | 0.0 | front |
| intern_ts_replace | 26.0 [14.0, 38.0] | – | – | 0.00801 | – | 0.0308 | 5.31 | 100.0 |  |
| intern_ts_params_replace | 18.0 [8.0, 30.0] | – | – | 0.00779 | – | 0.0433 | 5.62 | 81.3 |  |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

