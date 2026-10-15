## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_shadow | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0418 | 0.0000 | 12.6 | 7.5000 | 0.0000 | 45481.3 | 7.5000 | 0 | 0.0000 | 0.0418 | 0.0707 | 0.4529 |
| ours_stop_guard | 50 | 0.9400 | 0.8600 | 1.0000 | 0.0582 | 0.0000 | 16.7 | 9.9200 | 3.4200 | 63520.6 | 9.9200 | 0 | 0.0015 | 0.0582 | 0.1004 | 0.4712 |
| ours_stop_guard_s2lite * | 50 | 0.8600 | 0.7600 | 0.9400 | 0.0130 | 0.0000 | 6.2522 | 9.8600 | 3.2600 | 63365.1 | 9.8600 | 0 | 0.0000 | 0.0130 | 0.0198 | 0.3164 |
| ours_ts_assist_k3 * | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0174 | 0.0000 | 15.0 | 7.6200 | 7.6200 | 10272.7 | 7.6200 | 0 | 0.0013 | 0.0174 | 0.0174 | 0.0000 |
| ours_ts_assist_k3_s2lite * | 50 | 0.8200 | 0.7200 | 0.9200 | 0.0036 | 0.0000 | 13.2 | 7.7600 | 7.7600 | 10278.2 | 7.7600 | 0 | 0.0000 | 0.0036 | 0.0036 | 0.0000 |
| ours_ts_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0491 | 0.0000 | 11.5 | 9.3000 | 8.0400 | 56659.4 | 8.0400 | 0 | 0.0000 | 0.0491 | 0.0881 | 0.4898 |
| ours_ts_params_replace | 50 | 0.7800 | 0.6600 | 0.8800 | 0.0095 | 0.0000 | 20.0 | 3.6000 | 21.8 | 5389.6 | 9.3400 | 0 | 0.0016 | 0.0095 | 0.0095 | 0.0000 |
| ours_ts_replace * | 50 | 0.8800 | 0.7800 | 0.9600 | 0.0164 | 0.0000 | 14.7 | 7.8400 | 7.8400 | 10222.2 | 7.8400 | 0 | 0.0077 | 0.0164 | 0.0173 | 0.0056 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### ours_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 375 | 375 | 0.9627 | nan | nan | 1.0000 | nan | nan | 0.0331 | 0.0230 | 1.0000 |
| tool_params/shadow | 531 | 420 | 0.9310 | nan | nan | 1.0000 | 0.9000 | 0.9000 | 0.0305 | 0.0462 | 0.9420 |
| tool_select/shadow | 375 | 375 | 0.9840 | 0.9653 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0312 | 0.0207 | 0.9201 |

### ours_stop_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 171 | 171 | 0.2749 | nan | 0.2749 | 0.0000 | nan | nan | 0.6707 | 0.5807 | nan |
| tool_select/off | 496 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### ours_stop_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 163 | 163 | 0.4908 | nan | 0.4908 | 0.0000 | nan | nan | 0.4781 | 0.4505 | 0.6697 |
| tool_select/off | 493 | 0 | nan | nan | nan | 0.0000 | 0.9899 | 0.9899 | nan | nan | nan |

### ours_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 381 | 381 | 0.9764 | 0.9580 | 0.9764 | 0.0000 | 0.9843 | 0.9843 | 0.0213 | 0.0247 | 0.9253 |

### ours_ts_assist_k3_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 388 | 388 | 0.9794 | 0.9613 | 0.9794 | 0.0000 | 0.9665 | 0.9665 | 0.0243 | 0.0244 | 0.9201 |

### ours_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 402 | 402 | 0.8458 | nan | 0.8458 | 0.0000 | nan | nan | 0.1430 | 0.1206 | 0.8977 |

### ours_ts_params_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/generate | 168 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_params/replace | 622 | 552 | 0.7880 | nan | 0.8041 | 0.0209 | 1.0000 | 0.8080 | 0.1784 | 0.1883 | 0.6858 |
| tool_select/replace | 467 | 458 | 0.9476 | 0.8952 | 0.9476 | 0.0000 | nan | 0.9476 | 0.0607 | 0.0602 | 0.8109 |

### ours_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 392 | 391 | 0.9028 | 0.8798 | 0.9028 | 0.0000 | nan | 0.9028 | 0.0296 | 0.0621 | 0.9298 |


## Score card

- **Frontier cost-of-pass**: 0.00325 $/success over 7 arms (1 shadow arm(s) left out of the frontier) (best single arm: `ours_ts_assist_k3_s2lite`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.91, baseline alone –; front: `ours_stop_guard_s2lite`, `ours_ts_assist_k3`, `ours_ts_assist_k3_s2lite`, `ours_ts_replace`
- note: baseline 'llm_only' not found: baseline-relative metrics omitted
- note: unpriced models (cost counted as 0): jevagentrl-sft_v0

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| ours_ts_assist_k3 | 100.0 [100.0, 100.0] | – | – | 0.0174 | – | 0.0174 | 15.0 | 0.0 | front |
| ours_shadow | 100.0 [100.0, 100.0] | – | – | 0.0418 | – | 0.0418 | 12.6 | 0.0 |  |
| ours_ts_guard | 100.0 [100.0, 100.0] | – | – | 0.0491 | – | 0.0491 | 11.5 | 0.0 |  |
| ours_stop_guard | 94.0 [86.0, 100.0] | – | – | 0.0582 | – | 0.0619 | 16.7 | 0.0 |  |
| ours_ts_replace | 88.0 [78.0, 96.0] | – | – | 0.0164 | – | 0.0187 | 14.7 | 100.0 | front |
| ours_stop_guard_s2lite | 86.0 [76.0, 94.0] | – | – | 0.0130 | – | 0.0152 | 6.25 | 0.0 | front |
| ours_ts_assist_k3_s2lite | 82.0 [72.0, 92.0] | – | – | 0.00364 | – | 0.00444 | 13.2 | 0.0 | front |
| ours_ts_params_replace | 78.0 [66.0, 88.0] | – | – | 0.00952 | – | 0.0122 | 20.0 | 85.6 |  |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

