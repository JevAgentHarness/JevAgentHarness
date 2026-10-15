## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_shadow * | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0408 | 0.0000 | 9.8357 | 7.5000 | 0.0000 | 45425.9 | 7.5000 | 0 | 0.0000 | 0.0408 | 0.0706 | 0.4692 |
| jev_stop_guard | 50 | 0.9800 | 0.9400 | 1.0000 | 0.0447 | 0.0001 | 8.5263 | 8.1200 | 1.6200 | 50020.0 | 8.1200 | 0 | 0.0000 | 0.0447 | 0.0779 | 0.4759 |
| jev_stop_guard_s2lite | 50 | 0.7600 | 0.6400 | 0.8600 | 0.0132 | 0.0001 | 7.4182 | 10.1 | 3.5400 | 65089.8 | 10.1 | 0 | 0.0000 | 0.0132 | 0.0204 | 0.3304 |
| jev_ts_assist_k3 * | 50 | 0.9200 | 0.8400 | 0.9800 | 0.0202 | 0.0010 | 8.6068 | 8.3000 | 8.3000 | 11358.9 | 8.3000 | 0 | 0.0000 | 0.0202 | 0.0202 | 0.0000 |
| jev_ts_assist_k3_s2lite * | 50 | 0.7000 | 0.5800 | 0.8200 | 0.0048 | 0.0009 | 5.6313 | 8.1400 | 8.1400 | 10847.4 | 8.1400 | 0 | 0.0000 | 0.0048 | 0.0048 | 0.0000 |
| jev_ts_guard | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0413 | 0.0004 | 11.7 | 7.6200 | 7.5000 | 46212.4 | 7.5000 | 0 | 0.0000 | 0.0413 | 0.0722 | 0.4819 |
| jev_ts_params_replace * | 50 | 0.8000 | 0.6800 | 0.9000 | 0.0066 | 0.0015 | 5.3671 | 2.2800 | 18.8 | 2812.0 | 7.8400 | 0 | 0.0019 | 0.0066 | 0.0066 | 0.0000 |
| jev_ts_replace | 50 | 0.7800 | 0.6600 | 0.8800 | 0.0140 | 0.0008 | 12.7 | 6.7800 | 6.7800 | 7695.2 | 6.7800 | 0 | 0.0118 | 0.0140 | 0.0140 | 0.0000 |
| llm_only | 50 | 1.0000 | 1.0000 | 1.0000 | 0.0440 | 0.0000 | 11.2 | 7.5000 | 0.0000 | 45425.1 | 7.5000 | 0 | 0.0027 | 0.0440 | 0.0706 | 0.4143 |
| llm_only_s2lite * | 50 | 0.8200 | 0.7000 | 0.9200 | 0.0098 | 0.0000 | 4.7878 | 7.5400 | 0.0000 | 45388.0 | 7.5400 | 0 | 0.0000 | 0.0098 | 0.0142 | 0.3205 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### jev_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 375 | 375 | 0.9920 | nan | nan | 1.0000 | nan | nan | 0.0475 | 0.0081 | 1.0000 |
| tool_params/shadow | 531 | 433 | 0.9400 | nan | nan | 1.0000 | 0.9238 | 0.9238 | 0.0433 | 0.0521 | 0.8860 |
| tool_select/shadow | 375 | 375 | 0.9413 | 0.8667 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0604 | 0.0638 | 0.7159 |

### jev_stop_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 81 | 81 | 0.6049 | nan | 0.6049 | 0.0000 | nan | nan | 0.3844 | 0.2529 | nan |
| tool_select/off | 406 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### jev_stop_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 177 | 177 | 0.7684 | nan | 0.7684 | 0.0000 | nan | nan | 0.1390 | 0.1743 | 0.7952 |
| tool_select/off | 507 | 0 | nan | nan | nan | 0.0000 | 0.9901 | 0.9901 | nan | nan | nan |

### jev_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 415 | 415 | 0.8819 | 0.8241 | 0.8819 | 0.0000 | 0.8940 | 0.8940 | 0.0537 | 0.0883 | 0.7847 |

### jev_ts_assist_k3_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 407 | 407 | 0.9017 | 0.8477 | 0.9017 | 0.0000 | 0.9066 | 0.9066 | 0.0671 | 0.0881 | 0.7132 |

### jev_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 375 | 375 | 0.9840 | nan | 0.9840 | 0.0000 | nan | nan | 0.1277 | 0.0280 | nan |

### jev_ts_params_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/generate | 114 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_params/replace | 546 | 423 | 0.9858 | nan | 0.9858 | 0.0000 | nan | 0.9858 | 0.0212 | 0.0180 | 0.8965 |
| tool_select/replace | 392 | 383 | 0.8616 | 0.7598 | 0.8616 | 0.0000 | nan | 0.8616 | 0.0604 | 0.0706 | 0.9175 |

### jev_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 339 | 339 | 0.9587 | 0.8732 | 0.9587 | 0.0000 | nan | 0.9587 | 0.0332 | 0.0404 | 0.8467 |

### llm_only

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/off | 375 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### llm_only_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/off | 377 | 0 | nan | nan | nan | 0.0000 | 0.9947 | 0.9947 | nan | nan | nan |


## Score card (baseline: `llm_only`)

- **Iso-quality cost savings**: 6.2% [4.4, 8.1] with `jev_ts_guard` (Δsuccess +0.0 pp [0.0, 0.0], p_worse=1.000; not significantly worse, one-sided paired bootstrap, α=0.05; 2 arm(s) qualify)
- **Quality at iso-cost** (≤ 0.044 $/episode): `jev_ts_guard` 100.0% vs. reference 100.0% (Δ +0.0 pp, p_better=1.000); hull 100.0%
- **Iso-quality cost savings** (vs. `llm_only_s2lite`): -35.0% [-59.7, -13.2] with `jev_stop_guard_s2lite` (Δsuccess -6.0 pp [-14.0, 2.0], p_worse=0.127; not significantly worse, one-sided paired bootstrap, α=0.05; 1 arm(s) qualify)
- **Quality at iso-cost** (vs. `llm_only_s2lite`) (≤ 0.0098 $/episode): `llm_only_s2lite` 82.0% vs. reference 82.0% (Δ +0.0 pp, p_better=–); hull 82.0%
- **Frontier cost-of-pass**: 0.00436 $/success over 9 arms (1 shadow arm(s) left out of the frontier) (best single arm: `jev_ts_assist_k3_s2lite`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.853, baseline alone 0.0996; front: `jev_ts_assist_k3`, `jev_ts_assist_k3_s2lite`, `jev_ts_guard`, `jev_ts_params_replace`, `llm_only_s2lite`
- note: per-System-2 references: `llm_only_s2lite` (gemini-3.5-flash-lite) for 2 arm(s); every other arm vs. `llm_only`

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| llm_only | 100.0 [100.0, 100.0] | – | – | 0.0440 | – | 0.0440 | 11.2 | 0.0 | baseline |
| jev_shadow | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0408 | 7.2 | 0.0408 | 9.84 | 0.0 |  |
| jev_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0413 | 6.2 | 0.0413 | 11.7 | 0.0 | front,iso-best |
| jev_stop_guard | 98.0 [94.0, 100.0] | -2.0 | 0.384 | 0.0447 | -1.6 | 0.0456 | 8.53 | 0.0 | iso-q |
| jev_ts_assist_k3 | 92.0 [84.0, 98.0] | -8.0 | 0.017 | 0.0202 | 54.1 | 0.0220 | 8.61 | 0.0 | front,worse |
| llm_only_s2lite | 82.0 [70.0, 92.0] | -18.0 | <0.001 | 0.00980 | 77.7 | 0.0119 | 4.79 | 0.0 | front,worse |
| jev_ts_params_replace | 80.0 [68.0, 90.0] | -20.0 | <0.001 | 0.00659 | 85.0 | 0.00824 | 5.37 | 89.2 | front,worse |
| jev_ts_replace | 78.0 [66.0, 88.0] | -22.0 | <0.001 | 0.0140 | 68.2 | 0.0179 | 12.7 | 100.0 | worse |
| jev_stop_guard_s2lite | 76.0 [64.0, 86.0] | -6.0 | 0.127 | 0.0132 | -35.0 | 0.0174 | 7.42 | 0.0 | iso-best,vs llm_only_s2lite |
| jev_ts_assist_k3_s2lite | 70.0 [58.0, 82.0] | -12.0 | 0.043 | 0.00477 | 51.3 | 0.00681 | 5.63 | 0.0 | front,vs llm_only_s2lite |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

