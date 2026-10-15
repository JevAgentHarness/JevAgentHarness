## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rl_shadow_cal * | 150 | 1.0000 | 1.0000 | 1.0000 | 0.0437 | 0.0000 | 7.0241 | 7.5000 | 0.0000 | 45379.3 | 7.5000 | 0 | 0.0000 | 0.0437 | 0.0705 | 0.4117 |
| rl_shadow_cal_s2lite * | 150 | 0.8533 | 0.7933 | 0.9067 | 0.0098 | 0.0000 | 4.5174 | 7.5933 | 0.0000 | 45754.5 | 7.5933 | 0 | 0.0000 | 0.0098 | 0.0143 | 0.3271 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### rl_shadow_cal

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 1125 | 1125 | 1.0000 | nan | nan | 1.0000 | nan | nan | 0.0018 | 0.0000 | 1.0000 |
| tool_params/shadow | 1594 | 1328 | 0.9902 | nan | nan | 1.0000 | 0.9367 | 0.9367 | 0.0088 | 0.0082 | 0.9191 |
| tool_select/shadow | 1125 | 1125 | 1.0000 | 0.9822 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0016 | 0.0006 | nan |

### rl_shadow_cal_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 1139 | 1139 | 0.9921 | nan | nan | 1.0000 | nan | nan | 0.0089 | 0.0058 | 0.9996 |
| tool_params/shadow | 1612 | 1480 | 1.0000 | nan | nan | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | nan |
| tool_select/shadow | 1139 | 1139 | 1.0000 | 0.9824 | nan | 1.0000 | 0.9877 | 0.9877 | 0.0012 | 0.0003 | nan |


## Score card

- **Frontier cost-of-pass**: 0.0164 $/success over 2 arms (best single arm: `rl_shadow_cal_s2lite`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.873, baseline alone –; front: `rl_shadow_cal`, `rl_shadow_cal_s2lite`
- note: baseline 'llm_only' not found: baseline-relative metrics omitted
- note: unpriced models (cost counted as 0): jevagentrl-4b

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| rl_shadow_cal | 100.0 [100.0, 100.0] | – | – | 0.0437 | – | 0.0437 | 7.02 | 0.0 | front |
| rl_shadow_cal_s2lite | 85.3 [79.3, 90.7] | – | – | 0.00977 | – | 0.0114 | 4.52 | 0.0 | front |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

