# E22: the training recipe on tau^2-bench (exploratory, descriptive)

docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 'E22' and its amendment (the task as the unit of analysis). Test tasks of tau2-bench's official split; success = mean over tasks of the per-task mean over trials; Δ with a paired task bootstrap (trials stay with their task); NI verdict at 5 points by the certified ordered p (A6.21) on per-task majority pass. Retail NL judge: Gemini 3.5 Flash (not gpt-4.1).

GPU tariff q = 1.212 $/h x 0.36 h evaluation window / 480 GPU-arm episodes = 0.00091 $ per episode (E9's tariff: 0.00168).

| scope | arm | tasks | LLM-only % | arm % | Δ pt [task bootstrap] | NI p (tasks) | NI | saving % (GPU) | API-only % | time s (ref / arm) | LLM-free turns % | LLM calls / ep (ref / arm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | e22_final | 60 | 66.2 | 54.2 | -12.1 [-19.2, -5.4] | 0.979 | not shown | 4.6 | 5.8 | 19.8 / 26.9 | 3.9 | 11.95 / 11.30 |
| pooled | e22_tau05 | 60 | 66.2 | 59.2 | -7.1 [-13.8, -0.4] | 0.654 | not shown | 31.3 | 32.5 | 19.8 / 28.0 | 39.0 | 11.95 / 7.45 |
| retail | e22_final | 40 | 64.4 | 52.5 | -11.9 [-20.6, -3.8] | 0.857 | not shown | 5.8 | 7.0 | 19.6 / 26.7 | 5.6 | 11.54 / 10.69 |
| retail | e22_tau05 | 40 | 64.4 | 56.9 | -7.5 [-16.9, +1.9] | 0.560 | not shown | 32.4 | 33.6 | 19.6 / 26.8 | 41.0 | 11.54 / 7.01 |
| airline | e22_final | 20 | 70.0 | 57.5 | -12.5 [-25.0, -1.2] | 1.000 | not shown | 2.5 | 3.6 | 20.4 / 27.4 | 0.9 | 12.79 / 12.53 |
| airline | e22_tau05 | 20 | 70.0 | 63.7 | -6.2 [-15.0, +1.2] | 0.926 | not shown | 29.4 | 30.4 | 20.4 / 30.3 | 35.2 | 12.79 / 8.32 |

```
{
 "fit": {
  "tool_select": {
   "tau_hat": 0.9944593912273271,
   "rule": "projection",
   "n_paired": 997,
   "delta_c": 0.005891642978936811,
   "V": 1.0
  },
  "tool_params": {
   "tau_hat": 0.9980051159281186,
   "rule": "projection",
   "n_paired": 1436,
   "delta_c": 0.003855671030640668,
   "V": 1.0
  }
 },
 "dev_replay": {
  "tool_select": {
   "n": 1504,
   "acc": 84.57446808510637,
   "coverage_at_tau": 13.297872340425531,
   "accepted": 200,
   "accepted_wrong": 0
  },
  "tool_params": {
   "n": 1436,
   "acc": 95.47353760445682,
   "coverage_at_tau": 49.79108635097493,
   "accepted": 715,
   "accepted_wrong": 0
  }
 },
 "spend": {
  "host_usd": 51.734,
  "local_usd": 68.95916345,
  "harvest_usd": 68.67916345,
  "harvest_smoke_usd": 0.28
 },
 "plan": {
  "smoke_usd_per_sim": 0.0788,
  "smoke_sims": 4,
  "spent_before_eval": 69.275,
  "cap": 200.0,
  "always_defer": false,
  "plan": {
   "arms": [
    "llm_only",
    "e22_final",
    "e22_tau05"
   ],
   "trials": 4,
   "projection_usd": 143.0
  }
 }
}
```
