# E7 difficulty pilot (development seeds; never a result)

Generated 2026-10-06T15:27:51+00:00; Flash LLM-only, temperature 0, minimal reasoning, via OpenRouter. Total pilot spend $17.64.

| candidate | families | k | rule | max steps | seeds | n | errored | success % | by family | failures | $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `e7_pilot_amount_k12` | multi | 12 | amount | 45 | 7150-7179 | 30 | 0 | 66.7 | multi 20/30 | wrong_total 10 | 2.20 |
| `e7_pilot_amount_k8` | multi | 8 | amount | 40 | 7030-7059 | 30 | 0 | 100.0 | multi 30/30 | -- | 1.49 |
| `e7_pilot_k10` | v2 | 10 | given | 45 | 7000-7029 | 30 | 0 | 100.0 | clarify 15/15, multi 15/15 | -- | 1.10 |
| `e7_pilot_k6` | v2 | 6 | given | 30 | 7000-7029 | 30 | 0 | 100.0 | clarify 15/15, multi 15/15 | -- | 0.79 |
| `e7_pilot_tier_k10` | multi | 10 | tier | 45 | 7090-7119 | 30 | 0 | 76.7 | multi 23/30 | wrong_total 7 | 1.83 |
| `e7_pilot_tier_k11` | multi | 11 | tier | 45 | 7180-7209 | 30 | 0 | 80.0 | multi 24/30 | wrong_total 6 | 2.05 |
| `e7_pilot_tier_k11b` | multi | 11 | tier | 45 | 7210-7239 | 30 | 0 | 76.7 | multi 23/30 | wrong_total 7 | 2.13 |
| `e7_pilot_tier_k12` | multi | 12 | tier | 45 | 7120-7149 | 30 | 0 | 56.7 | multi 17/30 | wrong_total 13 | 2.25 |
| `e7_pilot_tier_k12b` | multi | 12 | tier | 45 | 7240-7269 | 30 | 0 | 36.7 | multi 11/30 | wrong_total 19 | 2.32 |
| `e7_pilot_tier_k8` | multi | 8 | tier | 40 | 7060-7089 | 30 | 0 | 90.0 | multi 27/30 | wrong_total 3 | 1.50 |

- aborted_direct_api: 2 episodes (seeds 7006, 7008), $0.00; direct Gemini API, about 15:00 UTC 2026-10-06: HTTP 503 'high demand'; discarded
- route_check: 4 episodes (seeds 7290-7291), $0.10; llm_only and jevagent_best, 6 customers, Flash and Jev via OpenRouter: the route works
- pilot spend in total: $17.75
- chosen: {'families': 'multi', 'multi_customers': 10, 'multi_rule': 'tier', 'max_steps': 45, 'rule': 'among the tier-rule candidates in the 60-80 % band, the one closest to 70 %'}
