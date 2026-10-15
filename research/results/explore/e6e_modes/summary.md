# E6E: module-map cells (exploratory (E6e))

- generated 2026-10-06T18:44:11+00:00 at git `aad6883d168b` by `scripts/explore_e6e_modes.py`; run `runs/e6e_modes`; paired task bootstrap B = 10000, seed 0
- spend (every E6E record, smoke included): $22.65 (Gemini $22.20, Jev $0.45) of the $60 cap
- verdict rule (descriptive): costs success if delta <= -3 pt; no saving if saving <= 0; takes over safely if delta >= -1 pt, saving > 0 and acted >= 5 %; else mixed (scripts/module_map.py, fixed 2026-10-05)
- cells, semantics and thresholds: `results/explore/e6e_modes/cells.md`; registration: `docs/PREREG_A7_STAGE2.md`, Deviations "Exploratory row E6E". No test; outside every Holm table.

- reference `llm_only`: n = 200 complete, success 99.5 %, $16.0 per 1k episodes, 8.1 s per episode
- reference `small_only`: n = 200 complete, success 80.0 %, $9.8 per 1k episodes, 5.3 s per episode

| cell | arm | ref. | n | succ. % | Δ pt [95 % CI] | wins / losses | $/1k ep. | saving % [95 % CI] | coverage % | acted % | acc. % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| memory_write/replace | `mw_replace` | llm_only | 200 | 100.0 | +0.5 [+0.0, +1.5] | 1 / 0 | 23.7 | -47.4 [-53.1, -42.4] | 100.0 | 100.0 (decided) | 69.4 | no saving |
| memory_read/replace | `mr_replace` | llm_only | 200 | 100.0 | +0.5 [+0.0, +1.5] | 1 / 0 | 17.8 | -11.1 [-13.9, -8.4] | 100.0 | 100.0 (decided) | 57.2 | no saving |
| reflection/replace | `rf_replace` | llm_only | 200 | 99.0 | -0.5 [-1.5, +0.0] | 0 / 1 | 17.2 | -7.2 [-9.8, -4.6] | 100.0 | 100.0 (decided) | 99.9 | no saving |
| stop/replace | `st_replace` | llm_only | 200 | 100.0 | +0.5 [+0.0, +1.5] | 1 / 0 | 17.1 | -6.8 [-10.8, -3.4] | 100.0 | 100.0 (decided) | 98.1 | no saving |
| routing/replace | `rt_replace` | small_only | 200 | 84.0 | +4.0 [+0.5, +7.5] | 11 / 3 | 10.4 | -6.7 [-8.2, -5.3] | 100.0 | 100.0 (decided) | 97.8 | no saving |

Coverage: share of the point's decisions executed from the decision model's answer (cascade: answered at confidence ≥ τ; trigger: 100 %, every gate decision is the model's; assist and guard: 0 by definition). Acted: cascade = coverage; assist = 100 % (shortlist at every decision); guard = proposals vetoed; trigger = questions on which it fired. Acc.: cascade = accepted answers; trigger and guard = the model's answers; assist = gold value in the shortlist (recall@k). Routing is compared with Flash-Lite alone (as C4). Stop cascade: the LLM's own stop decision is free (its tool call), so a decided answer changes the episode only where it differs from it (`overrode`: rejected finishes and early stops). Argument guard: `veto_on_other` = vetoes of a value outside the model's candidate list (which it cannot choose), `veto_llm_was_wrong` = vetoed values that were wrong, among the vetoes on the oracle's tool. Replace (E6e): coverage and acted = 100 % (the model's argmax decides every question; the LLM is never asked); acc. = its answers (memory read: its top memory, or none, in the gold set).

## Per cell

### memory_write × replace (`mw_replace`)

- same rule as: trigger at tau = 0.5: System-1 alone deciding store / skip is the write trigger at 0.5
- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 100.0 % vs 99.5 %; Δ +0.5 pt (bootstrap [+0.0, +1.5], Newcombe [-1.4, +2.8]); wins / losses 1 / 0
- cost $23.7 vs $16.0 per 1k episodes (saving -47.4 %); System-1 $0.56, System-2 $23.1 per 1k; LLM calls 7.62 vs 7.49, System-1 calls 6.63 per episode; wall 9.1 s vs 8.1 s (median 10.4 vs 9.0); episodes with an empty LLM reply 0 vs 1
- decision point: n_decisions 1326, n_gold 1325, coverage 100.0 %, acted 100.0 %, acc 69.4 %, model_yes_rate 98.5 %, gold_positive_rate 67.9 %

### memory_read × replace (`mr_replace`)

- same rule as: assist with k = 1: the read assist already injects System-1's selection without an LLM choice
- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 100.0 % vs 99.5 %; Δ +0.5 pt (bootstrap [+0.0, +1.5], Newcombe [-1.4, +2.8]); wins / losses 1 / 0
- cost $17.8 vs $16.0 per 1k episodes (saving -11.1 %); System-1 $0.57, System-2 $17.3 per 1k; LLM calls 7.50 vs 7.49, System-1 calls 5.50 per episode; wall 8.8 s vs 8.1 s (median 10.7 vs 9.0); episodes with an empty LLM reply 0 vs 1
- decision point: n_decisions 1100, n_gold 1100, coverage 100.0 %, acted 100.0 %, acc 57.2 %, mean_injected 1.00, final_covers 68.6 %

### reflection × replace (`rf_replace`)

- same rule as: trigger at tau = 0.5: System-1 alone deciding reflect / skip is the trigger at 0.5
- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 99.0 % vs 99.5 %; Δ -0.5 pt (bootstrap [-1.5, +0.0], Newcombe [-3.0, +1.6]); wins / losses 0 / 1
- cost $17.2 vs $16.0 per 1k episodes (saving -7.2 %); System-1 $0.37, System-2 $16.8 per 1k; LLM calls 8.39 vs 7.49, System-1 calls 6.49 per episode; wall 10.4 s vs 8.1 s (median 11.7 vs 9.0); episodes with an empty LLM reply 2 vs 1
- decision point: n_decisions 1297, n_gold 1115, coverage 100.0 %, acted 100.0 %, acc 99.9 %, model_yes_rate 14.0 %, gold_positive_rate 0.1 %

### stop × replace (`st_replace`)

- same rule as: cascade at tau = 0.5: System-1 alone deciding stop / continue at every step is the guard and the trigger together at 0.5
- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 100.0 % vs 99.5 %; Δ +0.5 pt (bootstrap [+0.0, +1.5], Newcombe [-1.4, +2.8]); wins / losses 1 / 0
- cost $17.1 vs $16.0 per 1k episodes (saving -6.8 %); System-1 $0.38, System-2 $16.8 per 1k; LLM calls 7.69 vs 7.49, System-1 calls 7.68 per episode; wall 9.8 s vs 8.1 s (median 10.5 vs 9.0); episodes with an empty LLM reply 2 vs 1
- decision point: n_decisions 1537, n_gold 1537, coverage 100.0 %, acted 100.0 %, acc 98.1 %, model_yes_rate 13.0 %, gold_positive_rate 14.9 %, finish_rejected 32, early_stops 1, n_finish_proposals 231, overrode 2.1 %

### routing × replace (`rt_replace`)

- same rule as: trigger at tau = 0.5: System-1 alone choosing the strong or the cheap LLM is the routing trigger at 0.5
- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 84.0 % vs 80.0 %; Δ +4.0 pt (bootstrap [+0.5, +7.5], Newcombe [+0.2, +8.0]); wins / losses 11 / 3
- cost $10.4 vs $9.8 per 1k episodes (saving -6.7 %); System-1 $0.35, System-2 $10.1 per 1k; LLM calls 7.57 vs 7.57, System-1 calls 7.57 per episode; wall 6.8 s vs 5.3 s (median 6.8 vs 5.3); episodes with an empty LLM reply 0 vs 0
- decision point: n_decisions 1513, n_gold 1444, coverage 100.0 %, acted 100.0 %, acc 97.8 %, model_yes_rate 4.6 %, gold_positive_rate 2.2 %
- vs Flash alone (`llm_only`, descriptive): Δ -15.5 pt, saving 34.9 %, wins / losses 1 / 32

