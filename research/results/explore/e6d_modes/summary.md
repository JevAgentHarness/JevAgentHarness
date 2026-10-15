# E6D: module-map cells (exploratory (E6d))

- generated 2026-10-06T17:01:35+00:00 at git `98e80247d023` by `scripts/explore_e6d_modes.py`; run `runs/e6d_modes`; paired task bootstrap B = 10000, seed 0
- spend (every E6D record, smoke included): $40.57 (Gemini $40.12, Jev $0.45) of the $120 cap
- verdict rule (descriptive): costs success if delta <= -3 pt; no saving if saving <= 0; takes over safely if delta >= -1 pt, saving > 0 and acted >= 5 %; else mixed (scripts/module_map.py, fixed 2026-10-05)
- cells, semantics and thresholds: `results/explore/e6d_modes/cells.md`; registration: `docs/PREREG_A7_STAGE2.md`, Deviations "Exploratory row E6D". No test; outside every Holm table.

- reference `llm_only`: n = 200 complete, success 100.0 %, $16.0 per 1k episodes, 8.0 s per episode
- reference `small_only`: n = 200 complete, success 25.5 %, $2.5 per 1k episodes, 4.5 s per episode
- reference `small_only (e6d_modes_routing)`: n = 200 complete, success 83.5 %, $9.8 per 1k episodes, 5.6 s per episode
- **excluded (infrastructure):** `rt_assist` and its reference `small_only` in `runs/e6d_modes`: episodes with an empty LLM reply {'rt_assist': 141, 'small_only': 147} of {'rt_assist': 200, 'small_only': 200}; the cell is read from its repair run instead (Deviations addendum)
- **excluded (infrastructure):** `rt_guard` and its reference `small_only` in `runs/e6d_modes`: episodes with an empty LLM reply {'rt_guard': 146, 'small_only': 147} of {'rt_guard': 200, 'small_only': 200}; the cell is read from its repair run instead (Deviations addendum)

| cell | arm | ref. | n | succ. % | Δ pt [95 % CI] | wins / losses | $/1k ep. | saving % [95 % CI] | coverage % | acted % | acc. % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| memory_write/assist | `mw_assist` | llm_only | 200 | 100.0 | +0.0 [+0.0, +0.0] | 0 / 0 | 27.8 | -74.2 [-79.1, -69.2] | 0.0 | 100.0 (advised) | 69.3 | no saving |
| reflection/assist | `rf_assist` | llm_only | 200 | 99.5 | -0.5 [-1.5, +0.0] | 0 / 1 | 26.4 | -65.7 [-70.5, -61.1] | 0.0 | 100.0 (advised) | 69.0 | no saving |
| stop/assist | `st_assist` | llm_only | 200 | 100.0 | +0.0 [+0.0, +0.0] | 0 / 0 | 18.6 | -16.7 [-19.8, -13.7] | 0.0 | 100.0 (advised) | 99.3 | no saving |
| routing/assist | `rt_assist` | small_only | 200 | 81.5 | -2.0 [-6.0, +2.0] | 7 / 11 | 12.8 | -31.0 [-33.9, -28.2] | 0.0 | 100.0 (advised) | 92.8 | no saving |
| reflection/guard | `rf_guard` | llm_only | 200 | 100.0 | +0.0 [+0.0, +0.0] | 0 / 0 | 25.7 | -61.2 [-65.6, -56.6] | 0.0 | 90.9 (vetoed) | 100.0 | no saving |
| routing/guard | `rt_guard` | small_only | 200 | 83.5 | +0.0 [-4.0, +4.0] | 8 / 8 | 12.0 | -22.6 [-24.3, -20.9] | 0.0 | 95.8 (vetoed) | 100.0 | no saving |
| memory_read/guard | `mr_guard` | llm_only | 200 | 100.0 | +0.0 [+0.0, +0.0] | 0 / 0 | 38.2 | -139.8 [-148.7, -130.9] | 0.0 | 41.2 (vetoed) | 58.7 | no saving |

Coverage: share of the point's decisions executed from the decision model's answer (cascade: answered at confidence ≥ τ; trigger: 100 %, every gate decision is the model's; assist and guard: 0 by definition). Acted: cascade = coverage; assist = 100 % (shortlist at every decision); guard = proposals vetoed; trigger = questions on which it fired. Acc.: cascade = accepted answers; trigger and guard = the model's answers; assist = gold value in the shortlist (recall@k). Routing is compared with Flash-Lite alone (as C4). Stop cascade: the LLM's own stop decision is free (its tool call), so a decided answer changes the episode only where it differs from it (`overrode`: rejected finishes and early stops). Argument guard: `veto_on_other` = vetoes of a value outside the model's candidate list (which it cannot choose), `veto_llm_was_wrong` = vetoed values that were wrong, among the vetoes on the oracle's tool.

## Per cell

### memory_write × assist (`mw_assist`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 100.0 % vs 100.0 %; Δ +0.0 pt (bootstrap [+0.0, +0.0], Newcombe [-1.9, +1.9]); wins / losses 0 / 0
- cost $27.8 vs $16.0 per 1k episodes (saving -74.2 %); System-1 $0.38, System-2 $27.4 per 1k; LLM calls 14.01 vs 7.50, System-1 calls 6.50 per episode; wall 14.2 s vs 8.0 s (median 17.1 vs 9.2); episodes with an empty LLM reply 0 vs 0
- decision point: n_decisions 1301, n_gold 1301, coverage 0.0 %, acted 100.0 %, followed 2.7 %, model_yes_rate 100.0 %, llm_yes_rate 2.7 %, acc 69.3 %, llm_final_acc 33.4 %

### reflection × assist (`rf_assist`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 99.5 % vs 100.0 %; Δ -0.5 pt (bootstrap [-1.5, +0.0], Newcombe [-2.8, +1.4]); wins / losses 0 / 1
- cost $26.4 vs $16.0 per 1k episodes (saving -65.7 %); System-1 $0.35, System-2 $26.1 per 1k; LLM calls 14.01 vs 7.50, System-1 calls 6.50 per episode; wall 14.1 s vs 8.0 s (median 16.8 vs 9.2); episodes with an empty LLM reply 1 vs 0
- decision point: n_decisions 1301, n_gold 1301, coverage 0.0 %, acted 100.0 %, followed 69.1 %, model_yes_rate 30.9 %, llm_yes_rate 0.0 %, acc 69.0 %, llm_final_acc 99.9 %

### stop × assist (`st_assist`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 100.0 % vs 100.0 %; Δ +0.0 pt (bootstrap [+0.0, +0.0], Newcombe [-1.9, +1.9]); wins / losses 0 / 0
- cost $18.6 vs $16.0 per 1k episodes (saving -16.7 %); System-1 $0.35, System-2 $18.3 per 1k; LLM calls 7.50 vs 7.50, System-1 calls 7.50 per episode; wall 9.2 s vs 8.0 s (median 10.8 vs 9.2); episodes with an empty LLM reply 0 vs 0
- decision point: n_decisions 1501, n_gold 1501, coverage 0.0 %, acted 100.0 %, followed 99.3 %, model_yes_rate 12.6 %, llm_yes_rate 13.3 %, acc 99.3 %, llm_final_acc 100.0 %

### routing × assist (`rt_assist`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 81.5 % vs 83.5 %; Δ -2.0 pt (bootstrap [-6.0, +2.0], Newcombe [-6.4, +2.4]); wins / losses 7 / 11
- cost $12.8 vs $9.8 per 1k episodes (saving -31.0 %); System-1 $0.36, System-2 $12.5 per 1k; LLM calls 15.09 vs 7.57, System-1 calls 7.54 per episode; wall 11.9 s vs 5.6 s (median 12.3 vs 5.6); episodes with an empty LLM reply 1 vs 0
- decision point: n_decisions 1509, n_gold 1468, coverage 0.0 %, acted 100.0 %, followed 92.7 %, model_yes_rate 4.8 %, llm_yes_rate 2.7 %, acc 92.8 %, llm_final_acc 97.5 %

### reflection × guard (`rf_guard`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 100.0 % vs 100.0 %; Δ +0.0 pt (bootstrap [+0.0, +0.0], Newcombe [-1.9, +1.9]); wins / losses 0 / 0
- cost $25.7 vs $16.0 per 1k episodes (saving -61.2 %); System-1 $0.00, System-2 $25.7 per 1k; LLM calls 14.02 vs 7.50, System-1 calls 0.06 per episode; wall 12.8 s vs 8.0 s (median 15.7 vs 9.2); episodes with an empty LLM reply 0 vs 0
- decision point: n_decisions 1301, n_gold 1300, coverage 0.0 %, n_proposals 11, n_vetoed 10, acted 90.9 %, acc 100.0 %, llm_store_rate 0.8 %, veto_precision 100.0 %

### routing × guard (`rt_guard`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 83.5 % vs 83.5 %; Δ +0.0 pt (bootstrap [-4.0, +4.0], Newcombe [-4.2, +4.2]); wins / losses 8 / 8
- cost $12.0 vs $9.8 per 1k episodes (saving -22.6 %); System-1 $0.06, System-2 $11.9 per 1k; LLM calls 15.18 vs 7.57, System-1 calls 1.54 per episode; wall 11.1 s vs 5.6 s (median 10.8 vs 5.6); episodes with an empty LLM reply 0 vs 0
- decision point: n_decisions 1518, n_gold 1505, coverage 0.0 %, n_proposals 309, n_vetoed 296, acted 95.8 %, acc 100.0 %, llm_store_rate 20.4 %, veto_precision 100.0 %

### memory_read × guard (`mr_guard`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 100.0 % vs 100.0 %; Δ +0.0 pt (bootstrap [+0.0, +0.0], Newcombe [-1.9, +1.9]); wins / losses 0 / 0
- cost $38.2 vs $16.0 per 1k episodes (saving -139.8 %); System-1 $0.42, System-2 $37.8 per 1k; LLM calls 16.68 vs 7.50, System-1 calls 6.50 per episode; wall 16.5 s vs 8.0 s (median 16.7 vs 9.2); episodes with an empty LLM reply 0 vs 0
- decision point: n_decisions 1300, n_gold 1300, coverage 0.0 %, n_proposals 1300, n_vetoed 536, acted 41.2 %, acc 58.7 %, llm_store_rate 0.0 %, veto_precision 30.0 %, final_covers 65.8 %

