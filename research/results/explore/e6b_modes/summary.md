# E6b: the module map's remaining cells (exploratory (E6b))

- generated 2026-10-06T07:24:54+00:00 at git `4d4efa2275d2` by `scripts/explore_e6b_modes.py`; run `runs/e6b_modes`; paired task bootstrap B = 10000, seed 0
- spend (every E6b record, smoke included): $92.74 (Gemini $91.87, Jev $0.87) of the $150 cap
- verdict rule (descriptive): costs success if delta <= -3 pt; no saving if saving <= 0; takes over safely if delta >= -1 pt, saving > 0 and acted >= 5 %; else mixed (scripts/module_map.py, fixed 2026-10-05)
- cells, semantics and thresholds: `results/explore/e6b_modes/cells.md`; registration: `docs/PREREG_A7_STAGE2.md`, Deviations "Exploratory row E6b". No test; outside every Holm table.

- reference `llm_only`: n = 200 complete, success 99.5 %, $40.7 per 1k episodes, 13.0 s per episode
- reference `small_only`: n = 200 complete, success 85.0 %, $9.8 per 1k episodes, 5.0 s per episode

| cell | arm | ref. | n | succ. % | Δ pt [95 % CI] | wins / losses | $/1k ep. | saving % [95 % CI] | coverage % | acted % | acc. % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/assist | `tp_assist` | llm_only | 200 | 99.5 | +0.0 [+0.0, +0.0] | 0 / 0 | 37.1 | 8.9 [7.7, 10.2] | 0.0 | 100.0 (shortlist) | 98.8 | takes over safely |
| tool_params/guard | `tp_guard` | llm_only | 200 | 99.5 | +0.0 [+0.0, +0.0] | 0 / 0 | 43.4 | -6.7 [-9.0, -4.7] | 0.0 | 6.3 (vetoed) | 93.8 | no saving |
| memory_write/cascade | `mw_cascade` | llm_only | 200 | 98.5 | -1.0 [-3.0, +1.0] | 1 / 3 | 60.2 | -48.1 [-52.5, -43.4] | 44.0 | 44.0 (decided) | 100.0 | no saving |
| memory_write/guard | `mw_guard` | llm_only | 200 | 99.5 | +0.0 [+0.0, +0.0] | 0 / 0 | 65.2 | -60.3 [-65.1, -55.1] | 0.0 | 0.0 (vetoed) | 96.8 | no saving |
| memory_read/cascade | `mr_cascade` | llm_only | 200 | 99.5 | +0.0 [-1.5, +1.5] | 1 / 1 | 49.0 | -20.5 [-24.0, -17.2] | 54.4 | 54.4 (decided) | 76.2 | no saving |
| memory_read/trigger | `mr_trigger` | llm_only | 200 | 99.5 | +0.0 [+0.0, +0.0] | 0 / 0 | 41.0 | -0.9 [-1.6, -0.2] | 100.0 | 0.0 (fired) | 24.8 | no saving |
| reflection/cascade | `rf_cascade` | llm_only | 200 | 100.0 | +0.5 [+0.0, +1.5] | 1 / 0 | 51.0 | -25.4 [-27.1, -23.7] | 6.4 | 6.4 (decided) | 100.0 | no saving |
| stop/cascade | `st_cascade` | llm_only | 200 | 99.5 | +0.0 [+0.0, +0.0] | 0 / 0 | 41.0 | -0.7 [-1.4, -0.1] | 94.2 | 94.2 (decided) | 100.0 | no saving |
| routing/cascade | `rt_cascade` | small_only | 200 | 81.0 | -4.0 [-8.0, +0.0] | 4 / 12 | 20.9 | -113.1 [-121.6, -104.5] | 5.5 | 5.5 (decided) | 100.0 | costs success |

Coverage: share of the point's decisions executed from the decision model's answer (cascade: answered at confidence ≥ τ; trigger: 100 %, every gate decision is the model's; assist and guard: 0 by definition). Acted: cascade = coverage; assist = 100 % (shortlist at every decision); guard = proposals vetoed; trigger = questions on which it fired. Acc.: cascade = accepted answers; trigger and guard = the model's answers; assist = gold value in the shortlist (recall@k). Routing is compared with Flash-Lite alone (as C4). Stop cascade: the LLM's own stop decision is free (its tool call), so a decided answer changes the episode only where it differs from it (`overrode`: rejected finishes and early stops). Argument guard: `veto_on_other` = vetoes of a value outside the model's candidate list (which it cannot choose), `veto_llm_was_wrong` = vetoed values that were wrong, among the vetoes on the oracle's tool.

## Per cell

### tool_params × assist (`tp_assist`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 99.5 % vs 99.5 %; Δ +0.0 pt (bootstrap [+0.0, +0.0], Newcombe [-2.1, +2.1]); wins / losses 0 / 0
- cost $37.1 vs $40.7 per 1k episodes (saving 8.9 %); System-1 $1.10, System-2 $36.0 per 1k; LLM calls 7.50 vs 7.50, System-1 calls 10.89 per episode; wall 14.8 s vs 13.0 s (median 10.3 vs 7.7)
- decision point: n_decisions 678, n_gold 670, coverage 0.0 %, acted 100.0 %, binding 65.6 %, recall_k 98.8 %, acc 98.8 %, s1_acc_all 95.2 %, llm_final_acc 92.7 %, in_shortlist 100.0 %

### tool_params × guard (`tp_guard`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 99.5 % vs 99.5 %; Δ +0.0 pt (bootstrap [+0.0, +0.0], Newcombe [-2.1, +2.1]); wins / losses 0 / 0
- cost $43.4 vs $40.7 per 1k episodes (saving -6.7 %); System-1 $0.64, System-2 $42.7 per 1k; LLM calls 7.96 vs 7.50, System-1 calls 10.67 per episode; wall 16.5 s vs 13.0 s (median 10.4 vs 7.7)
- decision point: n_decisions 2134, n_gold 1743, coverage 0.0 %, n_proposals 1306, n_vetoed 82, acted 6.3 %, acc 93.8 %, n_veto_graded 47, veto_llm_was_wrong 0.0 %, veto_on_other 100.0 %, veto_tool_changed 32.9 %

### memory_write × cascade (`mw_cascade`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 98.5 % vs 99.5 %; Δ -1.0 pt (bootstrap [-3.0, +1.0], Newcombe [-3.9, +1.5]); wins / losses 1 / 3
- cost $60.2 vs $40.7 per 1k episodes (saving -48.1 %); System-1 $0.50, System-2 $59.7 per 1k; LLM calls 11.19 vs 7.50, System-1 calls 6.54 per episode; wall 19.8 s vs 13.0 s (median 13.0 vs 7.7)
- decision point: n_decisions 1308, n_gold 1305, coverage 44.0 %, acted 44.0 %, acc 100.0 %, s1_acc_all 69.1 %, llm_acc_deferred 67.5 %, n_deferred 733

### memory_write × guard (`mw_guard`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 99.5 % vs 99.5 %; Δ +0.0 pt (bootstrap [+0.0, +0.0], Newcombe [-2.1, +2.1]); wins / losses 0 / 0
- cost $65.2 vs $40.7 per 1k episodes (saving -60.3 %); System-1 $0.22, System-2 $65.0 per 1k; LLM calls 14.03 vs 7.50, System-1 calls 2.53 per episode; wall 24.4 s vs 13.0 s (median 15.0 vs 7.7)
- decision point: n_decisions 1303, n_gold 1302, coverage 0.0 %, n_proposals 506, n_vetoed 0, acted 0.0 %, acc 96.8 %, llm_store_rate 38.8 %, veto_precision --

### memory_read × cascade (`mr_cascade`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 99.5 % vs 99.5 %; Δ +0.0 pt (bootstrap [-1.5, +1.5], Newcombe [-2.3, +2.3]); wins / losses 1 / 1
- cost $49.0 vs $40.7 per 1k episodes (saving -20.5 %); System-1 $0.44, System-2 $48.6 per 1k; LLM calls 7.55 vs 7.50, System-1 calls 3.56 per episode; wall 15.2 s vs 13.0 s (median 9.1 vs 7.7)
- decision point: n_decisions 711, n_gold 711, coverage 54.4 %, acted 54.4 %, acc 76.2 %, s1_acc_all 57.2 %, llm_acc_deferred --, n_deferred 324, mean_injected 4.49

### memory_read × trigger (`mr_trigger`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 99.5 % vs 99.5 %; Δ +0.0 pt (bootstrap [+0.0, +0.0], Newcombe [-2.1, +2.1]); wins / losses 0 / 0
- cost $41.0 vs $40.7 per 1k episodes (saving -0.9 %); System-1 $0.34, System-2 $40.7 per 1k; LLM calls 7.50 vs 7.50, System-1 calls 6.50 per episode; wall 14.2 s vs 13.0 s (median 9.2 vs 7.7)
- decision point: n_decisions 1300, n_gold 1300, coverage 100.0 %, acted 0.0 %, acc 24.8 %, gold_positive_rate 75.2 %

### reflection × cascade (`rf_cascade`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 100.0 % vs 99.5 %; Δ +0.5 pt (bootstrap [+0.0, +1.5], Newcombe [-1.4, +2.8]); wins / losses 1 / 0
- cost $51.0 vs $40.7 per 1k episodes (saving -25.4 %); System-1 $0.34, System-2 $50.7 per 1k; LLM calls 13.74 vs 7.50, System-1 calls 6.53 per episode; wall 22.2 s vs 13.0 s (median 14.9 vs 7.7)
- decision point: n_decisions 1306, n_gold 1286, coverage 6.4 %, acted 6.4 %, acc 100.0 %, s1_acc_all 77.6 %, llm_acc_deferred 100.0 %, n_deferred 1223

### stop × cascade (`st_cascade`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 99.5 % vs 99.5 %; Δ +0.0 pt (bootstrap [+0.0, +0.0], Newcombe [-2.1, +2.1]); wins / losses 0 / 0
- cost $41.0 vs $40.7 per 1k episodes (saving -0.7 %); System-1 $0.35, System-2 $40.6 per 1k; LLM calls 7.50 vs 7.50, System-1 calls 7.50 per episode; wall 13.6 s vs 13.0 s (median 9.5 vs 7.7)
- decision point: n_decisions 1500, n_gold 1500, coverage 94.2 %, acted 94.2 %, acc 100.0 %, s1_acc_all 98.9 %, llm_acc_deferred 100.0 %, n_deferred 87, finish_rejected 0, early_stops 0, n_finish_proposals 199, overrode 0.0 %

### routing × cascade (`rt_cascade`)

- n = 200 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 81.0 % vs 85.0 %; Δ -4.0 pt (bootstrap [-8.0, +0.0], Newcombe [-8.2, +0.1]); wins / losses 4 / 12
- cost $20.9 vs $9.8 per 1k episodes (saving -113.1 %); System-1 $0.36, System-2 $20.5 per 1k; LLM calls 14.73 vs 7.57, System-1 calls 7.58 per episode; wall 11.2 s vs 5.0 s (median 12.0 vs 5.2)
- decision point: n_decisions 1515, n_gold 1087, coverage 5.5 %, acted 5.5 %, acc 100.0 %, s1_acc_all 90.6 %, llm_acc_deferred 96.2 %, n_deferred 1432
- vs Flash alone (`llm_only`, descriptive): Δ -18.5 pt, saving 48.7 %, wins / losses 1 / 38

