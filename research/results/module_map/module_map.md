# Module map (descriptive)

- generated 2026-10-07T18:37:23+00:00 at git `4d2e61d631c0` by `scripts/module_map.py`; B = 10000, seed 0; runs: E2a `runs/e2a_jevgym_tool_decisions`, E2b `runs/e2b_jevgym_meta_control`, E2c `runs/e2c_jevgym_baselines`; Stage 2 (separate column): `runs/stage2_jevgym`
- verdict rule (fixed before the numbers): costs success if Δ ≤ −3 pt; no saving if saving ≤ 0; takes over safely if Δ ≥ −1 pt, saving > 0 and the model acted at ≥ 5 % of the point's opportunities; else mixed. Registered cells keep their registered outcome (label: word).
- flags: none

## Verdict grid

```
Flash (routing: Flash-Lite agent handing steps to Flash, vs. Flash-Lite alone)
module         REPLACE              CASCADE              ASSIST               GUARD                TRIGGER              | module verdict (best shown cell)
tool choice    COSTS -19.0/67%      SAFE +0.0/14% C1F    COSTS -7.8/54% C2F   no-sav +0.0/-0%      not run              | takes over safely (cascade, jev_ts_cascade_tauhat; registered C1F: holds)
tool arguments COSTS -20.0/88%      SAFE +0.0/14% C1F    not run              not run              not run              | takes over safely (cascade, jev_ts_cascade_tauhat; registered C1F: holds)
memory write   not run (=trigger)   not run              not run              not run              no-sav -1.0/-39%     | no saving (trigger, jev_memw)
memory read    not run (=assist)    not run              no-sav +0.0/-5%      not run              not run              | no saving (assist, jev_mem)
reflection     not run (=trigger)   not run              not run              not run              SAFE +0.0/1%*        | takes over safely (trigger, jev_reflect_trigger)
stop           not run (=cascade)   not run              not run              no-sav -0.6/-6% C6F  no-sav +0.0/-0%      | no saving (trigger, jev_stop_trigger)
routing        not run (=trigger)   not run              not run              not run              no-sav +0.6/-36% C4  | no saving (trigger, jev_route_tau0.4; registered C4: not established)
action         not run              not run              not run              not run              not run              | not run

Flash-Lite (routing: Flash-Lite agent handing steps to Flash, vs. Flash-Lite alone)
module         REPLACE              CASCADE              ASSIST               GUARD                TRIGGER              | module verdict (best shown cell)
tool choice    not run              mixed +1.0/0% C1L    COSTS -3.4/52% C2L   no-sav +4.5/-7%      not run              | mixed (cascade, jev_ts_cascade_tau0.9_s2lite; registered C1L: not established)
tool arguments not run              mixed +1.0/0% C1L    not run              not run              not run              | mixed (cascade, jev_ts_cascade_tau0.9_s2lite; registered C1L: not established)
memory write   not run (=trigger)   not run              not run              not run              not run              | not run
memory read    not run (=assist)    not run              not run              not run              not run              | not run
reflection     not run (=trigger)   not run              not run              not run              no-sav +1.8/-2% C5   | no saving (trigger, jev_reflect_trigger_s2lite; registered C5: not established)
stop           not run (=cascade)   not run              not run              no-sav +1.6/-32% C6L not run              | no saving (guard, jev_stop_guard_s2lite; registered C6L: not established)
routing        not run (=trigger)   not run              not run              not run              no-sav +0.6/-36% C4  | no saving (trigger, jev_route_tau0.4; registered C4: not established)
action         not run              not run              not run              not run              not run              | not run

cell: verdict delta(pt)/saving(%) [registered label]; * = a replicate or the fresh-task run of the same configuration gave a verdict of another colour class (mixed and no saving share one)
exploratory E6b / E6d / E6e / E6f cells (runs/e6b_modes, runs/e6d_modes, runs/e6e_modes, runs/e6f_modes, runs/e6f_alfworld_eval, fresh JevGym seeds; not Stage 1): action/trigger no-sav -2.1/-111.2%; memory_read/cascade no-sav +0.0/-20.5%; memory_read/guard no-sav +0.0/-139.8%; memory_read/replace no-sav +0.5/-11.1%; memory_read/trigger no-sav +0.0/-0.9%; memory_write/assist no-sav +0.0/-74.2%; memory_write/cascade no-sav -1.0/-48.1%; memory_write/guard no-sav +0.0/-60.3%; memory_write/replace no-sav +0.5/-47.4%; reflection/assist no-sav -0.5/-65.7%; reflection/cascade no-sav +0.5/-25.4%; reflection/guard no-sav +0.0/-61.2%; reflection/replace no-sav -0.5/-7.2%; routing/assist no-sav -2.0/-31.0%; routing/cascade COSTS -4.0/-113.1%; routing/guard no-sav +0.0/-22.6%; routing/replace no-sav +4.0/-6.7%; stop/assist no-sav +0.0/-16.7%; stop/cascade no-sav +0.0/-0.7%; stop/replace no-sav +0.5/-6.8%; tool_params/assist SAFE +0.0/8.9%; tool_params/guard no-sav +0.0/-6.7%; tool_params/trigger no-sav +0.0/-1.2%; tool_select/trigger no-sav +0.0/-14.2%
undefined / equivalent cells (harness.config.UNDEFINED_CELLS): none (E6f defined the last three)
implemented with the rule of another cell (harness.config.EQUIVALENT_NOTES; run as their own arms by E6e): memory_write/replace = trigger at tau = 0.5; memory_read/replace = assist with k = 1; reflection/replace = trigger at tau = 0.5; stop/replace = cascade at tau = 0.5; routing/replace = trigger at tau = 0.5
exploratory action row (E6a, E6c, ALFWorld valid_seen; not Stage 1): flash/action/assist COSTS -40.7/-182.0%; flash/action/cascade SAFE +0.0/14.0%; flash/action/guard no-sav -0.7/-86.7%; flash/action/replace COSTS -64.3/90.7%
```

## Exploratory action row (E6a, E6c: ALFWorld valid_seen, `runs/e6a_alfworld_eval, runs/e6c_alfworld_eval`; not Stage 1)

Descriptive, outside every Holm table, not the registered S2-L3b (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-06). Same comparison and verdict rule as the Stage-1 rows; never counted in the grid above.

| cell | arm | setting | n | succ. % | Δ pt [95 % CI] | wins / losses | $/1k ep. | saving % [95 % CI] | cov. % | acted % | acc. % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| action/assist (E6a) | `jev_action_assist_k3` | action assist k=3 | 140 | 50.7 | -40.7 [-49.3, -32.1] | 1 / 58 | 52.0 | -182.0 [-268.6, -122.1] | -- | 100.0 | 23.6 | costs success |
| action/cascade (E6a) | `jev_action_cascade_tauhat` | action cascade tau=0.95 | 140 | 91.4 | +0.0 [-4.3, +4.3] | 4 / 4 | 15.9 | 14.0 [-3.0, 28.3] | 23.3 | 23.3 | 73.5 | takes over safely |
| action/replace (E6a) | `jev_action_replace` | action replace | 140 | 27.1 | -64.3 [-72.1, -56.4] | 0 / 90 | 1.7 | 90.7 [87.9, 92.6] | 100.0 | 100.0 | 17.7 | costs success |
| action/guard (E6c) | `jev_action_guard` | action guard tau=0.5 | 140 | 90.0 | -0.7 [-4.3, +2.9] | 3 / 4 | 35.7 | -86.7 [-125.6, -55.7] | -- | 46.5 | 65.5 | no saving |

## Exploratory cells E6b / E6d / E6e / E6f (`runs/e6b_modes, runs/e6d_modes, runs/e6e_modes, runs/e6f_modes, runs/e6f_alfworld_eval`: fresh JevGym seeds 300800-300999 / 301000-301399 / 301400-301599 / 301600-301799, and ALFWorld valid_seen for E6f's action trigger, Flash; not Stage 1)

Descriptive, outside every Holm table (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-06 and 2026-10-07; semantics: results/explore/e6{b,d,e,f}_modes/cells.md; full readouts: results/explore/e6{b,d,e,f}_modes/summary.md). Same comparison and verdict rule as the Stage-1 rows; never counted in the grid above. An E6e replace cell behaves as the cell after its "=" (harness.config.EQUIVALENT_NOTES) and is still run as its own arm.

| cell | row | arm | reference | n | Δ pt | wins / losses | saving % | cov. % | acted % | acc. % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| action/trigger | E6f | `jev_action_trigger` | llm_only | 140 | -2.1 | 3 / 6 | -111.2 | 100.0 | 74.9 | 67.1 | no saving |
| memory_read/cascade | E6b | `mr_cascade` | llm_only | 200 | +0.0 | 1 / 1 | -20.5 | 54.4 | 54.4 | 76.2 | no saving |
| memory_read/guard | E6d | `mr_guard` | llm_only | 200 | +0.0 | 0 / 0 | -139.8 | 0.0 | 41.2 | 58.7 | no saving |
| memory_read/replace (= assist with k = 1) | E6e | `mr_replace` | llm_only | 200 | +0.5 | 1 / 0 | -11.1 | 100.0 | 100.0 | 57.2 | no saving |
| memory_read/trigger | E6b | `mr_trigger` | llm_only | 200 | +0.0 | 0 / 0 | -0.9 | 100.0 | 0.0 | 24.8 | no saving |
| memory_write/assist | E6d | `mw_assist` | llm_only | 200 | +0.0 | 0 / 0 | -74.2 | 0.0 | 100.0 | 69.3 | no saving |
| memory_write/cascade | E6b | `mw_cascade` | llm_only | 200 | -1.0 | 1 / 3 | -48.1 | 44.0 | 44.0 | 100.0 | no saving |
| memory_write/guard | E6b | `mw_guard` | llm_only | 200 | +0.0 | 0 / 0 | -60.3 | 0.0 | 0.0 | 96.8 | no saving |
| memory_write/replace (= trigger at tau = 0.5) | E6e | `mw_replace` | llm_only | 200 | +0.5 | 1 / 0 | -47.4 | 100.0 | 100.0 | 69.4 | no saving |
| reflection/assist | E6d | `rf_assist` | llm_only | 200 | -0.5 | 0 / 1 | -65.7 | 0.0 | 100.0 | 69.0 | no saving |
| reflection/cascade | E6b | `rf_cascade` | llm_only | 200 | +0.5 | 1 / 0 | -25.4 | 6.4 | 6.4 | 100.0 | no saving |
| reflection/guard | E6d | `rf_guard` | llm_only | 200 | +0.0 | 0 / 0 | -61.2 | 0.0 | 90.9 | 100.0 | no saving |
| reflection/replace (= trigger at tau = 0.5) | E6e | `rf_replace` | llm_only | 200 | -0.5 | 0 / 1 | -7.2 | 100.0 | 100.0 | 99.9 | no saving |
| routing/assist | E6d | `rt_assist` | small_only | 200 | -2.0 | 7 / 11 | -31.0 | 0.0 | 100.0 | 92.8 | no saving |
| routing/cascade | E6b | `rt_cascade` | small_only | 200 | -4.0 | 4 / 12 | -113.1 | 5.5 | 5.5 | 100.0 | costs success |
| routing/guard | E6d | `rt_guard` | small_only | 200 | +0.0 | 8 / 8 | -22.6 | 0.0 | 95.8 | 100.0 | no saving |
| routing/replace (= trigger at tau = 0.5) | E6e | `rt_replace` | small_only | 200 | +4.0 | 11 / 3 | -6.7 | 100.0 | 100.0 | 97.8 | no saving |
| stop/assist | E6d | `st_assist` | llm_only | 200 | +0.0 | 0 / 0 | -16.7 | 0.0 | 100.0 | 99.3 | no saving |
| stop/cascade | E6b | `st_cascade` | llm_only | 200 | +0.0 | 0 / 0 | -0.7 | 94.2 | 94.2 | 100.0 | no saving |
| stop/replace (= cascade at tau = 0.5) | E6e | `st_replace` | llm_only | 200 | +0.5 | 1 / 0 | -6.8 | 100.0 | 100.0 | 98.1 | no saving |
| tool_params/assist | E6b | `tp_assist` | llm_only | 200 | +0.0 | 0 / 0 | 8.9 | 0.0 | 100.0 | 98.8 | takes over safely |
| tool_params/guard | E6b | `tp_guard` | llm_only | 200 | +0.0 | 0 / 0 | -6.7 | 0.0 | 6.3 | 93.8 | no saving |
| tool_params/trigger | E6f | `tp_trigger` | llm_only | 100 | +0.0 | 0 / 0 | -1.2 | 100.0 | 0.0 | 100.0 | no saving |
| tool_select/trigger | E6f | `ts_trigger` | llm_only | 100 | +0.0 | 0 / 0 | -14.2 | 100.0 | 13.7 | 86.3 | no saving |

## Flash

| cell | row | run | model | setting | status | n | succ. % | Δ pt [95 % CI] | $/1k ep. | saving % [95 % CI] | cov. % | acted % | acc. % | verdict | fresh: n, Δ, saving, verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | `jev_ts_replace` | E2a | Jev | tool_select replace | descriptive | 200 | 81.0 | -19.0 [-24.5, -13.5] | 14.2 | 67.0 [65.1, 69.1] | 100.0 | 100.0 (decided) | 95.4 | costs success | -- |
| tool_select/replace | `jev_ts_replace-e2c` | E2c | Jev | tool_select replace | replicate of jev_ts_replace@E2a | 200 | 81.5 | -18.5 [-24.0, -13.0] | 14.2 | 65.4 [63.4, 67.4] | 100.0 | 100.0 (decided) | 95.5 | costs success | -- |
| tool_select/replace | `lite_ts_replace` | E2c | Flash-Lite | tool_select replace | descriptive | 200 | 41.5 | -58.5 [-65.0, -51.5] | 24.2 | 40.9 [36.5, 45.3] | 100.0 | 100.0 (decided) | 82.7 | costs success | -- |
| tool_select/replace | `embed_ts_replace` | E2c | embedding | tool_select replace | descriptive | 200 | 0.0 | -100.0 [-100.0, -100.0] | 8.7 | 78.7 [70.0, 86.3] | 100.0 | 100.0 (decided) | 4.2 | costs success | -- |
| tool_select/replace | `bm25_ts_replace` | E2c | BM25 | tool_select replace | descriptive | 200 | 12.5 | -87.5 [-92.0, -83.0] | 15.0 | 63.3 [53.8, 72.1] | 100.0 | 100.0 (decided) | 8.4 | costs success | -- |
| tool_select/replace | `tfidf_ts_replace` | E2c | TF-IDF | tool_select replace | descriptive | 200 | 12.5 | -87.5 [-92.0, -83.0] | 18.7 | 54.4 [39.1, 68.0] | 100.0 | 100.0 (decided) | 10.0 | costs success | -- |
| tool_select/replace | `heuristic_ts_replace` | E2c | heuristic | tool_select replace | descriptive | 200 | 12.5 | -87.5 [-92.0, -83.0] | 1.7 | 95.7 [95.5, 95.9] | 100.0 | 100.0 (decided) | 35.3 | costs success | -- |
| tool_select/replace | `prior_recent_ts_replace` | E2c | prior (most recent) | tool_select replace | descriptive | 200 | 0.0 | -100.0 [-100.0, -100.0] | 40.4 | 1.3 [-11.9, 13.9] | 100.0 | 100.0 (decided) | 0.0 | costs success | -- |
| tool_select/replace | `random_ts_replace` | E2c | random | tool_select replace | descriptive | 200 | 0.0 | -100.0 [-100.0, -100.0] | 3.0 | 92.7 [91.2, 94.0] | 100.0 | 100.0 (decided) | 2.9 | costs success | -- |
| tool_select/cascade, tool_params/cascade (joint) | `jev_ts_cascade_tau0.5` | E2a | Jev | tool_select cascade tau=0.5; tool_params cascade tau=0.9 | descriptive | 200 | 78.5 | -21.5 [-27.5, -16.0] | 6.6 | 84.5 [82.0, 86.4] | 97.3 | 97.3 (decided) | 93.7 | costs success | -- |
| tool_select/cascade, tool_params/cascade (joint) | `jev_ts_cascade_tau0.7` | E2a | Jev | tool_select cascade tau=0.7; tool_params cascade tau=0.9 | descriptive | 200 | 76.0 | -24.0 [-30.0, -18.0] | 8.2 | 81.0 [79.5, 82.1] | 91.3 | 91.3 (decided) | 96.1 | costs success | -- |
| tool_select/cascade, tool_params/cascade (joint) | `jev_ts_cascade_tau0.8` | E2a | Jev | tool_select cascade tau=0.8; tool_params cascade tau=0.9 | descriptive | 200 | 76.5 | -23.5 [-29.5, -18.0] | 10.1 | 76.5 [74.5, 78.1] | 86.5 | 86.5 (decided) | 96.3 | costs success | -- |
| tool_select/cascade, tool_params/cascade (joint) | `jev_ts_cascade_tau0.9` | E2a | Jev | tool_select cascade tau=0.9; tool_params cascade tau=0.9 | descriptive | 200 | 79.5 | -20.5 [-26.5, -15.0] | 12.0 | 72.0 [70.3, 73.5] | 81.3 | 81.3 (decided) | 96.4 | costs success | -- |
| tool_select/cascade, tool_params/cascade (joint) | `jev_ts_cascade_tau0.95` | E2a | Jev | tool_select cascade tau=0.95; tool_params cascade tau=0.9 | descriptive | 200 | 79.5 | -20.5 [-26.0, -15.0] | 14.3 | 66.8 [64.5, 69.0] | 74.3 | 74.3 (decided) | 96.1 | costs success | -- |
| tool_select/cascade, tool_params/cascade (joint) | `jev_ts_cascade_tau0.99` | E2a | Jev | tool_select cascade tau=0.99; tool_params cascade tau=0.9 | descriptive | 200 | 92.0 | -8.0 [-12.0, -4.5] | 26.1 | 39.3 [35.7, 43.0] | 45.1 | 45.1 (decided) | 97.6 | costs success | -- |
| tool_select/cascade, tool_params/cascade (joint) | `jev_ts_cascade_tauhat` | E2a | Jev | tool_select cascade tau=1.0; tool_params cascade tau=0.99 | registered C1F: holds | 500 | 99.8 | +0.0 [+0.0, +0.0] | 36.4 | 14.2 [12.7, 15.7] | 27.2 | 27.2 (decided) | 100.0 | takes over safely | 500, +0.4, 18.8, takes over safely |
| tool_select/cascade, tool_params/cascade (joint) | `lite_ts_cascade_tau0.7` | E2a | Flash-Lite | tool_select cascade tau=0.7; tool_params cascade tau=0.9 | descriptive | 200 | 42.5 | -57.5 [-64.0, -50.5] | 26.7 | 37.9 [32.6, 42.9] | 99.9 | 99.9 (decided) | 80.6 | costs success | -- |
| tool_select/cascade, tool_params/cascade (joint) | `lite_ts_cascade_tau0.9` | E2a | Flash-Lite | tool_select cascade tau=0.9; tool_params cascade tau=0.9 | descriptive | 200 | 43.5 | -56.5 [-63.5, -49.5] | 26.3 | 38.8 [34.7, 42.9] | 98.3 | 98.3 (decided) | 82.7 | costs success | -- |
| tool_select/cascade, tool_params/cascade (joint) | `lite_ts_cascade_tau0.99` | E2a | Flash-Lite | tool_select cascade tau=0.99; tool_params cascade tau=0.9 | descriptive | 200 | 75.0 | -25.0 [-31.0, -19.0] | 43.2 | -0.7 [-4.8, 3.5] | 50.0 | 50.0 (decided) | 89.7 | costs success | -- |
| tool_select/assist | `jev_ts_assist_k3` | E2a | Jev | tool_select assist k=3 | registered C2F: not established, C3: holds | 500 | 92.0 | -7.8 [-10.2, -5.4] | 19.6 | 53.7 [52.8, 54.6] | -- | 100.0 (shortlist) | 94.6 | costs success | -- |
| tool_select/assist | `jev_ts_assist_k3-e2c` | E2c | Jev | tool_select assist k=3 | replicate of jev_ts_assist_k3@E2a | 200 | 90.5 | -9.5 [-14.0, -5.5] | 19.5 | 52.3 [50.7, 53.8] | -- | 100.0 (shortlist) | 94.1 | costs success | -- |
| tool_select/assist | `jev_ts_assist_k5` | E2a | Jev | tool_select assist k=5 | descriptive | 200 | 90.0 | -10.0 [-14.5, -6.0] | 22.2 | 48.3 [46.3, 50.1] | -- | 100.0 (shortlist) | 96.2 | costs success | -- |
| tool_select/assist | `jev_ts_assist_k5-e2c` | E2c | Jev | tool_select assist k=5 | replicate of jev_ts_assist_k5@E2a | 200 | 92.0 | -8.0 [-12.0, -4.5] | 22.4 | 45.2 [42.8, 47.3] | -- | 100.0 (shortlist) | 96.4 | costs success | -- |
| tool_select/assist | `lite_ts_assist_k3` | E2a | Flash-Lite | tool_select assist k=3 | descriptive | 200 | 87.5 | -12.5 [-17.5, -8.0] | 36.7 | 14.6 [11.5, 17.8] | -- | 100.0 (shortlist) | 89.3 | costs success | -- |
| tool_select/assist | `lite_ts_assist_k3-e2c` | E2c | Flash-Lite | tool_select assist k=3 | replicate of lite_ts_assist_k3@E2a | 200 | 86.5 | -13.5 [-18.5, -9.0] | 36.8 | 10.2 [6.5, 13.8] | -- | 100.0 (shortlist) | 88.0 | costs success | -- |
| tool_select/assist | `embed_ts_assist_k3` | E2a | embedding | tool_select assist k=3 | descriptive | 200 | 28.0 | -72.0 [-78.0, -65.5] | 9.2 | 78.5 [75.2, 81.6] | -- | 100.0 (shortlist) | 25.4 | costs success | -- |
| tool_select/assist | `embed_ts_assist_k3-e2c` | E2c | embedding | tool_select assist k=3 | replicate of embed_ts_assist_k3@E2a | 200 | 30.0 | -70.0 [-76.0, -63.5] | 9.7 | 76.2 [72.4, 79.6] | -- | 100.0 (shortlist) | 24.8 | costs success | -- |
| tool_select/assist | `bm25_ts_assist_k3` | E2c | BM25 | tool_select assist k=3 | registered C3 (reference): holds | 200 | 17.0 | -83.0 [-88.0, -77.5] | 15.3 | 62.7 [53.8, 70.6] | -- | 100.0 (shortlist) | 16.2 | costs success | -- |
| tool_select/assist | `bm25_ts_assist_k5` | E2c | BM25 | tool_select assist k=5 | descriptive | 200 | 18.5 | -81.5 [-86.5, -76.0] | 16.3 | 60.1 [51.7, 67.7] | -- | 100.0 (shortlist) | 18.7 | costs success | -- |
| tool_select/assist | `bm25task_ts_assist_k3` | E2c | BM25 (task+obs.) | tool_select assist k=3 | descriptive | 200 | 24.5 | -75.5 [-81.5, -69.5] | 15.0 | 63.5 [56.9, 69.4] | -- | 100.0 (shortlist) | 10.7 | costs success | -- |
| tool_select/assist | `tfidf_ts_assist_k3` | E2c | TF-IDF | tool_select assist k=3 | descriptive | 200 | 33.0 | -67.0 [-73.5, -60.5] | 13.4 | 67.2 [61.7, 72.1] | -- | 100.0 (shortlist) | 28.8 | costs success | -- |
| tool_select/assist | `tfidf_ts_assist_k5` | E2c | TF-IDF | tool_select assist k=5 | descriptive | 200 | 25.0 | -75.0 [-81.0, -69.0] | 24.3 | 40.6 [30.9, 49.9] | -- | 100.0 (shortlist) | 25.6 | costs success | -- |
| tool_select/assist | `heuristic_ts_assist_k3` | E2c | heuristic | tool_select assist k=3 | descriptive | 200 | 31.0 | -69.0 [-75.5, -62.5] | 18.1 | 55.8 [46.9, 63.8] | -- | 100.0 (shortlist) | 13.7 | costs success | -- |
| tool_select/assist | `random_ts_assist_k3` | E2c | random | tool_select assist k=3 | descriptive | 200 | 6.0 | -94.0 [-97.0, -90.5] | 35.8 | 12.5 [0.8, 23.8] | -- | 100.0 (shortlist) | 6.5 | costs success | -- |
| tool_select/assist | `s2self_ts_assist_k3` | E2c | the agent LLM (typed question) | tool_select assist k=3 | descriptive | 200 | 94.5 | -5.5 [-9.0, -2.5] | 107.2 | -161.9 [-170.8, -153.8] | -- | 100.0 (shortlist) | 92.8 | costs success | -- |
| tool_select/guard | `jev_ts_guard` | E2a | Jev | tool_select guard tau=0.5 | descriptive | 200 | 100.0 | +0.0 [+0.0, +0.0] | 43.0 | -0.2 [-1.3, 0.9] | -- | 2.1 (vetoed) | 98.0 | no saving | -- |
| tool_select/guard | `lite_ts_guard` | E2a | Flash-Lite | tool_select guard tau=0.5 | descriptive | 200 | 100.0 | +0.0 [+0.0, +0.0] | 53.6 | -24.9 [-27.9, -22.0] | -- | 11.0 (vetoed) | 89.6 | no saving | -- |
| tool_params/replace, tool_select/replace (joint) | `jev_ts_params_replace` | E2a | Jev | tool_select replace; tool_params replace | descriptive | 200 | 80.0 | -20.0 [-25.5, -14.5] | 5.1 | 88.1 [85.6, 90.0] | 85.7 | 85.7 (decided) | 98.2 | costs success | -- |
| tool_params/replace, tool_select/replace (joint) | `jev_ts_params_replace_answercopy` | E2c | Jev | tool_select replace; tool_params replace (answer as copy) | descriptive | 200 | 80.0 | -20.0 [-25.5, -14.5] | 5.5 | 86.6 [83.4, 89.1] | 84.9 | 84.9 (decided) | 98.2 | costs success | -- |
| tool_params/replace, tool_select/replace (joint) | `lite_ts_params_replace` | E2a | Flash-Lite | tool_select replace; tool_params replace | descriptive | 200 | 39.5 | -60.5 [-67.5, -53.5] | 27.1 | 36.9 [30.7, 42.7] | 85.9 | 85.9 (decided) | 91.8 | costs success | -- |
| memory_write/trigger | `jev_memw` | E2c | Jev | memory_write trigger tau=0.5; memory_read always | descriptive | 200 | 99.0 | -1.0 [-2.5, +0.0] | 56.9 | -38.9 [-42.2, -35.5] | -- | 99.5 (fired) | 69.2 | no saving | -- |
| memory_write/trigger | `heuristic_memw` | E2c | heuristic | memory_write trigger tau=0.5; memory_read always | descriptive | 200 | 97.5 | -2.5 [-5.0, -0.5] | 51.7 | -26.3 [-30.1, -22.7] | -- | 92.2 (fired) | 63.9 | no saving | -- |
| memory_write/trigger | `ratem_memw` | E2c | random, matched rate | memory_write trigger tau=0.5; memory_read always | descriptive | 200 | 98.5 | -1.5 [-3.5, +0.0] | 55.6 | -35.8 [-40.0, -31.9] | -- | 99.4 (fired) | 67.4 | no saving | -- |
| memory_write/always | `mem_always` | E2b | no decision model | memory_write always; memory_read always | descriptive | 200 | 98.0 | -2.0 [-4.0, -0.5] | 55.4 | -36.3 [-44.0, -29.7] | -- | -- (-) | -- | -- | -- |
| memory_write/llm | `mem_llm_gate` | E2b | no decision model | memory_write llm; memory_read always | descriptive | 200 | 100.0 | +0.0 [+0.0, +0.0] | 64.8 | -59.6 [-64.2, -54.5] | -- | -- (-) | -- | -- | -- |
| memory_read/assist, memory_write/trigger (joint) | `jev_mem` | E2b | Jev | memory_write trigger tau=0.5; memory_read assist k=3 | descriptive | 200 | 100.0 | +0.0 [+0.0, +0.0] | 42.6 | -4.8 [-6.2, -3.3] | -- | 100.0 (shortlist) | 57.6 | no saving | -- |
| reflection/trigger | `jev_reflect_trigger` | E2b | Jev | reflection trigger tau=0.5 | descriptive | 200 | 100.0 | +0.0 [+0.0, +0.0] | 40.3 | 0.9 [0.0, 1.7] | -- | 13.9 (fired) | 100.0 | takes over safely | 200, -1.0, -1.0, no saving |
| reflection/trigger | `jev_reflect_trigger-e2c` | E2c | Jev | reflection trigger tau=0.5 | replicate of jev_reflect_trigger@E2b | 200 | 100.0 | +0.0 [+0.0, +0.0] | 42.9 | -4.8 [-5.8, -3.8] | -- | 14.1 (fired) | 100.0 | no saving | -- |
| reflection/trigger | `heuristic_reflect_trigger` | E2c | heuristic | reflection trigger tau=0.5 | descriptive | 200 | 100.0 | +0.0 [+0.0, +0.0] | 40.7 | 0.6 [-0.1, 1.3] | -- | 0.1 (fired) | 100.0 | mixed | -- |
| reflection/trigger | `rate10_reflect_trigger` | E2c | random, 10 % | reflection trigger tau=0.5 | descriptive | 200 | 100.0 | +0.0 [+0.0, +0.0] | 42.3 | -3.4 [-4.6, -2.3] | -- | 8.7 (fired) | 100.0 | no saving | -- |
| reflection/trigger | `rate30_reflect_trigger` | E2c | random, 30 % | reflection trigger tau=0.5 | descriptive | 200 | 99.0 | -1.0 [-2.5, +0.0] | 46.1 | -12.5 [-15.0, -10.3] | -- | 30.2 (fired) | 99.8 | no saving | -- |
| reflection/trigger | `rate50_reflect_trigger` | E2c | random, 50 % | reflection trigger tau=0.5 | descriptive | 200 | 98.5 | -1.5 [-3.5, +0.0] | 50.7 | -24.0 [-27.2, -20.7] | -- | 49.5 (fired) | 99.7 | no saving | -- |
| reflection/trigger | `s2self_reflect_trigger` | E2c | the agent LLM (typed question) | reflection trigger tau=0.5 | descriptive | 200 | 100.0 | +0.0 [+0.0, +0.0] | 52.8 | -29.0 [-30.4, -27.6] | -- | 2.4 (fired) | 100.0 | no saving | -- |
| reflection/always | `reflect_always` | E2b | no decision model | reflection always | descriptive | 200 | 96.0 | -4.0 [-7.0, -1.5] | 73.1 | -80.0 [-87.9, -73.1] | -- | -- (-) | -- | -- | -- |
| reflection/always | `reflect_always-e2c` | E2c | no decision model | reflection always | replicate of reflect_always@E2b | 200 | 97.5 | -2.5 [-5.0, -0.5] | 67.7 | -65.3 [-70.9, -59.6] | -- | -- (-) | -- | -- | -- |
| stop/guard | `jev_stop_guard` | E2b | Jev | stop guard tau=0.5 | registered C6F: holds | 500 | 99.2 | -0.6 [-1.4, +0.0] | 43.3 | -6.2 [-9.1, -4.0] | -- | 17.3 (vetoed) | 82.7 | no saving | -- |
| stop/guard | `jev_stop_guard-e2c` | E2c | Jev | stop guard tau=0.5 | replicate of jev_stop_guard@E2b | 200 | 100.0 | +0.0 [+0.0, +0.0] | 42.1 | -2.9 [-4.4, -1.6] | -- | 10.7 (vetoed) | 89.3 | no saving | -- |
| stop/guard | `lite_stop_guard` | E2b | Flash-Lite | stop guard tau=0.5 | descriptive | 200 | 99.5 | -0.5 [-1.5, +0.0] | 42.4 | -4.5 [-9.8, -0.6] | -- | 20.7 (vetoed) | 79.3 | no saving | -- |
| stop/guard | `heuristic_stop_guard` | E2c | heuristic | stop guard tau=0.5 | descriptive | 200 | 100.0 | +0.0 [+0.0, +0.0] | 40.5 | 1.1 [0.4, 1.8] | -- | 0.0 (vetoed) | 100.0 | mixed | -- |
| stop/guard | `rate50_stop_guard` | E2c | random, 50 % | stop guard tau=0.5 | descriptive | 200 | 100.0 | +0.0 [+0.0, +0.0] | 46.4 | -13.2 [-16.0, -10.7] | -- | 48.1 (vetoed) | 51.9 | no saving | -- |
| stop/trigger | `jev_stop_trigger` | E2b | Jev | stop trigger tau=0.9 | descriptive | 200 | 100.0 | +0.0 [+0.0, +0.0] | 40.6 | -0.1 [-0.7, 0.6] | -- | 0.0 (fired) | 100.0 | no saving | -- |
| stop/trigger | `jev_stop_trigger-e2c` | E2c | Jev | stop trigger tau=0.9 | replicate of jev_stop_trigger@E2b | 200 | 100.0 | +0.0 [+0.0, +0.0] | 40.6 | 0.8 [0.2, 1.5] | -- | 0.0 (fired) | 100.0 | mixed | -- |
| stop/trigger | `heuristic_stop_trigger` | E2c | heuristic | stop trigger tau=0.9 | descriptive | 200 | 50.0 | -50.0 [-57.0, -43.0] | 31.1 | 24.1 [20.7, 27.4] | -- | 9.9 (fired) | 90.1 | costs success | -- |
| stop/trigger | `rate10_stop_trigger` | E2c | random, 10 % | stop trigger tau=0.9 | descriptive | 200 | 50.5 | -49.5 [-56.5, -42.5] | 29.2 | 28.7 [24.0, 33.2] | -- | 10.4 (fired) | 89.6 | costs success | -- |

## Flash-Lite

| cell | row | run | model | setting | status | n | succ. % | Δ pt [95 % CI] | $/1k ep. | saving % [95 % CI] | cov. % | acted % | acc. % | verdict | fresh: n, Δ, saving, verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/cascade, tool_params/cascade (joint) | `jev_ts_cascade_tau0.9_s2lite` | E2a | Jev | tool_select cascade tau=0.99 (always defer); tool_params cascade tau=0.13 | registered C1L: not established | 500 | 81.2 | +1.0 [-1.0, +3.2] | 9.8 | 0.2 [-0.3, 0.7] | 0.0 | 0.0 (decided) | -- | mixed | -- |
| tool_select/assist | `jev_ts_assist_k3_s2lite` | E2a | Jev | tool_select assist k=3 | registered C2L: not established | 500 | 76.8 | -3.4 [-7.2, +0.6] | 4.7 | 52.0 [51.2, 52.7] | -- | 100.0 (shortlist) | 94.8 | costs success | -- |
| tool_select/assist | `bm25_ts_assist_k3_s2lite` | E2a | BM25 | tool_select assist k=3 | descriptive | 200 | 12.5 | -69.5 [-76.0, -63.0] | 2.1 | 79.0 [73.8, 83.6] | -- | 100.0 (shortlist) | 18.8 | costs success | -- |
| tool_select/guard | `jev_ts_guard_s2lite` | E2a | Jev | tool_select guard tau=0.5 | descriptive | 200 | 86.5 | +4.5 [+0.0, +9.0] | 10.4 | -6.7 [-8.1, -5.5] | -- | 3.0 (vetoed) | 97.8 | no saving | -- |
| reflection/trigger | `jev_reflect_trigger_s2lite` | E2c | Jev | reflection trigger tau=0.5 | registered C5: not established | 500 | 83.0 | +1.8 [-1.4, +5.0] | 9.9 | -1.9 [-5.1, -0.1] | -- | 14.8 (fired) | 83.5 | no saving | -- |
| reflection/trigger | `jev_reflect_trigger_s2lite_e2b-e2b` | E2b | Jev | reflection trigger tau=0.5 | replicate of jev_reflect_trigger_s2lite@E2c | 200 | 84.5 | +3.5 [-0.5, +7.5] | 9.8 | -0.2 [-1.0, 0.6] | -- | 14.0 (fired) | 97.3 | no saving | -- |
| reflection/trigger | `heuristic_reflect_trigger_s2lite` | E2c | heuristic | reflection trigger tau=0.5 | descriptive | 200 | 83.0 | +1.5 [-1.5, +4.5] | 9.7 | -0.3 [-0.7, 0.1] | -- | 0.2 (fired) | 97.3 | no saving | -- |
| reflection/trigger | `rate10_reflect_trigger_s2lite` | E2c | random, 10 % | reflection trigger tau=0.5 | descriptive | 200 | 87.0 | +5.5 [+1.5, +10.0] | 10.0 | -3.3 [-4.3, -2.3] | -- | 9.5 (fired) | 89.3 | no saving | -- |
| reflection/trigger | `rate30_reflect_trigger_s2lite` | E2c | random, 30 % | reflection trigger tau=0.5 | descriptive | 200 | 88.0 | +6.5 [+2.0, +11.5] | 10.6 | -9.7 [-11.6, -7.8] | -- | 29.3 (fired) | 70.1 | no saving | -- |
| reflection/trigger | `rate50_reflect_trigger_s2lite` | E2c | random, 50 % | reflection trigger tau=0.5 | descriptive | 200 | 90.0 | +8.5 [+4.0, +13.5] | 11.6 | -19.5 [-21.8, -17.0] | -- | 50.1 (fired) | 50.3 | no saving | -- |
| reflection/trigger | `ratem_reflect_trigger_s2lite` | E2c | random, matched rate | reflection trigger tau=0.5 | registered C5 (reference): not established | 500 | 85.4 | +4.2 [+1.0, +7.4] | 10.3 | -5.9 [-6.8, -5.0] | -- | 14.7 (fired) | 83.6 | no saving | -- |
| reflection/trigger | `s2self_reflect_trigger_s2lite` | E2c | the agent LLM (typed question) | reflection trigger tau=0.5 | descriptive | 200 | 84.0 | +2.5 [-0.5, +6.0] | 12.2 | -25.8 [-27.0, -24.6] | -- | 0.2 (fired) | 97.4 | no saving | -- |
| reflection/always | `reflect_always_s2lite` | E2b | no decision model | reflection always | descriptive | 200 | 89.5 | +8.5 [+4.0, +13.0] | 14.3 | -46.4 [-49.9, -42.7] | -- | -- (-) | -- | -- | -- |
| stop/guard | `jev_stop_guard_s2lite` | E2b | Jev | stop guard tau=0.5 | registered C6L: not established | 500 | 82.0 | +1.6 [-0.8, +4.0] | 13.0 | -32.1 [-39.4, -25.0] | -- | 72.6 (vetoed) | 75.4 | no saving | -- |
| stop/guard | `jev_stop_guard_s2lite-e2c` | E2c | Jev | stop guard tau=0.5 | replicate of jev_stop_guard_s2lite@E2b | 200 | 84.5 | +3.0 [+0.0, +6.5] | 12.0 | -23.8 [-34.6, -14.1] | -- | 65.3 (vetoed) | 78.3 | no saving | -- |
| stop/guard | `heuristic_stop_guard_s2lite` | E2c | heuristic | stop guard tau=0.5 | descriptive | 200 | 86.5 | +5.0 [+1.5, +8.5] | 9.7 | -0.1 [-0.5, 0.3] | -- | 0.0 (vetoed) | 86.5 | no saving | -- |
| stop/guard | `ratem_stop_guard_s2lite` | E2c | random, matched rate | stop guard tau=0.5 | descriptive | 200 | 83.0 | +1.5 [-2.0, +5.0] | 10.2 | -5.3 [-6.9, -4.0] | -- | 30.3 (vetoed) | 62.0 | no saving | -- |

## Routing (Flash-Lite → Flash) vs. Flash-Lite alone

| cell | row | run | model | setting | status | n | succ. % | Δ pt [95 % CI] | $/1k ep. | saving % [95 % CI] | cov. % | acted % | acc. % | verdict | fresh: n, Δ, saving, verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| routing/trigger | `jev_route_tau0.2` | E2b | Jev | routing trigger tau=0.2 | descriptive | 200 | 97.0 | +15.0 [+10.5, +20.0] | 29.1 | -198.0 [-212.6, -183.1] | -- | 59.6 (fired) | 99.0 | no saving; vs Flash alone Δ -3.0 pt, saving 28.4 % | -- |
| routing/trigger | `jev_route_tau0.2-e2c` | E2c | Jev | routing trigger tau=0.2 | replicate of jev_route_tau0.2@E2b | 200 | 96.0 | +14.5 [+10.0, +19.5] | 30.9 | -217.8 [-231.7, -204.2] | -- | 59.4 (fired) | 98.7 | no saving; vs Flash alone Δ -4.0 pt, saving 24.6 % | -- |
| routing/trigger | `jev_route_tau0.4` | E2b | Jev | routing trigger tau=0.4 | registered C4: not established | 500 | 81.4 | +0.6 [-1.6, +2.8] | 13.4 | -36.2 [-39.3, -33.3] | -- | 10.1 (fired) | 97.3 | no saving; vs Flash alone Δ -18.4 pt, saving 67.3 % | -- |
| routing/trigger | `jev_route_tau0.4-e2c` | E2c | Jev | routing trigger tau=0.4 | replicate of jev_route_tau0.4@E2b | 200 | 83.0 | +1.5 [-1.5, +4.5] | 12.9 | -33.3 [-37.4, -29.4] | -- | 9.7 (fired) | 97.5 | no saving; vs Flash alone Δ -17.0 pt, saving 68.4 % | -- |
| routing/trigger | `jev_route_tau0.5` | E2c | Jev | routing trigger tau=0.5 | descriptive | 200 | 84.5 | +3.0 [-0.5, +6.5] | 11.5 | -18.8 [-21.3, -16.3] | -- | 5.0 (fired) | 97.8 | no saving; vs Flash alone Δ -15.5 pt, saving 71.8 % | -- |
| routing/trigger | `jev_route_tau0.6` | E2b | Jev | routing trigger tau=0.6 | descriptive | 200 | 83.0 | +1.0 [-2.0, +4.0] | 10.1 | -3.3 [-3.9, -2.8] | -- | 0.0 (fired) | 97.7 | no saving; vs Flash alone Δ -17.0 pt, saving 75.2 % | -- |
| routing/trigger | `jev_route_tau0.6-e2c` | E2c | Jev | routing trigger tau=0.6 | replicate of jev_route_tau0.6@E2b | 200 | 82.0 | +0.5 [-2.5, +3.5] | 10.1 | -3.9 [-4.9, -3.1] | -- | 0.1 (fired) | 97.6 | no saving; vs Flash alone Δ -18.0 pt, saving 75.4 % | -- |
| routing/trigger | `jev_route_tau0.8` | E2b | Jev | routing trigger tau=0.8 | descriptive | 200 | 84.0 | +2.0 [-1.5, +5.5] | 10.1 | -3.4 [-4.0, -2.8] | -- | 0.0 (fired) | 97.9 | no saving; vs Flash alone Δ -16.0 pt, saving 75.2 % | -- |
| routing/trigger | `jev_route_tau0.8-e2c` | E2c | Jev | routing trigger tau=0.8 | replicate of jev_route_tau0.8@E2b | 200 | 84.0 | +2.5 [-1.0, +6.0] | 10.1 | -3.7 [-4.1, -3.2] | -- | 0.0 (fired) | 97.9 | no saving; vs Flash alone Δ -16.0 pt, saving 75.4 % | -- |
| routing/trigger | `lite_route` | E2b | Flash-Lite | routing trigger tau=0.5 | descriptive | 200 | 85.0 | +3.0 [+0.0, +6.0] | 14.4 | -47.8 [-51.9, -43.8] | -- | 6.4 (fired) | 97.9 | no saving; vs Flash alone Δ -15.0 pt, saving 64.5 % | -- |
| routing/trigger | `lite_route-e2c` | E2c | Flash-Lite | routing trigger tau=0.5 | replicate of lite_route@E2b | 200 | 84.0 | +2.5 [-1.0, +6.0] | 14.2 | -46.7 [-50.5, -43.0] | -- | 5.8 (fired) | 97.8 | no saving; vs Flash alone Δ -16.0 pt, saving 65.2 % | -- |
| routing/trigger | `heuristic_route` | E2c | heuristic | routing trigger tau=0.5 | descriptive | 200 | 83.5 | +2.0 [-1.0, +5.0] | 9.8 | -1.3 [-2.4, -0.3] | -- | 0.3 (fired) | 97.8 | no saving; vs Flash alone Δ -16.5 pt, saving 76.0 % | -- |
| routing/trigger | `rate10_route` | E2c | random, 10 % | routing trigger tau=0.5 | descriptive | 200 | 83.0 | +1.5 [-1.5, +4.5] | 13.2 | -35.5 [-40.8, -30.5] | -- | 11.3 (fired) | 97.5 | no saving; vs Flash alone Δ -17.0 pt, saving 67.9 % | -- |
| routing/trigger | `rate30_route` | E2c | random, 30 % | routing trigger tau=0.5 | descriptive | 200 | 86.5 | +5.0 [+0.5, +10.0] | 19.3 | -99.0 [-106.3, -91.7] | -- | 31.0 (fired) | 97.4 | no saving; vs Flash alone Δ -13.5 pt, saving 52.8 % | -- |
| routing/trigger | `rate50_route` | E2c | random, 50 % | routing trigger tau=0.5 | descriptive | 200 | 89.0 | +7.5 [+3.0, +12.5] | 25.5 | -163.1 [-171.6, -155.0] | -- | 51.3 (fired) | 97.0 | no saving; vs Flash alone Δ -11.0 pt, saving 37.6 % | -- |
| routing/trigger | `ratem_route` | E2c | random, matched rate | routing trigger tau=0.5 | descriptive | 200 | 83.5 | +2.0 [-1.0, +5.5] | 11.4 | -17.0 [-21.2, -13.0] | -- | 4.8 (fired) | 97.7 | no saving; vs Flash alone Δ -16.5 pt, saving 72.3 % | -- |
| routing/trigger | `randscore_route_tau0.2` | E2c | random score | routing trigger tau=0.2 | descriptive | 200 | 96.5 | +15.0 [+10.0, +20.0] | 33.8 | -247.6 [-255.8, -239.4] | -- | 78.5 (fired) | 97.8 | no saving; vs Flash alone Δ -3.5 pt, saving 17.5 % | -- |
| routing/trigger | `randscore_route_tau0.4` | E2c | random score | routing trigger tau=0.4 | descriptive | 200 | 94.0 | +12.5 [+8.0, +17.5] | 28.9 | -197.4 [-206.9, -187.7] | -- | 58.6 (fired) | 98.1 | no saving; vs Flash alone Δ -6.0 pt, saving 29.5 % | -- |
| routing/trigger | `randscore_route_tau0.6` | E2c | random score | routing trigger tau=0.6 | descriptive | 200 | 88.0 | +6.5 [+2.5, +10.5] | 21.6 | -122.2 [-130.6, -113.8] | -- | 37.5 (fired) | 97.5 | no saving; vs Flash alone Δ -12.0 pt, saving 47.3 % | -- |
| routing/trigger | `randscore_route_tau0.8` | E2c | random score | routing trigger tau=0.8 | descriptive | 200 | 87.0 | +5.5 [+2.0, +9.5] | 15.9 | -64.0 [-70.4, -57.6] | -- | 19.9 (fired) | 97.8 | no saving; vs Flash alone Δ -13.0 pt, saving 61.1 % | -- |
| routing/trigger | `s2self_route` | E2c | the agent LLM (typed question) | routing trigger tau=0.5 | descriptive | 200 | 82.5 | +1.0 [-1.5, +4.0] | 22.6 | -133.2 [-138.3, -127.8] | -- | 1.3 (fired) | 97.6 | no saving; vs Flash alone Δ -17.5 pt, saving 44.7 % | -- |

