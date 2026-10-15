# E7b: JevGym-Hard with 12 customers (exploratory (E7), descriptive)

- generated 2026-10-06T19:47:14+00:00 at git `dfd239c2a559` by `scripts/explore_e7_hard.py`; run `runs/e7b_hard`; paired task bootstrap B = 10000, seed 0
- setting: families multi, 12 customers per task, rule `tier`, 60 near distractors, 20 noise lines, strict True, max steps 45; seeds 310200-310399; System-2 `google/gemini-3.5-flash` via openrouter; 12 workers, task-major
- spend: evaluation + smoke $31.47 (System-2 $29.08, Jev $2.387); the pilot counts under E7; cap $60
- GPU tariff q = $0.00992 per JevAgentRL episode (g5.8xlarge, $2.448/h × 0.810 h / 200 episodes; sensitivity: S2-4's q $0.00196, q = 0)
- registration: `docs/PREREG_A7_STAGE2.md`, Deviations "E7b (JevGym-Hard, 12 customers)". No test; outside every Holm table.

- tuning target: Flash alone 60-80 % on the evaluated setting; the development pilot gave 46.7 % (n = 60, e7_pilot_tier_k12, e7_pilot_tier_k12b); the evaluation gives 66.0 % (n = 200), inside the band
- `llm_only`: n = 200, success 66.0 %, $73.3 per 1k episodes, wall 41.6 s mean / 40.6 s median; failures 68 failed: wrong final total 68, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 68})

| arm | n | success % | Δ pt [95 % CI] | Newcombe | wins / losses | $/1k ep. (ref) | saving % [95 % CI] | tool coverage % | acc. % (accepted) | reflection fired % | wall mean / median s (ref) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `jevagent_best` | 200 | 69.5 | +3.5 [-4.5, +11.5] | [-4.7, +11.6] | 38 / 31 | 69.9 (73.3) | 4.6 [3.1, 6.1] | 65.7 | 99.6 (100.0) | 5.6 | 71.5 / 70.3 (41.6 / 40.6) |
| `jevagent_best_rl` | 200 | 0.5 | -65.5 [-72.0, -59.0] | [-71.7, -58.3] | 0 / 131 | 22.3 (73.3) | 69.5 [66.8, 72.2] | 75.7 | 95.2 (96.2) | 3.5 | 59.3 / 38.7 (41.6 / 40.6) |

Cost: API as logged (System-2 via OpenRouter at its reported prices; Jev at its list price); the JevAgentRL arm adds the GPU tariff q per episode (its API-only saving and the sensitivity rows are below). Coverage: share of tool decisions (tool choice and arguments) executed from the decision model's answer. Acc.: the model's answers against the gold label on the run's own states (all of its tool answers; in brackets the accepted ones).

## Per arm

### `jevagent_best`

- paired n = 200 (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 69.5 % vs 66.0 %
- failures (all complete episodes): 61 failed: wrong final total 61, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 61}); `llm_only`: 68 failed: wrong final total 68, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 68})
- API cost $69.9 per 1k episodes, saving 4.6 % [3.1, 6.1]
- tool_select: 7400 decisions, coverage 60.4 %, model acc. 99.6 % (n gold 7400), accepted 100.0 %, LLM on deferred 99.2 %
- tool_params: 8088 decisions, coverage 70.5 %, model acc. 99.6 % (n gold 8088), accepted 100.0 %, LLM on deferred -- %
- tool: 15488 decisions, coverage 65.7 %, model acc. 99.6 % (n gold 15488), accepted 100.0 %, LLM on deferred 99.2 %
- cancel-vs-ship decisions (tool choice where the gold next tool is a write; the tier rule): 2411, coverage 58.1 %, model acc. 99.8 %, accepted 100.0 % (0 accepted wrong), LLM on deferred 98.9 %; the model named the other write 4 times
- reflection trigger: 7200 questions, fired 5.6 %, acc. 99.3 % (gold positive 0.7 %)
- per episode: 37.0 steps, 27.8 LLM calls, 113.4 System-1 calls (22.5 s), System-2 $58.1 / System-1 $11.82 per 1k

### `jevagent_best_rl`

- paired n = 200 (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 0.5 % vs 66.0 %
- failures (all complete episodes): 199 failed: wrong final total 5, wrong state-changing call 169, ask / finish error 25, other 0 (types {'strict_stop': 169, 'premature_finish': 24, 'no_answer': 1, 'wrong_total': 5}); `llm_only`: 68 failed: wrong final total 68, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 68})
- API cost $12.4 per 1k episodes, saving 83.1 % [80.3, 85.7]
- sensitivity q = s2_4: $14.4 per 1k, saving 80.4 % [77.7, 83.0]
- sensitivity q = zero: $12.4 per 1k, saving 83.1 % [80.3, 85.7]
- tool_select: 2881 decisions, coverage 100.0 %, model acc. 92.6 % (n gold 2881), accepted 92.6 %, LLM on deferred -- %
- tool_params: 4751 decisions, coverage 61.0 %, model acc. 96.9 % (n gold 4370), accepted 100.0 %, LLM on deferred -- %
- tool: 7632 decisions, coverage 75.7 %, model acc. 95.2 % (n gold 7251), accepted 96.2 %, LLM on deferred -- %
- cancel-vs-ship decisions (tool choice where the gold next tool is a write; the tier rule): 957, coverage 100.0 %, model acc. 81.5 %, accepted 81.5 % (177 accepted wrong), LLM on deferred -- %; the model named the other write 170 times
- reflection trigger: 2681 questions, fired 3.5 %, acc. 92.3 % (gold positive 7.7 %)
- per episode: 14.4 steps, 7.3 LLM calls, 51.6 System-1 calls (39.8 s), System-2 $12.4 / System-1 $0.00 per 1k


## Latency breakdown (descriptive; changes no reported number)

Calls per arm grouped by component / decision point / served provider (System-2: OpenRouter's `meta.served_provider`; `1tool` = the forced single-tool argument call of a delegated tool); tail = calls over 5 s; shares are of the arm's summed episode wall time.

### `llm_only`: wall 41.6 s per episode; System-1 0.0 s, System-2 38.9 s, System-2 tail 0.8 s per episode

| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |
|---|---|---|---|---|---|---|---|
| s2/tool_select/Google | 7397 | 0.97 | 1.18 | 1.04 | 17 | 92.8 | 1.2 |
| s2/tool_select/Google AI Studio | 3 | 17.77 | 18.50 | 17.73 | 3 | 0.6 | 0.6 |

### `jevagent_best`: wall 71.5 s per episode; System-1 22.5 s, System-2 45.6 s, System-2 tail 9.0 s per episode

| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |
|---|---|---|---|---|---|---|---|
| s2/tool_params/1tool/Google | 2216 | 1.31 | 5.10 | 2.30 | 230 | 35.7 | 10.9 |
| s2/tool_select/Google | 2928 | 0.98 | 1.30 | 1.12 | 6 | 23.0 | 0.4 |
| s1/tool_params/typesafe/jev-1.13-20260917 | 8088 | 0.18 | 0.26 | 0.20 | 0 | 11.1 | 0.0 |
| s1/tool_select/typesafe/jev-1.13-20260917 | 7400 | 0.19 | 0.26 | 0.20 | 0 | 10.5 | 0.0 |
| s1/reflection/typesafe/jev-1.13-20260917 | 7200 | 0.18 | 0.26 | 0.20 | 0 | 9.9 | 0.0 |
| s2/reflection/Google | 406 | 1.36 | 1.59 | 1.53 | 4 | 4.3 | 0.4 |
| s2/tool_params/1tool/Google AI Studio | 9 | 11.98 | 12.53 | 12.13 | 9 | 0.8 | 0.8 |

### `jevagent_best_rl`: wall 59.3 s per episode; System-1 39.8 s, System-2 17.9 s, System-2 tail 6.8 s per episode

| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |
|---|---|---|---|---|---|---|---|
| s2/tool_params/1tool/Google | 1364 | 1.32 | 5.54 | 2.44 | 181 | 28.1 | 10.7 |
| s1/tool_params/jevagentrl-4b | 4751 | 0.60 | 1.26 | 0.64 | 0 | 25.6 | 0.0 |
| s1/tool_select/jevagentrl-4b | 2881 | 1.00 | 1.65 | 1.04 | 0 | 25.2 | 0.0 |
| s1/reflection/jevagentrl-4b | 2681 | 0.65 | 1.38 | 0.72 | 0 | 16.3 | 0.0 |
| s2/reflection/Google | 95 | 1.56 | 1.77 | 1.59 | 0 | 1.3 | 0.0 |
| s2/tool_params/1tool/Google AI Studio | 7 | 11.84 | 15.07 | 12.73 | 7 | 0.8 | 0.8 |

- `jevagent_best` vs `llm_only`: +29.9 s per episode (System-1 +22.5 s, System-2 +6.7 s); the System-2 tail over 5 s adds +8.2 s, 27.4 % of the extra time
- `jevagent_best_rl` vs `llm_only`: +17.7 s per episode (System-1 +39.8 s, System-2 -21.0 s); the System-2 tail over 5 s adds +6.0 s, 33.8 % of the extra time

