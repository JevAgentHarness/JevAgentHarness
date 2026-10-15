# E7: JevGym-Hard (exploratory (E7), descriptive)

- generated 2026-10-06T16:36:09+00:00 at git `f88b3474bd5d` by `scripts/explore_e7_hard.py`; run `runs/e7_hard`; paired task bootstrap B = 10000, seed 0
- setting: families multi, 10 customers per task, rule `tier`, 60 near distractors, 20 noise lines, strict True, max steps 45; seeds 310000-310199; System-2 `google/gemini-3.5-flash` via openrouter; 12 workers, task-major
- spend: evaluation + smoke $25.26 (System-2 $23.29, Jev $1.973); pilot $17.75; cap $150
- GPU tariff q = $0.00821 per JevAgentRL episode (g5.8xlarge, $2.448/h × 0.671 h / 200 episodes; sensitivity: S2-4's q $0.00196, q = 0)
- registration: `docs/PREREG_A7_STAGE2.md`, Deviations "Exploratory row E7 (JevGym-Hard)". No test; outside every Holm table.

- tuning target: Flash alone 60-80 % on the evaluated setting; the development pilot gave 76.7 % (n = 30, e7_pilot_tier_k10); the evaluation gives 88.0 % (n = 200), outside the band
- `llm_only`: n = 200, success 88.0 %, $59.4 per 1k episodes, wall 33.1 s mean / 32.5 s median; failures 24 failed: wrong final total 24, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 24})

| arm | n | success % | Δ pt [95 % CI] | Newcombe | wins / losses | $/1k ep. (ref) | saving % [95 % CI] | tool coverage % | acc. % (accepted) | reflection fired % | wall mean / median s (ref) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `jevagent_best` | 200 | 86.5 | -1.5 [-7.5, +4.5] | [-7.5, +4.5] | 16 / 19 | 55.7 (59.4) | 6.2 [4.4, 8.0] | 66.2 | 99.7 (100.0) | 6.5 | 47.6 / 46.9 (33.1 / 32.5) |
| `jevagent_best_rl` | 200 | 3.0 | -85.0 [-90.0, -79.5] | [-89.2, -78.7] | 1 / 171 | 18.2 (59.4) | 69.4 [67.0, 71.7] | 77.6 | 95.5 (96.2) | 3.6 | 60.5 / 48.0 (33.1 / 32.5) |

Cost: API as logged (System-2 via OpenRouter at its reported prices; Jev at its list price); the JevAgentRL arm adds the GPU tariff q per episode (its API-only saving and the sensitivity rows are below). Coverage: share of tool decisions (tool choice and arguments) executed from the decision model's answer. Acc.: the model's answers against the gold label on the run's own states (all of its tool answers; in brackets the accepted ones).

## Per arm

### `jevagent_best`

- paired n = 200 (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 86.5 % vs 88.0 %
- failures (all complete episodes): 27 failed: wrong final total 27, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 27}); `llm_only`: 24 failed: wrong final total 24, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 24})
- API cost $55.7 per 1k episodes, saving 6.2 % [4.4, 8.0]
- tool_select: 6200 decisions, coverage 62.5 %, model acc. 99.8 % (n gold 6200), accepted 100.0 %, LLM on deferred 99.2 %
- tool_params: 6842 decisions, coverage 69.7 %, model acc. 99.7 % (n gold 6842), accepted 100.0 %, LLM on deferred -- %
- tool: 13042 decisions, coverage 66.2 %, model acc. 99.7 % (n gold 13042), accepted 100.0 %, LLM on deferred 99.2 %
- cancel-vs-ship decisions (tool choice where the gold next tool is a write; the tier rule): 2009, coverage 56.3 %, model acc. 99.7 %, accepted 100.0 % (0 accepted wrong), LLM on deferred 99.0 %; the model named the other write 2 times
- reflection trigger: 6000 questions, fired 6.5 %, acc. 99.6 % (gold positive 0.4 %)
- per episode: 31.0 steps, 23.3 LLM calls, 95.2 System-1 calls (18.5 s), System-2 $45.9 / System-1 $9.76 per 1k

### `jevagent_best_rl`

- paired n = 200 (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 3.0 % vs 88.0 %
- failures (all complete episodes): 194 failed: wrong final total 5, wrong state-changing call 152, ask / finish error 37, other 0 (types {'strict_stop': 152, 'premature_finish': 37, 'wrong_total': 5}); `llm_only`: 24 failed: wrong final total 24, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 24})
- API cost $10.0 per 1k episodes, saving 83.2 % [80.8, 85.5]
- sensitivity q = s2_4: $11.9 per 1k, saving 79.9 % [77.5, 82.2]
- sensitivity q = zero: $10.0 per 1k, saving 83.2 % [80.8, 85.5]
- tool_select: 2670 decisions, coverage 99.9 %, model acc. 92.6 % (n gold 2670), accepted 92.6 %, LLM on deferred 100.0 %
- tool_params: 4370 decisions, coverage 64.0 %, model acc. 97.4 % (n gold 4038), accepted 99.8 %, LLM on deferred -- %
- tool: 7040 decisions, coverage 77.6 %, model acc. 95.5 % (n gold 6708), accepted 96.2 %, LLM on deferred 100.0 %
- cancel-vs-ship decisions (tool choice where the gold next tool is a write; the tier rule): 877, coverage 99.9 %, model acc. 81.6 %, accepted 81.7 % (160 accepted wrong), LLM on deferred 100.0 %; the model named the other write 151 times
- reflection trigger: 2470 questions, fired 3.6 %, acc. 91.8 % (gold positive 8.2 %)
- per episode: 13.3 steps, 6.4 LLM calls, 47.5 System-1 calls (51.6 s), System-2 $10.0 / System-1 $0.00 per 1k

