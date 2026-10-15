# E7e: JevGym-Hard (12 customers), refitted Jev cascade, itemized final answer (exploratory (E7), descriptive)

- generated 2026-10-06T22:38:27+00:00 at git `4c7028846a99` by `scripts/explore_e7_hard.py`; run `runs/e7e_hard`; paired task bootstrap B = 10000, seed 0
- setting: families multi, 12 customers per task, rule `tier`, 60 near distractors, 20 noise lines, strict True, max steps 45; seeds 310500-310599; System-2 `gemini-3.5-flash` via gemini; 16 workers, task-major
- spend: evaluation + smoke $45.75 (System-2 $44.35, Jev $1.396); the pilot counts under E7; cap $60
- GPU tariff: no window record
- registration: `docs/PREREG_A7_STAGE2.md`, Deviations "E7e: itemized final answer". No test; outside every Holm table.
- **route:** Flash on the direct Gemini API in both arms, Jev on OpenRouter with the author's Jev-only key: dollars compare within this row only; thresholds from E7c's calibration arm; the task asks for one line per customer before the total in both arms
- `rl_refit`: not run (not run live: its calibration turns its tool choice off (E7c))

- tuning target: Flash alone 60-80 % on the evaluated setting; the development pilot gave 46.7 % (n = 60, e7_pilot_tier_k12, e7_pilot_tier_k12b); the evaluation gives 87.0 % (n = 100), outside the band
- `llm_only`: n = 100, success 87.0 %, $264.6 per 1k episodes, wall 63.5 s mean / 63.2 s median; failures 13 failed: wrong final total 13, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 13})

| arm | n | success % | Δ pt [95 % CI] | Newcombe | wins / losses | $/1k ep. (ref) | saving % [95 % CI] | tool coverage % | acc. % (accepted) | reflection fired % | wall mean / median s (ref) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `jev_refit` | 100 | 89.0 | +2.0 [-6.0, +10.0] | [-6.4, +10.4] | 9 / 7 | 183.8 (264.6) | 30.5 [29.8, 31.3] | 76.5 | 97.5 (99.2) | 4.9 | 104.1 / 103.6 (63.5 / 63.2) |

Cost: API as logged (System-2 via OpenRouter at its reported prices; Jev at its list price); the JevAgentRL arm adds the GPU tariff q per episode (its API-only saving and the sensitivity rows are below). Coverage: share of tool decisions (tool choice and arguments) executed from the decision model's answer. Acc.: the model's answers against the gold label on the run's own states (all of its tool answers; in brackets the accepted ones).

## Per arm

### `jev_refit`

- paired n = 100 (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 89.0 % vs 87.0 %
- failures (all complete episodes): 11 failed: wrong final total 11, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 11}); `llm_only`: 13 failed: wrong final total 13, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 13})
- API cost $183.8 per 1k episodes, saving 30.5 % [29.8, 31.3]
- tool_select: 3758 decisions, coverage 96.3 %, model acc. 98.8 % (n gold 3758), accepted 99.2 %, LLM on deferred 100.0 %
- tool_params: 5979 decisions, coverage 64.0 %, model acc. 96.7 % (n gold 5920), accepted 99.3 %, LLM on deferred -- %
- tool: 9737 decisions, coverage 76.5 %, model acc. 97.5 % (n gold 9678), accepted 99.2 %, LLM on deferred 100.0 %
- cancel-vs-ship decisions (tool choice where the gold next tool is a write; the tier rule): 1200, coverage 99.6 %, model acc. 100.0 %, accepted 100.0 % (0 accepted wrong), LLM on deferred 100.0 %; the model named the other write 0 times
- reflection trigger: 3658 questions, fired 4.9 %, acc. 99.7 % (gold positive 0.3 %)
- per episode: 37.6 steps, 25.3 LLM calls, 133.9 System-1 calls (35.7 s), System-2 $170.1 / System-1 $13.68 per 1k

## Final-answer form

Form of each complete episode's final answer (`answer_form`): the amount alone, prose with one or two amounts (e.g. only the total), or itemized (per-customer lines plus the total). Success is the arm's success among those answers.

| arm | itemized: n (share %), success % | sentence: n (share %), success % | number: n (share %), success % | none: n (share %), success % |
|---|---|---|---|---|
| `llm_only` | 100 (100.0), 87.0 | 0 (0.0), -- | 0 (0.0), -- | 0 (0.0), -- |
| `jev_refit` | 100 (100.0), 89.0 | 0 (0.0), -- | 0 (0.0), -- | 0 (0.0), -- |


## Latency breakdown (descriptive; changes no reported number)

Calls per arm grouped by component / decision point / served provider (System-2: OpenRouter's `meta.served_provider`; `1tool` = the forced single-tool argument call of a delegated tool); tail = calls over 5 s; shares are of the arm's summed episode wall time.

### `llm_only`: wall 63.5 s per episode; System-1 0.0 s, System-2 59.2 s, System-2 tail 16.5 s per episode

| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |
|---|---|---|---|---|---|---|---|
| s2/tool_select/gemini | 3700 | 1.07 | 2.03 | 1.60 | 175 | 93.2 | 26.0 |

### `jev_refit`: wall 104.1 s per episode; System-1 35.7 s, System-2 56.5 s, System-2 tail 27.9 s per episode

| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |
|---|---|---|---|---|---|---|---|
| s2/tool_params/1tool/gemini | 2217 | 0.97 | 6.18 | 2.16 | 289 | 46.1 | 22.7 |
| s1/tool_params/typesafe/jev-1.13-20260917 | 5979 | 0.24 | 0.35 | 0.26 | 0 | 15.1 | 0.0 |
| s1/tool_select/typesafe/jev-1.13-20260917 | 3758 | 0.25 | 0.35 | 0.27 | 0 | 9.7 | 0.0 |
| s1/reflection/typesafe/jev-1.13-20260917 | 3658 | 0.24 | 0.37 | 0.27 | 0 | 9.5 | 0.0 |
| s2/reflection/gemini | 178 | 1.31 | 5.95 | 2.62 | 21 | 4.5 | 2.1 |
| s2/tool_select/gemini | 138 | 1.31 | 5.71 | 2.77 | 17 | 3.7 | 2.0 |

- `jev_refit` vs `llm_only`: +40.6 s per episode (System-1 +35.7 s, System-2 -2.7 s); the System-2 tail over 5 s adds +11.4 s, 28.0 % of the extra time

