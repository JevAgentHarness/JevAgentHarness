# E7c: JevGym-Hard (12 customers) with refitted thresholds, Jev row (local) (exploratory (E7), descriptive)

- generated 2026-10-06T22:38:24+00:00 at git `4c7028846a99` by `scripts/explore_e7_hard.py`; run `runs/e7cj_hard`; paired task bootstrap B = 10000, seed 0
- setting: families multi, 12 customers per task, rule `tier`, 60 near distractors, 20 noise lines, strict True, max steps 45; seeds 310400-310499; System-2 `gemini-3.5-flash` via gemini; 16 workers, task-major
- spend: evaluation + smoke $0.00 (System-2 $0.00, Jev $0.000); the pilot counts under E7; cap $90
- GPU tariff: no window record
- registration: `docs/PREREG_A7_STAGE2.md`, Deviations "E7c Jev row, local". No test; outside every Holm table.
- **route:** Flash on the direct Gemini API in both arms, Jev on OpenRouter with the author's Jev-only key (the org's OpenRouter budget is exhausted): dollars compare within this row only; the thresholds were fitted on E7c's calibration arm (OpenRouter route)
- `rl_refit`: not run (not run live: its calibration turns its tool choice off (the reported result))

- tuning target: Flash alone 60-80 % on the evaluated setting; the development pilot gave 46.7 % (n = 60, e7_pilot_tier_k12, e7_pilot_tier_k12b); the evaluation gives 42.0 % (n = 100), outside the band
- `llm_only`: n = 100, success 42.0 %, $262.9 per 1k episodes, wall 66.6 s mean / 64.9 s median; failures 58 failed: wrong final total 58, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 58})

| arm | n | success % | Δ pt [95 % CI] | Newcombe | wins / losses | $/1k ep. (ref) | saving % [95 % CI] | tool coverage % | acc. % (accepted) | reflection fired % | wall mean / median s (ref) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `jev_refit` | 100 | 19.0 | -23.0 [-35.0, -11.0] | [-34.7, -10.3] | 11 / 34 | 181.4 (262.9) | 31.0 [30.3, 31.7] | 76.6 | 97.8 (99.5) | 5.3 | 118.3 / 114.4 (66.6 / 64.9) |

Cost: API as logged (System-2 via OpenRouter at its reported prices; Jev at its list price); the JevAgentRL arm adds the GPU tariff q per episode (its API-only saving and the sensitivity rows are below). Coverage: share of tool decisions (tool choice and arguments) executed from the decision model's answer. Acc.: the model's answers against the gold label on the run's own states (all of its tool answers; in brackets the accepted ones).

## Per arm

### `jev_refit`

- paired n = 100 (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 19.0 % vs 42.0 %
- failures (all complete episodes): 81 failed: wrong final total 81, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 81}); `llm_only`: 58 failed: wrong final total 58, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 58})
- API cost $181.4 per 1k episodes, saving 31.0 % [30.3, 31.7]
- tool_select: 3740 decisions, coverage 96.7 %, model acc. 99.1 % (n gold 3740), accepted 99.4 %, LLM on deferred 100.0 %
- tool_params: 5966 decisions, coverage 64.1 %, model acc. 97.0 % (n gold 5925), accepted 99.5 %, LLM on deferred -- %
- tool: 9706 decisions, coverage 76.6 %, model acc. 97.8 % (n gold 9665), accepted 99.5 %, LLM on deferred 100.0 %
- cancel-vs-ship decisions (tool choice where the gold next tool is a write; the tier rule): 1200, coverage 99.3 %, model acc. 99.9 %, accepted 100.0 % (0 accepted wrong), LLM on deferred 100.0 %; the model named the other write 1 times
- reflection trigger: 3640 questions, fired 5.3 %, acc. 98.3 % (gold positive 1.7 %)
- per episode: 37.4 steps, 25.2 LLM calls, 133.5 System-1 calls (37.9 s), System-2 $167.9 / System-1 $13.53 per 1k

## Final-answer form

Form of each complete episode's final answer (`answer_form`): the amount alone, prose with one or two amounts (e.g. only the total), or itemized (per-customer lines plus the total). Success is the arm's success among those answers.

| arm | itemized: n (share %), success % | sentence: n (share %), success % | number: n (share %), success % | none: n (share %), success % |
|---|---|---|---|---|
| `llm_only` | 51 (51.0), 82.4 | 18 (18.0), 0.0 | 31 (31.0), 0.0 | 0 (0.0), -- |
| `jev_refit` | 26 (26.0), 61.5 | 0 (0.0), -- | 74 (74.0), 4.1 | 0 (0.0), -- |


## Calibration fit (E7c) (JevGym-Hard, 12 customers; calibration seeds 5300-5399)

Generated 2026-10-06T19:47:44+00:00; run `runs/e7c_cal`; estimator: threshold_theory --calibration-arm --task-value 1.0 --lam-per-hour 0 --v-fallback W (A6.5 / A6.11).

| calibration arm | decision point | tau-hat | rule | paired | V | dC | coverage at tau % | model acc. taken % | LLM acc. taken % | model acc. all % | LLM acc. all % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `jev_shadow_cal` | tool_select | 0.91 | projection | 3699 | 1.0 | 0.001991415486486487 | 90.1 | 100.0 | 100.0 | 99.8 | 100.0 |
| `jev_shadow_cal` | tool_params | 0.87 | projection | 5998 | 1.0 | 0.001136279225 | 80.2 | 100.0 | 100.0 | 97.4 | 100.0 |
| `rl_shadow_cal` | tool_select | always defer | projection | 3561 | 1.0 | 0.001839197851727043 | 0.0 | -- | -- | 86.5 | 100.0 |
| `rl_shadow_cal` | tool_params | 1 | projection | 5781 | 1.0 | 0.001052616554229372 | 62.9 | 100.0 | 100.0 | 94.8 | 100.0 |

**Under the registered estimator, the procedure turns the frozen JevAgentRL-4B's tool choice off on this setting**: its tool_select fit accepts no confidence level (always defer), so the refitted 4B profile leaves tool choice to the LLM (its tool_params threshold is reached only inside System-1-chosen calls).

Cancel-vs-ship decisions (tool choice where the gold next tool is a write), at the fitted tau:

- `jev_shadow_cal`: 1200 decisions, coverage 99.5 %, model acc. taken 100.0 %, model acc. all 100.0 %, LLM acc. all 100.0 %
- `rl_shadow_cal`: 1153 decisions, coverage 0.0 %, model acc. taken -- %, model acc. all 81.1 %, LLM acc. all 100.0 %


## Latency breakdown (descriptive; changes no reported number)

Calls per arm grouped by component / decision point / served provider (System-2: OpenRouter's `meta.served_provider`; `1tool` = the forced single-tool argument call of a delegated tool); tail = calls over 5 s; shares are of the arm's summed episode wall time.

### `llm_only`: wall 66.6 s per episode; System-1 0.0 s, System-2 62.0 s, System-2 tail 14.2 s per episode

| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |
|---|---|---|---|---|---|---|---|
| s2/tool_select/gemini | 3700 | 1.20 | 2.34 | 1.68 | 147 | 93.2 | 21.3 |

### `jev_refit`: wall 118.3 s per episode; System-1 37.9 s, System-2 64.1 s, System-2 tail 35.9 s per episode

| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |
|---|---|---|---|---|---|---|---|
| s2/tool_params/1tool/gemini | 2205 | 1.05 | 6.97 | 2.46 | 345 | 45.8 | 25.5 |
| s1/tool_params/typesafe/jev-1.13-20260917 | 5966 | 0.24 | 0.41 | 0.28 | 0 | 14.1 | 0.0 |
| s1/tool_select/typesafe/jev-1.13-20260917 | 3740 | 0.25 | 0.41 | 0.29 | 0 | 9.1 | 0.0 |
| s1/reflection/typesafe/jev-1.13-20260917 | 3640 | 0.25 | 0.44 | 0.29 | 0 | 8.9 | 0.0 |
| s2/reflection/gemini | 193 | 1.40 | 8.19 | 3.00 | 33 | 4.9 | 2.8 |
| s2/tool_select/gemini | 125 | 1.45 | 8.72 | 3.27 | 21 | 3.5 | 2.0 |

- `jev_refit` vs `llm_only`: +51.7 s per episode (System-1 +37.9 s, System-2 +2.0 s); the System-2 tail over 5 s adds +21.7 s, 42.0 % of the extra time

