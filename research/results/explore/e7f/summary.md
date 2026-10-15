# E7f: JevGym-Hard (12 customers, itemized answer), AgentDecision-4B (tier) live (exploratory (E7), descriptive)

- generated 2026-10-07T07:17:46+00:00 at git `514bc3b434a5` by `scripts/explore_e7_hard.py`; run `runs/e7f_hard; runs/e7fx_hard`; paired task bootstrap B = 10000, seed 0
- setting: families multi, 12 customers per task, rule `tier`, 60 near distractors, 20 noise lines, strict True, max steps 45, answer style `itemized`; seeds 310600-310799 (pooled parts: first 80 tasks, ext 120 tasks); System-2 `gemini-3.5-flash` via gemini; 12 workers, task-major
- spend: evaluation + smoke $99.67 (System-2 $99.67, Jev $0.000); the pilot counts under E7; cap $115
- GPU tariff q = $0.02340 per JevAgentRL episode (g5.8xlarge, $2.448/h × 1.911 h / 200 episodes; sensitivity: S2-4's q $0.00196, q = 0)
- registration: `docs/PREREG_A7_STAGE2.md`, Deviations "E7f: AgentDecision-4B (tier) live". No test; outside every Holm table.
- **route:** Flash on the direct Gemini API in both arms: dollars compare within this row only; rl_tier = AgentDecision-4B (tier) (E7d's continued checkpoint) served on the host at E7d's fitted thresholds (tool choice 0.99993, arguments 0.99291); the task asks for one line per customer before the total
- `rl_frozen_refit`: not run (the frozen 4B at its registered refit (E7c) always defers at tool choice, so the arm would cost about as much as llm_only and show nothing E7c's fit has not shown)

- tuning target: Flash alone 60-80 % on the evaluated setting; the development pilot gave 46.7 % (n = 60, e7_pilot_tier_k12, e7_pilot_tier_k12b); the evaluation gives 83.5 % (n = 200), outside the band
- `llm_only`: n = 200, success 83.5 %, $264.3 per 1k episodes, wall 74.2 s mean / 63.2 s median; failures 33 failed: wrong final total 33, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 33})

| arm | n | success % | Δ pt [95 % CI] | Newcombe | wins / losses | $/1k ep. (ref) | saving % [95 % CI] | tool coverage % | acc. % (accepted) | reflection fired % | wall mean / median s (ref) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `rl_tier` | 200 | 81.5 | -2.0 [-8.5, +4.5] | [-8.8, +4.8] | 21 / 25 | 253.0 (264.3) | 4.3 [2.1, 6.6] | 58.7 | 98.9 (100.0) | 9.3 | 327.2 / 325.8 (74.2 / 63.2) |

Cost: API as logged (System-2 via gemini at the list prices of configs/prices/models.yaml; Jev at its list price); the JevAgentRL arm adds the GPU tariff q per episode (its API-only saving and the sensitivity rows are below). Coverage: share of tool decisions (tool choice and arguments) executed from the decision model's answer. Acc.: the model's answers against the gold label on the run's own states (all of its tool answers; in brackets the accepted ones).

## Per arm

### `rl_tier`

- paired n = 200 (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 81.5 % vs 83.5 %
- failures (all complete episodes): 37 failed: wrong final total 35, wrong state-changing call 0, ask / finish error 2, other 0 (types {'wrong_total': 35, 'premature_finish': 2}); `llm_only`: 33 failed: wrong final total 33, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 33})
- API cost $229.6 per 1k episodes, saving 13.1 % [10.9, 15.4]
- sensitivity q = s2_4: $231.6 per 1k, saving 12.4 % [10.2, 14.7]
- sensitivity q = zero: $229.6 per 1k, saving 13.1 % [10.9, 15.4]
- tool_select: 7394 decisions, coverage 52.7 %, model acc. 98.5 % (n gold 7394), accepted 99.9 %, LLM on deferred 100.0 %
- tool_params: 6092 decisions, coverage 65.9 %, model acc. 99.3 % (n gold 6092), accepted 100.0 %, LLM on deferred -- %
- tool: 13486 decisions, coverage 58.7 %, model acc. 98.9 % (n gold 13486), accepted 100.0 %, LLM on deferred 100.0 %
- cancel-vs-ship decisions (tool choice where the gold next tool is a write; the tier rule): 2398, coverage 51.6 %, model acc. 96.3 %, accepted 100.0 % (0 accepted wrong), LLM on deferred 100.0 %; the model named the other write 88 times
- reflection trigger: 7194 questions, fired 9.3 %, acc. 99.4 % (gold positive 0.6 %)
- per episode: 37.0 steps, 31.6 LLM calls, 103.4 System-1 calls (232.3 s), System-2 $229.6 / System-1 $0.00 per 1k

## Final-answer form

Form of each complete episode's final answer (`answer_form`): the amount alone, prose with one or two amounts (e.g. only the total), or itemized (per-customer lines plus the total). Success is the arm's success among those answers.

| arm | itemized: n (share %), success % | sentence: n (share %), success % | number: n (share %), success % | none: n (share %), success % |
|---|---|---|---|---|
| `llm_only` | 200 (100.0), 83.5 | 0 (0.0), -- | 0 (0.0), -- | 0 (0.0), -- |
| `rl_tier` | 200 (100.0), 81.5 | 0 (0.0), -- | 0 (0.0), -- | 0 (0.0), -- |


## Latency breakdown (descriptive; changes no reported number)

Calls per arm grouped by component / decision point / served provider (System-2: OpenRouter's `meta.served_provider`; `1tool` = the forced single-tool argument call of a delegated tool); tail = calls over 5 s; shares are of the arm's summed episode wall time.

### `llm_only`: wall 74.2 s per episode; System-1 0.0 s, System-2 70.7 s, System-2 tail 22.9 s per episode

| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |
|---|---|---|---|---|---|---|---|
| s2/tool_select/gemini | 7401 | 1.15 | 4.02 | 1.91 | 565 | 95.3 | 30.9 |

### `rl_tier`: wall 327.2 s per episode; System-1 232.3 s, System-2 86.0 s, System-2 tail 45.8 s per episode

| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |
|---|---|---|---|---|---|---|---|
| s1/tool_select/agentdecision-4b-tier | 7394 | 2.50 | 3.68 | 2.48 | 25 | 28.1 | 0.2 |
| s1/reflection/agentdecision-4b-tier | 7194 | 2.17 | 3.34 | 2.15 | 4 | 23.7 | 0.0 |
| s1/tool_params/agentdecision-4b-tier | 6092 | 2.07 | 3.27 | 2.07 | 1 | 19.2 | 0.0 |
| s2/tool_select/gemini | 3494 | 1.21 | 5.81 | 2.38 | 441 | 12.7 | 5.8 |
| s2/tool_params/1tool/gemini | 2159 | 1.46 | 7.47 | 3.03 | 506 | 10.0 | 6.0 |
| s2/reflection/gemini | 669 | 1.65 | 8.25 | 3.50 | 170 | 3.6 | 2.2 |

- `rl_tier` vs `llm_only`: +253.0 s per episode (System-1 +232.3 s, System-2 +15.2 s); the System-2 tail over 5 s adds +22.9 s, 9.1 % of the extra time


## Diagnosis view (as results/explore/e7f/DIAGNOSIS.md; descriptive): `rl_tier` in `runs/e7f_hard; runs/e7fx_hard`

- episodes: 200 complete of 200 tasks
- thresholds ran with: tool_select cascade tau 0.9999322567940803; tool_params cascade tau 0.9929112792015076; reflection trigger tau 0.5
- tool_select: 7394 cascade decisions; accepted / deferred as the threshold says: 7394 (of which 0 deferred because the answer was OTHER); violations 0
- tool_params: 6092 cascade decisions; accepted / deferred as the threshold says: 6092 (of which 1486 deferred because the answer was OTHER); violations 0
- System-1 endpoint per episode: {"backend": "agentdecision-4b-tier", "base_url": "http://127.0.0.1:8081", "extra_body": {}, "model": "agentdecision-4b-tier", "provider": "typesafe"} x 200
- System-1 calls by backend / model: agentdecision-4b-tier / agentdecision-4b-tier x 20680
- write states (gold next tool cancel_order / ship_order): 2398; both writes offered 2398; the policy sentence in the task 2398; the order identified from the current observation 2398, and of those the customer's tier and the amount both in the state 2398; longest state 5249 characters (harness cap 12000; states at the cap 0)

| tier \| amount vs. the tier's limit \| gold | n | model wrong | accepted wrong |
|---|---|---|---|
| basic | above limit | gold cancel_order | 544 | 18 (3.3 %) | 0 |
| basic | at or below limit | gold ship_order | 248 | 2 (0.8 %) | 0 |
| gold | gold tier | gold ship_order | 777 | 0 (0.0 %) | 0 |
| silver | above limit | gold cancel_order | 273 | 25 (9.2 %) | 0 |
| silver | at or below limit | gold ship_order | 556 | 43 (7.7 %) | 0 |

Cancel rate by tier and amount band ($150 bands; the rule cancels basic above $300, silver above $600, never gold):

| tier \| amount band ($) | n | model says cancel % | rule cancels % |
|---|---|---|---|
| basic | 0-150 | 109 | 0.0 | 0.0 |
| basic | 150-300 | 139 | 1.4 | 0.0 |
| basic | 300-450 | 147 | 91.8 | 100.0 |
| basic | 450-600 | 134 | 95.5 | 100.0 |
| basic | 600-750 | 135 | 100.0 | 100.0 |
| basic | 750-900 | 128 | 100.0 | 100.0 |
| gold | 0-150 | 97 | 0.0 | 0.0 |
| gold | 150-300 | 147 | 0.0 | 0.0 |
| gold | 300-450 | 136 | 0.0 | 0.0 |
| gold | 450-600 | 138 | 0.0 | 0.0 |
| gold | 600-750 | 137 | 0.0 | 0.0 |
| gold | 750-900 | 122 | 0.0 | 0.0 |
| silver | 0-150 | 131 | 0.0 | 0.0 |
| silver | 150-300 | 141 | 0.0 | 0.0 |
| silver | 300-450 | 140 | 12.9 | 0.0 |
| silver | 450-600 | 144 | 17.4 | 0.0 |
| silver | 600-750 | 132 | 87.9 | 100.0 |
| silver | 750-900 | 141 | 93.6 | 100.0 |

Failed episodes by type, with the decision that made the last step:

- 22 x wrong_total: last step finish chosen by the LLM
- 13 x wrong_total: last step finish chosen by the model
- 2 x premature_finish: finish chosen by the model
- early finishes, by the number of writes done before the finish: 11 writes: 2


## By part (each part's own paired comparison; the table above pools them)

| part | run | seeds | arm | n | LLM-only success % | arm success % | Δ pt [95 % CI] | wins / losses | saving % [95 % CI] | wall mean s (ref) | spend $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| first | `runs/e7f_hard` | 310600-310679 | `rl_tier` | 80 | 77.5 | 76.2 | -1.2 [-12.5, +8.8] | 9 / 10 | 3.4 [0.0, 6.9] | 336.0 (58.4) | 39.64 |
| ext | `runs/e7fx_hard` | 310680-310799 | `rl_tier` | 120 | 87.5 | 85.0 | -2.5 [-10.8, +5.8] | 12 / 15 | 4.9 [1.9, 7.9] | 321.3 (84.7) | 59.14 |
