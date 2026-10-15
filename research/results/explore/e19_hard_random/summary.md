# E19: JevGym-Hard (12 customers, itemized answer), AgentDecision-4B (random rules) live (exploratory (E19), descriptive)

- generated 2026-10-08T07:57:24+00:00 at git `f31a4bffd2b1` by `scripts/explore_e7_hard.py`; run `runs/e19_hard; runs/e19x_hard`; paired task bootstrap B = 10000, seed 0
- setting: families multi, 12 customers per task, rule `tier`, 60 near distractors, 20 noise lines, strict True, max steps 45, answer style `itemized`; seeds 310600-310799 (pooled parts: first 180 tasks, ext 20 tasks); System-2 `gemini-3.5-flash` via gemini; 12 workers, task-major
- spend: evaluation + smoke $53.79 (System-2 $53.79, Jev $0.000); the pilot counts under E7; cap $62
- GPU tariff q = $0.01144 per JevAgentRL episode (g5.2xlarge, $1.212/h × 1.887 h / 200 episodes; sensitivity: S2-4's q $0.00196, q = 0)
- registration: `docs/PREREG_A7_STAGE2.md`, Deviations "E19: the rule-randomised model on DecisionGym-Hard". No test; outside every Holm table.
- **route:** Flash on the direct Gemini API, as E7f's arms (the same route and list prices); rl_random = AgentDecision-4B (random rules) (E12b's checkpoint, never trained on the tier rule) served on the host at thresholds refitted on E7d's 20 calibration tasks; llm_only is E7f's logged arm on the same tasks, not re-run; the task asks for one line per customer before the total

- tuning target: Flash alone 60-80 % on the evaluated setting; the development pilot gave 46.7 % (n = 60, e7_pilot_tier_k12, e7_pilot_tier_k12b); the evaluation gives 83.5 % (n = 200), outside the band
- `llm_only`: n = 200, success 83.5 %, $264.3 per 1k episodes, wall 74.2 s mean / 63.2 s median; failures 33 failed: wrong final total 33, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 33})

| arm | n | success % | Δ pt [95 % CI] | Newcombe | wins / losses | $/1k ep. (ref) | saving % [95 % CI] | tool coverage % | acc. % (accepted) | reflection fired % | wall mean / median s (ref) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `rl_random` | 200 | 84.5 | +1.0 [-4.5, +7.0] | [-4.9, +6.9] | 18 / 16 | 277.5 (264.3) | -5.0 [-7.1, -2.8] | 43.4 | 99.0 (100.0) | 14.2 | 389.4 / 389.9 (74.2 / 63.2) |

Cost: API as logged (System-2 via gemini at the list prices of configs/prices/models.yaml; Jev at its list price); the JevAgentRL arm adds the GPU tariff q per episode (its API-only saving and the sensitivity rows are below). Coverage: share of tool decisions (tool choice and arguments) executed from the decision model's answer. Acc.: the model's answers against the gold label on the run's own states (all of its tool answers; in brackets the accepted ones).

## Per arm

### `rl_random`

- paired n = 200 (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 84.5 % vs 83.5 %
- failures (all complete episodes): 31 failed: wrong final total 31, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 31}); `llm_only`: 33 failed: wrong final total 33, wrong state-changing call 0, ask / finish error 0, other 0 (types {'wrong_total': 33})
- API cost $266.1 per 1k episodes, saving -0.7 % [-2.7, 1.5]
- sensitivity q = s2_4: $268.1 per 1k, saving -1.4 % [-3.5, 0.8]
- sensitivity q = zero: $266.1 per 1k, saving -0.7 % [-2.7, 1.5]
- tool_select: 7400 decisions, coverage 32.0 %, model acc. 98.9 % (n gold 7400), accepted 100.0 %, LLM on deferred 100.0 %
- tool_params: 3875 decisions, coverage 65.2 %, model acc. 99.1 % (n gold 3875), accepted 100.0 %, LLM on deferred -- %
- tool: 11275 decisions, coverage 43.4 %, model acc. 99.0 % (n gold 11275), accepted 100.0 %, LLM on deferred 100.0 %
- cancel-vs-ship decisions (tool choice where the gold next tool is a write; the tier rule): 2400, coverage 33.4 %, model acc. 96.6 %, accepted 100.0 % (0 accepted wrong), LLM on deferred 100.0 %; the model named the other write 81 times
- reflection trigger: 7200 questions, fired 14.2 %, acc. 99.5 % (gold positive 0.5 %)
- per episode: 37.0 steps, 37.1 LLM calls, 92.4 System-1 calls (336.5 s), System-2 $266.1 / System-1 $0.00 per 1k
- vs `rl_tier` (vs_tier: the logged arm of `e7f_hard; e7fx_hard` on the same tasks): paired n = 200, success 84.5 % vs 81.5 %, Δ +3.0 [-3.5, +9.5], wins / losses 26 / 20; API cost $266.1 vs $229.6 per 1k (saving -15.9 %, GPU tariffs of two hosts not added); wall 389.4 vs 327.2 s

## Final-answer form

Form of each complete episode's final answer (`answer_form`): the amount alone, prose with one or two amounts (e.g. only the total), or itemized (per-customer lines plus the total). Success is the arm's success among those answers.

| arm | itemized: n (share %), success % | sentence: n (share %), success % | number: n (share %), success % | none: n (share %), success % |
|---|---|---|---|---|
| `llm_only` | 200 (100.0), 83.5 | 0 (0.0), -- | 0 (0.0), -- | 0 (0.0), -- |
| `rl_random` | 200 (100.0), 84.5 | 0 (0.0), -- | 0 (0.0), -- | 0 (0.0), -- |


## Latency breakdown (descriptive; changes no reported number)

Calls per arm grouped by component / decision point / served provider (System-2: OpenRouter's `meta.served_provider`; `1tool` = the forced single-tool argument call of a delegated tool); tail = calls over 5 s; shares are of the arm's summed episode wall time.

### `llm_only`: wall 74.2 s per episode; System-1 0.0 s, System-2 70.7 s, System-2 tail 22.9 s per episode

| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |
|---|---|---|---|---|---|---|---|
| s2/tool_select/gemini | 7401 | 1.15 | 4.02 | 1.91 | 565 | 95.3 | 30.9 |

### `rl_random`: wall 389.4 s per episode; System-1 336.5 s, System-2 48.4 s, System-2 tail 9.8 s per episode

| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |
|---|---|---|---|---|---|---|---|
| s1/tool_select/agentdecision-4b-random | 7400 | 3.91 | 4.93 | 3.85 | 624 | 36.6 | 4.3 |
| s1/reflection/agentdecision-4b-random | 7200 | 3.60 | 4.64 | 3.53 | 285 | 32.7 | 1.9 |
| s1/tool_params/agentdecision-4b-random | 3875 | 3.53 | 4.52 | 3.45 | 110 | 17.1 | 0.7 |
| s2/tool_select/gemini | 5035 | 1.00 | 1.45 | 1.17 | 36 | 7.6 | 0.8 |
| s2/tool_params/1tool/gemini | 1357 | 0.78 | 1.52 | 1.41 | 43 | 2.5 | 0.9 |
| s2/reflection/gemini | 1019 | 1.15 | 1.49 | 1.82 | 23 | 2.4 | 0.9 |

- `rl_random` vs `llm_only`: +315.2 s per episode (System-1 +336.5 s, System-2 -22.3 s); the System-2 tail over 5 s adds -13.1 s, -4.2 % of the extra time


## Diagnosis view (as results/explore/e7f/DIAGNOSIS.md; descriptive): `rl_random` in `runs/e19_hard; runs/e19x_hard`

- episodes: 200 complete of 200 tasks
- thresholds ran with: tool_select cascade tau 0.9999538704122877; tool_params cascade tau 0.9862272143363953; reflection trigger tau 0.5
- tool_select: 7400 cascade decisions; accepted / deferred as the threshold says: 7400 (of which 0 deferred because the answer was OTHER); violations 0
- tool_params: 3875 cascade decisions; accepted / deferred as the threshold says: 3875 (of which 937 deferred because the answer was OTHER); violations 0
- System-1 endpoint per episode: {"backend": "agentdecision-4b-random", "base_url": "http://127.0.0.1:8081", "extra_body": {}, "model": "agentdecision-4b-random", "provider": "typesafe"} x 200
- System-1 calls by backend / model: agentdecision-4b-random / agentdecision-4b-random x 18475
- write states (gold next tool cancel_order / ship_order): 2400; both writes offered 2400; the policy sentence in the task 2400; the order identified from the current observation 2400, and of those the customer's tier and the amount both in the state 2400; longest state 5263 characters (harness cap 12000; states at the cap 0)

| tier \| amount vs. the tier's limit \| gold | n | model wrong | accepted wrong |
|---|---|---|---|
| basic | above limit | gold cancel_order | 545 | 2 (0.4 %) | 0 |
| basic | at or below limit | gold ship_order | 248 | 1 (0.4 %) | 0 |
| gold | gold tier | gold ship_order | 778 | 0 (0.0 %) | 0 |
| silver | above limit | gold cancel_order | 273 | 0 (0.0 %) | 0 |
| silver | at or below limit | gold ship_order | 556 | 78 (14.0 %) | 0 |

Cancel rate by tier and amount band ($150 bands; the rule cancels basic above $300, silver above $600, never gold):

| tier \| amount band ($) | n | model says cancel % | rule cancels % |
|---|---|---|---|
| basic | 0-150 | 109 | 0.0 | 0.0 |
| basic | 150-300 | 139 | 0.7 | 0.0 |
| basic | 300-450 | 147 | 99.3 | 100.0 |
| basic | 450-600 | 135 | 99.3 | 100.0 |
| basic | 600-750 | 135 | 100.0 | 100.0 |
| basic | 750-900 | 128 | 100.0 | 100.0 |
| gold | 0-150 | 97 | 0.0 | 0.0 |
| gold | 150-300 | 147 | 0.0 | 0.0 |
| gold | 300-450 | 136 | 0.0 | 0.0 |
| gold | 450-600 | 138 | 0.0 | 0.0 |
| gold | 600-750 | 137 | 0.0 | 0.0 |
| gold | 750-900 | 123 | 0.0 | 0.0 |
| silver | 0-150 | 131 | 0.0 | 0.0 |
| silver | 150-300 | 141 | 0.0 | 0.0 |
| silver | 300-450 | 140 | 25.7 | 0.0 |
| silver | 450-600 | 144 | 29.2 | 0.0 |
| silver | 600-750 | 132 | 100.0 | 100.0 |
| silver | 750-900 | 141 | 100.0 | 100.0 |

Failed episodes by type, with the decision that made the last step:

- 27 x wrong_total: last step finish chosen by the LLM
- 4 x wrong_total: last step finish chosen by the model


## By part (each part's own paired comparison; the table above pools them)

| part | run | seeds | arm | n | LLM-only success % | arm success % | Δ pt [95 % CI] | wins / losses | saving % [95 % CI] | wall mean s (ref) | spend $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| first | `runs/e19_hard` | 310600-310779 | `rl_random` | 180 | 83.9 | 85.0 | +1.1 [-5.0, +7.2] | 17 / 15 | -4.8 [-7.0, -2.5] | 394.9 (73.4) | 47.80 |
| ext | `runs/e19x_hard` | 310780-310799 | `rl_random` | 20 | 80.0 | 80.0 | +0.0 [-15.0, +15.0] | 1 / 1 | -6.9 [-13.1, -0.2] | 340.2 (81.4) | 5.42 |

## Registered reading (E19 (k))

- paired n = 200, wins 18, losses 16, Δ +1.0 points (Tango 95 % [-4.9, +6.9]); certified ordered p = 0.091 at the 3-point margin (A6.21 rule bb_ordered): **non-inferiority at 3 points not shown on the 200 tasks**
