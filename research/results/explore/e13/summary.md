# E13: other agent LLMs on DecisionGym-Hard (exploratory (E13), descriptive)

- generated 2026-10-07T16:05:19+00:00 at git `f5b6486706c2` by `scripts/explore_e13_llms.py`; paired task bootstrap B = 10000, seed 0
- setting (E7e's, unchanged): multi tasks, 12 customers, `tier` rule, itemized final answer, 60 near distractors, 20 noise lines, strict, max steps 45; arms `llm_only` and `jev_refit` (Jev's tool-choice and argument cascades at thresholds refitted per LLM on its own calibration arm, plus the fixed reflection trigger tau 0.5); task-major (seed 732)
- registration: `docs/PREREG_A7_STAGE2.md`, Deviations "E13: other agent LLMs on DecisionGym-Hard". No test; outside every Holm table. Dollars are each route's list prices (the saving is paired within an LLM's row).

| agent LLM | runner | seeds | n | LLM-only % | Jev refit % | Δ pt [95 % CI] | wins / losses | $/1k ep. (LLM-only) | saving % [95 % CI] | wall mean s (LLM-only) | delegated % | acc. accepted % | τ̂ tool / args | spend $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-6-Luna | local | 311000-311099 | 100 | 45.0 | 83.0 | +38.0 [+26.0, +50.0] | 44 / 6 | 22.8 (2.7) | -737.4 [-814.1, -672.2] | 67.4 (53.9) | 82.0 | 99.1 | 0.49 / 0.61 | 1.46 + Jev 2.19 |
| DeepSeek-V4-Flash | collaborator | 311100-311199 | -- | | | | | | | | | | | not run |
| Gemini 3.5 Flash (E7e, reference) | -- | 310500-310599 | 100 | 87.0 | 89.0 | +2.0 [-6.0, +10.0] | 9 / 7 | 183.8 (264.6) | 30.5 [29.8, 31.3] | 104.1 (63.5) | 76.5 | 99.2 | 0.91 / 0.87 (E7c) | (E7e) |

Delegated: share of the tool decisions (tool choice and arguments) executed from Jev's answer. Acc. accepted: Jev's accuracy against the gold label on the decisions it took. The Flash row is E7e (other seeds, thresholds fitted for Flash in E7c), shown for context; it is not part of E13.

## GPT-6-Luna (luna)

- route: OpenAI Responses API directly (reasoning effort none, its lowest; temperature 0; stateless); Jev via OpenRouter on the author's Jev-only key
- runner: local
- caps: LLM $60, Jev $5
- source: `results/explore/e13`; started 2026-10-07T14:12:21+00:00 (host tag 5bce98f7, git 5462386fc831); finished 2026-10-07T15:48:44+00:00
- gate: run: calibration 100, evaluation 100; before the evaluation: run: calibration 100, evaluation 100
- spend: LLM $1.46, Jev $2.190 (smoke, calibration and evaluation)

- agent-LLM input billed as cache hits (descriptive): `llm_only` 95.5 % of 165,922 input tokens per episode, 29.1 steps; `jev_refit` 0.9 % of 87,740 input tokens per episode, 37.5 steps

### Cost diagnosis (descriptive; changes no registered number)

Agent-LLM calls of `jev_refit` by kind (100 episodes) against `llm_only`:

| arm | call kind | calls | share of calls % | share of input tokens % | input billed as cache hits % | calls with a hit | tokens written to the cache |
|---|---|---|---|---|---|---|---|
| `jev_refit` | forced argument calls of delegated steps (one tool offered) | 1,861 | 87.8 | 86.2 | 0.7 | 18 | 7,197,123 |
| `jev_refit` | reflection calls (no tools) | 255 | 12.0 | 13.4 | 0.3 | 1 | 1,154,891 |
| `jev_refit` | the LLM's own tool-choice calls (every tool) | 4 | 0.2 | 0.4 | 78.6 | 4 | 7,429 |
| `llm_only` | the LLM's own tool-choice calls (every tool) | 2,906 | 100.0 | 100.0 | 95.5 | 2,809 | 744,472 |

- The forced argument call of a delegated step offers ONE tool (`agent._delegated_call`), so its prompt prefix (the tool list comes first) differs from every earlier call and the provider's prefix cache cannot hit; LLM-only's calls all offer the same list. The history is appended the same way in both arms (the reflection note, appended to the last tool message when the trigger fires, is the only other prefix change).
- Cost gap: $20.1 per 1k episodes (the cascade costs 8.4 times LLM-only); Jev's own calls are $13.7 of it (68.2 %), the agent LLM $9.1 against $2.7 for LLM-only.
- Had the cascade's LLM input been billed at LLM-only's effective price per input token (a perfect prefix-cache fix), its LLM part would be about $1.4 and the arm about $15.2 per 1k episodes: still about 5.6 times LLM-only. With this LLM the cascade's gain is success, not cost. (`tool_params.extra.full_tools`, off by default, now offers the full list in the forced call; it has not been run.)
- Not in the logged costs: cache writes are billed at 1.25 times the input price; the surcharge is about $0.2 per 1k episodes for `llm_only`, $2.1 per 1k episodes for `jev_refit`.
- order: fit written 2026-10-07T14:56:05+00:00, first evaluation invocation 2026-10-07T15:11:26+00:00: fit first, as registered

### Calibration fit

Generated 2026-10-07T14:56:04+00:00; run `runs/e13_luna_cal`; estimator: threshold_theory --calibration-arm --task-value 1.0 --lam-per-hour 0 --v-fallback W (A6.5 / A6.11).

| calibration arm | decision point | tau-hat | rule | paired | V | dC | coverage at tau % | model acc. taken % | LLM acc. taken % | model acc. all % | LLM acc. all % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `jev_shadow_cal` | tool_select | 0.49 | projection | 2790 | 0.7543859649122807 | 9.007662365591399e-05 | 100.0 | 99.9 | 98.5 | 99.9 | 98.5 |
| `jev_shadow_cal` | tool_params | 0.61 | projection | 4469 | 1.0 | 5.311432535242784e-05 | 90.2 | 100.0 | 100.0 | 97.2 | 100.0 |

Cancel-vs-ship decisions (tool choice where the gold next tool is a write), at the fitted tau:

- `jev_shadow_cal`: 911 decisions, coverage 100.0 %, model acc. taken 100.0 %, model acc. all 100.0 %, LLM acc. all 95.3 %

### Evaluation (E7's readout)

#### E13: GPT-6-Luna on DecisionGym-Hard (exploratory (E13), descriptive)

- generated 2026-10-07T16:05:20+00:00 at git `f5b6486706c2` by `scripts/explore_e7_hard.py`; run `results/explore/e13/records/luna/e13_luna_eval`; paired task bootstrap B = 10000, seed 0
- setting: families multi, 12 customers per task, rule `tier`, 60 near distractors, 20 noise lines, strict True, max steps 45, answer style `itemized`; seeds 311000-311099; System-2 `gpt-6-luna` via openai_responses; 6 workers, task-major
- spend: calibration + smoke + evaluation $3.65 (System-2 $1.46, Jev $2.190); cap $65
- registration: `docs/PREREG_A7_STAGE2.md`, Deviations "E13: other agent LLMs on DecisionGym-Hard". No test; outside every Holm table.
- **route:** OpenAI Responses API directly (reasoning effort none, its lowest; temperature 0; stateless); Jev via OpenRouter on the author's Jev-only key

- `llm_only`: n = 100, success 45.0 %, $2.7 per 1k episodes, wall 53.9 s mean / 62.3 s median; failures 55 failed: wrong final total 11, wrong state-changing call 44, ask / finish error 0, other 0 (types {'strict_stop': 44, 'wrong_total': 11})

| arm | n | success % | Δ pt [95 % CI] | Newcombe | wins / losses | $/1k ep. (ref) | saving % [95 % CI] | tool coverage % | acc. % (accepted) | reflection fired % | wall mean / median s (ref) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `jev_refit` | 100 | 83.0 | +38.0 [+26.0, +50.0] | [+25.4, +48.9] | 44 / 6 | 22.8 (2.7) | -737.4 [-814.1, -672.2] | 82.0 | 97.7 (99.1) | 7.0 | 67.4 / 65.4 (53.9 / 62.3) |

Cost: API as logged (System-2 via openai_responses at the list prices of configs/prices/models.yaml; Jev at its list price); the JevAgentRL arm adds the GPU tariff q per episode (its API-only saving and the sensitivity rows are below). Coverage: share of tool decisions (tool choice and arguments) executed from the decision model's answer. Acc.: the model's answers against the gold label on the run's own states (all of its tool answers; in brackets the accepted ones).

#### Per arm

##### `jev_refit`

- paired n = 100 (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 83.0 % vs 45.0 %
- failures (all complete episodes): 17 failed: wrong final total 8, wrong state-changing call 1, ask / finish error 8, other 0 (types {'premature_finish': 7, 'wrong_total': 8, 'no_answer': 1, 'strict_stop': 1}); `llm_only`: 55 failed: wrong final total 11, wrong state-changing call 44, ask / finish error 0, other 0 (types {'strict_stop': 44, 'wrong_total': 11})
- API cost $22.8 per 1k episodes, saving -737.4 % [-814.1, -672.2]
- tool_select: 3755 decisions, coverage 99.9 %, model acc. 98.8 % (n gold 3752), accepted 98.9 %, LLM on deferred 50.0 %
- tool_params: 6065 decisions, coverage 70.9 %, model acc. 97.0 % (n gold 5964), accepted 99.2 %, LLM on deferred -- %
- tool: 9820 decisions, coverage 82.0 %, model acc. 97.7 % (n gold 9716), accepted 99.1 %, LLM on deferred 50.0 %
- cancel-vs-ship decisions (tool choice where the gold next tool is a write; the tier rule): 1182, coverage 100.0 %, model acc. 100.0 %, accepted 100.0 % (0 accepted wrong), LLM on deferred -- %; the model named the other write 0 times
- reflection trigger: 3655 questions, fired 7.0 %, acc. 99.7 % (gold positive 0.3 %)
- per episode: 37.5 steps, 21.2 LLM calls, 134.8 System-1 calls (32.6 s), System-2 $9.1 / System-1 $13.72 per 1k

#### Final-answer form

Form of each complete episode's final answer (`answer_form`): the amount alone, prose with one or two amounts (e.g. only the total), or itemized (per-customer lines plus the total). Success is the arm's success among those answers.

| arm | itemized: n (share %), success % | sentence: n (share %), success % | number: n (share %), success % | none: n (share %), success % |
|---|---|---|---|---|
| `llm_only` | 56 (56.0), 80.4 | 0 (0.0), -- | 0 (0.0), -- | 44 (44.0), 0.0 |
| `jev_refit` | 98 (98.0), 84.7 | 0 (0.0), -- | 0 (0.0), -- | 2 (2.0), 0.0 |


#### Latency breakdown (descriptive; changes no reported number)

Calls per arm grouped by component / decision point / served provider (System-2: OpenRouter's `meta.served_provider`; `1tool` = the forced single-tool argument call of a delegated tool); tail = calls over 5 s; shares are of the arm's summed episode wall time.

##### `llm_only`: wall 53.9 s per episode; System-1 0.0 s, System-2 52.6 s, System-2 tail 5.0 s per episode

| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |
|---|---|---|---|---|---|---|---|
| s2/tool_select/openai_responses | 2906 | 1.54 | 2.39 | 1.81 | 59 | 97.6 | 9.3 |

##### `jev_refit`: wall 67.4 s per episode; System-1 32.6 s, System-2 32.9 s, System-2 tail 2.2 s per episode

| group | n | p50 s | p90 s | mean s | > 5 s | share of wall % | tail share % |
|---|---|---|---|---|---|---|---|
| s2/tool_params/1tool/openai_responses | 1861 | 1.22 | 2.22 | 1.46 | 20 | 40.3 | 2.3 |
| s1/tool_params/typesafe/jev-1.13-20260917 | 6065 | 0.22 | 0.30 | 0.24 | 0 | 21.4 | 0.0 |
| s1/tool_select/typesafe/jev-1.13-20260917 | 3755 | 0.23 | 0.31 | 0.25 | 0 | 13.9 | 0.0 |
| s1/reflection/typesafe/jev-1.13-20260917 | 3655 | 0.22 | 0.31 | 0.24 | 0 | 13.1 | 0.0 |
| s2/reflection/openai_responses | 255 | 1.91 | 2.65 | 2.20 | 7 | 8.3 | 1.0 |
| s2/tool_select/openai_responses | 4 | 1.87 | 3.45 | 2.38 | 0 | 0.1 | 0.0 |

- `jev_refit` vs `llm_only`: +13.6 s per episode (System-1 +32.6 s, System-2 -19.7 s); the System-2 tail over 5 s adds -2.8 s, -20.5 % of the extra time


## DeepSeek-V4-Flash (deepseek)

- route: OpenRouter pinned to StreamLake (fp8), fallbacks off, reasoning off; Jev via OpenRouter
- runner: collaborator; run by a collaborator on their own keys (date in the execution record)
- caps: LLM $15, Jev $5
- state: not run
