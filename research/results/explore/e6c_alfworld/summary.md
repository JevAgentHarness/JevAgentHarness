# E6c: the action guard on ALFWorld valid_seen (exploratory (E6c))

- **descriptive, revises no registered result, not the registered S2-L3b.** Outside every Holm table; no test. Generated 2026-10-06T07:58:50+00:00 at git `715fa3372270` by `scripts/explore_e6c_alfworld.py`.
- Games: 140 valid_seen games (list sha256 `5a914a32fa171b1d`, E6a's list and order); run `runs/e6c_alfworld_eval`. Agent LLM gemini-3.5-flash (temperature 0, reasoning_effort minimal); decision model Jev via OpenRouter (typesafe/jev-1.13); max 50 steps; task_major, seed 732.
- GUARD: the LLM proposes an admissible action; Jev vetoes it when p_yes < tau = 0.5 (fixed: the Stage-1 tool guard's tau on Flash (jev_ts_guard, E2a)); the LLM regenerates once with the rejection note and the rewrite is executed unchecked.
- LLM only: success 90.7 % (n 140, errored 0), $0.0191 per episode, wall time mean / median 37.2 / 24.6 s, 13.0 steps; its executed action matched the single gold action at 45.7 % of its steps (a lower bound).

| arm | n | success % | Δ pt vs LLM only [95 % bootstrap] | Newcombe | wins / losses | $ / episode (Gemini + Jev) | saving % [95 %] | wall s mean / median | steps | LLM calls | DM calls | coverage (vetoed) % | verdict acc. % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| guard (`jev_action_guard`, tau 0.5) | 140 | 90.0 | -0.7 [-4.3, +2.9] | [-5.2, +3.7] | 3 / 4 | 0.0357 (0.0353 + 0.0004) | -86.7 [-127.4, -56.6] | 56.9 / 36.1 | 15.1 | 22.1 | 15.1 | 46.5 | 65.5 | no saving |

Reference on the same pairs: $0.0191 per episode, wall time mean / median 37.2 / 24.6 s; paired median wall-time difference +8.6 s.
Guard decisions: 2114 proposals checked, 983 vetoed (7.02 per episode); executed from the model's answer: 0.0 % (0 by definition). Against the single gold action (lower bounds; 2114 proposals with a gold action): verdict accuracy 65.5 %, the LLM's proposals 44.6 %, vetoed proposals that were not the gold action 72.4 %, non-gold proposals vetoed 60.8 %, rewrites of vetoed proposals 17.1 % (the rewrite repeated the vetoed action 24.7 %), executed actions 39.7 %; mean p_yes 0.546.
Verdict: the module map's fixed descriptive rule (scripts/module_map.py `verdict`; activity = proposals vetoed).

E6a's `llm_only` on the same games (another run of the same configuration; descriptive, not paired): success 91.4 % (n 140); per game, 137 of 140 outcomes agree with this run's reference (E6a only 2, E6c only 1).

## Success by task type (descriptive, small n)

| task type | n | LLM only | guard |
|---|---|---|---|
| pick_and_place_simple | 35 | 100 | 100 |
| look_at_obj_in_light | 13 | 100 | 100 |
| pick_clean_then_place_in_recep | 27 | 74 | 74 |
| pick_heat_then_place_in_recep | 16 | 94 | 94 |
| pick_cool_then_place_in_recep | 25 | 80 | 76 |
| pick_two_obj_and_place | 24 | 100 | 100 |

## Spend (logged, list prices)

Total $7.70 (Gemini $7.65, Jev $0.056; 284 records incl. errored attempts) over `e6c_alfworld_smoke` $0.03, `e6c_alfworld_eval` $7.67; cap $25.

