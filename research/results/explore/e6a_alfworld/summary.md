# E6a: action-only control on ALFWorld valid_seen (exploratory (E6a))

- **descriptive, revises no registered result, not the registered S2-L3b.** Outside every Holm table; no test. Generated 2026-10-06T06:23:50+00:00 at git `6b8175ef6fa5` by `scripts/explore_e6a_alfworld.py`.
- Games: 140 valid_seen games (list sha256 `5a914a32fa171b1d`); run `runs/e6a_alfworld_eval`. Agent LLM gemini-3.5-flash (temperature 0, reasoning_effort minimal); decision model Jev via OpenRouter (typesafe/jev-1.13); max 50 steps; task_major, seed 732.
- CASCADE threshold fitted on 48 ALFWorld train games before any valid_seen episode: tau-hat = 0.9500 (rule `projection`, 707 paired decisions, V = 0.09 [counterfactual], dC = $0.00148).
- LLM only: success 91.4 % (n 140, errored 0), $0.0184 per episode, wall time mean / median 24.1 / 13.7 s, 12.7 steps; the LLM's executed action matched the single gold action at 46.7 % of its steps (a lower bound).

| arm | n | success % | Δ pt vs LLM only [95 % bootstrap] | Newcombe | wins / losses | $ / episode (Gemini + Jev) | saving % [95 %] | wall s mean / median | steps | LLM calls | DM calls | coverage % | DM acc. % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| replace (`jev_action_replace`) | 140 | 27.1 | -64.3 [-72.1, -56.4] | [-71.4, -55.2] | 0 / 90 | 0.0017 (0.0000 + 0.0017) | 90.7 [88.0, 92.6] | 18.7 / 19.8 | 40.5 | 0.0 | 40.5 | 100.0 | 17.7 |
| cascade (`jev_action_cascade_tauhat`) | 140 | 91.4 | +0.0 [-4.3, +3.6] | [-4.7, +4.7] | 4 / 4 | 0.0159 (0.0153 + 0.0005) | 14.0 [-2.8, 28.2] | 22.3 / 12.9 | 13.2 | 10.1 | 13.2 | 23.3 | 73.5 |
| assist (`jev_action_assist_k3`) | 140 | 50.7 | -40.7 [-49.3, -32.1] | [-48.9, -31.9] | 1 / 58 | 0.0520 (0.0507 + 0.0013) | -182.0 [-267.3, -122.6] | 57.1 / 53.1 | 30.0 | 30.0 | 30.0 | 0.0 | 37.5 |

Reference cost per episode on the same pairs: replace $0.0184; cascade $0.0184; assist $0.0184.
Coverage = share of the arm's action decisions executed from the decision model's answer (ASSIST: 0 by definition; it shortlists k = 3 actions at every step and the LLM chooses). DM acc. = accuracy against the single gold action (lower bound): REPLACE its answer, CASCADE its accepted answers, ASSIST gold in its top 3.

## Success by task type (descriptive, small n)

| task type | n | LLM only | replace | cascade | assist |
|---|---|---|---|---|---|
| pick_and_place_simple | 35 | 100 | 46 | 100 | 71 |
| look_at_obj_in_light | 13 | 100 | 38 | 100 | 46 |
| pick_clean_then_place_in_recep | 27 | 74 | 30 | 78 | 37 |
| pick_heat_then_place_in_recep | 16 | 94 | 19 | 88 | 56 |
| pick_cool_then_place_in_recep | 25 | 84 | 8 | 84 | 24 |
| pick_two_obj_and_place | 24 | 100 | 17 | 100 | 62 |

## Spend (logged, list prices)

Total $14.09 (Gemini $13.55, Jev $0.542; 665 records incl. errored attempts) over `e6a_alfworld_cal` $1.07, `e6a_alfworld_smoke` $0.12, `e6a_alfworld_eval` $12.33, `e6a_alfworld_cal_container_aborted` $0.57; cap $80.

