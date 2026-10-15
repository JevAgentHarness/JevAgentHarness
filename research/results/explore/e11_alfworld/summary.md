# E11: the training recipe on ALFWorld (exploratory (E11))

- **descriptive, revises no registered result; a new checkpoint, not AgentDecision-4B.** Outside every Holm table; no test. Generated 2026-10-07T10:20:49+00:00 at git `687ddb765d13` by `scripts/explore_e11_alfworld.py`.
- Data (ALFWorld train games only): 9159 train rows / 965 development rows; sources flash 360 episodes (success 89.7 %), scripted 600 episodes (success 96.8 %); train rows after an off-plan step 78.1 %; rows dropped from the expert's first fallback on 1742, duplicates dropped 505.
- Training: Qwen3-4B + LoRA (r 32), 9159 rows x 2 epochs, soft CE + 0.5 Brier, lr 1e-4, 158 min (gc_3072).
- CASCADE threshold on 48 calibration train games: tau-hat = 0.1208 (rule `projection`, 711 paired decisions); on those states the trained model matched the gold at 71.0 %, Flash at 40.5 %.
- GPU tariff q = $0.00213 per GPU-arm episode (g5.4xlarge, 0.367 h / 280 episodes); sensitivity: q = 0 and S2-4's q.
- LLM only: success 90.7 % (n 140, errored 0), $0.0194 per episode, wall mean / median 37.0 / 25.4 s, 13.1 steps; its executed action matched the single gold action at 46.4 % of its steps (a lower bound).

| arm | n | success % | Δ pt [95 % bootstrap] | wins / losses | $ / episode API (+ q) | saving % with q [95 %] | saving % API only | wall s mean / median | steps | LLM calls | coverage % | acc. on taken % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cascade (`e11_action_cascade_tauhat`) | 140 | 95.0 | +4.3 [-2.1, +10.7] | 13 / 7 | 0.0000 (+ 0.0021) | 89.0 [85.6, 91.4] | 100.0 | 18.2 / 13.7 | 12.1 | 0.0 | 100.0 | 64.6 |
| replace (`e11_action_replace`) | 140 | 95.7 | +5.0 [-0.7, +11.4] | 13 / 6 | 0.0000 (+ 0.0021) | 89.0 [85.6, 91.4] | 100.0 | 17.9 / 14.5 | 12.1 | 0.0 | 100.0 | 64.7 |

## Offline, development rows (held-out train games)

| model | file | n | acc. % | ECE | mean conf. |
|---|---|---|---|---|---|
| trained | dev | 965 | 84.7 | 0.026 | 0.839 |
| trained | dev_off_plan | 741 | 88.1 | 0.029 | 0.866 |
| trained | dev_flash | 405 | 81.5 | 0.047 | 0.831 |
| base | dev | 965 | 45.4 | 0.516 | 0.970 |
| base | dev_off_plan | 741 | 46.2 | 0.504 | 0.965 |
| base | dev_flash | 405 | 39.0 | 0.576 | 0.966 |

## Success by task type (descriptive, small n)

| task type | n | LLM only | cascade | replace |
|---|---|---|---|---|
| pick_and_place_simple | 35 | 100 | 97 | 97 |
| look_at_obj_in_light | 13 | 100 | 100 | 100 |
| pick_clean_then_place_in_recep | 27 | 67 | 100 | 100 |
| pick_heat_then_place_in_recep | 16 | 94 | 81 | 81 |
| pick_cool_then_place_in_recep | 25 | 88 | 92 | 96 |
| pick_two_obj_and_place | 24 | 100 | 96 | 96 |

E6a (Jev, same games, earlier run): LLM only 91.4 %; Jev cascade 91.4 % (+0.0 pt, saving 14.0 %); Jev alone 27.1 %.

## Spend (logged, list prices)

Gemini $11.61 (1434 records incl. errored attempts) over `e11_harvest_flash` $7.80, `e11_harvest_scripted` $0.00, `e11_alfworld_cal` $1.08, `e11_alfworld_smoke` $0.01, `e11_alfworld_eval` $2.72; cap $60.

