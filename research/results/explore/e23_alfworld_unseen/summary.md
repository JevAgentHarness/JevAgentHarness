# E23: E11's checkpoint in ALFWorld's unseen rooms (exploratory (E23))

- **descriptive, revises no registered result; E11's frozen checkpoint and threshold, no refit, no training.** Outside every Holm table; no confirmatory test. Generated 2026-10-08T19:48:58+00:00 at git `c541ab6f2d6c` by `scripts/explore_e23_alfworld_unseen.py`.
- Games: valid_unseen, 134 games (list sha256 3c46e12b31bfcb62; registered 3c46e12b31bfcb62); scenes {'10': 77, '219': 11, '308': 27, '424': 19} (none in E11's training or calibration games).
- Model: e11-alfworld-4b (E11's frozen checkpoint; serve_jev); checkpoint sha256 check against E11's record: ok; CASCADE at E11's tau-hat 0.120764 (no refit).
- GPU tariff q = $0.00163 per GPU-arm episode (g5.2xlarge, 0.360 h / 268 episodes); sensitivity: q = 0 and S2-4's q. E11 (valid_seen): q = $0.00213.
- LLM only, unseen: success 88.8 % (n 134, errored 0), $0.0215 per episode, wall mean / median 35.1 / 22.7 s, 14.8 steps; its executed action matched the single gold action at 36.9 % of its steps. Seen (E11): 90.7 %, $0.0194, 37.0 / 25.4 s, 13.1 steps.

## Unseen rooms (E23) beside valid_seen (E11)

| arm | split | n | success % (ref.) | Δ pt [95 % bootstrap] | wins / losses | NI (3 pt): p, verdict [Tango 95 %] | saving % with q [95 %] | saving % API only | wall s mean / median (ref.) | steps | LLM calls | coverage % | taken ≠ gold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cascade | unseen | 134 | 89.6 (88.8) | +0.7 [-6.7, +8.2] | 14 / 13 | 0.175, not shown [-7.1, +8.7] | 92.0 [89.7, 93.6] | 99.6 | 18.2 / 12.5 (35.1 / 22.7) | 16.4 | 0.1 | 99.4 | 805 of 2178 (37.0 %) |
| cascade | seen (E11) | 140 | 95.0 (90.7) | +4.3 [-2.1, +10.7] | 13 / 7 | 0.014, non-inferior [-2.1, +11.1] | 89.0 [85.6, 91.4] | 100.0 | 18.2 / 13.7 (37.0 / 25.4) | 12.1 | 0.0 | 100.0 | -- |
| replace | unseen | 134 | 88.1 (88.8) | -0.7 [-8.2, +7.5] | 14 / 15 | 0.301, not shown [-8.9, +7.4] | 92.4 [90.2, 93.9] | 100.0 | 18.8 / 12.9 (35.1 / 22.7) | 16.7 | 0.0 | 100.0 | 838 of 2239 (37.4 %) |
| replace | seen (E11) | 140 | 95.7 (90.7) | +5.0 [-0.7, +11.4] | 13 / 6 | 0.007, non-inferior [-1.2, +11.7] | 89.0 [85.6, 91.4] | 100.0 | 17.9 / 14.5 (37.0 / 25.4) | 12.1 | 0.0 | 100.0 | -- |

Seen-to-unseen change in success (points; unpaired, different games): llm_only -1.9, cascade -5.4, replace -7.7.

## The trained model's actions (unseen)

| arm | actions | taken | taken with gold | ≠ gold | deferred | mean conf. | conf. < tau-hat % | acc. on taken % | deferred LLM acc. % |
|---|---|---|---|---|---|---|---|---|---|
| cascade | 2191 | 2178 | 2178 | 805 | 13 | 0.785 | 0.6 | 63.0 | 0.0 |
| replace | 2239 | 2239 | 2239 | 838 | 0 | 0.785 | 0.6 | 62.6 | -- |

## Success by task type (descriptive, small n; seen = E11 on valid_seen)

| task type | n unseen | LLM only | cascade | replace | n seen | LLM only | cascade | replace |
|---|---|---|---|---|---|---|---|---|
| pick_and_place_simple | 24 | 100 | 88 | 88 | 35 | 100 | 97 | 97 |
| look_at_obj_in_light | 18 | 100 | 78 | 78 | 13 | 100 | 100 | 100 |
| pick_clean_then_place_in_recep | 31 | 68 | 94 | 90 | 27 | 67 | 100 | 100 |
| pick_heat_then_place_in_recep | 23 | 87 | 96 | 91 | 16 | 94 | 81 | 81 |
| pick_cool_then_place_in_recep | 21 | 90 | 86 | 86 | 25 | 88 | 92 | 96 |
| pick_two_obj_and_place | 17 | 100 | 94 | 94 | 24 | 100 | 96 | 96 |

Games played before E23 in development (pilot P3b; never reported), success here: `pick_and_place_simple-SaltShaker-None-Cabinet-10` llm_only True, e11_action_cascade_tauhat True, e11_action_replace True; `pick_heat_then_place_in_recep-Potato-None-GarbageCan-10` llm_only True, e11_action_cascade_tauhat True, e11_action_replace True; `pick_clean_then_place_in_recep-Bowl-None-Cabinet-10` llm_only True, e11_action_cascade_tauhat True, e11_action_replace True; `look_at_obj_in_light-Book-None-DeskLamp-308` llm_only True, e11_action_cascade_tauhat False, e11_action_replace False; `pick_cool_then_place_in_recep-Bread-None-CounterTop-10` llm_only True, e11_action_cascade_tauhat True, e11_action_replace True.

## Spend (logged, list prices) and host

Gemini $2.90 (408 records incl. errored attempts; probes $0.0006) over `e23_alfworld_smoke` $0.01, `e23_alfworld_unseen_eval` $2.89; cap $15. Host g5.2xlarge <instance-id>: 0.74 host-hours (2026-10-08T18:57:29Z to 2026-10-08T19:42:12Z); cap 4.

