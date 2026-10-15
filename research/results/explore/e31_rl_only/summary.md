# E31: GRPO in the agent loop without SFT (RL only, from the untrained Qwen3-4B)

Generated 2026-10-10T15:20:47+00:00 by `python scripts/explore_e31.py summarize`. exploratory (E31), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-10 'E31: GRPO in the agent loop without SFT (RL only, from the untrained base)'; outside every Holm table.

| Row | L1 | C | shift L2 | must-ask | shift L3 |
|---|---|---|---|---|---|
| Qwen3-4B, untrained | 71.4 | 50.1 | 52.9 | 27.6 | 13.5 |
| RL only (E31) | 71.4 | 48.5 | 50.2 | 27.6 | 11.5 |
| + SFT v1 | 78.4 | 99.2 | 98.4 | 95.0 | 96.0 |
| SFT + GRPO (v1) | 78.8 | 99.2 | 98.6 | 90.8 | 96.0 |

Share of SFT v1's gain over the untrained model that RL alone reaches: l1 0 %, c -3 %, l2 -6 %, mustask 0 %, l3 -2 %
Reading (registration (e)): {'l2_within_3pt': False, 'l3_lower_end_ok': False, 'matches_sft': False}

## Training

- iterations 15; GRPO stage 95.6 min; API $21.34 (rollouts $20.28, greedy evaluations $1.06)
- zero-variance group share 75.0 % (v1's first 15: 60.0 %)
- decisions 9608, made by the model 9429.0, updated 3274; iterations without an update 0
- rollout success first 3 / last 3 iterations 38.0 / 39.6 %; decision accuracy vs the answer key 70.4 / 80.3 %
- greedy development success by iteration: {0: {'rl_ts_params_replace': 0.125, 'rl_ts_assist_k3': 0.3125}, 10: {'rl_ts_params_replace': 0.125, 'rl_ts_assist_k3': 0.25}, 15: {'rl_ts_params_replace': 0.125, 'rl_ts_assist_k3': 0.3541666666666667}}
- stop record: None

| it | reward | success | zero-std | decisions | made | updated | defer | dec acc | KL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.286 | 48.4 | 87.5 | 329 | 321 | 110 | 11.1 | 79.7 | 0.00608 |
| 2 | 0.231 | 37.5 | 62.5 | 1127 | 1122 | 493 | 1.1 | 57.2 | 0.00865 |
| 3 | 0.050 | 28.1 | 87.5 | 603 | 595 | 105 | 10.0 | 74.2 | 0.02398 |
| 4 | 0.112 | 28.1 | 62.5 | 982 | 974 | 476 | 2.3 | 60.4 | 0.01769 |
| 5 | 0.253 | 50.0 | 75.0 | 957 | 943 | 350 | 20.0 | 61.2 | 0.01037 |
| 6 | 0.380 | 57.8 | 62.5 | 547 | 542 | 258 | 4.8 | 83.1 | 0.09957 |
| 7 | 0.052 | 25.0 | 62.5 | 656 | 640 | 354 | 33.3 | 71.7 | 0.11485 |
| 8 | 0.044 | 25.0 | 75.0 | 407 | 401 | 107 | 5.6 | 87.2 | 0.04264 |
| 9 | 0.155 | 39.1 | 75.0 | 613 | 568 | 286 | 16.2 | 82.8 | 0.20009 |
| 10 | 0.243 | 46.9 | 75.0 | 371 | 363 | 110 | 11.1 | 94.4 | 0.11003 |
| 11 | 0.197 | 37.5 | 87.5 | 944 | 928 | 96 | 33.3 | 59.2 | 0.2135 |
| 12 | 0.184 | 37.5 | 87.5 | 524 | 524 | 132 | 0.0 | 89.5 | 0.79905 |
| 13 | 0.256 | 43.8 | 87.5 | 744 | 728 | 98 | 4.7 | 69.0 | 0.01617 |
| 14 | 0.131 | 37.5 | 75.0 | 424 | 400 | 151 | 36.9 | 81.9 | 0.88266 |
| 15 | 0.122 | 37.5 | 62.5 | 380 | 380 | 148 | 0.0 | 90.1 | 0.06814 |

## L3 shift

n 200 (errored 0), success 11.5 %, own-state accuracy 40.0 %, Gemini $0.8842491000000002
- vs Qwen3-4B, untrained: -2.0 [-5.8, +1.7] (n 200)
- vs + SFT v1: -84.5 [-89.0, -77.7] (n 200)
- vs SFT + GRPO (v1): -84.5 [-88.7, -78.1] (n 200)

Logged reference values (cross-check): {'base': {'c': 0.5010455808575052, 'l3': 0.135}, 'sft_v1': {'c': 0.9919093147483258, 'l3': 0.96}, 'rl': {'c': 0.9920083177147069, 'l3': 0.96}}
