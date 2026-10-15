| arm | run | n | success | $/ep | S2 calls | S1 ms/call | stop/guard | stop/shadow | tool_params/replace | tool_params/shadow | tool_select/assist | tool_select/guard | tool_select/replace | tool_select/shadow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_shadow | e2dev_rl_sft_v1 | 50 | 1.000 | 0.0432 | 7.5 | 384 | - | 1.000 (375) | - | 0.995 (424) | - | - | - | 1.000 (375) |
| ours_stop_guard | e2dev_rl_sft_v1 | 50 | 1.000 | 0.0406 | 7.5 | 76 | 1.000 (50) | - | - | - | - | - | - | - |
| ours_stop_guard_s2lite | e2dev_rl_sft_v1 | 50 | 0.920 | 0.0100 | 7.7 | 76 | 0.909 (55) | - | - | - | - | - | - | - |
| ours_ts_assist_k3 | e2dev_rl_sft_v1 | 50 | 0.980 | 0.0179 | 7.8 | 1053 | - | - | - | - | 1.000 (392) | - | - | - |
| ours_ts_assist_k3_s2lite | e2dev_rl_sft_v1 | 50 | 0.840 | 0.0037 | 7.9 | 1205 | - | - | - | - | 1.000 (395) | - | - | - |
| ours_ts_guard | e2dev_rl_sft_v1 | 50 | 1.000 | 0.0409 | 7.5 | 61 | - | - | - | - | - | 1.000 (375) | - | - |
| ours_ts_params_replace | e2dev_rl_sft_v1 | 50 | 1.000 | 0.0033 | 1.5 | 859 | - | - | 1.000 (531) | - | - | - | 1.000 (375) | - |
| ours_ts_replace | e2dev_rl_sft_v1 | 50 | 1.000 | 0.0145 | 7.5 | 1076 | - | - | - | - | - | - | 1.000 (375) | - |
