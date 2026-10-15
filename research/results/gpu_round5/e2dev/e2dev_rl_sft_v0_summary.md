| arm | run | n | success | $/ep | S2 calls | S1 ms/call | stop/guard | stop/shadow | tool_params/replace | tool_params/shadow | tool_select/assist | tool_select/guard | tool_select/replace | tool_select/shadow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours_shadow | e2dev_rl_sft_v0 | 50 | 1.000 | 0.0418 | 7.5 | 287 | - | 0.963 (375) | - | 0.931 (420) | - | - | - | 0.984 (375) |
| ours_stop_guard | e2dev_rl_sft_v0 | 50 | 0.940 | 0.0582 | 9.9 | 68 | 0.275 (171) | - | - | - | - | - | - | - |
| ours_stop_guard_s2lite | e2dev_rl_sft_v0 | 50 | 0.860 | 0.0130 | 9.9 | 65 | 0.491 (163) | - | - | - | - | - | - | - |
| ours_ts_assist_k3 | e2dev_rl_sft_v0 | 50 | 1.000 | 0.0174 | 7.6 | 427 | - | - | - | - | 0.976 (381) | - | - | - |
| ours_ts_assist_k3_s2lite | e2dev_rl_sft_v0 | 50 | 0.820 | 0.0036 | 7.8 | 1196 | - | - | - | - | 0.979 (388) | - | - | - |
| ours_ts_guard | e2dev_rl_sft_v0 | 50 | 1.000 | 0.0491 | 9.3 | 60 | - | - | - | - | - | 0.846 (402) | - | - |
| ours_ts_params_replace | e2dev_rl_sft_v0 | 50 | 0.780 | 0.0095 | 3.6 | 743 | - | - | 0.788 (552) | - | - | - | 0.948 (458) | - |
| ours_ts_replace | e2dev_rl_sft_v0 | 50 | 0.880 | 0.0164 | 7.8 | 473 | - | - | - | - | - | - | 0.903 (391) | - |
