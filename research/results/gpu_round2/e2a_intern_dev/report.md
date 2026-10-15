## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intern08_ts_assist_k3 | 10 | 0.4000 | 0.1000 | 0.7000 | 0.0393 | 0.0004 | 26.9 | 14.1 | 28.2 | 27916.3 | 14.1 | 0 | 0.0393 | 0.0454 | 0.0333 |
| intern08_ts_replace | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0483 | 0.0004 | 36.3 | 16.0 | 32.0 | 36727.2 | 16.0 | 0 | 0.0483 | 0.0600 | 0.0654 |
| intern2b_ts_assist_k3 | 10 | 0.1000 | 0.0000 | 0.3000 | 0.0151 | 0.0002 | 13.3 | 8.7000 | 17.4 | 8593.9 | 8.7000 | 0 | 0.0151 | 0.0151 | 0.0000 |
| intern2b_ts_replace | 10 | 0.1000 | 0.0000 | 0.3000 | 0.0340 | 0.0003 | 30.4 | 8.7000 | 17.4 | 28280.3 | 8.7000 | 0 | 0.0340 | 0.0451 | 0.0633 |
| intern4b_uncal_ts_guard | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0423 | 0.0001 | 15.7 | 8.0000 | 7.6000 | 48346.7 | 7.6000 | 0 | 0.0423 | 0.0753 | 0.4953 |
| intern_shadow | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0393 | 0.0000 | 20.8 | 7.5000 | 0.0000 | 45310.1 | 7.5000 | 0 | 0.0393 | 0.0705 | 0.4981 |
| intern_shadow_s2lite | 10 | 0.7000 | 0.4000 | 1.0000 | 0.0096 | 0.0000 | 5.7126 | 7.6000 | 0.0000 | 45829.1 | 7.6000 | 0 | 0.0096 | 0.0143 | 0.3476 |
| intern_stop_guard | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0414 | 0.0000 | 14.3 | 7.7000 | 1.2000 | 46750.7 | 7.7000 | 0 | 0.0414 | 0.0728 | 0.4779 |
| intern_stop_guard_s2lite | 10 | 0.6000 | 0.3000 | 0.9000 | 0.0113 | 0.0000 | 7.4774 | 9.0000 | 2.4000 | 56088.4 | 9.0000 | 0 | 0.0113 | 0.0175 | 0.3703 |
| intern_ts_assist_k3 | 10 | 0.6000 | 0.3000 | 0.9000 | 0.0193 | 0.0003 | 20.7 | 8.5000 | 17.0 | 11157.0 | 8.9000 | 4 | 0.0193 | 0.0193 | 0.0000 |
| intern_ts_assist_k3_s2lite * | 10 | 0.4000 | 0.1000 | 0.7000 | 0.0030 | 0.0002 | 11.4 | 6.9000 | 13.8 | 7493.3 | 7.4000 | 5 | 0.0030 | 0.0030 | 0.0000 |
| intern_ts_assist_k5 | 10 | 0.5000 | 0.2000 | 0.8000 | 0.0176 | 0.0002 | 22.0 | 7.7000 | 15.4 | 10171.7 | 8.2000 | 5 | 0.0176 | 0.0176 | 0.0000 |
| intern_ts_cascade_tau0.5 | 10 | 0.1000 | 0.0000 | 0.3000 | 0.0047 | 0.0001 | 7.7262 | 3.0000 | 10.4 | 3126.7 | 3.1000 | 0 | 0.0047 | 0.0055 | 0.0577 |
| intern_ts_cascade_tau0.7 | 10 | 0.1000 | 0.0000 | 0.3000 | 0.0082 | 0.0001 | 8.6368 | 2.9000 | 9.9000 | 6650.8 | 3.1000 | 0 | 0.0082 | 0.0108 | 0.2703 |
| intern_ts_cascade_tau0.8 | 10 | 0.1000 | 0.0000 | 0.3000 | 0.0123 | 0.0001 | 9.0817 | 3.5000 | 10.7 | 11379.6 | 3.9000 | 2 | 0.0123 | 0.0180 | 0.3097 |
| intern_ts_cascade_tau0.9 | 10 | 0.2000 | 0.0000 | 0.5000 | 0.0215 | 0.0002 | 13.3 | 5.2000 | 14.1 | 22768.6 | 5.9000 | 5 | 0.0215 | 0.0356 | 0.4327 |
| intern_ts_cascade_tau0.95 | 10 | 0.4000 | 0.1000 | 0.7000 | 0.0256 | 0.0002 | 13.0 | 5.6000 | 13.5 | 28024.2 | 6.3000 | 5 | 0.0256 | 0.0436 | 0.4526 |
| intern_ts_cascade_tau0.99 | 10 | 0.5000 | 0.2000 | 0.8000 | 0.0296 | 0.0002 | 15.0 | 5.8000 | 12.0 | 33431.3 | 6.4000 | 5 | 0.0296 | 0.0518 | 0.4861 |
| intern_ts_cascade_tau0.9_s2lite | 10 | 0.2000 | 0.0000 | 0.5000 | 0.0056 | 0.0002 | 10.8 | 5.3000 | 14.0 | 23251.6 | 5.9000 | 5 | 0.0056 | 0.0075 | 0.2343 |
| intern_ts_cascade_tauhat | 10 | 0.2000 | 0.0000 | 0.5000 | 0.0216 | 0.0002 | 14.7 | 5.3000 | 14.1 | 22796.4 | 5.9000 | 5 | 0.0216 | 0.0357 | 0.4319 |
| intern_ts_guard | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0420 | 0.0001 | 17.8 | 8.0000 | 7.6000 | 48345.8 | 7.6000 | 0 | 0.0420 | 0.0752 | 0.5024 |
| intern_ts_guard_s2lite * | 10 | 0.7000 | 0.4000 | 1.0000 | 0.0093 | 0.0001 | 7.4725 | 7.2000 | 7.0000 | 42970.6 | 7.1000 | 1 | 0.0093 | 0.0135 | 0.3295 |
| intern_ts_params_replace | 10 | 0.1000 | 0.0000 | 0.3000 | 0.0210 | 0.0003 | 32.3 | 6.0000 | 26.0 | 11518.3 | 8.4000 | 0 | 0.0210 | 0.0210 | 0.0000 |
| intern_ts_replace | 10 | 0.1000 | 0.0000 | 0.3000 | 0.0121 | 0.0002 | 16.8 | 5.4000 | 10.8 | 6956.7 | 5.4000 | 0 | 0.0121 | 0.0121 | 0.0000 |
| jev_shadow | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0396 | 0.0000 | 13.0 | 7.5000 | 0.0000 | 45305.7 | 7.5000 | 0 | 0.0396 | 0.0704 | 0.4912 |
| jev_ts_assist_k3 | 10 | 0.8000 | 0.5000 | 1.0000 | 0.0219 | 0.0010 | 10.3 | 8.9000 | 8.9000 | 12346.1 | 8.9000 | 0 | 0.0219 | 0.0219 | 0.0000 |
| jev_ts_guard | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0413 | 0.0004 | 11.0 | 7.7000 | 7.5000 | 46615.1 | 7.5000 | 0 | 0.0413 | 0.0729 | 0.4899 |
| jev_ts_replace * | 10 | 0.9000 | 0.7000 | 1.0000 | 0.0142 | 0.0008 | 15.4 | 6.9000 | 6.9000 | 7785.4 | 6.9000 | 0 | 0.0142 | 0.0142 | 0.0000 |
| llm_only * | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0393 | 0.0000 | 11.4 | 7.5000 | 0.0000 | 45310.3 | 7.5000 | 0 | 0.0393 | 0.0704 | 0.4981 |
| llm_only_s2lite | 10 | 0.6000 | 0.3000 | 0.9000 | 0.0096 | 0.0000 | 5.4934 | 7.6000 | 0.0000 | 45827.9 | 7.6000 | 0 | 0.0096 | 0.0143 | 0.3476 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md)._

## Decision-point metrics

### intern08_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 141 | 119 | 0.0924 | 0.0840 | 0.0924 | 0.0000 | 0.2941 | 0.2941 | 0.3076 | 0.1703 | 0.7290 |

### intern08_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 160 | 133 | 0.0451 | 0.0301 | 0.0451 | 0.0000 | nan | 0.0451 | 0.5265 | 0.3251 | 0.6890 |

### intern2b_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 87 | 84 | 0.1429 | 0.0833 | 0.1429 | 0.0000 | 0.2976 | 0.2976 | 0.4155 | 0.2630 | 0.7361 |

### intern2b_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 87 | 77 | 0.1558 | 0.0649 | 0.1558 | 0.0000 | nan | 0.1558 | 0.3982 | 0.2531 | 0.7692 |

### intern4b_uncal_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 76 | 76 | 0.9605 | nan | 0.9605 | 0.0000 | nan | nan | 0.0712 | 0.0375 | 0.9600 |

### intern_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 75 | 75 | 0.9333 | nan | nan | 1.0000 | nan | nan | 0.2049 | 0.0720 | 0.9792 |
| tool_params/shadow | 106 | 94 | 0.8723 | nan | nan | 1.0000 | 0.9681 | 0.9681 | 0.1725 | 0.1071 | 0.9370 |
| tool_select/shadow | 75 | 75 | 0.7600 | 0.7200 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0653 | 0.1598 | 0.7427 |

### intern_shadow_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 76 | 76 | 0.9211 | nan | nan | 1.0000 | nan | nan | 0.2193 | 0.0793 | 0.9731 |
| tool_params/shadow | 107 | 100 | 0.9200 | nan | nan | 1.0000 | 1.0000 | 1.0000 | 0.2146 | 0.1185 | 0.8995 |
| tool_select/shadow | 76 | 66 | 0.7121 | 0.6667 | nan | 1.0000 | 0.9868 | 0.9868 | 0.0996 | 0.2047 | 0.6517 |

### intern_stop_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 12 | 12 | 0.8333 | nan | 0.8333 | 0.0000 | nan | nan | 0.2918 | 0.1265 | nan |
| tool_select/off | 77 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### intern_stop_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 24 | 24 | 0.5417 | nan | 0.5417 | 0.0000 | nan | nan | 0.3760 | 0.2975 | 0.5175 |
| tool_select/off | 90 | 0 | nan | nan | nan | 0.0000 | 0.9889 | 0.9889 | nan | nan | nan |

### intern_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 85 | 85 | 0.7176 | 0.6941 | 0.7176 | 0.0000 | 0.6824 | 0.6824 | 0.0806 | 0.2013 | 0.6305 |

### intern_ts_assist_k3_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 69 | 69 | 0.6812 | 0.6377 | 0.6812 | 0.0000 | 0.7101 | 0.7101 | 0.1066 | 0.1893 | 0.7311 |

### intern_ts_assist_k5

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 77 | 77 | 0.7013 | 0.6623 | 0.7013 | 0.0000 | 0.7922 | 0.7922 | 0.1102 | 0.1986 | 0.6626 |

### intern_ts_cascade_tau0.5

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 42 | 16 | 1.0000 | nan | 1.0000 | 0.6667 | nan | 1.0000 | 0.1684 | 0.0383 | nan |
| tool_params/generate | 4 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 31 | 31 | 0.5161 | 0.4516 | 0.5357 | 0.0968 | 1.0000 | 0.5806 | 0.2865 | 0.3271 | 0.4708 |

### intern_ts_cascade_tau0.7

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 37 | 13 | 0.9231 | nan | 1.0000 | 0.6486 | nan | 1.0000 | 0.2280 | 0.0830 | 0.9167 |
| tool_params/generate | 2 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 31 | 31 | 0.6129 | 0.5161 | 0.5714 | 0.3226 | 1.0000 | 0.7097 | 0.2345 | 0.2977 | 0.4518 |

### intern_ts_cascade_tau0.8

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 33 | 12 | 1.0000 | nan | 1.0000 | 0.6364 | nan | 1.0000 | 0.2613 | 0.1185 | nan |
| tool_params/generate | 2 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 37 | 37 | 0.5676 | 0.4865 | 0.5789 | 0.4865 | 1.0000 | 0.7838 | 0.2374 | 0.3065 | 0.5000 |

### intern_ts_cascade_tau0.9

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 33 | 22 | 0.9545 | nan | 1.0000 | 0.6667 | nan | 1.0000 | 0.2169 | 0.1084 | 0.9524 |
| tool_params/generate | 1 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 54 | 54 | 0.6481 | 0.5926 | 0.7647 | 0.6852 | 1.0000 | 0.9259 | 0.1396 | 0.2377 | 0.6316 |

### intern_ts_cascade_tau0.95

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 19 | 15 | 1.0000 | nan | 1.0000 | 0.5263 | nan | 1.0000 | 0.1821 | 0.0606 | nan |
| tool_params/generate | 2 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 58 | 58 | 0.7069 | 0.6552 | 0.9091 | 0.8103 | 1.0000 | 0.9828 | 0.1047 | 0.1996 | 0.6743 |

### intern_ts_cascade_tau0.99

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 2 | 2 | 1.0000 | nan | 1.0000 | 0.0000 | nan | 1.0000 | 0.0627 | 0.0042 | nan |
| tool_select/cascade | 59 | 59 | 0.6949 | 0.6441 | 1.0000 | 0.9831 | 1.0000 | 1.0000 | 0.1288 | 0.2081 | 0.6897 |

### intern_ts_cascade_tau0.9_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 32 | 23 | 0.9565 | nan | 1.0000 | 0.6562 | nan | 1.0000 | 0.2331 | 0.1180 | 0.9545 |
| tool_params/generate | 1 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 54 | 54 | 0.7037 | 0.6481 | 0.8125 | 0.7037 | 1.0000 | 0.9444 | 0.1713 | 0.2157 | 0.6151 |

### intern_ts_cascade_tauhat

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 33 | 22 | 0.9545 | nan | 1.0000 | 0.6970 | nan | 1.0000 | 0.2172 | 0.1070 | 0.9524 |
| tool_params/generate | 1 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 54 | 54 | 0.6111 | 0.5556 | 0.7647 | 0.6852 | 1.0000 | 0.9259 | 0.1724 | 0.2497 | 0.6652 |

### intern_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 76 | 76 | 0.9605 | nan | 0.9605 | 0.0000 | nan | nan | 0.1406 | 0.0439 | 0.9600 |

### intern_ts_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 70 | 70 | 0.9714 | nan | 0.9714 | 0.0000 | nan | nan | 0.1172 | 0.0371 | 0.9779 |

### intern_ts_params_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/generate | 60 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_params/replace | 92 | 69 | 0.2319 | nan | 0.2319 | 0.0000 | nan | 0.2319 | 0.6021 | 0.5578 | 0.4410 |
| tool_select/replace | 84 | 76 | 0.8026 | 0.7632 | 0.8026 | 0.0000 | nan | 0.8026 | 0.2474 | 0.1990 | 0.5290 |

### intern_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 54 | 54 | 0.4074 | 0.3704 | 0.4074 | 0.0000 | nan | 0.4074 | 0.2783 | 0.3021 | 0.6676 |

### jev_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 75 | 75 | 0.9867 | nan | nan | 1.0000 | nan | nan | 0.0469 | 0.0081 | 1.0000 |
| tool_params/shadow | 106 | 94 | 0.9681 | nan | nan | 1.0000 | 0.9681 | 0.9681 | 0.0177 | 0.0230 | 0.9597 |
| tool_select/shadow | 75 | 75 | 0.9333 | 0.8800 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0411 | 0.0624 | 0.7714 |

### jev_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 89 | 89 | 0.7978 | 0.7528 | 0.7978 | 0.0000 | 0.8202 | 0.8202 | 0.1056 | 0.1146 | 0.8764 |

### jev_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 75 | 75 | 0.9733 | nan | 0.9733 | 0.0000 | nan | nan | 0.1213 | 0.0264 | nan |

### jev_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 69 | 69 | 0.9710 | 0.8841 | 0.9710 | 0.0000 | nan | 0.9710 | 0.0703 | 0.0367 | 0.8993 |

### llm_only

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/off | 75 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### llm_only_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/off | 76 | 0 | nan | nan | nan | 0.0000 | 0.9868 | 0.9868 | nan | nan | nan |


## Score card (baseline: `llm_only`)

- **Iso-quality cost savings**: 63.9% [56.8, 75.3] with `jev_ts_replace` (Δsuccess -10.0 pp [-30.0, 0.0], p_worse=0.349; not significantly worse, one-sided paired bootstrap, α=0.05; 6 arm(s) qualify)
- **Quality at iso-cost** (≤ 0.0393 $/episode): `llm_only` 100.0% vs. reference 100.0% (Δ +0.0 pp, p_better=–); hull 100.0%
- **Iso-quality cost savings** (vs. `llm_only_s2lite`): 69.1% [64.1, 73.0] with `intern_ts_assist_k3_s2lite` (Δsuccess -20.0 pp [-50.0, 0.0], p_worse=0.103; not significantly worse, one-sided paired bootstrap, α=0.05; 3 arm(s) qualify)
- **Quality at iso-cost** (vs. `llm_only_s2lite`) (≤ 0.00957 $/episode): `intern_ts_guard_s2lite` 70.0% vs. reference 60.0% (Δ +10.0 pp, p_better=0.352); hull 70.0%
- **Frontier cost-of-pass**: 0.0111 $/success over 27 arms (3 shadow arm(s) left out of the frontier) (best single arm: `intern_ts_assist_k3_s2lite`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.697, baseline alone 0.144; front: `intern_ts_assist_k3_s2lite`, `intern_ts_guard_s2lite`, `jev_ts_replace`, `llm_only`
- **Sweep `intern_ts_cascade_tau*`** (6 arms): AIQ 0.617
- note: per-System-2 references: `llm_only_s2lite` (gemini-3.5-flash-lite) for 5 arm(s); every other arm vs. `llm_only`

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| llm_only | 100.0 [100.0, 100.0] | – | – | 0.0393 | – | 0.0393 | 11.4 | 0.0 | front,baseline |
| intern_shadow | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0393 | -0.0 | 0.0393 | 20.8 | 0.0 |  |
| jev_shadow | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0396 | -0.6 | 0.0396 | 13.0 | 0.0 |  |
| jev_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0413 | -5.0 | 0.0413 | 11.0 | 0.0 | iso-q |
| intern_stop_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0414 | -5.2 | 0.0414 | 14.3 | 0.0 | iso-q |
| intern_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0420 | -6.9 | 0.0420 | 17.8 | 0.0 | iso-q |
| intern4b_uncal_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0423 | -7.6 | 0.0423 | 15.7 | 0.0 | iso-q |
| jev_ts_replace | 90.0 [70.0, 100.0] | -10.0 | 0.349 | 0.0142 | 63.9 | 0.0158 | 15.4 | 100.0 | front,iso-best |
| jev_ts_assist_k3 | 80.0 [50.0, 100.0] | -20.0 | 0.107 | 0.0219 | 44.3 | 0.0274 | 10.3 | 0.0 | iso-q |
| intern_ts_guard_s2lite | 70.0 [40.0, 100.0] | +10.0 | 1.000 | 0.00929 | 3.0 | 0.0133 | 7.47 | 0.0 | front,iso-q,vs llm_only_s2lite |
| intern_shadow_s2lite | 70.0 [40.0, 100.0] | +10.0 | 1.000 | 0.00957 | 0.0 | 0.0137 | 5.71 | 0.0 | vs llm_only_s2lite |
| llm_only_s2lite | 60.0 [30.0, 90.0] | -40.0 | 0.008 | 0.00957 | 75.7 | 0.0160 | 5.49 | 0.0 |  |
| intern_stop_guard_s2lite | 60.0 [30.0, 90.0] | +0.0 | 1.000 | 0.0113 | -17.6 | 0.0188 | 7.48 | 0.0 | iso-q,vs llm_only_s2lite |
| intern_ts_assist_k3 | 60.0 [30.0, 90.0] | -40.0 | 0.006 | 0.0193 | 51.0 | 0.0321 | 20.7 | 0.0 | worse |
| intern_ts_assist_k5 | 50.0 [20.0, 80.0] | -50.0 | 0.002 | 0.0176 | 55.3 | 0.0351 | 22.0 | 0.0 | worse |
| intern_ts_cascade_tau0.99 | 50.0 [20.0, 80.0] | -50.0 | 0.002 | 0.0296 | 24.7 | 0.0593 | 15.0 | 4.9 | worse |
| intern_ts_assist_k3_s2lite | 40.0 [10.0, 70.0] | -20.0 | 0.103 | 0.00296 | 69.1 | 0.00740 | 11.4 | 0.0 | front,iso-best,vs llm_only_s2lite |
| intern_ts_cascade_tau0.95 | 40.0 [10.0, 70.0] | -60.0 | <0.001 | 0.0256 | 34.9 | 0.0641 | 13.0 | 25.3 | worse |
| intern08_ts_assist_k3 | 40.0 [10.0, 70.0] | -60.0 | <0.001 | 0.0393 | 0.2 | 0.0981 | 26.9 | 0.0 | worse |
| intern_ts_cascade_tau0.9_s2lite | 20.0 [0.0, 50.0] | -40.0 | 0.003 | 0.00565 | 41.0 | 0.0282 | 10.8 | 31.0 | worse,vs llm_only_s2lite |
| intern_ts_cascade_tau0.9 | 20.0 [0.0, 50.0] | -80.0 | <0.001 | 0.0215 | 45.3 | 0.108 | 13.3 | 31.8 | worse |
| intern_ts_cascade_tauhat | 20.0 [0.0, 50.0] | -80.0 | <0.001 | 0.0216 | 45.2 | 0.108 | 14.7 | 30.7 | worse |
| intern_ts_cascade_tau0.5 | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.00473 | 88.0 | 0.0473 | 7.73 | 54.5 | worse |
| intern_ts_cascade_tau0.7 | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.00820 | 79.2 | 0.0820 | 8.64 | 48.6 | worse |
| intern_ts_replace | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0121 | 69.2 | 0.121 | 16.8 | 100.0 | worse |
| intern_ts_cascade_tau0.8 | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0123 | 68.8 | 0.123 | 9.08 | 43.1 | worse |
| intern2b_ts_assist_k3 | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0151 | 61.7 | 0.151 | 13.3 | 0.0 | worse |
| intern_ts_params_replace | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0210 | 46.5 | 0.210 | 32.3 | 74.6 | worse |
| intern2b_ts_replace | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0340 | 13.6 | 0.340 | 30.4 | 100.0 | worse |
| intern08_ts_replace | 0.0 [0.0, 0.0] | -100.0 | <0.001 | 0.0483 | -22.7 | ∞ | 36.3 | 100.0 | worse |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

