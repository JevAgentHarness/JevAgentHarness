## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intern08_ts_assist_k3 | 10 | 0.3000 | 0.0000 | 0.6000 | 0.0355 | 0.0003 | 16.2 | 12.7 | 25.4 | 25512.7 | 12.7 | 0 | 0.0000 | 0.0355 | 0.0416 | 0.0333 |
| intern08_ts_replace | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0564 | 0.0005 | 26.3 | 18.7 | 37.4 | 41166.3 | 18.7 | 0 | 0.0053 | 0.0564 | 0.0686 | 0.0685 |
| intern2b_ts_assist_k3 | 10 | 0.1000 | 0.0000 | 0.3000 | 0.0165 | 0.0002 | 12.6 | 9.5000 | 19.0 | 9244.2 | 9.5000 | 0 | 0.0000 | 0.0165 | 0.0165 | 0.0000 |
| intern2b_ts_replace | 10 | 0.1000 | 0.0000 | 0.3000 | 0.0304 | 0.0003 | 14.0 | 8.4000 | 16.8 | 24217.2 | 8.4000 | 0 | 0.0238 | 0.0304 | 0.0385 | 0.0523 |
| intern4b_uncal_ts_guard | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0423 | 0.0001 | 10.6 | 8.0000 | 7.6000 | 48346.6 | 7.6000 | 0 | 0.0000 | 0.0423 | 0.0753 | 0.4981 |
| intern_shadow | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0396 | 0.0000 | 9.6410 | 7.5000 | 0.0000 | 45305.7 | 7.5000 | 0 | 0.0133 | 0.0396 | 0.0704 | 0.4912 |
| intern_shadow_s2lite | 10 | 0.7000 | 0.4000 | 0.9000 | 0.0097 | 0.0000 | 4.7437 | 7.6000 | 0.0000 | 45831.5 | 7.6000 | 0 | 0.0000 | 0.0097 | 0.0143 | 0.3414 |
| intern_stop_guard | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0420 | 0.0000 | 9.2016 | 7.8000 | 1.3000 | 47477.7 | 7.8000 | 0 | 0.0000 | 0.0420 | 0.0741 | 0.4885 |
| intern_stop_guard_s2lite | 10 | 0.7000 | 0.4000 | 1.0000 | 0.0130 | 0.0000 | 7.7013 | 10.1 | 3.2000 | 65429.0 | 10.1 | 0 | 0.0000 | 0.0130 | 0.0204 | 0.3695 |
| intern_ts_assist_k3 | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0271 | 0.0004 | 22.7 | 10.4 | 20.8 | 15974.3 | 10.4 | 0 | 0.0000 | 0.0271 | 0.0271 | 0.0000 |
| intern_ts_assist_k3_s2lite * | 10 | 0.7000 | 0.4000 | 1.0000 | 0.0040 | 0.0003 | 13.6 | 8.2000 | 16.4 | 10529.1 | 8.2000 | 0 | 0.0000 | 0.0040 | 0.0040 | 0.0000 |
| intern_ts_assist_k5 * | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0242 | 0.0003 | 19.4 | 9.4000 | 18.8 | 14246.3 | 9.4000 | 0 | 0.0000 | 0.0242 | 0.0242 | 0.0000 |
| intern_ts_cascade_tau0.5 | 10 | 0.1000 | 0.0000 | 0.3000 | 0.0047 | 0.0001 | 6.8134 | 3.1000 | 10.8 | 3235.0 | 3.2000 | 0 | 0.0000 | 0.0047 | 0.0057 | 0.0686 |
| intern_ts_cascade_tau0.7 | 10 | 0.2000 | 0.0000 | 0.5000 | 0.0097 | 0.0001 | 8.1089 | 3.5000 | 11.4 | 8084.8 | 3.6000 | 0 | 0.0286 | 0.0097 | 0.0131 | 0.2943 |
| intern_ts_cascade_tau0.8 | 10 | 0.2000 | 0.0000 | 0.5000 | 0.0143 | 0.0002 | 9.8513 | 4.1000 | 12.5 | 12937.1 | 4.2000 | 0 | 0.0000 | 0.0143 | 0.0205 | 0.3093 |
| intern_ts_cascade_tau0.9 | 10 | 0.7000 | 0.4000 | 1.0000 | 0.0290 | 0.0003 | 16.8 | 6.8000 | 18.9 | 28659.6 | 7.0000 | 0 | 0.0000 | 0.0290 | 0.0452 | 0.4035 |
| intern_ts_cascade_tau0.95 | 10 | 0.9000 | 0.7000 | 1.0000 | 0.0339 | 0.0003 | 16.8 | 7.1000 | 17.3 | 36688.4 | 7.3000 | 0 | 0.0000 | 0.0339 | 0.0577 | 0.4509 |
| intern_ts_cascade_tau0.99 | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0392 | 0.0003 | 15.9 | 7.4000 | 15.2 | 44723.1 | 7.5000 | 0 | 0.0000 | 0.0392 | 0.0698 | 0.4934 |
| intern_ts_guard | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0420 | 0.0001 | 9.8819 | 8.0000 | 7.6000 | 48346.7 | 7.6000 | 0 | 0.0000 | 0.0420 | 0.0752 | 0.5024 |
| intern_ts_guard_s2lite | 10 | 0.7000 | 0.4000 | 1.0000 | 0.0098 | 0.0001 | 7.3974 | 7.6000 | 7.4000 | 45737.0 | 7.4000 | 0 | 0.0000 | 0.0098 | 0.0144 | 0.3396 |
| intern_ts_params_replace | 10 | 0.1000 | 0.0000 | 0.3000 | 0.0134 | 0.0003 | 13.7 | 4.7000 | 22.3 | 7129.6 | 7.2000 | 0 | 0.0213 | 0.0134 | 0.0134 | 0.0000 |
| intern_ts_replace | 10 | 0.1000 | 0.0000 | 0.3000 | 0.0109 | 0.0002 | 9.8641 | 5.2000 | 10.4 | 6186.4 | 5.2000 | 0 | 0.0192 | 0.0109 | 0.0109 | 0.0000 |
| jev_shadow | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0396 | 0.0000 | 7.1510 | 7.5000 | 0.0000 | 45305.7 | 7.5000 | 0 | 0.0044 | 0.0396 | 0.0704 | 0.4912 |
| jev_ts_assist_k3 | 10 | 0.8000 | 0.5000 | 1.0000 | 0.0216 | 0.0010 | 9.4962 | 8.8000 | 8.8000 | 12168.0 | 8.8000 | 0 | 0.0000 | 0.0216 | 0.0216 | 0.0000 |
| jev_ts_guard | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0406 | 0.0004 | 8.0110 | 7.6000 | 7.5000 | 45885.8 | 7.5000 | 0 | 0.0000 | 0.0406 | 0.0717 | 0.4898 |
| jev_ts_replace * | 10 | 0.9000 | 0.7000 | 1.0000 | 0.0145 | 0.0008 | 6.8553 | 7.1000 | 7.1000 | 7901.3 | 7.1000 | 0 | 0.0410 | 0.0145 | 0.0145 | 0.0000 |
| llm_only | 10 | 1.0000 | 1.0000 | 1.0000 | 0.0422 | 0.0000 | 6.7588 | 7.5000 | 0.0000 | 45310.1 | 7.5000 | 0 | 0.0000 | 0.0422 | 0.0704 | 0.4451 |
| llm_only_s2lite * | 10 | 0.8000 | 0.5000 | 1.0000 | 0.0096 | 0.0000 | 4.9898 | 7.6000 | 0.0000 | 45831.9 | 7.6000 | 0 | 0.0000 | 0.0096 | 0.0143 | 0.3555 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### intern08_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 127 | 109 | 0.0917 | 0.0826 | 0.0917 | 0.0000 | 0.3119 | 0.3119 | 0.3081 | 0.1718 | 0.7152 |

### intern08_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 187 | 157 | 0.0382 | 0.0255 | 0.0382 | 0.0000 | nan | 0.0382 | 0.5235 | 0.3125 | 0.7130 |

### intern2b_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 95 | 92 | 0.1304 | 0.0761 | 0.1304 | 0.0000 | 0.2717 | 0.2717 | 0.4308 | 0.2716 | 0.7156 |

### intern2b_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 84 | 74 | 0.1622 | 0.0676 | 0.1622 | 0.0000 | nan | 0.1622 | 0.4051 | 0.2627 | 0.7661 |

### intern4b_uncal_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 76 | 76 | 0.9605 | nan | 0.9605 | 0.0000 | nan | nan | 0.0703 | 0.0391 | 0.9600 |

### intern_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 75 | 75 | 0.9333 | nan | nan | 1.0000 | nan | nan | 0.1963 | 0.0725 | 0.9792 |
| tool_params/shadow | 106 | 94 | 0.8723 | nan | nan | 1.0000 | 0.9681 | 0.9681 | 0.1724 | 0.1071 | 0.9339 |
| tool_select/shadow | 75 | 75 | 0.7600 | 0.7200 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0767 | 0.1654 | 0.7232 |

### intern_shadow_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 76 | 76 | 0.9737 | nan | nan | 1.0000 | nan | nan | 0.2298 | 0.0725 | 1.0000 |
| tool_params/shadow | 107 | 100 | 0.9100 | nan | nan | 1.0000 | 1.0000 | 1.0000 | 0.2104 | 0.1159 | 0.9145 |
| tool_select/shadow | 76 | 76 | 0.7368 | 0.6974 | nan | 1.0000 | 0.9868 | 0.9868 | 0.0880 | 0.1781 | 0.7179 |

### intern_stop_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 13 | 13 | 0.7692 | nan | 0.7692 | 0.0000 | nan | nan | 0.3119 | 0.1441 | nan |
| tool_select/off | 78 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### intern_stop_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 32 | 32 | 0.8750 | nan | 0.8750 | 0.0000 | nan | nan | 0.1916 | 0.1217 | 0.9245 |
| tool_select/off | 101 | 0 | nan | nan | nan | 0.0000 | 0.9604 | 0.9604 | nan | nan | nan |

### intern_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 104 | 104 | 0.7596 | 0.7404 | 0.7596 | 0.0000 | 0.7404 | 0.7404 | 0.0784 | 0.1781 | 0.6618 |

### intern_ts_assist_k3_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 82 | 82 | 0.7195 | 0.6951 | 0.7195 | 0.0000 | 0.7927 | 0.7927 | 0.0491 | 0.1525 | 0.8077 |

### intern_ts_assist_k5

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 94 | 94 | 0.7553 | 0.7234 | 0.7553 | 0.0000 | 0.8191 | 0.8191 | 0.0981 | 0.1658 | 0.7177 |

### intern_ts_cascade_tau0.5

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 44 | 16 | 1.0000 | nan | 1.0000 | 0.6818 | 1.0000 | 1.0000 | 0.1684 | 0.0383 | nan |
| tool_params/generate | 4 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 32 | 32 | 0.4688 | 0.4062 | 0.5172 | 0.0938 | 1.0000 | 0.5625 | 0.3042 | 0.3188 | 0.5490 |

### intern_ts_cascade_tau0.7

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 42 | 19 | 0.8947 | nan | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 0.2215 | 0.0720 | 1.0000 |
| tool_params/generate | 3 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 36 | 36 | 0.6111 | 0.5278 | 0.6400 | 0.3056 | 1.0000 | 0.7500 | 0.1846 | 0.2633 | 0.5812 |

### intern_ts_cascade_tau0.8

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 41 | 18 | 0.9444 | nan | 1.0000 | 0.6829 | 1.0000 | 1.0000 | 0.2835 | 0.1248 | 0.8235 |
| tool_params/generate | 2 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 42 | 42 | 0.5714 | 0.5000 | 0.6087 | 0.4524 | 1.0000 | 0.7857 | 0.2561 | 0.3064 | 0.5579 |

### intern_ts_cascade_tau0.9

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 49 | 38 | 0.9737 | nan | 1.0000 | 0.6735 | 1.0000 | 1.0000 | 0.2415 | 0.1086 | 0.9459 |
| tool_params/generate | 4 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 70 | 70 | 0.7143 | 0.6714 | 0.8571 | 0.6000 | 1.0000 | 0.9429 | 0.0996 | 0.1938 | 0.6970 |

### intern_ts_cascade_tau0.95

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 27 | 23 | 1.0000 | nan | 1.0000 | 0.5185 | 1.0000 | 1.0000 | 0.1906 | 0.0623 | nan |
| tool_params/generate | 2 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 73 | 73 | 0.7671 | 0.7260 | 0.9333 | 0.7945 | 1.0000 | 0.9863 | 0.0670 | 0.1606 | 0.7185 |

### intern_ts_cascade_tau0.99

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 2 | 2 | 1.0000 | nan | 1.0000 | 0.0000 | nan | 1.0000 | 0.0655 | 0.0049 | nan |
| tool_select/cascade | 75 | 75 | 0.7467 | 0.7067 | 1.0000 | 0.9867 | 1.0000 | 1.0000 | 0.0738 | 0.1657 | 0.7444 |

### intern_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 76 | 76 | 0.9605 | nan | 0.9605 | 0.0000 | nan | nan | 0.1419 | 0.0451 | 0.9600 |

### intern_ts_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 74 | 74 | 0.9595 | nan | 0.9595 | 0.0000 | nan | nan | 0.1279 | 0.0352 | 0.9859 |

### intern_ts_params_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/generate | 47 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_params/replace | 79 | 54 | 0.2963 | nan | 0.2963 | 0.0000 | nan | 0.2963 | 0.5319 | 0.5080 | 0.4243 |
| tool_select/replace | 72 | 68 | 0.7500 | 0.7206 | 0.7500 | 0.0000 | nan | 0.7500 | 0.2084 | 0.1920 | 0.6113 |

### intern_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 52 | 52 | 0.4231 | 0.3846 | 0.4231 | 0.0000 | nan | 0.4231 | 0.2644 | 0.2829 | 0.7121 |

### jev_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 75 | 75 | 0.9867 | nan | nan | 1.0000 | nan | nan | 0.0449 | 0.0074 | 1.0000 |
| tool_params/shadow | 106 | 94 | 0.9681 | nan | nan | 1.0000 | 0.9681 | 0.9681 | 0.0213 | 0.0231 | 0.9597 |
| tool_select/shadow | 75 | 75 | 0.9333 | 0.8800 | nan | 1.0000 | 1.0000 | 1.0000 | 0.0369 | 0.0629 | 0.7957 |

### jev_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 88 | 88 | 0.8295 | 0.7841 | 0.8295 | 0.0000 | 0.8295 | 0.8295 | 0.0727 | 0.1185 | 0.8178 |

### jev_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 75 | 75 | 0.9867 | nan | 0.9867 | 0.0000 | nan | nan | 0.1197 | 0.0260 | nan |

### jev_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 71 | 71 | 0.9437 | 0.8592 | 0.9437 | 0.0000 | nan | 0.9437 | 0.0723 | 0.0400 | 0.9478 |

### llm_only

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/off | 75 | 0 | nan | nan | nan | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |

### llm_only_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/off | 76 | 0 | nan | nan | nan | 0.0000 | 0.9868 | 0.9868 | nan | nan | nan |


## Score card (baseline: `llm_only`)

- **Iso-quality cost savings**: 65.7% [58.9, 76.5] with `jev_ts_replace` (Δsuccess -10.0 pp [-30.0, 0.0], p_worse=0.349; not significantly worse, one-sided paired bootstrap, α=0.05; 10 arm(s) qualify)
- **Quality at iso-cost** (≤ 0.0422 $/episode): `intern_ts_assist_k5` 100.0% vs. reference 100.0% (Δ +0.0 pp, p_better=1.000); hull 100.0%
- **Iso-quality cost savings** (vs. `llm_only_s2lite`): 58.3% [52.1, 65.2] with `intern_ts_assist_k3_s2lite` (Δsuccess -10.0 pp [-30.0, 0.0], p_worse=0.352; not significantly worse, one-sided paired bootstrap, α=0.05; 3 arm(s) qualify)
- **Quality at iso-cost** (vs. `llm_only_s2lite`) (≤ 0.00963 $/episode): `llm_only_s2lite` 80.0% vs. reference 80.0% (Δ +0.0 pp, p_better=–); hull 80.0%
- **Frontier cost-of-pass**: 0.00855 $/success over 25 arms (3 shadow arm(s) left out of the frontier) (best single arm: `intern_ts_assist_k3_s2lite`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.862, baseline alone 0.181; front: `intern_ts_assist_k3_s2lite`, `intern_ts_assist_k5`, `jev_ts_replace`, `llm_only_s2lite`
- **Sweep `intern_ts_cascade_tau*`** (6 arms): AIQ 0.704
- note: per-System-2 references: `llm_only_s2lite` (gemini-3.5-flash-lite) for 4 arm(s); every other arm vs. `llm_only`

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| llm_only | 100.0 [100.0, 100.0] | – | – | 0.0422 | – | 0.0422 | 6.76 | 0.0 | baseline |
| intern_ts_assist_k5 | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0242 | 42.6 | 0.0242 | 19.4 | 0.0 | front,iso-q |
| intern_ts_assist_k3 | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0271 | 35.7 | 0.0271 | 22.7 | 0.0 | iso-q |
| intern_ts_cascade_tau0.99 | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0392 | 7.1 | 0.0392 | 15.9 | 3.9 | iso-q |
| intern_shadow | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0396 | 6.2 | 0.0396 | 9.64 | 0.0 |  |
| jev_shadow | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0396 | 6.2 | 0.0396 | 7.15 | 0.0 |  |
| jev_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0406 | 3.8 | 0.0406 | 8.01 | 0.0 | iso-q |
| intern_stop_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0420 | 0.5 | 0.0420 | 9.20 | 0.0 | iso-q |
| intern_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0420 | 0.4 | 0.0420 | 9.88 | 0.0 | iso-q |
| intern4b_uncal_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0423 | -0.3 | 0.0423 | 10.6 | 0.0 | iso-q |
| jev_ts_replace | 90.0 [70.0, 100.0] | -10.0 | 0.349 | 0.0145 | 65.7 | 0.0161 | 6.86 | 100.0 | front,iso-best |
| intern_ts_cascade_tau0.95 | 90.0 [70.0, 100.0] | -10.0 | 0.349 | 0.0339 | 19.7 | 0.0376 | 16.8 | 27.5 | iso-q |
| llm_only_s2lite | 80.0 [50.0, 100.0] | -20.0 | 0.101 | 0.00963 | 77.2 | 0.0120 | 4.99 | 0.0 | front |
| jev_ts_assist_k3 | 80.0 [50.0, 100.0] | -20.0 | 0.107 | 0.0216 | 48.8 | 0.0270 | 9.50 | 0.0 | iso-q |
| intern_ts_assist_k3_s2lite | 70.0 [40.0, 100.0] | -10.0 | 0.352 | 0.00401 | 58.3 | 0.00574 | 13.6 | 0.0 | front,iso-best,vs llm_only_s2lite |
| intern_shadow_s2lite | 70.0 [40.0, 90.0] | -10.0 | 0.344 | 0.00973 | -1.1 | 0.0139 | 4.74 | 0.0 | vs llm_only_s2lite |
| intern_ts_guard_s2lite | 70.0 [40.0, 100.0] | -10.0 | 0.344 | 0.00975 | -1.3 | 0.0139 | 7.40 | 0.0 | iso-q,vs llm_only_s2lite |
| intern_stop_guard_s2lite | 70.0 [40.0, 100.0] | -10.0 | 0.344 | 0.0130 | -34.9 | 0.0186 | 7.70 | 0.0 | iso-q,vs llm_only_s2lite |
| intern_ts_cascade_tau0.9 | 70.0 [40.0, 100.0] | -30.0 | 0.032 | 0.0290 | 31.4 | 0.0414 | 16.8 | 35.8 |  |
| intern08_ts_assist_k3 | 30.0 [0.0, 60.0] | -70.0 | <0.001 | 0.0355 | 15.8 | 0.118 | 16.2 | 0.0 | worse |
| intern_ts_cascade_tau0.7 | 20.0 [0.0, 50.0] | -80.0 | <0.001 | 0.00971 | 77.0 | 0.0486 | 8.11 | 48.1 | worse |
| intern_ts_cascade_tau0.8 | 20.0 [0.0, 50.0] | -80.0 | <0.001 | 0.0143 | 66.2 | 0.0714 | 9.85 | 42.4 | worse |
| intern_ts_cascade_tau0.5 | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.00466 | 88.9 | 0.0466 | 6.81 | 53.8 | worse |
| intern_ts_replace | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0109 | 74.2 | 0.109 | 9.86 | 100.0 | worse |
| intern_ts_params_replace | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0134 | 68.3 | 0.134 | 13.7 | 76.3 | worse |
| intern2b_ts_assist_k3 | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0165 | 61.0 | 0.165 | 12.6 | 0.0 | worse |
| intern2b_ts_replace | 10.0 [0.0, 30.0] | -90.0 | <0.001 | 0.0304 | 28.0 | 0.304 | 14.0 | 100.0 | worse |
| intern08_ts_replace | 0.0 [0.0, 0.0] | -100.0 | <0.001 | 0.0564 | -33.6 | ∞ | 26.3 | 100.0 | worse |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

