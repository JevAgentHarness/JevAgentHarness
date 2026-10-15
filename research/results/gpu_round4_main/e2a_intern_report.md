## Episode metrics  (* = on cost/success Pareto front)

| config | n | success | success_ci_lo | success_ci_hi | cost_usd | s1_cost_usd | latency_s | llm_calls | s1_calls | llm_input_tokens | steps | errors | retry_rate | cost_usd_cache_modelled | cost_usd_no_cache | cached_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intern08_ts_assist_k3 | 200 | 0.1700 | 0.1200 | 0.2250 | 0.0311 | 0.0003 | 11.0 | 11.2 | 22.4 | 21704.8 | 11.2 | 0 | 0.0000 | 0.0311 | 0.0353 | 0.0245 |
| intern08_ts_replace | 200 | 0.0350 | 0.0150 | 0.0600 | 0.0402 | 0.0004 | 14.5 | 14.4 | 28.7 | 30479.0 | 14.4 | 0 | 0.0000 | 0.0402 | 0.0498 | 0.0577 |
| intern2b_ts_assist_k3 | 200 | 0.2550 | 0.1950 | 0.3150 | 0.0278 | 0.0003 | 12.2 | 11.0 | 22.0 | 18753.8 | 11.0 | 0 | 0.0005 | 0.0278 | 0.0310 | 0.0195 |
| intern2b_ts_replace | 200 | 0.0300 | 0.0100 | 0.0550 | 0.0246 | 0.0002 | 11.1 | 9.1850 | 18.4 | 18310.7 | 9.1850 | 0 | 0.0000 | 0.0246 | 0.0302 | 0.0326 |
| intern4b_uncal_ts_guard | 200 | 1.0000 | 1.0000 | 1.0000 | 0.0446 | 0.0001 | 29.7 | 7.9600 | 7.6300 | 48196.6 | 7.6300 | 0 | 0.0075 | 0.0446 | 0.0750 | 0.4542 |
| intern_shadow | 500 | 0.9980 | 0.9940 | 1.0000 | 0.0425 | 0.0000 | 10.2 | 7.5100 | 0.0000 | 45547.2 | 7.5100 | 0 | 0.0088 | 0.0425 | 0.0708 | 0.4399 |
| intern_shadow_s2lite | 500 | 0.8160 | 0.7820 | 0.8500 | 0.0098 | 0.0000 | 4.7750 | 7.5680 | 0.0000 | 45599.1 | 7.5680 | 0 | 0.0011 | 0.0098 | 0.0143 | 0.3194 |
| intern_stop_guard | 500 | 0.9920 | 0.9840 | 0.9980 | 0.0449 | 0.0000 | 10.4 | 7.9140 | 1.3900 | 48532.8 | 7.9140 | 0 | 0.0018 | 0.0449 | 0.0756 | 0.4456 |
| intern_stop_guard_s2lite | 500 | 0.7740 | 0.7360 | 0.8100 | 0.0144 | 0.0000 | 7.7464 | 11.1 | 4.3980 | 72250.2 | 11.1 | 0 | 0.0000 | 0.0144 | 0.0225 | 0.3359 |
| intern_ts_assist_k3 | 500 | 0.9600 | 0.9400 | 0.9760 | 0.0205 | 0.0003 | 17.3 | 8.5580 | 17.1 | 12033.4 | 8.5580 | 0 | 0.0030 | 0.0205 | 0.0206 | 0.0002 |
| intern_ts_assist_k3_s2lite * | 500 | 0.6860 | 0.6420 | 0.7260 | 0.0046 | 0.0003 | 12.8 | 8.5960 | 17.2 | 12337.3 | 8.5960 | 0 | 0.0000 | 0.0046 | 0.0046 | 0.0002 |
| intern_ts_assist_k5 * | 200 | 0.9750 | 0.9500 | 0.9950 | 0.0205 | 0.0003 | 19.8 | 8.0600 | 16.1 | 11993.9 | 8.0600 | 0 | 0.0012 | 0.0205 | 0.0205 | 0.0000 |
| intern_ts_cascade_tau0.5 | 200 | 0.2850 | 0.2250 | 0.3450 | 0.0084 | 0.0002 | 10.5 | 4.3050 | 15.8 | 5148.9 | 4.5300 | 0 | 0.0000 | 0.0084 | 0.0090 | 0.0281 |
| intern_ts_cascade_tau0.7 | 200 | 0.3600 | 0.2950 | 0.4250 | 0.0161 | 0.0002 | 11.2 | 4.7950 | 15.7 | 11644.7 | 4.9700 | 0 | 0.0000 | 0.0161 | 0.0189 | 0.0805 |
| intern_ts_cascade_tau0.8 | 200 | 0.4950 | 0.4250 | 0.5650 | 0.0216 | 0.0003 | 13.3 | 5.4650 | 16.3 | 18182.8 | 5.6250 | 0 | 0.0101 | 0.0216 | 0.0289 | 0.2184 |
| intern_ts_cascade_tau0.9 | 200 | 0.6800 | 0.6150 | 0.7450 | 0.0281 | 0.0003 | 15.5 | 6.3550 | 16.7 | 27292.8 | 6.4600 | 0 | 0.0110 | 0.0281 | 0.0430 | 0.3531 |
| intern_ts_cascade_tau0.95 | 200 | 0.8600 | 0.8100 | 0.9100 | 0.0352 | 0.0003 | 16.6 | 6.9550 | 16.0 | 36943.6 | 7.0600 | 0 | 0.0029 | 0.0352 | 0.0579 | 0.4130 |
| intern_ts_cascade_tau0.99 | 200 | 1.0000 | 1.0000 | 1.0000 | 0.0412 | 0.0003 | 16.4 | 7.4950 | 15.0 | 45437.1 | 7.5000 | 0 | 0.0000 | 0.0412 | 0.0709 | 0.4668 |
| intern_ts_cascade_tau0.9_s2lite * | 500 | 0.8280 | 0.7940 | 0.8620 | 0.0098 | 0.0000 | 4.3898 | 7.5720 | 0.0000 | 45627.8 | 7.5720 | 0 | 0.0000 | 0.0098 | 0.0143 | 0.3221 |
| intern_ts_cascade_tauhat | 500 | 0.9980 | 0.9940 | 1.0000 | 0.0437 | 0.0000 | 9.6106 | 7.5180 | 0.0000 | 45619.0 | 7.5180 | 0 | 0.0024 | 0.0437 | 0.0709 | 0.4184 |
| intern_ts_guard | 200 | 1.0000 | 1.0000 | 1.0000 | 0.0432 | 0.0001 | 9.7547 | 8.0150 | 7.6600 | 48523.8 | 7.6600 | 0 | 0.0000 | 0.0432 | 0.0755 | 0.4800 |
| intern_ts_guard_s2lite * | 200 | 0.8450 | 0.7950 | 0.8900 | 0.0103 | 0.0001 | 7.1590 | 7.8900 | 7.5500 | 47663.4 | 7.5500 | 0 | 0.0000 | 0.0103 | 0.0150 | 0.3187 |
| intern_ts_params_replace * | 200 | 0.1400 | 0.0900 | 0.1900 | 0.0021 | 0.0002 | 6.4398 | 1.0950 | 15.0 | 986.6 | 4.3900 | 0 | 0.0502 | 0.0021 | 0.0021 | 0.0000 |
| intern_ts_replace | 200 | 0.2650 | 0.2050 | 0.3250 | 0.0079 | 0.0001 | 8.2355 | 4.5550 | 9.1100 | 4468.4 | 4.5550 | 0 | 0.0417 | 0.0079 | 0.0080 | 0.0004 |
| jev_shadow * | 200 | 1.0000 | 1.0000 | 1.0000 | 0.0411 | 0.0000 | 10.3 | 7.5050 | 0.0000 | 45509.0 | 7.5050 | 0 | 0.0006 | 0.0411 | 0.0707 | 0.4645 |
| jev_ts_assist_k3 * | 200 | 0.9150 | 0.8750 | 0.9500 | 0.0195 | 0.0009 | 9.1041 | 8.0550 | 8.0550 | 10940.3 | 8.0550 | 0 | 0.0006 | 0.0195 | 0.0195 | 0.0000 |
| jev_ts_guard | 200 | 1.0000 | 1.0000 | 1.0000 | 0.0421 | 0.0004 | 11.6 | 7.6600 | 7.5000 | 46553.9 | 7.5000 | 0 | 0.0013 | 0.0421 | 0.0728 | 0.4718 |
| jev_ts_replace | 200 | 0.7900 | 0.7300 | 0.8450 | 0.0140 | 0.0008 | 5.9685 | 6.8050 | 6.8050 | 7706.0 | 6.8050 | 0 | 0.0004 | 0.0140 | 0.0140 | 0.0000 |
| llm_only | 500 | 0.9960 | 0.9900 | 1.0000 | 0.0436 | 0.0000 | 8.8335 | 7.5220 | 0.0000 | 45617.7 | 7.5220 | 0 | 0.0021 | 0.0436 | 0.0709 | 0.4203 |
| llm_only_s2lite * | 500 | 0.8340 | 0.8000 | 0.8660 | 0.0099 | 0.0000 | 5.6386 | 7.5740 | 0.0000 | 45645.0 | 7.5740 | 0 | 0.0026 | 0.0099 | 0.0143 | 0.3150 |

_cost_usd: the ledger's cost (what the provider bills: cached input tokens at the cached rate when it reported them); cost_usd_cache_modelled: the System-2 calls re-priced from their logged tokens with the current price table (measured cached tokens at `cached_input_per_m`); cost_usd_no_cache: list price, no cache; cached_frac: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md); retry_rate: share of logged calls that needed more than one HTTP attempt (over calls with a known attempt count; backoff is excluded from latency), errors: episodes that ended in an infrastructure error._

## Decision-point metrics

### intern08_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 2238 | 1955 | 0.1095 | 0.0793 | 0.1095 | 0.0000 | 0.2425 | 0.2425 | 0.3263 | 0.2018 | 0.6635 |

### intern08_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 2873 | 2507 | 0.0435 | 0.0255 | 0.0435 | 0.0000 | nan | 0.0435 | 0.4958 | 0.2952 | 0.6145 |

### intern2b_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 2198 | 2033 | 0.1815 | 0.1481 | 0.1815 | 0.0000 | 0.3079 | 0.3079 | 0.3948 | 0.2917 | 0.7063 |

### intern2b_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 1837 | 1615 | 0.1152 | 0.0768 | 0.1152 | 0.0000 | nan | 0.1152 | 0.4636 | 0.3141 | 0.6635 |

### intern4b_uncal_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 1526 | 1526 | 0.9607 | nan | 0.9607 | 0.0000 | nan | nan | 0.0600 | 0.0319 | 0.9697 |

### intern_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 3755 | 3755 | 0.9473 | nan | nan | 1.0000 | nan | nan | 0.2024 | 0.0708 | 0.9901 |
| tool_params/shadow | 5317 | 4202 | 0.8265 | nan | nan | 1.0000 | 0.9072 | 0.9072 | 0.1349 | 0.1076 | 0.9313 |
| tool_select/shadow | 3755 | 3755 | 0.7675 | 0.7055 | nan | 1.0000 | 0.9984 | 0.9984 | 0.0526 | 0.1618 | 0.7112 |

### intern_shadow_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 3784 | 3784 | 0.9289 | nan | nan | 1.0000 | nan | nan | 0.2079 | 0.0795 | 0.9720 |
| tool_params/shadow | 5352 | 4932 | 0.9053 | nan | nan | 1.0000 | 0.9996 | 0.9996 | 0.2075 | 0.1181 | 0.9292 |
| tool_select/shadow | 3784 | 3784 | 0.7799 | 0.7183 | nan | 1.0000 | 0.9910 | 0.9910 | 0.0533 | 0.1551 | 0.7170 |

### intern_stop_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 695 | 695 | 0.7137 | nan | 0.7137 | 0.0000 | nan | nan | 0.3430 | 0.1829 | nan |
| tool_select/off | 3957 | 0 | nan | nan | nan | 0.0000 | 0.9970 | 0.9970 | nan | nan | nan |

### intern_stop_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/guard | 2199 | 2199 | 0.5771 | nan | 0.5771 | 0.0000 | nan | nan | 0.2592 | 0.2975 | 0.6405 |
| tool_select/off | 5537 | 0 | nan | nan | nan | 0.0000 | 0.9841 | 0.9841 | nan | nan | nan |

### intern_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 4279 | 4244 | 0.7434 | 0.6708 | 0.7434 | 0.0000 | 0.8648 | 0.8648 | 0.0312 | 0.1627 | 0.7400 |

### intern_ts_assist_k3_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 4298 | 4248 | 0.7199 | 0.6448 | 0.7199 | 0.0000 | 0.8072 | 0.8072 | 0.0661 | 0.1738 | 0.7413 |

### intern_ts_assist_k5

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 1612 | 1612 | 0.8002 | 0.7270 | 0.8002 | 0.0000 | 0.9200 | 0.9200 | 0.0508 | 0.1469 | 0.7082 |

### intern_ts_cascade_tau0.5

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 1338 | 723 | 0.9253 | nan | 1.0000 | 0.7369 | 0.9928 | 0.9945 | 0.2121 | 0.1102 | 0.9321 |
| tool_params/generate | 106 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 906 | 906 | 0.7141 | 0.5861 | 0.7241 | 0.0397 | 0.8889 | 0.7307 | 0.0700 | 0.1920 | 0.6780 |

### intern_ts_cascade_tau0.7

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 1145 | 705 | 0.9532 | nan | 1.0000 | 0.6821 | 0.9798 | 0.9858 | 0.2266 | 0.1064 | 0.9520 |
| tool_params/generate | 93 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 994 | 994 | 0.7143 | 0.6157 | 0.7874 | 0.2807 | 0.9928 | 0.8451 | 0.0827 | 0.1949 | 0.6719 |

### intern_ts_cascade_tau0.8

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 1012 | 697 | 0.9684 | nan | 1.0000 | 0.6512 | 0.9846 | 0.9900 | 0.2366 | 0.1100 | 0.9584 |
| tool_params/generate | 104 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 1125 | 1125 | 0.7262 | 0.6400 | 0.8264 | 0.4471 | 0.9980 | 0.9031 | 0.0760 | 0.1881 | 0.6813 |

### intern_ts_cascade_tau0.9

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 764 | 559 | 0.9893 | nan | 1.0000 | 0.6113 | 0.9880 | 0.9928 | 0.2383 | 0.1035 | 0.9301 |
| tool_params/generate | 99 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 1292 | 1292 | 0.7399 | 0.6672 | 0.8629 | 0.6331 | 1.0000 | 0.9497 | 0.0706 | 0.1803 | 0.6868 |

### intern_ts_cascade_tau0.95

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 382 | 292 | 1.0000 | nan | 1.0000 | 0.5628 | 1.0000 | 1.0000 | 0.2106 | 0.0828 | nan |
| tool_params/generate | 51 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 1412 | 1412 | 0.7649 | 0.6983 | 0.8798 | 0.8350 | 0.9992 | 0.9795 | 0.0533 | 0.1665 | 0.6888 |

### intern_ts_cascade_tau0.99

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/cascade | 3 | 3 | 1.0000 | nan | 1.0000 | 0.3333 | 1.0000 | 1.0000 | 0.0997 | 0.0124 | nan |
| tool_params/generate | 1 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_select/cascade | 1500 | 1500 | 0.7727 | 0.7107 | 1.0000 | 0.9987 | 1.0000 | 1.0000 | 0.0532 | 0.1595 | 0.7056 |

### intern_ts_cascade_tau0.9_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/cascade | 3786 | 0 | nan | nan | nan | 1.0000 | 0.9905 | 0.9905 | nan | nan | nan |

### intern_ts_cascade_tauhat

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/cascade | 3759 | 0 | nan | nan | nan | 1.0000 | 0.9973 | 0.9973 | nan | nan | nan |

### intern_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 1532 | 1532 | 0.9550 | nan | 0.9550 | 0.0000 | nan | nan | 0.1378 | 0.0406 | 0.9572 |

### intern_ts_guard_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 1510 | 1510 | 0.9570 | nan | 0.9570 | 0.0000 | nan | nan | 0.1013 | 0.0381 | 0.9573 |

### intern_ts_params_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_params/generate | 203 | 0 | nan | nan | nan | 1.0000 | nan | nan | nan | nan | nan |
| tool_params/replace | 1236 | 567 | 0.7549 | nan | 0.7549 | 0.0138 | nan | 0.7549 | 0.1432 | 0.1774 | 0.6331 |
| tool_select/replace | 878 | 874 | 0.6510 | 0.5057 | 0.6510 | 0.0000 | nan | 0.6510 | 0.0874 | 0.2289 | 0.6196 |

### intern_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 911 | 911 | 0.6915 | 0.5675 | 0.6915 | 0.0000 | nan | 0.6915 | 0.0827 | 0.1931 | 0.7106 |

### jev_shadow

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop/shadow | 1501 | 1501 | 0.9927 | nan | nan | 1.0000 | nan | nan | 0.0479 | 0.0087 | 1.0000 |
| tool_params/shadow | 2126 | 1681 | 0.9280 | nan | nan | 1.0000 | 0.9090 | 0.9090 | 0.0415 | 0.0606 | 0.8855 |
| tool_select/shadow | 1501 | 1501 | 0.9400 | 0.8694 | nan | 1.0000 | 0.9993 | 0.9993 | 0.0683 | 0.0665 | 0.6934 |

### jev_ts_assist_k3

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/assist | 1611 | 1611 | 0.8970 | 0.8374 | 0.8970 | 0.0000 | 0.9150 | 0.9150 | 0.0482 | 0.0822 | 0.7517 |

### jev_ts_guard

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/guard | 1500 | 1500 | 0.9787 | nan | 0.9787 | 0.0000 | nan | nan | 0.1312 | 0.0299 | nan |

### jev_ts_replace

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/replace | 1361 | 1361 | 0.9493 | 0.8707 | 0.9493 | 0.0000 | nan | 0.9493 | 0.0271 | 0.0437 | 0.8614 |

### llm_only

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/off | 3761 | 0 | nan | nan | nan | 0.0000 | 0.9965 | 0.9965 | nan | nan | nan |

### llm_only_s2lite

| dp/mode | n | n_gold | s1_acc | s1_acc_canonical | accepted_acc | defer_rate | s2_acc | final_acc | ece | brier | auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/off | 3787 | 0 | nan | nan | nan | 0.0000 | 0.9902 | 0.9902 | nan | nan | nan |


## Score card (baseline: `llm_only`)

- **Iso-quality cost savings**: 5.3% [4.3, 6.3] with `intern_ts_cascade_tau0.99` (Δsuccess +0.0 pp [0.0, 0.0], p_worse=1.000; not significantly worse, one-sided paired bootstrap, α=0.05; 6 arm(s) qualify)
- **Quality at iso-cost** (≤ 0.0436 $/episode): `intern_ts_cascade_tau0.99` 100.0% vs. reference 99.6% (Δ +0.0 pp, p_better=1.000); hull 100.0%
- **Iso-quality cost savings** (vs. `llm_only_s2lite`): 1.0% [0.5, 1.4] with `intern_ts_cascade_tau0.9_s2lite` (Δsuccess -0.6 pp [-2.8, 1.6], p_worse=0.347; not significantly worse, one-sided paired bootstrap, α=0.05; 2 arm(s) qualify)
- **Quality at iso-cost** (vs. `llm_only_s2lite`) (≤ 0.00988 $/episode): `llm_only_s2lite` 83.4% vs. reference 83.4% (Δ +0.0 pp, p_better=–); hull 83.4%
- **Frontier cost-of-pass**: 0.0073 $/success over 27 arms (3 shadow arm(s) left out of the frontier) (best single arm: `intern_ts_assist_k3_s2lite`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.687, baseline alone 0.0789; front: `intern_ts_assist_k3_s2lite`, `intern_ts_assist_k5`, `intern_ts_cascade_tau0.99`, `intern_ts_cascade_tau0.9_s2lite`, `intern_ts_guard_s2lite`, `intern_ts_params_replace`, `jev_ts_assist_k3`, `llm_only_s2lite`
- **Sweep `intern_ts_cascade_tau*`** (6 arms): AIQ 0.58
- note: per-System-2 references: `llm_only_s2lite` (gemini-3.5-flash-lite) for 5 arm(s); every other arm vs. `llm_only`

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | flags |
|---|---|---|---|---|---|---|---|---|---|
| llm_only | 99.6 [99.0, 100.0] | – | – | 0.0436 | – | 0.0438 | 8.83 | 0.0 | baseline |
| jev_shadow | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0411 | 5.5 | 0.0411 | 10.3 | 0.0 |  |
| intern_ts_cascade_tau0.99 | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0412 | 5.3 | 0.0412 | 16.4 | 0.3 | front,iso-best |
| jev_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0421 | 3.1 | 0.0421 | 11.6 | 0.0 | iso-q |
| intern_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0432 | 0.7 | 0.0432 | 9.75 | 0.0 | iso-q |
| intern4b_uncal_ts_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0446 | -2.4 | 0.0446 | 29.7 | 0.0 | iso-q |
| intern_shadow | 99.8 [99.4, 100.0] | +0.2 | 1.000 | 0.0425 | 2.6 | 0.0426 | 10.2 | 0.0 |  |
| intern_ts_cascade_tauhat | 99.8 [99.4, 100.0] | +0.2 | 1.000 | 0.0437 | -0.2 | 0.0438 | 9.61 | 0.0 | iso-q |
| intern_stop_guard | 99.2 [98.4, 99.8] | -0.4 | 0.137 | 0.0449 | -3.0 | 0.0453 | 10.4 | 0.0 | iso-q |
| intern_ts_assist_k5 | 97.5 [95.0, 99.5] | -2.5 | 0.007 | 0.0205 | 53.0 | 0.0210 | 19.8 | 0.0 | front |
| intern_ts_assist_k3 | 96.0 [94.0, 97.6] | -3.6 | <0.001 | 0.0205 | 53.0 | 0.0214 | 17.3 | 0.0 | worse |
| jev_ts_assist_k3 | 91.5 [87.5, 95.0] | -8.5 | <0.001 | 0.0195 | 55.2 | 0.0213 | 9.10 | 0.0 | front,worse |
| intern_ts_cascade_tau0.95 | 86.0 [81.0, 91.0] | -14.0 | <0.001 | 0.0352 | 19.2 | 0.0409 | 16.6 | 21.7 | worse |
| intern_ts_guard_s2lite | 84.5 [79.5, 89.0] | -0.5 | 0.461 | 0.0103 | -4.4 | 0.0122 | 7.16 | 0.0 | front,iso-q,vs llm_only_s2lite |
| llm_only_s2lite | 83.4 [80.0, 86.6] | -16.2 | <0.001 | 0.00988 | 77.4 | 0.0118 | 5.64 | 0.0 | front,worse |
| intern_ts_cascade_tau0.9_s2lite | 82.8 [79.4, 86.2] | -0.6 | 0.347 | 0.00978 | 1.0 | 0.0118 | 4.39 | 0.0 | front,iso-best,vs llm_only_s2lite |
| intern_shadow_s2lite | 81.6 [78.2, 85.0] | -1.8 | 0.058 | 0.00981 | 0.7 | 0.0120 | 4.77 | 0.0 | vs llm_only_s2lite |
| jev_ts_replace | 79.0 [73.0, 84.5] | -21.0 | <0.001 | 0.0140 | 67.8 | 0.0177 | 5.97 | 100.0 | worse |
| intern_stop_guard_s2lite | 77.4 [73.6, 81.0] | -6.0 | <0.001 | 0.0144 | -45.3 | 0.0185 | 7.75 | 0.0 | worse,vs llm_only_s2lite |
| intern_ts_assist_k3_s2lite | 68.6 [64.2, 72.6] | -14.8 | <0.001 | 0.00458 | 53.7 | 0.00667 | 12.8 | 0.0 | front,worse,vs llm_only_s2lite |
| intern_ts_cascade_tau0.9 | 68.0 [61.5, 74.5] | -32.0 | <0.001 | 0.0281 | 35.4 | 0.0413 | 15.5 | 35.8 | worse |
| intern_ts_cascade_tau0.8 | 49.5 [42.5, 56.5] | -50.5 | <0.001 | 0.0216 | 50.4 | 0.0436 | 13.3 | 43.5 | worse |
| intern_ts_cascade_tau0.7 | 36.0 [29.5, 42.5] | -64.0 | <0.001 | 0.0161 | 63.0 | 0.0447 | 11.2 | 48.3 | worse |
| intern_ts_cascade_tau0.5 | 28.5 [22.5, 34.5] | -71.5 | <0.001 | 0.00836 | 80.8 | 0.0293 | 10.5 | 52.0 | worse |
| intern_ts_replace | 26.5 [20.5, 32.5] | -73.5 | <0.001 | 0.00792 | 81.8 | 0.0299 | 8.24 | 100.0 | worse |
| intern2b_ts_assist_k3 | 25.5 [19.5, 31.5] | -74.5 | <0.001 | 0.0278 | 36.1 | 0.109 | 12.2 | 0.0 | worse |
| intern08_ts_assist_k3 | 17.0 [12.0, 22.5] | -83.0 | <0.001 | 0.0311 | 28.5 | 0.183 | 11.0 | 0.0 | worse |
| intern_ts_params_replace | 14.0 [9.0, 19.0] | -86.0 | <0.001 | 0.00213 | 95.1 | 0.0152 | 6.44 | 90.5 | front,worse |
| intern08_ts_replace | 3.5 [1.5, 6.0] | -96.5 | <0.001 | 0.0402 | 7.6 | 1.15 | 14.5 | 100.0 | worse |
| intern2b_ts_replace | 3.0 [1.0, 5.5] | -97.0 | <0.001 | 0.0246 | 43.6 | 0.819 | 11.1 | 100.0 | worse |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

