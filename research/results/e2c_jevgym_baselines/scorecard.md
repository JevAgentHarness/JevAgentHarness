## Score card (baseline: `llm_only`)

- **Iso-quality cost savings**: 1.1% [0.4, 1.8] with `heuristic_stop_guard` (Δsuccess +0.0 pp [0.0, 0.0], p_worse=1.000; non-inferior at margin 3.0 pp, α=0.05; 8 arm(s) qualify)
- **Quality at iso-cost** (≤ 0.0409 $/episode): `heuristic_stop_guard` 100.0% vs. reference 100.0% (Δ +0.0 pp, p_better=1.000); hull 100.0%
- **Iso-quality cost savings** (vs. `llm_only_s2lite`): -0.1% [-0.5, 0.3] with `heuristic_stop_guard_s2lite` (Δsuccess +5.0 pp [1.5, 9.0], p_worse=0.998; non-inferior at margin 3.0 pp, α=0.05; 10 arm(s) qualify)
- **Quality at iso-cost** (vs. `llm_only_s2lite`) (≤ 0.00973 $/episode): `heuristic_stop_guard_s2lite` 86.5% vs. reference 81.2% (Δ +5.0 pp, p_better=0.004); hull 86.5%
- **Frontier cost-of-pass**: ∞ $/success over 65 arms (best single arm: `jev_ts_params_replace_answercopy`)
- **Hypervolume** (log-cost × success, normalised): all arms 0.715, baseline alone 0.274; front: `heuristic_stop_guard`, `heuristic_stop_guard_s2lite`, `heuristic_ts_replace`, `jev_route_tau0.2`, `jev_ts_assist_k3`, `jev_ts_assist_k5`, `jev_ts_params_replace_answercopy`, `randscore_route_tau0.2`, `randscore_route_tau0.4`, `rate10_reflect_trigger_s2lite`, `rate30_reflect_trigger_s2lite`, `rate50_reflect_trigger_s2lite`
- **Sweep `jev_route_tau*`** (5 arms): AIQ 0.899
- **Sweep `randscore_route_tau*`** (4 arms): AIQ 0.851
- note: per-System-2 references: `llm_only_s2lite` (gemini-3.5-flash-lite) for 10 arm(s); every other arm vs. `llm_only`

| arm | success | Δ pp | p_worse | cost $ | save % | CoP $ | lat s | deleg % | McNemar p | lat save % | LoP s | LLM-call ratio | strong-call ratio | PGR | EuB | HV contrib | LLM-free steps % | cache-mod $ | no-cache $ | cached % | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 100.0 [100.0, 100.0] | – | – | 0.0409 | – | 0.0409 | 14.8 | 0.0 | – | – | 14.8 | 1.00 | 1.00 | – | 0.500 | 0.000 | 0.0 | 0.0409 | 0.0707 | 46.7 | baseline |
| heuristic_stop_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0405 | 1.1 | 0.0405 | 9.85 | 0.0 | 1.000 | 33.6 | 9.85 | 1.00 | 1.00 | – | 0.505 | 0.000 | 0.0 | 0.0405 | 0.0707 | 47.6 | front,iso-best |
| jev_stop_trigger | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0406 | 0.8 | 0.0406 | 8.70 | 0.0 | 1.000 | 41.4 | 8.70 | 1.00 | 1.00 | – | 0.504 | 0.000 | 0.0 | 0.0406 | 0.0710 | 47.9 | iso-q |
| heuristic_reflect_trigger | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0407 | 0.6 | 0.0407 | 8.42 | 0.0 | 1.000 | 43.2 | 8.42 | 1.00 | 1.00 | – | 0.503 | 0.000 | 0.0 | 0.0407 | 0.0707 | 47.3 | iso-q |
| jev_stop_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0421 | -2.9 | 0.0421 | 9.00 | 0.0 | 1.000 | 39.3 | 9.00 | 1.02 | 1.02 | – | 0.486 | 0.000 | 0.0 | 0.0421 | 0.0723 | 46.4 | iso-q |
| rate10_reflect_trigger | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0423 | -3.4 | 0.0423 | 8.03 | 0.0 | 1.000 | 45.9 | 8.03 | 1.08 | 1.08 | – | 0.483 | 0.000 | 0.0 | 0.0423 | 0.0728 | 47.0 | iso-q |
| jev_reflect_trigger | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0429 | -4.8 | 0.0429 | 9.05 | 0.0 | 1.000 | 39.0 | 9.05 | 1.13 | 1.13 | – | 0.476 | 0.000 | 0.0 | 0.0429 | 0.0729 | 46.1 | iso-q |
| rate50_stop_guard | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0464 | -13.2 | 0.0464 | 11.9 | 0.0 | 1.000 | 20.0 | 11.9 | 1.12 | 1.12 | – | 0.436 | 0.000 | 0.0 | 0.0464 | 0.0805 | 47.7 | iso-q |
| s2self_reflect_trigger | 100.0 [100.0, 100.0] | +0.0 | 1.000 | 0.0528 | -29.0 | 0.0528 | 12.4 | 0.0 | 1.000 | 16.1 | 12.4 | 1.02 | 1.02 | – | 0.355 | 0.000 | 0.0 | 0.0528 | 0.0835 | 48.1 | iso-q |
| rate30_reflect_trigger | 99.0 [97.5, 100.0] | -1.0 | 0.144 | 0.0461 | -12.5 | 0.0465 | 9.34 | 0.0 | 0.500 | 37.1 | 9.43 | 1.28 | 1.28 | – | 0.437 | 0.000 | 0.0 | 0.0461 | 0.0782 | 47.7 |  |
| jev_memw | 99.0 [97.5, 100.0] | -1.0 | 0.125 | 0.0569 | -38.9 | 0.0574 | 11.1 | 0.0 | 0.500 | 25.4 | 11.2 | 1.01 | 1.01 | – | 0.319 | 0.000 | 0.0 | 0.0569 | 0.0805 | 36.9 |  |
| rate50_reflect_trigger | 98.5 [96.5, 100.0] | -1.5 | 0.054 | 0.0507 | -24.0 | 0.0515 | 11.0 | 0.0 | 0.250 | 26.2 | 11.1 | 1.46 | 1.46 | – | 0.383 | 0.000 | 0.0 | 0.0507 | 0.0832 | 46.6 |  |
| ratem_memw | 98.5 [96.5, 100.0] | -1.5 | 0.044 | 0.0556 | -35.8 | 0.0565 | 10.5 | 0.0 | 0.250 | 29.3 | 10.6 | 1.02 | 1.02 | – | 0.329 | 0.000 | 0.0 | 0.0556 | 0.0808 | 38.9 |  |
| heuristic_memw | 97.5 [95.0, 99.5] | -2.5 | 0.004 | 0.0517 | -26.3 | 0.0530 | 10.7 | 0.0 | 0.062 | 27.6 | 11.0 | 1.02 | 1.02 | – | 0.363 | 0.000 | 0.0 | 0.0517 | 0.0809 | 43.7 |  |
| reflect_always | 97.5 [95.0, 99.5] | -2.5 | 0.006 | 0.0677 | -65.3 | 0.0694 | 16.9 | 0.0 | 0.062 | -13.6 | 17.3 | 1.95 | 1.95 | – | 0.281 | 0.000 | 0.0 | 0.0677 | 0.100 | 41.5 |  |
| randscore_route_tau0.2 | 96.5 [93.5, 99.0] | -3.5 | <0.001 | 0.0338 | 17.5 | 0.0350 | 7.73 | 0.0 | 0.016 | 47.9 | 8.01 | 1.00 | 0.79 | – | 0.569 | 0.000 | 0.0 | 0.0338 | 0.0587 | 43.9 | front,worse |
| jev_route_tau0.2 | 96.0 [93.0, 98.5] | -4.0 | <0.001 | 0.0309 | 24.6 | 0.0321 | 9.29 | 0.0 | 0.008 | 37.4 | 9.68 | 1.01 | 0.60 | – | 0.602 | 0.000 | 0.0 | 0.0309 | 0.0478 | 38.2 | front,worse |
| s2self_ts_assist_k3 | 94.5 [91.0, 97.5] | -5.5 | <0.001 | 0.107 | -161.9 | 0.113 | 42.8 | 0.0 | <0.001 | -188.3 | 45.3 | 1.05 | 1.05 | – | 0.101 | 0.000 | 0.0 | 0.107 | 0.107 | 0.0 | worse |
| randscore_route_tau0.4 | 94.0 [90.5, 97.0] | -6.0 | <0.001 | 0.0289 | 29.5 | 0.0307 | 9.95 | 0.0 | <0.001 | 33.0 | 10.6 | 1.00 | 0.59 | – | 0.613 | 0.000 | 0.0 | 0.0289 | 0.0475 | 38.5 | front,worse |
| jev_ts_assist_k5 | 92.0 [88.0, 95.5] | -8.0 | <0.001 | 0.0224 | 45.2 | 0.0244 | 11.8 | 0.0 | <0.001 | 20.3 | 12.9 | 1.13 | 1.13 | – | 0.673 | 0.001 | 0.0 | 0.0224 | 0.0224 | 0.0 | front,worse |
| jev_ts_assist_k3 | 90.5 [86.0, 94.5] | -9.5 | <0.001 | 0.0195 | 52.3 | 0.0216 | 8.57 | 0.0 | <0.001 | 42.3 | 9.47 | 1.08 | 1.08 | – | 0.689 | 0.000 | 0.0 | 0.0195 | 0.0195 | 0.0 | front,worse |
| rate50_reflect_trigger_s2lite | 90.0 [85.5, 94.0] | +8.5 | 1.000 | 0.0116 | -19.5 | 0.0129 | 7.33 | 0.0 | 0.001 | -57.7 | 8.15 | 1.43 | 1.43 | – | 0.777 | 0.002 | 0.0 | 0.0116 | 0.0170 | 36.3 | front,iso-q,vs llm_only_s2lite |
| rate50_route | 89.0 [84.5, 93.5] | -11.0 | <0.001 | 0.0255 | 37.6 | 0.0287 | 10.4 | 0.0 | <0.001 | 30.1 | 11.6 | 1.00 | 0.51 | – | 0.618 | 0.000 | 0.0 | 0.0255 | 0.0430 | 40.0 | worse |
| rate30_reflect_trigger_s2lite | 88.0 [83.5, 92.5] | +6.5 | 0.998 | 0.0106 | -9.7 | 0.0121 | 6.14 | 0.0 | 0.011 | -32.1 | 6.98 | 1.25 | 1.25 | – | 0.770 | 0.000 | 0.0 | 0.0106 | 0.0158 | 36.1 | front,iso-q,vs llm_only_s2lite |
| randscore_route_tau0.6 | 88.0 [83.5, 92.0] | -12.0 | <0.001 | 0.0216 | 47.3 | 0.0245 | 6.29 | 0.0 | <0.001 | 57.6 | 7.15 | 1.01 | 0.38 | – | 0.653 | 0.000 | 0.0 | 0.0216 | 0.0355 | 37.1 | worse |
| rate10_reflect_trigger_s2lite | 87.0 [82.5, 91.5] | +5.5 | 0.996 | 0.0100 | -3.3 | 0.0115 | 5.18 | 0.0 | 0.019 | -11.5 | 5.96 | 1.08 | 1.08 | – | 0.768 | 0.000 | 0.0 | 0.0100 | 0.0147 | 33.4 | front,iso-q,vs llm_only_s2lite |
| randscore_route_tau0.8 | 87.0 [82.0, 91.0] | -13.0 | <0.001 | 0.0159 | 61.1 | 0.0183 | 5.92 | 0.0 | <0.001 | 60.1 | 6.81 | 1.00 | 0.20 | – | 0.706 | 0.000 | 0.0 | 0.0159 | 0.0255 | 34.5 | worse |
| heuristic_stop_guard_s2lite | 86.5 [81.5, 91.0] | +5.0 | 0.998 | 0.00972 | -0.1 | 0.0112 | 4.59 | 0.0 | 0.013 | 1.3 | 5.31 | 1.00 | 1.00 | – | 0.766 | 0.000 | 0.0 | 0.00972 | 0.0142 | 32.3 | front,iso-best,vs llm_only_s2lite |
| rate30_route | 86.5 [82.0, 90.5] | -13.5 | <0.001 | 0.0193 | 52.8 | 0.0223 | 6.79 | 0.0 | <0.001 | 54.3 | 7.85 | 1.01 | 0.31 | – | 0.670 | 0.000 | 0.0 | 0.0193 | 0.0317 | 37.0 | worse |
| lite_ts_assist_k3 | 86.5 [81.5, 91.0] | -13.5 | <0.001 | 0.0368 | 10.2 | 0.0425 | 24.7 | 0.0 | <0.001 | -66.3 | 28.5 | 1.01 | 1.01 | – | 0.467 | 0.000 | 0.0 | 0.0368 | 0.0368 | 0.0 | worse |
| ratem_reflect_trigger_s2lite | 85.4 [82.4, 88.6] | +4.2 | 0.996 | 0.0103 | -5.9 | 0.0121 | 5.56 | 0.0 | 0.016 | -21.2 | 6.52 | 1.13 | 1.13 | – | 0.751 | 0.000 | 0.0 | 0.0103 | 0.0151 | 33.8 | iso-q,vs llm_only_s2lite |
| jev_route_tau0.5 | 84.5 [79.5, 89.5] | -15.5 | <0.001 | 0.0115 | 71.8 | 0.0136 | 6.75 | 0.0 | <0.001 | 54.5 | 7.99 | 1.01 | 0.05 | – | 0.735 | 0.000 | 0.0 | 0.0115 | 0.0170 | 33.2 | worse |
| jev_stop_guard_s2lite | 84.5 [79.5, 89.0] | +3.0 | 0.974 | 0.0120 | -23.8 | 0.0142 | 6.34 | 0.0 | 0.146 | -36.5 | 7.51 | 1.22 | 1.22 | – | 0.747 | 0.000 | 0.0 | 0.0120 | 0.0182 | 32.5 | iso-q,vs llm_only_s2lite |
| jev_route_tau0.8 | 84.0 [79.0, 89.0] | -16.0 | <0.001 | 0.0101 | 75.4 | 0.0120 | 6.17 | 0.0 | <0.001 | 58.5 | 7.34 | 1.01 | 0.00 | – | 0.742 | 0.000 | 0.0 | 0.0101 | 0.0146 | 32.3 | worse |
| s2self_reflect_trigger_s2lite | 84.0 [78.5, 89.0] | +2.5 | 0.952 | 0.0122 | -25.8 | 0.0145 | 8.06 | 0.0 | 0.227 | -73.4 | 9.60 | 1.00 | 1.00 | – | 0.722 | 0.000 | 0.0 | 0.0122 | 0.0167 | 32.1 | iso-q,vs llm_only_s2lite |
| lite_route | 84.0 [78.5, 89.0] | -16.0 | <0.001 | 0.0142 | 65.2 | 0.0170 | 9.66 | 0.0 | <0.001 | 34.9 | 11.5 | 1.01 | 0.06 | – | 0.703 | 0.000 | 0.0 | 0.0142 | 0.0199 | 33.5 | worse |
| heuristic_route | 83.5 [78.5, 88.5] | -16.5 | <0.001 | 0.00983 | 76.0 | 0.0118 | 4.65 | 0.0 | <0.001 | 68.7 | 5.57 | 1.01 | 0.00 | – | 0.739 | 0.000 | 0.0 | 0.00983 | 0.0144 | 32.2 | worse |
| ratem_route | 83.5 [78.0, 88.5] | -16.5 | <0.001 | 0.0114 | 72.3 | 0.0136 | 4.83 | 0.0 | <0.001 | 67.5 | 5.78 | 1.01 | 0.05 | – | 0.726 | 0.000 | 0.0 | 0.0114 | 0.0169 | 32.6 | worse |
| heuristic_reflect_trigger_s2lite | 83.0 [77.5, 88.0] | +1.5 | 0.868 | 0.00974 | -0.3 | 0.0117 | 4.64 | 0.0 | 0.508 | 0.1 | 5.60 | 1.00 | 1.00 | – | 0.736 | 0.000 | 0.0 | 0.00974 | 0.0142 | 32.0 | iso-q,vs llm_only_s2lite |
| jev_reflect_trigger_s2lite | 83.0 [79.8, 86.2] | +1.8 | 0.860 | 0.00991 | -1.9 | 0.0119 | 7.05 | 0.0 | 0.336 | -53.5 | 8.49 | 1.14 | 1.14 | – | 0.736 | 0.000 | 0.0 | 0.00991 | 0.0153 | 39.3 | iso-q,vs llm_only_s2lite |
| ratem_stop_guard_s2lite | 83.0 [77.5, 88.0] | +1.5 | 0.833 | 0.0102 | -5.3 | 0.0123 | 4.98 | 0.0 | 0.581 | -7.1 | 6.00 | 1.06 | 1.06 | – | 0.731 | 0.000 | 0.0 | 0.0102 | 0.0152 | 33.8 | iso-q,vs llm_only_s2lite |
| jev_route_tau0.4 | 83.0 [77.5, 88.0] | -17.0 | <0.001 | 0.0129 | 68.4 | 0.0156 | 7.64 | 0.0 | <0.001 | 48.5 | 9.20 | 1.01 | 0.10 | – | 0.708 | 0.000 | 0.0 | 0.0129 | 0.0193 | 35.0 | worse |
| rate10_route | 83.0 [77.5, 88.0] | -17.0 | <0.001 | 0.0132 | 67.9 | 0.0159 | 5.94 | 0.0 | <0.001 | 60.0 | 7.16 | 1.01 | 0.11 | – | 0.703 | 0.000 | 0.0 | 0.0132 | 0.0205 | 34.3 | worse |
| s2self_route | 82.5 [77.0, 87.5] | -17.5 | <0.001 | 0.0226 | 44.7 | 0.0274 | 13.9 | 0.0 | <0.001 | 6.5 | 16.8 | 1.01 | 0.01 | – | 0.615 | 0.000 | 0.0 | 0.0226 | 0.0274 | 32.9 | worse |
| jev_route_tau0.6 | 82.0 [76.5, 87.0] | -18.0 | <0.001 | 0.0101 | 75.4 | 0.0123 | 6.21 | 0.0 | <0.001 | 58.2 | 7.57 | 1.01 | 0.00 | – | 0.725 | 0.000 | 0.0 | 0.0101 | 0.0146 | 32.5 | worse |
| jev_ts_replace | 81.5 [76.0, 86.5] | -18.5 | <0.001 | 0.0142 | 65.4 | 0.0174 | 7.12 | 100.0 | <0.001 | 52.0 | 8.74 | 0.92 | 0.92 | – | 0.647 | 0.000 | 0.0 | 0.0142 | 0.0142 | 0.0 | worse |
| llm_only_s2lite | 81.2 [77.8, 84.6] | -18.5 | <0.001 | 0.00973 | 76.3 | 0.0120 | 4.59 | 0.0 | <0.001 | 68.7 | 5.65 | 1.01 | 1.01 | – | 0.721 | 0.000 | 0.0 | 0.00973 | 0.0142 | 32.5 | worse |
| jev_ts_params_replace_answercopy | 80.0 [74.5, 85.5] | -20.0 | <0.001 | 0.00548 | 86.6 | 0.00685 | 6.70 | 90.6 | <0.001 | 54.8 | 8.38 | 0.25 | 0.25 | – | 0.747 | 0.089 | 74.9 | 0.00548 | 0.00548 | 0.0 | front,worse |
| rate10_stop_trigger | 50.5 [43.5, 57.5] | -49.5 | <0.001 | 0.0292 | 28.7 | 0.0578 | 5.13 | 0.0 | <0.001 | 65.5 | 10.1 | 0.77 | 0.77 | – | 0.282 | 0.000 | 0.0 | 0.0292 | 0.0491 | 43.6 | worse |
| heuristic_stop_trigger | 50.0 [43.0, 56.5] | -50.0 | <0.001 | 0.0311 | 24.1 | 0.0622 | 5.56 | 0.0 | <0.001 | 62.6 | 11.1 | 0.81 | 0.81 | – | 0.335 | 0.000 | 0.0 | 0.0311 | 0.0511 | 44.1 | worse |
| lite_ts_replace | 41.5 [35.0, 49.0] | -58.5 | <0.001 | 0.0242 | 40.9 | 0.0583 | 18.4 | 100.0 | <0.001 | -23.9 | 44.3 | 0.76 | 0.76 | – | 0.240 | 0.000 | 0.0 | 0.0242 | 0.0242 | 0.0 | worse |
| tfidf_ts_assist_k3 | 33.0 [26.5, 39.5] | -67.0 | <0.001 | 0.0134 | 67.2 | 0.0407 | 5.24 | 0.0 | <0.001 | 64.7 | 15.9 | 0.97 | 0.97 | – | 0.289 | 0.000 | 0.0 | 0.0134 | 0.0134 | 0.0 | worse |
| heuristic_ts_assist_k3 | 31.0 [24.5, 38.0] | -69.0 | <0.001 | 0.0181 | 55.8 | 0.0584 | 6.67 | 0.0 | <0.001 | 55.1 | 21.5 | 1.25 | 1.25 | – | 0.271 | 0.000 | 0.0 | 0.0181 | 0.0184 | 0.4 | worse |
| embed_ts_assist_k3 | 30.0 [23.5, 36.5] | -70.0 | <0.001 | 0.00975 | 76.2 | 0.0325 | 4.91 | 0.0 | <0.001 | 66.9 | 16.4 | 0.67 | 0.67 | – | 0.223 | 0.000 | 0.0 | 0.00975 | 0.00975 | 0.0 | worse |
| tfidf_ts_assist_k5 | 25.0 [19.0, 31.0] | -75.0 | <0.001 | 0.0243 | 40.6 | 0.0972 | 10.8 | 0.0 | <0.001 | 27.0 | 43.4 | 1.46 | 1.46 | – | 0.230 | 0.000 | 0.0 | 0.0243 | 0.0243 | 0.0 | worse |
| bm25task_ts_assist_k3 | 24.5 [18.5, 30.5] | -75.5 | <0.001 | 0.0150 | 63.5 | 0.0610 | 6.73 | 0.0 | <0.001 | 54.7 | 27.5 | 1.17 | 1.17 | – | 0.222 | 0.000 | 0.0 | 0.0150 | 0.0151 | 0.1 | worse |
| bm25_ts_assist_k3 | 18.8 [15.4, 22.2] | -83.0 | <0.001 | 0.0161 | 62.7 | 0.0854 | 5.81 | 0.0 | <0.001 | 62.3 | 30.9 | 1.04 | 1.04 | – | 0.175 | 0.000 | 0.0 | 0.0161 | 0.0167 | 0.5 | worse |
| bm25_ts_assist_k5 | 18.5 [13.5, 24.0] | -81.5 | <0.001 | 0.0163 | 60.1 | 0.0883 | 7.10 | 0.0 | <0.001 | 52.2 | 38.4 | 1.00 | 1.00 | – | 0.159 | 0.000 | 0.0 | 0.0163 | 0.0166 | 0.2 | worse |
| heuristic_ts_replace | 12.5 [8.0, 17.5] | -87.5 | <0.001 | 0.00175 | 95.7 | 0.0140 | 1.53 | 100.0 | <0.001 | 89.7 | 12.3 | 0.28 | 0.28 | – | 0.121 | 0.033 | 0.0 | 0.00175 | 0.00175 | 0.0 | front,worse |
| bm25_ts_replace | 12.5 [8.0, 17.5] | -87.5 | <0.001 | 0.0150 | 63.3 | 0.120 | 6.19 | 100.0 | <0.001 | 58.3 | 49.5 | 1.16 | 1.16 | – | 0.042 | 0.000 | 0.0 | 0.0150 | 0.0150 | 0.0 | worse |
| tfidf_ts_replace | 12.5 [8.0, 17.0] | -87.5 | <0.001 | 0.0187 | 54.4 | 0.149 | 6.07 | 100.0 | <0.001 | 59.1 | 48.5 | 0.97 | 0.97 | – | 0.042 | 0.000 | 0.0 | 0.0187 | 0.0233 | 2.7 | worse |
| random_ts_assist_k3 | 6.0 [3.0, 9.5] | -94.0 | <0.001 | 0.0358 | 12.5 | 0.597 | 10.4 | 0.0 | <0.001 | 29.9 | 173. | 1.98 | 1.98 | – | 0.032 | 0.000 | 0.0 | 0.0358 | 0.0358 | 0.0 | worse |
| random_ts_replace | 0.0 [0.0, 0.0] | -100.0 | <0.001 | 0.00300 | 92.7 | ∞ | 2.37 | 100.0 | <0.001 | 84.0 | ∞ | 0.44 | 0.44 | – | 0.000 | 0.000 | 0.0 | 0.00300 | 0.00300 | 0.0 | worse |
| embed_ts_replace | 0.0 [0.0, 0.0] | -100.0 | <0.001 | 0.00871 | 78.7 | ∞ | 4.34 | 100.0 | <0.001 | 70.7 | ∞ | 0.57 | 0.57 | – | 0.000 | 0.000 | 0.0 | 0.00871 | 0.00871 | 0.0 | worse |
| prior_recent_ts_replace | 0.0 [0.0, 0.0] | -100.0 | <0.001 | 0.0404 | 1.3 | ∞ | 18.9 | 100.0 | <0.001 | -27.1 | ∞ | 2.88 | 2.88 | – | 0.000 | 0.000 | 0.0 | 0.0404 | 0.0451 | 2.7 | worse |

_success: task-level bootstrap 95% CI; Δ/p_worse: paired bootstrap vs. the arm's reference (one-sided, small = worse; the baseline, or the LLM-only arm of the arm's System-2 model: flag `vs <ref>`); save: 1 − cost/reference cost; CoP: cost-of-pass ($ per success); deleg: decisions taken by System-1 without the LLM; flags: front = Pareto-optimal, worse = significantly worse (Holm), iso-best / iso-q = qualifies for iso-quality savings._

_cost $: the billed (ledger) cost, the primary figure; cache-mod $: System-2 calls re-priced from their logged tokens with the measured cached tokens at the model's cached rate; no-cache $: list price with no cache; cached %: measured cached share of System-2 input tokens (results/pilot_jev/CACHE_NOTES.md)._

### Non-inferiority (A6 procedure; margin 3.0 pp)

| arm | reference | n | Δ pp | 95% LCB pp | p (rule) | Holm p | α_adj | Tango LB(1−α_adj) pp | discordant (w/l) | exact p (loss UB pp) | MDE Δ pp (80%) | p_boot | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heuristic_stop_guard | llm_only | 200 | +0.0 | +0.0 | 0.002 (bb_ordered) | 0.128 | 0.0009 | -4.6 | 0 (0/0) | 0.002 (3.42) | – | <0.001 | inconclusive at this n |
| jev_stop_trigger | llm_only | 200 | +0.0 | +0.0 | 0.002 (bb_ordered) | 0.128 | 0.0010 | -4.6 | 0 (0/0) | 0.002 (3.39) | – | <0.001 | inconclusive at this n |
| heuristic_reflect_trigger | llm_only | 200 | +0.0 | +0.0 | 0.002 (bb_ordered) | 0.128 | 0.0009 | -4.6 | 0 (0/0) | 0.002 (3.43) | – | <0.001 | inconclusive at this n |
| jev_stop_guard | llm_only | 200 | +0.0 | +0.0 | 0.002 (bb_ordered) | 0.128 | 0.0010 | -4.6 | 0 (0/0) | 0.002 (3.40) | – | <0.001 | inconclusive at this n |
| rate10_reflect_trigger | llm_only | 200 | +0.0 | +0.0 | 0.002 (bb_ordered) | 0.128 | 0.0010 | -4.5 | 0 (0/0) | 0.002 (3.39) | – | <0.001 | inconclusive at this n |
| jev_reflect_trigger | llm_only | 200 | +0.0 | +0.0 | 0.002 (bb_ordered) | 0.128 | 0.0010 | -4.6 | 0 (0/0) | 0.002 (3.41) | – | <0.001 | inconclusive at this n |
| rate50_stop_guard | llm_only | 200 | +0.0 | +0.0 | 0.002 (bb_ordered) | 0.128 | 0.0010 | -4.5 | 0 (0/0) | 0.002 (3.38) | – | <0.001 | inconclusive at this n |
| s2self_reflect_trigger | llm_only | 200 | +0.0 | +0.0 | 0.002 (bb_ordered) | 0.128 | 0.0011 | -4.5 | 0 (0/0) | 0.002 (3.37) | – | <0.001 | inconclusive at this n |
| rate30_reflect_trigger | llm_only | 200 | -1.0 | -2.5 | 0.059 (bb_ordered) | 1.000 | 0.0011 | -6.2 | 2 (0/2) | 0.059 (5.43) | -0.3 | 0.016 | inconclusive at this n |
| jev_memw | llm_only | 200 | -1.0 | -2.5 | 0.059 (bb_ordered) | 1.000 | 0.0011 | -6.2 | 2 (0/2) | 0.059 (5.44) | -0.3 | 0.019 | inconclusive at this n |
| rate50_reflect_trigger | llm_only | 200 | -1.5 | -3.0 | 0.155 (bb_ordered) | 1.000 | 0.0011 | -7.0 | 3 (0/3) | 0.147 (6.29) | +0.3 | 0.076 | inconclusive at this n |
| ratem_memw | llm_only | 200 | -1.5 | -3.0 | 0.155 (bb_ordered) | 1.000 | 0.0012 | -7.0 | 3 (0/3) | 0.147 (6.28) | +0.3 | 0.081 | inconclusive at this n |
| heuristic_memw | llm_only | 200 | -2.5 | -4.5 | 0.459 (bb_ordered) | 1.000 | 0.0012 | -8.5 | 5 (0/5) | 0.443 (7.88) | +1.3 | 0.391 | not NI |
| reflect_always | llm_only | 200 | -2.5 | -4.5 | 0.459 (bb_ordered) | 1.000 | 0.0012 | -8.5 | 5 (0/5) | 0.443 (7.87) | +1.3 | 0.361 | not NI |
| randscore_route_tau0.2 | llm_only | 200 | -3.5 | -5.5 | 0.747 (bb_ordered) | 1.000 | 0.0013 | -9.9 | 7 (0/7) | 0.746 (9.35) | +2.1 | 0.693 | not NI |
| jev_route_tau0.2 | llm_only | 200 | -4.0 | -6.5 | 0.856 (bb_ordered) | 1.000 | 0.0013 | -10.6 | 8 (0/8) | 0.850 (10.06) | +2.4 | 0.813 | not NI |
| s2self_ts_assist_k3 | llm_only | 200 | -5.5 | -8.0 | 0.983 (bb_ordered) | 1.000 | 0.0013 | -12.5 | 11 (0/11) | 0.982 (12.12) | +3.3 | 0.968 | not NI |
| randscore_route_tau0.4 | llm_only | 200 | -6.0 | -9.0 | 0.993 (bb_ordered) | 1.000 | 0.0014 | -13.2 | 12 (0/12) | 0.992 (12.78) | +3.6 | 0.986 | not NI |
| jev_ts_assist_k5 | llm_only | 200 | -8.0 | -11.0 | 1.000 (bb_ordered) | 1.000 | 0.0024 | -15.2 | 16 (0/16) | 1.000 (14.91) | +4.3 | 1.000 | not NI |
| jev_ts_assist_k3 | llm_only | 200 | -9.5 | -13.0 | 1.000 (bb_ordered) | 1.000 | 0.0023 | -17.0 | 19 (0/19) | 1.000 (16.82) | +5.0 | 1.000 | not NI |
| rate50_reflect_trigger_s2lite | llm_only_s2lite | 200 | +8.5 | +4.5 | <0.001 (bb_ordered) | 0.001 | 0.0050 | +2.4 | 25 (21/4) | – | +5.5 | <0.001 | NI |
| rate50_route | llm_only | 200 | -11.0 | -14.5 | 1.000 (bb_ordered) | 1.000 | 0.0083 | -17.4 | 22 (0/22) | 1.000 (17.38) | +4.5 | 1.000 | not NI |
| rate30_reflect_trigger_s2lite | llm_only_s2lite | 200 | +6.5 | +2.5 | <0.001 (bb_ordered) | 0.001 | 0.0071 | +0.7 | 23 (18/5) | – | +4.8 | <0.001 | NI |
| randscore_route_tau0.6 | llm_only | 200 | -12.0 | -15.5 | 1.000 (bb_ordered) | 1.000 | 0.0045 | -19.3 | 24 (0/24) | 1.000 (19.18) | +5.4 | 1.000 | not NI |
| rate10_reflect_trigger_s2lite | llm_only_s2lite | 200 | +5.5 | +2.0 | <0.001 (bb_ordered) | 0.001 | 0.0083 | +0.3 | 19 (15/4) | – | +4.0 | <0.001 | NI |
| randscore_route_tau0.8 | llm_only | 200 | -13.0 | -16.5 | 1.000 (bb_ordered) | 1.000 | 0.0050 | -20.3 | 26 (0/26) | 1.000 (20.26) | +5.7 | 1.000 | not NI |
| heuristic_stop_guard_s2lite | llm_only_s2lite | 200 | +5.0 | +2.0 | <0.001 (bb_ordered) | 0.001 | 0.0063 | +0.4 | 14 (12/2) | – | +3.2 | <0.001 | NI |
| rate30_route | llm_only | 200 | -13.5 | -17.5 | 1.000 (bb_ordered) | 1.000 | 0.0071 | -20.5 | 27 (0/27) | 1.000 (20.47) | +5.5 | 1.000 | not NI |
| lite_ts_assist_k3 | llm_only | 200 | -13.5 | -17.5 | 1.000 (bb_ordered) | 1.000 | 0.0029 | -21.5 | 27 (0/27) | 1.000 (21.39) | +6.3 | 1.000 | not NI |
| ratem_reflect_trigger_s2lite | llm_only_s2lite | 500 | +4.2 | +1.4 | <0.001 (bb_ordered) | 0.001 | 0.0056 | -0.0 | 69 (45/24) | – | +2.6 | <0.001 | NI |
| jev_route_tau0.5 | llm_only | 200 | -15.5 | -19.5 | 1.000 (bb_ordered) | 1.000 | 0.0020 | -24.2 | 31 (0/31) | 1.000 (24.10) | +7.3 | 1.000 | not NI |
| jev_stop_guard_s2lite | llm_only_s2lite | 200 | +3.0 | +0.0 | 0.001 (bb_ordered) | 0.005 | 0.0100 | -1.3 | 12 (9/3) | – | +2.4 | 0.001 | NI |
| jev_route_tau0.8 | llm_only | 200 | -16.0 | -20.0 | 1.000 (bb_ordered) | 1.000 | 0.0022 | -24.7 | 32 (0/32) | 1.000 (24.60) | +7.4 | 1.000 | not NI |
| s2self_reflect_trigger_s2lite | llm_only_s2lite | 200 | +2.5 | +0.0 | 0.002 (bb_ordered) | 0.008 | 0.0125 | -1.5 | 11 (8/3) | – | +2.1 | <0.001 | NI |
| lite_route | llm_only | 200 | -16.0 | -20.0 | 1.000 (bb_ordered) | 1.000 | 0.0028 | -24.4 | 32 (0/32) | 1.000 (24.34) | +7.2 | 1.000 | not NI |
| heuristic_route | llm_only | 200 | -16.5 | -20.5 | 1.000 (bb_ordered) | 1.000 | 0.0017 | -25.6 | 33 (0/33) | 1.000 (25.44) | +7.8 | 1.000 | not NI |
| ratem_route | llm_only | 200 | -16.5 | -20.5 | 1.000 (bb_ordered) | 1.000 | 0.0100 | -23.5 | 33 (0/33) | 1.000 (23.51) | +6.1 | 1.000 | not NI |
| heuristic_reflect_trigger_s2lite | llm_only_s2lite | 200 | +1.5 | -1.0 | 0.005 (bb_ordered) | 0.010 | 0.0250 | -1.7 | 9 (6/3) | – | +1.2 | 0.003 | NI |
| jev_reflect_trigger_s2lite | llm_only_s2lite | 500 | +1.8 | -1.0 | 0.002 (bb_ordered) | 0.008 | 0.0167 | -1.8 | 69 (39/30) | – | +1.9 | 0.001 | NI |
| ratem_stop_guard_s2lite | llm_only_s2lite | 200 | +1.5 | -1.5 | 0.012 (bb_ordered) | 0.012 | 0.0500 | -1.6 | 13 (8/5) | – | +1.5 | 0.009 | NI |
| jev_route_tau0.4 | llm_only | 200 | -17.0 | -21.0 | 1.000 (bb_ordered) | 1.000 | 0.0019 | -26.0 | 34 (0/34) | 1.000 (25.87) | +7.8 | 1.000 | not NI |
| rate10_route | llm_only | 200 | -17.0 | -21.0 | 1.000 (bb_ordered) | 1.000 | 0.0056 | -24.7 | 34 (0/34) | 1.000 (24.74) | +6.8 | 1.000 | not NI |
| s2self_route | llm_only | 200 | -17.5 | -21.5 | 1.000 (bb_ordered) | 1.000 | 0.0125 | -24.3 | 35 (0/35) | 1.000 (24.36) | +6.1 | 1.000 | not NI |
| jev_route_tau0.6 | llm_only | 200 | -18.0 | -22.0 | 1.000 (bb_ordered) | 1.000 | 0.0021 | -27.0 | 36 (0/36) | 1.000 (26.92) | +8.1 | 1.000 | not NI |
| jev_ts_replace | llm_only | 200 | -18.5 | -23.0 | 1.000 (bb_ordered) | 1.000 | 0.0026 | -27.3 | 37 (0/37) | 1.000 (27.24) | +8.0 | 1.000 | not NI |
| llm_only_s2lite | llm_only | 200 | -18.5 | -23.0 | 1.000 (bb_ordered) | 1.000 | 0.0033 | -27.0 | 37 (0/37) | 1.000 (26.99) | +7.7 | 1.000 | not NI |
| jev_ts_params_replace_answercopy | llm_only | 200 | -20.0 | -24.5 | 1.000 (bb_ordered) | 1.000 | 0.0025 | -29.0 | 40 (0/40) | 1.000 (28.98) | +8.5 | 1.000 | not NI |
| rate10_stop_trigger | llm_only | 200 | -49.5 | -55.5 | 1.000 (bb_ordered) | 1.000 | 0.0063 | -58.2 | 99 (0/99) | 1.000 (58.51) | +13.5 | 1.000 | not NI |
| heuristic_stop_trigger | llm_only | 200 | -50.0 | -56.0 | 1.000 (bb_ordered) | 1.000 | 0.0017 | -60.1 | 100 (0/100) | 1.000 (60.46) | +15.7 | 1.000 | not NI |
| lite_ts_replace | llm_only | 200 | -58.5 | -65.0 | 1.000 (bb_ordered) | 1.000 | 0.0031 | -67.6 | 117 (0/117) | 1.000 (67.94) | +16.2 | 1.000 | not NI |
| tfidf_ts_assist_k3 | llm_only | 200 | -67.0 | -72.0 | 1.000 (bb_ordered) | 1.000 | 0.0167 | -73.6 | 134 (0/134) | 1.000 (73.98) | +14.1 | 1.000 | not NI |
| heuristic_ts_assist_k3 | llm_only | 200 | -69.0 | -74.0 | 1.000 (bb_ordered) | 1.000 | 0.0018 | -77.6 | 138 (0/138) | 1.000 (78.09) | +18.9 | 1.000 | not NI |
| embed_ts_assist_k3 | llm_only | 200 | -70.0 | -75.5 | 1.000 (bb_ordered) | 1.000 | 0.0016 | -78.6 | 140 (0/140) | 1.000 (79.09) | +19.3 | 1.000 | not NI |
| tfidf_ts_assist_k5 | llm_only | 200 | -75.0 | -80.0 | 1.000 (bb_ordered) | 1.000 | 0.0250 | -80.5 | 150 (0/150) | 1.000 (80.84) | +14.1 | 1.000 | not NI |
| bm25task_ts_assist_k3 | llm_only | 200 | -75.5 | -80.5 | 1.000 (bb_ordered) | 1.000 | 0.0015 | -83.3 | 151 (0/151) | 1.000 (83.86) | +20.2 | 1.000 | not NI |
| bm25_ts_assist_k3 | llm_only | 200 | -83.0 | -87.0 | 1.000 (bb_ordered) | 1.000 | 0.0014 | -89.5 | 166 (0/166) | 1.000 (90.06) | +21.5 | 1.000 | not NI |
| bm25_ts_assist_k5 | llm_only | 200 | -81.5 | -85.5 | 1.000 (bb_ordered) | 1.000 | 0.0014 | -88.3 | 163 (0/163) | 1.000 (88.85) | +21.3 | 1.000 | not NI |
| heuristic_ts_replace | llm_only | 200 | -87.5 | -91.5 | 1.000 (bb_ordered) | 1.000 | 0.0019 | -92.8 | 175 (0/175) | 1.000 (93.35) | +21.6 | 1.000 | not NI |
| bm25_ts_replace | llm_only | 200 | -87.5 | -91.5 | 1.000 (bb_ordered) | 1.000 | 0.0015 | -92.9 | 175 (0/175) | 1.000 (93.46) | +22.1 | 1.000 | not NI |
| tfidf_ts_replace | llm_only | 200 | -87.5 | -91.5 | 1.000 (bb_ordered) | 1.000 | 0.0500 | -90.9 | 175 (0/175) | 1.000 (91.16) | +13.4 | 1.000 | not NI |
| random_ts_assist_k3 | llm_only | 200 | -94.0 | -96.5 | 1.000 (bb_ordered) | 1.000 | 0.0038 | -97.1 | 188 (0/188) | 1.000 (97.57) | +20.9 | 1.000 | not NI |
| random_ts_replace | llm_only | 200 | -100.0 | -100.0 | 1.000 (bb_ordered) | 1.000 | 0.0042 | -100.0 | 200 (0/200) | 1.000 (100.00) | +21.5 | 1.000 | not NI |
| embed_ts_replace | llm_only | 200 | -100.0 | -100.0 | 1.000 (bb_ordered) | 1.000 | 0.0016 | -100.0 | 200 (0/200) | 1.000 (100.00) | +23.6 | 1.000 | not NI |
| prior_recent_ts_replace | llm_only | 200 | -100.0 | -100.0 | 1.000 (bb_ordered) | 1.000 | 0.0036 | -100.0 | 200 (0/200) | 1.000 (100.00) | +21.8 | 1.000 | not NI |

_A6.14: p_T = 1 − Φ(Z(−m)), Tango's one-sided score test of H0: Δ ≤ −m on the paired (wins, losses, n) counts, enters Holm with the family (every acting arm vs. the same reference); NI holds iff p_T ≤ α_adj of its Holm step, equivalently iff the Tango (1 − α_adj) lower bound is above −m. With zero wins or < 5 discordant tasks the exact binomial on losses (p = P[L ≤ l | n, m]) decides instead (`rule` = exact; it is the p that enters Holm, `Holm p`) and the one-sided 95% upper bound on the loss rate (1 − 0.05^(1/n) at zero losses) replaces a zero-width interval. A primary that fails while its two-sided 95% CI covers 0 is *inconclusive at this n*, not inferior; MDE: the smallest true Δ shown NI with 80% power at the achieved discordance and α_adj. p: the certified Berger-Boos p-value of one ordering of all (wins, losses) outcomes (A6.21; rule bb_ordered). Tango LB: descriptive score interval._

### System-1 decisions

| arm | n scored | S1 acc | ECE | ACE | Brier | AUROC | AURC | E-AURC | acc@80%cov | harmful % | guard FAR | guard FRR | rescue % | yes/no n | yes/no ECE | yes/no Brier | yes/no AUROC | mem stored % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heuristic_stop_guard | 0 | – | – | – | – | – | – | – | – | – | – | 0.000 | – | 200 | 0.010 | 0.000 | – | – |
| jev_stop_trigger | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1300 | 0.029 | 0.001 | – | – |
| heuristic_reflect_trigger | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1300 | 0.010 | 0.000 | – | – |
| jev_stop_guard | 0 | – | – | – | – | – | – | – | – | – | – | 0.104 | – | 224 | 0.203 | 0.081 | – | – |
| rate10_reflect_trigger | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1193 | 0.001 | 0.000 | – | – |
| jev_reflect_trigger | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1122 | 0.159 | 0.029 | – | – |
| rate50_stop_guard | 0 | – | – | – | – | – | – | – | – | – | – | 0.480 | – | 385 | 0.481 | 0.480 | – | – |
| s2self_reflect_trigger | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1272 | 0.050 | 0.003 | – | – |
| rate30_reflect_trigger | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 918 | 0.001 | 0.002 | 0.500 | – |
| jev_memw | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1310 | 0.267 | 0.218 | 0.889 | 99.5 |
| rate50_reflect_trigger | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 669 | 0.002 | 0.003 | 0.500 | – |
| ratem_memw | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1326 | 0.325 | 0.325 | 0.497 | 99.4 |
| heuristic_memw | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1328 | 0.351 | 0.354 | 0.486 | 92.2 |
| randscore_route_tau0.2 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 323 | 0.077 | 0.031 | 0.431 | – |
| jev_route_tau0.2 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 612 | 0.138 | 0.032 | 0.779 | – |
| s2self_ts_assist_k3 | 1574 | 0.915 | 0.075 | 0.075 | 0.077 | 0.734 | 0.047 | 0.044 | 0.957 | – | – | – | – | 0 | – | – | – | – |
| randscore_route_tau0.4 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 623 | 0.180 | 0.063 | 0.522 | – |
| jev_ts_assist_k5 | 1688 | 0.867 | 0.047 | 0.036 | 0.089 | 0.821 | 0.045 | 0.036 | 0.940 | – | – | – | – | 0 | – | – | – | – |
| jev_ts_assist_k3 | 1619 | 0.893 | 0.045 | 0.043 | 0.080 | 0.774 | 0.045 | 0.039 | 0.947 | – | – | – | – | 0 | – | – | – | – |
| rate50_reflect_trigger_s2lite | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1311 | 0.496 | 0.496 | 0.563 | – |
| rate50_route | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 733 | 0.029 | 0.030 | 0.500 | – |
| rate30_reflect_trigger_s2lite | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1309 | 0.298 | 0.298 | 0.534 | – |
| randscore_route_tau0.6 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 943 | 0.268 | 0.126 | 0.507 | – |
| rate10_reflect_trigger_s2lite | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1310 | 0.106 | 0.107 | 0.550 | – |
| randscore_route_tau0.8 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1207 | 0.384 | 0.222 | 0.510 | – |
| heuristic_stop_guard_s2lite | 0 | – | – | – | – | – | – | – | – | – | 1.000 | 0.000 | 0.0 | 200 | 0.125 | 0.132 | 0.500 | – |
| rate30_route | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1041 | 0.025 | 0.026 | 0.500 | – |
| lite_ts_assist_k3 | 1514 | 0.853 | 0.135 | 0.135 | 0.140 | 0.651 | 0.100 | 0.088 | 0.879 | – | – | – | – | 0 | – | – | – | – |
| ratem_reflect_trigger_s2lite | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 3293 | 0.163 | 0.163 | 0.494 | – |
| jev_route_tau0.5 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1434 | 0.187 | 0.064 | 0.490 | – |
| jev_stop_guard_s2lite | 0 | – | – | – | – | – | – | – | – | – | 0.259 | 0.142 | 3.7 | 530 | 0.128 | 0.159 | 0.856 | – |
| jev_route_tau0.8 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1510 | 0.203 | 0.075 | 0.498 | – |
| s2self_reflect_trigger_s2lite | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1310 | 0.016 | 0.026 | 0.491 | – |
| lite_route | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1424 | 0.027 | 0.026 | 0.213 | – |
| heuristic_route | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1509 | 0.012 | 0.022 | 0.500 | – |
| ratem_route | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1438 | 0.022 | 0.023 | 0.500 | – |
| heuristic_reflect_trigger_s2lite | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1310 | 0.017 | 0.027 | 0.499 | – |
| jev_reflect_trigger_s2lite | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 3306 | 0.206 | 0.102 | 0.631 | – |
| ratem_stop_guard_s2lite | 0 | – | – | – | – | – | – | – | – | – | 0.738 | 0.324 | 4.8 | 287 | 0.379 | 0.379 | 0.475 | – |
| jev_route_tau0.4 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1362 | 0.174 | 0.059 | 0.563 | – |
| rate10_route | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1339 | 0.024 | 0.025 | 0.500 | – |
| s2self_route | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1489 | 0.029 | 0.024 | 0.473 | – |
| jev_route_tau0.6 | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1508 | 0.200 | 0.076 | 0.481 | – |
| jev_ts_replace | 1376 | 0.955 | 0.031 | 0.038 | 0.041 | 0.867 | 0.008 | 0.007 | 0.988 | 2.7 | – | – | – | 0 | – | – | – | – |
| jev_ts_params_replace_answercopy | 3381 | 0.944 | 0.033 | 0.034 | 0.045 | 0.888 | 0.009 | 0.007 | 0.979 | 3.4 | – | – | – | 0 | – | – | – | – |
| rate10_stop_trigger | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 955 | 0.104 | 0.103 | – | – |
| heuristic_stop_trigger | 0 | – | – | – | – | – | – | – | – | – | – | – | – | 1009 | 0.107 | 0.097 | – | – |
| lite_ts_replace | 1133 | 0.827 | 0.163 | 0.162 | 0.169 | 0.588 | 0.138 | 0.122 | 0.839 | 16.2 | – | – | – | 0 | – | – | – | – |
| tfidf_ts_assist_k3 | 1408 | 0.156 | 0.579 | 0.579 | 0.509 | 0.446 | 0.887 | 0.332 | 0.161 | – | – | – | – | 0 | – | – | – | – |
| heuristic_ts_assist_k3 | 1756 | 0.106 | 0.884 | 0.884 | 0.876 | 0.442 | 0.915 | 0.259 | 0.109 | – | – | – | – | 0 | – | – | – | – |
| embed_ts_assist_k3 | 1005 | 0.125 | 0.153 | 0.136 | 0.120 | 0.650 | 0.781 | 0.166 | 0.137 | – | – | – | – | 0 | – | – | – | – |
| tfidf_ts_assist_k5 | 2028 | 0.090 | 0.764 | 0.763 | 0.724 | 0.242 | 0.970 | 0.277 | 0.077 | – | – | – | – | 0 | – | – | – | – |
| bm25task_ts_assist_k3 | 1676 | 0.081 | 0.722 | 0.722 | 0.625 | 0.703 | 0.875 | 0.160 | 0.083 | – | – | – | – | 0 | – | – | – | – |
| bm25_ts_assist_k3 | 3717 | 0.122 | 0.808 | 0.808 | 0.767 | 0.451 | 0.921 | 0.299 | 0.139 | – | – | – | – | 0 | – | – | – | – |
| bm25_ts_assist_k5 | 1452 | 0.116 | 0.811 | 0.811 | 0.779 | 0.405 | 0.930 | 0.296 | 0.109 | – | – | – | – | 0 | – | – | – | – |
| heuristic_ts_replace | 425 | 0.353 | 0.637 | 0.637 | 0.634 | 0.517 | 0.625 | 0.345 | 0.368 | 64.7 | – | – | – | 0 | – | – | – | – |
| bm25_ts_replace | 1689 | 0.084 | 0.801 | 0.801 | 0.728 | 0.440 | 0.944 | 0.236 | 0.078 | 50.1 | – | – | – | 0 | – | – | – | – |
| tfidf_ts_replace | 1416 | 0.100 | 0.760 | 0.747 | 0.669 | 0.434 | 0.887 | 0.216 | 0.066 | 40.6 | – | – | – | 0 | – | – | – | – |
| random_ts_assist_k3 | 2640 | 0.022 | 0.007 | 0.007 | 0.022 | 0.524 | 0.977 | 0.082 | 0.022 | – | – | – | – | 0 | – | – | – | – |
| random_ts_replace | 666 | 0.029 | 0.014 | 0.014 | 0.028 | 0.590 | 0.962 | 0.091 | 0.032 | 97.1 | – | – | – | 0 | – | – | – | – |
| embed_ts_replace | 792 | 0.042 | 0.432 | 0.432 | 0.231 | 0.690 | 0.903 | 0.077 | 0.041 | 95.8 | – | – | – | 0 | – | – | – | – |
| prior_recent_ts_replace | 3391 | 0.000 | 0.990 | 0.990 | 0.980 | 0.500 | 1.000 | 0.002 | 0.000 | 100.0 | – | – | – | 0 | – | – | – | – |

_Calibration over System-1 Choice decisions with gold labels (top-label confidence); yes/no: guard / trigger decisions calibrated as forecasts P(yes) of the binary gold; harmful: delegated decisions that were wrong in failed episodes (per delegated decision); FAR/FRR: guard false accept / false reject rates; rescue: wrong LLM proposals caught by the guard in episodes that succeeded._

### Delegation coverage and native LLM calls (A6.22 (a))

| arm | tool cov. % | arg cov. % (S1 / slots) | arg cov. in S1 calls % | LLM-free steps % | LLM calls / ep. | calls avoided / ep. | avoided % | reference |
|---|---|---|---|---|---|---|---|---|
| jev_ts_replace | 100.0 | 0.0 (0/2262) | – | 0.0 | 6.88 | +0.62 | 8.3 | llm_only |
| jev_ts_params_replace_answercopy | 100.0 | 84.9 (2156/2538) | 84.9 | 74.9 | 1.91 | +5.59 | 74.5 | llm_only |
| lite_ts_replace | 100.0 | 0.0 (0/1854) | – | 0.0 | 5.67 | +1.83 | 24.5 | llm_only |
| heuristic_ts_replace | 100.0 | 0.0 (0/839) | – | 0.0 | 2.12 | +5.38 | 71.7 | llm_only |
| bm25_ts_replace | 100.0 | 0.0 (0/3001) | – | 0.0 | 8.73 | -1.23 | -16.5 | llm_only |
| tfidf_ts_replace | 100.0 | 0.0 (0/2446) | – | 0.0 | 7.24 | +0.26 | 3.5 | llm_only |
| random_ts_replace | 100.0 | 0.0 (0/979) | – | 0.0 | 3.33 | +4.17 | 55.6 | llm_only |
| embed_ts_replace | 100.0 | 0.0 (0/1741) | – | 0.0 | 4.29 | +3.21 | 42.9 | llm_only |
| prior_recent_ts_replace | 100.0 | 0.0 (0/5162) | – | 0.0 | 21.59 | -14.09 | -187.9 | llm_only |

_tool cov.: tool choices executed from System-1 per tool-selection opportunity; arg cov.: argument values System-1 chose per argument slot of the executed calls (a deferred tool choice leaves the whole call, and its slots, to the LLM); in S1 calls: the same share inside System-1-chosen calls (– when it chose none: the argument thresholds were never reached); calls avoided: paired per-task mean of the reference's minus the arm's System-2 calls (trajectory length included)._

### Meta-decision golds: coverage

| arm | dp/mode | n | labelled | coverage | doomed | unobserved | no_oracle | cf_error |
|---|---|---|---|---|---|---|---|---|
| llm_only | tool_select/off | 1500 | 1500 | 100.0% | 0 | 0 | 0 | 0 |
| heuristic_stop_guard | tool_select/off | 1501 | 1501 | 100.0% | 0 | 0 | 0 | 0 |
| heuristic_stop_guard | stop/guard | 200 | 200 | 100.0% | 0 | 0 | 0 | 0 |
| jev_stop_trigger | tool_select/off | 1500 | 1500 | 100.0% | 0 | 0 | 0 | 0 |
| jev_stop_trigger | stop/trigger | 1300 | 1300 | 100.0% | 0 | 0 | 0 | 0 |
| heuristic_reflect_trigger | tool_select/off | 1501 | 1501 | 100.0% | 0 | 0 | 0 | 0 |
| heuristic_reflect_trigger | reflection/trigger | 1301 | 1300 | 99.9% | 0 | 1 | 0 | 0 |
| jev_stop_guard | tool_select/off | 1525 | 1525 | 100.0% | 0 | 0 | 0 | 0 |
| jev_stop_guard | stop/guard | 224 | 224 | 100.0% | 0 | 0 | 0 | 0 |
| rate10_reflect_trigger | tool_select/off | 1506 | 1506 | 100.0% | 0 | 0 | 0 | 0 |
| rate10_reflect_trigger | reflection/trigger | 1306 | 1193 | 91.3% | 0 | 113 | 0 | 0 |
| jev_reflect_trigger | tool_select/off | 1506 | 1506 | 100.0% | 0 | 0 | 0 | 0 |
| jev_reflect_trigger | reflection/trigger | 1306 | 1122 | 85.9% | 0 | 184 | 0 | 0 |
| rate50_stop_guard | tool_select/off | 1685 | 1685 | 100.0% | 0 | 0 | 0 | 0 |
| rate50_stop_guard | stop/guard | 385 | 385 | 100.0% | 0 | 0 | 0 | 0 |
| s2self_reflect_trigger | tool_select/off | 1503 | 1503 | 100.0% | 0 | 0 | 0 | 0 |
| s2self_reflect_trigger | reflection/trigger | 1303 | 1272 | 97.6% | 0 | 31 | 0 | 0 |
| rate30_reflect_trigger | tool_select/off | 1516 | 1516 | 100.0% | 0 | 0 | 0 | 0 |
| rate30_reflect_trigger | reflection/trigger | 1316 | 918 | 69.8% | 0 | 398 | 0 | 0 |
| jev_memw | tool_select/off | 1510 | 1510 | 100.0% | 0 | 0 | 0 | 0 |
| jev_memw | memory_write/trigger | 1312 | 1310 | 99.8% | 2 | 0 | 0 | 0 |
| rate50_reflect_trigger | tool_select/off | 1526 | 1526 | 100.0% | 0 | 0 | 0 | 0 |
| rate50_reflect_trigger | reflection/trigger | 1326 | 669 | 50.5% | 0 | 657 | 0 | 0 |
| ratem_memw | tool_select/off | 1526 | 1526 | 100.0% | 0 | 0 | 0 | 0 |
| ratem_memw | memory_write/trigger | 1329 | 1326 | 99.8% | 3 | 0 | 0 | 0 |
| heuristic_memw | tool_select/off | 1528 | 1528 | 100.0% | 0 | 0 | 0 | 0 |
| heuristic_memw | memory_write/trigger | 1333 | 1328 | 99.6% | 5 | 0 | 0 | 0 |
| reflect_always | tool_select/off | 1562 | 1562 | 100.0% | 0 | 0 | 0 | 0 |
| randscore_route_tau0.2 | routing/trigger | 1505 | 323 | 21.5% | 0 | 1182 | 0 | 0 |
| randscore_route_tau0.2 | tool_select/off | 1505 | 1505 | 100.0% | 0 | 0 | 0 | 0 |
| jev_route_tau0.2 | routing/trigger | 1508 | 612 | 40.6% | 0 | 896 | 0 | 0 |
| jev_route_tau0.2 | tool_select/off | 1508 | 1508 | 100.0% | 0 | 0 | 0 | 0 |
| s2self_ts_assist_k3 | tool_select/assist | 1574 | 1574 | 100.0% | 0 | 0 | 0 | 0 |
| randscore_route_tau0.4 | routing/trigger | 1505 | 623 | 41.4% | 0 | 882 | 0 | 0 |
| randscore_route_tau0.4 | tool_select/off | 1505 | 1505 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_assist_k5 | tool_select/assist | 1688 | 1688 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_assist_k3 | tool_select/assist | 1619 | 1619 | 100.0% | 0 | 0 | 0 | 0 |
| rate50_reflect_trigger_s2lite | tool_select/off | 1511 | 1511 | 100.0% | 0 | 0 | 0 | 0 |
| rate50_reflect_trigger_s2lite | reflection/trigger | 1311 | 1311 | 100.0% | 0 | 0 | 0 | 0 |
| rate50_route | routing/trigger | 1504 | 733 | 48.7% | 0 | 771 | 0 | 0 |
| rate50_route | tool_select/off | 1504 | 1504 | 100.0% | 0 | 0 | 0 | 0 |
| rate30_reflect_trigger_s2lite | tool_select/off | 1509 | 1509 | 100.0% | 0 | 0 | 0 | 0 |
| rate30_reflect_trigger_s2lite | reflection/trigger | 1309 | 1309 | 100.0% | 0 | 0 | 0 | 0 |
| randscore_route_tau0.6 | routing/trigger | 1509 | 943 | 62.5% | 0 | 566 | 0 | 0 |
| randscore_route_tau0.6 | tool_select/off | 1509 | 1509 | 100.0% | 0 | 0 | 0 | 0 |
| rate10_reflect_trigger_s2lite | tool_select/off | 1510 | 1510 | 100.0% | 0 | 0 | 0 | 0 |
| rate10_reflect_trigger_s2lite | reflection/trigger | 1310 | 1310 | 100.0% | 0 | 0 | 0 | 0 |
| randscore_route_tau0.8 | routing/trigger | 1507 | 1207 | 80.1% | 0 | 300 | 0 | 0 |
| randscore_route_tau0.8 | tool_select/off | 1507 | 1507 | 100.0% | 0 | 0 | 0 | 0 |
| heuristic_stop_guard_s2lite | tool_select/off | 1511 | 1511 | 100.0% | 0 | 0 | 0 | 0 |
| heuristic_stop_guard_s2lite | stop/guard | 200 | 200 | 100.0% | 0 | 0 | 0 | 0 |
| rate30_route | routing/trigger | 1508 | 1041 | 69.0% | 0 | 467 | 0 | 0 |
| rate30_route | tool_select/off | 1508 | 1508 | 100.0% | 0 | 0 | 0 | 0 |
| lite_ts_assist_k3 | tool_select/assist | 1514 | 1514 | 100.0% | 0 | 0 | 0 | 0 |
| ratem_reflect_trigger_s2lite | tool_select/off | 3793 | 3793 | 100.0% | 0 | 0 | 0 | 0 |
| ratem_reflect_trigger_s2lite | reflection/trigger | 3293 | 3293 | 100.0% | 0 | 0 | 0 | 0 |
| jev_route_tau0.5 | routing/trigger | 1510 | 1434 | 95.0% | 0 | 76 | 0 | 0 |
| jev_route_tau0.5 | tool_select/off | 1510 | 1510 | 100.0% | 0 | 0 | 0 | 0 |
| jev_stop_guard_s2lite | tool_select/off | 1846 | 1846 | 100.0% | 0 | 0 | 0 | 0 |
| jev_stop_guard_s2lite | stop/guard | 530 | 530 | 100.0% | 0 | 0 | 0 | 0 |
| jev_route_tau0.8 | routing/trigger | 1510 | 1510 | 100.0% | 0 | 0 | 0 | 0 |
| jev_route_tau0.8 | tool_select/off | 1510 | 1510 | 100.0% | 0 | 0 | 0 | 0 |
| s2self_reflect_trigger_s2lite | tool_select/off | 1510 | 1510 | 100.0% | 0 | 0 | 0 | 0 |
| s2self_reflect_trigger_s2lite | reflection/trigger | 1310 | 1310 | 100.0% | 0 | 0 | 0 | 0 |
| lite_route | routing/trigger | 1511 | 1424 | 94.2% | 0 | 87 | 0 | 0 |
| lite_route | tool_select/off | 1511 | 1511 | 100.0% | 0 | 0 | 0 | 0 |
| heuristic_route | routing/trigger | 1513 | 1509 | 99.7% | 0 | 4 | 0 | 0 |
| heuristic_route | tool_select/off | 1513 | 1513 | 100.0% | 0 | 0 | 0 | 0 |
| ratem_route | routing/trigger | 1510 | 1438 | 95.2% | 0 | 72 | 0 | 0 |
| ratem_route | tool_select/off | 1510 | 1510 | 100.0% | 0 | 0 | 0 | 0 |
| heuristic_reflect_trigger_s2lite | tool_select/off | 1510 | 1510 | 100.0% | 0 | 0 | 0 | 0 |
| heuristic_reflect_trigger_s2lite | reflection/trigger | 1310 | 1310 | 100.0% | 0 | 0 | 0 | 0 |
| jev_reflect_trigger_s2lite | tool_select/off | 3810 | 3806 | 99.9% | 4 | 0 | 0 | 0 |
| jev_reflect_trigger_s2lite | reflection/trigger | 3310 | 3306 | 99.9% | 4 | 0 | 0 | 0 |
| ratem_stop_guard_s2lite | tool_select/off | 1603 | 1603 | 100.0% | 0 | 0 | 0 | 0 |
| ratem_stop_guard_s2lite | stop/guard | 287 | 287 | 100.0% | 0 | 0 | 0 | 0 |
| jev_route_tau0.4 | routing/trigger | 1509 | 1362 | 90.3% | 0 | 147 | 0 | 0 |
| jev_route_tau0.4 | tool_select/off | 1509 | 1509 | 100.0% | 0 | 0 | 0 | 0 |
| rate10_route | routing/trigger | 1510 | 1339 | 88.7% | 0 | 171 | 0 | 0 |
| rate10_route | tool_select/off | 1510 | 1510 | 100.0% | 0 | 0 | 0 | 0 |
| s2self_route | routing/trigger | 1509 | 1489 | 98.7% | 0 | 20 | 0 | 0 |
| s2self_route | tool_select/off | 1509 | 1509 | 100.0% | 0 | 0 | 0 | 0 |
| jev_route_tau0.6 | routing/trigger | 1510 | 1508 | 99.9% | 0 | 2 | 0 | 0 |
| jev_route_tau0.6 | tool_select/off | 1510 | 1510 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_replace | tool_select/replace | 1376 | 1376 | 100.0% | 0 | 0 | 0 | 0 |
| llm_only_s2lite | tool_select/off | 3783 | 3783 | 100.0% | 0 | 0 | 0 | 0 |
| jev_ts_params_replace_answercopy | tool_select/replace | 1522 | 1504 | 98.8% | 18 | 0 | 0 | 0 |
| jev_ts_params_replace_answercopy | tool_params/replace | 1877 | 1877 | 100.0% | 0 | 0 | 0 | 0 |
| rate10_stop_trigger | tool_select/off | 1056 | 1056 | 100.0% | 0 | 0 | 0 | 0 |
| rate10_stop_trigger | stop/trigger | 955 | 955 | 100.0% | 0 | 0 | 0 | 0 |
| heuristic_stop_trigger | tool_select/off | 1109 | 1109 | 100.0% | 0 | 0 | 0 | 0 |
| heuristic_stop_trigger | stop/trigger | 1009 | 1009 | 100.0% | 0 | 0 | 0 | 0 |
| lite_ts_replace | tool_select/replace | 1133 | 1133 | 100.0% | 0 | 0 | 0 | 0 |
| tfidf_ts_assist_k3 | tool_select/assist | 1462 | 1408 | 96.3% | 54 | 0 | 0 | 0 |
| heuristic_ts_assist_k3 | tool_select/assist | 1870 | 1756 | 93.9% | 114 | 0 | 0 | 0 |
| embed_ts_assist_k3 | tool_select/assist | 1005 | 1005 | 100.0% | 0 | 0 | 0 | 0 |
| tfidf_ts_assist_k5 | tool_select/assist | 2192 | 2028 | 92.5% | 164 | 0 | 0 | 0 |
| bm25task_ts_assist_k3 | tool_select/assist | 1757 | 1676 | 95.4% | 81 | 0 | 0 | 0 |
| bm25_ts_assist_k3 | tool_select/assist | 3908 | 3717 | 95.1% | 191 | 0 | 0 | 0 |
| bm25_ts_assist_k5 | tool_select/assist | 1496 | 1452 | 97.1% | 44 | 0 | 0 | 0 |
| heuristic_ts_replace | tool_select/replace | 425 | 425 | 100.0% | 0 | 0 | 0 | 0 |
| bm25_ts_replace | tool_select/replace | 1747 | 1689 | 96.7% | 58 | 0 | 0 | 0 |
| tfidf_ts_replace | tool_select/replace | 1448 | 1416 | 97.8% | 32 | 0 | 0 | 0 |
| random_ts_assist_k3 | tool_select/assist | 2969 | 2640 | 88.9% | 329 | 0 | 0 | 0 |
| random_ts_replace | tool_select/replace | 666 | 666 | 100.0% | 0 | 0 | 0 | 0 |
| embed_ts_replace | tool_select/replace | 857 | 792 | 92.4% | 65 | 0 | 0 | 0 |
| prior_recent_ts_replace | tool_select/replace | 4318 | 3391 | 78.5% | 927 | 0 | 0 | 0 |

### Live gates vs. their gold (consistency check; not H2 inputs)

| arm | gate/mode | opportunities (retries) | labelled share | n+ / n− labelled | AUROC (observed labels) | J(τ) | realised fire rate |
|---|---|---|---|---|---|---|---|
| heuristic_stop_guard | stop_guard/guard | 200 (0) | 100.0% | 0 / 200 | – | – | 0.000 |
| jev_stop_trigger | stop_trigger/trigger | 1300 (0) | 100.0% | 0 / 1300 | – | – | 0.000 |
| heuristic_reflect_trigger | reflection/trigger | 1301 (0) | 99.9% | 0 / 1300 | – | selective | 0.001 |
| jev_stop_guard | stop_guard/guard | 221 (3) | 100.0% | 0 / 221 | – | – | 0.104 |
| rate10_reflect_trigger | reflection/trigger | 1306 (0) | 91.3% | 0 / 1193 | – | selective | 0.087 |
| jev_reflect_trigger | reflection/trigger | 1306 (0) | 85.9% | 0 / 1122 | – | selective | 0.141 |
| rate50_stop_guard | stop_guard/guard | 304 (81) | 100.0% | 0 / 304 | – | – | 0.480 |
| s2self_reflect_trigger | reflection/trigger | 1303 (0) | 97.6% | 0 / 1272 | – | selective | 0.024 |
| rate30_reflect_trigger | reflection/trigger | 1316 (0) | 69.8% | 2 / 916 | 0.500 | selective | 0.302 |
| jev_memw | memory_write/trigger | 1312 (0) | 99.8% | 900 / 410 | 0.889 | 0.017 | 0.995 |
| rate50_reflect_trigger | reflection/trigger | 1326 (0) | 50.5% | 2 / 667 | 0.500 | selective | 0.495 |
| ratem_memw | memory_write/trigger | 1329 (0) | 99.8% | 900 / 426 | 0.497 | -0.005 | 0.994 |
| heuristic_memw | memory_write/trigger | 1333 (0) | 99.6% | 900 / 428 | 0.486 | -0.027 | 0.922 |
| randscore_route_tau0.2 | routing/trigger | 1505 (0) | 21.5% | 7 / 316 | 0.431 | selective | 0.785 |
| jev_route_tau0.2 | routing/trigger | 1508 (0) | 40.6% | 8 / 604 | 0.779 | selective | 0.594 |
| randscore_route_tau0.4 | routing/trigger | 1505 (0) | 41.4% | 12 / 611 | 0.522 | selective | 0.586 |
| rate50_reflect_trigger_s2lite | reflection/trigger | 1311 (0) | 100.0% | 9 / 645 | 0.500 | selective | 0.501 |
| rate50_route | routing/trigger | 1504 (0) | 48.7% | 22 / 711 | 0.500 | selective | 0.513 |
| rate30_reflect_trigger_s2lite | reflection/trigger | 1309 (0) | 100.0% | 16 / 909 | 0.500 | selective | 0.293 |
| randscore_route_tau0.6 | routing/trigger | 1509 (0) | 62.5% | 24 / 919 | 0.507 | selective | 0.375 |
| rate10_reflect_trigger_s2lite | reflection/trigger | 1310 (0) | 100.0% | 21 / 1165 | 0.500 | selective | 0.095 |
| randscore_route_tau0.8 | routing/trigger | 1507 (0) | 80.1% | 26 / 1181 | 0.510 | selective | 0.199 |
| heuristic_stop_guard_s2lite | stop_guard/guard | 200 (0) | 100.0% | 27 / 173 | 0.500 | 0.000 | 0.000 |
| rate30_route | routing/trigger | 1508 (0) | 69.0% | 27 / 1014 | 0.500 | selective | 0.310 |
| ratem_reflect_trigger_s2lite | reflection/trigger | 3293 (0) | 100.0% | 64 / 2744 | 0.500 | selective | 0.147 |
| jev_route_tau0.5 | routing/trigger | 1510 (0) | 95.0% | 31 / 1403 | 0.490 | selective | 0.050 |
| jev_stop_guard_s2lite | stop_guard/guard | 251 (279) | 100.0% | 54 / 197 | 0.892 | 0.599 | 0.271 |
| jev_route_tau0.8 | routing/trigger | 1510 (0) | 100.0% | 32 / 1478 | 0.498 | selective | 0.000 |
| s2self_reflect_trigger_s2lite | reflection/trigger | 1310 (0) | 100.0% | 32 / 1276 | 0.492 | selective | 0.002 |
| lite_route | routing/trigger | 1511 (0) | 94.2% | 32 / 1392 | 0.213 | selective | 0.058 |
| heuristic_route | routing/trigger | 1513 (0) | 99.7% | 33 / 1476 | 0.500 | selective | 0.003 |
| ratem_route | routing/trigger | 1510 (0) | 95.2% | 33 / 1405 | 0.500 | selective | 0.048 |
| heuristic_reflect_trigger_s2lite | reflection/trigger | 1310 (0) | 100.0% | 34 / 1274 | 0.500 | selective | 0.002 |
| jev_reflect_trigger_s2lite | reflection/trigger | 3310 (0) | 99.9% | 81 / 2739 | 0.635 | selective | 0.148 |
| ratem_stop_guard_s2lite | stop_guard/guard | 255 (32) | 100.0% | 42 / 213 | 0.469 | -0.062 | 0.314 |
| jev_route_tau0.4 | routing/trigger | 1509 (0) | 90.3% | 34 / 1328 | 0.563 | selective | 0.097 |
| rate10_route | routing/trigger | 1510 (0) | 88.7% | 34 / 1305 | 0.500 | selective | 0.113 |
| s2self_route | routing/trigger | 1509 (0) | 98.7% | 35 / 1454 | 0.473 | selective | 0.013 |
| jev_route_tau0.6 | routing/trigger | 1510 (0) | 99.9% | 36 / 1472 | 0.481 | selective | 0.001 |
| rate10_stop_trigger | stop_trigger/trigger | 955 (0) | 100.0% | 0 / 955 | – | – | 0.104 |
| heuristic_stop_trigger | stop_trigger/trigger | 1009 (0) | 100.0% | 0 / 1009 | – | – | 0.099 |

_Live trigger arms label a reflection / routing opportunity only where the gate did not fire (selective labels): AUROC is computed on the labels observed at executed actions, J / accuracy / estimability are not reported, and these rows never enter H2 (META_GOLD_SPEC §9.2, §11.5–6). The realised fire rate counts every opportunity once (a stop-guard resubmission of a rejected answer is a retry)._

### Counterfactual efficacy (§4)

| arm | dp | n | n+ / n− | e | h | placebo δ | final share | CF coverage |
|---|---|---|---|---|---|---|---|---|

### Tool-selection recall@k vs. the acceptable tool set (C3)

| arm | n | recall@1 | recall@3 | recall@5 (undefined) | recall@3 (canonical) | offered recall (offered k) |
|---|---|---|---|---|---|---|
| s2self_ts_assist_k3 | 1574 | 0.915 | 0.928 | 0.936 (0) | 0.860 | 0.956 (3.9) |
| jev_ts_assist_k5 | 1688 | 0.867 | 0.941 | 0.964 (0) | 0.915 | 0.969 (5.8) |
| jev_ts_assist_k3 | 1619 | 0.893 | 0.941 | 0.964 (0) | 0.914 | 0.965 (3.8) |
| lite_ts_assist_k3 | 1514 | 0.853 | 0.880 | 0.884 (0) | 0.793 | 0.882 (3.7) |
| jev_ts_replace | 1376 | 0.955 | 0.991 | 0.993 (0) | 0.946 | – (–) |
| jev_ts_params_replace_answercopy | 1504 | 0.897 | 0.985 | 0.989 (0) | 0.945 | – (–) |
| lite_ts_replace | 1133 | 0.827 | 0.859 | 0.862 (0) | 0.780 | – (–) |
| tfidf_ts_assist_k3 | 1408 | 0.156 | 0.288 | 0.462 (0) | 0.159 | 0.335 (4.0) |
| heuristic_ts_assist_k3 | 1756 | 0.106 | 0.137 | 0.179 (0) | 0.094 | 0.137 (4.0) |
| embed_ts_assist_k3 | 1005 | 0.125 | 0.248 | 0.336 (0) | 0.208 | 0.303 (4.0) |
| tfidf_ts_assist_k5 | 2028 | 0.090 | 0.219 | 0.256 (0) | 0.069 | 0.281 (6.0) |
| bm25task_ts_assist_k3 | 1676 | 0.081 | 0.107 | 0.150 (0) | 0.071 | 0.137 (4.0) |
| bm25_ts_assist_k3 | 3717 | 0.122 | 0.172 | 0.229 (0) | 0.099 | 0.197 (4.0) |
| bm25_ts_assist_k5 | 1452 | 0.116 | 0.146 | 0.187 (0) | 0.084 | 0.213 (6.0) |
| heuristic_ts_replace | 425 | 0.353 | 0.412 | 0.438 (0) | 0.235 | – (–) |
| bm25_ts_replace | 1689 | 0.084 | 0.108 | 0.109 (0) | 0.079 | – (–) |
| tfidf_ts_replace | 1416 | 0.100 | 0.106 | 0.106 (0) | 0.071 | – (–) |
| random_ts_assist_k3 | 2640 | 0.022 | 0.065 | 0.106 (0) | 0.044 | 0.069 (4.0) |
| random_ts_replace | 666 | 0.029 | 0.075 | 0.110 (0) | 0.057 | – (–) |
| embed_ts_replace | 792 | 0.042 | 0.110 | 0.186 (0) | 0.091 | – (–) |
| prior_recent_ts_replace | 3391 | 0.000 | 0.036 | 0.102 (0) | 0.035 | – (–) |

_recall@k: the top-k of System-1's full ranking (tournaments log it as `s1_rank`: finalists, then each round's eliminated options by score); records whose ranking is incomplete at k (older tournament logs with only their finalists) are undefined and left out (count in brackets). Offered k = mean number of tools an assist arm offered System-2 (top-k + finish)._

### Tool arguments by fill kind (H3; §7.4)

| arm | kind | n | delegated | delegation | harmful |
|---|---|---|---|---|---|
| jev_ts_params_replace_answercopy | bounded | 986 | 986 | 100.0% | 4.1% |
| jev_ts_params_replace_answercopy | copy | 1328 | 1170 | 88.1% | 0.1% |
| jev_ts_params_replace_answercopy | free | 224 | 0 | 0.0% | – |

### Sweeps

| sweep | arms | values | APGR | CPT(50%) | CPT(80%) | AIQ | AIQ zero-router | ΔAIQ |
|---|---|---|---|---|---|---|---|---|
| jev_route_tau | 5 | 0.2, 0.4, 0.5, 0.6, 0.8 | – | – | – | 0.899 | – | – |
| randscore_route_tau | 4 | 0.2, 0.4, 0.6, 0.8 | – | – | – | 0.851 | – | – |

_APGR / CPT: x = strong-model calls relative to the baseline, anchors weak (PGR 0) and baseline (PGR 1); AIQ: mean success of the non-decreasing convex hull over the linear cost range between the two references (sweep + references); zero-router: hull of the references alone._

### Paired comparisons (arm vs. control)

| arm | control | n | Δ success pp | 95% CI | p_better | McNemar p | TOST ±2 pt (p, discordant) | Δ cost $ | Δ lat s | ΔU (W=0.1) [CI], p | ΔU (W=1) [CI], p | ΔU (W=10) [CI], p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_reflect_trigger_s2lite | ratem_reflect_trigger_s2lite | 500 | -2.4 | [-5.8, 0.8] | 0.934 | 0.189 | not shown (0.623, 70 discordant) | -0.0004 | +1.48 | -0.0020 [-0.0054, +0.0013], 0.890 | -0.0236 [-0.0574, +0.0085], 0.918 | -0.2396 [-0.5794, +0.0805], 0.918 |
| jev_ts_assist_k3 | bm25_ts_assist_k3 | 200 | +73.5 | [65.0, 82.0] | <0.001 | <0.001 | not shown (1.000, 179 discordant) | +0.0042 | +2.97 | +0.0693 [+0.0597, +0.0792], <0.001 | +0.7308 [+0.6441, +0.8151], <0.001 | +7.3458 [+6.4941, +8.1943], <0.001 |
| jev_route_tau0.5 | ratem_route | 200 | +1.0 | [-2.5, 4.5] | 0.344 | 0.791 | not shown (0.336, 14 discordant) | +0.0002 | +1.92 | +0.0008 [-0.0028, +0.0046], 0.329 | +0.0098 [-0.0253, +0.0452], 0.319 | +0.0998 [-0.2503, +0.4502], 0.319 |
| jev_memw | ratem_memw | 200 | +0.5 | [0.0, 1.5] | 0.372 | 1.000 | not shown (–, 1 discordant) | +0.0013 | +0.59 | -0.0008 [-0.0022, +0.0016], 0.779 | +0.0037 [-0.0022, +0.0151], 0.372 | +0.0487 [-0.0022, +0.1501], 0.372 |
| jev_stop_guard_s2lite | ratem_stop_guard_s2lite | 200 | +1.5 | [-2.0, 5.0] | 0.234 | 0.581 | not shown (0.443, 13 discordant) | +0.0018 | +1.37 | -0.0003 [-0.0041, +0.0036], 0.549 | +0.0132 [-0.0221, +0.0488], 0.234 | +0.1482 [-0.2021, +0.4988], 0.234 |

_Paired task bootstrap of arm − control; p_better one-sided (small = the arm is better); TOST: equivalence within ±2 pt of success (90% paired-bootstrap CI inside the margin; the pre-registered H2 test of a System-1 gate against its matched random gate); U = W·success − cost − λ·latency (λ = 0.0 USD/h)._

### Gate VoI: predicted vs. observed (W = 1; docs/META_GOLD_SPEC.md §9.1)

| pair | gate | meas. arm | System-1 answers | n+/n− | π | AUROC | J(τ) | e | h | n̄ | rate live/ctrl | pred. ΔU gross | obs. ΔU gross [CI] | estimable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_reflect_trigger_s2lite vs ratem_reflect_trigger_s2lite | reflection | meta_shadow_s2lite | measurement arm log | 91/3195 | 0.028 | 0.413 | -0.123 | 0.330 | 0.002 | 6.62 | 0.148/0.147 | -0.0072 | -0.0233 [-0.0571, +0.0088] | yes |
| jev_route_tau0.5 vs ratem_route | routing | meta_shadow_s2lite | measurement arm log | 91/3695 | 0.024 | 0.470 | -0.054 | 0.824 | 0.000 | 7.55 | 0.050/0.048 | -0.0079 | +0.0102 [-0.0249, +0.0455] | yes |
| jev_memw vs ratem_memw | memory_write | meta_shadow | measurement arm log | 900/401 | 0.692 | 0.820 | 0.000 | – | – | 6.56 | 0.995/0.994 | – | +0.0043 [-0.0017, +0.0156] | yes |
| jev_stop_guard_s2lite vs ratem_stop_guard_s2lite | stop_guard | meta_shadow_s2lite | measurement arm log | 91/409 | 0.182 | 0.975 | 0.581 | 0.011 | 0.049 | 1.25 | 0.271/0.314 | +0.0065 | +0.0133 [-0.0220, +0.0489] | yes |

_π, AUROC, J: the LIVE arm's System-1 on the measurement arm's shadow golds at the live τ -- the measurement arm's own log when it ran the same backend spec, else that spec's replay (`--replay`); pairs with neither get no prediction (see the notes). e, h: counterfactual efficacy net of the placebo; predicted ΔU = n̄·W·(e+h)·π(1−π)·J (gross of System-1's price); observed: paired task bootstrap of the gross utility difference._

