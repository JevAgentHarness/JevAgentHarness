# τ² external reference agreement (A7.13 (iv); descriptive)

Labels: reference-agreement labels, A6.22 (b) 1-2: tool_select = the reference next step (singleton; respond_to_user when no reference action is pending and grounded); tool_params = the reference action's value in the option-label space (OTHER outside the candidates), only when the executed tool is the reference next step. Agreement with these labels is not oracle accuracy and is not evidence of τ² success or savings. Task-weighted means (τ² task, trials and arms pooled) with task-bootstrap 95% intervals.

## Coverage

| stratum | logged | occurrences (state logged) | labelled | coverage | tasks | distinct labelled questions |
|---|---|---|---|---|---|---|
| airline_test/tool_params | 3363 | 3363 | 1709 | 0.5082 | 29 | 1639 |
| airline_test/tool_select | 14636 | 11992 | 11992 | 1.0 | 35 | 10991 |
| retail/tool_params | 3973 | 3973 | 2314 | 0.5824 | 108 | 2283 |
| retail/tool_select | 11921 | 7942 | 7942 | 1.0 | 114 | 7791 |

## Agreement

| stratum | frozen | intern | base | majority label | frozen − others |
|---|---|---|---|---|---|
| airline_test/tool_select | 56.0 [50.1, 61.6] | 46.2 [41.3, 50.8] | 54.6 [47.3, 61.6] | 65.4 (respond_to_user) | intern +9.9 [+5.0, +14.8]; base +1.5 [-3.1, +6.1] |
| airline_test/tool_params | 85.7 [77.9, 92.6] | 81.0 [71.7, 89.3] | 79.6 [69.6, 88.7] | – | intern +4.7 [+1.5, +7.9]; base +6.1 [+2.9, +9.8] |
| retail/tool_select | 52.9 [50.2, 55.6] | 40.9 [37.9, 43.9] | 37.0 [34.4, 39.5] | 50.7 (respond_to_user) | intern +12.0 [+9.1, +15.1]; base +16.0 [+13.2, +18.9] |
| retail/tool_params | 86.0 [83.1, 88.7] | 76.2 [73.1, 79.3] | 69.0 [65.1, 73.0] | – | intern +9.8 [+7.1, +12.5]; base +17.0 [+13.7, +20.4] |
| all/tool_select | 53.7 [51.2, 56.2] | 42.1 [39.5, 44.8] | 41.1 [38.3, 44.0] | 54.1 (respond_to_user) | intern +11.5 [+9.0, +14.1]; base +12.6 [+10.0, +15.2] |
| all/tool_params | 85.9 [83.2, 88.6] | 77.2 [74.1, 80.2] | 71.3 [67.4, 75.0] | – | intern +8.7 [+6.5, +11.0]; base +14.7 [+11.9, +17.6] |

## On the occurrences where Jev's live answer was logged

| stratum | n | Jev (logged) | frozen (Δ vs Jev) | intern (Δ vs Jev) | base (Δ vs Jev) |
|---|---|---|---|---|---|
| airline_test/tool_select | 11992 | 61.4 | 56.0 (-5.3 [-11.1, +0.3]) | 46.2 (-15.2 [-19.2, -11.1]) | 54.6 (-6.8 [-14.3, +0.5]) |
| airline_test/tool_params | 1709 | 87.4 | 85.7 (-1.8 [-5.8, +2.1]) | 81.0 (-6.5 [-12.5, -0.7]) | 79.6 (-7.8 [-14.5, -1.8]) |
| retail/tool_select | 7942 | 59.4 | 52.9 (-6.5 [-8.9, -4.1]) | 40.9 (-18.5 [-21.5, -15.7]) | 37.0 (-22.5 [-25.6, -19.4]) |
| retail/tool_params | 2314 | 90.7 | 86.0 (-4.7 [-7.3, -1.8]) | 76.2 (-14.6 [-17.9, -10.9]) | 69.0 (-21.7 [-25.0, -18.4]) |
| all/tool_select | 19934 | 59.9 | 53.7 (-6.2 [-8.5, -3.9]) | 42.1 (-17.8 [-20.1, -15.4]) | 41.1 (-18.8 [-21.9, -15.6]) |
| all/tool_params | 4023 | 90.0 | 85.9 (-4.1 [-6.3, -1.6]) | 77.2 (-12.8 [-15.8, -9.6]) | 71.3 (-18.8 [-21.9, -15.8]) |

Answered: frozen 22704 questions (0 errors, 0 invalid); intern 22704 questions (0 errors, 0 invalid); base 22704 questions (0 errors, 0 invalid)
