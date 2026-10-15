<!-- SENSITIVITY, not the registered readout: h5_summary.py on the run as is, except that the two predictions whose swebench evaluation timed out on the loaded box and resolved on an idle-box re-evaluation (action_guard and route, scikit-learn__scikit-learn-14710; results/hard/swe/reeval/) count as resolved. sympy__sympy-19040 (action_guard) timed out again and stays unresolved. -->
## Arms (k = 1; time = agent-run wall clock)

| arm | n | resolved | success | unevaluated | errors | time_p50_s | time_p90_s | s2_cost_mean | s1_cost_mean | cost_total | cost_per_resolved | steps_mean | solved_by_600s | solved_by_1800s | solved_by_3600s | exit_statuses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| action_guard | 100 | 77 | 0.7700 | 0 | 0 | 132.5 | 425.4 | 0.5492 | 0.0046 | 55.4 | 0.7193 | 58.5 | 0.7600 | 0.7700 | 0.7700 | {"Submitted": 100} |
| baseline | 100 | 75 | 0.7500 | 0 | 0 | 110.9 | 435.4 | 0.5628 | 0.0000 | 56.3 | 0.7505 | 60.6 | 0.7400 | 0.7500 | 0.7500 | {"Submitted": 94, "LimitsExceeded": 6} |
| route | 100 | 78 | 0.7800 | 0 | 0 | 116.3 | 326.0 | 0.4594 | 0.0041 | 46.4 | 0.5943 | 54.1 | 0.7500 | 0.7800 | 0.7800 | {"Submitted": 100} |
| stop_guard | 100 | 74 | 0.7400 | 0 | 0 | 124.8 | 555.8 | 0.6192 | 0.0001 | 61.9 | 0.8369 | 63.5 | 0.7300 | 0.7400 | 0.7400 | {"Submitted": 93, "LimitsExceeded": 7} |

## H5 vs baseline (A5.2 / A6.6 / A6.16 (j), labels A6.22 (d) 1; exploratory, outside the Holm family). success_preserving = the exploratory success screen (LB90 of d_success > -0.05 and wins >= losses; not an NI test); strictly_better = LB90 > 0 and sign p_wins < 0.05; faster = median d time on instances both resolve < 0 and exact one-sided sign test p < 0.05 (wilcoxon_p descriptive). McNemar (k = 1) and d_time_median_all / solved_by_t / d_cost_median are descriptive.

| arm | view | tasks | n_errored_tasks | success_base | success_arm | d_success | d_success_lb | wins | losses | sign_p | sign_p_wins | mcnemar_p | n_both_solved | d_time_median_solved | time_sign_p | wilcoxon_p | d_time_median_all | d_cost_median | solved_by_600s | solved_by_600s_base | solved_by_1800s | solved_by_1800s_base | solved_by_3600s | solved_by_3600s_base | success_preserving | strictly_better | faster | h5_verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| action_guard | full set | 100 | 0 | 0.7500 | 0.7700 | 0.0200 | -0.0200 | 7 | 5 | 0.7744 | 0.3872 | 0.7744 | 70 | 10.6 | 0.9639 | 0.9872 | 8.8 | 0.0194 | 0.7600 | 0.7400 | 0.7700 | 0.7500 | 0.7700 | 0.7500 | True | False | False | success-preserving |
| action_guard | complete case | 100 | 0 | 0.7500 | 0.7700 | 0.0200 | -0.0200 | 7 | 5 | 0.7744 | 0.3872 | 0.7744 | 70 | 10.6 | 0.9639 | 0.9872 | 8.8 | 0.0194 | 0.7600 | 0.7400 | 0.7700 | 0.7500 | 0.7700 | 0.7500 | True | False | False | success-preserving |
| route | full set | 100 | 0 | 0.7500 | 0.7800 | 0.0300 | -0.0200 | 8 | 5 | 0.5811 | 0.2905 | 0.5811 | 70 | -0.2792 | 0.4525 | 0.5942 | -2.4 | -0.0276 | 0.7500 | 0.7400 | 0.7800 | 0.7500 | 0.7800 | 0.7500 | True | False | False | success-preserving |
| route | complete case | 100 | 0 | 0.7500 | 0.7800 | 0.0300 | -0.0200 | 8 | 5 | 0.5811 | 0.2905 | 0.5811 | 70 | -0.2792 | 0.4525 | 0.5942 | -2.4 | -0.0276 | 0.7500 | 0.7400 | 0.7800 | 0.7500 | 0.7800 | 0.7500 | True | False | False | success-preserving |
| stop_guard | full set | 100 | 0 | 0.7500 | 0.7400 | -0.0100 | -0.0600 | 6 | 7 | 1.0000 | 0.7095 | 1.0000 | 68 | 7.8 | 0.9429 | 0.9095 | 11.3 | 0.0307 | 0.7300 | 0.7400 | 0.7400 | 0.7500 | 0.7400 | 0.7500 | False | False | False | neither |
| stop_guard | complete case | 100 | 0 | 0.7500 | 0.7400 | -0.0100 | -0.0600 | 6 | 7 | 1.0000 | 0.7095 | 1.0000 | 68 | 7.8 | 0.9429 | 0.9095 | 11.3 | 0.0307 | 0.7300 | 0.7400 | 0.7400 | 0.7500 | 0.7400 | 0.7500 | False | False | False | neither |
