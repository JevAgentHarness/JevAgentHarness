## Arms (k = 1; time = agent-run wall clock)

| arm | n | resolved | success | unevaluated | errors | time_p50_s | time_p90_s | s2_cost_mean | s1_cost_mean | cost_total | cost_per_resolved | steps_mean | solved_by_600s | solved_by_1800s | solved_by_3600s | exit_statuses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| action_guard | 2 | 1 | 0.5000 | 0 | 0 | 129.7 | 136.1 | 0.2697 | 0.0026 | 0.5446 | 0.5446 | 34.0 | 0.5000 | 0.5000 | 0.5000 | {"Submitted": 2} |
| baseline | 2 | 1 | 0.5000 | 0 | 0 | 141.2 | 165.2 | 0.2488 | 0.0000 | 0.4976 | 0.4976 | 33.5 | 0.5000 | 0.5000 | 0.5000 | {"Submitted": 2} |
| route | 2 | 1 | 0.5000 | 0 | 0 | 130.1 | 139.9 | 0.2827 | 0.0032 | 0.5718 | 0.5718 | 41.0 | 0.5000 | 0.5000 | 0.5000 | {"Submitted": 2} |
| stop_guard | 2 | 1 | 0.5000 | 0 | 0 | 207.4 | 291.7 | 0.4248 | 0.0001 | 0.8499 | 0.8499 | 46.0 | 0.5000 | 0.5000 | 0.5000 | {"Submitted": 2} |

## H5 vs baseline (A5.2 / A6.6 / A6.16 (j), labels A6.22 (d) 1; exploratory, outside the Holm family). success_preserving = the exploratory success screen (LB90 of d_success > -0.05 and wins >= losses; not an NI test); strictly_better = LB90 > 0 and sign p_wins < 0.05; faster = median d time on instances both resolve < 0 and exact one-sided sign test p < 0.05 (wilcoxon_p descriptive). McNemar (k = 1) and d_time_median_all / solved_by_t / d_cost_median are descriptive.

| arm | view | tasks | n_errored_tasks | success_base | success_arm | d_success | d_success_lb | wins | losses | sign_p | sign_p_wins | mcnemar_p | n_both_solved | d_time_median_solved | time_sign_p | wilcoxon_p | d_time_median_all | d_cost_median | solved_by_600s | solved_by_600s_base | solved_by_1800s | solved_by_1800s_base | solved_by_3600s | solved_by_3600s_base | success_preserving | strictly_better | faster | h5_verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| action_guard | full set | 2 | 0 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 1.0 | 1.0 | 1 | 26.3 | 1.0 | 1.0 | -11.6 | 0.0235 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | True | False | False | success-preserving |
| action_guard | complete case | 2 | 0 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 1.0 | 1.0 | 1 | 26.3 | 1.0 | 1.0 | -11.6 | 0.0235 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | True | False | False | success-preserving |
| route | full set | 2 | 0 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 1.0 | 1.0 | 1 | 31.0 | 1.0 | 1.0 | -11.1 | 0.0371 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | True | False | False | success-preserving |
| route | complete case | 2 | 0 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 1.0 | 1.0 | 1 | 31.0 | 1.0 | 1.0 | -11.1 | 0.0371 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | True | False | False | success-preserving |
| stop_guard | full set | 2 | 0 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 1.0 | 1.0 | 1 | -9.3 | 0.5000 | 0.5000 | 66.2 | 0.1761 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | True | False | False | success-preserving |
| stop_guard | complete case | 2 | 0 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 1.0 | 1.0 | 1 | -9.3 | 0.5000 | 0.5000 | 66.2 | 0.1761 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | True | False | False | success-preserving |
