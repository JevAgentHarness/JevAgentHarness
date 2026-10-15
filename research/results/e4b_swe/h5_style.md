## Arms (k = 1; time = agent-run wall clock)

| arm | n | resolved | success | unevaluated | errors | time_p50_s | time_p90_s | s2_cost_mean | s1_cost_mean | cost_total | cost_per_resolved | steps_mean | solved_by_600s | solved_by_1800s | solved_by_3600s | exit_statuses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_deep | 100 | 70 | 0.7000 | 0 | 0 | 906.9 | 1354.8 | 1.0 | 0.0000 | 101.6 | 1.5 | 93.6 | 0.2600 | 0.6900 | 0.7000 | {"Submitted": 76, "LimitsExceeded": 24} |
| always_fast | 50 | 40 | 0.8000 | 0 | 0 | 537.1 | 1028.2 | 0.6597 | 0.0000 | 33.0 | 0.8246 | 70.1 | 0.5000 | 0.8000 | 0.8000 | {"Submitted": 47, "LimitsExceeded": 3} |
| jevagent | 100 | 75 | 0.7500 | 0 | 0 | 720.8 | 1102.8 | 0.9376 | 0.0071 | 94.5 | 1.3 | 83.1 | 0.3500 | 0.7400 | 0.7500 | {"Submitted": 83, "LimitsExceeded": 17} |

## H5 vs always_deep (A5.2 / A6.6 / A6.16 (j), labels A6.22 (d) 1; exploratory, outside the Holm family). success_preserving = the exploratory success screen (LB90 of d_success > -0.05 and wins >= losses; not an NI test); strictly_better = LB90 > 0 and sign p_wins < 0.05; faster = median d time on instances both resolve < 0 and exact one-sided sign test p < 0.05 (wilcoxon_p descriptive). McNemar (k = 1) and d_time_median_all / solved_by_t / d_cost_median are descriptive.

| arm | view | tasks | n_errored_tasks | success_base | success_arm | d_success | d_success_lb | wins | losses | sign_p | sign_p_wins | mcnemar_p | n_both_solved | d_time_median_solved | time_sign_p | wilcoxon_p | d_time_median_all | d_cost_median | solved_by_600s | solved_by_600s_base | solved_by_1800s | solved_by_1800s_base | solved_by_3600s | solved_by_3600s_base | success_preserving | strictly_better | faster | h5_verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_fast | full set | 50 | 0 | 0.7200 | 0.8000 | 0.0800 | 0.0200 | 6 | 2 | 0.2891 | 0.1445 | 0.2891 | 34 | -332.7 | 0.0000 | 0.0000 | -271.7 | -0.3392 | 0.5000 | 0.2600 | 0.8000 | 0.7200 | 0.8000 | 0.7200 | True | False | True | passes the exploratory success screen; faster among jointly solved tasks |
| always_fast | complete case | 50 | 0 | 0.7200 | 0.8000 | 0.0800 | 0.0200 | 6 | 2 | 0.2891 | 0.1445 | 0.2891 | 34 | -332.7 | 0.0000 | 0.0000 | -271.7 | -0.3392 | 0.5000 | 0.2600 | 0.8000 | 0.7200 | 0.8000 | 0.7200 | True | False | True | passes the exploratory success screen; faster among jointly solved tasks |
| jevagent | full set | 100 | 0 | 0.7000 | 0.7500 | 0.0500 | 0.0100 | 8 | 3 | 0.2266 | 0.1133 | 0.2266 | 67 | -120.5 | 0.0007 | 0.0000 | -123.3 | -0.0051 | 0.3500 | 0.2600 | 0.7400 | 0.6900 | 0.7500 | 0.7000 | True | False | True | passes the exploratory success screen; faster among jointly solved tasks |
| jevagent | complete case | 100 | 0 | 0.7000 | 0.7500 | 0.0500 | 0.0100 | 8 | 3 | 0.2266 | 0.1133 | 0.2266 | 67 | -120.5 | 0.0007 | 0.0000 | -123.3 | -0.0051 | 0.3500 | 0.2600 | 0.7400 | 0.6900 | 0.7500 | 0.7000 | True | False | True | passes the exploratory success screen; faster among jointly solved tasks |
