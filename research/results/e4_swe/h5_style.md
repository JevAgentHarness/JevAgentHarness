## Arms (k = 1; time = agent-run wall clock)

| arm | n | resolved | success | unevaluated | errors | time_p50_s | time_p90_s | s2_cost_mean | s1_cost_mean | cost_total | cost_per_resolved | steps_mean | solved_by_600s | solved_by_1800s | solved_by_3600s | exit_statuses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_deep | 100 | 81 | 0.8100 | 0 | 0 | 237.8 | 922.9 | 0.0411 | 0.0000 | 4.1 | 0.0508 | 40.1 | 0.6600 | 0.8000 | 0.8100 | {"Submitted": 100} |
| always_fast | 100 | 78 | 0.7800 | 0 | 0 | 186.6 | 602.7 | 0.0108 | 0.0000 | 1.1 | 0.0139 | 46.9 | 0.7400 | 0.7800 | 0.7800 | {"Submitted": 100} |
| jevagent | 100 | 81 | 0.8100 | 0 | 0 | 219.0 | 726.4 | 0.0291 | 0.0033 | 3.2 | 0.0400 | 37.4 | 0.6900 | 0.8100 | 0.8100 | {"Submitted": 100} |

## H5 vs always_deep (A5.2 / A6.6 / A6.16 (j), labels A6.22 (d) 1; exploratory, outside the Holm family). success_preserving = the exploratory success screen (LB90 of d_success > -0.05 and wins >= losses; not an NI test); strictly_better = LB90 > 0 and sign p_wins < 0.05; faster = median d time on instances both resolve < 0 and exact one-sided sign test p < 0.05 (wilcoxon_p descriptive). McNemar (k = 1) and d_time_median_all / solved_by_t / d_cost_median are descriptive.

| arm | view | tasks | n_errored_tasks | success_base | success_arm | d_success | d_success_lb | wins | losses | sign_p | sign_p_wins | mcnemar_p | n_both_solved | d_time_median_solved | time_sign_p | wilcoxon_p | d_time_median_all | d_cost_median | solved_by_600s | solved_by_600s_base | solved_by_1800s | solved_by_1800s_base | solved_by_3600s | solved_by_3600s_base | success_preserving | strictly_better | faster | h5_verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_fast | full set | 100 | 0 | 0.8100 | 0.7800 | -0.0300 | -0.0700 | 3 | 6 | 0.5078 | 0.9102 | 0.5078 | 75 | -38.6 | 0.0001 | 0.0000 | -59.3 | -0.0104 | 0.7400 | 0.6600 | 0.7800 | 0.8000 | 0.7800 | 0.8100 | False | False | True | faster |
| always_fast | complete case | 100 | 0 | 0.8100 | 0.7800 | -0.0300 | -0.0700 | 3 | 6 | 0.5078 | 0.9102 | 0.5078 | 75 | -38.6 | 0.0001 | 0.0000 | -59.3 | -0.0104 | 0.7400 | 0.6600 | 0.7800 | 0.8000 | 0.7800 | 0.8100 | False | False | True | faster |
| jevagent | full set | 100 | 0 | 0.8100 | 0.8100 | 0.0000 | -0.0300 | 3 | 3 | 1.0 | 0.6562 | 1.0 | 78 | -25.8 | 0.0084 | 0.0027 | -21.9 | -0.0014 | 0.6900 | 0.6600 | 0.8100 | 0.8000 | 0.8100 | 0.8100 | True | False | True | passes the exploratory success screen; faster among jointly solved tasks |
| jevagent | complete case | 100 | 0 | 0.8100 | 0.8100 | 0.0000 | -0.0300 | 3 | 3 | 1.0 | 0.6562 | 1.0 | 78 | -25.8 | 0.0084 | 0.0027 | -21.9 | -0.0014 | 0.6900 | 0.6600 | 0.8100 | 0.8000 | 0.8100 | 0.8100 | True | False | True | passes the exploratory success screen; faster among jointly solved tasks |
