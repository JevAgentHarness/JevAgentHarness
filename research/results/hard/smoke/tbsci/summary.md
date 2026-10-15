| job | task | reward | tests_passed | tests_total | error | llm_cost_usd | llm_input_tokens | agent_steps | agent_setup_s | agent_run_s | verifier_s | trial_s | s1_requests | blocks | stop_nudges | reflections | s1_ms_mean | s1_cost_usd_est | s1_errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | symbolic-regression | 1.0 | 2 | 2 | – | – | – | – | 0.0001 | 0.2201 | 39.0 | 77.2 | 0 | 0 | 0 | 0 | – | 0.0000 | 0 |
| oracle | variable-star-vetting | 1.0 | 6 | 6 | – | – | – | – | 0.0000 | 59.7 | 18.1 | 129.2 | 0 | 0 | 0 | 0 | – | 0.0000 | 0 |
| nop | symbolic-regression | 0.0000 | 1 | 2 | – | – | – | – | 0.0001 | 0.0004 | 18.3 | 57.2 | 0 | 0 | 0 | 0 | – | 0.0000 | 0 |
| nop | variable-star-vetting | 0.0000 | 2 | 6 | – | – | – | – | 0.0000 | 0.0001 | 18.2 | 57.0 | 0 | 0 | 0 | 0 | – | 0.0000 | 0 |
| baseline | symbolic-regression | 0.0000 | 1 | 2 | – | 0.4467 | 596485 | 47 | 21.0 | 1005.6 | 24.3 | 1083.1 | 0 | 0 | 0 | 0 | – | 0.0000 | 0 |
| baseline | variable-star-vetting | 0.0000 | 2 | 6 | – | 1.2 | 3058970 | 100 | 12.1 | 1274.0 | 21.8 | 1347.8 | 0 | 0 | 0 | 0 | – | 0.0000 | 0 |
| jev_stop_reflect | symbolic-regression | 0.0000 | 1 | 2 | – | 1.2 | 2642962 | 83 | 21.3 | 1676.3 | 23.0 | 1739.6 | 10 | 0 | 2 | 7 | 253.9 | 0.0007 | 0 |
| jev_stop_reflect | variable-star-vetting | 0.0000 | 4 | 6 | – | 1.1 | 2621546 | 100 | 12.5 | 852.1 | 18.5 | 901.5 | 4 | 0 | 1 | 3 | 282.7 | 0.0003 | 0 |
| jev_gate_stop | symbolic-regression | 0.0000 | 1 | 2 | – | 0.6053 | 921126 | 63 | 17.1 | 1376.6 | 81.6 | 1503.3 | 48 | 7 | 2 | 0 | 324.5 | 0.0056 | 0 |
| jev_gate_stop | variable-star-vetting | 0.0000 | 2 | 6 | – | 1.2 | 2939778 | 100 | 9.5 | 879.3 | 18.3 | 935.1 | 72 | 20 | 0 | 0 | 294.4 | 0.0072 | 0 |

| job | n | tasks | success | timeouts | agent_run_p50_s | agent_run_p90_s | trial_p50_s | trial_p90_s | llm_cost_mean | s1_cost_mean | cost_per_solved | steps_mean | s1_requests_mean | solved_by_600s | solved_by_1800s | solved_by_3600s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 2 | 2 | 1.0 | 0 | 30.0 | 53.7 | 103.2 | 124.0 | – | 0.0000 | 0.0000 | – | 0.0000 | 1.0 | 1.0 | 1.0 |
| nop | 2 | 2 | 0.0000 | 0 | 0.0002 | 0.0004 | 57.1 | 57.2 | – | 0.0000 | – | – | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | 2 | 2 | 0.0000 | 0 | 1139.8 | 1247.2 | 1215.5 | 1321.3 | 0.8267 | 0.0000 | – | 73.5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| jev_stop_reflect | 2 | 2 | 0.0000 | 0 | 1264.2 | 1593.9 | 1320.5 | 1655.8 | 1.2 | 0.0005 | – | 91.5 | 7.0 | 0.0000 | 0.0000 | 0.0000 |
| jev_gate_stop | 2 | 2 | 0.0000 | 0 | 1127.9 | 1326.8 | 1219.2 | 1446.5 | 0.9081 | 0.0064 | – | 81.5 | 60.0 | 0.0000 | 0.0000 | 0.0000 |

paired vs baseline (per-task attempt means; d = arm - baseline)
| job | tasks | success_base | success_arm | d_success_mean | wins | losses | sign_test_p | d_agent_run_s_median | faster_tasks | n_both_solved | d_agent_run_s_median_both_solved | d_cost_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 2 | 0.0000 | 1.0 | 1.0 | 2 | 0 | 0.5000 | -1109.8 | 2 | 0 | – | -0.8267 |
| nop | 2 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | – | -1139.8 | 2 | 0 | – | -0.8267 |
| jev_stop_reflect | 2 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | – | 124.4 | 1 | 0 | – | 0.3297 |
| jev_gate_stop | 2 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | – | -11.9 | 1 | 0 | – | 0.0878 |

H5 verdict (A5.2 / A6.6 / A6.16 (j), labels A6.22 (d) 1; exploratory): h5_better = the success-preservation screen (LB90(d_success) > -0.05 and wins >= losses; not an NI test); h5_strictly_better = LB90 > 0 and sign p < 0.05; faster = median d agent-run time on tasks both solve < 0 and exact one-sided sign test p < 0.05 (wilcoxon_p descriptive); d_agent_run_s_median (all tasks, timeouts at the cap) and solved_by_t (arm, _base) descriptive. Full set: an errored trial (any Harbor exception except AgentTimeoutError) counts as a failure; complete case: a task with an errored trial in either arm leaves the pair (A6.22 (e) 5)
| job | view | tasks | d_success_mean | d_success_lb90 | wins | losses | sign_p_wins | n_h5_both_solved | d_solved_run_s_median | time_sign_p | wilcoxon_p | d_agent_run_s_median | solved_by_600s | solved_by_600s_base | solved_by_1800s | solved_by_1800s_base | solved_by_3600s | solved_by_3600s_base | h5_better | h5_strictly_better | h5_faster | h5_verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | full set | 2 | 1.0 | 1.0 | 2 | 0 | 0.2500 | 0 | nan | 1.0 | 1.0 | -1109.8 | 1.0 | 0.0000 | 1.0 | 0.0000 | 1.0 | 0.0000 | True | False | False | success-preserving |
| nop | full set | 2 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 0 | nan | 1.0 | 1.0 | -1139.8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | False | success-preserving |
| jev_stop_reflect | full set | 2 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 0 | nan | 1.0 | 1.0 | 124.4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | False | success-preserving |
| jev_gate_stop | full set | 2 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 0 | nan | 1.0 | 1.0 | -11.9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | False | success-preserving |
| oracle | complete case | 2 | 1.0 | 1.0 | 2 | 0 | 0.2500 | 0 | nan | 1.0 | 1.0 | -1109.8 | 1.0 | 0.0000 | 1.0 | 0.0000 | 1.0 | 0.0000 | True | False | False | success-preserving |
| nop | complete case | 2 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 0 | nan | 1.0 | 1.0 | -1139.8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | False | success-preserving |
| jev_stop_reflect | complete case | 2 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 0 | nan | 1.0 | 1.0 | 124.4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | False | success-preserving |
| jev_gate_stop | complete case | 2 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 0 | nan | 1.0 | 1.0 | -11.9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | False | False | success-preserving |
