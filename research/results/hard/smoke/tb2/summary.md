| job | task | reward | tests_passed | tests_total | error | llm_cost_usd | llm_input_tokens | agent_steps | agent_setup_s | agent_run_s | verifier_s | trial_s | s1_requests | blocks | stop_nudges | reflections | s1_ms_mean | s1_cost_usd_est | s1_errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | overfull-hbox | 1.0 | 4 | 4 | – | 0.4345 | 579524 | 43 | 19.6 | 360.3 | 32.6 | 424.8 | 0 | 0 | 0 | 0 | – | 0.0000 | 0 |
| baseline | prove-plus-comm | 1.0 | 4 | 4 | – | 0.0494 | 29168 | 12 | 18.2 | 16.0 | 9.9 | 60.9 | 0 | 0 | 0 | 0 | – | 0.0000 | 0 |
| jev_stop | overfull-hbox | 1.0 | 4 | 4 | – | 0.2562 | 268614 | 34 | 39.4 | 262.2 | 32.9 | 352.4 | 1 | 0 | 0 | 0 | 235.8 | 0.0001 | 0 |
| jev_stop | prove-plus-comm | 1.0 | 4 | 4 | – | 0.0412 | 23569 | 16 | 38.3 | 119.4 | 7.9 | 185.6 | 1 | 0 | 0 | 0 | 208.0 | 0.0001 | 0 |
| jev_gate_stop | overfull-hbox | 1.0 | 4 | 4 | – | 0.3927 | 435840 | 50 | 32.8 | 197.5 | 32.9 | 282.5 | 41 | 7 | 0 | 0 | 234.5 | 0.0024 | 0 |
| jev_gate_stop | prove-plus-comm | 1.0 | 4 | 4 | – | 0.0289 | 17547 | 15 | 33.9 | 188.5 | 5.1 | 248.3 | 6 | 2 | 0 | 0 | 325.1 | 0.0002 | 0 |

| job | n | tasks | success | timeouts | agent_run_p50_s | agent_run_p90_s | trial_p50_s | trial_p90_s | llm_cost_mean | s1_cost_mean | cost_per_solved | steps_mean | s1_requests_mean | solved_by_600s | solved_by_1800s | solved_by_3600s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 2 | 2 | 1.0 | 0 | 188.1 | 325.9 | 242.9 | 388.4 | 0.2419 | 0.0000 | 0.2419 | 27.5 | 0.0000 | 1.0 | 1.0 | 1.0 |
| jev_stop | 2 | 2 | 1.0 | 0 | 190.8 | 247.9 | 269.0 | 335.8 | 0.1487 | 0.0001 | 0.1488 | 25.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| jev_gate_stop | 2 | 2 | 1.0 | 0 | 193.0 | 196.6 | 265.4 | 279.1 | 0.2108 | 0.0013 | 0.2121 | 32.5 | 23.5 | 1.0 | 1.0 | 1.0 |

paired vs baseline (per-task attempt means; d = arm - baseline)
| job | tasks | success_base | success_arm | d_success_mean | wins | losses | sign_test_p | d_agent_run_s_median | faster_tasks | n_both_solved | d_agent_run_s_median_both_solved | d_cost_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_stop | 2 | 1.0 | 1.0 | 0.0000 | 0 | 0 | – | 2.7 | 1 | 2 | 2.7 | -0.0932 |
| jev_gate_stop | 2 | 1.0 | 1.0 | 0.0000 | 0 | 0 | – | 4.8 | 1 | 2 | 4.8 | -0.0299 |

H5 verdict (A5.2 / A6.6 / A6.16 (j), labels A6.22 (d) 1; exploratory): h5_better = the success-preservation screen (LB90(d_success) > -0.05 and wins >= losses; not an NI test); h5_strictly_better = LB90 > 0 and sign p < 0.05; faster = median d agent-run time on tasks both solve < 0 and exact one-sided sign test p < 0.05 (wilcoxon_p descriptive); d_agent_run_s_median (all tasks, timeouts at the cap) and solved_by_t (arm, _base) descriptive. Full set: an errored trial (any Harbor exception except AgentTimeoutError) counts as a failure; complete case: a task with an errored trial in either arm leaves the pair (A6.22 (e) 5)
| job | view | tasks | d_success_mean | d_success_lb90 | wins | losses | sign_p_wins | n_h5_both_solved | d_solved_run_s_median | time_sign_p | wilcoxon_p | d_agent_run_s_median | solved_by_600s | solved_by_600s_base | solved_by_1800s | solved_by_1800s_base | solved_by_3600s | solved_by_3600s_base | h5_better | h5_strictly_better | h5_faster | h5_verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_stop | full set | 2 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 2 | 2.7 | 0.7500 | 0.7500 | 2.7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | True | False | False | success-preserving |
| jev_gate_stop | full set | 2 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 2 | 4.8 | 0.7500 | 0.7500 | 4.8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | True | False | False | success-preserving |
| jev_stop | complete case | 2 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 2 | 2.7 | 0.7500 | 0.7500 | 2.7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | True | False | False | success-preserving |
| jev_gate_stop | complete case | 2 | 0.0000 | 0.0000 | 0 | 0 | 1.0 | 2 | 4.8 | 0.7500 | 0.7500 | 4.8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | True | False | False | success-preserving |
