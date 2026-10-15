# E1 day-1 go/no-go, round 2 — NO-GO

Run `runs/explore_telecom_rl_validation2` (RL arm only), validation tasks `validation_gonogo_2` of `configs/tau2/explore/telecom_rl_task_ids.json` (5 tasks); decided 2026-10-05T07:56:17+00:00 by `scripts/explore_telecom_rl.py gonogo` under the criteria of the E1 Deviations entry (d). validation tasks: adapter checks only; no success number read.

| check | ok | details |
|---|---|---|
| complete_no_error | ok | `{"n_expected": 5, "n_got": 5, "errors": {}}` |
| s1_answers | ok | `{"tool_select_decisions": 118, "with_s1_answer": 118, "malformed": [], "delegated": 73, "tool_params_decisions": 49, "tool_params_with_s1_answer": 49}` |
| ledger | ok | `{"s1_models": {"jevagentrl-4b": 167}, "unpriced_models": [], "nonzero_or_unpriced_s1_calls": [], "s1_calls": 167, "s2_cost_usd": 1.1343279000000002}` |
| no_crash_or_loop | **FAIL** | `{"termination": {"user_stop": 5}, "repeated_identical_calls_gt3": [{"task": "[mms_issue]airplane_mode_on|bad_network_preference|break_apn_mms_setting|break_app_storage_permission|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]", "repeats": 23}], "agent_tool_errors": 0, "executed_calls_to_unknown_tools": 0, "messages_per_sim": [38, 56, 82, 64, 86]}` |
| projected_spend | ok | `{"rl_arm_usd_per_sim": 0.25204619800000005, "llm_only_usd_per_sim_assumed": 0.16452332233928568, "spent_before": 1.7875171900000002, "projected_total": 64.27294524089287, "cap": 75.0}` |

Diagnosis (operational fields only): Round 2 (after the fix: no undeclared-tool call, no tool error, every conversation ended by the user simulator) failed no_crash_or_loop on one of 5 tasks: 23 identical get_data_usage(customer_id C1001, line_id L1002) calls (step 11 and steps 13-34, 22 of them back to back with no message between). System-1 chose get_data_usage above tau at 10 of those steps although its state shows the preceding identical calls and results (state rendering checked: the history is present), and Flash, on deferral, continued the pattern. Not an adapter defect but the frozen checkpoint's behaviour on tau2 telecom; S2-L3a's two arms had 0 of 1,120 simulations with more than 3 identical consecutive calls. Decision: NO-GO; E1 is not run (E1 entry (d); Deviations 2026-10-05, round 2).
