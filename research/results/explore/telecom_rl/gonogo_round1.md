# E1 day-1 go/no-go, round 1 — NO-GO

Run `runs/explore_telecom_rl_validation` (RL arm only), validation tasks `validation_gonogo` of `configs/tau2/explore/telecom_rl_task_ids.json` (5 tasks); decided 2026-10-05T07:55:04+00:00 by `scripts/explore_telecom_rl.py gonogo` under the criteria of the E1 Deviations entry (d). validation tasks: adapter checks only; no success number read.

| check | ok | details |
|---|---|---|
| complete_no_error | ok | `{"n_expected": 5, "n_got": 5, "errors": {}}` |
| s1_answers | ok | `{"tool_select_decisions": 70, "with_s1_answer": 70, "malformed": [], "delegated": 26, "tool_params_decisions": 10, "tool_params_with_s1_answer": 10}` |
| ledger | ok | `{"s1_models": {"jevagentrl-4b": 80}, "unpriced_models": [], "nonzero_or_unpriced_s1_calls": [], "s1_calls": 80, "s2_cost_usd": 0.4819443}` |
| no_crash_or_loop | **FAIL** | `{"termination": {"too_many_errors": 3, "user_stop": 2}, "repeated_identical_calls_gt3": [], "agent_tool_errors": 37, "executed_calls_to_unknown_tools": 36, "messages_per_sim": [26, 26, 28, 36, 62]}` |
| projected_spend | ok | `{"rl_arm_usd_per_sim": 0.10545724000000001, "llm_only_usd_per_sim_assumed": 0.16452332233928568, "spent_before": 0.5272862, "projected_total": 41.024370550892854, "cap": 75.0}` |

Diagnosis (operational fields only): Round 1 failed no_crash_or_loop: 3 of 5 conversations ended by too_many_errors; 36 of 37 tool errors were executed calls to undeclared tools. Cause: System-1 answered respond_to_user above tau at 61 of 70 tool_select decisions; the adapter asked Flash for the message under tool_choice=none; Flash returned a text-less call to an invented tool, which the registered fallback executed. An adapter defect: fixed by respond_fallback: defer (Deviations 2026-10-05); round 2 on validation tasks 6-10.
