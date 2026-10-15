# E4: JevAgentHarness on mini-SWE-agent, SWE-bench Verified eval (100 instances)

**exploratory (E4), descriptive.** Outside every Holm table; revises no registered result.

## Arms (k = 1; cost = API $ per instance: agent LLM as logged (E4 DeepSeek: OpenRouter `usage.cost`; E4b Gemini: token usage at the verified list price) + Jev; time = agent-run wall clock per instance)

| arm | n | resolved | unevaluated | errors | cost_mean | llm_cost_mean | s1_cost_mean | time_median_s | time_mean_s | time_p90_s | steps_mean | llm_calls_mean | fast_share | exit_statuses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_deep | 100 | 81 | 0 | 0 | 0.0411 | 0.0411 | 0.0000 | 237.8 | 399.5 | 922.9 | 40.1 | 40.1 | 0.0000 | Submitted 100 |
| always_fast | 100 | 78 | 0 | 0 | 0.0108 | 0.0108 | 0.0000 | 186.6 | 272.4 | 602.7 | 46.9 | 46.9 | 1.0000 | Submitted 100 |
| jevagent | 100 | 81 | 0 | 0 | 0.0324 | 0.0291 | 0.0033 | 219.0 | 354.1 | 726.4 | 37.4 | 37.4 | 0.2355 | Submitted 100 |

## Where the time goes (descriptive): mean latency per call by path, share of agent-run time spent waiting for the agent LLM, share of input tokens served from the prompt cache

| arm | deep_call_mean_s | fast_call_mean_s | s1_call_mean_s | s1_calls_mean | llm_time_share | cached_input_share | time_total_h | cost_total |
|---|---|---|---|---|---|---|---|---|
| always_deep | 9.495 | – | – | 0.000 | 0.952 | 0.917 | 11.096 | 4.113 |
| always_fast | – | 5.334 | – | 0.000 | 0.919 | 0.963 | 7.568 | 1.082 |
| jevagent | 10.296 | 4.119 | 0.185 | 37.410 | 0.934 | 0.904 | 9.837 | 3.236 |

## Paired vs always_deep (descriptive; p values are descriptive, no test)

| arm | tasks | d_resolved | wins | losses | sign_p_two_sided | mcnemar_p | saving_pct | mean_time_saving_pct | n_cosolved | d_time_median_cosolved_s | cosolved_faster | cosolved_slower | cosolved_time_sign_p_one_sided | d_time_median_all_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_fast | 100 | -3 | 3 | 6 | 0.508 | 0.508 | 73.687 | 31.796 | 75 | -38.574 | 54 | 21 | 0.000 | -59.317 |
| jevagent | 100 | 0 | 3 | 3 | 1.000 | 1.000 | 21.321 | 11.347 | 78 | -25.750 | 50 | 28 | 0.008 | -21.853 |
