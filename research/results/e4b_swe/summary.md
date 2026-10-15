# E4B: JevAgentHarness on mini-SWE-agent, SWE-bench Verified eval (100 instances)

**exploratory (E4b), descriptive.** Outside every Holm table; revises no registered result.

## Arms (k = 1; cost = API $ per instance: agent LLM as logged (E4 DeepSeek: OpenRouter `usage.cost`; E4b Gemini: token usage at the verified list price) + Jev; time = agent-run wall clock per instance)

| arm | n | resolved | unevaluated | errors | cost_mean | llm_cost_mean | s1_cost_mean | time_median_s | time_mean_s | time_p90_s | steps_mean | llm_calls_mean | fast_share | exit_statuses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_deep | 100 | 70 | 0 | 0 | 1.0164 | 1.0164 | 0.0000 | 906.9 | 886.0 | 1354.8 | 93.6 | 93.6 | 0.0000 | Submitted 76, LimitsExceeded 24 |
| always_fast | 50 | 40 | 0 | 0 | 0.6597 | 0.6597 | 0.0000 | 537.1 | 557.7 | 1028.2 | 70.1 | 70.1 | 1.0000 | Submitted 47, LimitsExceeded 3 |
| jevagent | 100 | 75 | 0 | 0 | 0.9448 | 0.9376 | 0.0071 | 720.8 | 728.2 | 1102.8 | 83.1 | 83.1 | 0.2161 | Submitted 83, LimitsExceeded 17 |

## Where the time goes (descriptive): mean latency per call by path, share of agent-run time spent waiting for the agent LLM, share of input tokens served from the prompt cache

| arm | deep_call_mean_s | fast_call_mean_s | s1_call_mean_s | s1_calls_mean | llm_time_share | cached_input_share | time_total_h | cost_total |
|---|---|---|---|---|---|---|---|---|
| always_deep | 8.609 | – | – | 0.000 | 0.909 | 0.885 | 24.612 | 101.641 |
| always_fast | – | 7.230 | – | 0.000 | 0.909 | 0.831 | 7.746 | 32.986 |
| jevagent | 7.979 | 7.488 | 0.182 | 83.110 | 0.899 | 0.824 | 20.229 | 94.478 |

## Paired vs always_deep (descriptive; p values are descriptive, no test)

| arm | tasks | d_resolved | wins | losses | sign_p_two_sided | mcnemar_p | saving_pct | mean_time_saving_pct | n_cosolved | d_time_median_cosolved_s | cosolved_faster | cosolved_slower | cosolved_time_sign_p_one_sided | d_time_median_all_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_fast | 50 | 4 | 6 | 2 | 0.289 | 0.289 | 36.247 | 35.300 | 34 | -332.749 | 30 | 4 | 0.000 | -271.710 |
| jevagent | 100 | 5 | 8 | 3 | 0.227 | 0.227 | 7.047 | 17.810 | 67 | -120.479 | 47 | 20 | 0.001 | -123.313 |
