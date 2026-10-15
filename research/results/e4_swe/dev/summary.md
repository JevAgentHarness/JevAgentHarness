# E4: JevAgentHarness on mini-SWE-agent (dev: 45 instances, used only to fix the routing threshold; development numbers, never reported as results)

**exploratory (E4), descriptive.** Outside every Holm table; revises no registered result.

## Arms (k = 1; cost = API $ per instance: DeepSeek via OpenRouter `usage.cost` + Jev; time = agent-run wall clock per instance)

| arm | n | resolved | unevaluated | errors | cost_mean | llm_cost_mean | s1_cost_mean | time_median_s | time_mean_s | time_p90_s | steps_mean | llm_calls_mean | fast_share | exit_statuses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_deep | 45 | 36 | 0 | 0 | 0.0341 | 0.0341 | 0.0000 | 209.8 | 336.9 | 680.6 | 38.2 | 38.2 | 0.0000 | Submitted 45 |
| always_fast | 45 | 34 | 0 | 0 | 0.0125 | 0.0125 | 0.0000 | 277.8 | 344.9 | 675.4 | 49.4 | 49.4 | 1.0000 | Submitted 45 |
| jevagent_t050 | 45 | 38 | 0 | 0 | 0.0343 | 0.0309 | 0.0034 | 300.9 | 369.5 | 781.4 | 40.4 | 40.4 | 0.2140 | Submitted 45 |
| jevagent_t060 | 45 | 38 | 0 | 0 | 0.0287 | 0.0252 | 0.0035 | 314.4 | 366.7 | 715.4 | 40.6 | 40.6 | 0.3905 | Submitted 45 |
| jevagent_t070 | 45 | 33 | 0 | 0 | 0.0240 | 0.0200 | 0.0040 | 286.4 | 394.7 | 832.7 | 46.4 | 46.4 | 0.7282 | Submitted 45 |

## Where the time goes (descriptive): mean latency per call by path, share of agent-run time spent waiting for the agent LLM, share of input tokens served from the prompt cache

| arm | deep_call_mean_s | fast_call_mean_s | s1_call_mean_s | s1_calls_mean | llm_time_share | cached_input_share | time_total_h | cost_total |
|---|---|---|---|---|---|---|---|---|
| always_deep | 8.327 | – | – | 0.000 | 0.943 | 0.916 | 4.212 | 1.533 |
| always_fast | – | 6.427 | – | 0.000 | 0.920 | 0.920 | 4.312 | 0.562 |
| jevagent_t050 | 9.206 | 5.340 | 0.181 | 40.400 | 0.916 | 0.890 | 4.619 | 1.544 |
| jevagent_t060 | 9.853 | 5.707 | 0.190 | 40.578 | 0.911 | 0.890 | 4.584 | 1.291 |
| jevagent_t070 | 12.381 | 6.018 | 0.186 | 46.444 | 0.912 | 0.909 | 4.933 | 1.079 |

## Paired vs always_deep (descriptive; p values are descriptive, no test)

| arm | tasks | d_resolved | wins | losses | sign_p_two_sided | mcnemar_p | saving_pct | mean_time_saving_pct | n_cosolved | d_time_median_cosolved_s | cosolved_faster | cosolved_slower | cosolved_time_sign_p_one_sided | d_time_median_all_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_fast | 45 | -2 | 2 | 4 | 0.688 | 0.688 | 63.304 | -2.375 | 32 | 19.552 | 12 | 20 | 0.945 | 16.260 |
| jevagent_t050 | 45 | 2 | 3 | 1 | 0.625 | 0.625 | -0.737 | -9.666 | 35 | 20.906 | 14 | 21 | 0.912 | 16.462 |
| jevagent_t060 | 45 | 2 | 4 | 2 | 0.688 | 0.688 | 15.788 | -8.829 | 34 | 15.869 | 10 | 24 | 0.995 | 18.271 |
| jevagent_t070 | 45 | -3 | 3 | 6 | 0.508 | 0.508 | 29.622 | -17.137 | 30 | 24.733 | 11 | 19 | 0.951 | 35.774 |
