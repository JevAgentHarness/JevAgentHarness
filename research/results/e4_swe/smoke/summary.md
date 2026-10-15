# E4: JevAgentHarness on mini-SWE-agent (smoke: the H5 smoke pair, development only, never reported)

**exploratory (E4), descriptive.** Outside every Holm table; revises no registered result.

## Arms (k = 1; cost = API $ per instance: DeepSeek via OpenRouter `usage.cost` + Jev; time = agent-run wall clock per instance)

| arm | n | resolved | unevaluated | errors | cost_mean | llm_cost_mean | s1_cost_mean | time_median_s | time_mean_s | time_p90_s | steps_mean | llm_calls_mean | fast_share | exit_statuses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_deep | 2 | 1 | 0 | 0 | 0.0231 | 0.0231 | 0.0000 | 244.4 | 244.4 | 339.2 | 31.0 | 31.0 | 0.0000 | Submitted 2 |
| always_fast | 2 | 1 | 0 | 0 | 0.0082 | 0.0082 | 0.0000 | 179.7 | 179.7 | 249.9 | 34.5 | 34.5 | 1.0000 | Submitted 2 |
| jevagent_t050 | 2 | 1 | 0 | 0 | 0.0161 | 0.0137 | 0.0024 | 217.6 | 217.6 | 292.5 | 28.0 | 28.0 | 0.4464 | Submitted 2 |

## Paired vs always_deep (descriptive; p values are descriptive, no test)

| arm | tasks | d_resolved | wins | losses | sign_p_two_sided | mcnemar_p | saving_pct | mean_time_saving_pct | n_cosolved | d_time_median_cosolved_s | cosolved_faster | cosolved_slower | cosolved_time_sign_p_one_sided | d_time_median_all_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_fast | 2 | 0 | 0 | 0 | – | 1.000 | 64.412 | 26.456 | 1 | -33.869 | 1 | 0 | 0.500 | -64.645 |
| jevagent_t050 | 2 | 0 | 0 | 0 | – | 1.000 | 30.336 | 10.968 | 1 | -1.916 | 1 | 0 | 0.500 | -26.801 |
