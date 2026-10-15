# Stage 2: fixed-behaviour economics and timing audit (descriptive, post hoc)

Label: descriptive sensitivity, post hoc (review round 12); not a registered analysis. Paired task bootstrap B = 10000, seed 0. All 61 reproduction checks passed (frozen prices reproduce every logged System-2 call, every episode's logged cost, the registered per-episode costs, savings and their intervals, and the registered mean model-call latencies).

## (b) Re-pricing System-2 tokens only

Prices ($ per 1M tokens: input / cached input / output): sol = `gpt-6.1-sol` 2.0 / 0.1 / 10.0; astra = `gpt-6-astra` 10.0 / 1.0 / 50.0; frozen = each call's own model: `deepseek/deepseek-v4-flash` 0.028 / 0.0056 / 0.056; `deepseek/deepseek-v4-pro` 0.2088 / 0.0174 / 0.4176; `gemini-3.5-flash` 1.5 / 0.15 / 9.0.
System-1 cost and GPU tariffs as recorded; tokens, cache hits and calls as logged. Another LLM's reasoning tokens, caching and behaviour are not modelled (logged reasoning tokens in these arms: 0).

| arm | status | n | tariff q | $ / ep. frozen | Sol | Astra | S2 share (frozen) | savings frozen [95 %] | Sol | Astra | abs. saving $ / ep. frozen | Sol | Astra | × (Astra / frozen) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `llm_only` | reference | 500 | – | 0.0413 | 0.0523 | 0.273 | 100.0 % | – | – | – | – | – | – | – |
| `best_single` | descriptive | 500 | – | 0.0335 | 0.0425 | 0.215 | 96.6 % | 18.8 [17.2, 20.3] | 18.9 [17.3, 20.4] | 21.2 [19.6, 22.8] | 0.00775 | 0.00987 | 0.0577 | 7.45 |
| `jevagent_best` | S2-1 | 500 | – | 0.0245 | 0.0306 | 0.151 | 93.1 % | 40.7 [38.9, 42.5] | 41.6 [39.8, 43.3] | 44.6 [42.7, 46.3] | 0.0168 | 0.0218 | 0.122 | 7.22 |
| `jevagent_best_rl` | S2-4 | 500 | 0.00196 | 0.00779 | 0.00955 | 0.0399 | 74.8 % | 81.1 [80.7, 81.6] | 81.7 [81.3, 82.2] | 85.4 [84.9, 85.8] | 0.0335 | 0.0428 | 0.233 | 6.95 |
| `jevagent_best_sl4b` | S2-4S | 500 | 0.00217 | 0.0419 | 0.0531 | 0.265 | 94.8 % | -1.4 [-2.5, -0.4] | -1.5 [-2.5, -0.5] | 2.9 [1.9, 3.8] | -0.000594 | -0.000777 | 0.00793 | -13.35 |
| `jevagent_default` | descriptive | 500 | – | 0.0338 | 0.0428 | 0.217 | 96.7 % | 18.1 [16.6, 19.6] | 18.2 [16.7, 19.7] | 20.5 [19.0, 22.0] | 0.00748 | 0.00952 | 0.0559 | 7.47 |
| `jevagent_minus_tool` | descriptive | 200 | – | 0.0413 | 0.0519 | 0.27 | 99.2 % | -1.0 [-3.0, 0.5] | -0.3 [-2.2, 1.3] | -0.1 [-1.9, 1.4] | -0.000429 | -0.000137 | -0.000166 | 0.39 |
| `jevagent_best_v2` | exploratory | 500 | 0.0124 | 0.0187 | 0.0206 | 0.0535 | 33.8 % | 54.8 [53.8, 55.8] | 60.7 [59.8, 61.4] | 80.4 [79.9, 80.8] | 0.0226 | 0.0317 | 0.219 | 9.68 |
| `llm_only_x` | reference | 500 | – | 0.00203 | 0.0212 | 0.128 | 100.0 % | – | – | – | – | – | – | – |
| `jevagent_best_x` | S2-1X | 500 | – | 0.00246 | 0.0224 | 0.133 | 85.3 % | -20.8 [-22.0, -19.6] | -5.6 [-6.7, -4.5] | -3.9 [-4.9, -3.0] | -0.000423 | -0.00118 | -0.00499 | 11.81 |
| `llm_only_xsmall` | descriptive | 500 | – | 0.000503 | – | – | 100.0 % | 75.3 [74.8, 75.7] | – | – | 0.00153 | – | – | – |

Savings are vs the arm's LLM-only reference (`llm_only` for Flash, `llm_only_x` for DeepSeek-V4), re-priced the same way. `llm_only_xsmall` not re-priced: a cheaper-LLM reference: its row is defined by its own lower price, which re-pricing at another model's price would remove.

## (c) Timing (s per episode)

| arm | conc. | same invocation as ref. | wall-clock mean | median | Δ mean vs ref. [95 %] | Δ median [95 %] | model-call latency mean | median | Δ mean [95 %] | Δ median [95 %] | retry back-off mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `llm_only` | 8 | – | 6.83 | 7.05 | –  | –  | 6.80 | 7.01 | –  | –  | 0.000 |
| `best_single` | 8 | yes | 7.61 | 7.58 | +0.78 [+0.59, +0.97] | +0.53 [-0.70, +1.05] | 7.92 | 8.08 | +1.12 [+0.92, +1.29] | +1.07 [-0.24, +1.33] | 0.000 |
| `jevagent_best` | 8 | yes | 9.21 | 10.00 | +2.39 [+2.07, +2.73] | +2.95 [+1.84, +3.17] | 9.35 | 10.09 | +2.54 [+2.22, +2.89] | +3.08 [+2.68, +3.69] | 0.004 |
| `jevagent_best_rl` | 4 | no | 10.21 | 9.10 | +3.38 [+2.85, +3.95] | +2.05 [+0.93, +3.77] | 9.77 | 8.43 | +2.97 [+2.52, +3.42] | +1.42 [+0.78, +3.01] | 0.007 |
| `jevagent_best_sl4b` | 4 | no | 13.99 | 15.93 | +7.17 [+6.67, +7.67] | +8.88 [+4.78, +9.48] | 13.81 | 15.74 | +7.01 [+6.51, +7.51] | +8.73 [+4.69, +9.27] | 0.000 |
| `jevagent_default` | 8 | yes | 7.67 | 7.72 | +0.84 [+0.65, +1.02] | +0.67 [-0.47, +1.04] | 7.96 | 8.07 | +1.16 [+0.96, +1.33] | +1.06 [-0.16, +1.33] | 0.000 |
| `jevagent_minus_tool` | 8 | yes | 8.26 | 9.27 | +2.16 [+1.87, +2.43] | +2.34 [+1.58, +3.58] | 8.32 | 9.47 | +2.25 [+1.96, +2.51] | +2.58 [+1.69, +3.69] | 0.000 |
| `jevagent_best_v2` | 4 | no | 22.79 | 23.64 | +15.96 [+15.27, +16.65] | +16.59 [+8.30, +23.05] | 22.58 | 21.41 | +15.78 [+15.11, +16.46] | +14.40 [+8.21, +22.68] | 0.007 |
| `llm_only_x` | 8 | – | 33.80 | 35.51 | –  | –  | 33.53 | 35.47 | –  | –  | 0.002 |
| `jevagent_best_x` | 8 | yes | 37.82 | 40.52 | +4.02 [+2.61, +5.33] | +5.01 [+2.67, +8.88] | 37.45 | 40.09 | +3.92 [+2.69, +5.04] | +4.62 [+2.17, +8.21] | 0.006 |
| `llm_only_xsmall` | 8 | yes | 24.04 | 22.44 | -9.76 [-11.58, -7.93] | -13.07 [-14.95, -3.80] | 23.39 | 22.25 | -10.15 [-11.79, -8.54] | -13.22 [-15.09, -3.98] | 0.111 |

Wall-clock = the agent loop's elapsed time per episode (`wall_ms`), measured while `conc.` episodes ran in parallel; it includes tool execution, harness overhead and retry back-off. Model-call latency = the sum of the logged call latencies (`usage.total.latency_ms`, the registered S2-1T quantity; System-1 calls of one step may overlap, so it can exceed the wall-clock). Arms not run in the reference's invocation (`same invocation` = no) ran in another time window and at another concurrency: their differences mix the arm with the serving conditions.

## (a) Reproduction checks

| arm | check | registered | recomputed | ok |
|---|---|---|---|---|
| `llm_only` | input sha256 = registered provenance | fba5397498f8 | fba5397498f8 | yes |
| `best_single` | input sha256 = registered provenance | 43495c708896 | 43495c708896 | yes |
| `jevagent_best` | input sha256 = registered provenance | defe93054b0b | defe93054b0b | yes |
| `jevagent_best_rl` | input sha256 = registered provenance | 97da04a8e4a7 | 97da04a8e4a7 | yes |
| `jevagent_best_sl4b` | input sha256 = registered provenance | b334cfba2276 | b334cfba2276 | yes |
| `jevagent_default` | input sha256 = registered provenance | 9d121a653e0b | 9d121a653e0b | yes |
| `jevagent_minus_tool` | input sha256 = registered provenance | 14fae4e9d093 | 14fae4e9d093 | yes |
| `jevagent_best_v2` | input sha256 = registered provenance | 79a154973b72 | 79a154973b72 | yes |
| `llm_only_x` | input sha256 = registered provenance | 3e82d6a70e5c | 3e82d6a70e5c | yes |
| `jevagent_best_x` | input sha256 = registered provenance | 03aa7d1c43c3 | 03aa7d1c43c3 | yes |
| `llm_only_xsmall` | input sha256 = registered provenance | 37f500d9d921 | 37f500d9d921 | yes |
| `llm_only` | mean $ / episode = family/S2-1/cost/cost_ref | 0.0412984074 | 0.0412984074 | yes |
| `llm_only` | mean $ / episode = descriptive/scorecard/llm_only/cost_usd | 0.0412984074 | 0.0412984074 | yes |
| `llm_only` | mean model-call latency (s) = descriptive/scorecard/llm_only/latency_s | 6.80479705969 | 6.80479705969 | yes |
| `llm_only` | mean model-call latency (s) = secondary/S2-1T/latency_ref_s | 6.80479705969 | 6.80479705969 | yes |
| `best_single` | mean $ / episode = family/S2-2/cost/cost_ref | 0.033544701708 | 0.033544701708 | yes |
| `best_single` | mean $ / episode = descriptive/scorecard/best_single/cost_usd | 0.033544701708 | 0.033544701708 | yes |
| `best_single` | mean model-call latency (s) = descriptive/scorecard/best_single/latency_s | 7.91988162098 | 7.91988162098 | yes |
| `best_single` | savings = descriptive/scorecard/best_single/vs_baseline/cost_savings | 0.187748297819 | 0.187748297819 | yes |
| `best_single` | savings bootstrap lo = descriptive/scorecard/best_single/vs_baseline/cost_savings_ci | 0.172206598027 | 0.172206598027 | yes |
| `best_single` | savings bootstrap hi = descriptive/scorecard/best_single/vs_baseline/cost_savings_ci | 0.203304338352 | 0.203304338352 | yes |
| `jevagent_best` | mean $ / episode = family/S2-1/cost/cost_arm | 0.024478962048 | 0.024478962048 | yes |
| `jevagent_best` | mean model-call latency (s) = secondary/S2-1T/latency_arm_s | 9.34683184522 | 9.34683184522 | yes |
| `jevagent_best` | savings = family/S2-1/cost/cost_savings | 0.407266197679 | 0.407266197679 | yes |
| `jevagent_best` | savings bootstrap lo = family/S2-1/cost/cost_savings_ci | 0.388899954612 | 0.388899954612 | yes |
| `jevagent_best` | savings bootstrap hi = family/S2-1/cost/cost_savings_ci | 0.424501853186 | 0.424501853186 | yes |
| `jevagent_best_rl` | mean $ / episode = family/S2-4/cost/cost_arm | 0.007794759 | 0.007794759 | yes |
| `jevagent_best_rl` | mean $ / episode = descriptive/scorecard/jevagent_best_rl/cost_usd | 0.005832939 | 0.005832939 | yes |
| `jevagent_best_rl` | mean model-call latency (s) = descriptive/scorecard/jevagent_best_rl/latency_s | 9.77373558219 | 9.77373558219 | yes |
| `jevagent_best_rl` | savings = family/S2-4/cost/cost_savings | 0.81125763702 | 0.81125763702 | yes |
| `jevagent_best_rl` | savings bootstrap lo = family/S2-4/cost/cost_savings_ci | 0.806852040869 | 0.806852040869 | yes |
| `jevagent_best_rl` | savings bootstrap hi = family/S2-4/cost/cost_savings_ci | 0.815577598364 | 0.815577598364 | yes |
| `jevagent_best_sl4b` | mean $ / episode = secondary/S2-4S/cost/cost_arm | 0.0418924143 | 0.0418924143 | yes |
| `jevagent_best_sl4b` | mean $ / episode = descriptive/scorecard/jevagent_best_sl4b/cost_usd | 0.0397250703 | 0.0397250703 | yes |
| `jevagent_best_sl4b` | mean model-call latency (s) = descriptive/scorecard/jevagent_best_sl4b/latency_s | 13.8113396545 | 13.8113396545 | yes |
| `jevagent_best_sl4b` | savings = secondary/S2-4S/cost/cost_savings | -0.0143832883008 | -0.0143832883008 | yes |
| `jevagent_best_sl4b` | savings bootstrap lo = secondary/S2-4S/cost/cost_savings_ci | -0.0249402746401 | -0.0249402746401 | yes |
| `jevagent_best_sl4b` | savings bootstrap hi = secondary/S2-4S/cost/cost_savings_ci | -0.00423598420083 | -0.00423598420083 | yes |
| `jevagent_default` | mean $ / episode = descriptive/scorecard/jevagent_default/cost_usd | 0.033818257872 | 0.033818257872 | yes |
| `jevagent_default` | mean model-call latency (s) = descriptive/scorecard/jevagent_default/latency_s | 7.95984414381 | 7.95984414381 | yes |
| `jevagent_default` | savings = descriptive/jevagent_default/cost/cost_savings | 0.181124406458 | 0.181124406458 | yes |
| `jevagent_default` | savings bootstrap lo = descriptive/jevagent_default/cost/cost_savings_ci | 0.166447171552 | 0.166447171552 | yes |
| `jevagent_default` | savings bootstrap hi = descriptive/jevagent_default/cost/cost_savings_ci | 0.196213471906 | 0.196213471906 | yes |
| `jevagent_minus_tool` | mean $ / episode = descriptive/logo/tool/cost/cost_arm | 0.04131913434 | 0.04131913434 | yes |
| `jevagent_best_v2` | mean $ / episode = descriptive/stage2_v2/profile/cost/cost_arm | 0.01865643325 | 0.01865643325 | yes |
| `jevagent_best_v2` | savings = descriptive/stage2_v2/profile/cost/cost_savings | 0.548252961203 | 0.548252961203 | yes |
| `jevagent_best_v2` | savings bootstrap lo = descriptive/stage2_v2/profile/cost/cost_savings_ci | 0.538149803258 | 0.538149803258 | yes |
| `jevagent_best_v2` | savings bootstrap hi = descriptive/stage2_v2/profile/cost/cost_savings_ci | 0.557620601939 | 0.557620601939 | yes |
| `llm_only_x` | mean $ / episode = secondary/S2-1X/cost/cost_ref | 0.0020334733764 | 0.0020334733764 | yes |
| `llm_only_x` | mean $ / episode = descriptive/scorecard/llm_only_x/cost_usd | 0.0020334733764 | 0.0020334733764 | yes |
| `llm_only_x` | mean model-call latency (s) = descriptive/scorecard/llm_only_x/latency_s | 33.5347125566 | 33.5347125566 | yes |
| `jevagent_best_x` | mean $ / episode = secondary/S2-1X/cost/cost_arm | 0.0024565002312 | 0.0024565002312 | yes |
| `jevagent_best_x` | mean model-call latency (s) = descriptive/scorecard/jevagent_best_x/latency_s | 37.4507657023 | 37.4507657023 | yes |
| `jevagent_best_x` | savings = secondary/S2-1X/cost/cost_savings | -0.208031666266 | -0.208031666266 | yes |
| `jevagent_best_x` | savings bootstrap lo = secondary/S2-1X/cost/cost_savings_ci | -0.220060589595 | -0.220060589595 | yes |
| `jevagent_best_x` | savings bootstrap hi = secondary/S2-1X/cost/cost_savings_ci | -0.196185452602 | -0.196185452602 | yes |
| `llm_only_xsmall` | mean $ / episode = descriptive/scorecard/llm_only_xsmall/cost_usd | 0.000502691336 | 0.000502691336 | yes |
| `llm_only_xsmall` | mean model-call latency (s) = descriptive/scorecard/llm_only_xsmall/latency_s | 23.3851632876 | 23.3851632876 | yes |
| `llm_only_xsmall` | savings = descriptive/llm_only_xsmall/cost/cost_savings | 0.752791778917 | 0.752791778917 | yes |
| `llm_only_xsmall` | savings bootstrap lo = descriptive/llm_only_xsmall/cost/cost_savings_ci | 0.747971390887 | 0.747971390887 | yes |
| `llm_only_xsmall` | savings bootstrap hi = descriptive/llm_only_xsmall/cost/cost_savings_ci | 0.75712664086 | 0.75712664086 | yes |

Generated 2026-10-05T07:12:54+00:00 at git 4e95fbb390ef; inputs: `runs/stage2_jevgym/llm_only.jsonl` (fba5397498f8), `runs/stage2_jevgym/best_single.jsonl` (43495c708896), `runs/stage2_jevgym/jevagent_best.jsonl` (defe93054b0b), `runs/stage2_jevgym/jevagent_best_rl.jsonl` (97da04a8e4a7), `runs/stage2_jevgym/jevagent_best_sl4b.jsonl` (b334cfba2276), `runs/stage2_jevgym/jevagent_default.jsonl` (9d121a653e0b), `runs/stage2_jevgym/jevagent_minus_tool.jsonl` (14fae4e9d093), `runs/stage2_v2/jevagent_best_v2.jsonl` (79a154973b72), `runs/stage2_x/llm_only_x.jsonl` (3e82d6a70e5c), `runs/stage2_x/jevagent_best_x.jsonl` (03aa7d1c43c3), `runs/stage2_x/llm_only_xsmall.jsonl` (37f500d9d921).
