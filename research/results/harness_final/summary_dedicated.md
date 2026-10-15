# The final JevAgentHarness, live: dedicated GPU (at most 1 harness episode per GPU) (exploratory, descriptive)

Label: exploratory, descriptive (post hoc; PREREG_A7_STAGE2 Deviations 2026-10-05 harness_final); no test. Generated 2026-10-05T21:14:00+00:00 at git 19f4022ee184 by `python scripts/harness_final.py analyze`; inputs: `runs/harness_final_dedicated/llm_only.jsonl` (4042b40dca9aba98), `runs/harness_final_dedicated/harness_final.jsonl` (3c1319c66b38921b); tariff `results/harness_final/tariff_dedicated.json`.

Arms `llm_only` and `harness_final` (the frozen JevAgentRL-4B, tool + argument cascade at its frozen thresholds, reflection off, argument questions batched per step, `serve_jev --fast`), one invocation, task-major interleaved, concurrency 1; 200 complete pairs of 200 expected (latest record errored: {'llm_only': 0, 'harness_final': 0}). Host: g5.4xlarge (1x NVIDIA A10G 24 GB); System-1 latencies are figures of that GPU (S2-4's JevAgentRL rows ran on one L40S).

| quantity | llm_only | harness_final | difference (harness_final − llm_only) |
|---|---|---|---|
| success | 99.5 % | 100.0 % | +0.5 pt [+0.0, +1.5] |
| cost $ / episode (GPU tariff incl.) | 0.0431 | 0.0102 (API 0.0058 + q 0.00438) | saving 76.3 % [75.6, 77.1] |
| wall-clock s / episode, mean (median) | 13.59 (8.87) | 9.59 (7.89) | -4.00 s [-7.98, -0.61]; median -0.98 [-1.32, -0.07]; 29.4 % faster |
| model-call time s / episode, mean (median) | 11.68 (8.84) | 9.04 (7.80) | -2.64 s [-5.16, -0.44] |
| System-2 calls / episode | 7.53 | 3.04 | |
| System-1 requests (questions) / episode | – | 14.00 (18.12) | |

Discordant pairs: llm_only only 0, harness_final only 1. Intervals: paired task bootstrap (95 %), descriptive (no test).

Final harness decomposition (s / episode): System-1 3.93 (server compute 3.83, waiting for the engine 0.00: 0.0 % of server time), System-2 5.11, rest 0.55; prefix-cache hit share 37.6 % of the prompt tokens; tool decisions by System-1 100.0 %, arguments 85.5 %.
