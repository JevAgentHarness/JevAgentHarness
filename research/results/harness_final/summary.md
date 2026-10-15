# The final JevAgentHarness, live: shared GPU (about 4 harness episodes per GPU) (exploratory, descriptive)

Label: exploratory, descriptive (post hoc; PREREG_A7_STAGE2 Deviations 2026-10-05 harness_final); no test. Generated 2026-10-05T19:38:09+00:00 at git df6945cecd69 by `python scripts/harness_final.py analyze`; inputs: `runs/harness_final_live/llm_only.jsonl` (661c75d4fed581cc), `runs/harness_final_live/harness_final.jsonl` (b350dce8b7dc6e00); tariff `results/harness_final/tariff.json`.

Arms `llm_only` and `harness_final` (the frozen JevAgentRL-4B, tool + argument cascade at its frozen thresholds, reflection off, argument questions batched per step, `serve_jev --fast`), one invocation, task-major interleaved, concurrency 8; 200 complete pairs of 200 expected (latest record errored: {'llm_only': 0, 'harness_final': 0}). Host: g5.4xlarge (1x NVIDIA A10G 24 GB); System-1 latencies are figures of that GPU (S2-4's JevAgentRL rows ran on one L40S).

| quantity | llm_only | harness_final | difference (harness_final − llm_only) |
|---|---|---|---|
| success | 99.5 % | 100.0 % | +0.5 pt [+0.0, +1.5] |
| cost $ / episode (GPU tariff incl.) | 0.0434 | 0.0078 (API 0.0058 + q 0.00192) | saving 82.1 % [81.5, 82.7] |
| wall-clock s / episode, mean (median) | 8.70 (8.94) | 22.39 (20.72) | +13.69 s [+12.08, +15.13]; median +11.78 [+10.76, +14.36]; 157.4 % slower |
| model-call time s / episode, mean (median) | 8.43 (8.89) | 22.05 (20.47) | +13.62 s [+12.16, +14.93] |
| System-2 calls / episode | 7.52 | 3.05 | |
| System-1 requests (questions) / episode | – | 14.00 (18.12) | |

Discordant pairs: llm_only only 0, harness_final only 1. Intervals: paired task bootstrap (95 %), descriptive (no test).

Final harness decomposition (s / episode): System-1 16.41 (server compute 3.85, waiting for the engine 12.44: 76.3 % of server time), System-2 5.63, rest 0.34; prefix-cache hit share 37.6 % of the prompt tokens; tool decisions by System-1 100.0 %, arguments 85.4 %.
