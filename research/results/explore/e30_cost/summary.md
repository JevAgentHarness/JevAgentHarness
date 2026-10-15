# E30: every open decision model at its cheapest common deployment (exploratory, descriptive)

exploratory (E30), descriptive; post hoc re-measurement; PREREG_A7_STAGE2 Deviations 2026-10-09 'E30: every open decision model at its cheapest common deployment'; outside every Holm table.

Generated 2026-10-10T00:50:16+00:00 at git 8aba289220fc by `python scripts/explore_e30_cost.py summarize`.

Instance: g6.xlarge (NVIDIA L4 24 GB), $0.8048 per hour (AWS price list (us-east-1, on demand, Linux, shared tenancy, effective 2026-10-01)). Engine: vLLM 0.31.0 in process (token-id prompts, prefix caching, continuous batching, CUDA graphs); the 27B models on llama.cpp (Q4_K_M GGUF files), E27 SP on our stack (serve_jev --backend clef). Workload: E9's 3,991 L1 decisions in one fixed shuffled order; saturated decisions/s = the best of C = 64 and C = 256 (llama.cpp: P = 8 slots; E27 SP: 16 and 64 requests); cost per 1,000 = $/h / (3,600 x dps) x 1,000; latency = C = 1 over the first 1,000 decisions. Plotted point = the cheapest precision whose accuracy is at least the reference minus 1 point.

## Plotted points

| model | precision | engine | L1 acc % (reference) | decisions/s (C) | $ per 1,000 | x cheaper than E9's basis | p50 / p90 ms (C = 1) | flag |
|---|---|---|---|---|---|---|---|---|
| AgentDecision-4B | FP8 | vLLM | 84.8 (85.1) | 50.2 (256) | 0.0045 | 31.4 | 31 / 52 | – |
| Frozen GRPO 4B | FP8 | vLLM | 78.6 (78.8) | 50.2 (256) | 0.0045 | 20.9 | 31 / 52 | – |
| StartLux-4B + our recipe | FP8 | vLLM | 85.5 (85.5) | 30.3 (64) | 0.0074 | 39.3 | 45 / 70 | – |
| AgentDecision-0.8B (StartLux-0.8B + our recipe, E28) | FP8 | vLLM | 80.5 (80.5) | 122.5 (64) | 0.0018 | 5.5 | 27 / 30 | – |
| StartLux-Decision-0.8B | FP8 | vLLM | 69.3 (70.0) | 131.3 (64) | 0.0017 | 7.6 | 32 / 34 | – |
| StartLux-Decision-2B | FP8 | vLLM | 76.0 (75.7) | 81.6 (64) | 0.0027 | 8.0 | 34 / 36 | – |
| StartLux-Decision-4B | FP8 | vLLM | 83.0 (83.1) | 33.0 (64) | 0.0068 | 7.7 | 44 / 69 | – |
| StartLux-Decision-9B | FP8 | vLLM | 83.3 (83.4) | 19.5 (64) | 0.0114 | 7.3 | 66 / 118 | – |
| StartLux-Decision-27B | Q4_K_M (GGUF) | llama.cpp | 87.5 (87.7) | 1.3 (8) | 0.1659 | 6.6 | 649 / 1224 | – |
| Intern-Decision-4B | FP8 | vLLM | 73.5 (73.4) | 18.0 (64) | 0.0124 | 3.1 | 61 / 125 | – |
| Qwen3.5-0.8B variant (E27, scoring head) | BF16 | serve_jev (scoring head) | 77.3 (77.2) | 26.7 (16) | 0.0084 | 10.9 | 60 / 63 | – |
| OpenJev-27B | Q4_K_M (GGUF) | llama.cpp | 83.0 (82.7) | 1.3 (8) | 0.1775 | 1.4 | 744 / 1231 | – |
| decider-4b | FP8 | vLLM | 79.4 (79.3) | 37.1 (64) | 0.0060 | 9.0 | 39 / 66 | – |
| JevK5 (v0.3, E15's weights) | FP8 | vLLM | 70.1 (70.3) | 25.1 (64) | 0.0089 | 5.4 | 46 / 105 | – |
| JevK5 v0.2 (the registered tag, not E15's weights) | FP8 | vLLM | 72.7 (70.3) | 24.6 (64) | 0.0091 | 5.3 | 47 / 108 | agreement 87.3 % < 95 % |
| Clef-Flash 9B (scoring head) | – | – | – (81.4) | – | – | – | – | not measured |
| Imajev-4B | – | – | – (78.9) | – | – | – | – | not measured |
| OpenJev (DiffusionGemma-26B-A4B) | FP8 | its own image (g6e.xlarge) | 79.1 (78.8) | 45.0 (64) | 0.0115 | 7.9 | 144 / 162 | – |

## Every precision measured

| model | precision | state | acc % | tool choice / args / When2Call | agreement with the reference record % | decisions/s per pass | $ per 1,000 | p50 / p90 ms | load s | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| AgentDecision-4B | BF16 | ok | 85.2 | 89.3 / 90.7 / 68.6 | 99.6 | 37.8 (C 64), 35.4 (C 256) | 0.0059 | 45 / 73 | 122 | – |
| AgentDecision-4B | FP8 | ok | 84.8 | 89.0 / 90.5 / 67.7 | 98.6 | 49.5 (C 64), 50.2 (C 256) | 0.0045 | 31 / 52 | 93 | – |
| Frozen GRPO 4B | BF16 | ok | 78.8 | 88.5 / 85.1 / 54.0 | 99.8 | 37.5 (C 64), 35.4 (C 256) | 0.0060 | 45 / 75 | 102 | – |
| Frozen GRPO 4B | FP8 | ok | 78.6 | 88.3 / 84.8 / 54.1 | 98.5 | 49.6 (C 64), 50.2 (C 256) | 0.0045 | 31 / 52 | 95 | – |
| StartLux-4B + our recipe | BF16 | ok | 85.5 | 89.5 / 91.8 / 67.2 | 99.9 | 23.4 (C 64), 22.9 (C 256) | 0.0096 | 58 / 99 | 157 | – |
| StartLux-4B + our recipe | FP8 | ok | 85.5 | 89.6 / 91.9 / 67.3 | 98.7 | 30.3 (C 64), 29.6 (C 256) | 0.0074 | 45 / 70 | 180 | – |
| AgentDecision-0.8B (StartLux-0.8B + our recipe, E28) | BF16 | ok | 80.5 | 87.2 / 87.2 / 58.3 | 99.9 | 113.8 (C 64), 109.6 (C 256) | 0.0020 | 31 / 34 | 136 | – |
| AgentDecision-0.8B (StartLux-0.8B + our recipe, E28) | FP8 | ok | 80.5 | 87.3 / 87.3 / 58.1 | 98.3 | 122.5 (C 64), 121.5 (C 256) | 0.0018 | 27 / 30 | 142 | – |
| StartLux-Decision-0.8B | BF16 | ok | 69.9 | 88.3 / 66.9 / 54.1 | 99.1 | 122.2 (C 64), 116.1 (C 256) | 0.0018 | 35 / 37 | 139 | 54 wide-choice reads |
| StartLux-Decision-0.8B | FP8 | ok | 69.3 | 87.6 / 66.5 / 53.2 | 96.9 | 131.3 (C 64), 128.7 (C 256) | 0.0017 | 32 / 34 | 153 | 54 wide-choice reads |
| StartLux-Decision-2B | BF16 | ok | 75.9 | 90.2 / 73.1 / 64.5 | 99.4 | 65.2 (C 64), 63.3 (C 256) | 0.0034 | 37 / 40 | 135 | 54 wide-choice reads |
| StartLux-Decision-2B | FP8 | ok | 76.0 | 90.3 / 73.7 / 63.7 | 97.4 | 81.6 (C 64), 80.4 (C 256) | 0.0027 | 34 / 36 | 154 | 54 wide-choice reads |
| StartLux-Decision-4B | BF16 | ok | 83.1 | 91.1 / 81.2 / 77.5 | 99.7 | 25.3 (C 64), 24.4 (C 256) | 0.0088 | 56 / 98 | 157 | 54 wide-choice reads |
| StartLux-Decision-4B | FP8 | ok | 83.0 | 91.3 / 81.2 / 76.7 | 98.6 | 33.0 (C 64), 32.2 (C 256) | 0.0068 | 44 / 69 | 182 | 54 wide-choice reads |
| StartLux-Decision-9B | BF16 | ok | 83.5 | 92.1 / 81.1 / 78.1 | 99.8 | 12.9 (C 64), 12.7 (C 256) | 0.0173 | 99 / 195 | 545 | engine {'max_num_seqs': 64}; 54 wide-choice reads |
| StartLux-Decision-9B | FP8 | ok | 83.3 | 92.3 / 80.6 / 78.2 | 98.7 | 19.5 (C 64), 18.8 (C 256) | 0.0114 | 66 / 118 | 241 | 54 wide-choice reads |
| StartLux-Decision-27B | Q4_K_M (GGUF) | ok | 87.5 | 91.3 / 88.7 / 80.6 | 98.4 | 1.3 (C 8) | 0.1659 | 649 / 1224 | – | – |
| Intern-Decision-4B | BF16 | ok | 73.4 | 88.3 / 71.1 / 60.2 | 99.4 | 13.6 (C 64), 13.4 (C 256) | 0.0164 | 83 / 172 | 177 | – |
| Intern-Decision-4B | FP8 | ok | 73.5 | 88.3 / 71.2 / 60.5 | 97.1 | 18.0 (C 64), 17.8 (C 256) | 0.0124 | 61 / 125 | 187 | – |
| Qwen3.5-0.8B variant (E27, scoring head) | BF16 | ok | 77.3 | 83.2 / 89.1 / 45.4 | 99.8 | 26.7 (C 16), 24.4 (C 64) | 0.0084 | 60 / 63 | – | – |
| OpenJev-27B | Q4_K_M (GGUF) | ok | 83.0 | 90.1 / 81.9 / 76.7 | 98.4 | 1.3 (C 8) | 0.1775 | 744 / 1231 | – | – |
| decider-4b | BF16 | ok | 79.3 | 91.0 / 76.8 / 70.5 | 99.8 | 29.0 (C 64), 27.6 (C 256) | 0.0077 | 53 / 96 | 157 | – |
| decider-4b | FP8 | ok | 79.4 | 91.1 / 76.8 / 70.9 | 99.0 | 37.1 (C 64), 36.2 (C 256) | 0.0060 | 39 / 66 | 174 | – |
| JevK5 (v0.3, E15's weights) | BF16 | ok | 70.3 | 86.4 / 62.4 / 67.3 | 99.4 | 19.3 (C 64), 18.4 (C 256) | 0.0116 | 63 / 154 | 155 | 109 wide-choice reads |
| JevK5 (v0.3, E15's weights) | FP8 | ok | 70.1 | 86.8 / 62.0 / 66.9 | 96.4 | 25.1 (C 64), 24.6 (C 256) | 0.0089 | 46 / 105 | 170 | 109 wide-choice reads |
| JevK5 v0.2 (the registered tag, not E15's weights) | BF16 | ok | 73.1 | 88.7 / 68.7 / 63.6 | 86.7 | 18.6 (C 64), 18.4 (C 256) | 0.0120 | 64 / 153 | 151 | 109 wide-choice reads |
| JevK5 v0.2 (the registered tag, not E15's weights) | FP8 | ok | 72.7 | 88.5 / 67.7 / 64.1 | 87.3 | 24.6 (C 64), 24.4 (C 256) | 0.0091 | 47 / 108 | 170 | 109 wide-choice reads |
| Clef-Flash 9B (scoring head) | – | not measured | – | – | – | – | – | – | – | – |
| Imajev-4B | – | not measured | – | – | – | – | – | – | – | – |
| OpenJev (DiffusionGemma-26B-A4B) | FP8 | ok | 79.1 | 91.3 / 79.8 / 62.9 | 96.1 | 33.9 (C 16), 45.0 (C 64) | 0.0115 | 144 / 162 | – | – |

## Not re-measured

- Imajev-4B: its read-out is a trained 255-row code readout over four rotated presentations of the options; no vLLM port was ready by the amendment's 00:30 UTC limit (the remaining host-hours went to the other rows), and its own server (about 2.9 s per decision as E15 logged it) does not fit the cap.
- Clef-Flash 9B (scoring head): not run for time. Its first attempt (21:53 UTC) stopped at load before any answer (the v2 venv has no image processor class for the checkpoint); the corrected stage (a new venv for the card's function, then about an hour of one-at-a-time 9B passes) would have ended after the time set for E30's end (about 1.5 h from 23:45 UTC), so the chain ended after JevK5 (execution as run).
- The llama.cpp rows' P = 16 passes (registration (f)'s saturation check) were not run (time; execution as run): the 27B models' rate and cost come from P = 8.
- Outside this entry (not drawn in Figure 4a): Clef 27B (withdrawn), Bosun, CLM, Laya, Verdict, E14's 0.8B heads (all under 70 %).

## Frontier of Figure 4a's drawn points on this basis

Open models at their E30 point (at least 68 %), Jev / OpenAI Decisions / Flash / Flash-Lite / Flash's own tool call at their E9 keys; E9's rule. By cost: sl08b -> ours08 -> ours -> jev -> sl27b -> flash.

## Reading

- Cheapest per decision: StartLux-Decision-0.8B at $0.0017 per 1,000 (FP8).
- AgentDecision-4B: FP8, 84.8 % (reference 85.1), 50.2 decisions/s at C = 256, $0.0045 per 1,000, 31.4x below E9's basis.
- Frozen GRPO 4B: FP8, 78.6 % (reference 78.8), 50.2 decisions/s at C = 256, $0.0045 per 1,000, 20.9x below E9's basis.
- StartLux-4B + our recipe: FP8, 85.5 % (reference 85.5), 30.3 decisions/s at C = 64, $0.0074 per 1,000, 39.3x below E9's basis.
- AgentDecision-0.8B (StartLux-0.8B + our recipe, E28): FP8, 80.5 % (reference 80.5), 122.5 decisions/s at C = 64, $0.0018 per 1,000, 5.5x below E9's basis.
- StartLux-Decision-0.8B: FP8, 69.3 % (reference 70.0), 131.3 decisions/s at C = 64, $0.0017 per 1,000, 7.6x below E9's basis.
- StartLux-Decision-2B: FP8, 76.0 % (reference 75.7), 81.6 decisions/s at C = 64, $0.0027 per 1,000, 8.0x below E9's basis.
- StartLux-Decision-4B: FP8, 83.0 % (reference 83.1), 33.0 decisions/s at C = 64, $0.0068 per 1,000, 7.7x below E9's basis.
- StartLux-Decision-9B: FP8, 83.3 % (reference 83.4), 19.5 decisions/s at C = 64, $0.0114 per 1,000, 7.3x below E9's basis.
- StartLux-Decision-27B: Q4_K_M (GGUF), 87.5 % (reference 87.7), 1.3 decisions/s at C = 8, $0.1659 per 1,000, 6.6x below E9's basis.
- Intern-Decision-4B: FP8, 73.5 % (reference 73.4), 18.0 decisions/s at C = 64, $0.0124 per 1,000, 3.1x below E9's basis.
- Qwen3.5-0.8B variant (E27, scoring head): BF16, 77.3 % (reference 77.2), 26.7 decisions/s at C = 16, $0.0084 per 1,000, 10.9x below E9's basis.
- OpenJev-27B: Q4_K_M (GGUF), 83.0 % (reference 82.7), 1.3 decisions/s at C = 8, $0.1775 per 1,000, 1.4x below E9's basis.
- decider-4b: FP8, 79.4 % (reference 79.3), 37.1 decisions/s at C = 64, $0.0060 per 1,000, 9.0x below E9's basis.
- JevK5 (v0.3, E15's weights): FP8, 70.1 % (reference 70.3), 25.1 decisions/s at C = 64, $0.0089 per 1,000, 5.4x below E9's basis.
- OpenJev (DiffusionGemma-26B-A4B): FP8, 79.1 % (reference 78.8), 45.0 decisions/s at C = 64, $0.0115 per 1,000, 7.9x below E9's basis.
