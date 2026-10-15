# E21: DecisionGym-Shift (exploratory, descriptive)

exploratory (E21), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-08 'E21: a shifted DecisionGym split (DecisionGym-Shift)'; outside every Holm table

Generated 2026-10-09T06:16:51+00:00 by `python scripts/explore_e21_shift.py summarize`. L2 = the shift corpus (LLM-only and Jev REPLACE trajectories on the 100 L2 tasks, built as C; task means); L3 = REPLACE at the tool choice, Flash, the 200 L3 tasks; in-domain = the logged C accuracy and live success on the first 200 Stage-2 tasks (the ladder rows of 'E21 addendum 2', Qwen3-4B untrained and + SFT v1: their Stage-2 rows, live on the first 100).

Corpus: 100 tasks, 1250 occurrences, 1093 questions, 129.0 options per question; subsets (occurrences / tasks): all 1250 / 100, mustask 100 / 60, offplan 81 / 25, renamed 1069 / 100, newkind 307 / 30, genkind 943 / 70, askany 263 / 60

## The shift table

| model | in-domain C | shift L2 | must-ask | after off-plan | renamed-tool | new kinds | generator kinds | surface cost | distraction cost | in-domain live | shift L3 (n) | L3 new kinds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Agent LLM alone (Flash) | – | – | – | – | – | – | – | – | – | 99.5 | 100.0 (200) | 100.0 |
| Qwen3-4B (untrained) | 50.1 | 52.9 | 27.6 | 79.8 | 44.7 | 65.1 | 47.6 | – | – | 13.0 | 13.5 (200) | 15.0 |
| Qwen3-4B + SFT v1 | 99.2 | 98.4 | 95.0 | 99.2 | 98.1 | 96.8 | 99.1 | – | – | 100.0 | 96.0 (200) | 86.7 |
| AgentDecision-4B | 99.2 | 98.6 | 90.8 | 99.2 | 98.3 | 99.3 | 98.3 | +1.1 | -0.2 | 100.0 | 96.0 (200) | 95.0 |
| AgentDecision-4B (mixed) | 99.1 | 94.9 | 67.4 | 98.4 | 93.8 | 93.7 | 95.3 | +3.7 | +0.1 | 100.0 | 91.5 (200) | 75.0 |
| StartLux-4B (mixed) | 99.6 | 98.5 | 100.0 | 100.0 | 98.0 | 94.9 | 100.0 | +1.4 | -0.8 | 100.0 | 98.0 (200) | 95.0 |
| 0.8B scoring head | 99.2 | 73.2 | 77.2 | 72.4 | 68.1 | 65.9 | 76.4 | +21.9 | -2.6 | 100.0 | 17.5 (200) | 0.0 |
| Jev | 91.5 | 91.8 | 58.3 | 94.4 | 90.2 | 95.6 | 90.2 | -2.8 | +0.1 | 83.0 | 82.0 (200) | 98.3 |
| OpenAI Decisions API | 90.9 | 86.1 | 41.7 | 95.8 | 83.4 | 84.8 | 86.7 | +0.6 | +2.7 | 78.5 | 66.0 (200) | 78.3 |
| StartLux-4B | 89.1 | 86.5 | 36.7 | 87.8 | 84.1 | 85.6 | 86.9 | +1.6 | -0.0 | 60.0 | 49.0 (200) | 28.3 |
| StartLux-9B | 91.8 | 90.9 | 55.0 | 96.4 | 89.6 | 91.0 | 90.8 | -0.9 | +0.7 | 74.0 | 69.5 (200) | 66.7 |
| StartLux-27B (Q8_0) | 93.9 | 94.7 | 65.0 | 98.4 | 93.8 | 95.6 | 94.3 | – | – | 63.0 | 79.9 (164) | 95.9 |
| OpenJev-27B | 92.4 | 92.4 | 58.3 | 92.6 | 91.1 | 94.3 | 91.6 | -0.1 | -0.0 | 63.0 | 69.5 (200) | 81.7 |
| Intern-Decision-4B | 75.2 | 75.9 | 33.9 | 83.4 | 71.4 | 78.8 | 74.7 | +0.6 | +1.9 | 26.5 | 34.0 (200) | 43.3 |

Axis cost = accuracy with the axis undone (re-rendered states) minus accuracy on the recorded states, in points (positive = the axis costs the model).

## Paired L2 deltas (task bootstrap, B = 2,000, seed 0)

| model | vs AgentDecision-4B | vs Jev | surface cost [95 %] | distraction cost [95 %] |
|---|---|---|---|---|
| Qwen3-4B (untrained) | -45.7 [-49.5, -41.9] | -38.9 [-43.4, -34.5] | – | – |
| Qwen3-4B + SFT v1 | -0.2 [-1.2, +0.9] | +6.6 [+4.6, +8.7] | – | – |
| AgentDecision-4B | – | +6.8 [+5.0, +8.6] | +1.1 [+0.2, +2.1] | -0.2 [-1.1, +0.7] |
| AgentDecision-4B (mixed) | -3.7 [-5.0, -2.5] | +3.1 [+1.6, +4.6] | +3.7 [+2.2, +5.3] | +0.1 [-0.9, +1.1] |
| StartLux-4B (mixed) | -0.1 [-1.4, +1.1] | +6.7 [+4.6, +8.9] | +1.4 [+0.6, +2.3] | -0.8 [-1.4, -0.2] |
| 0.8B scoring head | -25.4 [-28.8, -22.1] | -18.5 [-22.2, -14.9] | +21.9 [+18.9, +24.9] | -2.6 [-4.0, -1.2] |
| Jev | -6.8 [-8.6, -5.0] | – | -2.8 [-4.1, -1.5] | +0.1 [-0.6, +0.9] |
| OpenAI Decisions API | -12.5 [-14.7, -10.4] | -5.7 [-7.7, -3.9] | +0.6 [-1.0, +2.3] | +2.7 [+0.8, +4.4] |
| StartLux-4B | -12.1 [-14.7, -9.6] | -5.2 [-7.2, -3.4] | +1.6 [-0.1, +3.5] | -0.0 [-1.3, +1.2] |
| StartLux-9B | -7.7 [-9.8, -5.8] | -0.9 [-2.3, +0.5] | -0.9 [-1.8, -0.0] | +0.7 [+0.0, +1.5] |
| StartLux-27B (Q8_0) | -3.9 [-5.3, -2.4] | +3.0 [+1.5, +4.5] | – | – |
| OpenJev-27B | -6.2 [-8.0, -4.4] | +0.6 [-0.4, +1.6] | -0.1 [-1.2, +0.9] | -0.0 [-0.7, +0.6] |
| Intern-Decision-4B | -22.7 [-25.6, -19.7] | -15.9 [-18.4, -13.2] | +0.6 [-2.1, +3.2] | +1.9 [-0.5, +4.4] |

## L3 rows

| row | n | errored | success | own-state acc | steps | S1 / S2 calls | API $ / ep. | wall s | vs LLM-only [Newcombe] | vs Jev [Newcombe] | refusals |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Agent LLM alone (Flash) | 200 | 0 | 100.0 | – | 6.6 | 0.0 / 6.6 | 0.0535 | 8.0 | – | +18.0 [+12.9, +23.9] | 0 |
| Qwen3-4B (untrained) | 200 | 0 | 13.5 | 40.5 | 2.4 | 2.4 / 2.4 | 0.0033 | 3.2 | -86.5 [-90.6, -80.8] | -68.5 [-74.2, -61.2] | 0 |
| Qwen3-4B + SFT v1 | 200 | 0 | 96.0 | 96.8 | 6.5 | 6.5 / 6.5 | 0.0164 | 8.3 | -4.0 [-7.7, -1.3] | +14.0 [+7.7, +20.4] | 0 |
| AgentDecision-4B | 200 | 0 | 96.0 | 98.0 | 6.6 | 6.6 / 6.6 | 0.0169 | 8.2 | -4.0 [-7.7, -1.3] | +14.0 [+8.6, +19.8] | 0 |
| AgentDecision-4B (mixed) | 200 | 0 | 91.5 | 95.5 | 6.4 | 6.4 / 6.4 | 0.0161 | 8.5 | -8.5 [-13.2, -4.9] | +9.5 [+2.5, +16.4] | 0 |
| StartLux-4B (mixed) | 200 | 0 | 98.0 | 97.1 | 7.0 | 7.0 / 7.0 | 0.019 | 13.4 | -2.0 [-5.0, +0.2] | +16.0 [+10.4, +22.0] | 0 |
| 0.8B scoring head | 200 | 0 | 17.5 | 42.7 | 10.6 | 10.6 / 10.6 | 0.0408 | 10.5 | -82.5 [-87.1, -76.3] | -64.5 [-72.0, -55.0] | 0 |
| Jev | 200 | 0 | 82.0 | 87.3 | 6.2 | 6.2 / 6.2 | 0.0167 | 6.9 | -18.0 [-23.9, -12.9] | – | 0 |
| OpenAI Decisions API | 200 | 0 | 66.0 | 76.5 | 7.2 | 7.2 / 7.2 | 0.0229 | 10.1 | -34.0 [-40.8, -27.5] | -16.0 [-22.5, -9.5] | 0 |
| StartLux-4B | 200 | 0 | 49.0 | 80.7 | 5.4 | 5.4 / 5.4 | 0.0135 | 13.9 | -51.0 [-57.8, -43.9] | -33.0 [-39.3, -26.2] | 0 |
| StartLux-9B | 200 | 0 | 69.5 | 87.5 | 5.8 | 5.8 / 5.8 | 0.0147 | 24.5 | -30.5 [-37.2, -24.2] | -12.5 [-17.3, -7.8] | 0 |
| StartLux-27B (Q8_0) | 164 | 0 | 79.9 | 92.6 | 6.2 | 6.2 / 6.2 | 0.0159 | 132.7 | -20.1 [-26.9, -14.2] | -1.8 [-7.1, +3.3] | 0 |
| OpenJev-27B | 200 | 0 | 69.5 | 89.8 | 5.9 | 5.9 / 5.9 | 0.0154 | 47.9 | -30.5 [-37.2, -24.2] | -12.5 [-17.3, -7.8] | 0 |
| Intern-Decision-4B | 200 | 0 | 34.0 | 65.1 | 5.2 | 10.5 / 5.2 | 0.0128 | 14.4 | -66.0 [-72.2, -58.9] | -48.0 [-54.6, -40.4] | 0 |

L3 success per kind: Agent LLM alone (Flash): clarify.ask_after_lookup 100 (20), clarify.ask_twice 100 (20), clarify.method 100 (20), clarify.name 100 (20), clarify.order 100 (20), clarify.speed 100 (20), multi 100 (60), notify 100 (20); Jev: clarify.ask_after_lookup 100 (20), clarify.ask_twice 95 (20), clarify.method 25 (20), clarify.name 100 (20), clarify.order 100 (20), clarify.speed 0 (20), multi 100 (60), notify 100 (20); OpenAI Decisions API: clarify.ask_after_lookup 100 (20), clarify.ask_twice 50 (20), clarify.method 30 (20), clarify.name 90 (20), clarify.order 90 (20), clarify.speed 20 (20), multi 65 (60), notify 85 (20); AgentDecision-4B: clarify.ask_after_lookup 85 (20), clarify.ask_twice 100 (20), clarify.method 90 (20), clarify.name 100 (20), clarify.order 100 (20), clarify.speed 85 (20), multi 100 (60), notify 100 (20); 0.8B scoring head: clarify.ask_after_lookup 0 (20), clarify.ask_twice 0 (20), clarify.method 100 (20), clarify.name 75 (20), clarify.order 0 (20), clarify.speed 0 (20), multi 0 (60), notify 0 (20); AgentDecision-4B (mixed): clarify.ask_after_lookup 30 (20), clarify.ask_twice 95 (20), clarify.method 100 (20), clarify.name 100 (20), clarify.order 100 (20), clarify.speed 95 (20), multi 98 (60), notify 100 (20); StartLux-9B: clarify.ask_after_lookup 90 (20), clarify.ask_twice 15 (20), clarify.method 0 (20), clarify.name 100 (20), clarify.order 95 (20), clarify.speed 0 (20), multi 100 (60), notify 95 (20); OpenJev-27B: clarify.ask_after_lookup 95 (20), clarify.ask_twice 50 (20), clarify.method 0 (20), clarify.name 100 (20), clarify.order 50 (20), clarify.speed 0 (20), multi 100 (60), notify 100 (20); StartLux-27B (Q8_0): clarify.ask_after_lookup 100 (16), clarify.ask_twice 88 (16), clarify.method 0 (17), clarify.name 100 (16), clarify.order 76 (17), clarify.speed 41 (17), multi 100 (48), notify 100 (17); StartLux-4B: clarify.ask_after_lookup 10 (20), clarify.ask_twice 15 (20), clarify.method 0 (20), clarify.name 95 (20), clarify.order 35 (20), clarify.speed 0 (20), multi 92 (60), notify 60 (20); Intern-Decision-4B: clarify.ask_after_lookup 45 (20), clarify.ask_twice 40 (20), clarify.method 5 (20), clarify.name 60 (20), clarify.order 55 (20), clarify.speed 0 (20), multi 30 (60), notify 45 (20); StartLux-4B (mixed): clarify.ask_after_lookup 85 (20), clarify.ask_twice 100 (20), clarify.method 100 (20), clarify.name 100 (20), clarify.order 100 (20), clarify.speed 95 (20), multi 100 (60), notify 100 (20); Qwen3-4B (untrained): clarify.ask_after_lookup 45 (20), clarify.ask_twice 0 (20), clarify.method 0 (20), clarify.name 0 (20), clarify.order 90 (20), clarify.speed 0 (20), multi 0 (60), notify 0 (20); Qwen3-4B + SFT v1: clarify.ask_after_lookup 60 (20), clarify.ask_twice 100 (20), clarify.method 100 (20), clarify.name 100 (20), clarify.order 100 (20), clarify.speed 100 (20), multi 100 (60), notify 100 (20)

## Lead of the trained rows (best trained minus best off-the-shelf, points)

- l2_all: AgentDecision-4B 98.6 vs StartLux-27B (Q8_0) 94.7: gap +3.9, keeps the lead: True
- l2_mustask: StartLux-4B (mixed) 100.0 vs StartLux-27B (Q8_0) 65.0: gap +35.0, keeps the lead: True
- l2_offplan: StartLux-4B (mixed) 100.0 vs StartLux-27B (Q8_0) 98.4: gap +1.6, keeps the lead: True
- l2_renamed: AgentDecision-4B 98.3 vs StartLux-27B (Q8_0) 93.8: gap +4.6, keeps the lead: True
- l2_newkind: AgentDecision-4B 99.3 vs StartLux-27B (Q8_0) 95.6: gap +3.6, keeps the lead: True
- l2_genkind: StartLux-4B (mixed) 100.0 vs StartLux-27B (Q8_0) 94.3: gap +5.7, keeps the lead: True
- l3: StartLux-4B (mixed) 98.0 vs Jev 82.0: gap +16.0, keeps the lead: True
- l3_newkind: AgentDecision-4B 95.0 vs Jev 98.3: gap -3.3, keeps the lead: False
- in_c: StartLux-4B (mixed) 99.6 vs StartLux-27B (Q8_0) 93.9: gap +5.7, keeps the lead: True
- in_live: AgentDecision-4B 100.0 vs Jev 83.0: gap +17.0, keeps the lead: True
- cost_surface: trained mean +7.0, off-the-shelf mean -0.2, difference +7.2 (n 4 / 6)
- cost_distraction: trained mean -0.9, off-the-shelf mean +0.9, difference -1.8 (n 4 / 6)

## Wrong answers on the recorded states, by kind (post hoc, descriptive; occurrences)

- Qwen3-4B (untrained): added distractor 4, ask 24, finish 215, generator distractor 38, other core tool 374; at must-ask states: finish 3, generator distractor 8, other core tool 48
- Qwen3-4B + SFT v1: ask 5, generator distractor 1, other core tool 8; at must-ask states: other core tool 4
- AgentDecision-4B: ask 1, generator distractor 1, other core tool 10; at must-ask states: other core tool 8
- AgentDecision-4B (mixed): ask 7, generator distractor 6, other core tool 31; at must-ask states: other core tool 27
- StartLux-4B (mixed): ask 1, generator distractor 12, other core tool 1; at must-ask states: none
- 0.8B scoring head: ask 78, finish 31, generator distractor 21, other core tool 195; at must-ask states: other core tool 23
- Jev: ask 7, generator distractor 18, other core tool 47; at must-ask states: other core tool 32
- OpenAI Decisions API: added distractor 1, ask 15, generator distractor 42, other core tool 83; at must-ask states: generator distractor 12, other core tool 30
- StartLux-4B: generator distractor 30, other core tool 103; at must-ask states: other core tool 45
- StartLux-9B: finish 1, generator distractor 13, other core tool 75; at must-ask states: generator distractor 1, other core tool 33
- StartLux-27B (Q8_0): added distractor 1, generator distractor 14, other core tool 30; at must-ask states: generator distractor 1, other core tool 27
- OpenJev-27B: generator distractor 1, other core tool 64; at must-ask states: other core tool 32
- Intern-Decision-4B: added distractor 13, ask 4, generator distractor 60, other core tool 189; at must-ask states: added distractor 1, generator distractor 2, other core tool 45

## Kendall tau-b (in-domain vs shift)

- c_vs_l2: 0.56 (n 11: e12, e12c, e14a, intern, jev, oaidec, openjevhf, rl, sl27b, sl4b, sl9b)
- live_vs_l3: 0.49 (n 11: e12, e12c, e14a, intern, jev, oaidec, openjevhf, rl, sl27b, sl4b, sl9b)
- l2_vs_l3: 0.84 (n 11: e12, e12c, e14a, intern, jev, oaidec, openjevhf, rl, sl27b, sl4b, sl9b)
- c_vs_l3: 0.55 (n 11: e12, e12c, e14a, intern, jev, oaidec, openjevhf, rl, sl27b, sl4b, sl9b)
- l2_vs_l3_ladder: 0.88 (n 13: base, e12, e12c, e14a, intern, jev, oaidec, openjevhf, rl, sft_v1, sl27b, sl4b, sl9b)

## Spend

- direct Gemini $61.87 (this container $25.15, the host $36.71, probes $0.008); OpenAI $1.23 (replays $0.82); Jev $0.84 (replays $0.52); OpenRouter other than the Jev key $0.

## Records

- answers per model and rendering: `results/explore/e21_shift/c/answers_<slug>[__<variant>].jsonl.gz`; live slim copies: `results/explore/e21_shift/live/`; host records: `results/explore/e21_shift/box/`.

- base: 1093 answers, 0 errors, 0 refusals, 0 missing, logged $0.0000, median 1155.8762110000202 ms
- e12: 1093 answers, 0 errors, 0 refusals, 0 missing, logged $0.0000, median 1221.0776539996004 ms
- e12c: 1093 answers, 0 errors, 0 refusals, 0 missing, logged $0.0000, median 1751.0803730001498 ms
- e14a: 1093 answers, 0 errors, 0 refusals, 0 missing, logged $0.0000, median 340.2348280001206 ms
- intern: 1093 answers, 0 errors, 0 refusals, 0 missing, logged $0.0508, median 569.4995169978938 ms
- jev: 1093 answers, 0 errors, 0 refusals, 0 missing, logged $0.1940, median 231.28290100021331 ms
- oaidec: 1093 answers, 0 errors, 0 refusals, 0 missing, logged $0.3108, median 148.92159700002594 ms
- openjevhf: 1093 answers, 0 errors, 0 refusals, 0 missing, logged $0.0000, median 8298.735427997599 ms
- rl: 1093 answers, 0 errors, 0 refusals, 0 missing, logged $0.0000, median 1225.136361000068 ms
- sft_v1: 1093 answers, 0 errors, 0 refusals, 0 missing, logged $0.0000, median 1223.5917530000506 ms
- sl27b: 1093 answers, 0 errors, 0 refusals, 0 missing, logged $0.0000, median 22049.814006000815 ms
- sl4b: 1093 answers, 0 errors, 0 refusals, 0 missing, logged $0.0000, median 2569.1772409991245 ms
- sl9b: 1093 answers, 0 errors, 0 refusals, 0 missing, logged $0.0000, median 4125.871485999596 ms
