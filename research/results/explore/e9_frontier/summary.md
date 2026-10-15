# E9: the cost-accuracy frontier of multiple-choice decisions (exploratory, descriptive)

exploratory (E9), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-07 'E9: cost-accuracy frontier of multiple-choice decisions'; outside every Holm table.

Generated 2026-10-08T14:45:03+00:00 at git 0f49df238eeb by `python scripts/explore_e9_frontier.py summarize`. Inventory and cost rules: `DATA.md`.

## Main readout: E1's 3,991 L1 decisions (no new call)

Items: BFCL tool choice 1,114, BFCL arguments 1,949, When2Call 928. Cost per decision: API models as logged (list price, reasoning tokens included); open-weights models $2.24208 per GPU-hour × logged serving time per decision (request latency / questions in the request; four requests in flight, so an upper bound). Latency: median request latency as logged on each model's host. Frontier = no drawn model matches or beats it on both cost and accuracy.

| model | kind | accuracy % [Wilson 95 %] | BFCL tool choice | BFCL arguments | When2Call | $ per 1,000 decisions | median latency ms | on the frontier |
|---|---|---|---|---|---|---|---|---|
| StartLux-0.8B | dm (gpu) | 70.0 [68.5, 71.4] | 88.2 | 67.1 | 54.2 | 0.0127 | 17 | yes |
| Jev | dm (api) | 86.4 [85.3, 87.4] | 89.9 | 89.6 | 75.4 | 0.0163 | 206 | yes |
| StartLux-2B | dm (gpu) | 75.7 [74.4, 77.0] | 89.9 | 73.1 | 64.1 | 0.0218 | 50 | no |
| Intern-Decision-4B | dm (gpu) | 73.4 [72.0, 74.7] | 88.2 | 71.2 | 60.1 | 0.0389 | 91 | no |
| StartLux-4B | dm (gpu) | 83.1 [81.9, 84.3] | 91.3 | 81.2 | 77.4 | 0.0524 | 111 | no |
| StartLux-9B | dm (gpu) | 83.4 [82.3, 84.6] | 92.2 | 81.0 | 78.1 | 0.0830 | 182 | no |
| AgentDecision-4B | dm (gpu) | 78.8 [77.5, 80.0] | 88.4 | 85.0 | 54.0 | 0.0934 | 268 | no |
| StartLux-27B (Q8_0) | dm (gpu) | 87.7 [86.7, 88.7] | 90.8 | 89.1 | 81.2 | 1.0922 | 2689 | no |
| Gemini 3.5 Flash-Lite | llm (api) | 83.3 [82.1, 84.5] | 90.7 | 89.0 | 62.6 | 0.1861 | 681 | no |
| Flash, own function call | fc (api) | 78.7 [77.4, 79.9] | 84.2 | 96.3 | 35.2 | 0.7272 | 884 | no |
| Gemini 3.5 Flash | llm (api) | 88.1 [87.1, 89.1] | 90.3 | 93.6 | 74.0 | 0.8108 | 2570 | yes |
| BM25 | ctrl (api) | 39.4 [37.9, 40.9] | 61.4 | 31.5 | 29.5 | 0.0000 | 1 | not drawn |
| TF-IDF | ctrl (api) | 39.9 [38.4, 41.4] | 62.3 | 31.6 | 30.4 | 0.0000 | 1 | not drawn |
| Gemini embedding | ctrl (api) | 51.7 [50.1, 53.2] | 79.3 | 40.0 | 43.2 | 0.0256 | 196 | not drawn |
| Qwen3-4B, untrained | attr (gpu) | 71.4 [70.0, 72.8] | 87.5 | 70.5 | 53.9 | 0.0930 | 266 | not drawn |
| SFT only (v1) | attr (gpu) | 78.4 [77.1, 79.7] | 86.9 | 85.5 | 53.3 | 0.0944 | 270 | not drawn |
| Flash-Lite, label + confidence | llm (api) | 84.1 [82.9, 85.2] | 90.4 | 89.7 | 64.9 | 0.1363 | 582 | not drawn |
| Gemma 4 26B (unpriced) | llm (api) | 80.2 [79.0, 81.4] | 89.1 | 87.4 | 54.4 | unpriced | 1739 | not drawn |

Frontier (by cost): StartLux-0.8B → Jev → Gemini 3.5 Flash.

Against each LLM (Jev − LLM, points, paired item bootstrap B = 2,000; cost ratio = LLM / Jev; the cheapest decision model at least as accurate as the LLM and how many times cheaper it is):

| LLM | Jev − LLM [95 % CI] | LLM cost / Jev cost | cheapest decision model ≥ LLM accuracy | cost ratio |
|---|---|---|---|---|
| Gemini 3.5 Flash | -1.7 [-2.6, -0.9] | 49.7× | none | – |
| Gemini 3.5 Flash-Lite | +3.1 [+2.0, +4.0] | 11.4× | Jev | 11.4× |
| Flash-Lite, label + confidence | +2.3 [+1.3, +3.2] | 8.4× | Jev | 8.4× |
| Flash, own function call | +7.7 [+6.2, +9.0] | 44.6× | Jev | 44.6× |

Most accurate decision model: StartLux-27B (Q8_0) (87.7 %), -0.4 points vs Flash [-1.3, +0.5].

Sensitivity (open-weights cost only; accuracy unchanged):

- `zero`: frontier = StartLux-27B (Q8_0) → Gemini 3.5 Flash; open-weights $ per 1,000 decisions: Intern-Decision-4B 0.0000, StartLux-0.8B 0.0000, StartLux-2B 0.0000, StartLux-4B 0.0000, StartLux-9B 0.0000, StartLux-27B (Q8_0) 0.0000, AgentDecision-4B 0.0000.
- `token_tariff`: frontier = Intern-Decision-4B → StartLux-9B → StartLux-27B (Q8_0) → Gemini 3.5 Flash; open-weights $ per 1,000 decisions: Intern-Decision-4B 0.0030, StartLux-0.8B 0.0106, StartLux-2B 0.0106, StartLux-4B 0.0106, StartLux-9B 0.0106, StartLux-27B (Q8_0) 0.0106, AgentDecision-4B 0.0108.
- provisioned per-token tariff `rl` (results/stage2/tariff.json): $0.0942 / M System-1 input tokens (q = $0.00196182 × 100 episodes / 2,083,666 tokens; $7.97e-05 per System-1 call).
- provisioned per-token tariff `sl4b` (results/stage2/tariff_sl4b.json): $0.0429 / M System-1 input tokens (q = $0.002167344 × 100 episodes / 5,051,739 tokens; $1.37e-04 per System-1 call).

## The check: E1's 299-item development prefix (new calls)

Items: 299 (the registered dev prefix, excluded from C8). Spend: $0.59 (smoke $0.05, full $0.54); cap $30. Every row measured in this session (a new answer cache).

| decider | accuracy % [Wilson 95 %] | BFCL tool choice | BFCL arguments | When2Call | $ per 1,000 decisions | median latency ms | errors | Jev − decider [95 % CI] | cost / Jev |
|---|---|---|---|---|---|---|---|---|---|
| Jev | 91.3 [87.6, 94.0] | 96.5 | 95.1 | 77.8 | 0.0168 | 202 | 0 | – | – |
| GPT-6-Luna | 84.6 [80.1, 88.3] | 94.1 | 91.5 | 59.7 | 0.0445 | 1147 | 0 | +6.7 [+3.7, +10.0] | 2.6× |
| Gemini 3.5 Flash-Lite | 86.3 [81.9, 89.7] | 95.3 | 93.7 | 61.1 | 0.1854 | 767 | 0 | +5.0 [+1.7, +8.0] | 11.0× |
| Gemini 3.5 Flash | 91.6 [87.9, 94.3] | 94.1 | 97.2 | 77.8 | 0.8137 | 1122 | 0 | -0.3 [-2.7, +2.0] | 48.4× |
| GPT-6.1-Sol | 90.6 [86.8, 93.4] | 94.1 | 95.8 | 76.4 | 0.9063 | 2322 | 0 | +0.7 [-2.0, +3.3] | 53.9× |

Frontier of the check (by cost): Jev → Gemini 3.5 Flash.

## Reading

- On E1, the frontier holds 2 decision model(s) (StartLux-0.8B, Jev) and 1 LLM point(s) (Gemini 3.5 Flash); every frontier point cheaper than the most accurate LLM is a decision model: yes.
- Gemini 3.5 Flash (88.1 %, $0.811 per 1,000): Jev is -1.7 points [-2.6, -0.9] at 1/50 of its cost; no decision model is as accurate.
- Gemini 3.5 Flash-Lite (83.3 %, $0.186 per 1,000): Jev is +3.1 points [+2.0, +4.0] at 1/11 of its cost; the cheapest decision model at least as accurate is Jev, 11.4× cheaper.
- Flash-Lite, label + confidence (84.1 %, $0.136 per 1,000): Jev is +2.3 points [+1.3, +3.2] at 1/8 of its cost; the cheapest decision model at least as accurate is Jev, 8.4× cheaper.
- Flash, own function call (78.7 %, $0.727 per 1,000): Jev is +7.7 points [+6.2, +9.0] at 1/45 of its cost; the cheapest decision model at least as accurate is Jev, 44.6× cheaper.
- Sensitivity `zero`: Jev is not on the frontier; frontier = StartLux-27B (Q8_0), Gemini 3.5 Flash; only decision models below the most accurate LLM: yes.
- Sensitivity `token_tariff`: Jev is not on the frontier; frontier = Intern-Decision-4B, StartLux-9B, StartLux-27B (Q8_0), Gemini 3.5 Flash; only decision models below the most accurate LLM: yes.
- Check (299 dev items): Jev − Gemini 3.5 Flash -0.3 points [-2.7, +2.0], Gemini 3.5 Flash costs 48.4× Jev.
- Check (299 dev items): Jev − Gemini 3.5 Flash-Lite +5.0 points [+1.7, +8.0], Gemini 3.5 Flash-Lite costs 11.0× Jev.
- Check (299 dev items): Jev − GPT-6-Luna +6.7 points [+3.7, +10.0], GPT-6-Luna costs 2.6× Jev.
- Check (299 dev items): Jev − GPT-6.1-Sol +0.7 points [-2.0, +3.3], GPT-6.1-Sol costs 53.9× Jev.
- Check frontier: Jev, Gemini 3.5 Flash.

## E12's and E12c's points (exploratory E12 / E12c; E9's frontier above is unchanged)

AgentDecision-4B (mixed, E12): 85.1 % [83.9, 86.1] (BFCL tool choice 89.3, BFCL arguments 90.5, When2Call 68.5), $0.1397 per 1,000 decisions (E9's rule, its own host), median latency 344 ms; vs AgentDecision-4B +6.3 points [+5.1, +7.5], vs StartLux-4B +1.9 [+0.7, +3.1], vs Jev -1.3 [-2.4, -0.3]. On the frontier (alone with E9's drawn set): no (dominated by Jev).
  Host adjustment (the frozen checkpoint re-served on the same host: $0.1428 vs $0.0934 per 1,000 as logged; factor 0.654): $0.0914 per 1,000; on the frontier: no.
StartLux-4B (mixed, E12c): 85.5 % [84.3, 86.5] (BFCL tool choice 89.5, BFCL arguments 91.9, When2Call 67.1), $0.2946 per 1,000 decisions (E9's rule, its own host), median latency 713 ms; vs AgentDecision-4B +6.7 points [+5.4, +8.1], vs StartLux-4B +2.3 [+1.1, +3.6], vs Jev -0.9 [-2.1, +0.2]. On the frontier (alone with E9's drawn set): no (dominated by Jev).
  Host adjustment (the frozen checkpoint re-served on the E12 host (same instance type); factor 0.654): $0.1927 per 1,000; on the frontier: no.
AgentDecision-4B (mixed+, E16): 84.3 % [83.2, 85.4] (BFCL tool choice 88.8, BFCL arguments 90.1, When2Call 66.9), $0.0909 per 1,000 decisions (E9's rule, its own host), median latency 260 ms; vs AgentDecision-4B +5.6 points [+4.4, +6.8], vs StartLux-4B +1.2 [-0.1, +2.4], vs Jev -2.1 [-3.2, -1.0]. On the frontier (alone with E9's drawn set): no (dominated by Jev).
  Host adjustment (the frozen checkpoint re-served on the same host: $0.0925 vs $0.0934 per 1,000 as logged; factor 1.010): $0.0918 per 1,000; on the frontier: no.
StartLux-9B (mixed, E16): 84.8 % [83.7, 85.9] (BFCL tool choice 90.0, BFCL arguments 91.9, When2Call 63.6), $0.2404 per 1,000 decisions (E9's rule, its own host), median latency 604 ms; vs AgentDecision-4B +6.1 points [+4.6, +7.5], vs StartLux-4B +1.7 [+0.5, +3.0], vs Jev -1.6 [-2.8, -0.5]. On the frontier (alone with E9's drawn set): no (dominated by Jev).
  Host adjustment (the frozen checkpoint re-served on the same host: $0.0925 vs $0.0934 per 1,000 as logged; factor 1.010): $0.2427 per 1,000; on the frontier: no.
Frontier with the new point(s): StartLux-0.8B → Jev → Gemini 3.5 Flash; changed: no.
