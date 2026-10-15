# E15: more decision-model baselines (exploratory, descriptive)

exploratory (E15), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-07 'E15: more decision-model baselines'; outside every Holm table.

Generated 2026-10-08T03:17:50+00:00 at git 269a102418e8 by `python scripts/explore_e15_baselines.py summarize`. L1 = E1's 3,991 decisions (E9's cost rule: API at logged list price; open weights at $2.24208 per GPU-hour x logged serving time per decision); L2 = the common states C (5,314 occurrences, 200 tasks; task means); L3 = REPLACE at the tool choice, Flash, the first 200 fresh Stage-2 tasks.

## Candidates

| model | licence | served by | read-out | wire check (299 dev items): acc / errors / refusals |
|---|---|---|---|---|
| OpenAI Decisions API (gpt-6-luna) | proprietary API | OpenAI, POST /v1/decisions (public beta) | hosted option scores | 85.6 / 0 / 1 |
| Clef-Flash 9B | Apache-2.0 | the card's systemone() (bf16) | joint schema head | 86.6 / 0 / 0 |
| Clef 27B (8-bit) | Apache-2.0 | the card's systemone() (LLM.int8 backbone) | joint schema head | 89.6 / 0 / 0 |
| OpenJev (DiffusionGemma-26B-A4B) | Apache-2.0 | razorback16/openjev:0.5.0 | masked-slot read of a diffusion LM | 79.9 / 0 / 0 |
| OpenJev 27B (HF, FP8) | CC BY-NC 4.0 (weights) | vLLM 0.29.0 + the card's helper | first-token letter scores | 86.0 / 0 / 0 |
| JevK5 v0.2 (4B) | Apache-2.0 | razorback16/openjev-jevk5:0.5.1 | answer-letter logits | 74.9 / 0 / 0 |
| CLM v0.1 (8B) | Apache-2.0 | razorback16/openjev-clm:0.5.1 | contrastive heads on embeddings | 47.2 / 0 / 0 |
| Laya (421M) | Apache-2.0 | razorback16/openjev-laya:0.5.1 | encoder head, 1,024 tokens | 51.5 / 0 / 0 |
| Verdict 1.4 (151M) | Apache-2.0 | razorback16/openjev-verdict:0.5.1 | encoder head, 512 tokens | 46.8 / 0 / 0 |
| decider-4b v2.1 | Apache-2.0 | its own server (decider.serve) | answer-letter logits | 85.3 / 0 / 0 |
| Imajev-4B | Apache-2.0 | its own server (4 option orders, calibration) | option-code logits | 83.3 / 0 / 0 |
| Bosun 3.1 1.7B | Apache-2.0 | the card's model.predict() | decision-token slots | 65.2 / 0 / 0 |

Dropped: clef (27B, 8-bit) C and live: not run: its C run (about 1.6 s per request) would have used the GPU time reserved for the live rows and was stopped after 15 answers (set aside unused); the author then withdrew the 27B's remaining levels (2026-10-08). L1 only.; codiv hosted OpenJev endpoint: not used: it needs a sign-up (registration (b) 2); the self-hosted image and the HF checkpoint ran instead.; GPT-6-Luna / GPT-6.1-Sol / GPT-6-Astra as Responses-API deciders: not E15 rows (the author: not decision models); only the Decisions API was run.; written: 2026-10-08T03:17:39+00:00.

## The three levels

| model | L1 acc % [Wilson] | tool choice / args / When2Call | $ / 1,000 | median ms | L1 − Jev [95 %] | BFCL tool choice (static) | C acc | after off-plan | ask-vs-act | ECE | C − Jev [95 %] | live success | live − Jev [Newcombe] | errors L1 / L2 | refusals L1 / L2 / L3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenAI Decisions API (gpt-6-luna) | 79.2 [77.9, 80.4] | 90.6 / 81.4 / 60.7 | 0.0274 | 158 | -7.2 [-8.3, -6.2] | 90.6 | 90.9 | 74.3 | 75.9 | 0.193 | -0.5 [-1.5, +0.5] | 78.5 (n 200) | -4.5 [-10.9, +1.9] | 0 / 0 | 10 / 0 / 0 |
| Clef-Flash 9B | 81.4 [80.1, 82.6] | 87.5 / 85.1 / 66.2 | 0.1086 | 249 | -5.0 [-6.1, -4.0] | 87.5 | 47.7 | 48.8 | 25.0 | 0.252 | -43.8 [-45.4, -42.0] | – | – | 0 / 0 | 0 / 0 / – |
| Clef 27B (8-bit) | 84.0 [82.8, 85.1] | 86.3 / 88.8 / 71.3 | 0.6694 | 1578 | -2.4 [-3.3, -1.5] | 86.3 | – | – | – | – | – | – | – | 0 / – | 0 / – / – |
| OpenJev (DiffusionGemma-26B-A4B) | 78.8 [77.5, 80.1] | 91.2 / 79.8 / 62.0 | 0.0908 | 244 | -7.6 [-8.6, -6.5] | 91.2 | 91.1 | 82.4 | 87.5 | 0.030 | -0.4 [-1.6, +0.9] | 62.5 (n 200) | -20.5 [-26.2, -14.8] | 0 / 0 | 0 / 0 / 0 |
| OpenJev 27B (HF, FP8) | 82.7 [81.5, 83.9] | 90.2 / 81.6 / 76.2 | 0.2382 | 542 | -3.7 [-4.6, -2.8] | 90.2 | 92.4 | 80.1 | 63.5 | 0.021 | +0.9 [+0.1, +1.8] | 63.0 (n 200) | -20.0 [-25.7, -14.4] | 0 / 0 | 0 / 0 / 0 |
| JevK5 v0.2 (4B) | 70.3 [68.8, 71.7] | 86.4 / 62.4 / 67.3 | 0.0476 | 111 | -16.1 [-17.5, -14.9] | 86.4 | 65.0 | 50.0 | 60.0 | 0.228 | -26.5 [-29.4, -23.6] | – | – | 0 / 0 | 0 / 0 / – |
| CLM v0.1 (8B) | 46.2 [44.7, 47.8] | 71.0 / 41.1 / 27.0 | 0.0314 | 80 | -40.2 [-41.8, -38.5] | 71.0 | 11.9 | 5.1 | 36.2 | 0.371 | -79.5 [-81.6, -77.3] | – | – | 0 / 0 | 0 / 0 / – |
| Laya (421M) | 49.4 [47.8, 50.9] | 79.8 / 39.8 / 32.9 | 0.0370 | 90 | -37.0 [-38.7, -35.4] | 79.8 | 11.0 | 13.2 | 1.5 | 0.303 | -80.5 [-82.5, -78.5] | – | – | 0 / 0 | 0 / 0 / – |
| Verdict 1.4 (151M) | 47.2 [45.6, 48.7] | 78.3 / 34.9 / 35.5 | 0.0218 | 53 | -39.2 [-40.8, -37.5] | 78.3 | 19.0 | 22.1 | 0.0 | 0.148 | -72.4 [-73.9, -71.0] | – | – | 11 / 0 | 0 / 0 / – |
| decider-4b v2.1 | 79.3 [78.0, 80.5] | 90.8 / 76.9 / 70.4 | 0.0539 | 100 | -7.1 [-8.2, -6.0] | 90.8 | 63.2 | 43.8 | 57.8 | 0.261 | -28.2 [-31.0, -25.3] | – | – | 0 / 0 | 0 / 0 / – |
| Imajev-4B | 78.9 [77.6, 80.1] | 87.9 / 77.6 / 70.6 | 1.1397 | 2874 | -7.5 [-8.6, -6.5] | 87.9 | 76.9 | 64.1 | 49.0 | 0.095 | -14.5 [-16.1, -13.0] | – | – | 0 / 0 | 0 / 0 / – |
| Bosun 3.1 1.7B | 63.3 [61.8, 64.8] | 85.8 / 56.4 / 50.8 | 0.1297 | 361 | -23.1 [-24.6, -21.5] | 85.8 | 36.3 | 29.5 | 49.2 | 0.290 | -55.2 [-57.3, -53.0] | – | – | 0 / 0 | 0 / 0 / – |

Reference rows (logged): Jev L1 86.4 %, C 91.5 %, live 83.0 %; AgentDecision-4B L1 78.8 %, C 99.2 %, live 100.0 % (E9 summary, explore/startlux_live).

Frontier with the E15 points (by cost): sl0.8b → jev → flash; changed from E9's (sl0.8b → jev → flash): no.
- OpenAI Decisions API (gpt-6-luna): on the frontier alone with E9's drawn set: no (dominated by jev).
- Clef-Flash 9B: on the frontier alone with E9's drawn set: no (dominated by jev, sl4b, sl9b); sensitivity zero: off; token_tariff: off.
- Clef 27B (8-bit): on the frontier alone with E9's drawn set: no (dominated by jev); sensitivity zero: off; token_tariff: off.
- OpenJev (DiffusionGemma-26B-A4B): on the frontier alone with E9's drawn set: no (dominated by jev, sl4b, sl9b); sensitivity zero: off; token_tariff: on.
- OpenJev 27B (HF, FP8): on the frontier alone with E9's drawn set: no (dominated by jev, lite, sl4b, sl9b); sensitivity zero: off; token_tariff: on.
- JevK5 v0.2 (4B): on the frontier alone with E9's drawn set: no (dominated by intern, jev, sl2b); sensitivity zero: off; token_tariff: off.
- CLM v0.1 (8B): on the frontier alone with E9's drawn set: no (dominated by jev, sl0.8b, sl2b); sensitivity zero: off; token_tariff: off.
- Laya (421M): on the frontier alone with E9's drawn set: no (dominated by jev, sl0.8b, sl2b); sensitivity zero: off; token_tariff: off.
- Verdict 1.4 (151M): on the frontier alone with E9's drawn set: no (dominated by jev, sl0.8b, sl2b); sensitivity zero: off; token_tariff: off.
- decider-4b v2.1: on the frontier alone with E9's drawn set: no (dominated by jev, sl4b); sensitivity zero: off; token_tariff: on.
- Imajev-4B: on the frontier alone with E9's drawn set: no (dominated by flash, jev, lite, sl27b, sl4b, sl9b); sensitivity zero: off; token_tariff: on.
- Bosun 3.1 1.7B: on the frontier alone with E9's drawn set: no (dominated by intern, jev, rl, sl0.8b, sl2b, sl4b, sl9b); sensitivity zero: off; token_tariff: off.

L3 selection (registration (c) 4): oaidec, openjevhf, openjev; L2 ranking of the GPU models: openjevhf, openjev, imajev, jevk5, decider, clefflash, bosun, verdict, clm, laya.

## RQ1: Kendall's tau-b over old and new rows

- old + new: static vs live 0.04, common vs live 0.62 (11 rows with all three: e15:oaidec, e15:openjev, e15:openjevhf, frozen, intern, jev, sl0.8b, sl27b, sl2b, sl4b, sl9b); static vs common 0.56 (19 rows); RQ1's reading holds (static-live below common-live and below 0.5): True.
- old rows only: static vs live 0.15, common vs live 0.71 (8 rows with all three: frozen, intern, jev, sl0.8b, sl27b, sl2b, sl4b, sl9b); static vs common 0.30 (8 rows); RQ1's reading holds (static-live below common-live and below 0.5): True.
- off the shelf (without AgentDecision-4B): static vs live 0.18, common vs live 0.54 (10 rows); static vs common 0.63 (18 rows); reading holds: True; live-success spread (best − worst) 76.5 points (jev − sl2b).
- without any trained model (no E12 / E12c / E14 row is in this set): static vs live 0.18, common vs live 0.54 (10 rows); static vs common 0.63 (18 rows); reading holds: True; live-success spread (best − worst) 76.5 points (jev − sl2b).
- live-success spread over every row with a live run: 93.5 points.

## Spend

OpenAI $1.13 (offline $0.86, live System-1 $0.27); direct Gemini (live) $9.93; OpenRouter $0; Jev $0. GPU: {'gpu_hours': 8.8, 'host_hours_at_final_stage': 8.797, 'instance_type': 'g6e.4xlarge', 'gpu': 'NVIDIA L40S 48 GB', 'instance_id': '<instance-id>', 'launched_utc': '2026-10-07T18:22:53Z', 'final_stage_utc': '2026-10-08T03:10:55Z', 'stop': 'self-initiated shutdown one minute after the final stage (instance-initiated shutdown behaviour: stop)', 'note': 'host hours from the first boot to the final stage; the host was never stopped between its three chains, so host hours = GPU hours; cap 12', 'written': '2026-10-08T03:16:20+00:00'}.
