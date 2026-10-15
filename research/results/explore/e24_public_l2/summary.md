# E24: agent-state accuracy (L2) on public benchmarks -- Part 1, L2-ALFWorld (exploratory (E24), descriptive)

- **descriptive; no test; revises nothing.** Generated 2026-10-09T06:47:57+00:00 at git `6b820ba3cf4a` by `scripts/explore_e24_public_l2.py`.
- Corpus: 3355 labelled states (3263 distinct questions) from 274 games of Flash alone (valid_seen: E11's run, 1660 states; valid_unseen: E23's run, 1695 states). Replay check: {'games_desynced': 0, 'obs_mismatch': 0, 'options_mismatch': 0, 'steps': 3824}; dropped states by reason: {'after_expert_fallback': 327, 'expert_random_fallback': 142}.
- Labels: **acceptable** (lead, amendment) = the lenient set plus every action after which ALFWorld's hand-coded expert needs fewer steps to win (consequence check by replay); lenient = the strict action plus the expert's equally valid choices; **strict** = ALFWorld's single expert action (agreement with the expert; a lower bound); logged = the gold the live runs logged (unpatched expert). Score: game means (every game weighs the same); bootstrap 95 % over games.
- Consequence check: 9120 (state, action) pairs evaluated, 1411 accepted; expert baseline on 1997 of 2843 evaluated states; game errors 0.
- Set sizes on the sampled states (every option evaluated): seen: n 40, options 33.2, lenient 4.65, acceptable 8.32 (25.4 % of options); unseen: n 40, options 25.9, lenient 2.98, acceptable 3.88 (15.0 % of options)

## L2-ALFWorld accuracy %, game means

| model | group | acceptable seen | acceptable unseen | acceptable pooled [95 %] | lenient seen / unseen | strict (expert agreement) seen / unseen | logged seen / unseen | after off-plan / on plan | errors | missing pairs | $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ALFWorld-trained 4B (E11) | alfworld | 93.9 | 91.5 | 92.8 [91.3, 94.1] | 89.2 / 87.9 | 83.2 / 85.9 | 79.2 / 80.5 | 96.9 / 66.8 | 0 | 0 | 0.000 |
| Qwen3-4B (untrained base) | offshelf | 64.7 | 57.3 | 61.1 [58.3, 63.8] | 57.4 / 50.1 | 45.8 / 39.0 | 45.0 / 38.6 | 64.5 / 37.1 | 0 | 0 | 0.000 |
| AgentDecision-4B (released) | transfer | 66.5 | 62.9 | 64.7 [61.6, 67.7] | 59.2 / 55.8 | 47.7 / 45.9 | 47.4 / 45.5 | 69.5 / 31.1 | 0 | 0 | 0.000 |
| 0.8B scoring head | transfer | 49.2 | 48.4 | 48.8 [46.1, 51.4] | 38.5 / 39.1 | 30.6 / 29.8 | 30.8 / 30.4 | 47.8 / 44.3 | 0 | 0 | 0.000 |
| StartLux-4B | offshelf | 69.2 | 63.6 | 66.5 [63.5, 69.6] | 61.7 / 56.8 | 49.0 / 46.1 | 49.8 / 46.0 | 69.1 / 45.1 | 0 | 0 | 0.000 |
| StartLux-9B | offshelf | 74.9 | 70.6 | 72.8 [70.1, 75.5] | 67.8 / 63.3 | 55.6 / 54.3 | 55.6 / 52.7 | 74.0 / 54.4 | 0 | 0 | 0.000 |
| StartLux-27B (Q8_0) | offshelf | 73.1 | 68.2 | 70.7 [68.0, 73.3] | 67.2 / 63.6 | 55.7 / 54.7 | 55.4 / 53.1 | 74.8 / 37.1 | 0 | 0 | 0.000 |
| Jev | offshelf | 70.5 | 65.3 | 67.9 [64.8, 70.9] | 61.6 / 58.5 | 49.7 / 48.3 | 50.2 / 49.0 | 70.1 / 46.7 | 0 | 0 | 0.124 |
| OpenAI Decisions API | offshelf | 72.0 | 64.2 | 68.2 [65.2, 71.1] | 62.9 / 56.9 | 49.7 / 46.4 | 50.4 / 46.2 | 71.7 / 41.4 | 0 | 0 | 0.205 |
| tau^2-trained 0.8B head (E22) | tau2 | not run | | | | | | | | | |
| Flash (the agent LLM's own action; in sample) | reference | 81.1 | 80.8 | 80.9 [78.2, 83.8] | 74.9 / 71.7 | 59.3 / 58.1 | 59.6 / 57.7 | 85.3 / 54.9 | -- | 0 | -- |

## By task type (acceptable, pooled seen + unseen)

| model | place | look | clean | heat | cool | pick2 |
|---|---|---|---|---|---|---|
| ALFWorld-trained 4B (E11) | 89.3 | 90.7 | 93.8 | 91.7 | 94.3 | 97.0 |
| Qwen3-4B (untrained base) | 77.1 | 40.8 | 54.3 | 55.8 | 50.1 | 80.1 |
| AgentDecision-4B (released) | 80.1 | 38.5 | 60.9 | 59.5 | 55.1 | 83.4 |
| 0.8B scoring head | 65.4 | 42.3 | 39.4 | 51.2 | 37.4 | 53.5 |
| StartLux-4B | 84.4 | 52.7 | 61.6 | 50.4 | 51.3 | 90.4 |
| StartLux-9B | 83.6 | 56.6 | 70.2 | 64.4 | 65.4 | 89.5 |
| StartLux-27B (Q8_0) | 78.2 | 72.1 | 63.7 | 56.5 | 67.3 | 85.9 |
| Jev | 83.2 | 35.0 | 64.8 | 63.1 | 60.4 | 88.2 |
| OpenAI Decisions API | 81.3 | 54.1 | 62.0 | 60.2 | 60.0 | 85.3 |
| Flash (the agent LLM's own action; in sample) | 88.3 | 80.5 | 70.6 | 82.1 | 72.0 | 94.2 |

## Live (quoted from E11, E23, E6a; not recomputed)

| model | split | L2 acceptable | L2 strict | live alone (REPLACE) | live cascade |
|---|---|---|---|---|---|
| ALFWorld-trained 4B (E11) | seen | 93.9 | 83.2 | 95.7 (E11) | 95.0 |
| ALFWorld-trained 4B (E11) | unseen | 91.5 | 85.9 | 88.1 (E23) | 89.6 |
| Jev | seen | 70.5 | 49.7 | 27.1 (E6A) | 91.4 |
| Flash alone | seen | 81.1 (own states) | 59.3 | 90.7 (E11) | -- |
| Flash alone | unseen | 80.8 (own states) | 58.1 | 88.8 (E23) | -- |

## Kendall tau-b, L2 vs live alone (descriptive)

- acceptable: across (model, split) cells with a live row: tau-b 1.00 (n = 3); within e11 (split x task type): 0.08 (n = 12); within jev (split x task type): 0.07 (n = 6); within pooled (split x task type): 0.49 (n = 18)
- strict: across (model, split) cells with a live row: tau-b 0.33 (n = 3); within e11 (split x task type): -0.38 (n = 12); within jev (split x task type): 0.07 (n = 6); within pooled (split x task type): 0.31 (n = 18)

## Reading rules (registration (d) 5, on the acceptable label)

- spread (pooled, best - worst): 44.0 points (e11 vs e14a); ranking ['e11', 'sl9b', 'sl27b', 'oaidec', 'jev', 'sl4b', 'e12', 'qwen4b', 'e14a']
- best_alfworld: e11 92.8
- best_transfer: e12 64.7
- best_offshelf: sl9b 72.8
- lead_over_transfer_pt: +28.1
- lead_over_offshelf_pt: +20.0
- seen-to-unseen change (points): e11 -2.4, qwen4b -7.4, e12 -3.6, e14a -0.8, sl4b -5.6, sl9b -4.2, sl27b -4.8, jev -5.2, oaidec -7.8

## Part 2: L2-tau^2 (E22's test episodes; write-due states; task means)

- 422 labelled states of 46 tasks (422 tool questions, 1445 argument questions); by domain {"airline": {"arg_not_asked": 144, "arg_other_label": 4, "arg_questions": 424, "not_write_due": 924, "sims": 80, "turns": 1023, "write_due": 99}, "retail": {"arg_not_asked": 396, "arg_other_label": 14, "arg_questions": 1021, "not_write_due": 1523, "sims": 160, "turns": 1846, "write_due": 323}}

| model | tool retail | tool airline | tool pooled [95 %] | complete call pooled | wrong write pooled | arguments pooled | errors | $ |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B (untrained base) | 65.1 | 40.4 | 59.2 [49.9, 68.1] | 48.1 | 9.0 | 86.7 | 0 | 0.000 |
| AgentDecision-4B (released) | 68.2 | 45.3 | 62.7 [53.7, 71.0] | 53.4 | 9.7 | 89.1 | 0 | 0.000 |
| 0.8B scoring head | 68.9 | 39.2 | 61.8 [50.7, 71.6] | 24.0 | 16.1 | 64.5 | 0 | 0.000 |
| StartLux-4B | 64.7 | 41.7 | 59.2 [50.9, 67.3] | 48.0 | 8.9 | 79.8 | 0 | 0.000 |
| StartLux-9B | 63.9 | 44.6 | 59.3 [51.1, 67.1] | 55.2 | 7.6 | 93.0 | 0 | 0.000 |
| StartLux-27B (Q8_0) | 54.2 | 43.5 | 51.6 [43.8, 59.3] | 48.9 | 5.4 | 88.8 | 0 | 0.000 |
| Jev | 54.9 | 42.8 | 52.0 [43.9, 59.9] | 46.1 | 5.2 | 91.3 | 0 | 0.318 |
| OpenAI Decisions API | 39.6 | 31.5 | 37.7 [30.4, 44.9] | 34.8 | 3.3 | 90.2 | 0 | 0.612 |
| tau^2-trained 0.8B head (E22) | 47.0 | 45.2 | 46.6 [39.2, 54.0] | 42.7 | 4.6 | 92.1 | 0 | 0.000 |
| Flash (own action; in sample) | 49.8 | 41.0 | 47.7 [40.1, 55.1] | 46.0 | 4.0 | 98.1 | -- | -- |

- Kendall tau-b, L2-ALFWorld vs L2-tau^2 over the rows with both: -0.43 (n = 8)

## Spend and host

- hosted: {'jev': 0.4426714319999999, 'oaidec': 0.8173555999999981} (caps: Jev $3, OpenAI $3); Gemini $0.
- host: {"instance_id": "<instance-id>", "instance_type": "g6e.2xlarge", "az": "us-east-1d", "setup_utc": "2026-10-09T01:04:54Z", "gpu": "NVIDIA L40S, 46068 MiB", "mins_left": 404}; model runs: [{"slug": "e11", "corpus": "alfworld", "served": false, "minutes": 0}, {"slug": "e12", "corpus": "alfworld", "served": true, "minutes": 8, "used_h": 0.425}, {"slug": "e14a", "corpus": "alfworld", "served": true, "minutes": 5, "used_h": 0.511}, {"slug": "qwen4b", "corpus": "alfworld", "served": true, "minutes": 7, "used_h": 0.635}, {"slug": "sl4b", "corpus": "alfworld", "served": true, "minutes": 13, "used_h": 0.862}, {"slug": "e11", "corpus": "alfworld", "served": true, "minutes": 7, "used_h": 1.055}, {"slug": "sl9b", "corpus": "alfworld", "served": true, "minutes": 8, "used_h": 1.25}, {"slug": "sl27b", "corpus": "alfworld", "served": true, "minutes": 70, "used_h": 2.42}, {"slug": "e22", "corpus": "tau2", "served": true, "minutes": 4, "used_h": 2.527}, {"slug": "e12", "corpus": "tau2", "served": true, "minutes": 7, "used_h": 2.657}, {"slug": "e14a", "corpus": "tau2", "served": true, "minutes": 3, "used_h": 2.717}, {"slug": "qwen4b", "corpus": "tau2", "served": true, "minutes": 7, "used_h": 2.835}, {"slug": "sl4b", "corpus": "tau2", "served": true, "minutes": 26, "used_h": 3.282}, {"slug": "sl9b", "corpus": "tau2", "served": true, "minutes": 39, "used_h": 3.94}, {"slug": "sl27b", "corpus": "tau2", "served": true, "minutes": 165, "used_h": 6.696}]
- host hours: [{"final_utc": "2026-10-09T00:23:37Z", "used_h": 0.864}, {"final_utc": "2026-10-09T00:42:28Z", "used_h": 1.057}, {"final_utc": "2026-10-09T02:23:31Z", "used_h": 2.422}, {"final_utc": "2026-10-09T06:40:05Z", "used_h": 6.698}]
