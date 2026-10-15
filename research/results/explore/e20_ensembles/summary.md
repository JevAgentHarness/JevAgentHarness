# E20: decision-model ensembles and a hosted decision API behind the gate

Generated 2026-10-08T15:46:37+00:00 at git `92fc96518012` by `python scripts/explore_e20_ensembles.py summarize`.

**Part A** (post hoc, descriptive; stored answers only (no model call, no GPU); revises no registered result; no registration entry (the convention of scripts/explore_drift.py)): dev iff int(sha256('e20-split-v1|' + key)[:16], 16) % 2 == 0; C split by task, L1 by item (`split.json`); members and m chosen on the development half (`choice.json`), every number below on the held-out half. No LLM is a voter.

## Part A, common states C: held-out half (2,821 occurrences, 108 tasks)

| family | rule | members | acc % | must-ask % | predicted live % | gate coverage % | accepted errors | accepted acc % | own-confidence errors at equal coverage |
|---|---|---|---|---|---|---|---|---|---|
| ots | best_single | sl27b | 93.9 | 25.5 | 78.5 | | | | |
| ots | majority | sl27b, openjev | 94.9 | 47.3 | 81.0 | | | | |
| ots | weighted | sl27b, openjev | 94.9 | 47.3 | 81.0 | | | | |
| ots | gate 4 of 4 | sl27b, openjevhf, sl9b, openjev | | | 86.1 | 88.8 | 63 | 97.5 | sl27b 50; openjevhf 70; sl9b 59; openjev 110; rl 0; e14a 0 |
| mixed | best_single | e12c | 99.9 | 100.0 | 99.5 | | | | |
| mixed | majority | e12c, sl0.8b, clm, e16b, openjev | 99.5 | 100.0 | 99.1 | | | | |
| mixed | weighted | e12c, sl0.8b, clm, e16b, openjev | 99.6 | 100.0 | 99.1 | | | | |
| mixed | gate 2 of 2 | e12c, e16b | | | 100.0 | 99.0 | 3 | 99.9 | e12c 1; e16b 5; rl 11; e14a 16 |
| hosted | best_single | oaidec | 90.6 | 37.3 | 77.5 | | | | |
| hosted | majority | oaidec, jev | 92.2 | 31.5 | 75.0 | | | | |
| hosted | weighted | oaidec, jev | 92.2 | 31.5 | 75.0 | | | | |
| hosted | gate 2 of 2 | oaidec, jev | | | 79.9 | 88.2 | 103 | 95.9 | oaidec 158; jev 101; rl 0; e14a 0 |
| ours | rl at its threshold 0.4999999999978404 | AgentDecision-4B | 99.7 | 100.0 | 100.0 | 100.0 | 12 | 99.6 | |
| ours | e14a at its threshold 0.9999999997650646 | 0.8B scoring head (E14 A) | 99.5 | 100.0 | 99.5 | 78.2 | 0 | 100.0 | |
| pair | AgentDecision-4B + Jev (weighted; agreement gate) | rl, jev | 99.7 | 100.0 | 100.0 | 88.4 | 1 | 100.0 | |

Hosted pair as the live backend gates it (both agree and the smaller confidence >= t) against each model's own confidence at the same coverage (accepted errors):

| t | coverage % | accepted errors (agree) | Decisions API alone | Jev alone |
|---|---|---|---|---|
| 0.0 | 88.2 | 103 | 158.0 | 101.0 |
| 0.5 | 76.1 | 82 | 110.2 | 62.0 |
| 0.7 | 55.5 | 56 | 66.0 | 39.1 |
| 0.9 | 25.6 | 21 | 18.6 | 0.8 |
| 0.95 | 14.7 | 14 | 13.4 | 0.5 |
| 0.99 | 4.3 | 0 | 2.3 | 0.1 |
| 1.0 | 1.3 | 0 | 0.5 | 0.0 |

### Error correlation (all of C, no selection)

| models | median Jaccard of error sets | if independent | ratio | same wrong answer among joint errors % | must-ask states failed by all % | by half % | by none % |
|---|---|---|---|---|---|---|---|
| all_ots (18) | 24.9 | 7.1 | 1.8 | 73.1 | 39.3 | 62.5 | 0.0 |
| top6_ots (6) | 43.8 | 3.3 | 13.4 | 86.3 | 39.3 | 59.2 | 37.5 |

Must-ask states (267 occurrences): OpenJev-27B fails 164, StartLux-27B 165, both 162 (the same wrong tool on 162, in 73 tasks).

## Part A, L1 fixed decisions (E1): held-out half (2,069 decisions)

| family | rule | members | acc % | must-ask % | predicted live % | gate coverage % | accepted errors | accepted acc % | own-confidence errors at equal coverage |
|---|---|---|---|---|---|---|---|---|---|
| ots | best_single | sl27b | 87.7 | n/a | n/a | | | | |
| ots | majority | sl27b, clef, jev, clefflash | 86.4 | n/a | n/a | | | | |
| ots | weighted | sl27b, clef, jev, clefflash, laya | 86.4 | n/a | n/a | | | | |
| ots | gate 8 of 8 | sl27b, jev, clef, sl9b, sl4b, openjevhf, clefflash, oaidec | | | n/a | 68.9 | 94 | 93.4 | sl27b 57; jev 70; clef 122; sl9b 102; sl4b 118; openjevhf 141; clefflash 165; oaidec 164; rl 307; e14a 404 |
| mixed | best_single | sl27b | 87.7 | n/a | n/a | | | | |
| mixed | majority | sl27b, e12, e16b, clef, jev, verdict | 87.8 | n/a | n/a | | | | |
| mixed | weighted | sl27b, e12, e16b, clef, jev, verdict | 88.0 | n/a | n/a | | | | |
| mixed | gate 7 of 7 | sl27b, jev, e12, clef, e12c, e16b, e16a | | | n/a | 68.6 | 57 | 96.0 | sl27b 55; jev 69; e12 77; clef 122; e12c 71; e16b 116; e16a 82; rl 307; e14a 403 |
| hosted | best_single | jev | 85.6 | n/a | n/a | | | | |
| hosted | majority | jev, oaidec | 84.1 | n/a | n/a | | | | |
| hosted | weighted | jev, oaidec | 84.1 | n/a | n/a | | | | |
| hosted | gate 2 of 2 | jev, oaidec | | | n/a | 86.8 | 224 | 87.5 | jev 164; oaidec 311; rl 350; e14a 537 |
| ours | rl at its threshold 0.4999999999978404 | AgentDecision-4B | 78.2 | n/a | n/a | 99.9 | 450 | 78.2 | |
| ours | e14a at its threshold 0.9999999997650646 | 0.8B scoring head (E14 A) | 67.4 | n/a | n/a | 33.8 | 191 | 72.7 | |
| pair | AgentDecision-4B + Jev (weighted; agreement gate) | rl, jev | 78.9 | n/a | n/a | 80.3 | 156 | 90.6 | |

Hosted pair as the live backend gates it (both agree and the smaller confidence >= t) against each model's own confidence at the same coverage (accepted errors):

| t | coverage % | accepted errors (agree) | Decisions API alone | Jev alone |
|---|---|---|---|---|
| 0.0 | 86.8 | 224 | 311.5 | 164.4 |
| 0.5 | 74.8 | 143 | 207.6 | 98.0 |
| 0.7 | 65.2 | 94 | 153.3 | 55.0 |
| 0.9 | 50.8 | 52 | 76.5 | 22.3 |
| 0.95 | 43.3 | 31 | 47.8 | 15.2 |
| 0.99 | 29.0 | 10 | 19.3 | 7.6 |
| 1.0 | 21.0 | 4 | 11.6 | 5.5 |

## Part A, DecisionGym-Hard (E7c shadow calibration arms, the states both reached)

392 common labelled tool-choice states (Jev's arm 3699, the 4B's 3367).

| states | n (tasks) | Jev % | AgentDecision-4B % | vote (larger confidence) % | agreement gate coverage % | accepted errors | accepted acc % |
|---|---|---|---|---|---|---|---|
| tier | 97 (61) | 100.0 | 79.4 | 83.5 | 79.4 | 0 | 100.0 |
| other | 295 (91) | 100.0 | 99.0 | 99.0 | 99.0 | 0 | 100.0 |
| all | 392 (91) | 100.0 | 94.1 | 95.2 | 94.1 | 0 | 100.0 |

Transfer of the chosen vote members across levels (held-out half): ots/majority/c_members_on_l1 84.8; ots/weighted/c_members_on_l1 84.8; mixed/majority/c_members_on_l1 83.8; mixed/weighted/c_members_on_l1 84.4; hosted/majority/c_members_on_l1 84.1; hosted/majority/l1_members_on_c 92.3; hosted/weighted/c_members_on_l1 84.1; hosted/weighted/l1_members_on_c 92.3

## Part B (exploratory (E20), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-08 'E20: decision-model ensembles and a hosted decision API behind the gate'; outside every Holm table)

### Thresholds on the calibration states (rl_shadow_cal)

| row | point | tau-hat | n | accuracy % | coverage at tau-hat % | accepted errors | agreement % |
|---|---|---|---|---|---|---|---|
| oaidec | tool_select | inf | 1125 | 93.7 | 0.0 | 0 | n/a |
| oaidec | tool_params | inf | 1328 | 86.8 | 0.0 | 0 | n/a |
| agree | tool_select | inf | 1125 | 94.9 | 0.0 | 0 | 89.5 |
| agree | tool_params | 0.99 | 1328 | 91.9 | 50.2 | 0 | 75.7 |

Descriptive (no fit): on the same labelled states

- tool_select: n 1125, the live LLM right on 1125; oaidec right on 1054 (confidence 1.0: 34 answers, 3 wrong); jev right on 1060 (confidence 1.0: 353 answers, 1 wrong); members agree on 1007 (963 right; both confidences >= 0.99 on 49, 2 wrong)
- tool_params: n 1328, the live LLM right on 1244; oaidec right on 1153 (confidence 1.0: 626 answers, 12 wrong); jev right on 1272 (confidence 1.0: 870 answers, 0 wrong); members agree on 1206 (1152 right; both confidences >= 0.99 on 666, 0 wrong)

Live rows: oaidec: E20 (c): both fitted thresholds always defer (the row equals LLM-only; not run live); agree: E20 amendment (the fitted thresholds): the tool choice always defers and the argument cascade is reached only through a System-1 tool choice, so the row makes no System-1 call and equals LLM-only; not run live

Spend (calibration + every live record, smokes included): gemini $0.00 of $25, openai $0.33 of $10, jev $0.21 of $3

