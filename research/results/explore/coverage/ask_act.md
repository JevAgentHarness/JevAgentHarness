# Ask-vs-act states on the common-state corpus C (descriptive, post hoc)

Generated 2026-10-06T05:41:40+00:00 at git `749cecfdf7cb` by `scripts/coverage_ask_act.py` from `results/stage2/s23` (PREREG_A7_STAGE2 A7.10 (c)); 200 eligible tasks. descriptive, post hoc (coverage of decision-model roles); S2-3's scoring on corpus C; not a registered analysis.

Accuracy = S2-3's score (answer in the logged acceptable tool set; mean over a task's occurrences, then over tasks). Strata: `ask_now` = the oracle's answer is `ask_user`; `ask_only` = `ask_user` is the only acceptable tool; `ask_optional` = `ask_user` acceptable, the oracle's answer another tool; `act` = `ask_user` not acceptable. Over-ask = share of `act` states answered `ask_user` (task-weighted).

| backend | all (5,314 / 200 tasks) | after_off_plan (386 / 57 tasks) | ask_now (582 / 100 tasks) | ask_only (267 / 100 tasks) | ask_optional (167 / 25 tasks) | act (4,565 / 200 tasks) | over-ask on act |
|---|---|---|---|---|---|---|---|
| JevAgentRL-4B (frozen) | 99.2 | 94.7 | 100.0 | 100.0 | 100.0 | 98.9 | 0.8 |
| JevAgentRL-v2 (exploratory) | 99.4 | 95.0 | 100.0 | 100.0 | 100.0 | 99.2 | 0.6 |
| SFT v1 (JevAgentRL recipe before RL) | 99.2 | 93.8 | 100.0 | 100.0 | 100.0 | 98.9 | 0.8 |
| untrained base of JevAgentRL-4B | 50.1 | 26.7 | 47.3 | 24.0 | 60.2 | 54.9 | 1.5 |
| Jev | 91.5 | 81.5 | 64.4 | 27.0 | 100.0 | 94.2 | 1.2 |
| Intern-Decision-4B | 75.2 | 47.6 | 59.6 | 25.0 | 86.3 | 76.1 | 1.4 |
| StartLux-Decision 0.8B | 48.3 | 31.5 | 49.4 | 25.0 | 60.2 | 53.1 | 8.5 |
| StartLux-Decision 2B | 68.4 | 47.2 | 61.8 | 27.0 | 82.1 | 69.7 | 1.1 |
| StartLux-Decision 4B | 89.1 | 76.8 | 84.1 | 47.0 | 66.1 | 91.3 | 0.3 |
| StartLux-Decision 9B | 91.8 | 74.3 | 87.1 | 50.0 | 98.9 | 91.8 | 0.9 |
| StartLux-Decision 27B | 93.9 | 85.1 | 63.4 | 26.0 | 100.0 | 98.7 | 0.7 |

## Answers on the 582 `ask_now` states (pooled counts)

acted = a state-changing call (`cancel_order`, `refund`, `ship_order`) instead of asking.

| backend | asked | other acceptable read | acted | finished | other wrong tool | failed |
|---|---|---|---|---|---|---|
| JevAgentRL-4B (frozen) | 503 | 79 | 0 | 0 | 0 | 0 |
| JevAgentRL-v2 (exploratory) | 582 | 0 | 0 | 0 | 0 | 0 |
| SFT v1 (JevAgentRL recipe before RL) | 581 | 1 | 0 | 0 | 0 | 0 |
| untrained base of JevAgentRL-4B | 264 | 27 | 258 | 4 | 29 | 0 |
| Jev | 270 | 148 | 162 | 2 | 0 | 0 |
| Intern-Decision-4B | 120 | 269 | 166 | 1 | 26 | 0 |
| StartLux-Decision 0.8B | 301 | 1 | 279 | 0 | 1 | 0 |
| StartLux-Decision 2B | 137 | 260 | 169 | 1 | 15 | 0 |
| StartLux-Decision 4B | 156 | 310 | 109 | 2 | 5 | 0 |
| StartLux-Decision 9B | 162 | 311 | 105 | 3 | 1 | 0 |
| StartLux-Decision 27B | 102 | 313 | 165 | 2 | 0 | 0 |

## Answers on the 267 `ask_only` states (pooled counts)

acted = a state-changing call (`cancel_order`, `refund`, `ship_order`) instead of asking.

| backend | asked | other acceptable read | acted | finished | other wrong tool | failed |
|---|---|---|---|---|---|---|
| JevAgentRL-4B (frozen) | 267 | 0 | 0 | 0 | 0 | 0 |
| JevAgentRL-v2 (exploratory) | 267 | 0 | 0 | 0 | 0 | 0 |
| SFT v1 (JevAgentRL recipe before RL) | 267 | 0 | 0 | 0 | 0 | 0 |
| untrained base of JevAgentRL-4B | 96 | 0 | 166 | 1 | 4 | 0 |
| Jev | 105 | 0 | 162 | 0 | 0 | 0 |
| Intern-Decision-4B | 100 | 0 | 164 | 1 | 2 | 0 |
| StartLux-Decision 0.8B | 100 | 0 | 167 | 0 | 0 | 0 |
| StartLux-Decision 2B | 105 | 0 | 161 | 1 | 0 | 0 |
| StartLux-Decision 4B | 156 | 0 | 109 | 0 | 2 | 0 |
| StartLux-Decision 9B | 162 | 0 | 105 | 0 | 0 | 0 |
| StartLux-Decision 27B | 102 | 0 | 165 | 0 | 0 | 0 |

## Reproduction of the published C rows: 61 / 61 keys match

| backend | stratum | key | published | reproduced |
|---|---|---|---|---|
| rl | all | `stage2_rl/frozen/corpus_acc` | 99.2 | 99.2 |
| rl | all | `stage2_s1/frozen/corpus_acc` | 99.2 | 99.2 |
| rl | after_off_plan | `stage2_rl/frozen/corpus_acc_offplan` | 94.7 | 94.7 |
| rl | after_off_plan | `stage2_s1/frozen/corpus_acc_offplan` | 94.7 | 94.7 |
| rl | all | `diff/c/all/rl/acc` | 99.2 | 99.2 |
| rl | after_off_plan | `diff/c/after_off_plan/rl/acc` | 94.7 | 94.7 |
| rl | ask_now | `diff/c/ask_gold/rl/acc` | 100.0 | 100.0 |
| v2 | all | `stage2_v2/corpus_acc` | 99.4 | 99.4 |
| v2 | after_off_plan | `stage2_v2/corpus_acc_offplan` | 95.0 | 95.0 |
| v2 | all | `diff/c/all/v2/acc` | 99.4 | 99.4 |
| v2 | after_off_plan | `diff/c/after_off_plan/v2/acc` | 95.0 | 95.0 |
| v2 | ask_now | `diff/c/ask_gold/v2/acc` | 100.0 | 100.0 |
| sft_v1 | all | `stage2_rl/sft_v1/corpus_acc` | 99.2 | 99.2 |
| sft_v1 | after_off_plan | `stage2_rl/sft_v1/corpus_acc_offplan` | 93.8 | 93.8 |
| sft_v1 | all | `diff/c/all/sft_v1/acc` | 99.2 | 99.2 |
| sft_v1 | after_off_plan | `diff/c/after_off_plan/sft_v1/acc` | 93.8 | 93.8 |
| sft_v1 | ask_now | `diff/c/ask_gold/sft_v1/acc` | 100.0 | 100.0 |
| base | all | `stage2_rl/base/corpus_acc` | 50.1 | 50.1 |
| base | after_off_plan | `stage2_rl/base/corpus_acc_offplan` | 26.7 | 26.7 |
| base | all | `diff/c/all/base/acc` | 50.1 | 50.1 |
| base | after_off_plan | `diff/c/after_off_plan/base/acc` | 26.7 | 26.7 |
| base | ask_now | `diff/c/ask_gold/base/acc` | 47.3 | 47.3 |
| jev | all | `stage2_rl/jev/corpus_acc` | 91.5 | 91.5 |
| jev | all | `stage2_s1/jev/corpus_acc` | 91.5 | 91.5 |
| jev | after_off_plan | `stage2_rl/jev/corpus_acc_offplan` | 81.5 | 81.5 |
| jev | after_off_plan | `stage2_s1/jev/corpus_acc_offplan` | 81.5 | 81.5 |
| jev | all | `diff/c/all/jev/acc` | 91.5 | 91.5 |
| jev | after_off_plan | `diff/c/after_off_plan/jev/acc` | 81.5 | 81.5 |
| jev | ask_now | `diff/c/ask_gold/jev/acc` | 64.4 | 64.4 |
| intern | all | `stage2_rl/intern/corpus_acc` | 75.2 | 75.2 |
| intern | all | `stage2_s1/intern/corpus_acc` | 75.2 | 75.2 |
| intern | after_off_plan | `stage2_rl/intern/corpus_acc_offplan` | 47.6 | 47.6 |
| intern | after_off_plan | `stage2_s1/intern/corpus_acc_offplan` | 47.6 | 47.6 |
| intern | all | `diff/c/all/intern/acc` | 75.2 | 75.2 |
| intern | after_off_plan | `diff/c/after_off_plan/intern/acc` | 47.6 | 47.6 |
| intern | ask_now | `diff/c/ask_gold/intern/acc` | 59.6 | 59.6 |
| sl0.8b | all | `stage2_ladder/sl0.8b/corpus_acc` | 48.3 | 48.3 |
| sl0.8b | after_off_plan | `stage2_ladder/sl0.8b/corpus_acc_offplan` | 31.5 | 31.5 |
| sl0.8b | all | `diff/c/all/sl0.8b/acc` | 48.3 | 48.3 |
| sl0.8b | after_off_plan | `diff/c/after_off_plan/sl0.8b/acc` | 31.5 | 31.5 |
| sl0.8b | ask_now | `diff/c/ask_gold/sl0.8b/acc` | 49.4 | 49.4 |
| sl2b | all | `stage2_ladder/sl2b/corpus_acc` | 68.4 | 68.4 |
| sl2b | after_off_plan | `stage2_ladder/sl2b/corpus_acc_offplan` | 47.2 | 47.2 |
| sl2b | all | `diff/c/all/sl2b/acc` | 68.4 | 68.4 |
| sl2b | after_off_plan | `diff/c/after_off_plan/sl2b/acc` | 47.2 | 47.2 |
| sl2b | ask_now | `diff/c/ask_gold/sl2b/acc` | 61.8 | 61.8 |
| sl4b | all | `stage2_s1/sl4b/corpus_acc` | 89.1 | 89.1 |
| sl4b | after_off_plan | `stage2_s1/sl4b/corpus_acc_offplan` | 76.8 | 76.8 |
| sl4b | all | `diff/c/all/sl4b/acc` | 89.1 | 89.1 |
| sl4b | after_off_plan | `diff/c/after_off_plan/sl4b/acc` | 76.8 | 76.8 |
| sl4b | ask_now | `diff/c/ask_gold/sl4b/acc` | 84.1 | 84.1 |
| sl9b | all | `stage2_ladder/sl9b/corpus_acc` | 91.8 | 91.8 |
| sl9b | after_off_plan | `stage2_ladder/sl9b/corpus_acc_offplan` | 74.3 | 74.3 |
| sl9b | all | `diff/c/all/sl9b/acc` | 91.8 | 91.8 |
| sl9b | after_off_plan | `diff/c/after_off_plan/sl9b/acc` | 74.3 | 74.3 |
| sl9b | ask_now | `diff/c/ask_gold/sl9b/acc` | 87.1 | 87.1 |
| sl27b | all | `stage2_s1/sl27b/corpus_acc` | 93.9 | 93.9 |
| sl27b | after_off_plan | `stage2_s1/sl27b/corpus_acc_offplan` | 85.1 | 85.1 |
| sl27b | all | `diff/c/all/sl27b/acc` | 93.9 | 93.9 |
| sl27b | after_off_plan | `diff/c/after_off_plan/sl27b/acc` | 85.1 | 85.1 |
| sl27b | ask_now | `diff/c/ask_gold/sl27b/acc` | 63.4 | 63.4 |

## What these states can and cannot show

- C holds pre-action tool-choice states of four source agents on the first 200 fresh Stage-2 tasks; the ask states come from JevGym's clarify tasks only (one missing slot, asked once).
- Each state is scored as one replayed decision: C cannot show whether a wrong answer would have been recovered later in a live episode, nor the cost of a needless question to a real user.
- JevAgentRL (and SFT v1, v2) were trained on the same generator and oracle: their rows are in-generator.
- An 'acted' answer on an ask_now state is a state-changing call before the required question; JevGym's strict mode ends the episode at such a call, so these are irreversible errors in a live run (a replace-mode agent would make them; a cascade defers the low-confidence ones).
- The ask_now accuracy is the 'ask_user next' column of the difficulty appendix (Table tab:hardstates; same states, same scoring, checked above); new here are the answer kinds, the ask-only / ask-optional split and the over-ask rate.
- Descriptive and post hoc: no test, no interval; the strata were fixed in code before any answer was read.
