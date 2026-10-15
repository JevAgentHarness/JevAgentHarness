# Why a decision model can match Jev on the common states yet lose live (drift analysis)

Generated 2026-10-08T05:29:19+00:00 at git `c8560a809adf` by `python scripts/explore_drift.py summarize`. post hoc, descriptive; stored live logs only (no API call, no GPU, no Jev call); revises no registered result; no registration entry (the convention of scripts/benchmark_difficulty.py). Live = REPLACE at the tool choice, Gemini 3.5 Flash, the first 200 fresh Stage-2 DecisionGym tasks (seeds 300000-300199), latest record per task, errored tasks left out. C = the common-state corpus `results/stage2/s23` (3,939 distinct questions, 5,314 occurrences, 200 tasks; recorded from the LLM-only, Jev, Intern-Decision-4B and AgentDecision-4B live arms). A live state is *in C* when its corpus hash (`bench/stage2_corpus.question_hash`: the exact System-1 state, option list, option descriptions and question text) is a C question, else *novel*; novel states are split into *on plan* (every earlier executed tool was acceptable) and *after an off-plan step*. Correctness is the benchmark's rule (membership in the logged acceptable tool set). State accuracies are pooled over labelled decisions; the C column is the published task mean. Paper keys: `paper/generated/drift.tex` (`\res{drift/...}`).

The two E15 GPU live logs are read from S3 copies (`fetch`): openjev: `s3://<bucket>/e15_baselines/runs/e15_live_openjev/openjev_ts_replace_f.jsonl`; openjevhf: `s3://<bucket>/e15_baselines/runs/e15_live_openjevhf/openjevhf_ts_replace_f.jsonl`.

Rows with fewer live tasks than the others: Qwen3-4B (untrained) 100; SFT only (v1) 100.

## 1. Verification

The corpus hash recomputed on the C source arms' own logs (their labelled tool-choice states must be in C by construction):

| source arm | labelled states | in C | match % |
|---|---|---|---|
| llm_only | 1,498 | 1,498 | 100.0 |
| jev | 1,391 | 1,391 | 100.0 |
| intern | 925 | 925 | 100.0 |
| rl | 1,500 | 1,500 | 100.0 |

Published numbers against this script's reproduction (the C accuracy as the task mean over the corpus; live success over the latest records), the in-C labels against the corpus's, fall-throughs (System-1 answer not executed) and unlabelled states (excluded, as in C):

| model | C acc published | reproduced | live success published | reproduced | in-C label mismatches | fall-throughs | unlabelled |
|---|---|---|---|---|---|---|---|
| Jev | 91.5 | 91.5 | 83.0 | 83.0 | 0 | 0 | 0 |
| Intern-Decision-4B | 75.2 | 75.2 | 26.5 | 26.5 | 0 | 0 | 2 |
| StartLux-0.8B | 48.3 | 48.3 | 17.5 | 17.5 | 0 | 0 | 255 |
| StartLux-2B | 68.4 | 68.4 | 6.5 | 6.5 | 0 | 0 | 2 |
| StartLux-4B | 89.1 | 89.1 | 60.0 | 60.0 | 0 | 0 | 0 |
| StartLux-9B | 91.8 | 91.8 | 74.0 | 74.0 | 0 | 0 | 1 |
| StartLux-27B | 93.9 | 93.9 | 63.0 | 63.0 | 0 | 0 | 0 |
| OpenAI Decisions API (gpt-6-luna) | 90.9 | 90.9 | 78.5 | 78.5 | 0 | 0 | 10 |
| OpenJev (DiffusionGemma-26B-A4B) | 91.1 | 91.1 | 62.5 | 62.5 | 0 | 0 | 10 |
| OpenJev-27B (HF, FP8) | 92.4 | 92.4 | 63.0 | 63.0 | 0 | 0 | 0 |
| AgentDecision-4B (JevAgentRL-4B, frozen) | 99.2 | 99.2 | 100.0 | 100.0 | 0 | 0 | 0 |
| AgentDecision-4B + public data (E12) | 99.1 | 99.1 | 100.0 | 100.0 | 0 | 0 | 0 |
| StartLux-4B + our recipe (E12c) | 99.6 | 99.6 | 100.0 | 100.0 | 0 | 0 | 0 |
| 0.8B scoring head, agent states (E14 A) | 99.2 | 99.2 | 100.0 | 100.0 | 0 | 0 | 0 |
| 0.8B scoring head, generic mix (E14 G) | 23.9 | 23.9 | 0.0 | 0.0 | 0 | 0 | 0 |
| Qwen3-4B (untrained) | 50.1 | 50.1 | 13.0 | 13.0 | 0 | 0 | 18 |
| SFT only (v1) | 99.2 | 99.2 | 100.0 | 100.0 | 0 | 0 | 0 |
| JevAgentRL-v2 (exploratory) | 99.4 | 99.4 | 100.0 | 100.0 | 0 | 0 | 0 |

## 2. Main table (the paper's Table tab:drift)

| model | live success % | C offline acc % | live acc, in-C states % | live acc, novel states % | novel-state share % | first wrong choice: in C / novel / none (% of tasks) | irreversible first error (% of tasks) | wrong decisions below tau-hat % (tau-hat) | below 0.9: wrong / right % |
|---|---|---|---|---|---|---|---|---|---|
| Jev | 83.0 | 91.5 | 95.1 | n/a | 0.0 | 23.0 / 0.0 / 77.0 | 17.0 | 98.5 (1.00, own) | 86.8 / 18.9 |
| Intern-Decision-4B | 26.5 | 75.2 | 65.7 | n/a | 0.0 | 84.5 / 0.0 / 15.5 | 59.5 | 100.0 (always defer, own) | 82.0 / 60.9 |
| StartLux-0.8B | 17.5 | 48.3 | 52.1 | 9.2 | 75.2 | 65.5 / 17.5 / 17.0 | 28.0 | 100.0 (1.00, Jev arm) | 100.0 / 94.7 |
| StartLux-2B | 6.5 | 68.4 | 73.9 | 26.1 | 31.9 | 66.0 / 28.5 / 5.5 | 83.0 | 100.0 (1.00, Jev arm) | 97.8 / 88.9 |
| StartLux-4B | 60.0 | 89.1 | 89.0 | 96.5 | 44.6 | 37.5 / 4.5 / 58.0 | 35.0 | 100.0 (0.9756, own) | 73.4 / 43.1 |
| StartLux-9B | 74.0 | 91.8 | 94.7 | 94.4 | 47.8 | 19.5 / 6.5 / 74.0 | 25.5 | 100.0 (1.00, Jev arm) | 83.1 / 35.1 |
| StartLux-27B | 63.0 | 93.9 | 92.1 | 97.5 | 44.8 | 29.5 / 7.5 / 63.0 | 37.0 | 100.0 (1.00, Jev arm) | 62.2 / 3.4 |
| OpenAI Decisions API (gpt-6-luna) | 78.5 | 90.9 | 92.3 | 83.9 | 58.2 | 22.5 / 14.5 / 63.0 | 17.5 | 100.0 (1.00, Jev arm) | 92.9 / 66.6 |
| OpenJev (DiffusionGemma-26B-A4B) | 62.5 | 91.1 | 91.3 | 65.3 | 59.0 | 29.5 / 17.0 / 53.5 | 34.0 | 100.0 (1.00, Jev arm) | 52.4 / 31.0 |
| OpenJev-27B (HF, FP8) | 63.0 | 92.4 | 92.0 | 97.4 | 45.6 | 29.5 / 8.0 / 62.5 | 37.0 | 100.0 (1.00, Jev arm) | 26.7 / 2.6 |
| AgentDecision-4B (JevAgentRL-4B, frozen) | 100.0 | 99.2 | 100.0 | n/a | 0.0 | 0.0 / 0.0 / 100.0 | 0.0 | n/a (0.50, own) | n/a / 0.2 |
| AgentDecision-4B + public data (E12) | 100.0 | 99.1 | 100.0 | 100.0 | 51.0 | 0.0 / 0.0 / 100.0 | 0.0 | n/a (1.00, Jev arm) | n/a / 6.6 |
| StartLux-4B + our recipe (E12c) | 100.0 | 99.6 | 100.0 | 100.0 | 51.9 | 0.0 / 0.0 / 100.0 | 0.0 | n/a (1.00, Jev arm) | n/a / 9.1 |
| 0.8B scoring head, agent states (E14 A) | 100.0 | 99.2 | 100.0 | 100.0 | 52.3 | 0.0 / 0.0 / 100.0 | 0.0 | n/a (1.00, own) | n/a / 0.0 |
| 0.8B scoring head, generic mix (E14 G) | 0.0 | 23.9 | 11.5 | 0.0 | 3.6 | 96.0 / 4.0 / 0.0 | 100.0 | 100.0 (always defer, own) | 63.0 / 100.0 |
| Qwen3-4B (untrained) | 13.0 | 50.1 | 48.6 | 16.7 | 55.6 | 73.0 / 14.0 / 13.0 | 73.0 | 100.0 (1.00, Jev arm) | 100.0 / 100.0 |
| SFT only (v1) | 100.0 | 99.2 | 100.0 | 100.0 | 52.0 | 0.0 / 0.0 / 100.0 | 0.0 | n/a (1.00, Jev arm) | n/a / 0.0 |
| JevAgentRL-v2 (exploratory) | 100.0 | 99.4 | 100.0 | 100.0 | 50.3 | 0.0 / 0.0 / 100.0 | 0.0 | n/a (0.9987, own) | n/a / 0.0 |

## 3. (a) Faithfulness: the in-C states each model visited live, live answer against its offline C answer

| model | in-C states | live acc % | offline acc on the same states % | same answer % | live wrong, offline right | live right, offline wrong | sign test p | offline answer missing |
|---|---|---|---|---|---|---|---|---|
| Jev | 1,391 | 95.1 | 94.9 | 99.3 | 1 | 4 | 0.375 | 0 |
| Intern-Decision-4B | 925 | 65.7 | 65.7 | 100.0 | 0 | 0 | n/a | 0 |
| StartLux-0.8B | 284 | 52.1 | 52.1 | 100.0 | 0 | 0 | n/a | 0 |
| StartLux-2B | 514 | 73.9 | 73.9 | 100.0 | 0 | 0 | n/a | 0 |
| StartLux-4B | 683 | 89.0 | 89.0 | 100.0 | 0 | 0 | n/a | 0 |
| StartLux-9B | 739 | 94.7 | 94.7 | 100.0 | 0 | 0 | n/a | 0 |
| StartLux-27B | 746 | 92.1 | 92.1 | 100.0 | 0 | 0 | n/a | 0 |
| OpenAI Decisions API (gpt-6-luna) | 613 | 92.3 | 92.3 | 100.0 | 0 | 0 | n/a | 0 |
| OpenJev (DiffusionGemma-26B-A4B) | 677 | 91.3 | 91.1 | 96.9 | 7 | 8 | 1.000 | 0 |
| OpenJev-27B (HF, FP8) | 736 | 92.0 | 92.0 | 100.0 | 0 | 0 | n/a | 0 |
| AgentDecision-4B (JevAgentRL-4B, frozen) | 1,500 | 100.0 | 100.0 | 99.8 | 0 | 0 | n/a | 0 |
| AgentDecision-4B + public data (E12) | 735 | 100.0 | 100.0 | 100.0 | 0 | 0 | n/a | 0 |
| StartLux-4B + our recipe (E12c) | 722 | 100.0 | 100.0 | 100.0 | 0 | 0 | n/a | 0 |
| 0.8B scoring head, agent states (E14 A) | 715 | 100.0 | 100.0 | 100.0 | 0 | 0 | n/a | 0 |
| 0.8B scoring head, generic mix (E14 G) | 217 | 11.5 | 11.5 | 100.0 | 0 | 0 | n/a | 0 |
| Qwen3-4B (untrained) | 144 | 48.6 | 47.9 | 97.2 | 0 | 1 | 1.000 | 0 |
| SFT only (v1) | 360 | 100.0 | 100.0 | 100.0 | 0 | 0 | n/a | 0 |
| JevAgentRL-v2 (exploratory) | 746 | 100.0 | 100.0 | 100.0 | 0 | 0 | n/a | 0 |

## 4. (b) Drift: novel states and the first wrong tool choice

| model | states | in C % | novel on plan % | novel after off-plan % | acc in C % | acc novel on plan % | acc novel after off-plan % | acc in C after off-plan % (n) | tasks with a wrong choice | first wrong choice in C / novel | median step (share at step 0-1) | failed tasks | failed with a wrong choice % | recovered after a wrong choice % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev | 1,391 | 100.0 | 0.0 | 0.0 | 95.1 | n/a | n/a | 82.0 (122) | 46 | 46 / 0 | 1.0 (80.4) | 34 | 100.0 | 26.1 |
| Intern-Decision-4B | 925 | 100.0 | 0.0 | 0.0 | 65.7 | n/a | n/a | 43.9 (264) | 169 | 169 / 0 | 1 (51.5) | 147 | 100.0 | 13.0 |
| StartLux-0.8B | 1,145 | 24.8 | 8.6 | 66.6 | 52.1 | 64.6 | 2.0 | 0.0 (5) | 166 | 131 / 35 | 0.0 (90.4) | 165 | 100.0 | 0.6 |
| StartLux-2B | 755 | 68.1 | 13.1 | 18.8 | 73.9 | 42.4 | 14.8 | 0.0 (2) | 189 | 132 / 57 | 2 (32.8) | 187 | 100.0 | 1.1 |
| StartLux-4B | 1,232 | 55.4 | 42.1 | 2.4 | 89.0 | 98.3 | 66.7 | n/a (0) | 84 | 75 / 9 | 2.0 (36.9) | 80 | 100.0 | 4.8 |
| StartLux-9B | 1,416 | 52.2 | 46.0 | 1.8 | 94.7 | 98.0 | 0.0 | n/a (0) | 52 | 39 / 13 | 2.0 (1.9) | 52 | 100.0 | 0.0 |
| StartLux-27B | 1,352 | 55.2 | 44.8 | 0.0 | 92.1 | 97.5 | n/a | n/a (0) | 74 | 59 / 15 | 2.0 (0.0) | 74 | 100.0 | 0.0 |
| OpenAI Decisions API (gpt-6-luna) | 1,466 | 41.8 | 41.6 | 16.6 | 92.3 | 95.2 | 55.6 | 60.0 (5) | 74 | 45 / 29 | 1.0 (60.8) | 43 | 100.0 | 41.9 |
| OpenJev (DiffusionGemma-26B-A4B) | 1,650 | 41.0 | 29.1 | 29.9 | 91.3 | 92.9 | 38.3 | n/a (0) | 93 | 59 / 34 | 2 (5.4) | 75 | 100.0 | 19.4 |
| OpenJev-27B (HF, FP8) | 1,353 | 54.4 | 45.5 | 0.1 | 92.0 | 97.4 | 100.0 | n/a (0) | 75 | 59 / 16 | 2 (0.0) | 74 | 100.0 | 1.3 |
| AgentDecision-4B (JevAgentRL-4B, frozen) | 1,500 | 100.0 | 0.0 | 0.0 | 100.0 | n/a | n/a | n/a (0) | 0 | 0 / 0 | n/a (n/a) | 0 | n/a | n/a |
| AgentDecision-4B + public data (E12) | 1,500 | 49.0 | 51.0 | 0.0 | 100.0 | 100.0 | n/a | n/a (0) | 0 | 0 / 0 | n/a (n/a) | 0 | n/a | n/a |
| StartLux-4B + our recipe (E12c) | 1,500 | 48.1 | 51.9 | 0.0 | 100.0 | 100.0 | n/a | n/a (0) | 0 | 0 / 0 | n/a (n/a) | 0 | n/a | n/a |
| 0.8B scoring head, agent states (E14 A) | 1,500 | 47.7 | 52.3 | 0.0 | 100.0 | 100.0 | n/a | n/a (0) | 0 | 0 / 0 | n/a (n/a) | 0 | n/a | n/a |
| 0.8B scoring head, generic mix (E14 G) | 225 | 96.4 | 3.6 | 0.0 | 11.5 | 0.0 | n/a | n/a (0) | 200 | 192 / 8 | 0.0 (100.0) | 200 | 100.0 | 0.0 |
| Qwen3-4B (untrained) | 324 | 44.4 | 13.6 | 42.0 | 48.6 | 68.2 | 0.0 | 0.0 (1) | 87 | 73 / 14 | 1 (87.4) | 87 | 100.0 | 0.0 |
| SFT only (v1) | 750 | 48.0 | 52.0 | 0.0 | 100.0 | 100.0 | n/a | n/a (0) | 0 | 0 / 0 | n/a (n/a) | 0 | n/a | n/a |
| JevAgentRL-v2 (exploratory) | 1,500 | 49.7 | 50.3 | 0.0 | 100.0 | 100.0 | n/a | n/a (0) | 0 | 0 / 0 | n/a (n/a) | 0 | n/a | n/a |

First wrong tool choices on an in-C state: what the model answered offline on that very state, and the error types (the difficulty analysis's classifier):

| model | first errors in C | offline right on that state | offline wrong | offline gave the same wrong tool | error types of all first errors |
|---|---|---|---|---|---|
| Jev | 46 | 0 | 46 | 45 | wrong core tool 27; changed state before asking 9; unneeded question 9; other look-alike 1 |
| Intern-Decision-4B | 169 | 0 | 169 | 169 | wrong core tool 84; changed state before asking 55; look-alike of the gold tool 20; unneeded question 8; other look-alike 2 |
| StartLux-0.8B | 131 | 0 | 131 | 131 | unneeded question 109; wrong core tool 56; other look-alike 1 |
| StartLux-2B | 132 | 0 | 132 | 132 | wrong core tool 124; changed state before asking 58; look-alike of the gold tool 4; unneeded question 3 |
| StartLux-4B | 75 | 0 | 75 | 75 | changed state before asking 53; wrong core tool 31 |
| StartLux-9B | 39 | 0 | 39 | 39 | changed state before asking 50; unneeded question 1; wrong core tool 1 |
| StartLux-27B | 59 | 0 | 59 | 59 | changed state before asking 74 |
| OpenAI Decisions API (gpt-6-luna) | 45 | 0 | 45 | 45 | wrong core tool 39; other look-alike 14; changed state before asking 11; unneeded question 9; finished too early 1 |
| OpenJev (DiffusionGemma-26B-A4B) | 59 | 7 | 52 | 47 | changed state before asking 50; wrong core tool 20; look-alike of the gold tool 9; other look-alike 8; unrelated distractor 4; unneeded question 2 |
| OpenJev-27B (HF, FP8) | 59 | 0 | 59 | 59 | changed state before asking 74; wrong core tool 1 |
| AgentDecision-4B (JevAgentRL-4B, frozen) | 0 | 0 | 0 | 0 |  |
| AgentDecision-4B + public data (E12) | 0 | 0 | 0 | 0 |  |
| StartLux-4B + our recipe (E12c) | 0 | 0 | 0 | 0 |  |
| 0.8B scoring head, agent states (E14 A) | 0 | 0 | 0 | 0 |  |
| 0.8B scoring head, generic mix (E14 G) | 192 | 0 | 192 | 192 | wrong core tool 125; changed state before asking 75 |
| Qwen3-4B (untrained) | 73 | 0 | 73 | 70 | wrong core tool 73; unneeded question 5; look-alike of the gold tool 5; other look-alike 2; changed state before asking 2 |
| SFT only (v1) | 0 | 0 | 0 | 0 |  |
| JevAgentRL-v2 (exploratory) | 0 | 0 | 0 | 0 |  |

The ask-or-act states (the answer key is `ask_user`: 582 of C's occurrences, 11.0 %, in 100 tasks) and what each model did there live; the *must-ask* states among them (the acceptable set is exactly `ask_user`: 267 occurrences, 5.0 % of C, 100 tasks), how many a model met live (exposure: a model that asks earlier, where asking is one acceptable tool among others, never reaches the later must-ask state) and in how many tasks it asked early that way; the consequence of the first wrong tool choice (a wrong write action or a premature finish ends the task; a wrong lookup, an unneeded question or a read-only look-alike can be recovered from), with the share of such tasks that still succeeded:

| model | C ask-or-act acc % (task mean) | live ask-or-act states (in C) | live acc % | acted instead of asking (% of ask states) | must-ask states: C acc % (task mean) | live n (tasks) | live acc % | asked early (tasks) | first errors: wrong write / premature finish / recoverable | recovered % by kind (write / finish / recoverable) | irreversible first error: % of tasks, % of failed tasks | loss (% of tasks): at an irreversible first error / after a recoverable one / no wrong choice |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev | 64.4 | 93 (93) | 90.3 | 9.7 | 27.0 | 34 (34) | 73.5 | 25 | 34 / 0 / 12 | 0.0 / n/a / 100.0 | 17.0, 100.0 | 17.0 / 0.0 / 0.0 |
| Intern-Decision-4B | 59.6 | 180 (180) | 65.0 | 29.4 | 25.0 | 79 (78) | 31.6 | 10 | 119 / 0 / 50 | 0.0 / n/a / 44.0 | 59.5, 81.0 | 59.5 / 14.0 / 0.0 |
| StartLux-0.8B | 49.4 | 75 (75) | 100.0 | 0.0 | 25.0 | 25 (25) | 100.0 | 25 | 56 / 0 / 110 | 0.0 / n/a / 0.9 | 28.0, 33.9 | 28.0 / 54.5 / 0.0 |
| StartLux-2B | 61.8 | 177 (154) | 66.1 | 30.5 | 27.0 | 78 (78) | 33.3 | 2 | 166 / 0 / 23 | 0.0 / n/a / 8.7 | 83.0, 88.8 | 83.0 / 10.5 / 0.0 |
| StartLux-4B | 84.1 | 190 (177) | 72.1 | 27.9 | 47.0 | 90 (90) | 41.1 | 0 | 70 / 0 / 14 | 0.0 / n/a / 28.6 | 35.0, 87.5 | 35.0 / 5.0 / 0.0 |
| StartLux-9B | 87.1 | 200 (184) | 75.0 | 25.0 | 50.0 | 100 (100) | 50.0 | 0 | 51 / 0 / 1 | 0.0 / n/a / 0.0 | 25.5, 98.1 | 25.5 / 0.5 / 0.0 |
| StartLux-27B | 63.4 | 200 (185) | 63.0 | 37.0 | 26.0 | 100 (100) | 26.0 | 0 | 74 / 0 / 0 | 0.0 / n/a / n/a | 37.0, 100.0 | 37.0 / 0.0 / 0.0 |
| OpenAI Decisions API (gpt-6-luna) | 75.9 | 98 (97) | 88.8 | 11.2 | 38.5 | 38 (38) | 71.1 | 22 | 34 / 1 / 39 | 0.0 / 0.0 / 79.5 | 17.5, 81.4 | 17.5 / 4.0 / 0.0 |
| OpenJev (DiffusionGemma-26B-A4B) | 87.5 | 198 (181) | 74.7 | 25.3 | 50.0 | 98 (98) | 49.0 | 2 | 68 / 0 / 25 | 8.8 / n/a / 48.0 | 34.0, 82.7 | 31.0 / 6.5 / 0.0 |
| OpenJev-27B (HF, FP8) | 63.5 | 200 (185) | 63.0 | 37.0 | 26.0 | 100 (100) | 26.0 | 0 | 74 / 0 / 1 | 0.0 / n/a / 100.0 | 37.0, 100.0 | 37.0 / 0.0 / 0.0 |
| AgentDecision-4B (JevAgentRL-4B, frozen) | 100.0 | 109 (109) | 100.0 | 0.0 | 100.0 | 54 (54) | 100.0 | 0 | 0 / 0 / 0 | n/a / n/a / n/a | 0.0, n/a | 0.0 / 0.0 / 0.0 |
| AgentDecision-4B + public data (E12) | 99.9 | 100 (95) | 100.0 | 0.0 | 99.7 | 50 (50) | 100.0 | 0 | 0 / 0 / 0 | n/a / n/a / n/a | 0.0, n/a | 0.0 / 0.0 / 0.0 |
| StartLux-4B + our recipe (E12c) | 100.0 | 100 (96) | 100.0 | 0.0 | 100.0 | 50 (50) | 100.0 | 0 | 0 / 0 / 0 | n/a / n/a / n/a | 0.0, n/a | 0.0 / 0.0 / 0.0 |
| 0.8B scoring head, agent states (E14 A) | 100.0 | 100 (98) | 100.0 | 0.0 | 100.0 | 50 (50) | 100.0 | 0 | 0 / 0 / 0 | n/a / n/a / n/a | 0.0, n/a | 0.0 / 0.0 / 0.0 |
| 0.8B scoring head, generic mix (E14 G) | 25.0 | 75 (75) | 33.3 | 66.7 | 25.0 | 25 (25) | 100.0 | 0 | 200 / 0 / 0 | 0.0 / n/a / n/a | 100.0, 100.0 | 100.0 / 0.0 / 0.0 |
| Qwen3-4B (untrained) | 47.3 | 64 (39) | 56.2 | 3.1 | 24.0 | 12 (12) | 100.0 | 13 | 73 / 0 / 14 | 0.0 / n/a / 0.0 | 73.0, 83.9 | 73.0 / 14.0 / 0.0 |
| SFT only (v1) | 100.0 | 50 (50) | 100.0 | 0.0 | 100.0 | 25 (25) | 100.0 | 0 | 0 / 0 / 0 | n/a / n/a / n/a | 0.0, n/a | 0.0 / 0.0 / 0.0 |
| JevAgentRL-v2 (exploratory) | 100.0 | 100 (97) | 100.0 | 0.0 | 100.0 | 50 (50) | 100.0 | 0 | 0 / 0 / 0 | n/a / n/a / n/a | 0.0, n/a | 0.0 / 0.0 / 0.0 |

Across the 18 models, Kendall's tau-b of live success with the novel-state share: 0.13; with the share of states after an off-plan step and outside C: -0.49; with the novel on-plan share: 0.50.

## 5. (c) Gate recovery: wrong and right live decisions below a confidence threshold

tau-hat = the model's own fitted tool-choice threshold (results/calibration, results/gpu_round5/cal, results/explore/e14_small/cal) where one exists, else the Jev arm's (1.00); a cascade accepts System-1's answer at confidence >= tau and otherwise asks the LLM, so a decision *below* tau is handed back. 'Failed tasks' = share of failed tasks whose first wrong tool choice was below tau (an upper bound on what a gate hands back; whether the LLM then chooses right is not measured here).

| model | wrong decisions | mean conf right / wrong % | tau-hat (source) | wrong below tau-hat % | right below tau-hat % | failed tasks' first error below tau-hat % | at 0.5: wrong / right / failed % | at 0.9: wrong / right / failed % |
|---|---|---|---|---|---|---|---|---|
| Jev | 68 | 93.4 / 63.8 | 1.00 (own) | 98.5 | 60.3 | 97.1 | 20.6 / 1.1 / 0.0 | 86.8 / 18.9 / 73.5 |
| Intern-Decision-4B | 317 | 80.6 / 70.4 | always defer (own) | 100.0 | 100.0 | 100.0 | 9.5 / 1.8 / 3.4 | 82.0 / 60.9 / 66.7 |
| StartLux-0.8B | 918 | 56.7 / 47.9 | 1.00 (Jev arm) | 100.0 | 100.0 | 100.0 | 57.1 / 33.5 / 58.2 | 100.0 / 94.7 / 100.0 |
| StartLux-2B | 312 | 58.0 / 57.1 | 1.00 (Jev arm) | 100.0 | 100.0 | 100.0 | 41.3 / 41.3 / 29.9 | 97.8 / 88.9 / 96.3 |
| StartLux-4B | 94 | 86.8 / 64.0 | 0.9756 (own) | 100.0 | 75.3 | 100.0 | 34.0 / 2.3 / 26.2 | 73.4 / 43.1 / 68.8 |
| StartLux-9B | 77 | 90.6 / 69.8 | 1.00 (Jev arm) | 100.0 | 100.0 | 100.0 | 13.0 / 0.2 / 7.7 | 83.1 / 35.1 / 75.0 |
| StartLux-27B | 74 | 98.0 / 82.1 | 1.00 (Jev arm) | 100.0 | 100.0 | 100.0 | 1.4 / 0.0 / 1.4 | 62.2 / 3.4 / 62.2 |
| OpenAI Decisions API (gpt-6-luna) | 184 | 73.4 / 50.4 | 1.00 (Jev arm) | 100.0 | 96.5 | 100.0 | 54.9 / 18.4 / 32.6 | 92.9 / 66.6 / 86.0 |
| OpenJev (DiffusionGemma-26B-A4B) | 397 | 90.4 / 82.3 | 1.00 (Jev arm) | 100.0 | 100.0 | 100.0 | 5.8 / 1.2 / 9.3 | 52.4 / 31.0 / 34.7 |
| OpenJev-27B (HF, FP8) | 75 | 98.6 / 91.9 | 1.00 (Jev arm) | 100.0 | 99.1 | 100.0 | 1.3 / 0.0 / 0.0 | 26.7 / 2.6 / 25.7 |
| AgentDecision-4B (JevAgentRL-4B, frozen) | 0 | 99.9 / n/a | 0.50 (own) | n/a | 0.0 | n/a | n/a / 0.1 / n/a | n/a / 0.2 / n/a |
| AgentDecision-4B + public data (E12) | 0 | 97.7 / n/a | 1.00 (Jev arm) | n/a | 100.0 | n/a | n/a / 0.0 / n/a | n/a / 6.6 / n/a |
| StartLux-4B + our recipe (E12c) | 0 | 95.9 / n/a | 1.00 (Jev arm) | n/a | 100.0 | n/a | n/a / 0.0 / n/a | n/a / 9.1 / n/a |
| 0.8B scoring head, agent states (E14 A) | 0 | 100.0 / n/a | 1.00 (own) | n/a | 16.8 | n/a | n/a / 0.0 / n/a | n/a / 0.0 / n/a |
| 0.8B scoring head, generic mix (E14 G) | 200 | 57.7 / 77.7 | always defer (own) | 100.0 | 100.0 | 100.0 | 5.0 / 0.0 / 5.0 | 63.0 / 100.0 / 63.0 |
| Qwen3-4B (untrained) | 224 | 68.8 / 52.0 | 1.00 (Jev arm) | 100.0 | 100.0 | 100.0 | 47.8 / 15.0 / 39.1 | 100.0 / 100.0 / 100.0 |
| SFT only (v1) | 0 | 100.0 / n/a | 1.00 (Jev arm) | n/a | 3.6 | n/a | n/a / 0.0 / n/a | n/a / 0.0 / n/a |
| JevAgentRL-v2 (exploratory) | 0 | 100.0 / n/a | 0.9987 (own) | n/a | 1.7 | n/a | n/a / 0.0 / n/a | n/a / 0.0 / n/a |

## 7. An offline score that counts consequences: predicting live success from the C answers alone

No live log enters this score. For each task, each reference trajectory recorded in C (the LLM-only, Jev, Intern-Decision-4B and AgentDecision-4B source arms) is walked in step order with the model's offline C answers; the first state where the answer is wrong decides: the task is predicted lost iff that error is irreversible (a wrong write action or a premature finish), else predicted solved; the task's prediction is the mean over its available trajectories, the model's over its live tasks. Variants: the LLM-only trajectory alone; the walk stopped at the reference's own first off-plan step; an acceptable early ask (asking where asking is acceptable but not canonical) ends the walk as a success.

| model | C mean % | must-ask acc % | predicted live % (4 sources) | LLM-only traj. | on-plan prefix | early ask resolves | actual live % | error (4 sources) |
|---|---|---|---|---|---|---|---|---|
| Jev | 91.5 | 27.0 | 72.9 | 63.5 | 73.4 | 80.0 | 83.0 | -10.1 |
| Intern-Decision-4B | 75.2 | 25.0 | 42.9 | 32.5 | 43.1 | 44.9 | 26.5 | +16.4 |
| StartLux-0.8B | 48.3 | 25.0 | 63.7 | 59.5 | 64.4 | 72.1 | 17.5 | +46.2 |
| StartLux-2B | 68.4 | 27.0 | 24.5 | 18.0 | 27.3 | 25.2 | 6.5 | +18.0 |
| StartLux-4B | 89.1 | 47.0 | 67.4 | 51.0 | 68.2 | 67.4 | 60.0 | +7.4 |
| StartLux-9B | 91.8 | 50.0 | 71.0 | 60.5 | 72.4 | 71.0 | 74.0 | -3.0 |
| StartLux-27B | 93.9 | 26.0 | 78.9 | 63.0 | 79.6 | 78.9 | 63.0 | +15.9 |
| OpenAI Decisions API (gpt-6-luna) | 90.9 | 38.5 | 77.4 | 67.0 | 77.9 | 80.1 | 78.5 | -1.1 |
| OpenJev (DiffusionGemma-26B-A4B) | 91.1 | 50.0 | 71.5 | 67.0 | 72.6 | 71.5 | 62.5 | +9.0 |
| OpenJev-27B (HF, FP8) | 92.4 | 26.0 | 75.6 | 63.0 | 76.6 | 75.6 | 63.0 | +12.6 |
| AgentDecision-4B (JevAgentRL-4B, frozen) | 99.2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 |
| AgentDecision-4B + public data (E12) | 99.1 | 99.7 | 99.8 | 100.0 | 100.0 | 99.8 | 100.0 | -0.2 |
| StartLux-4B + our recipe (E12c) | 99.6 | 100.0 | 99.8 | 99.5 | 99.9 | 99.8 | 100.0 | -0.2 |
| 0.8B scoring head, agent states (E14 A) | 99.2 | 100.0 | 99.4 | 98.0 | 99.5 | 99.4 | 100.0 | -0.6 |
| 0.8B scoring head, generic mix (E14 G) | 23.9 | 25.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | +0.0 |
| Qwen3-4B (untrained) | 50.1 | 24.0 | 21.8 | 18.0 | 23.0 | 30.0 | 13.0 | +8.8 |
| SFT only (v1) | 99.2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 |
| JevAgentRL-v2 (exploratory) | 99.4 | 100.0 | 99.9 | 99.5 | 99.9 | 99.9 | 100.0 | -0.1 |

| row set (n) | Kendall tau-b with live: C mean | must-ask acc | predicted (4 sources) | LLM-only traj. | on-plan prefix | early ask resolves | MAE points: predicted / LLM-only / on-plan / early ask |
|---|---|---|---|---|---|---|---|
| all (18) | 0.78 | 0.72 | 0.83 | 0.83 | 0.83 | 0.84 | 8.3 / 7.0 / 8.6 / 9.0 |
| offtheshelf (10) | 0.54 | 0.26 | 0.63 | 0.57 | 0.63 | 0.63 | 14.0 / 11.7 / 14.5 / 14.4 |

Off-the-shelf rows = E15's set (every row except the paper's trained models and the attribution rows): jev, intern, sl0.8b, sl2b, sl4b, sl9b, sl27b, oaidec, openjev, openjevhf.

## 6. Inputs, rows not found and runs not used

| model | live log (sha256 prefix) | offline C answers |
|---|---|---|
| Jev | `runs/stage2_rl_states/jev_ts_replace_f.jsonl` (c4051d8652571fa4) | `results/stage2/s23/answers_jev.jsonl` |
| Intern-Decision-4B | `runs/stage2_rl_states/intern_ts_replace_f.jsonl` (7d6210b6cc07ec2c) | `results/stage2/s23/answers_intern.jsonl` |
| StartLux-0.8B | `runs/explore_startlux_live/sl0.8b_ts_replace_f.jsonl` (5b2c681839995a75) | `results/stage2/s23/answers_sl0.8b.jsonl` |
| StartLux-2B | `runs/explore_startlux_live/sl2b_ts_replace_f.jsonl` (dc2197ec21f334c5) | `results/stage2/s23/answers_sl2b.jsonl` |
| StartLux-4B | `runs/stage2_startlux_states/sl4b_ts_replace_f.jsonl` (95075ba3bfa2aba4) | `results/stage2/s23/answers_sl4b.jsonl` |
| StartLux-9B | `runs/explore_startlux_live/sl9b_ts_replace_f.jsonl` (bdef92c1d9613609) | `results/stage2/s23/answers_sl9b.jsonl` |
| StartLux-27B | `runs/stage2_startlux_states/sl27b_ts_replace_f.jsonl` (268bc82b3837d7ea) | `results/stage2/s23/answers_sl27b.jsonl` |
| OpenAI Decisions API (gpt-6-luna) | `runs/e15_live_oaidec/oaidec_ts_replace_f.jsonl` (946d3b7db96cf126) | `results/explore/e15_baselines/c/answers_oaidec.jsonl.gz` |
| OpenJev (DiffusionGemma-26B-A4B) | `<tmp>` (a21c0998291842aa) | `results/explore/e15_baselines/c/answers_openjev.jsonl.gz` |
| OpenJev-27B (HF, FP8) | `<tmp>` (f264b9344f64e3ae) | `results/explore/e15_baselines/c/answers_openjevhf.jsonl.gz` |
| AgentDecision-4B (JevAgentRL-4B, frozen) | `runs/stage2_rl_states/rl_ts_replace_f.jsonl` (decae4fb1fffd1e3) | `results/stage2/s23/answers_rl.jsonl` |
| AgentDecision-4B + public data (E12) | `runs/e12_live/e12_ts_replace_f.jsonl` (d39d62bae53c5e8c) | `results/explore/e12/box/s23/answers_e12_mixed.jsonl` |
| StartLux-4B + our recipe (E12c) | `runs/e12c_live/e12c_ts_replace_f.jsonl` (9c5c15a8375aa9e5) | `results/explore/e12/box_c/s23/answers_e12c_sl4b.jsonl` |
| 0.8B scoring head, agent states (E14 A) | `runs/e14_live_a/e14a_ts_replace_f.jsonl` (9455eb1c77955a8a) | `results/explore/e14_small/records/s23/answers_e14a.jsonl.gz` |
| 0.8B scoring head, generic mix (E14 G) | `runs/e14_live_a_g/e14g_ts_replace_f.jsonl` (415ed22a7c0c3528) | `results/explore/e14_small/records/s23/answers_e14g.jsonl.gz` |
| Qwen3-4B (untrained) | `runs/stage2_rl_attribution/rl_base_ts_replace_f.jsonl` (03fbe072f79015ef) | `results/stage2/s23/answers_base.jsonl` |
| SFT only (v1) | `runs/stage2_rl_attribution/rl_sft_v1_ts_replace_f.jsonl` (c1b4b43c1c8ba5d5) | `results/stage2/s23/answers_sft_v1.jsonl` |
| JevAgentRL-v2 (exploratory) | `runs/stage2_v2_states/v2_ts_replace_f.jsonl` (bb780bb76560fa9f) | `results/stage2/s23/answers_v2.jsonl` |

Not rows: e14ag (E14 arm AG): no live run on seeds 300000-300199 (runs/e14_live_b is arm e14_final on seeds 300500+); E5 students: live on 100 tasks of seeds 300700+ through OpenRouter (runs/e5_small_live), another protocol; E7/E7c/E7e/E7f rows: DecisionGym-Hard (seeds 310000+), another suite; stage2_x / stage2_sol: DeepSeek-V4-Pro / GPT-6.1-Sol as the agent LLM, not Flash, and profile arms (cascade); E16 arms: in progress in another session; no record read; llm_only (Flash alone): not a decision model; its states are a C source (hash check only).
