# Is JevAgentBench's tool selection too easy? (difficulty analysis)

Generated 2026-10-05T19:10:49+00:00 at git `3e0181ef17df` by `scripts/benchmark_difficulty.py`. post hoc, descriptive; stored answers only (no API call, no GPU); revises no registered result. Subsets with fewer than 300 items are merged or dropped (listed per block). Point estimates in the main tables; the 95 % bootstrap intervals (tasks resampled for JevGym / tau^2, items for L1; B = 2000, seed 0) are in the appendix tables at the end. Paper keys: `paper/generated/difficulty.tex` (`\res{diff/...}`).

## 1. The ceiling is the agent LLM's, not the decision models'

| agent LLM, LLM-only | tasks | success % | own tool-decision accuracy % | failures | first wrong decision of the failed episodes |
|---|---|---|---|---|---|
| stage1_flash | 500 | 99.8 | 99.8 (3,756) | 1 (clarify 1) | unneeded question 1 |
| stage1_lite | 500 | 80.2 | 99.0 (3,789) | 99 (multi 98, clarify 1) | every call on plan, final answer wrong 97; finished too early 1; near-duplicate distractor 1 |
| stage2_flash | 500 | 99.4 | 99.6 (3,759) | 3 (clarify 2, multi 1) | unneeded question 2; wrong core tool (skip / order) 1 |

## 2. Common-state corpus C: hard subsets (tool selection, task-weighted accuracy %)

Hard subsets are fixed by rule from the state and its label, never from a model's answer. The agent LLM row is Flash's own choice on the occurrences of its own trajectories (`llm_states`). Counts per rule: all 5314, hard 957, after_off_plan 386, ask_gold 582, state_change_trap 1339, finish_gold 603, long_history 770, llm_states 1498, must_ask 267. Merged / dropped: must_ask: n = 267 < 300: merged into ask_gold. Flash chose an unacceptable tool at 1 of its 1498 own-state occurrences.

| model | all (n 5,314; 200 clusters) | hard (n 957; 135 clusters) | after_off_plan (n 386; 57 clusters) | ask_gold (n 582; 100 clusters) | state_change_trap (n 1,339; 200 clusters) | finish_gold (n 603; 200 clusters) | long_history (n 770; 104 clusters) | llm_states (n 1,498; 200 clusters) |
|---|---|---|---|---|---|---|---|---|
| agent LLM (own choice) | 99.9 (n 1498) | 100.0 (n 200) | – | 100.0 (n 200) | 99.9 (n 471) | 100.0 (n 199) | 100.0 (n 198) | 99.9 |
| Jev | 91.5 | 69.4 | 81.5 | 64.4 | 100.0 | 97.7 | 96.5 | 92.0 |
| Intern-Decision-4B | 75.2 | 55.7 | 47.6 | 59.6 | 89.4 | 98.5 | 93.5 | 76.3 |
| StartLux-0.8B | 48.3 | 42.7 | 31.5 | 49.4 | 83.2 | 97.5 | 71.9 | 52.1 |
| StartLux-2B | 68.4 | 56.2 | 47.2 | 61.8 | 98.0 | 74.7 | 72.8 | 73.7 |
| StartLux-4B | 89.1 | 81.7 | 76.8 | 84.1 | 100.0 | 97.8 | 95.1 | 89.6 |
| StartLux-9B | 91.8 | 82.7 | 74.3 | 87.1 | 99.8 | 98.1 | 94.5 | 92.9 |
| StartLux-27B | 93.9 | 71.2 | 85.1 | 63.4 | 100.0 | 98.6 | 96.8 | 92.6 |
| JevAgentRL-4B | 99.2 | 98.2 | 94.7 | 100.0 | 100.0 | 99.2 | 97.0 | 100.0 |
| Qwen3-4B (untrained) | 50.1 | 39.9 | 26.7 | 47.3 | 91.0 | 99.4 | 81.2 | 54.4 |
| SFT only (v1) | 99.2 | 98.0 | 93.8 | 100.0 | 100.0 | 99.2 | 97.0 | 100.0 |
| JevAgentRL-v2 (exploratory) | 99.4 | 98.2 | 95.0 | 100.0 | 99.8 | 99.2 | 97.0 | 100.0 |
| *spread max - min over the decision models scored (pt; number of models)* | 50.9 (8) | 55.5 (8) | 63.2 (8) | 50.6 (8) | 16.8 (8) | 24.5 (8) | 25.1 (8) | 47.9 (8) |

- `all`: every occurrence
- `hard`: after an off-plan step, or the next tool is ask_user (pre-declared union)
- `after_off_plan`: after an earlier executed tool outside its acceptable set
- `ask_gold`: the canonical next tool is ask_user (clarify; the must-ask stratum with ask_user as the only acceptable tool, n < 300, is merged here)
- `state_change_trap`: the next tool changes the world and a state-changing look-alike is offered
- `finish_gold`: the next tool is finish (stop)
- `long_history`: step >= 8 (history window full)
- `llm_states`: occurrences of the LLM-only (Flash) trajectories

Subsets with at least 300 occurrences but fewer than 100 tasks (the task is the resampling unit, so their intervals are wide): after_off_plan (386 occurrences from 57 tasks).

Error types on C (wrong answers by kind; counts over occurrences):

| model | changed state before asking | finished too early | look-alike of the gold tool | other look-alike | unneeded question | unrelated distractor | wrong core tool |
|---|---|---|---|---|---|---|---|
| Jev | 162 | 2 | 2 | 3 | 71 | 0 | 110 |
| Intern-Decision-4B | 181 | 19 | 130 | 42 | 83 | 0 | 816 |
| StartLux-0.8B | 346 | 488 | 14 | 13 | 522 | 0 | 1624 |
| StartLux-2B | 188 | 38 | 20 | 6 | 66 | 0 | 1456 |
| StartLux-4B | 109 | 12 | 2 | 0 | 17 | 1 | 330 |
| StartLux-9B | 105 | 13 | 9 | 1 | 50 | 0 | 174 |
| StartLux-27B | 165 | 6 | 0 | 0 | 42 | 0 | 25 |
| JevAgentRL-4B | 0 | 0 | 0 | 0 | 44 | 0 | 16 |
| Qwen3-4B (untrained) | 324 | 325 | 70 | 30 | 84 | 3 | 2023 |
| SFT only (v1) | 0 | 0 | 0 | 1 | 45 | 0 | 15 |
| JevAgentRL-v2 (exploratory) | 0 | 0 | 0 | 0 | 36 | 0 | 15 |

Accuracy by source arm of the occurrence (each model on the states of each policy; a home-state check):

| source arm (n) | Jev | Intern-Decision-4B | StartLux-0.8B | StartLux-2B | StartLux-4B | StartLux-9B | StartLux-27B | JevAgentRL-4B | Qwen3-4B (untrained) | SFT only (v1) | JevAgentRL-v2 (exploratory) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_only (1498) | 92.0 | 76.3 | 52.1 | 73.7 | 89.6 | 92.9 | 92.6 | 100.0 | 54.4 | 100.0 | 100.0 |
| jev_ts_replace_f (1391) | 90.9 | 77.0 | 52.0 | 67.9 | 87.4 | 90.3 | 98.1 | 99.5 | 53.5 | 99.6 | 99.7 |
| intern_ts_replace_f (925) | 86.2 | 62.2 | 31.6 | 69.6 | 82.9 | 87.9 | 88.9 | 98.9 | 35.7 | 98.8 | 99.1 |
| rl_ts_replace_f (1500) | 94.2 | 77.6 | 50.2 | 67.1 | 92.7 | 94.7 | 97.2 | 100.0 | 51.4 | 100.0 | 100.0 |

## 3. Stage-1 shadow logs: the same states for the agent LLM, Jev and Intern-Decision-4B

Evaluation seeds 200000-200499 (Stage 1). Only states asked identically in both shadow runs are scored (matched by state, question and options). StartLux, JevAgentRL-4B and the other rows need a replay (`replay_plan.md`).

### flash_lite states (3067 of 8719 decisions matched; distinct questions {'tool_select': 3787, 'tool_params': 4866})

| model | tool_select/all (n 1,393; 500 clusters) | tool_select/ask_gold (n 436; 223 clusters) | tool_params/all (n 1,674; 439 clusters) | tool_params/copy (n 950; 324 clusters) | tool_params/many_candidates (n 633; 324 clusters) |
|---|---|---|---|---|---|
| agent LLM (own choice) | 98.3 | 97.6 | 100.0 | 100.0 | 100.0 |
| Jev | 90.5 | 73.0 | 100.0 | 100.0 | 100.0 |
| Intern-Decision-4B | 72.6 | 67.7 | 81.6 | 78.0 | 61.2 |
| *spread max - min over the decision models scored (pt; number of models)* | 17.9 (2) | 5.3 (2) | 18.4 (2) | 22.0 (2) | 38.8 (2) |

Dropped (n < 300): tool_select/after_off_plan: n = 17 < 300: dropped; tool_select/state_change: n = 171 < 300: dropped; tool_select/long_history: n = 8 < 300: dropped; tool_params/gold_outside: n = 0 < 300: dropped

### flash states (3327 of 7952 decisions matched; distinct questions {'tool_select': 3756, 'tool_params': 4128})

| model | tool_select/all (n 1,693; 500 clusters) | tool_select/ask_gold (n 410; 212 clusters) | tool_params/all (n 1,492; 447 clusters) | tool_params/copy (n 797; 388 clusters) | tool_params/many_candidates (n 449; 250 clusters) |
|---|---|---|---|---|---|
| agent LLM (own choice) | 99.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| Jev | 93.0 | 77.7 | 100.0 | 100.0 | 100.0 |
| Intern-Decision-4B | 68.6 | 72.5 | 91.1 | 92.2 | 79.7 |
| *spread max - min over the decision models scored (pt; number of models)* | 24.4 (2) | 5.2 (2) | 8.9 (2) | 7.8 (2) | 20.3 (2) |

Dropped (n < 300): tool_select/after_off_plan: n = 0 < 300: dropped; tool_select/state_change: n = 250 < 300: dropped; tool_select/long_history: n = 4 < 300: dropped; tool_params/gold_outside: n = 142 < 300: dropped

## 4. tau^2 reference corpus (agreement %, agreement with one reference trajectory (A6.22 (b)), not oracle accuracy; Jev = as logged at decision time (its occurrences only))

| model | tool_select/all (n 19,934; 149 clusters) | tool_select/non_majority (n 8,283; 143 clusters) | tool_params/all (n 4,023; 137 clusters) | tool_params/many_candidates (n 2,635; 134 clusters) |
|---|---|---|---|---|
| Jev | 59.9 | 51.0 | 90.0 | 84.0 |
| JevAgentRL-4B | 53.7 | 50.7 | 85.9 | 76.9 |
| Intern-Decision-4B | 42.1 | 59.9 | 77.2 | 74.3 |
| Qwen3-4B (untrained) | 41.1 | 37.9 | 71.3 | 66.6 |
| *spread max - min over the decision models scored (pt; number of models)* | 17.8 (3) | 9.2 (3) | 12.8 (3) | 9.7 (3) |

## 5. L1 static (E1, C8 population; accuracy %)

| model | bfcl_tool_select (n 1,114; 1114 clusters) | bfcl_irrelevance (n 370; 370 clusters) | bfcl_multi_tool (n 372; 372 clusters) | bfcl_arguments (n 1,949; 1949 clusters) | bfcl_live_arguments (n 1,033; 1033 clusters) | when2call (n 928; 928 clusters) | when2call_noncall (n 599; 599 clusters) |
|---|---|---|---|---|---|---|---|
| agent LLM (own choice) | 84.2 | 55.1 | 97.8 | 96.3 | 94.2 | 42.1 (n 776) | 0.0 (n 449) |
| Jev | 89.9 | 74.1 | 96.8 | 89.6 | 89.0 | 75.4 | 69.4 |
| Intern-Decision-4B | 88.2 | 67.8 | 97.0 | 71.2 | 65.2 | 60.1 | 48.2 |
| StartLux-0.8B | 88.2 | 69.5 | 97.8 | 67.1 | 60.6 | 54.2 | 44.9 |
| StartLux-2B | 89.9 | 77.8 | 96.2 | 73.1 | 68.3 | 64.1 | 58.3 |
| StartLux-4B | 91.3 | 79.2 | 97.6 | 81.2 | 79.0 | 77.4 | 75.8 |
| StartLux-9B | 92.2 | 81.1 | 97.0 | 81.0 | 76.6 | 78.1 | 71.8 |
| StartLux-27B | 90.8 | 76.5 | 97.6 | 89.1 | 87.4 | 81.2 | 80.8 |
| JevAgentRL-4B | 88.4 | 84.9 | 91.7 | 85.0 | 81.5 | 54.0 | 83.6 |
| Qwen3-4B (untrained) | 87.5 | 70.0 | 95.7 | 70.5 | 66.5 | 53.9 | 32.6 |
| SFT only (v1) | 86.9 | 85.7 | 87.9 | 85.5 | 82.0 | 53.3 | 82.6 |
| *spread max - min over the decision models scored (pt; number of models)* | 3.9 (8) | 17.0 (8) | 6.2 (8) | 22.5 (8) | 28.4 (8) | 27.3 (8) | 38.7 (8) |

## 6. Stage-1 meta decisions in shadow (Jev only; the other models need a replay)

| states | decision | n | distinct questions | Jev % | majority label | majority % | minority n | Jev on minority % |
|---|---|---|---|---|---|---|---|---|
| flash | stop | 1501 | 1472 | 98.9 | False | 86.7 | 200 | 92.0 |
| flash | memory_write | 1301 | 1300 | 69.2 | True | 69.2 | 401 | 0.0 |
| flash | reflection | 1301 | 1300 | 77.8 | False | 100.0 | 0 | 0.0 |
| flash_lite | routing | 3786 | 3707 | 92.3 | False | 97.6 | 91 | 0.0 |
| flash_lite | stop | 3786 | 3707 | 98.9 | False | 89.2 | 409 | 98.8 |
| flash_lite | memory_write | 3286 | 3286 | 69.4 | True | 69.3 | 1009 | 0.3 |
| flash_lite | reflection | 3286 | 3286 | 84.3 | False | 97.2 | 91 | 1.1 |

## 7. Development stress slice (development data (seeds 2000-2199; v2 training track); not an evaluation)

| checkpoint / reference | n | success % | own-state tool acc % | argument acc % | call-count-penalised reward |
|---|---|---|---|---|---|
| grpo_i15 | 200 | 62.5 | 94.9 | 92.3 | 0.41 |
| grpo | 200 | 60.0 | 93.9 | 90.9 | 0.404 |
| opd_i12 | 200 | 66.5 | 94.7 | 92.8 | 0.473 |
| opd | 200 | 65.0 | 93.9 | 96.2 | 0.453 |
| refs_intern | 200 | 11.5 | 61.1 | 56.8 | -0.046 |
| refs_jev | 200 | 49.0 | 69.1 | 86.5 | 0.323 |
| sft_v2 | 200 | 71.5 | 95.4 | 93.3 | 0.502 |

## Appendix: 95 % bootstrap intervals

### C

| model | all (n 5,314; 200 clusters) | hard (n 957; 135 clusters) | after_off_plan (n 386; 57 clusters) | ask_gold (n 582; 100 clusters) | state_change_trap (n 1,339; 200 clusters) | finish_gold (n 603; 200 clusters) | long_history (n 770; 104 clusters) | llm_states (n 1,498; 200 clusters) |
|---|---|---|---|---|---|---|---|---|
| agent LLM (own choice) | 99.9 [99.8, 100.0] (n 1498) | 100.0 [100.0, 100.0] (n 200) | – | 100.0 [100.0, 100.0] (n 200) | 99.9 [99.6, 100.0] (n 471) | 100.0 [100.0, 100.0] (n 199) | 100.0 [100.0, 100.0] (n 198) | 99.9 [99.8, 100.0] |
| Jev | 91.5 [90.0, 92.9] | 69.4 [63.5, 74.9] | 81.5 [74.4, 88.1] | 64.4 [56.8, 71.2] | 100.0 [100.0, 100.0] | 97.7 [96.2, 99.1] | 96.5 [94.2, 98.5] | 92.0 [90.7, 93.2] |
| Intern-Decision-4B | 75.2 [73.2, 77.1] | 55.7 [49.3, 61.7] | 47.6 [37.4, 57.1] | 59.6 [51.9, 66.8] | 89.4 [86.0, 92.8] | 98.5 [97.4, 99.5] | 93.5 [89.7, 96.8] | 76.3 [74.4, 78.2] |
| StartLux-0.8B | 48.3 [45.9, 51.1] | 42.7 [37.3, 47.8] | 31.5 [23.7, 39.3] | 49.4 [42.2, 56.5] | 83.2 [80.0, 86.3] | 97.5 [95.8, 98.8] | 71.9 [66.8, 77.1] | 52.1 [49.8, 54.6] |
| StartLux-2B | 68.4 [66.6, 70.3] | 56.2 [50.5, 62.2] | 47.2 [38.6, 55.4] | 61.8 [54.0, 68.8] | 98.0 [96.7, 99.0] | 74.7 [69.4, 80.0] | 72.8 [67.9, 77.6] | 73.7 [71.6, 75.7] |
| StartLux-4B | 89.1 [87.4, 90.8] | 81.7 [77.6, 85.7] | 76.8 [67.7, 84.9] | 84.1 [79.9, 87.7] | 100.0 [100.0, 100.0] | 97.8 [96.5, 99.0] | 95.1 [91.7, 98.0] | 89.6 [88.1, 91.0] |
| StartLux-9B | 91.8 [90.0, 93.5] | 82.7 [78.8, 86.3] | 74.3 [64.9, 82.8] | 87.1 [84.4, 89.7] | 99.8 [99.2, 100.0] | 98.1 [96.9, 99.2] | 94.5 [90.9, 97.6] | 92.9 [91.6, 94.2] |
| StartLux-27B | 93.9 [92.7, 95.1] | 71.2 [65.2, 76.6] | 85.1 [78.1, 91.0] | 63.4 [55.7, 70.4] | 100.0 [100.0, 100.0] | 98.6 [97.7, 99.4] | 96.8 [93.9, 99.1] | 92.6 [91.3, 93.9] |
| JevAgentRL-4B | 99.2 [98.4, 99.8] | 98.2 [96.8, 99.3] | 94.7 [90.1, 98.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.2 [98.6, 99.8] | 97.0 [94.2, 99.2] | 100.0 [100.0, 100.0] |
| Qwen3-4B (untrained) | 50.1 [48.0, 52.5] | 39.9 [34.5, 45.6] | 26.7 [19.9, 33.5] | 47.3 [39.7, 54.6] | 91.0 [89.1, 92.9] | 99.4 [98.9, 99.9] | 81.2 [76.6, 85.8] | 54.4 [52.4, 56.5] |
| SFT only (v1) | 99.2 [98.4, 99.8] | 98.0 [96.5, 99.2] | 93.8 [89.3, 97.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 99.2 [98.6, 99.8] | 97.0 [94.3, 99.2] | 100.0 [100.0, 100.0] |
| JevAgentRL-v2 (exploratory) | 99.4 [98.7, 99.8] | 98.2 [96.9, 99.3] | 95.0 [90.8, 98.2] | 100.0 [100.0, 100.0] | 99.8 [99.5, 100.0] | 99.2 [98.5, 99.7] | 97.0 [94.2, 99.2] | 100.0 [99.8, 100.0] |
| *spread max - min over the decision models scored (pt; number of models)* | 50.9 (8) | 55.5 (8) | 63.2 (8) | 50.6 (8) | 16.8 (8) | 24.5 (8) | 25.1 (8) | 47.9 (8) |

### Stage-1 shadow, flash_lite states

| model | tool_select/all (n 1,393; 500 clusters) | tool_select/ask_gold (n 436; 223 clusters) | tool_params/all (n 1,674; 439 clusters) | tool_params/copy (n 950; 324 clusters) | tool_params/many_candidates (n 633; 324 clusters) |
|---|---|---|---|---|---|
| agent LLM (own choice) | 98.3 [97.4, 99.1] | 97.6 [96.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| Jev | 90.5 [89.1, 92.0] | 73.0 [68.2, 77.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| Intern-Decision-4B | 72.6 [70.2, 74.9] | 67.7 [62.8, 72.4] | 81.6 [79.1, 84.1] | 78.0 [75.7, 80.2] | 61.2 [56.7, 65.4] |
| *spread max - min over the decision models scored (pt; number of models)* | 17.9 (2) | 5.3 (2) | 18.4 (2) | 22.0 (2) | 38.8 (2) |

### Stage-1 shadow, flash states

| model | tool_select/all (n 1,693; 500 clusters) | tool_select/ask_gold (n 410; 212 clusters) | tool_params/all (n 1,492; 447 clusters) | tool_params/copy (n 797; 388 clusters) | tool_params/many_candidates (n 449; 250 clusters) |
|---|---|---|---|---|---|
| agent LLM (own choice) | 99.8 [99.4, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| Jev | 93.0 [91.8, 94.1] | 77.7 [73.3, 81.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| Intern-Decision-4B | 68.6 [66.0, 70.9] | 72.5 [67.7, 76.9] | 91.1 [89.1, 92.9] | 92.2 [90.6, 93.7] | 79.7 [75.5, 83.7] |
| *spread max - min over the decision models scored (pt; number of models)* | 24.4 (2) | 5.2 (2) | 8.9 (2) | 7.8 (2) | 20.3 (2) |

### tau^2

| model | tool_select/all (n 19,934; 149 clusters) | tool_select/non_majority (n 8,283; 143 clusters) | tool_params/all (n 4,023; 137 clusters) | tool_params/many_candidates (n 2,635; 134 clusters) |
|---|---|---|---|---|
| Jev | 59.9 [57.4, 62.3] | 51.0 [47.2, 54.7] | 90.0 [87.6, 92.2] | 84.0 [80.5, 87.1] |
| JevAgentRL-4B | 53.7 [51.2, 56.1] | 50.7 [46.6, 54.9] | 85.9 [83.1, 88.6] | 76.9 [72.3, 81.2] |
| Intern-Decision-4B | 42.1 [39.5, 44.6] | 59.9 [56.1, 63.4] | 77.2 [74.1, 80.2] | 74.3 [69.9, 78.4] |
| Qwen3-4B (untrained) | 41.1 [38.3, 43.9] | 37.9 [33.9, 42.1] | 71.3 [67.5, 75.0] | 66.6 [61.6, 71.6] |
| *spread max - min over the decision models scored (pt; number of models)* | 17.8 (3) | 9.2 (3) | 12.8 (3) | 9.7 (3) |

### L1 static

| model | bfcl_tool_select (n 1,114; 1114 clusters) | bfcl_irrelevance (n 370; 370 clusters) | bfcl_multi_tool (n 372; 372 clusters) | bfcl_arguments (n 1,949; 1949 clusters) | bfcl_live_arguments (n 1,033; 1033 clusters) | when2call (n 928; 928 clusters) | when2call_noncall (n 599; 599 clusters) |
|---|---|---|---|---|---|---|---|
| agent LLM (own choice) | 84.2 [82.0, 86.2] | 55.1 [50.0, 60.3] | 97.8 [96.2, 99.2] | 96.3 [95.4, 97.1] | 94.2 [92.8, 95.5] | 42.1 [38.7, 45.5] (n 776) | 0.0 [0.0, 0.0] (n 449) |
| Jev | 89.9 [88.2, 91.7] | 74.1 [69.7, 78.4] | 96.8 [94.9, 98.4] | 89.6 [88.2, 90.9] | 89.0 [87.1, 90.8] | 75.4 [72.6, 78.1] | 69.4 [65.6, 73.0] |
| Intern-Decision-4B | 88.2 [86.3, 89.9] | 67.8 [63.2, 72.4] | 97.0 [95.2, 98.7] | 71.2 [69.3, 73.3] | 65.2 [62.3, 68.2] | 60.1 [57.1, 63.3] | 48.2 [44.6, 52.4] |
| StartLux-0.8B | 88.2 [86.4, 89.9] | 69.5 [64.9, 74.3] | 97.8 [96.2, 99.2] | 67.1 [65.0, 69.2] | 60.6 [57.5, 63.6] | 54.2 [50.9, 57.3] | 44.9 [40.7, 48.7] |
| StartLux-2B | 89.9 [88.1, 91.6] | 77.8 [73.8, 81.9] | 96.2 [94.1, 98.1] | 73.1 [71.2, 75.1] | 68.3 [65.6, 71.2] | 64.1 [60.9, 67.1] | 58.3 [54.3, 62.3] |
| StartLux-4B | 91.3 [89.7, 92.9] | 79.2 [75.1, 83.5] | 97.6 [96.0, 98.9] | 81.2 [79.5, 82.9] | 79.0 [76.5, 81.5] | 77.4 [74.7, 80.1] | 75.8 [72.5, 79.1] |
| StartLux-9B | 92.2 [90.7, 93.7] | 81.1 [77.0, 84.9] | 97.0 [95.2, 98.7] | 81.0 [79.2, 82.7] | 76.6 [74.1, 79.1] | 78.1 [75.4, 80.7] | 71.8 [68.1, 75.3] |
| StartLux-27B | 90.8 [89.0, 92.4] | 76.5 [72.4, 80.8] | 97.6 [96.0, 98.9] | 89.1 [87.7, 90.5] | 87.4 [85.5, 89.4] | 81.2 [78.8, 83.7] | 80.8 [77.5, 83.8] |
| JevAgentRL-4B | 88.4 [86.4, 90.3] | 84.9 [81.1, 88.4] | 91.7 [88.7, 94.4] | 85.0 [83.4, 86.6] | 81.5 [79.1, 83.8] | 54.0 [50.9, 57.3] | 83.6 [80.6, 86.3] |
| Qwen3-4B (untrained) | 87.5 [85.5, 89.4] | 70.0 [65.7, 74.6] | 95.7 [93.5, 97.6] | 70.5 [68.5, 72.7] | 66.5 [63.8, 69.3] | 53.9 [50.6, 57.1] | 32.6 [28.9, 36.4] |
| SFT only (v1) | 86.9 [84.7, 88.9] | 85.7 [82.2, 89.2] | 87.9 [84.4, 91.1] | 85.5 [83.9, 87.1] | 82.0 [79.7, 84.3] | 53.3 [50.2, 56.6] | 82.6 [79.6, 85.5] |
| *spread max - min over the decision models scored (pt; number of models)* | 3.9 (8) | 17.0 (8) | 6.2 (8) | 22.5 (8) | 28.4 (8) | 27.3 (8) | 38.7 (8) |
