# Train / evaluation hygiene audit (JevAgentRL v1, exploratory v2)

Generated 2026-10-05T19:07:34+00:00 at git `4917cfbc99bd` by `scripts/leakage_audit.py` (full). Post hoc and descriptive; no API call; changes no registered result. Machine-readable: `results/training/leakage_audit.json`; paper keys: `paper/generated/leakage.tex` (`\res{leak/...}`).

## Verdicts

| item | verdict | evidence |
|---|---|---|
| 1 Seeds | clean | 2898 training seeds read from files (+14000 pool seeds) vs 1250 evaluation seeds: 0 shared |
| 2 Content overlap | clean at instance level; generator-level overlap by design | 0 shared gold plans and 0 shared worlds with the Stage-2 tasks; 174 of 5314 C occurrences (3.3%) repeat a training state verbatim after noise removal (within C across tasks: 2.6%); Jaccard >= 0.8: 10.4% vs 10.4% within C; JevAgentRL-4B on C 99.2% -> 99.2% without them; tau^2 and BFCL / When2Call: no exact or near duplicate (v1) |
| 3 Labels / oracle | clean (operator barrier procedural: one disclosed schema read; one Stage-1 file in the exploratory v2 checkout) | teacher cache 7552/7552 requests re-keyed from training rows; wire canary absent: True |
| 4 Model selection | clean | chain inputs development-only; tau-hat* fits on calibration seeds: True |
| 5 External models | cannot be ruled out in principle; dated evidence against | JevGym first commit b9cb0a8 2026-09-30T19:54:47+00:00; Jev snapshots ['typesafe/jev-1.13-20260917'] |

## Findings and fixes (none is a train / evaluation leak)

| finding | size / evidence | fix |
|---|---|---|
| Appendix (app:rl, 'Stages and hyper-parameters', line 2230) misstates the OPD run | stated: seeds 40,000-40,511, 16 tasks per iteration, LR 3e-5, teacher budget $2, development rollouts 1,000-1,031 (these match configs/train/opd_jevgym.yaml's batch / LR / dev size, not the round-5 run); the run: pool 40,000-43,999 (864 tasks drawn, 40,011-43,998), 36 tasks per iteration x 24, LR 5e-5, teacher hard stop $8 (spent $0.38), greedy development rollouts on seeds 1,000-1,047 (configs/train/opd_round5.yaml, results/gpu_round5/opd_metrics.jsonl, opd_summary.json); 751 of the 864 drawn training tasks lie outside the stated range (all inside the training pool) | replace the sentence's values with the round-5 run's (draft in leakage_paper_draft.tex); reporting accuracy only, no leak |
| data v0's When2Call replay was not decontaminated | 15 of 2700 rows share an 8-gram with an evaluation question ({'when2call_test': 12, 'bfcl_only': 3}); v0's 300-row When2Call dev file: 2. Data v1 dropped them (v1 replay: 0); only SFT v0 (R2, a development ablation row never scored on L1) trained on v0's replay | state it in the data-v0 description; no reported evaluation number is affected |
| v0's 24-row MC stand-in file shares tasks with the 210-decision gate set | 6 tasks; the file is a pipeline sanity set that no recipe trains on | none needed; noted for completeness |
| The v2 operator's checkout contained one Stage-1 output file | results/primaries/c7_selection.json (E2a non-inferiority and cost summaries of the C7 candidate arms, committed 2026-10-02 21:55 UTC) came in with the fork point; the v2 manifest records that no evaluation output was opened; the v1 checkout held start records only | disclose in the v2 appendix paragraph; v2 is exploratory and enters no test |
| Generator-level overlap: verbatim step-0 / step-1 states | 174 of 5314 C occurrences (3.3%) repeat a training state after noise removal (instruction-only states such as the name-less clarify request); the same rate within C across tasks is 2.6% | report the re-scored accuracies (unchanged for the trained model); no gold plan, world or (plan, step) is shared |

## 1. Seeds

Training-side sources (seeds read from the files unless marked reconstructed / pool):

| source | kind | how | tasks | seeds | range(s) |
|---|---|---|---|---|---|
| `data_v0_train_rows` | training | row ids | 77 | 77 | 10000-10004, 10006-10013, 10015-10027, 12000-12010, 12012-12017, 12019-12021, ... (12 runs) |
| `data_v0_mc_standin_unused` | unused | row ids (pipeline sanity set, never trained on) | 22 | 22 | 12001, 12003, 12006, 12009-12011, 12014, 12018-12019, ... (16 runs) |
| `data_v0_episodes` | training | episode task ids (v0 trajectories) | 90 | 90 | 10000-10029, 12000-12059 |
| `data_v1_train_rows` | training | row ids | 587 | 587 | 10000-10004, 10006-10013, 10015-10027, 12000-12010, 12012-12017, 12019-12021, ... (88 runs) |
| `data_v1_build_inputs` | training | task ids of the builder's work files (gold rows before labelling + latest episodes, incl. the v1 dev tasks held out by task; data/cache/train_v1/_work) | 677 | 677 | 10000-10004, 10006-10013, 10015-10027, 12000-12010, 12012-12017, 12019-12021, ... (13 runs) |
| `data_v1_episodes` | training | episode task ids (data/cache/train_v1/train_v1_jevgym) | 600 | 600 | 30000-30599 |
| `rollouts_opd_v1` | training | reconstructed: opd.py: random.Random(0).shuffle(range(n)), 36 per iteration, no reshuffle before 4000 | 864 | 864 | 40011, 40017, 40019-40020, 40022, 40025, 40045, ... (673 runs) |
| `rollouts_grpo_v1` | training | reconstructed: grpo_online.py: random.Random(0).shuffle(range(n)), 8 per iteration | 240 | 240 | 50003, 50017, 50020, 50038, 50072, 50081, ... (223 runs) |
| `rollouts_opd_v2` | training | reconstructed: miles_port/rollout.py: random.Random(0).shuffle(range(n)); order[(rid*k+j) % n] | 864 | 864 | 60011, 60017, 60019-60020, 60022, 60025, 60045, ... (673 runs) |
| `rollouts_grpo_v2` | training | as logged (results/gpu_round6_miles/grpo_rollouts.jsonl) | 240 | 240 | 70003, 70017, 70020, 70038, 70072, 70081, ... (223 runs) |
| `pool_opd_v1` | training_pool | configs/train/opd_round5.yaml: seeds 40000-43999 |  | 4000 | 40000-43999 |
| `pool_grpo_v1` | training_pool | configs/train/grpo_online_round5.yaml: seeds 50000-52999 |  | 3000 | 50000-52999 |
| `pool_opd_v2` | training_pool | configs/train/miles_opd_v2.yaml: seeds 60000-63999 |  | 4000 | 60000-63999 |
| `pool_grpo_v2` | training_pool | configs/train/miles_grpo_v2.yaml: seeds 70000-72999 |  | 3000 | 70000-72999 |
| `gate_set_v0_dev` | development | row ids (210-decision gate set) | 13 | 13 | 10005, 10014, 10028-10029, 12011, 12018, 12022, ... (11 runs) |
| `data_v1_dev_rows` | development | row ids | 90 | 90 | 30006, 30034, 30043, 30048, 30068, 30086-30088, ... (75 runs) |
| `greedy_dev_rollouts_v1` | development | OPD --eval-seed0 1000 / GRPO eval_suite seed0 1000, eval_n 48 |  | 48 | 1000-1047 |
| `e2_dev_slice` | development | configs/experiments/jevgym_dev_rl.yaml |  | 50 | 1100-1149 |
| `stress_dev_v2` | development | configs/experiments/jevgym_stress_dev_miles.yaml |  | 200 | 2000-2199 |

Reconstruction check (v2 GRPO, the only rollout log with seeds): 240 of 240 logged task seeds reproduced in order (identical: True). The v1 OPD / GRPO samplers are the same `random.Random(0).shuffle` code at the training commits (`cf767b1`, `805a185`), so their draws are reconstructed the same way.

Evaluation-side sources:

| source | how | seeds | range(s) |
|---|---|---|---|
| `stage1_task_list` | configs/eval/jevgym_eval_tasks.json | 500 | 200000-200499 |
| `stage2_task_list` | configs/eval/jevgym_stage2_tasks.json | 500 | 300000-300499 |
| `s23_corpus_tasks` | results/stage2/s23/corpus.json eligible_tasks | 200 | 300000-300199 |
| `harness_final_spare` | configs/experiments/harness_final_live.yaml suite seed0 300500, n 200 | 200 | 300500-300699 |

`runs/` directories that touched a registered evaluation range:

| run dir | seeds | range(s) | classes |
|---|---|---|---|
| `e2a_jevgym_tool_decisions` | 500 | 200000-200499 | {'stage1_eval': 500} |
| `e2a_jevgym_tool_decisions_intern` | 500 | 200000-200499 | {'stage1_eval': 500} |
| `e2b_jevgym_meta_control` | 500 | 200000-200499 | {'stage1_eval': 500} |
| `e2c_jevgym_baselines` | 500 | 200000-200499 | {'stage1_eval': 500} |
| `explore_startlux_live` | 200 | 300000-300199 | {'stage2_eval': 200} |
| `pilot_jev_n50/e2b_jevgym_meta_control` | 100 | 0-49, 100000-100049 | {'development': 50, 'retired': 50} |
| `pilot_jev_n50_split/e2b_jevgym_meta_control_new` | 50 | 100000-100049 | {'retired': 50} |
| `stage2_jevgym` | 500 | 300000-300499 | {'stage2_eval': 500} |
| `stage2_rl_attribution` | 100 | 300000-300099 | {'stage2_eval': 100} |
| `stage2_rl_states` | 200 | 300000-300199 | {'stage2_eval': 200} |
| `stage2_sol` | 500 | 300000-300499 | {'stage2_eval': 500} |
| `stage2_startlux_states` | 200 | 300000-300199 | {'stage2_eval': 200} |
| `stage2_v2` | 500 | 300000-300499 | {'stage2_eval': 500} |
| `stage2_v2_states` | 200 | 300000-300199 | {'stage2_eval': 200} |
| `stage2_x` | 500 | 300000-300499 | {'stage2_eval': 500} |

Calibration (tau-hat*) fits and their seeds (read from each fit file):

| fit file | seeds | run |
|---|---|---|
| `results/calibration/tauhat_intern_shadow_cal_s2lite_tool_params.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal_intern` |
| `results/calibration/tauhat_intern_shadow_cal_s2lite_tool_select.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal_intern` |
| `results/calibration/tauhat_intern_shadow_cal_tool_params.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal_intern` |
| `results/calibration/tauhat_intern_shadow_cal_tool_select.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal_intern` |
| `results/calibration/tauhat_jev_shadow_cal_s2lite_tool_params.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal` |
| `results/calibration/tauhat_jev_shadow_cal_s2lite_tool_select.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal` |
| `results/calibration/tauhat_jev_shadow_cal_sol_tool_params.json` | 5150-5299 | `runs/e2a_jevgym_shadow_cal_sol` |
| `results/calibration/tauhat_jev_shadow_cal_sol_tool_select.json` | 5150-5299 | `runs/e2a_jevgym_shadow_cal_sol` |
| `results/calibration/tauhat_jev_shadow_cal_tool_params.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal` |
| `results/calibration/tauhat_jev_shadow_cal_tool_select.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal` |
| `results/calibration/tauhat_jev_shadow_cal_x_tool_params.json` | 5150-5299 | `runs/e2a_jevgym_shadow_cal_x` |
| `results/calibration/tauhat_jev_shadow_cal_x_tool_select.json` | 5150-5299 | `runs/e2a_jevgym_shadow_cal_x` |
| `results/calibration/tauhat_sl4b_shadow_cal_tool_params.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal_startlux` |
| `results/calibration/tauhat_sl4b_shadow_cal_tool_select.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal_startlux` |
| `results/calibration/tauhat_tau2_jev_shadow_tool_params.json` | None-None | `runs/tau2/airline_shadow` |
| `results/calibration/tauhat_tau2_jev_shadow_tool_select.json` | None-None | `runs/tau2/airline_shadow` |
| `results/calibration/tauhat_v2_shadow_cal_tool_params.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal_v2` |
| `results/calibration/tauhat_v2_shadow_cal_tool_select.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal_v2` |
| `results/gpu_round5/cal/tauhat_rl_shadow_cal_s2lite_tool_params.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal_rl` |
| `results/gpu_round5/cal/tauhat_rl_shadow_cal_s2lite_tool_select.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal_rl` |
| `results/gpu_round5/cal/tauhat_rl_shadow_cal_tool_params.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal_rl` |
| `results/gpu_round5/cal/tauhat_rl_shadow_cal_tool_select.json` | 5000-5149 | `runs/e2a_jevgym_shadow_cal_rl` |

Disjointness checks (each must be 0):

- `train_vs_eval_files`: 0
- `train_vs_registered_eval_ranges`: 0
- `train_vs_calibration_fits`: 0
- `train_vs_dev_instruments`: 0
- `train_below_10000`: 0
- `train_in_calibration_range`: 0
- `dev_vs_eval`: 0
- `calibration_vs_eval`: 0
- `v1_train_tasks_vs_v1_dev_tasks`: 0
- `v1_train_tasks_vs_gate_tasks`: 0
- `v0_train_tasks_vs_gate_tasks`: 0
- `seed_guard_train_forbidden`: [[200000, 200499], [100000, 100499], [5000, 5999], [300000, 300999]]
- training pools pairwise: {'pool_opd_v1~pool_opd_v2': 0, 'pool_grpo_v1~pool_opd_v1': 0, 'pool_grpo_v1~pool_opd_v2': 0, 'pool_grpo_v1~pool_grpo_v2': 0, 'pool_grpo_v2~pool_opd_v1': 0, 'pool_grpo_v2~pool_opd_v2': 0}
- note (not a training input): the 24-row MC stand-in file `data/train_v0/mc_standin.jsonl`, a pipeline sanity set no recipe trains on, shares tasks with the 210-decision gate set: ['jevgym-cancel-12035-d30-mixed-strict', 'jevgym-cancel-12055-d30-mixed-strict', 'jevgym-refund-12011-d30-mixed-strict', 'jevgym-refund-12041-d30-mixed-strict', 'jevgym-ship-12018-d30-mixed-strict', 'jevgym-stock-12022-d30-mixed-strict']

## 2. Content overlap

(a) **Task specs.** 2898 training tasks (data v0 / v1, the v1 / v2 rollout draws) and the 14000 tasks of the four rollout pools, regenerated from their seeds (`JevGymEnv`, task generation only), against every evaluation task. `plan` = family, variant, instruction, customer reply and the gold plan with every argument (user ids, order ids, amounts, speeds, reasons) and the final answer; `world` = all users, orders and stock.

| evaluation set | vs | n | world dup | plan dup | instruction dup | (name, user id) dup | (order id, amount) dup |
|---|---|---|---|---|---|---|---|
| stage1_eval | training tasks | 500 | 0 | 0 | 45.4% | 0.0% | 0.0% |
| stage2_eval | training tasks | 500 | 0 | 0 | 43.2% | 0.2% | 0.0% |
| s23_corpus_tasks | training tasks | 200 | 0 | 0 | 43.5% | 0.0% | 0.0% |
| harness_final_spare | training tasks | 200 | 0 | 0 | 44.0% | 0.0% | 0.0% |
| stage1_eval | rollout pools | 500 | 0 | 0 | 50.0% | 0.6% | 0.2% |
| stage2_eval | rollout pools | 500 | 0 | 0 | 50.0% | 0.6% | 0.0% |
| s23_corpus_tasks | rollout pools | 200 | 0 | 0 | 50.0% | 0.5% | 0.0% |
| harness_final_spare | rollout pools | 200 | 0 | 0 | 50.0% | 0.5% | 0.0% |
| stage1_eval | dev + calibration | 500 | 0 | 0 | 21.0% | 0.0% | 0.0% |
| stage2_eval | dev + calibration | 500 | 0 | 0 | 23.2% | 0.2% | 0.0% |
| s23_corpus_tasks | dev + calibration | 200 | 0 | 0 | 21.5% | 0.5% | 0.0% |
| harness_final_spare | dev + calibration | 200 | 0 | 0 | 18.0% | 0.0% | 0.0% |
| stage2_eval | stage1_eval (baseline, no training) | 500 | 0 | 0 | 23.8% | 0.0% | 0.0% |

Instruction duplicates by variant (Stage-2 vs training tasks): method 48/63, multi 0/250, name 62/62, order 55/63, speed 51/62. The `name` variant's instruction names no customer, so it is the same text in every such task.

(b) **States: training rows vs the common-state corpus C.** 16943 JevGym training rows (5173 distinct noise-free states) vs 3939 distinct questions / 5314 occurrences of C. Normalisation: heartbeat noise lines and truncation fragments dropped, lower case, whitespace collapsed; masked = names, user ids, order ids and numbers replaced.

| measure | vs training rows: occurrences (rate) | tasks | within C, other tasks: occurrences (rate) |
|---|---|---|---|
| exact, instance text | 174 (3.3%) | 43 | 137 (2.6%) |
| exact, entity-masked (plan shape) | 1705 (32.1%) | 200 | 1986 (37.4%) |
| MinHash Jaccard >= 0.8 (word 5-grams) | 551 (10.4%) | 121 | 555 (10.4%) |
| MinHash Jaccard >= 0.9 (word 5-grams) | 174 (3.3%) | 43 | 137 (2.6%) |

Max estimated Jaccard per C question: to training rows median 0.31, p90 0.55, max 1.00; to C questions of other tasks median 0.32, p90 0.62. Every pair whose MinHash estimate reached 0.65 was re-scored with the exact Jaccard (best of the five MinHash candidates); thresholds are applied to the exact values (311 and 359 verified; mean estimate minus exact +0.015).

(gold-plan fingerprint, step) duplicates between training rows and C: 0.

Where the exact instance duplicates sit: steps {'0': 172, '1': 2}; gold {'ask_user': 160, 'find_user': 14}. Examples: `task: a customer wrote in asking us to ship their pending order (express if they are gold tier, otherwise standard), but did not say who they are. current obser` / `task: refund the full amount of ivan wang's most recent delivered order (highest order id). current observation: you have access to the customer-operations tool` / `task: refund the full amount of ivan smith's most recent delivered order (highest order id). current observation: you have access to the customer-operations too`

**Re-scored C accuracy** (task-weighted, % ; `without_x` drops every occurrence whose question is in overlap set x; `only_x` keeps only those):

| model | all | without exact | only exact | without near90 | only near90 | without near80 | only near80 | without masked | only masked |
|---|---|---|---|---|---|---|---|---|---|
| jev | 91.5 | 90.9 (n 5140) | 100.0 (n 174) | 90.9 (n 5140) | 100.0 (n 174) | 89.3 (n 4763) | 96.5 (n 551) | 89.4 (n 3609) | 93.6 (n 1705) |
| intern | 75.2 | 74.2 (n 5140) | 93.0 (n 174) | 74.2 (n 5140) | 93.0 (n 174) | 71.3 (n 4763) | 92.5 (n 551) | 74.1 (n 3609) | 73.2 (n 1705) |
| rl | 99.2 | 99.2 (n 5140) | 100.0 (n 174) | 99.2 (n 5140) | 100.0 (n 174) | 99.1 (n 4763) | 100.0 (n 551) | 99.0 (n 3609) | 100.0 (n 1705) |
| sft_v1 | 99.2 | 99.2 (n 5140) | 100.0 (n 174) | 99.2 (n 5140) | 100.0 (n 174) | 99.1 (n 4763) | 100.0 (n 551) | 99.0 (n 3609) | 100.0 (n 1705) |
| base | 50.1 | 48.3 (n 5140) | 95.3 (n 174) | 48.3 (n 5140) | 95.3 (n 174) | 44.9 (n 4763) | 80.0 (n 551) | 54.4 (n 3609) | 40.5 (n 1705) |
| v2 | 99.4 | 99.3 (n 5140) | 100.0 (n 174) | 99.3 (n 5140) | 100.0 (n 174) | 99.3 (n 4763) | 100.0 (n 551) | 99.2 (n 3609) | 100.0 (n 1705) |
| sl0.8b | 48.3 | 46.6 (n 5140) | 100.0 (n 174) | 46.6 (n 5140) | 100.0 (n 174) | 42.7 (n 4763) | 82.4 (n 551) | 52.7 (n 3609) | 36.6 (n 1705) |
| sl2b | 68.4 | 67.3 (n 5140) | 97.7 (n 174) | 67.3 (n 5140) | 97.7 (n 174) | 64.6 (n 4763) | 93.1 (n 551) | 63.5 (n 3609) | 79.6 (n 1705) |
| sl4b | 89.1 | 88.6 (n 5140) | 97.7 (n 174) | 88.6 (n 5140) | 97.7 (n 174) | 88.6 (n 4763) | 88.3 (n 551) | 89.8 (n 3609) | 88.0 (n 1705) |
| sl9b | 91.8 | 91.2 (n 5140) | 100.0 (n 174) | 91.2 (n 5140) | 100.0 (n 174) | 89.6 (n 4763) | 99.8 (n 551) | 89.9 (n 3609) | 95.5 (n 1705) |
| sl27b | 93.9 | 93.5 (n 5140) | 100.0 (n 174) | 93.5 (n 5140) | 100.0 (n 174) | 92.1 (n 4763) | 100.0 (n 551) | 92.6 (n 3609) | 96.6 (n 1705) |

(c) **tau^2 reference corpus.** 18782 distinct states of 22704 questions: 0 exact duplicates of a training state (JevGym rows and both When2Call replays); MinHash max Jaccard {'median': 0.0, 'p99': 0.03125, 'max': 0.0390625}; >= 0.5: 0, >= 0.8: 0. Its airline tasks: 35, of which in the tau-hat* refit split: 0.

(d) **BFCL / When2Call.** 6687 evaluation questions (every BFCL item of the six tool categories and every When2Call test question; a superset of the C8 population). Rule: 8-gram overlap of the row's user text (data_v1.query_text) with any BFCL (six tool categories, every item) or When2Call test question.

| file | rows | 8-gram hits | exact query duplicates | near-dup >= 0.8 (unique queries) |
|---|---|---|---|---|
| `v1_train_jevgym` | 15694 | 0 | 0 |  |
| `v1_dev` | 2377 | 0 | 0 |  |
| `v1_replay_when2call` | 2685 | 0 | 0 | 0 |
| `v0_train` | 1225 | 0 | 0 |  |
| `v0_replay_when2call` | 2700 | 15 | 0 | 0 |
| `v0_dev_when2call` | 300 | 2 | 0 | 0 |

v0 replay hits absent from v1: 15 (by evaluation source: {'when2call_test': 12, 'bfcl_only': 3}); E1 dev-prefix ids in training: 0.

## 3. Labels and oracle access

| file | rows | label sources | rows whose state / question carries an oracle word |
|---|---|---|---|
| v1_train | 15694 | {'gold': 864, 'teacher': 14830} | 0 |
| v1_dev | 2377 | {'gold': 2377} | 0 |
| v0_train | 1225 | {'gold': 1225} | 0 |
| v0_gate_dev | 210 | {'gold': 210} | 0 |

**Teacher (Jev) cache, re-keyed.** `data/cache/teacher_v1.sqlite` holds 7552 entries; re-keying the 5175 train requests (rows grouped by state, <= 16 questions) and 2377 dev requests (one question each) from the training files gives 7552 matches, 0 cache entries with no training row and 0 rows never sent. Seeds of those rows: 10000-30599 (in an evaluation range: 0). Model that answered: {'typesafe/jev-1.13-20260917': 7552}. The OPD v1 teacher (6549 calls, $0.38) answered the student's own rollout states on the training pool; its cache stayed on the GPU host (not re-keyed here).

**Wire canary (offline).** Gold in `Question.meta`: True; in the Jev wire questions: False; in `serve_jev`'s training / serving prompt: False (checked: True). Transport-level test: tests/test_wire_canary.py (transport-level canary for every backend, the harness loop).

**Operator barrier (A7.10 (b)).** Procedural, recorded in the training manifests (one disclosed schema read of one Stage-1 record, 2026-10-02 ~20:40 UTC, ruled on in the Stage-2 Deviations). Git evidence:

- v1 (<worktree>): 9 commits merged by `487e212`; evaluation-output paths touched: none; forked at `9271976 2026-10-02T19:52:42+00:00`; evaluation-side files in the branch tree: ['results/main/START_E1.txt', 'results/main/START_E2a.txt'].
- v2 (<worktree>): 26 commits merged by `6670c7a`; evaluation-output paths touched: none; forked at `2dd05a9 2026-10-03T01:19:40+00:00`; evaluation-side files in the branch tree: ['results/main/RESUME_E2a.txt', 'results/main/RESUME_E2b.txt', 'results/main/START_C7.txt', 'results/main/START_E1.txt', 'results/main/START_E2a.txt', 'results/main/START_E2b.txt', 'results/main/START_E2c.txt', 'results/primaries/c7_selection.json'].

Times: {"training_manifest_v1_commit": ["9945d04 2026-10-02T22:11:52+00:00"], "training_manifest_v2_commit": ["98db847 2026-10-03T03:03:56+00:00"], "stage1_primaries_first_commit": ["7b17627 2026-10-03T12:07:07+00:00"], "freeze_v1": "2026-10-03T10:17:50Z", "freeze_v2": "2026-10-04T00:53:20Z", "first_stage2_run": "2026-10-03T06:57:18Z", "first_rl_eval_run": "2026-10-03T18:02:46Z", "first_rl_states_run": "2026-10-03T18:24:41Z", "teacher_cache_mtime": "2026-10-02T03:13:01.424697+00:00", "stage1_main_run_start": "2026-10-02T17:01:32Z"}

## 4. Model selection

- v1: kept ['sft_v1', 'grpo_i15']; retention: grpo_i15: highest v1-dev accuracy among passing continuations (0.99706 vs 0.99411). Inputs: v1 dev (training-seed tasks 30006-30597), recovery and knockout subsets, the 210-decision gate set (training-seed tasks 10005-12055), greedy dev rollouts 1000-1047, the E2 development slice 1100-1149 (r5_chain.py docstring; TRAINING_MANIFEST section 4). `chain_decision.json` sha256 equals the freeze record's: True. Temperatures: training/calibrate.py: golden-section NLL fit of a temperature per (kind x option-count bucket, min bucket 200) and a global one, on data/train_v1/dev.jsonl.gz (2,377 decisions, held-out training-seed tasks); applied by serve_jev --calibration.
- v2: inputs v1 dev, recovery, the gate set, the stress instrument 2000-2199 (development seeds), the E2 development slice 1100-1149 (r6_chain.py; TRAINING_MANIFEST section 5).
- Stage-2 profile thresholds (fit files named in `configs/harness/jevagent_best.yaml`; all on calibration seeds: True):
  - system2/gemini-3.5-flash/backends/jevagent_rl / tool_select: `tauhat_rl_shadow_cal_tool_select.json`, seeds [5000, 5149], sha256 as recorded: True
  - system2/gemini-3.5-flash/backends/jevagent_rl / tool_params: `tauhat_rl_shadow_cal_tool_params.json`, seeds [5000, 5149], sha256 as recorded: True
  - system2/gemini-3.5-flash/backends/startlux_4b / tool_select: `tauhat_sl4b_shadow_cal_tool_select.json`, seeds [5000, 5149], sha256 as recorded: True
  - system2/gemini-3.5-flash/backends/startlux_4b / tool_params: `tauhat_sl4b_shadow_cal_tool_params.json`, seeds [5000, 5149], sha256 as recorded: True
  - system2/gemini-3.5-flash/backends/jevagentrl_v2 / tool_select: `tauhat_v2_shadow_cal_tool_select.json`, seeds [5000, 5149], sha256 as recorded: True
  - system2/gemini-3.5-flash/backends/jevagentrl_v2 / tool_params: `tauhat_v2_shadow_cal_tool_params.json`, seeds [5000, 5149], sha256 as recorded: True
- Stage-2 profile chosen on Stage-1 evaluation results by the registered rule (scripts/select_profile.py, A7.1) and confirmed on fresh seeds 300000-300499.

## 5. External models

- Repository first commit: b9cb0a8 2026-09-30T19:54:47+00:00; JevGym first commit: b9cb0a8 2026-09-30T19:54:47+00:00.
- Jev responses by snapshot (every local decision cache): {'typesafe/jev-1.13-20260917': 192326}.
- Jev (TypeSafe): launched 2026-09-15 (docs/JEV_NOTES.md); every cached response names snapshot typesafe/jev-1.13-20260917.
- Intern-Decision-0.8B/2B/4B: 2026-09-26 (docs/DECISION_MODEL_LANDSCAPE.md); HF snapshot 0e5e6aa7 (TRAINING_MANIFEST v1).
- StartLux-Decision: 2026-09-30 per the main session; package a9101e9 of 2026-10-01 (A7.18 (b)).
- Qwen3-4B (base): HF snapshot 1cfa9a72 (TRAINING_MANIFEST v1).
- Closed training data cannot be audited; JevGym did not exist before its first commit, its seeds were never released, and the evaluation tasks were generated after every external model's recorded release. Public L1 items (BFCL, When2Call) may be in any vendor's data.

