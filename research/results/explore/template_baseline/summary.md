# Template baselines on the common states: leakage, or a learnable generator?

Generated 2026-10-08T20:45:25+00:00 at git `a7981f803c7d` by `scripts/explore_template_baseline.py`. post hoc, descriptive; stored data only (no API call, no GPU, no model call); baselines fitted on the training rows only; revises no registered result; no registration entry (the convention of scripts/benchmark_difficulty.py / scripts/explore_drift.py). Paper keys: `paper/generated/template_baseline.tex` (`\res{tpl/...}`); App. `app:tplbase`.

## Training rows (all the baselines see)

`data/train_v1/train.jsonl.gz`: 4,493 JevGym tool-choice rows of 587 training tasks (without the knockout-final duplicates); label = argmax of the training target. Distinct keys: (b) 1,220, (c) 220.

Labels: get_orders 1228, find_user 1181, ship_order 554, finish 492, ask_user 470, cancel_order 469, refund 89, check_stock 10.

Families (keyword rule) in training: `ship:no_speed` 463, `refund:no_method` 457, `ship:no_name` 439, `cancel` 425, `multi:cancel,ship,cancel` 370, `multi:cancel,ship,ship` 369, `multi:ship,cancel,cancel` 357, `multi:ship,cancel,ship` 340, `multi:cancel,cancel,cancel` 301, `multi:ship,ship,cancel` 300, `multi:cancel,cancel,ship` 279, `multi:ship,ship,ship` 253, `ship` 44, `total` 40, `refund` 36, `stock` 20.

Families on C (occurrences): `multi:ship,ship,ship` 617, `multi:ship,cancel,cancel` 604, `multi:cancel,ship,cancel` 521, `ship:no_name` 505, `refund:no_method` 480, `multi:cancel,cancel,ship` 450, `ship:no_speed` 405, `cancel` 404, `multi:cancel,cancel,cancel` 385, `multi:ship,cancel,ship` 372, `multi:cancel,ship,ship` 358, `multi:ship,ship,cancel` 213.

## C: accuracy (task mean %, the published C metric)

| row | all (n 5,314; 200 tasks) | must_ask (n 267; 100 tasks) | after_off_plan (n 386; 57 tasks) | ask_gold (n 582; 100 tasks) | hard (n 957; 135 tasks) | state_change_trap (n 1,339; 200 tasks) | finish_gold (n 603; 200 tasks) | long_history (n 770; 104 tasks) | llm_states (n 1,498; 200 tasks) |
|---|---|---|---|---|---|---|---|---|---|
| (a) most frequent training label | 26.6 | 0.0 | 18.6 | 12.7 | 15.4 | 4.0 | 0.0 | 3.4 | 30.5 |
| (b) template lookup: family + executed-tool sequence | 94.9 | 99.3 | 39.0 | 99.1 | 81.8 | 95.5 | 94.1 | 88.0 | 98.2 |
| (c) template lookup: family + last executed tool | 85.2 | 50.0 | 50.9 | 87.3 | 75.9 | 72.9 | 60.0 | 46.4 | 78.6 |
| (b+) lookup (b), backing off to (c), then (a) | 97.5 | 99.3 | 62.9 | 99.5 | 89.4 | 97.9 | 95.7 | 91.8 | 97.8 |
| (e) best tool per (b)-key, chosen on the scored set itself (bound, in-sample) | 99.9 | 100.0 | 99.4 | 100.0 | 99.8 | 100.0 | 100.0 | 99.6 | 100.0 |
| AgentDecision-4B | 99.2 | 100.0 | 94.7 | 100.0 | 98.2 | 100.0 | 99.2 | 97.0 | 100.0 |
| SFT v1 (the SFT stage of AgentDecision-4B) | 99.2 | 100.0 | 93.8 | 100.0 | 98.0 | 100.0 | 99.2 | 97.0 | 100.0 |
| 0.8B scoring head (E14 A) | 99.2 | 100.0 | 94.1 | 100.0 | 97.9 | 100.0 | 99.0 | 96.5 | 99.8 |
| Jev | 91.5 | 27.0 | 81.5 | 64.4 | 69.4 | 100.0 | 97.7 | 96.5 | 92.0 |
| StartLux-27B | 93.9 | 26.0 | 85.1 | 63.4 | 71.2 | 100.0 | 98.6 | 96.8 | 92.6 |
| Qwen3-4B (untrained) | 50.1 | 24.0 | 26.7 | 47.3 | 39.9 | 91.0 | 99.4 | 81.2 | 54.4 |

Subset definitions: `all`: every occurrence; `must_ask`: the acceptable set is exactly {ask_user}; `after_off_plan`: after an earlier executed tool outside its acceptable set; `ask_gold`: the canonical next tool is ask_user; `hard`: after an off-plan step, or ask_user next; `state_change_trap`: the next tool changes the world and a state-changing look-alike is offered; `finish_gold`: the next tool is finish; `long_history`: step >= 8 (history window full); `llm_states`: occurrences of the LLM-only (Flash) trajectories.

Pooled occurrence accuracy on all of C: (a) most frequent training label 28.7, (b) template lookup: family + executed-tool sequence 94.2, (c) template lookup: family + last executed tool 83.8, (b+) lookup (b), backing off to (c), then (a) 96.8, (e) best tool per (b)-key, chosen on the scored set itself (bound, in-sample) 99.8, AgentDecision-4B 98.9, SFT v1 (the SFT stage of AgentDecision-4B) 98.9, 0.8B scoring head (E14 A) 98.8, Jev 93.4, StartLux-27B 95.5, Qwen3-4B (untrained) 46.2.

(b') the lookup with the coarse family (`multi` without the request kinds): 88.4 % (key coverage 93.5 %).

How the lookups answered (occurrences): lookup: key 4874, unseen 440; lookup_last: key 5293, unseen 21; lookup_backoff: key 4874, backoff_last 419, prior 21.

## (d) Key coverage on C and accuracy on covered / uncovered occurrences (task mean %)

| lookup | keys in training | keys on C (seen) | covered occurrences | covered questions | covered must-ask | covered after off-plan | (a) most frequent training label: covered / uncovered | (b) template lookup: family + executed-tool sequence: covered / uncovered | (c) template lookup: family + last executed tool: covered / uncovered | (b+) lookup (b), backing off to (c), then (a): covered / uncovered | AgentDecision-4B: covered / uncovered | SFT v1 (the SFT stage of AgentDecision-4B): covered / uncovered | 0.8B scoring head (E14 A): covered / uncovered | Jev: covered / uncovered | StartLux-27B: covered / uncovered | Qwen3-4B (untrained): covered / uncovered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lookup | 1,220 | 412 (187) | 4,874 of 5,314 (91.7 %) | 88.9 % | 99.3 % | 17.9 % | 26.5 / 20.5 | 100.0 / 20.5 | 86.8 / 67.4 | 100.0 / 67.4 | 99.9 / 96.3 | 100.0 / 95.6 | 100.0 / 95.1 | 91.8 / 90.2 | 94.7 / 90.8 | 50.3 / 55.6 |
| lookup_last | 220 | 90 (84) | 5,293 of 5,314 (99.6 %) | 99.5 % | 100.0 % | 94.6 % | 26.6 / 14.3 | 95.3 / 14.3 | 85.6 / 14.3 | 97.9 / 14.3 | 99.2 / 92.9 | 99.2 / 92.9 | 99.2 / 92.9 | 91.5 / 100.0 | 94.0 / 89.3 | 50.4 / 17.9 |

The 440 occurrences with an unseen (b)-key: 72.0 % after an off-plan step, 38.6 % with a long history (step >= 8), 0.5 % must-ask.

## Where the lookup fails and AgentDecision-4B does not

- AgentDecision-4B minus lookup: +4.3 points (95 % task bootstrap [+3.2, +5.4], 200 tasks; descriptive).
- AgentDecision-4B minus lookup_backoff: +1.7 points (95 % task bootstrap [+0.8, +2.5], 200 tasks; descriptive).
- AgentDecision-4B minus lookup_last: +14.0 points (95 % task bootstrap [+12.7, +15.3], 200 tasks; descriptive).
- lookup: wrong where AgentDecision-4B is right at 270 occurrences (0 of them with a covered key); right where it is wrong at 22. By answer key: find_user 142, ship_order 45, finish 44, cancel_order 27, ask_user 8, refund 4. By family: multi 171, cancel 38, ship:no_speed 27, ship:no_name 22, refund:no_method 12. In strata: all 270, must_ask 2, after_off_plan 198, ask_gold 8, hard 198, state_change_trap 76, finish_gold 44, long_history 92, llm_states 36. Most frequent wrong answers: get_orders (gold find_user) x142; get_orders (gold ship_order) x45; get_orders (gold finish) x44; get_orders (gold cancel_order) x27; get_orders (gold ask_user) x8; get_orders (gold refund) x4.
- lookup_backoff: wrong where AgentDecision-4B is right at 140 occurrences (0 of them with a covered key); right where it is wrong at 28. By answer key: find_user 50, finish 32, ship_order 27, get_orders 21, cancel_order 6, ask_user 4. By family: multi 106, ship:no_name 15, ship:no_speed 12, refund:no_method 6, cancel 1. In strata: all 140, must_ask 2, after_off_plan 94, ask_gold 4, hard 94, state_change_trap 33, finish_gold 32, long_history 56, llm_states 43. Most frequent wrong answers: ship_order (gold find_user) x29; find_user (gold finish) x28; find_user (gold get_orders) x20; get_orders (gold ship_order) x15; cancel_order (gold find_user) x13; find_user (gold ship_order) x11; get_orders (gold find_user) x7; ship_order (gold cancel_order) x6; ship_order (gold ask_user) x3; cancel_order (gold finish) x2; get_orders (gold finish) x2; refund (gold ask_user) x1.
- lookup_last: wrong where AgentDecision-4B is right at 827 occurrences (811 of them with a covered key); right where it is wrong at 28. By answer key: finish 271, ship_order 208, cancel_order 125, ask_user 108, find_user 63, refund 29, get_orders 23. By family: multi 632, refund:no_method 96, ship:no_speed 83, ship:no_name 15, cancel 1. In strata: all 827, must_ask 105, after_off_plan 118, ask_gold 108, hard 221, state_change_trap 362, finish_gold 271, long_history 379, llm_states 327. Most frequent wrong answers: find_user (gold finish) x267; cancel_order (gold ship_order) x142; ship_order (gold cancel_order) x79; refund (gold ask_user) x59; find_user (gold ship_order) x51; ship_order (gold ask_user) x49; find_user (gold cancel_order) x46; ship_order (gold find_user) x30; find_user (gold refund) x29; cancel_order (gold find_user) x22; find_user (gold get_orders) x22; get_orders (gold ship_order) x15.

## E21's shifted corpus (renamed core tools, reworded requests): the baselines only

The same baselines (fitted on the same training rows) on `results/explore/e21_shift/c` with E21's subset rules; the decision models' rows on this corpus are E21's readout (App. app:e21).

| row | all (n 1,250; 100 tasks) | must_ask (n 100; 60 tasks) | after_off_plan (n 81; 25 tasks) | renamed (n 1,069; 100 tasks) |
|---|---|---|---|---|
| (a) most frequent training label | 15.9 | 0.0 | 39.8 | 0.0 |
| (b) template lookup: family + executed-tool sequence | 15.9 | 0.0 | 39.8 | 0.0 |
| (c) template lookup: family + last executed tool | 15.9 | 0.0 | 39.8 | 0.0 |
| (b+) lookup (b), backing off to (c), then (a) | 15.9 | 0.0 | 39.8 | 0.0 |
| (e) best tool per (b)-key, chosen on the scored set itself (bound, in-sample) | 95.2 | 77.2 | 99.2 | 94.3 |

(b) key coverage on the shifted corpus: 9.1 % of 1,250 occurrences (renamed subset 10.7 %).
