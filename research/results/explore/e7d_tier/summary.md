# E7d: AgentDecision-4B (tier)

_exploratory (E7d), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-06 'E7d'; outside every Holm table_

| | AgentDecision-4B | AgentDecision-4B (tier) |
|---|---|---|
| held-out tier states, cancel/ship tool choice (%) | 76.5 | 95.3 |
| held-out tier states, other tool choices (%) | 91.6 | 100.0 |
| held-out tier states, arguments (%) | 95.1 | 98.2 |
| E7c calibration states: tool-choice tau-hat | always defer | 0.9999 |
|   coverage at tau-hat (%) | 0.0 | 50.0 |
|   System-1 accuracy, all tool choices (%) | 85.8 | 98.5 |
|   System-1 accuracy, cancel/ship (%) | 79.2 | 96.7 |
| E7c calibration states: argument tau-hat | 1 | 0.9929 |
|   coverage at tau-hat (%) | 70.1 | 90.0 |
|   System-1 accuracy, all arguments (%) | 94.8 | 99.4 |
| corpus C accuracy (%) | 99.2 | 99.1 |
| v1 dev accuracy (%) | 99.7 | -- |

Corpus C, new minus registered checkpoint: -0.1 points (paired task bootstrap 95 % [-0.4, +0.1]).

Calibration states: the first 20 of the 100 E7c calibration tasks (sorted task ids) were re-answered in full within the GPU cap; both checkpoints are fitted on those tasks (the registered checkpoint from its logged E7c answers).
Serving-path check (registered checkpoint re-asked on the first 5 tasks): agreement 100.0 % (tool choice, n 185), 100.0 % (arguments, n 300).

Reference (E7c fit of the registered checkpoint, all 100 tasks): tool-choice tau-hat None, argument tau-hat 1.0.

Reference (E7c fit, Jev on the same calibration states): tool-choice tau-hat 0.91, argument tau-hat 0.87.

Generated 2026-10-07T01:15:06+00:00 from `results/explore/e7_hard/box_e7d`.
