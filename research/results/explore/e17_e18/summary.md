# E17 and E18: read-out ablation and a deferral-aware objective (exploratory; descriptive)

exploratory (E17, E18), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-08 'E17 and E18: read-out ablation and a deferral-aware objective'; outside every Holm table

Generated 2026-10-08T13:56:51+00:00 from `runs/e17_e18/box`; GPU host hours used 5.738333333333333; Gemini spend logged $0.66.

## E17 training (the generative arm; the head is E14 arm A, not retrained)

- projection: {"setting": "nogc_6144", "s_per_step_smoke": 22.695, "steps_per_epoch": 265, "minutes_per_epoch": 100.23625, "cap_min": 280.0, "max_epochs": 1, "epochs": 1, "projected_minutes": 100.23625, "rule": "2 epochs when both fit within cap_min, else 1, else not run (registration (b) 2); at most max_epochs (the amendment of 2026-10-08 06:10 UTC: one epoch within 280 minutes)"}
- runs: [{"arm": "gen", "exit": 0, "minutes": 52, "epochs": 1, "setting": "nogc_6144", "end_utc": "2026-10-08T08:21:37Z"}]; train_runtime_s 3111.0; train_loss 0.3769; training peak MiB 17789.0
- read-out check: {"model": "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17", "codes_single_token": true, "code_after_prompt_is_bare_code": true, "date_in_render": false, "render_tail": "el only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "pad_token": "<|endoftext|>", "eos_token": "<|im_end|>", "ok": true}

## SFT v1 development files

| model | v1 acc | tool choice | arguments | stop | after off-plan | knockout | ECE |
|---|---|---|---|---|---|---|---|
| head | 99.7 | 99.8 | 100.0 | 99.3 | 99.7 | 100.0 | 0.001 |
| gen | 99.7 | 99.7 | 100.0 | 99.3 | 99.5 | 100.0 | 0.001 |
| gen_raw | 99.7 | 99.7 | 100.0 | 99.3 | 99.5 | 100.0 | 0.032 |
| e18f | 99.7 | 99.9 | 99.9 | 99.3 | 99.7 | 100.0 | 0.002 |

## The common states C

| model | C acc | after off-plan | ask-vs-act | ECE |
|---|---|---|---|---|
| head | 99.2 | 94.1 | 100.0 | 0.011 |
| gen | 99.0 | 92.3 | 100.0 | 0.014 |
| e18f | 99.1 | 86.2 | 100.0 | 0.012 |
| rl | 99.2 | 94.7 | 100.0 | 0.010 |
| sft_v1 | 99.2 | 93.8 | 100.0 | 0.011 |
- gen_vs_head: all -0.2 [-0.4, -0.0]; offplan -1.8 [-3.6, -0.2]; ask +0.0 [+0.0, +0.0]
- gen_vs_rl: all -0.2 [-0.5, +0.0]; offplan -2.4 [-4.4, -0.8]; ask +0.0 [+0.0, +0.0]
- gen_vs_sft_v1: all -0.2 [-0.4, -0.0]; offplan -1.5 [-3.6, +0.7]; ask +0.0 [+0.0, +0.0]
- e18f_vs_head: all -0.1 [-0.3, +0.2]; offplan -8.0 [-14.0, -2.6]; ask +0.0 [+0.0, +0.0]
- e18f_vs_rl: all -0.1 [-0.4, +0.4]; offplan -8.6 [-14.8, -2.9]; ask +0.0 [+0.0, +0.0]
- e18f_vs_sft_v1: all -0.0 [-0.3, +0.4]; offplan -7.6 [-14.0, -1.8]; ask +0.0 [+0.0, +0.0]

## Calibration states (E5's procedure): tau-hat and coverage

| read-out | point | tau-hat | paired | coverage | S1 acc (taken) | all taken right | LLM acc (taken) | cascade acc |
|---|---|---|---|---|---|---|---|---|
| head | tool_select | 0.9999999997650646 | 1125 | 80.0 | 100.0 | True | 100.0 | 100.0 |
| head | tool_params | 0.9999999999989255 | 1328 | 78.8 | 100.0 | True | 100.0 | 100.0 |
| gen | tool_select | 0.9996190716719411 | 1125 | 100.0 | 100.0 | True | 100.0 | 100.0 |
| gen | tool_params | 0.9996734857559204 | 1328 | 88.5 | 100.0 | True | 100.0 | 100.0 |
| 4b_logged | tool_select | 0.4999999999978404 | 1125 | 100.0 | 100.0 | True | 100.0 | 100.0 |
| 4b_logged | tool_params | 0.9999998807907104 | 1328 | 77.5 | 99.9 | False | 100.0 | 99.9 |

## Serving (one session, one A10G)

| served | c=1 p50 | c=1 p90 | c=4 p50 | c=4 p90 | S1/step c=1 p50 | peak MiB |
|---|---|---|---|---|---|---|
| 4b | 318 | 441 | 875 | 1305 | 377 | 22561.0 |
| head | 92 | 119 | 204 | 377 | 157 | 2537.0 |
| gen | 247 | 348 | 796 | 1061 | 487 | 4591.0 |

## E18: the defer head

- weights [3.398634737148808, 1.2891355874699166, 0.3289306209858388, -1.0769827061634323] on ['1', 'log10(1 - p1) clipped at -15', '1[tool choice]', 'ln(n options)']; fit {"weights": [3.398634737148808, 1.2891355874699166, 0.3289306209858388, -1.0769827061634323], "lambda": 1.0, "iterations": 9, "loss": 54.872634923642494, "log_loss_mean": 0.0217723832412569, "n": 2453, "positives": 15, "p_mean": 0.0061149612719203035, "p_positives_mean": 0.2907129905189626}
- by decision point: {"tool_select": {"n": 1125, "positives": 3, "k": 0.0058322, "llm_right": 1125, "ref_right": 1122, "deferred_in_sample": 3, "coverage_in_sample": 0.9973333333333333, "p_quantiles": {"0.05": 0.002015252965410386, "0.5": 0.002015252965476572, "0.95": 0.0021127101817873524, "0.99": 0.0023186684739430544}, "positives_deferred": 2}, "tool_params": {"n": 1328, "positives": 12, "k": 0.0034651068524096384, "llm_right": 1244, "ref_right": 1313, "deferred_in_sample": 645, "coverage_in_sample": 0.5143072289156627, "p_quantiles": {"0.05": 0.0016748774303546223, "0.5": 0.0033103660253844, "0.95": 0.006164976634312454, "0.99": 0.11425667416878543}, "positives_deferred": 11}}

| rule on C | n | coverage | accepted acc | accepted errors | deferred acc | llm_only: deferred | LLM right among them |
|---|---|---|---|---|---|---|---|
| arm A + the defer head: accept iff P(defer) <= k | 5314 | 98.5 | 99.1 | 46 | 78.5 | 6 | 100.0 |
| arm A at its fitted tau-hat | 5314 | 78.0 | 100.0 | 0 | 94.6 | 327 | 100.0 |
| arm A, every decision accepted | 5314 | 100.0 | 98.8 | 63 | – | 0 | – |
| AgentDecision-4B at its frozen threshold | 5314 | 99.9 | 98.9 | 57 | 0.0 | 0 | – |
| the generative arm at its fitted tau-hat | 5314 | 98.4 | 98.7 | 66 | 86.9 | 11 | 100.0 |
| E18-F: accept iff p_defer <= k | 5314 | 100.0 | 98.8 | 63 | – | 0 | – |

## Live: the final harness with E18 (seeds 300500-300699; 200 common tasks; errored {"e18_final": 0, "llm_only": 0, "harness_final": 0, "e14_final": 0})

| arm | success | API $ / ep | q $ / ep | $ / ep incl. GPU | wall s | S1 s | S2 s | S2 calls | tool cov. | arg cov. |
|---|---|---|---|---|---|---|---|---|---|---|
| e18_final | 100.0 | 0.0033 | 0.00134 | 0.0046 | 2.98 | 1.43 | 1.52 | 1.51 | 100.0 | 98.7 |
| llm_only | 99.5 | 0.0431 | 0.00000 | 0.0431 | 13.59 | 0.00 | 11.68 | 7.53 | 0.0 | – |
| harness_final | 100.0 | 0.0058 | 0.00433 | 0.0102 | 9.59 | 3.93 | 5.11 | 3.04 | 100.0 | 85.5 |
| e14_final | 100.0 | 0.0136 | 0.00186 | 0.0154 | 4.12 | 1.35 | 2.74 | 3.60 | 85.5 | 86.4 |
- vs llm_only: success +0.5 [-1.4, +2.8]; saving incl. GPU 89.3 % [88.7, 89.8]; wall -10.61 s
- vs harness_final: success +0.0 [-1.9, +1.9]; saving incl. GPU 54.4 % [51.0, 57.5]; wall -6.62 s
- vs e14_final: success +0.0 [-1.9, +1.9]; saving incl. GPU 70.0 % [68.0, 71.8]; wall -1.15 s

## Registered readings

- e17/tool_select: {"winner": "gen", "coverage": {"head": 0.8, "gen": 1.0}, "accepted_all_right": {"head": true, "gen": true}, "tau": {"head": 0.9999999997650646, "gen": 0.9996190716719411}}
- e17/tool_params: {"winner": "gen", "coverage": {"head": 0.7884036144578314, "gen": 0.884789156626506}, "accepted_all_right": {"head": true, "gen": true}, "tau": {"head": 0.9999999999989255, "gen": 0.9996734857559204}}
- e18/live: {"holds": true, "success_minus_llm_pt": 0.5000000000000004, "tool_coverage": 1.0, "arg_coverage": 0.9868235294117647, "e14_coverage": {"tool": 0.855, "arg": 0.864}, "coverage_higher": true}
- e18/c: {"coverage": 0.9851336093338352, "a_coverage": 0.7803914188934888, "accepted_errors": 46, "a_accepted_errors": 0, "holds": false}
