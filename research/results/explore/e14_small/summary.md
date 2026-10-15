# E14: a sub-1B decision model with a scoring head (exploratory; descriptive)

exploratory (E14), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-07 'E14: a sub-1B decision model with a scoring head'; outside every Holm table.

Generated 2026-10-08T00:22:36+00:00 from `runs/e14_small/box`; GPU host hours used 6.481333333333334; Gemini spend logged $5.82. Qwen/Qwen3.5-0.8B (revision 2fc06364) + a fresh Clef joint schema head (Unsloth 2026.10.2, its from-LM recipe and defaults); one seed per arm.

## Training (development data only)

| arm | records | epoch | minutes | s/step | loss first / last | dev acc | selected | dev acc calibrated | dev ECE calibrated | head T |
|---|---|---|---|---|---|---|---|---|---|---|
| e14a | 14787 | epoch1 | 58.57 | 3.796 | [2.1837, 0.1406] | 99.71 | 1 | 99.71 | 0.002 | 0.291 |
| e14a | 14787 | epoch2 | 57.34 | 3.716 | [0.1307, 0.1314] | 99.66 | 1 | 99.71 | 0.002 | 0.291 |
| e14a (fresh head, before training) | | | | | | 39.08 | | | | |
| e14g | 12078 | epoch1 | 12.88 | 1.021 | [1.394, 0.5111] | 86.50 | 1 | 86.50 | 0.040 | 0.825 |
| e14g (fresh head, before training) | | | | | | 35.25 | | | | |
| e14ag | 27137 | epoch1 | 71.03 | 2.51 | [1.873, 0.3902] | 99.66 | 1 | 99.66 | 0.002 | 0.313 |
| e14ag (fresh head, before training) | | | | | | 39.34 | | | | |

Second epoch rule (registration (d)): e14a: {"considered": true, "hours_used": 1.335, "projection_h": 5.811, "cap_h": 10.0, "allowed": true}; e14g: {"considered": false, "hours_used": 3.1, "projection_h": 6.815, "cap_h": 10.0, "allowed": false}; e14ag: {"considered": false, "hours_used": 6.0, "projection_h": 10.684, "cap_h": 10.0, "allowed": false}

## Development read-outs (SFT v1 dev files; the served temperatures)

| model | v1 acc | tool choice | arguments | stop | after off-plan | knockout | ECE |
|---|---|---|---|---|---|---|---|
| e14a | 99.71 | 99.77 | 100.00 | 99.25 | 99.67 | 100.00 | 0.001 |
| e14g | 39.59 | 37.22 | 33.13 | 50.67 | 40.76 | 82.01 | 0.248 |
| e14ag | 99.66 | 99.77 | 99.88 | 99.25 | 99.67 | 100.00 | 0.002 |
| SFT v1 (4B, the same recipe) | 99.71 | 99.66 | 100.00 | 99.40 | 99.83 | 99.85 | 0.029 |
| JevAgentRL-4B (frozen) | 99.71 | 99.66 | 100.00 | 99.40 | 99.83 | 99.54 | 0.004 |
| Qwen3-1.7B | 99.58 | 99.66 | 100.00 | 98.95 | 99.67 | 100.00 | 0.029 |
| Qwen3-0.6B | 99.79 | 99.66 | 100.00 | 99.70 | 99.67 | 100.00 | 0.032 |
| MiniCPM5-1B | 99.28 | 98.98 | 99.88 | 98.95 | 99.01 | 100.00 | 0.034 |

## Unsloth's evaluation sets (G's reproduction check)

| model | typed-decisions | BANKING77 | CLINC150 | mean |
|---|---|---|---|---|
| base | 35.9 | 7.0 | 19.6 | 20.8 |
| e14a | 42.1 | 57.2 | 54.0 | 51.1 |
| e14g | 60.2 | 75.0 | 73.6 | 69.6 |
| e14ag | 60.6 | 71.8 | 75.8 | 69.4 |
| Unsloth's guide, Qwen3.5-0.8B (before -> after) | 36 -> 73 | 7 -> 74 | 19 -> 76 | 20.7 -> 74.3 |

G reproduced (mean >= 69.3 %): True.

## L1: E1's 3,991 fixed decisions (E9's cost rule)

| model | acc | BFCL tool choice | BFCL args | When2Call | $ / 1k decisions | median latency ms | vs AgentDecision-4B (pt) | vs StartLux-0.8B (pt) | on E9's frontier |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B + Clef head, agent states (A) | 67.8 | 80.5 | 70.1 | 47.4 | 0.1231 | 239 | -11.0 [-12.202455524931095, -9.796416938110752] | -2.2 [-3.7584565271861687, -0.6514657980456027] | False |
| Qwen3.5-0.8B + Clef head, generic mix (G) | 52.3 | 81.8 | 41.9 | 38.7 | 0.0997 | 235 | -26.5 [-28.288649461287896, -24.704961162615888] | -17.7 [-19.09295915810574, -16.26158857429216] | False |
| Qwen3.5-0.8B + Clef head, both (AG) | 68.8 | 84.8 | 68.9 | 49.5 | 0.0997 | 236 | -9.9 [-11.049862189927337, -8.694562766224005] | -1.2 [-2.531320471059885, 0.32573289902280134] | False |
| rl (E9) | 78.8 | 88.4 | 85.0 | 54.0 | 0.0934 | 268 | | | |
| sl0.8b (E9) | 70.0 | 88.2 | 67.1 | 54.2 | 0.0127 | 17 | | | |
| sl2b (E9) | 75.7 | 89.9 | 73.1 | 64.1 | 0.0218 | 50 | | | |
| sl4b (E9) | 83.1 | 91.3 | 81.2 | 77.4 | 0.0524 | 111 | | | |
| jev (E9) | 86.4 | 89.9 | 89.6 | 75.4 | 0.0163 | 206 | | | |
| flash (E9) | 88.1 | 90.3 | 93.6 | 74.0 | 0.8108 | 2570 | | | |
| lite (E9) | 83.3 | 90.7 | 89.0 | 62.6 | 0.1861 | 681 | | | |
| intern (E9) | 73.4 | 88.2 | 71.2 | 60.1 | 0.0389 | 91 | | | |

E9's frontier: ['sl0.8b', 'jev', 'flash']; with the E14 points: ['sl0.8b', 'jev', 'flash']. A minus G on E1 (pt): {"est": 15.484840892007016, "ci": [13.730894512653471, 17.188674517664744], "n": 3991, "n_boot": 2000, "seed": 0}; per family: {"toolchoice": [-1.3, [-4.0, 1.3]], "args": [28.3, [26.1, 30.6]], "w2c": [8.7, [3.9, 13.6]]}.

### L1 cost by serving engine (arm A; v1 = one request per pass, the original engine; v2 = queued requests share a pass, warm-up at load, the causal_conv1d kernel if installed)

| run | engine | acc | $ / 1k (E9 rule) | $ / 1k (wall clock) | wall s | median ms | p90 ms | requests > 5 s | agrees with v1 |
|---|---|---|---|---|---|---|---|---|---|
| e14a | v1 | 67.8 | 0.1231 | 0.0317 | 203 | 239 | 246 | 7 | – |
| e14g | v1 | 52.3 | 0.0997 | 0.0259 | 166 | 235 | 242 | 7 | – |
| e14ag | v1 | 68.8 | 0.0997 | 0.0259 | 166 | 236 | 243 | 0 | – |

Engine check on development items (registration 'serving-engine fix' (c)): pass = False; SFT v1 dev: {"n": 2377, "same_argmax": 2377, "differ_near_tie": 0, "differ": 0, "max_abs_prob_diff": 0.04359338191911949, "pass": false, "tie": 0.001, "tol": 0.001, "records": 2065}; E1 dev prefix: {"n": 299, "same_argmax": 298, "differ_near_tie": 0, "differ": 1, "max_abs_prob_diff": 0.5070189797882032, "pass": false, "tie": 0.001, "tol": 0.001, "items": 299, "pred_agree": 298, "conf_max_abs": 0.05844587283795988}.

## L2: the common states C

| model | C acc | after off-plan | ask-vs-act | ECE |
|---|---|---|---|---|
| e14a | 99.2 | 94.1 | 100.0 | 0.011 |
| e14g | 23.9 | 22.4 | 25.0 | 0.530 |
| e14ag | 99.1 | 92.7 | 100.0 | 0.013 |
| rl | 99.2 | 94.7 | 100.0 | 0.010 |
| sft_v1 | 99.2 | 93.8 | 100.0 | 0.011 |
| sl0.8b | 48.3 | 31.5 | 49.4 | 0.056 |
| e5_q06 | 99.1 | 88.7 | 100.0 | 0.011 |
| e5_mc1 | 98.6 | 92.3 | 100.0 | 0.017 |

Paired task bootstrap (B = 2,000, seed 0), model - reference, points; E5's reading rule.

| contrast | all states | after off-plan | ask-vs-act |
|---|---|---|---|
| e14a_vs_rl | +0.0 [-0.2, +0.2] (keeps the reference's accuracy) | -0.6 [-2.2, +0.7] (inconclusive) | +0.0 [+0.0, +0.0] (keeps the reference's accuracy) |
| e14a_vs_sft_v1 | +0.0 [-0.1, +0.2] (keeps the reference's accuracy) | +0.3 [-1.5, +2.1] (inconclusive) | +0.0 [+0.0, +0.0] (keeps the reference's accuracy) |
| e14a_vs_sl0.8b | +50.9 [+48.0, +53.3] (keeps the reference's accuracy) | +62.6 [+54.2, +71.2] (keeps the reference's accuracy) | +50.6 [+43.5, +57.8] (keeps the reference's accuracy) |
| e14a_vs_e5_q06 | +0.1 [-0.2, +0.3] (keeps the reference's accuracy) | +5.4 [+1.1, +10.7] (keeps the reference's accuracy) | +0.0 [+0.0, +0.0] (keeps the reference's accuracy) |
| e14a_vs_e5_mc1 | +0.6 [+0.4, +0.9] (keeps the reference's accuracy) | +1.9 [+0.1, +4.4] (keeps the reference's accuracy) | +0.0 [+0.0, +0.0] (keeps the reference's accuracy) |
| e14g_vs_rl | -75.3 [-76.5, -73.9] (loses accuracy) | -72.3 [-79.7, -64.7] (loses accuracy) | -75.0 [-83.0, -66.0] (loses accuracy) |
| e14g_vs_sft_v1 | -75.2 [-76.5, -73.9] (loses accuracy) | -71.4 [-78.7, -63.7] (loses accuracy) | -75.0 [-83.0, -66.0] (loses accuracy) |
| e14g_vs_sl0.8b | -24.4 [-26.8, -22.0] (loses accuracy) | -9.1 [-17.0, -0.8] (loses accuracy) | -24.4 [-29.6, -19.4] (loses accuracy) |
| e14g_vs_e5_q06 | -75.2 [-76.5, -73.8] (loses accuracy) | -66.3 [-76.2, -55.4] (loses accuracy) | -75.0 [-83.0, -66.0] (loses accuracy) |
| e14g_vs_e5_mc1 | -74.6 [-76.0, -73.2] (loses accuracy) | -69.8 [-77.8, -61.4] (loses accuracy) | -75.0 [-83.0, -66.0] (loses accuracy) |
| e14ag_vs_rl | -0.1 [-0.3, +0.1] (keeps the reference's accuracy) | -2.0 [-4.0, -0.4] (loses accuracy) | +0.0 [+0.0, +0.0] (keeps the reference's accuracy) |
| e14ag_vs_sft_v1 | -0.1 [-0.2, +0.1] (keeps the reference's accuracy) | -1.1 [-3.3, +1.0] (inconclusive) | +0.0 [+0.0, +0.0] (keeps the reference's accuracy) |
| e14ag_vs_sl0.8b | +50.8 [+47.9, +53.3] (keeps the reference's accuracy) | +61.2 [+52.2, +70.3] (keeps the reference's accuracy) | +50.6 [+43.5, +57.8] (keeps the reference's accuracy) |
| e14ag_vs_e5_q06 | -0.0 [-0.4, +0.2] (keeps the reference's accuracy) | +4.0 [-0.8, +9.5] (keeps the reference's accuracy) | +0.0 [+0.0, +0.0] (keeps the reference's accuracy) |
| e14ag_vs_e5_mc1 | +0.5 [+0.3, +0.8] (keeps the reference's accuracy) | +0.4 [-1.7, +3.0] (inconclusive) | +0.0 [+0.0, +0.0] (keeps the reference's accuracy) |
| e14a_vs_e14g | +75.3 [+73.9, +76.5] (keeps the reference's accuracy) | +71.7 [+64.3, +78.9] (keeps the reference's accuracy) | +75.0 [+66.0, +83.0] (keeps the reference's accuracy) |

## Calibration states (E5's offline procedure): tau-hat and coverage

| System-1 | decision point | tau | paired | coverage | S1 acc (taken) | LLM acc (taken) | cascade acc | LLM acc | confidence quantiles |
|---|---|---|---|---|---|---|---|---|---|
| e14a | tool_select | 0.999999999765 | 1125 | 80.0 | 100.00 | 100.00 | 100.00 | 100.00 | {"q5": 1.0, "q25": 1.0, "q50": 1.0, "q75": 1.0, "q95": 1.0} |
| e14a | tool_params | 0.999999999999 | 1328 | 78.8 | 100.00 | 100.00 | 100.00 | 100.00 | {"q5": 1.0, "q25": 1.0, "q50": 1.0, "q75": 1.0, "q95": 1.0} |
| e14g | tool_select | always defers | 1125 | 0.0 | – | – | 100.00 | 100.00 | {"q5": 0.4817, "q25": 0.6226, "q50": 0.8317, "q75": 0.9452, "q95": 0.9763} |
| e14g | tool_params | always defers | 1328 | 0.0 | – | – | 100.00 | 100.00 | {"q5": 0.1433, "q25": 0.2019, "q50": 0.3521, "q75": 0.6607, "q95": 0.9841} |
| e14ag | tool_select | 0.999838426209 | 1125 | 90.0 | 100.00 | 100.00 | 100.00 | 100.00 | {"q5": 0.9997, "q25": 1.0, "q50": 1.0, "q75": 1.0, "q95": 1.0} |
| e14ag | tool_params | 0.999999999994 | 1328 | 70.0 | 100.00 | 100.00 | 100.00 | 100.00 | {"q5": 0.9995, "q25": 1.0, "q50": 1.0, "q75": 1.0, "q95": 1.0} |
| 4b_logged | tool_select | 0.499999999998 | 1125 | 100.0 | 100.00 | 100.00 | 100.00 | 100.00 | {"q5": 1.0, "q25": 1.0, "q50": 1.0, "q75": 1.0, "q95": 1.0} |
| 4b_logged | tool_params | 0.999999880791 | 1328 | 77.5 | 99.90 | 100.00 | 99.92 | 100.00 | {"q5": 0.9995, "q25": 1.0, "q50": 1.0, "q75": 1.0, "q95": 1.0} |

## Serving: the final harness's request stream, one session on the E14 host's A10G

| served | c=1 p50 | c=1 p90 | c=4 p50 | c=4 p90 | S1/step c=1 p50 | serve peak MiB | speed-up vs 4B c=1 / c=4 (p50) |
|---|---|---|---|---|---|---|---|
| 4b | 319 | 443 | 879 | 1298 | 381 | 22505 | – / – |
| e14a | 91 | 118 | 203 | 386 | 153 | 2437 | 3.49 / 4.33 |
| e14g | 91 | 118 | 207 | 377 | 153 | 2437 | 3.51 / 4.25 |

Engines and settings: the E14 arms are served by `serve_jev --backend clef` engine v1 (Unsloth 2026.10.2 loader, the merged bf16 checkpoint, every question of a request in one forward pass, one request per pass under a lock, no prefix cache, flash-linear-attention 0.5.2, the causal_conv1d reference PyTorch path; the engine-v2 check failed, so no v2 timing is used); AgentDecision-4B by `serve_jev --fast` (its session prefix cache, fused multi-question passes, read-out gather); both alone on the same A10G (g5.2xlarge) in one session, `scripts/harness_final.py bench --path new` (System-2 time emulated from the logged workload). E1's timings (L1 above) are engine v1 under E1's 4 workers.

Training peak memory (nvidia-smi, MiB): {"A": 6429.0, "G": 5281.0, "AG": 6485.0}.

## L3 (a): REPLACE at the tool choice, the first 200 fresh Stage-2 tasks

| row | n | success | own-state acc | steps | API $ / ep | wall s / ep | vs frozen (pt) | vs Jev (pt) | vs StartLux-0.8B (pt) |
|---|---|---|---|---|---|---|---|---|---|
| e14a | 200 | 100.0 | 100.0 | 7.5 | 0.0145 | 6.4 | +0.0 [-1.9, +1.9] | +17.0 [+12.1, +22.8] | +82.5 [+76.3, +87.1] |
| e14g | 200 | 0.0 | 6.2 | 1.1 | 0.0007 | 0.9 | -100.0 [-100.0, -97.3] | -83.0 [-87.6, -76.9] | -17.5 [-23.4, -12.5] |
| frozen | 200 | 100.0 | 100.0 | 7.5 | 0.0145 | 8.8 |  |  |  |
| jev | 200 | 83.0 | 91.0 | 7.0 | 0.0144 | 7.3 |  |  |  |
| sl0.8b | 200 | 17.5 | 35.8 | 7.0 | 0.0143 | 7.7 |  |  |  |

## L3 (b): the final harness with arm A against the logged dedicated rows (seeds 300500-300699)

200 common tasks; errored latest records {'e14_final': 0, 'llm_only': 0, 'harness_final': 0}. GPU tariff q = $1.624/h (the logged rows' tariff host, g5.4xlarge) x the episode's wall-clock for the arms with a System-1 model (E9's $2.24208/h as sensitivity); the logged 4B row's own tariff file gives q = $0.00438.

| arm | success | API $ / ep | q $ / ep | $ / ep incl. GPU | $ incl. GPU (E9 rate) | wall s / ep mean (median) | S1 s / ep | S2 s / ep | S2 calls | tool cov. | arg cov. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e14_final | 100.0 | 0.0136 | 0.00186 | 0.0154 | 0.0161 | 4.12 (4.67) | 1.35 | 2.74 | 3.60 | 85.5 | 86.4 |
| llm_only | 99.5 | 0.0431 | 0.00000 | 0.0431 | 0.0431 | 13.59 (8.87) | 0.00 | 11.68 | 7.53 | 0.0 | – |
| harness_final | 100.0 | 0.0058 | 0.00433 | 0.0102 | 0.0118 | 9.59 (7.89) | 3.93 | 5.11 | 3.04 | 100.0 | 85.5 |

| contrast (e14_final minus) | success pt [Newcombe] | saving incl. GPU [boot] | saving incl. GPU, E9 rate | wall s [boot] |
|---|---|---|---|---|
| llm_only | +0.5 [-1.4, +2.8] | 64.2 % [62.4, 66.0] | 62.6 % | -9.47 [-13.29, -6.28] |
| harness_final | +0.0 [-1.9, +1.9] | -51.9 % [-63.3, -41.4] | -36.7 % | -5.47 [-6.53, -4.50] |

## Registered reading rules (registration (j); descriptive)

- i_usable_confidences/e14a: **does not hold** ({"tau": {"tool_select": 0.9999999997650646, "tool_params": 0.9999999999989255}, "coverage": {"tool_select": 0.8, "tool_params": 0.7884036144578314}})
- i_usable_confidences/e14g: **does not hold** ({"tau": {"tool_select": null, "tool_params": null}, "coverage": {"tool_select": 0.0, "tool_params": 0.0}})
- i_usable_confidences/e14ag: **does not hold** ({"tau": {"tool_select": 0.9998384262091261, "tool_params": 0.9999999999941991}, "coverage": {"tool_select": 0.8995555555555556, "tool_params": 0.6995481927710844}})
- ii_near_4b_at_fraction_of_cost/e14a: **does not hold** ({"c_within_1pt": true, "c_lo": -0.0020309903584293833, "c1_p50_below_4b": true, "c1_p50": [91.43303600103536, 319.2705100009334], "e9_cost_below_4b": false, "e9_cost_per_1k": [0.12312684807062388, 0.09341654545698216]})
- iii_fixed_questions_misrank/A_vs_G: **does not hold** ({"g_ahead_on_l1": false, "a_ahead_on_l2": true, "l1_a_minus_g": {"est": 15.484840892007016, "ci": [13.730894512653471, 17.188674517664744], "n": 3991, "n_boot": 2000, "seed": 0}, "l2_a_minus_g": {"delta": 0.7526047682324998, "lo": 0.7390476539165127, "hi": 0.764909747913374, "n_tasks": 200, "reading": "keeps the reference's accuracy"}})

