# Stage 2: the confirmatory family and the secondary table (A7.4 as amended)

## Family (Holm, FWER 0.05)

| test | arm → reference | n′ | clauses | decisive p | Holm α | verdict |
|---|---|---|---|---|---|---|
| S2-1 | `jevagent_best` → `llm_only` | 500 | ni: p 0.1779, est -2.2 [-3.8, -0.8], boundary -3.0 (not NI: loss of 2.2 pt (95 % CI entirely below 0)); cost: p <0.0001, est 0.01682 [0.01572, 0.01792], boundary 0 (cost saving established) | 0.1779 | 0.0250 | not rejected: the registered joint claim was not established |
| S2-2 | `jevagent_best` → `best_single` | 500 | ni: p 0.3255, est -2.6 [-4.2, -1.2], boundary -3.0 (not NI: loss of 2.6 pt (95 % CI entirely below 0)); cost: p <0.0001, est 0.009066 [0.008055, 0.01008], boundary 0 (cost saving established) | 0.3255 | 0.0500 | not rejected: the registered joint claim was not established |
| S2-3 | `rl` → `intern, jev` | 200 | intern: p <0.0001, est +24.0 [+22.1, +25.9], boundary +0.0 (accuracy gain over Intern-Decision-4B established); jev: p <0.0001, est +7.7 [+6.3, +9.1], boundary -5.0 (non-inferiority to Jev (no more than 5 pt less accurate) established) | <0.0001 | 0.0125 | rejects: the claim holds |
| S2-4 | `jevagent_best_rl` → `llm_only` | 500 | ni: p 0.0001, est +0.6 [+0.0, +1.4], boundary -3.0 (NI); cost: p <0.0001, est 0.0335 [0.03248, 0.03453], boundary 0 (cost saving established) | 0.0001 | 0.0167 | rejects: the claim holds |

## Secondary (Holm within, six rows; A7.9 (c), A7.12 (g), A7.18 (f))

| test | arm → reference | n′ | clauses | decisive p | Holm α | verdict |
|---|---|---|---|---|---|---|
| S2-1T | `jevagent_best` → `llm_only` | 500 | latency: p 1.0000, est -2.542 [-2.873, -2.211], boundary 0 (the wrong direction: latency saving reversed (95 % CI entirely below 0)) | 1.0000 | 0.0167 | not rejected: the registered joint claim was not established |
| S2-1L | `jevagent_best_s2lite` → `llm_only_s2lite` | – | – | 1.0000 | 0.0125 | not tested (no decision point qualified in Stage 1; the best-practice profile is LLM-only) -- A7.1 (g) 5 |
| S2-L3a | `jevagent_best` → `llm_only` | 560 | ni: p 0.0274, est -1.6 [-5.0, +1.8], boundary -5.0 (NI); cost: p 0.9728, est -0.003357 [-0.006771, 5.688e-05], boundary 0 (no cost saving established at this n) | 0.9728 | 0.0083 | not rejected: the registered joint claim was not established |
| S2-L3b | `jevagent_best` → `llm_react` | – | – | 1.0000 | 0.0500 | not run (A7.1 (h): E6 not run in Stage 1) |
| S2-1X | `jevagent_best_x` → `llm_only_x` | 500 | ni: p 0.0001, est +7.2 [+3.8, +10.6], boundary -3.0 (NI); cost: p 1.0000, est -0.000423 [-0.0004519, -0.0003942], boundary 0 (the wrong direction: cost saving reversed (95 % CI entirely below 0)) | 1.0000 | 0.0250 | not rejected: the registered joint claim was not established |
| S2-4S | `jevagent_best_sl4b` → `llm_only` | 500 | ni: p 0.0001, est +0.0 [-0.6, +0.6], boundary -3.0 (NI); cost: p 0.9974, est -0.000594 [-0.001008, -0.0001797], boundary 0 (the wrong direction: cost saving reversed (95 % CI entirely below 0)) | 0.9974 | 0.0100 | not rejected: the registered joint claim was not established |

## S2-3: the common-state corpus (A7.10 (c))

5314 occurrences, 3939 unique questions, 200 eligible tasks (0 excluded); source weights {"llm_only": 0.282, "jev_ts_replace_f": 0.262, "intern_ts_replace_f": 0.174, "rl_ts_replace_f": 0.282}.

| stratum | occurrences | tasks | rl | intern | jev |
|---|---|---|---|---|---|
| on_plan | 4928 | 200 | 99.9 | 76.1 | 92.0 |
| after_off_plan | 386 | 57 | 94.7 | 47.6 | 81.5 |
| llm_only_source | 1498 | 200 | 100.0 | 76.3 | 92.0 |

## Descriptive rows (no test)

| row | n′ | Δ success [Newcombe 95 %] | savings [95 %] | Δ latency s | S1 / S2 calls (arm) |
|---|---|---|---|---|---|
| LOGO without reflection | 200 | +3.5 [+0.9, +7.0] | -35.2 % [-42.0, -28.5] | -1.758 | 11.42 / 6.15 |
| LOGO without tool | 200 | +2.0 [-0.8, +5.4] | -69.2 % [-76.7, -61.6] | -0.7106 | 6.565 / 8.505 |
| llm_only_xsmall vs llm_only_x | 500 | +20.4 [+16.9, +24.2] | 75.3 % [+74.8, +75.7] | -10.15 | 0 / 7.634 |

best_control: no listed control satisfied the prespecified Stage-1 screen

## JevAgentRL attribution (A7.13; descriptive)

| row | acc on C | acc on C, after off-plan | ECE | live success (first 100) | own-state acc | E1 acc | status |
|---|---|---|---|---|---|---|---|
| jev | 91.5 | 81.5 | 0.04275 | 84.0 | 90.9 | 89.9 |  |
| intern | 75.2 | 47.6 | 0.0493 | 30.0 | 63.8 | 88.2 |  |
| base | 50.1 | 26.7 | 0.1597 | 13.0 | 38.4 | 87.5 |  |
| sft_v1 | 99.2 | 93.8 | 0.01059 | 100.0 | 100.0 | 86.9 |  |
| sft_v1_opd | – | – | – | – | – | – | not run (dropped by the chain): OPD does not precede the frozen checkpoint in its chain (freeze record, A7.13 (i)) |
| frozen | 99.2 | 94.7 | 0.009999 | 100.0 | 100.0 | 88.4 |  |

## StartLux-Decision size ladder on C (A7.18 (e) 2; descriptive)

| row | acc on C (Δ vs Jev [bootstrap]) | acc on C, after off-plan | ECE |
|---|---|---|---|
| sl0.8b | 48.3 (-43.1 [-46.3, -39.9]) | 31.5 | 0.05635 |
| sl2b | 68.4 (-23.0 [-25.2, -20.9]) | 47.2 | 0.1767 |
| sl9b | 91.8 (+0.3 [-0.7, +1.4]) | 74.3 | 0.07087 |

## Static vs in the loop (A7.18 (h); descriptive)

| row | E1 acc (Δ vs Jev [Newcombe]) | acc on C (Δ vs Jev [bootstrap]) | live success, first 200 (Δ vs Jev [Newcombe]) | own-state acc | status |
|---|---|---|---|---|---|
| jev | 89.9 | 91.5 | 83.0 | 91.0 |  |
| intern | 88.2 (-1.7 [-2.9, -0.5]) | 75.2 (-16.3 [-18.1, -14.5]) | 26.5 (-56.5 [-62.8, -49.0]) | 62.2 |  |
| frozen | 88.4 (-1.5 [-3.4, +0.3]) | 99.2 (+7.7 [+6.4, +9.1]) | 100.0 (+17.0 [+12.1, +22.8]) | 100.0 |  |
| sl4b | 91.3 (+1.3 [+0.1, +2.6]) | 89.1 (-2.4 [-3.5, -1.3]) | 60.0 (-23.0 [-28.9, -17.0]) | 81.6 |  |
| sl27b | 90.8 (+0.8 [-0.5, +2.1]) | 93.9 (+2.5 [+1.6, +3.3]) | 63.0 (-20.0 [-25.7, -14.4]) | 87.7 |  |

Kendall's τ-b of the rank orders over jev, intern, frozen, sl4b, sl27b: E1 vs C 0, E1 vs live -0.2. StartLux E1 rows: public BFCL / When2Call items; overlap with the vendor's training data cannot be excluded (A7.18 (c))

## JevAgentRL-v2 — exploratory (A7.19): descriptive, outside every Holm table (A7.19 (f))

JevAgentRL (S2-3, S2-4) is A7.10's checkpoint; v2 is the exploratory Miles track (freeze record {"name": "JevAgentRL-v2 (Miles track; exploratory, A7.19)", "frozen_from": "sft_v2 (checkpoints/r6/sft_v2)", "label": "GO: the frozen JevAgentRL-v2 passes G1-G6 (development; generator-adapted vs off-the-shelf)", "frozen_at": "2026-10-04T00:53:20Z"}). No test.

| row | acc on C (Δ vs Jev [bootstrap]) | acc on C, after off-plan | ECE | live success, first 200 (Δ vs Jev [Newcombe]) | own-state acc | steps |
|---|---|---|---|---|---|---|
| JevAgentRL (A7.10) | 99.2 (+7.7 [+6.4, +9.1]) | 94.7 | 0.009999 | 100.0 (+17.0 [+12.1, +22.8]) | 100.0 | – |
| JevAgentRL-v2, exploratory (A7.19) | 99.4 (+7.9 [+6.5, +9.3]) | 95.0 | 0.009568 | 100.0 (+17.0 [+12.1, +22.8]) | 100.0 | 7.5 |

| profile row → `llm_only` | n′ | success arm / ref | Δ success [Newcombe 95 %] | cost savings incl. tariff [95 %] | q $ / episode |
|---|---|---|---|---|---|
| `jevagent_best_rl` (S2-4, A7.10) | 500 | 100.0 / 99.4 | +0.6 [-0.3, +1.7] | 81.1 % [+80.7, +81.6] | 0.001962 |
| `jevagent_best_v2` (exploratory (A7.19)) | 500 | 100.0 / 99.4 | +0.6 [-0.3, +1.7] | 54.8 % [+53.8, +55.8] | 0.01236 |

v2 tariff sensitivity (cost savings): zero (q 0): 84.7 % [+84.3, +85.2]; busy_throughput (q 0.01202): 55.6 % [+54.6, +56.6]; jev_api_price (q 0.0008838): 82.6 % [+82.1, +83.0]; reference_host (q 0.003656): 75.9 % [+75.4, +76.4]. Δ latency 15.78 s [15.11, 16.46] (descriptive; serving host and queueing).

## Intervention reporting (A7.16 (d); descriptive)

| arm | n | success | tool cov. | arg cov. | S2 calls / ep. | calls avoided | trigger questions | fires | fire rate | tasks with a fire |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 500 | 99.4 | 0.0 | 0.0 | 7.518 | – | 0 | 0 | – | 0 |
| best_single | 500 | 99.8 | 28.9 | 23.1 | 6.23 | 1.288 | 0 | 0 | – | 0 |
| jevagent_best | 500 | 97.2 | 52.5 | 51.4 | 5.372 | 2.146 | 3270 | 469 | 14.3 | 416 |
| jevagent_minus_tool | 200 | 98.5 | 0.0 | 0.0 | 8.505 | -0.987 | 1313 | 188 | 14.3 | 167 |
| jevagent_best_rl | 500 | 100.0 | 99.9 | 75.0 | 3.032 | 4.486 | 3250 | 0 | 0.0 | 0 |
| jevagent_best_sl4b | 500 | 99.4 | 17.3 | 6.8 | 7.478 | 0.04 | 3254 | 1 | 0.0 | 1 |
| jevagent_default | 500 | 99.8 | 28.2 | 22.7 | 6.262 | 1.256 | 0 | 0 | – | 0 |
| llm_only_x | 500 | 78.2 | 0.0 | 0.0 | 7.412 | – | 0 | 0 | – | 0 |
| jevagent_best_x | 500 | 85.4 | 0.0 | 0.0 | 8.374 | -0.962 | 3238 | 449 | 13.9 | 424 |
| llm_only_xsmall | 500 | 98.6 | 0.0 | 0.0 | 7.634 | -0.222 | 0 | 0 | – | 0 |
| telecom_llm_only | 560 | 24.3 | 0.0 | 0.0 | 16.72 | – | 0 | 0 | – | 0 |
| telecom_jevagent_best | 560 | 22.7 | 8.1 | 0.7 | 16.81 | -0.08571 | 0 | 0 | – | 0 |

Jev profile (S2-1 arm): 14 failures, the trigger fired in 14 of them.
First 200 fresh seeds, successes: llm_only 199, jevagent_minus_tool 197, best_single 200, jevagent_best 193; interaction contrast (both - cascade) - (trigger - LLM-only) -2.5 [-5.0, +0.0] (paired task bootstrap; descriptive, not a test).
