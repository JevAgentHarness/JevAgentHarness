# E2 — StartLux-Decision size ladder, live (exploratory)

exploratory (E2): descriptive, no test; revises no registered result. Live: tool_select replace, Flash, the first 200 fresh Stage-2 tasks (seeds 300000–300199), latest record per task, errored tasks left out. E1: BFCL tool_select items of the C8 population. C: the common-state corpus accuracy as reported in results/stage2/stage2.json.

| row | size (B) | source | E1 acc (Δ vs Jev [Newcombe]) | C acc (Δ vs Jev) | live success (Δ vs Jev [Newcombe]) | own-state acc | steps | S1 / S2 calls | API $ / ep. | n live |
|---|---|---|---|---|---|---|---|---|---|---|
| sl0.8b | 0.8 | E2 | 88.2 (-1.7 [-3.2, -0.3]) | 48.3 (-43.1) | 17.5 (-65.5 [-71.4, -58.1]) | 35.8 | 7.0 | 7.0 / 7.0 | 0.0143 | 200 |
| sl2b | 2.0 | E2 | 89.9 (0.0 [-1.5, 1.5]) | 68.4 (-23.0) | 6.5 (-76.5 [-81.7, -69.4]) | 59.5 | 3.8 | 3.8 / 3.8 | 0.0049 | 200 |
| sl4b | 4.0 | A7.18 | 91.3 (1.3 [0.1, 2.6]) | 89.1 (-2.4) | 60.0 (-23.0 [-28.9, -17.0]) | 81.6 | 6.2 | 6.2 / 6.2 | 0.0116 | 200 |
| sl9b | 9.0 | E2 | 92.2 (2.2 [1.1, 3.4]) | 91.8 (0.3) | 74.0 (-9.0 [-13.3, -4.9]) | 91.0 | 7.1 | 7.1 / 7.1 | 0.0142 | 200 |
| sl27b | 27.0 | A7.18 (27B Q8_0) | 90.8 (0.8 [-0.5, 2.1]) | 93.9 (2.5) | 63.0 (-20.0 [-25.7, -14.4]) | 87.7 | 6.8 | 6.8 / 6.8 | 0.0133 | 200 |
| jev | – | A7.10 | 89.9 | 91.5 | 83.0 | 91.0 | 7.0 | 7.0 / 7.0 | 0.0144 | 200 |
| intern | – | A7.10 | 88.2 (-1.7 [-2.9, -0.5]) | 75.2 (-16.3) | 26.5 (-56.5 [-62.8, -49.0]) | 62.2 | 4.6 | 9.3 / 4.6 | 0.0080 | 200 |
| frozen | – | A7.10 | 88.4 (-1.5 [-3.4, 0.3]) | 99.2 (7.7) | 100.0 (17.0 [12.1, 22.8]) | 100.0 | 7.5 | 7.5 / 7.5 | 0.0145 | 200 |

Kendall's τ-b (size vs each StartLux measure; the *_all_rows entries rank every row with all three measures): e1_acc 0.60 (n 5); corpus_acc 1.00 (n 5); live_success 0.60 (n 5); own_acc 0.80 (n 5); e1_vs_live_all_rows 0.15 (n 8); corpus_vs_live_all_rows 0.71 (n 8).

StartLux E1 rows: public BFCL / When2Call items; overlap with the vendor's training data cannot be excluded (A7.18 (c)). The 27B row is the Q8_0 GGUF (A7.18 (g)). StartLux System-1 calls are priced at an exact 0 (API cost only).
