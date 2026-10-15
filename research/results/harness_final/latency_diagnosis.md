# Why the cheaper profiles were slower: latency diagnosis of the Stage-2 logs

Label: exploratory, descriptive (post hoc; PREREG_A7_STAGE2 Deviations 2026-10-05 harness_final); no test. Generated 2026-10-05T17:26:50+00:00 at git f42839905d4f by `python scripts/harness_final.py diagnose`; inputs: `runs/stage2_jevgym/llm_only.jsonl` (fba5397498f8d4a0), `runs/stage2_jevgym/best_single.jsonl` (43495c708896d392), `runs/stage2_jevgym/jevagent_best.jsonl` (defe93054b0b1c91), `runs/stage2_jevgym/jevagent_best_rl.jsonl` (97da04a8e4a71976), `runs/stage2_dev_tariff_rl/jevagent_best_rl.jsonl` (0c3806538a628d12).

## 1. Where an episode's time goes (s per episode, means)

| arm | conc. | n | success | wall | System-1 | System-2 | rest (tools, harness, cache) | steps | S1 requests | S1 questions | S2 calls | S2 ms / call (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `llm_only` | 8 | 500 | 99.4 | 6.83 | 0.00 | 6.80 | 0.02 | 7.52 | 0.00 | 0.00 | 7.52 | 905 (771) |
| `best_single` | 8 | 500 | 99.8 | 7.61 | 2.32 | 5.60 | -0.31 | 7.51 | 11.31 | 11.31 | 6.23 | 898 (769) |
| `jevagent_best` | 8 | 500 | 97.2 | 9.21 | 4.26 | 5.09 | -0.13 | 7.54 | 21.16 | 21.16 | 5.37 | 948 (799) |
| `jevagent_best_rl` | 4 | 500 | 100.0 | 10.21 | 7.40 | 2.37 | 0.44 | 7.50 | 24.62 | 24.62 | 3.03 | 782 (612) |
| `dev_tariff_rl` | None | 100 | 100.0 | 11.63 | 6.21 | 4.07 | 1.35 | 7.50 | 24.61 | 24.61 | 3.10 | 1313 (859) |

System-1 and System-2 calls of one episode are sequential in the reference loop, so wall ≈ System-1 + System-2 + rest. `jevagent_best_rl` and `dev_tariff_rl` ran on the GPU host (AWS, concurrency 4, one JevAgentRL server); the other arms ran in another invocation and location (concurrency 8, hosted Jev): their System-2 calls were slower per call, so cross-invocation differences mix the arm with the serving conditions.

## 2. System-1 requests by decision point (per episode)

| arm | decision point | requests | questions | s | ms / request |
|---|---|---|---|---|---|
| `best_single` | tool_params | 3.80 | 3.80 | 0.75 | 199 |
| `best_single` | tool_select | 7.51 | 7.51 | 1.57 | 209 |
| `jevagent_best` | reflection | 6.54 | 6.54 | 1.31 | 200 |
| `jevagent_best` | tool_params | 7.08 | 7.08 | 1.40 | 198 |
| `jevagent_best` | tool_select | 7.54 | 7.54 | 1.54 | 205 |
| `jevagent_best_rl` | reflection | 6.50 | 6.50 | 1.65 | 253 |
| `jevagent_best_rl` | tool_params | 10.62 | 10.62 | 2.74 | 258 |
| `jevagent_best_rl` | tool_select | 7.50 | 7.50 | 3.02 | 402 |
| `dev_tariff_rl` | reflection | 6.50 | 6.50 | 1.33 | 205 |
| `dev_tariff_rl` | tool_params | 10.61 | 10.61 | 2.23 | 210 |
| `dev_tariff_rl` | tool_select | 7.50 | 7.50 | 2.65 | 354 |

Every System-1 request carried exactly one question (the registered call graph, A6.22 (a)).

## 3. Per step

| arm | S1 requests / step | S1 ms / step | S2 ms / step | state tokens mean (p90) |
|---|---|---|---|---|
| `llm_only` | 0.00 | 0 | 905 | – (–) |
| `best_single` | 1.51 | 310 | 745 | 2726 (3345) |
| `jevagent_best` | 2.81 | 564 | 675 | 2752 (3426) |
| `jevagent_best_rl` | 3.28 | 987 | 316 | 790 (1456) |
| `dev_tariff_rl` | 3.28 | 828 | 543 | 790 (1453) |

## 4. Queueing at the single JevAgentRL server (`jevagent_best_rl`)

12222 requests over 1279 s; the server was busy (≥ 1 request in flight) 97.3 % of the time. Reconstructed mean service 102 ms and queueing 199 ms per request: 66.1 % of the System-1 latency was waiting for the other episodes' requests.

| decision point | requests | service ms mean (median) | queueing ms mean |
|---|---|---|---|
| tool_select | 3750 | 220 (219) | 183 |
| tool_params | 5227 | 48 (47) | 209 |
| reflection | 3245 | 53 (50) | 201 |

## 4. Queueing at the single JevAgentRL server (`dev_tariff_rl`)

2445 requests over 284 s; the server was busy (≥ 1 request in flight) 87.9 % of the time. Reconstructed mean service 102 ms and queueing 148 ms per request: 59.3 % of the System-1 latency was waiting for the other episodes' requests.

| decision point | requests | service ms mean (median) | queueing ms mean |
|---|---|---|---|
| tool_select | 750 | 220 (220) | 133 |
| tool_params | 1046 | 48 (48) | 157 |
| reflection | 649 | 53 (50) | 152 |

## 5. Gaps between calls (s per episode)

| arm | in-step s1->s1 | in-step s1->s2 | in-step s2->s1 | step boundary s1->s1 | step boundary s1->s2 | step boundary s2->s1 | step boundary s2->s2 |
|---|---|---|---|---|---|---|---|
| `llm_only` | – | – | – | – | – | – | 0.021 |
| `best_single` | 0.023 | 0.015 | – | 0.008 | 0.000 | 0.025 | – |
| `jevagent_best` | 0.082 | 0.014 | 0.005 | 0.019 | – | 0.019 | – |
| `jevagent_best_rl` | 0.182 | 0.230 | – | 0.047 | – | 0.022 | – |
| `dev_tariff_rl` | 0.182 | 1.169 | – | 0.046 | – | 0.022 | – |

## 6. The reflection trigger

| arm | questions / episode | questions | fired | max p(yes) |
|---|---|---|---|---|
| `jevagent_best` | 6.54 | 3270 | 469 | 0.8300 |
| `jevagent_best_rl` | 6.50 | 3250 | 0 | 0.4006 |
| `dev_tariff_rl` | 6.50 | 650 | 0 | 0.2808 |

## 7. Reading

- **The LLM calls JevAgentRL removed were cheap in time.** `jevagent_best_rl` made 3.03 System-2 calls per episode instead of 7.52 (2.37 s instead of 6.80 s), but it spent 7.40 s in 24.62 System-1 requests (3.28 per step, one question each).
- **Request count.** Per episode: reflection 6.50, tool choice 7.50, arguments 10.62 requests. The reflection trigger was asked 3250 times and never fired (max p(yes) 0.4006 vs τ 0.5); 4.00 steps per episode sent two or more argument questions one request at a time although they share the state.
- **Queueing on one GPU.** Four episodes shared one server that answers one request at a time: it was busy 97.3 % of the run and 66.1 % of the System-1 latency was queueing (mean service 102 ms, queueing 199 ms per request). The tool choice (69 options: a two-round knockout over chunks of 52 with every option's description re-read after the state at every step) is the most expensive request (service 220 ms).
- **A closed loop at saturation.** With 4 episodes, a server busy time of 332 ms per step and 375 ms of other work per step, the interactive-system law R = N·S − Z predicts 952 ms of System-1 time per step; the logs show 987 ms. One GPU serving this path completes at most 3.0 steps per second, so the episode time is set by the server's work per step times the number of episodes sharing it, not by the per-request latency.
- **Rest.** 0.44 s per episode outside model calls vs 0.02 s for LLM-only: the System-1 decision cache (one SQLite read, insert and commit per request), the gold bookkeeping of the extra decision logs and the per-request HTTP handling.
