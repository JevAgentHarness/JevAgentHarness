# Offline serving benchmark: replayed System-1 requests (no LLM call)

Label: exploratory, descriptive (post hoc; PREREG_A7_STAGE2 Deviations 2026-10-05 harness_final); no test. Generated 2026-10-05T19:11:05+00:00 by `python scripts/harness_final.py bench-report` from the GPU host's `bench_*.json` (`scripts/aws/harness_final_gpu.sh bench`). Workload: `runs/stage2_dev_tariff_rl/jevagent_best_rl.jsonl` (100 development episodes; 2461 requests in the old call graph, 1398 in the new one). Episodes replayed `c` at a time, steps in order, with each step's logged System-2 time (capped at 5 s) and the episode's rest (capped at 2 s) slept in between; c = 1 replays the first 25 episodes.

Servers: `v0` = the Stage-2 server (serve_jev defaults); `v2` = `serve_jev --fast` (fused multi-question pass, length-grouped suffix batches, LM head at the read-out positions only, prompts tokenized outside the engine lock); `p` = the same with device-synchronised phase timers. Host: one NVIDIA A10G 24 GB (g5.4xlarge); S2-4 ran on one L40S. Two fast replicas on the one GPU ran out of memory (not measured). Call graphs: `old` = reflection question + tool choice + one request per argument question (Stage 2, S2-4); `new` = tool choice + all argument questions of the step in one request (the final harness).

| server | graph | c | episodes | S1 requests / ep. | S1 ms / step (p50) | S1 s / ep. | emulated wall s / ep. | episodes / GPU-hour | cache-hit share | decisions differing from the logged ones / questions |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | new | 4 | 100 | 13.98 | 2440 (2562) | 18.30 | 21.63 | 659 | 37.7 % | 1 / 1811 |
| v0 | new | 8 | 100 | 13.98 | 5206 (5577) | 39.04 | 42.38 | 662 | 37.7 % | 1 / 1811 |
| v0 | old | 1 | 25 | 15.24 | 648 (592) | 3.24 | 6.15 | 585 | 45.4 % | 0 / 381 |
| v0 | old | 4 | 100 | 24.61 | 2409 (2479) | 18.07 | 21.40 | 666 | 54.6 % | 1 / 2461 |
| v0 | old | 8 | 100 | 24.61 | 5166 (5446) | 38.74 | 42.08 | 668 | 54.6 % | 1 / 2461 |
| v0p | new | 1 | 25 | 9.00 | 637 (548) | 3.19 | 6.09 | 591 | 30.8 % | 0 / 281 |
| v0p | old | 1 | 25 | 15.24 | 659 (609) | 3.30 | 6.20 | 580 | 45.4 % | 0 / 381 |
| v2 | new | 1 | 25 | 9.00 | 455 (374) | 2.28 | 5.18 | 694 | 30.8 % | 0 / 281 |
| v2 | new | 2 | 100 | 13.98 | 784 (762) | 5.88 | 9.21 | 777 | 37.7 % | 3 / 1811 |
| v2 | new | 4 | 100 | 13.98 | 1623 (1706) | 12.17 | 15.50 | 919 | 37.7 % | 3 / 1811 |
| v2 | new | 8 | 100 | 13.98 | 3564 (3837) | 26.73 | 30.06 | 934 | 37.7 % | 3 / 1811 |
| v2 | old | 4 | 100 | 24.61 | 1719 (1785) | 12.89 | 16.23 | 878 | 54.6 % | 3 / 2461 |
| v2p | new | 1 | 25 | 9.00 | 460 (379) | 2.30 | 5.21 | 691 | 30.8 % | 0 / 281 |

The logged decisions are the harness's own outcomes in the replayed Stage-2 development run (one L40S): the chosen value and whether the frozen cascade acted on it (tool choice τ ≈ 0.5, arguments τ ≈ 1 − 1.2e-7, the reflection trigger τ 0.5). A differing decision on this host can come from the GPU (A10G vs L40S bf16 numerics), the batch composition, or a knockout near-tie; the cross-replay table below isolates the server and call graph on one host.

Per decision point (ms per request, mean; server time in parentheses):

| server | graph | c | tool_select | tool_params | reflection |
|---|---|---|---|---|---|
| v0 | new | 4 | 1428 (1426) | 1172 (1170) | – |
| v0 | new | 8 | 2927 (2925) | 2637 (2635) | – |
| v0 | old | 1 | 488 (487) | 69 (68) | 92 (90) |
| v0 | old | 4 | 981 (979) | 623 (622) | 630 (629) |
| v0 | old | 8 | 1843 (1841) | 1456 (1454) | 1457 (1455) |
| v0p | new | 1 | 514 (512) | 154 (153) | – |
| v0p | old | 1 | 495 (493) | 72 (70) | 93 (92) |
| v2 | new | 1 | 370 (369) | 106 (104) | – |
| v2 | new | 2 | 529 (527) | 295 (294) | – |
| v2 | new | 4 | 964 (962) | 762 (761) | – |
| v2 | new | 8 | 2009 (2008) | 1799 (1797) | – |
| v2 | old | 4 | 664 (662) | 459 (457) | 468 (466) |
| v2p | new | 1 | 374 (372) | 108 (106) | – |

Work per step, hardware-free (`python scripts/harness_final.py positions`: the same 100 episodes replayed through a tiny random model with the served model's tokenizer, so token positions, forward passes and LM-head rows are exact; GFLOP = Qwen3-4B dense matmuls):

| server / graph | requests / step | forward passes / step | positions / step | LM-head rows / step | GFLOP / step | prefix-cache hit share |
|---|---|---|---|---|---|---|
| v0_old | 3.28 | 5.28 | 2534 | 1121 | 19286 | 54.6 % |
| v0_new | 1.86 | 4.38 | 2680 | 1524 | 20658 | 37.7 % |
| v2_new | 1.86 | 4.41 | 1924 | 4 | 13985 | 37.7 % |

Server phase profile (ms per step, device-synchronised timers; profiled servers only):

| server | graph | c | encode | lookup | prefill | store | suffix | sum | S1 ms / step (client) |
|---|---|---|---|---|---|---|---|---|---|
| v0p | new | 1 | 6.9 | 2.3 | 157.0 | 2.4 | 462.4 | 631.0 | 637.2 |
| v0p | old | 1 | 8.1 | 3.4 | 247.7 | 3.7 | 386.6 | 649.4 | 659.1 |
| v2p | new | 1 | 1.3 | 2.2 | 329.1 | 2.7 | 113.4 | 448.7 | 460.2 |

Decision equivalence between replays (every shared question; the frozen cascade's accept / defer and argmax):

| replay | vs | questions | decision flips | max abs Δp |
|---|---|---|---|---|
| v0/new/4 | v0/old/1 | 281 | 0 | 0.0038 |
| v0/new/4 | v0/old/4 | 1811 | 2 | 0.79 |
| v0/new/4 | v0/old/8 | 1811 | 2 | 0.79 |
| v0/new/8 | v0/old/1 | 281 | 0 | 0.0038 |
| v0/new/8 | v0/old/4 | 1811 | 2 | 0.79 |
| v0/new/8 | v0/old/8 | 1811 | 2 | 0.79 |
| v0p/new/1 | v0/old/1 | 281 | 0 | 0.0038 |
| v0p/new/1 | v0/old/4 | 281 | 0 | 0.0038 |
| v0p/new/1 | v0/old/8 | 281 | 0 | 0.0038 |
| v0p/old/1 | v0/old/1 | 381 | 0 | 0 |
| v0p/old/1 | v0/old/4 | 381 | 0 | 0.021 |
| v0p/old/1 | v0/old/8 | 381 | 0 | 0.021 |
| v2/new/1 | v0/old/1 | 281 | 0 | 0.19 |
| v2/new/1 | v0/old/4 | 281 | 0 | 0.19 |
| v2/new/1 | v0/old/8 | 281 | 0 | 0.19 |
| v2/new/2 | v0/old/1 | 281 | 0 | 0.19 |
| v2/new/2 | v0/old/4 | 1811 | 2 | 0.29 |
| v2/new/2 | v0/old/8 | 1811 | 2 | 0.29 |
| v2/new/4 | v0/old/1 | 281 | 0 | 0.19 |
| v2/new/4 | v0/old/4 | 1811 | 2 | 0.29 |
| v2/new/4 | v0/old/8 | 1811 | 2 | 0.29 |
| v2/new/8 | v0/old/1 | 281 | 0 | 0.19 |
| v2/new/8 | v0/old/4 | 1811 | 2 | 0.29 |
| v2/new/8 | v0/old/8 | 1811 | 2 | 0.29 |
| v2/old/4 | v0/old/1 | 381 | 0 | 0.19 |
| v2/old/4 | v0/old/4 | 2461 | 2 | 0.29 |
| v2/old/4 | v0/old/8 | 2461 | 2 | 0.29 |
| v2p/new/1 | v0/old/1 | 281 | 0 | 0.19 |
| v2p/new/1 | v0/old/4 | 281 | 0 | 0.19 |
| v2p/new/1 | v0/old/8 | 281 | 0 | 0.19 |
