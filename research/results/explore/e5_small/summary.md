# E5: small decision models (exploratory; descriptive)

exploratory (E5), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-06 'Exploratory row E5'; outside every Holm table.

Generated 2026-10-06T19:40:59+00:00 from `runs/e5_small`; host g5.8xlarge (NVIDIA A10G); GPU hours used 10.75.

## Accuracy (v1 dev = development; C = common states, descriptive)

| model | params (non-emb.) | setting | SFT min | v1 acc | ts | tp | stop | recov. | knockout | v1 ECE | C acc | C after off-plan | C ask-vs-act | C ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 1.72 (1.41) | nogc_3072 | 190.0 | 99.58 | 99.66 | 100.00 | 98.95 | 99.67 | 100.00 | 0.029 | 98.9 | 90.2 | 100.0 | 0.016 |
| Qwen3-0.6B | 0.6 (0.44) | nogc_3072 | 119.0 | 99.79 | 99.66 | 100.00 | 99.70 | 99.67 | 100.00 | 0.032 | 99.1 | 88.7 | 100.0 | 0.011 |
| MiniCPM5-1B | 1.08 (0.68) | nogc_3072 | 101.0 | 99.28 | 98.98 | 99.88 | 98.95 | 99.01 | 100.00 | 0.034 | 98.6 | 92.3 | 100.0 | 0.017 |
| SFT v1 (4B, the same recipe) | – (–) | – | – | 99.71 | 99.66 | 100.00 | 99.40 | 99.83 | 99.85 | 0.029 | 99.2 | 93.8 | 100.0 | 0.011 |
| JevAgentRL-4B (frozen) | – (–) | – | – | 99.71 | 99.66 | 100.00 | 99.40 | 99.83 | 99.54 | 0.004 | 99.2 | 94.7 | 100.0 | 0.010 |

Paired task bootstrap on C (B = 2,000, seed 0), model − reference, points; reading rule of plan §8 (keeps if the lower end ≥ −1.0; loses if the upper end < 0; else inconclusive). One seed per model.

| contrast | all states | after off-plan | ask-vs-act | v1 acc diff |
|---|---|---|---|---|
| q17_vs_sftv1 | -0.3 [-0.6, -0.1] (keeps the 4B's accuracy within 1 point (small loss: interval below 0)) | -3.6 [-6.7, -0.7] (loses accuracy) | +0.0 [+0.0, +0.0] (keeps the 4B's accuracy) | -0.13 |
| q17_vs_rl | -0.3 [-0.7, +0.0] (keeps the 4B's accuracy within 1 point (small loss: interval below 0)) | -4.5 [-7.5, -2.1] (loses accuracy) | +0.0 [+0.0, +0.0] (keeps the 4B's accuracy) | -0.13 |
| q06_vs_sftv1 | -0.1 [-0.3, +0.3] (keeps the 4B's accuracy) | -5.1 [-10.4, -0.3] (loses accuracy) | +0.0 [+0.0, +0.0] (keeps the 4B's accuracy) | 0.08 |
| q06_vs_rl | -0.1 [-0.4, +0.3] (keeps the 4B's accuracy) | -6.0 [-11.0, -1.5] (loses accuracy) | +0.0 [+0.0, +0.0] (keeps the 4B's accuracy) | 0.08 |
| mc1_vs_sftv1 | -0.6 [-0.9, -0.4] (keeps the 4B's accuracy within 1 point (small loss: interval below 0)) | -1.5 [-4.3, +0.9] (inconclusive) | +0.0 [+0.0, +0.0] (keeps the 4B's accuracy) | -0.42 |
| mc1_vs_rl | -0.6 [-0.9, -0.3] (keeps the 4B's accuracy within 1 point (small loss: interval below 0)) | -2.4 [-5.0, -0.5] (loses accuracy) | +0.0 [+0.0, +0.0] (keeps the 4B's accuracy) | -0.42 |

## Latency and memory (one session, one GPU; per request on the final harness's request stream)

Serve peak = nvidia-smi memory in use while serving (PyTorch's allocator cache included); train peak = the same during SFT. The 4B's thresholds stay frozen; its row "4b" in the fits below is the refit on its replayed answers (information only).

| served | c=1 p50 | c=1 p90 | c=4 p50 | c=4 p90 | S1/step c=1 p50 | S1/step c=4 p50 | synth. c=1 p50/p90 | synth. c=4 p50/p90 | speed-up c=1 / c=4 (p50) | serve peak MiB | train peak MiB | KV B/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4b | 317 | 439 | 876 | 1,298 | 375 | 1,725 | 847 / 849 | 3,322 / 3,326 | – / – | 21,795 | – | 147,456 |
| e5_q17 | 172 | 215 | 360 | 603 | 221 | 697 | 410 / 411 | 1,574 / 1,576 | 1.84× / 2.44× | 14,399 | 14,777 | 114,688 |
| e5_q06 | 127 | 159 | 245 | 448 | 171 | 465 | 284 / 285 | 1,073 / 1,076 | 2.50× / 3.57× | 11,973 | 8,725 | 114,688 |
| e5_mc1 | 107 | 127 | 174 | 319 | 141 | 327 | 191 / 193 | 693 / 695 | 2.95× / 5.03× | 4,495 | 9,709 | 24,576 |
| 4b_end | 317 | 440 | – | – | 375 | – | – / – | – / – | – / – | – | – | – |

## Live row

Model choice (registered rule, offline records only): {'run': True, 'slot': 'e5_mc1', 'eligible': {'e5_q17': 0.9889181441247148, 'e5_q06': 0.9913609586108261, 'e5_mc1': 0.9857599145760598}, 'within_1pt': ['e5_q17', 'e5_q06', 'e5_mc1'], 'c1_p50': {'e5_q17': 172.45588900004805, 'e5_q06': 126.89185199997155, 'e5_mc1': 107.30497300028219}, 'rule': 'highest C accuracy >= 95.0 %; within 1.0 point the lower c = 1 p50 (Deviations 2026-10-06 (g))'}.

Offline threshold fits (entry (g)): e5_q17: tool_select τ̂ 0.9999999999 (projection, n 1125), tool_params τ̂ 1 (projection, n 1328); e5_q06: tool_select τ̂ 0.9999816351 (projection, n 1125), tool_params τ̂ 0.9996356964 (projection, n 1328); e5_mc1: tool_select τ̂ 0.9999996544 (projection, n 1125), tool_params τ̂ 0.9999979734 (projection, n 1328); 4b: tool_select τ̂ 0.7878495933 (projection, n 1125), tool_params τ̂ 0.9999998808 (projection, n 1328)

Coverage on the calibration states at the fitted thresholds (frozen 4B: frozen thresholds): share of decisions the cascade takes, System-1 accuracy on them, the LLM's on the same, the cascade's overall accuracy vs the LLM alone.

| System-1 | decision point | τ | paired | coverage | S1 acc (taken) | LLM acc (taken) | cascade acc | LLM acc |
|---|---|---|---|---|---|---|---|---|
| e5_q17 | tool_select | 0.9999999999 | 1125 | 90.0 | 100.00 | 100.00 | 100.00 | 100.00 |
| e5_q17 | tool_params | 1 | 1328 | 78.5 | 100.00 | 100.00 | 100.00 | 100.00 |
| e5_q06 | tool_select | 0.9999816351 | 1125 | 90.0 | 100.00 | 100.00 | 100.00 | 100.00 |
| e5_q06 | tool_params | 0.9996356964 | 1328 | 86.8 | 100.00 | 100.00 | 100.00 | 100.00 |
| e5_mc1 | tool_select | 0.9999996544 | 1125 | 50.0 | 100.00 | 100.00 | 100.00 | 100.00 |
| e5_mc1 | tool_params | 0.9999979734 | 1328 | 70.0 | 100.00 | 100.00 | 100.00 | 100.00 |
| 4b_logged | tool_select | 0.5 | 1125 | 100.0 | 100.00 | 100.00 | 100.00 | 100.00 |
| 4b_logged | tool_params | 0.9999998808 | 1328 | 77.5 | 99.90 | 100.00 | 99.92 | 100.00 |
| 4b | tool_select | 0.5 | 1125 | 100.0 | 100.00 | 100.00 | 100.00 | 100.00 |
| 4b | tool_params | 0.9999998808 | 1328 | 77.4 | 100.00 | 100.00 | 100.00 | 100.00 |

4B replay check: tool_select agreement 0.9991, tool_params agreement 0.9987 (ok True).

### Live row

System-2: {'provider': 'openrouter', 'model': 'google/gemini-3.5-flash', 'temperature': 0, 'reasoning_effort': 'minimal'}. Dollar figures: within-E5 only: System-2 through OpenRouter, which reports about half the direct API's input tokens (Deviations 2026-10-06 ~18:00 UTC); not comparable with harness_final's direct-API rows.

n = 90 tasks complete in every arm; host $2.448/h; q per episode llm_only $0.0000, harness_final $0.0115, e5_final $0.0096.

| arm | success | API $/ep | $/ep incl. q | wall s/ep | S1 s/ep | S2 s/ep | tool cov. | arg cov. |
|---|---|---|---|---|---|---|---|---|
| llm_only | 100.0 | 0.0181 | 0.0181 | 8.46 | 0.00 | 8.44 | 0.0 | – |
| harness_final | 100.0 | 0.0026 | 0.0141 | 16.96 | 3.90 | 13.03 | 100.0 | 86.0 |
| e5_final | 100.0 | 0.0127 | 0.0223 | 14.12 | 1.09 | 13.00 | 39.0 | 80.4 |

| contrast | success Δ pt [95 %] | cost saving (incl. q) | wall Δ s [95 %] |
|---|---|---|---|
| e5_final_vs_llm_only | +0.0 [+0.0, +0.0] | -23.3 % | +5.66 [+4.15, +7.15] |
| harness_final_vs_llm_only | +0.0 [+0.0, +0.0] | 21.9 % | +8.50 [+6.76, +10.31] |
| e5_final_vs_harness_final | +0.0 [+0.0, +0.0] | -57.8 % | -2.84 [-4.84, -0.94] |

LLM (System-2) call latency, seconds: per decision point and per served provider.

| arm | group | calls | p50 | p90 | mean | calls > 5 s | their share of LLM time |
|---|---|---|---|---|---|---|---|
| llm_only | all | 675 | 0.98 | 1.37 | 1.13 | 5 | 6.6 |
| llm_only | dp tool_select | 675 | 0.98 | 1.37 | 1.13 | 5 | 6.6 |
| llm_only | served by Google | 673 | 0.98 | 1.37 | 1.08 | 3 | 2.5 |
| llm_only | served by Google AI Studio | 2 | 15.86 | 16.21 | 15.86 | 2 | 100.0 |
| harness_final | all | 269 | 1.36 | 11.54 | 4.36 | 93 | 78.5 |
| harness_final | dp tool_params | 269 | 1.36 | 11.54 | 4.36 | 93 | 78.5 |
| harness_final | served by Google | 219 | 1.18 | 7.92 | 2.69 | 43 | 57.2 |
| harness_final | served by Google AI Studio | 50 | 11.56 | 12.22 | 11.66 | 50 | 100.0 |
| e5_final | all | 565 | 1.01 | 5.82 | 2.07 | 62 | 50.7 |
| e5_final | dp tool_params | 152 | 1.72 | 11.76 | 4.64 | 59 | 80.3 |
| e5_final | dp tool_select | 413 | 0.97 | 1.33 | 1.12 | 3 | 5.7 |
| e5_final | served by Google | 538 | 1.00 | 2.58 | 1.58 | 35 | 32.1 |
| e5_final | served by Google AI Studio | 27 | 11.92 | 12.20 | 11.87 | 27 | 100.0 |

