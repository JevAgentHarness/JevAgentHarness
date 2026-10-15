# E16: the best open-weights decision model on all three levels

_exploratory (E16), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-08 'E16: the best open-weights decision model on all three levels'; outside every Holm table_

L1 = E1's 3,991 fixed decisions (accuracy; $ per 1,000 by E9's rule); L2 = the common states C (5,314 occurrences, task means); L3 = REPLACE at the tool choice, Flash, the first 200 fresh Stage-2 tasks (success).

## The best open-weights decision model we did not train, per level (registration (d) 4)

- L1: StartLux-27B (Q8_0) 87.7 % (then clef, sl9b, sl4b)
- L2: StartLux-27B (Q8_0) 93.9 % (then openjevhf, sl9b, openjev)
- L3: StartLux-9B 74.0 % (then sl27b, openjevhf, openjev)

## The arms

| arm | L1 all | tool choice / arguments / When2Call | L2 (C) | L3 (live) | best open L1 / L2 / L3 (gap) |
|---|---|---|---|---|---|
| AgentDecision-4B (mixed+) | 84.3 | 88.8 / 90.1 / 66.9 | 99.1 | 100.0 (n 200) | no (-3.4) / yes (+5.2) / yes (+26.0) |
| StartLux-9B (mixed) | 84.8 | 90.0 / 91.9 / 63.6 | 99.3 | 100.0 (n 200) | no (-2.9) / yes (+5.4) / yes (+26.0) |

Reference rows: AgentDecision-4B L1 78.8 / L2 99.2 / L3 100.0; E12 85.1 / 99.1 / 100.0; E12c 85.5 / 99.6 / 100.0.

## Arm A: AgentDecision-4B (mixed+)

Selected epoch 2 (dev prefix 87.2 %, v1 dev 99.7 %); candidates: epoch 1: 87.2 / 99.7; epoch 2: 87.2 / 99.7. Training 145 min (nogc_3072).
L1: 84.3 % [83.2, 85.4]; $0.0909 per 1,000 (E9's rule, this host), median 260 ms. When2Call by gold label: call 69.3 (n 329), sayi 64.6 (n 333), askt 66.9 (n 266). Confusion (gold->pred): {'askt->askt': 178, 'askt->call': 80, 'askt->sayi': 8, 'call->askt': 85, 'call->call': 228, 'call->sayi': 16, 'sayi->askt': 49, 'sayi->call': 69, 'sayi->sayi': 215}.
- vs frozen (rl, 78.8 %): +5.6 [+4.4, +6.8]; tool choice +0.4, arguments +5.1, When2Call +12.9
- vs jev (jev, 86.4 %): -2.1 [-3.2, -1.0]; tool choice -1.2, arguments +0.5, When2Call -8.5
- vs sl4b (sl4b, 83.1 %): +1.2 [-0.1, +2.4]; tool choice -2.5, arguments +8.9, When2Call -10.5
- vs sl9b (sl9b, 83.4 %): +0.9 [-0.4, +2.1]; tool choice -3.4, arguments +9.1, When2Call -11.2
- vs sl27b (sl27b, 87.7 %): -3.4 [-4.5, -2.2]; tool choice -2.0, arguments +1.0, When2Call -14.3
- vs e12 (None, 85.1 %): -0.7 [-1.6, +0.1]; tool choice -0.5, arguments -0.4, When2Call -1.6
- vs bestopen (sl27b, 87.7 %): -3.4 [-4.5, -2.2]; tool choice -2.0, arguments +1.0, When2Call -14.3
L2: 99.1 % (after off-plan 93.5, ask-vs-act 100.0); vs AgentDecision-4B -0.1 [-0.3, +0.2], non-inferior at 1 point: yes; vs the best open model +5.2 [+4.2, +6.2]; live gate: yes; missing 0, errors 0.
L3: 100.0 % (n 200; own-state accuracy 99.9 %, 7.5 steps, $0.0145 per episode, 11 s); vs frozen +0.0 [-1.9, +1.9]; vs jev +17.0 [+12.1, +22.8]; vs bestopen +26.0 [+20.1, +32.5]

## Arm B: StartLux-9B (mixed)

Selected epoch 2 (dev prefix 87.9 %, v1 dev 99.8 %); candidates: epoch 1: 86.9 / 99.7; epoch 2: 87.9 / 99.8. Training 303 min (gc_3072).
Plan: {"b": "run", "setting": "gc_3072", "s_per_step": 28.425, "projected_min_per_epoch": 205, "cap_min": 420, "minutes_left_at_smoke": 626, "epochs": 2, "reason": null}
L1: 84.8 % [83.7, 85.9]; $0.2404 per 1,000 (E9's rule, this host), median 604 ms. When2Call by gold label: call 93.6 (n 329), sayi 73.3 (n 333), askt 14.3 (n 266). Confusion (gold->pred): {'askt->askt': 38, 'askt->call': 216, 'askt->sayi': 12, 'call->askt': 4, 'call->call': 308, 'call->sayi': 17, 'sayi->askt': 1, 'sayi->call': 88, 'sayi->sayi': 244}.
- vs frozen (rl, 78.8 %): +6.1 [+4.6, +7.5]; tool choice +1.6, arguments +6.9, When2Call +9.6
- vs jev (jev, 86.4 %): -1.6 [-2.8, -0.5]; tool choice +0.1, arguments +2.4, When2Call -11.9
- vs sl4b (sl4b, 83.1 %): +1.7 [+0.5, +3.0]; tool choice -1.3, arguments +10.7, When2Call -13.8
- vs sl9b (sl9b, 83.4 %): +1.4 [+0.2, +2.6]; tool choice -2.2, arguments +11.0, When2Call -14.5
- vs sl27b (sl27b, 87.7 %): -2.9 [-4.0, -1.7]; tool choice -0.7, arguments +2.9, When2Call -17.7
- vs e12c (None, 85.5 %): -0.7 [-1.7, +0.4]; tool choice +0.5, arguments +0.1, When2Call -3.6
- vs bestopen (sl27b, 87.7 %): -2.9 [-4.0, -1.7]; tool choice -0.7, arguments +2.9, When2Call -17.7
L2: 99.3 % (after off-plan 95.0, ask-vs-act 100.0); vs AgentDecision-4B +0.1 [-0.1, +0.4], non-inferior at 1 point: yes; vs the best open model +5.4 [+4.4, +6.3]; live gate: yes; missing 0, errors 0.
L3: 100.0 % (n 200; own-state accuracy 100.0 %, 7.5 steps, $0.0145 per episode, 22 s); vs frozen +0.0 [-1.9, +1.9]; vs jev +17.0 [+12.1, +22.8]; vs bestopen +26.0 [+20.1, +32.5]

## Same-engine timing (vLLM, merged LoRA; $ per 1,000 decisions, wall clock, four states in flight)

- e16a: 0.0047 (E9's rule on this engine 0.0188; 7.5 ms per decision; p50 56 ms at 4, 20 ms at 1; agreement with the served answers 99.6 %; 8.0 GB)
- e16b: 0.0140 (E9's rule on this engine 0.0560; 22.5 ms per decision; p50 171 ms at 4, 50 ms at 1; agreement with the served answers 99.9 %; 17.9 GB)
- frozen: 0.0047 (E9's rule on this engine 0.0188; 7.5 ms per decision; p50 56 ms at 4, 20 ms at 1; agreement with the served answers 99.8 %; 8.0 GB)

Serving-path check: the frozen checkpoint re-served on the E16 host agrees with its logged E1 answers on 99.8 % (rule >= 99 %: ok); same-host cost $0.0925 per 1,000 (host factor 1.01).

Spend: $5.85 of Gemini (cap $8). Host: g6e.2xlarge, 10.2 host-hours.

Generated 2026-10-08T14:44:42+00:00.
