# E12: AgentDecision-4B (mixed) (more training on public tool-use data)

_exploratory (E12), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-07 'E12: more training on public tool-use data'; outside every Holm table_

## Pre-registered outcomes

(i) **Fixed questions (E1, 3,991 decisions):** AgentDecision-4B (mixed) 85.1 % vs AgentDecision-4B 78.8 % (+6.3 points, paired item bootstrap 95 % [+5.1, +7.5]).
(ii) **Common states C (5,314):** 99.1 % vs 99.2 % (-0.1 points, paired task bootstrap [-0.3, +0.1]); non-inferior at the 1-point margin: yes.
(iii) **On E9's frontier:** no (frontier changed by the new point: no; with the same-host cost adjustment: no).
(iv) **Live success (replace mode, Flash, seeds 300000-300199):** 100.0 % (n 200) vs AgentDecision-4B 100.0 % (+0.0 points, Newcombe [-1.9, +1.9]).

## E1 by family

| | AgentDecision-4B (mixed) | AgentDecision-4B | delta [95 %] |
|---|---|---|---|
| BFCL tool choice | 89.3 | 88.4 | +0.9 [-0.9, +2.7] |
| BFCL arguments | 90.5 | 85.0 | +5.5 [+3.9, +7.0] |
| When2Call | 68.5 | 54.0 | +14.5 [+11.2, +17.9] |
| all | 85.1 | 78.8 | +6.3 |
| $ per 1,000 decisions (E9's rule) | 0.1397 | 0.0934 | |

When2Call by gold label (E12): asktheuserformissingrequiredinformation 74.8 % (n 266), calloneoftheavailabletools 59.6 % (n 329), sayitcannothelp:noavailabletoolfitsandthequestionneedsone 72.4 % (n 333).
Serving-path check: the frozen checkpoint re-served on the E12 host agrees with its logged E1 answers on 99.8 % of 3991 decisions (rule >= 99 %: ok); its same-host cost is $0.1428 per 1,000 decisions (logged on the Stage-2 host: $0.0934).

Frontier without the new point: sl0.8b -> jev -> flash; with it: sl0.8b -> jev -> flash.

## Offline (E12 host)

| file | AgentDecision-4B (mixed) | AgentDecision-4B |
|---|---|---|
| pubdev/dev_args | 96.0 | 90.8 |
| pubdev/dev_tool | 97.7 | 93.0 |
| pubdev/dev_w2c | 96.7 | 67.1 |
| v1/dev | 99.7 | 99.7 |
| v1/dev_knockout | 99.8 | 99.7 |
| v1/dev_recovery | 99.8 | 99.8 |

## Live (tool_select replace, Flash, the first 200 fresh Stage-2 tasks)

| row | success % | n | own-state acc % | steps | API $ / ep. |
|---|---|---|---|---|---|
| AgentDecision-4B (mixed) | 100.0 | 200 | 100.0 | 7.5 | 0.0145 |
| AgentDecision-4B (A7.10 row) | 100.0 | 200 | 100.0 | 7.5 | 0.0145 |
| Jev (A7.10 row) | 83.0 | 200 | 91.0 | 7.0 | 0.0144 |

## Data and leakage audit

New items 7,000 (by family {"bfcl_tool": 1500, "w2c_call": 1000, "bfcl_args": 2000, "bfcl_none": 500, "w2c_cannot": 1000, "w2c_ask": 1000}), replay 7,000 SFT v1 rows, public dev 1,000. Units: {"total": 19909, "toolace": 5033, "when2call_train": 14876, "duplicate_rows_dropped": {"when2call-train": 124, "toolace": 3}, "clean": 19702, "train": 17830, "dev": 1872}.
Audit (exact token sequence or a shared 8-gram of the user request (shorter requests: identical tokens); a unit with any hit is dropped): 207 units (485 items) dropped; by target {"e1": {"exact": 0, "ngram": 175}, "e1_dev": {"exact": 0, "ngram": 14}, "public_test_pools": {"exact": 2, "ngram": 205}}; by source {"when2call-train": {"e1:ngram": 57, "public_test_pools:ngram": 67, "units_dropped": 69, "e1_dev:ngram": 9, "public_test_pools:exact": 2}, "toolace": {"e1:ngram": 118, "public_test_pools:ngram": 138, "units_dropped": 138, "e1_dev:ngram": 5}}.

## E12b: reading the order rule (held-out `tier` states, zero-shot for E12b)

| stratum | AgentDecision-4B (random rules, E12b) | AgentDecision-4B (this host) | AgentDecision-4B (E7d) | AgentDecision-4B (tier, E7d: trained on tier) |
|---|---|---|---|---|
| cancel/ship tool choice | 91.0 | 76.5 | 76.5 | 95.3 |
| other tool choices | 100.0 | 91.6 | 91.6 | 100.0 |
| arguments | 98.7 | 95.1 | 95.1 | 98.2 |
E12b data: {"rows": {"train_mix": 3000, "b_dev": 5189, "cal_dev_small": 2000}, "strata": {"op": 900, "ts_other": 450, "tp": 450}, "tasks": {"train": 540, "dev": 60}, "rule_kinds": {"above": 204, "below": 184, "tiered": 212}, "train_seeds": [322000, 600]}.

## E12c: the recipe on StartLux-Decision-4B (CC BY-NC 4.0)

Base and memory setting: {"base": "startlux", "path": "/opt/startlux/models/StartLux-Decision-4B", "setting": "gc_3072", "s_per_step": 47.75, "projected_min": 341}.
Checkpoint selection (E1 dev prefix + v1 dev only): {"epoch": 1, "path": "/opt/jev/ck_e12c/train/checkpoint-214", "acc": {"e1_devprefix": 0.89597, "dev": 0.99537}, "mean": 0.94567}.

| outcome | value |
|---|---|
| (i) E1: StartLux-4B (mixed) vs StartLux-4B | 85.5 vs 83.1 % (+2.3 [+1.1, +3.6]); beats: yes |
| (ii) E1: gap to Jev | -0.9 [-2.1, +0.2] (Jev 86.4 %) |
| (iii) C vs AgentDecision-4B | 99.6 vs 99.2 % (+0.4 [+0.0, +1.0]); non-inferior at 1 point: yes |
| (iv) live success | 100.0 % (n 200) vs AgentDecision-4B 100.0 %, Jev 83.0 %, StartLux-4B 60.0 % |
| E1 $ per 1,000 (E9's rule, own stack) | 0.2946 (Jev 0.0163, StartLux-4B 0.0524) |
| E1 BFCL tool choice | 89.5 (StartLux-4B 91.3, Jev 89.9) |
| E1 BFCL arguments | 91.9 (StartLux-4B 81.2, Jev 89.6) |
| E1 When2Call | 67.1 (StartLux-4B 77.4, Jev 75.4) |

## Same-engine serving cost (vllm 0.31.0 torch 2.13.0+cu130, merged LoRA, E1's 3,991 decisions, 4 states in flight)

| model | GPU ms / decision (wall) | $ / 1,000 (wall) | $ / 1,000 (E9's rule, this engine) | $ / 1,000 (E9's rule, own stack) | agreement with served answers |
|---|---|---|---|---|---|
| AgentDecision-4B | 23.7 | 0.0147 | 0.0589 | 0.0934 | 99.8 |
| AgentDecision-4B (mixed) | 23.8 | 0.0148 | 0.0592 | 0.1397 | 99.6 |
| StartLux-4B (mixed) | 39.5 | 0.0246 | 0.0983 | 0.2946 | 99.8 |
| StartLux-4B | 39.8 | 0.0248 | 0.0990 | 0.0524 | 96.3 |
StartLux-4B is timed on serve_jev's prompt (not its vendor's), so its agreement is not an accuracy.

Spend: $2.93 of API (Gemini, live smoke + live), cap $30.
GPU: 8.4 host hours; training 146 min.

Generated 2026-10-07T22:28:54+00:00.
