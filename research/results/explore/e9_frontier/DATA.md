# E9: data inventory for the cost-accuracy frontier of multiple-choice decisions

Exploratory row E9 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E9: cost-accuracy frontier"), descriptive.
Written 2026-10-07 before any E9 model call. It lists which logged item sets hold answers of both decision models and
LLM deciders to the same multiple-choice questions, and which models have which numbers. Nothing here is a result;
the numbers are in `summary.{md,json}` (`scripts/explore_e9_frontier.py`).

## Item sets considered

| item set | items | where | decision models with answers | LLM deciders on the same questions | usable for the frontier? |
|---|---|---|---|---|---|
| **L1 static (E1, the C8 population)**: BFCL v4 tool choice, BFCL arguments, When2Call (call / ask / answer / refuse) | 3,991 decisions (1,114 + 1,949 + 928), dev prefix excluded | `runs/e1_bfcl_when2call*/` (per-item records: answer, gold, correct, confidence, tokens, cost, latency) | Jev, Intern-Decision-4B, StartLux-Decision 0.8B / 2B / 4B / 9B / 27B, AgentDecision-4B (`jevagent_rl`), its untrained base and SFT-only checkpoint | Gemini 3.5 Flash and Flash-Lite as multiple-choice deciders (verbalised distribution; Flash-Lite also verbalised label), Gemma 4 26B (unpriced), Flash's own native function call; controls (embedding, BM25, TF-IDF) | **yes: the one common set** |
| E1 development prefix | 299 decisions (85 + 142 + 72) | `configs/eval/e1_dev_prefix_ids.json` (A6.13: exposed in development rounds, excluded from C8); `runs/dev/e1_dev/` | none logged (Jev was not reachable in the development rounds; no GPU arm) | Flash, Flash-Lite (both modes), Gemma 4 (first 120), Flash native FC | only with new calls (the E9 check: Jev, GPT-6-Luna, GPT-6.1-Sol) |
| S2-3 common-state corpus C (Stage 2) | 5,314 tool-choice occurrences, 200 tasks | `results/stage2/s23/` (seeds 300000-300199, an evaluation block) | Jev, Intern-4B, StartLux 0.8B-27B, AgentDecision-4B, base, SFT v1, v2; E5 students (accuracy only, `results/explore/e5_small`) | none (the agent LLM's own choice exists only on its own trajectories) | no: evaluation states, never re-scored by a new model |
| JevGym calibration shadow logs (seeds 5000-5149; Sol and DeepSeek logs 5150-5299) | 2,719 tool decisions per log (1,125 choice + 1,594 arguments; 2,449-2,453 with an answer key) | `runs/e2a_jevgym_shadow_cal*/`, E5 replays `runs/e5_small/cal/` | each log one model: Jev, Intern-4B, StartLux-4B, AgentDecision-4B (+ the E5 students replayed on its states), v2 | the agent LLM's own native choice (Flash, Flash-Lite, GPT-6.1-Sol, DeepSeek-V4-Pro), each on its own log | no (see below) |
| E8 sample | 1,000 decisions of the Flash calibration log | `results/explore/e8_gates/` | Jev | Flash as a yes / no verifier of Jev's answer (not a multiple-choice decider) | no |
| JevGym-Hard calibration (E7c / E7d, seeds 5300-5399) | 3,570-3,699 tool choices per log | `runs/e7c_cal/`, `runs/e7d_cal_*` | Jev and the frozen 4B on separate trajectories; the tier checkpoint replayed on 20 tasks | the agent's own choice | no |

Why the calibration logs are not the frontier's item set:

- **Ceiling.** Against the registered answer key (the acceptable tool set AT(s), `threshold_theory.decision_pairs`),
  the agent LLM's own choice is right on 98.9-100.0 % of the labelled tool choices and arguments in every
  calibration log (Flash, Flash-Lite, GPT-6.1-Sol, DeepSeek-V4-Pro; JevGym-Hard included). AgentDecision-4B and v2
  score 99.0-100.0 %, Jev 87.5-100.0 %, StartLux-4B 91 %, Intern-4B 77-90 % (each on its own log). The trained
  models were tuned on JevGym development tasks, and the LLMs sit at the ceiling, so these states would show little
  about the cost-accuracy trade-off between model kinds.
- **Different states.** The shadow logs are separate runs whose Flash trajectories diverge. Any two logs share
  1,115-1,211 of their 2,719 tool decisions; the states common to the Jev, Intern, StartLux-4B, AgentDecision-4B and
  v2 logs are 632 labelled decisions, 514 of them at steps 0-1. Only AgentDecision-4B and the three E5 students
  answer the same full set of states (`rl_shadow_cal`).
- The open models cannot be re-run here (no GPU), so a new sample of calibration states would carry Jev and at most
  AgentDecision-4B and the E5 students.

## Which model has which numbers on the frontier's item set (E1, 3,991 decisions)

| model | role | record file | accuracy | cost per decision | latency |
|---|---|---|---|---|---|
| Jev 1.13 | decision model (API) | `e1_bfcl_when2call/s1_jev.jsonl` | yes | logged API price ($0.042 / M input, output free) | logged |
| Intern-Decision-4B | decision model (open weights, in process) | `e1_bfcl_when2call_startlux/s1_intern_4b.jsonl` | yes | GPU tariff × logged serving time | logged |
| StartLux-Decision 4B, 27B (Q8_0) | decision model (open weights) | `e1_bfcl_when2call_startlux/s1_startlux_{4b,27b}.jsonl` | yes | GPU tariff × logged serving time | logged |
| StartLux-Decision 0.8B, 2B, 9B | decision model (open weights) | `e1_bfcl_when2call_startlux_ladder/s1_startlux_*.jsonl` | yes | GPU tariff × logged serving time | logged |
| AgentDecision-4B (`jevagent_rl`, frozen) | decision model (open weights, ours) | `e1_bfcl_when2call_rl/s1_jevagent_rl.jsonl` | yes | GPU tariff × logged serving time | logged |
| Qwen3-4B base, SFT v1 | attribution rows | `e1_bfcl_when2call_rl/s1_rl_{base,sft_v1}.jsonl` | yes | as above | logged |
| Gemini 3.5 Flash, multiple-choice (`s2_self`, verbalised distribution) | LLM decider | `e1_bfcl_when2call/s1_gemini_flash_typed.jsonl` | yes | logged list price (reasoning tokens included) | logged |
| Gemini 3.5 Flash-Lite, multiple-choice (distribution; label) | LLM decider | `s1_gemini_lite_{dist,label}.jsonl` | yes | logged list price | logged |
| Gemma 4 26B-A4B, multiple-choice | LLM decider | `s1_gemma4_dist.jsonl` | yes | **unpriced** (no verified price in `configs/prices/models.yaml`): listed, not plotted | logged |
| Gemini 3.5 Flash, native function call | the agent LLM's own call (not multiple choice) | `fc_gemini_flash_fc.jsonl` | yes | logged list price (When2Call judge excluded) | logged |
| Gemini embedding, BM25, TF-IDF | controls | `s1_{gemini_embed,bm25,tfidf}.jsonl` | yes | logged | logged |

Not on this item set: the E5 students (scored on C and on the calibration states only), JevAgentRL-v2, GPT-6-Luna,
GPT-6.1-Sol, GPT-6-Astra and DeepSeek-V4 as multiple-choice deciders (no record anywhere). DeepSeek cannot be added
now: it is reached only through OpenRouter, whose organisation budget is exhausted, and the author's Jev-only key is
never used for another model.

## Cost per decision (one rule per kind of model)

- API models (Jev, Gemini): the per-item `cost_usd` of the records (the ledger's tariff on measured tokens, reasoning
  tokens billed as output; a batched request's cost split over its questions). The records sum to each ledger total.
- Open-weights models: the paper's GPU tariff, $2.24208 per GPU-hour (the Stage-2 host, one L40S, A7.13 (iii),
  App. app:cost), times the logged serving time per decision (the request's latency divided by the questions it
  carried). Requests were sent four at a time, so this charges queueing too (an upper bound on GPU time).
  Sensitivity rows: the provisioned per-token tariff of the registered windows (`results/stage2/tariff.json`,
  `tariff_sl4b.json`: $0.094 / M and $0.043 / M System-1 input tokens), Intern-Decision's registered per-token
  price ($0.010 / M, `configs/prices/models.yaml`), and zero.

## The E9 check (new calls, registered before any call)

The cheapest LLM tier has no record on E1. The check scores GPT-6-Luna and GPT-6.1-Sol (OpenAI) next to Jev and the
two Gemini deciders on the 299 development-prefix items (`configs/explore/e9_frontier/e1_dev_check.yaml`). It is
reported in the text beside the figure, never drawn into it (different items).
