# E21b: the deployed cascades on DecisionGym-Shift (exploratory (E21b), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-08 'E21b: the deployed confidence-gated cascades on DecisionGym-Shift'; outside every Holm table)

Generated 2026-10-09T02:02:18+00:00 at git 028277744147 by `python scripts/explore_e21b_shift_cascade.py summarize`. Reference: E21's logged LLM-only row (runs/e21_live_local/llm_only.jsonl; n = 200, success 100.0 %, $0.0535 and 8.0 s per episode; 4 workers from the container). Arms: the final harness at the in-domain thresholds, 1 worker, the model alone on the host's GPU. Cost includes the GPU tariff $1.624/h × episode wall-clock.

| arm | n | err | success | Δ vs LLM-only [Newcombe] | W/L | NI p (3 pt) | $/ep. incl. GPU | saving | wall s (mean / median) | tool cov. | arg cov. | in-domain tool / arg cov. | accepted wrong tool / arg | must-ask taken / wrong | REPLACE (E21) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Frozen 4B (final harness) | 200 | 0 | 94.0 | -6.0 [-10.2, -2.8] | 0 / 12 | 0.993 (not shown) | 0.0189 | 64.8 % | 9.5 / 7.0 | 99.6 | 58.9 | 100.0 / 85.5 | 62 of 1364 / 17 of 959 | 113 of 113 / 7 | 96.0 |
| 0.8B head + defer head (E18) | 200 | 0 | 64.5 | -35.5 [-42.3, -28.9] | 0 / 71 | 1.000 (not shown) | 0.0403 | 24.8 % | 7.0 / 5.0 | 67.2 | 56.6 | 100.0 / 98.7 | 383 of 984 / 61 of 446 | 139 of 146 / 33 | 17.5 |
| AgentDecision-4B (E12) | 200 | 0 | 92.5 | -7.5 [-12.0, -4.0] | 0 / 15 | 1.000 (not shown) | 0.0162 | 69.7 % | 8.4 / 6.0 | 100.0 | 62.2 | 100.0 / 92.3 | 22 of 1272 / 0 of 1039 | 82 of 82 / 2 | 91.5 |
| AgentDecision-4B (E12), in-domain | 200 | 0 | 100.0 | +0.5 [-1.4, +2.8] | 1 / 0 | 0.002 (non-inferior) | 0.0059 | 86.3 % | 5.4 / 6.2 | 100.0 | 92.3 | – / – | 0 of 1500 / 0 of 1962 | 50 of 50 / 0 | – |

## Per arm

- **Frozen 4B (final harness)**: success 94.0 vs 100.0 (paired bootstrap -6.0 [-9.5, -3.0]); NI not shown (p = 0.993, Tango [-10.2, -3.5]); API $0.0146 + GPU $0.0043 vs $0.0535; saving 64.8 % [60.9, 67.8] (API only 72.8 %; host tariff 66.8 %); wall 9.5 s vs 8.0 s; steps 6.9, LLM calls 5.0, System-1 requests 12.7; handed back 0.4 % of the tool choices; deferred tool choices right by the LLM 100.0 %.
  by kind: clarify.ask_after_lookup 75 (20), clarify.ask_twice 100 (20), clarify.method 90 (20), clarify.name 100 (20), clarify.order 100 (20), clarify.speed 75 (20), multi 100 (60), notify 100 (20)
  vs E21's REPLACE row (rl_ts_replace_f, tracked True): 96.0 → cascade -2.0 [-5.8, +1.5], W/L 3 / 7
  in-domain (runs/harness_final_dedicated/harness_final.jsonl): success 100.0, tool / arg coverage 100.0 / 85.5
- **0.8B head + defer head (E18)**: success 64.5 vs 100.0 (paired bootstrap -35.5 [-42.0, -29.0]); NI not shown (p = 1.000, Tango [-42.3, -29.2]); API $0.0371 + GPU $0.0032 vs $0.0535; saving 24.8 % [17.1, 32.1] (API only 30.7 %; host tariff 26.3 %); wall 7.0 s vs 8.0 s; steps 7.3, LLM calls 6.1, System-1 requests 11.6; handed back 32.8 % of the tool choices; deferred tool choices right by the LLM 97.9 %.
  by kind: clarify.ask_after_lookup 100 (20), clarify.ask_twice 35 (20), clarify.method 100 (20), clarify.name 85 (20), clarify.order 100 (20), clarify.speed 0 (20), multi 52 (60), notify 70 (20)
  vs E21's REPLACE row (e14a_ts_replace_f, tracked True): 17.5 → cascade +47.0 [+39.6, +53.5], W/L 94 / 0
  in-domain (runs/e18_live/e18_final.jsonl): success 100.0, tool / arg coverage 100.0 / 98.7
- **AgentDecision-4B (E12)**: success 92.5 vs 100.0 (paired bootstrap -7.5 [-11.5, -4.0]); NI not shown (p = 1.000, Tango [-12.0, -4.6]); API $0.0124 + GPU $0.0038 vs $0.0535; saving 69.7 % [67.9, 71.6] (API only 76.8 %; host tariff 71.5 %); wall 8.4 s vs 8.0 s; steps 6.4, LLM calls 4.3, System-1 requests 11.8; handed back 0.0 % of the tool choices; deferred tool choices right by the LLM – %.
  by kind: clarify.ask_after_lookup 35 (20), clarify.ask_twice 95 (20), clarify.method 100 (20), clarify.name 100 (20), clarify.order 100 (20), clarify.speed 95 (20), multi 100 (60), notify 100 (20)
  vs E21's REPLACE row (e12_ts_replace_f, tracked True): 91.5 → cascade +1.0 [-2.2, +4.4], W/L 5 / 3
  in-domain (runs/e21b_e12_in/e12_final_in.jsonl): success 100.0, tool / arg coverage 100.0 / 92.3
- **AgentDecision-4B (E12), in-domain**: success 100.0 vs 99.5 (paired bootstrap +0.5 [+0.0, +1.5]); NI non-inferior (p = 0.002, Tango [-1.4, +2.8]); API $0.0035 + GPU $0.0024 vs $0.0431; saving 86.3 % [85.8, 86.7] (API only 91.9 %; host tariff 87.7 %); wall 5.4 s vs 13.6 s; steps 7.5, LLM calls 1.8, System-1 requests 14.0; handed back 0.0 % of the tool choices; deferred tool choices right by the LLM – %.

## Registered reading (E21b (f))

- Frozen 4B (final harness): protects success (NI at 3 points) False; still saves True; defers almost everything False
- 0.8B head + defer head (E18): protects success (NI at 3 points) False; still saves True; defers almost everything False
- AgentDecision-4B (E12): protects success (NI at 3 points) False; still saves True; defers almost everything False

## E12's in-domain fit

```
{
 "fits": {
  "tool_select": {
   "tau_hat": 0.44503032941063925,
   "n_paired": 1125,
   "tau_rule": "projection",
   "path": "results/explore/e21b_shift_cascade/cal/tauhat_e21bcal_e12_tool_select.json"
  },
  "tool_params": {
   "tau_hat": 0.9958731532096863,
   "n_paired": 1328,
   "tau_rule": "projection",
   "path": "results/explore/e21b_shift_cascade/cal/tauhat_e21bcal_e12_tool_params.json"
  }
 },
 "coverage_at_tau": {
  "tool_select": {
   "tau": 0.44503032941063925,
   "always_defer": false,
   "n_paired": 1125,
   "accepted": 1125,
   "coverage": 1.0,
   "accepted_errors": 0,
   "s1_acc_all": 1.0,
   "s2_acc_all": 1.0
  },
  "tool_params": {
   "tau": 0.9958731532096863,
   "always_defer": false,
   "n_paired": 1328,
   "accepted": 1112,
   "coverage": 0.8373493975903614,
   "accepted_errors": 0,
   "s1_acc_all": 0.9962349397590361,
   "s2_acc_all": 1.0
  }
 },
 "profile": {
  "path": "configs/harness/explore_e21b_e12_final.yaml",
  "sha256": "c460546766f2de8e40b1ef71015027dd2ca3799745c41bb9043c0b9696d6a21a",
  "backend": {
   "tool_select": {
    "tau": 0.44503032941063925
   },
   "tool_params": {
    "tau": 0.9958731532096863
   },
   "fitted_from": {
    "tool_select": {
     "path": "results/explore/e21b_shift_cascade/cal/tauhat_e21bcal_e12_tool_select.json",
     "sha256": "ed598114470acc889f751e33a681b5ab4550aa5905776bb05dbe8dd76a20457e"
    },
    "tool_params": {
     "path": "results/explore/e21b_shift_cascade/cal/tauhat_e21bcal_e12_tool_params.json",
     "sha256": "b1a65eddb359c829368b8558de7159c3ccf68d3ad366815ed3fe0bfd02433339"
    }
   },
   "notes": "E21b (b) 3: E12's in-domain fit by the registered command on rl_shadow_cal (seeds 5000-5149)"
  },
  "rule": "E21b (b) 3: E12's in-domain fit by the registered command on rl_shadow_cal (seeds 5000-5149)"
 }
}
```

Spend: Gemini $13.70 (probes $0.0022) in 808 records; OpenRouter, Jev and OpenAI $0.
