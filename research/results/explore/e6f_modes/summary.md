# E6F: the trigger cells at tool choice, tool arguments and action (exploratory (E6f))

- generated 2026-10-07T18:32:22+00:00 at git `4d2e61d631c0` by `scripts/explore_e6f_modes.py`; runs `runs/e6f_modes, runs/e6f_alfworld_eval`; paired task bootstrap B = 10000, seed 0
- spend (every E6f record, smokes included): Gemini $43.55 of the $60 cap, Jev $0.125 of the $2 cap
- verdict rule (descriptive): costs success if delta <= -3 pt; no saving if saving <= 0; takes over safely if delta >= -1 pt, saving > 0 and acted >= 5 %; else mixed (scripts/module_map.py, fixed 2026-10-05)
- definition, questions, verification notes and thresholds: `results/explore/e6f_modes/cells.md`; registration: `docs/PREREG_A7_STAGE2.md`, Deviations "Exploratory row E6f". No test; outside every Holm table.

- reference `llm_only` (jevgym, `runs/e6f_modes`): n = 100 complete, success 100.0 %, $42.2 per 1k episodes, 48.8 s per episode
- reference `llm_only` (alfworld, `runs/e6f_alfworld_eval`): n = 140 complete, success 90.7 %, $19.6 per 1k episodes, 73.0 s per episode

| cell | arm | τ | n | succ. % | Δ pt [95 % CI] | wins / losses | $/1k ep. | saving % [95 % CI] | verified % | fixed / broke | acc. % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_select/trigger | `ts_trigger` | 0.5 | 100 | 100.0 | +0.0 [+0.0, +0.0] | 0 / 0 | 48.2 | -14.2 [-17.4, -11.3] | 13.7 | 0 / 0 | 86.3 | no saving |
| tool_params/trigger | `tp_trigger` | 0.9 | 100 | 100.0 | +0.0 [+0.0, +0.0] | 0 / 0 | 42.7 | -1.2 [-2.8, 0.3] | 0.0 | 0 / 0 | 100.0 | no saving |
| action/trigger | `jev_action_trigger` | 0.5 | 140 | 88.6 | -2.1 [-6.4, +2.1] | 3 / 6 | 41.4 | -111.2 [-148.3, -79.6] | 74.9 | 9 / 23 | 67.1 | no saving |

| always-verify reference | arm | n | Δ pt vs LLM-only | saving % | verified % | fixed / broke | trigger's share of its extra cost | of its Δ | of its fixes | trigger − always, Δ pt |
|---|---|---|---|---|---|---|---|---|---|---|
| tool_select | `ts_verify_all` | 100 | +0.0 [+0.0, +0.0] | -92.6 | 100.0 | 0 / 0 | 15.3 % | -- % | -- % | +0.0 [+0.0, +0.0] |
| tool_params | `tp_verify_all` | 100 | +0.0 [+0.0, +0.0] | -38.1 | 100.0 | 0 / 1 | 3.3 % | -- % | -- % | +0.0 [+0.0, +0.0] |
| action | `action_verify_all` | 140 | -2.9 [-6.4, +0.0] | -159.5 | 100.0 | 3 / 49 | 69.7 % | 75.0 % | 300.0 % | +0.7 [-1.4, +3.6] |

Verified: the share of the point's proposals on which the LLM ran its verification call (trigger: the decision model fired, the module map's "acted"; always: 100 %). Fixed / broke: verifications that turned a wrong proposal into a right one / a right one into a wrong one, by the gold (tool choice: the tool is in the acceptable set AT(s); tool arguments: the call is progress; action: the single gold action, a lower bound). Acc.: the trigger's answer (fire = the proposal is wrong) against that gold. Share of the extra cost: (C_trigger − C_LLM-only) / (C_always − C_LLM-only); share of Δ only where always-verify's Δ ≠ 0.

## Per cell

### tool_select × trigger (`ts_trigger`, τ = 0.5; jevgym, `runs/e6f_modes`)

- n = 100 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 100.0 % vs 100.0 %; Δ +0.0 pt (bootstrap [+0.0, +0.0], Newcombe [-3.7, +3.7]); wins / losses 0 / 0
- cost $48.2 vs $42.2 per 1k episodes (saving -14.2 %); System-1 $0.36, System-2 $47.8 per 1k; LLM calls 8.53 vs 7.50 per episode (8.53 vs 7.50 at this point), System-1 calls 7.50; wall 58.2 s vs 48.8 s (median 46.2 vs 38.5); episodes with an empty LLM reply 0 vs 0
- trigger: 750 proposals, verified 13.7 % (103); wrong proposals 0.0 % of 750 with a gold, of which verified -- % (right ones verified 13.7 %); its answers' accuracy 86.3 %
- what the verifications did: fixed 0, broke 0, confirmed right 103, confirmed wrong 0, revised right 0, revised wrong 0; changed 0.0 % of the verified proposals; replies without a call 0; whole call: fixed 0, broke 0, confirmed right 103, confirmed wrong 0, revised right 0, revised wrong 0
- always-verify (`ts_verify_all`): n = 100; success 100.0 %, Δ +0.0 pt [+0.0, +0.0] (wins / losses 0 / 0); $81.2 per 1k (saving -92.6 %); 15.00 LLM calls per episode at this point; wall 96.6 s; 750 proposals verified, fixed 0, broke 0, confirmed right 733, confirmed wrong 0, revised right 17, revised wrong 0; whole call: fixed 0, broke 0, confirmed right 707, confirmed wrong 0, revised right 43, revised wrong 0
- trigger vs always-verify: share of its extra cost 15.3 %, of its Δ -- %, of its fixes -- % (net of breaks -- %); paired trigger − always Δ +0.0 pt [+0.0, +0.0], wins / losses 0 / 0

### tool_params × trigger (`tp_trigger`, τ = 0.9; jevgym, `runs/e6f_modes`)

- n = 100 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 100.0 % vs 100.0 %; Δ +0.0 pt (bootstrap [+0.0, +0.0], Newcombe [-3.7, +3.7]); wins / losses 0 / 0
- cost $42.7 vs $42.2 per 1k episodes (saving -1.2 %); System-1 $0.36, System-2 $42.3 per 1k; LLM calls 7.50 vs 7.50 per episode (0.00 vs 0.00 at this point), System-1 calls 7.50; wall 53.2 s vs 48.8 s (median 43.9 vs 38.5); episodes with an empty LLM reply 0 vs 0
- trigger: 750 proposals, verified 0.0 % (0); wrong proposals 0.0 % of 750 with a gold, of which verified -- % (right ones verified 0.0 %); its answers' accuracy 100.0 %
- what the verifications did: fixed 0, broke 0, confirmed right 0, confirmed wrong 0, revised right 0, revised wrong 0; changed -- % of the verified proposals; replies without a call 0
- always-verify (`tp_verify_all`): n = 100; success 100.0 %, Δ +0.0 pt [+0.0, +0.0] (wins / losses 0 / 0); $58.3 per 1k (saving -38.1 %); 7.51 LLM calls per episode at this point; wall 191.5 s; 751 proposals verified, fixed 0, broke 1, confirmed right 703, confirmed wrong 0, revised right 47, revised wrong 0
- trigger vs always-verify: share of its extra cost 3.3 %, of its Δ -- %, of its fixes -- % (net of breaks -0.0 %); paired trigger − always Δ +0.0 pt [+0.0, +0.0], wins / losses 0 / 0

### action × trigger (`jev_action_trigger`, τ = 0.5; alfworld, `runs/e6f_alfworld_eval`)

- n = 140 paired tasks (0 dropped: {'missing_arm': 0, 'missing_ref': 0, 'errored_arm': 0, 'errored_ref': 0}); success 88.6 % vs 90.7 %; Δ -2.1 pt (bootstrap [-6.4, +2.1], Newcombe [-7.1, +2.6]); wins / losses 3 / 6
- cost $41.4 vs $19.6 per 1k episodes (saving -111.2 %); System-1 $0.35, System-2 $41.1 per 1k; LLM calls 23.99 vs 13.21 per episode (23.99 vs 13.21 at this point), System-1 calls 13.71; wall 139.0 s vs 73.0 s (median 63.0 vs 33.3); episodes with an empty LLM reply 0 vs 0
- trigger: 1920 proposals, verified 74.9 % (1439); wrong proposals 59.8 % of 1920 with a gold, of which verified 85.2 % (right ones verified 59.7 %); its answers' accuracy 67.1 %
- what the verifications did: fixed 9, broke 23, confirmed right 438, confirmed wrong 888, revised right 0, revised wrong 81; changed 7.9 % of the verified proposals; replies without a call 0
- always-verify (`action_verify_all`): n = 140; success 87.9 %, Δ -2.9 pt [-6.4, +0.0] (wins / losses 1 / 5); $50.9 per 1k (saving -159.5 %); 28.67 LLM calls per episode at this point; wall 166.3 s; 2007 proposals verified, fixed 3, broke 49, confirmed right 753, confirmed wrong 1115, revised right 0, revised wrong 87
- trigger vs always-verify: share of its extra cost 69.7 %, of its Δ 75.0 %, of its fixes 300.0 % (net of breaks 30.4 %); paired trigger − always Δ +0.7 pt [-1.4, +3.6], wins / losses 2 / 1

