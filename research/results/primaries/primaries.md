# Primary family (11 tests, Holm, FWER 0.05)

- generated 2026-10-03T11:46:12+00:00 at git `29d9541f5232` by `scripts/primaries.py`; registry `scripts/prereg_effective.py PRIMARIES`; B = 10000, seed 0
- runs: E2a `runs/e2a_jevgym_tool_decisions`, E2a_intern `runs/e2a_jevgym_tool_decisions_intern`, E2b `runs/e2b_jevgym_meta_control`, E2c `runs/e2c_jevgym_baselines`, tau2 `runs/tau2`, E1 `runs/e1_bfcl_when2call`
- competence gate (`results/gpu_round2/gate_verdict.json`): intern_4B co-primary → two-family conjunction
- flags: 0 ERROR(s), 0 WARN(s)

| # | test | arm → reference | n′ / n (dropped) | Δ (95 % CI), pt | W / L | p (rule) | Holm: rank, α_adj, p_Holm | decision | label | MDE, pt |
|---|---|---|---|---|---|---|---|---|---|---|
| C1F | NI | jev: `jev_ts_cascade_tauhat` → `llm_only` | 500 / 500 (0) | +0.0 [+0.0, +0.0] | 0 / 0 | 0.0001 (bb_ordered) | 2, 0.0050, 0.0010 ✓ | NI: policy NI with zero coverage for Intern-Decision (A6.11 (c).4, A6.12, A6.16 (m), A6.20 (e), A6.22 (a)) | cross-backend replication of policy NI | – |
|  |  | intern: `intern_ts_cascade_tauhat` → `llm_only` | 500 / 500 (0) | +0.2 [+0.0, +0.6] | 1 / 0 | 0.0001 (bb_ordered) |  | NI |  | -2.5 |
| C1L | NI | jev: `jev_ts_cascade_tau0.9_s2lite` → `llm_only_s2lite` | 500 / 500 (0) | +1.0 [-1.0, +3.2] | 17 / 12 | 0.0004 (bb_ordered) | 5, 0.0071, 0.1683 | not NI as a conjunction: Jev NI; Intern-Decision inconclusive at this n | split (Jev passes) | +0.5 |
|  |  | intern: `intern_ts_cascade_tau0.9_s2lite` → `llm_only_s2lite` | 500 / 500 (0) | -0.6 [-2.8, +1.6] | 15 / 18 | 0.0240 (bb_ordered) |  | inconclusive at this n |  | +0.8 |
| C2F | NI | jev: `jev_ts_assist_k3` → `llm_only` | 500 / 500 (0) | -7.8 [-10.2, -5.4] | 1 / 40 | 1.0000 (bb_ordered) | 10, 0.0250, 1.0000 | not NI as a conjunction: Jev not NI: loss of 7.8 pt (95 % CI entirely below 0); Intern-Decision not NI: loss of 3.6 pt (95 % CI entirely below 0) | not reached (Holm step-down stopped at an earlier primary) | +0.6 |
|  |  | intern: `intern_ts_assist_k3` → `llm_only` | 500 / 500 (0) | -3.6 [-5.4, -2.0] | 2 / 20 | 0.7666 (bb_ordered) |  | not NI: loss of 3.6 pt (95 % CI entirely below 0) |  | -0.4 |
| C2L | NI | jev: `jev_ts_assist_k3_s2lite` → `llm_only_s2lite` | 500 / 500 (0) | -3.4 [-7.2, +0.6] | 43 / 60 | 0.5905 (bb_ordered) | 11, 0.0500, 1.0000 | not NI as a conjunction: Jev inconclusive at this n; Intern-Decision not NI: loss of 14.8 pt (95 % CI entirely below 0) | not reached (Holm step-down stopped at an earlier primary) | +2.0 |
|  |  | intern: `intern_ts_assist_k3_s2lite` → `llm_only_s2lite` | 500 / 500 (0) | -14.8 [-19.0, -10.6] | 27 / 101 | 1.0000 (bb_ordered) |  | not NI: loss of 14.8 pt (95 % CI entirely below 0) |  | +2.6 |
| C3 | Sup. | `jev_ts_assist_k3` → `bm25_ts_assist_k3` | 500 / 500 (0) | +73.2 [+68.0, +78.2] | 402 / 36 | <0.0001 (sign_exact) | 1, 0.0045, <0.0001 ✓ | superior |  | +14.4 |
| C4 | Sup. | `jev_route_tau0.4` → `small_only` | 500 / 500 (0) | +0.6 [-1.6, +2.8] | 17 / 14 | 0.3601 (sign_exact) | 6, 0.0083, 1.0000 | superiority not established (Holm step-down stopped earlier) |  | +3.6 |
| C5 | Sup. (ΔU) | `jev_reflect_trigger_s2lite` → `ratem_reflect_trigger_s2lite` | 500 / 500 (0) | ΔU $-0.0233 [-0.0557, +0.0087] | 499 ≠ 0 | 0.9175 (paired_t) | 8, 0.0125, 1.0000 | superiority not established (Holm step-down stopped earlier) |  | $0.0516 |
| C6F | NI | jev: `jev_stop_guard` → `llm_only` | 500 / 500 (0) | -0.6 [-1.4, +0.0] | 0 / 3 | 0.0003 (bb_ordered) | 4, 0.0063, 0.0023 ✓ | NI | cross-backend replication | -1.9 |
|  |  | intern: `intern_stop_guard` → `llm_only` | 500 / 500 (0) | -0.4 [-1.0, +0.0] | 0 / 2 | 0.0001 (bb_ordered) |  | NI |  | -2.2 |
| C6L | NI | jev: `jev_stop_guard_s2lite` → `llm_only_s2lite` | 500 / 500 (0) | +1.6 [-0.8, +4.0] | 23 / 15 | 0.0003 (bb_ordered) | 9, 0.0167, 1.0000 | not NI as a conjunction: Jev inconclusive at this n; Intern-Decision not NI: loss of 6.0 pt (95 % CI entirely below 0) | not reached (Holm step-down stopped at an earlier primary) | +0.6 |
|  |  | intern: `intern_stop_guard_s2lite` → `llm_only_s2lite` | 500 / 500 (0) | -6.0 [-9.2, -3.0] | 17 / 47 | 0.9746 (bb_ordered) |  | not NI: loss of 6.0 pt (95 % CI entirely below 0) |  | +1.7 |
| C7 | NI | `jev_ts_cascade_tauhat_refit` → `llm_only` | 149 / 149 (0) | -8.1 [-16.1, +0.0] | 15 / 27 | 0.7662 (bb_ordered) | 7, 0.0100, 1.0000 | transfer NI was not established; inconclusive at this n |  | +8.7 |
| C8 | NI | `jev` → `gemini_flash_typed` | 1114 / 1114 (0) | -0.4 [-1.5, +0.8] | 21 / 25 | 0.0001 (bb_ordered) | 3, 0.0056, 0.0013 ✓ | NI |  | -1.0 |

Full-set sensitivity row (errored / missing tasks count as failures in the arm where they errored (§1, A6.9)): Holm rejects C1F, C3, C6F, C8; identical decisions

p enters Holm as listed (C1 / C2 / C6: max(p_Jev, p_Intern) when Intern-Decision is co-primary; the row of each constituent shows its own p). NI rows: certified Berger–Boos p of one ordering of all (wins, losses) outcomes (`bb_ordered`, A6.21); superiority rows: exact sign test (C3, C4) / one-sided paired t-test (C5; the sign-flip p is descriptive). Δ in percentage points (arm − reference; C5 in $ of utility at W = $1). MDE: the true Δ at which the registered test has 80 % power at the achieved discordance and the primary's Holm level.

## Per primary

### C1F — tool_select + tool_params CASCADE at the deployed plug-in τ̂\* (the calibration-arm fit: scalar projection, A6.5 / A6.11 (c).7 / A6.19) loses ≤ 3 pt success vs LLM-only on Flash

- p entering Holm: 0.0001 (max(p_Jev, p_Intern)); decision: **NI: policy NI with zero coverage for Intern-Decision (A6.11 (c).4, A6.12, A6.16 (m), A6.20 (e), A6.22 (a))**; label: **cross-backend replication of policy NI**
- jev: `jev_ts_cascade_tauhat` (runs/e2a_jevgym_tool_decisions) vs `llm_only` (runs/e2a_jevgym_tool_decisions); System-2 gemini-3.5-flash / gemini-3.5-flash
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 99.8 vs 99.8 %; Δ +0.0 pt [+0.0, +0.0]; wins 0, losses 0 (discordance 0.0 %)
  - decisive p 0.0001 (bb_ordered); asymptotic Tango p <0.0001; Tango lower bound at α_adj -1.3 pt; bootstrap p_NI 0.0000 (descriptive); loss-rate UB 0.6 %; reading: NI; MDE n/a (no discordant task: the exact loss-rate upper bound is reported instead (A6.1))
  - full-set row (errored / missing = failure): n 500, Δ +0.0 pt, p 0.0001 (bb_ordered)
  - delegation (A6.22 (a) 5, 5 % rule of A6.20 (e)): **non-negligible** -- tool-decision coverage 27.2 %, argument coverage 20.8 % (1264 of 6072 executed argument slots; inside System-1-chosen calls 67.7 %); pooled delegation ratio 40.6 % (descriptive)
  - cost / episode $0.0364 vs $0.0424: savings 14.2 % [+12.7, +15.7]; System-2 input tokens 33929 vs 45549 (savings 25.5 %)
- intern: `intern_ts_cascade_tauhat` (runs/e2a_jevgym_tool_decisions_intern) vs `llm_only` (runs/e2a_jevgym_tool_decisions_intern); System-2 gemini-3.5-flash / gemini-3.5-flash
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 99.8 vs 99.6 %; Δ +0.2 pt [+0.0, +0.6]; wins 1, losses 0 (discordance 0.2 %)
  - decisive p 0.0001 (bb_ordered); asymptotic Tango p <0.0001; Tango lower bound at α_adj -1.1 pt; bootstrap p_NI 0.0000 (descriptive); loss-rate UB 0.6 %; reading: NI; MDE -2.5 pt
  - full-set row (errored / missing = failure): n 500, Δ +0.2 pt, p 0.0001 (bb_ordered)
  - delegation (A6.22 (a) 5, 5 % rule of A6.20 (e)): **zero** -- tool-decision coverage 0.0 %, argument coverage 0.0 % (0 of 6076 executed argument slots; inside System-1-chosen calls – %); pooled delegation ratio 0.0 % (descriptive)
  - cost / episode $0.0437 vs $0.0436: savings -0.2 % [-0.7, +0.4]; System-2 input tokens 45619 vs 45618 (savings -0.0 %)

### C1L — the same plug-in CASCADE policy on Flash-Lite

- p entering Holm: 0.0240 (max(p_Jev, p_Intern)); decision: **not NI as a conjunction: Jev NI; Intern-Decision inconclusive at this n**; label: **split (Jev passes)**
- jev: `jev_ts_cascade_tau0.9_s2lite` (runs/e2a_jevgym_tool_decisions) vs `llm_only_s2lite` (runs/e2a_jevgym_tool_decisions); System-2 gemini-3.5-flash-lite / gemini-3.5-flash-lite
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 81.2 vs 80.2 %; Δ +1.0 pt [-1.0, +3.2]; wins 17, losses 12 (discordance 5.8 %)
  - decisive p 0.0004 (bb_ordered); asymptotic Tango p 0.0004; Tango lower bound at α_adj -1.8 pt; bootstrap p_NI 0.0000 (descriptive); loss-rate UB – %; reading: NI; MDE +0.5 pt
  - full-set row (errored / missing = failure): n 500, Δ +1.0 pt, p 0.0004 (bb_ordered)
  - delegation (A6.22 (a) 5, 5 % rule of A6.20 (e)): **zero** -- tool-decision coverage 0.0 %, argument coverage 0.0 % (0 of 6103 executed argument slots; inside System-1-chosen calls – %); pooled delegation ratio 0.0 % (descriptive)
  - cost / episode $0.0098 vs $0.0098: savings 0.2 % [-0.3, +0.7]; System-2 input tokens 45585 vs 45656 (savings 0.2 %)
- intern: `intern_ts_cascade_tau0.9_s2lite` (runs/e2a_jevgym_tool_decisions_intern) vs `llm_only_s2lite` (runs/e2a_jevgym_tool_decisions_intern); System-2 gemini-3.5-flash-lite / gemini-3.5-flash-lite
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 82.8 vs 83.4 %; Δ -0.6 pt [-2.8, +1.6]; wins 15, losses 18 (discordance 6.6 %)
  - decisive p 0.0240 (bb_ordered); asymptotic Tango p 0.0243; Tango lower bound at α_adj -3.7 pt; bootstrap p_NI 0.0230 (descriptive); loss-rate UB – %; reading: inconclusive at this n; MDE +0.8 pt
  - full-set row (errored / missing = failure): n 500, Δ -0.6 pt, p 0.0240 (bb_ordered)
  - delegation (A6.22 (a) 5, 5 % rule of A6.20 (e)): **zero** -- tool-decision coverage 0.0 %, argument coverage 0.0 % (0 of 6107 executed argument slots; inside System-1-chosen calls – %); pooled delegation ratio 0.0 % (descriptive)
  - cost / episode $0.0098 vs $0.0099: savings 1.0 % [+0.5, +1.4]; System-2 input tokens 45628 vs 45645 (savings 0.0 %)

### C2F — tool_select ASSIST k = 3 (System-1 prunes to top-3 + finish, System-2 decides) loses ≤ 3 pt vs LLM-only on Flash

- p entering Holm: 1.0000 (max(p_Jev, p_Intern)); decision: **not NI as a conjunction: Jev not NI: loss of 7.8 pt (95 % CI entirely below 0); Intern-Decision not NI: loss of 3.6 pt (95 % CI entirely below 0)**; label: **not reached (Holm step-down stopped at an earlier primary)**
- jev: `jev_ts_assist_k3` (runs/e2a_jevgym_tool_decisions) vs `llm_only` (runs/e2a_jevgym_tool_decisions); System-2 gemini-3.5-flash / gemini-3.5-flash
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 92.0 vs 99.8 %; Δ -7.8 pt [-10.2, -5.4]; wins 1, losses 40 (discordance 8.2 %)
  - decisive p 1.0000 (bb_ordered); asymptotic Tango p 1.0000; Tango lower bound at α_adj -10.5 pt; bootstrap p_NI 1.0000 (descriptive); loss-rate UB – %; reading: not NI: loss of 7.8 pt (95 % CI entirely below 0); MDE +0.6 pt
  - full-set row (errored / missing = failure): n 500, Δ -7.8 pt, p 1.0000 (bb_ordered)
  - cost / episode $0.0196 vs $0.0424: savings 53.7 % [+52.8, +54.6]; System-2 input tokens 11016 vs 45549 (savings 75.8 %)
- intern: `intern_ts_assist_k3` (runs/e2a_jevgym_tool_decisions_intern) vs `llm_only` (runs/e2a_jevgym_tool_decisions_intern); System-2 gemini-3.5-flash / gemini-3.5-flash
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 96.0 vs 99.6 %; Δ -3.6 pt [-5.4, -2.0]; wins 2, losses 20 (discordance 4.4 %)
  - decisive p 0.7666 (bb_ordered); asymptotic Tango p 0.7526; Tango lower bound at α_adj -5.7 pt; bootstrap p_NI 0.7727 (descriptive); loss-rate UB – %; reading: not NI: loss of 3.6 pt (95 % CI entirely below 0); MDE -0.4 pt
  - full-set row (errored / missing = failure): n 500, Δ -3.6 pt, p 0.7666 (bb_ordered)
  - cost / episode $0.0205 vs $0.0436: savings 53.0 % [+50.5, +55.1]; System-2 input tokens 12033 vs 45618 (savings 73.6 %)

### C2L — tool_select ASSIST k = 3 on Flash-Lite (the "where System-1 adds success" reading)

- p entering Holm: 1.0000 (max(p_Jev, p_Intern)); decision: **not NI as a conjunction: Jev inconclusive at this n; Intern-Decision not NI: loss of 14.8 pt (95 % CI entirely below 0)**; label: **not reached (Holm step-down stopped at an earlier primary)**
- jev: `jev_ts_assist_k3_s2lite` (runs/e2a_jevgym_tool_decisions) vs `llm_only_s2lite` (runs/e2a_jevgym_tool_decisions); System-2 gemini-3.5-flash-lite / gemini-3.5-flash-lite
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 76.8 vs 80.2 %; Δ -3.4 pt [-7.2, +0.6]; wins 43, losses 60 (discordance 20.6 %)
  - decisive p 0.5905 (bb_ordered); asymptotic Tango p 0.5784; Tango lower bound at α_adj -6.8 pt; bootstrap p_NI 0.5935 (descriptive); loss-rate UB – %; reading: inconclusive at this n; MDE +2.0 pt
  - full-set row (errored / missing = failure): n 500, Δ -3.4 pt, p 0.5905 (bb_ordered)
  - cost / episode $0.0047 vs $0.0098: savings 52.0 % [+51.2, +52.7]; System-2 input tokens 10701 vs 45656 (savings 76.6 %)
- intern: `intern_ts_assist_k3_s2lite` (runs/e2a_jevgym_tool_decisions_intern) vs `llm_only_s2lite` (runs/e2a_jevgym_tool_decisions_intern); System-2 gemini-3.5-flash-lite / gemini-3.5-flash-lite
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 68.6 vs 83.4 %; Δ -14.8 pt [-19.0, -10.6]; wins 27, losses 101 (discordance 25.6 %)
  - decisive p 1.0000 (bb_ordered); asymptotic Tango p 1.0000; Tango lower bound at α_adj -18.4 pt; bootstrap p_NI 1.0000 (descriptive); loss-rate UB – %; reading: not NI: loss of 14.8 pt (95 % CI entirely below 0); MDE +2.6 pt
  - full-set row (errored / missing = failure): n 500, Δ -14.8 pt, p 1.0000 (bb_ordered)
  - cost / episode $0.0046 vs $0.0099: savings 53.7 % [+50.3, +56.6]; System-2 input tokens 12337 vs 45645 (savings 73.0 %)

### C3 — Jev ASSIST k = 3 beats BM25 ASSIST at the same k on Flash (superiority of success)

- p entering Holm: <0.0001 (sign_exact); decision: **superior**
- jev: `jev_ts_assist_k3` (runs/e2a_jevgym_tool_decisions) vs `bm25_ts_assist_k3` (runs/e2c_jevgym_baselines); System-2 gemini-3.5-flash / gemini-3.5-flash
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 92.0 vs 18.8 %; Δ +73.2 pt [+68.0, +78.2]; wins 402, losses 36 (discordance 87.6 %)
  - decisive p <0.0001 (exact sign test, p_wins); bootstrap p_sup 0.0000 (descriptive); MDE +14.4 pt
  - full-set row (errored / missing = failure): n 500, Δ +73.2 pt, p <0.0001 (sign_exact)

### C4 — Jev routing TRIGGER (Flash-Lite by default, Flash when it fires) beats small-only

- p entering Holm: 0.3601 (sign_exact); decision: **superiority not established (Holm step-down stopped earlier)**
- opportunity rule (A5.5 / A6.3): `meta_shadow_s2lite` holds 91 labelled routing positives (3695 negatives, 500 episodes) — threshold 30
- jev: `jev_route_tau0.4` (runs/e2b_jevgym_meta_control) vs `small_only` (runs/e2b_jevgym_meta_control); System-2 gemini-3.5-flash / gemini-3.5-flash-lite
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 81.4 vs 80.8 %; Δ +0.6 pt [-1.6, +2.8]; wins 17, losses 14 (discordance 6.2 %)
  - decisive p 0.3601 (exact sign test, p_wins); bootstrap p_sup 0.3245 (descriptive); MDE +3.6 pt
  - full-set row (errored / missing = failure): n 500, Δ +0.6 pt, p 0.3601 (sign_exact)
  - cost / episode $0.0134 vs $0.0098: savings -36.2 % [-39.3, -33.3]; System-2 input tokens 45636 vs 45605 (savings -0.1 %)

### C5 — Jev reflection TRIGGER beats the matched-rate random gate on Flash-Lite in paired ΔU at W = $1, λ = 0 (gross of System-1's price, net alongside)

- p entering Holm: 0.9175 (paired_t); decision: **superiority not established (Holm step-down stopped earlier)**
- jev: `jev_reflect_trigger_s2lite` (runs/e2c_jevgym_baselines) vs `ratem_reflect_trigger_s2lite` (runs/e2c_jevgym_baselines); System-2 gemini-3.5-flash-lite / gemini-3.5-flash-lite
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - ΔU gross -0.0233 $ (95 % CI -0.0557, +0.0087), net -0.0236 (CI -0.0561, +0.0083); success 83.0 vs 85.4 %; realised firing rates 0.1480 (arm) / 0.1473 (control)
  - full-set row (errored = failure): ΔU -0.0233, p 0.9175, n 500
  - one-step oracle ceiling (W = 1): 0.05862661354755442 CI [0.03364893193223935, 0.08462340100424035], no headroom: False; without budget-kind fatals: 0.05862661354755442

### C6F — stop GUARD (System-1 vets the proposed finish) loses ≤ 3 pt vs LLM-only on Flash

- p entering Holm: 0.0003 (max(p_Jev, p_Intern)); decision: **NI**; label: **cross-backend replication**
- jev: `jev_stop_guard` (runs/e2b_jevgym_meta_control) vs `llm_only` (runs/e2b_jevgym_meta_control); System-2 gemini-3.5-flash / gemini-3.5-flash
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 99.2 vs 99.8 %; Δ -0.6 pt [-1.4, +0.0]; wins 0, losses 3 (discordance 0.6 %)
  - decisive p 0.0003 (bb_ordered); asymptotic Tango p 0.0008; Tango lower bound at α_adj -2.3 pt; bootstrap p_NI 0.0000 (descriptive); loss-rate UB 1.5 %; reading: NI; MDE -1.9 pt
  - full-set row (errored / missing = failure): n 500, Δ -0.6 pt, p 0.0003 (bb_ordered)
  - cost / episode $0.0433 vs $0.0408: savings -6.2 % [-9.1, -4.0]; System-2 input tokens 47099 vs 45604 (savings -3.3 %)
- intern: `intern_stop_guard` (runs/e2a_jevgym_tool_decisions_intern) vs `llm_only` (runs/e2a_jevgym_tool_decisions_intern); System-2 gemini-3.5-flash / gemini-3.5-flash
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 99.2 vs 99.6 %; Δ -0.4 pt [-1.0, +0.0]; wins 0, losses 2 (discordance 0.4 %)
  - decisive p 0.0001 (bb_ordered); asymptotic Tango p 0.0003; Tango lower bound at α_adj -1.9 pt; bootstrap p_NI 0.0000 (descriptive); loss-rate UB 1.3 %; reading: NI; MDE -2.2 pt
  - full-set row (errored / missing = failure): n 500, Δ -0.4 pt, p 0.0001 (bb_ordered)
  - cost / episode $0.0449 vs $0.0436: savings -3.0 % [-6.0, -0.4]; System-2 input tokens 48533 vs 45618 (savings -6.4 %)

### C6L — stop GUARD on Flash-Lite

- p entering Holm: 0.9746 (max(p_Jev, p_Intern)); decision: **not NI as a conjunction: Jev inconclusive at this n; Intern-Decision not NI: loss of 6.0 pt (95 % CI entirely below 0)**; label: **not reached (Holm step-down stopped at an earlier primary)**
- jev: `jev_stop_guard_s2lite` (runs/e2b_jevgym_meta_control) vs `llm_only_s2lite` (runs/e2b_jevgym_meta_control); System-2 gemini-3.5-flash-lite / gemini-3.5-flash-lite
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 82.0 vs 80.4 %; Δ +1.6 pt [-0.8, +4.0]; wins 23, losses 15 (discordance 7.6 %)
  - decisive p 0.0003 (bb_ordered); asymptotic Tango p 0.0003; Tango lower bound at α_adj -1.1 pt; bootstrap p_NI 0.0002 (descriptive); loss-rate UB – %; reading: inconclusive at this n; MDE +0.6 pt
  - full-set row (errored / missing = failure): n 500, Δ +1.6 pt, p 0.0003 (bb_ordered)
  - cost / episode $0.0130 vs $0.0098: savings -32.1 % [-39.4, -25.0]; System-2 input tokens 63432 vs 45698 (savings -38.8 %)
- intern: `intern_stop_guard_s2lite` (runs/e2a_jevgym_tool_decisions_intern) vs `llm_only_s2lite` (runs/e2a_jevgym_tool_decisions_intern); System-2 gemini-3.5-flash-lite / gemini-3.5-flash-lite
  - n′ = 500 of 500; arm status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 500, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 77.4 vs 83.4 %; Δ -6.0 pt [-9.2, -3.0]; wins 17, losses 47 (discordance 12.8 %)
  - decisive p 0.9746 (bb_ordered); asymptotic Tango p 0.9733; Tango lower bound at α_adj -9.6 pt; bootstrap p_NI 0.9757 (descriptive); loss-rate UB – %; reading: not NI: loss of 6.0 pt (95 % CI entirely below 0); MDE +1.7 pt
  - full-set row (errored / missing = failure): n 500, Δ -6.0 pt, p 0.9746 (bb_ordered)
  - cost / episode $0.0144 vs $0.0099: savings -45.3 % [-53.1, -37.7]; System-2 input tokens 72250 vs 45645 (savings -58.3 %)

### C7 — the most cost-saving E2 (Flash, Jev) System-1 configuration that is NI on E2 at the unadjusted level (A6.18 (c)) — eligible: tool+args CASCADE at each swept τ and at τ̂\*, tool ASSIST k = 3, policy GUARD (E2a's `jev_ts_guard`

- p entering Holm: 0.7662 (bb_ordered); decision: **transfer NI was not established; inconclusive at this n**
- configuration (selected on E2, results/primaries/c7_selection.json): `jev_ts_cascade_tauhat` — most cost-saving NI configuration (14.2 % savings); tau² arm family jev_ts_cascade (tau-hat* refit, A5.4 (c))
- jev: `jev_ts_cascade_tauhat_refit` (runs/tau2) vs `llm_only` (runs/tau2); System-2 gemini/gemini-3.5-flash / gemini/gemini-3.5-flash
  - n′ = 149 of 149; arm status {'ok': 149, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 149, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 57.7 vs 65.8 %; Δ -8.1 pt [-16.1, +0.0]; wins 15, losses 27 (discordance 28.2 %)
  - decisive p 0.7662 (bb_ordered); asymptotic Tango p 0.7612; Tango lower bound at α_adj -18.3 pt; bootstrap p_NI 0.7587 (descriptive); loss-rate UB – %; reading: inconclusive at this n; MDE +8.7 pt
  - full-set row (errored / missing = failure): n 149, Δ -8.1 pt, p 0.7662 (bb_ordered)
  - cost / episode $0.0901 vs $0.0762: savings -18.1 % [-24.6, -12.0]; System-2 input tokens 60540 vs 80868 (savings 25.1 %)
  - pass^1 (descriptive): 59.1 vs 64.0 %, Δ -4.9 pt [-10.3, +0.4]
- secondary row `jev_ts_cascade_tauhat_e2` (Holm within the τ² table): Δ -0.7 pt, p 0.1730, NI within the table: False
- secondary row `jev_ts_assist_k3` (Holm within the τ² table): Δ -8.6 pt, p 0.7011, NI within the table: False

### C8 — Jev answering the typed tool-selection question loses ≤ 3 pt item-weighted accuracy vs the System-2 model answering the same typed question through structured output, over the BFCL items of the six categories

- p entering Holm: 0.0001 (bb_ordered); decision: **NI**
- jev: `jev` (runs/e1_bfcl_when2call) vs `gemini_flash_typed` (runs/e1_bfcl_when2call); System-2 – / –
  - n′ = 1114 of 1114; arm status {'ok': 1114, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}, reference status {'ok': 1114, 'errored': 0, 'retry_pending': 0, 'missing': 0, 'incomplete': 0}
  - success 89.9 vs 90.3 %; Δ -0.4 pt [-1.5, +0.8]; wins 21, losses 25 (discordance 4.1 %)
  - decisive p 0.0001 (bb_ordered); asymptotic Tango p <0.0001; Tango lower bound at α_adj -2.0 pt; bootstrap p_NI 0.0000 (descriptive); loss-rate UB – %; reading: NI; MDE -1.0 pt
  - full-set row (errored / missing = failure): n 1114, Δ -0.4 pt, p 0.0001 (bb_ordered)
  - $ / 1k decisions 0.0167 vs 0.6295; p50 latency 206 vs 2551 ms (ratio 0.081)

## Flags

- none

## Provenance

- E2a `runs/e2a_jevgym_tool_decisions`
  - `jev_ts_cascade_tauhat`: `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tauhat.jsonl` sha256 `752dd92afc9e01a5` records 500
  - `llm_only`: `runs/e2a_jevgym_tool_decisions/llm_only.jsonl` sha256 `14d2c35fc9f1d284` records 500
  - `jev_ts_cascade_tau0.9_s2lite`: `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tau0.9_s2lite.jsonl` sha256 `37a604be802b3020` records 500
  - `llm_only_s2lite`: `runs/e2a_jevgym_tool_decisions/llm_only_s2lite.jsonl` sha256 `1f98ab49eebdc9de` records 500
  - `jev_ts_assist_k3`: `runs/e2a_jevgym_tool_decisions/jev_ts_assist_k3.jsonl` sha256 `9065c8675fedc242` records 500
  - `jev_ts_assist_k3_s2lite`: `runs/e2a_jevgym_tool_decisions/jev_ts_assist_k3_s2lite.jsonl` sha256 `d98f2bf7390be459` records 500
  - `jev_ts_guard`: `runs/e2a_jevgym_tool_decisions/jev_ts_guard.jsonl` sha256 `ac9567ef11cc6465` records 200
  - `jev_ts_cascade_tau0.5`: `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tau0.5.jsonl` sha256 `e67717a961f6f47e` records 200
  - `jev_ts_cascade_tau0.7`: `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tau0.7.jsonl` sha256 `59cba2fc40cc148d` records 200
  - `jev_ts_cascade_tau0.8`: `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tau0.8.jsonl` sha256 `eec8555f9d3613a0` records 201
  - `jev_ts_cascade_tau0.9`: `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tau0.9.jsonl` sha256 `e9bd115d95f88b8c` records 200
  - `jev_ts_cascade_tau0.95`: `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tau0.95.jsonl` sha256 `4894df1f752e863f` records 200
  - `jev_ts_cascade_tau0.99`: `runs/e2a_jevgym_tool_decisions/jev_ts_cascade_tau0.99.jsonl` sha256 `e9c00020314ca37a` records 200
- E2a_intern `runs/e2a_jevgym_tool_decisions_intern`
  - `intern_ts_cascade_tauhat`: `runs/e2a_jevgym_tool_decisions_intern/intern_ts_cascade_tauhat.jsonl` sha256 `f05da80182d5b687` records 500
  - `llm_only`: `runs/e2a_jevgym_tool_decisions_intern/llm_only.jsonl` sha256 `4f689e6c3498f25f` records 500
  - `intern_ts_cascade_tau0.9_s2lite`: `runs/e2a_jevgym_tool_decisions_intern/intern_ts_cascade_tau0.9_s2lite.jsonl` sha256 `66684b6e76063ef6` records 500
  - `llm_only_s2lite`: `runs/e2a_jevgym_tool_decisions_intern/llm_only_s2lite.jsonl` sha256 `ede44c7cab8f7766` records 500
  - `intern_ts_assist_k3`: `runs/e2a_jevgym_tool_decisions_intern/intern_ts_assist_k3.jsonl` sha256 `42c5a7047e6aba78` records 500
  - `intern_ts_assist_k3_s2lite`: `runs/e2a_jevgym_tool_decisions_intern/intern_ts_assist_k3_s2lite.jsonl` sha256 `052a9d72b19e044e` records 500
  - `intern_stop_guard`: `runs/e2a_jevgym_tool_decisions_intern/intern_stop_guard.jsonl` sha256 `3e07503e55fd2dd5` records 500
  - `intern_stop_guard_s2lite`: `runs/e2a_jevgym_tool_decisions_intern/intern_stop_guard_s2lite.jsonl` sha256 `e4fcab0f32ce97b7` records 500
- E2b `runs/e2b_jevgym_meta_control`
  - `jev_stop_guard`: `runs/e2b_jevgym_meta_control/jev_stop_guard.jsonl` sha256 `ae755cc7cfffa9b4` records 500
  - `llm_only`: `runs/e2b_jevgym_meta_control/llm_only.jsonl` sha256 `632ab45cc7ad833c` records 500
  - `jev_stop_guard_s2lite`: `runs/e2b_jevgym_meta_control/jev_stop_guard_s2lite.jsonl` sha256 `d9e853e54ef99435` records 500
  - `llm_only_s2lite`: `runs/e2b_jevgym_meta_control/llm_only_s2lite.jsonl` sha256 `3b7e8b91a70d1e05` records 500
  - `meta_shadow_s2lite`: `runs/e2b_jevgym_meta_control/meta_shadow_s2lite.jsonl` sha256 `4d6c2bc28bf63608` records 500
  - `jev_route_tau0.4`: `runs/e2b_jevgym_meta_control/jev_route_tau0.4.jsonl` sha256 `3e63174d0ed6a92d` records 500
  - `small_only`: `runs/e2b_jevgym_meta_control/small_only.jsonl` sha256 `cf69e79ea717ec5b` records 500
- E2c `runs/e2c_jevgym_baselines`
  - `bm25_ts_assist_k3`: `runs/e2c_jevgym_baselines/bm25_ts_assist_k3.jsonl` sha256 `d74fb5e5a875b191` records 500
  - `jev_reflect_trigger_s2lite`: `runs/e2c_jevgym_baselines/jev_reflect_trigger_s2lite.jsonl` sha256 `044c0d35a44050e1` records 500
  - `ratem_reflect_trigger_s2lite`: `runs/e2c_jevgym_baselines/ratem_reflect_trigger_s2lite.jsonl` sha256 `b3e53887c9816dae` records 500
- tau2 `runs/tau2`
  - `llm_only`: `runs/tau2/airline/llm_only.jsonl` sha256 `6fb538f76ae21cb9` records 150
  - `llm_only`: `runs/tau2/retail/llm_only.jsonl` sha256 `25b4ea4d6d9f75a4` records 342
  - `jev_ts_cascade_tauhat_refit`: `runs/tau2/airline/jev_ts_cascade_tauhat_refit.jsonl` sha256 `807c3119cc0294c8` records 150
  - `jev_ts_cascade_tauhat_refit`: `runs/tau2/retail/jev_ts_cascade_tauhat_refit.jsonl` sha256 `ababe709333ec51a` records 342
  - `jev_ts_cascade_tauhat_e2`: `runs/tau2/airline/jev_ts_cascade_tauhat_e2.jsonl` sha256 `0c36e4564c7d63d7` records 150
  - `jev_ts_cascade_tauhat_e2`: `runs/tau2/retail/jev_ts_cascade_tauhat_e2.jsonl` sha256 `be03fb6872969dc5` records 342
  - `jev_ts_assist_k3`: `runs/tau2/airline/jev_ts_assist_k3.jsonl` sha256 `526e6aa0f74aab25` records 150
- E1 `runs/e1_bfcl_when2call`
  - `jev`: `runs/e1_bfcl_when2call/s1_jev.jsonl` sha256 `f4e4621adf1a0050` records 3991
  - `gemini_flash_typed`: `runs/e1_bfcl_when2call/s1_gemini_flash_typed.jsonl` sha256 `e0a736fd939463a1` records 3991
