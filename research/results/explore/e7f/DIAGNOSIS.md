# E7f step 1: why the frozen 4B collapsed on JevGym-Hard (offline, $0)

Exploratory (E7f), descriptive. Written 2026-10-07 before E7f was registered or run. Input: the logged records of
the frozen 4B arm `jevagent_best_rl` (paper name AgentDecision-4B; records JevAgentRL-4B) in E7 (10 customers, seeds
310000-310199, 3.0 % success) and E7b (12 customers, seeds 310200-310399, 0.5 % success), fetched from
`s3://<bucket>/e7_hard/runs/e7_hard/` and `.../e7_hard/e7b/runs/e7b_hard/`. The files match the
sha256 prefixes the E7 / E7b execution records give (`95df248654cce2a2`, `b96084ba12ec4552`). The host records come
from `results/explore/e7_hard/box{,_e7b}/`.

Command: `python scripts/explore_e7_hard.py diagnose --run <run dir> --arm jevagent_best_rl --out ...`. Full
tables: `diagnosis_e7.{json,md}` and `diagnosis_e7b.{json,md}` in this folder.

## Verdict

**No pipeline bug found. The collapse comes from the model under the new rule.** The 4B ran at the registered
thresholds and was served from the sha256-verified frozen checkpoint. At every cancel-or-ship decision its input held
the rule, the customer's tier and the order amount, and both write tools were offered. It applies the rule wrongly:
it cancels about half of the basic- and silver-tier orders that should be shipped. It also ends long tasks too early.
At its frozen tool-choice threshold (0.5) it takes almost every tool choice. In strict mode one wrong write ends the
episode, so almost every episode fails. Nothing needs fixing before E7f.

## Checks (E7b; E7 in brackets)

1. **Thresholds used.** The runner's resolved profile (`profiles.json`) gives the frozen profile's `jevagent_rl`
   backend block: tool choice cascade τ 0.4999999999978404, arguments cascade τ 0.9999998807907104, reflection trigger
   τ 0.5. The E7 / E7b registration names exactly these values. The decision log was checked against them. All 2,881
   (2,670) tool-choice decisions were accepted exactly when confidence ≥ τ. All 4,751 (4,370) argument decisions were
   accepted exactly when the answer was not OTHER and confidence ≥ τ; 997 (853) were deferred because the answer was
   OTHER. Violations: 0 (0).
2. **Served checkpoint.** In all 200 (200) episodes the System-1 endpoint is `jevagentrl-4b` at
   `http://127.0.0.1:8081`, and all 10,313 (9,510) System-1 calls name backend and model `jevagentrl-4b`.
   `frozen_verify.json` shows that all seven files of the frozen row match their sha256 (adapter, adapter config,
   calibration, tokenizer files, chat template, decision config). `serve.cmd` serves Qwen3-4B (snapshot `1cfa9a72…`)
   with the frozen adapter and its `calibration.json`, without extra arguments (S2-4's server).
3. **What the 4B saw at the rule's decisions.** There were 957 (877) tool-choice states whose gold next tool is a
   write. At every one of them, both `cancel_order` and `ship_order` were among the 69 options, and the task text held
   the policy sentence ("cancel it if the customer is basic tier and the amount is above $300.00, or silver tier and
   the amount is above $600.00 (never cancel a gold-tier customer's order); otherwise ship it"). In 950 of the 950
   (876 of 876) states where the order in view could be identified, the state also showed the customer's
   `"tier"` (from `find_user`, in HISTORY) and the order's `"amount"` (from `get_orders`, the CURRENT OBSERVATION).
   The 7 (1) remaining states are off-plan: an earlier customer had been skipped. The longest state had 5,102 (5,097)
   characters, against the harness cap of 12,000, so no state was cut as a whole. The `...[truncated]...` markers
   come from the history's 400-character limit per observation and remove only the trailing noise lines: the user
   and order records are printed first. The Jev arm saw the same state format in the same runs and was right on
   99.8 % (99.7 %) of these decisions (summary_e7b.md / summary.md).
4. **How the 4B decides cancel vs. ship.** It never cancels a gold-tier order (0 % in every amount band), as the rule
   says. For basic and silver tiers it does not apply the tier's limit:

   | tier | amount band ($) | model says cancel % (E7b) | rule cancels % |
   |---|---|---|---|
   | basic | 0-150 | 35.6 | 0 |
   | basic | 150-300 | 59.6 | 0 |
   | basic | 300-450 | 94.6 | 98.2 |
   | silver | 150-300 | 35.3 | 0 |
   | silver | 300-450 | 72.1 | 0 |
   | silver | 450-600 | 68.1 | 0 |
   | silver | 600-750 | 83.3 | 100 |

   By cell, it is wrong on 47.8 % (51.0 %) of basic orders at or below $300 and on 48.0 % (43.9 %) of silver orders at
   or below $600, all of them accepted (no deferral at τ 0.5). It is wrong on 4.8 % (3.3 %) of basic orders above the
   limit, 17.4 % (9.8 %) of silver orders above it, and 1.8 % (3.3 %) of gold-tier orders.
5. **Failure breakdown** (each failed episode with the decision that made its last step):

   | failure | E7b (0.5 %) | E7 (3.0 %) |
   |---|---|---|
   | wrong write, strict stop: the 4B chose `cancel_order` where the gold was `ship_order` | 139 | 134 |
   | wrong write, strict stop: the 4B chose `ship_order` where the gold was `cancel_order` | 30 | 17 |
   | wrong write, strict stop: the 4B chose the right tool, with a wrong argument | 0 | 1 |
   | early finish: the 4B chose `finish` with customers left | 24 | 37 |
   | wrong final total | 5 | 5 |
   | no final answer | 1 | 0 |
   | failed in total | 199 of 200 | 194 of 200 |

   The 4B chose the tool at every failing step. The LLM never made a failing write. The early finishes came after 6-11
   of 12 writes (E7b) and 4-9 of 10 writes (E7). The 4B's training tasks had 3 customers (the JevGym default).

## What this implies for E7f

The failing decisions are exactly those that E7d retrained: cancel vs. ship at the rule's write states, and the other
tool choices, which include finishing. Offline, AgentDecision-4B (tier) was right on 95.3 % of held-out cancel/ship
choices (frozen: 76.5 %) and 100.0 % of other tool choices (frozen: 91.6 %). Refitted on the E7c calibration states,
it takes 50 % of tool choices instead of none (`results/explore/e7d_tier/summary.md`). E7f runs it live. A
frozen-4B arm with refitted thresholds would add nothing. Its registered fit (E7c, all 100 calibration tasks) turns
its tool choice off, so it would cost about as much as LLM-only.
