# Decision-point modes on the Terminal-Bench 2.0 dev split

**Development data, not a result.** 15 dev tasks (`splits/tb2_dev.txt`), one trial per task, run on 2026-10-15 to
choose settings for the decision-point modes of [docs/modes.md](../../docs/modes.md). The test split was not used.
On this many tasks the run-to-run noise is larger than most effects (see "Noise"), so read the success column as a
sanity check and the decision columns (what each point did) as the measurement. The summary and the recommended
settings are in [docs/modes.md, "Dev-split measurements"](../../docs/modes.md#dev-split-measurements).

## Setup

- **Harnesses** (Harbor 0.24.0): mini-swe-agent (`mswe`; one tool, `bash`; it submits with
  `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`) and pi 1.2.0 (`read`, `bash`, `edit`, `write`; it ends with a text
  reply). Three concurrent trials per arm.
- **Tiers and decision model**: the README's "lite" pair. Strong Gemini 4 Argon (thinking high), fast Gemini 3.5
  Flash-Lite (thinking low), Jev (`typesafe/jev-1.13`) on OpenRouter, routing tau 0.75, first step strong.
- **Arms**: one proxy per phase, one model name per arm (`configs/gemini_argon_lite_modes.yaml`), arms of a phase
  run side by side on one host (m7i.8xlarge). Phase 1 (P1, row `dev-modes`, 08:02-09:11 UTC), phase 2 (P2,
  `dev-modes2`, 09:02-10:01 UTC), phase 3 (P3, `dev-modes3`, 10:04-11:02 UTC). Compare an arm with the default
  arm of its own phase.
- **Measures**: success = Harbor reward 1; $/task = LLM + decision-model dollars logged by the proxy; agent s/task =
  Harbor's agent-execution time; decision calls and what each point did from `decisions.jsonl`
  (`bench/summarize.py` for the first columns).
- **Spend**: $53.18 in total (P1 $20.04, P2 $19.47, P3 $13.67), of which Jev $0.64, and one m7i.8xlarge host for
  about 3.1 hours (stopped afterwards). The proxy cap of each phase bounded the spend; no session was refused.
- **Failures that were not the agent's answer**: 17 agent timeouts (counted as failures, as in the protocol), 2
  non-zero agent exits (kv-store-grpc in pi:jah-shadow P1 and in mswe:jah-auto-t75 P3), no infrastructure errors, no
  LLM errors. One of 6,533 decision calls failed (an OpenRouter HTTP 499, P2); the step fell back to the strong tier.
  The proxy now retries a 499.

## Arms

| harness:arm | phase | what it is | success | $/task | agent s/task | strong share | steps/task | LLM calls/step | decision calls/task | decision $/task (share) | point acted/task |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mswe:jah-shadow | P1 | default + every other point in shadow | 12/15 (80%) | 0.185 | 280 | 36.4% | 21.7 | 1.00 | 63.5 | 0.0067 (3.6%) | - |
| mswe:jah-safe | P1 | default + action guard (safe preset) | 12/15 (80%) | 0.411 | 420 | 48.3% | 26.1 | 1.00 | 25.5 | 0.0029 (0.7%) | action 0 |
| mswe:jah-auto-t75 | P2 | default (README setting) | 10/15 (67%) | 0.246 | 311 | 40.9% | 23.7 | 1.00 | 22.7 | 0.0025 (1.0%) | - |
| mswe:jah-auto-t75 | P3 | default (README setting) | 11/15 (73%) | 0.284 | 358 | 48.3% | 24.8 | 1.00 | 23.8 | 0.0027 (0.9%) | - |
| mswe:jah-submit | P2 | default + finish check on mswe's submit (stop shadow, `submit`) | 12/15 (80%) | 0.430 | 438 | 56.3% | 28.8 | 1.13 | 31.3 | 0.0035 (0.8%) | - |
| mswe:jah-reflect | P2 | default + reflection trigger (tau 0.8) | 13/15 (87%) | 0.231 | 288 | 41.3% | 21.1 | 1.00 | 40.1 | 0.0044 (1.9%) | reflection 3.33 |
| mswe:jah-reflect | P3 | default + reflection trigger (tau 0.8) | 13/15 (87%) | 0.304 | 380 | 47.9% | 21.9 | 1.00 | 41.9 | 0.0047 (1.5%) | reflection 4.73 |
| mswe:jah-routeguard | P2 | routing guard (tau 0.5) instead of the cascade | 13/15 (87%) | 0.390 | 415 | 35.5% | 28.8 | 1.34 | 27.5 | 0.0033 (0.8%) | routing 9.67 |
| pi:jah-shadow | P1 | default + every other point in shadow | 11/15 (73%) | 0.389 | 564 | 48.7% | 27.1 | 1.15 | 84.1 | 0.0065 (1.7%) | - |
| pi:jah-stopguard | P1 | default with the stop guard (tau 0.9) instead of the finish check | 9/15 (60%) | 0.351 | 567 | 57.4% | 28.1 | 1.16 | 31.7 | 0.0024 (0.7%) | stop 4.53 |
| pi:jah-reflect | P3 | default + reflection trigger (tau 0.8) | 13/15 (87%) | 0.323 | 461 | 59.0% | 22.7 | 1.14 | 43.3 | 0.0032 (1.0%) | reflection 3.73 |

"point acted/task": how often an acting point changed a step (reflection: a step sent to the strong tier with the
re-check note; routing guard: a strong redo of a fast step; stop guard: a strong redo of a fast final reply; action
guard: a strong re-check). Decision $ includes the routing question.

## What each point did

| Point and mode | Arm (phase) | Asked per task | Fired or acted | What acting changed |
|---|---|---|---|---|
| `tool_select: shadow` | mswe:jah-shadow (P1) | 21.7 | At p ≥ 0.99 it would have forced `bash` on 65% of steps. | Nothing to learn: one tool (`bash` or a text reply); `jah fit` leaves these out. |
| `tool_select: shadow` | pi:jah-shadow (P1) | 27.1 | Jev's top tool equalled pi's in 68% (`bash` 84%, `read` 30%, `write` 28%). At p ≥ 0.99: 7.9% of steps, 96% agreement; a tool other than `bash`: 0.8% of steps. | No threshold. Flash-Lite's 60 premature text replies got a tool prediction at p ≥ 0.99 in 0 cases. |
| `stop: shadow` | pi:jah-shadow (P1) | 4.0 | p(done) ≥ 0.9 on 0 of 60 fast final replies. | (labels) The strong redo went on working in 51 of 60. |
| `stop: guard`, tau 0.9 | pi:jah-stopguard (P1) | 4.7 | Accepted 2 of 70 fast final replies, redid 68. | The redo went on working in 58 of 68 (the guard changed the step) and ended the turn in 10. The 2 accepted replies were in tasks that passed. |
| `stop: shadow` with `submit` | mswe:jah-submit (P2) | 3.7 | p(done) ≥ 0.9 on 0 of 55 fast submissions (up to 12 in one task). | (finish check) The strong redo went on working in 44 of 55; Argon redos cost $0.063 per task. |
| `action: shadow` | mswe / pi shadow (P1) | 0.5 / 0.9 | Pre-filter matches: `rm -f` of build or temporary files, `pkill` of the agent's own server, `DROP INDEX`. p(risky) ≥ 0.5 on 1 of 7 / 1 of 13 (0.74 and 0.50). | – |
| `action: guard`, tau 0.5 (`safe`) | mswe:jah-safe (P1) | 0.5 | 0 of 8 pre-filtered calls flagged (max p(risky) 0.02). | No re-check. 0.08 s of decision waiting per task. |
| `reflection: shadow` | mswe / pi shadow (P1) | 20.7 / 26.1 | p(yes) ≥ 0.8 on 17.1% / 17.4% of steps; 20.7% vs 15.6% (mswe) and 23.6% vs 14.6% (pi) of the steps of failing vs passing tasks (AUC 0.65 / 0.58). | 51 of 53 (mswe) and 59 of 68 (pi) firings fell on steps the router sent to Argon anyway. |
| `reflection: trigger`, tau 0.8 | mswe:jah-reflect (P2, P3), pi:jah-reflect (P3) | 20.1 / 20.9 / 21.7 | Fired on 16.6% / 22.6% / 17.2% of steps (3.3 / 4.7 / 3.7 per task). | Moved 6 / 6 / 4 steps off the fast tier; every other firing only added the re-check note to a strong step. |
| `routing: guard`, tau 0.5 | mswe:jah-routeguard (P2) | 27.5 | Strong redo of 35% of the fast steps (9.7 per task). | 1.34 LLM calls per step. Whether the redo changed the step was not logged in this run (`recheck` was added to the routing guard afterwards). |

## `jah fit` on the shadow logs

**pi:jah-shadow P1** (`jah fit`, target 95%, at least 30 decisions, z 1.645)

| point | labelled | fitted tau | acts on | agreement | lower bound | note |
|---|---|---|---|---|---|---|
| tool_select | 406 | - | - | - | - | no threshold reached 95% agreement on >= 30 decisions |
| stop | 60 | - | - | - | - | no threshold reached 95% agreement on >= 30 decisions |

| point:mode | calls | errors | acted | fires at logged tau | p50 latency s | cost $ |
|---|---|---|---|---|---|---|
| action:shadow | 13 | 0 | 0 | 7.7% | 0.143 | 0.000995 |
| reflection:shadow | 391 | 0 | 0 | 17.4% | 0.144 | 0.029196 |
| routing:cascade | 391 | 0 | 391 | 34.0% | 0.143 | 0.030609 |
| stop:shadow | 60 | 0 | 0 | 0.0% | 0.137 | 0.004875 |
| tool_select:shadow | 406 | 0 | 0 | - | 0.144 | 0.032105 |

**mswe:jah-shadow P1** (`jah fit`, target 95%, at least 30 decisions, z 1.645)

| point | labelled | fitted tau | acts on | agreement | lower bound | note |
|---|---|---|---|---|---|---|
| tool_select | 0 | - | - | - | - | 325 decisions offered a single tool (left out: not a choice of tool); 0 labelled decisions with two or more tools |
| stop | 1 | - | - | - | - | too few labelled decisions (1 < 30) |

| point:mode | calls | errors | acted | fires at logged tau | p50 latency s | cost $ |
|---|---|---|---|---|---|---|
| action:shadow | 7 | 0 | 0 | 14.3% | 0.15 | 0.000826 |
| reflection:shadow | 310 | 0 | 0 | 17.1% | 0.15 | 0.032786 |
| routing:cascade | 310 | 0 | 310 | 40.3% | 0.147 | 0.033906 |
| stop:shadow | 1 | 0 | 0 | 0.0% | 0.151 | 5.6e-05 |
| tool_select:shadow | 325 | 0 | 0 | - | 0.146 | 0.033184 |

**mswe:jah-submit P2** (`jah fit`, target 95%, at least 30 decisions, z 1.645)

| point | labelled | fitted tau | acts on | agreement | lower bound | note |
|---|---|---|---|---|---|---|
| tool_select | 0 | - | - | - | - | too few labelled decisions (0 < 30) |
| stop | 55 | - | - | - | - | no threshold reached 95% agreement on >= 30 decisions |

| point:mode | calls | errors | acted | fires at logged tau | p50 latency s | cost $ |
|---|---|---|---|---|---|---|
| routing:cascade | 415 | 1 | 414 | 45.2% | 0.15 | 0.046345 |
| stop:shadow | 55 | 0 | 0 | 0.0% | 0.15 | 0.005746 |

## Noise

The default policy ran four times on mini-swe-agent: the shadow arm (it only logs), the safe arm (its guard never
acted) and `jah-auto-t75` in P2 and P3. They solved 12, 12, 10 and 11 of 15 tasks at $0.185, $0.411, $0.246 and
$0.284 per task. The shadow and safe arms of P1 made the same kind of calls, yet one cost 2.2 times as much: two
tasks (extract-elf, extract-moves-from-video) ran into the time limit in the safe arm and failed early in the shadow
arm. Differences of up to two or three tasks, or up to 2x in cost per task, are therefore noise on this split.
Every acting arm of P2 solved 2 or 3 tasks more than the P2 default, which points to an unlucky default draw as
much as to the modes.

Paired with the default of the same phase (wins / losses over the 15 tasks; cost and agent-time change on the same
tasks):

| Arm | vs | Wins / losses | Cost | Agent time |
|---|---|---|---|---|
| mswe:jah-safe P1 | mswe:jah-shadow P1 | 0 / 0 | +122% | +50% |
| mswe:jah-submit P2 | mswe:jah-auto-t75 P2 | 2 / 0 | +75% | +41% |
| mswe:jah-routeguard P2 | mswe:jah-auto-t75 P2 | 3 / 0 | +59% | +34% |
| mswe:jah-reflect P2 | mswe:jah-auto-t75 P2 | 3 / 0 | -6% | -7% |
| mswe:jah-reflect P3 | mswe:jah-auto-t75 P3 | 3 / 1 | +7% | +6% |
| pi:jah-stopguard P1 | pi:jah-shadow P1 | 0 / 2 | -10% | +1% |
| pi:jah-reflect P3 | pi:jah-shadow P1 (other phase) | 3 / 1 | -17% | -18% |

The reflection trigger is the only mode that kept the default's cost and was ahead in each of its three samples
(mini-swe-agent: 6 wins, 1 loss over two phases, a one-sided sign test gives p = 0.06). That is a reason to register
it as a test row, not a result.

Per task (✓ solved, ✗ not, $ per task, T = agent timeout):

mswe:

| task | jah-shadow P1 | jah-safe P1 | jah-auto-t75 P2 | jah-auto-t75 P3 | jah-submit P2 | jah-reflect P2 | jah-reflect P3 | jah-routeguard P2 |
|---|---|---|---|---|---|---|---|---|
| build-pov-ray | ✓ 0.08 | ✓ 0.16 | ✓ 0.19 | ✓ 0.07 | ✓ 0.51 | ✓ 0.05 | ✓ 0.20 | ✓ 0.10 |
| chess-best-move | ✓ 0.20 | ✓ 0.20 | ✓ 0.16 | ✓ 0.21 | ✓ 0.25 | ✓ 0.20 | ✓ 0.19 | ✓ 0.25 |
| constraints-scheduling | ✓ 0.10 | ✓ 0.19 | ✓ 0.08 | ✓ 0.14 | ✓ 0.11 | ✓ 0.09 | ✓ 0.13 | ✓ 0.08 |
| count-dataset-tokens | ✓ 0.28 | ✓ 0.33 | ✓ 0.26 | ✓ 0.18 | ✓ 0.17 | ✓ 0.17 | ✗ 0.08 | ✓ 0.18 |
| extract-elf | ✗ 0.05 | ✗ 1.08 T | ✗ 0.25 | ✗ 0.36 | ✗ 0.62 | ✓ 0.28 | ✗ 1.31 T | ✓ 1.13 T |
| extract-moves-from-video | ✗ 0.65 | ✗ 2.27 T | ✗ 0.63 | ✗ 1.28 T | ✗ 2.37 T | ✓ 0.73 | ✓ 0.93 | ✗ 1.69 T |
| fix-ocaml-gc | ✓ 0.65 | ✓ 0.98 | ✓ 1.32 | ✓ 0.94 | ✓ 0.93 | ✓ 1.02 | ✓ 0.69 | ✓ 1.44 |
| kv-store-grpc | ✓ 0.02 | ✓ 0.05 | ✗ 0.16 | ✗ 0.13 | ✗ 0.05 | ✗ 0.02 | ✓ 0.19 | ✓ 0.06 |
| log-summary-date-ranges | ✓ 0.02 | ✓ 0.02 | ✓ 0.02 | ✓ 0.05 | ✓ 0.07 | ✓ 0.02 | ✓ 0.03 | ✓ 0.03 |
| pypi-server | ✓ 0.02 | ✓ 0.05 | ✗ 0.01 | ✗ 0.01 | ✓ 0.11 | ✗ 0.03 | ✓ 0.02 | ✗ 0.06 |
| pytorch-model-cli | ✓ 0.09 | ✓ 0.12 | ✓ 0.14 | ✓ 0.12 | ✓ 0.28 | ✓ 0.13 | ✓ 0.10 | ✓ 0.11 |
| query-optimize | ✗ 0.26 | ✗ 0.17 | ✗ 0.16 | ✓ 0.35 | ✓ 0.43 | ✓ 0.21 | ✓ 0.22 | ✓ 0.25 |
| regex-log | ✓ 0.14 | ✓ 0.28 | ✓ 0.12 | ✓ 0.06 | ✓ 0.24 | ✓ 0.17 | ✓ 0.15 | ✓ 0.24 |
| sparql-university | ✓ 0.14 | ✓ 0.22 | ✓ 0.14 | ✓ 0.30 | ✓ 0.25 | ✓ 0.28 | ✓ 0.27 | ✓ 0.19 |
| sqlite-db-truncate | ✓ 0.08 | ✓ 0.06 | ✓ 0.04 | ✓ 0.04 | ✓ 0.06 | ✓ 0.08 | ✓ 0.04 | ✓ 0.04 |

pi:

| task | jah-shadow P1 | jah-stopguard P1 | jah-reflect P3 |
|---|---|---|---|
| build-pov-ray | ✓ 0.42 | ✓ 0.40 | ✓ 0.26 |
| chess-best-move | ✓ 0.22 | ✓ 0.20 | ✓ 0.21 |
| constraints-scheduling | ✗ 0.08 T | ✗ 0.09 T | ✓ 0.10 |
| count-dataset-tokens | ✓ 0.28 | ✓ 0.11 | ✓ 0.06 |
| extract-elf | ✓ 1.06 T | ✗ 0.70 T | ✗ 0.96 T |
| extract-moves-from-video | ✗ 0.89 T | ✗ 1.01 T | ✗ 0.88 T |
| fix-ocaml-gc | ✓ 1.03 | ✓ 0.31 | ✓ 0.64 |
| kv-store-grpc | ✗ 0.08 | ✗ 0.21 | ✓ 0.20 |
| log-summary-date-ranges | ✓ 0.08 | ✓ 0.07 | ✓ 0.07 |
| pypi-server | ✓ 0.14 | ✓ 0.08 | ✓ 0.13 |
| pytorch-model-cli | ✗ 0.65 T | ✗ 0.80 | ✓ 0.33 |
| query-optimize | ✓ 0.26 | ✗ 0.67 T | ✓ 0.45 |
| regex-log | ✓ 0.25 | ✓ 0.24 | ✓ 0.28 |
| sparql-university | ✓ 0.33 | ✓ 0.30 | ✓ 0.19 |
| sqlite-db-truncate | ✓ 0.07 | ✓ 0.07 | ✓ 0.08 |
