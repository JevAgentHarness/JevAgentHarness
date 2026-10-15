# Replay plan: scoring every decision model on the new hard sets (draft, nothing launched)

Status: **plan only.** No replay, GPU job or API call was started for it. Item counts come from
`results/benchmark_difficulty/difficulty.json` (`replay_inventory`, written by `scripts/benchmark_difficulty.py` from the
stored logs); per-question times are the measured S2-3 replay times on C (L40S hosts, `results/stage2/s23/answers_*.jsonl`
`wall_ms`), and the Jev price is the measured OpenRouter cost of the C replay ($0.462 for 3,939 questions, i.e. $1.17e-4
per question at C's mean of 6,581 characters per question).

## What is already scored at no cost (in `difficulty.md`)

| set | items | models with stored answers |
|---|---|---|
| C, the S2-3 common-state corpus (fresh tasks 300000-300199), seven rule-based subsets, all >= 300 occurrences | 5,314 occurrences (957 in the "hard" union) | all eight decision models + base, SFT-only, v2; Flash on its own states |
| Stage-1 shadow logs, Flash and Flash-Lite states (seeds 200000-200499), states asked identically in both shadow runs | 1,393 / 1,693 tool decisions, 1,674 / 1,492 argument decisions | agent LLM, Jev, Intern-Decision-4B |
| tau^2 reference corpus (airline test + retail) | 19,934 tool / 4,023 argument occurrences | Jev (as logged), JevAgentRL-4B, Intern-Decision-4B, base |
| L1 static (C8 population): BFCL irrelevance, multi-tool, live arguments; When2Call non-call | 370 / 372 / 1,033 / 599 items | all eight decision models + base, SFT-only; Flash native FC |
| Stage-1 meta decisions in shadow (stop, reflection, memory, routing) | 18,058 distinct Noul questions | Jev only |

Not usable: the H5 hard-task runs (SWE-bench Verified, Terminal-Bench-Science) carry outcome labels only, no per-decision
gold; the Flash-Lite shadow logs have **136** tool decisions after an off-plan step (Flash-Lite's 99 failures are 97
final-answer errors on `multi` tasks with every call on plan), so they do not add recovery states.

## New hard sets that need a replay

| id | set (stored questions; no episode is re-run) | distinct questions | what it adds (subsets >= 300) | answered already |
|---|---|---|---|---|
| **R1** | **Recovery extension**: tool decisions after an off-plan step in the seven live replace arms that are *not* C's sources (StartLux 0.8B / 2B / 4B / 9B / 27B, base, SFT-only, v2; fresh tasks 300000-300199) | 1,092 (1,095 occurrences; 762 from the StartLux-0.8B arm) | C's after-off-plan stratum grows from 386 to about 1,480 occurrences | only the arm's own model, live |
| **R2a** | Stage-1 Flash-Lite shadow, every tool decision of the Jev run | 3,787 | a second common-state set on 500 other tasks: ask_user 528, state change 1,004, step >= 8 506 | Jev (logged), Flash-Lite, Intern on 1,393 |
| **R2b** | Stage-1 Flash-Lite shadow, copy arguments with >= 20 candidates | <= 2,437 | many-candidate arguments (Intern 61 % vs Jev 100 % on the matched part) | Jev, Flash-Lite, Intern on 633 |
| **R3** | Stage-1 Flash shadow, arguments whose gold is outside the candidate list (correct answer: OTHER) | 405 | the only "none of these" argument set with n >= 300 | Jev, Intern on part |
| R4 | tau^2 reference corpus, stratified sample (non-majority tool decisions and >= 10-candidate arguments, per domain) | 3,000 of 22,704 | StartLux rows on tau^2 (full corpus: ~48 A10G hours for six open models; not proposed) | Jev, JevAgentRL-4B, Intern, base |
| R5 | Stage-1 meta shadow sample: every minority-label question (stop yes, memory no, reflection yes, routing yes) + an equal majority sample | ~4,000 of 18,058 | meta decisions where Jev is at its majority-label baseline (memory, reflection) | Jev only |

## Cost and time (planning numbers)

Per-question A10G time = measured L40S time on C x **3** (planning factor for compute-bound prefill on an A10G vs an
L40S; check it with a 50-question timing smoke per model before the full pass) x (characters per question / C's).
Six open core rows = JevAgentRL-4B, StartLux 0.8B / 2B / 4B / 9B, Intern-Decision-4B; extra rows = base, SFT-only, v2.

| set | questions | A10G h, six open core rows (RL / SL0.8 / SL2 / SL4 / SL9 / Intern) | A10G h, extra rows | StartLux-27B (L40S h) | Jev via OpenRouter |
|---|---|---|---|---|---|
| R1 | 1,092 | 1.7 (0.23 / 0.08 / 0.14 / 0.31 / 0.52 / 0.42) | 0.8 | 0.9 | $0.13 |
| R2a | 3,787 | 5.6 (0.76 / 0.27 / 0.45 / 1.03 / 1.70 / 1.39) | 2.6 | 3.1 | logged ($0.42 if replayed) |
| R2b | <= 2,437 | 1.3 | 0.6 | 0.7 | logged ($0.09) |
| R3 | 405 | 0.2 | 0.1 | 0.1 | logged ($0.02) |
| R4 | 3,000 | 6.3 | 2.9 | 3.4 | logged ($0.48) |
| R5 | 4,000 | 2.2 | 1.0 | 1.2 | logged ($0.17) |
| **Tier 1 = R1** | 1,092 | **1.7** | 0.8 | 0.9 | **$0.13** |
| **Tier 2 = R2a + R2b + R3** | ~6,600 | **7.1** | 3.3 | 3.9 | $0 (logged) |
| Tier 3 = R4 + R5 | ~7,000 | 8.5 | 3.9 | 4.6 | $0 (logged) |

Add about 5 minutes per model load (about ten loads per tier, 0.8 h). At the g5.4xlarge on-demand price ($1.624 / h,
the harness_final tariff), with every open row (core + extra + loads): Tier 1 ≈ 3.3 h ≈ $5, Tier 2 ≈ 11.2 h ≈ $18
(more than the box's 10-hour guard: two invocations), Tier 3 ≈ 13.2 h ≈ $21. Jev costs at most $1.30 if every set is
replayed through Jev instead of using its logged answers (full tau^2: $3.60).

**StartLux-27B does not fit the A10G**: its evaluated row is the Q8_0 GGUF (about 28 GB) and the A10G has 24 GB. Either
(a) run the 27B on an L40S host (g6e; capacity was missing in us-east-1d on 2026-10-05), or (b) serve the Q4_K_M GGUF on
the A10G as a separately labelled row (a different quantisation than every reported 27B number), or (c) report seven
models on the new sets. Recommendation: (a) if a g6e is available, else (c). StartLux-9B (19.4 GB in BF16) fits with
about 4 GB to spare at one request at a time, as the vendor server serves.

## Procedure (when the author approves; nothing here runs automatically)

1. **Do not share the GPU with a measured run.** The A10G box is busy with the replay benchmark until about 19:15 UTC and
   then with the harness_final live run, whose wall-clock and System-1 times are a reported quantity. Start only after
   the live run's last pass and its tariff window are complete and recorded (`results/harness_final/`), and stop the
   box afterwards (10-hour guard).
2. **Freeze the item lists first** (no model answer involved): `scripts/benchmark_difficulty.py` gets a `freeze` command
   that writes `results/benchmark_difficulty/replay/{r1,r2a,r2b,r3}/questions.jsonl.gz` + `occurrences.jsonl.gz` with
   sha256 (C's format; R2 / R3 questions are the stored shadow questions with their option descriptions re-attached from
   the suite factories, as `stage2_corpus.build_corpus` does). The questions come from stored logs only: no JevGym episode
   is run on any evaluation seed (the `JEV_EVAL_RUNS` guard is not touched).
3. **One replay path**, as for C (A7.10 (c)): `scripts/stage2_corpus.py replay --corpus <set dir> --name <row> --s1 <spec>
   --cache cache/s1_difficulty_replay.sqlite` (a new cache, separate from every rollout cache), sorted question hash,
   one question per request, two retries then incorrect. Frozen checkpoints are sha256-checked against
   `results/stage2/rl_eval_plan.json` / `v2_eval_plan.json` before serving; StartLux with the vendor server
   (`scripts/aws/serve_startlux.sh`); Intern in-process.
4. **Order**: Tier 1 for all rows (≈ 2.5 h with the extra rows), then Tier 2 core rows, then Tier 2 extra rows, then Tier 3
   only if time remains. A 50-question timing smoke per model first; if the measured A10G factor exceeds 4, cut Tier 2
   to R2a.
5. **Score** with `scripts/benchmark_difficulty.py` (same subset rules, MIN_N = 300, task bootstrap in the appendix
   only); label every new row descriptive and post hoc (Deviations entry before the first replay).

## What it would show

R1 tests whether JevAgentRL-4B's 94.7 % after an off-plan step (C, 57 tasks) holds on about four times as many recovery
states, most of them reached by a different policy (StartLux-0.8B) than any training-data policy; R2 gives every model a
second common-state set on 500 Stage-1 tasks with Flash-Lite as the agent, where Jev (90.5 %) and Intern-Decision-4B
(72.6 %) already differ by 18 points on the matched tool decisions and by 39 points on many-candidate arguments; R3 is the
only sizeable "none of the candidates" argument set.
