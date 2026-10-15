# E29b: step-type routing with a write guard (exploratory, descriptive)

Registered in docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 ~19:55 UTC, E29b, after E29's pilot was read and before any E29b call. E29's host-side proxy asks the decision model which kind of step comes next (inspect / edit / run / finish) while the previous action runs; the fast LLM writes a step only when the answer is inspect with c = p(inspect) >= tau (sticky, at most k = 6 fast steps in a row); the proxy executes a fast-LLM action only if a fixed classifier finds it read-only, else discards it and the strong LLM writes the step. Cost = proxy-logged LLM (discarded calls included) + decision-model $ per task; time = agent execution seconds per task.

## Fitted gates (typed shadow calibration)

| harness | tau | precision % | steps / tasks | read-only base rate % | Jev says inspect (n) | routed (n) | implied routed share % | recall % | guard would accept % | routed cost share % | simulated LLM saving % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tb | 0.92 | 100.0 | 205 / 10 | 24.9 | 98 | 4 | 2.0 | 7.8 | 75.0 | 0.6 | 0.5 |
| mswe | 0.98 | 100.0 | 709 / 20 | 48.4 | 531 | 6 | 0.8 | 1.7 | 100.0 | 0.3 | 0.1 |

Answers: tb: {"inspect": 98, "edit": 66, "run": 39, "finish": 2}; mswe: {"inspect": 531, "edit": 90, "run": 70, "finish": 18}

Post hoc, descriptive (the same calibration records; not part of the registered fit):

- tb: AUROC of p(inspect) for a read-only next step 0.56; read-only steps carried 14.1 % of the strong LLM's cost; read-only share of the strong LLM's next step by Jev's answer: inspect n 98: 25.5 %; edit n 66: 25.8 %; run n 39: 23.1 %; finish n 2: 0.0 %; the fast LLM's response was read-only in 60.8 % of the steps where the strong LLM's was and in 9.1 % where it was not.
- mswe: AUROC of p(inspect) for a read-only next step 0.70; read-only steps carried 41.9 % of the strong LLM's cost; read-only share of the strong LLM's next step by Jev's answer: inspect n 531: 53.7 %; edit n 90: 44.4 %; run n 70: 22.9 %; finish n 18: 11.1 %; the fast LLM's response was read-only in 70.5 % of the steps where the strong LLM's was and in 25.5 % where it was not.

## Pilot (30 tasks per harness; decided the go)

| harness | n | success base / gated % | delta pt [95 %] | W / L | NI p (3 pt) | cost $/task base / gated | saving % [95 %] | mean s base / gated | time saving % | median-time saving % | fast steps % | guard rejections |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pi on Terminal-Bench 2.0 | 28 | 53.6 / 46.4 | -7.1 [-25.0, +10.7] | 2 / 4 | 0.702 (not shown) | 0.5524 / 0.4668 | 15.5 [-2.4, 34.9] | 247 / 225 | 8.8 | 24.0 | 6.2 | 46 of 109 fast calls |
| mini-SWE-agent on SWE-bench Verified | 30 | 66.7 / 66.7 | 0.0 [-10.0, +10.0] | 1 / 1 | 0.774 (not shown) | 0.0388 / 0.0470 | -21.1 [-76.1, 16.5] | 469 / 497 | -6.0 | -8.3 | 1.2 | 0 of 15 fast calls |

- pi on Terminal-Bench 2.0: gated arm 38.0 LLM requests per task (baseline 43.6), 1017 executed steps of which 63 by the fast LLM; 1.1 fast segments per task of mean length 1.97 (15.6 % ended by k = 6); guard: 63 accepted, 46 rejected (discarded $0.0051 per task); decision model 36.3 calls, $0.0033 and 7.2 s per task, of which requests waited 4.53 s; fail-open steps 0, decision errors 0; route reasons {"first": 28, "enter": 69, "continue": 40, "below_tau": 875, "guard_reject": 46, "cap_k": 5}; cached input share 81.8 % (baseline) vs 85.1 % (gated); infrastructure errors 0 / 0; complete case n 28, delta -7.1, NI p 0.702. Missing: {"baseline": 2, "gated": 2}.
- mini-SWE-agent on SWE-bench Verified: gated arm 41.8 LLM requests per task (baseline 41.0), 1246 executed steps of which 15 by the fast LLM; 0.5 fast segments per task of mean length 1.07 (0.0 % ended by k = 6); guard: 15 accepted, 0 rejected (discarded $0.0000 per task); decision model 41.5 calls, $0.0055 and 7.8 s per task, of which requests waited 1.24 s; fail-open steps 0, decision errors 0; route reasons {"first": 30, "enter": 14, "below_tau": 1201, "repeat:below_tau": 8, "continue": 1}; cached input share 91.5 % (baseline) vs 91.1 % (gated); infrastructure errors 0 / 0; complete case n 30, delta 0.0, NI p 0.774. Missing: {"baseline": 0, "gated": 0}.

## Go decision (registered rule, pilot only)

```
{
 "decision": "no-go",
 "per_harness": {
  "tb": {
   "n": 28,
   "saving": 0.15490150173588557,
   "drop": 2,
   "solved_base": 15,
   "solved_gated": 13,
   "no_gate": false,
   "have_pilot": true,
   "pass": false,
   "ok": true,
   "status": null
  },
  "mswe": {
   "n": 30,
   "saving": -0.21093561280064033,
   "drop": 0,
   "solved_base": 20,
   "solved_gated": 20,
   "no_gate": false,
   "have_pilot": true,
   "pass": false,
   "ok": true,
   "status": null
  }
 },
 "rule": "GO iff (pass(tb) and ok(mswe)) or (pass(mswe) and ok(tb)); pass = saving >= 20 % and at most 3 of 30 tasks lost; ok = at most 3 lost; a gate that routes nothing: saving 0, drop 0",
 "registered": "docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 ~19:55 UTC, E29b",
 "written": "2026-10-09T21:51:07Z"
}
```

## Spend

```
{
 "tb": {
  "harness": "tb",
  "logged": {
   "llm": 34.51958,
   "decision_model": 0.121625,
   "total": 34.641205,
   "by_arm": {
    "baseline": 16.089775,
    "gated": 14.215454,
    "shadow": 4.122381,
    "smoke_gated": 0.213595
   },
   "calls": 2899,
   "decisions": 1351,
   "unpriced_calls": 0
  },
  "or_key_usage_start": "23.652674787",
  "or_key_usage_end": "27.722433124",
  "written": "2026-10-09T21:51:14Z"
 },
 "mswe": {
  "harness": "mswe",
  "logged": {
   "llm": 3.685591,
   "decision_model": 0.262544,
   "total": 3.948134,
   "by_arm": {
    "baseline": 1.163531,
    "gated": 1.408962,
    "shadow": 1.368979,
    "smoke_gated": 0.006663
   },
   "calls": 3986,
   "decisions": 2002,
   "unpriced_calls": 13
  },
  "or_key_usage_start": "23.652674787",
  "or_key_usage_end": "27.722433124",
  "written": "2026-10-09T21:53:46Z"
 },
 "swea": {
  "harness": "swea",
  "logged": {
   "llm": 0,
   "decision_model": 0,
   "total": 0,
   "by_arm": {},
   "calls": 0,
   "decisions": 0,
   "unpriced_calls": 0
  },
  "or_key_usage_start": "",
  "or_key_usage_end": "27.722433124",
  "written": "2026-10-09T21:52:29Z"
 },
 "total": 38.5893,
 "cap": 250.0,
 "e29_spent": 13.3822,
 "e29_plus_e29b": 51.9715,
 "total_cap": 300.0
}
```

## Leaks

```
{
 "tb": {
  "checks": 4,
  "max_files": 0,
  "stages": [
   "smoke",
   "cal",
   "pilot",
   "final"
  ]
 },
 "mswe": {
  "checks": 4,
  "max_files": 0,
  "stages": [
   "smoke",
   "cal",
   "pilot",
   "final"
  ]
 },
 "swea": {
  "checks": 1,
  "max_files": 0,
  "stages": [
   "final"
  ]
 }
}
```

