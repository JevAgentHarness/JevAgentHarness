# E29: the full harness in open-source harnesses (exploratory, descriptive)

Registered in docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 ~16:05 UTC, E29 before any E29 code, call or episode. The decision layer is a host-side model proxy (`jevharness.adapters.route_proxy`) that holds the keys: a typed fast/strong question to Jev asked while the previous action runs, a gate fitted on shadow logs, sticky fast segments of at most 4 steps, fail-open. Cost = proxy-logged LLM + decision-model $ per task; time = agent execution seconds per task.

## Pilot (30 tasks per harness; decided the go)

| harness | n | success base / gated % | delta pt [95 %] | W / L | NI p (3 pt) | cost $/task base / gated | saving % [95 %] | mean s base / gated | time saving % | median-time saving % | fast share % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pi on Terminal-Bench 2.0 | -- | the gate kept every step on the strong LLM | | | | | | | | | |
| mini-SWE-agent on SWE-bench Verified | 30 | 70.0 / 70.0 | 0.0 [-13.3, +13.3] | 2 / 2 | 0.577 (not shown) | 0.0514 / 0.0459 | 10.7 [-62.9, 42.2] | 543 / 561 | -3.3 | -11.2 | 0.7 |

- mini-SWE-agent on SWE-bench Verified: gated arm 42.5 requests per task (baseline 44.7), 0.2 fast segments per task of mean length 1.50 (0.0 % ended by the cap k = 4), 0.4 switches per task; decision model 42.3 calls, $0.0057 and 8.1 s per task, of which requests waited 1.06 s; fail-open steps 0, decision errors 0; cached input share 91.5 % (baseline) vs 91.3 % (gated); infrastructure errors 0 / 0; complete case n 30, delta 0.0, NI p 0.577. Missing: {"baseline": 0, "gated": 0}.

## Fitted gates (calibration, shadow mode)

| harness | tau | steps / tasks | agree fast / self % | routed: agree fast / self % | simulated fast share % | dC $ |
|---|---|---|---|---|---|---|
| tb | none | 187 / 10 | 36.9 / 60.4 | -- / -- | 0.0 | 0.01214 |
| mswe | 0.99 | 735 / 20 | 7.8 / 15.1 | 44.4 / 33.3 | 2.4 | 0.00087 |

Post hoc, descriptive (from the same calibration records; not part of the registered fit): agreement with the strong LLM's action of the fast LLM / of a second strong sample.

| harness | steps | exact action % (fast / self) | same tool % (fast / self) | Jev says fast: n, exact % (fast / self) | Jev says strong: n, exact % (fast / self) | AUROC of c for fast = strong | share c >= 0.5 % |
|---|---|---|---|---|---|---|---|
| tb | 187 | 36.9 / 60.4 | 77.0 / 96.3 | 123, 40.7 / 65.0 | 64, 29.7 / 51.6 | 0.52 | 65.8 |

- tb by the strong action's tool: bash n 106: 26.4 / 39.6; read n 46: 63.0 / 89.1; write n 21: 47.6 / 85.7; none n 10: 10.0 / 90.0; edit n 4: 25.0 / 75.0
| mswe | 735 | 7.8 / 15.1 | 99.6 / 99.0 | 535, 8.8 / 15.9 | 200, 5.0 / 13.0 | 0.70 | 72.8 |

- mswe by the strong action's tool: bash n 735: 7.8 / 15.1

## Go decision (registered rule, pilot only)

```
{
 "decision": "no-go",
 "per_harness": {
  "tb": {
   "n": 0,
   "saving": 0.0,
   "drop": 99,
   "solved_base": null,
   "solved_gated": null,
   "pass": false,
   "not_worse": false,
   "no_gate": true
  },
  "mswe": {
   "n": 30,
   "saving": 0.10720800549442366,
   "drop": 0,
   "solved_base": 21,
   "solved_gated": 21,
   "pass": false,
   "not_worse": true,
   "no_gate": false
  }
 },
 "rule": "GO iff pass(tb) and pass(mswe), or pass of one and not_worse of the other; pass = saving >= 20 % and at most 3 tasks of 30 lost; not_worse = saving >= the reference row's (pi 9.6 %, mini-SWE-agent 10.5 %) and at most 3 tasks lost",
 "registered": "docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 ~16:05 UTC, E29",
 "written": "2026-10-09T18:53:37Z"
}
```

## Spend

```
{
 "tb": {
  "harness": "tb",
  "logged": {
   "llm": 6.998803,
   "decision_model": 0.025783,
   "total": 7.024586,
   "by_arm": {
    "baseline": 0.148418,
    "shadow": 6.704068,
    "smoke_gated": 0.1721
   },
   "calls": 778,
   "decisions": 279,
   "unpriced_calls": 0
  },
  "or_key_usage_start": "18.286930267",
  "or_key_usage_end": "23.652674787",
  "written": "2026-10-09T18:53:43Z"
 },
 "mswe": {
  "harness": "mswe",
  "logged": {
   "llm": 5.061621,
   "decision_model": 0.274013,
   "total": 5.335635,
   "by_arm": {
    "baseline": 1.553128,
    "gated": 1.376086,
    "shadow": 2.397741,
    "smoke_gated": 0.008681
   },
   "calls": 5016,
   "decisions": 2064,
   "unpriced_calls": 13
  },
  "or_key_usage_start": "18.314529979",
  "or_key_usage_end": "23.652674787",
  "written": "2026-10-09T19:03:34Z"
 },
 "swea": {
  "harness": "swea",
  "logged": {
   "llm": 0.986849,
   "decision_model": 0.004329,
   "total": 0.991178,
   "by_arm": {
    "baseline": 0.284034,
    "shadow": 0.634945,
    "smoke_gated": 0.072199
   },
   "calls": 185,
   "decisions": 70,
   "unpriced_calls": 0
  },
  "or_key_usage_start": "18.286930267",
  "or_key_usage_end": "23.652674787",
  "written": "2026-10-09T18:53:22Z"
 },
 "local_smoke": {
  "logged": {
   "llm": 0.030435,
   "decision_model": 0.000321,
   "total": 0.030756
  },
  "what": "local development smoke of the proxy with pi 0.99.2 and litellm against Gemini (Jev via the author Jev-only key, decision/jev.py), 2026-10-09 16:27-16:31 UTC; development only, never reported as a result"
 },
 "total": 13.3822,
 "cap": 300.0
}
```

## Leaks

```
{
 "tb": {
  "checks": 3,
  "max_files": 0,
  "stages": [
   "smoke",
   "cal",
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
  "checks": 2,
  "max_files": 0,
  "stages": [
   "smoke",
   "final"
  ]
 }
}
```

