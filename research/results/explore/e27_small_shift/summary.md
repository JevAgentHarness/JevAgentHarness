# E27: a 0.8B decision model that holds under shift

_exploratory (E27), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-09 'E27: a 0.8B decision model that holds under shift'; outside every Holm table_

Generated 2026-10-09T15:17:11+00:00 (git 5663845208bc).

Vocabulary disjointness: overlap 0 (augmentation 193 words, sha256 1e86eb005297ec70; E21 1011 words, sha256 0012a84bf8c961b1).

Selected: **SP** (L2 shift).

| row | L1 | L2 C | L2 shift all | must-ask | surface cost | L3 shift (n) |
|---|---|---|---|---|---|---|
| 0.8B + surface augmentation (S) | 67.7 | 98.8 | 93.7 | 77.2 | 1.6 | 72.0 (200) |
| 0.8B + surface augmentation + public data (SP) | 77.2 | 99.2 | 96.3 | 93.9 | -1.7 | 80.0 (200) |
| 0.8B head (E14 arm A) | 67.8 | 99.2 | 73.2 | 77.2 | 21.9 | 17.5 (200) |
| AgentDecision-4B (E12) | 85.1 | 99.1 | 94.9 | 67.4 | 3.7 | 91.5 (200) |
| Checkpoint before public data | 78.8 | 99.2 | 98.6 | 90.8 | 1.1 | 96.0 (200) |

## AgentDecision-0.8B

- L3 in-domain REPLACE: 100.0 % (n 200)
- final_in: 99.0 % vs 99.5 % LLM-only (n 200), NI not shown (p 0.059); saving incl. GPU 30.3 %; wall 6.5 s vs 13.6 s; tool coverage 48.5 %, args 89.0 %
- final_shift: 100.0 % vs 100.0 % LLM-only (n 200), NI non-inferior (p 0.002); saving incl. GPU 6.0 %; wall 8.1 s vs 8.0 s; tool coverage 27.7 %, args 71.3 %

## Readings (registration (i))

```
{
 "selected": "SP",
 "l2_holds": true,
 "l3_holds": false,
 "holds_under_shift": false,
 "keeps_c": true,
 "final_in_ni": "not shown",
 "keeps_in_domain": false,
 "rule": "registration (i): holds under shift iff L2-shift accuracy >= AgentDecision-4B's (E12) - 3 points and the lower end of the paired Newcombe interval of L3-shift success minus E12's REPLACE row >= -5 points; keeps its in-domain level iff C keeps E14 arm A's by E5's rule and (if run) the in-domain final harness is non-inferior at 3 points"
}
```

Spend: {"gemini_runs": 24.070245749999998, "gemini_probes": 0.002799999999999999, "gemini": 24.07304575, "cap": 30.0, "runs": {"e27_final_in": {"gemini": 5.4246984, "records": 200}, "e27_final_shift": {"gemini": 9.327494549999997, "records": 200}, "e27_live_in": {"gemini": 2.896203, "records": 200}, "e27_live_shift_s": {"gemini": 2.9731259999999993, "records": 200}, "e27_live_shift_sp": {"gemini": 3.210596999999999, "records": 200}, "e27_smoke_final_in": {"gemini": 0.055241399999999996, "records": 2}, "e27_smoke_final_shift": {"gemini": 0.0665634, "records": 2}, "e27_smoke_in": {"gemini": 0.028540500000000003, "records": 2}, "e27_smoke_shift_s": {"gemini": 0.0448485, "records": 2}, "e27_smoke_shift_sp": {"gemini": 0.042933, "records": 2}}}

