# E28: our recipe on StartLux-Decision-0.8B

_exploratory (E28), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-09 'E28: our recipe on StartLux-Decision-0.8B'; outside every Holm table_

Generated 2026-10-09T20:04:51+00:00 (git b697f555907e). Base: startlux-models/StartLux-Decision-0.8B @ bd4f76a600e2 (CC BY-NC 4.0; the trained adapter is a derivative under the same terms).

Vocabulary disjointness: overlap 0 (augmentation 193 words, sha256 1e86eb005297ec70; E21 1011 words, sha256 0012a84bf8c961b1).

Data: 7000 agent-state rows (3463 augmented on 2014 of 4063 states) + 7000 public items = 14000 rows (sha256 2e5170527c081033).

Checkpoint selection (development data only): {"epoch": 2, "path": "/opt/jev/ck_e28/train/checkpoint-398", "acc": {"e1_devprefix": 0.8255, "dev": 0.99453}, "mean": 0.910015}.

| row | L1 | L1 tool choice | $ / 1k (E9, own stack) | L2 C | L2 shift all | must-ask | surface cost | L3 shift (n) |
|---|---|---|---|---|---|---|---|---|
| StartLux-0.8B (mixed), E28 | 80.5 | 87.1 | 0.1786 | 98.5 | 95.6 | 93.5 | 1.2 | 82.0 (200) |
| StartLux-0.8B (untrained) | 70.0 | 88.2 | 0.0127 | 48.3 | 51.7 | 27.2 | 5.7 | 13.0 (200) |
| E27's SP (Qwen3.5-0.8B head) | 77.2 | 83.2 | 0.0910 | 99.2 | 96.3 | 93.9 | -1.7 | 80.0 (200) |
| StartLux-4B (mixed), E12c | 85.5 | 89.5 | 0.2946 | 99.6 | 98.5 | 100.0 | 1.4 | 98.0 (200) |
| AgentDecision-4B (E12) | 85.1 | 89.3 | 0.1397 | – | 94.9 | 67.4 | 3.7 | 91.5 (200) |

L1 cost bases: **e28/l1/usdk and e9/e28/usdk = the vendor's server on the merged checkpoint** (the basis of e9/sl0.8b): $0.0101 per 1,000 decisions, median 18 ms (accuracy of that run, a check only: 80.8 %); e28/l1/usdk_own = E28's own stack (serve_jev): $0.1786, median 456 ms. L1 accuracy (e28/l1/acc, e9/e28/acc) is the own stack's.

## StartLux-0.8B (mixed) in the agent

- L3 in-domain REPLACE: 100.0 % (n 200)
- final_in: 100.0 % vs 99.5 % LLM-only (n 200), NI non-inferior (p 0.002); saving incl. GPU -6.0 %; wall 14.9 s vs 13.6 s; tool coverage 35.9 %, args 66.9 %
  - final_in time split (s per episode): model 2.4, LLM 12.4 (6.4 calls, 1.93 s each); LLM-only reference: LLM 11.7 (7.5 calls, 1.55 s each)
- final_shift: 98.0 % vs 100.0 % LLM-only (n 200), NI not shown (p 0.306); saving incl. GPU 13.2 %; wall 23.8 s vs 8.0 s; tool coverage 61.7 %, args 31.9 %
  - final_shift time split (s per episode): model 2.2, LLM 20.9 (6.0 calls, 3.51 s each); LLM-only reference: LLM 7.9 (6.6 calls, 1.20 s each)

## Readings (registration (i))

```
{
 "l2_holds": true,
 "l3_holds": false,
 "holds_under_shift": false,
 "keeps_c": false,
 "final_in_ni": "non-inferior",
 "keeps_in_domain": false,
 "rule": "registration (i) (= E27 (i)): holds under shift iff L2-shift accuracy >= AgentDecision-4B's (E12) - 3 points and the lower end of the paired Newcombe interval of L3-shift success minus E12's REPLACE row >= -5 points; keeps its in-domain level iff C keeps the frozen 4B's by E5's rule and (if run) the in-domain final harness is non-inferior at 3 points"
}
```

Spend: {"gemini_runs": 26.64379305, "gemini_probes": 0.002799999999999999, "gemini": 26.64659305, "cap": 30.0, "runs": {"e28_final_in": {"gemini": 7.795539299999999, "records": 200}, "e28_final_shift": {"gemini": 7.152599100000001, "records": 200}, "e28_live_in": {"gemini": 2.8960770000000005, "records": 200}, "e28_live_shift": {"gemini": 3.3809789999999986, "records": 200}, "e28_live_shift_base": {"gemini": 5.136715199999999, "records": 200}, "e28_smoke_final_in": {"gemini": 0.0757641, "records": 2}, "e28_smoke_final_shift": {"gemini": 0.07472835, "records": 2}, "e28_smoke_in": {"gemini": 0.030078, "records": 2}, "e28_smoke_shift": {"gemini": 0.012621, "records": 2}, "e28_smoke_shift_base": {"gemini": 0.088692, "records": 2}}}

