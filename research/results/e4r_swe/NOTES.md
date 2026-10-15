# E4r: E4's step routing on 200 new held-out SWE-bench Verified instances (exploratory, descriptive)

Registered in `docs/PREREG_A7_STAGE2.md`, Deviations 2026-10-07, in three steps:

1. "E4r: replicate and extension of E4" (`e4448cc`).
2. "E4r: route and host" (`f3291f9`).
3. "E4r: minimum spend" (`50e59d0`), which drops the replicate, caps spend at $40 and pools the 200 new instances with
   E4's 100.

All three were committed before any E4r episode. The execution-as-run entry follows them.

E4r is outside every Holm table and revises no registered result. Every number below comes from `summary.json`,
written by `scripts/explore_e4_swe.py --exp e4r readout`.

## Setup

E4r uses E4's evaluation unchanged:

- mini-SWE-agent 2.4.6, unmodified;
- deep path: DeepSeek V4 Pro with reasoning on; fast path: DeepSeek V4 Flash with reasoning off; both pinned to
  StreamLake with no fallback;
- Jev via OpenRouter decides before each step at τ = 0.5 (E4's registered value);
- caps per instance: $0.42 and 250 steps;
- k = 1, 24 workers in one task-major pool, local swebench 4.1.0.

**Instances.** The 200 SWE-bench Verified instances no earlier run had touched (`configs/explore/e4r_swe/`). The primary
view pools them with E4's own run on its 100 evaluation instances, giving 300 distinct held-out instances paired by
instance. All three arms of an instance come from the same run, on the same host type (m7i.8xlarge).

**Host and key.** The authors' AWS host `<instance-id>`, with one OpenRouter inference key (hard limit $150). The
registered cap for E4r was $40.

## Results

| view | arm | resolved | success % | $ / instance | mean s | Δ vs deep (pt) | wins / losses | cost saved % | time saved % |
|---|---|---|---|---|---|---|---|---|---|
| pooled 300 | always_deep | 232 | 77.3 | 0.0397 | 457 | – | – | – | – |
| pooled 300 | jevagent | 229 | 76.3 | 0.0355 | 436 | −1.0 [−4.0, +2.0] | 9 / 12 | 10.5 [0.4, 19.5] | 4.7 [−2.4, 11.1] |
| pooled 300 | always_fast | 227 | 75.7 | 0.0116 | 330 | −1.7 [−5.0, +1.7] | 10 / 15 | 70.7 [66.5, 74.3] | 27.9 [21.0, 34.1] |
| new 200 | always_deep | 151 | 75.5 | 0.0389 | 486 | – | – | – | – |
| new 200 | jevagent | 148 | 74.0 | 0.0370 | 476 | −1.5 [−5.0, +2.0] | 6 / 9 | 4.8 [−8.5, 16.9] | 2.0 [−6.9, 10.4] |
| new 200 | always_fast | 149 | 74.5 | 0.0120 | 359 | −1.0 [−5.0, +3.0] | 7 / 9 | 69.1 [63.4, 73.8] | 26.3 [17.4, 34.3] |

Intervals are paired task-bootstrap 95 % intervals (B = 10,000, seed 0). Tango's interval for routing on the pooled
300 is [−4.3, +2.2].

The paper's one-sided non-inferiority test at 3 points (`metrics.ni_certified_p`) was reported descriptively. It does
not hold for routing (p = 0.108 pooled, 0.240 on the new 200) or for the fast path (p = 0.229 pooled).

Routing sent 21.7 % of the steps on the new 200 to the fast path (E4: 23.5 %). Jev cost $0.0034 per instance. One
instance hit its limits in each of `always_deep` and `jevagent`. There were no infrastructure errors and no
unevaluated instance.

**Reading.** On 300 held-out instances, step routing kept success within the paired interval of the deep path (−1.0
points). It cost 10.5 % less and took 4.7 % less mean agent time. On the 200 new instances alone, the cost saving
shrank to 4.8 %, with an interval that includes no saving. E4's 21.3 % saving on its 100 instances is therefore the
favourable end. The fast path alone stayed within 2 points of the deep path's success at 71 % lower cost, as in E4.

## Execution

| when (UTC, 2026-10-07) | what |
|---|---|
| 15:20 | host started; guards armed |
| 15:34 | stages `rearm update smoke gate eval final` sent; code at `50e59d0`; key check: inference key, limit $150, usage $0 |
| 15:34–15:39 | smoke: `scikit-learn__scikit-learn-13496` × 3 arms, 0 errors, $0.0196 |
| 15:39 | gate: all five conditions held; decision full, k = 200, projection $25.29 against the $40 cap |
| 15:39–19:17 | agent runs: 600 jobs, 0 infrastructure errors (no re-run) |
| 19:17–19:50 | local SWE-bench evaluation; summary; leak check (0 files hold the key) |
| 19:50 | outputs synced to S3, host powered itself off |

**Spend:** $17.61 logged (smoke $0.02, eval $17.59), equal to the key's usage of $17.61. The watchdog never fired.

## Files

- `summary.md` and `summary.json`: the readout.
- `per_instance.csv`: E4r's 600 runs.
- `box/eval/summary/rows.json`: the rows the readout reads.
- `box/{eval,smoke}/<arm>/`: SWE-bench reports and arm configs.
- `box/status/`: driver status, gate and spend.
- `paper/generated/e4r_swe.tex`: keys `e4r/...`.
- `s3://<bucket>/e4r/runs/`: trajectories and full records (not in git).
