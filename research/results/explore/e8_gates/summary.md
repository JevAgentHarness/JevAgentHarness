# E8: what decides whether the decision model's answer is accepted (exploratory, descriptive)

Registered: docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "Exploratory row E8". Outside both stages; no test; revises no registered result, selects nothing and changes no threshold.

- Data: `runs/e2a_jevgym_shadow_cal/jev_shadow_cal.jsonl` (sha256 `fff21516e0aad243…`), the registered calibration shadow arm: JevGym calibration seeds 5000–5149, Gemini 3.5 Flash as the agent, Jev answering in shadow at every state. Sample: 1000 decisions from 150 tasks (500 per decision point, seed 0, stratified in proportion by whether the decision model is right).
- Spend: $2.49 logged (G3 $0.05, G4 $2.44); cap $40.
- Saving = the best expected saving per decision against the LLM alone over the gate's thresholds, on these decisions (an in-sample upper bound), V = $1 per wrong decision, with the decision model's own call and the gate's call charged; in % of the LLM's cost of the decision. Latency = critical-path time saved per decision at that threshold (negative = slower). 'If free' = the same optimum with the gate's call not charged (its ranking alone). 'Held-out' (post hoc) = the threshold chosen on half of the tasks and scored on the other half, both ways, mean of 10 random halvings: the in-sample optimum favours gates with many distinct scores.

## tool choice (n = 500, 30 wrong answers of the decision model; its accuracy 94.0 %, the LLM's 100.0 %)

LLM cost of the decision ΔC = $0.00571 (1160 ms); decision-model call κ = $0.000115 (224 ms); G3 call $0.000048 (203 ms); G4 call $0.00329 (1781 ms) = 0.58 × ΔC (medians: LLM 938 ms, model 208 ms, G3 189 ms, G4 1010 ms). Registered τ̂ = 1 (32.8 % of decisions at or above it).

| gate | extra call | AUROC [95 % CI] | Δ vs G1 [95 % CI] | sel. acc. @25/50/75/90 % | saving % of ΔC (USD) | coverage | latency ms | saving if free | held-out saving |
|---|---|---|---|---|---|---|---|---|---|
| G1 model's confidence (registered) | none | 0.735 [0.655, 0.808] | – | 100.0/97.8/95.5/95.3 | +28.3 % ($+0.00161) | 32.8 % | +170 | +28.3 % | -1.2 % |
| G1 at the registered tau-hat (fixed, the deployed rule) | none | 0.735 [0.655, 0.808] | – | 100.0/97.8/95.5/95.3 | +28.3 % ($+0.00161) | 32.8 % | +170 | +28.3 % | +28.3 % |
| G2 top-two margin | none | 0.730 [0.647, 0.805] | -0.005 [-0.021, +0.010] | 100.0/97.4/95.5/95.3 | +33.4 % ($+0.00191) | 38.4 % | +224 | +33.4 % | -24.5 % |
| G3 second decision-model call (yes/no) | 1 decision-model call | 0.734 [0.643, 0.816] | -0.001 [-0.040, +0.036] | 100.0/96.8/95.7/96.2 | +25.4 % ($+0.00145) | 31.2 % | -35 | +26.3 % | -27.2 % |
| G4 LLM verifier (Flash, yes/no + 0-100) | 1 LLM call | 0.500 [0.500, 0.500] | -0.235 [-0.308, -0.155] | 94.0/94.0/94.0/94.0 | -59.6 % ($-0.00340) | 0.0 % | -2005 | -2.0 % | -59.6 % |
| G4v LLM verifier, its verdict as the score (post hoc) | 1 LLM call | 0.689 [0.597, 0.776] | -0.046 [-0.094, +0.001] | 96.3/96.3/96.3/96.3 | -59.6 % ($-0.00340) | 0.0 % | -2005 | -2.0 % | -59.6 % |
| G5 agreement with the LLM's own choice | the agent's own LLM call | 0.929 [0.909, 0.947] | +0.193 [+0.117, +0.274] | 100.0/100.0/100.0/96.8 | -2.0 % ($-0.00012) | 0.0 % | +0 | -2.0 % | -2.0 % |
| G1, then G3 below tau-hat | 1 decision-model call below tau-hat | 0.748 [0.664, 0.823] | +0.013 [-0.022, +0.047] | 100.0/97.3/95.7/96.2 | +33.3 % ($+0.00190) | 40.2 % | +128 | +33.8 % | -20.8 % |
| G1, then G4 below tau-hat | 1 LLM call below tau-hat | 0.674 [0.651, 0.696] | -0.061 [-0.132, +0.014] | 100.0/96.9/95.0/94.3 | -8.1 % ($-0.00046) | 32.8 % | -1108 | +28.3 % | -8.1 % |
| G1, then G4v below tau-hat (post hoc) | 1 LLM call below tau-hat | 0.788 [0.723, 0.847] | +0.053 [+0.018, +0.091] | 100.0/98.0/96.8/96.4 | -8.1 % ($-0.00046) | 32.8 % | -1108 | +28.3 % | -8.1 % |
| answer key (reference, not a gate) | none | 1.000 [–, –] | – | 100.0/100.0/100.0/100.0 | +91.3 % ($+0.00521) | 94.0 % | +862 | +91.3 % | +91.3 % |

Saving (% of ΔC) by the value of a wrong decision V:

| gate | V = $0.1 | V = $1 | V = $10 |
|---|---|---|---|
| G1 model's confidence (registered) | +30.2 % (cov. 46 %) | +28.3 % (cov. 33 %) | +28.3 % (cov. 33 %) |
| G1 at the registered tau-hat (fixed, the deployed rule) | +28.3 % (cov. 33 %) | +28.3 % (cov. 33 %) | +28.3 % (cov. 33 %) |
| G2 top-two margin | +33.4 % (cov. 38 %) | +33.4 % (cov. 38 %) | +33.4 % (cov. 38 %) |
| G3 second decision-model call (yes/no) | +27.5 % (cov. 91 %) | +25.4 % (cov. 31 %) | +25.4 % (cov. 31 %) |
| G4 LLM verifier (Flash, yes/no + 0-100) | -59.6 % (cov. 0 %) | -59.6 % (cov. 0 %) | -59.6 % (cov. 0 %) |
| G4v LLM verifier, its verdict as the score (post hoc) | -29.0 % (cov. 92 %) | -59.6 % (cov. 0 %) | -59.6 % (cov. 0 %) |
| G5 agreement with the LLM's own choice | -2.0 % (cov. 0 %) | -2.0 % (cov. 0 %) | -2.0 % (cov. 0 %) |
| G1, then G3 below tau-hat | +33.3 % (cov. 40 %) | +33.3 % (cov. 40 %) | +33.3 % (cov. 40 %) |
| G1, then G4 below tau-hat | -8.1 % (cov. 33 %) | -8.1 % (cov. 33 %) | -8.1 % (cov. 33 %) |
| G1, then G4v below tau-hat (post hoc) | -7.8 % (cov. 92 %) | -8.1 % (cov. 33 %) | -8.1 % (cov. 33 %) |
| answer key (reference, not a gate) | +91.3 % (cov. 94 %) | +91.3 % (cov. 94 %) | +91.3 % (cov. 94 %) |

G4 replies: 461 yes, 39 no, 0 without a verdict; confidence 100 in 500 of 500; 39 'no' with confidence ≥ 50 and 0 'yes' below 50; AUROC of the verdict alone 0.689.

## tool arguments (n = 500, 22 wrong answers of the decision model; its accuracy 95.6 %, the LLM's 100.0 %)

LLM cost of the decision ΔC = $0.00338 (696 ms); decision-model call κ = $0.000059 (224 ms); G3 call $0.000051 (206 ms); G4 call $0.00159 (1658 ms) = 0.47 × ΔC (medians: LLM 491 ms, model 208 ms, G3 188 ms, G4 724 ms). Registered τ̂ = 0.99 (72.6 % of decisions at or above it).

| gate | extra call | AUROC [95 % CI] | Δ vs G1 [95 % CI] | sel. acc. @25/50/75/90 % | saving % of ΔC (USD) | coverage | latency ms | saving if free | held-out saving |
|---|---|---|---|---|---|---|---|---|---|
| G1 model's confidence (registered) | none | 0.885 [0.850, 0.917] | – | 100.0/100.0/99.5/97.1 | +59.1 % ($+0.00199) | 72.6 % | +208 | +59.1 % | +1.1 % |
| G1 at the registered tau-hat (fixed, the deployed rule) | none | 0.885 [0.850, 0.917] | – | 100.0/100.0/99.5/97.1 | +59.1 % ($+0.00199) | 72.6 % | +208 | +59.1 % | +59.1 % |
| G2 top-two margin | none | 0.883 [0.849, 0.913] | -0.003 [-0.010, +0.005] | 100.0/100.0/99.2/96.9 | +61.9 % ($+0.00209) | 74.2 % | +225 | +61.9 % | +9.8 % |
| G3 second decision-model call (yes/no) | 1 decision-model call | 0.833 [0.781, 0.878] | -0.052 [-0.098, -0.015] | 100.0/100.0/97.7/96.7 | +62.9 % ($+0.00212) | 68.0 % | +52 | +64.4 % | +45.1 % |
| G4 LLM verifier (Flash, yes/no + 0-100) | 1 LLM call | 0.500 [0.500, 0.500] | -0.385 [-0.417, -0.350] | 95.6/95.6/95.6/95.6 | -49.0 % ($-0.00165) | 0.0 % | -1881 | -1.8 % | -49.0 % |
| G4v LLM verifier, its verdict as the score (post hoc) | 1 LLM call | 0.921 [0.845, 0.989] | +0.036 [-0.047, +0.110] | 99.4/99.4/99.4/99.4 | -49.0 % ($-0.00165) | 0.0 % | -1881 | -1.8 % | -75.4 % |
| G5 agreement with the LLM's own choice | the agent's own LLM call | 1.000 [1.000, 1.000] | +0.115 [+0.083, +0.150] | 100.0/100.0/100.0/100.0 | -1.8 % ($-0.00006) | 0.0 % | +0 | -1.8 % | -1.8 % |
| G1, then G3 below tau-hat | 1 decision-model call below tau-hat | 0.942 [0.917, 0.963] | +0.057 [+0.028, +0.084] | 100.0/100.0/100.0/98.0 | +81.1 % ($+0.00274) | 87.6 % | +310 | +81.4 % | +80.7 % |
| G1, then G4 below tau-hat | 1 LLM call below tau-hat | 0.880 [0.863, 0.897] | -0.005 [-0.034, +0.022] | 100.0/100.0/99.5/96.9 | +47.7 % ($+0.00161) | 72.6 % | -246 | +59.1 % | +47.7 % |
| G1, then G4v below tau-hat (post hoc) | 1 LLM call below tau-hat | 0.973 [0.953, 0.990] | +0.088 [+0.051, +0.124] | 100.0/100.0/99.9/99.5 | +47.7 % ($+0.00161) | 72.6 % | -246 | +59.1 % | +15.4 % |
| answer key (reference, not a gate) | none | 1.000 [–, –] | – | 100.0/100.0/100.0/100.0 | +95.0 % ($+0.00321) | 95.6 % | +448 | +95.0 % | +95.0 % |

Saving (% of ΔC) by the value of a wrong decision V:

| gate | V = $0.1 | V = $1 | V = $10 |
|---|---|---|---|
| G1 model's confidence (registered) | +59.1 % (cov. 73 %) | +59.1 % (cov. 73 %) | +59.1 % (cov. 73 %) |
| G1 at the registered tau-hat (fixed, the deployed rule) | +59.1 % (cov. 73 %) | +59.1 % (cov. 73 %) | +59.1 % (cov. 73 %) |
| G2 top-two margin | +61.9 % (cov. 74 %) | +61.9 % (cov. 74 %) | +61.9 % (cov. 74 %) |
| G3 second decision-model call (yes/no) | +62.9 % (cov. 68 %) | +62.9 % (cov. 68 %) | +62.9 % (cov. 68 %) |
| G4 LLM verifier (Flash, yes/no + 0-100) | -49.0 % (cov. 0 %) | -49.0 % (cov. 0 %) | -49.0 % (cov. 0 %) |
| G4v LLM verifier, its verdict as the score (post hoc) | +26.7 % (cov. 94 %) | -49.0 % (cov. 0 %) | -49.0 % (cov. 0 %) |
| G5 agreement with the LLM's own choice | -1.8 % (cov. 0 %) | -1.8 % (cov. 0 %) | -1.8 % (cov. 0 %) |
| G1, then G3 below tau-hat | +81.1 % (cov. 88 %) | +81.1 % (cov. 88 %) | +81.1 % (cov. 88 %) |
| G1, then G4 below tau-hat | +47.7 % (cov. 73 %) | +47.7 % (cov. 73 %) | +47.7 % (cov. 73 %) |
| G1, then G4v below tau-hat (post hoc) | +62.6 % (cov. 94 %) | +47.7 % (cov. 73 %) | +47.7 % (cov. 73 %) |
| answer key (reference, not a gate) | +95.0 % (cov. 96 %) | +95.0 % (cov. 96 %) | +95.0 % (cov. 96 %) |

G4 replies: 471 yes, 29 no, 0 without a verdict; confidence 100 in 500 of 500; 29 'no' with confidence ≥ 50 and 0 'yes' below 50; AUROC of the verdict alone 0.921.

## Reading

- Tool choice: AUROC G1 0.735, G1@tau 0.735, G2 0.730, G3 0.734, G4 0.500, G4v 0.689, G5 0.929, G1>G3 0.748, G1>G4 0.674, G1>G4v 0.788. Best saving per decision (V = $1): G1 +28.3 %, G1@tau +28.3 %, G2 +33.4 %, G3 +25.4 % (free: +26.3 %), G4 -59.6 %, G4v -59.6 % (free: -2.0 %), G5 -2.0 %, G1>G3 +33.3 %, G1>G4 -8.1 %, G1>G4v -8.1 % (free: +28.3 %); held out: G1 -1.2 %, G1@tau +28.3 %, G2 -24.5 %, G3 -27.2 %, G4 -59.6 %, G4v -59.6 %, G5 -2.0 %, G1>G3 -20.8 %, G1>G4 -8.1 %, G1>G4v -8.1 %; a perfect free gate +91.3 %. Highest: G2 top-two margin. One G4 call costs 0.58 × the LLM's cost of the decision; G4's 0-100 confidence was 100 in 500 of 500 replies.
- Tool arguments: AUROC G1 0.885, G1@tau 0.885, G2 0.883, G3 0.833, G4 0.500, G4v 0.921, G5 1.000, G1>G3 0.942, G1>G4 0.880, G1>G4v 0.973. Best saving per decision (V = $1): G1 +59.1 %, G1@tau +59.1 %, G2 +61.9 %, G3 +62.9 % (free: +64.4 %), G4 -49.0 %, G4v -49.0 % (free: -1.8 %), G5 -1.8 %, G1>G3 +81.1 %, G1>G4 +47.7 %, G1>G4v +47.7 % (free: +59.1 %); held out: G1 +1.1 %, G1@tau +59.1 %, G2 +9.8 %, G3 +45.1 %, G4 -49.0 %, G4v -75.4 %, G5 -1.8 %, G1>G3 +80.7 %, G1>G4 +47.7 %, G1>G4v +15.4 %; a perfect free gate +95.0 %. Highest: G1, then G3 below tau-hat. One G4 call costs 0.47 × the LLM's cost of the decision; G4's 0-100 confidence was 100 in 500 of 500 replies.

Notes: AUROC with ties counted ½; selective accuracy resolves ties in expectation (`metrics.selective_accuracy`); intervals are a task-level bootstrap (B = 2,000, seed 0). ΔC and κ are the logged per-decision costs of the registered fit (argument decisions share their step's LLM call); G3 and G4 costs and latencies come from E8's own ledger (`calls_g3.jsonl.gz`, `calls_g4.jsonl.gz`). G5 needs the agent's own LLM call on every decision, so accepting saves nothing; its decision-model call is taken to run beside the LLM (no added time). Two-stage rows: the decision model's confidence accepts at the registered τ̂; the second gate is asked (and paid) only below it, so their baseline is G1 at the registered τ̂ (fitted on the whole calibration log, which contains these decisions: no threshold is chosen here, so its held-out value equals its in-sample one). G4v and G1>G4v are post hoc (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E8: the verifier's verdict as its score"): every G4 reply gave confidence 100 whatever its verdict, so the registered G4 score is constant; G4v scores the verdict (yes 1, no 0) from the same replies, no new call. Latencies of G3 and G4 were measured on 2026-10-07, those of ΔC and κ in the calibration run (2026-10-01): compare them with care.
