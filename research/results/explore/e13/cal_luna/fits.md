# E13 calibration fit, GPT-6-Luna (DecisionGym-Hard as E7e; calibration seeds 5400+, n = 100)

Generated 2026-10-07T14:56:04+00:00; run `runs/e13_luna_cal`; estimator: threshold_theory --calibration-arm --task-value 1.0 --lam-per-hour 0 --v-fallback W (A6.5 / A6.11).

| calibration arm | decision point | tau-hat | rule | paired | V | dC | coverage at tau % | model acc. taken % | LLM acc. taken % | model acc. all % | LLM acc. all % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `jev_shadow_cal` | tool_select | 0.49 | projection | 2790 | 0.7543859649122807 | 9.007662365591399e-05 | 100.0 | 99.9 | 98.5 | 99.9 | 98.5 |
| `jev_shadow_cal` | tool_params | 0.61 | projection | 4469 | 1.0 | 5.311432535242784e-05 | 90.2 | 100.0 | 100.0 | 97.2 | 100.0 |

Cancel-vs-ship decisions (tool choice where the gold next tool is a write), at the fitted tau:

- `jev_shadow_cal`: 911 decisions, coverage 100.0 %, model acc. taken 100.0 %, model acc. all 100.0 %, LLM acc. all 95.3 %
