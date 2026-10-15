# E7c calibration fit (JevGym-Hard, 12 customers; calibration seeds 5300-5399)

Generated 2026-10-06T19:47:44+00:00; run `runs/e7c_cal`; estimator: threshold_theory --calibration-arm --task-value 1.0 --lam-per-hour 0 --v-fallback W (A6.5 / A6.11).

| calibration arm | decision point | tau-hat | rule | paired | V | dC | coverage at tau % | model acc. taken % | LLM acc. taken % | model acc. all % | LLM acc. all % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `jev_shadow_cal` | tool_select | 0.91 | projection | 3699 | 1.0 | 0.001991415486486487 | 90.1 | 100.0 | 100.0 | 99.8 | 100.0 |
| `jev_shadow_cal` | tool_params | 0.87 | projection | 5998 | 1.0 | 0.001136279225 | 80.2 | 100.0 | 100.0 | 97.4 | 100.0 |
| `rl_shadow_cal` | tool_select | always defer | projection | 3561 | 1.0 | 0.001839197851727043 | 0.0 | -- | -- | 86.5 | 100.0 |
| `rl_shadow_cal` | tool_params | 1 | projection | 5781 | 1.0 | 0.001052616554229372 | 62.9 | 100.0 | 100.0 | 94.8 | 100.0 |

**Under the registered estimator, the procedure turns the frozen JevAgentRL-4B's tool choice off on this setting**: its tool_select fit accepts no confidence level (always defer), so the refitted 4B profile leaves tool choice to the LLM (its tool_params threshold is reached only inside System-1-chosen calls).

Cancel-vs-ship decisions (tool choice where the gold next tool is a write), at the fitted tau:

- `jev_shadow_cal`: 1200 decisions, coverage 99.5 %, model acc. taken 100.0 %, model acc. all 100.0 %, LLM acc. all 100.0 %
- `rl_shadow_cal`: 1153 decisions, coverage 0.0 %, model acc. taken -- %, model acc. all 81.1 %, LLM acc. all 100.0 %
