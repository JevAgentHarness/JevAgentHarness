# Stage 2, the optional `gpt-6.1-sol` row — descriptive (A7.12 (i)): no test, outside every Holm table

Status: **complete**. System-2 `gpt-6.1-sol` (reasoning effort low, its lowest), small model `gpt-6-luna` (effort none), OpenAI Responses API; Jev via OpenRouter as System-1; fresh Stage-2 tasks.

Procedure: eligibility True (n = 500; envelope $80.0, predicted {"500": 23.33, "300": 15.78}).
Calibration (seeds 5150–5299): tool_select: no accepted level; tool_params τ̂* 0.57.
Derived profile `configs/harness/jevagent_best_sol.yaml` (selected): {"tool_select": {"mode": "cascade", "tau": 0.99, "always_defer": true}, "tool_params": {"mode": "cascade", "tau": 0.57}, "reflection": {"mode": "trigger", "tau": 0.5}}.
Spend so far (ledgers + cache-write surcharge, probe and smoke included): $14.89.

## Arms

| arm | n | success [Wilson 95 %] | $ / episode | $ incl. cache writes | latency s | wall s | LLM calls | S1 calls | input tok | cached | reasoning tok | served model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_only_sol | 500 / 500 | 99.0 [97.7, 99.6] | 0.009272 | 0.01049 | 19.1 | 19.2 | 7.5 | 0 | 25921 | 90.5 % | 0 | gpt-6.1-sol |
| jevagent_best_sol | 500 / 500 | 98.4 [96.9, 99.2] | 0.01111 | 0.0123 | 23.3 | 23.4 | 8.43 | 6.5 | 26808 | 89.5 % | 0 | gpt-6.1-sol |
| llm_only_solsmall | 500 / 500 | 99.8 [98.9, 100.0] | 0.0007168 | 0.0008127 | 12.8 | 12.8 | 7.5 | 0 | 25939 | 85.1 % | 0 | gpt-6-luna |

## Against `llm_only_sol` (descriptive; no test)

| row | n′ | success arm / ref | Δ success [paired bootstrap 95 %] | Newcombe 95 % | wins / losses | savings [95 %] | savings incl. cache writes | Δ latency s | Δ wall s |
|---|---|---|---|---|---|---|---|---|---|
| jevagent_best_sol | 500 | 98.4 / 99.0 | -0.6 [-2.0, +0.6] | [-2.2, +0.9] | 4 / 7 | -19.8 % [-25.6, -14.2] | -17.3 % [-23.7, -11.2] | +4.15 | +4.21 |
| llm_only_solsmall | 500 | 99.8 / 99.0 | +0.8 [-0.2, +1.8] | [-0.3, +2.1] | 5 / 1 | 92.3 % [+92.1, +92.5] | 92.3 % [+92.0, +92.5] | -6.35 | -6.35 |

## Intervention reporting (A7.16 (d); descriptive)

| arm | n | tool cov. | arg cov. | S2 calls / ep. | calls avoided | trigger questions | fires | fire rate | tasks with a fire |
|---|---|---|---|---|---|---|---|---|---|
| llm_only_sol | 500 | 0.0 | 0.0 | 7.5 | – | 0 | 0 | – | 0 |
| jevagent_best_sol | 500 | 0.0 | 0.0 | 8.426 | -0.926 | 3250 | 463 | 14.2 | 413 |
| llm_only_solsmall | 500 | 0.0 | 0.0 | 7.504 | -0.004 | 0 | 0 | – | 0 |

_Δ success: arm − `llm_only_sol` on the paired tasks, paired task bootstrap 95 % percentile interval (Newcombe score interval beside it). Savings: 1 − cost(arm) / cost(`llm_only_sol`) per episode at the frozen price (Jev's calls included), paired task bootstrap; 'incl. cache writes' adds the pricing page's cache-write surcharge (a sensitivity, not the registered price). Latency: summed model-call latency per episode (backoff excluded); wall: episode wall clock. Coverage is never called intervention (A7.16 (d))._
