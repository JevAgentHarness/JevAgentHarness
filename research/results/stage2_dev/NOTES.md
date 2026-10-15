# Stage-2 development check (DEVELOPMENT data: never a paper number)

- Config: `configs/dev/stage2_dev.yaml` (`stage2_dev`); development seeds **900–999** (inside 0–999, never used by a
  pilot or dev round), frozen suite settings (v2, near distractors, 60 distractors, 20 noise lines, strict); 100 tasks
  per arm; System-1 Jev via OpenRouter (`{type: jev, provider: openrouter}`, fresh cache `cache/s1_stage2_dev.sqlite`);
  System-2 `gemini-3.5-flash` / `gemini-3.5-flash-lite` (temperature 0, minimal thinking).
- Run 2026-10-02 20:23–20:52 UTC from the worktree at commit `b41e906` (workers 3; one container restart at ≈ 20:32,
  resumed by task id: every arm holds exactly 100 records, none errored). Spend ≈ $13 of episode cost (plus a 2-task
  smoke, ≈ $0.25).
- Numbers: `summary.json` (written by `scripts/stage2_dev_summary.py runs/stage2_dev/stage2_dev --out
  results/stage2_dev`; provenance = path, sha256, record count of every arm file). The run files stay in the worktree's
  gitignored `runs/stage2_dev/`.

## What was checked

| arm | System-2 | success | $ / episode | latency s / ep. | LLM calls / ep. | System-1 calls / ep. | coverage (tool / arg) |
|---|---|---|---|---|---|---|---|
| `llm_only` | Flash | 100 / 100 | 0.0432 | 7.7 | 7.50 | 0 | – |
| `jevagent_default` (cascade τ 1.00 / 0.99 + stop trigger τ 0.9) | Flash | 100 / 100 | 0.0343 | 10.6 | 6.43 | 17.4 | 0.25 / 0.20 |
| `jev_ts_cascade_tauhat_dev` (the default's tool group alone) | Flash | 100 / 100 | 0.0340 | 11.7 | 6.41 | 10.9 | 0.25 / 0.21 |
| `llm_only_s2lite` | Flash-Lite | 88 / 100 | 0.0097 | 5.3 | 7.56 | 0 | – |
| `jevagent_default_s2lite` (reflection trigger τ 0.5) | Flash-Lite | 83 / 100 | 0.0099 | 8.5 | 8.44 | 6.6 | – |

Paired, against the same System-2's LLM-only arm (n = 100; descriptive):

| pair | wins / losses | Δ success | NI p (3 pt, `bb_ordered`) | cost saving [95 % CI] | p (cheaper) | latency Δ s / ep. |
|---|---|---|---|---|---|---|
| default vs LLM-only (Flash) | 0 / 0 | 0 | 0.048 | 20.6 % [17.8, 23.2] | < 0.001 | 2.9 slower |
| cascade alone vs LLM-only (Flash) | 0 / 0 | 0 | 0.048 | 21.3 % [18.7, 24.0] | < 0.001 | 4.0 slower |
| default vs cascade alone (Flash) | 0 / 0 | 0 | 0.048 | −0.9 % [−2.8, 0.8] | 0.85 | 1.1 faster (p 0.09) |
| default vs LLM-only (Flash-Lite) | 4 / 9 | −5 pt [−12, +1] | 0.73 | −1.2 % [−2.7, 0.3] | 0.94 | 3.2 slower |

- **End to end.** Every arm ran 100 episodes with no infrastructure error; the profile arms carry `meta.profile`
  (path, sha256, section) and the runner wrote `profiles.json`. The composed Flash profile logs both groups' decisions
  (tool_select / tool_params cascade and 650 stop-trigger decisions).
- **Ledger.** For every episode of every arm the per-call records (`calls`, measurement calls excluded) sum exactly to
  the ledger totals (`usage.total` calls and cost): 0 of 500 episodes differ; no unpriced model.
- **Direction (Flash).** The cascade at the calibration τ̂\* keeps 100 / 100 at −21 % cost (System-1 decides 25 % of
  tool choices and 21 % of argument slots; 14 % of steps need no LLM call). It is *slower* per episode (+3–4 s): it adds
  ≈ 11 Jev requests per episode at ≈ 0.2 s each (median 0.21 s) while removing ≈ 1.1 of 7.5 LLM calls, and the forced
  argument-filling calls are not faster than native ones. This is why A7.9 moved latency out of S2-1.
- **Composition (Flash).** The stop trigger never fired (0 of 650 decisions): it adds ≈ 6.5 System-1 requests per
  episode and ≈ 1 % cost to the cascade without changing any episode. On this suite the composed default is the cascade
  plus overhead; Stage 1's selection rule (A7.1) would keep the stop group with the LLM on evidence like this.
- **Flash-Lite.** The reflection trigger fired on 87 of 657 opportunities (13 %, close to the C5 control rate 0.14) and
  read 83 vs 88 (4 wins, 9 losses; the n = 50 pilots had read +8 and +4 pt). The development evidence on it is mixed.

## Consequence for the default profile (development evidence only)

`configs/harness/jevagent_default.yaml` is updated after this check: the Flash section keeps the tool cascade at the
calibration τ̂\* and drops the stop trigger (it never fired and only added calls); the Flash-Lite section delegates
nothing (the reflection trigger's development evidence is mixed: +8 / +4 pt in two n = 50 pilots, −5 pt here). No
Stage-1 main-run result and no Stage-2 seed entered this; the default stays a descriptive Stage-2 row (A7.3 (a)).
