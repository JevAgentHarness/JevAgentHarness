# Benchmark protocol

Written on 2026-10-13, before any benchmark run. Changes after a run starts are appended under "Changes", with the
date and the reason; nothing above that heading is edited once a run has started.

## Question

When an unmodified agent harness (Codex, DeepSeek Harness, pi) runs through JevAgentHarness (`jah-auto`), does it
solve as many tasks as the same harness on the strong model alone (`jah-strong`), at a lower cost and in less time?

## Setup

- **Harnesses:** Codex CLI 0.162.1, pi 1.2.0 and DeepSeek Harness 0.2.1-alpha.2, each through Harbor 0.24.0. Each
  harness uses its own prompts and tools unchanged; only the model endpoint is the proxy.
- **Tiers:** strong = Gemini 4 Argon (thinking `high`); fast = Gemini 3.8 Flash (thinking `low`). Both run on Google's
  direct API at the list prices of 2026-10-13 (`prices.py`).
- **Decision model:** Jev (`typesafe/jev-1.13` on OpenRouter), τ = 0.5. Router defaults: first step strong, finish
  check on, fail-safe to strong.
- **Arms** (one proxy, one host, run side by side): `jah-strong` (baseline), `jah-auto` (routed), `jah-fast`
  (reference: the fast tier alone).
- **Benchmark:** Terminal-Bench 2.0 (89 tasks). A fixed hash split (`splits/`) gives a **dev** set of 15 tasks
  and a **test** set of 74 tasks.
- **Measures, per task:** success (Harbor reward = 1); cost = LLM + decision-model dollars logged by the proxy;
  time = Harbor's agent-execution seconds. Arms are compared on the tasks both completed, with paired bootstrap
  intervals for the success difference. Infrastructure errors count as failures and are reported.

## Phases

1. **Dev (15 tasks):** plumbing, cost per task, and router behaviour. Router settings may change here (τ, the
   question, stickiness, the finish check). The dev numbers are never reported as results.
2. **Test (74 tasks):** one run per harness × arm with the router frozen after dev. These are the reported numbers.

## What counts as the claim holding (per harness, on test)

- Success of `jah-auto` is no more than 3 points below `jah-strong`.
- Cost per task of `jah-auto` is at least 40% below `jah-strong`.
- Agent time per task of `jah-auto` is lower than `jah-strong`.

With 74 tasks, a 3-point margin cannot be shown statistically; the interval is reported next to the point
estimate, and the README states the result for each harness, including harnesses where the claim fails.

## Budget

At most $1,000 in total across all phases (Gemini plus Jev); the proxy refuses new sessions at the cap set for
each run.

## Changes

- **2026-10-14, after the dev phase, before any test run.** Dev (15 tasks; rows dev1-dev4) showed that with Gemini 4
  Argon / 3.8 Flash the routed arm kept success (Codex 15/15 vs 11/15, dsh 11/15 vs 9/15, pi 12/15 vs 13/15) but saved
  only 0-30 % of cost, with large run-to-run noise (the same Codex setting gave +12 % and -5 % cost in two runs).
  Raising tau to 0.75 / 0.85, a medium-thinking fast tier and Argon effort routing did not help on dev. Decisions:
  (1) the router stays at its defaults (tau 0.5, first step strong, finish check); (2) **Codex** uses OpenAI's own
  models on OpenAI's direct API, strong = GPT-6.1 Sol (medium reasoning), fast = GPT-6 Luna (low reasoning), config
  `configs/openai_sol_luna.yaml` (author: GPT-6 Astra is too expensive; base models stay off OpenRouter, which serves
  only Jev); (3) **pi and DeepSeek Harness** keep Gemini 4 Argon / 3.8 Flash (`configs/gemini_argon_flash.yaml`);
  (4) for release speed the OpenAI pair goes to the test split without a separate dev round, and the arms run on two
  hosts to reduce load. Test arms: Codex strong / auto / fast; pi and dsh strong / auto.
- **2026-10-14, before any test result.** To keep OpenAI spend small (author), the Codex row on the OpenAI pair runs
  only the first 20 tasks of the test split (`splits/tb2_test20.txt`) with the arms `jah-strong` and `jah-auto`; its
  numbers are reported as a small check, not as a claim. More harnesses run on the Gemini pair instead: pi and
  DeepSeek Harness on all 74 test tasks (host 2), and mini-swe-agent and Terminus-2 (Harbor's reference harness for
  Terminal-Bench) on all 74 test tasks after a two-task plumbing smoke on dev tasks (host 1). Arms: `jah-strong` and
  `jah-auto`; the router stays at its defaults.
- **2026-10-14 22:20 UTC, test runs still going.** Interim test numbers (not final) show the Argon / Flash pair keeps
  success but saves little cost (0-16 %, Terminus-2 costs more): the routed arm sends about half of its steps to Argon
  and takes about a third more steps, and Flash is only ~2.7x cheaper per token. A **dev-only exploration** starts on a
  third host with a cheaper fast tier, Gemini 3.5 Flash-Lite (`configs/gemini_argon_lite.yaml`), for pi and
  mini-swe-agent on the 15 dev tasks, arms `jah-strong`, `jah-auto` (tau 0.5) and `jah-auto-t75`. The running test
  rows are not changed. A setting from this exploration goes to the test split only as a new, separately reported row,
  written here before it starts.
- **2026-10-15 00:15 UTC, registered before the run starts: test row "lite".** Dev exploration result (15 dev tasks,
  rows dev-lite + dev-lite2, $37.7): with Gemini 3.5 Flash-Lite as the fast tier, `jah-auto-t75` kept success and
  cut cost by about half (mini-swe-agent 66.7 % vs 66.7 %, 64 % cheaper, 51 % faster; pi 80.0 % vs 73.3 %, 48 %
  cheaper, 14 % faster); `jah-auto` (tau 0.5) saved 21-43 %. The registered test row: config
  `configs/gemini_argon_lite.yaml`, arm `jah-auto-t75` (tau 0.75, first step strong, finish check on), harnesses pi,
  mini-swe-agent, DeepSeek Harness and Terminus-2, all 74 test tasks, one run each (host 3). The baseline is each
  harness's `jah-strong` arm of the main test rows: the same strong tier (Gemini 4 Argon, high thinking), harness
  version and task set, run on 2026-10-14. The claim criteria above apply unchanged. The main test rows stay as they
  are and are reported next to this row.
- **2026-10-15, dev only, no test run.** The decision-point modes of `docs/modes.md` (the shadow preset, the stop
  guard, a finish check on mini-swe-agent's submit command, the action guard, the reflection trigger and the routing
  guard) were run on the 15 dev tasks with mini-swe-agent and pi on the Flash-Lite pair
  (`configs/gemini_argon_lite_modes.yaml`, rows dev-modes, dev-modes2, dev-modes3, $53.2; `results/dev_modes.md`).
  No registered row changed. The reflection trigger (tau 0.8) is the only candidate for a later test row, which
  would be registered here before it starts.
