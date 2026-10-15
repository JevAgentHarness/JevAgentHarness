# Integrating System-1 decisions into existing harnesses

We study the same WHERE × HOW configurations in four places:

| Harness | Language / license | Integration | Code | Benchmarks |
|---------|--------------------|-------------|------|------------|
| **Reference loop** (ours) | Python | native: every decision point & mode is a config | `src/jevharness/harness/agent.py` | JevGym, τ²-bench, ALFWorld, BFCL/When2Call (static) |
| **pi** (earendil-works/pi, ex badlogic/pi-mono) | TypeScript, MIT | extensions (`pi -e x.ts`): `tool_call` block/mutate, `setActiveTools`, `turn_end`/`agent_before_settle` continue, `registerVirtualModel` routing, `appendEntry` memory; built-in `classify()` with TypeSafe provider | `adapters/pi/` | Terminal-Bench 2.0 and Terminal-Bench-Science via Harbor (`pi --print --mode json --extension ...`) |
| **Codex CLI** (openai/codex) | Rust, Apache-2.0 | command hooks (`hooks.json`): PreToolUse can only **deny** (allow/ask outputs are rejected by Codex's parser), PermissionRequest allow/deny (runs *before* the Guardian → natural Jev-vs-Guardian comparison point; needs `approvals_reviewer="auto_review"` or `--approve-for-me` under `codex exec`), PostToolUse, Stop (block → continue), UserPromptSubmit; baseline = Guardian `auto_review`. Harbor's codex agent runs with `--dangerously-bypass-approvals-and-sandbox`, so Guardian arms need an agent subclass | `adapters/codex/`, `src/jevharness/adapters/hooks.py` | Terminal-Bench subset (Harbor `codex` agent), `codex exec --json` task sets |
| **DeepSeek Harness** (deepseek-ai/deepseek-harness, `dsh`) | TypeScript, MIT (dev preview) | command-hook bridges (Codex dialect and a generic one) (PreToolUse deny/ask, PostToolUse, Stop force-continue, UserPromptSubmit); native Cordis plugins (`tools/pre-execute`, `agent/turn-stopping`, `agent/request`) | `adapters/deepseek/` | generalisation demo (not in Harbor) |
| **mini-SWE-agent** | Python, MIT (~190-line agent) | subclass `DefaultAgent`: command guard, submit verification, per-step routing, reflection trigger | `src/jevharness/adapters/mini_swe.py`, `adapters/mini_swe/` | SWE-bench Verified subset |

## Decision point → hook mapping

| Decision (paper) | pi | Codex / dsh hooks | mini-SWE-agent | Jev question |
|------------------|----|---------------------------------|----------------|--------------|
| tool_select / guard (action approval) | `tool_call` → `{block, reason}` | PreToolUse → deny (Codex) / allow·ask·deny (dsh); PermissionRequest → allow/deny (Codex) | before `execute_actions` | Choice {allow, ask, block} (+ Noul destructive?) |
| tool_select / assist | `before_agent_start`/turn → `pi.setActiveTools(topk)` | – (needs proxy) | – | Choice / Score per tool |
| tool_params | `tool_call` mutates `event.input` (candidates only) | PreToolUse `updatedInput` (Codex) | – | Choice over candidates + OTHER |
| stop / guard | `turn_end` / `agent_before_settle` → `continue: true` | Stop → block + reason (loop-guarded) | on submit | Noul "task complete & verified?" |
| routing / trigger | `registerVirtualModel({route})` (extends official `jev-router.ts`) | UserPromptSubmit (log) / Responses proxy | swap model per step | Noul "needs strong model?" |
| memory write / read | `tool_result` → `appendEntry`; `context` inject | PostToolUse → note file (+ additionalContext) | – | Noul "worth remembering?", Choice relevant notes |
| reflection / trigger | after error → inject prompt | PostToolUse additionalContext | after failed/repeated cmd | Noul "stuck / failed?" |
| compaction | `session_before_compact` | PreCompact | – | Score per message |

All adapters log decisions in the same JSONL schema
`{ts, harness, session, dp, mode, question, options, s1_value, s1_conf, s1_dist, final_value, latency_ms, input_tokens, model, error}`
so `jevbench report`-style analysis applies across harnesses; hooks **fail open** (never break the host harness).

**DeepSeek Harness caveats**: dsh strips env vars matching `/KEY|PASSWORD|SECRET|TOKEN/i` from hook processes, so pass the Jev key via `JEV_ENV_FILE=<dotenv path>`; dsh "ask" fails closed (deny) without an approval answerer, and it never sets `stop_hook_active`, so the adapter's per-session counter is the only Stop loop guard.

## Running on Terminal-Bench 2.0 (Harbor)

Harbor (github.com/harbor-framework/harbor, Apache-2.0) is the official TB-2.0 harness and ships installed-agent
adapters for pi, codex, mini_swe_agent, openhands, etc. See `adapters/pi/README.md` and
`adapters/codex/README.md` for exact commands; budget ~$30–300 per full 89-task pass per arm.

## L3 smoke (2026-10-01): every Level-3 integration end to end with real models

Each Level-3 integration was run at tiny scale, with real models, to check the pipelines before the
full runs. The agent was Gemini 3.5 Flash (minimal thinking). System-1 was the Gemini 3.5 Flash-Lite
stand-in, not Jev: `openrouter.ai` is blocked here. Outputs are under `runs/dev/l3_smoke/`, which is
gitignored. Numbers are smoke-scale, n = 1–3 per arm, and are **not** results. Total spend was about $4–5.
The regression tests for every bug found are in `tests/test_l3_smoke.py`.

| Integration | Scale | Arms | Outcome | $ per task (agent) | Wall clock | Details |
|---|---|---|---|---|---|---|
| SWE-bench Verified via mini-swe-agent (Docker) | 2 instances | `baseline`, `action_guard` (τ 0.3) | resolved 1/2 vs 2/2 (local swebench 4.1.0 evaluation); baseline pytest-7432 hit the smoke's step limit (80) | 0.07–0.08 solved, 0.78 looping | 61–65 s solved, 345 s looping; evaluation 59 s for both arms | [`adapters/mini_swe/README.md`](../adapters/mini_swe/README.md#l3-smoke-2026-10-01) |
| Terminal-Bench 2.0 via pi + Harbor 0.23 (Docker) | 2 tasks | `baseline`, `jev_gate_stop` (tool gate + stop verifier, τ 0.5) | prove-plus-comm solved by both arms; overfull-hbox timed out (750 s) in both | 0.02–0.08 solved, 0.66–0.68 timed out (+ S1 ≤ $0.03) | about 90 s solved, about 820 s timed out | [`adapters/pi/README.md`](../adapters/pi/README.md#l3-smoke-2026-10-01) |
| τ²-bench retail | 3 tasks × 1 trial | `llm_only`, `jev_guard` (policy + transfer guard) | 3/3 both (Flash user simulator); 1/3 vs 3/3 with the Flash-Lite simulator, which is simulator noise | 0.074–0.082 (+ S1 0.0012, user 0.007) | about 50 s per arm at concurrency 3 | [`docs/TAU2.md`](TAU2.md#retail-smoke-l3-2026-10-01) |
| ALFWorld (regression) | 1 game | `llm_react`, `s1_action_shadow`, `s1_action_assist_k3` | 1/1 all; the shadow arm reproduces the LLM trajectory exactly | 0.005 / 0.005 (+0.004 shadow) / 0.033 | 53 s for all three arms | [`docs/ALFWORLD.md`](ALFWORLD.md#8-l3-smoke-2026-10-01-regression-check-after-the-harness-changes) |
| LoCoMo (regression) | 10 questions | the 5 pilot arms | LLM write gate stores 44% (0% before the fix), the same as the pilot | 0.0002–0.026 per question | 437 s, mostly the Flash write gate | [`docs/MEMORY.md`](MEMORY.md#l3-smoke-2026-10-01-regression-check-after-the-harness-changes) |

**Bugs found and fixed**

- **mini-swe** (`adapters/mini_swe/run_swebench.sh`):
  - It did not export `.env`, so litellm had no `GEMINI_API_KEY`.
  - It had no way to add config overlays. New: `CONFIG_OVERLAY` with `gemini.yaml` and
    `s1_gemini_standin.yaml`.
  - It could not evaluate against an offline dataset directory: the SWE-bench harness needs the `.jsonl`
    file, not the directory. `huggingface.co` is blocked here.
  - A missing evaluation report aborted the arm loop under `pipefail`.
- **τ² launcher** (`bench/suites/tau2.py`, `scripts/run_tau2.sh`): tau2 hard-codes the retail
  NL-assertion judge to gpt-4.1. New: `--jev-nl-judge` / `NL_JUDGE_LLM`, which is recorded in the
  side-car. The τ² gold heuristic kept a bypassed READ action, such as authentication by name+zip after
  the agent had used the e-mail, as the gold next step for the rest of the episode. It is now dropped.
- **TB2** (`adapters/pi/`):
  - New: `run_tb2.sh`, one Harbor job per arm, including a `jev_shadow` arm.
  - New: `summarize_harbor.py`, per-trial reward, cost, phase timings and S1 stats.
  - New: `harbor_host_gateway.yaml`, so containers can reach a host decision server.
  - New: `tb2_sandbox_tasks.sh` for restricted networks.
  - `run_tb2.sh` no longer lets the shell glob-expand task patterns.

**Blockers specific to this sandbox; none applies with open egress**

- SWE-bench: `huggingface.co` is blocked, so the dataset came from a GitHub mirror. Use the HF dataset.
- TB2:
  - Harbor's dataset registry (`*.supabase.co`) is blocked; use the git clone.
  - `astral.sh` is blocked; the TB2 verifiers install uv from it (smoke tasks patched).
  - The Debian mirrors are blocked, so only Ubuntu-based tasks run.
  - TLS interception needs the proxy CA inside the task containers (`EXTRA_CA`).
- Docker Hub answers anonymous pulls with intermittent 429 / 502.

**Failure modes to budget for at full scale**

- Looping agents dominate cost. They run to the step or cost limit (SWE) or the agent timeout (TB2) at
  about 10× the cost of a solved task.
- The Flash-Lite user simulator ends retail conversations right after confirming. Use Flash or gpt-4.1
  as the τ² user.
- The stand-in S1 is overconfident: the SWE guard never fired, while the TB2 gate blocked 19 of 69
  proposals. Jev numbers will differ.

**Full-scale commands** (Gemini agent, stand-in S1; drop the stand-in overlay or `S1` override once Jev
is reachable):

```bash
# E8 SWE-bench Verified, 100 instances, all default arms, local evaluation (swebench==4.1.0 in .venv-eval)
CONFIG_OVERLAY="adapters/mini_swe/gemini.yaml adapters/mini_swe/s1_gemini_standin.yaml" N=100 WORKERS=4 \
  EVAL=local EVAL_PY=.venv-eval/bin/python EVAL_WORKERS=4 adapters/mini_swe/run_swebench.sh
# E9 Terminal-Bench 2.0, 89 tasks (decision server on the host first; see adapters/pi/README.md)
HARBOR=<py3.12 venv>/bin/harbor N_CONCURRENT=8 adapters/pi/run_tb2.sh baseline jev_shadow jev_gate_stop
# E5 tau2 airline + retail, 3 trials, all arms
AGENT_LLM=gemini/gemini-3.5-flash USER_LLM=gemini/gemini-3.5-flash SMALL_LLM=gemini/gemini-3.5-flash-lite \
  AGENT_LLM_ARGS='{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}' \
  USER_LLM_ARGS='{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}' \
  NL_JUDGE_LLM=gemini/gemini-3.5-flash S1='{type: llm_generative, llm: {provider: gemini, model: gemini-3.5-flash-lite}, reasoning_effort: minimal}' \
  MAX_CONCURRENCY=8 bash scripts/run_tau2.sh
# E6 ALFWorld 134 games / E7 LoCoMo 10 conversations
jevbench run configs/experiments/alfworld_action_only.yaml --out runs
python -m jevharness.bench.suites.memory_qa --config configs/pilot/locomo_gemini.yaml --out runs \
  -o dataset.n_conversations=10 dataset.max_questions=null
```

Rough budgets at smoke rates on Gemini 3.5 Flash, per arm:

| Benchmark | Budget per arm |
|---|---|
| SWE-bench, 100 instances | $30–50 |
| Terminal-Bench, 89 tasks, one pass | $25–50 |
| τ², 492 simulations | about $40 |

## Terminal-Bench-Science (hard L3 suite): faster *and* better

Terminal-Bench-Science 0.1.0 is our hard L3 suite. It has 70 expert-written research workflows: earth 8,
engineering 9, life 19, mathematical 17, physical 17. Every task gets 8 h of agent budget, and the
verifier runs in a separate no-network container. It runs through the same pi + Harbor agent as TB2. The
goal here is that Jev makes the agent both **faster** (wall clock) and **better** (success).

- Full write-up, with the format survey, the TB2 differences, commands, budgets and failure modes:
  [`adapters/pi/README.md`](../adapters/pi/README.md#terminal-bench-science-via-harbor).
- Code:
  - `run_tbsci.sh`: `run_tb2.sh` with TB-science defaults, i.e. the Hub dataset `@0.1.0`, a 1 h agent
    cap, a 100-turn cap and `SUBSET=`.
  - New `run_tb2.sh` arms and knobs: `jev_reflect`, `jev_stop_reflect`, `oracle`, `nop`, `MAX_TURNS`,
    `AGENT_TIMEOUT_MULT`, `EXTRA_CA_VERIFIER`. Relative paths are now made absolute.
  - `tbsci_tasks.sh`: flattens the nested `tasks/<domain>/<field>/<task>` layout for `harbor run -p`, with
    the optional sandbox CA patch.
  - `tbsci.py`: survey, subsets, `patch-ca`.
  - `summarize_harbor.py`: p50/p90, solved-by-t, paired deltas.
  - Tests are in `tests/test_tbsci.py`.

**Hard subset.**
- **Static pool.** `adapters/pi/tbsci_hard_pool.txt` has 37 tasks: no GPU, ≤ 4 CPUs / 16 GB, image
  build and verifier ≤ 30 min, expert estimate 4–24 h.
- **Calibration.**
  - The `oracle` and `nop` arms make no API calls. Drop a task if the reference solution fails here,
    or if it takes more than half the agent cap.
    `nop` gives the floor of tests that pass with no output: `ode-law-discovery` passes 4 of 12 that
    way.
  - Then one capped baseline attempt per pool task. Keep a task if the baseline solved it or beat its
    floor.
  - Keep about 20 tasks, balanced across domains, and freeze them as `tbsci_hard.txt` before the
    evaluation.
- **Evaluation.** Every arm is rerun on the frozen list with k = 3 attempts. The calibration runs are
  never reused, which avoids selection bias.

**Arms.**
- `baseline`.
- `jev_stop_reflect` (primary): the stop verifier catches premature "done", and the reflection trigger
  breaks loops.
- `jev_gate_stop`, to compare with E9.

**Measurement.** Everything below comes from `summarize_harbor.py --baseline baseline --budgets 600,1800,3600`.
- Success per trial and per task.
- Tests passed above the `nop` floor.
- Agent-run and trial wall clock, p50 and p90 (timeouts count at the cap).
- Steps.
- LLM + S1 cost, and cost per solved trial.
- Paired per-task deltas: task wins and losses with a sign test, median Δ time over all tasks and over
  tasks both arms solved, and median Δ cost.
- The "solved by t" curve, which shows both claims in one figure.

```bash
python -m jevharness.decision.server --host 0.0.0.0 --port 8090 --name jev-1.13 --s1 '{"type":"jev","provider":"openrouter"}' &
# E9s-cal: calibration over the 37-task pool: oracle + nop ($0), then 1 capped baseline attempt
HARBOR=<py3.12 venv>/bin/harbor SUBSET=hard_pool OUT=runs/tbsci_cal N_CONCURRENT=4 adapters/pi/run_tbsci.sh oracle nop baseline
# E9s: evaluation on the frozen list, k=3, same caps for every arm
# (run_tbsci.sh defaults: 1 h agent cap, MAX_TURNS=100)
HARBOR=<py3.12 venv>/bin/harbor JEV_WIRE_MODEL=jev-1.13 SUBSET=hard N_ATTEMPTS=3 N_CONCURRENT=4 \
  adapters/pi/run_tbsci.sh nop baseline jev_stop_reflect jev_gate_stop
```

**Budget** (Gemini 3.5 Flash, worst case with every trial at `MAX_TURNS=100`):
- **Per trial.** pi does not compact within Gemini's 1M context, so the input per turn grows by about
  500–1,000 tokens and the cost is roughly quadratic in turns. That is about $1.8–3.3 per trial at 100
  turns and $6.6–12.5 at 200.
- **Calibration.** `oracle` and `nop` cost $0; the baseline pass is ≤ $125.
- **Evaluation.** ≤ $200 per arm, about 1.5–5 h per arm at concurrency 4.
- **Jev itself.** ≤ $0.01 per trial.

**Sandbox.** Unlike TB2, TB-science builds its images, so the TLS-intercepting proxy breaks
`pip`/`uv`/`curl` inside `docker build`.
- `PATCH_CA` is a sandbox deviation that makes the builds trust the proxy CA.
- Debian mirrors are blocked here, so only Ubuntu-family tasks run (`tbsci_sandbox.txt`).
- Hugging Face and OpenRouter are blocked from containers. The decision server runs on the host.

**Dev smoke (DEV, real Jev, not paper numbers).** Results are in the
[README](../adapters/pi/README.md#dev-smoke-2026-10-01-dev-not-paper-numbers). It ran 2 tasks × 2 arms
(`baseline`, `jev_stop_reflect`), plus `oracle` and `nop`. The whole pipeline worked: CA-patched builds,
pi install, the agent, real Jev through the host server, artifact hand-off and the separate verifier.
- **Cost.** Gemini $1.31, Jev $0.0005.
- **Outcome.** Nothing was solved under the 20- and 40-turn caps.
- **What Jev did.** `reflect` fired 3 of the 4 times it was asked; the stop verifier was never reached.
- **Oracle.** It solved `inelastic-constitutive-discovery` in 32 min. `ode-law-discovery`'s reference
  solution did not finish within 1 h.

## Why not fork the harnesses?

Every integration above uses the harness's *public* extension surface (extensions / hooks / subclassing), so
(1) results reflect realistic deployment, (2) we can pin upstream versions, (3) the same Jev-wire backend
(hosted Jev, OpenJev, or our trained model via `training/serve_jev.py`) plugs into all of them.

## Harness affordances limit which modes are deployable (finding for the paper)

| Mode | Reference loop | pi (extensions) | Codex (hooks) | DeepSeek Harness (hook bridge) | mini-SWE-agent (subclass) |
|------|----------------|-----------------|---------------|-------------------------------|---------------------------|
| Replace / Cascade at tool_select | ✓ | ✗ (no way to answer a model request without the LLM) | ✗ | ✗ (native plugin could) | ✗ (bash-only action space) |
| Assist (prune tools) | ✓ | ✓ `setActiveTools` | ✗ | ✗ | – |
| Guard (approve/deny) | ✓ | ✓ `tool_call` block | ✓ deny-only PreToolUse; PermissionRequest allow/deny | ✓ (ask fails closed) | ✓ |
| Trigger: stop | ✓ | ✓ `agent_before_settle` continue | ✓ Stop block | ✓ (no stop_hook_active; counter guard) | ✓ |
| Trigger: routing | ✓ | ✓ virtual model | log only (needs Responses proxy) | ✓ `agent/request` plugin | ✓ |
| Trigger: memory / reflection | ✓ | ✓ | ✓ (additionalContext) | ✓ | ✓ (prompt injection) |

Default thresholds differ: pi extensions default `JEV_TAU=0.5`, the Python `DPConfig` defaults to 0.8 —
always set `JEV_TAU` explicitly when comparing harnesses.

## OpenAI Responses API as System-2 and System-1 (2026-10-01)

`provider: openai_responses` (`src/jevharness/llm/openai_responses.py`) speaks OpenAI's current `POST /v1/responses`
directly over `httpx`, so it is testable offline with a fake transport (`tests/test_openai_responses.py`) and needs no SDK.
`provider: openai` remains the Chat Completions path (`openai_compat.py`). The reason for a second client: the GPT-5.x/6
docs state that Chat Completions supports function calling only with `reasoning_effort: none` and that reasoning with
tools needs Responses (https://developers.openai.com/api/docs/guides/latest-model). **No live call has been made yet**
(`OPENAI_API_KEY` is not set in the sandbox); `bash scripts/smoke_openai.sh` is the one-command check once it is.

Differences from Chat Completions that affected the harness:

| Topic | Chat Completions (`openai_compat.py`) | Responses (`openai_responses.py`) |
|-------|---------------------------------------|-----------------------------------|
| Conversation | `messages` | `input` *items*: the leading system prompt becomes `instructions` (re-sent on every call, it is not carried by `previous_response_id`); user/assistant messages are `{role, content}` items; a tool call is a `function_call` item (`call_id`, `name`, `arguments`), its result a `function_call_output` item (`call_id`, `output`) |
| Tools | `{type: function, function: {name, ...}}` | flattened `{type: function, name, description, parameters, strict?}`; `tool_choice: {type: function, name}`; `strict_tools: true` needs `additionalProperties: false` + every property required, so it is off by default |
| Reasoning | `reasoning_effort` | `reasoning: {effort}` with `none | minimal | low | medium | high | xhigh | max` (model-dependent: gpt-6-luna and gpt-5.4-nano accept `none`, gpt-6.1-sol accepts neither `none` nor `minimal`, gpt-6-astra not `none`). Opaque `reasoning` items come back in `output`; with `store: false` (our default) they carry `encrypted_content` (`include: ["reasoning.encrypted_content"]`) and the docs ask for them to be passed back untouched between a function call and its output. The client keeps them in `ToolCall.extra["openai"]["reasoning"]` (echoed by the harness as `extra_content`, the same path as Gemini thought signatures) or in `ChatResponse.extra` for text-only replies, and re-emits them in front of the call. |
| Sampling | `temperature`, `top_p` | rejected unless the effort is `none` ("remove temperature, top_p and top_logprobs", latest-model guide). The client drops them up front for reasoning models and counts it (`stats["dropped_params"]`); an unexpected 400 naming one of them is retried once without them. |
| Token limit | `max_tokens` | `max_output_tokens`, documented minimum 16 (the logprob decider's `max_tokens=1` is clamped; position 0 is still what is read) |
| Logprobs | `logprobs: true, top_logprobs: n` | `top_logprobs: n` + `include: ["message.output_text.logprobs"]`, returned per position inside the message's `output_text` part. **Only with effort `none` or on non-reasoning models** (gpt-4.1 / gpt-4o families). Otherwise the client raises `LogprobsUnavailableError` (before sending when the effort is known, or on the API's 400); a 200 without logprobs also raises unless `allow_missing_logprobs: true`. Verbalised deciders (`llm_generative`, `s2_self`) work with every model. |
| Usage | `prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens`, `completion_tokens_details.reasoning_tokens` | `input_tokens`, `output_tokens` (reasoning included), `input_tokens_details.{cached_tokens, cache_write_tokens}`, `output_tokens_details.reasoning_tokens`. Cost = `cached_input_per_m` on cached tokens + input price on the rest + output price; cache *writes* (1.25× input on gpt-5.6+) and the > 272k-token uplift are logged (`cache_write_tokens` in the ledger meta) but not priced. |
| Retention | n/a | `store: false` by default (nothing kept on OpenAI's side); the returned model id (a dated snapshot) is mapped back to the configured alias for pricing |
| State | stateless | `server_state: true` chains calls with `previous_response_id` and sends only the items appended since the last call (forces `store: true`). Off by default: the conversation-state docs say that all previous input tokens of the chain are billed again on every turn, so chaining saves wire bytes and client bookkeeping, not tokens — the "incremental state" argument of `DECISION_MODEL_LANDSCAPE.md` §3 therefore has to be measured. Every call records `wire_input_tokens_est` (estimate of what was sent) next to the billed `input_tokens` and `cached_input_tokens` in the ledger meta and in `client.stats`. A history that is not the previous request plus the echo of its own output (a System-1 *replace* of the model's call, a stop-verifier note, the early-stop branch) breaks the chain and falls back to a full stateless send (`stats["chain_breaks"]`). |
| Errors | SDK exceptions | `OpenAIResponsesError(status, message)`; 429/5xx retried with backoff and `Retry-After` (latency of the successful attempt only, as in the other clients); 401 and a missing key point at `OPENAI_API_KEY` in `.env` |

Model ids and prices were read from https://developers.openai.com/api/docs/pricing and the model pages on 2026-10-01
(`configs/prices/models.yaml`: `verified` vs `verify` per row). The dev configs use `gpt-6.1-sol` (main, effort `low`)
and `gpt-6-luna` (lite, effort `none`, the cheapest current model with function calling); confirm against
`GET /v1/models` before a paid run. Unverified without a key: that the current models accept `reasoning.encrypted_content`
with `store: false`, the exact 400 wording for rejected sampling parameters, and which models actually return
`message.output_text.logprobs` (the docs rule above is implemented; the smoke script reports what the API does).
