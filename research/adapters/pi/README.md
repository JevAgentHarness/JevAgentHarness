# Jev System-One extensions for pi

TypeScript extensions that put a fast System-One decision model (TypeSafe's Jev, an OpenJev
server, or our own trained model) at the decision points of the
[pi coding agent](https://github.com/earendil-works/pi). Each extension implements one or two cells of
the paper's WHERE x HOW design space (`src/jevharness/harness/config.py`) inside a real, unmodified
agent harness, and logs every decision in the same JSONL schema as our other harness adapters.

| File | Decision point / mode | pi hook | What System One decides |
|---|---|---|---|
| `jev-tool-gate.ts` | `tool_select` / `guard` | `tool_call` (block + reason) | Is the proposed call the right next step? Below tau the call is blocked and the LLM retries. |
| `jev-active-tools.ts` | `tool_select` / `assist` | `before_agent_start`, `turn_end` -> `pi.setActiveTools()` | Which k tools the LLM is shown for the next request. |
| `jev-stop-verify.ts` | `stop` / `guard` (or `trigger`) | `agent_before_settle` -> `continue: true` (`turn_end` for trigger) | Is the task really complete? Below tau the agent gets a verification nudge and continues. |
| `jev-router.ts` | `routing` / `trigger` | `pi.registerVirtualModel()` (`s1/router`) | Strong or cheap model for each request (plus an error-escalation rule). |
| `jev-memory.ts` | `memory_write` / `trigger`, `memory_read` / `assist` | `turn_end` (store), `context` (inject) | Which tool results become notes; which k notes are shown for the next request. |
| `jev-reflect.ts` | `reflection` / `trigger` | `turn_end` -> custom message | After a failed or repeated call: should the agent pause and reflect? |
| `jev-compact.ts` | `compaction` (addition to the paper's list) | `session_before_compact` | Which messages survive compaction (prefilter for pi's summarizer, or an extractive summary with no LLM). |
| `jev-common.ts` | - | - | Shared: env parsing, state rendering, decision-model client, timing, JSONL log. Not an extension. |
| `harbor_jev_pi.py` | - | - | Harbor agent (`harbor_jev_pi:JevPi`) that runs pi with any subset of these extensions. |
| `pilot_gemini.sh` | - | - | Small real-API pilot: toy coding tasks × {plain pi, tool gate + stop verifier} with a Gemini LLM and a local decision server; writes `runs/pilot/pi/summary.md`. |

The cells mirror the Python harness (`src/jevharness/harness/agent.py`): the question texts are the
same, and so are the defaults of `history_window` (8), `obs_truncate_chars` (2000), and
`s1_state_max_chars` (12000). `tool_params`, and the `replace` / `cascade` modes of `tool_select`, are
not implemented here: pi gives extensions no way to answer a model request without the LLM.

Every extension also has a shadow mode (`JEV_<SCOPE>_MODE=log`) that asks and logs without acting,
for collecting decision data on unmodified runs, and every extension fails open: if the decision
model errors or times out, pi behaves as it would without the extension.

## Install

Needs pi >= 0.99.0 (classifier models and virtual models; actionable `turn_end` /
`agent_before_settle` boundaries arrived in 0.87.0). Developed and tested against 0.99.2.

```bash
npm install -g @earendil-works/pi-coding-agent@0.99.2
export TYPESAFE_API_KEY=...          # default decision model: typesafe/jev-latest
export GEMINI_API_KEY=...         # or whatever provider the LLM uses
```

pi loads TypeScript extensions with jiti, so there is no build step. The files import
`./jev-common.ts`, so keep them together and load them by path (`-e`, alias `--extension`). Do not
copy the directory flat into `~/.pi/agent/extensions/`: pi would try to load `jev-common.ts` as an
extension and report that it has no factory.

## Run

```bash
# One extension
pi -e adapters/pi/jev-tool-gate.ts --model google/gemini-3.5-flash "Fix the failing test"

# Several: repeat -e. Handlers run in load order.
pi -e adapters/pi/jev-tool-gate.ts -e adapters/pi/jev-stop-verify.ts -e adapters/pi/jev-memory.ts \
   --model google/gemini-3.5-flash

# The router is a model: select it
JEV_ROUTER_STRONG=google/gemini-3.5-flash JEV_ROUTER_CHEAP=google/gemini-3.5-flash-lite \
  pi -e adapters/pi/jev-router.ts --model s1/router

# Non-interactive, as Harbor runs it. Redirect stdin: when it is not a TTY, pi waits to read a
# piped prompt from it (src/main.ts readPipedStdin), and a script without </dev/null can hang.
JEV_LOG=runs/pi-decisions.jsonl pi --print --mode json -e adapters/pi/jev-stop-verify.ts \
  --model google/gemini-3.5-flash "Say hello using bash." </dev/null
```

Combining notes:

- All seven extensions can be loaded together; they use separate env scopes, so each can have its own
  tau (`JEV_GATE_TAU`, `JEV_STOP_TAU`, ...) next to a shared `JEV_TAU`.
- `jev-active-tools` and `jev-tool-gate` compose: the gate only sees calls to tools the LLM was shown.
  Calls to pruned tools fail in pi as unknown tools (which is part of what assist mode trades off).
- `jev-stop-verify` (guard) and `jev-reflect` both append messages at turn boundaries; they chain, since
  each returns `[...event.entries, draft]`.
- Changing the declared tools (`jev-active-tools`) or the model (`jev-router`) mid-session can cost
  prompt-cache hits. Both only change something when the decision changes; the router additionally has
  `JEV_ROUTER_MIN_DWELL` and `JEV_ROUTER_GRANULARITY=user`.
- The router is listed as `s1/router`, not `jev/...`: Harbor treats `<PROVIDER>_BASE_URL` as the LLM
  endpoint of an unknown provider, so a `jev` provider would collide with `JEV_BASE_URL`.

## Choosing the decision model

The model is swappable without code changes.

**Through pi's model registry** (default). `JEV_MODEL=provider/id` names a classifier model, and the
extension calls `ctx.modelRegistry.classify()` with pi's own credentials:

| Decision model | Setting |
|---|---|
| TypeSafe Jev (hosted) | `JEV_MODEL=typesafe/jev-latest` (default), `TYPESAFE_API_KEY` |
| Jev via OpenRouter | `JEV_MODEL=openrouter/typesafe/jev-1.13`, `OPENROUTER_API_KEY` |
| Jev via Cloudflare / Vercel / OpenCode | `cloudflare-workers-ai/typesafe/jev`, `vercel-ai-gateway/typesafe-ai/jev`, `opencode/jev-1.13` |
| A llama.cpp chat model read through logprobs | `JEV_MODEL=llama.cpp/<model-id>`, `LLAMA_BASE_URL`; `JEV_TEMPERATURE=2` softens overconfident label probabilities |

**Directly over the Jev wire format**. `JEV_BASE_URL` bypasses the registry and POSTs
`{state, model, questions}` to `{JEV_BASE_URL}/v1/systemone` exactly as `src/jevharness/decision/jev.py`
does (bool questions sent as `noul`, retries on 408/429/5xx/529 with exponential backoff):

```bash
# OpenJev / openjev-server
JEV_BASE_URL=http://localhost:8080 JEV_WIRE_MODEL=openjev-latest JEV_EXTRA_BODY='{"samples":8,"steps":4}' pi -e ...
# Our own trained decision model
python -m jevharness.training.serve_jev --model Qwen/Qwen3-4B --adapter checkpoints/jev4b-sft --port 8081
JEV_BASE_URL=http://localhost:8081 JEV_WIRE_MODEL=ours-4b pi -e ...
```

The direct path also records score-level distributions in `s1_dist` (pi's `classify()` returns only
the expected score). An OpenJev server can also go through the registry by overriding the TypeSafe
base URL in `~/.pi/agent/models.json` (`{"providers": {"typesafe": {"baseUrl": "http://host:8080/v1"}}}`,
with any non-empty `TYPESAFE_API_KEY`).

## Environment variables

Every setting `JEV_<NAME>` can be overridden for one extension as `JEV_<SCOPE>_<NAME>`, where the
scope is `GATE`, `TOOLS`, `STOP`, `ROUTER`, `MEMORY`, `REFLECT`, or `COMPACT`.

Shared (`jev-common.ts`):

| Variable | Default | Meaning |
|---|---|---|
| `JEV_MODEL` | `typesafe/jev-latest` | Classifier model in pi's registry, `provider/id` |
| `JEV_BASE_URL` | unset | Jev-compatible server; set to bypass the registry |
| `JEV_WIRE_MODEL` | id part of `JEV_MODEL` | `model` field sent to `JEV_BASE_URL` |
| `JEV_API_KEY` | `OPENJEV_API_KEY`, then `TYPESAFE_API_KEY` | Bearer token for `JEV_BASE_URL` |
| `JEV_EXTRA_BODY` | unset | JSON object merged into each request (e.g. OpenJev `samples`, `steps`, `think`) |
| `JEV_TIMEOUT_MS` / `JEV_MAX_RETRIES` | `10000` / `2` | Per request |
| `JEV_TEMPERATURE` | unset | Logit temperature for logprob classifiers (ignored by System One) |
| `JEV_TAU` | `0.5` | Threshold for guard / trigger decisions (the Python `DPConfig` default is 0.8; set it explicitly when comparing harnesses) |
| `JEV_HISTORY_WINDOW` / `JEV_OBS_CHARS` / `JEV_STATE_MAX_CHARS` | `8` / `2000` / `12000` | State rendering, as in `HarnessConfig` |
| `JEV_LOG` | unset (no log) | JSONL decision log path |
| `JEV_LOG_STATE` | `0` | Also log the rendered state (training-data harvest) |

Per extension (`off` disables any of them):

| Extension | Variables (defaults) |
|---|---|
| tool gate | `JEV_GATE_MODE=guard\|log`, `JEV_GATE_QUESTION=bool\|choice` (allow/ask/block), `JEV_GATE_TOOLS=` (all), `JEV_GATE_MAX_BLOCKS=1` (consecutive blocks before the next proposal runs unverified, like `guard_max_retries`) |
| active tools | `JEV_TOOLS_MODE=assist\|log`, `JEV_TOOLS_K=3`, `JEV_TOOLS_KEEP=` (always declared), `JEV_TOOLS_METHOD=choice\|score`, `JEV_TOOLS_CANDIDATES=` (default: tools active at session start) |
| stop verify | `JEV_STOP_MODE=guard\|trigger\|log`, `JEV_STOP_MAX_NUDGES=2` (per user prompt), `JEV_STOP_NUDGE=` (custom text) |
| router | `JEV_ROUTER_STRONG=google/gemini-3.5-flash`, `JEV_ROUTER_CHEAP=google/gemini-3.5-flash-lite`, `JEV_ROUTER_MODE=trigger\|log`, `JEV_ROUTER_GRANULARITY=request\|user`, `JEV_ROUTER_MIN_DWELL=1`, `JEV_ROUTER_ERROR_WINDOW=4`, `JEV_ROUTER_ERROR_ESCALATE=2`, `JEV_ROUTER_FALLBACK=strong`, `JEV_ROUTER_PROVIDER=s1`, `JEV_ROUTER_ID=router`, `JEV_ROUTER_CONTEXT_WINDOW=200000`, `JEV_ROUTER_MAX_TOKENS=64000` |
| memory | `JEV_MEMORY_WRITE_MODE=trigger\|always\|log`, `JEV_MEMORY_READ_MODE=assist\|always\|log`, `JEV_MEMORY_K=3`, `JEV_MEMORY_RECENCY_K=20`, `JEV_MEMORY_MAX_NOTES=500`, `JEV_MEMORY_FILE=` (JSONL shared across sessions) |
| reflect | `JEV_REFLECT_MODE=trigger\|always\|log`, `JEV_REFLECT_EVERY_TURN=0`, `JEV_REFLECT_REPEAT_WINDOW=6`, `JEV_REFLECT_COOLDOWN=2`, `JEV_REFLECT_PROMPT=` |
| compact | `JEV_COMPACT_MODE=prefilter\|extractive\|log`, `JEV_COMPACT_MIN_SCORE=1.0` (expected level 0-3), `JEV_COMPACT_BATCH=32`, `JEV_COMPACT_MESSAGE_CHARS=800`, `JEV_COMPACT_MIN_MESSAGES=4` |

`always` modes are the paper's non-S1 references (store every result, inject the most recent notes,
reflect on every failure or repeat).

## Decision log

With `JEV_LOG` set, every decision appends one line:

```json
{"ts": "2026-09-30T20:14:10.309Z", "harness": "pi", "session": "01a0f3f4-...", "extension": "jev-tool-gate",
 "dp": "tool_select", "mode": "guard",
 "question": "The agent proposes to call `bash` with arguments {\"command\":\"rm -rf build\"}. Is this the right next step toward completing the task?",
 "options": null, "s1_value": false, "s1_conf": 0.7, "s1_dist": {"yes": 0.3, "no": 0.7},
 "final_value": "block", "tau": 0.5, "latency_ms": 7.3, "input_tokens": 124, "model": "jev-1.13.0",
 "request_id": "wh0f61kt", "n_questions": 1, "tool": "bash", "s1_verdict": "block"}
```

- `s1_value` is the model's own answer (bool: `p_yes >= 0.5`; choice: the argmax; score: the expected
  level), `s1_dist` its distribution, and `final_value` what the harness did, so logs can be
  re-thresholded offline.
- Batched requests (memory writes, compaction, per-tool scores) produce one record per question with a
  shared `request_id`, `latency_ms`, and `input_tokens`; deduplicate by `request_id` before summing.
- Records with `model: null` made no model call (retry or nudge limits, the router's error rule).
- `error` is present when the decision model failed; the harness then fell back to plain pi.
- `JEV_LOG_STATE=1` adds the rendered `state` (on the first record of each request).

pi's own session JSONL also records the extensions' effects: `custom_message` entries
`jev-stop-verify`, `jev-stop-trigger`, and `jev-reflect`, `custom` entries `jev-memory` (notes) and
`pi.virtual-model-state` (router tier), and the physical model on every assistant message.

## Terminal-Bench 2.0 via Harbor

Harbor's built-in `pi` agent (`harbor/agents/installed/pi.py`) installs pi with npm and runs
`pi --print --mode json --session-dir /logs/agent/pi/sessions --provider P --model M ... "<instruction>"`,
but only adds its own `--extension` flags. `harbor_jev_pi.py` subclasses it: it uploads the selected
extension files to `/tmp/harbor-pi-jev/` in the task container, appends one `--extension` flag each,
and points `JEV_LOG` at `/logs/agent/jev-decisions.jsonl` so the decisions are downloaded with the
trial's agent logs (next to pi's session and Harbor's ATIF `trajectory.json`).

Run from the repository root (Harbor needs Python >= 3.12 and Docker, or `--env daytona` etc.). Install Harbor from git
at the commit the smokes used, `pip install "harbor @ git+https://github.com/laude-institute/harbor@4f848e367e6e2627c2c0b1826bb3b20ceab1198f"`
(pyproject 0.23.0, 2026-09-30): the PyPI `harbor==0.23.0` release rejects `--ak max_turns` and writes no ATIF
`agent/trajectory.json` for pi (found by the 2026-10-02 hard-task smoke, `results/hard/NOTES.md`):

```bash
export GEMINI_API_KEY=... TYPESAFE_API_KEY=...

# Baseline: plain pi through the same wrapper
PYTHONPATH=adapters/pi harbor run -d terminal-bench@2.0 -a harbor_jev_pi:JevPi \
  -m google/gemini-3.5-flash --ak version=0.99.2 -n 4

# Stop verifier + tool gate
PYTHONPATH=adapters/pi harbor run -d terminal-bench@2.0 -a harbor_jev_pi:JevPi \
  -m google/gemini-3.5-flash --ak version=0.99.2 --ak jev_extensions=stop-verify,tool-gate \
  --ae TYPESAFE_API_KEY="$TYPESAFE_API_KEY" --ae JEV_STOP_TAU=0.5 --ae JEV_GATE_TAU=0.3 -n 4

# Router: select the virtual model; credentials for the routed models go through --ae
PYTHONPATH=adapters/pi harbor run -d terminal-bench@2.0 -a harbor_jev_pi:JevPi \
  -m s1/router --ak version=0.99.2 --ak jev_extensions=router \
  --ae GEMINI_API_KEY="$GEMINI_API_KEY" --ae TYPESAFE_API_KEY="$TYPESAFE_API_KEY" \
  --ae JEV_ROUTER_STRONG=google/gemini-3.5-flash --ae JEV_ROUTER_CHEAP=google/gemini-3.5-flash-lite -n 4

# Our own decision model served where the containers can reach it
PYTHONPATH=adapters/pi harbor run -d terminal-bench@2.0 -a harbor_jev_pi:JevPi \
  -m google/gemini-3.5-flash --ak version=0.99.2 --ak jev_extensions=stop-verify \
  --ae JEV_BASE_URL=http://<reachable-host>:8081 --ae JEV_WIRE_MODEL=ours-4b -n 4
```

- `jev_extensions` takes any subset of `tool-gate, active-tools, stop-verify, router, memory, reflect,
  compact`; empty runs plain pi. All other `--ak` options of Harbor's `pi` agent work (`thinking`,
  `max_turns`, `model_api` for custom endpoints, ...).
- Pin `--ak version=0.99.2` (or newer): the wrapper refuses versions older than 0.99.0.
- `-a module:Class` is Harbor's current import-path syntax (`--agent-import-path` is deprecated).
- The Harbor `pi` MCP adapter is limited to pi < 0.86, so tasks that configure MCP servers cannot use
  these extensions.
- Every `JEV_*` variable and decision-model key must be passed with `--ae`; the container does not see
  your shell environment. For the router, Harbor resolves no credentials for the `s1` provider, so pass
  the routed models' keys with `--ae` too.
- Terminal-Bench prompts are one long user turn, so pi compacts them as a split turn; `jev-compact`
  handles this (it scores `turnPrefixMessages` as well as `messagesToSummarize`).

### Launcher, summary, and restricted networks

`run_tb2.sh` runs one Harbor job per arm with the flags below. The arms are `baseline`, `jev_shadow`
(gate and stop verifier in log mode), `jev_gate`, `jev_stop`, `jev_gate_stop`, `jev_reflect` (reflection
trigger after a failed or repeated call) and `jev_stop_reflect`. `MAX_TURNS` (`--ak max_turns`) and
`AGENT_TIMEOUT_MULT` (`--agent-timeout-multiplier`) cap the agent; a capped run is a budget deviation.
`summarize_harbor.py` prints one row per trial: reward, tests passed, pi's LLM cost and tokens, agent
steps, phase and trial wall-clock, and S1 requests, blocks, nudges, reflections, latency and estimated S1
cost from `jev-decisions.jsonl`. It then prints per-arm statistics (success, p50/p90 wall-clock, cost per
solved trial, `--budgets` "solved by t") and, with `--baseline <arm>`, paired per-task deltas against
that arm. The S1 cost is **not** in pi's cost, so pass `--s1-price-per-m`.

```bash
# decision server on the host (Gemini stand-in until Jev is reachable); containers reach it via host-gateway
python -m jevharness.decision.server --host 0.0.0.0 --port 8090 --name gemini-s1 --cache cache/s1_pi_gemini.sqlite \
  --s1 '{"type":"llm_generative","llm":{"provider":"gemini","model":"gemini-3.5-flash-lite"},"reasoning_effort":"minimal"}' &
HARBOR=<py3.12 venv>/bin/harbor N_CONCURRENT=4 adapters/pi/run_tb2.sh baseline jev_shadow jev_gate_stop   # full TB2
python adapters/pi/summarize_harbor.py runs/tb2/baseline runs/tb2/jev_shadow runs/tb2/jev_gate_stop --s1-price-per-m 0.30
```

The L3 smoke ran behind an egress proxy that blocks some hosts and intercepts TLS. These are the
blockers and their workarounds; none applies on a host with open egress.

| Blocker | Symptom | Workaround |
|---|---|---|
| Harbor 0.23 resolves `-d terminal-bench@2.0` through its hosted registry (`*.supabase.co`), which was blocked | `Dataset 'terminal-bench' (version: '2.0') not found` | `tb2_sandbox_tasks.sh <dir> [task ...]` copies tasks from the pinned git commit in Harbor's `registry.json` (`laude-institute/terminal-bench-2@69671fb`); run with `TASKS_DIR=<dir>` (`harbor run -p`). |
| TLS interception: task containers do not trust the proxy CA | the nvm / Node / npm downloads during pi's install fail certificate checks; so do pi's Gemini calls and the verifier's downloads | `EXTRA_CA=/path/to/extra-ca-bundle.crt`: the bundle is bind-mounted into the container (`--mounts-json`) and exported as `NODE_EXTRA_CA_CERTS`, `CURL_CA_BUNDLE`, `GIT_SSL_CAINFO` and `SSL_CERT_FILE` to pi's install and run (`--ae`, which Harbor also applies to agent setup) and to the verifier (`--ve`). |
| `astral.sh` blocked | every TB2 verifier runs `curl ... astral.sh/uv/<ver>/install.sh \| sh`, so uv is missing and the reward is 0 | `tb2_sandbox_tasks.sh` (default `PATCH_UV=1`) replaces only that line with the same uv release from GitHub. Results on patched tasks are a sandbox deviation. |
| Debian mirrors (`deb.debian.org`) blocked; the Ubuntu archive is reachable | `python:3.13-slim-bookworm` tasks cannot `apt-get install curl`, which pi's installer and the verifiers need | Only Ubuntu-based tasks run here (for example `prove-plus-comm`, `overfull-hbox`), so the sandbox cannot run the full suite. Use open egress, or `--env daytona` / `modal`. |
| Decision server on the host | `127.0.0.1` inside the container is the container | `harbor_host_gateway.yaml` (`--extra-docker-compose`) maps `host.docker.internal` to the host gateway, and the server binds `0.0.0.0` |
| Anonymous Docker Hub pulls | intermittent `429 Too Many Requests` / `502` | pre-pull the task images (`docker_image` in each `task.toml`) with retries, or `docker login` |

### L3 smoke (2026-10-01)

pi 0.99.2 with Gemini 3.5 Flash (`--thinking minimal`) on two Ubuntu-based TB2 tasks (patched uv
install). Arms: `baseline` and `jev_gate_stop` (`tool-gate` + `stop-verify`, τ = 0.5, Gemini Flash-Lite
stand-in behind `decision.server`). Outputs are in `runs/dev/l3_smoke/tb2/jobs/`.

```bash
TB2_CLONE=<clone> adapters/pi/tb2_sandbox_tasks.sh runs/dev/l3_smoke/tb2/tasks prove-plus-comm overfull-hbox
HARBOR=<harbor venv>/bin/harbor TASKS_DIR=runs/dev/l3_smoke/tb2/tasks EXTRA_CA=/path/to/extra-ca-bundle.crt \
  OUT=runs/dev/l3_smoke/tb2/jobs N_CONCURRENT=2 adapters/pi/run_tb2.sh baseline jev_gate_stop
```

| arm | task | reward | pi LLM $ | LLM input tokens | pi install s | agent run s | verifier s | S1 requests | gate blocks | S1 ms/request |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | prove-plus-comm | 1 | 0.078 | 45,521 | 26 | 24 | 8 | – | – | – |
| baseline | overfull-hbox | 0 (agent timeout 750 s) | 0.676 | 1,291,463 | 35 | 750 | 34 | – | – | – |
| jev_gate_stop | prove-plus-comm | 1 | 0.024 | 13,019 | 31 | 14 | 8 | 6 | 2 | 702 |
| jev_gate_stop | overfull-hbox | 0 (agent timeout 750 s) | 0.655 | 1,058,837 | 31 | 750 | 33 | 69 | 19 | 813 |

- **End to end on the agent side.**
  - Harbor installed pi 0.99.2 in the task container: apt `curl`, nvm, Node 22, npm, about 30 s.
  - pi called Gemini 3.5 Flash directly, using `GEMINI_API_KEY` from `--env-file`.
  - The extensions reached the host decision server through `host.docker.internal`. There were no S1
    errors, and the stop verifier asked once on `prove-plus-comm` and accepted.
  - Harbor downloaded `jev-decisions.jsonl`, pi's session and the ATIF `trajectory.json` with the trial.
  - The patched verifier installed uv from GitHub and ran pytest. On `overfull-hbox` it ran 3 of 4 tests
    passed, and the remaining overfull boxes failed the task.
- **Failure mode: timeouts are expensive.** Both `overfull-hbox` runs looped over synonym edits and
  `pdflatex` until Harbor's 750 s agent timeout, at about $0.66 each with no prompt-cache reads reported
  by pi's `google` provider. That is 10× a solved task. Budget for it, or cap with `--ak max_turns=<n>`.
- **The stand-in gate is noisy.** It blocked 19 of 69 proposals on `overfull-hbox`, and on
  `prove-plus-comm` it blocked `coqc` before the fix (defensible) and a final `ls -la`. Each block costs
  one extra LLM turn. S1 cost estimated from input tokens at the Flash-Lite price
  (`summarize_harbor.py --s1-price-per-m 0.30`): $0.0013 and $0.027 per trial. That excludes the
  stand-in's output tokens; Jev bills input tokens only.
- **Wall clock:** about 90 s per solved task. A timed-out task takes 750 s plus about 70 s for setup and
  the verifier. Two concurrent trials ran fine on 4 CPUs.
- **Full scale** (E9: 89 tasks × arms; on a host with open egress, so no `EXTRA_CA` and no patched tasks):
  ```bash
  HARBOR=<harbor bin> N_CONCURRENT=8 adapters/pi/run_tb2.sh baseline jev_shadow jev_gate_stop
  ```
  At this smoke's rates, budget about $0.1 for each solved task and $0.7 for each timed-out one, so
  roughly $25–50 per arm per pass with Gemini 3.5 Flash. Image pulls total several GB, and pulls hit
  Docker Hub rate limits, so `docker login` first.

## Terminal-Bench-Science via Harbor

[Terminal-Bench-Science](https://github.com/harbor-framework/terminal-bench-science) (TB-science)
is a Harbor benchmark of expert-written research workflows. It is our hard L3 suite, where the goal is
for Jev to make the agent both faster (wall clock) and better (success). The same `harbor_jev_pi:JevPi`
agent and the same arms run it. `run_tbsci.sh` is `run_tb2.sh` with TB-science defaults.

### What it is (pinned 2026-10-01)

- **Dataset.** Harbor Hub package `terminal-bench-science/terminal-bench-science@0.1.0`: public, tags
  `0.1.0` / `v0.1` / `latest`, revision 10, published 2026-08-26. It has 70 tasks. By directory:
  earth 8, engineering 9, life 19, mathematical 17, physical 17. Ignore `[metadata] domain`: its
  spelling varies between tasks.
- **Git.** We pin `cfff307` (2026-09-28). All 70 `tasks/dataset.toml` digests equal the Hub's 0.1.0
  digests. The content of 67 tasks hashes to them. Three tasks changed after the release:
  `rv-astrometry-fitting`, `variable-star-vetting` and `inverse-waveguide-shape`.
- **`lite` set.** Empty at the pin: only `lite/README.md`. It is meant for well-built tasks that are
  too easy for frontier models, it is not published, and it has no tasks to run.
- **Task format.** Each task has `task.toml`, `instruction.md`, `environment/Dockerfile` (the agent
  image), `tests/Dockerfile` + `tests/test.sh` (the verifier image), `solution/` and a README.
  `tbsci.py survey <clone>` prints the per-task facts.

| | Terminal-Bench 2.0 | Terminal-Bench-Science 0.1.0 |
|---|---|---|
| Layout | flat `<task>/` | nested `tasks/<domain>/<field>/<task>/`. Harbor's `-p` scans one level only, so use `-d` or flatten with `tbsci_tasks.sh` |
| Images | prebuilt (`docker_image`), pulled | built locally from Dockerfiles (`build_timeout_sec` 300–7,200 s; 40 tasks ≤ 900 s) |
| Verifier | same container; `test.sh` installs uv from astral.sh at run time | `environment_mode = "separate"`: its own image from `tests/Dockerfile` with the tests baked in, `network_mode = "no-network"`; files listed in `artifacts` are copied over from the agent container |
| Agent budget | per task, minutes (e.g. 750 s) | 8 h (28,800 s) for every task, stated in the instruction |
| Verifier timeout | minutes | 120–12,000 s (46 tasks ≤ 600 s; 16 tasks > 1,800 s) |
| Resources | small | 1–4 CPUs, 2–16 GB RAM, 2–50 GB disk; no GPUs at the pin (`gpus = 0` for all 70) |
| Network | open | agent phase `public` (open internet is benchmark policy); verifier none |
| Data | in the image | no git LFS in the task tree. 12 tasks download inputs or models at image build time from Hugging Face (dataset `harborframework/terminal-bench-science-lfs`); others ship data in `environment/` |
| Base images | mostly Ubuntu | final agent image Debian-family (`python:*-slim`, `node:*-bookworm`) for 47 tasks, Ubuntu-family for 23; 5 tasks add compose sidecars |
| Reward | binary | binary (pytest → `reward.txt`), per-test results in `ctrf.json` |
| Expert estimate | – | 1.6–600 h (median 12 h) |

### Run

```bash
# decision server on the host: real Jev through OpenRouter (or the Gemini stand-in, as for TB2)
python -m jevharness.decision.server --host 0.0.0.0 --port 8090 --name jev-1.13 --cache cache/s1_pi_jev.sqlite \
  --s1 '{"type":"jev","provider":"openrouter"}' &

# open egress, published dataset from Harbor Hub (the default); caps: 1 h agent time, 100 turns
# 1) calibration over the 37-task hard pool: validity + floor ($0), then one baseline attempt (see "Hard subset")
HARBOR=<py3.12 venv>/bin/harbor SUBSET=hard_pool OUT=runs/tbsci_cal N_CONCURRENT=4 \
  adapters/pi/run_tbsci.sh oracle nop baseline
# 2) evaluation on the frozen list (tbsci_hard.txt), 3 attempts; nop first warms the build cache
HARBOR=<py3.12 venv>/bin/harbor JEV_WIRE_MODEL=jev-1.13 SUBSET=hard N_ATTEMPTS=3 N_CONCURRENT=4 \
  adapters/pi/run_tbsci.sh nop baseline jev_stop_reflect jev_gate_stop
python adapters/pi/summarize_harbor.py runs/tbsci/baseline runs/tbsci/jev_stop_reflect runs/tbsci/jev_gate_stop \
  --s1-price-per-m 0.042 --baseline baseline --budgets 300,900,1800,3600

# pinned git commit instead of the Hub: flatten first (@all, @hard_pool, @sandbox, or slugs)
TBSCI_CLONE=<clone at cfff307> adapters/pi/tbsci_tasks.sh runs/tbsci/tasks @hard_pool
TASKS_DIR=runs/tbsci/tasks ... adapters/pi/run_tbsci.sh ...
```

- `run_tbsci.sh` defaults:
  - `DATASET=terminal-bench-science/terminal-bench-science@0.1.0`, `OUT=runs/tbsci`, `N_CONCURRENT=2`.
  - `AGENT_TIMEOUT_MULT=0.125`, which caps the 8 h budget at 1 h. Set it empty or to 1 for the official
    budget.
  - `MAX_TURNS=100`. Set it empty for no turn cap. See the budget below.
  - `EXTRA_CA_VERIFIER=0`.
  - `SUBSET` (`hard_pool`, `sandbox`, `hard`, or a file) adds one `-i` per slug and picks the
    `terminal-bench-science/<slug>` names that Hub tasks use.
- Every other variable is `run_tb2.sh`'s: arms (including `oracle` and `nop`), `EXTRA_CA`, `JEV_*_TAU`,
  `HARBOR_ARGS`. Relative paths are taken from the current directory.
- Any cap is a budget deviation: the instruction still tells the agent it has 28,800 s. Apply the same
  caps to every arm and report them.
- Harbor builds the images per trial and removes them with the containers (`down --rmi local`). The
  BuildKit cache makes later builds fast: in the smoke, environment setup took 38–79 s cold and 8–9 s
  warm. So arm order biases the trial wall clock. Compare arms on the agent-run time, run `nop`
  (or `oracle`) first to warm the cache, and keep the build cache between arms. Images were
  0.2–0.6 GB in the smoke. Tasks request 2–50 GB of disk (`storage_mb`).

### Hard subset for the paper and how to measure "faster and better"

**Selection.** There are two stages, both fixed before any evaluation run.

1. **Static pool (`tbsci_hard_pool.txt`, 37 tasks).** Generated by `tbsci.py list <clone> hard_pool`; the
   rule is `HARD_POOL_RULES`. A task is in the pool when all of these hold:
   - it is published and needs no GPU;
   - it needs at most 4 CPUs, 16 GB RAM and 20 GB disk;
   - its image builds in at most 30 min and its verifier runs in at most 30 min;
   - its expert estimate is 4–24 h.

   The pool has life 11, physical 11, earth 5, engineering 5 and mathematical 5 tasks, with a median
   expert estimate of 8.5 h. The rule excludes 33 tasks:
   - 16 have long verifiers (up to 3.3 h). They cost throughput, not API dollars. Relaxing the cap to
     2 h adds 9 of them back.
   - 12 have long image builds or large disks (Lean/Mathlib, Julia/Rust toolchains, PyTorch, large HF
     data).
   - 6 have expert estimates under 4 h, and 9 over 24 h (up to 600 h). Every arm would likely score 0
     on the very long ones, and floor effects carry no signal.
   - Some tasks fail more than one rule.
2. **Calibration → `tbsci_hard.txt` (about 20 tasks).** Use a separate `OUT` for these runs.
   - **Validity and floor, no API cost:** `SUBSET=hard_pool adapters/pi/run_tbsci.sh oracle nop`.
     - Drop a task if `oracle` (the reference solution) does not reach reward 1 in our environment.
     - Also drop it if the oracle's own run time exceeds half the agent cap; otherwise raise the cap
       for every arm. Reference solutions can be heavy: the `inelastic-constitutive-discovery` oracle
       computed for 32 min on its 2 CPUs.
     - `nop` gives each task's floor: the verifier tests that pass with no agent output at all. In the
       smoke, `ode-law-discovery` passed 4 of 12 tests without writing any file. So a raw
       `tests_passed > 0` means nothing; always compare against the floor.
   - **One capped baseline attempt** per remaining task (`baseline`, `N_ATTEMPTS=1`). Keep a task when
     the baseline solved it or passed more tests than its `nop` floor.

   Fill up to about 20 tasks, balanced across domains, using the lowest expert estimates. Then freeze
   the list (with the git commit) before the evaluation runs.
   - **Exact rule and tool (PREREGISTRATION A6.22 (d) 2).** `python adapters/pi/tbsci.py select runs/tbsci_cal
     --clone <clone at cfff307>` applies validity (oracle reward 1 within 1,800 s; a build / verifier failing
     twice drops the task), floor, signal, the fill rule (20 tasks, 4 per domain, lowest expert estimate first,
     spare slots round-robin earth → physical) and writes `tbsci_hard.txt` with the pool hash, the calibration
     run id and the commit in its header; it prints the file's sha256 and size for the Deviations section. It
     reads the Harbor results only, refuses to freeze while a task's only trial of a stage is an
     infrastructure error (retry it once), and never overwrites an existing list.
   - **Selection bias.** The calibration attempts are never reused in the evaluation. Every arm,
     including the baseline, is rerun on the frozen list, so regression to the mean affects all arms
     equally.

**Arms.** The arms target the two failure modes we see on hard tasks: looping until the timeout, and
declaring done with missing or invalid outputs.

| Arm | Role |
|---|---|
| `baseline` | plain pi |
| `jev_stop_reflect` | **primary**. Stop verifier (guard): rejects premature "done" (better). Reflection trigger: breaks failure loops early (faster) |
| `jev_gate_stop` | comparable with the E9 TB2 arm |
| `jev_shadow` (optional) | gate and stop in log mode, for calibration data; the same behaviour as baseline |
| `oracle`, `nop` | Harbor's reference-solution and do-nothing agents. They make no API calls and serve validity and floor checks; they are not compared |

**Metrics.** Every metric comes from `summarize_harbor.py` with `--baseline baseline`.

- **Better.**
  - Success = verifier reward 1, per trial and as the per-task mean over k = 3 attempts.
  - Secondary: verifier tests passed above the task's `nop` floor, a graded signal on hard tasks.
- **Faster.**
  - Agent-run wall clock (`agent_execution`; timeouts count at the cap): p50 and p90 per arm.
  - Trial wall clock including setup and the verifier.
  - Agent steps and S1 time (`s1_s_total`), because the S1 latency is part of the arm's cost in time.
- **Cost.**
  - LLM $ from pi's usage plus S1 $ (Jev input tokens × $0.042/M).
  - Cost per solved trial.
- **Paired, per task** (attempt means first, then arm − baseline over the common tasks):
  - task wins and losses on success, with an exact sign test;
  - median Δ agent-run time over all tasks, and over the tasks both arms solved;
  - median Δ cost.
- **One figure for both claims:** "solved by t" (`--budgets`), the fraction of trials solved within t
  seconds of agent time, per arm.
- **The claim criterion.** "Faster and better" means all three of these:
  - wins ≥ losses on success;
  - solved-by-t at least as high as the baseline at every budget;
  - median Δ agent-run time < 0 on the tasks both arms solved.

  Report the sign-test p and per-task bootstrap CIs. With about 20 tasks × 3 attempts, the success
  test is underpowered for small effects, so lead with the solved-by-t curve and the paired time
  deltas.

**Budget per arm (Gemini 3.5 Flash, minimal thinking; estimates from the dev smoke, not results).**
- **Why the cost is quadratic.** pi compacts only near the model's context window, which is 1M tokens
  for Gemini. So the prompt grows by about 500–1,000 tokens per turn on these tasks, and the input
  tokens of an N-turn trial are about 5k·N + b·N²/2, with b ≈ 500–1,000. The dev smoke measured this
  at 20 and 40 turns.
- **Effective price.** With cache reads (38–82% of input), including output, the price is about
  $0.6 per million input tokens.
- **Turns.** The turn cap binds: no smoke run finished on its own. One turn took 2–10 s, so 100 turns
  is 3–16 min of agent time, and the 1 h cap rarely binds before the turn cap.
- **Jev.** About 2.3–3.1k input tokens and 0.5–0.7 s per request, which is ≤ $0.01 per trial.

| `MAX_TURNS` | input tokens per trial | $ per trial |
|---|---|---|
| 60 | 1.2–2.1M | 0.7–1.3 |
| 100 (default) | 3.0–5.5M | 1.8–3.3 |
| 150 | 6.3–11.9M | 3.8–7.2 |
| 200 | 11–21M | 6.6–12.5 |

| Run (worst case: every trial hits 100 turns) | Trials | API $ | Wall clock |
|---|---|---|---|
| Calibration: hard pool × `oracle` + `nop` | 74 | $0 | build + reference solution + verifier per task |
| Calibration: hard pool × 1 attempt, `baseline` | 37 | ≤ $125 | about 37 × 5–20 min / concurrency |
| Evaluation: hard subset (20) × 3 attempts, per arm | 60 | ≤ $200 per arm | 1.5–5 h per arm at concurrency 4 |
| Full suite, 70 tasks × 1 attempt, per arm | 70 | ≤ $235 per arm | 1.5–6 h at concurrency 4 |
| Official 8 h budget, no turn cap | – | unbounded in practice: 8 h is thousands of turns | days |

For a smaller budget, use `MAX_TURNS=60`, k = 2 attempts, or 15 tasks. A compaction threshold well
below 1M tokens would make the cost linear in turns; `jev-compact` is the System-1 arm for that, and it
was not run here.

### Restricted networks (the dev sandbox), on top of the TB2 table

| Blocker | Symptom | Workaround |
|---|---|---|
| TLS interception during image builds: TB-science builds images, and TB2 pulled them | `pip install` / `uv` / `curl` in the Dockerfiles fail certificate checks; `EXTRA_CA` only reaches running containers | `PATCH_CA=/path/to/extra-ca-bundle.crt tbsci_tasks.sh <dir> ...` (`tbsci.py patch-ca`) inserts, after every `FROM`, a `COPY` of the bundle plus `PIP_CERT` / `SSL_CERT_FILE` / ... `ENV`s. **Sandbox deviation**: the bundle and variables stay in the agent and verifier images |
| Debian mirrors blocked (HTTP and HTTPS, every mirror tried) | 27 tasks `apt-get` on Debian-family images at build time; on 47 tasks pi's installer would `apt-get install curl` | only Ubuntu-family tasks run here: `tbsci_sandbox.txt` (12 by heuristic) |
| Container egress differs from the host's: from containers, `huggingface.co`, `download.pytorch.org`, Posit/CRAN, Julia, rustup, `astral.sh` and `openrouter.ai` are blocked | HF-data tasks (12) cannot build; the pi extensions cannot call OpenRouter directly | run the decision server on the host (`JEV_BASE_URL=http://host.docker.internal:8090`) |
| Separate verifiers get no mounts or compose overlays (Harbor warns on every trial) | `--ve SSL_CERT_FILE=<mounted path>` would point at a missing file | `run_tbsci.sh` sets `EXTRA_CA_VERIFIER=0`; the verifiers have no network anyway |
| A missing artifact (the agent never wrote the file) | `Engine cp failed ... Could not find the file` in `trial.log` | expected: Harbor logs it and the verifier then fails the schema test; this is not an infrastructure error |

### Dev smoke (2026-10-01, DEV, not paper numbers)

pi 0.99.2 with Gemini 3.5 Flash (`--thinking minimal`). System-1 was **real Jev**
(`typesafe/jev-1.13` via OpenRouter) behind the host decision server. The tasks were CA-patched local
copies (a sandbox deviation). Caps: 900 s of agent time (`AGENT_TIMEOUT_MULT=0.03125`) and
`MAX_TURNS` = 20 or 40. Outputs are in `runs/dev/tbsci_smoke/`.

```bash
TBSCI_CLONE=<clone> PATCH_CA=/path/to/extra-ca-bundle.crt adapters/pi/tbsci_tasks.sh runs/dev/tbsci_smoke/tasks \
  inelastic-constitutive-discovery ode-law-discovery
HARBOR=<harbor venv>/bin/harbor TASKS_DIR=runs/dev/tbsci_smoke/tasks INCLUDE=inelastic-constitutive-discovery \
  EXTRA_CA=/path/to/extra-ca-bundle.crt OUT=runs/dev/tbsci_smoke/jobs N_CONCURRENT=1 MAX_TURNS=20 \
  AGENT_TIMEOUT_MULT=0.03125 JEV_WIRE_MODEL=jev-1.13 adapters/pi/run_tbsci.sh baseline jev_stop_reflect
```

Then `ode-law-discovery` with `OUT=runs/dev/tbsci_smoke/jobs_ode MAX_TURNS=40`, and `oracle nop` on
both tasks with `OUT=runs/dev/tbsci_smoke/jobs_floor`.

| arm | task (turn cap) | reward | tests passed (`nop` floor) | pi LLM $ | input tokens (cached) | env setup s | pi install s | agent run s | verifier s | trial s | S1 requests | reflections fired | S1 ms/request |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | inelastic-constitutive-discovery (20) | 0 | 0/4 (0/4) | 0.167 | 194k (113k) | 38 | 22 | 40 | 31 | 148 | – | – | – |
| jev_stop_reflect | inelastic-constitutive-discovery (20) | 0 | 0/4 (0/4) | 0.155 | 128k (48k) | 8 | 22 | 48 | 20 | 115 | 1 | 1 | 492 |
| baseline | ode-law-discovery (40) | 0 | 4/12 (4/12) | 0.579 | 1,018k (836k) | 79 | 21 | 385 | 70 | 572 | – | – | – |
| jev_stop_reflect | ode-law-discovery (40) | 0 | 4/12 (4/12) | 0.412 | 572k (422k) | 9 | 20 | 234 | 30 | 310 | 3 | 2 | 576 |

- **`oracle` (reference solution, $0, cap `AGENT_TIMEOUT_MULT=0.125` = 1 h).**
  - `inelastic-constitutive-discovery`: reward 1 (4/4 tests) after 1,902 s of compute, so the patched
    environment and the separate verifier work.
  - `ode-law-discovery`: no result within 1 h (`AgentTimeoutError` at 3,600 s; the 4-CPU host was shared
    with the other oracle and pytest). At a 1 h cap this task fails the calibration rule; it needs a
    larger cap or must be dropped.
  - `nop` floors: 0/4 and 4/12.
- **Spend.**
  - Gemini (pi's reported cost): $1.31 for the four agent trials.
  - Jev: $0.00047, measured as the change in the OpenRouter key's `usage` and covering every probe; the
    summarizer's token estimate at $0.042/M is $0.00045.
  - `oracle` and `nop` cost nothing.
- **The pipeline works end to end with real Jev.**
  - Harbor built both CA-patched images, ran pi's install (nvm, Node 22, about 20 s), and ran the agent
    with the 900 s and turn caps.
  - The `jev-reflect` and `jev-stop-verify` extensions reached the host decision server through
    `host.docker.internal`, and from there real Jev on OpenRouter. There were no S1 errors.
  - The declared artifacts went to the separate no-network verifier image, which scored them.
  - `jev-decisions.jsonl`, pi's session and `trajectory.json` were downloaded with each trial.
- **What Jev did.**
  - `reflect` asked 4 times, each time after a failed or repeated call: `p_yes` was 0.89, 0.91, 0.87 and
    0.24. It fired 3 times (τ = 0.5), and the reflection prompt reached the agent as a custom message.
  - The stop verifier never asked, because the turn cap ended every run before the agent tried to stop.
    For the stop verifier to act on these tasks, the cap has to leave room to finish.
- **Nothing was solved, and no arm effect can be read from n = 1 per cell.**
  - The Jev arm used fewer tokens and less agent time on `ode-law-discovery`: −150 s and −$0.17 on the
    same 40 turns. That is one paired task, not evidence.
  - The baseline trials also built their images cold (38–79 s of environment setup, versus 8–9 s warm),
    which is why we compare on agent-run time.
- **`tests_passed` needs the `nop` floor.** On `ode-law-discovery`, both arms "passed" 4/12 tests
  without writing any output file. The same 4 pass with no agent at all. The calibration rule uses the
  floor for this reason.
- **Relative `OUT` caused failures.** With a relative `OUT`, Harbor's `docker compose cp` resolved
  paths against the task's environment dir and failed (`invalid output path`). Harbor recovered through
  engine `cp` and a tar stream. `run_tb2.sh` now makes `OUT`, `TASKS_DIR`, `ENV_FILE` and `EXTRA_CA`
  absolute.
- **Harmless warnings.** Harbor warns on every trial that mounts and compose overlays do not reach
  separate verifiers. A declared artifact the agent never wrote shows up as `Engine cp failed ... Could
  not find the file` before the verifier fails its schema test.

## pi APIs used

References are to the pi repository at 0.99.2 (`packages/coding-agent/...` unless noted).

- Extension factory, `pi.on()` result types: `src/core/extensions/types.ts` (`ExtensionAPI` at 1540,
  handler signature 1535). A throwing `tool_call` handler blocks the tool (`docs/extensions.md` 256),
  so the gate catches its own errors.
- `tool_call` -> `{ block, reason }`: `ToolCallEventResult` at types.ts 1402; nested calls carry
  `parentToolCallId` (1150).
- `turn_end` / `agent_before_settle` boundaries: `BoundaryState` / `BoundaryResult` at types.ts
  967-977, `TurnEndEvent` 1015, `AgentBeforeSettleEvent` 980; entries are replaced, not appended, per
  handler (`src/core/extensions/runner.ts` 1020-1062), hence `[...event.entries, draft]`; continuation
  needs runnable context (`src/core/agent-session.ts` 965-993, 1820-1843).
- Active tools: `pi.getActiveTools()` / `pi.setActiveTools()` / `pi.getAllTools()` at types.ts
  1711-1723. pi snapshots the active tools when it prepares the next request, after `turn_end`
  (`packages/agent/src/agent-loop.ts` 184-207, agent-session.ts 864-878), and reads
  `setActiveTools()` changes at `before_agent_start` (agent-session.ts 1994-2000).
- `context` event (request-local messages): types.ts 851, `ContextEventResult` 1391.
- Virtual models: `pi.registerVirtualModel()` types.ts 1856; `ModelRouteRequest` / `ModelRoute` /
  state semantics in `src/core/virtual-models.ts` 52-99 and `docs/virtual-models.md`; the official
  `examples/extensions/jev-router.ts`.
- Classification: `ctx.modelRegistry.findOfType()` / `classify()` in `src/core/model-registry.ts`
  71-78 and 167-174; question and answer types in `packages/ai/src/types.ts` 633-690; TypeSafe wire
  mapping (`bool` <-> `noul`) in `packages/ai/src/api/system-one-shared.ts` 80-158 and
  `typesafe-system-one.ts` (POST `{baseUrl}/systemone`, catalog base URL `https://api.typesafe.ai/v1/`).
- Compaction: `SessionBeforeCompactEvent` types.ts 751, `CompactionPreparation` in
  `src/core/compaction/compaction.ts` 772; the same preparation object goes to the default compaction
  (agent-session.ts 2719-2760 and 3051-3095), which `prefilter` relies on.
- Session state: `ctx.sessionManager.buildSessionProjection()` / `getBranch()` / `getSessionId()`
  (`ReadonlySessionManager`, `src/core/session-manager.ts` 245); `convertToLlm` turns `custom`
  messages into user messages (`src/core/messages.ts` 148).
- Relative imports between extension files are resolved by jiti (`src/core/extensions/loader.ts`
  552-583), as in `examples/extensions/plan-mode/index.ts`.

## Validation

- Type-checked (`tsc --noEmit`, strict, `erasableSyntaxOnly`) against the published
  `@earendil-works/pi-coding-agent@0.99.2` typings, together with pi's official `jev-router.ts` and
  `permission-gate.ts` examples; formatted and linted with pi's own biome configuration.
- Run end to end with the published pi 0.99.2 CLI (`--print --mode json`) against a scripted
  OpenAI-compatible mock LLM and a mock Jev server: every extension, both backends (pi's registry
  `classify()` with the TypeSafe provider pointed at the mock, and `JEV_BASE_URL`), the router via
  `--model s1/router` and via Harbor's `--provider s1 --model router`, auto-compaction in all three
  compaction modes, and fail-open with an unreachable decision server.
- `harbor_jev_pi.JevPi` exercised with Harbor's own code (agent factory import path, option
  validation, `install()` uploads, and the rendered `pi` command) without Docker.

- Pilot with real models (`pilot_gemini.sh`; results in `docs/PILOT_RESULTS.md`): pi 0.99.2
  `--print --mode json` with Gemini 3.5 Flash (pi's `google` provider, `--thinking minimal`) and
  `jev-tool-gate` + `jev-stop-verify` against the Gemini Flash-Lite stand-in served by
  `python -m jevharness.decision.server` (`JEV_BASE_URL`). Two toy coding tasks, both arms solved both,
  and no decision-model errors. Behind the agent proxy, Node's `fetch` needs `NODE_USE_ENV_PROXY=1`, with
  `NO_PROXY=127.0.0.1,localhost` for the local decision server.
- The pilot found a state-rendering bug, now fixed in `renderState`. pi appends the assistant message
  before running its tool calls, so at `tool_call` the call under review was already the last `history`
  step, with no result. The decision model read it as already done and rejected correct calls: the
  gate blocked the correct one-line fix with p_yes ≈ 1e-6. Calls without a result are now left out of
  `history`; the gate passes the call as `proposed_call`.

- L3 smoke (2026-10-01): Docker Terminal-Bench 2.0 trials through Harbor 0.23 (`harbor_jev_pi:JevPi`),
  baseline and `tool-gate` + `stop-verify`, with real Gemini models; see
  [L3 smoke](#l3-smoke-2026-10-01).

- TB-science dev smoke (2026-10-01): 2 Terminal-Bench-Science tasks × `baseline` / `jev_stop_reflect`
  with **real Jev** (`typesafe/jev-1.13` via OpenRouter) behind the host decision server, plus `oracle`
  and `nop`; see [Dev smoke](#dev-smoke-2026-10-01-dev-not-paper-numbers).

Not validated here: OpenJev answers, the stop verifier and tool gate with real Jev (the TB-science smoke
only triggered `jev-reflect`), and the interactive (TUI) paths such as the gate's `ask` confirmation.
