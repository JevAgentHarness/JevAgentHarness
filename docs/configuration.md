# Configuration reference

`jah serve` reads one YAML file: the one given with `-c`, else `~/.jah/config.yaml` if it exists, else the built-in
starter config. `jah init` writes the starter config to `~/.jah/config.yaml` with comments. The starter config is the
setting of the README results: Gemini 4 Argon (strong), Gemini 3.5 Flash-Lite (fast), Jev, `tau: 0.75`.

```bash
jah init                                   # writes ~/.jah/config.yaml (use --force to overwrite)
jah serve                                  # picks up ~/.jah/config.yaml; or: jah serve -c other.yaml
```

Single values can be overridden on the command line with `--set dotted.key=<YAML value>` (repeatable; missing
parents are created):

```bash
jah serve -c ~/.jah/config.yaml --set router.tau=0.75 --set budget_usd=20 \
    --set tiers.fast.model=gemini-3.5-flash-lite --set 'router.decision={type: rules}'
```

`--host`, `--port` and `--log-dir` override `listen.host`, `listen.port` and `log_dir`, and `--preset` overrides
`preset`. Keys are never written in the config: it names environment variables (`key_env`), and the proxy reads them
at startup.

A complete example with every section:

```yaml
listen: {host: 127.0.0.1, port: 8787}
log_dir: ~/.jah/logs
budget_usd: null
log_states: false

upstreams:
  gemini: {type: gemini, base_url: "https://generativelanguage.googleapis.com/v1beta", key_env: GEMINI_API_KEY}
  openrouter: {type: openai, base_url: "https://openrouter.ai/api/v1", key_env: OPENROUTER_API_KEY,
               effort_field: reasoning}
  vllm: {type: openai, base_url: "http://localhost:8000/v1", key_env: ""}

tiers:
  strong: {upstream: gemini, model: gemini-4-argon, effort: high}
  fast:   {upstream: gemini, model: gemini-3.8-flash, effort: low, cache: explicit}
  lite:   {upstream: gemini, model: gemini-3.5-flash-lite, effort: low}

router:
  decision: {type: jev, provider: openrouter}
  tau: 0.5
  first_steps_strong: 1
  stick_strong: 0
  max_fast_streak: 0
  verify_finish: true
  on_error: strong
  decision_timeout_s: 8

preset: null                  # route | paper | safe | shadow (docs/modes.md)
decisions:                    # decision points x modes; unset points keep the router's behaviour
  tool_select: {mode: shadow}
  action: {mode: guard}

models:
  jah-auto: {policy: route}
  jah-auto-t75: {policy: route, tau: 0.75}
  jah-auto-lite: {policy: route, fast: lite}
  jah-auto-tools: {policy: route, decisions: {routing: off, tool_select: {mode: cascade, tau: 0.97}}}
  jah-strong: {tier: strong}
  jah-fast: {tier: fast}
default_model: jah-auto

prices:
  my-finetune: {input: 0.20, output: 0.80, cached_input: 0.02}
```

## Top level

| Key | Default | Meaning |
|---|---|---|
| `listen.host` | `127.0.0.1` | interface to bind. The proxy has no authentication and spends your keys: bind elsewhere only on a trusted network |
| `listen.port` | `8787` | port |
| `log_dir` | `~/.jah/logs` | where `steps.jsonl` (and `states.jsonl`) are written |
| `budget_usd` | `null` | spend cap in USD, see [Budget](#budget) |
| `log_states` | `false` | also write the decision model's input state per step to `states.jsonl`, see [Logs](#logs) |
| `upstreams` | | the LLM providers, see [Upstreams](#upstreams) |
| `tiers` | | named LLM settings (upstream + model + effort), see [Tiers](#tiers) |
| `router` | | the routing policy, see [Router](#router) |
| `preset` | `null` | a set of decision-point modes: `route` (the default behaviour), `paper`, `safe`, `shadow`; see [Decision points](#decision-points) |
| `decisions` | `{}` | the mode and settings of each decision point (`routing`, `tool_select`, `stop`, `reflection`, `action`); see [Decision points](#decision-points) |
| `models` | `{jah-auto: {policy: route}}` | the model names clients may ask for, see [Model names](#model-names) |
| `default_model` | `jah-auto` | the name used for an unknown model name; must be a key of `models` |
| `prices` | `{}` | added or corrected token prices, see [Prices](#prices) |

## Upstreams

An upstream is one provider endpoint. Tiers refer to upstreams by name, and the two tiers may use different
providers.

| Key | Default | Meaning |
|---|---|---|
| `type` | required | `gemini` (Gemini's native `generateContent` API), `openai` (any OpenAI-compatible Chat Completions endpoint: OpenAI, OpenRouter, DeepSeek, vLLM, SGLang, ...) or `openai_responses` (the OpenAI Responses API) |
| `base_url` | required | e.g. `https://generativelanguage.googleapis.com/v1beta`, `https://api.openai.com/v1`, `https://openrouter.ai/api/v1` |
| `key_env` | `""` | environment variable that holds the key; empty for a local server without a key |
| `effort_field` | `reasoning_effort` | `openai` only: the request field for the reasoning effort. `reasoning` sends OpenRouter's `{"reasoning": {"effort": ...}}` |
| `headers` | `{}` | extra HTTP headers sent with every upstream request |
| `timeout_s` | `900` | timeout of one attempt, in seconds |
| `retries` | `3` | retries on HTTP 408, 409, 425, 429, 500, 502, 503, 504 and 529, with exponential backoff |

How each type sends a request:

- `gemini`: `POST {base_url}/models/{model}:generateContent` with the key in `x-goog-api-key`. The effort becomes the
  thinking level (`minimal`, `low`, `medium`, `high`; `xhigh` maps to `high`, `none` to `minimal`). Gemini thought
  signatures are kept across turns and replaced by Google's documented placeholder when the router switches models.
- `openai`: `POST {base_url}/chat/completions` with `Authorization: Bearer <key>`. When the upstream reports the
  billed amount (`usage.cost`, as OpenRouter does), that amount is logged instead of a price-table estimate.
- `openai_responses`: `POST {base_url}/responses`, stateless (`store: false`) with the whole history each time and
  `prompt_cache_key` set to the session, so OpenAI's prompt cache keeps hitting. Encrypted reasoning items are kept
  per call and sent back to the model that produced them.

**Key pools.** `key_env` may hold several comma-separated keys, for example
`GEMINI_API_KEY=key-a,key-b,key-c`, to spread rate limits. Each session sticks to one key of the pool (Gemini context
caches belong to the project of the key that made them), and sessions spread evenly over the keys.

The proxy honours `HTTPS_PROXY` / `HTTP_PROXY` and a custom CA bundle in `SSL_CERT_FILE` or `REQUESTS_CA_BUNDLE`.

`jah serve` refuses to start when a key variable of an upstream that a tier uses is not set.

## Tiers

A tier is a model on an upstream with fixed settings. The router picks a tier for every step.

| Key | Default | Meaning |
|---|---|---|
| `upstream` | required | an upstream name |
| `model` | required | the provider's model id |
| `effort` | `null` | reasoning effort for this tier (`minimal`, `low`, `medium`, `high`, ...). When set, it replaces the effort the client asks for; when `null`, the client's own effort passes through |
| `extra` | `{}` | fields merged into the upstream request body, e.g. `{generationConfig: {temperature: 0}}` for Gemini or `{temperature: 0}` for an OpenAI-compatible upstream |
| `cache` | `implicit` | `explicit` (Gemini only): the proxy keeps one context cache per session, see below |

**Explicit context caching** (`cache: explicit`). An agent re-sends its whole history at every step. With an explicit
cache, the proxy stores the system instruction, the tools and the history up to some turn as a Gemini
`cachedContents` entry per session, and sends later requests as that cache plus the new turns only, so the history is
billed at the cached-input price. The cache is refreshed in the background once the uncached tail has grown (at a
point that balances re-writing the cache against paying full price for the tail), and the cache it replaces is
deleted. Each cache write is logged as its own record (`kind: cache`) at the input price plus storage for the
10-minute TTL, and counted in the session's cost. If a cached request fails because the cache is gone, it is re-sent
in full. `implicit` relies on the provider's automatic caching.

## Router

| Key | Default | Meaning |
|---|---|---|
| `decision` | `{type: rules}` in code; `{type: jev, provider: openrouter}` in the starter config | the decision model, see [Decision models](#decision-models) |
| `tau` | `0.5` in code; `0.75` in the starter config | a step goes to the strong tier when p(hard) >= `tau`. Higher values send fewer steps to the strong tier. The routing decision point's threshold (`decisions.routing.tau`, when set, wins) |
| `first_steps_strong` | `1` | the first N steps of a session go to the strong tier (the agent reads the task and plans) |
| `stick_strong` | `0` | after the decision model escalates a step, keep the next N steps on the strong tier |
| `max_fast_streak` | `0` | when > 0, force a strong step after N fast steps in a row |
| `verify_finish` | `true` | when a fast-tier reply has no tool call (the agent wants to end its turn) and the request offered tools, the strong tier re-does that step. The same as `decisions.stop: strong` (`false`: `off`); a preset or `decisions.stop` wins |
| `on_error` | `strong` | the tier used when the decision model fails or times out |
| `decision_timeout_s` | `8` | time limit for one decision |
| `strong` | `strong` | the tier name that plays the strong role |
| `fast` | `fast` | the tier name that plays the fast role |

The router is the `routing` decision point in its default `cascade` mode. Its other modes (`guard`, `shadow`, `off`)
are set with `decisions.routing`; see [docs/modes.md](modes.md#routing).

A step's index is the number of assistant turns already in the request, so step 0 is the first request of an agent
session. The decision model is asked only for steps that could go to the fast tier (not for the first steps, a
stick-strong step or a forced strong step). A fast-tier call that fails after its retries is re-done once on the
strong tier. The `strong`, `fast` and `on_error` tiers must exist.

## Model names

`models` maps the model names a client may send to what the proxy does with them:

- `{policy: route}`: the router picks the tier per step.
- `{tier: <name>}`: every step on that tier (no decision model call); logged with reason `pinned`.

A routed name may override these router settings for its own requests: `tau`, `first_steps_strong`, `stick_strong`,
`max_fast_streak`, `verify_finish`, `on_error`, `strong` and `fast`. One proxy can therefore serve several variants
side by side, for example `jah-auto-t75: {policy: route, tau: 0.75}` or `jah-auto-lite: {policy: route, fast: lite}`.
The decision model and `decision_timeout_s` are shared by all names. Unknown keys are rejected at startup.

Any name may also set its own `preset` and `decisions` (merged over the top-level ones, point by point), for example
`jah-auto-safe: {policy: route, preset: safe}`. A pinned name (`tier:`) is a baseline: the top-level `preset` and
`decisions` do not apply to it, only its own (and it has no routing point). See
[docs/modes.md](modes.md#per-model-name).

Besides the names in `models`, `jah-<tier>` works for every tier (for example `jah-lite` pins the `lite` tier). Any
other name is served as `default_model`. `GET /v1/models` lists the accepted names.

## Decision points

Besides routing, the proxy can ask the decision model at four more decision points of each step, each in a mode that
says who decides: the next tool (`tool_select`), ending the turn (`stop`), pausing to re-check (`reflection`) and
risky calls (`action`). The defaults keep the behaviour of the router above, and every point has a `shadow` mode that
only logs. [docs/modes.md](modes.md) describes every mode, what the paper found for it, and how to evaluate it.

```yaml
preset: shadow                       # route | paper | safe | shadow
decisions:
  routing:     {mode: cascade}       # off | shadow | cascade | guard; tau defaults to router.tau
  tool_select: {mode: cascade, tau: 0.97, tier: fast, k: 5, min_tools: 12, max_options: 100, force: true}
  stop:        {mode: strong}        # off | shadow | strong | guard | trigger; tau 0.9, scope: fast | all
  reflection:  {mode: off}           # off | shadow | trigger | always; tau 0.8, every: 5
  action:      {mode: guard}         # off | shadow | guard; tau 0.5, prefilter: risky | command | all,
                                     # patterns: [...], tools: [...], on_error: accept | recheck
```

A string is a shorthand for the mode (`action: guard`). Unknown points, modes and keys are rejected at startup, with
the reason for points the proxy does not offer (`tool_params`, `memory_write`, `memory_read`). `--set` works on these
keys too, for example `--set decisions.tool_select.mode=shadow`.

`jah fit <log_dir>` fits the thresholds of `tool_select` and `stop` on logged runs and prints a config snippet; see
[docs/modes.md](modes.md#fitting-thresholds).

## Decision models

`router.decision` selects the backend:

| Config | Notes |
|---|---|
| `{type: jev, provider: openrouter}` | Jev through OpenRouter; key in `OPENROUTER_API_KEY`; default model `typesafe/jev-1.13` |
| `{type: jev, provider: typesafe}` | Jev from TypeSafe; key in `TYPESAFE_API_KEY`; default model `jev-latest` |
| `{type: jev, provider: local, base_url: http://host:8090, model: <name>}` | a self-hosted model that speaks the Jev wire format (below); no key by default |
| `{type: rules}` | no model, no key: `hard` when the last tool result shows an error or failing tests, or the agent repeats its last action. It also answers the reflection question (the same signals) and the action question (the risky patterns), but not the tool-choice or finish questions: those points then fall back to their LLM-only behaviour |
| `{type: none}` | never answers, so every routed step goes to the `on_error` tier |

Options of `type: jev`: `provider`, `model`, `base_url`, `key_env` (each overrides the provider's default),
`timeout_s` (HTTP timeout per attempt, default 8) and `retries` (default 2). `router.decision_timeout_s` caps the whole
decision, retries included.

`jah decide -c <config>` sends one example state to the configured backend and prints the probabilities, the cost and
the latency: a quick check of the key and the backend. `jah decide --state <file>` sends your own state text.

### The question and the state

For each step the decision model gets a text state and one multiple-choice question: is the agent's next step
`routine` or `hard`? The state has the task (the first user message, up to 3,000 characters), the number of steps so
far, and the last six steps, each an action (tool calls, up to 400 characters) and its result (up to 300 characters;
the latest result up to its last 2,500 characters). The exact question text is `QUESTION` in
`src/jevagentharness/router.py`. The other decision points ask their questions about the same state (with the
proposed reply, final answer or call appended for the checks after the call); their texts are in
`src/jevagentharness/points.py`.

### Jev wire format (self-hosting)

A self-hosted backend answers `POST {base_url}/v1/systemone`:

```json
{
  "state": "TASK: fix the failing test\n\nSTEPS SO FAR: 1\n[1] ACTION: bash({\"command\": \"pytest -q\"})\n    RESULT: 1 failed, 12 passed",
  "model": "<model>",
  "questions": {
    "next_step": {
      "type": "choice",
      "instructions": "You are watching an AI agent work on the task below, ...",
      "criteria": {"routine": "The next step is routine: ...", "hard": "The next step needs careful reasoning: ..."}
    }
  }
}
```

with a probability per option:

```json
{"answers": {"next_step": {"probabilities": {"routine": 0.18, "hard": 0.82}}},
 "usage": {"input_tokens": 412, "cost": 0.0000412},
 "model": "<model>"}
```

An answer with only `"choice": "hard"` counts as probability 1 for that option. `usage.cost` is optional; without it
the cost is estimated at $0.10 per million input tokens. The key, when `key_env` is set, is sent as
`Authorization: Bearer <key>`.

## Prices

Costs are computed from a built-in table of list prices (`src/jevagentharness/prices.py`, with the date they were
read) unless the upstream reports the billed amount. Prices change: check them before you report numbers, and add or
correct models under `prices:` (USD per million tokens):

| Key | Meaning |
|---|---|
| `input` | input tokens |
| `output` | output tokens; thinking tokens are billed at this rate too |
| `cached_input` | cached input tokens (default: the input price) |
| `cache_write` | prompt-cache writes, as OpenAI bills them (default: the input price) |
| `cache_storage_per_hour` | explicit Gemini context caches, per million tokens per hour |

A model is looked up by its full id, then by the part after the last `/` (`google/gemini-3.8-flash` finds
`gemini-3.8-flash`). A call to a model with no price is logged with cost 0 and `priced: false`, and `jah report` counts
such calls as `unpriced_calls`.

## Budget

With `budget_usd` set, the proxy stops starting new agent sessions once the logged spend reaches the cap: the first
request of a new session gets HTTP 429, while sessions already running finish. At startup the proxy adds up the cost
of every record already in `<log_dir>/steps.jsonl`, so the cap covers everything logged in that directory, across
restarts. Use a fresh `log_dir` for a fresh budget. `GET /stats` shows the spend so far.

## Logs

`<log_dir>/steps.jsonl` gets one JSON line per agent step, and never contains prompt text, completion text, headers or
keys:

| Field | Meaning |
|---|---|
| `ts`, `session`, `model_name`, `step` | time, session key (`<model name>/<session>`), the client's model name, step index |
| `tier`, `reason` | the tier that wrote the step and why: `first_steps`, `routine`, `hard`, `stick`, `fast_streak`, `finish_check`, `fast_error`, `decision_error`, `pinned`; `repeat:<reason>` for a retried request |
| `p_hard`, `decision` | the decision model's p(hard), and its model, cost, latency and input tokens (or the error) |
| `calls` | every LLM call of the step: tier, model, latency, attempts, cost, `priced` |
| `llm_cost_usd`, `decision_cost_usd`, `cost_usd` | LLM cost, decision cost, and their sum |
| `latency_s` | wall-clock seconds for the step, decision included |
| `input_tokens`, `cached_tokens`, `cache_write_tokens`, `output_tokens`, `reasoning_tokens` | token counts of the final reply |
| `n_tool_calls`, `finish` | tool calls in the reply, finish reason |
| `finish_check`, `fast_error`, `error` | present when the finish check re-did the step, when a fast call failed, or when the step failed |

Records with `kind: cache` are context-cache writes (cost only, not steps). The reasons above are those of the
default decision points. Other modes add `tool_cascade`, `reflection`, `stop_trigger`, `stop_guard`, `tool_guard`,
`route_guard`, `action_guard`, `guard`, `shadow` and `routing_off`, and the fields `points` and `redo`
([docs/modes.md](modes.md#logs)). `decision_cost_usd` covers every decision call of the step.

`<log_dir>/decisions.jsonl` gets one line per decision call: the decision point and mode, the question, the options,
the probabilities, what the point did, the LLM's own choice when known, latency and cost (fields in
[docs/modes.md](modes.md#logs)). It holds no prompt text. `jah fit` reads it, and `jah report` prints a summary of it.

With `log_states: true` the proxy also writes `<log_dir>/states.jsonl`: per decided step, the exact state text the
decision model saw, with its p(hard), tier and reason, and one line per question of the other decision points (with
a `point` field). This is training data for your own decision model. It contains
task text and tool output, so it is off by default.

`jah report <log_dir>` summarizes `steps.jsonl` by model name (`--by <field>` for another field, `--json` for every
number) and, when `decisions.jsonl` exists, the decision calls per decision point and mode. `GET /stats` shows spend,
steps and tier shares of the running proxy, the default model name's decision points and the decision calls so far.
