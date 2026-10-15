# Harness setup

How to point an agent harness at JevAgentHarness for everyday use. For the benchmark runs, see
[`bench/README.md`](../bench/README.md). `jah setup <harness>` prints the short version of each snippet
(`codex`, `pi`, `dsh`, `mini-swe-agent`, `openai-sdk`); add `--base <url>` when the proxy is not on
`http://127.0.0.1:8787/v1`.

Every harness needs the same three things:

- **Base URL:** `http://127.0.0.1:8787/v1` (or the per-session form `http://127.0.0.1:8787/s/<session>/v1`, see
  [Sessions](#sessions)).
- **API key:** any value. The proxy ignores the client's key and uses its own provider keys, so no real key enters the
  harness or its sandbox. The examples use `JAH_API_KEY=jah`.
- **Model name:** `jah-auto` (routed), `jah-strong` (strong tier only, the baseline) or `jah-fast` (fast tier only).
  Any other name defined under `models:` in the config works too; an unknown name is served as `default_model`.

Start the proxy first (`jah serve`, or `jah serve -c ~/.jah/config.yaml` for your own config) and check it with
`jah ask "say hi"`.

Two notes apply to every harness:

- **Reasoning effort.** A tier with `effort` set uses that effort and ignores the one the harness asks for. Leave
  `effort` unset on a tier to pass the harness's own setting through.
- **HTTP proxies.** If your shell sets `HTTP_PROXY` / `HTTPS_PROXY`, add the proxy's host to `NO_PROXY`
  (for example `NO_PROXY=localhost,127.0.0.1`), or the harness may send local requests to the HTTP proxy.

## Codex CLI

Codex speaks the OpenAI Responses API, which the proxy serves at `/v1/responses`. Add a model provider to
`~/.codex/config.toml` (or to `$CODEX_HOME/config.toml` to keep it separate from your normal Codex setup):

```toml
model_provider = "jah"
model = "jah-auto"                  # jah-strong / jah-fast pin one tier
web_search = "disabled"             # hosted tools are not available through the proxy
model_catalog_json = "/absolute/path/to/jevagentharness/bench/codex_catalog.json"

[model_providers.jah]
name = "JevAgentHarness"
base_url = "http://127.0.0.1:8787/v1"
env_key = "JAH_API_KEY"             # any value: the proxy holds the real provider keys
wire_api = "responses"
stream_idle_timeout_ms = 900000     # the proxy streams keep-alives while a strong model thinks
request_max_retries = 4
stream_max_retries = 3

[agents]
enabled = false                     # optional: no multi-agent tools

[analytics]
enabled = false                     # optional
```

```bash
export JAH_API_KEY=jah
codex exec --skip-git-repo-check "fix the failing test"     # or plain `codex` for the interactive UI
```

**Why the model catalog.** Codex chooses its system prompt, its tools and its context settings from metadata for the
model slug. `jah-auto` is not a slug Codex knows, so without a catalog Codex falls back to generic unknown-model
defaults; among other things it does not offer its freeform `apply_patch` tool. `bench/codex_catalog.json` gives the
names `jah-auto`, `jah-strong`, `jah-fast` and `jah-auto-m` the metadata Codex uses for GPT-5.5 (its prompts, the
freeform `apply_patch` tool, a 272k-token context window and the reasoning levels), so Codex behaves as it does with
an OpenAI model and only the endpoint changes. The catalog needs Codex 0.124.0 or newer; the benchmarks used 0.162.1.
For another model name, copy the `jah-auto` entry and change `slug` and `display_name`.

Codex sends a session id with every request, so each Codex conversation is one proxy session without further setup.
Codex's namespaced tools (MCP servers, multi-agent tools) work through the proxy; hosted tools such as `web_search`
are not offered to the upstream model.

## pi

pi talks to the proxy as an OpenAI-compatible provider. Write `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "jah": {
      "baseUrl": "http://127.0.0.1:8787/v1",
      "api": "openai-completions",
      "apiKey": "JAH_API_KEY",
      "models": [
        {"id": "jah-auto", "name": "JevAgentHarness (routed)", "reasoning": true, "contextWindow": 1000000, "maxTokens": 65536},
        {"id": "jah-strong", "name": "strong tier only", "reasoning": true, "contextWindow": 1000000, "maxTokens": 65536},
        {"id": "jah-fast", "name": "fast tier only", "reasoning": true, "contextWindow": 1000000, "maxTokens": 65536}
      ]
    }
  }
}
```

```bash
export JAH_API_KEY=jah
pi --provider jah --model jah-auto "fix the failing test"
```

Set `contextWindow` and `maxTokens` to what your tiers support (the values above fit the Gemini defaults). When you
run several pi sessions on the same task at once, give each its own base URL (`http://127.0.0.1:8787/s/<name>/v1`),
so the proxy does not mistake them for one session (see [Sessions](#sessions)).

## DeepSeek Harness (`dsh`)

dsh reaches non-DeepSeek models through its pi-ai adapter, configured with a Cordis patch. Save as `jah.patch.yml`:

```yaml
- id: llm-pi-ai
  config:
    providers:
      jah:
        api: openai-completions
        baseURL: "http://127.0.0.1:8787/v1"
        apiKeyEnv: JAH_API_KEY
        compat: {supportsStore: false}
        models:
          - {id: jah-auto, contextWindow: 1000000, maxTokens: 65536}
          - {id: jah-strong, contextWindow: 1000000, maxTokens: 65536}
          - {id: jah-fast, contextWindow: 1000000, maxTokens: 65536}
- id: agent-default-model
  config: {provider: jah, model: jah-auto}
# Optional, as in the benchmark runs:
- {id: session-title-llm, disabled: true}     # saves one extra LLM call per session
- {id: tool-web, disabled: true}              # the web tools need a DeepSeek key
- {id: web-search-deepseek, disabled: true}
- {id: web-fetch-http, disabled: true}
```

```bash
export JAH_API_KEY=jah
dsh --profile headless --patch jah.patch.yml "fix the failing test" --json
dsh --profile headless --patch jah.patch.yml --dump-config     # check the composed config
```

The benchmark agent (`bench/harbor_agents.py`, `dsh_patch` and `Dsh`) writes the same patch with the per-trial session
URL, and also sets `DSH_TELEMETRY_MODE=DISABLED`. It runs dsh with `DSH_PERMISSION_MODE=danger-full-access`, which
is only reasonable inside a throwaway container.

## mini-swe-agent

mini-swe-agent calls models through LiteLLM; the `openai/` prefix selects LiteLLM's OpenAI-compatible route. Either
set the endpoint in the environment:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
export OPENAI_API_BASE=$OPENAI_BASE_URL       # older LiteLLM versions read this name
export OPENAI_API_KEY=jah                     # any value
export MSWEA_COST_TRACKING=ignore_errors      # LiteLLM has no price for jah-*; the proxy logs the cost
mini -m openai/jah-auto -t "fix the failing test"
```

or in a config file merged into mini-swe-agent's default config (`jah setup mini-swe-agent` prints it):

```yaml
# jah.yaml
model:
  model_name: openai/jah-auto
  model_kwargs:
    api_base: "http://127.0.0.1:8787/v1"
    api_key: jah
  cost_tracking: ignore_errors
```

```bash
mini -c mini.yaml -c jah.yaml -t "fix the failing test"
```

Under Harbor (`bench/run_harbor.sh`, `HARNESS=mswe`) the dummy key goes in `MSWEA_API_KEY`, the key variable of
Harbor's mini-swe-agent agent, which also sets `MSWEA_COST_TRACKING=ignore_errors` itself.

## Terminus-2 and other OpenAI-compatible clients

Anything that speaks OpenAI Chat Completions or Responses works with a base URL and a dummy key.

Terminus-2 is Harbor's reference agent for Terminal-Bench and runs only under Harbor. It calls the model from the
Harbor process (through LiteLLM), so the proxy is on `localhost` from its point of view:

```bash
OPENAI_API_KEY=jah harbor run -d terminal-bench@2.0 -a terminus-2 -m openai/jah-auto \
    --ak api_base=http://localhost:8787/v1 \
    --ak 'model_info={"max_input_tokens":1048576,"max_output_tokens":65536,"input_cost_per_token":0,"output_cost_per_token":0}'
```

`model_info` tells LiteLLM the context size of the unknown `jah-*` name and sets its price to zero; the proxy's log has
the real cost. `bench/run_harbor.sh` (`HARNESS=t2`) runs the same agent with a per-trial session.

OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="jah")
r = client.chat.completions.create(model="jah-auto", messages=[{"role": "user", "content": "hello"}])
r = client.responses.create(model="jah-auto", input="hello")       # the Responses API works too
```

LiteLLM:

```python
import litellm

r = litellm.completion(model="openai/jah-auto", api_base="http://127.0.0.1:8787/v1", api_key="jah",
                       messages=[{"role": "user", "content": "hello"}])
```

curl:

```bash
curl -s http://127.0.0.1:8787/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "jah-auto", "messages": [{"role": "user", "content": "hello"}]}'
```

Both APIs work with and without `stream: true`. Streams are sent from the complete upstream reply, with keep-alive
comments while the model works, so set generous stream idle timeouts in the client.

## Sessions

The router keeps state per agent session: the decision already made for each step (a retried request is routed the
same way), the counters behind `stick_strong` and `max_fast_streak`, the explicit context cache, and the session's
cost in the log. The proxy finds the session for a request in this order:

1. the base URL `http://127.0.0.1:8787/s/<session>/v1` (every endpoint is also served under `/s/<session>/...`);
2. the header `X-JAH-Session: <session>`;
3. the client's own session id: for the Responses API the `prompt_cache_key` field or the `session-id`,
   `session_id` or `thread-id` header (Codex sends these); for Chat Completions the `user` field;
4. otherwise a hash of the system prompt and the first user message.

The model name is always part of the session key, so a `jah-auto` run and a `jah-strong` run of the same task are
separate sessions, also when they run at the same time. Name sessions explicitly when several agents run the same task
with the same prompt in parallel, or when you want to find a run in the log: the session name appears in
`steps.jsonl` as `<model name>/<session>`. Use letters, digits, `.`, `_` and `-` in session names.

## Checking a setup

```bash
curl -s localhost:8787/v1/models         # the model names the proxy accepts
curl -s localhost:8787/stats             # spend, steps, tier shares, sessions
jah report ~/.jah/logs                   # per model name: sessions, steps, strong share, cost and LLM time
```

In `steps.jsonl`, `reason` says why each step went where it did: `first_steps`, `routine`, `hard`, `stick`,
`fast_streak`, `finish_check`, `fast_error`, `decision_error` or `pinned` (a `jah-strong` / `jah-fast` step); a
retried request keeps its first route and gets the prefix `repeat:`. Many `decision_error` steps mean the decision
model is not reachable (check its key with `jah decide`); those steps run on the strong tier.
