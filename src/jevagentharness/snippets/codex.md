# Codex CLI -> JevAgentHarness

Codex speaks the OpenAI Responses API; the proxy serves it at `{BASE}/responses`.
Add this to `~/.codex/config.toml` (or a separate `CODEX_HOME`):

```toml
model_provider = "jah"
model = "jah-auto"              # jah-strong / jah-fast pin one tier (baselines)
web_search = "disabled"         # hosted tools are not available through the proxy

[model_providers.jah]
name = "JevAgentHarness"
base_url = "{BASE}"
env_key = "JAH_API_KEY"         # any value: the proxy holds the real provider keys
wire_api = "responses"
stream_idle_timeout_ms = 600000

[agents]
enabled = false                 # optional: no multi-agent tools
```

Then:

```bash
export JAH_API_KEY=jah
codex exec --skip-git-repo-check "fix the failing test"
```
