# mini-SWE-agent (or anything built on litellm) -> JevAgentHarness

Save as `jah.yaml`:

```yaml
model:
  model_name: openai/jah-auto
  model_kwargs:
    api_base: "{BASE}"
    api_key: jah                  # any value: the proxy holds the real provider keys
  cost_tracking: ignore_errors    # litellm has no price for jah-*; the proxy logs the cost
```

Then (the first `-c` keeps mini-swe-agent's default config; the second is merged into it):

```bash
mini -c mini.yaml -c jah.yaml -t "fix the failing test"
```
