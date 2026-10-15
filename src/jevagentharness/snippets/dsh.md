# DeepSeek Harness (dsh) -> JevAgentHarness

dsh reaches non-DeepSeek models through its pi-ai adapter. Save this Cordis patch as `jah.patch.yml`:

```yaml
- id: llm-pi-ai
  config:
    providers:
      jah:
        api: openai-completions
        baseURL: "{BASE}"
        apiKeyEnv: JAH_API_KEY
        compat: {supportsStore: false}
        models:
          - {id: jah-auto, contextWindow: 1000000, maxTokens: 65536}
          - {id: jah-strong, contextWindow: 1000000, maxTokens: 65536}
          - {id: jah-fast, contextWindow: 1000000, maxTokens: 65536}
- id: agent-default-model
  config: {provider: jah, model: jah-auto}
```

Then:

```bash
export JAH_API_KEY=jah
dsh --profile headless --patch jah.patch.yml "fix the failing test" --json
```

The proxy's tiers set the reasoning effort, so dsh's own effort setting is ignored. Check the composed config with
`dsh --profile headless --patch jah.patch.yml --dump-config`.
