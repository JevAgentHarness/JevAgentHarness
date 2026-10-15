# pi -> JevAgentHarness

pi talks to the proxy as an OpenAI-compatible provider. Write `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "jah": {
      "baseUrl": "{BASE}",
      "api": "openai-completions",
      "apiKey": "JAH_API_KEY",
      "models": [
        {"id": "jah-auto", "name": "JevAgentHarness (routed)", "reasoning": true, "contextWindow": 1000000, "maxTokens": 65536},
        {"id": "jah-strong", "name": "strong tier only", "reasoning": true, "contextWindow": 1000000, "maxTokens": 65536}
      ]
    }
  }
}
```

Then `export JAH_API_KEY=jah` and run `pi --provider jah --model jah-auto`.
