# Any OpenAI SDK client -> JevAgentHarness

```python
from openai import OpenAI
client = OpenAI(base_url="{BASE}", api_key="jah")
r = client.chat.completions.create(model="jah-auto", messages=[{"role": "user", "content": "hello"}])
```

Name the session explicitly (recommended for agents that run in parallel) with the header `X-JAH-Session: <id>`
or the base URL `.../s/<id>/v1`.
