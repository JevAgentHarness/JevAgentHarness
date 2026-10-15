#!/usr/bin/env bash
# Live smoke test of the OpenAI Responses provider (src/jevharness/llm/openai_responses.py). Needs OPENAI_API_KEY in
# .env; prints usage, cost and client stats, never the key. Spend: a few cents.
#   bash scripts/smoke_openai.sh                               # MAIN=gpt-6.1-sol  LITE=gpt-6-luna
#   LITE=gpt-5.4-nano MAIN=gpt-5.4-mini bash scripts/smoke_openai.sh
# Steps: 0 key + connectivity, 1 one tool-calling turn (System-2), 2 one verbalised decision (System-1, any model),
# 3 a logprob probe (needs reasoning_effort none), 4 a previous_response_id round trip (wire vs billed tokens).
set -uo pipefail
cd "$(dirname "$0")/.."
LITE=${LITE:-gpt-6-luna}
MAIN=${MAIN:-gpt-6.1-sol}
MAIN_EFFORT=${MAIN_EFFORT:-low}     # gpt-6.1-sol accepts neither none nor minimal
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python
fail=0

echo "== 0. key + connectivity"
"$PY" - <<'EOF' || { echo "!! add OPENAI_API_KEY=... to .env (never commit it) and re-run"; exit 1; }
import os
from dotenv import load_dotenv
load_dotenv(".env")
k = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEYS", "")
print("  OPENAI_API_KEY(S):", "set" if k else "MISSING")
raise SystemExit(0 if k else 1)
EOF
code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 15 https://api.openai.com/v1/models 2>/dev/null || true)
echo "  https://api.openai.com/v1/models -> ${code:-000} (401 without a key is fine; 000 = host blocked, allow-list api.openai.com)"

echo "== 1. one tool-calling turn on $MAIN (reasoning $MAIN_EFFORT) + usage/cost"
"$PY" - "$MAIN" "$MAIN_EFFORT" <<'EOF' || fail=1
import json, sys
from dotenv import load_dotenv
load_dotenv(".env")
from jevharness.accounting import Ledger
from jevharness.llm import make_llm

model, effort = sys.argv[1], sys.argv[2]
llm = make_llm({"provider": "openai_responses", "model": model, "reasoning_effort": effort})
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}}]
led = Ledger()
msgs = [{"role": "system", "content": "You are a terse assistant. Use tools when they apply."},
        {"role": "user", "content": "What is the weather in Paris right now?"}]
r = llm.chat(msgs, tools=tools, ledger=led, decision_point="smoke")
print("  turn 1:", [(t.name, t.arguments) for t in r.tool_calls] or r.text[:100], "| finish:", r.finish_reason,
      "| reasoning items:", len((r.tool_calls[0].extra or {}).get("openai", {}).get("reasoning", [])) if r.tool_calls else 0)
if r.tool_calls:
    tc = r.tool_calls[0]
    msgs += [{"role": "assistant", "content": None, "tool_calls": [tc.to_openai()]},
             {"role": "tool", "tool_call_id": tc.id, "content": "18C, light rain"}]
    r2 = llm.chat(msgs, tools=tools, ledger=led, decision_point="smoke")
    print("  turn 2:", r2.text[:120].replace("\n", " "))
s = led.summary()
print("  ledger total:", json.dumps(s["total"]), "| unpriced_models:", s["unpriced_models"])
print("  client stats:", json.dumps({k: v for k, v in llm.stats.items() if v}))
EOF

echo "== 2. verbalised decision on $LITE (reasoning none) via jevbench ping"
jevbench ping --s1 "{\"type\": \"llm_generative\", \"llm\": {\"provider\": \"openai_responses\", \"model\": \"$LITE\"}, \"reasoning_effort\": \"none\"}" \
  || { echo "!! verbalised decider failed"; fail=1; }

echo "== 3. logprob probe on $LITE (reasoning none; a LogprobsUnavailableError here means: use the verbalised decider)"
jevbench ping --s1 "{\"type\": \"llm_logprob\", \"llm\": {\"provider\": \"openai_responses\", \"model\": \"$LITE\", \"reasoning_effort\": \"none\"}}" \
  || echo "!! logprob decider failed on $LITE"
"$PY" - "$LITE" <<'EOF'
import sys
from dotenv import load_dotenv
load_dotenv(".env")
from jevharness.llm import make_llm
from jevharness.llm.openai_responses import LogprobsUnavailableError

llm = make_llm({"provider": "openai_responses", "model": sys.argv[1], "reasoning_effort": "none"})
try:
    d = llm.option_logprobs([{"role": "system", "content": "Answer with the option letter only."},
                             {"role": "user", "content": "Which one is a fruit?\nA. carrot\nB. apple\nC. bread"}],
                            ["A", "B", "C"], temperature=0)
    print("  option logprobs at position 0:", d)
except LogprobsUnavailableError as e:
    print("  no logprobs:", e)
EOF

echo "== 4. previous_response_id round trip on $LITE: wire (sent) vs billed input tokens"
"$PY" - "$LITE" <<'EOF' || fail=1
import json, sys
from dotenv import load_dotenv
load_dotenv(".env")
from jevharness.accounting import Ledger
from jevharness.llm import make_llm

llm = make_llm({"provider": "openai_responses", "model": sys.argv[1], "reasoning_effort": "none", "server_state": True})
led = Ledger()
msgs = [{"role": "system", "content": "Answer in one short sentence."}, {"role": "user", "content": "Name a prime number."}]
r1 = llm.chat(msgs, ledger=led)
msgs += [r1.assistant_message(), {"role": "user", "content": "And the next one after it?"}]
r2 = llm.chat(msgs, ledger=led)
for rec in led.records:
    print("  call:", {"chained": rec.meta["chained"], "wire_input_tokens_est": rec.meta["wire_input_tokens_est"],
                      "billed_input_tokens": rec.input_tokens, "cached": rec.meta["cached_input_tokens"]})
print("  stats:", json.dumps({k: v for k, v in llm.stats.items() if v}))
EOF
echo "done (fail=$fail). Prices used: configs/prices/models.yaml (rows marked 'verify' need a check against the pricing page)."
exit $fail
