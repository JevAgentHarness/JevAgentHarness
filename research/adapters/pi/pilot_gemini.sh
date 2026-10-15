#!/usr/bin/env bash
# PILOT (real APIs, small): pi end to end with a Gemini LLM and the stand-in System-1 server.
#
# Two toy coding tasks x two arms, each run in a fresh temp directory with `pi --print --mode json`:
#   baseline : plain pi (no extensions)
#   jev      : -e jev-tool-gate.ts -e jev-stop-verify.ts  (tool_select/guard + stop/guard, JEV_TAU=0.5)
# The decision model is reached over the Jev wire format at $JEV_BASE_URL. Until Jev itself is reachable
# (openrouter.ai), start the Gemini stand-in first:
#   python -m jevharness.decision.server --port 8090 --name gemini-s1 --cache cache/s1_pi_gemini.sqlite \
#     --s1 '{"type":"llm_generative","llm":{"provider":"gemini","model":"gemini-3.5-flash-lite"},"reasoning_effort":"minimal"}'
# then:  bash adapters/pi/pilot_gemini.sh            (writes runs/pilot/pi/{summary.md,summary.json,...})
#
# Env: PI (pi binary; default `pi` on PATH), MODEL (gemini-3.5-flash), PROVIDER (google), THINKING (minimal),
#      ARMS ("baseline jev"), TASKS ("fizzbuzz fixbug"), OUT (runs/pilot/pi), JEV_BASE_URL
#      (http://127.0.0.1:8090), JEV_WIRE_MODEL (gemini-s1), TIMEOUT (300 s per run), TRIALS (1).
# Keys: GEMINI_API_KEY is read from the environment or from .env (never printed).
# Node's fetch ignores HTTPS_PROXY unless NODE_USE_ENV_PROXY=1 (Node >= 22.21); NO_PROXY keeps the local
# decision server off the proxy.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PI=${PI:-pi}
MODEL=${MODEL:-gemini-3.5-flash}
PROVIDER=${PROVIDER:-google}
THINKING=${THINKING:-minimal}
ARMS=${ARMS:-"baseline jev"}
TASKS=${TASKS:-"fizzbuzz fixbug"}
TRIALS=${TRIALS:-1}
OUT=$(mkdir -p "${OUT:-$ROOT/runs/pilot/pi}" && cd "${OUT:-$ROOT/runs/pilot/pi}" && pwd)
TIMEOUT=${TIMEOUT:-300}
export JEV_BASE_URL=${JEV_BASE_URL:-http://127.0.0.1:8090}
export JEV_WIRE_MODEL=${JEV_WIRE_MODEL:-gemini-s1}
export JEV_TAU=${JEV_TAU:-0.5}
export JEV_LOG=${JEV_LOG:-$OUT/jev-decisions.jsonl}
if [ -z "${GEMINI_API_KEY:-}" ] && [ -f "$ROOT/.env" ]; then
  GEMINI_API_KEY=$(grep -E '^GEMINI_API_KEY=' "$ROOT/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')
  export GEMINI_API_KEY
fi
[ -n "${GEMINI_API_KEY:-}" ] || { echo "GEMINI_API_KEY not set" >&2; exit 2; }
curl -s --noproxy '*' -m 5 -o /dev/null "$JEV_BASE_URL/v1/systemone" -X POST -H 'content-type: application/json' \
  -d '{}' || echo "WARNING: no decision server at $JEV_BASE_URL (the jev arm will fail open = plain pi)" >&2

setup_task() {  # $1 task, $2 dir -> prints the prompt
  case $1 in
    fizzbuzz)
      echo "Create fizzbuzz.py that prints the numbers 1 to 15, one per line, but prints Fizz for multiples of 3, Buzz for multiples of 5 and FizzBuzz for multiples of both. Run it to check the output." ;;
    fixbug)
      cat > "$2/stats.py" <<'EOF'
def mean(xs):
    return sum(xs) / (len(xs) - 1)


def median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
EOF
      cat > "$2/test_stats.py" <<'EOF'
from stats import mean, median

assert mean([1, 2, 3, 4]) == 2.5, mean([1, 2, 3, 4])
assert median([3, 1, 2]) == 2
assert median([4, 1, 3, 2]) == 2.5
print("all tests passed")
EOF
      sha256sum "$2/test_stats.py" > "$2/.test_sha"
      echo "The tests in test_stats.py fail. Fix the bug in stats.py (do not modify test_stats.py) and run python3 test_stats.py to verify." ;;
  esac
}

check_task() {  # $1 task, $2 dir -> exit 0 on success
  case $1 in
    fizzbuzz)
      [ "$(cd "$2" && timeout 10 python3 fizzbuzz.py 2>/dev/null)" = "$(printf '1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz')" ] ;;
    fixbug)
      sha256sum -c --quiet "$2/.test_sha" >/dev/null 2>&1 && (cd "$2" && timeout 10 python3 test_stats.py >/dev/null 2>&1) ;;
  esac
}

AGENT_DIR=$(mktemp -d)   # empty pi agent dir: no user settings, models.json or auth leak into the pilot
for arm in $ARMS; do
  ext=()
  [ "$arm" = jev ] && ext=(-e "$ROOT/adapters/pi/jev-tool-gate.ts" -e "$ROOT/adapters/pi/jev-stop-verify.ts")
  for task in $TASKS; do
    for trial in $(seq 1 "$TRIALS"); do
      run="$arm-$task-$trial"
      work=$(mktemp -d)
      prompt=$(setup_task "$task" "$work")
      t0=$(date +%s.%N)
      (cd "$work" && NODE_USE_ENV_PROXY=1 NO_PROXY=127.0.0.1,localhost PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 \
        PI_TELEMETRY=0 PI_CODING_AGENT_DIR="$AGENT_DIR" timeout "$TIMEOUT" "$PI" --print --mode json --no-session \
        --no-extensions --no-skills --no-context-files --no-prompt-templates --provider "$PROVIDER" \
        --model "$MODEL" --thinking "$THINKING" ${ext[@]+"${ext[@]}"} "$prompt" </dev/null >"$OUT/$run.jsonl" 2>"$OUT/$run.err")
      rc=$?
      wall=$(python3 -c "import time; print(round(time.time() - $t0, 2))")
      if check_task "$task" "$work"; then ok=true; else ok=false; fi
      echo "{\"run\": \"$run\", \"arm\": \"$arm\", \"task\": \"$task\", \"trial\": $trial, \"exit\": $rc, \"success\": $ok, \"wall_s\": $wall, \"workdir\": \"$work\"}" >>"$OUT/runs.jsonl"
      echo "[pi-pilot] $run exit=$rc success=$ok wall=${wall}s" >&2
    done
  done
done
rm -rf "$AGENT_DIR"
python3 - "$OUT" <<'EOF'
import json, sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
runs = {}
for line in (out / "runs.jsonl").read_text().splitlines():
    r = json.loads(line)
    runs[r["run"]] = r  # last attempt per run wins
decisions = [json.loads(x) for x in (out / "jev-decisions.jsonl").read_text().splitlines()] \
    if (out / "jev-decisions.jsonl").exists() else []
by_session = defaultdict(list)
for d in decisions:
    by_session[d.get("session")].append(d)
rows = []
for r in runs.values():
    llm = inp = outp = 0
    cost = 0.0
    tools = []
    session = None
    for line in (out / f"{r['run']}.jsonl").read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "session":
            session = e.get("id")
        if e.get("type") == "message_end" and e["message"].get("role") == "assistant":
            u = e["message"].get("usage") or {}
            llm += 1
            inp += int(u.get("input") or 0) + int(u.get("cacheRead") or 0)
            outp += int(u.get("output") or 0)
            cost += float((u.get("cost") or {}).get("total") or 0.0)
        if e.get("type") == "tool_execution_start":
            tools.append(e.get("toolName"))
    ds = by_session.get(session, [])
    s1_requests = {d.get("request_id") for d in ds if d.get("model")}
    rows.append({**r, "session": session, "llm_calls": llm, "llm_input_tokens": inp, "llm_output_tokens": outp,
                 "pi_cost_usd": round(cost, 6), "tool_calls": len(tools),
                 "s1_calls": len(s1_requests), "s1_errors": sum(1 for d in ds if d.get("error")),
                 "s1_latency_ms": round(sum(d.get("latency_ms") or 0 for d in ds if d.get("model")), 1),
                 "gate_blocks": sum(1 for d in ds if d.get("extension") == "jev-tool-gate"
                                    and d.get("final_value") in ("block", "ask:block")),
                 "stop_nudges": sum(1 for d in ds if d.get("extension") == "jev-stop-verify"
                                    and d.get("final_value") == "continue"),
                 "decisions": [{k: d.get(k) for k in ("extension", "dp", "mode", "s1_value", "s1_conf", "final_value", "tool")}
                               for d in ds]})
rows.sort(key=lambda x: x["run"])
(out / "summary.json").write_text(json.dumps(rows, indent=1))
cols = ["run", "success", "wall_s", "llm_calls", "llm_input_tokens", "pi_cost_usd", "tool_calls", "s1_calls",
        "s1_latency_ms", "gate_blocks", "stop_nudges", "s1_errors", "exit"]
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
(out / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
EOF
