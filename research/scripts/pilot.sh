#!/usr/bin/env bash
# Day-1 Jev pilot: run this first once openrouter.ai is reachable (docs/HANDOFF.md §1).
# Checks connectivity, pings every API System-1 backend, takes a latency snapshot, then runs small
# versions of E1 / E2a / E2b so a broken arm shows up in minutes, not after a full-scale run.
#
#   bash scripts/pilot.sh            # N=10 JevGym tasks, 200 static items
#   N=5 ITEMS=100 bash scripts/pilot.sh
#   OUT=runs/pilot_jev bash scripts/pilot.sh
set -uo pipefail
cd "$(dirname "$0")/.."
N=${N:-10}
ITEMS=${ITEMS:-200}
OUT=${OUT:-runs/pilot_jev}
JEV='{"type": "jev", "provider": "openrouter"}'
LITE='{"type": "llm_generative", "llm": {"provider": "gemini", "model": "gemini-3.5-flash-lite"}, "reasoning_effort": "minimal"}'
EMB='{"type": "embedding", "provider": "gemini", "model": "gemini-embedding-001"}'
fail=0

echo "== 0. connectivity"
for url in https://openrouter.ai/api/v1/models https://generativelanguage.googleapis.com/; do
  code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 15 "$url" 2>/dev/null) || true
  code=${code:-000}
  echo "  $url -> $code"
  if [ "$code" = 000 ]; then  # no HTTP response at all (DNS / proxy CONNECT refused / timeout)
    echo "!! $url unreachable: allow-list the host in the environment's network settings (docs/HANDOFF.md §1)"
    fail=1
  fi
done
[ "$fail" = 1 ] && [ "${FORCE:-0}" != 1 ] && { echo "stopping (FORCE=1 to continue anyway)"; exit 1; }

echo "== 1. ping System-1 backends"
jevbench ping --s1 "$JEV"  || { echo "!! Jev via OpenRouter failed (OPENROUTER_API_KEY in .env?)"; fail=1; }
jevbench ping --s1 "$LITE" || echo "!! Gemini Flash-Lite stand-in failed (GEMINI_API_KEY(S) in .env?)"
jevbench ping --s1 "$EMB"  || echo "   (embedding answers Choice questions only; a Noul/Score error here is expected)"
[ -n "${OPENJEV_BASE_URL:-}" ] && { jevbench ping --s1 '{"type": "openjev"}' || echo "!! OpenJev server failed"; }

echo "== 2. latency snapshot (Jev, concurrency 1 and 8)"
python scripts/measure_latency.py --s1 "$JEV" --concurrency 1 8 --n 20 || echo "!! latency snapshot failed"

echo "== 3. E1 static decisions ($ITEMS items)"
jevbench decide configs/decision/bfcl_when2call.yaml --max-items "$ITEMS" --out "$OUT" || fail=1

echo "== 4. E2a / E2b JevGym ($N tasks)"
jevbench run configs/experiments/jevgym_tool_select.yaml --max-tasks "$N" --out "$OUT" -o workers=4 || fail=1
jevbench run configs/experiments/jevgym_meta_control.yaml --max-tasks "$N" --out "$OUT" -o workers=4 || fail=1
for d in "$OUT"/e2a_jevgym_tool_decisions "$OUT"/e2b_jevgym_meta_control; do
  [ -d "$d" ] || continue
  jevbench report "$d" > "$d/report.md" || true
  python -m jevharness.bench.scorecard "$d" --baseline llm_only --format md > "$d/scorecard.md" 2>/dev/null || true
  echo "  -> $d/report.md, $d/scorecard.md"
done

echo "== 5. what the pilot produced (keys under $OUT; not for the paper)"
python -m jevharness.bench.paper_numbers "$OUT" --json "$OUT/numbers.json" 2>&1 | tail -3
grep -l '"error": "' "$OUT"/*/*.jsonl 2>/dev/null | head -5 | sed 's/^/!! episodes with errors: /'   # "error": null is not an error
echo "done (fail=$fail). Check unpriced_models in the reports and Jev usage.cost before scaling up."
exit $fail
