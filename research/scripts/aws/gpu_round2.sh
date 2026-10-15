#!/usr/bin/env bash
# GPU round 2 on an AWS box (runs as ubuntu via SSM; see docs/HANDOFF.md §5 "AWS").
#   1. pull the code bundle from the project bucket, fast-forward the clone, reuse the venv from setup_gpu.sh
#   2. read API keys from SSM Parameter Store (${SSM_PREFIX:?set SSM_PREFIX}/*) into the environment (never written to disk)
#   3. A5 competence gate: Intern-Decision 0.8B / 2B / 4B on the v0 dev set (Jev reference: results/pilot_jev/gate_jev_dev.json)
#   4. E-arch: scripts/bench_serving.py (dense vs hybrid, stateless vs session cache, state lengths)
#   5. E1 dev prefix with Intern-Decision-2B/4B (BFCL + When2Call, 299 items; same ids as results/dev_round2)
#   6. E2a slice (n=10) with Intern-Decision-2B as System-1, Gemini Flash as System-2 (pilot-scale, ≈ $1)
# Every stage syncs its outputs to s3://$BUCKET/gpu_round2/ so the parent can read them while the box runs.
# Budget guard: STOP_AFTER (default 150 min) powers the box off whatever is running.
set -uo pipefail
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
STOP_AFTER=${STOP_AFTER:-150}
REGION=${AWS_REGION:-us-east-1}
B="s3://$BUCKET"
OUT=runs/gpu_round2
sudo shutdown -h +"$STOP_AFTER" "gpu_round2: budget guard" 2>/dev/null || true

cd /opt/jev && aws s3 cp "$B/code/jev.bundle" jev.bundle --only-show-errors
cd JevAgentHarness/research && git pull -q --ff-only /opt/jev/jev.bundle main && git log --oneline -1
. .venv/bin/activate
mkdir -p "$OUT"
st() { echo "$(date -u +%H:%M:%S) $*" >> "$OUT/status.txt"; aws s3 sync "$OUT" "$B/gpu_round2" --only-show-errors --exclude '*.safetensors' --region "$REGION"; }

# keys from SSM (SecureString); exported for this shell only
for v in GEMINI_API_KEY GEMINI_API_KEYS OPENROUTER_API_KEY; do
  val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value --output text 2>/dev/null) || val=""
  [ -n "$val" ] && [ "$val" != None ] && export "$v=$val"
done
st "start $(git log --oneline -1) keys: gemini=$([ -n "${GEMINI_API_KEY:-}${GEMINI_API_KEYS:-}" ] && echo yes || echo no) openrouter=$([ -n "${OPENROUTER_API_KEY:-}" ] && echo yes || echo no)"

# 3. competence gate
for m in internlm/Intern-Decision-0.8B internlm/Intern-Decision-2B internlm/Intern-Decision-4B; do
  python scripts/gate_s1_dev.py --s1 "{\"type\":\"intern_decision\",\"model\":\"$m\"}" --out "$OUT/gate_${m##*/}.json" > "$OUT/gate_${m##*/}.log" 2>&1
  st "gate ${m##*/} exit=$?"
done

# 4. E-arch latency bench (skipped if the script is absent)
if [ -f scripts/bench_serving.py ]; then
  python scripts/bench_serving.py --out "$OUT/bench_serving" > "$OUT/bench_serving.log" 2>&1; st "bench_serving exit=$?"
else st "bench_serving: script absent, skipped"; fi

# 5. E1 dev prefix with Intern-Decision (needs Gemini for the FC/judge arms; S1 arms are local)
if [ -n "${GEMINI_API_KEY:-}${GEMINI_API_KEYS:-}" ]; then
  jevbench decide configs/dev/e1_dev_intern.yaml --out "$OUT" --max-items 300 > "$OUT/e1_intern.log" 2>&1; st "e1_intern exit=$?"
  # 6. E2a slice with Intern-Decision-2B as System-1
  jevbench run configs/experiments/jevgym_tool_select_intern.yaml --out "$OUT" --max-tasks 10 -o workers=4 name=e2a_intern2b_dev \
    > "$OUT/e2a_intern2b.log" 2>&1; st "e2a_intern2b exit=$?"
  d="$OUT/e2a_intern2b_dev"; [ -d "$d" ] && { jevbench report "$d" > "$d/report.md" 2>/dev/null; python -m jevharness.bench.scorecard "$d" --baseline llm_only --format md > "$d/scorecard.md" 2>/dev/null; }
else st "E1/E2a skipped: no Gemini key"; fi
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv > "$OUT/nvidia_smi_end.txt" 2>&1
st "done"
