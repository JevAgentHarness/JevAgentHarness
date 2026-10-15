#!/usr/bin/env bash
# GPU round 4: the registered Intern-Decision main run (docs/MAIN_RUN_PLAN.md step 7) and the A5.1 latency measurement
# that replaces the assumed Intern-Decision price. No freeze tag (authors' decision 2026-10-02): it refuses unless
# prereg_check passes on the pulled commit, and records that commit and the plan hash in START_E2a_intern.txt.
#   1. measure_latency (A5.1): Intern-Decision-4B, concurrency 1 and 4, 200 requests, 1.5k-token states, 3 questions
#   2. E2a' = jevgym_tool_select_intern.yaml on the frozen evaluation list, three passes (JEV_EVAL_RUNS=1, workers 4)
# Outputs sync to s3://$BUCKET/gpu_round4/ after every stage; the box stops itself at the end (and at STOP_AFTER min).
# One L4 (g6.4xlarge): about 7-8 h; System-2 API about $200 (MAIN_RUN_PLAN §4-5).
set -uo pipefail
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}; B="s3://$BUCKET"; REGION=${AWS_REGION:-us-east-1}
STOP_AFTER=${STOP_AFTER:-600}
sudo shutdown -c 2>/dev/null || true; sudo shutdown -h +"$STOP_AFTER" "gpu_round4: budget guard" 2>/dev/null || true
cd /opt/jev && aws s3 cp "$B/code/jev.bundle" jev.bundle --only-show-errors
cd JevAgentHarness/research && git checkout -q -- . && git fetch -q /opt/jev/jev.bundle 'refs/tags/*:refs/tags/*' 2>/dev/null
git pull -q --ff-only /opt/jev/jev.bundle main && git log --oneline -1
. .venv/bin/activate
OUT=runs/gpu_round4; mkdir -p "$OUT"
st() { echo "$(date -u +%H:%M:%S) $*" >> "$OUT/status.txt"; aws s3 sync "$OUT" "$B/gpu_round4" --only-show-errors --region "$REGION"; }
python scripts/prereg_check.py --quiet \
  || { st "refusing: prereg_check fails"; sudo shutdown -h +2 "gpu_round4: prereg_check"; exit 2; }
printf 'step E2a_intern\nstarted %s\ncommit %s\nplan sha256 %s\n' "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" \
  "$(sha256sum docs/PREREGISTRATION.md | cut -c1-16)" > "$OUT/START_E2a_intern.txt"
for v in GEMINI_API_KEY GEMINI_API_KEYS OPENROUTER_API_KEY; do
  val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value --output text 2>/dev/null) || val=""
  [ -n "$val" ] && [ "$val" != None ] && export "$v=$val"
done
[ -n "${GEMINI_API_KEY:-}${GEMINI_API_KEYS:-}" ] && [ -n "${OPENROUTER_API_KEY:-}" ] \
  || { st "refusing: Gemini and OpenRouter keys are both needed"; sudo shutdown -h +2 "gpu_round4: keys"; exit 2; }
st "start $(git log --oneline -1) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python scripts/measure_latency.py --s1 '{"type": "intern_decision", "model": "internlm/Intern-Decision-4B", "device": "auto"}' \
  --concurrency 1 4 --n 200 --state-tokens 1500 --questions 3 --gpu-hour-usd "${GPU_HOUR_USD:-1.32}" \
  > "$OUT/latency_intern4b.txt" 2>&1; st "measure_latency exit=$?"

d=runs/e2a_jevgym_tool_decisions_intern
for pass in 1 2 3; do
  JEV_EVAL_RUNS=1 jevbench run configs/experiments/jevgym_tool_select_intern.yaml --out runs -o workers=4 \
    >> "$OUT/e2a_intern_main.log" 2>&1; st "e2a' pass $pass exit=$?"
  aws s3 sync "$d" "$B/main/e2a_jevgym_tool_decisions_intern" --only-show-errors --region "$REGION"
done
jevbench report "$d" > "$d/report.md" 2>/dev/null; cp "$d/report.md" "$OUT/e2a_intern_report.md" 2>/dev/null
grep -c '"error": "' "$d"/*.jsonl > "$OUT/e2a_intern_errors.txt" 2>/dev/null
aws s3 sync "$d" "$B/main/e2a_jevgym_tool_decisions_intern" --only-show-errors --region "$REGION"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv > "$OUT/nvidia_smi_end.txt" 2>&1
st "round4 done"
sudo shutdown -h +2 "gpu_round4: finished" 2>/dev/null || true
