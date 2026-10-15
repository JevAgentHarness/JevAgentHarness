#!/usr/bin/env bash
# GPU round 3b (after gpu_round3_train.sh): the Intern-Decision E2a dev slice re-run that round 3 could not resume.
# Round 3 tried to resume runs/gpu_round2/e2a_intern2b_dev, but the runner refused (correctly): the Intern backend's
# spec id changed with the shared-weights / OOM-retry fix, and resuming would mix two backends in one arm. So the
# slice runs again from scratch into a NEW directory, on the dev seeds 0-9 that round 2 used (never the evaluation
# range), 10 tasks per arm. The two tau-hat arms stay skipped (`calibration: pending`, A6.16 (b)). ~20-30 min of L4.
set -uo pipefail
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}; B="s3://$BUCKET"; REGION=${AWS_REGION:-us-east-1}
STOP_AFTER=${STOP_AFTER:-75}
sudo shutdown -c 2>/dev/null || true; sudo shutdown -h +"$STOP_AFTER" "gpu_round3b: budget guard" 2>/dev/null || true
cd /opt/jev && aws s3 cp "$B/code/jev.bundle" jev.bundle --only-show-errors
cd JevAgentHarness/research && git pull -q --ff-only /opt/jev/jev.bundle main && git log --oneline -1
. .venv/bin/activate
OUT=runs/gpu_round3; mkdir -p "$OUT"
st() { echo "$(date -u +%H:%M:%S) $*" >> "$OUT/status.txt"; aws s3 sync "$OUT" "$B/gpu_round3" --only-show-errors --region "$REGION"; }
for v in GEMINI_API_KEY GEMINI_API_KEYS OPENROUTER_API_KEY; do
  val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value --output text 2>/dev/null) || val=""
  [ -n "$val" ] && [ "$val" != None ] && export "$v=$val"
done
st "round3b start $(git log --oneline -1)"
if [ -n "${GEMINI_API_KEY:-}${GEMINI_API_KEYS:-}" ]; then
  d=runs/gpu_round3/e2a_intern_oomfix_dev
  jevbench run configs/experiments/jevgym_tool_select_intern.yaml --out runs/gpu_round3 --max-tasks 10 -o workers=4 \
    name=e2a_intern_oomfix_dev suite.seed0=0 > "$OUT/e2a_intern_oomfix.log" 2>&1; st "e2a_intern_oomfix exit=$?"
  jevbench report "$d" > "$d/report.md" 2>/dev/null
  python -m jevharness.bench.scorecard "$d" --baseline llm_only --format md > "$d/scorecard.md" 2>/dev/null
  grep -c '"error"' "$d"/*.jsonl > "$OUT/e2a_intern_oomfix_errors.txt" 2>/dev/null
  st "e2a_intern_oomfix reported"
else st "e2a_intern_oomfix skipped: no Gemini key"; fi
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv > "$OUT/nvidia_smi_end_3b.txt" 2>&1
st "round3b done"
sudo shutdown -h +2 "gpu_round3b: finished" 2>/dev/null || true
