#!/usr/bin/env bash
# Resume GPU round 4 (E2a' = jevgym_tool_select_intern.yaml, three passes) after the 600-min budget guard of
# gpu_round4_main.sh stopped the box in pass 1 (2026-10-03 ~03:11 UTC; the A10G stand-in is slower than the L4 the
# guard was sized for). Same code as the original start (no bundle pull: the checkout stays at its start commit), the
# run directory resumes by task id (bench/runner.py), and passes 1-3 follow as registered. A record line goes to
# runs/gpu_round4/RESUME_E2a_intern.txt; any final JSONL line cut by the shutdown is dropped first (listed there).
#   aws ssm send-command ... 'aws s3 cp s3://<bucket>/code/gpu_round4_resume.sh /opt/jev/ && nohup setsid bash /opt/jev/gpu_round4_resume.sh &'
set -uo pipefail
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}; B="s3://$BUCKET"; REGION=${AWS_REGION:-us-east-1}
STOP_AFTER=${STOP_AFTER:-720}
sudo shutdown -c 2>/dev/null || true; sudo shutdown -h +"$STOP_AFTER" "gpu_round4 resume: budget guard" 2>/dev/null || true
cd /opt/jev/JevAgentHarness/research && . .venv/bin/activate
OUT=runs/gpu_round4; d=runs/e2a_jevgym_tool_decisions_intern; mkdir -p "$OUT"
st() { echo "$(date -u +%H:%M:%S) $*" >> "$OUT/status.txt"; aws s3 sync "$OUT" "$B/gpu_round4" --only-show-errors --region "$REGION"; }
cut=$(python - "$d" <<'PY'
import os, sys, glob
out = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.jsonl"))):
    with open(f, "rb+") as h:
        data = h.read()
        if data and not data.endswith(b"\n"):
            h.truncate(data.rfind(b"\n") + 1)
            out.append(os.path.basename(f))
print(" ".join(out) or "none")
PY
)
printf 'step E2a_intern (resumed)\nreason the 600-min budget guard of gpu_round4_main.sh stopped the box in pass 1 (about 2026-10-03T03:11Z)\nresumed %s\ncommit %s (unchanged checkout; no bundle pulled)\nplan sha256 %s\ncut final lines dropped: %s\nruns resume by task id; passes 1-3 follow as registered\n' \
  "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$(sha256sum docs/PREREGISTRATION.md | cut -c1-16)" "$cut" > "$OUT/RESUME_E2a_intern.txt"
for v in GEMINI_API_KEY GEMINI_API_KEYS OPENROUTER_API_KEY; do
  val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value --output text 2>/dev/null) || val=""
  [ -n "$val" ] && [ "$val" != None ] && export "$v=$val"
done
[ -n "${GEMINI_API_KEY:-}${GEMINI_API_KEYS:-}" ] && [ -n "${OPENROUTER_API_KEY:-}" ] \
  || { st "resume refusing: Gemini and OpenRouter keys are both needed"; sudo shutdown -h +2 "gpu_round4: keys"; exit 2; }
aws s3 sync "$d" "$B/main/e2a_jevgym_tool_decisions_intern" --only-show-errors --region "$REGION"
st "resume start $(git log --oneline -1) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) cut=$cut"
for pass in 1 2 3; do
  JEV_EVAL_RUNS=1 jevbench run configs/experiments/jevgym_tool_select_intern.yaml --out runs -o workers=4 \
    >> "$OUT/e2a_intern_main.log" 2>&1; st "e2a' pass $pass (resumed run) exit=$?"
  aws s3 sync "$d" "$B/main/e2a_jevgym_tool_decisions_intern" --only-show-errors --region "$REGION"
done
jevbench report "$d" > "$d/report.md" 2>/dev/null; cp "$d/report.md" "$OUT/e2a_intern_report.md" 2>/dev/null
grep -c '"error": "' "$d"/*.jsonl > "$OUT/e2a_intern_errors.txt" 2>/dev/null
aws s3 sync "$d" "$B/main/e2a_jevgym_tool_decisions_intern" --only-show-errors --region "$REGION"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv > "$OUT/nvidia_smi_end.txt" 2>&1
st "round4 done"
sudo shutdown -h +2 "gpu_round4: finished" 2>/dev/null || true
