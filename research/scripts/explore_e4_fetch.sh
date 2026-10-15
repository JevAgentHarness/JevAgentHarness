#!/usr/bin/env bash
# Fetch the E4 / E4b outputs from S3 (written by scripts/aws/e4_swe.sh and e4b_chain.sh) and write their readouts:
#   results/<exp>_swe/{summary.md, summary.json, per_instance.csv, h5_style.md}, paper/generated/<exp>_swe.tex,
#   results/e4b_swe/dev/ (E4b's dev summary and the threshold the box chose).
#   bash scripts/explore_e4_fetch.sh            # both rows (whatever has finished)
#   bash scripts/explore_e4_fetch.sh e4b        # one row
# Needs an AWS login (read access to the bucket); no API call. NOTES.md is written by hand afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."
B=s3://$BUCKET
A() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws --region us-east-1 "$@"; }
mkdir -p runs/e4_swe
A s3 cp "$B/e4/status/e4.txt" runs/e4_swe/e4_status.txt --only-show-errors
grep -E "\[(e4|e4b):(eval|chain)\]" runs/e4_swe/e4_status.txt | tail -12
for exp in "${@:-e4 e4b}"; do
  for exp1 in $exp; do
    for phase in dev eval; do
      A s3 sync "$B/$exp1/runs/$phase" "runs/${exp1}_swe/$phase" --only-show-errors \
        --exclude "*/*/*.traj.json" --exclude "*/eval/logs/*" || true
    done
    run="runs/${exp1}_swe/eval"
    if [ "$exp1" = e4b ] && [ -d runs/e4b_swe/dev/summary ]; then
      mkdir -p results/e4b_swe/dev
      .venv/bin/python scripts/explore_e4_swe.py --exp e4b summarize runs/e4b_swe/dev --phase dev --out-dir results/e4b_swe/dev \
        --title "E4b: dev (30 instances, used only to fix the routing threshold; development numbers)" > /dev/null
      cp runs/e4b_swe/dev/threshold.json results/e4b_swe/dev/threshold.json 2>/dev/null || true
      cp runs/e4b_swe/dev/threshold.json configs/explore/e4b_swe/threshold.json 2>/dev/null || true   # the box's choice
    fi
    if ls "$run"/*.jsonl > /dev/null 2>&1; then
      dev_sum="results/${exp1}_swe/dev/summary.json"
      .venv/bin/python scripts/explore_e4_swe.py --exp "$exp1" summarize "$run" --phase eval --tex --dev-summary "$dev_sum" \
        --title "${exp1^^}: JevAgentHarness on mini-SWE-agent, SWE-bench Verified eval (100 instances)"
      arms=$(ls "$run"/*.jsonl | xargs -n1 basename | sed 's/\.jsonl$//' | tr '\n' ' ')
      .venv/bin/python adapters/mini_swe/h5_summary.py "$run" --baseline always_deep --arms $arms \
        --json "results/${exp1}_swe/h5_style.json" > "results/${exp1}_swe/h5_style.md"
    else
      echo "!! $exp1: no eval readout yet ($run/*.jsonl missing)"
    fi
  done
done
