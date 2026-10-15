#!/usr/bin/env bash
# Budget-gated Terminal-Bench-Science evaluation (H5, exploratory) on the TB box: passes 1..PASSES of
# `PHASE=tbsci_eval_pass hard_tb.sh` (one attempt per arm per pass; same arms, caps, retries as tbsci_eval). Pass p is
# started only if the logged API spend so far plus the projected cost of the pass stays within CAP_USD; while a pass
# runs, a watchdog stops it cleanly (SIGINT to Harbor: in-flight trials end as CancelledError and are reported) if the
# logged spend plus the in-flight trials at the running mean cost would exceed CAP_USD. A pass that is not started or is
# stopped is a budget deviation (A6.22 (e) 9), reported as such; it never changes caps, arms or rules.
#   PRIOR_USD   spend outside this box's runs (e.g. the SWE-bench runs), counted against the same cap (required)
#   CAP_USD     440;  MARGIN 1.15 (projection = trials in the pass x mean cost per trial x MARGIN);  PASSES 3
#   Logged spend here = Harbor's agent_result.cost_usd (Gemini, from pi's usage) of the calibration baseline attempts
#   and of every evaluation trial, plus the Jev estimate from jev-decisions.jsonl (summarize_harbor, $0.042 / 1M input).
# Run detached:  nohup setsid env PRIOR_USD=227.05 bash scripts/aws/hard_tb_eval_budget.sh > /opt/jev/hard_status/tb_eval.log 2>&1 &
set -uo pipefail
PRIOR_USD=${PRIOR_USD:?PRIOR_USD (spend already counted against the cap) is required}
CAP_USD=${CAP_USD:-440}; MARGIN=${MARGIN:-1.15}; PASSES=${PASSES:-3}; STOP_WHEN_DONE=${STOP_WHEN_DONE:-1}
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}; REGION=${AWS_REGION:-us-east-1}
J=/opt/jev; R=$J/JevAgentHarness/research; S=$J/hard_status
cd "$R" || exit 1
st() { echo "$(date -u +%FT%TZ) [tb:eval_budget] $*" | tee -a "$S/tb.txt"
       aws s3 sync "$S" "s3://$BUCKET/hard/tb/status" --only-show-errors --region "$REGION" || true; }
# prints: <logged spend USD> <finished eval trials> <in-flight eval trials> <mean cost per finished eval trial or calibration mean>
spend() { .venv/bin/python - "$R" <<'PY'
import glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "adapters/pi"))
from summarize_harbor import trial_row
R = sys.argv[1]
def rows(pattern):
    out, inflight = [], 0
    for d in glob.glob(pattern):
        if not Path(d).is_dir():
            continue
        if Path(d, "result.json").exists():
            out.append(trial_row(Path(d), 0.042))
        else:
            inflight += 1
    return out, inflight
cost = lambda r: float(r.get("llm_cost_usd") or 0) + float(r.get("s1_cost_usd_est") or 0)
cal, _ = rows(f"{R}/runs/tbsci_cal/baseline/*__*")
ev, inflight = rows(f"{R}/runs/tbsci/pass*/*/*__*")
hard = {l.strip() for l in open(f"{R}/adapters/pi/tbsci_hard.txt") if l.strip() and not l.startswith("#")}
cal_sel = [cost(r) for r in cal if r["task"] in hard]
mean = (sum(cost(r) for r in ev) / len(ev)) if len(ev) >= 10 else (sum(cal_sel) / max(len(cal_sel), 1))
print(f"{sum(map(cost, cal)) + sum(map(cost, ev)):.2f} {len(ev)} {inflight} {mean:.4f}")
PY
}
n_tasks=$(grep -v '^#' adapters/pi/tbsci_hard.txt | grep -c .)
per_pass=$((3 * n_tasks))
st "start: cap \$$CAP_USD, prior \$$PRIOR_USD, $n_tasks task(s) x 3 arms = $per_pass trials per pass, margin $MARGIN, passes <= $PASSES"
for p in $(seq 1 "$PASSES"); do
  read -r box_usd n_ev n_fly mean < <(spend)
  proj=$(python3 -c "print(f'{$per_pass * $mean * $MARGIN:.2f}')")
  total=$(python3 -c "print(f'{$PRIOR_USD + $box_usd:.2f}')")
  if python3 -c "import sys; sys.exit(0 if $PRIOR_USD + $box_usd + $proj <= $CAP_USD else 1)"; then
    st "pass $p: spent \$$total (box \$$box_usd), projected pass cost \$$proj (mean \$$mean/trial) -> start"
  else
    st "pass $p NOT started: spent \$$total + projected \$$proj > cap \$$CAP_USD (budget deviation, A6.22 (e) 9)"
    break
  fi
  env PHASE=tbsci_eval_pass PASS=$p bash scripts/aws/hard_tb.sh > "$S/tb_tbsci_eval_pass$p.log" 2>&1 < /dev/null &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    read -r box_usd n_ev n_fly mean < <(spend)
    if python3 -c "import sys; sys.exit(0 if $PRIOR_USD + $box_usd + $n_fly * $mean > $CAP_USD else 1)"; then
      st "pass $p STOPPED by the watchdog: spent \$$(python3 -c "print(f'{$PRIOR_USD + $box_usd:.2f}')") + $n_fly in flight x \$$mean > cap"
      pkill -INT -f 'harbor run'; sleep 60; pkill -KILL -f 'harbor run'
      wait "$pid"; break 2
    fi
  done
  wait "$pid"
  st "pass $p done (exit $?)"
done
read -r box_usd n_ev n_fly mean < <(spend)
st "eval done: $n_ev evaluation trial(s); logged spend on this box \$$box_usd (calibration + evaluation); total with prior \$$(python3 -c "print(f'{$PRIOR_USD + $box_usd:.2f}')")"
if [ "$STOP_WHEN_DONE" = 1 ]; then st "stopping the box"; sudo shutdown -h +2 "jev-hard: tbsci eval done"; fi
