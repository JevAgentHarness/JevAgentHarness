#!/usr/bin/env bash
# E6d (exploratory, descriptive; docs/PREREG_A7_STAGE2.md Deviations 2026-10-06 "Exploratory row E6d") on the CPU box
# jev-hard-swe (<instance-id>, m7i.8xlarge; prepared by hard_setup.sh), SSM only, as ubuntu:
#   PHASE=smoke bash e6d_modes.sh   # every arm on 2 development tasks (configs/explore/e6d_modes/smoke.yaml), then the
#                                   # registered projection (scripts/explore_e6d_modes.py project -> projection.json)
#   PHASE=eval  bash e6d_modes.sh   # configs/explore/e6d_modes/eval.yaml at the n the projection allows (200 or the
#                                   # first 100; refuses otherwise), three passes (passes 2-3 re-run infrastructure
#                                   # errors only), then the summary; STOP_WHEN_DONE=1 powers the box off at the end
# Every phase first pulls the latest bundle (fast-forward to the branch head), so the configs it runs are the committed
# ones. Keys come from SSM Parameter Store into this process's environment only. A watchdog stops the runner when the
# logged spend of every E6d run (smoke + eval) reaches CAP. Outputs: runs/e6d_modes{_smoke}/ -> s3://<bucket>/e6d/runs/,
# status lines -> s3://<bucket>/e6d/status/, results/explore/e6d_modes/ -> s3://<bucket>/e6d/results/.
# Run detached:  nohup setsid env PHASE=smoke bash e6d_modes.sh > /opt/jev/e6d_status/smoke.out 2>&1 < /dev/null &
set -uo pipefail
PHASE=${PHASE:?PHASE=smoke|eval|routing_smoke|routing}
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}; B="s3://$BUCKET"; REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
CAP=${CAP:-120}; WORKERS=${WORKERS:-8}; GUARD_MIN=${GUARD_MIN:-360}
J=/opt/jev; R=$J/JevAgentHarness/research; S=$J/e6d_status
mkdir -p "$S"
export PATH="$HOME/.local/bin:$PATH"
st() { echo "$(date -u +%FT%TZ) [e6d:$PHASE] $*" | tee -a "$S/e6d.txt"
       aws s3 sync "$S" "$B/e6d/status" --only-show-errors --region "$REGION" || true; }
sync_out() { for d in "$R"/runs/e6d_modes*; do
               [ -d "$d" ] && aws s3 sync "$d" "$B/e6d/runs/$(basename "$d")" --only-show-errors --region "$REGION"
             done
             [ -d "$R/results/explore/e6d_modes" ] && aws s3 sync "$R/results/explore/e6d_modes" "$B/e6d/results" \
               --only-show-errors --region "$REGION"
             true; }
cd "$R" || { st "no repo at $R"; exit 1; }
sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "jev e6d: power-off guard" 2>/dev/null || true

# the committed code and configs (tracked files become exactly the bundle's branch head; runs/ and the venv are kept)
aws s3 cp "$B/code/e6d.bundle" "$J/e6d.bundle" --only-show-errors --region "$REGION" || { st "bundle download failed"; exit 1; }
git fetch -q "$J/e6d.bundle" "$BRANCH" || { st "fetch from the bundle failed"; exit 1; }
git checkout -q -f "$BRANCH" 2>/dev/null || git checkout -q -f -b "$BRANCH"
git reset -q --hard FETCH_HEAD || { st "reset to the bundle failed"; exit 1; }
st "code at $(git log --oneline -1 | cut -c1-80); $(nproc) vCPU, $(df -BG / | awk 'NR==2{print $4}') free"
# the E6d modes must be in the installed package (editable install of this checkout)
.venv/bin/python -c "from jevharness.harness.config import E6D_CELLS; assert len(E6D_CELLS) == 7" \
  || { st "refusing: the venv does not import the E6d harness"; exit 2; }

for v in GEMINI_API_KEY OPENROUTER_API_KEY; do
  val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value \
        --output text 2>/dev/null) || val=""
  [ -n "$val" ] && [ "$val" != None ] && export "$v=$val"
done
unset val
[ -n "${GEMINI_API_KEY:-}" ] && [ -n "${OPENROUTER_API_KEY:-}" ] || { st "refusing: GEMINI_API_KEY and OPENROUTER_API_KEY needed"; exit 2; }
spend() { .venv/bin/python scripts/explore_e6d_modes.py spend | .venv/bin/python -c 'import json,sys; print(round(json.load(sys.stdin)["total"], 3))'; }

run_once() {   # $1 = config, $2.. = runner overrides; one invocation under the watchdog
  local cfg=$1; shift
  export JEV_WORKERS=$WORKERS
  setsid .venv/bin/python -m jevharness.bench.runner "$cfg" --out runs "$@" >> "$S/$PHASE.log" 2>&1 < /dev/null &
  local pid=$!
  .venv/bin/python scripts/explore_e6d_modes.py watch --pid "$pid" --cap "$CAP" --every 30 >> "$S/$PHASE.watch.log" 2>&1 &
  local wpid=$!
  ( while kill -0 "$pid" 2>/dev/null; do sleep 300; sync_out; done ) & local spid=$!
  wait "$pid"; local rc=$?
  kill "$wpid" "$spid" 2>/dev/null
  sync_out
  return $rc
}

case "$PHASE" in
  smoke)
    st "start smoke; spend so far \$$(spend)"
    run_once configs/explore/e6d_modes/smoke.yaml; RC=$?
    .venv/bin/python scripts/explore_e6d_modes.py project > "$S/projection.json" 2>&1
    st "smoke runner exit=$RC; spend \$$(spend); projection: $(.venv/bin/python -c 'import json; print(json.load(open("results/explore/e6d_modes/projection.json"))["decision"])' 2>/dev/null)"
    ;;
  eval)
    [ -f results/explore/e6d_modes/projection.json ] || .venv/bin/python scripts/explore_e6d_modes.py project > /dev/null 2>&1
    DEC=$(.venv/bin/python -c 'import json; print(json.load(open("results/explore/e6d_modes/projection.json"))["decision"])' 2>/dev/null)
    case "$DEC" in
      "run all 200") OV=() ;;
      "run the first 100") OV=(--max-tasks 100) ;;
      *) st "refusing eval: projection says '$DEC'"; exit 3 ;;
    esac
    export JEV_EVAL_RUNS=1
    for pass in 1 2 3; do
      [ -f runs/e6d_STOP.json ] && { st "watchdog stop recorded; no further pass"; break; }
      st "eval pass $pass ($DEC); spend so far \$$(spend)"
      run_once configs/explore/e6d_modes/eval.yaml "${OV[@]}"; RC=$?
      st "eval pass $pass runner exit=$RC; spend \$$(spend)"
    done
    .venv/bin/python scripts/explore_e6d_modes.py summarize > "$S/summary.out" 2>&1 && st "summary written" \
      || st "summary FAILED (see summary.out)"
    sync_out
    ;;
  routing_smoke)     # the routing repair (Deviations addendum ~16:55 UTC): smoke of the trio
    st "start routing smoke; spend so far \$$(spend)"
    run_once configs/explore/e6d_modes/smoke_routing.yaml; RC=$?
    st "routing smoke runner exit=$RC; spend \$$(spend)"
    ;;
  routing)           # the routing repair: the trio on seeds 301200-301399, three passes, then the summary
    for pass in 1 2 3; do
      [ -f runs/e6d_STOP.json ] && { st "watchdog stop recorded; no further pass"; break; }
      st "routing pass $pass; spend so far \$$(spend)"
      run_once configs/explore/e6d_modes/eval_routing.yaml; RC=$?
      st "routing pass $pass runner exit=$RC; spend \$$(spend)"
    done
    .venv/bin/python scripts/explore_e6d_modes.py summarize > "$S/summary_routing.out" 2>&1 && st "summary written" \
      || st "summary FAILED (see summary_routing.out)"
    sync_out
    ;;
  *) st "unknown PHASE"; exit 2 ;;
esac
leak=0; for k in "$GEMINI_API_KEY" "$OPENROUTER_API_KEY"; do
  leak=$((leak + $(grep -rlF -- "$k" runs/e6d_modes* "$S" results/explore/e6d_modes 2>/dev/null | wc -l))); done
st "done; key-leak files: $leak"
aws s3 sync "$S" "$B/e6d/status" --only-show-errors --region "$REGION" || true
if [ "${STOP_WHEN_DONE:-0}" = 1 ]; then st "powering off"; sudo shutdown -h now; fi
exit 0
