#!/usr/bin/env bash
# E6a (exploratory, descriptive; docs/PREREG_A7_STAGE2.md Deviations 2026-10-06 "Exploratory row E6a") on the CPU box
# jev-hard-tb (<instance-id>, m7i.8xlarge; prepared by hard_setup.sh), SSM only, as ubuntu:
#   PHASE=setup  bash e6a_alfworld.sh   # repo <- s3://<bucket>/code/e6a.bundle, alfworld 0.4.2 + textworld 1.7.0 into its
#                                       # venv, ALFWorld data (valid_seen + the 48 calibration train games) <- e6a/data.tgz
#   PHASE=cal|smoke|eval|rerun          # one runner invocation of configs/explore/e6a_alfworld/<phase>.yaml (rerun = eval
#                                       # again: re-runs episodes whose latest record ended in an infrastructure error)
# Every phase first pulls the latest bundle (fast-forward only), so the configs it runs are the committed ones. Keys come
# from SSM Parameter Store into this process's environment only. A watchdog stops the runner when the logged spend of
# every E6a run (scripts/explore_e6a_alfworld.py spend, incl. the aborted container attempt) reaches CAP. Outputs:
# runs/e6a_alfworld_<phase>/ -> s3://<bucket>/e6a/runs/, status lines -> s3://<bucket>/e6a/status/.
# Run detached:  nohup setsid env PHASE=cal bash e6a_alfworld.sh > /opt/jev/e6a_status/cal.out 2>&1 < /dev/null &
# EXP=e6c (Deviations 2026-10-06 "Exploratory row E6c": the action guard, same box, games and settings) runs
# configs/explore/e6c_alfworld/<smoke|eval>.yaml (rerun = eval again) with scripts/explore_e6c_alfworld.py as the spend
# watchdog (CAP 25 over the E6c runs), bundle s3://<bucket>/code/e6c.bundle, status and runs under s3://<bucket>/e6c/;
# the box keeps E6a's setup (alfworld, data). EXP defaults to e6a, whose behaviour is unchanged.
set -uo pipefail
PHASE=${PHASE:?PHASE=setup|cal|smoke|eval|rerun}
EXP=${EXP:-e6a}
case "$EXP" in
  e6a) DEF_CAP=80; DEF_WORKERS=8 ;;
  e6c) DEF_CAP=25; DEF_WORKERS=16 ;;
  *) echo "unknown EXP=$EXP"; exit 2 ;;
esac
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}; B="s3://$BUCKET"; REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
CAP=${CAP:-$DEF_CAP}; WORKERS=${WORKERS:-$DEF_WORKERS}
SPEND="scripts/explore_${EXP}_alfworld.py"
J=/opt/jev; R=$J/JevAgentHarness/research; S=$J/${EXP}_status
mkdir -p "$S"
export PATH="$HOME/.local/bin:$PATH"
st() { echo "$(date -u +%FT%TZ) [$EXP:$PHASE] $*" | tee -a "$S/$EXP.txt"
       aws s3 sync "$S" "$B/$EXP/status" --only-show-errors --region "$REGION" || true; }
sync_runs() { for d in "$R"/runs/${EXP}_alfworld_*; do
                [ -d "$d" ] && aws s3 sync "$d" "$B/$EXP/runs/$(basename "$d")" --only-show-errors --region "$REGION"
              done; true; }
cd "$R" || { st "no repo at $R"; exit 1; }

# the committed code and configs
aws s3 cp "$B/code/$EXP.bundle" "$J/$EXP.bundle" --only-show-errors --region "$REGION" || { st "bundle download failed"; exit 1; }
# tracked files become exactly the bundle's branch head (untracked runs/, data/ and the venv are kept)
git fetch -q "$J/$EXP.bundle" "$BRANCH" || { st "fetch from the bundle failed"; exit 1; }
git checkout -q -f "$BRANCH" 2>/dev/null || git checkout -q -f -b "$BRANCH"
git reset -q --hard FETCH_HEAD || { st "reset to the bundle failed"; exit 1; }
SRC_PIN=${SRC_PIN:-}       # pin the harness source (src/) at a commit, e.g. the calibration's (other sessions edit src/)
if [ -n "$SRC_PIN" ]; then git checkout -q "$SRC_PIN" -- src/ || { st "src pin $SRC_PIN failed"; exit 1; }; fi
st "code at $(git log --oneline -1 | cut -c1-60); src pinned at ${SRC_PIN:-head} (src tree $(git rev-parse HEAD:src 2>/dev/null | cut -c1-12) vs worktree $(git write-tree --prefix=src/ 2>/dev/null | cut -c1-12))"

if [ "$PHASE" = setup ]; then
  [ "$EXP" = e6a ] || { st "setup is E6a's (the box keeps it)"; exit 2; }
  sudo shutdown -h +720 "jev e6a: 12 h guard" 2>/dev/null || true      # user-data guards run at first boot only
  uv pip install -q --python .venv "alfworld==0.4.2" "textworld==1.7.0" || { st "pip alfworld failed"; exit 1; }
  aws s3 cp "$B/e6a/data.tgz" - --region "$REGION" | tar xz -C "$R" || { st "data download failed"; exit 1; }
  n=$(.venv/bin/python - <<'PY'
from jevharness.bench.suites.alfworld import collect_game_files as c
print(len(c("valid_seen", data_dir="data/raw/alfworld")), len(c("train", data_dir="data/raw/alfworld_e6a")))
PY
)
  sha=$(.venv/bin/python -c "import json,hashlib; d=json.load(open('configs/explore/e6a_alfworld/eval_games.json')); \
from jevharness.bench.suites.alfworld import load_alfworld as L; ids=[f().task_id for f in L('valid_seen', data_dir='data/raw/alfworld', stratify=True)]; \
print('eval list ok' if ids==d['task_ids'] else 'EVAL LIST MISMATCH')")
  st "setup done: alfworld $(.venv/bin/python -c 'import importlib.metadata as m; print(m.version("alfworld"), m.version("textworld"))'), games valid_seen / train = $n; $sha; $(nproc) vCPU, $(df -BG / | awk 'NR==2{print $4}') free"
  exit 0
fi

case "$PHASE" in
  cal|smoke|eval) CFG=configs/explore/${EXP}_alfworld/$PHASE.yaml ;;
  rerun) CFG=configs/explore/${EXP}_alfworld/eval.yaml ;;
  *) st "unknown PHASE"; exit 2 ;;
esac
[ -f "$CFG" ] || { st "no $CFG"; exit 2; }
for v in GEMINI_API_KEY OPENROUTER_API_KEY; do
  val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value \
        --output text 2>/dev/null) || val=""
  [ -n "$val" ] && [ "$val" != None ] && export "$v=$val"
done
unset val
[ -n "${GEMINI_API_KEY:-}" ] && [ -n "${OPENROUTER_API_KEY:-}" ] || { st "refusing: GEMINI_API_KEY and OPENROUTER_API_KEY needed"; exit 2; }
if grep -q "calibration: pending" "$CFG"; then st "refusing: $CFG still has a pending tau"; exit 2; fi
export JEV_WORKERS=$WORKERS
st "start $CFG (workers $(grep -m1 '^workers:' "$CFG" | awk '{print $2}'), cap \$$CAP); spend so far \$$(.venv/bin/python "$SPEND" spend | .venv/bin/python -c 'import json,sys; print(round(json.load(sys.stdin)["total"], 3))')"
setsid .venv/bin/python -m jevharness.bench.runner "$CFG" --out runs > "$S/$PHASE.log" 2>&1 < /dev/null &
PID=$!
.venv/bin/python "$SPEND" watch --pid "$PID" --cap "$CAP" --every 30 > "$S/$PHASE.watch.log" 2>&1 &
WPID=$!
( while kill -0 "$PID" 2>/dev/null; do sleep 300; sync_runs; done ) & SYNC=$!
wait "$PID"; RC=$?
kill "$WPID" "$SYNC" 2>/dev/null
sync_runs
leak=0; for k in "$GEMINI_API_KEY" "$OPENROUTER_API_KEY"; do
  leak=$((leak + $(grep -rlF -- "$k" runs/${EXP}_alfworld_* "$S" 2>/dev/null | wc -l))); done
st "runner exit=$RC; $(tail -1 "$S/$PHASE.watch.log"); key-leak files: $leak"
.venv/bin/python "$SPEND" spend > "$S/spend_$PHASE.json" 2>&1
aws s3 sync "$S" "$B/$EXP/status" --only-show-errors --region "$REGION" || true
exit $RC
