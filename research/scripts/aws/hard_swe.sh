#!/usr/bin/env bash
# H5 (exploratory) on SWE-bench Verified with mini-SWE-agent 2.4.6, on a hard-task box prepared by hard_setup.sh.
# System-2 Gemini 3.5 Flash, minimal thinking (adapters/mini_swe/gemini.yaml); System-1 Jev via OpenRouter
# (adapters/mini_swe/s1_jev_openrouter.yaml). Registered caps (A6.22 (e) 9): step_limit 250, cost_limit $3 per instance,
# 60 s per command (swebench_jev.yaml), k = 1. Arms (ROADMAP E8): baseline, action_guard, stop_guard, route.
#   PHASE=smoke bash hard_swe.sh   # 2 instances OUTSIDE the evaluation list (shuffled positions 100-101) x every arm
#   PHASE=full  bash hard_swe.sh   # the 100-instance list (shuffled positions 0-99) x every arm, local evaluation
#   PHASE=rerun bash hard_swe.sh   # re-run instances whose record carries an infrastructure error (A6.22 (e) 4: <= 2x)
# Arms run as parallel processes (one run_swebench.sh per arm, WORKERS containers each), so they share the box's load
# and the API's rate limits over the same period. Each arm has its own System-1 SQLite cache (no cross-process locks).
# Keys come from SSM Parameter Store into this process's environment only (never written to disk or logs).
# Run detached:  nohup setsid env PHASE=full bash hard_swe.sh > /opt/jev/hard_status/swe_full.log 2>&1 < /dev/null &
set -uo pipefail
PHASE=${PHASE:?PHASE=smoke|full|rerun}
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}; B="s3://$BUCKET"; REGION=${AWS_REGION:-us-east-1}
BOX=${BOX:-swe}
R=/opt/jev/JevAgentHarness/research; S=/opt/jev/hard_status; mkdir -p "$S"
ARMS=${ARMS:-"baseline action_guard stop_guard route"}
WORKERS=${WORKERS:-6}; EVAL_WORKERS=${EVAL_WORKERS:-4}
ARM_COST_LIMIT=${ARM_COST_LIMIT:-0}         # MSWEA_GLOBAL_COST_LIMIT per arm process (USD; 0 = none): a spend guard
case "$PHASE" in
  smoke) OUT=${OUT:-$R/runs/hard_smoke/swe}; SLICE=${SLICE:-100:102}; NAME=${NAME:-hard_smoke_swe} ;;
  full|rerun) OUT=${OUT:-$R/runs/swe_verified_100}; SLICE=${SLICE:-0:100}; NAME=${NAME:-swe_verified_100} ;;
  *) echo "unknown PHASE $PHASE" >&2; exit 2 ;;
esac
cd "$R" || exit 1
export PATH="$HOME/.local/bin:$PATH"
st() { echo "$(date -u +%FT%TZ) [swe:$PHASE] $*" | tee -a "$S/swe.txt"
       aws s3 sync "$S" "$B/hard/$BOX/status" --only-show-errors --region "$REGION" || true; }
sync_out() { aws s3 sync "$OUT" "$B/hard/$BOX/runs/$NAME" --only-show-errors --region "$REGION" --exclude '*.sqlite' || true; }

for v in GEMINI_API_KEY OPENROUTER_API_KEY; do
  val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value \
        --output text 2>/dev/null) || val=""
  [ -n "$val" ] && [ "$val" != None ] && export "$v=$val"
done
unset val
[ -n "${GEMINI_API_KEY:-}" ] && [ -n "${OPENROUTER_API_KEY:-}" ] || { st "refusing: GEMINI_API_KEY and OPENROUTER_API_KEY needed"; exit 2; }
or_usage() { curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
             | jq -r '.data.usage // "?"' 2>/dev/null; }
leak_check() {  # files under $OUT that contain a key value (count only; values never printed)
  local n=0 k
  for k in "$GEMINI_API_KEY" "$OPENROUTER_API_KEY"; do n=$((n + $(grep -rlF -- "$k" "$OUT" 2>/dev/null | wc -l))); done
  echo "$n"
}

st "start $(git log --oneline -1) arms=[$ARMS] workers=$WORKERS out=$OUT openrouter_usage=$(or_usage)"
mkdir -p "$OUT"
IDS="$OUT/instance_ids.txt"
if [ ! -s "$IDS" ]; then
  .venv/bin/python -m jevharness.adapters.mini_swe ids --subset verified --split test --shuffle --slice "$SLICE" \
    -o "$IDS" > /dev/null 2>"$S/swe_ids.err" || { st "ids failed: $(tail -1 "$S/swe_ids.err")"; exit 1; }
fi
st "instances: $(wc -l < "$IDS") (sha256 $(sha256sum "$IDS" | cut -c1-16))"

if [ "$PHASE" = "rerun" ]; then
  # A6.22 (e) 4: an instance whose latest record has `error` set (an exception escaped the agent: Docker, API after
  # retries, harness) is re-run from scratch; the errored record is kept under <arm>/_errored/<iid>.<n>/.
  for arm in $ARMS; do
    d="$OUT/$arm"
    for f in "$d"/*/*.jev.json; do
      [ -f "$f" ] || continue
      iid=$(basename "$(dirname "$f")")
      if .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('error') else 1)" "$f"; then
        n=$(ls -d "$d/_errored/$iid".* 2>/dev/null | wc -l)
        if [ "$n" -ge 2 ]; then st "$arm/$iid: errored after 2 re-runs, left as is"; continue; fi
        mkdir -p "$d/_errored"; mv "$d/$iid" "$d/_errored/$iid.$((n + 1))"
        .venv/bin/python - "$d/preds.json" "$iid" <<'PY'
import json, sys
p, iid = sys.argv[1], sys.argv[2]
d = json.load(open(p)); d.pop(iid, None); json.dump(d, open(p, "w"), indent=2)
PY
        st "$arm/$iid: errored record moved aside (re-run $((n + 1)))"
      fi
    done
  done
fi

# pre-pull the instance images (anonymous Docker Hub: 100 pulls/h/IP; mirror.gcr.io first, see hard_setup.sh).
# An image already on the box is not pulled again (no registry request); before a pull, wait (up to 1 h) while the
# anonymous budget left is below HUB_RESERVE (the HEAD request that reads it is not counted by Docker Hub).
HUB_RESERVE=${HUB_RESERVE:-10}; export HUB_RESERVE
hub_remaining() { local tok
  tok=$(curl -s -m 20 "https://auth.docker.io/token?service=registry.docker.io&scope=repository:ratelimitpreview/test:pull" \
        | jq -r .token 2>/dev/null)
  curl -s -m 20 --head -H "Authorization: Bearer $tok" https://registry-1.docker.io/v2/ratelimitpreview/test/manifests/latest \
    | tr -d '\r' | awk -F'[ ;]' 'tolower($1)=="ratelimit-remaining:"{print $2}'; }
pull() { local img="docker.io/swebench/sweb.eval.x86_64.$(echo "${1//__/_1776_}" | tr 'A-Z' 'a-z'):latest" i r
         docker image inspect "$img" >/dev/null 2>&1 && { echo "CACHED $img"; return 0; }
         for i in $(seq 1 30); do r=$(hub_remaining); { [ -z "$r" ] || [ "$r" -ge "$HUB_RESERVE" ]; } && break
           echo "WAIT hub_remaining=$r $img"; sleep 120; done
         for i in 1 2 3 4 5; do docker pull -q "$img" >/dev/null 2>&1 && { echo "PULLED $img"; return 0; }; sleep $((i * 20)); done
         echo "PULL FAILED $img"; }
export -f pull hub_remaining
t0=$(date +%s)
sed 's/#.*//; /^\s*$/d' "$IDS" | xargs -P 6 -I{} bash -c 'pull "$@"' _ {} > "$S/swe_pull_$PHASE.txt" 2>&1
st "pre-pull done in $(( $(date +%s) - t0 )) s; cached $(grep -c ^CACHED "$S/swe_pull_$PHASE.txt"), pulled $(grep -c ^PULLED "$S/swe_pull_$PHASE.txt"), waits $(grep -c ^WAIT "$S/swe_pull_$PHASE.txt"), failures: $(grep -c 'PULL FAILED' "$S/swe_pull_$PHASE.txt"); disk $(df -BG / | awk 'NR==2{print $3"/"$2}')"

( while sleep 600; do sync_out; done ) & SYNC=$!
pids=()
for arm in $ARMS; do
  ( export MSWEA_GLOBAL_COST_LIMIT=$ARM_COST_LIMIT MSWEA_SILENT_STARTUP=1
    ENV_FILE="" SUBSET=verified OUT="$OUT" IDS="$IDS" WORKERS="$WORKERS" EVAL=local EVAL_PY="$R/.venv-eval/bin/python" \
    EVAL_WORKERS="$EVAL_WORKERS" CONFIG_OVERLAY="adapters/mini_swe/gemini.yaml adapters/mini_swe/s1_jev_openrouter.yaml" \
    RUN_ARGS="-c jev.s1_cache=cache/jev_swe_${NAME}_${arm}.sqlite -c environment.pull_timeout=900" \
    bash adapters/mini_swe/run_swebench.sh "$arm" > "$OUT/run_$arm.$PHASE.log" 2>&1
    echo "exit=$?" >> "$OUT/run_$arm.$PHASE.log" ) &
  pids+=($!)
  sleep 20
done
for p in "${pids[@]}"; do wait "$p"; done
kill "$SYNC" 2>/dev/null

for arm in $ARMS; do
  cnt=$(.venv/bin/python - "$OUT/$arm" <<'PY'
import glob, json, sys
recs = [json.load(open(f)) for f in glob.glob(f"{sys.argv[1]}/*/*.jev.json")]
print(f"instances={len(recs)} errors={sum(bool(r.get('error')) for r in recs)}")
PY
)
  st "$arm: $(tail -1 "$OUT/run_$arm.$PHASE.log") $cnt reports=$(ls "$OUT/$arm"/eval/*.jev_$arm.json 2>/dev/null | wc -l)"
done
.venv/bin/python -m jevharness.bench.scorecard "$OUT" --baseline baseline > "$OUT/scorecard.md" 2> "$OUT/scorecard.err"
st "scorecard exit=$? ($(wc -l < "$OUT/scorecard.md") lines)"
.venv/bin/python adapters/mini_swe/h5_summary.py "$OUT" --baseline baseline --json "$OUT/h5_summary.json" \
  > "$OUT/h5_summary.md" 2> "$OUT/h5_summary.err"
st "h5 summary exit=$?"
st "leak check: $(leak_check) file(s) under $OUT contain a key value"
sync_out
st "done; openrouter_usage=$(or_usage); disk $(df -BG / | awk 'NR==2{print $3"/"$2}')"
