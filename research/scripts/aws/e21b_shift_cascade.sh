#!/usr/bin/env bash
# Exploratory row E21b, the deployed confidence-gated cascades on DecisionGym-Shift (docs/PREREG_A7_STAGE2.md, Deviations
# 2026-10-08 "E21b: the deployed confidence-gated cascades on DecisionGym-Shift"; descriptive, outside every Holm table).
# One NEW GPU host launched by this driver from E3's clone image (tag Name=jev-e21b-shift-cascade), SSM only. Each model
# is served alone on the GPU as in its logged final-harness row; the final harness runs at the model's in-domain
# thresholds (no refit) on E21's 200 shifted L3 tasks, Flash on the direct Gemini API (the key from SSM Parameter Store
# into the live stage's process only; never printed, never on disk).
#
#   LOCAL:  bash scripts/aws/e21b_shift_cascade.sh launch [itype ...]   # one new host (refuses if a jev-e21b host exists);
#                                                                       # default g5.2xlarge, then g6e.2xlarge; user-data
#                                                                       # arms the power-off guard (GUARD_MIN, 230 min)
#           E21B_IID=i-... bash scripts/aws/e21b_shift_cascade.sh push | send "STAGES" | status [N] | stop | itype
#   BOX stages (in the order sent; a failed rearm / setup stops the chain; `final` always runs when sent):
#     rearm        idle stop (IDLE_MIN) + power-off guard (GUARD_END absolute UTC, else GUARD_MIN), host, GPU, disk
#     setup        the committed branch (bundle) in /opt/jev/e21b; the Unsloth venv for the 0.8B head (as E21 / E18);
#                  the frozen 4B, E12 and E14 arm A from S3, sha256-checked; the calibration questions (sha256-checked)
#     cal_e12      E12 served (serve_jev --fast) answers the 2,719 calibration questions (scripts/e5_cal_replay.py)
#     live_rl      the frozen 4B (serve_jev --fast): smoke, gate, the 200 shifted tasks (rl_final)
#     live_e18     E14 arm A + the defer head (serve_jev --backend clef --clef-defer-head): smoke, gate, e18_final
#     live_e12     E12 (serve_jev --fast) at its committed in-domain fit: smoke, gate, e12_final
#     live_e12_in  the secondary in-domain row (registration (d)), only if the gate and the GPU guard leave room
#     final        stop the server, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e21b_shift_cascade/box on the box -> s3://$BUCKET/e21b_shift_cascade/box; runs -> .../runs.
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E21B_IID=${E21B_IID:-}
# every other project host (never touched by this driver): the never-touch instances, E21's, E22's, E23's hosts and every
# earlier row's host; a host is only ever addressed after its Name tag is checked to be this driver's
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e21b-shift-cascade
BUNDLE=e21b.bundle
S3OUT="$B/e21b_shift_cascade"
CAP_USD=${CAP_USD:-35}
GUARD_MIN=${GUARD_MIN:-230}          # 4 GPU host-hours (registration (g)); GUARD_END (absolute UTC) wins when set
IDLE_MIN=${IDLE_MIN:-30}
POWEROFF=${POWEROFF:-1}
CAL_REQ_SHA=d46701e8e9bdc69e6a9e047227eabb1e52578ddc527342dda10b33f7dc5fff5e

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E21B_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E21B_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E21B_IID "* ]] && { echo "refused: $E21B_IID is not the E21b host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E21B_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E21B_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e21b-shift-cascade: power-off guard (4 GPU host-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g5.2xlarge g6e.2xlarge)
  for IT in "${types[@]}"; do
    for SUBNET in $SUBNETS; do
      out=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$IT" --subnet-id "$SUBNET" --security-group-ids "$SG" \
        --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior stop --count 1 \
        --user-data "file://$ud" --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
        --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name,LaunchTime]' --output text 2>&1)
      rc=$?
      echo "$IT $SUBNET: $(echo "$out" | tail -1 | cut -c1-200)"
      (( rc == 0 )) && { rm -f "$ud"; exit 0; }
      [[ "$out" == *InsufficientInstanceCapacity* || "$out" == *Unsupported* ]] || { rm -f "$ud"; exit 1; }
    done
  done
  rm -f "$ud"; exit 1
fi
if [[ "$MODE" == push ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null || { echo "bundle failed"; exit 1; }
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors || exit 1
  git -C "$ROOT" show "$BRANCH:scripts/aws/e21b_shift_cascade.sh" > "$tmp/e21b_shift_cascade.sh"   # the committed driver
  awsl s3 cp "$tmp/e21b_shift_cascade.sh" "$B/code/e21b_shift_cascade.sh" --only-show-errors || exit 1
  req="$ROOT/runs/e21b_cal/cal_requests.jsonl.gz"     # the calibration questions (built locally: cal-build)
  [[ "$(sha256sum "$req" 2>/dev/null | cut -d' ' -f1)" == "$CAL_REQ_SHA" ]] || { echo "no registered calibration questions at $req"; exit 1; }
  awsl s3 cp "$req" "$S3OUT/inputs/cal_requests.jsonl.gz" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e21b_shift_cascade.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e21b_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e21b_shift_cascade.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q GUARD_MIN=%q GUARD_END=%q IDLE_MIN=%q POWEROFF=%q E21B_IID=%q\n' \
      "$CAP_USD" "$GUARD_MIN" "${GUARD_END:-}" "$IDLE_MIN" "$POWEROFF" "$E21B_IID"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e21b_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e21b_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E21B_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e21b_driver.sh /tmp/e21b_driver.sh --only-show-errors && bash /tmp/e21b_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 stop-instances --instance-ids "$E21B_IID" --output text --query 'StoppingInstances[].[InstanceId,CurrentState.Name]'
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E21B_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime,StateTransitionReason]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,26p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
W=${W:-/opt/jev}
cd "$W" || exit 1
imds() {
  curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" "http://169.254.169.254/latest/meta-data/$1"
}
MY_IID=$(imds instance-id)
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E21B_IID" && "$MY_IID" != "$E21B_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E21b host ${E21B_IID:-?}"; exit 1
fi
R=$W/e21b
A=$R/results/explore/e21b_shift_cascade/box
STATUS_DIR=$W/e21b_status
HV=$W/hvenv                 # harness venv (runner, serve_jev for the 4B rows)
EV=$W/e14venv               # Unsloth 2026.10.2: serve_jev --backend clef (the 0.8B head, as E14 / E18 / E21)
CK=$W/ck                    # the frozen 4B: $CK/<plan dir>
CKD=$W/ck_e21b              # E12, E14 arm A
PLAN=results/stage2/rl_eval_plan.json
EX=scripts/explore_e21b_shift_cascade.py
CFG=configs/explore/e21b_shift_cascade
HEAD_FILE=results/explore/e17_e18/e18_head.json
HEAD_SHA=23dbbbee77c7f9df29ef6613ce703068a89c378f08340cd2748aec8a7e09187f
STOP=runs/e21b_STOP.json
PROBE_LOG=results/explore/e21b_shift_cascade/box/api_probe_log.txt
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$STATUS_DIR" "$CKD"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  aws s3 cp "$STATUS_DIR/status.txt" "$S3OUT/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION"
  local d
  for d in "$R"/runs/e21b_*; do
    [[ -d "$d" && "$(basename "$d")" != e21b_cal ]] && aws s3 sync "$d" "$S3OUT/runs/$(basename "$d")" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  true
}
PY() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }
plan() { (cd "$R" && PY -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"); }
qbase() { cat "$W/base_model_path.txt"; }
frozen_dir() { echo "$CK/$(plan "d['rows']['frozen']['dir']")"; }
frozen_cal() { echo "$CK/$(plan "d['rows']['frozen']['calibration']")"; }
first_boot() { cat "$W/e21b_first_boot" 2>/dev/null || date -d "$(uptime -s)" +%s; }
hours_used() { python3 -c "import time; print(round((time.time() - $(first_boot)) / 3600, 3))"; }
mins_left() { echo $(( GUARD_MIN - ( $(date +%s) - $(first_boot) ) / 60 )); }
keys() {  # the direct Gemini API key(s) from SSM into this stage's environment only (never on disk, never printed)
  local val v
  for v in GEMINI_API_KEY GEMINI_API_KEYS; do
    val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption \
          --query Parameter.Value --output text 2>/dev/null) || val=""
    [[ -n "$val" && "$val" != None ]] && export "$v=$val"
  done
  unset val
  [[ -n "${GEMINI_API_KEY:-}" ]] || { echo "GEMINI_API_KEY missing in SSM"; return 1; }
}
leak_check() {  # no key value in any record or status file
  local n=0 k ks=${GEMINI_API_KEYS:-}
  for k in ${GEMINI_API_KEY:-} ${ks//,/ }; do
    [[ ${#k} -ge 16 ]] || continue
    n=$(( n + $(grep -rlF -- "$k" "$R/runs" "$A" "$STATUS_DIR" 2>/dev/null | wc -l) ))
  done
  echo "key-leak files: $n"
  [[ "$n" == 0 ]]
}
sha_check() {  # sha_check FILE REFERENCE_SHA256 LABEL
  local got; got=$(sha256sum "$1" 2>/dev/null | cut -d' ' -f1)
  echo "$3 $(basename "$1") $got $([[ "$got" == "$2" ]] && echo ok || echo "MISMATCH (want $2)")" >> "$A/ckpt_sha256.txt"
  [[ "$got" == "$2" ]]
}
verify_frozen() {
  (cd "$R" && PY - "$PLAN" "$CK" "$A/frozen_verify.json" <<'EOF'
import hashlib, json, os, sys
plan, ck, out = sys.argv[1:4]
r = json.load(open(plan))["rows"]["frozen"]
res, bad = {}, []
for rel, sha in (r.get("sha256") or {}).items():
    p = os.path.join(ck, r.get("dir") or "", rel)
    got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    res[rel] = {"expected": sha, "got": got, "ok": got == sha}
    bad += [rel] if got != sha else []
json.dump({"ok": not bad and bool(res), "mismatch": bad, "files": res}, open(out, "w"), indent=1)
print("frozen checkpoint sha256:", "all match" if not bad and res else f"MISMATCH {bad}")
sys.exit(1 if bad or not res else 0)
EOF
  )
}
prewarm() {  # read files in 64 MB chunks, 32 at a time (a new EBS volume loads lazily from its snapshot)
  local d
  for d in "$@"; do
    find -L "$d" -type f -size +1M 2>/dev/null | while read -r f; do
      local s n i; s=$(stat -L -c %s "$f"); n=$(( (s + 67108863) / 67108864 ))
      for i in $(seq 0 $((n - 1))); do echo "$f $i"; done
    done
  done | xargs -P 32 -n 2 sh -c 'dd if="$0" of=/dev/null bs=64M skip=$1 count=1 status=none'
}

# ------------------------------------------------------------------------------------------------ servers
stop_server() {
  local f p
  for f in "$W"/e21b_srv_*.pid; do
    [[ -f "$f" ]] || continue
    p=$(cat "$f"); kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$p" 2>/dev/null; kill -9 "$p" 2>/dev/null
    rm -f "$f"
  done
  sleep 3
  true
}
serve() {  # serve SLUG: one model alone on the GPU (the serving of its logged final-harness row)
  local slug=$1 port name
  stop_server
  case "$slug" in
    rl) verify_frozen || return 1; port=8081; name=jevagentrl-4b
        set -- env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev --model "$(qbase)" --name "$name" \
          --port "$port" --adapter "$(frozen_dir)" --calibration "$(frozen_cal)" --fast ;;
    e12) sha_check "$CKD/e12/adapter_model.safetensors" 5c9542d5eb264c726488f2efa8896de4202df66b00c500e26f5a2801b544fa5e e12 || return 1
         sha_check "$CKD/e12/calibration.json" 5b027abfd95a65af8e9375b2b90aa7ffc463528d91d31c7a35eee780b254323e e12 || return 1
         port=8081; name=agentdecision-4b-mixed
         set -- env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev --model "$(qbase)" --name "$name" \
           --port "$port" --adapter "$CKD/e12" --calibration "$CKD/e12/calibration.json" --fast ;;
    e18) sha_check "$CKD/e14a/joint_head.safetensors" a7ce7a7e78d50305a34fa900a69a9b6544cc8e6b3bfe7aa4caf9cb8c60a1c200 e14a || return 1
         sha_check "$R/$HEAD_FILE" "$HEAD_SHA" e18_head || return 1
         port=8083; name=e18
         set -- env PYTHONPATH="$R/src" "$EV/bin/python" -m jevharness.training.serve_jev --backend clef --model "$CKD/e14a" \
           --name "$name" --port "$port" --clef-defer-head "$R/$HEAD_FILE" ;;
    *) echo "unknown slug $slug"; return 1 ;;
  esac
  echo "$(date -u +%FT%TZ) ${*:2}" >> "$A/logs/serve_$slug.cmd"
  setsid nohup "$@" > "$A/logs/serve_$slug.out" 2>&1 < /dev/null &
  echo $! > "$W/e21b_srv_$slug.pid"
  local end=$(( $(date +%s) + 40 * 60 ))
  while (( $(date +%s) < end )); do
    if curl -sf -m 10 "http://127.0.0.1:$port/health" > "$A/logs/health_$slug.json" 2>/dev/null; then
      curl -sf -m 10 "http://127.0.0.1:$port/v1/models" > "$A/logs/models_$slug.json" 2>/dev/null
      grep -q "\"$name\"" "$A/logs/models_$slug.json" || { echo "served model id is not $name"; return 1; }
      printf '{"server_up": "%s", "slug": "%s", "name": "%s", "port": %s, "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" \
        "$slug" "$name" "$port" "$(imds instance-type)" >> "$A/serve_windows.jsonl"
      return 0
    fi
    kill -0 "$(cat "$W/e21b_srv_$slug.pid")" 2>/dev/null || { echo "server died"; tail -30 "$A/logs/serve_$slug.out"; return 1; }
    sleep 10
  done
  echo "server did not come up in 40 min"; return 1
}

# ------------------------------------------------------------------------------------------------ live (pause rule)
probe_ok() {  # two consecutive clean probe rounds (E14's rule, registration (e)); waits up to PROBE_WAIT_MIN
  local end=$(( $(date +%s) + ${PROBE_WAIT_MIN:-60} * 60 )) clean=0
  while (( $(date +%s) < end )); do
    if PY "$EX" probe --log "$PROBE_LOG" > /dev/null 2>&1; then clean=$((clean + 1)); else clean=0; fi
    (( clean >= 2 )) && return 0
    sleep $(( clean > 0 ? 30 : 300 ))
  done
  return 1
}
PAUSED_S=0
run_live() {  # run_live CONFIG RUN [jevbench args]: probe-gated, paused on 503 / overload, capped; one re-run of errors
  local cfg=$1 run=$2 pid wpid spid round t0 ne; shift 2
  for round in 1 2 3 4 5 6 7 8 9 10; do
    [[ -f "$STOP" ]] && { st "spend cap reached earlier; $cfg not run further"; return 3; }
    t0=$(date +%s)
    probe_ok || { st "probes not clean for ${PROBE_WAIT_MIN:-60} min; $cfg stops (round $round)"; return 4; }
    (( round > 1 )) && PAUSED_S=$(( PAUSED_S + $(date +%s) - t0 ))
    st "$cfg: start (round $round) $*"
    setsid env PYTHONPATH="$R/src" JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$cfg" --out runs "$@" >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PY "$EX" watch --pid "$pid" --run "$run" --runs runs --log "$PROBE_LOG" --cap "$CAP_USD" --every 30 >> "$A/watch.log" 2>&1 &
    wpid=$!
    ( while kill -0 "$pid" 2>/dev/null; do sleep 300; sync_out; done ) & spid=$!
    wait "$pid"
    kill "$wpid" "$spid" 2>/dev/null
    sync_out
    if [[ -f "runs/$run/PAUSED_overload.json" ]]; then
      st "$cfg paused on an overload error (round $round); waiting for clean probes"
      mv -f "runs/$run/PAUSED_overload.json" "$A/paused_${run}_round$round.json"
      continue
    fi
    break
  done
  [[ -f "$STOP" ]] && return 3
  ne=$(PY "$EX" errored --run "runs/$run")
  if [[ "$ne" != 0 ]]; then
    st "$ne errored episode(s) in $run after full coverage: one re-run after clean probes"
    t0=$(date +%s)
    probe_ok || { st "probes not clean; re-run skipped"; return 0; }
    PAUSED_S=$(( PAUSED_S + $(date +%s) - t0 ))
    setsid env PYTHONPATH="$R/src" JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$cfg" --out runs "$@" >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PY "$EX" watch --pid "$pid" --run "$run" --runs runs --log "$PROBE_LOG" --cap "$CAP_USD" --every 30 --no-pause >> "$A/watch.log" 2>&1 &
    wpid=$!
    wait "$pid"; kill "$wpid" 2>/dev/null
  fi
  st "$cfg done; errored now $(PY "$EX" errored --run "runs/$run"); $(PY "$EX" spend --runs runs --log "$PROBE_LOG" --short 2>/dev/null)"
  sync_out
  [[ -f "$STOP" ]] && return 3
  return 0
}
live_arm() {  # live_arm SLUG [MIN_MINUTES]: serve, smoke, gate, the row, window record, stop (registration (c), (e))
  local slug=$1 need=${2:-0} g nfit rc t0 mt=null
  cd "$R" || return 1
  if (( need > 0 && $(mins_left) < need )); then st "$slug: not run ($(mins_left) min left of the GPU guard, $need needed)"; return 0; fi
  if [[ "$slug" == e12 || "$slug" == e12_in ]]; then
    grep -q '^status: development' configs/harness/explore_e21b_e12_final.yaml || { echo "E12's profile is still pending"; return 1; }
    grep -q 'E21b: E12' docs/PREREG_A7_STAGE2.md || { echo "no amendment with E12's fitted thresholds"; return 1; }
  fi
  keys || return 1
  serve "${slug%_in}" > "$A/logs/serve_${slug}_start.log" 2>&1 || { st "$slug: did not serve: $(tail -3 "$A/logs/serve_${slug}_start.log" | tr '\n' ' ')"; return 1; }
  run_live "$CFG/smoke_$slug.yaml" "e21b_smoke_$slug" || { st "$slug: smoke failed"; stop_server; return 1; }
  g=$(PY "$EX" gate --smoke "runs/e21b_smoke_$slug" --runs runs --log "$PROBE_LOG" --n 200 --cap "$CAP_USD")
  echo "{\"slug\": \"$slug\", \"gate\": $g}" >> "$A/gates.jsonl"
  nfit=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['n_fit'])" "$g" 2>/dev/null || echo 0)
  st "$slug gate: $g"
  if [[ "$slug" == e12_in && "$nfit" != 200 ]]; then st "e12_in: not run (registration (d): the gate does not fit all 200)"; stop_server; return 0; fi
  (( nfit > 0 )) || { st "$slug: the gate refuses the row"; stop_server; return 3; }
  printf 'arm %s\nstarted %s\ncommit %s\nconfig %s %s\nprereg sha256 %s\ncap_usd %s\nhost %s %s\n' "$slug" "$(date -u +%FT%TZ)" \
    "$(git rev-parse HEAD)" "$CFG/live_$slug.yaml" "$(sha256sum "$CFG/live_$slug.yaml" | cut -c1-16)" \
    "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" "$CAP_USD" "$MY_IID" "$(imds instance-type)" > "$A/START_$slug.txt"
  t0=$(date -u +%FT%TZ)
  PAUSED_S=0
  if (( nfit < 200 )); then mt=$nfit; run_live "$CFG/live_$slug.yaml" "e21b_$slug" --max-tasks "$nfit"
  else run_live "$CFG/live_$slug.yaml" "e21b_$slug"; fi
  rc=$?
  printf '{"window_start": "%s", "window_end": "%s", "instance_type": "%s", "instance_id": "%s", "workers": 1, "run": "runs/e21b_%s", "max_tasks": %s, "paused_s": %s, "exit": %s}\n' \
    "$t0" "$(date -u +%FT%TZ)" "$(imds instance-type)" "$MY_IID" "$slug" "$mt" "$PAUSED_S" "$rc" > "$A/window_$slug.json"
  curl -sf -m 10 "http://127.0.0.1:$([[ $slug == e18 ]] && echo 8083 || echo 8081)/health" > "$A/logs/health_${slug}_end.json" 2>/dev/null
  stop_server
  leak_check || { unset GEMINI_API_KEY GEMINI_API_KEYS; return 1; }
  unset GEMINI_API_KEY GEMINI_API_KEYS
  sync_out
  return $rc
}

# ------------------------------------------------------------------------------------------------ stages
stage_rearm() {
  [[ -s "$W/e21b_first_boot" ]] || date -d "$(uptime -s)" +%s > "$W/e21b_first_boot"
  local left
  if [[ -n "${GUARD_END:-}" ]]; then left=$(( ( $(date -u -d "$GUARD_END" +%s) - $(date -u +%s) ) / 60 )); else left=$(mins_left); fi
  (( left > 0 )) || { echo "guard already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$left" "jev-e21b-shift-cascade: power-off guard (GPU cap)"
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E21b chain / server / runner process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e21b_[0-9]|jevbench|explore_e21b|serve_jev|e5_cal_replay|uv pip|pip install' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e21b-shift-cascade: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $left min left, idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU; $(df -BG / | awk 'NR==2{print $4}') free; used $(hours_used) h"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
}
stage_code() {  # the committed branch from the bundle (records already on the host are kept: untracked files)
  local myb="$W/e21b.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A/logs" runs
  st "code at $(git log --oneline -1 | cut -c1-72)"
  grep -q 'E21b: the deployed confidence-gated cascades on DecisionGym-Shift' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E21b"; return 1; }
}
stage_setup() {
  stage_code || return 1
  ( prewarm "$(qbase)" "$HV/lib" ) &
  local pw=$!
  command -v uv > /dev/null || bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  PY -c "import jevharness, torch, transformers, sys; p = jevharness.__file__; print(p, torch.__version__, transformers.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_hvenv.txt" 2>&1 || { cat "$A/env_hvenv.txt"; echo "the hvenv does not import this checkout"; return 1; }
  [[ -x "$HV/bin/jevbench" ]] || { echo "no jevbench in the hvenv"; return 1; }
  grep -q "1cfa9a7208912126459214e8b04321603b3df60c" "$W/base_model_path.txt" || { echo "the base snapshot is not Qwen3-4B 1cfa9a72"; return 1; }
  cp -f "$W/base_model_path.txt" "$A/base_model_path.txt"
  if [[ ! -x "$EV/bin/python" ]] || ! "$EV/bin/python" -c "import unsloth" > /dev/null 2>&1; then
    uv venv -q -p 3.11 "$EV" || return 1
    timeout 45m uv pip install -q --python "$EV" "unsloth==2026.10.2" fastapi uvicorn httpx pydantic pyyaml tqdm \
      python-dotenv openai > "$A/logs/e14venv_install.log" 2>&1 || echo "Unsloth venv install failed (the 0.8B head will not serve)"
    timeout 20m uv pip install -q --python "$EV" flash-linear-attention >> "$A/logs/e14venv_install.log" 2>&1 \
      || echo "flash-linear-attention not installed (the torch path is used)" | tee -a "$A/logs/e14venv_install.log"
  fi
  "$EV/bin/python" -c "import unsloth, torch, transformers; print('unsloth', unsloth.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.is_available())" \
    > "$A/env_e14venv.txt" 2>&1 || echo "e14venv check failed: $(tail -2 "$A/env_e14venv.txt")"
  : > "$A/ckpt_sha256.txt"
  if ! verify_frozen; then
    aws s3 sync "$(plan "d['rows']['frozen']['s3']")" "$(frozen_dir)" --only-show-errors --region "$REGION" || return 1
    verify_frozen || return 1
  fi
  aws s3 sync "$B/checkpoints/e12_mixed/" "$CKD/e12" --only-show-errors --region "$REGION" --exclude 'checkpoint-*' --exclude '*.bin' || return 1
  sha_check "$CKD/e12/adapter_model.safetensors" 5c9542d5eb264c726488f2efa8896de4202df66b00c500e26f5a2801b544fa5e e12 || return 1
  sha_check "$CKD/e12/calibration.json" 5b027abfd95a65af8e9375b2b90aa7ffc463528d91d31c7a35eee780b254323e e12 || return 1
  aws s3 sync "$B/checkpoints/e14_small/A/final/" "$CKD/e14a" --only-show-errors --region "$REGION" || return 1
  sha_check "$CKD/e14a/joint_head.safetensors" a7ce7a7e78d50305a34fa900a69a9b6544cc8e6b3bfe7aa4caf9cb8c60a1c200 e14a || return 1
  sha_check "$CKD/e14a/unsloth_decision_config.json" a9f94cadfa2bacf5702883936c48da8c157e681b30bd80cabb963c9dce998ed4 e14a || true
  sha_check "$R/$HEAD_FILE" "$HEAD_SHA" e18_head || return 1
  mkdir -p "$A/cal"
  aws s3 cp "$S3OUT/inputs/cal_requests.jsonl.gz" "$W/e21b_cal_requests.jsonl.gz" --only-show-errors --region "$REGION" || return 1
  sha_check "$W/e21b_cal_requests.jsonl.gz" "$CAL_REQ_SHA" cal_requests || return 1
  wait "$pw"
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s", "gpu": "%s", "mins_left": %s}\n' "$MY_IID" \
    "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" \
    "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" "$(mins_left)" > "$A/host.json"
  cat "$A/ckpt_sha256.txt" "$A/env_hvenv.txt" "$A/env_e14venv.txt" 2>/dev/null
  df -BG / | awk 'NR==2{print "disk free: "$4}'
}
stage_cal_e12() {  # E12's answers to the calibration questions (registration (b) 3); no LLM call, $0
  cd "$R" || return 1
  serve e12 > "$A/logs/serve_cal_e12_start.log" 2>&1 || { tail -5 "$A/logs/serve_cal_e12_start.log"; return 1; }
  local t0; t0=$(date -u +%FT%TZ)
  PY scripts/e5_cal_replay.py answer --requests "$W/e21b_cal_requests.jsonl.gz" --url http://127.0.0.1:8081 \
    --name agentdecision-4b-mixed --out "$A/cal/answers_e12_cal.jsonl" --timeout 120 > "$A/logs/cal_e12.log" 2>&1
  local rc=$?
  printf '{"window_start": "%s", "window_end": "%s", "exit": %s}\n' "$t0" "$(date -u +%FT%TZ)" "$rc" > "$A/cal/window.json"
  tail -2 "$A/logs/cal_e12.log"
  stop_server
  sync_out
  return $rc
}
stage_live_rl() { live_arm rl; }
stage_live_e18() { live_arm e18; }
stage_live_e12() { live_arm e12; }
stage_live_e12_in() { live_arm e12_in 45; }
stage_final() {
  stop_server
  nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  st "final: used $(hours_used) h; $(cd "$R" && PY "$EX" spend --runs runs --log "$PROBE_LOG" --short 2>/dev/null)"
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; sudo shutdown -h +2 "jev-e21b-shift-cascade: done"; fi
}

st "chain start: $STAGES (host $MY_IID, used $(hours_used) h)"
for s in $STAGES; do
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $s"; continue; fi
  t0=$(date +%s)
  cd "$R" 2>/dev/null || cd "$W"
  ( "stage_$s" ) > "$STATUS_DIR/stage_$s.log" 2>&1
  rc=$?
  [[ -d "$A/logs" ]] && cp -f "$STATUS_DIR/stage_$s.log" "$A/logs/stage_$s.log"
  st "stage $s exit=$rc ($(( ($(date +%s) - t0) / 60 )) min): $(tail -2 "$STATUS_DIR/stage_$s.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  if (( rc != 0 )) && [[ "$s" == rearm || "$s" == code || "$s" == setup ]]; then
    st "chain stopped at $s"
    [[ " $STAGES " == *" final "* ]] && stage_final
    exit 1
  fi
done
st "chain end [$STAGES]"
