#!/usr/bin/env bash
# Exploratory row E28, our recipe on StartLux-Decision-0.8B (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 "E28: our
# recipe on StartLux-Decision-0.8B"; descriptive, outside every Holm table). One NEW GPU host launched by this driver from
# E3's clone image (tag Name=jev-e28-sl08b), SSM only. StartLux-Decision-0.8B (CC BY-NC 4.0; revision bd4f76a6, the E2
# ladder's) + a fresh LoRA, E12c's recipe (SFT, 2 epochs) on E12c's mix with E27's surface augmentation on a seeded half
# of its agent states and E12's public items at 1 : 1; E12c's checkpoint selection and temperatures; L1 (E1), L2 C, L2
# shift (main, surface-undone), the calibration answers ($0); the untrained StartLux-0.8B on the shift (L2, then L3) by
# the vendor's server; L3 shift, L3 in-domain; the gated final harness in-domain and (caps allowing) on the shift. Flash
# on the direct Gemini API (the key from SSM into the live stages' process only, never printed or on disk).
#
#   LOCAL:  bash scripts/aws/e28_sl08b.sh launch [itype ...]   # one new host (refuses if a jev-e28-sl08b host exists);
#                                                              # default g6e.2xlarge, then g5.2xlarge, every zone
#           E28_IID=i-... bash scripts/aws/e28_sl08b.sh push | send "STAGES" | status [N] | start | stop | itype | state
#   BOX stages (in the order sent; a failed rearm / code / setup / data / train / select / temps / wait_prev stops the
#   chain; `final` always runs when sent):
#     rearm        idle stop (IDLE_MIN) + power-off guard (GUARD_MIN from the first boot), host record
#     code         the committed branch in /opt/jev/e28 (records already on the host are kept: untracked files)
#     setup        code; peft into the v2 venv; StartLux-0.8B at the pinned revision (sha256, licence); E12's data
#                  (sha256), E1's dev prefix; the calibration questions (sha256)
#     data         E28's build (registration (d); vocabulary check and seed guard first; refuses on any overlap)
#     train        fresh LoRA, E12c's recipe, 150-minute cap, OOM fallback only; epoch-end adapters -> S3
#     select       epoch-end adapters on E1's dev prefix + v1 dev (the higher mean; tie: epoch 2)
#     temps        temperatures of the selected adapter on E12's cal_dev
#     e1 | c_replay | shift_l2 | cal    served on :8081 (startlux-0.8b-mixed, v2 venv): L1, L2 C, L2 shift, cal answers
#     base_l2      the vendor's server (StartLux-0.8B untrained, :8092): L2 shift (main, then surface-undone)
#     shift | shift_base | live_in      L3 shift (E28), L3 shift (untrained), L3 in-domain: smoke, gate, row
#     wait_prev    wait for every older E28 chain on this host to end
#     vendor_e1    the merged checkpoint behind the vendor's server: E1's decisions, timing only (amendment of ~16:25)
#     final_in | final_shift            the gated final harness (profile explore_e28_final.yaml fitted and committed)
#     final        stop the servers, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e28_sl08b/box -> s3://$BUCKET/e28_sl08b/box; runs -> e28_sl08b/runs; the adapters ->
# s3://$BUCKET/checkpoints/e28_sl08b/ (no optimizer state).
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E28_IID=${E28_IID:-}
FORBIDDEN_FILE=configs/explore/e28_sl08b/forbidden_iids.txt   # every other project host (registration (h))
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e28-sl08b
BUNDLE=e28.bundle
S3OUT="$B/e28_sl08b"
CAP_USD=${CAP_USD:-30}               # registration (h): direct Gemini at most 30 USD
GUARD_MIN=${GUARD_MIN:-470}          # 8 host-hours: the guard 7 h 50 min after the first boot
IDLE_MIN=${IDLE_MIN:-30}
POWEROFF=${POWEROFF:-1}
PORT=8081                            # serve_jev (v2 venv): StartLux-0.8B (mixed)
PORTB=8092                           # the vendor's server: StartLux-0.8B untrained (its logged rows' port)
NAME=startlux-0.8b-mixed
BASE_REV=bd4f76a600e23227547fee7bfc1825e12a32764c
CAL_REQ_SHA=d46701e8e9bdc69e6a9e047227eabb1e52578ddc527342dda10b33f7dc5fff5e
E21_Q_SHA=827a7678f87c586ae2da9e72d74a5a3c7fd0181e6fab5ba8b26639f661283397

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
forbidden_local() {
  local root; root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  grep -oE '^i-0[0-9a-f]{16}' "$root/$FORBIDDEN_FILE" | tr '\n' ' '
}
iid_ok() {
  [[ "$E28_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E28_IID is not set to an instance id"; return 1; }
  local forb; forb=$(forbidden_local)
  [[ -n "$forb" ]] || { echo "no forbidden list ($FORBIDDEN_FILE)"; return 1; }
  [[ " $forb " == *" $E28_IID "* ]] && { echo "refused: $E28_IID is not the E28 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E28_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E28_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e28-sl08b: power-off guard (8 host-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g6e.2xlarge g5.2xlarge)
  for IT in "${types[@]}"; do
    for SUBNET in $SUBNETS; do
      out=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$IT" --subnet-id "$SUBNET" --security-group-ids "$SG" \
        --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior stop --count 1 \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=300,VolumeType=gp3,Throughput=500,Iops=6000,DeleteOnTermination=true}' \
        --user-data "file://$ud" --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
        --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name]' --output text 2>&1)
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/e28_sl08b.sh" > "$tmp/e28_sl08b.sh"   # the committed driver
  awsl s3 cp "$tmp/e28_sl08b.sh" "$B/code/e28_sl08b.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e28_sl08b.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e28_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e28_sl08b.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q GUARD_MIN=%q IDLE_MIN=%q POWEROFF=%q E28_IID=%q WAIT_MAX_MIN=%q PROBE_WAIT_MIN=%q SHIFT_FINAL_MIN=%q FORBIDDEN_IIDS=%q\n' \
      "$CAP_USD" "$GUARD_MIN" "$IDLE_MIN" "$POWEROFF" "$E28_IID" "${WAIT_MAX_MIN:-420}" "${PROBE_WAIT_MIN:-60}" \
      "${SHIFT_FINAL_MIN:-50}" "$(forbidden_local)"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e28_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e28_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E28_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e28_driver.sh /tmp/e28_driver.sh --only-show-errors && bash /tmp/e28_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E28_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype || "$MODE" == state ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E28_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime,StateReason.Code]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,33p' "$0"
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
if [[ -z "${FORBIDDEN_IIDS:-}" || " ${FORBIDDEN_IIDS:-} " == *" $MY_IID "* || -z "$E28_IID" || "$MY_IID" != "$E28_IID" ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E28 host ${E28_IID:-?} (or no forbidden list)"; exit 1
fi
R=$W/e28
A=$R/results/explore/e28_sl08b/box
STATUS_DIR=$W/e28_status
HV=$W/hvenv                 # harness venv (runner, replays, E12's data tools)
VV=$W/v2venv                # A7.19 (f)'s v2 venv (+ peft): training and serve_jev on the StartLux base, as E12c
SLB=/opt/startlux           # StartLux: the vendor's package, venv and weights (serve_startlux.sh)
SL08=$SLB/models/StartLux-Decision-0.8B
D=$W/e28_data
CKC=$W/ck_e28/train         # the trainer's output: checkpoint-<step> per epoch + the final adapter
SEL=$W/ck_e28/selected      # the selected adapter (+ temperatures)
EX=scripts/explore_e28_sl08b.py
E12X=scripts/explore_e12_train.py
CFG=configs/explore/e28_sl08b
STOP=runs/e28_STOP.json
PROBE_LOG=$A/api_probe_log.txt
V1=(data/train_v1/dev.jsonl.gz data/train_v1/dev_recovery.jsonl.gz data/train_v1/dev_knockout.jsonl.gz)
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$STATUS_DIR"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  aws s3 cp "$STATUS_DIR/status.txt" "$S3OUT/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION" --exclude 'data/*' \
    --exclude 's23/corpus.json' --exclude 's23/occurrences.jsonl.gz' --exclude 's23/questions.jsonl.gz' \
    --exclude 'c_shift/questions*' --exclude 'c_shift/occurrences.jsonl.gz' --exclude 'c_shift/corpus.json'
  local d
  for d in "$R"/runs/e1_bfcl_when2call_e28* "$R"/runs/e28_*; do
    [[ -d "$d" ]] && aws s3 sync "$d" "$S3OUT/runs/$(basename "$d")" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  true
}
PY() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }
PV() { PYTHONPATH="$R/src" "$VV/bin/python" "$@"; }
first_boot() { cat "$W/e28_first_boot" 2>/dev/null || date -d "$(uptime -s)" +%s; }
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
  echo "$3 $(basename "$1") $got $([[ "$got" == "$2" ]] && echo ok || echo "MISMATCH (want $2)")" >> "$A/sha256_checks.txt"
  [[ "$got" == "$2" ]]
}
smi_start() { nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 1000 > "$1" 2>/dev/null & SMI=$!; }
smi_stop() { [[ -n "${SMI:-}" ]] && kill "$SMI" 2>/dev/null; SMI=""; }
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
stop_servers() {
  local p
  if [[ -f "$W/e28_serve.pid" ]]; then
    p=$(cat "$W/e28_serve.pid"); kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$p" 2>/dev/null; kill -9 "$p" 2>/dev/null
  fi
  rm -f "$W/e28_serve.pid"
  (cd "$R" 2>/dev/null && STOP=1 BASE=$SLB LOG_DIR=$SLB/logs bash scripts/aws/serve_startlux.sh > /dev/null 2>&1)
  pkill -f 'startlux_decision.server' 2>/dev/null
  sleep 5
  true
}
server_up() { curl -sf -m 10 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; }
serve_c() {  # the selected adapter + temperatures on :8081 (v2 venv; no extra arguments, as E12c)
  stop_servers
  [[ -s "$SEL/adapter_model.safetensors" && -s "$SEL/calibration.json" ]] || { echo "no selected adapter / temperatures"; return 1; }
  local args=(--model "$SL08" --name "$NAME" --port "$PORT" --adapter "$SEL" --calibration "$SEL/calibration.json")
  echo "$(date -u +%FT%TZ) serve_jev ${args[*]} (venv $VV)" >> "$A/logs/serve.cmd"
  setsid nohup env PYTHONPATH="$R/src" "$VV/bin/python" -m jevharness.training.serve_jev "${args[@]}" \
    > "$A/logs/serve.out" 2>&1 < /dev/null &
  echo $! > "$W/e28_serve.pid"
  for _ in $(seq 1 240); do
    if curl -sf -m 10 "http://127.0.0.1:$PORT/health" > "$A/logs/serve_health.json" 2>/dev/null; then
      curl -sf -m 10 "http://127.0.0.1:$PORT/v1/models" > "$A/logs/serve_models.json"
      grep -q "\"$NAME\"" "$A/logs/serve_models.json" || { echo "served model id is not $NAME"; return 1; }
      printf '{"server_up": "%s", "name": "%s", "port": %s, "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" "$NAME" "$PORT" \
        "$(imds instance-type)" >> "$A/serve_windows.jsonl"
      return 0
    fi
    kill -0 "$(cat "$W/e28_serve.pid")" 2>/dev/null || { echo "serve_jev died"; tail -30 "$A/logs/serve.out"; return 1; }
    sleep 5
  done
  echo "serve_jev did not come up"; return 1
}
serve_base() {  # the untrained StartLux-0.8B behind the vendor's server at the pinned revision (its logged rows' path)
  stop_servers
  (cd "$R" && SIZES=0.8B BASE=$SLB LOG_DIR=$SLB/logs PIN_MANIFEST="$R/$CFG/startlux_pin.json" MANIFEST="$A/manifest_0.8B.json" \
    PYTHON=python3 bash scripts/aws/serve_startlux.sh > "$A/logs/serve_base.log" 2>&1)
  python3 -c "import json,sys; m=json.load(open('$A/manifest_0.8B.json')); s=[x for x in m['sizes'] if x['size']=='0.8B'][0]; print(s['status'], s.get('weights_revision'), s.get('port'), s.get('why')); sys.exit(0 if s['status']=='original' and s.get('weights_revision')=='$BASE_REV' and int(s.get('port'))==$PORTB else 1)" \
    || return 1
  printf '{"server_up": "%s", "name": "StartLux-Decision-0.8B", "port": %s, "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" \
    "$PORTB" "$(imds instance-type)" >> "$A/serve_windows.jsonl"
}
spec_c() { printf '{"type": "jev", "name": "%s", "base_url": "http://127.0.0.1:%s", "model": "%s", "api_key": "", "timeout": 120.0, "price_model": "StartLux-Decision-0.8B"}' "$NAME" "$PORT" "$NAME"; }
spec_base() { printf '{"type": "jev", "name": "startlux-0.8b", "base_url": "http://127.0.0.1:%s", "model": "StartLux-Decision-0.8B", "api_key": "", "timeout": 60.0, "price_model": "StartLux-Decision-0.8B"}' "$PORTB"; }

# ------------------------------------------------------------------------------------------------ live (E21's pause rule)
probe_ok() {  # two consecutive clean probe rounds; waits up to PROBE_WAIT_MIN
  local end=$(( $(date +%s) + ${PROBE_WAIT_MIN:-60} * 60 )) clean=0
  while (( $(date +%s) < end )); do
    if PY "$EX" probe --log "$PROBE_LOG" > /dev/null 2>&1; then clean=$((clean + 1)); else clean=0; fi
    (( clean >= 2 )) && return 0
    sleep $(( clean > 0 ? 30 : 300 ))
  done
  return 1
}
run_live() {  # run_live CONFIG RUN_DIR [jevbench args]: probe-gated, paused on 503 / overload, capped; one re-run of errors
  local cfg=$1 run=$2 pid wpid spid round; shift 2
  for round in 1 2 3 4 5 6 7 8; do
    [[ -f "$STOP" ]] && { st "spend cap reached earlier; $cfg not run further"; return 3; }
    probe_ok || { st "probes not clean; $cfg paused (round $round)"; return 4; }
    st "$cfg: start (round $round) $*"
    setsid env PYTHONPATH="$R/src" JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$cfg" --out runs "$@" >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PY "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --log "$PROBE_LOG" --every 30 >> "$A/watch.log" 2>&1 &
    wpid=$!
    ( while kill -0 "$pid" 2>/dev/null; do sleep 300; sync_out; done ) & spid=$!
    wait "$pid"
    kill "$wpid" "$spid" 2>/dev/null
    sync_out
    if [[ -f "$run/PAUSED_overload.json" ]]; then
      st "$cfg paused on an overload error (round $round); waiting for clean probes"
      mv -f "$run/PAUSED_overload.json" "$run/paused_overload_round$round.json"
      continue
    fi
    break
  done
  [[ -f "$STOP" ]] && return 3
  local ne
  ne=$(PY "$EX" errored --run "$run")
  if [[ "$ne" != 0 ]]; then
    st "$ne errored episode(s) in $run: one re-run after clean probes"
    probe_ok || { st "probes not clean; re-run skipped"; return 0; }
    setsid env PYTHONPATH="$R/src" JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$cfg" --out runs "$@" >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PY "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --log "$PROBE_LOG" --every 30 --no-pause >> "$A/watch.log" 2>&1 &
    wpid=$!
    wait "$pid"; kill "$wpid" 2>/dev/null
  fi
  st "$cfg done; errored now $(PY "$EX" errored --run "$run"); $(PY "$EX" spend --runs runs --log "$PROBE_LOG" --short 2>/dev/null)"
  sync_out
  [[ -f "$STOP" ]] && return 3
  return 0
}
live_row() {  # live_row TAG WHICH(e28|base) SMOKE_CFG CFG [MIN_MINUTES]: serve, smoke, gate, the row (registration (g))
  local tag=$1 which=$2 smoke=$3 cfg=$4 need=${5:-0} g nfit rc t0 mt=null
  cd "$R" || return 1
  if (( need > 0 && $(mins_left) < need )); then st "$tag: not run ($(mins_left) min left of the guard, $need needed)"; return 0; fi
  if [[ -f "$STOP" ]]; then st "$tag: not run (the spend cap was reached earlier)"; return 3; fi
  keys || return 1
  if [[ "$which" == base ]]; then
    serve_base > "$A/logs/serve_${tag}_start.log" 2>&1 || { st "$tag: did not serve: $(tail -3 "$A/logs/serve_${tag}_start.log" | tr '\n' ' ')"; return 1; }
  else
    serve_c > "$A/logs/serve_${tag}_start.log" 2>&1 || { st "$tag: did not serve: $(tail -3 "$A/logs/serve_${tag}_start.log" | tr '\n' ' ')"; return 1; }
  fi
  run_live "$CFG/$smoke.yaml" "runs/e28_$smoke" || { st "$tag: smoke failed"; stop_servers; unset GEMINI_API_KEY GEMINI_API_KEYS; return 1; }
  g=$(PY "$EX" gate --smoke "runs/e28_$smoke" --runs runs --log "$PROBE_LOG" --n 200 --cap "$CAP_USD")
  echo "{\"row\": \"$tag\", \"gate\": $g}" >> "$A/gates.jsonl"
  nfit=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['n_fit'])" "$g" 2>/dev/null || echo 0)
  st "$tag gate: $g"
  (( nfit > 0 )) || { st "$tag: the gate refuses the row"; stop_servers; unset GEMINI_API_KEY GEMINI_API_KEYS; return 3; }
  printf 'row %s\nmodel %s\nstarted %s\ncommit %s\nconfig %s %s\nprereg sha256 %s\ncap_usd %s\nhost %s %s\n' "$tag" "$which" \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$CFG/$cfg.yaml" "$(sha256sum "$CFG/$cfg.yaml" | cut -c1-16)" \
    "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" "$CAP_USD" "$MY_IID" "$(imds instance-type)" > "$A/START_$tag.txt"
  t0=$(date -u +%FT%TZ)
  smi_start "$A/logs/smi_live_$tag.csv"
  if (( nfit < 200 )); then mt=$nfit; run_live "$CFG/$cfg.yaml" "runs/e28_$cfg" --max-tasks "$nfit"
  else run_live "$CFG/$cfg.yaml" "runs/e28_$cfg"; fi
  rc=$?
  smi_stop
  printf '{"row": "%s", "model": "%s", "window_start": "%s", "window_end": "%s", "instance_type": "%s", "instance_id": "%s", "run": "runs/e28_%s", "max_tasks": %s, "exit": %s}\n' \
    "$tag" "$which" "$t0" "$(date -u +%FT%TZ)" "$(imds instance-type)" "$MY_IID" "$cfg" "$mt" "$rc" > "$A/window_$tag.json"
  stop_servers
  leak_check || { unset GEMINI_API_KEY GEMINI_API_KEYS; return 1; }
  unset GEMINI_API_KEY GEMINI_API_KEYS
  sync_out
  return $rc
}

# ------------------------------------------------------------------------------------------------ stages
stage_rearm() {
  [[ -s "$W/e28_first_boot" ]] || date -d "$(uptime -s)" +%s > "$W/e28_first_boot"
  local left; left=$(mins_left)
  (( left > 0 )) || { echo "guard already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$left" "jev-e28-sl08b: power-off guard (host-hour cap)"
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E28 chain / runner / trainer / replay process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e28_[0-9]|jevbench|train_decision|eval_decisions|training.calibrate|explore_e28|explore_e21_shift|explore_e12_train|stage2_corpus|decision_eval|e5_cal_replay|snapshot_download|uv pip|pip install' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e28-sl08b: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $left min left, idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU; $(df -BG / | awk 'NR==2{print $4}') free; used $(hours_used) h"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
}
stage_code() {
  local myb="$W/e28.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A/logs" "$A/eval" "$A/sel" "$A/cal" runs
  st "code at $(git log --oneline -1 | cut -c1-72)"
  grep -q 'E28: our recipe on StartLux-Decision-0.8B' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E28"; return 1; }
}
stage_setup() {
  stage_code || return 1
  ( prewarm "$VV/lib" "$HV/lib" ) &
  local pw=$!
  PY -c "import jevharness, torch, sys; p = jevharness.__file__; print(p, torch.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_hvenv.txt" 2>&1 || { cat "$A/env_hvenv.txt"; echo "the hvenv does not import this checkout"; return 1; }
  [[ -x "$HV/bin/jevbench" ]] || { echo "no jevbench in the hvenv"; return 1; }
  command -v uv > /dev/null || bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  "$VV/bin/python" -c "import peft" 2>/dev/null || timeout 15m uv pip install -q --python "$VV" "peft==0.21.2" > "$A/logs/v2venv_peft.log" 2>&1 \
    || { tail -5 "$A/logs/v2venv_peft.log"; return 1; }
  PV -c "import jevharness, torch, transformers, peft, fla, causal_conv1d, sys; p = jevharness.__file__; print(p, torch.__version__, transformers.__version__, peft.__version__, fla.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_v2venv.txt" 2>&1 || { cat "$A/env_v2venv.txt"; echo "the v2 venv does not import this checkout"; return 1; }
  : > "$A/sha256_checks.txt"
  # StartLux-Decision-0.8B at the pinned revision (registration (b)): sha256 and licence recorded
  mkdir -p "$SL08"
  PY - "$BASE_REV" "$SL08" > "$A/logs/hf_startlux_08b.log" 2>&1 <<'EOF' || { tail -5 "$A/logs/hf_startlux_08b.log"; return 1; }
import sys
from huggingface_hub import HfApi, snapshot_download
rev, d = sys.argv[1], sys.argv[2]
snapshot_download("startlux-models/StartLux-Decision-0.8B", revision=rev, local_dir=d)
info = HfApi().model_info("startlux-models/StartLux-Decision-0.8B", revision=rev)
print("sha", info.sha)
print("license", (info.card_data or {}).get("license") if info.card_data else None)
EOF
  grep -q "^sha $BASE_REV" "$A/logs/hf_startlux_08b.log" || { echo "the downloaded revision is not $BASE_REV"; return 1; }
  printf '{"repo": "startlux-models/StartLux-Decision-0.8B", "requested": "%s", "got": "%s", "license": "%s", "at": "%s"}\n' "$BASE_REV" \
    "$(grep '^sha ' "$A/logs/hf_startlux_08b.log" | cut -d' ' -f2)" "$(grep '^license ' "$A/logs/hf_startlux_08b.log" | cut -d' ' -f2-)" \
    "$(date -u +%FT%TZ)" >> "$A/downloads.jsonl"
  (cd "$SL08" && sha256sum ./*.safetensors config.json tokenizer.json 2>/dev/null) > "$A/startlux_08b_sha256.txt"
  for f in LICENSE LICENSE.md LICENSE.txt README.md; do [[ -f "$SL08/$f" ]] && cp -f "$SL08/$f" "$A/startlux_08b_$f"; done
  # E12's data (sha256 against E12's build record), E1's dev prefix, the calibration questions
  mkdir -p "$A/data"
  aws s3 sync "$B/e12/data" "$A/data" --only-show-errors --region "$REGION" || return 1
  PY "$E12X" verify --data "$A/data" > "$A/logs/e12_verify.log" 2>&1 || { tail -5 "$A/logs/e12_verify.log"; return 1; }
  PY "$E12X" devprefix --out "$A/data/e1_devprefix.jsonl.gz" > "$A/devprefix.json" || return 1
  aws s3 cp "$B/e5_small/inputs/cal_requests.jsonl.gz" "$W/e28_cal_requests.jsonl.gz" --only-show-errors --region "$REGION" || return 1
  sha_check "$W/e28_cal_requests.jsonl.gz" "$CAL_REQ_SHA" cal_requests || return 1
  wait "$pw"
  ( prewarm "$SL08" )
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s", "gpu": "%s", "mins_left": %s}\n' "$MY_IID" \
    "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" \
    "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" "$(mins_left)" > "$A/host.json"
  cat "$A/sha256_checks.txt" "$A/env_v2venv.txt" "$A/startlux_08b_sha256.txt"
  df -BG / | awk 'NR==2{print "disk free: "$4}'
  sync_out
}
stage_data() {
  cd "$R" || return 1
  mkdir -p "$D"
  PY "$EX" build --e12-mix "$A/data/train_mix.jsonl.gz" --out "$D" > "$A/logs/build.log" 2>&1 || { tail -5 "$A/logs/build.log"; return 1; }
  cp -f "$D/build_e28.json" "$A/build_e28.json"; cp -f "$D/vocab_check.json" "$A/vocab_check.json"
  st "data: $(python3 -c "import json; d=json.load(open('$A/build_e28.json')); print('vocab overlap', d['vocab_check']['overlap_count'], '; rows', d['mix']['rows'], '(agent', d['agent_rows'], 'public', d['public_rows'], '); augmented states', d['augmentation']['augmented_states'], 'of', d['augmentation']['states'])")"
  sync_out
}
stage_train() {  # E12c (d): fresh LoRA, 2 epochs, OOM fallback only; 150-minute cap (registration (c))
  cd "$R" || return 1
  stop_servers
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$D/vocab_check.json'))['ok'] else 1)" || { echo "vocabulary check not ok"; return 1; }
  local setting rc=1 t0 flags
  for setting in nogc_3072 gc_3072 gc_2048; do
    flags=$(PY scripts/train/ablation_box.py setting "$setting") || return 1
    rm -rf "$CKC"; mkdir -p "$CKC"
    t0=$(date +%s)
    smi_start "$A/logs/train_smi_$setting.csv"
    # shellcheck disable=SC2086
    timeout 150m env PYTHONPATH="$R/src" "$VV/bin/python" -m jevharness.training.train_decision --model "$SL08" \
      --train "$D/train_mix_e28.jsonl.gz" --out "$CKC" --loss soft_ce_brier --brier-weight 0.5 --lora-r 32 --lora-alpha 64 \
      --lr 1e-4 --epochs 2 --max-len 4096 --seed 0 --report-to none $flags > "$A/logs/train_$setting.log" 2>&1
    rc=$?
    smi_stop
    printf '{"setting": "%s", "exit": %s, "minutes": %s}\n' "$setting" "$rc" "$(( ($(date +%s) - t0) / 60 ))" >> "$A/train_runs.jsonl"
    (( rc == 0 )) && break
    grep -qE 'out of memory|OutOfMemoryError' "$A/logs/train_$setting.log" || break
  done
  ls -d "$CKC"/checkpoint-* > "$A/checkpoints.txt" 2>/dev/null
  local c
  for c in "$CKC"/checkpoint-* "$CKC"; do
    [[ -s "$c/adapter_model.safetensors" ]] && sha256sum "$c/adapter_model.safetensors" >> "$A/e28_sha256.txt"
  done
  aws s3 sync "$CKC" "$B/checkpoints/e28_sl08b/train/" --only-show-errors --region "$REGION" \
    --exclude '*optimizer*' --exclude '*.pt' --exclude 'rng_state*' --exclude 'scheduler*'
  st "train: exit $rc; $(grep -oE "'train_runtime': '?[0-9.]+" "$A/logs/train_$setting.log" | tail -1); checkpoints $(wc -l < "$A/checkpoints.txt" 2>/dev/null)"
  sync_out
  (( rc == 0 )) || [[ -n "$(ls -d "$CKC"/checkpoint-* 2>/dev/null)" ]] || { tail -30 "$A/logs/train_$setting.log"; return 1; }
  return 0
}
stage_select() {  # E12c (e): epoch-end adapters on E1's dev prefix + v1 dev; the higher mean of the two (tie: epoch 2)
  cd "$R" || return 1
  stop_servers
  local c i=0 cands=()
  for c in $(ls -d "$CKC"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do [[ -s "$c/adapter_model.safetensors" ]] && cands+=("$c"); done
  [[ ${#cands[@]} -ge 1 ]] || { echo "no epoch checkpoint"; return 1; }
  for c in "${cands[@]}"; do   # the trainer's epoch checkpoints hold no tokenizer: the final one's (the base's) is copied
    cp -n "$CKC"/tokenizer.json "$CKC"/tokenizer_config.json "$CKC"/chat_template.jinja "$c/" 2>/dev/null
    cp -n "$SL08"/tokenizer.json "$SL08"/tokenizer_config.json "$SL08"/chat_template.jinja "$c/" 2>/dev/null
    i=$((i + 1))
    PV -m jevharness.training.eval_decisions --model "$SL08" --adapter "$c" \
      --dev "$A/data/e1_devprefix.jsonl.gz" data/train_v1/dev.jsonl.gz --n-perm 0 --max-len 4096 --batch-size 8 \
      --out "$A/sel/epoch$i.json" > "$A/logs/sel_epoch$i.log" 2>&1 || { tail -20 "$A/logs/sel_epoch$i.log"; return 1; }
    echo "$c" > "$A/sel/epoch$i.path"
  done
  PY - "$A/sel" <<'PYEOF' || return 1
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = []
for f in sorted(d.glob("epoch*.json")):
    ev = json.loads(f.read_text())["files"]
    acc = {Path(k).name.split(".")[0]: v["overall"]["acc"] for k, v in ev.items()}
    rows.append({"epoch": int(f.stem[5:]), "path": (d / f"{f.stem}.path").read_text().strip(), "acc": acc,
                 "mean": (acc["e1_devprefix"] + acc["dev"]) / 2})
best = max(rows, key=lambda r: (r["mean"], r["epoch"]))      # tie: the later epoch (the recipe's endpoint)
out = {"rule": "E28 (c) = E12c (e): higher mean of E1 dev-prefix and v1-dev accuracy; tie -> epoch 2", "candidates": rows,
       "selected": best}
(d / "selection.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out))
PYEOF
  local pick; pick=$(python3 -c "import json; print(json.load(open('$A/sel/selection.json'))['selected']['path'])")
  rm -rf "$SEL"; mkdir -p "$SEL"
  cp -f "$pick"/adapter_model.safetensors "$pick"/adapter_config.json "$SEL/"
  cp -f "$CKC"/tokenizer* "$CKC"/chat_template.jinja "$SEL/" 2>/dev/null || cp -f "$SL08"/tokenizer* "$SL08"/chat_template.jinja "$SEL/" 2>/dev/null
  st "selected: $(python3 -c "import json; s=json.load(open('$A/sel/selection.json'))['selected']; print(s['epoch'], s['acc'])")"
  sync_out
}
stage_temps() {
  cd "$R" || return 1
  stop_servers
  PV -m jevharness.training.calibrate --model "$SL08" --adapter "$SEL" --dev "$A/data/cal_dev.jsonl.gz" \
    --out "$SEL/calibration.json" --min-bucket 200 --batch-size 8 > "$A/logs/calibrate.log" 2>&1 || { tail -20 "$A/logs/calibrate.log"; return 1; }
  cp -f "$SEL/calibration.json" "$A/e28_calibration.json"
  sha256sum "$SEL/adapter_model.safetensors" "$SEL/calibration.json" > "$A/selected_sha256.txt"
  aws s3 sync "$SEL" "$B/checkpoints/e28_sl08b/selected/" --only-show-errors --region "$REGION"
  sync_out
}
stage_e1() {
  cd "$R" || return 1
  server_up || serve_c || return 1
  local k
  smi_start "$A/logs/smi_e1.csv"
  for k in 1 2; do   # two invocations as E1 / E12c (the second resumes any failed request from the cache)
    PY -m jevharness.bench.decision_eval "$CFG/e1_e28.yaml" --out runs --only e28 > "$A/logs/e1_$k.log" 2>&1
    st "e1 invocation $k exit=$?"
  done
  smi_stop
  sync_out
}
stage_c_replay() {
  cd "$R" || return 1
  server_up || serve_c || return 1
  mkdir -p "$A/s23"
  local f k
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz; do cp -f "results/stage2/s23/$f" "$A/s23/$f" || return 1; done
  for k in 1 2 3; do   # the replay resumes
    PY scripts/stage2_corpus.py replay --corpus "$A/s23" --name e28 --s1 "$(spec_c)" --cache cache/s1_e28_c.sqlite \
      > "$A/logs/replay_c_$k.log" 2>&1
  done
  st "corpus C replayed: $(wc -l < "$A/s23/answers_e28.jsonl" 2>/dev/null) answers"
  sync_out
}
shift_replay() {  # shift_replay SLUG SPEC: E21's corpus, main rendering then surface-undone (registration (f) 3 / 6)
  local slug=$1 spec=$2 v k f
  mkdir -p "$A/c_shift"
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz questions_nosurf.jsonl.gz; do
    cp -f "results/explore/e21_shift/c/$f" "$A/c_shift/$f" || return 1
  done
  sha_check "$A/c_shift/questions.jsonl.gz" "$E21_Q_SHA" e21_questions || return 1
  for v in main nosurf; do
    for k in 1 2; do
      PY scripts/explore_e21_shift.py replay --corpus "$A/c_shift" --slug "$slug" --variant "$v" --s1 "$spec" \
        --cache "cache/s1_e28_${slug}_shift_$v.sqlite" --workers 4 > "$A/logs/replay_shift_${slug}_${v}_$k.log" 2>&1 && break
    done
    st "shift L2 $slug $v: $(PY scripts/explore_e21_shift.py quickc --corpus "$A/c_shift" --slug "$slug" --variant "$v" 2>/dev/null)"
  done
}
stage_shift_l2() {
  cd "$R" || return 1
  server_up || serve_c || return 1
  shift_replay e28 "$(spec_c)"
  sync_out
}
stage_cal() {  # the 2,719 calibration questions answered once ($0; registration (f) 7); fit locally
  cd "$R" || return 1
  server_up || serve_c > "$A/logs/serve_cal_start.log" 2>&1 || { tail -5 "$A/logs/serve_cal_start.log"; return 1; }
  local t0 rc; t0=$(date -u +%FT%TZ)
  PY scripts/e5_cal_replay.py answer --requests "$W/e28_cal_requests.jsonl.gz" --url "http://127.0.0.1:$PORT" \
    --name "$NAME" --out "$A/cal/answers_cal.jsonl" --timeout 120 > "$A/logs/cal.log" 2>&1
  rc=$?
  printf '{"window_start": "%s", "window_end": "%s", "exit": %s}\n' "$t0" "$(date -u +%FT%TZ)" "$rc" > "$A/cal/window.json"
  st "cal: exit $rc, $(wc -l < "$A/cal/answers_cal.jsonl" 2>/dev/null) answers"
  sync_out
  return $rc
}
stage_base_l2() {
  cd "$R" || return 1
  serve_base || { st "base_l2: the vendor's server did not come up: $(tail -3 "$A/logs/serve_base.log" | tr '\n' ' ')"; return 1; }
  shift_replay sl08b "$(spec_base)"
  stop_servers
  sync_out
}
stage_vendor_e1() {  # amendment "E28: L1 cost on the untrained model's serving engine": timing only, the GPU alone
  cd "$R" || return 1
  stop_servers
  local M=$W/merged_e28_vendor PKG=$SLB/startlux-decision VPY=$SLB/venv/bin/python chk k
  [[ -s "$SEL/adapter_model.safetensors" ]] || { echo "no selected adapter"; return 1; }
  grep -q "E28: L1 cost on the untrained model's serving engine" docs/PREREG_A7_STAGE2.md || { echo "no amendment"; return 1; }
  rm -rf "$M"
  PV "$EX" merge-vendor --base "$SL08" --adapter "$SEL" --out "$M" > "$A/logs/merge_vendor.log" 2>&1 || { tail -5 "$A/logs/merge_vendor.log"; return 1; }
  cp -f "$M/e28_merge.json" "$A/vendor_merge.json"
  git -C "$PKG" rev-parse HEAD > "$A/vendor_pkg_commit.txt" 2>&1
  chk=$(CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PKG" "$VPY" -m startlux_decision.check "$M" 2>&1 | tail -3)
  echo "$chk" > "$A/logs/vendor_check.txt"
  echo "$chk" | grep -q "fast kernels: active" || { st "vendor_e1: the kernel check failed: $(echo "$chk" | tr '\n' ' ' | cut -c1-200)"; return 1; }
  echo "$(date -u +%FT%TZ) startlux_decision.server --model $M --port $PORTB --name StartLux-Decision-0.8B-e28 (pkg $(cat "$A/vendor_pkg_commit.txt"))" >> "$A/logs/serve_vendor.cmd"
  setsid nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PKG" "$VPY" -m startlux_decision.server --model "$M" --host 127.0.0.1 \
    --port "$PORTB" --name StartLux-Decision-0.8B-e28 > "$A/logs/serve_vendor_e28.out" 2>&1 < /dev/null &
  echo $! > "$W/e28_serve.pid"
  for k in $(seq 1 180); do
    curl -sf -m 10 "http://127.0.0.1:$PORTB/health" > "$A/logs/serve_vendor_health.json" 2>/dev/null && break
    kill -0 "$(cat "$W/e28_serve.pid")" 2>/dev/null || { st "vendor_e1: the vendor's server died: $(tail -3 "$A/logs/serve_vendor_e28.out" | tr '\n' ' ' | cut -c1-200)"; return 1; }
    sleep 5
  done
  curl -sf -m 10 "http://127.0.0.1:$PORTB/health" > /dev/null || { st "vendor_e1: the vendor's server did not come up"; stop_servers; return 1; }
  printf '{"server_up": "%s", "name": "StartLux-Decision-0.8B-e28", "port": %s, "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" \
    "$PORTB" "$(imds instance-type)" >> "$A/serve_windows.jsonl"
  smi_start "$A/logs/smi_vendor_e1.csv"
  for k in 1 2; do
    PY -m jevharness.bench.decision_eval "$CFG/e1_e28_vendor.yaml" --out runs --only e28v > "$A/logs/e1_vendor_$k.log" 2>&1
    st "vendor e1 invocation $k exit=$?"
  done
  smi_stop
  stop_servers
  rm -rf "$M"
  sync_out
}
stage_shift() { live_row shift e28 smoke_shift live_shift; }
stage_shift_base() { live_row shift_base base smoke_shift_base live_shift_base; }
stage_live_in() { live_row live_in e28 smoke_in live_in; }
fitted_ok() {
  grep -q '^status: development' configs/harness/explore_e28_final.yaml || { echo "the E28 profile has no fitted thresholds (status not development)"; return 1; }
  grep -q 'E28: the fitted thresholds' docs/PREREG_A7_STAGE2.md || { echo "no amendment with E28's fitted thresholds"; return 1; }
  if grep -q 'final_rows: not run' configs/harness/explore_e28_final.yaml; then echo "the fit always defers at the tool choice: the final-harness rows are not run"; return 2; fi
  return 0
}
stage_final_in() {
  local rc
  cd "$R" || return 1
  fitted_ok; rc=$?; (( rc == 2 )) && { st "final_in: not run (always defers at the tool choice)"; return 0; }; (( rc == 0 )) || return 1
  live_row final_in e28 smoke_final_in final_in
}
stage_final_shift() {
  local rc
  cd "$R" || return 1
  fitted_ok; rc=$?; (( rc == 2 )) && { st "final_shift: not run (always defers at the tool choice)"; return 0; }; (( rc == 0 )) || return 1
  live_row final_shift e28 smoke_final_shift final_shift "${SHIFT_FINAL_MIN:-50}"
}
stage_wait_prev() {  # wait until every older E28 chain on this host has ended (WAIT_MAX_MIN at most)
  local me t end=$(( $(date +%s) + ${WAIT_MAX_MIN:-420} * 60 ))
  me=$(basename "$0" .sh); me=${me#e28_}
  while (( $(date +%s) < end )); do
    local older=0
    for t in $(ps -eo args | grep -oE '/opt/jev/e28_[0-9]+\.sh box' | grep -oE '[0-9]{6,}' | sort -u); do
      [[ "$t" < "$me" ]] && older=1
    done
    (( older == 0 )) && return 0
    sleep 30
  done
  echo "an older chain is still running after ${WAIT_MAX_MIN:-420} min"; return 1
}
stage_final() {
  stop_servers
  nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  printf '{"final_utc": "%s", "instance_id": "%s", "instance_type": "%s", "hours_used": %s}\n' "$(date -u +%FT%TZ)" "$MY_IID" \
    "$(imds instance-type)" "$(hours_used)" > "$A/host_final.json"
  st "final: used $(hours_used) h; $(cd "$R" && PY "$EX" spend --runs runs --log "$PROBE_LOG" --short 2>/dev/null)"
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; sudo shutdown -h +2 "jev-e28-sl08b: done"; fi
}

st "chain start: $STAGES (host $MY_IID, used $(hours_used) h)"
for s in $STAGES; do
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $s"; continue; fi
  t0=$(date +%s)
  cd "$R" 2>/dev/null || cd "$W"
  ( "stage_$s" ) > "$STATUS_DIR/stage_$s.log" 2>&1
  rc=$?
  [[ -d "$A/logs" ]] && cp -f "$STATUS_DIR/stage_$s.log" "$A/logs/stage_$s.log"
  echo "{\"stage\": \"$s\", \"exit\": $rc, \"minutes\": $(( ($(date +%s) - t0) / 60 )), \"end_utc\": \"$(date -u +%FT%TZ)\"}" >> "$STATUS_DIR/stage_times.jsonl"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/stage_times.jsonl" "$A/stage_times.jsonl"
  st "stage $s exit=$rc ($(( ($(date +%s) - t0) / 60 )) min): $(tail -2 "$STATUS_DIR/stage_$s.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  if (( rc != 0 )) && [[ " rearm code setup data train select temps wait_prev " == *" $s "* ]]; then
    st "chain stopped at $s"
    stop_servers
    [[ " $STAGES " == *" final "* ]] && stage_final
    exit 1
  fi
done
sync_out
st "chain end [$STAGES]"
