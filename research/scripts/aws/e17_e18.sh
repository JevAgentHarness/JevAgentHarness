#!/usr/bin/env bash
# Exploratory rows E17 and E18 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E17 and E18: read-out ablation and a
# deferral-aware objective"; descriptive, outside every Holm table). One NEW GPU host launched by this driver from E3's
# clone image (tag Name=jev-e17-e18), SSM only. E17: E14 arm A's scoring head (its checkpoint from the bucket, not
# retrained) against the SFT v1 one-token generative read-out on the same backbone (Qwen3.5-0.8B) and the same rows;
# E18: arm A served with the logistic defer head (results/explore/e17_e18/e18_head.json, fitted locally from logged
# records) in the final harness on the 200 dedicated-row tasks; E18-F (optional, offline): the option trained into a
# new checkpoint. The live stage holds the Gemini key in its process only (from SSM; never printed, never on disk).
#
#   LOCAL:  bash scripts/aws/e17_e18.sh launch [itype ...]   # one new host (refuses if a jev-e17-e18 host exists)
#           E17_IID=i-... bash scripts/aws/e17_e18.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages (in the order sent; a failed stage stops the chain, `final` always runs when sent):
#     rearm      idle stop (IDLE_MIN) + power-off guard (GUARD_MIN from the first boot), host record
#     setup      the committed branch in /opt/jev/e17; the hvenv imports it; Unsloth's venv; the v2 venv; frozen 4B
#     code       refresh the checkout from the bucket's bundle (fixes between stages)
#     data       the records (built locally, sha256-checked), arm A's checkpoint, the calibration requests, the backbone
#     live_e18   arm A + the defer head on :8083 (serve_jev --backend clef --clef-defer-head): smoke, gate, 200 tasks
#     smoke_gen  the one-token read-out check and the 4-step memory smoke -> the memory setting and the epoch projection
#     train_gen  the SFT v1 recipe (10_sft.sh, E5's env) on the same rows; adapter -> s3://.../checkpoints/e17_gen/
#     eval_gen   eval_decisions (raw + temperatures), calibrate, served on :8082 (--fast): corpus C, calibration states
#     latency    harness_final.py bench c = 1 / 4: the frozen 4B (:8081), arm A (:8083, clef), the generative arm (:8082)
#     e18f_train | e18f_eval   optional: the option trained into a new checkpoint (Unsloth), dev read-out, corpus C
#     final      stop the servers, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e17_e18/box on the box -> s3://$BUCKET/e17_e18/box; runs -> s3://$BUCKET/e17_e18/runs.
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E17_IID=${E17_IID:-}
# every other project host: never touched by this driver
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e17-e18
BUNDLE=e17.bundle
S3OUT="$B/e17_e18"
CAP_USD=${CAP_USD:-8}
CAP_H=${CAP_H:-8}
GUARD_MIN=${GUARD_MIN:-470}          # 8 host-hours cap (registration (d)): the guard 7 h 50 min after the first boot
IDLE_MIN=${IDLE_MIN:-30}
POWEROFF=${POWEROFF:-1}
PORT=8083                            # arm A (clef; with the defer head in live_e18; E18-F in e18f_eval)
PORTG=8082                           # the generative arm (serve_jev --fast)
PORT4B=8081                          # the frozen 4B (latency anchor)
GEN_CAP_MIN=${GEN_CAP_MIN:-280}      # Deviations "E17: the generative arm's epoch and micro-batch": ONE epoch within 280 minutes
GEN_MAX_EPOCHS=${GEN_MAX_EPOCHS:-1}
GEN_SETTINGS=${GEN_SETTINGS:-"nogc_6144 nogc_3072 gc_3072 gc_2048 gc_1536 gc_1024"}   # the first, then E3's ladder on OOM
BASE_REPO=Qwen/Qwen3.5-0.8B
BASE_REV=2fc06364715b967f1860aea9cf38778875588b17
MIX_SHA=9fe1088d8e5be533f38a2115fe369b9b238018367bbb6b4e729c19f9f94ebe37
CALREQ_SHA=d46701e8e9bdc69e6a9e047227eabb1e52578ddc527342dda10b33f7dc5fff5e

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E17_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E17_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E17_IID "* ]] && { echo "refused: $E17_IID is not the E17/E18 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E17_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E17_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e17-e18: power-off guard (8 host-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g5.2xlarge g5.4xlarge g5.8xlarge)
  for IT in "${types[@]}"; do
    for SUBNET in $SUBNETS; do
      out=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$IT" --subnet-id "$SUBNET" --security-group-ids "$SG" \
        --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior stop --count 1 \
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/e17_e18.sh" > "$tmp/e17_e18.sh"   # the committed driver
  awsl s3 cp "$tmp/e17_e18.sh" "$B/code/e17_e18.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e17_e18.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e17_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e17_e18.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q CAP_H=%q GUARD_MIN=%q IDLE_MIN=%q POWEROFF=%q E17_IID=%q GEN_CAP_MIN=%q GEN_MAX_EPOCHS=%q GEN_SETTINGS=%q KEEP_SERVER=%q WAIT_MAX_MIN=%q PROBE_WAIT_MIN=%q GEN_EPOCHS=%q\n' \
      "$CAP_USD" "$CAP_H" "$GUARD_MIN" "$IDLE_MIN" "$POWEROFF" "$E17_IID" "$GEN_CAP_MIN" "$GEN_MAX_EPOCHS" "$GEN_SETTINGS" "${KEEP_SERVER:-0}" "${WAIT_MAX_MIN:-300}" "${PROBE_WAIT_MIN:-45}" "${GEN_EPOCHS:-}"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e17_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e17_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E17_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e17_driver.sh /tmp/e17_driver.sh --only-show-errors && bash /tmp/e17_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E17_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E17_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,27p' "$0"
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
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E17_IID" && "$MY_IID" != "$E17_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E17/E18 host ${E17_IID:-?}"; exit 1
fi
R=$W/e17
A=$R/results/explore/e17_e18/box
STATUS_DIR=$W/e17_status
HV=$W/hvenv
EV=$W/e14venv                       # Unsloth 2026.10.2 (the clef backend, E18-F)
VV=$W/v2venv                        # A7.19 (f): transformers 5.18 + fla / causal-conv1d (+ peft): the generative arm
D=$W/e17_data
CKA=$W/ck_e14/A/final               # E14 arm A's merged checkpoint (from the bucket; never retrained)
CKG=$W/ck_e17_gen                   # the generative arm's adapter
CKF=$W/ck_e18f                      # E18-F
CK=$W/ck                            # the frozen 4B
PLAN=results/stage2/rl_eval_plan.json
STOP=runs/e18_STOP.json
EX=scripts/explore_e17_e18.py
HEAD=results/explore/e17_e18/e18_head.json
PROFILE=configs/harness/explore_e18_final.yaml
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$STATUS_DIR"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  aws s3 cp "$STATUS_DIR/status.txt" "$S3OUT/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION" --exclude 's23/corpus.json' \
    --exclude 's23/occurrences.jsonl.gz' --exclude 's23/questions.jsonl.gz' --exclude 'inputs/*'
  local d
  for d in "$R"/runs/e18_*; do
    [[ -d "$d" ]] && aws s3 sync "$d" "$S3OUT/runs/$(basename "$d")" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  true
}
PY() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }
EPY() { PYTHONPATH="$R/src" "$EV/bin/python" "$@"; }
VPY() { PYTHONPATH="$R/src" "$VV/bin/python" "$@"; }
plan() { (cd "$R" && PY -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"); }
base4b() { cat "$W/base_model_path.txt"; }
frozen_dir() { echo "$CK/$(plan "d['rows']['frozen']['dir']")"; }
frozen_cal() { echo "$CK/$(plan "d['rows']['frozen']['calibration']")"; }
snap() { cat "$D/base_snapshot.txt" 2>/dev/null; }
hours_used() {  # host hours since the first boot recorded by rearm
  local b; b=$(cat "$W/e17_first_boot" 2>/dev/null || date -d "$(uptime -s)" +%s)
  python3 -c "import time; print(round((time.time() - $b) / 3600, 3))"
}
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
stop_port() {
  local f="$W/e17_serve_$1.pid" p
  if [[ -f "$f" ]]; then
    p=$(cat "$f"); kill "$p" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -9 "$p" 2>/dev/null
  fi
  rm -f "$f"
}
wait_up() {  # wait_up PORT NAME TAG
  local port=$1 name=$2 tag=$3
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$port/health" > "$A/logs/serve_${tag}_health.json"; then
      curl -sf "http://127.0.0.1:$port/v1/models" > "$A/logs/serve_${tag}_models.json"
      grep -q "\"$name\"" "$A/logs/serve_${tag}_models.json" || { echo "served model id is not $name"; return 1; }
      printf '{"server_up": "%s", "name": "%s", "port": %s, "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" "$name" "$port" \
        "$(imds instance-type)" >> "$A/serve_windows.jsonl"
      return 0
    fi
    kill -0 "$(cat "$W/e17_serve_$port.pid")" 2>/dev/null || { echo "server died"; tail -30 "$A/logs/serve_$tag.out"; return 1; }
    sleep 5
  done
  echo "server did not come up"; return 1
}
serve_a() {  # serve_a TAG [serve_jev flags...]: E14 arm A's checkpoint behind serve_jev --backend clef on :8083
  local tag=$1; shift
  stop_port "$PORT"
  [[ -s "$CKA/joint_head.safetensors" ]] || { echo "no arm A checkpoint at $CKA"; return 1; }
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then echo "port $PORT is served by another process"; return 1; fi
  echo "$(date -u +%FT%TZ) serve_jev --backend clef --model $CKA --name $tag --port $PORT $*" >> "$A/logs/serve_$tag.cmd"
  nohup env PYTHONPATH="$R/src" "$EV/bin/python" -m jevharness.training.serve_jev --backend clef --model "$CKA" \
    --name "$tag" --port "$PORT" "$@" > "$A/logs/serve_$tag.out" 2>&1 &
  echo $! > "$W/e17_serve_$PORT.pid"
  wait_up "$PORT" "$tag" "$tag"
}
serve_f() {  # E18-F's merged checkpoint with the option on :8083
  stop_port "$PORT"
  [[ -s "$CKF/F/final/joint_head.safetensors" ]] || { echo "no E18-F checkpoint"; return 1; }
  echo "$(date -u +%FT%TZ) serve_jev --backend clef --model $CKF/F/final --name e18f --port $PORT --clef-defer" >> "$A/logs/serve_e18f.cmd"
  nohup env PYTHONPATH="$R/src" "$EV/bin/python" -m jevharness.training.serve_jev --backend clef --model "$CKF/F/final" \
    --name e18f --port "$PORT" --clef-defer > "$A/logs/serve_e18f.out" 2>&1 &
  echo $! > "$W/e17_serve_$PORT.pid"
  wait_up "$PORT" e18f e18f
}
serve_gen() {  # the generative arm (adapter + temperatures) behind serve_jev --fast on :8082 (the v2 venv)
  stop_port "$PORTG"
  [[ -s "$CKG/adapter_model.safetensors" && -s "$CKG/calibration.json" ]] || { echo "no generative adapter / temperatures"; return 1; }
  local args=(--model "$(snap)" --adapter "$CKG" --calibration "$CKG/calibration.json" --name e17-gen --port "$PORTG" --fast)
  echo "serve_jev ${args[*]} (venv $VV)" > "$A/logs/serve_gen.cmd"
  nohup env PYTHONPATH="$R/src" "$VV/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$A/logs/serve_e17-gen.out" 2>&1 &
  echo $! > "$W/e17_serve_$PORTG.pid"
  wait_up "$PORTG" e17-gen e17-gen
}
serve_4b() {
  stop_port "$PORT4B"
  verify_frozen || return 1
  local args=(--model "$(base4b)" --adapter "$(frozen_dir)" --calibration "$(frozen_cal)" --name jevagentrl-4b
              --port "$PORT4B" --fast)
  echo "serve_jev ${args[*]}" > "$A/logs/serve_4b.cmd"
  nohup env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$A/logs/serve_4b.out" 2>&1 &
  echo $! > "$W/e17_serve_$PORT4B.pid"
  wait_up "$PORT4B" jevagentrl-4b 4b
}
spec_of() {
  printf '{"type": "jev", "name": "%s", "base_url": "http://127.0.0.1:%s", "model": "%s", "api_key": "", "timeout": 120.0}' \
    "$1" "$2" "$1"
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
smi_start() { nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 > "$1" 2>/dev/null & SMI=$!; }
smi_stop() { [[ -n "${SMI:-}" ]] && kill "$SMI" 2>/dev/null; SMI=""; }
sha_ok() { [[ "$(sha256sum "$1" | cut -d' ' -f1)" == "$2" ]]; }

stage_rearm() {
  [[ -s "$W/e17_first_boot" ]] || date -d "$(uptime -s)" +%s > "$W/e17_first_boot"
  local used_min left
  used_min=$(( ( $(date +%s) - $(cat "$W/e17_first_boot") ) / 60 ))
  left=$(( GUARD_MIN - used_min ))
  (( left > 0 )) || { echo "guard already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$left" "jev-e17-e18: power-off guard (GPU cap)"
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E17/E18 chain / server / runner / trainer process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e17_[0-9]|jevbench|e14_clef|e18_defer|explore_e17|stage2_corpus|decision_eval|e5_cal_replay|harness_final|train_decision|eval_decisions|calibrate|uv pip|pip install|snapshot_download' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e17-e18: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $left min left, idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU; $(df -BG / | awk 'NR==2{print $4}') free; used $(hours_used) h"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}
fetch_code() {
  local myb="$W/e17.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A/logs" "$A/eval" "$A/cal" "$A/s23" "$A/latency/bench" runs
  st "code at $(git log --oneline -1 | cut -c1-72)"
}
stage_code() { fetch_code; }
stage_kill_prev() {  # stop every older chain of this driver and the processes its stage left (the restart of the amendment)
  local me t
  me=$(basename "$0" .sh); me=${me#e17_}
  for t in $(ps -eo pid,args | grep -E 'e17_[0-9]+\.sh box' | grep -v grep | awk '{print $1}'); do
    [[ "$t" == "$$" || "$t" == "$BASHPID" ]] && continue   # never this chain nor this stage's own forked subshell
    kill "$t" 2>/dev/null
  done
  pkill -f 'harness_final.py bench' 2>/dev/null; pkill -f 'e18_defer.py train' 2>/dev/null
  pkill -f 'stage2_corpus.py replay' 2>/dev/null; pkill -f 'e5_cal_replay.py answer' 2>/dev/null
  stop_port "$PORT"; stop_port "$PORTG"; stop_port "$PORT4B"
  pkill -f 'jevharness.training.serve_jev' 2>/dev/null
  sleep 5; sudo shutdown -c 2>/dev/null; stage_rearm
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
}
stage_setup() {
  fetch_code || return 1
  cd "$R" || return 1
  grep -q 'E17 and E18: read-out ablation and a deferral-aware objective' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E17/E18"; return 1; }
  PY -c "import jevharness, torch, sys; p = jevharness.__file__; print(p, torch.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_hvenv.txt" 2>&1 || { cat "$A/env_hvenv.txt"; echo "the hvenv does not import this checkout"; return 1; }
  command -v uv > /dev/null || bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  if [[ ! -x "$EV/bin/python" ]] || ! "$EV/bin/python" -c "import unsloth" > /dev/null 2>&1; then
    uv venv -q -p 3.11 "$EV" || return 1
    timeout 45m uv pip install -q --python "$EV" "unsloth==2026.10.2" fastapi uvicorn httpx pydantic pyyaml tqdm \
      python-dotenv openai > "$A/logs/e14venv_install.log" 2>&1 || { tail -20 "$A/logs/e14venv_install.log"; return 1; }
    timeout 20m uv pip install -q --python "$EV" flash-linear-attention >> "$A/logs/e14venv_install.log" 2>&1 \
      || echo "flash-linear-attention not installed (the torch path is used)" | tee -a "$A/logs/e14venv_install.log"
  fi
  EPY -c "import unsloth, torch, transformers, peft, trl; from unsloth import FastDecisionModel, DecisionTrainer; print('unsloth', unsloth.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'trl', trl.__version__, 'cuda', torch.cuda.is_available())" \
    > "$A/env_e14venv.txt" 2>&1 || { tail -20 "$A/env_e14venv.txt"; return 1; }
  "$EV/bin/pip" freeze > "$A/e14venv_freeze.txt" 2>/dev/null || uv pip freeze --python "$EV" > "$A/e14venv_freeze.txt"
  if [[ -x "$VV/bin/python" ]]; then
    # the image's v2 venv has no peft (E12c installed it on its own host): the generative arm's LoRA needs it
    "$VV/bin/python" -c "import peft" > /dev/null 2>&1 || timeout 15m uv pip install -q --python "$VV" "peft==0.21.2" \
      > "$A/logs/v2venv_peft.log" 2>&1 || echo "peft install failed (logs/v2venv_peft.log)"
    VPY -c "import jevharness, torch, transformers, peft; print(jevharness.__file__, torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available())" \
      > "$A/env_v2venv.txt" 2>&1 || { cat "$A/env_v2venv.txt"; echo "the v2 venv does not import this checkout"; }
    VPY -c "import fla; print('fla', fla.__version__)" >> "$A/env_v2venv.txt" 2>&1 || true
  else
    echo "no v2 venv on this image: the generative arm cannot run" | tee "$A/env_v2venv.txt"
  fi
  cat "$A/env_e14venv.txt" "$A/env_v2venv.txt"
  [[ -s "$W/base_model_path.txt" ]] || { echo "no base_model_path.txt"; return 1; }
  if ! verify_frozen; then
    st "the frozen 4B is absent or different in $CK; fetching"
    aws s3 sync "$(plan "d['rows']['frozen']['s3']")" "$(frozen_dir)" --only-show-errors --region "$REGION" || return 1
    verify_frozen || return 1
  fi
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s", "gpu": "%s"}\n' "$MY_IID" \
    "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" \
    "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" > "$A/host.json"
  df -BG / | awk 'NR==2{print "disk free: "$4}'
}
stage_data() {
  cd "$R" || return 1
  mkdir -p "$D" "$A/inputs" "$CKA"
  # 1. the records built locally (E18-F: train_f / dev_f / cal_f; arm A's dev records), sha256-checked against the commit
  aws s3 sync "$S3OUT/data" "$D" --only-show-errors --region "$REGION" || return 1
  PY - "$D" results/explore/e17_e18/data_sha256.json <<'EOF' || return 1
import hashlib, json, os, sys
d, ref = sys.argv[1], json.load(open(sys.argv[2]))
bad = []
for rel, sha in ref["sha256"].items():
    p = os.path.join(d, rel)
    got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    bad += [rel] if got != sha else []
print("records sha256:", "all match" if not bad else f"MISMATCH {bad}")
sys.exit(1 if bad else 0)
EOF
  # 2. E14 arm A's merged checkpoint (never retrained); the head and decision config hashes recorded
  aws s3 sync "$B/checkpoints/e14_small/A/final" "$CKA" --only-show-errors --region "$REGION" || return 1
  (cd "$CKA" && sha256sum joint_head.safetensors unsloth_decision_config.json joint_head_config.json) > "$A/arm_a_sha256.txt" || return 1
  cat "$A/arm_a_sha256.txt"
  # 3. the calibration requests (E5's build)
  aws s3 cp "$B/e5_small/inputs/cal_requests.jsonl.gz" "$A/inputs/cal_requests.jsonl.gz" --only-show-errors --region "$REGION" || return 1
  sha_ok "$A/inputs/cal_requests.jsonl.gz" "$CALREQ_SHA" || { echo "cal_requests sha256 mismatch"; return 1; }
  # 4. the backbone at the pinned revision (both arms; the generative arm loads the snapshot path)
  PY - "$BASE_REPO" "$BASE_REV" "$D/base_snapshot.txt" <<'EOF' || return 1
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
open(sys.argv[3], "w").write(p)
print(p)
EOF
  # 5. the corpus C copy
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz; do cp -f "results/stage2/s23/$f" "$A/s23/$f" || return 1; done
  sync_out
}
run_live() {  # run_live CONFIG RUN_DIR: probe-gated, paused on 503 / overload, capped spend; then one re-run of errors
  local cfg=$1 run=$2 rc pid wpid round
  for round in 1 2 3 4 5 6 7 8; do
    [[ -f "$STOP" ]] && { st "spend cap reached earlier; $cfg not run further"; return 3; }
    probe_ok || { st "probes not clean; $cfg paused (round $round)"; return 4; }
    st "$cfg: start (round $round)"
    setsid env PYTHONPATH="$R/src" JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$cfg" --out runs >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PY "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --every 30 >> "$A/watch.log" 2>&1 &
    wpid=$!
    ( while kill -0 "$pid" 2>/dev/null; do sleep 300; sync_out; done ) & local spid=$!
    wait "$pid"; rc=$?
    kill "$wpid" "$spid" 2>/dev/null
    sync_out
    if [[ -f "$run/PAUSED_overload.json" ]]; then
      st "$cfg paused on an overload error (round $round); waiting for clean probes"
      rm -f "$run/PAUSED_overload.json"
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
    setsid env PYTHONPATH="$R/src" JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$cfg" --out runs >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PY "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --every 30 --no-pause >> "$A/watch.log" 2>&1 &
    wpid=$!
    wait "$pid"; kill "$wpid" 2>/dev/null
  fi
  st "$cfg done; errored now $(PY "$EX" errored --run "$run"); $(PY "$EX" spend --runs runs --short 2>/dev/null)"
  sync_out
  return 0
}
probe_ok() {  # two consecutive clean probe rounds (E14's pause rule); waits up to PROBE_WAIT_MIN
  local end=$(( $(date +%s) + ${PROBE_WAIT_MIN:-45} * 60 )) clean=0
  while (( $(date +%s) < end )); do
    if PY "$EX" probe --log "$A/api_probe_log.txt" > /dev/null 2>&1; then clean=$((clean + 1)); else clean=0; fi
    (( clean >= 2 )) && return 0
    sleep $(( clean > 0 ? 30 : 300 ))
  done
  return 1
}
gate() {  # gate SMOKE_RUN N_EPISODES: the smoke's mean $ per episode x N x 1.5 + spend so far <= CAP_USD
  PY "$EX" gate --runs runs --smoke "$1" --n "$2" --cap "$CAP_USD" > "$A/logs/gate_$(basename "$1").json" 2>&1
}
stage_live_e18() {  # the final harness with arm A + the defer head (registration (c) 6)
  cd "$R" || return 1
  [[ -s "$HEAD" ]] || { echo "no defer head $HEAD"; return 1; }
  grep -q 'status: development' "$PROFILE" && ! grep -q 'pending' "$PROFILE" || { echo "the E18 profile is not ready"; return 1; }
  PY "$EX" check-profile || return 1
  keys || return 1
  serve_a e18 --clef-defer-head "$HEAD" || return 1
  run_live configs/explore/e17_e18/live_smoke_e18.yaml runs/e18_live_smoke || return 1
  gate runs/e18_live_smoke 200 || { st "gate refuses live_e18: $(cat "$A/logs/gate_e18_live_smoke.json")"; return 3; }
  printf '{"window_start": "%s", "instance_type": "%s", "instance_id": "%s", "workers": 1, "run": "runs/e18_live"}\n' \
    "$(date -u +%FT%TZ)" "$(imds instance-type)" "$MY_IID" > "$A/live_window.json"
  run_live configs/explore/e17_e18/live_e18.yaml runs/e18_live
  local rc=$?
  printf '{"window_end": "%s"}\n' "$(date -u +%FT%TZ)" >> "$A/live_window.json"
  curl -sf "http://127.0.0.1:$PORT/health" > "$A/logs/serve_e18_health_end.json"
  stop_port "$PORT"
  leak_check || return 1
  return $rc
}
# ---------------------------------------------------------------- E17 (ii): the generative one-token read-out
flags_of() {  # flags_of SETTING: the trainer's memory flags (E3's ladder, plus nogc_6144 = the amended first setting)
  if [[ "$1" == nogc_6144 ]]; then echo "--max-batch-tokens 6144 --eff-batch 64 --no-gradient-checkpointing"; else PY scripts/train/ablation_box.py setting "$1"; fi
}
gen_mix() {  # SFT v1's exact training input (10_sft.sh's data mix), sha256-checked
  mkdir -p "$D/gen"
  [[ -s "$D/gen/train_mix.jsonl" ]] || PY -m jevharness.training.data mix --in data/train_v1/train.jsonl.gz:1 \
    data/train_v1/replay_when2call.jsonl.gz:0.45 --out "$D/gen/train_mix.jsonl" --seed 0 > "$A/logs/gen_mix.log" 2>&1 || return 1
  sha_ok "$D/gen/train_mix.jsonl" "$MIX_SHA" || { echo "train_mix sha256 mismatch"; return 1; }
}
sft_run() {  # sft_run OUT SETTING LOG [MAX_STEPS] [EPOCHS]: E5's sft_run (10_sft.sh, the SFT v1 recipe), BASE = the snapshot
  local out=$1 setting=$2 log=$3 steps=${4:--1} epochs=${5:-2} flags
  flags=$(flags_of "$setting") || return 1
  rm -rf "$out"; mkdir -p "$out"
  env GPUS=1 RECIPE=r3 MODEL="$(snap)" TRAIN="$D/gen/train_mix.jsonl" DEV=data/train_v0/dev.jsonl REPLAY= OUT="$out" \
      LR=1e-4 EPOCHS="$epochs" BATCH=2 EFF_BATCH=64 MAX_LEN=4096 SEED=0 MAX_STEPS="$steps" PY="$VV/bin/python" \
      PYTHONPATH="$R/src" SFT_EXTRA="$flags" DRY_RUN=0 bash scripts/train/10_sft.sh > "$log" 2>&1
}
stage_smoke_gen() {
  cd "$R" || return 1
  [[ -x "$VV/bin/python" ]] || { st "no v2 venv: the generative arm is not run"; echo '{"not_run": "no v2 venv"}' > "$A/gen_projection.json"; return 0; }
  gen_mix || return 1
  VPY scripts/train/e5_box.py readout --model "$(snap)" --out "$A/gen_readout.json" > "$A/logs/gen_readout.log" 2>&1 || true
  cat "$A/gen_readout.json"
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$A/gen_readout.json')).get('ok') else 1)" \
    || { st "the one-token read-out check failed: the generative arm is not run"; echo '{"not_run": "readout check"}' > "$A/gen_projection.json"; return 0; }
  PY "$EX" smoke-file --mix "$D/gen/train_mix.jsonl" --out "$D/gen/smoke.jsonl" || return 1
  local setting ok=0 log
  for setting in $GEN_SETTINGS; do
    log="$A/logs/gen_smoke_$setting.log"
    st "generative smoke at $setting"
    sft_smoke "$setting" "$log" && { ok=1; break; }
    if PY scripts/train/e5_box.py oom "$log"; then st "out of memory at $setting; next setting"; continue; fi
    tail -20 "$log"; break
  done
  (( ok )) || { st "the generative smoke failed: the arm is not run"; echo '{"not_run": "smoke failed"}' > "$A/gen_projection.json"; return 0; }
  echo "$setting" > "$D/gen/mem_setting.txt"
  PY "$EX" projection --log "$log" --rows 16902 --cap-min "$GEN_CAP_MIN" --max-epochs "$GEN_MAX_EPOCHS" --setting "$setting" --out "$A/gen_projection.json" || return 1
  cat "$A/gen_projection.json"
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$A/gen_projection.json')).get('epochs') else 1)" \
    || st "the projection allows no epoch: the generative arm is not run"
  return 0
}
sft_smoke() {  # sft_smoke SETTING LOG: 4 optimizer steps on the smoke rows (the 48 longest prompts + 48 random rows)
  local setting=$1 log=$2 flags
  flags=$(flags_of "$setting") || return 1
  rm -rf "$CKG.smoke"; mkdir -p "$CKG.smoke"
  env GPUS=1 RECIPE=r3 MODEL="$(snap)" TRAIN="$D/gen/smoke.jsonl" DEV= REPLAY= OUT="$CKG.smoke" LR=1e-4 EPOCHS=2 BATCH=2 \
      EFF_BATCH=64 MAX_LEN=4096 SEED=0 MAX_STEPS=4 PY="$VV/bin/python" PYTHONPATH="$R/src" SFT_EXTRA="$flags" DRY_RUN=0 \
      bash scripts/train/10_sft.sh > "$log" 2>&1
}
stage_train_gen() {
  cd "$R" || return 1
  local setting epochs t0 rc
  python3 -c "import json,sys; d=json.load(open('$A/gen_projection.json')); sys.exit(0 if d.get('epochs') else 1)" 2>/dev/null \
    || { st "train_gen skipped: the generative arm is not run ($(cat "$A/gen_projection.json" 2>/dev/null | cut -c1-120))"; return 0; }
  setting=$(cat "$D/gen/mem_setting.txt" 2>/dev/null) || { echo "no memory setting (smoke_gen first)"; return 1; }
  epochs=${GEN_EPOCHS:-$(python3 -c "import json; print(json.load(open('$A/gen_projection.json'))['epochs'])")}
  st "generative arm: $epochs epoch(s) at $setting (used $(hours_used) h)"
  t0=$(date +%s)
  smi_start "$A/logs/train_smi_gen.csv"
  sft_run "$CKG" "$setting" "$A/logs/train_gen.log" -1 "$epochs"
  rc=$?
  smi_stop
  printf '{"arm": "gen", "exit": %s, "minutes": %s, "epochs": %s, "setting": "%s", "end_utc": "%s"}\n' "$rc" \
    "$(( ($(date +%s) - t0) / 60 ))" "$epochs" "$setting" "$(date -u +%FT%TZ)" >> "$A/train_runs.jsonl"
  (( rc == 0 )) || { tail -30 "$A/logs/train_gen.log"; return 1; }
  sha256sum "$D/gen/train_mix.jsonl" > "$A/gen_train_mix.sha256"
  aws s3 sync "$CKG" "$B/checkpoints/e17_gen" --only-show-errors --region "$REGION" --exclude 'checkpoint-*/*' --exclude '*optimizer*'
  sync_out
}
stage_eval_gen() {
  cd "$R" || return 1
  local V1=(data/train_v1/dev.jsonl.gz data/train_v1/dev_recovery.jsonl.gz data/train_v1/dev_knockout.jsonl.gz) bs
  [[ -s "$CKG/adapter_model.safetensors" ]] || { st "eval_gen skipped: no generative adapter"; return 0; }
  VPY -m jevharness.training.eval_decisions --model "$(snap)" --adapter "$CKG" --dev "${V1[@]}" --n-perm 0 --batch-size 16 \
    --out "$A/eval/gen_v1_raw.json" > "$A/logs/eval_gen_raw.log" 2>&1 || { tail -5 "$A/logs/eval_gen_raw.log"; return 1; }
  for bs in 16 8 4; do
    VPY -m jevharness.training.calibrate --model "$(snap)" --adapter "$CKG" --dev data/train_v1/dev.jsonl.gz \
      --out "$CKG/calibration.json" --min-bucket 200 --batch-size "$bs" > "$A/logs/calibrate_gen.log" 2>&1 && break
    st "calibrate gen: failed at batch $bs"
  done
  [[ -s "$CKG/calibration.json" ]] || return 1
  cp -f "$CKG/calibration.json" "$A/gen_calibration.json"
  VPY -m jevharness.training.eval_decisions --model "$(snap)" --adapter "$CKG" --calibration "$CKG/calibration.json" \
    --dev "${V1[@]}" --n-perm 0 --batch-size 16 --out "$A/eval/gen_v1.json" > "$A/logs/eval_gen.log" 2>&1 || { tail -5 "$A/logs/eval_gen.log"; return 1; }
  st "gen v1: $(python3 -c "import json; d=json.load(open('$A/eval/gen_v1.json')); print({k[-20:]: v['overall']['acc'] for k, v in d['files'].items()})")"
  aws s3 cp "$CKG/calibration.json" "$B/checkpoints/e17_gen/calibration.json" --only-show-errors --region "$REGION"
  serve_gen || return 1
  local k
  for k in 1 2 3; do   # the replay resumes
    PY scripts/stage2_corpus.py replay --corpus "$A/s23" --name e17gen --s1 "$(spec_of e17-gen "$PORTG")" \
      --cache "cache/s1_e17_gen_c.sqlite" > "$A/logs/replay_c_gen_$k.log" 2>&1
  done
  st "corpus C gen: $(wc -l < "$A/s23/answers_e17gen.jsonl" 2>/dev/null) answers"
  for k in 1 2; do
    PY scripts/e5_cal_replay.py answer --requests "$A/inputs/cal_requests.jsonl.gz" --url "http://127.0.0.1:$PORTG" \
      --name e17-gen --out "$A/cal/answers_e17gen.jsonl" > "$A/logs/cal_replay_gen_$k.log" 2>&1 && break
  done
  st "calibration-state replay gen: $(wc -l < "$A/cal/answers_e17gen.jsonl" 2>/dev/null) answers"
  curl -sf "http://127.0.0.1:$PORTG/health" > "$A/logs/serve_gen_health_end.json"
  stop_port "$PORTG"
  sync_out
}
bench_one() {  # bench_one LABEL PORT: harness_final.py bench c = 1 (25 episodes) and c = 4 (100)
  local label=$1 port=$2 c lim
  for c in 1 4; do
    lim=(); [[ $c == 1 ]] && lim=(--limit 25)
    smi_start "$A/latency/smi_${label}_c$c.csv"
    PY scripts/harness_final.py bench --url "http://127.0.0.1:$port" --label "$label" --path new --concurrency "$c" \
      --out "$A/latency/bench" "${lim[@]}" >> "$A/latency/bench.log" 2>&1
    local rc=$?
    smi_stop
    (( rc == 0 )) || return 1
    st "latency $label bench c=$c done"
  done
  curl -sf "http://127.0.0.1:$port/v1/models" > "$A/latency/models_$label.json"
}
stage_latency() {  # one session, one model on the GPU at a time: the frozen 4B, arm A (head), the generative arm
  cd "$R" || return 1
  stop_port "$PORT"; stop_port "$PORTG"; stop_port "$PORT4B"
  serve_4b || return 1
  bench_one 4b "$PORT4B" || return 1
  stop_port "$PORT4B"
  serve_a e14-a || return 1
  bench_one head "$PORT" || return 1
  stop_port "$PORT"
  if [[ -s "$CKG/adapter_model.safetensors" && -s "$CKG/calibration.json" ]]; then
    serve_gen || return 1
    bench_one gen "$PORTG" || return 1
    stop_port "$PORTG"
  else
    st "latency: no generative arm to time"
  fi
  sync_out
}
# ---------------------------------------------------------------- E18-F (optional, offline): the option trained in
stage_e18f_train() {
  cd "$R" || return 1
  local used need=2.2
  used=$(hours_used)
  python3 -c "import sys; sys.exit(0 if $used + $need <= $CAP_H - 0.25 else 1)" \
    || { st "E18-F not run: $used h used, $need h needed, cap $CAP_H"; echo "{\"not_run\": \"cap\", \"hours_used\": $used}" > "$A/e18f_not_run.json"; return 0; }
  stop_port "$PORT"; stop_port "$PORTG"
  local t0 rc
  t0=$(date +%s)
  smi_start "$A/logs/train_smi_F.csv"
  EPY scripts/train/e18_defer.py train --data "$D/f" --out "$CKF" --hours-used "$used" --cap-h "$CAP_H" \
    --v1-out "$A/eval/e18f_v1.json" > "$A/logs/train_F.log" 2>&1
  rc=$?
  smi_stop
  cp -f "$CKF/F/train_record.json" "$A/train_record_F.json" 2>/dev/null
  printf '{"arm": "F", "exit": %s, "minutes": %s, "end_utc": "%s"}\n' "$rc" "$(( ($(date +%s) - t0) / 60 ))" "$(date -u +%FT%TZ)" >> "$A/train_runs.jsonl"
  (( rc == 0 )) || { tail -30 "$A/logs/train_F.log"; return 1; }
  aws s3 sync "$CKF/F/final" "$B/checkpoints/e18f/final" --only-show-errors --region "$REGION"
  sync_out
}
stage_e18f_eval() {
  cd "$R" || return 1
  [[ -s "$CKF/F/final/joint_head.safetensors" ]] || { st "E18-F eval: no checkpoint"; return 0; }
  EPY scripts/train/e18_defer.py devread --ckpt "$CKF/F/final" --data "$D/f" --out "$A/eval/e18f_devread.json" \
    > "$A/logs/e18f_devread.log" 2>&1 || { tail -5 "$A/logs/e18f_devread.log"; return 1; }
  serve_f || return 1
  local k
  for k in 1 2 3; do
    PY scripts/stage2_corpus.py replay --corpus "$A/s23" --name e18f --s1 "$(spec_of e18f "$PORT")" \
      --cache "cache/s1_e18f_c.sqlite" > "$A/logs/replay_c_e18f_$k.log" 2>&1
  done
  st "corpus C e18f: $(wc -l < "$A/s23/answers_e18f.jsonl" 2>/dev/null) answers"
  stop_port "$PORT"
  sync_out
}
stage_final() {
  stop_port "$PORT"; stop_port "$PORTG"; stop_port "$PORT4B"
  nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  printf '{"final_utc": "%s", "instance_id": "%s", "instance_type": "%s", "hours_used": %s}\n' "$(date -u +%FT%TZ)" "$MY_IID" \
    "$(imds instance-type)" "$(hours_used)" > "$A/host_final.json"
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; sudo shutdown -h +2 "jev-e17-e18: done"; fi
}

broke=0
for s in $STAGES; do
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $s"; continue; fi
  st "stage $s start (host $MY_IID, used $(hours_used) h)"
  t0=$(date +%s)
  ( "stage_$s" ) > "$STATUS_DIR/stage_$s.log" 2>&1
  rc=$?
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/stage_$s.log" "$A/stage_$s.log"
  echo "{\"stage\": \"$s\", \"exit\": $rc, \"minutes\": $(( ($(date +%s) - t0) / 60 )), \"end_utc\": \"$(date -u +%FT%TZ)\"}" >> "$STATUS_DIR/stage_times.jsonl"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/stage_times.jsonl" "$A/stage_times.jsonl"
  st "stage $s exit=$rc ($(( ($(date +%s) - t0) / 60 )) min)"
  if (( rc != 0 )) && [[ "$s" != final ]]; then st "stopping the chain after $s failed"; broke=1; break; fi
done
if (( broke )); then
  sync_out
  st "chain stopped; the box stops itself after $IDLE_MIN idle minutes (a server alone does not keep it up)"
fi
if [[ "${KEEP_SERVER:-0}" != 1 ]]; then stop_port "$PORT"; stop_port "$PORTG"; stop_port "$PORT4B"; fi
st "box done [$STAGES]"
