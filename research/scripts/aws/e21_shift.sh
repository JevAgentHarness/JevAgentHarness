#!/usr/bin/env bash
# Exploratory row E21, DecisionGym-Shift (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E21: a shifted DecisionGym
# split (DecisionGym-Shift)"; descriptive, outside every Holm table). One NEW GPU host launched by this driver from E3's
# clone image (tag Name=jev-e21-shift), SSM only. Each open or trained decision model is served on the host, one at a
# time, as in its logged rows, and answers the L2 corpus (results/explore/e21_shift/c, committed with the code) and its
# two axis re-renderings (registration (d) 3, 5); the four live rows (registration (e)) run right after their model's
# main replay, Flash on the direct Gemini API (the key from SSM into the live process only, never printed or on disk).
#
#   LOCAL:  bash scripts/aws/e21_shift.sh launch [itype ...]   # one new host (refuses if a jev-e21-shift host exists)
#           E21_IID=i-... bash scripts/aws/e21_shift.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages (in the order sent; a failed model stage is logged and the chain goes on; `final` always runs when sent):
#     rearm        idle stop (IDLE_MIN) + power-off guard (GUARD_MIN from the first boot), host record
#     code         the committed branch in /opt/jev/e21 (from the bundle; the corpus arrives with a later push)
#     setup        code + venvs (peft in the v2 venv, Unsloth's venv for the 0.8B head);
#                  docker + the NVIDIA runtime (llama.cpp's image for the 27B); the checkpoints from S3, sha256-checked
#     r_<slug>     as m_<slug> without the live row (a model whose live row ran in an earlier chain; addendum 2: base sft_v1,
#                  main rendering only)
#     l_<slug>     the addendum's live row only (openjevhf sl27b sl4b intern e12c; sl27b time-boxed, BOX_RESERVE_MIN;
#                  addendum 2: base sft_v1)
#     m_<slug>     serve the model, replay the main corpus, the live row (rl e14a e12 sl9b), then the re-renderings
#                  (skipped when the guard is near: registration (d) 5), stop, delete large weights
#                  (slugs: rl e14a e12 sl9b e12c sl4b intern openjevhf sl27b; "E21 addendum 2": base sft_v1, the
#                  untrained Qwen3-4B and Qwen3-4B + SFT v1 served as their Stage-2 rows, results/stage2/rl_eval_plan.json)
#     final        stop the servers, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e21_shift/box -> s3://$BUCKET/e21_shift/box; answers -> e21_shift/c; runs -> e21_shift/runs.
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E21_IID=${E21_IID:-}
# every other project host: never touched by this driver
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e21-shift
BUNDLE=e21.bundle
S3OUT="$B/e21_shift"
CAP_USD=${CAP_USD:-43}               # the host's share of E21's direct-Gemini cap (registration (h): 32; the addendum: 43)
GUARD_MIN=${GUARD_MIN:-710}          # 12 host-hours: the guard 11 h 50 min after the first boot
IDLE_MIN=${IDLE_MIN:-30}
POWEROFF=${POWEROFF:-1}
LLAMA_IMG=ghcr.io/ggml-org/llama.cpp@sha256:d87b02aafa74797a2d599b281c50468704bd0bcdc57bfcf31b570d593f1d285b

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E21_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E21_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E21_IID "* ]] && { echo "refused: $E21_IID is not the E21 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E21_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E21_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e21-shift: power-off guard (12 host-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g6e.2xlarge g6e.4xlarge g5.8xlarge)
  for IT in "${types[@]}"; do
    for SUBNET in $SUBNETS; do
      out=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$IT" --subnet-id "$SUBNET" --security-group-ids "$SG" \
        --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior stop --count 1 \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=400,VolumeType=gp3,Throughput=500,Iops=6000,DeleteOnTermination=true}' \
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/e21_shift.sh" > "$tmp/e21_shift.sh"   # the committed driver
  awsl s3 cp "$tmp/e21_shift.sh" "$B/code/e21_shift.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e21_shift.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e21_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e21_shift.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q GUARD_MIN=%q IDLE_MIN=%q POWEROFF=%q E21_IID=%q USED_MIN=%q\n' \
      "$CAP_USD" "$GUARD_MIN" "$IDLE_MIN" "$POWEROFF" "$E21_IID" "${USED_MIN:-}"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e21_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e21_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E21_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e21_driver.sh /tmp/e21_driver.sh --only-show-errors && bash /tmp/e21_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E21_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E21_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,22p' "$0"
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
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E21_IID" && "$MY_IID" != "$E21_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E21 host ${E21_IID:-?}"; exit 1
fi
R=$W/e21
A=$R/results/explore/e21_shift/box
C=$R/results/explore/e21_shift/c
STATUS_DIR=$W/e21_status
HV=$W/hvenv                 # harness venv (runner, serve_jev for the Qwen3-4B rows, Intern-Decision in-process)
VV=$W/v2venv                # A7.19 (f)'s v2 venv (+ peft): serve_jev on the StartLux-4B base (E12c)
EV=$W/e14venv               # Unsloth 2026.10.2: serve_jev --backend clef (the 0.8B head, as E14 / E18)
LV=$W/e21_vllm_venv         # vLLM 0.29.0 + the OpenJev (HF) helper (as E15)
CK=$W/ck                    # the frozen 4B: $CK/<plan dir>
CKD=$W/ck_e21               # E12, E12c, the 0.8B head
MD=$W/e21_models            # OpenJev-27B FP8 and its helper
SLB=/opt/startlux           # StartLux: the vendor's package, venv and weights (serve_startlux.sh)
PLAN=results/stage2/rl_eval_plan.json
EX=scripts/explore_e21_shift.py
CFG=configs/explore/e21_shift
STOP=runs/e21_STOP.json
LIVE_ROWS=" rl e14a e12 sl9b "
MAIN_ONLY=" base sft_v1 "     # "E21 addendum 2" (c): the main rendering only (no axis re-renderings)
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=$W/e21_hf
mkdir -p "$STATUS_DIR" "$HF_HOME" "$CKD" "$MD"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  aws s3 cp "$STATUS_DIR/status.txt" "$S3OUT/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION"
  [[ -d "$C" ]] && aws s3 sync "$C" "$S3OUT/c" --only-show-errors --region "$REGION" --exclude '*' --include 'answers_*' \
    --include 'replay_*'
  local d
  for d in "$R"/runs/e21_*; do
    [[ -d "$d" ]] && aws s3 sync "$d" "$S3OUT/runs/$(basename "$d")" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  true
}
PY() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }
PV() { PYTHONPATH="$R/src" "$VV/bin/python" "$@"; }
plan() { (cd "$R" && PY -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"); }
qbase() { cat "$W/base_model_path.txt"; }
frozen_dir() { echo "$CK/$(plan "d['rows']['frozen']['dir']")"; }
frozen_cal() { echo "$CK/$(plan "d['rows']['frozen']['calibration']")"; }
hours_used() {
  local b; b=$(cat "$W/e21_first_boot" 2>/dev/null || date -d "$(uptime -s)" +%s)
  python3 -c "import time; print(round((time.time() - $b) / 3600, 3))"
}
mins_left() { local b; b=$(cat "$W/e21_first_boot" 2>/dev/null || date -d "$(uptime -s)" +%s); echo $(( GUARD_MIN - ( $(date +%s) - b ) / 60 )); }
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
leak_check() {
  local n=0 k ks=${GEMINI_API_KEYS:-}
  for k in ${GEMINI_API_KEY:-} ${ks//,/ }; do
    [[ ${#k} -ge 16 ]] || continue
    n=$(( n + $(grep -rlF -- "$k" "$R/runs" "$A" "$C" "$STATUS_DIR" 2>/dev/null | wc -l) ))
  done
  echo "key-leak files: $n"
  [[ "$n" == 0 ]]
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
sha_check() {  # sha_check FILE REFERENCE_SHA256 LABEL
  local got; got=$(sha256sum "$1" 2>/dev/null | cut -d' ' -f1)
  echo "$3 $(basename "$1") $got $([[ "$got" == "$2" ]] && echo ok || echo "MISMATCH (want $2)")" >> "$A/ckpt_sha256.txt"
  [[ "$got" == "$2" ]]
}

# ------------------------------------------------------------------------------------------------ servers
stop_servers() {
  local f p
  for f in "$W"/e21_srv_*.pid; do
    [[ -f "$f" ]] || continue
    p=$(cat "$f"); kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$p" 2>/dev/null; kill -9 "$p" 2>/dev/null
    rm -f "$f"
  done
  (cd "$R" && STOP=1 BASE=$SLB LOG_DIR=$SLB/logs bash scripts/aws/serve_startlux.sh > /dev/null 2>&1)
  sudo docker ps -q 2>/dev/null | xargs -r sudo docker stop > /dev/null 2>&1
  pkill -f 'vllm serve' 2>/dev/null; pkill -f 'helper/shim.py' 2>/dev/null; pkill -f 'startlux_decision' 2>/dev/null
  sleep 5
  true
}
bgsrv() {  # bgsrv TAG CMD...: a server process in its own process group; its pid file
  local tag=$1; shift
  setsid nohup "$@" > "$A/logs/serve_$tag.out" 2>&1 < /dev/null &
  echo $! > "$W/e21_srv_$tag.pid"
  echo "$(date -u +%FT%TZ) $*" >> "$A/logs/serve_$tag.cmd"
}
wait_health() {  # wait_health PORT NAME TAG [MINUTES]: /health, and /v1/models lists NAME
  local port=$1 name=$2 tag=$3 mins=${4:-40} end
  end=$(( $(date +%s) + mins * 60 ))
  while (( $(date +%s) < end )); do
    if curl -sf -m 10 "http://127.0.0.1:$port/health" > "$A/logs/health_$tag.json" 2>/dev/null; then
      curl -sf -m 10 "http://127.0.0.1:$port/v1/models" > "$A/logs/models_$tag.json" 2>/dev/null
      grep -q "\"$name\"" "$A/logs/models_$tag.json" || { echo "served model id is not $name"; return 1; }
      printf '{"server_up": "%s", "slug": "%s", "name": "%s", "port": %s, "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" \
        "$tag" "$name" "$port" "$(imds instance-type)" >> "$A/serve_windows.jsonl"
      return 0
    fi
    [[ -f "$W/e21_srv_$tag.pid" ]] && ! kill -0 "$(cat "$W/e21_srv_$tag.pid")" 2>/dev/null && \
      { echo "server died"; tail -30 "$A/logs/serve_$tag.out"; return 1; }
    sleep 10
  done
  echo "server did not come up in $mins min"; return 1
}
serve_sl() {  # serve_sl SIZE: the vendor's server (serve_startlux.sh) at the pinned revision; 27B = Q8_0 GGUF
  local size=$1
  (cd "$R" && SIZES=$size BASE=$SLB LOG_DIR=$SLB/logs PIN_MANIFEST="$R/$CFG/startlux_pin.json" MANIFEST="$A/manifest_$size.json" \
    PYTHON=python3 LLAMA_SERVER="$SLB/bin/llama-server" bash scripts/aws/serve_startlux.sh > "$A/logs/serve_sl_$size.log" 2>&1)
  python3 -c "import json,sys; m=json.load(open('$A/manifest_$size.json')); s=[x for x in m['sizes'] if x['size']=='$size'][0]; print(s['status'], s.get('weights_revision'), s.get('why')); sys.exit(0 if s['status'] in ('original','gguf_q8_0') else 1)"
}
start_model() {  # start_model SLUG: serve it (the spec in configs/explore/e21_shift/backends.yaml names its port)
  local slug=$1 d
  case "$slug" in
    rl)
      verify_frozen || return 1
      bgsrv rl env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev --model "$(qbase)" --name jevagentrl-4b \
        --port 8081 --adapter "$(frozen_dir)" --calibration "$(frozen_cal)"
      wait_health 8081 jevagentrl-4b rl ;;
    e12)
      d=$CKD/e12
      bgsrv e12 env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev --model "$(qbase)" \
        --name agentdecision-4b-mixed --port 8081 --adapter "$d" --calibration "$d/calibration.json"
      wait_health 8081 agentdecision-4b-mixed e12 ;;
    e12c)
      d=$CKD/e12c
      [[ -s "$SLB/models/StartLux-Decision-4B/config.json" ]] || serve_sl 4B > /dev/null 2>&1   # the base at its pinned revision
      stop_servers
      bgsrv e12c env PYTHONPATH="$R/src" "$VV/bin/python" -m jevharness.training.serve_jev --model "$SLB/models/StartLux-Decision-4B" \
        --name startlux-4b-mixed --port 8081 --adapter "$d" --calibration "$d/calibration.json"
      wait_health 8081 startlux-4b-mixed e12c ;;
    e14a)
      bgsrv e14a env PYTHONPATH="$R/src" "$EV/bin/python" -m jevharness.training.serve_jev --backend clef --model "$CKD/e14a" \
        --name e14-a --port 8083
      wait_health 8083 e14-a e14a ;;
    base)     # addendum 2 (b): the untrained Qwen3-4B as its Stage-2 row (no adapter; the plan's temperature file)
      verify_ladder base || return 1
      bgsrv base env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev --model "$(qbase)" --name rl-base \
        --port 8081 --calibration "$R/$(plan "d['rows']['base']['calibration']")"
      wait_health 8081 rl-base base ;;
    sft_v1)   # addendum 2 (b): Qwen3-4B + the SFT v1 LoRA as its Stage-2 row (from S3, sha256-checked; the plan's temperatures)
      d=$CK/$(plan "d['rows']['sft_v1']['dir']")
      if ! verify_ladder sft_v1; then
        mkdir -p "$d"
        aws s3 sync "$(plan "d['rows']['sft_v1']['s3']")" "$d" --only-show-errors --region "$REGION" || return 1
        verify_ladder sft_v1 || return 1
      fi
      bgsrv sft_v1 env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev --model "$(qbase)" --name rl-sft-v1 \
        --port 8081 --adapter "$d" --calibration "$R/$(plan "d['rows']['sft_v1']['calibration']")"
      wait_health 8081 rl-sft-v1 sft_v1 ;;
    sl4b) serve_sl 4B ;;
    sl9b) wait_predl sl9b; serve_sl 9B ;;
    sl27b) wait_predl sl27b; setup_llama || return 1; serve_sl 27B ;;
    openjevhf) serve_openjevhf ;;
    intern) wait_predl intern
      PY -c "import transformers, torch; print('in process', transformers.__version__, torch.cuda.is_available())" ;;
    *) echo "unknown slug $slug"; return 1 ;;
  esac
}
hfdl() {  # hfdl REPO DIR REV [patterns...]: a pinned snapshot into its own directory; the revision is recorded
  local repo=$1 dir=$2 rev=$3; shift 3
  mkdir -p "$dir"
  PY - "$repo" "$dir" "$rev" "$@" > "$A/logs/hf_$(basename "$dir").log" 2>&1 <<'EOF' || return 1
import sys
from huggingface_hub import HfApi, snapshot_download
repo, d, rev, pats = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
snapshot_download(repo, revision=rev, local_dir=d, allow_patterns=pats or None)
print(HfApi().model_info(repo, revision=rev).sha)
EOF
  printf '{"repo": "%s", "requested": "%s", "sha": "%s", "at": "%s"}\n' "$repo" "$rev" "$(tail -1 "$A/logs/hf_$(basename "$dir").log")" \
    "$(date -u +%FT%TZ)" >> "$A/downloads.jsonl"
}
predl() {  # predl SLUG REPO DIR REV [patterns]: a background download (setup), marked running / done
  local slug=$1; shift
  touch "$W/predl_$slug.running"
  ( hfdl "$@" && touch "$W/predl_$slug.ok"; rm -f "$W/predl_$slug.running" ) &
}
wait_predl() {  # wait_predl SLUG: a background download of setup still running -> wait for it (at most 60 min)
  local n=0
  while [[ -f "$W/predl_$1.running" ]] && (( n < 360 )); do sleep 10; n=$((n + 1)); done
  true
}
serve_openjevhf() {  # E15's OpenJev (HF, FP8) serving, unchanged: vLLM 0.29.0 + the card's helper (registration (d) 3)
  local d="$MD/openjev-fp8"
  wait_predl openjevhf
  hfdl openjev/openjev-FP8 "$d" main || return 1
  hfdl openjev/openjev "$MD/openjev-helper" ac97900fd034fdd7e7e536f3d4c21b836cae0750 helper/shim.py serve/SERVE.md NOTICE LICENSE LICENSE-APACHE-2.0 || return 1
  sha256sum "$MD/openjev-helper/helper/shim.py" > "$A/openjevhf_shim.sha256"
  if [[ ! -x "$LV/bin/vllm" ]]; then
    uv venv -q -p 3.12 "$LV" || return 1
    timeout 40m uv pip install -q --python "$LV" "vllm==0.29.0" "openai==3.16.2" "httpx==0.28.1" > "$A/logs/vllm_install.log" 2>&1 \
      || { tail -20 "$A/logs/vllm_install.log"; return 1; }
  fi
  bgsrv openjevhf_vllm env VLLM_USE_FLASHINFER_SAMPLER=0 "$LV/bin/vllm" serve "$d" --host 127.0.0.1 --served-model-name qwen \
    --port 8000 --enable-prefix-caching --max-model-len 16384 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":1}' \
    --trust-remote-code --max-num-seqs 128 --max-logprobs 64 --gdn-prefill-backend triton
  local vp; vp=$(cat "$W/e21_srv_openjevhf_vllm.pid")
  for _ in $(seq 1 240); do curl -sf -m 5 http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && break
    kill -0 "$vp" 2>/dev/null || { tail -30 "$A/logs/serve_openjevhf_vllm.out"; return 1; }; sleep 10; done
  bgsrv openjevhf env VLLM=http://localhost:8000/v1 TOKENIZER="$d" READOUT_T=0.85 READOUT_NOUL_T=1.829074 READOUT_NOUL_BIAS=0 \
    READOUT_TARGETED=1 READOUT_INSTR_STYLE=pyrepr SHIM_STAGGER=1 "$LV/bin/python" "$MD/openjev-helper/helper/shim.py" \
    --host 127.0.0.1 --port 8090
  for _ in $(seq 1 120); do   # the helper has no /health: a 2-option Choice answered on :8090 is its readiness check
    if curl -sf -m 120 -H 'Content-Type: application/json' -X POST "http://127.0.0.1:8090/v1/systemone" \
        -d '{"model": "openjev", "state": "The user asks to ship order 7.", "questions": {"q0": {"type": "choice", "instructions": "Which tool next?", "criteria": {"ship_order": null, "cancel_order": null}}}}' \
        > "$A/logs/ready_openjevhf.json" 2>/dev/null && grep -q '"answers"' "$A/logs/ready_openjevhf.json"; then
      printf '{"server_up": "%s", "slug": "openjevhf", "port": 8090}\n' "$(date -u +%FT%TZ)" >> "$A/serve_windows.jsonl"
      return 0
    fi
    sleep 10
  done
  return 1
}
setup_llama() {  # llama.cpp's CUDA server image at the digest of the logged 27B row (results/stage2/startlux/)
  [[ -x "$SLB/bin/llama-server" ]] && grep -q "$LLAMA_IMG" "$SLB/bin/llama-server" && return 0
  sudo mkdir -p "$SLB/bin" && sudo chown "$(id -un)" "$SLB/bin"
  timeout 40m sudo docker pull -q "$LLAMA_IMG" > "$A/logs/llama_pull.log" 2>&1 || { tail -5 "$A/logs/llama_pull.log"; return 1; }
  cat > "$SLB/bin/llama-server" <<EOF
#!/bin/bash
# llama.cpp's CUDA server image ($LLAMA_IMG); the model directory is mounted read-only at the same path
exec sudo docker run --rm --init --gpus "device=\${CUDA_VISIBLE_DEVICES:-all}" --network host \\
  -v $SLB/models:$SLB/models:ro $LLAMA_IMG "\$@"
EOF
  chmod +x "$SLB/bin/llama-server"
  "$SLB/bin/llama-server" --version > "$A/llama_server_version.txt" 2>&1
  echo "$LLAMA_IMG" > "$A/llama_image_digest.txt"
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
verify_ladder() {  # verify_ladder ROW: the plan's sha256 of the row's checkpoint files and repo files, and the base snapshot
  (cd "$R" && PY - "$PLAN" "$CK" "$1" "$A/ladder_verify_$1.json" "$(qbase)" <<'EOF'
import hashlib, json, os, sys
plan, ck, row, out, base = sys.argv[1:6]
d = json.load(open(plan))
r = d["rows"][row]
items = [(os.path.join(ck, r.get("dir") or "", k), v) for k, v in (r.get("sha256") or {}).items()]
items += list((r.get("repo_sha256") or {}).items())
res, bad = {}, []
for p, sha in items:
    got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    res[p] = {"expected": sha, "got": got, "ok": got == sha}
    bad += [p] if got != sha else []
rev_ok = d["base_model_revision"] in base and os.path.exists(os.path.join(base, "config.json"))
json.dump({"row": row, "ok": not bad and bool(res) and rev_ok, "mismatch": bad, "files": res, "base_model_path": base,
           "base_model_revision": d["base_model_revision"], "base_revision_in_path": rev_ok}, open(out, "w"), indent=1)
print(f"{row} sha256:", "all match" if not bad and res else f"MISMATCH {bad}", "| base snapshot",
      "ok" if rev_ok else "NOT the plan's revision")
sys.exit(1 if bad or not res or not rev_ok else 0)
EOF
  )
}

# ------------------------------------------------------------------------------------------------ stages
stage_rearm() {
  if [[ -n "${USED_MIN:-}" ]]; then echo $(( $(date +%s) - USED_MIN * 60 )) > "$W/e21_first_boot"; fi
  [[ -s "$W/e21_first_boot" ]] || date -d "$(uptime -s)" +%s > "$W/e21_first_boot"
  local left; left=$(mins_left)
  (( left > 0 )) || { echo "guard already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$left" "jev-e21-shift: power-off guard (host-hour cap)"
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E21 chain / server / runner process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e21_[0-9]|jevbench|explore_e21|serve_jev|startlux_decision|vllm|shim.py|uv pip|pip install|docker pull|snapshot_download' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e21-shift: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $left min left, idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU; $(df -BG / | awk 'NR==2{print $4}') free; used $(hours_used) h"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
}
stage_code() {  # the committed branch from the bundle (answers already on the host are kept: untracked files)
  local myb="$W/e21.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A/logs" runs
  st "code at $(git log --oneline -1 | cut -c1-72)"
  grep -q 'E21: a shifted DecisionGym split (DecisionGym-Shift)' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E21"; return 1; }
  if [[ -s "$C/questions.jsonl.gz" && -s "$C/questions_nosurf.jsonl.gz" && -s "$C/questions_nodist.jsonl.gz" ]]; then
    echo "L2 corpus: $(python3 -c "import json; print(json.load(open('$C/corpus.json'))['summary'])")"
  else
    echo "the L2 corpus is not in this bundle yet (model stages refuse to run)"
  fi
}
stage_setup() {
  stage_code || return 1
  ( prewarm "$(qbase)" "$HV/lib" "$VV/lib" "$SLB/models/StartLux-Decision-4B" ) &
  local pw=$!
  command -v uv > /dev/null || bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  PY -c "import jevharness, torch, transformers, sys; p = jevharness.__file__; print(p, torch.__version__, transformers.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_hvenv.txt" 2>&1 || { cat "$A/env_hvenv.txt"; echo "the hvenv does not import this checkout"; return 1; }
  [[ -x "$HV/bin/jevbench" ]] || { echo "no jevbench in the hvenv"; return 1; }
  "$VV/bin/python" -c "import peft" 2>/dev/null || timeout 15m uv pip install -q --python "$VV" "peft==0.21.2" > "$A/logs/v2venv_peft.log" 2>&1 \
    || echo "peft install failed (logs/v2venv_peft.log)"
  PV -c "import jevharness, torch, transformers, peft; print(jevharness.__file__, torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available())" \
    > "$A/env_v2venv.txt" 2>&1 || echo "v2 venv check failed (E12c will not serve): $(tail -2 "$A/env_v2venv.txt")"
  if [[ ! -x "$EV/bin/python" ]] || ! "$EV/bin/python" -c "import unsloth" > /dev/null 2>&1; then
    uv venv -q -p 3.11 "$EV" || return 1
    timeout 45m uv pip install -q --python "$EV" "unsloth==2026.10.2" fastapi uvicorn httpx pydantic pyyaml tqdm \
      python-dotenv openai > "$A/logs/e14venv_install.log" 2>&1 || echo "Unsloth venv install failed (the 0.8B head will not serve)"
    timeout 20m uv pip install -q --python "$EV" flash-linear-attention >> "$A/logs/e14venv_install.log" 2>&1 \
      || echo "flash-linear-attention not installed (the torch path is used)" | tee -a "$A/logs/e14venv_install.log"
  fi
  "$EV/bin/python" -c "import unsloth, torch, transformers; print('unsloth', unsloth.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.is_available())" \
    > "$A/env_e14venv.txt" 2>&1 || echo "e14venv check failed: $(tail -2 "$A/env_e14venv.txt")"
  # docker + the NVIDIA container runtime (llama.cpp's image for the 27B)
  if ! command -v docker > /dev/null; then
    curl -fsSL https://get.docker.com | sudo sh > "$A/logs/docker_install.log" 2>&1 || echo "docker install failed (the 27B will not serve)"
  fi
  if command -v docker > /dev/null && ! sudo docker info 2>/dev/null | grep -qi nvidia; then
    if ! command -v nvidia-ctk > /dev/null; then
      (curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
          | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
          | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null \
        && sudo apt-get update -q && sudo apt-get install -y -q nvidia-container-toolkit) > "$A/logs/nvctk_install.log" 2>&1 \
        || echo "nvidia-container-toolkit install failed"
    fi
    sudo nvidia-ctk runtime configure --runtime=docker > /dev/null 2>&1 && sudo systemctl restart docker
  fi
  # checkpoints from S3, sha256-checked against the logged rows' records
  if ! verify_frozen; then
    aws s3 sync "$(plan "d['rows']['frozen']['s3']")" "$(frozen_dir)" --only-show-errors --region "$REGION" || return 1
    verify_frozen || return 1
  fi
  : > "$A/ckpt_sha256.txt"
  aws s3 sync "$B/checkpoints/e12_mixed/" "$CKD/e12" --only-show-errors --region "$REGION" --exclude 'checkpoint-*' --exclude '*.bin' || return 1
  sha_check "$CKD/e12/adapter_model.safetensors" 5c9542d5eb264c726488f2efa8896de4202df66b00c500e26f5a2801b544fa5e e12 || echo "E12 adapter sha256 mismatch"
  aws s3 sync "$B/checkpoints/e12c_sl4b/selected/" "$CKD/e12c" --only-show-errors --region "$REGION" || return 1
  sha_check "$CKD/e12c/adapter_model.safetensors" 652e3beba4d05c4aa6bd8b568e5541ff08ef1c0e06ed71a0adda514099393c01 e12c || echo "E12c adapter sha256 mismatch"
  sha_check "$CKD/e12c/calibration.json" c99a91beb9f61313bef5cab0bfacc15777f724a6e0af5d1647fff74a313dc93f e12c || echo "E12c temperatures sha256 mismatch"
  aws s3 sync "$B/checkpoints/e14_small/A/final/" "$CKD/e14a" --only-show-errors --region "$REGION" || return 1
  sha_check "$CKD/e14a/joint_head.safetensors" a7ce7a7e78d50305a34fa900a69a9b6544cc8e6b3bfe7aa4caf9cb8c60a1c200 e14a || echo "0.8B head sha256 mismatch"
  sha_check "$CKD/e14a/unsloth_decision_config.json" a9f94cadfa2bacf5702883936c48da8c157e681b30bd80cabb963c9dce998ed4 e14a || true
  if [[ -s "$SLB/models/StartLux-Decision-4B/config.json" ]]; then
    (cd "$SLB/models/StartLux-Decision-4B" && sha256sum ./*.safetensors config.json tokenizer.json 2>/dev/null) > "$A/startlux_4b_sha256.txt"
  fi
  ls "$SLB" "$SLB/models" "$SLB/venv/bin/python" > "$A/startlux_image_contents.txt" 2>&1
  wait "$pw"
  # the vendor's StartLux environment (serve_startlux.sh installs it when the image lacks it) and the 4B base of E12c:
  # served once here and stopped (its manifest records the revision); then the large weights in the background
  sudo mkdir -p "$SLB" && sudo chown -R "$(id -un):$(id -gn)" "$SLB" 2>/dev/null
  serve_sl 4B > "$A/logs/setup_sl4b.log" 2>&1 && echo "StartLux env + 4B: ok" || echo "StartLux env / 4B failed: $(tail -2 "$A/logs/setup_sl4b.log" | tr '\n' ' ')"
  stop_servers
  predl sl9b startlux-models/StartLux-Decision-9B "$SLB/models/StartLux-Decision-9B" 342aad64840be8eee314455428a5a775f18ef547
  predl sl27b startlux-models/StartLux-Decision-27B-Q8_0-GGUF "$SLB/models/StartLux-Decision-27B-Q8_0" a54b311e3aede0c869bf006116bcfdddb1ef466f
  predl openjevhf openjev/openjev-FP8 "$MD/openjev-fp8" main
  touch "$W/predl_intern.running"     # Intern-Decision-4B into the HF cache the in-process backend reads
  ( PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('internlm/Intern-Decision-4B'))" \
      > "$A/logs/predl_intern.log" 2>&1; rm -f "$W/predl_intern.running" ) &
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s", "gpu": "%s", "mins_left": %s}\n' "$MY_IID" \
    "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" \
    "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" "$(mins_left)" > "$A/host.json"
  cat "$A/ckpt_sha256.txt" "$A/env_hvenv.txt" "$A/env_v2venv.txt" "$A/env_e14venv.txt" 2>/dev/null
  df -BG / | awk 'NR==2{print "disk free: "$4}'
}
probe_ok() {  # two consecutive clean probe rounds (registration (g)); waits up to PROBE_WAIT_MIN
  local end=$(( $(date +%s) + ${PROBE_WAIT_MIN:-180} * 60 )) clean=0
  while (( $(date +%s) < end )); do
    if PY "$EX" probe --log "$A/api_probe_log.txt" > /dev/null 2>&1; then clean=$((clean + 1)); else clean=0; fi
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
    PY "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --every 30 >> "$A/watch.log" 2>&1 &
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
  [[ -n "$(ls "$W"/e21_timebox_* 2>/dev/null)" && -f "$W/e21_timebox_${run##*e21_live_}" ]] && { st "$cfg stopped by its time box"; sync_out; return 0; }
  local ne
  ne=$(PY "$EX" errored --run "$run")
  if [[ "$ne" != 0 ]]; then
    st "$ne errored episode(s) in $run: one re-run after clean probes"
    probe_ok || { st "probes not clean; re-run skipped"; return 0; }
    setsid env PYTHONPATH="$R/src" JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$cfg" --out runs "$@" >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PY "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --every 30 --no-pause >> "$A/watch.log" 2>&1 &
    wpid=$!
    wait "$pid"; kill "$wpid" 2>/dev/null
  fi
  st "$cfg done; errored now $(PY "$EX" errored --run "$run"); $(PY "$EX" spend --runs runs --short 2>/dev/null)"
  sync_out
  return 0
}
live_stage() {  # live_stage SLUG [BOX_RESERVE_MIN]: smoke, gate, the row (registration (e)); the model is already served.
  # BOX_RESERVE_MIN (the addendum's time box, (b)): the runner is stopped once fewer minutes than this remain before the
  # power-off guard; the tasks it completed are the row
  local slug=$1 box=${2:-} g nfit tb=""
  cd "$R" || return 1
  keys || return 1
  run_live "$CFG/live_smoke_$slug.yaml" "runs/e21_live_smoke_$slug" || return 1
  g=$(PY "$EX" gate --smoke "runs/e21_live_smoke_$slug" --runs runs --n 200 --arms 1 --cap "$CAP_USD") \
    || { st "gate refuses live $slug: $g"; echo "$g" >> "$A/gates.jsonl"; return 3; }
  echo "$g" >> "$A/gates.jsonl"
  nfit=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['n_fit'])" "$g")
  st "live $slug gate: $g"
  if [[ -n "$box" ]]; then   # the time box: stop the runner (and its watchdog) at the reserve; records are kept
    ( while (( $(mins_left) > box )); do sleep 30; done
      st "live $slug: time box reached ($(mins_left) min left, reserve $box); runner stopped"
      touch "$W/e21_timebox_$slug"; pkill -f "jevbench run $CFG/live_$slug.yaml" ) & tb=$!
  fi
  if (( nfit < 200 )); then run_live "$CFG/live_$slug.yaml" "runs/e21_live_$slug" --max-tasks "$nfit"
  else run_live "$CFG/live_$slug.yaml" "runs/e21_live_$slug"; fi
  local rc=$?
  [[ -n "$tb" ]] && kill "$tb" 2>/dev/null
  leak_check || { unset GEMINI_API_KEY GEMINI_API_KEYS; return 1; }
  unset GEMINI_API_KEY GEMINI_API_KEYS
  return $rc
}
# minutes one model's main replay is expected to take (load included), for the re-rendering rule of registration (d) 5
est_min() { case "$1" in rl|e12|e14a|base|sft_v1) echo 20 ;; e12c|sl4b) echo 25 ;; sl9b) echo 35 ;; intern) echo 45 ;; openjevhf|sl27b) echo 90 ;; *) echo 30 ;; esac; }
cf_ok() {  # cf_ok SLUG: time for this model's two re-renderings before the guard, after the remaining models' main replays
  local need=0 s
  for s in ${REMAINING:-}; do [[ "$s" == [mr]_* ]] && need=$(( need + $(est_min "${s#[mr]_}") )); done
  for s in ${REMAINING:-}; do [[ "$s" == m_* && "$LIVE_ROWS" == *" ${s#m_} "* ]] && need=$(( need + 35 )); done
  need=$(( need + 20 + $(est_min "$1") ))     # final + this model's re-renderings (about one main replay)
  (( $(mins_left) >= need ))
}
replay() {  # replay SLUG VARIANT
  local slug=$1 v=$2 w=4 rc
  [[ "$slug" == intern ]] && w=1
  PY "$EX" replay --corpus "$C" --slug "$slug" --variant "$v" --cache "cache/s1_e21_${slug}_$v.sqlite" --workers "$w" \
    > "$A/logs/replay_${slug}_$v.log" 2>&1
  rc=$?
  st "$slug $v replay exit=$rc: $(PY "$EX" quickc --corpus "$C" --slug "$slug" --variant "$v" 2>/dev/null)"
  return $rc
}
model_stage() {  # model_stage SLUG [nolive]: nolive = the replays only (the live row ran in an earlier chain)
  local slug=$1 nolive=${2:-} t0 rc k
  t0=$(date +%s)
  [[ -s "$C/questions.jsonl.gz" && -s "$C/questions_nosurf.jsonl.gz" && -s "$C/questions_nodist.jsonl.gz" ]] \
    || { st "$slug: the L2 corpus is not on the host (send code first)"; return 1; }
  st "$slug: start (used $(hours_used) h, $(mins_left) min left)"
  stop_servers
  smi_start "$A/logs/smi_$slug.csv"
  start_model "$slug" > "$A/logs/start_$slug.log" 2>&1
  rc=$?
  if (( rc != 0 )); then
    smi_stop; stop_servers
    st "$slug: did not serve (exit $rc): $(tail -3 "$A/logs/start_$slug.log" | tr '\n' ' ' | cut -c1-300)"
    printf '{"slug": "%s", "served": false, "minutes": %s}\n' "$slug" "$(( ($(date +%s) - t0) / 60 ))" >> "$A/model_runs.jsonl"
    sync_out; return 0
  fi
  st "$slug: serving after $(( ($(date +%s) - t0) / 60 )) min"
  for k in 1 2; do replay "$slug" main && break; done
  sync_out
  if [[ "$LIVE_ROWS" == *" $slug "* && -z "$nolive" ]]; then
    live_stage "$slug" > "$A/logs/live_$slug.log" 2>&1
    st "$slug live exit=$?: $(tail -2 "$A/logs/live_$slug.log" | tr '\n' ' ' | cut -c1-200)"
  fi
  local cf=0
  if [[ "$MAIN_ONLY" == *" $slug "* ]]; then
    st "$slug: main rendering only (E21 addendum 2 (c))"
  elif cf_ok "$slug"; then
    cf=1
    replay "$slug" nosurf; replay "$slug" nodist
  else
    st "$slug: re-renderings skipped ($(mins_left) min left; registration (d) 5)"
  fi
  smi_stop
  stop_servers
  printf '{"slug": "%s", "served": true, "variants": %s, "minutes": %s, "used_h": %s}\n' "$slug" "$cf" \
    "$(( ($(date +%s) - t0) / 60 ))" "$(hours_used)" >> "$A/model_runs.jsonl"
  case "$slug" in openjevhf) rm -rf "$MD/openjev-fp8" ;; sl27b) rm -rf "$SLB/models/"*27B* ;; esac
  st "$slug: done in $(( ($(date +%s) - t0) / 60 )) min (used $(hours_used) h)"
  sync_out
  return 0
}
live_only_stage() {  # live_only_stage SLUG: the addendum's live row (serve, smoke, gate, row, stop); sl27b time-boxed
  local slug=$1 t0 rc
  t0=$(date +%s)
  st "$slug live row: start (used $(hours_used) h, $(mins_left) min left)"
  stop_servers
  start_model "$slug" > "$A/logs/start_live_$slug.log" 2>&1 || { st "$slug: did not serve for the live row: $(tail -2 "$A/logs/start_live_$slug.log" | tr '\n' ' ' | cut -c1-200)"; stop_servers; return 0; }
  st "$slug: serving after $(( ($(date +%s) - t0) / 60 )) min"
  if [[ "$slug" == sl27b ]]; then live_stage "$slug" "${BOX_RESERVE_MIN:-140}" > "$A/logs/live_$slug.log" 2>&1
  else live_stage "$slug" > "$A/logs/live_$slug.log" 2>&1; fi
  rc=$?
  st "$slug live exit=$rc: $(tail -2 "$A/logs/live_$slug.log" | tr '\n' ' ' | cut -c1-200)"
  stop_servers
  case "$slug" in openjevhf) rm -rf "$MD/openjev-fp8" ;; esac
  printf '{"slug": "%s", "live_row": true, "exit": %s, "minutes": %s, "used_h": %s}\n' "$slug" "$rc" \
    "$(( ($(date +%s) - t0) / 60 ))" "$(hours_used)" >> "$A/model_runs.jsonl"
  sync_out
  return 0
}
stage_final() {
  stop_servers
  sync_out
  st "final: used $(hours_used) h; $(PY "$EX" spend --runs runs --short 2>/dev/null)"
  if [[ "$POWEROFF" == 1 ]]; then sudo shutdown -h +1 "jev-e21-shift: done"; fi
}

st "chain start: $STAGES (host $MY_IID, used $(hours_used) h)"
set -- $STAGES
while (( $# > 0 )); do
  s=$1; shift
  export REMAINING="$*"
  t0=$(date +%s)
  cd "$R" 2>/dev/null || cd "$W"      # every stage runs from the checkout (relative paths), as setup leaves it
  case "$s" in
    rearm|code|setup|final) "stage_$s" > "$STATUS_DIR/stage_$s.log" 2>&1; rc=$? ;;
    m_*) model_stage "${s#m_}" > "$STATUS_DIR/stage_$s.log" 2>&1; rc=$? ;;
    r_*) model_stage "${s#r_}" nolive > "$STATUS_DIR/stage_$s.log" 2>&1; rc=$? ;;
    l_*) live_only_stage "${s#l_}" > "$STATUS_DIR/stage_$s.log" 2>&1; rc=$? ;;
    *) echo "unknown stage $s"; rc=1 ;;
  esac
  [[ -d "$A/logs" ]] && cp -f "$STATUS_DIR/stage_$s.log" "$A/logs/stage_$s.log" 2>/dev/null
  st "stage $s exit=$rc ($(( ($(date +%s) - t0) / 60 )) min): $(tail -2 "$STATUS_DIR/stage_$s.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  if (( rc != 0 )) && [[ "$s" == rearm || "$s" == code || "$s" == setup ]]; then
    st "chain stopped at $s"
    [[ " $STAGES " == *" final "* ]] && stage_final
    exit 1
  fi
done
st "chain end"
