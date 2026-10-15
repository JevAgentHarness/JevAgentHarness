#!/usr/bin/env bash
# Exploratory row E24, agent-state accuracy (L2) on public benchmarks (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08
# "E24: agent-state accuracy (L2) on public benchmarks"; descriptive, outside every Holm table). One NEW GPU host launched
# by this driver from E3's clone image (tag Name=jev-e24-public-l2), SSM only. Each open or trained decision model is
# served on the host, one at a time, as in its logged rows, and answers the L2-ALFWorld corpus
# (results/explore/e24_public_l2/alfworld, committed with the code). No agent LLM runs here and NO API key is read: the
# host makes no LLM call and plays no episode (offline answers only).
#
#   LOCAL:  bash scripts/aws/e24_public_l2.sh launch [itype ...]   # one new host (refuses if a jev-e24-public-l2 host exists)
#           E24_IID=i-... bash scripts/aws/e24_public_l2.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages (in the order sent; a failed model stage is logged and the chain goes on; `final` always runs when sent):
#     rearm        idle stop (IDLE_MIN) + power-off guard (GUARD_MIN of host time in total, USED_MIN carried over)
#     code         the committed branch in /opt/jev/e24 (from the bundle)
#     setup        code + the Unsloth venv (the 0.8B head); docker + the NVIDIA runtime (llama.cpp's image, 27B only);
#                  the checkpoints from S3, sha256-checked against their rows' records
#     m_<slug>     serve the model, answer the corpus (CORPUS, default alfworld), stop, delete large weights
#                  (slugs: e11 e12 e14a qwen4b sl4b sl9b sl27b; Part 2 adds e22, with CORPUS=tau2)
#     alfenv       a venv with alfworld 0.4.2 / textworld 1.7.0 (as E23's avenv) and the 274 games from S3 (sha256-checked)
#     labels       the consequence labels of the amendment (every answered (state, action) pair; CPU, LABEL_WORKERS)
#     final        stop the servers, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e24_public_l2/box -> s3://$BUCKET/e24_public_l2/box; answers -> e24_public_l2/<corpus>.
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E24_IID=${E24_IID:-}
# every other project host: never touched by this driver (E21's and E22's hosts included)
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e24-public-l2
BUNDLE=e24.bundle
S3OUT="$B/e24_public_l2"
GUARD_MIN=${GUARD_MIN:-470}          # 8 host-hours in total (registration (e)): the guard 7 h 50 min of host time
IDLE_MIN=${IDLE_MIN:-30}
POWEROFF=${POWEROFF:-1}
CORPUS=${CORPUS:-alfworld}
LLAMA_IMG=ghcr.io/ggml-org/llama.cpp@sha256:d87b02aafa74797a2d599b281c50468704bd0bcdc57bfcf31b570d593f1d285b

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E24_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E24_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E24_IID "* ]] && { echo "refused: $E24_IID is not the E24 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E24_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E24_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e24-public-l2: power-off guard (8 host-hours)"\n' "$GUARD_MIN" > "$ud"
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/e24_public_l2.sh" > "$tmp/e24_public_l2.sh"   # the committed driver
  awsl s3 cp "$tmp/e24_public_l2.sh" "$B/code/e24_public_l2.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e24_public_l2.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e24_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e24_public_l2.sh \$f --only-show-errors --region $REGION"
    printf 'export GUARD_MIN=%q IDLE_MIN=%q POWEROFF=%q E24_IID=%q USED_MIN=%q CORPUS=%q\n' \
      "$GUARD_MIN" "$IDLE_MIN" "$POWEROFF" "$E24_IID" "${USED_MIN:-}" "$CORPUS"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e24_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e24_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E24_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e24_driver.sh /tmp/e24_driver.sh --only-show-errors && bash /tmp/e24_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E24_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E24_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,20p' "$0"
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
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E24_IID" && "$MY_IID" != "$E24_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E24 host ${E24_IID:-?}"; exit 1
fi
R=$W/e24
A=$R/results/explore/e24_public_l2/box
C=$R/results/explore/e24_public_l2/$CORPUS
STATUS_DIR=$W/e24_status
HV=$W/hvenv                 # harness venv (serve_jev for the Qwen3-4B rows, the replays)
EV=$W/e14venv               # Unsloth 2026.10.2: serve_jev --backend clef (the 0.8B head, as E14 / E18 / E21)
CKD=$W/ck_e24               # E11, E12, the 0.8B head
SLB=/opt/startlux           # StartLux: the vendor's package, venv and weights (serve_startlux.sh)
EX=scripts/explore_e24_public_l2.py
CFG=configs/explore/e24_public_l2
AV=$W/e24_avenv             # alfworld 0.4.2 / textworld 1.7.0 + this checkout (the consequence labels)
LABEL_WORKERS=${LABEL_WORKERS:-6}
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=$W/e24_hf
mkdir -p "$STATUS_DIR" "$HF_HOME" "$CKD"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  aws s3 cp "$STATUS_DIR/status.txt" "$S3OUT/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION"
  [[ -d "$C" ]] && aws s3 sync "$C" "$S3OUT/$CORPUS" --only-show-errors --region "$REGION" --exclude '*' --include 'answers_*' \
    --include 'replay_*' --include 'consequence*'
  true
}
PY() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }
qbase() { cat "$W/base_model_path.txt"; }
hours_used() {
  local b; b=$(cat "$W/e24_first_boot" 2>/dev/null || date -d "$(uptime -s)" +%s)
  python3 -c "import time; print(round((time.time() - $b) / 3600, 3))"
}
mins_left() { local b; b=$(cat "$W/e24_first_boot" 2>/dev/null || date -d "$(uptime -s)" +%s); echo $(( GUARD_MIN - ( $(date +%s) - b ) / 60 )); }
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
  for f in "$W"/e24_srv_*.pid; do
    [[ -f "$f" ]] || continue
    p=$(cat "$f"); kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$p" 2>/dev/null; kill -9 "$p" 2>/dev/null
    rm -f "$f"
  done
  (cd "$R" && STOP=1 BASE=$SLB LOG_DIR=$SLB/logs bash scripts/aws/serve_startlux.sh > /dev/null 2>&1)
  sudo docker ps -q 2>/dev/null | xargs -r sudo docker stop > /dev/null 2>&1
  pkill -f 'startlux_decision' 2>/dev/null
  sleep 5
  true
}
bgsrv() {  # bgsrv TAG CMD...: a server process in its own process group; its pid file
  local tag=$1; shift
  setsid nohup "$@" > "$A/logs/serve_$tag.out" 2>&1 < /dev/null &
  echo $! > "$W/e24_srv_$tag.pid"
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
    [[ -f "$W/e24_srv_$tag.pid" ]] && ! kill -0 "$(cat "$W/e24_srv_$tag.pid")" 2>/dev/null && \
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
setup_llama() {  # llama.cpp's CUDA server image at the digest of the logged 27B row (results/stage2/startlux/)
  [[ -x "$SLB/bin/llama-server" ]] && grep -q "$LLAMA_IMG" "$SLB/bin/llama-server" && return 0
  if ! command -v docker > /dev/null; then
    curl -fsSL https://get.docker.com | sudo sh > "$A/logs/docker_install.log" 2>&1 || { echo "docker install failed"; return 1; }
  fi
  if ! sudo docker info 2>/dev/null | grep -qi nvidia; then
    if ! command -v nvidia-ctk > /dev/null; then
      (curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
          | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
          | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null \
        && sudo apt-get update -q && sudo apt-get install -y -q nvidia-container-toolkit) > "$A/logs/nvctk_install.log" 2>&1 \
        || { echo "nvidia-container-toolkit install failed"; return 1; }
    fi
    sudo nvidia-ctk runtime configure --runtime=docker > /dev/null 2>&1 && sudo systemctl restart docker
  fi
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
hfdl() {  # hfdl REPO DIR REV [patterns...]: a pinned snapshot into its own directory; the revision is recorded
  local repo=$1 dir=$2 rev=$3; shift 3
  mkdir -p "$dir"
  PY - "$repo" "$dir" "$rev" "$@" > "$A/logs/hf_$(basename "$dir").log" 2>&1 <<'EOF2' || return 1
import sys
from huggingface_hub import HfApi, snapshot_download
repo, d, rev, pats = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
snapshot_download(repo, revision=rev, local_dir=d, allow_patterns=pats or None)
print(HfApi().model_info(repo, revision=rev).sha)
EOF2
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
gpu_gib() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
start_model() {  # start_model SLUG: serve it (configs/explore/e24_public_l2/backends.yaml names its port)
  local slug=$1 d
  case "$slug" in
    e11)
      d=$CKD/e11
      bgsrv e11 env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev --model "$(qbase)" --name e11-alfworld-4b \
        --port 8081 --adapter "$d" --calibration "$d/calibration.json"
      wait_health 8081 e11-alfworld-4b e11 ;;
    e12)
      d=$CKD/e12
      bgsrv e12 env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev --model "$(qbase)" \
        --name agentdecision-4b-mixed --port 8081 --adapter "$d" --calibration "$d/calibration.json"
      wait_health 8081 agentdecision-4b-mixed e12 ;;
    qwen4b)
      bgsrv qwen4b env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev --model "$(qbase)" --name qwen3-4b \
        --port 8081
      wait_health 8081 qwen3-4b qwen4b ;;
    e14a)
      bgsrv e14a env PYTHONPATH="$R/src" "$EV/bin/python" -m jevharness.training.serve_jev --backend clef --model "$CKD/e14a" \
        --name e14-a --port 8083
      wait_health 8083 e14-a e14a ;;
    e22)
      bgsrv e22 env PYTHONPATH="$R/src" "$EV/bin/python" -m jevharness.training.serve_jev --backend clef --model "$CKD/e22" \
        --name e22-tau2-0.8b --port 8083
      wait_health 8083 e22-tau2-0.8b e22 ;;
    sl4b) serve_sl 4B ;;
    sl9b) (( $(gpu_gib) >= 32 )) || { echo "9B needs >= 32 GiB (A7.18 (g)); not run on $(gpu_gib) GiB"; return 1; }
      wait_predl sl9b; serve_sl 9B ;;
    sl27b) (( $(gpu_gib) >= 40 )) || { echo "27B (Q8_0) needs >= 40 GiB (A7.18 (g)); not run on $(gpu_gib) GiB"; return 1; }
      wait_predl sl27b; setup_llama || return 1; serve_sl 27B ;;
    *) echo "unknown slug $slug"; return 1 ;;
  esac
}

# ------------------------------------------------------------------------------------------------ stages
stage_rearm() {
  if [[ -n "${USED_MIN:-}" ]]; then echo $(( $(date +%s) - USED_MIN * 60 )) > "$W/e24_first_boot"; fi
  [[ -s "$W/e24_first_boot" ]] || date -d "$(uptime -s)" +%s > "$W/e24_first_boot"
  local left; left=$(mins_left)
  (( left > 0 )) || { echo "guard already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$left" "jev-e24-public-l2: power-off guard (host-hour cap)"
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E24 chain / server / replay process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e24_[0-9]|explore_e24|serve_jev|startlux_decision|uv pip|pip install|docker pull|snapshot_download' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e24-public-l2: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $left min left, idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU; $(df -BG / | awk 'NR==2{print $4}') free; used $(hours_used) h"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
}
stage_code() {  # the committed branch from the bundle (answers already on the host are kept: untracked files)
  local myb="$W/e24.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A/logs"
  st "code at $(git log --oneline -1 | cut -c1-72)"
  grep -q 'E24: agent-state accuracy (L2) on public benchmarks' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E24"; return 1; }
  if [[ -s "$C/questions.jsonl.gz" ]]; then
    echo "corpus $CORPUS: $(python3 -c "import json; s=json.load(open('$C/corpus.json'))['stats']; print({k: s[k] for k in ('games', 'states', 'unique_questions')})")"
  else
    echo "the $CORPUS corpus is not in this bundle (model stages refuse to run)"
  fi
}
stage_setup() {
  stage_code || return 1
  ( prewarm "$(qbase)" "$HV/lib" "$SLB/models/StartLux-Decision-4B" ) &
  local pw=$!
  command -v uv > /dev/null || bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  PY -c "import jevharness, torch, transformers, peft, sys; p = jevharness.__file__; print(p, torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_hvenv.txt" 2>&1 || { cat "$A/env_hvenv.txt"; echo "the hvenv does not import this checkout"; return 1; }
  [[ -s "$W/base_model_path.txt" ]] || { echo "no base_model_path.txt"; return 1; }
  cp -f "$W/base_model_path.txt" "$A/base_model_path.txt"
  grep -q "1cfa9a7208912126459214e8b04321603b3df60c" "$W/base_model_path.txt" || { echo "base snapshot is not E11's / E12's"; return 1; }
  if [[ ! -x "$EV/bin/python" ]] || ! "$EV/bin/python" -c "import unsloth" > /dev/null 2>&1; then
    uv venv -q -p 3.11 "$EV" || return 1
    timeout 45m uv pip install -q --python "$EV" "unsloth==2026.10.2" fastapi uvicorn httpx pydantic pyyaml tqdm \
      python-dotenv openai > "$A/logs/e14venv_install.log" 2>&1 || echo "Unsloth venv install failed (the 0.8B head will not serve)"
    timeout 20m uv pip install -q --python "$EV" flash-linear-attention >> "$A/logs/e14venv_install.log" 2>&1 \
      || echo "flash-linear-attention not installed (the torch path is used)" | tee -a "$A/logs/e14venv_install.log"
  fi
  "$EV/bin/python" -c "import unsloth, torch, transformers; print('unsloth', unsloth.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.is_available())" \
    > "$A/env_e14venv.txt" 2>&1 || echo "e14venv check failed: $(tail -2 "$A/env_e14venv.txt")"
  # checkpoints from S3, sha256-checked against the logged rows' records (registration (c))
  : > "$A/ckpt_sha256.txt"
  local ok=1
  # the whole checkpoint directory, as E23 (serve_jev reads the tokenizer and chat template from the adapter directory)
  aws s3 sync "$B/checkpoints/e11_alfworld/" "$CKD/e11" --only-show-errors --region "$REGION" || return 1
  sha_check "$CKD/e11/adapter_model.safetensors" 53c5893b60c262d6ac2356ad9fd537ee0e722085abb5a7e14d2cf2ae7cc6b659 e11 || ok=0
  sha_check "$CKD/e11/adapter_config.json" 06d8e3d54909591eaa3e9a09a9b83bc5dedeebf4ac9e88deeb8e39abb349fa1d e11 || ok=0
  sha_check "$CKD/e11/calibration.json" d089d0e76e08cc68dd9743d0fd36fcbf04044c16afe3ae76ddacf4cc1515b382 e11 || ok=0
  (( ok == 1 )) || { echo "E11 checkpoint sha256 mismatch: the e11 row stops (registration (c))"; touch "$W/e24_e11_bad"; }
  aws s3 sync "$B/checkpoints/e12_mixed/" "$CKD/e12" --only-show-errors --region "$REGION" --exclude 'checkpoint-*' --exclude '*.bin' || return 1
  sha_check "$CKD/e12/adapter_model.safetensors" 5c9542d5eb264c726488f2efa8896de4202df66b00c500e26f5a2801b544fa5e e12 || { echo "E12 adapter sha256 mismatch"; touch "$W/e24_e12_bad"; }
  aws s3 sync "$B/checkpoints/e14_small/A/final/" "$CKD/e14a" --only-show-errors --region "$REGION" || return 1
  sha_check "$CKD/e14a/joint_head.safetensors" a7ce7a7e78d50305a34fa900a69a9b6544cc8e6b3bfe7aa4caf9cb8c60a1c200 e14a || { echo "0.8B head sha256 mismatch"; touch "$W/e24_e14a_bad"; }
  sha_check "$CKD/e14a/unsloth_decision_config.json" a9f94cadfa2bacf5702883936c48da8c157e681b30bd80cabb963c9dce998ed4 e14a || true
  # E24 Part 2: E22's tau^2-trained 0.8B head (sha256 against E22's record results/explore/e22_tau2/box/train/final_sha256.txt)
  aws s3 sync "$B/checkpoints/e22_tau2/final/" "$CKD/e22" --only-show-errors --region "$REGION" || return 1
  sha_check "$CKD/e22/joint_head.safetensors" fbc4278c64c64f063220926ba36f09cbc7326a1e63719a28732de4cbeeafcd1f e22 || { echo "E22 head sha256 mismatch"; touch "$W/e24_e22_bad"; }
  sha_check "$CKD/e22/unsloth_decision_config.json" a196c06a3ef955a4fb3585dc9963a7d43b6a604610e962304a079baca7a9d42e e22 || { echo "E22 config sha256 mismatch"; touch "$W/e24_e22_bad"; }
  wait "$pw"
  # the vendor's StartLux environment (serve_startlux.sh installs it when the image lacks it) and the 4B: served once
  # here and stopped (its manifest records the revision); then the 9B / 27B weights in the background
  sudo mkdir -p "$SLB" && sudo chown -R "$(id -un):$(id -gn)" "$SLB" 2>/dev/null
  serve_sl 4B > "$A/logs/setup_sl4b.log" 2>&1 && echo "StartLux env + 4B: ok" || echo "StartLux env / 4B failed: $(tail -2 "$A/logs/setup_sl4b.log" | tr '\n' ' ')"
  stop_servers
  if (( $(gpu_gib) >= 32 )); then
    predl sl9b startlux-models/StartLux-Decision-9B "$SLB/models/StartLux-Decision-9B" 342aad64840be8eee314455428a5a775f18ef547
  fi
  if (( $(gpu_gib) >= 40 )); then
    predl sl27b startlux-models/StartLux-Decision-27B-Q8_0-GGUF "$SLB/models/StartLux-Decision-27B-Q8_0" a54b311e3aede0c869bf006116bcfdddb1ef466f
  fi
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s", "gpu": "%s", "mins_left": %s}\n' "$MY_IID" \
    "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" \
    "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" "$(mins_left)" > "$A/host.json"
  cat "$A/ckpt_sha256.txt" "$A/env_hvenv.txt" "$A/env_e14venv.txt" 2>/dev/null
  df -BG / | awk 'NR==2{print "disk free: "$4}'
}
model_stage() {  # model_stage SLUG
  local slug=$1 t0 rc k
  t0=$(date +%s)
  [[ -s "$C/questions.jsonl.gz" ]] || { st "$slug: the $CORPUS corpus is not on the host (send code first)"; return 1; }
  [[ -f "$W/e24_${slug}_bad" ]] && { st "$slug: checkpoint sha256 mismatch; not run"; return 0; }
  st "$slug: start (used $(hours_used) h, $(mins_left) min left)"
  stop_servers
  smi_start "$A/logs/smi_${CORPUS}_$slug.csv"
  start_model "$slug" > "$A/logs/start_${CORPUS}_$slug.log" 2>&1
  rc=$?
  if (( rc != 0 )); then
    smi_stop; stop_servers
    st "$slug: did not serve (exit $rc): $(tail -3 "$A/logs/start_${CORPUS}_$slug.log" | tr '\n' ' ' | cut -c1-300)"
    printf '{"slug": "%s", "corpus": "%s", "served": false, "minutes": %s}\n' "$slug" "$CORPUS" "$(( ($(date +%s) - t0) / 60 ))" >> "$A/model_runs.jsonl"
    sync_out; return 0
  fi
  st "$slug: serving after $(( ($(date +%s) - t0) / 60 )) min"
  for k in 1 2; do
    PY "$EX" replay --corpus "$C" --slug "$slug" --cache "cache/s1_e24_${CORPUS}_${slug}.sqlite" --workers 4 \
      > "$A/logs/replay_${CORPUS}_${slug}.log" 2>&1 && break
  done
  st "$slug replay: $(PY "$EX" quick --corpus "$C" --slug "$slug" 2>/dev/null)"
  smi_stop
  stop_servers
  printf '{"slug": "%s", "corpus": "%s", "served": true, "minutes": %s, "used_h": %s}\n' "$slug" "$CORPUS" \
    "$(( ($(date +%s) - t0) / 60 ))" "$(hours_used)" >> "$A/model_runs.jsonl"
  case "$slug" in sl27b) rm -rf "$SLB/models/"*27B* ;; esac
  st "$slug: done in $(( ($(date +%s) - t0) / 60 )) min (used $(hours_used) h)"
  sync_out
  return 0
}
stage_alfenv() {  # the consequence labels' environment: E23's avenv recipe, and the games the corpus was replayed in
  cd "$R" || return 1
  if [[ ! -x "$AV/bin/python" ]] || ! "$AV/bin/python" -c "import alfworld, textworld" > /dev/null 2>&1; then
    uv venv -q -p 3.11 "$AV" || return 1
    timeout 30m uv pip install -q --python "$AV" -e ".[all]" "alfworld==0.4.2" "textworld==1.7.0" > "$A/logs/avenv_install.log" 2>&1 \
      || { tail -5 "$A/logs/avenv_install.log"; return 1; }
  fi
  PYTHONPATH="$R/src" "$AV/bin/python" -c "import importlib.metadata as m, jevharness; print(m.version('alfworld'), m.version('textworld'), m.version('fast-downward-textworld'), jevharness.__file__)" \
    > "$A/env_avenv.txt" 2>&1 || { cat "$A/env_avenv.txt"; return 1; }
  grep -q "^0.4.2 1.7.0 20.6.4 $R/src" "$A/env_avenv.txt" || { echo "avenv versions differ from the corpus build: $(cat "$A/env_avenv.txt")"; return 1; }
  mkdir -p data/raw/alfworld
  aws s3 cp "$S3OUT/alfworld_games.tar.gz" "$W/e24_games.tar.gz" --only-show-errors --region "$REGION" || return 1
  echo "$(sha256sum "$W/e24_games.tar.gz" | cut -d' ' -f1)" > "$A/games_tar_sha256.txt"
  grep -q "$(cat "$R/$CFG/alfworld_games.sha256")" "$A/games_tar_sha256.txt" || { echo "games archive sha256 mismatch"; return 1; }
  tar -xzf "$W/e24_games.tar.gz" -C data/raw/alfworld || return 1
  echo "games: $(find data/raw/alfworld/json_2.1.1 -name game.tw-pddl | wc -l)"
}
stage_labels() {  # every answered (state, action) pair: the amendment's consequence rule, LABEL_WORKERS games at a time
  cd "$R" || return 1
  [[ -x "$AV/bin/python" ]] || { echo "no avenv (send alfenv first)"; return 1; }
  local t0; t0=$(date +%s)
  st "labels: start ($(ls "$C"/answers_*.jsonl 2>/dev/null | wc -l) answer files; used $(hours_used) h)"
  TMPDIR=/dev/shm PYTHONPATH="$R/src" "$AV/bin/python" "$EX" consequence --corpus "$C" --workers "$LABEL_WORKERS" \
    > "$A/logs/consequence.log" 2>&1
  local rc=$?
  st "labels: exit $rc in $(( ($(date +%s) - t0) / 60 )) min: $(tail -1 "$A/logs/consequence.log" | cut -c1-200)"
  sync_out
  return $rc
}
stage_final() {
  stop_servers
  printf '{"final_utc": "%s", "used_h": %s}\n' "$(date -u +%FT%TZ)" "$(hours_used)" >> "$A/host_hours.jsonl"
  sync_out
  st "final: used $(hours_used) h"
  if [[ "$POWEROFF" == 1 ]]; then sudo shutdown -h +1 "jev-e24-public-l2: done"; fi
}

st "chain start: $STAGES (host $MY_IID, used $(hours_used) h, corpus $CORPUS)"
set -- $STAGES
while (( $# > 0 )); do
  s=$1; shift
  export REMAINING="$*"
  t0=$(date +%s)
  cd "$R" 2>/dev/null || cd "$W"      # every stage runs from the checkout (relative paths), as setup leaves it
  case "$s" in
    rearm|code|setup|alfenv|labels|final) "stage_$s" > "$STATUS_DIR/stage_$s.log" 2>&1; rc=$? ;;
    m_*) model_stage "${s#m_}" > "$STATUS_DIR/stage_$s.log" 2>&1; rc=$? ;;
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
