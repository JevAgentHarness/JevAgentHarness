#!/usr/bin/env bash
# GPU round 6 = JevAgentRL-v2 (exploratory, PREREG amendment A7.19): the System-1 post-training chain on the Miles RL
# framework (SGLang rollout + FSDP2 trainer, https://github.com/radixark/miles). Plan: docs/MILES_PORT.md; rules:
# results/gpu_round6_miles/TRAINING_MANIFEST.md. Development / training data only: nothing here is an evaluation number.
#
#   base rule (StartLux-Decision-4B vs Qwen3-4B) -> SFT v2 (Miles, data v1) -> OPD v2 (Miles rollouts, Jev teacher)
#   -> GRPO v2 (Miles rollouts, call-count-penalised success) -> gates G1-G5 + the stress instrument -> freeze
#
# Seeds (training/seed_guard.py where seeds are drawn): train >= 10000 outside 100000-100499, 200000-200499,
# 300000-300999 and 5000-5999 (OPD pool 60000-63999, GRPO pool 70000-72999); development 0-4999 outside 5000-5149
# (stress instrument 2000-2199). Never calibration, evaluation or Stage-2 seeds.
#
#   LOCAL (AWS CLI with the role session: env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws --region us-east-1):
#     bash scripts/aws/gpu_round6_miles.sh launch       # g6e.12xlarge -> p4d.24xlarge -> p5.48xlarge, first AZ with
#                                                       # capacity; Name=jev-rl-miles, Project=jev; shutdown behaviour
#                                                       # stop; user-data: 30 h power-off guard + 45-min idle stop
#     bash scripts/aws/gpu_round6_miles.sh push         # git bundle of this branch + this script -> s3://$BUCKET/code/
#     bash scripts/aws/gpu_round6_miles.sh send "STAGES" [ENV=VAL ...]   # SSM: box side, detached
#     bash scripts/aws/gpu_round6_miles.sh status [N]   # the box's status.txt from S3
#     bash scripts/aws/gpu_round6_miles.sh stop         # stop the jev-rl-miles instance (never any other box)
#   BOX (as ubuntu, under /opt/jev; called by `send`):  bash gpu_round6_miles.sh box "setup smoke_engine ..."
# Every stage logs to runs/gpu_round6_miles/<stage>.log, appends to runs/gpu_round6_miles/status.txt and syncs it to
# s3://$BUCKET/gpu_round6_miles/; checkpoints go to s3://$BUCKET/checkpoints/round6_miles/. API keys come from SSM
# Parameter Store into process environments only (never on disk, never printed).
set -uo pipefail
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-<worktree>}
TAG=jev-rl-miles
BUNDLE=jev-rl-miles.bundle
SCRIPT=gpu_round6_miles.sh
S3OUT="$B/gpu_round6_miles"
S3CK="$B/checkpoints/round6_miles"

# =============================================================================================== local side
awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws --region "$REGION" "$@"; }

instance_id() {  # the jev-rl-miles box (pending / running / stopping / stopped); exactly one expected
  awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text
}

if [[ "$MODE" == launch ]]; then
  existing=$(instance_id)
  if [[ -n "$existing" ]]; then echo "already exists: $existing" >&2; exit 1; fi
  TYPES=${TYPES:-"g6e.12xlarge p4d.24xlarge p5.48xlarge"}
  SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
  SG=${SG:?set SG to a security group id}            # jev-train-noingress (SSM only)
  PROFILE=${PROFILE:?set PROFILE to an instance profile}         # SSM + the bucket + ssm:GetParameter on ${SSM_PREFIX:?set SSM_PREFIX}/*
  AMI=${AMI:-<aws-resource-id>}         # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.14 (Ubuntu 26.04) 20260926
  GUARD_MIN=${GUARD_MIN:-1800}
  IDLE_MIN=${IDLE_MIN:-45}
  UD=$(mktemp)
  trap 'rm -f "$UD" "$UD.err"' EXIT
  cat > "$UD" <<EOF
#!/bin/bash
shutdown -h +$GUARD_MIN "jev-rl-miles: 30 h guard"
mkdir -p /opt/jev && chown ubuntu:ubuntu /opt/jev
cat > /usr/local/bin/jev-idle-stop <<'IDLE'
#!/bin/bash
# stop the box after $IDLE_MIN min without a stage process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'jevharness|jevbench|setup_gpu|gpu_round6_miles|opt.jev.r6_|hf download|uv pip|docker pull|miles|sglang|ray::' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-rl-miles: idle $IDLE_MIN min"; fi
IDLE
chmod +x /usr/local/bin/jev-idle-stop
echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' > /etc/cron.d/jev-idle-stop
EOF
  for t in $TYPES; do
    for s in $SUBNETS; do
      if id=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$t" --subnet-id "$s" \
            --security-group-ids "$SG" --iam-instance-profile "Name=$PROFILE" \
            --instance-initiated-shutdown-behavior stop --user-data "file://$UD" \
            --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":700,"VolumeType":"gp3","Throughput":750,"Iops":8000,"DeleteOnTermination":true}}]' \
            --metadata-options HttpTokens=required,HttpEndpoint=enabled \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG},{Key=Project,Value=jev}]" \
              "ResourceType=volume,Tags=[{Key=Name,Value=$TAG},{Key=Project,Value=jev}]" \
            --query 'Instances[0].InstanceId' --output text 2>"$UD.err"); then
        echo "$id $t $s"
        exit 0
      fi
      if ! grep -q -E 'InsufficientInstanceCapacity|Unsupported|not supported|InstanceLimitExceeded|VcpuLimitExceeded' "$UD.err"; then
        cat "$UD.err" >&2
        exit 1
      fi
      echo "no capacity: $t in $s ($(grep -o -E '[A-Za-z]+(Capacity|Limit)[A-Za-z]*|Unsupported' "$UD.err" | head -1))" >&2
    done
  done
  echo "no capacity for any of: $TYPES" >&2
  exit 1
fi

if [[ "$MODE" == push ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors
  awsl s3 cp "$ROOT/scripts/aws/$SCRIPT" "$B/code/$SCRIPT" --only-show-errors
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH")"
  rm -rf "$tmp"
  exit 0
fi

if [[ "$MODE" == send ]]; then
  STAGES=${1:?usage: send \"STAGES\" [ENV_ASSIGNMENTS...]}
  shift
  iid=$(instance_id)
  [[ -n "$iid" ]] || { echo "no $TAG instance" >&2; exit 1; }
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    # a private copy per invocation: bash reads a running script incrementally, so a later send never overwrites it
    echo 'f=/opt/jev/r6_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/$SCRIPT \$f --only-show-errors --region $REGION"
    printf 'cd /opt/jev && setsid nohup env %s bash "$f" box %q >> /opt/jev/r6_box.log 2>&1 < /dev/null &\n' \
      "${*:-R6=1}" "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/r6_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$iid" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/r6_driver.sh /tmp/r6_driver.sh --only-show-errors && bash /tmp/r6_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi

if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi

if [[ "$MODE" == stop ]]; then
  iid=$(instance_id)
  [[ -n "$iid" ]] || { echo "no $TAG instance" >&2; exit 1; }
  awsl ec2 stop-instances --instance-ids "$iid" --query 'StoppingInstances[].[InstanceId,CurrentState.Name]' --output text
  exit 0
fi

if [[ "$MODE" == start ]]; then
  iid=$(instance_id)
  [[ -n "$iid" ]] || { echo "no $TAG instance" >&2; exit 1; }
  awsl ec2 start-instances --instance-ids "$iid" --query 'StartingInstances[].[InstanceId,CurrentState.Name]' --output text
  exit 0
fi

if [[ "$MODE" != box ]]; then
  sed -n '2,26p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
W=/opt/jev
cd "$W" || exit 1
# the idle-stop counter survives a stop / start in /var/tmp: reset it so a restarted box is not stopped at once
sudo sh -c 'echo 0 > /var/tmp/jev-idle-count' 2>/dev/null
MYB="$W/bundle.$$.bundle"
aws s3 cp "$B/code/$BUNDLE" "$MYB" --only-show-errors --region "$REGION"
if [[ ! -d "$W/JevRL" ]]; then
  git clone -q -b "$BRANCH" "$MYB" "$W/JevRL"
fi
cd "$W/JevRL" || exit 1
git fetch -q "$MYB" "$BRANCH" && git reset -q --hard FETCH_HEAD
rm -f "$MYB"
OUT=runs/gpu_round6_miles
EVAL=$OUT/eval
CK=checkpoints/r6
mkdir -p "$OUT" "$EVAL" "$CK"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$OUT/status.txt"
  aws s3 sync "$OUT" "$S3OUT" --only-show-errors --region "$REGION" --exclude '*.jsonl.gz' --exclude 'labelled/*'
}
sync_ck() {  # sync_ck NAME: HF weights + metrics (no optimizer / DCP state) -> s3://.../checkpoints/round6_miles/NAME
  aws s3 sync "$CK/$1" "$S3CK/$1" --only-show-errors --region "$REGION" --exclude 'dcp/*' --exclude '*optimizer*'
}
for v in GEMINI_API_KEY GEMINI_API_KEYS OPENROUTER_API_KEY HF_TOKEN; do
  val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value \
        --output text 2>/dev/null) || val=""
  if [[ -n "$val" && "$val" != None ]]; then export "$v=$val"; fi
done
unset val
export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
st "box start $(git log --oneline -1) stages=[$STAGES] gpu=${GPU} x${NGPU}"

IMAGE=${MILES_IMAGE:-radixark/miles:latest}
QWEN=Qwen/Qwen3-4B
QWEN_REV=1cfa9a7208912126459214e8b04321603b3df60c
STARTLUX=startlux-models/StartLux-Decision-4B
STARTLUX_REV=3890ae0c174215e1555cc94c882ef45c5fe87cad
MODELS=$W/models
D0=data/train_v0
D1=data/train_v1

# Miles runs inside its image; the repository, the models and the outputs are bind-mounted at the same paths, and the
# API keys are passed by name (docker -e NAME copies the value from this process's environment; nothing is printed).
DOCKER=docker
docker info > /dev/null 2>&1 || DOCKER="sudo docker"
miles() {  # miles "COMMAND": run COMMAND in the Miles image with every GPU (PYTHONPATH = this checkout's src)
  local rc
  # MILES_NAME (optional): a container name, so a budget watchdog can stop exactly this run
  $DOCKER run --rm ${MILES_NAME:+--name "$MILES_NAME"} --gpus all --ipc=host --network=host --shm-size=64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$W:$W" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -w "$W/JevRL" \
    -e PYTHONPATH="$W/JevRL/src/jevharness/training/miles_port/_startup:$W/JevRL/src" -e PYTHONUNBUFFERED=1 -e TOKENIZERS_PARALLELISM=false \
    -e GEMINI_API_KEY -e GEMINI_API_KEYS -e OPENROUTER_API_KEY -e HF_TOKEN -e JEV_MILES_CONFIG -e JEV_MILES_OUT \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$IMAGE" bash -lc "$1"
  rc=$?
  sudo chown -R "$(id -u):$(id -g)" "$W/JevRL/runs" "$W/JevRL/checkpoints" "$W/models" 2>/dev/null
  return $rc
}

stage_setup() {  # project venv (evaluation side), the Miles image, pinned model snapshots
  RUN_TESTS=0 CHECK_HF=0 bash scripts/train/setup_gpu.sh
  . .venv/bin/activate
  uv pip install --python .venv/bin/python flash-linear-attention==0.5.2 || echo "fla not installed in the venv (slow path)"
  $DOCKER pull "$IMAGE" || return 1
  $DOCKER inspect --format '{{index .RepoDigests 0}}' "$IMAGE" > "$OUT/miles_image_digest.txt"
  mkdir -p "$MODELS"
  python - "$MODELS" "$QWEN" "$QWEN_REV" "$STARTLUX" "$STARTLUX_REV" <<'EOF'
import sys
from huggingface_hub import snapshot_download
root, pairs = sys.argv[1], sys.argv[2:]
for repo, rev in zip(pairs[::2], pairs[1::2]):
    p = snapshot_download(repo, revision=rev, local_dir=f"{root}/{repo.split('/')[-1]}")
    print(repo, rev, p)
EOF
  python - <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("internlm/Intern-Decision-4B"))
EOF
}

stage_smoke_engine() {  # A: the image on this GPU; B: SGLang option log-probs vs the HF forward, per base (no training)
  miles "nvidia-smi; python -m jevharness.training.miles_port.smoke env" > "$OUT/smoke_env.txt" 2>&1
  local rc=0
  for m in Qwen3-4B StartLux-Decision-4B; do
    miles "python -m jevharness.training.miles_port.smoke engine --model $MODELS/$m --dev $D1/dev.jsonl.gz --n 32 \
      --out $OUT/smoke_engine_$m.json" > "$OUT/smoke_engine_$m.log" 2>&1 || rc=1
  done
  return $rc
}

stage_poweroff() {
  aws s3 sync "$OUT" "$S3OUT" --only-show-errors --region "$REGION" --exclude '*.jsonl.gz' --exclude 'labelled/*'
  sudo shutdown -h +3 "jev-rl-miles: chain finished"
}

# ----------------------------------------------------------------------------------------------- stage runner
if [[ -f scripts/aws/gpu_round6_stages.sh ]]; then
  # shellcheck source=/dev/null
  . scripts/aws/gpu_round6_stages.sh   # the training / evaluation stages (kept in the bundle, versioned with the code)
fi
for s in $STAGES; do
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $s"; continue; fi
  if [[ -z "${VIRTUAL_ENV:-}" && -x .venv/bin/python ]]; then . .venv/bin/activate; fi
  st "stage $s start"
  t0=$(date +%s)
  ( while sleep 300; do aws s3 sync "$OUT" "$S3OUT" --only-show-errors --region "$REGION" --exclude '*.jsonl.gz' \
      --exclude 'labelled/*' > /dev/null 2>&1; done ) &
  syncer=$!
  ( set -o pipefail; "stage_$s" ) > "$OUT/$s.log" 2>&1
  rc=$?
  kill "$syncer" 2>/dev/null; wait "$syncer" 2>/dev/null
  st "stage $s exit=$rc ($(( ($(date +%s) - t0) / 60 )) min)"
  if [[ $rc -ne 0 && "${STOP_ON_FAIL:-1}" == 1 ]]; then st "stopping this invocation after the failed stage $s"; break; fi
done
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv > "$OUT/nvidia_smi_last.txt" 2>&1
st "box done [$STAGES]"
