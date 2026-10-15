#!/usr/bin/env bash
# GPU round 5 = JevAgentRL phase 1 (docs/SCOPE_V2.md deliverable 4; docs/TRAINING_RECIPES.md §17-§18): train our own
# System-1 decider (Qwen3-4B LoRA) so that it is strong INSIDE the harness, then freeze it by the §18.4 gates.
# Development / training data only: nothing here is an evaluation number (seed rules below).
#
#   SFT v1 (R3-v1, data/train_v1)  ->  OPD from Jev at scale (R9, opd.py)  ->  episode-level GRPO in the harness with a
#   success - lambda * cost reward (R8, grpo_online.py)  ->  temperature calibration (v1 dev) + tau-hat* (seeds 5000-5149,
#   the registered command, as for Intern-Decision)  ->  frozen checkpoint JevAgentRL-4B on S3.
#
# Seeds (hard rules, enforced by training/seed_guard.py where seeds are drawn):
#   train      >= 10000, never 100000-100499, 200000-200499, 300000-300999 (Stage-2 range)
#   dev/gates  0-999 and 1000-4999 (E2 dev slices, greedy in-training evaluations)
#   calibration 5000-5149 only for the tau-hat* fit of the frozen checkpoint (stage `cal`)
#   never      200000-200499 / 300000-300499 in this phase
#
# One script, two sides.
#   LOCAL (the cloud session; AWS CLI with the SSO session, `env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws`):
#     bash scripts/aws/gpu_round5_rl.sh launch          # p5.4xlarge (1x H100 80 GB) -> g6e.12xlarge -> p4d.24xlarge ->
#                                                       # g6e.2xlarge, first AZ with capacity; Name=jev-rl-round5,
#                                                       # Project=jev; shutdown behaviour stop; user-data: 30 h power-off
#                                                       # guard + 45-min idle stop (no jevharness / jevbench process)
#     bash scripts/aws/gpu_round5_rl.sh push            # git bundle of this branch + this script -> s3://$BUCKET/code/
#     bash scripts/aws/gpu_round5_rl.sh send "STAGES"   # SSM: run the box side detached (setsid nohup) with STAGES
#     bash scripts/aws/gpu_round5_rl.sh status          # the box's status.txt from S3
#     bash scripts/aws/gpu_round5_rl.sh stop            # stop the jev-rl-round5 instance (never touches any other box)
#   BOX (as ubuntu, under /opt/jev; called by `send`):
#     bash gpu_round5_rl.sh box "setup fetch sft eval_sft ..."
# Every stage logs to runs/gpu_round5/<stage>.log, appends a line to runs/gpu_round5/status.txt and syncs
# runs/gpu_round5 -> s3://$BUCKET/gpu_round5/ (small artefacts) and checkpoints/r5/* -> s3://$BUCKET/checkpoints/round5/.
# API keys come from SSM Parameter Store into the box process environment only (never on disk, never printed).
set -uo pipefail
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-<worktree>}
TAG=jev-rl-round5
BUNDLE=jev-rl-round5.bundle
S3OUT="$B/gpu_round5"
S3CK="$B/checkpoints/round5"

# =============================================================================================== local side
awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws --region "$REGION" "$@"; }

instance_id() {  # the running/pending/stopped jev-rl-round5 box (exactly one expected)
  awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text
}

if [[ "$MODE" == launch ]]; then
  existing=$(instance_id)
  if [[ -n "$existing" ]]; then echo "already exists: $existing" >&2; exit 1; fi
  TYPES=${TYPES:-"p5.4xlarge g6e.12xlarge p4d.24xlarge g6e.2xlarge"}
  # us-east-1 a b c d e f (default subnets of the project VPC)
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
shutdown -h +$GUARD_MIN "jev-rl-round5: 30 h guard"
mkdir -p /opt/jev && chown ubuntu:ubuntu /opt/jev
cat > /usr/local/bin/jev-idle-stop <<'IDLE'
#!/bin/bash
# stop the box after $IDLE_MIN min without a jevharness / jevbench / setup process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'jevharness|jevbench|setup_gpu|gpu_round5_rl|opt.jev.r5_|hf download|uv pip' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-rl-round5: idle $IDLE_MIN min"; fi
IDLE
chmod +x /usr/local/bin/jev-idle-stop
echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' > /etc/cron.d/jev-idle-stop
EOF
  for t in $TYPES; do
    for s in $SUBNETS; do
      if id=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$t" --subnet-id "$s" \
            --security-group-ids "$SG" --iam-instance-profile "Name=$PROFILE" \
            --instance-initiated-shutdown-behavior stop --user-data "file://$UD" \
            --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","Throughput":500,"Iops":6000,"DeleteOnTermination":true}}]' \
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
  awsl s3 cp "$ROOT/scripts/aws/gpu_round5_rl.sh" "$B/code/gpu_round5_rl.sh" --only-show-errors
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
    # a private copy per invocation: bash reads a running script incrementally, so a later send must never overwrite it
    echo 'f=/opt/jev/r5_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/gpu_round5_rl.sh \$f --only-show-errors --region $REGION"
    printf 'cd /opt/jev && setsid nohup env %s bash "$f" box %q >> /opt/jev/r5_box.log 2>&1 < /dev/null &\n' \
      "${*:-R5=1}" "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/r5_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$iid" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/r5_driver.sh /tmp/r5_driver.sh --only-show-errors && bash /tmp/r5_driver.sh'\"]" \
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

if [[ "$MODE" != box ]]; then
  sed -n '2,30p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
W=/opt/jev
cd "$W" || exit 1
MYB="$W/bundle.$$.bundle"   # private copy: concurrent invocations never read a bundle that is being rewritten
aws s3 cp "$B/code/$BUNDLE" "$MYB" --only-show-errors --region "$REGION"
if [[ ! -d "$W/JevRL" ]]; then
  git clone -q -b "$BRANCH" "$MYB" "$W/JevRL"
fi
cd "$W/JevRL" || exit 1
git fetch -q "$MYB" "$BRANCH" && git reset -q --hard FETCH_HEAD
rm -f "$MYB"
OUT=runs/gpu_round5
EVAL=$OUT/eval
CK=checkpoints/r5
mkdir -p "$OUT" "$EVAL" "$CK" runs/train_logs
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$OUT/status.txt"
  aws s3 sync "$OUT" "$S3OUT" --only-show-errors --region "$REGION" --exclude '*.jsonl.gz' --exclude 'labelled/*'
}
sync_ck() {  # sync_ck NAME: adapter + metrics (no optimizer state) -> s3://.../checkpoints/round5/NAME
  aws s3 sync "$CK/$1" "$S3CK/$1" --only-show-errors --region "$REGION" --exclude '*optimizer*' \
    --exclude '*/trainer_state.pt' --exclude 'checkpoint-*' --exclude 'ckpt-*'
}
for v in GEMINI_API_KEY GEMINI_API_KEYS OPENROUTER_API_KEY; do
  val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value \
        --output text 2>/dev/null) || val=""
  if [[ -n "$val" && "$val" != None ]]; then export "$v=$val"; fi
done
unset val
if [[ -x .venv/bin/python ]]; then . .venv/bin/activate; fi
export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
st "box start $(git log --oneline -1) stages=[$STAGES] gpu=${GPU} x${NGPU}"

MODEL=Qwen/Qwen3-4B
INTERN='{"type": "intern_decision", "model": "internlm/Intern-Decision-4B"}'
D0=data/train_v0
D1=data/train_v1
GATE_FILES=("$D0/dev.jsonl" "$D0/dev_when2call.jsonl.gz")
V1_FILES=("$D1/dev.jsonl.gz" "$D1/dev_recovery.jsonl.gz" "$D1/dev_knockout.jsonl.gz")
EVAL_BATCH=${EVAL_BATCH:-16}

evaluate() {  # evaluate ADAPTER TAG: gate files (+ flip rate) and the v1 in-harness files
  local adapter=$1 tag=$2
  if [[ ! -e "$EVAL/eval_${tag}_gate.json" ]]; then
    python -m jevharness.training.eval_decisions --model "$MODEL" --adapter "$adapter" --dev "${GATE_FILES[@]}" \
      --n-perm 1 --max-len 4096 --batch-size "$EVAL_BATCH" --out "$EVAL/eval_${tag}_gate.json"
  fi
  if [[ ! -e "$EVAL/eval_${tag}_v1.json" ]]; then
    python -m jevharness.training.eval_decisions --model "$MODEL" --adapter "$adapter" --dev "${V1_FILES[@]}" \
      --n-perm 0 --max-len 4096 --batch-size "$EVAL_BATCH" --out "$EVAL/eval_${tag}_v1.json"
  fi
}
gonogo_of() {  # gonogo_of TAG [BASELINE_TAG]
  local tag=$1 base=${2:-}
  local args=(--eval "$EVAL/eval_${tag}_gate.json" "$EVAL/eval_${tag}_v1.json" --jev-v1 "$D1/dev_teacher.summary.json"
              --out "$EVAL/gonogo_${tag}.json")
  if [[ -e "$EVAL/gate_intern4b_v1dev.json" ]]; then args+=(--intern-v1 "$EVAL/gate_intern4b_v1dev.json"); fi
  if [[ -n "$base" ]]; then args+=(--baseline "$EVAL/eval_${base}_gate.json" "$EVAL/eval_${base}_v1.json"); fi
  python -m jevharness.training.data_v1 gonogo "${args[@]}" > /dev/null
}

stage_setup() {
  RUN_TESTS=0 CHECK_HF=0 bash scripts/train/setup_gpu.sh
  . .venv/bin/activate
  for m in Qwen/Qwen3-4B internlm/Intern-Decision-4B; do
    python - "$m" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))
EOF
  done
}

stage_fetch() {  # round 3's R2 (SFT on v0): the "SFT v0" row of the ablation
  mkdir -p checkpoints/v0/r2
  aws s3 sync "$B/checkpoints/v0/r2" checkpoints/v0/r2 --only-show-errors --region "$REGION" \
    --exclude 'checkpoint-*' --exclude 'train_mix.jsonl'
}

stage_sft() {  # R3-v1: Qwen3-4B LoRA r32, soft CE + 0.5 Brier + permutation, v1 train + 0.45 When2Call replay
  # token-budget micro-batches (length-sorted, ~3k padded tokens) and no activation checkpointing: ~2x faster on one
  # L40S than the round-4 preset (measured: 4.1k vs 2.4k tokens/s at equal padding; far less padding)
  HW=l40s BATCH=${SFT_BATCH:-2} EFF_BATCH=64 EPOCHS=${SFT_EPOCHS:-2} MAX_LEN=4096 OUT="$CK/sft_v1" \
    SFT_EXTRA="${SFT_EXTRA:---max-batch-tokens 3072 --eff-batch 64 --no-gradient-checkpointing}" \
    EVAL_DIR="$EVAL" STAGES=sft bash scripts/train/12_sft_v1.sh
  local rc=$?
  cp -f "$CK/sft_v1/trainer_state.json" "$OUT/sft_v1_trainer_state.json" 2>/dev/null
  sync_ck sft_v1
  return $rc
}

stage_eval_sft() {  # decision-level gates of SFT v0 (R2) and SFT v1 + Intern-Decision-4B on v1 dev; go / no-go
  evaluate "$CK/sft_v1" sft_v1
  if [[ -e checkpoints/v0/r2/adapter_config.json ]]; then evaluate checkpoints/v0/r2 sft_v0; fi
  if [[ ! -e "$EVAL/gate_intern4b_v1dev.json" ]]; then
    for f in dev dev_recovery; do gzip -dc "$D1/$f.jsonl.gz" > "$EVAL/v1_$f.jsonl"; done
    python scripts/gate_s1_dev.py --s1 "$INTERN" --dev "$EVAL/v1_dev.jsonl" --out "$EVAL/gate_intern4b_v1dev.json"
    python scripts/gate_s1_dev.py --s1 "$INTERN" --dev "$EVAL/v1_dev_recovery.jsonl" \
      --out "$EVAL/gate_intern4b_v1recovery.json"
    rm -f "$EVAL/v1_dev.jsonl" "$EVAL/v1_dev_recovery.jsonl"
  fi
  gonogo_of sft_v1 sft_v0
  if [[ -e "$EVAL/eval_sft_v0_v1.json" ]]; then gonogo_of sft_v0; fi
}

# ----------------------------------------------------------------------------------------------- on-policy stages
latest_ckpt() { ls -d "$1"/ckpt-* 2>/dev/null | sort | tail -1; }

stage_opd() {  # R9: on-policy distillation from Jev at scale, from SFT v1 (configs/train/opd_round5.yaml)
  python -m jevharness.training.opd --config configs/train/opd_round5.yaml --resume auto ${OPD_ARGS:-}
  local rc=$?
  cp -f "$CK/opd/metrics.jsonl" "$OUT/opd_metrics.jsonl" 2>/dev/null
  cp -f "$CK/opd/summary.json" "$OUT/opd_summary.json" 2>/dev/null
  cp -f "$CK/opd/rollouts.jsonl" "$OUT/opd_rollouts.jsonl" 2>/dev/null
  sync_ck opd
  aws s3 sync "$CK/opd/labelled" "$S3CK/opd/labelled" --only-show-errors --region "$REGION"
  return $rc
}

stage_poweroff() {  # the last stage of a chain invocation: stop the host (EBS persists; outputs are on S3)
  aws s3 sync "$OUT" "$S3OUT" --only-show-errors --region "$REGION" --exclude '*.jsonl.gz' --exclude 'labelled/*'
  sudo shutdown -h +3 "jev-rl-round5: chain finished"
}

stage_await_eval_sft() {  # chain the next invocation behind a running `sft eval_sft` one (no idle gap, no overlap)
  until grep -q "stage eval_sft exit" "$OUT/status.txt" 2>/dev/null; do sleep 60; done
}

stage_grpo_validity() {  # A7.10 (c): the sampler / replay / normalisation / KL tests on the training commit
  python -m pytest -p no:warnings tests/test_grpo_validity.py > "$OUT/grpo_validity.txt" 2>&1
  local rc=$?
  echo "pytest_exit=$rc" >> "$OUT/grpo_validity.txt"
  return $rc
}

stage_chain() {  # the mechanical keep / drop / freeze decision (scripts/train/r5_chain.py; TRAINING_WHITELIST.md)
  python scripts/train/r5_chain.py --eval-dir "$EVAL" --e2dev "$OUT/e2dev" --grpo-validity "$OUT/grpo_validity.txt" \
    --out "$OUT/chain_decision.json"
}

stage_grpo() {  # R8: episode-level GRPO in the harness from the last KEPT stage (r5_chain.py: OPD if kept, else SFT v1)
  local resume init
  resume=$(latest_ckpt "$CK/grpo")
  stage_chain > /dev/null || return 1
  init=$(python -c "import json; print(json.load(open('$OUT/chain_decision.json'))['kept'][-1])") || return 1
  echo "GRPO init = $init (chain_decision.json)"
  python -m jevharness.training.grpo_online --config configs/train/grpo_online_round5.yaml --init "$CK/$init" \
    ${resume:+--resume "$resume"} ${GRPO_ARGS:-}
  local rc=$?
  cp -f "$CK/grpo/metrics.jsonl" "$OUT/grpo_metrics.jsonl" 2>/dev/null
  cp -f "$CK/grpo/rollouts.jsonl" "$OUT/grpo_rollouts.jsonl" 2>/dev/null
  sync_ck grpo
  return $rc
}

candidate() {  # candidate STAGE ITER TAG: the stage's checkpoint at iteration ITER as its own adapter dir (a continuation)
  local src
  src=$(printf '%s/%s/ckpt-%05d' "$CK" "$1" "$2")
  [[ -e "$src/adapter_model.safetensors" ]] || { echo "missing $src"; return 1; }
  rm -rf "${CK:?}/$3" && mkdir -p "$CK/$3"
  cp -f "$src"/adapter_config.json "$src"/adapter_model.safetensors "$src"/decision_config.json "$CK/$3/"
  cp -f "$src"/tokenizer* "$src"/chat_template.jinja "$src"/special_tokens_map.json "$CK/$3/" 2>/dev/null
  sync_ck "$3"
}
# continuations (TRAINING_MANIFEST.md): OPD at iterations 12 and 24 (final), GRPO at 15 and 30 (final); the gonogo
# baseline is the stage's init (SFT v1 for OPD, the chain's GRPO init for GRPO)
stage_eval_opd() {
  candidate opd 12 opd_i12
  for t in opd_i12 opd; do evaluate "$CK/$t" "$t"; gonogo_of "$t" sft_v1; done
}
stage_eval_grpo() {
  local init
  init=$(python -c "import json; print(json.load(open('$OUT/chain_decision.json'))['grpo_init'])")
  candidate grpo 15 grpo_i15
  for t in grpo_i15 grpo; do evaluate "$CK/$t" "$t"; gonogo_of "$t" "$init"; done
}

# ----------------------------------------------------------------------------------------------- E2 dev slices
E2D=runs/gpu_round5_e2dev_s1100   # development seeds 1100-1149 (configs/experiments/jevgym_dev_rl.yaml)
DEVCFG=configs/experiments/jevgym_dev_rl.yaml
PORT=8081
SERVER_PID=""
serve_start() {  # serve_start ADAPTER NAME [CALIBRATION]: the checkpoint behind the Jev wire API on :$PORT
  local adapter=$1 name=$2 cal=${3:-}
  local args=(--model "$MODEL" --adapter "$adapter" --name "$name" --port "$PORT")
  if [[ -n "$cal" ]]; then args+=(--calibration "$cal"); fi
  python -m jevharness.training.serve_jev "${args[@]}" > "$OUT/serve_$name.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died"; return 1; fi
    sleep 5
  done
  return 1
}
serve_stop() {
  if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""; fi
}
e2dev_run() {  # e2dev_run NAME "PREFIXES" [SET...]: a subset of jevgym_dev_rl.yaml's arms as its own run directory
  local name=$1 prefixes=$2
  shift 2
  mkdir -p "$E2D" "$OUT/e2dev"
  # shellcheck disable=SC2086  # PREFIXES is a whitespace-separated list
  python scripts/train/r5_select_arms.py "$DEVCFG" "$E2D/cfg_$name.yaml" --prefix $prefixes \
    --set "name=$name" "workers=${E2DEV_WORKERS:-8}" "$@" || return 1
  jevbench run "$E2D/cfg_$name.yaml" --out "$E2D" ${E2DEV_MAX:+--max-tasks "$E2DEV_MAX"}
  local rc=$?
  jevbench report "$E2D/$name" > "$E2D/$name/report.md" 2>/dev/null
  cp -f "$E2D/$name/report.md" "$OUT/e2dev/${name}_report.md"
  python scripts/train/r5_summary.py "$E2D/$name" --out "$OUT/e2dev/${name}_summary" > /dev/null
  aws s3 sync "$E2D/$name" "$S3OUT/e2dev_runs_s1100/$name" --only-show-errors --region "$REGION"
  return $rc
}
e2dev_ckpt() {  # e2dev_ckpt TAG ADAPTER [CALIBRATION]: serve one checkpoint and run the ours_* arms
  local tag=$1 adapter=$2 cal=${3:-} name="jevagentrl-$1"
  serve_start "$adapter" "$name" "$cal" || { serve_stop; return 1; }
  e2dev_run "e2dev_rl_$tag" "${E2DEV_ARMS:-ours_shadow ours_ts_replace ours_ts_params_replace ours_ts_assist_k3 ours_stop_guard ours_ts_guard}" \
    "s1.name=$name" "s1.model=$name"
  local rc=$?
  curl -sf "http://127.0.0.1:$PORT/v1/models" > "$OUT/e2dev/serve_${tag}_models.json" 2>/dev/null
  serve_stop
  return $rc
}
stage_e2dev_refs_api() { e2dev_run e2dev_rl_refs_api "llm_only jev_"; }    # no GPU: runs next to training
stage_e2dev_refs_intern() { E2DEV_WORKERS=4 e2dev_run e2dev_rl_refs_intern "intern_"; }   # Intern in-process (GPU)
stage_e2dev_sft_v0() { e2dev_ckpt sft_v0 checkpoints/v0/r2; }
stage_e2dev_sft_v1() { e2dev_ckpt sft_v1 "$CK/sft_v1"; }
stage_e2dev_opd() { e2dev_ckpt opd_i12 "$CK/opd_i12"; e2dev_ckpt opd "$CK/opd"; }
stage_e2dev_grpo() { e2dev_ckpt grpo_i15 "$CK/grpo_i15"; e2dev_ckpt grpo "$CK/grpo"; }
stage_e2dev_final() {  # the frozen checkpoint, calibrated, every ours_* arm
  E2DEV_ARMS=ours_ e2dev_ckpt final "$CK/final" "$CK/final/calibration.json"
}
stage_e2dev_summary() {
  python scripts/train/r5_summary.py "$E2D"/e2dev_rl_* --out "$OUT/e2dev/summary_all"
}

# ----------------------------------------------------------------------------------------------- freeze + calibrate
stage_freeze() {  # the chain's frozen stage (chain_decision.json) -> checkpoints/r5/final; temperatures on v1 dev
  stage_chain || return 1
  local from
  from=$(python -c "import json; print(json.load(open('$OUT/chain_decision.json'))['frozen'])") || return 1
  if [[ -n "${FINAL_FROM:-}" && "$FINAL_FROM" != "$from" ]]; then
    echo "FINAL_FROM=$FINAL_FROM disagrees with the chain ($from)"
    return 1
  fi
  FINAL_FROM=$from
  local src="$CK/$FINAL_FROM"
  rm -rf "$CK/final" && mkdir -p "$CK/final"
  cp -f "$src"/adapter_config.json "$src"/adapter_model.safetensors "$src"/decision_config.json "$CK/final/"
  cp -f "$src"/tokenizer* "$src"/chat_template.jinja "$src"/special_tokens_map.json "$CK/final/" 2>/dev/null
  python -m jevharness.training.calibrate --model "$MODEL" --adapter "$CK/final" --dev "$D1/dev.jsonl.gz" \
    --out "$CK/final/calibration.json" --min-bucket 200 || return 1
  evaluate "$CK/final" final
  python -m jevharness.training.eval_decisions --model "$MODEL" --adapter "$CK/final" --calibration \
    "$CK/final/calibration.json" --dev "${V1_FILES[@]}" --n-perm 0 --batch-size "$EVAL_BATCH" \
    --out "$EVAL/eval_final_calibrated_v1.json"
  gonogo_of final sft_v1
  # the freeze record (A7.10 (a)): checkpoint, adapter, tokenizer, temperature file, serving commit, freeze time
  { echo "name JevAgentRL-4B"; echo "base $MODEL (HF snapshot $(basename "$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/* | head -1)"))"
    echo "from $FINAL_FROM ($src)"; echo "serving / training commit $(git rev-parse HEAD) (training/serve_jev.py)"
    echo "chain decision $(sha256sum "$OUT/chain_decision.json" | cut -c1-64)"
    echo "frozen $(date -u +%FT%TZ)"
    (cd "$CK/final" && sha256sum adapter_model.safetensors adapter_config.json calibration.json tokenizer.json \
      tokenizer_config.json chat_template.jinja decision_config.json 2>/dev/null); } > "$CK/final/FROZEN.txt"
  cp -f "$CK/final/FROZEN.txt" "$OUT/final_FROZEN.txt"
  cp -f "$CK/final/calibration.json" "$OUT/final_calibration.json"
  aws s3 sync "$CK/final" "$S3CK/jevagentrl-4b" --only-show-errors --region "$REGION"
}

stage_base_control() {  # A7.13 (i): untrained Qwen3-4B, same prompt / readout, temperatures fitted on v1 dev as for ours
  mkdir -p "$CK/base"
  python -m jevharness.training.calibrate --model "$MODEL" --dev "$D1/dev.jsonl.gz" --out "$CK/base/calibration.json" \
    --min-bucket 200 || return 1
  cp -f "$CK/base/calibration.json" "$OUT/base_calibration.json"
  if [[ ! -e "$EVAL/eval_base_gate.json" ]]; then
    python -m jevharness.training.eval_decisions --model "$MODEL" --dev "${GATE_FILES[@]}" --n-perm 1 \
      --batch-size "$EVAL_BATCH" --out "$EVAL/eval_base_gate.json"
  fi
  if [[ ! -e "$EVAL/eval_base_v1.json" ]]; then
    python -m jevharness.training.eval_decisions --model "$MODEL" --dev "${V1_FILES[@]}" --n-perm 0 \
      --batch-size "$EVAL_BATCH" --out "$EVAL/eval_base_v1.json"
  fi
}

stage_hashes() {  # sha256 of every stage checkpoint kept for the attribution controls (A7.13 (ii))
  {
    for d in sft_v1 opd_i12 opd grpo_i15 grpo final base; do
      if [[ -d "$CK/$d" ]]; then sha256sum "$CK/$d"/adapter_model.safetensors "$CK/$d"/adapter_config.json \
        "$CK/$d"/calibration.json 2>/dev/null; fi
    done
    sha256sum checkpoints/v0/r2/adapter_model.safetensors 2>/dev/null
  } > "$OUT/checkpoint_sha256.txt"
}

stage_cal() {  # tau-hat* of the frozen checkpoint: calibration shadow arms on seeds 5000-5149, the registered fit
  serve_start "$CK/final" jevagentrl-4b "$CK/final/calibration.json" || { serve_stop; return 1; }
  jevbench run configs/experiments/jevgym_shadow_cal_rl.yaml --out runs
  local rc=$?
  serve_stop
  local d=runs/e2a_jevgym_shadow_cal_rl
  mkdir -p "$OUT/cal"
  jevbench report "$d" > "$d/report.md" 2>/dev/null
  cp -f "$d/report.md" "$OUT/cal/shadow_cal_rl_report.md"
  for arm in rl_shadow_cal rl_shadow_cal_s2lite; do for dp in tool_select tool_params; do
    # the registered fitting procedure (A6.5 / A6.11 (c)): the same command as the Jev and Intern fits
    python -m jevharness.bench.threshold_theory "$d" --arm $arm --dp $dp --calibration-arm --task-value 1.0 \
      --lam-per-hour 0 --v-fallback W --out "$OUT/cal/tauhat_${arm}_${dp}.json" > "$OUT/cal/tauhat_${arm}_${dp}.log" 2>&1
  done; done
  python scripts/train/r5_summary.py "$d" --out "$OUT/cal/shadow_cal_rl_summary" > /dev/null
  aws s3 sync "$d" "$B/e2a_jevgym_shadow_cal_rl" --only-show-errors --region "$REGION"
  return $rc
}

stage_latency() {  # System-1 latency per decision, same protocol as round 4's Intern measurement (A5.1)
  local spec='{"type": "jev", "name": "jevagentrl-4b", "base_url": "http://127.0.0.1:8081", "model": "jevagentrl-4b", "api_key": ""}'
  local args=(--concurrency 1 4 --n 200 --state-tokens 1500 --questions 3)
  serve_start "$CK/final" jevagentrl-4b "$CK/final/calibration.json" || { serve_stop; return 1; }
  python scripts/measure_latency.py --s1 "$spec" "${args[@]}" --gpu-hour-usd "${GPU_HOUR_USD:-3.004}" \
    > "$OUT/latency_jevagentrl4b.txt" 2>&1
  serve_stop
  python scripts/measure_latency.py --s1 '{"type": "intern_decision", "model": "internlm/Intern-Decision-4B", "device": "auto"}' \
    "${args[@]}" --gpu-hour-usd "${GPU_HOUR_USD:-3.004}" > "$OUT/latency_intern4b.txt" 2>&1
  python scripts/measure_latency.py --s1 '{"type": "jev", "provider": "openrouter"}' --concurrency 1 4 --n 100 \
    --state-tokens 1500 --questions 3 > "$OUT/latency_jev.txt" 2>&1
}

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
done
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv > "$OUT/nvidia_smi_last.txt" 2>&1
st "box done [$STAGES]"
