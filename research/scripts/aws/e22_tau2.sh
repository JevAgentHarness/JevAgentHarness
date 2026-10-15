#!/usr/bin/env bash
# Exploratory row E22 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E22: the training recipe on tau^2-bench";
# descriptive, outside every Holm table). One NEW GPU host launched by this driver from E3's clone image
# (tag Name=jev-e22-tau2), SSM only: E14 arm A's 0.8B scoring head continued on the tau^2 train-task states harvested
# locally (records from the bucket, sha256-checked), the threshold fit on the dev tasks' questions, and the final
# harness against LLM-only on tau^2-bench's official test tasks (retail 40, airline 20; k trials). The evaluation
# stages hold the Gemini key in their processes only (from SSM; never printed, never on disk).
#
#   LOCAL:  bash scripts/aws/e22_tau2.sh launch [itype ...]   # one new host (refuses if a jev-e22-tau2 host exists)
#           E22_IID=i-... BEFORE_USD=x bash scripts/aws/e22_tau2.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages (in the order sent; a failed stage stops the chain, `final` always runs when sent):
#     rearm   idle stop (IDLE_MIN) + power-off guard (GUARD_MIN from the first boot), host record
#     setup   the committed branch in /opt/jev/e22; Unsloth's venv; the tau^2 venv (tau2-bench 5bfa7e3 + jevharness)
#     data    the records (sha256 against the commit), arm A's checkpoint (head sha256 recorded)
#     train   scripts/train/e22_clef.py (E14's recipe continued from arm A, max length 8,192); final -> S3
#     replay  the trained model on :8090 answers every labelled dev question (explore_e22_tau2.py replay)
#     fit     tau-hat per decision point (threshold_theory --calibration-arm ...) -> configs/explore/e22_tau2/final.yaml
#     smoke   e22_final on two dev tasks per domain (one trial; never reported) -> the evaluation gate (eval_plan.json)
#     eval    per domain, the planned arms concurrently on the test tasks (run-arm: pause / resume, cap, one re-run)
#     final   stop the server, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e22_tau2/box on the box -> s3://$BUCKET/e22_tau2/box; runs -> s3://$BUCKET/e22_tau2/runs.
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E22_IID=${E22_IID:-}
# every other project host: never touched by this driver
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e22-tau2
BUNDLE=e22.bundle
S3OUT="$B/e22_tau2"
CAP_USD=${CAP_USD:-200}
BEFORE_USD=${BEFORE_USD:-0}
CAP_H=${CAP_H:-10}
GUARD_MIN=${GUARD_MIN:-590}          # 10 host-hours cap (registration (j)): the guard 9 h 50 min after the first boot
IDLE_MIN=${IDLE_MIN:-30}
POWEROFF=${POWEROFF:-1}
PORT=8090
NAME=e22-tau2-0.8b
TAU2_COMMIT=5bfa7e3

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E22_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E22_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E22_IID "* ]] && { echo "refused: $E22_IID is not the E22 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E22_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E22_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e22-tau2: power-off guard (10 host-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g5.2xlarge g6e.2xlarge)
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/e22_tau2.sh" > "$tmp/e22_tau2.sh"   # the committed driver
  awsl s3 cp "$tmp/e22_tau2.sh" "$B/code/e22_tau2.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e22_tau2.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e22_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e22_tau2.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q BEFORE_USD=%q CAP_H=%q GUARD_MIN=%q IDLE_MIN=%q POWEROFF=%q E22_IID=%q KEEP_SERVER=%q PROBE_WAIT_MIN=%q NO_SECOND=%q\n' \
      "$CAP_USD" "$BEFORE_USD" "$CAP_H" "$GUARD_MIN" "$IDLE_MIN" "$POWEROFF" "$E22_IID" "${KEEP_SERVER:-0}" "${PROBE_WAIT_MIN:-45}" "${NO_SECOND:-0}"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e22_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e22_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E22_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e22_driver.sh /tmp/e22_driver.sh --only-show-errors && bash /tmp/e22_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E22_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E22_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,24p' "$0"
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
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E22_IID" && "$MY_IID" != "$E22_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E22 host ${E22_IID:-?}"; exit 1
fi
R=$W/e22
A=$R/results/explore/e22_tau2/box
STATUS_DIR=$W/e22_status
EV=$W/e14venv                       # Unsloth 2026.10.2 (training, the clef server)
TV=$W/t2venv                        # tau2-bench 5bfa7e3 + jevharness (harness runs, replay, fit)
T2=$W/tau2-bench
D=$W/e22_data
CKA=$W/ck_e14/A/final               # E14 arm A's merged checkpoint (the start; never modified)
CKE=$W/ck_e22                       # E22's checkpoint
RUNS=$W/e22_runs
EX=scripts/explore_e22_tau2.py
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
export E22_SCRATCH=$RUNS LITELLM_LOCAL_MODEL_COST_MAP=True
mkdir -p "$STATUS_DIR" "$RUNS"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  aws s3 cp "$STATUS_DIR/status.txt" "$S3OUT/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION"
  [[ -d "$RUNS" ]] && aws s3 sync "$RUNS" "$S3OUT/runs" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  true
}
EPY() { PYTHONPATH="$R/src" "$EV/bin/python" "$@"; }
TPY() { PYTHONPATH="$R/src" TAU2_DATA_DIR="$T2/data" "$TV/bin/python" "$@"; }
hours_used() {  # host hours since the first boot recorded by rearm
  local b; b=$(cat "$W/e22_first_boot" 2>/dev/null || date -d "$(uptime -s)" +%s)
  python3 -c "import time; print(round((time.time() - $b) / 3600, 3))"
}
keys() {  # the direct Gemini API key from SSM into this stage's environment only (never on disk, never printed)
  local val
  val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/GEMINI_API_KEY" --with-decryption \
        --query Parameter.Value --output text 2>/dev/null) || val=""
  [[ -n "$val" && "$val" != None ]] || { echo "GEMINI_API_KEY missing in SSM"; return 1; }
  export GEMINI_API_KEY="$val"
  unset val
}
leak_check() {  # no key value in any record or status file
  local n=0
  if [[ ${#GEMINI_API_KEY} -ge 16 ]]; then
    n=$(grep -rlF -- "$GEMINI_API_KEY" "$RUNS" "$A" "$STATUS_DIR" 2>/dev/null | wc -l)
  fi
  echo "key-leak files: $n"
  [[ "$n" == 0 ]]
}
stop_port() {
  local f="$W/e22_serve_$1.pid" p
  if [[ -f "$f" ]]; then
    p=$(cat "$f"); kill "$p" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -9 "$p" 2>/dev/null
  fi
  rm -f "$f"
}
serve() {  # the trained checkpoint behind serve_jev --backend clef on :8090
  stop_port "$PORT"
  [[ -s "$CKE/final/joint_head.safetensors" ]] || { echo "no E22 checkpoint at $CKE/final"; return 1; }
  echo "$(date -u +%FT%TZ) serve_jev --backend clef --model $CKE/final --name $NAME --port $PORT" >> "$A/logs/serve.cmd"
  nohup env PYTHONPATH="$R/src" "$EV/bin/python" -m jevharness.training.serve_jev --backend clef --model "$CKE/final" \
    --name "$NAME" --port "$PORT" > "$A/logs/serve_$(date +%s).out" 2>&1 &
  echo $! > "$W/e22_serve_$PORT.pid"
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/health" > "$A/logs/serve_health.json"; then
      curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$NAME\"" || { echo "served model id is not $NAME"; return 1; }
      printf '{"server_up": "%s", "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" "$(imds instance-type)" >> "$A/serve_windows.jsonl"
      return 0
    fi
    kill -0 "$(cat "$W/e22_serve_$PORT.pid")" 2>/dev/null || { echo "server died"; return 1; }
    sleep 5
  done
  echo "server did not come up"; return 1
}

stage_rearm() {
  [[ -s "$W/e22_first_boot" ]] || date -d "$(uptime -s)" +%s > "$W/e22_first_boot"
  local used_min left
  used_min=$(( ( $(date +%s) - $(cat "$W/e22_first_boot") ) / 60 ))
  left=$(( GUARD_MIN - used_min ))
  (( left > 0 )) || { echo "guard already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$left" "jev-e22-tau2: power-off guard (GPU cap)"
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E22 chain / runner / trainer / install process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e22_[0-9]|e22_clef|explore_e22|jevharness.bench.suites.tau2|uv pip|pip install|git clone|snapshot_download' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e22-tau2: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $left min left, idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU; $(df -BG / | awk 'NR==2{print $4}') free; used $(hours_used) h"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}
fetch_code() {
  local myb="$W/e22.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A/logs" "$A/cal" "$A/train"
  st "code at $(git log --oneline -1 | cut -c1-72)"
}
stage_code() { fetch_code; }
stage_setup() {
  fetch_code || return 1
  cd "$R" || return 1
  grep -q 'E22: the training recipe on τ²-bench' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E22"; return 1; }
  command -v uv > /dev/null || bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  if [[ ! -x "$EV/bin/python" ]] || ! "$EV/bin/python" -c "import unsloth" > /dev/null 2>&1; then
    uv venv -q -p 3.11 "$EV" || return 1
    timeout 45m uv pip install -q --python "$EV" "unsloth==2026.10.2" fastapi uvicorn httpx pydantic pyyaml tqdm \
      python-dotenv openai > "$A/logs/e14venv_install.log" 2>&1 || { tail -20 "$A/logs/e14venv_install.log"; return 1; }
    timeout 20m uv pip install -q --python "$EV" flash-linear-attention >> "$A/logs/e14venv_install.log" 2>&1 \
      || echo "flash-linear-attention not installed (the torch path is used)" | tee -a "$A/logs/e14venv_install.log"
  fi
  EPY -c "import unsloth, torch, transformers, peft; from unsloth import FastDecisionModel, DecisionTrainer; print('unsloth', unsloth.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'cuda', torch.cuda.is_available())" \
    > "$A/env_e14venv.txt" 2>&1 || { tail -20 "$A/env_e14venv.txt"; return 1; }
  # the tau^2 venv: tau2-bench at the pinned commit (its runtime deps as docs/TAU2.md), jevharness from this checkout
  if [[ ! -d "$T2/.git" ]]; then
    timeout 10m git clone -q https://github.com/sierra-research/tau2-bench "$T2" || return 1
  fi
  git -C "$T2" fetch -q origin 2>/dev/null; git -C "$T2" checkout -q "$TAU2_COMMIT" || return 1
  if [[ ! -x "$TV/bin/python" ]] || ! TPY -c "import tau2, jevharness.bench.suites.tau2" > /dev/null 2>&1; then
    uv venv -q -p 3.11 "$TV" || return 1
    timeout 20m uv pip install -q --python "$TV" -e "$R" rich "tabulate>=0.9.0" "fastapi>=0.115.11" "uvicorn>=0.34.0" \
      "pandas>=2.2.3" "psutil>=7.0.0" "loguru>=0.7.3" "docstring-parser>=0.16" "litellm>=1.80.15,<1.82.7" \
      "tenacity>=9.0.0" "deepdiff>=8.4.2" "addict>=2.4.0" "toml>=0.10.2" "typer>=0.12.5" "requests>=2.31.0" websockets \
      pip > "$A/logs/t2venv_install.log" 2>&1 || { tail -20 "$A/logs/t2venv_install.log"; return 1; }
    "$TV/bin/python" -m pip install -q --no-deps --ignore-requires-python -e "$T2" >> "$A/logs/t2venv_install.log" 2>&1 || return 1
  fi
  TPY -c "import tau2, litellm, jevharness, jevharness.bench.suites.tau2 as jt; print(tau2.__file__, jevharness.__file__, jt.HAVE_TAU2)" \
    > "$A/env_t2venv.txt" 2>&1 || { cat "$A/env_t2venv.txt"; return 1; }
  git -C "$T2" rev-parse --short HEAD >> "$A/env_t2venv.txt"
  uv pip freeze --python "$TV" > "$A/t2venv_freeze.txt" 2>/dev/null
  TPY "$EX" split > "$A/logs/split_check.log" 2>&1 || return 1
  git diff --quiet configs/explore/e22_tau2/split.json || { echo "the tau2 split on the box differs from the committed one"; return 1; }
  cat "$A/env_e14venv.txt" "$A/env_t2venv.txt"
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s", "gpu": "%s"}\n' "$MY_IID" \
    "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" \
    "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" > "$A/host.json"
  df -BG / | awk 'NR==2{print "disk free: "$4}'
}
stage_data() {
  cd "$R" || return 1
  mkdir -p "$D" "$CKA"
  aws s3 sync "$S3OUT/data" "$D" --only-show-errors --region "$REGION" || return 1
  TPY - "$D" results/explore/e22_tau2/data_sha256.json <<'EOF' || return 1
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
  aws s3 sync "$B/checkpoints/e14_small/A/final" "$CKA" --only-show-errors --region "$REGION" || return 1
  (cd "$CKA" && sha256sum joint_head.safetensors unsloth_decision_config.json joint_head_config.json) > "$A/arm_a_sha256.txt" || return 1
  cat "$A/arm_a_sha256.txt"
  sync_out
}
stage_train() {
  cd "$R" || return 1
  local nos=""
  [[ "${NO_SECOND:-0}" == 1 ]] && nos="--no-second"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 5 > "$A/train/gpu.csv" 2>/dev/null &
  local smi=$!
  EPY scripts/train/e22_clef.py train --data "$D" --init "$CKA" --out "$CKE" --hours-used "$(hours_used)" --cap-h 9.5 $nos \
    > "$A/train/train.log" 2>&1
  local rc=$?
  kill "$smi" 2>/dev/null
  cp -f "$CKE/train_record.json" "$A/train/train_record.json" 2>/dev/null
  (( rc == 0 )) || { tail -40 "$A/train/train.log"; return 1; }
  (cd "$CKE/final" && sha256sum joint_head.safetensors unsloth_decision_config.json) > "$A/train/final_sha256.txt"
  aws s3 sync "$CKE/final" "$B/checkpoints/e22_tau2/final" --only-show-errors --region "$REGION" || return 1
  sync_out
}
stage_replay() {
  cd "$R" || return 1
  serve || return 1
  TPY "$EX" replay --inp "$D/dev_replay.jsonl.gz" --base-url "http://127.0.0.1:$PORT" --name "$NAME" \
    --out "$A/cal/answers_dev.jsonl.gz" --workers 4 > "$A/cal/replay.log" 2>&1 || { tail -30 "$A/cal/replay.log"; return 1; }
  tail -2 "$A/cal/replay.log"
  sync_out
}
stage_fit() {
  cd "$R" || return 1
  TPY "$EX" fit --inp "$D/dev_replay.jsonl.gz" --answers "$A/cal/answers_dev.jsonl.gz" --out-dir "$A/cal" \
    --profile configs/explore/e22_tau2/final.yaml --base-url "http://127.0.0.1:$PORT" --name "$NAME" \
    > "$A/cal/fit.log" 2>&1 || { tail -30 "$A/cal/fit.log"; return 1; }
  cp -f configs/explore/e22_tau2/final.yaml "$A/cal/final.yaml"
  sha256sum configs/explore/e22_tau2/final.yaml > "$A/cal/final_yaml_sha256.txt"
  date -u +%FT%TZ > "$A/cal/final_yaml_written_utc.txt"
  cat "$A/cal/fit.log"
  sync_out
}
host_spend() {  # this host's logged spend (every E22 run under $RUNS)
  TPY "$EX" spend --json 2>/dev/null | python3 -c "import json,sys; print(round(json.load(sys.stdin)['local_usd'], 4))"
}
write_spend() { printf '{"total_usd": %s, "utc": "%s"}\n' "$(host_spend)" "$(date -u +%FT%TZ)" > "$A/spend_host.json"; }
stage_smoke() {
  cd "$R" || return 1
  keys || return 1
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || serve || return 1
  local always d
  always=$(TPY -c "import yaml; c=yaml.safe_load(open('configs/explore/e22_tau2/final.yaml')); print(int(bool(c['tool_select'].get('always_defer') or c['tool_params'].get('always_defer'))))") || return 1
  for d in retail airline; do
    TPY "$EX" run-arm --arm "$( [[ $always == 1 ]] && echo e22_tau05 || echo e22_final )" --domain "$d" --smoke \
      --out "$RUNS/smoke/$d" --probe-log "$A/api_probe_log.txt" --cap "$CAP_USD" --before "$BEFORE_USD" \
      > "$A/logs/smoke_$d.log" 2>&1 || { tail -30 "$A/logs/smoke_$d.log"; return 1; }
  done
  write_spend
  TPY - "$RUNS/smoke" "$CAP_USD" "$BEFORE_USD" "$always" "$A/eval_plan.json" <<'EOF' || return 1
import json, sys
sys.path.insert(0, "scripts")
from pathlib import Path
import explore_e22_tau2 as X
smoke, cap, before, always, out = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "1", sys.argv[5]
rows = [X.run_spend(p) for p in sorted(smoke.iterdir()) if p.is_dir()]
n = sum(r["simulations"] for r in rows) or 1
per = sum(r["total_usd"] for r in rows) / n
spent = before + sum(r["total_usd"] for r in rows)
plan = None
arms_full = ["llm_only", "e22_tau05"] if always else ["llm_only", "e22_final", "e22_tau05"]
for arms, k in ((arms_full, 4), (arms_full[:2], 4), (arms_full[:2], 2)):
    proj = per * 60 * k * len(arms) * 1.3 + spent
    if proj <= cap:
        plan = {"arms": arms, "trials": k, "projection_usd": round(proj, 2)}
        break
rec = {"smoke_usd_per_sim": round(per, 4), "smoke_sims": n, "spent_before_eval": round(spent, 3), "cap": cap,
       "always_defer": always, "plan": plan}
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps(rec))
sys.exit(0 if plan else 3)
EOF
  leak_check || return 1
  sync_out
}
stage_eval() {
  cd "$R" || return 1
  keys || return 1
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || serve || return 1
  local arms k d a pids
  arms=$(python3 -c "import json; print(' '.join(json.load(open('$A/eval_plan.json'))['plan']['arms']))") || return 1
  k=$(python3 -c "import json; print(json.load(open('$A/eval_plan.json'))['plan']['trials'])") || return 1
  local before_eval
  before_eval=$(python3 -c "print($BEFORE_USD + $(host_spend))")
  for d in retail airline; do
    st "eval $d: arms [$arms], k $k"
    printf '{"domain": "%s", "start_utc": "%s", "t": %s, "arms": "%s", "trials": %s}\n' "$d" "$(date -u +%FT%TZ)" "$(date +%s)" "$arms" "$k" >> "$A/eval_windows.jsonl"
    pids=""
    for a in $arms; do
      TPY "$EX" run-arm --arm "$a" --domain "$d" --trials "$k" --concurrency 6 --out "$RUNS/eval/$d/$a" \
        --probe-log "$A/api_probe_log.txt" --cap "$CAP_USD" --before "$BEFORE_USD" > "$A/logs/eval_${d}_$a.log" 2>&1 &
      pids="$pids $!"
    done
    ( while true; do sleep 300; write_spend; sync_out; done ) & local spid=$!
    for p in $pids; do wait "$p" || st "eval $d: an arm exited non-zero (see logs/eval_${d}_*.log)"; done
    kill "$spid" 2>/dev/null
    printf '{"domain": "%s", "end_utc": "%s", "t": %s}\n' "$d" "$(date -u +%FT%TZ)" "$(date +%s)" >> "$A/eval_windows.jsonl"
    write_spend
    st "eval $d done; host spend $(host_spend) USD (+ $BEFORE_USD before the host)"
    sync_out
    python3 -c "import sys; sys.exit(0 if $BEFORE_USD + $(host_spend) < $CAP_USD else 1)" || { st "spend cap reached"; return 3; }
  done
  curl -sf "http://127.0.0.1:$PORT/health" > "$A/logs/serve_health_end.json"
  leak_check || return 1
  sync_out
}
stage_final() {
  stop_port "$PORT"
  nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  write_spend 2>/dev/null
  printf '{"final_utc": "%s", "instance_id": "%s", "instance_type": "%s", "hours_used": %s}\n' "$(date -u +%FT%TZ)" "$MY_IID" \
    "$(imds instance-type)" "$(hours_used)" > "$A/host_final.json"
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; sudo shutdown -h +2 "jev-e22-tau2: done"; fi
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
if [[ "${KEEP_SERVER:-0}" != 1 ]]; then stop_port "$PORT"; fi
st "box done [$STAGES]"
