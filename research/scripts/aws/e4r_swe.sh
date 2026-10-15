#!/usr/bin/env bash
# E4r (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07, "E4r: replicate and extension of E4"; exploratory, descriptive):
# E4's three eval arms (always_deep, jevagent at E4's registered tau 0.5, always_fast) on mini-SWE-agent 2.4.6, DeepSeek
# V4 Pro (deep) / Flash (fast) and Jev through OpenRouter, on E4's 100 eval instances (replicate) and 200 new held-out
# SWE-bench Verified instances (extension; configs/explore/e4r_swe/). One NEW CPU host of its own (tag Name=jev-e4r-swe,
# SSM only, no inbound). The phases run through E4's own driver (scripts/aws/e4_swe.sh, EXP=e4r): the same caps,
# workers, task-major schedule (seed 732), one re-run of infrastructure errors, local swebench 4.1.0 evaluation,
# OpenRouter spend watchdog and idle stop.
#
#   LOCAL: bash scripts/aws/e4r_swe.sh launch [itype ...]   # one new host (refuses if a jev-e4r-swe host exists)
#          E4R_IID=i-... bash scripts/aws/e4r_swe.sh push | send "STAGES" | status [N] | start | stop | itype | fetch
#   BOX stages (in the order sent; a failed stage stops the chain, `final` always runs when sent):
#     rearm    power-off guard (GUARD_MIN from now), idle stop (30 min without a driver pid or a container)
#     setup    Docker CE (mirror.gcr.io), uv + Python 3.11, the branch from code/e4r.bundle, .venv (.[all] +
#              mini-swe-agent 2.4.6), .venv-eval (swebench 4.1.0), offline adapter + E4/E4r tests
#     update   fast-forward the checkout to the latest code/e4r.bundle (the registration commit)
#     prepull  the instance images of the smoke and eval lists (mirror.gcr.io first; Docker Hub reserve respected)
#     smoke    one smoke instance x 3 arms (development only); the key must be an inference key; its usage now is E4r's
#              start, and the watchdog stops the runner at start + CAP_USD
#     gate     the registered smoke checks and cost projection (explore_e4_swe.py --exp e4r gate -> e4r_gate.json)
#     eval     the 200 extension instances x 3 arms (or the gate's first 100)
#     final    spend (logged + key usage), leak check, sync, power-off (POWEROFF=1)
# Docker needs the group of a new login: send `rearm setup` alone, then the other stages in a second `send`.
# Records: /opt/jev/hard_status/e4r_* and e4.txt -> s3://$BUCKET/e4r/status; runs -> s3://$BUCKET/e4r/runs/<phase>.
# The OpenRouter key is read from SSM Parameter Store (${SSM_PREFIX:?set SSM_PREFIX}/OPENROUTER_API_KEY) into the stage's process only and
# is never printed or written; nothing in this script writes a secret anywhere.
set -uo pipefail
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E4R_IID=${E4R_IID:-}
# every other project host (E3's, E5's, E6a's / H5's CPU boxes, Stage 2's, E7's, E11's): never touched by this driver
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
TAG=jev-e4r-swe
BUNDLE=e4r.bundle
CAP_USD=${CAP_USD:-40}              # OpenRouter (DeepSeek + Jev) for E4r in total, smoke included (minimum-spend amendment)
GUARD_MIN=${GUARD_MIN:-240}         # power-off guard from `rearm` (setup + pre-pull session); eval re-arms EVAL_GUARD_MIN
EVAL_GUARD_MIN=${EVAL_GUARD_MIN:-600}
POWEROFF=${POWEROFF:-1}

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E4R_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E4R_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E4R_IID "* ]] && { echo "refused: $E4R_IID is not the E4r host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E4R_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E4R_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  AMI=${AMI:-$(awsl ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
    --query Parameter.Value --output text)}
  ud=$(mktemp)
  cat > "$ud" <<EOF
#!/bin/bash
shutdown -h +$GUARD_MIN "jev-e4r-swe: power-off guard ($GUARD_MIN min from boot)"
snap install aws-cli --classic || (apt-get update -qq && apt-get install -y -qq unzip && cd /tmp \\
  && curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o awscliv2.zip && unzip -q awscliv2.zip && ./aws/install)
EOF
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(m7i.8xlarge m6i.8xlarge c7i.8xlarge)
  for IT in "${types[@]}"; do
    for s in $SUBNETS; do
      out=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$IT" --subnet-id "$s" --security-group-ids "$SG" \
        --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior stop --count 1 \
        --user-data "file://$ud" \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG},{Key=Project,Value=jev}]" \
          "ResourceType=volume,Tags=[{Key=Name,Value=$TAG},{Key=Project,Value=jev}]" \
        --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name]' --output text 2>&1)
      rc=$?
      echo "$IT $s: $out"
      (( rc == 0 )) && { rm -f "$ud"; exit 0; }
      [[ "$out" == *InsufficientInstanceCapacity* || "$out" == *Unsupported* || "$out" == *"not supported"* ]] \
        || { rm -f "$ud"; exit 1; }
    done
  done
  rm -f "$ud"; exit 1
fi
if [[ "$MODE" == push ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null || { echo "bundle failed"; exit 1; }
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors || exit 1
  git -C "$ROOT" show "$BRANCH:scripts/aws/e4r_swe.sh" > "$tmp/e4r_swe.sh"   # the committed driver
  awsl s3 cp "$tmp/e4r_swe.sh" "$B/code/e4r_swe.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e4r_swe.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e4r_$(date +%s%N).sh'
    echo "sudo mkdir -p /opt/jev && sudo chown \$(id -u):\$(id -g) /opt/jev"
    echo "aws s3 cp $B/code/e4r_swe.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q GUARD_MIN=%q EVAL_GUARD_MIN=%q POWEROFF=%q\n' "$CAP_USD" "$GUARD_MIN" "$EVAL_GUARD_MIN" "$POWEROFF"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e4r_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e4r_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E4R_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e4r_driver.sh /tmp/e4r_driver.sh --only-show-errors && bash /tmp/e4r_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$B/e4r/status/e4r_box.txt" - 2>/dev/null | tail -n "${1:-30}"
  echo "--- e4.txt (E4's driver, EXP=e4r)"
  awsl s3 cp "$B/e4r/status/e4.txt" - 2>/dev/null | tail -n "${1:-30}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E4R_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E4R_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]'
  exit 0
fi
if [[ "$MODE" == fetch ]]; then   # small files only: summaries, rows, reports, status (trajectories stay in S3)
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  d="$ROOT/results/e4r_swe/box"; mkdir -p "$d"
  awsl s3 sync "$B/e4r/status" "$d/status" --only-show-errors --exclude '*' --include 'e4r_*' --include 'e4.txt' \
    --include 'e4_gate.txt' --exclude 'e4r_*.log'
  for ph in smoke eval; do
    awsl s3 sync "$B/e4r/runs/$ph/summary" "$d/$ph/summary" --only-show-errors
    for arm in always_deep always_fast jevagent; do
      awsl s3 sync "$B/e4r/runs/$ph/$arm/eval" "$d/$ph/$arm" --only-show-errors --exclude '*' --include "*.e4r_${ph}_${arm}.json"
      awsl s3 cp "$B/e4r/runs/$ph/$arm/jev_arm.json" "$d/$ph/$arm/jev_arm.json" --only-show-errors 2>/dev/null || true
    done
  done
  du -sh "$d"
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,27p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
W=/opt/jev; R=$W/JevAgentHarness/research; S=$W/hard_status
mkdir -p "$S"
export PATH="$HOME/.local/bin:/snap/bin:$PATH"
STATUS_S3="$B/e4r/status"; export STATUS_S3
st() { echo "$(date -u +%FT%TZ) [e4r:box] $*" | tee -a "$S/e4r_box.txt"
       aws s3 sync "$S" "$STATUS_S3" --only-show-errors --region "$REGION" --exclude '*' --include 'e4*' || true; }
echo $$ > "$S/e4r_driver.pid"
trap 'rm -f "$S/e4r_driver.pid"' EXIT

or_key() { aws ssm get-parameter --region "$REGION" --name ${SSM_PREFIX:?set SSM_PREFIX}/OPENROUTER_API_KEY --with-decryption \
             --query Parameter.Value --output text 2>/dev/null; }
# GET an OpenRouter key endpoint; the key goes through curl's stdin (-H @-), never through argv, a file or a log
or_get() { local k; k=$(or_key) || return 1; [ -n "$k" ] && [ "$k" != None ] || return 1
           printf 'Authorization: Bearer %s\n' "$k" | curl -s -m 20 -H @- "https://openrouter.ai/api/v1/$1"; }
or_usage() { or_get key | jq -r '.data.usage // "?"' 2>/dev/null; }

stage_rearm() {
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "E4r guard" 2>/dev/null
  cat > "$W/e4r_idle_guard.sh" <<'EOF'
#!/usr/bin/env bash
# E4r idle stop: power off after 30 min with no live driver (any /opt/jev/hard_status/*.pid) and no running container
S=/opt/jev/hard_status; idle=0
while sleep 300; do
  alive=0
  for f in "$S"/*.pid; do pid=$(cat "$f" 2>/dev/null); [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1; done
  if [ "$alive" = 1 ] || [ -n "$(docker ps -q 2>/dev/null; sudo -n docker ps -q 2>/dev/null)" ]; then idle=0; else idle=$((idle + 5)); fi
  if [ "$idle" -ge 30 ]; then
    echo "$(date -u +%FT%TZ) [e4r:idle] 30 min without a driver or container: powering off" >> "$S/e4r_box.txt"
    PATH="/snap/bin:$PATH" aws s3 sync "$S" s3://$BUCKET/e4r/status --only-show-errors --region us-east-1 \
      --exclude '*' --include 'e4*' || true
    sudo shutdown -h now
  fi
done
EOF
  pgrep -f e4r_idle_guard.sh >/dev/null || nohup setsid bash "$W/e4r_idle_guard.sh" > /dev/null 2>&1 < /dev/null &
  st "rearm: power-off in $GUARD_MIN min, idle stop armed; $(nproc) vCPU, $(free -g | awk '/Mem:/{print $2}') GB RAM, disk $(df -BG / | awk 'NR==2{print $3"/"$2}')"
}

stage_setup() {
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq && sudo apt-get install -y -qq git jq unzip build-essential ca-certificates curl >/dev/null \
    || { st "setup: apt failed"; return 1; }
  if ! command -v docker >/dev/null; then
    curl -fsSL https://get.docker.com | sudo sh >/dev/null 2>&1 || { st "setup: docker install failed"; return 1; }
  fi
  sudo usermod -aG docker ubuntu
  sudo mkdir -p /etc/docker
  cat <<'JSON' | sudo tee /etc/docker/daemon.json >/dev/null
{
  "default-address-pools": [{"base": "10.128.0.0/12", "size": 24}],
  "registry-mirrors": ["https://mirror.gcr.io"],
  "max-concurrent-downloads": 8,
  "log-driver": "json-file",
  "log-opts": {"max-size": "50m", "max-file": "2"},
  "features": {"buildkit": true}
}
JSON
  sudo systemctl restart docker || { st "setup: docker restart failed"; return 1; }
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || { st "setup: uv failed"; return 1; }
  uv python install 3.11 >/dev/null 2>&1 || { st "setup: python failed"; return 1; }
  stage_update || return 1
  cd "$R" || return 1
  [ -x .venv/bin/python ] || uv venv -q -p 3.11 .venv || return 1
  uv pip install -q --python .venv -e ".[all]" "mini-swe-agent==2.4.6" || { st "setup: pip repo failed"; return 1; }
  [ -x .venv-eval/bin/python ] || uv venv -q -p 3.11 .venv-eval || return 1
  uv pip install -q --python .venv-eval "swebench==4.1.0" || { st "setup: pip swebench failed"; return 1; }
  sudo docker run --rm hello-world >/dev/null 2>&1 || { st "setup: docker run failed"; return 1; }
  MSWEA_SILENT_STARTUP=1 .venv/bin/python -m pytest -q tests/test_explore_e4_swe.py tests/test_mini_swe_adapter.py \
    > "$S/e4r_pytest.txt" 2>&1
  st "setup done: $(git log --oneline -1 | cut -c1-60); mini-swe $(.venv/bin/pip show mini-swe-agent 2>/dev/null | awk '/^Version/{print $2}'), swebench $(.venv-eval/bin/python -c 'import swebench; print(swebench.__version__)' 2>/dev/null); pytest: $(tail -1 "$S/e4r_pytest.txt")"
}

stage_update() {
  aws s3 cp "$B/code/$BUNDLE" "$W/$BUNDLE" --only-show-errors --region "$REGION" || { st "update: no bundle"; return 1; }
  if [ ! -d "$R/.git" ]; then
    git clone -q -b "$BRANCH" "$W/$BUNDLE" "$R" || { st "update: clone failed"; return 1; }
  else
    git -C "$R" checkout -q -- . && git -C "$R" pull -q --ff-only "$W/$BUNDLE" "$BRANCH" || { st "update: pull failed"; return 1; }
  fi
  st "update: $(git -C "$R" log --oneline -1 | cut -c1-72)"
}

HUB_RESERVE=${HUB_RESERVE:-10}; export HUB_RESERVE
hub_remaining() { local tok
  tok=$(curl -s -m 20 "https://auth.docker.io/token?service=registry.docker.io&scope=repository:ratelimitpreview/test:pull" \
        | jq -r .token 2>/dev/null)
  curl -s -m 20 --head -H "Authorization: Bearer $tok" https://registry-1.docker.io/v2/ratelimitpreview/test/manifests/latest \
    | tr -d '\r' | awk -F'[ ;]' 'tolower($1)=="ratelimit-remaining:"{print $2}'; }
pull() { local img="docker.io/swebench/sweb.eval.x86_64.$(echo "${1//__/_1776_}" | tr 'A-Z' 'a-z'):latest" i r
         sudo docker image inspect "$img" >/dev/null 2>&1 && { echo "CACHED $img"; return 0; }
         for i in $(seq 1 30); do r=$(hub_remaining); { [ -z "$r" ] || [ "$r" -ge "$HUB_RESERVE" ]; } && break
           echo "WAIT hub_remaining=$r $img"; sleep 120; done
         for i in 1 2 3 4 5; do sudo docker pull -q "$img" >/dev/null 2>&1 && { echo "PULLED $img"; return 0; }; sleep $((i * 20)); done
         echo "PULL FAILED $img"; }
export -f pull hub_remaining

stage_prepull() {
  cd "$R" || return 1
  local t0; t0=$(date +%s)
  cat configs/explore/e4r_swe/smoke_ids.txt configs/explore/e4r_swe/eval_ids.txt | sed 's/#.*//; /^\s*$/d' \
    | xargs -P 6 -I{} bash -c 'pull "$@"' _ {} > "$S/e4r_prepull.txt" 2>&1
  st "prepull: $(( $(date +%s) - t0 )) s; cached $(grep -c ^CACHED "$S/e4r_prepull.txt"), pulled $(grep -c ^PULLED "$S/e4r_prepull.txt"), failures $(grep -c 'PULL FAILED' "$S/e4r_prepull.txt"); disk $(df -BG / | awk 'NR==2{print $3"/"$2}')"
}

cap_usage() { cat "$S/e4r_cap_usage.txt" 2>/dev/null; }

stage_smoke() {
  cd "$R" || return 1
  local info u0
  info=$(or_get auth/key | jq -c '{m: .data.is_management_key, p: .data.is_provisioning_key, limit: .data.limit, usage: .data.usage}' 2>/dev/null)
  st "key check: $info"
  case "$info" in *'"m":true'*|*'"p":true'*|'') st "refusing: the key in SSM is not an inference key (or unreadable)"; return 1 ;; esac
  u0=$(or_usage)
  case "$u0" in ''|'?'|null) st "refusing: key usage unreadable"; return 1 ;; esac
  echo "$u0" > "$S/e4r_usage_start.txt"
  awk -v u="$u0" -v c="$CAP_USD" 'BEGIN{printf "%.4f\n", u + c}' > "$S/e4r_cap_usage.txt"
  st "smoke: key usage at E4r start \$$u0; watchdog stops the runner at \$$(cap_usage) of key usage (cap \$$CAP_USD)"
  env EXP=e4r PHASE=smoke STOP_WHEN_DONE=0 NO_GUARD_REARM=1 CAP_USAGE="$(cap_usage)" PHASE_COST_LIMIT=6 \
    bash scripts/aws/e4_swe.sh > "$S/e4r_smoke.log" 2>&1
  st "smoke driver exit $?; logged $(.venv/bin/python scripts/explore_e4_swe.py --exp e4r spend runs/e4r_swe/smoke 2>/dev/null | tail -1)"
}

stage_gate() {
  cd "$R" || return 1
  MSWEA_SILENT_STARTUP=1 .venv/bin/python scripts/explore_e4_swe.py --exp e4r gate runs/e4r_swe/smoke --cap "$CAP_USD" \
    --out "$S/e4r_gate.json" > "$S/e4r_gate.txt" 2>&1
  local rc=$?
  st "gate exit $rc: $(tail -1 "$S/e4r_gate.txt")"
  return $rc
}

stage_eval() {
  cd "$R" || return 1
  [ -s "$S/e4r_gate.json" ] && [ -s "$S/e4r_cap_usage.txt" ] || { st "refusing the eval: no gate decision or cap"; return 1; }
  local dec k rf="" spent left
  dec=$(jq -r .decision "$S/e4r_gate.json"); k=$(jq -r .k "$S/e4r_gate.json")
  case "$dec" in
    full) ;;
    cut) rf="--arm-limit always_deep=$k --arm-limit always_fast=$k --arm-limit jevagent=$k" ;;
    *) st "refusing the eval: gate decision $dec"; return 1 ;;
  esac
  spent=$(.venv/bin/python scripts/explore_e4_swe.py --exp e4r spend runs/e4r_swe/smoke 2>/dev/null | tail -1 | jq -r .total)
  left=$(awk -v c="$CAP_USD" -v s="${spent:-0}" 'BEGIN{printf "%.2f", c - s}')
  st "eval: gate $dec (k = $k); mini-swe global limit \$$left (cap \$$CAP_USD - logged smoke \$${spent:-0}); key-usage cap \$$(cap_usage)"
  env EXP=e4r PHASE=eval STOP_WHEN_DONE=0 CAP_USAGE="$(cap_usage)" PHASE_COST_LIMIT="$left" GUARD_MIN="$EVAL_GUARD_MIN" \
    RUN_FLAGS="$rf" bash scripts/aws/e4_swe.sh > "$S/e4r_eval.log" 2>&1
  st "eval driver exit $?"
}

stage_final() {
  cd "$R" 2>/dev/null || true
  local u1 u0 sp n k
  u0=$(cat "$S/e4r_usage_start.txt" 2>/dev/null); u1=$(or_usage)
  sp=$(.venv/bin/python scripts/explore_e4_swe.py --exp e4r spend runs/e4r_swe/smoke runs/e4r_swe/eval 2>/dev/null | tail -1)
  printf '{"logged": %s, "key_usage_start": "%s", "key_usage_end": "%s", "written": "%s"}\n' "${sp:-null}" "${u0:-}" "${u1:-}" \
    "$(date -u +%FT%TZ)" > "$S/e4r_spend.json"
  n=0
  k=$(or_key) && [ -n "$k" ] && [ "$k" != None ] && n=$(grep -rlF -- "$k" runs/e4r_swe "$S" /opt/jev/e4r_box.log 2>/dev/null | wc -l)
  unset k
  st "final: spend $(cat "$S/e4r_spend.json"); leak check: $n file(s) contain the key value"
  for ph in smoke eval; do
    [ -d "runs/e4r_swe/$ph" ] && aws s3 sync "runs/e4r_swe/$ph" "$B/e4r/runs/$ph" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  cp /opt/jev/e4r_box.log "$S/e4r_box_log.txt" 2>/dev/null
  st "final: synced"
  if [ "$POWEROFF" = 1 ]; then st "powering off"; rm -f "$S/e4r_driver.pid"; sudo shutdown -h +1 "E4r done"; fi
}

st "stages: $STAGES"
failed=0
for stg in $STAGES; do
  [ "$stg" = final ] && continue
  if [ "$failed" = 0 ]; then
    if declare -F "stage_$stg" >/dev/null; then
      "stage_$stg" || { failed=1; st "stage $stg FAILED; the chain stops"; }
    else
      failed=1; st "unknown stage $stg"
    fi
  fi
done
[[ " $STAGES " == *" final "* ]] && stage_final
st "chain end (failed=$failed)"
