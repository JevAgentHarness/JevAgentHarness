#!/usr/bin/env bash
# E25 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09, "E25"; exploratory, descriptive): per-step routing by the
# decision model in two more open-source harnesses through Harbor, one NEW CPU host per harness (SSM only, no inbound):
#   H=tb   pi 0.99.2 on Terminal-Bench 2.0      host tag Name=jev-e25-tb   (adapters/pi/run_tb2.sh)
#   H=swe  SWE-agent v1.1.0 on SWE-bench Verified  host tag Name=jev-e25-swe  (adapters/swe_agent/run_swe.sh)
# Arms baseline / jev_router / always_cheap (the last only when the gate's projection fits), 5 concurrent trials each,
# k = 1, Harbor's retries of infrastructure errors; Jev through OpenRouter behind the host decision server (no cache).
#
#   LOCAL: bash scripts/aws/e25_harnesses.sh launch tb|swe [itype ...]   # refuses if a host with the tag exists
#          bash scripts/aws/e25_harnesses.sh push                          # bundle + this driver -> s3 code/
#          bash scripts/aws/e25_harnesses.sh send tb|swe "STAGES" | status tb|swe [N] | start|stop|itype tb|swe
#          bash scripts/aws/e25_harnesses.sh fetch tb|swe                  # small records -> results/explore/e25_harnesses/box/
#   BOX stages (in the order sent; a failed stage stops the chain; `final` always runs when sent):
#     rearm    power-off guard (GUARD_MIN from now) and the 30-minute idle stop
#     setup    Docker CE, uv, the branch from code/e25.bundle, .venv (.[all]), Harbor at 4f848e3, offline E25 tests
#     update   fast-forward the checkout to the latest code/e25.bundle
#     smoke    decision server, then the smoke tasks x 3 arms (development only) -> runs/e25/<H>/smoke
#     resmoke  keep the finished smoke as runs/e25/<H>/smoke_<k>, then `smoke` again (E25 (e): one re-smoke after a fix)
#     gate     scripts/explore_e25_harnesses.py gate (E25 (e)) -> e25_<H>_gate.json; exit 1 = no-go
#     eval     the gate's tasks x the gate's arms in parallel, spend watchdog at CAP - 15 -> runs/e25/<H>/eval
#     evalcheap  (tb) always_cheap as a third job next to a running eval, after the amended gate admits it
#     evalrouter (tb) the router arm again with the thought-signature fix; the defective run -> runs/e25/tb/eval_defect
#     rerun    one re-run of every trial that still carries an infrastructure error -> runs/e25/<H>/eval_rerun
#     final    spend, leak check, sync, power-off (POWEROFF=1)
# Docker needs the group of a new login: send `rearm setup` alone, then the other stages in a second `send`.
# Keys come from SSM Parameter Store into the stage's processes only (GEMINI_API_KEY for Harbor, which hands it to the
# agent through its own model-connection resolution; OPENROUTER_API_KEY only for the decision server); nothing here
# prints or writes a key, and `final` counts the files that contain one (must be 0).
set -uo pipefail
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
# hosts of other rows and agents: never touched by this driver
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
BUNDLE=e25.bundle
GUARD_MIN=${GUARD_MIN:-300}
EVAL_GUARD_MIN=${EVAL_GUARD_MIN:-720}
POWEROFF=${POWEROFF:-1}
N_CONC=${N_CONC:-5}
OR_CAP=${OR_CAP:-25}

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
tag_of() { case "$1" in tb) echo jev-e25-tb ;; swe) echo jev-e25-swe ;; *) return 1 ;; esac; }
cap_of() { case "$1" in tb) echo 140 ;; swe) echo 100 ;; esac; }
iid_of() {  # the one live host with the harness's tag
  local tag; tag=$(tag_of "$1") || { echo "harness must be tb or swe" >&2; return 1; }
  local ids; ids=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$tag" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [ "$(echo "$ids" | wc -w)" = 1 ] || { echo "expected one $tag host, found: '$ids'" >&2; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $ids "* ]] && { echo "refused: $ids is another row's host" >&2; return 1; }
  echo "$ids"
}

if [[ "$MODE" == launch ]]; then
  H=${1:?usage: launch tb|swe [itype ...]}; shift
  TAG=$(tag_of "$H") || exit 1
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  AMI=${AMI:-$(awsl ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
    --query Parameter.Value --output text)}
  ud=$(mktemp)
  cat > "$ud" <<EOF
#!/bin/bash
shutdown -h +$GUARD_MIN "$TAG: power-off guard ($GUARD_MIN min from boot)"
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/e25_harnesses.sh" > "$tmp/e25_harnesses.sh"   # the committed driver
  awsl s3 cp "$tmp/e25_harnesses.sh" "$B/code/e25_harnesses.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e25_harnesses.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  H=${1:?usage: send tb|swe "STAGES"}; STAGES=${2:?usage: send tb|swe "STAGES"}
  IID=$(iid_of "$H") || exit 1
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e25_$(date +%s%N).sh'
    echo "sudo mkdir -p /opt/jev && sudo chown \$(id -u):\$(id -g) /opt/jev"
    echo "aws s3 cp $B/code/e25_harnesses.sh \$f --only-show-errors --region $REGION"
    printf 'export H=%q GUARD_MIN=%q EVAL_GUARD_MIN=%q POWEROFF=%q N_CONC=%q OR_CAP=%q\n' "$H" "$GUARD_MIN" \
      "$EVAL_GUARD_MIN" "$POWEROFF" "$N_CONC" "$OR_CAP"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e25_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e25_driver_$H.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e25_driver_$H.sh /tmp/e25_driver.sh --only-show-errors && bash /tmp/e25_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  H=${1:?usage: status tb|swe [N]}
  awsl s3 cp "$B/e25/$H/status/e25_box.txt" - 2>/dev/null | tail -n "${2:-30}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  H=${1:?usage: start|stop tb|swe}
  IID=$(iid_of "$H") || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  H=${1:?usage: itype tb|swe}
  IID=$(iid_of "$H") || exit 1
  awsl ec2 describe-instances --instance-ids "$IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]'
  exit 0
fi
if [[ "$MODE" == fetch ]]; then   # small records only (result.json, decision / call logs, pi sessions stay in S3)
  H=${1:?usage: fetch tb|swe}
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  d="$ROOT/runs/e25/$H"; mkdir -p "$d"
  awsl s3 sync "$B/e25/$H/status" "$ROOT/results/explore/e25_harnesses/box/$H/status" --only-show-errors \
    --exclude '*.log'
  awsl s3 sync "$B/e25/$H/runs" "$d" --only-show-errors --exclude '*' --include '*/result.json' \
    --include '*/agent/jev-decisions.jsonl' --include '*/agent/jev-calls.jsonl' --include '*/agent/pi/sessions/*' \
    --include '*/agent/trajectory.json' \
    --include '*/verifier/ctrf.json' --include '*/verifier/reward.txt' \
    --include '*/config.json' --include '*/run_*.log' 2>/dev/null
  du -sh "$d"
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,24p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
H=${H:?H=tb|swe}
W=/opt/jev; R=$W/JevAgentHarness/research; S=$W/e25_status
mkdir -p "$S"
export PATH="$HOME/.local/bin:/snap/bin:$PATH"
STATUS_S3="$B/e25/$H/status"
CAP=$(cap_of "$H")
RUNS=$R/runs/e25/$H
st() { echo "$(date -u +%FT%TZ) [e25:$H] $*" | tee -a "$S/e25_box.txt"
       aws s3 sync "$S" "$STATUS_S3" --only-show-errors --region "$REGION" || true; }
echo $$ > "$S/e25_driver_$$.pid"   # one pid file per chain (a second chain may run next to the first)
trap 'rm -f "$S/e25_driver_$$.pid"' EXIT

ssm_val() { aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$1" --with-decryption \
              --query Parameter.Value --output text 2>/dev/null; }
or_usage() {  # OpenRouter key usage (the key goes through curl's stdin, never argv / a file / a log)
  local k; k=$(ssm_val OPENROUTER_API_KEY) || return 1; [ -n "$k" ] && [ "$k" != None ] || return 1
  printf 'Authorization: Bearer %s\n' "$k" | curl -s -m 20 -H @- https://openrouter.ai/api/v1/key | jq -r '.data.usage // "?"'
}
leak_check() {  # files under the given paths that contain a key value (count only)
  local n=0 k v
  for v in GEMINI_API_KEY OPENROUTER_API_KEY; do
    k=$(ssm_val "$v") || continue; [ -n "$k" ] && [ "$k" != None ] || continue
    n=$((n + $(grep -rlF -- "$k" "$@" 2>/dev/null | wc -l)))
  done
  unset k; echo "$n"
}

stage_rearm() {
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "E25 guard" 2>/dev/null
  cat > "$W/e25_idle_guard.sh" <<EOF
#!/usr/bin/env bash
# E25 idle stop: power off after 30 min with no live driver (/opt/jev/e25_status/*.pid) and no running container
S=/opt/jev/e25_status; idle=0
while sleep 300; do
  alive=0
  for f in "\$S"/*.pid; do pid=\$(cat "\$f" 2>/dev/null); [ -n "\$pid" ] && kill -0 "\$pid" 2>/dev/null && alive=1; done
  if [ "\$alive" = 1 ] || [ -n "\$(sudo -n docker ps -q 2>/dev/null)" ]; then idle=0; else idle=\$((idle + 5)); fi
  if [ "\$idle" -ge 30 ]; then
    echo "\$(date -u +%FT%TZ) [e25:idle] 30 min without a driver or container: powering off" >> "\$S/e25_box.txt"
    PATH="/snap/bin:\$PATH" aws s3 sync "\$S" $STATUS_S3 --only-show-errors --region $REGION || true
    sudo shutdown -h now
  fi
done
EOF
  pgrep -f e25_idle_guard.sh >/dev/null || nohup setsid bash "$W/e25_idle_guard.sh" > /dev/null 2>&1 < /dev/null &
  st "rearm: power-off in $GUARD_MIN min, idle stop armed; $(nproc) vCPU, $(free -g | awk '/Mem:/{print $2}') GB RAM, disk $(df -BG / | awk 'NR==2{print $3"/"$2}')"
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
  uv python install 3.11 3.12 >/dev/null 2>&1 || { st "setup: python failed"; return 1; }
  stage_update || return 1
  cd "$R" || return 1
  [ -x .venv/bin/python ] || uv venv -q -p 3.11 .venv || return 1
  uv pip install -q --python .venv -e ".[all]" || { st "setup: pip repo failed"; return 1; }
  [ -x "$W/harbor-venv/bin/harbor" ] || uv venv -q -p 3.12 "$W/harbor-venv" || return 1
  uv pip install -q --python "$W/harbor-venv" \
    "harbor @ git+https://github.com/laude-institute/harbor@4f848e367e6e2627c2c0b1826bb3b20ceab1198f" \
    || { st "setup: pip harbor failed"; return 1; }
  sudo docker run --rm hello-world >/dev/null 2>&1 || { st "setup: docker run failed"; return 1; }
  .venv/bin/python -m pytest -q tests/test_explore_e25.py > "$S/e25_pytest.txt" 2>&1
  PYTHONPATH="adapters/pi:adapters/swe_agent" "$W/harbor-venv/bin/python" -c \
    'import harbor_jev_pi, harbor_jev_swe_agent; print("harbor agents import ok")' >> "$S/e25_pytest.txt" 2>&1
  st "setup done: $(git log --oneline -1 | cut -c1-60); harbor $("$W/harbor-venv/bin/python" -c 'import importlib.metadata as m; print(m.version("harbor"))'); pytest: $(grep -E 'passed|failed|error' "$S/e25_pytest.txt" | tail -1); $(tail -1 "$S/e25_pytest.txt")"
}

ensure_server() {
  if ! curl -s --noproxy '*' -m 5 http://127.0.0.1:8090/health > /dev/null; then
    ( k=$(ssm_val OPENROUTER_API_KEY); [ -n "$k" ] && [ "$k" != None ] || exit 1
      cd "$R" && OPENROUTER_API_KEY="$k" setsid nohup .venv/bin/python -m jevharness.decision.server --host 0.0.0.0 \
        --port 8090 --name jev-1.13 --s1 '{"type":"jev","provider":"openrouter"}' >> "$S/decision_server.log" 2>&1 < /dev/null & )
    for _ in $(seq 1 30); do curl -s --noproxy '*' -m 2 http://127.0.0.1:8090/health > /dev/null && break; sleep 2; done
  fi
  local h; h=$(curl -s --noproxy '*' -m 5 http://127.0.0.1:8090/health) || { st "decision server DOWN"; return 1; }
  st "decision server: $h"
}

# run the arms of the harness as parallel Harbor jobs; env: OUT, N, ARMS_LIST, and INCLUDE (tb) / IDS_FILE (swe)
run_arms() {
  local out=$1; shift
  local pids=() a g
  g=$(ssm_val GEMINI_API_KEY) || { st "no GEMINI_API_KEY in SSM"; return 1; }
  mkdir -p "$out"
  for a in "$@"; do
    if [ "$H" = tb ]; then
      ( cd "$R" && env -u OPENROUTER_API_KEY GEMINI_API_KEY="$g" HARBOR="$W/harbor-venv/bin/harbor" OUT="$out" \
          ENV_FILE=/nonexistent N_CONCURRENT="$N" N_ATTEMPTS=1 MAX_TURNS=100 THINKING=minimal JEV_WIRE_MODEL=jev-1.13 \
          INCLUDE="${INCLUDE:-}" HARBOR_ARGS="--max-retries 2 --retry-exclude AgentTimeoutError" \
          bash adapters/pi/run_tb2.sh "$a" > "$out/run_$a.log" 2>&1; echo "exit=$?" >> "$out/run_$a.log" ) &
    else
      ( cd "$R" && env -u OPENROUTER_API_KEY GEMINI_API_KEY="$g" HARBOR="$W/harbor-venv/bin/harbor" OUT="$out" \
          N_CONCURRENT="$N" N_ATTEMPTS=1 IDS_FILE="$IDS_FILE" \
          HARBOR_ARGS="--max-retries 2 --retry-exclude AgentTimeoutError" \
          bash adapters/swe_agent/run_swe.sh "$a" > "$out/run_$a.log" 2>&1; echo "exit=$?" >> "$out/run_$a.log" ) &
    fi
    pids+=($!); sleep 20
  done
  unset g
  echo "${pids[*]}" > "$S/e25_arm_pids.txt"
  for a in "${pids[@]}"; do wait "$a"; done
}

watchdog() {  # stop the runs at CAP - 15 of logged LLM spend (smoke included) or OR_CAP - 1 of logged Jev spend
  local done_flag=${1:-$S/e25_eval_done}
  local limit; limit=$(awk -v c="$CAP" 'BEGIN{print c - 15}')
  while sleep 300; do
    [ -f "$done_flag" ] && return 0
    local sp tot jev
    sp=$(cd "$R" && .venv/bin/python scripts/explore_e25_harnesses.py spend "$RUNS"/smoke* "$RUNS/eval" "$RUNS/eval_rerun" "$RUNS/eval_defect" 2>/dev/null)
    tot=$(echo "$sp" | jq -r .total); jev=$(echo "$sp" | jq -r .jev)
    echo "$(date -u +%FT%TZ) $sp" >> "$S/e25_spend_log.txt"
    if awk -v t="${tot:-0}" -v l="$limit" -v j="${jev:-0}" -v o="$OR_CAP" 'BEGIN{exit !((t >= l) || (j >= o - 1))}'; then
      st "WATCHDOG: logged spend $tot (Jev $jev) reached the stop line ($limit / OR $((OR_CAP - 1))): stopping the runs"
      pkill -INT -f "harbor run" ; sleep 60; pkill -KILL -f "harbor run"
      touch "$S/e25_watchdog_fired"
      return 0
    fi
  done
}

stage_smoke() {
  cd "$R" || return 1
  echo "$(or_usage)" > "$S/e25_or_usage_start.txt"
  st "smoke: OpenRouter key usage at start \$$(cat "$S/e25_or_usage_start.txt")"
  ensure_server || return 1
  local out=$RUNS/smoke
  if [ "$H" = tb ]; then
    N=2 INCLUDE="$(sed 's/^/*/' configs/explore/e25_harnesses/tb2_smoke_tasks.txt | tr '\n' ' ')" \
      run_arms "$out" baseline jev_router always_cheap
  else
    N=2 IDS_FILE="$R/configs/explore/e25_harnesses/swe_smoke_ids.txt" run_arms "$out" baseline jev_router always_cheap
  fi
  st "smoke done: $(.venv/bin/python scripts/explore_e25_harnesses.py spend "$out" 2>/dev/null)"
  aws s3 sync "$out" "$B/e25/$H/runs/smoke" --only-show-errors --region "$REGION" || true
}

stage_resmoke() {  # E25 (e): one re-smoke after an implementation fix; the first smoke stays as smoke_<k> (its spend counts)
  cd "$R" || return 1
  local k=1
  while [ -e "$RUNS/smoke_$k" ]; do k=$((k + 1)); done
  [ -d "$RUNS/smoke" ] || { st "resmoke: no smoke to keep"; return 1; }
  mv "$RUNS/smoke" "$RUNS/smoke_$k" && aws s3 sync "$RUNS/smoke_$k" "$B/e25/$H/runs/smoke_$k" --only-show-errors --region "$REGION"
  st "resmoke: the first smoke is kept as runs/e25/$H/smoke_$k ($(.venv/bin/python scripts/explore_e25_harnesses.py spend "$RUNS/smoke_$k" 2>/dev/null | jq -c '{total, trials}'))"
  stage_smoke
}

stage_gate() {
  cd "$R" || return 1
  local leaks rc
  leaks=$(leak_check "$RUNS/smoke" /opt/jev/e25_box.log "$S")
  # spent = every smoke of this harness (a re-smoke after an implementation fix keeps the first one as smoke_<k>)
  local spent; spent=$(.venv/bin/python scripts/explore_e25_harnesses.py spend "$RUNS"/smoke* 2>/dev/null | jq -r .total)
  .venv/bin/python scripts/explore_e25_harnesses.py gate --harness "$H" --smoke "$RUNS/smoke" --leaks "$leaks" \
    --spent "${spent:-0}" \
    --out "$S/e25_${H}_gate.json" --include-out "$S/e25_${H}_tasks.txt" > "$S/e25_${H}_gate.txt" 2>&1
  rc=$?
  st "gate exit $rc: decision $(jq -r .decision "$S/e25_${H}_gate.json" 2>/dev/null), n $(jq -r .n "$S/e25_${H}_gate.json" 2>/dev/null), arms $(jq -c .arms "$S/e25_${H}_gate.json" 2>/dev/null), leaks $leaks"
  return $rc
}

disk_pruner() {  # harness B: per-trial SWE-bench image builds grow BuildKit's cache by ~5 GB per instance; keep it bounded
  cat > "$W/e25_disk_pruner.sh" <<'EOF2'
#!/usr/bin/env bash
while sleep 600; do
  u=$(df --output=pcent / | tail -1 | tr -dc 0-9)
  if [ "$u" -ge 55 ]; then
    sudo docker builder prune -f --keep-storage 150g >/dev/null 2>&1
    sudo docker image prune -f >/dev/null 2>&1
    echo "$(date -u +%FT%TZ) [e25:disk] used ${u}% -> builder cache pruned to 150 GB; now $(df --output=pcent / | tail -1 | tr -dc 0-9)%" >> /opt/jev/e25_status/e25_box.txt
  fi
done
EOF2
  pgrep -f e25_disk_pruner.sh >/dev/null || (nohup setsid bash "$W/e25_disk_pruner.sh" >/dev/null 2>&1 < /dev/null &)
}

stage_eval() {
  cd "$R" || return 1
  [ "$H" = swe ] && disk_pruner   # (started by hand on the first harness-B eval at 02:03 UTC; the same script)
  [ "$(jq -r .decision "$S/e25_${H}_gate.json" 2>/dev/null)" = go ] || { st "refusing the eval: no go from the gate"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$EVAL_GUARD_MIN" "E25 eval guard" 2>/dev/null
  ensure_server || return 1
  local arms; arms=$(jq -r '.arms | join(" ")' "$S/e25_${H}_gate.json")
  rm -f "$S/e25_eval_done"
  watchdog & local wd=$!
  st "eval: arms [$arms], n $(jq -r .n "$S/e25_${H}_gate.json"), $N_CONC concurrent per arm; guard $EVAL_GUARD_MIN min; watchdog at \$$((CAP - 15)) logged"
  if [ "$H" = tb ]; then
    local inc=""
    [ -s "$S/e25_${H}_tasks.txt" ] && inc="$(sed 's/^/*/' "$S/e25_${H}_tasks.txt" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    N=$N_CONC INCLUDE="$inc" run_arms "$RUNS/eval" $arms
  else
    # shellcheck disable=SC2086
    N=$N_CONC IDS_FILE="$S/e25_${H}_tasks.txt" run_arms "$RUNS/eval" $arms
  fi
  touch "$S/e25_eval_done"; kill "$wd" 2>/dev/null
  .venv/bin/python scripts/explore_e25_harnesses.py spend "$RUNS"/smoke* "$RUNS/eval" "$RUNS/eval_rerun" "$RUNS/eval_defect" > "$S/e25_spend_logged.json" 2>/dev/null
  st "eval done: $(cat "$S/e25_spend_logged.json")"
  aws s3 sync "$RUNS/eval" "$B/e25/$H/runs/eval" --only-show-errors --region "$REGION" || true
}

stage_evalcheap() {  # harness A's always_cheap as a third job next to the running evaluation (amendment ~01:35 UTC)
  cd "$R" || return 1
  [ "$H" = tb ] || { st "evalcheap: harness A only"; return 1; }
  local spent; spent=$(.venv/bin/python scripts/explore_e25_harnesses.py spend "$RUNS"/smoke* 2>/dev/null | jq -r .total)
  .venv/bin/python scripts/explore_e25_harnesses.py gate --harness tb --smoke "$RUNS/smoke" --spent "${spent:-0}" \
    --out "$S/e25_tb_gate_amended.json" > "$S/e25_tb_gate_amended.txt" 2>&1
  jq -e '.decision == "go" and (.arms | index("always_cheap")) and .full_dataset' "$S/e25_tb_gate_amended.json" >/dev/null \
    || { st "evalcheap: the amended gate does not admit always_cheap on the 89 tasks: $(jq -c '{decision, n, arms}' "$S/e25_tb_gate_amended.json")"; return 1; }
  ensure_server || return 1
  rm -f "$S/e25_cheap_done"
  local deadline=$(( $(date +%s) + EVAL_GUARD_MIN * 60 ))
  (  # hold a power-off that the main chain's `final` schedules ("E25 done") until this job ends; keep the guard
    while [ ! -f "$S/e25_cheap_done" ]; do
      u=$(sudo cat /run/systemd/shutdown/scheduled 2>/dev/null | sed -n 's/^USEC=//p')
      if [ -n "$u" ] && [ $(( u / 1000000 )) -lt $(( deadline - 300 )) ]; then
        sudo shutdown -c; sudo shutdown -h +$(( (deadline - $(date +%s)) / 60 )) "E25 guard (evalcheap)"
        st "evalcheap: moved a scheduled power-off to the guard deadline while always_cheap runs"
      fi
      sleep 20
    done ) &
  local hold=$!
  watchdog "$S/e25_cheap_done" & local wd=$!
  st "evalcheap: always_cheap on the 89 tasks, $N_CONC concurrent; amended gate $(jq -c '{n, arms, projected: .checks["5_projection"].projected}' "$S/e25_tb_gate_amended.json")"
  N=$N_CONC INCLUDE="" run_arms "$RUNS/eval" always_cheap
  touch "$S/e25_cheap_done"; kill "$wd" "$hold" 2>/dev/null
  st "evalcheap done: $(.venv/bin/python scripts/explore_e25_harnesses.py spend "$RUNS"/smoke* "$RUNS/eval" "$RUNS/eval_rerun" "$RUNS/eval_defect" 2>/dev/null)"
  aws s3 sync "$RUNS/eval" "$B/e25/$H/runs/eval" --only-show-errors --region "$REGION" || true
}

stage_evalrouter() {  # amendment 2026-10-09 ~03:05 UTC: harness A's router arm again, with the thought-signature fix
  cd "$R" || return 1
  [ "$H" = tb ] || { st "evalrouter: harness A only"; return 1; }
  grep -q "fillMissingThoughtSignatures" adapters/pi/jev-router.ts || { st "evalrouter: the checkout lacks the fix"; return 1; }
  pgrep -af "harbor run" | grep -q -- "--job-name jev_router" && { st "evalrouter: a jev_router job still runs"; return 1; }
  if [ -d "$RUNS/eval/jev_router" ]; then
    mkdir -p "$RUNS/eval_defect" && mv "$RUNS/eval/jev_router" "$RUNS/eval_defect/jev_router" || return 1
    mv "$RUNS/eval/run_jev_router.log" "$RUNS/eval_defect/" 2>/dev/null
    aws s3 sync "$RUNS/eval_defect" "$B/e25/$H/runs/eval_defect" --only-show-errors --region "$REGION" || true
    st "evalrouter: the defective router run is kept as runs/e25/tb/eval_defect/jev_router ($(ls "$RUNS"/eval_defect/jev_router/*/result.json 2>/dev/null | wc -l) trials)"
  fi
  ensure_server || return 1
  rm -f "$S/e25_router_done"
  local deadline=$(( $(date +%s) + EVAL_GUARD_MIN * 60 ))
  (  # hold power-offs that other chains' `final` schedule until this job ends; keep the guard
    while [ ! -f "$S/e25_router_done" ]; do
      u=$(sudo cat /run/systemd/shutdown/scheduled 2>/dev/null | sed -n 's/^USEC=//p')
      if [ -n "$u" ] && [ $(( u / 1000000 )) -lt $(( deadline - 300 )) ]; then
        sudo shutdown -c; sudo shutdown -h +$(( (deadline - $(date +%s)) / 60 )) "E25 guard (evalrouter)"
        st "evalrouter: moved a scheduled power-off to the guard deadline while jev_router runs"
      fi
      sleep 20
    done ) &
  local hold=$!
  watchdog "$S/e25_router_done" & local wd=$!
  st "evalrouter: jev_router (fixed) on the 89 tasks, $N_CONC concurrent"
  N=$N_CONC INCLUDE="" run_arms "$RUNS/eval" jev_router
  touch "$S/e25_router_done"; kill "$wd" "$hold" 2>/dev/null
  st "evalrouter done: $(.venv/bin/python scripts/explore_e25_harnesses.py spend "$RUNS"/smoke* "$RUNS/eval" "$RUNS/eval_rerun" "$RUNS/eval_defect" 2>/dev/null)"
  aws s3 sync "$RUNS/eval" "$B/e25/$H/runs/eval" --only-show-errors --region "$REGION" || true
}

stage_rerun() {  # amendment 2026-10-09 ~02:30 UTC: one re-run of every trial that still carries an infrastructure error
  cd "$R" || return 1
  local lst="$S/e25_${H}_errored.txt" a f inc pids=() p
  rm -f "$S/e25_rerun_done"
  # hold the power-off that another chain's `final` schedules ("E25 done", +1 min) while this stage waits and re-runs
  # (a rerun chain sent next to a running `eval final`); the guard stays: GUARD_MIN from the start of this stage
  local deadline=$(( $(date +%s) + GUARD_MIN * 60 ))
  (
    while [ ! -f "$S/e25_rerun_done" ]; do
      u=$(sudo cat /run/systemd/shutdown/scheduled 2>/dev/null | sed -n 's/^USEC=//p')
      if [ -n "$u" ] && [ $(( u / 1000000 )) -lt $(( deadline - 300 )) ]; then
        sudo shutdown -c; sudo shutdown -h +$(( (deadline - $(date +%s)) / 60 )) "E25 guard (rerun)"
        st "rerun: moved a scheduled power-off to the guard deadline while the re-run waits or runs"
      fi
      sleep 20
    done ) &
  local hold=$!
  while pgrep -f "harbor run" >/dev/null; do sleep 60; done   # after every arm of the harness has ended
  .venv/bin/python scripts/explore_e25_harnesses.py errored --harness "$H" "$RUNS/eval" > "$lst" \
    || { touch "$S/e25_rerun_done"; kill "$hold" 2>/dev/null; return 1; }
  if [ ! -s "$lst" ]; then
    st "rerun: no infrastructure error to re-run"; touch "$S/e25_rerun_done"; kill "$hold" 2>/dev/null; return 0
  fi
  ensure_server || { touch "$S/e25_rerun_done"; kill "$hold" 2>/dev/null; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "E25 rerun guard" 2>/dev/null
  st "rerun: $(wc -l < "$lst") trial(s): $(tr '\n' ';' < "$lst")"
  watchdog "$S/e25_rerun_done" & local wd=$!
  for a in $(cut -d' ' -f1 "$lst" | sort -u); do
    if [ "$H" = tb ]; then
      inc=$(awk -v a="$a" '$1==a{printf "*%s ", $2}' "$lst")
      ( N=$N_CONC INCLUDE="$inc" run_arms "$RUNS/eval_rerun" "$a" ) &
    else
      f="$S/e25_${H}_rerun_$a.txt"; awk -v a="$a" '$1==a{print $2}' "$lst" > "$f"
      ( N=$N_CONC IDS_FILE="$f" run_arms "$RUNS/eval_rerun" "$a" ) &
    fi
    pids+=($!); sleep 5
  done
  for p in "${pids[@]}"; do wait "$p"; done
  touch "$S/e25_rerun_done"; kill "$wd" "$hold" 2>/dev/null
  aws s3 sync "$RUNS/eval_rerun" "$B/e25/$H/runs/eval_rerun" --only-show-errors --region "$REGION" || true
  st "rerun done: still errored $(.venv/bin/python scripts/explore_e25_harnesses.py errored --harness "$H" "$RUNS/eval_rerun" | wc -l); spend $(.venv/bin/python scripts/explore_e25_harnesses.py spend "$RUNS"/smoke* "$RUNS/eval" "$RUNS/eval_rerun" "$RUNS/eval_defect" 2>/dev/null)"
}

stage_final() {
  cd "$R" 2>/dev/null || true
  local u0 u1 n
  u0=$(cat "$S/e25_or_usage_start.txt" 2>/dev/null); u1=$(or_usage)
  .venv/bin/python scripts/explore_e25_harnesses.py spend "$RUNS"/smoke* "$RUNS/eval" "$RUNS/eval_rerun" "$RUNS/eval_defect" > "$S/e25_spend_logged.json" 2>/dev/null
  printf '{"logged": %s, "or_key_usage_start": "%s", "or_key_usage_end": "%s", "written": "%s"}\n' \
    "$(cat "$S/e25_spend_logged.json" 2>/dev/null || echo null)" "${u0:-}" "${u1:-}" "$(date -u +%FT%TZ)" > "$S/e25_spend.json"
  n=$(leak_check "$RUNS" /opt/jev/e25_box.log "$S")
  st "final: spend $(cat "$S/e25_spend.json"); leak check: $n file(s) contain a key value"
  for d in "$RUNS"/smoke* "$RUNS/eval" "$RUNS/eval_rerun" "$RUNS/eval_defect"; do
    [ -d "$d" ] && aws s3 sync "$d" "$B/e25/$H/runs/$(basename "$d")" --only-show-errors --region "$REGION"
  done
  cp /opt/jev/e25_box.log "$S/e25_box_log.txt" 2>/dev/null
  st "final: synced"
  if [ "$POWEROFF" = 1 ]; then st "powering off"; rm -f "$S/e25_driver_$$.pid"; sudo shutdown -h +1 "E25 done"; fi
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
