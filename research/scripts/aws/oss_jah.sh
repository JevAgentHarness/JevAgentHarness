#!/usr/bin/env bash
# Open-source JevAgentHarness benchmarks (open_source/jevagentharness): Codex / pi / DeepSeek Harness on Terminal-Bench 2.0
# through the jah proxy, one CPU host (SSM only, no inbound), tag Name=jah-oss.
#
#   LOCAL: bash scripts/aws/oss_jah.sh launch [itype ...]     # refuses if a jah-oss host exists
#          bash scripts/aws/oss_jah.sh push                    # git bundle of the branch + this driver -> s3 code/
#          bash scripts/aws/oss_jah.sh send "STAGES"           # runs on the box, in order (env: ROW, RUN_SPEC, CAP, N)
#          bash scripts/aws/oss_jah.sh status [N] | start | stop | fetch
#   BOX stages:
#     setup   Docker CE, uv, Python 3.12 venv with harbor 0.24.0 + the jah package (from the bundle), guard + idle stop
#     guard   re-arm the power-off guard (GUARD_MIN) and the idle stop (IDLE_MIN) on a host set up earlier
#     update  fast-forward the checkout to the latest bundle and reinstall the package
#     proxy   (re)start `jah serve` (bench/configs/$CONFIG) on 0.0.0.0:8787 with budget CAP; keys from SSM, proxy only
#     run     every "harness:arm" in RUN_SPEC on TASKS (a split file) in parallel jobs, N trials each -> sync
#     final   sync, leak check, power off (POWEROFF=1)
# Keys: GEMINI_API_KEY (the pool ${SSM_PREFIX:?set SSM_PREFIX}/GEMINI_API_KEY_POOL, comma-separated, else ${SSM_PREFIX:?set SSM_PREFIX}/GEMINI_API_KEY) and the Jev key (${SSM_PREFIX:?set SSM_PREFIX}/OPENROUTER_API_KEY_JAH) are read from SSM
# into the proxy process only. Agents get a dummy key and the proxy URL, so no key enters a task container. Nothing here
# prints or writes a key; `final` greps the run directory for the key values (must be 0 files).
set -uo pipefail
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
TAG=${TAG:-jah-oss}     # a second host: TAG=jah-oss2
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
BUNDLE=jah.bundle
ROW=${ROW:-pilot}
GUARD_MIN=${GUARD_MIN:-2880}   # power-off guard, minutes from setup/guard
IDLE_MIN=${IDLE_MIN:-240}      # idle stop: minutes without harbor, jah or a container
POWEROFF=${POWEROFF:-0}

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN AWS_PAGER="" aws --region "$REGION" "$@"; }
iid() {
  local ids; ids=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [ "$(echo "$ids" | wc -w)" = 1 ] || { echo "expected one $TAG host, found: '$ids'" >&2; return 1; }
  echo "$ids"
}

case "$MODE" in
launch)
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  AMI=${AMI:-$(awsl ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
    --query Parameter.Value --output text)}
  ud=$(mktemp)
  cat > "$ud" <<EOF
#!/bin/bash
shutdown -h +$GUARD_MIN "$TAG: power-off guard ($GUARD_MIN min from boot)"
snap install aws-cli --classic || true
EOF
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(m7i.8xlarge m6i.8xlarge c7i.8xlarge)
  for IT in "${types[@]}"; do
    for s in $SUBNETS; do
      out=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$IT" --subnet-id "$s" --security-group-ids "$SG" \
        --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior stop --count 1 --user-data "file://$ud" \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":400,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG},{Key=Project,Value=jev}]" \
          "ResourceType=volume,Tags=[{Key=Name,Value=$TAG},{Key=Project,Value=jev}]" \
        --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name]' --output text 2>&1)
      rc=$?; echo "$IT $s: $out"
      (( rc == 0 )) && { rm -f "$ud"; exit 0; }
      [[ "$out" == *InsufficientInstanceCapacity* || "$out" == *Unsupported* || "$out" == *"not supported"* ]] || { rm -f "$ud"; exit 1; }
    done
  done
  rm -f "$ud"; exit 1 ;;
push)
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null || { echo "bundle failed"; exit 1; }
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors || exit 1
  git -C "$ROOT" show "$BRANCH:scripts/aws/oss_jah.sh" > "$tmp/oss_jah.sh"
  awsl s3 cp "$tmp/oss_jah.sh" "$B/code/oss_jah.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72)"
  rm -rf "$tmp"; exit 0 ;;
send)
  STAGES=${1:?usage: send "STAGES"}
  IID=$(iid) || exit 1
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo "mkdir -p /opt/jah && f=/opt/jah/driver_\$(date +%s%N).sh"
    echo "aws s3 cp $B/code/oss_jah.sh \$f --only-show-errors --region $REGION"
    printf 'export TAG=%q\n' "$TAG"
    printf 'export IDLE_MIN=%q\n' "$IDLE_MIN"
    printf 'export ROW=%q RUN_SPEC=%q TASKS=%q N=%q CAP=%q CONFIG=%q DATASET=%q GUARD_MIN=%q POWEROFF=%q EXTRA=%q PORT=%q\n' \
      "$ROW" "${RUN_SPEC:-}" "${TASKS:-bench/splits/tb2_dev.txt}" "${N:-4}" "${CAP:-50}" \
      "${CONFIG:-gemini_argon_flash.yaml}" "${DATASET:-terminal-bench@2.0}" "$GUARD_MIN" "$POWEROFF" \
      "${EXTRA:---agent-setup-timeout-multiplier 4}" "${PORT:-8787}"
    printf 'cd /opt/jah && setsid nohup bash "$f" box %q >> /opt/jah/box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/jah_driver.sh" --only-show-errors; rm -f "$tmp"
  awsl ssm send-command --instance-ids "$IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"aws s3 cp $B/code/jah_driver.sh /tmp/jah_driver.sh --only-show-errors && bash /tmp/jah_driver.sh\"]" \
    --query 'Command.CommandId' --output text
  exit 0 ;;
status)
  awsl s3 cp "$B/jah/status_$TAG.txt" - 2>/dev/null | tail -n "${1:-30}"; exit 0 ;;
start) IID=$(iid) || exit 1; awsl ec2 start-instances --instance-ids "$IID" --query 'StartingInstances[0].CurrentState.Name' --output text; exit 0 ;;
stop)  IID=$(iid) || exit 1; awsl ec2 stop-instances --instance-ids "$IID" --query 'StoppingInstances[0].CurrentState.Name' --output text; exit 0 ;;
fetch)
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  awsl s3 sync "$B/jah/$ROW/" "$ROOT/runs/jah/$ROW/" --only-show-errors --exclude "*/agent/sessions/*" --exclude "*/agent/pi/sessions/*" --exclude "*/agent/pi.txt" --exclude "*/agent/codex.txt" --exclude "*/agent/dsh.jsonl*" && echo "fetched runs/jah/$ROW"
  exit 0 ;;
box) ;;
*) sed -n 2,20p "$0"; exit 0 ;;
esac

# ------------------------------------------------------------------------------------------------ on the box (root)
STAGES=${1:?stages}
R=/opt/jah/repo; V=/opt/jah/venv; RUNS=/opt/jah/runs/$ROW; P=open_source/jevagentharness
export PATH="/root/.local/bin:$V/bin:$PATH"
log() { echo "$(date -u +%FT%TZ) [$ROW] $*" | tee -a /opt/jah/status.txt; awsl s3 cp /opt/jah/status.txt "$B/jah/status_$TAG.txt" --only-show-errors || true; }
ssm() { awsl ssm get-parameter --with-decryption --name "$1" --query Parameter.Value --output text; }
gemini_keys() { ssm ${SSM_PREFIX:?set SSM_PREFIX}/GEMINI_API_KEY_POOL 2>/dev/null || ssm ${SSM_PREFIX:?set SSM_PREFIX}/GEMINI_API_KEY; }
sync_runs() { awsl s3 sync "$RUNS" "$B/jah/$ROW/" --only-show-errors --exclude "*.tar*" || true; }

stage_guard() {
  shutdown -c 2>/dev/null; shutdown -h +"$GUARD_MIN" "jah-oss guard" || true
  # idle stop: power off when no harbor/jah process and no container ran for IDLE_MIN minutes
  local idle_s=$(( IDLE_MIN * 60 ))
  cat > /etc/cron.d/jah-idle <<EOF
*/5 * * * * root if pgrep -f "harbor run|jah serve" >/dev/null || [ -n "\$(docker ps -q)" ]; then date +\%s > /opt/jah/last_busy; elif [ \$(( \$(date +\%s) - \$(cat /opt/jah/last_busy 2>/dev/null || date +\%s) )) -gt $idle_s ]; then shutdown -h now "jah idle"; fi
EOF
  date +%s > /opt/jah/last_busy
  log "guard: power off in $GUARD_MIN min, idle stop after $IDLE_MIN min"
}

stage_setup() {
  if ! command -v docker >/dev/null; then
    apt-get update -qq && apt-get install -y -qq ca-certificates curl git jq >/dev/null
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable" \
      > /etc/apt/sources.list.d/docker.list
    apt-get update -qq && apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin >/dev/null
  fi
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  stage_update || return 1
  uv venv -q -p 3.12 "$V" && VIRTUAL_ENV=$V uv pip install -q "harbor==0.24.0" -e "$R/$P[dev]" || return 1
  (cd "$R/$P" && "$V/bin/python" -m pytest -q) || return 1
  stage_guard
  log "setup ok: $(docker --version | cut -c1-40); harbor $($V/bin/harbor --version 2>/dev/null)"
}

stage_update() {
  aws s3 cp "$B/code/$BUNDLE" /opt/jah/$BUNDLE --only-show-errors --region "$REGION" || return 1
  if [ -d "$R/.git" ]; then git -C "$R" fetch -q /opt/jah/$BUNDLE "$BRANCH" && git -C "$R" reset -q --hard FETCH_HEAD
  else git clone -q -b "$BRANCH" /opt/jah/$BUNDLE "$R"; fi
  [ -x "$V/bin/python" ] && VIRTUAL_ENV=$V uv pip install -q -e "$R/$P[dev]"
  log "checkout $(git -C "$R" log --oneline -1 | cut -c1-60)"
}

stage_proxy() {
  pkill -f "jah serve.*--port $PORT" 2>/dev/null; sleep 2
  mkdir -p "$RUNS/proxy"
  GEMINI_API_KEY=$(gemini_keys) OPENROUTER_API_KEY=$(ssm ${SSM_PREFIX:?set SSM_PREFIX}/OPENROUTER_API_KEY_JAH) \
    OPENAI_API_KEY=$(ssm ${SSM_PREFIX:?set SSM_PREFIX}/OPENAI_API_KEY) \
    setsid nohup "$V/bin/jah" serve -c "$R/$P/bench/configs/$CONFIG" --host 0.0.0.0 --port "$PORT" \
    --log-dir "$RUNS/proxy" --set "budget_usd=$CAP" > "$RUNS/proxy/serve.log" 2>&1 < /dev/null &
  for _ in $(seq 30); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 1; done
  curl -sf "localhost:$PORT/health" >/dev/null || { log "proxy failed: $(tail -3 "$RUNS/proxy/serve.log")"; return 1; }
  log "proxy up ($CONFIG, cap \$$CAP, gemini keys: $(gemini_keys | tr , "\n" | grep -c .)): $(head -1 "$RUNS/proxy/serve.log" | cut -c1-160)"
}

stage_run() {
  [ -n "${RUN_SPEC:-}" ] || { log "RUN_SPEC is empty"; return 1; }
  curl -sf "localhost:$PORT/health" >/dev/null || stage_proxy || return 1
  mkdir -p "$RUNS/jobs"
  local pids=()
  for spec in $RUN_SPEC; do
    h=${spec%%:*}; a=${spec#*:}
    ( cd "$R/$P" && HARNESS=$h ARM=$a TASKS="$R/$P/$TASKS" DATASET=$DATASET N=$N PORT=$PORT JOBS_DIR="$RUNS/jobs" \
        JOB="$h-$a" HARBOR="$V/bin/harbor" EXTRA="${EXTRA:-}" bash bench/run_harbor.sh > "$RUNS/jobs/$h-$a.log" 2>&1 ) &
    pids+=($!)
    log "started $h:$a on $TASKS (n=$N)"
    sleep "${STAGGER_S:-30}"     # Harbor downloads tasks into a shared cache: simultaneous starts can race on it
  done
  ( while sleep 300; do sync_runs; curl -s "localhost:$PORT/stats" > "$RUNS/proxy/stats.json"; done ) & local syncer=$!
  for p in "${pids[@]}"; do wait "$p"; done
  kill "$syncer" 2>/dev/null
  curl -s "localhost:$PORT/stats" > "$RUNS/proxy/stats.json"
  sync_runs
  log "run done: $RUN_SPEC; spent USD $(jq -r .spent_usd "$RUNS/proxy/stats.json")"
}

stage_final() {
  sync_runs
  local leaks=0 k
  local IFS_OLD=$IFS; IFS=", "; local gk=($(gemini_keys) $(ssm ${SSM_PREFIX:?set SSM_PREFIX}/GEMINI_API_KEY)); IFS=$IFS_OLD
  for k in "${gk[@]}" "$(ssm ${SSM_PREFIX:?set SSM_PREFIX}/OPENROUTER_API_KEY_JAH)" "$(ssm ${SSM_PREFIX:?set SSM_PREFIX}/OPENAI_API_KEY)"; do
    [ -n "$k" ] && leaks=$(( leaks + $(grep -rlF -- "$k" "$RUNS" 2>/dev/null | wc -l) ))
  done
  log "final: leak check $leaks files (must be 0)"
  [ "$POWEROFF" = 1 ] && { log "power off"; shutdown -h +1 "jah final"; }
  return 0
}

for st in $STAGES; do
  log "stage $st"
  "stage_$st" || { log "stage $st FAILED"; [ "$st" = final ] || { stage_final; exit 1; }; }
done
