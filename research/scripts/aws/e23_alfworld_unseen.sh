#!/usr/bin/env bash
# Exploratory row E23, E11's checkpoint in ALFWorld's unseen rooms (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08
# "E23: the E11 checkpoint in ALFWorld's unseen rooms (valid_unseen)"; descriptive, outside every Holm table). One NEW
# GPU host launched by this driver from E3's clone image (tag Name=jev-e23-alfworld-unseen), SSM only. E11's frozen
# checkpoint (S3, sha256-checked against E11's record) served by serve_jev on :8081 at E11's tau-hat (no refit, no
# training); smoke on 2 calibration train games, the registered gate, then E11's three arms on the 134 valid_unseen
# games under E14's pause-and-resume rule. Gemini on the direct API (key from SSM Parameter Store into the stage's
# process only; never printed, never on disk).
#
#   LOCAL:  bash scripts/aws/e23_alfworld_unseen.sh launch [itype ...]   # one new host (refuses if a jev-e23 host
#                                                                         # exists); types in order (default g5.4xlarge
#                                                                         # g5.2xlarge g6e.2xlarge); user-data arms the
#                                                                         # power-off guard (GUARD_MIN, 230 min)
#           E23_IID=i-... bash scripts/aws/e23_alfworld_unseen.sh push | send "STAGES" | status [N] | stop | itype
#   BOX stages (in the order sent; a failed stage stops the chain, `final` always runs when sent):
#     rearm    idle stop (IDLE_MIN) + power-off guard (GUARD_END absolute UTC, else GUARD_MIN), host, GPU, disk
#     setup    the committed branch (bundle) in /opt/jev/e23; avenv (runner + alfworld 0.4.2 / textworld 1.7.0); hvenv
#              (serving) imports this checkout; E6a's data (the 48 calibration games); valid_unseen extracted from the
#              ALFWorld release archives (sha256 as in E6a's record) and the registered list sha256 checked; E11's
#              checkpoint from S3 into $CK, sha256 checked against E11's record
#     serve    serve_jev (E11's adapter + temperatures) on :8081 as e11-alfworld-4b
#     smoke    smoke.yaml (probe-gated, pause on overload) + the registered gate (projection)
#     eval     eval.yaml at the n the gate allows (probe-gated; paused on 503 / overload and resumed after two clean
#              probe rounds; one re-run of errored episodes after full coverage); evaluation window
#     final    stop the server, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e23_alfworld_unseen/box on the box -> s3://$BUCKET/e23_alfworld_unseen/box; runs ->
# .../runs/<run>.
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E23_IID=${E23_IID:-}
# every other project host (never touched by this driver): the never-touch instances, E11's stopped host (not
# restarted), E21's and E22's running hosts, and every earlier row's host
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
# the default VPC's subnets, us-east-1d first (E11's), then the others (as E21 / E22's drivers try them)
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e23-alfworld-unseen
BUNDLE=e23.bundle
S3OUT="$B/e23_alfworld_unseen"
S3CK="$B/checkpoints/e11_alfworld"
CAP_USD=${CAP_USD:-15}
GUARD_MIN=${GUARD_MIN:-230}          # 4 GPU-hours (registration (h)); GUARD_END (absolute UTC) wins when set
IDLE_MIN=${IDLE_MIN:-30}
WORKERS=${WORKERS:-8}
POWEROFF=${POWEROFF:-1}
NAME=e11-alfworld-4b
PORT=8081

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E23_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E23_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E23_IID "* ]] && { echo "refused: $E23_IID is not the E23 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E23_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E23_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e23-alfworld-unseen: power-off guard (4 GPU-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g5.4xlarge g5.2xlarge g6e.2xlarge)
  for IT in "${types[@]}"; do
    for SUBNET in $SUBNETS; do
      out=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$IT" --subnet-id "$SUBNET" --security-group-ids "$SG" \
        --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior stop --count 1 \
        --user-data "file://$ud" --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
        --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name,LaunchTime]' --output text 2>&1)
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/e23_alfworld_unseen.sh" > "$tmp/e23_alfworld_unseen.sh"   # the committed driver
  awsl s3 cp "$tmp/e23_alfworld_unseen.sh" "$B/code/e23_alfworld_unseen.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e23_alfworld_unseen.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e23_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e23_alfworld_unseen.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q GUARD_MIN=%q GUARD_END=%q IDLE_MIN=%q WORKERS=%q POWEROFF=%q E23_IID=%q\n' \
      "$CAP_USD" "$GUARD_MIN" "${GUARD_END:-}" "$IDLE_MIN" "$WORKERS" "$POWEROFF" "$E23_IID"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e23_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e23_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E23_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e23_driver.sh /tmp/e23_driver.sh --only-show-errors && bash /tmp/e23_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 stop-instances --instance-ids "$E23_IID" --output text --query 'StoppingInstances[].[InstanceId,CurrentState.Name]'
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E23_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime,StateTransitionReason]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,25p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
W=${W:-/opt/jev}
DRY_RUN=${DRY_RUN:-0}
cd "$W" || exit 1
imds() {
  [[ "$DRY_RUN" == 1 ]] && { echo "dry-run-$1"; return 0; }
  curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" "http://169.254.169.254/latest/meta-data/$1"
}
if [[ "$DRY_RUN" == 1 ]]; then MY_IID=${E23_IID:-i-dryrun}; else MY_IID=$(imds instance-id); fi
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E23_IID" && "$MY_IID" != "$E23_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E23 host ${E23_IID:-?}"; exit 1
fi
R=$W/e23
A=$R/results/explore/e23_alfworld_unseen/box
STATUS_DIR=$W/e23_status
HV=$W/hvenv
AV=$W/avenv
CK=$W/ck_e23
ZIPS=$W/alfworld_zips
STOP=runs/e23_STOP.json
EX=scripts/explore_e23_alfworld_unseen.py
PROBE_LOG=results/explore/e23_alfworld_unseen/box/api_probe_log.txt
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$STATUS_DIR"
# DRY_RUN=1: every heavy command (GPU, S3, network, runner) is written to $STATUS_DIR/dry_run.log instead of run
X() { if [[ "$DRY_RUN" == 1 ]]; then echo "+ $*" >> "$STATUS_DIR/dry_run.log"; return 0; fi; "$@"; }
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  X aws s3 cp "$STATUS_DIR/status.txt" "$S3OUT/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && X aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION"
  local d
  for d in e23_alfworld_smoke e23_alfworld_unseen_eval; do
    [[ -d "$R/runs/$d" ]] && X aws s3 sync "$R/runs/$d" "$S3OUT/runs/$d" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  true
}
PYA() { PYTHONPATH="$R/src" "$AV/bin/python" "$@"; }    # runner, probes, spend, readout (alfworld installed)
PYH() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }    # serving (torch)
base() { cat "$W/base_model_path.txt"; }
keys() {  # the direct Gemini API key(s) from SSM into this stage's environment only (never on disk, never printed)
  [[ "$DRY_RUN" == 1 ]] && return 0
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
probe_ok() {  # two consecutive clean probe rounds (E14's rule, registration (f)); waits up to PROBE_WAIT_MIN
  [[ "$DRY_RUN" == 1 ]] && return 0
  local end=$(( $(date +%s) + ${PROBE_WAIT_MIN:-60} * 60 )) clean=0
  while (( $(date +%s) < end )); do
    if PYA "$EX" probe --log "$PROBE_LOG" > /dev/null 2>&1; then clean=$((clean + 1)); else clean=0; fi
    (( clean >= 2 )) && return 0
    sleep $(( clean > 0 ? 30 : 300 ))
  done
  return 1
}
PAUSED_S=0
run_live() {  # run_live CONFIG RUN [runner args]: probe-gated, paused on 503 / overload, capped spend; then one re-run
  local cfg=$1 run=$2; shift 2
  local rc pid wpid spid round t0 ne
  for round in 1 2 3 4 5 6 7 8 9 10; do
    [[ -f "$STOP" ]] && { st "spend cap reached earlier; $cfg not run further"; return 3; }
    t0=$(date +%s)
    probe_ok || { st "probes not clean for ${PROBE_WAIT_MIN:-60} min; $cfg stops (round $round)"; return 4; }
    (( round > 1 )) && PAUSED_S=$(( PAUSED_S + $(date +%s) - t0 ))
    st "$cfg: start (round $round)"
    if [[ "$DRY_RUN" == 1 ]]; then X PYA -m jevharness.bench.runner "$cfg" --out runs "$@"; break; fi
    setsid env PYTHONPATH="$R/src" "$AV/bin/python" -m jevharness.bench.runner "$cfg" --out runs "$@" >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PYA "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --run "$run" --log "$PROBE_LOG" --every 30 >> "$A/watch.log" 2>&1 &
    wpid=$!
    ( while kill -0 "$pid" 2>/dev/null; do sleep 300; sync_out; done ) & spid=$!
    wait "$pid"; rc=$?
    kill "$wpid" "$spid" 2>/dev/null
    sync_out
    if [[ -f "runs/$run/PAUSED_overload.json" ]]; then
      st "$cfg paused on an overload error (round $round, runner exit $rc); waiting for clean probes"
      mv -f "runs/$run/PAUSED_overload.json" "$A/paused_${run}_round$round.json"
      continue
    fi
    break
  done
  [[ -f "$STOP" ]] && return 3
  [[ "$DRY_RUN" == 1 ]] && return 0
  ne=$(PYA "$EX" errored --run "runs/$run")
  if [[ "$ne" != 0 ]]; then
    st "$ne errored episode(s) in $run after full coverage: one re-run after clean probes"
    t0=$(date +%s)
    probe_ok || { st "probes not clean; re-run skipped"; return 0; }
    PAUSED_S=$(( PAUSED_S + $(date +%s) - t0 ))
    setsid env PYTHONPATH="$R/src" "$AV/bin/python" -m jevharness.bench.runner "$cfg" --out runs "$@" >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PYA "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --run "$run" --log "$PROBE_LOG" --every 30 --no-pause >> "$A/watch.log" 2>&1 &
    wpid=$!
    wait "$pid"; kill "$wpid" 2>/dev/null
  fi
  st "$cfg done; errored now $(PYA "$EX" errored --run "runs/$run"); $(PYA "$EX" spend --runs runs --log "$PROBE_LOG" --short 2>/dev/null)"
  sync_out
  [[ -f "$STOP" ]] && return 3
  return 0
}
stop_server() {
  local p
  if [[ -f "$W/e23_serve.pid" ]]; then
    while read -r p; do [[ -n "$p" ]] && kill "$p" 2>/dev/null; done < "$W/e23_serve.pid"
  fi
  rm -f "$W/e23_serve.pid"; [[ "$DRY_RUN" == 1 ]] || sleep 5
}
server_up() { [[ "$DRY_RUN" == 1 ]] || curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; }

stage_rearm() {
  if [[ -n "${GUARD_END:-}" ]]; then   # an absolute end (UTC): the GPU cap counts from launch
    GUARD_MIN=$(( ( $(date -u -d "$GUARD_END" +%s) - $(date -u +%s) ) / 60 ))
  fi
  (( GUARD_MIN > 0 )) || { echo "guard end already passed"; return 1; }
  X sudo shutdown -c 2>/dev/null; X sudo shutdown -h +"$GUARD_MIN" "jev-e23-alfworld-unseen: power-off guard (GPU cap)"
  if [[ "$DRY_RUN" != 1 ]]; then
    sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E23 chain / server / runner process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e23_[0-9]|jevharness|serve_jev|explore_e23' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e23-alfworld-unseen: idle $IDLE_MIN min"; fi
IDLE
    sudo chmod +x /usr/local/bin/jev-idle-stop
    sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
    echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
    echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
    echo "guard $GUARD_MIN min, idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU; $(df -BG / | awk 'NR==2{print $4}') free"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  fi
}
stage_setup() {
  local myb="$W/e23.$$.bundle"
  X aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  if [[ "$DRY_RUN" != 1 ]]; then
    [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
    cd "$R" || return 1
    git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
    rm -f "$myb"
  fi
  cd "$R" || return 1
  mkdir -p "$A" runs
  st "code at $(git log --oneline -1 | cut -c1-72)"
  grep -q "E23: the E11 checkpoint in ALFWorld's unseen rooms" docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E23"; return 1; }
  command -v uv > /dev/null || X bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  [[ -x "$AV/bin/python" ]] || X uv venv -q -p 3.11 "$AV" || return 1
  X uv pip install -q --python "$AV" -e ".[all]" "alfworld==0.4.2" "textworld==1.7.0" || return 1
  if [[ "$DRY_RUN" != 1 ]]; then
    PYA -c "import jevharness, alfworld, textworld, sys; p = jevharness.__file__; print(p); sys.exit(0 if p.startswith('$R/src') else 1)" \
      || { echo "the avenv does not import this checkout"; return 1; }
    PYH -c "import jevharness, torch, transformers, peft, sys; p = jevharness.__file__; print(p, torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
      > "$A/env.txt" 2>&1 || { cat "$A/env.txt"; echo "the hvenv does not import this checkout"; return 1; }
    uv pip freeze --python "$AV" > "$A/avenv_freeze.txt" 2>/dev/null
    cat "$A/env.txt"
    [[ -s "$W/base_model_path.txt" ]] || { echo "no base_model_path.txt"; return 1; }
    cp -f "$W/base_model_path.txt" "$A/base_model_path.txt"
    grep -q "1cfa9a7208912126459214e8b04321603b3df60c" "$W/base_model_path.txt" || { echo "base snapshot is not E11's"; return 1; }
  fi
  # E6a's data (the 48 calibration train games for the smoke; valid_seen comes with it and is not run)
  if [[ ! -d data/raw/alfworld_e6a ]]; then
    X bash -c "aws s3 cp '$B/e6a/data.tgz' - --region '$REGION' | tar xz -C '$R'" || { echo "E6a data download failed"; return 1; }
  fi
  # the ALFWorld release archives (E6a's record: sha256) from E11's S3 copy, then valid_unseen extracted and checked
  mkdir -p "$ZIPS"
  local k f
  for k in json twpddl; do
    f="$ZIPS/$k.zip"
    X aws s3 cp "$B/e11_alfworld/archives/$k.zip" "$f" --only-show-errors --region "$REGION" || { echo "download of $k failed"; return 1; }
  done
  X PYA "$EX" extract-unseen --json-zip "$ZIPS/json.zip" --twpddl-zip "$ZIPS/twpddl.zip" --data-dir data/raw/alfworld \
    > "$A/extract_unseen.log" 2>&1 || { cat "$A/extract_unseen.log"; return 1; }
  cat "$A/extract_unseen.log"
  X PYA "$EX" eval-list --check --data-dir data/raw/alfworld || return 1
  # E11's frozen checkpoint from S3 into a new directory; sha256 against E11's record before any use
  mkdir -p "$CK"
  X aws s3 sync "$S3CK/" "$CK/" --only-show-errors --region "$REGION" || { echo "checkpoint download failed"; return 1; }
  X PYA "$EX" verify-ckpt --dir "$CK" --out "$A/ckpt_verify.json" || { echo "checkpoint sha256 differs from E11's record"; return 1; }
  rm -f "$ZIPS"/*.zip
  df -BG / | awk 'NR==2{print "disk free: "$4}'
}
stage_serve() {
  cd "$R" || return 1
  stop_server
  [[ "$DRY_RUN" == 1 ]] && { X serve_jev "$NAME"; return 0; }
  PYA "$EX" verify-ckpt --dir "$CK" --out "$A/ckpt_verify.json" || { echo "checkpoint sha256 differs from E11's record"; return 1; }
  local args=(--model "$(base)" --name "$NAME" --port "$PORT" --adapter "$CK" --calibration "$CK/calibration.json")
  echo "serve_jev ${args[*]}" > "$A/serve.cmd"
  nohup env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$A/serve.out" 2>&1 &
  echo $! > "$W/e23_serve.pid"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" > "$A/serve_health.json"; then
      curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/serve_models.json"
      grep -q "\"$NAME\"" "$A/serve_models.json" || { echo "served model id is not $NAME"; return 1; }
      return 0
    fi
    kill -0 "$(cat "$W/e23_serve.pid")" 2>/dev/null || { echo "serve_jev died"; tail -20 "$A/serve.out"; return 1; }
    sleep 5
  done
  echo "serve_jev did not come up"; return 1
}
stage_smoke() {
  cd "$R" || return 1
  keys || return 1
  server_up || { echo "server not up"; return 1; }
  run_live configs/explore/e23_alfworld_unseen/smoke.yaml e23_alfworld_smoke
  local rc=$?
  X PYA "$EX" project --runs runs --log "$PROBE_LOG" --out "$A/projection.json" > "$A/projection.out" 2>&1
  st "smoke exit=$rc; projection: $(PYA -c "import json; print(json.load(open('$A/projection.json'))['decision'])" 2>/dev/null)"
  leak_check || return 1
  return $rc
}
stage_eval() {
  cd "$R" || return 1
  keys || return 1
  server_up || { echo "server not up"; return 1; }
  local dec ov=() rc t0 t1 mt=null
  dec=$(PYA -c "import json; print(json.load(open('$A/projection.json'))['decision'])" 2>/dev/null)
  [[ "$DRY_RUN" == 1 ]] && dec="run all 134"
  case "$dec" in
    "run all "[0-9]*) ov=() ;;
    "run the first "[0-9]*) ov=(--max-tasks "${dec##* }"); mt=${dec##* } ;;
    *) st "refusing eval: projection says '$dec'"; return 3 ;;
  esac
  printf 'step e23_alfworld_unseen_eval\nstarted %s\ncommit %s\nconfig eval.yaml %s\nprereg sha256 %s\ncap_usd %s\nworkers %s\nhost %s %s\n' \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$(sha256sum configs/explore/e23_alfworld_unseen/eval.yaml | cut -c1-16)" \
    "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" "$CAP_USD" "$WORKERS" "$MY_IID" "$(imds instance-type)" > "$A/START_eval.txt"
  t0=$(date -u +%FT%TZ)
  PAUSED_S=0
  run_live configs/explore/e23_alfworld_unseen/eval.yaml e23_alfworld_unseen_eval "${ov[@]}"
  rc=$?
  t1=$(date -u +%FT%TZ)
  printf '{"window_start": "%s", "window_end": "%s", "instance_type": "%s", "instance_id": "%s", "workers": %s, "run": "runs/e23_alfworld_unseen_eval", "max_tasks": %s, "paused_s": %s, "exit": %s}\n' \
    "$t0" "$t1" "$(imds instance-type)" "$MY_IID" "$WORKERS" "$mt" "$PAUSED_S" "$rc" > "$A/eval_window.json"
  st "eval exit=$rc; $(PYA "$EX" spend --runs runs --log "$PROBE_LOG" --short 2>/dev/null)"
  leak_check || return 1
  [[ -f "$STOP" ]] && return 1
  return $rc
}
stage_final() {
  stop_server
  [[ "$DRY_RUN" == 1 ]] || nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; X sudo shutdown -h +2 "jev-e23-alfworld-unseen: done"; fi
}

broke=0
for s in $STAGES; do
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $s"; continue; fi
  st "stage $s start (host $MY_IID)"
  t0=$(date +%s)
  mkdir -p "$STATUS_DIR"
  ( "stage_$s" ) > "$STATUS_DIR/stage_$s.log" 2>&1
  rc=$?
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/stage_$s.log" "$A/stage_$s.log"
  st "stage $s exit=$rc ($(( ($(date +%s) - t0) / 60 )) min)"
  if (( rc != 0 )) && [[ "$s" != final ]]; then st "stopping the chain after $s failed"; broke=1; break; fi
done
if (( broke )); then
  stop_server
  sync_out
  st "chain stopped; server stopped; the box stops itself after $IDLE_MIN idle minutes"
fi
st "box done [$STAGES]"
