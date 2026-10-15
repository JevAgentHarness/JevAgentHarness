#!/usr/bin/env bash
# Exploratory row E31, GRPO in the agent loop without SFT (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-10 "E31: GRPO in
# the agent loop without SFT (RL only, from the untrained base)"; descriptive, outside every Holm table). One NEW GPU host
# launched by this driver from E3's clone image (tag Name=jev-e31-rlonly), SSM only. v1's GRPO stage
# (configs/train/grpo_online_round5.yaml) from the untrained Qwen3-4B: configs/train/grpo_online_e31_rlonly.yaml (a fresh
# LoRA, KL to the base, 15 iterations, everything else v1's), then the ladder's measurements of the iteration-15 adapter:
# temperatures on v1 dev, L1 (E1's 3,991), C (Stage 2's common states), L2 shift (E21's corpus) and L3 shift (REPLACE at
# the tool choice, E21's 200 tasks). Gemini on the direct API (the key from SSM into the training and live stages'
# processes only, never printed or on disk); no other key.
#
#   LOCAL:  bash scripts/aws/e31_rl_only.sh launch [itype ...]   # one new host (refuses if a jev-e31-rlonly host exists);
#                                                                # default g6e.2xlarge, g6e.4xlarge, g6e.xlarge, every zone
#           E31_IID=i-... bash scripts/aws/e31_rl_only.sh push | send "STAGES" | status [N] | start | stop | itype | state
#   BOX stages (in the order sent; a failed rearm / code / setup / smoke / train / select stops the chain; `final` always
#   runs when sent):
#     rearm      idle stop (IDLE_MIN) + power-off guard (GUARD_MIN from the first boot), host record
#     code       the committed branch in /opt/jev/e31 (records already on the host are kept: untracked files)
#     setup      code; the harness venv imports this checkout; the base snapshot at revision 1cfa9a72; versions; GPU
#     smoke      the trainer with the simulated executor (1 iteration, 2 tasks x 2 episodes; no API call; never reported)
#     train      the registered run (registration (b)); watchdog: training spend cap, wall cap, degeneracy rule (c)
#     select     the iteration-15 adapter (else the last saved one, with its iteration) -> S3, sha256
#     temps      temperatures on data/train_v1/dev.jsonl.gz (calibrate.py --min-bucket 200)
#     v1dev      eval_decisions on the three v1 dev files (diagnostic, development data)
#     e1 | c | l2  served on :8081 (rl-only-e31): L1, C, L2 shift
#     l3         L3 shift: probes, smoke (339070-339071), gate, the row (200 tasks), one re-run of errored episodes
#     final      stop the server, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e31_rl_only/box -> s3://$BUCKET/e31_rl_only/box; runs -> e31_rl_only/runs; adapters ->
# s3://$BUCKET/checkpoints/e31_rl_only/ (no optimizer state).
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E31_IID=${E31_IID:-}
FORBIDDEN_FILE=configs/explore/e31_rl_only/forbidden_iids.txt   # every other instance of the account (registration (f))
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e31-rlonly
BUNDLE=e31.bundle
S3OUT="$B/e31_rl_only"
S3CK="$B/checkpoints/e31_rl_only"
CAP_USD=${CAP_USD:-45}               # registration (f): direct Gemini at most 45 USD in total
TRAIN_CAP_USD=${TRAIN_CAP_USD:-30}   # of which training (rollouts + greedy evaluations) at most 30 USD
TRAIN_MAX_MIN=${TRAIN_MAX_MIN:-390}  # training wall cap
GUARD_MIN=${GUARD_MIN:-590}          # 10 host-hours: the guard 9 h 50 min after the first boot
IDLE_MIN=${IDLE_MIN:-30}
POWEROFF=${POWEROFF:-1}
PORT=8081
NAME=rl-only-e31
BASE_REV=1cfa9a7208912126459214e8b04321603b3df60c
E21_Q_SHA=827a7678f87c586ae2da9e72d74a5a3c7fd0181e6fab5ba8b26639f661283397

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
forbidden_local() {
  local root; root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  grep -oE '^i-0[0-9a-f]{16}' "$root/$FORBIDDEN_FILE" | tr '\n' ' '
}
iid_ok() {
  [[ "$E31_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E31_IID is not set to an instance id"; return 1; }
  local forb; forb=$(forbidden_local)
  [[ -n "$forb" ]] || { echo "no forbidden list ($FORBIDDEN_FILE)"; return 1; }
  [[ " $forb " == *" $E31_IID "* ]] && { echo "refused: $E31_IID is not the E31 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E31_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E31_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e31-rlonly: power-off guard (10 host-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g6e.2xlarge g6e.4xlarge g6e.xlarge)
  for IT in "${types[@]}"; do
    for SUBNET in $SUBNETS; do
      out=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$IT" --subnet-id "$SUBNET" --security-group-ids "$SG" \
        --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior stop --count 1 \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=300,VolumeType=gp3,Throughput=500,Iops=6000,DeleteOnTermination=true}' \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled \
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/e31_rl_only.sh" > "$tmp/e31_rl_only.sh"   # the committed driver
  awsl s3 cp "$tmp/e31_rl_only.sh" "$B/code/e31_rl_only.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e31_rl_only.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e31_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e31_rl_only.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q TRAIN_CAP_USD=%q TRAIN_MAX_MIN=%q GUARD_MIN=%q IDLE_MIN=%q POWEROFF=%q E31_IID=%q PROBE_WAIT_MIN=%q FORBIDDEN_IIDS=%q\n' \
      "$CAP_USD" "$TRAIN_CAP_USD" "$TRAIN_MAX_MIN" "$GUARD_MIN" "$IDLE_MIN" "$POWEROFF" "$E31_IID" "${PROBE_WAIT_MIN:-60}" \
      "$(forbidden_local)"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e31_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e31_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E31_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e31_driver.sh /tmp/e31_driver.sh --only-show-errors && bash /tmp/e31_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E31_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype || "$MODE" == state ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E31_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime,StateReason.Code]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,30p' "$0"
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
if [[ -z "${FORBIDDEN_IIDS:-}" || " ${FORBIDDEN_IIDS:-} " == *" $MY_IID "* || -z "$E31_IID" || "$MY_IID" != "$E31_IID" ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E31 host ${E31_IID:-?} (or no forbidden list)"; exit 1
fi
R=$W/e31
A=$R/results/explore/e31_rl_only/box
STATUS_DIR=$W/e31_status
HV=$W/hvenv                 # the harness venv of E3's clone image (trainer, server, runner, replays)
CKT=$W/ck_e31/grpo          # the trainer's output: ckpt-00005, ckpt-00010 and the final (iteration-15) adapter
SEL=$W/ck_e31/selected      # the evaluated adapter (+ temperatures)
EX=scripts/explore_e31.py
TCFG=configs/train/grpo_online_e31_rlonly.yaml
CFG=configs/explore/e31_rl_only
STOP=runs/e31_STOP.json
PROBE_LOG=$A/api_probe_log.txt
TSPEND=$A/train_spend.json
V1=(data/train_v1/dev.jsonl.gz data/train_v1/dev_recovery.jsonl.gz data/train_v1/dev_knockout.jsonl.gz)
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  HF_HUB_DISABLE_TELEMETRY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$STATUS_DIR"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  aws s3 cp "$STATUS_DIR/status.txt" "$S3OUT/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION" \
    --exclude 's23/corpus.json' --exclude 's23/occurrences.jsonl.gz' --exclude 's23/questions.jsonl.gz' \
    --exclude 'c_shift/questions*' --exclude 'c_shift/occurrences.jsonl.gz' --exclude 'c_shift/corpus.json'
  local d
  for d in "$R"/runs/e1_bfcl_when2call_e31 "$R"/runs/e31_*; do
    [[ -d "$d" ]] && aws s3 sync "$d" "$S3OUT/runs/$(basename "$d")" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  true
}
PY() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }
qbase() { cat "$W/base_model_path.txt"; }
first_boot() { cat "$W/e31_first_boot" 2>/dev/null || date -d "$(uptime -s)" +%s; }
hours_used() { python3 -c "import time; print(round((time.time() - $(first_boot)) / 3600, 3))"; }
mins_left() { echo $(( GUARD_MIN - ( $(date +%s) - $(first_boot) ) / 60 )); }
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
leak_check() {  # no key value in any record, checkpoint directory or status file
  local n=0 k ks=${GEMINI_API_KEYS:-}
  for k in ${GEMINI_API_KEY:-} ${ks//,/ }; do
    [[ ${#k} -ge 16 ]] || continue
    n=$(( n + $(grep -rlF -- "$k" "$R/runs" "$A" "$STATUS_DIR" "$W/ck_e31" "$W/e31_box.log" 2>/dev/null | wc -l) ))
  done
  echo "key-leak files: $n"
  [[ "$n" == 0 ]]
}
smi_start() { nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 5000 > "$1" 2>/dev/null & SMI=$!; }
smi_stop() { [[ -n "${SMI:-}" ]] && kill "$SMI" 2>/dev/null; SMI=""; }
gz_copy() { python3 -c "import gzip,sys; d=open(sys.argv[1],'rb').read(); f=open(sys.argv[2],'wb'); g=gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0); g.write(d); g.close(); f.close()" "$1" "$2"; }

# ------------------------------------------------------------------------------------------------ server
stop_servers() {
  local p
  if [[ -f "$W/e31_serve.pid" ]]; then
    p=$(cat "$W/e31_serve.pid"); kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$p" 2>/dev/null; kill -9 "$p" 2>/dev/null
  fi
  rm -f "$W/e31_serve.pid"
  pkill -f 'jevharness.training.serve_jev' 2>/dev/null
  sleep 3
  true
}
server_up() { curl -sf -m 10 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; }
serve_c() {  # the evaluated adapter + its temperatures on :8081 (as E21's ladder rows: serve_jev in the harness venv)
  server_up && grep -q "\"$NAME\"" <(curl -sf -m 10 "http://127.0.0.1:$PORT/v1/models") && return 0
  stop_servers
  [[ -s "$SEL/adapter_model.safetensors" && -s "$SEL/calibration.json" ]] || { echo "no selected adapter / temperatures"; return 1; }
  local args=(--model "$(qbase)" --name "$NAME" --port "$PORT" --adapter "$SEL" --calibration "$SEL/calibration.json")
  echo "$(date -u +%FT%TZ) serve_jev ${args[*]}" >> "$A/logs/serve.cmd"
  setsid nohup env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev "${args[@]}" \
    > "$A/logs/serve.out" 2>&1 < /dev/null &
  echo $! > "$W/e31_serve.pid"
  for _ in $(seq 1 240); do
    if curl -sf -m 10 "http://127.0.0.1:$PORT/health" > "$A/logs/serve_health.json" 2>/dev/null; then
      curl -sf -m 10 "http://127.0.0.1:$PORT/v1/models" > "$A/logs/serve_models.json"
      grep -q "\"$NAME\"" "$A/logs/serve_models.json" || { echo "served model id is not $NAME"; return 1; }
      printf '{"server_up": "%s", "name": "%s", "port": %s, "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" "$NAME" "$PORT" \
        "$(imds instance-type)" >> "$A/serve_windows.jsonl"
      return 0
    fi
    kill -0 "$(cat "$W/e31_serve.pid")" 2>/dev/null || { echo "serve_jev died"; tail -30 "$A/logs/serve.out"; return 1; }
    sleep 5
  done
  echo "serve_jev did not come up"; return 1
}
spec_c() { printf '{"type": "jev", "name": "%s", "base_url": "http://127.0.0.1:%s", "model": "%s", "api_key": "", "timeout": 60.0, "price_model": "jevagentrl-4b"}' "$NAME" "$PORT" "$NAME"; }

# ------------------------------------------------------------------------------------------------ live (E21's pause rule)
probe_ok() {  # two consecutive clean probe rounds; waits up to PROBE_WAIT_MIN
  local end=$(( $(date +%s) + ${PROBE_WAIT_MIN:-60} * 60 )) clean=0
  while (( $(date +%s) < end )); do
    if PY "$EX" probe --log "$PROBE_LOG" > /dev/null 2>&1; then clean=$((clean + 1)); else clean=0; fi
    (( clean >= 2 )) && return 0
    sleep $(( clean > 0 ? 30 : 300 ))
  done
  return 1
}
run_live() {  # run_live CONFIG RUN_DIR [jevbench args]: probe-gated, paused on 503 / overload, capped; one re-run of errors
  local cfg=$1 run=$2 pid wpid spid round; shift 2
  local W_ARGS=(--runs runs --cap "$CAP_USD" --stop "$STOP" --log "$PROBE_LOG" --train-spend "$TSPEND" --every 30)
  for round in 1 2 3 4 5 6 7 8; do
    [[ -f "$STOP" ]] && { st "spend cap reached earlier; $cfg not run further"; return 3; }
    probe_ok || { st "probes not clean; $cfg paused (round $round)"; return 4; }
    st "$cfg: start (round $round) $*"
    setsid env PYTHONPATH="$R/src" JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$cfg" --out runs "$@" >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PY "$EX" watch --pid "$pid" "${W_ARGS[@]}" >> "$A/watch.log" 2>&1 &
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
  local ne
  ne=$(PY "$EX" errored --run "$run")
  if [[ "$ne" != 0 ]]; then
    st "$ne errored episode(s) in $run: one re-run after clean probes"
    probe_ok || { st "probes not clean; re-run skipped"; return 0; }
    setsid env PYTHONPATH="$R/src" JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$cfg" --out runs "$@" >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PY "$EX" watch --pid "$pid" "${W_ARGS[@]}" --no-pause >> "$A/watch.log" 2>&1 &
    wpid=$!
    wait "$pid"; kill "$wpid" 2>/dev/null
  fi
  st "$cfg done; errored now $(PY "$EX" errored --run "$run"); $(PY "$EX" spend --runs runs --log "$PROBE_LOG" --train-spend "$TSPEND" --short 2>/dev/null)"
  sync_out
  [[ -f "$STOP" ]] && return 3
  return 0
}

# ------------------------------------------------------------------------------------------------ stages
stage_rearm() {
  [[ -s "$W/e31_first_boot" ]] || date -d "$(uptime -s)" +%s > "$W/e31_first_boot"
  local left; left=$(mins_left)
  (( left > 0 )) || { echo "guard already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$left" "jev-e31-rlonly: power-off guard (host-hour cap)"
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E31 chain / trainer / server / runner / replay process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e31_[0-9]|jevbench|grpo_online|eval_decisions|training.calibrate|serve_jev|explore_e31|explore_e21_shift|stage2_corpus|decision_eval|uv pip|pip install' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e31-rlonly: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $left min left, idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU; $(df -BG / | awk 'NR==2{print $4}') free; used $(hours_used) h"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
}
stage_code() {
  local myb="$W/e31.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A/logs" "$A/eval" "$A/train" runs cache
  st "code at $(git log --oneline -1 | cut -c1-72)"
  grep -q 'E31: GRPO in the agent loop without SFT' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E31"; return 1; }
}
stage_setup() {
  stage_code || return 1
  PY -c "import jevharness, torch, sys; p = jevharness.__file__; print(p, torch.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') and torch.cuda.is_available() else 1)" \
    > "$A/env_hvenv.txt" 2>&1 || { cat "$A/env_hvenv.txt"; echo "the hvenv does not import this checkout (or no CUDA)"; return 1; }
  [[ -x "$HV/bin/jevbench" ]] || { echo "no jevbench in the hvenv"; return 1; }
  [[ -s "$(qbase)/config.json" && "$(qbase)" == *"$BASE_REV"* ]] || { echo "the base snapshot is not at revision $BASE_REV: $(qbase)"; return 1; }
  PY - "$A/versions.json" "$(qbase)" <<'EOF' || return 1
import importlib, json, platform, sys
v = {"python": platform.python_version(), "base_model_path": sys.argv[2]}
for m in ("torch", "transformers", "peft", "accelerate", "safetensors", "numpy", "pydantic"):
    try:
        v[m] = importlib.import_module(m).__version__
    except Exception as e:  # noqa: BLE001
        v[m] = f"missing ({e})"
import torch
v["cuda"] = torch.version.cuda
v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
v["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else None
json.dump(v, open(sys.argv[1], "w"), indent=1)
print(json.dumps(v))
sys.exit(0 if (v["gpu_mem_gb"] or 0) >= 40 else 1)     # registration (f): one L40S 48 GB (v1's GRPO GPU)
EOF
  for f in "${V1[@]}" data/train_v1/dev.jsonl.gz results/stage2/s23/questions.jsonl.gz results/explore/e21_shift/c/questions.jsonl.gz; do
    [[ -s "$f" ]] || { echo "missing $f"; return 1; }
  done
  [[ "$(sha256sum results/explore/e21_shift/c/questions.jsonl.gz | cut -d' ' -f1)" == "$E21_Q_SHA" ]] || { echo "E21 corpus sha256 differs"; return 1; }
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s", "gpu": "%s", "mins_left": %s, "first_boot": %s}\n' \
    "$MY_IID" "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" \
    "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" "$(mins_left)" "$(first_boot)" > "$A/host.json"
  df -BG / | awk 'NR==2{print "disk free: "$4}'
  sync_out
}
stage_smoke() {  # registration (b): the trainer with the simulated executor, no API call, a scratch directory
  cd "$R" || return 1
  stop_servers
  local so=$W/e31_smoke sim='{provider: simulated, model: scripted, error_rate: 0.1, simulated_latency_ms: 0}' t0 rc
  rm -rf "$so"
  t0=$(date +%s)
  PY -m jevharness.training.grpo_online --config "$TCFG" --model "$(qbase)" --out "$so" --iterations 1 \
    --set grpo.tasks_per_iter=2 grpo.group_size=2 suite.n=4 eval_every=0 save_every=0 workers=4 \
    "executor=$sim" "executors={flash: $sim}" > "$A/logs/smoke.log" 2>&1
  rc=$?
  python3 - "$so/metrics.jsonl" "$A/smoke.json" "$rc" "$(( $(date +%s) - t0 ))" <<'EOF'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1])] if __import__("os").path.exists(sys.argv[1]) else []
r = rows[-1] if rows else {}
keep = ("iter", "episodes", "decisions", "trainable_decisions", "updated_decisions", "frac_zero_std_groups", "success",
        "loss", "kl", "time_rollout_s", "time_update_s", "cascade_forced_defer_frac", "defer_rate")
out = {"exit": int(sys.argv[3]), "seconds": int(sys.argv[4]), "row": {k: r.get(k) for k in keep},
       "note": "pipeline smoke with the simulated executor (registration (b)); never reported"}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out))
EOF
  rm -rf "$so"
  sync_out
  return $rc
}
stage_train() {  # registration (b), (c): the registered run with its three guards
  cd "$R" || return 1
  stop_servers
  if [[ -s "$A/train_run.json" ]]; then echo "already trained: $(cat "$A/train_run.json")"; return 0; fi
  keys || return 1
  mkdir -p "$CKT" "$A/train"
  local t0 t1 rc pid wpid spid it
  rm -f "$A/train_stop.json"
  t0=$(date +%s)
  printf 'stage train\nstarted %s\ncommit %s\nconfig %s %s\nprereg sha256 %s\nbase %s\ntraining cap usd %s\nwall cap min %s\nhost %s %s\n' \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$TCFG" "$(sha256sum "$TCFG" | cut -c1-16)" \
    "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" "$(qbase)" "$TRAIN_CAP_USD" "$TRAIN_MAX_MIN" "$MY_IID" \
    "$(imds instance-type)" > "$A/START_train.txt"
  smi_start "$A/logs/smi_train.csv"
  setsid env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.grpo_online --config "$TCFG" --model "$(qbase)" \
    --out "$CKT" > "$A/logs/train.log" 2>&1 < /dev/null &
  pid=$!
  PY "$EX" trainwatch --pid "$pid" --metrics "$CKT/metrics.jsonl" --cap "$TRAIN_CAP_USD" --max-min "$TRAIN_MAX_MIN" \
    --stop "$A/train_stop.json" --every 60 >> "$A/logs/trainwatch.log" 2>&1 &
  wpid=$!
  ( while kill -0 "$pid" 2>/dev/null; do
      sleep 300
      cp -f "$CKT/metrics.jsonl" "$A/train/metrics.jsonl" 2>/dev/null
      st "train: $(python3 -c "import json; r=[json.loads(x) for x in open('$CKT/metrics.jsonl')][-1]; print('iteration', r.get('iter'), 'success', r.get('success'), 'zero-std', r.get('frac_zero_std_groups'), 'updated', r.get('updated_decisions'))" 2>/dev/null)"
      sync_out
    done ) & spid=$!
  wait "$pid"; rc=$?
  t1=$(date +%s)
  kill "$spid" 2>/dev/null
  for _ in $(seq 1 90); do kill -0 "$wpid" 2>/dev/null || break; sleep 1; done
  kill "$wpid" 2>/dev/null
  smi_stop
  cp -f "$CKT/metrics.jsonl" "$A/train/metrics.jsonl" 2>/dev/null
  cp -f "$CKT/grpo_config.yaml" "$A/train/grpo_config.yaml" 2>/dev/null
  [[ -s "$CKT/rollouts.jsonl" ]] && gz_copy "$CKT/rollouts.jsonl" "$A/train/rollouts.jsonl.gz"
  PY "$EX" trainspend --metrics "$CKT/metrics.jsonl" --out "$TSPEND" > /dev/null
  it=$(python3 -c "import json; print(max([json.loads(x).get('iter', 0) for x in open('$CKT/metrics.jsonl')] or [0]))" 2>/dev/null || echo 0)
  printf '{"exit": %s, "start": "%s", "end": "%s", "minutes": %s, "iteration_reached": %s, "stopped": %s, "final_adapter": %s, "instance_type": "%s", "peak_gpu_mem_mib": %s}\n' \
    "$rc" "$(date -u -d "@$t0" +%FT%TZ)" "$(date -u -d "@$t1" +%FT%TZ)" "$(python3 -c "print(round(($t1 - $t0) / 60, 1))")" "$it" \
    "$([[ -s "$A/train_stop.json" ]] && echo true || echo false)" "$([[ -s "$CKT/adapter_model.safetensors" ]] && echo true || echo false)" \
    "$(imds instance-type)" "$(awk -F', ' '{if ($3 > m) m = $3} END {print m + 0}' "$A/logs/smi_train.csv" 2>/dev/null || echo null)" \
    > "$A/train_run.json"
  leak_check || { unset GEMINI_API_KEY GEMINI_API_KEYS; st "train: KEY LEAK CHECK FAILED"; return 1; }
  unset GEMINI_API_KEY GEMINI_API_KEYS
  # adapters (no optimizer state): the final one and the saved iterations
  local d
  for d in "$CKT" "$CKT"/ckpt-*; do
    [[ -s "$d/adapter_model.safetensors" ]] || continue
    aws s3 sync "$d" "$S3CK/$( [[ "$d" == "$CKT" ]] && echo final || basename "$d")" --only-show-errors --region "$REGION" \
      --exclude '*' --include 'adapter_*' --include 'decision_config.json' --include 'tokenizer*' --include 'chat_template.jinja' \
      --include 'special_tokens_map.json' --include 'vocab.json' --include 'merges.txt' --include 'added_tokens.json'
  done
  st "train: exit $rc, iteration $it, $(cat "$A/train_run.json" | tr -d '\n' | cut -c1-200); spend $(python3 -c "import json; print(round(json.load(open('$TSPEND'))['total'], 3))")"
  sync_out
  [[ -s "$CKT/adapter_model.safetensors" ]] || ls -d "$CKT"/ckpt-* > /dev/null 2>&1 || return 1
  return 0
}
stage_select() {  # registration (c), (f): the iteration-15 adapter, else the last saved one (reported with its iteration)
  cd "$R" || return 1
  local src it
  if [[ -s "$CKT/adapter_model.safetensors" ]]; then src=$CKT
  else src=$(ls -d "$CKT"/ckpt-* 2>/dev/null | sort | tail -1); fi
  [[ -n "$src" && -s "$src/adapter_model.safetensors" ]] || { echo "no adapter to evaluate"; return 1; }
  it=$(python3 -c "import json; print(json.load(open('$src/decision_config.json')).get('iteration'))")
  rm -rf "$SEL" && mkdir -p "$SEL"
  cp -f "$src"/adapter_* "$src"/decision_config.json "$SEL/" && cp -f "$src"/tokenizer* "$src"/chat_template.jinja "$SEL/" 2>/dev/null
  (cd "$SEL" && sha256sum adapter_model.safetensors adapter_config.json decision_config.json) > "$A/selected_sha256.txt"
  printf '{"source": "%s", "iteration": %s, "adapter_sha256": "%s", "at": "%s"}\n' "$src" "$it" \
    "$(sha256sum "$SEL/adapter_model.safetensors" | cut -d' ' -f1)" "$(date -u +%FT%TZ)" > "$A/selected.json"
  st "select: iteration $it from $src"
  sync_out
}
stage_temps() {
  cd "$R" || return 1
  stop_servers
  PY -m jevharness.training.calibrate --model "$(qbase)" --adapter "$SEL" --dev data/train_v1/dev.jsonl.gz \
    --out "$SEL/calibration.json" --min-bucket 200 > "$A/logs/calibrate.log" 2>&1 || { tail -5 "$A/logs/calibrate.log"; return 1; }
  cp -f "$SEL/calibration.json" "$A/e31_calibration.json"
  sha256sum "$SEL/calibration.json" >> "$A/selected_sha256.txt"
  aws s3 sync "$SEL" "$S3CK/selected" --only-show-errors --region "$REGION"
  st "temps: $(python3 -c "import json; d=json.load(open('$SEL/calibration.json')); print({k: round(v, 3) for k, v in (d.get('temperatures') or d).items() if isinstance(v, (int, float))})" 2>/dev/null)"
  sync_out
}
stage_v1dev() {  # diagnostic, development data: as round 5's evaluate (uncalibrated, n-perm 0)
  cd "$R" || return 1
  stop_servers
  PY -m jevharness.training.eval_decisions --model "$(qbase)" --adapter "$SEL" --dev "${V1[@]}" --n-perm 0 --max-len 4096 \
    --batch-size 16 --out "$A/eval/e31_v1.json" > "$A/logs/eval_v1.log" 2>&1 || { tail -5 "$A/logs/eval_v1.log"; return 1; }
  sync_out
}
stage_e1() {
  cd "$R" || return 1
  serve_c > "$A/logs/serve_e1_start.log" 2>&1 || { cat "$A/logs/serve_e1_start.log"; return 1; }
  local k
  for k in 1 2; do   # two invocations as E1 (the second resumes any failed request from the cache)
    PY -m jevharness.bench.decision_eval "$CFG/e1_e31.yaml" --out runs --only e31 > "$A/logs/e1_$k.log" 2>&1
    st "e1 invocation $k exit=$?"
  done
  st "e1: $(wc -l < runs/e1_bfcl_when2call_e31/s1_e31.jsonl 2>/dev/null) answers"
  sync_out
}
stage_c() {
  cd "$R" || return 1
  serve_c > "$A/logs/serve_c_start.log" 2>&1 || { cat "$A/logs/serve_c_start.log"; return 1; }
  mkdir -p "$A/s23"
  local f k
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz; do cp -f "results/stage2/s23/$f" "$A/s23/$f" || return 1; done
  for k in 1 2 3; do   # the replay resumes
    PY scripts/stage2_corpus.py replay --corpus "$A/s23" --name e31 --s1 "$(spec_c)" --cache cache/s1_e31_c.sqlite \
      > "$A/logs/replay_c_$k.log" 2>&1
  done
  st "C replayed: $(wc -l < "$A/s23/answers_e31.jsonl" 2>/dev/null) answers"
  sync_out
}
stage_l2() {
  cd "$R" || return 1
  serve_c > "$A/logs/serve_l2_start.log" 2>&1 || { cat "$A/logs/serve_l2_start.log"; return 1; }
  mkdir -p "$A/c_shift"
  local f k rc=1
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz; do cp -f "results/explore/e21_shift/c/$f" "$A/c_shift/$f" || return 1; done
  for k in 1 2; do
    PY scripts/explore_e21_shift.py replay --corpus "$A/c_shift" --slug e31 --variant main --s1 "$(spec_c)" \
      --cache cache/s1_e31_l2.sqlite --workers 4 > "$A/logs/replay_l2_$k.log" 2>&1
    rc=$?
    (( rc == 0 )) && break
  done
  st "L2 shift exit=$rc: $(PY scripts/explore_e21_shift.py quickc --corpus "$A/c_shift" --slug e31 2>/dev/null)"
  sync_out
  return $rc
}
stage_l3() {  # registration (d) 4: smoke, gate, the row
  cd "$R" || return 1
  local g nfit rc t0 mt=null
  if (( $(mins_left) < 45 )); then st "l3: not run ($(mins_left) min left of the guard)"; return 0; fi
  [[ -f "$STOP" ]] && { st "l3: not run (the spend cap was reached earlier)"; return 3; }
  [[ -s "$TSPEND" ]] || { st "l3: no training spend record"; return 1; }
  keys || return 1
  serve_c > "$A/logs/serve_l3_start.log" 2>&1 || { st "l3: did not serve"; unset GEMINI_API_KEY GEMINI_API_KEYS; return 1; }
  run_live "$CFG/smoke_shift.yaml" runs/e31_live_smoke_shift || { st "l3: smoke failed"; unset GEMINI_API_KEY GEMINI_API_KEYS; return 1; }
  g=$(PY "$EX" gate --smoke runs/e31_live_smoke_shift --runs runs --n 200 --cap "$CAP_USD" --log "$PROBE_LOG" --train-spend "$TSPEND")
  echo "{\"row\": \"l3_shift\", \"gate\": $g}" >> "$A/gates.jsonl"
  nfit=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['n_fit'])" "$g" 2>/dev/null || echo 0)
  st "l3 gate: $g"
  (( nfit > 0 )) || { st "l3: the gate refuses the row"; unset GEMINI_API_KEY GEMINI_API_KEYS; return 3; }
  printf 'row l3_shift\nstarted %s\ncommit %s\nconfig %s %s\nprereg sha256 %s\ncap_usd %s\nhost %s %s\n' "$(date -u +%FT%TZ)" \
    "$(git rev-parse HEAD)" "$CFG/live_shift.yaml" "$(sha256sum "$CFG/live_shift.yaml" | cut -c1-16)" \
    "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" "$CAP_USD" "$MY_IID" "$(imds instance-type)" > "$A/START_l3.txt"
  t0=$(date -u +%FT%TZ)
  if (( nfit < 200 )); then mt=$nfit; run_live "$CFG/live_shift.yaml" runs/e31_live_shift --max-tasks "$nfit"
  else run_live "$CFG/live_shift.yaml" runs/e31_live_shift; fi
  rc=$?
  printf '{"row": "l3_shift", "window_start": "%s", "window_end": "%s", "instance_type": "%s", "instance_id": "%s", "max_tasks": %s, "exit": %s}\n' \
    "$t0" "$(date -u +%FT%TZ)" "$(imds instance-type)" "$MY_IID" "$mt" "$rc" > "$A/window_l3.json"
  leak_check || { unset GEMINI_API_KEY GEMINI_API_KEYS; st "l3: KEY LEAK CHECK FAILED"; return 1; }
  unset GEMINI_API_KEY GEMINI_API_KEYS
  sync_out
  return $rc
}
stage_final() {
  stop_servers
  printf '{"host_hours": %s, "at": "%s", "instance_type": "%s", "first_boot": %s}\n' "$(hours_used)" "$(date -u +%FT%TZ)" \
    "$(imds instance-type)" "$(first_boot)" > "$A/host_final.json"
  sync_out
  st "final: used $(hours_used) h; $(PY "$EX" spend --runs runs --log "$PROBE_LOG" --train-spend "$TSPEND" --short 2>/dev/null)"
  if [[ "$POWEROFF" == 1 ]]; then sudo shutdown -h +1 "jev-e31-rlonly: done"; fi
}

st "chain start: $STAGES (host $MY_IID, used $(hours_used) h)"
set -- $STAGES
while (( $# > 0 )); do
  s=$1; shift
  t0=$(date +%s)
  cd "$R" 2>/dev/null || cd "$W"      # every stage runs from the checkout (relative paths), as setup leaves it
  if declare -F "stage_$s" > /dev/null; then "stage_$s" > "$STATUS_DIR/stage_$s.log" 2>&1; rc=$?
  else echo "unknown stage $s" > "$STATUS_DIR/stage_$s.log"; rc=1; fi
  [[ -d "$A/logs" ]] && cp -f "$STATUS_DIR/stage_$s.log" "$A/logs/stage_$s.log" 2>/dev/null
  st "stage $s exit=$rc ($(( ($(date +%s) - t0) / 60 )) min): $(tail -2 "$STATUS_DIR/stage_$s.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  if (( rc != 0 )) && [[ " rearm code setup smoke train select " == *" $s "* ]]; then
    st "chain stopped at $s"
    [[ " $* " == *" final "* ]] && stage_final
    exit 1
  fi
done
st "chain end"
