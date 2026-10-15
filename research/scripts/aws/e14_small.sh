#!/usr/bin/env bash
# Exploratory row E14, a sub-1B decision model with a scoring head (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07
# "E14: a sub-1B decision model with a scoring head" and "E14: pause and resume on API overload"; descriptive, outside
# every Holm table). One NEW GPU host launched by this driver from E3's clone image (tag Name=jev-e14-small), SSM only.
# Qwen3.5-0.8B + a fresh Clef joint schema head trained with Unsloth 2026.10.2 (its own venv, run only) on the agent's
# states (A), on Unsloth's generic mix (G) and, if the cap allows, on both (AG); then L1 (E1's 3,991 decisions), L2
# (corpus C), the offline calibration-state replay, the same-session latency against the frozen 4B, and the live rows
# (Flash on the direct Gemini API; the key from SSM into the live stages' process only, never printed, never on disk).
#
#   LOCAL:  bash scripts/aws/e14_small.sh launch [itype ...]   # one new host (refuses if a jev-e14-small host exists)
#           E14_IID=i-... bash scripts/aws/e14_small.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages (in the order sent; a failed stage stops the chain, `final` always runs when sent):
#     rearm      idle stop (IDLE_MIN) + power-off guard (GUARD_MIN from the first boot), host record
#     setup      the committed branch in /opt/jev/e14; the hvenv imports it; Unsloth's venv; frozen 4B verified
#     data       A's records (SFT v1's exact input, sha256-checked) + the calibration requests; G's mix in background
#     train_a | train_g | train_ag     Unsloth's from-LM recipe (scripts/train/e14_clef.py train)
#     eval_a | eval_g | eval_ag        served on :8083: E1 (L1), corpus C (L2), calibration-state answers; Unsloth's sets
#     base_sets  the fresh-head base on Unsloth's three sets (G's reproduction check)
#     latency    harness_final.py bench c = 1 / 4: the frozen 4B (:8081, serve_jev --fast), then e14a (and e14g)
#     serve_a    arm A on :8083 (kept up for the live stages)
#     live_a | live_b | live_ag        live configs (probe-gated; watchdog pauses on 503 / overload, caps the spend)
#     final      stop the servers, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e14_small/box on the box -> s3://$BUCKET/e14_small/box; runs -> s3://$BUCKET/e14_small/runs;
# merged checkpoints -> s3://$BUCKET/checkpoints/e14_small/<arm>/.
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E14_IID=${E14_IID:-}
# every other project host: never touched by this driver
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e14-small
BUNDLE=e14.bundle
S3OUT="$B/e14_small"
CAP_USD=${CAP_USD:-40}
CAP_H=${CAP_H:-10}
GUARD_MIN=${GUARD_MIN:-590}          # 10 GPU-hours cap (registration (i)): the guard 9 h 50 min after the first boot
IDLE_MIN=${IDLE_MIN:-30}
POWEROFF=${POWEROFF:-1}
PORT=8083
PORT4B=8081

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E14_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E14_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E14_IID "* ]] && { echo "refused: $E14_IID is not the E14 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E14_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E14_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e14-small: power-off guard (10 GPU-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g5.2xlarge g5.4xlarge g5.8xlarge)
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/e14_small.sh" > "$tmp/e14_small.sh"   # the committed driver
  awsl s3 cp "$tmp/e14_small.sh" "$B/code/e14_small.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e14_small.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e14_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e14_small.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q CAP_H=%q GUARD_MIN=%q IDLE_MIN=%q POWEROFF=%q E14_IID=%q NO_SECOND=%q KEEP_SERVER=%q LAT_ARMS=%q AG_EPOCHS=%q WAIT_MAX_MIN=%q PROBE_WAIT_MIN=%q\n' \
      "$CAP_USD" "$CAP_H" "$GUARD_MIN" "$IDLE_MIN" "$POWEROFF" "$E14_IID" "${NO_SECOND:-0}" "${KEEP_SERVER:-0}" "${LAT_ARMS:-A}" "${AG_EPOCHS:-1}" "${WAIT_MAX_MIN:-300}" "${PROBE_WAIT_MIN:-45}"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e14_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e14_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E14_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e14_driver.sh /tmp/e14_driver.sh --only-show-errors && bash /tmp/e14_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E14_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E14_IID" --output text \
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
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E14_IID" && "$MY_IID" != "$E14_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E14 host ${E14_IID:-?}"; exit 1
fi
R=$W/e14
A=$R/results/explore/e14_small/box
STATUS_DIR=$W/e14_status
HV=$W/hvenv
EV=$W/e14venv
D=$W/e14_data
CKE=$W/ck_e14
CK=$W/ck
PLAN=results/stage2/rl_eval_plan.json
STOP=runs/e14_STOP.json
EX=scripts/explore_e14_small.py
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$STATUS_DIR"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  aws s3 cp "$STATUS_DIR/status.txt" "$S3OUT/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION" --exclude 's23/corpus.json' \
    --exclude 's23/occurrences.jsonl.gz' --exclude 's23/questions.jsonl.gz' --exclude 'inputs/*'
  local d
  for d in "$R"/runs/e1_bfcl_when2call_e14 "$R"/runs/e14_*; do
    [[ -d "$d" ]] && aws s3 sync "$d" "$S3OUT/runs/$(basename "$d")" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  true
}
PY() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }
EPY() { PYTHONPATH="$R/src" "$EV/bin/python" "$@"; }
plan() { (cd "$R" && PY -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"); }
base4b() { cat "$W/base_model_path.txt"; }
frozen_dir() { echo "$CK/$(plan "d['rows']['frozen']['dir']")"; }
frozen_cal() { echo "$CK/$(plan "d['rows']['frozen']['calibration']")"; }
hours_used() {  # host hours since the first boot recorded by rearm
  local b; b=$(cat "$W/e14_first_boot" 2>/dev/null || date -d "$(uptime -s)" +%s)
  python3 -c "import time; print(round((time.time() - $b) / 3600, 3))"
}
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
leak_check() {  # no key value in any record or status file
  local n=0 k ks=${GEMINI_API_KEYS:-}
  for k in ${GEMINI_API_KEY:-} ${ks//,/ }; do
    [[ ${#k} -ge 16 ]] || continue
    n=$(( n + $(grep -rlF -- "$k" "$R/runs" "$A" "$STATUS_DIR" 2>/dev/null | wc -l) ))
  done
  echo "key-leak files: $n"
  [[ "$n" == 0 ]]
}
stop_port() {
  local f="$W/e14_serve_$1.pid" p
  if [[ -f "$f" ]]; then
    p=$(cat "$f"); kill "$p" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -9 "$p" 2>/dev/null
  fi
  rm -f "$f"
}
wait_up() {  # wait_up PORT NAME TAG
  local port=$1 name=$2 tag=$3
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$port/health" > "$A/logs/serve_${tag}_health.json"; then
      curl -sf "http://127.0.0.1:$port/v1/models" > "$A/logs/serve_${tag}_models.json"
      grep -q "\"$name\"" "$A/logs/serve_${tag}_models.json" || { echo "served model id is not $name"; return 1; }
      printf '{"server_up": "%s", "name": "%s", "port": %s, "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" "$name" "$port" \
        "$(imds instance-type)" >> "$A/serve_windows.jsonl"
      return 0
    fi
    kill -0 "$(cat "$W/e14_serve_$port.pid")" 2>/dev/null || { echo "server died"; tail -30 "$A/logs/serve_$tag.out"; return 1; }
    sleep 5
  done
  echo "server did not come up"; return 1
}
engine_default() {  # v2 once the registered check passed (Deviations "E14: serving-engine fix" (d)), else v1
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$A/engine_check.json')).get('pass') else 1)" 2>/dev/null \
    && echo v2 || echo v1
}
serve_arm() {  # serve_arm ARM [v1|v2]: the arm's merged checkpoint behind serve_jev --backend clef on :8083
  local arm=$1 eng=${2:-$(engine_default)} name="e14-${1,,}" flags=()
  [[ "$eng" == v2 ]] && flags=(--clef-coalesce --clef-warmup 2)
  stop_port "$PORT"
  [[ -s "$CKE/$arm/final/joint_head.safetensors" ]] || { echo "no final checkpoint for $arm"; return 1; }
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then echo "port $PORT is served by another process"; return 1; fi
  echo "$(date -u +%FT%TZ) serve_jev --backend clef --model $CKE/$arm/final --name $name --port $PORT ${flags[*]} (engine $eng)" \
    >> "$A/logs/serve_$name.cmd"
  nohup env PYTHONPATH="$R/src" "$EV/bin/python" -m jevharness.training.serve_jev --backend clef --model "$CKE/$arm/final" \
    --name "$name" --port "$PORT" "${flags[@]}" > "$A/logs/serve_${name}_$eng.out" 2>&1 &
  echo $! > "$W/e14_serve_$PORT.pid"
  wait_up "$PORT" "$name" "${name}_$eng"
}
serve_4b() {
  stop_port "$PORT4B"
  verify_frozen || return 1
  local args=(--model "$(base4b)" --adapter "$(frozen_dir)" --calibration "$(frozen_cal)" --name jevagentrl-4b
              --port "$PORT4B" --fast)
  echo "serve_jev ${args[*]}" > "$A/logs/serve_4b.cmd"
  nohup env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$A/logs/serve_4b.out" 2>&1 &
  echo $! > "$W/e14_serve_$PORT4B.pid"
  wait_up "$PORT4B" jevagentrl-4b 4b
}
spec_of() {
  printf '{"type": "jev", "name": "%s", "base_url": "http://127.0.0.1:%s", "model": "%s", "api_key": "", "timeout": 120.0}' \
    "$1" "$2" "$1"
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
smi_start() { nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 > "$1" 2>/dev/null & SMI=$!; }
smi_stop() { [[ -n "${SMI:-}" ]] && kill "$SMI" 2>/dev/null; SMI=""; }

stage_wait_prev() {  # wait until every older E14 chain on this host has ended (a queued chain; WAIT_MAX_MIN at most)
  local me t end=$(( $(date +%s) + ${WAIT_MAX_MIN:-300} * 60 ))
  me=$(basename "$0" .sh); me=${me#e14_}
  while (( $(date +%s) < end )); do
    local older=0
    for t in $(ps -eo args | grep -oE '/opt/jev/e14_[0-9]+\.sh box' | grep -oE '[0-9]{6,}' | sort -u); do
      [[ "$t" < "$me" ]] && older=1
    done
    (( older == 0 )) && return 0
    sleep 30
  done
  echo "an older chain is still running after ${WAIT_MAX_MIN:-300} min"; return 1
}
stage_rearm() {
  [[ -s "$W/e14_first_boot" ]] || date -d "$(uptime -s)" +%s > "$W/e14_first_boot"
  local used_min left
  used_min=$(( ( $(date +%s) - $(cat "$W/e14_first_boot") ) / 60 ))
  left=$(( GUARD_MIN - used_min ))
  (( left > 0 )) || { echo "guard already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$left" "jev-e14-small: power-off guard (GPU cap)"
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E14 chain / server / runner / trainer process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e14_[0-9]|jevbench|e14_clef|explore_e14|stage2_corpus|decision_eval|e5_cal_replay|harness_final|uv pip|pip install' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e14-small: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $left min left, idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU; $(df -BG / | awk 'NR==2{print $4}') free; used $(hours_used) h"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}
stage_setup() {
  local myb="$W/e14.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A/logs" "$A/eval" "$A/cal" "$A/latency/bench" runs
  st "code at $(git log --oneline -1 | cut -c1-72)"
  grep -q 'E14: a sub-1B decision model with a scoring head' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E14"; return 1; }
  PY -c "import jevharness, torch, sys; p = jevharness.__file__; print(p, torch.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_hvenv.txt" 2>&1 || { cat "$A/env_hvenv.txt"; echo "the hvenv does not import this checkout"; return 1; }
  command -v uv > /dev/null || bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  if [[ ! -x "$EV/bin/python" ]] || ! "$EV/bin/python" -c "import unsloth" > /dev/null 2>&1; then
    uv venv -q -p 3.11 "$EV" || return 1
    timeout 45m uv pip install -q --python "$EV" "unsloth==2026.10.2" fastapi uvicorn httpx pydantic pyyaml tqdm \
      python-dotenv openai > "$A/logs/e14venv_install.log" 2>&1 || { tail -20 "$A/logs/e14venv_install.log"; return 1; }
    timeout 20m uv pip install -q --python "$EV" flash-linear-attention >> "$A/logs/e14venv_install.log" 2>&1 \
      || echo "flash-linear-attention not installed (the torch path is used)" | tee -a "$A/logs/e14venv_install.log"
  fi
  EPY -c "import unsloth, torch, transformers, peft, trl; from unsloth import FastDecisionModel, DecisionTrainer; print('unsloth', unsloth.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'trl', trl.__version__, 'cuda', torch.cuda.is_available())" \
    > "$A/env_e14venv.txt" 2>&1 || { tail -20 "$A/env_e14venv.txt"; return 1; }
  EPY -c "import fla; print('fla', fla.__version__)" >> "$A/env_e14venv.txt" 2>&1 || true
  "$EV/bin/pip" freeze > "$A/e14venv_freeze.txt" 2>/dev/null || uv pip freeze --python "$EV" > "$A/e14venv_freeze.txt"
  cat "$A/env_e14venv.txt"
  [[ -s "$W/base_model_path.txt" ]] || { echo "no base_model_path.txt"; return 1; }
  if ! verify_frozen; then
    st "the frozen 4B is absent or different in $CK; fetching"
    aws s3 sync "$(plan "d['rows']['frozen']['s3']")" "$(frozen_dir)" --only-show-errors --region "$REGION" || return 1
    verify_frozen || return 1
  fi
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s", "gpu": "%s"}\n' "$MY_IID" \
    "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" \
    "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" > "$A/host.json"
  df -BG / | awk 'NR==2{print "disk free: "$4}'
}
stage_data() {
  cd "$R" || return 1
  mkdir -p "$D" "$A/inputs"
  PY scripts/train/e14_clef.py build-a --out "$D" > "$A/logs/build_a.log" 2>&1 || { tail -5 "$A/logs/build_a.log"; return 1; }
  cp -f "$D/build_a.json" "$A/build_a.json"
  aws s3 cp "$B/e5_small/inputs/cal_requests.jsonl.gz" "$A/inputs/cal_requests.jsonl.gz" --only-show-errors --region "$REGION" || return 1
  sha256sum "$A/inputs/cal_requests.jsonl.gz" > "$A/cal_requests.sha256"
  if [[ ! -s "$D/build_g.json" ]]; then   # G's mix (downloads; CPU) beside A's training
    (cd "$R" && EPY scripts/train/e14_clef.py build-g --out "$D" > "$A/logs/build_g.log" 2>&1 \
      && cp -f "$D/build_g.json" "$A/build_g.json" && st "G's mix built: $(python3 -c "import json; d=json.load(open('$D/build_g.json')); print(d['rows'], 'rows; sources', d['sources'], 'skipped', list(d['skipped']))")" \
      || st "G's mix build FAILED (logs/build_g.log)") &
  fi
  sync_out
}
train_arm() {  # train_arm ARM [extra args]
  local arm=$1; shift
  cd "$R" || return 1
  stop_port "$PORT"
  local rc t0
  t0=$(date +%s)
  smi_start "$A/logs/train_smi_$arm.csv"
  EPY scripts/train/e14_clef.py train --arm "$arm" --data "$D" --out "$CKE" --hours-used "$(hours_used)" --cap-h "$CAP_H" \
    --v1-out "$A/eval/e14${arm,,}_v1.json" "$@" > "$A/logs/train_$arm.log" 2>&1
  rc=$?
  smi_stop
  cp -f "$CKE/$arm/train_record.json" "$A/train_record_$arm.json" 2>/dev/null
  printf '{"arm": "%s", "exit": %s, "minutes": %s}\n' "$arm" "$rc" "$(( ($(date +%s) - t0) / 60 ))" >> "$A/train_runs.jsonl"
  (( rc == 0 )) || { tail -30 "$A/logs/train_$arm.log"; return 1; }
  aws s3 sync "$CKE/$arm/final" "$B/checkpoints/e14_small/$arm/final" --only-show-errors --region "$REGION"
  aws s3 sync "$CKE/$arm" "$B/checkpoints/e14_small/$arm" --only-show-errors --region "$REGION" --exclude 'final/*' \
    --exclude '_run/*' --exclude '*optimizer*'
  st "train $arm: $(python3 -c "import json; d=json.load(open('$A/train_record_$arm.json')); print('epochs', [(e['tag'], e['minutes'], e['dev']['accuracy']) for e in d['epochs']], 'selected', d['selected_epoch'], 'cal', d.get('calibration_dev', {}).get('accuracy'))" 2>/dev/null)"
  sync_out
}
stage_train_a() { train_arm A $( [[ "${NO_SECOND:-0}" == 1 ]] && echo --no-second ); }
stage_train_g() {
  [[ -s "$D/build_g.json" ]] || { local end=$(( $(date +%s) + 1800 )); while [[ ! -s "$D/build_g.json" ]] && (( $(date +%s) < end )) && pgrep -f 'e14_clef.py build-g' > /dev/null; do sleep 20; done; }
  [[ -s "$D/build_g.json" ]] || { echo "G's mix is not built"; return 1; }
  train_arm G --no-second
}
stage_train_ag() { train_arm AG --epochs "${AG_EPOCHS:-1}" $( [[ "${AG_EPOCHS:-1}" == 1 ]] && echo --no-second ); }
eval_arm() {  # eval_arm ARM: L1 (E1), L2 (C), the calibration-state answers, through the served arm
  local arm=$1 key="e14${1,,}" name="e14-${1,,}" k rc
  cd "$R" || return 1
  serve_arm "$arm" || return 1
  for k in 1 2; do   # two invocations as E1 (the second resumes any failed request from the cache)
    PY -m jevharness.bench.decision_eval configs/explore/e14_small/e1_e14.yaml --out runs --only "$key" > "$A/logs/e1_${key}_$k.log" 2>&1
    st "e1 $key invocation $k exit=$?"
  done
  mkdir -p "$A/s23"
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz; do cp -f "results/stage2/s23/$f" "$A/s23/$f" || return 1; done
  for k in 1 2 3; do   # the replay resumes
    PY scripts/stage2_corpus.py replay --corpus "$A/s23" --name "$key" --s1 "$(spec_of "$name" "$PORT")" \
      --cache "cache/s1_e14_${key}_c.sqlite" > "$A/logs/replay_c_${key}_$k.log" 2>&1
  done
  st "corpus C $key: $(wc -l < "$A/s23/answers_$key.jsonl" 2>/dev/null) answers"
  for k in 1 2; do
    PY scripts/e5_cal_replay.py answer --requests "$A/inputs/cal_requests.jsonl.gz" --url "http://127.0.0.1:$PORT" \
      --name "$name" --out "$A/cal/answers_$key.jsonl" > "$A/logs/cal_replay_${key}_$k.log" 2>&1
    rc=$?
    st "calibration-state replay $key invocation $k exit=$rc ($(wc -l < "$A/cal/answers_$key.jsonl" 2>/dev/null) answers)"
    (( rc == 0 )) && break
  done
  curl -sf "http://127.0.0.1:$PORT/health" > "$A/logs/serve_${key}_health_end.json"
  sync_out
}
evalsets_arm() {
  local arm=$1
  cd "$R" || return 1
  [[ -s "$D/build_g.json" ]] || { st "evalsets $arm: G's sets not built; skipped"; return 0; }
  EPY scripts/train/e14_clef.py evalsets --ckpt "$CKE/$arm/final" --data "$D" --out "$A/eval/evalsets_$arm.json" \
    > "$A/logs/evalsets_$arm.log" 2>&1 || { tail -5 "$A/logs/evalsets_$arm.log"; return 1; }
  st "Unsloth's sets $arm: $(python3 -c "import json; d=json.load(open('$A/eval/evalsets_$arm.json')); print({k: v['accuracy'] for k, v in d['sets'].items()}, 'mean', round(d['mean_accuracy'], 4))")"
}
stage_eval_a() { eval_arm A; }
stage_eval_g() { eval_arm G; }
stage_eval_ag() { eval_arm AG; }
stage_sets_a() { stop_port "$PORT"; evalsets_arm A; }
stage_sets_g() { stop_port "$PORT"; evalsets_arm G; }
stage_sets_ag() { stop_port "$PORT"; evalsets_arm AG; }
stage_base_sets() {
  cd "$R" || return 1
  stop_port "$PORT"
  [[ -s "$D/build_g.json" ]] || { echo "G's sets not built"; return 1; }
  EPY scripts/train/e14_clef.py evalsets --base --data "$D" --out "$A/eval/evalsets_base.json" > "$A/logs/evalsets_base.log" 2>&1 \
    || { tail -5 "$A/logs/evalsets_base.log"; return 1; }
  st "Unsloth's sets, fresh-head base: $(python3 -c "import json; d=json.load(open('$A/eval/evalsets_base.json')); print({k: v['accuracy'] for k, v in d['sets'].items()})")"
}
bench_one() {  # bench_one LABEL PORT: harness_final.py bench c = 1 (25 episodes) and c = 4 (100)
  local label=$1 port=$2 c lim
  for c in 1 4; do
    lim=(); [[ $c == 1 ]] && lim=(--limit 25)
    smi_start "$A/latency/smi_${label}_c$c.csv"
    PY scripts/harness_final.py bench --url "http://127.0.0.1:$port" --label "$label" --path new --concurrency "$c" \
      --out "$A/latency/bench" "${lim[@]}" >> "$A/latency/bench.log" 2>&1
    local rc=$?
    smi_stop
    (( rc == 0 )) || return 1
    st "latency $label bench c=$c done"
  done
  curl -sf "http://127.0.0.1:$port/v1/models" > "$A/latency/models_$label.json"
}
stage_latency() {  # one session, one model on the GPU at a time: the frozen 4B, then the E14 arms
  cd "$R" || return 1
  stop_port "$PORT"; stop_port "$PORT4B"
  serve_4b || return 1
  bench_one 4b "$PORT4B" || return 1
  stop_port "$PORT4B"
  local arm
  for arm in ${LAT_ARMS:-A}; do
    serve_arm "$arm" || return 1
    bench_one "e14${arm,,}" "$PORT" || return 1
    stop_port "$PORT"
  done
  sync_out
}
stage_serve_a() { cd "$R" && serve_arm A; }
# ---------------------------------------------------------------- the serving-engine fix (Deviations "E14: serving-engine fix")
devcheck() {  # devcheck ENGINE: SFT v1 dev records + E1's 299-item dev prefix through arm A served with ENGINE
  local eng=$1
  cd "$R" || return 1
  serve_arm A "$eng" || return 1
  PY "$EX" devcheck --url "http://127.0.0.1:$PORT" --data "$D/dev_a.jsonl.gz" --out "$A/eval/devcheck_$eng.jsonl" \
    > "$A/logs/devcheck_$eng.log" 2>&1 || { tail -5 "$A/logs/devcheck_$eng.log"; return 1; }
  PY -m jevharness.bench.decision_eval configs/explore/e14_small/e1_dev_check.yaml --out runs --only "e14a_$eng" \
    --max-items 300 > "$A/logs/e1dev_$eng.log" 2>&1 || { tail -5 "$A/logs/e1dev_$eng.log"; return 1; }
  curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/logs/models_e14a_$eng.json"
  stop_port "$PORT"
  st "devcheck $eng: $(cat "$A/logs/devcheck_$eng.log" | tail -1)"
  sync_out
}
stage_devcheck_v1() { devcheck v1; }
stage_kernels() {  # the causal_conv1d kernel, only from a published wheel, 20 minutes at most (registration (b) 3)
  if EPY -c "import causal_conv1d" > /dev/null 2>&1; then st "kernels: causal_conv1d already present"; return 0; fi
  if timeout 20m uv pip install -q --python "$EV" --only-binary :all: causal-conv1d > "$A/logs/kernels.log" 2>&1 \
     && EPY -c "import causal_conv1d, torch; print('causal_conv1d', causal_conv1d.__version__)" >> "$A/logs/kernels.log" 2>&1; then
    st "kernels: $(tail -1 "$A/logs/kernels.log")"
  else
    uv pip uninstall -q --python "$EV" causal-conv1d > /dev/null 2>&1
    st "kernels: no causal_conv1d wheel installed (reference path stays): $(tail -2 "$A/logs/kernels.log" | tr '\n' ' ' | cut -c1-200)"
  fi
  "$EV/bin/pip" freeze > "$A/e14venv_freeze_v2.txt" 2>/dev/null || uv pip freeze --python "$EV" > "$A/e14venv_freeze_v2.txt"
  return 0
}
stage_devcheck_v2() {
  devcheck v2 || return 1
  PY "$EX" compare-engines --dev-v1 "$A/eval/devcheck_v1.jsonl" --dev-v2 "$A/eval/devcheck_v2.jsonl" \
    --e1-v1 runs/e14_e1dev_check/s1_e14a_v1.jsonl --e1-v2 runs/e14_e1dev_check/s1_e14a_v2.jsonl \
    --out "$A/engine_check.json" > "$A/logs/engine_check.log" 2>&1
  st "engine check: $(tail -1 "$A/logs/engine_check.log" | cut -c1-400)"
  sync_out
  return 0   # reported either way; engine_default reads the verdict
}
stage_e1_v2() {  # E1's 3,991 decisions once more through engine v2 (re-timing; answers compared with v1's)
  cd "$R" || return 1
  [[ "$(engine_default)" == v2 ]] || { st "e1_v2: the engine check did not pass; not re-timed"; return 0; }
  serve_arm A v2 || return 1
  local t0 t1
  t0=$(date +%s.%N)
  PY -m jevharness.bench.decision_eval configs/explore/e14_small/e1_e14.yaml --out runs --only e14a_v2 > "$A/logs/e1_e14a_v2.log" 2>&1
  t1=$(date +%s.%N)
  printf '{"engine": "v2", "start_epoch": %s, "end_epoch": %s, "decisions": 3991}\n' "$t0" "$t1" > "$A/e1_v2_window.json"
  stop_port "$PORT"
  st "e1 e14a v2 done ($(python3 -c "print(round($t1 - $t0, 1))") s)"
  sync_out
}
stage_latency_v2() {  # engine v2 on the request stream (c = 1 / 4), then the 4B again at c = 1 (drift check)
  cd "$R" || return 1
  [[ "$(engine_default)" == v2 ]] || { st "latency_v2: the engine check did not pass; not re-timed"; return 0; }
  stop_port "$PORT"; stop_port "$PORT4B"
  serve_arm A v2 || return 1
  bench_one e14a_v2 "$PORT" || return 1
  stop_port "$PORT"
  serve_4b || return 1
  smi_start "$A/latency/smi_4b_end_c1.csv"
  PY scripts/harness_final.py bench --url "http://127.0.0.1:$PORT4B" --label 4b_end --path new --concurrency 1 --limit 25 \
    --out "$A/latency/bench" >> "$A/latency/bench.log" 2>&1
  smi_stop
  stop_port "$PORT4B"
  st "latency v2 and the 4B drift check done"
  sync_out
}
probe_ok() {  # two consecutive clean probe rounds (registration amendment (b)); waits up to PROBE_WAIT_MIN
  local end=$(( $(date +%s) + ${PROBE_WAIT_MIN:-45} * 60 )) clean=0
  while (( $(date +%s) < end )); do
    if PY "$EX" probe --log "$A/api_probe_log.txt" > /dev/null 2>&1; then clean=$((clean + 1)); else clean=0; fi
    (( clean >= 2 )) && return 0
    sleep $(( clean > 0 ? 30 : 300 ))
  done
  return 1
}
run_live() {  # run_live CONFIG RUN_DIR: probe-gated, paused on 503 / overload, capped spend; then one re-run of errors
  local cfg=$1 run=$2 rc pid wpid round
  for round in 1 2 3 4 5 6 7 8; do
    [[ -f "$STOP" ]] && { st "spend cap reached earlier; $cfg not run further"; return 3; }
    probe_ok || { st "probes not clean; $cfg paused (round $round)"; return 4; }
    st "$cfg: start (round $round)"
    setsid env PYTHONPATH="$R/src" JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$cfg" --out runs >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PY "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --every 30 >> "$A/watch.log" 2>&1 &
    wpid=$!
    ( while kill -0 "$pid" 2>/dev/null; do sleep 300; sync_out; done ) & local spid=$!
    wait "$pid"; rc=$?
    kill "$wpid" "$spid" 2>/dev/null
    sync_out
    if [[ -f "$run/PAUSED_overload.json" ]]; then
      st "$cfg paused on an overload error (round $round); waiting for clean probes"
      rm -f "$run/PAUSED_overload.json"
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
    setsid env PYTHONPATH="$R/src" JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$cfg" --out runs >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PY "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --every 30 --no-pause >> "$A/watch.log" 2>&1 &
    wpid=$!
    wait "$pid"; kill "$wpid" 2>/dev/null
  fi
  st "$cfg done; errored now $(PY "$EX" errored --run "$run"); $(PY "$EX" spend --runs runs --short 2>/dev/null)"
  sync_out
  return 0
}
gate() {  # gate SMOKE_RUN N_EPISODES: the smoke's mean $ per episode x N x 1.5 + spend so far <= CAP_USD
  PY "$EX" gate --runs runs --smoke "$1" --n "$2" --cap "$CAP_USD" > "$A/logs/gate_$(basename "$1").json" 2>&1
}
stage_live_a() {  # REPLACE at the tool choice, arm A (registration (g) (a))
  cd "$R" || return 1
  keys || return 1
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || serve_arm A || return 1
  run_live configs/explore/e14_small/live_smoke_a.yaml runs/e14_live_smoke_a || return 1
  gate runs/e14_live_smoke_a 200 || { st "gate refuses live_a: $(cat "$A/logs/gate_e14_live_smoke_a.json")"; return 3; }
  run_live configs/explore/e14_small/live_a.yaml runs/e14_live_a
  leak_check || return 1
}
stage_live_b() {  # the final harness with arm A at its tau-hat (registration (g) (b))
  cd "$R" || return 1
  keys || return 1
  grep -q 'status: development' configs/harness/explore_e14_final.yaml && ! grep -q 'tau: pending' configs/harness/explore_e14_final.yaml \
    || { echo "the E14 profile has no fitted thresholds"; return 1; }
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || serve_arm A || return 1
  run_live configs/explore/e14_small/live_smoke_b.yaml runs/e14_live_smoke_b || return 1
  gate runs/e14_live_smoke_b 200 || { st "gate refuses live_b: $(cat "$A/logs/gate_e14_live_smoke_b.json")"; return 3; }
  printf '{"window_start": "%s", "instance_type": "%s", "instance_id": "%s", "workers": 1, "run": "runs/e14_live_b"}\n' \
    "$(date -u +%FT%TZ)" "$(imds instance-type)" "$MY_IID" > "$A/live_b_window.json"
  run_live configs/explore/e14_small/live_b.yaml runs/e14_live_b
  local rc=$?
  printf '{"window_end": "%s"}\n' "$(date -u +%FT%TZ)" >> "$A/live_b_window.json"
  leak_check || return 1
  return $rc
}
stage_live_g() {  # REPLACE with arm G (only if G is reproduced; registration (d), (g))
  cd "$R" || return 1
  keys || return 1
  serve_arm G || return 1
  run_live configs/explore/e14_small/live_smoke_g.yaml runs/e14_live_smoke_g || return 1
  gate runs/e14_live_smoke_g 200 || { st "gate refuses live_g: $(cat "$A/logs/gate_e14_live_smoke_g.json")"; return 3; }
  run_live configs/explore/e14_small/live_a_g.yaml runs/e14_live_a_g
  local rc=$?
  stop_port "$PORT"
  leak_check || return 1
  return $rc
}
stage_final() {
  stop_port "$PORT"; stop_port "$PORT4B"
  nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  printf '{"final_utc": "%s", "instance_id": "%s", "instance_type": "%s", "hours_used": %s}\n' "$(date -u +%FT%TZ)" "$MY_IID" \
    "$(imds instance-type)" "$(hours_used)" > "$A/host_final.json"
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; sudo shutdown -h +2 "jev-e14-small: done"; fi
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
if [[ "${KEEP_SERVER:-0}" != 1 ]]; then stop_port "$PORT"; stop_port "$PORT4B"; fi
st "box done [$STAGES]"
