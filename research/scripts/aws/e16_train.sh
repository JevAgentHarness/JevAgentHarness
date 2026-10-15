#!/usr/bin/env bash
# Exploratory row E16, the best open-weights decision model on all three levels (docs/PREREG_A7_STAGE2.md, Deviations
# 2026-10-08 "E16: the best open-weights decision model on all three levels"; descriptive, outside every Holm table).
# One NEW GPU host launched by this driver from E3's clone image (tag Name=jev-e16-train), SSM only. Two arms on one
# host, in this order:
#   A  AgentDecision-4B's frozen adapter (sha256-checked) continued on the E16 mix (14,000 public tool-use items in
#      E1's format + 14,000 SFT v1 replay rows; built locally by scripts/explore_e16_best_open.py build, sha256-checked
#      here), E12's continuation recipe for 2 epochs; the epoch chosen on E1's dev prefix + JevGym v1 dev only
#   B  StartLux-Decision-9B (CC BY-NC 4.0; downloaded here at the pinned revision) + a fresh LoRA, E12c's recipe on the
#      same mix; 2 epochs when the smoke's step time fits the remaining GPU budget, else 1, else not run (recorded)
# Each arm: temperatures, E1's 3,991 decisions (served by training/serve_jev.py), corpus C, then LIVE (replace mode,
# Flash on the direct Gemini API, seeds 300000-300199) only when its C accuracy is within 1 point of AgentDecision-4B's
# (the registered gate, scripts/explore_e16_best_open.py c-gate). Then the same-engine timing (vLLM, merged LoRA) of
# both arms and the frozen model. Gemini key from SSM Parameter Store into the live stages' process only (never
# printed, never on disk); no other key.
#
#   LOCAL:  bash scripts/aws/e16_train.sh launch [itype ...]    # one new host (refuses if a jev-e16-train host exists);
#                                                                # types in order (default g6e.2xlarge g6e.4xlarge
#                                                                # g5.8xlarge), each in every us-east-1 zone;
#                                                                # user-data arms the power-off guard (GUARD_MIN)
#           bash scripts/aws/e16_train.sh data DIR              # the locally built data -> s3://$BUCKET/e16/data/
#           E16_IID=i-... bash scripts/aws/e16_train.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages (in the order sent; a failed stage stops the chain, `final` always runs when sent):
#     rearm       idle stop (IDLE_MIN) + power-off guard (GUARD_END absolute UTC, else GUARD_MIN); the guard end recorded
#     setup       the committed branch (bundle) in /opt/jev/e16; peft into the v2 venv; the frozen 4B verified (fetched
#                 from S3 if absent); model files pre-read; the 9B download and the vLLM venv install start in the background
#     data        s3://$BUCKET/e16/data -> $A/data, sha256 against results/explore/e16_best_open/build_info.json; dev prefix
#     a_train     continue the frozen adapter (lr 5e-5, 2 epochs, 64 rows / step, max_len 4096, seed 0; nogc_3072, then
#                 gc_3072 / gc_2048 on out-of-memory only; A_CAP_MIN cap); the trainer saves each epoch; adapters -> S3
#     a_select    epoch-end adapters on E1's dev prefix + v1 dev (eval_decisions); the higher mean (tie: epoch 2)
#     a_temps     temperatures of the selected adapter on E12's cal_dev (calibrate.py --min-bucket 200)
#     rehost      the frozen 4B served on :8081: E1's 3,991 decisions (serving-path check + same-host cost reference)
#     a_e1        the selected A adapter served on :8081 (agentdecision-4b-mixedplus): E1's 3,991 decisions
#     a_c         corpus C replayed through the served checkpoint (a new cache); the C gate -> $A/a_cgate.json
#     a_live_smoke / a_live   configs/explore/e16_best_open/live_e16a*.yaml (JEV_EVAL_RUNS=1 for live), only when the
#                 C gate passed; the $8 projection gate; passes 2-3 re-run infrastructure errors only
#     b_smoke     waits for the 9B; 4 training steps per memory setting -> the setting, the epochs (2 / 1 / none)
#     b_train     fresh LoRA, E12c's recipe (lr 1e-4, 64 rows / step, max_len 4096, seed 0), the epochs of b_smoke,
#                 capped at the remaining budget; the trainer saves each epoch; adapters -> S3
#     b_select / b_temps / b_e1 / b_c / b_live_smoke / b_live   as A, in the v2 venv (startlux-9b-mixed)
#     fast        vLLM venv; LoRA merges (frozen, A, B); E1 prompts per tokenizer; the timing (scripts/e12_fast_engine.py)
#     final       stop the server, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e16_best_open/box on the box -> s3://$BUCKET/e16/box; runs -> s3://$BUCKET/e16/runs/<run>;
# the adapters -> s3://$BUCKET/checkpoints/e16_a/ and e16_b/ (no optimizer state).
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E16_IID=${E16_IID:-}
# every other project host (E3's, E5's, Stage 2's, E6a's, E7's, E11's, E4r's, E12's, E12c's, E14's, E15's, the author's):
# never touched by this driver
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
# us-east-1d first (E3's / E11's zone), then the other zones of the same VPC
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e16-train
BUNDLE=e16.bundle
S3OUT="$B/e16"
CAP_USD=${CAP_USD:-8}
GUARD_MIN=${GUARD_MIN:-830}          # 14 host-hours cap (registration (h)): the guard 830 min after launch
IDLE_MIN=${IDLE_MIN:-30}
WORKERS=${WORKERS:-4}
POWEROFF=${POWEROFF:-1}
A_CAP_MIN=${A_CAP_MIN:-330}          # arm A's training cap (minutes)
B_CAP_MIN=${B_CAP_MIN:-420}          # arm B's training cap (minutes), never beyond the remaining budget
RESERVE_MIN=${RESERVE_MIN:-170}      # minutes kept after B's training for b_select .. final
NAME_A=agentdecision-4b-mixedplus
NAME_B=startlux-9b-mixed
NAME_F=jevagentrl-4b
PORT=8081
SL9B_REPO=startlux-models/StartLux-Decision-9B
SL9B_REV=342aad64840be8eee314455428a5a775f18ef547

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E16_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E16_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E16_IID "* ]] && { echo "refused: $E16_IID is not the E16 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E16_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E16_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e16-train: power-off guard (14 host-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g6e.2xlarge g6e.4xlarge g5.8xlarge)
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
if [[ "$MODE" == data ]]; then
  D=${1:?usage: data DIR}
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  python3 "$ROOT/scripts/explore_e16_best_open.py" verify --data "$D" || exit 1   # the committed record
  awsl s3 sync "$D" "$S3OUT/data" --only-show-errors --exclude '*' --include '*.jsonl.gz' --include '*.json' || exit 1
  echo "data -> $S3OUT/data"
  exit 0
fi
if [[ "$MODE" == push ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null || { echo "bundle failed"; exit 1; }
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors || exit 1
  git -C "$ROOT" show "$BRANCH:scripts/aws/e16_train.sh" > "$tmp/e16_train.sh"   # the committed driver
  awsl s3 cp "$tmp/e16_train.sh" "$B/code/e16_train.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e16_train.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e16_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e16_train.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q GUARD_MIN=%q GUARD_END=%q IDLE_MIN=%q WORKERS=%q POWEROFF=%q E16_IID=%q A_CAP_MIN=%q B_CAP_MIN=%q RESERVE_MIN=%q\n' \
      "$CAP_USD" "$GUARD_MIN" "${GUARD_END:-}" "$IDLE_MIN" "$WORKERS" "$POWEROFF" "$E16_IID" "$A_CAP_MIN" "$B_CAP_MIN" "$RESERVE_MIN"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e16_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e16_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E16_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e16_driver.sh /tmp/e16_driver.sh --only-show-errors && bash /tmp/e16_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E16_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E16_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,45p' "$0"
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
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E16_IID" && "$MY_IID" != "$E16_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E16 host ${E16_IID:-?}"; exit 1
fi
R=$W/e16
A=$R/results/explore/e16_best_open/box
STATUS_DIR=$W/e16_status
HV=$W/hvenv                                # harness venv (Qwen3-4B arm, runner, merges, prompts)
VV=$W/v2venv                               # A7.19 (f)'s v2 venv: transformers 5.18 + fla / causal-conv1d (+ peft): StartLux
VLV=$W/vllmvenv                            # the fast engine (vLLM)
CK=$W/ck                                   # the frozen 4B: $CK/<plan dir>
SL9=$W/models/StartLux-Decision-9B
CKA=$W/ck_e16/a_train;  SELA=$W/ck_e16/a_selected
CKB=$W/ck_e16/b_train;  SELB=$W/ck_e16/b_selected
PLAN=results/stage2/rl_eval_plan.json
STOP=runs/e16_STOP.json
E16=scripts/explore_e16_best_open.py
E12=scripts/explore_e12_train.py
FAST=scripts/e12_fast_engine.py
E1_CFG=configs/explore/e16_best_open/e1_e16.yaml
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$STATUS_DIR"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  aws s3 cp "$STATUS_DIR/status.txt" "$S3OUT/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION" --exclude 'data/*' \
    --exclude 's23/corpus.json' --exclude 's23/occurrences.jsonl.gz' --exclude 's23/questions.jsonl.gz' --exclude 'fast/prompts_*'
  local d
  for d in e1_bfcl_when2call_e16 e16a_live_smoke e16a_live e16b_live_smoke e16b_live; do
    [[ -d "$R/runs/$d" ]] && aws s3 sync "$R/runs/$d" "$S3OUT/runs/$d" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  true
}
PY() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }        # harness venv
PV() { PYTHONPATH="$R/src" "$VV/bin/python" "$@"; }        # v2 venv (StartLux kernels)
plan() { (cd "$R" && PY -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"); }
qbase() { cat "$W/base_model_path.txt"; }                  # Qwen3-4B (AgentDecision-4B's base)
frozen_dir() { echo "$CK/$(plan "d['rows']['frozen']['dir']")"; }
frozen_cal() { echo "$CK/$(plan "d['rows']['frozen']['calibration']")"; }
guard_end() { cat "$STATUS_DIR/guard_end_epoch.txt" 2>/dev/null || echo 0; }
mins_left() { echo $(( ( $(guard_end) - $(date +%s) ) / 60 )); }
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
leak_check() {
  local n=0 k ks=${GEMINI_API_KEYS:-}
  for k in ${GEMINI_API_KEY:-} ${ks//,/ }; do
    [[ ${#k} -ge 16 ]] || continue
    n=$(( n + $(grep -rlF -- "$k" "$R/runs" "$A" "$STATUS_DIR" 2>/dev/null | wc -l) ))
  done
  echo "key-leak files: $n"
  [[ "$n" == 0 ]]
}
run_watched() {
  local cfg=$1; shift
  setsid env PYTHONPATH="$R/src" "$HV/bin/jevbench" run "$cfg" --out runs "$@" >> "$A/runner.log" 2>&1 < /dev/null &
  local pid=$!
  PY "$E16" watch --pid "$pid" --cap "$CAP_USD" --runs runs --every 30 >> "$A/watch.log" 2>&1 &
  local wpid=$!
  ( while kill -0 "$pid" 2>/dev/null; do sleep 300; sync_out; done ) & local spid=$!
  wait "$pid"; local rc=$?
  kill "$wpid" "$spid" 2>/dev/null
  sync_out
  return $rc
}
errored() {
  PY - "$1" <<'PYEOF'
import glob, json, sys
n = 0
for f in glob.glob(sys.argv[1] + "/*.jsonl"):
    last = {}
    for l in open(f):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if isinstance(r, dict) and "task_id" in r:
            last[r["task_id"]] = r
    n += sum(1 for r in last.values() if r.get("error"))
print(n)
PYEOF
}
stop_server() {
  local p
  if [[ -f "$W/e16_serve.pid" ]]; then
    while read -r p; do [[ -n "$p" ]] && kill "$p" 2>/dev/null; done < "$W/e16_serve.pid"
  fi
  rm -f "$W/e16_serve.pid" "$W/e16_serve.name"; sleep 8
}
server_up() { [[ "$(cat "$W/e16_serve.name" 2>/dev/null)" == "$1" ]] && curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; }
serve() {  # serve VENV NAME BASE ADAPTER CALIBRATION: one server on :8081 (no extra arguments, as the frozen row)
  local venv=$1 name=$2 base=$3 adapter=$4 cal=$5
  stop_server
  [[ -s "$adapter/adapter_model.safetensors" && -s "$cal" ]] || { echo "no adapter / temperatures for $name"; return 1; }
  local args=(--model "$base" --name "$name" --port "$PORT" --adapter "$adapter" --calibration "$cal")
  echo "serve_jev ${args[*]} (venv $venv)" > "$A/serve_$name.cmd"
  nohup env PYTHONPATH="$R/src" "$venv/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$A/logs/serve_$name.out" 2>&1 &
  echo $! > "$W/e16_serve.pid"
  echo "$name" > "$W/e16_serve.name"
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/health" > "$A/serve_health_$name.json"; then
      curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/serve_models_$name.json"
      grep -q "\"$name\"" "$A/serve_models_$name.json" || { echo "served model id is not $name"; return 1; }
      printf '{"server_up": "%s", "name": "%s", "instance_type": "%s", "instance_id": "%s"}\n' "$(date -u +%FT%TZ)" \
        "$name" "$(imds instance-type)" "$MY_IID" >> "$A/serve_windows.jsonl"
      return 0
    fi
    kill -0 "$(cat "$W/e16_serve.pid")" 2>/dev/null || { echo "serve_jev died"; tail -20 "$A/logs/serve_$name.out"; return 1; }
    sleep 5
  done
  echo "serve_jev did not come up"; return 1
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
prewarm() {  # read files in 64 MB chunks, 32 at a time (a new EBS volume loads lazily from its snapshot)
  local d
  for d in "$@"; do
    find -L "$d" -type f -size +1M 2>/dev/null | while read -r f; do
      local s n i; s=$(stat -L -c %s "$f"); n=$(( (s + 67108863) / 67108864 ))
      for i in $(seq 0 $((n - 1))); do echo "$f $i"; done
    done
  done | xargs -P 32 -n 2 sh -c 'dd if="$0" of=/dev/null bs=64M skip=$1 count=1 status=none'
}
train_a() {  # train_a OUT SETTING [extra]: E12's continuation of the frozen adapter, 2 epochs
  local out=$1 setting=$2; shift 2
  local flags; flags=$(PY scripts/train/ablation_box.py setting "$setting") || return 1
  rm -rf "$out"; mkdir -p "$out"
  # shellcheck disable=SC2086
  PY -m jevharness.training.train_decision --model "$(qbase)" --init "$(frozen_dir)" --train "$A/data/train_mix.jsonl.gz" \
    --out "$out" --loss soft_ce_brier --brier-weight 0.5 --lr 5e-5 --epochs 2 --batch-size 2 --max-len 4096 --seed 0 \
    --report-to none $flags "$@"
}
train_b() {  # train_b OUT SETTING EPOCHS [extra]: E12c's SFT recipe from a fresh LoRA on StartLux-9B
  local out=$1 setting=$2 epochs=$3; shift 3
  local flags; flags=$(PY scripts/train/ablation_box.py setting "$setting") || return 1
  rm -rf "$out"; mkdir -p "$out"
  # shellcheck disable=SC2086
  PV -m jevharness.training.train_decision --model "$SL9" --train "$A/data/train_mix.jsonl.gz" --out "$out" \
    --loss soft_ce_brier --brier-weight 0.5 --lora-r 32 --lora-alpha 64 --lr 1e-4 --epochs "$epochs" --max-len 4096 \
    --seed 0 --report-to none $flags "$@"
}
sl9b_ready() { [[ -s "$SL9/config.json" && -s "$SL9/model.safetensors.index.json" && -f "$A/sl9b_download.ok" ]]; }
download_9b() {  # the pinned revision, public weights, no token; weights + config + tokenizer + licence (not the assets)
  PV - "$SL9" <<EOF > "$A/logs/sl9b_download.log" 2>&1 && touch "$A/sl9b_download.ok"
import sys
from huggingface_hub import snapshot_download
p = snapshot_download("$SL9B_REPO", revision="$SL9B_REV", local_dir=sys.argv[1],
                      allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "LICENSE", "NOTICE", "README.md"])
print(p)
EOF
}

stage_rearm() {
  if [[ -n "${GUARD_END:-}" ]]; then GUARD_MIN=$(( ( $(date -u -d "$GUARD_END" +%s) - $(date -u +%s) ) / 60 )); fi
  (( GUARD_MIN > 0 )) || { echo "guard end already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "jev-e16-train: power-off guard (GPU cap)"
  echo $(( $(date +%s) + GUARD_MIN * 60 )) > "$STATUS_DIR/guard_end_epoch.txt"
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
f=/var/tmp/jev-idle-count
if pgrep -f 'e16_[0-9]|jevharness|jevbench|serve_jev|explore_e16|train_decision|stage2_corpus|e12_fast_engine|vllm|pip|uv |snapshot_download' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e16-train: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $GUARD_MIN min (end $(date -u -d @"$(guard_end)" +%FT%TZ)), idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU; $(df -BG / | awk 'NR==2{print $4}') free"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}
stage_setup() {
  local myb="$W/e16.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A/logs" "$A/eval" "$A/sel_a" "$A/sel_b" "$A/fast" "$A/cal" runs "$W/models"
  st "code at $(git log --oneline -1 | cut -c1-72)"
  grep -q 'E16: the best open-weights decision model on all three levels' docs/PREREG_A7_STAGE2.md \
    || { echo "no Deviations entry for E16"; return 1; }
  ( prewarm "$(qbase)" "$VV" "$HV/lib" ) &
  local pw=$!
  command -v uv > /dev/null || bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  "$VV/bin/python" -c "import peft" 2>/dev/null || uv pip install -q --python "$VV" "peft==0.21.2" || return 1
  PV -c "import jevharness, torch, transformers, peft, fla, causal_conv1d, sys; p = jevharness.__file__; print(p, torch.__version__, transformers.__version__, peft.__version__, fla.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_v2venv.txt" 2>&1 || { cat "$A/env_v2venv.txt"; echo "the v2 venv does not import this checkout"; return 1; }
  PY -c "import jevharness, torch, transformers, peft, sys; p = jevharness.__file__; print(p, torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_hvenv.txt" 2>&1 || { cat "$A/env_hvenv.txt"; return 1; }
  cat "$A/env_v2venv.txt" "$A/env_hvenv.txt"
  [[ -x "$HV/bin/jevbench" ]] || { echo "no jevbench in the hvenv"; return 1; }
  cp -f "$W/base_model_path.txt" "$A/base_model_path.txt"
  # the 9B download and the vLLM venv in the background (CPU / network only; waited for by b_smoke and fast)
  rm -f "$A/sl9b_download.ok"
  ( download_9b ) &
  if [[ ! -x "$VLV/bin/python" ]]; then
    ( uv venv -q -p 3.11 "$VLV" && uv pip install -q --python "$VLV" vllm > "$A/logs/vllm_install.log" 2>&1 && touch "$A/vllm_install.ok" ) &
  else
    touch "$A/vllm_install.ok"
  fi
  wait "$pw"
  if ! verify_frozen; then
    st "the frozen 4B is absent or different in $CK; fetching $(plan "d['rows']['frozen']['s3']")"
    aws s3 sync "$(plan "d['rows']['frozen']['s3']")" "$(frozen_dir)" --only-show-errors --region "$REGION" || return 1
    verify_frozen || return 1
  fi
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s", "guard_end_utc": "%s"}\n' "$MY_IID" \
    "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" "$(date -u -d @"$(guard_end)" +%FT%TZ)" > "$A/host_setup.json"
  df -BG / | awk 'NR==2{print "disk free: "$4}'
}
stage_data() {
  cd "$R" || return 1
  mkdir -p "$A/data"
  aws s3 sync "$S3OUT/data" "$A/data" --only-show-errors --region "$REGION" || return 1
  PY "$E16" verify --data "$A/data" || return 1
  cp -f "$A/data/build_info.json" "$A/build_info_s3.json" 2>/dev/null || true
  PY "$E12" devprefix --out "$A/data/e1_devprefix.jsonl.gz" > "$A/devprefix.json" || return 1
  cat "$A/devprefix.json"
}
# ---------------------------------------------------------------- arm A
stage_a_train() {
  cd "$R" || return 1
  stop_server
  verify_frozen || return 1
  local setting rc=1 t0
  for setting in nogc_3072 gc_3072 gc_2048; do
    t0=$(date +%s)
    nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 10000 > "$A/logs/a_train_smi_$setting.csv" 2>/dev/null &
    local smi=$!
    timeout "${A_CAP_MIN}m" bash -c "$(declare -f train_a PY qbase frozen_dir plan); A='$A' R='$R' HV='$HV' W='$W' CK='$CK' PLAN='$PLAN' train_a '$CKA' $setting" \
      > "$A/logs/a_train_$setting.log" 2>&1
    rc=$?
    kill "$smi" 2>/dev/null
    printf '{"arm": "a", "setting": "%s", "exit": %s, "minutes": %s}\n' "$setting" "$rc" "$(( ($(date +%s) - t0) / 60 ))" >> "$A/train_runs.jsonl"
    (( rc == 0 )) && break
    grep -qE 'out of memory|OutOfMemoryError' "$A/logs/a_train_$setting.log" || break
  done
  ls -d "$CKA"/checkpoint-* > "$A/a_checkpoints.txt" 2>/dev/null
  local c
  for c in "$CKA"/checkpoint-* "$CKA"; do
    [[ -s "$c/adapter_model.safetensors" ]] && sha256sum "$c/adapter_model.safetensors" >> "$A/a_sha256.txt"
  done
  aws s3 sync "$CKA" "$B/checkpoints/e16_a/train/" --only-show-errors --region "$REGION" \
    --exclude '*optimizer*' --exclude '*.pt' --exclude 'rng_state*' --exclude 'scheduler*'
  sync_out
  (( rc == 0 )) || [[ -n "$(ls -d "$CKA"/checkpoint-* 2>/dev/null)" ]] || return 1   # timed out: epoch 1 still selectable
  return 0
}
select_arm() {  # select_arm ARM CKDIR SELDIR PYFN TOKSRC BASE: epoch-end adapters on E1's dev prefix + v1 dev; higher mean (tie: later)
  local arm=$1 ckdir=$2 seldir=$3 pyfn=$4 toksrc=$5 base=$6
  local c i=0 cands=() S="$A/sel_$arm"
  for c in $(ls -d "$ckdir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do [[ -s "$c/adapter_model.safetensors" ]] && cands+=("$c"); done
  [[ ${#cands[@]} -ge 1 ]] || { echo "no epoch checkpoint for arm $arm"; return 1; }
  for c in "${cands[@]}"; do   # the trainer's epoch checkpoints hold no tokenizer: the final one's (the base's) is copied
    cp -n "$ckdir"/tokenizer.json "$ckdir"/tokenizer_config.json "$ckdir"/chat_template.jinja "$c/" 2>/dev/null
    cp -n "$toksrc"/tokenizer.json "$toksrc"/tokenizer_config.json "$toksrc"/chat_template.jinja "$c/" 2>/dev/null
    i=$((i + 1))
    "$pyfn" -m jevharness.training.eval_decisions --model "$base" --adapter "$c" \
      --dev "$A/data/e1_devprefix.jsonl.gz" data/train_v1/dev.jsonl.gz --n-perm 0 --max-len 4096 --batch-size 8 \
      --out "$S/epoch$i.json" > "$A/logs/${arm}_sel_epoch$i.log" 2>&1 || return 1
    echo "$c" > "$S/epoch$i.path"
  done
  PY - "$S" <<'PYEOF' || return 1
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = []
for f in sorted(d.glob("epoch*.json")):
    ev = json.loads(f.read_text())["files"]
    acc = {Path(k).name.split(".")[0]: v["overall"]["acc"] for k, v in ev.items()}
    rows.append({"epoch": int(f.stem[5:]), "path": (d / f"{f.stem}.path").read_text().strip(), "acc": acc,
                 "mean": (acc["e1_devprefix"] + acc["dev"]) / 2})
best = max(rows, key=lambda r: (r["mean"], r["epoch"]))      # tie: the later epoch (the recipe's endpoint)
out = {"rule": "higher mean of E1 dev-prefix and v1-dev accuracy; tie -> the later epoch", "candidates": rows, "selected": best}
(d / "selection.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out))
PYEOF
  local pick; pick=$(python3 -c "import json; print(json.load(open('$S/selection.json'))['selected']['path'])")
  rm -rf "$seldir"; mkdir -p "$seldir"
  cp -f "$pick"/adapter_model.safetensors "$pick"/adapter_config.json "$seldir/"
  cp -f "$pick"/tokenizer* "$pick"/chat_template.jinja "$seldir/" 2>/dev/null
  cp -n "$toksrc"/tokenizer* "$toksrc"/chat_template.jinja "$seldir/" 2>/dev/null
  st "arm $arm selected: $(python3 -c "import json; s=json.load(open('$S/selection.json'))['selected']; print(s['epoch'], s['acc'])")"
  sync_out
}
stage_a_select() { cd "$R" || return 1; stop_server; select_arm a "$CKA" "$SELA" PY "$(frozen_dir)" "$(qbase)"; }
temps_arm() {  # temps_arm ARM SELDIR PYFN BASE S3DIR
  local arm=$1 seldir=$2 pyfn=$3 base=$4 s3dir=$5
  "$pyfn" -m jevharness.training.calibrate --model "$base" --adapter "$seldir" --dev "$A/data/cal_dev.jsonl.gz" \
    --out "$seldir/calibration.json" --min-bucket 200 --batch-size 8 > "$A/logs/${arm}_calibrate.log" 2>&1 || return 1
  cp -f "$seldir/calibration.json" "$A/cal/${arm}_calibration.json"
  sha256sum "$seldir/adapter_model.safetensors" "$seldir/calibration.json" > "$A/${arm}_selected_sha256.txt"
  aws s3 sync "$seldir" "$s3dir" --only-show-errors --region "$REGION"
  sync_out
}
stage_a_temps() { cd "$R" || return 1; stop_server; temps_arm a "$SELA" PY "$(qbase)" "$B/checkpoints/e16_a/selected/"; }
e1_arm() {  # e1_arm BACKEND: E1's decisions, two invocations as E1 (the second resumes any failed request from the cache)
  local k
  for k in 1 2; do
    PY -m jevharness.bench.decision_eval "$E1_CFG" --out runs --only "$1" > "$A/logs/e1_$1_$k.log" 2>&1
    st "e1 $1 invocation $k exit=$?"
  done
  sync_out
}
stage_rehost() {
  cd "$R" || return 1
  verify_frozen || return 1
  serve "$HV" "$NAME_F" "$(qbase)" "$(frozen_dir)" "$(frozen_cal)" || return 1
  e1_arm frozen_rehost
  stop_server
  true   # the agreement with the logged E1 answers is computed off the host (the logged records are not in the bundle)
}
stage_a_e1() {
  cd "$R" || return 1
  serve "$HV" "$NAME_A" "$(qbase)" "$SELA" "$SELA/calibration.json" || return 1
  e1_arm e16a
}
c_arm() {  # c_arm ARM NAME: corpus C through the served checkpoint (a new cache); then the registered live gate
  local arm=$1 name=$2 f k spec
  mkdir -p "$A/s23"
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz; do cp -f "results/stage2/s23/$f" "$A/s23/$f" || return 1; done
  spec=$(printf '{"type": "jev", "name": "%s", "base_url": "http://127.0.0.1:%s", "model": "%s", "api_key": "", "timeout": 60.0}' \
    "$name" "$PORT" "$name")
  for k in 1 2 3; do   # the replay resumes; a scoring failure is retried inside
    PY scripts/stage2_corpus.py replay --corpus "$A/s23" --name "e16$arm" --s1 "$spec" --cache "cache/s1_e16${arm}_c.sqlite" \
      > "$A/logs/${arm}_replay_c_$k.log" 2>&1
  done
  st "arm $arm corpus C replayed: $(wc -l < "$A/s23/answers_e16$arm.jsonl" 2>/dev/null) answers"
  PY "$E16" c-gate --answers "$A/s23/answers_e16$arm.jsonl" --out "$A/${arm}_cgate.json" > "$A/logs/${arm}_cgate.log" 2>&1
  st "arm $arm C gate: $(python3 -c "import json; d=json.load(open('$A/${arm}_cgate.json')); print('live' if d['live'] else 'no live', round(100*d['c_acc'],2), 'vs', round(100*d['frozen_c_acc'],2))" 2>/dev/null)"
  sync_out
}
stage_a_c() {
  cd "$R" || return 1
  server_up "$NAME_A" || serve "$HV" "$NAME_A" "$(qbase)" "$SELA" "$SELA/calibration.json" || return 1
  c_arm a "$NAME_A"
}
live_gate_ok() { python3 -c "import json,sys; sys.exit(0 if json.load(open('$A/$1_cgate.json'))['live'] else 1)" 2>/dev/null; }
live_smoke_arm() {  # live_smoke_arm ARM CFG
  local arm=$1 cfg=$2 rc
  live_gate_ok "$arm" || { st "arm $arm: no live (C gate)"; return 0; }
  keys || return 1
  run_watched "$cfg"
  rc=$?
  PY "$E16" project --arm "$arm" --runs runs --out "$A/${arm}_projection.json" > "$A/logs/${arm}_projection.out" 2>&1
  st "arm $arm smoke exit=$rc; projection: $(python3 -c "import json; print(json.load(open('$A/${arm}_projection.json'))['decision'])" 2>/dev/null)"
  leak_check || return 1
  return $rc
}
live_arm() {  # live_arm ARM CFG RUN
  local arm=$1 cfg=$2 run=$3 dec pass rc t0 t1 ne
  live_gate_ok "$arm" || { st "arm $arm: no live (C gate)"; return 0; }
  keys || return 1
  dec=$(python3 -c "import json; print(json.load(open('$A/${arm}_projection.json'))['decision'])" 2>/dev/null)
  [[ "$dec" == "run all 200" ]] || { st "arm $arm: refusing live: projection says '$dec'"; return 0; }
  export JEV_EVAL_RUNS=1      # this stage's subshell only (the Stage-2 evaluation seeds 300000-300199)
  printf 'step %s\nstarted %s\ncommit %s\nconfig %s %s\nprereg sha256 %s\ncap_usd %s\nhost %s %s\n' "$run" \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$cfg" "$(sha256sum "$cfg" | cut -c1-16)" \
    "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" "$CAP_USD" "$MY_IID" "$(imds instance-type)" > "$A/START_${run}.txt"
  t0=$(date -u +%FT%TZ)
  for pass in 1 2 3; do
    [[ -f "$STOP" ]] && { st "watchdog stop recorded; no further pass"; break; }
    if (( pass > 1 )); then
      ne=$(errored "runs/$run")
      [[ "$ne" == 0 ]] && break
      st "$ne errored episode(s) after pass $((pass - 1)); pass $pass"; sleep 120
    fi
    st "arm $arm live pass $pass"
    run_watched "$cfg"
    rc=$?
    st "arm $arm live pass $pass exit=$rc; $(PY "$E16" spend --runs runs --short 2>/dev/null)"
  done
  t1=$(date -u +%FT%TZ)
  printf '{"window_start": "%s", "window_end": "%s", "instance_type": "%s", "instance_id": "%s", "workers": 4, "run": "runs/%s"}\n' \
    "$t0" "$t1" "$(imds instance-type)" "$MY_IID" "$run" > "$A/${arm}_live_window.json"
  unset JEV_EVAL_RUNS
  leak_check || return 1
  [[ -f "$STOP" ]] && return 1
  return 0
}
stage_a_live_smoke() {
  cd "$R" || return 1
  live_gate_ok a || { st "arm a: no live (C gate)"; return 0; }
  server_up "$NAME_A" || serve "$HV" "$NAME_A" "$(qbase)" "$SELA" "$SELA/calibration.json" || return 1
  live_smoke_arm a configs/explore/e16_best_open/live_e16a_smoke.yaml
}
stage_a_live() {
  cd "$R" || return 1
  live_gate_ok a || { st "arm a: no live (C gate)"; return 0; }
  server_up "$NAME_A" || serve "$HV" "$NAME_A" "$(qbase)" "$SELA" "$SELA/calibration.json" || return 1
  live_arm a configs/explore/e16_best_open/live_e16a.yaml e16a_live
  stop_server
}
# ---------------------------------------------------------------- arm B
stage_b_smoke() {  # waits for the 9B; 4 training steps per memory setting: the setting, then the epochs that fit the budget
  cd "$R" || return 1
  stop_server
  local end=$(( $(date +%s) + 60 * 60 ))
  until sl9b_ready; do (( $(date +%s) < end )) || { st "arm b: the 9B did not download in time"; echo '{"b": "not run", "reason": "9B download did not finish"}' > "$A/b_plan.json"; return 0; }; sleep 30; done
  (cd "$SL9" && sha256sum ./*.safetensors config.json tokenizer.json 2>/dev/null) > "$A/startlux_9b_sha256.txt"
  cp -f "$SL9/LICENSE" "$A/startlux_9b_LICENSE.txt" 2>/dev/null; head -c 4000 "$SL9/README.md" > "$A/startlux_9b_README_head.txt" 2>/dev/null
  local setting rc s sps steps proj1 proj2 cap epochs=0 left
  for setting in nogc_3072 gc_3072 gc_2048; do
    s=$(date +%s)
    timeout 40m bash -c "$(declare -f train_b PV PY); A='$A' R='$R' HV='$HV' VV='$VV' SL9='$SL9' train_b '$W/ck_e16/b_smoke' $setting 2 --max-steps 4" \
      > "$A/logs/b_smoke_$setting.log" 2>&1
    rc=$?
    sps=$(PY -c "
import re
t = open('$A/logs/b_smoke_$setting.log').read()
m = re.search(r\"'train_runtime': '?([0-9.]+)\", t)
print(float(m.group(1)) / 4 if m else '')")
    steps=$(grep -oE 'token budget [0-9]+: [0-9]+ micro-batches/epoch, [0-9.]+ rows each on average, grad-accum [0-9]+' "$A/logs/b_smoke_$setting.log" | head -1)
    printf '{"arm": "b", "setting": "%s", "exit": %s, "wall_s": %s, "s_per_step": "%s", "budget": "%s"}\n' "$setting" \
      "$rc" "$(( $(date +%s) - s ))" "$sps" "$steps" >> "$A/b_smoke.jsonl"
    if (( rc == 0 )); then
      proj1=$(PY -c "
import re
t = open('$A/logs/b_smoke_$setting.log').read()
m = re.search(r'token budget \d+: (\d+) micro-batches/epoch, [0-9.]+ rows each on average, grad-accum (\d+)', t)
print(round(-(-int(m.group(1)) // int(m.group(2))) * float('$sps') / 60))")
      proj2=$(( proj1 * 2 ))
      left=$(mins_left); cap=$(( left - RESERVE_MIN )); (( cap > B_CAP_MIN )) && cap=$B_CAP_MIN
      if (( proj2 + 10 <= cap )); then epochs=2; elif (( proj1 + 10 <= cap )); then epochs=1; else epochs=0; fi
      st "arm b smoke: $setting ok, ${sps}s / step, one epoch about $proj1 min, cap $cap min (left $left): epochs $epochs"
      python3 - "$A/b_plan.json" "$setting" "$sps" "$proj1" "$cap" "$epochs" "$left" <<'PYEOF'
import json, sys
p, setting, sps, proj1, cap, epochs, left = sys.argv[1:8]
json.dump({"b": "run" if int(epochs) > 0 else "not run", "setting": setting, "s_per_step": float(sps), "projected_min_per_epoch": int(proj1),
           "cap_min": int(cap), "minutes_left_at_smoke": int(left), "epochs": int(epochs),
           "reason": None if int(epochs) > 0 else "one epoch does not fit the remaining GPU budget"}, open(p, "w"), indent=1)
PYEOF
      rm -rf "$W/ck_e16/b_smoke"
      return 0
    fi
    grep -qE 'out of memory|OutOfMemoryError' "$A/logs/b_smoke_$setting.log" || break
  done
  st "arm b smoke: no memory setting trained 4 steps (last exit $rc)"
  printf '{"b": "not run", "reason": "no memory setting passed the training smoke (exit %s)"}\n' "$rc" > "$A/b_plan.json"
  return 0
}
b_planned() { python3 -c "import json,sys; d=json.load(open('$A/b_plan.json')); sys.exit(0 if d.get('b') == 'run' else 1)" 2>/dev/null; }
stage_b_train() {
  cd "$R" || return 1
  stop_server
  b_planned || { st "arm b: not run ($(python3 -c "import json; print(json.load(open('$A/b_plan.json')).get('reason'))" 2>/dev/null))"; return 0; }
  local setting epochs cap rc=1 t0 order
  setting=$(python3 -c "import json; print(json.load(open('$A/b_plan.json'))['setting'])") || return 1
  epochs=$(python3 -c "import json; print(json.load(open('$A/b_plan.json'))['epochs'])") || return 1
  cap=$(( $(mins_left) - RESERVE_MIN )); (( cap > B_CAP_MIN )) && cap=$B_CAP_MIN
  (( cap >= 30 )) || { st "arm b: no time left for training ($cap min)"; echo '{"b": "not run", "reason": "no time left"}' > "$A/b_plan.json"; return 0; }
  case "$setting" in nogc_3072) order="nogc_3072 gc_3072 gc_2048" ;; gc_3072) order="gc_3072 gc_2048" ;; *) order="gc_2048" ;; esac
  for setting in $order; do
    t0=$(date +%s)
    nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 10000 > "$A/logs/b_train_smi_$setting.csv" 2>/dev/null &
    local smi=$!
    timeout "${cap}m" bash -c "$(declare -f train_b PV PY); A='$A' R='$R' HV='$HV' VV='$VV' SL9='$SL9' train_b '$CKB' $setting $epochs" \
      > "$A/logs/b_train_$setting.log" 2>&1
    rc=$?
    kill "$smi" 2>/dev/null
    printf '{"arm": "b", "setting": "%s", "epochs": %s, "cap_min": %s, "exit": %s, "minutes": %s}\n' "$setting" "$epochs" "$cap" "$rc" "$(( ($(date +%s) - t0) / 60 ))" >> "$A/train_runs.jsonl"
    (( rc == 0 )) && break
    grep -qE 'out of memory|OutOfMemoryError' "$A/logs/b_train_$setting.log" || break
  done
  ls -d "$CKB"/checkpoint-* > "$A/b_checkpoints.txt" 2>/dev/null
  local c
  for c in "$CKB"/checkpoint-* "$CKB"; do
    [[ -s "$c/adapter_model.safetensors" ]] && sha256sum "$c/adapter_model.safetensors" >> "$A/b_sha256.txt"
  done
  aws s3 sync "$CKB" "$B/checkpoints/e16_b/train/" --only-show-errors --region "$REGION" \
    --exclude '*optimizer*' --exclude '*.pt' --exclude 'rng_state*' --exclude 'scheduler*'
  sync_out
  if (( rc != 0 )) && [[ -z "$(ls -d "$CKB"/checkpoint-* 2>/dev/null)" ]]; then
    st "arm b: training ended without an epoch checkpoint (exit $rc)"
    printf '{"b": "not run", "reason": "training ended without an epoch checkpoint (exit %s)"}\n' "$rc" > "$A/b_plan.json"
  fi
  return 0
}
stage_b_select() { cd "$R" || return 1; b_planned || return 0; stop_server; select_arm b "$CKB" "$SELB" PV "$SL9" "$SL9"; }
stage_b_temps() { cd "$R" || return 1; b_planned || return 0; stop_server; temps_arm b "$SELB" PV "$SL9" "$B/checkpoints/e16_b/selected/"; }
stage_b_e1() {
  cd "$R" || return 1
  b_planned || return 0
  serve "$VV" "$NAME_B" "$SL9" "$SELB" "$SELB/calibration.json" || return 1
  e1_arm e16b
}
stage_b_c() {
  cd "$R" || return 1
  b_planned || return 0
  server_up "$NAME_B" || serve "$VV" "$NAME_B" "$SL9" "$SELB" "$SELB/calibration.json" || return 1
  c_arm b "$NAME_B"
}
stage_b_live_smoke() {
  cd "$R" || return 1
  b_planned || return 0
  live_gate_ok b || { st "arm b: no live (C gate)"; return 0; }
  server_up "$NAME_B" || serve "$VV" "$NAME_B" "$SL9" "$SELB" "$SELB/calibration.json" || return 1
  live_smoke_arm b configs/explore/e16_best_open/live_e16b_smoke.yaml
}
stage_b_live() {
  cd "$R" || return 1
  b_planned || return 0
  live_gate_ok b || { st "arm b: no live (C gate)"; return 0; }
  server_up "$NAME_B" || serve "$VV" "$NAME_B" "$SL9" "$SELB" "$SELB/calibration.json" || return 1
  live_arm b configs/explore/e16_best_open/live_e16b.yaml e16b_live
  stop_server
}
# ---------------------------------------------------------------- the same-engine timing (E12c (h)'s readout)
stage_fast() {
  cd "$R" || return 1
  stop_server
  local F=$A/fast M=$W/merged end
  mkdir -p "$F" "$M"
  end=$(( $(date +%s) + 45 * 60 ))
  until [[ -f "$A/vllm_install.ok" ]]; do (( $(date +%s) < end )) || { st "fast: vLLM install did not finish"; tail -3 "$A/logs/vllm_install.log"; return 0; }; sleep 20; done
  "$VLV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" > "$F/vllm_version.txt" 2>&1
  cat "$F/vllm_version.txt"
  [[ -s "$M/frozen/config.json" ]] || PY "$FAST" merge --base "$(qbase)" --adapter "$(frozen_dir)" --out "$M/frozen" > "$A/logs/merge_frozen.log" 2>&1
  [[ -s "$M/e16a/config.json" || ! -s "$SELA/adapter_model.safetensors" ]] || PY "$FAST" merge --base "$(qbase)" --adapter "$SELA" --out "$M/e16a" > "$A/logs/merge_e16a.log" 2>&1
  [[ -s "$M/e16b/config.json" || ! -s "$SELB/adapter_model.safetensors" ]] || PV "$FAST" merge --base "$SL9" --adapter "$SELB" --out "$M/e16b" > "$A/logs/merge_e16b.log" 2>&1
  [[ -s "$F/prompts_qwen.jsonl.gz" ]] || PY "$FAST" prompts --tokenizer "$(qbase)" --out "$F/prompts_qwen.jsonl.gz" > "$F/prompts_qwen.json" 2>&1
  [[ -s "$F/prompts_sl9.jsonl.gz" || ! -s "$SL9/tokenizer.json" ]] || PV "$FAST" prompts --tokenizer "$SL9" --out "$F/prompts_sl9.jsonl.gz" > "$F/prompts_sl9.json" 2>&1
  local m path pr
  for m in e16a e16b frozen; do
    case "$m" in
      frozen) path=$M/frozen; pr=$F/prompts_qwen.jsonl.gz ;;
      e16a) path=$M/e16a; pr=$F/prompts_qwen.jsonl.gz ;;
      e16b) path=$M/e16b; pr=$F/prompts_sl9.jsonl.gz ;;
    esac
    [[ -s "$path/config.json" ]] || { echo "{\"model\": \"$m\", \"state\": \"not available\"}" > "$F/time_$m.json"; continue; }
    (( $(mins_left) > 25 )) || { echo "{\"model\": \"$m\", \"state\": \"no time left before the guard\"}" > "$F/time_$m.json"; continue; }
    timeout 40m "$VLV/bin/python" "$FAST" time --model "$path" --prompts "$pr" --label "$m" --out "$F/time_$m.json" \
      > "$A/logs/fast_$m.log" 2>&1 || echo "{\"model\": \"$m\", \"state\": \"vLLM failed: $(tail -1 "$A/logs/fast_$m.log" | tr '"' "'" | cut -c1-200)\"}" > "$F/time_$m.json"
    st "fast $m: $(python3 -c "import json; d=json.load(open('$F/time_$m.json')); print(d.get('state') or round(d['usd_per_1000_wall'], 4))" 2>/dev/null)"
    sync_out
  done
}
stage_final() {
  stop_server
  nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  printf '{"final_utc": "%s", "instance_id": "%s", "instance_type": "%s", "guard_end_utc": "%s"}\n' "$(date -u +%FT%TZ)" "$MY_IID" \
    "$(imds instance-type)" "$(date -u -d @"$(guard_end)" +%FT%TZ 2>/dev/null)" > "$A/host_final.json"
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; sudo shutdown -h +2 "jev-e16-train: done"; fi
}

broke=0
for s in $STAGES; do
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $s"; continue; fi
  st "stage $s start (host $MY_IID)"
  t0=$(date +%s)
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
