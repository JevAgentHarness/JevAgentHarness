#!/usr/bin/env bash
# Exploratory row E12c, the recipe on a stronger base, and same-engine serving cost (docs/PREREG_A7_STAGE2.md,
# Deviations 2026-10-07 "E12c: the recipe on a stronger base, and same-engine serving cost"; descriptive, outside every
# Holm table). A SECOND new GPU host launched by this driver from E3's clone image (tag Name=jev-e12c-train), SSM only.
# StartLux-Decision-4B (CC BY-NC 4.0; on the image at /opt/startlux/models) + a fresh LoRA, SFT v1 recipe on E12's
# 14,000-row mix (sha256-checked), epoch-end checkpoints selected on E1's dev prefix + JevGym v1 dev only, then E1's
# 3,991 decisions, corpus C and live (replace mode, Flash on the direct Gemini API, seeds 300000-300199); then the
# same-engine timing (vLLM, merged LoRA) of E12, E12c, AgentDecision-4B and StartLux-4B. Gemini key from SSM into the
# live stages' process only (never printed, never on disk); no other key.
#
#   LOCAL:  bash scripts/aws/e12c_train.sh launch [itype ...]   # one new host (refuses if a jev-e12c-train host exists);
#                                                                # types in order (default g6e.2xlarge g6e.xlarge
#                                                                # g6e.4xlarge g5.8xlarge g5.4xlarge), each in every
#                                                                # us-east-1 zone; user-data arms the power-off guard
#           E12C_IID=i-... bash scripts/aws/e12c_train.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages (in the order sent; a failed stage stops the chain, `final` always runs when sent):
#     rearm      idle stop (IDLE_MIN) + power-off guard (GUARD_END absolute UTC, else GUARD_MIN)
#     setup      the committed branch (bundle) in /opt/jev/e12c; peft into the v2 venv (fla / causal-conv1d kernels);
#                StartLux-4B's weight sha256 recorded; the frozen 4B verified; model files pre-read (EBS lazy load)
#     data       E12's data from s3://$BUCKET/e12/data, sha256 against results/explore/e12/build_info.json
#     smoke      4 training steps per memory setting -> the setting and the projection; the Qwen3-8B fallback (E12c (b))
#     train      fresh LoRA, SFT v1 recipe (lr 1e-4, 2 epochs, 64 rows / step, max_len 4096, seed 0), 360 min cap;
#                the trainer saves each epoch; adapters -> S3
#     select     epoch-end adapters on E1's dev prefix + v1 dev (eval_decisions); the higher mean (tie: epoch 2)
#     temps      temperatures of the selected adapter on E12's cal_dev
#     offline    the selected adapter on the v1 dev files and E12's public dev split
#     e1         served on :8081 (startlux-4b-mixed, v2 venv): E1's 3,991 decisions; the server stays up
#     c_replay   corpus C replayed through the served checkpoint (a new cache)
#     live_smoke configs/explore/e12_train/live_e12c_smoke.yaml + the projection gate ($15)
#     live       configs/explore/e12_train/live_e12c.yaml (JEV_EVAL_RUNS=1), passes 2-3 infrastructure errors only
#     fast       vLLM venv; LoRA merges (AgentDecision-4B, E12 from S3, E12c); E1 prompts per tokenizer; the timing of
#                the four models (scripts/e12_fast_engine.py)
#     final      stop the server, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e12/box_c on the box -> s3://$BUCKET/e12c/box; runs -> s3://$BUCKET/e12c/runs/<run>;
# the adapters -> s3://$BUCKET/checkpoints/e12c_sl4b/ (no optimizer state).
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E12C_IID=${E12C_IID:-}
# every other project host (E3's, E5's, Stage 2's, E6a's, E7's, E11's, E4r's, the author's): never touched by this driver
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
# us-east-1d first (E3's / E11's zone), then the other zones of the same VPC
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e12c-train
BUNDLE=e12c.bundle
S3OUT="$B/e12c"
CAP_USD=${CAP_USD:-15}
GUARD_MIN=${GUARD_MIN:-590}          # 10 GPU-hours cap (E12c (i)): the guard 590 min after launch
IDLE_MIN=${IDLE_MIN:-30}
WORKERS=${WORKERS:-4}
POWEROFF=${POWEROFF:-1}
NAME=startlux-4b-mixed
PORT=8081

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E12C_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E12C_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E12C_IID "* ]] && { echo "refused: $E12C_IID is not the E12c host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E12C_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E12C_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e12c-train: power-off guard (10 GPU-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g6e.2xlarge g6e.xlarge g6e.4xlarge g5.8xlarge g5.4xlarge)
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/e12c_train.sh" > "$tmp/e12c_train.sh"   # the committed driver
  awsl s3 cp "$tmp/e12c_train.sh" "$B/code/e12c_train.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e12c_train.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e12c_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e12c_train.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q GUARD_MIN=%q GUARD_END=%q IDLE_MIN=%q WORKERS=%q POWEROFF=%q E12C_IID=%q\n' \
      "$CAP_USD" "$GUARD_MIN" "${GUARD_END:-}" "$IDLE_MIN" "$WORKERS" "$POWEROFF" "$E12C_IID"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e12c_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e12c_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E12C_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e12c_driver.sh /tmp/e12c_driver.sh --only-show-errors && bash /tmp/e12c_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E12C_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E12C_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,36p' "$0"
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
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E12C_IID" && "$MY_IID" != "$E12C_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E12c host ${E12C_IID:-?}"; exit 1
fi
R=$W/e12c
A=$R/results/explore/e12/box_c
STATUS_DIR=$W/e12c_status
HV=$W/hvenv                                # harness venv (runner, merges, prompts)
VV=$W/v2venv                               # A7.19 (f)'s v2 venv: transformers 5.18 + fla / causal-conv1d (+ peft)
VLV=$W/vllmvenv                            # the fast engine (vLLM)
CK=$W/ck                                   # the frozen 4B: $CK/<plan dir>
SL=/opt/startlux/models/StartLux-Decision-4B
CKC=$W/ck_e12c/train                       # the trainer's output: checkpoint-<step> per epoch + the final adapter
SEL=$W/ck_e12c/selected                    # the selected adapter (+ temperatures)
PLAN=results/stage2/rl_eval_plan.json
STOP=runs/e12c_STOP.json
SPEND=scripts/explore_e12_train.py
FAST=scripts/e12_fast_engine.py
V1=(data/train_v1/dev.jsonl.gz data/train_v1/dev_recovery.jsonl.gz data/train_v1/dev_knockout.jsonl.gz)
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
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
  for d in e1_bfcl_when2call_e12c e12c_live_smoke e12c_live; do
    [[ -d "$R/runs/$d" ]] && aws s3 sync "$R/runs/$d" "$S3OUT/runs/$d" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  true
}
PY() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }        # harness venv
PV() { PYTHONPATH="$R/src" "$VV/bin/python" "$@"; }        # v2 venv (StartLux kernels)
plan() { (cd "$R" && PY -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"); }
qbase() { cat "$W/base_model_path.txt"; }                  # Qwen3-4B (AgentDecision-4B's and E12's base)
frozen_dir() { echo "$CK/$(plan "d['rows']['frozen']['dir']")"; }
base_c() { python3 -c "import json; print(json.load(open('$A/base_choice.json'))['path'])" 2>/dev/null || echo "$SL"; }
venv_c() { if [[ "$(base_c)" == "$SL" ]]; then echo "$VV"; else echo "$HV"; fi; }
PC() { PYTHONPATH="$R/src" "$(venv_c)/bin/python" "$@"; }   # the venv of E12c's base
keys() {
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
  PY "$SPEND" watch --exp e12c --pid "$pid" --cap "$CAP_USD" --runs runs --every 30 >> "$A/watch.log" 2>&1 &
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
  if [[ -f "$W/e12c_serve.pid" ]]; then
    while read -r p; do [[ -n "$p" ]] && kill "$p" 2>/dev/null; done < "$W/e12c_serve.pid"
  fi
  rm -f "$W/e12c_serve.pid"; sleep 8
}
server_up() { curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; }
serve_c() {  # the selected adapter + temperatures on :8081 (v2 venv for StartLux; no extra arguments)
  stop_server
  [[ -s "$SEL/adapter_model.safetensors" && -s "$SEL/calibration.json" ]] || { echo "no selected adapter / temperatures"; return 1; }
  local args=(--model "$(base_c)" --name "$NAME" --port "$PORT" --adapter "$SEL" --calibration "$SEL/calibration.json")
  echo "serve_jev ${args[*]} (venv $(venv_c))" > "$A/serve.cmd"
  nohup env PYTHONPATH="$R/src" "$(venv_c)/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$A/logs/serve.out" 2>&1 &
  echo $! > "$W/e12c_serve.pid"
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/health" > "$A/serve_health.json"; then
      curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/serve_models.json"
      grep -q "\"$NAME\"" "$A/serve_models.json" || { echo "served model id is not $NAME"; return 1; }
      printf '{"server_up": "%s", "instance_type": "%s", "instance_id": "%s"}\n' "$(date -u +%FT%TZ)" \
        "$(imds instance-type)" "$MY_IID" >> "$A/serve_windows.jsonl"
      return 0
    fi
    kill -0 "$(cat "$W/e12c_serve.pid")" 2>/dev/null || { echo "serve_jev died"; tail -20 "$A/logs/serve.out"; return 1; }
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
train_c() {  # train_c OUT SETTING [extra args]: the SFT v1 recipe from a fresh LoRA on E12's mix
  local out=$1 setting=$2; shift 2
  local flags; flags=$(PY scripts/train/ablation_box.py setting "$setting") || return 1
  rm -rf "$out"; mkdir -p "$out"
  # shellcheck disable=SC2086
  PC -m jevharness.training.train_decision --model "$(base_c)" --train "$A/data/train_mix.jsonl.gz" --out "$out" \
    --loss soft_ce_brier --brier-weight 0.5 --lora-r 32 --lora-alpha 64 --lr 1e-4 --epochs 2 --max-len 4096 --seed 0 \
    --report-to none $flags "$@"
}

stage_rearm() {
  if [[ -n "${GUARD_END:-}" ]]; then GUARD_MIN=$(( ( $(date -u -d "$GUARD_END" +%s) - $(date -u +%s) ) / 60 )); fi
  (( GUARD_MIN > 0 )) || { echo "guard end already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "jev-e12c-train: power-off guard (GPU cap)"
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
f=/var/tmp/jev-idle-count
if pgrep -f 'e12c_[0-9]|jevharness|jevbench|serve_jev|explore_e12|train_decision|stage2_corpus|e12_fast_engine|vllm|pip|uv ' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e12c-train: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $GUARD_MIN min, idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}
stage_setup() {
  local myb="$W/e12c.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A/logs" "$A/eval" "$A/sel" "$A/fast" runs
  st "code at $(git log --oneline -1 | cut -c1-72)"
  grep -q 'E12c: the recipe on a stronger base, and same-engine serving cost' docs/PREREG_A7_STAGE2.md \
    || { echo "no Deviations entry for E12c"; return 1; }
  ( prewarm "$SL" "$(qbase)" "$VV" "$HV/lib" ) &
  local pw=$!
  command -v uv > /dev/null || bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  "$VV/bin/python" -c "import peft" 2>/dev/null || uv pip install -q --python "$VV" "peft==0.21.2" || return 1
  PV -c "import jevharness, torch, transformers, peft, fla, causal_conv1d, sys; p = jevharness.__file__; print(p, torch.__version__, transformers.__version__, peft.__version__, fla.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_v2venv.txt" 2>&1 || { cat "$A/env_v2venv.txt"; echo "the v2 venv does not import this checkout"; return 1; }
  PY -c "import jevharness, torch, transformers, peft, sys; p = jevharness.__file__; print(p, torch.__version__, transformers.__version__, peft.__version__); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_hvenv.txt" 2>&1 || { cat "$A/env_hvenv.txt"; return 1; }
  cat "$A/env_v2venv.txt" "$A/env_hvenv.txt"
  wait "$pw"
  (cd "$SL" && sha256sum ./*.safetensors config.json tokenizer.json 2>/dev/null) > "$A/startlux_4b_sha256.txt"
  grep -h '"weights_revision"\|revision' "$SL"/.cache/huggingface/download/*.metadata 2>/dev/null | head -3 > "$A/startlux_4b_revision.txt"
  cp -f "$W/base_model_path.txt" "$A/base_model_path.txt"
  if ! verify_frozen; then
    aws s3 sync "$(plan "d['rows']['frozen']['s3']")" "$(frozen_dir)" --only-show-errors --region "$REGION" || return 1
    verify_frozen || return 1
  fi
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s"}\n' "$MY_IID" \
    "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" > "$A/host_setup.json"
}
stage_data() {
  cd "$R" || return 1
  mkdir -p "$A/data"
  aws s3 sync "$B/e12/data" "$A/data" --only-show-errors --region "$REGION" || return 1
  PY "$SPEND" verify --data "$A/data" || return 1
  PY "$SPEND" devprefix --out "$A/data/e1_devprefix.jsonl.gz" > "$A/devprefix.json" || return 1
  cat "$A/devprefix.json"
}
stage_smoke() {  # 4 training steps per memory setting: the setting and the projection of the two epochs (E12c (b))
  cd "$R" || return 1
  stop_server
  local base setting rc s steps sps proj
  for base in startlux qwen3-8b; do
    if [[ "$base" == qwen3-8b ]]; then
      st "smoke: falling back to Qwen3-8B (E12c (b))"
      PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B', local_dir='$W/models/Qwen3-8B'))" \
        > "$A/logs/qwen3_8b_download.log" 2>&1 || return 1
      printf '{"base": "qwen3-8b", "path": "%s"}\n' "$W/models/Qwen3-8B" > "$A/base_choice.json"
    else
      printf '{"base": "startlux", "path": "%s"}\n' "$SL" > "$A/base_choice.json"
    fi
    for setting in nogc_3072 gc_3072 gc_2048; do
      s=$(date +%s)
      timeout 40m bash -c "$(declare -f train_c PC PY base_c venv_c); A='$A' R='$R' HV='$HV' VV='$VV' SL='$SL' train_c '$W/ck_e12c/smoke' $setting --max-steps 4" \
        > "$A/logs/smoke_${base}_$setting.log" 2>&1
      rc=$?
      steps=$(grep -oE 'micro-batches/epoch, [0-9.]+ rows each on average, grad-accum [0-9]+' "$A/logs/smoke_${base}_$setting.log" | head -1)
      sps=$(PY -c "
import re, sys
t = open('$A/logs/smoke_${base}_$setting.log').read()
m = re.search(r\"'train_runtime': '?([0-9.]+)\", t)
print(float(m.group(1)) / 4 if m else '')")
      printf '{"base": "%s", "setting": "%s", "exit": %s, "wall_s": %s, "s_per_step": "%s", "budget": "%s"}\n' "$base" "$setting" \
        "$rc" "$(( $(date +%s) - s ))" "$sps" "$steps" >> "$A/smoke.jsonl"
      if (( rc == 0 )); then
        proj=$(PY -c "
import re
t = open('$A/logs/smoke_${base}_$setting.log').read()
m = re.search(r'token budget \d+: (\d+) micro-batches/epoch, [0-9.]+ rows each on average, grad-accum (\d+)', t)
steps = 2 * -(-int(m.group(1)) // int(m.group(2)))
print(round(steps * float('$sps') / 60))")
        st "smoke: $base $setting ok, ${sps}s / step, projected $proj min for two epochs"
        if (( proj <= 360 )); then
          python3 - "$A/base_choice.json" "$setting" "$sps" "$proj" <<'PYEOF'
import json, sys
p, setting, sps, proj = sys.argv[1:5]
d = json.load(open(p)); d.update({"setting": setting, "s_per_step": float(sps), "projected_min": int(proj)})
json.dump(d, open(p, "w")); print(d)
PYEOF
          rm -rf "$W/ck_e12c/smoke"
          return 0
        fi
        break    # too slow with the fastest setting that fits: the fallback base
      fi
      grep -qE 'out of memory|OutOfMemoryError' "$A/logs/smoke_${base}_$setting.log" || break
    done
  done
  echo "no base / setting passed the smoke within the cap"; return 1
}
stage_train() {
  cd "$R" || return 1
  stop_server
  local setting rc=1 t0 order
  setting=$(python3 -c "import json; print(json.load(open('$A/base_choice.json'))['setting'])") || return 1
  case "$setting" in nogc_3072) order="nogc_3072 gc_3072 gc_2048" ;; gc_3072) order="gc_3072 gc_2048" ;; *) order="gc_2048" ;; esac
  for setting in $order; do
    t0=$(date +%s)
    nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 10000 > "$A/logs/train_smi_$setting.csv" 2>/dev/null &
    local smi=$!
    timeout 360m bash -c "$(declare -f train_c PC PY base_c venv_c); A='$A' R='$R' HV='$HV' VV='$VV' SL='$SL' train_c '$CKC' $setting" \
      > "$A/logs/train_$setting.log" 2>&1
    rc=$?
    kill "$smi" 2>/dev/null
    printf '{"setting": "%s", "exit": %s, "minutes": %s}\n' "$setting" "$rc" "$(( ($(date +%s) - t0) / 60 ))" >> "$A/train_runs.jsonl"
    (( rc == 0 )) && break
    grep -qE 'out of memory|OutOfMemoryError' "$A/logs/train_$setting.log" || break
  done
  ls -d "$CKC"/checkpoint-* > "$A/checkpoints.txt" 2>/dev/null
  local c
  for c in "$CKC"/checkpoint-* "$CKC"; do
    [[ -s "$c/adapter_model.safetensors" ]] && sha256sum "$c/adapter_model.safetensors" >> "$A/e12c_sha256.txt"
  done
  aws s3 sync "$CKC" "$B/checkpoints/e12c_sl4b/train/" --only-show-errors --region "$REGION" \
    --exclude '*optimizer*' --exclude '*.pt' --exclude 'rng_state*' --exclude 'scheduler*'
  sync_out
  (( rc == 0 )) || [[ -n "$(ls -d "$CKC"/checkpoint-* 2>/dev/null)" ]] || return 1   # timed out: epoch 1 still selectable
  return 0
}
stage_select() {  # E12c (e): epoch-end adapters on E1's dev prefix + v1 dev; the higher mean of the two (tie: epoch 2)
  cd "$R" || return 1
  stop_server
  local c i=0 cands=()
  for c in $(ls -d "$CKC"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do [[ -s "$c/adapter_model.safetensors" ]] && cands+=("$c"); done
  [[ ${#cands[@]} -ge 1 ]] || { echo "no epoch checkpoint"; return 1; }
  for c in "${cands[@]}"; do   # the trainer's epoch checkpoints hold no tokenizer: the final one's (the base's) is copied
    cp -n "$CKC"/tokenizer.json "$CKC"/tokenizer_config.json "$CKC"/chat_template.jinja "$c/" 2>/dev/null
    i=$((i + 1))
    PC -m jevharness.training.eval_decisions --model "$(base_c)" --adapter "$c" \
      --dev "$A/data/e1_devprefix.jsonl.gz" data/train_v1/dev.jsonl.gz --n-perm 0 --max-len 4096 --batch-size 8 \
      --out "$A/sel/epoch$i.json" > "$A/logs/sel_epoch$i.log" 2>&1 || return 1
    echo "$c" > "$A/sel/epoch$i.path"
  done
  PY - "$A/sel" <<'PYEOF' || return 1
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
out = {"rule": "higher mean of E1 dev-prefix and v1-dev accuracy; tie -> epoch 2", "candidates": rows, "selected": best}
(d / "selection.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out))
PYEOF
  local pick; pick=$(python3 -c "import json; print(json.load(open('$A/sel/selection.json'))['selected']['path'])")
  rm -rf "$SEL"; mkdir -p "$SEL"
  cp -f "$pick"/adapter_model.safetensors "$pick"/adapter_config.json "$SEL/"
  cp -f "$CKC"/tokenizer* "$CKC"/chat_template.jinja "$SEL/" 2>/dev/null || cp -f "$(base_c)"/tokenizer* "$(base_c)"/chat_template.jinja "$SEL/" 2>/dev/null
  st "selected: $(python3 -c "import json; s=json.load(open('$A/sel/selection.json'))['selected']; print(s['epoch'], s['acc'])")"
  sync_out
}
stage_temps() {
  cd "$R" || return 1
  stop_server
  PC -m jevharness.training.calibrate --model "$(base_c)" --adapter "$SEL" --dev "$A/data/cal_dev.jsonl.gz" \
    --out "$SEL/calibration.json" --min-bucket 200 --batch-size 8 > "$A/logs/calibrate.log" 2>&1 || return 1
  cp -f "$SEL/calibration.json" "$A/e12c_calibration.json"
  sha256sum "$SEL/adapter_model.safetensors" "$SEL/calibration.json" > "$A/selected_sha256.txt"
  aws s3 sync "$SEL" "$B/checkpoints/e12c_sl4b/selected/" --only-show-errors --region "$REGION"
  sync_out
}
stage_offline() {
  cd "$R" || return 1
  stop_server
  PC -m jevharness.training.eval_decisions --model "$(base_c)" --adapter "$SEL" --calibration "$SEL/calibration.json" \
    --dev "$A/data/dev_w2c.jsonl.gz" "$A/data/dev_tool.jsonl.gz" "$A/data/dev_args.jsonl.gz" --n-perm 0 --max-len 4096 \
    --batch-size 8 --out "$A/eval/e12c_pubdev.json" > "$A/logs/eval_pub.log" 2>&1 || return 1
  PC -m jevharness.training.eval_decisions --model "$(base_c)" --adapter "$SEL" --calibration "$SEL/calibration.json" \
    --dev "${V1[@]}" --n-perm 0 --max-len 4096 --batch-size 8 --out "$A/eval/e12c_v1.json" > "$A/logs/eval_v1.log" 2>&1 || return 1
  sync_out
}
stage_e1() {
  cd "$R" || return 1
  serve_c || return 1
  local k
  for k in 1 2; do
    PY -m jevharness.bench.decision_eval configs/explore/e12_train/e1_e12c.yaml --out runs --only e12c_sl4b > "$A/logs/e1_$k.log" 2>&1
    st "e1 invocation $k exit=$?"
  done
  sync_out
}
stage_c_replay() {
  cd "$R" || return 1
  server_up || serve_c || return 1
  mkdir -p "$A/s23"
  local f k spec
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz; do cp -f "results/stage2/s23/$f" "$A/s23/$f" || return 1; done
  spec=$(printf '{"type": "jev", "name": "%s", "base_url": "http://127.0.0.1:%s", "model": "%s", "api_key": "", "timeout": 60.0}' \
    "$NAME" "$PORT" "$NAME")
  for k in 1 2 3; do
    PY scripts/stage2_corpus.py replay --corpus "$A/s23" --name e12c_sl4b --s1 "$spec" --cache cache/s1_e12c_c.sqlite \
      > "$A/logs/replay_c_$k.log" 2>&1
  done
  st "corpus C replayed: $(wc -l < "$A/s23/answers_e12c_sl4b.jsonl" 2>/dev/null) answers"
  sync_out
}
stage_live_smoke() {
  cd "$R" || return 1
  keys || return 1
  server_up || serve_c || return 1
  run_watched configs/explore/e12_train/live_e12c_smoke.yaml
  local rc=$?
  PY "$SPEND" project --exp e12c --runs runs --out "$A/projection.json" > "$A/logs/projection.out" 2>&1
  st "smoke exit=$rc; projection: $(python3 -c "import json; print(json.load(open('$A/projection.json'))['decision'])" 2>/dev/null)"
  leak_check || return 1
  return $rc
}
stage_live() {
  cd "$R" || return 1
  keys || return 1
  server_up || serve_c || return 1
  local dec pass rc t0 t1 ne
  dec=$(python3 -c "import json; print(json.load(open('$A/projection.json'))['decision'])" 2>/dev/null)
  [[ "$dec" == "run all 200" ]] || { st "refusing live: projection says '$dec'"; return 3; }
  export JEV_EVAL_RUNS=1
  printf 'step e12c_live\nstarted %s\ncommit %s\nconfig %s\nprereg sha256 %s\ncap_usd %s\nhost %s %s\n' \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$(sha256sum configs/explore/e12_train/live_e12c.yaml | cut -c1-16)" \
    "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" "$CAP_USD" "$MY_IID" "$(imds instance-type)" > "$A/START_live.txt"
  t0=$(date -u +%FT%TZ)
  for pass in 1 2 3; do
    [[ -f "$STOP" ]] && { st "watchdog stop recorded; no further pass"; break; }
    if (( pass > 1 )); then
      ne=$(errored runs/e12c_live)
      [[ "$ne" == 0 ]] && break
      st "$ne errored episode(s) after pass $((pass - 1)); pass $pass"; sleep 120
    fi
    st "live pass $pass"
    run_watched configs/explore/e12_train/live_e12c.yaml
    rc=$?
    st "live pass $pass exit=$rc; $(PY "$SPEND" spend --exp e12c --runs runs --short 2>/dev/null)"
  done
  t1=$(date -u +%FT%TZ)
  printf '{"window_start": "%s", "window_end": "%s", "instance_type": "%s", "instance_id": "%s", "workers": 4, "run": "runs/e12c_live"}\n' \
    "$t0" "$t1" "$(imds instance-type)" "$MY_IID" > "$A/live_window.json"
  leak_check || return 1
  [[ -f "$STOP" ]] && return 1
  return 0
}
stage_fast() {  # E12c (h): vLLM, merged LoRA; the four models on E1's 3,991 decisions, one at a time
  cd "$R" || return 1
  stop_server
  local F=$A/fast M=$W/merged
  mkdir -p "$F" "$M"
  if [[ ! -x "$VLV/bin/python" ]]; then
    uv venv -q -p 3.11 "$VLV" && timeout 40m uv pip install -q --python "$VLV" vllm > "$A/logs/vllm_install.log" 2>&1 \
      || { st "fast: vLLM install failed"; tail -5 "$A/logs/vllm_install.log"; return 1; }
  fi
  "$VLV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" > "$F/vllm_version.txt" 2>&1
  cat "$F/vllm_version.txt"
  # merged checkpoints (harness venv: peft + transformers 5.18)
  [[ -s "$M/frozen/config.json" ]] || PY "$FAST" merge --base "$(qbase)" --adapter "$(frozen_dir)" --out "$M/frozen" > "$A/logs/merge_frozen.log" 2>&1
  if [[ ! -s "$M/e12/config.json" ]]; then
    aws s3 sync "$B/checkpoints/e12_mixed/" "$W/ck_e12_from_s3/" --only-show-errors --region "$REGION" --exclude 'checkpoint-*' \
      && [[ -s "$W/ck_e12_from_s3/adapter_model.safetensors" ]] \
      && PY "$FAST" merge --base "$(qbase)" --adapter "$W/ck_e12_from_s3" --out "$M/e12" > "$A/logs/merge_e12.log" 2>&1
  fi
  [[ -s "$M/e12c/config.json" ]] || PV "$FAST" merge --base "$(base_c)" --adapter "$SEL" --out "$M/e12c" > "$A/logs/merge_e12c.log" 2>&1
  # prompts per tokenizer (serve_jev's rendering)
  [[ -s "$F/prompts_qwen.jsonl.gz" ]] || PY "$FAST" prompts --tokenizer "$(qbase)" --out "$F/prompts_qwen.jsonl.gz" > "$F/prompts_qwen.json" 2>&1
  [[ -s "$F/prompts_sl.jsonl.gz" ]] || PV "$FAST" prompts --tokenizer "$SL" --out "$F/prompts_sl.jsonl.gz" > "$F/prompts_sl.json" 2>&1
  local m path pr
  for m in frozen e12 e12c startlux4b; do
    case "$m" in
      frozen) path=$M/frozen; pr=$F/prompts_qwen.jsonl.gz ;;
      e12) path=$M/e12; pr=$F/prompts_qwen.jsonl.gz ;;
      e12c) path=$M/e12c; pr=$F/prompts_$( [[ "$(base_c)" == "$SL" ]] && echo sl || echo qwen ).jsonl.gz ;;
      startlux4b) path=$SL; pr=$F/prompts_sl.jsonl.gz ;;
    esac
    [[ -s "$path/config.json" ]] || { echo "{\"model\": \"$m\", \"state\": \"not available\"}" > "$F/time_$m.json"; continue; }
    timeout 60m "$VLV/bin/python" "$FAST" time --model "$path" --prompts "$pr" --label "$m" --out "$F/time_$m.json" \
      > "$A/logs/fast_$m.log" 2>&1 || echo "{\"model\": \"$m\", \"state\": \"vLLM failed: $(tail -1 "$A/logs/fast_$m.log" | tr '"' "'" | cut -c1-200)\"}" > "$F/time_$m.json"
    st "fast $m: $(python3 -c "import json; d=json.load(open('$F/time_$m.json')); print(d.get('state') or round(d['usd_per_1000_wall'], 4))" 2>/dev/null)"
  done
  sync_out
}
stage_final() {
  stop_server
  nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  printf '{"final_utc": "%s", "instance_id": "%s", "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" "$MY_IID" \
    "$(imds instance-type)" > "$A/host_final.json"
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; sudo shutdown -h +2 "jev-e12c-train: done"; fi
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
