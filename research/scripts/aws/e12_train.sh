#!/usr/bin/env bash
# Exploratory row E12, more training on public tool-use data (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E12:
# more training on public tool-use data"; descriptive, outside every Holm table). One NEW GPU host launched by this
# driver from E3's clone image (tag Name=jev-e12-train), SSM only. AgentDecision-4B's frozen adapter (sha256-checked
# against results/stage2/rl_eval_plan.json) is continued on the 14,000-row mix (7,000 public tool-use items in E1's
# format + 7,000 SFT v1 replay rows; built locally by scripts/explore_e12_train.py build, sha256-checked here), then
# scored offline (E1's 3,991 decisions, corpus C, held-out JevGym dev rows, the public dev split) and live (replace
# mode, Flash on the direct Gemini API, seeds 300000-300199). Gemini key from SSM Parameter Store into the live stages'
# process only (never printed, never on disk); no other key.
#
#   LOCAL:  bash scripts/aws/e12_train.sh launch [itype ...]    # one new host (refuses if a jev-e12-train host exists);
#                                                                # types in order (default g6e.2xlarge g6e.xlarge
#                                                                # g5.8xlarge g5.4xlarge), each in us-east-1d then the
#                                                                # other zones; user-data arms the power-off guard
#           bash scripts/aws/e12_train.sh data DIR              # the locally built data -> s3://$BUCKET/e12/data/
#           E12_IID=i-... bash scripts/aws/e12_train.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages (in the order sent; a failed stage stops the chain, `final` always runs when sent):
#     rearm      idle stop (IDLE_MIN) + power-off guard (GUARD_END absolute UTC, else GUARD_MIN), host record
#     setup      the committed branch (bundle) in /opt/jev/e12; the hvenv imports this checkout; the frozen 4B verified
#                (fetched from S3 if absent); base model path
#     data       s3://$BUCKET/e12/data -> $A/data, sha256 against results/explore/e12/build_info.json
#     train      continue the frozen adapter on train_mix (lr 5e-5, 1 epoch, 64 rows / step, max_len 4096, seed 0;
#                nogc_3072, then gc_3072 / gc_2048 on out-of-memory only; 180 min cap); adapter -> S3
#     temps      temperatures on cal_dev (calibrate.py --min-bucket 200)
#     offline    eval_decisions: both checkpoints on the v1 dev files and on the public dev split
#     e1_frozen  the frozen 4B served on :8081 (S2-4's server): E1's 3,991 decisions (serving-path check, same-host cost)
#     e1_e12     the new checkpoint served on :8081 (agentdecision-4b-mixed): E1's 3,991 decisions; the server stays up
#     c_replay   corpus C replayed through the served new checkpoint (scripts/stage2_corpus.py, a new cache)
#     live_smoke configs/explore/e12_train/live_smoke.yaml (2 development tasks) + the registered projection gate
#     live       configs/explore/e12_train/live.yaml (JEV_EVAL_RUNS=1), passes 2-3 re-run infrastructure errors only
#     final      stop the server, sync, power-off (POWEROFF=1)
#   E12b (Deviations "E12b: reading the order rule"; $0 API): b_inputs (D100, E7d's held-out tier files) and b_harvest
#     (CPU: harvest + rows; may run in a second chain beside `train`), b_wait (waits for b_harvest's rows), b_train
#     (E7d's continuation recipe, 120 min cap), b_eval (temperatures; E7d's held-out tier states, both checkpoints)
# Records: $A = results/explore/e12/box on the box -> s3://$BUCKET/e12/box; runs -> s3://$BUCKET/e12/runs/<run>; the
# adapter -> s3://$BUCKET/checkpoints/e12_mixed/ (no optimizer state).
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E12_IID=${E12_IID:-}
# every other project host (E3's, E5's, Stage 2's, E6a's, E7's, E11's, E4r's, the author's): never touched by this driver
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
# us-east-1d first (E3's / E11's zone), then the other zones of the same VPC
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e12-train
BUNDLE=e12.bundle
S3OUT="$B/e12"
CAP_USD=${CAP_USD:-30}
GUARD_MIN=${GUARD_MIN:-690}          # 12 GPU-hours cap (registration (i)): the guard 11.5 h after launch
IDLE_MIN=${IDLE_MIN:-30}
WORKERS=${WORKERS:-4}
POWEROFF=${POWEROFF:-1}
NAME=agentdecision-4b-mixed
PORT=8081

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E12_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E12_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E12_IID "* ]] && { echo "refused: $E12_IID is not the E12 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E12_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E12_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e12-train: power-off guard (12 GPU-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g6e.2xlarge g6e.xlarge g5.8xlarge g5.4xlarge)
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
  python3 "$ROOT/scripts/explore_e12_train.py" verify --data "$D" || exit 1   # the committed record
  awsl s3 sync "$D" "$S3OUT/data" --only-show-errors --exclude '*' --include '*.jsonl.gz' --include '*.json' || exit 1
  echo "data -> $S3OUT/data"
  exit 0
fi
if [[ "$MODE" == push ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null || { echo "bundle failed"; exit 1; }
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors || exit 1
  git -C "$ROOT" show "$BRANCH:scripts/aws/e12_train.sh" > "$tmp/e12_train.sh"   # the committed driver
  awsl s3 cp "$tmp/e12_train.sh" "$B/code/e12_train.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e12_train.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e12_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e12_train.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q GUARD_MIN=%q GUARD_END=%q IDLE_MIN=%q WORKERS=%q POWEROFF=%q E12_IID=%q\n' \
      "$CAP_USD" "$GUARD_MIN" "${GUARD_END:-}" "$IDLE_MIN" "$WORKERS" "$POWEROFF" "$E12_IID"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e12_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e12_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E12_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e12_driver.sh /tmp/e12_driver.sh --only-show-errors && bash /tmp/e12_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E12_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E12_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,38p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
W=${W:-/opt/jev}
DRY_RUN=${DRY_RUN:-0}
cd "$W" || exit 1
imds() {
  curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" "http://169.254.169.254/latest/meta-data/$1"
}
if [[ "$DRY_RUN" == 1 ]]; then MY_IID=${E12_IID:-i-dryrun}; else MY_IID=$(imds instance-id); fi
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E12_IID" && "$MY_IID" != "$E12_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E12 host ${E12_IID:-?}"; exit 1
fi
R=$W/e12
A=$R/results/explore/e12/box
STATUS_DIR=$W/e12_status
HV=$W/hvenv
CK=$W/ck                                   # the frozen 4B: $CK/<plan dir>
E12_CK=$W/ck_e12/mixed                     # AgentDecision-4B (mixed): the continued adapter (a NEW checkpoint)
PLAN=results/stage2/rl_eval_plan.json
STOP=runs/e12_STOP.json
SPEND=scripts/explore_e12_train.py
V1=(data/train_v1/dev.jsonl.gz data/train_v1/dev_recovery.jsonl.gz data/train_v1/dev_knockout.jsonl.gz)
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
  [[ -d "$A" ]] && X aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION" --exclude 'data/*' \
    --exclude 's23/corpus.json' --exclude 's23/occurrences.jsonl.gz' --exclude 's23/questions.jsonl.gz' --exclude 'fast/prompts_*'
  local d
  for d in e1_bfcl_when2call_e12 e12_live_smoke e12_live; do
    [[ -d "$R/runs/$d" ]] && X aws s3 sync "$R/runs/$d" "$S3OUT/runs/$d" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  true
}
PY() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }
plan() { (cd "$R" && PY -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"); }
base() { cat "$W/base_model_path.txt"; }
frozen_dir() { echo "$CK/$(plan "d['rows']['frozen']['dir']")"; }
frozen_cal() { echo "$CK/$(plan "d['rows']['frozen']['calibration']")"; }
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
run_watched() {  # run_watched CONFIG [runner args]: one runner invocation under the $ watchdog
  local cfg=$1; shift
  if [[ "$DRY_RUN" == 1 ]]; then X jevbench run "$cfg" --out runs "$@"; return 0; fi
  setsid env PYTHONPATH="$R/src" "$HV/bin/jevbench" run "$cfg" --out runs "$@" >> "$A/runner.log" 2>&1 < /dev/null &
  local pid=$!
  PY "$SPEND" watch --pid "$pid" --cap "$CAP_USD" --runs runs --every 30 >> "$A/watch.log" 2>&1 &
  local wpid=$!
  ( while kill -0 "$pid" 2>/dev/null; do sleep 300; sync_out; done ) & local spid=$!
  wait "$pid"; local rc=$?
  kill "$wpid" "$spid" 2>/dev/null
  sync_out
  return $rc
}
errored() {  # errored RUN: tasks whose latest record (any arm) ended in an infrastructure error
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
  if [[ -f "$W/e12_serve.pid" ]]; then
    while read -r p; do [[ -n "$p" ]] && kill "$p" 2>/dev/null; done < "$W/e12_serve.pid"
  fi
  rm -f "$W/e12_serve.pid"; [[ "$DRY_RUN" == 1 ]] || sleep 8
}
server_up() { [[ "$DRY_RUN" == 1 ]] || curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; }
serve() {  # serve NAME ADAPTER CALIBRATION: one server on :8081 (no extra arguments: the plan's serve_args are empty)
  local name=$1 adapter=$2 cal=$3
  stop_server
  [[ "$DRY_RUN" == 1 ]] && { X serve_jev "$name" "$adapter" "$cal"; return 0; }
  local args=(--model "$(base)" --name "$name" --port "$PORT" --adapter "$adapter" --calibration "$cal")
  echo "serve_jev ${args[*]}" > "$A/serve_$name.cmd"
  nohup env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$A/logs/serve_$name.out" 2>&1 &
  echo $! > "$W/e12_serve.pid"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" > "$A/serve_health_$name.json"; then
      curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/serve_models_$name.json"
      grep -q "\"$name\"" "$A/serve_models_$name.json" || { echo "served model id is not $name"; return 1; }
      printf '{"server_up": "%s", "name": "%s", "instance_type": "%s", "instance_id": "%s"}\n' "$(date -u +%FT%TZ)" \
        "$name" "$(imds instance-type)" "$MY_IID" >> "$A/serve_windows.jsonl"
      return 0
    fi
    kill -0 "$(cat "$W/e12_serve.pid")" 2>/dev/null || { echo "serve_jev died"; tail -20 "$A/logs/serve_$name.out"; return 1; }
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
e12_ready() { [[ -s "$E12_CK/adapter_model.safetensors" && -s "$A/cal/e12_calibration.json" ]] && \
  sha256sum -c --quiet "$A/e12_sha256.txt" > /dev/null 2>&1; }

stage_rearm() {
  if [[ -n "${GUARD_END:-}" ]]; then   # an absolute end (UTC): the GPU cap counts from launch
    GUARD_MIN=$(( ( $(date -u -d "$GUARD_END" +%s) - $(date -u +%s) ) / 60 ))
  fi
  (( GUARD_MIN > 0 )) || { echo "guard end already passed"; return 1; }
  X sudo shutdown -c 2>/dev/null; X sudo shutdown -h +"$GUARD_MIN" "jev-e12-train: power-off guard (GPU cap)"
  if [[ "$DRY_RUN" != 1 ]]; then
    sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E12 chain / server / runner / trainer process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e12_[0-9]|jevharness|jevbench|serve_jev|explore_e12|train_decision|stage2_corpus' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e12-train: idle $IDLE_MIN min"; fi
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
  local myb="$W/e12.$$.bundle"
  X aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  if [[ "$DRY_RUN" != 1 ]]; then
    [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
    cd "$R" || return 1
    git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
    rm -f "$myb"
  fi
  cd "$R" || return 1
  mkdir -p "$A/logs" "$A/eval" "$A/cal" runs
  st "code at $(git log --oneline -1 | cut -c1-72)"
  grep -q 'E12: more training on public tool-use data' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E12"; return 1; }
  [[ "$DRY_RUN" == 1 ]] && return 0
  PY -c "import jevharness, torch, transformers, peft, sys; p = jevharness.__file__; print(p, torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env.txt" 2>&1 || { cat "$A/env.txt"; echo "the hvenv does not import this checkout"; return 1; }
  cat "$A/env.txt"
  [[ -x "$HV/bin/jevbench" ]] || { echo "no jevbench in the hvenv"; return 1; }
  [[ -s "$W/base_model_path.txt" ]] || { echo "no base_model_path.txt"; return 1; }
  cp -f "$W/base_model_path.txt" "$A/base_model_path.txt"
  if ! verify_frozen; then
    st "the frozen 4B is absent or different in $CK; fetching $(plan "d['rows']['frozen']['s3']")"
    aws s3 sync "$(plan "d['rows']['frozen']['s3']")" "$(frozen_dir)" --only-show-errors --region "$REGION" || return 1
    verify_frozen || return 1
  fi
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s"}\n' "$MY_IID" \
    "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" > "$A/host_setup.json"
  df -BG / | awk 'NR==2{print "disk free: "$4}'
}
stage_data() {
  cd "$R" || return 1
  mkdir -p "$A/data"
  X aws s3 sync "$S3OUT/data" "$A/data" --only-show-errors --region "$REGION" || return 1
  [[ "$DRY_RUN" == 1 ]] && return 0
  PY "$SPEND" verify --data "$A/data" || return 1
  cp -f "$A/data/build_info.json" "$A/build_info_s3.json" 2>/dev/null || true
}
stage_train() {  # continue the frozen adapter (E7d's continuation recipe); nogc_3072, then gc_3072 / gc_2048 on OOM only;
  # capped at 240 min (Deviations "E12: training cap on the A10G host"; registered 180 min)
  cd "$R" || return 1
  stop_server
  verify_frozen || return 1
  local setting flags rc=1 t0 t1
  for setting in nogc_3072 gc_3072 gc_2048; do
    flags=$(PY scripts/train/ablation_box.py setting "$setting") || return 1
    X rm -rf "$E12_CK"; mkdir -p "$E12_CK"
    t0=$(date +%s)
    [[ "$DRY_RUN" == 1 ]] || { nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 5000 \
      > "$A/logs/train_smi_$setting.csv" 2>/dev/null & echo $! > "$W/e12_smi.pid"; }
    # shellcheck disable=SC2086
    X timeout 240m env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.train_decision --model "$(base)" \
      --init "$(frozen_dir)" --train "$A/data/train_mix.jsonl.gz" --out "$E12_CK" --loss soft_ce_brier --brier-weight 0.5 \
      --lr 5e-5 --epochs 1 --batch-size 2 --max-len 4096 --seed 0 --report-to none $flags > "$A/logs/train_$setting.log" 2>&1
    rc=$?
    [[ -f "$W/e12_smi.pid" ]] && { kill "$(cat "$W/e12_smi.pid")" 2>/dev/null; rm -f "$W/e12_smi.pid"; }
    t1=$(date +%s)
    printf '{"setting": "%s", "exit": %s, "minutes": %s}\n' "$setting" "$rc" "$(( (t1 - t0) / 60 ))" >> "$A/train_runs.jsonl"
    (( rc == 0 )) && break
    grep -qE 'out of memory|OutOfMemoryError' "$A/logs/train_$setting.log" || return 1
  done
  (( rc == 0 )) || return 1
  [[ "$DRY_RUN" == 1 ]] && return 0
  cp -f "$(frozen_dir)"/tokenizer* "$(frozen_dir)"/chat_template.jinja "$E12_CK/" 2>/dev/null
  sha256sum "$E12_CK"/adapter_model.safetensors "$E12_CK"/adapter_config.json > "$A/e12_sha256.txt"
  PY - "$A" <<'PYEOF'
import json, sys
from pathlib import Path
a = Path(sys.argv[1])
runs = [json.loads(x) for x in (a / "train_runs.jsonl").read_text().splitlines() if x.strip()]
ok = [r for r in runs if r["exit"] == 0][-1]
info = json.loads((a / "data" / "build_info.json").read_text())
rec = {"base_model": (a / "base_model_path.txt").read_text().strip(), "init": "the frozen AgentDecision-4B adapter",
       "rows": info["rows"]["train_mix"], "epochs": 1, "lr": 5e-5, "loss": "soft_ce_brier", "brier_weight": 0.5,
       "rows_per_step": 64, "max_len": 4096, "seed": 0, "memory_setting": ok["setting"], "minutes": ok["minutes"],
       "attempts": runs, "adapter_sha256": (a / "e12_sha256.txt").read_text().split()[0::2]}
rec["summary"] = (f"AgentDecision-4B's adapter continued on {rec['rows']} rows (7,000 public + 7,000 replay), 1 epoch, "
                  f"lr 5e-5, soft CE + 0.5 Brier, {ok['minutes']} min ({ok['setting']})")
(a / "train_record.json").write_text(json.dumps(rec, indent=1) + "\n")
print(rec["summary"])
PYEOF
  aws s3 sync "$E12_CK" "$B/checkpoints/e12_mixed/" --only-show-errors --region "$REGION" --exclude 'checkpoint-*' \
    --exclude '*optimizer*' --exclude '*.pt' --exclude 'rng_state*'
  sync_out
}
stage_temps() {
  cd "$R" || return 1
  stop_server
  X PY -m jevharness.training.calibrate --model "$(base)" --adapter "$E12_CK" --dev "$A/data/cal_dev.jsonl.gz" \
    --out "$A/cal/e12_calibration.json" --min-bucket 200 --batch-size 8 > "$A/logs/calibrate.log" 2>&1 || return 1
  [[ "$DRY_RUN" == 1 ]] && return 0
  cp -f "$A/cal/e12_calibration.json" "$E12_CK/calibration.json"
  sha256sum "$A/cal/e12_calibration.json" >> "$A/e12_sha256.txt"
  aws s3 cp "$A/cal/e12_calibration.json" "$B/checkpoints/e12_mixed/calibration.json" --only-show-errors --region "$REGION"
  sync_out
}
stage_offline() {  # both checkpoints, the same pipeline: v1 dev files (n-perm 0) and the public dev split per family
  cd "$R" || return 1
  stop_server
  local pub=("$A/data/dev_w2c.jsonl.gz" "$A/data/dev_tool.jsonl.gz" "$A/data/dev_args.jsonl.gz")
  [[ "$DRY_RUN" == 1 ]] || e12_ready || { echo "no verified E12 adapter / temperatures"; return 1; }
  X PY -m jevharness.training.eval_decisions --model "$(base)" --adapter "$E12_CK" --calibration "$A/cal/e12_calibration.json" \
    --dev "${pub[@]}" --n-perm 0 --max-len 4096 --batch-size 8 --out "$A/eval/e12_pubdev.json" > "$A/logs/eval_e12_pub.log" 2>&1 || return 1
  X PY -m jevharness.training.eval_decisions --model "$(base)" --adapter "$(frozen_dir)" --calibration "$(frozen_cal)" \
    --dev "${pub[@]}" --n-perm 0 --max-len 4096 --batch-size 8 --out "$A/eval/frozen_pubdev.json" > "$A/logs/eval_frozen_pub.log" 2>&1 || return 1
  st "offline: public dev done"
  X PY -m jevharness.training.eval_decisions --model "$(base)" --adapter "$E12_CK" --calibration "$A/cal/e12_calibration.json" \
    --dev "${V1[@]}" --n-perm 0 --max-len 4096 --batch-size 8 --out "$A/eval/e12_v1.json" > "$A/logs/eval_e12_v1.log" 2>&1 || return 1
  X PY -m jevharness.training.eval_decisions --model "$(base)" --adapter "$(frozen_dir)" --calibration "$(frozen_cal)" \
    --dev "${V1[@]}" --n-perm 0 --max-len 4096 --batch-size 8 --out "$A/eval/frozen_v1.json" > "$A/logs/eval_frozen_v1.log" 2>&1 || return 1
  sync_out
}
e1() {  # e1 BACKEND: E1's decisions, two invocations as E1 (the second resumes any failed request from the cache)
  local k
  for k in 1 2; do
    X PY -m jevharness.bench.decision_eval configs/explore/e12_train/e1_e12.yaml --out runs --only "$1" > "$A/logs/e1_$1_$k.log" 2>&1
    st "e1 $1 invocation $k exit=$?"
  done
  sync_out
}
stage_e1_frozen() {
  cd "$R" || return 1
  verify_frozen || return 1
  serve jevagentrl-4b "$(frozen_dir)" "$(frozen_cal)" || return 1
  e1 frozen_rehost
  stop_server
  X PY "$SPEND" rehost --run runs/e1_bfcl_when2call_e12 --out "$A/rehost.json" > "$A/logs/rehost.log" 2>&1
  st "serving-path check: $(tr '\n' ' ' < "$A/logs/rehost.log" | cut -c1-200)"
  true   # reported either way (registration (f) 1)
}
stage_e1_e12() {
  cd "$R" || return 1
  [[ "$DRY_RUN" == 1 ]] || e12_ready || { echo "no verified E12 adapter / temperatures"; return 1; }
  serve "$NAME" "$E12_CK" "$A/cal/e12_calibration.json" || return 1
  e1 e12_mixed
}
stage_c_replay() {
  cd "$R" || return 1
  server_up || serve "$NAME" "$E12_CK" "$A/cal/e12_calibration.json" || return 1
  mkdir -p "$A/s23"
  local f k spec
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz; do cp -f "results/stage2/s23/$f" "$A/s23/$f" || return 1; done
  spec=$(printf '{"type": "jev", "name": "%s", "base_url": "http://127.0.0.1:%s", "model": "%s", "api_key": "", "timeout": 60.0}' \
    "$NAME" "$PORT" "$NAME")
  for k in 1 2 3; do   # the replay resumes; a scoring failure is retried inside
    X PY scripts/stage2_corpus.py replay --corpus "$A/s23" --name e12_mixed --s1 "$spec" --cache cache/s1_e12_c.sqlite \
      > "$A/logs/replay_c_$k.log" 2>&1
  done
  st "corpus C replayed: $(wc -l < "$A/s23/answers_e12_mixed.jsonl" 2>/dev/null) answers"
  sync_out
}
stage_live_smoke() {
  cd "$R" || return 1
  keys || return 1
  server_up || serve "$NAME" "$E12_CK" "$A/cal/e12_calibration.json" || return 1
  run_watched configs/explore/e12_train/live_smoke.yaml
  local rc=$?
  X PY "$SPEND" project --runs runs --out "$A/projection.json" > "$A/logs/projection.out" 2>&1
  st "smoke exit=$rc; projection: $(PY -c "import json; print(json.load(open('$A/projection.json'))['decision'])" 2>/dev/null)"
  leak_check || return 1
  return $rc
}
stage_live() {
  cd "$R" || return 1
  keys || return 1
  server_up || serve "$NAME" "$E12_CK" "$A/cal/e12_calibration.json" || return 1
  local dec pass rc t0 t1 ne
  dec=$(PY -c "import json; print(json.load(open('$A/projection.json'))['decision'])" 2>/dev/null)
  [[ "$DRY_RUN" == 1 ]] && dec="run all 200"
  [[ "$dec" == "run all 200" ]] || { st "refusing live: projection says '$dec'"; return 3; }
  export JEV_EVAL_RUNS=1      # this stage's subshell only (the Stage-2 evaluation seeds 300000-300199)
  printf 'step e12_live\nstarted %s\ncommit %s\nconfig live.yaml %s\nprereg sha256 %s\ncap_usd %s\nhost %s %s\n' \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$(sha256sum configs/explore/e12_train/live.yaml | cut -c1-16)" \
    "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" "$CAP_USD" "$MY_IID" "$(imds instance-type)" > "$A/START_live.txt"
  t0=$(date -u +%FT%TZ)
  for pass in 1 2 3; do
    [[ -f "$STOP" ]] && { st "watchdog stop recorded; no further pass"; break; }
    if (( pass > 1 )); then
      ne=$(errored runs/e12_live)
      [[ "$ne" == 0 ]] && { st "no errored episode after pass $((pass - 1)); no further pass"; break; }
      st "$ne errored episode(s) after pass $((pass - 1)); pass $pass"
      [[ "$DRY_RUN" == 1 ]] || sleep 120
    fi
    st "live pass $pass"
    run_watched configs/explore/e12_train/live.yaml
    rc=$?
    st "live pass $pass exit=$rc; $(PY "$SPEND" spend --runs runs --short 2>/dev/null)"
    [[ "$DRY_RUN" == 1 ]] && break
  done
  t1=$(date -u +%FT%TZ)
  printf '{"window_start": "%s", "window_end": "%s", "instance_type": "%s", "instance_id": "%s", "workers": 4, "run": "runs/e12_live"}\n' \
    "$t0" "$t1" "$(imds instance-type)" "$MY_IID" > "$A/live_window.json"
  leak_check || return 1
  [[ -f "$STOP" ]] && return 1
  return 0
}
# ---------------------------------------------------------------- E12b (Deviations "E12b: reading the order rule"; $0 API)
B_CK=$W/ck_e12b/random                    # AgentDecision-4B (random rules): E7d's continuation on random-rule states
B_A=$A/b
stage_b_inputs() {  # D100 (E3's builder), E7d's held-out tier files (S3, sha256-checked by build-b)
  cd "$R" || return 1
  grep -q 'E12b: reading the order rule' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E12b"; return 1; }
  mkdir -p "$B_A/data" "$B_A/e7d" "$B_A/eval" "$B_A/cal" "$B_A/logs"
  X PY scripts/train/ablation_subsets.py build --out "$B_A/data" --arms D100 > "$B_A/logs/d100.log" 2>&1 \
    || { tail -5 "$B_A/logs/d100.log"; return 1; }
  local f
  for f in build_info.json heldout_op.jsonl heldout_ts_other.jsonl heldout_tp.jsonl; do
    X aws s3 cp "$B/e7_hard/e7d/box/data/$f" "$B_A/e7d/$f" --only-show-errors --region "$REGION" || return 1
  done
}
stage_b_harvest() {  # $0, CPU: the simulated oracle executor + NearMiss System-1 on random-rule tasks (seeds 322000-322599)
  cd "$R" || return 1
  X PY "$SPEND" harvest-b --out runs/e12b_harvest --workers 16 > "$B_A/logs/harvest.log" 2>&1 \
    || { tail -5 "$B_A/logs/harvest.log"; return 1; }
  X PY "$SPEND" build-b --harvest runs/e12b_harvest --d100 "$B_A/data/D100.jsonl" --e7d-data "$B_A/e7d" --out "$B_A/data" \
    > "$B_A/logs/build.log" 2>&1 || { tail -5 "$B_A/logs/build.log"; return 1; }
  cat "$B_A/logs/build.log"
  sync_out
}
stage_b_wait() {  # b_harvest may run in a second chain: wait for its rows (at most B_WAIT_MIN minutes)
  local end=$(( $(date +%s) + ${B_WAIT_MIN:-150} * 60 ))
  while (( $(date +%s) < end )); do
    [[ -s "$B_A/data/build_info.json" ]] && { st "E12b rows ready"; return 0; }
    if ! pgrep -f 'harvest-b|build-b|ablation_subsets' > /dev/null; then
      sleep 30; [[ -s "$B_A/data/build_info.json" ]] && return 0
      echo "no E12b harvest process and no rows"; return 1
    fi
    sleep 30
  done
  echo "E12b rows not ready in time"; return 1
}
stage_b_train() {  # E7d's continuation recipe: gc_3072, then gc_2048 on out-of-memory only; capped at 120 min
  cd "$R" || return 1
  stop_server
  verify_frozen || return 1
  [[ "$DRY_RUN" == 1 || -s "$B_A/data/train_mix.jsonl" ]] || { echo "no E12b train_mix.jsonl (b_harvest)"; return 1; }
  local setting flags rc=1 t0
  for setting in gc_3072 gc_2048; do
    flags=$(PY scripts/train/ablation_box.py setting "$setting") || return 1
    X rm -rf "$B_CK"; mkdir -p "$B_CK"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    X timeout 120m env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.train_decision --model "$(base)" \
      --init "$(frozen_dir)" --train "$B_A/data/train_mix.jsonl" --out "$B_CK" --loss soft_ce_brier --brier-weight 0.5 \
      --lr 5e-5 --epochs 1 --batch-size 2 --max-len 4096 --seed 0 --report-to none $flags > "$B_A/logs/train_$setting.log" 2>&1
    rc=$?
    printf '{"setting": "%s", "exit": %s, "minutes": %s}\n' "$setting" "$rc" "$(( ($(date +%s) - t0) / 60 ))" >> "$B_A/train_runs.jsonl"
    (( rc == 0 )) && break
    grep -qE 'out of memory|OutOfMemoryError' "$B_A/logs/train_$setting.log" || return 1
  done
  (( rc == 0 )) || return 1
  [[ "$DRY_RUN" == 1 ]] && return 0
  sha256sum "$B_CK"/adapter_model.safetensors "$B_CK"/adapter_config.json > "$B_A/e12b_sha256.txt"
  PY -c "
import json
runs = [json.loads(x) for x in open('$B_A/train_runs.jsonl') if x.strip()]
ok = [r for r in runs if r['exit'] == 0][-1]
json.dump({'rows': sum(1 for _ in open('$B_A/data/train_mix.jsonl')), 'epochs': 1, 'lr': 5e-5, 'memory_setting': ok['setting'],
           'minutes': ok['minutes'], 'attempts': runs}, open('$B_A/train_record.json', 'w'), indent=1)"
  aws s3 sync "$B_CK" "$B/checkpoints/e12b_random/" --only-show-errors --region "$REGION" --exclude 'checkpoint-*' \
    --exclude '*optimizer*' --exclude '*.pt' --exclude 'rng_state*'
  sync_out
}
stage_b_eval() {  # temperatures (1,000 v1 dev + 1,000 E12b dev rows); E7d's held-out tier states, both checkpoints
  cd "$R" || return 1
  stop_server
  local held=("$B_A/data/heldout_op.jsonl" "$B_A/data/heldout_ts_other_s.jsonl" "$B_A/data/heldout_tp_s.jsonl")
  X PY -m jevharness.training.calibrate --model "$(base)" --adapter "$B_CK" --dev "$B_A/data/cal_dev_small.jsonl" \
    --out "$B_A/cal/e12b_calibration.json" --min-bucket 200 --batch-size 8 > "$B_A/logs/calibrate.log" 2>&1 || return 1
  X PY -m jevharness.training.eval_decisions --model "$(base)" --adapter "$B_CK" --calibration "$B_A/cal/e12b_calibration.json" \
    --dev "${held[@]}" --n-perm 0 --max-len 4096 --batch-size 8 --out "$B_A/eval/e12b_heldout.json" > "$B_A/logs/eval_e12b.log" 2>&1 || return 1
  X PY -m jevharness.training.eval_decisions --model "$(base)" --adapter "$(frozen_dir)" --calibration "$(frozen_cal)" \
    --dev "${held[@]}" --n-perm 0 --max-len 4096 --batch-size 8 --out "$B_A/eval/frozen_heldout.json" > "$B_A/logs/eval_frozen.log" 2>&1 || return 1
  sync_out
}
# ---------------------------------------------------------------- E12c (h): the same-engine timing on THIS host
# (Deviations "E12c: same-engine timing on the E12 host"): vLLM with merged LoRA; AgentDecision-4B, E12, StartLux-4B,
# then E12c once its selected adapter is in s3://$BUCKET/checkpoints/e12c_sl4b/selected/ (waits FAST_WAIT_MIN at most)
VV=$W/v2venv
VLV=$W/vllmvenv
SLB=/opt/startlux/models/StartLux-Decision-4B
FAST=scripts/e12_fast_engine.py
stage_fast() {
  cd "$R" || return 1
  stop_server
  local F=$A/fast M=$W/merged m path pr
  mkdir -p "$F" "$M"
  command -v uv > /dev/null || bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  "$VV/bin/python" -c "import peft" 2>/dev/null || uv pip install -q --python "$VV" "peft==0.21.2" || return 1
  if [[ ! -x "$VLV/bin/python" ]]; then
    uv venv -q -p 3.11 "$VLV" && timeout 40m uv pip install -q --python "$VLV" vllm > "$A/logs/vllm_install.log" 2>&1 \
      || { st "fast: vLLM install failed"; tail -5 "$A/logs/vllm_install.log"; return 1; }
  fi
  "$VLV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" > "$F/vllm_version.txt" 2>&1
  cat "$F/vllm_version.txt"
  [[ -s "$M/frozen/config.json" ]] || PY "$FAST" merge --base "$(base)" --adapter "$(frozen_dir)" --out "$M/frozen" > "$A/logs/merge_frozen.log" 2>&1
  [[ -s "$M/e12/config.json" ]] || PY "$FAST" merge --base "$(base)" --adapter "$E12_CK" --out "$M/e12" > "$A/logs/merge_e12.log" 2>&1
  [[ -s "$F/prompts_qwen.jsonl.gz" ]] || PY "$FAST" prompts --tokenizer "$(base)" --out "$F/prompts_qwen.jsonl.gz" > "$F/prompts_qwen.json" 2>&1
  [[ -s "$F/prompts_sl.jsonl.gz" ]] || PYTHONPATH="$R/src" "$VV/bin/python" "$FAST" prompts --tokenizer "$SLB" \
    --out "$F/prompts_sl.jsonl.gz" > "$F/prompts_sl.json" 2>&1
  for m in frozen e12 startlux4b e12c; do
    if [[ "$m" == e12c && ! -s "$M/e12c/config.json" ]]; then
      local end=$(( $(date +%s) + ${FAST_WAIT_MIN:-120} * 60 ))
      until aws s3 ls "$B/checkpoints/e12c_sl4b/selected/calibration.json" --region "$REGION" > /dev/null 2>&1; do
        (( $(date +%s) < end )) || break; sleep 60
      done
      aws s3 sync "$B/checkpoints/e12c_sl4b/selected/" "$W/ck_e12c_selected/" --only-show-errors --region "$REGION" \
        && [[ -s "$W/ck_e12c_selected/adapter_model.safetensors" ]] \
        && PYTHONPATH="$R/src" "$VV/bin/python" "$FAST" merge --base "$SLB" --adapter "$W/ck_e12c_selected" --out "$M/e12c" \
          > "$A/logs/merge_e12c.log" 2>&1
    fi
    case "$m" in
      frozen) path=$M/frozen; pr=$F/prompts_qwen.jsonl.gz ;;
      e12) path=$M/e12; pr=$F/prompts_qwen.jsonl.gz ;;
      startlux4b) path=$SLB; pr=$F/prompts_sl.jsonl.gz ;;
      e12c) path=$M/e12c; pr=$F/prompts_sl.jsonl.gz ;;
    esac
    [[ -s "$path/config.json" ]] || { echo "{\"model\": \"$m\", \"state\": \"not available\"}" > "$F/time_$m.json"; continue; }
    timeout 60m "$VLV/bin/python" "$FAST" time --model "$path" --prompts "$pr" --label "$m" --out "$F/time_$m.json" \
      > "$A/logs/fast_$m.log" 2>&1 \
      || echo "{\"model\": \"$m\", \"state\": \"vLLM failed: $(tail -1 "$A/logs/fast_$m.log" | tr '"' "'" | cut -c1-200)\"}" > "$F/time_$m.json"
    st "fast $m: $(python3 -c "import json; d=json.load(open('$F/time_$m.json')); print(d.get('state') or round(d['usd_per_1000_wall'], 4))" 2>/dev/null)"
    sync_out
  done
}
stage_final() {
  stop_server
  [[ "$DRY_RUN" == 1 ]] || nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  printf '{"final_utc": "%s", "instance_id": "%s", "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" "$MY_IID" \
    "$(imds instance-type)" > "$A/host_final.json"
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; X sudo shutdown -h +2 "jev-e12-train: done"; fi
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
