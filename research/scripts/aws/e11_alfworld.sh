#!/usr/bin/env bash
# Exploratory row E11, the training recipe on ALFWorld (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E11: the
# training recipe on ALFWorld"; descriptive, outside every Holm table). One NEW GPU host launched by this driver from
# E3's clone image (tag Name=jev-e11-alfworld), SSM only. Harvest on ALFWorld TRAIN games (Flash LLM-only + the scripted
# executor), SFT of Qwen3-4B + LoRA (hvenv), temperatures, serve_jev on :8081, the calibration shadow arm and the tau
# fit on E6a's 48 calibration train games, smoke, then the evaluation on E6a's 140 valid_seen games. Gemini on the
# direct API (key from SSM Parameter Store into the stage's process only; never printed, never on disk).
#
#   LOCAL:  bash scripts/aws/e11_alfworld.sh launch [itype ...]   # one new host (refuses if a jev-e11-alfworld host exists);
#                                                                  # tries the types in order (default g6e.2xlarge
#                                                                  # g5.8xlarge g5.4xlarge); user-data arms the power-off guard
#           E11_IID=i-... bash scripts/aws/e11_alfworld.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages (in the order sent; a failed stage stops the chain, `final` always runs when sent):
#     rearm    idle stop (IDLE_MIN) + power-off guard (GUARD_END absolute UTC, else GUARD_MIN), host, GPU, disk
#     setup    the committed branch (bundle) in /opt/jev/e11; avenv (runner + alfworld 0.4.2 / textworld 1.7.0); hvenv
#              (training, serving) imports this checkout; E6a's data (valid_seen + the 48 calibration games); the ALFWorld
#              release archives (S3 copy, else GitHub), sha256-checked against E6a's record
#     select   the 600 harvest games (explore_e11_alfworld.py select) -> configs/explore/e11_alfworld/harvest_games.json
#     harvest  Flash LLM-only on the 360 Flash games (watchdog) and, in parallel, the scripted executor on the 600 games
#              (chunks of CHUNK games per process: TextWorld keeps a 32 MB library copy per game until the process exits)
#     build    SFT rows -> $A/data (train / dev files + build_manifest.json); data -> S3
#     train    SFT (SFT v1 recipe; nogc_3072, then gc_3072 / gc_2048 on out-of-memory only); adapter -> S3
#     temps    temperatures on the development rows; offline eval (trained + temperatures; base without adapter)
#     serve    serve_jev (adapter + temperatures) on :8081 as e11-alfworld-4b
#     cal      the calibration shadow arm (cal.yaml) under the watchdog
#     fit      tau-hat (explore_e11_alfworld.py fit) and set-tau (eval.yaml / smoke.yaml, the one pending line)
#     smoke    smoke.yaml + the registered projection gate
#     eval     eval.yaml at the n the gate allows, passes 2-3 re-run infrastructure errors only; evaluation window
#     final    stop the server, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e11_alfworld/box on the box -> s3://$BUCKET/e11_alfworld/box; runs -> .../runs/<run>;
# adapter -> s3://$BUCKET/checkpoints/e11_alfworld/ (no optimizer state).
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E11_IID=${E11_IID:-}
# every other project host: E3's, E5's, Stage 2's, E6a's CPU box, E7's (E7d / E7f): never touched by this driver
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SUBNET=<aws-resource-id>; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
TAG=jev-e11-alfworld
BUNDLE=e11.bundle
S3OUT="$B/e11_alfworld"
CAP_USD=${CAP_USD:-60}
GUARD_MIN=${GUARD_MIN:-600}          # 10 GPU-hours (registration (k)); GUARD_END (absolute UTC) wins when set
IDLE_MIN=${IDLE_MIN:-30}
WORKERS=${WORKERS:-8}
CHUNK=${CHUNK:-150}
POWEROFF=${POWEROFF:-1}
NAME=e11-alfworld-4b
PORT=8081

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E11_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E11_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E11_IID "* ]] && { echo "refused: $E11_IID is not the E11 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E11_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E11_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e11-alfworld: power-off guard (10 GPU-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g6e.2xlarge g5.8xlarge g5.4xlarge)
  for IT in "${types[@]}"; do
    out=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$IT" --subnet-id "$SUBNET" --security-group-ids "$SG" \
      --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior stop --count 1 \
      --user-data "file://$ud" --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
      --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name]' --output text 2>&1)
    rc=$?
    echo "$IT: $out"
    (( rc == 0 )) && { rm -f "$ud"; exit 0; }
    [[ "$out" == *InsufficientInstanceCapacity* || "$out" == *Unsupported* ]] || { rm -f "$ud"; exit 1; }
  done
  rm -f "$ud"; exit 1
fi
if [[ "$MODE" == push ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null || { echo "bundle failed"; exit 1; }
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors || exit 1
  git -C "$ROOT" show "$BRANCH:scripts/aws/e11_alfworld.sh" > "$tmp/e11_alfworld.sh"   # the committed driver
  awsl s3 cp "$tmp/e11_alfworld.sh" "$B/code/e11_alfworld.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e11_alfworld.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e11_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e11_alfworld.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q GUARD_MIN=%q GUARD_END=%q IDLE_MIN=%q WORKERS=%q CHUNK=%q POWEROFF=%q E11_IID=%q\n' \
      "$CAP_USD" "$GUARD_MIN" "${GUARD_END:-}" "$IDLE_MIN" "$WORKERS" "$CHUNK" "$POWEROFF" "$E11_IID"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e11_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e11_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E11_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e11_driver.sh /tmp/e11_driver.sh --only-show-errors && bash /tmp/e11_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E11_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E11_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,30p' "$0"
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
if [[ "$DRY_RUN" == 1 ]]; then MY_IID=${E11_IID:-i-dryrun}; else MY_IID=$(imds instance-id); fi
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E11_IID" && "$MY_IID" != "$E11_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E11 host ${E11_IID:-?}"; exit 1
fi
R=$W/e11
A=$R/results/explore/e11_alfworld/box
STATUS_DIR=$W/e11_status
HV=$W/hvenv
AV=$W/avenv
CK=$W/ck_e11
ZIPS=$W/alfworld_zips
STOP=runs/e11_STOP.json
SPEND=scripts/explore_e11_alfworld.py
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
  for d in e11_harvest_flash e11_harvest_scripted e11_alfworld_cal e11_alfworld_smoke e11_alfworld_eval; do
    [[ -d "$R/runs/$d" ]] && X aws s3 sync "$R/runs/$d" "$S3OUT/runs/$d" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  true
}
PYA() { PYTHONPATH="$R/src" "$AV/bin/python" "$@"; }    # runner, harvest, build, fit, readout (alfworld installed)
PYH() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }    # training, temperatures, offline eval, serving (torch)
base() { cat "$W/base_model_path.txt"; }
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
  watched -m jevharness.bench.runner "$cfg" --out runs "$@"
}
watched() {  # watched PYTHON-ARGS...: one avenv python process under the $ watchdog (its process group is stopped at the cap)
  if [[ "$DRY_RUN" == 1 ]]; then X PYA "$@"; return 0; fi
  setsid env PYTHONPATH="$R/src" "$AV/bin/python" "$@" >> "$A/runner.log" 2>&1 < /dev/null &
  local pid=$!
  PYA "$SPEND" watch --pid "$pid" --cap "$CAP_USD" --every 30 >> "$A/watch.log" 2>&1 &
  local wpid=$!
  ( while kill -0 "$pid" 2>/dev/null; do sleep 300; sync_out; done ) & local spid=$!
  wait "$pid"; local rc=$?
  kill "$wpid" "$spid" 2>/dev/null
  sync_out
  return $rc
}
errored() {  # errored RUN: tasks whose latest record (any arm) ended in an infrastructure error
  PYA - "$1" <<'PYEOF'
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
  if [[ -f "$W/e11_serve.pid" ]]; then
    while read -r p; do [[ -n "$p" ]] && kill "$p" 2>/dev/null; done < "$W/e11_serve.pid"
  fi
  rm -f "$W/e11_serve.pid"; [[ "$DRY_RUN" == 1 ]] || sleep 5
}
server_up() { [[ "$DRY_RUN" == 1 ]] || curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; }

stage_rearm() {
  if [[ -n "${GUARD_END:-}" ]]; then   # an absolute end (UTC): the GPU cap counts from launch
    GUARD_MIN=$(( ( $(date -u -d "$GUARD_END" +%s) - $(date -u +%s) ) / 60 ))
  fi
  (( GUARD_MIN > 0 )) || { echo "guard end already passed"; return 1; }
  X sudo shutdown -c 2>/dev/null; X sudo shutdown -h +"$GUARD_MIN" "jev-e11-alfworld: power-off guard (GPU cap)"
  if [[ "$DRY_RUN" != 1 ]]; then
    sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E11 chain / server / runner / trainer process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e11_[0-9]|jevharness|serve_jev|explore_e11|train_decision' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e11-alfworld: idle $IDLE_MIN min"; fi
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
  local myb="$W/e11.$$.bundle"
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
  grep -q 'E11: the training recipe on ALFWorld' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E11"; return 1; }
  command -v uv > /dev/null || X bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  [[ -x "$AV/bin/python" ]] || X uv venv -q -p 3.11 "$AV" || return 1
  X uv pip install -q --python "$AV" -e ".[all]" "alfworld==0.4.2" "textworld==1.7.0" || return 1
  if [[ "$DRY_RUN" != 1 ]]; then
    PYA -c "import jevharness, alfworld, textworld, sys; p = jevharness.__file__; print(p); sys.exit(0 if p.startswith('$R/src') else 1)" \
      || { echo "the avenv does not import this checkout"; return 1; }
    PYH -c "import jevharness, torch, transformers, peft, sys; p = jevharness.__file__; print(p, torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
      > "$A/env.txt" 2>&1 || { cat "$A/env.txt"; echo "the hvenv does not import this checkout"; return 1; }
    cat "$A/env.txt"
    [[ -s "$W/base_model_path.txt" ]] || { echo "no base_model_path.txt"; return 1; }
    cp -f "$W/base_model_path.txt" "$A/base_model_path.txt"
  fi
  # E6a's data (valid_seen + the 48 calibration train games)
  if [[ ! -d data/raw/alfworld/json_2.1.1/valid_seen || ! -d data/raw/alfworld_e6a ]]; then
    X bash -c "aws s3 cp '$B/e6a/data.tgz' - --region '$REGION' | tar xz -C '$R'" || { echo "E6a data download failed"; return 1; }
  fi
  if [[ "$DRY_RUN" != 1 ]]; then
    PYA - <<'PYEOF' || return 1
import json, sys
sys.path.insert(0, "scripts")
import explore_e6a_alfworld as E6A
from jevharness.bench.suites.alfworld import collect_game_files
ids = E6A.eval_ids()
ok = ids == json.loads(E6A.EVAL_LIST.read_text())["task_ids"]
n_cal = len(collect_game_files("train", data_dir="data/raw/alfworld_e6a"))
print(f"valid_seen list {'matches' if ok else 'MISMATCH'} ({len(ids)}); calibration games {n_cal}")
sys.exit(0 if ok and n_cal == 48 else 1)
PYEOF
  fi
  # the ALFWorld release archives (E6a's record: URLs and sha256): S3 copy, else GitHub (then copied to S3)
  mkdir -p "$ZIPS"
  local k url sha f
  for k in json twpddl; do
    f="$ZIPS/$k.zip"
    url=$(PYA -c "import json; print(json.load(open('configs/explore/e6a_alfworld/train_games.json'))['archives']['$k']['url'])")
    sha=$(PYA -c "import json; print(json.load(open('configs/explore/e6a_alfworld/train_games.json'))['archives']['$k']['sha256'])")
    if [[ "$DRY_RUN" == 1 ]]; then X fetch "$url" "$f"; continue; fi
    [[ -s "$f" && "$(sha256sum "$f" | cut -d' ' -f1)" == "$sha" ]] && continue
    aws s3 cp "$S3OUT/archives/$k.zip" "$f" --only-show-errors --region "$REGION" 2>/dev/null \
      || curl -sSL --retry 3 -o "$f" "$url" || { echo "download of $k failed"; return 1; }
    [[ "$(sha256sum "$f" | cut -d' ' -f1)" == "$sha" ]] || { echo "$k archive sha256 differs from E6a's record"; return 1; }
    aws s3 cp "$f" "$S3OUT/archives/$k.zip" --only-show-errors --region "$REGION" || true
  done
  df -BG / | awk 'NR==2{print "disk free: "$4}'
}
stage_select() {
  cd "$R" || return 1
  X PYA "$SPEND" select --json-zip "$ZIPS/json.zip" --twpddl-zip "$ZIPS/twpddl.zip" || return 1
  [[ "$DRY_RUN" == 1 ]] || cp -f configs/explore/e11_alfworld/harvest_games.json "$A/harvest_games.json"
}
stage_harvest() {  # Flash (watched, background) and the scripted executor (chunks of CHUNK games per process)
  cd "$R" || return 1
  keys || return 1
  [[ -s configs/explore/e11_alfworld/harvest_games.json || "$DRY_RUN" == 1 ]] || cp -f "$A/harvest_games.json" configs/explore/e11_alfworld/ \
    || { echo "no harvest_games.json"; return 1; }
  watched "$SPEND" harvest-flash --out runs & local fpid=$!   # the harvest-label patch of the expert (addendum)
  local n rc=0
  for n in $(seq "$CHUNK" "$CHUNK" 600) 600; do
    X PYA "$SPEND" harvest-scripted --out runs --workers "$WORKERS" --max-tasks "$n" >> "$A/harvest_scripted.log" 2>&1 \
      || { rc=1; break; }
  done
  wait "$fpid"; local frc=$?
  st "harvest: scripted exit=$rc, flash exit=$frc; $(PYA "$SPEND" spend --short 2>/dev/null)"
  leak_check || return 1
  [[ -f "$STOP" ]] && return 1
  (( rc == 0 && frc == 0 )) || return 1
  if [[ "$DRY_RUN" != 1 ]]; then
    ne=$(errored runs/e11_harvest_flash)
    if [[ "$ne" != 0 ]]; then
      st "$ne errored Flash harvest episode(s); one re-run pass"
      watched "$SPEND" harvest-flash --out runs
    fi
  fi
  sync_out
}
stage_build() {
  cd "$R" || return 1
  X PYA "$SPEND" build --runs runs --out "$A/data" > "$A/build.log" 2>&1 || { tail -20 "$A/build.log"; return 1; }
  cat "$A/build.log"
  X aws s3 sync "$A/data" "$S3OUT/box/data" --only-show-errors --region "$REGION"
}
stage_train() {  # SFT v1 recipe on the train rows: fresh LoRA r 32 / alpha 64, soft CE + 0.5 Brier, permutation,
  # lr 1e-4, 2 epochs, 64 rows per step, max_len 4096, seed 0; memory setting nogc_3072, then gc_3072 / gc_2048 on OOM
  cd "$R" || return 1
  stop_server
  local setting flags rc=1 t0 t1
  for setting in nogc_3072 gc_3072 gc_2048; do
    flags=$(PYA scripts/train/ablation_box.py setting "$setting") || return 1
    X rm -rf "$CK"; mkdir -p "$CK" "$A/logs"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    X env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.train_decision --model "$(base)" \
      --train "$A/data/train.jsonl" --out "$CK" --loss soft_ce_brier --brier-weight 0.5 --lora-r 32 --lora-alpha 64 \
      --lr 1e-4 --epochs 2 --max-len 4096 --seed 0 --report-to none $flags > "$A/logs/train_$setting.log" 2>&1
    rc=$?
    t1=$(date +%s)
    printf '{"setting": "%s", "exit": %s, "minutes": %s}\n' "$setting" "$rc" "$(( (t1 - t0) / 60 ))" >> "$A/train_runs.jsonl"
    (( rc == 0 )) && break
    grep -qE 'out of memory|OutOfMemoryError' "$A/logs/train_$setting.log" || return 1
  done
  (( rc == 0 )) || return 1
  [[ "$DRY_RUN" == 1 ]] && return 0
  sha256sum "$CK"/adapter_model.safetensors "$CK"/adapter_config.json > "$A/adapter_sha256.txt"
  PYA - "$A" <<'PYEOF'
import json, sys
from pathlib import Path
a = Path(sys.argv[1])
runs = [json.loads(x) for x in (a / "train_runs.jsonl").read_text().splitlines() if x.strip()]
ok = [r for r in runs if r["exit"] == 0][-1]
rows = sum(1 for _ in open(a / "data" / "train.jsonl"))
rec = {"base_model": (a / "base_model_path.txt").read_text().strip(), "rows": rows, "epochs": 2, "lr": 1e-4,
       "lora": {"r": 32, "alpha": 64, "target": "all-linear"}, "loss": "soft_ce_brier", "brier_weight": 0.5,
       "rows_per_step": 64, "max_len": 4096, "seed": 0, "memory_setting": ok["setting"], "minutes": ok["minutes"],
       "attempts": runs, "adapter_sha256": (a / "adapter_sha256.txt").read_text().split()[0::2]}
rec["summary"] = (f"Qwen3-4B + LoRA (r 32), {rows} rows x 2 epochs, soft CE + 0.5 Brier, lr 1e-4, "
                  f"{ok['minutes']} min ({ok['setting']})")
(a / "train_record.json").write_text(json.dumps(rec, indent=1) + "\n")
print(rec["summary"])
PYEOF
  aws s3 sync "$CK" "$B/checkpoints/e11_alfworld/" --only-show-errors --region "$REGION" --exclude 'checkpoint-*' \
    --exclude '*optimizer*' --exclude '*.pt' --exclude 'rng_state*'
}
stage_temps() {  # temperatures on the development rows; offline eval (trained + temperatures; base, no adapter)
  cd "$R" || return 1
  stop_server
  mkdir -p "$A/cal" "$A/eval" "$A/logs"
  local dev=("$A/data/dev.jsonl" "$A/data/dev_off_plan.jsonl" "$A/data/dev_flash.jsonl")
  X PYH -m jevharness.training.calibrate --model "$(base)" --adapter "$CK" --dev "$A/data/dev.jsonl" \
    --out "$A/cal/calibration.json" --min-bucket 200 --batch-size 8 > "$A/logs/calibrate.log" 2>&1 || return 1
  X cp -f "$A/cal/calibration.json" "$CK/calibration.json"
  X PYH -m jevharness.training.eval_decisions --model "$(base)" --adapter "$CK" --calibration "$A/cal/calibration.json" \
    --dev "${dev[@]}" --n-perm 0 --max-len 4096 --batch-size 8 --out "$A/eval/dev_trained.json" > "$A/logs/eval_trained.log" 2>&1 || return 1
  X PYH -m jevharness.training.eval_decisions --model "$(base)" \
    --dev "${dev[@]}" --n-perm 0 --max-len 4096 --batch-size 8 --out "$A/eval/dev_base.json" > "$A/logs/eval_base.log" 2>&1 || return 1
  [[ "$DRY_RUN" == 1 ]] || sha256sum "$A/cal/calibration.json" >> "$A/adapter_sha256.txt"
  X aws s3 cp "$A/cal/calibration.json" "$B/checkpoints/e11_alfworld/calibration.json" --only-show-errors --region "$REGION"
}
stage_serve() {
  cd "$R" || return 1
  stop_server
  [[ "$DRY_RUN" == 1 ]] && { X serve_jev "$NAME"; return 0; }
  [[ -s "$CK/adapter_model.safetensors" && -s "$A/cal/calibration.json" ]] || { echo "no adapter / temperatures"; return 1; }
  sha256sum -c --quiet <(head -2 "$A/adapter_sha256.txt") || { echo "adapter sha256 mismatch"; return 1; }
  local args=(--model "$(base)" --name "$NAME" --port "$PORT" --adapter "$CK" --calibration "$A/cal/calibration.json")
  echo "serve_jev ${args[*]}" > "$A/serve.cmd"
  nohup env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$A/serve.out" 2>&1 &
  echo $! > "$W/e11_serve.pid"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" > "$A/serve_health.json"; then
      curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/serve_models.json"
      grep -q "\"$NAME\"" "$A/serve_models.json" || { echo "served model id is not $NAME"; return 1; }
      return 0
    fi
    kill -0 "$(cat "$W/e11_serve.pid")" 2>/dev/null || { echo "serve_jev died"; tail -20 "$A/serve.out"; return 1; }
    sleep 5
  done
  echo "serve_jev did not come up"; return 1
}
stage_cal() {
  cd "$R" || return 1
  keys || return 1
  server_up || { echo "server not up"; return 1; }
  run_watched configs/explore/e11_alfworld/cal.yaml
  local rc=$?
  st "calibration exit=$rc; $(PYA "$SPEND" spend --short 2>/dev/null)"
  leak_check || return 1
  [[ -f "$STOP" ]] && return 1
  return $rc
}
stage_fit() {
  cd "$R" || return 1
  X PYA "$SPEND" fit > "$A/fit.log" 2>&1 || { cat "$A/fit.log"; return 1; }
  X PYA "$SPEND" set-tau --record "$A/set_tau.json" >> "$A/fit.log" 2>&1 || { cat "$A/fit.log"; return 1; }
  [[ "$DRY_RUN" == 1 ]] || cp -f configs/explore/e11_alfworld/tauhat_action.json configs/explore/e11_alfworld/eval.yaml \
    configs/explore/e11_alfworld/smoke.yaml "$A/"
  st "fit: $(tail -c 400 "$A/fit.log" | tr '\n' ' ' | cut -c1-300)"
}
stage_smoke() {
  cd "$R" || return 1
  keys || return 1
  server_up || { echo "server not up"; return 1; }
  grep -q "calibration: pending" configs/explore/e11_alfworld/smoke.yaml && { echo "smoke.yaml still pending (run fit)"; return 1; }
  run_watched configs/explore/e11_alfworld/smoke.yaml
  local rc=$?
  X PYA "$SPEND" project --out "$A/projection.json" > "$A/projection.out" 2>&1
  st "smoke exit=$rc; projection: $(PYA -c "import json; print(json.load(open('$A/projection.json'))['decision'])" 2>/dev/null)"
  leak_check || return 1
  return $rc
}
stage_eval() {
  cd "$R" || return 1
  keys || return 1
  server_up || { echo "server not up"; return 1; }
  grep -q "calibration: pending" configs/explore/e11_alfworld/eval.yaml && { echo "eval.yaml still pending (run fit)"; return 1; }
  local dec ov=() pass rc t0 t1 ne
  dec=$(PYA -c "import json; print(json.load(open('$A/projection.json'))['decision'])" 2>/dev/null)
  [[ "$DRY_RUN" == 1 ]] && dec="run all 140"
  case "$dec" in
    "run all "[0-9]*) ov=() ;;
    "run the first "[0-9]*) ov=(--max-tasks "${dec##* }") ;;
    *) st "refusing eval: projection says '$dec'"; return 3 ;;
  esac
  printf 'step e11_alfworld_eval\nstarted %s\ncommit %s\nconfig eval.yaml %s\nprereg sha256 %s\ncap_usd %s\nworkers %s\nhost %s %s\n' \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$(sha256sum configs/explore/e11_alfworld/eval.yaml | cut -c1-16)" \
    "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" "$CAP_USD" "$WORKERS" "$MY_IID" "$(imds instance-type)" > "$A/START_eval.txt"
  t0=$(date -u +%FT%TZ)
  for pass in 1 2 3; do
    [[ -f "$STOP" ]] && { st "watchdog stop recorded; no further pass"; break; }
    if (( pass > 1 )); then
      ne=$(errored runs/e11_alfworld_eval)
      [[ "$ne" == 0 ]] && { st "no errored episode after pass $((pass - 1)); no further pass"; break; }
      st "$ne errored episode(s) after pass $((pass - 1)); pass $pass"
      [[ "$DRY_RUN" == 1 ]] || sleep 120
    fi
    st "eval pass $pass ($dec)"
    run_watched configs/explore/e11_alfworld/eval.yaml "${ov[@]}"
    rc=$?
    st "eval pass $pass exit=$rc; $(PYA "$SPEND" spend --short 2>/dev/null)"
    [[ "$DRY_RUN" == 1 ]] && break
  done
  t1=$(date -u +%FT%TZ)
  printf '{"window_start": "%s", "window_end": "%s", "instance_type": "%s", "instance_id": "%s", "workers": %s, "run": "runs/e11_alfworld_eval"}\n' \
    "$t0" "$t1" "$(imds instance-type)" "$MY_IID" "$WORKERS" > "$A/eval_window.json"
  leak_check || return 1
  [[ -f "$STOP" ]] && return 1
  return 0
}
stage_final() {
  stop_server
  [[ "$DRY_RUN" == 1 ]] || nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; X sudo shutdown -h +2 "jev-e11-alfworld: done"; fi
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
