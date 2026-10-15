#!/usr/bin/env bash
# Exploratory row E5: small decision models (docs/PREREG_A7_STAGE2.md, Deviations entry of 2026-10-06 "Exploratory row
# E5"; plan results/explore/small_models/e5_plan.md). Built from E3's driver (scripts/aws/train_ablation.sh): the SFT v1
# recipe on E3's D100 file for Qwen/Qwen3-1.7B, Qwen/Qwen3-0.6B and openbmb/MiniCPM5-1B (pinned revisions in
# scripts/train/e5_box.py MODELS; ibm-granite/granite-4.0-1b replaces MiniCPM5-1B only if its smoke fails), the offline
# evaluation (v1 dev, temperatures, corpus C through serve_jev --fast, the calibration-state replay of entry (g)), and
# the latency bench of every model and the frozen JevAgentRL-4B in one session on one GPU. Offline stages: no LLM call
# and no key (the preamble unsets every *_API_KEY(S); any priced record stops the chain). The live stages (entry (g))
# load the System-2 key from SSM inside their own subshell only.
#
# Host: a NEW instance launched from an image of the E3 host, given by id: E5_IID (no default). The driver refuses
# the E3 host and the other project hosts, and never writes outside s3://$BUCKET/e5_small/, .../checkpoints/e5_small/
# and .../code/e5_small*.
#
#   LOCAL:  E5_IID=i-... bash scripts/aws/e5_small.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages, in the order sent (a failed stage stops the chain; e5_final / live_final always run when sent):
#     rearm         idle stop (IDLE_MIN) + power-off guard (GUARD_MIN), instance id / type, GPU, versions, host.json
#     e5_inputs     D100 rebuilt and sha256-checked, corpus C copied, the calibration-replay requests from S3 (sha256),
#                   the frozen 4B verified, every model downloaded at its pinned revision into the HF cache
#     e5_smoke      per model: the read-out check (tokenizer) and a 4-step memory smoke (E3's fallback order); a failed
#                   MiniCPM5-1B smoke swaps in the backup (swap.json)
#     e5_arms       every model in order (or e5_arm:<slot>): cap check, SFT, eval_decisions, temperatures, serve_jev
#                   --fast + C replay (three invocations) + calibration-state replay, adapter -> S3, scored
#     e5_latency    the frozen 4B, then every finished model, each served alone: harness_final.py bench c = 1 (25
#                   episodes) and c = 4 (100), measure_latency.py c = 1, 4; nvidia-smi sampled; the 4B again at c = 1
#     e5_cal4b      the frozen 4B's calibration-state replay (the check of entry (g))
#     e5_final      cost check of every record, scoring, final sync, power-off (POWEROFF=1)
#     live_serve    the frozen 4B on :8081 and the chosen E5 model on :8082 (both serve_jev --fast, one GPU)
#     live_smoke    5 development episodes (seeds 900-904) of the three live arms, one worker
#     live          configs/experiments/e5_small_live.yaml, one worker, three passes, $ watchdog (LIVE_CAP_USD)
#     live_final    stop the servers, sync, power-off (POWEROFF=1)
#   Send-time env: CAP_H [14], GUARD_MIN [(CAP_H+3)*60], IDLE_MIN [60], POWEROFF [1], WAIT_PREV [1], LIVE_CAP_USD [60],
#   E5_PART [''] (a further host's part of the row: records under s3://$BUCKET/e5_small/part_<E5_PART>/; e.g.
#   E5_PART=b E5_IID=i-... send "rearm e5_inputs e5_smoke:e5_q06 e5_arm:e5_q06 e5_final"; adapters as always under
#   checkpoints/e5_small/<slot>; the latency session stays on the main host).
#   DRY_RUN=1 (box side, run locally: W=<dir> DRY_RUN=1 bash scripts/aws/e5_small.sh box "STAGES"): every heavy
#   command (GPU, S3, sudo, downloads, the data build) goes to $A/dry_run.log instead of running; light checks run.
# Records: runs/e5_small on the box -> s3://$BUCKET/e5_small/ (no data files, no corpus copies, no caches);
# adapters -> s3://$BUCKET/checkpoints/e5_small/<slot>/ (no optimizer state); live runs -> s3://$BUCKET/e5_small/runs/.
set -uo pipefail
SELF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"   # the repo of this file (DRY_RUN only)
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E5_IID=${E5_IID:-}
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}   # E3's host and the other project hosts
BUNDLE=e5_small.bundle
E5_PART=${E5_PART:-}                         # a second / third host's part of the row (Deviations 2026-10-06, hosts)
[[ -z "$E5_PART" || "$E5_PART" =~ ^[a-z0-9]+$ ]] || { echo "E5_PART must be [a-z0-9]+"; exit 1; }
S3OUT="$B/e5_small${E5_PART:+/part_$E5_PART}"
S3IN="$B/e5_small/inputs"                     # the row's inputs (one copy for every host)
S3CK="$B/checkpoints/e5_small"
S3_FROZEN="$B/checkpoints/round5/jevagentrl-4b"                                # read only (fetched if absent)
CAP_H=${CAP_H:-14}
GUARD_MIN=${GUARD_MIN:-$(( (${CAP_H%.*} + 3) * 60 ))}
IDLE_MIN=${IDLE_MIN:-60}
POWEROFF=${POWEROFF:-1}
WAIT_PREV=${WAIT_PREV:-1}
LIVE_CAP_USD=${LIVE_CAP_USD:-60}

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws --region "$REGION" "$@"; }
iid_ok() {  # the coordinator's instance id, never one of the forbidden hosts
  [[ "$E5_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E5_IID is not set to an instance id (given by the coordinator)"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E5_IID "* ]] && { echo "refused: $E5_IID is not the E5 host"; return 1; }
  return 0
}

if [[ "$MODE" == push ]]; then
  ROOT=$SELF_ROOT
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null || { echo "bundle failed"; exit 1; }
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors || exit 1
  git -C "$ROOT" show "$BRANCH:scripts/aws/e5_small.sh" > "$tmp/e5_small.sh"   # the committed driver
  awsl s3 cp "$tmp/e5_small.sh" "$B/code/e5_small.sh" --only-show-errors || exit 1
  awsl s3 ls "$S3IN/cal_requests.jsonl.gz" > /dev/null 2>&1 \
    || echo "WARNING: $S3IN/cal_requests.jsonl.gz is missing: e5_inputs will fail"
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e5_small.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e5_$(date +%s%N).sh'      # a private copy per invocation (bash reads a running script lazily)
    echo "aws s3 cp $B/code/e5_small.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_H=%q GUARD_MIN=%q IDLE_MIN=%q POWEROFF=%q WAIT_PREV=%q LIVE_CAP_USD=%q E5_IID=%q E5_PART=%q\n' \
      "$CAP_H" "$GUARD_MIN" "$IDLE_MIN" "$POWEROFF" "$WAIT_PREV" "$LIVE_CAP_USD" "$E5_IID" "$E5_PART"
    for v in LIVE_CFG LIVE_RUN SMOKE_CFG SMOKE_RUN LIVE_KEYS; do   # live-row overrides (the route diagnostic)
      [[ -n "${!v:-}" ]] && printf 'export %s=%q\n' "$v" "${!v}"
    done
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e5_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  drv=e5_driver${E5_PART:+_$E5_PART}.sh
  awsl s3 cp "$tmp" "$B/code/$drv" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E5_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/$drv /tmp/e5_driver.sh --only-show-errors && bash /tmp/e5_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E5_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E5_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,39p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
DRY_RUN=${DRY_RUN:-0}
W=${W:-/opt/jev}
mkdir -p "$W" && cd "$W" || exit 1
my_ts=$(basename "$0" .sh); my_ts=${my_ts#e5_}
others_running() {  # another driver's chain, or an older E5 chain, is still running (never touch its tree or GPU)
  local t
  ps -eo args | grep -E '/opt/jev/(hf|ex|g2|s2|sl|r5|r6|ta)_[0-9]+\.sh box' | grep -v -e grep -e "$0" | grep -q . && return 0
  for t in $(ps -eo args | grep -oE '/opt/jev/e5_[0-9]+\.sh box' | grep -oE '[0-9]{6,}' | sort -u); do
    [[ "$t" < "$my_ts" ]] && return 0
  done
  return 1
}
waited=0
if [[ "$WAIT_PREV" == 1 && "$DRY_RUN" != 1 ]]; then
  while others_running; do sleep 30; waited=$((waited + 30)); done
fi
imds() {  # imds PATH: instance metadata (IMDSv2)
  curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" "http://169.254.169.254/latest/meta-data/$1"
}
if [[ "$DRY_RUN" == 1 ]]; then
  CHECKOUT=${CHECKOUT:-$SELF_ROOT}
  [[ -n "$CHECKOUT" && -f "$CHECKOUT/scripts/aws/e5_small.sh" ]] && cd "$CHECKOUT" || { echo "no checkout"; exit 1; }
  MY_IID=${DRY_IID:-i-0e5dry00000000000}
else
  MY_IID=$(imds instance-id)
  # the image is a clone of E3's host: refuse to run on any host but the E5 one (E3's tree and S3 prefix stay untouched)
  if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E5_IID" && "$MY_IID" != "$E5_IID" ) ]]; then
    echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E5 host ${E5_IID:-?}"; exit 1
  fi
  myb="$W/e5.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || exit 1
  [[ -d "$W/stage2" ]] || git clone -q -b "$BRANCH" "$myb" "$W/stage2"
  cd "$W/stage2" || exit 1
  # the checkout may be a stale copy of E3's: reset to the committed E5 branch (E5's records live in the untracked
  # runs/e5_small and survive the reset)
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || exit 1
  rm -f "$myb"
fi
A=${E5_RUN:-runs/e5_small}
mkdir -p "$A"/{data,s23,eval,cal,logs,settings,done,mem,smoke,models,inputs,latency/bench} cache
HV=$W/hvenv
if [[ "$DRY_RUN" == 1 ]]; then PY=${PY:-python3}; else PY=$HV/bin/python; set +u; . "$HV/bin/activate"; set -u; fi
CKE=$W/ck_e5                                   # E5 adapters (one directory per model slot)
CK=$W/ck                                       # the frozen JevAgentRL-4B (as scripts/aws/stage2_gpu.sh fetched it)
PLAN=results/stage2/rl_eval_plan.json
PORT=8082
PORT4B=8081
LIVE_RUN=${LIVE_RUN:-runs/e5_small_live}      # the route diagnostic (Deviations ~19:40 UTC): runs/e5_small_live_direct
LIVE_CFG=${LIVE_CFG:-configs/experiments/e5_small_live.yaml}
LIVE_CACHE_TOKENS=${LIVE_CACHE_TOKENS:-16384}   # per server, live row (two models on one 24 GB GPU)
SMOKE_CFG=${SMOKE_CFG:-configs/dev/e5_small_smoke.yaml}
SMOKE_RUN=${SMOKE_RUN:-runs/e5_small_smoke}
GPU_WAIT_MIN=${GPU_WAIT_MIN:-20}
V1_FILES=(data/train_v1/dev.jsonl.gz data/train_v1/dev_recovery.jsonl.gz data/train_v1/dev_knockout.jsonl.gz)
CALREQ=$A/inputs/cal_requests.jsonl.gz
BASE4B=$(cat "$W/base_model_path.txt" 2>/dev/null || echo "${DRY_BASE:-/BASE_MODEL}")
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=0

dry() { printf 'DRY:'; printf ' %q' "$@"; printf '\n'; }
X() {  # a heavy command (GPU, S3, sudo, downloads, long CPU): run it, or under DRY_RUN=1 only record it
  if [[ "$DRY_RUN" == 1 ]]; then dry "$@" | tee -a "$A/dry_run.log"; return 0; fi
  "$@"
}
dry_skip() {  # dry_skip NAME: true (skip) under DRY_RUN, where the inputs of a check were never produced
  [[ "$DRY_RUN" == 1 ]] || return 1
  echo "DRY: skipped $1" | tee -a "$A/dry_run.log"
}
S3() { X aws "$@" --only-show-errors --region "$REGION"; }
box() { "$PY" scripts/train/e5_box.py "$@"; }
sync_out() {
  S3 s3 sync "$A" "$S3OUT" --exclude 'data/*' --exclude 'smoke/*.jsonl' --exclude 's23/occurrences.jsonl.gz' \
    --exclude 's23/questions.jsonl.gz' --exclude 'inputs/*' --exclude '*.sqlite' --exclude 'dry_run.log'
}
sync_run() { [[ -d "$1" ]] && S3 s3 sync "$1" "$S3OUT/$1" --exclude '*.sqlite'; }
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$A/status.txt"
  [[ "$DRY_RUN" == 1 ]] || sync_out > /dev/null 2>&1
}
sync_ck() {  # sync_ck SLOT: adapter, adapter config, calibration, tokenizer (no optimizer state)
  S3 s3 sync "$CKE/$1" "$S3CK/$1" --exclude 'checkpoint-*' --exclude '*optimizer*' --exclude '*.pt' \
    --exclude 'rng_state*' --exclude 'train_mix.jsonl'
}
itype() { if [[ "$DRY_RUN" == 1 ]]; then echo "${DRY_ITYPE:-g5.8xlarge}"; else imds instance-type; fi; }
gpu_name() {
  if [[ "$DRY_RUN" == 1 ]]; then echo "${DRY_GPU:-NVIDIA A10G}"; return; fi
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
}
gpu_free() {  # no other process on the GPU (never killed): wait up to GPU_WAIT_MIN minutes
  [[ "$DRY_RUN" == 1 ]] && return 0
  local i apps=""
  for i in $(seq 1 $(( GPU_WAIT_MIN * 2 ))); do
    apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
    [[ -z "$apps" ]] && return 0
    sleep 30
  done
  echo "GPU still busy after $GPU_WAIT_MIN min: $apps"
  return 1
}
smi_start() {  # smi_start FILE: sample timestamp, utilisation, memory.used every 500 ms (pid in $SMI)
  SMI=""
  [[ "$DRY_RUN" == 1 ]] && return 0
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 > "$1" 2>/dev/null &
  SMI=$!
}
smi_stop() { [[ -n "${SMI:-}" ]] && kill "$SMI" 2>/dev/null; SMI=""; }
arm_stops() {  # power-off guard + idle stop (cron, IDLE_MIN); replaces the cloned image's E3 idle stop
  X sudo shutdown -c 2>/dev/null
  X sudo shutdown -h +"$GUARD_MIN" "jev-e5-small: total guard" 2>/dev/null
  [[ "$DRY_RUN" == 1 ]] && { dry install /usr/local/bin/jev-idle-stop "IDLE_MIN=$IDLE_MIN" >> "$A/dry_run.log"; return 0; }
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E5 chain / trainer / server / runner process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e5_[0-9]|jevharness|jevbench|train_decision|serve_jev|stage2_corpus|e5_cal_replay|harness_final|measure_latency|snapshot_download|hf download|uv pip' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e5-small: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
}
registered() {  # the Deviations entry of E5 is in the checkout's registration before any training step (smoke included)
  [[ "$DRY_RUN" == 1 ]] && return 0
  grep -q 'Exploratory row E5' docs/PREREG_A7_STAGE2.md && return 0
  st "no Deviations entry 'Exploratory row E5' in docs/PREREG_A7_STAGE2.md: no training"
  return 1
}
score_box() {  # the run scored as it stands (descriptive; never used to choose anything on the box)
  "$PY" scripts/e5_analysis.py --run "$A" --out "$A/score" --no-tex > "$A/logs/score.log" 2>&1 || true
}
model_path() {  # model_path SLOT: the local snapshot of the slot's pinned revision
  if [[ "$DRY_RUN" == 1 ]]; then echo "$W/hf/$1"; return; fi
  "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['path'])" "$A/models/$1.json" 2>/dev/null
}
slots() { box models --run "$A"; }

# ---------------------------------------------------------------------------------------------- serving
stop_port() {  # stop_port PORT: the server this driver started on PORT
  local f="$W/e5_serve_$1.pid" p
  if [[ -f "$f" ]]; then
    p=$(cat "$f")
    kill "$p" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -9 "$p" 2>/dev/null
  fi
  rm -f "$f"
}
stop_all() { stop_port "$PORT"; stop_port "$PORT4B"; }
serve() {  # serve PORT NAME BASE ADAPTER CALIBRATION TAG [serve_jev flags...]: serve_jev --fast on PORT
  local port=$1 name=$2 base=$3 adapter=$4 cal=$5 tag=$6
  shift 6
  local args=(--model "$base" --adapter "$adapter" --calibration "$cal" --name "$name" --port "$port" --fast "$@")
  stop_port "$port"
  echo "serve_jev ${args[*]}" >> "$A/logs/serve.cmd"
  if [[ "$DRY_RUN" == 1 ]]; then dry "$PY" -m jevharness.training.serve_jev "${args[@]}" | tee -a "$A/dry_run.log"; return 0; fi
  if curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; then echo "port $port is served by another process"; return 1; fi
  nohup "$PY" -m jevharness.training.serve_jev "${args[@]}" > "$A/logs/serve_$tag.out" 2>&1 &
  echo $! > "$W/e5_serve_$port.pid"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$port/health" > "$A/logs/serve_${tag}_health.json"; then
      curl -sf "http://127.0.0.1:$port/v1/models" > "$A/logs/serve_${tag}_models.json"
      return 0
    fi
    kill -0 "$(cat "$W/e5_serve_$port.pid")" 2>/dev/null || { echo "serve_jev died"; tail -20 "$A/logs/serve_$tag.out"; return 1; }
    sleep 5
  done
  echo "serve_jev did not come up"
  return 1
}
plan() { "$PY" -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"; }
verify_4b() {  # the frozen row of the plan (sha256 list of results/stage2/rl_eval_plan.json); fetched from S3 if absent
  dry_skip "frozen 4B sha256" && return 0
  local out=$A/frozen_verify.json
  "$PY" - "$PLAN" "$CK" "$out" <<'EOF'
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
}
serve_4b() {  # serve_4b TAG [serve_jev flags...]: the frozen JevAgentRL-4B on :8081 (sha256-checked), its own temperatures
  local adapter cal name tag=$1
  shift
  [[ "$DRY_RUN" == 1 || -s "$W/base_model_path.txt" ]] || { echo "no base_model_path.txt"; return 1; }
  verify_4b || return 1
  adapter=$(plan "d['rows']['frozen'].get('adapter') or ''")
  cal=$(plan "d['rows']['frozen'].get('calibration') or ''")
  name=$(plan "d['rows']['frozen']['served_name']")
  [[ "$name" == jevagentrl-4b ]] || { echo "served name $name != jevagentrl-4b"; return 1; }
  serve "$PORT4B" "$name" "$BASE4B" "$CK/$adapter" "$CK/$cal" "$tag" "$@"
}
serve_e5() {  # serve_e5 SLOT TAG [serve_jev flags...]: the slot's merged adapter + its v1-dev temperatures on :8082
  local slot=$1 tag=$2
  shift 2
  serve "$PORT" "e5-$slot" "$(model_path "$slot")" "$CKE/$slot" "$A/cal/${slot}_calibration.json" "$tag" "$@"
}
spec_of() {  # spec_of NAME PORT: the jev backend spec of a served model
  printf '{"type": "jev", "name": "%s", "base_url": "http://127.0.0.1:%s", "model": "%s", "api_key": "", "timeout": 60.0}' \
    "$1" "$2" "$1"
}
cal_replay() {  # cal_replay NAME PORT OUT: the calibration-state questions of entry (g) through the served model
  X "$PY" scripts/e5_cal_replay.py answer --requests "$CALREQ" --url "http://127.0.0.1:$2" --name "$1" --out "$3"
}

# ---------------------------------------------------------------------------------------------- training + evaluation
oom() { [[ -f "$1" ]] && box oom "$1"; }
sft_run() {  # sft_run SLOT OUT_DIR TRAIN SETTING LOG [MAX_STEPS]: 10_sft.sh with the SFT v1 recipe, BASE = the slot's model
  local slot=$1 out=$2 train=$3 setting=$4 log=$5 steps=${6:--1} flags base
  flags=$(box setting "$setting") || return 1
  base=$(model_path "$slot")
  [[ -n "$base" ]] || { echo "no local snapshot for $slot"; return 1; }
  [[ "$DRY_RUN" == 1 ]] || rm -rf "$out"
  local envs=(GPUS=1 RECIPE=r3 MODEL="$base" TRAIN="$train" DEV=data/train_v0/dev.jsonl
              REPLAY=data/train_v1/replay_when2call.jsonl.gz REPLAY_FACTOR=0.45 OUT="$out" LR=1e-4 EPOCHS=2 BATCH=2
              EFF_BATCH=64 MAX_LEN=4096 SEED=0 MAX_STEPS="$steps" PY="$PY" SFT_EXTRA="$flags")
  if [[ "$DRY_RUN" == 1 ]]; then   # 10_sft.sh's own dry run prints its commands (data mix, trainer)
    env "${envs[@]}" DRY_RUN=1 bash scripts/train/10_sft.sh 2>&1 | sed 's/^/DRY(10_sft.sh): /' | tee -a "$A/dry_run.log" > "$log"
    return 0
  fi
  env "${envs[@]}" DRY_RUN=0 bash scripts/train/10_sft.sh > "$log" 2>&1
}
eval_ck() {  # eval_ck SLOT ADAPTER: eval_decisions (raw) on the v1 dev files + temperatures on v1 dev (batch 16, 8, 4 on OOM)
  local slot=$1 ad=$2 bs ebs ok=0 base
  base=$(model_path "$slot")
  if [[ "$DRY_RUN" != 1 && -s "$A/eval/${slot}_v1.json" && -s "$A/cal/${slot}_calibration.json" ]]; then
    echo "$slot: evaluation and temperatures from an earlier attempt kept"; return 0
  fi
  for bs in 16 8 4; do
    X "$PY" -m jevharness.training.eval_decisions --model "$base" --adapter "$ad" --dev "${V1_FILES[@]}" --n-perm 0 \
      --max-len 4096 --batch-size "$bs" --out "$A/eval/${slot}_v1.json" > "$A/logs/eval_$slot.log" 2>&1 && { ok=1; break; }
    oom "$A/logs/eval_$slot.log" || break
    st "eval $slot: out of memory at batch $bs"
  done
  (( ok )) || { echo "eval_decisions failed ($slot)"; return 1; }
  ebs=$bs ok=0
  for bs in 16 8 4; do
    X "$PY" -m jevharness.training.calibrate --model "$base" --adapter "$ad" --dev data/train_v1/dev.jsonl.gz \
      --out "$A/cal/${slot}_calibration.json" --min-bucket 200 --batch-size "$bs" > "$A/logs/cal_$slot.log" 2>&1 && { ok=1; break; }
    oom "$A/logs/cal_$slot.log" || break
    st "calibrate $slot: out of memory at batch $bs"
  done
  (( ok )) || { echo "calibrate failed ($slot)"; return 1; }
  echo "eval_decisions batch $ebs; calibrate batch $bs" > "$A/settings/${slot}_eval_batch.txt"
}
replay_c() {  # replay_c SLOT: every distinct question of C (three invocations; replay resumes) + the calibration states
  local slot=$1 k rc name="e5-$1"
  serve_e5 "$slot" "$slot" || return 1
  for k in 1 2 3; do
    X "$PY" scripts/stage2_corpus.py replay --corpus "$A/s23" --name "e5_$slot" --s1 "$(spec_of "$name" "$PORT")" \
      --cache "cache/s1_e5_${slot}.sqlite" > "$A/logs/replay_${slot}_$k.log" 2>&1
    st "replay $slot invocation $k exit=$?"
  done
  for k in 1 2; do
    cal_replay "$name" "$PORT" "$A/cal/answers_$slot.jsonl" > "$A/logs/cal_replay_${slot}_$k.log" 2>&1
    rc=$?
    st "calibration-state replay $slot invocation $k exit=$rc"
    (( rc == 0 )) && break
  done
  stop_port "$PORT"
  dry_skip "cost check and completeness ($slot)" && return 0
  box cost --run "$A" > "$A/logs/cost_$slot.log" 2>&1 || { st "a replay record is priced or not local: STOP (cost_check.json)"; return 9; }
  box answers --run "$A" --arm "$slot" > "$A/logs/answers_$slot.txt" 2>&1; rc=$?
  st "replay $slot: $(cat "$A/logs/answers_$slot.txt"); calibration states $(wc -l < "$A/cal/answers_$slot.jsonl" 2>/dev/null || echo 0) answered"
  return $(( rc == 0 ? 0 : 1 ))                    # an incomplete replay never marks a model done (a rerun resumes it)
}

# ---------------------------------------------------------------------------------------------- stages
start_rec() {
  printf 'step e5_small (E5)\nstarted %s\ncommit %s\nprereg sha256 %s\ninputs sha256 %s\nhost %s %s\ncap_h %s guard_min %s idle_min %s\n' \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD 2>/dev/null)" "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" \
    "$(sha256sum results/explore/e5_small/inputs.json | cut -c1-16)" "$MY_IID" "$(itype)" "$CAP_H" "$GUARD_MIN" \
    "$IDLE_MIN" > "$A/START_e5_small.txt"
  echo "part ${E5_PART:-a} -> $S3OUT" >> "$A/START_e5_small.txt"
}
stage_rearm() {
  local it gpu
  arm_stops
  it=$(itype); gpu=$(gpu_name)
  echo "$it" > "$A/instance_type.txt"
  X nvidia-smi > "$A/nvidia_smi_start.txt" 2>&1
  X "$PY" -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'cuda', torch.cuda.is_available())" > "$A/env.txt" 2>&1
  box host --out "$A/host.json" --itype "$it" --gpu "$gpu" --cap-h "$CAP_H" --instance-id "$MY_IID" || return 1
  (compgen -e | grep -E '_API_KEYS?$' || true) > "$A/keys_in_env.txt"      # names only; must be empty
  [[ -s "$A/keys_in_env.txt" ]] && { echo "an API key is in the environment"; return 1; }
  [[ "$DRY_RUN" != 1 && -e .env ]] && { echo "a .env file is in the checkout (never read here): refusing"; return 1; }
  df -h "$W" "$HOME" > "$A/disk_start.txt" 2>&1
  if [[ "$DRY_RUN" != 1 ]]; then   # three models (~7 GB), adapters, caches and logs
    local avail; avail=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
    (( ${avail:-0} >= 20 )) || { echo "only ${avail:-?} GB free on $W (need 20)"; return 1; }
  fi
  [[ -f "$A/START_e5_small.txt" ]] || start_rec
  (( waited > 0 )) && st "waited $(( waited / 60 )) min for another chain before touching the checkout"
  cat "$A/host.json"
}
download() {  # download SLOT: the pinned revision into the HF cache; path + file hashes -> A/models/SLOT.json
  local repo rev
  repo=$(box model --arm "$1" --field repo); rev=$(box model --arm "$1" --field revision)
  if [[ "$DRY_RUN" == 1 ]]; then
    dry "$PY" -c "snapshot_download($repo, revision=$rev)" | tee -a "$A/dry_run.log"
    printf '{"slot": "%s", "repo": "%s", "revision": "%s", "path": "%s"}\n' "$1" "$repo" "$rev" "$W/hf/$1" > "$A/models/$1.json"
    return 0
  fi
  "$PY" - "$1" "$repo" "$rev" "$A/models/$1.json" <<'EOF'
import hashlib, json, os, sys
from huggingface_hub import snapshot_download
slot, repo, rev, out = sys.argv[1:5]
p = snapshot_download(repo, revision=rev)
files = {}
for f in sorted(os.listdir(p)):
    fp = os.path.join(p, f)
    if os.path.isfile(fp) and (f.endswith((".json", ".safetensors", ".model", ".txt", ".jinja")) or f == "tokenizer.model"):
        h = hashlib.sha256()
        with open(fp, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        files[f] = h.hexdigest()
assert os.path.basename(os.path.dirname(p)) == "snapshots" and os.path.basename(p) == rev, (p, rev)
json.dump({"slot": slot, "repo": repo, "revision": rev, "path": p, "sha256": files}, open(out, "w"), indent=1)
print(slot, repo, rev, p, len(files), "files")
EOF
}
stage_e5_inputs() {
  local f s
  X "$PY" scripts/train/ablation_subsets.py build --out "$A/data" --arms D100 > "$A/logs/build.log" 2>&1 || return 1
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz; do
    cp results/stage2/s23/$f "$A/s23/$f" && cmp -s results/stage2/s23/$f "$A/s23/$f" || return 1
  done
  S3 s3 cp "$S3IN/cal_requests.jsonl.gz" "$CALREQ" || return 1
  dry_skip "inputs check" || box inputs --run "$A" || return 1
  if ! verify_4b; then   # the cloned image carries the frozen checkpoint; fetch it only if it is not there
    st "the frozen 4B is absent or different in $CK; fetching $S3_FROZEN"
    S3 s3 sync "$S3_FROZEN" "$CK/$(plan "d['rows']['frozen']['dir']")" || return 1
    verify_4b || return 1
  fi
  [[ "$DRY_RUN" == 1 || -s "$BASE4B/config.json" ]] || { echo "the 4B base snapshot $BASE4B is missing"; return 1; }
  for s in $(slots); do download "$s" > "$A/logs/download_$s.log" 2>&1 || { echo "download $s failed"; tail -5 "$A/logs/download_$s.log"; return 1; }; done
  sha256sum results/stage2/s23/{corpus.json,occurrences.jsonl.gz,questions.jsonl.gz} data/train_v1/*.jsonl.gz \
    data/train_v0/dev.jsonl results/training/ablation_inputs.json results/explore/e5_small/inputs.json > "$A/inputs.sha256"
}
smoke_one() {  # smoke_one SLOT -> 0 ok (A/mem/SLOT.txt) | 1 failed (reason in A/smoke/SLOT.json)
  local slot=$1 s=nogc_3072 tried="" rc peak reason=""
  if [[ -s "$A/mem/$slot.txt" && "${SMOKE_AGAIN:-0}" != 1 ]]; then echo "$slot: smoke done earlier ($(cat "$A/mem/$slot.txt"))"; return 0; fi
  [[ -s "$A/models/$slot.json" ]] || download "$slot" > "$A/logs/download_$slot.log" 2>&1 || reason="download failed"
  if [[ -z "$reason" ]] && ! dry_skip "readout check ($slot)"; then
    box readout --model "$(model_path "$slot")" --out "$A/smoke/${slot}_readout.json" > "$A/logs/readout_$slot.log" 2>&1 \
      || reason="read-out check failed ($(tail -1 "$A/logs/readout_$slot.log" | cut -c1-120))"
  fi
  while [[ -z "$reason" && "$s" != none ]]; do
    smi_start "$A/logs/smoke_${slot}_${s}_mem.csv"
    sft_run "$slot" "$CKE/_smoke_$slot" "$A/smoke/smoke.jsonl" "$s" "$A/logs/smoke_${slot}_$s.log" 4; rc=$?
    smi_stop
    [[ "$DRY_RUN" == 1 ]] || rm -rf "$CKE/_smoke_$slot"
    if (( rc == 0 )); then
      if [[ "$DRY_RUN" != 1 ]] && box bad-loss "$A/logs/smoke_${slot}_$s.log"; then reason="non-finite loss"; break; fi
      tried+="$s:ok"; break
    fi
    if oom "$A/logs/smoke_${slot}_$s.log"; then tried+="$s:oom,"; s=$(box setting --next "$s"); continue; fi
    reason="smoke failed for a reason other than memory (logs/smoke_${slot}_$s.log)"
  done
  [[ -z "$reason" && "$s" == none ]] && reason="every memory setting ran out of memory"
  peak=$(box peak "$A/logs/smoke_${slot}_${s}_mem.csv" 2>/dev/null)
  printf '{"slot": "%s", "setting": "%s", "tried": "%s", "peak_mem_mib": "%s", "instance_type": "%s", "failed": "%s"}\n' \
    "$slot" "$s" "$tried" "${peak:-}" "$(itype)" "$reason" > "$A/smoke/$slot.json"
  if [[ -n "$reason" ]]; then st "smoke $slot: FAILED: $reason"; return 1; fi
  echo "$s" > "$A/mem/$slot.txt"
  st "smoke $slot: memory setting $s (tried $tried; peak ${peak:-?} MiB)"
}
stage_e5_smoke() {  # every slot in order (or e5_smoke:SLOT[,SLOT]); a failed MiniCPM5-1B smoke swaps in its backup
  local slot backup list
  list=$(if [[ -n "${1:-}" ]]; then echo "$1" | tr ',' ' '; else slots; fi)
  registered || return 1
  gpu_free || return 1
  dry_skip "smoke rows" || box smoke-file --run "$A" --out "$A/smoke/smoke.jsonl" || return 1
  for slot in $list; do
    smoke_one "$slot" && continue
    backup=$("$PY" -c "import sys; sys.path.insert(0, 'scripts/train'); import e5_box as E; print(E.BACKUP.get('$slot', ''))")
    [[ -n "$backup" ]] || { st "smoke: $slot failed and has no backup: stop"; return 1; }
    printf '{"swap": {"%s": "%s"}, "reason": "%s", "utc": "%s"}\n' "$slot" "$backup" \
      "$("$PY" -c "import json; print(json.load(open('$A/smoke/$slot.json'))['failed'])" | tr '"' "'")" \
      "$(date -u +%FT%TZ)" > "$A/swap.json"
    st "smoke: $slot failed; the backup $backup replaces it (swap.json, Deviations entry (b))"
    smoke_one "$backup" || { st "smoke: the backup $backup failed too"; return 1; }
  done
}
run_arm() {  # run_arm SLOT -> 0 done | 1 failed | 3 skipped by the cap | 9 priced record (stop everything)
  local slot=$1 out="$CKE/$1" setting log rc t0 m=0 tried="" mix_sha peak verdict
  [[ -f "$A/done/$slot" ]] && { echo "$slot: done earlier"; return 0; }
  verdict=$(box cap --run "$A" --arm "$slot" --cap-h "$CAP_H"); rc=$?
  st "cap: $verdict"
  (( rc == 0 )) || return 3
  gpu_free || return 1
  if [[ -f "$A/settings/$slot.json" && -f "$out/adapter_model.safetensors" ]]; then
    st "$slot: trained earlier; resuming at the evaluation"
  else
    rm -f "$A/s23/answers_e5_$slot.jsonl" "$A/s23/replay_e5_$slot.json" "$A/eval/${slot}_v1.json" \
      "$A/cal/${slot}_calibration.json" "$A/cal/answers_$slot.jsonl" cache/s1_e5_"${slot}".sqlite
    setting=$(cat "$A/mem/$slot.txt" 2>/dev/null || echo nogc_3072)
    while :; do
      log="$A/logs/sft_${slot}_$setting.log"
      t0=$(date +%s)
      smi_start "$A/logs/sft_${slot}_${setting}_mem.csv"
      sft_run "$slot" "$out" "$A/data/D100.jsonl" "$setting" "$log"; rc=$?
      smi_stop
      m=$(( ($(date +%s) - t0) / 60 ))
      st "stage sft_$slot exit=$rc ($m min)"
      if (( rc == 0 )); then tried+="$setting:ok"; break; fi
      if oom "$log"; then
        tried+="$setting:oom,"
        setting=$(box setting --next "$setting")
        [[ "$setting" == none ]] && { st "$slot: every memory setting ran out of memory"; return 1; }
        echo "$setting" > "$A/mem/$slot.txt"
        st "$slot: out of memory; retrying with $setting"
        continue
      fi
      st "$slot: SFT failed (logs/sft_${slot}_$setting.log)"
      return 1
    done
    if [[ "$DRY_RUN" != 1 ]] && box bad-loss "$log"; then st "$slot: non-finite training loss: model failed"; return 1; fi
    mix_sha=$(sha256sum "$out/train_mix.jsonl" 2>/dev/null | cut -c1-64)
    peak=$(box peak "$A/logs/sft_${slot}_${setting}_mem.csv" 2>/dev/null)
    box record --run "$A" --arm "$slot" --setting "$setting" --attempts "$tried" --sft-min "$m" --itype "$(itype)" \
      --log "$log" --mix-sha "${mix_sha:-}" --peak-mib "${peak:-}" || return 1
    if [[ "$DRY_RUN" != 1 ]] && ! "$PY" -c "import json, sys; sys.exit(0 if json.load(open('$A/settings/$slot.json')).get('train_mix_ok') else 1)"; then
      st "$slot: the training file differs from SFT v1's (train_mix sha256): model failed"
      return 1
    fi
    cp -f "$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -V | tail -1)/trainer_state.json" \
      "$A/logs/trainer_state_$slot.json" 2>/dev/null
    [[ "$DRY_RUN" == 1 ]] || rm -rf "$out"/checkpoint-* "$out/train_mix.jsonl"
  fi
  t0=$(date +%s)
  eval_ck "$slot" "$out" || return 1
  [[ "$DRY_RUN" == 1 ]] || cp -f "$A/cal/${slot}_calibration.json" "$out/calibration.json"
  sync_ck "$slot"
  replay_c "$slot"; rc=$?
  (( rc == 0 )) || return "$rc"
  box arm-time --run "$A" --arm "$slot" --eval-min "$(( ($(date +%s) - t0) / 60 ))"
  touch "$A/done/$slot"
  score_box
  st "model $slot done (setting $(cat "$A/mem/$slot.txt" 2>/dev/null); eval + replays $(( ($(date +%s) - t0) / 60 )) min)"
}
arms_run() {  # arms_run SLOT...: in order; a model the cap skips is reported as not run and the chain goes on
  local slot rc fails=0
  registered || return 1
  if [[ "$DRY_RUN" != 1 ]]; then box inputs --run "$A" > /dev/null || { st "arms: inputs check failed"; return 1; }; fi
  for slot in "$@"; do
    [[ -s "$A/mem/$slot.txt" || "$DRY_RUN" == 1 ]] || { st "$slot: no smoke record (send e5_smoke first)"; fails=$((fails + 1)); continue; }
    run_arm "$slot"; rc=$?
    (( rc == 9 )) && return 9
    if (( rc == 3 )); then st "model $slot NOT RUN (cap)"; continue; fi
    if (( rc != 0 )); then
      fails=$((fails + 1)); st "model $slot failed (rc $rc)"
      (( fails >= 2 )) && { st "two models in a row failed: stopping the chain"; return 1; }
    else
      fails=0
    fi
  done
}
stage_e5_arms() { arms_run $(slots); }
stage_e5_arm() { arms_run "${1:?e5_arm:SLOT}"; }
bench_one() {  # bench_one LABEL PORT: harness_final.py bench c = 1 (25 episodes) and c = 4 (100), measure_latency c = 1, 4
  local label=$1 port=$2 c lim
  for c in 1 4; do
    lim=()
    [[ $c == 1 ]] && lim=(--limit 25)
    smi_start "$A/latency/smi_${label}_c$c.csv"
    X "$PY" scripts/harness_final.py bench --url "http://127.0.0.1:$port" --label "$label" --path new \
      --concurrency "$c" --out "$A/latency/bench" "${lim[@]}" >> "$A/latency/bench.log" 2>&1
    local rc=$?
    smi_stop
    (( rc == 0 )) || return 1
    st "latency $label bench c=$c done"
  done
  X "$PY" scripts/measure_latency.py --s1 "$(spec_of "$3" "$port")" --concurrency 1 4 --n 200 --state-tokens 1500 \
    --questions 3 > "$A/latency/${label}_synthetic.json" 2> "$A/latency/${label}_synthetic.err" || return 1
  [[ "$DRY_RUN" == 1 ]] || curl -sf "http://127.0.0.1:$port/v1/models" > "$A/latency/models_$label.json"
  st "latency $label synthetic done"
}
stage_e5_latency() {  # one session, one GPU, one model served at a time: the frozen 4B first, the models, the 4B again (c = 1)
  local slot
  gpu_free || return 1
  stop_all
  if [[ ! -s "$A/latency/4b_synthetic.json" || "$DRY_RUN" == 1 ]]; then
    serve_4b 4b_lat || return 1
    bench_one 4b "$PORT4B" jevagentrl-4b || return 1
    stop_port "$PORT4B"
  fi
  for slot in $(slots); do
    [[ -f "$A/done/$slot" || "$DRY_RUN" == 1 ]] || { st "latency: $slot has no finished model; not benched"; continue; }
    [[ -s "$A/latency/${slot}_synthetic.json" && "$DRY_RUN" != 1 ]] && continue
    serve_e5 "$slot" "${slot}_lat" || return 1
    bench_one "$slot" "$PORT" "e5-$slot" || return 1
    stop_port "$PORT"
  done
  serve_4b 4b_lat_end || return 1     # drift check: the 4B again at c = 1 at the end of the session
  smi_start "$A/latency/smi_4b_end_c1.csv"
  X "$PY" scripts/harness_final.py bench --url "http://127.0.0.1:$PORT4B" --label 4b_end --path new --concurrency 1 \
    --limit 25 --out "$A/latency/bench" >> "$A/latency/bench.log" 2>&1
  smi_stop
  stop_port "$PORT4B"
  dry_skip "compact bench files" || box compact-bench --dir "$A/latency/bench" --out "$A/latency/bench_compact"
  st "latency: session done"
}
stage_e5_cal4b() {  # entry (g)'s check: the frozen 4B through the same calibration-state replay
  local verdict
  verdict=$("$PY" -c "import sys; sys.path.insert(0, 'scripts/train'); import ablation_box as BX; u = BX.used_hours(__import__('pathlib').Path('$A')); print('go' if u + 0.5 <= $CAP_H else 'skip', round(u, 2))")
  st "cap (4B calibration replay, 0.5 h): $verdict"
  [[ "$verdict" == go* ]] || return 0
  gpu_free || return 1
  serve_4b 4b_cal || return 1
  cal_replay jevagentrl-4b "$PORT4B" "$A/cal/answers_4b.jsonl" > "$A/logs/cal_replay_4b.log" 2>&1
  local rc=$?
  stop_port "$PORT4B"
  (( rc == 0 )) || return 1
  dry_skip "4B calibration check" && return 0
  "$PY" scripts/e5_cal_replay.py check --requests "$CALREQ" --answers "$A/cal/answers_4b.jsonl" --out "$A/cal/check_4b.json" \
    > "$A/logs/cal_check_4b.log" 2>&1
  st "4B calibration-state replay: $(cat "$A/logs/cal_check_4b.log" | tail -1 | cut -c1-200)"
  return 0
}
stage_e5_final() {
  local rc=0 used price
  stop_all
  if ! dry_skip "final cost check"; then box cost --run "$A" || rc=1; fi
  X nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  score_box
  used=$(box used-h --run "$A")
  price=$("$PY" -c "import json; print(json.load(open('$A/host.json')).get('usd_per_h') or '')" 2>/dev/null)
  printf '{"cost_check_ok": %s, "gpu_hours_used": %s, "usd_per_h": "%s", "models_done": "%s", "instance_id": "%s", "finished_utc": "%s"}\n' \
    "$( (( rc == 0 )) && echo true || echo false)" "$used" "$price" "$(ls "$A/done" 2>/dev/null | tr '\n' ' ')" "$MY_IID" \
    "$(date -u +%FT%TZ)" > "$A/final_check.json"
  for slot in $(ls "$A/done" 2>/dev/null); do sync_ck "$slot"; done
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; X sudo shutdown -h +2 "jev-e5-small: done"; fi
  return $rc
}

# ---------------------------------------------------------------------------------------------- live row (entry (g))
LIVE_KEYS=${LIVE_KEYS:-OPENROUTER_API_KEY}   # Deviations 2026-10-06 ~18:00 UTC: System-2 through OpenRouter
keys() {  # the System-2 key from SSM (never on disk, never printed); only inside a live stage's subshell
  [[ "$DRY_RUN" == 1 ]] && { export OPENROUTER_API_KEY=dry-run-not-a-key; return 0; }
  local v val
  for v in $LIVE_KEYS; do
    val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value \
          --output text 2>/dev/null) || val=""
    if [[ -n "$val" && "$val" != None ]]; then export "$v=$val"; fi
  done
  unset val
  local any=0
  for v in $LIVE_KEYS; do [[ -n "${!v:-}" ]] && any=1; done
  (( any )) || { echo "no key of [$LIVE_KEYS] in SSM"; return 1; }
}
live_slot() {  # the E5 slot of the live config (s1_backends.e5_small.name = e5-<slot>)
  "$PY" -c "import yaml,sys; n=yaml.safe_load(open(sys.argv[1]))['s1_backends']['e5_small']['name']; print(n[3:] if n.startswith('e5-') else '')" "$LIVE_CFG"
}
live_ready() {  # both servers up, the config's profile has no pending block, the registration is in the checkout
  [[ "$DRY_RUN" == 1 ]] && return 0
  grep -q 'Exploratory row E5' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E5"; return 1; }
  curl -sf "http://127.0.0.1:$PORT4B/health" > /dev/null || { echo "the 4B server is not up"; return 1; }
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "the E5 server is not up"; return 1; }
  ! grep -q 'calibration: pending' configs/harness/explore_e5_final.yaml || { echo "explore_e5_final.yaml is pending"; return 1; }
}
stage_live_serve() {
  local slot
  slot=$(live_slot)
  [[ -n "$slot" ]] || { echo "no E5 model in $LIVE_CFG"; return 1; }
  if [[ "$DRY_RUN" != 1 && ! -f "$CKE/$slot/adapter_model.safetensors" ]]; then
    st "live: fetching the $slot adapter from $S3CK/$slot"
    S3 s3 sync "$S3CK/$slot" "$CKE/$slot" || return 1
  fi
  [[ "$DRY_RUN" == 1 || -s "$A/cal/${slot}_calibration.json" ]] || cp -f "$CKE/$slot/calibration.json" "$A/cal/${slot}_calibration.json"
  gpu_free || return 1
  # both models share the one A10G: each server's session cache is capped (exact read-outs; reuse only), else the
  # 4B's cache grows to ~18 GB and the E5 server runs out of memory (first live attempt, 18:05 UTC)
  serve_4b live_4b --cache-tokens "$LIVE_CACHE_TOKENS" || return 1
  serve_e5 "$slot" "live_$slot" --cache-tokens "$LIVE_CACHE_TOKENS" || return 1
  st "live: serving jevagentrl-4b on :$PORT4B and e5-$slot on :$PORT"
}
ops() {  # operational fields only: tasks, errored, $/episode, System-1 models
  "$PY" - "$1" <<'EOF'
import collections, glob, json, sys
for f in sorted(glob.glob(sys.argv[1] + "/*.jsonl")):
    last = {}
    for l in open(f):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if isinstance(r, dict) and "task_id" in r:
            last[r["task_id"]] = r
    errs = sum(1 for r in last.values() if r.get("error"))
    c = sum(float(((r.get("usage") or {}).get("total") or {}).get("cost_usd") or 0) for r in last.values())
    m = collections.Counter(x.get("model") for r in last.values() for x in r.get("calls") or [] if x.get("component") == "s1")
    print(f"{f}: tasks {len(last)} latest-errored {errs} cost/ep ${c / max(1, len(last)):.4f} s1 {dict(m)}")
EOF
}
live_spend() { box spend "$SMOKE_RUN"/*.jsonl "$LIVE_RUN"/*.jsonl 2>/dev/null | "$PY" -c "import json,sys; print(json.load(sys.stdin)['usd'])"; }
stage_live_smoke() {
  keys || return 1
  live_ready || return 1
  X "$HV/bin/jevbench" run "$SMOKE_CFG" --out runs -o workers=1 > "$A/logs/live_smoke.log" 2>&1
  local rc=$? h
  [[ "$DRY_RUN" == 1 ]] || ops "$SMOKE_RUN" >> "$A/live_smoke_ops.txt"
  sync_run "$SMOKE_RUN"
  st "live smoke exit=$rc (spend so far \$$(live_spend))"
  if [[ "$DRY_RUN" != 1 ]] && ! h=$(box health "$SMOKE_RUN"/*.jsonl); then
    st "live smoke: route not healthy ($h): the live row is not started"; return 1
  fi
  st "live smoke health: ${h:-dry}"
  return $rc
}
watchdog() {  # kill the live runner when the logged spend reaches LIVE_CAP_USD (every attempt counted), the route degrades
  # (a new errored episode or more than 4 new retried LLM calls) or a System-1 server fails (a new JevAPIError episode)
  local files=("$LIVE_RUN"/llm_only.jsonl "$LIVE_RUN"/harness_final.jsonl "$LIVE_RUN"/e5_final.jsonl) base lim
  base=$(box health "${files[@]}" --max-errors 999999 --max-s1-errors 999999 --max-retried 999999)
  lim=$("$PY" -c "import json,sys; b=json.loads(sys.argv[1]); print(b['errored_route'] + b['errored_other'], b['errored_s1'], b['s2_retried'] + 4)" "$base")
  read -r lim_e lim_s1 lim_r <<< "$lim"
  echo "$(date -u +%FT%TZ) watchdog baseline $base" >> "$A/live_watchdog.log"
  while sleep 30; do
    local s h
    if ! h=$(box health "${files[@]}" --max-errors "$lim_e" --max-s1-errors "$lim_s1" --max-retried "$lim_r"); then
      echo "$(date -u +%FT%TZ) route or System-1 degraded ($h): pausing the live row (Deviations 2026-10-06 ~18:00 UTC)" \
        > "$A/live_watchdog_stop.txt"
      pkill -f "jevbench run $LIVE_CFG"
      return 0
    fi
    s=$(live_spend)
    if "$PY" -c "import sys; sys.exit(0 if float('${s:-0}') >= float('$LIVE_CAP_USD') else 1)"; then
      echo "$(date -u +%FT%TZ) spend \$$s >= cap \$$LIVE_CAP_USD: stopping the live runner" > "$A/live_watchdog_stop.txt"
      pkill -f "jevbench run $LIVE_CFG"
      return 0
    fi
  done
}
stage_live() {  # entry (g): one invocation per pass, one worker, three passes (passes 2-3 re-run errored tasks only)
  local pass wd rc
  keys || return 1
  live_ready || return 1
  printf 'step e5_small_live\nstarted %s\ncommit %s\nconfig %s %s\nprofile %s\nprereg sha256 %s\ncap_usd %s\n' \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD 2>/dev/null)" "$LIVE_CFG" "$(sha256sum "$LIVE_CFG" | cut -c1-16)" \
    "$(sha256sum configs/harness/explore_e5_final.yaml | cut -c1-16)" "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" \
    "$LIVE_CAP_USD" > "$A/START_e5_small_live.txt"
  [[ "$DRY_RUN" == 1 ]] || curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/live_server_models_start.json"
  if [[ -f "$A/live_watchdog_stop.txt" ]]; then      # an earlier stop, reported already: keep it, start a fresh watch
    mv "$A/live_watchdog_stop.txt" "$A/live_watchdog_stop_$(date -u +%H%M%S).txt"
  fi
  if [[ "$DRY_RUN" != 1 ]]; then watchdog & wd=$!; fi
  for pass in 1 2 3; do
    [[ -f "$A/live_watchdog_stop.txt" ]] && { st "live: watchdog stop: $(cat "$A/live_watchdog_stop.txt")"; break; }
    X env JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$LIVE_CFG" --out runs -o workers=1 > "$A/logs/live_pass$pass.log" 2>&1
    rc=$?
    [[ "$DRY_RUN" == 1 ]] || ops "$LIVE_RUN" >> "$A/live_ops.txt"
    sync_run "$LIVE_RUN"
    st "live pass $pass exit=$rc (spend so far \$$(live_spend))"
  done
  [[ -n "${wd:-}" ]] && kill "$wd" 2>/dev/null
  [[ "$DRY_RUN" == 1 ]] || curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/live_server_models_end.json"
  [[ -f "$A/live_watchdog_stop.txt" ]] && return 1
  return 0
}
stage_live_final() {
  stop_all
  sync_run "$SMOKE_RUN"
  sync_run "$LIVE_RUN"
  X nvidia-smi > "$A/nvidia_smi_live_end.txt" 2>&1
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; X sudo shutdown -h +2 "jev-e5-small: live done"; fi
}

broke=0
for tok in $STAGES; do
  s=${tok%%:*}; arg=""
  [[ "$tok" == *:* ]] && arg=${tok#*:}
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $tok"; continue; fi
  st "stage $tok start ($(git log --oneline -1 2>/dev/null | cut -c1-50); host $MY_IID $(itype))"
  t0=$(date +%s)
  printf '{"stage": "%s", "start": "%s"}\n' "$tok" "$(date -u +%FT%TZ)" > "$A/cur_stage.json"
  ( "stage_$s" ${arg:+"$arg"} ) > "$A/logs/stage_${tok//[:,+]/_}.log" 2>&1
  rc=$?
  secs=$(( $(date +%s) - t0 ))
  rm -f "$A/cur_stage.json"
  printf '{"stage": "%s", "min": %s}\n' "$tok" "$(awk -v s="$secs" 'BEGIN { printf "%.2f", s / 60 }')" >> "$A/stage_times.jsonl"
  st "stage $tok exit=$rc ($(( secs / 60 )) min)"
  if (( rc != 0 )) && [[ "$s" != e5_final && "$s" != live_final ]]; then st "stopping the chain after $tok failed"; broke=1; break; fi
done
stop_all
if (( broke )); then
  score_box
  sync_out
  st "chain stopped; the box stops itself after $IDLE_MIN idle minutes"
fi
st "box done [$STAGES]"
