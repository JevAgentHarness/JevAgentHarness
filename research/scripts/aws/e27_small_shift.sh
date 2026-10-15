#!/usr/bin/env bash
# Exploratory row E27, a 0.8B decision model that holds under shift (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09
# "E27: a 0.8B decision model that holds under shift"; descriptive, outside every Holm table). One NEW GPU host launched
# by this driver from E3's clone image (tag Name=jev-e27-08b), SSM only. E14 arm A's recipe (Qwen3.5-0.8B + a fresh Clef
# head, Unsloth 2026.10.2 in its own venv) on surface-augmented agent states (arm S) and on S + E12's public items (SP),
# one epoch each; L1 (E1's 3,991 decisions), L2 in-domain (corpus C), L2 shift (E21's corpus, main rendering, then the
# surface-undone re-rendering), L3 shift (REPLACE, E21's 200 tasks); the selection rule picks AgentDecision-0.8B, which
# answers the calibration questions and gets L3 in-domain, the gated final harness in-domain and (caps allowing) on the
# shift. Flash on the direct Gemini API (the key from SSM into the live stages' process only, never printed or on disk).
#
#   LOCAL:  bash scripts/aws/e27_small_shift.sh launch [itype ...]   # one new host (refuses if a jev-e27-08b host exists)
#           E27_IID=i-... bash scripts/aws/e27_small_shift.sh push | send "STAGES" | status [N] | start | stop | itype | state
#   BOX stages (in the order sent; a failed rearm / code / setup / data stops the chain; `final` always runs when sent):
#     rearm        idle stop (IDLE_MIN) + power-off guard (GUARD_MIN from the first boot), host record
#     code         the committed branch in /opt/jev/e27 (records already on the host are kept: untracked files)
#     setup        code + Unsloth's venv; the calibration questions (sha256); the frozen 4B (latency only, sha256)
#     data         E27's build (vocabulary check first; refuses on any overlap): train_s, train_sp, dev_a
#     train_s | train_sp      one epoch (scripts/train/e27_clef.py train) -> s3 checkpoints/e27_small_shift/<arm>/final
#     eval_s | eval_sp        served on :8083: L1 (E1), L2 C, L2 shift (main, then surface-undone)
#     select       the registered selection rule -> box/selected.json (AgentDecision-0.8B)
#     cal_sel      AgentDecision-0.8B answers the 2,719 calibration questions ($0)
#     shift_sel | shift_other     L3 shift REPLACE (smoke, gate, row) for the selected / the other arm
#     live_in      L3 in-domain REPLACE (selected)
#     wait_prev    wait for every older E27 chain on this host to end
#     final_in | final_shift      the gated final harness (profile explore_e27_final.yaml must be fitted and committed)
#     latency      harness_final.py bench c = 1 / 4: AgentDecision-0.8B, then the frozen 4B (lowest priority)
#     final        stop the servers, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e27_small_shift/box -> s3://$BUCKET/e27_small_shift/box; runs -> e27_small_shift/runs.
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E27_IID=${E27_IID:-}
# every other project host (every instance id in the other drivers, and E21's, E24's and E25's hosts): never touched
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e27-08b
BUNDLE=e27.bundle
S3OUT="$B/e27_small_shift"
CAP_USD=${CAP_USD:-30}               # registration (h): direct Gemini at most 30 USD
GUARD_MIN=${GUARD_MIN:-590}          # 10 host-hours: the guard 9 h 50 min after the first boot
IDLE_MIN=${IDLE_MIN:-30}
POWEROFF=${POWEROFF:-1}
PORT=8083
PORT4B=8081

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E27_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E27_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E27_IID "* ]] && { echo "refused: $E27_IID is not the E27 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E27_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E27_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e27-08b: power-off guard (10 host-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g6e.2xlarge g6e.4xlarge g5.2xlarge)
  for IT in "${types[@]}"; do
    for SUBNET in $SUBNETS; do
      out=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$IT" --subnet-id "$SUBNET" --security-group-ids "$SG" \
        --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior stop --count 1 \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=300,VolumeType=gp3,Throughput=500,Iops=6000,DeleteOnTermination=true}' \
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/e27_small_shift.sh" > "$tmp/e27_small_shift.sh"   # the committed driver
  awsl s3 cp "$tmp/e27_small_shift.sh" "$B/code/e27_small_shift.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e27_small_shift.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e27_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e27_small_shift.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q GUARD_MIN=%q IDLE_MIN=%q POWEROFF=%q E27_IID=%q WAIT_MAX_MIN=%q PROBE_WAIT_MIN=%q SHIFT_FINAL_MIN=%q LAT_MIN=%q\n' \
      "$CAP_USD" "$GUARD_MIN" "$IDLE_MIN" "$POWEROFF" "$E27_IID" "${WAIT_MAX_MIN:-420}" "${PROBE_WAIT_MIN:-60}" \
      "${SHIFT_FINAL_MIN:-60}" "${LAT_MIN:-45}"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e27_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e27_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E27_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e27_driver.sh /tmp/e27_driver.sh --only-show-errors && bash /tmp/e27_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E27_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype || "$MODE" == state ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E27_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime,StateReason.Code]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,28p' "$0"
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
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E27_IID" && "$MY_IID" != "$E27_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E27 host ${E27_IID:-?}"; exit 1
fi
R=$W/e27
A=$R/results/explore/e27_small_shift/box
STATUS_DIR=$W/e27_status
HV=$W/hvenv                 # harness venv (runner, replays, the frozen 4B for the latency stage)
EV=$W/e14venv               # Unsloth 2026.10.2: training and serve_jev --backend clef (as E14 / E18 / E21)
D=$W/e27_data
CKE=$W/ck_e27
CK=$W/ck
PLAN=results/stage2/rl_eval_plan.json
EX=scripts/explore_e27_small_shift.py
CFG=configs/explore/e27_small_shift
STOP=runs/e27_STOP.json
PROBE_LOG=$A/api_probe_log.txt
CAL_REQ_SHA=d46701e8e9bdc69e6a9e047227eabb1e52578ddc527342dda10b33f7dc5fff5e
E12_MIX_SHA=fcdb3b9a9a1f1194ebf2335251d500a75c37dc75ad5d3cafa1229ce71e539521
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$STATUS_DIR"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  aws s3 cp "$STATUS_DIR/status.txt" "$S3OUT/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION" --exclude 's23/corpus.json' \
    --exclude 's23/occurrences.jsonl.gz' --exclude 's23/questions.jsonl.gz' --exclude 'c_shift/questions*' \
    --exclude 'c_shift/occurrences.jsonl.gz' --exclude 'c_shift/corpus.json' --exclude 'inputs/*'
  local d
  for d in "$R"/runs/e1_bfcl_when2call_e27 "$R"/runs/e27_*; do
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
first_boot() { cat "$W/e27_first_boot" 2>/dev/null || date -d "$(uptime -s)" +%s; }
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
leak_check() {  # no key value in any record or status file
  local n=0 k ks=${GEMINI_API_KEYS:-}
  for k in ${GEMINI_API_KEY:-} ${ks//,/ }; do
    [[ ${#k} -ge 16 ]] || continue
    n=$(( n + $(grep -rlF -- "$k" "$R/runs" "$A" "$STATUS_DIR" 2>/dev/null | wc -l) ))
  done
  echo "key-leak files: $n"
  [[ "$n" == 0 ]]
}
sha_check() {  # sha_check FILE REFERENCE_SHA256 LABEL
  local got; got=$(sha256sum "$1" 2>/dev/null | cut -d' ' -f1)
  echo "$3 $(basename "$1") $got $([[ "$got" == "$2" ]] && echo ok || echo "MISMATCH (want $2)")" >> "$A/sha256_checks.txt"
  [[ "$got" == "$2" ]]
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
smi_start() { nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 1000 > "$1" 2>/dev/null & SMI=$!; }
smi_stop() { [[ -n "${SMI:-}" ]] && kill "$SMI" 2>/dev/null; SMI=""; }
sel_arm() { python3 -c "import json; print(json.load(open('$A/selected.json'))['selected'])" 2>/dev/null; }
other_arm() { local s; s=$(sel_arm); [[ "$s" == S ]] && echo SP || { [[ "$s" == SP ]] && echo S; }; }

# ------------------------------------------------------------------------------------------------ servers
stop_port() {
  local f="$W/e27_serve_$1.pid" p
  if [[ -f "$f" ]]; then
    p=$(cat "$f"); kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$p" 2>/dev/null; kill -9 "$p" 2>/dev/null
  fi
  rm -f "$f"
}
wait_up() {  # wait_up PORT NAME TAG
  local port=$1 name=$2 tag=$3
  for _ in $(seq 1 240); do
    if curl -sf -m 10 "http://127.0.0.1:$port/health" > "$A/logs/serve_${tag}_health.json" 2>/dev/null; then
      curl -sf -m 10 "http://127.0.0.1:$port/v1/models" > "$A/logs/serve_${tag}_models.json"
      grep -q "\"$name\"" "$A/logs/serve_${tag}_models.json" || { echo "served model id is not $name"; return 1; }
      printf '{"server_up": "%s", "name": "%s", "port": %s, "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" "$name" "$port" \
        "$(imds instance-type)" >> "$A/serve_windows.jsonl"
      return 0
    fi
    kill -0 "$(cat "$W/e27_serve_$port.pid")" 2>/dev/null || { echo "server died"; tail -30 "$A/logs/serve_$tag.out"; return 1; }
    sleep 5
  done
  echo "server did not come up"; return 1
}
serve_arm() {  # serve_arm ARM NAME: the arm's merged checkpoint behind serve_jev --backend clef on :8083 (engine v1)
  local arm=$1 name=$2
  stop_port "$PORT"
  [[ -s "$CKE/$arm/final/joint_head.safetensors" ]] || { echo "no final checkpoint for $arm"; return 1; }
  if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then echo "port $PORT is served by another process"; return 1; fi
  echo "$(date -u +%FT%TZ) serve_jev --backend clef --model $CKE/$arm/final --name $name --port $PORT" >> "$A/logs/serve_$name.cmd"
  setsid nohup env PYTHONPATH="$R/src" "$EV/bin/python" -m jevharness.training.serve_jev --backend clef --model "$CKE/$arm/final" \
    --name "$name" --port "$PORT" > "$A/logs/serve_$name.out" 2>&1 < /dev/null &
  echo $! > "$W/e27_serve_$PORT.pid"
  wait_up "$PORT" "$name" "$name"
}
serve_4b() {
  stop_port "$PORT4B"
  verify_frozen || return 1
  echo "serve_jev --model $(base4b) --adapter $(frozen_dir) --calibration $(frozen_cal) --name jevagentrl-4b --port $PORT4B --fast" > "$A/logs/serve_4b.cmd"
  setsid nohup env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev --model "$(base4b)" --adapter "$(frozen_dir)" \
    --calibration "$(frozen_cal)" --name jevagentrl-4b --port "$PORT4B" --fast > "$A/logs/serve_4b.out" 2>&1 < /dev/null &
  echo $! > "$W/e27_serve_$PORT4B.pid"
  wait_up "$PORT4B" jevagentrl-4b 4b
}
spec_of() {
  printf '{"type": "jev", "name": "%s", "base_url": "http://127.0.0.1:%s", "model": "%s", "api_key": "", "timeout": 120.0, "price_model": "jevagentrl-4b"}' \
    "$1" "$2" "$1"
}

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
  for round in 1 2 3 4 5 6 7 8; do
    [[ -f "$STOP" ]] && { st "spend cap reached earlier; $cfg not run further"; return 3; }
    probe_ok || { st "probes not clean; $cfg paused (round $round)"; return 4; }
    st "$cfg: start (round $round) $*"
    setsid env PYTHONPATH="$R/src" JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$cfg" --out runs "$@" >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PY "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --log "$PROBE_LOG" --every 30 >> "$A/watch.log" 2>&1 &
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
    PY "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --log "$PROBE_LOG" --every 30 --no-pause >> "$A/watch.log" 2>&1 &
    wpid=$!
    wait "$pid"; kill "$wpid" 2>/dev/null
  fi
  st "$cfg done; errored now $(PY "$EX" errored --run "$run"); $(PY "$EX" spend --runs runs --log "$PROBE_LOG" --short 2>/dev/null)"
  sync_out
  [[ -f "$STOP" ]] && return 3
  return 0
}
live_row() {  # live_row TAG ARM NAME SMOKE_CFG CFG [MIN_MINUTES]: serve, smoke, gate, the row (registration (g))
  local tag=$1 arm=$2 name=$3 smoke=$4 cfg=$5 need=${6:-0} g nfit rc t0 mt=null
  cd "$R" || return 1
  if (( need > 0 && $(mins_left) < need )); then st "$tag: not run ($(mins_left) min left of the guard, $need needed)"; return 0; fi
  keys || return 1
  serve_arm "$arm" "$name" > "$A/logs/serve_${tag}_start.log" 2>&1 || { st "$tag: did not serve: $(tail -3 "$A/logs/serve_${tag}_start.log" | tr '\n' ' ')"; return 1; }
  run_live "$CFG/$smoke.yaml" "runs/e27_$smoke" || { st "$tag: smoke failed"; stop_port "$PORT"; return 1; }
  g=$(PY "$EX" gate --smoke "runs/e27_$smoke" --runs runs --log "$PROBE_LOG" --n 200 --cap "$CAP_USD")
  echo "{\"row\": \"$tag\", \"gate\": $g}" >> "$A/gates.jsonl"
  nfit=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['n_fit'])" "$g" 2>/dev/null || echo 0)
  st "$tag gate: $g"
  (( nfit > 0 )) || { st "$tag: the gate refuses the row"; stop_port "$PORT"; unset GEMINI_API_KEY GEMINI_API_KEYS; return 3; }
  printf 'row %s\narm %s\nstarted %s\ncommit %s\nconfig %s %s\nprereg sha256 %s\ncap_usd %s\nhost %s %s\n' "$tag" "$arm" \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$CFG/$cfg.yaml" "$(sha256sum "$CFG/$cfg.yaml" | cut -c1-16)" \
    "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" "$CAP_USD" "$MY_IID" "$(imds instance-type)" > "$A/START_$tag.txt"
  t0=$(date -u +%FT%TZ)
  smi_start "$A/logs/smi_live_$tag.csv"
  if (( nfit < 200 )); then mt=$nfit; run_live "$CFG/$cfg.yaml" "runs/e27_$cfg" --max-tasks "$nfit"
  else run_live "$CFG/$cfg.yaml" "runs/e27_$cfg"; fi
  rc=$?
  smi_stop
  printf '{"row": "%s", "arm": "%s", "window_start": "%s", "window_end": "%s", "instance_type": "%s", "instance_id": "%s", "run": "runs/e27_%s", "max_tasks": %s, "exit": %s}\n' \
    "$tag" "$arm" "$t0" "$(date -u +%FT%TZ)" "$(imds instance-type)" "$MY_IID" "$cfg" "$mt" "$rc" > "$A/window_$tag.json"
  curl -sf -m 10 "http://127.0.0.1:$PORT/health" > "$A/logs/health_${tag}_end.json" 2>/dev/null
  stop_port "$PORT"
  leak_check || { unset GEMINI_API_KEY GEMINI_API_KEYS; return 1; }
  unset GEMINI_API_KEY GEMINI_API_KEYS
  sync_out
  return $rc
}

# ------------------------------------------------------------------------------------------------ stages
stage_rearm() {
  [[ -s "$W/e27_first_boot" ]] || date -d "$(uptime -s)" +%s > "$W/e27_first_boot"
  local left; left=$(mins_left)
  (( left > 0 )) || { echo "guard already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$left" "jev-e27-08b: power-off guard (host-hour cap)"
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E27 chain / runner / trainer / replay process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e27_[0-9]|jevbench|e27_clef|explore_e27|explore_e21_shift|stage2_corpus|decision_eval|e5_cal_replay|harness_final|uv pip|pip install' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e27-08b: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $left min left, idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU; $(df -BG / | awk 'NR==2{print $4}') free; used $(hours_used) h"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
}
stage_code() {
  local myb="$W/e27.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A/logs" "$A/eval" "$A/cal" "$A/latency/bench" runs
  st "code at $(git log --oneline -1 | cut -c1-72)"
  grep -q 'E27: a 0.8B decision model that holds under shift' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E27"; return 1; }
}
stage_setup() {
  stage_code || return 1
  PY -c "import jevharness, torch, sys; p = jevharness.__file__; print(p, torch.__version__, torch.cuda.is_available()); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_hvenv.txt" 2>&1 || { cat "$A/env_hvenv.txt"; echo "the hvenv does not import this checkout"; return 1; }
  [[ -x "$HV/bin/jevbench" ]] || { echo "no jevbench in the hvenv"; return 1; }
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
  "$EV/bin/pip" freeze > "$A/e14venv_freeze.txt" 2>/dev/null || uv pip freeze --python "$EV" > "$A/e14venv_freeze.txt"
  : > "$A/sha256_checks.txt"
  mkdir -p "$A/inputs"
  aws s3 cp "$B/e5_small/inputs/cal_requests.jsonl.gz" "$W/e27_cal_requests.jsonl.gz" --only-show-errors --region "$REGION" || return 1
  sha_check "$W/e27_cal_requests.jsonl.gz" "$CAL_REQ_SHA" cal_requests || return 1
  aws s3 cp "$B/e12/data/train_mix.jsonl.gz" "$W/e27_e12_train_mix.jsonl.gz" --only-show-errors --region "$REGION" || return 1
  sha_check "$W/e27_e12_train_mix.jsonl.gz" "$E12_MIX_SHA" e12_train_mix || return 1
  [[ -s "$W/base_model_path.txt" ]] || echo "no base_model_path.txt (the latency stage will not serve the 4B)"
  if ! verify_frozen; then
    aws s3 sync "$(plan "d['rows']['frozen']['s3']")" "$(frozen_dir)" --only-show-errors --region "$REGION" || echo "frozen 4B fetch failed"
    verify_frozen || echo "the frozen 4B does not verify (the latency stage will not run the 4B)"
  fi
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s", "gpu": "%s", "mins_left": %s}\n' "$MY_IID" \
    "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" \
    "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" "$(mins_left)" > "$A/host.json"
  cat "$A/sha256_checks.txt" "$A/env_e14venv.txt"
  df -BG / | awk 'NR==2{print "disk free: "$4}'
  sync_out
}
stage_data() {
  cd "$R" || return 1
  mkdir -p "$D"
  PY scripts/train/e27_clef.py build --out "$D" --e12-mix "$W/e27_e12_train_mix.jsonl.gz" > "$A/logs/build.log" 2>&1 \
    || { tail -5 "$A/logs/build.log"; return 1; }
  cp -f "$D/build_e27.json" "$A/build_e27.json"; cp -f "$D/vocab_check.json" "$A/vocab_check.json"
  st "data: $(python3 -c "import json; d=json.load(open('$A/build_e27.json')); print('vocab overlap', d['vocab_check']['overlap_count'], '; S', d['files']['train_s']['n'], 'SP', d['files']['train_sp']['n'], 'records; augmented states', d['augmentation']['augmented_states'], 'of', d['augmentation']['states'])")"
  sync_out
}
train_arm() {  # train_arm ARM
  local arm=$1 rc t0
  cd "$R" || return 1
  stop_port "$PORT"
  [[ -s "$D/train_${arm,,}.jsonl.gz" ]] || { echo "no training file for $arm"; return 1; }
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$D/vocab_check.json'))['ok'] else 1)" || { echo "vocabulary check not ok"; return 1; }
  t0=$(date +%s)
  smi_start "$A/logs/train_smi_$arm.csv"
  EPY scripts/train/e27_clef.py train --arm "$arm" --data "$D" --out "$CKE" --hours-used "$(hours_used)" \
    --v1-out "$A/eval/e27${arm,,}_v1.json" > "$A/logs/train_$arm.log" 2>&1
  rc=$?
  smi_stop
  cp -f "$CKE/$arm/train_record.json" "$A/train_record_$arm.json" 2>/dev/null
  printf '{"arm": "%s", "exit": %s, "minutes": %s}\n' "$arm" "$rc" "$(( ($(date +%s) - t0) / 60 ))" >> "$A/train_runs.jsonl"
  (( rc == 0 )) || { tail -30 "$A/logs/train_$arm.log"; return 1; }
  aws s3 sync "$CKE/$arm/final" "$B/checkpoints/e27_small_shift/$arm/final" --only-show-errors --region "$REGION"
  (cd "$CKE/$arm/final" && sha256sum joint_head.safetensors unsloth_decision_config.json 2>/dev/null) > "$A/ckpt_sha256_$arm.txt"
  st "train $arm: $(python3 -c "import json; d=json.load(open('$A/train_record_$arm.json')); print('records', d['train_records'], 'epochs', [(e['tag'], e['minutes'], e['dev']['accuracy']) for e in d['epochs']], 'cal', d.get('calibration_dev', {}).get('accuracy'), 'peak', d.get('peak_reserved_gb'))" 2>/dev/null)"
  sync_out
}
stage_train_s() { train_arm S; }
stage_train_sp() { train_arm SP; }
eval_arm() {  # eval_arm ARM: L1 (E1), L2 C, L2 shift (main, then surface-undone), through the served arm
  local arm=$1 key="e27${1,,}" name="e27-${1,,}" k f
  cd "$R" || return 1
  serve_arm "$arm" "$name" || return 1
  smi_start "$A/logs/smi_eval_$arm.csv"
  for k in 1 2; do   # two invocations as E1 / E14 (the second resumes any failed request from the cache)
    PY -m jevharness.bench.decision_eval "$CFG/e1_e27.yaml" --out runs --only "$key" > "$A/logs/e1_${key}_$k.log" 2>&1
    st "e1 $key invocation $k exit=$?"
  done
  mkdir -p "$A/s23" "$A/c_shift"
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz; do cp -f "results/stage2/s23/$f" "$A/s23/$f" || return 1; done
  for k in 1 2 3; do   # the replay resumes
    PY scripts/stage2_corpus.py replay --corpus "$A/s23" --name "$key" --s1 "$(spec_of "$name" "$PORT")" \
      --cache "cache/s1_e27_${key}_c.sqlite" > "$A/logs/replay_c_${key}_$k.log" 2>&1
  done
  st "corpus C $key: $(wc -l < "$A/s23/answers_$key.jsonl" 2>/dev/null) answers"
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz questions_nosurf.jsonl.gz; do
    cp -f "results/explore/e21_shift/c/$f" "$A/c_shift/$f" || return 1
  done
  sha_check "$A/c_shift/questions.jsonl.gz" 827a7678f87c586ae2da9e72d74a5a3c7fd0181e6fab5ba8b26639f661283397 e21_questions || return 1
  for v in main nosurf; do
    for k in 1 2; do
      PY scripts/explore_e21_shift.py replay --corpus "$A/c_shift" --slug "$key" --variant "$v" --s1 "$(spec_of "$name" "$PORT")" \
        --cache "cache/s1_e27_${key}_shift_$v.sqlite" --workers 4 > "$A/logs/replay_shift_${key}_${v}_$k.log" 2>&1 && break
    done
    st "shift L2 $key $v: $(PY scripts/explore_e21_shift.py quickc --corpus "$A/c_shift" --slug "$key" --variant "$v" 2>/dev/null)"
  done
  smi_stop
  curl -sf -m 10 "http://127.0.0.1:$PORT/health" > "$A/logs/serve_${key}_health_end.json"
  stop_port "$PORT"
  sync_out
}
stage_eval_s() { eval_arm S; }
stage_eval_sp() { eval_arm SP; }
stage_select() {
  cd "$R" || return 1
  PY "$EX" select --box "$A" --runs runs --out "$A/selected.json" > "$A/logs/select.log" 2>&1 || { cat "$A/logs/select.log"; return 1; }
  st "select: $(cat "$A/logs/select.log" | tail -1)"
  sync_out
}
stage_cal_sel() {  # AgentDecision-0.8B answers the calibration questions (registration (f) 6); no LLM call, $0
  local arm; arm=$(sel_arm)
  [[ -n "$arm" ]] || { echo "no selected arm"; return 1; }
  cd "$R" || return 1
  serve_arm "$arm" e27-sel > "$A/logs/serve_cal_start.log" 2>&1 || { tail -5 "$A/logs/serve_cal_start.log"; return 1; }
  local t0; t0=$(date -u +%FT%TZ)
  PY scripts/e5_cal_replay.py answer --requests "$W/e27_cal_requests.jsonl.gz" --url "http://127.0.0.1:$PORT" \
    --name e27-sel --out "$A/cal/answers_sel_cal.jsonl" --timeout 120 > "$A/logs/cal_sel.log" 2>&1
  local rc=$?
  printf '{"arm": "%s", "window_start": "%s", "window_end": "%s", "exit": %s}\n' "$arm" "$t0" "$(date -u +%FT%TZ)" "$rc" > "$A/cal/window.json"
  st "cal_sel ($arm): exit $rc, $(wc -l < "$A/cal/answers_sel_cal.jsonl" 2>/dev/null) answers"
  stop_port "$PORT"
  sync_out
  return $rc
}
stage_shift_sel() { local a; a=$(sel_arm); [[ -n "$a" ]] || return 1; live_row "shift_${a,,}" "$a" "e27-${a,,}" "smoke_shift_${a,,}" "live_shift_${a,,}"; }
stage_shift_other() { local a; a=$(other_arm); [[ -n "$a" ]] || return 1; live_row "shift_${a,,}" "$a" "e27-${a,,}" "smoke_shift_${a,,}" "live_shift_${a,,}"; }
stage_live_in() { local a; a=$(sel_arm); [[ -n "$a" ]] || return 1; live_row live_in "$a" e27-sel smoke_in live_in; }
fitted_ok() {
  grep -q '^status: development' configs/harness/explore_e27_final.yaml || { echo "the E27 profile has no fitted thresholds (status not development)"; return 1; }
  grep -q 'E27: the fitted thresholds' docs/PREREG_A7_STAGE2.md || { echo "no amendment with E27's fitted thresholds"; return 1; }
  if grep -q 'final_rows: not run' configs/harness/explore_e27_final.yaml; then echo "the fit always defers at the tool choice: the final-harness rows are not run"; return 2; fi
  return 0
}
stage_final_in() {
  local a rc; a=$(sel_arm); [[ -n "$a" ]] || return 1
  cd "$R" || return 1
  fitted_ok; rc=$?; (( rc == 2 )) && { st "final_in: not run (always defers at the tool choice)"; return 0; }; (( rc == 0 )) || return 1
  live_row final_in "$a" e27-sel smoke_final_in final_in
}
stage_final_shift() {
  local a rc; a=$(sel_arm); [[ -n "$a" ]] || return 1
  cd "$R" || return 1
  fitted_ok; rc=$?; (( rc == 2 )) && { st "final_shift: not run (always defers at the tool choice)"; return 0; }; (( rc == 0 )) || return 1
  live_row final_shift "$a" e27-sel smoke_final_shift final_shift "${SHIFT_FINAL_MIN:-60}"
}
stage_wait_prev() {  # wait until every older E27 chain on this host has ended (WAIT_MAX_MIN at most)
  local me t end=$(( $(date +%s) + ${WAIT_MAX_MIN:-420} * 60 ))
  me=$(basename "$0" .sh); me=${me#e27_}
  while (( $(date +%s) < end )); do
    local older=0
    for t in $(ps -eo args | grep -oE '/opt/jev/e27_[0-9]+\.sh box' | grep -oE '[0-9]{6,}' | sort -u); do
      [[ "$t" < "$me" ]] && older=1
    done
    (( older == 0 )) && return 0
    sleep 30
  done
  echo "an older chain is still running after ${WAIT_MAX_MIN:-420} min"; return 1
}
bench_one() {  # bench_one LABEL PORT: harness_final.py bench c = 1 (25 episodes) and c = 4 (100), E14 (f) 4
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
stage_latency() {  # lowest priority (registration (f) 9): AgentDecision-0.8B, then the frozen 4B, one model at a time
  local a; a=$(sel_arm); [[ -n "$a" ]] || return 1
  cd "$R" || return 1
  if (( $(mins_left) < ${LAT_MIN:-45} )); then st "latency: not run ($(mins_left) min left)"; return 0; fi
  stop_port "$PORT"; stop_port "$PORT4B"
  serve_arm "$a" e27-sel || return 1
  bench_one e27sel "$PORT" || return 1
  stop_port "$PORT"
  serve_4b || { st "latency: the frozen 4B did not serve"; return 1; }
  bench_one 4b "$PORT4B" || return 1
  stop_port "$PORT4B"
  sync_out
}
stage_final() {
  stop_port "$PORT"; stop_port "$PORT4B"
  nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  printf '{"final_utc": "%s", "instance_id": "%s", "instance_type": "%s", "hours_used": %s}\n' "$(date -u +%FT%TZ)" "$MY_IID" \
    "$(imds instance-type)" "$(hours_used)" > "$A/host_final.json"
  st "final: used $(hours_used) h; $(cd "$R" && PY "$EX" spend --runs runs --log "$PROBE_LOG" --short 2>/dev/null)"
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; sudo shutdown -h +2 "jev-e27-08b: done"; fi
}

st "chain start: $STAGES (host $MY_IID, used $(hours_used) h)"
for s in $STAGES; do
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $s"; continue; fi
  t0=$(date +%s)
  cd "$R" 2>/dev/null || cd "$W"
  ( "stage_$s" ) > "$STATUS_DIR/stage_$s.log" 2>&1
  rc=$?
  [[ -d "$A/logs" ]] && cp -f "$STATUS_DIR/stage_$s.log" "$A/logs/stage_$s.log"
  echo "{\"stage\": \"$s\", \"exit\": $rc, \"minutes\": $(( ($(date +%s) - t0) / 60 )), \"end_utc\": \"$(date -u +%FT%TZ)\"}" >> "$STATUS_DIR/stage_times.jsonl"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/stage_times.jsonl" "$A/stage_times.jsonl"
  st "stage $s exit=$rc ($(( ($(date +%s) - t0) / 60 )) min): $(tail -2 "$STATUS_DIR/stage_$s.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  if (( rc != 0 )) && [[ "$s" == rearm || "$s" == code || "$s" == setup || "$s" == data || "$s" == wait_prev ]]; then
    st "chain stopped at $s"
    [[ " $STAGES " == *" final "* ]] && stage_final
    exit 1
  fi
done
sync_out
st "chain end [$STAGES]"
