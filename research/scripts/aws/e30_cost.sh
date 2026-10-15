#!/usr/bin/env bash
# Exploratory row E30, every open decision model at its cheapest common deployment (docs/PREREG_A7_STAGE2.md,
# Deviations 2026-10-09 "E30: every open decision model at its cheapest common deployment"; descriptive, post hoc
# re-measurement, outside every Holm table). One NEW g6.xlarge (1x L4 24 GB) launched by this driver from E3's clone image
# (tag Name=jev-e30-cost), SSM only. vLLM 0.31.0 in process (scripts/e30_bench.py time-vllm), BF16 and FP8 per model, E9's
# 3,991 L1 decisions in one fixed shuffled order, saturated at C = 64 / 256, C = 1 latency on 1,000 decisions;
# StartLux-27B as the vendor's Q4_K_M GGUF on llama.cpp; E27's scoring head on our stack over the Jev wire format.
# No API key anywhere: Hugging Face and S3 through the host's role.
#
#   LOCAL:  bash scripts/aws/e30_cost.sh launch [itype ...]     # one new host (refuses if a jev-e30-cost host exists);
#                                                                # default g6.xlarge, then g6.2xlarge (same GPU)
#           E30_IID=i-... bash scripts/aws/e30_cost.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages (in the order sent; a failed rearm / code / setup stops the chain; `final` always runs when sent):
#     rearm      idle stop (IDLE_MIN) + power-off guard (GUARD_MIN from the first boot), host record
#     code       the committed branch in /opt/jev/e30
#     setup      code; the vendor's package at the pinned commit; vLLM 0.31.0 in its own venv; the fixed order
#     wait_prev  wait until every older E30 chain on this host has ended
#     smoke      the engine path on 300 decisions of StartLux-0.8B BF16 (the 7 wide choices first); never a result
#     e12 rl e12c e28                         our trained models: merge (bf16), serve_jev's prompts, BF16 + FP8
#     sl08b sl2b sl4b sl9b                    StartLux (vendor read-out) at E9's revisions, BF16 + FP8
#     intern                                  Intern-Decision-4B (vendored compiler, one read per field), BF16 + FP8
#     sl27b sl27b16                           StartLux-27B Q4_K_M on llama.cpp: P = 8 with C = 1 latency; P = 16
#     e27sp                                   E27 arm SP's head behind serve_jev --backend clef (Unsloth venv)
#     openjevhf openjevhf16                   OpenJev-27B: the publisher's Q4_K_M GGUF on llama.cpp, the helper's read-out
#                                             (E30 amendment): P = 8 with C = 1 latency; P = 16
#     decider jevk5                           E15's decider-4b and JevK5 v0.2 on vLLM, their own prompt code (amendment)
#     jevk5m                                  JevK5 at E15's commit c4f7fdb3 (v0.3; amendment "JevK5 at the weights E15 loaded")
#     clefflash                               E15's Clef-Flash 9B through the card's systemone() (scoring head; amendment)
#     setup_min openjev                       E30_TAG=jev-e30-cost-l40s E30_S3=e30_cost_l40s: OpenJev's image on a g6e.xlarge
#     final                                   stop servers, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e30_cost/box -> s3://$BUCKET/e30_cost/box (timing files with every answer: runs/*.json.gz).
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$|_TOKEN$'); do unset "$v"; done   # no key in this process tree
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E30_IID=${E30_IID:-}
FORBIDDEN_FILE=configs/explore/e30_cost/forbidden_iids.txt
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=${E30_TAG:-jev-e30-cost}          # jev-e30-cost-l40s: the g6e.xlarge of amendment "E15's open baselines" (c) 5
BUNDLE=e30.bundle
S3OUT="$B/${E30_S3:-e30_cost}"        # e30_cost_l40s for the g6e.xlarge
GUARD_MIN=${GUARD_MIN:-355}          # 6 host-hours (registration (h)): the guard 5 h 55 min after the first boot
IDLE_MIN=${IDLE_MIN:-30}
POWEROFF=${POWEROFF:-1}
MIN_LEFT=${MIN_LEFT:-20}             # a model stage starts only with at least this many guard minutes left
VLLM_SPEC="vllm==0.31.0"
PKG_COMMIT=a9101e918e097b26fea5ead9ed893b823e875e46
LLAMA_IMG=ghcr.io/ggml-org/llama.cpp@sha256:d87b02aafa74797a2d599b281c50468704bd0bcdc57bfcf31b570d593f1d285b   # the logged 27B row's
# Hugging Face revisions (registration (b)): E9's rows (E2 ladder manifests, Stage 2's manifest), E28's base, Intern, the 27B GGUF
REV_08B=bd4f76a600e23227547fee7bfc1825e12a32764c
REV_2B=3e2e456409a7fe44be69eee231566f7830872ea2
REV_9B=342aad64840be8eee314455428a5a775f18ef547
REV_27B_Q4=ce98e33e03fe238638b9ec9df3071231bc933a9f
REV_INTERN=0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd
REV_OPENJEV=ac97900fd034fdd7e7e536f3d4c21b836cae0750
REV_OPENJEV_GGUF=24a4776974ef7b6f906065fc33e5ad73275d3d1b
REV_DECIDER=eb5fbdfc9448473ec25e399882912863afbdb70e
REV_JEVK5=v0.2                     # the Hub tag E15's JevK5 v0.2 row names (the commit is recorded)
JEVK5_PKG_TAG=v0.2.2
REV_JEVK5_MAIN=c4f7fdb3aeab5582336406e78d3bef11bf98833d   # amendment "JevK5 at the weights E15 loaded": E15's revision=main (v0.3)
REV_CLEF=fde727a287004204b7518dcc983fe64379776712
OPENJEV_IMG=razorback16/openjev@sha256:53741f5005c81e4d19a8a8d66e717213c173bb11a46aac1147eea1dc9244b854   # E15's digest
OPENJEV_WEIGHTS=RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
forbidden_local() {
  local root; root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  grep -oE '^i-0[0-9a-f]{16}' "$root/$FORBIDDEN_FILE" | tr '\n' ' '
}
iid_ok() {
  [[ "$E30_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E30_IID is not set to an instance id"; return 1; }
  local forb; forb=$(forbidden_local)
  [[ -n "$forb" ]] || { echo "no forbidden list ($FORBIDDEN_FILE)"; return 1; }
  [[ " $forb " == *" $E30_IID "* ]] && { echo "refused: $E30_IID is not the E30 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E30_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E30_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "%s: power-off guard (host-hour cap from launch)"\n' "$GUARD_MIN" "$TAG" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g6.xlarge g6.2xlarge)
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/e30_cost.sh" > "$tmp/e30_cost.sh"   # the committed driver
  awsl s3 cp "$tmp/e30_cost.sh" "$B/code/e30_cost.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e30_cost.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e30_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e30_cost.sh \$f --only-show-errors --region $REGION"
    printf 'export GUARD_MIN=%q IDLE_MIN=%q POWEROFF=%q MIN_LEFT=%q E30_IID=%q FORBIDDEN_IIDS=%q E30_TAG=%q E30_S3=%q\n' \
      "$GUARD_MIN" "$IDLE_MIN" "$POWEROFF" "$MIN_LEFT" "$E30_IID" "$(forbidden_local)" "$TAG" "${E30_S3:-e30_cost}"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e30_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e30_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E30_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e30_driver.sh /tmp/e30_driver.sh --only-show-errors && bash /tmp/e30_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E30_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype || "$MODE" == state ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E30_IID" --output text \
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
if [[ -z "${FORBIDDEN_IIDS:-}" || " ${FORBIDDEN_IIDS} " == *" $MY_IID "* || ( -n "$E30_IID" && "$MY_IID" != "$E30_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E30 host ${E30_IID:-?} (or no forbidden list)"; exit 1
fi
R=$W/e30
A=$R/results/explore/e30_cost/box
STATUS_DIR=$W/e30_status
HV=$W/hvenv                 # harness venv (prompts, merges of the Qwen3-4B models, the wire client)
VV=$W/v2venv                # A7.19 (f)'s v2 venv (+ peft): merges on the StartLux base (as E12c)
LV=$W/e30vllm               # vLLM 0.31.0
EV=$W/e14venv               # Unsloth 2026.10.2: serve_jev --backend clef (as E14 / E27)
DV=$W/e30_dmvenv            # E15's card-function venv (Clef-Flash's systemone)
SLB=/opt/startlux
MD=$SLB/models
PKG=$SLB/startlux-decision  # the vendor's package (pinned commit)
M=$W/e30_models             # merged checkpoints and new downloads
P=$W/e30_prompts
CK=$W/ck                    # the frozen 4B: $CK/<plan dir>
PLAN=results/stage2/rl_eval_plan.json
BENCH=scripts/e30_bench.py
FAST=scripts/e12_fast_engine.py
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$STATUS_DIR"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  aws s3 cp "$STATUS_DIR/status.txt" "$S3OUT/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION"
  true
}
PY() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }
PV() { PYTHONPATH="$R/src" "$VV/bin/python" "$@"; }
LPY() { "$LV/bin/python" "$@"; }
first_boot() { cat "$W/e30_first_boot" 2>/dev/null || date -d "$(uptime -s)" +%s; }
hours_used() { python3 -c "import time; print(round((time.time() - $(first_boot)) / 3600, 3))"; }
mins_left() { echo $(( GUARD_MIN - ( $(date +%s) - $(first_boot) ) / 60 )); }
qbase() { cat "$W/base_model_path.txt"; }                  # Qwen3-4B (AgentDecision-4B's and the frozen 4B's base)
plan() { (cd "$R" && PY -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"); }
frozen_dir() { echo "$CK/$(plan "d['rows']['frozen']['dir']")"; }
prewarm() {  # read files in 64 MB chunks, 16 at a time (a new EBS volume loads lazily from its snapshot)
  local d
  for d in "$@"; do
    find -L "$d" -type f -size +1M 2>/dev/null | while read -r f; do
      local s n i; s=$(stat -L -c %s "$f"); n=$(( (s + 67108863) / 67108864 ))
      for i in $(seq 0 $((n - 1))); do echo "$f $i"; done
    done
  done | xargs -P 16 -n 2 sh -c 'dd if="$0" of=/dev/null bs=64M skip=$1 count=1 status=none'
}
hfdl() {  # hfdl REPO REV DIR [allow patterns...]: a pinned snapshot; the revision and the weights' sha256 are recorded
  local repo=$1 rev=$2 dir=$3; shift 3
  mkdir -p "$dir"
  PY - "$repo" "$rev" "$dir" "$@" > "$A/logs/hf_$(basename "$dir").log" 2>&1 <<'EOF' || { tail -5 "$A/logs/hf_$(basename "$dir").log"; return 1; }
import sys
from huggingface_hub import HfApi, snapshot_download
repo, rev, d, pats = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
snapshot_download(repo, revision=rev, local_dir=d, allow_patterns=pats or None)
info = HfApi().model_info(repo, revision=rev)
print("sha", info.sha)
EOF
  if [[ "$rev" =~ ^[0-9a-f]{40}$ ]]; then
    grep -q "^sha $rev" "$A/logs/hf_$(basename "$dir").log" || { echo "the downloaded revision of $repo is not $rev"; return 1; }
  fi
  printf '{"repo": "%s", "revision": "%s", "dir": "%s", "at": "%s"}\n' "$repo" "$rev" "$dir" "$(date -u +%FT%TZ)" >> "$A/downloads.jsonl"
  (cd "$dir" && find . -maxdepth 1 \( -name '*.safetensors' -o -name '*.gguf' -o -name config.json \) -print0 | sort -z \
    | xargs -0 -r sha256sum) > "$A/sha256_$(basename "$dir").txt" 2>/dev/null
  return 0
}
smi_start() { nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 1000 > "$1" 2>/dev/null & SMI=$!; }
smi_stop() { [[ -n "${SMI:-}" ]] && kill "$SMI" 2>/dev/null; SMI=""; }
enough_time() {  # enough_time LABEL: refuse a model stage with fewer than MIN_LEFT guard minutes left
  local left; left=$(mins_left)
  if (( left < MIN_LEFT )); then st "$1: not run ($left guard minutes left < $MIN_LEFT)"; return 1; fi
  return 0
}

# ------------------------------------------------------------------------------------------------ measurement helpers
prompts() {  # prompts SLUG FAMILY MODEL_DIR [CODE_DIR]: the decision prompts for one tokenizer (harness venv)
  local slug=$1 fam=$2 dir=$3 code=${4:-$PKG}
  [[ -s "$P/$slug.jsonl.gz" ]] && return 0
  mkdir -p "$P"
  PY "$BENCH" prompts --family "$fam" --model "$dir" --vendor "$code" --out "$P/$slug.jsonl.gz" > "$A/prompts/$slug.json" 2> "$A/logs/prompts_$slug.log" \
    || { tail -5 "$A/logs/prompts_$slug.log"; return 1; }
  cat "$A/prompts/$slug.json"
}
time_vllm() {  # time_vllm SLUG DIR PRECISION [CODE_DIR]: one vLLM process, one precision (registration (c), (e), (f))
  local slug=$1 dir=$2 prec=$3 code=${4:-$PKG} out="$A/runs/${1}_${3}.json.gz"
  [[ -s "$out" ]] && { st "$slug $prec: already measured"; return 0; }
  enough_time "$slug $prec" || return 1
  smi_start "$A/smi/${slug}_${prec}.csv"
  timeout 45m "$LV/bin/python" "$R/$BENCH" time-vllm --model "$dir" --prompts "$P/$slug.jsonl.gz" --order "$A/order.json" \
    --precision "$prec" --vendor "$code" --label "$slug" --out "$out" > "$A/logs/time_${slug}_${prec}.out" 2> "$A/logs/time_${slug}_${prec}.err"
  local rc=$?
  smi_stop
  st "$slug $prec: rc=$rc $(python3 - "$out" <<'EOF' 2>/dev/null
import gzip, json, sys
d = json.load(gzip.open(sys.argv[1], "rt"))
if d.get("state") != "ok":
    print(d.get("state"), (d.get("errors") or [""])[-1][:200])
else:
    s = d["saturated"]
    print(f"{s['dps']:.1f} dec/s at C={s['c']}; c1 p50 {d['c1']['p50_ms']:.0f} ms; load {d['load_s']:.0f} s; args {d.get('extra_args')}")
EOF
)"
  if (( rc != 0 )) && [[ ! -s "$out" ]]; then
    tail -c 1500 "$A/logs/time_${slug}_${prec}.err"
    printf '{"state": "failed", "rc": %s, "slug": "%s", "precision": "%s"}' "$rc" "$slug" "$prec" | gzip > "$out"
  fi
  sync_out
  return 0
}
both() {  # both SLUG DIR [CODE_DIR]: BF16 then FP8
  time_vllm "$1" "$2" bf16 "${3:-$PKG}"
  time_vllm "$1" "$2" fp8 "${3:-$PKG}"
}
merge_q() {  # merge_q OUT BASE ADAPTER VENV(PY|PV): LoRA merged into its base in bf16 (E12c (h)'s merge)
  local out=$1 base=$2 ad=$3 venv=$4
  [[ -s "$out/config.json" ]] && return 0
  $venv "$R/$FAST" merge --base "$base" --adapter "$ad" --out "$out" > "$A/logs/merge_$(basename "$out").log" 2>&1 \
    || { tail -5 "$A/logs/merge_$(basename "$out").log"; return 1; }
  (cd "$out" && sha256sum ./*.safetensors config.json) > "$A/sha256_merged_$(basename "$out").txt" 2>/dev/null
  return 0
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

# ------------------------------------------------------------------------------------------------ stages
stage_rearm() {
  [[ -s "$W/e30_first_boot" ]] || date -d "$(uptime -s)" +%s > "$W/e30_first_boot"
  local left; left=$(mins_left)
  (( left > 0 )) || { echo "guard already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$left" "jev-e30-cost: power-off guard (host-hour cap)"
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E30 chain / bench / server process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e30_[0-9]|e30_bench|e12_fast_engine|serve_jev|llama-server|snapshot_download|uv pip|pip install|docker pull' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e30-cost: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $left min left, idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU; $(free -g | awk '/Mem:/{print $2}') GiB RAM; $(df -BG / | awk 'NR==2{print $4}') free; used $(hours_used) h"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
}
stage_code() {
  local myb="$W/e30.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A/logs" "$A/runs" "$A/prompts" "$A/smi"
  st "code at $(git log --oneline -1 | cut -c1-72)"
  grep -q 'E30: every open decision model at its cheapest common deployment' docs/PREREG_A7_STAGE2.md \
    || { echo "no Deviations entry for E30"; return 1; }
}
stage_setup() {
  stage_code || return 1
  ( prewarm "$HV/lib" "$VV/lib" ) &
  local pw=$!
  PY -c "import jevharness, torch, peft, sys; p = jevharness.__file__; print(p, torch.__version__, peft.__version__); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_hvenv.txt" 2>&1 || { cat "$A/env_hvenv.txt"; echo "the hvenv does not import this checkout"; return 1; }
  command -v uv > /dev/null || bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  "$VV/bin/python" -c "import peft" 2>/dev/null || timeout 15m uv pip install -q --python "$VV" "peft==0.21.2" > "$A/logs/v2venv_peft.log" 2>&1 \
    || { tail -5 "$A/logs/v2venv_peft.log"; return 1; }
  # the vendor's package at the pinned commit (registration (d) 2)
  if [[ ! -d "$PKG/.git" ]]; then sudo mkdir -p "$SLB" && sudo chown "$(id -un)" "$SLB" && git clone -q https://github.com/StartLuxLabs/startlux-decision "$PKG"; fi
  if [[ "$(git -C "$PKG" rev-parse HEAD)" != "$PKG_COMMIT" ]]; then
    git -C "$PKG" fetch -q --depth 1 origin "$PKG_COMMIT" && git -C "$PKG" checkout -q "$PKG_COMMIT" || { echo "vendor package not at $PKG_COMMIT"; return 1; }
  fi
  git -C "$PKG" rev-parse HEAD > "$A/vendor_commit.txt"
  [[ "$(cat "$A/vendor_commit.txt")" == "$PKG_COMMIT" ]] || return 1
  # vLLM 0.31.0 in its own venv (registration (c))
  if [[ ! -x "$LV/bin/python" ]] || ! "$LV/bin/python" -c "import vllm" > /dev/null 2>&1; then
    uv venv -q -p 3.11 "$LV" && timeout 40m uv pip install -q --python "$LV" "$VLLM_SPEC" > "$A/logs/vllm_install.log" 2>&1 \
      || { tail -5 "$A/logs/vllm_install.log"; echo "vLLM install failed"; return 1; }
  fi
  "$LV/bin/python" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)" \
    > "$A/vllm_version.txt" 2>&1 || { cat "$A/vllm_version.txt"; return 1; }
  uv pip freeze --python "$LV" > "$A/vllm_freeze.txt" 2>/dev/null
  # the fixed order (registration (f))
  PY "$BENCH" order --out "$A/order.json" > "$A/order_info.json" 2> "$A/logs/order.log" || { tail -5 "$A/logs/order.log"; return 1; }
  wait "$pw"
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s", "gpu": "%s", "vcpu": %s, "ram_gib": %s, "mins_left": %s}\n' \
    "$MY_IID" "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" \
    "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" "$(nproc)" \
    "$(free -g | awk '/Mem:/{print $2}')" "$(mins_left)" > "$A/host.json"
  cat "$A/vllm_version.txt" "$A/order_info.json" "$A/host.json"
  df -BG / | awk 'NR==2{print "disk free: "$4}'
  sync_out
}
older_running() {  # an E30 chain started before this one (its /opt/jev/e30_<ns>.sh has a smaller ns), or a bench outside one
  # (no pipeline status is tested: under pipefail an early-exiting grep or a false last test would decide it)
  local mine ns hit="" lines orphans
  mine=$(basename "$0" | sed -E 's/^e30_([0-9]+)\.sh$/\1/')
  lines=$(ps -eo args | grep -oE "e30_[0-9]+\.sh box" | sed -E 's/^e30_([0-9]+)\.sh box$/\1/' | sort -u || true)
  for ns in $lines; do
    if [[ -n "$mine" && "$ns" -lt "$mine" ]]; then hit=1; fi
  done
  orphans=$(ps -eo ppid,args | grep -E "e30_bench\.py time-" | grep -v grep | awk '$1 == 1' || true)
  [[ -n "$hit" || -n "$orphans" ]]
}
stage_wait_prev() {  # wait while an older E30 chain (one /opt/jev/e30_<ns>.sh box) or an orphaned bench is still running
  local n=0
  while older_running; do
    (( n % 10 == 0 )) && st "wait_prev: an older E30 chain or bench is still running"
    n=$(( n + 1 )); sleep 30
  done
  pkill -f "nvidia-smi --query-gpu=timestamp" 2>/dev/null       # an orphaned GPU logger of an older chain
  sudo docker rm -f e30llama > /dev/null 2>&1                   # an older chain's llama-server
  sync_out
  st "wait_prev: no older chain or bench"
}
stage_smoke() {  # the engine path on 300 decisions (StartLux-0.8B BF16; the order's 7 wide choices first): never a result
  enough_time smoke || return 0
  hfdl startlux-models/StartLux-Decision-0.8B "$REV_08B" "$(sl_dir 0.8B)" || return 1
  prompts sl0.8b startlux "$(sl_dir 0.8B)" || return 1
  mkdir -p "$A/smoke"
  python3 - "$A/order.json" "$P/sl0.8b.jsonl.gz" "$A/smoke/order_smoke.json" <<'EOF' || return 1
import gzip, json, sys
o = json.load(open(sys.argv[1]))
wide = [json.loads(l)["id"] for l in gzip.open(sys.argv[2], "rt") if '"kind": "wide"' in l]
o["order"] = wide + [i for i in o["order"] if i not in set(wide)]
json.dump(o, open(sys.argv[3], "w"))
EOF
  timeout 20m "$LV/bin/python" "$R/$BENCH" time-vllm --model "$(sl_dir 0.8B)" --prompts "$P/sl0.8b.jsonl.gz" \
    --order "$A/smoke/order_smoke.json" --precision bf16 --vendor "$PKG" --label smoke --limit 300 \
    --out "$A/smoke/sl0.8b_bf16_smoke.json.gz" > "$A/logs/smoke.out" 2> "$A/logs/smoke.err"
  local rc=$?
  st "smoke: rc=$rc $(tail -c 600 "$A/logs/smoke.out" | tr '\n' ' ')"
  (( rc == 0 )) || tail -c 3000 "$A/logs/smoke.err"
  sync_out
  return $rc
}
stage_e12() {  # AgentDecision-4B (E12's checkpoint) merged into Qwen3-4B
  enough_time e12 || return 0
  mkdir -p "$M"
  aws s3 sync "$B/checkpoints/e12_mixed/" "$M/ad_e12/" --only-show-errors --region "$REGION" --exclude 'checkpoint-*' || return 1
  merge_q "$M/e12" "$(qbase)" "$M/ad_e12" PY || return 1
  prompts e12 code "$M/e12" || return 1
  both e12 "$M/e12"
}
stage_rl() {  # the frozen GRPO 4B (freeze record, sha256-verified) merged into Qwen3-4B
  enough_time rl || return 0
  local fd; fd=$(frozen_dir)
  [[ -s "$fd/adapter_model.safetensors" ]] || aws s3 sync "$(plan "d['rows']['frozen']['s3']")" "$fd/" --only-show-errors --region "$REGION"
  verify_frozen || return 1
  merge_q "$M/rl" "$(qbase)" "$fd" PY || return 1
  [[ -s "$P/e12.jsonl.gz" ]] && { mkdir -p "$P"; cp -f "$P/e12.jsonl.gz" "$P/rl.jsonl.gz"; cp -f "$A/prompts/e12.json" "$A/prompts/rl.json"; }   # same tokenizer
  prompts rl code "$M/rl" || return 1
  both rl "$M/rl"
  rm -rf "$M/rl"
}
sl_dir() { echo "$MD/StartLux-Decision-$1"; }
stage_sl08b() {
  enough_time sl0.8b || return 0
  hfdl startlux-models/StartLux-Decision-0.8B "$REV_08B" "$(sl_dir 0.8B)" || return 1
  prompts sl0.8b startlux "$(sl_dir 0.8B)" || return 1
  both sl0.8b "$(sl_dir 0.8B)"
}
stage_sl2b() {
  enough_time sl2b || return 0
  hfdl startlux-models/StartLux-Decision-2B "$REV_2B" "$(sl_dir 2B)" || return 1
  prompts sl2b startlux "$(sl_dir 2B)" || return 1
  both sl2b "$(sl_dir 2B)"
  rm -rf "$(sl_dir 2B)"
}
stage_sl4b() {  # the image's StartLux-4B (E12c's base); sha256 against E12c's record
  enough_time sl4b || return 0
  (cd "$(sl_dir 4B)" && sha256sum ./*.safetensors config.json) > "$A/sha256_StartLux-Decision-4B.txt"
  diff <(sort "$A/sha256_StartLux-Decision-4B.txt") <(sort "$R/results/explore/e12/box_c/startlux_4b_sha256.txt") > /dev/null \
    && echo "StartLux-4B sha256 = E12c's base" || echo "StartLux-4B sha256 differs from E12c's record (stated)"
  prompts sl4b startlux "$(sl_dir 4B)" || return 1
  both sl4b "$(sl_dir 4B)"
}
stage_sl9b() {
  enough_time sl9b || return 0
  hfdl startlux-models/StartLux-Decision-9B "$REV_9B" "$(sl_dir 9B)" || return 1
  prompts sl9b startlux "$(sl_dir 9B)" || return 1
  both sl9b "$(sl_dir 9B)"
  rm -rf "$(sl_dir 9B)"
}
stage_e12c() {  # StartLux-4B + our recipe (E12c's selected adapter), merged (v2 venv, as E12c (h))
  enough_time e12c || return 0
  aws s3 sync "$B/checkpoints/e12c_sl4b/selected/" "$M/ad_e12c/" --only-show-errors --region "$REGION" || return 1
  merge_q "$M/e12c" "$(sl_dir 4B)" "$M/ad_e12c" PV || return 1
  prompts e12c code "$M/e12c" || return 1
  both e12c "$M/e12c"
  rm -rf "$M/e12c"
}
stage_e28() {  # E28's StartLux-0.8B (mixed): only if its selected adapter is on S3 (E28's host and files untouched)
  enough_time e28 || return 0
  aws s3 ls "$B/checkpoints/e28_sl08b/selected/adapter_model.safetensors" > /dev/null 2>&1 \
    || { st "e28: no selected adapter on S3 (not run)"; return 0; }
  aws s3 sync "$B/checkpoints/e28_sl08b/selected/" "$M/ad_e28/" --only-show-errors --region "$REGION" || return 1
  (cd "$M/ad_e28" && sha256sum adapter_model.safetensors adapter_config.json) > "$A/sha256_ad_e28.txt"
  hfdl startlux-models/StartLux-Decision-0.8B "$REV_08B" "$(sl_dir 0.8B)" || return 1
  merge_q "$M/e28" "$(sl_dir 0.8B)" "$M/ad_e28" PV || return 1
  prompts e28 code "$M/e28" || return 1
  both e28 "$M/e28"
}
stage_intern() {
  enough_time intern || return 0
  hfdl internlm/Intern-Decision-4B "$REV_INTERN" "$M/Intern-Decision-4B" || return 1
  prompts intern intern "$M/Intern-Decision-4B" || return 1
  both intern "$M/Intern-Decision-4B"
  rm -rf "$M/Intern-Decision-4B"
}
docker_ready() {  # docker with the NVIDIA runtime (as E21 / E24)
  if ! command -v docker > /dev/null; then
    curl -fsSL https://get.docker.com | sudo sh > "$A/logs/docker_install.log" 2>&1 || { echo "docker install failed"; return 1; }
  fi
  if ! sudo docker info 2>/dev/null | grep -qi nvidia; then
    if ! command -v nvidia-ctk > /dev/null; then
      (curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
          | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
          | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null \
        && sudo apt-get update -q && sudo apt-get install -y -q nvidia-container-toolkit) > "$A/logs/nvctk_install.log" 2>&1 \
        || { echo "nvidia-container-toolkit install failed"; return 1; }
    fi
    sudo nvidia-ctk runtime configure --runtime=docker > /dev/null 2>&1 && sudo systemctl restart docker
  fi
}
setup_llama() {  # llama.cpp's CUDA server image at the digest of the logged 27B row (as E21 / E24)
  docker_ready || return 1
  timeout 30m sudo docker pull -q "$LLAMA_IMG" > "$A/logs/llama_pull.log" 2>&1 || { tail -5 "$A/logs/llama_pull.log"; return 1; }
  echo "$LLAMA_IMG" > "$A/llama_image.txt"
}
llama_up() {  # llama_up GGUF SLOTS PORT: llama-server with SLOTS parallel slots, 4,096 tokens each, all layers on the GPU
  sudo docker rm -f e30llama > /dev/null 2>&1
  sudo docker run -d --name e30llama --init --gpus all --network host -v "$MD:$MD:ro" "$LLAMA_IMG" \
    -m "$1" -ngl 99 -c $(( $2 * 4096 )) --parallel "$2" --port "$3" > "$A/logs/llama_cid_$2.txt" 2>&1 || return 1
  for _ in $(seq 1 120); do
    curl -sf -m 5 "http://127.0.0.1:$3/health" > /dev/null 2>&1 && { echo "llama-server up (P=$2)"; return 0; }
    sudo docker ps -q -f name=e30llama | grep -q . || { sudo docker logs e30llama 2>&1 | tail -20; return 1; }
    sleep 5
  done
  return 1
}
llama_down() { sudo docker logs e30llama > "$A/logs/llama_$1.log" 2>&1; sudo docker rm -f e30llama > /dev/null 2>&1; true; }
llama_rows() {  # llama_rows SLUG GGUF TOKDIR P...: one llama-server run per P (registration (f), llama.cpp rule; C = 1 at P = 8)
  local slug=$1 gguf=$2 tok=$3 port=8191 s; shift 3
  for s in "$@"; do
    local out="$A/runs/${slug}_q4_k_m_p$s.json.gz"
    [[ -s "$out" ]] && continue
    enough_time "$slug P=$s" || break
    llama_up "$gguf" "$s" "$port" || { llama_down "${slug}_p$s"; st "$slug: llama-server did not start (P=$s)"; return 1; }
    smi_start "$A/smi/${slug}_p$s.csv"
    local extra=(); [[ "$s" != 8 ]] && extra=(--skip-c1)
    timeout 100m env PYTHONPATH="$R/src" "$HV/bin/python" "$R/$BENCH" time-llama --url "http://127.0.0.1:$port" --model-dir "$tok" \
      --prompts "$P/$slug.jsonl.gz" --order "$A/order.json" --slots "$s" --vendor "$PKG" --label "$slug" "${extra[@]}" --out "$out" \
      > "$A/logs/time_${slug}_p$s.out" 2> "$A/logs/time_${slug}_p$s.err"
    local rc=$?
    smi_stop
    llama_down "${slug}_p$s"
    st "$slug P=$s: rc=$rc $(tail -c 300 "$A/logs/time_${slug}_p$s.out" | tr '\n' ' ')"
    sync_out
  done
}
sl27b_prep() {
  local d="$MD/StartLux-Decision-27B-Q4_K_M"
  hfdl startlux-models/StartLux-Decision-27B-Q4_K_M-GGUF "$REV_27B_Q4" "$d" || return 1
  setup_llama || return 1
  prompts sl27b startlux "$d" || return 1
}
stage_sl27b() {  # the vendor's Q4_K_M GGUF on llama.cpp (registration (c), (e)): P = 8 with the C = 1 latency
  enough_time sl27b || return 0
  sl27b_prep || return 1
  llama_rows sl27b "$MD/StartLux-Decision-27B-Q4_K_M/StartLux-Decision-27B-Q4_K_M.gguf" "$MD/StartLux-Decision-27B-Q4_K_M" 8
}
stage_sl27b16() {  # the same at P = 16 (the saturation check)
  enough_time sl27b16 || return 0
  sl27b_prep || return 1
  llama_rows sl27b "$MD/StartLux-Decision-27B-Q4_K_M/StartLux-Decision-27B-Q4_K_M.gguf" "$MD/StartLux-Decision-27B-Q4_K_M" 16
}
stage_e27sp() {  # E27 arm SP's head behind our stack (serve_jev --backend clef, coalescing, warmed up) on :8083
  enough_time e27sp || return 0
  local port=8083 name=e27-sp ck="$M/e27_SP"
  aws s3 sync "$B/checkpoints/e27_small_shift/SP/final/" "$ck/" --only-show-errors --region "$REGION" || return 1
  (cd "$ck" && sha256sum ./*.safetensors) > "$A/sha256_e27_SP.txt"
  if [[ ! -x "$EV/bin/python" ]] || ! "$EV/bin/python" -c "import unsloth" > /dev/null 2>&1; then
    uv venv -q -p 3.11 "$EV" || return 1
    timeout 45m uv pip install -q --python "$EV" "unsloth==2026.10.2" fastapi uvicorn httpx pydantic pyyaml tqdm \
      python-dotenv openai > "$A/logs/e14venv_install.log" 2>&1 || { tail -20 "$A/logs/e14venv_install.log"; return 1; }
    timeout 20m uv pip install -q --python "$EV" flash-linear-attention >> "$A/logs/e14venv_install.log" 2>&1 \
      || echo "flash-linear-attention not installed (the torch path is used)" | tee -a "$A/logs/e14venv_install.log"
  fi
  uv pip freeze --python "$EV" > "$A/e14venv_freeze.txt" 2>/dev/null
  echo "serve_jev --backend clef --model $ck --name $name --port $port --clef-coalesce --clef-warmup 3" > "$A/logs/serve_e27sp.cmd"
  setsid nohup env PYTHONPATH="$R/src" "$EV/bin/python" -m jevharness.training.serve_jev --backend clef --model "$ck" --name "$name" \
    --port "$port" --clef-coalesce --clef-warmup 3 > "$A/logs/serve_e27sp.out" 2>&1 < /dev/null &
  local sp=$!
  for _ in $(seq 1 120); do curl -sf -m 5 "http://127.0.0.1:$port/health" > /dev/null 2>&1 && break
    kill -0 "$sp" 2>/dev/null || { tail -30 "$A/logs/serve_e27sp.out"; return 1; }; sleep 5; done
  curl -sf "http://127.0.0.1:$port/v1/models" > "$A/serve_e27sp_models.json"
  smi_start "$A/smi/e27sp.csv"
  timeout 60m env PYTHONPATH="$R/src" "$HV/bin/python" "$R/$BENCH" time-wire --url "http://127.0.0.1:$port" --model "$name" \
    --order "$A/order.json" --label e27sp --out "$A/runs/e27sp_bf16.json.gz" > "$A/logs/time_e27sp.out" 2> "$A/logs/time_e27sp.err"
  local rc=$?
  smi_stop
  kill -- -"$sp" 2>/dev/null; kill "$sp" 2>/dev/null; sleep 5; kill -9 "$sp" 2>/dev/null
  st "e27sp: rc=$rc $(tail -c 300 "$A/logs/time_e27sp.out" | tr '\n' ' ')"
  sync_out
}
openjevhf_prep() {
  local d="$MD/OpenJev-27B-Q4_K_M" tok="$M/openjev-tok"
  if [[ ! -s "$A/openjevhf_gguf_sha256.txt" ]]; then
    hfdl openjev/openjev-GGUF "$REV_OPENJEV_GGUF" "$d" OpenJev-Q4_K_M.gguf MANIFEST.json || return 1
    python3 - "$d" > "$A/openjevhf_gguf_sha256.txt" <<'PYEOF' || { echo "OpenJev-Q4_K_M.gguf sha256 differs from the publisher's manifest"; rm -f "$A/openjevhf_gguf_sha256.txt"; return 1; }
import hashlib, json, sys
d = sys.argv[1]
want = json.load(open(f"{d}/MANIFEST.json"))["files"]["OpenJev-Q4_K_M.gguf"]["sha256"]
h = hashlib.sha256()
with open(f"{d}/OpenJev-Q4_K_M.gguf", "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
print("sha256", h.hexdigest(), "manifest", want)
sys.exit(0 if h.hexdigest() == want else 1)
PYEOF
  fi
  hfdl openjev/openjev "$REV_OPENJEV" "$tok" tokenizer.json tokenizer_config.json chat_template.jinja config.json || return 1
  setup_llama || return 1
  prompts openjevhf openjev "$tok" || return 1
}
stage_openjevhf() {  # E30 amendment "OpenJev-27B through its official 4-bit GGUF": the publisher's Q4_K_M on llama.cpp, P = 8
  enough_time openjevhf || return 0
  openjevhf_prep || return 1
  llama_rows openjevhf "$MD/OpenJev-27B-Q4_K_M/OpenJev-Q4_K_M.gguf" "$M/openjev-tok" 8
}
stage_openjevhf16() {  # the same at P = 16
  enough_time openjevhf16 || return 0
  openjevhf_prep || return 1
  llama_rows openjevhf "$MD/OpenJev-27B-Q4_K_M/OpenJev-Q4_K_M.gguf" "$M/openjev-tok" 16
}
stage_decider() {  # E30 amendment "E15's open baselines" (c) 1: decider-4b on vLLM, its own row construction
  enough_time decider || return 0
  local d="$M/decider-4b"
  hfdl Mapika/decider-4b "$REV_DECIDER" "$d" || return 1
  prompts decider decider "$d" "$d" || return 1
  both decider "$d" "$d"
  rm -f "$d"/*.safetensors
}
stage_jevk5() {  # (c) 2: JevK5 v0.2 on vLLM, the jevk5 package 0.2.2
  enough_time jevk5 || return 0
  local d="$M/JevK5-v0.2" c="$W/e30_jevk5"
  hfdl alibiserikbay/JevK5 "$REV_JEVK5" "$d" || return 1
  grep '^sha ' "$A/logs/hf_JevK5-v0.2.log" > "$A/jevk5_weights_commit.txt"
  if [[ ! -d "$c/.git" ]]; then git clone -q --depth 1 --branch "$JEVK5_PKG_TAG" https://github.com/allebee/jevk5 "$c" || return 1; fi
  git -C "$c" rev-parse HEAD > "$A/jevk5_package_commit.txt"
  prompts jevk5 jevk5 "$d" "$c" || return 1
  both jevk5 "$d" "$c"
  rm -f "$d"/*.safetensors
}
stage_jevk5m() {  # amendment "E30: JevK5 at the weights E15 loaded": the same as stage_jevk5 at E15's commit (v0.3)
  enough_time jevk5m || return 0
  local d="$M/JevK5-main" c="$W/e30_jevk5"
  hfdl alibiserikbay/JevK5 "$REV_JEVK5_MAIN" "$d" || return 1
  grep '^sha ' "$A/logs/hf_JevK5-main.log" > "$A/jevk5m_weights_commit.txt"
  cp "$d/jevk5_config.json" "$A/jevk5m_config.json" 2>/dev/null
  if [[ ! -d "$c/.git" ]]; then git clone -q --depth 1 --branch "$JEVK5_PKG_TAG" https://github.com/allebee/jevk5 "$c" || return 1; fi
  git -C "$c" rev-parse HEAD > "$A/jevk5m_package_commit.txt"
  prompts jevk5m jevk5 "$d" "$c" || return 1
  both jevk5m "$d" "$c"
  rm -f "$d"/*.safetensors
}
stage_clefflash() {  # (c) 3: Clef-Flash 9B through the card's systemone() behind E15's wrapper (scoring head; marked)
  enough_time clefflash || return 0
  local d="$M/clef-flash" port=8090 name=clef-flash
  hfdl Cloudflare/clef-flash "$REV_CLEF" "$d" || return 1
  # E15's card-function venv (torch 2.11, transformers 5.10.2, torchvision + Pillow for the card's processor); the v2 venv
  # lacks an image processor class for the checkpoint
  if [[ ! -x "$DV/bin/python" ]] || ! "$DV/bin/python" -c "import torchvision, PIL, fastapi, uvicorn" > /dev/null 2>&1; then
    uv venv -q -p 3.11 "$DV" || return 1
    timeout 40m uv pip install -q --python "$DV" "torch==2.11.*" torchvision "transformers==5.10.2" accelerate peft safetensors \
      "huggingface_hub[cli]" fastapi uvicorn pydantic pillow numpy > "$A/logs/dmvenv_install.log" 2>&1 \
      || { tail -20 "$A/logs/dmvenv_install.log"; return 1; }
    timeout 20m uv pip install -q --python "$DV" flash-linear-attention >> "$A/logs/dmvenv_install.log" 2>&1 \
      || echo "flash-linear-attention not installed (the torch path is used)" | tee -a "$A/logs/dmvenv_install.log"
  fi
  "$DV/bin/python" -c "import fastapi, uvicorn, transformers, torch; print('dmvenv', transformers.__version__, torch.__version__)" \
    > "$A/env_clefflash.txt" 2>&1 || { cat "$A/env_clefflash.txt"; return 1; }
  uv pip freeze --python "$DV" > "$A/dmvenv_freeze.txt" 2>/dev/null
  echo "e15_wrap_server.py --kind clef --model $d --name $name --port $port (dmvenv)" > "$A/logs/serve_clefflash.cmd"
  setsid nohup "$DV/bin/python" -I "$R/scripts/aws/e15_wrap_server.py" --kind clef --model "$d" --name "$name" --port "$port" \
    > "$A/logs/serve_clefflash.out" 2>&1 < /dev/null &
  local sp=$!
  for _ in $(seq 1 120); do curl -sf -m 5 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1 && break
    kill -0 "$sp" 2>/dev/null || { tail -30 "$A/logs/serve_clefflash.out"; return 1; }; sleep 5; done
  smi_start "$A/smi/clefflash.csv"
  timeout 90m env PYTHONPATH="$R/src" "$HV/bin/python" "$R/$BENCH" time-wire --url "http://127.0.0.1:$port" --model "$name" \
    --order "$A/order.json" --label clefflash --out "$A/runs/clefflash_bf16.json.gz" > "$A/logs/time_clefflash.out" 2> "$A/logs/time_clefflash.err"
  local rc=$?
  smi_stop
  kill -- -"$sp" 2>/dev/null; kill "$sp" 2>/dev/null; sleep 5; kill -9 "$sp" 2>/dev/null
  st "clefflash: rc=$rc $(tail -c 300 "$A/logs/time_clefflash.out" | tr '\n' ' ')"
  sync_out
  rm -f "$d"/*.safetensors
}
stage_setup_min() {  # the g6e.xlarge of (c) 5: code, the harness venv and the fixed order only (no vLLM)
  stage_code || return 1
  PY -c "import jevharness, sys; p = jevharness.__file__; print(p); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_hvenv.txt" 2>&1 || { cat "$A/env_hvenv.txt"; return 1; }
  PY "$BENCH" order --out "$A/order.json" > "$A/order_info.json" 2> "$A/logs/order.log" || { tail -5 "$A/logs/order.log"; return 1; }
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s", "gpu": "%s", "vcpu": %s, "ram_gib": %s, "mins_left": %s}\n' \
    "$MY_IID" "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" \
    "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" "$(nproc)" \
    "$(free -g | awk '/Mem:/{print $2}')" "$(mins_left)" > "$A/host.json"
  cat "$A/order_info.json" "$A/host.json"
  sync_out
}
stage_openjev() {  # (c) 5: OpenJev through its own image with the FP8 weights, as E15 ran it (marked; the g6e.xlarge)
  enough_time openjev || return 0
  local port=8090 mid=openjev-latest hf="$HOME/.cache/huggingface"
  docker_ready || return 1
  timeout 30m sudo docker pull -q "$OPENJEV_IMG" > "$A/logs/openjev_pull.log" 2>&1 || { tail -5 "$A/logs/openjev_pull.log"; return 1; }
  echo "$OPENJEV_IMG OPENJEV_MODEL=$OPENJEV_WEIGHTS" > "$A/openjev_image.txt"
  mkdir -p "$hf"
  sudo docker rm -f e30openjev > /dev/null 2>&1
  sudo docker run -d --name e30openjev --gpus all --ipc=host -p "127.0.0.1:$port:8080" -v "$hf:/root/.cache/huggingface" \
    -e OPENJEV_MODEL="$OPENJEV_WEIGHTS" "$OPENJEV_IMG" > "$A/logs/openjev_cid.txt" 2>&1 || return 1
  local up=0
  for _ in $(seq 1 240); do
    if curl -sf -m 5 "http://127.0.0.1:$port/v1/models" > "$A/openjev_models.json" 2>/dev/null; then up=1; break; fi
    sudo docker ps -q -f name=e30openjev | grep -q . || break
    sleep 10
  done
  if (( up == 0 )); then sudo docker logs e30openjev > "$A/logs/openjev_docker.log" 2>&1; st "openjev: the image did not serve"; sync_out; return 1; fi
  smi_start "$A/smi/openjev.csv"
  timeout 100m env PYTHONPATH="$R/src" "$HV/bin/python" "$R/$BENCH" time-wire --url "http://127.0.0.1:$port" --model "$mid" \
    --order "$A/order.json" --precision fp8 --label openjev --out "$A/runs/openjev_fp8.json.gz" > "$A/logs/time_openjev.out" 2> "$A/logs/time_openjev.err"
  local rc=$?
  smi_stop
  sudo docker logs e30openjev > "$A/logs/openjev_docker.log" 2>&1
  sudo docker rm -f e30openjev > /dev/null 2>&1
  st "openjev: rc=$rc $(tail -c 300 "$A/logs/time_openjev.out" | tr '\n' ' ')"
  sync_out
}
stage_final() {
  sudo docker rm -f e30llama e30openjev > /dev/null 2>&1
  pkill -f 'serve_jev' 2>/dev/null; pkill -f 'e15_wrap_server' 2>/dev/null
  nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  printf '{"final_utc": "%s", "instance_id": "%s", "instance_type": "%s", "hours_used": %s}\n' "$(date -u +%FT%TZ)" "$MY_IID" \
    "$(imds instance-type)" "$(hours_used)" > "$A/host_final.json"
  st "final: used $(hours_used) h; disk $(df -BG / | awk 'NR==2{print $4}') free"
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; sudo shutdown -h +2 "jev-e30-cost: done"; fi
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
  if (( rc != 0 )) && [[ "$s" == rearm || "$s" == code || "$s" == setup ]]; then
    st "chain stopped at $s"
    [[ " $STAGES " == *" final "* ]] && stage_final
    exit 1
  fi
done
sync_out
st "chain end [$STAGES]"
