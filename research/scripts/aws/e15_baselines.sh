#!/usr/bin/env bash
# Exploratory row E15, more decision-model baselines (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E15: more
# decision-model baselines"; descriptive, outside every Holm table). One NEW GPU host launched by this driver from E3's
# clone image (tag Name=jev-e15-baselines), SSM only. Each public decision model is served on :8090, one at a time, by
# its vendor's documented server (a Docker image, its own server, or the model card's function behind a thin wrapper
# of ours, scripts/aws/e15_wrap_server.py), and answers (1) E1's 299-item development prefix (wire check), (2) E1's
# 3,991 decisions (L1) and (3) the common states C (L2); then the live REPLACE rows of the two GPU models with the
# highest L2 accuracy (Flash on the direct Gemini API; the key from SSM into the live process only, never printed,
# never on disk). Downloaded code and weights run on this host only, each in its own directory or container.
#
#   LOCAL:  bash scripts/aws/e15_baselines.sh launch [itype ...]   # one new host (refuses if a jev-e15-baselines host exists)
#           E15_IID=i-... bash scripts/aws/e15_baselines.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages (in the order sent; a failed model stage is logged and the chain goes on; `final` always runs when sent):
#     rearm        idle stop (IDLE_MIN) + power-off guard (GUARD_MIN from the first boot), host record
#     setup        the committed branch in /opt/jev/e15; docker + the NVIDIA runtime; the model venv; the corpus C copy
#     m_<slug>     serve the model, wire check, L1, L2, stop, delete its weights   (slugs: see SLUGS below)
#     live_<slug>  REPLACE live row of a GPU model (probe-gated; paused on 503 / overload; spend-capped)
#     live_top2    the live rows of the two GPU models with the highest L2 accuracy (registration (c) 4)
#     final        stop the servers, sync, power-off (POWEROFF=1)
# Records: $A = results/explore/e15_baselines/box on the box -> s3://$BUCKET/e15_baselines/box; runs ->
# s3://$BUCKET/e15_baselines/runs.
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E15_IID=${E15_IID:-}
# every other project host: never touched by this driver (E12, E12c, E4r, E14 and the older hosts)
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
TAG=jev-e15-baselines
BUNDLE=e15.bundle
S3OUT="$B/e15_baselines"
CAP_USD=${CAP_USD:-35}
GUARD_MIN=${GUARD_MIN:-710}          # 12 GPU-hours cap (registration (e)): the guard 11 h 50 min after the first boot
IDLE_MIN=${IDLE_MIN:-30}
POWEROFF=${POWEROFF:-1}
PORT=8090

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E15_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E15_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E15_IID "* ]] && { echo "refused: $E15_IID is not the E15 host"; return 1; }
  local tag
  tag=$(awsl ec2 describe-instances --instance-ids "$E15_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
  [[ "$tag" == "$TAG" ]] || { echo "refused: $E15_IID is tagged '$tag', not $TAG"; return 1; }
  return 0
}

if [[ "$MODE" == launch ]]; then
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  ud=$(mktemp)
  printf '#!/bin/bash\nshutdown -h +%s "jev-e15-baselines: power-off guard (12 GPU-hours from launch)"\n' "$GUARD_MIN" > "$ud"
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g6e.2xlarge g6e.4xlarge)
  for IT in "${types[@]}"; do
    for SUBNET in $SUBNETS; do
      out=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$IT" --subnet-id "$SUBNET" --security-group-ids "$SG" \
        --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior stop --count 1 \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=400,VolumeType=gp3,Throughput=500,Iops=6000,DeleteOnTermination=true}' \
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/e15_baselines.sh" > "$tmp/e15_baselines.sh"   # the committed driver
  awsl s3 cp "$tmp/e15_baselines.sh" "$B/code/e15_baselines.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e15_baselines.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e15_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/e15_baselines.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q GUARD_MIN=%q IDLE_MIN=%q POWEROFF=%q E15_IID=%q KEEP_WEIGHTS=%q USED_MIN=%q\n' \
      "$CAP_USD" "$GUARD_MIN" "$IDLE_MIN" "$POWEROFF" "$E15_IID" "${KEEP_WEIGHTS:-0}" "${USED_MIN:-}"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e15_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/e15_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E15_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/e15_driver.sh /tmp/e15_driver.sh --only-show-errors && bash /tmp/e15_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E15_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E15_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,22p' "$0"
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
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E15_IID" && "$MY_IID" != "$E15_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E15 host ${E15_IID:-?}"; exit 1
fi
R=$W/e15
A=$R/results/explore/e15_baselines/box
STATUS_DIR=$W/e15_status
HV=$W/hvenv
DV=$W/e15_dmvenv            # torch + transformers for the card-function models (Clef, Bosun) and decider's server
IV=$W/e15_imajev_venv       # Imajev's own package
VV=$W/e15_vllm_venv         # vLLM 0.29.0 + the OpenJev (HF) helper
HFD=$W/e15_hf               # Hugging Face cache of the host processes (owned by ubuntu)
HFDOCK=$W/e15_hf_docker     # the containers' own cache (they write as root; kept apart so HFD stays writable)
MD=$W/e15_models            # snapshots / repositories, one directory per model
STOP=runs/e15_STOP.json
EX=scripts/explore_e15_baselines.py
E1CFG=configs/explore/e15_baselines/e1_e15.yaml
DEVCFG=configs/explore/e15_baselines/e1dev_e15.yaml
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="$HFD"
mkdir -p "$STATUS_DIR" "$HFD" "$HFDOCK" "$MD"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  aws s3 cp "$STATUS_DIR/status.txt" "$S3OUT/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && aws s3 sync "$A" "$S3OUT/box" --only-show-errors --region "$REGION" --exclude 's23/corpus.json' \
    --exclude 's23/occurrences.jsonl.gz' --exclude 's23/questions.jsonl.gz'
  local d
  for d in "$R"/runs/e1_bfcl_when2call_e15 "$R"/runs/e1dev_e15 "$R"/runs/e15_*; do
    [[ -d "$d" ]] && aws s3 sync "$d" "$S3OUT/runs/$(basename "$d")" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  true
}
PY() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }
hours_used() {
  local b; b=$(cat "$W/e15_first_boot" 2>/dev/null || date -d "$(uptime -s)" +%s)
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
leak_check() {
  local n=0 k ks=${GEMINI_API_KEYS:-}
  for k in ${GEMINI_API_KEY:-} ${ks//,/ }; do
    [[ ${#k} -ge 16 ]] || continue
    n=$(( n + $(grep -rlF -- "$k" "$R/runs" "$A" "$STATUS_DIR" 2>/dev/null | wc -l) ))
  done
  echo "key-leak files: $n"
  [[ "$n" == 0 ]]
}
smi_start() { nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 1000 > "$1" 2>/dev/null & SMI=$!; }
smi_stop() { [[ -n "${SMI:-}" ]] && kill "$SMI" 2>/dev/null; SMI=""; }
hfdl() {  # hfdl REPO DIR [REV] [files...]: a pinned snapshot into its own directory; the revision is recorded
  local repo=$1 dir=$2 rev=${3:-main}; shift 3 2>/dev/null || shift $#
  mkdir -p "$dir"
  "$DV/bin/hf" download "$repo" --revision "$rev" --local-dir "$dir" "$@" > "$A/logs/hf_$(basename "$dir").log" 2>&1 || return 1
  "$DV/bin/python" -c "from huggingface_hub import HfApi; i = HfApi().model_info('$repo', revision='$rev'); print(i.sha)" \
    > "$dir.sha" 2>/dev/null || true
  printf '{"repo": "%s", "requested": "%s", "sha": "%s", "at": "%s"}\n' "$repo" "$rev" "$(cat "$dir.sha" 2>/dev/null)" \
    "$(date -u +%FT%TZ)" >> "$A/downloads.jsonl"
}

# ------------------------------------------------------------------------------------------------ servers
stop_server() {
  sudo docker rm -f e15srv > /dev/null 2>&1
  local f p
  for f in "$W"/e15_srv_*.pid; do
    [[ -f "$f" ]] || continue
    p=$(cat "$f"); kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$p" 2>/dev/null; kill -9 "$p" 2>/dev/null
    rm -f "$f"
  done
  pkill -f 'vllm serve' 2>/dev/null; pkill -f 'helper/shim.py' 2>/dev/null
  for _ in $(seq 1 30); do curl -s -m 2 "http://127.0.0.1:$PORT/" > /dev/null 2>&1 || break; sleep 2; done
  true
}
wait_up() {  # wait_up MODEL_ID TAG [MINUTES]: a 2-option Choice answered on :8090 (the readiness check of every server)
  local mid=$1 tag=$2 mins=${3:-45} end
  end=$(( $(date +%s) + mins * 60 ))
  while (( $(date +%s) < end )); do
    if curl -sf -m 120 -H 'Content-Type: application/json' -X POST "http://127.0.0.1:$PORT/v1/systemone" \
        -d "{\"model\": \"$mid\", \"state\": \"The user asks to ship order 7.\", \"questions\": {\"q0\": {\"type\": \"choice\", \"instructions\": \"Which tool next?\", \"criteria\": {\"ship_order\": null, \"cancel_order\": null}}}}" \
        > "$A/logs/ready_$tag.json" 2>/dev/null && grep -q '"answers"' "$A/logs/ready_$tag.json"; then
      printf '{"server_up": "%s", "slug": "%s", "model": "%s", "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" "$tag" "$mid" \
        "$(imds instance-type)" >> "$A/serve_windows.jsonl"
      return 0
    fi
    if [[ -n "${SRV_PID:-}" ]] && ! kill -0 "$SRV_PID" 2>/dev/null; then echo "server died"; return 1; fi
    if [[ "${SRV_DOCKER:-0}" == 1 ]] && [[ -z "$(sudo docker ps -q -f name=e15srv)" ]]; then
      sudo docker logs --tail 40 e15srv > "$A/logs/docker_$tag.tail" 2>&1; echo "container exited"; return 1
    fi
    sleep 10
  done
  echo "server did not come up in $mins min"; return 1
}
bgsrv() {  # bgsrv CMD...: a server process (tag $TAGNOW) in its own process group; its pid file
  setsid nohup "$@" > "$A/logs/serve_$TAGNOW.out" 2>&1 < /dev/null &
  SRV_PID=$!
  echo "$SRV_PID" > "$W/e15_srv_$TAGNOW.pid"
}
serve_docker() {  # serve_docker SLUG IMAGE MODEL_ID [docker -e args...]
  local slug=$1 img=$2 mid=$3; shift 3
  stop_server
  SRV_DOCKER=1; SRV_PID=""
  timeout 40m sudo docker pull -q "$img" > "$A/logs/pull_$slug.log" 2>&1 || { tail -5 "$A/logs/pull_$slug.log"; return 1; }
  sudo docker image inspect --format '{{index .RepoDigests 0}}' "$img" > "$A/image_$slug.digest" 2>/dev/null
  echo "docker run $img $*" > "$A/logs/serve_$slug.cmd"
  sudo docker run -d --name e15srv --gpus all --ipc=host -p "127.0.0.1:$PORT:8080" \
    -v "$HFDOCK:/root/.cache/huggingface" "$@" "$img" > "$A/logs/serve_$slug.cid" 2>&1 || return 1
  wait_up "$mid" "$slug" 60
  local rc=$?
  sudo docker logs e15srv > "$A/logs/docker_$slug.log" 2>&1
  return $rc
}
serve_wrap() {  # serve_wrap SLUG KIND MODEL_DIR NAME [extra]: the card's function behind scripts/aws/e15_wrap_server.py
  local slug=$1 kind=$2 dir=$3 name=$4; shift 4
  stop_server
  SRV_DOCKER=0; TAGNOW=$slug
  echo "e15_wrap_server.py --kind $kind --model $dir --name $name --port $PORT $*" > "$A/logs/serve_$slug.cmd"
  bgsrv "$DV/bin/python" -I "$R/scripts/aws/e15_wrap_server.py" --kind "$kind" --model "$dir" --name "$name" --port "$PORT" "$@"
  wait_up "$name" "$slug" 40
}

# per-model serving (registration (b)); every model answers on :8090
start_model() {  # start_model SLUG -> serves it; sets MID (the wire model id)
  local slug=$1 d
  case "$slug" in
    clefflash)
      d="$MD/clef-flash"; hfdl Cloudflare/clef-flash "$d" fde727a287004204b7518dcc983fe64379776712 || return 1
      MID=clef-flash; serve_wrap clefflash clef "$d" "$MID" ;;
    clef)
      d="$MD/clef"; hfdl Cloudflare/clef "$d" ed3eed331870db2eff4b0db01237128ede8a00ce || return 1
      MID=clef; serve_wrap clef clef "$d" "$MID" --load-8bit ;;
    openjev)
      MID=openjev-latest
      serve_docker openjev razorback16/openjev:0.5.0 "$MID" && return 0
      st "openjev: the NVFP4 weights did not serve; the image's OPENJEV_MODEL switch to the FP8 weights (registration (b) 2)"
      echo fp8 > "$A/openjev_variant.txt"
      serve_docker openjev razorback16/openjev:0.5.0 "$MID" -e OPENJEV_MODEL=RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic ;;
    jevk5)
      MID=jevk5-0.2; serve_docker jevk5 razorback16/openjev-jevk5:0.5.1 "$MID" ;;
    clm)
      MID=clm-v0.1; serve_docker clm razorback16/openjev-clm:0.5.1 "$MID" ;;
    laya)
      MID=laya-1.0; serve_docker laya razorback16/openjev-laya:0.5.1 "$MID" ;;
    verdict)
      MID=verdict-1.4; serve_docker verdict razorback16/openjev-verdict:0.5.1 "$MID" ;;
    decider)
      d="$MD/decider-4b"; hfdl Mapika/decider-4b "$d" main || return 1
      stop_server; SRV_DOCKER=0; TAGNOW=decider; MID=decider-4b
      echo "DECIDER_MODEL=$d uvicorn decider.serve:app --host 127.0.0.1 --port $PORT (cwd $d)" > "$A/logs/serve_decider.cmd"
      (cd "$d" && DECIDER_MODEL="$d" bgsrv "$DV/bin/python" -m uvicorn decider.serve:app --host 127.0.0.1 --port "$PORT")
      SRV_PID=$(cat "$W/e15_srv_decider.pid")
      wait_up "$MID" decider 40 ;;
    imajev)
      d="$MD/imajev"
      if [[ ! -d "$d/.git" ]]; then git clone -q https://github.com/mohit67890/imajev "$d" || return 1; fi
      git -C "$d" rev-parse HEAD > "$A/imajev_commit.txt"
      if [[ ! -x "$IV/bin/python" ]]; then
        uv venv -q -p 3.11 "$IV" || return 1
        (cd "$d" && timeout 40m uv pip install -q --python "$IV" -e ".[serve,torch]") > "$A/logs/imajev_install.log" 2>&1 \
          || { tail -20 "$A/logs/imajev_install.log"; return 1; }
      fi
      (cd "$d" && timeout 40m "$IV/bin/python" scripts/download_model.py --model 4b) > "$A/logs/imajev_download.log" 2>&1 \
        || { tail -10 "$A/logs/imajev_download.log"; return 1; }
      hfdl mohit67890/imajev-4b "$d/adapters/imajev-4b" main || return 1
      stop_server; SRV_DOCKER=0; TAGNOW=imajev; MID=imajev-4b
      echo "server.py --backend torch --model-bundle artifacts/model-qwen4b.json --adapter adapters/imajev-4b --rotations 4 --calibration adapters/imajev-4b/calibration-rot4.json --model-name imajev-4b --port $PORT" \
        > "$A/logs/serve_imajev.cmd"
      (cd "$d" && PYTHONPATH=src:scripts bgsrv "$IV/bin/python" scripts/playground/server.py --backend torch \
        --model-bundle artifacts/model-qwen4b.json --adapter adapters/imajev-4b --rotations 4 \
        --calibration adapters/imajev-4b/calibration-rot4.json --model-name imajev-4b --port "$PORT")
      SRV_PID=$(cat "$W/e15_srv_imajev.pid")
      wait_up "$MID" imajev 40 ;;
    bosun)
      d="$MD/bosun"; hfdl Hanno-Labs/bosun-v3.1-1.7b "$d" main || return 1
      MID=bosun-3.1-1.7b; serve_wrap bosun bosun "$d" "$MID" ;;
    openjevhf)
      d="$MD/openjev-fp8"; hfdl openjev/openjev-FP8 "$d" main || return 1
      hfdl openjev/openjev "$MD/openjev-helper" ac97900fd034fdd7e7e536f3d4c21b836cae0750 helper/shim.py serve/SERVE.md NOTICE LICENSE LICENSE-APACHE-2.0 || return 1
      sha256sum "$MD/openjev-helper/helper/shim.py" > "$A/openjevhf_shim.sha256"
      if [[ ! -x "$VV/bin/vllm" ]]; then
        uv venv -q -p 3.12 "$VV" || return 1
        timeout 40m uv pip install -q --python "$VV" "vllm==0.29.0" "openai==3.16.2" "httpx==0.28.1" > "$A/logs/vllm_install.log" 2>&1 \
          || { tail -20 "$A/logs/vllm_install.log"; return 1; }
      fi
      stop_server; SRV_DOCKER=0; TAGNOW=openjevhf_vllm
      # the card's flags (measured on an 80 GB H100) except --max-num-seqs: on the 48 GB L40S vLLM 0.29.0 refuses 256
      # (only 165 Mamba cache blocks fit); 128 is the most concurrent sequences this host serves, far above the helper's
      # 16 questions in flight times our 4 workers. It changes capacity only, never an answer.
      echo "VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve $d --host 127.0.0.1 --served-model-name qwen --port 8000 --enable-prefix-caching --max-model-len 16384 --gpu-memory-utilization 0.90 --limit-mm-per-prompt {\"image\":1} --trust-remote-code --max-num-seqs 128 --max-logprobs 64 --gdn-prefill-backend triton" \
        > "$A/logs/serve_openjevhf.cmd"
      # VLLM_USE_FLASHINFER_SAMPLER=0: FlashInfer's sampler is JIT-compiled at the first request and needs nvcc, which
      # this image lacks (the engine failed at start); vLLM's own sampler takes over. The helper reads the option
      # letters' log-probabilities, which are computed before sampling, so no answer changes (the openjev images set
      # the same variable for the same reason).
      bgsrv env VLLM_USE_FLASHINFER_SAMPLER=0 "$VV/bin/vllm" serve "$d" --host 127.0.0.1 --served-model-name qwen --port 8000 --enable-prefix-caching \
        --max-model-len 16384 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":1}' --trust-remote-code \
        --max-num-seqs 128 --max-logprobs 64 --gdn-prefill-backend triton
      local vp=$SRV_PID
      for _ in $(seq 1 240); do curl -sf -m 5 http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && break
        kill -0 "$vp" 2>/dev/null || { tail -30 "$A/logs/serve_openjevhf_vllm.out"; return 1; }; sleep 10; done
      TAGNOW=openjevhf; MID=openjev
      bgsrv env VLLM=http://localhost:8000/v1 TOKENIZER="$d" READOUT_T=0.85 READOUT_NOUL_T=1.829074 READOUT_NOUL_BIAS=0 \
        READOUT_TARGETED=1 READOUT_INSTR_STYLE=pyrepr SHIM_STAGGER=1 \
        "$VV/bin/python" "$MD/openjev-helper/helper/shim.py" --host 127.0.0.1 --port "$PORT"
      wait_up "$MID" openjevhf 20 ;;
    *) echo "unknown slug $slug"; return 1 ;;
  esac
}
drop_weights() {  # weights deleted after each model (registration (e)); KEEP_WEIGHTS=1 keeps a live model's
  local slug=$1
  [[ "${KEEP_WEIGHTS:-0}" == 1 ]] && return 0
  case "$slug" in
    clefflash) rm -rf "$MD/clef-flash" ;;
    clef) rm -rf "$MD/clef" ;;
    decider) rm -rf "$MD/decider-4b" ;;
    bosun) rm -rf "$MD/bosun" ;;
    openjevhf) rm -rf "$MD/openjev-fp8" ;;
    imajev) rm -rf "$MD/imajev/adapters" "$MD/imajev/models" "$MD/imajev/artifacts/"*.safetensors 2>/dev/null ;;
  esac
  df -BG / | awk 'NR==2{print "disk free: "$4}'
}

# ------------------------------------------------------------------------------------------------ stages
stage_rearm() {
  # after a stop / start the wall time since the first boot overstates the GPU time: USED_MIN (the running minutes so
  # far, from the instance's EC2 state history, passed by `send`) resets the clock the guard and hours_used() read
  if [[ -n "${USED_MIN:-}" ]]; then echo $(( $(date +%s) - USED_MIN * 60 )) > "$W/e15_first_boot"; fi
  [[ -s "$W/e15_first_boot" ]] || date -d "$(uptime -s)" +%s > "$W/e15_first_boot"
  local used_min left
  used_min=$(( ( $(date +%s) - $(cat "$W/e15_first_boot") ) / 60 ))
  left=$(( GUARD_MIN - used_min ))
  (( left > 0 )) || { echo "guard already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$left" "jev-e15-baselines: power-off guard (GPU cap)"
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E15 chain / server / runner process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e15_[0-9]|jevbench|explore_e15|stage2_corpus|decision_eval|uv pip|pip install|docker pull|hf download' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e15-baselines: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $left min left, idle $IDLE_MIN min; host $MY_IID $(imds instance-type); $(nproc) vCPU; $(df -BG / | awk 'NR==2{print $4}') free; used $(hours_used) h"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
}
stage_setup() {
  local myb="$W/e15.$$.bundle"
  sudo chown -R "$(id -un):$(id -gn)" "$HFD" 2>/dev/null   # a container of an earlier chain wrote into it as root
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A/logs" runs
  st "code at $(git log --oneline -1 | cut -c1-72)"
  grep -q 'E15: more decision-model baselines' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for E15"; return 1; }
  PY -c "import jevharness, sys; p = jevharness.__file__; print(p); sys.exit(0 if p.startswith('$R/src') else 1)" \
    > "$A/env_hvenv.txt" 2>&1 || { cat "$A/env_hvenv.txt"; echo "the hvenv does not import this checkout"; return 1; }
  # docker + the NVIDIA container runtime (the openjev images)
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
        || { tail -5 "$A/logs/nvctk_install.log"; return 1; }
    fi
    sudo nvidia-ctk runtime configure --runtime=docker > /dev/null 2>&1 && sudo systemctl restart docker
  fi
  sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L > "$A/logs/docker_gpu_check.txt" 2>&1 || { cat "$A/logs/docker_gpu_check.txt"; return 1; }
  command -v uv > /dev/null || bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1' || return 1
  if [[ ! -x "$DV/bin/python" ]] || ! "$DV/bin/python" -c "import transformers, fastapi, huggingface_hub, torchvision" > /dev/null 2>&1; then
    [[ -x "$DV/bin/python" ]] || uv venv -q -p 3.11 "$DV" || return 1
    # torchvision: Clef's processor (Qwen3.5's video processor) imports it
    timeout 40m uv pip install -q --python "$DV" "torch==2.11.*" torchvision "transformers==5.10.2" accelerate peft safetensors \
      "huggingface_hub[cli]" fastapi uvicorn pydantic pillow bitsandbytes numpy > "$A/logs/dmvenv_install.log" 2>&1 \
      || { tail -20 "$A/logs/dmvenv_install.log"; return 1; }
    timeout 20m uv pip install -q --python "$DV" flash-linear-attention >> "$A/logs/dmvenv_install.log" 2>&1 \
      || echo "flash-linear-attention not installed (the torch path is used)" | tee -a "$A/logs/dmvenv_install.log"
  fi
  "$DV/bin/python" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())" \
    > "$A/env_dmvenv.txt" 2>&1 || { cat "$A/env_dmvenv.txt"; return 1; }
  uv pip freeze --python "$DV" > "$A/dmvenv_freeze.txt" 2>/dev/null
  mkdir -p "$A/s23"
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz; do cp -f "results/stage2/s23/$f" "$A/s23/$f" || return 1; done
  printf '{"instance_id": "%s", "instance_type": "%s", "az": "%s", "setup_utc": "%s", "gpu": "%s", "driver": "%s"}\n' "$MY_IID" \
    "$(imds instance-type)" "$(imds placement/availability-zone)" "$(date -u +%FT%TZ)" \
    "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" > "$A/host.json"
  cat "$A/env_dmvenv.txt"
  df -BG / | awk 'NR==2{print "disk free: "$4}'
}
spec_of() {  # spec_of SLUG MODEL_ID: the harness's jev backend for the served model (registration (b))
  PY "$EX" spec --slug "$1" --model "$2" --base-url "http://127.0.0.1:$PORT"
}
eval_model() {  # eval_model SLUG: wire check, L1, L2 through the model on :8090
  local slug=$1 k rc spec
  cd "$R" || return 1
  spec=$(spec_of "$slug" "$MID") || return 1
  echo "$spec" > "$A/spec_$slug.json"
  PY -m jevharness.bench.decision_eval "$DEVCFG" --out runs --max-items 300 --only "$slug" > "$A/logs/dev_$slug.log" 2>&1
  st "$slug wire check exit=$?: $(PY "$EX" quick --run runs/e1dev_e15 --slug "$slug" 2>/dev/null)"
  PY "$EX" wire-ok --run runs/e1dev_e15 --slug "$slug" > "$A/logs/wire_$slug.json" 2>&1 \
    || { st "$slug: the wire check failed (more than half the answers errored); L1 / L2 not run"; return 1; }
  for k in 1 2; do   # two invocations as E1 (the second resumes failed requests from the cache)
    PY -m jevharness.bench.decision_eval "$E1CFG" --out runs --only "$slug" > "$A/logs/e1_${slug}_$k.log" 2>&1
    st "$slug L1 invocation $k exit=$?: $(PY "$EX" quick --run runs/e1_bfcl_when2call_e15 --slug "$slug" 2>/dev/null)"
  done
  sync_out
  for k in 1 2 3; do   # the replay resumes
    PY "$EX" replay --corpus "$A/s23" --name "$slug" --s1 "$spec" --cache "cache/s1_e15_${slug}_c.sqlite" --workers 4 \
      > "$A/logs/replay_c_${slug}_$k.log" 2>&1
    rc=$?
    st "$slug L2 invocation $k exit=$rc: $(wc -l < "$A/s23/answers_$slug.jsonl" 2>/dev/null) answers; $(PY "$EX" quickc --corpus "$A/s23" --slug "$slug" 2>/dev/null)"
    (( rc == 0 )) && break
  done
  sync_out
}
model_stage() {  # model_stage SLUG
  local slug=$1 t0 rc
  t0=$(date +%s)
  if [[ "$slug" == clef ]]; then   # registration (b) 4: only if >= 3 GPU-hours remain beyond the 2.5 h live reserve
    local left
    left=$(python3 -c "print(round($GUARD_MIN / 60 - $(hours_used) - 2.5, 2))")
    if python3 -c "import sys; sys.exit(0 if $left >= 3 else 1)"; then st "clef: $left GPU-hours beyond the live reserve; run"
    else st "clef: only $left GPU-hours beyond the live reserve (< 3); not run (registration (b) 4)"
      printf '{"slug": "clef", "served": false, "skipped": "GPU-hours", "left_h": %s}\n' "$left" >> "$A/model_runs.jsonl"
      return 0; fi
  fi
  st "$slug: start (used $(hours_used) h)"
  smi_start "$A/logs/smi_$slug.csv"
  start_model "$slug"
  rc=$?
  if (( rc != 0 )); then
    smi_stop
    st "$slug: did not serve (exit $rc); see logs/serve_$slug.* (dropped unless a later stage serves it)"
    printf '{"slug": "%s", "served": false, "minutes": %s}\n' "$slug" "$(( ($(date +%s) - t0) / 60 ))" >> "$A/model_runs.jsonl"
    stop_server; drop_weights "$slug"; sync_out
    return 0
  fi
  st "$slug: serving ($MID) after $(( ($(date +%s) - t0) / 60 )) min"
  eval_model "$slug"
  rc=$?
  smi_stop
  stop_server
  printf '{"slug": "%s", "served": true, "eval_exit": %s, "minutes": %s, "used_h": %s}\n' "$slug" "$rc" \
    "$(( ($(date +%s) - t0) / 60 ))" "$(hours_used)" >> "$A/model_runs.jsonl"
  drop_weights "$slug"
  st "$slug: done in $(( ($(date +%s) - t0) / 60 )) min (used $(hours_used) h)"
  sync_out
  return 0
}
probe_ok() {  # two consecutive clean probe rounds (registration (d)); waits up to PROBE_WAIT_MIN
  local end=$(( $(date +%s) + ${PROBE_WAIT_MIN:-240} * 60 )) clean=0
  while (( $(date +%s) < end )); do
    if PY "$EX" probe --log "$A/api_probe_log.txt" > /dev/null 2>&1; then clean=$((clean + 1)); else clean=0; fi
    (( clean >= 2 )) && return 0
    sleep $(( clean > 0 ? 30 : 300 ))
  done
  return 1
}
run_live() {  # run_live CONFIG RUN_DIR: probe-gated, paused on 503 / overload, capped spend; then one re-run of errors
  local cfg=$1 run=$2 rc pid wpid round spid
  for round in 1 2 3 4 5 6 7 8; do
    [[ -f "$STOP" ]] && { st "spend cap reached earlier; $cfg not run further"; return 3; }
    probe_ok || { st "probes not clean; $cfg paused (round $round)"; return 4; }
    st "$cfg: start (round $round)"
    setsid env PYTHONPATH="$R/src" JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$cfg" --out runs >> "$A/runner.log" 2>&1 < /dev/null &
    pid=$!
    PY "$EX" watch --pid "$pid" --cap "$CAP_USD" --runs runs --stop "$STOP" --every 30 >> "$A/watch.log" 2>&1 &
    wpid=$!
    ( while kill -0 "$pid" 2>/dev/null; do sleep 300; sync_out; done ) & spid=$!
    wait "$pid"; rc=$?
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
live_stage() {  # live_stage SLUG: the model's REPLACE row (registration (c) 4), smoke + gate first
  local slug=$1 cfg smoke
  cd "$R" || return 1
  cfg=configs/explore/e15_baselines/live_$slug.yaml
  smoke=configs/explore/e15_baselines/live_smoke_$slug.yaml
  [[ -f "$cfg" && -f "$smoke" ]] || { st "live $slug: no committed live config"; return 1; }
  keys || return 1
  start_model "$slug" || { st "live $slug: the model did not serve"; return 1; }
  run_live "$smoke" "runs/e15_live_smoke_$slug" || { stop_server; return 1; }
  PY "$EX" gate --runs runs --smoke "runs/e15_live_smoke_$slug" --n 200 --cap "$CAP_USD" > "$A/logs/gate_$slug.json" 2>&1 \
    || { st "gate refuses live $slug: $(cat "$A/logs/gate_$slug.json")"; stop_server; return 3; }
  run_live "$cfg" "runs/e15_live_$slug"
  local rc=$?
  stop_server
  leak_check || return 1
  return $rc
}
stage_live_top2() {  # registration (c) 4: the two GPU models with the highest L2 accuracy (complete L2 only), then live
  cd "$R" || return 1
  local top s rc
  top=$(PY "$EX" rank --corpus "$A/s23" --l1 runs/e1_bfcl_when2call_e15 --top 2 --out "$A/live_selection.json") || return 1
  st "live selection (registration (c) 4): $top"
  for s in $top; do
    live_stage "$s"; rc=$?
    st "live_$s exit=$rc"
  done
}
stage_final() {
  stop_server
  sync_out
  st "final: used $(hours_used) h; $(PY "$EX" spend --runs runs --short 2>/dev/null)"
  if [[ "$POWEROFF" == 1 ]]; then sudo shutdown -h +1 "jev-e15-baselines: done"; fi
}

st "chain start: $STAGES (host $MY_IID, used $(hours_used) h)"
for s in $STAGES; do
  t0=$(date +%s)
  case "$s" in
    rearm|setup|final) "stage_$s" > "$STATUS_DIR/stage_$s.log" 2>&1; rc=$? ;;
    m_*) model_stage "${s#m_}" > "$STATUS_DIR/stage_$s.log" 2>&1; rc=$? ;;
    live_top2) stage_live_top2 > "$STATUS_DIR/stage_$s.log" 2>&1; rc=$? ;;
    live_*) live_stage "${s#live_}" > "$STATUS_DIR/stage_$s.log" 2>&1; rc=$? ;;
    *) echo "unknown stage $s"; rc=1 ;;
  esac
  [[ -d "$A/logs" ]] && cp -f "$STATUS_DIR/stage_$s.log" "$A/logs/stage_$s.log" 2>/dev/null
  st "stage $s exit=$rc ($(( ($(date +%s) - t0) / 60 )) min): $(tail -2 "$STATUS_DIR/stage_$s.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  if (( rc != 0 )) && [[ "$s" == rearm || "$s" == setup ]]; then
    st "chain stopped at $s"
    [[ " $STAGES " == *" final "* ]] && stage_final
    exit 1
  fi
done
st "chain end"
