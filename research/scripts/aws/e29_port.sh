#!/usr/bin/env bash
# E29 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 ~16:05 UTC, "E29: the full harness in open-source harnesses";
# exploratory, descriptive): the decision layer as a host-side model proxy (jevharness.adapters.route_proxy) in three
# open-source harnesses, one NEW CPU host per harness (SSM only, no inbound):
#   H=tb    pi 0.99.2 on Terminal-Bench 2.0 (Harbor)          proxy :8091 (Gemini)      tag Name=jev-e29-tb
#   H=mswe  mini-SWE-agent 2.4.6 on SWE-bench Verified         proxy :8093 (OpenRouter)  tag Name=jev-e29-mswe
#   H=swea  SWE-agent v1.1.0 on SWE-bench Verified (Harbor)    proxy :8092 (Gemini)      tag Name=jev-e29-swea
# Arms: baseline (the harness unchanged: every step on the strong LLM) and gated (the full harness), both through the
# proxy, run concurrently on the same host; calibration runs the shadow arm on the disjoint calibration split.
#
#   LOCAL: bash scripts/aws/e29_port.sh launch tb|mswe|swea [itype ...]   # refuses if a host with the tag exists
#          bash scripts/aws/e29_port.sh push                             # bundle + this driver -> s3 code/
#          bash scripts/aws/e29_port.sh send H "STAGES" | status H [N] | start|stop|itype H | fetch H
#   BOX stages (in the order sent; a failed stage stops the chain; `final` always runs when sent):
#     rearm    power-off guard (GUARD_MIN from now) and the 30-minute idle stop
#     setup    Docker CE, uv, the branch from code/e29.bundle, .venv (.[all] + mini-swe-agent 2.4.6), .venv-eval
#              (swebench 4.1.0; mswe), Harbor at 4f848e3 (tb, swea), offline E29 tests
#     update   fast-forward the checkout to the latest code/e29.bundle
#     prepull  (mswe) the SWE-bench instance images of every E29 list (mirror.gcr.io first)
#     smoke    smoke task(s) x {baseline, smoke_gated (tau 0), shadow} -> registered checks (development only)
#     cal      the calibration split in shadow mode -> fit -> gate_<H>.json (registration (c))
#     pilot    (tb, mswe) the first 30 evaluation tasks x {baseline, gated} -> summaries/pilot_<H>.json
#     go       (tb, mswe) wait for both pilot summaries in S3, the registered go rule -> go.json; no-go stops the chain
#     waitgo   (swea) wait for both pilot summaries in S3, the same go rule; no-go stops the chain
#     full     every evaluation task x {baseline, gated} -> summaries/full_<H>.json
#     rerun    one re-run of trials with an infrastructure error (both arms), within the cap -> summary again
#     final    spend, leak check, sync, power-off (POWEROFF=1)
# Docker needs the group of a new login: send `rearm setup` alone, then the other stages in a second `send`.
# E29b (Deviations 2026-10-09 ~19:55 UTC, "E29b: step-type routing with a write guard"): the same driver with ROW=e29b
# (S3 prefix e29b/, runs/e29b/<H>, /opt/jev/e29b_status, scripts/explore_e29b_route.py): typed step-kind question,
# write guard, k = 6, typed shadow calibration (strong + one fast shadow call), the E29b caps and go rule, and the
# mini-SWE-agent full run on configs/explore/e29b_route/mswe_full.txt. E29's three hosts are restarted (`start H`);
# E29b chains: tb / mswe "rearm update smoke cal pilot go full rerun final", swea "rearm update waitgo smoke cal full
# rerun final" (send with ROW=e29b in the environment).
# Keys come from SSM Parameter Store into the PROXY process only (GEMINI_API_KEY for the Gemini upstream,
# OPENROUTER_API_KEY for OpenRouter and Jev); Harbor and the agents get a dummy key and the proxy's URL, so no key
# enters a task container. Nothing here prints or writes a key; every stage ends with a leak check (must be 0 files).
set -uo pipefail
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
# hosts of other rows and agents: never touched by this driver (it acts only on the one host tagged jev-e29-<H>)
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
BUNDLE=e29.bundle
GUARD_MIN=${GUARD_MIN:-600}
WAIT_GO_MIN=${WAIT_GO_MIN:-420}
POWEROFF=${POWEROFF:-1}
DUMMY_KEY=e29-the-proxy-holds-the-key
ROW=${ROW:-e29}     # e29 (E29) | e29b (E29b)
case "$ROW" in e29|e29b) ;; *) echo "ROW must be e29 or e29b" >&2; exit 2 ;; esac
if [ "$ROW" = e29b ]; then PYS=scripts/explore_e29b_route.py; LOCAL_BOX=results/explore/e29b_route/box
else PYS=scripts/explore_e29_port.py; LOCAL_BOX=results/explore/e29_port/box; fi

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_PAGER="" aws --region "$REGION" "$@"; }
tag_of() { case "$1" in tb|mswe|swea) echo "jev-e29-$1" ;; *) return 1 ;; esac; }
iid_of() {  # the one live host with the harness's tag
  local tag; tag=$(tag_of "$1") || { echo "harness must be tb, mswe or swea" >&2; return 1; }
  local ids; ids=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$tag" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [ "$(echo "$ids" | wc -w)" = 1 ] || { echo "expected one $tag host, found: '$ids'" >&2; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $ids "* ]] && { echo "refused: $ids is another row's host" >&2; return 1; }
  echo "$ids"
}

if [[ "$MODE" == launch ]]; then
  H=${1:?usage: launch tb|mswe|swea [itype ...]}; shift
  TAG=$(tag_of "$H") || exit 1
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  AMI=${AMI:-$(awsl ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
    --query Parameter.Value --output text)}
  ud=$(mktemp)
  cat > "$ud" <<EOF
#!/bin/bash
shutdown -h +$GUARD_MIN "$TAG: power-off guard ($GUARD_MIN min from boot)"
snap install aws-cli --classic || (apt-get update -qq && apt-get install -y -qq unzip && cd /tmp \\
  && curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o awscliv2.zip && unzip -q awscliv2.zip && ./aws/install)
EOF
  types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(m7i.8xlarge m6i.8xlarge c7i.8xlarge)
  for IT in "${types[@]}"; do
    for s in $SUBNETS; do
      out=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$IT" --subnet-id "$s" --security-group-ids "$SG" \
        --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior stop --count 1 \
        --user-data "file://$ud" \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG},{Key=Project,Value=jev}]" \
          "ResourceType=volume,Tags=[{Key=Name,Value=$TAG},{Key=Project,Value=jev}]" \
        --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name]' --output text 2>&1)
      rc=$?
      echo "$IT $s: $out"
      (( rc == 0 )) && { rm -f "$ud"; exit 0; }
      [[ "$out" == *InsufficientInstanceCapacity* || "$out" == *Unsupported* || "$out" == *"not supported"* ]] \
        || { rm -f "$ud"; exit 1; }
    done
  done
  rm -f "$ud"; exit 1
fi
if [[ "$MODE" == push ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null || { echo "bundle failed"; exit 1; }
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors || exit 1
  git -C "$ROOT" show "$BRANCH:scripts/aws/e29_port.sh" > "$tmp/e29_port.sh"   # the committed driver
  awsl s3 cp "$tmp/e29_port.sh" "$B/code/e29_port.sh" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e29_port.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  H=${1:?usage: send H "STAGES"}; STAGES=${2:?usage: send H "STAGES"}
  IID=$(iid_of "$H") || exit 1
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo "f=/opt/jev/${ROW}_\$(date +%s%N).sh"
    echo "sudo mkdir -p /opt/jev && sudo chown \$(id -u):\$(id -g) /opt/jev"
    echo "aws s3 cp $B/code/e29_port.sh \$f --only-show-errors --region $REGION"
    printf 'export H=%q ROW=%q GUARD_MIN=%q WAIT_GO_MIN=%q POWEROFF=%q\n' "$H" "$ROW" "$GUARD_MIN" "$WAIT_GO_MIN" "$POWEROFF"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/%s_box.log 2>&1 < /dev/null &\n' "$STAGES" "$ROW"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/${ROW}_driver_$H.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/${ROW}_driver_$H.sh /tmp/${ROW}_driver.sh --only-show-errors && bash /tmp/${ROW}_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  H=${1:?usage: status H [N]}
  awsl s3 cp "$B/$ROW/$H/status/e29_box.txt" - 2>/dev/null | tail -n "${2:-30}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  H=${1:?usage: start|stop H}
  IID=$(iid_of "$H") || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  H=${1:?usage: itype H}
  IID=$(iid_of "$H") || exit 1
  awsl ec2 describe-instances --instance-ids "$IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]'
  exit 0
fi
if [[ "$MODE" == fetch ]]; then   # slim records only: status, summaries, gates, proxy logs (gz), Harbor result.json files
  H=${1:?usage: fetch H}
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  d="$ROOT/$LOCAL_BOX"; mkdir -p "$d/$H"
  awsl s3 sync "$B/$ROW/$H/status" "$d/$H/status" --only-show-errors --exclude '*.log' --exclude '*.pid'
  awsl s3 sync "$B/$ROW/summaries" "$d/summaries" --only-show-errors
  awsl s3 sync "$B/$ROW/gates" "$d/gates" --only-show-errors
  awsl s3 cp "$B/$ROW/go.json" "$d/go.json" --only-show-errors 2>/dev/null || true
  awsl s3 sync "$B/$ROW/$H/proxy" "$d/$H/proxy" --only-show-errors --exclude '*' --include '*.jsonl.gz'
  awsl s3 sync "$B/$ROW/$H/runs" "$d/$H/runs" --only-show-errors --exclude '*' --include '*/result.json' \
    --include '*.jev_slim.json' --include '*/eval/*.json' 2>/dev/null
  du -sh "$d/$H"
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,32p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
H=${H:?H=tb|mswe|swea}
W=/opt/jev; R=$W/JevAgentHarness/research; S=$W/${ROW}_status
mkdir -p "$S"
export PATH="$HOME/.local/bin:/snap/bin:$PATH"
STATUS_S3="$B/$ROW/$H/status"
RUNS=$R/runs/$ROW/$H
CFGD=$R/configs/explore/e29_port
BOXLOG=/opt/jev/${ROW}_box.log
if [ "$ROW" = e29b ]; then   # E29b (b): typed question, write guard, k = 6; typed shadow with one fast shadow call
  ARMS_SMOKE='{"smoke_gated": {"mode": "gated", "tau": 0.0, "k": 6, "question": "step_type", "guard": true}, "shadow": {"mode": "shadow", "question": "step_type", "shadow_roles": ["fast"]}}'
  SMOKE_ARMS="smoke_gated shadow"
  ARMS_CAL='{"shadow": {"mode": "shadow", "question": "step_type", "shadow_roles": ["fast"]}}'
  K=6; GATED_EXTRA=', "question": "step_type", "guard": true'
else
  ARMS_SMOKE='{"baseline": {"mode": "strong"}, "smoke_gated": {"mode": "gated", "tau": 0.0, "k": 4}, "shadow": {"mode": "shadow"}}'
  SMOKE_ARMS="baseline smoke_gated shadow"
  ARMS_CAL='{"shadow": {"mode": "shadow"}}'
  K=4; GATED_EXTRA=''
fi
eval_list() {  # the evaluation list of a stage (E29b: mini-SWE-agent's full run is a registered 100-instance subset)
  if [ "$1" = pilot ]; then echo "$CFGD/${H}_pilot.txt"
  elif [ "$ROW" = e29b ] && [ "$H" = mswe ]; then echo "$R/configs/explore/e29b_route/mswe_full.txt"
  else echo "$CFGD/${H}_eval.txt"; fi
}
arms_eval() { echo "{\"baseline\": {\"mode\": \"strong\"}, \"gated\": {\"mode\": \"gated\", \"tau\": $1, \"k\": $K$GATED_EXTRA}}"; }
case "$H" in
  tb)   PORT=8091; FMT=gemini ;;
  swea) PORT=8092; FMT=gemini ;;
  mswe) PORT=8093; FMT=openai ;;
  *) echo "H must be tb, mswe or swea" >&2; exit 2 ;;
esac
# cumulative proxy caps per host (registration (e), (f)): the pilot stages, then the full run
cap_upto() {  # cap_upto <stage>
  if [ "$ROW" = e29b ]; then  # E29b (f), (g): pilot (smoke + calibration + pilot) pi 35 / mini-SWE-agent 5; full +95 / +14; SWE-agent 60
    case "$H:$1" in
      tb:smoke|tb:cal|tb:pilot) echo 35 ;;  tb:*) echo 130 ;;
      mswe:smoke|mswe:cal|mswe:pilot|mswe:prepull) echo 5 ;;  mswe:*) echo 19 ;;
      swea:*) echo 60 ;;
    esac
    return 0
  fi
  case "$H:$1" in
    tb:smoke|tb:cal|tb:pilot) echo 38 ;;  tb:*) echo 138 ;;
    mswe:smoke|mswe:cal|mswe:pilot|mswe:prepull) echo 7 ;;  mswe:*) echo 32 ;;
    swea:*) echo 80 ;;
  esac
}
st() { echo "$(date -u +%FT%TZ) [$ROW:$H] $*" | tee -a "$S/e29_box.txt"
       aws s3 sync "$S" "$STATUS_S3" --only-show-errors --region "$REGION" --exclude '*.pid' || true; }
echo $$ > "$S/e29_driver_$$.pid"
trap 'rm -f "$S/e29_driver_$$.pid"' EXIT

ssm_val() { aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$1" --with-decryption \
              --query Parameter.Value --output text 2>/dev/null; }
or_usage() {  # OpenRouter key usage (the key goes through curl's stdin, never argv / a file / a log)
  local k; k=$(ssm_val OPENROUTER_API_KEY) || return 1; [ -n "$k" ] && [ "$k" != None ] || return 1
  printf 'Authorization: Bearer %s\n' "$k" | curl -s -m 20 -H @- https://openrouter.ai/api/v1/key | jq -r '.data.usage // "?"'
}
leak_check() {  # files under the given paths that contain a key value (count only)
  local n=0 k v
  for v in GEMINI_API_KEY OPENROUTER_API_KEY; do
    k=$(ssm_val "$v") || continue; [ -n "$k" ] && [ "$k" != None ] || continue
    n=$((n + $(grep -rlF -f <(printf '%s\n' "$k") "$@" 2>/dev/null | wc -l)))
  done
  unset k; echo "$n"
}
leak_gate() {  # registration (f): the leak check runs after every stage; a non-zero count stops the chain
  local n; n=$(leak_check "$RUNS" "$S" "$BOXLOG")
  echo "{\"stage\": \"$1\", \"files\": $n, \"time\": \"$(date -u +%FT%TZ)\"}" >> "$S/e29_leaks.jsonl"
  st "leak check after $1: $n file(s) contain a key value"
  [ "$n" = 0 ]
}
spent_all() { (cd "$R" && .venv/bin/python "$PYS" spend "$RUNS"/proxy/* 2>/dev/null | jq -r .total); }
sync_runs() { aws s3 sync "$RUNS" "$B/$ROW/$H/runs" --only-show-errors --region "$REGION" --exclude 'proxy/*' \
                --exclude '*.sqlite' || true
              for f in "$RUNS"/proxy/*/*.jsonl; do [ -f "$f" ] && gzip -kf "$f"; done
              aws s3 sync "$RUNS/proxy" "$B/$ROW/$H/proxy" --only-show-errors --region "$REGION" --exclude '*' \
                --include '*.jsonl.gz' || true; }

stage_rearm() {
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "E29 guard" 2>/dev/null
  cat > "$W/${ROW}_idle_guard.sh" <<EOF
#!/usr/bin/env bash
# E29 idle stop: power off after 30 min with no live driver ($S/*.pid) and no running container
S=$S; idle=0
while sleep 300; do
  alive=0
  for f in "\$S"/*.pid; do pid=\$(cat "\$f" 2>/dev/null); [ -n "\$pid" ] && kill -0 "\$pid" 2>/dev/null && alive=1; done
  if [ "\$alive" = 1 ] || [ -n "\$(sudo -n docker ps -q 2>/dev/null)" ]; then idle=0; else idle=\$((idle + 5)); fi
  if [ "\$idle" -ge 30 ]; then
    echo "\$(date -u +%FT%TZ) [$ROW:idle] 30 min without a driver or container: powering off" >> "\$S/e29_box.txt"
    PATH="/snap/bin:\$PATH" aws s3 sync "\$S" $STATUS_S3 --only-show-errors --region $REGION --exclude '*.pid' || true
    sudo shutdown -h now
  fi
done
EOF
  pgrep -f "${ROW}_idle_guard.sh" >/dev/null || nohup setsid bash "$W/${ROW}_idle_guard.sh" > /dev/null 2>&1 < /dev/null &
  st "rearm: power-off in $GUARD_MIN min, idle stop armed; $(nproc) vCPU, $(free -g | awk '/Mem:/{print $2}') GB RAM, disk $(df -BG / | awk 'NR==2{print $3"/"$2}')"
}

stage_update() {
  aws s3 cp "$B/code/$BUNDLE" "$W/$BUNDLE" --only-show-errors --region "$REGION" || { st "update: no bundle"; return 1; }
  if [ ! -d "$R/.git" ]; then
    git clone -q -b "$BRANCH" "$W/$BUNDLE" "$R" || { st "update: clone failed"; return 1; }
  else
    git -C "$R" checkout -q -- . && git -C "$R" pull -q --ff-only "$W/$BUNDLE" "$BRANCH" || { st "update: pull failed"; return 1; }
  fi
  st "update: $(git -C "$R" log --oneline -1 | cut -c1-72)"
  if [ "$ROW" = e29b ]; then   # E29b: the offline tests of the extended proxy gate the chain
    (cd "$R" && .venv/bin/python -m pytest -q -p no:warnings tests/test_e29_port.py tests/test_e29b_route.py) \
      > "$S/e29_pytest.txt" 2>&1 || { st "update: E29b tests FAILED: $(tail -2 "$S/e29_pytest.txt" | tr '\n' ' ')"; return 1; }
    st "update: E29b tests passed"
  fi
}

stage_setup() {
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq && sudo apt-get install -y -qq git jq unzip build-essential ca-certificates curl >/dev/null \
    || { st "setup: apt failed"; return 1; }
  if ! command -v docker >/dev/null; then
    curl -fsSL https://get.docker.com | sudo sh >/dev/null 2>&1 || { st "setup: docker install failed"; return 1; }
  fi
  sudo usermod -aG docker ubuntu
  sudo mkdir -p /etc/docker
  cat <<'JSON' | sudo tee /etc/docker/daemon.json >/dev/null
{
  "default-address-pools": [{"base": "10.128.0.0/12", "size": 24}],
  "registry-mirrors": ["https://mirror.gcr.io"],
  "max-concurrent-downloads": 8,
  "log-driver": "json-file",
  "log-opts": {"max-size": "50m", "max-file": "2"},
  "features": {"buildkit": true}
}
JSON
  sudo systemctl restart docker || { st "setup: docker restart failed"; return 1; }
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || { st "setup: uv failed"; return 1; }
  uv python install 3.11 3.12 >/dev/null 2>&1 || { st "setup: python failed"; return 1; }
  stage_update || return 1
  cd "$R" || return 1
  [ -x .venv/bin/python ] || uv venv -q -p 3.11 .venv || return 1
  uv pip install -q --python .venv -e ".[all]" "mini-swe-agent==2.4.6" || { st "setup: pip repo failed"; return 1; }
  if [ "$H" = mswe ]; then
    [ -x .venv-eval/bin/python ] || uv venv -q -p 3.11 .venv-eval || return 1
    uv pip install -q --python .venv-eval "swebench==4.1.0" || { st "setup: pip swebench failed"; return 1; }
  else
    [ -x "$W/harbor-venv/bin/harbor" ] || uv venv -q -p 3.12 "$W/harbor-venv" || return 1
    uv pip install -q --python "$W/harbor-venv" \
      "harbor @ git+https://github.com/laude-institute/harbor@4f848e367e6e2627c2c0b1826bb3b20ceab1198f" \
      || { st "setup: pip harbor failed"; return 1; }
    PYTHONPATH="adapters/pi:adapters/swe_agent" "$W/harbor-venv/bin/python" -c \
      'import harbor_jev_pi, harbor_jev_swe_agent; print("harbor agents import ok")' > "$S/e29_harbor_import.txt" 2>&1
  fi
  sudo docker run --rm hello-world >/dev/null 2>&1 || { st "setup: docker run failed"; return 1; }
  .venv/bin/python -m pytest -q tests/test_e29_port.py > "$S/e29_pytest.txt" 2>&1
  st "setup done: $(git log --oneline -1 | cut -c1-60); pytest: $(tail -1 "$S/e29_pytest.txt"); $(cat "$S/e29_harbor_import.txt" 2>/dev/null)"
}

stage_prepull() {  # mswe: the instance images of every E29 list (as E4's driver; mirror.gcr.io first)
  [ "$H" = mswe ] || return 0
  cd "$R" || return 1
  pull() { local img="docker.io/swebench/sweb.eval.x86_64.$(echo "${1//__/_1776_}" | tr 'A-Z' 'a-z'):latest" i
           sudo docker image inspect "$img" >/dev/null 2>&1 && { echo "CACHED $img"; return 0; }
           if sudo docker pull -q "mirror.gcr.io/${img#docker.io/}" >/dev/null 2>&1; then
             sudo docker tag "mirror.gcr.io/${img#docker.io/}" "$img" && sudo docker rmi "mirror.gcr.io/${img#docker.io/}" >/dev/null 2>&1
             echo "PULLED $img (mirror)"; return 0; fi
           for i in 1 2 3 4 5; do sudo docker pull -q "$img" >/dev/null 2>&1 && { echo "PULLED $img"; return 0; }; sleep $((i * 30)); done
           echo "PULL FAILED $img"; }
  export -f pull
  local t0 lists=${1:-"mswe_smoke mswe_cal mswe_pilot"} f; t0=$(date +%s)
  local tag; tag=$(echo "$lists" | tr ' /' '__' | tr -cd 'A-Za-z0-9_.-' | tail -c 80)
  for f in $lists; do case "$f" in /*) cat "$f" ;; *) cat "$CFGD/$f.txt" ;; esac; done | sort -u \
    | xargs -P 6 -I{} bash -c 'pull "$@"' _ {} > "$S/e29_pull_$tag.txt" 2>&1
  f="$S/e29_pull_$tag.txt"
  st "prepull [$lists]: $(( $(date +%s) - t0 )) s; cached $(grep -c ^CACHED "$f"), pulled $(grep -c ^PULLED "$f"), failures $(grep -c 'PULL FAILED' "$f"); disk $(df -BG / | awk 'NR==2{print $3"/"$2}')"
}

# ---------------------------------------------------------------- the proxy
proxy_stop() { pkill -f "jevharness.adapters.route_proxy --config" 2>/dev/null; sleep 2; }
proxy_start() {  # proxy_start <stage> <arms json>: one proxy for the stage, its own log dir, the remaining cap
  local stage=$1 arms=$2 cap spent g o
  proxy_stop
  spent=$(spent_all); spent=${spent:-0}
  cap=$(awk -v c="$(cap_upto "$stage")" -v s="$spent" 'BEGIN{printf "%.4f", c - s}')
  mkdir -p "$RUNS/proxy/$stage"
  g=$(ssm_val GEMINI_API_KEY); o=$(ssm_val OPENROUTER_API_KEY)
  [ -n "$o" ] && [ "$o" != None ] || { st "proxy: no OPENROUTER_API_KEY in SSM"; return 1; }
  if [ "$FMT" = gemini ]; then [ -n "$g" ] && [ "$g" != None ] || { st "proxy: no GEMINI_API_KEY in SSM"; return 1; }; fi
  ( cd "$R" && env GEMINI_API_KEY="$g" OPENROUTER_API_KEY="$o" setsid nohup .venv/bin/python -m jevharness.adapters.route_proxy \
      --config "$CFGD/proxy_$H.json" --set "arms=$arms" --set "cap_usd=$cap" --host 0.0.0.0 --port "$PORT" \
      --log-dir "$RUNS/proxy/$stage" >> "$S/e29_proxy_$stage.log" 2>&1 < /dev/null & )
  unset g o
  for _ in $(seq 1 30); do curl -s --noproxy '*' -m 2 "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 2; done
  local h; h=$(curl -s --noproxy '*' -m 5 "http://127.0.0.1:$PORT/health") || { st "proxy DOWN"; return 1; }
  st "proxy ($stage): $(echo "$h" | jq -c '{arms: (.arms | map_values(.mode + (if .tau != null then "@" + (.tau|tostring) else "" end))), cap_usd, spent_usd, backend: .decision_backend.model}'); host spend before this stage \$$spent"
}
watch_cap() {  # stop the harness runs once the proxy reports the cap (it refuses new episodes from then on)
  local flag=$1
  while sleep 30; do
    [ -f "$flag" ] && return 0
    if curl -s --noproxy '*' -m 5 "http://127.0.0.1:$PORT/health" | jq -e '.capped' >/dev/null 2>&1; then
      st "WATCHDOG: the proxy reached its cap: stopping the runs"
      pkill -INT -f "harbor run" ; pkill -INT -f "jevharness.adapters.mini_swe run"; sleep 60
      pkill -KILL -f "harbor run"; pkill -KILL -f "jevharness.adapters.mini_swe run"
      touch "$S/e29_watchdog_fired"; return 0
    fi
  done
}

# ---------------------------------------------------------------- harness runs (one job per arm, in parallel)
harbor_job() {  # harbor_job <out> <arm> <ids file> <n concurrent> <dataset>
  local out=$1 arm=$2 ids=$3 n=$4 ds=$5 agent inc=() id model
  mkdir -p "$out"
  while IFS= read -r id; do id="${id%%#*}"; id="$(echo "$id" | tr -d '[:space:]')"; [ -n "$id" ] && inc+=(-i "$id" -i "*/$id"); done < "$ids"
  if [ "$H" = tb ]; then
    agent=(-a harbor_jev_pi:JevPi -m google/gemini-3.5-flash --ak version=0.99.2 --ak thinking=minimal --ak max_turns=100
           --ak "llm_proxy=http://host.docker.internal:$PORT/e29/$arm")
  else
    agent=(-a harbor_jev_swe_agent:JevSweAgent -m gemini/gemini-3.5-flash --ak version=v1.1.0 --ak litellm_version=1.82.6
           --ak jev_router=off --ak "llm_proxy=http://host.docker.internal:$PORT/e29/$arm")
    inc=(); while IFS= read -r id; do id="$(echo "${id%%#*}" | tr -d '[:space:]')"; [ -n "$id" ] && inc+=(-i "*$id"); done < "$ids"
  fi
  ( cd "$R" && env -u OPENROUTER_API_KEY GEMINI_API_KEY="$DUMMY_KEY" PYTHONPATH="$R/adapters/pi:$R/adapters/swe_agent" \
      "$W/harbor-venv/bin/harbor" run -d "$ds" "${agent[@]}" -o "$out" -n "$n" -k 1 -y \
      --extra-docker-compose "$R/adapters/pi/harbor_host_gateway.yaml" --job-name "$arm" \
      --max-retries 2 --retry-exclude AgentTimeoutError "${inc[@]}" > "$out/run_$arm.log" 2>&1
    echo "exit=$?" >> "$out/run_$arm.log" )
}
mswe_job() {  # mswe_job <out> <arm> <ids file> <workers>
  local out=$1 arm=$2 ids=$3 w=$4 rep
  mkdir -p "$out/$arm"
  sed "s/__ARM__/$arm/" "$R/adapters/mini_swe/e29_proxy.yaml" > "$out/e29_proxy_$arm.yaml"
  ( cd "$R" && env -u OPENROUTER_API_KEY -u GEMINI_API_KEY MSWEA_SILENT_STARTUP=1 MSWEA_GLOBAL_COST_LIMIT=40 \
      .venv/bin/python -m jevharness.adapters.mini_swe run -c adapters/mini_swe/swebench_jev.yaml \
      -c adapters/mini_swe/s1_jev_openrouter.yaml -c adapters/mini_swe/e4_deepseek.yaml -c "$out/e29_proxy_$arm.yaml" \
      -c environment.pull_timeout=900 --arm llm_only --name "$arm" --subset verified --split test --instance-ids "$ids" \
      -o "$out/$arm" -w "$w" > "$out/run_$arm.log" 2>&1; echo "exit=$?" >> "$out/run_$arm.log" )
  ( d="$out/$arm"; mkdir -p "$d/eval"
    cd "$d/eval" && "$R/.venv-eval/bin/python" -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified \
      --split test --predictions_path "$d/preds.json" --max_workers 5 --run_id "${ROW}_$(basename "$out")_$arm" --report_dir "$d/eval" \
      > "$d/eval/eval.log" 2>&1
    rep=$(ls "$d"/eval/*."${ROW}_$(basename "$out")_$arm".json 2>/dev/null | head -1)
    cd "$R" && .venv/bin/python -m jevharness.adapters.mini_swe collect "$d" ${rep:+--report "$rep"} -o "$out/$arm.jsonl" \
      > "$d/collect.log" 2>&1 )
}
run_arms() {  # run_arms <stage> <out> <ids file> <n> <arm> ...
  local stage=$1 out=$2 ids=$3 n=$4; shift 4
  local pids=() a ds
  case "$H:$stage" in tb:smoke|tb:cal) ds=terminal-bench-pro@1.0 ;; tb:*) ds=terminal-bench@2.0 ;; *) ds=swebench-verified@1.0 ;; esac
  rm -f "$S/e29_${stage}_done"
  watch_cap "$S/e29_${stage}_done" & local wd=$!
  for a in "$@"; do
    if [ "$H" = mswe ]; then mswe_job "$out" "$a" "$ids" "$n" & else harbor_job "$out" "$a" "$ids" "$n" "$ds" & fi
    pids+=($!); sleep 10
  done
  for a in "${pids[@]}"; do wait "$a"; done
  touch "$S/e29_${stage}_done"; kill "$wd" 2>/dev/null
}
summary() {  # summary <stage> <out> <tasks file>   (the stage's re-run, if any, replaces its errored trials)
  local stage=$1 out=$2 tasks=$3 rr=()
  mkdir -p "$S/summaries"
  [ -d "$RUNS/${stage}_rerun" ] && rr=(--rerun-run "$RUNS/${stage}_rerun" --rerun-logs "$RUNS/proxy/${stage}_rerun")
  (cd "$R" && .venv/bin/python "$PYS" summary --harness "$H" --stage "$stage" --run "$out" \
     --logs "$RUNS/proxy/$stage" --tasks "$tasks" "${rr[@]}" --out "$S/summaries/${stage}_$H.json") \
     > "$S/e29_summary_$stage.txt" 2>&1
  aws s3 cp "$S/summaries/${stage}_$H.json" "$B/$ROW/summaries/${stage}_$H.json" --only-show-errors --region "$REGION"
  st "summary $stage: $(tail -1 "$S/e29_summary_$stage.txt")"
}
no_pilot() {  # no_pilot <status>: a pilot summary that says why there is none (the other host's go stage stops waiting)
  mkdir -p "$S/summaries"
  printf '{"harness": "%s", "stage": "pilot", "n": 0, "status": "%s", "no_gate": %s}\n' "$H" "$1" "${2:-false}" \
    > "$S/summaries/pilot_$H.json"
  aws s3 cp "$S/summaries/pilot_$H.json" "$B/$ROW/summaries/pilot_$H.json" --only-show-errors --region "$REGION"
}

disk_pruner() {  # swea: per-trial SWE-bench image builds grow BuildKit's cache by ~5 GB per instance (E25); keep it bounded
  [ "$H" = swea ] || return 0
  cat > "$W/e29_disk_pruner.sh" <<'EOF2'
#!/usr/bin/env bash
while sleep 600; do
  u=$(df --output=pcent / | tail -1 | tr -dc 0-9)
  if [ "$u" -ge 55 ]; then
    sudo docker builder prune -f --keep-storage 150g >/dev/null 2>&1
    sudo docker image prune -f >/dev/null 2>&1
    echo "$(date -u +%FT%TZ) [e29:disk] used ${u}% -> builder cache pruned to 150 GB; now $(df --output=pcent / | tail -1 | tr -dc 0-9)%" >> STATUS_DIR/e29_box.txt
  fi
done
EOF2
  sed -i "s#STATUS_DIR#$S#" "$W/e29_disk_pruner.sh"
  pgrep -f e29_disk_pruner.sh >/dev/null || (nohup setsid bash "$W/e29_disk_pruner.sh" >/dev/null 2>&1 < /dev/null &)
}

# ---------------------------------------------------------------- stages
stage_smoke() {
  cd "$R" || return 1
  disk_pruner
  echo "$(or_usage)" > "$S/e29_or_usage_start.txt"
  st "smoke: OpenRouter key usage at start \$$(cat "$S/e29_or_usage_start.txt")"
  proxy_start smoke "$ARMS_SMOKE" || return 1
  local n=2; [ "$H" = mswe ] && n=1
  # shellcheck disable=SC2086
  run_arms smoke "$RUNS/smoke" "$CFGD/${H}_smoke.txt" "$n" $SMOKE_ARMS
  sync_runs
  local leaks; leaks=$(leak_check "$RUNS" "$S" "$BOXLOG")
  .venv/bin/python "$PYS" smoke --harness "$H" --logs "$RUNS/proxy/smoke" --run "$RUNS/smoke" \
    --leaks "$leaks" --out "$S/e29_smoke_check.json" > "$S/e29_smoke_check.txt" 2>&1
  local rc=$?
  st "smoke checks (exit $rc): $(tail -1 "$S/e29_smoke_check.txt"); spend \$$(spent_all)"
  [ "$rc" = 0 ] || { [ "$H" != swea ] && no_pilot "stopped: smoke failed"; return 1; }
  leak_gate smoke
}
rearm_guard() { sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "E29 guard ($1)" 2>/dev/null; }
stage_cal() {
  cd "$R" || return 1
  rearm_guard cal
  proxy_start cal "$ARMS_CAL" || return 1
  local n=5; [ "$H" = mswe ] && n=10
  st "cal: $(wc -l < "$CFGD/${H}_cal.txt") calibration tasks in shadow mode, $n concurrent"
  run_arms cal "$RUNS/cal" "$CFGD/${H}_cal.txt" "$n" shadow
  proxy_stop
  sync_runs
  mkdir -p "$S/gates"
  .venv/bin/python "$PYS" fit --harness "$H" --logs "$RUNS/proxy/cal" --out "$S/gates/gate_$H.json" \
    > "$S/e29_fit.txt" 2>&1 || { st "fit FAILED: $(tail -3 "$S/e29_fit.txt")"; [ "$H" != swea ] && no_pilot "stopped: fit failed"; return 1; }
  aws s3 cp "$S/gates/gate_$H.json" "$B/$ROW/gates/gate_$H.json" --only-show-errors --region "$REGION"
  st "fit: $(tail -1 "$S/e29_fit.txt"); spend \$$(spent_all)"
  leak_gate cal
}
gate_tau() { jq -r '.tau // "null"' "$S/gates/gate_$H.json" 2>/dev/null; }
stage_eval() {  # stage_eval pilot|full
  local stage=$1 ids n tau
  cd "$R" || return 1
  rearm_guard "$stage"
  tau=$(gate_tau)
  [ -n "$tau" ] || { st "$stage: no gate file"; return 1; }
  if [ "$tau" = null ]; then
    st "$stage: the fitted gate routes nothing (registration (c) 3): the gated arm is not run"
    [ "$stage" = pilot ] && no_pilot "the gate kept every step on the strong LLM" true
    return 0
  fi
  ids=$(eval_list "$stage")
  case "$H:$stage" in tb:*) n=6 ;; mswe:pilot) n=8 ;; mswe:full) n=10 ;; swea:*) n=6 ;; esac
  proxy_start "$stage" "$(arms_eval "$tau")" || return 1
  st "$stage: $(wc -l < "$ids") tasks x {baseline, gated}, $n concurrent per arm, tau $tau"
  run_arms "$stage" "$RUNS/$stage" "$ids" "$n" baseline gated
  proxy_stop
  sync_runs
  summary "$stage" "$RUNS/$stage" "$ids"
  leak_gate "$stage"
}
stage_pilot() { stage_eval pilot || { no_pilot "stopped: pilot failed"; return 1; }; }
stage_full() { stage_eval full; }
go_decision() {  # wait for both pilot summaries in S3, then the registered rule (same inputs -> same decision everywhere)
  local deadline=$(( $(date +%s) + WAIT_GO_MIN * 60 )) h
  mkdir -p "$S/summaries"
  while :; do
    for h in tb mswe; do aws s3 cp "$B/$ROW/summaries/pilot_$h.json" "$S/summaries/pilot_$h.json" --only-show-errors --region "$REGION" 2>/dev/null; done
    [ -f "$S/summaries/pilot_tb.json" ] && [ -f "$S/summaries/pilot_mswe.json" ] && break
    [ "$(date +%s)" -lt "$deadline" ] || { st "go: no pilot summary of both harnesses after $WAIT_GO_MIN min: no-go"; return 1; }
    sleep 120
  done
  (cd "$R" && .venv/bin/python "$PYS" go --tb "$S/summaries/pilot_tb.json" \
     --mswe "$S/summaries/pilot_mswe.json" --out "$S/go.json") > "$S/e29_go.txt" 2>&1
  local rc=$?
  aws s3 cp "$S/go.json" "$B/$ROW/go_$H.json" --only-show-errors --region "$REGION"
  [ "$H" = tb ] && aws s3 cp "$S/go.json" "$B/$ROW/go.json" --only-show-errors --region "$REGION"
  st "go: $(jq -c '{decision, per_harness}' "$S/go.json")"
  return $rc
}
stage_go() {  # mswe: the evaluation list's images are pulled while this host waits for the other pilot
  local pp=""
  if [ "$H" = mswe ]; then (stage_prepull "$( [ "$ROW" = e29b ] && eval_list full || echo mswe_eval)") & pp=$!; fi
  go_decision; local rc=$?
  [ -n "$pp" ] && wait "$pp"
  return $rc
}
stage_waitgo() {
  local deadline=$(( $(date +%s) + GUARD_MIN * 60 ))
  (  # keep the power-off guard ahead while this host waits for the pilots
    while [ ! -f "$S/e29_waitgo_done" ]; do sleep 600; sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "E29 guard (waitgo)" 2>/dev/null; done ) &
  local hold=$!
  go_decision; local rc=$?
  touch "$S/e29_waitgo_done"; kill "$hold" 2>/dev/null
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "E29 guard" 2>/dev/null
  return $rc
}
stage_rerun() {  # one re-run of the trials with an infrastructure error (E25 / E4r), both arms, within the cap
  cd "$R" || return 1
  local f="$S/summaries/full_$H.json" tau lst ids
  rearm_guard rerun
  [ -f "$f" ] || { st "rerun: no full summary"; return 0; }
  [ -f "$S/e29_watchdog_fired" ] && { st "rerun: the watchdog fired: nothing is re-run"; return 0; }
  lst=$(jq -r '[.per_task[] | to_entries[] | select(.value.infra_error and (.value.cancelled | not)) | .key] | unique | .[]' "$f")
  [ -n "$lst" ] || { st "rerun: no infrastructure error"; return 0; }
  ids="$S/e29_rerun_ids.txt"; echo "$lst" > "$ids"
  tau=$(gate_tau)
  proxy_start full_rerun "$(arms_eval "$tau")" || return 1
  st "rerun: $(wc -l < "$ids") task(s) in both arms: $(tr '\n' ' ' < "$ids")"
  run_arms full_rerun "$RUNS/full_rerun" "$ids" 5 baseline gated
  proxy_stop
  sync_runs
  summary full "$RUNS/full" "$(eval_list full)"
  leak_gate rerun
}
stage_final() {
  cd "$R" 2>/dev/null || true
  proxy_stop
  local u0 u1 n
  u0=$(cat "$S/e29_or_usage_start.txt" 2>/dev/null); u1=$(or_usage)
  (cd "$R" && .venv/bin/python "$PYS" spend "$RUNS"/proxy/*) > "$S/e29_spend_logged.json" 2>/dev/null
  printf '{"harness": "%s", "logged": %s, "or_key_usage_start": "%s", "or_key_usage_end": "%s", "written": "%s"}\n' "$H" \
    "$(cat "$S/e29_spend_logged.json" 2>/dev/null || echo null)" "${u0:-}" "${u1:-}" "$(date -u +%FT%TZ)" > "$S/e29_spend.json"
  sync_runs
  n=$(leak_check "$RUNS" "$S" "$BOXLOG")
  echo "{\"stage\": \"final\", \"files\": $n, \"time\": \"$(date -u +%FT%TZ)\"}" >> "$S/e29_leaks.jsonl"
  st "final: spend $(cat "$S/e29_spend.json"); leak check: $n file(s) contain a key value"
  cp "$BOXLOG" "$S/e29_box_log.txt" 2>/dev/null
  st "final: synced"
  if [ "$POWEROFF" = 1 ]; then st "powering off"; rm -f "$S/e29_driver_$$.pid"; sudo shutdown -h +1 "E29 done"; fi
}

st "stages: $STAGES"
failed=0
for stg in $STAGES; do
  [ "$stg" = final ] && continue
  if [ "$failed" = 0 ]; then
    if declare -F "stage_$stg" >/dev/null; then
      "stage_$stg" || { failed=1; st "stage $stg FAILED or stopped the chain"; }
    else
      failed=1; st "unknown stage $stg"
    fi
  fi
done
[[ " $STAGES " == *" final "* ]] && stage_final
st "chain end (failed=$failed)"
