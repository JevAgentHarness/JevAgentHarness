#!/usr/bin/env bash
# Exploratory row E7, JevGym-Hard (docs/PREREG_A7_STAGE2.md, Deviations entry of 2026-10-06 "Exploratory row E7
# (JevGym-Hard)"; descriptive, outside every Holm table). One NEW GPU host launched from the E3 clone image (tag
# Name=jev-e7-hard): the frozen JevAgentRL-4B (sha256-checked against results/stage2/rl_eval_plan.json's frozen row)
# behind training/serve_jev.py with the plan's serve_args (S2-4's server) on :8081, and the runner with the three arms
# task-major in one pool on the same host. System-2 and Jev via OpenRouter (keys from SSM into the stage's process only).
#
#   LOCAL:  bash scripts/aws/e7_hard.sh launch [itype]      # one new instance (refuses if a jev-e7-hard host exists)
#           E7_IID=i-... bash scripts/aws/e7_hard.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages (in the order sent; a failed stage stops the chain, `final` always runs when sent):
#     rearm   idle stop (IDLE_MIN) + power-off guard (GUARD_MIN, or the absolute GUARD_END: the GPU cap), host, GPU
#     setup   the committed branch (bundle) in /opt/jev/stage2, the venv imports this checkout, the frozen 4B verified
#             (fetched from S3 if absent)
#     serve   serve_jev with the frozen row on :8081 (waits for /health); START record
#     smoke   configs/explore/e7_hard/smoke.yaml (the three arms on 2 development tasks), then the registered projection
#     eval    configs/explore/e7_hard/eval.yaml at the n the projection allows, three passes (passes 2-3 re-run
#             infrastructure errors only) under the $ watchdog (CAP_USD minus PRIOR_USD, the pilot's spend); the
#             evaluation window (tariff) is recorded in eval_window.json
#     cal     (EXP=e7c) the calibration shadow arms of configs/explore/e7_hard/cal_c.yaml under the $ watchdog
#     final   stop the server, sync, power-off (POWEROFF=1)
# Records: runs/e7_hard{,_smoke} and results/explore/e7_hard/box on the box -> s3://$BUCKET/e7_hard/.
# EXP=e7b (send-time): the addendum E7b (configs/explore/e7_hard/{smoke,eval}_b.yaml, runs/e7b_hard{,_smoke},
# results/explore/e7_hard/box_e7b -> s3://$BUCKET/e7_hard/e7b/; its own $ cap via CAP_USD; Deviations "E7b").
# EXP=e7f: E7d's continued checkpoint (AgentDecision-4B (tier), sha256-checked against box_e7d/tier_sha256.txt and its
# temperatures against TIER_CAL_SHA) served on :8081 instead of the frozen 4B; Flash on the direct Gemini API
# (configs/explore/e7_hard/{smoke,eval}_e7f.yaml; Deviations 2026-10-07 "E7f: AgentDecision-4B (tier) live").
# EXP=e7fx: the E7f extension (Deviations 2026-10-07 "E7f extension"): as e7f on seeds 310680-310799 (eval_e7fx.yaml),
# no smoke; stage `xproj` writes the gate's projection from the first part's records (runs/e7f_hard, from S3 if absent).
# EXP=e19 (Deviations 2026-10-08 "E19: the rule-randomised model on DecisionGym-Hard"): E12b's checkpoint
# "AgentDecision-4B (random rules)" (sha256-checked against results/explore/e12/box/b/e12b_sha256.txt, its temperatures
# against RND_CAL_SHA) served on :8081 as agentdecision-4b-random on ONE NEW g5.2xlarge (tag jev-e19-hard, launched
# with E12's zone fallback; `launch` with EXP=e19); stage `refit` = the registered calibration-arm estimator on E7c's
# calibration states re-answered by the served model (E7d's 20 tasks, $0); the one arm `rl_random` on E7f's 200 tasks
# (configs/explore/e7_hard/{smoke,eval}_e19.yaml, Flash on the direct Gemini API), paired with E7f's logged arms;
# smoke and eval are probe-gated and pause on API overload (E14's rule; scripts/explore_e19_hard_random.py).
# Records: results/explore/e19_hard_random/box and runs/e19_hard{,_smoke} -> s3://$BUCKET/e7_hard/e19/.
set -uo pipefail
for v in $(compgen -e | grep -E '_API_KEYS?$'); do unset "$v"; done   # no key in this process tree, local or box
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
E7_IID=${E7_IID:-}
# E3's hosts, E5's hosts, the Stage-2 host: never touched by this driver
FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
AMI=${AMI:?set AMI to a GPU image id}; SUBNET=<aws-resource-id>; SG=${SG:?set SG to a security group id}; PROFILE=${PROFILE:?set PROFILE to an instance profile}
TAG=jev-e7-hard
BUNDLE=e7_hard.bundle
S3OUT="$B/e7_hard"
CAP_USD=${CAP_USD:-150}
PRIOR_USD=${PRIOR_USD:-0}
GUARD_MIN=${GUARD_MIN:-330}
IDLE_MIN=${IDLE_MIN:-30}
WORKERS=${WORKERS:-12}
POWEROFF=${POWEROFF:-1}
EXP=${EXP:-e7}                      # e7, or e7b (the addendum: 12 customers, seeds 310200-310399, its own records)
RUN_CAL=""
case "$EXP" in
  e7)  RUN_EVAL=e7_hard;  RUN_SMOKE=e7_hard_smoke;  SUF="";      ENTRY='row E7 (JevGym-Hard)' ;;
  e7b) RUN_EVAL=e7b_hard; RUN_SMOKE=e7b_hard_smoke; SUF="_b";    ENTRY='E7b (JevGym-Hard, 12 customers)' ;;
  e7c) RUN_EVAL=e7c_hard; RUN_SMOKE=e7c_hard_smoke; SUF="_c";    ENTRY='E7c (JevGym-Hard, refitted thresholds)'
       RUN_CAL=e7c_cal ;;
  e7cd) RUN_EVAL=e7cd_hard; RUN_SMOKE=e7cd_hard_smoke; SUF="_cd"; ENTRY='E7c direct-API row' ;;   # Flash direct API
  e7d) RUN_EVAL=e7d_harvest; RUN_SMOKE=e7d_none; SUF="_d"; ENTRY='the recipe on the new rule' ;;  # GPU only
  e7f) RUN_EVAL=e7f_hard; RUN_SMOKE=e7f_hard_smoke; SUF="_e7f"; ENTRY='E7f: AgentDecision-4B (tier) live' ;;  # direct API
  e7fx) RUN_EVAL=e7fx_hard; RUN_SMOKE=e7fx_none; SUF="_e7fx"; ENTRY='"E7f extension"' ;;           # direct API, no smoke
  e19) RUN_EVAL=e19_hard; RUN_SMOKE=e19_hard_smoke; SUF="_e19"; ENTRY='E19: the rule-randomised model on DecisionGym-Hard' ;;
  e19x) RUN_EVAL=e19x_hard; RUN_SMOKE=e19x_none; SUF="_e19x"; ENTRY='E19 extension: the remaining 20 tasks' ;;  # no smoke
  *) echo "EXP must be e7, e7b, e7c, e7cd, e7d, e7f, e7fx, e19 or e19x"; exit 1 ;;
esac
if [[ "$EXP" == e7 ]]; then S3X="$S3OUT"; else S3X="$S3OUT/$EXP"; fi
DRIVER=e7_hard.sh; SENDER=e7_driver.sh          # the S3 keys of the committed driver and of the send stub
E19=0; [[ "$EXP" == e19 || "$EXP" == e19x ]] && E19=1
if (( E19 )); then   # E19 (and its extension): its own NEW host (never E7's or any other project host), its own S3 keys
  TAG=jev-e19-hard; DRIVER=e7_hard_e19.sh; SENDER=e19_driver.sh
  FORBIDDEN_IIDS=${FORBIDDEN_IIDS:-}
fi
# us-east-1d first (E3's zone), then the other zones of the same VPC (E12's launch; E19 only)
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws --region "$REGION" "$@"; }
iid_ok() {
  [[ "$E7_IID" =~ ^i-[0-9a-f]{8,17}$ ]] || { echo "E7_IID is not set to an instance id"; return 1; }
  [[ " $FORBIDDEN_IIDS " == *" $E7_IID "* ]] && { echo "refused: $E7_IID is not the E7 host"; return 1; }
  if (( E19 )); then   # E19 acts only on a host it launched (tag jev-e19-hard)
    local tag
    tag=$(awsl ec2 describe-instances --instance-ids "$E7_IID" --query 'Reservations[0].Instances[0].Tags[?Key==`Name`]|[0].Value' --output text 2>/dev/null)
    [[ "$tag" == "$TAG" ]] || { echo "refused: $E7_IID is tagged '$tag', not $TAG"; return 1; }
  fi
  return 0
}

if [[ "$MODE" == launch ]]; then
  IT=${1:-g5.4xlarge}
  have=$(awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
         "Name=instance-state-name,Values=pending,running,stopping,stopped" --query 'Reservations[].Instances[].InstanceId' --output text)
  [[ -z "$have" ]] || { echo "a $TAG host exists already: $have"; exit 1; }
  if [[ "$EXP" == e19 ]]; then   # one g5.2xlarge (default), us-east-1d then the other zones; user-data arms the power-off
    # guard GUARD_MIN minutes after launch (the 4 host-hour cap: GUARD_MIN=230)
    ud=$(mktemp)
    printf '#!/bin/bash\nshutdown -h +%s "jev-e19-hard: power-off guard (4 host-hours from launch)"\n' "$GUARD_MIN" > "$ud"
    types=("$@"); [[ ${#types[@]} -gt 0 ]] || types=(g5.2xlarge)
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
  awsl ec2 run-instances --image-id "$AMI" --instance-type "$IT" --subnet-id "$SUBNET" --security-group-ids "$SG" \
    --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior stop --count 1 \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
    --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name]' --output text
  exit $?
fi
if [[ "$MODE" == push ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null || { echo "bundle failed"; exit 1; }
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors || exit 1
  git -C "$ROOT" show "$BRANCH:scripts/aws/e7_hard.sh" > "$tmp/e7_hard.sh"   # the committed driver
  awsl s3 cp "$tmp/e7_hard.sh" "$B/code/$DRIVER" --only-show-errors || exit 1
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/e7_hard.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  iid_ok || exit 1
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/e7_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/$DRIVER \$f --only-show-errors --region $REGION"
    printf 'export CAP_USD=%q PRIOR_USD=%q GUARD_MIN=%q GUARD_END=%q IDLE_MIN=%q WORKERS=%q POWEROFF=%q E7_IID=%q EXP=%q PROBE_WAIT_MIN=%q\n' \
      "$CAP_USD" "$PRIOR_USD" "$GUARD_MIN" "${GUARD_END:-}" "$IDLE_MIN" "$WORKERS" "$POWEROFF" "$E7_IID" "$EXP" "${PROBE_WAIT_MIN:-60}"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/e7_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/$SENDER" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$E7_IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/$SENDER /tmp/$SENDER --only-show-errors && bash /tmp/$SENDER'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3X/box/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  iid_ok || exit 1
  awsl ec2 "$MODE-instances" --instance-ids "$E7_IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  iid_ok || exit 1
  awsl ec2 describe-instances --instance-ids "$E7_IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,22p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
W=/opt/jev
cd "$W" || exit 1
imds() {
  curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" "http://169.254.169.254/latest/meta-data/$1"
}
MY_IID=$(imds instance-id)
if [[ " $FORBIDDEN_IIDS " == *" $MY_IID "* || ( -n "$E7_IID" && "$MY_IID" != "$E7_IID" ) ]]; then
  echo "$(date -u +%FT%TZ) refused: this host is $MY_IID, not the E7 host ${E7_IID:-?}"; exit 1
fi
R=$W/stage2
if [[ "$EXP" == e7 ]]; then A=$R/results/explore/e7_hard/box
elif [[ "$EXP" == e19 ]]; then A=$R/results/explore/e19_hard_random/box
elif [[ "$EXP" == e19x ]]; then A=$R/results/explore/e19_hard_random/box_e19x
else A=$R/results/explore/e7_hard/box_$EXP; fi
STATUS_DIR=$W/${EXP}_status
HV=$W/hvenv
CK=$W/ck
PLAN=results/stage2/rl_eval_plan.json
PORT=8081
SMOKE_CFG=configs/explore/e7_hard/smoke$SUF.yaml
CAL_CFG=configs/explore/e7_hard/cal$SUF.yaml
EVAL_CFG=configs/explore/e7_hard/eval$SUF.yaml
STOP=runs/${EXP}_STOP.json
[[ "$EXP" == e19x ]] && STOP=runs/e19_STOP.json     # one $ cap over E19's smoke, first part and extension
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$STATUS_DIR"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$STATUS_DIR/status.txt"
  [[ -d "$A" ]] && cp -f "$STATUS_DIR/status.txt" "$A/status.txt"
  aws s3 cp "$STATUS_DIR/status.txt" "$S3X/box/status.txt" --only-show-errors --region "$REGION" || true
}
sync_out() {
  [[ -d "$A" ]] && aws s3 sync "$A" "$S3X/box" --only-show-errors --region "$REGION" --exclude 'data/D100.jsonl' \
    --exclude 's23/corpus.json' --exclude 's23/occurrences.jsonl.gz' --exclude 's23/questions.jsonl.gz' \
    --exclude 'cal/cal_requests.jsonl.gz' 
  local d
  for d in "runs/$RUN_EVAL" "runs/$RUN_SMOKE" ${RUN_CAL:+"runs/$RUN_CAL"}; do
    [[ -d "$R/$d" ]] && aws s3 sync "$R/$d" "$S3X/$d" --only-show-errors --region "$REGION" --exclude '*.sqlite'
  done
  true
}
PY() { PYTHONPATH="$R/src" "$HV/bin/python" "$@"; }
plan() { (cd "$R" && PY -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"); }
keys() {  # System-2 key(s) from SSM into this stage's environment only (never on disk, never printed): OpenRouter, or
  # for EXP=e7cd the direct Gemini API key(s)
  local val v
  local names=(OPENROUTER_API_KEY)
  [[ "$EXP" == e7cd || "$EXP" == e7f || "$EXP" == e7fx || "$EXP" == e19 || "$EXP" == e19x ]] && names=(GEMINI_API_KEY GEMINI_API_KEYS)
  for v in "${names[@]}"; do
    val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption \
          --query Parameter.Value --output text 2>/dev/null) || val=""
    [[ -n "$val" && "$val" != None ]] && export "$v=$val"
  done
  unset val
  KEYVAL=${OPENROUTER_API_KEY:-${GEMINI_API_KEY:-}}
  [[ -n "$KEYVAL" ]] || { echo "${names[0]} missing in SSM"; return 1; }
}
leak_check() {  # no key value in any record or status file
  local n k ks=${GEMINI_API_KEYS:-}
  n=0
  for k in ${OPENROUTER_API_KEY:-} ${GEMINI_API_KEY:-} ${ks//,/ }; do
    [[ ${#k} -ge 16 ]] || continue
    n=$(( n + $(grep -rlF -- "$k" "$R/runs/$RUN_EVAL" "$R/runs/$RUN_SMOKE" ${RUN_CAL:+"$R/runs/$RUN_CAL"} "$A" \
      "$STATUS_DIR" 2>/dev/null | wc -l) ))
  done
  echo "key-leak files: $n"
  [[ "$n" == 0 ]]
}

stage_rearm() {
  if [[ -n "${GUARD_END:-}" ]]; then   # an absolute end (UTC, e.g. "2026-10-06 20:45"): the GPU cap counts from launch
    GUARD_MIN=$(( ( $(date -u -d "$GUARD_END" +%s) - $(date -u +%s) ) / 60 ))
  fi
  (( GUARD_MIN > 0 )) || { echo "guard end already passed"; return 1; }
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "jev-e7-hard: power-off guard (GPU cap)" 2>/dev/null
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without an E7 chain / server / runner process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'e7_[0-9]|jevharness|jevbench|serve_jev|explore_e7_hard|explore_e19|e5_cal_replay|explore_e7d' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-e7-hard: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  sudo rm -f /etc/cron.d/jev-idle-stop-* 2>/dev/null
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
  echo "guard $GUARD_MIN min, idle $IDLE_MIN min; host $MY_IID $(imds instance-type)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}
stage_setup() {
  local myb="$W/e7.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  [[ -d "$R" ]] || git clone -q -b "$BRANCH" "$myb" "$R"
  cd "$R" || return 1
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD || return 1
  rm -f "$myb"
  mkdir -p "$A" runs
  st "code at $(git log --oneline -1 | cut -c1-72)"
  PY -c "import jevharness, sys; p = jevharness.__file__; print(p); sys.exit(0 if p.startswith('$R/src') else 1)" \
    || { echo "the venv does not import this checkout"; return 1; }
  PY -c "from jevharness.bench.suites.jevgym_hard import MULTI_RULES; assert 'tier' in MULTI_RULES" || return 1
  grep -q "$ENTRY" docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for $EXP"; return 1; }   # one line of the entry
  if (( E19 )); then   # E19: the random-rules checkpoint (the frozen 4B is not used); a new EBS volume loads
    # lazily from its snapshot, so the base model and the venv are read once in the background (E16's prewarm)
    mkdir -p "$A/cal" "$A/logs"
    ( prewarm "$(cat "$W/base_model_path.txt")" "$HV/lib" ) > /dev/null 2>&1 &
    if ! verify_random; then
      st "the random-rules checkpoint is absent or different in $RND_CK; fetching $B/checkpoints/e12b_random/"
      aws s3 sync "$B/checkpoints/e12b_random/" "$RND_CK" --only-show-errors --region "$REGION" || return 1
      verify_random || return 1
    fi
  elif ! verify_4b; then
    st "the frozen 4B is absent or different in $CK; fetching $(plan "d['rows']['frozen']['s3']")"
    aws s3 sync "$(plan "d['rows']['frozen']['s3']")" "$CK/$(plan "d['rows']['frozen']['dir']")" --only-show-errors \
      --region "$REGION" || return 1
    verify_4b || return 1
  fi
  if [[ "$EXP" == e7f || "$EXP" == e7fx ]] && ! verify_tier; then
    st "the tier checkpoint is absent or different in $D_CK; fetching $B/checkpoints/e7d_tier/"
    aws s3 sync "$B/checkpoints/e7d_tier/" "$D_CK" --only-show-errors --region "$REGION" || return 1
    verify_tier || return 1
  fi
  [[ -s "$W/base_model_path.txt" ]] || { echo "no base_model_path.txt"; return 1; }
  cp -f "$W/base_model_path.txt" "$A/base_model_path.txt"
  PY -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'cuda', torch.cuda.is_available())" > "$A/env.txt" 2>&1
  cat "$A/env.txt"
}
verify_4b() {
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
TIER_SHA=results/explore/e7_hard/box_e7d/tier_sha256.txt                   # E7d's record of the continued adapter
TIER_CAL=results/explore/e7_hard/box_e7d/cal/tier_calibration.json          # E7d's refitted temperatures (committed)
TIER_CAL_SHA=8d7e0b780e75c6925cb8ccfb5d28d573200be085620bb7ece6cdcea093efd865
verify_tier() {  # E7f: the continued adapter in $D_CK matches E7d's tier_sha256.txt; the temperatures match TIER_CAL_SHA
  (cd "$R" && PY - "$TIER_SHA" "$D_CK" "$TIER_CAL" "$TIER_CAL_SHA" "$A/tier_verify.json" <<'EOF'
import hashlib, json, os, sys
rec, ck, cal, cal_sha, out = sys.argv[1:6]
res, bad = {}, []
for line in open(rec):
    sha, path = line.split()
    p = os.path.join(ck, os.path.basename(path))
    got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    res[os.path.basename(path)] = {"expected": sha, "got": got, "ok": got == sha}
    bad += [p] if got != sha else []
got = hashlib.sha256(open(cal, "rb").read()).hexdigest() if os.path.exists(cal) else None
res[cal] = {"expected": cal_sha, "got": got, "ok": got == cal_sha}
bad += [cal] if got != cal_sha else []
json.dump({"ok": not bad and len(res) == 3, "mismatch": bad, "files": res}, open(out, "w"), indent=1)
print("tier checkpoint sha256:", "all match" if not bad and len(res) == 3 else f"MISMATCH {bad}")
sys.exit(1 if bad or len(res) != 3 else 0)
EOF
  )
}
RND_CK=$W/ck_e12b/random                                                      # E12b's checkpoint: AgentDecision-4B (random rules)
RND_SHA=results/explore/e12/box/b/e12b_sha256.txt                             # E12b's record of the adapter
RND_CAL=results/explore/e12/box/b/cal/e12b_calibration.json                   # E12b's fitted temperatures (committed)
RND_CAL_SHA=aa035e01c4ef703754219059744e0b751df3bb4856e0a39c8618dd10b22a8a78
RND_NAME=agentdecision-4b-random
verify_random() {  # E19: the adapter in $RND_CK matches E12b's e12b_sha256.txt; the temperatures match RND_CAL_SHA
  (cd "$R" && PY - "$RND_SHA" "$RND_CK" "$RND_CAL" "$RND_CAL_SHA" "$A/random_verify.json" <<'EOF'
import hashlib, json, os, sys
rec, ck, cal, cal_sha, out = sys.argv[1:6]
res, bad = {}, []
for line in open(rec):
    sha, path = line.split()
    p = os.path.join(ck, os.path.basename(path))
    got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    res[os.path.basename(path)] = {"expected": sha, "got": got, "ok": got == sha}
    bad += [p] if got != sha else []
got = hashlib.sha256(open(cal, "rb").read()).hexdigest() if os.path.exists(cal) else None
res[cal] = {"expected": cal_sha, "got": got, "ok": got == cal_sha}
bad += [cal] if got != cal_sha else []
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"ok": not bad and len(res) == 3, "mismatch": bad, "files": res}, open(out, "w"), indent=1)
print("random-rules checkpoint sha256:", "all match" if not bad and len(res) == 3 else f"MISMATCH {bad}")
sys.exit(1 if bad or len(res) != 3 else 0)
EOF
  )
}
prewarm() {  # read files in 64 MB chunks, 32 at a time (a new EBS volume loads lazily from its snapshot; E16's helper)
  local d
  for d in "$@"; do
    find -L "$d" -type f -size +1M 2>/dev/null | while read -r f; do
      local s n i; s=$(stat -L -c %s "$f"); n=$(( (s + 67108863) / 67108864 ))
      for i in $(seq 0 $((n - 1))); do echo "$f $i"; done
    done
  done | xargs -P 32 -n 2 sh -c 'dd if="$0" of=/dev/null bs=64M skip=$1 count=1 status=none'
}
serve_random() {  # E19: AgentDecision-4B (random rules) on :8081 as agentdecision-4b-random (no extra arguments, as E7f / E12)
  verify_random || return 1
  stop_server
  local args=(--model "$(cat "$W/base_model_path.txt")" --name "$RND_NAME" --port "$PORT" --adapter "$RND_CK"
              --calibration "$R/$RND_CAL")
  echo "serve_jev ${args[*]}" > "$A/serve.cmd"
  nohup env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$A/serve.out" 2>&1 &
  echo $! > "$W/e7_serve.pid"
  for _ in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:$PORT/health" > "$A/serve_health.json"; then
      curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/serve_models.json"
      grep -q "\"$RND_NAME\"" "$A/serve_models.json" || { echo "served model id is not $RND_NAME"; return 1; }
      printf '{"server_up": "%s", "instance_type": "%s", "instance_id": "%s"}\n' "$(date -u +%FT%TZ)" \
        "$(imds instance-type)" "$MY_IID" > "$A/serve_window_start.json"
      return 0
    fi
    kill -0 "$(cat "$W/e7_serve.pid")" 2>/dev/null || { echo "serve_jev died"; tail -20 "$A/serve.out"; return 1; }
    sleep 5
  done
  echo "serve_jev did not come up"; return 1
}
stop_server() {
  local p
  if [[ -f "$W/e7_serve.pid" ]]; then
    while read -r p; do [[ -n "$p" ]] && kill "$p" 2>/dev/null; done < "$W/e7_serve.pid"
  fi
  rm -f "$W/e7_serve.pid"; sleep 5
}
stage_serve() {
  cd "$R" || return 1
  [[ "$EXP" == e7f || "$EXP" == e7fx ]] && { serve_tier; return $?; }
  (( E19 )) && { serve_random; return $?; }
  verify_4b || return 1
  local adapter cal name
  adapter=$(plan "d['rows']['frozen'].get('adapter') or ''")
  cal=$(plan "d['rows']['frozen'].get('calibration') or ''")
  name=$(plan "d['rows']['frozen']['served_name']")
  [[ "$name" == jevagentrl-4b ]] || { echo "served name $name != jevagentrl-4b"; return 1; }
  stop_server
  local args=(--model "$(cat "$W/base_model_path.txt")" --name "$name" --port "$PORT" --adapter "$CK/$adapter"
              --calibration "$CK/$cal")
  # shellcheck disable=SC2207
  args+=($(plan "' '.join(d.get('serve_args') or [])"))     # S2-4's server (the plan's serve_args)
  echo "serve_jev ${args[*]}" > "$A/serve.cmd"
  nohup env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$A/serve.out" 2>&1 &
  echo $! > "$W/e7_serve.pid"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" > "$A/serve_health.json"; then
      curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/serve_models.json"
      printf '{"server_up": "%s", "instance_type": "%s", "instance_id": "%s"}\n' "$(date -u +%FT%TZ)" \
        "$(imds instance-type)" "$MY_IID" > "$A/serve_window_start.json"
      return 0
    fi
    kill -0 "$(cat "$W/e7_serve.pid")" 2>/dev/null || { echo "serve_jev died"; tail -20 "$A/serve.out"; return 1; }
    sleep 5
  done
  echo "serve_jev did not come up"; return 1
}
serve_tier() {  # E7f: AgentDecision-4B (tier) on :8081 as agentdecision-4b-tier (E7d's serving: no extra arguments)
  verify_tier || return 1
  stop_server
  local args=(--model "$(cat "$W/base_model_path.txt")" --name agentdecision-4b-tier --port "$PORT" --adapter "$D_CK"
              --calibration "$R/$TIER_CAL")
  echo "serve_jev ${args[*]}" > "$A/serve.cmd"
  nohup env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$A/serve.out" 2>&1 &
  echo $! > "$W/e7_serve.pid"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" > "$A/serve_health.json"; then
      curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/serve_models.json"
      grep -q '"agentdecision-4b-tier"' "$A/serve_models.json" || { echo "served model id is not agentdecision-4b-tier"; return 1; }
      printf '{"server_up": "%s", "instance_type": "%s", "instance_id": "%s"}\n' "$(date -u +%FT%TZ)" \
        "$(imds instance-type)" "$MY_IID" > "$A/serve_window_start.json"
      return 0
    fi
    kill -0 "$(cat "$W/e7_serve.pid")" 2>/dev/null || { echo "serve_jev died"; tail -20 "$A/serve.out"; return 1; }
    sleep 5
  done
  echo "serve_jev did not come up"; return 1
}
run_watched() {  # run_watched CONFIG [runner args]: one runner invocation under the $ watchdog
  local cfg=$1; shift
  setsid env PYTHONPATH="$R/src" "$HV/bin/jevbench" run "$cfg" --out runs -o "workers=$WORKERS" "$@" \
    >> "$A/runner.log" 2>&1 < /dev/null &
  local pid=$!
  if (( E19 )); then   # E19: the $ cap and the pause on API overload (E14's rule; NO_PAUSE=1 for the re-run)
    PY scripts/explore_e19_hard_random.py watch --pid "$pid" --cap "$CAP_USD" --prior "$PRIOR_USD" --every 30 \
      ${NO_PAUSE:+--no-pause} >> "$A/watch.log" 2>&1 &
  else
    PY scripts/explore_e7_hard.py --exp "$EXP" watch --pid "$pid" --cap "$CAP_USD" --prior "$PRIOR_USD" --every 30 \
      >> "$A/watch.log" 2>&1 &
  fi
  local wpid=$!
  ( while kill -0 "$pid" 2>/dev/null; do sleep 300; sync_out; done ) & local spid=$!
  wait "$pid"; local rc=$?
  kill "$wpid" "$spid" 2>/dev/null
  sync_out
  return $rc
}
route_ok() {  # one probe call to the System-2 route (Flash via OpenRouter, or the direct API for e7cd); 0 when it answers
  PY - <<'PYEOF'
import os, sys
from openai import OpenAI
try:
    if os.environ.get("EXP") in ("e7cd", "e7f", "e7fx", "e19", "e19x"):
        key = os.environ["GEMINI_API_KEY"].split(",")[0]
        c = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=key, timeout=30, max_retries=0)
        model = "gemini-3.5-flash"
    else:
        c = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=30,
                   max_retries=0)
        model = "google/gemini-3.5-flash"
    r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": "Say OK"}],
                                  reasoning_effort="minimal", max_tokens=8)
    sys.exit(0 if r.choices else 1)
except Exception as e:  # noqa: BLE001
    print("route probe failed:", type(e).__name__)
    sys.exit(1)
PYEOF
}
wait_route() {  # wait (up to 30 min) until the System-2 route answers again
  local i
  for i in $(seq 1 30); do route_ok && return 0; sleep 60; done
  return 1
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
stage_cal() {  # E7c: the calibration shadow arms (development / calibration seeds; no JEV_EVAL_RUNS) under the watchdog
  [[ -n "$RUN_CAL" ]] || { echo "no calibration stage for $EXP"; return 1; }
  cd "$R" || return 1
  keys || return 1
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "server not up"; return 1; }
  printf '{"cal_start": "%s", "instance_type": "%s"}\n' "$(date -u +%FT%TZ)" "$(imds instance-type)" > "$A/cal_start.json"
  run_watched "$CAL_CFG"
  local rc=$?
  st "calibration exit=$rc; $(PY scripts/explore_e7_hard.py --exp "$EXP" spend --prior "$PRIOR_USD" --short 2>/dev/null)"
  leak_check || return 1
  [[ -f "$STOP" ]] && return 1
  return $rc
}
stage_smoke() {
  cd "$R" || return 1
  keys || return 1
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "server not up"; return 1; }
  local rc
  if [[ "$EXP" == e19 ]]; then live_loop "$SMOKE_CFG" "runs/$RUN_SMOKE"; rc=$?; else run_watched "$SMOKE_CFG"; rc=$?; fi
  PY scripts/explore_e7_hard.py --exp "$EXP" project --prior "$PRIOR_USD" --out "$A/projection.json" > "$A/projection.out" 2>&1
  st "smoke exit=$rc; projection: $(PY -c "import json; print(json.load(open('$A/projection.json'))['decision'])" 2>/dev/null)"
  leak_check || return 1
  return $rc
}
stage_xproj() {  # E7f extension: the gate's projection from the first part's logged per-episode costs (no smoke)
  cd "$R" || return 1
  if [[ "$EXP" == e19x ]]; then   # E19 extension: the first part's logged $ per rl_random episode x 20 x 1.05 + the spend so far
    mkdir -p "$A"
    [[ -s runs/e19_hard/rl_random.jsonl ]] || \
      aws s3 sync "$S3OUT/e19/runs/e19_hard" runs/e19_hard --only-show-errors --region "$REGION" --exclude '*.sqlite' || return 1
    [[ -s runs/e19_hard_smoke/rl_random.jsonl ]] || \
      aws s3 sync "$S3OUT/e19/runs/e19_hard_smoke" runs/e19_hard_smoke --only-show-errors --region "$REGION" --exclude '*.sqlite' || return 1
    sha256sum runs/e19_hard/rl_random.jsonl runs/e19_hard_smoke/rl_random.jsonl > "$A/basis.sha256"
    PY scripts/explore_e7_hard.py --exp "$EXP" project --prior "$PRIOR_USD" --out "$A/projection.json" > "$A/projection.out" 2>&1
    st "projection (basis runs/e19_hard, cap \$$CAP_USD over all E19 runs): $(PY -c "import json; print(json.load(open('$A/projection.json'))['decision'])" 2>/dev/null)"
    [[ -s "$A/projection.json" ]]
    return
  fi
  [[ "$EXP" == e7fx ]] || { echo "EXP=e7fx only"; return 1; }
  [[ -s runs/e7f_hard/rl_tier.jsonl && -s runs/e7f_hard/llm_only.jsonl ]] || \
    aws s3 sync "$S3OUT/e7f/runs/e7f_hard" runs/e7f_hard --only-show-errors --region "$REGION" --exclude '*.sqlite' || return 1
  sha256sum runs/e7f_hard/llm_only.jsonl runs/e7f_hard/rl_tier.jsonl > "$A/basis.sha256"
  PY scripts/explore_e7_hard.py --exp "$EXP" project --prior "$PRIOR_USD" --out "$A/projection.json" > "$A/projection.out" 2>&1
  st "projection (basis runs/e7f_hard): $(PY -c "import json; print(json.load(open('$A/projection.json'))['decision'])" 2>/dev/null)"
  [[ -s "$A/projection.json" ]]
}
probe_ok() {  # E19 (h): two consecutive clean probe rounds (six one-word Flash calls each); waits up to PROBE_WAIT_MIN
  local end=$(( $(date +%s) + ${PROBE_WAIT_MIN:-60} * 60 )) clean=0
  while (( $(date +%s) < end )); do
    if PY scripts/explore_e19_hard_random.py probe --log "$A/api_probe_log.txt" > /dev/null 2>&1; then clean=$((clean + 1)); else clean=0; fi
    (( clean >= 2 )) && return 0
    sleep $(( clean > 0 ? 30 : 300 ))
  done
  return 1
}
live_loop() {  # live_loop CFG RUN [runner args]: probe-gated; paused on 503 / overload (records kept) and resumed after
  # two clean probe rounds (the runner skips completed tasks); then one re-run of the errored episodes (E19 (h))
  local cfg=$1 run=$2; shift 2
  local rc=0 round ne
  for round in 1 2 3 4 5 6 7 8; do
    [[ -f "$STOP" ]] && { st "spend cap reached; $cfg not run further"; return 3; }
    probe_ok || { st "probes not clean for ${PROBE_WAIT_MIN:-60} min; $cfg paused (round $round)"; return 4; }
    st "$cfg: start (round $round)"
    run_watched "$cfg" "$@"; rc=$?
    if [[ -f "$run/PAUSED_overload.json" ]]; then
      mv -f "$run/PAUSED_overload.json" "$A/paused_$(basename "$run")_$round.json"
      st "$cfg paused on an overload error (round $round); waiting for clean probes"
      continue
    fi
    break
  done
  [[ -f "$STOP" ]] && return 3
  ne=$(errored "$run")
  if [[ "$ne" != 0 ]]; then
    st "$ne errored episode(s) in $run: one re-run after clean probes"
    if probe_ok; then NO_PAUSE=1 run_watched "$cfg" "$@"; rc=$?; else st "probes not clean; re-run skipped"; fi
  fi
  st "$cfg done; errored now $(errored "$run"); $(PY scripts/explore_e7_hard.py --exp "$EXP" spend --prior "$PRIOR_USD" --short 2>/dev/null)"
  return $rc
}
stage_refit() {  # E19 (c): the registered calibration-arm estimator on E7c's calibration states (E7d's 20 tasks)
  # re-answered by the served random-rules model through the harness's asking path ($0 API)
  cd "$R" || return 1
  [[ "$EXP" == e19 ]] || { echo "EXP=e19 only"; return 1; }
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "server not up"; return 1; }
  mkdir -p "$A/cal" "$A/logs"
  [[ -s runs/e7c_cal/rl_shadow_cal.jsonl ]] || \
    aws s3 sync "$S3OUT/e7c/runs/e7c_cal" runs/e7c_cal --only-show-errors --region "$REGION" --exclude '*.sqlite' || return 1
  local sha; sha=$(sha256sum runs/e7c_cal/rl_shadow_cal.jsonl | cut -c1-64)
  [[ "$sha" == 6fd651974c30cfd2cac814633b6272777bb1a7e9f787e5e2f7ad56cf00551595 ]] \
    || { echo "runs/e7c_cal/rl_shadow_cal.jsonl sha256 $sha differs from E7d's record"; return 1; }
  PY scripts/e5_cal_replay.py build --run runs/e7c_cal --arm rl_shadow_cal --out "$A/cal/cal_requests.jsonl.gz" \
    > "$A/logs/cal_build.log" 2>&1 || { tail -5 "$A/logs/cal_build.log"; return 1; }
  PY - "$A/cal/cal_requests.json" <<'EOF' || return 1
import json, sys
d = json.load(open(sys.argv[1]))      # the rebuilt counts must equal E7d's record
assert d["tasks"] == 100 and d["questions"] == {"tool_select": 3367, "tool_params": 5460}, d
print("request counts match E7d's record")
EOF
  PY scripts/explore_e7d_tier.py head-tasks --requests "$A/cal/cal_requests.jsonl.gz" --tasks 20 \
    --out "$A/cal/cal20_requests.jsonl" > "$A/logs/cal_head.log" 2>&1 || { tail -3 "$A/logs/cal_head.log"; return 1; }
  st "calibration replay starts: $(tail -1 "$A/logs/cal_head.log" | cut -c1-120)"
  printf '{"replay_start": "%s"}\n' "$(date -u +%FT%TZ)" > "$A/cal/replay_window.json"
  PY scripts/e5_cal_replay.py answer --requests "$A/cal/cal20_requests.jsonl" --url "http://127.0.0.1:$PORT" \
    --name "$RND_NAME" --out "$A/cal/answers_random.jsonl" > "$A/logs/cal_answer.log" 2>&1 \
    || { tail -5 "$A/logs/cal_answer.log"; return 1; }
  printf '{"replay_end": "%s"}\n' "$(date -u +%FT%TZ)" >> "$A/cal/replay_window.json"
  PY scripts/explore_e7d_tier.py complete --requests "$A/cal/cal20_requests.jsonl" --answers "$A/cal/answers_random.jsonl" \
    --out "$A/cal" > "$A/logs/cal_complete.log" 2>&1 || { tail -3 "$A/logs/cal_complete.log"; return 1; }
  mv -f "$A/cal/answers_tier_cal.jsonl" "$A/cal/answers_random_cal.jsonl"
  st "calibration states: $(tail -1 "$A/logs/cal_complete.log" | cut -c1-200)"
  PY scripts/explore_e7d_tier.py cal --src runs/e7c_cal --arm rl_shadow_cal --answers "$A/cal/answers_random_cal.jsonl" \
    --name random --out "$A/cal" --work runs/e19_cal_work --tasks "$A/cal/cal_tasks.json" > "$A/logs/cal_fit.log" 2>&1 \
    || { tail -5 "$A/logs/cal_fit.log"; return 1; }
  st "fit: $(tail -1 "$A/logs/cal_fit.log" | cut -c1-200)"
  rm -f "$A/cal/cal_requests.jsonl.gz" "$A/cal/cal20_requests.jsonl"     # large; rebuilt from runs/e7c_cal at any time
  sync_out
}
eval_e19() {  # E19: the gate's n from the smoke's projection, then live_loop (probe-gated, paused on overload, one
  # re-run of the errored episodes); the evaluation window (tariff) is recorded in eval_window.json
  cd "$R" || return 1
  keys || return 1
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "server not up"; return 1; }
  local dec ov=() rc t0 t1
  dec=$(PY -c "import json; print(json.load(open('$A/projection.json'))['decision'])" 2>/dev/null)
  case "$dec" in
    "run all "[0-9]*) ov=() ;;
    "run the first "[0-9]*) ov=(--max-tasks "${dec##* }") ;;
    *) st "refusing eval: projection says '$dec'"; return 3 ;;
  esac
  printf 'step %s\nstarted %s\ncommit %s\nconfig %s %s\nprofile %s\nprereg sha256 %s\ncap_usd %s prior_usd %s\nworkers %s\nhost %s %s\n' \
    "$RUN_EVAL" "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$EVAL_CFG" "$(sha256sum "$EVAL_CFG" | cut -c1-16)" \
    "$(sha256sum configs/harness/jevagent_best.yaml | cut -c1-16)" "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" \
    "$CAP_USD" "$PRIOR_USD" "$WORKERS" "$MY_IID" "$(imds instance-type)" > "$A/START_$RUN_EVAL.txt"
  curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/eval_server_models_start.json"
  t0=$(date -u +%FT%TZ)
  export JEV_EVAL_RUNS=1      # this stage's subshell only (the registered evaluation seeds)
  st "eval ($dec)"
  live_loop "$EVAL_CFG" "runs/$RUN_EVAL" "${ov[@]}"
  rc=$?
  t1=$(date -u +%FT%TZ)
  curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/eval_server_models_end.json"
  printf '{"window_start": "%s", "window_end": "%s", "instance_type": "%s", "instance_id": "%s", "workers": %s, "run": "runs/%s"}\n' \
    "$t0" "$t1" "$(imds instance-type)" "$MY_IID" "$WORKERS" "$RUN_EVAL" > "$A/eval_window.json"
  leak_check || return 1
  [[ -f "$STOP" ]] && return 1
  return $rc
}
stage_eval() {
  cd "$R" || return 1
  (( E19 )) && { eval_e19; return $?; }
  keys || return 1
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "server not up"; return 1; }
  local dec ov=() pass rc t0 t1 ne
  dec=$(PY -c "import json; print(json.load(open('$A/projection.json'))['decision'])" 2>/dev/null)
  case "$dec" in
    "run all "[0-9]*) ov=() ;;
    "run the first "[0-9]*) ov=(--max-tasks "${dec##* }") ;;
    *) st "refusing eval: projection says '$dec'"; return 3 ;;
  esac
  printf 'step %s\nstarted %s\ncommit %s\nconfig %s %s\nprofile %s\nprereg sha256 %s\ncap_usd %s prior_usd %s\nworkers %s\nhost %s %s\n' \
    "$RUN_EVAL" "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$EVAL_CFG" "$(sha256sum "$EVAL_CFG" | cut -c1-16)" \
    "$(sha256sum configs/harness/jevagent_best.yaml | cut -c1-16)" "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" \
    "$CAP_USD" "$PRIOR_USD" "$WORKERS" "$MY_IID" "$(imds instance-type)" > "$A/START_$RUN_EVAL.txt"
  curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/eval_server_models_start.json"
  t0=$(date -u +%FT%TZ)
  export JEV_EVAL_RUNS=1      # this stage's subshell only (the registered evaluation seeds)
  for pass in 1 2 3; do
    [[ -f "$STOP" ]] && { st "watchdog stop recorded; no further pass"; break; }
    if (( pass > 1 )); then
      ne=$(errored "runs/$RUN_EVAL")
      [[ "$ne" == 0 ]] && { st "no errored episode after pass $((pass - 1)); no further pass"; break; }
      st "$ne errored episode(s) after pass $((pass - 1)); waiting for the System-2 route"
      wait_route || st "the System-2 route did not answer for 30 min; pass $pass runs anyway"
    fi
    st "eval pass $pass ($dec)"
    run_watched "$EVAL_CFG" "${ov[@]}"
    rc=$?
    st "eval pass $pass exit=$rc; $(PY scripts/explore_e7_hard.py --exp "$EXP" spend --prior "$PRIOR_USD" --short 2>/dev/null)"
  done
  t1=$(date -u +%FT%TZ)
  curl -sf "http://127.0.0.1:$PORT/v1/models" > "$A/eval_server_models_end.json"
  printf '{"window_start": "%s", "window_end": "%s", "instance_type": "%s", "instance_id": "%s", "workers": %s, "run": "runs/%s"}\n' \
    "$t0" "$t1" "$(imds instance-type)" "$MY_IID" "$WORKERS" "$RUN_EVAL" > "$A/eval_window.json"
  leak_check || return 1
  [[ -f "$STOP" ]] && return 1
  return 0
}
# ---------------------------------------------------------------------------------- E7d (GPU only, $0 API, no key)
D_CK=$W/ck_e7d/tier                     # AgentDecision-4B (tier): the continued adapter (a NEW checkpoint)
D_PORT=8082
D_NAME=agentdecision-4b-tier
D_V1=(data/train_v1/dev.jsonl.gz data/train_v1/dev_recovery.jsonl.gz data/train_v1/dev_knockout.jsonl.gz)
D_HELD=("$A/data/heldout_op.jsonl" "$A/data/heldout_ts_other.jsonl" "$A/data/heldout_tp.jsonl")
d_frozen() { plan "d['rows']['frozen']['dir']"; }
stage_d_inputs() {  # D100 (E3's builder), corpus C, the E7c calibration run (S3) and its replay requests
  cd "$R" || return 1
  [[ "$EXP" == e7d ]] || { echo "EXP=e7d only"; return 1; }
  mkdir -p "$A/data" "$A/s23" "$A/cal" "$A/eval" "$A/logs"
  PY scripts/train/ablation_subsets.py build --out "$A/data" --arms D100 > "$A/logs/d100.log" 2>&1 \
    || { tail -5 "$A/logs/d100.log"; return 1; }
  [[ -s "$A/data/D100.jsonl" ]] || { echo "no D100.jsonl"; return 1; }
  local f
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz; do cp -f results/stage2/s23/$f "$A/s23/$f" || return 1; done
  aws s3 sync "$S3OUT/e7c/runs/e7c_cal" runs/e7c_cal --only-show-errors --region "$REGION" || return 1
  PY scripts/e5_cal_replay.py build --run runs/e7c_cal --arm rl_shadow_cal --out "$A/cal/cal_requests.jsonl.gz" \
    > "$A/logs/cal_build.log" 2>&1 || { tail -5 "$A/logs/cal_build.log"; return 1; }
  sha256sum "$A/data/D100.jsonl" results/stage2/s23/* runs/e7c_cal/*.jsonl "$A/cal/cal_requests.jsonl.gz" > "$A/inputs.sha256"
}
stage_d_harvest() {  # $0: the simulated oracle executor + NearMiss System-1 (in-process); labels = oracle gold
  cd "$R" || return 1
  PY scripts/explore_e7d_tier.py harvest --kind train --out runs/e7d_harvest --workers 24 > "$A/logs/harvest_train.log" 2>&1 \
    || { tail -5 "$A/logs/harvest_train.log"; return 1; }
  PY scripts/explore_e7d_tier.py harvest --kind heldout --out runs/e7d_harvest --workers 24 > "$A/logs/harvest_held.log" 2>&1 \
    || { tail -5 "$A/logs/harvest_held.log"; return 1; }
  PY scripts/explore_e7d_tier.py build --harvest runs/e7d_harvest --d100 "$A/data/D100.jsonl" --out "$A/data" \
    > "$A/logs/build.log" 2>&1 || { tail -5 "$A/logs/build.log"; return 1; }
  cat "$A/logs/build.log"
}
stage_d_train() {  # continue the frozen adapter for 1 epoch (gc_3072; gc_2048 on out-of-memory), capped at 100 min
  cd "$R" || return 1
  verify_4b || return 1
  local setting flags rc t0
  stop_server
  for setting in gc_3072 gc_2048; do
    flags=$(PY scripts/train/ablation_box.py setting "$setting") || return 1
    rm -rf "$D_CK"; mkdir -p "$D_CK"
    nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 2000 > "$A/logs/train_smi.csv" 2>/dev/null &
    local smi=$!
    t0=$(date +%s)
    # shellcheck disable=SC2086
    timeout 100m env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.train_decision \
      --model "$(cat "$W/base_model_path.txt")" --init "$CK/$(d_frozen)" --train "$A/data/train_mix.jsonl" --out "$D_CK" \
      --loss soft_ce_brier --brier-weight 0.5 --lr 5e-5 --epochs 1 --batch-size 2 --max-len 4096 --seed 0 $flags \
      > "$A/logs/train_$setting.log" 2>&1
    rc=$?
    kill "$smi" 2>/dev/null
    printf '{"setting": "%s", "exit": %s, "minutes": %s}\n' "$setting" "$rc" "$(( ($(date +%s) - t0) / 60 ))" >> "$A/train_runs.jsonl"
    (( rc == 0 )) && break
    grep -q "out of memory" "$A/logs/train_$setting.log" || return 1
  done
  (( rc == 0 )) || return 1
  cp -f "$CK/$(d_frozen)/tokenizer"* "$CK/$(d_frozen)/chat_template.jinja" "$D_CK/" 2>/dev/null
  aws s3 sync "$D_CK" "$B/checkpoints/e7d_tier/" --only-show-errors --region "$REGION" --exclude 'checkpoint-*' \
    --exclude '*optimizer*' --exclude '*.pt' --exclude 'rng_state*'
  sha256sum "$D_CK"/adapter_model.safetensors "$D_CK"/adapter_config.json > "$A/tier_sha256.txt"
}
stage_d_wait() {  # a second chain started while the first chain's d_train subshell still runs (Deviations "E7d:
  # scoring within the GPU cap"): wait for that subshell's last step (tier_sha256.txt), at most until D_TRAIN_END
  local end; end=$(date -u -d "${D_TRAIN_END:-2026-10-07 00:10}" +%s)
  while (( $(date -u +%s) < end )); do
    [[ -s "$A/tier_sha256.txt" ]] && { st "training done: $(tail -1 "$A/train_runs.jsonl" 2>/dev/null)"; return 0; }
    if ! pgrep -f 'jevharness.training.train_decision|checkpoints/e7d_tier' > /dev/null; then
      sleep 60; [[ -s "$A/tier_sha256.txt" ]] && continue
      echo "no training process and no tier_sha256.txt"; tail -3 "$A/train_runs.jsonl" 2>/dev/null; return 1
    fi
    sleep 30
  done
  echo "training not done by ${D_TRAIN_END:-2026-10-07 00:10}"; return 1
}
stage_d_eval() {  # temperatures (1,000 v1 dev + 1,000 tier dev rows), held-out tier states (write-state 600, other
  # tool choice 225, arguments 225) for both checkpoints; no v1 dev pass (Deviations "E7d: scoring within the GPU cap")
  cd "$R" || return 1
  local base; base=$(cat "$W/base_model_path.txt")
  local held=("$A/data/heldout_op.jsonl" "$A/data/heldout_ts_other_s.jsonl" "$A/data/heldout_tp_s.jsonl")
  stop_server
  PY scripts/explore_e7d_tier.py trim --data "$A/data" > "$A/logs/trim.log" 2>&1 || { tail -5 "$A/logs/trim.log"; return 1; }
  PY -m jevharness.training.calibrate --model "$base" --adapter "$D_CK" --dev "$A/data/cal_dev_small.jsonl" \
    --out "$A/cal/tier_calibration.json" --min-bucket 200 --batch-size 8 > "$A/logs/calibrate.log" 2>&1 || return 1
  cp -f "$A/cal/tier_calibration.json" "$D_CK/calibration.json"
  st "temperatures refitted"
  PY -m jevharness.training.eval_decisions --model "$base" --adapter "$D_CK" --calibration "$A/cal/tier_calibration.json" \
    --dev "${held[@]}" --n-perm 0 --max-len 4096 --batch-size 8 --out "$A/eval/tier_heldout.json" > "$A/logs/eval_tier_held.log" 2>&1 || return 1
  st "held-out: tier done"
  PY -m jevharness.training.eval_decisions --model "$base" --adapter "$CK/$(d_frozen)" \
    --calibration "$CK/$(plan "d['rows']['frozen'].get('calibration') or ''")" \
    --dev "${held[@]}" --n-perm 0 --max-len 4096 --batch-size 8 --out "$A/eval/frozen_heldout.json" > "$A/logs/eval_frozen_held.log" 2>&1 || return 1
}
stage_d_replay() {  # ONE server at a time (Deviations "E7d: one server at a time": two 4B servers ran out of GPU
  # memory on the long JevGym-Hard states). The registered checkpoint alone: serving-path check (first 5 tasks). Then
  # the new checkpoint alone: corpus C, then the E7c calibration states until D_REPLAY_END; `complete` here, the
  # fits and the score run off the box (CPU, $0)
  cd "$R" || return 1
  local base k left; base=$(cat "$W/base_model_path.txt")
  stop_server
  [[ -f "$A/cal/answers_frozen_check.jsonl" ]] && mv -f "$A/cal/answers_frozen_check.jsonl" "$A/cal/answers_frozen_check_oom.jsonl"
  stage_serve || return 1                                       # the registered checkpoint on :8081 (S2-4's server)
  [[ -s "$A/cal/check_requests.jsonl" ]] || PY scripts/explore_e7d_tier.py head-tasks \
    --requests "$A/cal/cal_requests.jsonl.gz" --out "$A/cal/check_requests.jsonl" > "$A/logs/check_build.log" 2>&1 || return 1
  PY scripts/e5_cal_replay.py answer --requests "$A/cal/check_requests.jsonl" --url "http://127.0.0.1:$PORT" \
    --name jevagentrl-4b --out "$A/cal/answers_frozen_check.jsonl" > "$A/logs/cal_frozen_check.log" 2>&1
  PY scripts/e5_cal_replay.py check --requests "$A/cal/check_requests.jsonl" --answers "$A/cal/answers_frozen_check.jsonl" \
    --out "$A/cal/check_frozen.json" > "$A/logs/cal_check.log" 2>&1
  st "serving-path check: $(tail -1 "$A/logs/cal_check.log")"
  stop_server
  nohup env PYTHONPATH="$R/src" "$HV/bin/python" -m jevharness.training.serve_jev --model "$base" --name "$D_NAME" \
    --port "$D_PORT" --adapter "$D_CK" --calibration "$A/cal/tier_calibration.json" > "$A/logs/serve_tier.out" 2>&1 &
  echo $! >> "$W/e7_serve.pid"
  for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$D_PORT/health" > /dev/null && break; sleep 5; done
  curl -sf "http://127.0.0.1:$D_PORT/v1/models" > "$A/serve_tier_models.json" || { echo "tier server not up"; return 1; }
  local spec; spec=$(printf '{"type": "jev", "name": "%s", "base_url": "http://127.0.0.1:%s", "model": "%s", "api_key": "", "timeout": 60.0}' \
    "$D_NAME" "$D_PORT" "$D_NAME")
  for k in 1 2 3; do
    PY scripts/stage2_corpus.py replay --corpus "$A/s23" --name e7d_tier --s1 "$spec" --cache cache/s1_e7d_tier.sqlite \
      > "$A/logs/replay_c_$k.log" 2>&1
  done
  st "corpus C replayed: $(wc -l < "$A/s23/answers_e7d_tier.jsonl" 2>/dev/null) answers"
  left=$(( $(date -u -d "${D_REPLAY_END:-2026-10-07 01:14}" +%s) - $(date -u +%s) ))
  if (( left > 120 )); then
    timeout "$left" env PYTHONPATH="$R/src" "$HV/bin/python" scripts/e5_cal_replay.py answer \
      --requests "$A/cal/cal_requests.jsonl.gz" --url "http://127.0.0.1:$D_PORT" --name "$D_NAME" \
      --out "$A/cal/answers_tier.jsonl" > "$A/logs/cal_tier.log" 2>&1
  fi
  stop_server
  PY scripts/explore_e7d_tier.py complete --requests "$A/cal/cal_requests.jsonl.gz" --answers "$A/cal/answers_tier.jsonl" \
    --out "$A/cal" > "$A/logs/cal_complete.log" 2>&1
  st "calibration states: $(tail -1 "$A/logs/cal_complete.log" | cut -c1-200)"
  true
}
stage_final() {
  stop_server
  nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; sudo shutdown -h +2 "jev-e7-hard: done"; fi
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
