#!/usr/bin/env bash
# Exploratory row E3: the SFT training ablation of the decision model (results/training/ablation_plan.md, addendum of
# 2026-10-05; Deviations draft results/training/ablation_deviation_draft.md, appended to docs/PREREG_A7_STAGE2.md before
# the first training step). GPU only: no LLM call and no key, ever (the preamble unsets every *_API_KEY(S); every
# replay record must carry cost_usd == 0, else the chain stops). Host: the persistent Stage-2 GPU host
# <instance-id> (one L40S 48 GB as g6e.2xlarge, or one A10G 24 GB as g5.4xlarge); `rearm` reads and records the
# instance type, and the cap projection, the anchor's temperature tolerance and the smoke adapt to it.
#
#   LOCAL:  bash scripts/aws/train_ablation.sh push | send "STAGES" | status [N] | stop | itype
#   BOX stages, in the order sent (a failed stage stops the chain; abl_final always runs when sent):
#     rearm           idle stop (IDLE_MIN) + power-off guard (GUARD_MIN), instance type, GPU, library versions, host.json
#     abl_inputs      the teacher-only input from S3; every tier-1 arm file built and sha256-checked against
#                     results/training/ablation_inputs.json; a copy of the common-state corpus C (results/stage2/s23)
#     abl_anchor      the existing SFT v1 checkpoint (sha256; fetched from S3 when absent) through eval / temperatures /
#                     serve / C replay: the go / no-go of the plan (anchor_check.json); fixes the serving mode
#     abl_smoke       memory smoke (4 optimizer steps on the longest rows); first fitting setting -> mem_setting.txt
#                     (abl_smoke and every arm stage refuse to train unless the checkout's docs/PREREG_A7_STAGE2.md
#                     holds the Deviations entry "Exploratory row E3")
#     abl_core        the core arms (scripts/train/ablation_subsets.py CORE), in order
#     abl_ext         the extended arms (EXTENDED), in order
#     abl_arms:LIST   any arms, e.g. abl_arms:D25,S_canon+S_match ("+" binds arms into one cap unit)
#     abl_final       cost check of every replay record, nvidia-smi, scoring, final sync, power-off (POWEROFF=1)
#   Per arm: SFT with the SFT v1 recipe (scripts/train/10_sft.sh; on a CUDA out-of-memory error the next memory setting
#   of ablation_box.SETTINGS, never another LR / epochs / batch / loss / rank / data), eval_decisions on the three v1 dev
#   files, temperatures on v1 dev, serve_jev on :8082 + the C replay (three invocations), adapter -> S3, local
#   checkpoint-* deleted, the run scored. Before each cap unit: used + projected GPU hours <= CAP_H, else the unit is
#   skipped and the chain goes on. Resumable: finished arms (done/<arm>) are skipped, a trained arm resumes at eval.
#   Send-time env: CAP_H [24], GUARD_MIN [(CAP_H+4)*60], IDLE_MIN [60], SERVE_FAST [1], POWEROFF [1], WAIT_PREV [1].
#   DRY_RUN=1 (box side, run locally: W=<dir> DRY_RUN=1 bash scripts/aws/train_ablation.sh box "STAGES"): every heavy
#   command (GPU, S3, sudo, the arm build) is written to $A/dry_run.log instead of run; light checks still run.
# Records: runs/train_ablation on the box -> s3://$BUCKET/train_ablation/ (no arm data files, no corpus copies);
# adapters -> s3://$BUCKET/checkpoints/train_ablation/<arm>/ (no optimizer state).
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
IID=${IID:?set IID to the instance id}
BUNDLE=train_ablation.bundle
S3OUT="$B/train_ablation"
S3CK="$B/checkpoints/train_ablation"
S3_SFT1="$B/checkpoints/round5/sft_v1"
CAP_H=${CAP_H:-24}
GUARD_MIN=${GUARD_MIN:-$(( (${CAP_H%.*} + 4) * 60 ))}
IDLE_MIN=${IDLE_MIN:-60}
SERVE_FAST=${SERVE_FAST:-1}
POWEROFF=${POWEROFF:-1}
WAIT_PREV=${WAIT_PREV:-1}

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws --region "$REGION" "$@"; }

if [[ "$MODE" == push ]]; then
  ROOT=$SELF_ROOT
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null || { echo "bundle failed"; exit 1; }
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors || exit 1
  git -C "$ROOT" show "$BRANCH:scripts/aws/train_ablation.sh" > "$tmp/train_ablation.sh"   # the committed driver
  awsl s3 cp "$tmp/train_ablation.sh" "$B/code/train_ablation.sh" --only-show-errors || exit 1
  awsl s3 ls "$S3OUT/inputs/train_teacher_only.jsonl.gz" > /dev/null 2>&1 \
    || echo "WARNING: $S3OUT/inputs/train_teacher_only.jsonl.gz is missing (plan section 4.1): abl_inputs will fail"
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH" | cut -c1-72); driver $(sha256sum "$tmp/train_ablation.sh" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/ta_$(date +%s%N).sh'      # a private copy per invocation (bash reads a running script lazily)
    echo "aws s3 cp $B/code/train_ablation.sh \$f --only-show-errors --region $REGION"
    printf 'export CAP_H=%q GUARD_MIN=%q IDLE_MIN=%q SERVE_FAST=%q POWEROFF=%q WAIT_PREV=%q\n' "$CAP_H" "$GUARD_MIN" \
      "$IDLE_MIN" "$SERVE_FAST" "$POWEROFF" "$WAIT_PREV"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/ta_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/ta_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/ta_driver.sh /tmp/ta_driver.sh --only-show-errors && bash /tmp/ta_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == stop ]]; then
  awsl ec2 stop-instances --instance-ids "$IID" --output text --query 'StoppingInstances[].[InstanceId,CurrentState.Name]'
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  awsl ec2 describe-instances --instance-ids "$IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,32p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
DRY_RUN=${DRY_RUN:-0}
W=${W:-/opt/jev}
mkdir -p "$W" && cd "$W" || exit 1
my_ts=$(basename "$0" .sh); my_ts=${my_ts#ta_}
others_running() {  # another driver's chain, or an older E3 chain, is still running (never touch its tree or GPU)
  local t
  ps -eo args | grep -E '/opt/jev/(hf|ex|g2|s2|sl|r5|r6)_[0-9]+\.sh box' | grep -v -e grep -e "$0" | grep -q . && return 0
  for t in $(ps -eo args | grep -oE '/opt/jev/ta_[0-9]+\.sh box' | grep -oE '[0-9]{6,}' | sort -u); do
    [[ "$t" < "$my_ts" ]] && return 0
  done
  return 1
}
waited=0
if [[ "$WAIT_PREV" == 1 && "$DRY_RUN" != 1 ]]; then
  while others_running; do sleep 30; waited=$((waited + 30)); done
fi
if [[ "$DRY_RUN" == 1 ]]; then
  CHECKOUT=${CHECKOUT:-$SELF_ROOT}
  [[ -n "$CHECKOUT" && -f "$CHECKOUT/scripts/aws/train_ablation.sh" ]] && cd "$CHECKOUT" || { echo "no checkout"; exit 1; }
else
  myb="$W/ta.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || exit 1
  [[ -d "$W/stage2" ]] || git clone -q -b "$BRANCH" "$myb" "$W/stage2"
  cd "$W/stage2" || exit 1
  # every box-written status.txt under results/ survives the reset (tracked: the reset would restore the committed copy)
  keep="$W/ta_keep.$$"
  mkdir -p "$keep"
  git diff --name-only HEAD 2>/dev/null | grep -E '^results/.*status\.txt$' | while read -r f; do
    mkdir -p "$keep/$(dirname "$f")" && cp -p "$f" "$keep/$f"; done
  git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD
  (cd "$keep" && find . -type f) | while read -r f; do mkdir -p "$(dirname "$f")"; cp -p "$keep/$f" "$f"; done
  rm -rf "$keep" "$myb"
fi
A=${ABL_RUN:-runs/train_ablation}
mkdir -p "$A"/{data,s23,eval,cal,logs,settings,done,smoke} cache
HV=$W/hvenv
if [[ "$DRY_RUN" == 1 ]]; then PY=${PY:-python3}; else PY=$HV/bin/python; set +u; . "$HV/bin/activate"; set -u; fi
ACK=$W/ck_abl                                  # E3 adapters (one directory per arm)
SFT1_DIR=$W/ck/sft_v1                          # SFT v1 as fetched by scripts/aws/stage2_gpu.sh stage_fetch
PLAN=results/stage2/rl_eval_plan.json          # its sha256 list (= results/gpu_round5/FREEZE_RECORD.json)
TEACH=$W/inputs/train_teacher_only.jsonl.gz
PORT=8082
URL=http://127.0.0.1:$PORT
SERVE_PID=$W/abl_serve.pid
GPU_WAIT_MIN=${GPU_WAIT_MIN:-20}
V1_FILES=(data/train_v1/dev.jsonl.gz data/train_v1/dev_recovery.jsonl.gz data/train_v1/dev_knockout.jsonl.gz)
ANCHOR=sft_v1
BASE=$(cat "$W/base_model_path.txt" 2>/dev/null || echo "${DRY_BASE:-/BASE_MODEL}")
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dry() { printf 'DRY:'; printf ' %q' "$@"; printf '\n'; }
X() {  # a heavy command (GPU, S3, sudo, long CPU): run it, or under DRY_RUN=1 only record it
  if [[ "$DRY_RUN" == 1 ]]; then dry "$@" | tee -a "$A/dry_run.log"; return 0; fi
  "$@"
}
dry_skip() {  # dry_skip NAME: true (skip) under DRY_RUN, where the inputs of a check were never produced
  [[ "$DRY_RUN" == 1 ]] || return 1
  echo "DRY: skipped $1" | tee -a "$A/dry_run.log"
}
S3() { X aws "$@" --only-show-errors --region "$REGION"; }
box() {  # box SUBCOMMAND ...: scripts/train/ablation_box.py (CPU-only checks); box units SPEC: the cap units of SPEC
  if [[ "${1:-}" == units ]]; then shift; "$PY" scripts/train/ablation_subsets.py units "$@"; return; fi
  "$PY" scripts/train/ablation_box.py "$@"
}
sync_out() {
  S3 s3 sync "$A" "$S3OUT" --exclude 'data/*.jsonl' --exclude 'smoke/*.jsonl' --exclude 's23/occurrences.jsonl.gz' \
    --exclude 's23/questions.jsonl.gz' --exclude '*.sqlite' --exclude 'dry_run.log'
}
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$A/status.txt"
  [[ "$DRY_RUN" == 1 ]] || sync_out > /dev/null 2>&1
}
sync_ck() {  # sync_ck ARM: adapter, adapter config, calibration, decision config, tokenizer (no optimizer state)
  S3 s3 sync "$ACK/$1" "$S3CK/$1" --exclude 'checkpoint-*' --exclude '*optimizer*' --exclude '*.pt' \
    --exclude 'rng_state*' --exclude 'train_mix.jsonl'
}
itype() {
  if [[ "$DRY_RUN" == 1 ]]; then echo "${DRY_ITYPE:-g5.4xlarge}"; return; fi
  curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/instance-type
}
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
arm_stops() {  # power-off guard + idle stop (cron, IDLE_MIN), as the other drivers
  X sudo shutdown -c 2>/dev/null
  X sudo shutdown -h +"$GUARD_MIN" "jev-train-ablation: total guard" 2>/dev/null
  [[ "$DRY_RUN" == 1 ]] && { dry install /usr/local/bin/jev-idle-stop "IDLE_MIN=$IDLE_MIN" >> "$A/dry_run.log"; return 0; }
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without a chain / trainer / server / runner process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'ta_[0-9]|hf_[0-9]|g2_[0-9]|ex_[0-9]|s2_[0-9]|sl_[0-9]|jevharness|jevbench|train_decision|serve_jev|stage2_corpus|ablation_|startlux_decision|llama-server|uv pip|hf download' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-train-ablation: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
}
registered() {  # the Deviations entry of E3 is in the checkout's registration before any training step (smoke included)
  [[ "$DRY_RUN" == 1 ]] && return 0
  grep -q 'Exploratory row E3' docs/PREREG_A7_STAGE2.md && return 0
  st "no Deviations entry 'Exploratory row E3' in docs/PREREG_A7_STAGE2.md (results/training/ablation_deviation_draft.md): no training"
  return 1
}
serve_mode() { cat "$A/serve_mode.txt" 2>/dev/null || { (( SERVE_FAST )) && echo fast || echo plain; }; }
score_box() {  # the run scored as it stands (descriptive; never used to choose anything)
  "$PY" scripts/train/ablation_score.py --run "$A" --out "$A/score" > "$A/logs/score.log" 2>&1 || true
}

# ---------------------------------------------------------------------------------------------- serving + replay
stop_abl_server() {
  local p
  if [[ -f "$SERVE_PID" ]]; then
    p=$(cat "$SERVE_PID")
    kill "$p" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -9 "$p" 2>/dev/null
  fi
  rm -f "$SERVE_PID"
}
serve_abl() {  # serve_abl ADAPTER CALIBRATION NAME MODE: serve_jev on :8082 (--fast when MODE = fast)
  local args=(--model "$BASE" --adapter "$1" --calibration "$2" --name "$3" --port "$PORT")
  [[ "$4" == fast ]] && args+=(--fast)
  stop_abl_server
  echo "serve_jev ${args[*]}" >> "$A/logs/serve.cmd"
  if [[ "$DRY_RUN" == 1 ]]; then dry "$PY" -m jevharness.training.serve_jev "${args[@]}" | tee -a "$A/dry_run.log"; return 0; fi
  if curl -sf "$URL/health" > /dev/null 2>&1; then echo "port $PORT is served by another process"; return 1; fi
  nohup "$PY" -m jevharness.training.serve_jev "${args[@]}" > "$A/logs/serve_$3.out" 2>&1 &
  echo $! > "$SERVE_PID"
  for _ in $(seq 1 180); do
    if curl -sf "$URL/health" > "$A/logs/serve_$3_health.json"; then
      curl -sf "$URL/v1/models" > "$A/logs/serve_$3_models.json"
      return 0
    fi
    kill -0 "$(cat "$SERVE_PID")" 2>/dev/null || { echo "serve_jev died"; tail -20 "$A/logs/serve_$3.out"; return 1; }
    sleep 5
  done
  echo "serve_jev did not come up"
  return 1
}
replay_ck() {  # replay_ck ARM ADAPTER MODE: every distinct question of C (three invocations; replay resumes); rc 9 = priced
  local arm=$1 mode=$3 name="abl-$1" k rc
  serve_abl "$2" "$A/cal/${arm}_calibration.json" "$name" "$mode" || return 1
  local spec="{\"type\": \"jev\", \"name\": \"$name\", \"base_url\": \"$URL\", \"model\": \"$name\", \"api_key\": \"\", \"timeout\": 60.0}"
  for k in 1 2 3; do
    X "$PY" scripts/stage2_corpus.py replay --corpus "$A/s23" --name "abl_$arm" --s1 "$spec" \
      --cache "cache/s1_abl_${arm}_$mode.sqlite" > "$A/logs/replay_${arm}_${mode}_$k.log" 2>&1
    st "replay $arm ($mode) invocation $k exit=$?"
  done
  stop_abl_server
  dry_skip "cost check and completeness ($arm)" && return 0
  box cost --run "$A" > "$A/logs/cost_$arm.log" 2>&1 || { st "a replay record is priced or not local: STOP (cost_check.json)"; return 9; }
  box answers --run "$A" --arm "$arm" > "$A/logs/answers_$arm.txt" 2>&1; rc=$?
  st "replay $arm: $(cat "$A/logs/answers_$arm.txt")"
  return $(( rc == 0 ? 0 : 1 ))                    # an incomplete replay never marks an arm done (a rerun resumes it)
}

# ---------------------------------------------------------------------------------------------- training + evaluation
oom() { [[ -f "$1" ]] && box oom "$1"; }
sft_run() {  # sft_run NAME TRAIN SEED REPLAY_FACTOR SETTING LOG [MAX_STEPS]: 10_sft.sh with the SFT v1 recipe
  local out="$ACK/$1" train=$2 seed=$3 rf=$4 setting=$5 log=$6 steps=${7:--1} flags rep=""
  flags=$(box setting "$setting") || return 1
  [[ "$rf" != 0 && "$rf" != 0.0 ]] && rep=data/train_v1/replay_when2call.jsonl.gz
  [[ "$DRY_RUN" == 1 ]] || rm -rf "$out"
  local envs=(GPUS=1 RECIPE=r3 MODEL="$BASE" TRAIN="$train" DEV=data/train_v0/dev.jsonl REPLAY="$rep"
              REPLAY_FACTOR="$rf" OUT="$out" LR=1e-4 EPOCHS=2 BATCH=2 EFF_BATCH=64 MAX_LEN=4096 SEED="$seed"
              MAX_STEPS="$steps" PY="$PY" SFT_EXTRA="$flags")
  if [[ "$DRY_RUN" == 1 ]]; then   # 10_sft.sh's own dry run prints its commands (data mix, trainer)
    env "${envs[@]}" DRY_RUN=1 bash scripts/train/10_sft.sh 2>&1 | sed 's/^/DRY(10_sft.sh): /' | tee -a "$A/dry_run.log" > "$log"
    return 0
  fi
  env "${envs[@]}" DRY_RUN=0 bash scripts/train/10_sft.sh > "$log" 2>&1
}
eval_ck() {  # eval_ck ARM ADAPTER: eval_decisions (raw) on the v1 dev files + temperatures on v1 dev (batch 16, 8, 4 on OOM)
  local arm=$1 ad=$2 bs ebs ok=0
  if [[ "$DRY_RUN" != 1 && -s "$A/eval/${arm}_v1.json" && -s "$A/cal/${arm}_calibration.json" ]]; then
    echo "$arm: evaluation and temperatures from an earlier attempt kept"; return 0
  fi
  for bs in 16 8 4; do
    X "$PY" -m jevharness.training.eval_decisions --model "$BASE" --adapter "$ad" --dev "${V1_FILES[@]}" --n-perm 0 \
      --max-len 4096 --batch-size "$bs" --out "$A/eval/${arm}_v1.json" > "$A/logs/eval_$arm.log" 2>&1 && { ok=1; break; }
    oom "$A/logs/eval_$arm.log" || break
    st "eval $arm: out of memory at batch $bs"
  done
  (( ok )) || { echo "eval_decisions failed ($arm)"; return 1; }
  ebs=$bs ok=0
  for bs in 16 8 4; do
    X "$PY" -m jevharness.training.calibrate --model "$BASE" --adapter "$ad" --dev data/train_v1/dev.jsonl.gz \
      --out "$A/cal/${arm}_calibration.json" --min-bucket 200 --batch-size "$bs" > "$A/logs/cal_$arm.log" 2>&1 && { ok=1; break; }
    oom "$A/logs/cal_$arm.log" || break
    st "calibrate $arm: out of memory at batch $bs"
  done
  (( ok )) || { echo "calibrate failed ($arm)"; return 1; }
  echo "eval_decisions batch $ebs; calibrate batch $bs" > "$A/settings/${arm}_eval_batch.txt"
}

# ---------------------------------------------------------------------------------------------- stages
start_rec() {
  printf 'step train_ablation (E3)\nstarted %s\ncommit %s\nprereg sha256 %s\nplan sha256 %s\ninputs sha256 %s\nhost %s\ncap_h %s guard_min %s idle_min %s serve_fast %s\n' \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD 2>/dev/null)" "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" \
    "$(sha256sum results/training/ablation_plan.md | cut -c1-16)" "$(sha256sum results/training/ablation_inputs.json | cut -c1-16)" \
    "$(itype)" "$CAP_H" "$GUARD_MIN" "$IDLE_MIN" "$SERVE_FAST" > "$A/START_train_ablation.txt"
}
stage_rearm() {
  local it gpu
  arm_stops
  it=$(itype); gpu=$(gpu_name)
  echo "$it" > "$A/instance_type.txt"
  X nvidia-smi > "$A/nvidia_smi_start.txt" 2>&1
  X "$PY" -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'cuda', torch.cuda.is_available())" > "$A/env.txt" 2>&1
  [[ -f "$W/base_model_path.txt" ]] && cp "$W/base_model_path.txt" "$A/base_model_path.txt"
  box host --out "$A/host.json" --itype "$it" --gpu "$gpu" --cap-h "$CAP_H" --serve-mode "$( (( SERVE_FAST )) && echo fast || echo plain)" || return 1
  (compgen -e | grep -E '_API_KEYS?$' || true) > "$A/keys_in_env.txt"      # names only; must be empty
  [[ -s "$A/keys_in_env.txt" ]] && { echo "an API key is in the environment"; return 1; }
  [[ -f "$A/START_train_ablation.txt" ]] || start_rec
  (( waited > 0 )) && st "waited $(( waited / 60 )) min for another chain before touching the checkout"
  cat "$A/host.json"
}
stage_abl_inputs() {
  mkdir -p "$W/inputs"
  S3 s3 cp "$S3OUT/inputs/train_teacher_only.jsonl.gz" "$TEACH" || return 1
  S3 s3 cp "$S3OUT/inputs/train_teacher_only.jsonl.summary.json" "$W/inputs/" || return 1
  X "$PY" scripts/train/ablation_subsets.py build --out "$A/data" --teacher-only "$TEACH" > "$A/logs/build.log" 2>&1 || return 1
  dry_skip "inputs check" || box inputs --run "$A" --teacher "$TEACH" || return 1
  local f
  for f in corpus.json occurrences.jsonl.gz questions.jsonl.gz; do
    cp results/stage2/s23/$f "$A/s23/$f" && cmp -s results/stage2/s23/$f "$A/s23/$f" || return 1
  done
  sha256sum results/stage2/s23/{corpus.json,occurrences.jsonl.gz,questions.jsonl.gz} data/train_v1/*.jsonl.gz \
    data/train_v0/dev.jsonl results/training/ablation_inputs.json > "$A/inputs.sha256"
}
verify_sft1() {  # verify_sft1 DIR: the two sha256 of SFT v1 in the plan (results/stage2/rl_eval_plan.json rows.sft_v1)
  dry_skip "sft_v1 sha256 ($1)" && return 0
  "$PY" - "$PLAN" "$1" "$A/sft_v1_verify.json" <<'EOF'
import hashlib, json, os, sys
plan, d, out = sys.argv[1:4]
want = json.load(open(plan))["rows"]["sft_v1"]["sha256"]
res = {}
for rel, sha in want.items():
    p = os.path.join(d, rel)
    got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    res[rel] = {"expected": sha, "got": got, "ok": got == sha}
ok = all(r["ok"] for r in res.values())
json.dump({"dir": d, "ok": ok, "files": res}, open(out, "w"), indent=1)
print("sft_v1 sha256:", "all match" if ok else "MISMATCH", d)
sys.exit(0 if ok else 1)
EOF
}
stage_abl_anchor() {
  local sft1=$SFT1_DIR mode rc t0
  if [[ "$DRY_RUN" != 1 && "${REDO_ANCHOR:-0}" != 1 ]] \
     && "$PY" -c "import json, sys; sys.exit(0 if json.load(open('$A/anchor_check.json')).get('ok') else 1)" 2>/dev/null; then
    echo "anchor: GO recorded earlier (anchor_check.json); not repeated"; return 0
  fi
  rm -f "$A/eval/${ANCHOR}_v1.json" "$A/cal/${ANCHOR}_calibration.json" "$A/anchor_check.json"
  gpu_free || return 1
  if ! verify_sft1 "$sft1"; then
    sft1=$ACK/_anchor_sft_v1           # never written into the shared $W/ck tree
    st "anchor: $SFT1_DIR absent or different; fetching $S3_SFT1"
    S3 s3 sync "$S3_SFT1" "$sft1" --exclude 'train_mix.jsonl' || return 1
    verify_sft1 "$sft1" || return 1
  fi
  echo "$sft1" > "$A/anchor_adapter_dir.txt"
  t0=$(date +%s)
  eval_ck "$ANCHOR" "$sft1" || return 1
  mode=$(serve_mode)
  replay_ck "$ANCHOR" "$sft1" "$mode"; rc=$?
  (( rc == 0 )) || return "$rc"
  box arm-time --run "$A" --arm "$ANCHOR" --eval-min "$(( ($(date +%s) - t0) / 60 ))"
  if dry_skip "anchor check"; then echo "$mode" > "$A/serve_mode.txt"; return 0; fi
  box anchor --run "$A" --serve-mode "$mode"; rc=$?
  if (( rc == 2 )) && [[ "$mode" == fast ]]; then   # fixed rule: only the C answers differ -> the Stage-2 (plain) server
    st "anchor: C answers differ under --fast; replaying with the plain Stage-2 server (fixed rule of the plan addendum)"
    mkdir -p "$A/s23/superseded"
    mv "$A/s23/answers_abl_$ANCHOR.jsonl" "$A/s23/superseded/answers_abl_${ANCHOR}_fast.jsonl"
    mv "$A/s23/replay_abl_$ANCHOR.json" "$A/s23/superseded/replay_abl_${ANCHOR}_fast.json"
    cp "$A/anchor_check.json" "$A/anchor_check_fast.json"
    mode=plain
    replay_ck "$ANCHOR" "$sft1" plain; rc=$?
    (( rc == 0 )) || return "$rc"
    box anchor --run "$A" --serve-mode plain; rc=$?
  fi
  echo "$mode" > "$A/serve_mode.txt"
  score_box
  if (( rc != 0 )); then st "anchor: NO-GO (anchor_check.json); no arm is trained"; return 1; fi
  st "anchor: GO (serving mode $mode)"
}
stage_abl_smoke() {  # memory: the recipe's setting first, then the fallbacks of ablation_box.SETTINGS
  local s=nogc_3072 tried="" rc smi peak
  registered || return 1
  gpu_free || return 1
  if [[ -s "$A/mem_setting.txt" && "${SMOKE_AGAIN:-0}" != 1 ]]; then echo "smoke done earlier: $(cat "$A/mem_setting.txt")"; return 0; fi
  dry_skip "smoke rows" || box smoke-file --run "$A" --out "$A/smoke/smoke.jsonl" || return 1
  while [[ "$s" != none ]]; do
    smi=""
    if [[ "$DRY_RUN" != 1 ]]; then
      nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -lms 1000 > "$A/logs/smoke_${s}_mem.csv" 2>/dev/null &
      smi=$!
    fi
    sft_run _smoke "$A/smoke/smoke.jsonl" 0 0 "$s" "$A/logs/smoke_$s.log" 4; rc=$?
    [[ -n "$smi" ]] && kill "$smi" 2>/dev/null
    [[ "$DRY_RUN" == 1 ]] || rm -rf "$ACK/_smoke"
    if (( rc == 0 )); then tried+="$s:ok"; break; fi
    if oom "$A/logs/smoke_$s.log"; then tried+="$s:oom,"; s=$(box setting --next "$s"); continue; fi
    st "smoke: $s failed for a reason other than memory (logs/smoke_$s.log)"
    return 1
  done
  [[ "$s" == none ]] && { st "smoke: every memory setting ran out of memory"; return 1; }
  echo "$s" > "$A/mem_setting.txt"
  peak=$(cat "$A"/logs/smoke_"${s}"_mem.csv 2>/dev/null | cut -d, -f1 | sort -n | tail -1)
  printf '{"chosen": "%s", "tried": "%s", "peak_mem_mib": "%s", "instance_type": "%s"}\n' "$s" "$tried" "${peak:-}" \
    "$(itype)" > "$A/smoke.json"
  st "smoke: memory setting $s (tried $tried; peak ${peak:-?} MiB)"
}
run_arm() {  # run_arm ARM -> 0 done | 1 failed | 9 priced record (stop everything)
  local arm=$1 out="$ACK/$1" data seed rf setting log rc t0 m=0 tried="" mix_sha
  [[ -f "$A/done/$arm" ]] && { echo "$arm: done earlier"; return 0; }
  data=$("$PY" -c "import sys; sys.path.insert(0, 'scripts/train'); import ablation_subsets as A; print(A.data_arm('$arm'))")
  seed=$("$PY" -c "import sys; sys.path.insert(0, 'scripts/train'); import ablation_subsets as A; print(A.trainer_seed('$arm'))")
  rf=$("$PY" -c "import json; print(json.load(open('results/training/ablation_inputs.json'))['arms']['$data']['replay_factor'])")
  [[ "$DRY_RUN" == 1 || -s "$A/data/$data.jsonl" ]] || { st "$arm: no data file $A/data/$data.jsonl"; return 1; }
  gpu_free || return 1
  if [[ -f "$A/settings/$arm.json" && -f "$out/adapter_model.safetensors" ]]; then
    st "$arm: trained earlier; resuming at the evaluation"
  else
    rm -f "$A/s23/answers_abl_$arm.jsonl" "$A/s23/replay_abl_$arm.json" "$A/eval/${arm}_v1.json" \
      "$A/cal/${arm}_calibration.json" cache/s1_abl_"${arm}"_*.sqlite
    setting=$(cat "$A/mem_setting.txt" 2>/dev/null || echo nogc_3072)
    while :; do
      log="$A/logs/sft_${arm}_$setting.log"
      t0=$(date +%s)
      sft_run "$arm" "$A/data/$data.jsonl" "$seed" "$rf" "$setting" "$log"; rc=$?
      m=$(( ($(date +%s) - t0) / 60 ))
      st "stage sft_$arm exit=$rc ($m min)"
      if (( rc == 0 )); then tried+="$setting:ok"; break; fi
      if oom "$log"; then
        tried+="$setting:oom,"
        setting=$(box setting --next "$setting")
        [[ "$setting" == none ]] && { st "$arm: every memory setting ran out of memory"; return 1; }
        echo "$setting" > "$A/mem_setting.txt"          # later arms start from the setting that fits
        st "$arm: out of memory; retrying with $setting"
        continue
      fi
      st "$arm: SFT failed (logs/sft_${arm}_$setting.log)"
      return 1
    done
    mix_sha=$(sha256sum "$out/train_mix.jsonl" 2>/dev/null | cut -c1-64)
    box record --run "$A" --arm "$arm" --setting "$setting" --attempts "$tried" --sft-min "$m" --itype "$(itype)" \
      --log "$log" --mix-sha "${mix_sha:-}" || return 1
    if [[ "$DRY_RUN" != 1 ]] && ! "$PY" -c "import json, sys; sys.exit(0 if json.load(open('$A/settings/$arm.json')).get('train_mix_ok') else 1)"; then
      st "$arm: the training file differs from results/training/ablation_inputs.json (train_mix): arm failed"
      return 1
    fi
    cp -f "$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -V | tail -1)/trainer_state.json" \
      "$A/logs/trainer_state_$arm.json" 2>/dev/null
    [[ "$DRY_RUN" == 1 ]] || rm -rf "$out"/checkpoint-* "$out/train_mix.jsonl"
  fi
  t0=$(date +%s)
  eval_ck "$arm" "$out" || return 1
  [[ "$DRY_RUN" == 1 ]] || cp -f "$A/cal/${arm}_calibration.json" "$out/calibration.json"
  sync_ck "$arm"
  replay_ck "$arm" "$out" "$(serve_mode)"; rc=$?
  (( rc == 0 )) || return "$rc"
  box arm-time --run "$A" --arm "$arm" --eval-min "$(( ($(date +%s) - t0) / 60 ))"
  touch "$A/done/$arm"
  score_box
  st "arm $arm done (setting $(cat "$A/mem_setting.txt" 2>/dev/null); eval + replay $(( ($(date +%s) - t0) / 60 )) min)"
}
arms_run() {  # arms_run SPEC: cap units in order; a unit that does not fit under CAP_H is skipped, the chain goes on
  local spec=$1 unit arm verdict rc fails=0
  registered || return 1
  if [[ "$DRY_RUN" != 1 ]]; then
    "$PY" -c "import json, sys; sys.exit(0 if json.load(open('$A/anchor_check.json')).get('ok') else 1)" 2>/dev/null \
      || { st "arms: no anchor GO in anchor_check.json"; return 1; }
    box inputs --run "$A" > /dev/null || { st "arms: inputs check failed"; return 1; }
    [[ -s "$A/mem_setting.txt" ]] || stage_abl_smoke || return 1
  fi
  while read -r unit; do
    [[ -z "$unit" ]] && continue
    # shellcheck disable=SC2086
    verdict=$(box cap --run "$A" --cap-h "$CAP_H" --arms $unit); rc=$?
    st "cap: $verdict"
    (( rc == 0 )) || continue
    for arm in $unit; do
      run_arm "$arm"; rc=$?
      (( rc == 9 )) && return 9
      if (( rc != 0 )); then
        fails=$((fails + 1)); st "arm $arm failed (rc $rc)"
        (( fails >= 2 )) && { st "two arms in a row failed: stopping the chain"; return 1; }
      else
        fails=0
      fi
    done
  done < <(box units "$spec")
}
stage_abl_core() { arms_run core; }
stage_abl_ext() { arms_run extended; }
stage_abl_arms() { arms_run "${1:?abl_arms:LIST}"; }
stage_abl_final() {
  local rc=0 used price
  stop_abl_server
  if ! dry_skip "final cost check"; then box cost --run "$A" || rc=1; fi
  X nvidia-smi > "$A/nvidia_smi_end.txt" 2>&1
  score_box
  used=$(box used-h --run "$A")
  price=$("$PY" -c "import json; print(json.load(open('$A/host.json')).get('usd_per_h') or '')" 2>/dev/null)
  printf '{"cost_check_ok": %s, "gpu_hours_used": %s, "usd_per_h": "%s", "arms_done": "%s", "finished_utc": "%s"}\n' \
    "$( (( rc == 0 )) && echo true || echo false)" "$used" "$price" "$(ls "$A/done" 2>/dev/null | tr '\n' ' ')" \
    "$(date -u +%FT%TZ)" > "$A/final_check.json"
  for arm in $(ls "$A/done" 2>/dev/null); do [[ "$arm" == "$ANCHOR" ]] || sync_ck "$arm"; done
  sync_out
  if [[ "$POWEROFF" == 1 ]]; then st "power-off in 2 min (POWEROFF=1)"; X sudo shutdown -h +2 "jev-train-ablation: done"; fi
  return $rc
}

broke=0
for tok in $STAGES; do
  s=${tok%%:*}; arg=""
  [[ "$tok" == *:* ]] && arg=${tok#*:}
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $tok"; continue; fi
  st "stage $tok start ($(git log --oneline -1 2>/dev/null | cut -c1-50); host $(itype))"
  t0=$(date +%s)
  printf '{"stage": "%s", "start": "%s"}\n' "$tok" "$(date -u +%FT%TZ)" > "$A/cur_stage.json"
  ( "stage_$s" ${arg:+"$arg"} ) > "$A/logs/stage_${tok//[:,+]/_}.log" 2>&1
  rc=$?
  secs=$(( $(date +%s) - t0 ))
  rm -f "$A/cur_stage.json"
  printf '{"stage": "%s", "min": %s}\n' "$tok" "$(awk -v s="$secs" 'BEGIN { printf "%.2f", s / 60 }')" >> "$A/stage_times.jsonl"
  st "stage $tok exit=$rc ($(( secs / 60 )) min)"
  if (( rc != 0 )) && [[ "$s" != abl_final ]]; then st "stopping the chain after $tok failed"; broke=1; break; fi
done
stop_abl_server
if (( broke )); then
  score_box
  sync_out
  st "chain stopped; the box stops itself after $IDLE_MIN idle minutes"
fi
st "box done [$STAGES]"
