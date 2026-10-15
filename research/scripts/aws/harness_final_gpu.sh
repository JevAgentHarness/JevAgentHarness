#!/usr/bin/env bash
# The final JevAgentHarness (exploratory; docs/PREREG_A7_STAGE2.md, Deviations entry of 2026-10-05 "harness_final";
# docs/JEVAGENT_HARNESS.md §8) on the persistent Stage-2 GPU host <instance-id> (tag Name=jev-stage2-startlux,
# one L40S 48 GB, g6e.2xlarge). One model on the GPU: the frozen JevAgentRL-4B (sha256-checked against
# results/stage2/rl_eval_plan.json's frozen row before it is served), behind training/serve_jev.py on :8081.
#
#   LOCAL:  bash scripts/aws/harness_final_gpu.sh push | send "STAGES" | status [N] | start | stop | itype
#   BOX stages:
#     rearm                     idle stop (IDLE_MIN) + total guard (GUARD_MIN), instance type, nvidia-smi
#     bench                     offline replay benchmark, no LLM call (scripts/harness_final.py bench): the Stage-2 server
#                               (v0) with the old and the new call graph, then the fast server (`--fast`, v2) with the
#                               new call graph; c = 1 (25 episodes), 2, 4, 8 (100 development episodes, think time
#                               emulated from the logged System-2 time)
#     serve_fast                the final harness's server (serve_jev --fast) for the live stages
#     tariff                    the harness_final configuration on the S2-4 tariff's development tasks (seeds 600-699,
#                               concurrency 4) inside a recorded serving window -> tariff window record
#     smoke                     5 development episodes of both live arms (seeds 900-904), operational check only
#     live                      runs/harness_final_live: llm_only and harness_final interleaved (task-major), 3 passes
#     final                     stop the server, sync
# Code: the committed branch (git bundle) + an overlay of this session's uncommitted files (sha256 in $OUT).
# Records: results/harness_final/gpu on the box -> s3://$BUCKET/harness_final_gpu/; runs -> .../runs/.
# While the live run is in flight only operational fields are read (errors, cost per episode, served model).
set -uo pipefail
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
IID=${IID:?set IID to the instance id}
BUNDLE=harness_final.bundle
OVERLAY=harness_final_overlay.tgz
S3OUT="$B/harness_final_gpu"
IDLE_MIN=${IDLE_MIN:-45}
GUARD_MIN=${GUARD_MIN:-600}
WORKERS=${WORKERS:-8}
OVERLAY_FILES=(scripts/aws/harness_final_gpu.sh scripts/harness_final.py
  src/jevharness/harness/agent.py src/jevharness/decision/types.py src/jevharness/decision/base.py
  src/jevharness/decision/jev.py src/jevharness/training/serve_jev.py src/jevharness/training/decision_model.py
  configs/harness/jevagent_final.yaml configs/experiments/harness_final_live.yaml configs/dev/harness_final_tariff.yaml
  configs/dev/harness_final_smoke.yaml results/harness_final/replay_workload.jsonl.gz configs/prices/models.yaml
  configs/experiments/harness_final_dedicated.yaml configs/dev/harness_final_tariff_dedicated.yaml
  docs/PREREG_A7_STAGE2.md)

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws --region "$REGION" "$@"; }

if [[ "$MODE" == push ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors
  tar -C "$ROOT" -czf "$tmp/$OVERLAY" "${OVERLAY_FILES[@]}"
  awsl s3 cp "$tmp/$OVERLAY" "$B/code/$OVERLAY" --only-show-errors
  cp "$ROOT/scripts/aws/harness_final_gpu.sh" "$tmp/harness_final_gpu.sh"
  awsl s3 cp "$tmp/harness_final_gpu.sh" "$B/code/harness_final_gpu.sh" --only-show-errors
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH"); overlay $(sha256sum "$tmp/$OVERLAY" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/hf_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/harness_final_gpu.sh \$f --only-show-errors --region $REGION"
    printf 'export IDLE_MIN=%q GUARD_MIN=%q WORKERS=%q LIVE_REPLICAS=%q WAIT_PREV=%q\n' "$IDLE_MIN" "$GUARD_MIN" "$WORKERS" \
      "${LIVE_REPLICAS:-1}" "${WAIT_PREV:-0}"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/hf_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/hf_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/hf_driver.sh /tmp/hf_driver.sh --only-show-errors && bash /tmp/hf_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == start || "$MODE" == stop ]]; then
  awsl ec2 "$MODE-instances" --instance-ids "$IID" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
  exit 0
fi
if [[ "$MODE" == itype ]]; then
  awsl ec2 describe-instances --instance-ids "$IID" --output text \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]'
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,24p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
W=/opt/jev
cd "$W" || exit 1
if [[ "${WAIT_PREV:-0}" == 1 ]]; then   # queue behind a chain that is still running (never reset its working tree)
  while ps -eo args | grep "hf_[0-9]*\.sh box" | grep -v -F "$0" | grep -q .; do sleep 30; done
fi
myb="$W/hf.$$.bundle"
aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || exit 1
[[ -d "$W/stage2" ]] || git clone -q -b "$BRANCH" "$myb" "$W/stage2"
cd "$W/stage2" || exit 1
# keep the previous chain's status lines: status.txt is tracked, so the reset would restore the committed copy
st_keep="$W/hf_status.$$.txt"; cp -f results/harness_final/gpu/status.txt "$st_keep" 2>/dev/null || true
git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD
[[ -s "$st_keep" ]] && mkdir -p results/harness_final/gpu && mv -f "$st_keep" results/harness_final/gpu/status.txt
rm -f "$myb"
aws s3 cp "$B/code/$OVERLAY" "$W/$OVERLAY" --only-show-errors --region "$REGION" && tar -xzf "$W/$OVERLAY" -C "$W/stage2"
OUT=results/harness_final/gpu
mkdir -p "$OUT" runs cache
sha256sum "$W/$OVERLAY" > "$OUT/overlay.sha256"
HV=$W/hvenv
CK=$W/ck
PLAN=results/stage2/rl_eval_plan.json
RL_URL=http://127.0.0.1:8081
LIVE=configs/experiments/harness_final_live.yaml
LIVE_REPLICAS=${LIVE_REPLICAS:-1}
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$OUT/status.txt"
  aws s3 sync "$OUT" "$S3OUT" --only-show-errors --region "$REGION"
}
sync_run() { [[ -d "$1" ]] && aws s3 sync "$1" "$S3OUT/$1" --only-show-errors --region "$REGION" --exclude '*.sqlite'; }
plan() { "$HV/bin/python" -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"; }
itype() {
  curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/instance-type
}
keys() {  # the System-2 key from SSM (never on disk, never printed)
  local v val
  for v in GEMINI_API_KEY GEMINI_API_KEYS; do
    val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value \
          --output text 2>/dev/null) || val=""
    if [[ -n "$val" && "$val" != None ]]; then export "$v=$val"; fi
  done
  unset val
  [[ -n "${GEMINI_API_KEY:-}${GEMINI_API_KEYS:-}" ]] || { echo "keys missing in SSM"; return 1; }
}
arm_stops() {  # total guard + idle stop (cron, IDLE_MIN), as scripts/aws/explore_gpu.sh
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "jev-harness-final: total guard" 2>/dev/null
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without a chain / runner / server / install process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'hf_[0-9]|g2_[0-9]|ex_[0-9]|jevharness|jevbench|harness_final|startlux_decision|llama-server|uv pip|hf download' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-harness-final: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
}
start_rec() {  # START file with the overlay and registration hashes
  printf 'step %s\nstarted %s\ncommit %s\noverlay sha256 %s\nprereg sha256 %s\nhost %s\n%s\n' "$1" \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$(cut -c1-16 "$OUT/overlay.sha256")" \
    "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" "$(itype)" "${2:-}" > "$OUT/START_$1.txt"
}

stage_rearm() { arm_stops; itype > "$OUT/instance_type.txt"; nvidia-smi > "$OUT/nvidia_smi_start.txt" 2>&1;
  "$HV/bin/python" -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'cuda', torch.cuda.is_available())" > "$OUT/env.txt" 2>&1
  [[ -f "$W/base_model_path.txt" ]] && cp "$W/base_model_path.txt" "$OUT/base_model_path.txt"; cat "$OUT/env.txt"; }

verify_ck() {
  "$HV/bin/python" - "$PLAN" "$CK" "$OUT/checkpoint_verify_frozen.json" <<'EOF'
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
serve() {  # serve TAG [serve_jev flags...]: the frozen row of the plan behind serve_jev; REPLICAS processes on :8081..
  local tag=$1 adapter cal name k port
  shift
  [[ -f "$W/base_model_path.txt" ]] || { echo "no base_model_path.txt"; return 1; }
  verify_ck || return 1
  adapter=$(plan "d['rows']['frozen'].get('adapter') or ''")
  cal=$(plan "d['rows']['frozen'].get('calibration') or ''")
  name=$(plan "d['rows']['frozen']['served_name']")
  [[ "$name" == jevagentrl-4b ]] || { echo "served name $name != jevagentrl-4b"; return 1; }
  stop_server
  local n=${REPLICAS:-1} extra=()
  (( n > 1 )) && extra=(--cache-tokens ${REPLICA_CACHE_TOKENS:-12288})
  URLS=""
  for k in $(seq 0 $((n - 1))); do
    port=$((8081 + k))
    local args=(--model "$(cat "$W/base_model_path.txt")" --name "$name" --port "$port" --adapter "$CK/$adapter"
                --calibration "$CK/$cal" "${extra[@]}" "$@")
    echo "serve_jev ${args[*]}" >> "$OUT/serve_$tag.cmd"
    JEV_SERVE_PROFILE=${PROFILE:-0} nohup "$HV/bin/python" -m jevharness.training.serve_jev "${args[@]}" \
      > "$OUT/serve_${tag}_$k.out" 2>&1 &
    echo $! >> "$W/serve_ck.pid"
    URLS="${URLS:+$URLS,}http://127.0.0.1:$port"
  done
  for k in $(seq 0 $((n - 1))); do
    port=$((8081 + k))
    local up=0
    for _ in $(seq 1 180); do
      if curl -sf "http://127.0.0.1:$port/health" > "$OUT/serve_${tag}_${k}_health.json"; then
        curl -sf "http://127.0.0.1:$port/v1/models" > "$OUT/serve_${tag}_${k}_models.json"; up=1; break
      fi
      sleep 5
    done
    (( up )) || { echo "serve_jev :$port did not come up"; tail -20 "$OUT/serve_${tag}_$k.out"; return 1; }
  done
  echo "$URLS" > "$W/serve_urls.txt"
}
stop_server() {
  [[ -f "$W/serve_ck.pid" ]] && while read -r p; do kill "$p" 2>/dev/null; done < "$W/serve_ck.pid"
  rm -f "$W/serve_ck.pid"; sleep 5
}
bench() {  # bench LABEL PATH C... : replay against the running server(s) (no LLM call), GPU utilisation sampled
  local label=$1 path=$2
  shift 2
  local c smi
  for c in "$@"; do
    local lim=()
    [[ $c == 1 ]] && lim=(--limit 25)
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 \
      > "$OUT/bench/smi_${label}_${path}_c$c.csv" 2>/dev/null &
    smi=$!
    "$HV/bin/python" scripts/harness_final.py bench --url "$(cat "$W/serve_urls.txt")" --label "$label" --path "$path" \
      --concurrency "$c" --out "$OUT/bench" "${lim[@]}" >> "$OUT/bench.log" 2>&1
    local rc=$?
    kill "$smi" 2>/dev/null
    (( rc == 0 )) || return 1
    st "bench $label $path c=$c done"
  done
}

stage_bench() {
  mkdir -p "$OUT/bench"
  # phase profiles first (JEV_SERVE_PROFILE=1: device-synchronised phase timers, c = 1, 25 episodes), then timed runs
  PROFILE=1 serve v0p || return 1
  bench v0p old 1 || return 1
  bench v0p new 1 || return 1
  PROFILE=1 serve v2p --fast || return 1
  bench v2p new 1 || return 1
  serve v0 || return 1                         # the Stage-2 server (defaults)
  bench v0 old 1 4 8 || return 1
  bench v0 new 4 8 || return 1
  serve v2 --fast || return 1                  # the final harness's server
  bench v2 new 1 2 4 8 || return 1
  bench v2 old 4 || return 1
  REPLICAS=2 serve v2r2 --fast || return 1     # two replicas of the fast server on the one GPU, sticky per episode
  bench v2r2 new 4 8 || return 1
  stop_server
}
stage_serve_fast() { REPLICAS=${LIVE_REPLICAS:-1} serve fast --fast; }

check_replicas() {  # check_replicas CONFIG: the config's System-1 replicas = the servers that are up
  local want
  want=$("$HV/bin/python" -c "import yaml,sys; b=yaml.safe_load(open(sys.argv[1]))['s1_backends']['jevagent_rl']; print(len(b.get('replicas') or [b['base_url']]))" "$1")
  [[ "$want" == "$(tr ',' '\n' < "$W/serve_urls.txt" | grep -c .)" ]] || { echo "$1 wants $want replica(s); serving $(cat "$W/serve_urls.txt")"; return 1; }
}

stage_tariff() {  # S2-4's tariff workload (development seeds 600-699, concurrency 4) with the final configuration
  keys || return 1
  curl -sf "$RL_URL/health" > /dev/null || { echo "server not up"; return 1; }
  check_replicas configs/dev/harness_final_tariff.yaml || return 1
  start_rec harness_final_tariff "configs/dev/harness_final_tariff.yaml"
  local t0 t1
  t0=$(date -u +%FT%TZ)
  "$HV/bin/jevbench" run configs/dev/harness_final_tariff.yaml --out runs -o workers=4 > "$OUT/tariff.log" 2>&1
  local rc=$?
  t1=$(date -u +%FT%TZ)
  printf '{"window_start": "%s", "window_end": "%s", "exit": %d, "run": "runs/harness_final_tariff", "instance_type": "%s", "concurrency": 4}\n' \
    "$t0" "$t1" "$rc" "$(itype)" > "$OUT/tariff_window.json"
  sync_run runs/harness_final_tariff
  return $rc
}
stage_smoke() {  # 5 development episodes of both live arms (seeds 900-904): operational fields only
  keys || return 1
  curl -sf "$RL_URL/health" > /dev/null || { echo "server not up"; return 1; }
  check_replicas configs/dev/harness_final_smoke.yaml || return 1
  "$HV/bin/jevbench" run configs/dev/harness_final_smoke.yaml --out runs -o "workers=$WORKERS" > "$OUT/smoke.log" 2>&1
  local rc=$?
  ops runs/harness_final_smoke >> "$OUT/smoke_ops.txt"
  sync_run runs/harness_final_smoke
  return $rc
}
ops() {  # operational fields only: tasks, errored, $/episode, System-1 models
  "$HV/bin/python" - "$1" <<'EOF'
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
stage_live() {  # the registered-in-Deviations live confirmation: three passes (errored tasks re-run), interleaved arms
  keys || return 1
  curl -sf "$RL_URL/health" > /dev/null || { echo "server not up"; return 1; }
  check_replicas "$LIVE" || return 1
  grep -q 'harness_final' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for harness_final"; return 1; }
  start_rec harness_final_live "config $LIVE ($(sha256sum "$LIVE" | cut -c1-16)); profile $(sha256sum configs/harness/jevagent_final.yaml | cut -c1-16); workers $WORKERS"
  curl -sf "$RL_URL/v1/models" > "$OUT/live_server_models_start.json"
  local pass
  for pass in 1 2 3; do
    JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$LIVE" --out runs -o "workers=$WORKERS" > "$OUT/live_pass$pass.log" 2>&1
    st "live pass $pass exit=$?"
    ops runs/harness_final_live >> "$OUT/live_ops.txt"
    sync_run runs/harness_final_live
  done
  curl -sf "$RL_URL/v1/models" > "$OUT/live_server_models_end.json"
}
stage_tariff_ded() {  # the dedicated-GPU tariff: S2-4's tariff tasks (dev seeds 600-699), ONE episode at a time
  keys || return 1
  curl -sf "$RL_URL/health" > /dev/null || { echo "server not up"; return 1; }
  check_replicas configs/dev/harness_final_tariff_dedicated.yaml || return 1
  start_rec harness_final_tariff_dedicated "configs/dev/harness_final_tariff_dedicated.yaml"
  local t0 t1
  t0=$(date -u +%FT%TZ)
  "$HV/bin/jevbench" run configs/dev/harness_final_tariff_dedicated.yaml --out runs -o workers=1 > "$OUT/tariff_ded.log" 2>&1
  local rc=$?
  t1=$(date -u +%FT%TZ)
  printf '{"window_start": "%s", "window_end": "%s", "exit": %d, "run": "runs/harness_final_tariff_dedicated", "instance_type": "%s", "concurrency": 1}\n' \
    "$t0" "$t1" "$rc" "$(itype)" > "$OUT/tariff_dedicated_window.json"
  sync_run runs/harness_final_tariff_dedicated
  return $rc
}
stage_live_ded() {  # Deviations 2026-10-05 harness_final_dedicated: one worker (<= 1 harness episode on the GPU), 3 passes
  keys || return 1
  curl -sf "$RL_URL/health" > /dev/null || { echo "server not up"; return 1; }
  local CFG=configs/experiments/harness_final_dedicated.yaml
  check_replicas "$CFG" || return 1
  grep -q 'harness_final_dedicated' docs/PREREG_A7_STAGE2.md || { echo "no Deviations entry for harness_final_dedicated"; return 1; }
  start_rec harness_final_dedicated "config $CFG ($(sha256sum "$CFG" | cut -c1-16)); profile $(sha256sum configs/harness/jevagent_final.yaml | cut -c1-16); workers 1"
  local pass
  for pass in 1 2 3; do
    JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run "$CFG" --out runs -o workers=1 > "$OUT/live_ded_pass$pass.log" 2>&1
    st "live_ded pass $pass exit=$?"
    ops runs/harness_final_dedicated >> "$OUT/live_ded_ops.txt"
    sync_run runs/harness_final_dedicated
  done
}
stage_final() {
  stop_server
  nvidia-smi > "$OUT/nvidia_smi_end.txt" 2>&1
  for d in runs/harness_final_live runs/harness_final_tariff runs/harness_final_smoke runs/harness_final_dedicated \
           runs/harness_final_tariff_dedicated; do sync_run "$d"; done
}

for s in $STAGES; do
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $s"; continue; fi
  st "stage $s start ($(git log --oneline -1 | cut -c1-50); overlay $(cut -c1-12 "$OUT/overlay.sha256"))"
  t0=$(date +%s)
  ( "stage_$s" ) > "$OUT/$s.log" 2>&1
  rc=$?
  st "stage $s exit=$rc ($(( ($(date +%s) - t0) / 60 )) min)"
  if [[ $rc -ne 0 && $s != final ]]; then st "stopping the chain after $s failed"; broke=1; break; fi
done
if [[ "${broke:-0}" == 1 || " $STAGES " == *" final "* ]]; then
  stop_server
  st "server stopped; the box stops itself after $IDLE_MIN idle minutes"
fi
st "box done [$STAGES]"
