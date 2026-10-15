#!/usr/bin/env bash
# The remaining Stage-2 GPU chain (docs/PREREG_A7_STAGE2.md A7.10, A7.13, A7.1 (f), A7.18 (e) 2) on ONE persistent box:
# the Stage-2 L40S host <instance-id> (tag Name=jev-stage2-startlux, g6e.2xlarge, 1x L40S 48 GB). Everything is
# queued in one `send`; every stage syncs to s3://$BUCKET/stage2_gpu/ as it goes, so the chain completes without this
# session. Two GATES wait (up to GATE_MIN) for this session to COMMIT what the box produced before it is used:
#   gate1  results/stage2/tariff.json (A7.13 (iii): q frozen before any priced evaluation episode); the profile's
#          backends.jevagent_rl block is filled from the operator's post-freeze fits (results/gpu_round5/cal/, A7.1 (f))
#          and committed before the chain starts, so gate1 only re-checks it;
#   gate2  the frozen common-state corpus results/stage2/s23/{corpus.json,occurrences,questions} (A7.10 (c): C frozen
#          before any answer is scored; replay order = sorted question hash, cache and backend description: this script).
# A gate passes when a newer code bundle on S3 carries byte-identical files; on timeout the chain goes on with the box's
# own files and status.txt says so (a dated Deviations entry then).
#
#   LOCAL:  bash scripts/aws/stage2_gpu.sh push | send "STAGES" | status [N] | start | stop
#   BOX:    the full chain, in the registered order (one model on the GPU at a time):
#     setup fetch serve_rl tariff_rl gate1 s24 states_rl e1_rl stop_rl states_jev states_intern corpus gate2
#     replay_jev replay_intern serve_rl replay_rl stop_rl attribution replay_sl4b ladder replay_sl27b final
#   then (A7.13 (iv), after C7 and the freeze): rearm t2ref final  (IDLE_MIN / GUARD_MIN re-armed after a restart)
#   then (A7.19 (f), JevAgentRL-v2, exploratory; results/stage2/v2_eval_plan.json; the box waits for two commits):
#     rearm v2_setup v2_fetch v2_serve v2_cal v2_gate_tau v2_tariff v2_gate_tariff v2_profile v2_states v2_replay v2_final final
# The RL checkpoints, their temperature files and the rows of A7.13 (i) come from results/stage2/rl_eval_plan.json
# (written from the freeze record, committed before the chain); sha256 of every downloaded artefact is checked against
# the freeze record's list before any request. While a config runs only operational fields are read (A7.6).
set -uo pipefail
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
TAG=${TAG:-jev-stage2-startlux}
BUNDLE=stage2_gpu.bundle
S3OUT="$B/stage2_gpu"
GATE_MIN=${GATE_MIN:-240}        # how long a gate waits for the commit
IDLE_MIN=${IDLE_MIN:-120}        # the box stops this long after the queue empties
GUARD_MIN=${GUARD_MIN:-2880}     # total guard (48 h)

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws --region "$REGION" "$@"; }
instance_id() {
  awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text
}

if [[ "$MODE" == push ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors
  git -C "$ROOT" show "$BRANCH:scripts/aws/stage2_gpu.sh" > "$tmp/stage2_gpu.sh"
  awsl s3 cp "$tmp/stage2_gpu.sh" "$B/code/stage2_gpu.sh" --only-show-errors
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH")"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  STAGES=${1:?usage: send \"STAGES\"}
  iid=$(instance_id)
  [[ -n "$iid" ]] || { echo "no $TAG instance" >&2; exit 1; }
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/g2_$(date +%s%N).sh'      # a private copy per invocation (bash reads a running script lazily)
    echo "aws s3 cp $B/code/stage2_gpu.sh \$f --only-show-errors --region $REGION"
    printf 'export IDLE_MIN=%q GUARD_MIN=%q\n' "$IDLE_MIN" "$GUARD_MIN"   # the stop settings travel with the queue
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/g2_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/g2_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$iid" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/g2_driver.sh /tmp/g2_driver.sh --only-show-errors && bash /tmp/g2_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" == stop || "$MODE" == start ]]; then
  iid=$(instance_id)
  [[ -n "$iid" ]] || { echo "no $TAG instance" >&2; exit 1; }
  awsl ec2 "$MODE-instances" --instance-ids "$iid" --output text \
    --query "$( [[ $MODE == stop ]] && echo StoppingInstances || echo StartingInstances )[].[InstanceId,CurrentState.Name]"
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
fetch_bundle() {  # fetch_bundle -> FETCH_HEAD = the newest pushed bundle
  local myb="$W/bundle.$$.bundle"
  aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || return 1
  git -C "$W/stage2" fetch -q "$myb" "$BRANCH" 2>/dev/null
  local rc=$?
  rm -f "$myb"
  return $rc
}
if [[ ! -d "$W/stage2" ]]; then
  aws s3 cp "$B/code/$BUNDLE" "$W/b.bundle" --only-show-errors --region "$REGION" && git clone -q -b "$BRANCH" "$W/b.bundle" "$W/stage2"
fi
cd "$W/stage2" || exit 1
fetch_bundle && git reset -q --hard FETCH_HEAD
OUT=results/stage2/gpu                       # box records (synced to S3; committed from the cloud session)
S23=results/stage2/s23
HV=$W/hvenv                                  # the harness venv (Intern-Decision in-process, serve_jev, the runner)
CK=$W/ck                                     # the checkpoints of the freeze record (rl_eval_plan.json)
PLAN=results/stage2/rl_eval_plan.json
export BASE=/opt/startlux PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
mkdir -p "$OUT" "$S23" runs "$CK"
RL_URL=http://127.0.0.1:8081
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$OUT/status.txt"
  aws s3 sync "$OUT" "$S3OUT" --only-show-errors --region "$REGION"
}
sync_run() { aws s3 sync "runs/$1" "$S3OUT/runs/$1" --only-show-errors --region "$REGION"; }
sync_s23() { aws s3 sync "$S23" "$S3OUT/s23" --only-show-errors --region "$REGION"; }
plan() { "$HV/bin/python" -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"; }
keys() {  # the System-2 / Jev keys from SSM (never on disk, never printed)
  local v val
  for v in GEMINI_API_KEY GEMINI_API_KEYS OPENROUTER_API_KEY; do
    val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value \
          --output text 2>/dev/null) || val=""
    if [[ -n "$val" && "$val" != None ]]; then export "$v=$val"; fi
  done
  unset val
  [[ -n "${GEMINI_API_KEY:-}${GEMINI_API_KEYS:-}" && -n "${OPENROUTER_API_KEY:-}" ]] || { echo "keys missing in SSM"; return 1; }
}
ops() {  # operational fields only: tasks, latest-errored, cost per episode, System-1 models
  "$HV/bin/python" - "$@" <<'EOF'
import json, sys, glob, os, collections
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        last = {}
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "task_id" in r:
                last[r["task_id"]] = r
        if not last:
            continue
        errs = sum(1 for r in last.values() if r.get("error"))
        cost = sum(float(((r.get("usage") or {}).get("total") or {}).get("cost_usd") or 0) for r in last.values())
        m = collections.Counter(c.get("model") for r in last.values() for c in r.get("calls") or [] if c.get("component") == "s1")
        print(f"{d}/{os.path.basename(f)[:-6]}: tasks {len(last)} latest-errored {errs} cost/ep ${cost / len(last):.4f} s1 {dict(m)}")
EOF
}
three() {  # three NAME CMD...: the three registered passes (EVAL=1 sets JEV_EVAL_RUNS=1 for evaluation commands only)
  local name=$1 rc; shift
  for pass in 1 2 3; do
    if [[ "${EVAL:-0}" == 1 ]]; then JEV_EVAL_RUNS=1 "$@"; else "$@"; fi
    rc=$?
    st "$name pass $pass exit=$rc"
  done
}
start_rec() {  # start_rec STEP: a START file in the run_main.sh format (once)
  [[ -f "$OUT/START_$1.txt" ]] || printf 'step %s\nstarted %s\ncommit %s\nplan sha256 %s\nhost %s\n' "$1" \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" \
    "$(cat "$OUT/instance_type.txt" 2>/dev/null)" > "$OUT/START_$1.txt"
}
gate() {  # gate RUN [ARGS]: the registered launch gate on this host's tree
  "$HV/bin/python" scripts/stage2_spec.py --gate "$@" > "$OUT/gate_${1}${2:+_${2#--}}.txt" 2>&1 \
    || { echo "launch gate $* FAILED"; return 2; }
}
rl_spec() {  # rl_spec NAME: the Jev-wire spec of the checkpoint served on :8081 under NAME
  printf '{type: jev, name: %s, base_url: "%s", model: %s, api_key: ""}' "$1" "$RL_URL" "$1"
}

serve_ck() {  # serve_ck ROW: the row's checkpoint behind serve_jev on :8081 (one at a time), waits for /health
  local row=$1 adapter cal name args
  adapter=$(plan "d['rows']['$row'].get('adapter') or ''")
  cal=$(plan "d['rows']['$row'].get('calibration') or ''")
  name=$(plan "d['rows']['$row']['served_name']")
  stop_ck
  args=(--model "$(cat "$W/base_model_path.txt")" --name "$name" --port 8081)   # the pinned base snapshot (stage fetch)
  [[ -n "$adapter" ]] && args+=(--adapter "$CK/$adapter")
  if [[ -n "$cal" ]]; then [[ -f "$cal" ]] && args+=(--calibration "$cal") || args+=(--calibration "$CK/$cal"); fi
  # shellcheck disable=SC2207
  args+=($(plan "' '.join(d.get('serve_args') or [])"))
  echo "serve_jev ${args[*]}" > "$OUT/serve_$row.cmd"
  nohup "$HV/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$OUT/serve_$row.out" 2>&1 &
  echo $! > "$W/serve_ck.pid"
  for _ in $(seq 1 180); do
    if curl -sf "$RL_URL/health" > "$OUT/serve_${row}_health.json"; then
      curl -sf "$RL_URL/v1/models" > "$OUT/serve_${row}_models.json"; return 0
    fi
    kill -0 "$(cat "$W/serve_ck.pid")" 2>/dev/null || { echo "serve_jev died"; tail -20 "$OUT/serve_$row.out"; return 1; }
    sleep 5
  done
  echo "serve_jev did not come up"; return 1
}
stop_ck() {
  [[ -f "$W/serve_ck.pid" ]] && kill "$(cat "$W/serve_ck.pid")" 2>/dev/null
  rm -f "$W/serve_ck.pid"; sleep 5
}

arm_stops() {  # the total guard (a scheduled shutdown does not survive a reboot) and the idle stop (cron, IDLE_MIN)
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "jev-stage2-gpu: total guard" 2>/dev/null
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without a chain / runner / server / install process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'g2_|jevharness|jevbench|startlux_decision|llama-server|uv pip|hf download' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-stage2-gpu: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
}
stage_rearm() { arm_stops; }   # after a restart (e.g. an instance-type change): send "rearm ..." with IDLE_MIN / GUARD_MIN
stage_wait_chain() {  # wait until every other chain on this box has exited (queue behind it without this session)
  local others
  while others=$(pgrep -af 'g2_[0-9]+\.sh box' | grep -v -F -- "$0" | grep -v pgrep); [[ -n "$others" ]]; do sleep 60; done
}
stage_setup() {  # guard + idle stop, host facts, the harness venv with the serving / Intern stack, the sl4b box run moved
  arm_stops
  nvidia-smi > "$OUT/nvidia_smi_start.txt" 2>&1
  curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/instance-type > "$OUT/instance_type.txt"
  command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; } || return 1
  [[ -x "$HV/bin/python" ]] || uv venv -q -p 3.11 "$HV" || return 1
  uv pip install -q -p "$HV/bin/python" -e ".[dev,bench,serve]" || return 1
  local specs
  specs=$(plan "' '.join(d.get('gpu_python_specs') or [])" 2>/dev/null)
  "$HV/bin/python" -c "import torch, transformers, peft" 2>/dev/null || {
    uv pip install -q -p "$HV/bin/python" "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130 &&
    uv pip install -q -p "$HV/bin/python" "transformers==5.18.0" "accelerate>=1.0" safetensors peft; } || return 1
  # shellcheck disable=SC2086
  [[ -n "$specs" ]] && { uv pip install -q -p "$HV/bin/python" $specs || return 1; }
  "$HV/bin/python" -c "import torch, transformers, peft, fastapi, uvicorn; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'cuda', torch.cuda.is_available())" \
    > "$OUT/env.txt" 2>&1; cat "$OUT/env.txt"
  uv pip freeze -p "$HV/bin/python" > "$OUT/hvenv_pip_freeze.txt"
  # the S2-4S box run of stage2_jevgym is archived beside, so this run directory holds only this chain's arm
  if [[ -d runs/stage2_jevgym && ! -d runs/_sl4b_box_stage2_jevgym ]]; then mv runs/stage2_jevgym runs/_sl4b_box_stage2_jevgym; fi
  grep -q 'torch' "$OUT/env.txt"
}

stage_fetch() {  # every checkpoint row of the plan from S3, verified against the freeze record's sha256 list
  local row src dst
  for row in $(plan "' '.join(d['rows'])"); do
    src=$(plan "d['rows']['$row'].get('s3') or ''")
    dst=$(plan "d['rows']['$row'].get('dir') or ''")
    [[ -z "$src" ]] && continue
    mkdir -p "$CK/$dst"
    aws s3 sync "$src" "$CK/$dst" --only-show-errors --region "$REGION" || return 1
  done
  # the base model at the snapshot of the freeze record (base_model_hf_snapshot), served from its local path
  "$HV/bin/python" -c "import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], revision=sys.argv[2]))" \
    "$(plan "d['base_model_repo']")" "$(plan "d['base_model_revision']")" > "$W/base_model_path.txt" || return 1
  cp "$W/base_model_path.txt" "$OUT/base_model_path.txt"
  "$HV/bin/python" - "$PLAN" "$CK" "$OUT/checkpoint_verify.json" <<'EOF' || return 1
import hashlib, json, os, sys
plan, ck, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(plan))
res, bad = {}, []
for row, r in d["rows"].items():
    for rel, sha in list((r.get("sha256") or {}).items()) + [(f"REPO:{k}", v) for k, v in (r.get("repo_sha256") or {}).items()]:
        p = rel[5:] if rel.startswith("REPO:") else os.path.join(ck, r.get("dir") or "", rel)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
        res[f"{row}/{rel}"] = {"expected": sha, "got": got, "ok": got == sha}
        if got != sha:
            bad.append(f"{row}/{rel}")
json.dump({"ok": not bad, "mismatch": bad, "files": res}, open(out, "w"), indent=1)
print("checkpoint sha256:", "all match" if not bad else f"MISMATCH {bad}")
sys.exit(1 if bad else 0)
EOF
}

stage_serve_rl() { serve_ck frozen; }
stage_stop_rl() { stop_ck; }

stage_tariff_rl() {  # A7.13 (iii): the committed plan results/stage2/tariff_rl_plan.json (dev seeds), window recorded
  keys || return 1
  curl -sf "$RL_URL/health" > /dev/null || { echo "RL server not up"; return 1; }
  local p=results/stage2/tariff_rl_plan.json wseed wn seed n name usd
  read -r wseed wn seed n name usd < <("$HV/bin/python" -c "import json; p = json.load(open('$p'));
print(p['warmup']['seed0'], p['warmup']['n'], p['workload']['seed0'], p['workload']['n'], p['workload']['run_name'], p['host']['usd_per_hour'])")
  local arm="[{name: jevagent_best_rl, s1: jevagent_rl, profile: configs/harness/jevagent_best.yaml, profile_s2: gemini-3.5-flash, profile_backend: jevagent_rl}]"
  local common=(-o workers=4 -o "s1_backends.jevagent_rl=$(rl_spec jevagentrl-4b)" -o "arms=$arm" -o s1_cache=cache/s1_dev_tariff_rl.sqlite)
  "$HV/bin/jevbench" run configs/experiments/stage2_jevgym.yaml --out runs -o "name=${name}_warmup" -o suite.seed0="$wseed" \
    -o suite.n="$wn" -o n_tasks="$wn" "${common[@]}" > "$OUT/tariff_rl_warmup.log" 2>&1
  local t0 t1 rc
  t0=$(date -u +%FT%TZ)
  "$HV/bin/jevbench" run configs/experiments/stage2_jevgym.yaml --out runs -o "name=$name" -o suite.seed0="$seed" \
    -o suite.n="$n" -o n_tasks="$n" "${common[@]}" > "$OUT/tariff_rl_workload.log" 2>&1
  rc=$?
  t1=$(date -u +%FT%TZ)
  printf '{"window_start": "%s", "window_end": "%s", "exit": %s, "run": "runs/%s", "instance_type": "%s"}\n' "$t0" "$t1" \
    "$rc" "$name" "$(cat "$OUT/instance_type.txt")" > "$OUT/tariff_rl_window.json"
  ops "runs/$name" > "$OUT/tariff_rl_ops.txt"
  sync_run "$name"; sync_run "${name}_warmup"
  [[ $rc -eq 0 ]] || return $rc
  local sens
  sens=$("$HV/bin/python" - "runs/$name/jevagent_best_rl.jsonl" "$usd" <<'EOF'
import datetime as dt, json, sys
iv, tok, n = [], 0, 0
for line in open(sys.argv[1]):
    r = json.loads(line); n += 1
    for c in r.get("calls") or []:
        if c.get("component") == "s1" and not c.get("cached"):
            end = dt.datetime.fromisoformat(c["ts"]).timestamp(); iv.append((end - c["latency_ms"] / 1000, end))
            tok += int(c.get("input_tokens") or 0)
iv.sort(); busy = 0.0; cs, ce = iv[0]
for s, e in iv[1:]:
    if s > ce: busy += ce - cs; cs, ce = s, e
    else: ce = max(ce, e)
busy += ce - cs
print(f"--sensitivity busy_throughput={float(sys.argv[2]) * busy / 3600 / n} --sensitivity jev_api_price={tok * 0.042 / 1e6 / n} --sensitivity zero=0.0")
EOF
)
  # shellcheck disable=SC2086
  "$HV/bin/python" scripts/stage2_tariff.py --run "runs/$name" --arm jevagent_best_rl --served-model jevagentrl-4b \
    --host-usd-per-hour "$usd" --window-start "$t0" --window-end "$t1" \
    --hardware "1x NVIDIA L40S 48 GB ($(cat "$OUT/instance_type.txt"), us-east-1, on demand)" --precision bf16 \
    --concurrency 4 $sens --workload-matches --out results/stage2/tariff.json > "$OUT/tariff_rl.txt" 2>&1 || return 1
  mkdir -p "$OUT/gate1"; cp results/stage2/tariff.json "$OUT/tariff_rl_window.json" "$OUT/gate1/"
}

wait_commit() {  # wait_commit NAME FILE...: until a pushed bundle carries byte-identical FILEs (or GATE_MIN passes)
  local name=$1; shift
  local t_end=$(( $(date +%s) + GATE_MIN * 60 )) f ok
  st "$name: waiting up to $GATE_MIN min for the commit of: $*"
  while (( $(date +%s) < t_end )); do
    if fetch_bundle; then
      ok=1
      for f in "$@"; do
        [[ "$(git show "FETCH_HEAD:$f" 2>/dev/null | sha256sum | cut -c1-64)" == "$(sha256sum < "$f" | cut -c1-64)" ]] || ok=0
      done
      if [[ $ok == 1 ]]; then
        git reset -q --hard FETCH_HEAD
        st "$name: committed at $(git log --oneline -1 | cut -c1-70)"
        return 0
      fi
    fi
    sleep 120
  done
  st "$name: TIMEOUT after $GATE_MIN min: going on with the box's own files (uncommitted; record a Deviations entry)"
  return 0
}
stage_gate1() {
  wait_commit gate1 configs/harness/jevagent_best.yaml results/stage2/tariff.json
}

stage_s24() {  # S2-4: jevagent_best_rl on the fresh seeds (stage2_jevgym, this arm only), three passes, gate --rl
  keys || return 1
  curl -sf "$RL_URL/health" > /dev/null || { echo "RL server not up"; return 1; }
  gate stage2_jevgym --rl || return 2
  start_rec stage2_jevgym_rl
  local arm="[{name: jevagent_best_rl, s1: jevagent_rl, profile: configs/harness/jevagent_best.yaml, profile_s2: gemini-3.5-flash, profile_backend: jevagent_rl}]"
  EVAL=1 three s24 "$HV/bin/jevbench" run configs/experiments/stage2_jevgym.yaml --out runs -o workers=4 \
    -o "s1_backends.jevagent_rl=$(rl_spec jevagentrl-4b)" -o "arms=$arm"
  ops runs/stage2_jevgym > "$OUT/s24_ops.txt"
  sync_run stage2_jevgym
}

stage_states_rl() {  # A7.10 (c): rl_ts_replace_f (n = 200), three passes (the RL arm only; the server is up)
  keys || return 1
  gate stage2_rl_states || return 2
  start_rec stage2_rl_states
  EVAL=1 three states_rl "$HV/bin/jevbench" run configs/experiments/stage2_rl_states.yaml --out runs -o workers=4 \
    -o "s1_backends.jevagent_rl=$(rl_spec jevagentrl-4b)" \
    -o 'arms=[{name: rl_ts_replace_f, s1: jevagent_rl, tool_select: {mode: replace}}]'
  ops runs/stage2_rl_states > "$OUT/states_ops.txt"
  sync_run stage2_rl_states
}
stage_states_jev() {  # jev_ts_replace_f, the header's command (workers 8; the placeholder arms are skipped)
  keys || return 1
  gate stage2_rl_states || return 2
  EVAL=1 three states_jev "$HV/bin/jevbench" run configs/experiments/stage2_rl_states.yaml --out runs -o workers=8
  ops runs/stage2_rl_states > "$OUT/states_ops.txt"
  sync_run stage2_rl_states
}
stage_states_intern() {  # intern_ts_replace_f, in-process Intern-Decision-4B (the RL server is stopped: sequential)
  keys || return 1
  gate stage2_rl_states || return 2
  stop_ck
  EVAL=1 three states_intern "$HV/bin/jevbench" run configs/experiments/stage2_rl_states.yaml --out runs -o workers=4 \
    -o 's1_backends.intern={type: intern_decision, model: internlm/Intern-Decision-4B, device: auto, max_length: 8192}'
  ops runs/stage2_rl_states > "$OUT/states_ops.txt"
  sync_run stage2_rl_states
}

e1() {  # e1 BACKEND: the E1 static decisions (configs/decision/bfcl_when2call_rl.yaml), two invocations as E1
  local k
  for k in 1 2; do
    "$HV/bin/python" -m jevharness.bench.decision_eval configs/decision/bfcl_when2call_rl.yaml --out runs --only "$1" \
      > "$OUT/e1_$1_$k.log" 2>&1
    st "e1 $1 invocation $k exit=$?"
  done
  sync_run e1_bfcl_when2call_rl
}
stage_e1_rl() { curl -sf "$RL_URL/health" > /dev/null || { echo "RL server not up"; return 1; }; e1 jevagent_rl; }

stage_corpus() {  # A7.10 (c): the corpus from the four sources (Stage 2's llm_only: its first 200 tasks, from S3)
  mkdir -p runs/stage2_jevgym
  [[ -f runs/stage2_jevgym/llm_only.jsonl ]] || aws s3 cp "$S3OUT/inputs/stage2_jevgym/llm_only.jsonl" \
    runs/stage2_jevgym/llm_only.jsonl --only-show-errors --region "$REGION" || return 1
  sha256sum runs/stage2_jevgym/llm_only.jsonl > "$OUT/llm_only_source.sha256"
  "$HV/bin/python" scripts/stage2_corpus.py build --runs runs --out "$S23" > "$OUT/corpus_build.txt" 2>&1 || return 1
  sync_s23
  mkdir -p "$OUT/gate2"; cp "$S23/corpus.json" "$OUT/gate2/"
}
stage_gate2() {
  local f files=()
  for f in "$S23"/corpus.json "$S23"/*.jsonl.gz; do [[ -f "$f" ]] && files+=("$f"); done
  wait_commit gate2 "${files[@]}"
}

replay() {  # replay NAME SPEC [CACHE]: every distinct question of C, one replay path (scripts/stage2_corpus.py)
  local cache=${3:-cache/s1_stage2_replay.sqlite} k
  for k in 1 2 3; do   # replay resumes; a scoring failure is retried twice inside, then recorded as incorrect
    "$HV/bin/python" scripts/stage2_corpus.py replay --corpus "$S23" --name "$1" --s1 "$2" --cache "$cache" \
      > "$OUT/replay_$1_$k.log" 2>&1
    st "replay $1 invocation $k exit=$?"
  done
  sync_s23
}
stage_replay_jev() { keys || return 1; replay jev '{"type": "jev", "provider": "openrouter"}'; }
stage_replay_intern() {
  stop_ck
  replay intern '{"type": "intern_decision", "model": "internlm/Intern-Decision-4B", "device": "auto", "max_length": 8192}'
}
stage_replay_rl() {
  curl -sf "$RL_URL/health" > /dev/null || { echo "RL server not up"; return 1; }
  replay rl "{\"type\": \"jev\", \"name\": \"jevagentrl-4b\", \"base_url\": \"$RL_URL\", \"model\": \"jevagentrl-4b\", \"api_key\": \"\"}"
}

stage_calib_sft_v1() {  # A7.13 (i) comparability: SFT v1's temperatures on v1 dev with the command and settings that
  # produced final_calibration.json and base_calibration.json (gpu_round5_rl.sh stages freeze / base_control: training/
  # calibrate.py --min-bucket 200 on data/train_v1/dev.jsonl.gz); first a reproduction check that refits the frozen model
  # and compares with results/gpu_round5/final_calibration.json. Sent on its own while the GPU is idle (the gate2 wait).
  local dev=data/train_v1/dev.jsonl.gz base
  base=$(cat "$W/base_model_path.txt")
  mkdir -p data/train_v1 "$OUT/rl_attr"
  [[ -f $dev ]] || aws s3 cp "$S3OUT/inputs/train_v1/dev.jsonl.gz" "$dev" --only-show-errors --region "$REGION" || return 1
  [[ "$(sha256sum < $dev | cut -c1-64)" == 9301ae530b83f953aafeffe2364e8998e7a6057f5b0cc0410afa890664e30858 ]] \
    || { echo "v1 dev sha256 differs from data/train_v1/MANIFEST.json"; return 1; }
  "$HV/bin/python" -m jevharness.training.calibrate --model "$base" --adapter "$CK/jevagentrl-4b" --dev "$dev" \
    --out "$OUT/rl_attr/frozen_refit_calibration.json" --min-bucket 200 > "$OUT/rl_attr/frozen_refit.log" 2>&1 || return 1
  "$HV/bin/python" - "$OUT/rl_attr/frozen_refit_calibration.json" results/gpu_round5/final_calibration.json \
    "$OUT/rl_attr/reproduction_check.json" <<'EOF2' || return 1
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
fa = {"global": a["global"], **{f"b:{k}": v for k, v in a.get("buckets", {}).items()}}
fb = {"global": b["global"], **{f"b:{k}": v for k, v in b.get("buckets", {}).items()}}
rel = {k: abs(fa[k] - fb[k]) / abs(fb[k]) for k in fb if k in fa}
ok = set(fa) == set(fb) and max(rel.values()) <= 1e-3
json.dump({"refit": fa, "recorded": fb, "max_rel_diff": max(rel.values()) if rel else None, "tolerance": 1e-3, "ok": ok},
          open(sys.argv[3], "w"), indent=1)
print("reproduction", "ok" if ok else "FAILED", max(rel.values()) if rel else None)
sys.exit(0 if ok else 1)
EOF2
  "$HV/bin/python" -m jevharness.training.calibrate --model "$base" --adapter "$CK/sft_v1" --dev "$dev" \
    --out "$OUT/rl_attr/sft_v1_calibration.json" --min-bucket 200 > "$OUT/rl_attr/sft_v1_fit.log" 2>&1 || return 1
  sha256sum "$OUT/rl_attr/sft_v1_calibration.json" > "$OUT/rl_attr/sft_v1_calibration.sha256"
}

stage_attribution() {  # A7.13 (i), (ii): each trained stage row of the freeze record, served in turn (n = 100 live)
  keys || return 1
  local row be name
  for row in $(plan "' '.join(r for r in d['attribution_rows'])"); do
    be=$(plan "d['rows']['$row']['backend']"); name=$(plan "d['rows']['$row']['served_name']")
    serve_ck "$row" || { st "attribution $row: server failed"; continue; }
    start_rec "stage2_rl_attribution_$row"
    EVAL=1 three "attribution_$row" "$HV/bin/jevbench" run configs/experiments/stage2_rl_attribution.yaml --out runs \
      -o workers=4 -o "s1_backends.$be=$(rl_spec "$name")"
    e1 "$be"
    replay "$row" "{\"type\": \"jev\", \"name\": \"$name\", \"base_url\": \"$RL_URL\", \"model\": \"$name\", \"api_key\": \"\"}"
    stop_ck
  done
  ops runs/stage2_rl_attribution > "$OUT/attribution_ops.txt"
  sync_run stage2_rl_attribution
}

sl_serve() {  # sl_serve SIZE: the vendor's server for one size (its own manifest file; one size on the GPU)
  stop_ck
  STOP=1 bash scripts/aws/serve_startlux.sh > /dev/null 2>&1; sudo docker ps -q | xargs -r sudo docker stop > /dev/null
  local py; py=$(uv python find 3.11 2>/dev/null)
  # PIN_MANIFEST: the 4B and 27B serve the revisions of the committed manifest (A7.18 (d)); the ladder sizes are new
  if [[ "$1" == 27B ]]; then
    SIZES=27B PYTHON="$py" LLAMA_SERVER="$BASE/bin/llama-server" MANIFEST="$OUT/manifest_replay_27B.json" \
      PIN_MANIFEST=results/stage2/startlux_manifest.json bash scripts/aws/serve_startlux.sh > "$OUT/serve_sl_$1.log" 2>&1
  else
    SIZES=$1 PYTHON="$py" MANIFEST="$OUT/manifest_replay_$1.json" PIN_MANIFEST=results/stage2/startlux_manifest.json \
      bash scripts/aws/serve_startlux.sh > "$OUT/serve_sl_$1.log" 2>&1
  fi
  "$HV/bin/python" -c "import json, sys; m = json.load(open('$OUT/manifest_replay_$1.json'));
s = [x for x in m['sizes'] if x['size'] == '$1' and x.get('status') in ('original', 'gguf_q8_0')]
sys.exit(0 if s else 1)"
}
sl_port() { "$HV/bin/python" -c "import sys; sys.path.insert(0, 'scripts'); import startlux_plan as P; print(P.SIZES['$1'][2])"; }
sl_replay() {  # sl_replay SIZE NAME MODEL
  local port; port=$(sl_port "$1")
  replay "$2" "{\"type\": \"jev\", \"name\": \"startlux-${1,,}\", \"base_url\": \"http://127.0.0.1:$port\", \"model\": \"$3\", \"api_key\": \"\", \"timeout\": 120.0}" \
    cache/s1_stage2_startlux_replay.sqlite
}
stage_replay_sl4b() { sl_serve 4B || return 1; sl_replay 4B sl4b StartLux-Decision-4B; STOP=1 bash scripts/aws/serve_startlux.sh; }
stage_ladder() {  # A7.18 (e) 2: the size ladder on C (descriptive; GPU time only)
  local s
  for s in 0.8B 2B 9B; do
    if sl_serve "$s"; then sl_replay "$s" "sl${s,,}" "StartLux-Decision-$s"; else st "ladder $s: server not started"; fi
    STOP=1 bash scripts/aws/serve_startlux.sh
  done
}
stage_replay_sl27b() {
  sl_serve 27B || return 1
  sl_replay 27B sl27b StartLux-Decision-27B-Q8_0
  STOP=1 bash scripts/aws/serve_startlux.sh; sudo docker ps -q | xargs -r sudo docker stop > /dev/null
}

# ---- A7.13 (iv): the τ² external reference agreement (scripts/tau2_reference_replay.py). The label-free questions come
# from S3 (inputs/t2ref/, 28 MB; checked against the committed corpus.json's sha256); the labels never leave the cloud
# session. The three rows run in parallel, one GPU each (Intern-Decision in two shards), each with its own replay cache.
T2=results/stage2/t2ref
sync_t2() { aws s3 sync "$T2" "$S3OUT/t2ref" --exclude 'questions.jsonl.gz' --only-show-errors --region "$REGION"; }
t2_serve() {  # t2_serve ROW PORT GPU: the row's checkpoint behind serve_jev (as serve_ck) on its own port and GPU
  local row=$1 port=$2 gpu=$3 adapter cal name args
  adapter=$(plan "d['rows']['$row'].get('adapter') or ''")
  cal=$(plan "d['rows']['$row'].get('calibration') or ''")
  name=$(plan "d['rows']['$row']['served_name']")
  args=(--model "$(cat "$W/base_model_path.txt")" --name "$name" --port "$port")
  [[ -n "$adapter" ]] && args+=(--adapter "$CK/$adapter")
  if [[ -n "$cal" ]]; then [[ -f "$cal" ]] && args+=(--calibration "$cal") || args+=(--calibration "$CK/$cal"); fi
  # shellcheck disable=SC2207
  args+=($(plan "' '.join(d.get('serve_args') or [])"))
  echo "CUDA_VISIBLE_DEVICES=$gpu serve_jev ${args[*]}" > "$OUT/t2ref_serve_$row.cmd"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$HV/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$OUT/t2ref_serve_$row.out" 2>&1 &
  echo $! > "$W/t2_$row.pid"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$port/health" > "$OUT/t2ref_serve_${row}_health.json"; then
      curl -sf "http://127.0.0.1:$port/v1/models" > "$OUT/t2ref_serve_${row}_models.json"; return 0
    fi
    kill -0 "$(cat "$W/t2_$row.pid")" 2>/dev/null || { echo "serve_jev $row died"; tail -20 "$OUT/t2ref_serve_$row.out"; return 1; }
    sleep 5
  done
  echo "serve_jev $row did not come up"; return 1
}
t2_replay() {  # t2_replay NAME SPEC GPU [SHARD]: resumes; three invocations as replay()
  local tag=$1${4:+_${4/\//of}} k
  for k in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$3 "$HV/bin/python" scripts/tau2_reference_replay.py replay --corpus "$T2" --name "$1" \
      --s1 "$2" --cache "cache/s1_t2ref_replay_$tag.sqlite" ${4:+--shard "$4"} > "$OUT/t2ref_${tag}_$k.log" 2>&1
    st "t2ref $tag invocation $k exit=$?"
  done
}
stage_t2ref() {
  local n g pids=() syncer want
  mkdir -p "$T2"
  aws s3 cp "$S3OUT/inputs/t2ref/questions.jsonl.gz" "$T2/questions.jsonl.gz" --only-show-errors --region "$REGION" || return 1
  want=$("$HV/bin/python" -c "import json; print(json.load(open('$T2/corpus.json'))['questions_sha256'])")
  [[ "$(sha256sum < "$T2/questions.jsonl.gz" | cut -c1-64)" == "$want" ]] || { echo "questions sha256 differs"; return 1; }
  stop_ck
  local is='{"type": "intern_decision", "model": "internlm/Intern-Decision-4B", "device": "auto", "max_length": 8192}'
  local fs='{"type": "jev", "name": "jevagentrl-4b", "base_url": "http://127.0.0.1:8082", "model": "jevagentrl-4b", "api_key": ""}'
  local bs='{"type": "jev", "name": "rl-base", "base_url": "http://127.0.0.1:8083", "model": "rl-base", "api_key": ""}'
  n=$(nvidia-smi -L | wc -l)
  ( while sleep 600; do sync_t2; done ) & syncer=$!
  if (( n >= 4 )); then   # one GPU per process: Intern in two shards, the two served rows on their own GPUs
    t2_serve frozen 8082 2 || return 1
    t2_serve base 8083 3 || return 1
    t2_replay intern "$is" 0 0/2 & pids+=($!)
    t2_replay intern "$is" 1 1/2 & pids+=($!)
    t2_replay frozen "$fs" 2 & pids+=($!)
    t2_replay base "$bs" 3 & pids+=($!)
    wait "${pids[@]}"
  else                    # one GPU: at most two models resident (no memory contention), the base after the frozen row
    t2_serve frozen 8082 0 || return 1
    t2_replay intern "$is" 0 & pids+=($!)
    t2_replay frozen "$fs" 0
    kill "$(cat "$W/t2_frozen.pid")" 2>/dev/null; rm -f "$W/t2_frozen.pid"; sleep 10
    t2_serve base 8083 0 || return 1
    t2_replay base "$bs" 0
    wait "${pids[@]}"
  fi
  kill "$syncer" 2>/dev/null
  for r in frozen base; do [[ -f "$W/t2_$r.pid" ]] && kill "$(cat "$W/t2_$r.pid")" 2>/dev/null; rm -f "$W/t2_$r.pid"; done
  sync_t2
  wc -l "$T2"/answers_*.jsonl > "$OUT/t2ref_answers_wc.txt"
}

stage_final() {  # every record to S3; the idle stop ends the box IDLE_MIN after this
  stop_ck
  nvidia-smi > "$OUT/nvidia_smi_end.txt" 2>&1
  for d in stage2_jevgym stage2_rl_states e1_bfcl_when2call_rl stage2_rl_attribution e2a_jevgym_shadow_cal_rl; do
    [[ -d runs/$d ]] && sync_run "$d"
  done
  sync_s23
}

# ---- A7.19 (f): the JevAgentRL-v2 rows (exploratory; descriptive, outside every Holm table, label "exploratory
# (A7.19)"), as the StartLux-4B rows of A7.18 (e) 2-5 (scripts/aws/stage2_startlux.sh): the frozen v2 checkpoint
# (results/gpu_round6_miles/final_FROZEN.txt) behind serve_jev on :8084 in its own venv (the Qwen3.5-hybrid kernels;
# hvenv and the StartLux vendor venv untouched), then calibration (5000-5149) -> [the session commits the fits and fills
# backends.jevagentrl_v2] -> tariff (dev seeds) -> [the session commits tariff_v2.json] -> the profile row (300000-300499)
# -> the live replace arm (300000-300199) -> the corpus replay of C. Plan: results/stage2/v2_eval_plan.json; records:
# $OUT/v2 (S3 stage2_gpu/v2/). One model on the GPU at a time: the v2 server's pid is serve_ck.pid, so stop_ck and the
# chain's end stop it like the RL server.
V2PLAN=results/stage2/v2_eval_plan.json
V2O=$OUT/v2                                  # box records of the v2 rows (synced with $OUT)
V2V=$W/v2venv                                # the v2 serving venv: hvenv's stack + flash-linear-attention / causal-conv1d
V2CK=$CK/jevagentrl-v2-final                 # the frozen checkpoint (the plan's checkpoint.dir)
V2_PORT=8084
V2_URL=http://127.0.0.1:$V2_PORT
V2_FITS=(results/calibration/tauhat_v2_shadow_cal_tool_select.json results/calibration/tauhat_v2_shadow_cal_tool_params.json)
v2plan() { "$HV/bin/python" -c "import json,sys; d=json.load(open('$V2PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"; }
v2_spec() {  # v2_spec [CACHE]: the configs' jevagentrl_v2 backend, bound (placeholder dropped), optionally its own cache
  printf '{type: jev, name: jevagentrl-v2, base_url: "%s", model: jevagentrl-v2, api_key: "", price_model: jevagentrl-v2, timeout: 60.0%s}' \
    "$V2_URL" "${1:+, cache: $1}"
}
v2_up() { curl -sf "$V2_URL/health" > /dev/null || { echo "v2 server not up on :$V2_PORT (stage v2_serve)"; return 1; }; }
v2_itype() {  # the instance type now (the host may have been resized since stage setup wrote instance_type.txt)
  curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/instance-type
}
v2_start_rec() {  # v2_start_rec STEP: a START file (run_main.sh format) + the v2 plan's sha256 and the instance type now
  mkdir -p "$V2O"
  [[ -f "$V2O/START_$1.txt" ]] || printf 'step %s\nstarted %s\ncommit %s\nplan sha256 %s\nv2 plan sha256 %s\nhost %s\n' \
    "$1" "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" \
    "$(sha256sum "$V2PLAN" | cut -c1-16)" "$(v2_itype)" > "$V2O/START_$1.txt"
}
v2_budget() {  # v2_budget STAGE: A7.19 (g), the cap is a ceiling -- every record of the v2 runs (errored attempts
  # included) + the stage's reserve must stay within the plan's cap_rows_usd, else the stage does not start
  mkdir -p "$V2O"
  "$HV/bin/python" - "$V2PLAN" "$1" "$V2O/spend.txt" <<'EOF'
import glob, json, os, sys
api, stage, out = json.load(open(sys.argv[1]))["api"], sys.argv[2], sys.argv[3]
spent, parts = 0.0, []
for d in api["runs"]:
    s = 0.0
    for f in sorted(glob.glob(os.path.join("runs", d, "*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and "task_id" in r:
                s += float(((r.get("usage") or {}).get("total") or {}).get("cost_usd") or 0)
    spent += s
    parts.append(f"{d} ${s:.2f}")
cap, reserve = float(api["cap_rows_usd"]), float(api["reserve_usd"].get(stage, 0.0))
ok = spent + reserve <= cap
line = (f"{stage}: spent ${spent:.2f} + reserve ${reserve:.2f} vs cap ${cap:.2f} -> "
        f"{'go' if ok else 'STOP (A7.19 (g))'} [{'; '.join(parts)}]")
open(out, "a").write(line + "\n")
print(line)
sys.exit(0 if ok else 1)
EOF
}
v2_fitted() {  # the profile's backends.jevagentrl_v2 is fitted from v2_shadow_cal with the committed fits (stage2_spec.py)
  "$HV/bin/python" -c "import sys; sys.path.insert(0, 'scripts'); import stage2_spec as S; ok, m = S._v2_fitted(); print(m); sys.exit(0 if ok else 1)"
}
v2_skip() {  # why the fitted profile runs no v2 arm (a block whose fit delegates nothing), else empty
  "$HV/bin/python" -c "import sys; sys.path.insert(0, 'src'); from jevharness.harness import profile as HP
print(HP.decision_points(HP.load_profile('configs/harness/jevagent_best.yaml'), 'gemini-3.5-flash', backend='jevagentrl_v2')[2] or '')"
}

stage_v2_setup() {  # the v2 serving venv (plan serving_env): hvenv's stack (its freeze as constraints, torch from the cu130
  # index, the plan's pins) + flash-linear-attention / causal-conv1d at the StartLux vendor venv's versions (its own
  # cp311 build of causal-conv1d from the pip wheel cache, else built as serve_startlux.sh builds it). Fails unless the
  # serving code is the freeze commit's and transformers' Qwen3.5 module binds the kernels and they run on the GPU.
  mkdir -p "$V2O"
  v2_itype > "$V2O/instance_type.txt"; echo >> "$V2O/instance_type.txt"
  nvidia-smi > "$V2O/nvidia_smi_setup.txt" 2>&1
  [[ -x "$HV/bin/python" ]] || { echo "no harness venv $HV (stage setup)"; return 1; }
  command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; } || return 1
  local fc code
  fc=$(v2plan "d['serving_commit_of_the_freeze']")
  code=$(v2plan "' '.join(d['serving_code'])")
  # shellcheck disable=SC2086
  if ! git diff --quiet "$fc" HEAD -- $code; then
    # shellcheck disable=SC2086
    git diff --stat "$fc" HEAD -- $code; echo "the serving code differs from the freeze commit $fc"; return 1
  fi
  echo "serving code ($code) identical to the freeze commit $fc at $(git rev-parse HEAD)" > "$V2O/serving_code.txt"
  [[ -x "$V2V/bin/python" ]] || uv venv -q -p 3.11 "$V2V" || return 1
  uv pip freeze -p "$HV/bin/python" | grep -v -E '^(-e |jevharness|torch==|triton==|nvidia-)' > "$V2O/v2venv_constraints.txt"
  local C=(-c "$V2O/v2venv_constraints.txt")
  uv pip install -q -p "$V2V/bin/python" "$(v2plan "d['serving_env']['torch']")" \
    --index-url "$(v2plan "d['serving_env']['torch_index']")" || return 1
  # shellcheck disable=SC2046
  uv pip install -q -p "$V2V/bin/python" "${C[@]}" -e ".[serve]" $(v2plan "' '.join(d['serving_env']['pins'])") \
    safetensors einops || return 1
  uv pip install -q -p "$V2V/bin/python" "${C[@]}" "$(v2plan "d['serving_env']['kernels'][0]")" || return 1
  if ! "$V2V/bin/python" -c "import torch, causal_conv1d_cuda" > /dev/null 2>&1; then
    local cc whl tmaj c
    cc=$(v2plan "d['serving_env']['kernels'][1]")
    whl=$(find "$HOME/.cache/pip/wheels" -name "causal_conv1d-${cc#*==}-cp311-cp311-linux_x86_64.whl" 2>/dev/null | head -1)
    if [[ -n "$whl" ]]; then
      sha256sum "$whl" > "$V2O/causal_conv1d_wheel.txt"
      uv pip install -q -p "$V2V/bin/python" "${C[@]}" "$whl"
    fi
    if ! "$V2V/bin/python" -c "import torch, causal_conv1d_cuda" > /dev/null 2>&1; then   # build (serve_startlux.sh's way)
      uv pip install -q -p "$V2V/bin/python" "${C[@]}" setuptools wheel packaging ninja || return 1
      if [[ -z "${CUDA_HOME:-}" ]] && ! command -v nvcc > /dev/null; then
        tmaj=$("$V2V/bin/python" -c "import torch; print((torch.version.cuda or '').split('.')[0])" 2>/dev/null)
        for c in /usr/local/cuda /usr/local/cuda-"$tmaj"* /opt/*/lib/python3*/site-packages/nvidia/cu"$tmaj" \
                 "$V2V"/lib/python3*/site-packages/nvidia/cu"$tmaj"; do
          if [[ -n "$tmaj" && -x "$c/bin/nvcc" ]] && "$c/bin/nvcc" --version | grep -q "release $tmaj\."; then
            export CUDA_HOME="$c"; break
          fi
        done
      fi
      [[ -n "${CUDA_HOME:-}" ]] && export PATH="$CUDA_HOME/bin:$PATH"
      echo "causal-conv1d built with CUDA_HOME=${CUDA_HOME:-none}" > "$V2O/causal_conv1d_build.txt"
      MAX_JOBS=${MAX_JOBS:-$(nproc)} uv pip install -q -p "$V2V/bin/python" "${C[@]}" --no-build-isolation \
        --reinstall-package causal-conv1d "$cc" >> "$V2O/causal_conv1d_build.txt" 2>&1 || return 1
    fi
  fi
  uv pip freeze -p "$V2V/bin/python" > "$V2O/v2venv_pip_freeze.txt"
  "$V2V/bin/python" - > "$V2O/kernels.json" <<'EOF'
import inspect, json, sys
from jevharness.training.miles_port.smoke import env_report
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
rep = env_report()      # versions + kernel_check(): causal_conv1d_fn and fla's chunk_gated_delta_rule run on the GPU
bound = {}
for f in ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
    fn = getattr(mq, f, None)
    nl = inspect.getclosurevars(fn).nonlocals if callable(fn) else {}
    bound[f] = {"kernel": bool(nl.get("is_new_implementation")),
                "module": getattr(nl.get("implementation"), "__module__", None)}
rep["qwen3_5_bound"] = bound
k = rep.get("kernels") or {}
rep["fast_kernels_active"] = (all(v["kernel"] for v in bound.values())
                              and all(str(k.get(x, "")).startswith("ok") for x in ("causal_conv1d", "fla_gated_delta_rule")))
print(json.dumps(rep, indent=1, default=str))
sys.exit(0 if rep["fast_kernels_active"] else 1)
EOF
  local rc=$?
  "$HV/bin/python" -c "import json; d = json.load(open('$V2O/kernels.json')); print('fast kernels active:', d['fast_kernels_active'], {k: v['kernel'] for k, v in d['qwen3_5_bound'].items()}, d.get('torch'), d.get('transformers'))"
  return $rc
}

stage_v2_fetch() {  # the frozen checkpoint (the plan's S3 directory) -> $V2CK; every sha256 line of the freeze record
  # (results/gpu_round6_miles/final_FROZEN.txt) checked and the downloaded FROZEN.txt byte-identical to it
  mkdir -p "$V2CK" "$V2O"
  aws s3 sync "$(v2plan "d['checkpoint']['s3']")" "$V2CK" --only-show-errors --region "$REGION" || return 1
  "$HV/bin/python" - "$(v2plan "d['freeze_record']")" "$V2CK" "$V2O/checkpoint_verify.json" <<'EOF' || return 1
import hashlib, json, os, sys
rec, ck, out = sys.argv[1:4]
want = {}
for line in open(rec):
    k, _, v = line.strip().partition(" ")
    if len(k) == 64 and all(c in "0123456789abcdef" for c in k):
        v = v.strip()
        want[v[2:] if v.startswith("./") else v] = k


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


files, bad = {}, []
for rel, s in sorted(want.items()):
    p = os.path.join(ck, rel)
    got = sha(p) if os.path.exists(p) else None
    files[rel] = {"expected": s, "got": got, "ok": got == s}
    if got != s:
        bad.append(rel)
fz = os.path.join(ck, "FROZEN.txt")
same = os.path.exists(fz) and open(fz, "rb").read() == open(rec, "rb").read()
need = {"model.safetensors", "config.json", "calibration.json", "tokenizer.json", "regression_image.json"}
unlisted = {f: sha(os.path.join(ck, f)) for f in sorted(os.listdir(ck))
            if f not in want and f != "FROZEN.txt" and os.path.isfile(os.path.join(ck, f))}
ok = need <= set(want) and not bad and same
json.dump({"ok": ok, "freeze_record": rec, "frozen_txt_identical": same, "mismatch": bad, "files": files,
           "unlisted_sha256": unlisted}, open(out, "w"), indent=1)
print("v2 checkpoint sha256:", "all match" if ok else f"MISMATCH {bad} (FROZEN.txt identical: {same})")
sys.exit(0 if ok else 1)
EOF
}

stage_v2_serve() {  # serve_jev in the v2 venv on :8084 with the freeze record's temperature file, after the regression
  # check: the host must reproduce the freeze-time option probabilities (Miles image) within the freeze's tolerance
  [[ -x "$V2V/bin/python" ]] || { echo "no v2 venv (stage v2_setup)"; return 1; }
  "$HV/bin/python" -c "import json, sys; sys.exit(0 if json.load(open('$V2O/checkpoint_verify.json'))['ok'] else 1)" \
    2>/dev/null || { echo "checkpoint not verified (stage v2_fetch)"; return 1; }
  stop_ck
  local cal dev tol
  cal="$V2CK/$(v2plan "d['checkpoint']['calibration']")"
  dev=$(v2plan "d['regression']['dev']")
  tol=$(v2plan "d['regression']['tol_max_abs_dprob']")
  [[ "$(sha256sum < "$dev" | cut -c1-64)" == "$(v2plan "d['regression']['dev_sha256']")" ]] \
    || { echo "regression corpus $dev: sha256 differs from the plan"; return 1; }
  "$V2V/bin/python" -m jevharness.training.miles_port.smoke regress --model "$V2CK" --calibration "$cal" --dev "$dev" \
    --n "$(v2plan "d['regression']['n']")" --out "$V2O/regression_host.json" > "$V2O/regression.log" 2>&1 \
    || { tail -20 "$V2O/regression.log"; return 1; }
  "$V2V/bin/python" -m jevharness.training.miles_port.smoke compare "$V2CK/regression_venv.json" \
    "$V2O/regression_host.json" --tol "$tol" --out "$V2O/regression_compare_venv.json" > /dev/null   # recorded beside
  "$V2V/bin/python" -m jevharness.training.miles_port.smoke compare "$V2CK/regression_image.json" \
    "$V2O/regression_host.json" --tol "$tol" --out "$V2O/regression_compare_image.json" > /dev/null \
    || { echo "the host does not reproduce the freeze-time probabilities: $V2O/regression_compare_image.json"; return 1; }
  local args=(--model "$V2CK" --calibration "$cal" --name "$(v2plan "d['checkpoint']['served_name']")" --port "$V2_PORT")
  # shellcheck disable=SC2207
  args+=($(v2plan "' '.join(d.get('serve_args') or [])"))
  echo "serve_jev ${args[*]} (venv $V2V)" > "$V2O/serve.cmd"
  nohup "$V2V/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$V2O/serve.out" 2>&1 &
  echo $! > "$W/serve_ck.pid"
  for _ in $(seq 1 180); do
    if curl -sf "$V2_URL/health" > "$V2O/serve_health.json"; then
      curl -sf "$V2_URL/v1/models" > "$V2O/serve_models.json"; return 0
    fi
    kill -0 "$(cat "$W/serve_ck.pid")" 2>/dev/null || { echo "serve_jev (v2) died"; tail -20 "$V2O/serve.out"; return 1; }
    sleep 5
  done
  echo "serve_jev (v2) did not come up"; return 1
}
stage_v2_stop() { stop_ck; nvidia-smi --query-gpu=memory.used --format=csv > "$V2O/gpu_after_stop.txt" 2>&1; }

stage_v2_cal() {  # A7.19 (f) / A7.18 (e) 4: v2_shadow_cal on the calibration seeds 5000-5149 (three passes; not an
  # evaluation run), then the registered fit per decision point -> results/calibration/ (the box's copy; v2_gate_tau
  # waits for the session to commit these files byte-identical and to fill backends.jevagentrl_v2 from them)
  keys || return 1; v2_up || return 1
  gate jevgym_shadow_cal_v2 || return 2
  v2_budget cal || return 1
  v2_start_rec jevgym_shadow_cal_v2
  three v2_cal "$HV/bin/jevbench" run configs/experiments/jevgym_shadow_cal_v2.yaml --out runs -o workers=4
  "$HV/bin/jevbench" report runs/e2a_jevgym_shadow_cal_v2 > /dev/null 2>&1
  ops runs/e2a_jevgym_shadow_cal_v2 > "$V2O/cal_ops.txt"
  sync_run e2a_jevgym_shadow_cal_v2
  mkdir -p results/calibration "$V2O/cal"
  local f dp
  for f in "${V2_FITS[@]}"; do
    dp=${f%.json}; dp=${dp#results/calibration/tauhat_v2_shadow_cal_}
    "$HV/bin/python" -m jevharness.bench.threshold_theory runs/e2a_jevgym_shadow_cal_v2 --arm v2_shadow_cal --dp "$dp" \
      --calibration-arm --task-value 1.0 --lam-per-hour 0 --v-fallback W --format json --out "$f" \
      > "$V2O/cal/fit_$dp.log" 2>&1 || return 1
    cp "$f" "$V2O/cal/"
  done
}

stage_v2_gate_tau() {  # the v2 fits committed byte-identical AND backends.jevagentrl_v2 filled from them (stage2_spec.py
  # _v2_fitted) before any v2 evaluation episode; a timeout stops the chain (no thresholds to go on with)
  local t_end=$(( $(date +%s) + GATE_MIN * 60 )) f ok
  st "v2_gate_tau: waiting up to $GATE_MIN min for the commit of ${V2_FITS[*]} and the filled backends.jevagentrl_v2"
  while (( $(date +%s) < t_end )); do
    if fetch_bundle; then
      ok=1
      for f in "${V2_FITS[@]}"; do
        [[ -f "$V2O/cal/$(basename "$f")" ]] || { echo "no box fit $f (stage v2_cal)"; return 1; }
        [[ "$(git show "FETCH_HEAD:$f" 2>/dev/null | sha256sum | cut -c1-64)" == \
           "$(sha256sum < "$V2O/cal/$(basename "$f")" | cut -c1-64)" ]] || ok=0
      done
      if [[ $ok == 1 ]]; then
        git reset -q --hard FETCH_HEAD
        if v2_fitted > "$V2O/tau_check.txt" 2>&1; then
          st "v2_gate_tau: committed at $(git log --oneline -1 | cut -c1-70): $(head -1 "$V2O/tau_check.txt")"
          return 0
        fi
      fi
    fi
    sleep 120
  done
  st "v2_gate_tau: TIMEOUT after $GATE_MIN min: the v2 chain stops here (no committed v2 thresholds)"
  return 1
}

stage_v2_tariff() {  # A7.19 (f) / A7.13 (iii): jevagent_best_v2's configuration (stage2_v2.yaml) on the committed plan's
  # development seeds (results/stage2/tariff_v2_plan.json), the window recorded, the host priced by its type now
  keys || return 1; v2_up || return 1
  v2_fitted > /dev/null || { echo "backends.jevagentrl_v2 not fitted (stage v2_gate_tau)"; return 1; }
  local why; why=$(v2_skip)
  if [[ -n "$why" ]]; then echo "no v2 tariff: $why" | tee "$V2O/tariff_skip.txt"; return 0; fi
  v2_budget tariff || return 1
  local p=results/stage2/tariff_v2_plan.json itype wseed wn seed n name usd ref
  itype=$(v2_itype)
  read -r wseed wn seed n name usd ref < <("$HV/bin/python" -c "import json, sys; p = json.load(open('$p')); h = p['host']
u = h['usd_per_hour_by_type'].get(sys.argv[1])
print(p['warmup']['seed0'], p['warmup']['n'], p['workload']['seed0'], p['workload']['n'], p['workload']['run_name'],
      'none' if u is None else u, h['usd_per_hour_by_type'][h['reference_type']])" "$itype")
  [[ -n "$usd" && "$usd" != none ]] || { echo "instance type '$itype' is not priced in $p: no tariff"; return 1; }
  local common=(-o workers=4 -o "s1_backends.jevagentrl_v2=$(v2_spec cache/s1_dev_tariff_v2.sqlite)")
  # warm-up (not in the window): the same configuration on its own dev seeds
  "$HV/bin/jevbench" run configs/experiments/stage2_v2.yaml --out runs -o "name=${name}_warmup" -o suite.seed0="$wseed" \
    -o suite.n="$wn" -o n_tasks="$wn" "${common[@]}" > "$V2O/tariff_warmup.log" 2>&1
  local t0 t1 rc
  t0=$(date -u +%FT%TZ)
  "$HV/bin/jevbench" run configs/experiments/stage2_v2.yaml --out runs -o "name=$name" -o suite.seed0="$seed" \
    -o suite.n="$n" -o n_tasks="$n" "${common[@]}" > "$V2O/tariff_workload.log" 2>&1
  rc=$?
  t1=$(date -u +%FT%TZ)
  printf '{"window_start": "%s", "window_end": "%s", "exit": %s, "run": "runs/%s", "instance_type": "%s", "usd_per_hour": %s}\n' \
    "$t0" "$t1" "$rc" "$name" "$itype" "$usd" > "$V2O/tariff_window.json"
  ops "runs/$name" > "$V2O/tariff_ops.txt"
  sync_run "$name"; sync_run "${name}_warmup"
  [[ $rc -eq 0 ]] || return $rc
  local sens   # the S2-4 tariff's sensitivity rows (stage tariff_rl), + the reference host's price on another type
  sens=$("$HV/bin/python" - "runs/$name/jevagent_best_v2.jsonl" "$usd" "$ref" "$t0" "$t1" <<'EOF'
import datetime as dt, json, sys
path, usd, ref, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
iv, tok, n, latest = [], 0, 0, {}
for line in open(path):
    r = json.loads(line); n += 1
    latest[r["task_id"]] = r
    for c in r.get("calls") or []:
        if c.get("component") == "s1" and not c.get("cached"):
            end = dt.datetime.fromisoformat(c["ts"]).timestamp(); iv.append((end - c["latency_ms"] / 1000, end))
            tok += int(c.get("input_tokens") or 0)
iv.sort(); busy = 0.0
if iv:
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce: busy += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    busy += ce - cs
out = [f"--sensitivity busy_throughput={usd * busy / 3600 / n}", f"--sensitivity jev_api_price={tok * 0.042 / 1e6 / n}",
       "--sensitivity zero=0.0"]
if abs(usd - ref) > 1e-9:
    h = (dt.datetime.fromisoformat(t1.replace("Z", "+00:00")) - dt.datetime.fromisoformat(t0.replace("Z", "+00:00")))
    done = sum(1 for r in latest.values() if not r.get("error"))
    out.append(f"--sensitivity reference_host={ref * h.total_seconds() / 3600 / max(1, done)}")
print(" ".join(out))
EOF
)
  # shellcheck disable=SC2086
  "$HV/bin/python" scripts/stage2_tariff.py --run "runs/$name" --arm jevagent_best_v2 --served-model jevagentrl-v2 \
    --host-usd-per-hour "$usd" --window-start "$t0" --window-end "$t1" \
    --hardware "1x NVIDIA L40S 48 GB ($itype, us-east-1, on demand)" --precision bf16 \
    --concurrency 4 $sens --workload-matches --out results/stage2/tariff_v2.json > "$V2O/tariff.txt" 2>&1 || return 1
  cp results/stage2/tariff_v2.json "$V2O/tariff_v2.json"
}

stage_v2_gate_tariff() {  # results/stage2/tariff_v2.json committed byte-identical before any priced v2 episode
  # (A7.13 (iii)); without the commit the v2 profile row does not start (the chain stops)
  if [[ -f "$V2O/tariff_skip.txt" ]]; then cat "$V2O/tariff_skip.txt"; return 0; fi
  [[ -f "$V2O/tariff_v2.json" ]] || { echo "no box tariff (stage v2_tariff)"; return 1; }
  wait_commit v2_gate_tariff results/stage2/tariff_v2.json
  git cat-file -e HEAD:results/stage2/tariff_v2.json 2>/dev/null && [[ "$(git show HEAD:results/stage2/tariff_v2.json \
    | sha256sum | cut -c1-64)" == "$(sha256sum < "$V2O/tariff_v2.json" | cut -c1-64)" ]] \
    || { echo "results/stage2/tariff_v2.json is not committed: the v2 profile row does not start"; return 1; }
}

stage_v2_profile() {  # A7.19 (f), as A7.18 (e) 5 without a test: jevagent_best_v2 (stage2_v2.yaml; read beside
  # stage2_jevgym's llm_only) on 300000-300499, three passes, after the launch gate (thresholds fitted, tariff frozen)
  keys || return 1; v2_up || return 1
  local why; why=$(v2_skip)
  if [[ -n "$why" ]]; then echo "v2 profile row not run: $why" | tee "$V2O/profile_skip.txt"; return 0; fi
  gate stage2_v2 || return 2
  v2_budget profile || return 1
  v2_start_rec stage2_v2
  EVAL=1 three v2_profile "$HV/bin/jevbench" run configs/experiments/stage2_v2.yaml --out runs -o workers=4 \
    -o "s1_backends.jevagentrl_v2=$(v2_spec cache/s1_stage2_v2.sqlite)"
  ops runs/stage2_v2 > "$V2O/profile_ops.txt"
  sync_run stage2_v2
}

stage_v2_states() {  # A7.19 (f), as A7.18 (e) 3: v2_ts_replace_f (stage2_v2_states.yaml, n = 200 on 300000-300199),
  # three passes
  keys || return 1; v2_up || return 1
  gate stage2_v2_states || return 2
  v2_budget states || return 1
  v2_start_rec stage2_v2_states
  EVAL=1 three v2_states "$HV/bin/jevbench" run configs/experiments/stage2_v2_states.yaml --out runs -o workers=4 \
    -o "s1_backends.jevagentrl_v2=$(v2_spec)"
  ops runs/stage2_v2_states > "$V2O/states_ops.txt"
  sync_run stage2_v2_states
}

stage_v2_replay() {  # A7.19 (f), as A7.18 (e) 2: every distinct question of the frozen corpus C answered by v2 through
  # the one replay path (scripts/stage2_corpus.py replay --name v2), its own replay cache; C's files checked first
  v2_up || return 1
  "$HV/bin/python" - "$S23" <<'EOF' || return 1
import hashlib, json, os, sys
d = sys.argv[1]
files = json.load(open(os.path.join(d, "corpus.json"))).get("files") or {}
bad = [n for n, h in files.items() if not os.path.exists(os.path.join(d, n))
       or hashlib.sha256(open(os.path.join(d, n), "rb").read()).hexdigest() != h]
print("corpus C:", "as frozen" if files and not bad else f"CHANGED {bad}")
sys.exit(0 if files and not bad else 1)
EOF
  replay v2 "{\"type\": \"jev\", \"name\": \"jevagentrl-v2\", \"base_url\": \"$V2_URL\", \"model\": \"jevagentrl-v2\", \"api_key\": \"\", \"price_model\": \"jevagentrl-v2\", \"timeout\": 60.0}" \
    "$(v2plan "d['rows']['corpus_replay']['cache']")"
}

stage_v2_final() {  # every v2 record to S3 and the v2 server stopped (send "... v2_final final": the chain-end stop)
  stop_ck
  mkdir -p "$V2O"
  nvidia-smi > "$V2O/nvidia_smi_end.txt" 2>&1
  local d
  for d in e2a_jevgym_shadow_cal_v2 stage2_dev_tariff_v2_warmup stage2_dev_tariff_v2 stage2_v2 stage2_v2_states; do
    [[ -d runs/$d ]] && sync_run "$d"
  done
  sync_s23
}

# a failed stage stops the chain while later stages depend on it; after gate2 every stage is independent (each replay,
# row and size serves its own model), and e1_rl feeds nothing downstream, so those go on after a failure
after_gate2=0
for s in $STAGES; do
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $s"; continue; fi
  st "stage $s start ($(git log --oneline -1 | cut -c1-60))"
  t0=$(date +%s)
  ( "stage_$s" ) > "$OUT/$s.log" 2>&1
  rc=$?
  st "stage $s exit=$rc ($(( ($(date +%s) - t0) / 60 )) min)"
  [[ $s == gate2 && $rc -eq 0 ]] && after_gate2=1
  if [[ $rc -ne 0 && "${KEEP_GOING:-0}" != 1 && $after_gate2 == 0 && $s != e1_rl ]]; then
    st "stopping the chain after $s failed"
    broke=1
    break
  fi
done
# servers are stopped when the chain ends (`final`) or breaks, so the idle stop can end the box; a partial queue
# (e.g. "setup fetch serve_rl") leaves the RL server up for the next send
if [[ "${broke:-0}" == 1 || " $STAGES " == *" final "* ]]; then
  stop_ck; STOP=1 bash scripts/aws/serve_startlux.sh > /dev/null 2>&1; sudo docker ps -q | xargs -r sudo docker stop > /dev/null
  st "servers stopped; the box stops itself after $IDLE_MIN idle minutes"
fi
st "box done [$STAGES]"
