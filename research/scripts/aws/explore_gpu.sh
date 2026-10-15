#!/usr/bin/env bash
# Exploratory rows E1 and E2 (docs/PREREG_A7_STAGE2.md, Deviations entries of 2026-10-05; descriptive, outside both
# registered stages, no test) on the persistent Stage-2 GPU host <instance-id> (tag Name=jev-stage2-startlux,
# one L40S 48 GB; g6e.2xlarge, or a larger single-GPU g6e type when that has no capacity). One model on the GPU at a time.
#   E1  live tau^2 telecom backend substitution: `llm_only` vs `jevagent_best_rl` (configs/tau2/explore/
#       jevagent_best_rl.yaml; System-1 = the frozen JevAgentRL-4B behind serve_jev on :8081) on the 150 never-run ids of
#       configs/tau2/explore/telecom_rl_task_ids.json, one attempt each, seed 300, A6.15 / A7.15 (e) settings; two chunks
#       of 75 tasks (both arms per chunk), the API cap ($75) checked before the second chunk; first the day-1 go/no-go
#       on the 5 validation tasks (RL arm only; stage e1_val), decided by this session before e1_main is sent.
#       Round 1 failed (no_crash_or_loop: S1's respond_to_user answered by a call to an undeclared tool); after the
#       adapter fix (tau2 `respond_fallback: defer`, Deviations 2026-10-05) round 2 runs on validation tasks 6-10
#       (stage e1_val2) and is decided the same way (`explore_telecom_rl.py gonogo --round 2`).
#   E2  the StartLux size ladder live: sl{0.8b,2b,9b}_ts_replace_f (configs/experiments/explore_startlux_live.yaml), each
#       size served by the vendor's server at the ladder replay's weights revision, three passes, API cap $15.
#
#   LOCAL:  bash scripts/aws/explore_gpu.sh push | send "STAGES" | status [N]
#   BOX:    rearm t2setup serve_rl e1_val      (then, after the local go/no-go:)  e1_main stop_rl e2 final
#           (round 2 of the go/no-go: rearm e1_val2, then the same e1_main stop_rl e2 final; on NO-GO: stop_rl e2 final)
# Code: the committed branch (git bundle code/explore.bundle) + an overlay of this session's uncommitted files
# (code/explore_overlay.tgz; sha256 in $OUT/overlay.sha256) + tau2-bench at the pinned commit 5bfa7e3 (git archive,
# code/tau2-bench-5bfa7e3.tar.gz) in its own venv ($W/t2venv; the harness venv and the StartLux venv are untouched).
# Records: results/explore/gpu on the box -> s3://$BUCKET/explore_gpu/; run directories -> s3://$BUCKET/explore_gpu/runs/.
# While a run is in flight only operational fields are read (errors, cost per episode / simulation, served model).
set -uo pipefail
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
IID=${IID:?set IID to the instance id}
BUNDLE=explore.bundle
OVERLAY=explore_overlay.tgz
T2TAR=tau2-bench-5bfa7e3.tar.gz
S3OUT="$B/explore_gpu"
IDLE_MIN=${IDLE_MIN:-60}
GUARD_MIN=${GUARD_MIN:-720}
OVERLAY_FILES=(scripts/aws/explore_gpu.sh scripts/explore_common.py scripts/explore_telecom_rl.py
  src/jevharness/bench/suites/tau2.py
  configs/tau2/explore/telecom_rl_task_ids.json configs/tau2/explore/jevagent_best_rl.yaml
  configs/tau2/explore/t2venv_constraints.txt configs/experiments/explore_startlux_live.yaml configs/prices/models.yaml
  configs/decision/explore_bfcl_when2call_startlux_ladder.yaml
  docs/PREREG_A7_STAGE2.md)

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws --region "$REGION" "$@"; }

if [[ "$MODE" == push ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors
  tar -C "$ROOT" -czf "$tmp/$OVERLAY" "${OVERLAY_FILES[@]}"
  awsl s3 cp "$tmp/$OVERLAY" "$B/code/$OVERLAY" --only-show-errors
  cp "$ROOT/scripts/aws/explore_gpu.sh" "$tmp/explore_gpu.sh"
  awsl s3 cp "$tmp/explore_gpu.sh" "$B/code/explore_gpu.sh" --only-show-errors
  if ! awsl s3 ls "$B/code/$T2TAR" > /dev/null 2>&1; then
    T2=${T2SRC:?set T2SRC to the tau2-bench clone (commit 5bfa7e3)}
    git -C "$T2" archive --format=tar.gz -o "$tmp/$T2TAR" 5bfa7e3
    awsl s3 cp "$tmp/$T2TAR" "$B/code/$T2TAR" --only-show-errors
    sha256sum "$tmp/$T2TAR"
  fi
  echo "pushed $(git -C "$ROOT" log --oneline -1 "$BRANCH"); overlay $(sha256sum "$tmp/$OVERLAY" | cut -c1-16)"
  rm -rf "$tmp"
  exit 0
fi
if [[ "$MODE" == send ]]; then
  STAGES=${1:?usage: send \"STAGES\"}
  tmp=$(mktemp)
  {
    echo '#!/bin/bash'
    echo 'f=/opt/jev/ex_$(date +%s%N).sh'
    echo "aws s3 cp $B/code/explore_gpu.sh \$f --only-show-errors --region $REGION"
    printf 'export IDLE_MIN=%q GUARD_MIN=%q\n' "$IDLE_MIN" "$GUARD_MIN"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/ex_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/ex_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$IID" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/ex_driver.sh /tmp/ex_driver.sh --only-show-errors && bash /tmp/ex_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi
if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi
if [[ "$MODE" != box ]]; then
  sed -n '2,20p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
W=/opt/jev
cd "$W" || exit 1
myb="$W/explore.$$.bundle"
aws s3 cp "$B/code/$BUNDLE" "$myb" --only-show-errors --region "$REGION" || exit 1
[[ -d "$W/stage2" ]] || git clone -q -b "$BRANCH" "$myb" "$W/stage2"
cd "$W/stage2" || exit 1
git fetch -q "$myb" "$BRANCH" && git reset -q --hard FETCH_HEAD
rm -f "$myb"
aws s3 cp "$B/code/$OVERLAY" "$W/$OVERLAY" --only-show-errors --region "$REGION" && tar -xzf "$W/$OVERLAY" -C "$W/stage2"
OUT=results/explore/gpu
mkdir -p "$OUT" runs cache
sha256sum "$W/$OVERLAY" > "$OUT/overlay.sha256"
HV=$W/hvenv                         # the harness venv: serve_jev, jevbench (E2)
T2V=$W/t2venv                       # tau2-bench + the harness (E1)
T2SRC=$W/tau2-bench-5bfa7e3
CK=$W/ck
PLAN=results/stage2/rl_eval_plan.json
RL_URL=http://127.0.0.1:8081
IDS=configs/tau2/explore/telecom_rl_task_ids.json
RLCFG=configs/tau2/explore/jevagent_best_rl.yaml
E1=runs/explore_telecom_rl
E1V=runs/explore_telecom_rl_validation
E1V2=runs/explore_telecom_rl_validation2
E2RUN=explore_startlux_live
export BASE=/opt/startlux PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
GEMINI_ARGS='{"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}'
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$OUT/status.txt"
  aws s3 sync "$OUT" "$S3OUT" --only-show-errors --region "$REGION"
}
sync_run() { [[ -d "$1" ]] && aws s3 sync "$1" "$S3OUT/$1" --only-show-errors --region "$REGION"; }
plan() { "$HV/bin/python" -c "import json,sys; d=json.load(open('$PLAN')); print(eval(sys.argv[1], {'d': d}))" "$1"; }
itype() {
  curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/instance-type
}
keys() {  # the System-2 keys from SSM (never on disk, never printed); litellm reads GEMINI_API_KEY
  local v val
  for v in GEMINI_API_KEY GEMINI_API_KEYS OPENROUTER_API_KEY; do
    val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value \
          --output text 2>/dev/null) || val=""
    if [[ -n "$val" && "$val" != None ]]; then export "$v=$val"; fi
  done
  unset val
  if [[ -z "${GEMINI_API_KEY:-}" && -n "${GEMINI_API_KEYS:-}" ]]; then
    GEMINI_API_KEY=$(printf '%s' "$GEMINI_API_KEYS" | tr ', ' '\n\n' | grep -v '^$' | head -1); export GEMINI_API_KEY
  fi
  [[ -n "${GEMINI_API_KEY:-}" ]] || { echo "keys missing in SSM"; return 1; }
}
arm_stops() {  # total guard + idle stop (cron, IDLE_MIN), as scripts/aws/stage2_gpu.sh
  sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "jev-explore-gpu: total guard" 2>/dev/null
  sudo tee /usr/local/bin/jev-idle-stop > /dev/null <<IDLE
#!/bin/bash
# stop the box after $IDLE_MIN min without a chain / runner / server / install process (checked every 5 min)
f=/var/tmp/jev-idle-count
if pgrep -f 'g2_[0-9]|ex_[0-9]|jevharness|jevbench|startlux_decision|llama-server|uv pip|hf download' >/dev/null; then echo 0 > \$f; exit 0; fi
n=\$(( \$(cat \$f 2>/dev/null || echo 0) + 1 )); echo \$n > \$f
if (( n * 5 >= $IDLE_MIN )); then shutdown -h now "jev-explore-gpu: idle $IDLE_MIN min"; fi
IDLE
  sudo chmod +x /usr/local/bin/jev-idle-stop
  echo '*/5 * * * * root /usr/local/bin/jev-idle-stop' | sudo tee /etc/cron.d/jev-idle-stop > /dev/null
  echo 0 | sudo tee /var/tmp/jev-idle-count > /dev/null
}
start_rec() {  # START file (run_main.sh format) with the overlay and registration hashes
  [[ -f "$OUT/START_$1.txt" ]] || printf 'step %s\nstarted %s\ncommit %s\noverlay sha256 %s\nplan sha256 %s\nhost %s\n%s\n' "$1" \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$(cut -c1-16 "$OUT/overlay.sha256")" \
    "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" "$(itype)" "${2:-}" > "$OUT/START_$1.txt"
}

stage_rearm() { arm_stops; itype > "$OUT/instance_type.txt"; nvidia-smi > "$OUT/nvidia_smi_start.txt" 2>&1; }

stage_t2setup() {  # tau2-bench at 5bfa7e3 + the harness in their own venv; tau2's runtime deps pinned to the cloud venv's
  if [[ ! -d "$T2SRC/src/tau2" ]]; then
    aws s3 cp "$B/code/$T2TAR" "$W/$T2TAR" --only-show-errors --region "$REGION" || return 1
    mkdir -p "$T2SRC" && tar -xzf "$W/$T2TAR" -C "$T2SRC" || return 1
  fi
  sha256sum "$W/$T2TAR" > "$OUT/tau2_tar.sha256" 2>/dev/null
  command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; } || return 1
  [[ -x "$T2V/bin/python" ]] || uv venv -q -p 3.11 "$T2V" || return 1
  local C=(-c configs/tau2/explore/t2venv_constraints.txt)
  uv pip install -q -p "$T2V/bin/python" "${C[@]}" -e . rich "tabulate>=0.9.0" "fastapi>=0.115.11" "uvicorn>=0.34.0" \
    "pandas>=2.2.3" "psutil>=7.0.0" "loguru>=0.7.3" "docstring-parser>=0.16" "litellm>=1.80.15,<1.82.7" "tenacity>=9.0.0" \
    "deepdiff>=8.4.2" "addict>=2.4.0" "toml>=0.10.2" "typer>=0.12.5" "requests>=2.31.0" websockets pip || return 1
  "$T2V/bin/python" -m pip install -q --no-deps --ignore-requires-python -e "$T2SRC" || return 1
  "$T2V/bin/python" -c "import jevharness.bench.suites.tau2 as T; assert T.HAVE_TAU2, T._TAU2_IMPORT_ERROR; import tau2, litellm; print('tau2 ok', tau2.__file__)" \
    > "$OUT/t2venv_check.txt" 2>&1 || { tail -5 "$OUT/t2venv_check.txt"; return 1; }
  uv pip freeze -p "$T2V/bin/python" > "$OUT/t2venv_pip_freeze.txt"
  tail -1 "$OUT/t2venv_check.txt"
}

stage_serve_rl() {  # the frozen row of the plan behind serve_jev on :8081 (as stage2_gpu.sh serve_ck), sha256 re-checked
  local adapter cal name args
  [[ -f "$W/base_model_path.txt" ]] || { echo "no base_model_path.txt (stage2_gpu.sh fetch)"; return 1; }
  "$HV/bin/python" - "$PLAN" "$CK" "$OUT/checkpoint_verify_frozen.json" <<'EOF' || return 1
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
  adapter=$(plan "d['rows']['frozen'].get('adapter') or ''")
  cal=$(plan "d['rows']['frozen'].get('calibration') or ''")
  name=$(plan "d['rows']['frozen']['served_name']")
  [[ "$name" == jevagentrl-4b ]] || { echo "served name $name != jevagentrl-4b"; return 1; }
  stage_stop_rl
  args=(--model "$(cat "$W/base_model_path.txt")" --name "$name" --port 8081)
  [[ -n "$adapter" ]] && args+=(--adapter "$CK/$adapter")
  if [[ -n "$cal" ]]; then [[ -f "$cal" ]] && args+=(--calibration "$cal") || args+=(--calibration "$CK/$cal"); fi
  # shellcheck disable=SC2207
  args+=($(plan "' '.join(d.get('serve_args') or [])"))
  echo "serve_jev ${args[*]}" > "$OUT/serve_frozen.cmd"
  nohup "$HV/bin/python" -m jevharness.training.serve_jev "${args[@]}" > "$OUT/serve_frozen.out" 2>&1 &
  echo $! > "$W/serve_ck.pid"
  for _ in $(seq 1 180); do
    if curl -sf "$RL_URL/health" > "$OUT/serve_frozen_health.json"; then
      curl -sf "$RL_URL/v1/models" > "$OUT/serve_frozen_models.json"; return 0
    fi
    kill -0 "$(cat "$W/serve_ck.pid")" 2>/dev/null || { echo "serve_jev died"; tail -20 "$OUT/serve_frozen.out"; return 1; }
    sleep 5
  done
  echo "serve_jev did not come up"; return 1
}
stage_stop_rl() { [[ -f "$W/serve_ck.pid" ]] && kill "$(cat "$W/serve_ck.pid")" 2>/dev/null; rm -f "$W/serve_ck.pid"; sleep 5; }

ids() {  # ids KEY [FROM TO]: task ids of the E1 list (task_ids or validation_gonogo), optionally a slice
  "$T2V/bin/python" -c "import json,sys; d=json.load(open('$IDS'))[sys.argv[1]]; a=int(sys.argv[2]) if len(sys.argv)>2 else 0
b=int(sys.argv[3]) if len(sys.argv)>3 else len(d); print('\n'.join(d[a:b]))" "$@"
}
t2arm() {  # t2arm RUNDIR ARM SUFFIX KEY FROM TO: one tau2 arm on a slice (run + rerun-errored + convert); RL windows logged
  local run=$1 arm=$2 sfx=$3 cfg rc t0 t1
  local -a TID
  mapfile -t TID < <(ids "$4" "$5" "$6")
  [[ ${#TID[@]} -gt 0 ]] || { echo "no task ids"; return 1; }
  cfg=configs/tau2/llm_only.yaml; [[ $arm == jevagent_best_rl ]] && cfg=$RLCFG
  local save; save="$(pwd)/$run/raw/telecom/$arm$sfx"
  mkdir -p "$save" "$run/telecom"
  local RUN=(--jev-config "$cfg" --jev-name "$arm" --domain telecom --save-to "$save" --agent jev_agent
    --agent-llm gemini/gemini-3.5-flash --user-llm gemini/gemini-3.5-flash-lite --num-trials 1 --max-concurrency 8
    --seed 300 --auto-resume --task-split-name full --task-ids "${TID[@]}" --agent-llm-args "$GEMINI_ARGS"
    --user-llm-args "$GEMINI_ARGS")
  t0=$(date -u +%FT%TZ)
  "$T2V/bin/python" -m jevharness.bench.suites.tau2 run "${RUN[@]}" > "$OUT/t2_${run##*/}_$arm$sfx.log" 2>&1
  rc=$?
  "$T2V/bin/python" -m jevharness.bench.suites.tau2 rerun-errored "$save" -- "${RUN[@]}" >> "$OUT/t2_${run##*/}_$arm$sfx.log" 2>&1
  t1=$(date -u +%FT%TZ)
  "$T2V/bin/python" -m jevharness.bench.suites.tau2 convert "$save/results.json" --arm "$arm" \
    --out "$run/telecom/$arm$sfx.jsonl" >> "$OUT/t2_${run##*/}_$arm$sfx.log" 2>&1 || rc=9
  if [[ $arm == jevagent_best_rl ]]; then
    "$T2V/bin/python" - "$OUT/${7:-e1_rl_windows}.json" "$t0" "$t1" "$(itype)" "$run$sfx" <<'EOF'
import json, os, sys
p, t0, t1, it, tag = sys.argv[1:6]
w = json.load(open(p)) if os.path.exists(p) else []
w.append({"start": t0, "end": t1, "instance_type": it, "run": tag})
json.dump(w, open(p, "w"), indent=1)
EOF
  fi
  # operational fields only: simulations, errors, API cost per simulation, System-1 models
  "$T2V/bin/python" - "$run/telecom/$arm$sfx.jsonl" >> "$OUT/e1_ops.txt" <<'EOF'
import json, sys, collections
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
errs = sum(1 for r in recs if r.get("error"))
c = sum(float(((r.get("meta") or {}).get("tau2") or {}).get("agent_cost") or 0) + float(((r.get("meta") or {}).get("tau2") or {}).get("user_cost") or 0) for r in recs)
m = collections.Counter(x.get("model") for r in recs for x in r.get("calls") or [] if x.get("component") == "s1")
print(f"{sys.argv[1]}: sims {len(recs)} errored {errs} api $/sim {c / max(1, len(recs)):.4f} s1 {dict(m)}")
EOF
  sync_run "$run"
  return $rc
}

stage_e1_val() {  # day-1 go/no-go: the RL arm on the 5 validation tasks (never evaluated); the session decides
  keys || return 1
  curl -sf "$RL_URL/health" > /dev/null || { echo "RL server not up"; return 1; }
  start_rec explore_telecom_rl_validation "tasks: validation_gonogo of $IDS ($(sha256sum "$IDS" | cut -c1-16))"
  t2arm "$E1V" jevagent_best_rl "" validation_gonogo 0 5 e1_val_windows
}

stage_e1_val2() {  # go/no-go round 2 (after the adapter fix): the RL arm on validation tasks 6-10; the session decides
  keys || return 1
  curl -sf "$RL_URL/health" > /dev/null || { echo "RL server not up"; return 1; }
  grep -q '^respond_fallback: defer' "$RLCFG" || { echo "$RLCFG lacks respond_fallback: defer"; return 1; }
  start_rec explore_telecom_rl_validation2 "tasks: validation_gonogo_2 of $IDS ($(sha256sum "$IDS" | cut -c1-16)); render $(sha256sum "$RLCFG" | cut -c1-16)"
  t2arm "$E1V2" jevagent_best_rl "" validation_gonogo_2 0 5 e1_val2_windows
}

stage_e1_main() {  # chunk A (ids 0-74: llm_only, then the RL arm), the cap check, chunk B (75-149)
  keys || return 1
  curl -sf "$RL_URL/health" > /dev/null || { echo "RL server not up"; return 1; }
  start_rec explore_telecom_rl "tasks: task_ids of $IDS ($(sha256sum "$IDS" | cut -c1-16)); render $(sha256sum "$RLCFG" | cut -c1-16)"
  t2arm "$E1" llm_only _A task_ids 0 75; st "e1 llm_only chunk A exit=$?"
  t2arm "$E1" jevagent_best_rl _A task_ids 0 75; st "e1 rl chunk A exit=$?"
  local verdict
  verdict=$("$T2V/bin/python" - "$E1V" "$E1" <<'EOF'
import sys, glob
sys.path.insert(0, "scripts")
import explore_telecom_rl as E
dirs = [d for d in glob.glob(sys.argv[1] + "*/raw/telecom/*") + glob.glob(sys.argv[2] + "/raw/telecom/*")]
tot = E.raw_spend(dirs)["total"]
a = E.raw_spend([d for d in dirs if d.endswith("_A")])["total"]
ok = tot + a <= E.CAP_USD
print(f"{'go' if ok else 'STOP'} spent ${tot:.2f}, chunk A ${a:.2f}, projected ${tot + a:.2f} vs cap ${E.CAP_USD:.0f}")
EOF
)
  echo "$verdict" >> "$OUT/e1_spend.txt"; st "e1 cap check: $verdict"
  [[ "$verdict" == go* ]] || { st "e1: chunk B not run (cap)"; return 0; }
  t2arm "$E1" llm_only _B task_ids 75 150; st "e1 llm_only chunk B exit=$?"
  t2arm "$E1" jevagent_best_rl _B task_ids 75 150; st "e1 rl chunk B exit=$?"
  "$T2V/bin/python" scripts/explore_telecom_rl.py spend "$E1V"*/raw/telecom/* "$E1"/raw/telecom/* --json > "$OUT/e1_spend.json"
}

# ---- E2: the StartLux ladder live (one size on the GPU; the vendor server at the ladder replay's revision)
sl_port() { "$HV/bin/python" -c "import sys; sys.path.insert(0, 'scripts'); import startlux_plan as P; print(P.SIZES['$1'][2])"; }
e2_budget() {  # e2_budget SIZE: every E2 record so far + a $3.50 reserve within the $15 cap
  "$HV/bin/python" - "runs/$E2RUN" "$1" <<'EOF'
import glob, json, sys
s = 0.0
for f in glob.glob(sys.argv[1] + "/*.jsonl"):
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and "task_id" in r:
            s += float(((r.get("usage") or {}).get("total") or {}).get("cost_usd") or 0)
ok = s + 3.5 <= 15.0
print(f"e2 {sys.argv[2]}: spent ${s:.2f} + reserve $3.50 vs cap $15 -> {'go' if ok else 'STOP'}")
sys.exit(0 if ok else 1)
EOF
}
stage_e2() {
  keys || return 1
  stage_stop_rl
  local s port key tag py v
  py=$(uv python find 3.11 2>/dev/null)
  for s in 0.8B 2B 9B; do
    v=$(e2_budget "$s"); rc=$?; echo "$v" >> "$OUT/e2_spend.txt"; st "$v"
    [[ $rc -eq 0 ]] || break
    STOP=1 bash scripts/aws/serve_startlux.sh > /dev/null 2>&1
    if ! SIZES=$s PYTHON="$py" MANIFEST="$OUT/manifest_e2_$s.json" PIN_MANIFEST="results/stage2/gpu/manifest_replay_$s.json" \
         bash scripts/aws/serve_startlux.sh > "$OUT/serve_sl_$s.log" 2>&1; then
      st "e2 $s: server not started (not run)"; continue
    fi
    "$HV/bin/python" -c "import json, sys; m = json.load(open('$OUT/manifest_e2_$s.json')); p = json.load(open('results/stage2/gpu/manifest_replay_$s.json'))
a = [x for x in m['sizes'] if x['size'] == '$s' and x.get('status') == 'original']
b = [x for x in p['sizes'] if x['size'] == '$s']
sys.exit(0 if a and b and a[0].get('weights_revision') == b[0].get('weights_revision') else 1)" \
      || { st "e2 $s: served revision differs from the ladder replay's (not run)"; STOP=1 bash scripts/aws/serve_startlux.sh; continue; }
    port=$(sl_port "$s"); tag=${s,,}; key=startlux_${tag/./_}
    start_rec "${E2RUN}_$tag" "size $s port $port revision pinned to results/stage2/gpu/manifest_replay_$s.json"
    # the static E1 decisions first (GPU only; C8 population, as A7.18 (e) 1), then the live arm
    "$HV/bin/python" -m jevharness.bench.decision_eval configs/decision/explore_bfcl_when2call_startlux_ladder.yaml \
      --out runs --only "$key" > "$OUT/e2_static_$tag.log" 2>&1
    st "e2 $s static E1 exit=$?"
    sync_run runs/e1_bfcl_when2call_startlux_ladder
    for pass in 1 2 3; do
      JEV_EVAL_RUNS=1 "$HV/bin/jevbench" run configs/experiments/$E2RUN.yaml --out runs -o workers=4 \
        -o "s1_backends.$key={type: jev, name: startlux-$tag, base_url: \"http://127.0.0.1:$port\", model: StartLux-Decision-$s, api_key: \"\", price_model: StartLux-Decision-$s, timeout: 60.0}" \
        -o "arms=[{name: sl${tag}_ts_replace_f, s1: $key, tool_select: {mode: replace}}]" > "$OUT/e2_${tag}_pass$pass.log" 2>&1
      st "e2 $s pass $pass exit=$?"
    done
    "$HV/bin/python" - "runs/$E2RUN/sl${tag}_ts_replace_f.jsonl" >> "$OUT/e2_ops.txt" <<'EOF'
import json, sys, collections
last = {}
for l in open(sys.argv[1]):
    try:
        r = json.loads(l)
    except ValueError:
        continue
    if "task_id" in r:
        last[r["task_id"]] = r
errs = sum(1 for r in last.values() if r.get("error"))
c = sum(float(((r.get("usage") or {}).get("total") or {}).get("cost_usd") or 0) for r in last.values())
m = collections.Counter(x.get("model") for r in last.values() for x in r.get("calls") or [] if x.get("component") == "s1")
print(f"{sys.argv[1]}: tasks {len(last)} latest-errored {errs} cost/ep ${c / max(1, len(last)):.4f} s1 {dict(m)}")
EOF
    STOP=1 bash scripts/aws/serve_startlux.sh
    sync_run "runs/$E2RUN"
  done
}

stage_final() {
  stage_stop_rl; STOP=1 bash scripts/aws/serve_startlux.sh > /dev/null 2>&1
  nvidia-smi > "$OUT/nvidia_smi_end.txt" 2>&1
  for d in "$E1V" "$E1V2" "$E1" "runs/$E2RUN" runs/e1_bfcl_when2call_startlux_ladder; do sync_run "$d"; done
}

for s in $STAGES; do
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $s"; continue; fi
  st "stage $s start ($(git log --oneline -1 | cut -c1-50); overlay $(cut -c1-12 "$OUT/overlay.sha256"))"
  t0=$(date +%s)
  ( "stage_$s" ) > "$OUT/$s.log" 2>&1
  rc=$?
  st "stage $s exit=$rc ($(( ($(date +%s) - t0) / 60 )) min)"
  if [[ $rc -ne 0 && $s != e2 && $s != final ]]; then st "stopping the chain after $s failed"; broke=1; break; fi
done
if [[ "${broke:-0}" == 1 || " $STAGES " == *" final "* ]]; then
  stage_stop_rl; STOP=1 bash scripts/aws/serve_startlux.sh > /dev/null 2>&1
  st "servers stopped; the box stops itself after $IDLE_MIN idle minutes"
fi
st "box done [$STAGES]"
