#!/usr/bin/env bash
# Stage-2 StartLux-Decision rows (docs/PREREG_A7_STAGE2.md A7.18) on one g6e.xlarge / g6e.2xlarge (1x L40S 48 GB), the
# patterns of scripts/aws/startlux_smoke.sh. One GPU: the 4B (original weights) rows first, then the 27B (Q8_0 GGUF via
# llama.cpp, A7.18 (g); one slot, 32k context) rows; serve_startlux.sh runs once per size.
#
#   LOCAL (cloud session; `env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws`):
#     bash scripts/aws/stage2_startlux.sh launch       # Name=jev-stage2-startlux, Project=jev; shutdown behaviour stop;
#                                                      # user-data: 24 h power-off guard
#     bash scripts/aws/stage2_startlux.sh push         # git bundle of $BRANCH -> s3://$BUCKET/code/stage2_startlux.bundle
#     bash scripts/aws/stage2_startlux.sh send "STAGES"   # SSM: the box side, detached, with STAGES
#     bash scripts/aws/stage2_startlux.sh status | stop | start
#   BOX (as ubuntu; called by `send`), stages in the registered order (local steps between them in brackets):
#     setup serve4b manifest          [commit results/stage2/startlux_manifest.json before any StartLux evaluation request]
#     cal                             [threshold_theory fits + scripts/fill_backend_tau.py, commit; tariff plan, commit]
#     tariff                          [scripts/stage2_tariff.py -> results/stage2/tariff_sl4b.json, commit]
#     sl4b states4b e1_4b stop4b e1_intern llama serve27b manifest   [commit the manifest with the 27B entry]
#     states27b e1_27b stopall manifest poweroff
# Outputs: results/stage2/startlux/ on the box -> s3://$BUCKET/stage2_startlux/; run directories -> s3://$BUCKET/
# stage2_startlux/runs/<run>/. While a config runs only operational fields are read (A7.6): errors, cost per episode,
# served model, unpriced models; the box writes no success number anywhere.
set -uo pipefail
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-main}
TAG=jev-stage2-startlux
BUNDLE=stage2_startlux.bundle
S3OUT="$B/stage2_startlux"

awsl() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws --region "$REGION" "$@"; }
instance_id() {
  awsl ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text
}

if [[ "$MODE" == launch ]]; then
  existing=$(instance_id)
  if [[ -n "$existing" ]]; then echo "already exists: $existing" >&2; exit 1; fi
  TYPES=${TYPES:-"g6e.xlarge g6e.2xlarge"}
  SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
  SG=${SG:?set SG to a security group id}            # jev-train-noingress (SSM only)
  PROFILE=${PROFILE:?set PROFILE to an instance profile}         # SSM + the bucket + ssm:GetParameter on ${SSM_PREFIX:?set SSM_PREFIX}/*
  AMI=${AMI:-<aws-resource-id>}         # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.14 (Ubuntu 26.04) 20260926
  GUARD_MIN=${GUARD_MIN:-1440}
  UD=$(mktemp)
  trap 'rm -f "$UD" "$UD.err"' EXIT
  cat > "$UD" <<EOF
#!/bin/bash
shutdown -h +$GUARD_MIN "jev-stage2-startlux: 24 h guard"
mkdir -p /opt/jev /opt/startlux && chown ubuntu:ubuntu /opt/jev /opt/startlux
EOF
  for t in $TYPES; do
    for s in $SUBNETS; do
      if id=$(awsl ec2 run-instances --image-id "$AMI" --instance-type "$t" --subnet-id "$s" \
            --security-group-ids "$SG" --iam-instance-profile "Name=$PROFILE" \
            --instance-initiated-shutdown-behavior stop --user-data "file://$UD" \
            --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","Throughput":250,"DeleteOnTermination":true}}]' \
            --metadata-options HttpTokens=required,HttpEndpoint=enabled \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG},{Key=Project,Value=jev}]" \
              "ResourceType=volume,Tags=[{Key=Name,Value=$TAG},{Key=Project,Value=jev}]" \
            --query 'Instances[0].InstanceId' --output text 2>"$UD.err"); then
        echo "$id $t $s"
        exit 0
      fi
      if ! grep -q -E 'InsufficientInstanceCapacity|Unsupported|not supported|InstanceLimitExceeded|VcpuLimitExceeded' "$UD.err"; then
        cat "$UD.err" >&2
        exit 1
      fi
      echo "no capacity: $t in $s ($(grep -o -E '[A-Za-z]+(Capacity|Limit)[A-Za-z]*|Unsupported' "$UD.err" | head -1))" >&2
    done
  done
  echo "no capacity for any of: $TYPES" >&2
  exit 1
fi

if [[ "$MODE" == push ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  tmp=$(mktemp -d)
  git -C "$ROOT" bundle create "$tmp/$BUNDLE" "$BRANCH" 2>/dev/null
  awsl s3 cp "$tmp/$BUNDLE" "$B/code/$BUNDLE" --only-show-errors
  git -C "$ROOT" show "$BRANCH:scripts/aws/stage2_startlux.sh" > "$tmp/stage2_startlux.sh"
  awsl s3 cp "$tmp/stage2_startlux.sh" "$B/code/stage2_startlux.sh" --only-show-errors
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
    echo 'f=/opt/jev/s2_$(date +%s%N).sh'      # a private copy per invocation (bash reads a running script lazily)
    echo "aws s3 cp $B/code/stage2_startlux.sh \$f --only-show-errors --region $REGION"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/s2_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/s2_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$iid" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/s2_driver.sh /tmp/s2_driver.sh --only-show-errors && bash /tmp/s2_driver.sh'\"]" \
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
  sed -n '2,20p' "$0"
  exit 0
fi

# =============================================================================================== box side
STAGES=${1:-}
W=/opt/jev
cd "$W" || exit 1
MYB="$W/bundle.$$.bundle"
aws s3 cp "$B/code/$BUNDLE" "$MYB" --only-show-errors --region "$REGION"
if [[ ! -d "$W/stage2" ]]; then git clone -q -b "$BRANCH" "$MYB" "$W/stage2"; fi
cd "$W/stage2" || exit 1
git fetch -q "$MYB" "$BRANCH" && git reset -q --hard FETCH_HEAD
rm -f "$MYB"
OUT=results/stage2/startlux                  # box outputs (synced to S3; committed from the cloud session as needed)
HV=$W/hvenv                                  # the harness venv (the serving venv is serve_startlux.sh's own)
export BASE=/opt/startlux
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER JEV_ALLOW_PENDING_TAU
mkdir -p "$OUT" runs
SL4B='{type: jev, name: startlux-4b, base_url: "http://127.0.0.1:8090", model: StartLux-Decision-4B, api_key: "", price_model: StartLux-Decision-4B, timeout: 60.0, cache: cache/s1_stage2_startlux.sqlite}'
SL27B='{type: jev, name: startlux-27b, base_url: "http://127.0.0.1:8091", model: StartLux-Decision-27B, api_key: "", price_model: StartLux-Decision-27B, timeout: 120.0, cache: cache/s1_stage2_startlux.sqlite}'
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$OUT/status.txt"
  aws s3 sync "$OUT" "$S3OUT" --only-show-errors --region "$REGION"
}
sync_run() { aws s3 sync "runs/$1" "$S3OUT/runs/$1" --only-show-errors --region "$REGION"; }
py311() { uv python find 3.11 2>/dev/null; }
keys() {  # the System-2 keys from SSM (never on disk, never printed)
  local v val
  for v in GEMINI_API_KEY GEMINI_API_KEYS OPENROUTER_API_KEY; do
    val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value \
          --output text 2>/dev/null) || val=""
    if [[ -n "$val" && "$val" != None ]]; then export "$v=$val"; fi
  done
  unset val
  [[ -n "${GEMINI_API_KEY:-}${GEMINI_API_KEYS:-}" ]] || { echo "no Gemini key in SSM"; return 1; }
}
up() { curl -sf "http://127.0.0.1:$1/health" > /dev/null || { echo "server on :$1 not up"; return 1; }; }
ops() {  # operational fields only: records, latest-errored tasks, cost per episode, System-1 models
  "$HV/bin/python" - "$@" <<'EOF'
import json, sys, glob, os, collections
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        last = {}
        for line in open(f):
            if line.strip():
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
        print(f"{os.path.basename(f)[:-6]}: tasks {len(last)} latest-errored {errs} cost/ep ${cost / len(last):.4f} s1 {dict(m)}")
EOF
}
three() {  # three <name>: three passes of the remaining args (evaluation commands only get JEV_EVAL_RUNS=1 via EVAL=1)
  local name=$1 rc; shift
  for pass in 1 2 3; do
    if [[ "${EVAL:-0}" == 1 ]]; then JEV_EVAL_RUNS=1 "$@"; else "$@"; fi
    rc=$?
    st "$name pass $pass exit=$rc"
  done
}

stage_setup() {  # host facts, uv + Python 3.11, the harness venv (with the test extras for the launch gate)
  nvidia-smi > "$OUT/nvidia_smi_start.txt" 2>&1
  { echo "kernel $(uname -r)"; echo "os $(. /etc/os-release; echo "$PRETTY_NAME")"; echo "nproc $(nproc)";
    free -g | awk '/Mem:/ {print "ram_gb " $2}'; df -h / | tail -1; command -v docker && docker --version;
    python3 --version; curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT http://169.254.169.254/latest/api/token \
      -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/instance-type; echo; } \
    > "$OUT/host.txt" 2>&1
  command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; } || return 1
  uv python install 3.11 || return 1
  [[ -x "$HV/bin/python" ]] || uv venv -q -p 3.11 "$HV" || return 1
  uv pip install -q -p "$HV/bin/python" -e ".[dev,bench]" || return 1
  "$HV/bin/python" -c "import jevharness, sys; print('harness ok', sys.version.split()[0])"
}

stage_serve4b() {  # the vendor's server for the 4B on :8090 (original weights)
  SIZES=4B PYTHON="$(py311)" MANIFEST="$OUT/manifest_4B.json" bash scripts/aws/serve_startlux.sh
  local rc=$?
  "$BASE/venv/bin/pip" freeze > "$OUT/serving_pip_freeze.txt" 2>/dev/null
  return $rc
}

stage_manifest() {  # results/stage2/startlux_manifest.json from the per-size invocations (4B, then 27B)
  "$HV/bin/python" - "$OUT" <<'EOF'
import json, os, sys
out = sys.argv[1]
man = None
for size in ("4B", "27B"):
    p = os.path.join(out, f"manifest_{size}.json")
    if not os.path.exists(p):
        continue
    m = json.load(open(p))
    if man is None:
        man = {**m, "invocations": []}
        for k in ("host", "environment", "sizes"):
            man[k] = []
    man["invocations"].append({"sizes": size, "file": os.path.basename(p), "written": m.get("written")})
    for k in ("host", "environment", "sizes"):
        man[k] += m.get(k, [])
if man:
    man["note"] = ("Stage-2 serving manifest (A7.18 (d)): one L40S; serve_startlux.sh run once per size (SIZES=4B, then "
                   "SIZES=27B after the 4B rows), the 4B stopped before the 27B started")
    json.dump(man, open(os.path.join(out, "startlux_manifest.json"), "w"), indent=1)
    print("merged", [i["sizes"] for i in man["invocations"]])
EOF
}

stage_cal() {  # A7.18 (e) 4: sl4b_shadow_cal on the calibration seeds 5000-5149 (not an evaluation run)
  keys || return 1; up 8090 || return 1
  three cal "$HV/bin/jevbench" run configs/experiments/jevgym_shadow_cal_startlux.yaml --out runs -o workers=4
  "$HV/bin/jevbench" report runs/e2a_jevgym_shadow_cal_startlux > /dev/null 2>&1
  ops runs/e2a_jevgym_shadow_cal_startlux > "$OUT/cal_ops.txt"
  sync_run e2a_jevgym_shadow_cal_startlux
}

stage_tariff() {  # A7.18 (e) 5 / A7.13 (iii): the committed plan results/stage2/tariff_sl4b_plan.json (dev seeds)
  keys || return 1; up 8090 || return 1
  local plan=results/stage2/tariff_sl4b_plan.json
  [[ -f "$plan" ]] || { echo "no committed tariff plan"; return 1; }
  local wseed wn seed n name
  read -r wseed wn seed n name < <("$HV/bin/python" -c "import json; p = json.load(open('$plan'));
print(p['warmup']['seed0'], p['warmup']['n'], p['workload']['seed0'], p['workload']['n'], p['workload']['run_name'])")
  local arm="[{name: jevagent_best_sl4b, s1: startlux_4b, profile: configs/harness/jevagent_best.yaml, profile_s2: gemini-3.5-flash, profile_backend: startlux_4b}]"
  # warm-up (not in the window): the same configuration on its own dev seeds, covering the input lengths
  "$HV/bin/jevbench" run configs/experiments/stage2_jevgym.yaml --out runs -o "name=${name}_warmup" suite.seed0="$wseed" \
    suite.n="$wn" n_tasks="$wn" workers=4 "s1_backends.startlux_4b=$SL4B" "arms=$arm" \
    "s1_backends.startlux_4b.cache=cache/s1_dev_tariff_sl4b.sqlite" > "$OUT/tariff_warmup.log" 2>&1
  local t0 t1 rc
  t0=$(date -u +%FT%TZ)
  "$HV/bin/jevbench" run configs/experiments/stage2_jevgym.yaml --out runs -o "name=$name" suite.seed0="$seed" \
    suite.n="$n" n_tasks="$n" workers=4 "s1_backends.startlux_4b=$SL4B" "arms=$arm" \
    "s1_backends.startlux_4b.cache=cache/s1_dev_tariff_sl4b.sqlite" > "$OUT/tariff_workload.log" 2>&1
  rc=$?
  t1=$(date -u +%FT%TZ)
  printf '{"window_start": "%s", "window_end": "%s", "exit": %s, "run": "runs/%s", "instance_type": "%s"}\n' "$t0" "$t1" \
    "$rc" "$name" "$(tail -1 "$OUT/host.txt")" > "$OUT/tariff_window.json"
  ops "runs/$name" > "$OUT/tariff_ops.txt"
  sync_run "$name"; sync_run "${name}_warmup"
  return $rc
}

stage_sl4b() {  # S2-4S: jevagent_best_sl4b on the fresh seeds (stage2_jevgym, this arm only), three passes
  keys || return 1; up 8090 || return 1
  "$HV/bin/python" scripts/stage2_spec.py --gate stage2_jevgym --startlux > "$OUT/gate_sl4b.txt" 2>&1 \
    || { echo "gate failed"; return 2; }
  [[ -f "$OUT/START_stage2_jevgym_sl4b.txt" ]] || printf 'step %s\nstarted %s\ncommit %s\nplan sha256 %s\nhost %s\n' \
    stage2_jevgym_sl4b "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" \
    "$(tail -1 "$OUT/host.txt")" > "$OUT/START_stage2_jevgym_sl4b.txt"
  local arm="[{name: jevagent_best_sl4b, s1: startlux_4b, profile: configs/harness/jevagent_best.yaml, profile_s2: gemini-3.5-flash, profile_backend: startlux_4b}]"
  EVAL=1 three sl4b "$HV/bin/jevbench" run configs/experiments/stage2_jevgym.yaml --out runs -o workers=4 \
    "s1_backends.startlux_4b=$SL4B" "arms=$arm"
  ops runs/stage2_jevgym > "$OUT/sl4b_ops.txt"
  sync_run stage2_jevgym
}

stage_states4b() {  # A7.18 (e) 3: sl4b_ts_replace_f (n = 200), three passes; the 27B arm stays a placeholder (skipped)
  keys || return 1; up 8090 || return 1
  "$HV/bin/python" scripts/stage2_spec.py --gate stage2_startlux_states > "$OUT/gate_states4b.txt" 2>&1 \
    || { echo "gate failed"; return 2; }
  [[ -f "$OUT/START_stage2_startlux_states.txt" ]] || printf 'step %s\nstarted %s\ncommit %s\nplan sha256 %s\n' \
    stage2_startlux_states "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)" "$(sha256sum docs/PREREG_A7_STAGE2.md | cut -c1-16)" \
    > "$OUT/START_stage2_startlux_states.txt"
  EVAL=1 three states4b "$HV/bin/jevbench" run configs/experiments/stage2_startlux_states.yaml --out runs \
    -o "s1_backends.startlux_4b=$SL4B"
  ops runs/stage2_startlux_states > "$OUT/states4b_ops.txt"
  sync_run stage2_startlux_states
}

stage_states27b() {  # A7.18 (e) 3: sl27b_ts_replace_f (27B (Q8_0)), three passes; the 4B arm is complete (placeholder again)
  keys || return 1; up 8091 || return 1
  EVAL=1 three states27b "$HV/bin/jevbench" run configs/experiments/stage2_startlux_states.yaml --out runs \
    -o "s1_backends.startlux_27b=$SL27B"
  ops runs/stage2_startlux_states > "$OUT/states27b_ops.txt"
  sync_run stage2_startlux_states
}

e1() {  # e1 <backend>: the E1 static decisions of A7.18 (e) 1 (a second invocation re-asks errored items only)
  for k in 1 2; do
    "$HV/bin/python" -m jevharness.bench.decision_eval configs/decision/bfcl_when2call_startlux.yaml --out runs --only "$1"
    st "e1 $1 invocation $k exit=$?"
  done
  sync_run e1_bfcl_when2call_startlux
}
stage_e1_4b() { up 8090 || return 1; e1 startlux_4b; }
stage_e1_27b() { up 8091 || return 1; e1 startlux_27b; }
stage_e1_intern() {  # Intern-Decision-4B in-process (torch + transformers as the Stage-1 GPU rounds: 2.14.1 cu130, 5.18.0)
  "$HV/bin/python" -c "import torch, transformers" 2>/dev/null || {
    uv pip install -q -p "$HV/bin/python" torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 &&
    uv pip install -q -p "$HV/bin/python" transformers==5.18.0 "accelerate>=1.0" safetensors; } || return 1
  "$HV/bin/python" -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())" \
    > "$OUT/intern_env.txt" 2>&1
  e1 intern_4b
}

stage_stop4b() { STOP=1 bash scripts/aws/serve_startlux.sh; sleep 5; nvidia-smi --query-gpu=memory.used --format=csv; }

stage_llama() {  # llama-server with CUDA for the 27B (Q8_0): the official CUDA server image, else a source build
  local bin=$BASE/bin img=ghcr.io/ggml-org/llama.cpp:server-cuda
  mkdir -p "$bin"
  if command -v docker > /dev/null && sudo docker pull -q "$img"; then
    cat > "$bin/llama-server" <<EOF
#!/bin/bash
# llama.cpp's CUDA server image ($img); the model directory is mounted read-only at the same path
exec sudo docker run --rm --init --gpus "device=\${CUDA_VISIBLE_DEVICES:-all}" --network host \\
  -v $BASE/models:$BASE/models:ro $img "\$@"
EOF
    chmod +x "$bin/llama-server"
    sudo docker inspect --format '{{index .RepoDigests 0}}' "$img" > "$OUT/llama_image_digest.txt"
  else
    sudo apt-get install -y -q cmake build-essential > /dev/null
    [[ -d $BASE/llama.cpp ]] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$BASE/llama.cpp"
    cmake -S "$BASE/llama.cpp" -B "$BASE/llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null || return 1
    cmake --build "$BASE/llama.cpp/build" --target llama-server -j "$(nproc)" > /dev/null || return 1
    ln -sf "$BASE/llama.cpp/build/bin/llama-server" "$bin/llama-server"
    git -C "$BASE/llama.cpp" rev-parse HEAD > "$OUT/llama_source_commit.txt"
  fi
  "$bin/llama-server" --version > "$OUT/llama_server_version.txt" 2>&1
  cat "$OUT/llama_server_version.txt"
}

stage_serve27b() {  # the 27B on :8091 (one L40S < 64 GiB: the Q8_0 GGUF via llama.cpp, A7.18 (g))
  SIZES=27B PYTHON="$(py311)" LLAMA_SERVER="$BASE/bin/llama-server" MANIFEST="$OUT/manifest_27B.json" \
    bash scripts/aws/serve_startlux.sh
}

stage_stopall() {
  STOP=1 bash scripts/aws/serve_startlux.sh; sudo docker ps -q | xargs -r sudo docker stop > /dev/null
  mkdir -p "$OUT/logs"
  for f in "$BASE"/logs/*.log; do [[ -f "$f" ]] && tail -n 200 "$f" > "$OUT/logs/$(basename "$f")"; done
  nvidia-smi > "$OUT/nvidia_smi_end.txt" 2>&1
}

stage_poweroff() { aws s3 sync "$OUT" "$S3OUT" --only-show-errors --region "$REGION"; sudo shutdown -h +1 "jev-stage2-startlux: done"; }

for s in $STAGES; do
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $s"; continue; fi
  st "stage $s start ($(git log --oneline -1 | cut -c1-60))"
  t0=$(date +%s)
  ( "stage_$s" ) > "$OUT/$s.log" 2>&1
  rc=$?
  st "stage $s exit=$rc ($(( ($(date +%s) - t0) / 60 )) min)"
  if [[ $rc -ne 0 && "${KEEP_GOING:-0}" != 1 && "$s" != poweroff ]]; then
    st "stopping the chain after $s failed"
    break
  fi
done
st "box done [$STAGES]"
