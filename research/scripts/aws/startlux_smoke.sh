#!/usr/bin/env bash
# GPU smoke test of the StartLux-Decision servers (docs/PREREG_A7_STAGE2.md A7.18) on one g6e.xlarge (1x L40S 48 GB):
# scripts/aws/serve_startlux.sh for the 4B (original weights, fast kernels) and then the 27B (Q8_0 GGUF via llama.cpp,
# A7.18 (g)) one after the other on the one GPU, and a DEVELOPMENT-ONLY end-to-end check of the 4B through the runner
# (JevGym seeds 0-19, Flash, one tool_select replace arm, the Jev wire backend on the local server, one worker, no
# System-1 cache). Not an evaluation: no success or accuracy number is computed, printed or synced here; the end-to-end
# summary reads only errors, System-1 call latencies / tokens and the served model.
#
#   LOCAL (cloud session; `env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws`):
#     bash scripts/aws/startlux_smoke.sh launch          # g6e.xlarge, first AZ with capacity, then g6e.2xlarge;
#                                                        # Name=jev-startlux-smoke, Project=jev; shutdown behaviour stop;
#                                                        # user-data: 4 h power-off guard
#     bash scripts/aws/startlux_smoke.sh push            # git bundle of this branch -> s3://$BUCKET/code/startlux_smoke.bundle
#     bash scripts/aws/startlux_smoke.sh send "STAGES"   # SSM: run the box side detached with STAGES
#     bash scripts/aws/startlux_smoke.sh status          # the box's status.txt from S3
#     bash scripts/aws/startlux_smoke.sh stop            # stop the jev-startlux-smoke instance (no other box)
#   BOX (as ubuntu; called by `send`):
#     bash startlux_smoke.sh box "setup serve4b probe4b e2e stop4b llama serve27b probe27b stopall manifest poweroff"
# Outputs: results/stage2/startlux_smoke/ on the box, synced to s3://$BUCKET/startlux_smoke/ (the dev run's
# trajectories go to s3://$BUCKET/startlux_smoke/dev_run/ for audit and are not copied anywhere else).
set -uo pipefail
MODE=${1:-help}
shift || true
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}
B="s3://$BUCKET"
REGION=${AWS_REGION:-us-east-1}
BRANCH=${BRANCH:-startlux-gpu-smoke}
TAG=jev-startlux-smoke
BUNDLE=startlux_smoke.bundle
S3OUT="$B/startlux_smoke"

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
  GUARD_MIN=${GUARD_MIN:-240}
  UD=$(mktemp)
  trap 'rm -f "$UD" "$UD.err"' EXIT
  cat > "$UD" <<EOF
#!/bin/bash
shutdown -h +$GUARD_MIN "jev-startlux-smoke: 4 h guard"
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
  git -C "$ROOT" show "$BRANCH:scripts/aws/startlux_smoke.sh" > "$tmp/startlux_smoke.sh"
  awsl s3 cp "$tmp/startlux_smoke.sh" "$B/code/startlux_smoke.sh" --only-show-errors
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
    echo 'f=/opt/jev/sl_$(date +%s%N).sh'      # a private copy per invocation (bash reads a running script lazily)
    echo "aws s3 cp $B/code/startlux_smoke.sh \$f --only-show-errors --region $REGION"
    printf 'cd /opt/jev && setsid nohup bash "$f" box %q >> /opt/jev/sl_box.log 2>&1 < /dev/null &\n' "$STAGES"
    echo 'echo started'
  } > "$tmp"
  awsl s3 cp "$tmp" "$B/code/sl_driver.sh" --only-show-errors
  rm -f "$tmp"
  awsl ssm send-command --instance-ids "$iid" --document-name AWS-RunShellScript \
    --parameters "commands=[\"sudo -u ubuntu -H bash -lc 'aws s3 cp $B/code/sl_driver.sh /tmp/sl_driver.sh --only-show-errors && bash /tmp/sl_driver.sh'\"]" \
    --query 'Command.CommandId' --output text
  exit 0
fi

if [[ "$MODE" == status ]]; then
  awsl s3 cp "$S3OUT/status.txt" - 2>/dev/null | tail -n "${1:-40}"
  exit 0
fi

if [[ "$MODE" == stop ]]; then
  iid=$(instance_id)
  [[ -n "$iid" ]] || { echo "no $TAG instance" >&2; exit 1; }
  awsl ec2 stop-instances --instance-ids "$iid" --query 'StoppingInstances[].[InstanceId,CurrentState.Name]' --output text
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
MYB="$W/bundle.$$.bundle"
aws s3 cp "$B/code/$BUNDLE" "$MYB" --only-show-errors --region "$REGION"
if [[ ! -d "$W/startlux-smoke" ]]; then git clone -q -b "$BRANCH" "$MYB" "$W/startlux-smoke"; fi
cd "$W/startlux-smoke" || exit 1
git fetch -q "$MYB" "$BRANCH" && git reset -q --hard FETCH_HEAD
rm -f "$MYB"
OUT=results/stage2/startlux_smoke
RUNS=runs/startlux_smoke                     # the development run directory (seeds 0-19)
DEVNAME=startlux_smoke_dev_s0
HV=$W/hvenv                                  # the harness venv (the serving venv is serve_startlux.sh's own)
export BASE=/opt/startlux
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH" PYTHONUNBUFFERED=1
unset JEV_EVAL_RUNS JEV_ALLOW_PLACEHOLDER
mkdir -p "$OUT" "$RUNS"
st() {
  echo "$(date -u +%m-%dT%H:%M:%S) $*" >> "$OUT/status.txt"
  aws s3 sync "$OUT" "$S3OUT" --only-show-errors --region "$REGION"
}
py311() { uv python find 3.11 2>/dev/null; }

stage_setup() {  # host facts, uv + Python 3.11 (the serving venv and the harness venv), the harness package
  nvidia-smi > "$OUT/nvidia_smi_start.txt" 2>&1
  { echo "kernel $(uname -r)"; echo "os $(. /etc/os-release; echo "$PRETTY_NAME")"; echo "nproc $(nproc)";
    free -g | awk '/Mem:/ {print "ram_gb " $2}'; df -h / | tail -1; command -v docker && docker --version;
    nvcc --version 2>/dev/null | tail -2; python3 --version; } > "$OUT/host.txt" 2>&1
  command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; } || return 1
  uv python install 3.11 || return 1
  [[ -x "$HV/bin/python" ]] || uv venv -q -p 3.11 "$HV" || return 1
  uv pip install -q -p "$HV/bin/python" -e . || return 1
  "$HV/bin/python" -c "import jevharness, sys; print('harness ok', sys.version.split()[0])"
}

stage_serve4b() {  # the vendor's server for the 4B on :8090 (original weights; installs the serving env on first use)
  SIZES=4B PYTHON="$(py311)" MANIFEST="$OUT/manifest_4B.json" bash scripts/aws/serve_startlux.sh
  local rc=$?
  "$BASE/venv/bin/pip" freeze > "$OUT/serving_pip_freeze.txt" 2>/dev/null
  return $rc
}

probe() {  # probe PORT TAG: informational, not registered: the two smoke requests repeated (one at a time), the
  # server's /health and /v1/models, and one long-state request (~8k tokens; the 27B's llama-server runs -c 16384
  # --parallel 4, i.e. 4096 tokens per slot unless its KV cache is unified)
  "$HV/bin/python" - "$1" "$2" "$OUT" <<'EOF'
import json, sys, time, urllib.request
sys.path.insert(0, "scripts")
import startlux_plan as SP
port, tag, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
base = f"http://127.0.0.1:{port}"
def get(p):
    try:
        with urllib.request.urlopen(base + p, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:300]}
def post(body, timeout=300):
    t0 = time.perf_counter()
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        bad = {k: SP.answer_problems(q, (d.get("answers") or {}).get(k)) for k, q in body["questions"].items()}
        return (time.perf_counter() - t0) * 1000, d, {k: v for k, v in bad.items() if v}
    except Exception as e:  # noqa: BLE001
        msg = e.read().decode()[:300] if hasattr(e, "read") else str(e)[:300]
        return (time.perf_counter() - t0) * 1000, None, {"_error": f"{type(e).__name__}: {msg}"}
def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None
res = {"port": port, "health": get("/health"), "models": get("/v1/models")}
small, wide = SP.smoke_requests()
for name, body, n in (("three_questions", small, 20), ("choice_69", wide, 10)):
    post(body)                                                   # warm-up, not timed
    ms, errs, toks = [], [], None
    for _ in range(n):
        t, d, bad = post(body)
        ms.append(t)
        if bad:
            errs.append(bad)
        elif d:
            toks = (d.get("usage") or {}).get("input_tokens")
    res[name] = {"n": n, "p50_ms": pct(ms, .5), "p95_ms": pct(ms, .95), "max_ms": max(ms), "errors": errs[:3],
                 "n_errors": len(errs), "input_tokens": toks}
lines = "\n".join(f"[{i:04d}] worker-{i % 7} heartbeat ok; queue depth {i % 13}; cache hit ratio 0.{i % 97:02d}"
                  for i in range(500))
long_body = {"state": {"log": lines, "ticket": "Customer asks to cancel pending order 7781."},
             "questions": {"next": {"type": "choice", "instructions": "Which tool should be called next?",
                                    "criteria": {"cancel_order": "Cancel a pending order", "lookup_order": "Look up an order",
                                                 "escalate": "Hand over to a human"}}}}
t, d, bad = post(long_body)
res["long_state"] = {"ms": t, "ok": not bad, "problems": bad or None,
                     "input_tokens": ((d or {}).get("usage") or {}).get("input_tokens")}
json.dump(res, open(f"{out}/probe_{tag}.json", "w"), indent=1)
print(json.dumps({k: (v if k in ("health",) else (v.get("p50_ms") if isinstance(v, dict) and "p50_ms" in v else v))
                  for k, v in res.items() if k != "models"}, default=str)[:2000])
EOF
}
stage_probe4b() { probe 8090 4B; }
stage_probe27b() { probe 8091 27B; }

stage_e2e() {  # DEVELOPMENT ONLY: seeds 0-19 (suite seed0 0, n 50), Flash, tool_select replace on the 4B, workers 1
  local v val
  for v in GEMINI_API_KEY GEMINI_API_KEYS; do
    val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value \
          --output text 2>/dev/null) || val=""
    if [[ -n "$val" && "$val" != None ]]; then export "$v=$val"; fi
  done
  unset val
  [[ -n "${GEMINI_API_KEY:-}${GEMINI_API_KEYS:-}" ]] || { echo "no Gemini key in SSM"; return 1; }
  curl -sf http://127.0.0.1:8090/health > "$OUT/e2e_server_health.json" || { echo "4B server not up"; return 1; }
  curl -sf http://127.0.0.1:8090/v1/models > "$OUT/e2e_server_models.json"
  "$HV/bin/jevbench" run configs/experiments/stage2_startlux_states.yaml --out "$RUNS" -o \
    "name=$DEVNAME" suite.seed0=0 suite.n=50 n_tasks=20 workers=1 s1_cache=null \
    's1_backends.startlux_4b={type: jev, name: startlux-4b, base_url: "http://127.0.0.1:8090", model: StartLux-Decision-4B, api_key: "", price_model: StartLux-Decision-4B, timeout: 60.0}' \
    'arms=[{name: dev_sl4b_ts_replace_f, s1: startlux_4b, tool_select: {mode: replace}}]'
  local rc=$?
  unset GEMINI_API_KEY GEMINI_API_KEYS
  # the summary reads errors, System-1 call latencies / tokens / models and the API spend -- never `result`
  "$HV/bin/python" - "$RUNS/$DEVNAME" "$OUT/e2e_summary.json" <<'EOF'
import collections, glob, json, os, sys
run, out = sys.argv[1], sys.argv[2]
def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 1) if xs else None
summ = {"run_dir": run, "arms": {}}
for f in sorted(glob.glob(os.path.join(run, "*.jsonl"))):
    last = {}
    for line in open(f):
        if line.strip():
            r = json.loads(line)
            last[r.get("task_id")] = r                       # loaders keep the last record per task id
    lat, ins, models, n_calls, cached, s2_cost, s2_calls, errs, s1_ids, s1_err = [], [], collections.Counter(), 0, 0, 0.0, 0, [], set(), 0
    for r in last.values():
        if r.get("error"):
            errs.append(str(r["error"])[:200])
        s1_ids.add(str(((r.get("meta") or {}).get("s1") or {}).get("id")))
        for c in r.get("calls") or []:
            if c.get("component") == "s1":
                n_calls += 1
                models[str(c.get("model"))] += 1
                if (c.get("meta") or {}).get("error"):
                    s1_err += 1
                if c.get("cached"):
                    cached += 1
                    continue
                lat.append(float(c.get("latency_ms") or 0))
                ins.append(int(c.get("input_tokens") or 0))
            elif c.get("component") == "s2":
                s2_calls += 1
                s2_cost += float(c.get("cost_usd") or 0)
    summ["arms"][os.path.basename(f)[:-6]] = {
        "episodes": len(last), "episodes_with_error": len(errs), "errors": sorted(set(errs))[:10],
        "s1_calls": n_calls, "s1_calls_cached": cached, "s1_calls_with_error_meta": s1_err,
        "s1_latency_ms_per_decision": {"n": len(lat), "p50": pct(lat, .5), "p95": pct(lat, .95),
                                       "mean": round(sum(lat) / len(lat), 1) if lat else None,
                                       "max": round(max(lat), 1) if lat else None},
        "s1_input_tokens_per_request": {"p50": pct(ins, .5), "p95": pct(ins, .95), "max": max(ins) if ins else None},
        "s1_models": dict(models), "s1_backend_ids": sorted(s1_ids),
        "s2_calls": s2_calls, "api_spend_usd": round(s2_cost, 4)}
json.dump(summ, open(out, "w"), indent=1)
print(json.dumps(summ, indent=1))
EOF
  aws s3 sync "$RUNS/$DEVNAME" "$S3OUT/dev_run/$DEVNAME" --only-show-errors --region "$REGION"
  return $rc
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

stage_serve27b() {  # the 27B on :8091 (one 45 GiB GPU < 64 GiB: the Q8_0 GGUF via llama.cpp, A7.18 (g))
  SIZES=27B PYTHON="$(py311)" LLAMA_SERVER="$BASE/bin/llama-server" MANIFEST="$OUT/manifest_27B.json" \
    bash scripts/aws/serve_startlux.sh
}

stage_stopall() { STOP=1 bash scripts/aws/serve_startlux.sh; sudo docker ps -q | xargs -r sudo docker stop > /dev/null; }

stage_manifest() {  # one manifest of both invocations (the script writes one per invocation) + the server logs
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
    man["note"] = ("GPU smoke test on one L40S (development only): serve_startlux.sh run once per size (SIZES=4B, then "
                   "SIZES=27B), the 4B stopped before the 27B started; not the Stage-2 serving manifest")
    json.dump(man, open(os.path.join(out, "startlux_manifest.json"), "w"), indent=1)
    print("merged", [i["sizes"] for i in man["invocations"]])
EOF
  mkdir -p "$OUT/logs"
  for f in "$BASE"/logs/*.log; do [[ -f "$f" ]] && tail -n 200 "$f" > "$OUT/logs/$(basename "$f")"; done
}

stage_poweroff() { aws s3 sync "$OUT" "$S3OUT" --only-show-errors --region "$REGION"; sudo shutdown -h +1 "jev-startlux-smoke: done"; }

for s in $STAGES; do
  if ! declare -F "stage_$s" > /dev/null; then st "unknown stage $s"; continue; fi
  st "stage $s start"
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
