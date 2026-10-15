#!/usr/bin/env bash
# StartLux-Decision servers for Stage 2 (docs/PREREG_A7_STAGE2.md A7.18): installs the vendor's inference package at the
# pinned commit and the fast linear-attention kernels in their OWN environment (transformers >= 5.8; the harness venv
# is left alone), downloads the weights, checks the kernels, starts one server per size on its own GPU, smoke-tests it
# (a 3-question request and a 69-option choice through the server's own knockout) and writes the manifest of A7.18 (d)
# (weights revision, package commit, kernel versions, GPU, mode, health, smoke). Run on the GPU host:
#
#   bash scripts/aws/serve_startlux.sh                       # SIZES="4B 27B" (default); ports 4B 8090, 27B 8091
#   SIZES="0.8B 2B 9B" bash scripts/aws/serve_startlux.sh    # the size ladder (corpus replay only, A7.18 (e) 2)
#   DRY_RUN=1 bash scripts/aws/serve_startlux.sh             # print the plan and the commands only
#   STOP=1 bash scripts/aws/serve_startlux.sh                # stop the servers this script started
#
# A7.18 (g): a size runs from its original weights on one GPU with enough memory (27B: >= 64 GiB); the 27B otherwise from
# the vendor's Q8_0 GGUF via llama.cpp + startlux_decision.gguf_server on >= 40 GiB ("27B (Q8_0)"); otherwise, or when
# the kernel check or the server start fails, the size is NOT RUN and the manifest says why. Nothing else is substituted.
# Keys: HF_TOKEN is not needed for public weights; no API key is read or printed.
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO_ROOT=$(pwd)

: "${SIZES:=4B 27B}"
: "${BASE:=/opt/startlux}"
: "${VENV:=$BASE/venv}"
: "${PKG_DIR:=$BASE/startlux-decision}"
: "${MODELS_DIR:=$BASE/models}"
: "${LOG_DIR:=$BASE/logs}"
: "${MANIFEST:=results/stage2/startlux_manifest.json}"
: "${PYTHON:=python3}"
: "${DRY_RUN:=}"
: "${STOP:=}"
: "${LLAMA_SERVER:=llama-server}"            # llama.cpp built with CUDA (only for the 27B Q8_0 fallback)
: "${START_TIMEOUT_S:=900}"
# transformers as the vendor tested it (requirements.txt at the pinned commit: 5.8.1; A7.18 (b) asks >= 5.8). An unpinned
# install takes the newest release, and 5.18 no longer defines modeling_qwen3_5.is_fast_path_available, which
# startlux_decision.model.fast_kernels_active reads: the kernel check then fails with the kernels installed and the server
# refuses to start on CUDA. Enforced on an existing venv too.
: "${TRANSFORMERS_SPEC:=transformers==5.8.1}"
# accelerate: transformers' from_pretrained(device_map=...) (startlux_decision.model.load_model) requires it; the
# vendor's requirements.txt brings it only through peft (fine-tuning), which is not installed here
ENSURE_SPECS=("$TRANSFORMERS_SPEC" "accelerate>=1.0")                   # per size: weights load (llama.cpp: 28.6 GB from disk), CUDA graphs record
PLAN_PY="$REPO_ROOT/scripts/startlux_plan.py"
PKG_REPO=$($PYTHON -c "import sys; sys.path.insert(0, 'scripts'); import startlux_plan as P; print(P.PKG_REPO)")
PKG_COMMIT=$($PYTHON -c "import sys; sys.path.insert(0, 'scripts'); import startlux_plan as P; print(P.PKG_COMMIT)")

run() { if [ -n "$DRY_RUN" ]; then printf '%q ' "$@"; echo; else "$@"; fi; }
log() { echo "$(date -u +%H:%M:%S) $*" >&2; }
# start_bg LOG_STEM GPU CMD...: CMD in the background on GPU (all GPUs when empty), output to LOG_STEM.log, pid to
# LOG_STEM.pid; a dry run prints the command
start_bg() {
  local stem=$1 g=$2; shift 2
  if [ -n "$DRY_RUN" ]; then printf 'CUDA_VISIBLE_DEVICES=%s ' "$g"; printf '%q ' "$@"; echo "> $stem.log &"; return; fi
  if [ -n "$g" ]; then CUDA_VISIBLE_DEVICES="$g" nohup "$@" > "$stem.log" 2>&1 & else nohup "$@" > "$stem.log" 2>&1 & fi
  echo $! > "$stem.pid"
}

if [ -n "$STOP" ]; then
  for f in "$LOG_DIR"/*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null && log "stopped $(basename "$f" .pid)"; rm -f "$f"; done
  exit 0
fi

mkdir -p "$LOG_DIR" "$MODELS_DIR" "$(dirname "$MANIFEST")"
WORK=$(mktemp -d)
GPUS_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
[ -n "$GPUS_MIB" ] || { log "no GPU visible (nvidia-smi): every StartLux row is NOT RUN (A7.18 (g))"; GPUS_MIB=""; }
GPU_ARGS=(); for m in $GPUS_MIB; do GPU_ARGS+=(--gpus-mib "$m"); done
[ "${#GPU_ARGS[@]}" -gt 0 ] || GPU_ARGS=(--gpus-mib 0)
# shellcheck disable=SC2086
$PYTHON "$PLAN_PY" plan "${GPU_ARGS[@]}" --sizes $SIZES > "$WORK/plan.jsonl"
cat "$WORK/plan.jsonl" >&2
{ printf '{"_section": "host", "hostname": "%s", "gpus": "%s", "started": "%s", "harness_commit": "%s"}\n' \
    "$(hostname)" "$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | tr '\n' ';')" \
    "$(date -u +%FT%TZ)" "$(git rev-parse HEAD 2>/dev/null)"; } > "$WORK/host.jsonl"

# ---- the serving environment: package at the pinned commit + kernels (A7.18 (b), (d))
if [ ! -d "$PKG_DIR/.git" ]; then
  run git clone -q "$PKG_REPO" "$PKG_DIR"
fi
run git -C "$PKG_DIR" fetch -q --depth 1 origin "$PKG_COMMIT"
run git -C "$PKG_DIR" checkout -q "$PKG_COMMIT"
if [ ! -x "$VENV/bin/python" ]; then
  run "$PYTHON" -m venv "$VENV"
  run "$VENV/bin/pip" install -q --upgrade pip wheel
  run "$VENV/bin/pip" install -q "torch>=2.8" "${ENSURE_SPECS[@]}" "huggingface_hub>=0.34" "safetensors>=0.4" pyarrow ninja \
    packaging
fi
VPY="$VENV/bin/python"
if [ -n "$DRY_RUN" ] || ! "$VPY" - "${ENSURE_SPECS[@]}" > /dev/null 2>&1 <<'EOF'
import importlib.metadata as md, sys
from packaging.requirements import Requirement
for spec in sys.argv[1:]:
    r = Requirement(spec)
    if not r.specifier.contains(md.version(r.name), prereleases=True):    # PackageNotFoundError also exits non-zero
        sys.exit(1)
EOF
then
  run "$VENV/bin/pip" install -q "${ENSURE_SPECS[@]}"
fi
# the kernels are (re)installed whenever they do not import, not only with a new venv: a failed causal-conv1d build
# must not leave a venv that every later run skips. causal-conv1d ships prebuilt wheels for some torch versions only and
# otherwise compiles against a CUDA toolkit of torch's CUDA major version; a host without nvcc on PATH (the Deep
# Learning AMI keeps its toolkit as pip wheels under /opt/<env>/lib/python3*/site-packages/nvidia/cu<major>) gets the
# first such toolkit found, or CUDA_HOME as given. The toolkit used is recorded in the manifest.
if [ -n "$DRY_RUN" ] || ! "$VPY" -c "import torch, fla, causal_conv1d_cuda" > /dev/null 2>&1; then
  run "$VENV/bin/pip" install -q "flash-linear-attention>=0.5.2"
  if [ -z "${CUDA_HOME:-}" ] && ! command -v nvcc > /dev/null && [ -z "$DRY_RUN" ]; then
    tmaj=$("$VPY" -c "import torch; print((torch.version.cuda or '').split('.')[0])" 2>/dev/null)
    for c in /usr/local/cuda /usr/local/cuda-"$tmaj"* /opt/*/lib/python3*/site-packages/nvidia/cu"$tmaj" \
             "$VENV"/lib/python3*/site-packages/nvidia/cu"$tmaj"; do
      if [ -n "$tmaj" ] && [ -x "$c/bin/nvcc" ] && "$c/bin/nvcc" --version | grep -q "release $tmaj\."; then
        export CUDA_HOME="$c"; break
      fi
    done
    log "causal-conv1d build toolkit: ${CUDA_HOME:-none found for CUDA $tmaj (set CUDA_HOME)}"
  fi
  [ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH"
  MAX_JOBS=${MAX_JOBS:-$(nproc)} run "$VENV/bin/pip" install -q --no-build-isolation "causal-conv1d>=1.7"
fi
export PYTHONPATH="$PKG_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [ -z "$DRY_RUN" ]; then
  "$VPY" - > "$WORK/env.jsonl" <<'EOF'
import json, importlib.metadata as md
v = {}
for p in ("torch", "transformers", "accelerate", "flash-linear-attention", "causal-conv1d", "huggingface_hub", "triton"):
    try:
        v[p] = md.version(p)
    except md.PackageNotFoundError:
        v[p] = None
import os, subprocess
ch = os.environ.get("CUDA_HOME")
try:
    nv = subprocess.run([os.path.join(ch, "bin", "nvcc") if ch else "nvcc", "--version"], capture_output=True,
                        text=True).stdout.strip().splitlines()[-1]
except (OSError, IndexError):
    nv = None
print(json.dumps({"_section": "environment", **v, "cuda_home": ch, "nvcc": nv}))
EOF
else
  : > "$WORK/env.jsonl"
fi

# ---- one server per planned size
: > "$WORK/sizes.jsonl"
while read -r line <&3; do                    # fd 3: no command in the loop can eat the remaining plan lines
  size=$(echo "$line" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['size'])")
  mode=$(echo "$line" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['mode'])")
  gpu=$(echo "$line" | "$PYTHON" -c "import json,sys; g=json.load(sys.stdin)['gpu']; print('' if g is None else g)")
  port=$(echo "$line" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['port'])")
  repo=$(echo "$line" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['repo'])")
  name=$(echo "$line" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['local_name'])")
  dir="$MODELS_DIR/$name"
  status="$mode"; why=""; rev=""
  if [ "$mode" = not_run ]; then
    why=$(echo "$line" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['reason'])")
  else
    # the revision is read first and the download pinned to it, so the manifest's sha is the one on disk (A7.18 (d))
    # PIN_MANIFEST (optional): a committed manifest whose revision for this size is reused, so a later invocation serves
    # the same weights as the first one (A7.18 (d): a different revision before the A7.18 rows end is a deviation)
    if [ -n "${PIN_MANIFEST:-}" ] && [ -f "$PIN_MANIFEST" ]; then
      rev=$("$PYTHON" -c "import json,sys; m = json.load(open(sys.argv[1])); r = [s.get('weights_revision') for s in m.get('sizes', []) if s.get('size') == sys.argv[2] and s.get('weights_revision')]; print(r[0] if r else '')" \
            "$PIN_MANIFEST" "$size" 2>/dev/null)
    fi
    if [ -z "$DRY_RUN" ] && [ -z "$rev" ]; then
      rev=$("$VPY" -c "import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)" \
            "$repo" 2>/dev/null)
    fi
    run "$VPY" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2] or None, local_dir=sys.argv[3])" \
      "$repo" "$rev" "$dir" || { status=not_run; why="download failed"; rev=""; }
  fi
  if [ "$status" = original ]; then
    if [ -z "$DRY_RUN" ]; then
      chk=$(CUDA_VISIBLE_DEVICES="$gpu" "$VPY" -m startlux_decision.check "$dir" 2>&1 | tail -3)
      echo "$chk" | grep -q "fast kernels: active" || { status=not_run; why="kernel check: $(echo "$chk" | tr '\n' ' ')"; }
    else
      run "$VPY" -m startlux_decision.check "$dir"
    fi
    if [ "$status" = original ]; then
      start_bg "$LOG_DIR/server_$size" "$gpu" "$VPY" -m startlux_decision.server --model "$dir" --host 127.0.0.1 \
        --port "$port" --name "StartLux-Decision-$size"
    fi
  elif [ "$status" = gguf_q8_0 ]; then
    if ! command -v "$LLAMA_SERVER" > /dev/null && [ -z "$DRY_RUN" ]; then
      status=not_run; why="llama-server (llama.cpp with CUDA) not installed: 27B (Q8_0) not run"
    else
      lport=$((port + 100))
      start_bg "$LOG_DIR/llama_$size" "$gpu" "$LLAMA_SERVER" -m "$dir/$name.gguf" -ngl 99 -c "${LLAMA_CTX:-32768}" --parallel 1 \
        --port "$lport"   # one slot with the whole context: the vendor flags (-c 16384 --parallel 4) cap a prompt at 4,096 tokens and JevGym states reach ~13k (smoke test 2026-10-03); the server answers one request at a time anyway
      start_bg "$LOG_DIR/server_$size" "" "$VPY" -m startlux_decision.gguf_server --model-dir "$dir" \
        --llama "http://127.0.0.1:$lport" --port "$port" --name "StartLux-Decision-$size-Q8_0"
    fi
  fi
  health=""; smoke=""
  if [ -z "$DRY_RUN" ] && [ "$status" != not_run ]; then
    # gguf_server answers /health at once, before llama-server has loaded the weights (llama-server: 503 while
    # loading), so the Q8_0 path waits for llama-server's own /health first; either process exiting ends the wait
    for _ in $(seq 1 $((START_TIMEOUT_S / 5))); do
      for p in server llama; do
        [ -f "$LOG_DIR/${p}_$size.pid" ] && ! kill -0 "$(cat "$LOG_DIR/${p}_$size.pid")" 2>/dev/null && break 2  # exited
      done
      if [ "$status" = gguf_q8_0 ] && ! curl -sf "http://127.0.0.1:$lport/health" > /dev/null 2>&1; then sleep 5; continue; fi
      health=$(curl -sf "http://127.0.0.1:$port/health" 2>/dev/null) && [ -n "$health" ] && break
      health=""
      sleep 5
    done
    if [ -z "$health" ]; then
      tails="$(tail -3 "$LOG_DIR/server_$size.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
      [ "$status" = gguf_q8_0 ] && tails="llama-server: $(tail -3 "$LOG_DIR/llama_$size.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300) | $tails"
      status=not_run; why="server did not start within ${START_TIMEOUT_S}s: $tails"
    else
      smoke=$("$PYTHON" "$PLAN_PY" smoke --port "$port") || { status=not_run; why="smoke test failed"; }
    fi
  fi
  "$PYTHON" - "$line" "$status" "$why" "$rev" "$health" "$smoke" >> "$WORK/sizes.jsonl" <<'EOF'
import json, sys
rec = json.loads(sys.argv[1])
def j(s):
    try:
        return json.loads(s) if s else None
    except ValueError:
        return s
rec.update({"_section": "sizes", "planned_mode": rec.pop("mode"), "status": sys.argv[2], "why": sys.argv[3] or None,
            "weights_revision": sys.argv[4] or None, "health": j(sys.argv[5]), "smoke": j(sys.argv[6])})
print(json.dumps(rec))
EOF
  log "$size: $status ${why:+($why)}"
done 3< "$WORK/plan.jsonl"

if [ -z "$DRY_RUN" ]; then
  "$PYTHON" "$PLAN_PY" manifest --out "$MANIFEST" --entry "$WORK/host.jsonl" --entry "$WORK/env.jsonl" --entry "$WORK/sizes.jsonl"
  log "manifest -> $MANIFEST (commit it before the first StartLux evaluation request, A7.18 (d))"
else
  cat "$WORK/sizes.jsonl"
fi
rm -rf "$WORK"
