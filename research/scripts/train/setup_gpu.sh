#!/usr/bin/env bash
# GPU host setup + verification for the training programme (docs/TRAINING_AGENT_BRIEF.md §2).
# Idempotent: re-running skips what is already installed and re-runs every check. Ends with
# runs/gpu_setup/env_report.json (paste it into the report) and a non-zero exit code if a check failed.
#
#   1. host     nvidia-smi (GPUs, memory, driver, highest CUDA the driver supports), disk, RAM
#   2. venv     $VENV via uv (uv is installed with pip if missing)
#   3. torch    torch==$TORCH_VERSION from $TORCH_INDEX, chosen from the DRIVER (not the toolkit):
#                 driver CUDA >= 13.0 -> PyPI default wheel (CUDA 13.0 runtime; needs driver >= 580)
#                 12.8 / 12.9 -> https://download.pytorch.org/whl/cu128      12.6 / 12.7 -> .../cu126
#               skipped when the installed torch already matches and sees the GPUs
#   4. deps     uv pip install -e ".[all,train]" -c configs/train/constraints-gpu.txt (pinned transformers / trl / peft /
#               accelerate / datasets: the versions the code and tests were verified with)
#   5. verify   scripts/train/_verify_gpu.py: versions vs pins, CUDA + bf16, TRL GRPOConfig fields, single-token option
#               codes, Qwen3 chat template, ONE forward/backward of the decision loss with LoRA on $CHECK_MODEL
#   6. tests    the offline pytest suite (RUN_TESTS=1)
#   7. vLLM     WITH_VLLM=1: a SEPARATE venv $VLLM_VENV (vLLM pins its own torch; never install it into $VENV), used as
#               the self-hosted System-2 executor (EXECUTOR=vllm) and for the Qwen3 logprob baselines
#
# Variables (defaults in brackets):
#   VENV [.venv]   PYTHON [3.11]   TORCH_VERSION [2.14.1]   TORCH_INDEX ['' = auto, see 3; 'pypi' = PyPI default]
#   CONSTRAINTS [configs/train/constraints-gpu.txt]   CHECK_MODEL [Qwen/Qwen3-0.6B]   SKIP_INSTALL [0] (checks only)
#   RUN_TESTS [1]   CHECK_HF [1] (gated xLAM access, warning only)   WITH_VLLM [0]   VLLM_VENV [.venv-vllm]
#   VLLM_VERSION ['' = latest]   ALLOW_CPU [0] (1 = run the checks on a CPU-only host, e.g. with a tiny CHECK_MODEL)
#   MIN_DISK_GB [150]   DRY_RUN [0]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${VENV:=.venv}"
: "${PYTHON:=3.11}"
: "${TORCH_VERSION:=2.14.1}"
: "${TORCH_INDEX:=}"
: "${CONSTRAINTS:=configs/train/constraints-gpu.txt}"
: "${CHECK_MODEL:=Qwen/Qwen3-0.6B}"
: "${SKIP_INSTALL:=0}"
: "${RUN_TESTS:=1}"
: "${CHECK_HF:=1}"
: "${WITH_VLLM:=0}"
: "${VLLM_VENV:=.venv-vllm}"
: "${VLLM_VERSION:=}"
: "${ALLOW_CPU:=0}"
: "${MIN_DISK_GB:=150}"
: "${DRY_RUN:=0}"
OUT_DIR=runs/gpu_setup
mkdir -p "$OUT_DIR"

log() { printf '[setup_gpu] %s\n' "$*" >&2; }
run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then
    "$@"
  fi
}

# ---------------------------------------------------------------- 1. host
driver_cuda=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee "$OUT_DIR/gpus.csv" >&2
  driver_cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9]*\).*/\1/p' | head -n1)
  log "driver supports CUDA ${driver_cuda:-?}; $(nvidia-smi -L | wc -l) GPU(s)"
  nvidia-smi topo -m > "$OUT_DIR/topo.txt" 2>&1 || true
elif [[ "$ALLOW_CPU" == 1 ]]; then
  log "no nvidia-smi: CPU-only checks (ALLOW_CPU=1)"
else
  log "nvidia-smi not found: no NVIDIA driver on this host (set ALLOW_CPU=1 for a CPU-only check)"
  exit 1
fi
free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
log "free disk here: ${free_gb} GB (HF cache: ${HF_HOME:-~/.cache/huggingface}); RAM: $(free -g 2>/dev/null | awk '/Mem:/ {print $2}') GB"
if (( free_gb < MIN_DISK_GB )); then
  log "WARNING: < ${MIN_DISK_GB} GB free. Qwen3-4B/8B weights (8/16 GB), full-FT checkpoints with optimizer state and" \
      "vLLM need room; point HF_HOME / CKPT at a bigger volume"
fi

# ---------------------------------------------------------------- 2. venv
if [[ "$SKIP_INSTALL" != 1 ]]; then
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    # PEP 668 distros (Ubuntu >= 23.04, e.g. the Ubuntu 26.04 Deep Learning AMI) refuse a plain `pip install --user`;
    # a --user install of uv does not touch system packages, so --break-system-packages is safe here.
    run python3 -m pip install --user -q uv || run python3 -m pip install --user -q --break-system-packages uv \
      || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
  fi
  if [[ ! -x "$VENV/bin/python" ]]; then
    run uv venv -p "$PYTHON" "$VENV"
  fi
fi
if [[ -f "$VENV/bin/activate" ]]; then
  set +u
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  set -u
fi
PY="${VENV}/bin/python"
[[ -x "$PY" ]] || PY=python

# ---------------------------------------------------------------- 3. torch
torch_index() {
  if [[ -n "$TORCH_INDEX" ]]; then
    [[ "$TORCH_INDEX" == pypi ]] && echo "" || echo "$TORCH_INDEX"
    return
  fi
  local major=${driver_cuda%%.*} minor=${driver_cuda#*.}
  if [[ -z "$driver_cuda" ]] || (( major >= 13 )); then
    echo ""                                             # PyPI default build (CUDA 13.0 runtime)
  elif (( major == 12 && minor >= 8 )); then
    echo "https://download.pytorch.org/whl/cu128"
  elif (( major == 12 && minor >= 6 )); then
    echo "https://download.pytorch.org/whl/cu126"
  else
    log "driver CUDA $driver_cuda is older than 12.6: upgrade the driver or set TORCH_INDEX / TORCH_VERSION by hand"
    exit 1
  fi
}
torch_ok() {
  "$PY" - "$TORCH_VERSION" "$ALLOW_CPU" <<'EOF' 2>/dev/null
import sys
import torch
ok = torch.__version__.split("+")[0] == sys.argv[1] and (torch.cuda.is_available() or sys.argv[2] == "1")
sys.exit(0 if ok else 1)
EOF
}
if [[ "$SKIP_INSTALL" != 1 ]]; then
  if torch_ok; then
    log "torch $TORCH_VERSION already installed and working"
  else
    idx=$(torch_index)
    if [[ -n "$idx" ]]; then
      run uv pip install --python "$PY" "torch==$TORCH_VERSION" --index-url "$idx" || {
        log "torch==$TORCH_VERSION is not on $idx. List what is there (pip index versions torch --index-url $idx)," \
            "then re-run with TORCH_VERSION=<that version> and update configs/train/constraints-gpu.txt to match"
        exit 1
      }
    else
      run uv pip install --python "$PY" "torch==$TORCH_VERSION"
    fi
  fi

  # ---------------------------------------------------------------- 4. deps (pinned)
  # torch is excluded from the constraints here so a CUDA-index build (2.14.1+cu128) is never swapped for PyPI's
  grep -v '^torch==' "$CONSTRAINTS" > "$OUT_DIR/constraints.no-torch.txt"
  run uv pip install --python "$PY" -e ".[all,train]" -c "$OUT_DIR/constraints.no-torch.txt"
fi

# ---------------------------------------------------------------- 5. verify
vargs=(--model "$CHECK_MODEL" --out "$OUT_DIR/env_report.json" --constraints "$CONSTRAINTS")
[[ "$ALLOW_CPU" == 1 ]] && vargs+=(--allow-cpu)
[[ "$CHECK_HF" == 1 ]] && vargs+=(--check-hf)
if [[ -z "${HF_TOKEN:-}" ]] && ! "$PY" -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 1)" 2>/dev/null; then
  log "WARNING: no HF token (HF_TOKEN or \`hf auth login\`): Qwen3 downloads work anonymously, gated xLAM does not"
fi
verify_rc=0
run "$PY" "$ROOT/scripts/train/_verify_gpu.py" "${vargs[@]}" || verify_rc=$?

# ---------------------------------------------------------------- 6. tests
tests_rc=0
if [[ "$RUN_TESTS" == 1 ]]; then
  run "$PY" -m pytest -q -p no:warnings -x tests > "$OUT_DIR/pytest.log" 2>&1 || tests_rc=$?
  tail -n 3 "$OUT_DIR/pytest.log" >&2 || true
fi

# ---------------------------------------------------------------- 7. vLLM (separate venv)
vllm_rc=0
if [[ "$WITH_VLLM" == 1 ]]; then
  if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
    run uv venv -p "$PYTHON" "$VLLM_VENV"
  fi
  run uv pip install --python "$VLLM_VENV/bin/python" "vllm${VLLM_VERSION:+==$VLLM_VERSION}" || vllm_rc=$?
  if (( vllm_rc == 0 )); then
    run "$VLLM_VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
      || vllm_rc=$?
    log "serve an executor: $VLLM_VENV/bin/vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 32768 &  then EXECUTOR=vllm" \
        "VLLM_BASE_URL=http://localhost:8000/v1 (70_grpo_online.sh / mc_rollouts via an experiment llm: {provider: vllm})"
  fi
fi

log "env report: $OUT_DIR/env_report.json   verify rc=$verify_rc   tests rc=$tests_rc   vllm rc=$vllm_rc"
if (( verify_rc != 0 || tests_rc != 0 || vllm_rc != 0 )); then
  log "setup NOT verified: see the failing check above and docs/TRAINING_AGENT_BRIEF.md §8"
  exit 1
fi
log "setup verified. Next: bash scripts/train/smoke_gpu.sh"
