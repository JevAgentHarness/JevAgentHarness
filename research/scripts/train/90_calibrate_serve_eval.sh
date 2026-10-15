#!/usr/bin/env bash
# T8 -- calibrate a trained decider, serve it behind the Jev wire API, and evaluate it exactly like every other
# System-1 backend:
#   1. training.calibrate      temperature per (type x option-count bucket) on DEV (held-out TASKS) -> $CAL
#   2. training.serve_jev      POST /v1/systemone on 127.0.0.1:$PORT (background; killed on exit)
#   3. jevbench ping           smoke check through the real JevBackend client
#   4. jevbench decide         E1 static decisions (BFCL v4 + When2Call) with backends = {$NAME: served model}
#   5. jevbench run            E2 JevGym arms (llm_only + every arm on the default System-1) with s1 = served model
# The backend spec used everywhere:
#   {"type":"jev","name":"$NAME","base_url":"http://localhost:$PORT","model":"$NAME","api_key":""}
#
# Variables (defaults in brackets):
#   MODEL [Qwen/Qwen3-4B]   ADAPTER [$CKPT/r4] ('' = zero-shot base model = recipe R0)
#   NAME  served model id [ours-<basename of ADAPTER>]   PORT [8081]
#   DEV   [$DATA/dev.jsonl]   CAL [$ADAPTER/calibration.json | $EVAL_DIR/...]   DO_CAL [1]   MIN_BUCKET [200]
#   DECIDE_CFG [configs/decision/bfcl_when2call.yaml]   DO_DECIDE [1]   MAX_ITEMS ['']   KEEP_BACKENDS [0]
#   EXP [configs/experiments/jevgym_tool_select.yaml]   DO_RUN [1]   MAX_TASKS ['']
#   RUN_SET   ';'-separated `jevbench run -o` overrides, e.g. 'suite.n=50;workers=4'           ['']
#   RUNS_DIR [runs]   EVAL_DIR (derived configs, server log) [$ADAPTER/eval or runs/eval_$NAME]
#   SERVE_URL  use an already-running server instead of starting one ('' = start)          ['']
#   SERVER_TIMEOUT seconds to wait for /health [600]   DRY_RUN
# Note: the E2 executor (System-2) in EXP is a real API (Gemini) -- budget it like any E2 arm (docs/ROADMAP.md §5).
source "$(dirname "$0")/_common.sh"

: "${ADAPTER=$CKPT/r4}"
: "${NAME:=ours-$(basename "${ADAPTER:-base}")}"
: "${PORT:=8081}"
: "${DEV:=$DATA/dev.jsonl}"
: "${DO_CAL:=1}"
: "${MIN_BUCKET:=200}"
: "${DECIDE_CFG:=configs/decision/bfcl_when2call.yaml}"
: "${DO_DECIDE:=1}"
: "${MAX_ITEMS:=}"
: "${KEEP_BACKENDS:=0}"
: "${EXP:=configs/experiments/jevgym_tool_select.yaml}"
: "${DO_RUN:=1}"
: "${MAX_TASKS:=}"
: "${RUN_SET:=}"
: "${RUNS_DIR:=runs}"
: "${EVAL_DIR:=${ADAPTER:+$ADAPTER/eval}}"
: "${EVAL_DIR:=runs/eval_$NAME}"
: "${CAL:=${ADAPTER:+$ADAPTER/calibration.json}}"
: "${CAL:=$EVAL_DIR/calibration.json}"
: "${SERVE_URL:=}"
: "${SERVER_TIMEOUT:=600}"

jb=("$PY" -m jevharness.bench.cli)   # == `jevbench`
base_url="${SERVE_URL:-http://localhost:$PORT}"
spec=$(printf '{"type": "jev", "name": "%s", "base_url": "%s", "model": "%s", "api_key": ""}' "$NAME" "$base_url" "$NAME")
adapter_args=()
if [[ -n "$ADAPTER" ]]; then
  need "$ADAPTER"
  adapter_args=(--adapter "$ADAPTER")
fi
run mkdir -p "$EVAL_DIR"

# 1. calibration
cal_args=()
if [[ "$DO_CAL" == 1 ]]; then
  need "$DEV"
  run "$PY" -m jevharness.training.calibrate --model "$MODEL" ${adapter_args[@]+"${adapter_args[@]}"} --dev "$DEV" \
    --out "$CAL" --min-bucket "$MIN_BUCKET"
fi
if [[ "$DO_CAL" == 1 || -e "$CAL" ]]; then
  cal_args=(--calibration "$CAL")
fi

# 2. server
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
if [[ -z "$SERVE_URL" ]]; then
  log "starting serve_jev on :$PORT (log: $EVAL_DIR/server.log)"
  printf '+'
  printf ' %q' "$PY" -m jevharness.training.serve_jev --model "$MODEL" ${adapter_args[@]+"${adapter_args[@]}"} \
    ${cal_args[@]+"${cal_args[@]}"} --name "$NAME" --port "$PORT"
  printf ' &\n'
  if [[ "$DRY_RUN" != 1 ]]; then
    "$PY" -m jevharness.training.serve_jev --model "$MODEL" ${adapter_args[@]+"${adapter_args[@]}"} \
      ${cal_args[@]+"${cal_args[@]}"} --name "$NAME" --port "$PORT" > "$EVAL_DIR/server.log" 2>&1 &
    server_pid=$!
    "$PY" - "$base_url" "$SERVER_TIMEOUT" "$server_pid" <<'EOF'
import os, sys, time
import httpx
url, timeout, pid = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
t0 = time.time()
while time.time() - t0 < timeout:
    try:
        if httpx.get(url + "/health", timeout=2).status_code == 200:
            print(f"server up after {time.time() - t0:.0f}s"); sys.exit(0)
    except httpx.HTTPError:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        sys.exit("server process exited; see server.log")
    time.sleep(2)
sys.exit(f"server not healthy after {timeout:.0f}s")
EOF
  fi
fi

# 3. ping
run "${jb[@]}" ping --s1 "$spec"

# 4. E1 static decisions
if [[ "$DO_DECIDE" == 1 ]]; then
  need "$DECIDE_CFG"
  dargs=()
  if [[ "$KEEP_BACKENDS" == 1 ]]; then
    dargs+=(--keep-backends)
  fi
  run "$PY" "$ROOT/scripts/train/_derive_eval_configs.py" decide "$DECIDE_CFG" "$EVAL_DIR/decide.yaml" \
    --spec "$spec" --name "$NAME" ${dargs[@]+"${dargs[@]}"}
  margs=()
  if [[ -n "$MAX_ITEMS" ]]; then
    margs=(--max-items "$MAX_ITEMS")
  fi
  run "${jb[@]}" decide "$EVAL_DIR/decide.yaml" --out "$RUNS_DIR/decision" ${margs[@]+"${margs[@]}"}
fi

# 5. E2 JevGym arms
if [[ "$DO_RUN" == 1 ]]; then
  need "$EXP"
  run "$PY" "$ROOT/scripts/train/_derive_eval_configs.py" run "$EXP" "$EVAL_DIR/run.yaml" --spec "$spec" --name "$NAME"
  rargs=()
  if [[ -n "$MAX_TASKS" ]]; then
    rargs+=(--max-tasks "$MAX_TASKS")
  fi
  overrides=()
  while IFS= read -r kv; do
    overrides+=("$kv")
  done < <(split_list "$RUN_SET" ";")
  if (( ${#overrides[@]} > 0 )); then
    rargs+=(-o "${overrides[@]}")
  fi
  run "${jb[@]}" run "$EVAL_DIR/run.yaml" --out "$RUNS_DIR" ${rargs[@]+"${rargs[@]}"}
  log "report: jevbench report $RUNS_DIR/<name printed above>; scorecard: python -m jevharness.bench.scorecard ..."
fi
log "done ($NAME)"
