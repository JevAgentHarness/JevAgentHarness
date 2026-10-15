# shellcheck shell=bash
# Box-side stages of GPU round 6 (JevAgentRL-v2, the Miles track), sourced by scripts/aws/gpu_round6_miles.sh `box`.
# Variables / helpers from the caller: OUT EVAL CK MODELS D0 D1 W B S3OUT S3CK REGION IMAGE DOCKER, miles() st() sync_ck().
# Rules: results/gpu_round6_miles/TRAINING_MANIFEST.md. Seeds: training 60000-63999 (OPD), 70000-72999 (GRPO);
# development 1100-1149 (E2 dev slice, v1's), 2000-2199 (stress instrument); smoke rollouts use the simulated
# executor on the GRPO pool's first tasks. Never calibration, evaluation or Stage-2 seeds.

GATE_FILES=("$D0/dev.jsonl" "$D0/dev_when2call.jsonl.gz")
V1_FILES=("$D1/dev.jsonl.gz" "$D1/dev_recovery.jsonl.gz" "$D1/dev_knockout.jsonl.gz")
EVAL_BATCH=${EVAL_BATCH:-16}
# Parallel invocations on the 8 GPUs (outside the training stages): EVAL_GPU (decision-level evaluation), SERVE_GPU +
# PORT (the served checkpoint of a live slice; one server container per port), REF_GPU (Intern reference).
EVAL_GPU=${EVAL_GPU:-0}
SERVE_GPU=${SERVE_GPU:-0}
PORT=${PORT:-8082}
SERVER_PID=""
mkdir -p "$OUT/stress" "$OUT/e2dev"

base_path() { python -c "import json; print(json.load(open('$OUT/base.json'))['path'])"; }

# Evaluation-side commands (eval_decisions, serve_jev, jevbench) run in the Miles image by default: it has the
# flash-linear-attention / causal-conv1d kernels that Qwen3.5-family models (StartLux-Decision, Intern-Decision) need
# for a usable speed; the project venv lacks causal-conv1d. EVAL_ENV=venv runs them in the venv instead. The freeze
# stage checks that the venv's serve_jev reproduces the container's option log-probabilities (regression corpus).
EVAL_ENV=${EVAL_ENV:-container}
evx() {  # evx "COMMAND"
  if [[ "$EVAL_ENV" == container ]]; then miles "$1"; else bash -c "$1"; fi
}

evaluate() {  # evaluate DIR TAG: gate set (+ flip rate) and the v1 in-harness files, raw temperature
  local ck=$1 tag=$2
  if [[ ! -e "$EVAL/eval_${tag}_gate.json" ]]; then
    evx "CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m jevharness.training.eval_decisions --model $ck --dev ${GATE_FILES[*]} \
      --n-perm 1 --max-len 4096 --batch-size $EVAL_BATCH --out $EVAL/eval_${tag}_gate.json" || return 1
  fi
  if [[ ! -e "$EVAL/eval_${tag}_v1.json" ]]; then
    evx "CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m jevharness.training.eval_decisions --model $ck --dev ${V1_FILES[*]} \
      --n-perm 0 --max-len 4096 --batch-size $EVAL_BATCH --out $EVAL/eval_${tag}_v1.json" || return 1
  fi
}
gonogo_of() {  # gonogo_of TAG [BASELINE_TAG]: G1-G5 with the actual Intern-Decision-4B v1-dev reference (round 5's file)
  local tag=$1 base=${2:-}
  local args=(--eval "$EVAL/eval_${tag}_gate.json" "$EVAL/eval_${tag}_v1.json" --jev-v1 "$D1/dev_teacher.summary.json"
              --intern-v1 results/gpu_round5/eval/gate_intern4b_v1dev.json --out "$EVAL/gonogo_${tag}.json")
  if [[ -n "$base" ]]; then args+=(--baseline "$EVAL/eval_${base}_gate.json" "$EVAL/eval_${base}_v1.json"); fi
  python -m jevharness.training.data_v1 gonogo "${args[@]}" > /dev/null
}

# ----------------------------------------------------------------------------------------------- serving + live slices
serve_start() {  # serve_start DIR NAME: a checkpoint behind the Jev wire API on :$PORT (training/serve_jev.py, $SERVE_GPU)
  if [[ "$EVAL_ENV" == container ]]; then
    $DOCKER rm -f "jev-serve-$PORT" > /dev/null 2>&1
    $DOCKER run -d --name "jev-serve-$PORT" --gpus all --ipc=host --network=host --shm-size=16g \
      -v "$W:$W" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -w "$W/JevRL" \
      -e PYTHONPATH="$W/JevRL/src" -e CUDA_VISIBLE_DEVICES="$SERVE_GPU" -e PYTHONUNBUFFERED=1 "$IMAGE" \
      bash -lc "python -m jevharness.training.serve_jev --model $1 --name $2 --port $PORT" > /dev/null || return 1
    SERVER_PID=container
  else
    CUDA_VISIBLE_DEVICES="$SERVE_GPU" python -m jevharness.training.serve_jev --model "$1" --name "$2" --port "$PORT" \
      > "$OUT/serve_$2.log" 2>&1 &
    SERVER_PID=$!
  fi
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then return 0; fi
    if [[ "$SERVER_PID" == container ]]; then
      $DOCKER ps -q -f name="jev-serve-$PORT" | grep -q . || {
        $DOCKER logs "jev-serve-$PORT" > "$OUT/serve_$2.log" 2>&1; echo "server died"; return 1; }
    elif ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died"; return 1; fi
    sleep 5
  done
  return 1
}
serve_stop() {
  if [[ "$SERVER_PID" == container ]]; then
    $DOCKER logs "jev-serve-$PORT" > "$OUT/serve_last_$PORT.log" 2>&1
    $DOCKER rm -f "jev-serve-$PORT" > /dev/null 2>&1
  elif [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; fi
  SERVER_PID=""
}
live_run() {  # live_run CONFIG RUNDIR NAME "PREFIXES" [SET...]: a subset of an experiment's arms as its own run
  local cfg=$1 rd=$2 name=$3 prefixes=$4
  shift 4
  # the API cap (manifest §3): no live run starts unless the ledgers leave room for one more (~$11)
  python scripts/train/r6_spend.py --dir "$OUT" --ck "$CK" --cap "${API_CAP:-125}" --reserve 11 --check \
    > "$OUT/spend_before_$name.txt" || { echo "API cap reached: $name not run"; return 1; }
  mkdir -p "$rd"
  # shellcheck disable=SC2086
  python scripts/train/r5_select_arms.py "$cfg" "$rd/cfg_$name.yaml" --prefix $prefixes --set "name=$name" "$@" || return 1
  evx "python -m jevharness.bench.cli run $rd/cfg_$name.yaml --out $rd"
  local rc=$?
  python -m jevharness.bench.cli report "$rd/$name" > "$rd/$name/report.md" 2>/dev/null
  aws s3 sync "$rd/$name" "$S3OUT/live_runs/$name" --only-show-errors --region "$REGION"
  return $rc
}
STRESS_CFG=configs/experiments/jevgym_stress_dev_miles.yaml
SRD=runs/gpu_round6_stress_s2000
stress_ckpt() {  # stress_ckpt TAG DIR: the stress instrument (manifest §4) on one checkpoint
  local tag=$1 dir=$2 name="jevagentrl-v2-$1"
  serve_start "$dir" "$name" || { serve_stop; return 1; }
  live_run "$STRESS_CFG" "$SRD" "stress_$tag" "ours_" "s1.name=$name" "s1.model=$name" "s1.base_url=http://127.0.0.1:$PORT"
  local rc=$?
  serve_stop
  python scripts/train/r5_summary.py "$SRD/stress_$tag" --out "$OUT/stress/stress_${tag}_summary" > /dev/null
  return $rc
}
E2CFG=configs/experiments/jevgym_dev_rl.yaml
E2RD=runs/gpu_round6_e2dev_s1100
e2dev_ckpt() {  # e2dev_ckpt TAG DIR "ARMS": v1's E2 development slice (seeds 1100-1149), descriptive comparison with v1
  local tag=$1 dir=$2 arms=$3 name="jevagentrl-v2-$1"
  serve_start "$dir" "$name" || { serve_stop; return 1; }
  # shellcheck disable=SC2086
  live_run "$E2CFG" "$E2RD" "e2dev_v2_$tag" "$arms" "s1.name=$name" "s1.model=$name" "s1.base_url=http://127.0.0.1:$PORT" \
    "workers=8"
  local rc=$?
  serve_stop
  python scripts/train/r5_summary.py "$E2RD/e2dev_v2_$tag" --out "$OUT/e2dev/e2dev_v2_${tag}_summary" > /dev/null
  return $rc
}

# ----------------------------------------------------------------------------------------------- sequencing / parallel
par() {  # par "STAGE[:VAR=VAL[+VAR=VAL...]]" ...: stages in parallel subshells (own GPU / port each); 1 if any failed
  local spec s vars p rc=0 pids=()
  for spec in "$@"; do
    s=${spec%%:*}
    vars=""
    if [[ "$spec" == *:* ]]; then vars=${spec#*:}; fi
    (
      for kv in ${vars//+/ }; do export "${kv?}"; done
      st "stage $s start (parallel; ${vars:-defaults})"
      t0=$(date +%s)
      ( set -o pipefail; "stage_$s" ) > "$OUT/$s.log" 2>&1
      r=$?
      st "stage $s exit=$r ($(( ($(date +%s) - t0) / 60 )) min; parallel)"
      exit $r
    ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
stage_await() {  # AWAIT=STAGE: wait (up to 12 h) until STAGE's latest status line is its exit; fail unless exit=0
  local n last
  for n in $(seq 1 1440); do
    last=$(tac "$OUT/status.txt" | grep -m1 -E "stage ${AWAIT:?} (start|exit=)")
    case "$last" in
      *"exit=0 "*) return 0 ;;
      *exit=*) echo "$AWAIT failed: $last"; return 1 ;;
    esac
    sleep 30
  done
  return 1
}
# after SFT: gates, stress, the zero-shot row and the Intern stress reference side by side (GPUs 0-3)
stage_post_sft() {
  par "eval_sft:EVAL_GPU=0" "stress_sft:SERVE_GPU=1+PORT=8083" "stress_refs_intern:REF_GPU=2" "zeroshot_row:EVAL_GPU=3"
  return 0   # failures are in the status lines; OPD does not depend on these read-outs
}

# ----------------------------------------------------------------------------------------------- host diagnostics
stage_nccl_bench() {  # all-gather / reduce-scatter bus bandwidth over the 8 GPUs under NCCL transport settings
  nvidia-smi topo -m > "$OUT/nvidia_topo.txt" 2>&1
  local v
  for v in "" "NCCL_P2P_LEVEL=SYS" "NCCL_P2P_DISABLE=1" "NCCL_P2P_LEVEL=SYS NCCL_PROTO=Simple"; do
    echo "=== env: ${v:-default}"
    miles "$v all_gather_perf -b 256M -e 1G -f 2 -g 8 -n 5 -w 2 2>&1 | tail -6; $v reduce_scatter_perf -b 256M -e 1G -f 2 -g 8 -n 5 -w 2 2>&1 | tail -6"
  done > "$OUT/nccl_bench.txt" 2>&1
}

# ----------------------------------------------------------------------------------------------- base rule
stage_base_zero() {  # S4 inputs: zero-shot v1-dev accuracy of both bases through serve_jev's prompt (raw temperature)
  for key in qwen startlux; do
    local m
    m=$([[ $key == qwen ]] && echo "$MODELS/Qwen3-4B" || echo "$MODELS/StartLux-Decision-4B")
    if [[ ! -e "$EVAL/eval_zeroshot_${key}_v1.json" ]]; then
      evx "python -m jevharness.training.eval_decisions --model $m --dev $D1/dev.jsonl.gz --n-perm 0 --max-len 4096 \
        --batch-size $EVAL_BATCH --out $EVAL/eval_zeroshot_${key}_v1.json" || return 1
    fi
  done
}
smoke_cfg() {  # smoke_cfg MODEL_DIR NAME [MAX_TOK] -> a short SFT config of the given base (written under $OUT)
  python - "$1" "$2" "$OUT/smoke_sft_$2.yaml" "${3:-}" <<'EOF'
import sys, yaml
src, name, dst, max_tok = sys.argv[1:]
c = yaml.safe_load(open("configs/train/miles_sft_v2.yaml"))
c.update(name=f"smoke_sft_{name}", base=src, init=None, out=f"checkpoints/r6/smoke_{name}")
c["miles"]["save_interval"] = 0
if max_tok:
    c["miles"]["max_tokens_per_gpu"] = int(max_tok)
yaml.safe_dump(c, open(dst, "w"))
EOF
}
stage_speed_test() {  # SPEED_MODEL (dir name) SPEED_MAX_TOK (comma list) SPEED_ENV: 3 SFT rollouts each, no save -> speed_*.txt
  local name=${SPEED_MODEL:-Qwen3-4B} tok tag rc=0
  for tok in $(echo "${SPEED_MAX_TOK:-3072}" | tr ',' ' '); do
    tag="${name}_${tok}_$(echo "${SPEED_ENV:-default}" | tr ' =' '__')"
    smoke_cfg "$MODELS/$name" "speed_$tag" "$tok" || return 1
    miles "${SPEED_ENV:-} JEV_MILES_NO_SAVE=1 JEV_MILES_NUM_ROLLOUT=3 python -m jevharness.training.miles_port.launch run \
      --config $OUT/smoke_sft_speed_$tag.yaml" > "$OUT/speed_$tag.log" 2>&1 || rc=1
    grep -E "perf [0-9]+: \{'perf/update|step [0-9]+: \{'train/loss|OutOfMemory|CUDA out of memory" "$OUT/speed_$tag.log" \
      | sed -E 's/^.*(perf [0-9]+|step [0-9]+)/\1/' > "$OUT/speed_$tag.txt"
    rm -rf "checkpoints/r6/smoke_speed_$tag"
  done
  return $rc
}
stage_smoke_train() {  # S1: 4 Miles FSDP SFT steps per base, custom loss, finite loss; then the strict HF export
  local rc=0
  for name in Qwen3-4B StartLux-Decision-4B; do
    smoke_cfg "$MODELS/$name" "$name" || return 1
    miles "JEV_MILES_NUM_ROLLOUT=4 python -m jevharness.training.miles_port.launch run --config $OUT/smoke_sft_$name.yaml" \
      > "$OUT/smoke_train_$name.log" 2>&1 || rc=1
    miles "python -m jevharness.training.miles_port.export --dcp checkpoints/r6/smoke_$name/dcp --origin $MODELS/$name \
      --out checkpoints/r6/smoke_${name}_hf" > "$OUT/smoke_export_$name.log" 2>&1 || rc=1
    rm -rf "checkpoints/r6/smoke_$name/dcp"
  done
  return $rc
}
stage_base_rule() {
  python scripts/train/r6_chain.py base --dir "$OUT" --models "$MODELS" --out "$OUT/base.json"
  python -c "import json,sys; sys.exit(0 if json.load(open('$OUT/base.json'))['path'] else 1)"
}

# ----------------------------------------------------------------------------------------------- SFT v2
stage_sft() {
  miles "python -m jevharness.training.miles_port.launch run --config configs/train/miles_sft_v2.yaml" || return 1
  miles "python -m jevharness.training.miles_port.export --dcp $CK/sft/dcp --origin $(base_path) --out $CK/sft_v2 \
    --extra '{\"stage\": \"sft_v2\"}'" || return 1
  cp -f "$CK/sft/rollout_metrics.jsonl" "$OUT/sft_rollout_metrics.jsonl" 2>/dev/null
  cp -f "$CK/sft/miles_cmd.txt" "$OUT/sft_miles_cmd.txt" 2>/dev/null
  echo "$W/JevRL/$CK/sft_v2" > "$OUT/init_opd.txt"
  sync_ck sft_v2
  rm -rf "$CK/sft/dcp"   # the exported HF directory is the artefact; DCP states (~75 GB each) only served resumes
}
stage_eval_sft() {
  evaluate "$CK/sft_v2" sft_v2 || return 1
  gonogo_of sft_v2
}
stage_stress_sft() { stress_ckpt sft_v2 "$CK/sft_v2"; }
stage_e2dev_sft() { e2dev_ckpt sft_v2 "$CK/sft_v2" "ours_ts_replace ours_ts_params_replace ours_ts_assist_k3"; }

# ----------------------------------------------------------------------------------------------- references (once)
stage_stress_refs_intern() {  # G6 reference on the stress slice: Intern-Decision-4B in-process (GPU ${REF_GPU:-1})
  local g=${REF_GPU:-1}
  python scripts/train/r5_select_arms.py "$STRESS_CFG" "$SRD/cfg_stress_refs_intern.yaml" --prefix intern_ \
    --set "name=stress_refs_intern" || return 1
  evx "CUDA_VISIBLE_DEVICES=$g python -m jevharness.bench.cli run $SRD/cfg_stress_refs_intern.yaml --out $SRD"
  local rc=$?
  python -m jevharness.bench.cli report "$SRD/stress_refs_intern" > "$SRD/stress_refs_intern/report.md" 2>/dev/null
  aws s3 sync "$SRD/stress_refs_intern" "$S3OUT/live_runs/stress_refs_intern" --only-show-errors --region "$REGION"
  python scripts/train/r5_summary.py "$SRD/stress_refs_intern" --out "$OUT/stress/stress_refs_intern_summary" > /dev/null
  return $rc
}
stage_stress_refs_jev() {  # descriptive reference on the stress slice: Jev via OpenRouter (API only, no GPU)
  live_run "$STRESS_CFG" "$SRD" stress_refs_jev "jev_"
  python scripts/train/r5_summary.py "$SRD/stress_refs_jev" --out "$OUT/stress/stress_refs_jev_summary" > /dev/null
}
stage_zeroshot_row() {  # the untrained base: decision level + stress (the "base" row of the ablation)
  local key
  key=$(python -c "import json; print(json.load(open('$OUT/base.json'))['base'])")
  evaluate "$(base_path)" "zeroshot_$key" || return 1
  gonogo_of "zeroshot_$key"
  # the untrained base is not run on the stress slice (API budget, manifest §9)
}

# ----------------------------------------------------------------------------------------------- OPD v2
export_cont() {  # export_cont STAGE STEP TAG: the stage's DCP checkpoint at iteration STEP -> HF dir $CK/TAG
  miles "python -m jevharness.training.miles_port.export --dcp $CK/$1/dcp --iter $2 --origin $(cat "$OUT/init_$1.txt") \
    --out $CK/$3 --extra '{\"stage\": \"$3\", \"iteration\": $2}'" || return 1
  sync_ck "$3"
}
train_with_retry() {  # train_with_retry CONFIG LOG_TAG [miles-env...]: one restart from the last checkpoint after a
  # crash (manifest §3); after an out-of-memory crash the restart uses a smaller micro-batch token budget
  local cfg=$1 tag=$2 rc tok=""
  shift 2
  local try
  for try in 1 2; do
    miles "JEV_MILES_MAX_TOKENS=$tok $* python -m jevharness.training.miles_port.launch run --config $cfg" \
      > "$OUT/${tag}_train_try$try.log" 2>&1
    rc=$?
    grep -E "step [0-9]+: \{'train/loss|perf [0-9]+: \{'perf/update" "$OUT/${tag}_train_try$try.log" | tail -3
    [[ $rc -eq 0 ]] && return 0
    echo "trainer attempt $try exited rc=$rc"
    grep -E "Traceback|Error" "$OUT/${tag}_train_try$try.log" | grep -v "audit op" | tail -5
    if grep -qiE "out of memory|OutOfMemoryError" "$OUT/${tag}_train_try$try.log"; then tok=8192; fi
  done
  return $rc
}
stage_opd() {
  if [[ ! -d "$CK/opd/dcp" ]]; then sudo rm -rf "$CK/opd"; fi   # nothing to resume: a fresh start (10:08 UTC attempt
                                                           # stopped at engine start-up, before any rollout)
  train_with_retry configs/train/miles_opd_v2.yaml opd || true
  local any=0
  if [[ -d "$CK/opd/dcp/iter_0000012" ]]; then export_cont opd 12 opd_i12 || return 1; any=1; fi
  if [[ -d "$CK/opd/dcp/iter_0000024" ]]; then export_cont opd 24 opd || return 1; any=1; fi
  (( any )) || { cp -f "$CK/opd/rollout_metrics.jsonl" "$OUT/opd_rollout_metrics.jsonl" 2>/dev/null; return 1; }
  for f in rollout_metrics.jsonl teacher_ledger.jsonl miles_cmd.txt; do cp -f "$CK/opd/$f" "$OUT/opd_$f" 2>/dev/null; done
  aws s3 sync "$CK/opd/labelled" "$S3CK/opd/labelled" --only-show-errors --region "$REGION"
  rm -rf "$CK/opd/dcp"
}
stage_eval_opd() {
  for t in opd_i12 opd; do evaluate "$CK/$t" "$t" || return 1; gonogo_of "$t" sft_v2; done
}
stage_stress_opd() { stress_ckpt opd_i12 "$CK/opd_i12"; stress_ckpt opd "$CK/opd"; }
# an absent continuation (OPD stopped before that iteration) is skipped: the chain reads it as "not trained"
stage_eval_opd_i12() { [[ -d "$CK/opd_i12" ]] || return 0; evaluate "$CK/opd_i12" opd_i12 && gonogo_of opd_i12 sft_v2; }
stage_eval_opd_last() { [[ -d "$CK/opd" ]] || return 0; evaluate "$CK/opd" opd && gonogo_of opd sft_v2; }
stage_stress_opd_i12() { [[ -d "$CK/opd_i12" ]] || return 0; stress_ckpt opd_i12 "$CK/opd_i12"; }
stage_stress_opd_last() { [[ -d "$CK/opd" ]] || return 0; stress_ckpt opd "$CK/opd"; }
stage_post_opd() {  # the OPD read-outs side by side; the chain (keep rules) needs all four
  par "eval_opd_i12:EVAL_GPU=0" "eval_opd_last:EVAL_GPU=1" "stress_opd_i12:SERVE_GPU=2+PORT=8083" \
    "stress_opd_last:SERVE_GPU=3+PORT=8084"
}

# ----------------------------------------------------------------------------------------------- GRPO v2
stage_grpo_validity() {  # A7.10 (f): v1's sampler / replay tests + the Miles-port parity and loss tests, on this commit
  python -m pytest -p no:warnings tests/test_grpo_validity.py tests/test_miles_port.py tests/test_r6_chain.py \
    > "$OUT/grpo_validity.txt" 2>&1
  local rc=$?
  echo "pytest_exit=$rc" >> "$OUT/grpo_validity.txt"   # r6_chain.validity_passed reads this line first
  return $rc
}
stage_chain() {
  python scripts/train/r6_chain.py decide --dir "$OUT" --out "$OUT/chain_decision.json"
}
stage_grpo_replay() {  # on-host replay check: stored SGLang behaviour log-probs vs the trainer's HF forward
  stage_chain > /dev/null || return 1
  local init
  init=$(python -c "import json; print(json.load(open('$OUT/chain_decision.json'))['grpo_init'])") || return 1
  echo "$W/JevRL/$CK/$init" > "$OUT/init_grpo.txt"
  miles "python -m jevharness.training.miles_port.smoke replay --model $(cat "$OUT/init_grpo.txt") \
    --config configs/train/miles_grpo_v2.yaml --out $OUT/grpo_replay_check.json"
}
stage_grpo() {
  python -c "import json,sys; sys.exit(0 if json.load(open('$OUT/grpo_replay_check.json')).get('pass') else 1)" || {
    echo "replay check failed: GRPO excluded (A7.10 (f))"; return 1; }
  grep -q "pytest_exit=0" "$OUT/grpo_validity.txt" || { echo "validity tests not passed: GRPO excluded (A7.10 (f))"; return 1; }
  # GRPO_END_EPOCH (the time the GPU budget left for GRPO runs out, manifest §3): GRPO starts only with >= 120 min
  # left (iteration 15 needs about half the stage); at the end time the trainer is stopped and the stage ends at its
  # last saved continuation (iteration 15 or 30). One restart from the last checkpoint after a crash.
  local end=${GRPO_END_EPOCH:-0} watcher="" rc=1 try tok="" left
  rm -f "$OUT/grpo_budget_stop"
  if [[ ! -d "$CK/grpo/dcp" ]]; then sudo rm -rf "$CK/grpo"; fi
  if (( end > 0 )); then
    left=$(( (end - $(date +%s)) / 60 ))
    echo "GPU budget left for GRPO: ${left} min"
    (( left >= 120 )) || { echo "less than 120 min left: GRPO not started"; return 1; }
    ( while (( $(date +%s) < end )); do sleep 60; done
      echo "GRPO budget end reached: stopping the trainer"; touch "$OUT/grpo_budget_stop"
      $DOCKER stop -t 60 jev-grpo > /dev/null 2>&1 ) &
    watcher=$!
  fi
  for try in 1 2; do
    MILES_NAME=jev-grpo miles "JEV_MILES_MAX_TOKENS=$tok python -m jevharness.training.miles_port.launch run \
      --config configs/train/miles_grpo_v2.yaml" > "$OUT/grpo_train_try$try.log" 2>&1
    rc=$?
    grep -E "step [0-9]+: \{'train/loss|perf [0-9]+: \{'perf/update" "$OUT/grpo_train_try$try.log" | tail -3
    if [[ $rc -eq 0 || -e "$OUT/grpo_budget_stop" ]]; then break; fi
    echo "trainer attempt $try exited rc=$rc"
    grep -E "Traceback|Error" "$OUT/grpo_train_try$try.log" | grep -v "audit op" | tail -5
    if grep -qiE "out of memory|OutOfMemoryError" "$OUT/grpo_train_try$try.log"; then tok=8192; fi
  done
  if [[ -n "$watcher" ]]; then kill "$watcher" 2>/dev/null; fi
  local any=0
  if [[ -d "$CK/grpo/dcp/iter_0000015" ]]; then export_cont grpo 15 grpo_i15 || return 1; any=1; fi
  if [[ -d "$CK/grpo/dcp/iter_0000030" ]]; then export_cont grpo 30 grpo || return 1; any=1; fi
  echo "GRPO trainer rc=$rc; continuations exported: $(ls -d "$CK"/grpo_i15 "$CK"/grpo 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
  (( any )) || return 1
  for f in rollout_metrics.jsonl rollouts.jsonl miles_cmd.txt; do cp -f "$CK/grpo/$f" "$OUT/grpo_$f" 2>/dev/null; done
  rm -rf "$CK/grpo/dcp"
}
smoke_stage_cfg() {  # smoke_stage_cfg STAGE: a one-iteration copy of the stage config, init SFT v2, nothing saved
  python - "$1" "$W/JevRL/$CK/sft_v2" "$OUT/smoke_$1.yaml" <<'EOF'
import sys, yaml
stage, init, dst = sys.argv[1:]
c = yaml.safe_load(open(f"configs/train/miles_{stage}_v2.yaml"))
c.update(name=f"smoke_{stage}", init=init, out=f"checkpoints/r6/smoke_{stage}")
c["rollout"]["iterations"] = 1
if stage == "opd":
    c["rollout"]["tasks_per_iter"] = 6
    c["opd"].update(fresh_rows=128, replay_rows=0, budget_usd=1.0)
else:
    c["rollout"]["tasks_per_iter"] = 2
c["miles"].update(save_interval=0, keep_steps=[])
yaml.safe_dump(c, open(dst, "w"))
EOF
}
smoke_stage() {  # one colocated Miles iteration (SGLang rollouts in the harness, the decision loss, weight sync)
  smoke_stage_cfg "$1" || return 1
  sudo rm -rf "checkpoints/r6/smoke_$1"
  miles "JEV_MILES_NO_SAVE=1 python -m jevharness.training.miles_port.launch run --config $OUT/smoke_$1.yaml" \
    > "$OUT/smoke_$1_train.log" 2>&1
  local rc=$?
  grep -E "step [0-9]+: \{'train/|perf [0-9]+: \{'perf/update" "$OUT/smoke_$1_train.log" | cut -c1-400 | tail -4
  grep -E "Traceback|Error" "$OUT/smoke_$1_train.log" | grep -v "audit op" | tail -5
  cp -f "checkpoints/r6/smoke_$1/rollout_metrics.jsonl" "$OUT/smoke_$1_rollout_metrics.jsonl" 2>/dev/null
  return $rc
}
stage_smoke_opd() { smoke_stage opd; }
stage_smoke_grpo() { smoke_stage grpo; }
stage_eval_grpo() {
  local init
  init=$(basename "$(cat "$OUT/init_grpo.txt")")
  for t in grpo_i15 grpo; do evaluate "$CK/$t" "$t" || return 1; gonogo_of "$t" "$init"; done
}
stage_stress_grpo() { stress_ckpt grpo_i15 "$CK/grpo_i15"; stress_ckpt grpo "$CK/grpo"; }
stage_eval_grpo_i15() { evaluate "$CK/grpo_i15" grpo_i15 && gonogo_of grpo_i15 "$(basename "$(cat "$OUT/init_grpo.txt")")"; }
stage_eval_grpo_last() {  # absent when the budget stopped GRPO before iteration 30 (the chain reads "not trained")
  [[ -d "$CK/grpo" ]] || { echo "no iteration-30 continuation"; return 0; }
  evaluate "$CK/grpo" grpo && gonogo_of grpo "$(basename "$(cat "$OUT/init_grpo.txt")")"
}
stage_stress_grpo_i15() { stress_ckpt grpo_i15 "$CK/grpo_i15"; }
stage_stress_grpo_last() { [[ -d "$CK/grpo" ]] || { echo "no iteration-30 continuation"; return 0; }; stress_ckpt grpo "$CK/grpo"; }
stage_post_grpo() {
  par "eval_grpo_i15:EVAL_GPU=0" "eval_grpo_last:EVAL_GPU=1" "stress_grpo_i15:SERVE_GPU=2+PORT=8083" \
    "stress_grpo_last:SERVE_GPU=3+PORT=8084"
}

# ----------------------------------------------------------------------------------------------- freeze
stage_freeze() {  # the chain's frozen continuation -> $CK/final (+ temperatures on v1 dev) and the freeze record
  stage_chain || return 1
  local from
  from=$(python -c "import json; print(json.load(open('$OUT/chain_decision.json'))['frozen'])") || return 1
  [[ -n "$from" && "$from" != None ]] || { echo "nothing to freeze"; return 1; }
  rm -rf "$CK/final" && cp -r "$CK/$from" "$CK/final"
  evx "python -m jevharness.training.calibrate --model $CK/final --dev $D1/dev.jsonl.gz \
    --out $CK/final/calibration.json --min-bucket 200" || return 1
  evx "python -m jevharness.training.eval_decisions --model $CK/final --calibration $CK/final/calibration.json \
    --dev ${V1_FILES[*]} --n-perm 0 --batch-size $EVAL_BATCH --out $EVAL/eval_final_calibrated_v1.json"
  # regression corpus (A7.10 (b) repairs): the serving path's probabilities on 64 dev_recovery rows, in the image and in
  # the project venv (the evaluation workspace's stack); both files and their comparison go with the freeze record
  evx "python -m jevharness.training.miles_port.smoke regress --model $CK/final --calibration $CK/final/calibration.json \
    --dev $D1/dev_recovery.jsonl.gz --n 64 --out $CK/final/regression_image.json"
  python -m jevharness.training.miles_port.smoke regress --model "$CK/final" --calibration "$CK/final/calibration.json" \
    --dev "$D1/dev_recovery.jsonl.gz" --n 64 --out "$CK/final/regression_venv.json"
  python -m jevharness.training.miles_port.smoke compare "$CK/final/regression_image.json" \
    "$CK/final/regression_venv.json" --out "$OUT/final_regression_compare.json"
  local commit
  commit=$(git rev-parse HEAD)
  {
    echo "name JevAgentRL-v2 (Miles track; exploratory, A7.19)"
    echo "frozen_from $from ($CK/$from)"
    echo "label $(python -c "import json; print(json.load(open('$OUT/chain_decision.json'))['label'])")"
    echo "base $(base_path) ($(python -c "import json; print(json.load(open('$OUT/base.json'))['name'])"))"
    echo "serving / training commit $commit (training/serve_jev.py, training/data.py render, decision_model.chat_text)"
    echo "miles image $(cat "$OUT/miles_image_digest.txt" 2>/dev/null)"
    echo "chain_decision sha256 $(sha256sum "$OUT/chain_decision.json" | cut -c1-64)"
    echo "frozen_at $(date -u +%FT%TZ)"
    (cd "$CK/final" && sha256sum ./*.safetensors config.json tokenizer.json tokenizer_config.json chat_template.jinja \
      calibration.json export.json decision_config.json regression_image.json regression_venv.json 2>/dev/null)
  } > "$CK/final/FROZEN.txt"
  cp -f "$CK/final/FROZEN.txt" "$OUT/final_FROZEN.txt"
  cp -f "$CK/final/calibration.json" "$OUT/final_calibration.json"
  aws s3 sync "$CK/final" "$S3CK/jevagentrl-v2-final" --only-show-errors --region "$REGION"
}
stage_e2dev_final() { e2dev_ckpt final "$CK/final" 'ours_ts_replace$ ours_ts_params_replace$ ours_ts_assist_k3$'; }   # minimum set (manifest §9)
stage_hashes() {
  { for d in sft_v2 opd_i12 opd grpo_i15 grpo final; do
      if [[ -d "$CK/$d" ]]; then (cd "$CK/$d" && sha256sum ./*.safetensors | sed "s|^|$d |"); fi
    done; } > "$OUT/checkpoint_sha256.txt"
}
stage_ablation() {
  python scripts/train/r6_chain.py ablation --dir "$OUT" --out "$OUT/ablation"
}
