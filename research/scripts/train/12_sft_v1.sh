#!/usr/bin/env bash
# GPU round 4 -- the next training iteration on data/train_v1 (docs/TRAINING_RECIPES.md §18), one GPU:
#   sft       R3-v1: Qwen3-4B LoRA, soft CE + 0.5 Brier + option permutation on data/train_v1/train.jsonl.gz (Jev soft labels on
#             the harness's own states: Flash-Lite shadow / near-miss replace / guard arms + v0 train + knockout-final
#             variants) + the decontaminated When2Call-train replay                          -> $OUT
#   eval      eval_decisions on the gate set (data/train_v0/dev.jsonl, 210 decisions: comparable with Jev 0.971 /
#             Intern-4B 0.862) + v0 When2Call dev (flip rate), and on v1 dev / dev_recovery / dev_knockout
#             (the in-harness states); also every checkpoint in BASELINES (e.g. round 3's checkpoints/v0/r2)
#   intern    Intern-Decision-4B on v1 dev + dev_recovery (scripts/gate_s1_dev.py; once, GPU, ~15 min): the in-harness
#             reference of the go / no-go (Jev's is data/train_v1/dev_teacher.summary.json)
#   gonogo    python -m jevharness.training.data_v1 gonogo -> $EVAL_DIR/gonogo_<name>.json (criteria: §18.4)
#   opd       R9 on-policy distillation from $OUT: the student drives JevGym in the EVALUATION SETTING on training seeds
#             40000.. (disjoint from v0 / v1 / the gate set), Jev teacher, Flash-Lite executor; greedy dev rollouts on
#             the training-side dev seeds 1000.. (never the calibration range 5000-5149 or evaluation seeds) -> $OPD_OUT
#   opd_eval  eval + gonogo of the OPD adapter (baseline: R3-v1)
#   e2dev     (opt-in, STAGES="... e2dev"; Gemini key, ~$1) serve the best adapter and run shadow / ts_replace /
#             ts_params_replace on the DEV seeds 0-9 with Flash System-2: paired with round 2's Intern-4B rows
#             (results/gpu_round2/e2a_intern_dev). No E1 (BFCL / When2Call test) here: model selection never sees it.
#
# Hardware presets (HW): l4 = 1x L4 24 GB, l40s = 1x L40S 48 GB (anything below is overridable).
#                         l4                                  l40s
#   SFT BATCH / EFF_BATCH 2 / 64 (grad-accum 32)              4 / 64 (grad-accum 16)
#   SFT EPOCHS            1  (~16.9k rows incl. replay)       2
#   MAX_LEN               4096 (prompt tail kept)             4096
#   OPD MICRO_BATCH       4 + --gradient-checkpointing        8
#   OPD WORKERS           8                                   16
#   OPD ITERATIONS        10 x 16 tasks                       20 x 16 tasks
#   EVAL_BATCH            8                                   16
# Variables (defaults in brackets): HW [l4]  MODEL [Qwen/Qwen3-4B]  D1 [data/train_v1]  D0 [data/train_v0]
#   NAME [r3v1]  OUT [$CKPT/v1/r3]  INIT [''] (e.g. checkpoints/v0/r2 with LR=5e-5)  LR [1e-4]  TRAIN_FACTOR [1.0]
#   REPLAY_FACTOR [0.45]  EVAL_DIR [runs/train_eval/v1]  BASELINES [checkpoints/v0/r2 if present]
#   OPD_OUT [$CKPT/v1/r9-opd]  OPD_BUDGET [2] (teacher $)  OPD_SEED0 [40000]  OPD_N_TASKS [512]  OPD_EVAL_SEED0 [1000]
#   OPD_ARMS [''] (default arm: replace on tool_select + tool_params)  E2DEV_N [10]  E2DEV_ADAPTER [$OUT]
#   STAGES ["sft eval intern gonogo opd opd_eval"]  DRY_RUN=1 prints every command.
# Example (round 4 on the L4 box; data/train_v1 ships with the repo, else `bash scripts/train/make_data_v1.sh`):
#   HW=l4 bash scripts/train/12_sft_v1.sh 2>&1 | tee runs/train_logs/v1_r3.log
source "$(dirname "$0")/_common.sh"

: "${HW:=l4}"
case "$HW" in
  l4)   : "${BATCH:=2}"; : "${EPOCHS:=1}"; : "${OPD_MICRO:=4}"; : "${OPD_EXTRA:=--gradient-checkpointing}"
        : "${OPD_WORKERS:=8}"; : "${OPD_ITERS:=10}"; : "${EVAL_BATCH:=8}" ;;
  l40s) : "${BATCH:=4}"; : "${EPOCHS:=2}"; : "${OPD_MICRO:=8}"; : "${OPD_EXTRA:=}"
        : "${OPD_WORKERS:=16}"; : "${OPD_ITERS:=20}"; : "${EVAL_BATCH:=16}" ;;
  *) log "unknown HW=$HW (l4 | l40s)"; exit 2 ;;
esac
: "${EFF_BATCH:=64}"
: "${MAX_LEN:=4096}"
: "${D1:=data/train_v1}"
: "${D0:=data/train_v0}"
: "${NAME:=r3v1}"
: "${OUT:=$CKPT/v1/r3}"
: "${INIT:=}"
: "${LR:=1e-4}"
: "${TRAIN_FACTOR:=1.0}"
: "${REPLAY_FACTOR:=0.45}"
: "${EVAL_DIR:=runs/train_eval/v1}"
if [[ -z "${BASELINES+x}" ]]; then
  BASELINES=""
  if [[ -e "$CKPT/v0/r2/adapter_config.json" ]]; then
    BASELINES="$CKPT/v0/r2"
  fi
fi
: "${OPD_OUT:=$CKPT/v1/r9-opd}"
: "${OPD_BUDGET:=2}"
: "${OPD_SEED0:=40000}"
: "${OPD_N_TASKS:=512}"
: "${OPD_EVAL_SEED0:=1000}"
: "${OPD_ARMS:=}"
: "${E2DEV_N:=10}"
: "${E2DEV_ADAPTER:=$OUT}"
: "${STAGES:=sft eval intern gonogo opd opd_eval}"

EVAL_SUITE='{families: v2, distractor_style: near, n_distractors: 60, noise_lines: 20, strict: true}'
GATE_FILES=("$D0/dev.jsonl" "$D0/dev_when2call.jsonl.gz")
V1_FILES=("$D1/dev.jsonl.gz" "$D1/dev_recovery.jsonl.gz" "$D1/dev_knockout.jsonl.gz")
INTERN='{"type": "intern_decision", "model": "internlm/Intern-Decision-4B"}'
has() { [[ " $STAGES " == *" $1 "* ]]; }
run mkdir -p "$EVAL_DIR"

# evaluate ADAPTER TAG: gate files with the flip rate (n-perm 1), v1 files without (n-perm 0: 2x faster)
evaluate() {
  local adapter=$1 tag=$2
  run "$PY" -m jevharness.training.eval_decisions --model "$MODEL" --adapter "$adapter" --dev "${GATE_FILES[@]}" \
    --n-perm 1 --max-len "$MAX_LEN" --batch-size "$EVAL_BATCH" --out "$EVAL_DIR/eval_${tag}_gate.json"
  run "$PY" -m jevharness.training.eval_decisions --model "$MODEL" --adapter "$adapter" --dev "${V1_FILES[@]}" \
    --n-perm 0 --max-len "$MAX_LEN" --batch-size "$EVAL_BATCH" --out "$EVAL_DIR/eval_${tag}_v1.json"
}
gonogo_of() {  # gonogo_of TAG [BASELINE_TAG]
  local tag=$1 base=${2:-}
  local args=(--eval "$EVAL_DIR/eval_${tag}_gate.json" "$EVAL_DIR/eval_${tag}_v1.json"
              --jev-v1 "$D1/dev_teacher.summary.json" --out "$EVAL_DIR/gonogo_${tag}.json")
  if [[ -e "$EVAL_DIR/gate_intern4b_v1dev.json" || "$DRY_RUN" == 1 ]]; then
    args+=(--intern-v1 "$EVAL_DIR/gate_intern4b_v1dev.json")
  fi
  if [[ -n "$base" ]]; then
    args+=(--baseline "$EVAL_DIR/eval_${base}_gate.json" "$EVAL_DIR/eval_${base}_v1.json")
  fi
  run "$PY" -m jevharness.training.data_v1 gonogo "${args[@]}"
}

need "$D1/train.jsonl.gz" "${V1_FILES[@]}" "$D1/dev_teacher.summary.json" "${GATE_FILES[@]}"

# ---------------------------------------------------------------- 1. SFT (R3-v1)
if has sft; then
  train="$D1/train.jsonl.gz"
  if [[ "$TRAIN_FACTOR" != "1.0" && "$TRAIN_FACTOR" != "1" ]]; then
    run mkdir -p "$OUT"
    run "$PY" -m jevharness.training.data mix --in "$D1/train.jsonl.gz:$TRAIN_FACTOR" --out "$OUT/train_sub.jsonl" --seed 0
    train="$OUT/train_sub.jsonl"
  fi
  RECIPE=r3 TRAIN="$train" DEV="$D0/dev.jsonl" REPLAY="$D1/replay_when2call.jsonl.gz" REPLAY_FACTOR="$REPLAY_FACTOR" \
    OUT="$OUT" INIT="$INIT" LR="$LR" EPOCHS="$EPOCHS" BATCH="$BATCH" EFF_BATCH="$EFF_BATCH" MAX_LEN="$MAX_LEN" \
    GPUS=1 bash "$(dirname "$0")/10_sft.sh"
fi

# ---------------------------------------------------------------- 2. decision-level eval (+ baselines)
if has eval; then
  evaluate "$OUT" "$NAME"
  for b in $BASELINES; do
    evaluate "$b" "$(basename "$(dirname "$b")")_$(basename "$b")"
  done
fi

# ---------------------------------------------------------------- 3. Intern-Decision-4B reference on v1 dev (once)
if has intern; then
  if [[ ! -e "$EVAL_DIR/gate_intern4b_v1dev.json" || "$DRY_RUN" == 1 ]]; then
    for f in dev dev_recovery; do   # scripts/gate_s1_dev.py reads plain JSONL
      if [[ "$DRY_RUN" != 1 ]]; then
        gzip -dc "$D1/$f.jsonl.gz" > "$EVAL_DIR/v1_$f.jsonl"
      fi
    done
    run "$PY" scripts/gate_s1_dev.py --s1 "$INTERN" --dev "$EVAL_DIR/v1_dev.jsonl" --out "$EVAL_DIR/gate_intern4b_v1dev.json"
    run "$PY" scripts/gate_s1_dev.py --s1 "$INTERN" --dev "$EVAL_DIR/v1_dev_recovery.jsonl" \
      --out "$EVAL_DIR/gate_intern4b_v1recovery.json"
  fi
fi

# ---------------------------------------------------------------- 4. go / no-go of R3-v1
if has gonogo; then
  first_base=""
  for b in $BASELINES; do
    first_base="$(basename "$(dirname "$b")")_$(basename "$b")"
    break
  done
  gonogo_of "$NAME" "$first_base"
fi

# ---------------------------------------------------------------- 5. OPD continuation (R9 from R3-v1)
if has opd; then
  CONFIG=configs/train/opd_jevgym.yaml MODEL="$MODEL" INIT="$OUT" OUT="$OPD_OUT" TEACHER=jev-openrouter \
    EXECUTOR=gemini-lite SEED0="$OPD_SEED0" N_TASKS="$OPD_N_TASKS" SUITE="$EVAL_SUITE" ARMS="$OPD_ARMS" \
    ITERATIONS="$OPD_ITERS" TASKS_PER_ITER=16 BUDGET_USD="$OPD_BUDGET" MICRO_BATCH="$OPD_MICRO" WORKERS="$OPD_WORKERS" \
    MAX_LEN="$MAX_LEN" EVAL_EVERY=5 EVAL_N=32 EVAL_SEED0="$OPD_EVAL_SEED0" RESUME=auto EXTRA="$OPD_EXTRA" \
    bash "$(dirname "$0")/65_opd.sh"
fi
if has opd_eval; then
  evaluate "$OPD_OUT" "${NAME}_opd"
  gonogo_of "${NAME}_opd" "$NAME"
fi

# ---------------------------------------------------------------- 6. E2 dev slice (opt-in): paired with round 2's Intern rows
if has e2dev; then
  arms="[{name: ours_shadow, tool_select: {mode: shadow}, tool_params: {mode: shadow}, stop: {mode: shadow}},"
  arms+=" {name: ours_ts_replace, tool_select: {mode: replace}},"
  arms+=" {name: ours_ts_params_replace, tool_select: {mode: replace}, tool_params: {mode: replace}}]"
  ADAPTER="$E2DEV_ADAPTER" NAME="ours-$NAME" DEV="$D1/dev.jsonl.gz" DO_DECIDE=0 DO_RUN=1 EVAL_DIR="$EVAL_DIR/serve_$NAME" \
    RUN_SET="name=e2dev_$NAME;suite.seed0=0;suite.n=$E2DEV_N;n_tasks=$E2DEV_N;sweeps=[];s1_backends={};arms=$arms" \
    bash "$(dirname "$0")/90_calibrate_serve_eval.sh"
fi
log "done: $EVAL_DIR (eval_*.json, gonogo_*.json); adapters $OUT ${OPD_OUT}"
