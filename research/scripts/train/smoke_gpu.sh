#!/usr/bin/env bash
# Whole training chain 00 -> 90 at tiny scale (minutes on one GPU) on Qwen3-0.6B: validates a GPU box, the
# installed stack and every script BEFORE real GPU hours are spent (docs/TRAINING_AGENT_BRIEF.md §3).
# Needs no API key: teacher = a local checkpoint, MC / in-harness GRPO executor = the SIMULATED LLM (pipeline only),
# E1 decide talks to the served checkpoint only (BFCL / When2Call eval data are downloaded from GitHub on first use).
# Nothing here is a result. Outputs: $SMOKE/{logs/<stage>.log, ckpt/, smoke_report.json}.
#
#   stage          what it runs                                                                  checks
#   00-harvest     harvest.py on data/train_v0/trajectories (gzipped)                            data path, rendering
#   10-sft         10_sft.sh R2 on data/train_v0 train + When2Call replay, STEPS steps            Trainer, LoRA, (DDP)
#   10-eval        eval_decisions.py on the dev files: R0 (base model) and R2                     gate metrics
#   20-teacher     20_teacher_soft.sh with a LOCAL teacher (= 10-sft ckpt) on 64 decisions + R3   teacher_label local
#   30-mc          30_mc_rollouts.sh, simulated executor, 6 v1 states                             MC plumbing
#   40-kl          40_kl_to_mc.sh (pi_ref recomputed from R3) -> R4                                local teacher + SFT
#   50-dpo, 55-pg  50_dpo.sh / 55_pg_exact.sh from R3                                             reference adapter
#   60-grpo-trl    60_grpo_trl.sh IMPL=trl (TRL GRPOTrainer), 2 steps                             TRL version
#   60-grpo-torch  60_grpo_trl.sh IMPL=torch, 2 steps                                             restricted softmax
#   70-online      70_grpo_online.sh EXECUTOR=sim, 1 iteration from R4                           in-harness RL
#   80-distill     80_distill_small.sh, FULL fine-tuning of the student (LORA_R=0)               full-FT save/load
#   90-serve-r4    90_calibrate_serve_eval.sh on R4 (LoRA): calibrate, serve_jev, jevbench ping, E1 decide (DECIDE)
#   90-serve-r7    same on R7 (full-FT checkpoint), no decide
#
# Variables (defaults in brackets):
#   MODEL [Qwen/Qwen3-0.6B]   (CPU check: MODEL=$(python scripts/train/make_tiny_model.py /tmp/tiny) MAX_LEN=512)
#   DATA_V0 [data/train_v0]   SMOKE [runs/smoke_gpu]   GPUS [1] (>1: the SFT/DPO/PG/TRL stages use accelerate DDP)
#   STEPS [10]   MAX_LEN [2048]   DECIDE [1] (E1 decide on DECIDE_ITEMS items)   DECIDE_ITEMS [20]   PORT [18081]
#   ONLY      space-separated stage numbers to run, e.g. ONLY="10 90" ('' = all; later stages need earlier ckpts)
#   KEEP_GOING [0]  1 = run the remaining stages after a failure
#   DRY_RUN=1  every stage prints its commands only (wiring check, seconds)
# The smoke default must be set BEFORE _common.sh, whose own default (Qwen3-4B) would otherwise win: on a 24 GB GPU
# that made every stage run on 4B and the TRL GRPO stage run out of memory (AWS g6.4xlarge smoke, 2026-10-01).
: "${MODEL:=Qwen/Qwen3-0.6B}"
source "$(dirname "$0")/_common.sh"

: "${DATA_V0:=data/train_v0}"
: "${SMOKE:=runs/smoke_gpu}"
: "${STEPS:=10}"
: "${MAX_LEN:=2048}"
: "${DECIDE:=1}"
: "${DECIDE_ITEMS:=20}"
: "${PORT:=18081}"
: "${ONLY:=}"
: "${KEEP_GOING:=0}"
export MODEL GPUS MAX_LEN DRY_RUN REPORT_TO=none
export MAX_STEPS="$STEPS"
HERE="$ROOT/scripts/train"
CK="$SMOKE/ckpt"
mkdir -p "$SMOKE/logs" "$CK" "$SMOKE/traj_v1"

need "$DATA_V0/train.jsonl" "$DATA_V0/dev.jsonl" "$DATA_V0/replay_when2call.jsonl.gz" \
  "$DATA_V0/trajectories/train_v0_jevgym_v1/shadow.jsonl.gz"
# small inputs (first rows; the full files are for real runs)
head -n 48 "$DATA_V0/dev.jsonl" > "$SMOKE/dev_small.jsonl"
head -n 64 "$DATA_V0/train.jsonl" > "$SMOKE/train_small.jsonl"
gzip -dc "$DATA_V0/trajectories/train_v0_jevgym_v1/shadow.jsonl.gz" > "$SMOKE/traj_v1/shadow.jsonl"
"$PY" - "$DATA_V0/trajectories/train_v0_jevgym_v1/experiment.yaml" "$SMOKE/traj_v1/experiment.yaml" <<'EOF'
import sys, yaml
exp = yaml.safe_load(open(sys.argv[1]))
# PIPELINE CHECK ONLY: the simulated executor follows the oracle with errors; never use its Q-values
exp["llm"] = {"provider": "simulated", "model": "scripted", "error_rate": 0.3, "simulated_latency_ms": 0}
yaml.safe_dump(exp, open(sys.argv[2], "w"), sort_keys=False)
EOF

STAGES=()
write_report() {
  "$PY" - "$SMOKE/smoke_report.json" "$MODEL" "$GPUS" "${STAGES[@]+"${STAGES[@]}"}" <<'EOF'
import json, platform, sys
out, model, gpus, *rows = sys.argv[1:]
env = {"python": platform.python_version()}
try:
    import torch
    env.update(torch=torch.__version__, cuda=torch.version.cuda, gpus_visible=torch.cuda.device_count(),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    import transformers, trl, peft, accelerate
    env.update(transformers=transformers.__version__, trl=trl.__version__, peft=peft.__version__,
               accelerate=accelerate.__version__)
except Exception as e:  # noqa: BLE001
    env["error"] = repr(e)
stages = [dict(zip(("stage", "rc", "seconds", "log"), r.split("|"))) for r in rows]
for s in stages:
    s["ok"] = s["rc"] == "0"
rep = {"model": model, "gpus": int(gpus), "env": env, "stages": stages, "all_ok": all(s["ok"] for s in stages)}
open(out, "w").write(json.dumps(rep, indent=1) + "\n")
print(json.dumps({"all_ok": rep["all_ok"], "report": out}))
EOF
}

# stage NAME CMD... : run CMD with its output in $SMOKE/logs/NAME.log, record rc + wall time
stage() {
  local name=$1
  shift
  if [[ -n "$ONLY" && " $ONLY " != *" ${name%%-*} "* ]]; then
    return 0
  fi
  local logf="$SMOKE/logs/$name.log" t0=$SECONDS rc=0
  ("$@") > "$logf" 2>&1 || rc=$?
  local dt=$(( SECONDS - t0 ))
  STAGES+=("$name|$rc|$dt|$logf")
  printf '%-14s %-4s %5ss  %s\n' "$name" "$([[ $rc == 0 ]] && echo ok || echo FAIL)" "$dt" "$logf"
  if (( rc != 0 )); then
    tail -n 40 "$logf" >&2
    if [[ "$KEEP_GOING" != 1 ]]; then
      write_report
      log "stage $name failed (see the failure playbook in docs/TRAINING_AGENT_BRIEF.md §8)"
      exit 1
    fi
  fi
}

log "model=$MODEL gpus=$GPUS steps=$STEPS max_len=$MAX_LEN out=$SMOKE"
lora=(LORA_R=16 LORA_ALPHA=32 BATCH=4 EFF_BATCH=8)
local_teacher=$(printf '{"type": "local", "model": "%s", "adapter": "%s", "max_len": %s, "batch_size": 8}' \
  "$MODEL" "$CK/r2" "$MAX_LEN")

stage 00-harvest env RUNS="$DATA_V0/trajectories/train_v0_jevgym_v1" WHEN2CALL=0 OUT="$SMOKE/harvest" \
  bash "$HERE/00_harvest.sh"
stage 10-sft env "${lora[@]}" TRAIN="$DATA_V0/train.jsonl" DEV="$SMOKE/dev_small.jsonl" \
  REPLAY="$DATA_V0/replay_when2call.jsonl.gz" REPLAY_FACTOR=0.2 OUT="$CK/r2" bash "$HERE/10_sft.sh"
eval_r0_r2() {  # R0 (base model, zero-shot) and R2 on the dev files: the gate metrics of TRAINING_AGENT_BRIEF §6
  local dev=("$SMOKE/dev_small.jsonl" "$DATA_V0/dev_when2call.jsonl.gz")
  run "$PY" -m jevharness.training.eval_decisions --model "$MODEL" --dev "${dev[@]}" --limit 48 --max-len "$MAX_LEN" \
    --out "$CK/r0_eval_dev.json"
  run "$PY" -m jevharness.training.eval_decisions --model "$MODEL" --adapter "$CK/r2" --dev "${dev[@]}" --limit 48 \
    --max-len "$MAX_LEN" --out "$CK/r2/eval_dev.json"
}
stage 10-eval eval_r0_r2
stage 20-teacher env "${lora[@]}" TEACHERS="r2=$local_teacher" IN="$SMOKE/train_small.jsonl" \
  LABELED="$SMOKE/train_teacher_local.jsonl" CACHE= MIN_AGREE=1 GOLD_FILTER=keep ON_DISAGREE=gold DO_TRAIN=1 \
  OUT="$CK/r3" INIT= bash "$HERE/20_teacher_soft.sh"
stage 30-mc env RUNS="$SMOKE/traj_v1/shadow.jsonl" EXP="$SMOKE/traj_v1/experiment.yaml" MAX_STATES=6 K_ALT=3 M=2 \
  WORKERS=4 OUT="$SMOKE/mc_sim.jsonl" bash "$HERE/30_mc_rollouts.sh"
stage 40-kl env BATCH=4 EFF_BATCH=8 INIT="$CK/r3" MC="$SMOKE/mc_sim.jsonl" REPLAY="$SMOKE/train_small.jsonl" \
  OUT="$CK/r4" bash "$HERE/40_kl_to_mc.sh"
stage 50-dpo env BATCH=4 EFF_BATCH=8 INIT="$CK/r3" MC="$SMOKE/mc_sim.jsonl" MIN_Q_GAP=0 DEV= OUT="$CK/r5" \
  bash "$HERE/50_dpo.sh"
stage 55-pg env BATCH=4 EFF_BATCH=8 INIT="$CK/r3" MC="$SMOKE/mc_sim.jsonl" DEV= OUT="$CK/r6" bash "$HERE/55_pg_exact.sh"
stage 60-grpo-trl env IMPL=trl INIT="$CK/r3" TRAIN="$SMOKE/train_small.jsonl" DEV= OUT="$CK/r6b" MAX_STEPS=2 \
  NUM_GEN=4 BATCH=8 GRAD_ACCUM=1 bash "$HERE/60_grpo_trl.sh"
stage 60-grpo-torch env IMPL=torch GPUS=1 INIT="$CK/r3" TRAIN="$SMOKE/train_small.jsonl" DEV="$SMOKE/dev_small.jsonl" \
  OUT="$CK/r6b-torch" MAX_STEPS=2 NUM_GEN=4 BATCH=8 GRAD_ACCUM=1 bash "$HERE/60_grpo_trl.sh"
stage 65-opd env CONFIG=configs/train/opd_smoke.yaml EXECUTOR=sim INIT="$CK/r3" OUT="$CK/r9-opd" MAX_LEN="$MAX_LEN" \
  TEACHER="r4=$(printf '{"type": "local", "model": "%s", "adapter": "%s", "max_len": %s, "batch_size": 8}' \
    "$MODEL" "$CK/r4" "$MAX_LEN")" TEACHER_CACHE= bash "$HERE/65_opd.sh"
stage 70-online env EXECUTOR=sim ITERATIONS=1 GPUS=1 SEEDS=0 INIT="$CK/r4" OUT="$CK/r8" \
  SET="grpo.tasks_per_iter=2;grpo.group_size=2;suite.n=4;eval_suite.n=2;eval_every=1;workers=4;decider.max_len=$MAX_LEN" \
  bash "$HERE/70_grpo_online.sh"
stage 80-distill env TEACHER_MODEL="$MODEL" TEACHER_ADAPTER="$CK/r4" STUDENT="$MODEL" IN="$SMOKE/train_small.jsonl" \
  LABELED="$SMOKE/train_distill.jsonl" OUT="$CK/r7" LORA_R=0 BATCH=4 EFF_BATCH=8 TEACHER_BATCH=8 \
  bash "$HERE/80_distill_small.sh"
stage 90-serve-r4 env ADAPTER="$CK/r4" NAME=smoke-r4 PORT="$PORT" DEV="$SMOKE/dev_small.jsonl" MIN_BUCKET=8 \
  DO_DECIDE="$DECIDE" MAX_ITEMS="$DECIDE_ITEMS" DO_RUN=0 EVAL_DIR="$SMOKE/eval_r4" RUNS_DIR="$SMOKE/runs" \
  bash "$HERE/90_calibrate_serve_eval.sh"
stage 90-serve-r7 env ADAPTER="$CK/r7" NAME=smoke-r7 PORT="$((PORT + 1))" DEV="$SMOKE/dev_small.jsonl" MIN_BUCKET=8 \
  DO_DECIDE=0 DO_RUN=0 EVAL_DIR="$SMOKE/eval_r7" RUNS_DIR="$SMOKE/runs" bash "$HERE/90_calibrate_serve_eval.sh"

write_report
log "done: $SMOKE/smoke_report.json (paste it into the report, docs/TRAINING_AGENT_BRIEF.md §7)"
