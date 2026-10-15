#!/usr/bin/env bash
# GPU round 3: the first real training on the AWS box (TRAINING_AGENT_BRIEF §0), sized for one L4 24 GB.
#   R2  = Qwen3-4B LoRA SFT on data/train_v0 (+ When2Call replay 0.45)   -> checkpoints/v0/r2  (~20-30 min)
#   eval_decisions on the v0 dev sets                                       -> checkpoints/v0/r2/eval_dev.json
#   OPD pilot (recipe R9): 2 iterations x 16 tasks from r2, Jev teacher, Flash-Lite executor, teacher budget $0.5
# Outputs sync to s3://$BUCKET/gpu_round3/ after every stage. Cancels any earlier shutdown and sets its own guard.
set -uo pipefail
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}; B="s3://$BUCKET"; REGION=${AWS_REGION:-us-east-1}
STOP_AFTER=${STOP_AFTER:-240}
sudo shutdown -c 2>/dev/null || true; sudo shutdown -h +"$STOP_AFTER" "gpu_round3: budget guard" 2>/dev/null || true
cd /opt/jev && aws s3 cp "$B/code/jev.bundle" jev.bundle --only-show-errors
cd JevAgentHarness/research && git pull -q --ff-only /opt/jev/jev.bundle main && git log --oneline -1
. .venv/bin/activate
OUT=runs/gpu_round3; mkdir -p "$OUT" runs/train_logs checkpoints/v0
st() { echo "$(date -u +%H:%M:%S) $*" >> "$OUT/status.txt"; aws s3 sync "$OUT" "$B/gpu_round3" --only-show-errors --region "$REGION"; }
for v in GEMINI_API_KEY GEMINI_API_KEYS OPENROUTER_API_KEY; do
  val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value --output text 2>/dev/null) || val=""
  [ -n "$val" ] && [ "$val" != None ] && export "$v=$val"
done
st "start $(git log --oneline -1) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# E-arch (compact): dense vs hybrid x state length x stateless/prefix/session, batch 1 only, 1k/4k/16k. The 64k cells
# are left out on a 24 GB card (round 2 spent 85 min on them and the batch-8 cells OOM); see runs/gpu_round2 for those.
# --profile: per-phase wall time (encode / lookup / prefill / store / suffix) of the shared-prefix path, to see where the
# session-mode overhead measured in round 2 (96 vs 30 ms at 1k, one question) went after the fused single-pass change.
python scripts/bench_serving.py --lengths 1000 4000 16000 --batch 1 --questions 1 3 --repeats 3 --profile --out "$OUT/bench_serving" \
  > "$OUT/bench_serving.log" 2>&1; st "bench_serving(compact) exit=$?"

# E2a Intern-Decision slice: re-run the 11 arms that ended in CUDA OOM in round 2 (four separately loaded checkpoints;
# fixed by shared weights). The runner resumes by task id and retries errored tasks only, so the round-2 records are
# pulled back first; outputs go to gpu_round2/ in the bucket next to the originals (results/gpu_round2/NOTES.md §3).
if [ -n "${GEMINI_API_KEY:-}${GEMINI_API_KEYS:-}" ]; then
  mkdir -p runs/gpu_round2/e2a_intern2b_dev
  aws s3 sync "$B/gpu_round2/e2a_intern2b_dev" runs/gpu_round2/e2a_intern2b_dev --only-show-errors --region "$REGION"
  # suite.seed0=0: round 2 ran on the suite's then-default dev seeds 0-9; since A6.13 the config defaults to the evaluation
  # range 200000+, which the runner refuses without JEV_EVAL_RUNS=1 (and must never be touched by a dev re-run). The two
  # tau-hat arms carry `calibration: pending` and are skipped by the runner (A6.16 (b)); they were not among the OOM arms.
  jevbench run configs/experiments/jevgym_tool_select_intern.yaml --out runs/gpu_round2 --max-tasks 10 -o workers=4 name=e2a_intern2b_dev \
    suite.seed0=0 > "$OUT/e2a_intern_rerun.log" 2>&1; st "e2a_intern_rerun exit=$?"
  d=runs/gpu_round2/e2a_intern2b_dev; jevbench report "$d" > "$d/report.md" 2>/dev/null
  python -m jevharness.bench.scorecard "$d" --baseline llm_only --format md > "$d/scorecard.md" 2>/dev/null
  aws s3 sync "$d" "$B/gpu_round2/e2a_intern2b_dev" --only-show-errors --region "$REGION"; st "e2a_intern_rerun synced"
else st "e2a_intern_rerun skipped: no Gemini key"; fi

# Intern-Decision calibration shadow arms (PREREGISTRATION A6.5): tau-hat* for the Intern C1 arms is fitted on the
# calibration seeds 5000-5149 (never on evaluation tasks); ~1 GPU-h + ~$8 of System-2 API. Fit files -> gpu_round3/cal/.
if [ -n "${GEMINI_API_KEY:-}${GEMINI_API_KEYS:-}" ]; then
  jevbench run configs/experiments/jevgym_shadow_cal_intern.yaml --out runs -o workers=4 > "$OUT/shadow_cal_intern.log" 2>&1
  st "shadow_cal_intern exit=$?"
  d=runs/e2a_jevgym_shadow_cal_intern; mkdir -p "$OUT/cal"
  if [ -d "$d" ]; then
    jevbench report "$d" > "$d/report.md" 2>/dev/null
    for arm in intern_shadow_cal intern_shadow_cal_s2lite; do for dp in tool_select tool_params; do
      # the registered fitting procedure (A6.5 / A6.11 (c)): the same command as the Jev fits in results/calibration/
      python -m jevharness.bench.threshold_theory "$d" --arm $arm --dp $dp --calibration-arm --task-value 1.0 --lam-per-hour 0 \
        --v-fallback W --out "$OUT/cal/tauhat_${arm}_${dp}.json" > "$OUT/cal/tauhat_${arm}_${dp}.log" 2>&1
    done; done
    cp -f "$d/report.md" "$OUT/cal/shadow_cal_intern_report.md"; aws s3 sync "$d" "$B/e2a_jevgym_shadow_cal_intern" --only-show-errors --region "$REGION"
  fi
  st "shadow_cal_intern fitted"
else st "shadow_cal_intern skipped: no Gemini key"; fi

# R2: 4B LoRA SFT. L4 24 GB: BATCH=2 (brief §2.1), EFF_BATCH unchanged, MAX_LEN 4096.
if [ ! -f checkpoints/v0/r2/adapter_config.json ]; then
  BATCH=2 TRAIN=data/train_v0/train.jsonl DEV=data/train_v0/dev.jsonl REPLAY=data/train_v0/replay_when2call.jsonl.gz REPLAY_FACTOR=0.45 \
    OUT=checkpoints/v0/r2 bash scripts/train/10_sft.sh > "$OUT/r2_sft.log" 2>&1; st "r2_sft exit=$?"
else st "r2_sft: checkpoint exists, skipped"; fi
python -m jevharness.training.eval_decisions --model Qwen/Qwen3-4B --adapter checkpoints/v0/r2 \
  --dev data/train_v0/dev.jsonl data/train_v0/dev_when2call.jsonl.gz --out "$OUT/r2_eval_dev.json" > "$OUT/r2_eval.log" 2>&1; st "r2_eval exit=$?"
cp -f checkpoints/v0/r2/metrics.jsonl "$OUT/r2_metrics.jsonl" 2>/dev/null; cp -f checkpoints/v0/r2/*.json "$OUT/" 2>/dev/null

# OPD pilot from r2 (measure teacher $ / rollout time per iteration; TRAINING_RECIPES §17)
if [ -n "${OPENROUTER_API_KEY:-}" ] && [ -n "${GEMINI_API_KEY:-}${GEMINI_API_KEYS:-}" ]; then
  ITERATIONS=2 TASKS_PER_ITER=16 BUDGET_USD=0.5 INIT=checkpoints/v0/r2 EXECUTOR=gemini-lite MICRO_BATCH=4 EXTRA=--gradient-checkpointing \
    OUT=checkpoints/v0/r9-opd-pilot bash scripts/train/65_opd.sh > "$OUT/opd_pilot.log" 2>&1; st "opd_pilot exit=$?"
  cp -f checkpoints/v0/r9-opd-pilot/metrics.jsonl "$OUT/opd_metrics.jsonl" 2>/dev/null; cp -f checkpoints/v0/r9-opd-pilot/summary.json "$OUT/opd_summary.json" 2>/dev/null
else st "opd_pilot skipped: missing keys"; fi
# keep the adapters in the bucket (small), never the base weights
aws s3 sync checkpoints/v0/r2 "$B/checkpoints/v0/r2" --only-show-errors --exclude '*optimizer*' --region "$REGION"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv > "$OUT/nvidia_smi_end.txt" 2>&1
st "done"
