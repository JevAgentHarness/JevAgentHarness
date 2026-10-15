#!/usr/bin/env bash
# Regenerate data/train_v0 -- the committed, API-free starter data of the training programme
# (docs/TRAINING_AGENT_BRIEF.md §4). Every step is resumable / cached, so a re-run over existing runs costs $0.
#
#   1. episodes   JevGym TRAINING seeds, one `shadow` arm (LLM-only behaviour, System-1 asked in shadow):
#                   v2 calibrated suite  seeds 10000..10000+N_V2-1 (clarify+multi, 60 near distractors, noise 20)
#                   v1 families          seeds 12000..12000+N_V1-1 (cancel/refund/stock/ship/total, 30 mixed)
#                 System-2 = Gemini 3.5 Flash; System-1 (shadow) = Gemini 3.5 Flash-Lite verbalised distributions
#                 (STAND-IN: Jev was unreachable when v0 was made).                     configs/train/data_v0_jevgym.yaml
#   2. harvest    gold-labelled decisions (tool_select / tool_params / stop gates) -> train.jsonl / dev.jsonl,
#                 split by TASK; tool options rendered `name -- description` exactly as the served model sees them
#   3. replay     When2Call TRAIN (SFT file; ask / cannot-answer only) -> replay_when2call.jsonl.gz, dev_when2call.jsonl.gz
#   4. teacher    stand-in soft labels from Gemini Flash-Lite (teacher_label.py) -> train_teacher_standin.jsonl;
#                 its metrics on dev -> dev_teacher_standin.summary.json (dev itself stays gold)
#   5. mc         (MC=1) a few counterfactual Q-values with a Flash-Lite executor -> mc_standin.jsonl
#   6. copy the trajectories (gzipped) + experiment.yaml, write MANIFEST.json, check the size (< MAX_MB)
#
# Variables (defaults in brackets):
#   OUT [data/train_v0]   RUNS_DIR [runs/dev]   N_V2 [30]   N_V1 [60]   WORKERS [6]
#   W2C_N [3000] (When2Call-train rows; 10% held out as dev_when2call)   DEV_FRAC [0.15]   SEED [0]
#   TEACHER [lite=gemini-lite]   (Jev, once reachable: 'jev=jev-openrouter;lite=gemini-lite' -> see the brief)
#   MC [1]   MC_STATES [24]   MC_K [3]   MC_M [2]   MAX_MB [15]
#   SKIP_EPISODES=1  reuse RUNS_DIR as is (no API call for step 1)
#   RESTORE_FROM [data/train_v0]  episodes missing from RUNS_DIR are first restored from its trajectories/ (free)
#   NOTES [configs/train/data_v0_notes.json]  merged into MANIFEST.json ('' for a new data dir such as data/train_v1)
#   DRY_RUN=1        print the commands only
# Cost of the committed v0: data/train_v0/MANIFEST.json "cost_usd" (Ledger, ~ $4.7) + notes.cost_estimates (~ $5.2 all in).
source "$(dirname "$0")/_common.sh"

: "${OUT:=data/train_v0}"
: "${RUNS_DIR:=runs/dev}"
: "${N_V2:=30}"
: "${N_V1:=60}"
: "${WORKERS:=6}"
: "${W2C_N:=3000}"
: "${DEV_FRAC:=0.15}"
: "${SEED:=0}"
: "${TEACHER:=lite=gemini-lite}"
: "${MC:=1}"
: "${MC_STATES:=24}"
: "${MC_K:=3}"
: "${MC_M:=2}"
: "${MAX_MB:=15}"
: "${SKIP_EPISODES:=0}"
: "${RESTORE_FROM:=data/train_v0}"
: "${NOTES=configs/train/data_v0_notes.json}"   # NOTES= (empty) for other data dirs: the v0 notes describe v0

CFG=configs/train/data_v0_jevgym.yaml
V2="$RUNS_DIR/train_v0_jevgym_v2"
V1="$RUNS_DIR/train_v0_jevgym_v1"
TMP="$RUNS_DIR/train_v0_tmp"
jb=("$PY" -m jevharness.bench.cli)

# 0. restore committed episodes (e.g. data/train_v0/trajectories on a fresh clone) so the resumable runner skips them
for d in "$V2" "$V1"; do
  src="$RESTORE_FROM/trajectories/$(basename "$d")"
  if [[ ! -s "$d/shadow.jsonl" && -s "$src/shadow.jsonl.gz" ]]; then
    run mkdir -p "$d"
    if [[ "$DRY_RUN" != 1 ]]; then
      gzip -dc "$src/shadow.jsonl.gz" > "$d/shadow.jsonl"
    fi
    run cp "$src/experiment.yaml" "$d/experiment.yaml"
    log "restored $(basename "$d") from $src"
  fi
done

# 1. episodes (resumable: finished tasks are skipped)
if [[ "$SKIP_EPISODES" != 1 ]]; then
  run "${jb[@]}" run "$CFG" --out "$RUNS_DIR" -o "suite.n=$N_V2" "workers=$WORKERS"
  run "${jb[@]}" run "$CFG" --out "$RUNS_DIR" -o name=train_v0_jevgym_v1 "suite.n=$N_V1" suite.families=v1 \
    suite.seed0=12000 suite.distractor_style=mixed suite.n_distractors=30 suite.noise_lines=0 "workers=$WORKERS"
fi
need "$V2/shadow.jsonl" "$V1/shadow.jsonl"
run mkdir -p "$OUT" "$TMP"

# 2. JevGym decisions (gold), split by task
run "$PY" -m jevharness.training.harvest --runs "$V2" "$V1" --out "$OUT" --dev-frac "$DEV_FRAC" --seed "$SEED" \
  --min-jevgym-seed 1000

# 3. When2Call-train replay (gold; never the test split)
run "$PY" -m jevharness.training.harvest --when2call-train --when2call-max "$W2C_N" --out "$TMP/w2c" --dev-frac 0.1 \
  --seed "$SEED"
run "$PY" -m jevharness.training.data mix --in "$TMP/w2c/train.jsonl" --out "$OUT/replay_when2call.jsonl.gz" --seed "$SEED"
run "$PY" -m jevharness.training.data mix --in "$TMP/w2c/dev.jsonl" --out "$OUT/dev_when2call.jsonl.gz" --seed "$SEED"

# 4. stand-in teacher soft labels on the JevGym TRAIN decisions (dev stays gold)
CACHE=cache/teacher_v0.sqlite TEACHERS="$TEACHER" IN="$OUT/train.jsonl" LABELED="$OUT/train_teacher_standin.jsonl" \
  MIN_AGREE=1 ALPHA=0.3 GOLD_FILTER=gold ON_DISAGREE=gold WORKERS=8 DO_TRAIN=0 SEED="$SEED" \
  bash "$(dirname "$0")/20_teacher_soft.sh"

# 4b. the stand-in teacher's metrics on DEV (reference of the T3 gate "R3 >= 90% of teacher acc"); only the summary
#     is kept, dev.jsonl stays gold
targs=()
while IFS= read -r t; do
  targs+=(--teacher "$t")
done < <(split_list "$TEACHER" ";")
run "$PY" -m jevharness.training.teacher_label --in "$OUT/dev.jsonl" --out "$TMP/dev_teacher_standin.jsonl" "${targs[@]}" \
  --alpha 0.3 --min-agree 1 --gold-filter keep --on-disagree gold --cache cache/teacher_v0.sqlite --workers 8
run cp "$TMP/dev_teacher_standin.jsonl.summary.json" "$OUT/dev_teacher_standin.summary.json"

# 5. counterfactual MC Q-values (Flash-Lite executor at T=0.7, v1 states; tiny: a pipeline / sanity set)
if [[ "$MC" == 1 ]]; then
  run "$PY" - "$V1/experiment.yaml" "$TMP/experiment_mc_lite.yaml" <<'EOF'
import sys, yaml
exp = yaml.safe_load(open(sys.argv[1]))
exp["llm"] = {"provider": "gemini", "model": "gemini-3.5-flash-lite", "temperature": 0.7, "reasoning_effort": "minimal"}
yaml.safe_dump(exp, open(sys.argv[2], "w"), sort_keys=False)
EOF
  RUNS="$V1/shadow.jsonl" EXP="$TMP/experiment_mc_lite.yaml" MAX_STATES="$MC_STATES" K_ALT="$MC_K" M="$MC_M" \
    WORKERS=8 OUT="$OUT/mc_standin.jsonl" SEED="$SEED" bash "$(dirname "$0")/30_mc_rollouts.sh"
fi

# 6. trajectories + manifest
for d in "$V2" "$V1"; do
  name=$(basename "$d")
  run mkdir -p "$OUT/trajectories/$name"
  run cp "$d/experiment.yaml" "$OUT/trajectories/$name/experiment.yaml"
  if [[ "$DRY_RUN" != 1 ]]; then
    gzip -9 -n -c "$d/shadow.jsonl" > "$OUT/trajectories/$name/shadow.jsonl.gz"
  else
    printf '+ gzip -9 -n -c %q > %q\n' "$d/shadow.jsonl" "$OUT/trajectories/$name/shadow.jsonl.gz"
  fi
done
margs=(--max-mb "$MAX_MB")
if [[ -n "$NOTES" ]]; then
  margs+=(--notes "$NOTES")
fi
run "$PY" "$ROOT/scripts/train/_manifest_v0.py" "$OUT" "${margs[@]}"
log "done -> $OUT (MANIFEST.json); commit it: data/train_v0 is not gitignored"
