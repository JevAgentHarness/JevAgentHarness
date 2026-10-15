# Plan: an exploratory SFT training ablation of the decision model (GPU only, no LLM API)

Written 2026-10-05 by the recipe-takeaways session. **Nothing here has been launched.** The AWS session is expired;
the orchestrator gets a new login before any step below. Context and the existing evidence:
[`recipe_takeaways.md`](recipe_takeaways.md) (T1-T3 name the questions the records leave open).

## 1. Questions and what stays fixed

The records support "SFT on visited states carries the gain" but cannot say *which ingredient of data v1* did it:
v0 -> v1 changed the states, the size, the task setting and the labels at once (T1), no gold-only or teacher-only
model exists (T2), and every full-data checkpoint saturates v1 dev (99.7 %) and the development slice (T6). Three
factors, each varied with everything else held at the SFT v1 recipe:

| factor | question | arms |
|---|---|---|
| (i) data size | how many harness episodes does the recipe need? | D10, D25, D50 vs D100 (nested task subsets) |
| (ii) label source | does the teacher's soft mass matter, given that every v1 target already has the gold option as its argmax? | gold only / teacher only / mixed (as v1), at 100 % and at 25 % |
| (iii) state distribution | do the states after an off-plan step matter beyond their count? | S_canon (no earlier off-plan step) vs S_match (random rows, same size) |

Fixed (the SFT v1 recipe of `results/gpu_round5/TRAINING_MANIFEST.md` §3 stage 1, `scripts/aws/gpu_round5_rl.sh
stage_sft`): Qwen3-4B (HF snapshot `1cfa9a7208912126459214e8b04321603b3df60c`), LoRA r 32 / alpha 64 on all linear
layers, soft cross-entropy + 0.5 Brier, fresh option permutation per row, LR 1e-4 cosine, warm-up 3 %, 2 epochs,
effective batch 64 rows, token-budget micro-batches <= 3,072 padded tokens, no activation checkpointing, bf16,
`MAX_LEN` 4,096 (prompt tail kept), held-out gate set `data/train_v0/dev.jsonl` during training, trainer seed 0
(replicates: 1). The When2Call replay keeps SFT v1's share: `REPLAY_FACTOR = 0.45 x rows / 15,694`. Epochs are fixed,
so smaller arms also get fewer optimizer steps (D10: about 55 vs 536); that is the recipe as a user would run it on
less data, and it is stated with every size result (an optional matched-steps arm is listed in tier 2).

## 2. What can be scored without an LLM

| instrument | content | labels | LLM needed? | use |
|---|---|---|---|---|
| v1 dev / dev_recovery / dev_knockout (`data/train_v1/*.jsonl.gz`, committed) | 2,377 / 606 / 656 decisions on held-out training-seed tasks (30006-30597), all three decision types | JevGym oracle gold | no (`eval_decisions`, offline) | accuracy by decision type, after an off-plan step, on-plan, knockout; raw ECE (development numbers) |
| common-state corpus C = S2-3's corpus (`results/stage2/s23/`, committed; A7.10 (c)) | 3,939 distinct tool-selection questions, 5,314 occurrences, 200 fresh Stage-2 tasks (seeds 300000-300199), 386 occurrences after an off-plan step | oracle acceptable sets stored in `occurrences.jsonl.gz` | no: the served checkpoint answers (`scripts/stage2_corpus.py replay`); Jev's and the published rows' answers are already committed | tool-selection accuracy (task mean; after an off-plan step), ECE with the arm's temperatures; paired task bootstrap against the reference arm |
| tau2 reference-agreement corpus (`results/stage2/t2ref/`, A7.13 (iv)) | 22,704 label-free tau2 questions (airline test, retail) | reference-trajectory agreement labels (not oracle accuracy) | no (served replay, `scripts/tau2_reference_replay.py`) | optional tier 2: out-of-generator agreement for three arms |
| E2 development slice, v2 stress slice, any live arm | live episodes | - | **yes** (Gemini System-2) | **not used** |

Note: the task brief's "common-state corpus C of `results/stage2/t2ref/`" refers to two corpora. C is
`results/stage2/s23/` (S2-3); `results/stage2/t2ref/` is the tau2 corpus. Both are scorable without an LLM; C is the
primary instrument, t2ref is optional.

## 3. Run matrix

Row counts are computed offline from the committed data (`python scripts/train/ablation_subsets.py plan`, seed 0);
"SFT min" scales SFT v1's measured 134 min for 16,902 rows on one L40S (`results/gpu_round5/status.txt`) by the
arm's rows incl. replay, plus 3 min load / save; "eval min" = 28 (section 5).

**Tier 1 (13 checkpoints, 12 trainings).**

| # | arm | factor | rows (selector) | labels | tasks | after off-plan | replay factor | rows + replay | trainer seed | SFT min | eval min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `sft_v1` (anchor) | - | existing SFT v1 checkpoint, no training | mixed | 587 | 30.9 % | 0.45 | 16,902 | (0) | 0 | 28 |
| 1 | `D100` | reference | all 15,694 rows, retrained on the ablation host | mixed | 587 | 30.9 % | 0.4500 | 16,902 | 0 | 137 | 28 |
| 2 | `D25` | size | 3,831 rows of 147 tasks (25 % of each block) | mixed | 147 | 36.4 % | 0.1098 | 4,126 | 0 | 36 | 28 |
| 3 | `L_gold_25` | labels | as D25 | gold one-hot | 147 | 36.4 % | 0.1098 | 4,126 | 0 | 36 | 28 |
| 4 | `L_teach_25` | labels | as D25 | Jev only (alpha 0) | 147 | 36.4 % | 0.1098 | 4,126 | 0 | 36 | 28 |
| 5 | `S_canon` | states | 10,851 rows with no earlier off-plan step | mixed | 587 | 0 % | 0.3111 | 11,686 | 0 | 96 | 28 |
| 6 | `S_match` | states | 10,851 random rows (seed 0) | mixed | 587 | 30.6 % | 0.3111 | 11,686 | 0 | 96 | 28 |
| 7 | `L_gold_100` | labels | all 15,694 rows | gold one-hot | 587 | 30.9 % | 0.4500 | 16,902 | 0 | 137 | 28 |
| 8 | `L_teach_100` | labels | all 15,694 rows | Jev only (alpha 0) | 587 | 30.9 % | 0.4500 | 16,902 | 0 | 137 | 28 |
| 9 | `D10` | size | 1,627 rows of 59 tasks | mixed | 59 | 35.3 % | 0.0467 | 1,752 | 0 | 17 | 28 |
| 10 | `D50` | size | 7,586 rows of 293 tasks | mixed | 293 | 30.1 % | 0.2175 | 8,170 | 0 | 68 | 28 |
| 11 | `D10_s1` | seed noise | as D10 | mixed | 59 | 35.3 % | 0.0467 | 1,752 | 1 | 17 | 28 |
| 12 | `D25_s1` | seed noise | as D25 | mixed | 147 | 36.4 % | 0.1098 | 4,126 | 1 | 36 | 28 |

Order of execution = the table order (the anchor first, as the go / no-go of the pipeline; then the reference; then the
cheap 25 % label block, so that a stop half-way still answers (ii) at 25 %). The D10 / D25 / D50 subsets are nested and
stratified by data block (v0's 77 tasks, v1's 510). S_canon has a different decision-type mix (stop 743 vs 1,399 rows
in S_match: stop questions sit mostly after off-plan steps); per-type accuracies are reported for that reason.

**Tier 2 (optional, after tier 1 and only within the caps).**

| arm | what | GPU min |
|---|---|---|
| `D100_s1` | D100 with trainer seed 1 (seed noise at full size) | 165 |
| `S_nonm` / `S_nonm_match` | without the scripted near-miss arm (7,611 rows, 17.9 % after an off-plan step) vs 7,611 random rows | 2 x 96 |
| `D10_steps` | D10 data for SFT v1's 536 optimizer steps (`MAX_STEPS=536`), separating data size from update count | about 160 |
| t2ref for `D100`, `L_gold_100`, `L_teach_100` | out-of-generator agreement (does gold-only training transfer worse than teacher-labelled?) | 3 x 132 |

## 4. Exact commands

### 4.1 Local, before the launch (CPU only, $0)

```bash
# (a) teacher-only targets (alpha 0, no gold fallback) from the cached Jev answers of the data-v1 build. No key in the
#     environment: a cache miss cannot be billed (it fails); require calls_fresh == 0 and 15,694 kept rows.
env -u OPENROUTER_API_KEY -u GEMINI_API_KEY -u GEMINI_API_KEYS .venv/bin/python -m jevharness.training.teacher_label \
  --in data/cache/train_v1/_work/train_gold.jsonl --out data/cache/train_v1/train_teacher_only.jsonl \
  --teacher jev=jev-openrouter --alpha 0 --min-agree 1 --gold-filter keep --on-disagree gold \
  --cache data/cache/teacher_v1.sqlite --workers 8 --max-questions 16 --max-errors 1
python3 -c "import json; s=json.load(open('data/cache/train_v1/train_teacher_only.jsonl.summary.json')); \
  l=s['teachers']['jev']['ledger']; assert l['calls_fresh']==0 and s['outcomes']=={'kept':15694}, (l, s['outcomes'])"
gzip -k data/cache/train_v1/train_teacher_only.jsonl && sha256sum data/cache/train_v1/train_teacher_only.jsonl.gz
# (b) dry build of every tier-1 arm (row counts must equal section 3), then upload the teacher file
.venv/bin/python scripts/train/ablation_subsets.py build --out runs/train_ablation/data \
  --teacher-only data/cache/train_v1/train_teacher_only.jsonl
env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws s3 cp data/cache/train_v1/train_teacher_only.jsonl.gz \
  s3://<bucket>/train_ablation/inputs/ --region us-east-1
# (c) commit this plan, the Deviations entry (section 8), the builder / scorer and the driver; then
bash scripts/aws/train_ablation.sh push        # bundle of the committed branch + the driver -> s3://.../code/
```

`train_gold.jsonl` (gitignored, local) is the pre-label version of `data/train_v1/train.jsonl.gz`, row for row
(same ids and order; the builder's gold relabelling reproduces its targets exactly, checked 2026-10-05).

### 4.2 Box side, per arm (`$W=/opt/jev`, checkout `$W/stage2`, `HV=$W/hvenv`, `BASE=$(cat $W/base_model_path.txt)`)

```bash
. "$HV/bin/activate"; A=runs/train_ablation; ACK=$W/ck_abl; PORT=8082
# data (deterministic from the committed data/train_v1 + the uploaded teacher file; sha256 checked)
python scripts/train/ablation_subsets.py build --out $A/data --teacher-only $W/inputs/train_teacher_only.jsonl.gz
# C: a copy of the committed corpus (answers go next to it; results/stage2/s23 stays untouched)
mkdir -p $A/s23 && cp results/stage2/s23/{corpus.json,occurrences.jsonl.gz,questions.jsonl.gz} $A/s23/
# SFT (ARM in the matrix; data file = ARM without "_s1"; SEED = 1 for *_s1, else 0)
RF=$(python -c "import json; print(json.load(open('$A/data/manifest.json'))['arms']['$DATA']['replay_factor'])")
GPUS=1 RECIPE=r3 MODEL="$BASE" TRAIN=$A/data/$DATA.jsonl DEV=data/train_v0/dev.jsonl \
  REPLAY=data/train_v1/replay_when2call.jsonl.gz REPLAY_FACTOR=$RF OUT=$ACK/$ARM LR=1e-4 EPOCHS=2 BATCH=2 \
  EFF_BATCH=64 MAX_LEN=4096 SEED=$SEED PY=python \
  SFT_EXTRA="--max-batch-tokens 3072 --eff-batch 64 --no-gradient-checkpointing" bash scripts/train/10_sft.sh
# development read-outs (raw) and the temperature fit (v1 dev, as for SFT v1 and the frozen model)
python -m jevharness.training.eval_decisions --model "$BASE" --adapter $ACK/$ARM \
  --dev data/train_v1/dev.jsonl.gz data/train_v1/dev_recovery.jsonl.gz data/train_v1/dev_knockout.jsonl.gz \
  --n-perm 0 --max-len 4096 --batch-size 16 --out $A/eval/${ARM}_v1.json
python -m jevharness.training.calibrate --model "$BASE" --adapter $ACK/$ARM --dev data/train_v1/dev.jsonl.gz \
  --out $A/cal/${ARM}_calibration.json --min-bucket 200
# C through serve_jev with the arm's temperatures; served name abl-<arm> keeps the replay cache entries apart
python -m jevharness.training.serve_jev --model "$BASE" --adapter $ACK/$ARM --calibration $A/cal/${ARM}_calibration.json \
  --name abl-$ARM --port $PORT &            # wait for /health (as explore_gpu.sh stage_serve_rl)
for k in 1 2 3; do python scripts/stage2_corpus.py replay --corpus $A/s23 --name abl_$ARM \
  --s1 "{\"type\": \"jev\", \"name\": \"abl-$ARM\", \"base_url\": \"http://127.0.0.1:$PORT\", \"model\": \"abl-$ARM\", \"api_key\": \"\"}" \
  --cache cache/s1_ablation_replay.sqlite; done   # resumes; three invocations as stage2_gpu.sh replay()
kill %1
```

The anchor (`ARM=sft_v1`) skips SFT and uses `$W/ck/sft_v1` (sha256 `6736730a...` of `FREEZE_RECORD.json`
`attribution_rows_A7_13_i.sft_v1`, re-checked). Tier-2 t2ref: `scripts/tau2_reference_replay.py replay --corpus
$A/t2ref --name abl_<arm> ...` on a copy of `results/stage2/t2ref/{corpus.json,questions.jsonl.gz,occurrences.jsonl.gz}`,
then `score --corpus $A/t2ref --rows abl_D100,abl_L_gold_100,abl_L_teach_100 --tex $A/t2ref_abl.tex` (never the
default `--tex`, which is the paper's `generated/t2ref.tex`).

### 4.3 Scoring (local, after `aws s3 sync s3://.../train_ablation runs/train_ablation`)

```bash
python scripts/train/ablation_score.py --run runs/train_ablation --out results/training/ablation \
  --tex paper/generated/recipe_ablation.tex
```

## 5. GPU hours and cost on one L40S (g6e.2xlarge, $2.24208 / h on demand, `results/stage2/tariff.json`)

Evidence for the per-step times (all on one L40S): SFT v1 134 min for 16,902 rows x 2 epochs (round 5, g6e.4xlarge,
token-budget batching); `eval_decisions` on the three v1 files 206 + 54 + 85 s (`results/gpu_round5/eval/
eval_sft_v1_v1.json`) + model load, about 7 min; `calibrate` about 3 min (round 5 `base_control` stage: 9 min with its
evaluations); C replay 16-17 min per checkpoint on the g6e.2xlarge Stage-2 host (`results/stage2/gpu/status.txt`:
`replay_rl` 16 min, the SFT v1 replay 17 min) + 1 min server start; t2ref about 132 min per checkpoint (same file).
Per checkpoint eval = 7 + 3 + 18 = 28 min.

| block | GPU hours | $ (g6e.2xlarge) |
|---|---|---|
| tier 1: 12 trainings (849 min) + 13 evaluations (364 min) + setup (0.5 h) | 20.7 | 46.45 |
| tier 2: D100_s1, S_nonm pair, t2ref for 3 arms (D10_steps not included) | 12.6 | 28.14 |
| all | 33.3 | 74.59 |

If the host is still a g6e.16xlarge ($7.57719 / h, one L40S; the resize to g6e.2xlarge awaits the user's OK), tier 1
costs about $157 for the same GPU hours. API cost: $0 by construction (no key is loaded on the box; section 6).
Wall clock: tier 1 runs sequentially in about 21 h on one GPU. A 4-GPU host (g6e.12xlarge) could run four arms in
parallel (`CUDA_VISIBLE_DEVICES`, four serve ports) in about 6 h for the same GPU hours, at its own hourly price.

## 6. Driver stage plan

New driver `scripts/aws/train_ablation.sh`, built from the skeleton of `scripts/aws/explore_gpu.sh` (local side `push`
/ `send` / `status`; box side preamble, `st`, `sync_run`, `itype`, `arm_stops`, `start_rec`, and the serve / health
loop of `stage_serve_rl`) plus `stage2_gpu.sh`'s three-invocation `replay()` and `gpu_round5_rl.sh`'s `sync_ck`.

- **Host:** the persistent Stage-2 GPU host `<instance-id>` (tag `jev-stage2-startlux`; one L40S) after the
  resize to g6e.2xlarge. It already has `$W/hvenv` (editable install of `$W/stage2`; torch 2.14.1 cu130,
  transformers 5.18.0, peft 0.21.2, accelerate 1.15.0: `results/stage2/gpu/hvenv_pip_freeze.txt`), the pinned base
  snapshot (`$W/base_model_path.txt`) and `$W/ck/sft_v1`. Code: the committed branch as `code/train_ablation.bundle`
  checked out in `$W/stage2` (other agents are changing `serve_jev.py` / `decision_model.py` today, so the anchor's
  reproduction check below is the guard against a serving or readout change).
- **No keys:** the driver never calls `keys()`; its preamble `unset GEMINI_API_KEY GEMINI_API_KEYS OPENROUTER_API_KEY`
  and `abl_final` asserts that every replay record has `cost_usd == 0`.
- **Guards:** `GUARD_MIN=2400` (40 h power-off), `IDLE_MIN=60`; a GPU-hour cap of 36 h checked before each arm
  (host start time from `START_train_ablation.txt`); shutdown behaviour stop.

| stage | does | writes |
|---|---|---|
| `rearm` | `arm_stops`, instance type, `nvidia-smi` | `$A/instance_type.txt`, `nvidia_smi_start.txt` |
| `abl_inputs` | fetch `train_ablation/inputs/train_teacher_only.jsonl.gz`, check sha256; `ablation_subsets.py build`; copy C (and t2ref) with sha256 against the committed files | `$A/data/manifest.json`, `$A/inputs.sha256` |
| `abl_anchor` | sha256 of `$W/ck/sft_v1`; eval / calibrate / serve / replay as 4.2 with `ARM=sft_v1`; **go / no-go of the pipeline:** v1-dev accuracy equals `results/gpu_round5/eval/eval_sft_v1_v1.json` (0.99706) within 0.001, the refitted temperatures match `results/stage2/gpu/rl_attr/sft_v1_calibration.json` (relative 1e-3, as `reproduction_check.json`), and the C answers agree with `results/stage2/s23/answers_sft_v1.jsonl` on >= 99 % of questions; otherwise stop and report | `$A/anchor_check.json` |
| `abl_smoke` | `D10` with `MAX_STEPS=4` (memory, throughput; checkpoint discarded) | `$A/logs/smoke.log` |
| `abl_arms "<list>"` | per arm, in the table order: `sft_<ARM>` (timed line `stage sft_<ARM> exit=0 (<m> min)` for the scorer), eval, calibrate, serve + replay, `sync_ck`; delete local `checkpoint-*` | `$A/eval/`, `$A/cal/`, `$A/s23/answers_abl_*`, `$A/logs/` |
| `abl_t2ref` (tier 2) | replay + score of the tau2 copy for the named arms | `$A/t2ref/`, `$A/t2ref_abl.tex` |
| `abl_final` | `cost_usd == 0` check over every replay record, `nvidia-smi`, final sync, power-off | `$A/final_check.json` |

**S3 (bucket `<bucket>`).** `train_ablation/` <- `runs/train_ablation/` (status, manifests,
eval / cal JSON, replay answers and `replay_abl_*.json`, logs, trainer states; not the arm data files, which are
rebuilt from committed data + the teacher input); `train_ablation/inputs/` (teacher-only file + summary);
`checkpoints/train_ablation/<arm>/` (adapter, adapter config, calibration, decision config, tokenizer files; no optimizer
state); `code/train_ablation.bundle`, `code/train_ablation.sh`. Results to commit afterwards:
`results/training/ablation/{ablation.json, ablation.md, manifest.json, status.txt, eval/, cal/, anchor_check.json}` and
the answer files gzipped.

## 7. Evaluation, analysis keys and reading rules

`scripts/train/ablation_score.py` (tested offline in `tests/test_train_ablation.py`; it reproduces the published C rows
of SFT v1, the frozen model, Jev and the base exactly from the committed answer files) writes
`results/training/ablation/ablation.{json,md}` and, with `--tex`, `paper/generated/recipe_ablation.tex`
(`main.tex` would need `\InputIfFileExists{generated/recipe_ablation}{}{}`):

- `recipe_abl/<arm>/{rows, tasks, sft_min}`;
- `recipe_abl/<arm>/{v1_acc, v1_tool_select, v1_tool_params, v1_stop, v1_recovery, v1_recovery_tool_select, v1_onplan}`
  (percent) and `v1_ece` (raw), `v1_ece_cal` when a calibrated evaluation exists;
- `recipe_abl/<arm>/{c_acc, c_acc_offplan}` (percent, task means) and `c_ece` (with the arm's temperatures);
- `recipe_abl/contrast/<name>/{v1_acc_diff, v1_recovery_diff, v1_recovery_tool_select_diff}` (points) and
  `recipe_abl/contrast/<name>/c_acc_{delta, lo, hi}` (points; paired task bootstrap over the 200 tasks of C, B = 2,000,
  seed 0) for the contrasts `size_{10,25,50}` (vs D100), `{gold,teacher}_vs_mixed_100` (vs D100),
  `gold_vs_teacher_100`, `{gold,teacher}_vs_mixed_25` (vs D25), `canon_vs_match`, `nonm_vs_match`,
  `seed_{10,25,100}`, `retrain_vs_anchor` (D100 vs the existing SFT v1) and `anchor_vs_published`.

Reading rules, fixed before the first training step (descriptive; no test, no multiplicity claim): a factor is
described as making a difference on C when its paired 95 % interval excludes 0 **and** its |delta| exceeds the
largest |delta| of the seed contrasts at a comparable size; v1-dev differences below 0.5 points are described as
within the saturated range (no per-row output, so no interval); the after-off-plan stratum of C (57 tasks) and v1
recovery are the primary read-outs for factor (iii), v1-dev ECE and C ECE for factor (ii). Every number is a
development number (v1 dev) or descriptive (C); none is a registered result, none selects a checkpoint, and no
ablation checkpoint is served in any other experiment.

## 8. Deviations entry (draft; to append to `docs/PREREG_A7_STAGE2.md` "Deviations" before the first training step)

- 2026-10-0X, recorded about HH:MM UTC, before any ablation training step and before any ablation checkpoint answers a
  question. **Exploratory row E3: an SFT training ablation of the decision model** (`results/training/ablation_plan.md`).
  Outside both registered stages; descriptive, labelled "exploratory (E3)"; it revises no registered result, freezes no
  checkpoint and changes nothing in JevAgentRL (A7.10) or JevAgentRL-v2 (A7.19). (a) *Purpose.* The freeze records
  show that SFT on data v1 carries JevAgentRL's development accuracy, but data v1 differs from data v0 in states, size,
  task setting and labels at once; E3 varies data size (task subsets of 10, 25, 50 and 100 %), label source (gold
  one-hot, Jev alone, the v1 mixture; at 100 and 25 %) and state distribution (rows without an earlier off-plan step vs
  a size-matched random subset), with two seed replicates. (b) *Data and seeds.* Every training row is a row of
  `data/train_v1/train.jsonl.gz` (sha256 `9c0450c2...`), selected and relabelled by `scripts/train/ablation_subsets.py`
  (seed 0); no episode is run and no JevGym seed is drawn, so every training seed stays the data-v1 seed (>= 10000 and
  outside 100000-100499, 200000-200499, 300000-300999 and 5000-5999; `training/seed_guard.TRAIN_FORBIDDEN`, re-checked
  by the builder). The Jev-only targets are rebuilt from the cached Jev answers of the data-v1 build with no key in the
  environment (`calls_fresh` 0, $0). (c) *Training.* The SFT v1 recipe unchanged (round-5 manifest §3 stage 1:
  Qwen3-4B at snapshot `1cfa9a72...`, LoRA r 32 / alpha 64, soft CE + 0.5 Brier, option permutation, LR 1e-4 cosine,
  warm-up 3 %, 2 epochs, effective batch 64, token budget 3,072, bf16), replay factor 0.45 x rows / 15,694, trainer
  seed 0 (replicates 1). (d) *Evaluation, GPU only, no API.* `eval_decisions` on v1 dev, dev_recovery and dev_knockout
  (training-side development data); temperatures fitted on v1 dev (`training/calibrate.py --min-bucket 200`); the
  common-state corpus C (`results/stage2/s23/`, built from Stage-2 seeds 300000-300199 and already analysed) replayed
  by `scripts/stage2_corpus.py replay` into a copy of the corpus with its own cache (`cache/s1_ablation_replay.sqlite`);
  optionally the tau2 reference-agreement corpus (A7.13 (iv)) for three arms. No live episode, no tau2 simulation, no
  LLM call. C is used post hoc and only descriptively: no checkpoint, threshold or recipe is selected on it.
  (e) *Host and caps.* One L40S (the Stage-2 GPU host, g6e.2xlarge); GPU <= 36 host hours (expected about 21 for
  tier 1); API $0 (any priced record stops the run). The existing SFT v1 checkpoint is first re-evaluated through the
  same pipeline; if it does not reproduce its recorded v1-dev accuracy (0.99706 within 0.001), its temperatures
  (relative 1e-3) and >= 99 % of its C answers, E3 stops. (f) *Analysis.* `scripts/train/ablation_score.py` ->
  `results/training/ablation/ablation.{json,md}`, keys `recipe_abl/...`: per arm v1-dev accuracy by decision type,
  after an off-plan step and on-plan, raw ECE; C accuracy (all, after an off-plan step) and ECE; contrasts as listed in
  the plan with paired task-bootstrap intervals on C (B = 2,000, seed 0) and the reading rules of the plan §7.
  Descriptive only; outside every Holm table.

## 9. Pre-launch checklist

1. New AWS login (the session is expired); confirm the host's instance type and the user's OK for the g6e.2xlarge
   resize.
2. Section 4.1 (a)-(b) locally: `calls_fresh == 0`, 15,694 kept rows, arm row counts as in section 3.
3. Write `scripts/aws/train_ablation.sh` (section 6), `bash -n` it, run it with `DRY_RUN=1` for one arm.
4. Commit the plan, the Deviations entry, the driver, `scripts/train/ablation_{subsets,score}.py` and their test; run
   `python3 scripts/check_no_secrets.py`; then `push`, `send "rearm abl_inputs abl_anchor"`, read `anchor_check.json`,
   then `send "abl_smoke abl_arms '<tier-1 list>' abl_final"`.
5. After the run: sync, score, commit `results/training/ablation/`, add the host window to `scripts/spend_ledger.py`'s
   GPU lines.

## 10. Risks

- **Saturation.** Full-data arms may all sit at 99.7 % (v1 dev) and 99.2 % (C); the 25 % and 10 % arms and C's
  after-off-plan stratum carry the signal. That is why labels are crossed with 25 %.
- **Stack drift.** The ablation host's library versions and today's code changes differ from round 5's; D100 retrained
  on the same host is the reference of every contrast, and the anchor check stops the run on a readout change.
- **Teacher file.** If the cached relabel misses any request, the teacher arms are skipped (the builder refuses rows
  without an alpha-0 target); gold and size arms still run.
- **Compute confound.** Fixed epochs give small arms fewer updates (section 1; `D10_steps` in tier 2 separates them).

## 11. Addendum 2026-10-05 (about 20:10 UTC): budget split, host, 25 % state pair, cap

Written by the E3 preparation session before any ablation training step, before any ablation checkpoint answered a
question and before the anchor ran; nothing above is edited. Where this section and sections 1-10 differ, this section
holds.

**11.1 Done locally (section 4.1).** (a) Teacher-only relabel from the cache with no key in the environment
(`env -u OPENROUTER_API_KEY -u GEMINI_API_KEY -u GEMINI_API_KEYS -u OPENAI_API_KEY ...`): 15,694 kept rows, 5,175
cached calls, `calls_fresh` 0, $0; Jev's argmax differs from gold on 864 rows (no gold fallback, as intended). The file
was gzipped with `gzip -n` (reproducible header). (b) Dry build of every tier-1 arm: row counts equal section 3, plus
the two new arms below. (c) D100's training file as 10_sft.sh writes it (D100 rows + 0.45 x When2Call replay,
`data.py mix`, seed 0) is byte-identical to SFT v1's stored `train_mix.jsonl` (s3 `checkpoints/round5/sft_v1/`,
sha256 `9fe1088d...`): D100 reproduces SFT v1's training input exactly. Every hash (teacher file, each arm file, each
arm's training mix incl. the seed-1 replicates) is in `results/training/ablation_inputs.json`; the driver checks the
arm files at `abl_inputs` and each arm's training mix after its SFT (a mismatch fails the arm).

**11.2 Two new arms: the state pair at 25 %** (`scripts/train/ablation_subsets.py`, seed 0, nested and deterministic).

| arm | rows (selector) | tasks | after off-plan | stop rows | replay factor | rows + replay | SFT min (L40S) |
|---|---|---|---|---|---|---|---|
| `S_canon_25` | D25's rows with no earlier off-plan step (= S_canon restricted to D25's tasks) | 147 | 0 % | 154 | 0.0699 | 2,626 | 24 |
| `S_match_25` | a random subsample of D25's rows of the same size (2,438) | 146 | 36.7 % | 318 | 0.0699 | 2,626 | 24 |

New contrast `canon_vs_match_25` (S_canon_25 - S_match_25) in `scripts/train/ablation_score.py`, and, because the
core block below has no D100, `size_10_vs_50`, `size_25_vs_50` (within the ablation host) and
`size_{10,25,50}_vs_anchor` (vs the existing SFT v1 checkpoint: a cross-stack comparison, bounded by the anchor check).

**11.3 Host.** The Stage-2 GPU host <instance-id> is a g5.4xlarge (one A10G 24 GB, $1.624 / h) today; the
orchestrator may resize it to a g6e.2xlarge (one L40S 48 GB, $2.24208 / h) first. The driver works on both: `rearm`
reads the instance type and GPU and writes `host.json` (price, priors of the cap projection, the anchor's temperature
tolerance); every arm records the instance type it trained on.

**11.4 GPU hours and cost.** Assumptions: L40S as section 5 (SFT 134 min per 16,902 rows incl. replay + 3 min load /
save; evaluation 28 min per checkpoint = eval_decisions 7 + temperatures 3 + C replay 16.7 measured (the summed wall
time of `answers_sft_v1.jsonl`) + 1 server start; setup 0.5 h). A10G: training x 2.9 (bf16 dense tensor throughput,
about 362 vs 125 TFLOPS), + 30 % when activation checkpointing is needed (a sensitivity at + 50 %: with LoRA the frozen
weights get no weight gradients, so one recomputed forward is about half of the remaining work), + 5 min load / save;
evaluation and replay x 2.21 = the measured tool-selection service time per request, A10G 487 ms
(`results/harness_final/replay_bench.md`, Stage-2 server, c = 1) vs L40S 220 ms (`latency_diagnosis.md` section 4,
Stage-2 host), and the fast server x 0.758 (370 vs 488 ms per tool-selection request, same table) -> eval_decisions
15.5 + temperatures 6.6 + C replay 28.0 + server start 1.5 = 52 min per checkpoint (61 with the plain server); setup
0.75 h (smoke with up to two out-of-memory attempts). Cost = hours x on-demand price. The core includes the anchor's
evaluation and the setup.

| host | core (h / $) | extended (h / $) | all tier 1 (h / $) |
|---|---|---|---|
| L40S, g6e.2xlarge | 8.2 / 18.44 | 14.2 / 31.80 | 22.4 / 50.23 |
| A10G, g5.4xlarge, no checkpointing | 18.8 / 30.50 | 37.2 / 60.49 | 56.0 / 90.99 |
| A10G, checkpointing + 30 % | 21.9 / 35.64 | 46.4 / 75.42 | 68.4 / 111.06 |
| A10G, checkpointing + 50 % (sensitivity) | 24.1 / 39.07 | 52.6 / 85.37 | 76.6 / 124.44 |

Per arm, SFT + evaluation in minutes (L40S / A10G without / with checkpointing + 30 %): D25, L_gold_25, L_teach_25 64 /
152 / 180 each; S_canon_25, S_match_25 52 / 117 / 135 each; D10 45 / 97 / 109; D50 96 / 245 / 301; D25_s1 64 / 152 /
180; D10_s1 45 / 97 / 109; D100, L_gold_100, L_teach_100 165 / 445 / 562 each; S_canon, S_match 124 / 325 / 406 each.
On 24 GB the recipe's setting (3,072-token micro-batches without checkpointing; single rows reach the 4,096-token
`MAX_LEN`) probably runs out of memory (about 8 GB of bf16 weights plus roughly 4 MB of stored activations per token
without checkpointing), so the checkpointing rows are the likely A10G case: the core then exceeds the 20-hour target
by about 2 h. API: $0.

**11.5 Core and extended, execution order.** Core (answers all three questions at 25 % scale): the anchor (go /
no-go), then `D25`, `L_gold_25`, `L_teach_25`, `S_canon_25` + `S_match_25`, `D10`, `D50`. Extended (the rest of tier
1): `D25_s1`, `D10_s1`, `D100`, `L_gold_100` + `L_teach_100`, `S_canon` + `S_match`. Order within the core: the 25 %
label and state contrasts first (each needs D25 or its pair), then the size points (D25 and the anchor already give
one size contrast); D50, the costliest core arm, last. Order within the extended block: the seed replicates first
(cheap; they let the seed clause of the section 7 reading rules apply to the core's 25 % contrasts), then D100 (the
reference of every 100 % contrast), then the two 100 % pairs. `CORE` / `EXTENDED` in `ablation_subsets.py`; "+" binds
a pair into one cap unit (one arm of a pair alone answers nothing). Expected under the default cap: the L40S runs all
of tier 1 (22.4 h); the A10G runs the core plus D25_s1 and D10_s1 without checkpointing, and the core plus D10_s1 with
it (+ 30 %); at the + 50 % sensitivity D50 does not fit and the two seed arms run in its place.

**11.6 Cap.** `CAP_H` = 24 GPU-hours on either host by default (the orchestrator may raise it; on the A10G with
checkpointing, 26 secures the whole core at the + 50 % sensitivity). GPU hours = the summed duration of the E3 stages on
the box (waiting for another chain excluded). Before each cap unit, the driver projects used + the unit's SFT and
evaluation minutes: host priors (section 11.4) until an arm has finished, then the run's measured minutes per row incl.
replay and its measured evaluation minutes. A unit that does not fit is skipped and the next one is tried; a skipped
unit is reported as not run, never as a result. Power-off guard `(CAP_H + 4) x 60` minutes from the last `rearm`, idle
stop 60 min, power-off two minutes after `abl_final` (`POWEROFF=1`). Every finished arm is complete on its own (eval,
temperatures, C answers, adapter on S3) and the run is re-scored after each arm (`runs/train_ablation/score/`), so a
stop half-way leaves every finished arm scored.

**11.7 Memory fallback (fixed before any run).** The recipe's setting first (`nogc_3072`); on a CUDA out-of-memory
error, in this order: activation checkpointing with the same 3,072-token budget (`gc_3072`, same maths), then
checkpointing with 2,048, 1,536 and 1,024-token budgets (same 64 rows per optimizer step; the trainer derives the
accumulation, so the maths is the same up to float order and the rows grouped into each step). `abl_smoke` runs 4
optimizer steps on the 48 longest D100 prompts plus 48 random D10 rows and fixes the starting setting; an arm that
still runs out of memory moves to the next setting (later arms start there). LR, epochs, effective batch, loss, LoRA
rank and data never change. Each arm records its setting, attempts, accumulation and instance type in
`settings/<arm>.json` (also a column of the scorer's table). Evaluation and temperature scoring fall back from batch 16
to 8 and 4 on out-of-memory (`calibrate.py` gains a `--batch-size` flag, default 16 = the old behaviour).

**11.8 Serving.** `serve_jev --fast` for the C replay: the replay sends one question per request, and
`results/harness_final/replay_bench.md` shows no decision flip at c = 1 for the fast server against the Stage-2 server
on the same host (0 / 281 questions for v2 and v2p vs v0/old/1; largest probability change 0.19). The anchor check is
the guard: if under `--fast` only the anchor's C answers fail (v1 dev, temperatures and cost pass), the anchor is
replayed once with the plain Stage-2 server and, if that passes, every arm uses the plain server (`serve_mode.txt`;
the fast answers are kept under `s23/superseded/`). Each arm has its own replay cache (`cache/s1_abl_<arm>_<mode>.sqlite`).

**11.9 Anchor tolerance by host.** v1-dev accuracy within 0.001 of 0.99706 and >= 99 % agreement with the published C
answers (unchanged); the refitted temperatures within relative 1e-3 on an L40S (the GPU type of the recorded fit) and
within relative 2e-2 on another GPU type (bf16 kernels differ between GPU types, and a temperature fit on a
99.7 %-accurate set rests on few low-margin rows).

**11.10 Reading rules, added.** If the extended block does not run, the seed clause of section 7 cannot be applied: a
factor is then described by its paired interval alone, and the text says that seed noise was not measured at that
size. Contrasts against the anchor are cross-stack (other host and library versions); `retrain_vs_anchor` (D100 vs the
anchor) measures that gap when D100 runs. Nothing else in section 7 changes.

**11.11 Launch (orchestrator).** (1) Append `results/training/ablation_deviation_draft.md` to the Deviations of
`docs/PREREG_A7_STAGE2.md` with host and time filled in, regenerate `docs/STAGE2_EFFECTIVE.md`, commit, then `bash
scripts/aws/train_ablation.sh push` (the box refuses `abl_smoke` and every arm stage without the "Exploratory row E3"
entry in its checkout). (2) `bash scripts/aws/train_ablation.sh send "rearm abl_inputs abl_anchor"`; read
`s3://<bucket>/train_ablation/anchor_check.json`. (3) `bash scripts/aws/train_ablation.sh send
"rearm abl_smoke abl_core abl_ext abl_final"`. One send `"rearm abl_inputs abl_anchor abl_smoke abl_core abl_ext
abl_final"` is equivalent (a NO-GO anchor stops the chain before any training). Start every send with `rearm` (a
stopped and restarted box loses the scheduled power-off). Send-time variables: `CAP_H`, `GUARD_MIN`, `IDLE_MIN`,
`SERVE_FAST`, `POWEROFF`, `WAIT_PREV` (default 1: the chain waits for another driver's chain to end before it touches
the checkout or the GPU). (4) Afterwards: `aws s3 sync s3://.../train_ablation runs/train_ablation`, score (section 4.3),
commit `results/training/ablation/`, add the host window to `scripts/spend_ledger.py`.

Not in the driver: the tier-2 arms and the t2ref replay (section 3, tier 2).
