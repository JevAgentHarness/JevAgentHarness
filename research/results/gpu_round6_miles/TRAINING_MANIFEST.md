# JevAgentRL-v2 training manifest (GPU round 6, the Miles track; exploratory, PREREG amendment A7.19)

Written and committed on 2026-10-03 (~03:15 UTC) by the operator of the v2 chain, before any v2 training step (no Miles
training step, not even the 4-step smoke, has run) and before any Stage-1 or Stage-2 evaluation output was opened by
the operator. JevAgentRL-v2 is a second, exploratory post-training track on the Miles RL framework
(https://github.com/radixark/miles, read at commit `49576fd`). It never replaces the frozen v1 JevAgentRL
(`results/gpu_round5/TRAINING_MANIFEST.md`, branch `<worktree>`); S2-3 and S2-4 stay with v1.
Plan and design: [`docs/MILES_PORT.md`](../../docs/MILES_PORT.md). The recipe configs and scripts named below are
those of the commit that adds this file (`git log -1 --format=%H -- results/gpu_round6_miles/TRAINING_MANIFEST.md`);
the GPU host runs that commit's bundle or a later commit of the same branch whose changes are listed in §9
(infrastructure repairs only). From this commit on, `scripts/train/r6_chain.py`, not a person, picks the base, keeps
or drops each stage, retains one continuation per stage and names the frozen checkpoint.

## 1. Actors, workspace and access controls (A7.10 (b), applied to v2)

- **Operator / developer:** the JevAgentRL-v2 training operator (a separate working copy
  of this repository, apart from the coordinating operator).
  It takes every v2 model-development decision until the freeze record exists and does not open Stage-1 or Stage-2
  evaluation outputs. The coordinating session reads Stage-1 results; any later message from it that would change the
  chain (stages, configs, thresholds, checkpoint choice) is recorded in §9 as a dated deviation and makes every
  comparison involving the resulting checkpoint exploratory; operational messages (budget, stop / start, restarts)
  are not deviations. The coordinator's instruction of 2026-10-03 ~03:00 UTC to add a harder development instrument
  (§4.2) arrived before this manifest and is part of it.
- **Workspace:** the worktree above and the GPU host tagged `Name=jev-rl-miles`, `Project=jev` (us-east-1; type in §3).
  The host receives only the git bundle of this branch (`s3://<bucket>/code/jev-rl-miles.bundle`),
  `code/gpu_round6_miles.sh` / `code/r6_driver.sh`, the Miles Docker image (`radixark/miles:latest`, digest recorded in
  `runs/gpu_round6_miles/miles_image_digest.txt`), the pinned model snapshots (§2) and HF / API access; it writes only
  `gpu_round6_miles/` and `checkpoints/round6_miles/` in the bucket. The instance role can technically read the whole
  bucket; the separation is procedural: the box script reads no other prefix, and the operator never syncs main-run
  directories, `results/primaries/`, `results/stage2/` or `results/main/` to the host.
- **Not allowed to the operator before the freeze:** anything under the main tree's `runs/` except `runs/dev*`,
  `runs/pilot*` and `runs/gpu_round3`; `results/primaries/`, `results/stage2/`, `results/main/`; any Stage-1 or Stage-2
  trajectory, score card, primary, profile selection, failure example or aggregate; seeds 100000-100499,
  200000-200499, 300000-300999, and the calibration seeds 5000-5149 (v2 runs no calibration arm before the freeze).
- **Access record.** Before this manifest the operator read: the documents named in the task brief (SCOPE_V2,
  the project rules, TRAINING_RECIPES §16-18, PREREG_A7_STAGE2 A7.10-A7.13, results/gpu_round3/NOTES.md), v1's manifest and
  its committed development artefacts on the v1 branch (round 5 `eval/` gate files, the SFT v1 gonogo, the E2
  development-slice summaries on seeds 1100-1149, `status.txt`, `NOTES.md`; all development data), Miles' public
  repository and documentation, and the public StartLux-Decision model card. No Stage-1 / Stage-2 output was opened.
  v1's branch code was merged into this branch (read only; nothing on v1's side changed).

## 2. Training inputs (the whitelist)

| input | content | notes |
|---|---|---|
| data v0 | `data/train_v0/dev.jsonl` = the **210-decision gate set**; `dev_when2call.jsonl.gz` (flip rate) | evaluation of G1 / G2 only (v2 trains on v1 data) |
| data v1 | `data/train_v1/{train,replay_when2call}.jsonl.gz` (training); `dev.jsonl.gz`, `dev_recovery.jsonl.gz`, `dev_knockout.jsonl.gz`, `dev_teacher.summary.json` (gates) | unchanged since 9271976 |
| v1 references | `results/gpu_round5/eval/gate_intern4b_v1dev.json` (Intern-Decision-4B on v1 dev: G3's actual reference), `results/gpu_round2/gate_verdict.json` (G1 / G2 references); v1's development numbers (round 5 `eval/`, `e2dev/`) for the ablation table only | committed development artefacts |
| training seeds | JevGym seeds >= 10000 outside 100000-100499, 200000-200499, 300000-300999 and 5000-5999 (`training/seed_guard.py`, checked when a stage config loads): OPD pool **60000-63999**, GRPO pool **70000-72999** (disjoint from data v0 10000-12059, data v1 30000-30599, v1's OPD 40000-43999 and GRPO 50000-52999; test-enforced) | generated on the host |
| development seeds | 0-4999 outside 5000-5149: the **stress instrument 2000-2199** (§4.2); v1's E2 development slice **1100-1149** (`configs/experiments/jevgym_dev_rl.yaml`, descriptive comparison with v1 only) | |
| models | `Qwen/Qwen3-4B` (snapshot `1cfa9a7208912126459214e8b04321603b3df60c`); `startlux-models/StartLux-Decision-4B` (snapshot `3890ae0c174215e1555cc94c882ef45c5fe87cad`, CC BY-NC 4.0: research use; a checkpoint trained from it inherits the licence); `internlm/Intern-Decision-4B` (stress reference for G6 only); Jev `typesafe/jev-1.13` via OpenRouter (OPD teacher; stress reference, descriptive); Gemini 3.5 Flash / Flash-Lite (System-2) | |

## 3. Recipe, budgets and the stage order

Host stages (`scripts/aws/gpu_round6_miles.sh box "<stages>"` + `scripts/aws/gpu_round6_stages.sh`), in this order:

| # | stage | what (fixed values) |
|---|---|---|
| 0 | `setup`, `smoke_engine` | project venv (evaluation side), the Miles image, pinned snapshots; per base: SGLang serving + option log-probs vs the HF forward on 32 v1-dev prompts (S2); kernel report (S3). No training. |
| 1 | `base_zero`, `smoke_train`, `base_rule` | zero-shot v1-dev accuracy of both bases through `serve_jev`'s prompt (S4); 4 Miles FSDP SFT steps per base with the decision loss (S1) + strict HF export; `r6_chain.py base` -> `base.json` (rule in §5). The smoke steps train nothing that is kept. |
| 2 | `sft`, `eval_sft`, `stress_sft`, `e2dev_sft` | **SFT v2** (`configs/train/miles_sft_v2.yaml`): from the base; data v1 train + 0.45 x When2Call replay (16,902 rows); soft CE + 0.5 Brier over the presented option codes; fresh option permutation per row and epoch; 2 epochs; 64 rows per optimizer step (529 steps); full fine-tuning under FSDP2 (8 ranks), LR 1e-5 cosine to 1e-6, warmup 3 %, Adam (0.9, 0.98), weight decay 0, clip 1.0, bf16, activation checkpointing, `max_tokens_per_gpu` 3072; export -> HF dir; gates (§4.1), stress (§4.2), E2 dev slice (ts_replace, ts_params_replace, ts_assist_k3; descriptive) |
| 3 | `stress_refs`, `zeroshot_row` | stress references: Intern-Decision-4B (G6) and Jev (descriptive), once; the untrained base on the gates and the stress slice (ablation row) |
| 4 | `opd`, `eval_opd`, `stress_opd` | **OPD v2** (`configs/train/miles_opd_v2.yaml`): init SFT v2; six arms (ts + params replace with stop guard; ts replace; assist k 3; cascade tau 0.9 on ts / params; ts guard + stop guard; Flash shadow on all three), executor Flash-Lite T 0.7 (Flash T 0.3 for the shadow arm), max 20 steps, 24 iterations x 36 tasks (6 per arm) x 1 rollout, rollout temperature 1; teacher Jev (`jev-openrouter`); target 0.7 Jev + 0.3 oracle gold, gold one-hot where Jev's argmax is not the gold; forward KL over the presented options; 512 fresh labelled visits + 256 rows of the previous iteration per iteration (shortfall resampled), fresh permutation per row; 12 optimizer steps per iteration (64 rows each), LR 3e-6 constant, warmup 3 %; FSDP2 on 8 ranks colocated with one SGLang engine per GPU; continuations **opd_i12**, **opd** (iteration 24) |
| 5 | `grpo_validity`, `chain`, `grpo_replay` | the validity tests on the training commit (§6); the chain names the GRPO init (§5); the on-host replay check (§6) |
| 6 | `grpo`, `eval_grpo`, `stress_grpo` | **GRPO v2** (`configs/train/miles_grpo_v2.yaml`): init = the retained continuation of the last kept stage; four arms (ts + params replace with stop guard; assist k 3; cascade tau 0.9; ts guard + stop guard), all with Flash-Lite T 0.7; 30 iterations x 8 tasks x G 8; PPO-clip 0.2 / 0.28 against the stored SGLang behaviour log-probabilities; no KL, no entropy term; group-normalised advantages + GiGPO (omega 1, gamma 0.95); error / zero-signal episodes dropped; per-episode loss mean; 16 episodes per optimizer step (4 steps per iteration); LR 1e-6 constant; continuations **grpo_i15**, **grpo** (iteration 30) |
| 7 | `chain`, `freeze`, `hashes`, `e2dev_final`, `ablation` | the decision (§5); the frozen continuation -> `checkpoints/r6/final`; temperatures per (type x option-count bucket) on v1 dev (`training/calibrate.py`, min bucket 200); freeze record (§7); E2 dev slice of the frozen model (six arms, descriptive); the ablation table |

**Host.** Preference g6e.12xlarge (4 x L40S 48 GB), else p4d.24xlarge, else g6e.24xlarge / g6e.48xlarge, else an
8 x 24 GB host (g5.48xlarge or g6.48xlarge), p5.48xlarge only if necessary; the first type with capacity in us-east-1
at launch is used. g6e.12xlarge, p4d.24xlarge, g6e.24xlarge, g6e.48xlarge and p4de.24xlarge had no capacity in any
us-east-1 zone at 02:20-02:52 UTC; the host is **`<instance-id>`, g5.48xlarge (8 x NVIDIA A10G 24 GB, 192 vCPU,
768 GiB), us-east-1d, on-demand $16.288 / h**, launched 2026-10-03 02:51 UTC. The configs above are set for it: SFT on
8 FSDP ranks; OPD / GRPO on 8 FSDP ranks colocated with one SGLang engine per GPU (Miles offloads the trainer between
phases); `max_tokens_per_gpu` 3072 (data v1 prompts: mean ~1.15k, p99 ~2.6k tokens with either tokenizer).
Shutdown behaviour stop, 30 h power-off guard, 45-min idle stop.

**Maximum budgets.** GPU **$400** (host $ / h x hours: 24.5 h of the g5.48xlarge). API **$100** in
all: OPD teacher hard stop **$8** (ledger, `budget_usd`); executors of the OPD / GRPO rollouts (planned ~$25), stress
runs (~$2-3 per checkpoint, references ~$4), E2 dev slices (~$10). If the API ledgers reach $100, no further live run
starts; a rule whose input is missing then fails (§5). Deadline: freeze record by **2026-10-07 12:00 UTC** (target
2026-10-05).

**Allowed infrastructure adaptations (no deviation; each recorded in §9 with its time):** the GPU layout
(`actor_gpus` 4-8 dividing the global batch, `rollout_gpus` 0-2, colocation) and `max_tokens_per_gpu` (1024-8192), activation checkpointing,
SGLang memory fraction and attention backend (never FA3 on non-Hopper GPUs), the host type in the order above,
restarting a stage from its last checkpoint after a crash, Miles / SGLang flags that change no objective value
(logging, timeouts, weight-sync transport). Never: hyper-parameters of §3 (LR, steps, batch in rows / episodes, targets,
arms, seeds, reward), the rules of §4-§5, the data. A stage that cannot finish within the budgets ends at its last
checkpoint, which is then its final continuation.

## 4. The development instruments

### 4.1 Standard gates G1-G5 (`python -m jevharness.training.data_v1 gonogo`, TRAINING_RECIPES §18.4)

| gate | rule | threshold |
|---|---|---|
| G1 minimum | gate-set acc >= 0.8 x Jev's gate acc (0.9714) | >= 0.7771 |
| G2 static parity | gate-set acc >= Intern-4B's gate acc (0.8619) - 0.01 | >= 0.8519 |
| G3 in-harness states | v1-dev tool_select acc >= Intern-4B's v1-dev tool_select + 0.05 and v1-dev overall >= Intern-4B's + 0.02 (actual Intern reference, round 5's file) | >= 0.7977 / >= 0.8631 |
| G4 vs teacher | v1-dev acc >= 0.8 x Jev's on tool_select / tool_params / stop (0.9163 / 0.9794 / 0.9865) | >= 0.7330 / 0.7835 / 0.7892 |
| G5 calibration | v1-dev ECE (raw) <= max(0.10, Intern-4B's v1-dev ECE) | <= 0.10 |

Read-outs per checkpoint X used by §5: V = v1-dev accuracy, Rc = `dev_recovery` accuracy, E = v1-dev raw ECE
(`eval_decisions`, raw temperature, through `serve_jev`'s prompt and the 52-option knockout). Evaluation-side commands
(`eval_decisions`, `serve_jev`, `jevbench`) run in the Miles image, which has the flash-linear-attention and
causal-conv1d kernels Qwen3.5-family models need for a usable speed (the project venv lacks causal-conv1d); the freeze
stage records a 64-row regression corpus of the serving path in the image and in the project venv and their
comparison (max |delta p| <= 0.02 expected; reported, not a gate).

### 4.2 The stress instrument (harder; development seeds only)

`configs/experiments/jevgym_stress_dev_miles.yaml`: JevGym **development seeds 2000-2199** (n = 200), generator
settings no training data used: families v2 (clarify + multi), **120 distractors** (31 near duplicates + 89 random;
training: 60), **40 noise lines** (20), **`multi` tasks with 5 customers** (3), strict, max 30 steps; about 128 tools,
so every tool choice goes through the serving knockout (three 52-option chunks). One arm: the checkpoint served by
`training/serve_jev.py` replaces System-2 on tool_select and tool_params (`ours_stress_ts_params_replace`); System-2 =
Gemini 3.5 Flash, T = 0, for the rest. Replace decisions are argmax decisions, so the read-outs do not depend on
temperatures. Read-outs per checkpoint X (`scripts/train/r5_summary.py`):

- S = task success (200 episodes);
- A = own-state tool_select accuracy (pooled decisions, against the oracle's acceptable set, as `jevbench report`);
- P = own-state tool_params accuracy (descriptive);
- R = mean call-count-penalised success, the GRPO reward formula of §6 on each episode (errored episodes excluded);
- C = (S + A) / 2 (tie-break only).

References on the same slice, once: Intern-Decision-4B (`intern_stress_ts_params_replace`, in-process; G6) and Jev
(descriptive). **G6 (stress vs Intern-Decision-4B):** A >= Intern's A + 0.05 and S >= Intern's S.

Every evaluated stage is reported on both instruments (decision-level G1-G5 read-outs and S / A / P / R).

## 5. Base, keep, retain and freeze (`scripts/train/r6_chain.py`, verbatim in its docstring)

1. **Base.** StartLux-Decision-4B iff S1 (4 Miles FSDP SFT steps of the decision loss with finite loss), S2 (SGLang
   option log-probabilities within 0.1 nats (max |delta| of the restricted log-softmax) of the HF forward on 32 v1-dev
   prompts), S3 (the fla gated-delta-rule and causal-conv1d kernels run on the GPU) and S4 (zero-shot v1-dev accuracy
   through `serve_jev`'s prompt >= 0.766 = 0.8 x Jev's 0.958, and >= Qwen3-4B's on the same rows); else Qwen3-4B, which
   must pass S1 and S2, otherwise the track stops (no-go report).
2. **SFT v2** is the base of the chain (tag `sft_v2`); kept when it was evaluated.
3. **OPD** continuations `opd_i12`, `opd`, each against SFT v2. PASS iff
   - keep: V >= SFT - 0.005, Rc >= SFT - 0.010, E <= SFT + 0.010, S >= SFT - 0.015, A >= SFT - 0.010; and
   - gain: [V >= SFT + 0.010 and Rc >= SFT] or S >= SFT + 0.030 or A >= SFT + 0.020.
4. **GRPO** continuations `grpo_i15`, `grpo`, trained from the retained continuation of the last kept stage, each
   against that init. PASS iff the validity tests passed and the on-host replay check passed (§6), V >= init - 0.010,
   Rc >= init - 0.010, E <= init + 0.020, R > init's R + 0.010, and S >= init - 0.010.
5. **Retention.** A stage is kept iff at least one continuation passes; among passing continuations the retained one
   has the highest C, then the highest V, then the later step.
6. **Freeze.** The retained continuation of the last kept stage that passes G1-G5 (G3 on the actual Intern reference)
   **and** G6 is frozen as **GO**; if no kept stage passes them, the retained continuation of the last kept stage is
   frozen as **NO-GO** (a training-study row). No RL claim unless the frozen stage is GRPO.
7. **Deterministic fallbacks.** A missing input fails the rule it feeds (the continuation does not pass; the reason is
   written to `chain_decision.json`). A stage that crashes beyond repair keeps its last exported checkpoint as its final
   continuation; if none exists, the stage is not trained and the chain continues from the last kept stage.

## 6. GRPO validity (A7.10 (f)) and the reward (A7.10 (g))

Behaviour policy pi_T = softmax(z / T) over the scored option codes (T = 1.0), z = the logits at the decision position
restricted to the codes of the presented order; served by the Miles SGLang engine as full-vocabulary log-softmax values
(`token_ids_logprob`, `SGLANG_RETURN_ORIGINAL_LOGPROB=1`; a per-row constant away from the logits). Samplers and events
are v1's (`grpo_online.TorchDecider`, unchanged; `SGLangDecider` replaces only the forward pass):

| mode (decision points trained) | sampled action | stored behaviour log-probability |
|---|---|---|
| replace (tool_select, tool_params) | a ~ pi_T, categorical | log pi_T(a) |
| cascade tau 0.9 (tool_select, tool_params) | a ~ pi_T; commit a iff pi_1(a) >= tau and a is not OTHER, else defer | commit: log pi_T(a); defer: log sum over {a' failing the gate} of pi_T(a'); no gradient when every option fails |
| assist k 3 (tool_select) | ordered top-k by Gumbel-top-k = Plackett-Luce without replacement from pi_T | sum_i [log pi_T(s_i) - log(1 - sum_{j<i} pi_T(s_j))] |
| guard tau 0.5 (tool guard, stop guard: Noul gates) | yes / no ~ pi_T over the two options; the gate follows the sample | log pi_T(sample) |

Knockout rounds (> 52 options) are deterministic top-2 cuts with no probability and no gradient; the trained event is
the final round's categorical choice. Greedy decisions and assist decisions that keep every option carry no event.
Options are scored in the presented order; padded codes do not exist in the Miles loss (each sample carries exactly its
option codes). **KL estimator: none** (no KL term). In Miles one sample = one decision event (prompt tokens + one
option-code token); the custom loss (`miles_port/loss.py`) restricts the decision-position logits to the sample's codes
and computes ratio = exp(log pi_T,current(event) - stored behaviour log-prob), PPO-clip 0.2 / 0.28.

Tests (on the training commit, stage `grpo_validity`): `tests/test_grpo_validity.py` (v1's samplers: stored = replayed
within 1e-4, normalisation, Monte-Carlo frequencies, ratio after an update, knockout) and `tests/test_miles_port.py`
(the SGLang decider reproduces v1's decider event for event and log-prob for log-prob in every mode and the knockout;
the Miles loss replays the stored behaviour log-probabilities within 1e-4 before any update (ratio 1) on the restricted
option set, equals v1's objectives, masks placeholders; the rollout functions' samples). **On-host replay check**
(stage `grpo_replay`, `miles_port/smoke.py replay`): 8 tasks x 2 episodes of the GRPO arms on the SGLang engine with
the GRPO init (simulated System-2, no API cost), every trainable event replayed through the trainer's model class in
bf16 on the identical token ids: pass iff mean |delta log-prob| <= 0.02 and max <= 0.2 nats. Failing it excludes GRPO
(the chain records it; no RL claim).

**Reward, verbatim** (`configs/train/miles_grpo_v2.yaml`; `grpo_online.episode_reward`, kind `gated`):
R = 1[success] x (1 - lambda x min(1, C / C0)) - 0.2 x 1[irreversible error]; lambda = 0.5; C = the episode's
System-2 (LLM) calls (ledger `usage.s2.calls`); C0 = 10 calls; irreversible penalty 0.2 (strict JevGym). Episodes
that raised an infrastructure exception get no credit and no blame. Paper wording: **call-count-penalised success**.

## 7. The freeze record and the attribution artefacts

`checkpoints/r6/final/FROZEN.txt` (copied to `results/gpu_round6_miles/` and S3
`checkpoints/round6_miles/jevagentrl-v2-final/`): the frozen continuation and its stage chain, the label (GO / NO-GO),
the base and its snapshot, the serving / training commit (prompt rendering `training/data.py`, `decision_model.chat_text`,
option ordering, the 52-option knockout and `serve_jev` live in that commit), the Miles image digest, the sha256 of
`chain_decision.json`, the freeze time and the sha256 of every weight file, `config.json`, the tokenizer files, the
chat template, `calibration.json` (the temperature file), `export.json` and `decision_config.json`. Every evaluated row
of the ablation (untrained base, SFT v2, OPD continuations, GRPO continuations, final) keeps its sha256
(`checkpoint_sha256.txt`).

## 8. What is not done in v2

No calibration arms (tau-hat*) and no Stage-2 rows before the freeze; no change to v1, its configs or its host
(`<instance-id>`); no use of StartLux-Decision-27B (MILES_PORT.md §4).

## 9. Deviations and infrastructure adaptations (dated)

- **2026-10-03 ~03:20 UTC, infrastructure repair (commit 1bf1be4).** Miles saves no checkpoint unless a save interval
  is set; the launcher now always passes `--save-interval` (0 in a config = the last rollout only). No recipe value
  changed.
- **2026-10-03 ~03:25 UTC, deviation: the S2 metric of the base rule (§5.1), before any base decision and before any
  training step.** The engine smoke (stage `smoke_engine`, 03:04-03:11 UTC) measured, SGLang vs the bf16 HF forward on
  the same 32 v1-dev prompts: Qwen3-4B max |delta log p| over every option 1.25 nats (mean of the per-prompt maxima
  0.33), StartLux-Decision-4B 0.125 (0.043); for both, argmax agreement 1.00 and identical accuracy (0.9375). The
  written S2 (max |delta log p| <= 0.1 over every option) is dominated by bf16 rounding on options that carry no
  probability and would stop the track with two faithfully served models. S2 is replaced by a test of what it was meant
  to check (the engine serves the model faithfully), applied identically to both bases: on the same 32 prompts, SGLang
  vs an **fp32** HF forward (the reference), argmax agreement >= 0.95, mean total-variation distance between the option
  distributions <= 0.02 and max <= 0.10. The written quantity stays in the report (`pass_s2_as_written`). The engine
  smoke is re-run with the amended read-out before `base_rule`. S1, S3 and S4 are unchanged. The change was made
  before the base decision and before any training step, on engine-smoke data from 32 v1-dev prompts only (no S4
  zero-shot accuracy and no S1 smoke result had been read), and both bases failed the as-written check, so it cannot
  favour either. Approved by the coordinating session on 2026-10-03 (~03:30 UTC).
- **2026-10-03 ~03:35 UTC, budget change (approved by the coordinating session): API cap $100 -> $125.** The stress
  instrument costs ~$10.5 per run as measured (Jev reference, 03:19-03:27 UTC: $0.053 per episode, Flash System-2 at
  9.6 calls per episode), so the planned runs exceed $100 (projected $105-115). Made before any trained checkpoint was
  scored on the stress slice. Nothing else changes (System-2 of the stress slice stays Flash, so the Jev reference stays
  valid). The E2 development slices (descriptive) are cut to the minimum (frozen model: ts_replace, ts_params_replace,
  ts_assist_k3; SFT v2 not run) and the untrained-base row is not run on the stress slice.
- **2026-10-03 ~03:35 UTC, infrastructure:** the stress references are run as two stages (`stress_refs_jev`, API only,
  run in parallel with other stages; `stress_refs_intern`, Intern-Decision-4B on GPU 1).
- **2026-10-03 ~03:50 UTC, infrastructure repair.** The 4-step training smoke (03:32-03:40 UTC) ran the whole Miles
  stack for both bases (Ray, the rollout function, data conversion, the FSDP2 forward of Qwen3-4B and of
  StartLux-Decision-4B's Qwen3.5 class) and stopped at the first loss call: `batch["metadata"]` was missing because the
  `.pth` line of the trainer key patch had been written to a site directory this (Debian-patched) venv does not
  process. The patch is now installed through the venv's purelib directory and a `sitecustomize` on PYTHONPATH that
  Ray workers inherit, and the launcher checks it in a fresh interpreter before training. No step trained; S1 is
  re-run.
- **2026-10-03 ~04:10 UTC, infrastructure repair.** The re-run (03:52-04:00 UTC) showed the patch active and the
  metadata still missing: Miles' rollout-side data-parallel shard packer (`_package_shards`) copies a fixed key list
  that also omits `metadata`. The import hook now patches it as well (metadata partitioned like every per-sample key),
  with a test and the fresh-interpreter check covering both patches. No step trained; S1 is re-run.
- **2026-10-03 ~04:20 UTC, infrastructure (throughput).** The S1 smoke trained (finite losses) at ~130 s per
  optimizer step on the g5.48xlarge (Miles perf: 2.1 TFLOPs per GPU; 28 s of the step in a forward-only pass whose
  full-vocabulary log-probabilities the decision loss never reads), i.e. ~19 h for SFT v2. Two changes that leave
  every objective value unchanged: (a) the trainer's forward-only "old log-prob" pass is skipped for the custom loss;
  (b) the language-model head runs only at the decision positions (`logits_to_keep`; the loss reads one row per sample
  and is identical, test `test_rows_only_logits_match_the_packed_path`), which removes the full-vocabulary logits
  that bounded the micro-batch size. With the logits gone, `max_tokens_per_gpu` may go above the §3 range (up to
  16384): it only changes gradient accumulation (the loss is a per-rollout mean normalised by the global batch), not
  the update. NCCL bandwidth of the host is measured (`nccl_bench.txt`) to choose the transport settings.
- **2026-10-03 ~05:35 UTC, infrastructure (throughput, continued).** The host has no GPU peer-to-peer path (every GPU
  pair crosses the PCIe host bridge, `nvidia_topo.txt`); NCCL's default shared-memory transport reaches ~1.0-1.6 GB/s
  bus bandwidth (`nccl_bench.txt`). `NCCL_SHM_USE_CUDA_MEMCPY=1` reaches 5-7 GB/s in nccl-tests but deadlocked the
  first FSDP initialisation (broadcast timeout after 600 s; speed test 04:52-05:07 UTC) and an 8-process stand-alone
  reproduction (broadcasts while the caching allocator grows; 05:17-05:33 UTC) under every variant tried (pageable,
  pinned or no host-to-device copies; copy mode 2), so it is not used: the default transport stays. With the
  rows-only head and no old-log-prob pass, an SFT step of StartLux-Decision-4B takes ~26 s at `max_tokens_per_gpu`
  16384 (speed test 05:09-05:16 UTC, 3 steps, nothing saved; 190-220 s per step at 3072 before the change), i.e. one
  micro-batch per rank per step. `max_tokens_per_gpu` is set to 16384 for SFT, OPD and GRPO (allowed by the entry
  above); nothing else changes. Projected GPU time at this rate: SFT ~4 h, OPD ~5 h, GRPO ~8-9 h, evaluations ~3 h;
  a stage that would exceed the $400 GPU budget ends at its last checkpoint (§3).
- **2026-10-03 ~06:10 UTC, infrastructure repair (before any GRPO stage ran; no rule changed).** With the repository's
  `addopts = "-q"` plus the stage's `-q`, pytest prints no "N passed" line, which `r6_chain.py` read as the only pass
  signal, so passing validity tests would have counted as failed (the same defect was found and repaired on the v1
  track at ~05:05 UTC). The stage now records `pytest_exit=N` and the chain decides from it (fallback: the pytest
  output), test `test_validity_detection_reads_the_exit_line_and_quiet_pytest_output`; the GRPO stage does not start
  unless the validity tests passed (A7.10 (f)).
- **2026-10-03 ~06:20 UTC, infrastructure (wall time; before OPD or GRPO started).** Concurrent harness episodes per
  rollout: OPD 16 -> 36 (one wave per iteration), GRPO 16 -> 32 (two waves). The GPUs idle while episodes wait on the
  System-2 API, so this shortens every iteration; each episode's seed depends only on (seed, rollout id, group,
  index), not on the worker count, and nothing else changes (tasks, arms, executors, samplers, rows, steps).
- **2026-10-03 ~17:05 UTC, operations and an infrastructure repair (no rule, recipe value or data changed).** SFT v2
  finished at 09:39 UTC (529 steps, 239 min) and its read-outs ran 09:39-10:08 UTC (`post_sft`). The AWS session of
  the operating container expired at ~09:47 UTC; nothing was logged in (reported to the coordinating session) and the
  queued invocation ran on its own. OPD v2 stopped at 10:20 UTC during engine start-up, before any rollout or update:
  in a colocated stage SGLang runs under TorchMemorySaver, which refuses PyTorch's expandable segments, and the
  launcher had set them for every stage. Repair: colocated stages run without expandable segments
  (`launch.alloc_env`, test `test_launcher_env_overrides_and_allocator`). The box stopped itself (45-min idle guard,
  ~11:10 UTC). Restarted 17:03 UTC with the AWS session back (16:55 UTC); the idle counter in `/var/tmp` survived the
  stop and stopped the box again at its first check, so the box runner now resets it at start. Before the rest of
  the chain: one-iteration smokes of the colocated OPD / GRPO loops (`smoke_opd`, `smoke_grpo`: init SFT v2, 6 / 2
  tasks of each pool, nothing saved; development of the pipeline only). Then the remaining chain is queued in one
  invocation that ends by powering the box off: `opd post_opd grpo_validity grpo_replay grpo post_grpo freeze hashes
  e2dev_final ablation poweroff`, without stopping at a failed stage (a stage that produces nothing is "not trained"
  for the chain, §5). Each trainer gets one restart from its last checkpoint after a crash (§3), with
  `max_tokens_per_gpu` 8192 if the crash was out of memory. GPU caps on the box: a hard `shutdown` at the end of the
  $400 budget (estimated from 8.3-8.7 h used before 11:10-11:30 UTC) and a GRPO end time 1.75 h earlier (GRPO only
  starts with >= 2 h left; at the end time the trainer stops and GRPO ends at its last saved continuation); API cap:
  no live run starts unless the ledgers (`scripts/train/r6_spend.py`) leave $11 under $125. The E2 dev slice of the
  frozen model selects its three arms by exact name (`r5_select_arms.py`: a trailing `$`).
- **2026-10-03 ~18:15 UTC, infrastructure repairs found by the smokes (before any OPD / GRPO step that is kept).**
  (a) Miles' SGLang router rejects batched `/generate` requests with per-prompt `token_ids_logprob` lists, so the
  rollout client now sends one prompt per request, concurrently (identical log-probabilities; test
  `test_router_format_one_prompt_per_request`); after it the OPD smoke ran one iteration end to end (6 episodes,
  teacher labels, 2 updates; 18:00-18:08 UTC). (b) The GRPO smoke stopped in Miles' trainer-side micro-batch
  planner, which on the FSDP backend counts samples, not episodes (Miles plans compact rollouts on the rollout side
  only for backends that advertise the full parallel config, as Megatron does). The FSDP backend now advertises it
  (`keys_patch.full_parallel_config`), so the global batch of 16 counts episodes and an episode's decisions stay in
  one step, as §3 specifies; and `rollout_mask_sums` now reaches the loss, so an episode's decisions average to one
  term (the per-episode mean of §3) instead of summing. SFT and OPD samples are their own rollouts (mask sum 1), so
  neither changes for them. Tests: `test_keys_patch_carries_metadata_to_the_trainer`,
  `test_startup_sitecustomize_patches_a_fresh_interpreter`; both smokes are re-run before the chain is queued.
- **2026-10-03 ~22:05 UTC, a lost record and its repair (no rule or result affected).** OPD v2 ran 18:38-21:26 UTC
  (24 iterations, 288 updates; continuations `opd_i12`, `opd`). The iteration-24 export writes into the stage
  directory, and the exporter cleared that directory first, which deleted OPD's own records (`rollout_metrics.jsonl`,
  `teacher_ledger.jsonl`, `stage_config.yaml`) after training and before they were copied out. The continuations, the
  Miles training log (`opd_train_try1.log`: per-step losses, per-rollout Miles metrics) and the teacher cache survive.
  OPD's API spend is therefore an estimate from the per-iteration rows read during the run (iterations 0-1 and 8-15:
  $0.009-0.013 per executor episode, teacher $0.254 after 16 iterations): executors ~$9.9, teacher ~$0.4. The exporter
  now removes only the files an export writes (test `test_strict_export_round_trip`), and the host checkout was
  updated at 22:02 UTC, before GRPO's first export. The chain read OPD mechanically (§5): neither continuation passes
  the keep clause against SFT v2 (v1-dev accuracy, recovery and stress success drop; `chain_decision.json`), so
  GRPO starts from SFT v2.
- **2026-10-04 ~00:45 UTC, status at the end of the operating container's AWS session (no rule involved).** GRPO v2
  ran 21:58-23:52 UTC (30 iterations x 64 episodes, 120 updates; validity tests 30 passed, replay check mean |delta
  log p| 0.0031, max 0.085). The chain (00:20 UTC, `chain_decision.json`): neither GRPO continuation passes (stress R
  and S fall below SFT v2), so the kept chain is SFT v2 alone and the frozen continuation is **SFT v2, GO on G1-G6**
  (`rl_claim` false). The freeze stage was running (temperatures fitted: global 0.513; buckets noul:2 0.577,
  choice:3-5 0.050, choice:17-52 0.450); read on the host at 00:44 UTC: `checkpoints/r6/final/model.safetensors`
  sha256 d7bfe3ed50bc8d3ff4821500b0975a9c8ba7b626a7129f47a979d99d4c686282, `calibration.json` 10f5889f...e625,
  `tokenizer.json` 5f9e4d49...1cb42, `chat_template.jinja` a4aee8af...f715, serving / training commit 4c0bdf8.
  The authoritative freeze record is `FROZEN.txt` in `s3://<bucket>/checkpoints/round6_miles/
  jevagentrl-v2-final/` (and `gpu_round6_miles/final_FROZEN.txt`), written by the queued invocation, which then runs
  `hashes e2dev_final ablation poweroff` and stops the box (also armed: idle stop, hard stop 08:55 UTC).
