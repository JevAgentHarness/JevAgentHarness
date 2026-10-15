# JevAgentRL training manifest (GPU round 5; PREREGISTRATION A7.10 (a)-(b), (f)-(h); A7.13)

Written and committed on 2026-10-02 (~22:10 UTC) by the operator of the JevAgentRL chain, **before any Stage-1 main-run
result was read by the operator and before evaluation outputs were released to anyone who develops the model**. The recipe
configs and scripts named below are those of the commit that adds this file
(`git log -1 --format=%H -- results/gpu_round5/TRAINING_MANIFEST.md`); the GPU host runs exactly that commit's bundle
(`scripts/aws/gpu_round5_rl.sh push`; every host status line names the commit it runs). From this commit on, the chain runs
mechanically: `scripts/train/r5_chain.py` (the gonogo script of A7.10 (a)), not a person, keeps or drops each stage, retains
one continuation per stage and names the frozen checkpoint.

## 1. Actors, workspace and access controls (A7.10 (b))

- **Operator / developer:** the JevAgentRL training operator (a separate working copy of
  this repository, apart from the coordinating operator). It takes every
  model-development decision until the freeze record exists, and does not open Stage-1 or Stage-2 evaluation outputs
  before then. The coordinating (main) session will read Stage-1 results; from that moment, any message from it that would
  change the chain (stages, configs, thresholds, checkpoint choice) is recorded below as a dated deviation and, per A7.10 (b),
  makes every comparison involving the resulting checkpoint exploratory (S2-3 / S2-4 enter with p = 1); operational
  messages (AWS budget, stop / start, crash restarts) are not deviations.
- **Workspace:** the worktree above and the GPU host `<instance-id>` (`jev-rl-round5`, g6e.4xlarge, us-east-1d).
  The host receives only the git bundle of this branch (`s3://<bucket>/code/jev-rl-round5.bundle`),
  `code/gpu_round5_rl.sh` / `code/r5_driver.sh`, round 3's R2 adapter (`checkpoints/v0/r2/`) and HF / API access; it writes
  only `gpu_round5/`, `checkpoints/round5/` and `e2a_jevgym_shadow_cal_rl/` in the bucket. The instance role can technically
  read the whole project bucket (which also holds Stage-1 outputs under `main/` and `gpu_round4/`); the separation is
  procedural: the box script reads no other prefix, and the operator never syncs main-run directories, `results/primaries/`,
  `results/stage2/` or `results/main/` to the host.
- **Not allowed to the operator before the freeze:** anything under the main tree's `runs/` except `runs/dev*`,
  `runs/pilot*` and `runs/gpu_round3`; `results/primaries/`, `results/stage2/`, `results/main/`; any Stage-1 or Stage-2
  trajectory, score card, primary, profile selection, failure example or aggregate number; seeds 100000-100499,
  200000-200499, 300000-300999.
- **Access record (dated).** On 2026-10-02 at about 20:40 UTC, before the barrier rule reached the operator (about 21:35
  UTC), the operator printed the *schema* of the first record of one Stage-1 E2a trajectory file in the main tree
  (`runs/e2a_jevgym_tool_decisions/`, a `jev_ts_replace*` arm; one episode on an evaluation seed): its key names, that
  episode's `usage` block and its first decision-log entry (the 69-tool option list, `s1_value`, part of `s1_dist`), to
  learn the trajectory format for `scripts/train/r5_summary.py`. No other record, aggregate or score card was opened. The
  information is one Jev decision on one task; no recipe value, seed, threshold or stage was chosen on it (the recipe
  below predates or does not depend on it). It is disclosed here so that the evaluation side can rule on A7.10 (b).
- No Stage-1 output has been read since; none will be before the freeze record is committed.

## 2. Training inputs (the whitelist)

| input | content | notes |
|---|---|---|
| data v0 | `data/train_v0/{train.jsonl, replay_when2call.jsonl.gz}`; `dev.jsonl` = the **210-decision gate set**; `dev_when2call.jsonl.gz` | unchanged since 9271976 |
| data v1 | `data/train_v1/{train,replay_when2call}.jsonl.gz`; dev files `{dev,dev_recovery,dev_knockout}.jsonl.gz`, `dev_teacher.summary.json` | unchanged since 9271976 |
| training seeds | JevGym seeds >= 10000 outside 100000-100499, 200000-200499, 300000-300999 and 5000-5999 (`training/seed_guard.TRAIN_FORBIDDEN`, the code form of `registry.training_exclusions()`; enforced by `opd.py --strict-seeds`, `grpo_online` `seed_policy: strict`, `data_v1`, `harvest.py`): OPD pool **40000-43999**, GRPO pool **50000-52999** | generated on the host |
| development seeds | >= 1000 outside 5000-5149: greedy in-training rollouts **1000-1047** (OPD `--eval-seed0 1000`; GRPO `eval_suite.seed0: 1000` set in the run config, the code default of A6.4 unchanged); the E2 development slice **1100-1149** (`configs/experiments/jevgym_dev_rl.yaml`) | a first reference run on seeds 0-49 (llm_only + Jev arms, 21:15-21:30 UTC) is superseded by 1100-1149 and not used by the chain |
| calibration seeds | **5000-5149**, only for the tau-hat* fit of the frozen checkpoint, after the freeze (`configs/experiments/jevgym_shadow_cal_rl.yaml`, the registered fit command) | stage `cal` |
| models | Qwen/Qwen3-4B (HF snapshot `1cfa9a7208912126459214e8b04321603b3df60c`); internlm/Intern-Decision-4B (`0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd`, references only); Jev `typesafe/jev-1.13` via OpenRouter (OPD teacher; references); Gemini 3.5 Flash / Flash-Lite (System-2) | |
| earlier checkpoint | round 3's R2 adapter (SFT on v0): the "SFT v0" ablation row only | S3 `checkpoints/v0/r2/` |

## 3. Recipe, budgets and the stage order

Host stages (`scripts/aws/gpu_round5_rl.sh box "<stages>"`, in this order):

| # | stage | what (fixed values) |
|---|---|---|
| 1 | `sft` | **SFT v1** (R3-v1) from Qwen3-4B: data v1 train + 0.45 x When2Call replay (16,902 rows); LoRA r 32 / alpha 64 all-linear; soft CE + 0.5 Brier; option permutation; LR 1e-4 cosine, warmup 3 %; 2 epochs; 64 rows / optimiser step; token-budget micro-batches <= 3,072 padded tokens, no activation checkpointing; bf16 (started 21:15 UTC on commit 805a185, before this manifest; the recipe is `scripts/train/12_sft_v1.sh` unchanged except the batching) |
| 2 | `eval_sft` | `eval_decisions` of SFT v1 and SFT v0 on the gate set (+ flip rate) and v1 dev / recovery / knockout; Intern-Decision-4B on v1 dev + recovery (`scripts/gate_s1_dev.py`; G3's actual reference); gonogo |
| 3 | `e2dev_refs_api`, `e2dev_sft_v1`, `e2dev_refs_intern` | E2 development slice, seeds 1100-1149, Flash T 0 (and Flash-Lite T 0 for the `*_s2lite` arms): llm_only, Jev, SFT v1 (served by `serve_jev`), Intern-Decision-4B |
| 4 | `opd` | **OPD** (`configs/train/opd_round5.yaml`): init SFT v1; six arms (ts + params replace with stop guard; ts replace; assist k 3; cascade tau 0.9 on ts / params; ts guard + stop guard; Flash shadow on all three); teacher Jev; target 0.7 Jev + 0.3 oracle gold, the gold one-hot where Jev's argmax is not the gold; forward KL; 24 iterations x 36 tasks; <= 600 labelled decisions / iteration; replay 0.5; LR 5e-5; rollout temperature 1; executor Flash-Lite T 0.7 (Flash T 0.3 for the shadow arm); checkpoints every 2 iterations |
| 5 | `eval_opd`, `e2dev_opd` | continuations **opd_i12** (iteration 12) and **opd** (iteration 24): decision level + gonogo against SFT v1; E2 dev slice |
| 6 | `grpo_validity` | `tests/test_grpo_validity.py` on the training commit -> `grpo_validity.txt` |
| 7 | `grpo` | **GRPO** (`configs/train/grpo_online_round5.yaml`): init = the retained continuation of the last kept stage (`chain_decision.json`); 30 iterations x 8 tasks x G 8; four arms (ts + params replace with stop guard; assist k 3; cascade tau 0.9; Flash ts guard + stop guard); PPO-clip 0.2 / 0.28; exact KL 0.02 to the init; GiGPO omega 1, gamma 0.95; LR 1e-5; executor Flash-Lite T 0.7 (Flash T 0.3 for the guard arm); checkpoints every 5 iterations |
| 8 | `eval_grpo`, `e2dev_grpo` | continuations **grpo_i15** and **grpo** (iteration 30) against the GRPO init |
| 9 | `chain`, `freeze`, `hashes` | `r5_chain.py` -> `chain_decision.json`; the frozen continuation -> `checkpoints/r5/final`; temperatures per (type x option-count bucket) fitted on v1 dev (`training/calibrate.py`, min bucket 200); freeze record |
| 10 | after the freeze | `cal` (tau-hat* on 5000-5149), `latency`, `e2dev_final`, `base_control` (A7.13 (i)), `e2dev_sft_v0`; optionally the A7.13 frozen-policy distillation control (`opd.py --behaviour-frozen`) if it does not delay the freeze |

**Maximum budgets.** Training GPU: **48 GPU-hours** reserved (A7.10 (h); this round's host is one L40S at $3.004 / h, so the
cap is about $144; expected 15-20 h); serving / development slices on the same host within the round's $250 GPU cap. Teacher:
**$8** (OPD's ledger hard stop, `budget_usd`). API in all (teacher, executors, development slices, calibration arms): **$200**.
Deadlines: freeze target 2026-10-04; the freeze record must exist by **2026-10-07 12:00 UTC** (A7.10 (h)).

**Allowed ranges.** The hyper-parameters are the fixed values above; there is no search. The operator may, without a
deviation, only repair infrastructure (OOM, network, host restart): resume a stage from its last checkpoint with the same
config, lower the update token budget to no less than 1,536 or the rollout decider batch to no less than 2. A stage that
cannot finish within the budgets ends at its last checkpoint, which is then its final continuation.

## 4. Keep, retain and freeze (`scripts/train/r5_chain.py`, verbatim in its docstring)

Accuracies on [0, 1]; "v1 dev" = `data/train_v1/dev.jsonl.gz`, "recovery" = `dev_recovery.jsonl.gz`, ECE raw (before
temperature); live Flash arms = ours_ts_replace, ours_ts_params_replace, ours_ts_assist_k3, ours_stop_guard, ours_ts_guard on
seeds 1100-1149; own-state accuracy = tool_select accuracy of ours_ts_replace and ours_ts_params_replace pooled by
decisions; call-count-penalised success as in §5.

1. **SFT v1** is the base; always kept.
2. **OPD** continuations opd_i12, opd. One PASSES against SFT v1 iff (A, §18.4) v1-dev acc >= SFT + 0.010, recovery >=
   SFT, ECE <= SFT + 0.010; **or** (B, in-harness) v1-dev acc >= SFT - 0.005, recovery >= SFT - 0.010, ECE <= SFT + 0.010,
   own-state accuracy >= SFT + 0.020 and mean live success >= SFT - 0.010.
3. **GRPO** continuations grpo_i15, grpo, trained from the retained continuation of the last kept stage. One PASSES
   against its init iff the GRPO validity tests passed (§5), v1-dev acc >= init - 0.010, recovery >= init - 0.010, ECE <=
   init + 0.020, mean live call-count-penalised success strictly above the init's, and mean live success >= init - 0.010.
4. **Retention.** A stage is kept iff at least one continuation passes; among passing continuations the retained one has
   the highest v1-dev accuracy, then the later step.
5. **Freeze.** The retained continuation of the last kept stage that passes the §18.4 gates G1-G5 (GO); G3 must use the
   actual Intern-Decision-4B v1-dev reference (a gonogo built on the round-2 proxy counts as not passing). If no kept stage
   passes, the retained continuation of the last kept stage is frozen and labelled NO-GO (Intern-Decision-4B stays the open
   System-1; JevAgentRL is reported as a training study row). No RL claim unless the frozen stage is GRPO.
6. **Deterministic fallbacks.** A missing input fails the rule it feeds (the continuation does not pass; the reason is
   written to `chain_decision.json`). A stage that crashes beyond repair keeps its last checkpoint as its final continuation;
   if none exists, the stage is not trained and the chain continues from the last kept stage.

**Gates G1-G5** (`python -m jevharness.training.data_v1 gonogo`, TRAINING_RECIPES §18.4):

| gate | rule | threshold |
|---|---|---|
| G1 minimum | gate-set acc >= 0.8 x Jev's gate acc (0.9714) | >= 0.7771 |
| G2 static parity | gate-set acc >= Intern-4B's gate acc (0.8619) - 0.01 | >= 0.8519 |
| G3 in-harness states | v1-dev tool_select acc >= Intern-4B's v1-dev tool_select + 0.05 **and** v1-dev overall >= Intern-4B's + 0.02 (Intern on the same rows) | measured in stage 2 |
| G4 vs teacher | v1-dev acc >= 0.8 x Jev's on tool_select / tool_params / stop (Jev 0.9163 / 0.9794 / 0.9865) | >= 0.7330 / 0.7835 / 0.7892 |
| G5 calibration | v1-dev ECE (raw) <= max(0.10, Intern-4B's v1-dev ECE when known) | <= 0.10 |

Reported with the decision (§18.4 confirmation, descriptive): on seeds 1100-1149, ours_shadow tool_select accuracy >=
intern_shadow's; ours_ts_replace success and own-state accuracy >= intern_ts_replace's. JevGym comparisons with
Intern-Decision-4B are "generator-adapted vs off-the-shelf" (A7.13 (vi)); no state-of-the-art claim.

## 5. GRPO validity (A7.10 (f)) and the reward (A7.10 (g))

Behaviour policy pi_T = softmax(z / T) over the scored options (T = 1.0), pi_1 at T = 1:

| mode (decision points trained) | sampled action | stored behaviour log-probability |
|---|---|---|
| replace (tool_select, tool_params) | a ~ pi_T, categorical | log pi_T(a) |
| cascade tau 0.9 (tool_select, tool_params) | a ~ pi_T; commit a iff pi_1(a) >= tau and a is not OTHER, else defer to System-2 | commit: log pi_T(a); defer: log sum over {a' failing the gate} of pi_T(a'); no gradient when every option fails |
| assist k 3 (tool_select) | ordered top-k by Gumbel-top-k = Plackett-Luce sampling without replacement from pi_T | sum_i [log pi_T(s_i) - log(1 - sum_{j<i} pi_T(s_j))] |
| guard tau 0.5 (tool guard, stop guard: Noul gates) | yes / no ~ pi_T over the two options; the gate follows the sample | log pi_T(sample) |

Knockout rounds (> 52 options) are deterministic top-2 cuts with no probability and no gradient: the trained event is the
final round's categorical choice over the surviving pool. Greedy (evaluation) decisions and assist decisions that keep every
option carry no event. The options are scored in the presented order (no permutation in GRPO); padded option codes are
masked (-inf, no mass). The KL term is the exact KL(pi_1 || pi_ref) over the same option set, pi_ref = the frozen init.
Tests (`tests/test_grpo_validity.py`, tolerance 1e-4 in log-probability): stored behaviour log-probs equal the ones
replayed through the update path before any update; the distributions are normalised; Monte Carlo frequencies of every
sampler match its stated rule (replace, cascade, assist, guard); after an update the ratio is exp(current-policy
log-prob - stored behaviour log-prob) on the masked option set with mixed option counts; knockout decisions train only the
final-round choice and replay exactly; the KL uses the same distributions.

**Reward, verbatim** (`configs/train/grpo_online_round5.yaml`; `grpo_online.episode_reward`, kind `gated`):
R = 1[success] x (1 - lambda x min(1, C / C0)) - 0.2 x 1[irreversible error]; lambda = 0.5; C = the episode's System-2 (LLM)
calls (ledger `usage.s2.calls`); C0 = 10 calls; irreversible penalty 0.2 (strict JevGym). Episodes that raised an
infrastructure exception get no credit and no blame. Paper wording: **call-count-penalised success**.

## 6. The freeze record and the attribution artefacts

The freeze record (`checkpoints/r5/final/FROZEN.txt`, copied to `results/gpu_round5/` and S3
`checkpoints/round5/jevagentrl-4b/`) lists: the frozen continuation and its stage chain, the sha256 of the adapter weights
and config, the tokenizer files and chat template (prompt), the decision config, the temperature file, the base model
snapshot, the serving / training commit (prompt rendering, option ordering, the 52-option knockout and `serve_jev` live in
that commit), the sha256 of `chain_decision.json`, and the freeze time. Every evaluated row of A7.13 (i) is kept with
sha256 (`checkpoint_sha256.txt`): the untrained base with its v1-dev temperature fit (`base_control`), SFT v1, the OPD
continuations, the GRPO continuations, the frozen model.

## 7. Deviations (dated)

- **2026-10-03 ~05:05 UTC, operator, operational repair (no rule, threshold, stage or config changed; no evaluation
  information involved).** The stage `grpo_validity` ran `tests/test_grpo_validity.py` on the host at 04:52 UTC and all
  seven tests passed (`results/gpu_round5/grpo_validity.txt`: seven passes, `[100%]`, no failure). With the repository's
  `addopts = "-q"` plus the stage's `-q`, pytest printed no "7 passed" summary line, and `r5_chain.py` (commit 9945d04)
  detected a pass only from that line, so it would have read the passed tests as failed and made every GRPO continuation
  fail the validity clause. Repair (commit after 2dd05a9): the chain decides validity from an explicit `pytest_exit=N`
  line when present, else from the pytest output (a "N passed" summary without failures, or progress lines of passes /
  skips only reaching `[100%]`); the stage now records `pytest_exit=$?`. Unit test:
  `tests/test_r5_chain.py::test_validity_detection_handles_quiet_pytest_output` (includes the host's exact output). The
  GRPO stage itself is unaffected (its init, the retained SFT v1, came from the same chain run, which does not use the
  validity clause).
