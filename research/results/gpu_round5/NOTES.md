# GPU round 5: JevAgentRL phase 1 (development data only)

Development, training and calibration data only: nothing here is an evaluation number or a paper headline. The rules of
the chain are in [`TRAINING_MANIFEST.md`](TRAINING_MANIFEST.md) (PREREGISTRATION A7.10 / A7.13; committed 9945d04 before
any Stage-1 result was read by the operator); the host script is `scripts/aws/gpu_round5_rl.sh`; outputs on
`s3://<bucket>/gpu_round5/` and `.../checkpoints/round5/`. Comparisons with Intern-Decision-4B are
**in-domain** (generator-adapted vs off-the-shelf, A7.13 (vi)): JevAgentRL is trained on states of the JevGym generator.

## 0. Settings actually run (quoted by the paper's appendix)

| stage | init | data / tasks | setting | seeds | System-2 executor | key hyper-parameters |
|---|---|---|---|---|---|---|
| SFT v1 (R3-v1) | Qwen/Qwen3-4B | data/train_v1 train (15,694) + 0.45 x When2Call replay (1,208) = 16,902 rows | harness states of the evaluation setting (v1) + v0 rows | training seeds (v1: 30000-30599; v0: 10000-12059) | - | LoRA r 32 / alpha 64 all-linear; soft CE + 0.5 Brier; option permutation; LR 1e-4 cosine, warmup 3 %; 2 epochs (536 steps); 64 rows / step; token-budget micro-batches <= 3,072 padded tokens; no activation checkpointing; bf16; 134 min |
| OPD (R9) | SFT v1 | 24 iterations x 36 tasks = 864 episodes, 6 arms | evaluation setting (families v2, near distractors, 60 distractors, 20 noise lines, strict) | train 40000-43999; greedy dev evals 1000-1047 | Gemini 3.5 Flash-Lite T 0.7 (Flash T 0.3 for the shadow arm) | `configs/train/opd_round5.yaml`: Jev teacher, target 0.7 Jev + 0.3 gold, gold one-hot where Jev's argmax is wrong, forward KL, LR 5e-5, rollout T 1; 222 min |
| GRPO (R8) | SFT v1 (OPD not kept by the chain) | 30 iterations x 8 tasks x G 8 = 1,920 episodes, 4 arms | evaluation setting (as above) | train 50000-52999; greedy dev evals 1000-1047 (never 5000-5149) | Flash-Lite T 0.7 (Flash T 0.3 for the guard arm) | `configs/train/grpo_online_round5.yaml`: call-count-penalised success, PPO-clip 0.2 / 0.28, exact KL 0.02 to the init, GiGPO, LR 1e-5; 270 min |
| E2 dev slice | each stage + references | 50 tasks per arm | evaluation setting | dev seeds 1100-1149 (a first reference run on 0-49 is superseded) | Flash T 0 (E2a) and Flash-Lite T 0 for the *_s2lite arms | `configs/experiments/jevgym_dev_rl.yaml` |

The older `configs/train/grpo_online_jevgym.yaml` (30 mixed distractors, `eval_suite.seed0: 5000`) is **not** used in this
round: round-5 GRPO trains and evaluates in the frozen evaluation setting, with greedy development rollouts on seeds
1000-1047 (`seed_policy: strict`, `training/seed_guard.py`); no curriculum.

GRPO reward, verbatim: R = 1[success] x (1 - 0.5 x min(1, C / 10)) - 0.2 x 1[irreversible error], C = System-2 calls of
the episode ("call-count-penalised success").

## 1. Compute and spend

- Instance `<instance-id>`, g6e.4xlarge (1x NVIDIA L40S 48 GB, 16 vCPU, 128 GiB), us-east-1d, on-demand $3.004 / h,
  launched 2026-10-02 20:11 UTC (p5.4xlarge, g6e.12xlarge / 24xlarge, p4d / p4de.24xlarge and g6e.2xlarge had no capacity in
  any us-east-1 zone at launch). Guard rails: 30 h power-off, 45-min idle stop (cron), shutdown behaviour stop.
- GPU hours and $, API $ by item: §6 (`scripts/train/r5_spend.py`).

## 2. Engineering changes this round

- SFT speed: the round-4 preset (batch 4 with activation checkpointing) ran at 49 s / step on the L40S (7 h for 2 epochs).
  Token-budget micro-batches (length-sorted, <= 3,072 padded tokens, 0.2 % padding) without checkpointing: 15 s / step
  (about 4.4k tokens / s), 134 min for 2 epochs (`train_decision.py --max-batch-tokens`; the same option in `opd.py` and
  `grpo_online.py`).
- Seed guard with the Stage-2 range (`training/seed_guard.py`), per-arm System-2 executors and eval-arm subsets in OPD /
  GRPO, OPD `--gold-fallback` (data v1's rule) and `--behaviour-frozen` (the A7.13 frozen-policy control).
- GRPO validity tests (`tests/test_grpo_validity.py`, A7.10 (f)); the mechanical chain (`scripts/train/r5_chain.py`);
  development-slice tooling (`scripts/train/r5_select_arms.py`, `r5_summary.py`, `r5_ablation.py`).

## 3. What each stage bought (development data)

### 3.1 Decision level

Gate set = 210 decisions of `data/train_v0/dev.jsonl` (same generator as the training data: necessary, uninformative for
v0 / v1-trained checkpoints, §18.4); v1 dev = 2,377 decisions of held-out tasks of the harness's own states; recovery =
606 of them after an off-plan step.

| stage | gate acc | v1 dev acc | v1 tool_select | v1 tool_params | v1 stop | v1 recovery | v1 ECE | §18.4 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B, untrained (A7.13 (i)) | 0.729 | 0.701 | 0.516 | 0.816 | 0.804 | 0.663 | 0.278 | - |
| SFT v0 (R2, train_v0) | 0.995 | 0.952 | 0.928 | 0.953 | 0.982 | 0.922 | 0.022 | GO |
| SFT v1 (R3-v1, train_v1) | 1.000 | 0.997 | 0.997 | 1.000 | 0.994 | 0.998 | 0.029 | GO |
| + OPD, iteration 12 | 1.000 | 0.989 | 0.984 | 0.999 | 0.985 | 0.979 | 0.023 | GO |
| + OPD, iteration 24 | 0.948 | 0.989 | 0.985 | 0.998 | 0.984 | 0.980 | 0.021 | GO |
| + GRPO, iteration 15 | 1.000 | 0.997 | 0.997 | 1.000 | 0.994 | 0.998 | 0.004 | GO |
| + GRPO, iteration 30 | 1.000 | 0.994 | 0.991 | 1.000 | 0.991 | 0.995 | 0.003 | GO |
| JevAgentRL-4B (frozen, calibrated) | 1.000 | 0.997 | 0.997 | 1.000 | 0.994 | 0.998 | 0.004 | GO |
| Intern-Decision-4B | 0.862 | 0.843 | 0.748 | 0.875 | 0.930 | 0.820 | - | - |
| Jev (teacher) | 0.971 | 0.958 | 0.916 | 0.979 | 0.987 | 0.944 | 0.013 | - |

The frozen row's decision-level numbers are raw (before temperature); with its temperatures (`eval_final_calibrated_v1.json`)
v1-dev accuracy is unchanged (0.997) and ECE 0.001. Intern-Decision-4B on the same v1 rows: `scripts/gate_s1_dev.py`
(no ECE computed there); Jev: `data/train_v1/dev_teacher.summary.json` (one question per request), gate accuracies of
Jev and Intern from `results/gpu_round2/gate_verdict.json`.

### 3.2 E2 development slice (live, seeds 1100-1149, Flash System-2 T = 0)

| stage | n | shadow ts | shadow tp | shadow stop | ts-replace succ. | ts-replace own-state ts acc | ts+params-replace succ. | its ts acc | its tp acc | assist-k3 succ. | stop-guard succ. | ts-guard succ. | S1 ms/call p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B, untrained (A7.13 (i)) | - | - | - | - | - | - | - | - | - | - | - | - | - |
| SFT v0 (R2, train_v0) | 50 | 0.984 | 0.931 | 0.963 | 0.880 | 0.903 | 0.780 | 0.948 | 0.788 | 1.000 | 0.940 | 1.000 | 263 |
| SFT v1 (R3-v1, train_v1) | 50 | 1.000 | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.980 | 1.000 | 1.000 | 367 |
| + OPD, iteration 12 | 50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 413 |
| + OPD, iteration 24 | 50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 416 |
| + GRPO, iteration 15 | 50 | 1.000 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 443 |
| + GRPO, iteration 30 | 50 | 1.000 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 390 |
| JevAgentRL-4B (frozen, calibrated) | 50 | 1.000 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 314 |
| Intern-Decision-4B | 50 | 0.739 | 0.823 | 0.931 | 0.260 | 0.544 | 0.180 | 0.630 | 0.545 | 0.960 | 1.000 | 1.000 | 139 |
| Jev (teacher) | 50 | 0.949 | 0.916 | 0.989 | 0.800 | 0.970 | 0.800 | 0.969 | 0.975 | 0.920 | 0.980 | 1.000 | 157 |

Flash-Lite cells (same seeds, `*_s2lite`, success): assist k = 3 -- SFT v0 0.82, SFT v1 0.84, OPD i12 0.78, OPD 0.80,
GRPO i15 0.84, GRPO 0.82, frozen 0.76, Intern-Decision-4B 0.60, Jev 0.82, llm_only (Flash-Lite) 0.88; stop guard --
SFT v0 0.86 (guard accuracy 0.49), SFT v1 0.92, OPD i12 0.94, OPD 0.90, GRPO i15 0.92, GRPO 0.92, frozen 0.94 (0.92),
Intern-Decision-4B 0.92 (0.59), Jev 0.88 (0.84). "S1 ms/call p50" is the median System-1 call latency in the shadow /
replace arms under 8 concurrent episodes (a 69-option tool_select goes through serve_jev's 52-option knockout; see §5 for
the controlled measurement).

### 3.3 Reading

- **Data v1 closed the static-to-in-harness gap on this generator.** SFT v1 is at 1.000 own-state tool_select accuracy
  and 1.00 replace success on the live slice (Intern-Decision-4B: 0.544 / 0.26; Jev: 0.970 / 0.80; SFT v0 on v0 data:
  0.903 / 0.88). The gap that round 3 measured for R2 (0.45 in sampled OPD rollouts, v1 setting) is far smaller under
  greedy serving with Flash (0.90), and data v1 removes the rest. This is **in-domain**: JevAgentRL is trained on states of
  the JevGym generator (training seeds), Intern-Decision-4B and Jev are not (A7.13 (vi)); it is not evidence of general
  decision competence.
- **The development measures are saturated.** v1 dev (0.997) and the live slice (1.000 on every Flash arm) leave no room
  to separate SFT v1, OPD and GRPO; the gate set (210) is uninformative for v0 / v1-trained checkpoints (SFT v0 also passes
  every §18.4 gate). The untrained base (0.701 v1 dev, 0.516 tool_select) shows what the training adds overall.
- **OPD bought nothing measurable and was dropped by the chain.** Distilling Jev (0.88-0.94 tool_select argmax agreement with the
  student, 0.90-0.95 accuracy vs gold) into a student that is already at gold accuracy slightly lowered v1-dev and recovery
  accuracy (0.989 / 0.979-0.980 vs 0.997 / 0.998) even with the gold fallback (853 of 12,414 labelled rows), while
  lowering ECE (0.029 -> 0.021) -- the teacher's softer distributions transfer, its mistakes partly too.
- **GRPO kept by the rule, but its gain is small.** Both GRPO continuations pass (validity tests, no decision-level loss,
  call-count-penalised success above SFT v1's on the live slice: 0.6822 / 0.6846 vs 0.6788, i.e. +0.003 on 250 live
  episodes per checkpoint, with success 1.000 vs 0.996). Its clear effect is calibration: raw v1-dev ECE 0.029 -> 0.004
  (i15) / 0.003 (i30) at equal accuracy. In training the greedy dev rollouts (Flash-Lite T = 0.7 executor) stayed flat
  (ts+params replace 0.833 -> 0.854, assist 0.812 -> 0.792); many groups had zero reward spread (median 0.75 of groups).
  An "RL" claim beyond "GRPO did not hurt and sharpened confidence on these development data" is not supported.
- **Flash-Lite cells are not saturated** (assist k3 0.76-0.84 vs llm_only Flash-Lite 0.88): there the System-2 decides
  among the top-3 and its own errors dominate; differences between stages are within noise at n = 50.
- **Latency is the weak point** (§5): served on one L40S, JevAgentRL is slower per request than Jev's hosted API.

## 4. The chain's decision and the frozen checkpoint

`chain_decision.json` (`scripts/train/r5_chain.py`, rules of TRAINING_MANIFEST §4) at 10:01 UTC:

| stage | continuation | decision | why |
|---|---|---|---|
| SFT v1 | - | kept | base of the chain |
| OPD | i12, i24 | dropped | rule A: v1-dev acc 0.989 < SFT + 0.010; rule B: v1-dev acc < SFT - 0.005, recovery 0.979 / 0.980 < SFT - 0.010, own-state accuracy 1.000 not >= 1.000 + 0.020 |
| GRPO (init SFT v1) | i15, i30 | kept, i15 retained | both pass (validity tests 7 / 7; v1-dev 0.997 / 0.994 >= 0.987; recovery 0.998 / 0.995; ECE 0.004 / 0.003; call-count-penalised success 0.6822 / 0.6846 > 0.6788; success 1.000); retention: higher v1-dev accuracy (0.99706 vs 0.99411) |

Frozen: **JevAgentRL-4B = grpo_i15**, GO (G1-G5 pass, G3 on the actual Intern-Decision-4B v1-dev reference), frozen
2026-10-03 10:17:50 UTC on the host. §18.4 confirmation on seeds 1100-1149: ours_shadow tool_select 1.000 >= Intern 0.739;
ours_ts_replace success 1.00 >= 0.26 and own-state accuracy 1.000 >= 0.544. Freeze record: [`FREEZE_RECORD.json`](FREEZE_RECORD.json)
(adapter sha256 `b5eb0250228355212c2b9b08f0ac6312db1a8d632314febcc384174194994d1c`, re-verified from S3; temperatures
`0c4b9534...`; serving commit c6c1cc1; S3 `checkpoints/round5/jevagentrl-4b/`). Deviation: one operational repair of the
chain's test-output parser before the chain stage ran (TRAINING_MANIFEST §7); no Stage-1 or Stage-2 output was read by the
operator before the freeze.

## 5. Calibration (tau-hat*) and latency

**Temperatures** (v1 dev, `calibration.json`): global 0.775; noul:2 0.930; choice:17-52 0.249; choice:3-5 0.095 (the
model is under-confident on Choices before scaling; after it, v1-dev ECE 0.001).

**tau-hat\*** (calibration seeds 5000-5149, n = 150 per arm, the frozen model served with its temperatures, the
registered command; `cal/`): shadow accuracy on the calibration states 1.000 tool_select, 0.990 / 1.000 tool_params,
1.000 / 0.992 stop (Flash / Flash-Lite).

| System-2 | tool_select tau-hat\* | accept rate | tool_params tau-hat\* | accept rate | V |
|---|---|---|---|---|---|
| Flash | 0.500 | 1.00 | 0.99999988 | 0.80 | fallback W = 1 (no wrong System-2 episode) |
| Flash-Lite | 0.500 | 1.00 | 0.99874 | 1.00 | fallback W = 1 |

**Latency** (`scripts/measure_latency.py`, synthetic 1.5k-token state, 3 questions per request: a 21-option Choice, a
Noul and a Score, 200 requests; same protocol as round 4's Intern measurement):

| System-1 | where | p50 ms / request, c = 1 | c = 4 | requests / s |
|---|---|---|---|---|
| JevAgentRL-4B | serve_jev on 1x L40S (HTTP, bf16, session cache) | 307 | 1,213 | 3.3 |
| Intern-Decision-4B | in-process on the same L40S | 414 | 1,648 | 2.4 |
| Jev (typesafe/jev-1.13) | OpenRouter API | 157 | 161 | 6.2 (c = 1) / 22.9 (c = 4) |

Per question (3 per request) JevAgentRL takes about 100 ms at c = 1 against about 50 ms for Jev, and its engine
serialises requests, so latency grows linearly with concurrency. In the live slice a 69-option tool_select call (knockout)
took 0.3-0.9 s p50 under 8 concurrent episodes vs 0.16 s for Jev. The goal "much lower latency than Jev" is **not** met on
this hardware and serving stack; a faster GPU and batching across concurrent requests in serve_jev are the levers.

## 6. Spend

| item | $ |
|---|---|
| GPU: g6e.4xlarge 2026-10-02 20:11 -> 2026-10-03 11:15 UTC (15.1 h x $3.004) | 45.3 |
| E2 dev slices (Flash / Flash-Lite System-2): references Jev + llm_only 11.6, Intern 9.1, SFT v0 10.5, SFT v1 8.7, OPD i12 9.0, OPD 8.8, GRPO i15 8.8, GRPO 8.6, frozen 8.7; superseded seed 0-49 references 12.0 | 95.7 |
| OPD: executor rollouts 10.2, greedy dev evaluations 1.5, Jev teacher 0.38 (6,549 requests, 12,414 labelled decisions) | 12.1 |
| GRPO: executor rollouts (1,920 episodes) 25.3, greedy dev evaluations 1.2 | 26.5 |
| tau-hat\* calibration arms (300 episodes) | 8.0 |
| **API total** (ledger; latency probes of Jev < $0.05 not included) | **142.4** |

GPU budget $250 (training reservation 48 GPU-h): used 15.1 GPU-h / $45. API budget $200: used $142. The stopped instance
keeps its 500 GB gp3 volume (about $70 / month incl. provisioned throughput / IOPS) until it is terminated; every artefact
needed later is on S3.

## 7. What the evaluation (phase 2, another session) needs

- Artefacts: `s3://<bucket>/checkpoints/round5/jevagentrl-4b/` (adapter, tokenizer, chat template,
  decision config, `calibration.json`, `FROZEN.txt`); serving code at commit c6c1cc1 (`training/serve_jev.py`); verify the
  sha256 in `FREEZE_RECORD.json` before serving (A7.10 (b): operational repairs must reproduce the frozen predictions).
- Serve: `python -m jevharness.training.serve_jev --model Qwen/Qwen3-4B --adapter <dir> --calibration <dir>/calibration.json
  --name jevagentrl-4b --port 8081`; backend spec `{"type": "jev", "name": "jevagentrl-4b", "base_url":
  "http://127.0.0.1:8081", "model": "jevagentrl-4b", "api_key": ""}`.
- A7.10 (c) S2-3: `rl_ts_replace_f` (and the Jev / Intern arms) on fresh seeds 300000-300199, Flash, three passes, then the
  common-state replay; registered estimate ≈ $31 API + ≈ 2 GPU-h. A7.13: base / SFT v1 rows (no OPD row: OPD does not
  precede the frozen checkpoint) on the common-state corpus and E1, live `*_ts_replace` on 300000-300099 for the base and
  SFT v1 (≈ $13), the provisioned tariff workload (≈ $2-5), ≈ 3 GPU-h. The calibration arms are done (τ̂\* above); the
  Stage-1-seed mirror (`configs/experiments/jevgym_tool_select_rl.yaml`) is not run (A7.13 (v)).
- Memory: one served checkpoint ≈ 9 GB + session cache (≈ 147 KB per cached token at the default 65,536 tokens ≈ 9.6 GB);
  Intern-Decision-4B in-process ≈ 10 GB; both fit one L40S together, on one L4 load them sequentially (A7.10 (c)).
