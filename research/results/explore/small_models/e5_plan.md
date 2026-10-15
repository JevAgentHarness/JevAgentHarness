# Plan: exploratory row E5, small decision models (GPU first; an optional small live row)

Written 2026-10-06 by a research session. **Nothing here has been launched, committed or registered.** Model choice
and reasons: [`survey.md`](survey.md) §5. Built on the E3 driver and plan
([`results/training/ablation_plan.md`](../../training/ablation_plan.md), §§4–6 and the addendum §11). Every time and
cost below is an **estimate**, and the basis of each is stated.

## 0. Summary

| | |
|---|---|
| question | Does the frozen SFT v1 recipe carry the 4B's in-generator accuracy down to 1.7B, ~1B and 0.6B, and how much System-1 latency does that buy on the same GPU? |
| models | `Qwen/Qwen3-1.7B`, `openbmb/MiniCPM5-1B`, `Qwen/Qwen3-0.6B` (backup: `ibm-granite/granite-4.0-1b`); reference: the 4B (E3's D100 retrain, else the SFT v1 anchor; the frozen JevAgentRL-4B for latency and live) |
| training | SFT v1 recipe on E3's D100 file, unchanged except the base model; **no GRPO** (§7) |
| offline evaluation ($0 API) | exactly E3's: v1 dev / dev_recovery / dev_knockout, temperatures on v1 dev, replay of corpus C (`results/stage2/s23/`) through `serve_jev` |
| latency | p50 / p90 ms per request at c = 1 and c = 4 on one A10G, every model and the frozen 4B in one session |
| optional live row | final-harness profile with each small model, n = 100 fresh JevGym tasks on **seeds 300700–300799**, ≈ $40 API (≈ $21 with the offline τ̂\* fit) |
| GPU (A10G, estimate) | **≈ 9–11 h** offline + latency (cap 14 h), + ≈ 2.5 h for the live row; **≈ $22–27** on the g5.8xlarge ($2.448 / h), ≈ $15–18 on a g5.4xlarge |
| timeline | Deviations entry and driver 10-06; GPU chain 10-07 (after E3's chain); live row 10-07 / 10-08; paper text 10-08; buffer until 10-12 |

## 1. Questions and what stays fixed

1. **Size curve of the recipe.** v1-dev accuracy (by decision type, after an off-plan step, knockout) and accuracy on
   the common-state corpus C (all, after an off-plan step) for 0.6B / 1.7B / 4B of one family, plus one non-Qwen ~1B
   model.
2. **What the size buys.** System-1 latency per request on the agent's real request stream, at c = 1 (a dedicated
   GPU) and at c = 4 (the shared-GPU load of the final harness).
3. *(Optional)* **Does it turn the shared-GPU harness from slower into faster than LLM-only at the same success?**
   With the 4B, the shared-GPU final harness took 22.39 s per episode against LLM-only's 8.70 s. System-1 accounted
   for 16.41 s of that: 3.85 s of compute and 12.44 s of waiting for the engine (`results/harness_final/summary.md`).
   Back-of-envelope only: a model with 0.2–0.3× the service time cuts compute to about 1 s, and most of the queueing
   goes away. That would put an episode near 7–9 s, i.e. LLM-only's time or better. The live row measures this.

**Fixed (the SFT v1 recipe, as in E3 §1 and Deviations E3 (c)).** Training file = E3's D100
(`runs/train_ablation/data/D100.jsonl`: 15,694 rows, 587 tasks, sha256 `65fa28a1…` in
`results/training/ablation_inputs.json`) + the When2Call replay at factor 0.45. The resulting mix must be
byte-identical to SFT v1's `train_mix.jsonl` (sha256 `9fe1088d…`); the mix does not depend on the tokenizer, so the
check applies unchanged. Recipe: LoRA r 32 / alpha 64 on all linear layers, soft CE + 0.5 Brier, fresh option
permutation per row, LR 1e-4 cosine, warm-up 3 %, 2 epochs, 64 rows per optimizer step, 3,072-token micro-batches,
bf16, `MAX_LEN` 4,096 (tail kept), gate set `data/train_v0/dev.jsonl`, trainer seed 0. Memory fallback exactly as E3
§11.7. **Only the base model changes**, at pinned revisions:

| arm | base model | revision (HF, 2026-10-06) | params total / non-emb. | why |
|---|---|---|---|---|
| `e5_q17` | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 1.72 / 1.41 B | same family, main small point |
| `e5_mc1` | `openbmb/MiniCPM5-1B` | `87179e5c1f455ef22e6223592d2d61351b525bfc` | 1.08 / 0.68 B | non-Qwen ~1B, Apache-2.0, plain Llama |
| `e5_q06` | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | 0.60 / 0.44 B | same family, low end |
| `e5_gr1` (backup) | `ibm-granite/granite-4.0-1b` | `6a7381ba1f54d684ff508d991aeb7dc580157103` | 1.63 / 1.43 B | runs only if `e5_mc1` fails its smoke |

The LR was set for the 4B. Keeping it is the recipe "as a user would run it" on a smaller base, and the text says so.
An LR sweep is not part of E5.

## 2. Offline evaluation (GPU only, no API; identical to E3 (d))

Per arm: `eval_decisions` (raw) on `data/train_v1/{dev,dev_recovery,dev_knockout}.jsonl.gz`; temperatures fitted on
v1 dev (`training/calibrate.py --min-bucket 200`); C replayed through `serve_jev` (`--fast`, or the plain server if
E3's `serve_mode.txt` says the anchor needed it) into a copy of the corpus with a per-arm cache. Read-outs: v1
accuracy by decision type, recovery, on-plan, knockout, raw ECE; C accuracy (task mean, all and after an off-plan
step) and ECE with the arm's temperatures. One descriptive stratum is added: C occurrences whose oracle answer is
`ask_user` (582 occurrences, 100 tasks; the ask-vs-act decision). The published rows on it, computed for this plan
from the committed answer files, are JevAgentRL-4B 100.0, Jev 64.4, Intern-Decision-4B 59.6 and base Qwen3-4B 47.3
(see `coverage.md`).

The 4B reference: E3's `D100` (retrained on the same host and stack) when it exists, else E3's `sft_v1` anchor
(cross-stack, bounded by E3's anchor check). C is used post hoc and descriptively only, as in E3: no checkpoint,
threshold or model is selected on it.

## 3. Latency microbenchmark (same A10G host, one session)

Each model is served alone by `serve_jev --fast` with its adapter merged and its temperatures; the frozen
JevAgentRL-4B is served in the same session as the reference (sha256-checked as in `harness_final_gpu.sh`).

| instrument | command | output |
|---|---|---|
| (a) synthetic, round-5 protocol | `scripts/measure_latency.py --s1 '<jev spec of the served model>' --concurrency 1 4 --n 200 --state-tokens 1500 --questions 3` | p50 / p90 ms per request at c = 1, 4 (**add p90**: the script prints p50 / p95; one line) |
| (b) the final harness's request stream | `scripts/harness_final.py bench --url http://127.0.0.1:8082 --label e5_<arm> --path new --concurrency 1 --limit 25`, then `--concurrency 4` (100 episodes) on the committed `results/harness_final/replay_workload.jsonl.gz` | per-request ms by decision point (tool_select, tool_params) and System-1 ms per step; **add p90** from the per-request records (`by_dp` reports mean and p50 today) |
| (c) serving facts | `/health` of each server (`cache_info`) + `nvidia-smi` sampling (as `harness_final_gpu.sh bench`) | croppable, KV bytes / token, peak memory, GPU utilisation |

The request stream in (b) is fixed by the log: tool choice plus all argument questions of the logged step, and
69-option knockouts chunked by option count. It does not depend on the model's answers, so every model faces the
same work. The "decision flips" column of the bench is meaningless for a new model and is ignored. Expected
(estimates, survey §3.2): c = 1 latency 0.40–0.50× the 4B for 1.7B, 0.20–0.30× for MiniCPM5-1B and 0.6B; at c = 4,
larger gains, because the 4B's c = 4 time is mostly queueing.

## 4. Optional live JevGym row (needs the author's OK for API spend)

- **Seeds: 300700–300799 (n = 100).** This is inside the held-back spare block 300500–300999 of A7.2 (a), which
  only a dated A7 deviation can open; 300500–300699 are used by harness_final and its dedicated row (checked: no
  config or result uses 300700+). 300800–300999 stay spare. Frozen suite settings (`families: v2`,
  `distractor_style: near`, `n_distractors: 60`, `noise_lines: 20`, `strict: true`), `JEV_EVAL_RUNS=1`. The runner
  refuses the block without that variable, and the training guards already exclude 300000–300999.
- **Thresholds (τ̂\*).** Each small model gets its tool_select and tool_params thresholds from the registered
  calibration command. This is a shadow arm on calibration seeds 5000–5149 (n = 150, Flash, shadow at tool_select /
  tool_params / stop; a copy of `configs/experiments/jevgym_shadow_cal_startlux.yaml` with the E5 backend), then
  `python -m jevharness.bench.threshold_theory … --calibration-arm …` per decision point, then
  `scripts/fill_backend_tau.py --backend e5_<arm>` into a copy of the final profile
  (`configs/harness/explore_e5_final.yaml`: tool + argument cascade, reflection off, `batch_s1: true`). The frozen
  4B keeps its frozen thresholds. If a fit accepts no level at either point, the block for that model is **not run**,
  and is reported as "always defers, equals LLM-only by construction".
  *$0 alternative:* in shadow mode System-1 never acts, so the trajectories do not depend on it, and the recorded
  shadow questions of an existing Flash calibration trajectory (e.g. `runs/e2a_jevgym_shadow_cal_startlux/`) can be
  re-answered by each small model offline. This needs a small replay tool, and the shadow records must carry the
  full wire question (check first). If used, the Deviations entry says so.
- **Blocks.** One per System-1 (`e5_q17`, `e5_mc1`, `e5_q06`, and the frozen 4B as the in-session reference), each
  with arms `llm_only` and `harness_final` on the same 100 tasks: task-major interleaved (arm order seeded per task,
  seed 732), workers 8 (the shared-GPU provisioning of harness_final, where the 4B was slower than LLM-only), three
  passes (passes 2–3 re-run infrastructure errors only), one model on the GPU per block. Each block has its own
  `llm_only` arm, so drift in API latency between blocks does not enter a comparison.
- **System-2.** Gemini 3.5 Flash, temperature 0, minimal reasoning. By the spend policy, use the
  **OpenRouter** route for both arms of every block. Dollar figures are then comparable within E5 but not with
  harness_final's direct-API rows: the route smoke reported about half the input tokens
  (`results/harness_final/route_smoke.json`). If comparability with harness_final matters more, use the direct API
  (same order of cost, inside the remaining Gemini cap).
- **Tariff.** Per model, the harness_final tariff protocol: a window on development seeds 600–699 at c = 4,
  q = $/h × window hours / completed episodes.
- **Analysis.** `scripts/harness_final.py analyze` per block (success per arm and the paired difference, cost incl.
  q, wall-clock and model-call time per episode, the System-1 decomposition into compute and waiting, coverage).
  Descriptive; no test.
- **API cost (estimate).** Per block: 100 × ($0.0434 LLM-only + $0.0059 harness) ≈ $4.93; four blocks ≈ $19.7.
  τ̂\* fits: 3 × 150 × ≈ $0.043 ≈ $19.4 ($0 with the offline alternative). Smokes (5 development episodes per block)
  ≈ $1. **Total ≈ $40 (≈ $21 with the offline fit); n = 200 ≈ $60. Cap: $60.** GPU ≈ 2.5 h: fits 3 × ~20 min,
  tariff windows 3 × ~10 min, four blocks × ~35 min.

## 5. GPU hours and cost (A10G; estimates)

Basis. E3's D25 SFT on the 4B took 136 min for 4,126 rows (incl. replay) with activation checkpointing on one A10G,
i.e. 1.98 s per row over 2 epochs. That extrapolates to **557 min** for D100's 16,902 rows on the 4B. Scaled by
**total parameter count** as instructed (the conservative choice: compute scales with non-embedding parameters,
which would give 0.39 / 0.19 / 0.12 instead of 0.43 / 0.27 / 0.15). E3's evaluation + replay took 61 min per 4B
checkpoint; it is scaled the same way, with a floor of 10–15 min for model load, server start and 3,939 replay
requests. Models ≤ 1.7B probably fit the recipe's 3,072-token micro-batches without checkpointing on 24 GB
(Qwen3-1.7B: ~3.4 GB bf16 weights and roughly 0.6× the 4B's activations per token), which saves ~23 % (E3 §11.4's
1.3 factor).

| step (minutes) | `e5_q17` (×0.43) | `e5_mc1` (×0.27) | `e5_q06` (×0.15) | 4B ref | total |
|---|---|---|---|---|---|
| SFT D100, with checkpointing | 238 | 150 | 83 | – | 471 |
| SFT D100, without checkpointing | 183 | 115 | 64 | – | 362 |
| load / save | 5 | 5 | 5 | – | 15 |
| eval + temperatures + C replay | 30 | 20 | 15 | – (E3) | 65 |
| latency bench (a)–(c) | 15 | 15 | 15 | 15 | 60 |
| setup, pinned downloads, smoke (`MAX_STEPS=4` per model) | | | | | 45 |
| **total** | | | | | **656 min ≈ 10.9 h** (≈ 9.1 h without checkpointing) |

| host | hours | $ |
|---|---|---|
| g5.8xlarge (E3's host, $2.448 / h) | 9.1–10.9 | **22–27** |
| g5.4xlarge ($1.624 / h) | 9.1–10.9 | 15–18 |
| + optional live row (GPU) | ≈ 2.5 | + 4–6 |

Cap: 14 GPU-hours for the offline block and latency (each arm is projected before it starts, as in E3 §11.6; a unit
that does not fit is reported as not run), plus 3 h for the live row. API: $0 offline; ≤ $60 live.

## 6. Driver and code changes (small; offline tests)

1. `scripts/aws/train_ablation.sh` gains E5 stages (or a copy `scripts/aws/e5_small.sh`): `e5_inputs` downloads
   each base at its pinned revision into `$W/base_e5/<arm>` (ungated repos, no HF token) and records its sha256;
   `sft_run`, `eval_ck`, `replay_ck` and `serve_abl` take a per-arm `BASE` instead of
   `$W/base_model_path.txt`; `e5_latency` serves each model plus the frozen 4B and runs §3 (a)–(c); `e5_final`
   checks cost 0 and powers off. The box refuses E5 stages without the "Exploratory row E5" entry in its checkout,
   as E3 does.
2. `scripts/measure_latency.py` and `scripts/harness_final.py bench`: add p90.
3. Scoring: `scripts/train/ablation_score.py` learns the E5 arms (or a thin `scripts/train/e5_score.py`) →
   `results/explore/small_models/e5.{json,md}`, keys `e5/<arm>/{v1_acc, v1_tool_select, v1_tool_params, v1_stop,
   v1_recovery, v1_knockout, v1_ece, c_acc, c_acc_offplan, c_acc_ask, c_ece, sft_min, lat_c1_p50, lat_c1_p90,
   lat_c4_p50, lat_c4_p90, lat_ratio_c1, lat_ratio_c4}`, contrasts `e5/contrast/<arm>_vs_4b/c_acc_{delta,lo,hi}`
   (paired task bootstrap over C's 200 tasks, B = 2,000, seed 0), with `--tex paper/generated/explore_e5.tex`.
4. Local preflight before the launch (CPU, $0): `results/explore/small_models/preflight/check_readout.py`, already
   run (all three models pass); a dry run of the driver (`DRY_RUN=1`).

Box side, per arm (as E3 §4.2; `$W=/opt/jev`, `HV=$W/hvenv`, `A=runs/e5_small`, `PORT=8082`):

```bash
. "$HV/bin/activate"; BASE=$W/base_e5/$ARM            # pinned snapshot of the arm's model
GPUS=1 RECIPE=r3 MODEL="$BASE" TRAIN=runs/train_ablation/data/D100.jsonl DEV=data/train_v0/dev.jsonl \
  REPLAY=data/train_v1/replay_when2call.jsonl.gz REPLAY_FACTOR=0.45 OUT=$W/ck_e5/$ARM LR=1e-4 EPOCHS=2 BATCH=2 \
  EFF_BATCH=64 MAX_LEN=4096 SEED=0 PY=python \
  SFT_EXTRA="--max-batch-tokens 3072 --eff-batch 64 --no-gradient-checkpointing" bash scripts/train/10_sft.sh
sha256sum $W/ck_e5/$ARM/train_mix.jsonl            # must equal SFT v1's 9fe1088d...
python -m jevharness.training.eval_decisions --model "$BASE" --adapter $W/ck_e5/$ARM \
  --dev data/train_v1/dev.jsonl.gz data/train_v1/dev_recovery.jsonl.gz data/train_v1/dev_knockout.jsonl.gz \
  --n-perm 0 --max-len 4096 --batch-size 16 --out $A/eval/${ARM}_v1.json
python -m jevharness.training.calibrate --model "$BASE" --adapter $W/ck_e5/$ARM --dev data/train_v1/dev.jsonl.gz \
  --out $A/cal/${ARM}_calibration.json --min-bucket 200
python -m jevharness.training.serve_jev --model "$BASE" --adapter $W/ck_e5/$ARM \
  --calibration $A/cal/${ARM}_calibration.json --name e5-$ARM --port $PORT --fast &      # wait for /health
for k in 1 2 3; do python scripts/stage2_corpus.py replay --corpus $A/s23 --name e5_$ARM \
  --s1 "{\"type\": \"jev\", \"name\": \"e5-$ARM\", \"base_url\": \"http://127.0.0.1:$PORT\", \"model\": \"e5-$ARM\", \"api_key\": \"\"}" \
  --cache cache/s1_e5_$ARM.sqlite; done
python scripts/measure_latency.py --s1 "{\"type\": \"jev\", \"name\": \"e5-$ARM\", \"base_url\": \"http://127.0.0.1:$PORT\", \"model\": \"e5-$ARM\", \"api_key\": \"\"}" \
  --concurrency 1 4 --n 200 --state-tokens 1500 --questions 3 --gpu-hour-usd 2.448 > $A/latency/${ARM}_synthetic.json
python scripts/harness_final.py bench --url http://127.0.0.1:$PORT --label e5_$ARM --path new --concurrency 1 --limit 25 --out $A/latency/bench
python scripts/harness_final.py bench --url http://127.0.0.1:$PORT --label e5_$ARM --path new --concurrency 4 --out $A/latency/bench
kill %1
```

## 7. GRPO: not worth it for E5

The attribution rows (`results/stage2/stage2.md`, A7.13) compare SFT v1 with the frozen SFT + GRPO checkpoint:
C 99.2 vs 99.2 %, after an off-plan step 93.8 vs 94.7 % (57 tasks), ECE 0.0106 vs 0.0100, live success on the first
100 tasks 100 vs 100 %, own-state accuracy 100 vs 100 %. GRPO cost 270 L40S-minutes (≈ 13 A10G-hours at E3's 2.9×
factor), $25.3 of Flash-Lite rollouts and $1.2 of evaluations (`results/gpu_round5/spend.json`). It also needs an
API key on the box, while E3 and E5 run key-free. SFT carried the gain. E5 therefore trains SFT only. A GRPO stage
for a small model would get its own Deviations entry, and only if its SFT leaves a clear gap after off-plan steps
(say C after-off-plan < 90 %).

## 8. Reading rules (fixed before the first E5 training step; descriptive, no test)

- Per arm, all numbers are development (v1 dev) or descriptive (C, latency, live) numbers; outside every Holm table.
- On C, a small model **keeps the 4B's accuracy** if the lower end of its paired 95 % interval against the 4B
  reference is ≥ −1.0 point, and **loses accuracy** if the upper end is < 0; otherwise inconclusive. The after-off-plan
  stratum (57 tasks) and the `ask_user` stratum are reported beside it with their own intervals. One seed per arm:
  seed noise is not measured for E5, and the text says so (E3's seed contrasts give its size for the 4B at D10 / D25).
- v1-dev differences below 0.5 points are described as within the saturated range (no per-row output, no interval).
- Latency is reported as measured p50 / p90 per request at c = 1 and c = 4 with the ratio to the 4B measured in the
  same session on the same GPU; never compared with numbers from another GPU type or session.
- Live (optional): success per arm with the paired difference and its interval (appendix only), cost incl. q, wall
  and System-1 time per episode; nothing is tuned or re-run on 300700–300799 after any result is read.
- No E5 checkpoint replaces JevAgentRL (A7.10) or JevAgentRL-v2 (A7.19), is frozen, or is served in another
  experiment.

## 9. Timeline (UTC; deadline 2026-10-12)

| when | what |
|---|---|
| 10-06 | Author decides. Append the Deviations entry (§11) before any E5 training step; `python scripts/stage2_spec.py` (regenerate `docs/STAGE2_EFFECTIVE.md`); driver changes and offline tests; `DRY_RUN=1`; commit; `check_no_secrets.py`; push |
| 10-06 late / 10-07 | On the E3 host after E3's chain ends (`WAIT_PREV=1`; E3's 24 GPU-h cap ends by about 10-06 22:00), or on a second A10G host if capacity exists: `rearm e5_inputs e5_smoke e5_arms "e5_q17 e5_mc1 e5_q06" e5_latency e5_final`, ≈ 9–11 h wall |
| 10-07 afternoon | score; if approved, τ̂\* fits and the four live blocks (≈ 3–4 h wall) |
| 10-08 | analysis; appendix paragraph + one sentence in §7 "Can a small open model learn the agent's decisions?" from `paper/generated/explore_e5.tex` |
| 10-09 – 10-11 | buffer, review rounds |

Order inside the chain: the 1.7B first (the main question), then MiniCPM5-1B, then 0.6B, then latency for all. If
the cap stops the chain, the finished arms are complete on their own (eval, temperatures, C answers, adapter on S3).

## 10. Risks

- **Saturation.** As in E3, every arm may sit near 99.7 % on v1 dev; C's after-off-plan stratum, the knockout file
  and the `ask_user` stratum carry the signal.
- **LR fixed at the 4B's value.** A small model can under- or over-fit at 1e-4 with LoRA r 32. E5 reports the recipe
  as is; an LR arm would be new work.
- **MiniCPM5-1B is new** (on HF since 2026-05-21, last modified 2026-08-17). If its smoke fails (load, memory,
  NaNs), the backup Granite-4.0-1B runs in its slot, and the swap is recorded.
- **Host contention with E3.** Same host and GPU; the chain waits for E3. A second host saves a day, but only if
  capacity allows (E3 found none for several types).
- **Latency on an A10G** differs from the L40S rows of Stage 2. All E5 latency comparisons are within one session
  on one GPU type.

## 11. Deviations entry (draft; to append to `docs/PREREG_A7_STAGE2.md` "Deviations" before any E5 training step)

- 2026-10-0X, recorded about HH:MM UTC, before any E5 training step, before any E5 checkpoint answers a question
  and before any episode on seeds 300700–300799. **Exploratory row E5: small decision models**
  (`results/explore/small_models/{survey,e5_plan}.md`). Outside both registered stages; descriptive, labelled
  "exploratory (E5)"; it revises no registered result, freezes no checkpoint, serves no E5 checkpoint in any other
  experiment and changes nothing in JevAgentRL (A7.10) or JevAgentRL-v2 (A7.19). (a) *Purpose.* JevAgentRL is
  Qwen3-4B; E5 asks whether the SFT v1 recipe keeps its in-generator accuracy at 1.7B, ~1B and 0.6B, and what that
  buys in System-1 latency. (b) *Models.* `Qwen/Qwen3-1.7B` (revision `70d244cc…`), `openbmb/MiniCPM5-1B`
  (`87179e5c…`), `Qwen/Qwen3-0.6B` (`c1899de2…`); `ibm-granite/granite-4.0-1b` (`6a7381ba…`) replaces MiniCPM5-1B
  only if its four-step smoke fails, and the swap is recorded. Chosen from public model cards and a CPU-only
  tokenizer and template preflight (`results/explore/small_models/preflight/`), before any E5 number existed.
  (c) *Data and training.* E3's D100 file (sha256 `65fa28a1…`, the SFT v1 rows) with the When2Call replay at factor
  0.45, whose training mix is checked against SFT v1's (`9fe1088d…`); no episode is run and no JevGym seed is drawn
  for training. The SFT v1 recipe unchanged except the base model (LoRA r 32 / alpha 64 on all linear layers, soft
  CE + 0.5 Brier, option permutation, LR 1e-4 cosine, warm-up 3 %, 2 epochs, 64 rows per step, 3,072-token
  micro-batches, bf16, gate set `data/train_v0/dev.jsonl`, trainer seed 0), with E3's memory fallback order. No
  GRPO. (d) *Evaluation, GPU only, no API.* As E3 (d): `eval_decisions` on v1 dev, dev_recovery and dev_knockout;
  temperatures on v1 dev; the common-state corpus C replayed through `serve_jev`, with the added `ask_user`
  stratum; C used post hoc and only descriptively. Latency: `scripts/measure_latency.py` (c = 1, 4; n = 200;
  1,500-token states; 3 questions) and the harness_final replay benchmark (`results/harness_final/
  replay_workload.jsonl.gz`, new call graph; c = 1 on 25 episodes, c = 4 on 100), p50 and p90 per request, every
  model and the frozen JevAgentRL-4B served by `serve_jev --fast` in one session on one GPU. (e) *Host and caps.*
  One A10G (the E3 host, after E3's chain, or another A10G host), no key on the box, any priced record stops the
  run; GPU ≤ 14 host hours for (c)–(d). (f) *Optional live row* (run only after the author's approval, recorded
  here before its first episode). Seeds **300700–300799** (n = 100) of the spare block of A7.2 (a), opened by this
  entry; frozen suite settings; `JEV_EVAL_RUNS=1`. Thresholds per model by the registered calibration command
  (shadow arm on seeds 5000–5149, `threshold_theory --calibration-arm`, `fill_backend_tau.py`) into
  `configs/harness/explore_e5_final.yaml` (the final profile: tool + argument cascade, reflection off, `batch_s1`).
  A model whose fit accepts no level at either decision point is not run. One block per System-1 (the three E5
  models and the frozen JevAgentRL-4B at its frozen thresholds): `llm_only` and `harness_final`, task-major
  interleaved (seed 732), workers 8, three passes (errors only), Gemini 3.5 Flash (temperature 0, minimal
  reasoning) via OpenRouter for every arm. Tariff per model as harness_final (f). API ≤ $60, GPU ≤ 3 h. Analysis
  as harness_final (g); nothing is tuned or re-run on these seeds after any result is read. (g) *Analysis.*
  `results/explore/small_models/e5.{json,md}`, keys `e5/...`: per model the v1 read-outs, C accuracy (all, after an
  off-plan step, `ask_user`) and ECE, paired task-bootstrap intervals against the 4B reference (B = 2,000, seed 0)
  with the reading rules of the plan §8, latency p50 / p90 at c = 1 and 4 and its ratio to the 4B; the live blocks
  if run. Development and descriptive numbers only; outside every Holm table.
