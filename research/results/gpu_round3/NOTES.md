# GPU round 3 (2026-10-02, one NVIDIA L4 24 GB, <instance-id>)

Development and calibration data only: nothing here is an evaluation number or a paper headline. Script
`scripts/aws/gpu_round3_train.sh` at commit 5820767; outputs in `s3://<bucket>/gpu_round3/` (small artefacts copied
here: `status.txt`, `bench_serving.{json,md}`, `r2_eval_dev.json`, `opd_metrics.jsonl`, `opd_summary.json`).
Stage log (`status.txt`): serving bench 02:38–02:51, Intern calibration arms 02:53–04:07, R2 SFT 04:07–05:30,
R2 eval 05:30–05:33, OPD pilot 05:33–05:57; round 3b (Intern OOM re-run) chained after it.

## 1. Two stages failed on the first attempt (both bugs fixed, commit c939e87)

- **Serving benchmark**: all cells measured; writing the markdown crashed on stateless cells, which carry no phase
  profile under `--profile`. The JSON was already saved; `scripts/bench_serving.py --render-only` rebuilt the table.
- **Intern E2a OOM re-run**: the runner refused to resume round 2's directory because Intern-Decision's backend spec
  id changed with the shared-weights / OOM-retry fix (resuming would mix two backends in one arm). It re-runs from
  scratch into a new directory as round 3b (`scripts/aws/gpu_round3b_intern_rerun.sh`, dev seeds 0–9).

## 2. Serving: the fused single-pass session path (`bench_serving.md`)

Qwen3-0.6B, batch 1, ms per decision p50 (stateless / prefix / session):

| state tokens | questions | stateless | prefix | session | session speed-up |
|---|---|---|---|---|---|
| ~1.1k | 1 | 31.2 | 31.0 | 37.4 | 0.83× |
| ~1.1k | 3 | 35.6 | 24.9 | 25.6 | 1.39× |
| ~4.1k | 3 | 123.3 | 45.7 | 40.3 | 3.06× |
| ~16k | 3 | 846.7 | 220.1 | 182.2 | 4.65× |

Round 2 measured the session path at 96 ms vs 30 ms stateless for one question at 1k tokens; the fused path brings
that to 37 vs 31 ms. The session cache pays off from three questions per state and grows with state length. The
hybrid-attention model (Qwen3.5-0.8B, non-croppable cache) and MiniCPM5-1B rows are in `bench_serving.md`.

## 3. Intern-Decision calibration arms → the Intern C1 thresholds

`intern_shadow_cal` / `intern_shadow_cal_s2lite`, seeds 5000–5149, 150 tasks each, System-2 API $8.15. Fitted with
the registered command and written into the Intern C1 arms (PREREGISTRATION Deviations, 2026-10-02; A6.19):
tool_select never → always-defer on both System-2 models; tool_params 0.6309 (Flash), 0.8518 (Flash-Lite).

## 4. R2: Qwen3-4B LoRA SFT on train_v0

- 2 epochs, 4,813 s (80 min) on the L4, 1.01 examples/s, final train loss 0.63.
- `r2_eval_dev.json`: accuracy 0.995 on the 210-decision gate dev set (tool_select 1.00, tool_params 0.988, stop
  1.00; ECE 0.019) and 0.987 on 300 When2Call dev items.
- **Reading.** The gate dev set comes from the same generator and task families as train_v0, so 0.995 measures fit
  to the v0 distribution, not competence in the harness. It must not be compared with Intern-Decision's 0.862 or
  Jev's 0.971 as evidence that R2 is a better System-1 (those two were not trained on v0).

## 5. OPD pilot (recipe R9) from R2: the in-harness gap, measured on our own model

`opd_metrics.jsonl`, `opd_summary.json`: 2 iterations × 16 training tasks (seeds ≥ 10000), Jev teacher via
OpenRouter, Flash-Lite executor, forward-KL loss; greedy dev rollouts on 32 tasks.

| | iter 0 (R2) | iter 1 | iter 2 |
|---|---|---|---|
| rollout tool_select accuracy vs gold | – | 0.45 | 0.68 |
| rollout tool_params accuracy | – | 0.97 | 0.92 |
| teacher (Jev) accuracy on the same decisions | – | 0.97 | 0.90 |
| greedy dev decision accuracy | 0.654 | – | 0.570 |
| greedy dev task success | 0.69 | – | 0.66 |

- R2's tool-selection accuracy on the states it visits itself is 0.45 against 0.97 for the teacher, while it scores
  1.00 on static tool-selection items: the same static-to-sequential gap Intern-Decision-4B showed in round 2
  (0.81 static, 0.41 in replace). Training on logged, success-only states (train_v0) does not transfer to the states
  a System-1 reaches when it acts.
- Two iterations of 16 tasks are far too few to move the greedy dev numbers (the drop from 0.654 to 0.570 is within
  noise at 32 tasks); the pilot measures cost and time, not the effect.
- Cost and time: 211 teacher calls for $0.014 in total; rollout 49–122 s per iteration of 16 tasks. OPD at the scale
  of `scripts/train/12_sft_v1.sh` (hundreds of tasks per iteration) is therefore limited by rollout time, not by
  teacher cost.
- The pilot's greedy dev rollouts ran on seeds from 1000 (the config's 5000 was patched on the host before the stage
  started; never the calibration or evaluation ranges).

## 6. What this changes for round 4

- Train on v1 (`data/train_v1`: in-harness states, recovery states, guard questions), not v0.
- The go / no-go must not rest on the gate dev set for models trained on v0 / v1 data; G3 (v1-dev in-harness states)
  and the E2 dev slice decide (`docs/TRAINING_RECIPES.md` §18.4, updated).

## 7. Round 3b: the Intern E2a slice re-run without out-of-memory errors (`e2a_intern_oomfix_dev/`)

Same n = 10 development slice as round 2 (seeds 0–9, `suite.seed0=0`; never the calibration or evaluation ranges),
run 05:57–06:24 on commit 5820767 (it carries the shared-weights / OOM-retry fix of c939e87). Trajectories (19 MB)
stay on the bucket under `gpu_round3/e2a_intern_oomfix_dev/`; `report.md`, `scorecard.md` and `experiment.yaml` are
copied here. Nothing here is a headline number.

**No arm has an error** (round 2: 4–5 out-of-memory episodes in each of the assist and cascade arms). The arms that
erred in round 2, with System-1 tool-selection accuracy on the states it was asked:

| arm | success | cost $/ep. | tool_select s1_acc (n) | defer rate |
|---|---|---|---|---|
| llm_only (reference) | 10/10 | 0.042 | - | - |
| intern_ts_assist_k3 | 10/10 | 0.027 | 0.760 (104) | - |
| intern_ts_assist_k5 | 10/10 | 0.024 | 0.755 (94) | - |
| jev_ts_assist_k3 | 8/10 | 0.022 | 0.830 (88) | - |
| intern_ts_cascade_tau0.5 | 1/10 | 0.005 | 0.469 (32) | 0.09 |
| intern_ts_cascade_tau0.7 | 2/10 | 0.010 | 0.611 (36) | 0.31 |
| intern_ts_cascade_tau0.8 | 2/10 | 0.014 | 0.571 (42) | 0.45 |
| intern_ts_cascade_tau0.9 | 7/10 | 0.029 | 0.714 (70) | 0.60 |
| intern_ts_cascade_tau0.95 | 9/10 | 0.034 | 0.767 (73) | 0.79 |
| intern_ts_cascade_tau0.99 | 10/10 | 0.039 | 0.747 (75) | 0.99 |

The other arms reproduce round 2: shadow tool-selection accuracy 0.760 (Intern) vs 0.933 (Jev) on the same states,
replace 1/10 (Intern, s1_acc 0.42) vs 9/10 (Jev, 0.94), guards 10/10 with guard accuracy 0.96 (Intern) and 0.99 (Jev).

**Reading.**
- The cascade sweep is the in-harness picture behind the fitted Intern C1 policy (tool selection always defers;
  `results/calibration/tauhat_intern_*`): every threshold that accepts a meaningful share of Intern's tool choices
  loses episodes, and success returns to LLM-only only as the defer rate approaches 1, where the cost does too.
- Assist is where Intern helps: with the LLM choosing among its top-k, both k = 3 and k = 5 kept 10/10 at about
  60 % of LLM-only cost on this slice. At n = 10 this separates nothing from Jev's 8/10 (C2 is the registered test).
- The out-of-memory fix holds on the 24 GB L4 with four Intern specs bound in one experiment.

The instance stopped itself after the stage (`shutdown +2`); state checked from here.
