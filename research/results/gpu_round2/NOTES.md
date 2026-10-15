# GPU round 2 (2026-10-01, AWS g6.4xlarge / L4 24 GB, commit 5d95966; script `scripts/aws/gpu_round2.sh`)

Pilot-scale outputs for the second System-1 family (PREREGISTRATION A5.1) and the serving-architecture study.
Nothing here is a headline number: the E2a slice is n = 10 tasks on the *pre-A6* seed range (0-9, `d60-near-strict`),
E1 is the 299-item dev prefix, the serving bench is partial. Trajectories (`e2a_intern2b_dev/*.jsonl`, 18 MB)
and the full E1 records stay on `s3://<bucket>/gpu_round2/`; summaries are copied here.

## 1. Competence gate (A5.1): `scripts/gate_s1_dev.py` on `data/train_v0/dev.jsonl` (210 items)

| System-1 | acc | tool_select | tool_params | stop | wall s | gate (>= 0.8 x Jev 0.971 = 0.777) |
|---|---|---|---|---|---|---|
| Jev (`results/pilot_jev/gate_jev_dev.json`) | 0.971 | - | - | - | - | reference |
| Intern-Decision-0.8B | 0.571 | 0.274 | 0.547 | 0.903 | 24 | FAIL |
| Intern-Decision-2B | 0.638 | 0.290 | 0.674 | 0.935 | 30 | FAIL |
| **Intern-Decision-4B** | **0.862** | 0.806 | 0.814 | 0.984 | 69 | **PASS** |

Only the 4B passes; it is the registered second family (`configs/experiments/jevgym_tool_select_intern.yaml`,
`s1: Intern-Decision-4B`). Latency on the L4: 69 s / 210 forward passes = 330 ms per decision at batch 1
(bf16, ~1k-token states), i.e. slower than Jev over OpenRouter (p50 ~ 200 ms) on this card.

## 2. E1 dev prefix (`e1_dev_intern/summary.json`; BFCL 85 tool_select + 142 tool_params, When2Call 72)

| System-1 | BFCL tool_select acc / ECE | BFCL tool_params acc / ECE | When2Call acc / ECE |
|---|---|---|---|
| Jev | 0.965 / 0.036 | 0.951 / 0.044 | 0.778 / 0.118 |
| Intern-Decision-4B | 0.953 / 0.079 | 0.746 / 0.082 | 0.694 / 0.119 |
| Intern-Decision-2B | 0.906 / 0.070 | 0.570 / 0.079 | 0.569 / 0.157 |
| Intern-Decision-2B, T = 1 | 0.906 / 0.072 | 0.570 / 0.241 | 0.569 / 0.267 |
| Intern-Decision-0.8B | 0.929 / 0.189 | 0.493 / 0.028 | 0.375 / 0.108 |
| BM25 | 0.612 / 0.158 | 0.310 / 0.155 | 0.292 / 0.597 |

The checkpoints' fitted temperatures (2B: 2.10, 4B: 1.99) matter for calibration (ECE 0.24 -> 0.08 on tool_params)
and not for accuracy. The 4B matches Jev on static tool selection and trails it on argument filling (-20 pts) and on
the When2Call deferral decision (-8 pts).

## 3. E2a slice, n = 10 (`e2a_intern_dev/report.md`, `scorecard.md`; System-2 = Gemini 3.5 Flash, `_s2lite` = Flash-Lite)

Episode success and the System-1's own tool_select accuracy on the states it was asked (shadow = same states as
`llm_only`; replace / cascade = the states its own choices produced):

| arm | success | errors | tool_select s1_acc (n) | note |
|---|---|---|---|---|
| llm_only | 10/10 | 0 | - | reference |
| jev_shadow | 10/10 | 0 | 0.933 (75) | ECE 0.04 |
| intern_shadow (4B) | 10/10 | 0 | 0.760 (75) | ECE 0.07; stop 0.933, params 0.872 |
| jev_ts_replace | 9/10 | 0 | 0.971 (69) | |
| intern_ts_replace (4B) | 1/10 | 0 | 0.407 (54) | skips `get_orders` after `find_user`, calls `cancel_order` where gold is `ask_user` |
| intern2b_ts_replace | 1/10 | 0 | 0.156 (87) | |
| intern08_ts_replace | 0/10 | 0 | 0.045 (160) | 70 options -> two knockout rounds |
| jev_ts_assist_k3 | 8/10 | 0 | 0.798 (89) | |
| intern_ts_assist_k3 | 6/10 | 4 OOM | 0.718 (85) | |
| intern_ts_cascade_tau0.9 | 2/10 | 5 OOM | 0.648 (54), defer 0.69 | |
| intern_ts_cascade_tau0.99 | 5/10 | 5 OOM | 0.695 (59), defer 0.98 | |
| jev_ts_guard / intern_ts_guard / intern4b_uncal_ts_guard | 10/10 each | 0 | guard acc 0.973 / 0.961 / 0.961 | |
| intern_stop_guard | 10/10 | 0 | stop guard acc 0.833 (12) | |

**Errors.** Every error is `torch.OutOfMemoryError` (`e2a_intern2b.log` has no traceback: the runner records the
exception per task). Cause: the experiment binds four Intern specs (4B calibrated, 4B with `temperature: 1`, 2B, 0.8B)
and the backend loaded each spec's weights separately, so two 4B copies + 2B + 0.8B (~ 24 GB bf16) filled the 24 GB
card before the first long state. Fixed in `decision/intern_decision.py`: instances with the same checkpoint /
device / dtype share one copy of the weights, forward passes on a device are serialised across instances, and an
OOM is retried once after `torch.cuda.empty_cache()` (tests in `tests/test_intern_decision.py`). The arms with
errors are re-run in round 3 (`scripts/aws/gpu_round3_train.sh`).

**Reading.** On the states the harness actually visits, the 4B is a weaker *decider* than Jev (shadow s1_acc 0.76
vs 0.93; replace 0.41 vs 0.97), although both score 95-97 % on BFCL tool selection: its errors are sequential
(skipping the lookup step, acting instead of asking), i.e. exactly the in-harness decisions E1-style static sets do
not test. As a *guard* it is as good as Jev (0.96 vs 0.97 on proposal acceptance, episodes 10/10). So within A5 the
second family's guard / stop / shadow rows look like Jev's, its replace and cascade rows do not; both are reported as
what they are. Two cautions (review round 6): guard *accuracy* on mostly-correct proposals does not show that the
guard detects wrong proposals (the registered natural / probe FAR and FRR do), and the assist / cascade rows with
4-5 OOM tasks are a systems failure, not a model-quality estimate (re-run in round 3). JevGym pilot arms say nothing
about H5, which is the registered hard-task experiment. The static-vs-interactive gap is the motivation for
harness-level training (round 3: SFT on v0, then OPD) and is pre-registered as a reportable finding (A6.12).

**Accounting fix found here (applies to Jev too).** Deferred `tool_params` decisions in cascade / replace arms were
logged with `final_value: null` although System-2's forced call carried the argument (79 of 449 in the Jev n=50
pilot); `harness/agent.py` now back-fills `s2_value` / `final_value` from the executed call (test in `tests/test_agent.py`).
Reports generated before this fix undercount cascade `final_acc` on tool_params.

## 4. Serving bench (`bench_serving.log`, partial: killed after 85 min at the 64k cells; Qwen3-0.6B dense, L4)

ms per decision (p50) and prefilled tokens per decision, 6-step synthetic episodes, 5 repeats:

| state | questions | batch | stateless | prefix (shared prefill) | session cache | reuse (session) |
|---|---|---|---|---|---|---|
| 1k | 1 | 1 | 30 ms / 891 tok | 30 / 891 | **96** / 471 | 1.9x |
| 1k | 3 | 1 | 38 / 748 | 25 / 328 | 36 / 189 | 4.0x |
| 4k | 1 | 1 | 85 / 2376 | 82 / 2376 | 104 / 966 | 2.5x |
| 4k | 3 | 1 | 125 / 2238 | 47 / 824 | 50 / 354 | 6.3x |
| 16k | 1 | 1 | 408 / 8312 | 407 / 8312 | 436 / 2945 | 2.8x |
| 16k | 1 | 8 | 840 / 8308 | 841 / 8308 | (evictions: cache 75k tokens < 260k needed) | - |

Honest reading: the shared-prefill path pays off as designed (3 questions: 2.3-2.7x fewer prefilled tokens, 2.7x
lower latency at 4k). The session cache cuts prefilled tokens 1.9-6.3x but is **not faster** on this dense 0.6B
model at <= 16k: a fixed ~70 ms overhead per request (cache clone / crop / device transfer) eats the saving. The
hybrid (Qwen3.5) rows and the 64k cells did not run. Before the paper says anything about session serving, the
overhead has to be profiled and removed (round 3 re-runs the compact bench 1k/4k/16k at batch 1 after that).
