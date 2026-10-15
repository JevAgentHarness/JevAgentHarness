# JevAgentRL-v2: porting the System-1 post-training chain to Miles

**Status (2026-10-03 ~03:00 UTC): feasibility done, GO (conditional on a one-hour on-host smoke).** Exploratory second
track (amendment A7.19, being registered). It never replaces the frozen v1 JevAgentRL; S2-3 / S2-4 stay with v1.
Nothing in this file is an evaluation number. The barrier, seeds and mechanical rules live in
[`results/gpu_round6_miles/TRAINING_MANIFEST.md`](../results/gpu_round6_miles/TRAINING_MANIFEST.md), committed before
any training step.

Miles: https://github.com/radixark/miles, read at commit `49576fd` (2026-10-02). SGLang rollout engines + a Megatron-LM,
FSDP2 or torchtitan trainer, driven by Ray; SFT, GRPO / GSPO / PPO / REINFORCE++, on-policy distillation, LoRA
(Megatron only).

## 1. Answers to the feasibility questions

| question | answer | consequence |
|---|---|---|
| 4B dense on one multi-GPU host? | Yes. Qwen3-4B is Miles' quick-start model (`scripts/run_qwen3_dense.py`, `run_qwen3_0_6b_fsdp.py`); one node is the default layout (colocated or disaggregated). | 4 x L40S is enough: full fine-tuning of 4-5 B params under FSDP2 needs ~16 B/param = 75 GB spread over 3-4 GPUs. |
| L40S (sm_89)? | **Not on Miles' hardware list** (B/GB-series, H100/H200, A100 "FP8 off", AMD). The image is CUDA 13 / torch 2.13; FA2, Triton (fla) and SGLang's flashinfer / triton backends run on Ada; FA3 and DeepGEMM are Hopper-only and must not be selected. | First host stage is a smoke on the real box: image pull, `import miles`, SGLang serving the base with option log-probs, 4 FSDP SFT steps. Fallback host p4d.24xlarge (A100, supported). |
| StartLux-Decision-4B? | Verified on HF (commit `3890ae0`): `Qwen3_5ForConditionalGeneration`, `model_type qwen3_5` (hidden 2560, 32 layers, 3:1 Gated-DeltaNet : full attention, vocab 248,320, a vision tower), 4.66 B params, CC BY-NC 4.0, own readout over the letters A-Z (26 options per pass). That is Qwen3.5-4B's architecture, which Miles supports (Megatron recipe `Qwen3.5-4B`; FSDP spec `qwen3_5` with the packed-document reset for GatedDeltaNet). Our stack already loads this family text-only (`train_decision.load_causal_lm`, used for Intern-Decision-4B) and serves its non-croppable cache (`serve_jev`, §16). | Candidate base. Kernel caveat: without `causal_conv1d` / fla on the box the packed path silently drops per-document resets, so the smoke checks the kernels and the base rule falls back to Qwen3-4B if they are missing. |
| Custom rollout / reward / action space | `--rollout-function-path` replaces the rollout (plain `generate_rollout(args, rollout_id, data_source, evaluation)`); a rollout may return several **sibling samples per rollout** (`Sample.rollout_id`), and Miles normalises one shared reward per rollout and broadcasts it (episode-level GRPO is native). `Sample.train_metadata` reaches the trainer batch as `batch["metadata"]`; `--loss-type custom_loss --custom-loss-function-path` gets the full-vocabulary logits `[1, T, V]` of the micro-batch. | One sample = one System-1 decision: tokens = rendered chat prompt + one option-code token, response length 1; the custom loss reads the decision-position logits, restricts them to the presented option codes (exactly v1's readout) and computes the stage's loss. The categorical over the options of a typed question is therefore exact, not a full-vocabulary approximation. |
| OPD | Miles OPD = a per-token reverse-KL penalty subtracted from the advantages, teacher log-probs from an SGLang server with the **student's tokenizer** (`--opd-type sglang`, sampled-token or top-k "Rethinking OPD" estimate) or a Megatron teacher. | For a decision the response is one option code, so token-level OPD restricted to the option codes and renormalised *is* decision-level distillation. Miles' sampled-token estimator is the Monte Carlo version of a quantity we can compute exactly (the option set is enumerable), and its teacher must share the tokenizer and be an SGLang server; Jev is an API returning a distribution over options. So OPD runs inside Miles as a custom loss: exact forward KL from the teacher's option distribution (v1's `opd.opd_loss`), on states from Miles rollouts. |
| GSPO | Sequence-level ratio. With one response token per sample it coincides with token-level GRPO; an episode-level GSPO would need all decisions of an episode in one micro-batch, which Miles' scheduler does not guarantee. | GRPO (PPO-clip, clip-higher) only; stated, not run as a separate arm. |

**Verdict: feasible within the deadline** (freeze by 2026-10-07 12:00 UTC), with two conditions checked mechanically on
the host before any training step: the Miles smoke on the L40S box (else p4d.24xlarge), and the base rule (§4).

## 2. What runs where

```
data v1 rows ──(sft_rollout: permute, render, chat template, 1 code token)──► Miles trainer (FSDP2, custom loss: soft CE + 0.5 Brier)
                                                                                     │  --save-hf
JevGym tasks (training seeds) ──► harness_rollout ──► JevAgent episodes              ▼
        System-2: Gemini Flash-Lite (API)            System-1: SGLangDecider ◄── SGLang engine (weights synced by Miles)
        teacher (OPD): Jev via OpenRouter            samplers = v1 TorchDecider (replace / cascade / assist / guard)
                                     │
                                     └─► one Sample per decision (+ teacher target or GRPO advantage) ──► custom loss
HF checkpoint ──► training/serve_jev.py (Jev wire, unchanged) ──► eval_decisions / gonogo / E2 dev slice
```

- **Code** (`src/jevharness/training/miles_port/`, tested on CPU with a tiny Qwen3-style model and without Miles installed):
  `loss.py` (the custom loss for the three stages; slicing of packed / padded logits identical to Miles'
  `_iter_response_chunks` for CP = 1), `samples.py` (decision -> Miles `Sample` fields and back), `sglang_decider.py`
  (v1's `TorchDecider` with the forward pass replaced by SGLang `/generate` + `token_ids_logprob` over the option codes;
  log-softmax over the full vocabulary differs from the logits by a per-row constant, so the restricted softmax, the
  knockout and every sampler are unchanged), `rollout.py` (Miles rollout functions: `sft_rollout`, `harness_rollout`
  for OPD and GRPO), `export.py` (Miles HF save -> serve_jev directory + logit regression), `launch.py` (the Miles
  command lines).
- **Host**: the Miles Docker image (`radixark/miles`, digest recorded on the host) with our package installed
  `--no-deps` (its runtime deps, httpx / pydantic / pyyaml / dotenv / openai, are in the image). Evaluation
  (`eval_decisions`, gonogo, the E2 dev slice with `serve_jev`) runs in the project's own venv (`setup_gpu.sh`), as
  in v1, so the served model is read by exactly the code the evaluation side will use.
- **Layout.** Planned for 4 x L40S (SFT on four GPUs; OPD / GRPO: actor on three, one SGLang engine on the fourth).
  No 4 x L40S, A100 or H100 host had capacity in us-east-1 on 2026-10-03 02:20-02:52 UTC, so the host is a
  **g5.48xlarge (8 x A10G 24 GB, Ampere sm_86)**: SFT on 8 FSDP ranks (`--debug-train-only`, no engine); OPD / GRPO
  on 8 FSDP ranks colocated with one SGLang engine per GPU (`--colocate`: Miles offloads the trainer to host RAM
  while the engines generate); `max_tokens_per_gpu` 3072 (the full-vocabulary logits of a micro-batch dominate the
  activation memory on 24 GB). Attention: `--attn-implementation flash_attention_2`, SGLang `flashinfer` (never FA3,
  which is Hopper-only).

## 3. Stages (the manifest fixes the values)

1. **SFT v2** (Miles, custom loss `sft`): data v1 train + 0.45 x When2Call replay, soft CE + 0.5 Brier, random option
   permutation per row and epoch, full fine-tuning (FSDP has no LoRA), 2 epochs, LR 1e-5 cosine.
2. **OPD v2** (Miles rollouts, custom loss `opd`): init SFT v2; the six v1 OPD arms; teacher Jev; target 0.7 Jev + 0.3
   gold with the gold one-hot where Jev's argmax is not the gold (data v1's rule, as v1); forward KL; fresh training seed
   pool.
3. **GRPO v2** (Miles rollouts, custom loss `grpo`): init = the last kept stage; v1's four arms and samplers; reward
   `gated`, verbatim: R = 1[success] x (1 - 0.5 x min(1, C / 10)) - 0.2 x 1[irreversible error], C = System-2 calls
   ("call-count-penalised success"); group-normalised episode advantages (+ GiGPO anchors) computed in the rollout
   function exactly as v1's `compute_advantages`; PPO-clip 0.2 / 0.28 against the **stored SGLang behaviour
   log-probabilities**; no KL term (the reference distribution over options is not available inside Miles' loss; the
   declared KL estimator is "none").
4. **Validity gate (A7.10 (f))**: CPU tests (stored = replayed log-probs through the Miles loss path, samplers,
   masking, knockout) plus an on-host check at the first GRPO rollout: trainer-recomputed restricted log-probs vs the
   stored SGLang ones, mean |Δ| <= 0.02 and max <= 0.2 nats; failing it excludes GRPO (no RL claim).
5. **Gates and freeze**: `eval_decisions` + `data_v1 gonogo` (G1-G5) for every continuation, keep / retain / freeze by
   the manifest's rules (v1's, unchanged); temperatures on v1 dev; freeze record with sha256s.

## 4. Base and teacher

- **Base rule** (fixed before measurement): StartLux-Decision-4B iff (S1) Miles loads it under FSDP and 4 SFT steps run
  with finite loss, (S2) SGLang serves it and its option log-probs match the HF forward (max |Δ| <= 0.1 nats on 32
  v1-dev prompts), (S3) the fla and causal-conv1d kernels are active in the image, and (S4) zero-shot through
  `serve_jev`'s prompt its v1-dev accuracy is >= 0.766 (0.8 x Jev's 0.958) and >= Qwen3-4B's zero-shot accuracy on the
  same rows; else Qwen3-4B. (Licence: the CC BY-NC terms carry over to a StartLux-based checkpoint; research use.)
- **Teacher: Jev via OpenRouter** (decision-level distributions, as v1; ~$0.09 / 1k requests). StartLux-Decision-27B
  is not used: 54 GB of bf16 weights need two of the four L40S (SGLang TP > 1 is flagged as broken for Qwen3.5 in
  Miles' recipe notes), leaving too little for the actor and the engine; its quality on our states is unmeasured, and a
  second teacher would confound the comparison with v1.

## 5. Host, budget, timeline

- `g6e.12xlarge` (4 x L40S 48 GB, 48 vCPU, 384 GiB; ~$10.49 / h), else `p4d.24xlarge` (8 x A100 40 GB), else
  `p5.48xlarge`. Tag `Name=jev-rl-miles`, `Project=jev`; shutdown behaviour stop; 30 h power-off guard; idle stop.
  Script: `scripts/aws/gpu_round6_miles.sh` (launch / push / send / status / stop; touches only its own instance).
- Caps: GPU ~$400 (= ~38 h of g6e.12xlarge; plan ~20 h), API ~$100 (executor episodes ~$0.01, Jev teacher < $5,
  E2 dev slices ~$2 per arm of 50 tasks).
- Timeline: code + CPU tests + manifest (2026-10-03), smoke + base rule + SFT v2 + gates (2026-10-03/04), OPD and GRPO
  (2026-10-04/05), freeze by 2026-10-05 if possible, hard deadline 2026-10-07 12:00 UTC. If the Miles smoke has not
  passed on either host by 2026-10-04 06:00 UTC, the track stops with a no-go report.

## 6. What to expect (development evidence, v1 round 5)

v1's SFT v1 already scores 0.997 on v1 dev (0.998 on recovery states) and 1.000 success / System-1 accuracy on the
Flash E2 dev slice (seeds 1100-1149, n = 50 per arm): the development instruments are near their ceiling, so the
mechanical keep rules (OPD must add +1 pt v1-dev accuracy or +2 pts own-state accuracy) will rarely keep a continuation.
That is a property of the instruments, recorded before any v2 number exists; v2's question is whether the Miles chain
(and possibly a decision-pretrained base) reaches the same development level, and what OPD / GRPO add where the
instruments still discriminate (recovery states, Flash-Lite arms, cost).
