# Training roadmap: open decision models for agentic decision points (RQ5)

**Scope.** This is the plan of record for training open System-1 deciders: small causal LMs read out as a
restricted softmax over option codes. They answer the harness's typed `Choice` / `Noul` / `Score` questions and
are served behind the Jev wire API. It covers stages T0–T8 of [`ROADMAP.md`](ROADMAP.md) §6 (recipes R0–R8,
paper Table 3 / §7.5).

**Status (2026-10-01).**
- Every stage has code, a numbered script in `scripts/train/`, and a CPU test with a tiny Qwen3-style model
  (`tests/test_training.py`, `tests/test_training_rl.py`, `tests/test_training_kit.py`).
- **Operating the GPU box:** follow [`TRAINING_AGENT_BRIEF.md`](TRAINING_AGENT_BRIEF.md). It covers
  `setup_gpu.sh` (pinned stack + checks), `smoke_gpu.sh` (the whole chain 00 → 90 on Qwen3-0.6B), the stage order,
  the gates with `eval_decisions.py`, and the report template.
- **Starter data:** `data/train_v0` (committed, 12 MB) holds 1,435 gold-labelled JevGym decisions from training-seed
  Gemini episodes, stand-in teacher labels (Gemini Flash-Lite, not Jev), When2Call-train replay and 24 MC states.
  `data/train_v0/MANIFEST.json` has the provenance, and `scripts/train/make_data_v0.sh` regenerates it.
- The full script chain has been executed on CPU with a tiny model (`smoke_gpu.sh`, simulated executor, local
  teacher), including a live `serve_jev` → `jevbench ping` / `decide`.
- **New (2026-10-01): stage 65, on-policy distillation** (`opd.py`, `65_opd.sh`, §17): the student's own rollouts
  pick the states, the teacher labels them, KL over the option codes trains the student. Code + CPU tests
  (`tests/test_opd.py`) + smoke stage `65-opd`; no GPU run yet; a *candidate* for the training RQ, not pre-registered.
- **New (2026-10-02): training data v1 and the round-4 recipe** (§18): 15,694 Jev-soft-labelled decisions from the
  harness's own states in the evaluation setting (Flash-Lite shadow, near-miss replace and guard arms on training
  seeds 30000–30599, wrong-proposal guard rows, knockout-final variants, + v0), dev = 2,377 decisions on 90 held-out
  tasks; $5.28 (`results/training/train_v1/MANIFEST.md`). Round 4 runs `scripts/train/12_sft_v1.sh` (L4 / L40S presets)
  and decides with `data_v1 gonogo` (§18.4).
- **No GPU run exists yet, so this document contains no result numbers.** Compute figures are planning estimates,
  to be replaced after a 5% pilot.
- Numbers in the paper come only from `runs/` and checkpoints produced by these scripts.

Code map: `src/jevharness/training/`

| File | Role |
|------|------|
| `data.py` | Training format, harvesting helpers, MC target, `mix` / `stats` CLI |
| `harvest.py` | Gold / S2 labels from harness runs, When2Call-train, xLAM |
| `teacher_label.py` | Soft teacher targets from any backend, agreement filter, α-mix, cache, Ledger |
| `mc_rollouts.py` | Counterfactual Q̂ (sharded, threaded) |
| `decision_model.py` | Readout, losses, reference-adapter helpers |
| `train_decision.py` | HF Trainer: SFT / KL-to-MC / DPO / exact PG |
| `grpo_trl.py` | Decision-level GRPO (TRL, plus a pure-PyTorch fallback) |
| `grpo_online.py` | Episode-level GRPO inside the harness |
| `opd.py` | On-policy distillation inside the harness: student rollouts → teacher distributions at the reached states → KL (§17) |
| `calibrate.py` | Temperature per type × bucket |
| `serve_jev.py` | Jev-wire server (LoRA adapters and full fine-tuned checkpoints); prefix / episode cache (§16) |
| `eval_decisions.py` | Gate metrics on dev files through the served prompt: acc, NLL, Brier, ECE, AUROC, AURC, flip rate |

Scripts: `scripts/train/00_harvest.sh … 90_calibrate_serve_eval.sh` (see §11), plus `setup_gpu.sh`,
`smoke_gpu.sh` and `make_data_v0.sh` (see [`TRAINING_AGENT_BRIEF.md`](TRAINING_AGENT_BRIEF.md)).

---

## 0. Bottom line

1. **What we train.** A causal LM (Qwen3-4B main; 0.6B/1.7B for latency; 8B as the upper point) runs one prefill.
   Its next-token logits restricted to the option-code tokens A–Z, a–z give $p_\theta(a\mid s)$. Choices with more
   than 52 options use a knockout (chunks of 52, top-2 advance). The same prompt (`data.render` + `chat_text`) is
   used in training, calibration, GRPO rollouts and serving, byte for byte (§2).
2. **Default recipe.**
   - SFT with soft CE + 0.5·Brier and a random option permutation per example, on about 50% agentic decisions and
     50% generic replay.
   - Then teacher soft labels (Jev when reachable; the Gemini verbalised-distribution stand-in until then).
   - Then counterfactual MC targets in closed form, $p^\*\propto\pi_{\rm ref}\,e^{\hat Q/\beta}$.
   - Then post-hoc temperature per (type × option-count bucket).
3. **RL where it pays.**
   - Options are enumerable, so the bandit part of RL has exact, supervised solutions: R4 (KL-to-$p^\*$) and R6
     (exact PG).
   - Sampled RL is reserved for:
     - (a) verifiable decision rewards (R6b, decision-level GRPO);
     - (b) the only thing supervised targets cannot see: the *episode-level* consequences of committing vs.
       deferring vs. pruning inside the harness (R8, in-harness GRPO with a cost-aware, anti-cheapness reward).
4. **Success criterion.** Downstream harness success and cost at matched decision accuracy, not decision accuracy
   alone (RQ5).

## 1. Goals and hypotheses (decision modeling for agents)

The paper's claim is that *decision modeling* is its own layer of an agent: a bounded, typed, calibrated decision
made in ~0.1 s. Training asks what an open model of that layer must learn beyond "pick the gold label".

| ID | Hypothesis | Test (stage) | Primary metric |
|----|-----------|--------------|----------------|
| H1 | Agent decisions are learnable by a single-pass decider: SFT beats zero-shot readout by a wide margin | R1 vs R0 (T1) | held-out decision acc |
| H2 | Proper scoring + position debiasing give *usable* confidence: Brier + permutation lower ECE and flip-rate at equal accuracy | R2 vs R1 (T2) | ECE, flip-rate, AURC |
| H3 | Soft teacher distributions transfer calibration and uncertainty that one-hot gold lacks | R3 vs R2; soft vs hard ablation (T3) | NLL, ECE, teacher-agreement acc |
| H4 | *Consequence-aware* targets (MC Q̂) improve downstream success beyond decision accuracy | R4/R5/R6 vs R3 at matched acc (T4–T6) | JevGym success, $/success |
| H5 | In-harness RL learns *when to commit, defer, prune or stop* (the control policy the cascade threshold only approximates) | R8 vs R4 (T7) | success–cost Pareto front, defer rate, harmful automation |
| H6 | A 0.6–1.7B student keeps most of the gains at a fraction of the latency | R7 vs teacher (T8) | Δacc ≤ 2 pts, p50/p95 latency |

## 2. Decision model and model families

**Readout** (`decision_model.option_logits`). Let $c_1..c_K$ be the option-code tokens of the presented order.

$$z_k(s)=\text{logit}_\theta(c_k\mid \text{chat}(s,q,\text{options})),\qquad p_\theta(a_k\mid s)=\frac{e^{z_k/T}}{\sum_j e^{z_j/T}}$$

- Noul becomes `["yes -- …", "no -- …"]`; Score becomes `level k: …`. This matches exactly how `serve_jev.py`
  maps Jev wire questions (`grpo_online.question_to_example` mirrors it, and a test checks this).
- Prompt: `data.render` then `decision_model.chat_text`.
  - `chat_text` passes `enable_thinking=False` whenever the tokenizer's chat template knows that variable (Qwen3
    hybrid templates).
  - This is decided from the template, not the model path, so a tokenizer re-loaded from a checkpoint directory
    renders the same prompt as in training.
- The collator left-pads and keeps the prompt *tail* (`max_len`), so the question and options are never cut.

| Family | Sizes | Use | Notes |
|--------|-------|-----|-------|
| Qwen3 (dense, hybrid thinking) | 0.6B, 1.7B, **4B**, 8B | main line | codes A–Z/a–z are single tokens (checked at load: `code_token_ids` raises otherwise) |
| Any HF causal LM with single-token codes | — | ablation / replication | same code path; verify the chat template |
| Jev (`typesafe/jev-1.13`, hosted) | — | teacher (T3) and comparison row | never trained here; OpenRouter preset |
| Gemini 3.5 Flash-Lite, verbalised distribution | — | stand-in teacher until Jev is reachable | `llm_generative`, `reasoning_effort: minimal` |
| OpenJev (DiffusionGemma) | 26B-A4B | zero-shot comparison row | not trained (see §12 for diffusion deciders) |
| Encoders (ModernBERT-large cross-encoder) | ~0.4B | optional Noul gates (memory write, retrieve?) | §12 |

Default adapter: LoRA r=32, α=64 on all linear layers. Full fine-tuning is used for ≤1.7B students.

## 3. Data sources, labels and mixing

**Format** (`data.py`): one typed decision per row.

```
{"id": "<task>/<step>/<dp>", "source", "dp", "kind": choice|noul|score, "state", "question",
 "options": [...], "target": {option: prob}, "q_values": {option: Q̂} | null, "meta": {...}}
```

**Label sources** (they combine; every example keeps its provenance in `meta.label_source`):

| Label | Produced by | Signal | Cost |
|-------|-------------|--------|------|
| gold one-hot | `harvest.py` (JevGym oracle `gold`, When2Call-train, xLAM, ToolACE…) | what is correct | free |
| teacher soft $p_T$ | `teacher_label.py` (Jev / Gemini / any served checkpoint / local checkpoint) | how uncertain a strong decider is | Jev ≈ \$0.04 / 1k decisions; Flash-Lite ≈ \$0.13 / 1k (ROADMAP pilot figures) |
| MC $\hat Q(s,a)$ | `mc_rollouts.py` (replay prefix, force $a$, finish $M$ times) | which choice leads to success | $K\cdot M$ episodes per state |
| in-harness returns | `grpo_online.py` rollouts | commit / defer / prune / stop consequences, incl. LLM cost | $G$ episodes per task per iteration |

**Teacher labelling** (`teacher_label.py`):
- For each example, each teacher $t$ gives $p_t$ over the options.
- Pooling: $p_T=\sum_t w_t p_t$ (mean) or $p_T\propto\prod_t p_t^{w_t}$ (geo, product of experts), then tempered
  $p_T\propto p_T^{1/T}$.
- Agreement filter: keep the example if at least `min_agree` teachers have $\arg\max p_t=\arg\max p_T$.
- Gold filter: when the consensus disagrees with gold, `drop` | `gold` (keep one-hot) | `keep`.
- Target: $q=\alpha\,e_y+(1-\alpha)\,p_T$.
- Gold labels are only visible to *simulated* teachers (`Question.meta` is never on the wire, and it is left empty
  for real teachers; a test enforces this).
- SQLite cache: re-runs are free and the summary separates fresh from replayed cost.
- Per-teacher `Ledger`: calls, tokens, \$, latency, accuracy / NLL / ECE vs gold, in `<out>.summary.json`.
- Smoke-checked against the real Gemini stand-in: 3 decisions, 2 batched calls.

**Splits and contamination (non-negotiable)**:
- Split by *task* (`split_by_task`).
- Never train on BFCL or When2Call **test**.
- JevGym seeds:

  | Seeds | Use |
  |-------|-----|
  | 0–199 | E2 evaluation — never train, never tune |
  | 5000+ | GRPO dev / early stopping |
  | 10000+ | training data, MC states and GRPO training tasks |

- Harvest training trajectories with `jevbench run <exp> -o suite.seed0=10000 name=train_<exp>` (one `-o`,
  several pairs). `harvest.py --min-jevgym-seed 1000` (default) drops JevGym tasks with evaluation seeds and warns.
- `harvest.py` emits Choice decisions (tool, arguments, action) and Noul gates (tool-call guard, stop guard or
  trigger) whose oracle label is a bool. Arms must *consult* System-1 for decisions to be logged with options. The
  `shadow` arm is the cheapest way to log everything: its behaviour is LLM-only, and System-1 is asked in shadow.
- Harvest renders tool_select options `name -- description` exactly as `serve_jev` renders a wire Choice. The
  descriptions are rebuilt from the `experiment.yaml` saved next to the trajectories, or `--exp`.
  `mc_rollouts.py` does the same.
- Choices with more than 52 options (the calibrated v2 suite has 69 tools) keep the gold plus a seeded sample of
  51 others; they used to be dropped. Ids are unique per decision.
- When2Call **train** is the SFT file (`{tools, messages}`, no label column). It holds only *ask for info* and
  *cannot answer* rows (label: the assistant turn contains "?" → ask). xLAM rows add When2Call-format "call" items;
  without them, the replay teaches "never call".
- Files ending in `.jsonl.gz` are read and written transparently.
- 8-gram decontaminate external data against every eval set.

**Mixing** (per stage):

| Stage | Mix |
|-------|-----|
| T1/T2 SFT | ~50% agentic decisions (tool 40%, gates/stop/verify 35%, args 25%) + ~50% generic replay (When2Call-train, xLAM, ToolACE) |
| T3 | teacher-labelled agentic data + gold replay; disagreements fall back to gold (`ON_DISAGREE=gold`) |
| T4–T6 | MC data + 0.5× gold replay (`REPLAY_FACTOR`) against forgetting |
| T8 | teacher-labelled union of all training states (+ MC states) |

Agent-only data erodes general decision quality (Imajev: JevBench-hard 64.9→42.3; CLM top-1 69→56.2 without
replay), so replay is always on. `python -m jevharness.training.data mix --in a.jsonl:1 b.jsonl:0.5 --out m.jsonl`
does the mixing; `… data stats FILE` reports source / dp / kind / option-bucket counts and target softness.

## 4. Stages and objectives (T0–T8 ↔ R0–R8)

Notation:
- $s$ is the decision state (task, history, observation); $A(s)$ its options; $y$ the gold option.
- $\pi_\theta(a\mid s)$ is the readout at $T=1$.
- $\pi_{\rm ref}$ is the frozen reference: the `--init` adapter loaded a second time as `"ref"` (default when
  continuing), or the base model with LoRA disabled (`--ref base`).
- $\sigma$ is the random option permutation.

| Stage | Recipe | Init | Data | Objective (per decision) | Script |
|-------|--------|------|------|--------------------------|--------|
| T0 | R0 zero-shot | base | — | none: $p_\theta$ as is | `90 ADAPTER=` |
| T1 | R1 SFT-CE | base | gold | $-\log\pi_\theta(y\mid s)$, fixed order | `10 RECIPE=r1` |
| T2 | R2 SFT-CE+Brier+perm | base | gold | $-\sum_a q_a\log\pi_\theta(a\mid s_\sigma)+\lambda_B\sum_a(\pi_\theta(a\mid s_\sigma)-q_a)^2$ | `10` |
| T3 | R3 teacher-soft | base (or R2) | $q=\alpha e_y+(1-\alpha)p_T$ | as R2 | `20` |
| T4 | R4 KL-to-MC | R3 | MC + replay | $-\sum_a p^\*(a\mid s)\log\pi_\theta(a\mid s)$, $p^\*\propto\pi_{\rm ref}e^{\hat Q/\beta}$ | `30`, `40` |
| T5 | R5 DPO | R3 | MC pairs, gap ≥ 0.3 | $-\log\sigma\big(\beta[\Delta_\theta(a_w)-\Delta_\theta(a_l)]\big)$, $\Delta_\theta(a)=\log\pi_\theta(a)-\log\pi_{\rm ref}(a)$ | `50` |
| T6 | R6 exact PG | R3 | MC | $-\sum_a\pi_\theta(a\mid s)\hat Q(s,a)+\lambda_{KL}\,\mathrm{KL}(\pi_\theta\Vert\pi_{\rm ref})$ | `55` |
| T6 | R6b decision GRPO | R3 | gold / soft / Q̂ rewards | sampled-group PPO-clip (below) | `60` |
| T6c | R9 on-policy distillation (candidate, §17) | R3 | live JevGym states reached by the *student*; teacher $p_T$ at those states | $\mathrm{KL}(p_T\Vert\pi_\theta)$ (or reverse KL / JSD) over the option codes, fresh rollouts per iteration | `65` |
| T7 | R8 in-harness GRPO | R4 (best of T4–T6) | live JevGym (+τ²) episodes | episode-level PPO-clip on event log-probs (below) | `70` |
| T8 | R7 distil + calibrate + serve | small base | teacher readout | as R2 with $q=\alpha e_y+(1-\alpha)\,\mathrm{norm}(p_{\rm teacher}^{1/T})$; then temperature fit | `80`, `90` |

### R4: why the closed form is the RL solution

A single decision is a contextual bandit. The KL-regularised objective
$\max_\pi \mathbb E_{a\sim\pi}\hat Q(s,a)-\beta\,\mathrm{KL}(\pi\Vert\pi_{\rm ref})$ has the unique maximiser
$p^\*(a\mid s)\propto\pi_{\rm ref}(a\mid s)\exp(\hat Q(s,a)/\beta)$. Fitting $p^\*$ with soft CE is therefore
supervised RL with zero sampling variance.

- $\hat Q=(\text{wins}+1)/(M+2)$ (Laplace).
- By default `40_kl_to_mc.sh` recomputes $\pi_{\rm ref}$ from the R3 checkpoint (`teacher_label --mc-beta` with a
  `local` teacher), so $p^\*$ is the optimum relative to the policy being fine-tuned rather than the logged S1.
- MC rollouts use common random numbers across alternatives: rollout $r$ of every alternative shares an executor
  seed. $\hat Q$ differences thus reflect the action, not executor noise.
- `mc_rollouts.py` exits with an error if no logged state belongs to the `--exp` suite. Pass the
  `runs/<name>/experiment.yaml` that `jevbench run` saved; this is the default in `30_mc_rollouts.sh`.
- Alternatives that are not the taken tool get empty arguments. In strict JevGym, a wrong state-changing call
  scores near 0, which is the intended irreversibility signal.

### R6b: decision-level GRPO (`grpo_trl.py`)

For each prompt, sample $G$ options $a_1..a_G\sim\pi_{\theta_{\rm old}}^{(T)}$ and compute rewards
$r_i=\sum_k w_k r_k(a_i)$, where:

| Reward | Value |
|--------|-------|
| `correct` | $\mathbb 1[a=y]$ |
| `format` | 0 if the first token is a code (and, with `--strict-eos`, EOS follows), else −1 |
| `soft` | $q_a$ |
| `qvalue` | $\hat Q(s,a)$ |

$$A_i=\frac{r_i-\bar r}{\mathrm{std}(r)+\epsilon},\quad \rho_i=\frac{\pi_\theta(a_i)}{\pi_{\theta_{\rm old}}(a_i)},\quad
L=-\frac1G\sum_i\min\big(\rho_iA_i,\ \mathrm{clip}(\rho_i,1-\epsilon_{lo},1+\epsilon_{hi})A_i\big)+\beta\,\mathrm{KL}(\pi_\theta\Vert\pi_{\rm ref})$$

Two implementations:

| | `--impl trl` | `--impl torch` (fallback) |
|--|--------------|---------------------------|
| Sampling | TRL `GRPOTrainer` over the full vocabulary (1–2 tokens) | restricted option softmax, one prefill per prompt |
| KL | TRL's per-token estimator | exact over the options |
| Format failures | possible | none |
| Processes | multi-GPU | single process |

TRL notes (installed and verified: TRL 1.14.1, transformers 5.18, peft 0.21, accelerate 1.15):
- All `GRPOConfig` fields used exist (`num_generations`, `max_completion_length`, `beta`, `loss_type` (default
  `dapo`), `scale_rewards`, `epsilon`/`epsilon_high`, `num_iterations`, `reward_weights`).
- A `PeftModel` passed with `beta>0` gets a frozen `"ref"` copy of its adapter.
- `per_device_train_batch_size` counts *completions*.
- Prompts are pre-rendered strings (no second chat template).
- With a cold model, TRL samples non-code tokens, so every group has zero variance and no update happens. Start
  from an SFT adapter.

### R8: episode-level GRPO inside the harness (`grpo_online.py`)

The decider runs in-process as a `DecisionBackend` (`TorchDecider`); rollout prefills from concurrent episodes are
micro-batched. Every System-1 question is answered by *sampling*, and each record stores the log-probability of
the **event it caused in the harness** (behaviour policy at temperature $T$):

| Harness mode | Event $e$ | $\log P_\theta(e)$ |
|--------------|-----------|--------------------|
| replace / choice | option $a$ | $\log\pi(a)$ |
| cascade, commit | $a$ with $\pi_1(a)\ge\tau$ | $\log\pi(a)$ |
| cascade, defer | $D=\{a':\pi_1(a')<\tau\}$ | $\log\sum_{a'\in D}\pi(a')$ (no gradient if $D=A$) |
| assist top-$k$ | ordered subset $(s_1..s_k)$ (Gumbel-top-$k$) | $\sum_i\log\pi(s_i)-\log\big(1-\sum_{j<i}\pi(s_j)\big)$ (Plackett–Luce) |
| Noul gates (guard / trigger / stop / reflection / routing / memory) | sampled yes/no; the gate follows the sample | $\log\pi(\text{yes or no})$ |

Reward per episode, from the trajectory ledger:

$$R_{\rm lin}=\mathbb 1[\text{success}]-\lambda_{\$}\,\text{cost}-\lambda_{\rm calls}\,n_{\rm LLM},\qquad
R_{\rm gated}=\mathbb 1[\text{success}]\cdot\big(1-\lambda_{\rm gate}\min(1,c/c_{\rm scale})\big),$$

minus `irreversible_penalty` when strict JevGym reports an irreversible error.

Advantages over the $G$ rollouts of a task, plus optional GiGPO step groups over *anchor states* (identical
(dp, prompt) across the group), with $g_t=\gamma^{T-1-t}R$:

$$A^E=\frac{R-\overline{R}_G}{\mathrm{std}_G+\epsilon},\qquad A^S=\frac{g_t-\overline{g}_{\rm anchor}}{\mathrm{std}_{\rm anchor}+\epsilon},\qquad A=A^E+\omega A^S$$

$$L=\frac1{|D|}\sum_{d}\Big[\max\big(-A_d\rho_d,\ -A_d\,\mathrm{clip}(\rho_d,1-\epsilon,1+\epsilon_{hi})\big)+\beta\,\mathrm{KL}\big(\pi_\theta\Vert\pi_{\rm ref}\big)(s_d)-c_H\,H(s_d)\Big],\quad \rho_d=\frac{P_\theta(e_d)}{P_{\rm old}(e_d)}$$

- $\mu$ inner epochs over the buffer; exact KL over the option set at $T=1$.
- Aggregation per decision (default) or per episode.
- Records are skipped when:
  - their episode raised an exception;
  - they carry no signal at either level (zero reward spread in the task group *and* no anchor spread). This is
    DAPO-style dynamic sampling, and GiGPO still teaches "succeed in fewer steps" when every rollout succeeds.
- Several arms (WHERE × HOW cells) share one policy; arms are assigned round-robin to task groups.
- Outputs:
  - `metrics.jsonl`: reward, success, cost, calls, defer rate, `cascade_forced_defer_frac`, decision acc vs gold
    per dp (logged, never rewarded), KL, entropy, clip fraction, `updated_decisions`, periodic greedy
    `eval/<arm>/*`;
  - `rollouts.jsonl`;
  - `ckpt-XXXXX/` with optimizer state (`--resume`);
  - the final adapter, which `calibrate.py` / `serve_jev.py` load unchanged.

## 5. Rewards and anti-reward-hacking

| Hack | Where | Mitigation (implemented) |
|------|-------|--------------------------|
| **Cheap failure**: stop early, never call the LLM, collect the cost bonus | R8 linear reward | default `reward.kind: gated`. Cost only discounts *successful* episodes, so failing cheaply earns 0. A test checks this. Linear is kept as an ablation. |
| **Defer everything**: success via the LLM at full cost | R8 cascade arms | LLM calls enter the gated reward; report `defer_rate` next to success/cost; Pareto evaluation (§9) |
| **Premature stop / finish** | stop gates | success comes from `env.evaluate`, not the agent; stop-guard arm in training; `max_steps` bound |
| **Irreversible gambles** (strict JevGym) | R8, MC | `irreversible_penalty` (0.2) below a benign failure; strict mode in training and eval suites |
| **Over-confidence to clear τ** | cascade | exact KL anchor; confidence and ECE logged; calibration **refit after every policy change** (T8); τ* = q_L − ΔC/V recomputed |
| **Format gaming** (non-code tokens, no EOS) | R6b TRL | `format` reward −1; `--strict-eos`; the torch impl is immune |
| **Teacher bias / copying teacher errors** | R3, R7 | multi-teacher agreement, gold filter, α-mix with gold, gold-only fallback |
| **Simulator exploitation** | any | simulated executor / teachers are for pipeline tests only; training executor = real LLM (API or vLLM); evaluation on held-out seeds with the E2 executor |
| **Accuracy proxy** | R8 | decision accuracy vs oracle is logged but never part of the reward |
| **Entropy collapse** | R6b, R8 | clip-higher (ε_hi = 0.28), optional entropy bonus, KL β; monitor `entropy` / `clip_frac` |
| **Leakage of gold into a real teacher** | R3 | gold only in `Question.meta` for simulated teachers (never on the wire); test-enforced |

## 6. Hyper-parameters (defaults in the scripts; sweep marked ✱)

**SFT** (`10_sft.sh`, `20_teacher_soft.sh`, `80_distill_small.sh`):

| Size | Method | LR | LoRA r/α | Eff. batch | Epochs | Max len |
|------|--------|----|----------|-----------|--------|---------|
| 0.6B | full FT | 2e-5 | – | 64 | 2 | 4k |
| 1.7B | full / LoRA | 1e-5 / 1e-4 | 64/128 | 64 | 2 | 4k |
| 4B | LoRA all-linear | 1e-4 ✱{3e-5, 1e-4, 2e-4} | 32/64 | 64 | 2 | 4k (8k for τ² states) |
| 8B | LoRA | 1e-4 → 2e-5 | 32/64 | 64 | 1–2 | 4k |

- AdamW, weight decay 0, warmup 3%, cosine, grad-clip 1.0, bf16.
- Non-reentrant gradient checkpointing, `ddp_find_unused_parameters=False`.
- Brier weight 0.5 ✱{0, 0.25, 0.5, 1}.
- No label smoothing (it hurts selective prediction); use soft targets.
- Teacher α = 0.3 ✱{0, 0.3, 0.6}; distillation α = 0.2, T = 1 ✱{1, 2}.

**MC / KL / DPO / PG** (`30`–`55`):

| Knob | Default | Sweep |
|------|---------|-------|
| states / K_ALT / M | 2000 / 4 / 4 | M ✱{2, 4, 8} |
| β (MC target) | 0.2 | ✱{0.1, 0.2, 0.5} |
| R4 LR / epochs / replay | 3e-5 / 1 / 0.5× | |
| R5 β / LR / min Q-gap | 0.1 / 5e-6 / 0.3 | gap ✱{0.2, 0.3, 0.5} |
| R6 KL weight / LR | 0.02 / 1e-5 | KL ✱{0.005, 0.02, 0.1} |

**Decision-level GRPO** (`60_grpo_trl.sh`):

| Knob | Default | Sweep |
|------|---------|-------|
| G | 8 | ✱{4, 8, 16} |
| temperature | 1.0 | |
| β (KL) | 0.02 | ✱{0, 0.02, 0.05} |
| loss_type | dapo | |
| ε / ε_hi | 0.2 / 0.28 | |
| μ (num_iterations) | 1 | |
| LR | 1e-5 constant | |
| completions per device | 16 | |
| grad-accum | 4 | |
| rewards | `correct,format` (gold), `soft` (teacher), `qvalue` (MC) | |

**In-harness GRPO** (`configs/train/grpo_online_jevgym.yaml`):

| Knob | Default | Sweep |
|------|---------|-------|
| iterations × tasks/iter × G | 150 × 8 × 8 (9.6k episodes) | G ✱{4, 8} |
| inner epochs μ | 2 | |
| minibatch / micro-batch (decisions) | 64 / 8 | |
| ε / ε_hi | 0.2 / 0.28 | |
| β (exact KL to init) | 0.02 | ✱{0.005, 0.02, 0.05} |
| entropy coef | 0 | |
| GiGPO | on, ω = 1, γ = 0.95 | ✱{off, on} |
| reward | gated, λ_gate = 0.5, cost in LLM calls, scale 10, irreversible penalty 0.2 | λ ✱{0.25, 0.5, 1} |
| LR | 1e-5, warmup 5 | |
| workers | 16 concurrent episodes | |
| max_steps | 20 | |
| eval | every 10 iterations on 64 dev tasks (seeds 5000+) | |

## 7. Compute plan

These are planning estimates for bf16 on H100 80GB. Replace them with measured throughput after the 5% pilot
(`MAX_STEPS`, `LIMIT`, `ITERATIONS` knobs).

| Stage | 1×H100 | 8×H100 (DDP via `accelerate_8gpu.yaml`) | External cost |
|-------|--------|------------------------------------------|---------------|
| T1/T2 SFT 4B LoRA, ~20k decisions × 2 epochs | 5–10 h | ~1 h | — |
| T3 teacher labels, 20k decisions | minutes (API-bound, 8 workers) | same | Jev ≈ \$1, Flash-Lite ≈ \$3 per teacher pass (per-1k pilot prices × 20k) |
| T3 SFT | 10–20 h | 1.5–3 h | — |
| T4 MC rollouts, 2000 × 4 × 4 = 32k episodes | executor-bound; shard with `SHARDS` | same | ≈ \$350 with Gemini 3.5 Flash (\$0.011/episode pilot); ~5× less with Flash-Lite; GPU-only with a vLLM executor |
| T4–T6 fine-tunes | 3–6 h each | < 1 h each | — |
| T6b decision GRPO (1 epoch, G = 8) | 4–8 h (trl) / 1–2 h (torch) | 1 h (trl) | — |
| T7 in-harness GRPO, 9.6k episodes | 2–6 h wall (latency-bound at 16 workers); the decider needs < 20 GB | run 3 seeds / ablations in parallel, one per GPU | Flash-Lite executor ≈ \$15–30; vLLM executor: GPU only |
| T8 distil 0.6B full FT | 2–4 h | < 1 h | — |
| T8 calibrate + serve + E1/E2 eval | < 1 h + E2 run | same | as one E2 arm |

- Measured while building `data/train_v0`: an MC rollout episode costs $0.0027 with the Flash-Lite executor on
  the v1 suite (`mc_rollouts` now writes `<out>.summary.json` with the Ledger cost). A shadow-arm training episode
  with a Flash System-2 costs $0.081 on the calibrated v2 suite and $0.022 on v1.
- Full fine-tuning at ≥ 4B: switch the accelerate config to DeepSpeed ZeRO-2/3; the scripts are unchanged.
- The in-harness GRPO process is single-GPU by design: the decider never generates, and the bottleneck is episode
  latency. Scale it with `workers` and parallel seeds.

## 8. Ablations (each isolates one design choice; 3 seeds where cheap)

| Axis | Arms | Where |
|------|------|-------|
| Loss terms | CE vs soft CE vs +Brier (λ_B ∈ {0.25, 0.5, 1}) | T2 |
| Position debiasing | permute vs fixed order; eval flip-rate | T2 |
| Soft vs hard | one-hot gold vs teacher-soft vs α-mix; single vs multi-teacher; mean vs geo pooling; agreement filter on/off | T3 |
| Consequence targets | R4 vs R5 vs R6 vs R6b-qvalue at matched decision acc; β; M; π_ref = init vs logged S1 | T4–T6 |
| Free generation vs restricted readout | R6b `trl` vs `torch` | T6 |
| Reward shaping | gated vs linear; λ ∈ {0.25, 0.5, 1}; irreversible penalty on/off | T7 |
| Credit assignment | GRPO vs GiGPO (ω ∈ {0.5, 1}); decision vs episode aggregation | T7 |
| Group size / KL | G ∈ {4, 8, 16}; β ∈ {0, 0.02, 0.05}; ref = init vs base | T6b, T7 |
| Arms trained | single-cell (cascade only) vs multi-cell shared policy | T7 |
| Student size | 0.6B vs 1.7B vs 4B; distil from R4 vs R8 vs Jev directly | T8 |
| Calibration | none vs global T vs per-bucket T; refit after RL vs not | T8 |

## 9. Evaluation protocol

1. **Decision level** (held-out tasks: `dev.jsonl`, E1 BFCL v4 + When2Call test via `jevbench decide`):
   - accuracy, NLL, Brier, ECE (equal-width and equal-mass), AUROC (conf → correct), AURC / risk–coverage,
     coverage at 1% / 5% risk;
   - option-order flip rate (permute and compare);
   - latency p50/p95 at concurrency 1 / 16 / 64 through `serve_jev`.
2. **Downstream harness** (`jevbench run`, E2 JevGym seeds 0–199 strict, near + random distractors; then τ² / ALFWorld
   for the best checkpoint):
   - success, \$ cost, LLM calls, latency, cost-of-pass, APGR/CPT, hypervolume;
   - harmful automation: accepted wrong System-1 decision → failure.
   - Every trained decider is served behind the Jev API and plugged into the *same* arms as Jev / Gemini-lite
     (`90_calibrate_serve_eval.sh` derives the configs).
3. **Calibration**: reliability diagrams; empirical best τ vs τ* = q_L − ΔC/V (Prop. 1) per checkpoint.
   Temperatures are fitted on dev tasks and never on E1/E2 test.
4. **Statistics**:
   - 3 training seeds for T2/T4/T7 main rows;
   - bootstrap 95% CIs over tasks;
   - paired tests vs the previous stage on the same tasks;
   - simulated-executor runs are never reported as results.

## 10. Go / no-go per stage

Decision-level metrics come from `python -m jevharness.training.eval_decisions` (same dev files for every
checkpoint); downstream metrics come from the E2 runs of `90_calibrate_serve_eval.sh`. The operational version of this
table, with the v0 thresholds filled in, is [`TRAINING_AGENT_BRIEF.md`](TRAINING_AGENT_BRIEF.md) §6.

| Stage | Go if … (on dev tasks, then confirmed on E1/E2) | Else |
|-------|-----------------------------------------------|------|
| T1 | R1 ≥ R0 + 5 pts decision acc | check the prompt/template (e.g. `enable_thinking`), label noise, LR sweep |
| T2 | ECE ↓ and flip-rate ↓ vs R1 at ≤ 1 pt acc loss | keep R1 + temperature scaling; report H2 as negative |
| T3 | ≥ 90% of teacher acc, NLL ↓ vs R2 | raise α, tighten agreement, more replay |
| T4–T6 | JevGym success ↑ (or \$ / success ↓) vs R3 at matched decision acc (CI excludes 0) | report H4 negative; skip T7 and distil R3 |
| T6b | ≥ R3 decision acc with ECE not worse | drop from the main table (ablation only) |
| T7 | Pareto front dominates R4–R6 on ≥ 2 arms and `defer_rate` does not collapse to 0 or 1 | report as exploratory |
| T8 | student within 2 pts of the teacher, p50 latency ≤ ½ | use 1.7B |

Stop-loss: if T4 does not help by 10/6 (ROADMAP §7), skip T7 and spend the GPU time on T8 + E1/E2 evaluation.

## 11. Exact commands

Setup on the GPU host (details, hardware layouts, gates and the failure playbook:
[`TRAINING_AGENT_BRIEF.md`](TRAINING_AGENT_BRIEF.md)):

```bash
export HF_TOKEN=...                      # or `hf auth login`
bash scripts/train/setup_gpu.sh          # venv + pinned stack (configs/train/constraints-gpu.txt) + checks
bash scripts/train/smoke_gpu.sh          # every stage at tiny scale on Qwen3-0.6B, no API key needed
cp .env.example .env   # keys; never commit
```

Gate metrics for any checkpoint (R0 = no `--adapter`):
`python -m jevharness.training.eval_decisions --model Qwen/Qwen3-4B --adapter checkpoints/r2 --dev data/train_v0/dev.jsonl --out checkpoints/r2/eval_dev.json`.

Every script accepts `DRY_RUN=1` (print the commands) and documents its variables in its header. Common variables:
`MODEL`, `GPUS`, `DATA` (`data/decisions`), `CKPT` (`checkpoints`), `OUT`.

```bash
# T0 data (training seeds only!). One -o followed by several KEY=VALUE pairs (repeated -o flags accumulate too).
# Arms consult System-1 so every decision is logged with its options,
# state and oracle gold (a simulated 80%-accurate S1 produces realistic mistake/recovery states; labels are oracle
# gold; the executor is the config's real LLM).
jevbench run configs/experiments/jevgym_tool_select.yaml -o name=train_jevgym suite.seed0=10000 suite.n=1000 \
  's1={type: simulated, accuracy: 0.8}' 's1_backends={}' 'sweeps=[]' s1_cache=null \
  'arms=[{name: gen_replace, tool_select: {mode: replace}, tool_params: {mode: replace}, stop: {mode: trigger, tau: 0.5}},
         {name: gen_guard, tool_select: {mode: guard, tau: 0.5}, stop: {mode: guard, tau: 0.5}}]'
RUNS="runs/train_jevgym" XLAM=data/raw/xlam_function_calling_60k.json bash scripts/train/00_harvest.sh

# T1 / T2 SFT (8xH100)
RECIPE=r1 GPUS=8 bash scripts/train/10_sft.sh
GPUS=8 bash scripts/train/10_sft.sh                                                  # -> checkpoints/r2

# T3 teacher soft labels + SFT (stand-in now; Jev when openrouter.ai is allow-listed)
LIMIT=200 DO_TRAIN=0 bash scripts/train/20_teacher_soft.sh                           # budget probe
TEACHERS='jev=jev-openrouter;lite=gemini-lite' MIN_AGREE=2 GPUS=8 bash scripts/train/20_teacher_soft.sh   # -> r3

# T4-T6 counterfactual targets (MC reads the run's saved experiment.yaml, so suite/seeds match the trajectories)
jevbench run configs/experiments/jevgym_tool_select.yaml -o name=train_llm_only suite.seed0=10000 suite.n=500 \
  'arms=[{name: llm_only}]' 'sweeps=[]'
RUNS=runs/train_llm_only/llm_only.jsonl SHARDS=4 WORKERS=8 bash scripts/train/30_mc_rollouts.sh
GPUS=8 bash scripts/train/40_kl_to_mc.sh                                             # -> r4
GPUS=8 bash scripts/train/50_dpo.sh                                                  # -> r5
GPUS=8 bash scripts/train/55_pg_exact.sh                                             # -> r6
GPUS=8 bash scripts/train/60_grpo_trl.sh                                             # -> r6b (TRL)
IMPL=torch bash scripts/train/60_grpo_trl.sh                                         # fallback / ablation

# T6c on-policy distillation (§17): Jev teacher, Flash-Lite executor, stop at $5 of teacher spend; 1 GPU
BUDGET_USD=5 INIT=checkpoints/r3 EXECUTOR=gemini-lite bash scripts/train/65_opd.sh   # -> r9-opd
TEACHER='intern={"type":"intern_decision","model":"internlm/Intern-Decision-4B"}' OUT=checkpoints/r9-opd-intern \
  EXECUTOR=gemini-lite bash scripts/train/65_opd.sh                                  # second teacher family
LOSS=reverse_kl OUT=checkpoints/r9-opd-rkl EXECUTOR=gemini-lite bash scripts/train/65_opd.sh   # loss ablation

# T7 in-harness GRPO (3 seeds on 3 GPUs; EXECUTOR=vllm for a self-hosted executor)
GPUS=8 SEEDS="0 1 2" INIT=checkpoints/r4 bash scripts/train/70_grpo_online.sh        # -> r8-s{0,1,2}
GPUS=8 SEEDS=0 SET='grpo.gigpo.enabled=false' OUT=checkpoints/r8-nogigpo bash scripts/train/70_grpo_online.sh

# T8 distil + calibrate + serve + evaluate (E1 decide + E2 JevGym arms against the served model)
TEACHER_ADAPTER=checkpoints/r4 GPUS=8 bash scripts/train/80_distill_small.sh         # -> r7-0.6b
ADAPTER=checkpoints/r4 bash scripts/train/90_calibrate_serve_eval.sh
MODEL=Qwen/Qwen3-0.6B ADAPTER=checkpoints/r7-0.6b PORT=8082 bash scripts/train/90_calibrate_serve_eval.sh
ADAPTER= NAME=qwen3-4b-zeroshot DO_CAL=0 bash scripts/train/90_calibrate_serve_eval.sh   # R0 row
```

On 1×H100, drop `GPUS=8`: gradient accumulation is recomputed to keep the effective batch at 64.

Smoke on CPU with any tiny model: `MODEL=<dir> MAX_STEPS=3 LORA_R=4 BATCH=4 EFF_BATCH=4 …`. That is how the chain
was validated; `tests/test_training_rl.py` runs the unit and integration parts in about 15 s.

A served checkpoint is a System-1 backend anywhere:
`{"type":"jev","name":"ours-r4","base_url":"http://localhost:8081","model":"ours-r4","api_key":""}`.

## 12. Optional / exploratory

- **Encoder gates**: ModernBERT-large cross-encoder for Noul gates (memory write, retrieve?), about 5–30 ms.
- **Tool retriever**: Qwen3-Embedding-0.6B InfoNCE on ToolRet for catalogs of more than 255 tools.
- **Diffusion deciders**: LoRA on DiffusionGemma via NeMo Automodel (answer-slot loss); high risk.
- **Calibration-aware RL**: an RLCR-style Brier term in the R8 reward for gate questions.
- **τ² in-harness RL**: `grpo_online` is suite-agnostic for `load_suite` suites. τ² runs through its own
  orchestrator, so R8 on τ² needs a rollout adapter (not implemented).

## 13. Known limitations

- MC alternatives that are not the taken tool get empty arguments. Q̂ therefore measures "calling this tool now,
  with the executor unable to argue it". Fine for tool *selection*; argument-level MC needs an LLM argument
  proposer.
- R8 trains through the events the harness exposes. Knockout rounds for more than 52 options (deterministic
  top-2), and executor-side choices, receive no gradient.
- Cascade arms need a warm start (§4).
- Noul gates are *sampled* during R8 training but thresholded at evaluation (p_yes ≥ τ). Report both, and refit τ
  after training.
- TRL's KL is a sampled per-token estimator. The torch implementation's KL is exact.
- `data/train_v0` is a starter set:
  - Gemini 3.5 Flash is at ceiling on JevGym, so its states contain few recoveries.
  - Its teacher labels come from the Gemini stand-in, not Jev.
  - Its When2Call replay has two of four classes.
  - See `data/train_v0/MANIFEST.json` notes.
  - Its states differ from the harness's: §18.1 measures the gap and §18.2 builds `data/train_v1` to close it.
- No GPU numbers yet. Every compute figure above is an estimate.

## 14. Data sources (HF IDs; download on the GPU machine, the cloud sandbox cannot reach HF)

| Dataset | Content | License |
|---------|---------|---------|
| `nvidia/When2Call` (GitHub copy used by our loader) | call / ask / cannot-answer / direct | CC-BY-4.0 |
| `Salesforce/xlam-function-calling-60k` | verified single/multi calls | CC-BY-4.0 (gated); verify |
| `Team-ACE/ToolACE` | function-calling dialogues | Apache-2.0 |
| `NousResearch/hermes-function-calling-v1` | FC + JSON mode | Apache-2.0 |
| `Agent-Ark/Toucan-1.5M` | MCP trajectories | Apache-2.0 |
| `nvidia/Nemotron-SFT-Agentic-v2` | tool-use trajectories | verify |
| `nvidia/Open-SWE-Traces`, `SWE-bench/SWE-smith-trajectories` | SWE trajectories (resolved flag) | CC-BY-4.0 / MIT |
| `neulab/agent-data-collection` (ADP) | unified agent trajectories | per source |
| our JevGym / τ² / SWE runs | decision logs with gold / teacher / MC labels | ours |

**Never train on BFCL or When2Call *test*** (our evaluation sets).

## 15. References (arXiv IDs)

PriDe 2309.03882 · temperature scaling 1706.04599 · KnowNo 2307.01928 · RAPS 2009.14193 ·
DPO 2305.18290 · GRPO 2402.03300 · DAPO 2503.14476 · RLOO 2402.14740 · GiGPO 2505.10978 ·
ArCHer 2402.19446 · RLCR 2507.16806 · Memory-R1 2508.19828 · AgentPRM 2502.10325 · Math-Shepherd
2312.08935 · IPR 2406.11176 · When2Call (NAACL 2025) · xLAM/APIGen 2406.18518 · ToolACE 2409.00920 ·
ToolRet 2503.01763 · ModernBERT 2412.13663 · LLaDA 2502.09992 · Hinton KD 1503.02531 · xRouter 2510.08439.

Software: TRL `GRPOTrainer` (verified against TRL 1.14.1), PEFT LoRA (0.21), Accelerate (1.15).

Community Jev-likes: github.com/allebee/jevk5 · github.com/mohit67890/imajev · github.com/Contrastive-LM/CLM ·
github.com/NandhaKishorM/laya · github.com/Heman10x-NGU/Verdict-open-jev · JevBench github.com/fstandhartinger/jevbench.

## 16. Incremental decision serving (`serve_jev` session cache)

`serve_jev.py` renders the training prompt byte for byte (`data.render` + `chat_text`, §2) but no longer runs one
full prefill per question. Two levels of reuse (`docs/DECISION_MODEL_LANDSCAPE.md` §3), both exact: a token's
hidden states never depend on later tokens, so a prompt scored as *cached prefix + suffix* gives the same
`option_logits` as the stateless path (fp32 CPU regression tests assert ≤ 1e-4; `tests/test_serving.py`).

1. **Inside one request.** The k questions of a `POST /v1/systemone` request are tokenised, their longest common
   token prefix (the whole `STATE:` block, usually more) is prefilled once with the KV / recurrent cache, and each
   question's suffix is scored on a copy of that cache, batched (right-padded; each row reads its own last token).
2. **Across requests (episode cache).** An LRU of (token prefix, cache snapshot) pairs, bounded by cached tokens and
   entries. A new request reuses the stored snapshot that is the longest *exact* prefix of its common prefix and
   prefills only the rest. Dense / sliding-window KV caches can be **cropped** (`DynamicCache.crop`), so a longer
   stored snapshot is cropped to the match: the state at t+1 shares everything before the `CURRENT OBSERVATION`
   block of the state at t, and that is what gets reused. Hybrid caches with recurrent layers (Qwen3.5 / Qwen3-Next
   Gated DeltaNet) cannot be rolled back (detected once at start-up by trying `crop`); for them the prefill is
   split where the new state diverged from its nearest stored snapshot and where that divergence marker
   (e.g. `\n\nCURRENT OBSERVATION:`) reappears later in the state, and snapshots are taken at those points, so the
   next step still gets an exact-prefix hit (one step of lag on the first step of an episode).

Flags (`python -m jevharness.training.serve_jev --help`): `--session-cache / --no-session-cache` (default on),
`--cache-tokens N` (budget in cached tokens, default 65536; memory ≈ N × `bytes_per_token` from `describe()`, e.g.
≈ 115 KB/token for Qwen3-0.6B in bf16, so size it to the GPU), `--cache-entries`, `--prefill-chunk` (long prefills
run in chunks), `--batch-suffixes / --no-batch-suffixes` and `--suffix-batch-tokens` (the prefix cache is repeated
once per question for the batched suffix pass; above this many prefix tokens × questions the questions are scored
one by one). `JEV_SERVE_NO_CACHE=1` forces the stateless path (the old server; identical-output regression and A/B
latency runs). In-process scorers (`teacher_label`, `calibrate`,
`eval_decisions`) construct `Engine(...)` with the session cache off and keep batching distinct states.

Counters (wire `usage` per request; totals in `GET /v1/models` → `backend.counters` and `/health`):

| key | meaning |
|---|---|
| `stateless_tokens` | tokens the stateless path would have prefilled (sum of full prompt lengths) |
| `prefilled_tokens` | tokens actually run through the model (prefix delta + suffixes) |
| `cache_hit_tokens`, `cache_hit` | tokens taken from a stored snapshot for this request's common prefix |
| `cache_hits`, `snapshots_stored`, `evictions` | engine totals; `session.cached_tokens / entries` = current LRU size |

`describe()` also reports `croppable`, `cache_class`, `layer_types` and `bytes_per_token`. Benchmark:
`scripts/bench_serving.py` (stateless vs prefix vs session, 1 / 3 questions, batch 1 / 8, synthetic JevGym-like
states of 1k–64k tokens or `--trajectories <run>.jsonl` to replay logged per-step states; `--smoke` runs the tiny
model on CPU in under a minute; `--max-cache-gb` caps the session cache by memory and `--max-batch-tokens` sizes the
stateless path's padded batch, so 64k-token cells degrade to evictions / smaller batches instead of OOM); it writes
`bench_serving.{json,md}` to `--out`. Memory note: a 64k-token snapshot of Qwen3-0.6B is ≈ 7 GB, so one L4 holds one
such episode in the session cache, not eight.

**Measured on an L4 (GPU round 2, Qwen3-0.6B, `results/gpu_round2/NOTES.md` §4):** the shared-prefill path pays off
as designed (3 questions per state: 2.3-2.7× fewer prefilled tokens, 2.7× lower latency at 4k tokens), but the
session cache, while cutting prefilled tokens 1.9-6.3×, was *slower* than stateless at ≤ 16k tokens with one
question per request (96 vs 30 ms at 1k): that path ran two forward passes per request (prefix delta, then the
suffix on a copy) plus an intermediate snapshot at the predicted divergence point. Since then: (1) on a croppable
cache (dense / sliding-window KV) a one-question request runs the prefix delta and the suffix read-out in **one**
forward pass and crops the cache back to the prefix for the snapshot (two passes remain for several questions and
for recurrent / linear-attention caches); (2) croppable caches store no intermediate snapshots (the next step crops
the full-prefix entry); (3) `Engine(profile=True)` / `JEV_SERVE_PROFILE=1` and `bench_serving.py --profile` report
wall time per phase (encode / lookup / prefill / store / suffix, device-synchronised) so the next GPU bench shows
where the remaining time goes. Tests: `tests/test_serving.py::test_single_question_session_requests_use_one_forward_pass_and_match_stateless`.

## 17. On-policy distillation (OPD, stage 65 / recipe R9, `opd.py`)

**Status.** Implemented and CPU-tested (`tests/test_opd.py`, smoke stage `65-opd`); no GPU run and no numbers yet.
It is a *candidate* addition to the training RQ (hypothesis below), **not part of the pre-registration** (frozen
2026-10-03): adopting it means an append-only amendment, and until then its rows are exploratory.

### 17.1 Why on-policy for a decider inside an agent

Stages 20 / 40 / 80 fit the student to targets on *fixed* states: states harvested from Gemini-driven or shadow
episodes (20, 80) or the logged states of the MC rollouts (40). That is off-policy: the student is trained on the
state distribution of whichever policy wrote the logs, but once it is served it lives in the states its *own*
decisions produce -- in particular the recovery states after its own wrong tool or argument, which a near-ceiling
teacher-driven log (`data/train_v0`: every Flash episode succeeded) almost never contains. This is the compounding-
error argument of DAgger / exposure bias, and it is sharper for a decider than for a generator: one wrong
`tool_select` changes every later observation of the episode.

GRPO (60 / 70) is on-policy but pays for it twice: it needs a reward (and the anti-cheapness design of §5), and it
gets one scalar per episode (70) or per decision (60), so the gradient is high-variance and silent about the
options that were not sampled. OPD keeps the on-policy half (the student's own state distribution) and replaces
the reward by the teacher's **full distribution** over the options at every reached state:

- dense: every option of every visited question gets a target, not only the sampled one;
- calibrated: the teacher's uncertainty mass is the signal the cascade threshold relies on (H3), and it
  transfers with the distribution, not with an argmax;
- no reward design: nothing to hack (§5 rows "cheap failure", "defer everything", "format gaming" do not apply);
- cheap per state: one teacher request (Jev ≈ $0.09 per 1k requests, pilot r2 ledger) against $K\times M$ executor
  episodes per MC state (stage 30).

Options are enumerable, so the divergences below are **exact** over the option set (no sampled tokens, no
importance weights), as in the rest of this document.

### 17.2 The loop (`python -m jevharness.training.opd`, `scripts/train/65_opd.sh`)

Per iteration $i = 1..N$:

1. **Rollout.** The student is System-1 in `JevAgent` (default arm: `replace` on `tool_select` and `tool_params`;
   any arm list works, gates included) on `tasks_per_iter` JevGym tasks drawn from a pool of training seeds
   ($\ge 10000$; `--seeds < 10000` is refused), sampling its option at temperature $T$ (`--rollout-temperature`;
   0 = greedy = the served policy's state distribution). The in-process decider is `grpo_online.TorchDecider`
   (same prompts, knockout, micro-batching), subclassed so every decision keeps its state, question and scored
   option pool (`opd.Visit`).
2. **Teacher.** Every visited decision is re-asked, at the same state, with the same typed question restricted
   to the pool the student scored, to the teacher: any `make_backend` spec (`jev-openrouter`,
   `{"type": "intern_decision", ...}`, the Gemini verbalised stand-in, a served checkpoint) or
   `{"type": "local", "model": ..., "adapter": ...}` = a checkpoint read out in-process by `serve_jev.Engine` on
   the identical rendered prompt. Visits sharing a state go out in one request (`--questions-per-request`) when the
   backend batches. Gold labels reach *simulated* teachers only (`Question.meta`), never a real backend
   (test-enforced). Every call is on one `Ledger`; `--budget-usd` stops the run when the spend passes it (the labels
   already paid for are still trained on); `--max-questions` caps the labelled decisions per iteration.
3. **Update.** The labelled visits become rows of the `data.py` format (`labelled/iter-XXXXX.jsonl`, reusable by
   `10_sft.sh` / `80_distill_small.sh` off-policy) whose `render` is byte for byte the rollout prompt. One pass
   (`--epochs-per-iter`) of the divergence below over the fresh rows plus `replay_frac` × as many rows sampled from
   the previous iteration (against forgetting between iterations), random option permutation per row, LoRA (or
   full FT with `--lora-r 0`), AdamW, grad-clip 1, bf16 on CUDA.

Checkpoints (`ckpt-XXXXX/` with optimizer state; `--resume auto`), `metrics.jsonl`, `summary.json` and the final
adapter in `--out` (loads unchanged in `serve_jev` / `calibrate` / `eval_decisions`).

### 17.3 Objective

With $p_T$ the teacher's distribution over the presented options and $\pi_\theta$ the student's restricted softmax:

$$\mathcal L_{\rm fwd}=\mathrm{KL}(p_T\Vert\pi_\theta)=\sum_a p_T(a)\log\frac{p_T(a)}{\pi_\theta(a)}
=\underbrace{-\sum_a p_T(a)\log\pi_\theta(a)}_{\text{soft CE (stages 20/40/80)}}-H(p_T)$$

$$\mathcal L_{\rm rev}=\mathrm{KL}(\pi_\theta\Vert p_T),\qquad
\mathcal L_{\rm JSD}=\tfrac12\mathrm{KL}(p_T\Vert m)+\tfrac12\mathrm{KL}(\pi_\theta\Vert m),\ m=\tfrac12(p_T+\pi_\theta)$$

- `forward_kl` (default) has exactly the soft-CE gradient $\pi_\theta-p_T$; it is logged as a divergence so that
  0 means "student = teacher". Mean-seeking: it keeps the teacher's uncertainty mass, which is what calibration
  transfer (H3) wants.
- `reverse_kl` is mode-seeking (the objective on-policy distillation of *generators* minimises on sampled
  tokens; here it is exact). Expect sharper students: check ECE, not only accuracy.
- `jsd` is bounded by $\log 2$ and symmetric; a robustness option when the teacher is sometimes wrong.
- `--gold-alpha` mixes the JevGym oracle one-hot into $p_T$ (ablation only; 0 by default: OPD uses no gold).
- `--teacher-temperature` tempers $p_T$ as in stages 20 / 80.

### 17.4 Relation to stages 40 / 60 / 80

| Stage | States | Target | Signal per state | Executor episodes per state | Teacher calls per state |
|-------|--------|--------|------------------|----------------------------|-------------------------|
| 40 KL-to-MC | logged (off-policy) | $p^\*\propto\pi_{\rm ref}e^{\hat Q/\beta}$ | consequence-aware distribution | $K\cdot M$ | 1 (local $\pi_{\rm ref}$) |
| 60 decision GRPO | logged prompts, sampled options | reward of $G$ samples | $G$ scalars | 0 | 0 (gold) or 1 (`soft`) |
| 70 in-harness GRPO | **on-policy** | episode reward | 1 scalar per episode (+ GiGPO anchors) | $G$ per task | 0 |
| 80 distil | logged (off-policy) | $p_T$ | dense distribution | 0 | 1 |
| **65 OPD** | **on-policy** (student's rollouts) | $p_T$ at the reached states | dense distribution | $1/\text{steps}$ per state | 1 |

OPD learns *what the teacher would decide where the student actually goes*; it cannot learn what the teacher does
not know (consequences, 40) or what only the episode reveals (when to defer, 70). The natural chain is therefore
R3 → R9 (OPD) → optionally R8 (in-harness GRPO on top), and the comparison the hypothesis asks for is R3 → R4 → R8
vs R3 → R9 at equal teacher budget.

### 17.5 Hyper-parameters (`configs/train/opd_jevgym.yaml`; sweep marked ✱)

| Knob | Default | Sweep |
|------|---------|-------|
| iterations × tasks/iter × rollouts/task | 40 × 16 × 1 (640 episodes) | tasks/iter ✱{8, 16, 32} at fixed budget |
| rollout temperature | 1.0 (sampled = the student's policy) | ✱{0 (greedy), 1} |
| loss | forward_kl | ✱{reverse_kl, jsd} |
| replay_frac | 0.5 | ✱{0, 0.5, 1} |
| epochs/iter, minibatch, micro-batch | 1, 32, 8 | |
| LR, warmup, LoRA r/α | 3e-5, 5 steps, 32/64 (full FT ≤ 1.7B: `LORA_R=0`, LR 1e-5) | |
| gold_alpha / teacher_T | 0 / 1 | ✱{0, 0.3} / ✱{1, 2} |
| max_questions, questions/request | 0 (all) / 16 | |
| budget_usd | 5 | |
| eval | every 5 iterations, 32 greedy dev episodes (seeds 5000+), no teacher calls | |
| arms | `opd_replace` (tool_select + tool_params replace) | + cascade / gate arms as in `70` |

### 17.6 Cost model

Teacher questions per iteration $\approx$ tasks × rollouts × E[steps] × questions per step (replace arm: one
`tool_select` plus 1–2 `tool_params` per step). With $\approx 8$ steps and 2.5 questions per step: 16 × 8 × 2.5 ≈ 320
questions, ≈ 130 requests after per-state batching. Prices (`configs/prices/models.yaml`, verify):

| Teacher | $ per 1k requests | per iteration | 40 iterations |
|---------|-------------------|---------------|---------------|
| Jev via OpenRouter | ≈ 0.09 measured (pilot r2: $0.156 / 1,817 requests of ~3.5k tokens, 3 questions each) | ≈ $0.01–0.03 | ≈ $0.5–1 |
| Gemini Flash-Lite verbalised stand-in | ≈ 0.13 per 1k decisions (ROADMAP pilot) | ≈ $0.04 | ≈ $2 |
| Intern-Decision-4B (in-process) | GPU time only | – | – |
| local checkpoint (`serve_jev.Engine`) | GPU time only | – | – |

The executor dominates: Flash-Lite ≈ $0.01 per episode (HANDOFF §9 estimate; $0.022 measured for a Flash
shadow episode on v1) → ≈ $6 for 640 episodes; a vLLM executor is GPU time only. `metrics.jsonl` carries
`teacher_cost_iter / teacher_cost_total` (Ledger, cached replays separated) and the per-episode `cost_usd`;
`--budget-usd` is the hard stop.

### 17.7 GPU plan (single L4 24 GB or L40S 48 GB)

- Student Qwen3-4B LoRA bf16: ≈ 9 GB weights; rollout prefills micro-batched (`decider_batch 32`); the update at
  micro-batch 8 × 4k tokens fits an L40S; on an L4 use `MICRO_BATCH=4 EXTRA=--gradient-checkpointing`.
- Rollouts are executor-latency-bound: `workers 16` concurrent episodes with the Flash-Lite API, as in `70`.
- A local teacher shares the GPU: Qwen3-8B adapter in bf16 ≈ 16 GB + 4B student → L40S; on an L4 pair a 1.7B
  student with a 4B local teacher, or use Jev / Intern-Decision-4B (≈ 8 GB) as the teacher.
- Planning estimate per iteration: rollout 2–4 min (16 tasks, 16 workers) + teacher < 1 min + update ≈ 1 min →
  40 iterations ≈ 2–4 h wall, single process. Seeds / ablations run one per GPU (`CUDA_VISIBLE_DEVICES`).
- Smoke first: `EXECUTOR=sim ITERATIONS=1 TASKS_PER_ITER=2` (stage `65-opd` of `smoke_gpu.sh` does this with the
  R4 checkpoint as an in-process teacher), then `ITERATIONS=2 BUDGET_USD=0.5` live to read the measured $ and
  minutes per iteration before the full run.

### 17.8 Metrics (`metrics.jsonl`, one row per iteration)

`success`, `cost_usd`, `llm_calls`, `steps`, `irreversible_rate` of the rollouts; `decisions` (states visited),
`asked`, `labelled`, `teacher_skipped`; `agree_greedy` / `agree_sampled` (student argmax / sample vs teacher
argmax, overall and per decision point), `teacher_acc` and `dec_acc_greedy` vs the JevGym oracle (logged, never
optimised), `kl_teacher_student` = KL($p_T\Vert\pi$) on the fresh states *before* the update, `loss` (the chosen
divergence during the update), `fresh_n` / `replay_n` / `train_n`, `teacher_calls_iter`, `teacher_cost_iter`,
`teacher_cost_total`, `teacher_input_tokens_iter`, the three wall times, and `eval/<arm>/{success, cost_usd,
llm_calls, dec_acc, ...}` from greedy dev rollouts. Health: `agree_greedy` ↑ and `kl_teacher_student` ↓ across
iterations; `success` in the rollouts ↑ (sampled policy) and `eval/*/success` ↑ (greedy policy).

### 17.9 Hypothesis for the training RQ (candidate; NOT pre-registered)

> **H7 (candidate).** At equal teacher budget (teacher questions, or $; the MC rollouts of R4 count as teacher
> spend), SFT → OPD (R3 → R9) matches or exceeds SFT → KL-to-MC → in-harness GRPO (R3 → R4 → R8) on (a) decision
> accuracy vs the oracle on held-out dev tasks and (b) E2 JevGym success and $ / success in the `replace` and
> `cascade` arms (same E2 subset, bootstrap CIs over tasks); the gap in favour of OPD grows with the student's own
> error rate at R3 (the distribution-shift term), and `forward_kl` keeps ECE at or below R3's while `reverse_kl`
> raises accuracy at the cost of ECE.

Test: R9 vs R4 and R8 from the same R3, same seeds, teacher = Jev (second family: Intern-Decision-4B); matched
budget by `BUDGET_USD`; `eval_decisions` on the same dev files; E2 through `90_calibrate_serve_eval.sh`.
Go: dev acc ≥ R3 + 1 pt with ECE not worse, `agree_greedy` rising; else report as negative. This is a proposed
addition to the pre-registration's training section (it would be amendment A6, append-only); until it is accepted,
OPD rows are exploratory and never headline numbers.

### 17.10 Known limitations

- Teacher errors are copied (no agreement / gold filter by default); `--gold-alpha` is the only guard. Multi-
  teacher pooling as in `teacher_label.py` is not wired in.
- Choices with more than 52 options: the teacher is asked over the pool the student's knockout kept, not over all
  options (the student can only learn over what it scored).
- States depend on the executor: record which System-2 made the rollouts (the config is saved in `opd_config.yaml`).
- Replay reaches one iteration back; longer buffers are not implemented.
- `eval/*` greedy rollouts cost executor $ (no teacher $); set `eval_every 0` to skip them.

## 18. Training data v1 and GPU round 4: closing the static-to-in-harness gap

**Status (2026-10-02).** Data v1 is built (`data/train_v1`, 12 MB gzipped and committable like v0; the trajectories,
gold intermediates and Jev cache stay in the gitignored `data/cache/`, and `SKIP_EPISODES=1 bash
scripts/train/make_data_v1.sh` rebuilds it from them at $0; manifest
[`results/training/train_v1/MANIFEST.md`](../results/training/train_v1/MANIFEST.md)), the round-4 recipe is
`scripts/train/12_sft_v1.sh`, and the builder is `src/jevharness/training/data_v1.py` (tests:
`tests/test_training_data_v1.py`). No checkpoint has been trained on v1 yet. Every number below is a decision-level
dev number or a pilot number on development / training seeds. None is a headline number. The trained rows are
exploratory (H4 / RQ5 training section) and never replace the registered second family (A5.1).

### 18.1 Gap diagnosis (round-2 evidence)

Intern-Decision-4B passes the A5.1 gate statically, but the harness exposes a gap (`results/gpu_round2/NOTES.md`,
`gate_Intern-Decision-4B.json`, `e2a_intern_dev/report.md`, `e1_dev_intern/summary.json`):

| decision point | static gate (210, `data/train_v0/dev.jsonl`) | in-harness shadow (Flash states, n = 10 dev tasks) | own states (replace) | Jev: static / shadow / replace |
|---|---|---|---|---|
| tool_select | 0.806 | 0.760 (75), AUROC 0.74 | 0.407 (54); cascade AUROC 0.45–0.69 | 0.968 / 0.933 / 0.971 |
| tool_params | 0.814 | 0.872 (94) | 0.232 (69) under `ts_params_replace` | 0.954 / 0.968 / – |
| stop | 0.984 | 0.933 (75) | stop guard 0.833 (12, Flash), 0.542 (24, Flash-Lite) | 1.000 / 0.987 / – |
| E1 (static, BFCL / When2Call) | tool 0.953, args 0.746, W2C 0.694 | – | – | 0.965 / 0.951 / 0.778 |

The failure is sequential, not lexical. Static tool selection matches Jev on BFCL (0.95 vs 0.97). In the harness,
Intern skips the lookup (`cancel_order` straight after `find_user`), acts where the gold is `ask_user`, and fills an
argument for a call that should not happen. Its confidence carries little information on its own states: cascade
AUROC is 0.45–0.69, so a threshold cannot rescue it. As a guard on mostly correct proposals it matches Jev (0.96
vs 0.97), but those proposals hold too few wrong ones to measure detection (review round 6).

**What `data/train_v0` lacks** (measured with `data_v1.bucket` / `stats`; v0 numbers from `data/train_v0`, in-harness
numbers from the E2a dev slice `runs/dev/e2a_dev` on dev seeds 0–9):

| property | v0 train / gate dev | in-harness (E2a dev slice) | v1 train (new rows) / v1 dev |
|---|---|---|---|
| task setting of the tool_select rows | 47 % / 52 % from v1 families (cancel / refund / stock / ship / total, 30 mixed distractors, no noise) | 100 % v2 clarify + multi, 60 near distractors, 20 noise lines (= the frozen evaluation setting) | 100 % evaluation setting (training seeds 30000–30599) |
| median S1 state length | 563 / 528 chars | ~1,900 chars | 1,898 / 1,733 chars |
| states after a wrong action (`after_error`) | 0 (every Flash episode succeeded) | the states of every replace / cascade arm | 35 % / 25 % |
| `ask_user` gold among tool_select rows | 7.3 % / 6.5 % | 12 % | 10.7 % / 10.2 % |
| tool-guard questions (wrong proposals) | none | guard arms (E2a / E2b) | 1,042 yes + 526 no / 125 + 91 |
| stop-guard 'no' (premature or wrong finish) | 0 | stop-guard arms | 147 / 15 |
| teacher | Gemini Flash-Lite stand-in (89 % agreement with gold) | – | Jev (`typesafe/jev-1.13`, 94.5 % agreement with gold) |
| size | 1,225 rows, 77 tasks | – | 15,694 rows, 587 tasks |

### 18.2 Data v1 (`scripts/train/make_data_v1.sh`)

1. **Episodes.** The real harness (`bench.runner.run_experiment` + `JevAgent`) runs the evaluation *setting* on JevGym
   training seeds 30000–30599, with Gemini 3.5 Flash-Lite as the executor (T = 0). Three behaviour arms (1,000
   episodes, $4.65):
   - `lite_shadow` (250 tasks): Flash-Lite acts alone and System-1 is asked in shadow. These are natural states,
     including Flash-Lite's own mistakes (207 / 250 successful).
   - `nearmiss_replace` (600 tasks): System-1 is `NearMissBackend` in replace on tool_select + tool_params. It
     answers with the oracle gold, except that at rate ε (tool 0.2, argument 0.15, gate 0.1) it makes a *near
     miss* drawn from the round-2 error modes:
     - `skip`: a later step, or acting instead of asking (weight 0.4);
     - `near`: the JevGym near-duplicate that shadows the gold tool (0.35);
     - `over_ask` (0.1) and `random` (0.15);
     - for arguments, a confusable value of the same lexical shape (another order id or user id), OTHER, or a
       random value.

     This is DAgger-style state collection. It reaches the recovery states that a decider in replace or cascade
     visits and that no teacher-driven log contains (432 / 600 episodes with off-plan steps). The policy is a
     simulation component: it decides which states get visited and never supplies a label.
   - `lite_guard` (150 tasks): Flash-Lite proposes, and System-1 (the oracle with ε flips) guards every tool call
     and the stop. This yields the guard Noul questions on real proposals and the states after a rejection.
2. **Labels.**
   - Every row keeps the JevGym oracle gold.
   - Jev answers every train row at the same state, on the exact training prompt (`teacher_label.label_file`,
     batched per state). The training target is q = 0.3·gold + 0.7·p_Jev.
   - The 864 rows where Jev's argmax is not the gold keep the gold one-hot.
   - Dev stays gold. Jev's own dev metrics, asked one question per request, are the in-harness reference (18.5).
3. **Wrong-proposal guard rows.**
   - Source: an executed call outside the oracle's acceptable set AT(s) (`gold_aux.acceptable_tools`; 668 calls).
   - Each one becomes a guard question asked exactly as the harness asks it (`q_tool_guard` with the executed
     arguments, or `q_stop_guard` for a premature `finish`), labelled 'no'.
   - Without these rows the guard data hold 32 'no' against 1,167 'yes', and a guard learns "always yes".
4. **Knockout-final variants.**
   - The problem: `serve_jev` answers the 69-tool Choice of the evaluation suite through a 52-chunk knockout whose
     final round is the gold plus about three strong survivors. Harvested rows (gold + 51 random others) never show
     that distribution.
   - The fix: half of the train rows (and all of `dev_knockout`) get a variant with the gold plus its hardest
     negatives present (near-duplicates, the skipped-to step, `ask_user` / `finish`). That adds 2,133 rows.
5. **Mixing and splits.**
   - Duplicates across arms are removed (1,496, mostly step-0 states).
   - Balancing caps stop-trigger 'no' at 4× 'yes' and OTHER arguments at 25 %, but always keeps the recovery rows.
   - The split is by task: dev = 90 tasks (15 %) with the natural distribution.
   - All 1,225 v0 train rows are added to train.
   - The When2Call-train replay is v0's, minus the 15 rows that share an 8-gram with a BFCL / When2Call test question.
     The overlaps are stock phrases such as the "quick brown fox" pangram, not duplicated items.
6. **Hold-out rules** (`data_v1.check_seed_plan`, `verify_exclusions`, `assert_clean`; test-enforced). All four
   pass on the built files:
   - every JevGym seed is ≥ 10000 and outside 200000–200499, 100000–100499 and 5000–5999;
   - none of the 13 tasks of the 210-decision gate set is in v1;
   - no E1 dev-prefix id (`configs/eval/e1_dev_prefix_ids.json`) is in v1, and no BFCL item is in v1;
   - no external row shares an 8-gram with any of the 6,687 BFCL (six categories) + When2Call test questions.
7. **Cost (Ledger):** episodes $4.65 + Jev teacher $0.49 (train, 5,175 requests) + $0.14 (dev, 2,377 requests) =
   **$5.28**.

| file (`data/train_v1/`) | rows | content |
|---|---|---|
| `train.jsonl.gz` | 15,694 (587 tasks) | tool_select 8,194 (incl. 1,568 tool-guard Nouls), tool_params 5,512, stop 1,988; Jev-soft 14,830, gold 864 |
| `dev.jsonl.gz` | 2,377 (90 tasks) | gold; 606 after an off-plan step |
| `dev_recovery.jsonl.gz` | 606 | the dev rows after an off-plan step (the replace / cascade regime) |
| `dev_knockout.jsonl.gz` | 656 | knockout-final variants of the dev tool_select rows |
| `replay_when2call.jsonl.gz` | 2,685 | When2Call train, decontaminated |

**Jev on v1 dev** (gold targets, `dev_teacher.summary.json`):

| subset | acc | ECE |
|---|---|---|
| overall | 0.958 | 0.013 |
| tool_select, incl. guard Nouls | 0.916 | – |
| tool_select Choice | 0.895 | – |
| tool-guard 'no' | 0.989 | – |
| tool_params | 0.979 | – |
| stop | 0.987 | – |
| stop-guard 'no' | 0.67 (n = 15) | – |
| states after an error | 0.944 | – |

### 18.3 Recipe for GPU round 4 (`scripts/train/12_sft_v1.sh`, unchanged on an L4 or an L40S)

```bash
HW=l4 bash scripts/train/12_sft_v1.sh 2>&1 | tee runs/train_logs/v1_r3.log           # 1x L4 24 GB
HW=l40s bash scripts/train/12_sft_v1.sh 2>&1 | tee runs/train_logs/v1_r3.log         # 1x L40S 48 GB
STAGES="e2dev" bash scripts/train/12_sft_v1.sh                                        # opt-in, after a GO
```

| stage | what | L4 24 GB | L40S 48 GB |
|---|---|---|---|
| `sft` | R3-v1: Qwen3-4B LoRA r 32 / α 64, soft CE + 0.5 Brier + permutation, LR 1e-4 cosine, eff. batch 64, `MAX_LEN` 4096, train + 0.45× replay (~16.9k rows) | `BATCH=2`, 1 epoch | `BATCH=4`, 2 epochs |
| `eval` | `eval_decisions` on the gate set + v0 When2Call dev (flip rate) and on v1 dev / recovery / knockout; also on `BASELINES` (round 3's `checkpoints/v0/r2`) | eval batch 8 | eval batch 16 |
| `intern` | Intern-Decision-4B on v1 dev + recovery (`scripts/gate_s1_dev.py`, once) | ~15 min | ~8 min |
| `gonogo` | `python -m jevharness.training.data_v1 gonogo` → `gonogo_r3v1.json` (18.4) | – | – |
| `opd` | R9 from R3-v1: Jev teacher, Flash-Lite executor, evaluation setting on training seeds 40000–40511, greedy dev rollouts on seeds 1000–1031, `BUDGET_USD=2` teacher, 16 tasks / iteration | 10 iterations, micro-batch 4 + gradient checkpointing, 8 workers | 20 iterations, micro-batch 8, 16 workers |
| `opd_eval` | eval + gonogo of the OPD adapter against R3-v1 | – | – |
| `e2dev` (opt-in) | serve the adapter, fit temperatures on v1 dev, run `ours_shadow` / `ours_ts_replace` / `ours_ts_params_replace` with Flash System-2 on dev seeds 0–9: paired with round 2's Intern rows; no E1 (model selection never sees BFCL / When2Call test) | ~$1 | ~$1 |

Knobs: `TRAIN_FACTOR` (seeded subsample if time is short), `INIT=checkpoints/v0/r2 LR=5e-5` (continue round 3's R2
instead of starting fresh), `OPD_ARMS`, `OPD_BUDGET`, `STAGES`, `DRY_RUN=1`. The epoch / iteration counts assume the
planning throughput; replace them with round 3's measured R2 SFT throughput and OPD minutes per iteration (18.6).

### 18.4 Go / no-go: when is the trained decider a better System-1 than Intern-Decision-4B?

`gonogo` reads the `eval_decisions` outputs and the references and writes every check with its value and threshold:

| check | rule | threshold today |
|---|---|---|
| G1 minimum (A5.1 bar) | gate-set acc ≥ 0.8 × Jev's gate acc | ≥ 0.777 |
| G2 static parity | gate-set acc ≥ Intern-4B's gate acc − 1 pt | ≥ 0.852 |
| G3 in-harness states | v1-dev tool_select acc ≥ Intern-4B's v1-dev tool_select + 5 pts **and** v1-dev overall ≥ Intern-4B's + 2 pts (same rows; stage `intern`) | from round 4; proxy if missing: Intern's round-2 shadow tool_select 0.760 → ≥ 0.81 |
| G4 vs teacher | v1-dev acc ≥ 0.8 × Jev's v1-dev acc on each of tool_select / tool_params / stop | ≥ 0.733 / 0.784 / 0.789 |
| G5 calibration | v1-dev ECE (raw) ≤ max(0.10, Intern's v1-dev ECE) | ≤ 0.10 |

- **GO = G1 ∧ G2 ∧ G3 ∧ G4 ∧ G5.** The checkpoint is then the better open System-1 *on the states the harness
  visits*. It is confirmed on the E2 dev slice before any row reports it: on the same seeds 0–9 with Flash
  System-2, `ours_shadow` tool_select s1_acc ≥ Intern's 0.760, and `ours_ts_replace` success ≥ Intern's 1/10 with
  s1_acc ≥ 0.407.
- **NO-GO:** Intern-Decision-4B stays the open System-1, and the trained row is reported as is.
- For the OPD continuation, the same gonogo runs with R3-v1 as baseline. OPD is kept when v1-dev acc ≥ R3-v1 + 1 pt,
  `dev_recovery` acc does not drop, and ECE is not worse (§17.9's go rule).

The static bar alone is not enough: Intern passed it at 0.862, yet its tool_select accuracy fell from 0.81 (gate)
to 0.41 on its own states. G3 and the `dev_recovery` / E2-dev readouts are the criteria that test the gap §18.1 found.

**Round-3 evidence (results/gpu_round3/NOTES.md §4–5).** R2, trained on train_v0, scores 0.995 on the gate dev set
(tool_select 1.00) but 0.45 tool_select accuracy on the states it reaches in OPD rollouts, against 0.97 for Jev. The
gate dev set shares train_v0's generator and task families, so for any checkpoint trained on v0 or v1 data G1 / G2 are
necessary but carry no information about in-harness competence: they cannot be failed by fitting the training
distribution. For such checkpoints the decision rests on G3, G4 and G5 (v1-dev in-harness and recovery states) and the
E2 dev-slice confirmation; G1 / G2 are reported, not used to argue superiority over Intern-Decision-4B.

### 18.5 Caveats

- **Behaviour policy.** The near-miss policy is scripted (oracle + ε near misses). It approximates a weak decider's
  state distribution, not any particular one. OPD (stage `opd`) is the student-specific, on-policy complement.
- **Executor.** All v1 states come from Flash-Lite, while the Flash primary arms reach somewhat different states.
  v0's 1,225 Flash rows are kept in train for that reason.
- **Wrong-proposal labels.** They assume AT(s) contains every tool whose call can be progress (META_GOLD_SPEC §3.5).
  Args-only errors on an acceptable tool are not turned into 'no' rows.
- **Jev is not gold.** Its tool_select agreement on v1 dev is 0.895, and on premature-finish proposals 0.67
  (n = 15). The α = 0.3 gold mix and the gold fallback on disagreement bound the damage, so the teacher's mistakes
  enter only as soft mass.
- **Dev is a dev set.** v1 dev is training-side data on training seeds: it may select checkpoints and fit
  temperatures, and it is never an evaluation number. E1 / E2 evaluation stays on the registered items and seeds.

### 18.6 What round 4 needs from round 3

- R2 SFT: wall time, examples/s and peak memory on the L4 (`r2_sft.log`, `nvidia_smi`). This sets `EPOCHS` /
  `TRAIN_FACTOR` (v1 states are ~3.4× longer than v0's).
- R2 `eval_decisions` on the gate set (`r2_eval_dev.json`): the v0-only baseline that G2 / G3 improve on. Round 4
  also scores R2 on v1 dev (`BASELINES`).
- OPD pilot (`opd_metrics.jsonl`, `opd_summary.json`): minutes and teacher $ per iteration, peak memory at
  micro-batch 4 with gradient checkpointing. These set `OPD_ITERS` / `OPD_BUDGET`; an OOM means `OPD_MICRO=2`.
- Check the pilot's seeds. It ran with `configs/train/opd_jevgym.yaml`, whose `eval_seed0` is 5000. If the pilot
  reached its greedy evaluation (iteration 0 and the last iteration), those rollouts used seeds 5000–5031, inside
  the calibration range. They were not the calibration tasks: the setting differs (30 mixed distractors, all
  families vs v2 near d60), so the task ids differ. `opd.py` now warns about it, and round 4 passes
  `--eval-seed0 1000`. The config default should move to 1000 as well; `configs/train/` is outside this change.
