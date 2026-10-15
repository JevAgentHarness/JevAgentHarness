# Small open decision models (≤ 2B parameters), October 2026: survey for an exploratory row E5

Written 2026-10-06 by a research session (read-only on the repo; web research plus CPU-only checks on tokenizers,
chat templates and tiny random models; no weights downloaded, no GPU, no LLM API call). It updates
[`docs/DECISION_MODEL_LANDSCAPE.md`](../../../docs/DECISION_MODEL_LANDSCAPE.md) (2026-10-01) for one question: *can a
~1B model replace JevAgentRL-4B (Qwen3-4B + LoRA SFT + GRPO) as the decision model, and which ones should we try?*

Third-party facts come from model cards, configs, papers and blogs (sources at the end). Unless a row says
"measured" or "verified here", treat a number as the vendor's. **Latency columns are estimates** (method in §1.3),
not measurements.

## 0. Bottom line

- **Recommended for E5 (§5):** `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-0.6B`. They are the 4B's own family, with the same
  tokenizer and chat template, so the prompts are byte-identical and give a clean 0.6B / 1.7B / 4B size curve. The
  third model is `openbmb/MiniCPM5-1B`, a non-Qwen ~1B model: Apache-2.0, plain `LlamaForCausalLM`, a croppable KV
  cache that is 1/6 of the 4B's per token, and already served through our `serve_jev` in GPU round 3. Backup for the
  third slot: `ibm-granite/granite-4.0-1b`, which has the best published BFCL v3 of the general ≤ 2B non-Qwen models
  (54.8) but 40 layers and 1.6B parameters.
- **Ouro-1.4B is not an efficiency candidate.** It loops its 24 layers 4 times, so one forward pass costs about
  **1.4× the 4B's compute** and needs about **5× its KV bytes per token** during prefill. It needs
  `trust_remote_code`, and its card pins `transformers < 4.56`; our stack is 5.18, which has no `ouro` model type
  (checked in `.venv`). Its strength is reasoning per *parameter*, not per *FLOP*. A decision model pays for FLOPs.
- **Every one of the 20 tokenizers checked supports our readout.** All 52 option codes are single tokens, and the
  code right after the generation prompt does not merge with the preceding text. Four things need care:
  date-stamped chat templates (Llama 3.2, SmolLM3), hybrid or sliding-window caches that cannot be cropped (Qwen3.5,
  LFM2, Granite-H, Falcon-H1, Gemma 3), remote-code models (Ouro, MobileLLM-R1) and gated repos (Gemma, Llama,
  MobileLLM). Details are in §2 and §6.
- **Prior evidence in this repo.** Off-the-shelf decision-tuned small models do not transfer to our harness states.
  StartLux-Decision-0.8B / 2B reach 48.3 % / 68.4 % on the common-state corpus C and 17.5 % / 6.5 % live success,
  against 99.2 % and 100 % for JevAgentRL-4B (`results/explore/startlux_live/summary.md`). On the same L40S host and
  server, though, the 0.8B answered a C question in 0.25× the 4B's median time and the 2B in 0.43×. What E5 asks is
  whether in-generator SFT closes the accuracy gap at ≤ 2B.

## 1. Method

### 1.1 What was verified here (CPU only, `preflight/`)

1. **Readout preflight** (`preflight/check_readout.py` → `preflight/readout_check.json`). The script downloads only
   the tokenizer, template and config files (`preflight/fetch_tok.py`; ungated `unsloth/` mirrors stand in for the
   gated Gemma and Llama repos). It then runs the repo's own `decision_model.chat_text` and `code_token_ids` with
   transformers 5.18.0 (the box's version) and checks four things: (a) all 52 codes are single tokens; (b) the code
   token directly after the rendered prompt has the same id as the bare code; (c) how `enable_thinking` is handled
   and whether the template inserts a date; (d) the mean prompt length over 400 data-v1 dev rows (100 per decision
   type), relative to Qwen3-4B.
2. **Serving-path probe** (`preflight/probe_cache.py` → `preflight/cache_probe.json`). For each architecture, a tiny
   random model is built from the candidate's config (64-dim, 2–8 layers that keep every layer type). Two checks
   follow: `serve_jev`'s croppability probe, and whether the prefix-cache read-out (`option_logits_cached`, the
   served path) equals the stateless left-padded read-out (`option_logits`, the trained path).
   **Result: the read-outs are equal for every architecture (max |Δ logit| ≤ 4e-7 in fp32).** Croppability splits
   by architecture (§2).
3. **Offline numbers from committed files.** The median C-replay latency per question for the StartLux ladder,
   JevAgentRL-4B, Jev and Intern-Decision (`results/stage2/s23/answers_*.jsonl`, field `latency_ms`). The C
   accuracies recomputed from the same files reproduce the published rows exactly (rl 99.2, jev 91.5, intern 75.2,
   base 50.1, sl0.8b 48.3, sl2b 68.4, sl4b 89.1, sl9b 91.8).

### 1.2 Parameter counts

"Non-emb." means the parameters of the decoder blocks, computed from each `config.json`. That is what a prefill
spends its FLOPs on: the LM head runs only at the read-out positions (`logits_to_keep`), so the vocabulary does not
count. For Qwen3 the safetensors index lists more parameters (0.75 / 2.03 B for 0.6B / 1.7B) because the tied LM
head is stored as its own tensor. `docs/DECISION_MODEL_LANDSCAPE.md` quotes those stored sizes.

### 1.3 How the latency estimate is made (an estimate, not a measurement)

**Compute ratio vs Qwen3-4B** = (non-embedding params × recurrent loops) / 3.63 B × (mean prompt tokens / Qwen3-4B's,
measured in §2). At concurrency 1 a small model does not reach this ratio. Per-layer kernel launches, the LM head at
the read-out rows, HTTP and the knockout rounds of the 69-option tool choice set a floor. The repo shows the floor
on our own workload: on the same L40S host and the vendor's server, StartLux-0.8B / 2B / 4B / 9B answer a C question
in **86 / 152 / 349 / 593 ms** (median), i.e. **0.25× / 0.43× / 1× / 1.70×**. So the c = 1 estimate is about
max(compute ratio, ~0.25), raised for very deep stacks (≥ 40 layers). At c = 4 the 4B saturates one A10G: the final
harness's System-1 time was 76 % queueing (`results/harness_final/summary.md`). A model with a lower service time
therefore gains more than its compute ratio at c = 4. For reference, JevAgentRL-4B on one A10G with
`serve_jev --fast` takes 370 ms per tool-choice request and 455 ms of System-1 time per step (p50 374 ms) at c = 1,
and 1,623 ms per step at c = 4 (`results/harness_final/replay_bench.md`).

## 2. Readout and serving preflight (verified here)

All rows: 52 single-token codes ✓, code after the prompt = bare code ✓, system text kept ✓ (Gemma folds it into the
first user turn). "Croppable" = `serve_jev` can roll a cached prefix back, which enables `--fast`'s fused
multi-question pass and the reuse of a longer cached entry. Non-croppable models fall back to exact-prefix reuse, and
their read-outs stay exact.

| model (tokenizer checked) | `enable_thinking` in template → `chat_text` | prompt end | date in template | prompt tokens vs Qwen3-4B | cache classes (tiny probe) | croppable |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B, -1.7B, -0.6B | yes → `False` | `<think>\n\n</think>\n\n` | no | 1.00 | DynamicLayer | **yes** |
| Qwen/Qwen3.5-0.8B, -2B | yes → `False` | `<think>\n\n</think>\n\n` | no | 1.02 | LinearAttention + Dynamic | no |
| openbmb/MiniCPM5-1B | yes → `False` | `<think>\n\n</think>\n\n` | no | **0.85** | DynamicLayer | **yes** |
| ByteDance/Ouro-1.4B (-Thinking) | no (Thinking: yes) | ChatML `assistant\n` | no | 1.07 | not native (remote code) | – |
| unsloth/gemma-3-1b-it, functiongemma-270m-it | no | `<start_of_turn>model\n` | no | 1.04 | SlidingWindow + Dynamic | no (window 512) |
| unsloth/Llama-3.2-1B-Instruct | no | Llama-3 header (`end_header_id`) | **yes** ("Today Date") | 0.85 | DynamicLayer | yes |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | no | ChatML | no | 1.07 | DynamicLayer | yes |
| LiquidAI/LFM2-1.2B, LFM2.5-1.2B-Instruct | no | ChatML | no | 0.88 | LinearAttention (conv) + Dynamic | no |
| ibm-granite/granite-4.0-1b | no | Granite role tags (`start_of_role` … `end_of_role`) | no | **0.83** | DynamicLayer | **yes** |
| ibm-granite/granite-4.0-h-1b | no | same | no | 0.83 | LinearAttention (Mamba-2) + Dynamic | no |
| tiiuae/Falcon-H1-0.5B / 1.5B-Deep-Instruct | no | ChatML | no | 1.20 | LinearAttention+FullAttention (parallel) | no |
| Salesforce/xLAM-2-1b-fc-r, MadeAgents/Hammer2.1-1.5b | no | ChatML | no | 0.99–1.00 | DynamicLayer | yes |
| facebook/MobileLLM-R1-950M | – | – | – | – | gated: tokenizer not downloadable without an HF token | – |

Notes. The Ouro tokenizer loads, but its `config.json` `auto_map` makes transformers ask interactively whether to run
remote code; a non-interactive box would hang or fail at that prompt. SmolLM3-3B (out of scope at 3B) also stamps
"Today Date" through `strftime_now` (template source). A date-stamped template breaks `chat_text`'s rule that
training and serving render byte-identical prompts: the training date ends up inside the training prompts.

## 3. Candidates

### 3.1 Identity, licence, architecture, context

| model (HF id) | params total / non-emb. | licence (commercial? research?) | architecture; loading | layers | context |
|---|---|---|---|---|---|
| `Qwen/Qwen3-4B` (our base, reference) | 4.02 / 3.63 B | Apache-2.0 (yes / yes) | dense GQA 32:8; native `Qwen3ForCausalLM` | 36 | 32,768 |
| `Qwen/Qwen3-1.7B` | 1.72 / 1.41 B | Apache-2.0 (yes / yes) | dense GQA 16:8; native | 28 | 32,768 |
| `Qwen/Qwen3-0.6B` | 0.60 / 0.44 B | Apache-2.0 (yes / yes) | dense GQA 16:8; native | 28 | 32,768 |
| `Qwen/Qwen3.5-0.8B` (Mar 2026) | 0.8 B (text: hidden 1024, vocab 248k) | Apache-2.0 (yes / yes) | hybrid: 18 Gated DeltaNet + 6 gated attention; multimodal checkpoint (`Qwen3_5ForConditionalGeneration`), loaded text-only by our `load_causal_lm` | 24 | 262,144 |
| `Qwen/Qwen3.5-2B` (Mar 2026) | 2 B (hidden 2048) | Apache-2.0 (yes / yes) | as 0.8B | 24 | 262,144 |
| `ByteDance/Ouro-1.4B` (Oct 2025) | 1.43 / 1.23 B, ×4 loops ≈ 4.9 B effective | Apache-2.0 (yes / yes) | **looped LM**: 24 shared layers × `total_ut_steps` 4, MHA 16 heads, SmolLM2 vocab 49,152; **`trust_remote_code`** (`OuroForCausalLM` via `auto_map`); card: **transformers < 4.56**; not in transformers 5.18; vLLM supported it Oct 2025 – Jul 2026, then removed it | 24 (×4) | 4K trained, 64K ext. |
| `ByteDance/Ouro-1.4B-Thinking` | same | Apache-2.0 | SFT on reasoning data (~8.3M examples) | 24 (×4) | as base |
| `google/gemma-3-1b-it` | ~1.0 / 0.70 B (262k vocab) | Gemma Terms of Use (commercial allowed under the prohibited-use policy; gated) | dense, MQA (1 KV head), 5 local (512) : 1 global; native `Gemma3ForCausalLM` | 26 | 32K |
| `google/functiongemma-270m-it` (Dec 2025) | 0.27 / ~0.10 B | Gemma terms (gated) | Gemma 3 270M tuned for function calling, meant as a base for further tuning | 18 | 32K |
| `google/gemma-3n-E2B-it` | ~5 B stored / "2B effective" | Gemma terms | MatFormer, per-layer embeddings, KV sharing; multimodal | – | 32K |
| `google/gemma-4-E2B` (Apr 2026) | 5.1 B stored / 2.3 B effective | Apache-2.0 | per-layer embeddings, 512 sliding + global, shared KV in the last layers; multimodal | 35 | 128K |
| `meta-llama/Llama-3.2-1B-Instruct` | 1.24 / 0.97 B | Llama 3.2 Community Licence (commercial yes below 700M MAU, attribution; gated) | dense GQA 32:8, head_dim 64; native | **16** | 128K |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.71 / 1.61 B | Apache-2.0 | dense **MHA** 32 heads (KV 1.33× the 4B per token); native Llama | 24 | 8K |
| `HuggingFaceTB/SmolLM3-3B` (ref.) | 3.08 B (out of scope) | Apache-2.0 | GQA, NoPE 3:1 | 36 | 64K (128K YaRN) |
| `LiquidAI/LFM2-1.2B` / 700M / 350M | 1.17 / 0.74 / 0.35 B | **LFM Open License v1.0** (free commercial use only below $10M annual revenue; research yes) | hybrid: 10 short-conv + 6 GQA blocks; native `Lfm2ForCausalLM` (≥ 4.55) | 16 | 32,768 |
| `LiquidAI/LFM2.5-1.2B-Instruct` (Jan 2026) | 1.17 B | LFM Open License v1.0 | as LFM2, more pre-training and RL | 16 | 32,768 |
| `ibm-granite/granite-4.0-1b` (Oct 2025) | 1.63 / 1.43 B | Apache-2.0 (yes / yes) | dense GQA 16:4, μP multipliers; native `GraniteMoeHybridForCausalLM` (attention-only config) | **40** | 128K |
| `ibm-granite/granite-4.0-h-1b` | ~1.5 B | Apache-2.0 | hybrid: 36 Mamba-2 + 4 attention, NoPE | 40 | 128K |
| `ibm-granite/granite-4.0-350m`, `-h-350m` | ~0.35 B | Apache-2.0 | dense / hybrid | – | – |
| `facebook/MobileLLM-R1-950M` | 0.95 B | **FAIR Noncommercial Research** (no commercial use); gated | dense GQA 24:6; registers `Llama4ForCausalLM`; a reasoning model (math, code, science), not general chat | 22 | 32K |
| `tiiuae/Falcon-H1-1.5B-Deep-Instruct` / `-1.5B` / `-0.5B` | ~1.5 / 1.5 / 0.5 B | Falcon LLM Licence 2.0 (Apache-based; acceptable-use policy, attribution) | **parallel hybrid** (attention + Mamba-2 in each layer); native `FalconH1ForCausalLM` | **66** (Deep) | up to 131,072 |
| `tiiuae/Falcon-H1-Tiny-*` (2026) | 90M – 600M | Falcon licence | hybrid; includes a 90M tool-calling model | – | – |
| `Salesforce/xLAM-2-1b-fc-r` | 1.54 / 1.31 B (Qwen2.5-1.5B base) | **CC-BY-NC-4.0** (research only) | `Qwen2ForCausalLM`, GQA 12:2 | 28 | 32K |
| `MadeAgents/Hammer2.1-1.5b` (and 0.5b) | 1.54 B (Qwen2.5-Coder-1.5B base) | **CC-BY-NC-4.0** | `Qwen2ForCausalLM` | 28 | 32K |
| `openbmb/MiniCPM5-1B` (May 2026) | 1.08 / 0.68 B (untied 130k vocab) | Apache-2.0 (yes / yes) | dense GQA **16:2** (24 KB KV / token); native `LlamaForCausalLM`; hybrid thinking template | 24 | 131,072 |
| `openbmb/MiniCPM5-2B` (Sep 2026, ref.) | 2.52 / 1.98 B | Apache-2.0 | dense Llama | 42 | 131,072 |
| Decision-tuned ≤ 2B (ref.): Intern-Decision-0.8B / 2B (Apache-2.0), StartLux-Decision-0.8B / 2B (CC BY-NC 4.0) | 0.8 / 2 B | see repo | Qwen3.5-based decision models | 24 | 262K |

### 3.2 Reported results and estimated forward latency

Benchmarks are the vendors' unless marked. BFCL versions differ and are not comparable across rows (v2, v3, v4, or
an unspecified subset). "repo" marks numbers measured in this repository. Latency = the §1.3 estimate for one
System-1 request at c = 1 on the same GPU, relative to Qwen3-4B.

| model | tool use / function calling (as reported) | general MCQ / other | est. compute ratio | **est. c = 1 latency vs 4B** |
|---|---|---|---|---|
| Qwen3-4B | BFCL v3 57.6 (non-thinking) / 65.9 (thinking) | MMLU-Redux 77.3 / 83.7; repo R0 zero-shot v0 dev 72.9 %, When2Call dev 46.3 % | 1.00 | 1 (A10G: 370 ms per tool-choice request) |
| **Qwen3-1.7B** | BFCL v3 **52.2** / 56.6 | MMLU-Redux 64.4 / 73.9; repo R0 v0 dev 63.8 %, When2Call 10.3 % | 0.39 | **0.40–0.50** |
| **Qwen3-0.6B** | BFCL v3 **44.1** / 46.4 | MMLU-Redux 44.6 / 55.6; repo R0 v0 dev 37.6 %, When2Call 7.7 % | 0.12 | **0.20–0.30** |
| Qwen3.5-0.8B | BFCL-V4 25.3 (non-thinking) | MMLU-Redux 48.5, MMLU-Pro 29.7; repo: StartLux-0.8B C 48.3 %, live 17.5 % | ~0.15 | measured 0.25 (vendor server, L40S); in our `serve_jev` on an L4, 1.5–3.6× slower than Qwen3-0.6B (`results/gpu_round3/bench_serving.md`) |
| Qwen3.5-2B | BFCL-V4 43.6, τ²-Bench 48.8 | MMLU-Redux 69.2, MMLU-Pro 55.3, IFEval 61.2; repo: StartLux-2B C 68.4 %, live 6.5 % | ~0.45 | measured 0.43 (vendor server) |
| Ouro-1.4B (base) | none reported; a 2026 study finds looping helps compositional tool calls (2608.18171) | base, R4: MMLU 67.35, MMLU-Pro 48.62, BBH 71.02, GSM8K 78.92 (Qwen3-4B base: 73.19 / 51.40 / 70.95 / 72.86) | **1.45** | **1.3–1.6 (slower)**; ~5× KV bytes / token at prefill |
| Gemma 3 1B (IT) | BFCL v3 **16.3** (Qwen3 report) | IFEval 80.2, GSM8K 62.8, MMLU-Pro 14.7 | 0.20 | 0.25–0.35 (no fused path) |
| FunctionGemma-270M | BFCL simple 61.6, multiple 63.5, parallel 39, relevance 61.1, irrelevance 73.7 | – | 0.03 | 0.10–0.20 (overhead-bound) |
| Gemma 3n E2B (IT) | – | MMLU 60.1, LiveCodeBench 13.2 | ~0.6 | 0.5–0.7 (5 B in memory) |
| Gemma 4 E2B | τ² (avg of 3) 24.5 | MMLU-Pro 60.0, GPQA-D 43.4 | ~0.6 | 0.5–0.7 (5.1 B in memory) |
| Llama 3.2 1B (IT) | BFCL V2 **25.7**, Nexus 13.5 | MMLU 49.3, IFEval 59.5, GSM8K 44.4 | 0.23 | 0.20–0.30 (16 layers) |
| SmolLM2-1.7B (IT) | "BFCL 27 %" (subset not stated) | IFEval 56.7, MMLU-Pro 19.3, GSM8K 48.2 | 0.47 | 0.45–0.55 |
| LFM2-1.2B | – | MMLU 55.23, GPQA 31.47, IFEval 74.89, GSM8K 58.3 | ~0.25 | 0.25–0.40 (no fused path) |
| LFM2.5-1.2B-Instruct | BFCL v3 **49.12** | MMLU-Pro 44.35, GPQA 38.89, IFEval 86.23 | ~0.25 | 0.25–0.40 (no fused path) |
| **Granite-4.0-1B** | BFCL v3 **54.82** | MMLU 59.39, MMLU-Pro 34.02, IFEval 80.82, GSM8K 76.35 | 0.32 | **0.35–0.50** (40 layers) |
| Granite-4.0-H-1B | (family claims best-in-class BFCL v3; no number retrieved) | – | ~0.3 | 0.35–0.50 (no fused path) |
| MobileLLM-R1-950M | – | MATH500 74.0, GSM8K 67.5, MMLU 47.4 (base) | 0.21 | ~0.25 |
| Falcon-H1-1.5B-Deep (IT) | – | MMLU 66.11, MMLU-Pro 41.89, IFEval 83.5, GSM8K 82.34 | ~0.45 | 0.6–0.9 (66 layers, no fused path) |
| xLAM-2-1b-fc-r | BFCL v3 overall **58.90** (rank 36 at publication), τ-bench 21.8 | – | 0.36 | 0.40–0.45 |
| Hammer2.1-1.5b | BFCL v3: chart only on the card | – | 0.36 | 0.40–0.45 |
| **MiniCPM5-1B** | official numbers in a chart only (vs LFM2.5-1.2B-Thinking, Qwen3-0.6B, Qwen3.5-0.8B); a third-party aggregate lists BFCL v3 simple-AST 0.0 in non-thinking mode (a generation-format failure; unverified, and irrelevant to a scoring read-out) | third-party: MMLU 52.4, IFEval 64.7 (non-thinking); repo (round 3, L4): 37.9 ms vs Qwen3-0.6B 31.2 ms per decision at 1k-token states | 0.16 | **0.20–0.30** |
| MiniCPM5-2B (ref., > 2B) | BFCL v4 66.6, τ²-Telecom 97.1 | MMLU-Pro 70.8, IFEval 81.7 | 0.55 | 0.6–0.8 (42 layers) |

What the zero-shot rows mean for us: R0 (`results/training/aws_smoke_2026-10-01/README.md`) shows small Qwen3
models are poor zero-shot tool selectors (tool_select 24 % / 27 % / 57 % for 0.6B / 1.7B / 4B on v0 dev). SFT took
the 4B from 50.1 % (base) to 99.2 % on C (`results/stage2/stage2.md`, attribution rows). Published BFCL measures
*generating* calls. Our read-out only *scores* options, so a model's BFCL number says more about its prior knowledge
of tools than about how well it will take our SFT.

## 4. Notes per family

- **Qwen3 (0.6B, 1.7B).** Same tokenizer, template, rotary setup and GQA (8 KV heads) as the 4B. Only depth (28 vs 36
  layers) and width differ, which gives the cleanest size curve. Both are in the repo already: the GPU smoke ran all
  14 training stages on 0.6B in about 5 min (`results/training/aws_smoke_2026-10-01/README.md`);
  `80_distill_small.sh` defaults to 0.6B as its student and `10_sft.sh` documents a 0.6B example; prices exist in
  `configs/prices/models.yaml`. Both caches are
  croppable, so `serve_jev --fast` runs its full fast path. KV per token is 112 KB (0.78× the 4B), measured in round
  3 for 0.6B.
- **Qwen3.5 (0.8B, 2B).** These are newer, but on top of the size change they bring a different architecture
  (Gated DeltaNet, which cannot be cropped), a different tokenizer (248k) and a multimodal checkpoint. In our stack
  they were *slower* than Qwen3-0.6B at the agent's state lengths (round-3 bench on an L4: 111 vs 31 ms per decision
  at 1k tokens). The vendor-served StartLux ladder shows their size curve on our workload already. Not recommended
  for a latency-focused E5.
- **Ouro (looped).** The model card and the paper (2510.25741) show strong base-model reasoning for its parameter
  count: Ouro-1.4B R4 is about Qwen3-4B base on BBH and MMLU-Pro. But each token passes through the 24-layer stack 4
  times, with MHA (16 KV heads of 128) per pass. Prefill compute is therefore ~4.9 B effective non-embedding
  parameters per token (1.36× the 4B; 1.45× after its 7 % longer prompts) and the prefill KV cache is ~768 KB per
  token (5.3×). The paper's 4× KV saving applies to decoding, which we never do. The model would also need a separate
  transformers-4.54 environment and `trust_remote_code` support in `load_causal_lm` and `serve_jev`. Early exit
  (`early_exit_threshold`) could cut loops, but it is unsupported in vLLM and untested with our read-out. Verdict:
  interesting for a reasoning-per-parameter study, not for E5.
- **Gemma.** Gemma 3 1B: weak on function calling (BFCL v3 16.3), the 512-token sliding window makes the cache
  non-croppable past 512 tokens (our prompts average ~1.1k), the repo is gated, and the licence is the Gemma terms.
  FunctionGemma-270M could be a "how small can it go" curiosity point. Gemma 3n and Gemma 4 E2B hold 5 B weights in
  memory and are multimodal; "effective 2B" describes compute, not memory.
- **Llama 3.2 1B.** At 16 layers it is the shallowest candidate, which helps c = 1 latency. Against it: the weakest
  tool-use numbers (BFCL V2 25.7), the Llama licence (gated, attribution), and a template that stamps today's date
  (verified). Using it needs `chat_text` to pass a fixed `date_string` (a code change) so that training and serving
  prompts match.
- **SmolLM2 / SmolLM3.** SmolLM2-1.7B uses full MHA (KV 1.33× the 4B per token) and an 8K context. SmolLM3 is 3B
  and date-stamped. Neither has an edge.
- **LFM2 / LFM2.5.** These are the fastest architectures on paper (only 6 attention layers) and LFM2.5 has a good
  BFCL v3 (49.1). The licence withdraws free commercial use above $10M revenue, and the conv cache is not croppable.
  Fine for a research row, but not as the "deployable" recommendation.
- **Granite 4.0 Nano.** `granite-4.0-1b` has the best published BFCL v3 (54.8) of the general-purpose ≤ 2B non-Qwen
  models, an Apache-2.0 licence, a croppable cache and the most compact prompts (0.83×). It has 40 layers, though,
  and 1.6B parameters (the size of Qwen3-1.7B), so it adds a family point, not a smaller size point. The `-h-` hybrids
  are not croppable.
- **Falcon-H1.** A parallel hybrid. The 1.5B-Deep has 66 layers (a poor c = 1 latency floor) and its tokenizer is
  ~20 % longer on our prompts. Falcon-H1-Tiny (2026) adds 90M–600M models, including a 90M tool-caller.
- **Function-calling fine-tunes (xLAM-2-1b, Hammer2.1-1.5b).** Strong BFCL (xLAM-2-1b: 58.9) but CC-BY-NC-4.0, and
  both are Qwen2.5-1.5B fine-tunes tuned to *generate* calls. Under our SFT on harness states they would probably
  end up close to a Qwen2.5-1.5B base. Neither is worth a slot.
- **MiniCPM5-1B.** A conventional dense Llama stack (24 layers, GQA 16:2) with an Apache-2.0 licence and no remote
  code (its config was written by transformers 5.6.2). Its KV cache is the smallest of the candidates (24 KB per
  token), it is croppable, and its prompts are 0.85×. It is already in `scripts/bench_serving.py`'s default models and
  in `docs/DECISION_MODEL_LANDSCAPE.md` as "the dense control". Risks: it is new (on HF since 2026-05-21, last
  modified 2026-08-17), the official benchmark numbers are image-only, and its 0.40B untied embedding is a large
  share of 1.08B (memory, not compute).

## 5. Recommendation for E5 "small decision models"

1. **`Qwen/Qwen3-1.7B`** (revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`): the primary small model. Same family,
   byte-identical prompts, est. 0.40–0.50× the 4B's latency at c = 1, best published small-Qwen3 BFCL (52.2
   non-thinking vs 57.6 for the 4B), and memory headroom on a 24 GB A10G (probably no activation checkpointing).
2. **`Qwen/Qwen3-0.6B`** (revision `c1899de289a04d12100db370d81485cdf75e47ca`): the low end of the size curve,
   0.12× the 4B's compute. It has already run end to end through the training chain in this repo. If it holds C
   accuracy, the efficiency claim gets much stronger; if it does not, the curve shows where SFT stops carrying the
   gain.
3. **`openbmb/MiniCPM5-1B`** (revision `87179e5c1f455ef22e6223592d2d61351b525bfc`): the non-Qwen ~1B point. It runs
   on plain transformers (`LlamaForCausalLM`), passes the readout preflight (its hybrid-thinking template renders the
   same empty think block as Qwen3), is croppable, has the smallest KV cache, and has been served by `serve_jev`
   before (round 3). It tests whether the recipe transfers beyond the Qwen family at a third size (0.68B
   non-embedding).
   *Backup if MiniCPM5-1B fails the smoke:* `ibm-granite/granite-4.0-1b` (revision
   `6a7381ba1f54d684ff508d991aeb7dc580157103`; Apache-2.0, croppable, BFCL v3 54.8; 40 layers).

Not recommended: Ouro (no efficiency gain, unsupported stack), Qwen3.5 small (confounded and slower in our serving
stack), LFM2.5 (licence threshold, non-croppable), Gemma 3 / Llama 3.2 (weak tool priors; gated; Llama's date
stamp), Falcon-H1 (depth, hybrid), xLAM / Hammer (non-commercial, no gain for scoring), MobileLLM-R1
(non-commercial, reasoning-only).

## 6. What our trainer and read-out would not support as is

| issue | affects | consequence | fix |
|---|---|---|---|
| no `trust_remote_code` in `load_causal_lm` / `AutoTokenizer` calls; model type absent from transformers 5.18 | Ouro (card pins < 4.56), MobileLLM-R1 (custom registration) | load fails; the tokenizer prompt can hang a headless box | separate env + code change; out of scope for E5 |
| date-stamped chat template (`strftime_now`) | Llama 3.2 (verified), SmolLM3 | training and serving prompts differ by date | pass a fixed `date_string` in `chat_text` (applies to new models only) |
| non-croppable cache | Qwen3.5, LFM2 / 2.5, Granite-H, Falcon-H1, Gemma 3 past 512 tokens | `serve_jev --fast` loses the fused multi-question pass and longer-entry reuse (read-out stays exact: verified) | none needed for correctness; latency comparisons must say so |
| multimodal checkpoints | Qwen3.5 (handled), Gemma 3n / 4 (untested) | possible fallback to the multimodal class (memory) | test before use |
| gated repos | Gemma, Llama, MobileLLM | the box needs an HF token (a secret) | prefer ungated models (all three recommended are ungated) |
| `enable_thinking` detection by template string | any template that mentions it | `chat_text` passes `False`; for Qwen3 / Qwen3.5 / MiniCPM5 the prompt ends with an empty think block, as for the 4B (verified) | none |
| LoRA `all-linear` | untied LM heads (MiniCPM5, Ouro) | PEFT's `all-linear` leaves the output layer out; the head stays frozen, as for the tied 4B | none |
| μP logit scaling | Granite (`logits_scaling` 8) | raw logits scaled; the per-bucket temperature fit absorbs it | none |

## Sources

Models and cards: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) ·
[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) · [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) ·
[Qwen3 technical report, Tables 19–20](https://arxiv.org/abs/2505.09388) ·
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) · [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) ·
[Qwen3.5 small-model announcement](https://x.com/Alibaba_Qwen/status/2028460046510965160) ·
[Qwen3.8 (no ≤ 2B sizes)](https://github.com/QwenLM/Qwen3.8) ·
[Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) ·
[Ouro-1.4B config](https://huggingface.co/ByteDance/Ouro-1.4B/raw/main/config.json) ·
[Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking) ·
[Ouro paper, arXiv 2510.25741](https://arxiv.org/abs/2510.25741) ·
[vLLM added (Oct 2025) and removed (Jul 2026) Ouro](https://github.com/microprediction/latentreasoning/issues/10) ·
[Looped LMs and compositional tool calling, arXiv 2608.18171](https://arxiv.org/abs/2608.18171) ·
[Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3) ·
[gemma-3-1b-it config (ungated mirror)](https://huggingface.co/unsloth/gemma-3-1b-it) ·
[FunctionGemma-270M](https://huggingface.co/google/functiongemma-270m-it) ·
[FunctionGemma announcement](https://blog.google/technology/developers/functiongemma/) ·
[Gemma 3n model card](https://ai.google.dev/gemma/docs/gemma-3n/model_card) ·
[Gemma 3n blog](https://huggingface.co/blog/gemma3n) ·
[Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) · [Gemma 4 blog](https://huggingface.co/blog/gemma4) ·
[Llama 3.2 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) ·
[Llama-3.2-1B-Instruct (ungated mirror)](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) ·
[SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) ·
[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) ·
[LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B) ·
[LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) ·
[LFM2 technical report](https://arxiv.org/abs/2511.23404) · [LFM licence](https://www.liquid.ai/lfm-license) ·
[LFM2.5 release coverage](https://gigazine.net/gsc_news/en/20260107-lfm2-5-on-device-ai/) ·
[Granite-4.0-1B](https://huggingface.co/ibm-granite/granite-4.0-1b) ·
[Granite 4.0 Nano blog](https://huggingface.co/blog/ibm-granite/granite-4-nano) ·
[MobileLLM-R1-950M](https://huggingface.co/facebook/MobileLLM-R1-950M) ·
[Falcon-H1-1.5B-Deep-Instruct](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Deep-Instruct) ·
[Falcon licence](https://falconllm.tii.ae/falcon-terms-and-conditions.html) ·
[Falcon-H1-Tiny](https://tiiuae-tiny-h1-blogpost.hf.space/) ·
[Falcon-H1-Tiny tool-calling 90M](https://huggingface.co/tiiuae/Falcon-H1-Tiny-Tool-Calling-90M) ·
[xLAM-2-1b-fc-r](https://huggingface.co/Salesforce/xLAM-2-1b-fc-r) · [APIGen-MT / xLAM-2 paper](https://arxiv.org/abs/2504.03601) ·
[Hammer2.1-1.5b](https://huggingface.co/MadeAgents/Hammer2.1-1.5b) ·
[MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) · [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) ·
[MiniCPM5 release coverage](https://rits.shanghai.nyu.edu/ai/minicpm5-2b-tops-open-models-under-4b/) ·
[benchlm.ai MiniCPM5-1B (third-party)](https://benchlm.ai/models/minicpm5-1b) ·
[When2Call, arXiv 2504.18851](https://arxiv.org/abs/2504.18851).
HF API (`/api/models/<id>`) for revisions, creation dates and licence tags (2026-10-06).

Repo evidence: `results/gpu_round3/bench_serving.md` (L4 serving bench: Qwen3-0.6B, MiniCPM5-1B, Qwen3.5-0.8B),
`results/training/aws_smoke_2026-10-01/README.md` (R0 zero-shot), `results/stage2/stage2.md` and
`results/explore/startlux_live/summary.md` (StartLux ladder, attribution rows), `results/stage2/s23/answers_*.jsonl`
(per-question latencies), `results/harness_final/{replay_bench,summary}.md` (4B serving on an A10G).
