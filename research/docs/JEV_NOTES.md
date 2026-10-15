# Jev / OpenJev: facts we rely on (as of 2026-09-30)

[P] = read from primary source code (official SDK / repo); [S] = secondary (docs pages we could not
fetch directly, blogs, search snippets). Re-verify [S] items before the camera-ready.

## Jev (TypeSafe AI) — launched 2026-09-15 (early access)

- **Endpoint** [P]: `POST https://api.typesafe.ai/v1/systemone`, `GET /v1/models`, header
  `Authorization: Bearer $TYPESAFE_API_KEY`. Official Python SDK `pip install typesafe-sdk`
  (`TypeSafeClient().system_one(state=..., questions={...})`), JS `@typesafe-ai/sdk`.
- **Request** [P]: `{"state": str|dict|list, "model": "jev-latest", "questions": {id: Q}}`
  - `{"type":"choice","instructions":..., "criteria": {label: description|null}}` (≤255 options [S])
  - `{"type":"score","instructions":..., "criteria": [level0, level1, ...]}` (≤10 levels [S])
  - `{"type":"noul","instructions":..., "criteria": {"true":..., "false":...}}`
- **Response** [P]: `{"model": "jev-1.13.0", "answers": {id: A}, "usage": {"input_tokens", "output_tokens"}}`
  - choice → `{"choice", "confidence", "probabilities"}`; score → `{"score" (expected level index),
    "confidence", "legend", "probabilities"}`; noul → `{"noul": P(yes)}` (**no** confidence field)
  - live responses may carry extra fields (`stats`, `assets_used`) → clients must ignore unknown fields.
- Questions in one request share the state, run in parallel, never see each other's answers [P].
- **Price** [S]: $0.042 / M input tokens, output free. **Limits** [S]: 64k tokens/request (32k state +
  longest question), 250k tok/s, 1,200 req/min. Text only. English primary.
- **Models** [S]: aliases `jev-latest`, `jev-preview`; pinned `jev-1.13.0` (we pin this).
- Errors [P, via OpenJev's live-checked shapes]: 400 unknown model / bad criteria, 401/403, 422
  validation, 429, 529 overloaded. SDK retries 408/429/5xx with backoff 0.5s→5s.
- Vendor guidance [P, typesafe-ai/skills]: code owns thresholds and side effects; always include an
  `other`/`none` option; confidence = concentration, not permission; Jev "should not be the component
  that grants a capability"; compose multi-label/ranking/extraction from primitives.
- Vendor benchmark [S]: evals.typesafe.ai 4 workflows — Jev 67.8% @ ~$0.0004 & 0.4 s/case vs
  GPT-5.6 Terra 67.9% @ ~$0.03 & 10 s (labels = consensus of frontier models).
- Training reportedly "RLCD" (RL for calibrated decisions) — **unverified**, no tech report.

## Open alternatives

| Project | What | API | Notes |
|---------|------|-----|-------|
| **razorback16/openjev** (Apache-2.0) | DiffusionGemma-26B-A4B (NVFP4), zero-shot masked-slot read, re-read ×4 if entropy > 0.1 | Jev wire-compatible `/v1/systemone` (+ `samples`, `steps`, `think`, `images`, `sequential` extensions) | ≥24 GB GPU (`docker run razorback16/openjev:0.5.0`), Mac MLX; p50 94 ms @ 3 Q; free hosted `https://api.codiv.ai` (100M tokens) |
| **internlm/Intern-Decision-0.8B / 2B / 4B** (Apache-2.0, 2026-09-26) | Qwen3.5 decision-tune; one forward pass answers up to 16 fields of ≤ 62 options (more fields: one pass per 16; wider Choices: a multi-pass knockout): options -> single-token symbols A-Z a-z 0-9 (<= 62), assistant JSON skeleton with a `<decision>` placeholder per field, logits read before each placeholder, per-checkpoint temperature (2.748 / 2.101 / 1.992) | Jev-style request/response dicts (local `inference.py`, no HTTP); in-process backend `{type: intern_decision}` (docs/BASELINES.md §7) or behind our wire server | 1-16 questions/request, 8k tokens, text-only in our integration (their path also takes <= 8 images); self-reported Jevbench avg 79.4 / 84.7 / 90.0 vs Jev 88.7, 33-44 ms on a 4090 |
| **openjev/openjev (HF) + abhishekgahlot2/openjev-server** | 27B fine-tune (base unverified; weights CC-BY-NC-4.0) | Jev-like, first-token letter logprobs, ≤52 options/pass, `--perms 4` | 84.0% vs Jev 85.4% on its 10k-question benchmark |
| zhangcy122/OpenJev ("OpenJevPro") | client library (PolyForm-NC) | – | logprob readout + temperature, System-2→1 "crystallisation" flywheel |
| typesafe-ai/system-one-adapter-python | official LLM-backed Jev-compatible adapter | Jev wire | baseline idea |
| JevK5, Imajev-4B, decider-4b, Bosun, CLM, Laya, Verdict | community Jev-likes (Qwen3.5-4B LoRA etc.) | mostly Jev wire | see docs/TRAINING_RECIPES.md §1 |
| pi `classify()` + `llama-cpp-classify` | built into the pi agent | – | TypeSafe provider `typesafe/jev-latest` |

## Concurrent / prior work to position against (Sept 2026 arXiv, all v1)

- **REFLEX** 2609.26532 — Jev decision layer + escalation; 95% success with 72.7% fewer strong-model calls. *Closest.*
- **Jev-Mem** 2609.23986 — Jev-controlled memory; LoCoMo judge 0.777 (+11%), 6.6× faster construction.
- **JEV-Star** 2609.27331 — Jev action selector + GPT-6 planner in SC2 (v1 full text: 4/4 non-cheating full-game wins, micro wins 7/105 combined vs 3/105 Jev-only; $3.71/game, 96% planner cost. An earlier "9/1 wins" note here was not in the paper; corrected 2026-10-04 per docs/JEV_GALLERY_STUDY.md F9).
- 2609.28919 control-the-harness (routing coding agents) · 2609.28940 pentest harness (Jev ~250 ms vs Laya ~35 ms: vendor-documented figures the paper copies, not a controlled benchmark) ·
  2609.37647 benchmarking Jev (37 datasets) · 2609.32160 typed-decision audit · 2609.26550 Jev-as-a-judge ·
  2609.22753 edge orchestration · security/robustness: decision hijacking, JevOut, adversarial bench.

Our differentiation: (1) systematic WHERE × HOW design space rather than one decision point; (2) cost-
and latency-accounted benchmark across static / controllable / real-harness levels; (3) the same
configurations inside pi, Codex, DeepSeek Harness, mini-SWE-agent; (4) training recipes + open model.
