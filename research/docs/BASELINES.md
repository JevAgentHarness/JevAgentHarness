# Baselines and controls for System-1 decisions

**Reviewer question this answers:** *is the fast decider better than cheap non-learned alternatives, at the
same invocation rate and cost?* Every baseline below is an ordinary `DecisionBackend`
(`src/jevharness/decision/`). It plugs into every decision point and integration mode unchanged, and is
selected with one `s1` spec in the experiment YAML. The harness, prompts, System-2 model and suite stay
fixed, so arms differ only in who answers the typed question.

Configs: `configs/experiments/baselines_jevgym.yaml` (L2 JevGym, E2c) and
`configs/decision/baselines_static.yaml` (L1 BFCL / When2Call, next to E1).
Tests: `tests/test_baselines.py` (offline).

## 1. Baselines at a glance

Cost and latency figures are this repo's own measurements from the pipeline checks in §5, or the price
table in `configs/prices/models.yaml`. Latency is per S1 call, measured in-process. The free baselines never
touch the network.

| Baseline | Spec | Answers | What it is | Cost | Latency | Why it matters | Reference |
|---|---|---|---|---|---|---|---|
| **BM25** | `{type: bm25}` | Choice | Okapi BM25 (k1=1.2, b=0.75) between the query (rendered state + question) and each option's document (name + description). Softmax(T=1) gives the distribution. Tool-name-aware tokeniser (snake/camel/kebab, light stemming). | $0 | ~0.1 ms (L1 p50), ~1 ms in-harness | The canonical sparse tool/memory retriever. If Jev does not beat it, delegation is just retrieval. | Robertson & Zaragoza 2009; Gorilla's BM25 retriever `patil2024gorilla`; ToolLLM retriever comparison `qin2024toolllm`; ToolRet `shi2025toolret` |
| BM25, task query | `{type: bm25, query_mode: task_obs}` | Choice | As above, but the query is only TASK + CURRENT OBSERVATION: the usual "instruction as query" tool-retrieval setup. | $0 | ~1 ms | Isolates whether the history helps or hurts a lexical ranker. | as BM25; RAG-MCP `gan2025ragmcp` |
| **TF-IDF cosine** | `{type: tfidf}` | Choice | Sublinear-TF, smoothed-IDF cosine. Softmax(T=0.05), the same scale as the dense-embedding baseline. | $0 | ~0.5 ms (L1 p50), ~1.7 ms in-harness | The second classic sparse ranker. Brackets BM25's sensitivity to length normalisation. | Salton & Buckley 1988 |
| Dense embedding *(existing)* | `{type: embedding, provider: gemini, model: gemini-embedding-001}` | Choice | Cosine between state and option embeddings. | $0.15/M tok (verify) | ~370 ms (API) | The learned-retriever reference. | ToolRet `shi2025toolret` |
| **Rate (matched-rate random)** | `{type: rate, rate: r, seed: s}` | all | Noul answers yes with probability r (per decision point via `rates:`); Choice/Score are uniform-random. In guard modes, r is the acceptance rate. Hash-seeded, so reproducible and thread-safe. | $0 | <0.1 ms | Separates *which* steps a trigger picks from *how often* it fires. This is the control for reflection / stop / routing / memory triggers. | random-router reference in LLM routing (`ong2025routellm`) % CHECK |
| Random score (τ-matched) | `{type: rate, rate: null}` | Noul | p_yes ~ U(0,1), so the arm fires iff u ≥ τ, at rate 1−τ. Sweeping τ gives a random curve with the *same thresholds* as the Jev τ sweep. | $0 | <0.1 ms | A random reference curve on the cost–success plane. | — |
| **Prior** | `{type: prior, rule: majority\|first\|last\|recent, prior: {...}\|prior_file: ...}` | all | Always the prior's answer. `majority`: the highest-weight label from a fitted prior (fit on a *training* split with `fit_prior` / `fit_prior_from_trajectories`). `first`/`last`: positional. `recent`: "call the most recently useful tool". Noul gets a constant `p_yes`. | $0 | ~0.01 ms (L1), ~0.2 ms in-harness (`recent` parses the history) | The majority-class / "do what worked last" reference. It shows how much of a decider's accuracy is label imbalance. | — |
| **Heuristic** | `{type: heuristic}` | all | Deterministic rules on the rendered state text, per decision point (§2). No gold, no suite-specific tuning. | $0 | ~0.1 ms (Noul), ~1 ms (tool Choice) | The "a few if-statements" competitor that any learned trigger must beat. | Reflexion's trigger-on-failure `shinn2023reflexion` (rule inspiration) |
| **S2-self** | `{type: s2_self, llm: {provider: gemini, model: gemini-3.5-flash}, reasoning_effort: minimal}` | all | The System-2 LLM answers the same typed questions: `llm_generative` pointed at the S2 model, under its own backend name. | S2 price per decision | S2 latency | Asks whether a *separate* fast model is needed, or the big model can decide for itself. It is the upper reference for accuracy per dollar. | — |
| Gemini stand-in *(existing)* | `{type: llm_generative, llm: {provider: gemini, model: gemini-3.5-flash-lite}, reasoning_effort: minimal}` | all | A cheap generative LLM with a verbalised distribution. | $0.30 / $2.50 per M (verify) | ~1.2 s | The "just use a small LLM" alternative. | — |
| Non-S1 references *(existing modes)* | `mode: off \| always \| llm`, `llm: small_only` | — | LLM-only, reflect / store every step, the LLM gates memory itself, small model only. | — | — | Brackets every trigger arm. | — |

**Deliberate limitations.**
- Lexical backends cannot answer Noul / Score. Pass `fallback: {type: heuristic}` (or any spec) to mix them
  into trigger or guard arms; without a fallback they raise, as the embedding backend does.
- They never pick an abstain option ("none", OTHER) on lexical evidence. `abstain_score: t` adds a
  retrieval threshold, and `t` must be tuned on a dev split.
- The rules report a fixed confidence (default 0.99). Their trigger and guard outcomes are therefore
  τ-independent, and under cascade they behave like replace; set `confidence` below τ to make them defer.
  Rate reports 0.999 / 0.001 for the same reason.

## 2. Which baseline at which decision point

| Decision point (mode) | BM25 / TF-IDF | Heuristic rule | Prior | Rate | Other references |
|---|---|---|---|---|---|
| `tool_select` (replace / cascade / assist) | query ↔ name + description | BM25(task + latest obs) plus workflow rules. Finish goes first once the task looks complete and last before any progress. State-changing tools wait for a successful read-only call (look before you leap). Tools that already succeeded go down; dead ends (empty result, or 2 errors) go last. | `recent` / `majority` | uniform random (random top-k in assist) | embedding, stand-in, s2_self |
| `tool_select` (guard, Noul) | via `fallback` | reject an exact repeat of an earlier call | constant | accept w.p. r | — |
| `tool_params` (Choice over enum / copied spans + OTHER) | weak (spans are short) | enum: BM25 against the task. Spans: the most recent `arg: value` pair in the state, then spans typed like the argument (IDs for `*_id`, numbers for amounts, multi-word capitalised spans for names, ...). Otherwise OTHER, which defers to the LLM. | `first` (most recent span) / `last` (always defer) | random | — |
| `memory_write` (trigger) | — | keep iff the observation carries an ID / number / email / date and is not an error or empty | constant (≡ always / off) | store w.p. r | `always`, `llm` gate |
| `memory_read` (assist) | standard sparse RAG over memories | recency (newest first, "none" last) | position | random k | `always` (recency-k) |
| `reflection` (trigger) | — | fire on *trouble*: error words in the last observation, an empty / "no relevant result" observation, or the last action repeated within 3 steps | constant (≡ always / off) | fire w.p. r | `always`, s2_self |
| `stop` (trigger) | — | fire iff the last call was to a state-changing tool (generic verb list: cancel, refund, ship, send, create, update, delete, book, pay, ...) and it succeeded | constant | fire w.p. r | — |
| `stop` (guard) | — | accept iff (the instruction requests a change and a state-changing call succeeded) or (an information request and some call returned a result) | constant | accept w.p. r | — |
| `routing` (trigger) | — | strong model on *trouble* (the same signal as reflection) | constant | strong w.p. r | `small_only`, `llm_only`, s2_self |
| `action` (discrete envs) | query ↔ admissible actions | BM25 plus no recent repeats plus dead ends last | `first` | random | — |

Questions that arrive over the Jev wire (`decision/server.py`, the harness adapters) carry no decision
point. The controls recover it from the harness's question wording (`controls.infer_dp`), so the same
backends serve τ², pi, Codex and the other adapters behind `python -m jevharness.decision.server --s1 '{"type":"heuristic"}'`.

## 3. Matched-rate control protocol (meta-level triggers)

A trigger arm's benefit mixes two things: how often it fires (more reflection or strong-model calls cost
more and may help regardless of timing) and *which* steps it picks. The matched-rate control keeps the
first and randomises the second.

1. **Run the System-1 arm**, e.g. `jev_reflect_trigger` (τ = 0.5), on the full suite.
2. **Measure its per-opportunity rate** r̂ from the logs:
   ```bash
   python -c "from jevharness.decision.controls import trigger_rates as t; \
              print(t('runs/e2c_jevgym_baselines/jev_reflect_trigger.jsonl'))"
   # {'reflection/trigger': {'n': <opportunities>, 'rate': <r_hat>}}
   ```
   Trigger rows give the firing rate, guard rows the acceptance rate, routing rows the strong-model fraction,
   and `memory_write/<mode>` stored / offered.
3. **Run the rate control at r̂** on the same tasks, the same τ and the same System-2:
   `-o 's1_backends.rate_m={type: rate, rate: <r̂>, seed: 0}'` plus an arm `{name: ratem_reflect_trigger, s1: rate_m, reflection: {mode: trigger, tau: 0.5}}`.
   Use one `seed` per trial (seeds 0, 1, 2 for pass^k). The draws are hashes of (seed, state, question), so a
   rerun is identical and caching is safe.
4. **Check the realised rate.** A firing changes the trajectory, and with it the number of later
   opportunities, so the realised rate of the control can drift from r̂. Report it for both arms (`trigger_rates`
   again); if they differ by more than 0.05, rerun once with r corrected by the ratio.
5. **Compare at matched cost.** The triggered module (the reflection call, the strong model) dominates
   cost, and S1 cost is about 0 (rate) against about $0.04/M input tokens (Jev). Equal realised rates
   therefore mean near-equal $. Report success, $/episode, latency and realised rate, with a paired
   bootstrap / McNemar on per-task success (`bench/metrics.py`: `paired_bootstrap_diff`, `mcnemar`, Holm across
   decision points).
6. **The claim.** "Jev picks the right moments" holds for a decision point iff Jev beats `rate(r̂)` at the
   matched realised rate. It should also be compared with the deterministic heuristic, placed on the
   cost–success plane at its own realised rate, and with the fixed rates 0.1 / 0.3 / 0.5 that bracket typical
   trigger rates. The τ-matched `randscore` sweep gives a random *curve* next to the Jev τ sweep; Jev's curve
   should Pareto-dominate it.
7. **Object-level analogue.** For tool selection a System-1 call happens at every step in every arm, so
   invocation counts match by construction and the free baselines are cost-matched from below. For
   *cascade*, compare backends at matched **deferral rate**, not matched τ, because lexical and rule
   confidences are not calibrated. Take the τ from each backend's sweep whose `defer_rate`
   (`decision_table`) equals Jev's, or wrap the backend in `{type: calibrated, inner: ...}` first.

Honesty rules for every control: they read only the state string, question text, options and decision
point, never `Question.meta` (gold); a test enforces this. Priors are fitted on training splits only. The
rules use generic English cues (error words, state-changing verbs) and are not tuned to JevGym. If a rule
changes, bump `VERSION` in `controls.py` / `lexical.py`. `describe()`, and hence the `s1_cache` key,
includes it; otherwise cached answers from the old rules are replayed (this happened during the check below).

## 4. How to run

```bash
# L1 static (free, offline once data/raw is populated)
jevbench decide configs/decision/baselines_static.yaml                     # full E1 item set
# L2 JevGym, all arms (Jev default s1 needs openrouter.ai; ~46 arms x 200 tasks -- budget accordingly)
jevbench run configs/experiments/baselines_jevgym.yaml
# L2 without Jev: Gemini stand-in as the default s1, a subset of arms, fresh run name (the runner resumes by task id)
jevbench run configs/experiments/baselines_jevgym.yaml --max-tasks 5 -o name=e2c_check workers=4 'sweeps=[]' \
  's1={"type":"llm_generative","llm":{"provider":"gemini","model":"gemini-3.5-flash-lite"},"reasoning_effort":"minimal"}' \
  'arms=[{name: llm_only}, {name: bm25_ts_assist_k3, s1: bm25, tool_select: {mode: assist, k: 3}}]'
# harder variant (near-duplicate distractors hurt lexical rankers most)
jevbench run configs/experiments/baselines_jevgym.yaml -o suite.distractor_style=near name=e2c_jevgym_baselines_near
```

## 5. Pipeline check (2026-09-30), not paper numbers

**L1** (`runs/decision/e1_baselines_static/`): 150 shuffled items (BFCL tool_select 52, BFCL tool_params
68, When2Call 30), with every backend free and the p50 latency below 1.3 ms. Tool-select accuracy on BFCL:
TF-IDF 0.77, BM25 0.73, `prior_last` ("none") 0.42, random 0.42, `prior_first` 0.35. On When2Call:
TF-IDF 0.30, first 0.27, BM25 0.23, random 0.20, last 0.00. On BFCL tool_params, `prior_last` (always
OTHER) gets 0.47: on these items the gold value is often not among the copied candidates. The config's
full BFCL sample crashes in `load_bfcl`, because 4 `live_irrelevance` rows have an empty function list
(a pre-existing bug in `bench/suites/static.py`; E1's `bfcl_when2call.yaml` draws the same seed-0 sample and
crashes the same way; the fix is to skip rows without functions). The check skipped those rows.

**L2** (`runs/e2c_jevgym_baselines_check/`): JevGym strict, 30 distractors, 5 tasks per arm, gemini-3.5-flash
System-2, non-Jev arms only, about $0.88 in total. LLM-only solved 5/5 at $0.011/episode. On tool
selection, the Gemini stand-in assist-k3 solved 5/5, and embedding, TF-IDF and BM25 assist-k3 solved
3, 3 and 1 of 5. BM25, TF-IDF and `prior_recent` replace solved 0/5. The heuristic solved 4/5 with
assist and 1/5 with replace, under rules v2 (before the dead-end rule of v3). The lexical rankers miss
prerequisite tools (`find_user` is not lexically close to "cancel their order"), which is the ToolRet
observation. On meta-level controls, the tasks gave no trouble, so the heuristic never fired: reflection
and routing rates were 0. `heuristic_route` therefore ran everything on flash-lite at 5/5 and
$0.0023/episode. `rate30` realised 0.33 (reflection) and 0.41 (routing). These tasks are at the
LLM-only ceiling, so run the meta arms on the noisy / near-distractor variants for signal.

## 6. Citations to add to `paper/references.bib`

These are not in the bib yet; add them when the baselines are written up. BM25 and TF-IDF are cited in
`decision/lexical.py`.

```bibtex
@article{robertson2009bm25,
  title   = {The Probabilistic Relevance Framework: {BM25} and Beyond},
  author  = {Robertson, Stephen and Zaragoza, Hugo},
  journal = {Foundations and Trends in Information Retrieval},
  volume  = {3}, number = {4}, pages = {333--389}, year = {2009}
}
% CHECK
@article{salton1988termweighting,
  title   = {Term-weighting approaches in automatic text retrieval},
  author  = {Salton, Gerard and Buckley, Christopher},
  journal = {Information Processing \& Management},
  volume  = {24}, number = {5}, pages = {513--523}, year = {1988}
}
% CHECK
```
The existing keys used above are `patil2024gorilla`, `qin2024toolllm`, `shi2025toolret`, `gan2025ragmcp`,
`shinn2023reflexion` and `ong2025routellm` (check that RouteLLM reports a random-router baseline before
citing it for that).

## 7. Intern-Decision backend (in-process open System-1, next to Jev)

**What it is.** `internlm/Intern-Decision-{0.8B,2B,4B}` (InternLM, released 2026-09-26; Apache-2.0, derived from
Qwen3.5-{0.8B,2B,4B} with the Qwen licence preserved) are decision-tuned models that take a Jev-style request
`{state, questions: {id: {type: choice|score|noul, instructions, criteria}}}` and return the Jev response shape
(`answers` with `probabilities`, `confidence`, `choice` / `noul` = P(yes) / `score` = expected value, `usage`).
Self-reported "Jevbench" averages 79.4 / 84.7 / 90.0 vs Jev 88.7 (`docs/DECISION_MODEL_LANDSCAPE.md`). Backend:
`src/jevharness/decision/intern_decision.py`; tests: `tests/test_intern_decision.py` (offline, tiny Qwen3.5 checkpoint).

**How it differs from Jev / our served deciders.** Jev is an HTTP service; our trained deciders
(`training/serve_jev.py`) render one prompt per question and read the option-code logits at the *last* position.
Intern-Decision is a **multi-field read-out (one pass per ≤16 fields of ≤62 options) with placeholders**: every option is mapped to a
single-token symbol (`A..Z a..z 0..9`, so <= 62 options), ONE prompt holds the state, a "decision schema" with
every field and a complete *assistant* JSON skeleton `{"team": "<decision>", ...}` (chat template with the empty
`<think>` block), and one forward pass reads the logits at the token *before* each `<decision>` placeholder,
softmaxed over that field's symbols and temperature-scaled with the checkpoint's fitted calibration (0.8B: 2.748,
2B: 2.101, 4B: 1.992; argmax-preserving). k questions about one state therefore cost one prefill, not k; a wider Choice is a knockout (69 JevGym tools via the harness: 62+7 in one pass, then the final read — 2 passes, logged as `n_passes`/`n_rounds`), and no
text is generated. Up to 16 fields per pass (more are chunked), no images in our integration.

**Our mapping.** `Choice` -> `choice` with `criteria = {option: description | ""}`; `Noul` -> `noul`
(`meta["criteria"]` true/false descriptions as on the Jev wire; the model's option order is `no`, `yes`);
`Score` -> `score` in the object form keyed by **our** scale values (`{"1": ..., "2": ...}`), so `probabilities`
come back keyed by our labels and the expected `score` is already on our scale. Skeleton field names are the wire
client's field names (`decision/server.py`), else the harness decision point (`tool_select`, `stop`, ...), else
`q<i>`; nothing else from `Question.meta` is rendered (regression test). Choice over more than `max_options`
(62, configurable) runs the same knockout as `serve_jev` (chunks, top-2 advance), and the harness tournament
chunks to `max_options` before that. Prompts over `max_length` (8192) drop the oldest part of the state and flag
`raw["truncated_state"]` (`truncate_state: false` raises, as the reference does). Prompt compilation and the
calibration arithmetic are vendored from the checkpoint's `inference.py` with attribution (byte-identical prompt;
the reference file needs Python >= 3.12 and Pillow/torchvision, ours needs neither); the model is loaded
text-only through `training.train_decision.load_causal_lm` (`Qwen3_5ForCausalLM`, no vision tower).

**Accounting.** Every call goes through the `Ledger`: `input_tokens` = prompt tokens per forward pass,
`output_tokens` = scored fields (the reference's `usage` convention), `latency_ms` = forward-pass time (device
sync included; the harness's wall-clock still includes queueing when several workers share one model), cost from
`configs/prices/models.yaml` (`internlm/Intern-Decision-*`: an *assumed* amortised RTX-4090 cost, ~$0.008/M input,
to be recomputed from measured throughput). `describe()` carries model, dtype, device, max_length, temperature,
`batch: false` (one request per forward; questions inside a request are batched) and the knobs above, so the
cache key and `backend_id` separate a bf16 GPU run from an fp32 CPU run and a calibrated from an uncalibrated arm.

**Specs and runs.**
```yaml
s1: {type: intern_decision, model: internlm/Intern-Decision-2B}                 # device auto, T from the checkpoint
s1: {type: intern_decision, model: internlm/Intern-Decision-0.8B, device: cuda, dtype: bfloat16, max_length: 8192,
     temperature: 2.747760550703, max_questions_per_request: 16, max_options: 62, truncate_state: true, state_json: true}
```
```bash
# E2a with Intern-Decision as System-1 (same suite/modes as jevgym_tool_select.yaml, paired Jev arms included)
jevbench run configs/experiments/jevgym_tool_select_intern.yaml                # 2B default; 0.8B / 4B arms inside
jevbench run configs/experiments/jevgym_tool_select.yaml -o 's1={type: intern_decision, model: internlm/Intern-Decision-2B}' name=e2a_intern2b
# E1 (jevbench decide has no -o): add to `backends:` in configs/decision/bfcl_when2call.yaml
#   intern2b: {type: intern_decision, model: internlm/Intern-Decision-2B}
#   intern2b_uncal: {type: intern_decision, model: internlm/Intern-Decision-2B, temperature: 1}
# Behind the Jev wire for the adapters (pi, Codex hooks, tau2, mini-SWE-agent):
python -m jevharness.decision.server --port 8090 --name intern2b --s1 '{"type":"intern_decision","model":"internlm/Intern-Decision-2B"}'
```

**CPU smoke (2026-10-01, pipeline check, not a result).** `Intern-Decision-0.8B`, fp32, 4 CPU threads, the model
card's 3-question example (276 tokens): load 8 s, forward 1.2-2.1 s (the Gated-DeltaNet kernels fall back to the
PyTorch reference implementation without `flash-linear-attention` / `causal_conv1d`; the card reports 34 ms on an
RTX 4090); answers `billing` 0.936, score 1.29 (P = 0.15 / 0.41 / 0.44), P(yes) = 0.538 -- identical (max |dp| = 0)
to the checkpoint's own compiler + `Qwen3_5ForConditionalGeneration` forward on the same request. An 8k-token
state took 57 s on CPU: use a GPU for anything beyond pilots.
