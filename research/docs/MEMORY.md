# Jev-for-memory: long-term-memory QA (LoCoMo, LongMemEval)

`jevharness.bench.suites.memory_qa` measures what a fast System-1 model adds when it decides
**what to write** to an agent's long-term memory and **what to read** back, compared with
store-everything, recency, dense and lexical retrieval, and an LLM that makes the same decisions
itself. Store, gate and reader are the harness's own `MemoryStore` (`harness/memory.py`), the same
component JevAgent uses inside episodes.

```
for each conversation (LoCoMo) / question haystack (LongMemEval):
    stream memory candidates in time order    (turn | round = user+assistant pair | session)
        -> MemoryStore.offer(candidate, context)       WRITE: always | trigger(tau) [S1 Noul] | llm [S2 yes/no]
    for each question:
        -> recall                                       READ:  see table below
        -> System-2 LLM answers from the recalled memories (dataset-native prompt)
        -> EM / F1 (LoCoMo rules) + optional LLM judge (LongMemEval prompts)
```

| `read.policy` | what the answerer sees | decision maker |
|---|---|---|
| `full_context` | the whole conversation (newest sessions kept if over budget) | none (long-context baseline) |
| `none` | nothing (closed-book lower bound) | |
| `all` | every stored memory (a write gate is the only filter) | write gate |
| `recency` (k) | the k most recent stored memories (`MemoryStore` read=always) | |
| `s1` (k) | S1 `Choice` tournament over stored memories (`MemoryStore` read=assist) | **Jev** |
| `embedding` (k) | cosine top-k (`text-embedding-3-small`, or any OpenAI-compatible server) | embedder |
| `bm25` (k) | Okapi BM25 top-k (LongMemEval's `flat-bm25` style) | lexical |
| `s1_rerank` (k) | BM25/embedding prefilter to n, then S1 picks k (one Jev call when n <= 254) | **Jev** |

Write gate (`write.mode`): `always` stores everything. `trigger` asks the S1 backend a `Noul`
("Will this information likely be needed later ...?") and stores when `p_yes >= tau`. `llm` asks
the System-2 model (or `write_llm`) yes/no.

## Data

**LoCoMo** (ACL 2024, https://github.com/snap-research/locomo). `data/locomo10.json` is fetched from
raw.githubusercontent.com at pinned commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` into
`data/raw/locomo/locomo10.json` on first use. You can also pass `dataset.path`. It has
10 conversations (19-32 sessions, 369-689 turns, 13k-27k tokens each) and 1,986 QA: 1,540 answerable
plus 446 adversarial. Images are represented by their BLIP captions, as in LoCoMo's own scripts.
Evidence is turn-level (`D<session>:<turn>`); malformed ids such as `D:11:26` and `D8:6; D9:17` are
parsed tolerantly. Categories: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial
(the name mapping used by most follow-up work: Mem0, A-Mem, Zep). Category 5 has no gold answer
and is excluded by default. `include_adversarial: true` asks it in LoCoMo's multiple-choice form
(`(a) Not mentioned in the conversation (b) <adversarial answer>`).

**LongMemEval** (ICLR 2025, https://github.com/xiaowu0162/LongMemEval) is distributed **only** via
Hugging Face (`xiaowu0162/longmemeval-cleaned`, the 2025/09 cleaned release), plus a Google Drive
mirror linked from the README. Neither is reachable from the dev container, and no full copy exists
on GitHub (only small test fixtures). Download it on a machine with HF access:

```bash
mkdir -p data/raw/longmemeval && cd data/raw/longmemeval
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
# optional, ~500 sessions per question:
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json
# or: huggingface-cli download xiaowu0162/longmemeval-cleaned --repo-type dataset --local-dir .
```

Then pick it with `dataset.variant: s | m | oracle`, or point `dataset.path` at the file. There are
500 questions, and each carries its own haystack, so a LongMemEval "conversation" here is one
question. `_S` is ~115k tokens / ~50 sessions; `oracle` contains only the evidence sessions and is
good for debugging. Evidence groups are the `has_answer` turns; an evidence session without such
turns counts as one group. The 30 `_abs` abstention questions have no evidence and are judged with
the abstention template.

**synthetic**: a deterministic toy generator for offline tests and smoke runs (`dataset.name: synthetic`).

## Run

```bash
# offline plumbing check (no keys, no downloads; accuracy meaningless)
python -m jevharness.bench.suites.memory_qa --config configs/memory/synthetic_smoke.yaml --out runs/memory

# LoCoMo: cheap dry run first, then everything (needs OPENAI_API_KEY and JEV/TypeSafe credentials)
python -m jevharness.bench.suites.memory_qa --config configs/memory/locomo.yaml --out runs/memory \
    --max-conversations 1 --max-questions 20 --arms full_context,all_embed_k10,jevwrite_jevread_k10
python -m jevharness.bench.suites.memory_qa --config configs/memory/locomo.yaml --out runs/memory

# LongMemEval (after downloading): subset first
python -m jevharness.bench.suites.memory_qa --config configs/memory/longmemeval.yaml --out runs/memory -o dataset.n=50
python -m jevharness.bench.suites.memory_qa --config configs/memory/longmemeval.yaml --out runs/memory --report-only
```

Flags: `--arms a,b`, `--max-conversations N`, `--max-questions N` (per conversation), `--no-judge`,
`--report-only`, and `-o dotted.key=value` overrides. Runs are **resumable**: questions already in
`<out>/<name>/<arm>.jsonl` without an error are skipped. The S1 cache (`s1_cache`) makes reruns of
gate/read decisions free, and replayed latencies and costs keep the reported efficiency faithful.

Config keys: `dataset`, `granularity` (`turn`, or `round` for LongMemEval, or `session`),
`write_context_turns`, `max_item_chars`, `answer` (`style: short|cot`, `prompt: locomo|longmemeval`,
`max_context_tokens`, `max_memory_tokens`, `max_tokens`), `llm`, `write_llm`, `judge`, `s1`,
`s1_cache`, `s1_backends` / `llms` (arm-level `s1:` / `llm:` / `write_llm:` bindings, as in
`bench/runner.py`), `embedder` (`{type: openai, model, base_url}` or `{type: hashing}`), `workers`,
`arms`, and `sweeps` (`{base, set: write.tau, values}`). `dataset` also takes `max_questions` (per
conversation) and `question_seed`. The seed shuffles each conversation's questions before truncation.
LoCoMo lists its questions grouped by category, so a small pilot needs the seed to see every category.

**Pilot with Gemini** (`configs/pilot/locomo_gemini.yaml`: conv-26, 40 questions, 5 arms, Flash-Lite
answerer, Flash LLM write gate and judge, Flash-Lite as the stand-in System-1; results in
[`PILOT_RESULTS.md`](PILOT_RESULTS.md)):

```bash
python -m jevharness.bench.suites.memory_qa --config configs/pilot/locomo_gemini.yaml --out runs/pilot
# arms run one after another; to parallelise, start `--arms a,b` in separate processes, then `--report-only`
```

- The LLM write gate (`MemoryStore.offer`, `write.mode: llm`) asks with `max_tokens=4`. At that budget
  Gemini 3.x returns an empty text, so every candidate was rejected: 0/419 stored in the first pilot run.
  The suite now wraps the write LLM so it gets at least `write_gate_min_tokens` (default 16; 0 turns the
  wrapper off).
- A Gemini judge works with the suite's `max_tokens: 10`: with `reasoning_effort: minimal` it answers
  in one token.
- An S1 `read` over a store-everything memory is a `Choice` tournament. The stand-in accepts at most 52
  options per question, so 419 memories take 9 chunk calls plus a final call per question: ~$0.024 and
  ~14 s per question, about 135× the BM25 arm's cost. Jev's 255-option `Choice` needs 2–3 calls. With
  the stand-in, prefer `s1_rerank` with a `bm25` prefilter of n ≤ 51 (one call).
- With the LoCoMo short-answer prompt, Flash-Lite often answers temporal questions with the session date
  or a relative phrase ("last year", "next month") instead of the resolved date. All arms are affected;
  the `cot` style is the usual remedy.

## Outputs and metrics

`<out>/<name>/`:
* `<arm>.jsonl`: one record per question: prediction, EM/F1, judge label, evidence recall,
  recalled memory ids, context tokens, the question's ledger summary (read + answer calls), and wall time.
* `<arm>.conversations.jsonl`: one record per conversation: offered/stored candidates, memory tokens,
  write-gate confusion vs. evidence, and the write-phase ledger (gate calls, embedding index).
* `summary.json`, `summary.md`: per-arm table.

| metric | definition |
|---|---|
| `accuracy` | judge accuracy if a judge ran, else mean F1 (`accuracy_metric` says which) |
| `f1`, `em` | LoCoMo `task_eval/evaluation.py`: normalisation (lowercase, strip punctuation / `a an the and` / commas), Porter stemming (needs `nltk`, else unstemmed, recorded as `stemmer`), multi-hop = mean best F1 over comma-separated sub-answers, open-domain uses the part before `;`, adversarial = prediction contains "not mentioned" / "no information available" |
| `judge_acc` | LongMemEval `evaluate_qa.py` yes/no prompts (question-type specific; LoCoMo uses the generic one). Default judge `gpt-4o-2024-08-06`, as in LongMemEval |
| `evidence_stored` | fraction of the question's evidence groups present in the stored memory (write-gate recall) |
| `evidence_recall` | fraction of evidence groups in the context given to the answerer |
| `stored_fraction`, `memory_tokens` | stored / offered candidates; estimated tokens (chars/4) of the stored memory per conversation |
| `gate_precision`, `gate_recall` | write decisions vs. "candidate contains evidence of *some* question". Precision is a lower bound, because unasked facts count as false positives |
| `cost_usd_per_q` | (write + index + read + answer cost) / #questions. The write cost is amortised; judge cost is reported separately (`judge_cost_usd`) |
| `latency_ms_per_q` | model-call latency on the question's critical path (read + answer). Write latency is reported per candidate (`write_latency_ms_per_candidate`), since writes happen online while the conversation streams |
| `s1_calls_per_q`, `llm_calls_per_q`, `write_*_calls_per_conv` | call counts from the ledgers |

**Simulated / oracle S1 backends.** `MemoryStore` doesn't know the gold labels, so the engine wraps
the backend in `GoldHintBackend`. It attaches `Question.meta["gold"]`: for writes, "the candidate
contains evidence"; for reads, "the first option holding evidence for this question, else the *none*
option". Real backends ignore `meta`, and the decision cache key excludes it, so the wrapper is
always on and never changes a real run.

## Cost estimates (per arm; gpt-4.1-mini $0.40/M in, Jev $0.042/M, text-embedding-3-small $0.02/M, gpt-4o judge $2.50/M in; verify prices in `configs/prices/models.yaml`)

**LoCoMo** (1,540 questions, 5,882 turns, ~22k tokens per conversation):

| arm | dominant cost | tokens | USD |
|---|---|---|---|
| `full_context` | answerer reads ~22k tokens per question | ~35M | ~$14 (≈$4-5 with automatic prompt caching; prefix is shared) |
| `all_embed_k10`, `all_bm25_k10` | answer ~0.6k tokens per question, embedding index ~0.3M | ~1M | ~$0.5 |
| `all_jevread_k10` | S1 tournament over ~600 memories: 3-4 Jev calls x ~19k tokens per question | ~90M S1 | ~$4 S1 + $0.5 answer |
| `jevwrite_*` | 5.9k gate calls x ~250 tokens | ~1.5M S1 | ~$0.06 for writes; reads shrink with the store |
| `llmwrite_*` | 5.9k LLM gate calls x ~250 tokens | ~1.5M | ~$0.6 (and ~5.9k sequential calls) |
| judge (any arm) | 1,540 x ~200 tokens | ~0.3M | ~$0.8 |

**LongMemEval_S** (500 questions, ~115k-token haystacks, ~275 rounds each). Start with `-o dataset.n=50`:

| arm | tokens | USD |
|---|---|---|
| `full_context` | 500 x ~115k = ~58M | ~$23 (no shared prefix, so caching doesn't help) |
| embedding index (any `embedding`/`s1_rerank` arm) | ~58M embed | ~$1.2 |
| `all_embed_k10` answering (CoT) | ~2.5M | ~$1.5 |
| `jevwrite_*` gate | ~137k calls x ~600 tokens = ~80M S1 | ~$3.5 |
| `llmwrite_*` gate | ~80M | ~$33 plus hours of latency; run on a subset |
| `all_jevread_k10` | ~2 Jev calls x ~20k tokens per question | ~$0.4 |

## Known limitations (in `harness/memory.py`, which this suite reuses unchanged)

* `MemoryStore._select` shows the S1 reader only the first 280 characters of each memory. S1
  reads therefore work best at `turn`/`round` granularity; with `session` granularity the reader
  mostly sees session headers.
* The *none* option of the read `Choice` never produces an empty recall: `_select` takes
  `top_k(k+1)` and drops *none*, so k memories come back even when *none* wins.
* `MemoryStore(max_items=...)` assigns `id=len(items)`, so once capacity evicts items, ids repeat
  and `Choice` options can collide. The suite always uses unlimited capacity.
* The write-gate question text is the agent-harness one ("... needed later to complete the task").
  The suite adds a conversational TASK line to the gate context, but doesn't change the question,
  so episode and QA results stay comparable.

## L3 smoke (2026-10-01): regression check after the harness changes

This run re-checks the memory-gate token budget after the fix moved into `harness/memory.py`
(`max_tokens=16`). The suite's `MinTokensLLM` wrapper stays as a floor. The run used the LoCoMo pilot
config with the first 10 of conv-26's shuffled questions, all 5 arms, and Gemini models
(`configs/pilot/locomo_gemini.yaml`). Outputs are in `runs/dev/l3_smoke/locomo_smoke/`. Wall clock was
437 s, most of it the Flash write gate over 419 turns.

```bash
python -m jevharness.bench.suites.memory_qa --config configs/pilot/locomo_gemini.yaml --out runs/dev/l3_smoke \
  -o name=locomo_smoke dataset.max_questions=10
```

| arm | judge acc | evidence in context | stored fraction | $/q (incl. amortised write) | write $/conv | latency ms/q |
|---|---|---|---|---|---|---|
| full_context | 3/10 | 100% | – | 0.0027 | – | 998 |
| all_bm25_k5 | 0/10 | 20% | 100% | 0.0002 | – | 572 |
| all_s1read_k5 | 4/10 | 65% | 100% | 0.0240 | – | 14,012 |
| s1write_s1read_k5 | 3/10 | 65% | 73% | 0.0256 | 0.078 (S1) | 9,870 |
| llmwrite_bm25_k5 | 2/10 | 35% | 44% | 0.0207 | 0.205 (Flash) | 551 |

- **No regression.** The LLM write gate stored 44% of the turns, the same as the fixed pilot; before the
  fix it stored 0%. The S1 gate stored 73%, also the same. 0 errors.
- The S1 arms ran over the pilot's S1 cache (`cache/s1_gemini.sqlite`). Cache hits replay the original
  cost and latency, so their $/q and latency are the pilot's per-call figures, not fresh spend.
- n = 10 says nothing about accuracy (95% CI ≈ ±30 pp).
- Full scale (E7, 10 conversations, all 1,540 QA): `-o dataset.n_conversations=10 -o dataset.max_questions=null`.
  At pilot rates the write gates cost about $0.08 (S1 stand-in) and $0.2 (Flash) per conversation, and
  the S1-read arms about $0.024 per question with the 52-option stand-in. Prefer `s1_rerank` with a BM25
  prefilter until Jev's 255-option `Choice` is available.
