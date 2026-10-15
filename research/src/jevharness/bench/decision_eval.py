"""Decision-level evaluation: System-1 backends vs. an LLM's native function calling on the
same static decisions (BFCL / When2Call), reporting accuracy, calibration, latency and cost.

Config (configs/decision/*.yaml):

    name: bfcl_w2c
    suites:
      - {name: bfcl, categories: [simple_python, multiple, live_simple, live_multiple, irrelevance], max_per_cat: 150}
      - {name: when2call, max_items: 600}
    backends:                       # System-1 deciders (typed questions)
      jev: {type: jev, model: jev-1.13.0}
      qwen3_4b: {type: llm_logprob, llm: {provider: vllm, model: Qwen/Qwen3-4B}}
      gemma: {type: llm_generative, llm: {...}, eval: {workers: 2, max_items: 150}}   # per-backend eval options
    fc_llms:                        # System-2 baselines using native tool calling
      gpt41mini: {provider: openai, model: gpt-4.1-mini, temperature: 0}
    fc_judge: {provider: gemini, model: gemini-3.5-flash-lite, reasoning_effort: minimal}   # When2Call FC only
    batch_by_state: true            # send all questions about one state (and decision point) in one request
    workers: 16

``eval:`` inside a backend / fc_llm spec is read here and never passed to the backend: ``workers`` (request
concurrency, e.g. for a model with a low tokens-per-minute quota) and ``max_items`` (evaluate only the first
``max_items`` items of the same stratified order, so the backend's items are a subset of everyone else's).

Item sampling (``--max-items``) is stratified: rows are shuffled within each source category and interleaved
in proportion to the category sizes, whole rows at a time (a BFCL row's tool_select item and its tool_params
items stay together). The order is a prefix order, so a smaller dev run is a subset of a larger one.

Native FC on When2Call: a tool call is "call a tool"; a text reply is classified as "ask for information",
"cannot answer" or "answer directly" by ``fc_judge`` (When2Call's LLM-as-judge protocol). The judge's cost
is reported separately (``judge_cost_usd``), not as decision cost. Without ``fc_judge`` the FC arm skips
When2Call.

Every call goes through a per-backend :class:`~jevharness.accounting.Ledger`; ``ledger.json`` holds each
backend's totals and ``fresh_cost_usd`` (requests actually sent; cached replays excluded).

Caches (resume after a crash, and let a larger run reuse a dev run's requests): System-1 answers go to
``s1_cache``; native-FC and judge responses go to ``fc_cache`` (default: ``fc.sqlite`` next to ``s1_cache``;
``fc_cache: null`` turns it off). A cache hit replays the original tokens, cost and latency, so reported
efficiency is that of the original request, and is flagged ``cached`` in the records and the ledger.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from tqdm import tqdm

from ..accounting import CallRecord, Ledger, default_prices
from ..decision import make_backend
from ..decision.base import DecisionBackend
from ..harness.tools import OTHER
from ..llm import LLMClient, make_llm
from ..llm.base import ChatResponse, ToolCall
from .metrics import auroc, brier, ece
from .suites.static import META_PRIVATE, NONE_TOOL, W2C_OPTIONS, DecisionItem, load_static, norm


def _pub_meta(it: DecisionItem) -> dict[str, Any]:
    return {k: v for k, v in it.meta.items() if k not in META_PRIVATE}


# ------------------------------------------------------------------ sampling
def stratified_order(items: list[DecisionItem], seed: int = 0) -> list[list[DecisionItem]]:
    """Rows (units) in a proportional, stratified random order: units are shuffled within their source
    (``bfcl:<category>``, ``when2call:<source>``) and interleaved by their within-source rank / size,
    so every prefix has roughly the full pool's mix of sources."""
    units: dict[tuple[str, str], list[DecisionItem]] = {}
    for it in items:
        units.setdefault((it.source, it.unit), []).append(it)
    by_src: dict[str, list[list[DecisionItem]]] = defaultdict(list)
    for (src, _), its in units.items():
        by_src[src].append(its)
    rng = random.Random(seed)
    keyed: list[tuple[float, str, int, list[DecisionItem]]] = []
    for src in sorted(by_src):
        us = by_src[src]
        rng.shuffle(us)
        keyed.extend(((i + 0.5) / len(us), src, i, u) for i, u in enumerate(us))
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    return [u for *_, u in keyed]


def stratified_sample(items: list[DecisionItem], max_items: int | None, seed: int = 0) -> list[DecisionItem]:
    """The first ``max_items`` items (whole units only) of :func:`stratified_order`."""
    if not max_items:
        return list(items)
    out: list[DecisionItem] = []
    for u in stratified_order(items, seed):
        if len(out) + len(u) > max_items:
            break
        out.extend(u)
    return out


def _split_eval(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = dict(spec)
    return spec, dict(spec.pop("eval", None) or {})


def _limit(items: list[DecisionItem], opts: dict[str, Any]) -> list[DecisionItem]:
    return stratified_sample(items, int(opts["max_items"])) if opts.get("max_items") else items


# ------------------------------------------------------------------ System-1
def _s1_records(backend: DecisionBackend, items: list[DecisionItem], batch_by_state: bool,
                workers: int, name: str, ledger: Ledger | None = None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[DecisionItem]] = defaultdict(list)
    for it in items:
        # batch per (state, decision point): a tool_params question names the gold function ("The function
        # `f` will be called..."), so sharing a request with the tool_select question of the same state
        # would leak the tool_select answer to deciders that read all questions in one prompt
        groups[(it.state, it.dp) if batch_by_state else (it.id, it.dp)].append(it)

    def run(group: list[DecisionItem]) -> list[dict[str, Any]]:
        try:
            res = backend.decide(group[0].state, [it.question for it in group], ledger=ledger)
        except Exception as e:  # record failures instead of aborting the sweep
            return [{"id": it.id, "source": it.source, "dp": it.dp, "backend": name, "kind": "s1",
                     "error": f"{type(e).__name__}: {e}"[:500]} for it in group]
        u = res.usage
        out = []
        for it, a in zip(group, res.answers):
            pred = norm(a.value)
            rec = {"id": it.id, "source": it.source, "dp": it.dp, "backend": name, "kind": "s1",
                   "pred": pred, "gold": it.gold, "correct": pred in it.gold, "conf": a.confidence,
                   "latency_ms": u.latency_ms, "request_questions": len(group),
                   "input_tokens": u.input_tokens / len(group), "output_tokens": u.output_tokens / len(group),
                   "cost_usd": u.cost_usd / len(group), "cached": bool(u.cached),
                   "priced": u.priced is not False, "model": u.model, "meta": _pub_meta(it)}
            if a.raw and a.raw.get("refusal"):
                # E15: a refused question (OpenAI Decisions API) is no answer -- never correct, counted
                rec.update({"pred": "<refusal>", "correct": False, "conf": 0.0, "refusal": True})
            elif a.raw and a.raw.get("parse_ok") is False:
                rec["parse_fail"] = True  # unparseable reply -> uniform distribution (argmax = first option)
                rec["raw_reply"] = a.raw.get("text")
            elif a.raw and a.raw.get("salvaged"):
                rec["salvaged"] = True  # parsed from its own object in an otherwise malformed batched reply
            out.append(rec)
        return out

    recs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(run, groups.values()), total=len(groups), desc=f"s1:{name}"):
            recs.extend(r)
    return recs


# ------------------------------------------------------------------ native function calling
class CachedLLM(LLMClient):
    """SQLite response cache for the native-FC baseline and the When2Call judge (the System-1 arms have
    ``s1_cache``). A hit replays the original response with its tokens and latency and is booked in the ledger
    with ``cached=True`` (so ``fresh_cost_usd`` excludes it): a crashed or extended run resumes for free and the
    reported efficiency stays that of the original request. Only successful responses are stored. The key is
    the client spec (provider, model, sampling settings), the messages, the tools and the call options."""

    def __init__(self, inner: LLMClient, spec: dict[str, Any], path: str | Path):
        super().__init__(inner.model)
        self.inner = inner
        self.provider = inner.provider
        self.spec = {k: v for k, v in spec.items() if k not in ("api_key", "eval")}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self._db.commit()
        self._lock = threading.Lock()

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse:
        return self.inner._chat(messages, tools=tools, **kw)

    def _key(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, kw: dict[str, Any]) -> str:
        body = {"llm": self.spec, "m": messages, "t": tools, "kw": kw}
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def _dump(r: ChatResponse) -> str:
        return json.dumps({"text": r.text, "tool_calls": [{"id": t.id, "name": t.name, "arguments": t.arguments,
                                                           "extra": t.extra} for t in r.tool_calls],
                           "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                           "cached_input_tokens": r.cached_input_tokens, "reasoning_tokens": r.reasoning_tokens,
                           "latency_ms": r.latency_ms, "model": r.model, "finish_reason": r.finish_reason})

    @staticmethod
    def _load(v: str) -> ChatResponse:
        d = json.loads(v)
        d["tool_calls"] = [ToolCall(**t) for t in d.get("tool_calls") or []]
        return ChatResponse(**d, raw={"cached": True})

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             ledger: Ledger | None = None, decision_point: str | None = None, component: str = "s2",
             **kw: Any) -> ChatResponse:
        key = self._key(messages, tools, kw)
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        if row:
            r = self._load(row[0])
            if ledger is not None:
                m = r.model or self.model
                price = default_prices().get(m)
                ledger.add(CallRecord(component=component, backend=self.provider, model=m,
                                      decision_point=decision_point, input_tokens=r.input_tokens,
                                      output_tokens=r.output_tokens,
                                      cost_usd=default_prices().cost(m, r.input_tokens, r.output_tokens,
                                                                     r.cached_input_tokens),
                                      latency_ms=r.latency_ms, cached=True, priced=price is not None,
                                      meta={"cached_input_tokens": r.cached_input_tokens,
                                            "reasoning_tokens": r.reasoning_tokens}))
            return r
        r = self.inner.chat(messages, tools=tools, ledger=ledger, decision_point=decision_point,
                            component=component, **kw)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (key, self._dump(r)))
            self._db.commit()
        return r


def _was_cached(r: Any) -> bool:
    return isinstance(getattr(r, "raw", None), dict) and bool(r.raw.get("cached"))


def _fc_cache_path(exp: dict[str, Any]) -> str | None:
    """``fc_cache`` of the config; absent -> ``fc.sqlite`` next to ``s1_cache`` (none without it); null -> off."""
    if "fc_cache" in exp:
        return exp["fc_cache"] or None
    return str(Path(exp["s1_cache"]).with_name("fc.sqlite")) if exp.get("s1_cache") else None


JUDGE_PROMPT = """You classify an AI assistant's reply to a user's request. The assistant had these tools:
{tools}

USER: {question}

ASSISTANT REPLY (it did not call a tool): {reply}

Which kind of reply is it?
- request_for_info: it asks the user for missing information or clarification it needs to proceed
- cannot_answer: it says it cannot help / does not have a suitable tool or ability
- direct: it answers or acts on the request directly (gives an answer, or claims to have done it)
Reply with ONLY a JSON object: {{"label": "<request_for_info|cannot_answer|direct>"}}"""


def judge_w2c_reply(judge: LLMClient, it: DecisionItem, reply: str,
                    ledger: Ledger | None = None) -> tuple[str, Any]:
    """When2Call LLM-as-judge for a text (no tool call) reply -> label in W2C_OPTIONS minus tool_call."""
    tools = "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')[:150]}"
                      for t in it.tools) or "(none)"
    prompt = JUDGE_PROMPT.format(tools=tools, question=it.meta.get("user_message", it.state), reply=reply[:2000])
    r = judge.chat([{"role": "user", "content": prompt}], temperature=0.0, ledger=ledger, component="judge")
    text = (r.text or "").lower()
    for lab in ("request_for_info", "cannot_answer", "direct"):
        if lab in text:
            return lab, r
    raise ValueError(f"judge returned no label: {r.text[:200]!r}")


def _fc_cost(r: Any, llm: LLMClient) -> tuple[float, bool]:
    m = r.model or llm.model
    return (default_prices().cost(m, r.input_tokens, r.output_tokens, r.cached_input_tokens),
            default_prices().get(m) is not None)


def _fc_messages(it: DecisionItem) -> list[dict[str, Any]]:
    if it.meta.get("messages"):
        return [dict(m) for m in it.meta["messages"]]
    return [{"role": "user", "content": it.meta.get("user_message", it.state)}]


def fc_omitted_ok(r: dict[str, Any]) -> bool:
    """A native-FC call that left an optional argument out is correct when BFCL accepts omission."""
    return (r.get("kind") == "fc_llm" and r.get("dp") == "tool_params" and r.get("pred") == ""
            and bool((r.get("meta") or {}).get("omit_ok")))


def _fc_records(llm: LLMClient, items: list[DecisionItem], workers: int, name: str,
                judge: LLMClient | None = None, ledger: Ledger | None = None,
                judge_ledger: Ledger | None = None) -> list[dict[str, Any]]:
    """Native function calling baseline.

    tool_select (BFCL): free choice among the tools (no call = NONE). When2Call: tool call = "call a tool",
    a text reply is classified by ``judge``. tool_params: the gold function is forced, ONE call per BFCL row
    yields every argument, and its cost / tokens are split over the row's argument decisions (as a batched
    System-1 request is split over its questions)."""
    groups: dict[str, list[DecisionItem]] = defaultdict(list)
    for it in items:
        groups[f"{it.unit}|params" if it.dp == "tool_params" else it.id].append(it)

    def base(it: DecisionItem) -> dict[str, Any]:
        return {"id": it.id, "source": it.source, "dp": it.dp, "backend": name, "kind": "fc_llm", "gold": it.gold,
                "meta": _pub_meta(it)}

    def run(group: list[DecisionItem]) -> list[dict[str, Any]]:
        it = group[0]
        try:
            if it.dp == "tool_params":
                return _fc_params(group)
            msgs = _fc_messages(it)
            extra: dict[str, Any] = {}
            r = llm.chat(msgs, tools=it.tools or None, temperature=0.0, ledger=ledger, decision_point=it.dp)
            if it.source.startswith("when2call"):
                if r.tool_calls:
                    label = "tool_call"
                elif judge is None:
                    return [{**base(it), "skipped": "when2call text reply needs fc_judge"}]
                else:
                    label, jr = judge_w2c_reply(judge, it, r.text, ledger=judge_ledger)
                    jc, _ = _fc_cost(jr, judge)
                    extra = {"judge_cost_usd": jc, "reply": (r.text or "")[:300]}
                pred = norm(W2C_OPTIONS[label])
                extra["fc_label"] = label
            elif r.tool_calls:
                pred = norm(it.meta.get("name_map", {}).get(r.tool_calls[0].name, r.tool_calls[0].name))
            else:
                pred = norm(NONE_TOOL)
            cost, priced = _fc_cost(r, llm)
            return [{**base(it), "pred": pred, "correct": pred in it.gold, "conf": None, "latency_ms": r.latency_ms,
                     "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "cost_usd": cost,
                     "priced": priced, "model": r.model, "cached": _was_cached(r), **extra}]
        except Exception as e:
            return [{**base(g), "error": f"{type(e).__name__}: {e}"[:500]} for g in group]

    def _fc_params(group: list[DecisionItem]) -> list[dict[str, Any]]:
        it = group[0]
        tname = it.meta["gold_tool"].replace(".", "_")
        tools = [t for t in it.tools if t["function"]["name"] == tname]
        r = llm.chat(_fc_messages(it), tools=tools, temperature=0.0, ledger=ledger, decision_point="tool_params",
                     tool_choice={"type": "function", "function": {"name": tname}})
        args = r.tool_calls[0].arguments if r.tool_calls else {}
        cost, priced = _fc_cost(r, llm)
        n = len(group)
        out = []
        for g in group:
            val = args.get(g.meta["param"])
            pred = norm(val) if val is not None else ""
            rec = {**base(g), "pred": pred, "conf": None, "latency_ms": r.latency_ms,
                   "input_tokens": r.input_tokens / n, "output_tokens": r.output_tokens / n, "cost_usd": cost / n,
                   "request_questions": n, "priced": priced, "model": r.model, "called": bool(r.tool_calls),
                   "cached": _was_cached(r)}
            rec["correct"] = pred in g.meta.get("accept", g.gold) or fc_omitted_ok(rec)
            out.append(rec)
        return out

    recs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(run, groups.values()), total=len(groups), desc=f"fc:{name}"):
            recs.extend(r)
    return recs


# ------------------------------------------------------------------ summary
def _rescore(r: dict[str, Any]) -> bool:
    """Recompute correctness with the current normaliser, so records written by older versions
    (different string normalisation) are scored identically to new ones."""
    if r.get("pred") is None:
        return bool(r.get("correct"))
    pred = norm(r["pred"])
    if r.get("kind") == "fc_llm" and r.get("dp") == "tool_params":
        if fc_omitted_ok(r):
            return True
        accept = r.get("meta", {}).get("accept") or r.get("gold") or []
    else:
        accept = r.get("gold") or []
    return pred in {norm(g) for g in accept}


def summarize(recs: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    errors: dict[tuple[str, str, str], int] = defaultdict(int)
    for r in recs:
        if "error" in r and r.get("source") and r.get("dp"):
            # failed requests are not scored; count them so S1 and FC rows over different item sets are visible
            errors[(r["backend"], r["dp"], r["source"].split(":")[0])] += 1
            if ":" in r["source"] and r["source"].split(":", 1)[1]:
                errors[(r["backend"], r["dp"], r["source"])] += 1
        if "correct" not in r:
            continue
        r = {**r, "correct": _rescore(r)}
        src = r["source"].split(":")[0]
        by[(r["backend"], r["dp"], src)].append(r)
        if ":" in r["source"] and r["source"].split(":", 1)[1]:  # per category too (e.g. bfcl:irrelevance)
            by[(r["backend"], r["dp"], r["source"])].append(r)
    out = {}
    for (b, dp, src), rs in sorted(by.items()):
        corr = [float(r["correct"]) for r in rs]
        conf = [r["conf"] for r in rs if r.get("conf") is not None]
        lat = [r["latency_ms"] for r in rs]
        row = {"n": len(rs), "acc": float(np.mean(corr)),
               "latency_p50_ms": float(np.percentile(lat, 50)), "latency_p95_ms": float(np.percentile(lat, 95)),
               "cost_per_1k_usd": 1000 * float(np.mean([r.get("cost_usd", 0.0) for r in rs])),
               "input_tokens": float(np.mean([r.get("input_tokens", 0) for r in rs]))}
        n_err = errors.get((b, dp, src), 0)
        row["n_errors"] = n_err
        row["coverage"] = len(rs) / (len(rs) + n_err)
        n_pf = sum(1 for r in rs if r.get("parse_fail"))
        if n_pf:
            row["n_parse_fail"] = n_pf  # scored as answered with a uniform distribution
        jc = [r["judge_cost_usd"] for r in rs if r.get("judge_cost_usd") is not None]
        if jc:
            row["judge_cost_per_1k_usd"] = 1000 * float(np.sum(jc)) / len(rs)  # evaluation overhead, not decision cost
        if any(r.get("priced") is False for r in rs):
            row["unpriced"] = 1  # cost_per_1k_usd counts an unknown price as 0
        if len(conf) == len(rs):
            row.update({"ece": ece(conf, corr), "brier": brier(conf, corr), "auroc": auroc(conf, corr)})
        if dp == "tool_params":
            # Fair comparison: an S1 decider may answer OTHER (defer to the LLM), a native-FC LLM must
            # always generate a value. Report accuracy on slots whose gold value is among the offered
            # candidates for both, plus the S1 fill rate / precision (answers other than OTHER).
            inc = [float(r["correct"]) for r in rs if r.get("meta", {}).get("gold_in_candidates", True)]
            row["acc_gold_in_candidates"] = float(np.mean(inc)) if inc else float("nan")
            if rs and rs[0].get("kind") == "s1":
                other = norm(OTHER)
                filled = [r for r in rs if norm(r.get("pred", "")) != other]
                row["fill_rate"] = len(filled) / len(rs)
                row["fill_precision"] = float(np.mean([float(r["correct"]) for r in filled])) if filled else float("nan")
                row["defer_correct"] = float(np.mean([norm(r["pred"]) == other for r in rs
                                                      if not r.get("meta", {}).get("gold_in_candidates", True)]
                                                     or [float("nan")]))
            elif rs and rs[0].get("kind") == "fc_llm":
                row["omit_rate"] = float(np.mean([r.get("pred") == "" for r in rs]))
        out[f"{b}|{dp}|{src}"] = row
    for (b, dp, src), k in sorted(errors.items()):
        out.setdefault(f"{b}|{dp}|{src}", {"n": 0, "acc": float("nan"), "n_errors": k, "coverage": 0.0})
    return out


def resummarize(run_dir: str | Path) -> dict[str, Any]:
    """Recompute summary.json from the per-backend record files of an existing run."""
    recs: list[dict[str, Any]] = []
    for p in sorted(Path(run_dir).glob("*.jsonl")):
        recs.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
    summary = summarize(recs)
    (Path(run_dir) / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _ledger_entry(led: Ledger) -> dict[str, Any]:
    s = led.summary()
    s["fresh_cost_usd"] = sum(r.cost_usd for r in led.records if not r.cached)
    s["cached_calls"] = sum(1 for r in led.records if r.cached)
    return s


def _record_files(exp: dict[str, Any], out: Path) -> list[Path]:
    """Record files of the backends / fc_llms in ``exp`` (stale files of removed backends are ignored)."""
    names = [f"s1_{n}.jsonl" for n in (exp.get("backends") or {})] + [f"fc_{n}.jsonl" for n in (exp.get("fc_llms") or {})]
    return [out / n for n in names if (out / n).exists()]


def parent_id(item_id: str) -> str:
    """The source item (benchmark row) of a decision record id, the same rule as ``DecisionItem.unit``: a BFCL row's
    ``<row>/tool`` and ``<row>/param/<slot>`` items share ``<row>``; any other id is its own parent."""
    if "/param/" in item_id:
        return item_id.split("/param/", 1)[0]
    return item_id[: -len("/tool")] if item_id.endswith("/tool") else item_id


def exclude_by_parent(items: Sequence[DecisionItem], exclude_ids_file: str | Path) -> list[DecisionItem]:
    """PREREGISTRATION A6.16 (l): exclusion is applied by parent (source) item -- every decision record of a row that
    has any record in the exclusion list leaves the population, so a row exposed through one of its items is out
    entirely."""
    excl = json.loads(Path(exclude_ids_file).read_text())["ids"]
    parents = {parent_id(i) for i in excl}
    return [it for it in items if it.unit not in parents]


def load_population(exp: Mapping[str, Any]) -> list[DecisionItem]:
    """The items of ``exp``'s suites after the registered exclusion (before any ``max_items`` sampling)."""
    items: list[DecisionItem] = []
    for s in exp["suites"]:
        s = dict(s)
        items.extend(load_static(s.pop("name"), **s))
    if exp.get("exclude_ids_file"):
        items = exclude_by_parent(items, exp["exclude_ids_file"])
    return items


def population_counts(items: Sequence[DecisionItem]) -> dict[str, Any]:
    """Per-source and per-(source, dp) item counts plus the sha256 of the sorted id list (the C8 population file)."""
    by_source: dict[str, int] = {}
    by_source_dp: dict[str, int] = {}
    parents: set[str] = set()
    for it in items:
        by_source[it.source] = by_source.get(it.source, 0) + 1
        by_source_dp[f"{it.source}|{it.dp}"] = by_source_dp.get(f"{it.source}|{it.dp}", 0) + 1
        parents.add(it.unit)
    ids = sorted(it.id for it in items)
    return {"n_items": len(ids), "n_source_items": len(parents), "by_source": dict(sorted(by_source.items())),
            "by_source_dp": dict(sorted(by_source_dp.items())),
            "sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}


def run_decision_eval(exp: dict[str, Any], out_dir: str | Path, max_items: int | None = None,
                      only: list[str] | None = None) -> dict[str, Any]:
    """Run every backend / fc_llm of ``exp`` (or just those named in ``only``, keeping the other backends'
    record files from an earlier run of the same experiment) and write the records, summary.json and
    ledger.json to ``<out_dir>/<name>``."""
    out = Path(out_dir) / exp.get("name", "decision")
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))  # provenance (paper_numbers)
    items: list[DecisionItem] = []
    for s in exp["suites"]:
        s = dict(s)
        items.extend(load_static(s.pop("name"), **s))
    if exp.get("exclude_ids_file"):  # PREREGISTRATION A6.13 / A6.16 (l): the exposed dev-prefix items leave the C8 population
        before = len(items)
        items = exclude_by_parent(items, exp["exclude_ids_file"])
        print(f"[decision_eval] excluded {before - len(items)} item(s) whose source item is listed in "
              f"{exp['exclude_ids_file']} ({len(items)} remain)", file=sys.stderr)
    items = stratified_sample(items, max_items)
    workers = int(exp.get("workers", 8))
    ledgers: dict[str, Ledger] = {}
    for name, spec in (exp.get("backends") or {}).items():
        if only and name not in only:
            continue
        spec, opts = _split_eval(spec)
        b = make_backend(spec, cache=exp.get("s1_cache"))
        ledgers[name] = Ledger()
        recs = _s1_records(b, _limit(items, opts), bool(exp.get("batch_by_state", True)),
                           int(opts.get("workers", workers)), name, ledger=ledgers[name])
        (out / f"s1_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    fc_cache = _fc_cache_path(exp)

    def _llm(spec: dict[str, Any]) -> LLMClient:
        llm = make_llm(spec)
        return CachedLLM(llm, spec, fc_cache) if fc_cache else llm

    judge = _llm(exp["fc_judge"]) if exp.get("fc_judge") else None
    for name, spec in (exp.get("fc_llms") or {}).items():
        if only and name not in only:
            continue
        spec, opts = _split_eval(spec)
        ledgers[name] = Ledger()
        ledgers[f"{name}.judge"] = Ledger()
        recs = _fc_records(_llm(spec), _limit(items, opts), int(opts.get("workers", workers)), name,
                           judge=judge, ledger=ledgers[name], judge_ledger=ledgers[f"{name}.judge"])
        (out / f"fc_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    prev: dict[str, Any] = {}
    if only and (out / "ledger.json").exists():  # keep the other backends' ledger entries
        prev = json.loads((out / "ledger.json").read_text()).get("backends", {})
    entries = {**prev, **{k: _ledger_entry(v) for k, v in ledgers.items()}}
    this_run = sum(_ledger_entry(v)["fresh_cost_usd"] for v in ledgers.values())
    (out / "ledger.json").write_text(json.dumps(
        {"fresh_cost_usd": sum(e.get("fresh_cost_usd", 0.0) for e in entries.values()), "backends": entries},
        indent=1))
    print(f"[decide] {len(items)} items; API spend of this invocation (fresh requests): ${this_run:.4f}",
          file=sys.stderr)
    recs_all: list[dict[str, Any]] = []
    for p in _record_files(exp, out):
        recs_all.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
    summary = summarize(recs_all)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> None:
    """``python -m jevharness.bench.decision_eval CONFIG [--out runs] [--max-items N] [--only a,b]``
    (``jevbench decide`` without ``--only``: re-run some backends, keep the others' records)."""
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated backend / fc_llm names to (re-)run")
    a = ap.parse_args(argv)
    exp = yaml.safe_load(Path(a.config).read_text())
    res = run_decision_eval(exp, a.out, max_items=a.max_items, only=a.only.split(",") if a.only else None)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
