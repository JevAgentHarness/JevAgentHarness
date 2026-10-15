"""A generative LLM answering typed questions in JSON with *verbalised* probabilities.

This is the "generative LLM as System-1" baseline family: same questions, same schema, but answered
by an autoregressive model. It is the only option for providers without logprobs (current
Gemini models) and it quantifies what a non-generative decider (Jev) buys in latency/cost/calibration.

Two answer formats:
  mode="distribution"  the model states a probability for every option (verbalised distribution);
  mode="label"         the model states its answer + a confidence (classic verbalised confidence).

``batch=True`` answers all questions about one state in ONE call (mirrors Jev's multi-question
request: questions share the state and are answered independently); ``batch=False`` issues one call
per question in parallel.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..accounting import default_prices
from ..llm.base import ChatResponse, LLMClient
from .base import DecisionBackend
from .types import OPTION_LABELS, Answer, DecisionResult, QKind, Question, Usage, answer_from_distribution

SYSTEM = (
    "You are a fast decision module inside an AI agent. Read the state and answer each typed question "
    "independently. Be calibrated: probabilities must reflect how likely each option is correct. "
    "Reply with ONLY one JSON object, no prose, no code fences."
)


def _labels_for(q: Question) -> tuple[list[str], dict[str, str]]:
    """Answer keys shown to the model and the map back to our labels."""
    if q.kind == QKind.CHOICE:
        assert q.options is not None
        keys = OPTION_LABELS[: len(q.options)]
        return keys, dict(zip(keys, q.options))
    if q.kind == QKind.NOUL:
        return ["yes", "no"], {"yes": "yes", "no": "no"}
    labs = q.labels()
    return labs, {lab: lab for lab in labs}


def _render_question(qid: str, q: Question) -> str:
    if q.kind == QKind.CHOICE:
        return f"{qid} (choose one option): {q.text}\nOPTIONS:\n{q.render_options()}"
    if q.kind == QKind.NOUL:
        return f"{qid} (yes/no): {q.text}"
    lo, hi = q.scale  # type: ignore[misc]
    s = f"{qid} (score {lo}-{hi}): {q.text}"
    if q.rubric:
        s += f"\nRUBRIC: {q.rubric}"
    if q.levels:
        s += "\nLEVELS:\n" + "\n".join(f"{lo + i}: {d}" for i, d in enumerate(q.levels))
    return s


def build_prompt(state: str, qs: list[tuple[str, Question]], mode: str) -> str:
    parts = [f"STATE:\n{state}", "QUESTIONS:"]
    fmt: dict[str, Any] = {}
    for qid, q in qs:
        parts.append(_render_question(qid, q))
        keys, _ = _labels_for(q)
        if mode == "distribution":
            fmt[qid] = {k: "<prob>" for k in keys}
        else:
            fmt[qid] = {"answer": f"<one of {keys}>", "confidence": "<prob that the answer is correct>"}
    if mode == "distribution":
        instr = ("For EACH question give a probability for EVERY option key (they must sum to 1), "
                 "reflecting how likely each option is the correct answer.")
    else:
        instr = "For EACH question give your answer key and the probability that it is correct."
    parts.append(f"{instr}\nReply with JSON exactly in this shape:\n{json.dumps(fmt)}")
    return "\n\n".join(parts)


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"```(?:json)?", "", text or "")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


_QOBJ = re.compile(r'"(Q\d+)"\s*:\s*(\{[^{}]*\})')


def _salvage_questions(text: str) -> dict[str, Any]:
    """Per-question answer objects from a reply that does not parse as a whole, so one malformed answer does
    not void the other answers of a batched request (seen in the E1 dev run:
    ``{"Q1": {...}, "Q2": {"A": 0.1, "B": 0.9), "Q3": {...}}`` -> Q1 and Q3 are kept, Q2 stays unparsed)."""
    out: dict[str, Any] = {}
    for m in _QOBJ.finditer(text or ""):
        try:
            v = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        if isinstance(v, dict):
            out.setdefault(m.group(1), v)
    return out


def _to_float(x: Any) -> float | None:
    try:
        v = float(str(x).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    if v > 1.0 and v <= 100.0 and str(x).strip().endswith("%"):
        v /= 100.0
    return v if v >= 0 else None


def parse_answer(q: Question, obj: Any, mode: str) -> tuple[dict[str, float] | None, float | None]:
    """Returns (distribution over our labels or None if unparseable, verbalised confidence or None)."""
    keys, back = _labels_for(q)
    if not isinstance(obj, dict):
        return None, None
    if mode == "distribution":
        dist: dict[str, float] = {}
        for k, v in obj.items():
            key = str(k).strip()
            key = key if key in back else key.upper() if key.upper() in back else key.lower()
            f = _to_float(v)
            if key in back and f is not None:
                dist[back[key]] = dist.get(back[key], 0.0) + f
        if not dist or sum(dist.values()) <= 0:
            return None, None
        return dist, None
    ans = str(obj.get("answer", "")).strip()
    conf = _to_float(obj.get("confidence"))
    conf = 1.0 if conf is None else min(max(conf if conf <= 1 else conf / 100.0, 0.0), 1.0)
    if ans not in back:
        ans = ans.upper() if ans.upper() in back else ans.lower()
    if ans not in back and q.kind == QKind.CHOICE and q.options and ans in q.options:
        return {lab: (conf if lab == ans else (1 - conf) / max(1, len(keys) - 1)) for lab in q.labels()}, conf
    if ans not in back:
        return None, conf
    lab = back[ans]
    rest = (1.0 - conf) / max(1, len(keys) - 1)
    return {x: (conf if x == lab else rest) for x in q.labels()}, conf


class GenerativeLLMBackend(DecisionBackend):
    name = "llm_generative"
    max_options = len(OPTION_LABELS)

    def __init__(self, llm: LLMClient, mode: str = "distribution", batch: bool = True, max_workers: int = 8,
                 **chat_kw: Any):
        if mode not in ("distribution", "label"):
            raise ValueError("mode must be 'distribution' or 'label'")
        self.llm = llm
        self.model = llm.model
        self.mode = mode
        self.batch = batch
        self.batched_questions = batch
        self.max_workers = max_workers
        self.chat_kw = {"temperature": 0.0, **chat_kw}

    def _call(self, state: str, qs: list[tuple[str, Question]]) -> tuple[list[Answer], ChatResponse]:
        prompt = build_prompt(state, qs, self.mode)
        resp = self.llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                             **self.chat_kw)
        obj = _extract_json(resp.text)
        salvaged: set[str] = set()
        if any(qid not in obj for qid, _ in qs):
            extra = {k: v for k, v in _salvage_questions(resp.text).items() if k not in obj}
            salvaged = set(extra)
            obj = {**obj, **extra}
        answers = []
        for qid, q in qs:
            sub = obj.get(qid)
            if sub is None and len(qs) == 1:
                sub = obj  # model skipped the id wrapper
            dist, conf = parse_answer(q, sub, self.mode)
            parse_ok = dist is not None
            raw: dict[str, Any] = {"parse_ok": parse_ok}
            if parse_ok and qid in salvaged:
                raw["salvaged"] = True  # read from its own object in an otherwise malformed reply
            if dist is None:
                labs = q.labels()
                dist = {lab: 1.0 / len(labs) for lab in labs}
                # keep the reply so unparseable answers can be inspected (scored as uniform = first option)
                raw.update({"text": (resp.text or "")[:500], "finish_reason": resp.finish_reason})
            a = answer_from_distribution(q, dist, confidence=conf if self.mode == "label" and parse_ok else None,
                                         raw=raw)
            answers.append(a)
        return answers, resp

    def _usage(self, resps: list[ChatResponse], parallel: bool) -> Usage:
        u = Usage(backend=self.name, model=self.model, n_calls=len(resps), priced=True)
        for r in resps:
            u.priced = u.priced and default_prices().get(r.model or self.model) is not None
            u.input_tokens += r.input_tokens
            u.output_tokens += r.output_tokens
            u.cost_usd += default_prices().cost(r.model or self.model, r.input_tokens, r.output_tokens,
                                                r.cached_input_tokens)
            u.latency_ms = max(u.latency_ms, r.latency_ms) if parallel else u.latency_ms + r.latency_ms
            if r.model:
                u.model = r.model
        return u

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        tagged = [(f"Q{i + 1}", q) for i, q in enumerate(questions)]
        if self.batch:
            answers, resp = self._call(state, tagged)
            return DecisionResult(answers=answers, usage=self._usage([resp], parallel=False))
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            outs = list(ex.map(lambda t: self._call(state, [t]), tagged))
        return DecisionResult(answers=[a[0] for a, _ in outs], usage=self._usage([r for _, r in outs], parallel=True))

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model, "mode": self.mode, "batch": self.batch,
                **{k: v for k, v in self.chat_kw.items() if k in ("reasoning_effort", "temperature")}}
