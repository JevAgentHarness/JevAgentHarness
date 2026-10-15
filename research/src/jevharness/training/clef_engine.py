"""A Clef scoring-head decision model behind the Jev wire API (``serve_jev.py --backend clef``).

Exploratory row E14 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E14: a sub-1B decision model with a scoring
head"). A Clef model = a language-model backbone plus Cloudflare's joint schema head (Apache-2.0; trained here with
Unsloth, whose decision-model code is AGPL-3.0 and is only run, never vendored). A wire request
``{"state", "questions"}`` *is* a Clef record: every question of the request is encoded with Cloudflare's record
encoding (state first, then one field per question with its allowed options) and scored in one forward pass, one logit
per allowed option. The probabilities are a softmax at the checkpoint's served temperatures (Unsloth's ``calibrate``),
returned unrounded in the Jev answer format (the same answer shapes as ``serve_jev.Engine``).

The engine is built from three callables so the wire mapping can be tested without a GPU or Unsloth:

* ``encode(state, questions) -> record`` with ``record.input_ids`` and ``record.questions[i].question_id`` /
  ``.option_ids`` (Clef's ``encode_record``; choice options sorted by label, noul as ``true`` / ``false``);
* ``forward(records) -> list of per-question logit lists`` (all questions of all records, in record order);
* ``scales(records, questions_by_record, logits) -> list of temperatures`` (one per question).

:meth:`ClefEngine.load` wires them to Unsloth's loader, record encoding and served temperatures. Requests are scored
one batch at a time under a lock (``lock_wait_ms`` / ``compute_ms`` in the usage, as ``serve_jev.Engine``).
"""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Callable, Mapping, Sequence

QTYPES = ("noul", "choice", "score")
STAT_KEYS = ("requests", "questions", "input_tokens", "batches")

# Exploratory row E18 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E17 and E18"): the deferral option. Either an
# extra criterion of every choice question scored by the head itself (arm E18-F, ``DeferOption(mode="option")``) or a
# small logistic head on the served top probability (the E18 row, ``mode="head"``). Both report ``confidence`` =
# 1 - P(defer) so the harness's cascade implements "defer iff P(defer) > k" with tau = 1 - k, and keep the option
# probabilities over the real options (renormalised in option mode); ``defer`` carries P(defer) in the wire answer.
DEFER_LABEL = "__defer__"
DEFER_DESC = "Hand this decision to the agent LLM (the decision model gives no answer)."
Q_TOOL_SELECT_TEXT = "Which tool should the agent call next?"   # harness/prompts.Q_TOOL_SELECT (kept in sync by a test)


def _softmax(z: Sequence[float], t: float) -> list[float]:
    t = float(t) if t and t > 0 else 1.0
    m = max(z)
    e = [math.exp((v - m) / t) for v in z]
    s = sum(e)
    return [v / s for v in e]


def validate_question(qid: str, q: Mapping[str, Any]) -> None:
    """The wire checks Clef's ``systemone`` applies (a 400 from the server)."""
    t = q.get("type") if isinstance(q, Mapping) else None
    if t not in QTYPES:
        raise ValueError(f"{qid}: type must be noul, choice, or score")
    if t != "noul" and not q.get("criteria"):
        raise ValueError(f"{qid}: criteria must not be empty")
    if t == "noul" and q.get("criteria") is not None:
        crit = q["criteria"]
        if not isinstance(crit, Mapping) or set(crit) - {"true", "false"}:
            raise ValueError(f'{qid}: noul criteria may only have "true" and "false"')


def clef_question(q: Mapping[str, Any]) -> dict[str, Any]:
    """A wire question as Clef encodes it: choice criteria as a dict (a list of labels becomes ``{label: None}``),
    noul criteria only when given (absent -> Clef's default true / false descriptions)."""
    out: dict[str, Any] = {"type": q["type"], "instructions": q.get("instructions")}
    crit = q.get("criteria")
    if q["type"] == "choice":
        out["criteria"] = dict.fromkeys(crit) if isinstance(crit, list) else dict(crit)
    elif q["type"] == "score":
        out["criteria"] = list(crit)
    elif crit is not None:
        out["criteria"] = dict(crit)
    return out


def finish(q: Mapping[str, Any], option_ids: Sequence[str], probs: Sequence[float]) -> dict[str, Any]:
    """Per-option probabilities (Clef's option order) -> the Jev answer of one question, unrounded. Choice
    probabilities are listed in the wire's criteria order; noul answers P(true)."""
    p = {str(o): float(v) for o, v in zip(option_ids, probs)}
    t = q["type"]
    if t == "noul":
        return {"type": "noul", "noul": p["true"]}
    if t == "choice":
        crit = q["criteria"]
        labels = [str(k) for k in (crit if isinstance(crit, list) else crit.keys())]
        best = max(labels, key=lambda k: p[k])
        return {"type": "choice", "choice": best, "confidence": p[best], "probabilities": {k: p[k] for k in labels}}
    levels = [str(i) for i in range(len(q["criteria"]))]
    best = max(levels, key=lambda k: p[k])
    return {"type": "score", "score": sum(i * p[k] for i, k in enumerate(levels)), "confidence": p[best],
            "legend": dict(zip(levels, [c if isinstance(c, str) else json.dumps(c) for c in q["criteria"]])),
            "probabilities": {k: p[k] for k in levels}}


def head_features(p_best: float, tool_select: bool, n_options: int) -> list[float]:
    """E18 head features f(x) = [1, log10(1 - p1) clipped below at -15, 1[tool choice], ln(number of options)]."""
    gap = max(0.0, 1.0 - float(p_best))
    return [1.0, max(-15.0, math.log10(gap)) if gap > 0 else -15.0, 1.0 if tool_select else 0.0,
            math.log(max(1, int(n_options)))]


class DeferOption:
    """The E18 deferral rule applied to a scored choice question (``mode`` "option": the ``__defer__`` criterion's
    probability; "head": a logistic head on the served top probability, ``weights`` over :func:`head_features`)."""

    def __init__(self, mode: str = "option", weights: Sequence[float] | None = None, label: str = DEFER_LABEL,
                 description: str = DEFER_DESC, tool_select_text: str = Q_TOOL_SELECT_TEXT):
        if mode not in ("option", "head"):
            raise ValueError(f"unknown defer mode {mode!r}")
        if mode == "head" and (weights is None or len(weights) != 4):
            raise ValueError("the head mode needs 4 weights")
        self.mode, self.weights = mode, (list(map(float, weights)) if weights else None)
        self.label, self.description, self.tool_select_text = label, description, tool_select_text

    @classmethod
    def from_file(cls, path: str) -> "DeferOption":
        d = json.loads(open(path).read())
        return cls(mode="head", weights=d["weights"])

    def describe(self) -> dict[str, Any]:
        return {"mode": self.mode, "label": self.label if self.mode == "option" else None, "weights": self.weights}

    def add_option(self, q: Mapping[str, Any]) -> dict[str, Any]:
        """Option mode: the ``__defer__`` criterion appended to a choice question (a Clef-form question)."""
        if self.mode != "option" or q.get("type") != "choice" or self.label in q["criteria"]:
            return dict(q)
        return {**q, "criteria": {**q["criteria"], self.label: self.description}}

    def p_defer(self, q: Mapping[str, Any], p: Mapping[str, float]) -> float | None:
        """P(defer) of a choice question from its scored probabilities ``p`` (Clef option id -> probability)."""
        if q.get("type") != "choice":
            return None
        if self.mode == "option":
            return float(p.get(self.label, 0.0))
        crit = q["criteria"]
        labels = [str(k) for k in (crit if isinstance(crit, list) else crit.keys()) if str(k) != self.label]
        p_best = max(p[k] for k in labels)
        f = head_features(p_best, str(q.get("instructions")) == self.tool_select_text, len(labels))
        z = sum(w * x for w, x in zip(self.weights or [], f))
        return 1.0 / (1.0 + math.exp(-z))

    def finish(self, q: Mapping[str, Any], option_ids: Sequence[str], probs: Sequence[float]) -> dict[str, Any]:
        """The Jev answer of a choice question under the rule: ``probabilities`` over the real options (renormalised
        without the option's mass in option mode), ``confidence`` = 1 - P(defer), ``defer`` = P(defer),
        ``p_best`` = the top real-option probability as the head scored it."""
        if q["type"] != "choice":
            return finish(q, option_ids, probs)
        p = {str(o): float(v) for o, v in zip(option_ids, probs)}
        pd = self.p_defer(q, p)
        real = [str(k) for k in q["criteria"] if str(k) != self.label]
        mass = sum(p[k] for k in real)
        rp = {k: (p[k] / mass if mass > 0 else 1.0 / len(real)) for k in real}
        best = max(real, key=lambda k: rp[k])
        return {"type": "choice", "choice": best, "confidence": 1.0 - float(pd), "probabilities": rp,
                "defer": float(pd), "p_best": p[best], "rule": self.mode}


class ClefEngine:
    """Serves a Clef decision model through ``serve_jev.build_app`` (``answer_all``, ``stats``, ``tok``, ``name``,
    ``describe``). ``max_batch_tokens`` bounds the padded tokens of one forward when several records are scored
    together (evaluation); a served request is one record."""

    def __init__(self, encode: Callable[[Any, dict[str, Any]], Any], forward: Callable[[list[Any]], list[list[float]]],
                 scales: Callable[[list[Any], list[dict[str, Any]], list[list[float]]], list[float]],
                 tokenizer: Any = None, name: str = "clef", info: Mapping[str, Any] | None = None,
                 max_batch_tokens: int = 32768, max_batch: int = 16, coalesce: bool = False,
                 defer: DeferOption | None = None):
        self.encode, self.forward, self.scales = encode, forward, scales
        self.coalesce = bool(coalesce)   # engine v2 (E14 Deviations "serving-engine fix"): queued requests share a pass
        self.defer = defer               # E18: the deferral rule (None = E14's plain answers)
        self._pending: list[dict[str, Any]] = []
        self._plock = threading.Lock()
        self.tok = tokenizer
        self.name = name
        self.info = dict(info or {})
        self.max_batch_tokens = int(max_batch_tokens)
        self.max_batch = int(max_batch)
        self.lock = threading.Lock()
        self.stats: dict[str, int] = {k: 0 for k in STAT_KEYS}

    # ------------------------------------------------------------------ loading (GPU host, Unsloth installed)
    @classmethod
    def load(cls, path: str, name: str = "clef", max_len: int = 16384, device: str | None = None,
             **kw: Any) -> "ClefEngine":
        """A saved Clef checkpoint (Unsloth's ``save_pretrained_merged`` folder, or an adapter folder) via Unsloth's
        loader; record encoding = Cloudflare's ``encode_record`` (vendored in Unsloth); temperatures = Unsloth's
        served temperatures from the checkpoint's decision config."""
        from unsloth import FastDecisionModel  # Unsloth first: it patches transformers

        model, processor = FastDecisionModel.from_pretrained(path, max_seq_length=max_len)
        FastDecisionModel.for_inference(model)
        return cls.from_model(model, processor, name=name, max_len=max_len, **kw)

    @classmethod
    def from_model(cls, model: Any, processor: Any, name: str = "clef", max_len: int = 16384,
                   **kw: Any) -> "ClefEngine":
        """An in-memory Unsloth Clef model (``ClefDecisionModel``) -> engine (training-time evaluation uses this)."""
        import torch
        from unsloth.models import decision as D
        from unsloth.models.clef import encode_record

        tok = getattr(processor, "tokenizer", processor)
        dev = next(model.parameters()).device
        amp = D._clef_amp_dtype(model, dev)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        def encode(state: Any, questions: dict[str, Any]) -> Any:
            return encode_record(tok, {"state": state, "questions": questions}, max_length=max_len)

        @torch.no_grad()
        def forward(records: list[Any]) -> list[list[float]]:
            n = max(len(r.input_ids) for r in records)
            ids = torch.full((len(records), n), pad, dtype=torch.long)
            mask = torch.zeros((len(records), n), dtype=torch.long)
            for i, r in enumerate(records):
                ids[i, : len(r.input_ids)] = torch.tensor(r.input_ids)
                mask[i, : len(r.input_ids)] = 1
            was = model.training
            model.eval()
            try:
                with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                    z, _ = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), records=records)
            finally:
                model.train(was)
            z = z.float().cpu()
            rows, k = [], 0
            for r in records:
                for q in r.questions:
                    rows.append(z[k, : len(q.option_ids)].tolist())
                    k += 1
            return rows

        def scales(records: list[Any], questions: list[dict[str, Any]], logits: list[list[float]]) -> list[float]:
            items, i = [], 0
            for r, qs in zip(records, questions):
                for q in r.questions:
                    items.append({"qtype": D.QUESTION_TYPES.index(qs[q.question_id]["type"])})
                    i += 1
            return [float(s) for s in D._served_temperatures(model.decision_config, [torch.tensor(z) for z in logits],
                                                               items)]

        info = {"backend": "clef", "base_model": model.decision_config.get("base_model"),
                "head": dict(getattr(model.head, "config", {}) or {}), "max_len": max_len,
                "temperature": model.decision_config.get("temperature"),
                "head_temperature": model.decision_config.get("head_temperature"), "device": str(dev)}
        return cls(encode, forward, scales, tokenizer=tok, name=name, info=info, **kw)

    # ------------------------------------------------------------------ scoring
    def describe(self) -> dict[str, Any]:
        return {"name": self.name, **self.info, "max_batch_tokens": self.max_batch_tokens, "max_batch": self.max_batch,
                "coalesce": self.coalesce, "engine": "v2" if self.coalesce else "v1",
                "defer": self.defer.describe() if self.defer else None}

    def _batches(self, encoded: list[Any]) -> list[list[int]]:
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i].input_ids))
        out: list[list[int]] = []
        cur: list[int] = []
        for i in order:
            n = len(encoded[i].input_ids)
            if cur and (len(cur) >= self.max_batch or n * (len(cur) + 1) > self.max_batch_tokens):
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
        return out

    def score(self, requests: Sequence[tuple[Any, Mapping[str, Any]]]) -> list[dict[str, tuple[list[str], list[float]]]]:
        """(state, {qid: wire question}) pairs -> per request {qid: (Clef option ids, probabilities)}."""
        for _, questions in requests:
            if not isinstance(questions, Mapping) or not questions:
                raise ValueError("at least one question is required")
            for qid, q in questions.items():
                validate_question(qid, q)
        qs = [{qid: clef_question(q) for qid, q in questions.items()} for _, questions in requests]
        if self.defer is not None and self.defer.mode == "option":
            qs = [{qid: self.defer.add_option(q) for qid, q in d.items()} for d in qs]
        encoded = [self.encode(state, q) for (state, _), q in zip(requests, qs)]
        out: list[dict[str, tuple[list[str], list[float]]]] = [{} for _ in requests]
        for b in self._batches(encoded):
            recs = [encoded[i] for i in b]
            logits = self.forward(recs)
            temps = self.scales(recs, [qs[i] for i in b], logits)
            k = 0
            for i, r in zip(b, recs):
                for q in r.questions:
                    out[i][q.question_id] = (list(q.option_ids), _softmax(logits[k], temps[k]))
                    k += 1
            self.stats["batches"] += 1
        self.stats["requests"] += len(requests)
        self.stats["questions"] += sum(len(q) for q in qs)
        self.stats["input_tokens"] += sum(len(e.input_ids) for e in encoded)
        self._last_tokens = [len(e.input_ids) for e in encoded]
        return out

    def _finish(self, q: Mapping[str, Any], option_ids: Sequence[str], probs: Sequence[float]) -> dict[str, Any]:
        return self.defer.finish(q, option_ids, probs) if self.defer is not None else finish(q, option_ids, probs)

    def answer_batch(self, requests: Sequence[tuple[Any, Mapping[str, Any]]]
                     ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        t0 = time.perf_counter()
        with self.lock:
            t1 = time.perf_counter()
            scored = self.score(requests)
            toks = list(self._last_tokens)
            t2 = time.perf_counter()
        res = []
        for (_, questions), sc, n in zip(requests, scored, toks):
            answers = {qid: self._finish(q, *sc[qid]) for qid, q in questions.items()}
            res.append((answers, {"questions": len(questions), "encoded_tokens": n, "n_passes": 1, "n_rounds": 1,
                                  "lock_wait_ms": round((t1 - t0) * 1000, 3), "compute_ms": round((t2 - t1) * 1000, 3)}))
        return res

    def answer_all(self, state: Any, questions: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """One wire request. Engine v1 scores it alone under the lock; engine v2 (``coalesce``) queues it, and whichever
        request holds the GPU lock next scores every queued request in one padded pass (at most ``max_batch``; a lone
        request is never delayed)."""
        if not self.coalesce:
            return self.answer_batch([(state, questions)])[0]
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("at least one question is required")
        for qid, q in questions.items():   # an invalid request never joins (and fails) a shared pass
            validate_question(qid, q)
        slot: dict[str, Any] = {"req": (state, questions), "done": threading.Event(), "out": None, "err": None,
                                "t0": time.perf_counter()}
        with self._plock:
            self._pending.append(slot)
        while not slot["done"].is_set():
            with self.lock:
                if slot["done"].is_set():
                    break
                with self._plock:
                    batch, self._pending = self._pending[: self.max_batch], self._pending[self.max_batch:]
                if not batch:
                    continue
                t1 = time.perf_counter()
                try:
                    scored, err = self.score([b["req"] for b in batch]), None
                    toks = list(self._last_tokens)
                except Exception as e:  # noqa: BLE001 -- handed to every request of the pass
                    scored, toks, err = None, [], e
                t2 = time.perf_counter()
                for i, b in enumerate(batch):
                    if err is not None:
                        b["err"] = err
                    else:
                        _, qs = b["req"]
                        b["out"] = ({qid: self._finish(q, *scored[i][qid]) for qid, q in qs.items()},
                                    {"questions": len(qs), "encoded_tokens": toks[i], "n_passes": 1, "n_rounds": 1,
                                     "coalesced": len(batch), "lock_wait_ms": round((t1 - b["t0"]) * 1000, 3),
                                     "compute_ms": round((t2 - t1) * 1000, 3)})
                    b["done"].set()
                self.stats["coalesced_passes"] = self.stats.get("coalesced_passes", 0) + (len(batch) > 1)
        if slot["err"] is not None:
            raise slot["err"]
        return slot["out"]

    def warmup(self, rounds: int = 1, lengths: Sequence[int] = (256, 1500, 3500)) -> dict[str, Any]:
        """Synthetic requests at load (engine v2), so kernel compilation and autotuning happen before the server
        reports healthy: one request per length, then the lengths together in one pass; ``rounds`` times."""
        t0 = time.perf_counter()
        crit = {f"tool_{k}": f"Tool number {k}." for k in range(30)}
        reqs = [(" ".join(["observation"] * n), {"q0": {"type": "choice", "instructions": "Which tool?", "criteria": crit},
                                                 "q1": {"type": "noul", "instructions": "Stop now?"}}) for n in lengths]
        saved = dict(self.stats)
        for _ in range(max(0, int(rounds))):
            for r in reqs:
                self.score([r])
            self.score(reqs)
        self.stats.update(saved)   # warm-up passes are not served requests
        return {"rounds": rounds, "lengths": list(lengths), "seconds": round(time.perf_counter() - t0, 2)}

    # ------------------------------------------------------------------ eval_decisions compatibility
    def _probs(self, items: Sequence[tuple[str, str, list[str], str]]) -> list[list[float]]:
        """``eval_decisions.evaluate``'s interface: (state, question, rendered options, kind) -> probabilities over
        the rendered options, each item one record with one question (``clef_data`` maps the rendering)."""
        from .clef_data import row_question

        reqs, maps = [], []
        for state, question, options, kind in items:
            q, keymap = row_question({"kind": kind, "question": question, "options": options})
            reqs.append((state, {"q0": q}))
            maps.append(keymap)
        out = []
        for (opts_map, sc), (_, _, options, _) in zip(zip(maps, self.score(reqs)), items):
            ids, p = sc["q0"]
            pm = dict(zip(ids, p))
            row = [pm[opts_map[o]] for o in options]
            z = sum(row)
            out.append([v / z for v in row] if z > 0 and self.defer is not None else row)
        return out
