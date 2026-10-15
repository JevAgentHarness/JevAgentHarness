"""Zero-cost, non-learned *control* deciders: the answer to "is the System-1 model better than a cheap
rule at the SAME invocation rate / cost?".

All three are ordinary :class:`DecisionBackend` s, so they plug into every decision point and mode:

* :class:`RateBackend` -- **matched-rate random control.** NOUL is answered "yes" with a fixed probability
  ``r`` (per decision point if ``rates`` is given); CHOICE / SCORE are uniform-random. At a trigger
  (reflection / stop / routing / memory write) it fires the expensive module on a random ``r`` fraction of
  opportunities. Set ``r`` to the trigger rate Jev showed in the same arm (:func:`trigger_rates`); the
  control then isolates *which* steps Jev picks from *how often* it fires. In guard modes ``r`` is the
  acceptance rate. ``rate: null`` gives the τ-matched variant: ``p_yes ~ U(0, 1)``, so the harness
  threshold sets the rate (fires iff ``u >= τ``, i.e. rate ``1 - τ``).
* :class:`PriorBackend` -- **always the prior's answer.** It returns the most frequent label from a fitted
  prior (``prior`` / ``prior_file``), or a fixed rule: ``first`` / ``last`` option, or ``recent`` ("call
  the most recently useful tool"). NOUL gets a constant ``p_yes``.
* :class:`HeuristicBackend` -- **hand-written rules** on the rendered state text. Reflection fires on
  *trouble*: an error in the last observation, an empty result, or a repeated action. Routing goes to the
  strong model on *trouble*. The stop trigger fires after a state-changing tool succeeded. The memory
  write gate keeps observations that carry IDs or numbers. Memory read is by recency. Tool selection is
  BM25 plus workflow rules. Tool params copy key:value pairs and typed spans.

Honesty: these backends read only the state string, the question text, the options and the decision point.
They never read ``Question.meta``, which is where gold labels live. The rules use generic English cues
(error words, state-changing verbs), not anything tuned to one suite. The randomness is a hash of
(seed, state, question), so it is reproducible, thread-safe and cache-consistent: use a different ``seed``
per trial. Every answer is logged through the ledger with cost 0 and the measured (sub-millisecond) latency.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .base import DecisionBackend
from .lexical import ABSTAIN_RE, bm25_scores, split_sections, stem, tokenize
from .types import JEV_MAX_OPTIONS, Answer, DecisionResult, QKind, Question, Usage, answer_from_distribution

# Part of every describe() and hence of the decision-cache key (CachedBackend): bump it whenever a rule changes,
# otherwise a run with ``s1_cache`` replays answers produced by the old rules.
VERSION = 3

# ------------------------------------------------------------------ decision-point inference

# Questions sent over the Jev wire (decision/server.py, harness adapters) carry no decision point, so it
# is recovered from the harness's question wording. Order matters (first match wins).
DP_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("memory_write", re.compile(r"needed later|long-term memory|worth (storing|remembering)", re.I)),
    ("memory_read", re.compile(r"stored memor|most relevant memor", re.I)),
    ("reflection", re.compile(r"\bstuck\b|repeating itself|last action fail", re.I)),
    ("stop", re.compile(r"task (actually |already )?(been )?(fully )?complete|stop now|wants to (stop|finish|submit)",
                        re.I)),
    ("routing", re.compile(r"strong model|multi-step reasoning|routine action", re.I)),
    ("tool_params", re.compile(r"argument `|which value should", re.I)),
    ("tool_select", re.compile(r"which (tool|function)|proposes to call", re.I)),
    ("action", re.compile(r"which action", re.I)),
]
STOP_GUARD_RE = re.compile(r"wants to (stop|finish|submit)|is the task actually complete", re.I)
PROPOSAL_RE = re.compile(r"proposes to call `([^`]+)` with arguments (.*)\. Is this", re.S)
ARG_RE = re.compile(r"argument `([^`]+)`(?: of tool `([^`]+)`)?[^(]*(?:\((.*)\))?", re.S)


_QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"')


def infer_dp(q: Question) -> str | None:
    """The question's decision point: ``q.decision_point`` when set (the harness always sets it), else inferred from
    the harness wording. JSON-quoted spans (a proposed final answer or call arguments, e.g. the stop guard's
    ``... submit the final answer "I am stuck; 1234.56" ...``) are ignored first, so keywords inside an answer cannot
    steer the inference; the full text is the fallback when the unquoted frame matches nothing."""
    if q.decision_point:
        return q.decision_point
    frame = _QUOTED.sub('""', q.text)
    for text in (frame, q.text) if frame != q.text else (q.text,):
        for dp, pat in DP_PATTERNS:
            if pat.search(text):
                return dp
    return None


def _u01(*parts: Any) -> float:
    """Deterministic uniform [0, 1) from a hash of ``parts``."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def ranked_distribution(options: list[str], order: list[int], top: float) -> dict[str, float]:
    """``order`` lists option indices best-first. The first gets ``top`` mass; the rest share ``1 - top``
    with linearly decreasing weights, so ``Answer.top_k`` follows ``order`` (assist mode, memory recall)."""
    n = len(options)
    top = min(max(top, 2.0 / (n + 2) + 1e-3), 1.0 - 1e-6)
    rest = order[1:]
    z = len(rest) * (len(rest) + 1) / 2 or 1.0
    dist = {options[order[0]]: top}
    for r, i in enumerate(rest):
        dist[options[i]] = (1.0 - top) * (len(rest) - r) / z
    return dist


def _usage(name: str, model: str, t0: float) -> Usage:
    return Usage(backend=name, model=model, latency_ms=(time.perf_counter() - t0) * 1000)


# ------------------------------------------------------------------ state parsing + generic signals

_HIST_LINE = re.compile(r"^\[(\d+)\] (.*?) -> (.*)$")
_TOOL_OF = re.compile(r"^\s*([\w.\-]+)\(")

ERROR_RE = re.compile(
    r"\b(error|errors|exception|traceback|failed|failure|invalid|not found|no such|unknown|denied|forbidden|"
    r"unauthori[sz]ed|cannot|can't|unable|refused|timed? ?out|irreversible|nothing happens)\b|\bno \w+ found\b",
    re.I)
UNPRODUCTIVE_RE = re.compile(r"no relevant results?|no results?\b|\b0 results\b|no matches", re.I)
ID_RE = re.compile(r"\b[A-Za-z]+[-_#]?\d[\w\-#]*\b|(?<![\w.])\d{2,}(?:\.\d+)?(?![\w])|"
                   r"\b[\w.+-]+@[\w-]+\.[\w.]+\b|\b\d{4}-\d{2}-\d{2}\b")
# first word of a tool name / a task sentence that changes the world (generic English verbs, not suite-specific)
STATE_CHANGING_VERBS = frozenset(
    "cancel refund ship send create update delete remove book reserve transfer pay submit apply add modify change "
    "exchange write edit move put place merge archive reset schedule approve assign upload insert register "
    "purchase buy rename install deploy post set grant revoke close".split())
_CHANGE_STEMS = frozenset(stem(v) for v in STATE_CHANGING_VERBS)
FINISH_RE = re.compile(r"^(finish|final_answer|submit|done|stop|end|respond|answer|complete)\b", re.I)


def is_error(obs: str | None) -> bool:
    return bool(ERROR_RE.search(obs or ""))


def is_unproductive(obs: str | None) -> bool:
    s = (obs or "").strip()
    return s in ("", "[]", "{}", "null", "None") or bool(UNPRODUCTIVE_RE.search(s))


def is_success(obs: str | None) -> bool:
    return not is_error(obs) and not is_unproductive(obs)


def tool_of(action: str) -> str:
    """``find_user({"name": ...})`` -> ``find_user``; free-form actions (ALFWorld) are returned as is."""
    m = _TOOL_OF.match(action)
    return m.group(1) if m else action.strip()


@dataclass
class ParsedState:
    text: str
    structured: bool = False          # True for harness-rendered states (TASK / HISTORY / ... sections)
    task: str = ""
    memory: str = ""
    history: list[tuple[str, str]] = field(default_factory=list)   # (action, observation), oldest first
    observation: str = ""
    candidate: str | None = None      # memory-write gate: the candidate memory


def parse_state(state: str) -> ParsedState:
    sec = split_sections(state)
    if not sec:
        return ParsedState(text=state, observation=state[-2000:])
    hist: list[tuple[str, str]] = []
    for line in sec.get("HISTORY", "").splitlines():
        m = _HIST_LINE.match(line)
        if m:
            hist.append((m.group(2).strip(), m.group(3)))
        elif hist:  # multi-line observation
            hist[-1] = (hist[-1][0], hist[-1][1] + "\n" + line)
    return ParsedState(text=state, structured=True, task=sec.get("TASK", ""), memory=sec.get("MEMORY", ""),
                       history=hist, observation=sec.get("CURRENT OBSERVATION", ""),
                       candidate=sec.get("CANDIDATE MEMORY"))


# ------------------------------------------------------------------ matched-rate protocol helpers

def _load_trajs(src: str | Path | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trajectories of a ``.jsonl``, of every ``*.jsonl`` of a run directory, or the given dicts. Files are read
    like every other analysis reads them (``harness.trajectory.load_trajectories``): the LAST record per task id,
    partial lines skipped. The resumable runner appends a fresh record when it re-runs an errored episode, so
    reading raw lines would count that task twice (and a killed run's partial last line used to crash here)."""
    if isinstance(src, (str, Path)):
        from ..harness.trajectory import load_trajectories

        p = Path(src)
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
        return [t for f in files for t in load_trajectories(f)]
    return list(src)


def trigger_rates(src: str | Path | Iterable[dict[str, Any]], by_arm: bool = False) -> dict[str, Any]:
    """Empirical per-opportunity rates of the NOUL-based meta decisions in logged episodes (a trajectory
    ``.jsonl``, a run directory, or trajectory dicts): ``{"reflection/trigger": {"n": 120, "rate": 0.23, ...}}``.
    Trigger rows give the firing rate, guard rows the acceptance rate, and routing rows the fraction of
    steps routed to the strong model. ``memory_write/<mode>`` is ``stored / offered``. Feed the rate of
    the Jev arm into ``{"type": "rate", "rate": r}`` for the matched-rate control (docs/BASELINES.md).

    **One arm at a time.** Pass the arm's own ``<run>/<arm>.jsonl``: a run directory pools every arm in it (the Jev
    arms of both System-2 models, the rate grid, the heuristic, ...), which is never the matched rate -- with a
    directory, ``by_arm=True`` returns ``{arm: rates}`` (grouped by ``traj["config"]``), and pooling several arms
    warns.

    **Opportunities, not records.** ``n`` / ``rate`` count each opportunity once (``bench.metrics.opportunity_key``):
    a stop-guard resubmission of the same answer is a retry of one proposal, so its acceptance rate is the rate per
    *distinct proposal* (the rate the matched control must reproduce); ``n_all`` / ``rate_all`` (present only when retries were dropped)
    count every record and ``n_retries`` the re-queries dropped (round-3 review: a deterministic gate re-answering an identical rejected
    finish ~25 times collapsed the pooled acceptance rate of a rate-0.5 control to 0.036)."""
    from ..bench.metrics import opportunity_key

    trajs = _load_trajs(src)
    if by_arm:
        arms: dict[str, list[dict[str, Any]]] = {}
        for t in trajs:
            arms.setdefault(str(t.get("config")), []).append(t)
        return {a: trigger_rates(ts) for a, ts in sorted(arms.items())}
    names = {str(t.get("config")) for t in trajs}
    if len(names) > 1:
        import warnings

        warnings.warn(f"trigger_rates pools {len(names)} arms ({sorted(names)[:6]}...): the matched rate is one arm's "
                      "rate -- pass <run>/<arm>.jsonl or by_arm=True", stacklevel=2)
    counts: dict[str, list[int]] = {}
    for t in trajs:
        mem_logged = False
        seen: set[tuple[Any, ...]] = set()
        for i, st in enumerate(t.get("steps", [])):
            for d in st.get("decisions", []):
                if d.get("dp") == "memory_write":
                    mem_logged = True
                    if d.get("mode") in ("always", "llm") and isinstance(d.get("final_value"), bool):
                        c = counts.setdefault(f"memory_write/{d['mode']}", [0, 0, 0, 0])
                        c[0] += 1
                        c[1] += int(d["final_value"])
                        c[2] += 1
                        c[3] += int(d["final_value"])
                        continue
                if d.get("mode") in ("trigger", "guard") and isinstance(d.get("s1_value"), bool):
                    c = counts.setdefault(f"{d['dp']}/{d['mode']}", [0, 0, 0, 0])
                    c[2] += 1
                    c[3] += int(d["s1_value"])
                    k = opportunity_key({**d, "task_id": t.get("task_id"), "step": st.get("step", i)})
                    if k is not None and k in seen:
                        continue
                    if k is not None:
                        seen.add(k)
                    c[0] += 1
                    c[1] += int(d["s1_value"])
        # older logs: memory gates logged no DecisionLog, so the stored fraction stands in (skipped when the gate
        # logged its decisions -- counted above -- and for shadow gates, which never store anything)
        mem = (t.get("meta") or {}).get("memory") or {}
        mode = (((t.get("meta") or {}).get("config") or {}).get("memory_write") or {}).get("mode")
        if mem.get("offered") and mode not in (None, "off", "shadow") and not mem_logged:
            c = counts.setdefault(f"memory_write/{mode}", [0, 0, 0, 0])
            c[0] += int(mem["offered"])
            c[1] += int(mem.get("stored", 0))
            c[2] += int(mem["offered"])
            c[3] += int(mem.get("stored", 0))
    nan = float("nan")
    out: dict[str, Any] = {}
    for k, (n, f, na, fa) in sorted(counts.items()):
        out[k] = {"n": n, "rate": (f / n if n else nan)}
        if na != n:                       # retries were dropped: report the per-record numbers next to them
            out[k].update({"n_all": na, "rate_all": (fa / na if na else nan), "n_retries": na - n})
    return out


def _label_str(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def fit_prior(labels: Iterable[Any]) -> dict[str, float]:
    """Label frequencies (most frequent first). Fit on a TRAINING split, never on the evaluation items."""
    c = Counter(_label_str(v) for v in labels if v is not None)
    z = sum(c.values()) or 1
    return {k: v / z for k, v in c.most_common()}


def fit_prior_from_trajectories(src: str | Path | Iterable[dict[str, Any]],
                                dps: Iterable[str] | None = None) -> dict[str, dict[str, float]]:
    """Per-decision-point gold-label priors from logged episodes (``DecisionLog.gold``), for
    ``{"type": "prior", "prior_file": ...}``. Use training-split episodes only (e.g. other JevGym seeds)."""
    want = set(dps) if dps else None
    by_dp: dict[str, list[Any]] = {}
    for t in _load_trajs(src):
        for st in t.get("steps", []):
            for d in st.get("decisions", []):
                if d.get("gold") is not None and (want is None or d["dp"] in want):
                    by_dp.setdefault(d["dp"], []).append(d["gold"])
    return {dp: fit_prior(v) for dp, v in sorted(by_dp.items())}


# ------------------------------------------------------------------ RateBackend

class RateBackend(DecisionBackend):
    name = "rate"
    batched_questions = True
    max_options = JEV_MAX_OPTIONS

    def __init__(self, rate: float | None = 0.5, rates: dict[str, float | None] | None = None, seed: int = 0,
                 p_fire: float = 0.999):
        """
        rate    P(yes) for NOUL questions; ``None`` -> p_yes ~ U(0, 1) (rate set by the harness τ: 1 - τ).
        rates   per-decision-point overrides, e.g. ``{"reflection": 0.1, "routing": 0.3}``.
        seed    changes every random draw (use one seed per trial).
        p_fire  reported p_yes when firing (1 - p_fire when not), so the decision is τ-independent for
                any τ in (1 - p_fire, p_fire].
        """
        for r in [rate, *(rates or {}).values()]:
            if r is not None and not 0.0 <= float(r) <= 1.0:
                raise ValueError("rates must lie in [0, 1]")
        self.rate = rate
        self.rates = dict(rates or {})
        self.seed = seed
        self.p_fire = p_fire
        self.model = "rate(u)" if rate is None else f"rate({rate})"

    def _perm(self, state: str, q: Question, n: int) -> list[int]:
        order = list(range(n))
        random.Random(int(_u01(self.seed, "perm", state, q.text, q.options) * 2 ** 53)).shuffle(order)
        return order

    def _answer(self, state: str, q: Question) -> Answer:
        labels = q.labels()
        if q.kind == QKind.NOUL:
            dp = infer_dp(q)
            r = self.rates[dp] if dp in self.rates else self.rate
            u = _u01(self.seed, "noul", dp, state, q.text)
            if r is None:
                p = min(max(u, 1e-6), 1 - 1e-6)
            else:
                p = self.p_fire if u < float(r) else 1.0 - self.p_fire
            return answer_from_distribution(q, {"yes": p, "no": 1.0 - p}, raw={"u": round(u, 6), "rate": r})
        # CHOICE / SCORE: uniform-random pick; the ranking below it is a random permutation (assist: random k)
        order = self._perm(state, q, len(labels))
        n = len(labels)
        dist = {labels[i]: 1.0 + 1e-3 * (n - rank) for rank, i in enumerate(order)}
        return answer_from_distribution(q, dist)

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        t0 = time.perf_counter()
        answers = [self._answer(state, q) for q in questions]
        return DecisionResult(answers=answers, usage=_usage(self.name, self.model, t0))

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model, "version": VERSION, "rate": self.rate, "rates": self.rates,
                "seed": self.seed, "p_fire": self.p_fire}


# ------------------------------------------------------------------ PriorBackend

def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


class PriorBackend(DecisionBackend):
    name = "prior"
    batched_questions = True
    max_options = JEV_MAX_OPTIONS
    RULES = ("majority", "first", "last", "recent")

    def __init__(self, rule: str = "majority", prior: dict[str, Any] | None = None, prior_file: str | None = None,
                 p_yes: float | dict[str, float] | None = None, fallback_rule: str = "first",
                 confidence: float = 0.99):
        """
        rule           "majority": the highest-prior option present (flat ``{label: weight}`` or per decision
                       point ``{dp: {label: weight}}``; labels match case-insensitively); options missing from
                       the prior fall back to ``fallback_rule`` order.
                       "first" / "last": positional prior (e.g. BFCL "none" and the harness OTHER are last).
                       "recent": the option most recently used successfully in the HISTORY ("always call the
                       most recently useful tool"), then the most recently used, then ``fallback_rule``.
        p_yes          constant P(yes) for NOUL (float or per decision point). Default: the prior's yes/no
                       frequency for that decision point if present, else 0.5.
        confidence     probability reported for the chosen option under position / recent rules.
        """
        if rule not in self.RULES or fallback_rule not in ("first", "last"):
            raise ValueError(f"rule must be one of {self.RULES}; fallback_rule 'first' or 'last'")
        if prior_file:
            prior = {**json.loads(Path(prior_file).read_text()), **(prior or {})}
        self.rule = rule
        self.prior = prior or {}
        self.per_dp = any(isinstance(v, dict) for v in self.prior.values())
        self.p_yes = p_yes
        self.fallback_rule = fallback_rule
        self.confidence = confidence
        self.model = f"prior({rule})"

    def _prior_for(self, dp: str | None) -> dict[str, float]:
        if self.per_dp:
            return {_norm_label(k): float(v) for k, v in (self.prior.get(dp or "", {}) or {}).items()}
        return {_norm_label(k): float(v) for k, v in self.prior.items()}

    def _base_order(self, n: int) -> list[int]:
        return list(range(n)) if self.fallback_rule == "first" else list(range(n - 1, -1, -1))

    def _choice(self, state: str, q: Question, dp: str | None) -> Answer:
        opts = q.labels()
        n = len(opts)
        base = self._base_order(n)
        if self.rule == "majority":
            pri = self._prior_for(dp)
            if pri:
                rank = {i: r for r, i in enumerate(base)}
                w = {o: pri.get(_norm_label(o), 0.0) for o in opts}
                if any(w.values()):
                    # smoothed prior over the offered options; ties broken by the fallback order
                    z = sum(w.values())
                    dist = {o: (w[o] + 1e-3 * z) * (1.0 + 1e-6 * (n - rank[i])) for i, o in enumerate(opts)}
                    return answer_from_distribution(q, dist, raw={"rule": "majority"})
            return answer_from_distribution(q, ranked_distribution(opts, base, self.confidence),
                                            raw={"rule": self.fallback_rule})
        if self.rule in ("first", "last"):
            order = list(range(n)) if self.rule == "first" else list(range(n - 1, -1, -1))
            return answer_from_distribution(q, ranked_distribution(opts, order, self.confidence),
                                            raw={"rule": self.rule})
        # recent
        st = parse_state(state)
        last_ok: dict[int, int] = {}
        last_any: dict[int, int] = {}
        low = [o.lower() for o in opts]
        for t, (act, obs) in enumerate(st.history):
            for key in (act.lower(), tool_of(act).lower()):
                if key in low:
                    i = low.index(key)
                    last_any[i] = t
                    if is_success(obs):
                        last_ok[i] = t
        rank = {i: r for r, i in enumerate(base)}
        order = sorted(range(n), key=lambda i: (-last_ok.get(i, -1), -last_any.get(i, -1), rank[i]))
        return answer_from_distribution(q, ranked_distribution(opts, order, self.confidence), raw={"rule": "recent"})

    def _noul_p(self, dp: str | None) -> float:
        if isinstance(self.p_yes, dict):
            if dp in self.p_yes:
                return float(self.p_yes[dp])
        elif self.p_yes is not None:
            return float(self.p_yes)
        pri = self._prior_for(dp)
        if "yes" in pri or "no" in pri:
            y, no = pri.get("yes", 0.0), pri.get("no", 0.0)
            return y / (y + no) if y + no > 0 else 0.5
        return 0.5

    def _answer(self, state: str, q: Question) -> Answer:
        dp = infer_dp(q)
        if q.kind == QKind.NOUL:
            p = self._noul_p(dp)
            return answer_from_distribution(q, {"yes": p, "no": 1.0 - p}, raw={"rule": "constant"})
        if q.kind == QKind.CHOICE:
            return self._choice(state, q, dp)
        labels = q.labels()
        pri = self._prior_for(dp)
        dist = {lab: pri.get(lab, 0.0) + 1e-6 for lab in labels} if any(lab in pri for lab in labels) \
            else {lab: 1.0 for lab in labels}
        return answer_from_distribution(q, dist, raw={"rule": "prior"})

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        t0 = time.perf_counter()
        answers = [self._answer(state, q) for q in questions]
        return DecisionResult(answers=answers, usage=_usage(self.name, self.model, t0))

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model, "version": VERSION, "rule": self.rule, "prior": self.prior,
                "p_yes": self.p_yes, "fallback_rule": self.fallback_rule, "confidence": self.confidence}


# ------------------------------------------------------------------ HeuristicBackend

_KIND_CUES: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"e-?mail", re.I)),
    ("path", re.compile(r"\b(path|file|dir|directory|folder)\b|_path$|_file$", re.I)),
    ("date", re.compile(r"\b(date|time|day|when)\b|_date$|_time$", re.I)),
    ("id", re.compile(r"(^|_|\b)(id|ids|uuid|code|key|sku|ref|number)$|\bid\b|Id$", re.I)),
    ("number", re.compile(r"\b(amount|price|cost|count|total|quantity|qty|num|age|limit|size|rate|value|"
                          r"number of)\b", re.I)),
    ("name", re.compile(r"\b(name|city|location|place|person|title|country|author|destination)\b", re.I)),
]
_KIND_MATCH: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"^[\w.+-]+@[\w-]+\.[\w.]+$"),
    "path": re.compile(r"^\.?/[\w./\-]+$"),
    "date": re.compile(r"^\d{4}-\d{2}-\d{2}"),
    "id": re.compile(r"^(?=.*\d)[A-Za-z0-9][\w\-#]*$"),
    "number": re.compile(r"^-?\d+(\.\d+)?$"),
    "name": re.compile(r"^[A-Z][A-Za-z'.-]*( [A-Z][A-Za-z'.-]*)*$"),
}


class HeuristicBackend(DecisionBackend):
    name = "heuristic"
    batched_questions = True
    max_options = JEV_MAX_OPTIONS

    def __init__(self, confidence: float = 0.99, state_changing: list[str] | None = None,
                 unknown_p_yes: float = 0.5, repeat_window: int = 3):
        """
        confidence      probability reported for a rule's answer (NOUL: p_yes = confidence when the rule
                        fires, 1 - confidence otherwise), so trigger/guard outcomes do not depend on τ.
        state_changing  extra tool names treated as state-changing (besides the generic verb list).
        unknown_p_yes   p_yes for NOUL questions no rule covers.
        repeat_window   an action counts as repeated if it occurs twice within the last N steps.
        """
        self.confidence = confidence
        self.state_changing = set(state_changing or [])
        self.unknown_p_yes = unknown_p_yes
        self.repeat_window = repeat_window
        self.model = "heuristic"

    # -- signals -------------------------------------------------------------------------
    def is_state_changing(self, tool: str) -> bool:
        if tool in self.state_changing:
            return True
        toks = tokenize(tool, do_stem=False, stopwords=frozenset())
        return bool(toks) and toks[0] in STATE_CHANGING_VERBS

    @staticmethod
    def task_requests_change(task: str) -> bool:
        return bool(set(tokenize(task)) & _CHANGE_STEMS)

    def _last_obs(self, st: ParsedState) -> str:
        return st.observation or (st.history[-1][1] if st.history else "")

    def repeated(self, st: ParsedState) -> bool:
        acts = [a for a, _ in st.history[-self.repeat_window:]]
        return len(acts) >= 2 and acts.count(acts[-1]) >= 2

    def trouble(self, st: ParsedState) -> bool:
        """Error or empty result in the last observation, or the last action repeated."""
        if not st.history:
            return is_error(self._last_obs(st))
        obs = self._last_obs(st)
        return is_error(obs) or is_unproductive(obs) or self.repeated(st)

    def last_change_succeeded(self, st: ParsedState) -> bool:
        if not st.history:
            return False
        act, _ = st.history[-1]
        return self.is_state_changing(tool_of(act)) and is_success(self._last_obs(st))

    def task_complete(self, st: ParsedState) -> bool:
        """Stop-guard rule. Action tasks (the instruction has a state-changing verb) are complete once a
        state-changing call succeeded. Information tasks are complete once some call returned a result."""
        if not st.structured:
            return not is_error(st.observation)
        if self.task_requests_change(st.task):
            return any(self.is_state_changing(tool_of(a)) and is_success(o) for a, o in st.history)
        return any(is_success(o) for _, o in st.history)

    # -- NOUL ----------------------------------------------------------------------------
    def _noul(self, st: ParsedState, q: Question, dp: str | None) -> tuple[bool | None, str]:
        if dp in ("reflection", "routing"):
            return self.trouble(st), "trouble"
        if dp == "memory_write":
            cand = st.candidate if st.candidate is not None else st.observation
            obs = cand.split(" -> ", 1)[1] if " -> " in cand else cand
            return bool(ID_RE.search(obs)) and is_success(obs), "ids_or_numbers"
        if dp == "stop":
            if STOP_GUARD_RE.search(q.text):
                return self.task_complete(st), "task_complete"
            return self.last_change_succeeded(st), "state_change_succeeded"
        if dp == "tool_select":  # guard: reject exact repeats of an earlier call
            m = PROPOSAL_RE.search(q.text)
            if m:
                proposed = f"{m.group(1)}({m.group(2)[:200]})"
                return not any(a == proposed for a, _ in st.history), "no_repeat"
        return None, "unknown"

    # -- CHOICE ----------------------------------------------------------------------------
    def _tool_order(self, st: ParsedState, q: Question, dp: str | None) -> list[int]:
        """BM25(task + latest observation, option + description), with workflow rules on top:
        finish-like options go first once the task looks complete and last before that. Tools that
        already succeeded (or, for discrete actions, recently taken actions) go below fresh ones.
        State-changing tools wait until some read-only call has succeeded ("look before you leap").
        Dead ends -- options that returned nothing, or errored twice -- go last."""
        opts = q.options or []
        descs = q.option_descriptions or [None] * len(opts)
        query = f"{st.task}\n{st.observation}" if st.structured else f"{st.text}\n{q.text}"
        lex = bm25_scores(tokenize(query), [tokenize(o if not d else f"{o} {d}") for o, d in zip(opts, descs)])
        done = st.structured and self.task_complete(st)
        progressed = any(is_success(o) for _, o in st.history)
        looked = any(is_success(o) and not self.is_state_changing(tool_of(a)) for a, o in st.history)
        ok_tools = {tool_of(a) for a, o in st.history if is_success(o)}
        recent = {a for a, _ in st.history[-self.repeat_window:]}
        errors = Counter(tool_of(a) for a, o in st.history if is_error(o))
        dead = {tool_of(a) for a, o in st.history if is_unproductive(o)} | {t for t, c in errors.items() if c >= 2}

        def key(i: int) -> tuple[float, ...]:
            o = opts[i]
            fin = bool(FINISH_RE.match(o))
            pri = 1 if (fin and done) else (-1 if (fin and st.structured and not progressed) else 0)
            used = (o in recent) if dp == "action" else (o in ok_tools)
            leap = dp != "action" and st.structured and not looked and not fin and self.is_state_changing(o)
            return (-pri, float(o in dead), float(used), float(leap), -float(lex[i]), float(i))

        return sorted(range(len(opts)), key=key)

    def _param_order(self, st: ParsedState, q: Question) -> list[int]:
        """Bounded values: BM25 against the task + observation. Copy candidates: the most recent value of
        a ``arg: value`` pair in the state; then spans typed like the argument (IDs for ``*_id``, numbers
        for amounts, capitalised multi-word spans for names ...), most recent first; else OTHER (defer)."""
        opts = q.options or []
        n = len(opts)
        other = [i for i, o in enumerate(opts) if o.startswith("<OTHER")]
        query = f"{st.task}\n{st.observation}" if st.structured else st.text
        if not other:
            lex = bm25_scores(tokenize(query), [tokenize(o) for o in opts])
            return sorted(range(n), key=lambda i: (-float(lex[i]), i))
        m = ARG_RE.search(q.text)
        arg = m.group(1) if m else ""
        hint = f"{arg} {m.group(3) or ''}" if m else ""
        first: list[int] = []
        if arg:
            kv = re.findall(r"[\"']?\b" + re.escape(arg) + r"\b[\"']?\s*[:=]\s*[\"']?([^\"',}\]\n]+)", st.text)
            if kv:
                val = kv[-1].strip()
                first = [i for i, o in enumerate(opts) if o == val]
        kind = next((k for k, pat in _KIND_CUES if pat.search(arg) or pat.search(hint)), None)
        cands = [i for i in range(n) if i not in other and i not in first]
        typed = [i for i in cands if kind and _KIND_MATCH[kind].match(opts[i])]
        if kind == "name":  # prefer full (multi-word) names, then recency
            typed.sort(key=lambda i: (-len(opts[i].split()), i))
        rest = [i for i in cands if i not in typed]
        return first + typed + other + rest

    @staticmethod
    def _recency_order(q: Question) -> list[int]:
        opts = q.options or []
        ids = []
        for i, o in enumerate(opts):
            m = re.match(r"\s*\[m(\d+)\]", o)
            ids.append(int(m.group(1)) if m else -1)
        return sorted(range(len(opts)), key=lambda i: (-(ids[i] if not ABSTAIN_RE.match(opts[i]) else -2), i))

    # -- dispatch --------------------------------------------------------------------------
    def _answer(self, st: ParsedState, q: Question) -> Answer:
        dp = infer_dp(q)
        if q.kind == QKind.NOUL:
            fire, rule = self._noul(st, q, dp)
            p = self.unknown_p_yes if fire is None else (self.confidence if fire else 1.0 - self.confidence)
            return answer_from_distribution(q, {"yes": p, "no": 1.0 - p}, raw={"rule": rule, "dp": dp})
        if q.kind == QKind.CHOICE:
            assert q.options is not None
            if dp == "memory_read":
                order, rule = self._recency_order(q), "recency"
            elif dp == "tool_params":
                order, rule = self._param_order(st, q), "copy"
            else:
                order, rule = self._tool_order(st, q, dp), "lexical+workflow"
            return answer_from_distribution(q, ranked_distribution(q.options, order, self.confidence),
                                            raw={"rule": rule, "dp": dp})
        labels = q.labels()
        return answer_from_distribution(q, {lab: 1.0 for lab in labels}, raw={"rule": "uniform", "dp": dp})

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        t0 = time.perf_counter()
        st = parse_state(state)
        answers = [self._answer(st, q) for q in questions]
        return DecisionResult(answers=answers, usage=_usage(self.name, self.model, t0))

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model, "version": VERSION, "confidence": self.confidence,
                "state_changing": sorted(self.state_changing), "unknown_p_yes": self.unknown_p_yes,
                "repeat_window": self.repeat_window}
