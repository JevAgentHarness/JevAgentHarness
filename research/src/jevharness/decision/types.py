"""Typed decision primitives.

These mirror the three primitives exposed by TypeSafe's Jev "System One" model:

* ``Choice`` -- pick one option out of a bounded list (hosted Jev: up to 255 options per request; letter-code
  deciders read at most 52 per pass and answer larger sets with a multi-pass knockout, ``decision/tournament.py``),
* ``Score``  -- rate the state against a rubric on a bounded ordinal scale,
* ``Noul``   -- a yes/no question answered with a probability.

Every backend (Jev, OpenJev, a small LLM read through logprobs, a frontier LLM that
verbalises its answer, an embedding model, simulated oracles) consumes and produces the
same objects, so the harness can swap System-1 backends without touching agent code.
"""

from __future__ import annotations

import math
import string
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

# Letter labels A..Z then a..z -> 52 options: the single-pass limit of logprob-based deciders
# (one label = one token; same limit as openjev-server). Hosted Jev accepts up to 255 options
# per Choice and at most 10 Score levels; backends expose their own ``max_options``.
OPTION_LABELS: list[str] = list(string.ascii_uppercase) + list(string.ascii_lowercase)
MAX_OPTIONS = len(OPTION_LABELS)
JEV_MAX_OPTIONS = 255
JEV_MAX_SCORE_LEVELS = 10


class QKind(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"  # yes / no


class Question(BaseModel):
    """A single typed question about a shared state."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    kind: QKind
    text: str
    # CHOICE
    options: list[str] | None = None
    option_descriptions: list[str | None] | None = None
    # SCORE
    scale: tuple[int, int] | None = None
    rubric: str | None = None
    levels: list[str] | None = None  # optional per-level descriptions (Jev Score "criteria")
    # bookkeeping: which decision point in the harness produced this question
    decision_point: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "Question":
        if self.kind == QKind.CHOICE:
            if not self.options or len(self.options) < 2:
                raise ValueError("Choice questions need >= 2 options")
            if len(self.options) > JEV_MAX_OPTIONS:
                raise ValueError(f"Choice questions support at most {JEV_MAX_OPTIONS} options")
            if len(set(self.options)) != len(self.options):
                raise ValueError("Choice options must be unique")
            if self.option_descriptions and len(self.option_descriptions) != len(self.options):
                raise ValueError("option_descriptions must align with options")
        if self.kind == QKind.SCORE:
            if self.scale is None:
                self.scale = (1, len(self.levels)) if self.levels else (1, 5)
            lo, hi = self.scale
            if hi <= lo:
                raise ValueError("Score scale must satisfy hi > lo")
            if hi - lo + 1 > JEV_MAX_SCORE_LEVELS:
                raise ValueError(f"Score supports at most {JEV_MAX_SCORE_LEVELS} levels")
            if self.levels and len(self.levels) != hi - lo + 1:
                raise ValueError("levels must have one description per scale point")
        return self

    # --- helpers -------------------------------------------------------------------
    def labels(self) -> list[str]:
        """Answer labels a backend should produce a distribution over."""
        if self.kind == QKind.CHOICE:
            assert self.options is not None
            return list(self.options)
        if self.kind == QKind.NOUL:
            return ["yes", "no"]
        lo, hi = self.scale  # type: ignore[misc]
        return [str(v) for v in range(lo, hi + 1)]

    def render_options(self) -> str:
        assert self.options is not None
        lines = []
        for i, opt in enumerate(self.options):
            desc = ""
            if self.option_descriptions and self.option_descriptions[i]:
                desc = f" -- {self.option_descriptions[i]}"
            lines.append(f"{OPTION_LABELS[i]}. {opt}{desc}")
        return "\n".join(lines)


def Choice(text: str, options: list[str], descriptions: list[str | None] | None = None,
           decision_point: str | None = None, **meta: Any) -> Question:
    return Question(kind=QKind.CHOICE, text=text, options=options, option_descriptions=descriptions,
                    decision_point=decision_point, meta=meta)


def Score(text: str, rubric: str | None = None, scale: tuple[int, int] | None = None,
          levels: list[str] | None = None, decision_point: str | None = None, **meta: Any) -> Question:
    return Question(kind=QKind.SCORE, text=text, rubric=rubric, scale=scale, levels=levels,
                    decision_point=decision_point, meta=meta)


def Noul(text: str, decision_point: str | None = None, **meta: Any) -> Question:
    return Question(kind=QKind.NOUL, text=text, decision_point=decision_point, meta=meta)


class Answer(BaseModel):
    question_id: str
    kind: QKind
    distribution: dict[str, float]
    value: Any  # CHOICE -> option str, NOUL -> bool, SCORE -> expected score (float)
    confidence: float  # backend confidence in `value` (max prob unless backend reports its own)
    raw: dict[str, Any] | None = None

    @property
    def p_yes(self) -> float:
        return float(self.distribution.get("yes", float("nan")))

    def entropy(self) -> float:
        return -sum(p * math.log(p) for p in self.distribution.values() if p > 0)

    def top_k(self, k: int) -> list[str]:
        return [o for o, _ in sorted(self.distribution.items(), key=lambda kv: -kv[1])[:k]]


class Usage(BaseModel):
    backend: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    n_calls: int = 1
    cached: bool = False
    # False when cost_usd is 0 only because no price is known for the model (flagged as
    # ``unpriced_models`` in run summaries); None = backend did not say (treated as priced)
    priced: bool | None = None
    # A6 ruling (e), additive: transport bookkeeping (summed over the chunk requests of one decision)
    attempts: int | None = None
    retry_wait_ms: float | None = None
    request_id: str | None = None
    # Review P1-4 pass accounting, additive (None = not observable / not counted; ``decision/tournament.py`` has the
    # definitions). ``n_calls`` keeps its meaning: requests to a model service for API backends (Jev, LLM deciders,
    # embeddings), forward passes for the in-process Intern-Decision backend.
    n_passes: int | None = None   # decision-model forward passes behind the answers: counted in-process
                                  # (intern_decision) or reported by our own wire servers (serve_jev, decision/server);
                                  # None when the passes happen inside a provider (hosted Jev, LLM APIs)
    n_rounds: int | None = None   # sequential read-out rounds: 1 when every Choice fit one read, 1 + elimination
                                  # rounds of the widest Choice for a knockout over more options than one read takes
    # serving bookkeeping reported by our own wire servers (training/serve_jev.py), additive; None for hosted Jev and
    # every other backend: summed over the requests of one decision -- ``ms`` (server time incl. its queue),
    # ``lock_wait_ms`` (waiting for the engine), ``prefilled_tokens`` / ``cache_hit_tokens`` (session prefix cache)
    server: dict[str, float] | None = None


class DecisionResult(BaseModel):
    answers: list[Answer]
    usage: Usage

    def by_id(self) -> dict[str, Answer]:
        return {a.question_id: a for a in self.answers}

    def __getitem__(self, i: int) -> Answer:
        return self.answers[i]


# --- distribution utilities ---------------------------------------------------------

def normalize(dist: dict[str, float], labels: list[str], floor: float = 1e-6) -> dict[str, float]:
    """Restrict to `labels`, floor missing mass, renormalise."""
    out = {lab: max(float(dist.get(lab, 0.0)), floor) for lab in labels}
    z = sum(out.values())
    return {k: v / z for k, v in out.items()}


def answer_from_distribution(q: Question, dist: dict[str, float], confidence: float | None = None,
                             raw: dict[str, Any] | None = None) -> Answer:
    dist = normalize(dist, q.labels())
    if q.kind == QKind.CHOICE:
        value: Any = max(dist, key=dist.get)  # type: ignore[arg-type]
        conf = dist[value]
    elif q.kind == QKind.NOUL:
        value = dist["yes"] >= 0.5
        conf = max(dist["yes"], dist["no"])
    else:
        value = sum(float(k) * p for k, p in dist.items())
        conf = max(dist.values())
    return Answer(question_id=q.id, kind=q.kind, distribution=dist, value=value,
                  confidence=confidence if confidence is not None else conf, raw=raw)
