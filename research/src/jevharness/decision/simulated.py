"""Simulated System-1 backends for offline tests and *sensitivity analysis*.

``SimulatedBackend`` answers from gold labels with a controllable accuracy and a controllable
(mis)calibration. Replaying logged trajectories with it answers questions such as "how accurate
must a System-1 model be at tool selection before replacing the LLM stops hurting success?"
without spending API budget -- one of the analysis axes of the paper.

Gold labels are supplied through ``Question.meta['gold']`` (option string, bool, or int).
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from typing import Any

from .base import DecisionBackend
from .types import DecisionResult, Question, Usage, answer_from_distribution


class SimulatedBackend(DecisionBackend):
    name = "simulated"

    def __init__(self, accuracy: float = 0.9, concentration: float = 4.0, overconfidence: float = 0.0,
                 latency_ms: float = 100.0, price_per_m_input: float = 0.0, seed: int = 0):
        """
        accuracy        expected P(argmax == gold) when a gold label is present.
        concentration   per-item confidence c ~ Beta(accuracy*kappa, (1-accuracy)*kappa); lower kappa ->
                        more spread between easy and hard items -> more useful confidence (higher AUROC).
        overconfidence  added to the reported top probability (0 = perfectly calibrated:
                        P(correct | reported conf = c) = c by construction).
        """
        self.accuracy = accuracy
        self.concentration = concentration
        self.overconfidence = overconfidence
        self.latency_ms = latency_ms
        self.price = price_per_m_input
        self.seed = seed
        self._tl = threading.local()
        self._draws: dict[str, int] = {}
        self._draw_lock = threading.Lock()
        self.rng = random.Random(seed)
        self.model = f"sim(acc={accuracy},oc={overconfidence})"

    @property
    def rng(self) -> random.Random:
        """The stream of the question being answered (see :meth:`_rng_for`); the base RNG otherwise."""
        return getattr(self._tl, "rng", None) or self._base_rng

    @rng.setter
    def rng(self, value: random.Random) -> None:
        self._base_rng = value

    def _rng_for(self, state: str, q: Question) -> random.Random:
        """An independent stream per (seed, state, question, occurrence). The runner shares one backend
        across worker threads; with a single shared RNG an answer depended on which other questions
        happened to be drawn first, so simulation results changed with thread scheduling / worker count.
        Repeating an identical question still yields independent draws (occurrence counter)."""
        gold = q.meta.get("gold")
        key = hashlib.sha256(json.dumps([self.seed, state, q.kind.value, q.text, q.options, q.scale,
                                         None if gold is None else str(gold)], default=str).encode()).hexdigest()
        with self._draw_lock:
            n = self._draws.get(key, 0)
            self._draws[key] = n + 1
        return random.Random(f"{key}:{n}")

    def _dist(self, q: Question) -> dict[str, float]:
        labels = q.labels()
        gold = q.meta.get("gold")
        if isinstance(gold, bool):
            gold = "yes" if gold else "no"
        if gold is not None:
            gold = str(gold)
        if gold not in labels:
            w = {lab: self.rng.random() for lab in labels}
            z = sum(w.values())
            return {k: v / z for k, v in w.items()}
        k = self.concentration
        acc = min(max(self.accuracy, 1e-3), 1 - 1e-3)
        c = self.rng.betavariate(acc * k, (1 - acc) * k)
        c = min(0.999, max(1.0 / len(labels), c))
        correct = self.rng.random() < c
        top = gold if correct else self.rng.choice([lab for lab in labels if lab != gold])
        p_top = min(0.999, c + self.overconfidence)
        rest = (1 - p_top) / (len(labels) - 1)
        return {lab: (p_top if lab == top else rest) for lab in labels}

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        answers = []
        for q in questions:
            self._tl.rng = self._rng_for(state, q)
            try:
                answers.append(answer_from_distribution(q, self._dist(q)))
            finally:
                self._tl.rng = None
        n_in = max(1, len(state) // 4) + sum(len(q.text) // 4 for q in questions)
        return DecisionResult(answers=answers, usage=Usage(
            backend=self.name, model=self.model, input_tokens=n_in, cost_usd=n_in * self.price / 1e6,
            latency_ms=self.latency_ms))

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "accuracy": self.accuracy, "overconfidence": self.overconfidence,
                "concentration": self.concentration, "seed": self.seed}


class RandomBackend(SimulatedBackend):
    name = "random"

    def __init__(self, seed: int = 0):
        super().__init__(accuracy=0.0, seed=seed, latency_ms=0.0)

    def _dist(self, q: Question) -> dict[str, float]:
        labels = q.labels()
        pick = self.rng.choice(labels)  # uniform, with random tie-break so argmax is a random label
        eps = 1e-3
        return {lab: (1.0 + eps if lab == pick else 1.0) / (len(labels) + eps) for lab in labels}


class OracleBackend(SimulatedBackend):
    """Upper bound: always returns the gold label with probability 1 - eps."""

    name = "oracle"

    def __init__(self):
        super().__init__(accuracy=1.0, latency_ms=0.0)

    def _dist(self, q: Question) -> dict[str, float]:
        labels = q.labels()
        gold = q.meta.get("gold")
        if isinstance(gold, bool):
            gold = "yes" if gold else "no"
        gold = str(gold) if gold is not None else labels[0]
        return {lab: (0.999 if lab == gold else 0.001 / max(1, len(labels) - 1)) for lab in labels}
