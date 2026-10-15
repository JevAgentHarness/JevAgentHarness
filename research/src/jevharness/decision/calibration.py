"""Post-hoc calibration for *any* System-1 backend (API models included).

``TemperatureScaledBackend`` rescales every answer distribution as p_i^(1/T) / sum_j p_j^(1/T)
(equivalent to dividing logits by T), optionally with a separate T per question kind. Temperatures
are fitted on logged decisions that carry gold labels (JevGym trajectories, static decision records)
with ``fit_from_records`` -- used for the RQ3 "raw vs temperature-scaled" arms.

    python -m jevharness.decision.calibration fit runs/e2a/jev_ts_cascade_tau0.9.jsonl --out cal_jev.json
    # then in a config:  {type: calibrated, inner: {type: jev, provider: openrouter}, temperatures_file: cal_jev.json}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .base import DecisionBackend
from .types import DecisionResult, QKind, Question, answer_from_distribution


def scale_distribution(dist: dict[str, float], t: float) -> dict[str, float]:
    if t == 1.0:
        return dict(dist)
    logs = {k: math.log(max(v, 1e-12)) / t for k, v in dist.items()}
    m = max(logs.values())
    z = sum(math.exp(v - m) for v in logs.values())
    return {k: math.exp(v - m) / z for k, v in logs.items()}


class TemperatureScaledBackend(DecisionBackend):
    def __init__(self, inner: DecisionBackend, temperature: float = 1.0,
                 per_kind: dict[str, float] | None = None, temperatures_file: str | None = None):
        self.inner = inner
        self.name = f"{inner.name}+ts"
        self.model = inner.model
        self.batched_questions = inner.batched_questions
        self.max_options = inner.max_options
        cal = json.loads(Path(temperatures_file).read_text()) if temperatures_file else {}
        self.temperature = float(cal.get("global", temperature))
        self.per_kind = {**(cal.get("per_kind") or {}), **(per_kind or {})}

    def _t(self, q: Question) -> float:
        return float(self.per_kind.get(q.kind.value, self.temperature))

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        res = self.inner._decide(state, questions)
        answers = []
        for q, a in zip(questions, res.answers):
            dist = scale_distribution(a.distribution, self._t(q))
            answers.append(answer_from_distribution(q, dist, raw={**(a.raw or {}), "t": self._t(q)}))
        return DecisionResult(answers=answers, usage=res.usage)

    def describe(self) -> dict[str, Any]:
        return {**self.inner.describe(), "temperature": self.temperature, "per_kind": self.per_kind}


def _nll(dists: list[np.ndarray], gold: list[int], t: float) -> float:
    tot = 0.0
    for p, g in zip(dists, gold):
        z = np.log(np.clip(p, 1e-12, 1.0)) / t
        z = z - z.max()
        tot -= z[g] - math.log(np.exp(z).sum())
    return tot / max(1, len(gold))


def fit_temperature(dists: list[np.ndarray], gold: list[int], lo: float = 0.05, hi: float = 20.0) -> float:
    """Golden-section search on log T minimising NLL."""
    a, b = math.log(lo), math.log(hi)
    gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(60):
        if _nll(dists, gold, math.exp(c)) < _nll(dists, gold, math.exp(d)):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return math.exp((a + b) / 2)


def _norm(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def records_to_examples(records: Iterable[dict[str, Any]]) -> dict[str, tuple[list[np.ndarray], list[int]]]:
    """DecisionLog-like dicts (s1_dist + gold) -> per-kind (distributions, gold index)."""
    out: dict[str, tuple[list[np.ndarray], list[int]]] = {}
    for r in records:
        dist, gold = r.get("s1_dist"), r.get("gold")
        if not dist or gold is None:
            continue
        labels = list(dist)
        g = _norm(gold)
        if g not in labels:
            continue
        kind = QKind.NOUL.value if set(labels) == {"yes", "no"} else QKind.CHOICE.value
        ds, gs = out.setdefault(kind, ([], []))
        ds.append(np.array([dist[k] for k in labels], dtype=float))
        gs.append(labels.index(g))
    return out


def fit_from_records(records: Iterable[dict[str, Any]], min_per_kind: int = 50) -> dict[str, Any]:
    ex = records_to_examples(records)
    all_d = [d for ds, _ in ex.values() for d in ds]
    all_g = [g for _, gs in ex.values() for g in gs]
    cal: dict[str, Any] = {"global": fit_temperature(all_d, all_g) if all_d else 1.0, "per_kind": {},
                           "n": len(all_d)}
    for kind, (ds, gs) in ex.items():
        if len(ds) >= min_per_kind:
            cal["per_kind"][kind] = fit_temperature(ds, gs)
    return cal


def _trajectory_decisions(paths: list[str]) -> list[dict[str, Any]]:
    recs = []
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            t = json.loads(line)
            for st in t.get("steps", []):
                recs.extend(d for d in st.get("decisions", []) if d.get("s1_dist"))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit temperatures from trajectory JSONL files (decisions with gold)")
    f.add_argument("paths", nargs="+")
    f.add_argument("--out", required=True)
    f.add_argument("--min-per-kind", type=int, default=50)
    a = ap.parse_args()
    cal = fit_from_records(_trajectory_decisions(a.paths), a.min_per_kind)
    Path(a.out).write_text(json.dumps(cal, indent=1))
    print(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()
