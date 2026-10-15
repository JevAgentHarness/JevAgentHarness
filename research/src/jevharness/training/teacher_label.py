"""Soft teacher targets for decision examples (recipe R3 teacher distillation, R7 small-model distillation).

Every example of the ``training/data.py`` format is asked, as a typed question about its state, to one or more
*teachers*; their option distributions are pooled, filtered by agreement, mixed with the gold label and written
back in the same training format (``train_decision.py --loss soft_ce_brier`` consumes it unchanged).

  python -m jevharness.training.teacher_label --in data/decisions/train.jsonl \\
      --out data/decisions/train_teacher.jsonl \\
      --teacher jev=jev-openrouter --teacher lite=gemini-lite \\
      --alpha 0.3 --min-agree 2 --gold-filter drop --cache cache/teacher.sqlite --workers 8

Teachers (``--teacher [NAME=]SPEC``; SPEC is an alias below or a JSON ``make_backend`` spec):
  jev-openrouter   {"type": "jev", "provider": "openrouter"}                  TypeSafe Jev (needs openrouter.ai)
  gemini-lite      {"type": "llm_generative", "llm": {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
                    "reasoning_effort": "minimal"}                           stand-in until Jev is reachable
  oracle / sim     simulated backends (tests and simulation studies only; they read the gold label)
  {"type": "jev", "base_url": "http://localhost:8081", ...}                   any served checkpoint (serve_jev.py)
  {"type": "local", "model": "Qwen/Qwen3-4B", "adapter": "checkpoints/r4", "calibration": null,
   "batch_size": 32}                                                          in-process restricted-softmax readout
                    of a trained checkpoint on the EXACT training prompt (``data.render``); used by R7 distillation.

Per example (options o_1..o_K; Noul -> yes/no, Score -> levels, everything else -> Choice):
  p_t        teacher t's distribution over the options (floored at 1e-6 and renormalised)
  p_T        pooled teacher: weighted arithmetic mean (``--pool mean``) or normalised weighted geometric mean
             (``--pool geo``, product of experts), then tempered  p_T ∝ p_T^(1/T)  (``--temperature``)
  agree      c = argmax p_T; n_agree = #teachers with argmax p_t = c. Kept iff n_agree >= min_agree and
             max p_T >= min_conf; otherwise ``--on-disagree`` drop | gold (keep the one-hot gold target)
  gold       y = argmax of the input target when it is one-hot. If c != y: ``--gold-filter`` drop | gold | keep
  target     q = alpha * onehot(y) + (1 - alpha) * p_T            (no gold -> q = p_T)
  --mc-beta  examples with counterfactual ``q_values`` instead get  q = p* ∝ p_T * exp(Q/beta)  (recipe R4 with
             pi_ref = the teacher, e.g. ``{"type": "local", ...}`` = the checkpoint being fine-tuned)

Gold labels are only ever visible to simulated teachers (oracle / simulated / random): ``Question.meta`` is not
sent on the wire, and for every other teacher it is left empty. Calls go through a per-teacher ``Ledger``
(cost / latency / tokens in the summary; cached replays are reported separately) and, with ``--cache``, through
the SQLite ``CachedBackend`` so re-runs are free. Output: the labelled JSONL + ``<out>.summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from ..accounting import CallRecord, Ledger, default_prices
from ..decision.types import Choice, Noul, Question, Score
from .data import read_jsonl, render, write_jsonl

ALIASES: dict[str, dict[str, Any]] = {
    "jev-openrouter": {"type": "jev", "provider": "openrouter"},
    "jev": {"type": "jev", "provider": "openrouter"},
    "gemini-lite": {"type": "llm_generative", "llm": {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
                    "reasoning_effort": "minimal"},
    "gemini-flash": {"type": "llm_generative", "llm": {"provider": "gemini", "model": "gemini-3.5-flash"},
                     "reasoning_effort": "minimal"},
    "oracle": {"type": "oracle"},
    "sim": {"type": "simulated", "accuracy": 0.85},
}
SIMULATED_TYPES = {"oracle", "simulated", "random"}  # the only teachers allowed to see gold labels
FLOOR = 1e-6


# =============================================================================== specs
def parse_teacher(arg: str, i: int) -> tuple[str, dict[str, Any]]:
    """``[NAME=]SPEC`` -> (name, spec). SPEC: alias or JSON."""
    name, spec_s = None, arg.strip()
    if not spec_s.startswith("{") and "=" in spec_s:
        name, spec_s = spec_s.split("=", 1)
    if spec_s in ALIASES:
        spec = json.loads(json.dumps(ALIASES[spec_s]))
        name = name or spec_s
    else:
        try:
            spec = json.loads(spec_s)
        except json.JSONDecodeError as e:
            raise ValueError(f"teacher {arg!r} is neither an alias {sorted(ALIASES)} nor JSON: {e}") from e
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"teacher spec needs a 'type': {spec_s}")
    return name or f"t{i}", spec


# =============================================================================== questions
def _yes_index(options: list[str]) -> int | None:
    lo = [o.strip().lower() for o in options]
    if len(lo) == 2 and lo[0].startswith("yes") and lo[1].startswith("no"):
        return 0
    if len(lo) == 2 and lo[1].startswith("yes") and lo[0].startswith("no"):
        return 1
    return None


def example_question(ex: dict[str, Any], gold: str | None) -> tuple[Question, list[str]]:
    """Training example -> (typed question, question label of each option in ``ex['options']`` order).
    ``gold`` goes into ``meta`` (never sent on the wire); pass None for real teachers."""
    opts, kind, dp = ex["options"], ex.get("kind", "choice"), ex.get("dp")
    meta: dict[str, Any] = {}
    if kind == "noul" and (y := _yes_index(opts)) is not None:
        if gold is not None:
            meta["gold"] = opts.index(gold) == y
        return Noul(ex["question"], decision_point=dp, **meta), ["yes" if k == y else "no" for k in range(2)]
    if kind == "score" and 2 <= len(opts) <= 10:
        if gold is not None:
            meta["gold"] = str(opts.index(gold) + 1)
        q = Score(ex["question"], levels=list(opts), decision_point=dp, **meta)
        return q, q.labels()
    if gold is not None:
        meta["gold"] = gold
    return Choice(ex["question"], list(opts), decision_point=dp, **meta), list(opts)


def gold_of(ex: dict[str, Any]) -> str | None:
    """The gold option when the input target is one-hot, else None."""
    t = ex.get("target") or {}
    if not t:
        return None
    best = max(t, key=lambda o: t[o])
    if t[best] >= 1.0 - 1e-9 and best in ex["options"]:
        return best
    return None


def _norm(p: Sequence[float]) -> list[float]:
    q = [max(float(x), FLOOR) if math.isfinite(float(x)) else FLOOR for x in p]
    z = sum(q)
    return [x / z for x in q]


# =============================================================================== teachers
class LocalTeacher:
    """In-process readout of a (trained) causal-LM decider on the exact training prompt (``data.render``)."""

    def __init__(self, spec: dict[str, Any]):
        from .serve_jev import Engine

        s = dict(spec)
        s.pop("type", None)
        self.name = s.pop("name", "local")
        self.price_model = s.pop("price_model", None) or s.get("model")
        self.engine = Engine(s.pop("model"), s.pop("adapter", None), s.pop("calibration", None), name=self.name,
                             device=s.pop("device", None), max_len=int(s.pop("max_len", 4096)),
                             batch_size=int(s.pop("batch_size", 32)))
        if s:
            raise ValueError(f"unknown keys for a local teacher: {sorted(s)}")
        self.max_options = 52

    def distributions(self, exs: list[dict[str, Any]], ledger: Ledger, chunk: int = 256
                      ) -> list[list[float] | None]:
        out: list[list[float] | None] = []
        tok = self.engine.tok
        for i in range(0, len(exs), chunk):
            part = exs[i:i + chunk]
            t0 = time.perf_counter()
            probs = self.engine._probs([(e["state"], e["question"], e["options"], e.get("kind", "choice"))
                                        for e in part])
            ms = (time.perf_counter() - t0) * 1000
            n_in = sum(len(tok(render(e)[0], add_special_tokens=False).input_ids) for e in part)
            ledger.add(CallRecord(component="s1", backend="local", model=self.name, decision_point="teacher",
                                  input_tokens=n_in, cost_usd=default_prices().cost(self.price_model, n_in, 0),
                                  latency_ms=ms, priced=default_prices().get(self.price_model) is not None,
                                  meta={"n_questions": len(part)}))
            out += [_norm(p) for p in probs]
        return out


def backend_distributions(exs: list[dict[str, Any]], backend: Any, ledger: Ledger, *, expose_gold: bool,
                          workers: int = 8, max_questions: int = 16, max_errors: int = 50
                          ) -> tuple[list[list[float] | None], Counter]:
    """Ask a ``DecisionBackend`` about every example. Examples sharing a state are asked in one request (up to
    ``max_questions``) when the backend answers questions jointly (Jev, batched verbalised LLM)."""
    reasons: Counter = Counter()
    out: list[list[float] | None] = [None] * len(exs)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(exs):
        if len(e["options"]) > backend.max_options:
            reasons["too_many_options"] += 1
            continue
        groups[e["state"]].append(i)
    per = max(1, max_questions) if getattr(backend, "batched_questions", False) else 1
    jobs = [idx[j:j + per] for idx in groups.values() for j in range(0, len(idx), per)]
    lock = threading.Lock()
    errors: list[str] = []
    abort = threading.Event()

    def run(job: list[int]) -> None:
        if abort.is_set():
            return
        qs, labs = [], []
        for i in job:
            q, lab = example_question(exs[i], gold_of(exs[i]) if expose_gold else None)
            qs.append(q)
            labs.append(lab)
        try:
            res = backend.decide(exs[job[0]]["state"], qs, ledger=ledger, decision_point="teacher")
        except Exception as e:  # noqa: BLE001 - network / parse failures are counted, not fatal
            with lock:
                errors.append(f"{type(e).__name__}: {str(e)[:300]}")
                reasons["error"] += len(job)
                ok = sum(1 for x in out if x is not None)
                if len(errors) >= max_errors or (len(errors) >= min(20, len(jobs)) and ok == 0):
                    abort.set()
            return
        for i, q, lab, a in zip(job, qs, labs, res.answers):
            d = a.distribution
            if (a.raw or {}).get("parse_ok") is False:  # verbalised teachers: unparseable -> uniform
                reasons["unparseable"] += 1
                continue
            out[i] = _norm([d.get(x, 0.0) for x in lab])

    if workers <= 1:
        for j in jobs:
            run(j)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, jobs))
    if abort.is_set():
        raise RuntimeError(f"teacher {getattr(backend, 'name', '?')} failed {len(errors)} requests; first error: "
                           f"{errors[0]}")
    if errors:
        print(f"[teacher_label] {len(errors)} failed requests; first: {errors[0]}", flush=True)
    return out, reasons


def make_teacher(spec: dict[str, Any], cache: str | None) -> Any:
    if spec["type"] == "local":
        return LocalTeacher(spec)
    from ..decision import make_backend

    return make_backend(spec, cache=cache)


# =============================================================================== pooling & targets
def pool(dists: Sequence[Sequence[float]], weights: Sequence[float], how: str = "mean",
         temperature: float = 1.0) -> list[float]:
    w = [float(x) for x in weights]
    z = sum(w) or 1.0
    k = len(dists[0])
    if how == "geo":
        logs = [sum(wi * math.log(max(d[j], FLOOR)) for wi, d in zip(w, dists)) / z for j in range(k)]
        m = max(logs)
        p = [math.exp(v - m) for v in logs]
    elif how == "mean":
        p = [sum(wi * d[j] for wi, d in zip(w, dists)) / z for j in range(k)]
    else:
        raise ValueError(f"unknown pooling {how!r}")
    if temperature != 1.0:
        m = max(p)
        p = [(x / m) ** (1.0 / temperature) for x in p]
    return _norm(p)


def _argmax(p: Sequence[float]) -> int:
    return max(range(len(p)), key=lambda j: p[j])


def label_example(ex: dict[str, Any], dists: dict[str, list[float] | None], weights: dict[str, float], *,
                  alpha: float, min_agree: int, min_conf: float, how: str, temperature: float,
                  gold_filter: str, on_disagree: str, mc_beta: float | None = None
                  ) -> tuple[dict[str, Any] | None, str]:
    """Returns (labelled example or None, outcome). Outcome in kept | mc | gold_only | drop_*.

    With ``mc_beta`` and ``q_values`` on the example, the target is the KL-regularised optimum
    p*(a) ∝ p_T(a) exp(Q(a)/beta) with the pooled teacher as pi_ref (no agreement / gold filtering)."""
    opts = ex["options"]
    got = {n: d for n, d in dists.items() if d is not None}
    y = gold_of(ex)
    yi = opts.index(y) if y is not None else None

    def gold_only(reason: str) -> tuple[dict[str, Any] | None, str]:
        if yi is None:
            return None, f"drop_{reason}"
        return ({**ex, "target": {o: float(k == yi) for k, o in enumerate(opts)},
                 "meta": {**(ex.get("meta") or {}), "label_source": "gold", "teacher_outcome": reason}}, "gold_only")

    if not got:
        return gold_only("no_teacher") if on_disagree == "gold" else (None, "drop_no_teacher")
    names = list(got)
    p = pool([got[n] for n in names], [weights.get(n, 1.0) for n in names], how, temperature)
    qv = ex.get("q_values") or {}
    if mc_beta is not None and any(qv.get(o) is not None for o in opts):
        from .data import mc_target

        ref = dict(zip(opts, p))
        tgt_d = mc_target(ref, {o: float(qv[o]) for o in opts if qv.get(o) is not None}, mc_beta)
        meta = {**(ex.get("meta") or {}), "label_source": "mc_teacher_ref", "teachers": names, "mc_beta": mc_beta,
                "pi_ref": [round(v, 5) for v in p]}
        return {**ex, "target": tgt_d, "meta": meta}, "mc"
    c = _argmax(p)
    n_agree = sum(_argmax(got[n]) == c for n in names)
    if n_agree < min(min_agree, len(names)) or len(got) < min_agree or p[c] < min_conf:
        return gold_only("disagree") if on_disagree == "gold" else (None, "drop_disagree")
    if yi is not None and c != yi:
        if gold_filter == "drop":
            return None, "drop_gold_mismatch"
        if gold_filter == "gold":
            return gold_only("gold_mismatch")
    a = alpha if yi is not None else 0.0
    tgt = [(1 - a) * p[k] + (a if k == yi else 0.0) for k in range(len(opts))]
    meta = {**(ex.get("meta") or {}), "label_source": "teacher", "teachers": names, "teacher_agree": n_agree,
            "alpha": a, "gold": y, "teacher_dists": {n: [round(v, 5) for v in got[n]] for n in names}}
    return {**ex, "target": dict(zip(opts, tgt)), "meta": meta}, "kept"


# =============================================================================== metrics
def teacher_metrics(exs: list[dict[str, Any]], dists: list[list[float] | None]) -> dict[str, Any]:
    from ..bench.metrics import ece

    conf, corr, nll = [], [], []
    for e, d in zip(exs, dists):
        y = gold_of(e)
        if d is None or y is None:
            continue
        yi = e["options"].index(y)
        k = _argmax(d)
        conf.append(d[k])
        corr.append(float(k == yi))
        nll.append(-math.log(max(d[yi], FLOOR)))
    n = len(corr)
    return {"answered": sum(d is not None for d in dists), "with_gold": n,
            "acc": sum(corr) / n if n else None, "nll": sum(nll) / n if n else None,
            "ece": ece(conf, corr) if n else None}


def ledger_summary(ledger: Ledger) -> dict[str, Any]:
    fresh = [r for r in ledger.records if not r.cached]
    return {"calls": len(ledger.records), "calls_fresh": len(fresh),
            "input_tokens": sum(r.input_tokens for r in ledger.records),
            "output_tokens": sum(r.output_tokens for r in ledger.records),
            "cost_usd": sum(r.cost_usd for r in ledger.records),  # incl. replayed (cached) cost
            "cost_usd_fresh": sum(r.cost_usd for r in fresh),     # actually spent in this run
            "latency_ms_sum": sum(r.latency_ms for r in ledger.records),
            "unpriced_models": sorted({str(r.model) for r in ledger.records if not r.priced})}


# =============================================================================== driver
def label_file(examples: list[dict[str, Any]], teachers: list[tuple[str, dict[str, Any]]], *, alpha: float = 0.3,
               min_agree: int | None = None, min_conf: float = 0.0, how: str = "mean", temperature: float = 1.0,
               gold_filter: str = "drop", on_disagree: str = "drop", weights: dict[str, float] | None = None,
               cache: str | None = None, workers: int = 8, max_questions: int = 16, max_errors: int = 50,
               mc_beta: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label ``examples`` with every teacher; returns (output examples, summary)."""
    t0 = time.perf_counter()
    exs = [e for e in examples if 2 <= len(e.get("options") or []) and e.get("state") and e.get("question")]
    all_d: dict[str, list[list[float] | None]] = {}
    summary: dict[str, Any] = {"n_in": len(examples), "n_valid": len(exs), "teachers": {}}
    for name, spec in teachers:
        ledger = Ledger()
        teacher = make_teacher(spec, cache)
        if isinstance(teacher, LocalTeacher):
            d, reasons = teacher.distributions(exs, ledger), Counter()
        else:
            d, reasons = backend_distributions(exs, teacher, ledger, expose_gold=spec["type"] in SIMULATED_TYPES,
                                               workers=workers, max_questions=max_questions, max_errors=max_errors)
        all_d[name] = d
        summary["teachers"][name] = {"spec": spec, **teacher_metrics(exs, d), "skipped": dict(reasons),
                                     "ledger": ledger_summary(ledger)}
    min_agree = len(teachers) if min_agree is None else min_agree
    out, outcomes = [], Counter()
    for i, e in enumerate(exs):
        lab, outcome = label_example(e, {n: all_d[n][i] for n in all_d}, weights or {}, alpha=alpha,
                                     min_agree=min_agree, min_conf=min_conf, how=how, temperature=temperature,
                                     gold_filter=gold_filter, on_disagree=on_disagree, mc_beta=mc_beta)
        outcomes[outcome] += 1
        if lab is not None:
            out.append(lab)
    pooled = [[e["target"][o] for o in e["options"]] if (e.get("meta") or {}).get("label_source") == "teacher"
              else None for e in out]
    summary.update({"n_out": len(out), "outcomes": dict(outcomes),
                    "config": {"alpha": alpha, "min_agree": min_agree, "min_conf": min_conf, "pool": how,
                               "temperature": temperature, "gold_filter": gold_filter, "on_disagree": on_disagree,
                               "weights": weights or {}, "mc_beta": mc_beta},
                    "target_metrics": teacher_metrics(out, pooled),
                    "cost_usd_fresh": sum(t["ledger"]["cost_usd_fresh"] for t in summary["teachers"].values()),
                    "wall_s": time.perf_counter() - t0})
    return out, summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description="Soft teacher targets for decision examples")
    ap.add_argument("--in", dest="inp", required=True, help="training-format JSONL (harvest.py output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", action="append", required=True, metavar="[NAME=]SPEC",
                    help=f"alias {sorted(ALIASES)} or JSON make_backend spec, or {{\"type\": \"local\", ...}}; repeat")
    ap.add_argument("--weights", default=None, help="comma list aligned with --teacher (pooling weights)")
    ap.add_argument("--alpha", type=float, default=0.3, help="weight of the one-hot gold label in the target")
    ap.add_argument("--min-agree", type=int, default=None, help="teachers whose argmax must equal the consensus "
                    "(default: all)")
    ap.add_argument("--min-conf", type=float, default=0.0, help="drop when the pooled max prob is below this")
    ap.add_argument("--pool", choices=["mean", "geo"], default="mean")
    ap.add_argument("--temperature", type=float, default=1.0, help="temper the pooled distribution (T<1 sharpens)")
    ap.add_argument("--gold-filter", choices=["drop", "gold", "keep"], default="drop",
                    help="consensus != gold: drop the example | keep it with the gold one-hot | keep the mixture")
    ap.add_argument("--on-disagree", choices=["drop", "gold"], default="drop")
    ap.add_argument("--mc-beta", type=float, default=None,
                    help="examples with q_values get p* ∝ p_teacher * exp(Q/beta) (KL-to-MC with the teacher as "
                         "pi_ref, e.g. a local teacher = the init checkpoint); other examples are labelled as usual")
    ap.add_argument("--cache", default="cache/teacher.sqlite", help="SQLite decision cache ('' disables)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-questions", type=int, default=16, help="questions per request for batched backends")
    ap.add_argument("--max-errors", type=int, default=50, help="abort a teacher after this many failed requests")
    ap.add_argument("--limit", type=int, default=None, help="label only the first N examples (after --shuffle)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="parse inputs/teachers, print the plan, call nothing")
    a = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()
    teachers = [parse_teacher(t, i) for i, t in enumerate(a.teacher)]
    if len({n for n, _ in teachers}) != len(teachers):
        raise SystemExit("teacher names must be unique (use NAME=SPEC)")
    weights = None
    if a.weights:
        w = [float(x) for x in a.weights.split(",")]
        if len(w) != len(teachers):
            raise SystemExit("--weights must align with --teacher")
        weights = {n: x for (n, _), x in zip(teachers, w)}
    exs = read_jsonl(a.inp)
    if a.shuffle:
        random.Random(a.seed).shuffle(exs)
    if a.limit:
        exs = exs[: a.limit]
    if a.dry_run:
        plan = {"examples": len(exs), "teachers": {n: s for n, s in teachers}, "out": a.out}
        print(json.dumps(plan, indent=1))
        return plan
    out, summary = label_file(exs, teachers, alpha=a.alpha, min_agree=a.min_agree, min_conf=a.min_conf,
                              how=a.pool, temperature=a.temperature, gold_filter=a.gold_filter,
                              on_disagree=a.on_disagree, weights=weights, cache=a.cache or None, workers=a.workers,
                              max_questions=a.max_questions, max_errors=a.max_errors, mc_beta=a.mc_beta)
    write_jsonl(out, a.out)
    Path(f"{a.out}.summary.json").write_text(json.dumps(summary, indent=1, default=str))
    brief = {k: summary[k] for k in ("n_in", "n_out", "outcomes", "cost_usd_fresh", "wall_s")}
    brief["teachers"] = {n: {k: t[k] for k in ("answered", "acc", "ece")} for n, t in summary["teachers"].items()}
    print(json.dumps(brief, indent=1, default=str))
    return summary


if __name__ == "__main__":
    main()
