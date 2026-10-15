"""Review P1-4: Choices over more options than one read takes are multi-pass knockouts. Every backend records the
passes / rounds it actually used, and the reads are composed into ONE distribution over the full option set
(``decision/tournament.py``). Offline: Luce-consistent fake readouts (a softmax over fixed per-option strengths), for
which the linked full-set distribution must equal the single-read softmax exactly, run through the real knockout code
of ``intern_decision.py`` and ``training/serve_jev.py``, the harness tournament and the Jev wire client."""

from __future__ import annotations

import math
import random
import threading

import pytest

from jevharness.accounting import Ledger
from jevharness.decision import CachedBackend, Choice, Noul
from jevharness.decision.base import DecisionBackend
from jevharness.decision.intern_decision import InternDecisionBackend
from jevharness.decision.tournament import decision_pass_counts, full_set_distribution, knockout_plan
from jevharness.decision.types import MAX_OPTIONS, DecisionResult, QKind, Usage, answer_from_distribution

SIZES = (10, 52, 53, 200)


def strengths(n: int, seed: int = 0) -> dict[str, float]:
    rng = random.Random(f"w{n}-{seed}")
    return {f"opt_{i:03d}": math.exp(rng.gauss(0.0, 1.0)) for i in range(n)}


def truth(w: dict[str, float]) -> dict[str, float]:
    z = sum(w.values())
    return {o: v / z for o, v in w.items()}


def luce_read(w: dict[str, float], options: list[str]) -> dict[str, float]:
    z = sum(w[o] for o in options)
    return {o: w[o] / z for o in options}


# ------------------------------------------------------------------ the schedule
@pytest.mark.parametrize("n,rounds,reads,finalists", [(10, 1, 1, 10), (52, 1, 1, 52), (53, 2, 2, 3), (69, 2, 3, 4),
                                                      (200, 2, 5, 8), (255, 2, 6, 10)])
def test_knockout_plan_width_52(n, rounds, reads, finalists):
    p = knockout_plan(n, MAX_OPTIONS)
    assert (p["rounds"], p["reads"], p["finalists"]) == (rounds, reads, finalists)
    if n > MAX_OPTIONS:
        assert sum(p["chunks"][0]) == n and max(p["chunks"][0]) == MAX_OPTIONS


def test_knockout_plan_other_widths_and_the_harness_rule():
    assert knockout_plan(69, 62) == {"rounds": 2, "reads": 3, "chunks": [[62, 7]], "finalists": 4}   # Intern, JevGym
    assert knockout_plan(1353, 52)["rounds"] == 3
    # the harness reads a chunk only when it holds more than `keep` options: 54 = 52 + 2 -> one chunk read
    assert knockout_plan(54, 52, keep=2, read_if_more_than=2)["reads"] == 2
    assert knockout_plan(54, 52)["reads"] == 3


# ------------------------------------------------------------------ the composition
def _simulate(w: dict[str, float], width: int, keep: int = 2):
    """A knockout over Luce reads: (rounds as full_set_distribution takes them, final read)."""
    pool, rounds = list(w), []
    while len(pool) > width:
        rnd, nxt = [], []
        for a in range(0, len(pool), width):
            chunk = pool[a:a + width]
            if len(chunk) == 1:
                nxt += chunk
                continue
            p = luce_read(w, chunk)
            surv = sorted(chunk, key=lambda o: -p[o])[:keep]
            rnd.append((chunk, p, surv))
            nxt += surv
        rounds.append(rnd)
        pool = nxt
    return rounds, luce_read(w, pool)


@pytest.mark.parametrize("n", SIZES + (255,))
def test_linked_composition_is_exact_for_luce_reads(n):
    w = strengths(n)
    rounds, final = _simulate(w, MAX_OPTIONS)
    full = full_set_distribution(list(w), rounds, final)
    assert full == pytest.approx(truth(w), abs=1e-12)
    legacy = full_set_distribution(list(w), rounds, final, method="finalists")
    assert sum(legacy.values()) == pytest.approx(1.0) and set(o for o, v in legacy.items() if v > 0) == set(final)
    best = max(final, key=final.get)
    assert max(full, key=full.get) == best and legacy[best] >= full[best]
    if n > MAX_OPTIONS:  # the legacy confidence is conditional on the finalists: strictly too high
        assert legacy[best] > full[best]


@pytest.mark.parametrize("seed", range(20))
def test_linked_composition_keeps_argmax_and_finalist_odds_for_any_reads(seed):
    """Non-Luce reads (each read draws its own noise): the argmax and the finalists' odds are the final read's."""
    rng = random.Random(seed)
    opts = [f"o{i}" for i in range(rng.choice([53, 120, 200]))]
    pool, rounds = list(opts), []
    while len(pool) > MAX_OPTIONS:
        rnd, nxt = [], []
        for a in range(0, len(pool), MAX_OPTIONS):
            chunk = pool[a:a + MAX_OPTIONS]
            if len(chunk) == 1:
                nxt += chunk
                continue
            raw = {o: rng.random() ** 3 for o in chunk}
            z = sum(raw.values())
            p = {o: v / z for o, v in raw.items()}
            surv = sorted(chunk, key=lambda o: -p[o])[:2]
            rnd.append((chunk, p, surv))
            nxt += surv
        rounds.append(rnd)
        pool = nxt
    raw = {o: rng.random() for o in pool}
    final = {o: v / sum(raw.values()) for o, v in raw.items()}
    full = full_set_distribution(opts, rounds, final)
    assert sum(full.values()) == pytest.approx(1.0) and all(v >= 0 for v in full.values())
    assert max(full, key=full.get) == max(final, key=final.get)
    a, b = sorted(pool)[:2]
    assert full[a] / full[b] == pytest.approx(final[a] / final[b])
    with pytest.raises(ValueError):
        full_set_distribution(opts, rounds, final, method="product")


# ------------------------------------------------------------------ intern_decision: the real knockout, fake reads
class FakeIntern(InternDecisionBackend):
    """``InternDecisionBackend`` without a checkpoint: ``_score`` (one forward pass over <= max_q fields) returns Luce
    reads; ``_score_many`` / ``_knockout`` / ``_decide`` are the real ones."""

    def __init__(self, w: dict[str, float], max_options: int = MAX_OPTIONS, knockout_probs: str = "linked"):
        self.name, self.model, self.price_model = "intern_decision", "fake-intern", "fake-intern"
        self.max_q, self.max_options, self.temperature = 16, max_options, 1.0
        self.knockout_probs = knockout_probs
        self.w = w
        self.fields_per_pass: list[int] = []

    def describe(self):  # noqa: ANN201
        return {"backend": self.name, "model": self.model}

    def _score(self, state, questions, names, usage):  # noqa: ANN001, ANN202
        usage.n_calls += 1
        usage.input_tokens += 100
        usage.output_tokens += len(questions)
        self.fields_per_pass.append(len(questions))
        out = []
        for q in questions:
            if q.kind == QKind.NOUL:
                out.append({"type": "noul", "probabilities": {"no": 0.4, "yes": 0.6}, "confidence": 0.6, "noul": 0.6,
                            "decision": "yes"})
                continue
            p = luce_read(self.w, list(q.options))
            best = max(p, key=p.get)
            out.append({"type": "choice", "probabilities": p, "confidence": p[best], "decision": best, "choice": best})
        return out


@pytest.mark.parametrize("n", SIZES)
def test_intern_knockout_records_passes_rounds_and_the_full_set(n):
    w = strengths(n)
    b = FakeIntern(w)
    led = Ledger()
    res = b.decide("S", [Choice("Which tool?", list(w), decision_point="tool_select"), Noul("Done?")], ledger=led)
    a, u = res.answers[0], res.usage
    plan = knockout_plan(n, MAX_OPTIONS)
    if n <= MAX_OPTIONS:  # one forward pass answers both questions
        assert u.n_passes == u.n_calls == 1 and u.n_rounds == 1 and "knockout" not in a.raw
        assert b.fields_per_pass == [2]
    else:  # round 1 reads every chunk in one multi-field pass, the final read is a second pass, the Noul a third
        ko = a.raw["knockout"]
        assert ko["rounds"] == plan["rounds"] == 2 and ko["reads"] == plan["reads"] and ko["passes"] == 2
        assert len(ko["finalists"]) == plan["finalists"] and a.value in ko["finalists"]
        assert u.n_passes == u.n_calls == 3 and u.n_rounds == 2
        assert b.fields_per_pass == [len([c for c in plan["chunks"][0] if c > 1]), 1, 1]  # chunk fields, final, Noul
    assert a.distribution == pytest.approx(truth(w), abs=2e-6)  # 1e-6 floor of answer_from_distribution
    assert a.value == max(w, key=w.get) and a.confidence == pytest.approx(truth(w)[a.value], abs=2e-6)
    rec = led.records[-1]
    assert (rec.meta["n_passes"], rec.meta["n_rounds"], rec.meta["n_calls"]) == (u.n_passes, u.n_rounds, u.n_calls)
    assert decision_pass_counts(led.records) == {"s1_calls": 1, "s1_rounds": u.n_rounds, "s1_passes": u.n_passes}


def test_intern_knockout_finalists_mode_is_the_legacy_output():
    w = strengths(200)
    a = FakeIntern(w, knockout_probs="finalists").decide("S", [Choice("Which tool?", list(w))]).answers[0]
    fin = a.raw["knockout"]["finalists"]
    assert a.raw["knockout"]["full_set"] == "finalists"
    assert all(a.distribution[o] < 1e-5 for o in w if o not in fin)
    # the final read (up to the 1e-6 floor that answer_from_distribution gives each of the 192 eliminated options)
    assert a.distribution[a.value] == pytest.approx(luce_read(w, fin)[a.value], abs=5e-4)
    assert a.value == max(w, key=w.get)


# ------------------------------------------------------------------ serve_jev: the real knockout, fake logits
class LuceEngine:
    @staticmethod
    def make(w: dict[str, float], method: str = "linked"):  # noqa: ANN205
        pytest.importorskip("torch")
        from jevharness.training.serve_jev import STAT_KEYS, Engine

        class _E(Engine):
            def __init__(self) -> None:  # no model: only the request / knockout / accounting code is real
                self.cal, self.lock, self.stats = {}, threading.RLock(), {k: 0 for k in STAT_KEYS}
                self.knockout_probs, self.name = method, "luce"
                self.calls: list[int] = []

            def _option_logits(self, items):  # noqa: ANN001, ANN202
                self._item_stats = [{"prefilled": 10, "stateless": 10, "hit": 0, "passes": set()} for _ in items]
                self._new_pass(list(range(len(items))))  # one batched pass per call
                self.calls.append(len(items))
                return [[math.log(w.get(o.split(" -- ")[0], 1.0)) for o in opts] for _, _, opts, _ in items]

        return _E()


@pytest.mark.parametrize("n", SIZES)
def test_serve_jev_reports_passes_rounds_and_links_the_reads(n):
    w = strengths(n)
    eng = LuceEngine.make(w)
    q = {"type": "choice", "instructions": "Which tool?", "criteria": {o: None for o in w}}
    answers, usage = eng.answer_all("S", {"t": q, "done": {"type": "noul", "instructions": "Done?"}})
    a = answers["t"]
    probs = a["probabilities"]
    assert probs == pytest.approx(truth(w), abs=1e-6) and a["choice"] == max(w, key=w.get)  # float32 softmax
    plan = knockout_plan(n, MAX_OPTIONS)
    if n <= MAX_OPTIONS:
        assert usage["n_passes"] == 1 and usage["n_rounds"] == 1 and "knockout" not in a and eng.calls == [2]
    else:  # small questions in one call, then the knockout's chunk round and its final read
        ko = a["knockout"]
        assert ko["rounds"] == usage["n_rounds"] == 2 and ko["reads"] == plan["reads"]
        assert set(ko["finalists"]) == set(_simulate(w, MAX_OPTIONS)[1]) and len(ko["finalists"]) == plan["finalists"]
        assert usage["n_passes"] == 3 and eng.calls == [1, len([c for c in plan["chunks"][0] if c > 1]), 1]
    assert eng.stats["passes"] == usage["n_passes"]
    legacy = LuceEngine.make(w, "finalists").answer_all("S", {"t": q})[0]["t"]
    assert legacy["choice"] == a["choice"] and legacy["confidence"] >= a["confidence"]


def test_jev_client_reads_server_reported_passes_and_hosted_jev_stays_unobserved():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from jevharness.decision.jev import JevBackend
    from jevharness.decision.server import build_app
    from jevharness.training.serve_jev import build_app as build_serve_app

    w = strengths(200)
    for app in (build_app(FakeIntern(w), name="intern"), build_serve_app(_with_tok(LuceEngine.make(w)))):
        client = JevBackend(model="x", base_url="http://testserver", api_key="")
        client._http = TestClient(app)
        led = Ledger()
        r = client.decide("S", [Choice("Which tool?", list(w), decision_point="tool_select")], ledger=led)
        assert r.usage.n_rounds == 2 and r.usage.n_passes == 2 and r.usage.n_calls == 1
        assert led.records[-1].meta["n_passes"] == 2 and led.records[-1].meta["n_rounds"] == 2
        assert r.answers[0].distribution == pytest.approx(truth(w), abs=2e-6)
        assert r.answers[0].raw["knockout"]["rounds"] == 2

    class Hosted:  # a hosted-Jev response: no pass accounting -> unobserved, ledger meta unchanged
        def post(self, path, json=None, headers=None):  # noqa: ANN001, ANN202
            import httpx

            ans = {k: {"type": "choice", "choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}}
                   for k in json["questions"]}
            return httpx.Response(200, json={"model": "jev-1.13.0", "answers": ans,
                                             "usage": {"input_tokens": 5, "output_tokens": 0}})

    hosted = JevBackend(model="jev", base_url="http://x", api_key="k")
    hosted._http = Hosted()
    led = Ledger()
    r = hosted.decide("S", [Choice("q", ["a", "b"])], ledger=led)
    assert r.usage.n_passes is None and r.usage.n_rounds is None
    assert led.records[-1].meta == {"n_questions": 1, "n_calls": 1}


def _with_tok(eng):  # noqa: ANN001, ANN202
    class Tok:
        def __call__(self, text, add_special_tokens=False):  # noqa: ANN001, ANN204
            return type("Enc", (), {"input_ids": list(range(len(text.split())))})()

    eng.tok = Tok()
    eng.describe = lambda: {"name": "luce"}
    return eng


# ------------------------------------------------------------------ the harness tournament (agent.py)
class LuceBackend(DecisionBackend):
    name = "luce"
    max_options = MAX_OPTIONS

    def __init__(self, w: dict[str, float], batched: bool = True):
        self.w, self.batched_questions = w, batched

    def _decide(self, state, questions):  # noqa: ANN001, ANN202
        answers = [answer_from_distribution(q, luce_read(self.w, list(q.options))) for q in questions]
        return DecisionResult(answers=answers, usage=Usage(backend=self.name, n_calls=1, n_passes=1, n_rounds=1))


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("n", SIZES)
def test_harness_tournament_pass_counts(n, batched):
    from jevharness.harness.agent import JevAgent, Tap

    w = strengths(n)

    class Host:  # JevAgent._choice_tournament needs only `s1` and `_ask` (as bench/suites/tau2.py uses it)
        s1 = LuceBackend(w, batched)

        def _ask(self, state, q, ledger):  # noqa: ANN001, ANN202
            return self.s1.ask(state, q, ledger=ledger)

    tap = Tap(Ledger())
    rank: list[str] = []
    a = JevAgent._choice_tournament(Host(), "S", "Which tool?", list(w), "tool_select", tap, ranking=rank)
    plan = knockout_plan(n, MAX_OPTIONS, keep=2, read_if_more_than=2)
    pc = decision_pass_counts(tap.records)
    assert pc["s1_rounds"] == plan["rounds"] and a.value == max(w, key=w.get)
    if n <= MAX_OPTIONS:
        assert pc == {"s1_calls": 1, "s1_rounds": 1, "s1_passes": 1}
        assert all(r.tournament_round is None for r in tap.records)
    else:
        # batched: one call per round; unbatched: one call per chunk read, plus the final read
        assert pc["s1_calls"] == (plan["rounds"] if batched else plan["reads"]) == pc["s1_passes"]
        assert sum(r.meta["n_questions"] for r in tap.records) == plan["reads"]
        assert [r.tournament_round for r in tap.records][-1] == plan["rounds"] - 1
        assert len(a.distribution) == plan["finalists"] and set(rank) == set(w)
    # the same counts from the trajectory's per-call dicts
    assert decision_pass_counts([{"component": "s1", "meta": r.meta, "tournament_round": r.tournament_round}
                                 for r in tap.records]) == pc


def test_decision_pass_counts_unobserved_passes_and_empty():
    recs = [{"component": "s1", "meta": {"n_calls": 1}, "tournament_round": 0},
            {"component": "s1", "meta": {"n_calls": 1, "n_passes": 1}, "tournament_round": 1},
            {"component": "s2", "meta": {}}]
    assert decision_pass_counts(recs) == {"s1_calls": 2, "s1_rounds": 2, "s1_passes": None}
    assert decision_pass_counts([]) == {"s1_calls": 0, "s1_rounds": 0, "s1_passes": None}


def test_cache_key_tracks_the_composition_only_for_knockout_choices(tmp_path):
    w = strengths(60)
    small = FakeIntern(strengths(10))
    c1 = CachedBackend(small, tmp_path / "c.sqlite")
    q_small = [Choice("q", list(strengths(10)))]
    key_small = c1._key("S", q_small)
    small.knockout_probs = "finalists"
    assert c1._key("S", q_small) == key_small  # nothing knocked out: the old entries stay valid
    big = FakeIntern(w)
    c2 = CachedBackend(big, tmp_path / "c.sqlite")
    q_big = [Choice("q", list(w))]
    k_linked = c2._key("S", q_big)
    big.knockout_probs = "finalists"
    assert c2._key("S", q_big) != k_linked
    big.knockout_probs = "linked"
    r1, r2 = c2.decide("S", q_big), c2.decide("S", q_big)
    assert r2.usage.cached and r2.usage.n_passes == r1.usage.n_passes == 2 and r2.usage.n_rounds == 2
