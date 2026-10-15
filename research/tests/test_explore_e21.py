"""Exploratory row E21, DecisionGym-Shift (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E21: a shifted
DecisionGym split"): the shifted suite (``bench/suites/jevgym_shift.py``), its seed guards and the gold hygiene of the
alias boundary. Offline: no network, no model."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import pytest

from jevharness.bench import registry as R
from jevharness.bench.suites import jevgym as J
from jevharness.bench.suites import jevgym_shift as S

ROOT = Path(__file__).resolve().parents[1]
STAGE2 = dict(families="v2", n_distractors=60, noise_lines=20, strict=True, distractor_style="near")
DEV = range(R.E21_DEV_RANGE[0], R.E21_DEV_RANGE[0] + 60)
COMBOS = list(itertools.product((True, False), repeat=3))


def _env(seed: int, combo=(True, True, True), **kw) -> S.JevGymShiftEnv:
    return S.JevGymShiftEnv(seed, **dict(zip(S.AXES, combo)), **{**STAGE2, **kw})


def _play(e, check=None) -> dict:
    """The scripted oracle policy: execute the oracle's next action until it finishes."""
    e.reset()
    ans = None
    for _ in range(40):
        if check:
            check(e)
        a = e.oracle_action()
        assert a is not None
        tool, args = a
        if tool == "finish":
            ans = args["answer"]
            break
        r = e.step(tool, args)
        assert not r.done, (e.task_id, tool, r.observation)
    return e.evaluate(ans)


# ------------------------------------------------------------------------------------------- the oracle policy
@pytest.mark.parametrize("combo", COMBOS)
def test_oracle_policy_solves_every_shift_task(combo):
    for seed in DEV:
        e = _env(seed, combo)
        assert _play(e)["success"], e.task_id


def test_answer_key_is_in_the_names_the_agent_sees():
    """At every state of the oracle play: the oracle's tool and the acceptable sets are offered names (aliases under
    the surface axis), and they equal the canonical core's labels up to renaming."""
    def check(e):
        names = {t["function"]["name"] for t in e.tools()} | {"finish"}
        a = e.oracle_action()
        f, u = e.oracle_acceptable_tool_sets(30)
        assert a[0] in names and set(f) <= names and set(u) <= names
        cf, cu = e.core.oracle_acceptable_tool_sets(30)
        assert f == [e.to_ext(t) for t in cf] and u == [e.to_ext(t) for t in cu]
        assert e.core.oracle_action()[0] == e.to_canon(a[0])
        assert not (set(f) & set(S.ALIASES))            # no canonical core name in a label under the surface axis
        assert e.oracle_accepts(*a) is True

    for seed in DEV:
        _play(_env(seed), check)


def test_aliases_round_trip_and_names_are_unique():
    e = _env(next(iter(DEV)))
    names = [t["function"]["name"] for t in e.tools()]
    assert len(names) == len(set(names)) == 8 + 120 and "finish" not in names
    for c, (a, desc, _) in S.ALIASES.items():
        assert e.to_ext(c) == a and e.to_canon(a) == c
        assert c not in names and names.count(a) == 1
        assert e.to_canon(c).startswith(S.HIDDEN)              # a canonical core name is not offered
        spec = next(t for t in e.tools() if t["function"]["name"] == a)
        assert spec["function"]["description"] == desc
    pool = ([n for n, _ in J.DISTRACTOR_NAMES] + [t.name for t in J.NEAR_TOOLS] + [t.name for t in S.SHIFT_NEAR]
            + [n for n, _ in S.SHIFT_RANDOM] + [a for a, _, _ in S.ALIASES.values()] + list(S.ALIASES))
    assert len(pool) == len(set(pool))
    off = _env(next(iter(DEV)), (False, True, True))
    assert {"find_user", "ask_user"} <= {t["function"]["name"] for t in off.tools()}
    assert off.to_canon("find_user") == "find_user" and off.to_ext("find_user") == "find_user"


def test_hidden_canonical_names_are_unknown_to_the_oracle():
    e = _env(next(iter(DEV)) + 1)       # multi: find_user is the canonical first step
    e.reset()
    a = e.oracle_action()
    assert a[0] == "lookup_customer"
    assert e.oracle_accepts("find_user", a[1]) is False
    assert e.oracle_verdict("find_user", a[1])["verdict"] == "detour"
    assert e.oracle_verdict("lookup_customer", a[1])["verdict"] == "progress"


def test_every_axis_off_is_the_generator_itself():
    for seed in DEV:
        e, g = _env(seed, (False, False, False)), J.JevGymEnv(seed, **STAGE2)
        assert e.instruction == g.instruction and e.core._plan() == g._plan() and e.reset() == g.reset()
        assert e.tools() == g.tools()
        assert e.task_id.startswith("jevgym-shift-none-")


def test_surface_axis_paraphrases_every_text_and_names_the_alias():
    for seed in DEV:
        e, g = _env(seed, (True, False, False)), J.JevGymEnv(seed, **STAGE2)
        assert e.core._plan() == g._plan()                   # same task, renamed
        assert e.instruction != g.instruction
        obs = e.reset()
        assert "request_clarification" in obs and "ask_user" not in obs
        for c in S.ALIASES:
            assert c not in e.instruction


def test_distraction_axis_doubles_distractors_and_noise():
    e = _env(next(iter(DEV)) + 1, (False, True, False))
    assert e.n_distractors == 120 and e.noise_lines == 40
    names = {t["function"]["name"] for t in e.tools()}
    base = {t["function"]["name"] for t in J.JevGymEnv(next(iter(DEV)) + 1, **STAGE2).tools()}
    assert base < names and len(names) == 128
    assert {t.name for t in S.SHIFT_NEAR} <= names
    e.reset()
    obs = e.step("find_user", {"name": e.user["name"]}).observation
    assert obs.count("heartbeat ok") == 40


def test_shift_near_duplicates_behave():
    e = _env(next(iter(DEV)) + 1)
    e.reset()
    v = e.oracle_verdict("lookup_customer_by_email", {"email": "x@y.z"})
    assert v["verdict"] == "detour"
    w = e.oracle_verdict("cancel_purchase_plan", {"user_id": e.user["user_id"], "reason": "customer_request"})
    assert w["verdict"] == "fatal" and w["fatal_kind"] == "terminal"
    r = e.step("cancel_purchase_plan", {"user_id": e.user["user_id"], "reason": "customer_request"})
    assert r.done and not e.evaluate(None)["success"]


# ------------------------------------------------------------------------------------------- the structural kinds
def _plan_shape(plan) -> tuple:
    return tuple(t for t, _ in plan)


def test_new_plan_shapes_never_occur_in_the_training_generator():
    seen = set()
    for fams in ("all", "v2"):
        for k in (2, 3, 4):
            for seed in range(400):
                seen.add(_plan_shape(J.JevGymEnv(seed, families=fams, multi_customers=k, n_distractors=10)._plan()))
    new = {}
    for seed in DEV:
        e = _env(seed)
        if (e.family, e.variant) in S.NEW_KINDS:
            new[(e.family, e.variant)] = _plan_shape(e.core._plan())
    assert set(new) == set(S.NEW_KINDS)
    assert not set(new.values()) & seen
    assert new[("clarify", "ask_twice")].count("ask_user") == 2
    assert "send_message" in new[("notify", None)]


def test_kind_schedule_shares():
    kinds = [S.kind_of(s) for s in range(330000, 330100)]
    assert kinds.count(("multi", None)) == 30 and sum(k in S.NEW_KINDS for k in kinds) == 30
    assert len({S.kind_of(s) for s in range(330100, 330300)}) == 8


def _seed_of(kind) -> int:
    return next(s for s in DEV if S.kind_of(s) == kind)


def test_ask_twice_needs_both_questions_before_the_write():
    e = _env(_seed_of(("clarify", "ask_twice")))
    e.reset()
    f, _ = e.oracle_acceptable_tool_sets()
    assert f == ["request_clarification"]                          # nobody is known yet: a must-ask state
    e.step("request_clarification", {"missing": "customer_name"})
    e.step("lookup_customer", {"name": e.user["name"]})
    e.step("list_customer_orders", {"user_id": e.user["user_id"], "status": "pending"})
    f, _ = e.oracle_acceptable_tool_sets()
    assert f == ["request_clarification"]                          # the speed is still missing: must ask again
    assert e.oracle_action() == ("request_clarification", {"missing": "shipping_speed"})
    snap = e.oracle_snapshot()
    snap.step("request_clarification", {"missing": "shipping_speed"})
    assert e.core.asked_slots == {"customer_name"}                 # the snapshot never changes the episode
    r = e.step("dispatch_order", {"order_id": e.pending_order["order_id"], "speed": e.ship_speed})
    assert r.done and not e.evaluate("done")["success"]            # strict: shipping before the second question


def test_ask_after_lookup_is_ambiguous_only_after_the_lookup():
    e = _env(_seed_of(("clarify", "ask_after_lookup")))
    e.reset()
    f, _ = e.oracle_acceptable_tool_sets()
    assert "lookup_customer" in f
    e.step("lookup_customer", {"name": e.user["name"]})
    obs = e.step("list_customer_orders", {"user_id": e.user["user_id"], "status": "delivered"}).observation
    assert sum(1 for o in json.loads(obs.split("\n")[0])) >= 2
    f, _ = e.oracle_acceptable_tool_sets()
    assert f == ["request_clarification"]
    assert e.instruction.count("ORD-") == 0


def test_notify_needs_the_message_after_the_shipment():
    seed = _seed_of(("notify", None))
    e = _env(seed)
    e.reset()
    u, o = e.user, e.pending_order
    e.step("lookup_customer", {"name": u["name"]})
    e.step("list_customer_orders", {"user_id": u["user_id"], "status": "pending"})
    assert e.oracle_verdict("message_customer", {"user_id": u["user_id"], "text": "hi"})["verdict"] == "detour"
    f, _ = e.oracle_acceptable_tool_sets()
    assert f == ["dispatch_order"]
    e.step("message_customer", {"user_id": u["user_id"], "text": "soon"})     # too early: a detour, recoverable
    e.step("dispatch_order", {"order_id": o["order_id"], "speed": J._speed(u)})
    assert not e.evaluate("done")["success"]
    f, _ = e.oracle_acceptable_tool_sets()
    assert f == ["message_customer"]
    e.step("message_customer", {"user_id": u["user_id"], "text": "Your order has shipped."})
    assert e.evaluate("done")["success"]
    e2 = _env(seed)
    e2.reset()
    e2.step("lookup_customer", {"name": u["name"]})
    e2.step("list_customer_orders", {"user_id": u["user_id"], "status": "pending"})
    e2.step("dispatch_order", {"order_id": o["order_id"], "speed": J._speed(u)})
    e2.step("send_email", {"user_id": u["user_id"], "subject": "x", "body": "y"})   # e-mail is not the chat
    assert not e2.evaluate("done")["success"]


def test_suite_routing_and_options():
    from jevharness.bench.suites import load_suite

    f = load_suite("jevgym", n=2, seed0=339000, shift=True, **STAGE2)
    assert type(f[0]()) is S.JevGymShiftEnv and len(f) == 2
    g = load_suite("jevgym", n=1, seed0=339000, shift={"surface": False}, **STAGE2)[0]()
    assert not g.surface and g.distraction and g.structure
    with pytest.raises(ValueError):
        load_suite("jevgym", n=1, seed0=339000, shift={"colour": True}, **STAGE2)
    with pytest.raises(ValueError):
        load_suite("jevgym", n=1, seed0=339000, shift=True, multi_rule="tier", **STAGE2)
    ids = [fac().task_id for fac in load_suite("jevgym", n=300, seed0=330000, shift=True, **STAGE2)]
    assert len(set(ids)) == 300 and all(i.startswith("jevgym-shift-sdt-") for i in ids)


def test_frozen_generator_is_untouched():
    fz = json.loads((ROOT / "configs/eval/jevgym_eval_tasks.json").read_text())
    assert fz["generator"]["sha256"] == hashlib.sha256((ROOT / "src/jevharness/bench/suites/jevgym.py").read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------- seeds
def test_evaluation_seeds_are_refused_and_never_trained_on(monkeypatch):
    from jevharness.training import seed_guard

    monkeypatch.delenv(R.EVAL_RUNS_ENV, raising=False)
    for seed0, n in ((330000, 100), (330100, 200), (330999, 1), (329990, 20)):
        with pytest.raises(SystemExit):
            R.check_suite_seeds("jevgym", seed0, n)
    R.check_suite_seeds("jevgym", R.E21_DEV_RANGE[0], 100)        # development seeds: allowed, never reported
    for lo, hi in ((330000, 330999), (339000, 339099)):
        assert R.in_ranges(lo, R.training_exclusions()) and R.in_ranges(hi, R.training_exclusions())
        with pytest.raises(ValueError):
            seed_guard.check_train_pool(lo, 10)
    assert not R.overlaps(330000, 10000, R.reserved_ranges() + R.STAGE2_RESERVED_RANGES + ((310000, 322999),))
    monkeypatch.setenv(R.EVAL_RUNS_ENV, "1")
    R.check_suite_seeds("jevgym", 330000, 300)


# ------------------------------------------------------------------------------------------- gold hygiene
def test_no_gold_reaches_a_backend_through_the_shift_env():
    """The wire canary of tests/test_wire_canary.py on the shifted env: REPLACE at the tool choice with a Jev-wire
    System-1 and an OpenAI-compatible System-2 behind mock transports; no oracle value (canaried) reaches a request."""
    sys.path.insert(0, str(ROOT / "tests"))
    import test_wire_canary as W  # noqa: E402

    from jevharness.decision.jev import JevBackend
    from jevharness.harness.agent import JevAgent
    from jevharness.harness.config import HarnessConfig

    s1_all, s2_all, dumps = W.Wire(W.jev_answers), W.Wire(W.jev_answers), []
    for seed in (next(iter(DEV)), _seed_of(("clarify", "ask_twice")), _seed_of(("notify", None))):
        real = _env(seed, noise_lines=5)
        env = W.CanaryOracleEnv(real, canary_tool=False)
        s1_wire, s2_wire = W.Wire(W.s1_policy(real)), W.Wire(W.s2_policy(real))
        s1 = JevBackend(provider="openrouter", api_key="test-key", transport=s1_wire.transport())
        llm = W.compat_llm("gemini", "gemini-3.5-flash", s2_wire, temperature=0, reasoning_effort="minimal")
        cfg = HarnessConfig(name="e21_replace", max_steps=14, tool_select={"mode": "replace"})
        traj = JevAgent(llm, cfg, s1=s1).run(env)
        assert traj.error is None, traj.error
        s1_all.requests += s1_wire.requests
        s2_all.requests += s2_wire.requests
        dumps.append(json.dumps(traj.to_dict(), default=str))
        for r in s1_wire.requests + s2_wire.requests:     # the env's own hidden values never travel either
            body = r["body"].decode() if isinstance(r["body"], bytes) else str(r["body"])
            assert "acceptable_tools" not in body and S.HIDDEN not in body
            assert '"family"' not in body and "ask_twice" not in body
    s1_all.assert_clean(min_requests=2, key_names=False)
    s2_all.assert_clean(min_requests=2, key_names=False)
    assert any(W.CANARY in d for d in dumps)


# ------------------------------------------------------------------------------------------- readout helpers
def _driver():
    sys.path.insert(0, str(ROOT / "scripts"))
    import explore_e21_shift as X  # noqa: E402
    return X


def test_surface_undone_rendering_matches_the_generator_wording():
    X = _driver()
    for kind in (("clarify", "ask_twice"), ("clarify", "order"), ("multi", None), ("notify", None)):
        seed = _seed_of(kind)
        sh, ca = X._canon_texts(seed)
        sh.reset()
        reply = ""
        if sh.family == "clarify":
            slot = sh.core.slots()[0]
            reply = sh.step("request_clarification", {"missing": slot}).observation
        obs0 = S.OPENING[True] + S.POLICY_SHIFT.format(ask="request_clarification")
        state = (f"TASK: {sh.instruction}\n\nHISTORY:\n[0] request_clarification({{\"missing\": \"x\"}}) -> {reply}\n"
                 f"[1] lookup_customer_by_email({{}}) -> no\n\nCURRENT OBSERVATION:\n{obs0}")
        out = X.undo_surface(state, sh, ca)
        assert f"TASK: {ca.instruction}" in out and "request_clarification" not in out and "ask_user(" in out
        assert "lookup_customer_by_email" in out                      # a near-duplicate's longer name is untouched
        assert J.ASK_POLICY in out and "Rule: do not guess" not in out
        if reply:
            ca.reset()
            want = ca.step("ask_user", {"missing": slot}).observation
            assert want in out


def test_subset_rules_and_mapped_scoring():
    X = _driver()
    o = {"task": "jevgym-shift-sdt-clarify.ask_twice-330008-d120-near-strict", "qhash": "q", "stratum": "after_off_plan",
         "gold": "request_clarification", "gold_aux": {"acceptable_tools": ["request_clarification"]}, "gold_status": "ok"}
    assert X.occ_subsets(o) == {"all", "mustask", "askany", "offplan", "renamed", "newkind"}
    fin = {**o, "task": "jevgym-shift-sdt-multi-330001-d120-near-strict", "stratum": "on_plan", "gold": "finish",
           "gold_aux": {"acceptable_tools": ["finish"]}}
    assert X.occ_subsets(fin) == {"all", "genkind"}
    qs = {"q": {"map": {"ask_user": "request_clarification"}}}
    assert X.score_answers([o], qs, {"q": {"answer": "ask_user", "valid": True}}) == {0: 1.0}
    assert X.score_answers([o], {"q": {"map": {}}}, {"q": {"answer": "lookup_customer", "valid": True}}) == {0: 0.0}
    assert X.task_seed(o["task"]) == 330008
