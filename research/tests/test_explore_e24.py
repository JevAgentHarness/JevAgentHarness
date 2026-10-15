"""Exploratory row E24 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E24: agent-state accuracy (L2) on public
benchmarks"): the L2-ALFWorld corpus construction (labels, lenient sets, strata), the gold hygiene of the question file
and the replay path, and the scoring rules, on a tiny fixture. Offline: no network, no model; the one test that replays
a real ALFWorld game skips when ALFWorld or its data are absent."""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_e24_public_l2 as E  # noqa: E402

from jevharness.accounting import Ledger  # noqa: E402
from jevharness.decision.base import DecisionBackend  # noqa: E402
from jevharness.decision.types import DecisionResult, Usage, answer_from_distribution  # noqa: E402

OPTS = ["go to desk 1", "go to desk 2", "go to shelf 1", "go to shelf 2", "take book 1 from desk 1",
        "take book 3 from desk 1", "take pen 1 from desk 1", "take book 2 from shelf 1", "use desklamp 1", "look"]


def test_instance_equivalents_take_and_use():
    eq = E.instance_equivalents("take book 1 from desk 1", OPTS)
    assert eq == {"take book 1 from desk 1", "take book 3 from desk 1"}      # same class, same receptacle only
    assert E.instance_equivalents("use desklamp 1", OPTS + ["use desklamp 2", "use floorlamp 1"]) == \
        {"use desklamp 1", "use desklamp 2"}
    assert E.instance_equivalents("go to desk 1", OPTS) == {"go to desk 1"}
    assert E.instance_equivalents(None, OPTS) == set()


def test_lenient_set_rules():
    # strict only (no hand-coded label): its instance equivalents
    assert E.lenient_set("take book 3 from desk 1", None, None, OPTS) == ["take book 1 from desk 1",
                                                                        "take book 3 from desk 1"]
    # the hand-coded expert's search go-to adds its priority tier, restricted to offered actions
    info = {"kind": "search", "ret": "go to shelf 1", "tier": ["shelf 1", "shelf 2", "shelf 9"]}
    got = E.lenient_set("go to desk 2", "go to shelf 1", info, OPTS)
    assert got == ["go to desk 2", "go to shelf 1", "go to shelf 2"]
    # a tier is used only when it belongs to the hand-coded action
    info_other = {"kind": "search", "ret": "go to desk 1", "tier": ["desk 1", "desk 2"]}
    assert E.lenient_set("go to shelf 1", "go to shelf 1", info_other, OPTS) == ["go to shelf 1"]
    # a GOTO subgoal: every receptacle of the type
    goto = {"kind": "goto", "ret": "go to desk 1", "tier": ["desk 1", "desk 2"]}
    assert E.lenient_set("go to desk 1", "go to desk 1", goto, OPTS) == ["go to desk 1", "go to desk 2"]


def test_off_plan_flags():
    st = [{"strict": "a", "executed": "a"}, {"strict": "b", "executed": "x"}, {"strict": None, "executed": "y"},
          {"strict": "c", "executed": "c"}]
    assert E.off_plan_flags(st) == [False, False, True, True]
    st2 = [{"strict": None, "executed": "x"}, {"strict": "b", "executed": "b"}]
    assert E.off_plan_flags(st2) == [False, False]       # a state without a label never starts off-plan


class FakeEnv:
    """A scripted stand-in for AlfWorldEnv: a walkthrough, a hand-coded plan per history length, fixed options."""

    def __init__(self, game_file, split="eval_in_distribution", oracle="auto", plans=None, wt=None, obs=None):
        self._history: list[str] = []
        self._walk = list(wt or [])
        self._plans = plans or {}
        self._obs = obs or {}
        self._expert_plan: list[str] = []

    def reset(self):
        self._history = []
        self._expert_plan = ["look"]
        return "room"

    def admissible_actions(self):
        return list(OPTS)

    def on_walkthrough(self):
        return self._history == self._walk[: len(self._history)]

    def _load_walkthrough(self):
        return self._walk

    def step(self, tool, args):
        from jevharness.bench.env import StepResult

        self._history.append(args["action"])
        self._expert_plan = [self._plans.get(len(self._history), "look")]
        return StepResult(self._obs.get(len(self._history), "ok"), done=False)

    def evaluate(self, final_answer):
        return {"success": False}


def _traj(actions, golds, task_id="alfworld/valid_seen/pick_and_place_simple-Book-None-Desk-1/trial_T1"):
    steps = []
    for i, (a, g) in enumerate(zip(actions, golds)):
        steps.append({"step": i, "tool": "act", "args": {"action": a}, "observation": "ok",
                      "decisions": [{"dp": "action", "options": list(OPTS), "state": f"STATE {i}", "gold": g}]})
    return {"task_id": task_id, "steps": steps, "result": {"success": True, "task_type": "pick_and_place_simple"}}


def test_relabel_episode_rules():
    wt = ["go to desk 1", "take book 1 from desk 1", "go to shelf 1"]
    # after the agent leaves the walkthrough at step 1, the hand-coded plan labels; 'look' (fallback) loses track
    plans = {1: "take book 3 from desk 1", 2: "go to shelf 2", 3: "look", 4: "go to shelf 1"}
    t = _traj(["go to desk 1", "take pen 1 from desk 1", "go to shelf 2", "go to desk 2", "go to shelf 1"],
              ["go to desk 1", "take book 1 from desk 1", "go to shelf 2", "look", "go to shelf 1"])
    res = E.relabel_episode(t, Path("unused"), lambda *a, **k: FakeEnv(*a, plans=plans, wt=wt, **k))
    s = res["states"]
    assert res["checks"]["desync_from"] is None
    assert [x["strict"] for x in s] == ["go to desk 1", "take book 1 from desk 1", "go to shelf 2", None, None]
    assert [x["source"] for x in s[:3]] == ["walkthrough", "walkthrough", "handcoded"]
    assert [x["drop"] for x in s] == [None, None, None, "after_expert_fallback", "after_expert_fallback"]
    # on the walkthrough the hand-coded action joins the lenient set (with its instance equivalents)
    assert s[1]["hand"] == "take book 3 from desk 1"
    assert s[1]["lenient"] == ["take book 1 from desk 1", "take book 3 from desk 1"]
    assert s[1]["strict_instances"] == 2


def test_relabel_desync_drops_rest():
    t = _traj(["go to desk 1", "go to desk 2"], ["go to desk 1", "go to desk 2"])
    t["steps"][0]["observation"] = "something else"
    res = E.relabel_episode(t, Path("unused"), lambda *a, **k: FakeEnv(*a, wt=["go to desk 1", "go to desk 2"], **k))
    assert res["checks"]["desync_from"] == 1
    assert [x["drop"] for x in res["states"]] == [None, "replay_desync"]


def _corpus(tmp_path):
    wt = ["go to desk 1", "take book 1 from desk 1"]
    rel = {"seen": {}, "unseen": {}}
    for split, tid in (("seen", "alfworld/valid_seen/pick_and_place_simple-Book-None-Desk-1/trial_A"),
                       ("unseen", "alfworld/valid_unseen/look_at_obj_in_light-Book-None-DeskLamp-10/trial_B")):
        t = _traj(["go to desk 1", "take book 3 from desk 1"], ["go to desk 1", "take book 1 from desk 1"], tid)
        r = E.relabel_episode(t, Path("unused"), lambda *a, **k: FakeEnv(*a, plans={1: "take book 3 from desk 1"},
                                                                         wt=wt, **k))
        r["task_type"], r["success"] = tid.split("/")[2].split("-")[0], True
        rel[split][tid] = r
    c = E.assemble(rel, {"seen": (Path("a.jsonl"), "E11"), "unseen": (Path("b.jsonl"), "E23")})
    meta = E.write_corpus(c, tmp_path)
    return c, meta


def test_question_file_carries_no_label(tmp_path):
    c, meta = _corpus(tmp_path)
    with gzip.open(tmp_path / "questions.jsonl.gz", "rt") as f:
        qs = [json.loads(x) for x in f]
    assert qs and all(set(q) == {"qhash", "state", "options"} for q in qs)
    blob = (tmp_path / "questions.jsonl.gz").read_bytes()
    assert b"strict" not in gzip.decompress(blob) and b"lenient" not in gzip.decompress(blob)
    assert meta["stats"]["states"] == 4 and meta["stats"]["by_split"]["seen"]["states"] == 2
    occ = E.jl(tmp_path / "occurrences.jsonl.gz")
    assert {o["qhash"] for o in occ} == {q["qhash"] for q in qs}
    assert all(o["qhash"] == E.question_hash(q["state"], q["options"]) for o in occ for q in qs
               if q["qhash"] == o["qhash"])


class FirstOptionBackend(DecisionBackend):
    """Answers the first option of every Choice; records every question it saw (for the gold-hygiene check)."""
    name = "first"
    model = "first"
    max_options = 4

    def __init__(self):
        self.seen = []

    def _decide(self, state, questions):
        self.seen.extend(questions)
        ans = [answer_from_distribution(q, {q.options[0]: 0.9, **{o: 0.1 / max(1, len(q.options) - 1)
                                                                  for o in q.options[1:]}}) for q in questions]
        return DecisionResult(answers=ans, usage=Usage(backend="first", model="first", cost_usd=0.001))


def test_replay_path_and_gold_hygiene(tmp_path):
    _corpus(tmp_path)
    qs = E.jl(tmp_path / "questions.jsonl.gz")
    be = FirstOptionBackend()
    out = E.replay_questions(qs, be, retries=0)
    assert len(out) == len(qs) and all(r["error"] is None and r["valid"] for r in out)
    # 10 options > max_options 4: the harness knockout ran in rounds; every question asked is the action question
    assert len(be.seen) > len(qs)
    assert all(q.text == E.QUESTION and q.decision_point == "action" for q in be.seen)
    assert all(getattr(q, "gold", None) is None for q in be.seen)
    assert all(abs(r["cost_usd"] - 0.001 * r["requests"]) < 1e-9 for r in out)


def test_scoring_rules(tmp_path):
    c, _ = _corpus(tmp_path)
    corpus = E.load_corpus(tmp_path)
    occ = corpus["occ"]
    # state 1 of each game: strict = walkthrough 'take book 1', lenient adds 'take book 3' (instance), logged = 'take book 1'
    vals = {o["qhash"]: ("take book 3 from desk 1" if o["step"] == 1 else "go to desk 1") for o in occ}
    strict = E.game_means(occ, vals, "strict")
    lenient = E.game_means(occ, vals, "lenient")
    assert set(strict.values()) == {0.5} and set(lenient.values()) == {1.0}
    assert E.answer_value({"answer": "x", "error": "boom"}) is None
    assert E.answer_value({"answer": "x", "refusal": True}) is None
    assert E.answer_value({"answer": "x", "valid": False}) is None
    assert E.answer_value({"answer": "x", "valid": True}) == "x"
    # logged-gold sensitivity: from the logged fallback on, no logged label
    occ2 = [dict(occ[0], logged_gold="look", step=0), dict(occ[1], step=1)]
    E.mark_logged_lost(occ2)
    assert [o["logged_lost"] for o in occ2] == [True, True]
    assert E.occ_correct(occ2[1], "anything", "logged") is None


def test_kendall_tau_b():
    assert E.kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert E.kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert E.kendall_tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(2 / (2 * 3) ** 0.5)
    assert E.kendall_tau_b([1], [1]) is None


def _alfworld_ready() -> bool:
    try:
        import alfworld  # noqa: F401
        import textworld  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return (E.ALF_DATA / "valid_seen").is_dir() and E.ALF_SOURCES["seen"][0].exists()


@pytest.mark.skipif(not _alfworld_ready(), reason="ALFWorld, its valid_seen games or E11's logs are not present")
def test_instrumentation_changes_no_label():
    """Registration (b) 5: the E24 instrumentation is read-only. One logged game replayed in two fresh processes,
    with E11's labeler alone and with E11's labeler plus the instrumentation: identical labels at every state."""
    code = r"""
import json, sys
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
import explore_e24_public_l2 as E, explore_e11_alfworld as E11
from alfworld.agents.expert import handcoded_expert as H
mode = sys.argv[1]
E11.patch_handcoded_move()
if mode == 'e24':
    E.install_label_patches()
t = E.latest_by_task(E.ALF_SOURCES['seen'][0])
tids = sorted(t)[:3]
out = []
for tid in tids:
    H.random.seed(E._seed_for(tid))
    from jevharness.bench.suites.alfworld import AlfWorldEnv
    env = AlfWorldEnv(str(E._game_file(tid)), split='valid_seen', oracle='auto')
    env.reset()
    for st in t[tid]['steps']:
        out.append(list(env._expert_plan))
        r = env.step('act', {'action': st['args']['action']})
        if r.done:
            break
    env.evaluate(None)
print(json.dumps(out))
"""
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "TMPDIR": "/dev/shm" if Path("/dev/shm").is_dir() else "/tmp"}
    a = subprocess.run([sys.executable, "-c", code, "e11"], cwd=ROOT, capture_output=True, text=True, env=env, timeout=600)
    b = subprocess.run([sys.executable, "-c", code, "e24"], cwd=ROOT, capture_output=True, text=True, env=env, timeout=600)
    assert a.returncode == 0, a.stderr[-2000:]
    assert b.returncode == 0, b.stderr[-2000:]
    la, lb = json.loads(a.stdout.strip().splitlines()[-1]), json.loads(b.stdout.strip().splitlines()[-1])
    assert la == lb and len(la) > 5


def test_summarize_on_fixture(tmp_path, monkeypatch):
    _corpus(tmp_path)
    occ = E.jl(tmp_path / "occurrences.jsonl.gz")
    # one hosted and one served row; the hosted one errs on one question (scored wrong)
    with open(tmp_path / "answers_jev.jsonl", "w") as f:
        for i, o in enumerate(sorted({o["qhash"]: o for o in occ}.values(), key=lambda x: x["qhash"])):
            rec = {"qhash": o["qhash"], "answer": o["strict"], "confidence": 0.8, "valid": True, "error": None,
                   "cost_usd": 0.0001}
            if i == 0:
                rec.update(answer=None, error="RuntimeError: boom")
            f.write(json.dumps(rec) + "\n")
    with open(tmp_path / "answers_e11.jsonl", "w") as f:
        for o in {o["qhash"]: o for o in occ}.values():
            f.write(json.dumps({"qhash": o["qhash"], "answer": o["strict"], "confidence": 0.9, "valid": True,
                                "error": None, "cost_usd": 0.0}) + "\n")
    r = E.summarize(tmp_path, tmp_path / "no_box")
    assert r["models"]["e11"]["score"]["strict"]["pooled"]["acc"] == pytest.approx(1.0)
    assert r["models"]["jev"]["n_errors"] == 1
    assert r["models"]["jev"]["score"]["strict"]["pooled"]["acc"] < 1.0
    assert r["models"]["sl9b"]["run"] is False
    keys = E.tex_keys(r)
    assert keys["e24/e11/pooled/strict"] == "100.0" and keys["e24/e11/pooled/acceptable"] == "100.0"
    assert "e24/alf/n/states" in keys and all(k.startswith("e24/") for k in keys)
    md = E.to_markdown(r)
    assert "L2-ALFWorld" in md and "not run" in md
    tex = E.to_tex(r)
    assert tex.count("\\defres{") == len(keys)


class LineGame:
    """A scripted game for the consequence rule: win by doing g1, g2, g3 in order; 'noop' changes nothing; 'jump'
    does g1 and g2 at once. The 'expert' always proposes the next goal action."""
    GOAL = ["g1", "g2", "g3"]

    def __init__(self, *a, **k):
        self.k = 0
        self.t = 0

    def _infos(self):
        nxt = "look" if self.t == 0 else (self.GOAL[self.k] if self.k < 3 else "look")   # ALFWorld's opening look
        return {"won": [self.k >= 3], "admissible_commands": [["g1", "g2", "g3", "noop", "jump", "look"]],
                "extra.expert_plan": [[nxt]]}

    def reset(self):
        self.k, self.t = 0, 0
        return ["start"], self._infos()

    def step(self, acts):
        a = acts[0]
        self.t += 1
        if self.k < 3 and a == self.GOAL[self.k]:
            self.k += 1
        elif a == "jump" and self.k == 0:
            self.k = 2
        return ["ok"], [0], [self.k >= 3], self._infos()


def test_consequence_rule_on_a_scripted_game():
    rows = E.consequence_game("t", ["noop", "g1"], {0: ["g1", "noop", "look"], 1: ["g1", "noop", "jump", "g2"]},
                              "unused", factory=lambda *a, **k: LineGame())
    base = {r["step"]: r["h"] for r in rows if r["kind"] == "base"}
    acc = {(r["step"], r["action"]): r["accepted"] for r in rows if r["kind"] == "action"}
    assert base == {0: 3, 1: 3}         # the first state's opening look is not counted (H = 4 - 1)
    assert acc[(1, "g1")] and acc[(1, "jump")] and not acc[(1, "noop")] and not acc[(1, "g2")]
    assert acc[(0, "g1")] and not acc[(0, "noop")] and not acc[(0, "look")]


def test_acceptable_rule_uses_lenient_then_consequence():
    o = {"strict": "a", "lenient": ["a", "b"], "cons": {"c": True, "d": False}}
    assert E.occ_correct(o, "a", "acceptable") == 1.0 and E.occ_correct(o, "b", "acceptable") == 1.0
    assert E.occ_correct(o, "c", "acceptable") == 1.0 and E.occ_correct(o, "d", "acceptable") == 0.0
    assert E.occ_correct(o, "e", "acceptable") == 0.0 and E.occ_correct(o, None, "acceptable") == 0.0
    assert E.missing_pairs([dict(o, qhash="q")], {"q": "e"}) == 1
    assert E.missing_pairs([dict(o, qhash="q")], {"q": "c"}) == 0


def test_tau2_score_rules():
    w = ["cancel_pending_order", "modify_pending_order_items"]
    occ = [
        {"task": "retail/1", "qhash": "t1", "label_tool": "cancel_pending_order", "write_tools": w, "flash": "get_order_details",
         "args": [{"qhash": "a1", "label": "#W1", "flash": None}]},
        {"task": "retail/1", "qhash": "t2", "label_tool": "cancel_pending_order", "write_tools": w, "flash": "cancel_pending_order",
         "args": [{"qhash": "a2", "label": "<OTHER>", "flash": "<OTHER>"}]},
        {"task": "airline/7", "qhash": "t3", "label_tool": "modify_pending_order_items", "write_tools": w, "flash": "respond_to_user",
         "args": []},
    ]
    ans = {"t1": {"answer": "cancel_pending_order", "valid": True}, "a1": {"answer": "#W1", "valid": True},
           "t2": {"answer": "modify_pending_order_items", "valid": True}, "a2": {"answer": "<OTHER>", "valid": True},
           "t3": {"answer": "modify_pending_order_items", "valid": True}}
    sc = E.tau2_score(occ, ans)
    assert sc["retail"]["tool"] == pytest.approx(0.5) and sc["airline"]["tool"] == pytest.approx(1.0)
    assert sc["pooled"]["tool"] == pytest.approx(0.75)                       # task means: (0.5 + 1.0) / 2
    assert sc["retail"]["wrong_write"] == pytest.approx(0.5)                  # t2 chose another write tool
    assert sc["retail"]["call"] == pytest.approx(0.5) and sc["airline"]["call"] == pytest.approx(1.0)
    fl = E.tau2_score(occ, None, flash=True)
    assert fl["retail"]["tool"] == pytest.approx(0.5) and fl["airline"]["tool"] == pytest.approx(0.0)
    assert E.tau2_question_hash("s", ["a"], None, "q") != E.tau2_question_hash("s", ["a"], ["d"], "q")
