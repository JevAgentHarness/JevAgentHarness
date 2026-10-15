"""scripts/explore_template_baseline.py (offline): on a tiny synthetic fixture, the template key read from the state
text (keyword family, executed-tool sequence), the training labels (argmax of the target, augmented and yes/no rows
left out), the lookups with their fall-back / back-off restricted to the offered tools, the in-sample key bound,
task-mean scoring with the acceptable-set rule, key coverage with covered / uncovered accuracy, the disagreement
counts and the TeX keys (decision-model rows are never keyed on the shifted corpus)."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import explore_template_baseline as T  # noqa: E402

MULTI = "TASK: Process these requests in order: (1) ship Ann Lee's pending order; (2) cancel Bo Kim's pending order " \
        "(the customer requested it). Use express shipping for gold-tier customers and standard shipping otherwise."
NOSPEED = "TASK: Ship Ann Lee's pending order."
NONAME = ("TASK: A customer wrote in asking us to ship their pending order (express if they are gold tier, otherwise "
          "standard), but did not say who they are.")
OPTS = ["find_user", "get_orders", "ship_order", "cancel_order", "ask_user", "finish", "search_users"]


def state(task: str, tools: list[str], obs: str = "ok") -> str:
    hist = "\n".join(f'[{i}] {t}({{"x": 1}}) -> {{"y": 2}}\nlog: heartbeat ok #{i}' for i, t in enumerate(tools))
    return task + ("\n\nHISTORY:\n" + hist if tools else "") + f"\n\nCURRENT OBSERVATION:\n[9] not_a_tool(oops) {obs}"


def row(i: int, st: str, label: str, rid: str | None = None, question: str = T.TOOL_Q) -> dict:
    opts = [f"{o} -- desc of {o}" for o in OPTS]
    tgt = {o: (0.9 if o.startswith(label + " ") else 0.1 / (len(opts) - 1)) for o in opts}
    return {"id": rid or f"jevgym-multi-{30000 + i}-d60-near-strict/a:{i}/tool_select", "dp": "tool_select",
            "question": question, "state": st, "options": opts, "target": tgt, "meta": {}}


def item(i: int, task: str, st: str, at: list[str], off: bool = False) -> dict:
    return {"task": task, "qhash": f"q{i}", "state": st, "options": list(OPTS),
            "rec": {"dp": "tool_select", "gold": at[0], "gold_aux": {"v": 1, "acceptable_tools": at}, "gold_status": "ok"},
            "strata": {"all": True, "must_ask": at == ["ask_user"], "after_off_plan": off, "long_history": False}}


def test_key_from_state_text():
    assert T.family(MULTI) == "multi:ship,cancel"
    assert T.family(MULTI, coarse=True) == "multi"
    assert T.family(NOSPEED) == "ship:no_speed"
    assert T.family(NONAME) == "ship:no_name"
    assert T.family("TASK: Refund the full amount of Ann Lee's most recent delivered order (highest order id).") == "refund:no_method"
    assert T.family("TASK: The customer Ann Lee asked us to cancel their pending order. Please do it.") == "cancel"
    st = state(MULTI, ["find_user", "get_orders"])
    assert T.tool_history(st) == ("find_user", "get_orders")       # the observation's "[9] not_a_tool(" is ignored
    assert T.tool_history(state(MULTI, [])) == ()
    assert T.key_last(state(MULTI, [])) == ("multi:ship,cancel", T.START)
    assert T.key_full(st) == ("multi:ship,cancel", ("find_user", "get_orders"))


def test_training_rows_and_lookup(tmp_path: Path):
    rows = [row(0, state(MULTI, []), "find_user"), row(1, state(MULTI, []), "find_user"),
            row(2, state(MULTI, ["find_user"]), "get_orders"),
            row(3, state(MULTI, ["find_user", "get_orders"]), "ship_order"),
            row(4, state(NOSPEED, ["find_user", "get_orders"]), "ask_user"),
            row(5, state(MULTI, []), "finish", rid="jevgym-multi-30005-d60-near-strict/a:5/tool_select#ko"),
            row(6, state(MULTI, []), "finish", question="Is this the right next step?"),
            row(7, state(MULTI, []), "finish", rid="when2call-7")]
    p = tmp_path / "train.jsonl.gz"
    with gzip.open(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tr = T.training_rows(p)
    assert [r["label"] for r in tr] == ["find_user", "find_user", "get_orders", "ship_order", "ask_user"]
    assert len(T.training_rows(p, with_augmented=True)) == 6
    full, prior = T.Lookup(tr, T.key_full), T.Lookup(tr, None)
    assert full.predict(state(MULTI, ["find_user"]), OPTS) == ("get_orders", "key")
    assert full.predict(state(MULTI, ["search_users"]), OPTS) == ("find_user", "unseen")    # prior's top label
    assert full.predict(state(MULTI, ["find_user"]), ["finish", "ship_order"]) == ("ship_order", "key_not_offered")
    assert prior.predict("anything", ["ask_user", "get_orders"]) == ("ask_user", "prior")  # tie -> alphabetical
    assert T.pick({"a": 2, "b": 2}, ["b", "a"]) == "a" and T.pick({"a": 1}, ["z"]) is None


def test_scoring_coverage_and_tex(tmp_path: Path):
    tr = [{"id": "r", "task": "t", "state": state(MULTI, []), "options": OPTS, "label": "find_user"},
          {"id": "r", "task": "t", "state": state(MULTI, ["find_user"]), "options": OPTS, "label": "get_orders"},
          {"id": "r", "task": "t", "state": state(MULTI, ["find_user", "get_orders"]), "options": OPTS, "label": "ship_order"},
          {"id": "r", "task": "t", "state": state(NOSPEED, ["find_user", "get_orders"]), "options": OPTS, "label": "ask_user"},
          {"id": "r", "task": "t", "state": state(NOSPEED, []), "options": OPTS, "label": "find_user"}]
    items = [item(0, "A", state(MULTI, []), ["find_user"]),                                  # seen key, right
             item(1, "A", state(MULTI, ["find_user"]), ["get_orders"]),                     # seen key, right
             item(2, "A", state(MULTI, ["search_users", "find_user"]), ["get_orders"], off=True),  # unseen: prior wrong,
             item(3, "B", state(NOSPEED, ["find_user", "get_orders"]), ["ask_user"])]       # back-off (last) right
    answers = {"q0": "find_user", "q1": "get_orders", "q2": "get_orders", "q3": "ship_order"}
    ap = tmp_path / "answers_rl.jsonl"
    ap.write_text("".join(json.dumps({"qhash": q, "answer": a, "valid": True, "error": None}) + "\n" for q, a in answers.items()))
    res = T.score_set(items, tr, {"all": "", "must_ask": "", "after_off_plan": ""}, {"rl": T.load_answers(ap)})
    rows = res["subsets"]["all"]["rows"]
    # task means: A = (1 + 1 + 0) / 3, B = 1 -> (2/3 + 1) / 2
    assert abs(rows["lookup"]["acc"] - (2 / 3 + 1) / 2) < 1e-9
    assert abs(rows["lookup"]["pooled"] - 0.75) < 1e-9
    assert rows["lookup_backoff"]["acc"] == 1.0 and rows["key_ceiling"]["acc"] == 1.0
    assert abs(rows["rl"]["acc"] - (1 + 0) / 2) < 1e-9              # rl: A all right, B wrong
    assert abs(rows["prior"]["acc"] - (1 / 3 + 0) / 2) < 1e-9        # prior answers find_user everywhere
    assert res["subsets"]["must_ask"]["rows"]["lookup"]["acc"] == 1.0
    assert res["subsets"]["after_off_plan"]["rows"]["lookup"]["acc"] == 0.0
    cv = res["coverage"]["lookup"]
    assert (cv["covered_n"], cv["uncovered_n"], cv["share"]) == (3, 1, 0.75)
    assert cv["covered"]["lookup"]["acc"] == 1.0 and cv["uncovered"]["lookup"]["acc"] == 0.0
    assert cv["uncovered"]["rl"]["acc"] == 1.0 and cv["share_after_off_plan"] == 0.0
    d = res["disagreement"]["lookup"]
    assert (d["lookup_wrong_rl_right"], d["lookup_right_rl_wrong"], d["lookup_wrong_rl_right_covered"]) == (1, 1, 0)
    assert res["routes"]["lookup_backoff"] == {"key": 3, "backoff_last": 1}
    # TeX keys: C rows for every row; on the shifted corpus the baselines only
    full = {"train": {"rows": 5, "tasks": 1, "keys_lookup": 4, "keys_lookup_last": 4}, "c": res,
            "shift": T.score_set([dict(it) for it in items], tr, {"all": "", "renamed": ""}, {"rl": T.load_answers(ap)})}
    k = T.tex_rows(full)
    assert k["tpl/c/all/lookup/acc"] == "83.3" and k["tpl/c/cov/full/share"] == "75.0"
    assert k["tpl/c/all/n"] == "4" and k["tpl/c/dis/full/lw_rr"] == "1"
    assert "tpl/shift/all/lookup/acc" in k and "tpl/shift/all/rl/acc" not in k
    assert "tpl/c/cov/last/covered/rl/acc" not in k                  # only (b)'s coverage split is keyed
    tex = T.to_tex({"generated": "x", "git": "y", "label": "z", **full})
    assert "\\defres{tpl/c/all/lookup/acc}{83.3}" in tex
