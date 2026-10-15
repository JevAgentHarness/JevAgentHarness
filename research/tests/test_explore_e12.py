"""E12 (exploratory; docs/PREREG_A7_STAGE2.md Deviations 2026-10-07 "E12: more training on public tool-use data"):
E1's converter is reused unchanged, the training items render exactly as served, the audit and label rules, configs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_e12_train as E12  # noqa: E402

from jevharness.bench.suites import static  # noqa: E402

# sha256 over E1's 3,991 items (id, source, dp, state, question, options, descriptions, gold, tools, meta) as built by
# static.py before its per-row bodies became bfcl_row_items / when2call_item (commit 96a8249)
E1_FINGERPRINT = "d3b7200a42d02caacdbb235744c1fc569152a80a960adb5209bfc265274a5021"


@pytest.mark.skipif(not (ROOT / "data/raw/bfcl/BFCL_v4_live_multiple.json").exists()
                    or not (ROOT / "data/raw/when2call/test/when2call_test_mcq.jsonl").exists(),
                    reason="E1's raw files are not in this checkout (data/raw is gitignored)")
def test_e1_items_unchanged_by_the_refactor(monkeypatch):
    from jevharness.bench.decision_eval import load_population

    monkeypatch.chdir(ROOT)
    items = load_population(yaml.safe_load((ROOT / "configs/decision/bfcl_when2call_rl.yaml").read_text()))
    h = hashlib.sha256()
    for it in items:
        q = it.question
        h.update(json.dumps([it.id, it.source, it.dp, it.state, q.text, q.options, q.option_descriptions,
                             q.meta.get("gold"), it.gold, it.tools, dict(it.meta)], sort_keys=True,
                            default=str).encode())
    assert len(items) == 3991 and h.hexdigest() == E1_FINGERPRINT


def test_single_call_parser():
    assert E12.parse_single_call('[Get Template List(orderBy="generatedAt")]') == ("Get Template List",
                                                                                    {"orderBy": "generatedAt"})
    assert E12.parse_single_call("[NFTSearch(chain=\"eth\", offset=0.0, flag=true)]") == (
        "NFTSearch", {"chain": "eth", "offset": 0.0, "flag": True})
    assert E12.parse_single_call('[A(x=1), B(y=2)]') is None          # two calls
    assert E12.parse_single_call("I need the page number.") is None
    assert E12.parse_single_call('[A("positional")]') is None


def test_when2call_train_label_rule():
    assert E12.w2c_label("Could you please provide the order id?") == "request_for_info"
    assert E12.w2c_label("To proceed, I'll need the URL of the image.") == "request_for_info"
    assert E12.w2c_label("Apologies, I'm unable to provide that without a location. Please provide it.") == \
        "request_for_info"
    assert E12.w2c_label("Apologies, but I'm unable to perform calculations.") == "cannot_answer"
    assert E12.w2c_label("Here you go.") is None


def test_items_render_as_served_and_e1_format():
    from jevharness.decision.jev import question_to_wire

    tools = [{"name": "get_weather", "description": "Weather for a city.",
              "parameters": {"type": "dict", "properties": {"city": {"type": "string", "description": "City"}},
                             "required": ["city"]}},
             {"name": "get_time", "description": "Time in a zone.", "parameters": {"type": "dict", "properties": {}}}]
    row = {"id": "toolace-x", "question": [[{"role": "user", "content": "What is the weather in Paris today?"}]],
           "function": tools}
    items = static.bfcl_row_items(row, [{"get_weather": {"city": ["Paris"]}}], "toolace", source="toolace")
    assert [i.id for i in items] == ["toolace-x/tool", "toolace-x/param/city"]
    tool = E12.to_row(items[0], "bfcl_tool", "toolace-x")
    wire = question_to_wire(items[0].question)
    assert tool["question"] == wire["instructions"] == "Which function should be called to handle the user's request?"
    assert tool["options"][0] == "get_weather -- Weather for a city."          # serve_jev's "label -- description"
    assert tool["options"][-1].startswith(static.NONE_TOOL)
    assert max(tool["target"], key=tool["target"].get) == tool["options"][0] and sum(tool["target"].values()) == 1
    arg = E12.to_row(items[1], "bfcl_args", "toolace-x")
    assert max(arg["target"], key=arg["target"].get) == "Paris"
    w = static.when2call_item("toolace-x/when2call", tools, "What is the weather in Paris today?", "tool_call",
                              "toolace:w2c", id_prefix="")
    wr = E12.to_row(w, "w2c_call", "toolace-x")
    assert wr["options"] == list(static.W2C_OPTIONS.values())
    assert wr["state"].startswith("AVAILABLE TOOLS:\n- get_weather: Weather for a city.\n    * city (string, required)")
    assert max(wr["target"], key=wr["target"].get) == static.W2C_OPTIONS["tool_call"]


def test_when2call_train_required_moved_inside():
    t = E12.w2c_tools(['{"name": "f", "description": "d", "parameters": {"type": "dict", "properties": '
                       '{"a": {"type": "str"}}}, "required": ["a"]}'])
    assert t[0]["parameters"]["required"] == ["a"] and "required" not in {k for k in t[0] if k != "parameters"}


def test_audit_exact_and_ngram():
    a = E12.Audit({"e1": [E12.request_text("USER: Find the weather in San Diego for tomorrow at noon please"),
                          E12.request_text("AVAILABLE TOOLS:\n- f: g\n\nUSER: short one")]})
    assert a.hits("find the weather in San Diego for tomorrow at noon please!") == {"e1": "exact"}
    assert a.hits("Could you find the weather in San Diego for tomorrow at noon?") == {"e1": "ngram"}
    assert a.hits("Short one") == {"e1": "exact"}
    assert a.hits("a different request about stock prices entirely") == {}


def test_dev_rule_is_deterministic_and_about_ten_percent():
    units = [f"toolace-{i:012x}" for i in range(5000)]
    share = sum(E12.is_dev(u) for u in units) / len(units)
    assert 0.08 < share < 0.12 and E12.is_dev(units[3]) == E12.is_dev(units[3])


def test_quotas_and_ratio():
    assert sum(E12.QUOTA_TRAIN.values()) == E12.N_REPLAY == 7000      # new : replay = 1 : 1
    assert sum(E12.QUOTA_DEV.values()) == 1000


def test_configs_match_e1_and_the_frozen_live_row():
    e1 = yaml.safe_load((ROOT / "configs/decision/bfcl_when2call_rl.yaml").read_text())
    e12 = yaml.safe_load((ROOT / "configs/explore/e12_train/e1_e12.yaml").read_text())
    for k in ("suites", "exclude_ids_file", "batch_by_state", "workers"):
        assert e12[k] == e1[k]
    assert e12["s1_cache"] not in (e1["s1_cache"], "cache/s1_stage2.sqlite")
    assert e12["backends"]["e12_mixed"]["name"] == E12.SERVED
    assert e12["backends"]["frozen_rehost"]["name"] == e1["backends"]["jevagent_rl"]["name"] == "jevagentrl-4b"
    live = yaml.safe_load((ROOT / "configs/explore/e12_train/live.yaml").read_text())
    ref = yaml.safe_load((ROOT / "configs/experiments/stage2_rl_states.yaml").read_text())
    for k in ("suite", "n_tasks", "llm"):
        assert live[k] == ref[k]
    assert live["workers"] == 4 and live["n_tasks"] == E12.LIVE_N == 200
    assert live["arms"] == [{"name": "e12_ts_replace_f", "s1": "e12", "tool_select": {"mode": "replace"}}]
    assert live["s1_cache"] != ref["s1_cache"]
    smoke = yaml.safe_load((ROOT / "configs/explore/e12_train/live_smoke.yaml").read_text())
    assert smoke["suite"]["seed0"] + smoke["suite"]["n"] <= 1000 and smoke["arms"] == live["arms"]
    assert {k: v for k, v in smoke["suite"].items() if k not in ("n", "seed0")} == \
        {k: v for k, v in live["suite"].items() if k not in ("n", "seed0")}


def test_committed_build_record():
    p = ROOT / "results/explore/e12/build_info.json"
    if not p.exists():
        pytest.skip("no build record yet")
    b = json.loads(p.read_text())
    assert b["quota_train"] == E12.QUOTA_TRAIN and b["n_replay"] == E12.N_REPLAY
    assert b["rows"]["public_train"] == 7000 and b["rows"]["train_mix"] == 14000
    assert b["sources"]["toolace"]["sha256"] == E12.TOOLACE_SHA256
    assert set(b["sha256"]) >= {"train_mix.jsonl.gz", "cal_dev.jsonl.gz", "public_dev.jsonl.gz"}
    assert b["audit"]["units_with_hit"] == sum(v["units_dropped"] for v in b["audit"]["by_source"].values())


# ------------------------------------------------------------------------------------------- E12b: random rules
HARD = dict(families="multi", n_distractors=60, noise_lines=20, strict=True, distractor_style="near")


def _play_gold(e):
    e.reset()
    ans = None
    while True:
        a = e.oracle_action()
        if a is None:
            break
        tool, args = a
        if tool == "finish":
            ans = args["answer"]
            break
        e.step(tool, args)
    return e.evaluate(ans)


def test_random_rule_derives_the_operation_keeps_the_oracle_exact_and_never_reads_as_tier():
    from jevharness.bench.suites import jevgym as J
    from jevharness.bench.suites import jevgym_hard as H

    kinds, n_cancel, n_ship = set(), 0, 0
    for seed in range(322000, 322060):
        e = H.JevGymHardEnv(seed, multi_customers=6, multi_rule="random", **HARD)
        g = J.JevGymEnv(seed, multi_customers=6, **HARD)
        assert e.rule == H.random_rule(seed)                     # one rule per seed, its own stream
        assert [(u["user_id"], o["order_id"]) for _, u, o in e.jobs] == \
            [(u["user_id"], o["order_id"]) for _, u, o in g.jobs]  # the parent's world, customers and orders
        r = e.rule
        kinds.add(r["kind"])
        ts = [r["t"]] if r["kind"] != "tiered" else [v for v in r["t"].values() if v is not None]
        assert ts and all(t in H.RANDOM_GRID and t not in (300.0, 600.0) for t in ts)
        assert r != {"kind": "tiered", "t": {"basic": 300.0, "silver": 600.0, "gold": None}}
        assert e.policy_text() in e.instruction
        for op, u, o in e.jobs:
            assert op == ("cancel" if e.rule_cancels(u, o) else "ship")
            n_cancel += op == "cancel"
            n_ship += op == "ship"
        assert _play_gold(e)["success"]
    assert kinds == {"above", "below", "tiered"} and n_cancel and n_ship


def test_tier_and_amount_rules_unchanged_by_the_random_option():
    from jevharness.bench.suites import jevgym_hard as H

    e = H.JevGymHardEnv(321000, multi_customers=12, multi_rule="tier", **HARD)
    assert e.rule is None and e.policy_text() == (
        "cancel it if the customer is basic tier and the amount is above $300.00, or silver tier and the amount is "
        "above $600.00 (never cancel a gold-tier customer's order); otherwise ship it")
    assert H.JevGymHardEnv(7000, multi_customers=3, multi_rule="amount", **HARD).policy_text() == \
        "cancel it if its amount is above $450.00, otherwise ship it"


# task fingerprints of the existing rules (25 seeds each, from the module before E12b's `random` option, git 5462386):
# the option is strictly additive, so every E7-E7f / E13 / E7d held-out task is byte-identical
HARD_FP = {"e7e_tier_itemized": (310500, dict(multi_rule="tier", answer_style="itemized"), "2b46f44becfbd20d"),
           "e7f_tier_itemized": (310600, dict(multi_rule="tier", answer_style="itemized"), "f10a023d6fff9ce7"),
           "e7fx_tier_itemized": (310680, dict(multi_rule="tier", answer_style="itemized"), "26a2acb7711f1274"),
           "e13_tier_itemized": (311000, dict(multi_rule="tier", answer_style="itemized"), "c68a907f60d1406a"),
           "e7b_tier_total": (310200, dict(multi_rule="tier"), "4cf0734fd1cb75b8"),
           "e7d_heldout_tier": (321000, dict(multi_rule="tier"), "74df2a722fb2f477"),
           "amount": (7000, dict(multi_rule="amount"), "131194e05898f5bd"),
           "given": (7000, dict(multi_rule="given"), "43c0b42a3dd8a628")}


@pytest.mark.parametrize("name", sorted(HARD_FP))
def test_existing_hard_rules_produce_byte_identical_tasks(name):
    from jevharness.bench.suites import jevgym_hard as H

    s0, kw, want = HARD_FP[name]
    base = dict(families="multi", multi_customers=12, distractor_style="near", n_distractors=60, noise_lines=20,
                strict=True)
    h = hashlib.sha256()
    for seed in range(s0, s0 + 25):
        e = H.JevGymHardEnv(seed, **{**base, **kw})
        h.update(json.dumps([e.task_id, e.instruction, e._plan(), [(op, u["user_id"], o["order_id"])
                                                                     for op, u, o in e.jobs]], sort_keys=True).encode())
    assert h.hexdigest()[:16] == want


def test_e12c_configs_match_e12_except_the_model():
    e1 = yaml.safe_load((ROOT / "configs/decision/bfcl_when2call_rl.yaml").read_text())
    c = yaml.safe_load((ROOT / "configs/explore/e12_train/e1_e12c.yaml").read_text())
    for k in ("suites", "exclude_ids_file", "batch_by_state", "workers"):
        assert c[k] == e1[k]
    assert list(c["backends"]) == ["e12c_sl4b"] and c["backends"]["e12c_sl4b"]["name"] == "startlux-4b-mixed"
    for a, b in (("live.yaml", "live_e12c.yaml"), ("live_smoke.yaml", "live_e12c_smoke.yaml")):
        x = yaml.safe_load((ROOT / "configs/explore/e12_train" / a).read_text())
        y = yaml.safe_load((ROOT / "configs/explore/e12_train" / b).read_text())
        for k in ("n_tasks", "llm", "workers"):
            assert x[k] == y[k]
        assert {k: v for k, v in x["suite"].items() if k != "seed0"} == {k: v for k, v in y["suite"].items() if k != "seed0"}
        assert y["arms"] == [{"name": "e12c_ts_replace_f", "s1": "e12c", "tool_select": {"mode": "replace"}}]
        assert y["s1_cache"] != x["s1_cache"] and y["name"] != x["name"]
    assert yaml.safe_load((ROOT / "configs/explore/e12_train/live_e12c.yaml").read_text())["suite"]["seed0"] == 300000
    assert E12.EXPS["e12c"]["cap"] == 15.0 and E12.EXPS["e12"]["cap"] == 30.0
