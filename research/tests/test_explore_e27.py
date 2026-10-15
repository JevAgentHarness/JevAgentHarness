"""Exploratory row E27 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 "E27: a 0.8B decision model that holds under
shift"): the surface augmentation (consistent renames, rewording, near copies, labels moved only), the vocabulary
disjointness check against E21's split and re-renderings, the configurations and profile as registered, the driver's
guards and the selection rule. Offline; no model or API call."""
from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import explore_e27_small_shift as X  # noqa: E402

from jevharness.training import clef_data as CD  # noqa: E402
from jevharness.training import surface_aug as SA  # noqa: E402

CFG = ROOT / "configs/explore/e27_small_shift"
TRAIN = ROOT / "data/train_v1/train.jsonl.gz"


def _rows(n: int = 4000) -> list[dict]:
    out = []
    with gzip.open(TRAIN, "rt") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            out.append(json.loads(line))
    return out


@pytest.fixture(scope="module")
def rows():
    return _rows()


def test_vocabulary_disjoint_from_e21_and_its_rerenderings():
    aug = SA.aug_vocabulary()
    e21, info = SA.e21_vocabulary([ROOT / "src/jevharness/bench/suites/jevgym_shift.py",
                                   ROOT / "src/jevharness/bench/suites/jevgym.py"], ROOT / "results/explore/e21_shift/c")
    assert {Path(c["path"]).name for c in info["corpus_files"]} >= {"questions.jsonl.gz", "questions_nosurf.jsonl.gz",
                                                                    "questions_nodist.jsonl.gz"}
    # E21's aliases, near-duplicates and the generator's own names are in the E21 vocabulary ...
    for w in ("lookup", "customer", "dispatch", "reimburse", "clarification", "cancel", "order", "ship", "refund"):
        assert w in e21
    r = SA.disjointness(aug, e21)
    assert r["ok"] and r["overlap_count"] == 0, (r["overlap_tokens"], r["overlap_stems"])
    # ... and the stem rule catches inflections
    assert SA.stem("shipped") == SA.stem("ship") and SA.stem("bookings") == SA.stem("book")
    assert not SA.disjointness({"shipping"}, {"shipped"})["ok"]


def test_template_fields_are_world_values_only():
    allowed = {"c", "o", "os", "p", "ask", "nm", "oid", "item", "steps", "speed", "V_speed", *SA.FIELDS}
    assert SA.template_fields() <= allowed
    assert set(SA.FIELDS.values()) <= {"user_id", "tier", "status", "delivered", "store_credit", "original_payment",
                                      "express", "standard", "gold"}


def test_selection_is_half_of_the_states_and_deterministic(rows):
    states = sorted({r["state"] for r in rows})
    share = sum(SA.state_selected(s) for s in states) / len(states)
    assert 0.4 < share < 0.6
    s = states[0]
    a, b = SA.Aug(s), SA.Aug(s)
    assert a.names == b.names and a.descs == b.descs
    assert len(set(a.names.values())) == len(SA.CORE)


def test_augmentation_renames_consistently_and_moves_only_the_label(rows):
    out, st = SA.augment_rows(rows)
    assert len(out) == len(rows) and st["augmented_rows"] > 0
    assert st["task_forms"].get("kept", 0) == 0          # every request form of the generator is reworded
    n_tool = 0
    for a, b in zip(rows, out):
        assert a["id"] == b["id"] and a["dp"] == b["dp"]
        info = (b.get("meta") or {}).get("e27_aug")
        if not info:
            assert a == b or not SA.is_agent_row(a)
            continue
        names = info["names"]
        text = b["state"] + "\n" + b["question"]
        for t in SA.CORE:
            if t == "refund":
                assert not re.search(r"(?<![\w])refund\(", text) and "`refund`" not in text
            else:
                assert not re.search(rf"(?<![\w]){t}(?![\w])", text), (t, b["id"])
        assert "Policy: never guess" not in b["state"]
        # the answer key is only moved: same target values, the gold option renamed
        assert sorted(a["target"].values()) == sorted(v for o, v in b["target"].items()
                                                     if not (o not in a["options"] and o.split(SA.SEP)[0]
                                                             not in names.values() and v == 0.0))
        assert abs(sum(a["target"].values()) - sum(b["target"].values())) < 1e-9
        ga = max(a["target"], key=a["target"].get)
        gb = max(b["target"], key=b["target"].get)
        assert a["target"][ga] == b["target"][gb]
        if a["dp"] == "tool_select" and a.get("kind") == "choice":
            n_tool += 1
            la, lb = ga.split(SA.SEP)[0], gb.split(SA.SEP)[0]
            assert lb == names.get(la, la)
            assert len(set(b["options"])) == len(b["options"])
            assert 0 <= len(b["options"]) - len(a["options"]) <= SA.N_NEAR[1]
            for o in b["options"]:
                if o not in a["options"] and o.split(SA.SEP)[0] not in names.values():
                    assert b["target"][o] == 0.0          # a near copy is never acceptable
        else:
            assert len(b["options"]) == len(a["options"])
    assert n_tool > 0


def test_argument_rows_of_a_step_stay_one_record(rows):
    out, _ = SA.augment_rows(rows)
    assert len(CD.rows_to_records(out)) == len(CD.rows_to_records(rows))


def test_every_generator_request_form_parses():
    from jevharness.bench.suites.jevgym import JevGymEnv

    seen = set()
    for fam in ("cancel", "refund", "stock", "ship", "total", "clarify", "multi"):
        for seed in range(12000, 12040):
            env = JevGymEnv(seed, family=fam, ask_tool=fam in ("clarify", "multi"))
            _, form = SA.Aug("x").task(env.instruction)
            assert form != "kept", (fam, env.instruction)
            seen.add(form)
    assert seen >= {"cancel", "refund_credit", "stock", "ship_policy", "total", "multi", "refund_method", "ship_speed",
                    "ship_name"}


def test_public_rows_repeat_with_distinct_ids(tmp_path):
    import e27_clef as T

    pub = [{"id": f"toolace-{i}/param/x", "dp": "tool_params", "kind": "choice", "state": "USER: hi", "question": "q",
            "options": ["a", "b"], "target": {"a": 1.0, "b": 0.0}, "meta": {"e12": "bfcl_args"}} for i in range(3)]
    rep = {"id": "v1-row", "meta": {"e12": "replay_v1"}}
    p = tmp_path / "mix.jsonl.gz"
    with gzip.open(p, "wt") as f:
        for r in pub + [rep]:
            f.write(json.dumps(r) + "\n")
    T.E12_MIX_SHA = T.E14.sha256(p)
    rows, info = T.public_rows(p, 7)
    assert len(rows) == 7 and info["full_passes"] == 2 and info["sampled"] == 1
    assert len({r["id"] for r in rows}) == 7
    recs, bad = T.to_records(rows)
    assert len(recs) == 7 and bad == 0


def _cfg(name: str) -> dict:
    return yaml.safe_load((CFG / name).read_text())


def test_configs_as_registered():
    e21 = yaml.safe_load((ROOT / "configs/explore/e21_shift/live_e14a.yaml").read_text())
    for arm, sm in (("s", 339040), ("sp", 339042)):
        live, smoke = _cfg(f"live_shift_{arm}.yaml"), _cfg(f"smoke_shift_{arm}.yaml")
        assert live["suite"] == e21["suite"] and live["workers"] == 4 and live["n_tasks"] == 200
        assert smoke["suite"]["seed0"] == sm and smoke["suite"]["n"] == 2 and smoke["suite"]["shift"] is True
        for c in (live, smoke):
            (be, spec), = c["s1_backends"].items()
            assert spec["model"] == f"e27-{arm}" and spec["cache"] is None and "s1_cache" not in c
            assert c["arms"] == [{"name": f"e27{arm}_ts_replace_f", "s1": be, "tool_select": {"mode": "replace"}}]
    e14 = yaml.safe_load((ROOT / "configs/explore/e14_small/live_a.yaml").read_text())
    li = _cfg("live_in.yaml")
    assert li["suite"] == e14["suite"] and li["n_tasks"] == 200 and li["workers"] == 4
    assert _cfg("smoke_in.yaml")["suite"]["seed0"] == 940
    b = yaml.safe_load((ROOT / "configs/explore/e14_small/live_b.yaml").read_text())
    fi, fs = _cfg("final_in.yaml"), _cfg("final_shift.yaml")
    assert fi["suite"] == b["suite"] and fs["suite"] == e21["suite"]
    for c in (fi, fs, _cfg("smoke_final_in.yaml"), _cfg("smoke_final_shift.yaml")):
        assert c["workers"] == 1 and c["schedule"] == {"mode": "task_major", "seed": 732}
        arm, = c["arms"]
        assert arm["profile"] == "configs/harness/explore_e27_final.yaml" and arm["profile_backend"] == "e27"
    assert _cfg("smoke_final_in.yaml")["suite"]["seed0"] == 942 and _cfg("smoke_final_shift.yaml")["suite"]["seed0"] == 339044
    for name in ("smoke_shift_s", "smoke_shift_sp", "smoke_final_shift"):
        s0 = _cfg(f"{name}.yaml")["suite"]["seed0"]
        assert 339000 <= s0 <= 339098
    for name in ("smoke_in", "smoke_final_in"):
        assert _cfg(f"{name}.yaml")["suite"]["seed0"] + 1 <= 999
    e1 = _cfg("e1_e27.yaml")
    e14e1 = yaml.safe_load((ROOT / "configs/explore/e14_small/e1_e14.yaml").read_text())
    for k in ("suites", "exclude_ids_file", "batch_by_state", "workers"):
        assert e1[k] == e14e1[k]
    assert set(e1["backends"]) == {"e27s", "e27sp"}


def test_profile_pending_or_fitted():
    from jevharness.harness.profile import load_profile, resolve_arm

    prof = load_profile(X.PROFILE)
    base = load_profile(X.FINAL_PROFILE)
    sec, bsec = prof["system2"]["gemini-3.5-flash"], base["system2"]["gemini-3.5-flash"]
    assert sec["decision_points"] == bsec["decision_points"]
    assert sec["backends"]["jevagent_rl"] == bsec["backends"]["jevagent_rl"]
    c = _cfg("final_in.yaml")
    out, meta = resolve_arm(c["arms"][0], c)
    if prof["status"] == "pending":
        assert meta["skip"]
        return
    blk = sec["backends"]["e27"]
    fits = json.loads((X.CAL_OUT / "fits.json").read_text())
    for dp in X.DPS:
        f = blk["fitted_from"][dp]
        assert X.sha256_file(ROOT / f["path"]) == f["sha256"]
        t = fits["fits"][dp]["tau_hat"]
        if blk[dp].get("always_defer"):
            assert t is None or t > 1.0
        else:
            assert blk[dp]["tau"] == pytest.approx(float(t))


def test_driver_syntax_and_guards():
    p = ROOT / "scripts/aws/e27_small_shift.sh"
    subprocess.run(["bash", "-n", str(p)], check=True)
    txt = p.read_text()
    forb = txt.split("FORBIDDEN_IIDS=")[1].split("\n")[0]
    for iid in ("<instance-id>", "<instance-id>", "<instance-id>", "<instance-id>",
                "<instance-id>", "<instance-id>"):
        assert iid in forb
    others = set()
    for d in (ROOT / "scripts/aws").glob("*.sh"):
        if d.name != p.name:
            others |= set(re.findall(r"(?<![A-Za-z0-9])i-0[0-9a-f]{16}", d.read_text()))
    assert others <= set(forb.split('"')[1].split())
    assert "TAG=jev-e27-08b" in txt and "GUARD_MIN=${GUARD_MIN:-590}" in txt and "CAP_USD=${CAP_USD:-30}" in txt
    assert X.CAL_REQ_SHA in txt and "OPENROUTER" not in txt


def test_selection_rule(monkeypatch, tmp_path):
    def fake_l2(acc):
        return lambda cdir: {"rows": {k: {"main": {"all": {"acc": v}, "answers": 1}} for k, v in acc.items()}}

    monkeypatch.setattr(X, "l1_acc", lambda runs, key: {"e27s": 0.70, "e27sp": 0.80}[key])
    monkeypatch.setattr(X, "l2_shift", fake_l2({"e27s": 0.90, "e27sp": 0.85}))
    assert X.select(tmp_path, tmp_path)["selected"] == "S"
    monkeypatch.setattr(X, "l2_shift", fake_l2({"e27s": 0.85, "e27sp": 0.90}))
    assert X.select(tmp_path, tmp_path)["selected"] == "SP"
    monkeypatch.setattr(X, "l2_shift", fake_l2({"e27s": 0.88, "e27sp": 0.88}))
    r = X.select(tmp_path, tmp_path)
    assert r["selected"] == "SP" and r["by"].startswith("L1")
    monkeypatch.setattr(X, "l1_acc", lambda runs, key: 0.7)
    assert X.select(tmp_path, tmp_path)["selected"] == "S"
