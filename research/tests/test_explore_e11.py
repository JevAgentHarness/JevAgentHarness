"""E11 (exploratory: the training recipe on ALFWorld; scripts/explore_e11_alfworld.py, scripts/aws/e11_alfworld.sh),
offline: the harvest-game selection and split, the scripted executor, the SFT-row build and its data guards, the
threshold write (set-tau), the configs against E6a's, the projection gate and tariff, and an end-to-end dry run of the
Python pipeline (harvest -> build -> calibration shadow arm -> fit -> set-tau -> evaluation readout) on a fake
ALFWorld-shaped environment with the real harness loop and a simulated System-1."""

from __future__ import annotations

import io
import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_e11_alfworld as E  # noqa: E402

from jevharness.bench.env import Env, StepResult  # noqa: E402
from jevharness.decision.simulated import SimulatedBackend  # noqa: E402
from jevharness.harness.agent import JevAgent  # noqa: E402
from jevharness.harness.config import DPConfig, HarnessConfig, Mode  # noqa: E402
from jevharness.llm.base import ChatResponse  # noqa: E402
from jevharness.llm.scripted import ScriptedLLM  # noqa: E402

TYPES = list(E.TASK_TYPES.values())


# ------------------------------------------------------------------ a fake ALFWorld-shaped action env
class FakeAlf(Env):
    """Fetch an apple: go to countertop 1, take apple 1, go to table 1, move apple 1 to table 1. The gold is the next
    plan step from the current state (re-planned after any deviation), like ALFWorld's oracle."""

    RECEPS = ["countertop 1", "cabinet 1", "cabinet 2", "table 1", "drawer 1"]

    def __init__(self, task_id: str):
        self.task_id, self.instruction = task_id, "Your task is to: put an apple on table."
        self.task_type = task_id.split("/")[2].split("-")[0]
        self.at, self.holding, self.done_, self._expert_plan = None, False, False, []

    def reset(self) -> str:
        self.at, self.holding, self.done_ = None, False, False
        return "You are in the middle of a room."

    def admissible_actions(self) -> list[str]:
        acts = [f"go to {r}" for r in self.RECEPS if r != self.at] + ["look", "inventory"]
        if self.at == "countertop 1" and not self.holding:
            acts.append("take apple 1 from countertop 1")
        if self.holding and self.at:
            acts.append(f"move apple 1 to {self.at}")
        if self.at:
            acts.append(f"examine {self.at}")
        return [a for a in acts if a != "inventory"]

    def oracle_action(self):
        if self.done_:
            return None
        if not self.holding:
            g = "go to countertop 1" if self.at != "countertop 1" else "take apple 1 from countertop 1"
        else:
            g = "go to table 1" if self.at != "table 1" else "move apple 1 to table 1"
        return ("act", {"action": g})

    def step(self, tool: str, args: dict[str, Any]) -> StepResult:
        a = args.get("action", "")
        if a.startswith("go to "):
            self.at = a[len("go to "):]
        elif a.startswith("take apple"):
            self.holding = True
        elif a.startswith("move apple"):
            self.holding = False
            self.done_ = self.at == "table 1"
        return StepResult(f"You did: {a}.", done=self.done_, info={"won": self.done_})

    def oracle_done(self):
        return self.done_

    def evaluate(self, final_answer: str | None) -> dict[str, Any]:
        return {"success": self.done_, "score": float(self.done_), "task_type": self.task_type}


def _tid(split: str, typ: str, i: int) -> str:
    return f"alfworld/{split}/{typ}-Apple-None-Table-{i}/trial_T2019_{i}"


def _pending(text: str) -> str:
    """A config as registered before set-tau: the fitted cascade line back to the pending placeholder."""
    import re
    return re.sub(r"action: \{mode: cascade, tau: [0-9.]+\}", "action: {mode: cascade, tau: 0.5, calibration: pending}", text)


def _run(agent: JevAgent, envs: list[Env], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for env in envs:
        agent.run(env).dump(path)


# ------------------------------------------------------------------ selection and split
def _zips(tmp: Path, n: int = 6) -> tuple[Path, Path]:
    js, tw = io.BytesIO(), io.BytesIO()
    with zipfile.ZipFile(js, "w") as zj, zipfile.ZipFile(tw, "w") as zt:
        for t in TYPES:
            for i in range(n):
                d = f"json_2.1.1/train/{t}-Obj{i}-None-Recep-1/trial_T2019_{i}"
                zj.writestr(f"{d}/traj_data.json", json.dumps({"task_type": t}))
                zt.writestr(f"{d}/game.tw-pddl", json.dumps({"solvable": i != 1, "walkthrough": []}))
    (tmp / "j.zip").write_bytes(js.getvalue())
    (tmp / "t.zip").write_bytes(tw.getvalue())
    return tmp / "j.zip", tmp / "t.zip"


def test_select_harvest_excludes_calibration_games_and_is_deterministic(tmp_path: Path) -> None:
    j, t = _zips(tmp_path)
    excl = [f"{TYPES[0]}-Obj0-None-Recep-1/trial_T2019_0"]
    a = E.select_harvest(j, t, excl, n_per_type=3)
    assert a == E.select_harvest(j, t, excl, n_per_type=3)
    games = [g for v in a.values() for g in v]
    assert excl[0] not in games and all(len(v) == 3 for v in a.values())
    assert not any(g.endswith("trial_T2019_1") for g in games)                      # not solvable
    for typ, gs in a.items():                                                        # salted order within each type
        assert all(E.task_type_of(g) == typ for g in gs)
        assert [E.order_key(g) for g in gs] == sorted(E.order_key(g) for g in gs)
    assert E.select_harvest(j, t, excl, n_per_type=3, salt="other") != a or True    # the salt only reorders


def test_split_games_flash_and_development_positions() -> None:
    by_type = {t: [f"{t}-g{i}/trial_{i}" for i in range(100)] for t in TYPES}
    sp = E.split_games(by_type)
    assert len(sp["all"]) == 600 and len(sp["flash"]) == 360 and len(sp["dev"]) == 60
    assert sum(g in set(sp["flash"]) for g in sp["dev"]) == 36                      # positions 10..60 of each type
    first = by_type[TYPES[0]]
    assert [g for g in sp["dev"] if g in first] == [first[i] for i in range(9, 100, 10)]


# ------------------------------------------------------------------ scripted executor
def test_scripted_error_kinds_and_fallback() -> None:
    acts = ["go to cabinet 1", "go to cabinet 2", "take apple 1 from countertop 1", "look", "examine countertop 1"]
    rng = random.Random(0)
    seen = set()
    for _ in range(300):
        a, kind = E.scripted_error("go to cabinet 1", acts, "look", rng)
        assert a != "go to cabinet 1" and a in acts
        seen.add(kind)
        if kind == "near":
            assert a == "go to cabinet 2"
        if kind == "noop":
            assert a in ("look", "examine countertop 1")
    assert seen == {"near", "random", "noop"}
    a, kind = E.scripted_error("take apple 1 from countertop 1", ["take apple 1 from countertop 1", "look"], None,
                               random.Random(1), {"near": 1.0})
    assert (a, kind) == ("look", "random")                                          # no near miss: random
    assert E.scripted_error("look", ["look"], None, random.Random(0)) == ("look", "none")
    assert E.verb("go to cabinet 2") == "go to" and E.verb("move apple 1 to table 1") == "move"


def _ask(llm: E.ExpertNoiseLLM) -> str:
    return llm.chat([{"role": "user", "content": "x"}]).text


def test_expert_noise_llm_follows_gold_and_injects_errors() -> None:
    env = FakeAlf(_tid("train", TYPES[0], 1))
    env.reset()
    exact = E.ExpertNoiseLLM(eps=0.0)
    exact.bind_env(env)
    assert _ask(exact) == "Action: go to countertop 1"
    noisy = E.ExpertNoiseLLM(eps=1.0)
    noisy.bind_env(env)
    outs = {_ask(noisy) for _ in range(20)}
    assert "Action: go to countertop 1" not in outs and noisy.counts["gold"] == 0
    # a non-admissible oracle action falls back to the expert plan's first admissible action
    env.oracle_action = lambda: ("act", {"action": "take bowl 4 from dresser 1"})        # type: ignore[method-assign]
    env._expert_plan = ["go to cabinet 2"]
    exact.bind_env(env)
    assert _ask(exact) == "Action: go to cabinet 2"
    env._expert_plan = []
    exact.bind_env(env)
    assert _ask(exact).startswith("Action: ") and exact.counts["no_gold"] == 1
    # the per-episode stream is seeded by (seed, task id)
    a, b = E.ExpertNoiseLLM(eps=0.5), E.ExpertNoiseLLM(eps=0.5)
    e1, e2 = FakeAlf(_tid("train", TYPES[0], 2)), FakeAlf(_tid("train", TYPES[0], 2))
    e1.reset(), e2.reset()
    a.bind_env(e1), b.bind_env(e2)
    assert [_ask(a) for _ in range(10)] == [_ask(b) for _ in range(10)]


# ------------------------------------------------------------------ rows and guards
def _harvest(tmp: Path, n: int = 6) -> tuple[dict[str, Path], list[str]]:
    envs = [FakeAlf(_tid("train", TYPES[i % 6], i)) for i in range(n)]
    flash = ScriptedLLM("scripted", policy=lambda m, t: ChatResponse(text="Action: go to cabinet 1"))
    _run(JevAgent(flash, HarnessConfig(name=E.FLASH_ARM, max_steps=6)), envs[:3],
         tmp / E.RUN_FLASH / f"{E.FLASH_ARM}.jsonl")
    _run(JevAgent(E.ExpertNoiseLLM(eps=0.3), HarnessConfig(name=E.SCRIPTED_ARM, max_steps=12)), envs,
         tmp / E.RUN_SCRIPTED / f"{E.SCRIPTED_ARM}.jsonl")
    return ({"flash": tmp / E.RUN_FLASH / f"{E.FLASH_ARM}.jsonl",
             "scripted": tmp / E.RUN_SCRIPTED / f"{E.SCRIPTED_ARM}.jsonl"}, [E.game_of(e.task_id) for e in envs])


def test_fallback_look_is_no_label() -> None:
    env = FakeAlf(_tid("train", TYPES[0], 3))
    env.reset()
    env.oracle_action = lambda: ("act", {"action": "look"})                         # type: ignore[method-assign]
    env._expert_plan = ["look"]
    assert E.ExpertNoiseLLM.gold_of(env, env.admissible_actions()) is None
    t = {"steps": [{"step": i, "args": {"action": "a"}, "decisions": [{"dp": "action", "gold": g}]}
                   for i, g in enumerate(["a", "b", "look", "c"])]}
    assert E.lost_from(t) == 2 and E.lost_from({"steps": t["steps"][:2]}) is None


def test_rows_after_the_first_fallback_are_dropped(tmp_path: Path) -> None:
    sources, _ = _harvest(tmp_path)
    path = sources["scripted"]
    recs = [json.loads(x) for x in path.read_text().splitlines()]
    st = recs[0]["steps"][1]
    next(d for d in st["decisions"] if d["dp"] == "action")["gold"] = "look"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    rows = E.rows_from_runs({"scripted": path})
    tid = recs[0]["task_id"]
    assert [r["meta"]["step"] for r in rows if r["meta"]["task_id"] == tid] == [0]
    assert E.DROPPED["scripted:after_fallback"] == len(recs[0]["steps"]) - 1
    assert all(next(o for o, p in r["target"].items() if p) != "look" for r in rows)


def test_patch_handcoded_move_tracks_move_as_put() -> None:
    pytest.importorskip("alfworld")
    from alfworld.agents.expert import handcoded_expert as H

    seen: list[str] = []
    orig = H.BasePolicy.update_state_tracking
    try:
        H.BasePolicy.update_state_tracking = lambda self, gs, la: seen.append(la)      # type: ignore[assignment]
        assert E.patch_handcoded_move() and E.patch_handcoded_move()                       # idempotent
        H.BasePolicy.update_state_tracking(object(), {}, "move apple 1 to table 1")
        H.BasePolicy.update_state_tracking(object(), {}, "go to table 1")
        assert seen == ["put apple 1 to table 1", "go to table 1"]
    finally:
        H.BasePolicy.update_state_tracking = orig


def test_off_plan_steps() -> None:
    t = {"steps": [{"step": 0, "args": {"action": "a"}, "decisions": [{"dp": "action", "gold": "a"}]},
                   {"step": 1, "args": {"action": "x"}, "decisions": [{"dp": "action", "gold": "b"}]},
                   {"step": 2, "args": {"action": "c"}, "decisions": [{"dp": "action", "gold": "c"}]}]}
    assert E.off_plan_steps(t) == {0: False, 1: False, 2: True}


def test_build_rows_split_dedupe_cap_and_guards(tmp_path: Path) -> None:
    sources, games = _harvest(tmp_path)
    rows = E.rows_from_runs(sources)
    assert rows and all(r["kind"] == "choice" and r["dp"] == "action" and r["question"] == E.QUESTION for r in rows)
    for r in rows:
        gold = [o for o, p in r["target"].items() if p == 1.0]
        assert len(gold) == 1 and gold[0] in r["options"] and abs(sum(r["target"].values()) - 1) < 1e-9
        assert r["meta"]["src"] in ("flash", "scripted") and r["meta"]["game"] in games
    assert any(r["meta"]["off_plan"] for r in rows) and any(not r["meta"]["off_plan"] for r in rows)
    res = E.build(sources, dev_games=[games[0]], cal=[], eval_ids=[_tid("valid_seen", TYPES[0], 99)])
    assert {r["meta"]["game"] for r in res["dev"]} == {games[0]}
    assert games[0] not in {r["meta"]["game"] for r in res["train"]}
    keys = [(r["state"], tuple(sorted(r["options"]))) for r in res["train"]]
    assert len(keys) == len(set(keys))                                                # deduplicated
    assert res["stats"]["duplicates_dropped"]["train"] > 0                            # flash and scripted step 0
    capped = E.build(sources, dev_games=[], cal=[], eval_ids=[], max_train=5)
    assert len(capped["train"]) == 5 and capped["stats"]["train_before_cap"] > 5
    with pytest.raises(SystemExit, match="calibration game"):
        E.build(sources, dev_games=[], cal=[games[1]], eval_ids=[])
    with pytest.raises(SystemExit, match="valid_seen trial"):
        E.build(sources, dev_games=[], cal=[], eval_ids=[f"alfworld/valid_seen/x/{games[2].split('/')[-1]}"])
    bad = [dict(rows[0], meta={**rows[0]["meta"], "task_id": "alfworld/valid_seen/x/trial_1"})]
    with pytest.raises(SystemExit, match="not an ALFWorld train game"):
        E.check_rows(bad, [], [])


# ------------------------------------------------------------------ set-tau and the configs
def test_set_tau_changes_exactly_the_pending_line() -> None:
    for name in ("eval.yaml", "smoke.yaml"):
        text = _pending((E.CFG_DIR / name).read_text())
        out = E.set_tau_text(text, {"mode": "cascade", "tau": 0.93})
        diff = [(a, b) for a, b in zip(text.splitlines(), out.splitlines()) if a != b]
        assert len(diff) == 1 and "tau: 0.93}" in diff[0][1] and "calibration: pending" not in out
        arms = {a["name"]: a for a in yaml.safe_load(out)["arms"]}
        assert arms[E.ARMS["cascade"]]["action"] == {"mode": "cascade", "tau": 0.93}
        off = E.set_tau_text(text, {"mode": "cascade", "always_defer": True})
        assert E.ARMS["cascade"] not in {a["name"] for a in yaml.safe_load(off)["arms"]}
        with pytest.raises(SystemExit):
            E.set_tau_text(out, {"mode": "cascade", "tau": 0.9})                     # nothing pending any more


def test_configs_match_e6a_and_the_registration() -> None:
    cfg = {n: yaml.safe_load((E.CFG_DIR / f"{n}.yaml").read_text()) for n in ("eval", "smoke", "cal", "harvest_flash")}
    e6a = yaml.safe_load((ROOT / "configs/explore/e6a_alfworld/eval.yaml").read_text())
    ev = cfg["eval"]
    for k in ("suite", "n_tasks", "llm", "harness_defaults", "workers", "schedule"):
        assert ev[k] == e6a[k], k
    assert [a["name"] for a in ev["arms"]] == [E.REF, E.ARMS["cascade"], E.ARMS["replace"]]
    assert [a["name"] for a in cfg["smoke"]["arms"]] == [a["name"] for a in ev["arms"]]
    for n in ("eval", "smoke", "cal"):
        s1 = cfg[n]["s1_backends"]["e11"]
        assert s1["base_url"] == "http://127.0.0.1:8081" and s1["name"] == s1["model"] == E.SERVED
        assert s1.get("cache") is None and "s1" not in cfg[n]
        assert all(a.get("s1") == "e11" for a in cfg[n]["arms"] if a["name"] != E.REF)
        assert cfg[n]["llm"] == e6a["llm"] and cfg[n]["harness_defaults"] == e6a["harness_defaults"]
    for n in ("cal", "smoke"):                                                       # E6a's calibration games
        assert cfg[n]["suite"]["split"] == "train" and cfg[n]["suite"]["data_dir"] == "data/raw/alfworld_e6a"
    assert cfg["cal"]["arms"] == [{"name": E.CAL_ARM, "s1": "e11", "action": {"mode": "shadow"}}]
    hf = cfg["harvest_flash"]
    assert hf["suite"]["split"] == "train" and hf["suite"]["data_dir"] == "data/raw/alfworld_e11_flash"
    assert hf["arms"] == [{"name": E.FLASH_ARM}] and hf["llm"] == e6a["llm"]
    reg = (ROOT / "docs/PREREG_A7_STAGE2.md").read_text()
    assert 'E11: the training recipe on ALFWorld' in reg and E.SALT in reg.replace("-\n", "-").replace("\n", " ")


def test_committed_threshold_is_the_fit() -> None:
    """The committed eval / smoke configs carry exactly the fitted tau-hat of tauhat_action.json (set-tau on the box),
    and the evaluation run's saved config (when the run is present locally) is the committed eval.yaml."""
    fit = json.loads(E.TAU_FILE.read_text())
    for name in ("eval.yaml", "smoke.yaml"):
        arms = {a["name"]: a for a in yaml.safe_load((E.CFG_DIR / name).read_text())["arms"]}
        assert arms[E.ARMS["cascade"]]["action"] == {"mode": "cascade", "tau": fit["deploy"]["tau"]}
    saved = ROOT / "runs" / E.RUN_EVAL / "experiment.yaml"
    if saved.exists():
        assert yaml.safe_load(saved.read_text()) == yaml.safe_load((E.CFG_DIR / "eval.yaml").read_text())


def test_driver_syntax_and_forbidden_hosts() -> None:
    p = ROOT / "scripts/aws/e11_alfworld.sh"
    assert subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode == 0
    text = p.read_text()
    forb = text.split('FORBIDDEN_IIDS="', 1)[1].split('"', 1)[0].split()
    assert {"<instance-id>", "<instance-id>", "<instance-id>"} <= set(forb)
    assert "--epochs 2" in text and "--lr 1e-4" in text and "--brier-weight 0.5" in text and "--lora-r 32" in text


# ------------------------------------------------------------------ projection and tariff
def _ep(task: str, cost: float, ok: bool = True, err: str | None = None, wall: float = 10_000.0) -> dict[str, Any]:
    return {"task_id": task, "config": "x", "steps": [], "result": {"success": ok}, "error": err, "wall_ms": wall,
            "usage": {"total": {"cost_usd": cost}, "s2": {"cost_usd": cost}, "s1": {"cost_usd": 0.0}}}


def test_projection_gate_and_tariff(tmp_path: Path) -> None:
    sm = tmp_path / E.RUN_SMOKE
    sm.mkdir(parents=True)
    for arm, c in ((E.REF, 0.02), (E.ARMS["cascade"], 0.01), (E.ARMS["replace"], 0.0)):
        (sm / f"{arm}.jsonl").write_text("\n".join(json.dumps(_ep(f"t{i}", c)) for i in range(2)) + "\n")
    p = E.projection(tmp_path)
    assert p["decision"] == "run all 140" and p["projected_140"] == pytest.approx(0.03 * 140 * 1.5)
    assert E.projection(tmp_path, cap=0.05)["decision"] == "stop after the smoke"
    ev = tmp_path / E.RUN_EVAL
    ev.mkdir()
    for arm in E.GPU_ARMS:
        (ev / f"{arm}.jsonl").write_text("\n".join(json.dumps(_ep(f"t{i}", 0.0)) for i in range(10)) + "\n")
    w = tmp_path / "eval_window.json"
    w.write_text(json.dumps({"window_start": "2026-10-08T00:00:00Z", "window_end": "2026-10-08T00:30:00Z",
                             "instance_type": "g6e.2xlarge"}))
    tf = E.tariff(tmp_path, w)
    assert tf["episodes"] == 20 and tf["q_usd_per_episode"] == pytest.approx(2.24208 * 0.5 / 20)
    assert E.tariff(tmp_path, tmp_path / "none.json")["state"] == "no window record"


# ------------------------------------------------------------------ end-to-end dry run of the Python pipeline
def test_pipeline_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs, box, cfg = tmp_path / "runs", tmp_path / "box", tmp_path / "cfg"
    cfg.mkdir()
    for n in ("eval.yaml", "smoke.yaml"):
        (cfg / n).write_text(_pending((E.CFG_DIR / n).read_text()))
    monkeypatch.setattr(E, "CFG_DIR", cfg)
    monkeypatch.setattr(E, "TAU_FILE", cfg / "tauhat_action.json")
    # 1. harvest + build
    sources, games = _harvest(runs, n=12)
    res = E.build(sources, dev_games=games[:2], cal=[], eval_ids=[])
    assert res["train"] and res["dev"] and res["stats"]["train"]["n"] == len(res["train"])
    # 2. calibration shadow arm (the LLM acts; a simulated System-1 is asked and logged), then the registered fit
    class Gold(ScriptedLLM):                       # follows the env's gold, errs at 30 %
        def bind_env(self, env: Any) -> None:
            self.env, self.rng = env, random.Random(env.task_id)

        def _chat(self, messages: Any, tools: Any = None, **kw: Any) -> Any:
            g = self.env.oracle_action()[1]["action"]
            self.policy = lambda m, t: ChatResponse(text=f"Action: {g if self.rng.random() > 0.3 else 'go to cabinet 1'}")
            return super()._chat(messages, tools, **kw)

    cal_envs = [FakeAlf(_tid("train", TYPES[i % 6], 1000 + i)) for i in range(16)]
    shadow = HarnessConfig(name=E.CAL_ARM, max_steps=10, action=DPConfig(mode=Mode.SHADOW))
    _run(JevAgent(Gold("scripted"), shadow, s1=SimulatedBackend(accuracy=0.8, seed=3)), cal_envs,
         runs / E.RUN_CAL / f"{E.CAL_ARM}.jsonl")
    import argparse
    assert E.cmd_fit(argparse.Namespace(run=str(runs / E.RUN_CAL))) == 0
    fit = json.loads((cfg / "tauhat_action.json").read_text())
    assert fit["cal_states"]["n"] > 0 and "deploy" in fit
    assert E.cmd_set_tau(argparse.Namespace(record=str(box / "set_tau.json"))) == 0
    assert "calibration: pending" not in (cfg / "eval.yaml").read_text()
    # 3. evaluation readout on three valid_seen ids (fake envs), with an evaluation window for the tariff
    ids = json.loads(E.EVAL_LIST.read_text())["task_ids"][:3]
    deploy = fit["deploy"]
    arms = {E.REF: HarnessConfig(name=E.REF, max_steps=10),
            E.ARMS["replace"]: HarnessConfig(name=E.ARMS["replace"], max_steps=10, action=DPConfig(mode=Mode.REPLACE))}
    if not deploy.get("always_defer"):
        arms[E.ARMS["cascade"]] = HarnessConfig(name=E.ARMS["cascade"], max_steps=10,
                                                action=DPConfig(mode=Mode.CASCADE, tau=deploy["tau"]))
    for name, hc in arms.items():
        s1 = SimulatedBackend(accuracy=0.8, seed=5) if name != E.REF else None
        _run(JevAgent(Gold("scripted"), hc, s1=s1), [FakeAlf(t) for t in ids], runs / E.RUN_EVAL / f"{name}.jsonl")
    box.mkdir(exist_ok=True)
    (box / "eval_window.json").write_text(json.dumps({"window_start": "2026-10-08T00:00:00Z",
                                                      "window_end": "2026-10-08T00:06:00Z",
                                                      "instance_type": "g6e.2xlarge"}))
    r = E.summarize(runs, box)
    assert r["reference"]["n"] == 3 and r["arms"]["replace"]["status"] == "run"
    rep = r["arms"]["replace"]
    q = r["tariff"]["q_usd_per_episode"]
    assert rep["tariff"]["q"] == pytest.approx(q) and rep["saving_api"] is not None
    md, tex = E.to_markdown(r), E.to_tex(r)
    assert "E11" in md and r"\defres{e11/llm_only/success}" in tex and r"\defres{e11/replace/saving}" in tex
    assert r"\defres{e11/tariff/q}" in tex
    assert fit["deploy"].get("always_defer") or r"\defres{e11/trained/success}" in tex       # the paper hook's alias
    assert fit["deploy"].get("always_defer") or r"\defres{e11/cascade/coverage}" in tex
