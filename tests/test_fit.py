"""``jah fit``: thresholds from logged decisions, robust to small samples."""

from __future__ import annotations

import json
import random

from conftest import FakeGemini, ScriptedDecision, call_reply, start, text_reply
from jevagentharness import fit as F
from jevagentharness.cli import main


def tool_rec(conf: float, agree: bool, fit_ok: bool = True, choice: str = "bash", **kw) -> dict:
    return {"point": "tool_select", "mode": "shadow", "question": "next_tool", "choice": choice, "confidence": conf,
            "label": choice if agree else "other", "fit_ok": fit_ok, "error": None, "tau": 0.99,
            "model_name": "jah-auto", **kw}


def stop_rec(p_done: float, done: bool) -> dict:
    return {"point": "stop", "mode": "shadow", "question": "task_done", "p": {"done": p_done, "not_done": 1 - p_done},
            "p_act": p_done, "act_option": "done", "label": "done" if done else "not_done", "fit_ok": True,
            "error": None, "tau": 0.9, "model_name": "jah-auto"}


def synthetic(n: int = 600, seed: int = 0) -> list[dict]:
    """Confidence c; the decision model agrees with the LLM with probability c**4 (well below 95% until c ~ 0.99)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        c = round(rng.choice([0.6, 0.8, 0.9, 0.95, 0.99, 1.0]), 4)
        out.append(tool_rec(c, rng.random() < (1.0 if c >= 0.99 else c ** 4)))
    return out


def test_wilson_lower_bound():
    assert F.wilson_lower(0, 0, 1.645) == 0.0
    assert F.wilson_lower(10, 10, 0) == 1.0
    assert abs(F.wilson_lower(50, 50, 1.645) - 50 / (50 + 1.645 ** 2)) < 1e-9
    assert F.wilson_lower(95, 100, 1.645) < 0.95 < F.wilson_lower(990, 1000, 1.645)


def test_fit_picks_the_smallest_reliable_threshold():
    res = F.fit(synthetic())
    f = res["points"]["tool_select"]
    assert f["tau"] == 0.99 and f["lower"] >= 0.95 and f["n_accepted"] >= 30
    assert 0.2 < f["coverage"] < 0.5                     # 0.99 and 1.0: two of the six confidence levels
    assert res["points"]["stop"]["tau"] is None and "too few" in res["points"]["stop"]["note"]
    # without the bound, a lower threshold passes on the raw rate
    assert F.fit(synthetic(), z=0.0)["points"]["tool_select"]["tau"] <= 0.99


def test_fit_is_robust_to_small_and_biased_samples():
    few = [tool_rec(1.0, True) for _ in range(12)]
    f = F.fit(few)["points"]["tool_select"]
    assert f["tau"] is None and "too few" in f["note"]
    # decisions the cascade acted on (fit_ok false) and "no tool" answers are not labels
    acted = [tool_rec(1.0, False, fit_ok=False) for _ in range(100)] + [
        tool_rec(1.0, True, choice="reply_without_tool") for _ in range(100)]
    assert F.fit(acted)["points"]["tool_select"]["n_labelled"] == 100
    assert F.fit(acted)["points"]["tool_select"]["tau"] is None
    # agreement never reaches the target
    bad = [tool_rec(1.0, i % 5 != 0) for i in range(500)]
    f = F.fit(bad)["points"]["tool_select"]
    assert f["tau"] is None and "no threshold" in f["note"]


def test_fit_leaves_out_single_tool_requests():
    # mini-swe-agent offers one tool (bash): "agreement" there only says the agent calls it
    one = [tool_rec(1.0, True, options=["bash", "reply_without_tool"], n_options=2) for _ in range(200)]
    f = F.fit(one)["points"]["tool_select"]
    assert f["tau"] is None and f["n_labelled"] == 0 and f["single_tool"] == 200 and "single tool" in f["note"]
    many = [tool_rec(1.0, True, options=["read", "bash", "edit", "reply_without_tool"], n_options=4)
            for _ in range(100)]
    f = F.fit(one + many)["points"]["tool_select"]
    assert f["tau"] == 1.0 and f["n_labelled"] == 100 and f["single_tool"] == 200
    assert f["note"] == "200 single-tool decisions left out"
    # with tool_choice required, two tools are a real choice
    req = [tool_rec(1.0, True, options=["bash", "edit"], n_options=2) for _ in range(40)]
    assert F.fit(req)["points"]["tool_select"]["n_labelled"] == 40


def test_fit_stop():
    recs = [stop_rec(0.97, True) for _ in range(80)] + [stop_rec(0.7, i % 2 == 0) for i in range(80)]
    f = F.fit(recs)["points"]["stop"]
    assert f["tau"] == 0.97 and f["agreement"] == 1.0 and f["coverage"] == 0.5


def test_fit_rounds_up_to_a_round_threshold():
    recs = [tool_rec(0.9834, True) for _ in range(80)] + [tool_rec(0.5, False) for _ in range(80)]
    assert F.fit(recs)["points"]["tool_select"]["tau"] == 0.99 or \
        F.fit(recs)["points"]["tool_select"]["tau"] == 0.9834


def test_fit_cli_writes_a_snippet(tmp_path, capsys):
    (tmp_path / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in synthetic()) + "\n")
    out = tmp_path / "fit.yaml"
    assert main(["fit", str(tmp_path), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "| tool_select |" in printed and "tool_select: {tau: 0.99}" in printed
    text = out.read_text()
    assert text.startswith("# jah fit") and "decisions:" in text and "# stop: too few" in text
    assert main(["fit", str(tmp_path / "missing")]) == 1
    assert main(["fit", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["points"]["tool_select"]["tau"] == 0.99


def test_shadow_run_then_fit_end_to_end(run, tmp_path, capsys):
    """The shadow preset on the default router: every point logs, the strong tier labels the fast finals."""
    def script(model, body):
        n = sum(1 for c in body["contents"] if c["role"] == "model")
        if model == "gemini-3.8-flash" and n % 4 == 3:
            return 200, text_reply("done")
        return 200, call_reply("bash", {"command": f"step {n}"}, sig=None)

    fake = FakeGemini(script)
    dec = ScriptedDecision({"next_tool": {"bash": 0.98, "reply_without_tool": 0.02},
                            "task_done": {"done": 0.3, "not_done": 0.7}})

    async def go():
        client, up, proxy = await start(fake, tmp_path, dec, sets=["preset=shadow"])
        tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}},
                 {"type": "function", "function": {"name": "edit", "parameters": {"type": "object"}}}]
        try:
            for n in range(1, 25):
                msgs = [{"role": "user", "content": "fix it"}]
                for i in range(n):
                    msgs += [{"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function",
                              "function": {"name": "bash", "arguments": "{}"}}]},
                             {"role": "tool", "tool_call_id": f"c{i}", "content": "ok"}]
                await client.post("/v1/chat/completions", json={"model": "jah-auto", "messages": msgs,
                                                                 "tools": tools})
        finally:
            await client.close()
            await up.close()

    run(go())
    recs = F.read_decisions(tmp_path)
    tools = F.labelled(recs, "tool_select")
    stops = F.labelled(recs, "stop")
    assert len(tools) == 24 and all(r["fit_ok"] for r in tools)
    assert stops and all(r["label"] == "not_done" for r in stops)  # the fake strong tier goes on working
    assert sum(r["label"] == "reply_without_tool" for r in tools) == 6      # the fast tier's own final answers
    assert main(["fit", str(tmp_path), "--min-count", "5", "--z", "0"]) == 0
    assert "tool_select: no threshold reached 95%" in capsys.readouterr().out   # 18 of 24 agree
    assert main(["fit", str(tmp_path), "--min-count", "5", "--z", "0", "--target", "0.7"]) == 0
    printed = capsys.readouterr().out
    assert "tool_select: {tau: 0.98}" in printed
    assert main(["report", str(tmp_path)]) == 0
    assert "| tool_select:shadow |" in capsys.readouterr().out
