"""Decision points x modes (points.py) against the scripted fake upstreams: what each mode changes in the upstream
request, which tier writes the step, what is logged, and what happens when the decision model fails."""

from __future__ import annotations

import json
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from conftest import FakeGemini, FixedDecision, ScriptedDecision, call_reply, start, text_reply
from jevagentharness import points as P
from jevagentharness.config import Config
from jevagentharness.server import Proxy

STRONG, FAST = "gemini-4-argon", "gemini-3.8-flash"


def fn(name: str, desc: str = "") -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": desc or f"the {name} tool",
                                             "parameters": {"type": "object",
                                                            "properties": {"command": {"type": "string"}}}}}


TOOLS = [fn("bash", "Run a shell command.\nMore text."), fn("read_file"), fn("write_file"), fn("search"),
         fn("list_dir"), fn("http_get")]


def body(steps: int, tools: list[dict[str, Any]] | None = None, last: str = "bash", args: str = "ls",
         stream: bool = False, task: str = "fix the bug", model: str = "jah-auto") -> dict[str, Any]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "You are a coding agent."},
                                  {"role": "user", "content": task}]
    for i in range(steps):
        name = last if i == steps - 1 else "bash"
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(
                {"command": args if i == steps - 1 else f"cmd{i}"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"output {i}"})
    return {"model": model, "messages": msgs, "tools": TOOLS if tools is None else tools, "stream": stream}


async def serve(fake: FakeGemini, tmp_path: Any, backend: Any, sets: list[str], bodies: list[dict[str, Any]],
                path: str = "/v1/chat/completions") -> list[Any]:
    client, up, proxy = await start(fake, tmp_path, backend, sets=sets)
    out = []
    try:
        for b in bodies:
            r = await client.post(path, json=b)
            out.append(await r.text() if b.get("stream") else (r.status, await r.json()))
    finally:
        await client.close()
        await up.close()
    return out


def steps(tmp_path: Any) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (tmp_path / "steps.jsonl").read_text().splitlines()]


def decisions(tmp_path: Any, point: str | None = None) -> list[dict[str, Any]]:
    p = tmp_path / "decisions.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    return [r for r in rows if point is None or r["point"] == point]


def fcc(gemini_body: dict[str, Any]) -> dict[str, Any] | None:
    return (gemini_body.get("toolConfig") or {}).get("functionCallingConfig")


def last_text(gemini_body: dict[str, Any]) -> str:
    parts = gemini_body["contents"][-1]["parts"]
    return " ".join(p.get("text") or (p.get("functionResponse") or {}).get("response", {}).get("output", "")
                    for p in parts)


def models(fake: FakeGemini) -> list[str]:
    return [m for m, _, _ in fake.requests]


# ------------------------------------------------------------------------------------------------ defaults
def test_default_logs_routing_decisions_and_no_points(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    run(serve(fake, tmp_path, FixedDecision([0.1, 0.9]), [], [body(1), body(2)]))
    assert models(fake) == [FAST, STRONG]
    st = steps(tmp_path)
    assert all("points" not in s and "redo" not in s for s in st)
    d = decisions(tmp_path)
    assert [(r["point"], r["mode"], r["effect"], r["acted"]) for r in d] == [
        ("routing", "cascade", "routine", True), ("routing", "cascade", "hard", True)]
    assert d[1]["p"]["hard"] == 0.9 and d[1]["tau"] == 0.5 and "fix the bug" not in json.dumps(d)


# ------------------------------------------------------------------------------------------------ tool_select
def test_tool_shadow_logs_the_llms_choice_and_changes_nothing(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("read_file", {"command": "a.py"})))
    dec = ScriptedDecision({"next_tool": {"bash": 0.97, "read_file": 0.03}})
    run(serve(fake, tmp_path, dec, ["decisions.tool_select=shadow"], [body(0), body(1)]))
    assert models(fake) == [STRONG, FAST]
    for _, b, _ in fake.requests:
        assert fcc(b) is None and len(b["tools"][0]["functionDeclarations"]) == 6
    d = decisions(tmp_path, "tool_select")
    assert len(d) == 2                                   # shadow asks at every step, the opening one included
    assert d[1]["choice"] == "bash" and d[1]["label"] == "read_file" and d[1]["agree"] is False
    assert d[1]["fit_ok"] and not d[1]["acted"] and d[1]["label_tier"] == "fast"
    assert P.NO_TOOL in d[1]["options"] and len(d[1]["options"]) == 7
    asked = [o for q, _, o in dec.asked if q == "next_tool"][0]
    assert asked[:6] == ["bash", "read_file", "write_file", "search", "list_dir", "http_get"]
    assert [s["reason"] for s in steps(tmp_path)] == ["first_steps", "routine"]


def test_tool_cascade_forces_the_tool_on_the_fast_tier(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    dec = ScriptedDecision({"next_tool": [{"bash": 0.995}, {"bash": 0.6, "search": 0.4}, {P.NO_TOOL: 1.0}],
                            "next_step": {"routine": 0.9, "hard": 0.1}})
    sets = ["decisions.tool_select={mode: cascade, tau: 0.99}", "decisions.routing=off"]
    run(serve(fake, tmp_path, dec, sets, [body(0), body(1), body(2), body(3)]))
    # step 0: the opening step, on strong, not asked; 1: confident -> fast, forced; 2: unsure -> strong;
    # 3: "no tool" -> strong
    assert models(fake) == [STRONG, FAST, STRONG, STRONG]
    assert fcc(fake.requests[1][1]) == {"mode": "ANY", "allowedFunctionNames": ["bash"]}
    assert fcc(fake.requests[2][1]) is None and fcc(fake.requests[3][1]) is None
    assert "next_step" not in dec.ids()                  # routing off: no routing question
    st = steps(tmp_path)
    assert [s["reason"] for s in st] == ["first_steps", "tool_cascade", "routing_off", "routing_off"]
    d = decisions(tmp_path, "tool_select")
    assert [(r["acted"], r["effect"]) for r in d] == [(True, "forced"), (False, "defer_unsure"),
                                                       (False, "defer_no_tool")]
    assert d[0]["forced_tool"] == "bash" and d[0]["followed"] is True and d[0]["fit_ok"] is False
    assert d[1]["fit_ok"] is True and d[1]["label"] == "bash"


def test_tool_cascade_leaves_hard_steps_to_the_strong_tier(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    dec = ScriptedDecision({"next_tool": {"bash": 1.0}, "next_step": [{"hard": 0.9}, {"hard": 0.1}]})
    run(serve(fake, tmp_path, dec, ["decisions.tool_select={mode: cascade, tau: 0.99}"], [body(1), body(2)]))
    assert models(fake) == [STRONG, FAST]
    assert fcc(fake.requests[0][1]) is None and fcc(fake.requests[1][1])["allowedFunctionNames"] == ["bash"]
    assert [r["effect"] for r in decisions(tmp_path, "tool_select")] == ["defer_hard", "forced"]


def test_forced_tool_error_falls_back_unforced_on_strong(run, tmp_path):
    def script(model, b):
        if fcc(b):
            return 400, {"error": {"message": "forced tool choice not supported"}}
        return 200, call_reply("bash", {"command": "ls"})

    fake = FakeGemini(script)
    dec = ScriptedDecision({"next_tool": {"bash": 1.0}})
    out = run(serve(fake, tmp_path, dec, ["decisions.tool_select={mode: cascade}"], [body(1)]))
    assert out[0][0] == 200
    assert models(fake) == [FAST, STRONG] and fcc(fake.requests[1][1]) is None
    s = steps(tmp_path)[0]
    assert s["reason"] == "fast_error" and s["tier"] == "strong"


def test_tool_assist_shortlists_and_keeps_used_tools(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    dec = ScriptedDecision({"next_tool": [{"write_file": 0.5, "list_dir": 0.3, "bash": 0.2}]})
    sets = ["decisions.tool_select={mode: assist, k: 2, min_tools: 4}"]
    run(serve(fake, tmp_path, dec, sets, [body(2, last="search")]))
    names = [f["name"] for f in fake.requests[0][1]["tools"][0]["functionDeclarations"]]
    assert names == ["bash", "write_file", "search", "list_dir"]        # top 2 + the tools already used, in order
    d = decisions(tmp_path, "tool_select")[0]
    assert d["effect"] == "shortlist" and d["kept_tools"] == 4 and d["fit_ok"] is False


def test_tool_assist_failure_keeps_every_tool(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    dec = ScriptedDecision(fail={"next_tool"})
    run(serve(fake, tmp_path, dec, ["decisions.tool_select={mode: assist, min_tools: 2}"], [body(2)]))
    assert len(fake.requests[0][1]["tools"][0]["functionDeclarations"]) == 6
    assert decisions(tmp_path, "tool_select")[0]["effect"] == "error_all_tools"


def test_tool_guard_redoes_on_strong_with_a_note(run, tmp_path):
    def script(model, b):
        return 200, call_reply("read_file" if model == FAST else "bash", {"command": "x"})

    fake = FakeGemini(script)
    dec = ScriptedDecision({"next_tool": {"bash": 0.995}})
    out = run(serve(fake, tmp_path, dec, ["decisions.tool_select={mode: guard, tau: 0.99}"], [body(1)]))
    assert models(fake) == [FAST, STRONG]
    assert "[Note from the harness]" in last_text(fake.requests[1][1]) and "read_file" in last_text(fake.requests[1][1])
    assert "[Note from the harness]" not in last_text(fake.requests[0][1])
    assert out[0][1]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"
    s = steps(tmp_path)[0]
    assert s["reason"] == "tool_guard" and s["redo"] == ["tool_guard"] and s["tier"] == "strong"
    d = decisions(tmp_path, "tool_select")[0]
    assert d["label"] == "read_file" and d["after"] == "bash" and d["acted"] and d["fit_ok"]


def test_tool_guard_accepts_an_agreeing_call(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    dec = ScriptedDecision({"next_tool": {"bash": 1.0}})
    run(serve(fake, tmp_path, dec, ["decisions.tool_select=guard"], [body(1)]))
    assert models(fake) == [FAST]
    assert decisions(tmp_path, "tool_select")[0]["effect"] == "accepted"


# ------------------------------------------------------------------------------------------------ stop
def test_stop_guard_accepts_a_confident_final_answer(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, text_reply("done" if m == FAST else "strong says")))
    dec = ScriptedDecision({"task_done": {"done": 0.95, "not_done": 0.05}})
    out = run(serve(fake, tmp_path, dec, ["decisions.stop={mode: guard, tau: 0.9}"], [body(2)]))
    assert models(fake) == [FAST] and out[0][1]["choices"][0]["message"]["content"] == "done"
    state = [s for q, s, _ in dec.asked if q == "task_done"][0]
    assert "PROPOSED FINAL REPLY: message: done" in state
    assert decisions(tmp_path, "stop")[0]["effect"] == "accepted"


def test_stop_guard_redoes_an_unsure_final_answer_and_fails_safe(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, text_reply("done" if m == FAST else "strong says")))
    run(serve(fake, tmp_path, ScriptedDecision({"task_done": {"done": 0.3}}),
              ["decisions.stop={mode: guard, tau: 0.9}"], [body(2)]))
    assert models(fake) == [FAST, STRONG]
    assert steps(tmp_path)[0]["reason"] == "stop_guard"
    # a decision-model failure: the finish check (redo on strong), as in the default mode
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    fake2 = FakeGemini(lambda m, b: (200, text_reply("x")))
    run(serve(fake2, tmp2, ScriptedDecision(fail={"task_done"}), ["decisions.stop=guard"], [body(2)]))
    assert models(fake2) == [FAST, STRONG]
    s = steps(tmp2)[0]
    assert s["reason"] == "finish_check" and s["finish_check"] is True
    assert decisions(tmp2, "stop")[0]["effect"] == "error_redo"


def test_stop_shadow_behaves_as_strong_and_labels(run, tmp_path):
    replies = {FAST: text_reply("done"), STRONG: call_reply("bash", {"command": "pytest"})}
    fake = FakeGemini(lambda m, b: (200, replies[m]))
    run(serve(fake, tmp_path, ScriptedDecision({"task_done": {"done": 0.8}}), ["decisions.stop=shadow"],
              [body(2)]))
    assert models(fake) == [FAST, STRONG]
    d = decisions(tmp_path, "stop")[0]
    assert d["label"] == "not_done" and d["fit_ok"] and d["agree"] is False and d["p_act"] == 0.8
    assert steps(tmp_path)[0]["reason"] == "finish_check"


def test_stop_guard_scope_all_rechecks_a_strong_final_answer(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, text_reply("final")))
    dec = ScriptedDecision({"task_done": {"done": 0.2}, "next_step": {"hard": 0.9}})
    run(serve(fake, tmp_path, dec, ["decisions.stop={mode: guard, scope: all}"], [body(2)]))
    assert models(fake) == [STRONG, STRONG]
    assert P.STOP_REJECT_NOTE in last_text(fake.requests[1][1])
    assert steps(tmp_path)[0]["redo"] == ["stop_guard"]


def test_stop_trigger_sends_the_step_to_strong_with_a_note(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, text_reply("all done")))
    dec = ScriptedDecision({"already_done": [{"yes": 0.95}, {"yes": 0.1}]})
    run(serve(fake, tmp_path, dec, ["decisions.stop={mode: trigger, tau: 0.9}"], [body(2), body(3)]))
    # step 2: fired -> strong + note; step 3: not fired -> fast, whose final answer the strong tier re-does
    assert models(fake) == [STRONG, FAST, STRONG]
    assert P.STOP_TRIGGER_NOTE in last_text(fake.requests[0][1])
    assert P.STOP_TRIGGER_NOTE not in last_text(fake.requests[1][1])
    assert [s["reason"] for s in steps(tmp_path)] == ["stop_trigger", "finish_check"]


# ------------------------------------------------------------------------------------------------ reflection
def test_reflection_trigger_and_always(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    dec = ScriptedDecision({"needs_reflection": [{"yes": 0.9}, {"yes": 0.2}]})
    run(serve(fake, tmp_path, dec, ["decisions.reflection={mode: trigger, tau: 0.8}"], [body(1), body(2)]))
    assert models(fake) == [STRONG, FAST]
    assert P.REFLECT_NOTE in last_text(fake.requests[0][1]) and P.REFLECT_NOTE not in last_text(fake.requests[1][1])
    assert [s["reason"] for s in steps(tmp_path)] == ["reflection", "routine"]
    assert [r["acted"] for r in decisions(tmp_path, "reflection")] == [True, False]

    tmp2 = tmp_path / "always"
    tmp2.mkdir()
    fake2 = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    dec2 = ScriptedDecision()
    run(serve(fake2, tmp2, dec2, ["decisions.reflection={mode: always, every: 2}"], [body(1), body(2)]))
    assert models(fake2) == [FAST, STRONG] and "needs_reflection" not in dec2.ids()
    assert P.REFLECT_NOTE in last_text(fake2.requests[1][1])
    assert steps(tmp2)[1]["mark_reflection"] == "always"


# ------------------------------------------------------------------------------------------------ action
def test_action_guard_rechecks_a_risky_call(run, tmp_path):
    def script(model, b):
        return 200, call_reply("bash", {"command": "rm -rf build" if model == FAST else "ls build"})

    fake = FakeGemini(script)
    dec = ScriptedDecision({"action_risk": {"risky": 0.9}})
    out = run(serve(fake, tmp_path, dec, ["decisions.action=guard"], [body(1)]))
    assert models(fake) == [FAST, STRONG]
    note = last_text(fake.requests[1][1])
    assert "irreversible or risky" in note and "rm -rf build" in note
    assert json.loads(out[0][1]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "command": "ls build"}
    d = decisions(tmp_path, "action")[0]
    assert d["acted"] and d["recheck"] == "changed" and d["tool"] == "bash" and d["pattern"] is True
    assert "PROPOSED ACTION: bash(" in [s for q, s, _ in dec.asked if q == "action_risk"][0]


def test_action_prefilter_skips_safe_calls_and_errors_accept(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls -la"})))
    dec = ScriptedDecision({"action_risk": {"risky": 1.0}})
    run(serve(fake, tmp_path, dec, ["decisions.action=guard"], [body(1)]))
    assert models(fake) == [FAST] and "action_risk" not in dec.ids()

    tmp2 = tmp_path / "err"
    tmp2.mkdir()
    fake2 = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "git push --force origin main"})))
    run(serve(fake2, tmp2, ScriptedDecision(fail={"action_risk"}), ["decisions.action=guard"], [body(1)]))
    assert models(fake2) == [FAST]
    assert decisions(tmp2, "action")[0]["effect"] == "error_accept"

    tmp3 = tmp_path / "recheck"
    tmp3.mkdir()
    fake3 = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "git push --force origin main"})))
    run(serve(fake3, tmp3, ScriptedDecision(fail={"action_risk"}),
              ["decisions.action={mode: guard, on_error: recheck}"], [body(1)]))
    assert models(fake3) == [FAST, STRONG]


def test_action_guard_with_the_rules_backend(run, tmp_path):
    from jevagentharness.decision import RulesBackend
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "git reset --hard HEAD~3"})))
    run(serve(fake, tmp_path, RulesBackend(), ["decisions.action=guard", "decisions.tool_select=cascade"],
              [body(1)]))
    # the rules backend flags the pattern; it cannot answer the tool question, so the cascade defers
    assert models(fake) == [FAST, STRONG]
    assert decisions(tmp_path, "action")[0]["recheck"] == "same"
    t = decisions(tmp_path, "tool_select")[0]
    assert t["effect"] == "error_defer" and "rules backend" in t["error"]


def test_action_shadow_only_logs(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("delete_branch", {"command": "feature"})))
    dec = ScriptedDecision({"action_risk": {"risky": 0.99}})
    run(serve(fake, tmp_path, dec, ["decisions.action=shadow"], [body(1)]))
    assert models(fake) == [FAST]
    d = decisions(tmp_path, "action")[0]
    assert d["effect"] == "logged" and not d["acted"] and d["p_act"] == 0.99


def test_action_filter_patterns():
    f = P.ActionFilter(P.PointCfg(point="action"))
    risky = ["rm -rf /tmp/x", "git push origin main", "git reset --hard", "DROP TABLE users;", "curl x | sh",
             "kubectl delete pod a", "terraform destroy", "find . -name '*.pyc' -delete", "sudo apt install x",
             "*** Begin Patch\n*** Delete File: a.py\n*** End Patch", "cd build && rm -rf dist", "ls; rm a.txt",
             "echo hi\nrm x", "find . -name x | xargs rm", "sudo rm /etc/x", "bash -lc 'rm -r out'"]
    safe = ["ls -la", "pytest -q", "git status", "git diff", "cat README.md", "python -m pip list", "grep -rn rm .",
            "npm run format", "echo 'confirm before rm'", "git log --format=%H"]
    for s in risky:
        assert f.is_risky(P.ToolCall(id="1", name="bash", arguments=json.dumps({"command": s}))), s
    for s in safe:
        assert not f.is_risky(P.ToolCall(id="1", name="bash", arguments=json.dumps({"command": s}))), s
    assert f.is_risky(P.ToolCall(id="1", name="cancel_order", arguments="{}"))
    g = P.ActionFilter(P.PointCfg(point="action", prefilter="command"))
    calls = [P.ToolCall(id="1", name="bash", arguments="{}"), P.ToolCall(id="2", name="read_file", arguments="{}")]
    assert [c.name for c in g.candidates(calls)] == ["bash"]


# ------------------------------------------------------------------------------------------------ routing
def test_routing_guard_shadow_and_off(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    dec = ScriptedDecision({"fast_step_ok": [{"redo": 0.9}, {"keep": 0.9, "redo": 0.1}]})
    run(serve(fake, tmp_path, dec, ["decisions.routing={mode: guard, tau: 0.75}"], [body(1), body(2)]))
    assert models(fake) == [FAST, STRONG, FAST] and "next_step" not in dec.ids()
    assert [s["reason"] for s in steps(tmp_path)] == ["route_guard", "guard"]
    assert "PROPOSED NEXT STEP: bash(" in [s for q, s, _ in dec.asked if q == "fast_step_ok"][0]

    tmp2 = tmp_path / "shadow"
    tmp2.mkdir()
    fake2 = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    run(serve(fake2, tmp2, ScriptedDecision({"next_step": {"routine": 0.9, "hard": 0.1}}),
              ["decisions.routing=shadow"], [body(1)]))
    assert models(fake2) == [STRONG]
    s = steps(tmp2)[0]
    assert s["reason"] == "shadow" and s["p_hard"] == 0.1
    assert decisions(tmp2, "routing")[0]["acted"] is False

    tmp3 = tmp_path / "off"
    tmp3.mkdir()
    fake3 = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    dec3 = ScriptedDecision()
    run(serve(fake3, tmp3, dec3, ["decisions.routing=off"], [body(1)]))
    assert models(fake3) == [STRONG] and dec3.asked == [] and steps(tmp3)[0]["reason"] == "routing_off"


# ------------------------------------------------------------------------------------------------ latency, failures
def test_pre_call_questions_run_concurrently(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    dec = ScriptedDecision(delay={"*": 0.3})
    sets = ["decisions.tool_select=cascade", "decisions.reflection=trigger", "decisions.stop=trigger"]
    run(serve(fake, tmp_path, dec, sets, [body(2)]))
    starts = [t for _, kind, t in dec.events if kind == "start"]
    ends = [t for _, kind, t in dec.events if kind == "end"]
    assert sorted(dec.ids()) == ["already_done", "needs_reflection", "next_step", "next_tool"]
    assert max(starts) < min(ends)                       # all four were in flight at once
    assert steps(tmp_path)[0]["latency_s"] < 0.3 * 3


def test_decision_timeout_falls_back_and_never_breaks(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "rm -rf x"})))
    dec = ScriptedDecision({"next_step": {"routine": 1.0}}, delay={"next_tool": 1.0, "action_risk": 1.0})
    sets = ["decisions.tool_select=cascade", "decisions.action=guard", "router.decision_timeout_s=0.1"]
    out = run(serve(fake, tmp_path, dec, sets, [body(1)]))
    assert out[0][0] == 200 and models(fake) == [FAST]
    effects = {r["point"]: (r["effect"], r["error"]) for r in decisions(tmp_path)}
    assert effects["tool_select"][0] == "error_defer" and "TimeoutError" in effects["tool_select"][1]
    assert effects["action"][0] == "error_accept"


def test_retried_step_reuses_its_decisions(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    dec = ScriptedDecision({"next_tool": {"bash": 1.0}})
    run(serve(fake, tmp_path, dec, ["decisions.tool_select=cascade"], [body(1), body(1)]))
    assert dec.ids().count("next_tool") == 1
    assert fcc(fake.requests[0][1]) == fcc(fake.requests[1][1]) == {"mode": "ANY", "allowedFunctionNames": ["bash"]}
    assert [s["reason"] for s in steps(tmp_path)] == ["tool_cascade", "repeat:tool_cascade"]


def test_many_tools_use_a_knockout(run, tmp_path):
    tools = [fn(f"tool_{i:03d}") for i in range(30)]
    fake = FakeGemini(lambda m, b: (200, call_reply("tool_017", {})))

    def answer(state, q):
        return {o: (0.9 if o == "tool_017" else 0.1 / len(q.options)) for o in q.options}

    dec = ScriptedDecision({"next_tool": answer})
    run(serve(fake, tmp_path, dec, ["decisions.tool_select={mode: cascade, tau: 0.8, max_options: 10}"],
              [body(1, tools=tools, last="tool_003")]))
    sizes = [len(o) for q, _, o in dec.asked if q == "next_tool"]
    assert max(sizes) <= 10 and len(sizes) == 5           # 4 chunks of <= 10 (31 options), then the final
    d = decisions(tmp_path, "tool_select")[0]
    assert d["rounds"] == 2 and d["n_options"] == 31 and d["choice"] == "tool_017" and d["acted"]
    assert fcc(fake.requests[0][1])["allowedFunctionNames"] == ["tool_017"]


def test_states_are_logged_only_with_log_states(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    run(serve(fake, tmp_path, ScriptedDecision(), ["decisions.tool_select=shadow"],
              [body(1, task="secret task text")]))
    assert "secret task text" not in (tmp_path / "decisions.jsonl").read_text()
    assert not (tmp_path / "states.jsonl").exists()
    tmp2 = tmp_path / "states"
    tmp2.mkdir()
    fake2 = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))
    run(serve(fake2, tmp2, ScriptedDecision(), ["decisions.tool_select=shadow", "log_states=true"],
              [body(1, task="secret task text")]))
    rows = [json.loads(x) for x in (tmp2 / "states.jsonl").read_text().splitlines()]
    assert {r.get("point", "routing") for r in rows} == {"routing", "tool_select"}
    assert all("secret task text" in r["state"] for r in rows)


def test_forced_call_does_not_refresh_the_explicit_cache(run, tmp_path):
    import asyncio
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"}, sig=None)))
    big = "x" * 40000
    dec = ScriptedDecision({"next_tool": [{"bash": 1.0}, {"bash": 0.5, "search": 0.5}]})

    async def go():
        client, up, proxy = await start(fake, tmp_path, dec, sets=["decisions.tool_select=cascade"])
        try:
            for n in (1, 2):
                await client.post("/v1/chat/completions", json=body(n, task=big))
                await asyncio.sleep(0.05)
        finally:
            await client.close()
            await up.close()

    run(go())
    assert models(fake) == [FAST, FAST]
    assert len(fake.caches) == 1                          # made from the unforced step 2 only
    assert "toolConfig" not in fake.caches["cachedContents/c0"]


# ------------------------------------------------------------------------------------------------ client APIs
def test_chat_stream_with_forced_tool(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("search", {"command": "foo"})))
    dec = ScriptedDecision({"next_tool": {"search": 1.0}})
    out = run(serve(fake, tmp_path, dec, ["decisions.tool_select=cascade"], [body(1, stream=True)]))
    events = [json.loads(ln[6:]) for ln in out[0].splitlines() if ln.startswith("data: {")]
    calls = [tc for e in events for ch in e["choices"] for tc in ch["delta"].get("tool_calls", [])]
    assert calls[0]["function"]["name"] == "search" and out[0].rstrip().endswith("data: [DONE]")
    assert fcc(fake.requests[0][1])["allowedFunctionNames"] == ["search"]


def codex_body(stream: bool = True) -> dict[str, Any]:
    return {"model": "jah-auto", "stream": stream, "store": False, "prompt_cache_key": "codex-1",
            "input": [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "sys"}]},
                      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix calc"}]},
                      {"type": "function_call", "call_id": "k1", "name": "exec_command",
                       "arguments": json.dumps({"cmd": "cat calc.py"})},
                      {"type": "function_call_output", "call_id": "k1", "output": "def add(a,b): return a-b"}],
            "tools": [{"type": "function", "name": "exec_command", "description": "run",
                       "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
                      {"type": "custom", "name": "apply_patch", "description": "patch",
                       "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"}}]}


def test_responses_client_stream_with_forced_custom_tool_on_gemini(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("apply_patch", {"input": "*** Begin Patch\n*** End Patch\n"})))
    dec = ScriptedDecision({"next_tool": {"apply_patch": 1.0}})
    out = run(serve(fake, tmp_path, dec, ["decisions.tool_select=cascade"], [codex_body()], path="/v1/responses"))
    assert fcc(fake.requests[0][1])["allowedFunctionNames"] == ["apply_patch"]
    items = [json.loads(ln[6:])["item"] for ln in out[0].splitlines()
             if ln.startswith("data: ") and '"response.output_item.done"' in ln]
    assert items[0]["type"] == "custom_tool_call" and items[0]["input"].startswith("*** Begin Patch")


class FakeOpenAI:
    """A scripted OpenAI-compatible upstream: /v1/chat/completions and /v1/responses."""

    def __init__(self, chat_reply: dict[str, Any] | None = None, responses_reply: dict[str, Any] | None = None):
        self.chat_reply, self.responses_reply = chat_reply, responses_reply
        self.requests: list[tuple[str, dict[str, Any]]] = []

    async def chat(self, request: web.Request) -> web.Response:
        self.requests.append(("chat", await request.json()))
        return web.json_response(self.chat_reply)

    async def responses(self, request: web.Request) -> web.Response:
        self.requests.append(("responses", await request.json()))
        return web.json_response(self.responses_reply)

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_post("/v1/responses", self.responses)
        return app


async def _openai_proxy(fake: FakeOpenAI, tmp_path: Any, kind: str, dec: Any, decisions_: dict[str, Any]
                        ) -> tuple[TestClient, TestServer]:
    up = TestServer(fake.app())
    await up.start_server()
    base = str(up.make_url("")).rstrip("/") + "/v1"
    cfg = Config.from_dict({
        "log_dir": str(tmp_path),
        "upstreams": {"o": {"type": kind, "base_url": base, "key_env": "", "retries": 0}},
        "tiers": {"strong": {"upstream": "o", "model": "gpt-6.1-sol"},
                  "fast": {"upstream": "o", "model": "gpt-6-luna"}},
        "router": {"decision": {"type": "rules"}}, "decisions": decisions_})
    client = TestClient(TestServer(Proxy(cfg, backend=dec).app()))
    await client.start_server()
    return client, up


def test_forced_tool_choice_on_openai_chat_upstream(run, tmp_path):
    reply = {"id": "x", "model": "gpt-6-luna", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {
            "name": "bash", "arguments": "{\"command\": \"ls\"}"}}]}}], "usage": {"prompt_tokens": 10,
                                                                                "completion_tokens": 2}}
    fake = FakeOpenAI(chat_reply=reply)

    async def go():
        client, up = await _openai_proxy(fake, tmp_path, "openai", ScriptedDecision({"next_tool": {"bash": 1.0}}),
                                         {"tool_select": "cascade"})
        try:
            r = await client.post("/v1/chat/completions", json=body(1))
            return await r.json()
        finally:
            await client.close()
            await up.close()

    d = run(go())
    kind, sent = fake.requests[0]
    assert kind == "chat" and sent["model"] == "gpt-6-luna"
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "bash"}}
    assert d["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "bash"


def test_forced_tool_choice_on_openai_responses_upstream_from_codex(run, tmp_path):
    reply = {"id": "r", "model": "gpt-6-luna", "status": "completed", "output": [
        {"type": "function_call", "call_id": "t2", "name": "exec_command", "arguments": "{\"cmd\": \"pytest\"}"}],
        "usage": {"input_tokens": 10, "output_tokens": 3}}
    fake = FakeOpenAI(responses_reply=reply)

    async def go():
        client, up = await _openai_proxy(fake, tmp_path, "openai_responses",
                                         ScriptedDecision({"next_tool": {"exec_command": 1.0}}),
                                         {"tool_select": "cascade"})
        try:
            r = await client.post("/v1/responses", json=codex_body(stream=False))
            return await r.json()
        finally:
            await client.close()
            await up.close()

    d = run(go())
    kind, sent = fake.requests[0]
    assert kind == "responses" and sent["model"] == "gpt-6-luna"
    assert sent["tool_choice"] == {"type": "function", "name": "exec_command"}
    assert d["output"][0]["type"] == "function_call" and d["output"][0]["name"] == "exec_command"


# ------------------------------------------------------------------------------------------------ configuration
def test_config_legacy_keys_presets_and_model_overrides():
    from jevagentharness.config import load_config
    c = load_config(None)
    dc = c.decisions_for(c.models["jah-auto"])
    assert dc.modes() == {"routing": "cascade", "tool_select": "off", "stop": "strong", "reflection": "off",
                          "action": "off"}
    assert dc.routing.tau == c.router.tau == 0.75 and dc.only_routing()
    assert c.decisions_for(c.models["jah-strong"]).modes()["stop"] == "off"      # pinned: a plain baseline
    c = load_config(None, ["router.verify_finish=false"])
    assert c.decisions_for(c.models["jah-auto"]).stop.mode == "off"
    c = load_config(None, ["preset=paper"])
    ts = c.decisions_for(c.models["jah-auto"]).tool_select
    assert ts.mode == "cascade" and ts.tau == 0.99
    c = load_config(None, ["preset=shadow", "decisions.stop=strong"])
    assert c.decisions_for(c.models["jah-auto"]).modes() == {
        "routing": "cascade", "tool_select": "shadow", "stop": "strong", "reflection": "shadow", "action": "shadow"}
    c = load_config(None, ['models={"jah-auto": {"policy": "route"}, "jah-auto-t9": {"policy": "route", "tau": 0.9, '
                           '"verify_finish": false, "decisions": {"action": "guard", "tool_select": {"mode": '
                           '"shadow"}}}, "jah-safe": {"tier": "strong", "preset": "safe"}, "jah-strong": '
                           '{"tier": "strong"}}', "decisions.routing.tau=0.6"])
    a = c.decisions_for(c.models["jah-auto"])
    t9 = c.decisions_for(c.models["jah-auto-t9"], 0.9)
    assert a.routing.tau == 0.6 and t9.routing.tau == 0.9 and t9.stop.mode == "off"
    assert t9.action.mode == "guard" and t9.tool_select.mode == "shadow"
    assert c.decisions_for(c.models["jah-safe"]).modes() == {
        "routing": "off", "tool_select": "off", "stop": "off", "reflection": "off", "action": "guard"}
    assert c.decisions_for(c.models["jah-strong"]).modes()["action"] == "off"
    c = load_config(None, ["decisions.routing=trigger", "decisions.tool_select={mode: off}"])
    assert c.decisions_for(c.models["jah-auto"]).routing.mode == "cascade"       # an alias in a proxy


def test_config_rejects_bad_decisions():
    import pytest

    from jevagentharness.config import load_config
    for bad, msg in [("decisions.tool_params=cascade", "not offered"), ("decisions.memory_write=trigger", "memory"),
                     ("decisions.toolselect=shadow", "unknown decision point"),
                     ("decisions.stop=assist", "must be one of"), ("decisions.action={mode: guard, k: 3}", "unknown"),
                     ("decisions.tool_select={mode: cascade, tau: 1.5}", "tau"), ("preset=fast", "unknown preset"),
                     ("decisions.tool_select={mode: cascade, tier: nope}", "unknown tier"),
                     ("decisions.action={mode: guard, patterns: ['(']}", "bad regex"),
                     ('models={"x": {"tier": "strong", "decisions": {"routing": "guard"}}}', "pinned")]:
        with pytest.raises(ValueError, match=msg):
            load_config(None, [bad])


def test_with_notes_keeps_the_conversation_shape():
    from jevagentharness.types import Message, Request
    r = Request(messages=[Message.of("user", "task"), Message(role="assistant", tool_calls=[
        P.ToolCall(id="a", name="bash", arguments="{}")]), Message.of("tool", "out", tool_call_id="a")])
    r2 = P.with_notes(r, ["N1", "N2"])
    assert r2.modified and not r.modified and r.messages[-1].text == "out"
    assert r2.messages[-1].role == "tool" and r2.messages[-1].text == "out\n\nN1\n\nN2"
    r3 = P.with_notes(Request(messages=[Message.of("assistant", "hi")]), ["N"])
    assert r3.messages[-1].role == "user" and r3.messages[-1].text == "N"
    assert P.with_notes(r, []) is r


# ------------------------------------------------------------------------------------------------ model names, stats
def test_model_names_side_by_side_and_pinned_with_own_points(run, tmp_path):
    def script(model, b):
        return 200, call_reply("bash", {"command": "rm -rf build" if model == STRONG and len(fake.requests) == 1
                                        else "ls"})

    fake = FakeGemini(script)
    dec = ScriptedDecision({"action_risk": {"risky": 0.9}, "next_tool": {"bash": 1.0}})
    models_ = ('models={"jah-auto": {"policy": "route"}, "jah-auto-tools": {"policy": "route", "decisions": '
               '{"routing": "off", "tool_select": "cascade"}}, "jah-strong": {"tier": "strong"}, '
               '"jah-strong-safe": {"tier": "strong", "preset": "safe"}}')
    run(serve(fake, tmp_path, dec, [models_, "decisions.tool_select=shadow"], [
        body(1, model="jah-strong-safe"), body(1, model="jah-strong", task="b"), body(1, model="jah-auto-tools",
                                                                                     task="c")]))
    # pinned + safe: the risky call is re-checked on strong; the plain pinned name asks nothing; the tools arm forces
    assert models(fake) == [STRONG, STRONG, STRONG, FAST]
    st = steps(tmp_path)
    assert [s["reason"] for s in st] == ["action_guard", "pinned", "tool_cascade"]
    assert "points" not in st[1]
    d = decisions(tmp_path)
    assert [(r["model_name"], r["point"]) for r in d] == [("jah-strong-safe", "action"),
                                                          ("jah-auto-tools", "tool_select")]


def test_routing_guard_is_skipped_when_the_finish_check_redoes(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, text_reply("done")))
    dec = ScriptedDecision()
    run(serve(fake, tmp_path, dec, ["decisions.routing=guard"], [body(2)]))
    assert models(fake) == [FAST, STRONG] and "fast_step_ok" not in dec.ids()
    assert steps(tmp_path)[0]["reason"] == "finish_check"


def test_stats_show_decision_points(run, tmp_path):
    fake = FakeGemini(lambda m, b: (200, call_reply("bash", {"command": "ls"})))

    async def go():
        client, up, proxy = await start(fake, tmp_path, ScriptedDecision(), sets=["preset=shadow"])
        try:
            await client.post("/v1/chat/completions", json=body(2))
            return await (await client.get("/stats")).json()
        finally:
            await client.close()
            await up.close()

    s = run(go())
    assert s["decision_points"] == {"routing": "cascade", "tool_select": "shadow", "stop": "shadow",
                                    "reflection": "shadow", "action": "shadow"}
    assert s["decision_calls"]["tool_select:asked"] == 1 and s["decision_calls"]["routing:asked"] == 1
