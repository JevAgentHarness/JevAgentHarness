"""Regression tests for the E2 dev runs (runs/dev/E2_NOTES.md): bugs found by running E2a / E2b / E2c on the
calibrated JevGym v2 suite with the Gemini stand-in System-1. Offline only."""

from __future__ import annotations

import json
from typing import Any

import pytest

from jevharness.bench.env import Env, StepResult
from jevharness.bench.scorecard import compute_scorecard, family_references, is_llm_only_arm, main as scorecard_main
from jevharness.decision.base import DecisionBackend
from jevharness.decision.controls import trigger_rates
from jevharness.decision.types import DecisionResult, Usage, answer_from_distribution
from jevharness.harness.agent import JevAgent, append_note
from jevharness.harness.config import DPConfig, HarnessConfig, Mode
from jevharness.llm.base import ChatResponse, ToolCall
from jevharness.llm.scripted import ScriptedLLM


# ------------------------------------------------------------------ helpers
def _fn(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": f"{name} tool", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"}}, "required": []}}}


class StepEnv(Env):
    task_id = "toy"
    instruction = "Look things up, then finish."

    def __init__(self, done_after: int = 2):
        self.n, self.done_after = 0, done_after

    def reset(self) -> str:
        return "start"

    def tools(self) -> list[dict[str, Any]]:
        return [_fn("lookup")]

    def step(self, tool: str, args: dict[str, Any]) -> StepResult:
        self.n += 1
        return StepResult(f"result {self.n}")

    def oracle_done(self):
        return self.n >= self.done_after

    def evaluate(self, final_answer):
        return {"success": final_answer == "done", "score": float(final_answer == "done")}


class StepS1(DecisionBackend):
    """System-1 double answering yes/no by the number of executed steps in the rendered state."""

    name = "step_s1"
    model = "step-s1"

    def __init__(self, yes):
        self.yes = yes  # (question text, n history lines) -> bool

    def _decide(self, state, questions):
        n_hist = state.split("CURRENT OBSERVATION")[0].count("\n[")
        ans = [answer_from_distribution(q, {"yes": 0.99, "no": 0.01} if self.yes(q.text, n_hist)
                                        else {"yes": 0.01, "no": 0.99}) for q in questions]
        return DecisionResult(answers=ans, usage=Usage(backend=self.name, model=self.model))


def gemini_like(policy):
    """gemini-3.5-flash as observed in the E2b dev run: a tool-calling request that ends on a tool result while
    its history holds a user text message right after an earlier tool result gets an EMPTY reply."""
    requests: list[list[dict[str, Any]]] = []

    def wrapped(messages, tools):
        requests.append([dict(m) for m in messages])
        roles = [m["role"] for m in messages]
        if tools and roles[-1] == "tool" and any(a == "tool" and b == "user" for a, b in zip(roles, roles[1:])):
            return ChatResponse(text="", finish_reason=None)
        return policy(messages, tools)

    return wrapped, requests


def _script(names: list[str]):
    """LLM policy: reflection calls answer text; tool calls follow ``names`` (one per tool-calling request)."""
    i = {"n": 0}

    def policy(messages, tools):
        if tools is None:
            return ChatResponse(text="REFLECTION-TEXT")
        n = min(i["n"], len(names) - 1)
        i["n"] += 1
        name = names[n]
        args = {"answer": "done"} if name == "finish" else {"q": "x"}
        return ChatResponse(text="", tool_calls=[ToolCall(id=f"c{i['n']}", name=name, arguments=args)])

    return policy


# ------------------------------------------------------------------ bug 1: harness notes after a tool result
def test_append_note_merges_into_a_trailing_tool_result_else_appends_a_user_message():
    msgs = [{"role": "user", "content": "task"}]
    append_note(msgs, "[Verifier] no")
    assert msgs[-1] == {"role": "user", "content": "[Verifier] no"}
    msgs += [{"role": "assistant", "content": None, "tool_calls": []}, {"role": "tool", "tool_call_id": "c", "content": "obs"}]
    n = len(msgs)
    append_note(msgs, "[Reflection] r")
    assert len(msgs) == n and msgs[-1]["role"] == "tool" and msgs[-1]["tool_call_id"] == "c"
    assert msgs[-1]["content"] == "obs\n\n[Reflection] r"


def test_triggered_reflection_does_not_leave_a_user_message_after_a_tool_result():
    """E2b dev: jev_reflect_trigger failed 5/10 episodes on Flash -- every episode whose reflection fired at step 1
    but not at step 2 got an empty reply at step 2 (read as an empty finish)."""
    policy, requests = gemini_like(_script(["lookup", "lookup", "lookup", "finish"]))
    s1 = StepS1(lambda text, n: "stuck" in text and n == 1)     # reflection fires at step 1 only
    cfg = HarnessConfig(name="reflect_trigger", max_steps=6, reflection=DPConfig(mode=Mode.TRIGGER, tau=0.5))
    traj = JevAgent(ScriptedLLM("scripted", policy=policy), cfg, s1=s1).run(StepEnv())
    assert traj.error is None and traj.success and traj.final_answer == "done"
    assert [s.tool for s in traj.steps] == ["lookup", "lookup", "lookup", "finish"]
    assert traj.steps[1].reflection == "REFLECTION-TEXT" and "empty_llm_responses" not in traj.meta
    tool_reqs = [r for r in requests if r[-1]["role"] != "user" or "Pause" not in str(r[-1]["content"])]
    for r in tool_reqs:
        roles = [m["role"] for m in r]
        assert not any(a == "tool" and b == "user" for a, b in zip(roles, roles[1:]))
    # the reflection stays in the LLM's context for the rest of the episode
    assert any("[Reflection] REFLECTION-TEXT" in str(m.get("content")) for m in requests[-1])


def test_stop_guard_rejection_note_does_not_break_later_tool_calls():
    policy, _ = gemini_like(_script(["lookup", "finish", "lookup", "finish"]))
    # the verifier rejects the first finish (after 1 executed step), accepts later ones
    s1 = StepS1(lambda text, n: "complete" in text and n >= 2)
    cfg = HarnessConfig(name="stop_guard", max_steps=6, stop=DPConfig(mode=Mode.GUARD, tau=0.5))
    traj = JevAgent(ScriptedLLM("scripted", policy=policy), cfg, s1=s1).run(StepEnv())
    assert traj.error is None and traj.success, [(s.tool, s.observation) for s in traj.steps]
    assert [s.tool for s in traj.steps] == ["lookup", "finish", "lookup", "finish"]
    assert traj.steps[1].observation == "(finish rejected by stop verifier)"


def test_empty_llm_replies_are_counted_in_the_trajectory():
    def policy(messages, tools):
        return ChatResponse(text="", finish_reason=None)

    traj = JevAgent(ScriptedLLM("scripted", policy=policy), HarnessConfig()).run(StepEnv())
    assert traj.meta["empty_llm_responses"] == 1 and traj.final_answer == "" and not traj.success


# ------------------------------------------------------------------ bug 2: matched-rate protocol on resumed runs
def _rt(task: str, fires: list[bool], err: str | None = None) -> dict[str, Any]:
    return {"task_id": task, "config": "jev_reflect_trigger", "error": err, "result": {}, "meta": {},
            "steps": [{"step": i, "decisions": [{"dp": "reflection", "mode": "trigger", "s1_value": f}]}
                      for i, f in enumerate(fires)]}


def test_trigger_rates_count_the_last_record_per_task_and_skip_partial_lines(tmp_path):
    """A resumed run appends a fresh record for an errored episode: trigger_rates read raw lines, counted that
    task twice and crashed on a killed run's partial last line."""
    p = tmp_path / "jev_reflect_trigger.jsonl"
    lines = [_rt("t1", [True, True, True, True], err="RateLimitError"), _rt("t1", [False, False]), _rt("t2", [True, False])]
    p.write_text("".join(json.dumps(r) + "\n" for r in lines) + '{"task_id": "t3", "steps": [')
    with pytest.warns(UserWarning):
        r = trigger_rates(p)
    assert r["reflection/trigger"] == {"n": 4, "rate": 0.25}
    with pytest.warns(UserWarning):
        assert trigger_rates(tmp_path)["reflection/trigger"]["n"] == 4      # a run directory works the same


# ------------------------------------------------------------------ bug 3: one reference per System-2 model
OFF = {dp: {"mode": "off"} for dp in ("tool_select", "tool_params", "memory_write", "memory_read", "reflection",
                                      "stop", "routing", "action")}


def _ep(task: str, ok: bool, cost: float, llm: str, cfg: dict | None = None) -> dict[str, Any]:
    return {"task_id": task, "config": "x", "backend": None, "llm": llm, "steps": [], "final_answer": None,
            "result": {"success": ok}, "error": None, "meta": {"config": cfg or OFF},
            "usage": {"total": {"cost_usd": cost, "latency_ms": 1000.0}, "s2": {"calls": 2}},
            "calls": [{"component": "s2", "model": llm}] * 2}


def _family_runs() -> dict[str, list[dict[str, Any]]]:
    n = 12
    assist = {**OFF, "tool_select": {"mode": "assist", "k": 3}}
    return {
        "llm_only": [_ep(f"t{i}", True, 1.0, "flash") for i in range(n)],
        "assist": [_ep(f"t{i}", True, 0.6, "flash", assist) for i in range(n)],
        "llm_only_s2lite": [_ep(f"t{i}", i % 3 != 0, 0.2, "lite") for i in range(n)],     # 8/12
        "small_only": [_ep(f"t{i}", i % 3 != 0, 0.2, "lite") for i in range(n)],
        "assist_s2lite": [_ep(f"t{i}", True, 0.15, "lite", assist) for i in range(n)],     # 12/12
    }


def test_family_references_pair_each_arm_with_the_llm_only_arm_of_its_system2_model():
    runs = _family_runs()
    assert is_llm_only_arm(runs["llm_only_s2lite"]) and not is_llm_only_arm(runs["assist_s2lite"])
    refs = family_references(runs, "llm_only")
    assert refs == {"llm_only": "llm_only", "assist": "llm_only", "llm_only_s2lite": "llm_only",
                    "small_only": "llm_only_s2lite", "assist_s2lite": "llm_only_s2lite"}


def test_scorecard_compares_s2lite_arms_with_the_s2lite_reference():
    """E2a/E2b dev: the card compared every *_s2lite arm with the Flash llm_only, so a Flash-Lite arm that RAISES
    success over Flash-Lite LLM-only showed up as a loss, and Flash-Lite arms competed in Flash's iso-quality."""
    card = compute_scorecard(_family_runs(), baseline="llm_only", n_boot=300)
    a = card["arms"]
    vb = a["assist_s2lite"]["vs_baseline"]
    assert vb["reference"] == "llm_only_s2lite" and vb["delta"] == pytest.approx(4 / 12)
    assert vb["cost_savings"] == pytest.approx(0.25)
    assert a["llm_only_s2lite"]["vs_baseline"]["reference"] == "llm_only"          # the S2-strength contrast
    assert a["llm_only_s2lite"]["vs_baseline"]["delta"] == pytest.approx(-4 / 12)
    assert a["assist"]["vs_baseline"]["reference"] == "llm_only"
    iq = card["headline"]["iso_quality"]
    assert iq["arm"] == "assist" and set(iq["qualified"]) == {"assist"}            # no Flash-Lite arm competes
    iq_lite = card["headline"]["iso_quality_by_reference"]["llm_only_s2lite"]
    assert iq_lite["arm"] == "assist_s2lite" and iq_lite["baseline"] == "llm_only_s2lite"
    assert iq_lite["qualified"] == ["assist_s2lite"]          # small_only: an LLM-only replicate, not a candidate
    assert a["small_only"]["vs_baseline"]["reference"] == "llm_only_s2lite"
    assert any("per-System-2 references" in n for n in card["meta"]["notes"])
    # the pre-fix behaviour stays available
    single = compute_scorecard(_family_runs(), baseline="llm_only", n_boot=300, family_baselines=False)
    assert single["arms"]["assist_s2lite"]["vs_baseline"]["reference"] == "llm_only"
    assert single["arms"]["assist_s2lite"]["vs_baseline"]["delta"] == pytest.approx(0.0)


def test_scorecard_cli_single_baseline_flag(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    for arm, eps in _family_runs().items():
        (run / f"{arm}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    scorecard_main([str(run), "--n-boot", "200"])
    md = capsys.readouterr().out
    assert "vs llm_only_s2lite" in md and "(vs. `llm_only_s2lite`)" in md
    scorecard_main([str(run), "--n-boot", "200", "--single-baseline", "--format", "json"])
    card = json.loads(capsys.readouterr().out)
    assert card["arms"]["assist_s2lite"]["vs_baseline"]["reference"] == "llm_only"
    assert "iso_quality_by_reference" not in card["headline"]


def test_shadow_arms_stay_off_the_frontier_and_best_cost_of_pass():
    """E2a dev: jev_shadow (= the LLM-only harness, System-1 only logged) was 9% cheaper than llm_only through
    prompt-cache luck and took the Pareto front; jev_shadow_s2lite became "best cost-of-pass"."""
    shadow_cfg = {**OFF, "tool_select": {"mode": "shadow"}}
    dec = {"dp": "tool_select", "mode": "shadow", "options": ["a", "b"], "s1_value": "a", "s1_conf": 0.9,
           "gold": "a", "shadow": True}
    sh = [dict(_ep(f"t{i}", True, 0.9, "flash", shadow_cfg), steps=[{"step": 0, "decisions": [dict(dec)]}])
          for i in range(6)]
    runs = {"llm_only": [_ep(f"t{i}", True, 1.0, "flash") for i in range(6)], "jev_shadow": sh,
            "assist": [_ep(f"t{i}", i < 3, 0.6, "flash", {**OFF, "tool_select": {"mode": "assist"}})
                       for i in range(6)]}
    card = compute_scorecard(runs, baseline="llm_only", n_boot=200)
    a = card["arms"]
    assert a["jev_shadow"]["shadow_only"] and not a["jev_shadow"]["pareto"]
    assert a["llm_only"]["pareto"] and "jev_shadow" not in card["headline"]["hypervolume"]["front"]
    assert card["headline"]["best_cost_of_pass"] == "llm_only"
    assert "vs_baseline" in a["jev_shadow"]          # still compared (the shadow reproduction check)


# ------------------------------------------------------------------ bug 4: Pareto figure mixed System-2 models
def test_pareto_plot_draws_one_system2_family(tmp_path, capsys):
    from jevharness.bench.plots import pareto

    run = tmp_path / "run"
    run.mkdir()
    for arm, eps in _family_runs().items():
        (run / f"{arm}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    pareto(run, tmp_path / "flash.png")
    err = capsys.readouterr().err
    assert (tmp_path / "flash.png").exists()
    assert "left out 3 arm(s)" in err and "assist_s2lite" in err and "llm_only_s2lite" in err
    pareto(run, tmp_path / "lite.png", baseline="llm_only_s2lite")
    err = capsys.readouterr().err
    assert "left out 2 arm(s)" in err and "assist," in err
    pareto(run, tmp_path / "all.png", s2="all")
    assert "left out" not in capsys.readouterr().err


# ------------------------------------------------------------------ bug 5: instruction values crowded out of the options
def _noisy_state(n_noise: int = 80) -> str:
    noise = "\n".join(f"[{i}] get_orders(...) -> log: heartbeat ok #{100000 + 37 * i} order ORD-{5000 + i}"
                      for i in range(n_noise))
    return ("TASK: Process these requests in order: ship Goro Haddad's pending order, then cancel Lena Kim's.\n\n"
            f"HISTORY:\n{noise}\n\nCURRENT OBSERVATION:\nlog: heartbeat ok #999999")


def test_extract_candidates_keeps_the_instruction_values_first():
    from jevharness.harness.tools import extract_candidates, task_section_end

    st = _noisy_state()
    assert "Goro Haddad" not in extract_candidates(st)               # the pre-fix behaviour (default unchanged)
    c = extract_candidates(st, task_first=25)
    assert c[0] == "Goro Haddad" and "Lena Kim" in c[:5] and len(c) == 51
    assert "999999" in c                                             # the most recent spans still follow
    # not a harness state: task_first changes nothing
    assert task_section_end("no harness state") == 0
    assert extract_candidates("x ORD-1 ORD-2", task_first=5) == extract_candidates("x ORD-1 ORD-2")


class NoisyEnv(StepEnv):
    instruction = "Process these requests in order: ship Goro Haddad's pending order, then cancel Lena Kim's."

    def tools(self):
        return [{"type": "function", "function": {"name": "find_user", "description": "Look up a customer.",
                 "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                                "required": ["name"]}}}]

    def step(self, tool, args):
        self.n += 1
        return StepResult("\n".join(f"log: heartbeat ok #{100000 + 37 * (self.n * 70 + i)}" for i in range(70)))

    def oracle_action(self):
        return ("find_user", {"name": "Goro Haddad"}) if self.n < 3 else ("finish", {"answer": "done"})


def test_tool_params_options_contain_the_instruction_values_after_a_noisy_history():
    class PickName(DecisionBackend):
        name, model = "pick", "pick"

        def _decide(self, state, questions):
            out = []
            for q in questions:
                labels = q.labels()
                top = next((x for x in ("find_user", "Goro Haddad") if x in labels), None)
                if q.text.startswith("Which tool") and self_n["n"] >= 3:
                    top = "finish"
                dist = {lab: (0.9 if lab == top else 0.1 / max(1, len(labels) - 1)) for lab in labels}
                out.append(answer_from_distribution(q, dist))
            self_n["n"] += q.text.startswith("Which tool")
            return DecisionResult(answers=out, usage=Usage(backend=self.name, model=self.model))

    self_n = {"n": 0}
    cfg = HarnessConfig(name="params_replace", max_steps=5, tool_select=DPConfig(mode=Mode.REPLACE),
                        tool_params=DPConfig(mode=Mode.REPLACE))
    llm = ScriptedLLM("scripted", policy=lambda m, t: ChatResponse(text="done"))
    traj = JevAgent(llm, cfg, s1=PickName()).run(NoisyEnv())
    # (``generate`` records: free-text slots System-2 fills, e.g. finish.answer -- no options by design)
    params = [d for s in traj.steps for d in s.decisions if d.dp == "tool_params" and d.mode != "generate"]
    assert len(params) >= 3 and all("Goro Haddad" in d.options for d in params)
    assert all(s.args == {"name": "Goro Haddad"} for s in traj.steps if s.tool == "find_user")


# ------------------------------------------------------------------ gap: pre-registered arm-vs-control comparisons
def test_scorecard_pairs_arm_vs_control_with_utility(tmp_path, capsys):
    """C3 / C5 compare a System-1 arm with a control arm (not with LLM-only); the card had no such comparison."""
    runs = {"llm_only": [_ep(f"t{i}", True, 1.0, "flash") for i in range(10)],
            "jev": [_ep(f"t{i}", i < 8, 0.5, "flash") for i in range(10)],
            "bm25": [_ep(f"t{i}", i < 3, 0.4, "flash") for i in range(10)]}
    card = compute_scorecard(runs, baseline="llm_only", n_boot=500, pairs=[("jev", "bm25"), ("jev", "nope")],
                             task_values=(0.1, 10.0))
    p = card["headline"]["pairs"]["jev vs bm25"]
    assert p["n_paired"] == 10 and p["delta"] == pytest.approx(0.5) and p["p_better"] < 0.05
    assert (p["n_a_only"], p["n_b_only"]) == (5, 0) and p["mcnemar_p"] == pytest.approx(0.0625)
    assert p["delta_cost"] == pytest.approx(0.1)
    assert p["utility"]["0.1"]["delta_u"] == pytest.approx(0.1 * 0.5 - 0.1)       # W x dS - dC
    assert p["utility"]["10"]["delta_u"] == pytest.approx(10 * 0.5 - 0.1) and p["utility"]["10"]["p_better"] < 0.05
    assert any("nope" in n for n in card["meta"]["notes"])
    run = tmp_path / "run"
    run.mkdir()
    for arm, eps in runs.items():
        (run / f"{arm}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    scorecard_main([str(run), "--n-boot", "200", "--pair", "jev:bm25", "--task-value", "1"])
    md = capsys.readouterr().out
    assert "### Paired comparisons (arm vs. control)" in md and "| jev | bm25 | 10 | +50.0 |" in md
