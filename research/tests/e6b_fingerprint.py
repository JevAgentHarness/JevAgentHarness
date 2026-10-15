"""Behaviour + call-graph fingerprints of the harness modes that existed before E6b (2026-10-06), used by
``tests/test_e6b_modes.py`` to show that adding the E6b cells left every existing mode unchanged.

Each configuration runs offline (simulated System-2 that follows the env oracle with a seeded error rate, simulated
System-1 with a seeded per-question stream; a fresh backend per configuration) on a few JevGym tasks (and a tiny
action-only env for the ``action`` modes). The fingerprint of an episode is everything the harness decided and every
call it made, minus timing and random call ids: success, final answer, per step the executed tool / arguments,
LLM-call count, reflection and observation head, every DecisionLog (dp, mode, question, options, System-1 value and
confidence, deferral, System-2 / final values, gold, gold status / aux, retry, shadow) and every ledger record
(component, decision point, step, model, tokens, cost, shadow tag). ``python tests/e6b_fingerprint.py`` rewrites
``tests/fixtures/e6b_existing_modes_fingerprint.json`` -- run it only on the pre-E6b code (it was generated at
commit cbedd65 before any harness edit)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench.env import Env, StepResult  # noqa: E402
from jevharness.bench.suites.jevgym import JevGymEnv  # noqa: E402
from jevharness.decision.simulated import SimulatedBackend  # noqa: E402
from jevharness.harness.agent import JevAgent  # noqa: E402
from jevharness.harness.config import DPConfig, HarnessConfig, Mode  # noqa: E402
from jevharness.llm.base import ChatResponse  # noqa: E402
from jevharness.llm.simulated import SimulatedLLM  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "e6b_existing_modes_fingerprint.json"
N_TASKS = 6


class TinyActionEnv(Env):
    """A three-step action-only env with an oracle (go -> take -> put); success iff the plan is followed."""

    PLAN = ["go to shelf 1", "take mug 1 from shelf 1", "put mug 1 in/on desk 1"]
    NOISE = ["look", "inventory", "go to desk 1", "open drawer 1", "examine shelf 1"]

    def __init__(self, i: int):
        self.task_id = f"tiny-action-{i}"
        self.instruction = "Your task is to: put a mug on the desk."
        self.i_task = i

    def reset(self) -> str:
        self.t, self.done_ok = 0, True
        return "You are in a room. You see a shelf 1 and a desk 1."

    def admissible_actions(self) -> list[str]:
        acts = [self.PLAN[min(self.t, 2)]] + self.NOISE
        k = (self.i_task + self.t) % len(acts)
        return acts[k:] + acts[:k]

    def step(self, tool: str, args: dict[str, Any]) -> StepResult:
        a = args.get("action")
        if a == self.PLAN[min(self.t, 2)]:
            self.t += 1
            return StepResult(f"ok: {a}", done=self.t >= 3)
        self.done_ok = False if a and a.startswith("put") else self.done_ok
        return StepResult(f"nothing happens: {a}")

    def oracle_action(self) -> tuple[str, dict[str, Any]] | None:
        return ("act", {"action": self.PLAN[self.t]}) if self.t < 3 else None

    def evaluate(self, final_answer: str | None) -> dict[str, Any]:
        ok = self.t >= 3 and self.done_ok
        return {"success": ok, "score": float(ok)}


class ActionLLM(SimulatedLLM):
    """SimulatedLLM for the action env: answers ``Action: <oracle action>`` (or a seeded wrong one)."""

    def _policy(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ChatResponse:
        env = getattr(self._local, "env", None)
        if env is None or not hasattr(env, "admissible_actions"):
            return ChatResponse(text="no")
        o = env.oracle_action()
        acts = env.admissible_actions() or []
        rng = self._local.rng
        a = o[1]["action"] if o else (acts[0] if acts else "look")
        if rng.random() < self.error_rate and acts:
            a = rng.choice(acts)
        return ChatResponse(text=f"Thinking.\nAction: {a}")


def _r(x: Any, nd: int = 6) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {str(k): _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    return x


DEC_KEYS = ("dp", "mode", "question", "options", "s1_value", "s1_conf", "deferred", "s2_value", "final_value",
            "gold", "gold_status", "gold_aux", "retry", "shadow", "s1_rank")
CALL_KEYS = ("component", "decision_point", "step", "model", "input_tokens", "output_tokens", "cost_usd")


def episode_fingerprint(t: dict[str, Any]) -> dict[str, Any]:
    steps = []
    for s in t.get("steps") or []:
        steps.append({"tool": s.get("tool"), "args": s.get("args"), "llm_calls": s.get("llm_calls"),
                      "reflection": bool(s.get("reflection")), "obs": (s.get("observation") or "")[:120],
                      "decisions": [{k: d.get(k) for k in DEC_KEYS} for d in s.get("decisions") or []]})
    calls = [{**{k: c.get(k) for k in CALL_KEYS}, "shadow": bool((c.get("meta") or {}).get("shadow"))}
             for c in t.get("calls") or []]
    return _r({"task": t.get("task_id"), "success": (t.get("result") or {}).get("success"), "error": t.get("error"),
               "final_answer": t.get("final_answer"), "steps": steps, "calls": calls})


def _cfg(name: str, **dps: DPConfig) -> HarnessConfig:
    return HarnessConfig(name=name, **dps)


D = DPConfig
# the modes that existed before E6b (harness/config.py VALID_MODES at cbedd65), one configuration each (tool params
# with the tool decision they need; memory reads with a write policy that stores something)
EXISTING: dict[str, Callable[[], HarnessConfig]] = {
    "llm_only": lambda: _cfg("llm_only"),
    "ts_replace": lambda: _cfg("ts_replace", tool_select=D(mode=Mode.REPLACE)),
    "ts_params_replace": lambda: _cfg("ts_params_replace", tool_select=D(mode=Mode.REPLACE),
                                      tool_params=D(mode=Mode.REPLACE)),
    "ts_cascade": lambda: _cfg("ts_cascade", tool_select=D(mode=Mode.CASCADE, tau=0.9),
                               tool_params=D(mode=Mode.CASCADE, tau=0.9)),
    "ts_cascade_always_defer": lambda: _cfg("ts_cascade_always_defer",
                                            tool_select=D(mode=Mode.CASCADE, tau=0.99, extra={"always_defer": True}),
                                            tool_params=D(mode=Mode.CASCADE, tau=0.13)),
    "ts_assist": lambda: _cfg("ts_assist", tool_select=D(mode=Mode.ASSIST, k=3)),
    "ts_guard": lambda: _cfg("ts_guard", tool_select=D(mode=Mode.GUARD, tau=0.5)),
    "shadow_tools": lambda: _cfg("shadow_tools", tool_select=D(mode=Mode.SHADOW), tool_params=D(mode=Mode.SHADOW),
                                 stop=D(mode=Mode.SHADOW)),
    "mem_always": lambda: _cfg("mem_always", memory_write=D(mode=Mode.ALWAYS), memory_read=D(mode=Mode.ALWAYS)),
    "mem_llm": lambda: _cfg("mem_llm", memory_write=D(mode=Mode.LLM), memory_read=D(mode=Mode.ALWAYS)),
    "mem_trigger_assist": lambda: _cfg("mem_trigger_assist", memory_write=D(mode=Mode.TRIGGER, tau=0.5),
                                       memory_read=D(mode=Mode.ASSIST, k=3)),
    "mem_write_shadow": lambda: _cfg("mem_write_shadow", memory_write=D(mode=Mode.SHADOW),
                                     memory_read=D(mode=Mode.ALWAYS)),
    "reflect_always": lambda: _cfg("reflect_always", reflection=D(mode=Mode.ALWAYS)),
    "reflect_trigger": lambda: _cfg("reflect_trigger", reflection=D(mode=Mode.TRIGGER, tau=0.5)),
    "reflect_shadow": lambda: _cfg("reflect_shadow", reflection=D(mode=Mode.SHADOW)),
    "stop_guard": lambda: _cfg("stop_guard", stop=D(mode=Mode.GUARD, tau=0.5)),
    "stop_trigger": lambda: _cfg("stop_trigger", stop=D(mode=Mode.TRIGGER, tau=0.6)),
    "stop_shadow": lambda: _cfg("stop_shadow", stop=D(mode=Mode.SHADOW)),
    "route_trigger": lambda: _cfg("route_trigger", routing=D(mode=Mode.TRIGGER, tau=0.5)),
    "route_shadow": lambda: _cfg("route_shadow", routing=D(mode=Mode.SHADOW)),
    "action_off": lambda: _cfg("action_off"),
    "action_replace": lambda: _cfg("action_replace", action=D(mode=Mode.REPLACE)),
    "action_cascade": lambda: _cfg("action_cascade", action=D(mode=Mode.CASCADE, tau=0.8)),
    "action_assist": lambda: _cfg("action_assist", action=D(mode=Mode.ASSIST, k=2)),
    "action_shadow": lambda: _cfg("action_shadow", action=D(mode=Mode.SHADOW)),
}


# The modes added by E6b and E6c (fingerprinted on the pre-E6d code, commit 7097f3c; fixture
# tests/fixtures/e6d_prior_modes_fingerprint.json, ``python tests/e6b_fingerprint.py e6``)
FIXTURE_E6 = ROOT / "tests" / "fixtures" / "e6d_prior_modes_fingerprint.json"
PRIOR_E6: dict[str, Callable[[], HarnessConfig]] = {
    "e6b_tp_assist": lambda: _cfg("tp_assist", tool_select=D(mode=Mode.CASCADE, tau=0.9),
                                  tool_params=D(mode=Mode.ASSIST, k=3)),
    "e6b_tp_guard": lambda: _cfg("tp_guard", tool_params=D(mode=Mode.GUARD, tau=0.9)),
    "e6b_mw_cascade": lambda: _cfg("mw_cascade", memory_write=D(mode=Mode.CASCADE, tau=0.9),
                                   memory_read=D(mode=Mode.ALWAYS)),
    "e6b_mw_guard": lambda: _cfg("mw_guard", memory_write=D(mode=Mode.GUARD, tau=0.5), memory_read=D(mode=Mode.ALWAYS)),
    "e6b_mr_cascade": lambda: _cfg("mr_cascade", memory_write=D(mode=Mode.ALWAYS),
                                   memory_read=D(mode=Mode.CASCADE, tau=0.9, k=3)),
    "e6b_mr_trigger": lambda: _cfg("mr_trigger", memory_write=D(mode=Mode.ALWAYS),
                                   memory_read=D(mode=Mode.TRIGGER, tau=0.6)),
    "e6b_rf_cascade": lambda: _cfg("rf_cascade", reflection=D(mode=Mode.CASCADE, tau=0.9)),
    "e6b_st_cascade": lambda: _cfg("st_cascade", stop=D(mode=Mode.CASCADE, tau=0.7)),
    "e6b_rt_cascade": lambda: _cfg("rt_cascade", routing=D(mode=Mode.CASCADE, tau=0.9)),
    "action_guard": lambda: _cfg("action_guard", action=D(mode=Mode.GUARD, tau=0.5)),
}


# The modes added by E6d (fingerprinted on the pre-E6e code, commit 765565b; fixture
# tests/fixtures/e6e_prior_modes_fingerprint.json, ``python tests/e6b_fingerprint.py e6d``)
FIXTURE_E6D = ROOT / "tests" / "fixtures" / "e6e_prior_modes_fingerprint.json"
PRIOR_E6D: dict[str, Callable[[], HarnessConfig]] = {
    "e6d_mw_assist": lambda: _cfg("mw_assist", memory_write=D(mode=Mode.ASSIST), memory_read=D(mode=Mode.ALWAYS)),
    "e6d_rf_assist": lambda: _cfg("rf_assist", reflection=D(mode=Mode.ASSIST)),
    "e6d_st_assist": lambda: _cfg("st_assist", stop=D(mode=Mode.ASSIST)),
    "e6d_rt_assist": lambda: _cfg("rt_assist", routing=D(mode=Mode.ASSIST)),
    "e6d_rf_guard": lambda: _cfg("rf_guard", reflection=D(mode=Mode.GUARD, tau=0.5)),
    "e6d_rt_guard": lambda: _cfg("rt_guard", routing=D(mode=Mode.GUARD, tau=0.5)),
    "e6d_mr_guard": lambda: _cfg("mr_guard", memory_write=D(mode=Mode.ALWAYS),
                                 memory_read=D(mode=Mode.GUARD, tau=0.5, k=3)),
}


# The modes added by E6e (fingerprinted on the pre-E6f code, commit d04fea0; fixture
# tests/fixtures/e6f_prior_modes_fingerprint.json, ``python tests/e6b_fingerprint.py e6e``)
FIXTURE_E6E = ROOT / "tests" / "fixtures" / "e6f_prior_modes_fingerprint.json"
PRIOR_E6E: dict[str, Callable[[], HarnessConfig]] = {
    "e6e_mw_replace": lambda: _cfg("mw_replace", memory_write=D(mode=Mode.REPLACE), memory_read=D(mode=Mode.ALWAYS)),
    "e6e_mr_replace": lambda: _cfg("mr_replace", memory_write=D(mode=Mode.ALWAYS), memory_read=D(mode=Mode.REPLACE)),
    "e6e_rf_replace": lambda: _cfg("rf_replace", reflection=D(mode=Mode.REPLACE)),
    "e6e_st_replace": lambda: _cfg("st_replace", stop=D(mode=Mode.REPLACE)),
    "e6e_rt_replace": lambda: _cfg("rt_replace", routing=D(mode=Mode.REPLACE)),
}


class GateSimLLM(SimulatedLLM):
    """SimulatedLLM that answers yes / no gate prompts deterministically (alternating), so the gate paths run."""

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse:
        last = str(messages[-1].get("content") or "")
        if not tools and last.rstrip().endswith("Answer yes or no."):
            self._n = getattr(self, "_n", 0) + 1
            return ChatResponse(text="yes" if self._n % 2 else "no", input_tokens=len(last) // 4, output_tokens=1,
                                latency_ms=1.0)
        return super()._chat(messages, tools, **kw)


def run_config(name: str) -> list[dict[str, Any]]:
    if name in PRIOR_E6 or name in PRIOR_E6D or name in PRIOR_E6E:
        return _run_prior(name)
    cfg = EXISTING[name]()
    s1 = SimulatedBackend(accuracy=0.8, seed=1)
    if name.startswith("action_"):
        llm = ActionLLM("scripted", error_rate=0.2, simulated_latency_ms=5, seed=0)
        agent = JevAgent(llm, cfg, s1=s1)
        envs = [TinyActionEnv(i) for i in range(N_TASKS)]
    else:
        llm = SimulatedLLM("scripted", error_rate=0.1, simulated_latency_ms=5, seed=0)
        small = SimulatedLLM("scripted-small", error_rate=0.3, simulated_latency_ms=2, seed=1)
        agent = JevAgent(llm, cfg, s1=s1, small_llm=small)
        envs = [JevGymEnv(seed=i, families="v2", strict=True, n_distractors=20, distractor_style="near",
                          noise_lines=5) for i in range(N_TASKS)]
    return [episode_fingerprint(agent.run(e).to_dict()) for e in envs]


def _run_prior(name: str) -> list[dict[str, Any]]:
    cfg = (PRIOR_E6.get(name) or PRIOR_E6D.get(name) or PRIOR_E6E[name])()
    s1 = SimulatedBackend(accuracy=0.8, seed=1)
    if name.startswith("action_"):
        agent = JevAgent(ActionLLM("scripted", error_rate=0.2, simulated_latency_ms=5, seed=0), cfg, s1=s1)
        envs: list[Any] = [TinyActionEnv(i) for i in range(N_TASKS)]
    else:
        agent = JevAgent(GateSimLLM("scripted", error_rate=0.1, simulated_latency_ms=5, seed=0), cfg, s1=s1,
                         small_llm=GateSimLLM("scripted-small", error_rate=0.3, simulated_latency_ms=2, seed=1))
        envs = [JevGymEnv(seed=i, families="v2", strict=True, n_distractors=20, distractor_style="near",
                          noise_lines=5) for i in range(N_TASKS)]
    return [episode_fingerprint(agent.run(e).to_dict()) for e in envs]


def digest(fps: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(fps, sort_keys=True, default=str).encode()).hexdigest()


def all_digests() -> dict[str, str]:
    return {name: digest(run_config(name)) for name in EXISTING}


def prior_e6_digests() -> dict[str, str]:
    return {name: digest(run_config(name)) for name in PRIOR_E6}


def prior_e6d_digests() -> dict[str, str]:
    return {name: digest(run_config(name)) for name in PRIOR_E6D}


def prior_e6e_digests() -> dict[str, str]:
    return {name: digest(run_config(name)) for name in PRIOR_E6E}


if __name__ == "__main__" and sys.argv[1:] == ["e6e"]:
    out = prior_e6e_digests()
    FIXTURE_E6E.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_E6E.write_text(json.dumps({"generated_at_commit": "d04fea0 (pre-E6f harness code)", "n_tasks": N_TASKS,
                                       "digests": out}, indent=1) + "\n")
    print(json.dumps(out, indent=1))
elif __name__ == "__main__" and sys.argv[1:] == ["e6d"]:
    out = prior_e6d_digests()
    FIXTURE_E6D.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_E6D.write_text(json.dumps({"generated_at_commit": "765565b (pre-E6e harness code)", "n_tasks": N_TASKS,
                                       "digests": out}, indent=1) + "\n")
    print(json.dumps(out, indent=1))
elif __name__ == "__main__" and sys.argv[1:] == ["e6"]:
    out = prior_e6_digests()
    FIXTURE_E6.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_E6.write_text(json.dumps({"generated_at_commit": "7097f3c (pre-E6d harness code)", "n_tasks": N_TASKS,
                                      "digests": out}, indent=1) + "\n")
    print(json.dumps(out, indent=1))
elif __name__ == "__main__":
    out = all_digests()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({"generated_at_commit": "cbedd65 (pre-E6b harness code)", "n_tasks": N_TASKS,
                                   "digests": out}, indent=1) + "\n")
    print(json.dumps(out, indent=1))
