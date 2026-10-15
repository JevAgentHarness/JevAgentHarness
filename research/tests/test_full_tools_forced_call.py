"""``tool_params.extra.full_tools`` (off by default; E13 cache diagnosis, docs/PREREG_A7_STAGE2.md Deviations
2026-10-07 "E13: GPT-6-Luna execution as run"): the forced argument call of a delegated step offers the step's whole
tool list (same order) instead of the one chosen tool, still forced to it. Default arms are untouched; with the key the
simulated episodes make the same decisions, calls and outcomes, and only the offered tool list changes."""
from __future__ import annotations

from typing import Any

from jevharness.bench.suites.jevgym import JevGymEnv
from jevharness.decision.simulated import SimulatedBackend
from jevharness.harness.agent import JevAgent
from jevharness.harness.config import DPConfig, HarnessConfig, Mode
from jevharness.llm.simulated import SimulatedLLM


class RecordingLLM(SimulatedLLM):
    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.offered: list[tuple[list[str], Any]] = []
        self.episode = -1
        self.by_episode: list[list[tuple[list[str], Any]]] = []

    def _chat(self, messages, tools=None, **kw):  # noqa: ANN001, ANN201
        rec = ([t["function"]["name"] for t in tools or []], kw.get("tool_choice"))
        self.offered.append(rec)
        self.by_episode[self.episode].append(rec)
        return super()._chat(messages, tools, **kw)


def _episodes(full: bool | None, n: int = 4) -> tuple[list[dict[str, Any]], RecordingLLM]:
    extra = {} if full is None else {"full_tools": full}
    cfg = HarnessConfig(name="jev_refit", max_steps=20, tool_select=DPConfig(mode=Mode.CASCADE, tau=0.5),
                        tool_params=DPConfig(mode=Mode.CASCADE, tau=0.6, extra=extra),
                        reflection=DPConfig(mode=Mode.TRIGGER, tau=0.5))
    llm = RecordingLLM("scripted", error_rate=0.1, simulated_latency_ms=0, seed=0)
    agent = JevAgent(llm, cfg, s1=SimulatedBackend(accuracy=0.85, seed=1, latency_ms=0.0))
    out = []
    llm.step_tools = []
    for i in range(n):
        env = JevGymEnv(seed=i, families="multi", multi_customers=4, strict=True, n_distractors=20,
                        distractor_style="near", noise_lines=5)
        llm.episode = i
        llm.by_episode.append([])
        llm.step_tools.append([t["function"]["name"] for t in env.tools()] + [cfg.final_tool])
        out.append(agent.run(env).to_dict())
    return out, llm


def _view(trajs: list[dict[str, Any]]) -> list[Any]:
    return [(t["task_id"], t["result"], t["final_answer"],
             [(s["tool"], s["args"], [(d["dp"], d["s1_value"], d["final_value"], d["deferred"]) for d in s["decisions"]])
              for s in t["steps"]],
             [(c["component"], c["decision_point"]) for c in t["calls"]]) for t in trajs]


def test_default_and_explicit_off_offer_one_tool():
    base, llm = _episodes(None)
    off, _ = _episodes(False)
    assert _view(base) == _view(off)
    forced = [(names, tc) for names, tc in llm.offered if tc]
    assert forced and all(len(names) == 1 and names[0] == tc["function"]["name"] for names, tc in forced)


def test_full_tools_offers_the_step_list_and_changes_nothing_else():
    base, llm0 = _episodes(None)
    full, llm1 = _episodes(True)
    assert _view(full) == _view(base)                      # same decisions, calls and outcomes
    n_forced = 0
    for step_tools, calls in zip(llm1.step_tools, llm1.by_episode):
        for names, tc in calls:
            if tc:                                    # the full list of the step, in its order, forced to one tool
                assert names == step_tools and tc["function"]["name"] in names
                n_forced += 1
            elif names:
                assert names == step_tools            # the LLM's own tool-choice calls offer the same list
    assert n_forced and n_forced == len([1 for _, tc in llm0.offered if tc])
