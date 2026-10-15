"""scripts/explore_e6b_modes.py (exploratory row E6b) on simulated run directories written by the real harness (offline:
simulated System-2 and System-1 on JevGym development seeds): spend, the registered projection gate, the readout
(paired comparison per cell, coverage / activity / accuracy per mode, the fixed verdict rule) and the TeX keys; and
the module map's exploratory E6b cells."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import explore_e6b_modes as E  # noqa: E402

from jevharness.bench.runner import expand_arms  # noqa: E402
from jevharness.bench.suites.jevgym import JevGymEnv  # noqa: E402
from jevharness.decision.simulated import SimulatedBackend  # noqa: E402
from jevharness.harness.agent import JevAgent  # noqa: E402
from jevharness.harness.config import DP, E6B_CELLS, Mode  # noqa: E402
from jevharness.llm.simulated import SimulatedLLM  # noqa: E402

EVAL = ROOT / "configs/explore/e6b_modes/eval.yaml"
SMOKE = ROOT / "configs/explore/e6b_modes/smoke.yaml"


def test_configs_cover_every_e6b_cell_once_with_the_registered_settings() -> None:
    exp = yaml.safe_load(EVAL.read_text())
    cfgs = {c.name: c for c in expand_arms(exp)}
    assert set(cfgs) == set(E.ARMS)
    seen = set()
    for arm, (dp, mode) in E.CELLS.items():
        c = cfgs[arm].validate_modes()
        assert c.dp(DP(dp)).mode == Mode(mode)
        seen.add((DP(dp), Mode(mode)))
    assert seen == set(E6B_CELLS)
    # thresholds fixed a priori: every cascade 0.9; memory-write guard 0.5 (its trigger); argument guard and read
    # trigger 0.9 (no registered tau there); k = 3; the argument assist behind the C1 tool cascade at Flash's 1.00
    assert all(cfgs[a].dp(DP(dp)).tau == 0.9 for a, (dp, m) in E.CELLS.items() if m == "cascade")
    assert cfgs["mw_guard"].memory_write.tau == 0.5 and cfgs["tp_guard"].tool_params.tau == 0.9
    assert cfgs["mr_trigger"].memory_read.tau == 0.9 and cfgs["mr_cascade"].memory_read.k == 3
    assert cfgs["tp_assist"].tool_params.k == 3 and cfgs["tp_assist"].tool_select.tau == 1.0
    assert exp["suite"] == {"name": "jevgym", "n": 200, "seed0": 300800, "families": "v2", "distractor_style": "near",
                            "n_distractors": 60, "noise_lines": 20, "strict": True}
    assert exp["schedule"] == {"mode": "task_major", "seed": 732} and exp["llm"]["model"] == "gemini-3.5-flash"
    assert exp["llm"]["reasoning_effort"] == "minimal" and exp["s1"] == {"type": "jev", "provider": "openrouter"}
    sm = yaml.safe_load(SMOKE.read_text())
    assert sm["suite"]["seed0"] == 900 and sm["n_tasks"] == 2 and sm["arms"] == exp["arms"]


class GateLLM(SimulatedLLM):
    """SimulatedLLM that answers the yes / no gate prompts (memory write gate, E6b LLM gates) with "yes" on every other
    call, so the deferral and guard paths have proposals in the simulation."""

    def _chat(self, messages, tools=None, **kw):  # type: ignore[no-untyped-def]
        last = str(messages[-1].get("content") or "")
        if not tools and last.rstrip().endswith("Answer yes or no."):
            from jevharness.llm.base import ChatResponse
            self._n = getattr(self, "_n", 0) + 1
            return ChatResponse(text="yes" if self._n % 2 else "no", input_tokens=40, output_tokens=1, latency_ms=1.0)
        return super()._chat(messages, tools, **kw)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Simulated eval + smoke runs of every arm (JevGym development seeds 0-5; the real harness and ledger)."""
    root = tmp_path_factory.mktemp("runs")
    exp = yaml.safe_load(EVAL.read_text())
    cfgs = {c.name: c for c in expand_arms(exp)}
    for run, n in ((E.RUN_EVAL, 6), (E.RUN_SMOKE, 2)):
        d = root / run
        d.mkdir()
        for name, cfg in cfgs.items():
            if name == "tp_assist":   # the simulated decider never reports confidence 1.0: accept tools at 0.9 here
                cfg = cfg.model_copy(deep=True)
                cfg.tool_select.tau = 0.9
            llm = GateLLM("gemini-3.5-flash-lite" if name == "small_only" else "gemini-3.5-flash", error_rate=0.15,
                          simulated_latency_ms=5, seed=0)
            small = GateLLM("gemini-3.5-flash-lite", error_rate=0.3, simulated_latency_ms=2, seed=1)
            agent = JevAgent(llm, cfg, s1=SimulatedBackend(accuracy=0.8, seed=1, price_per_m_input=0.1),
                             small_llm=small)
            with open(d / f"{name}.jsonl", "w") as f:
                for i in range(n):
                    t = agent.run(JevGymEnv(seed=i, families="v2", strict=True, n_distractors=20,
                                            distractor_style="near", noise_lines=5)).to_dict()
                    f.write(json.dumps(t, default=str) + "\n")
    return root


def test_spend_and_projection_gate(runs: Path) -> None:
    s = E.total_spend(runs)
    assert s["records"] == 11 * 6 + 11 * 2 and s["total"] > 0
    p = E.projection(runs, n_eval=200, safety=1.5, cap=1e9)
    assert p["decision"] == "run all 200" and not p["missing_arms"]
    assert p["projected_200"] == pytest.approx(p["per_task_all_arms"] * 200 * 1.5)
    tight = E.projection(runs, cap=s["total"] + p["per_task_all_arms"] * 100 * 1.5 + 1e-9)
    assert tight["decision"] == "run the first 100"
    assert E.projection(runs, cap=s["total"])["decision"] == "stop after the smoke"


def test_summary_cells_metrics_and_tex(runs: Path) -> None:
    res = E.summarize(runs, n_boot=200, seed=0)
    cells = res["cells"]
    assert set(cells) == set(E.CELLS) and all(c["state"] == "exploratory" for c in cells.values())
    for arm, c in cells.items():
        k, pm = c["cmp"], c["points"]
        assert k["n"] == 6 and k["wins"] + k["losses"] <= 6
        assert pm["n_decisions"] > 0, arm
        assert 0.0 <= pm["acted"] <= 1.0 and 0.0 <= pm["coverage"] <= 1.0, (arm, pm)
        assert c["reference"] == ("small_only" if arm == "rt_cascade" else "llm_only")
    assert cells["tp_assist"]["points"]["coverage"] == 0.0 and cells["tp_assist"]["points"]["acted"] == 1.0
    assert cells["mr_trigger"]["points"]["coverage"] == 1.0 and cells["tp_guard"]["points"]["acted_kind"] == "vetoed"
    assert "vs_flash" in cells["rt_cascade"]
    keys = E.to_tex_keys(res)
    assert keys["e6b/count/run"] == "9" and "e6b/tp_assist/delta" in keys and "e6b/rt_cascade/flash/delta" in keys
    assert not any(v.lower() in ("nan", "inf") for v in keys.values())
    md = E.to_markdown(res)
    assert "| tool_params/assist | `tp_assist` | llm_only | 6 |" in md
    js = json.dumps(E.MM._clean(res), default=E.MM._json_default)
    assert "NaN" not in js and "Infinity" not in js
