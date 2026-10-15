"""scripts/filter_res_keys.py: the paper's \\res keys out of the full paper_numbers set (literal keys, prefixes of
templated keys, templated suffixes on Stage-1 main runs only)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("filter_res_keys", ROOT / "scripts" / "filter_res_keys.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

MACROS = r"""
\newcommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}
\newcommand{\resm}[1]{\ensuremath{\res{#1}}}
\newcommand{\ethree}[3]{#1 & \res{#2/#3/success} & \res{#2/#3/vs_baseline/delta}}
\newcommand{\lfam}[2]{#1 & \resm{primaries/#1/#2/delta} & \resm{primaries/#1/#2/delta_ci/lo}}
"""
BODY = r"""
Literal \res{e2a_jevgym_tool_decisions/llm_only/success} and \resm{tau2/retail/llm_only/cost_usd}.
\ethree{LLM-only}{tau2/airline_test}{llm_only}
\lfam{C1F}{jev}
% \res{commented/out/key}
"""
VALUES = {
    "e2a_jevgym_tool_decisions/llm_only/success": "90.0",             # literal
    "tau2/retail/llm_only/cost_usd": "0.08",                          # literal through \resm
    "tau2/airline_test/llm_only/success": "60.0",                     # literal through \ethree
    "e2a_jevgym_tool_decisions/jev_ts_replace/success": "80.0",       # templated suffix, main run
    "tau2/retail/jev_guard/vs_baseline/delta": "+1.0",                # templated suffix, main run
    "pilot_jev/e2a_jevgym_tool_decisions/jev_ts_replace/success": "70.0",  # suffix, but a pilot run: dropped
    "e2a_jevgym_tool_decisions/jev_ts_replace/ece": "0.05",           # no rule: dropped
    "primaries/C2F/intern/n_pairs": "500",                            # templated prefix
    "commented/out/key": "1",                                         # only in a comment: dropped
}


def _paper(tmp_path: Path) -> Path:
    p = tmp_path / "paper"
    (p / "sections").mkdir(parents=True)
    (p / "generated").mkdir()
    (p / "macros.tex").write_text(MACROS)
    (p / "sections" / "results.tex").write_text(BODY)
    (p / "generated" / "results.tex").write_text(r"\res{generated/only/key}")  # generated files are not sources
    return p


def test_rules_from_templated_keys(tmp_path):
    temps = F.templated_keys(_paper(tmp_path))
    assert {"#2/#3/success", "#2/#3/vs_baseline/delta", "primaries/#1/#2/delta", "primaries/#1/#2/delta_ci/lo"} <= temps
    prefixes, suffixes = F.rules(temps)
    assert prefixes == {"primaries/"}
    assert suffixes == {"/success", "/vs_baseline/delta"}          # the \resm pass-through "#1" gives nothing
    assert F.rules(["#1", "#1/#2", "abc/#1/#2_ci/lo", "#1/#2_ci/hi"]) == ({"abc/"}, {"_ci/hi"})


def test_filter_keeps_literal_prefix_and_main_run_suffix_keys(tmp_path):
    out = tmp_path / "results.tex"
    full = tmp_path / "full.json"
    full.write_text(json.dumps({"provenance": [{"run": "e2a_jevgym_tool_decisions", "included": True}],
                                "values": VALUES}))
    assert F.main([str(full), "--paper", str(_paper(tmp_path)), "--out", str(out),
                   "--json-out", str(tmp_path / "kept.json")]) == 0
    kept = json.loads((tmp_path / "kept.json").read_text())["values"]
    assert set(kept) == {"e2a_jevgym_tool_decisions/llm_only/success", "tau2/retail/llm_only/cost_usd",
                         "tau2/airline_test/llm_only/success", "e2a_jevgym_tool_decisions/jev_ts_replace/success",
                         "tau2/retail/jev_guard/vs_baseline/delta", "primaries/C2F/intern/n_pairs"}
    tex = out.read_text()
    assert r"\defres{tau2/retail/jev_guard/vs_baseline/delta}{+1.0}" in tex
    assert "pilot_jev/" not in tex.split("filtered by", 1)[1].split("\n", 1)[1]   # only the provenance mentions runs
    assert "% filtered by scripts/filter_res_keys.py: 6 of 9 keys (literal 3, templated prefix 1, " \
           "templated suffix on a Stage-1 main run 2)" in tex
    assert tex.count(r"\providecommand{\res}") == 1 and tex.rstrip().endswith(r"\makeatother")


def test_main_run_match_is_by_whole_run_path():
    assert F.is_main_run_key("tau2/airline_test/llm_only/success")
    assert F.is_main_run_key("e2a_jevgym_tool_decisions_intern/intern_ts_replace/success")
    assert not F.is_main_run_key("smoke_main/e2a_jevgym_tool_decisions/llm_only/success")
    assert not F.is_main_run_key("e2c_jevgym_baselines_check/llm_only/success")
    kept, counts = F.filter_keys({"e2c_jevgym_baselines_check/a/success": "1", "e2c_jevgym_baselines/a/success": "1"},
                                 [], [], ["/success"])
    assert set(kept) == {"e2c_jevgym_baselines/a/success"} and counts == {"literal": 0, "prefix": 0, "suffix": 1}
