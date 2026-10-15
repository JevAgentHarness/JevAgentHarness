"""runs/ -> paper numbers: formatting, simulated-run gating, wrapped-key checking."""

from __future__ import annotations

import json

import yaml

from jevharness.bench import paper_numbers as pn


def _decision_run(root, name, backends, rows):
    d = root / name
    d.mkdir(parents=True)
    (d / "experiment.yaml").write_text(yaml.safe_dump({"name": name, "backends": backends}))
    (d / "summary.json").write_text(json.dumps(rows))
    return d


def test_fmt_rules():
    assert pn.fmt(("r", "b", "tool_select", "bfcl", "acc"), 0.9166) == "91.7"
    assert pn.fmt(("r", "a", "vs_baseline", "delta"), -0.05) == "-5.0"
    assert pn.fmt(("r", "a", "success_ci", "lo"), 0.8) == "80.0"
    assert pn.fmt(("r", "a", "ece"), 0.08412) == "0.084"
    assert pn.fmt(("r", "a", "p_worse"), 0.0001) == "$<$0.001"
    assert pn.fmt(("r", "a", "latency_p50_ms"), 729.35) == "729"
    assert pn.fmt(("r", "a", "cost_per_1k_usd"), 0.13361) == "0.134"
    assert pn.fmt(("r", "a", "n"), 60.0) == "60"


def test_collect_decision_run_and_simulated_gate(tmp_path):
    row = {"n": 10, "acc": 0.9, "ece": 0.05, "latency_p50_ms": 700.0, "cost_per_1k_usd": 0.1}
    _decision_run(tmp_path, "e1", {"jev": {"type": "jev", "provider": "openrouter"}}, {"jev|tool_select|bfcl": row})
    _decision_run(tmp_path, "sim", {"s": {"type": "simulated"}}, {"s|tool_select|bfcl": row})
    values, prov = pn.collect(tmp_path)
    assert values["e1/jev/tool_select/bfcl/acc"] == "90.0"
    assert not any(k.startswith(("sim/", "SIM/")) for k in values)
    assert {p["run"]: p["included"] for p in prov} == {"e1": True, "sim": False}
    values, _ = pn.collect(tmp_path, include_simulated=True)
    assert values["SIM/sim/s/tool_select/bfcl/acc"] == "90.0"  # never collides with a headline key
    tex = pn.to_tex(values, prov)
    assert r"\defres{e1/jev/tool_select/bfcl/acc}{90.0}" in tex and r"\providecommand{\res}" in tex


def test_used_keys_expands_wrapper_macros(tmp_path):
    (tmp_path / "t.tex").write_text(
        "% \\res{commented/out}\n"
        "\\newcommand{\\eone}[2]{\\res{e1/#1/#2}}\n"
        "\\newcommand{\\eonerow}[2]{#1 & \\eone{#2}{tool_select/bfcl/acc}\n  & \\eone{#2}{tool_select/bfcl/ece}\\\\}\n"
        "\\eonerow{\\jev{}}{jev}\n"
        "Direct: \\res{e2/llm_only/success} and \\eone{fc}{tool_select/bfcl/acc}.\n")
    used = pn.used_keys(tmp_path)
    assert set(used) == {"e1/jev/tool_select/bfcl/acc", "e1/jev/tool_select/bfcl/ece", "e2/llm_only/success",
                         "e1/fc/tool_select/bfcl/acc"}


def test_non_trajectory_jsonl_is_ignored(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    (d / "x.jsonl").write_text(json.dumps({"foo": 1}) + "\n")
    values, prov = pn.collect(tmp_path)
    assert values == {} and prov == []


def test_bfcl_rows_without_functions_are_skipped(tmp_path, monkeypatch):
    """4 live_irrelevance rows have an empty function list; they must not become one-option Choices."""
    from jevharness.bench.suites import static

    rows = [{"id": "a", "question": [[{"role": "user", "content": "hi"}]], "function": []},
            {"id": "b", "question": [[{"role": "user", "content": "weather?"}]],
             "function": [{"name": "get_weather", "description": "d", "parameters": {"type": "dict", "properties": {}}}]}]
    d = tmp_path / "bfcl"
    d.mkdir()
    (d / "BFCL_v4_live_irrelevance.json").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(static, "_download", lambda url, dest: dest)
    items = static.load_bfcl(categories=["live_irrelevance"], data_dir=tmp_path)
    assert [i.id for i in items] == ["b/tool"]
