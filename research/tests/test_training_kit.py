"""Training-agent kit: data/train_v0, the GPU setup / smoke / data scripts, and the harvest / serving fixes they rely on
(docs/TRAINING_AGENT_BRIEF.md). Offline: simulated LLM / System-1, a tiny local Qwen3-style model."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V0 = ROOT / "data/train_v0"
SIM_LLM = {"provider": "simulated", "model": "scripted", "error_rate": 0.25, "simulated_latency_ms": 0}
SIM_S1 = {"type": "simulated", "accuracy": 0.8, "latency_ms": 0, "price_per_m_input": 0.042}
SHADOW_ARM = {"name": "shadow", "tool_select": {"mode": "shadow"}, "tool_params": {"mode": "shadow"},
              "stop": {"mode": "shadow"}}


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def v2_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A simulated shadow run on the calibrated v2 JevGym suite (69 tools > 52 option codes), training seeds."""
    from jevharness.bench.runner import run_experiment

    out = tmp_path_factory.mktemp("runs")
    exp = {"name": "kit_v2", "suite": {"name": "jevgym", "n": 4, "seed0": 10000, "families": "v2",
                                       "distractor_style": "near", "n_distractors": 60, "noise_lines": 20,
                                       "strict": True},
           "llm": SIM_LLM, "s1": SIM_S1, "workers": 2, "arms": [SHADOW_ARM]}
    run_experiment(exp, out, progress=False)
    return out / "kit_v2"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    sys.path.insert(0, str(ROOT / "scripts/train"))
    from make_tiny_model import make_tiny_model

    return make_tiny_model(tmp_path_factory.mktemp("tiny"))


# ------------------------------------------------------------------ harvest fixes (regressions)
def test_harvest_keeps_choices_over_52_options_with_gold(v2_run: Path):
    """Before: every v2 tool_select decision (69 tools) was dropped by harvest_trajectories."""
    from jevharness.training.data import harvest_trajectories

    ex = harvest_trajectories([v2_run / "shadow.jsonl"])
    ts = [e for e in ex if e["dp"] == "tool_select"]
    assert ts, "tool_select decisions of the 69-tool suite must be harvested"
    for e in ts:
        assert len(e["options"]) == 52 and e["meta"]["n_options_logged"] == 69
        gold = [o for o, v in e["target"].items() if v == 1.0]
        assert len(gold) == 1 and gold[0] in e["options"]
    again = harvest_trajectories([v2_run / "shadow.jsonl"])
    assert [e["options"] for e in again] == [e["options"] for e in ex]  # seeded subsample: deterministic
    assert len({e["id"] for e in ex}) == len(ex)  # several tool_params per step get unique ids


def test_harvested_tool_options_match_the_served_prompt(v2_run: Path):
    """Training options must be the strings serve_jev renders for the harness's Choice (`name -- description`)."""
    from jevharness.decision import Choice
    from jevharness.harness.tools import finish_tool, tool_desc, tool_name
    from jevharness.training.data import harvest_trajectories, render
    from jevharness.training.grpo_online import question_to_example
    from jevharness.training.harvest import tool_descriptions

    desc = tool_descriptions(v2_run / "experiment.yaml")
    ex = [e for e in harvest_trajectories([v2_run / "shadow.jsonl"], descriptions=desc) if e["dp"] == "tool_select"]
    e = ex[0]
    names = [o.split(" -- ")[0] for o in e["options"]]
    from jevharness.bench.suites import load_suite

    env = next(f() for f in load_suite("jevgym", n=4, seed0=10000, families="v2", distractor_style="near",
                                         n_distractors=60, noise_lines=20, strict=True)
               if f().task_id == e["id"].split("/")[0])
    by = {tool_name(t): t for t in env.tools() + [finish_tool()]}
    served = question_to_example(e["state"], Choice("Which tool should the agent call next?", names,
                                                    [tool_desc(by[n]) for n in names]))
    assert served["options"] == e["options"] and served["question"] == e["question"]
    assert render(e)[0] == render(served)[0]  # byte-identical prompt
    assert e["meta"]["options_rendered"] == "name -- description"


def test_harvest_cli_renders_descriptions_and_reads_gzip(v2_run: Path, tmp_path: Path):
    from jevharness.training import harvest
    from jevharness.training.data import read_jsonl

    d = tmp_path / "gz_run"
    d.mkdir()
    (d / "experiment.yaml").write_text((v2_run / "experiment.yaml").read_text())
    (d / "shadow.jsonl.gz").write_bytes(gzip.compress((v2_run / "shadow.jsonl").read_bytes()))
    sys_argv = sys.argv
    try:
        sys.argv = ["harvest", "--runs", str(d), "--out", str(tmp_path / "out"), "--dev-frac", "0.25"]
        harvest.main()
    finally:
        sys.argv = sys_argv
    ex = read_jsonl(tmp_path / "out/train.jsonl") + read_jsonl(tmp_path / "out/dev.jsonl")
    assert Counter(e["dp"] for e in ex)["tool_select"] > 0
    assert all(" -- " in o for e in ex if e["dp"] == "tool_select" for o in e["options"])


def test_jsonl_gzip_roundtrip(tmp_path: Path):
    from jevharness.training.data import read_jsonl, write_jsonl

    rows = [{"id": "a/0/x", "options": ["ü", "b"]}]
    write_jsonl(rows, tmp_path / "x.jsonl.gz")
    assert gzip.open(tmp_path / "x.jsonl.gz", "rt").read().startswith("{")
    assert read_jsonl(tmp_path / "x.jsonl.gz") == rows
    first = (tmp_path / "x.jsonl.gz").read_bytes()
    write_jsonl(rows, tmp_path / "x.jsonl.gz")
    assert (tmp_path / "x.jsonl.gz").read_bytes() == first  # reproducible bytes (no header timestamp)


# ------------------------------------------------------------------ When2Call train (regressions)
def _w2c_dirs(tmp_path: Path) -> Path:
    tools = [json.dumps({"name": "get_weather", "description": "Weather for a city.",
                         "parameters": {"type": "dict", "properties": {"city": {"type": "str", "description": "City"},
                                                                       "units": {"type": "str"}}},
                         "required": ["city"]})]
    d = tmp_path / "raw/when2call"
    (d / "train").mkdir(parents=True)
    (d / "test").mkdir(parents=True)
    train = [{"tools": tools, "messages": [{"role": "user", "content": f"Weather please {i}"},
                                           {"role": "assistant", "content": "Which city do you mean?" if i % 2 else
                                            "Sorry, I am unable to help with that."}]} for i in range(6)]
    (d / "train/when2call_train_sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    t = json.loads(tools[0])
    t["parameters"]["required"] = t.pop("required")  # the TEST file keeps `required` inside `parameters`
    test = {"uuid": "u1", "question": "Weather please 0", "tools": [json.dumps(t)], "correct_answer": "cannot_answer",
            "source": "x"}
    (d / "test/when2call_test_mcq.jsonl").write_text(json.dumps(test) + "\n")
    return tmp_path / "raw"


def test_when2call_train_parses_the_sft_schema_like_the_test_items(tmp_path: Path):
    """Before: load_when2call(split='train') raised KeyError('question') (the SFT file is {tools, messages})."""
    from jevharness.bench.suites.static import W2C_OPTIONS, load_when2call
    from jevharness.training.data import split_by_task
    from jevharness.training.harvest import from_when2call_train

    raw = _w2c_dirs(tmp_path)
    ex = from_when2call_train(data_dir=raw)
    assert len(ex) == 6 and len({e["id"] for e in ex}) == 6
    labels = Counter(e["meta"]["w2c_label"] for e in ex)
    assert labels == {"request_for_info": 3, "cannot_answer": 3}
    assert all(e["target"][W2C_OPTIONS[e["meta"]["w2c_label"]]] == 1.0 for e in ex)
    test = load_when2call("test", data_dir=raw)[0]
    assert ex[0]["state"] == test.state  # same tools -> same System-1 state (incl. "required" flags)
    assert "city (str, required)" in ex[0]["state"] and ex[0]["question"] == test.question.text
    assert ex[0]["options"] == test.question.options
    tr, dv = split_by_task(ex, 0.34)
    assert tr and dv  # one task per row (before: all rows shared the task "w2c")


def test_xlam_rows_add_when2call_format_call_items():
    from jevharness.bench.suites.static import W2C_OPTIONS
    from jevharness.training.harvest import from_xlam

    rows = [{"id": 7, "query": "Weather in Paris?", "answers": json.dumps([{"name": "w", "arguments": {"city": "Paris"}}]),
             "tools": json.dumps([{"name": "w", "description": "Weather",
                                   "parameters": {"city": {"type": "str", "description": "City"}}}])}]
    ex = {e["id"]: e for e in from_xlam(rows, n_distractors=0)}
    w = ex["xlam7/when2call"]
    assert w["target"][W2C_OPTIONS["tool_call"]] == 1.0 and "city (str, required)" in w["state"]


# ------------------------------------------------------------------ serving / MC fixes (regressions)
def test_engine_serves_a_full_finetuned_checkpoint_passed_as_adapter(tiny: Path, tmp_path: Path):
    """80_distill_small.sh trains the student with LORA_R=0; 90 then passes it as ADAPTER (before: PeftModel crash)."""
    from jevharness.training.data import write_jsonl
    from jevharness.training.serve_jev import Engine, resolve_checkpoint
    from jevharness.training.train_decision import main as train_main

    data = [{"id": f"t{i}/0/x", "state": "S", "question": "Q?", "options": ["a", "b"],
             "target": {"a": 0.0, "b": 1.0}} for i in range(8)]
    write_jsonl(data, tmp_path / "d.jsonl")
    out = tmp_path / "full"
    train_main(["--model", str(tiny), "--train", str(tmp_path / "d.jsonl"), "--out", str(out), "--lora-r", "0",
                "--lr", "1e-3", "--max-steps", "2", "--batch-size", "4", "--grad-accum", "1", "--max-len", "128",
                "--no-bf16"])
    assert not (out / "adapter_config.json").exists() and (out / "config.json").exists()
    assert resolve_checkpoint(str(tiny), str(out)) == (str(out), None)
    assert resolve_checkpoint("base", None) == ("base", None)
    eng = Engine(str(tiny), str(out), name="full", max_len=128)
    p = eng.answer("S", {"type": "choice", "instructions": "Q?", "criteria": {"a": None, "b": None}})
    assert abs(sum(p["probabilities"].values()) - 1) < 1e-5


def test_mc_rollouts_render_descriptions_and_write_a_cost_summary(v2_run: Path, tmp_path: Path):
    import yaml

    from jevharness.training import mc_rollouts
    from jevharness.training.data import read_jsonl

    exp = yaml.safe_load((v2_run / "experiment.yaml").read_text())
    exp["llm"] = SIM_LLM
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(exp))
    out = tmp_path / "mc.jsonl"
    mc_rollouts.main(["--exp", str(tmp_path / "exp.yaml"), "--runs", str(v2_run / "shadow.jsonl"),
                      "--max-states", "2", "--k-alt", "2", "--m", "2", "--out", str(out)])
    rows = read_jsonl(out)
    assert len(rows) == 2
    for r in rows:
        assert all(" -- " in o for o in r["options"]) and set(r["q_values"]) == set(r["options"])
        assert set(r["target"]) == set(r["options"]) and r["meta"]["label_source"] == "mc"
    s = json.loads(Path(f"{out}.summary.json").read_text())
    assert s["episodes"] == 2 * 2 * 2 and s["cost_usd"] >= 0 and s["executor"] == SIM_LLM


# ------------------------------------------------------------------ data/train_v0 integrity
def test_train_v0_manifest_matches_files_and_size_limit():
    m = json.loads((V0 / "MANIFEST.json").read_text())
    total = 0
    for rel, info in m["files"].items():
        p = V0 / rel
        assert p.exists(), rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == info["sha256"], f"{rel} changed: re-run the manifest"
        total += p.stat().st_size
    listed = {str(p.relative_to(V0)) for p in V0.rglob("*") if p.is_file() and p.name != "MANIFEST.json"}
    assert listed == set(m["files"])
    assert total < 15 * 2**20 and m["holdout"]["eval_seeds_found"] == []
    assert "STAND-IN" in json.dumps(m["notes"]["models"])


def test_train_v0_has_no_eval_items_and_valid_targets():
    from jevharness.training.data import read_jsonl
    from jevharness.training.harvest import jevgym_seed

    files = ["train.jsonl", "dev.jsonl", "train_teacher_standin.jsonl", "mc_standin.jsonl",
             "replay_when2call.jsonl.gz", "dev_when2call.jsonl.gz"]
    split_tasks = {}
    for f in files:
        ex = read_jsonl(V0 / f)
        assert ex, f
        for e in ex:
            s = jevgym_seed(e["id"])
            assert s is None or s >= 1000, (f, e["id"])          # E2 eval seeds 0..199 / dev 5000+ never here
            assert not e["id"].startswith(("w2c/", "bfcl", "simple", "multiple", "live_", "irrelevance")), e["id"]
            assert 2 <= len(e["options"]) <= 52 and set(e["target"]) <= set(e["options"])
            assert abs(sum(e["target"].values()) - 1) < 1e-6
            if e["dp"] == "tool_select" and e["id"].startswith("jevgym"):
                assert all(" -- " in o for o in e["options"])     # rendered like the served prompt
        if f in ("train.jsonl", "dev.jsonl"):
            split_tasks[f] = {e["id"].split("/")[0] for e in ex}
            assert {e["meta"]["label_source"] for e in ex} == {"gold"}
    assert split_tasks["train.jsonl"].isdisjoint(split_tasks["dev.jsonl"])
    teacher = read_jsonl(V0 / "train_teacher_standin.jsonl")
    assert len(teacher) == len(read_jsonl(V0 / "train.jsonl"))
    assert {e["meta"]["label_source"] for e in teacher} <= {"teacher", "gold"}
    summ = json.loads((V0 / "train_teacher_standin.jsonl.summary.json").read_text())
    assert [t["spec"]["llm"]["model"] for t in summ["teachers"].values()] == ["gemini-3.5-flash-lite"]


def test_train_v0_trajectories_are_training_seed_shadow_runs():
    import yaml

    for name in ("train_v0_jevgym_v2", "train_v0_jevgym_v1"):
        d = V0 / "trajectories" / name
        exp = yaml.safe_load((d / "experiment.yaml").read_text())
        assert exp["suite"]["seed0"] >= 1000 and [a["name"] for a in exp["arms"]] == ["shadow"]
        with gzip.open(d / "shadow.jsonl.gz", "rt") as f:
            first = json.loads(f.readline())
        assert first["config"] == "shadow" and int(re.search(r"-(\d+)-d", first["task_id"]).group(1)) >= 1000


# ------------------------------------------------------------------ scripts and brief
KIT = ["setup_gpu.sh", "smoke_gpu.sh", "make_data_v0.sh"]


@pytest.mark.parametrize("script", KIT)
def test_kit_scripts_parse(script: str):
    subprocess.run(["bash", "-n", str(ROOT / "scripts/train" / script)], check=True)


def test_smoke_gpu_defaults_to_the_small_model(tmp_path: Path):
    """_common.sh defaults MODEL to Qwen3-4B; the smoke chain must still run on Qwen3-0.6B when MODEL is unset."""
    env = {k: v for k, v in os.environ.items() if k != "MODEL"}
    env.update({"DRY_RUN": "1", "SMOKE": str(tmp_path / "smoke"), "ONLY": "10"})
    r = subprocess.run(["bash", str(ROOT / "scripts/train/smoke_gpu.sh")], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    sft = (tmp_path / "smoke/logs/10-sft.log").read_text()
    assert "--model Qwen/Qwen3-0.6B" in sft and "Qwen3-4B" not in sft


def test_smoke_gpu_dry_run_wires_every_stage(tiny: Path, tmp_path: Path):
    env = {**os.environ, "DRY_RUN": "1", "MODEL": str(tiny), "SMOKE": str(tmp_path / "smoke"), "GPUS": "2"}
    r = subprocess.run(["bash", str(ROOT / "scripts/train/smoke_gpu.sh")], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rep = json.loads((tmp_path / "smoke/smoke_report.json").read_text())
    names = [s["stage"] for s in rep["stages"]]
    assert names == ["00-harvest", "10-sft", "10-eval", "20-teacher", "30-mc", "40-kl", "50-dpo", "55-pg", "60-grpo-trl",
                     "60-grpo-torch", "65-opd", "70-online", "80-distill", "90-serve-r4", "90-serve-r7"] and rep["all_ok"]
    sft = (tmp_path / "smoke/logs/10-sft.log").read_text()
    assert "accelerate launch" in sft and "--num_processes 2" in sft and "--max-steps 10" in sft
    assert "--lora-r 0" in (tmp_path / "smoke/logs/80-distill.log").read_text()
    opd = (tmp_path / "smoke/logs/65-opd.log").read_text()  # local (in-process Engine) teacher = the R4 adapter
    assert "jevharness.training.opd" in opd and '"type": "local"' in opd and "--executor sim" in opd


def test_make_data_v0_dry_run(tmp_path: Path):
    env = {**os.environ, "DRY_RUN": "1", "SKIP_EPISODES": "1", "OUT": str(tmp_path / "v0"),
           "RUNS_DIR": str(tmp_path / "runs")}
    r = subprocess.run(["bash", str(ROOT / "scripts/train/make_data_v0.sh")], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    cmds = [c for c in r.stdout.splitlines() if c.startswith("+ ")]
    assert any("jevharness.training.harvest" in c and "--min-jevgym-seed 1000" in c for c in cmds)
    assert any("jevharness.training.teacher_label" in c for c in cmds)
    assert any("_manifest_v0.py" in c for c in cmds)


def test_eval_decisions_reports_gate_metrics(tiny: Path, tmp_path: Path):
    from jevharness.training.data import write_jsonl
    from jevharness.training.eval_decisions import main

    ex = [{"id": f"t{i}/0/x", "dp": "tool_select" if i % 2 else "stop", "kind": "choice" if i % 2 else "noul",
           "state": "S", "question": "Q?", "options": ["a", "b", "c"] if i % 2 else ["yes", "no"],
           "target": {"a": 0.0, "b": 1.0, "c": 0.0} if i % 2 else {"yes": 1.0, "no": 0.0}} for i in range(6)]
    write_jsonl(ex, tmp_path / "dev.jsonl.gz")
    res = main(["--model", str(tiny), "--dev", str(tmp_path / "dev.jsonl.gz"), "--n-perm", "2", "--max-len", "128",
                "--out", str(tmp_path / "e.json")])
    r = res["files"][str(tmp_path / "dev.jsonl.gz")]
    assert r["overall"]["n"] == 6 and set(r["by_dp"]) == {"tool_select", "stop"}
    assert {"acc", "nll", "brier", "ece", "auroc", "aurc", "flip_rate"} <= set(r["overall"])
    assert 0 <= r["overall"]["flip_rate"] <= 1 and 0 <= r["overall"]["acc"] <= 1
    json.loads((tmp_path / "e.json").read_text())  # strict JSON (no NaN)


def test_verify_gpu_checks_pass_on_cpu_with_tiny_model(tiny: Path, tmp_path: Path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/train/_verify_gpu.py"), "--model", str(tiny),
                        "--allow-cpu", "--max-len", "256", "--out", str(tmp_path / "env.json")],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rep = json.loads((tmp_path / "env.json").read_text())
    assert {c["name"] for c in rep["checks"]} >= {"versions", "cuda", "trl", "codes", "template", "step"}


def test_verify_gpu_grpo_fields_match_grpo_trl():
    """_verify_gpu.py checks exactly the GRPOConfig fields grpo_trl.py passes (keeps the TRL check honest)."""
    sys.path.insert(0, str(ROOT / "scripts/train"))
    import _verify_gpu

    src = (ROOT / "src/jevharness/training/grpo_trl.py").read_text()
    call = src[src.index("args = GRPOConfig("):]
    call = call[: call.index("\n    trainer = ")]
    used = set(re.findall(r"(\w+)=", call)) - {"args"}
    assert used == set(_verify_gpu.GRPO_FIELDS)


def test_brief_references_existing_scripts_and_states_hard_rules():
    text = (ROOT / "docs/TRAINING_AGENT_BRIEF.md").read_text()
    for ref in set(re.findall(r"scripts/train/[\w./-]+\.(?:sh|py)", text)):
        assert (ROOT / ref).exists(), ref
    for ref in set(re.findall(r"configs/train/[\w./-]+", text)):
        assert (ROOT / ref.rstrip(".")).exists(), ref
    for must in ("BFCL", "When2Call", "seeds < 1000", "never commit", "huggingface.co", "Report template",
                 "OOM", "NaN", "TRL"):
        assert must in text, must
