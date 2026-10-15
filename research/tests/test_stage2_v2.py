"""The exploratory JevAgentRL-v2 rows of docs/PREREG_A7_STAGE2.md A7.19 (f), as prepared before any v2 episode: the three
configs mirror the StartLux-4B rows of A7.18 (e) 2-5 except for System-1; the serving plan matches the freeze record;
the tariff plan uses development seeds only; the served name is an exact-0 price row; the profile's backends.jevagentrl_v2
block is added pending by select_profile.py's rule and only v2's own fit fills it; the box stages of
scripts/aws/stage2_gpu.sh exist in the registered order and their checkpoint check and tariff measurement work on stubs
(no AWS, no GPU, no model, no network)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from jevharness.bench import registry as R  # noqa: E402
from jevharness.bench.runner import arm_bindings, arm_profiles, arm_task_counts, expand_arms  # noqa: E402

EXP = ROOT / "configs/experiments"
PLAN = json.loads((ROOT / "results/stage2/v2_eval_plan.json").read_text())
V2_ROWS_DONE = "7a44310"   # the commit that completed every A7.19 (f) row (2026-10-05 04:02 UTC)
TARIFF_PLAN = json.loads((ROOT / "results/stage2/tariff_v2_plan.json").read_text())
FROZEN = ROOT / "results/gpu_round6_miles/final_FROZEN.txt"
BOX = ROOT / "scripts/aws/stage2_gpu.sh"


def _y(name: str) -> dict:
    return yaml.safe_load((EXP / name).read_text())


def test_v2_configs_mirror_the_startlux_rows_except_system1():
    s2, v2, st, cal = _y("stage2_jevgym.yaml"), _y("stage2_v2.yaml"), _y("stage2_v2_states.yaml"), _y("jevgym_shadow_cal_v2.yaml")
    sl_st, sl_cal = _y("stage2_startlux_states.yaml"), _y("jevgym_shadow_cal_startlux.yaml")
    # the profile row: every setting that is not System-1 is stage2_jevgym's (suite -> the same frozen task list)
    for k in ("suite", "n_tasks", "llm", "small_llm", "schedule"):
        assert v2[k] == s2[k], k
    assert v2["workers"] == 4 and [a.name for a in expand_arms(v2)] == ["jevagent_best_v2"]
    pr = arm_profiles(v2)["jevagent_best_v2"]
    assert pr["backend"] == "jevagentrl_v2" and pr["s2"] == "gemini-3.5-flash" and pr["status"] == "selected"
    assert pr["path"] == "configs/harness/jevagent_best.yaml" and arm_bindings(v2)["jevagent_best_v2"]["s1"] == "jevagentrl_v2"
    # the live replace arm and the calibration arm: the StartLux configs with v2 as System-1
    assert st["suite"] == sl_st["suite"] == s2["suite"] and st["n_tasks"] == 200 and st["llm"] == sl_st["llm"]
    assert [a.name for a in expand_arms(st)] == ["v2_ts_replace_f"] and arm_task_counts(st) == {"v2_ts_replace_f": 200}
    assert st["arms"][0]["tool_select"] == sl_st["arms"][0]["tool_select"] == {"mode": "replace"}
    assert cal["suite"] == sl_cal["suite"] and cal["suite"]["seed0"] == 5000 and cal["n_tasks"] == 150
    assert {k: v for k, v in cal["arms"][0].items() if k != "name"} == {k: v for k, v in sl_cal["arms"][0].items()
                                                                       if k != "name"}
    assert cal["arms"][0]["name"] == "v2_shadow_cal" and cal["llm"] == sl_cal["llm"] and cal["workers"] == 4
    # one v2 backend everywhere: the plan's port and served name, a placeholder until the host binds it
    ck = PLAN["checkpoint"]
    specs = [v2["s1_backends"]["jevagentrl_v2"], st["s1_backends"]["jevagentrl_v2"], cal["s1"]]
    for sp in specs:
        assert sp["base_url"] == f"http://127.0.0.1:{ck['port']}" and sp["name"] == sp["model"] == ck["served_name"]
        assert sp["price_model"] == ck["served_name"] and sp["timeout"] == 60.0 and sp["type"] == "jev"
    assert specs[0]["placeholder"] and specs[1]["placeholder"] and "placeholder" not in specs[2]
    assert {v2["s1_cache"], st["s1_cache"], cal["s1_cache"], specs[0]["cache"]} == {"cache/s1_stage2_v2.sqlite"}
    # seeds: calibration and evaluation blocks as registered (the runner refuses 300000.. without JEV_EVAL_RUNS=1)
    assert R.overlaps(int(v2["suite"]["seed0"]), int(v2["n_tasks"]), R.STAGE2_EVAL_RANGES)
    lo, hi = R.CALIBRATION_RANGE
    assert lo <= cal["suite"]["seed0"] and cal["suite"]["seed0"] + cal["n_tasks"] - 1 <= hi


def test_v2_plan_matches_the_freeze_record():
    lines = dict(ln.split(" ", 1) for ln in FROZEN.read_text().splitlines() if ln and " " in ln)
    assert lines["frozen_at"].strip() == PLAN["frozen_utc"] and PLAN["freeze_record"] == str(FROZEN.relative_to(ROOT))
    import stage2_analysis as S2

    rec = S2.read_v2_freeze(FROZEN)
    assert S2.v2_freeze_status(FROZEN)[0] is None                                   # before the A7.19 (e) deadline
    cal = ROOT / "results/gpu_round6_miles/final_calibration.json"
    assert rec["files"][PLAN["checkpoint"]["calibration"]] == hashlib.sha256(cal.read_bytes()).hexdigest()
    assert {"model.safetensors", "config.json", "regression_image.json", "regression_venv.json"} <= set(rec["files"])
    assert PLAN["serving_commit_of_the_freeze"] in rec["serving"]                  # "serving / training commit ..."
    assert PLAN["checkpoint"]["s3"].rstrip("/").endswith(PLAN["checkpoint"]["dir"])
    dev = ROOT / PLAN["regression"]["dev"]
    assert hashlib.sha256(dev.read_bytes()).hexdigest() == PLAN["regression"]["dev_sha256"]
    assert PLAN["regression"]["tol_max_abs_dprob"] == json.loads(
        (ROOT / "results/gpu_round6_miles/final_regression_compare.json").read_text())["tol"]
    env = json.loads((ROOT / "results/stage2/startlux_manifest.json").read_text())["environment"][0]   # A7.18 (d)
    assert PLAN["serving_env"]["kernels"] == [f"flash-linear-attention=={env['flash-linear-attention']}",
                                              f"causal-conv1d=={env['causal-conv1d']}"]
    assert PLAN["serving_env"]["torch"] == f"torch=={env['torch']}"
    api = PLAN["api"]                                                                 # A7.19 (g): <= $100 in all
    assert sum(api["reserve_usd"].values()) <= api["cap_rows_usd"] <= 100.0
    assert set(api["reserve_usd"]) == {"cal", "tariff", "profile", "states"}
    # The serving code is the freeze's over the span that produced every A7.19 (f) row (freeze commit .. V2_ROWS_DONE);
    # later serving changes for other work are recorded in PREREG_A7_STAGE2 Deviations 2026-10-06 ~06:25 UTC.
    have = [subprocess.run(["git", "cat-file", "-e", c], cwd=ROOT, capture_output=True).returncode == 0
            for c in (PLAN["serving_commit_of_the_freeze"], V2_ROWS_DONE)]
    if all(have):
        d = subprocess.run(["git", "diff", "--quiet", PLAN["serving_commit_of_the_freeze"], V2_ROWS_DONE, "--",
                            *PLAN["serving_code"]], cwd=ROOT)
        assert d.returncode == 0, "the serving code changed between the v2 freeze and its last row (record a Deviations entry)"


def test_v2_tariff_plan_dev_seeds_and_prices():
    rl = json.loads((ROOT / "results/stage2/tariff_rl_plan.json").read_text())
    for part in ("warmup", "workload"):
        p = TARIFF_PLAN[part]
        assert (p["seed0"], p["n"]) == (rl[part]["seed0"], rl[part]["n"])           # the S2-4 / S2-4S workload
        assert p["seed0"] + p["n"] - 1 < R.DEV_MAX_SEED
        for rng in (R.EVAL_SEED_RANGES, R.STAGE2_EVAL_RANGES, R.STAGE2_RESERVED_RANGES, (R.CALIBRATION_RANGE,)):
            assert not R.overlaps(p["seed0"], p["n"], rng)
    c = TARIFF_PLAN["configuration"]
    assert c["arm"] == "jevagent_best_v2" and c["config"] == "configs/experiments/stage2_v2.yaml" and c["workers"] == 4
    h = TARIFF_PLAN["host"]
    assert h["reference_type"] in h["usd_per_hour_by_type"]
    assert h["usd_per_hour_by_type"]["g6e.2xlarge"] == rl["host"]["usd_per_hour"]  # the S2-4 tariff's host price
    from jevharness.accounting import PriceTable

    p = PriceTable.load().prices["jevagentrl-v2"]                                     # A7.13 (iii): nothing twice
    assert p.input_per_m == 0 and p.output_per_m == 0 and p.per_call == 0


def test_add_pending_block_then_only_the_v2_fit_fills_it(tmp_path: Path):
    import fill_backend_tau as F

    from jevharness.harness import profile as HP

    prof = {"kind": "jevagent_profile", "version": 1, "status": "selected", "system2": {"gemini-3.5-flash": {
        "decision_points": {"tool_select": {"mode": "cascade", "tau": 1.0}, "tool_params": {"mode": "cascade", "tau": 0.99},
                            "reflection": {"mode": "trigger", "tau": 0.5}},
        "backends": {"jevagent_rl": {"tool_select": {"tau": 0.5}}}}}}
    out = F.add_pending(prof, "jevagentrl_v2")
    blk = out["system2"]["gemini-3.5-flash"]["backends"]
    assert blk["jevagentrl_v2"] == {"tool_select": {"calibration": "pending"}, "tool_params": {"calibration": "pending"}}
    assert blk["jevagent_rl"] == {"tool_select": {"tau": 0.5}}                       # nothing else changes
    with pytest.raises(SystemExit, match="exists already"):
        F.add_pending(out, "jevagentrl_v2")
    no_cascade = json.loads(json.dumps(prof))
    no_cascade["system2"]["gemini-3.5-flash"]["decision_points"] = {"reflection": {"mode": "trigger", "tau": 0.5}}
    with pytest.raises(SystemExit, match="no cascade"):
        F.add_pending(no_cascade, "jevagentrl_v2")
    p = tmp_path / "p.yaml"
    p.write_text("# header\n" + yaml.safe_dump(prof, sort_keys=False))
    assert F.main(["--profile", str(p), "--backend", "jevagentrl_v2", "--add-pending"]) == 0
    loaded = HP.load_profile(p)
    assert p.read_text().splitlines()[0] == "# header" and "--add-pending" in p.read_text().splitlines()[1]
    dps = HP.decision_points(loaded, "gemini-3.5-flash", backend="jevagentrl_v2")[0]
    assert dps["tool_select"]["calibration"] == "pending" and dps["reflection"] == {"mode": "trigger", "tau": 0.5}
    filled = F.fill(loaded, "jevagentrl_v2", {"tool_select": {"primary": {"tau_hat": 0.93}},
                                              "tool_params": {"primary": {"tau_hat": 0.995}}})
    assert filled["system2"]["gemini-3.5-flash"]["backends"]["jevagentrl_v2"]["tool_select"] == {"tau": 0.93}
    # the committed profile: the block exists (pending, or filled from v2's own calibration arm); other blocks untouched
    best = HP.load_profile(ROOT / "configs/harness/jevagent_best.yaml")
    sec = HP.section(best, "gemini-3.5-flash")
    v2b = sec["backends"]["jevagentrl_v2"]
    assert all(v.get("calibration") == "pending" for k, v in v2b.items() if k in ("tool_select", "tool_params")) or (
        (v2b.get("fitted_from") or {}).get("fit_arm") == "v2_shadow_cal")
    assert (sec["backends"]["startlux_4b"].get("fitted_from") or {}).get("fit_arm") == "sl4b_shadow_cal"
    assert (sec["backends"]["jevagent_rl"].get("fitted_from") or {}).get("fit_arm") == "rl_shadow_cal"


def _block() -> str:
    lines, on = [], False
    for ln in BOX.read_text().splitlines(keepends=True):
        if ln.startswith("# ---- A7.19 (f): the JevAgentRL-v2 rows"):
            on = True
        elif ln.startswith("# a failed stage stops the chain while later stages depend on it"):
            on = False
        if on:
            lines.append(ln)
    return "".join(lines)


def test_box_script_has_the_v2_stages_in_the_registered_order():
    assert subprocess.run(["bash", "-n", str(BOX)]).returncode == 0
    text, block = BOX.read_text(), _block()
    order = PLAN["order"]
    for s in order:
        assert f"\nstage_{s}() " in block, s
    usage = next(ln for ln in text.splitlines() if ln.startswith("#     rearm v2_setup"))
    assert usage.split()[2:-1] == order and usage.split()[-1] == "final"           # `final`: the chain-end server stop
    assert f"V2_PORT={PLAN['checkpoint']['port']}\n" in block and f"$CK/{PLAN['checkpoint']['dir']} " in block
    for cfg in ("jevgym_shadow_cal_v2.yaml", "stage2_v2.yaml", "stage2_v2_states.yaml"):
        assert f"configs/experiments/{cfg}" in block
    assert "EVAL=1 three v2_profile" in block and "EVAL=1 three v2_states" in block   # JEV_EVAL_RUNS=1: evaluation only
    assert "EVAL=1 three v2_cal" not in block and "gate stage2_v2 ||" in block


def _sh(script: str, cwd: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


@pytest.fixture()
def box(tmp_path: Path):
    """A throw-away tree with the v2 block sourced: stub `aws` (s3 sync copies a fixture checkpoint) and a stub runner
    that writes four records (one errored first attempt) and takes 2 s; the harness python is this interpreter."""
    tree, hv, bindir, ck = tmp_path / "tree", tmp_path / "hv/bin", tmp_path / "bin", tmp_path / "src_ck"
    for d in (tree / "results/stage2", tree / "runs", hv, bindir, ck):
        d.mkdir(parents=True)
    for d in ("scripts", "src", "configs"):
        (tree / d).symlink_to(ROOT / d)
    for f in ("v2_eval_plan.json", "tariff_v2_plan.json"):
        (tree / "results/stage2" / f).write_text((ROOT / "results/stage2" / f).read_text())
    # a wrapper, not a symlink: a venv interpreter reached through a symlink elsewhere loses its venv (no pyvenv.cfg
    # beside the link), which fails on a host whose base interpreter lacks the harness dependencies
    (hv / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (hv / "python").chmod(0o755)
    (tmp_path / "block.sh").write_text(_block())
    (bindir / "aws").write_text(f'#!/bin/bash\n[[ "$1 $2" == "s3 sync" ]] && cp -r {ck}/. "$4" && exit 0\nexit 1\n')
    (hv / "jevbench").write_text(
        "#!/bin/bash\nname=''; for a in \"$@\"; do case \"$a\" in name=*) name=${a#name=};; esac; done\n"
        "mkdir -p runs/$name; sleep 2\n" + f"{sys.executable} - runs/$name/jevagent_best_v2.jsonl <<'PY'\n"
        "import json, sys, datetime as dt\nt = dt.datetime(2026, 10, 5, 10, 0, 0)\nwith open(sys.argv[1], 'w') as f:\n"
        "    for i, err in ((0, 'Timeout'), (0, None), (1, None), (2, None)):\n"
        "        c = [{'component': 's1', 'ts': (t + dt.timedelta(seconds=10 * i + 2)).isoformat(), 'latency_ms': 500,"
        " 'input_tokens': 1000}]\n"
        "        f.write(json.dumps({'task_id': f't{i}', 'error': err, 'calls': c, 'usage': {'total': {'cost_usd': 0.03}}})"
        " + '\\n')\nPY\n")
    for f in (bindir / "aws", hv / "jevbench"):
        f.chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"}
    head = (f"set -u; HV={tmp_path / 'hv'} OUT={tmp_path / 'out'} W={tmp_path / 'w'} CK={tmp_path / 'ck'} "
            f"REGION=us-east-1 S23=results/stage2/s23 S3OUT=s3://x; source {tmp_path / 'block.sh'}\n"
            "keys() { return 0; }; v2_up() { return 0; }; ops() { :; }; sync_run() { :; }\n")
    return tmp_path, tree, ck, env, head


def test_box_v2_fetch_checks_every_sha256_of_the_freeze_record(box):
    tmp, tree, ck, env, head = box
    (ck / "model.safetensors").write_bytes(b"weights")
    for f in ("config.json", "calibration.json", "tokenizer.json", "regression_image.json", "unlisted.json"):
        (ck / f).write_text(f)
    listed = ("model.safetensors", "config.json", "calibration.json", "tokenizer.json", "regression_image.json")
    rec = "name test\nfrozen_at 2026-10-04T00:53:20Z\n" + "".join(
        f"{hashlib.sha256((ck / f).read_bytes()).hexdigest()}  {'./' if f.endswith('.safetensors') else ''}{f}\n"
        for f in listed)
    (tmp / "FROZEN.txt").write_text(rec)
    (ck / "FROZEN.txt").write_text(rec)
    plan = {**PLAN, "freeze_record": str(tmp / "FROZEN.txt")}
    (tmp / "plan.json").write_text(json.dumps(plan))
    r = _sh(head + f"V2PLAN={tmp / 'plan.json'}; stage_v2_fetch", tree, env)
    assert r.returncode == 0, r.stdout + r.stderr
    v = json.loads((tmp / "out/v2/checkpoint_verify.json").read_text())
    assert v["ok"] and v["frozen_txt_identical"] and list(v["unlisted_sha256"]) == ["unlisted.json"]
    (ck / "calibration.json").write_text("another temperature file")
    r = _sh(head + f"V2PLAN={tmp / 'plan.json'}; stage_v2_fetch", tree, env)
    assert r.returncode == 1 and "MISMATCH ['calibration.json']" in r.stdout


def test_box_v2_tariff_prices_the_host_type_and_refuses_an_unlisted_one(box):
    tmp, tree, ck, env, head = box
    fit = "v2_fitted() { echo fitted; }; v2_skip() { echo; }; "
    r = _sh(head + fit + "v2_itype() { echo g6e.16xlarge; }; stage_v2_tariff", tree, env)
    assert r.returncode == 0, r.stdout + r.stderr
    t = json.loads((tree / "results/stage2/tariff_v2.json").read_text())
    usd = TARIFF_PLAN["host"]["usd_per_hour_by_type"]
    assert t["served_model"] == "jevagentrl-v2" and t["host_usd_per_hour"] == usd["g6e.16xlarge"] and t["episodes"] == 3
    assert set(t["sensitivity"]) == {"busy_throughput", "jev_api_price", "zero", "reference_host"}
    assert abs(t["sensitivity"]["reference_host"] / t["q_usd_per_episode"] - usd["g6e.2xlarge"] / usd["g6e.16xlarge"]) < 1e-9
    w = json.loads((tmp / "out/v2/tariff_window.json").read_text())
    assert w["instance_type"] == "g6e.16xlarge" and w["exit"] == 0
    assert "tariff: spent" in (tmp / "out/v2/spend.txt").read_text()                  # the A7.19 (g) ledger check ran
    r = _sh(head + fit + "v2_itype() { echo g6e.xlarge; }; stage_v2_tariff", tree, env)
    assert r.returncode == 1 and "not priced" in r.stdout
    r = _sh(head + "v2_fitted() { echo fitted; }; v2_skip() { echo 'not tested (delegates nothing)'; }; "
            "stage_v2_tariff && stage_v2_gate_tariff", tree, env)
    assert r.returncode == 0 and "no v2 tariff: not tested (delegates nothing)" in r.stdout
