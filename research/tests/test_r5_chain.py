"""scripts/train/r5_chain.py: the mechanical keep / retain / freeze decision of the JevAgentRL chain (A7.10 (a))."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r5_chain", ROOT / "scripts/train/r5_chain.py")
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)  # type: ignore[union-attr]


def _eval(d: Path, tag: str, acc: float, rec: float, ece: float, go: bool, source: str = "v1 dev") -> None:
    files = {"data/train_v1/dev.jsonl.gz": {"overall": {"acc": acc, "ece": ece}},
             "data/train_v1/dev_recovery.jsonl.gz": {"overall": {"acc": rec}}}
    (d / f"eval_{tag}_v1.json").write_text(json.dumps({"files": files}))
    (d / f"gonogo_{tag}.json").write_text(json.dumps({"go": go, "checks": {"G1": {"pass": go}},
                                                      "references": {"intern_v1_dev": {"source": source}}}))


def _slice(d: Path, run: str, prefix: str, succ: float, own: float, reward: float) -> None:
    rows = {}
    for a in chain.LIVE + ("shadow",):
        r = {"success": succ, "reward_ccps": reward, "n": 50}
        if a in ("ts_replace", "ts_params_replace"):
            r.update({"tool_select/replace/s1_acc": own, "tool_select/replace/n": 100})
        if a == "shadow":
            r["tool_select/shadow/s1_acc"] = own
        rows[f"{run}::{prefix}_{a}"] = r
    (d / f"{run}_summary.json").write_text(json.dumps(rows))


def _dirs(tmp_path: Path) -> tuple[Path, Path]:
    ev, e2 = tmp_path / "eval", tmp_path / "e2"
    ev.mkdir()
    e2.mkdir()
    return ev, e2


def test_rule_b_retention_validity_and_freeze(tmp_path: Path):
    ev, e2 = _dirs(tmp_path)
    _eval(ev, "sft_v1", 0.95, 0.93, 0.03, True)
    _eval(ev, "opd_i12", 0.949, 0.93, 0.03, True)         # flat at decision level: rule B can hold
    _eval(ev, "opd", 0.948, 0.925, 0.035, True)
    _eval(ev, "grpo_i15", 0.95, 0.93, 0.03, True)
    _eval(ev, "grpo", 0.95, 0.93, 0.03, True)
    _slice(e2, "e2dev_rl_sft_v1", "ours", 0.70, 0.70, 0.40)
    _slice(e2, "e2dev_rl_opd_i12", "ours", 0.74, 0.78, 0.44)
    _slice(e2, "e2dev_rl_opd", "ours", 0.75, 0.80, 0.45)
    _slice(e2, "e2dev_rl_grpo_i15", "ours", 0.78, 0.81, 0.47)
    _slice(e2, "e2dev_rl_grpo", "ours", 0.80, 0.82, 0.50)
    _slice(e2, "e2dev_rl_refs_intern", "intern", 0.30, 0.45, 0.20)
    res = chain.decide(ev, e2, validity_ok=False)
    opd = res["stages"]["opd"]
    assert opd["kept"] and opd["retained"] == "opd_i12"   # both pass rule B; higher v1-dev acc wins
    assert not res["stages"]["grpo"]["kept"] and res["frozen"] == "opd_i12" and res["go"] and not res["rl_claim"]
    res2 = chain.decide(ev, e2, validity_ok=True)
    assert res2["grpo_init"] == "opd_i12" and res2["stages"]["grpo"]["retained"] == "grpo"  # tie on acc -> later step
    assert res2["kept"] == ["sft_v1", "opd_i12", "grpo"] and res2["frozen"] == "grpo" and res2["rl_claim"]
    assert res2["confirmation_dev_slice"]["ts_replace_success>=intern"]


def test_regression_drops_opd_and_proxy_g3_is_not_go(tmp_path: Path):
    ev, e2 = _dirs(tmp_path)
    _eval(ev, "sft_v1", 0.90, 0.85, 0.05, True, source="proxy: round-2 shadow s1_acc (n = 10 tasks)")
    _eval(ev, "opd", 0.85, 0.80, 0.05, True)             # decision-level regression: fails A and B
    _slice(e2, "e2dev_rl_sft_v1", "ours", 0.6, 0.6, 0.3)
    _slice(e2, "e2dev_rl_opd", "ours", 0.9, 0.9, 0.6)
    res = chain.decide(ev, e2, validity_ok=True)
    assert res["kept"] == ["sft_v1"] and res["frozen"] == "sft_v1" and not res["go"]   # proxy G3 never counts
    assert not res["stages"]["opd"]["kept"] and not res["stages"]["grpo"]["kept"]


def test_validity_detection_handles_quiet_pytest_output():
    host = 'No entry for terminal type "unknown";\nusing dumb terminal settings.\n' + "." * 7 + " " * 66 + "[100%]\n"
    assert chain.validity_passed(host)                                   # -qq: no summary line (round-5 host output)
    assert chain.validity_passed("7 passed in 3.2s\n")
    assert not chain.validity_passed("..F....   [100%]\n")
    assert not chain.validity_passed("6 passed, 1 failed in 3s\n")
    assert not chain.validity_passed("")
    assert chain.validity_passed(host + "pytest_exit=0\n") and not chain.validity_passed(host + "pytest_exit=1\n")
