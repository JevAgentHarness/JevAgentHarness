"""GPU round 5 (JevAgentRL) ablation table: what each training stage buys, on development data only.

    python scripts/train/r5_ablation.py --eval-dir runs/gpu_round5/eval --e2dev runs/gpu_round5/e2dev \
        --out results/gpu_round5/ablation

Rows: SFT v0 (round 3's R2), SFT v1, + OPD, + GRPO, the frozen JevAgentRL-4B, and the references Intern-Decision-4B and
Jev on the same items. Columns:
  static gate      acc on the 210-decision gate set (data/train_v0/dev.jsonl; same generator as v0 training data, so a
                   necessary bar, not evidence of in-harness competence -- TRAINING_RECIPES §18.4)
  G3 in-harness    v1 dev (held-out tasks of the harness's own states): overall / tool_select / tool_params / stop acc,
                   dev_recovery acc (states after an off-plan step), ECE (raw)
  E2 dev slice     development seeds 1100-1149, Flash System-2 (T = 0): shadow s1_acc per decision point (the System-2's states),
                   ts_params_replace success and s1_acc (the System-1's own states = live-policy accuracy), assist k = 3
                   and stop-guard success, System-1 latency per call
Development numbers: they select and gate the checkpoint, never headline numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

STAGES = [("base", "Qwen3-4B, untrained (A7.13 (i))"), ("sft_v0", "SFT v0 (R2, train_v0)"),
          ("sft_v1", "SFT v1 (R3-v1, train_v1)"), ("opd_i12", "+ OPD, iteration 12"), ("opd", "+ OPD, iteration 24"),
          ("grpo_i15", "+ GRPO, iteration 15"), ("grpo", "+ GRPO, iteration 30"),
          ("final", "JevAgentRL-4B (frozen, calibrated)")]


def _load(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text()) if p.exists() else None


def _file(ev: dict[str, Any] | None, suffix: str) -> dict[str, Any]:
    for f, m in ((ev or {}).get("files") or {}).items():
        if f.endswith(suffix) or f.endswith(suffix + ".gz"):
            return m
    return {}


def _acc(m: dict[str, Any], dp: str | None = None) -> float | None:
    x = (m.get("by_dp") or {}).get(dp) if dp else m.get("overall")
    return None if not x else x.get("acc")


def static_cols(eval_dir: Path, tag: str) -> dict[str, Any]:
    gate = _file(_load(eval_dir / f"eval_{tag}_gate.json"), "train_v0/dev.jsonl")
    v1 = _load(eval_dir / f"eval_{tag}_v1.json")
    dev, rec, ko = _file(v1, "train_v1/dev.jsonl"), _file(v1, "train_v1/dev_recovery.jsonl"), _file(v1, "dev_knockout.jsonl")
    go = _load(eval_dir / f"gonogo_{tag}.json") or {}
    return {"gate_acc": _acc(gate), "v1_acc": _acc(dev), "v1_tool_select": _acc(dev, "tool_select"),
            "v1_tool_params": _acc(dev, "tool_params"), "v1_stop": _acc(dev, "stop"), "v1_recovery": _acc(rec),
            "v1_knockout": _acc(ko), "v1_ece": (dev.get("overall") or {}).get("ece"), "go": go.get("go"),
            "gates": {k: v.get("pass") for k, v in (go.get("checks") or {}).items()}}


def intern_static(eval_dir: Path) -> dict[str, Any]:
    g = _load(eval_dir / "gate_intern4b_v1dev.json") or {}
    r = _load(eval_dir / "gate_intern4b_v1recovery.json") or {}
    bd = g.get("by_dp") or {}
    return {"gate_acc": 0.8619, "v1_acc": g.get("acc"), "v1_tool_select": (bd.get("tool_select") or {}).get("acc"),
            "v1_tool_params": (bd.get("tool_params") or {}).get("acc"), "v1_stop": (bd.get("stop") or {}).get("acc"),
            "v1_recovery": r.get("acc")}


def jev_static(teacher_summary: Path) -> dict[str, Any]:
    m = ((_load(teacher_summary) or {}).get("metrics") or {})
    return {"gate_acc": 0.9714, "v1_acc": (m.get("overall") or {}).get("acc"),
            "v1_tool_select": (m.get("dp:tool_select") or {}).get("acc"),
            "v1_tool_params": (m.get("dp:tool_params") or {}).get("acc"), "v1_stop": (m.get("dp:stop") or {}).get("acc"),
            "v1_recovery": (m.get("after_error:1") or {}).get("acc"),
            "v1_ece": (m.get("overall") or {}).get("ece")}


def e2_cols(rows: dict[str, Any], prefix: str) -> dict[str, Any]:
    def get(arm: str) -> dict[str, Any]:
        for k, v in rows.items():
            if k.split("::")[-1] == f"{prefix}_{arm}":
                return v
        return {}

    sh, rp, ts = get("shadow"), get("ts_params_replace"), get("ts_replace")
    asst, sg, tg = get("ts_assist_k3"), get("stop_guard"), get("ts_guard")
    lat = [r.get("s1_latency_ms_p50") for r in (sh, rp, ts) if r.get("s1_latency_ms_p50") is not None]
    return {"shadow_ts": sh.get("tool_select/shadow/s1_acc"), "shadow_tp": sh.get("tool_params/shadow/s1_acc"),
            "shadow_stop": sh.get("stop/shadow/s1_acc"),
            "replace_success": rp.get("success"), "replace_ts": rp.get("tool_select/replace/s1_acc"),
            "replace_tp": rp.get("tool_params/replace/s1_acc"), "ts_replace_success": ts.get("success"),
            "ts_replace_ts": ts.get("tool_select/replace/s1_acc"),
            "assist_success": asst.get("success"), "assist_cost": asst.get("cost_usd"),
            "stop_guard_success": sg.get("success"), "stop_guard_acc": sg.get("stop/guard/s1_acc"),
            "ts_guard_success": tg.get("success"), "ts_guard_acc": tg.get("tool_select/guard/s1_acc"),
            "s1_ms_p50": min(lat) if lat else None, "n": rp.get("n") or sh.get("n")}


def fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "GO" if v else "no-go"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval-dir", default="runs/gpu_round5/eval")
    ap.add_argument("--e2dev", default="runs/gpu_round5/e2dev")
    ap.add_argument("--teacher-summary", default="data/train_v1/dev_teacher.summary.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ev, e2 = Path(a.eval_dir), Path(a.e2dev)
    e2rows: dict[str, Any] = {}
    for f in sorted(e2.glob("e2dev_rl_*_summary.json")):
        e2rows.update(json.loads(f.read_text()))
    table: list[dict[str, Any]] = []
    for tag, label in STAGES:
        run_rows = {k: v for k, v in e2rows.items() if k.startswith(f"e2dev_rl_{tag}::")}
        if not (ev / f"eval_{tag}_v1.json").exists() and not run_rows:
            continue
        table.append({"row": label, "tag": tag, **static_cols(ev, tag), **e2_cols(run_rows, "ours")})
    table.append({"row": "Intern-Decision-4B", "tag": "intern", **intern_static(ev),
                  **e2_cols({k: v for k, v in e2rows.items() if k.startswith("e2dev_rl_refs_intern::")}, "intern")})
    table.append({"row": "Jev (teacher)", "tag": "jev", **jev_static(Path(a.teacher_summary)),
                  **e2_cols({k: v for k, v in e2rows.items() if k.startswith("e2dev_rl_refs_api::")}, "jev")})
    llm = next((v for k, v in e2rows.items() if k.endswith("::llm_only")), {})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"rows": table, "llm_only": llm}, indent=1) + "\n")
    h1 = ["stage", "gate acc", "v1 dev acc", "v1 tool_select", "v1 tool_params", "v1 stop", "v1 recovery", "v1 ECE",
          "§18.4"]
    h2 = ["stage", "n", "shadow ts", "shadow tp", "shadow stop", "ts-replace succ.", "ts-replace own-state ts acc",
          "ts+params-replace succ.", "its ts acc", "its tp acc", "assist-k3 succ.", "stop-guard succ.", "ts-guard succ.",
          "S1 ms/call p50"]
    md = ["### Decision level (static gate; G3 in-harness states = v1 dev)", "", "| " + " | ".join(h1) + " |",
          "|" + "---|" * len(h1)]
    for r in table:
        md.append("| " + " | ".join([r["row"], fmt(r.get("gate_acc")), fmt(r.get("v1_acc")), fmt(r.get("v1_tool_select")),
                                     fmt(r.get("v1_tool_params")), fmt(r.get("v1_stop")), fmt(r.get("v1_recovery")),
                                     fmt(r.get("v1_ece")), fmt(r.get("go"))]) + " |")
    md += ["", f"### E2 development slice (development seeds 1100-1149, Flash System-2 T = 0; llm_only success "
               f"{fmt(llm.get('success'))}, $ {fmt(llm.get('cost_usd'), 4)} / episode)", "",
           "| " + " | ".join(h2) + " |", "|" + "---|" * len(h2)]
    for r in table:
        md.append("| " + " | ".join([r["row"], fmt(r.get("n")), fmt(r.get("shadow_ts")), fmt(r.get("shadow_tp")),
                                     fmt(r.get("shadow_stop")), fmt(r.get("ts_replace_success")),
                                     fmt(r.get("ts_replace_ts")), fmt(r.get("replace_success")), fmt(r.get("replace_ts")),
                                     fmt(r.get("replace_tp")), fmt(r.get("assist_success")),
                                     fmt(r.get("stop_guard_success")), fmt(r.get("ts_guard_success")),
                                     fmt(r.get("s1_ms_p50"), 0)]) + " |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
