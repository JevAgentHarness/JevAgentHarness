#!/usr/bin/env python3
"""JevAgentRL development and freeze records -> \\res{} macros for the paper (paper/generated/rl.tex).

Reads only the committed records of the frozen model (no checkpoint, no API, no evaluation output):
  results/gpu_round5/ablation.json        development ablation per checkpoint (v1 dev, gate set, E2 dev slice)
  results/gpu_round5/chain_decision.json  the mechanical chain: kept / dropped stages and the live dev-slice readouts
  results/gpu_round5/latency_*.txt        System-1 latency (synthetic 1.5k-token state, 3 questions per request)
  results/gpu_round5/FREEZE_RECORD.json   frozen checkpoint and base model
These are development numbers (training and development seeds only); the evaluation numbers of the frozen model are
Stage-2 keys (stage2/S2-3, stage2/S2-4, stage2_rl/...).

Keys:
  rl/dev/<tag>/{v1_acc, v1_tool_select, v1_tool_params, v1_stop, v1_recovery, v1_ece, gate_acc, ts_replace_success,
      ts_replace_ts, s1_ms_p50, go}           tags: base, sft_v0, sft_v1, opd_i12, opd, grpo_i15, grpo, final, intern, jev
  rl/chain/{kept, frozen, base_model}
  rl/chain/<stage>/{reward_live, success_live}  stages: sft_v1, opd_i12, opd, grpo_i15, grpo
  rl/latency/<backend>/{p50_ms_c1, p95_ms_c1, p50_ms_c4, req_per_s_c1}   backends: jevagentrl, jev, intern
  rl/gpu
  rl2/...  JevAgentRL-v2 (A7.19, exploratory; results/gpu_round6_miles/): rl2/{base, kept, frozen, rl_claim, go},
      rl2/zeroshot/{startlux,qwen}/v1_acc, rl2/dev/<tag>/{gate_acc, v1_acc, v1_tool_select, v1_recovery, v1_ece}
      (tags zeroshot_startlux, sft_v2, opd_i12, opd, grpo_i15, grpo), rl2/stress/<tag>/{S, A, P, R} (the same trained
      tags plus the references intern and jev), rl2/stress/n
Accuracies and success in percent (one decimal), ECE and rewards as fractions (three / four decimals).
Usage: python scripts/rl_to_tex.py [--out FILE]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / "results" / "gpu_round5"
STAGE_NAMES = {"sft_v1": "SFT v1", "opd_i12": "OPD (iteration 12)", "opd": "OPD (iteration 24)",
               "grpo_i15": "GRPO (iteration 15)", "grpo": "GRPO (iteration 30)"}


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _pct(x) -> str | None:
    return None if x is None else f"{100 * x:.1f}"


def dev_rows(values: dict[str, str]) -> None:
    d = json.loads((R5 / "ablation.json").read_text())
    for r in d["rows"]:
        k = f"rl/dev/{r['tag']}"
        for f in ("v1_acc", "v1_tool_select", "v1_tool_params", "v1_stop", "v1_recovery", "gate_acc",
                  "ts_replace_success", "ts_replace_ts"):
            v = _pct(r.get(f))
            if v is not None:
                values[f"{k}/{f}"] = v
        if r.get("v1_ece") is not None:
            values[f"{k}/v1_ece"] = f"{r['v1_ece']:.3f}"
        if r.get("s1_ms_p50") is not None:
            values[f"{k}/s1_ms_p50"] = f"{r['s1_ms_p50']:.0f}"
        if r.get("go") is not None:
            values[f"{k}/go"] = "GO" if r["go"] else "NO-GO"
    lo = d.get("llm_only") or {}
    if lo.get("success") is not None:
        values["rl/dev/llm_only/success"] = _pct(lo["success"])
        values["rl/dev/llm_only/reward_ccps"] = f"{lo['reward_ccps']:.4f}"


def chain(values: dict[str, str]) -> None:
    c = json.loads((R5 / "chain_decision.json").read_text())
    values["rl/chain/kept"] = " + ".join(STAGE_NAMES.get(s, s) for s in c["kept"])
    values["rl/chain/frozen"] = STAGE_NAMES.get(c["frozen"], c["frozen"])
    for s, row in c.get("dev_slice", {}).items():
        values[f"rl/chain/{s}/reward_live"] = f"{row['reward_live']:.4f}"
        values[f"rl/chain/{s}/success_live"] = _pct(row["success_live"])
    fr = json.loads((R5 / "FREEZE_RECORD.json").read_text())
    values["rl/chain/base_model"] = fr["frozen_checkpoint"]["base_model"]


def latency(values: dict[str, str]) -> None:
    for name, f in (("jevagentrl", "latency_jevagentrl4b.txt"), ("jev", "latency_jev.txt"),
                    ("intern", "latency_intern4b.txt")):
        t = (R5 / f).read_text()
        d = json.loads(t[t.index("{"):])
        res = d["results"]
        values[f"rl/latency/{name}/p50_ms_c1"] = f"{res['c=1']['p50_ms']:.0f}"
        values[f"rl/latency/{name}/p95_ms_c1"] = f"{res['c=1']['p95_ms']:.0f}"
        values[f"rl/latency/{name}/p50_ms_c4"] = f"{res['c=4']['p50_ms']:,.0f}".replace(",", "{,}")
        values[f"rl/latency/{name}/req_per_s_c1"] = f"{res['c=1']['req_per_s']:.1f}"
    smi = (R5 / "nvidia_smi_last.txt")
    if smi.exists():
        m = re.search(r"NVIDIA ([A-Z][A-Z0-9]+)", smi.read_text())
        if m:
            values["rl/gpu"] = m.group(1)


R6 = ROOT / "results" / "gpu_round6_miles"
STAGE2_NAMES = {"sft_v2": "SFT v2", "opd_i12": "OPD (iteration 12)", "opd": "OPD (iteration 24)",
                "grpo_i15": "GRPO (iteration 15)", "grpo": "GRPO (iteration 30)"}


def v2(values: dict[str, str]) -> None:
    """JevAgentRL-v2 on Miles (A7.19, exploratory): development records only."""
    if not (R6 / "chain_decision.json").exists():
        return
    c = json.loads((R6 / "chain_decision.json").read_text())
    values["rl2/kept"] = " + ".join(STAGE2_NAMES.get(t, t) for t in c["kept"])
    values["rl2/frozen"] = STAGE2_NAMES.get(c["frozen"], c["frozen"])
    values["rl2/rl_claim"] = "yes" if c.get("rl_claim") else "no"
    values["rl2/go"] = "GO" if c.get("go") else "NO-GO"
    b = json.loads((R6 / "base.json").read_text())
    values["rl2/base"] = b["name"]
    for k, row in b["checks"].items():
        if row.get("zeroshot_v1_dev_acc") is not None:
            values[f"rl2/zeroshot/{k}/v1_acc"] = _pct(row["zeroshot_v1_dev_acc"])
    a = json.loads((R6 / "ablation.json").read_text())["v2"]
    for tag, row in a.items():
        d = row.get("decision") or {}
        k = f"rl2/dev/{tag}"
        for f, src in (("gate_acc", "gate_acc"), ("v1_acc", "acc"), ("v1_recovery", "recovery")):
            if d.get(src) is not None:
                values[f"{k}/{f}"] = _pct(d[src])
        if (d.get("by_dp") or {}).get("tool_select") is not None:
            values[f"{k}/v1_tool_select"] = _pct(d["by_dp"]["tool_select"])
        if d.get("ece") is not None:
            values[f"{k}/v1_ece"] = f"{d['ece']:.3f}"
        st = row.get("stress") or {}
        if st.get("S") is not None:
            values[f"rl2/stress/{tag}/S"] = _pct(st["S"])
            values[f"rl2/stress/{tag}/A"] = _pct(st["A"])
            values[f"rl2/stress/{tag}/P"] = _pct(st["P"])
            values[f"rl2/stress/{tag}/R"] = f"{st['R']:.3f}"
            values["rl2/stress/n"] = str(st["n"])
    si = c.get("stress_intern") or {}
    if si.get("S") is not None:
        for f in ("S", "A", "P"):
            values[f"rl2/stress/intern/{f}"] = _pct(si[f])
        values["rl2/stress/intern/R"] = f"{si['R']:.3f}"
    jf = R6 / "stress" / "stress_refs_jev_summary.json"
    if jf.exists():
        j = next(iter(json.loads(jf.read_text()).values()))
        values["rl2/stress/jev/S"] = _pct(j["success"])
        values["rl2/stress/jev/A"] = _pct(j["tool_select/replace/s1_acc"])
        values["rl2/stress/jev/P"] = _pct(j["tool_params/replace/s1_acc"])
        values["rl2/stress/jev/R"] = f"{j['reward_ccps']:.3f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "paper" / "generated" / "rl.tex"))
    a = ap.parse_args(argv)
    values: dict[str, str] = {}
    dev_rows(values)
    chain(values)
    latency(values)
    v2(values)
    lines = ["% AUTO-GENERATED by scripts/rl_to_tex.py -- do not edit by hand.",
             f"% generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} at git {_git_head()}",
             "% JevAgentRL development and freeze records (results/gpu_round5/); development numbers, never evaluation",
             "\\providecommand{\\res}[1]{\\ifcsname jevres@#1\\endcsname\\csname jevres@#1\\endcsname\\else\\tbd\\fi}",
             "\\providecommand{\\defres}[2]{\\expandafter\\def\\csname jevres@#1\\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{v}}}" for k, v in sorted(values.items())]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(values)} keys to {out}")


if __name__ == "__main__":
    main()
