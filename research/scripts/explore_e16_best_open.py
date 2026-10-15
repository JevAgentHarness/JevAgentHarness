#!/usr/bin/env python3
r"""E16 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E16: the best open-weights decision
model on all three levels"): does more public tool-use data (arm A, AgentDecision-4B continued) and / or a larger open
base (arm B, StartLux-Decision-9B with E12c's recipe) close the fixed-question gap to the best open-weights decision
model without losing the common-state and live levels? Two NEW checkpoints, "AgentDecision-4B (mixed+)" (served as
``agentdecision-4b-mixedplus``) and "StartLux-9B (mixed)" (served as ``startlux-9b-mixed``). The frozen checkpoint, E12
and E12c are unchanged. No test; outside every Holm table.

    python scripts/explore_e16_best_open.py build --toolace data.json --e12-data DIR --out DIR   # local, CPU, $0
    python scripts/explore_e16_best_open.py verify --data DIR                                    # the box: sha256
    python scripts/explore_e16_best_open.py spend | watch --pid P --cap C | project --arm a|b --out F   # live
    python scripts/explore_e16_best_open.py c-gate --answers F [--margin 1.0]                    # the live gate (L2)
    python scripts/explore_e16_best_open.py summarize [--box DIR]                                # the readout

Items: E12's builders unchanged (``explore_e12_train``: E1's converter ``bench/suites/static.py``, E12's label rule for
the When2Call training split, E12's ToolACE parser, E12's renderer ``to_row``), E12's leakage audit (exact match or a
shared 8-gram of the user request against E1's 3,991 decisions, E1's 299-item development prefix and the full public
test pools; a unit with any hit is dropped with all its items) and E12's development rule (a unit is a development
unit iff sha256("jev-explore-E12-dev\n" + unit) mod 10 == 0; never trained on). Only the training quotas differ (twice
E12's items, tilted to the When2Call-format families) and the replay is a seeded sample of 14,000 SFT v1 rows (1 : 1).
The public development split, its per-family files and the temperature set are E12's files, sha256-checked."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import explore_e12_train as E12  # noqa: E402

OUT = ROOT / "results" / "explore" / "e16_best_open"
BOX = OUT / "box"
TEX = ROOT / "paper" / "generated" / "e16_best_open.tex"
LABEL = ("exploratory (E16), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-08 'E16: the best open-weights decision "
         "model on all three levels'; outside every Holm table")
ARMS = {"a": {"name": "AgentDecision-4B (mixed+)", "served": "agentdecision-4b-mixedplus", "base": "Qwen3-4B",
              "recipe": "E12's continuation of the frozen adapter, 2 epochs, epoch chosen on development data",
              "e1_backend": "e16a", "live_run": "e16a_live", "smoke_run": "e16a_live_smoke", "arm": "e16a_ts_replace_f"},
        "b": {"name": "StartLux-9B (mixed)", "served": "startlux-9b-mixed", "base": "StartLux-Decision-9B",
              "recipe": "E12c's fresh-LoRA SFT recipe, at most 2 epochs, epoch chosen on development data",
              "e1_backend": "e16b", "live_run": "e16b_live", "smoke_run": "e16b_live_smoke", "arm": "e16b_ts_replace_f"}}
E1_RUN = "e1_bfcl_when2call_e16"
# training quotas per family (fixed before any item was built): twice E12's 7,000, tilted to the When2Call-format
# families (E12: call / ask / cannot 1,000 each, tool choice 1,500, "none" 500, arguments 2,000)
QUOTA_TRAIN = {"w2c_call": 2500, "w2c_ask": 3000, "w2c_cannot": 3000, "bfcl_tool": 1900, "bfcl_none": 1000,
               "bfcl_args": 2600}
N_REPLAY = 14000                      # = sum(QUOTA_TRAIN): new : replay = 1 : 1 (of 15,694 SFT v1 rows, seeded)
SEED = 0
E12_FILES = ("public_dev.jsonl.gz", "cal_dev.jsonl.gz", "dev_w2c.jsonl.gz", "dev_tool.jsonl.gz", "dev_args.jsonl.gz")
SL9B_REPO, SL9B_REV = "startlux-models/StartLux-Decision-9B", "342aad64840be8eee314455428a5a775f18ef547"
# live (L3): the same first 200 fresh Stage-2 tasks as E12 / E12c; one Gemini cap for both arms
LIVE_N = 200
CAP_USD = 8.0
PROJ_FACTOR = 1.2
C_MARGIN_PT = 1.0                     # the live gate: C accuracy within 1 point of AgentDecision-4B's
RUNS = {"e16": ("e16a_live_smoke", "e16a_live", "e16b_live_smoke", "e16b_live")}
NAN = float("nan")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _read(p: Path) -> list[dict[str, Any]]:
    return E12._read(p)


# ============================================================================== build (local, CPU, $0)
def _pick(units: Sequence[Mapping[str, Any]], fam: str, k: int, rng: random.Random, used: set[str],
          per_unit: bool = True) -> list[tuple[Any, str]]:
    return E12._pick(units, fam, k, rng, used, per_unit)


def select16(units: Sequence[Mapping[str, Any]], quota: Mapping[str, int], rng: random.Random
             ) -> dict[str, list[tuple[Any, str]]]:
    """Per family from disjoint units within a source, as E12: ToolACE "call" items first, then tool-choice items from
    the remaining ToolACE units (arguments from any ToolACE unit); When2Call-train "none" items first, then "cannot",
    then "ask". The family named in the author's question (When2Call "call") fills first."""
    ta = [u for u in units if u["source"] == "toolace"]
    wc = [u for u in units if u["source"] == "when2call-train"]
    used_ta: set[str] = set()
    used_wc: set[str] = set()
    return {"w2c_call": _pick(ta, "w2c_call", quota["w2c_call"], rng, used_ta),
            "bfcl_tool": _pick(ta, "bfcl_tool", quota["bfcl_tool"], rng, used_ta),
            "bfcl_args": _pick(ta, "bfcl_args", quota["bfcl_args"], rng, set(), per_unit=False),
            "bfcl_none": _pick(wc, "bfcl_none", quota["bfcl_none"], rng, used_wc),
            "w2c_cannot": _pick(wc, "w2c_cannot", quota["w2c_cannot"], rng, used_wc),
            "w2c_ask": _pick(wc, "w2c_ask", quota["w2c_ask"], rng, used_wc)}


def build(toolace: Path, e12_data: Path, out: Path) -> dict[str, Any]:
    from jevharness.bench.suites.static import WHEN2CALL_COMMIT
    from jevharness.harness.tools import OTHER

    if E12._sha(toolace) != E12.TOOLACE_SHA256:
        raise SystemExit("ToolACE data.json differs from the pinned revision's sha256")
    e12_info = json.loads((ROOT / "results" / "explore" / "e12" / "build_info.json").read_text())
    for f in E12_FILES:   # E12's development split and temperature set, unchanged
        if not (e12_data / f).exists() or E12._sha(e12_data / f) != e12_info["sha256"][f]:
            raise SystemExit(f"{f} is missing or differs from E12's record")
    rng = random.Random(SEED)
    wu, w_why = E12.when2call_units()
    tu, t_why = E12.toolace_units(toolace)
    units, seen, n_dup = [], set(), Counter()
    for u in wu + tu:
        if u["unit"] in seen:
            n_dup[u["source"]] += 1
            continue
        seen.add(u["unit"])
        units.append(u)
    aud = E12.Audit(E12.audit_targets())
    hits = {u["unit"]: h for u in units if (h := aud.hits(u["text"]))}
    e12_hits = json.loads((ROOT / "results" / "explore" / "e12" / "audit_hits.json").read_text())["hits"]
    table: dict[str, Counter] = {}
    for uid, h in hits.items():
        src = "toolace" if uid.startswith("toolace-") else "when2call-train"
        t = table.setdefault(src, Counter())
        for tgt, kind in h.items():
            t[f"{tgt}:{kind}"] += 1
        t["units_dropped"] += 1
    n_items = {u["unit"]: sum(len(v) for v in u["items"].values()) for u in units}
    clean = [u for u in units if u["unit"] not in hits]
    dev_u = [u for u in clean if E12.is_dev(u["unit"])]
    tr_u = [u for u in clean if not E12.is_dev(u["unit"])]
    sel_tr = select16(tr_u, QUOTA_TRAIN, rng)
    rows_tr = [r for fam, its in sel_tr.items() for it, unit in its if (r := E12.to_row(it, fam, unit))]
    dev_rows = _read(e12_data / "public_dev.jsonl.gz")
    dev_units = {r["meta"]["unit"] for r in dev_rows}
    assert not {r["meta"]["unit"] for r in rows_tr} & dev_units, "train / E12's dev split share a unit"
    assert not {r["meta"]["unit"] for r in rows_tr} & set(hits), "an audited unit got through"
    assert dev_units <= {u["unit"] for u in dev_u}, "E12's dev units are not this build's dev units"
    n_rep = min(N_REPLAY, len(rows_tr))          # 1 : 1 with what the quotas actually yield
    replay, rep_info = E12.replay_rows(n_rep, rng)
    for r in rows_tr:
        r["meta"]["e16"] = r["meta"].pop("e12")
    for r in replay:
        r["meta"]["e16"] = r["meta"].pop("e12")
    mix = rows_tr + replay
    rng.shuffle(mix)
    out.mkdir(parents=True, exist_ok=True)
    n = {"train_mix": E12._write(mix, out / "train_mix.jsonl.gz"), "public_train": len(rows_tr), "replay": len(replay)}
    for f in E12_FILES:
        shutil.copyfile(e12_data / f, out / f)
        n[f.split(".")[0]] = sum(1 for _ in _read(out / f))
    fam_of = lambda r: r["meta"]["e16"]  # noqa: E731
    info = {
        "label": LABEL, "generated": _now(), "seed": SEED, "quota_train": QUOTA_TRAIN, "n_replay": n_rep,
        "dev_rule": e12_info["dev_rule"], "rows": n,
        "train_by_family": dict(Counter(fam_of(r) for r in rows_tr)),
        "train_by_source": dict(Counter(r["source"].split(":")[0] for r in rows_tr)),
        "train_by_dp": dict(Counter(r["dp"] for r in mix)),
        "train_units": len({r["meta"]["unit"] for r in rows_tr}),
        "train_args_other": sum(1 for r in rows_tr if fam_of(r) == "bfcl_args" and max(r["target"], key=r["target"].get) == OTHER),
        "train_options_mean": {f: round(sum(len(r["options"]) for r in rows_tr if fam_of(r) == f)
                                        / max(1, sum(1 for r in rows_tr if fam_of(r) == f)), 2) for f in QUOTA_TRAIN},
        "pool_train_units": {"toolace": sum(1 for u in tr_u if u["source"] == "toolace"),
                             "when2call_train": sum(1 for u in tr_u if u["source"] == "when2call-train"),
                             "when2call_train_ask": sum(1 for u in tr_u if u["source"] == "when2call-train" and u["label"] == "request_for_info"),
                             "when2call_train_cannot": sum(1 for u in tr_u if u["source"] == "when2call-train" and u["label"] == "cannot_answer"),
                             "when2call_train_cannot_with_tools": sum(1 for u in tr_u if "bfcl_none" in u["items"])},
        "overlap_with_e12_train_units": None,     # E12's training units are not recorded (its items were sampled from the same pool)
        "sources": {"when2call_train": {"url_commit": WHEN2CALL_COMMIT, "file": "train/when2call_train_sft.jsonl",
                                        "sha256": E12._sha(E12.W2C_TRAIN), "license": "CC-BY-4.0", "rows_by_outcome": dict(w_why)},
                    "toolace": {"repo": E12.TOOLACE_REPO, "revision": E12.TOOLACE_REV, "url": E12.TOOLACE_URL,
                                "sha256": E12.TOOLACE_SHA256, "license": "Apache-2.0", "rows_by_outcome": dict(t_why)}},
        "units": {"total": len(units), "toolace": sum(1 for u in units if u["source"] == "toolace"),
                  "when2call_train": sum(1 for u in units if u["source"] == "when2call-train"),
                  "duplicate_rows_dropped": dict(n_dup), "clean": len(clean), "train": len(tr_u), "dev": len(dev_u)},
        "audit": {"targets": aud.sizes, "rule": e12_info["audit"]["rule"], "units_with_hit": len(hits),
                  "items_dropped": sum(n_items[u] for u in hits), "same_hits_as_e12": set(hits) == set(e12_hits),
                  "by_source": {k: dict(v) for k, v in table.items()},
                  "by_target": {t: {k: sum(1 for h in hits.values() if h.get(t) == k) for k in ("exact", "ngram")}
                                for t in aud.sizes}},
        "replay": rep_info,
        "e12_files": {f: e12_info["sha256"][f] for f in E12_FILES},
        "e12_record": {"generated": e12_info["generated"], "quota_train": e12_info["quota_train"], "n_replay": e12_info["n_replay"]},
        "startlux_9b": {"repo": SL9B_REPO, "revision": SL9B_REV, "license": "CC BY-NC 4.0 (model card, read 2026-10-08)"},
    }
    info["sha256"] = {f.name: E12._sha(f) for f in sorted(out.glob("*.jsonl.gz"))}
    OUT.mkdir(parents=True, exist_ok=True)
    (out / "build_info.json").write_text(json.dumps(info, indent=1) + "\n")
    (OUT / "build_info.json").write_text(json.dumps(info, indent=1) + "\n")
    (OUT / "audit_hits.json").write_text(json.dumps({"label": LABEL, "hits": hits}, indent=1, sort_keys=True) + "\n")
    return info


def verify(data: Path, record: Path = OUT / "build_info.json") -> int:
    want = json.loads(record.read_text())["sha256"]
    bad = [f for f, s in want.items() if not (data / f).exists() or E12._sha(data / f) != s]
    print("data sha256:", "all match" if not bad else f"MISMATCH {bad}")
    return 1 if bad else 0


# ============================================================================== live: spend, watchdog, gate
def total_spend(runs: Path) -> dict[str, Any]:
    import explore_e6a_alfworld as E6A

    per = {name: E6A.run_spend(runs / name) for name in RUNS["e16"] if (runs / name).is_dir()}
    return {"by_run": per, **{k: sum(v[k] for v in per.values()) for k in ("gemini", "jev", "total", "records")}}


def cmd_watch(a: argparse.Namespace) -> int:
    """At the cap: SIGTERM / SIGKILL to the runner's process group and ``<runs>/e16_STOP.json``."""
    import os
    import signal
    import time

    stop = Path(a.runs) / "e16_STOP.json"
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend(Path(a.runs))
        if s["total"] >= a.cap:
            stop.write_text(json.dumps({"at": _now(), "spend": s, "cap": a.cap}, indent=1))
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(a.pid), sig)
                except OSError:
                    pass
                time.sleep(10)
            print(f"CAP REACHED: ${s['total']:.2f} >= ${a.cap:.2f}; runner stopped", flush=True)
            return 1
        print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} spend ${s['total']:.3f} ({s['records']} records)", flush=True)
        time.sleep(a.every)


def projection(runs: Path, arm: str) -> dict[str, Any]:
    """Spend so far + 1.2 x (mean $ per episode of the arm's smoke, or of the other arm's finished live run when one
    exists, whichever is larger) x 200 must fit the $8 cap, else no live episode for this arm."""
    import explore_e6a_alfworld as E6A

    def per_episode(name: str) -> list[float]:
        eps = []
        d = runs / name
        for p in sorted(d.glob("*.jsonl")) if d.is_dir() else []:
            for t in E6A._latest(p).values():
                if not t.get("error"):
                    eps.append(E6A.episode_spend(t)["total"])
        return eps

    spent = total_spend(runs)["total"]
    smoke = per_episode(ARMS[arm]["smoke_run"])
    if not smoke:
        return {"decision": "stop: no smoke episode", "spent": spent}
    other = per_episode(ARMS["b" if arm == "a" else "a"]["live_run"])
    per = max([sum(smoke) / len(smoke)] + ([sum(other) / len(other)] if other else []))
    proj = spent + per * LIVE_N * PROJ_FACTOR
    return {"arm": arm, "smoke_episodes": len(smoke), "other_live_episodes": len(other), "usd_per_episode": per,
            "spent": spent, "projected": proj, "cap": CAP_USD,
            "decision": f"run all {LIVE_N}" if proj <= CAP_USD else "stop: projection above the cap"}


def c_gate(answers: Path, margin_pt: float = C_MARGIN_PT) -> dict[str, Any]:
    """The registered live gate: the arm's common-state accuracy (task means) within ``margin_pt`` of AgentDecision-4B's
    registered answers (``results/stage2/s23/answers_rl.jsonl``)."""
    import ablation_score as AS
    import e5_analysis as EA

    corpus = ROOT / "results" / "stage2" / "s23"
    occ, eligible = AS.load_corpus(corpus)
    arm, _ = EA.c_rows(occ, eligible, AS.load_answers(answers))
    ref, _ = EA.c_rows(occ, eligible, AS.load_answers(corpus / "answers_rl.jsonl"))
    gap = 100 * (arm["c_acc"] - ref["c_acc"])
    return {"c_acc": arm["c_acc"], "frozen_c_acc": ref["c_acc"], "gap_pt": gap, "margin_pt": margin_pt,
            "missing": arm["c_missing"], "errors": arm["c_errors"],
            "live": bool(gap >= -margin_pt and arm["c_missing"] == 0)}


# ============================================================================== cli
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--toolace", required=True)
    b.add_argument("--e12-data", required=True, help="E12's data directory (s3://.../e12/data): the dev and temperature files")
    b.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--data", required=True)
    s = sub.add_parser("spend")
    s.add_argument("--runs", default="runs")
    s.add_argument("--short", action="store_true")
    w = sub.add_parser("watch")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--cap", type=float, default=CAP_USD)
    w.add_argument("--runs", default="runs")
    w.add_argument("--every", type=int, default=30)
    p = sub.add_parser("project")
    p.add_argument("--arm", choices=("a", "b"), required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", required=True)
    g = sub.add_parser("c-gate")
    g.add_argument("--answers", required=True)
    g.add_argument("--margin", type=float, default=C_MARGIN_PT)
    g.add_argument("--out", default=None)
    sm = sub.add_parser("summarize")
    sm.add_argument("--box", default=str(BOX))
    sm.add_argument("--runs", default=str(ROOT / "runs"))
    a = ap.parse_args(argv)
    if a.cmd == "build":
        info = build(Path(a.toolace), Path(a.e12_data), Path(a.out))
        print(json.dumps({k: info[k] for k in ("rows", "train_by_family", "train_by_source", "units", "pool_train_units",
                                                "replay")}, indent=1))
        print(json.dumps(info["audit"], indent=1))
        return 0
    if a.cmd == "verify":
        return verify(Path(a.data))
    if a.cmd == "spend":
        s = total_spend(Path(a.runs))
        print(f"${s['total']:.3f} ({s['records']} records)" if a.short else json.dumps(s, indent=1))
        return 0
    if a.cmd == "watch":
        return cmd_watch(a)
    if a.cmd == "project":
        r = projection(Path(a.runs), a.arm)
        Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
        print(json.dumps(r, indent=1))
        return 0
    if a.cmd == "c-gate":
        r = c_gate(Path(a.answers), a.margin)
        if a.out:
            Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
        print(json.dumps(r, indent=1))
        return 0 if r["live"] else 3
    if a.cmd == "summarize":
        from explore_e16_summary import summarize   # the readout lives beside this file (written after the registration)

        return summarize(Path(a.box), Path(a.runs))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
