#!/usr/bin/env python3
"""Box-side helpers of scripts/aws/train_ablation.sh, the exploratory SFT training ablation E3 (plan:
results/training/ablation_plan.md, addendum 2026-10-05). CPU only: no GPU, no API call, no key.

    host        --out H [--itype T] [--gpu NAME] [--cap-h 24] [--serve-mode fast]  host profile + cap-projection priors
    inputs      --run A [--teacher F]           arm files / teacher input vs results/training/ablation_inputs.json
    smoke-file  --run A --out F                 the memory smoke rows: the longest rows of D100 + random D10 rows
    oom         LOG                              exit 0 when LOG shows a CUDA out-of-memory error, else 1
    setting     NAME | --next NAME              trainer flags of a memory setting / the next fallback ('none' at the end)
    used-h      --run A                          E3 GPU hours so far (finished stages + the running one)
    cap         --run A --arms X [Y ...]         one cap unit: prints 'go ...' (exit 0) or 'skip ...' (exit 3)
    record      --run A --arm X --setting S --attempts a:oom,b:ok --sft-min M --itype T [--log L] [--mix-sha H]
    arm-time    --run A --arm X --eval-min E     append the arm's minutes to A/arm_times.jsonl (cap projection)
    anchor      --run A --serve-mode fast|plain  plan section 6 go / no-go -> A/anchor_check.json
                                                 (exit 0 ok; 2 only the C answers differ; 1 otherwise)
    cost        --run A                          every replay record has cost_usd == 0 and a local backend
    answers     --run A --arm X                  every question of the corpus copy answered (else the replay resumes)

Memory settings (plan addendum: never LR, epochs, effective batch, loss, LoRA rank or data): nogc_3072 = the recipe
(3,072-token micro-batches, no activation checkpointing); gc_3072 = the same with activation checkpointing (same maths);
gc_2048 / gc_1536 / gc_1024 = a smaller token budget with checkpointing and the same 64 rows per optimizer step (the
trainer derives the accumulation; same maths up to float order and the rows grouped into each step).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import ablation_subsets as AB  # noqa: E402

INPUTS = ROOT / "results" / "training" / "ablation_inputs.json"
ANCHOR_EVAL = ROOT / "results" / "gpu_round5" / "eval" / "eval_sft_v1_v1.json"     # v1-dev 0.99706 (round 5)
ANCHOR_CAL = ROOT / "results" / "stage2" / "gpu" / "rl_attr" / "sft_v1_calibration.json"
ANCHOR_ANSWERS = ROOT / "results" / "stage2" / "s23" / "answers_sft_v1.jsonl"
V1_DEV = "data/train_v1/dev.jsonl.gz"
ANCHOR = AB.ANCHOR
ACC_TOL = 0.001                 # plan section 6
AGREE_MIN = 0.99                # plan section 6
# on-demand USD / h, us-east-1 (results/stage2/tariff_v2_plan.json, results/harness_final/tariff.json,
# results/gpu_round5 for the g6e.4xlarge); another type is recorded unpriced
PRICES = {"g6e.2xlarge": 2.24208, "g6e.4xlarge": 3.004, "g6e.16xlarge": 7.57719, "g5.4xlarge": 1.624, "g5.8xlarge": 2.448}  # g5.8xlarge: AWS on-demand, us-east-1 (2026-10-05; the E3 host: no L40S / g5.4xlarge capacity)
SFT_MIN_PER_ROW_L40S = 134 / 16902   # SFT v1: 134 min for 16,902 rows incl. replay, 2 epochs, one L40S, no checkpointing
# priors of the cap projection (plan addendum 2026-10-05); replaced by the measured minutes once an arm has finished
PRIORS: dict[str, dict[str, Any]] = {
    # plan section 5: eval 7 + calibrate 3 + C replay 18 min (plain server; --fast not measured on the L40S)
    "L40S": {"train_factor": 1.0, "gc_factor": 1.3, "load_min": 3.0, "eval_min": {"fast": 28.0, "plain": 28.0},
             "temp_rel_tol": 1e-3},
    # training x 2.9 (bf16 dense tensor throughput, about 362 vs 125 TFLOPS); evaluation and replay x 2.21 (measured
    # tool-selection service time, A10G 487 ms vs L40S 220 ms per request), the fast server x 0.758 (370 vs 488 ms)
    "A10G": {"train_factor": 2.9, "gc_factor": 1.3, "load_min": 5.0, "eval_min": {"fast": 52.0, "plain": 61.0},
             "temp_rel_tol": 2e-2},
}
SETTINGS: list[tuple[str, int, bool]] = [("nogc_3072", 3072, False), ("gc_3072", 3072, True), ("gc_2048", 2048, True),
                                         ("gc_1536", 1536, True), ("gc_1024", 1024, True)]
OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation|"
                    r"CUDA error: out of memory")
GA_RE = re.compile(r"token budget (\d+): (\d+) micro-batches/epoch, ([\d.]+) rows each on average, grad-accum (\d+)")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _jsonl(p: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ host
def gpu_class(name: str) -> str:
    n = (name or "").upper()
    return "L40S" if "L40S" in n else "A10G" if "A10G" in n else "other"


def host_profile(itype: str, gpu: str, cap_h: float, serve_mode: str = "fast") -> dict[str, Any]:
    cls = gpu_class(gpu) if gpu else ("L40S" if itype.startswith("g6e") else "A10G" if itype.startswith("g5") else "other")
    pri = PRIORS.get(cls, PRIORS["A10G"])          # an unknown GPU: the slower host's priors and tolerance
    return {"instance_type": itype, "gpu_name": gpu, "gpu_class": cls, "usd_per_h": PRICES.get(itype),
            "cap_h": cap_h, "serve_mode_planned": serve_mode, "priors": pri,
            "recorded_utc": _now().strftime("%Y-%m-%dT%H:%M:%SZ")}


# ------------------------------------------------------------------ inputs
def check_inputs(run: Path, teacher: Path | None = None, inputs: Path = INPUTS) -> dict[str, Any]:
    want = json.loads(inputs.read_text())
    man = json.loads((run / "data" / "manifest.json").read_text())
    res: dict[str, Any] = {"arms": {}, "mismatch": []}
    if man.get("source_sha256") != want["train_v1_sha256"]:
        res["mismatch"].append("data/train_v1/train.jsonl.gz")
    if man.get("teacher_only_content_sha256") != want["teacher_only"]["jsonl_sha256"]:
        res["mismatch"].append("teacher_only (content)")
    if teacher is not None and teacher.exists() and str(teacher).endswith(".gz"):
        res["teacher_gz_sha256"] = _sha(teacher)
        if res["teacher_gz_sha256"] != want["teacher_only"]["gz_sha256"]:
            res["mismatch"].append("teacher_only (gz)")
    for arm, w in want["arms"].items():
        got = (man.get("arms") or {}).get(arm) or {}
        ok = got.get("sha256") == w["sha256"] and got.get("rows") == w["rows"]
        res["arms"][arm] = {"rows": got.get("rows"), "sha256_ok": ok}
        if not ok:
            res["mismatch"].append(arm)
    res["ok"] = not res["mismatch"]
    return res


# ------------------------------------------------------------------ smoke rows
def prompt_chars(r: dict[str, Any]) -> int:
    return len(r.get("state") or "") + len(r.get("question") or "") + sum(len(o) + 4 for o in r.get("options") or [])


def smoke_rows(d100: list[dict[str, Any]], d10: list[dict[str, Any]], n_long: int = 48, n_rand: int = 48,
               seed: int = 0) -> list[dict[str, Any]]:
    """The memory smoke's rows: the n_long longest prompts of D100 (the longest micro-batches any arm can form) and
    n_rand random D10 rows (short rows packed many to a micro-batch). Deterministic."""
    longest = sorted(range(len(d100)), key=lambda i: (-prompt_chars(d100[i]), i))[:n_long]
    ids = {d100[i]["id"] for i in longest}
    pool = [r for r in d10 if r["id"] not in ids]
    rand = random.Random(f"{seed}:smoke").sample(range(len(pool)), min(n_rand, len(pool)))
    return [d100[i] for i in sorted(longest)] + [pool[i] for i in sorted(rand)]


# ------------------------------------------------------------------ settings, OOM
def setting_flags(name: str) -> str:
    for n, tok, gc in SETTINGS:
        if n == name:
            return f"--max-batch-tokens {tok} --eff-batch 64 " + ("--gradient-checkpointing" if gc
                                                                   else "--no-gradient-checkpointing")
    raise SystemExit(f"unknown memory setting {name!r}; known {[s[0] for s in SETTINGS]}")


def next_setting(name: str) -> str:
    names = [s[0] for s in SETTINGS]
    i = names.index(name)
    return names[i + 1] if i + 1 < len(names) else "none"


def is_oom(text: str) -> bool:
    return bool(OOM_RE.search(text))


# ------------------------------------------------------------------ GPU hours and the cap
def used_hours(run: Path, now: dt.datetime | None = None) -> float:
    now = now or _now()
    mins = 0.0
    p = run / "stage_times.jsonl"
    if p.exists():
        mins += sum(float(json.loads(x)["min"]) for x in p.read_text().splitlines() if x.strip())
    c = run / "cur_stage.json"
    if c.exists():
        start = dt.datetime.fromisoformat(json.loads(c.read_text())["start"].replace("Z", "+00:00"))
        mins += max(0.0, (now - start).total_seconds() / 60)
    return mins / 60


def rows_with_replay(arm: str, inputs: Path = INPUTS) -> int:
    if arm == ANCHOR:
        return 0
    return int(json.loads(inputs.read_text())["arms"][AB.data_arm(arm)]["rows_with_replay"])


def estimate_min(run: Path, arm: str, inputs: Path = INPUTS) -> float:
    """Projected GPU minutes of one arm: SFT + evaluation. Priors of the host (host.json) until an arm has finished on
    this run; then the measured minutes per row incl. replay and the measured evaluation minutes."""
    host = json.loads((run / "host.json").read_text()) if (run / "host.json").exists() else host_profile("", "", 24)
    pri = host["priors"]
    mode = (run / "serve_mode.txt").read_text().strip() if (run / "serve_mode.txt").exists() \
        else host.get("serve_mode_planned", "fast")
    setting = (run / "mem_setting.txt").read_text().strip() if (run / "mem_setting.txt").exists() else "nogc_3072"
    done = []
    if (run / "arm_times.jsonl").exists():
        done = [json.loads(x) for x in (run / "arm_times.jsonl").read_text().splitlines() if x.strip()]
    rows = rows_with_replay(arm, inputs)
    trained = [d for d in done if d.get("rows") and d.get("sft_min") is not None]
    if trained:
        rate = sum(max(0.0, d["sft_min"] - pri["load_min"]) for d in trained) / sum(d["rows"] for d in trained)
    else:
        rate = SFT_MIN_PER_ROW_L40S * pri["train_factor"] * (pri["gc_factor"] if setting.startswith("gc") else 1.0)
    evals = [d["eval_min"] for d in done if d.get("eval_min") is not None]
    ev = sum(evals) / len(evals) if evals else pri["eval_min"].get(mode, max(pri["eval_min"].values()))
    sft = 0.0 if arm == ANCHOR else rate * rows + pri["load_min"]
    return sft + ev


def cap_check(run: Path, arms: list[str], cap_h: float, inputs: Path = INPUTS) -> tuple[bool, str]:
    todo = [a for a in arms if not (run / "done" / a).exists()]
    used = used_hours(run)
    est = sum(estimate_min(run, a, inputs) for a in todo) / 60
    msg = f"unit {'+'.join(arms)}: used {used:.2f} h + projected {est:.2f} h vs cap {cap_h:g} h"
    return used + est <= cap_h, msg


def record_setting(run: Path, arm: str, setting: str, attempts: str, sft_min: float, itype: str,
                   log: Path | None = None, mix_sha: str = "", inputs: Path = INPUTS) -> dict[str, Any]:
    tok = next(t for n, t, _ in SETTINGS if n == setting)
    gc = next(g for n, _, g in SETTINGS if n == setting)
    rec: dict[str, Any] = {"arm": arm, "data": AB.data_arm(arm), "trainer_seed": AB.trainer_seed(arm),
                           "setting": setting, "max_batch_tokens": tok, "gradient_checkpointing": gc,
                           "eff_batch_rows": 64, "attempts": [a for a in attempts.split(",") if a],
                           "sft_min": sft_min, "instance_type": itype,
                           "rows_with_replay": rows_with_replay(arm)}
    want = ((json.loads(inputs.read_text()).get("train_mix") or {}).get("arms") or {}).get(arm) or {}
    if mix_sha:   # the arm's exact training file (rows + replay) vs the hash computed before the run
        rec.update({"train_mix_sha256": mix_sha, "train_mix_sha256_expected": want.get("sha256"),
                    "train_mix_ok": mix_sha == want.get("sha256")})
    if log is not None and log.exists():
        m = GA_RE.search(log.read_text(errors="replace"))
        if m:
            rec.update({"micro_batches_per_epoch": int(m.group(2)), "rows_per_micro_batch": float(m.group(3)),
                        "grad_accum": int(m.group(4)), "rows_per_step": round(int(m.group(4)) * float(m.group(3)), 1)})
    (run / "settings").mkdir(parents=True, exist_ok=True)
    (run / "settings" / f"{arm}.json").write_text(json.dumps(rec, indent=1) + "\n")
    with open(run / "arm_times.jsonl", "a") as f:
        f.write(json.dumps({"arm": arm, "rows": rec["rows_with_replay"], "sft_min": sft_min, "setting": setting})
                + "\n")
    return rec


def record_eval_time(run: Path, arm: str, eval_min: float) -> None:
    with open(run / "arm_times.jsonl", "a") as f:
        f.write(json.dumps({"arm": arm, "eval_min": eval_min}) + "\n")


# ------------------------------------------------------------------ anchor go / no-go (plan section 6)
def latest_answers(p: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if p.exists():
        for x in p.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                out[r["qhash"]] = r
    return out


def temps_flat(c: dict[str, Any]) -> dict[str, float]:
    return {"global": float(c["global"]), **{f"b:{k}": float(v) for k, v in (c.get("buckets") or {}).items()}}


def anchor_check(run: Path, serve_mode: str, host: dict[str, Any] | None = None) -> dict[str, Any]:
    host = host or (json.loads((run / "host.json").read_text()) if (run / "host.json").exists() else
                    host_profile("", "", 24))
    tol = float(host["priors"]["temp_rel_tol"])
    res: dict[str, Any] = {"arm": ANCHOR, "serve_mode": serve_mode, "instance_type": host.get("instance_type"),
                           "gpu_class": host.get("gpu_class")}
    # (1) v1-dev accuracy vs round 5's record
    want = json.loads(ANCHOR_EVAL.read_text())["files"]
    p = run / "eval" / f"{ANCHOR}_v1.json"
    got = json.loads(p.read_text())["files"] if p.exists() else {}
    v1: dict[str, Any] = {"tolerance": ACC_TOL}
    for f, w in want.items():
        g = (got.get(f) or {}).get("overall") or {}
        v1[f] = {"recorded": w["overall"]["acc"], "got": g.get("acc"), "n": g.get("n"),
                 "diff": None if g.get("acc") is None else g["acc"] - w["overall"]["acc"]}
    v1["ok"] = v1.get(V1_DEV, {}).get("diff") is not None and abs(v1[V1_DEV]["diff"]) <= ACC_TOL
    res["v1_dev"] = v1
    # (2) temperatures refitted on v1 dev vs the Stage-2 fit (results/stage2/gpu/rl_attr/sft_v1_calibration.json)
    rec = temps_flat(json.loads(ANCHOR_CAL.read_text()))
    p = run / "cal" / f"{ANCHOR}_calibration.json"
    fit = temps_flat(json.loads(p.read_text())) if p.exists() else {}
    rel = {k: abs(fit[k] - rec[k]) / abs(rec[k]) for k in rec if k in fit}
    res["temperatures"] = {"recorded": rec, "refit": fit, "rel_diff": rel, "tolerance": tol,
                           "tolerance_note": "1e-3 on an L40S (the GPU type of the recorded fit); 2e-2 on another GPU "
                                             "type (bf16 kernels differ; fixed in the plan addendum before any run)",
                           "ok": bool(fit) and set(fit) == set(rec) and max(rel.values()) <= tol}
    # (3) the common-state corpus C: the anchor's answers vs the published SFT v1 answers
    pub = latest_answers(ANCHOR_ANSWERS)
    mine = latest_answers(run / "s23" / f"answers_abl_{ANCHOR}.jsonl")
    common = set(pub) & set(mine)
    agree = sum(mine[q].get("answer") == pub[q].get("answer") for q in common)
    errors = sum(1 for r in mine.values() if r.get("error"))
    share = agree / len(pub) if pub else 0.0       # a missing answer counts as a disagreement
    res["c_answers"] = {"published": len(pub), "answered": len(mine), "errors": errors, "agree": agree,
                        "agreement": share, "min": AGREE_MIN, "ok": share >= AGREE_MIN and errors == 0}
    # (4) no priced replay record
    cost = cost_check(run)
    res["cost"] = {k: cost[k] for k in ("records", "priced", "ok")}
    res["ok"] = all(res[k]["ok"] for k in ("v1_dev", "temperatures", "c_answers", "cost"))
    res["checked_utc"] = _now().strftime("%Y-%m-%dT%H:%M:%SZ")
    return res


def answers_complete(run: Path, arm: str) -> dict[str, Any]:
    """Every distinct question of the corpus copy answered by the arm (an error record counts as answered and is
    scored incorrect, A7.10 (d)); a missing one means the replay must be resumed."""
    qs = {q["qhash"] for q in _jsonl(run / "s23" / "questions.jsonl.gz")}
    got = latest_answers(run / "s23" / f"answers_abl_{arm}.jsonl")
    return {"questions": len(qs), "answered": len(qs & set(got)),
            "errors": sum(1 for q in qs if (got.get(q) or {}).get("error")), "ok": qs <= set(got)}


def cost_check(run: Path) -> dict[str, Any]:
    recs = priced = 0
    nonlocal_backends: list[str] = []
    for p in sorted((run / "s23").glob("answers_abl_*.jsonl")):
        for x in p.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                recs += 1
                if r.get("cost_usd") is None or float(r["cost_usd"]) != 0.0:
                    priced += 1
    for p in sorted((run / "s23").glob("replay_abl_*.json")):
        url = str((json.loads(p.read_text()).get("backend") or {}).get("base_url") or "")
        if not url.startswith("http://127.0.0.1:"):
            nonlocal_backends.append(p.name)
    return {"records": recs, "priced": priced, "nonlocal_backends": nonlocal_backends,
            "ok": priced == 0 and not nonlocal_backends}


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("host")
    h.add_argument("--out", required=True)
    h.add_argument("--itype", default="")
    h.add_argument("--gpu", default="")
    h.add_argument("--cap-h", type=float, default=24.0)
    h.add_argument("--serve-mode", default="fast", choices=["fast", "plain"])
    i = sub.add_parser("inputs")
    i.add_argument("--run", required=True)
    i.add_argument("--teacher", default=None)
    s = sub.add_parser("smoke-file")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-long", type=int, default=48)
    s.add_argument("--n-rand", type=int, default=48)
    o = sub.add_parser("oom")
    o.add_argument("log")
    st = sub.add_parser("setting")
    st.add_argument("name")
    st.add_argument("--next", action="store_true")
    u = sub.add_parser("used-h")
    u.add_argument("--run", required=True)
    c = sub.add_parser("cap")
    c.add_argument("--run", required=True)
    c.add_argument("--arms", nargs="+", required=True)
    c.add_argument("--cap-h", type=float, required=True)
    r = sub.add_parser("record")
    r.add_argument("--run", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--setting", required=True)
    r.add_argument("--attempts", default="")
    r.add_argument("--sft-min", type=float, required=True)
    r.add_argument("--itype", default="")
    r.add_argument("--log", default=None)
    r.add_argument("--mix-sha", default="", help="sha256 of OUT/train_mix.jsonl (checked against ablation_inputs.json)")
    e = sub.add_parser("arm-time")
    e.add_argument("--run", required=True)
    e.add_argument("--arm", required=True)
    e.add_argument("--eval-min", type=float, required=True)
    an = sub.add_parser("anchor")
    an.add_argument("--run", required=True)
    an.add_argument("--serve-mode", required=True, choices=["fast", "plain"])
    co = sub.add_parser("cost")
    co.add_argument("--run", required=True)
    aw = sub.add_parser("answers")
    aw.add_argument("--run", required=True)
    aw.add_argument("--arm", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "host":
        prof = host_profile(a.itype, a.gpu, a.cap_h, a.serve_mode)
        Path(a.out).write_text(json.dumps(prof, indent=1) + "\n")
        print(json.dumps({k: prof[k] for k in ("instance_type", "gpu_class", "usd_per_h", "cap_h")}))
        return 0
    if a.cmd == "inputs":
        res = check_inputs(Path(a.run), Path(a.teacher) if a.teacher else None)
        Path(a.run, "inputs_check.json").write_text(json.dumps(res, indent=1) + "\n")
        print("inputs", "ok" if res["ok"] else f"MISMATCH {res['mismatch']}")
        return 0 if res["ok"] else 1
    if a.cmd == "smoke-file":
        run = Path(a.run)
        rows = smoke_rows(_jsonl(run / "data" / "D100.jsonl"), _jsonl(run / "data" / "D10.jsonl"), a.n_long, a.n_rand)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        print(f"smoke rows {len(rows)}; longest prompt {max(prompt_chars(x) for x in rows)} chars")
        return 0
    if a.cmd == "oom":
        return 0 if is_oom(Path(a.log).read_text(errors="replace")) else 1
    if a.cmd == "setting":
        print(next_setting(a.name) if a.next else setting_flags(a.name))
        return 0
    if a.cmd == "used-h":
        print(f"{used_hours(Path(a.run)):.3f}")
        return 0
    if a.cmd == "cap":
        ok, msg = cap_check(Path(a.run), a.arms, a.cap_h)
        print(("go " if ok else "skip ") + msg)
        return 0 if ok else 3
    if a.cmd == "record":
        rec = record_setting(Path(a.run), a.arm, a.setting, a.attempts, a.sft_min, a.itype,
                             Path(a.log) if a.log else None, a.mix_sha)
        print(json.dumps({k: rec.get(k) for k in ("arm", "setting", "grad_accum", "rows_per_step", "sft_min",
                                                  "train_mix_ok")}))
        return 0
    if a.cmd == "arm-time":
        record_eval_time(Path(a.run), a.arm, a.eval_min)
        return 0
    if a.cmd == "anchor":
        res = anchor_check(Path(a.run), a.serve_mode)
        Path(a.run, "anchor_check.json").write_text(json.dumps(res, indent=1) + "\n")
        print("anchor", "ok" if res["ok"] else "FAILED",
              {k: res[k]["ok"] for k in ("v1_dev", "temperatures", "c_answers", "cost")})
        if res["ok"]:
            return 0
        only_c = res["v1_dev"]["ok"] and res["temperatures"]["ok"] and res["cost"]["ok"] and not res["c_answers"]["ok"]
        return 2 if only_c else 1
    if a.cmd == "answers":
        res = answers_complete(Path(a.run), a.arm)
        print(f"answered {res['answered']} / {res['questions']} questions, {res['errors']} error records"
              + ("" if res["ok"] else " (INCOMPLETE)"))
        return 0 if res["ok"] else 1
    if a.cmd == "cost":
        res = cost_check(Path(a.run))
        Path(a.run, "cost_check.json").write_text(json.dumps(res, indent=1) + "\n")
        print("cost", "ok" if res["ok"] else "PRICED OR NON-LOCAL RECORDS", res)
        return 0 if res["ok"] else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
