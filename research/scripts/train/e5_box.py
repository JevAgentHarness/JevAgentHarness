#!/usr/bin/env python3
"""Box-side helpers of scripts/aws/e5_small.sh, exploratory row E5 "small decision models" (docs/PREREG_A7_STAGE2.md,
Deviations 2026-10-06 "Exploratory row E5"; plan results/explore/small_models/e5_plan.md). CPU only: no GPU, no API
call, no key. Reuses E3's helpers (scripts/train/ablation_box.py) where E5 does the same thing.

    models      [--run A]                         the model slots in run order, one per line (a recorded swap applied)
    model       --arm X --field repo|revision     one field of a model slot
    host        --out H [--itype T] [--gpu NAME] [--cap-h 14] [--instance-id I]
    inputs      --run A                           D100 file + training-mix expectation + corpus copy + replay requests
    readout     --model DIR --out J               tokenizer / template checks of the read-out (CPU; E5 smoke)
    smoke-file  --run A --out F                   the memory smoke rows: the 48 longest D100 prompts + 48 random rows
    bad-loss    LOG                               exit 0 when LOG shows a non-finite training loss
    oom         LOG                               exit 0 when LOG shows a CUDA out-of-memory error
    setting     NAME | --next NAME                trainer flags of a memory setting / the next fallback
    used-h      --run A                           E5 GPU hours so far (finished stages + the running one)
    cap         --run A --arm X --cap-h 14 [--reserve-h R]   'go ...' (exit 0) or 'skip ...' (exit 3)
    record      --run A --arm X --setting S --attempts .. --sft-min M --itype T [--log L] [--mix-sha H] [--peak-mib P]
    arm-time    --run A --arm X --eval-min E
    cost        --run A                           every replay / calibration-replay record cost 0 and local
    answers     --run A --arm X                   every question of the corpus copy answered
    peak        CSV                               peak memory.used (MiB) of an nvidia-smi sample file
    spend       FILE...                           API $ over every record of the live run files (watchdog; all attempts)
    health      FILE... [--max-errors 0] [--max-s1-errors 0] [--max-retried 4]   live run health: errored episodes
                                                  (latest record per task; route / System-1 server / other) and retried
                                                  LLM calls; exit 4 when over a limit
    compact-bench --dir D --out O                 harness_final.py bench files without the answers (request ms and dp,
                                                  System-1 ms per step): what the analysis reads, a small fraction of the size
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "src"))

import ablation_box as BX  # noqa: E402

# Deviations 2026-10-06 (b): the models at their pinned revisions (survey.md section 5), in run order; the backup
# replaces e5_mc1 only if e5_mc1 fails its smoke (recorded in A/swap.json)
MODELS: dict[str, dict[str, Any]] = {
    "e5_q17": {"repo": "Qwen/Qwen3-1.7B", "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
               "params_b": 1.72, "nonemb_b": 1.41, "label": "Qwen3-1.7B"},
    "e5_q06": {"repo": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
               "params_b": 0.60, "nonemb_b": 0.44, "label": "Qwen3-0.6B"},
    "e5_mc1": {"repo": "openbmb/MiniCPM5-1B", "revision": "87179e5c1f455ef22e6223592d2d61351b525bfc",
               "params_b": 1.08, "nonemb_b": 0.68, "label": "MiniCPM5-1B"},
    "e5_gr1": {"repo": "ibm-granite/granite-4.0-1b", "revision": "6a7381ba1f54d684ff508d991aeb7dc580157103",
               "params_b": 1.63, "nonemb_b": 1.43, "label": "Granite-4.0-1B", "backup_for": "e5_mc1"},
}
ORDER = ["e5_q17", "e5_q06", "e5_mc1"]
BACKUP = {"e5_mc1": "e5_gr1"}
REF_4B = {"label": "JevAgentRL-4B (Qwen3-4B)", "params_b": 4.02, "nonemb_b": 3.63}
INPUTS = ROOT / "results" / "training" / "ablation_inputs.json"
E5_INPUTS = ROOT / "results" / "explore" / "e5_small" / "inputs.json"
DATA_ARM = "D100"
# priors of the cap projection: E3's measured A10G minutes for the 4B (results/training/ablation status, D25 on a
# g5.8xlarge: SFT 136 min for 4,126 rows incl. replay with checkpointing; evaluation + temperatures + C replay with
# the fast server 61 min per checkpoint), scaled by total parameters (the plan's conservative choice)
SFT_MIN_PER_ROW_4B_A10G_GC = (136 - 5) / 4126
EVAL_MIN_4B_A10G = 61.0
GC_FACTOR = 1.3
LOAD_MIN = 5.0
EVAL_FLOOR_MIN = 15.0
CAL_REPLAY_MIN = 10.0               # the calibration-state replay of (g): 2,719 questions at a small model's speed
L40S_TRAIN, L40S_EVAL = 2.9, 2.21   # E3's A10G / L40S factors (ablation_plan.md section 11.4)
RESERVE_H = 1.5                     # the latency stage (and the 4B's calibration replay) after the arms


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ slots
def slots(run: Path | None = None) -> list[str]:
    """The run order with a recorded smoke swap applied (A/swap.json: {"e5_mc1": "e5_gr1", ...})."""
    swap = {}
    if run is not None and (run / "swap.json").exists():
        swap = json.loads((run / "swap.json").read_text()).get("swap") or {}
    return [swap.get(a, a) for a in ORDER]


def ratio(arm: str) -> float:
    return float(MODELS[arm]["params_b"]) / float(REF_4B["params_b"])


def expected_mix_sha(inputs: Path = INPUTS) -> str:
    return json.loads(inputs.read_text())["train_mix"]["arms"][DATA_ARM]["sha256"]


def rows_with_replay(inputs: Path = INPUTS) -> int:
    return int(json.loads(inputs.read_text())["arms"][DATA_ARM]["rows_with_replay"])


# ------------------------------------------------------------------ host
def host_profile(itype: str, gpu: str, cap_h: float, instance_id: str = "") -> dict[str, Any]:
    h = BX.host_profile(itype, gpu, cap_h, "fast")
    h.update({"instance_id": instance_id, "row": "E5", "reserve_h": RESERVE_H})
    return h


# ------------------------------------------------------------------ inputs
def check_inputs(run: Path, inputs: Path = INPUTS, e5_inputs: Path = E5_INPUTS) -> dict[str, Any]:
    want = json.loads(inputs.read_text())
    res: dict[str, Any] = {"mismatch": []}
    p = run / "data" / f"{DATA_ARM}.jsonl"
    got = _sha(p) if p.exists() else None
    res["D100"] = {"sha256": got, "expected": want["arms"][DATA_ARM]["sha256"]}
    if got != want["arms"][DATA_ARM]["sha256"]:
        res["mismatch"].append("D100")
    for f in ("corpus.json", "occurrences.jsonl.gz", "questions.jsonl.gz"):
        a, b = ROOT / "results" / "stage2" / "s23" / f, run / "s23" / f
        if not b.exists() or _sha(a) != _sha(b):
            res["mismatch"].append(f"s23/{f}")
    e5 = json.loads(e5_inputs.read_text())
    cr = run / "inputs" / "cal_requests.jsonl.gz"
    res["cal_requests"] = {"sha256": _sha(cr) if cr.exists() else None, "expected": e5["cal_requests"]["sha256"]}
    if res["cal_requests"]["sha256"] != e5["cal_requests"]["sha256"]:
        res["mismatch"].append("cal_requests")
    res["ok"] = not res["mismatch"]
    return res


# ------------------------------------------------------------------ read-out preflight on the box (tokenizer only)
def readout(model_dir: str) -> dict[str, Any]:
    """The survey's preflight on the downloaded snapshot: 52 single-token option codes, the code right after the
    rendered prompt is the bare code (no merge), no date stamp in the rendered prompt."""
    import re

    from transformers import AutoTokenizer

    from jevharness.training.data import render
    from jevharness.training.decision_model import chat_text, code_token_ids

    out: dict[str, Any] = {"model": model_dir}
    try:
        tok = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:  # noqa: BLE001
        return {**out, "ok": False, "error": f"tokenizer: {type(e).__name__}: {str(e)[:200]}"}
    try:
        codes = code_token_ids(tok)
    except Exception as e:  # noqa: BLE001
        return {**out, "ok": False, "error": f"codes: {str(e)[:200]}"}
    prompt = render({"state": "TASK: probe\n\nCURRENT OBSERVATION:\nok", "question": "Which tool?",
                     "options": ["find_user", "ask_user", "finish"]})[0]
    txt = chat_text(tok, prompt)
    base = tok(txt, add_special_tokens=False).input_ids
    bare = True
    for c, cid in zip("AZaz", [codes[0], codes[25], codes[26], codes[51]]):
        ids = tok(txt + c, add_special_tokens=False).input_ids
        bare &= len(ids) == len(base) + 1 and ids[-1] == cid and ids[:-1] == base
    date = bool(re.search(r"Today Date|Cutting Knowledge Date|Knowledge Cutoff", txt))
    out.update({"codes_single_token": True, "code_after_prompt_is_bare_code": bool(bare), "date_in_render": date,
                "render_tail": txt[-60:], "pad_token": tok.pad_token, "eos_token": tok.eos_token,
                "ok": bool(bare) and not date})
    return out


def smoke_file(run: Path, out: Path) -> int:
    """E3's smoke rows (ablation_box.smoke_rows) from D100 alone: the longest prompts and random other rows."""
    d100 = BX._jsonl(run / "data" / f"{DATA_ARM}.jsonl")
    rows = BX.smoke_rows(d100, d100)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    return len(rows)


BAD_LOSS_RE = re.compile(r"""['"](?:loss|train_loss)['"]\s*:\s*'?(?:nan|inf|-inf)\b""", re.IGNORECASE)


def bad_loss(text: str) -> bool:
    return bool(BAD_LOSS_RE.search(text))


# ------------------------------------------------------------------ cap projection
def estimate_min(run: Path, arm: str, inputs: Path = INPUTS) -> float:
    """Projected GPU minutes of one model: SFT + evaluation + C replay + calibration replay. Priors (E3's 4B on an
    A10G scaled by total parameters; L40S faster by E3's factors) until a model has finished on this run; then the
    measured minutes per row and per unit of parameter ratio."""
    host = json.loads((run / "host.json").read_text()) if (run / "host.json").exists() else host_profile("", "", 14)
    cls = host.get("gpu_class", "A10G")
    tf, ef = (1 / L40S_TRAIN, 1 / L40S_EVAL) if cls == "L40S" else (1.0, 1.0)
    setting = ((run / "mem").exists() and (run / "mem" / f"{arm}.txt").exists()
               and (run / "mem" / f"{arm}.txt").read_text().strip()) or "nogc_3072"
    done = []
    if (run / "arm_times.jsonl").exists():
        done = [json.loads(x) for x in (run / "arm_times.jsonl").read_text().splitlines() if x.strip()]
    rows = rows_with_replay(inputs)
    r = ratio(arm)
    trained = [d for d in done if d.get("rows") and d.get("sft_min") is not None and d.get("ratio")]
    if trained:
        rate = sum(max(0.0, d["sft_min"] - LOAD_MIN) / d["ratio"] for d in trained) / sum(d["rows"] for d in trained)
    else:
        rate = SFT_MIN_PER_ROW_4B_A10G_GC * tf / (1.0 if setting.startswith("gc") else GC_FACTOR)
    evals = [d for d in done if d.get("eval_min") is not None and d.get("ratio")]
    if evals:
        ev = max(EVAL_FLOOR_MIN, sum(d["eval_min"] / d["ratio"] for d in evals) / len(evals) * r)
    else:
        ev = max(EVAL_FLOOR_MIN, EVAL_MIN_4B_A10G * ef * r) + CAL_REPLAY_MIN
    return rate * rows * r + LOAD_MIN + ev


def cap_check(run: Path, arm: str, cap_h: float, reserve_h: float = RESERVE_H) -> tuple[bool, str]:
    if (run / "done" / arm).exists():
        return True, f"{arm}: done earlier"
    used = BX.used_hours(run)
    est = estimate_min(run, arm) / 60
    msg = (f"model {arm}: used {used:.2f} h + projected {est:.2f} h + reserved {reserve_h:g} h (latency) "
           f"vs cap {cap_h:g} h")
    return used + est + reserve_h <= cap_h, msg


def record_setting(run: Path, arm: str, setting: str, attempts: str, sft_min: float, itype: str,
                   log: Path | None = None, mix_sha: str = "", peak_mib: str = "",
                   inputs: Path = INPUTS) -> dict[str, Any]:
    tok = next(t for n, t, _ in BX.SETTINGS if n == setting)
    gc = next(g for n, _, g in BX.SETTINGS if n == setting)
    m = MODELS[arm]
    rec: dict[str, Any] = {"arm": arm, "repo": m["repo"], "revision": m["revision"], "data": DATA_ARM,
                           "trainer_seed": 0, "setting": setting, "max_batch_tokens": tok,
                           "gradient_checkpointing": gc, "eff_batch_rows": 64,
                           "attempts": [a for a in attempts.split(",") if a], "sft_min": sft_min,
                           "instance_type": itype, "rows_with_replay": rows_with_replay(inputs),
                           "peak_mem_mib": float(peak_mib) if str(peak_mib).strip() else None}
    if mix_sha:
        want = expected_mix_sha(inputs)
        rec.update({"train_mix_sha256": mix_sha, "train_mix_sha256_expected": want, "train_mix_ok": mix_sha == want})
    if log is not None and log.exists():
        mm = BX.GA_RE.search(log.read_text(errors="replace"))
        if mm:
            rec.update({"micro_batches_per_epoch": int(mm.group(2)), "rows_per_micro_batch": float(mm.group(3)),
                        "grad_accum": int(mm.group(4)), "rows_per_step": round(int(mm.group(4)) * float(mm.group(3)), 1)})
    (run / "settings").mkdir(parents=True, exist_ok=True)
    (run / "settings" / f"{arm}.json").write_text(json.dumps(rec, indent=1) + "\n")
    with open(run / "arm_times.jsonl", "a") as f:
        f.write(json.dumps({"arm": arm, "rows": rec["rows_with_replay"], "sft_min": sft_min, "setting": setting,
                            "ratio": ratio(arm)}) + "\n")
    return rec


def record_eval_time(run: Path, arm: str, eval_min: float) -> None:
    with open(run / "arm_times.jsonl", "a") as f:
        f.write(json.dumps({"arm": arm, "eval_min": eval_min, "ratio": ratio(arm)}) + "\n")


# ------------------------------------------------------------------ cost / completeness
def cost_check(run: Path) -> dict[str, Any]:
    recs = priced = 0
    nonlocal_backends: list[str] = []
    files = sorted((run / "s23").glob("answers_e5_*.jsonl")) + sorted((run / "cal").glob("answers_*.jsonl"))
    for p in files:
        for x in p.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                recs += 1
                if r.get("cost_usd") is None or float(r["cost_usd"]) != 0.0:
                    priced += 1
    for p in sorted((run / "s23").glob("replay_e5_*.json")):
        url = str((json.loads(p.read_text()).get("backend") or {}).get("base_url") or "")
        if not url.startswith("http://127.0.0.1:"):
            nonlocal_backends.append(p.name)
    return {"records": recs, "files": len(files), "priced": priced, "nonlocal_backends": nonlocal_backends,
            "ok": priced == 0 and not nonlocal_backends}


def answers_complete(run: Path, arm: str) -> dict[str, Any]:
    with gzip.open(run / "s23" / "questions.jsonl.gz", "rt") as f:
        qs = {json.loads(x)["qhash"] for x in f if x.strip()}
    got = BX.latest_answers(run / "s23" / f"answers_e5_{arm}.jsonl")
    return {"questions": len(qs), "answered": len(qs & set(got)),
            "errors": sum(1 for q in qs if (got.get(q) or {}).get("error")), "ok": qs <= set(got)}


def peak_mib(csv: Path) -> float | None:
    """Peak memory.used of an nvidia-smi --query-gpu=[timestamp,utilization.gpu,]memory.used[,memory.total] sample
    file (the column is found by position: 'memory.used,memory.total' -> first, 'timestamp,util,memory.used' -> last)."""
    if not csv.exists():
        return None
    vals = []
    for line in csv.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        cell = parts[-1] if len(parts) == 3 else parts[0]
        try:
            vals.append(float(cell))
        except ValueError:
            continue
    return max(vals) if vals else None


def spend(files: list[Path]) -> dict[str, Any]:
    """API $ over EVERY record line of the given run files (every attempt is billed, so errored and superseded
    records count too): the live watchdog's sum."""
    usd = 0.0
    n = 0
    for p in files:
        if not p.exists():
            continue
        for x in p.read_text(errors="replace").splitlines():
            if not x.strip():
                continue
            try:
                r = json.loads(x)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and "task_id" in r:
                n += 1
                usd += float(((r.get("usage") or {}).get("total") or {}).get("cost_usd") or 0.0)
    return {"records": n, "usd": round(usd, 4)}


def health(files: list[Path]) -> dict[str, Any]:
    """The live row's route health (Deviations 2026-10-06 ~18:00 UTC: pause on an errored episode or repeated
    retried LLM calls): errored episodes by the latest record of each task, and LLM (System-2) calls that needed more
    than one attempt or waited for a backoff, over every record line (every attempt counts)."""
    errored = tasks = calls = retried = 0
    kinds = {"route": 0, "s1": 0, "other": 0}
    for p in files:
        if not p.exists():
            continue
        last: dict[str, Any] = {}
        for x in p.read_text(errors="replace").splitlines():
            try:
                r = json.loads(x)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or "task_id" not in r:
                continue
            last[str(r["task_id"])] = r
            for c in r.get("calls") or []:
                if c.get("component") == "s2":
                    calls += 1
                    retried += int((c.get("attempts") or 1) > 1 or float(c.get("retry_wait_ms") or 0) > 0)
        tasks += len(last)
        for r in last.values():
            if r.get("error"):
                errored += 1
                kinds[error_kind(str(r["error"]))] += 1
    return {"tasks": tasks, "errored": errored, "errored_route": kinds["route"], "errored_s1": kinds["s1"],
            "errored_other": kinds["other"], "s2_calls": calls, "s2_retried": retried}


ROUTE_ERRORS = ("InternalServerError", "APIError", "APIConnectionError", "APITimeoutError", "RateLimitError",
                "ServiceUnavailable", "APIStatusError", "BadRequestError", "openrouter")


def error_kind(err: str) -> str:
    """An episode error's source: the System-1 server (JevAPIError), the LLM route (the OpenAI client's errors), other."""
    head = err.strip().splitlines()[0] if err.strip() else ""
    if head.startswith("JevAPIError") or "Jev API error" in head:
        return "s1"
    if any(x in head for x in ROUTE_ERRORS):
        return "route"
    return "other"


def compact_bench(d: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(d.glob("bench_*.json")):
        r = json.loads(p.read_text())
        keep = {k: r.get(k) for k in ("label", "path", "concurrency", "think", "episodes", "span_s", "replicas", "time")}
        keep["requests"] = [{"dp": x.get("dp"), "ms": x.get("ms"), "server_ms": x.get("server_ms"),
                             "n_questions": x.get("n_questions")} for x in r.get("requests") or []]
        keep["steps"] = [{"s1_ms": x.get("s1_ms"), "n_requests": x.get("n_requests")} for x in r.get("steps") or []]
        keep["episodes_rows"] = r.get("episodes_rows")
        (out / p.name).write_text(json.dumps(keep))
        n += 1
    return n


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("models")
    m.add_argument("--run", default=None)
    mo = sub.add_parser("model")
    mo.add_argument("--arm", required=True)
    mo.add_argument("--field", required=True)
    h = sub.add_parser("host")
    h.add_argument("--out", required=True)
    h.add_argument("--itype", default="")
    h.add_argument("--gpu", default="")
    h.add_argument("--cap-h", type=float, default=14.0)
    h.add_argument("--instance-id", default="")
    i = sub.add_parser("inputs")
    i.add_argument("--run", required=True)
    r = sub.add_parser("readout")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    sf = sub.add_parser("smoke-file")
    sf.add_argument("--run", required=True)
    sf.add_argument("--out", required=True)
    bl = sub.add_parser("bad-loss")
    bl.add_argument("log")
    o = sub.add_parser("oom")
    o.add_argument("log")
    st = sub.add_parser("setting")
    st.add_argument("name")
    st.add_argument("--next", action="store_true")
    u = sub.add_parser("used-h")
    u.add_argument("--run", required=True)
    c = sub.add_parser("cap")
    c.add_argument("--run", required=True)
    c.add_argument("--arm", required=True)
    c.add_argument("--cap-h", type=float, required=True)
    c.add_argument("--reserve-h", type=float, default=RESERVE_H)
    rc = sub.add_parser("record")
    rc.add_argument("--run", required=True)
    rc.add_argument("--arm", required=True)
    rc.add_argument("--setting", required=True)
    rc.add_argument("--attempts", default="")
    rc.add_argument("--sft-min", type=float, required=True)
    rc.add_argument("--itype", default="")
    rc.add_argument("--log", default=None)
    rc.add_argument("--mix-sha", default="")
    rc.add_argument("--peak-mib", default="")
    e = sub.add_parser("arm-time")
    e.add_argument("--run", required=True)
    e.add_argument("--arm", required=True)
    e.add_argument("--eval-min", type=float, required=True)
    co = sub.add_parser("cost")
    co.add_argument("--run", required=True)
    aw = sub.add_parser("answers")
    aw.add_argument("--run", required=True)
    aw.add_argument("--arm", required=True)
    pk = sub.add_parser("peak")
    pk.add_argument("csv")
    sp = sub.add_parser("spend")
    sp.add_argument("files", nargs="*")
    he = sub.add_parser("health")
    he.add_argument("files", nargs="*")
    he.add_argument("--max-errors", type=int, default=0, help="route + other errored episodes")
    he.add_argument("--max-s1-errors", type=int, default=0, help="System-1 server errored episodes")
    he.add_argument("--max-retried", type=int, default=4)
    cb = sub.add_parser("compact-bench")
    cb.add_argument("--dir", required=True)
    cb.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "models":
        print("\n".join(slots(Path(a.run) if a.run else None)))
        return 0
    if a.cmd == "model":
        print(MODELS[a.arm][a.field])
        return 0
    if a.cmd == "host":
        prof = host_profile(a.itype, a.gpu, a.cap_h, a.instance_id)
        Path(a.out).write_text(json.dumps(prof, indent=1) + "\n")
        print(json.dumps({k: prof[k] for k in ("instance_type", "instance_id", "gpu_class", "usd_per_h", "cap_h")}))
        return 0
    if a.cmd == "inputs":
        res = check_inputs(Path(a.run))
        Path(a.run, "inputs_check.json").write_text(json.dumps(res, indent=1) + "\n")
        print("inputs", "ok" if res["ok"] else f"MISMATCH {res['mismatch']}")
        return 0 if res["ok"] else 1
    if a.cmd == "readout":
        res = readout(a.model)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print("readout", "ok" if res["ok"] else f"FAILED {res.get('error') or res}")
        return 0 if res["ok"] else 1
    if a.cmd == "smoke-file":
        print(f"smoke rows {smoke_file(Path(a.run), Path(a.out))}")
        return 0
    if a.cmd == "bad-loss":
        return 0 if bad_loss(Path(a.log).read_text(errors="replace")) else 1
    if a.cmd == "oom":
        return 0 if BX.is_oom(Path(a.log).read_text(errors="replace")) else 1
    if a.cmd == "setting":
        print(BX.next_setting(a.name) if a.next else BX.setting_flags(a.name))
        return 0
    if a.cmd == "used-h":
        print(f"{BX.used_hours(Path(a.run)):.3f}")
        return 0
    if a.cmd == "cap":
        ok, msg = cap_check(Path(a.run), a.arm, a.cap_h, a.reserve_h)
        print(("go " if ok else "skip ") + msg)
        return 0 if ok else 3
    if a.cmd == "record":
        rec = record_setting(Path(a.run), a.arm, a.setting, a.attempts, a.sft_min, a.itype,
                             Path(a.log) if a.log else None, a.mix_sha, a.peak_mib)
        print(json.dumps({k: rec.get(k) for k in ("arm", "setting", "grad_accum", "sft_min", "train_mix_ok",
                                                  "peak_mem_mib")}))
        return 0
    if a.cmd == "arm-time":
        record_eval_time(Path(a.run), a.arm, a.eval_min)
        return 0
    if a.cmd == "cost":
        res = cost_check(Path(a.run))
        Path(a.run, "cost_check.json").write_text(json.dumps(res, indent=1) + "\n")
        print("cost", "ok" if res["ok"] else "PRICED OR NON-LOCAL RECORDS", res)
        return 0 if res["ok"] else 1
    if a.cmd == "answers":
        res = answers_complete(Path(a.run), a.arm)
        print(f"answered {res['answered']} / {res['questions']} questions, {res['errors']} error records"
              + ("" if res["ok"] else " (INCOMPLETE)"))
        return 0 if res["ok"] else 1
    if a.cmd == "peak":
        v = peak_mib(Path(a.csv))
        print("" if v is None else f"{v:.0f}")
        return 0
    if a.cmd == "health":
        h = health([Path(f) for f in a.files])
        bad = (h["errored_route"] + h["errored_other"] > a.max_errors or h["errored_s1"] > a.max_s1_errors
               or h["s2_retried"] > a.max_retried)
        print(json.dumps({**h, "ok": not bad}))
        return 4 if bad else 0
    if a.cmd == "compact-bench":
        print(f"compacted {compact_bench(Path(a.dir), Path(a.out))} bench files")
        return 0
    if a.cmd == "spend":
        print(json.dumps(spend([Path(f) for f in a.files])))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
