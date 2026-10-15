#!/usr/bin/env python3
r"""E30 readout (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 "E30: every open decision model at its cheapest common
deployment"; exploratory, descriptive, post hoc re-measurement): what one decision costs for each open decision model
of Figure 4a when every model is served the same, cheapest way (one g6.xlarge, vLLM in process, saturated).

    python scripts/explore_e30_cost.py fetch        # s3://.../e30_cost/box -> results/explore/e30_cost/box (records)
                                                    #   and the timing files with every answer -> runs/e30_cost/
    python scripts/explore_e30_cost.py summarize    # summary.{md,json}, preds/*.jsonl.gz, paper/generated/e30_cost.tex

Host side: ``scripts/e30_bench.py`` (driver ``scripts/aws/e30_cost.sh``). Each timing file holds, per decision, the
raw code logits of each read (or the finished wire answer); here they become the harness's prediction exactly as the
reference rows were parsed (serve_jev's labels for our trained models, the vendor's letter distribution through
``answer_from_wire``, Intern's temperature scaling and tie-break through ``answer_from_intern``), and a decision is
correct iff its prediction is in its gold set (E1). Cost per 1,000 decisions = the hourly price / (3,600 x saturated
decisions per second) x 1,000; ``lg`` = log10(cost per 1,000 / 1,000), Figure 4a's x coordinate. The plotted point of
a model is its cheapest precision whose accuracy is at least its reference minus 1 point (registration (e))."""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/explore/e30_cost"
BOX = OUT / "box"
RAW = ROOT / "runs/e30_cost"              # timing files with every answer (not versioned; kept on S3)
RUNS = ROOT / "runs"
TEX = ROOT / "paper/generated/e30_cost.tex"
GEN = ROOT / "paper/generated"
BUCKET = "<bucket>"
S3 = f"s3://{BUCKET}/e30_cost/box"
S3_L40S = f"s3://{BUCKET}/e30_cost_l40s/box"
BOX_L40S = OUT / "box_l40s"
INSTANCE, GPU = "g6.xlarge", "NVIDIA L4 24 GB"
PRICE_H = 0.8048                          # AWS price list, us-east-1 on demand, Linux, effective 2026-10-01 (registration (c))
PRICE_SOURCE = "AWS price list (us-east-1, on demand, Linux, shared tenancy, effective 2026-10-01)"
FALLBACK_PRICE_H = {"g6.2xlarge": 0.9776}
MARGIN_PT = 1.0                           # registration (e)
AGREE_MIN = 0.95                          # a plotted row whose answers agree less with its reference record is flagged
N_E1 = 3991
LABEL = ("exploratory (E30), descriptive; post hoc re-measurement; PREREG_A7_STAGE2 Deviations 2026-10-09 'E30: every "
         "open decision model at its cheapest common deployment'; outside every Holm table")

# slug -> (label, family, reference key, reference record file under runs/ (E9's row of the model))
MODELS: dict[str, tuple[str, str, str, str]] = {
    "e12": ("AgentDecision-4B", "code", "e9/e12/acc", "e1_bfcl_when2call_e12/s1_e12_mixed.jsonl"),
    "rl": ("Frozen GRPO 4B", "code", "e9/rl/acc", "e1_bfcl_when2call_rl/s1_jevagent_rl.jsonl"),
    "e12c": ("StartLux-4B + our recipe", "code", "e9/e12c/acc", "e1_bfcl_when2call_e12c/s1_e12c_sl4b.jsonl"),
    "e28": ("AgentDecision-0.8B (StartLux-0.8B + our recipe, E28)", "code", "e28/l1/acc", "e1_bfcl_when2call_e28/s1_e28.jsonl"),
    "sl0.8b": ("StartLux-Decision-0.8B", "startlux", "e9/sl0.8b/acc",
               "e1_bfcl_when2call_startlux_ladder/s1_startlux_0_8b.jsonl"),
    "sl2b": ("StartLux-Decision-2B", "startlux", "e9/sl2b/acc", "e1_bfcl_when2call_startlux_ladder/s1_startlux_2b.jsonl"),
    "sl4b": ("StartLux-Decision-4B", "startlux", "e9/sl4b/acc", "e1_bfcl_when2call_startlux/s1_startlux_4b.jsonl"),
    "sl9b": ("StartLux-Decision-9B", "startlux", "e9/sl9b/acc", "e1_bfcl_when2call_startlux_ladder/s1_startlux_9b.jsonl"),
    "sl27b": ("StartLux-Decision-27B", "startlux", "e9/sl27b/acc", "e1_bfcl_when2call_startlux/s1_startlux_27b.jsonl"),
    "intern": ("Intern-Decision-4B", "intern", "e9/intern/acc", "e1_bfcl_when2call_startlux/s1_intern_4b.jsonl"),
    "e27sp": ("Qwen3.5-0.8B variant (E27, scoring head)", "wire", "e9/e27sp/acc",
              "e1_bfcl_when2call_e27/s1_e27sp.jsonl"),
    # E30 amendment "OpenJev-27B through its official 4-bit GGUF": the reference record is E15's (results/, gzip)
    "openjevhf": ("OpenJev-27B", "openjev", "e9/openjevhf/acc", "results/explore/e15_baselines/l1/s1_openjevhf.jsonl.gz"),
    # E30 amendment "E15's open baselines on the same basis"
    "decider": ("decider-4b", "decider", "e9/decider/acc", "results/explore/e15_baselines/l1/s1_decider.jsonl.gz"),
    # amendment "E30: JevK5 at the weights E15 loaded": key slug jevk5 = E15's weights (commit c4f7fdb3, v0.3; runs jevk5m_*),
    # the v0.2 tag's rows (runs jevk5_*) stay as the descriptive row jevk5v02, never drawn
    "jevk5": ("JevK5 (v0.3, E15's weights)", "jevk5", "e9/jevk5/acc", "results/explore/e15_baselines/l1/s1_jevk5.jsonl.gz"),
    "jevk5v02": ("JevK5 v0.2 (the registered tag, not E15's weights)", "jevk5", "e9/jevk5/acc",
                 "results/explore/e15_baselines/l1/s1_jevk5.jsonl.gz"),
    "clefflash": ("Clef-Flash 9B (scoring head)", "wire", "e9/clefflash/acc",
                  "results/explore/e15_baselines/l1/s1_clefflash.jsonl.gz"),
    "imajev": ("Imajev-4B", "imajev", "e9/imajev/acc", "results/explore/e15_baselines/l1/s1_imajev.jsonl.gz"),
    "openjev": ("OpenJev (DiffusionGemma-26B-A4B)", "wire", "e9/openjev/acc",
                "results/explore/e15_baselines/l1/s1_openjev.jsonl.gz"),
}
# runs/<file slug>_<precision>.json.gz when a key slug reads another stage's files; the E9 slug of its reference keys
RUN_SLUG = {"jevk5": "jevk5m", "jevk5v02": "jevk5"}
E9_SLUG = {"jevk5v02": "jevk5"}
DESCRIPTIVE_ONLY = ("jevk5v02",)   # rows outside the summary counts (e30/meta/*), the reading and Figure 4a
# a model measured on another instance type, priced at that type's on-demand price (amendment (c) 5)
PRICE_BY_SLUG = {"openjev": 1.861}
INSTANCE_BY_SLUG = {"openjev": ("g6e.xlarge", "NVIDIA L40S 48 GB")}
# Figure 4a's drawn points (the coordinating session): figure coordinate name -> slug; open models at their E30 point,
# the API / LLM points at their E9 keys
FIG_OPEN = {"sl08b": "sl0.8b", "sl2b": "sl2b", "sl4b": "sl4b", "sl9b": "sl9b", "sl27b": "sl27b", "intern": "intern",
            "ours": "e12", "ours08": "e28", "openjev": "openjev", "jevk5": "jevk5", "decider": "decider",
            "clefflash": "clefflash", "openjevhf": "openjevhf", "imajev": "imajev"}
FIG_API = ("jev", "oaidec", "flash", "lite", "flashfc")
FIG_MIN_ACC = 68.0
REF_S3 = {"e28": f"s3://{BUCKET}/e28_sl08b/runs/e1_bfcl_when2call_e28/s1_e28.jsonl"}   # read only (E28's own record)
# precision -> (label for tables, TeX label, engine); runs/<slug>_<file stem>.json.gz
PRECISIONS: dict[str, tuple[str, str, str]] = {
    "bf16": ("BF16", "BF16", "vLLM"),
    "fp8": ("FP8", "FP8", "vLLM"),
    "q4_k_m": ("Q4_K_M (GGUF)", r"Q4\_K\_M", "llama.cpp"),
}
RUN_FILES: dict[str, list[str]] = {"q4_k_m": ["q4_k_m_p8", "q4_k_m_p16"]}   # llama.cpp: P = 8 (answers, c = 1), P = 16
# reasons for a model without a measured row (amendment "E15's open baselines" (c) 4, 5); shown only when it has none
NOT_MEASURED = {
    "imajev": "Imajev-4B: its read-out is a trained 255-row code readout over four rotated presentations of the options; no "
              "vLLM port was ready by the amendment's 00:30 UTC limit (the remaining host-hours went to the other rows), and "
              "its own server (about 2.9 s per decision as E15 logged it) does not fit the cap",
    "clefflash": "Clef-Flash 9B (scoring head): not run for time. Its first attempt (21:53 UTC) stopped at load before any "
                 "answer (the v2 venv has no image processor class for the checkpoint); the corrected stage (a new venv for the "
                 "card's function, then about an hour of one-at-a-time 9B passes) would have ended after the time set for "
                 "E30's end (about 1.5 h from 23:45 UTC), so the chain ended after JevK5 (execution as run)",
    "openjev": "OpenJev (DiffusionGemma-26B-A4B): its masked-slot read of a diffusion language model is served only by its own "
               "image, its FP8 weights (27.2 GB) do not fit 24 GB and no 4-bit file it loads exists; the g6e.xlarge of the "
               "amendment's (c) 5 did not run it",
}
OUTSIDE = ("Clef 27B (withdrawn)", "Bosun", "CLM", "Laya", "Verdict", "E14's 0.8B heads (all under 70 %)")
# the serving engine named in the keys and tables for rows that are not on vLLM / llama.cpp
ENGINE_BY_SLUG = {"e27sp": "serve\\_jev (scoring head)", "clefflash": "the card's systemone() (scoring head)",
                  "openjev": "its own image (g6e.xlarge)"}
FAMILIES = {"toolchoice": ("tool_select", "bfcl"), "args": ("tool_params", "bfcl"), "w2c": ("tool_select", "when2call")}


# ================================================================================================ io
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def read_json(path: Path) -> Any:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(Path(path).read_text())


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def generated_keys(gen: Path = GEN) -> dict[str, str]:
    """Every ``\\defres{key}{value}`` of paper/generated/*.tex (this readout's own file excluded)."""
    pat = re.compile(r"^\\defres\{([^}]*)\}\{(.*)\}\s*$")
    out: dict[str, str] = {}
    for p in sorted(gen.glob("*.tex")):
        if p.name == TEX.name:
            continue
        for line in p.read_text().splitlines():
            m = pat.match(line)
            if m:
                out.setdefault(m.group(1), m.group(2))
    return out


def ref_float(keys: Mapping[str, str], key: str) -> float | None:
    v = keys.get(key)
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


# ================================================================================================ predictions
def softmax(z: Sequence[float], t: float = 1.0) -> list[float]:
    m = max(z)
    e = [math.exp((v - m) / t) for v in z]
    s = sum(e)
    return [v / s for v in e]


def argmax(xs: Sequence[float]) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


def predict(family: str, question: Any, ans: Mapping[str, Any]) -> str:
    """One decision's prediction, parsed as the reference rows were (registration (d))."""
    from jevharness.bench.suites.static import norm
    from jevharness.decision.intern_decision import answer_from_intern, scale_probabilities
    from jevharness.decision.intern_decision import argmax as intern_argmax
    from jevharness.decision.jev import answer_from_wire

    if "pred" in ans:                                   # the wire row: the harness's own parse on the host
        return str(ans["pred"])
    if "answer" in ans:                                 # a finished wire answer (StartLux wide choice / one option)
        return norm(answer_from_wire(question, dict(ans["answer"])).value)
    meta = ans.get("meta") or {}
    z = ans["logits"][0]
    if family == "code":                                # serve_jev: the label of the most likely option code
        return meta["labels_norm"][argmax(z)]
    if family in ("startlux", "jevk5"):                 # the vendor's _probs and _answer (JevK5: jevk5.prompt.answer)
        p = softmax(z, float(meta.get("T") or 1.0))
        dist = dict(zip(meta["opt_ids"], p))
        best = max(dist, key=dist.get)
        wire = {"type": "choice", "choice": best, "confidence": dist[best], "probabilities": dist}
        return norm(answer_from_wire(question, wire).value)
    if family in ("openjev", "decider"):                # softmax at the temperature, rounded probabilities (the helper /
                                                        # decider.systemone.format_answer), the winner before rounding
        p = softmax(z, float(meta.get("T") or 1.0))
        opts = meta["opt_ids"]
        choice = opts[max(range(len(p)), key=lambda i: p[i])]          # the helper picks the winner before rounding
        probs = {k: round(v, int(meta.get("round", 4))) for k, v in zip(opts, p)}
        wire = {"type": "choice", "choice": choice, "probabilities": probs, "confidence": probs[choice]}
        return norm(answer_from_wire(question, wire).value)
    if family == "intern":                              # read_fields: softmax, temperature, the reference tie-break
        raw = dict(zip(meta["values"], softmax(z)))
        try:
            probs = scale_probabilities(raw, float(meta["T"]))
        except ArithmeticError:
            probs = raw
        best = intern_argmax(probs)
        a = {"type": "choice", "probabilities": probs, "confidence": probs[best], "decision": best, "choice": best}
        return norm(answer_from_intern(question, a).value)
    raise ValueError(f"unknown family {family}")


def family_of(item: Any) -> str | None:
    key = (item.dp, str(item.source).split(":")[0])
    return next((f for f, v in FAMILIES.items() if v == key), None)


# ================================================================================================ cost and the plotted point
def usd_per_1000(price_h: float, dps: float) -> float:
    """Registration (f): the hourly price / (3,600 x decisions per second) x 1,000."""
    return price_h / (3600.0 * dps) * 1000.0


def lg(usdk: float) -> float:
    """Figure 4a's coordinate: log10 of the dollars per decision."""
    return math.log10(usdk / 1000.0)


def plotted(rows: Mapping[str, Mapping[str, Any]], ref_pct: float | None) -> tuple[str | None, str]:
    """(precision, flag): the cheapest precision whose accuracy (%) is at least the reference minus MARGIN_PT; else the
    most accurate one ("below"). Without a reference, the BF16 accuracy is the reference ("no reference")."""
    ok = {p: r for p, r in rows.items() if r.get("acc_pct") is not None and r.get("usdk") is not None}
    if not ok:
        return None, "not measured"
    flag = ""
    if ref_pct is None:
        ref_pct = (ok.get("bf16") or max(ok.values(), key=lambda r: r["acc_pct"]))["acc_pct"]
        flag = "no reference"
    qual = [p for p, r in ok.items() if r["acc_pct"] >= ref_pct - MARGIN_PT - 1e-9]
    if qual:
        return min(qual, key=lambda p: (ok[p]["usdk"], -ok[p]["acc_pct"])), flag
    return max(ok, key=lambda p: (ok[p]["acc_pct"], -ok[p]["usdk"])), (flag + "; " if flag else "") + "below"


# ================================================================================================ scoring
def load_items() -> dict[str, Any]:
    import yaml

    from jevharness.bench.decision_eval import load_population

    items = load_population(yaml.safe_load((ROOT / "configs/decision/bfcl_when2call_rl.yaml").read_text()))
    return {it.id: it for it in items}


def load_ref(slug: str) -> dict[str, dict[str, Any]] | None:
    f = MODELS[slug][3]
    p = ROOT / f if f.startswith("results/") else RUNS / f
    if not p.exists() and slug in REF_S3:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["aws", "--region", "us-east-1", "s3", "cp", REF_S3[slug], str(p), "--only-show-errors"],
                           check=True, capture_output=True, timeout=300,
                           env={k: v for k, v in __import__("os").environ.items()
                                if k not in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")} | {"AWS_PAGER": ""})
        except (OSError, subprocess.SubprocessError):
            return None
    if not p.exists():
        return None
    return {r["id"]: r for r in read_jsonl(p)}


def score_run(slug: str, run: Mapping[str, Any], items: Mapping[str, Any],
              ref: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    fam = MODELS[slug][1]
    answers = run.get("answers") or {}
    preds: dict[str, str] = {}
    for i, it in items.items():
        if i in answers:
            preds[i] = predict(fam, it.question, answers[i])
    correct = {i: p in items[i].gold for i, p in preds.items()}
    n = len(items)
    k = sum(correct.values())
    fams: dict[str, Any] = {}
    for f in FAMILIES:
        ids = [i for i, it in items.items() if family_of(it) == f]
        if ids:
            fams[f] = sum(1 for i in ids if correct.get(i)) / len(ids)
    out: dict[str, Any] = {"n": n, "answered": len(preds), "missing": n - len(preds), "acc": k / n,
                           "acc_pct": 100.0 * k / n, "families": fams}
    if ref:
        common = [i for i in preds if i in ref and ref[i].get("pred") is not None]
        out["agree_ref"] = sum(1 for i in common if preds[i] == ref[i]["pred"]) / len(common) if common else None
        out["ref_acc_pct_records"] = 100.0 * sum(bool(r.get("correct")) and not r.get("error") for r in ref.values()) / len(ref)
    out["_preds"] = preds
    out["_correct"] = correct
    return out


def run_record(slug: str, prec: str, raw: Path = RAW) -> list[dict[str, Any]]:
    stems = RUN_FILES.get(prec, [prec])
    out = []
    for s in stems:
        p = raw / f"{RUN_SLUG.get(slug, slug)}_{s}.json.gz"
        if p.exists():
            d = read_json(p)
            d["_file"] = p.name
            out.append(d)
    return out


def measure(slug: str, prec: str, items: Mapping[str, Any], ref: Mapping[str, Any] | None, price_h: float,
            raw: Path = RAW) -> dict[str, Any] | None:
    """One (model, precision) row: accuracy from the first saturated pass's answers, the saturated rate = the best
    pass over every file of the precision (llama.cpp: P = 8 and P = 16), C = 1 latency from the file that has it."""
    recs = run_record(slug, prec, raw)
    if not recs:
        return None
    ok = [r for r in recs if r.get("state") == "ok"]
    if not ok:
        return {"state": recs[0].get("state") or "failed", "errors": recs[0].get("errors"), "files": [r["_file"] for r in recs]}
    first = ok[0]
    sc = score_run(slug, first, items, ref)
    passes = [dict(p, file=r["_file"]) for r in ok for p in r.get("passes") or []]
    best = max(passes, key=lambda p: p["dps"])
    c1 = next((r["c1"] for r in ok if r.get("c1")), None)
    usdk = usd_per_1000(price_h, best["dps"])
    row = {"state": "ok", "engine": first.get("engine"), "version": first.get("version"), "precision": prec,
           "files": [r["_file"] for r in ok], "dps": best["dps"], "conc": best["c"], "usdk": usdk, "lg": lg(usdk),
           "passes": [{k: p.get(k) for k in ("c", "dps", "wall_s", "agree_with_first", "file")} for p in passes],
           "lat_p50": c1.get("p50_ms") if c1 else None, "lat_p90": c1.get("p90_ms") if c1 else None,
           "lat_n": c1.get("decisions") if c1 else None, "load_s": first.get("load_s"),
           "engine_info": first.get("engine_info"), "extra_args": first.get("extra_args"),
           "longest_prompt": first.get("longest_prompt"), "reads": first.get("reads"),
           "wide_reads": first.get("wide_reads"), "widened_reads": first.get("widened_reads"),
           **{k: v for k, v in sc.items()}}
    return row


def analyse(items: Mapping[str, Any], keys: Mapping[str, str], price_h: float = PRICE_H, raw: Path = RAW
            ) -> dict[str, Any]:
    res: dict[str, Any] = {"models": {}, "price_h": price_h}
    for slug, (label, fam, refkey, _) in MODELS.items():
        ref = load_ref(slug)
        rows = {}
        for prec in PRECISIONS:
            m = measure(slug, prec, items, ref, PRICE_BY_SLUG.get(slug, price_h), raw)
            if m is not None:
                rows[prec] = m
        ref_pct = ref_float(keys, refkey)
        good = {p: r for p, r in rows.items() if r.get("state") == "ok"}
        pick, flag = plotted(good, ref_pct)
        e9_usdk = (ref_float(keys, f"e9/{E9_SLUG.get(slug, slug)}/usdk") if slug != "e28"
                   else ref_float(keys, "e28/l1/usdk"))
        if pick and (good[pick].get("agree_ref") or 1.0) < AGREE_MIN:      # not the same answers as the reference run
            flag = (flag + "; " if flag else "") + f"agreement {100 * good[pick]['agree_ref']:.1f} % < {100 * AGREE_MIN:.0f} %"
        entry = {"label": label, "family": fam, "ref_key": refkey, "ref_acc_pct": ref_pct, "rows": rows,
                 "plotted": pick, "flag": flag, "e9_usdk": e9_usdk}
        if pick:
            p = good[pick]
            entry["point"] = {k: p[k] for k in ("usdk", "lg", "acc_pct", "dps", "conc", "lat_p50", "lat_p90")}
            entry["point"]["prec"] = pick
            if e9_usdk:
                entry["point"]["cheaper_than_e9"] = e9_usdk / p["usdk"]
        res["models"][slug] = entry
    return res


# ================================================================================================ outputs
def _sig(x: float | None, d: int = 2) -> str:
    if x is None or not math.isfinite(x):
        return r"\tbd"
    if x == 0:
        return "0"
    p = d - 1 - int(math.floor(math.log10(abs(x))))
    return f"{round(x, p):.{max(p, 0)}f}"


def _num(x: float | None, digits: int = 1) -> str:
    if x is None or not math.isfinite(x):
        return r"\tbd"
    return f"{x:,.{digits}f}".replace(",", "{,}")


def _int(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return r"\tbd"
    return f"{int(round(x)):,}".replace(",", "{,}")


def _ratio(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return r"\tbd"
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def _dps(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return r"\tbd"
    return _int(x) if x >= 100 else f"{x:.1f}"


def _tex_text(x: str) -> str:
    """Plain text as a TeX key value: % would comment out the rest of the definition."""
    return (x.replace("\\", "/").replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
            .replace("<", "$<$").replace(">", "$>$"))


def tex_keys(res: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, str]:
    k: dict[str, str] = {"e30/meta/instance": meta["instance"], "e30/meta/gpu": meta["gpu"],
                         "e30/meta/price_h": f"{meta['price_h']:.4f}", "e30/meta/engine": meta["engine"],
                         "e30/meta/version": meta["version"], "e30/meta/n": _int(N_E1),
                         "e30/meta/conc": "64 and 256", "e30/meta/latn": _int(1000), "e30/meta/margin": f"{MARGIN_PT:.0f}"}
    if meta.get("llama"):
        k["e30/meta/llama"] = meta["llama"]
    if meta.get("hours") is not None:
        k["e30/meta/hosth"] = f"{meta['hours']:.1f}"
    pts = {s: m for s, m in res["models"].items() if m.get("point") and s not in DESCRIPTIVE_ONLY}
    if pts:   # summary keys for prose: models drawn, how many at FP8, the largest accuracy change, the lowest agreement
        diffs = [abs(m["point"]["acc_pct"] - m["ref_acc_pct"]) for m in pts.values() if m.get("ref_acc_pct") is not None]
        agrees = [m["rows"].get(m["point"]["prec"], {}).get("agree_ref") for m in pts.values()]
        agrees = [a for a in agrees if a is not None]
        k["e30/meta/models"] = str(len(pts))
        k["e30/meta/fp8n"] = str(sum(1 for m in pts.values() if m["point"]["prec"] == "fp8"))
        both = [m for m in pts.values() if all(m["rows"].get(p, {}).get("state") == "ok" for p in ("bf16", "fp8"))]
        nfp8 = sum(1 for m in both if m["point"]["prec"] == "fp8")
        k["e30/meta/nvllm"] = str(len(both))      # plotted models measured at both BF16 and FP8, and how many kept FP8
        k["e30/meta/fp8kept"] = "all of them" if both and nfp8 == len(both) else f"{nfp8} of them"
        if diffs:
            k["e30/meta/maxdiff"] = f"{max(diffs):.1f}"
        same = [m for m in pts.values() if (m["rows"].get(m["point"]["prec"], {}).get("agree_ref") or 1.0) >= AGREE_MIN]
        d2 = [abs(m["point"]["acc_pct"] - m["ref_acc_pct"]) for m in same if m.get("ref_acc_pct") is not None]
        a2 = [m["rows"][m["point"]["prec"]]["agree_ref"] for m in same
              if m["rows"].get(m["point"]["prec"], {}).get("agree_ref") is not None]
        k["e30/meta/nsame"] = str(len(same))
        k["e30/meta/nlowagree"] = str(len(pts) - len(same))
        k["e30/meta/samefor"] = "every model" if len(same) == len(pts) else f"{len(same)} of the {len(pts)} models"
        if d2:
            k["e30/meta/maxdiff_same"] = f"{max(d2):.1f}"
        if a2:
            k["e30/meta/minagree_same"] = f"{100 * min(a2):.1f}"
        if agrees:
            k["e30/meta/minagree"] = f"{100 * min(agrees):.1f}"
        fp8d = [abs(m["rows"]["fp8"]["acc_pct"] - m["rows"]["bf16"]["acc_pct"]) for m in pts.values()
                if m["rows"].get("fp8", {}).get("state") == "ok" and m["rows"].get("bf16", {}).get("state") == "ok"]
        if fp8d:
            k["e30/meta/fp8maxdiff"] = f"{max(fp8d):.1f}"
    for slug, m in res["models"].items():
        if m.get("ref_acc_pct") is not None:
            k[f"e30/{slug}/ref"] = f"{m['ref_acc_pct']:.1f}"
        for prec, r in m["rows"].items():
            base = f"e30/{slug}/{prec}"
            if r.get("state") != "ok":
                k[f"{base}/state"] = "not loaded"
                continue
            k.update({f"{base}/usdk": _sig(r["usdk"], 2), f"{base}/lg": f"{r['lg']:.4f}", f"{base}/acc": f"{r['acc_pct']:.1f}",
                      f"{base}/dps": _dps(r["dps"]), f"{base}/conc": str(r["conc"]),
                      f"{base}/lat_p50": _int(r["lat_p50"]), f"{base}/lat_p90": _int(r["lat_p90"])})
            if r.get("agree_ref") is not None:
                k[f"{base}/agree"] = f"{100 * r['agree_ref']:.1f}"
        pt = m.get("point")
        if not pt:
            continue
        k.update({f"e30/{slug}/usdk": _sig(pt["usdk"], 2), f"e30/{slug}/lg": f"{pt['lg']:.4f}",
                  f"e30/{slug}/acc": f"{pt['acc_pct']:.1f}", f"e30/{slug}/prec": PRECISIONS[pt["prec"]][1],
                  f"e30/{slug}/dps": _dps(pt["dps"]), f"e30/{slug}/conc": str(pt["conc"]),
                  f"e30/{slug}/lat_p50": _int(pt["lat_p50"]), f"e30/{slug}/lat_p90": _int(pt["lat_p90"]),
                  f"e30/{slug}/engine": ENGINE_BY_SLUG.get(slug, PRECISIONS[pt["prec"]][2]),
                  f"e30/{slug}/flag": _tex_text(m["flag"]) if m["flag"] else "none"})
        if pt.get("cheaper_than_e9"):
            k[f"e30/{slug}/vse9"] = _ratio(pt["cheaper_than_e9"])
        if slug in PRICE_BY_SLUG:   # priced on another instance type (amendment (c) 5)
            k[f"e30/{slug}/price_h"] = f"{PRICE_BY_SLUG[slug]:.4f}".rstrip("0")
            k[f"e30/{slug}/instance"], k[f"e30/{slug}/gpu"] = INSTANCE_BY_SLUG[slug]
        agree = m["rows"].get(pt["prec"], {}).get("agree_ref")
        if agree is not None:   # the plotted precision's agreement with the reference record, and its accuracy change
            k[f"e30/{slug}/agree"] = f"{100 * agree:.1f}"
        if m.get("ref_acc_pct") is not None:
            k[f"e30/{slug}/dacc"] = f"{pt['acc_pct'] - m['ref_acc_pct']:+.1f}"
    return k


def frontier(points: Mapping[str, tuple[float, float]]) -> list[str]:
    """E9's rule: the points no other point matches or beats on both cost (lower) and accuracy (higher), one strictly;
    ordered by cost."""
    out = []
    for k, (c, a) in points.items():
        if not any(c2 <= c and a2 >= a and (c2 < c or a2 > a) for k2, (c2, a2) in points.items() if k2 != k):
            out.append(k)
    return sorted(out, key=lambda k: (points[k][0], -points[k][1]))


def figure_points(res: Mapping[str, Any], keys: Mapping[str, str]) -> dict[str, tuple[float, float]]:
    """Figure 4a's drawn points on the new basis: (US$ per decision, accuracy %) by figure coordinate name."""
    pts: dict[str, tuple[float, float]] = {}
    for name, slug in FIG_OPEN.items():
        pt = (res["models"].get(slug) or {}).get("point")
        if pt and pt["acc_pct"] >= FIG_MIN_ACC:
            pts[name] = (pt["usdk"] / 1000.0, pt["acc_pct"])
    for name in FIG_API:
        lgv, acc = ref_float(keys, f"e9/{name}/lg"), ref_float(keys, f"e9/{name}/acc")
        if lgv is not None and acc is not None:
            pts[name] = (10 ** lgv, acc)
    return pts


def write_tex(vals: Mapping[str, str], path: Path = TEX) -> Path:
    lines = ["% AUTO-GENERATED by scripts/explore_e30_cost.py -- do not edit by hand.",
             "% Exploratory row E30, every open decision model at its cheapest common deployment (descriptive; post hoc; "
             "docs/PREREG_A7_STAGE2.md Deviations 2026-10-09), outside both stages.",
             f"% last written {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} at git "
             f"{(git_head() or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(vals.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + [r"\makeatother", ""]))
    return path


def host_meta(box: Path = BOX) -> dict[str, Any]:
    meta: dict[str, Any] = {"instance": INSTANCE, "gpu": GPU, "price_h": PRICE_H, "price_source": PRICE_SOURCE,
                            "engine": "vLLM", "version": "0.31.0"}
    vv = box / "vllm_version.txt"
    if vv.exists():
        m = re.search(r"vllm (\S+)", vv.read_text())
        if m:
            meta["version"] = m.group(1)
        meta["vllm_env"] = vv.read_text().strip()
    h = box / "host.json"
    if h.exists():
        try:
            meta["host"] = json.loads(h.read_text().splitlines()[0])
        except (ValueError, IndexError):
            pass
    hf = box / "host_final.json"
    if hf.exists():
        try:
            meta["host_final"] = json.loads(hf.read_text())
            meta["hours"] = float(meta["host_final"].get("hours_used"))
        except (ValueError, TypeError):
            pass
    li = box / "llama_image.txt"
    if li.exists():
        meta["llama"] = "llama.cpp"
        meta["llama_image"] = li.read_text().strip()
    used = ((meta.get("host") or {}).get("instance_type"))
    if used and used != INSTANCE:
        meta["instance_used"] = used
        meta["instance_used_price_h"] = FALLBACK_PRICE_H.get(used)
    return meta


def render_md(res: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    L = ["# E30: every open decision model at its cheapest common deployment (exploratory, descriptive)", "",
         LABEL + ".", "",
         f"Generated {res['written']} at git {(res.get('git') or '?')[:12]} by `python scripts/explore_e30_cost.py "
         "summarize`.", "",
         f"Instance: {meta['instance']} ({meta['gpu']}), ${meta['price_h']:.4f} per hour ({meta['price_source']})"
         + (f"; measured on a {meta['instance_used']} (same GPU, ${meta.get('instance_used_price_h')} per hour), priced "
            f"at the {meta['instance']} price as registered" if meta.get("instance_used") else "") + ". "
         f"Engine: vLLM {meta['version']} in process (token-id prompts, prefix caching, continuous batching, CUDA graphs); "
         "the 27B models on llama.cpp (Q4_K_M GGUF files), E27 SP on our stack (serve_jev --backend clef). "
         f"Workload: E9's {N_E1:,} L1 decisions in one fixed shuffled order; saturated decisions/s = the best of C = 64 "
         f"and C = 256 (llama.cpp: P = {_slots(res, 'q4_k_m') or '8 and 16'} slots; E27 SP: "
         f"{_slots(res, 'bf16', 'e27sp') or '16 and 64'} requests); cost per 1,000 = $/h / (3,600 x dps) "
         "x 1,000; latency = C = 1 over the first 1,000 decisions. Plotted point = the cheapest precision whose accuracy "
         f"is at least the reference minus {MARGIN_PT:.0f} point.", "",
         "## Plotted points", "",
         "| model | precision | engine | L1 acc % (reference) | decisions/s (C) | $ per 1,000 | x cheaper than E9's basis | "
         "p50 / p90 ms (C = 1) | flag |", "|---|---|---|---|---|---|---|---|---|"]
    for slug, m in res["models"].items():
        pt = m.get("point")
        if not pt:
            L.append(f"| {m['label']} | – | – | – ({_f(m.get('ref_acc_pct'))}) | – | – | – | – | {m['flag']} |")
            continue
        eng = ENGINE_BY_SLUG.get(slug, PRECISIONS[pt["prec"]][2]).replace("\\_", "_")
        vs = _f(pt.get("cheaper_than_e9"))
        L.append(f"| {m['label']} | {PRECISIONS[pt['prec']][0]} | {eng} | {pt['acc_pct']:.1f} ({_f(m.get('ref_acc_pct'))}) | "
                 f"{pt['dps']:.1f} ({pt['conc']}) | {pt['usdk']:.4f} | {vs} | "
                 f"{_f(pt['lat_p50'], 0)} / {_f(pt['lat_p90'], 0)} | {m['flag'] or '–'} |")
    L += ["", "## Every precision measured", "",
          "| model | precision | state | acc % | tool choice / args / When2Call | agreement with the reference record % | "
          "decisions/s per pass | $ per 1,000 | p50 / p90 ms | load s | notes |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for slug, m in res["models"].items():
        if not m["rows"]:
            L.append(f"| {m['label']} | – | not measured | – | – | – | – | – | – | – | – |")
        for prec, r in m["rows"].items():
            if r.get("state") != "ok":
                err = "; ".join((r.get("errors") or [])[-1:])[:160].replace("|", "/")
                L.append(f"| {m['label']} | {PRECISIONS[prec][0]} | {r.get('state')} | – | – | – | – | – | – | – | {err} |")
                continue
            fam = " / ".join(f"{100 * r['families'][f]:.1f}" if f in r["families"] else "–" for f in FAMILIES)
            passes = ", ".join(f"{p['dps']:.1f} (C {p['c']})" for p in r["passes"])
            notes = []
            if r.get("missing"):
                notes.append(f"{r['missing']} unanswered")
            if r.get("extra_args") and r["extra_args"] != {"max_num_seqs": 256}:     # the first capacity tried
                notes.append(f"engine {r['extra_args']}")
            if r.get("wide_reads"):
                notes.append(f"{r['wide_reads']} wide-choice reads")
            if r.get("widened_reads"):
                notes.append(f"{r['widened_reads']} reads widened")
            ei = r.get("engine_info") or {}
            if ei.get("enable_prefix_caching") is False:
                notes.append("prefix caching off (engine)")
            L.append(f"| {m['label']} | {PRECISIONS[prec][0]} | ok | {r['acc_pct']:.1f} | {fam} | "
                     f"{_f(100 * r['agree_ref'] if r.get('agree_ref') is not None else None)} | {passes} | "
                     f"{r['usdk']:.4f} | {_f(r['lat_p50'], 0)} / {_f(r['lat_p90'], 0)} | {_f(r.get('load_s'), 0)} | "
                     f"{'; '.join(notes) or '–'} |")
    L += ["", "## Not re-measured", ""]
    L += [f"- {v}." for k, v in NOT_MEASURED.items() if not res["models"].get(k, {}).get("point")]
    if _slots(res, "q4_k_m") == "8":
        L.append("- The llama.cpp rows' P = 16 passes (registration (f)'s saturation check) were not run (time; execution as "
                 "run): the 27B models' rate and cost come from P = 8.")
    L.append(f"- Outside this entry (not drawn in Figure 4a): {', '.join(OUTSIDE)}.")
    if res.get("frontier"):
        L += ["", "## Frontier of Figure 4a's drawn points on this basis", "",
              "Open models at their E30 point (at least 68 %), Jev / OpenAI Decisions / Flash / Flash-Lite / Flash's own "
              "tool call at their E9 keys; E9's rule. By cost: " + " -> ".join(res["frontier"]) + "."]
    L += ["", "## Reading", ""] + [f"- {x}" for x in res.get("reading", [])] + [""]
    return "\n".join(L)


def _slots(res: Mapping[str, Any], prec: str, slug: str | None = None) -> str:
    """The concurrency levels measured for one precision (one model, or every model), e.g. "8" or "16 and 64"."""
    cs = sorted({p["c"] for s, m in res["models"].items() if slug in (None, s)
                 for q, r in m["rows"].items() if q == prec and r.get("state") == "ok" for p in r.get("passes") or []})
    return " and ".join(str(c) for c in cs)


def _f(x: float | None, d: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def reading(res: Mapping[str, Any]) -> list[str]:
    out = []
    pts = {s: m for s, m in res["models"].items() if m.get("point") and s not in DESCRIPTIVE_ONLY}
    if not pts:
        return ["No model was measured."]
    cheap = min(pts, key=lambda s: pts[s]["point"]["usdk"])
    out.append(f"Cheapest per decision: {pts[cheap]['label']} at ${pts[cheap]['point']['usdk']:.4f} per 1,000 "
               f"({PRECISIONS[pts[cheap]['point']['prec']][0]}).")
    for s, m in pts.items():
        pt = m["point"]
        out.append(f"{m['label']}: {PRECISIONS[pt['prec']][0]}, {pt['acc_pct']:.1f} % (reference "
                   f"{_f(m.get('ref_acc_pct'))}), {pt['dps']:.1f} decisions/s at C = {pt['conc']}, "
                   f"${pt['usdk']:.4f} per 1,000"
                   + (f", {pt['cheaper_than_e9']:.1f}x below E9's basis" if pt.get("cheaper_than_e9") else "")
                   + (f" [{m['flag']}]" if m["flag"] else "") + ".")
    return out


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def write_preds(res: Mapping[str, Any], out: Path = OUT / "preds") -> list[str]:
    """Slim per-decision records (id, prediction, correct) of every measured precision (the raw answers stay on S3)."""
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for slug, m in res["models"].items():
        for prec, r in m["rows"].items():
            if r.get("state") != "ok":
                continue
            p = out / f"{slug}_{prec}.jsonl.gz"
            with open(p, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as g:
                for i in sorted(r["_preds"]):
                    g.write((json.dumps({"id": i, "pred": r["_preds"][i], "correct": r["_correct"][i]}) + "\n").encode())
            files.append(str(p.relative_to(ROOT)))
    return files


# ================================================================================================ commands
def cmd_fetch(a: argparse.Namespace) -> int:
    import os

    env = {k: v for k, v in os.environ.items() if k not in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")}
    env["AWS_PAGER"] = ""
    BOX.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    base = ["aws", "--region", "us-east-1", "s3", "sync"]
    subprocess.run(base + [S3, str(BOX), "--exclude", "runs/*", "--exclude", "logs/time_*.out", "--only-show-errors"],
                   check=True, env=env)
    subprocess.run(base + [f"{S3}/runs", str(RAW), "--exclude", "*smoke*", "--only-show-errors"], check=True, env=env)
    # the g6e.xlarge of the amendment "E15's open baselines" (c) 5 (OpenJev)
    subprocess.run(base + [S3_L40S, str(BOX_L40S), "--exclude", "runs/*", "--only-show-errors"], check=True, env=env)
    subprocess.run(base + [f"{S3_L40S}/runs", str(RAW), "--only-show-errors"], check=True, env=env)
    print(json.dumps({"box": str(BOX.relative_to(ROOT)), "raw": str(RAW.relative_to(ROOT)),
                      "runs": sorted(p.name for p in RAW.glob("*.json.gz"))}, indent=1))
    return 0


def cmd_summarize(a: argparse.Namespace) -> int:
    items = load_items()
    if len(items) != N_E1:
        raise SystemExit(f"E1 has {len(items)} decisions, not {N_E1}")
    keys = generated_keys()
    meta = host_meta()
    res = analyse(items, keys, PRICE_H)
    pts = figure_points(res, keys)
    res["figure_points"] = pts
    res["frontier"] = frontier(pts)
    res.update({"label": LABEL, "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "git": git_head(), "meta": meta, "not_measured": {k: v for k, v in NOT_MEASURED.items() if not res["models"].get(k, {}).get("point")},
                "outside": list(OUTSIDE)})
    res["reading"] = reading(res)
    OUT.mkdir(parents=True, exist_ok=True)
    res["preds_files"] = write_preds(res)
    (OUT / "summary.json").write_text(json.dumps(_jsonable(res), indent=1) + "\n")
    (OUT / "summary.md").write_text(render_md(res, meta))
    k = tex_keys(res, meta)
    if res["frontier"]:
        k.update({"e30/front/first": res["frontier"][0], "e30/front/rest": ",".join(res["frontier"][1:]),
                  "e30/front/n": str(len(res["frontier"]))})
    write_tex(k)
    print("\n".join(res["reading"]))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("summarize")
    a = ap.parse_args(list(argv) if argv is not None else None)
    return {"fetch": cmd_fetch, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
