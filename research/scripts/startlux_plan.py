#!/usr/bin/env python3
"""StartLux-Decision serving plan and manifest (docs/PREREG_A7_STAGE2.md A7.18 (d), (g)); standard library only, so the
GPU host runs it with any python3 before the serving environment exists. Used by scripts/aws/serve_startlux.sh.

    python3 scripts/startlux_plan.py plan --gpus-mib 23034 [--gpus-mib 81559] --sizes 4B 27B
        # -> one JSON line per size: {size, mode: original | gguf_q8_0 | not_run, gpu, port, repo, local_name, reason}
    python3 scripts/startlux_plan.py smoke --port 8090 [--name StartLux-Decision-4B]
        # one 3-question request and one 69-option choice (the JevGym tool count; the server's own knockout) -> JSON
    python3 scripts/startlux_plan.py manifest --out results/stage2/startlux_manifest.json --entry FILE ...

Rules (A7.18 (g)), decided by availability before the first request and never by an outcome: a size runs from its
original weights on one GPU with at least ORIGINAL_MIN_GIB of memory; the 27B, when no such GPU exists, runs from the
vendor's Q8_0 GGUF through `startlux_decision.gguf_server` on a GPU with at least 40 GiB (labelled "27B (Q8_0)");
otherwise the size is not run. Q4_K_M is never used. One size per GPU; the 4B takes the smallest GPU it fits, the
27B the largest."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from typing import Any, Sequence

PKG_REPO = "https://github.com/StartLuxLabs/startlux-decision"
PKG_COMMIT = "a9101e918e097b26fea5ead9ed893b823e875e46"     # HEAD read on 2026-10-03 (A7.18 (b))
HF_ORG = "startlux-models"
# size -> (original weights GB as listed, minimum GPU memory GiB at bf16 with activations, default port)
SIZES = {"0.8B": (1.8, 8, 8092), "2B": (4.6, 12, 8093), "4B": (9.3, 20, 8090), "9B": (19.4, 32, 8094),
         "27B": (55.6, 64, 8091)}
GGUF_Q8_MIN_GIB = {"27B": 40}       # only the 27B has a registered fallback (A7.18 (g))


def plan(sizes: Sequence[str], gpus_mib: Sequence[int]) -> list[dict[str, Any]]:
    gib = [m / 1024 for m in gpus_mib]
    free = list(range(len(gib)))
    out = []
    for size in sorted(sizes, key=lambda s: -SIZES[s][1]):        # the largest model chooses first
        _, need, port = SIZES[size]
        rec: dict[str, Any] = {"size": size, "port": port, "repo": f"{HF_ORG}/StartLux-Decision-{size}",
                               "local_name": f"StartLux-Decision-{size}"}
        fits = [i for i in free if gib[i] >= need]
        if fits:
            g = max(fits, key=lambda i: gib[i]) if size == "27B" else min(fits, key=lambda i: gib[i])
            rec.update({"mode": "original", "gpu": g, "reason": f"{gib[g]:.0f} GiB >= {need} GiB"})
            free.remove(g)
        elif size in GGUF_Q8_MIN_GIB and any(gib[i] >= GGUF_Q8_MIN_GIB[size] for i in free):
            g = max((i for i in free if gib[i] >= GGUF_Q8_MIN_GIB[size]), key=lambda i: gib[i])
            rec.update({"mode": "gguf_q8_0", "gpu": g, "repo": f"{HF_ORG}/StartLux-Decision-{size}-Q8_0-GGUF",
                        "local_name": f"StartLux-Decision-{size}-Q8_0",
                        "reason": f"no free GPU >= {need} GiB; Q8_0 GGUF on {gib[g]:.0f} GiB (A7.18 (g))"})
            free.remove(g)
        else:
            rec.update({"mode": "not_run", "gpu": None,
                        "reason": f"no free GPU with >= {need} GiB" + (
                            f" (or >= {GGUF_Q8_MIN_GIB[size]} GiB for the Q8_0 GGUF)" if size in GGUF_Q8_MIN_GIB else "")
                        + " -- A7.18 (g): not run"})
        out.append(rec)
    return sorted(out, key=lambda r: list(SIZES).index(r["size"]))


def smoke_requests() -> list[dict[str, Any]]:
    small = {"state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
             "questions": {"team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                                    "criteria": {"billing": "Payments, refunds and invoices",
                                                 "shipping": "Delivery and tracking",
                                                 "technical": "App, login and account problems"}},
                           "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
                           "severity": {"type": "score", "instructions": "How severe is the impact?",
                                        "criteria": ["cosmetic", "annoying", "blocks the customer"]}}}
    tools = {f"tool_{i:02d}": f"Customer-operations tool number {i}" for i in range(68)}
    tools["cancel_order"] = "Cancel a pending order of a customer"
    wide = {"state": "The customer asked to cancel their pending order 7781; the order was found and is pending.",
            "questions": {"next_tool": {"type": "choice", "instructions": "Which tool should be called next?",
                                        "criteria": tools}}}
    return [small, wide]


def answer_problems(q: dict[str, Any], a: Any) -> list[str]:
    """Format checks of one answer against its question (the Jev wire format the harness parses): a choice has a
    probability for every listed option and its `choice` among them, a noul a probability in [0, 1], a score one
    probability per level; probabilities sum to 1. Empty list = well-formed."""
    if not isinstance(a, dict):
        return ["no answer object"]
    t, bad = q.get("type", "choice"), []
    if a.get("type") not in (None, t):
        bad.append(f"type {a.get('type')!r} != {t!r}")
    if t == "noul":
        v = a.get("noul")
        if not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
            bad.append(f"noul {v!r} not a probability")
        return bad
    probs = a.get("probabilities")
    want = [str(k) for k in q["criteria"]] if t == "choice" else [str(i) for i in range(len(q["criteria"]))]
    if not isinstance(probs, dict) or sorted(probs) != sorted(want):
        return bad + [f"probabilities over {len(probs) if isinstance(probs, dict) else probs!r} keys, expected {len(want)}"]
    total = sum(float(v) for v in probs.values())
    if abs(total - 1.0) > 1e-3 or min(float(v) for v in probs.values()) < 0:
        bad.append(f"probabilities sum to {total:.4f}")
    if t == "choice" and a.get("choice") not in probs:
        bad.append(f"choice {a.get('choice')!r} not an option")
    return bad


def smoke(port: int, name: str | None, timeout: float = 120.0) -> dict[str, Any]:
    res: dict[str, Any] = {"port": port, "ok": True, "requests": []}
    for body in smoke_requests():
        if name:
            body = {**body, "model": name}
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            ans = data.get("answers") or {}
            problems = {k: answer_problems(q, ans.get(k)) for k, q in body["questions"].items()}
            problems = {k: v for k, v in problems.items() if v}
            if set(ans) - set(body["questions"]):
                problems["_extra"] = sorted(set(ans) - set(body["questions"]))
            ok = not problems
            n_opts = {k: len((a or {}).get("probabilities") or {}) for k, a in ans.items()}
            res["requests"].append({"questions": len(body["questions"]), "ok": ok, "model": data.get("model"),
                                    "ms": (time.perf_counter() - t0) * 1000, "n_probabilities": n_opts,
                                    "usage": data.get("usage"), "server_latency_ms": data.get("latency_ms"),
                                    **({"problems": problems} if problems else {})})
            res["ok"] = res["ok"] and ok
        except Exception as e:  # noqa: BLE001 - a failed smoke is recorded, never hidden
            res["requests"].append({"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            res["ok"] = False
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--gpus-mib", type=int, action="append", required=True, help="total memory of each GPU (MiB)")
    p.add_argument("--sizes", nargs="+", default=["4B", "27B"], choices=list(SIZES))
    s = sub.add_parser("smoke")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--name", default=None)
    m = sub.add_parser("manifest")
    m.add_argument("--out", required=True)
    m.add_argument("--entry", action="append", default=[], help="JSON files (host, plan, revisions, health, smoke)")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        for rec in plan(a.sizes, a.gpus_mib):
            print(json.dumps(rec))
        return 0
    if a.cmd == "smoke":
        r = smoke(a.port, a.name)
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    man: dict[str, Any] = {"registered": "PREREG_A7_STAGE2 A7.18 (d)", "package": {"repo": PKG_REPO, "commit": PKG_COMMIT},
                           "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for f in a.entry:
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    man.setdefault(rec.pop("_section", "entries"), []).append(rec)
    with open(a.out, "w") as fh:
        json.dump(man, fh, indent=1)
    print(f"manifest -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
