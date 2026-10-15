#!/usr/bin/env python3
"""E15 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E15: more decision-model baselines"): a thin Jev-wire HTTP
wrapper around a decision model's own documented Python function, for the models that ship no server. Runs on the E15
GPU host only (the vendor code it imports comes from the downloaded model directory; never run in the coordinating
container).

    python -I scripts/aws/e15_wrap_server.py --kind clef  --model /opt/jev/e15_models/clef-flash --name clef-flash --port 8090
    python -I scripts/aws/e15_wrap_server.py --kind clef  --model /opt/jev/e15_models/clef --name clef --port 8090 --load-8bit
    python -I scripts/aws/e15_wrap_server.py --kind bosun --model /opt/jev/e15_models/bosun --name bosun-3.1-1.7b --port 8090

``clef``: Cloudflare's model card -- ``load_release_model`` and ``systemone`` from the release's
``joint_schema_model.py`` (the card's loader and defaults; ``--load-8bit`` loads the backbone with bitsandbytes LLM.int8
through the loader's ``from_pretrained`` keyword arguments, the joint head stays bf16). The request body is passed to
``systemone`` unchanged and its response is returned unchanged.

``bosun``: Hanno Labs' model card -- ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`` and
``model.predict(state=, instructions=, candidates=, decision_type=)`` per question (the card's defaults for ``seed`` and
``row_id``); a Choice's criteria become candidates ``{id, label[, description]}``, a yes / no question the card's
``yes`` / ``no`` candidates, a Score's levels candidates ``"0", "1", ...``; answers in the Jev wire format.

One request at a time on the GPU (a lock); ``usage.input_tokens`` counts the tokens the model read; the
``server-timing`` header carries the model time.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:   # module level: FastAPI resolves the handler's (string) annotations in this module's globals
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except ImportError:   # pragma: no cover - the host venv has FastAPI; the coordinating container never runs this file
    FastAPI = Request = JSONResponse = None  # type: ignore[assignment,misc]


def load(kind: str, model_dir: str, load_8bit: bool) -> Any:
    import torch

    if kind == "clef":
        sys.path.insert(0, model_dir)
        from joint_schema_model import load_release_model, systemone  # type: ignore[import-not-found]

        kw: dict[str, Any] = {}
        if load_8bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model, processor = load_release_model(model_dir, device="cuda", dtype=torch.bfloat16, **kw)

        def answer(body: dict[str, Any]) -> dict[str, Any]:
            return systemone(model, processor, body)

        return answer
    if kind == "bosun":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype="auto", device_map="auto")
        model.eval()
        tok = model.tokenizer

        def one(state: Any, qid: str, q: dict[str, Any]) -> tuple[dict[str, Any], int]:
            t = q.get("type")
            instr = q.get("instructions") or qid
            if t == "choice":
                crit = q.get("criteria") or {}
                opts = list(crit) if isinstance(crit, dict) else [str(c) for c in crit]
                cands = []
                for o in opts:
                    c: dict[str, Any] = {"id": str(o), "label": str(o)}
                    d = crit.get(o) if isinstance(crit, dict) else None
                    if d:
                        c["description"] = str(d)
                    cands.append(c)
                out = model.predict(state=state, instructions=instr, candidates=cands, decision_type="choice")
                p = [float(x) for x in out["probabilities"]]
                probs = dict(zip(opts, p))
                best = max(opts, key=probs.__getitem__)
                ans = {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": probs}
            elif t == "noul":
                cands = [{"id": "yes", "label": "Yes"}, {"id": "no", "label": "No"}]
                out = model.predict(state=state, instructions=instr, candidates=cands, decision_type="noul")
                ans = {"type": "noul", "noul": float(out["probabilities"][0])}
            elif t == "score":
                levels = [str(x) for x in (q.get("criteria") or [])]
                cands = [{"id": str(i), "label": lv} for i, lv in enumerate(levels)]
                out = model.predict(state=state, instructions=instr, candidates=cands, decision_type="score")
                p = [float(x) for x in out["probabilities"]]
                ans = {"type": "score", "score": sum(i * v for i, v in enumerate(p)), "confidence": max(p),
                       "probabilities": {str(i): v for i, v in enumerate(p)}}
            else:
                raise ValueError(f"{qid}: unknown question type {t!r}")
            n_tok = len(tok(out["prompt"])["input_ids"]) if out.get("prompt") else 0
            return ans, n_tok

        def answer(body: dict[str, Any]) -> dict[str, Any]:
            qs = body.get("questions")
            if not isinstance(qs, dict) or not qs or "state" not in body:
                raise ValueError("state and at least one question are required")
            answers, n = {}, 0
            for qid, q in qs.items():
                answers[qid], k = one(body["state"], qid, q)
                n += k
            return {"model": body.get("model"), "answers": answers, "usage": {"input_tokens": n, "output_tokens": 0}}

        return answer
    raise SystemExit(f"unknown kind {kind}")


def build_app(answer: Any, name: str, kind: str, load_8bit: bool = False) -> Any:
    """The Jev-wire app around ``answer(body) -> response body``: one request at a time (a lock)."""
    lock = threading.Lock()
    app = FastAPI()
    counters = {"requests": 0, "errors": 0}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "name": name, "kind": kind, "load_8bit": load_8bit, **counters}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": name, "object": "model"}]}

    @app.post("/v1/systemone")
    async def systemone(req: Request) -> JSONResponse:
        body = await req.json()
        try:
            with lock:
                t = time.perf_counter()
                out = answer(body)
                ms = (time.perf_counter() - t) * 1000
            counters["requests"] += 1
            out["model"] = name
            return JSONResponse(out, headers={"server-timing": f"model;dur={ms:.1f}"})
        except Exception as e:  # noqa: BLE001 - a bad request is the client's error, never a crash
            counters["errors"] += 1
            return JSONResponse({"error": {"type": "api_usage_error", "message": f"{type(e).__name__}: {e}"[:500]}},
                                status_code=400)

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kind", choices=("clef", "bosun"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--load-8bit", action="store_true")
    a = ap.parse_args(argv)
    if not Path(a.model).is_dir():
        raise SystemExit(f"no model directory {a.model}")

    import uvicorn

    t0 = time.time()
    answer = load(a.kind, a.model, a.load_8bit)
    print(f"loaded {a.kind} {a.model} in {time.time() - t0:.0f} s", flush=True)
    app = build_app(answer, a.name, a.kind, a.load_8bit)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
