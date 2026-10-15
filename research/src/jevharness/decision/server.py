"""Serve ANY System-1 backend behind the Jev wire protocol (POST /v1/systemone).

Lets every harness adapter (pi extensions via JEV_BASE_URL, Codex / DeepSeek Harness
hooks via JEV_S1, mini-SWE-agent, tau2) use a backend that is not Jev itself -- e.g. a Gemini
verbalised-distribution decider, a vLLM small LM read through logprobs, or a simulated backend --
without code changes. Swapping in real Jev later is a base-URL change.

    python -m jevharness.decision.server --port 8090 --name gemini-s1 \\
        --s1 '{"type":"llm_generative","llm":{"provider":"gemini","model":"gemini-3.5-flash-lite",
               "reasoning_effort":"minimal"}}'
    curl -s localhost:8090/v1/systemone -d '{"state":"...","model":"gemini-s1","questions":{...}}'
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from typing import Any

try:  # module-level so FastAPI resolves the annotation under `from __future__ import annotations`
    from fastapi import Request
except ImportError:  # pragma: no cover - optional extra
    Request = Any  # type: ignore[misc,assignment]

from .base import DecisionBackend
from .types import Answer, Choice, Noul, QKind, Question, Score


def wire_to_question(qid: str, q: dict[str, Any]) -> Question:
    t = q.get("type")
    instr = q.get("instructions") or ""
    if not isinstance(instr, str):
        instr = json.dumps(instr)
    crit = q.get("criteria")
    if t == "choice":
        if not isinstance(crit, dict) or len(crit) < 2:
            raise ValueError(f"{qid}: choice needs >= 2 criteria")
        labels = list(crit)
        descs = [None if v is None else (v if isinstance(v, str) else json.dumps(v)) for v in crit.values()]
        return Choice(instr, labels, descs if any(descs) else None, wire_id=qid)
    if t == "noul":
        extra = {"criteria": crit} if isinstance(crit, dict) else {}
        return Noul(instr, wire_id=qid, **extra)
    if t == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise ValueError(f"{qid}: score needs >= 2 levels")
        levels = [c if isinstance(c, str) else json.dumps(c) for c in crit]
        return Score(instr, scale=(0, len(levels) - 1), levels=levels, wire_id=qid)
    raise ValueError(f"{qid}: unsupported question type {t!r}")


def answer_to_wire(q: Question, a: Answer) -> dict[str, Any]:
    if q.kind == QKind.CHOICE:
        out = {"type": "choice", "choice": a.value, "confidence": a.confidence, "probabilities": a.distribution}
        ko = (a.raw or {}).get("knockout")
        if isinstance(ko, dict):  # review P1-4: a multi-pass knockout says so (rounds, reads, finalists, composition)
            out["knockout"] = ko
        return out
    if q.kind == QKind.NOUL:
        return {"type": "noul", "noul": a.p_yes}
    probs = {str(int(k)): v for k, v in a.distribution.items()}
    return {"type": "score", "score": float(a.value), "confidence": a.confidence,
            "legend": {str(i): d for i, d in enumerate(q.levels or [])}, "probabilities": probs}


def build_app(backend: DecisionBackend, name: str = "s1"):  # noqa: ANN201
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title=f"jevharness System-1 server ({name})")
    stats = {"requests": 0, "questions": 0, "errors": 0}
    lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **stats}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"data": [{"id": name, "object": "model", "backend": backend.describe()}]}

    @app.post("/v1/systemone")
    async def systemone(req: Request) -> JSONResponse:
        body = await req.json()
        state = body.get("state")
        wire_qs = body.get("questions") or {}
        if state is None or not wire_qs:
            raise HTTPException(422, "state and questions are required")
        try:
            qs = [wire_to_question(k, v) for k, v in wire_qs.items()]
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        state_s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        t = time.perf_counter()
        try:
            # blocking backend call off the event loop: otherwise concurrent clients (parallel harness
            # workers) are served one at a time and their measured System-1 latency includes queueing
            res = await run_in_threadpool(backend.decide, state_s, qs)
        except Exception as e:  # surface as 502 so clients fail open
            with lock:
                stats["errors"] += 1
            raise HTTPException(502, f"backend error: {type(e).__name__}: {e}") from e
        ms = (time.perf_counter() - t) * 1000
        with lock:
            stats["requests"] += 1
            stats["questions"] += len(qs)
        answers = {q.meta["wire_id"]: answer_to_wire(q, a) for q, a in zip(qs, res.answers)}
        u = res.usage
        usage: dict[str, Any] = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}
        if u.priced is not False:  # an unknown price is not a billed $0: let the client flag it as unpriced
            usage["cost"] = u.cost_usd
        for k in ("n_passes", "n_rounds"):  # review P1-4: passes / rounds when the wrapped backend observes them
            if getattr(u, k) is not None:
                usage[k] = getattr(u, k)
        return JSONResponse({"model": name, "answers": answers, "usage": usage},
                            headers={"server-timing": f"model;dur={ms:.1f}"})

    return app


def main() -> None:
    from dotenv import load_dotenv

    from . import make_backend

    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True, help="backend spec JSON for make_backend")
    ap.add_argument("--name", default="s1")
    ap.add_argument("--cache", default=None, help="optional SQLite cache path")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    import uvicorn

    backend = make_backend(json.loads(a.s1), cache=a.cache)
    uvicorn.run(build_app(backend, a.name), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
