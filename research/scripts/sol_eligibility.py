#!/usr/bin/env python3
"""The optional closed-family row of docs/PREREG_A7_STAGE2.md A7.12 (i) (gpt-6.1-sol with gpt-6-luna as its small
model) under the procedure of A7.12 (a)-(c): capability manifest, eligibility, the outcome-blind cost gate that fixes
n, and the spend tracker. Reads development smoke files (seeds 0-49) and ledgers; never a success number.

    python scripts/sol_eligibility.py probe --write        # tiny capability calls -> results/stage2/sol_capability_manifest.json
    for p in 1 2 3; do jevbench run configs/dev/sol_eligibility.yaml --out runs/stage2_dev -o workers=4; done
    python scripts/sol_eligibility.py gate --write          # -> results/stage2/sol_cost_gate.json (n before calibration)
    python scripts/sol_eligibility.py spend [--cap 85]      # what every Sol run has cost so far (exit 3 above the cap)
    python scripts/sol_eligibility.py watch --cap 85 --kill-pattern stage2_sol.yaml   # stop a run before the cap

Eligibility per member (A7.12 (b)): malformed responses <= 5 % of steps (``x_eligibility._malformed``: a tool call
naming no offered tool, arguments that are not JSON, an empty response; a final-text answer is not malformed), no
infrastructure error left after the retries, no context-length error, and every System-2 call served by the requested
model: OpenAI is called directly (no router, so no provider fallback exists) and /v1/models lists no dated snapshot of
either id, so the id is the pin and the ledger's ``served_model`` (the API's own ``model`` field) must equal it.

Cost gate (A7.12 (c) with A7.12 (i)'s envelope): per-episode cost of each member at the registered frozen price
(gpt-6.1-sol $2.00 / $10.00 per M, cached input $0.10, A7.12 (i); gpt-6-luna $0.10 / $0.50 / $0.01,
configs/prices/models.yaml) plus the cache-write surcharge the pricing page lists (input x 1.25 on
``cache_write_tokens``; not in the frozen price, so it is counted here and in the spend, and reported as a cost
sensitivity), then predicted total = probe + smoke actual + calibration (150 Sol episodes with Jev's shadow calls) +
`llm_only_sol` (n) + `jevagent_best_sol` (n, bounded by `llm_only_sol`'s per-episode cost, as for X) +
`llm_only_solsmall` (n) + Jev's calls, x 1.2 contingency; n = 500 if it fits the envelope, else 300 if it fits, else
the row is not run (resource gate). ENVELOPE = $80, the upper end of A7.12 (i)'s "≈ $70-80", fixed in the dated
Deviations entry before the smoke; the operator's hard spend cap ($85, smokes and calibration included) is enforced
by ``watch`` and is not part of the rule."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jevharness.harness.trajectory import iter_records, latest_by_task  # noqa: E402
from x_eligibility import _malformed  # noqa: E402   (the one malformed-response rule of A7.12 (b))

SOL, LUNA = "gpt-6.1-sol", "gpt-6-luna"
# $ per 1M input / output / cached-input tokens: Sol frozen by A7.12 (i); Luna as configs/prices/models.yaml (verified)
FROZEN_PRICES = {SOL: (2.00, 10.00, 0.10), LUNA: (0.10, 0.50, 0.01)}
CACHE_WRITE_FACTOR = 1.25          # pricing page, 2026-10-05: cache writes $2.50 (Sol) / $0.125 (Luna) per M
MEMBERS = {"llm_only_sol": SOL, "llm_only_solsmall": LUNA}
MALFORMED_MAX, ENVELOPE, CONTINGENCY, N_CAL = 0.05, 80.0, 1.2, 150
JEV_CAL_PER_EPISODE = 0.006        # Jev's shadow calls per calibration episode (X's calibration: $0.882 / 150)
JEV_PER_EPISODE = 0.003            # Jev per profile episode (A7.12 (c)'s planning value; X's profile arm: $0.00036)
HARD_CAP = 85.0
SMOKE_RUN = ROOT / "runs/stage2_dev/sol_eligibility"
SOL_RUNS = (ROOT / "runs/stage2_dev/sol_eligibility", ROOT / "runs/e2a_jevgym_shadow_cal_sol", ROOT / "runs/stage2_sol")
MANIFEST = ROOT / "results/stage2/sol_capability_manifest.json"
OUT = ROOT / "results/stage2/sol_cost_gate.json"


def surcharge(model: str, cache_write_tokens: float) -> float:
    """$ the cache-write tokens cost on top of the frozen (plain input) price."""
    pi = FROZEN_PRICES.get(model, (0.0, 0.0, 0.0))[0]
    return (CACHE_WRITE_FACTOR - 1.0) * pi * float(cache_write_tokens or 0) / 1e6


def record_cost(t: Mapping[str, Any]) -> dict[str, float]:
    """One trajectory record's spend: the ledger (shadow System-1 calls included) + the cache-write surcharge."""
    u = t.get("usage") or {}
    led = float((u.get("total_with_shadow") or u.get("total") or {}).get("cost_usd") or 0.0)
    extra = sum(surcharge(str(c.get("model")), (c.get("meta") or {}).get("cache_write_tokens") or 0)
                for c in t.get("calls") or [] if c.get("component") == "s2")
    return {"ledger": led, "cache_write_surcharge": extra, "total": led + extra}


def member(path: Path, model: str) -> dict[str, Any]:
    recs = list(iter_records(path)) if path.exists() else []
    latest = latest_by_task(r for r in recs if "task_id" in r)
    eps = list(latest.values())
    errors = [t for t in eps if t.get("error")]
    ctx = [t for t in errors if "context" in str(t.get("error")).lower()]
    steps = sum(len(t.get("steps") or []) for t in eps if not t.get("error"))
    bad = sum(_malformed(t) for t in eps if not t.get("error"))
    served: dict[str, int] = {}
    tiers: dict[str, int] = {}
    tin = tout = tcache = twrite = treason = 0.0
    calls = 0
    for t in eps:
        for c in t.get("calls") or []:
            if c.get("component") != "s2":
                continue
            calls += 1
            m = c.get("meta") or {}
            sm = str(m.get("served_model") or "(none)")
            served[sm] = served.get(sm, 0) + 1
            st = str(m.get("service_tier") or "(none)")
            tiers[st] = tiers.get(st, 0) + 1
            tin += float(c.get("input_tokens") or 0)
            tout += float(c.get("output_tokens") or 0)
            tcache += float(m.get("cached_input_tokens") or 0)
            twrite += float(m.get("cache_write_tokens") or 0)
            treason += float(m.get("reasoning_tokens") or 0)
    pi, po, pc = FROZEN_PRICES[model]
    frozen = ((tin - tcache) * pi + tcache * pc + tout * po) / 1e6
    with_writes = frozen + surcharge(model, twrite)
    n = max(1, len(eps))
    rate = bad / steps if steps else float("nan")
    pin_ok = bool(served) and set(served) == {model}
    spent = sum(record_cost(t)["total"] for t in recs if "task_id" in t)      # every attempt, errored ones included
    return {"model": model, "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "episodes": len(eps), "attempts": len(recs), "steps": steps, "malformed": bad, "malformed_rate": rate,
            "infra_errors_left": len(errors), "context_errors": len(ctx), "s2_calls": calls,
            "served_models": served, "service_tiers": tiers, "pinned": model,
            "eligible": bool(eps) and steps > 0 and rate <= MALFORMED_MAX and not errors and not ctx and pin_ok,
            "cost_per_episode_frozen": frozen / n, "cost_per_episode": with_writes / n, "smoke_spend": spent,
            "tokens_per_episode": {"input": tin / n, "output": tout / n, "cached": tcache / n,
                                   "cache_write": twrite / n, "reasoning": treason / n}}


def gate(rows: Mapping[str, Mapping[str, Any]], probe_usd: float = 0.0, envelope: float = ENVELOPE) -> dict[str, Any]:
    cs = rows["llm_only_sol"]["cost_per_episode"]
    cl = rows["llm_only_solsmall"]["cost_per_episode"]
    smoke = probe_usd + sum(r["smoke_spend"] for r in rows.values())
    pred = {}
    for n in (500, 300):
        pred[n] = CONTINGENCY * (smoke + N_CAL * (cs + JEV_CAL_PER_EPISODE) + n * cs + n * cs + n * cl
                                 + n * JEV_PER_EPISODE)
    n = 500 if pred[500] <= envelope else 300 if pred[300] <= envelope else None
    eligible = all(r["eligible"] for r in rows.values())
    return {"eligible": eligible, "predicted_usd": {str(k): v for k, v in pred.items()}, "envelope_usd": envelope,
            "n": n if eligible else None,
            "decision": (f"n = {n}" if n else "not run (resource gate, A7.12 (c), (i))") if eligible else
            "pair ineligible (A7.12 (b)): the optional Sol row is not run (no other closed pair is registered)",
            "assumptions": {"probe_usd": probe_usd, "jev_calibration_per_episode_usd": JEV_CAL_PER_EPISODE,
                            "jev_profile_per_episode_usd": JEV_PER_EPISODE, "calibration_episodes": N_CAL,
                            "jevagent_best_sol_bound": "llm_only_sol per-episode cost", "contingency": CONTINGENCY,
                            "prices_per_M": {m: {"input": p[0], "output": p[1], "cached_input": p[2],
                                                 "cache_write": p[0] * CACHE_WRITE_FACTOR}
                                             for m, p in FROZEN_PRICES.items()}}}


# ------------------------------------------------------------------------------------------------ spend
def spend(runs: Sequence[Path] = SOL_RUNS, manifest: Path = MANIFEST) -> dict[str, Any]:
    """Spend of every Sol record so far (all attempts) + the capability probe, by run and arm."""
    by: dict[str, dict[str, float]] = {}
    tot = 0.0
    for d in runs:
        for p in sorted(Path(d).glob("*.jsonl")):
            if p.name.startswith("schedule_order"):
                continue
            s = {"records": 0, "ledger": 0.0, "cache_write_surcharge": 0.0, "total": 0.0}
            for t in iter_records(p):
                if "task_id" not in t:
                    continue
                c = record_cost(t)
                s["records"] += 1
                for k in ("ledger", "cache_write_surcharge", "total"):
                    s[k] += c[k]
            by[str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)] = s
            tot += s["total"]
    probe = float(json.loads(manifest.read_text()).get("probe_cost_usd") or 0.0) if manifest.exists() else 0.0
    return {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "probe_usd": probe,
            "runs": by, "total_usd": tot + probe}


def _alive(pgid: int | None, pattern: str | None) -> list[int]:
    pids: list[int] = []
    if pattern:
        pids += [int(p) for p in subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
                 if int(p) != os.getpid()]
    if pgid:
        try:
            os.killpg(pgid, 0)
            pids.append(-pgid)
        except (ProcessLookupError, PermissionError):
            pass
    return pids


def watch(cap: float, reserve: float, pattern: str | None, pgid: int | None, interval: float, log: Path) -> int:
    """Poll the spend while the watched run lives (process group ``pgid`` and / or processes whose command line
    contains ``pattern``); when spend + reserve reaches the cap, SIGTERM them (the runner keeps every finished
    episode; the in-flight ones are lost and re-run by a resume) until none is left, and exit 3. Exit 0 when the run
    ends below the cap."""
    log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        s = spend()
        live = _alive(pgid, pattern)
        with log.open("a") as f:
            f.write(json.dumps({"at": s["at"], "total_usd": round(s["total_usd"], 4), "live": len(live)}) + "\n")
        if s["total_usd"] + reserve >= cap:
            while live:
                for pid in live:
                    try:
                        (os.killpg(-pid, signal.SIGTERM) if pid < 0 else os.kill(pid, signal.SIGTERM))
                    except ProcessLookupError:
                        pass
                time.sleep(5)
                live = _alive(pgid, pattern)
            with log.open("a") as f:
                f.write(json.dumps({"at": s["at"], "stopped": True, "total_usd": s["total_usd"], "cap": cap}) + "\n")
            return 3
        if not live:
            return 0
        time.sleep(interval)


# ------------------------------------------------------------------------------------------------ probe
def probe() -> dict[str, Any]:
    """Tiny live calls (< $0.01): ids listed, accepted reasoning efforts, tool calling (auto and a forced named tool),
    a function-call round trip with the reasoning items echoed, previous_response_id chaining, structured output
    (text.format json_schema), and the account's rate limits. Prints no key."""
    import httpx
    from dotenv import load_dotenv

    from jevharness.accounting import Ledger
    from jevharness.llm import make_llm

    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENAI_API_KEY", "")
    H = {"Authorization": f"Bearer {key}"}
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    r = httpx.get("https://api.openai.com/v1/models", headers=H, timeout=30)
    listed = {m["id"]: m for m in r.json().get("data", [])}
    out: dict[str, Any] = {"registered": "docs/PREREG_A7_STAGE2.md A7.12 (i) under A7.12 (a): capability manifest of the "
                                         "optional closed-family pair, written before the eligibility smoke",
                           "written": now, "provider": "OpenAI, https://api.openai.com/v1/responses (Responses API), "
                           "called directly: no router, no provider fallback", "members": {}}
    led = Ledger()
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                              "required": ["city"]}}},
             {"type": "function", "function": {"name": "finish", "description": "Finish with an answer",
                                               "parameters": {"type": "object", "properties": {"answer": {"type": "string"}},
                                                              "required": ["answer"]}}}]
    msgs = [{"role": "system", "content": "You are a terse assistant. Use tools when they apply."},
            {"role": "user", "content": "What is the weather in Paris right now?"}]
    for role, (model, effort, temp) in {"X": (SOL, "low", None), "X-small": (LUNA, "none", 0)}.items():
        m: dict[str, Any] = {"id": model, "listed": model in listed,
                             "created": listed.get(model, {}).get("created"),
                             "dated_snapshots_listed": sorted(i for i in listed if i.startswith(model + "-")),
                             "prices_per_M": dict(zip(("input", "output", "cached_input"), FROZEN_PRICES[model]),
                                                  cache_write=FROZEN_PRICES[model][0] * CACHE_WRITE_FACTOR),
                             "price_source": ("A7.12 (i) frozen price" if model == SOL else
                                              "configs/prices/models.yaml (verified 2026-10-01)")
                             + "; pricing page re-read 2026-10-05 (standard tier, short context): input / cached input "
                               "/ cache writes / output as listed here; reasoning tokens are billed as output tokens "
                               "(usage.output_tokens includes them); long-context (> 272k) rates do not arise"}
        eff = {}
        for e in ("none", "minimal", "low"):
            rr = httpx.post("https://api.openai.com/v1/responses", headers=H, timeout=60,
                            json={"model": model, "input": "Say OK.", "store": False, "reasoning": {"effort": e},
                                  "max_output_tokens": 16})
            j = rr.json() if rr.headers.get("content-type", "").startswith("application/json") else {}
            eff[e] = {"status": rr.status_code, "served_model": j.get("model"),
                      "error": ((j.get("error") or {}).get("message") or "")[:200] or None,
                      "rate_limits": {h: rr.headers.get(h) for h in ("x-ratelimit-limit-requests",
                                                                     "x-ratelimit-limit-tokens") if rr.headers.get(h)}}
            if rr.status_code == 200:
                u = j.get("usage") or {}
                pi, po, pc = FROZEN_PRICES[model]
                led_cost = (int(u.get("input_tokens") or 0) * pi + int(u.get("output_tokens") or 0) * po) / 1e6
                out.setdefault("_raw_cost", 0.0)
                out["_raw_cost"] += led_cost
        m["reasoning_efforts"] = eff
        spec = {"provider": "openai_responses", "model": model, "reasoning_effort": effort}
        if temp is not None:
            spec["temperature"] = temp
        llm = make_llm(spec)
        checks: dict[str, Any] = {}
        a = llm.chat(msgs, tools=tools, ledger=led, decision_point="probe")
        checks["tool_call_auto"] = {"ok": bool(a.tool_calls) and a.tool_calls[0].name == "get_weather",
                                    "call": [(t.name, t.arguments) for t in a.tool_calls]}
        if a.tool_calls:
            tc = a.tool_calls[0]
            m2 = msgs + [{"role": "assistant", "content": None, "tool_calls": [tc.to_openai()]},
                         {"role": "tool", "tool_call_id": tc.id, "content": "18C, light rain"}]
            b = llm.chat(m2, tools=tools, ledger=led, decision_point="probe",
                         tool_choice={"type": "function", "function": {"name": "finish"}})
            checks["round_trip_forced_tool"] = {
                "ok": bool(b.tool_calls) and b.tool_calls[0].name == "finish",
                "reasoning_items_echoed": len(((tc.extra or {}).get("openai") or {}).get("reasoning") or []),
                "call": [(t.name, t.arguments) for t in b.tool_calls]}
        f = llm.chat([{"role": "user", "content": "Give the capital of France."}], ledger=led, decision_point="probe",
                     response_format={"type": "json_schema", "json_schema": {
                         "name": "answer", "strict": True,
                         "schema": {"type": "object", "properties": {"capital": {"type": "string"}},
                                    "required": ["capital"], "additionalProperties": False}}})
        try:
            parsed = json.loads(f.text)
        except json.JSONDecodeError:
            parsed = None
        checks["structured_output_json_schema"] = {"ok": isinstance(parsed, dict) and "capital" in parsed,
                                                   "text": f.text[:80]}
        ch = make_llm({**spec, "server_state": True})
        cm = [{"role": "system", "content": "Answer in one short sentence."}, {"role": "user", "content": "Name a prime."}]
        r1 = ch.chat(cm, ledger=led, decision_point="probe")
        cm += [r1.assistant_message(), {"role": "user", "content": "And the next one after it?"}]
        ch.chat(cm, ledger=led, decision_point="probe")
        checks["previous_response_id_chaining"] = {"ok": ch.stats["chained_calls"] == 1,
                                                   "chained_calls": ch.stats["chained_calls"],
                                                   "used_in_runs": False}
        m["checks"] = checks
        m["dropped_params"] = dict(llm.stats.get("dropped_params") or {})
        out["members"][role] = m
    served = sorted({str((c.meta or {}).get("served_model")) for c in led.records})
    tiers = sorted({str((c.meta or {}).get("service_tier")) for c in led.records})
    probe_cost = sum(c.cost_usd + surcharge(c.model, (c.meta or {}).get("cache_write_tokens") or 0)
                     for c in led.records) + float(out.pop("_raw_cost", 0.0))
    out.update({
        "served_models_seen": served, "service_tiers_seen": tiers, "probe_calls": len(led.records),
        "probe_cost_usd": probe_cost,
        "pin": "the model id itself: /v1/models lists no dated snapshot of either id and the API reports the served "
               "model per response (ledger meta served_model); a served model other than the id during the runs is "
               "a deviation; no provider routing exists, so no fallback can occur",
        "request_settings": {
            SOL: {"reasoning_effort": "low (the lowest it accepts: none and minimal are rejected with a 400; stands for "
                                      "A7.12 (a)'s 'reasoning off or minimal')",
                  "temperature": "not sent (the API rejects sampling parameters at an effort other than none)",
                  "store": False, "server_state": False},
            LUNA: {"reasoning_effort": "none", "temperature": 0, "store": False, "server_state": False},
            "source": "configs/dev/sol_eligibility.yaml, configs/experiments/jevgym_shadow_cal_sol.yaml, "
                      "configs/experiments/stage2_sol.yaml"},
        "harness_needs": "System-2 native function calling with tool_choice auto and a forced named tool (tool_params "
                         "fill, final tool), the reasoning items echoed between a call and its output; no structured "
                         "output and no chaining are used by the System-2 loop",
        "sources": [f"https://api.openai.com/v1/models (read {now})",
                    "https://developers.openai.com/api/docs/pricing (read 2026-10-05)"]})
    out["eligible_capabilities"] = all(c.get("ok") for mm in out["members"].values()
                                       for k, c in mm["checks"].items() if k != "previous_response_id_chaining")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--write", action="store_true")
    p.add_argument("--out", default=str(MANIFEST))
    g = sub.add_parser("gate")
    g.add_argument("--run", default=str(SMOKE_RUN))
    g.add_argument("--manifest", default=str(MANIFEST))
    g.add_argument("--write", action="store_true")
    g.add_argument("--out", default=str(OUT))
    s = sub.add_parser("spend")
    s.add_argument("--cap", type=float, default=HARD_CAP)
    w = sub.add_parser("watch")
    w.add_argument("--cap", type=float, default=HARD_CAP)
    w.add_argument("--reserve", type=float, default=1.5, help="$ kept free for the episodes in flight")
    w.add_argument("--kill-pattern", default=None, help="processes whose command line contains it")
    w.add_argument("--pgid", type=int, default=None, help="the run's process group (setsid)")
    w.add_argument("--interval", type=float, default=60.0)
    w.add_argument("--log", default=str(ROOT / "runs/stage2_logs/sol_spend.jsonl"))
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        res = probe()
        print(json.dumps({k: res[k] for k in ("eligible_capabilities", "served_models_seen", "probe_cost_usd")}))
        if a.write:
            Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        return 0
    if a.cmd == "gate":
        man = Path(a.manifest)
        probe_usd = float(json.loads(man.read_text()).get("probe_cost_usd") or 0.0) if man.exists() else 0.0
        rows = {arm: member(Path(a.run) / f"{arm}.jsonl", model) for arm, model in MEMBERS.items()}
        res = {"registered": "PREREG_A7_STAGE2 A7.12 (i) with A7.12 (b), (c)",
               "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "members": rows,
               **gate(rows, probe_usd)}
        print(json.dumps({k: res[k] for k in ("eligible", "predicted_usd", "n", "decision")}, indent=1))
        if a.write:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        return 0
    if a.cmd == "spend":
        res = spend()
        print(json.dumps(res, indent=1))
        return 3 if res["total_usd"] > a.cap else 0
    if not (a.kill_pattern or a.pgid):
        ap.error("watch needs --kill-pattern and / or --pgid")
    return watch(a.cap, a.reserve, a.kill_pattern, a.pgid, a.interval, Path(a.log))


if __name__ == "__main__":
    sys.exit(main())
