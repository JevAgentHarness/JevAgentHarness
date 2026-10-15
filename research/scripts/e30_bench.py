#!/usr/bin/env python3
r"""E30 host-side bench (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 "E30: every open decision model at its cheapest
common deployment"; exploratory, descriptive): every open decision model on ONE engine and ONE instance type
(vLLM in process on one L4), saturated, with each model's own read-out on E9's 3,991 L1 decisions.

    # harness venv (jevharness importable; --vendor = the StartLux package at the pinned commit)
    python scripts/e30_bench.py order --out ORDER.json
    python scripts/e30_bench.py prompts --family code|startlux|intern --model DIR --order ORDER.json --out P.jsonl.gz
    python scripts/e30_bench.py time-wire --url URL --model NAME --order ORDER.json --out R.json.gz
    # vLLM venv (vLLM 0.31.0)
    python scripts/e30_bench.py time-vllm --model DIR --prompts P --order ORDER.json --precision bf16|fp8 --out R.json.gz
    # harness venv: llama.cpp's server (the 27B Q4_K_M GGUF), the vendor's read-out against llama-server
    python scripts/e30_bench.py time-llama --url URL --model-dir DIR --prompts P --order ORDER.json --slots 8 --out R.json.gz

Families (registration (d)): ``code`` = our trained models (serve_jev's prompt, the option codes' logits at the last
position); ``startlux`` = the vendor's ``jevfmt`` rendering, the option letters' logits, the per-type temperature, the
vendor's own wide-choice code for > 26 options; ``intern`` = the vendored Intern-Decision compiler on E1's (state,
decision point) groups, one read per field (the prompt cut right after the position before its placeholder). Every
read is one forward pass of the prompt with one sampled token restricted to the read's codes; the codes' logits come
back as vLLM's processed logits (allowed-token mask, greedy), so nothing is generated beyond the read.

The timing output holds, per decision, the raw code logits of each read (or the finished wire answer for a wide
choice / the scoring-head row) plus what the readout needs to turn them into the harness's prediction
(``scripts/explore_e30_cost.py``), and the pass timings: warm-up (first WARMUP decisions, untimed), saturated passes
(C in flight, the prefix cache reset before each), and C = 1 latency over the first LAT_N decisions."""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
E1_CONFIG = ROOT / "configs/decision/bfcl_when2call_rl.yaml"   # E1's pools, exclusions, batching (as E9's rows)
ORDER_SEED = 0
WARMUP = 200                 # registration (f): the first 200 decisions of the order, untimed
LAT_N = 1000                 # registration (f): C = 1 latency over the first 1,000 decisions of the order
SAT_C = (64, 256)            # registration (f): saturation passes (vLLM); answers from the first
LLAMA_SLOTS = (8, 16)        # registration (f): llama.cpp parallel slots (C = P)
WIRE_C = (16, 64)            # registration (f): the scoring-head row (requests = E1's groups in flight)
MAX_LEN = 4096               # registration (c): maximum model length (8,192 if a prompt is longer)
GPU_MEM = 0.92               # registration (c)
# Capacity (never an answer): vLLM's defaults on this GPU are 2,048 batched tokens per step and 128 sequences (the
# smoke's engine record), which leave a saturated prefill-only workload step-bound; set before any measurement.
BATCH_TOKENS = 8192
INTERN_MAX_Q = 16            # decision/intern_decision.py: fields per prompt


# ================================================================================================ io helpers
def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj)
    if str(p).endswith(".gz"):
        with gzip.open(p, "wt") as f:
            f.write(data)
    else:
        p.write_text(data + "\n")


def read_json(path: str | Path) -> Any:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(Path(path).read_text())


def shuffled_order(ids: Sequence[str], seed: int = ORDER_SEED) -> list[str]:
    """Registration (f): ``random.Random(0).shuffle`` of the sorted item ids."""
    out = sorted(ids)
    random.Random(seed).shuffle(out)
    return out


def pct(xs: Sequence[float], q: float) -> float:
    """The q-quantile (linear interpolation, numpy's default) of a non-empty list."""
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def lat_summary(ms: Sequence[float]) -> dict[str, Any]:
    return {"n": len(ms), "p50_ms": statistics.median(ms) if ms else None, "p90_ms": pct(ms, 0.9) if ms else None,
            "mean_ms": statistics.mean(ms) if ms else None}


# ================================================================================================ E1 items and order
def e1_items() -> list[Any]:
    """E1's 3,991 L1 decisions in E1's load order (decision_eval.load_population: pools, then the dev-prefix exclusion)."""
    sys.path.insert(0, str(ROOT / "src"))
    import yaml

    from jevharness.bench.decision_eval import load_population

    return load_population(yaml.safe_load(E1_CONFIG.read_text()))


def e1_groups(items: Sequence[Any]) -> list[list[Any]]:
    """decision_eval._s1_records' batch_by_state groups: (state, decision point), in order of first appearance."""
    groups: dict[tuple[str, str], list[Any]] = {}
    for it in items:
        groups.setdefault((it.state, it.dp), []).append(it)
    return list(groups.values())


def cmd_order(a: argparse.Namespace) -> int:
    items = e1_items()
    ids = [it.id for it in items]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate E1 item ids")
    gid = {}
    for g, grp in enumerate(e1_groups(items)):
        for it in grp:
            gid[it.id] = g
    order = shuffled_order(ids)
    out = {"n": len(order), "seed": ORDER_SEED, "order": order, "group": gid, "warmup": WARMUP, "lat_n": LAT_N,
           "groups": len(set(gid.values()))}
    write_json(a.out, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("order", "group")}))
    return 0


# ================================================================================================ prompts (harness venv)
def prompts_code(model: str, items: Sequence[Any]) -> Iterable[dict[str, Any]]:
    """serve_jev's prompt token for token (scripts/e12_fast_engine.py prompts): one read per question at the last
    position over the option codes; the prediction is the label of the most likely code."""
    from transformers import AutoTokenizer

    from jevharness.bench.suites.static import norm
    from jevharness.training.data import render
    from jevharness.training.decision_model import DecisionCollator, code_token_ids
    from jevharness.training.grpo_online import question_to_example

    tok = AutoTokenizer.from_pretrained(model)
    col = DecisionCollator(tok, max_len=MAX_LEN)
    codes = code_token_ids(tok)
    for it in items:
        ex = question_to_example(it.state, it.question)
        if len(ex["options"]) > len(codes):
            raise SystemExit(f"{it.id}: {len(ex['options'])} options exceed the {len(codes)} codes (knockout)")
        ids = col.encode(render({"state": ex["state"], "question": ex["question"], "options": ex["options"]})[0])
        yield {"id": it.id, "kind": "single", "reads": [{"ids": ids, "allowed": codes[: len(ex["options"])]}],
               "meta": {"labels_norm": [norm(x) for x in ex["labels"]]}}


def _vendor(vendor: str | None) -> Any:
    if vendor:
        sys.path.insert(0, vendor)
    from startlux_decision import jevfmt as J  # noqa: E402

    return J


def prompts_startlux(model: str, items: Sequence[Any], vendor: str | None) -> Iterable[dict[str, Any]]:
    """The vendor's rendering (jevfmt at the pinned commit): one prompt per question, the option letters at the last
    position; a choice with one option is answered without the model and a choice wider than MAX_OPTIONS goes to the
    vendor's wide-choice code at timing (startlux_decision.model.StartLuxDecision._split)."""
    from transformers import AutoTokenizer

    from jevharness.decision.jev import question_to_wire

    J = _vendor(vendor)
    tok = AutoTokenizer.from_pretrained(model)
    letters = J.check_tokenizer(tok)
    cfg = json.loads((Path(model) / "decision_config.json").read_text())
    if letters != cfg["letter_token_ids"]:
        raise SystemExit("tokenizer letter ids differ from decision_config.json")
    temp = {k: float(v) for k, v in cfg["temperature_by_type"].items()}
    for it in items:
        q = question_to_wire(it.question)
        t = q.get("type", "choice")
        crit = q.get("criteria") or {}
        if t == "choice" and isinstance(crit, dict) and len(crit) == 1:
            only = next(iter(crit))
            yield {"id": it.id, "kind": "answer", "answer": {"type": "choice", "choice": only, "confidence": 1.0,
                                                              "probabilities": {only: 1.0}}, "reads": []}
        elif t == "choice" and isinstance(crit, dict) and len(crit) > J.MAX_OPTIONS:
            yield {"id": it.id, "kind": "wide", "state": it.state, "spec": q, "reads": []}
        else:
            row = J.from_systemone(it.state, q)
            order = [o["id"] for o in row["options"]]
            ids, _ = J.render_ids(row, tok, order, max_length=65536)
            yield {"id": it.id, "kind": "single", "reads": [{"ids": ids, "allowed": letters[: len(order)]}],
                   "meta": {"opt_ids": order, "type": row["type"], "T": temp.get(row["type"], 1.0)}}


OPENJEV_LETTERS = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)]   # helper/shim.py LETTERS
OPENJEV_T = 0.85             # E15's READOUT_T (scripts/aws/e15_baselines.sh)


def openjev_prompt(state: str, instructions: str, options: Sequence[tuple[str, str]]) -> str:
    """The card's helper text lane (helper/shim.py at ac97900f: _readout_once without an image, prefix_text without
    padding): the state, the question and the lettered options as one user message."""
    lines = "\n".join(f"[{OPENJEV_LETTERS[i]}] {k}: {d}" for i, (k, d) in enumerate(options))
    return f"State:\n{state}" + f"\n\nQuestion: {instructions}\nOptions:\n{lines}\n\nAnswer with the letter of the best option only."


def prompts_openjev(model: str, items: Sequence[Any]) -> Iterable[dict[str, Any]]:
    """OpenJev-27B (E30 amendment "OpenJev-27B through its official 4-bit GGUF"): the helper's prompt through the
    checkpoint's chat template with thinking off, tokenized without added special tokens (vLLM's chat endpoint), and the
    option letters' token ids (shim.letter_ids: the first token of each letter)."""
    from transformers import AutoTokenizer

    from jevharness.decision.jev import question_to_wire

    tok = AutoTokenizer.from_pretrained(model)
    ids_of = {L: tok.encode(L, add_special_tokens=False)[0] for L in OPENJEV_LETTERS}
    for it in items:
        q = question_to_wire(it.question)
        crit = q.get("criteria")
        if q.get("type") != "choice" or not isinstance(crit, dict) or not crit or len(crit) > 52:
            raise SystemExit(f"{it.id}: not a choice of 1..52 options")
        instr = q.get("instructions")
        instr = instr if isinstance(instr, str) else json.dumps(instr, ensure_ascii=False)
        opts = [(str(k), "" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)))
                for k, v in crit.items()]
        text = tok.apply_chat_template([{"role": "user", "content": openjev_prompt(it.state, instr, opts)}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        yield {"id": it.id, "kind": "single",
               "reads": [{"ids": ids, "allowed": [ids_of[OPENJEV_LETTERS[i]] for i in range(len(opts))]}],
               "meta": {"opt_ids": [k for k, _ in opts], "T": OPENJEV_T, "round": 4}}


def prompts_decider(model: str, items: Sequence[Any], code_dir: str | None) -> Iterable[dict[str, Any]]:
    """decider-4b (E30 amendment "E15's open baselines"): its own row construction at the checkpoint's revision
    (decider.systemone.render_question / plan_rows and decider.prompt_fast.build_rows, the plain layout of
    decider_config.json, every question its own row as its server's default `independent`), read at the row's slot (the
    last position) over the option labels (decider.prompt.letter_ids, the rows its engine reads from the LM head);
    softmax at the choice temperature, probabilities rounded to 4 decimals as decider.systemone.format_answer returns
    them."""
    from transformers import AutoTokenizer

    from jevharness.decision.jev import question_to_wire

    sys.path.insert(0, code_dir or model)
    from decider import systemone as S1
    from decider.prompt import letter_ids
    from decider.prompt_fast import build_rows

    tok = AutoTokenizer.from_pretrained(model)
    cfg = json.loads((Path(model) / "decider_config.json").read_text())
    if cfg.get("layout", "plain") != "plain" or cfg.get("schema_first"):
        raise SystemExit("only the plain state-first layout is ported")
    temp = float((cfg.get("temperature_by_type") or {}).get("choice", cfg.get("temperature", 1.0)))
    letters = letter_ids(tok)
    for it in items:
        rq = S1.render_question(question_to_wire(it.question))
        flat, _ = S1.plan_rows({"q": rq}, bool(cfg.get("isolated_levels", False)))
        rows, _ = build_rows(tok, S1.render_state(it.state), [[(r["question"], list(r["options"]))] for r in flat],
                             max_ctx_tokens=int(cfg.get("max_state_tokens", 32768)))
        if len(rows) != 1 or rows[0]["slots"] != [len(rows[0]["ids"]) - 1]:
            raise SystemExit(f"{it.id}: not one row read at its last position")
        n = rows[0]["nopts"][0]
        yield {"id": it.id, "kind": "single", "reads": [{"ids": rows[0]["ids"], "allowed": letters[:n]}],
               "meta": {"opt_ids": [str(x) for x in rq["names"]], "T": temp, "round": 4}}


def prompts_jevk5(model: str, items: Sequence[Any], code_dir: str | None) -> Iterable[dict[str, Any]]:
    """JevK5 (E30 amendment "E15's open baselines"): the jevk5 package 0.2.2's prompt (SemIf's JSON payload through the
    chat template, thinking off), the answer letters' logits at the last position over at most 16 options, softmax at
    jevk5_config.json's temperature; a choice of more than 16 options is read by the package's own knockout at timing."""
    from transformers import AutoTokenizer

    from jevharness.decision.jev import question_to_wire

    if code_dir:
        sys.path.insert(0, code_dir)
    from jevk5.prompt import LETTERS, decision_options, messages

    tok = AutoTokenizer.from_pretrained(model)
    temp = float(json.loads((Path(model) / "jevk5_config.json").read_text()).get("temperature", 1.0))
    slots = []
    for L in LETTERS:
        e = tok.encode(L, add_special_tokens=False)
        if len(e) != 1:
            raise SystemExit("an answer letter is not one token")
        slots.append(e[0])
    for it in items:
        q = question_to_wire(it.question)
        opts = decision_options(q)
        if len(opts) > len(LETTERS):
            yield {"id": it.id, "kind": "wide_jevk5", "state": it.state, "spec": q, "reads": [],
                   "meta": {"opt_ids": [k for k, _ in opts], "T": temp}}
            continue
        text = tok.apply_chat_template(messages(it.state, q["instructions"], [t for _, t in opts]), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        yield {"id": it.id, "kind": "single",
               "reads": [{"ids": tok.encode(text, add_special_tokens=False), "allowed": slots[: len(opts)]}],
               "meta": {"opt_ids": [k for k, _ in opts], "T": temp}}


def prompts_intern(model: str, items: Sequence[Any]) -> Iterable[dict[str, Any]]:
    """The vendored compiler and encoder of decision/intern_decision.py (InternDecisionBackend._encode, its
    truncation included) on E1's groups, at most 16 fields per prompt (_score_many); field j is read at
    ``positions[j]`` (the position before its placeholder), so its read is the prompt cut after that position."""
    from transformers import AutoTokenizer

    from jevharness.decision import intern_decision as I

    tok = AutoTokenizer.from_pretrained(model)
    if I.DECISION_TOKEN not in tok.get_added_vocab():
        raise SystemExit("no decision token in the tokenizer")
    enc = object.__new__(I.InternDecisionBackend)      # the encoder only: no weights are loaded
    enc.tok = tok
    enc.max_length = 8192
    enc.truncate_state = True
    enc.state_json = True
    enc.marker_id = int(tok.convert_tokens_to_ids(I.DECISION_TOKEN))
    enc.n_truncated = 0
    symbol_ids = {}
    for s in I.ANSWER_SYMBOLS:
        e = tok.encode(s, add_special_tokens=False)
        if len(e) != 1:
            raise SystemExit(f"answer symbol {s!r} is not one token")
        symbol_ids[s] = int(e[0])
    temp, src = I.checkpoint_temperature(model)
    if temp is None:
        raise SystemExit("no calibration temperature for the checkpoint")
    for grp in e1_groups(items):
        qs = [it.question for it in grp]
        names = I.field_names(qs)
        for i in range(0, len(qs), INTERN_MAX_Q):
            chunk, cnames, citems = qs[i:i + INTERN_MAX_Q], names[i:i + INTERN_MAX_Q], grp[i:i + INTERN_MAX_Q]
            compiled, wq, ids, positions, truncated = enc._encode(grp[0].state, chunk, cnames)
            row = ids[0].tolist()
            for j, (it, name) in enumerate(zip(citems, compiled.fields)):
                pos = int(positions[j])
                values = [v for v, _ in I._options(wq[name])]
                yield {"id": it.id, "kind": "single",
                       "reads": [{"ids": row[: pos + 1], "allowed": [symbol_ids[s] for s in compiled.symbols[name]]}],
                       "meta": {"values": values, "T": temp, "T_source": src, "field": name, "fields": len(chunk),
                                "truncated": bool(truncated)}}


def cmd_prompts(a: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    items = e1_items()
    if a.family == "code":
        rows = list(prompts_code(a.model, items))
    elif a.family == "startlux":
        rows = list(prompts_startlux(a.model, items, a.vendor))
    elif a.family == "intern":
        rows = list(prompts_intern(a.model, items))
    elif a.family == "openjev":
        rows = list(prompts_openjev(a.model, items))
    elif a.family == "decider":
        rows = list(prompts_decider(a.model, items, a.vendor))
    elif a.family == "jevk5":
        rows = list(prompts_jevk5(a.model, items, a.vendor))
    else:
        raise SystemExit(f"unknown family {a.family}")
    if {r["id"] for r in rows} != {it.id for it in items} or len(rows) != len(items):
        raise SystemExit("the prompts do not cover E1's decisions exactly once")
    lens = [len(rd["ids"]) for r in rows for rd in r["reads"]]
    write_jsonl(a.out, rows)
    info = {"family": a.family, "model": a.model, "decisions": len(rows), "reads": len(lens),
            "kinds": {k: sum(1 for r in rows if r["kind"] == k) for k in ("single", "wide", "answer", "wide_jevk5")},
            "mean_tokens": statistics.mean(lens) if lens else None, "max_tokens": max(lens) if lens else None,
            "max_allowed": max((len(rd["allowed"]) for r in rows for rd in r["reads"]), default=0)}
    print(json.dumps(info))
    return 0


# ================================================================================================ timing core
def limited(order: Sequence[str], limit: int | None) -> tuple[list[str], int, int]:
    """(order, warm-up, latency n): the registered values, or a smoke's first ``limit`` decisions (never a result)."""
    if not limit:
        return list(order), WARMUP, LAT_N
    return list(order[:limit]), max(1, min(WARMUP, limit // 5)), max(1, min(LAT_N, limit // 2))


async def run_passes(decide: Callable[[dict[str, Any]], Any], rows: Sequence[dict[str, Any]], order: Sequence[str],
                     sat_c: Sequence[int], reset: Callable[[], Any] | None, warmup: int = WARMUP,
                     lat_n: int = LAT_N) -> dict[str, Any]:
    """Warm-up (untimed), one saturated pass per C (prefix cache reset before each) and the C = 1 latency pass.
    ``decide(row)`` is a coroutine returning the decision's answer record (without the id)."""
    by_id = {r["id"]: r for r in rows}
    seq = [by_id[i] for i in order]

    async def one(r: dict[str, Any]) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        ans = await decide(r)
        return ans, (time.perf_counter() - t0) * 1000

    async def pass_(rs: Sequence[dict[str, Any]], c: int) -> tuple[dict[str, Any], list[float], float]:
        sem = asyncio.Semaphore(c)
        answers: dict[str, Any] = {}
        lats: list[float] = []

        async def task(r: dict[str, Any]) -> None:
            async with sem:
                ans, ms = await one(r)
                answers[r["id"]] = ans
                lats.append(ms)

        t0 = time.perf_counter()
        await asyncio.gather(*(task(r) for r in rs))
        return answers, lats, time.perf_counter() - t0

    async def do_reset() -> None:
        if reset is not None:
            res = reset()
            if asyncio.iscoroutine(res):
                await res

    out: dict[str, Any] = {"order_n": len(seq), "passes": []}
    t0 = time.perf_counter()
    await pass_(seq[:warmup], max(sat_c))
    out["warmup_s"] = time.perf_counter() - t0
    first_answers = None
    for c in sat_c:
        await do_reset()
        answers, lats, wall = await pass_(seq, c)
        out["passes"].append({"c": c, "decisions": len(seq), "wall_s": wall, "dps": len(seq) / wall,
                              "latency": lat_summary(lats)})
        if first_answers is None:
            first_answers = answers
            out["answers_c"] = c
        else:
            out["passes"][-1]["agree_with_first"] = sum(
                1 for i in answers if _same(answers[i], first_answers.get(i))) / len(answers)
    if lat_n > 0:
        await do_reset()
        _, lats, wall = await pass_(seq[:lat_n], 1)
        out["c1"] = {"c": 1, "decisions": min(lat_n, len(seq)), "wall_s": wall, **lat_summary(lats)}
    best = max(out["passes"], key=lambda p: p["dps"])
    out["saturated"] = {"c": best["c"], "dps": best["dps"], "wall_s": best["wall_s"]}
    out["answers"] = first_answers
    return out


def _argmax(xs: Sequence[float]) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


def _same(a: Any, b: Any) -> bool:
    """The same decision in two passes: the same most likely code of every read, or the same wire answer choice."""
    if a is None or b is None:
        return False
    if "logits" in a and "logits" in b:
        return [_argmax(x) for x in a["logits"]] == [_argmax(x) for x in b["logits"]]
    return (a.get("answer") or {}).get("choice") == (b.get("answer") or {}).get("choice") and \
        a.get("pred") == b.get("pred")


def softmax(z: Sequence[float], t: float = 1.0) -> list[float]:
    m = max(z)
    e = [math.exp((v - m) / t) for v in z]
    s = sum(e)
    return [v / s for v in e]


def make_wide(model_dir: str, vendor: str | None, read_sync: Callable[[list[int], list[int]], list[float]]) -> Any:
    """The vendor's StartLuxDecision with only the forward pass replaced (as the vendor's gguf_server does): prompt
    rendering, temperatures and the wide-choice rule are the package's own code."""
    _vendor(vendor)
    import torch
    from transformers import AutoTokenizer

    from startlux_decision import jevfmt as J
    from startlux_decision.model import StartLuxDecision

    class Remote(StartLuxDecision):
        def __init__(self, path: str):                    # no torch model: the engine runs the weights
            cfg = json.load(open(os.path.join(path, "decision_config.json")))
            self.tok = AutoTokenizer.from_pretrained(path)
            self.letters = J.check_tokenizer(self.tok)
            if self.letters != cfg["letter_token_ids"]:
                raise ValueError("tokenizer letter ids differ from decision_config.json")
            self.temperature = {k: float(v) for k, v in cfg["temperature_by_type"].items()}
            wide = cfg.get("wide_choice", {})
            self.group, self.keep, self.residual = wide.get("group", 25), wide.get("keep", 3), wide.get("residual", 1e-3)
            self.max_length, self.graphs = 65536, {}
            self.reads = 0

        def _logits(self, rows: list[dict[str, Any]]) -> tuple[list[Any], int]:
            out, tokens = [], 0
            for r in rows:
                order = [o["id"] for o in r["options"]]
                ids, _ = J.render_ids(r, self.tok, order, max_length=self.max_length)
                out.append(torch.tensor(read_sync(ids, self.letters[: len(order)]), dtype=torch.float32))
                tokens += len(ids)
                self.reads += 1
            return out, tokens

    return Remote(model_dir)


def jevk5_wide(model_dir: str, code_dir: str | None, read_sync: Callable[[list[int], list[int]], list[float]]
               ) -> Callable[[Any, dict[str, Any]], dict[str, Any]]:
    """JevK5Runtime.probabilities + answer with only the forward pass replaced: the package's own spread (knockout)."""
    from transformers import AutoTokenizer

    if code_dir:
        sys.path.insert(0, code_dir)
    from jevk5.prompt import LETTERS, answer, decision_options, messages, spread

    tok = AutoTokenizer.from_pretrained(model_dir)
    temp = float(json.loads((Path(model_dir) / "jevk5_config.json").read_text()).get("temperature", 1.0))
    slots = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    reads = [0]

    def decide(state: Any, q: dict[str, Any]) -> dict[str, Any]:
        options = decision_options(q)

        def read(texts: list[str]) -> list[float]:
            text = tok.apply_chat_template(messages(state, q["instructions"], texts), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            z = read_sync(tok.encode(text, add_special_tokens=False), slots[: len(texts)])
            reads[0] += 1
            return softmax([v / temp for v in z])

        probs = spread(read, [t for _, t in options], "knockout")
        return answer(q, {k: v for (k, _), v in zip(options, probs)}, 0)

    decide.reads = reads  # type: ignore[attr-defined]
    return decide


def wide_answer(dec: Any, state: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """StartLuxDecision._split's wide branch: the vendor's _wide, then the answer as the vendor builds it."""
    p, _ = dec._wide(state, spec)
    best = max(p, key=p.get)
    return {"type": "choice", "choice": best, "confidence": p[best], "probabilities": p}


# ------------------------------------------------------------------------------------------------ vLLM
def cmd_time_vllm(a: argparse.Namespace) -> int:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")      # as E12c (h): no FlashInfer JIT (no nvcc)
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    rows = read_jsonl(a.prompts)
    order, warm, lat_n = limited(read_json(a.order)["order"], a.limit)
    longest = max((len(rd["ids"]) for r in rows for rd in r["reads"]), default=1)
    max_len = MAX_LEN if longest + 8 <= MAX_LEN else 8192
    cfg = json.loads((Path(a.model) / "config.json").read_text())
    mm = "vision_config" in cfg
    base = dict(model=a.model, dtype="bfloat16", enable_prefix_caching=True, max_model_len=max_len,
                gpu_memory_utilization=GPU_MEM, seed=0, max_logprobs=64, logprobs_mode="processed_logits",
                max_num_batched_tokens=BATCH_TOKENS)
    if a.precision == "fp8":
        base["quantization"] = "fp8"
    elif a.precision != "bf16":
        raise SystemExit(f"unknown precision {a.precision}")
    if mm:
        base["limit_mm_per_prompt"] = {"image": 0, "video": 0}       # language model only
    engine, used, errors = None, None, []
    t_load = time.perf_counter()
    for extra in ({"max_num_seqs": 256}, {"max_num_seqs": 128}, {"max_num_seqs": 64}):   # capacity only, never an answer
        try:
            engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**base, **extra))
            used = extra
            break
        except Exception as e:  # noqa: BLE001 - recorded; the next capacity is tried
            errors.append(f"{type(e).__name__}: {str(e)[:400]}")
    if engine is None:
        write_json(a.out, {"state": "did not load", "model": a.model, "precision": a.precision, "errors": errors,
                           "vllm": vllm.__version__})
        print(json.dumps({"state": "did not load", "errors": errors}))
        return 2
    load_s = time.perf_counter() - t_load
    info: dict[str, Any] = {}
    try:
        vc = engine.vllm_config
        info = {"enable_prefix_caching": vc.cache_config.enable_prefix_caching,
                "max_num_seqs": vc.scheduler_config.max_num_seqs,
                "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
                "quantization": vc.model_config.quantization, "dtype": str(vc.model_config.dtype),
                "max_model_len": vc.model_config.max_model_len,
                "logprobs_mode": str(getattr(vc.model_config, "logprobs_mode", None)),
                "num_gpu_blocks": getattr(vc.cache_config, "num_gpu_blocks", None)}
    except Exception as e:  # noqa: BLE001
        info = {"error": repr(e)[:300]}
    loop = asyncio.new_event_loop()
    rid = [0]
    reads = [0]

    async def read(ids: list[int], allowed: list[int]) -> list[float]:
        rid[0] += 1
        reads[0] += 1
        sp = SamplingParams(max_tokens=1, temperature=0.0, allowed_token_ids=list(allowed), logprobs=len(allowed))
        final = None
        async for o in engine.generate(TokensPrompt(prompt_token_ids=list(ids)), sp, request_id=f"r{rid[0]}"):
            final = o
        lp = final.outputs[0].logprobs[0]
        vals = [float(lp[t].logprob) for t in allowed]
        if not all(math.isfinite(v) for v in vals):
            raise RuntimeError("a code's processed logit is not finite")
        return vals

    def read_sync(ids: list[int], allowed: list[int]) -> list[float]:
        return asyncio.run_coroutine_threadsafe(read(ids, allowed), loop).result()

    wide_dec = None
    if any(r["kind"] == "wide" for r in rows):
        wide_dec = make_wide(a.model, a.vendor, read_sync)
    k5 = jevk5_wide(a.model, a.vendor, read_sync) if any(r["kind"] == "wide_jevk5" for r in rows) else None

    async def decide(r: dict[str, Any]) -> dict[str, Any]:
        if r["kind"] == "answer":
            return {"answer": r["answer"]}
        if r["kind"] == "wide":
            return {"answer": await asyncio.to_thread(wide_answer, wide_dec, r["state"], r["spec"])}
        if r["kind"] == "wide_jevk5":
            return {"answer": await asyncio.to_thread(k5, r["state"], r["spec"])}
        return {"logits": [await read(rd["ids"], rd["allowed"]) for rd in r["reads"]]}

    async def probe() -> dict[str, Any]:
        r = next(x for x in rows if x["kind"] == "single")
        rd = r["reads"][0]
        sp = SamplingParams(max_tokens=1, temperature=0.0, allowed_token_ids=list(rd["allowed"]),
                            logprobs=len(rd["allowed"]))
        final = None
        async for o in engine.generate(TokensPrompt(prompt_token_ids=list(rd["ids"])), sp, request_id="probe"):
            final = o
        lp = final.outputs[0].logprobs[0]
        got = {int(k): float(v.logprob) for k, v in lp.items()}
        tok = final.outputs[0].token_ids[0]
        vals = [got.get(t) for t in rd["allowed"]]
        ok = all(v is not None and math.isfinite(v) for v in vals) and tok in rd["allowed"] and \
            rd["allowed"][_argmax(vals)] == tok
        return {"ok": bool(ok), "returned": len(got), "allowed": len(rd["allowed"]), "sampled_is_argmax": ok}

    async def main() -> dict[str, Any]:
        pr = await probe()
        if not pr["ok"]:
            raise RuntimeError(f"the read-out probe failed: {pr}")
        res = await run_passes(decide, rows, order, a.conc, engine.reset_prefix_cache, warm, lat_n)
        res["probe"] = pr
        return res

    try:
        res = loop.run_until_complete(main())
    finally:
        try:
            engine.shutdown()
        except Exception:  # noqa: BLE001
            pass
    meta = {r["id"]: r.get("meta") for r in rows if r.get("meta")}
    out = {"state": "ok", "engine": "vllm", "version": vllm.__version__, "model": a.model, "label": a.label,
           "precision": a.precision, "load_s": load_s, "engine_args": {k: v for k, v in base.items() if k != "model"},
           "extra_args": used, "load_errors": errors, "engine_info": info, "multimodal_checkpoint": mm,
           "longest_prompt": longest, "reads": reads[0],
           "wide_reads": getattr(wide_dec, "reads", 0) + (k5.reads[0] if k5 else 0),
           "env": {k: os.environ.get(k) for k in ("VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ATTENTION_BACKEND")},
           **{k: v for k, v in res.items() if k != "answers"},
           "answers": {i: {**ans, **({"meta": meta[i]} if i in meta else {})} for i, ans in res["answers"].items()}}
    write_json(a.out, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("answers",)}, default=str)[:4000])
    return 0


# ------------------------------------------------------------------------------------------------ llama.cpp
def cmd_time_llama(a: argparse.Namespace) -> int:
    """The vendor's gguf_server read-out against llama-server (letters' log-probabilities from the full-vocabulary
    log-probabilities at the answer position; ``n_probs`` widened until every letter is present), C = the server's
    parallel slots. Answers are identical whatever ``n_probs`` is (it only bounds the returned list)."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    rows = read_jsonl(a.prompts)
    order, warm, lat_n = limited(read_json(a.order)["order"], a.limit)
    mc = json.loads((Path(a.model_dir) / "config.json").read_text())
    vocab = int(mc.get("text_config", mc).get("vocab_size", 262144))
    url = a.url.rstrip("/") + "/completion"
    widened = [0]

    def read_sync(ids: list[int], allowed: list[int]) -> list[float]:
        for n in (256, 4096, vocab):
            body = json.dumps({"prompt": list(ids), "n_predict": 1, "temperature": -1, "n_probs": n,
                               "cache_prompt": True}).encode()
            req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                out = json.loads(r.read())
            rws = out.get("completion_probabilities", out.get("probs"))
            got = {t["id"]: t["logprob"] for t in rws[0]["top_logprobs"]}
            if all(t in got for t in allowed):
                return [float(got[t]) for t in allowed]
            widened[0] += 1
        raise RuntimeError("option letters missing from llama-server's log-probabilities")

    wide_dec = make_wide(a.model_dir, a.vendor, read_sync) if any(r["kind"] == "wide" for r in rows) else None
    pool = ThreadPoolExecutor(max(a.slots, 1) + 4)

    async def decide(r: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        if r["kind"] == "answer":
            return {"answer": r["answer"]}
        if r["kind"] == "wide":
            return {"answer": await loop.run_in_executor(pool, wide_answer, wide_dec, r["state"], r["spec"])}
        return {"logits": [await loop.run_in_executor(pool, read_sync, rd["ids"], rd["allowed"]) for rd in r["reads"]]}

    res = asyncio.run(run_passes(decide, rows, order, [a.slots], None, warm, 0 if a.skip_c1 else lat_n))
    pool.shutdown()
    meta = {r["id"]: r.get("meta") for r in rows if r.get("meta")}
    props = {}
    try:
        with urllib.request.urlopen(a.url.rstrip("/") + "/props", timeout=30) as r:
            props = json.loads(r.read())
        props = {k: props.get(k) for k in ("total_slots", "build_info", "model_path") if k in props}
    except Exception as e:  # noqa: BLE001
        props = {"error": repr(e)[:200]}
    out = {"state": "ok", "engine": "llama.cpp", "model": a.model_dir, "label": a.label, "precision": a.precision,
           "slots": a.slots, "server": props, "widened_reads": widened[0],
           **{k: v for k, v in res.items() if k != "answers"},
           "answers": {i: {**ans, **({"meta": meta[i]} if i in meta else {})} for i, ans in res["answers"].items()}}
    write_json(a.out, out)
    print(json.dumps({k: v for k, v in out.items() if k != "answers"}, default=str)[:3000])
    return 0


# ------------------------------------------------------------------------------------------------ Jev wire (our stack)
def cmd_time_wire(a: argparse.Namespace) -> int:
    """The scoring-head row (registration (d) 4): the harness's own Jev wire client against our server on this host,
    E1's groups as requests (the reference rows' batching), C requests in flight; the C = 1 latency pass sends one
    question per request (per-decision latency). Predictions are the harness's (answer_from_wire, norm)."""
    sys.path.insert(0, str(ROOT / "src"))
    from concurrent.futures import ThreadPoolExecutor

    from jevharness.bench.suites.static import norm
    from jevharness.decision.jev import JevBackend

    items = e1_items()
    by_id = {it.id: it for it in items}
    ordr = read_json(a.order)
    order, warm_n, lat_n = limited(ordr["order"], a.limit)
    groups = e1_groups(items)
    first = {}
    for k, i in enumerate(order):
        first.setdefault(ordr["group"][i], k)
    gorder = sorted(range(len(groups)), key=lambda g: first[g])
    clients = threading.local()

    def backend() -> Any:
        if not hasattr(clients, "b"):
            clients.b = JevBackend(model=a.model, base_url=a.url, api_key="", timeout=300.0, name="e30")
        return clients.b

    def ask(grp: list[Any]) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        res = backend()._decide(grp[0].state, [it.question for it in grp])
        ms = (time.perf_counter() - t0) * 1000
        return {it.id: {"pred": norm(ans.value), "answer": ans.raw} for it, ans in zip(grp, res.answers)}, ms

    def run(units: list[list[Any]], c: int) -> tuple[dict[str, Any], list[float], float]:
        answers: dict[str, Any] = {}
        lats: list[float] = []
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            for ans, ms in ex.map(ask, units):
                answers.update(ans)
                lats.append(ms)
        return answers, lats, time.perf_counter() - t0

    keep = set(order)
    units = [[it for it in groups[g] if it.id in keep] for g in gorder]
    units = [u for u in units if u]
    warm = set(order[:warm_n])
    run([u for u in units if any(it.id in warm for it in u)], max(a.conc))
    out: dict[str, Any] = {"passes": []}
    first_answers = None
    for c in a.conc:
        answers, lats, wall = run(units, c)
        p = {"c": c, "decisions": len(answers), "requests": len(units), "wall_s": wall, "dps": len(answers) / wall,
             "request_latency": lat_summary(lats)}
        if first_answers is None:
            first_answers, out["answers_c"] = answers, c
        else:
            p["agree_with_first"] = sum(1 for i in answers if answers[i]["pred"] == first_answers[i]["pred"]) / len(answers)
        out["passes"].append(p)
    _, lats, wall = run([[by_id[i]] for i in order[:lat_n]], 1)
    out["c1"] = {"c": 1, "decisions": min(lat_n, len(order)), "wall_s": wall, **lat_summary(lats),
                 "note": "one question per request"}
    best = max(out["passes"], key=lambda p: p["dps"])
    out["saturated"] = {"c": best["c"], "dps": best["dps"], "wall_s": best["wall_s"]}
    res = {"state": "ok", "engine": "serve_jev (wire)", "model": a.model, "url": a.url, "label": a.label,
           "precision": a.precision, **out, "answers": first_answers}
    write_json(a.out, res)
    print(json.dumps({k: v for k, v in res.items() if k != "answers"}, default=str)[:3000])
    return 0


# ================================================================================================ cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("order")
    o.add_argument("--out", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--family", required=True, choices=["code", "startlux", "intern", "openjev", "decider", "jevk5"])
    p.add_argument("--model", required=True)
    p.add_argument("--vendor", default=None, help="the StartLux package directory (pinned commit)")
    p.add_argument("--out", required=True)
    conc = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
    t = sub.add_parser("time-vllm")
    t.add_argument("--model", required=True)
    t.add_argument("--prompts", required=True)
    t.add_argument("--order", required=True)
    t.add_argument("--precision", required=True, choices=["bf16", "fp8"])
    t.add_argument("--vendor", default=None)
    t.add_argument("--conc", type=conc, default=list(SAT_C))
    t.add_argument("--label", default="")
    t.add_argument("--limit", type=int, default=None, help="smoke only: the first N decisions of the order")
    t.add_argument("--out", required=True)
    l = sub.add_parser("time-llama")  # noqa: E741
    l.add_argument("--url", required=True)
    l.add_argument("--model-dir", required=True)
    l.add_argument("--prompts", required=True)
    l.add_argument("--order", required=True)
    l.add_argument("--slots", type=int, required=True)
    l.add_argument("--precision", default="q4_k_m")
    l.add_argument("--vendor", default=None)
    l.add_argument("--label", default="")
    l.add_argument("--skip-c1", action="store_true", help="no C = 1 latency pass (measured once, at the first P)")
    l.add_argument("--limit", type=int, default=None, help="smoke only: the first N decisions of the order")
    l.add_argument("--out", required=True)
    w = sub.add_parser("time-wire")
    w.add_argument("--url", required=True)
    w.add_argument("--model", required=True)
    w.add_argument("--order", required=True)
    w.add_argument("--conc", type=conc, default=list(WIRE_C))
    w.add_argument("--precision", default="bf16")
    w.add_argument("--label", default="")
    w.add_argument("--limit", type=int, default=None, help="smoke only: the first N decisions of the order")
    w.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"order": cmd_order, "prompts": cmd_prompts, "time-vllm": cmd_time_vllm, "time-llama": cmd_time_llama,
            "time-wire": cmd_time_wire}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
