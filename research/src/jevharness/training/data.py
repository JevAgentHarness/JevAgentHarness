"""Decision-point training data: harvesting, formatting, and the shared prompt format.

A training example is one typed decision::

    {"id", "source", "dp", "kind": "choice"|"noul"|"score",
     "state": str, "question": str, "options": [str, ...],      # noul -> ["yes","no"]; score -> levels
     "target": {option: prob}                                   # one-hot, teacher-soft, or MC-derived
     "q_values": {option: float} | null,                        # counterfactual MC success estimates
     "meta": {...}}

Sources
  harvest_trajectories  our harness logs (``runs/**.jsonl``): S2 decisions = teacher labels,
                        oracle gold (JevGym) = gold labels, episode success = outcome signal
  from_static           BFCL / When2Call items (train splits only!)
  mc_targets            closed-form KL-regularised target  p*(a) ∝ pi_ref(a|s) * exp(Q(s,a)/beta)

The prompt format is shared by training, the logprob decider and the Jev-compatible server
(``serve_jev.py``) so a fine-tuned model can be dropped into the harness unchanged.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import random
from pathlib import Path
from typing import IO, Any, Iterable

from ..decision.types import OPTION_LABELS

SYSTEM = ("You are a fast decision module inside an AI agent. Read the state and answer the question "
          "with exactly one token: the label of your answer.")


def render(ex: dict[str, Any], order: list[int] | None = None) -> tuple[str, list[str]]:
    """Return (user prompt, code labels in presentation order). ``order`` permutes options."""
    opts = ex["options"]
    order = order or list(range(len(opts)))
    if len(opts) > len(OPTION_LABELS):
        raise ValueError("training format supports <= 52 options; use retrieval/tournament above that")
    codes = OPTION_LABELS[: len(opts)]
    lines = [f"{codes[i]}. {opts[j]}" for i, j in enumerate(order)]
    prompt = (f"STATE:\n{ex['state']}\n\nQUESTION: {ex['question']}\nOPTIONS:\n" + "\n".join(lines)
              + "\n\nAnswer with the option label only.")
    return prompt, codes


def permuted(ex: dict[str, Any], rng: random.Random | None) -> tuple[str, list[float], list[int]]:
    """Option order (random if ``rng`` is given: training-time position debiasing).
    Returns (prompt, target over presented codes, idx) with idx[pos] = original option index."""
    idx = list(range(len(ex["options"])))
    if rng is not None:
        rng.shuffle(idx)
    prompt, _ = render(ex, idx)
    tgt = [float(ex["target"].get(ex["options"][j], 0.0)) for j in idx]
    z = sum(tgt) or 1.0
    return prompt, [t / z for t in tgt], idx


# ------------------------------------------------------------------ harvesting
def _onehot(options: list[str], label: str) -> dict[str, float]:
    return {o: (1.0 if o == label else 0.0) for o in options}


def option_text(label: str, desc: Any = None) -> str:
    """An option line exactly as ``serve_jev.Engine.answer`` (and ``grpo_online.question_to_example``) renders a wire
    Choice criterion: the label, plus `` -- <description>`` when the option has one. Training data built from
    harness logs must use the same strings, or the model is trained on prompts it never sees when served."""
    if not desc:
        return label
    return f"{label} -- {desc if isinstance(desc, str) else json.dumps(desc)}"


def cap_options(options: list[str], gold: str, max_options: int, rng: random.Random) -> list[str]:
    """At most ``max_options`` options that always include ``gold``: the gold plus a uniform sample of the others,
    in their original order. The restricted readout has 52 option codes; when served, larger Choices are answered by
    a knockout over chunks of 52 (``serve_jev.Engine.choice_probs``), so a 52-option subset is what the model sees."""
    if len(options) <= max_options:
        return list(options)
    others = [o for o in options if o != gold]
    keep = set(rng.sample(others, max_options - 1)) | {gold}
    return [o for o in options if o in keep]


def harvest_trajectories(paths: Iterable[str | Path], label_source: str = "gold",
                         only_successful: bool = False,
                         descriptions: dict[str, dict[str, str]] | None = None,
                         max_options: int = len(OPTION_LABELS), seed: int = 0) -> list[dict[str, Any]]:
    """Extract decision examples from harness trajectories: Choice decisions (tool / args / action) and Noul
    gates (tool-call guard, stop guard / trigger) whose label is a bool.

    label_source: "gold" (oracle label, JevGym), "s2" (LLM decision as teacher), "final"
    only_successful: keep only decisions from successful episodes (for s2/final labels)
    descriptions: task_id -> {tool name: description} (``harvest.tool_descriptions``). tool_select options are then
                  rendered ``name -- description`` exactly as the served model sees them (``option_text``); without
                  it they stay bare names (and the harvest warns: the served prompt differs).
    max_options:  Choices with more options keep the gold + a seeded sample of the rest (``cap_options``) instead of
                  being dropped (the calibrated v2 JevGym suite has 69 tools).
    Example ids are unique: a second decision with the same ``task/step/dp`` (one per tool argument) gets ``#k``.
    """
    out = []
    seen: dict[str, int] = {}

    def uid(base: str) -> str:
        k = seen.get(base, 0)
        seen[base] = k + 1
        return base if k == 0 else f"{base}#{k}"

    for p in paths:
        with open_text(p) as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line)
                succ = bool(t["result"].get("success"))
                if only_successful and not succ:
                    continue
                desc = (descriptions or {}).get(t["task_id"])
                for st in t["steps"]:
                    for d in st["decisions"]:
                        opts = d.get("options")
                        state = d.get("state")
                        label = {"gold": d.get("gold"), "s2": d.get("s2_value"), "final": d.get("final_value")}[label_source]
                        logged = {k: d[k] for k in ("s1_value", "s1_conf", "s2_value") if d.get(k) is not None}
                        if not opts and state and isinstance(label, bool) and d.get("question"):
                            # Noul gate (guard / stop / trigger): logged without options, label is a bool
                            out.append({"id": uid(f"{t['task_id']}/{st['step']}/{d['dp']}:{d.get('mode')}"),
                                        "source": f"traj:{t['config']}", "dp": d["dp"], "kind": "noul",
                                        "state": state, "question": d["question"], "options": ["yes", "no"],
                                        "target": _onehot(["yes", "no"], "yes" if label else "no"), "q_values": None,
                                        "meta": {"episode_success": succ, "label_source": label_source,
                                                 "mode": d.get("mode"), "logged": logged}})
                            continue
                        if not opts or not state:
                            continue
                        if label is None or str(label) not in opts:
                            continue
                        eid = uid(f"{t['task_id']}/{st['step']}/{d['dp']}")
                        meta = {"episode_success": succ, "label_source": label_source, "mode": d.get("mode"),
                                "logged": logged}
                        if len(opts) > max_options:
                            meta["n_options_logged"] = len(opts)
                            opts = cap_options(list(opts), str(label), max_options, random.Random(f"{seed}:{eid}"))
                        if desc is not None and d["dp"] == "tool_select":
                            text = {o: option_text(o, desc.get(o)) for o in opts}
                            opts, label = [text[o] for o in opts], text[str(label)]
                            meta["options_rendered"] = "name -- description"
                        out.append({"id": eid, "source": f"traj:{t['config']}",
                                    "dp": d["dp"], "kind": "choice", "state": state,
                                    "question": _question_text(d), "options": opts,
                                    "target": _onehot(opts, str(label)), "q_values": None, "meta": meta})
    return out


def _question_text(d: dict[str, Any]) -> str:
    q = d.get("question") or ""
    return {"next tool": "Which tool should the agent call next?",
            "next action": "Which action should the agent take next?"}.get(q, q)


def from_static(items: list[Any]) -> list[dict[str, Any]]:
    """DecisionItems (bench/suites/static.py) -> training examples. Use TRAIN splits only."""
    out = []
    for it in items:
        q = it.question
        opts = q.options
        gold = q.meta.get("gold")
        if not opts or gold is None or len(opts) > len(OPTION_LABELS):
            continue
        out.append({"id": it.id, "source": it.source, "dp": it.dp, "kind": "choice", "state": it.state,
                    "question": q.text, "options": opts, "target": _onehot(opts, gold), "q_values": None,
                    "meta": {}})
    return out


def mc_target(pi_ref: dict[str, float], q_values: dict[str, float], beta: float = 0.2) -> dict[str, float]:
    """Closed-form KL-regularised optimum for a single-step decision (contextual bandit):
    p*(a) ∝ pi_ref(a) * exp(Q(a) / beta). Options without a Q estimate keep pi_ref mass."""
    logits = {a: math.log(max(pi_ref.get(a, 1e-6), 1e-6)) + q_values.get(a, 0.0) / beta for a in pi_ref}
    m = max(logits.values())
    z = sum(math.exp(v - m) for v in logits.values())
    return {a: math.exp(v - m) / z for a, v in logits.items()}


def mc_q_estimate(successes: int, rollouts: int) -> float:
    """Laplace-smoothed success probability from counterfactual rollouts."""
    return (successes + 1) / (rollouts + 2)


def split_by_task(examples: list[dict[str, Any]], dev_frac: float = 0.1, seed: int = 0
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by *task* (prefix of id before the first '/'), never by step, to avoid leakage."""
    tasks = sorted({e["id"].split("/")[0] for e in examples})
    random.Random(seed).shuffle(tasks)
    dev = set(tasks[: max(1, int(len(tasks) * dev_frac))])
    return [e for e in examples if e["id"].split("/")[0] not in dev], [e for e in examples if e["id"].split("/")[0] in dev]


def open_text(path: str | Path, mode: str = "r") -> IO[str]:
    """Text file handle; ``*.gz`` paths are (de)compressed transparently (``data/train_v0`` ships gzipped files).
    Gzip output has a zero header timestamp, so regenerating a file gives the same bytes (MANIFEST sha256)."""
    if str(path).endswith(".gz"):
        if "w" in mode or "a" in mode:
            raw = gzip.GzipFile(filename=str(path), mode=mode + "b", mtime=0)
            return io.TextIOWrapper(raw, encoding="utf-8")  # type: ignore[return-value]
        return gzip.open(path, mode + "t", encoding="utf-8")  # type: ignore[return-value]
    return open(path, mode, encoding="utf-8")


def write_jsonl(examples: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open_text(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------ mixing / stats CLI
def mix(sources: list[tuple[str | Path, float]], seed: int = 0) -> list[dict[str, Any]]:
    """Concatenate JSONL files, each scaled by a factor f: round(f * len(file)) examples, sampled without
    replacement for f <= 1 and with whole-file repeats + a remainder sample for f > 1. Shuffled."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for path, f in sources:
        ex = read_jsonl(path)
        n = int(round(f * len(ex)))
        reps, rem = divmod(n, len(ex)) if ex else (0, 0)
        out += ex * reps + rng.sample(ex, rem)
    rng.shuffle(out)
    return out


def stats(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts by source / decision point / kind / option-count bucket, and how soft the targets are."""
    from collections import Counter

    def bucket(k: int) -> str:
        return "2" if k == 2 else "3-5" if k <= 5 else "6-16" if k <= 16 else "17-52" if k <= 52 else ">52"

    ent, soft, with_q = 0.0, 0, 0
    for e in examples:
        t = [v for v in (e.get("target") or {}).values() if v > 0]
        ent += -sum(v * math.log(v) for v in t)
        soft += int(max(t, default=1.0) < 1.0 - 1e-9)
        with_q += int(bool(e.get("q_values")))
    n = max(1, len(examples))
    return {"n": len(examples), "tasks": len({e["id"].split("/")[0] for e in examples}),
            "by_source": Counter(str(e.get("source", "?")).split(":")[0] for e in examples),
            "by_dp": Counter(str(e.get("dp")) for e in examples),
            "by_kind": Counter(str(e.get("kind", "choice")) for e in examples),
            "by_options": Counter(bucket(len(e["options"])) for e in examples),
            "soft_frac": soft / n, "target_entropy_mean": ent / n, "with_q_values": with_q}


def main(argv: list[str] | None = None) -> None:
    """``python -m jevharness.training.data mix --in a.jsonl:1 b.jsonl:0.5 --out mixed.jsonl``
    ``python -m jevharness.training.data stats data/decisions/train.jsonl``"""
    import argparse

    ap = argparse.ArgumentParser(description="decision-data utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mix", help="mix JSONL files with per-file scale factors (PATH:FACTOR, default 1)")
    m.add_argument("--in", dest="inp", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("stats", help="dataset statistics")
    s.add_argument("path")
    a = ap.parse_args(argv)
    if a.cmd == "mix":
        srcs = []
        for spec in a.inp:
            path, _, f = spec.rpartition(":") if ":" in spec else (spec, "", "1")
            srcs.append((path or spec, float(f or 1)))
        ex = mix(srcs, a.seed)
        write_jsonl(ex, a.out)
        print(json.dumps({"out": a.out, **stats(ex)}, indent=1))
    else:
        print(json.dumps(stats(read_jsonl(a.path)), indent=1))


if __name__ == "__main__":
    main()
