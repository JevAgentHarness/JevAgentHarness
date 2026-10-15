"""Build decision-point training sets.

  python -m jevharness.training.harvest --runs runs/jevgym_* --label gold \
      --when2call-train --xlam data/raw/xlam_function_calling_60k.json \
      --out data/decisions [--dev-frac 0.1]

Sources (combine freely; keep ~50% generic replay per docs/TRAINING_RECIPES.md):
  --runs         harness trajectories (JevGym gold labels, or LLM decisions as teacher with --label s2); tool_select
                 options are rendered ``name -- description`` exactly as the served model sees them, with the
                 descriptions rebuilt from the ``experiment.yaml`` that ``jevbench run`` saved next to the JSONL
                 (or ``--exp``); Choices with more than 52 options keep the gold + a seeded sample of 51 others
  --when2call-train   NVIDIA When2Call train_sft (CC-BY-4.0): when to call / ask / refuse (``--when2call-max``)
  --xlam         Salesforce xlam-function-calling-60k JSON (download from HF on your machine; gated)
Never train on BFCL / When2Call *test* -- they are our evaluation sets.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path
from typing import Any

from ..decision.types import OPTION_LABELS
from ..harness.tools import OTHER, extract_candidates, finish_tool, tool_desc, tool_name
from ..bench.registry import ranges_spec, reserved_ranges  # noqa: F401  (re-exported for callers)
from .data import harvest_trajectories, open_text, split_by_task, write_jsonl

NONE_TOOL = "none (no available function fits the request)"


def seed_ranges(spec: str) -> list[tuple[int, int]]:
    """'100000-100499,5000-5999' -> [(100000, 100499), (5000, 5999)]; '' -> []."""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.append((int(lo), int(hi or lo)))
    return out


def seed_allowed(seed: int | None, min_seed: int, excluded: list[tuple[int, int]]) -> bool:
    """A harvested decision may be trained on iff its JevGym seed (None = not a JevGym task) is >= ``min_seed`` and
    outside every excluded range (the evaluation suite, PREREGISTRATION A6.4)."""
    if seed is None:
        return True
    if seed < min_seed:
        return False
    return not any(lo <= seed <= hi for lo, hi in excluded)


def jevgym_seed(example_id: str) -> int | None:
    """Seed of a JevGym task id ``jevgym-<family>-<seed>-d<k>[...]/step/dp`` (None for other suites)."""
    m = re.match(r"jevgym-[a-z0-9_]+-(\d+)-", example_id)
    return int(m.group(1)) if m else None


def tool_descriptions(exp_path: str | Path, task_ids: set[str] | None = None,
                      final_tool: str = "finish") -> dict[str, dict[str, str]]:
    """task_id -> {tool name: description as the harness sends it to System-1} (``tool_desc`` of every env tool plus
    the harness's ``finish`` tool) for the tasks of the suite in ``exp_path`` (the experiment.yaml that
    ``jevbench run`` saved next to its trajectories). Suites that cannot be rebuilt offline give {}."""
    import yaml

    from ..bench.suites import load_suite

    exp = yaml.safe_load(Path(exp_path).read_text()) or {}
    suite = dict(exp.get("suite") or {})
    name = suite.pop("name", None)
    try:
        factories = load_suite(name, **suite)
    except Exception as e:  # noqa: BLE001 - tau2 / ALFWorld need their own data; harvest still works without
        print(f"[harvest] WARNING: cannot rebuild suite {name!r} from {exp_path} ({type(e).__name__}: {e}); "
              "tool_select options stay bare names (the served prompt shows `name -- description`)")
        return {}
    fin = finish_tool(final_tool)
    out: dict[str, dict[str, str]] = {}
    for f in factories:
        env = f()
        if task_ids is not None and env.task_id not in task_ids:
            continue
        out[env.task_id] = {tool_name(t): tool_desc(t) for t in env.tools() + [fin]}
    return out


def _task_ids(path: str) -> set[str]:
    with open_text(path) as f:
        return {json.loads(line)["task_id"] for line in f if line.strip()}


def from_xlam(rows: list[dict[str, Any]], max_rows: int | None = None, seed: int = 0,
              distractor_pool: list[dict[str, Any]] | None = None, n_distractors: int = 5,
              w2c_format: bool = True) -> list[dict[str, Any]]:
    """xLAM rows {query, tools (json str), answers (json str)} -> tool_select + tool_params examples.
    Adds random distractor tools from other rows so selection is non-trivial. With ``w2c_format`` every row also
    gives a When2Call-format 4-way item labelled "call one of the available tools" (the class When2Call-train
    lacks)."""
    rng = random.Random(seed)
    rows = rows[:max_rows] if max_rows else rows
    pool = distractor_pool or [t for r in rows[:5000] for t in json.loads(r["tools"])]
    out = []
    for r in rows:
        tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
        answers = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
        if len(answers) != 1:
            continue
        call = answers[0]
        extra = [t for t in rng.sample(pool, min(n_distractors, len(pool))) if t["name"] not in {x["name"] for x in tools}]
        cand_tools = tools + extra
        names = list(dict.fromkeys(t["name"] for t in cand_tools)) + [NONE_TOOL]
        if len(names) > len(OPTION_LABELS):
            continue
        rng.shuffle(names)
        rid = f"xlam{r.get('id', len(out))}"
        state = f"USER: {r['query']}"
        out.append({"id": f"{rid}/tool", "source": "xlam", "dp": "tool_select", "kind": "choice", "state": state,
                    "question": "Which function should be called to handle the user's request?", "options": names,
                    "target": {n: float(n == call["name"]) for n in names}, "q_values": None, "meta": {}})
        if w2c_format:  # the "call" class the When2Call SFT file lacks (verified call, all values in the query)
            from ..bench.suites.static import W2C_OPTIONS

            labels = list(W2C_OPTIONS.values())
            w2c_tools = [{"name": t["name"], "description": t.get("description", ""),
                          "parameters": {"type": "dict", "properties": t.get("parameters") or {},
                                         "required": [k for k, v in (t.get("parameters") or {}).items()
                                                      if not (isinstance(v, dict) and "default" in v)]}}
                         for t in tools]
            gold = W2C_OPTIONS["tool_call"]
            out.append({"id": f"{rid}/when2call", "source": "xlam", "dp": "tool_select", "kind": "choice",
                        "state": w2c_state(w2c_tools, r["query"]), "question": W2C_QUESTION, "options": labels,
                        "target": {o: float(o == gold) for o in labels}, "q_values": None,
                        "meta": {"label_source": "gold", "w2c_label": "tool_call"}})
        cands = extract_candidates(state)
        for pname, val in (call.get("arguments") or {}).items():
            if isinstance(val, (dict, list)):
                continue
            sval = str(val)
            opts = list(dict.fromkeys(cands[:50] + [OTHER]))
            gold = sval if sval in opts else OTHER
            if len(opts) < 2:
                continue
            out.append({"id": f"{rid}/param/{pname}", "source": "xlam", "dp": "tool_params", "kind": "choice",
                        "state": state, "question": f"The function `{call['name']}` will be called. Which value should "
                        f"its argument `{pname}` take?", "options": opts,
                        "target": {o: float(o == gold) for o in opts}, "q_values": None, "meta": {}})
    return out


W2C_QUESTION = "How should the assistant respond to the user's message?"  # == bench/suites/static.load_when2call


def when2call_train_label(answer: str) -> str:
    """Label of a When2Call SFT-train row. ``when2call_train_sft.jsonl`` rows are ``{tools, messages}`` with no label
    column (unlike the test MCQ file), and the file holds only the two "do not call" behaviours: the assistant turn
    either asks a follow-up question (request_for_info, ~47%) or declines (cannot_answer, ~53%)."""
    return "request_for_info" if "?" in answer else "cannot_answer"


def w2c_state(tools: list[Any], user: str) -> str:
    """System-1 state of a When2Call item, built exactly like ``static.load_when2call`` builds the test items."""
    from ..bench.suites.static import w2c_tool_text

    tools = [json.loads(t) if isinstance(t, str) else t for t in (tools or [])]
    norm = []
    for t in tools:  # the SFT-train file keeps `required` next to `parameters`; the test file (and
        params = t.get("parameters") or {}  # w2c_tool_text) expects it inside: move it, or every arg reads "optional"
        if "required" in t and isinstance(params, dict) and "required" not in params:
            t = {**{k: v for k, v in t.items() if k != "required"}, "parameters": {**params, "required": t["required"]}}
        norm.append(t)
    tool_lines = "\n".join(w2c_tool_text(t) for t in norm) or "(none)"
    return f"AVAILABLE TOOLS:\n{tool_lines}\n\nUSER: {user}"


def from_when2call_train(max_items: int | None = None, seed: int = 0,
                         data_dir: str | Path = "data/raw") -> list[dict[str, Any]]:
    """When2Call TRAIN (SFT file, CC-BY-4.0) -> 4-way decisions in the format of the When2Call test items (E1).
    Only request_for_info / cannot_answer occur (see ``when2call_train_label``): mix with call examples (xLAM:
    ``from_xlam`` adds When2Call-format "call" items) or the decider learns never to call. Ids are unique per row
    (``w2ctrain-<sha1>/when2call``) so ``split_by_task`` treats every row as its own task."""
    import hashlib

    from ..bench.suites.static import W2C_OPTIONS, WHEN2CALL_COMMIT, WHEN2CALL_URL, _download

    fn = "train/when2call_train_sft.jsonl"
    path = _download(WHEN2CALL_URL.format(commit=WHEN2CALL_COMMIT, path=fn), Path(data_dir) / "when2call" / fn)
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if max_items:
        rows = random.Random(seed).sample(rows, min(max_items, len(rows)))
    labels = list(W2C_OPTIONS.values())
    out = []
    for r in rows:
        msgs = r.get("messages") or []
        user = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
        ans = next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "assistant"), None)
        if not user or ans is None:
            continue
        lab = when2call_train_label(ans)
        gold = W2C_OPTIONS[lab]
        uid = hashlib.sha1(json.dumps(r, sort_keys=True).encode()).hexdigest()[:12]
        out.append({"id": f"w2ctrain-{uid}/when2call", "source": "when2call-train", "dp": "tool_select",
                    "kind": "choice", "state": w2c_state(r.get("tools") or [], user), "question": W2C_QUESTION,
                    "options": labels, "target": {o: float(o == gold) for o in labels}, "q_values": None,
                    "meta": {"label_source": "gold", "w2c_label": lab,
                             "label_rule": "assistant turn contains '?' -> request_for_info, else cannot_answer"}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--label", default="gold", choices=["gold", "s2", "final"])
    ap.add_argument("--only-successful", action="store_true")
    ap.add_argument("--when2call-train", action="store_true")
    ap.add_argument("--when2call-max", type=int, default=None, help="random sample of N When2Call-train rows")
    ap.add_argument("--exp", default=None, help="experiment.yaml that produced --runs (default: the one saved next "
                    "to each JSONL); used to render tool_select options with their descriptions")
    ap.add_argument("--max-options", type=int, default=len(OPTION_LABELS),
                    help="Choices with more options keep the gold + a seeded sample of the rest")
    ap.add_argument("--xlam", default=None)
    ap.add_argument("--xlam-max", type=int, default=20000)
    ap.add_argument("--out", default="data/decisions")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-jevgym-seed", type=int, default=1000,
                    help="drop JevGym tasks with a smaller seed: pilot and dev tasks use seeds 0..999; "
                         "train on runs made with `-o suite.seed0=10000` (0 disables the guard)")
    from .seed_guard import train_exclude_spec

    ap.add_argument("--exclude-jevgym-seeds", default=train_exclude_spec(),
                    help="comma-separated seed ranges that are never harvested (PREREGISTRATION A6.13: the evaluation "
                         "suite and the retired, exposed range; the calibration seeds; the Stage-2 confirmation range "
                         f"300000-300999: {train_exclude_spec()}); '' disables")
    a = ap.parse_args()
    ex: list[dict[str, Any]] = []
    paths = [p for pat in a.runs for p in (glob.glob(f"{pat}/*.jsonl") + glob.glob(f"{pat}/*.jsonl.gz") + glob.glob(pat))
             if p.endswith((".jsonl", ".jsonl.gz"))]
    if paths:
        desc: dict[str, dict[str, str]] = {}
        for p in sorted(set(paths)):
            exp = a.exp or str(Path(p).parent / "experiment.yaml")
            if Path(exp).exists():
                desc.update(tool_descriptions(exp, _task_ids(p)))
            else:
                print(f"[harvest] WARNING: no experiment.yaml next to {p} (and no --exp): tool_select options stay "
                      "bare names, but the served model sees `name -- description`")
        got = harvest_trajectories(sorted(set(paths)), label_source=a.label, only_successful=a.only_successful,
                                   descriptions=desc, max_options=a.max_options, seed=a.seed)
        excl = seed_ranges(a.exclude_jevgym_seeds)
        kept = [e for e in got if seed_allowed(jevgym_seed(e["id"]), a.min_jevgym_seed, excl)]
        if len(kept) < len(got):
            print(f"[harvest] WARNING: dropped {len(got) - len(kept)} decisions from JevGym tasks with seed < "
                  f"{a.min_jevgym_seed} or in {a.exclude_jevgym_seeds or '-'} (dev / evaluation seeds); "
                  "harvest training-seed runs instead")
        ex += kept
    if a.when2call_train:
        ex += from_when2call_train(a.when2call_max, a.seed)
    if a.xlam:
        ex += from_xlam(json.load(open(a.xlam)), max_rows=a.xlam_max, seed=a.seed)
    train, dev = split_by_task(ex, a.dev_frac, a.seed)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    write_jsonl(train, Path(a.out) / "train.jsonl")
    write_jsonl(dev, Path(a.out) / "dev.jsonl")
    from collections import Counter
    print(json.dumps({"train": len(train), "dev": len(dev),
                      "by_source_dp": Counter(f"{e['source'].split(':')[0]}|{e['dp']}" for e in ex)}, indent=1))


if __name__ == "__main__":
    main()
