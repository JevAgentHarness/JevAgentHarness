#!/usr/bin/env python3
r"""E12 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E12: more training on public
tool-use data"): AgentDecision-4B (the frozen JevAgentRL-4B checkpoint of the records) is continued on public tool-use
training items, built in exactly E1's multiple-choice format, mixed 1:1 with a replay of its original SFT v1 states.
The continued model is a NEW checkpoint, "AgentDecision-4B (mixed)" (served as ``agentdecision-4b-mixed``); the frozen
checkpoint is unchanged and stays the paper's model. No test; outside every Holm table.

    python scripts/explore_e12_train.py build --toolace data.json --out DIR    # local, CPU, $0: items, audit, mix
    python scripts/explore_e12_train.py verify --data DIR                      # the box: sha256 vs the committed record
    python scripts/explore_e12_train.py spend | watch --pid P --cap C | project --out F    # live: spend, watchdog, gate
    python scripts/explore_e12_train.py rehost --run runs/e1_bfcl_when2call_e12           # the serving-path check
    python scripts/explore_e12_train.py score --box results/explore/e12/box               # the readout

Items (E1's converter, ``bench/suites/static.py``: ``bfcl_row_items`` / ``when2call_item``; rendered for training
exactly as ``serve_jev`` renders a wire question, ``grpo_online.question_to_example``):
  * When2Call TRAIN (NVIDIA, CC-BY-4.0; ``train/when2call_train_sft.jsonl`` at E1's pinned commit): its rows hold only
    "ask for missing information" and "cannot answer" replies (no label column; ``w2c_label``). They give When2Call-format
    ask / cannot items, and BFCL-format tool-choice items whose gold is "none" (cannot rows with at least one tool: no
    available tool fits the request).
  * ToolACE (Team-ACE/ToolACE, Apache-2.0, pinned revision; English rows whose first assistant turn is ONE function
    call): When2Call-format "call one of the available tools" items, BFCL-format tool-choice items (gold = the called
    function) and BFCL-format argument items (one per scalar argument of the call; candidates copied from the request,
    plus OTHER, exactly as E1).
Never BFCL, never When2Call TEST, never an E1 item: every unit (source row) is audited against the 3,991 E1 decisions,
E1's 299-item development prefix and the full public test pools (BFCL v4's six tool categories, When2Call test) by
exact match and 8-gram overlap of the user request; a unit with any hit is dropped with all its items."""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import gzip
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

OUT = ROOT / "results" / "explore" / "e12"
TEX = ROOT / "paper" / "generated" / "e12_train.tex"
LABEL = ("exploratory (E12), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-07 'E12: more training on public "
         "tool-use data'; outside every Holm table")
NEW_NAME = "AgentDecision-4B (mixed)"
OLD_NAME = "AgentDecision-4B"
SERVED = "agentdecision-4b-mixed"
E1_CFG = ROOT / "configs" / "decision" / "bfcl_when2call_rl.yaml"
DEV_IDS = ROOT / "configs" / "eval" / "e1_dev_prefix_ids.json"
W2C_TRAIN = ROOT / "data" / "raw" / "when2call" / "train" / "when2call_train_sft.jsonl"
V1_TRAIN = ROOT / "data" / "train_v1" / "train.jsonl.gz"
V1_DEV = ROOT / "data" / "train_v1" / "dev.jsonl.gz"
V1_REPLAY_W2C = ROOT / "data" / "train_v1" / "replay_when2call.jsonl.gz"
TOOLACE_REPO = "Team-ACE/ToolACE"
TOOLACE_REV = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
TOOLACE_SHA256 = "ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e"
TOOLACE_URL = f"https://huggingface.co/datasets/{TOOLACE_REPO}/resolve/{TOOLACE_REV}/data.json"
TOOLACE_MARK = "Here is a list of functions in JSON format that you can invoke:"
# rows (fixed before any item was built): training quotas per family, the public development split, the replay
QUOTA_TRAIN = {"w2c_call": 1000, "w2c_ask": 1000, "w2c_cannot": 1000, "bfcl_tool": 1500, "bfcl_none": 500,
               "bfcl_args": 2000}
QUOTA_DEV = {"w2c_call": 150, "w2c_ask": 150, "w2c_cannot": 150, "bfcl_tool": 200, "bfcl_none": 100, "bfcl_args": 250}
N_REPLAY = 7000                       # = sum(QUOTA_TRAIN): new : replay = 1 : 1
N_CAL_V1 = 1000                       # temperatures: a seeded sample of v1 dev + the whole public dev split
DEV_SALT = "jev-explore-E12-dev"      # a unit is a dev unit iff sha256(salt \n unit) mod 10 == 0 (10 %)
NGRAM = 8
SEED = 0
MAX_OPTIONS = 52
FAMILY_FILE = {"w2c": ("w2c_call", "w2c_ask", "w2c_cannot"), "tool": ("bfcl_tool", "bfcl_none"), "args": ("bfcl_args",)}
# the When2Call-train label rule: a reply that asks for something is request_for_info, else a refusal is cannot_answer
ASK_RE = re.compile(r"\?|please provide|could you (please )?(provide|specify|tell|share)|i('ll| will)? need|"
                    r"need to know|kindly provide|please specify|please share", re.I)
CANNOT_RE = re.compile(r"unable|cannot|can't|can not|not able|apolog|sorry|don't have|do not have|not possible|"
                       r"beyond my", re.I)
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
NAN = float("nan")


# ============================================================================== io
def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _read(p: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def _write(rows: Iterable[Mapping[str, Any]], p: Path) -> int:
    from jevharness.training.data import open_text

    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open_text(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
            n += 1
    return n


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ============================================================================== sources -> E1-format items
def w2c_label(reply: str) -> str | None:
    """When2Call-train rows carry no label: a reply asking for something is request_for_info, a refusal without a
    request is cannot_answer, anything else is dropped (the file holds only these two behaviours)."""
    if ASK_RE.search(reply):
        return "request_for_info"
    if CANNOT_RE.search(reply):
        return "cannot_answer"
    return None


def w2c_tools(tools: Sequence[Any]) -> list[dict[str, Any]]:
    """The SFT-train file keeps ``required`` next to ``parameters``; E1's test file (and ``w2c_tool_text``) has it
    inside: moved, as ``training/harvest.w2c_state`` does (else every argument would read "optional")."""
    out = []
    for t in tools or []:
        t = json.loads(t) if isinstance(t, str) else dict(t)
        params = t.get("parameters") or {}
        if "required" in t and isinstance(params, dict) and "required" not in params:
            t = {**{k: v for k, v in t.items() if k != "required"}, "parameters": {**params, "required": t["required"]}}
        out.append(t)
    return out


def _uid(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def when2call_units(path: Path = W2C_TRAIN) -> tuple[list[dict[str, Any]], Counter]:
    """One unit per When2Call-train row: its When2Call-format item, and for a cannot row with tools its BFCL-format
    "none" item. Ids ``w2ctrain-<sha1[:12]>`` as in SFT v1's replay (``training/harvest.from_when2call_train``)."""
    from jevharness.bench.suites.static import bfcl_row_items, when2call_item

    units, why = [], Counter()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        msgs = r.get("messages") or []
        user = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
        reply = next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "assistant"), None)
        if not user or reply is None:
            why["no_user_or_reply"] += 1
            continue
        lab = w2c_label(reply)
        if lab is None:
            why["no_label"] += 1
            continue
        uid = f"w2ctrain-{_uid(r)}"
        tools = w2c_tools(r.get("tools") or [])
        items = {("w2c_ask" if lab == "request_for_info" else "w2c_cannot"):
                 [when2call_item(f"{uid}/when2call", tools, user, lab, "when2call-train", id_prefix="")]}
        if lab == "cannot_answer" and tools:
            row = {"id": uid, "question": [[{"role": "user", "content": user}]], "function": tools}
            got = bfcl_row_items(row, None, "w2ctrain_irrelevance", with_params=False, source="when2call-train:bfcl")
            if got:
                items["bfcl_none"] = got
        units.append({"unit": uid, "source": "when2call-train", "text": user, "items": items, "label": lab})
        why[lab] += 1
    return units, why


def _literal(node: ast.AST) -> Any:
    """ast.literal_eval that also reads JSON's true / false / null."""
    if isinstance(node, ast.Name) and node.id in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


CALL_RE = re.compile(r"^\[(?P<name>[^\(\[\]]+?)\((?P<args>.*)\)\]$", re.S)


def parse_single_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """``[Name(a="x", b=2)]`` -> (Name, {a: "x", b: 2}); None for several calls, text or anything unparseable."""
    m = CALL_RE.match(text.strip())
    if not m:
        return None
    try:
        node = ast.parse("f(" + m.group("args") + ")", mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call) or node.args:
        return None
    try:
        return m.group("name").strip(), {k.arg: _literal(k.value) for k in node.keywords if k.arg}
    except (ValueError, TypeError, SyntaxError):
        return None


def toolace_units(path: Path) -> tuple[list[dict[str, Any]], Counter]:
    """One unit per usable ToolACE row: English user turn first, the first assistant turn ONE call of a listed tool.
    Items: the When2Call-format "call" item, the BFCL-format tool-choice item and its argument items (E1's
    converter; BFCL's answer format {name: {arg: [value]}})."""
    from jevharness.bench.suites.static import bfcl_row_items, when2call_item

    units, why = [], Counter()
    for r in json.loads(path.read_text()):
        sysmsg = r.get("system") or ""
        if TOOLACE_MARK not in sysmsg:
            why["other_system_format"] += 1
            continue
        try:
            tools, _ = json.JSONDecoder().raw_decode(sysmsg.split(TOOLACE_MARK, 1)[1].strip())
        except ValueError:
            why["bad_tool_json"] += 1
            continue
        conv = r.get("conversations") or []
        if len(conv) < 2 or conv[0].get("from") != "user" or conv[1].get("from") != "assistant":
            why["not_user_assistant"] += 1
            continue
        user = str(conv[0].get("value") or "")
        if not user.strip() or CJK_RE.search(user):
            why["not_english"] += 1
            continue
        call = parse_single_call(str(conv[1].get("value") or ""))
        if call is None:
            why["not_one_call"] += 1
            continue
        name, args = call
        tools = [t for t in tools if isinstance(t, dict) and t.get("name")]
        if name not in [t["name"] for t in tools]:
            why["call_not_listed"] += 1
            continue
        uid = f"toolace-{_uid(r)}"
        row = {"id": uid, "question": [[{"role": "user", "content": user}]], "function": tools}
        gt = [{name: {k: [v] for k, v in args.items()}}]
        got = bfcl_row_items(row, gt, "toolace", with_params=True, source="toolace")
        if not got:
            why["converter_skipped"] += 1
            continue
        items = {"bfcl_tool": [i for i in got if i.dp == "tool_select"],
                 "bfcl_args": [i for i in got if i.dp == "tool_params"],
                 "w2c_call": [when2call_item(f"{uid}/when2call", tools, user, "tool_call", "toolace:w2c", id_prefix="")]}
        units.append({"unit": uid, "source": "toolace", "text": user, "items": items, "label": "tool_call"})
        why["used"] += 1
    return units, why


def to_row(item: Any, family: str, unit: str) -> dict[str, Any] | None:
    """An E1-format DecisionItem -> a training row whose prompt is byte-identical to what ``serve_jev`` builds from
    the item's wire question (``grpo_online.question_to_example``); one-hot target on the gold option."""
    from jevharness.training.data import cap_options
    from jevharness.training.grpo_online import question_to_example

    ex = question_to_example(item.state, item.question)
    gold = item.question.meta.get("gold")
    if gold is None or gold not in ex["labels"]:
        return None
    opts = ex["options"]
    g = opts[ex["labels"].index(gold)]
    if len(opts) > MAX_OPTIONS:
        opts = cap_options(opts, g, MAX_OPTIONS, random.Random(f"{SEED}:{item.id}"))
    return {"id": item.id, "source": item.source, "dp": item.dp, "kind": "choice", "state": ex["state"],
            "question": ex["question"], "options": opts, "target": {o: float(o == g) for o in opts}, "q_values": None,
            "meta": {"label_source": "gold", "e12": family, "unit": unit}}


# ============================================================================== leakage audit
def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


ROLE_RE = re.compile(r"^(USER|ASSISTANT|SYSTEM|TOOL): ", re.M)


def request_text(state: str) -> str:
    """The user's request of an E1-format state, comparable with a training unit's raw request: after the tool list
    for When2Call-format states, else the BFCL state's turns without their role prefixes."""
    if state.startswith("AVAILABLE TOOLS:"):
        return state.rpartition("\n\nUSER: ")[2]
    return ROLE_RE.sub("", state)


class Audit:
    """Exact (identical token sequence) and near-duplicate (a shared 8-gram; shorter texts: identical tokens) matching
    of training requests against named target sets."""

    def __init__(self, targets: Mapping[str, Iterable[str]], n: int = NGRAM):
        self.n = n
        self.exact: dict[str, set[str]] = {}
        self.grams: dict[str, set[tuple[str, ...]]] = {}
        self.sizes: dict[str, int] = {}
        for name, texts in targets.items():
            ex, gr, k = set(), set(), 0
            for t in texts:
                tok = _tokens(t)
                ex.add(" ".join(tok))
                gr |= self._grams(tok)
                k += 1
            self.exact[name], self.grams[name], self.sizes[name] = ex, gr, k

    def _grams(self, tok: list[str]) -> set[tuple[str, ...]]:
        if len(tok) < self.n:
            return {tuple(tok)} if tok else set()
        return {tuple(tok[i:i + self.n]) for i in range(len(tok) - self.n + 1)}

    def hits(self, text: str) -> dict[str, str]:
        tok = _tokens(text)
        key, gr = " ".join(tok), self._grams(tok)
        out = {}
        for name in self.exact:
            if key in self.exact[name]:
                out[name] = "exact"
            elif gr & self.grams[name]:
                out[name] = "ngram"
        return out


def audit_targets() -> dict[str, list[str]]:
    """E1's 3,991 decisions, E1's 299-item development prefix (both from E1's own config and loader) and the full
    public test pools (data_v1.eval_texts: every BFCL v4 tool-category row, every When2Call test question)."""
    import yaml

    from jevharness.bench.decision_eval import exclude_by_parent, load_population
    from jevharness.training.data_v1 import eval_texts

    exp = yaml.safe_load(E1_CFG.read_text())
    e1 = load_population(exp)
    full = load_population({k: v for k, v in exp.items() if k != "exclude_ids_file"})
    e1_ids = {i.id for i in e1}
    dev = [i for i in full if i.id not in e1_ids]
    if len(e1) != 3991 or len(dev) != 299:
        raise SystemExit(f"E1 population {len(e1)} / dev prefix {len(dev)} (expected 3,991 / 299)")
    assert not {i.id for i in exclude_by_parent(dev, DEV_IDS)}, "dev items not all in the dev prefix"
    return {"e1": [request_text(i.state) for i in e1], "e1_dev": [request_text(i.state) for i in dev],
            "public_test_pools": [ROLE_RE.sub("", t) for t in eval_texts(ROOT / "data" / "raw")]}


# ============================================================================== build
def is_dev(unit: str) -> bool:
    return int(hashlib.sha256(f"{DEV_SALT}\n{unit}".encode()).hexdigest()[:8], 16) % 10 == 0


def _pick(units: Sequence[Mapping[str, Any]], fam: str, k: int, rng: random.Random, used: set[str],
          per_unit: bool = True) -> list[tuple[Any, str]]:
    """``k`` items of family ``fam`` (one per unit when ``per_unit``, else any of the units' items), seeded, from units
    not in ``used``; the chosen units join ``used`` when ``per_unit``."""
    if per_unit:
        pool = sorted((u for u in units if fam in u["items"] and u["unit"] not in used), key=lambda u: u["unit"])
        got = rng.sample(pool, min(k, len(pool)))
        used.update(u["unit"] for u in got)
        return [(u["items"][fam][0], u["unit"]) for u in got]
    pool = sorted(((it, u["unit"]) for u in units for it in u["items"].get(fam, [])), key=lambda x: x[0].id)
    return rng.sample(pool, min(k, len(pool)))


def select(units: Sequence[Mapping[str, Any]], quota: Mapping[str, int], rng: random.Random
           ) -> dict[str, list[tuple[Any, str]]]:
    """Per family; tool-choice before "call" items and "none" before "cannot" items, each from disjoint units, so a
    ToolACE row gives either its When2Call-format or its BFCL-format choice item (arguments come from any row)."""
    ta = [u for u in units if u["source"] == "toolace"]
    wc = [u for u in units if u["source"] == "when2call-train"]
    used_ta: set[str] = set()
    used_wc: set[str] = set()
    out = {"bfcl_tool": _pick(ta, "bfcl_tool", quota["bfcl_tool"], rng, used_ta),
           "w2c_call": _pick(ta, "w2c_call", quota["w2c_call"], rng, used_ta),
           "bfcl_args": _pick(ta, "bfcl_args", quota["bfcl_args"], rng, set(), per_unit=False),
           "bfcl_none": _pick(wc, "bfcl_none", quota["bfcl_none"], rng, used_wc),
           "w2c_cannot": _pick(wc, "w2c_cannot", quota["w2c_cannot"], rng, used_wc),
           "w2c_ask": _pick(wc, "w2c_ask", quota["w2c_ask"], rng, used_wc)}
    return out


def replay_rows(n: int, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A seeded uniform sample of the SFT v1 training rows (data/train_v1/train.jsonl.gz: AgentDecision-4B's SFT
    states; E3's D100 file is byte-identical), every JevGym seed re-checked against the training guards."""
    from jevharness.bench import registry as R
    from jevharness.training.data_v1 import jevgym_seed
    from jevharness.training.seed_guard import TRAIN_FORBIDDEN

    rows = _read(V1_TRAIN)
    got = rng.sample(rows, min(n, len(rows)))
    bad = []
    for r in got:
        s = jevgym_seed(r["id"])
        if s is None:
            continue
        if (s < R.TRAIN_MIN_SEED or R.in_ranges(s, R.training_exclusions() + (R.CALIBRATION_RANGE,))
                or any(lo <= s <= hi for lo, hi in TRAIN_FORBIDDEN)):
            bad.append(r["id"])
    if bad:
        raise SystemExit(f"replay rows on guarded seeds: {bad[:5]}")
    for r in got:
        r["meta"] = {**(r.get("meta") or {}), "e12": "replay_v1"}
    return got, {"path": str(V1_TRAIN.relative_to(ROOT)), "sha256": _sha(V1_TRAIN), "rows_available": len(rows),
                 "rows": len(got), "dp": dict(Counter(r["dp"] for r in got))}


def build(toolace: Path, out: Path) -> dict[str, Any]:
    from jevharness.bench.suites.static import WHEN2CALL_COMMIT
    from jevharness.harness.tools import OTHER

    if _sha(toolace) != TOOLACE_SHA256:
        raise SystemExit("ToolACE data.json differs from the pinned revision's sha256")
    rng = random.Random(SEED)
    wu, w_why = when2call_units()
    tu, t_why = toolace_units(toolace)
    units, seen, n_dup = [], set(), Counter()
    for u in wu + tu:   # identical source rows (same sha1) are one unit: the first is kept
        if u["unit"] in seen:
            n_dup[u["source"]] += 1
            continue
        seen.add(u["unit"])
        units.append(u)
    # leakage audit (before any split or sample): a unit with any hit is dropped with all its items
    aud = Audit(audit_targets())
    hits: dict[str, dict[str, str]] = {}
    for u in units:
        h = aud.hits(u["text"])
        if h:
            hits[u["unit"]] = h
    table: dict[str, Counter] = defaultdict(Counter)
    for uid, h in hits.items():
        src = "toolace" if uid.startswith("toolace-") else "when2call-train"
        for tgt, kind in h.items():
            table[src][f"{tgt}:{kind}"] += 1
        table[src]["units_dropped"] += 1
    n_items = {u["unit"]: sum(len(v) for v in u["items"].values()) for u in units}
    clean = [u for u in units if u["unit"] not in hits]
    dev_u = [u for u in clean if is_dev(u["unit"])]
    tr_u = [u for u in clean if not is_dev(u["unit"])]
    sel_tr = select(tr_u, QUOTA_TRAIN, rng)
    sel_dev = select(dev_u, QUOTA_DEV, rng)
    rows_tr = [r for fam, its in sel_tr.items() for it, unit in its if (r := to_row(it, fam, unit))]
    rows_dev = [r for fam, its in sel_dev.items() for it, unit in its if (r := to_row(it, fam, unit))]
    assert not {r["meta"]["unit"] for r in rows_tr} & {r["meta"]["unit"] for r in rows_dev}, "train / dev share a unit"
    assert not {r["meta"]["unit"] for r in rows_tr + rows_dev} & set(hits), "an audited unit got through"
    replay, rep_info = replay_rows(N_REPLAY, rng)
    mix = rows_tr + replay
    rng.shuffle(mix)
    v1dev = _read(V1_DEV)
    keep = sorted(rng.sample(range(len(v1dev)), min(N_CAL_V1, len(v1dev))))
    cal = [v1dev[i] for i in keep] + rows_dev
    out.mkdir(parents=True, exist_ok=True)
    n = {"train_mix": _write(mix, out / "train_mix.jsonl.gz"), "public_train": len(rows_tr),
         "public_dev": _write(rows_dev, out / "public_dev.jsonl.gz"), "cal_dev": _write(cal, out / "cal_dev.jsonl.gz")}
    for f, fams in FAMILY_FILE.items():   # the public dev split per E1 family (offline readout)
        n[f"dev_{f}"] = _write([r for r in rows_dev if r["meta"]["e12"] in fams], out / f"dev_{f}.jsonl.gz")
    seen_v1 = {r["id"] for r in _read(V1_REPLAY_W2C)} if V1_REPLAY_W2C.exists() else set()
    info = {
        "label": LABEL, "generated": _now(), "seed": SEED, "quota_train": QUOTA_TRAIN, "quota_dev": QUOTA_DEV,
        "n_replay": N_REPLAY, "n_cal_v1": N_CAL_V1, "dev_rule": f"sha256('{DEV_SALT}\\n<unit>') mod 10 == 0",
        "rows": n,
        "train_by_family": dict(Counter(r["meta"]["e12"] for r in rows_tr)),
        "dev_by_family": dict(Counter(r["meta"]["e12"] for r in rows_dev)),
        "train_by_dp": dict(Counter(r["dp"] for r in mix)),
        "train_args_other": sum(1 for r in rows_tr if r["meta"]["e12"] == "bfcl_args"
                                and max(r["target"], key=r["target"].get) == OTHER),
        "train_options_mean": {f: round(sum(len(r["options"]) for r in rows_tr if r["meta"]["e12"] == f)
                                        / max(1, sum(1 for r in rows_tr if r["meta"]["e12"] == f)), 2)
                               for f in QUOTA_TRAIN},
        "when2call_train_ids_in_sftv1_replay_file": sum(1 for r in rows_tr if r["id"] in seen_v1),
        "sources": {
            "when2call_train": {"url_commit": WHEN2CALL_COMMIT, "file": "train/when2call_train_sft.jsonl",
                                "sha256": _sha(W2C_TRAIN), "license": "CC-BY-4.0", "rows_by_outcome": dict(w_why)},
            "toolace": {"repo": TOOLACE_REPO, "revision": TOOLACE_REV, "url": TOOLACE_URL, "sha256": TOOLACE_SHA256,
                        "license": "Apache-2.0", "rows_by_outcome": dict(t_why)}},
        "units": {"total": len(units), "toolace": sum(1 for u in units if u["source"] == "toolace"),
                  "when2call_train": sum(1 for u in units if u["source"] == "when2call-train"),
                  "duplicate_rows_dropped": dict(n_dup), "clean": len(clean), "train": len(tr_u), "dev": len(dev_u)},
        "audit": {"targets": aud.sizes, "rule": f"exact token sequence or a shared {NGRAM}-gram of the user request "
                                               "(shorter requests: identical tokens); a unit with any hit is dropped",
                  "units_with_hit": len(hits), "items_dropped": sum(n_items[u] for u in hits),
                  "by_source": {k: dict(v) for k, v in table.items()},
                  "by_target": {t: {k: sum(1 for h in hits.values() if h.get(t) == k) for k in ("exact", "ngram")}
                                for t in aud.sizes}},
        "replay": rep_info,
        "cal_dev": {"v1_dev_rows": len(keep), "public_dev_rows": len(rows_dev), "v1_dev_sha256": _sha(V1_DEV)},
    }
    info["sha256"] = {f.name: _sha(f) for f in sorted(out.glob("*.jsonl.gz"))}
    (out / "build_info.json").write_text(json.dumps(info, indent=1) + "\n")
    (out / "audit_hits.json").write_text(json.dumps({"label": LABEL, "hits": hits}, indent=1, sort_keys=True) + "\n")
    return info


def devprefix(out: Path) -> dict[str, Any]:
    """E1's 299-item development prefix (excluded from E1's 3,991 decisions) as training-format rows (E1's converter,
    E12's renderer): E12c's checkpoint selection set. Never trained on (E12's audit dropped every overlapping unit)."""
    import yaml

    from jevharness.bench.decision_eval import load_population

    exp = yaml.safe_load(E1_CFG.read_text())
    e1 = {i.id for i in load_population(exp)}
    full = load_population({k: v for k, v in exp.items() if k != "exclude_ids_file"})
    dev = [i for i in full if i.id not in e1]
    fam = lambda i: "w2c" if i.source.startswith("when2call") else ("tool" if i.dp == "tool_select" else "args")  # noqa: E731
    rows = [r for i in dev if (r := to_row(i, fam(i), i.unit))]
    n = _write(rows, out)
    return {"items": len(dev), "rows": n, "by_family": dict(Counter(r["meta"]["e12"] for r in rows)), "sha256": _sha(out)}


def verify(data: Path, record: Path = OUT / "build_info.json") -> int:
    want = json.loads(record.read_text())["sha256"]
    bad = [f for f, s in want.items() if not (data / f).exists() or _sha(data / f) != s]
    print("data sha256:", "all match" if not bad else f"MISMATCH {bad}")
    return 1 if bad else 0


# ============================================================================== E12b: randomised order rules
# Secondary row E12b (Deviations 2026-10-07 "E12b: reading the order rule"): E7d's recipe (harvest, rows, continuation
# of the frozen adapter, temperatures) on JevGym-Hard tasks whose order rule is drawn per seed from rules that are never
# `tier` (jevgym_hard.random_rule), scored zero-shot on E7d's held-out `tier` states, beside E7d's offline numbers.
B_SETTING = {"name": "jevgym", "families": "multi", "multi_customers": 12, "multi_rule": "random",
             "distractor_style": "near", "n_distractors": 60, "noise_lines": 20, "strict": True}
B_TRAIN_SEEDS = (322000, 600)          # seed0, n: a new training block (>= 10000, outside every guarded range)
B_HELD = {"op": "heldout_op.jsonl", "ts_other": "heldout_ts_other.jsonl", "tp": "heldout_tp.jsonl"}
B_TRIM = {"ts_other": 225, "tp": 225}  # E7d's trimmed strata ("E7d: scoring within the GPU cap"), recomputed (seed 0)


def harvest_b(out: Path, n: int | None = None, workers: int = 16) -> Path:
    """E7d's $0 harvest (simulated oracle executor, NearMiss System-1, behaviour policies lite_shadow and
    nearmiss_replace; every label the oracle's gold) on the random-rule setting, training seeds only."""
    import explore_e7d_tier as E7D
    from jevharness.training import data_v1 as D

    llm = {"provider": "simulated", "model": "oracle-sim", "error_rate": E7D.ERROR_RATE, "simulated_latency_ms": 0}
    arms = [a for a in D.behaviour_arms() if a["name"] in ("lite_shadow", "nearmiss_replace")]
    seed0, n0 = B_TRAIN_SEEDS
    exp = D.build_experiment(name="e12b_train", seed0=seed0, n=n or n0, llm=llm, workers=workers,
                             suite={k: v for k, v in B_SETTING.items() if k != "name"}, arms=arms)
    exp["harness_defaults"] = {"max_steps": E7D.MAX_STEPS}
    D.collect(exp, out, s1=D.NearMissBackend(**E7D.EPS), progress=False)
    return out / "e12b_train"


def build_b(harvest_dir: Path, d100: Path, v1_dev: Path, e7d_data: Path, out: Path, seed: int = SEED) -> dict[str, Any]:
    """E7d's rows and quotas on the random-rule harvest: 900 write-state, 450 other tool-choice and 450 argument rows
    plus 1,200 D100 replay rows (train_mix.jsonl); the dev split (10 % of the tasks) sampled to 1,000 rows plus 1,000
    v1 dev rows (cal_dev_small.jsonl: the temperatures); E7d's held-out tier files, sha256-checked against E7d's
    build record, with its trimmed strata recomputed (seed 0)."""
    import explore_e7d_tier as E7D
    from jevharness.bench import registry as R

    rng = random.Random(seed)
    rows, s_tr = E7D.tier_rows(harvest_dir / "e12b_train")
    seeds = {int(E7D._task(r).split("-")[2]) for r in rows}
    bad = [s for s in seeds if R.in_ranges(s, R.training_exclusions() + (R.CALIBRATION_RANGE,)) or s < R.TRAIN_MIN_SEED]
    if bad:
        raise SystemExit(f"E12b training rows on guarded seeds: {sorted(bad)[:5]}")
    tasks = sorted({E7D._task(r) for r in rows})
    dev_tasks = set(rng.sample(tasks, max(1, round(E7D.DEV_TASK_FRAC * len(tasks)))))
    tr = [r for r in rows if E7D._task(r) not in dev_tasks]
    dv = [r for r in rows if E7D._task(r) in dev_tasks]
    new = E7D._sample(tr, E7D.N_TIER, rng)
    d100_rows = E7D._read(d100)
    replay = rng.sample(d100_rows, min(E7D.N_REPLAY, len(d100_rows)))
    for r in new:
        r.setdefault("meta", {})["e12b"] = "random_rule"
    for r in replay:
        r.setdefault("meta", {})["e12b"] = "replay_d100"
    mix = new + replay
    rng.shuffle(mix)
    out.mkdir(parents=True, exist_ok=True)
    n = {"train_mix": E7D._write(mix, out / "train_mix.jsonl"), "b_dev": E7D._write(dv, out / "b_dev.jsonl")}
    cal = E7D.trim(v1_dev, E7D.TRIM["cal_v1"], out / "cal_v1_small.jsonl", seed)
    calb = E7D.trim(out / "b_dev.jsonl", E7D.TRIM["cal_tier"], out / "cal_b_small.jsonl", seed)
    n["cal_dev_small"] = E7D._write(E7D._read(out / "cal_v1_small.jsonl") + E7D._read(out / "cal_b_small.jsonl"),
                                    out / "cal_dev_small.jsonl")
    e7d_info = json.loads((e7d_data / "build_info.json").read_text())
    held: dict[str, Any] = {}
    for s, f in B_HELD.items():
        got = _sha(e7d_data / f)
        if got != e7d_info["sha256"][f]:
            raise SystemExit(f"E7d held-out file {f} differs from E7d's build record")
        held[s] = {"file": f, "sha256": got}
    for s, k in B_TRIM.items():
        held[s]["trim"] = E7D.trim(e7d_data / B_HELD[s], k, out / f"heldout_{s}_s.jsonl", seed)
    (out / "heldout_op.jsonl").write_bytes((e7d_data / B_HELD["op"]).read_bytes())
    rules = Counter()
    for t in tasks:
        from jevharness.bench.suites.jevgym_hard import random_rule
        rules[random_rule(int(t.split("-")[2]))["kind"]] += 1
    info = {"label": LABEL.replace("(E12)", "(E12b)"), "generated": _now(), "setting": B_SETTING,
            "train_seeds": B_TRAIN_SEEDS, "quota": E7D.N_TIER, "replay": E7D.N_REPLAY, "seed": seed, "rows": n,
            "strata": dict(Counter(E7D.stratum(r) for r in new)), "available": dict(Counter(E7D.stratum(r) for r in tr)),
            "tasks": {"train": len(tasks) - len(dev_tasks), "dev": len(dev_tasks)}, "rule_kinds": dict(rules),
            "harvest": {k: v for k, v in s_tr.items() if k != "run"}, "cal": {"v1": cal, "b": calb},
            "d100": {"sha256": _sha(d100), "rows": len(d100_rows)}, "heldout": held,
            "sha256": {f.name: _sha(f) for f in sorted(out.glob("*.jsonl"))}}
    (out / "build_info.json").write_text(json.dumps(info, indent=1, default=str) + "\n")
    return info


def score_b(box: Path) -> dict[str, Any]:
    """Held-out tier states: the E12b checkpoint and the frozen checkpoint on the E12 host, beside E7d's recorded
    numbers (results/explore/e7d_tier/summary.json: AgentDecision-4B and AgentDecision-4B (tier))."""
    import explore_e7d_tier as E7D

    out: dict[str, Any] = {"available": False}
    for m in ("e12b", "frozen"):
        p = box / "b" / "eval" / f"{m}_heldout.json"
        if p.exists():
            ev = json.loads(p.read_text())
            out[m] = {s: E7D._file_acc(ev, f"heldout_{s}") for s in ("op", "ts_other", "tp")}
            out["available"] = True
    e7d = ROOT / "results" / "explore" / "e7d_tier" / "summary.json"
    if e7d.exists():
        d = json.loads(e7d.read_text())
        out["e7d"] = {m: (d["models"].get(m) or {}).get("heldout") for m in ("frozen", "tier")}
    for f in ("build_info.json",):
        p = box / "b" / "data" / f
        if p.exists():
            bi = json.loads(p.read_text())
            out["data"] = {k: bi.get(k) for k in ("rows", "strata", "tasks", "rule_kinds", "train_seeds")}
    p = box / "b" / "train_record.json"
    if p.exists():
        out["train"] = json.loads(p.read_text())
    return out


# ============================================================================== live: spend, watchdog, gate
RUN_SMOKE, RUN_LIVE = "e12_live_smoke", "e12_live"
CAP_USD = 30.0
EXPS = {"e12": {"runs": (RUN_SMOKE, RUN_LIVE), "cap": CAP_USD, "arm": "e12_ts_replace_f"},
        "e12c": {"runs": ("e12c_live_smoke", "e12c_live"), "cap": 15.0, "arm": "e12c_ts_replace_f"}}
LIVE_N = 200
PROJ_FACTOR = 1.5


def total_spend(runs: Path, exp: str = "e12") -> dict[str, Any]:
    import explore_e6a_alfworld as E6A

    per = {name: E6A.run_spend(runs / name) for name in EXPS[exp]["runs"] if (runs / name).is_dir()}
    return {"by_run": per, **{k: sum(v[k] for v in per.values()) for k in ("gemini", "jev", "total", "records")}}


def cmd_watch(a: argparse.Namespace) -> int:
    """At the cap: SIGTERM / SIGKILL to the runner's process group and ``<runs>/e12_STOP.json``."""
    import os
    import signal
    import time

    stop = Path(a.runs) / "e12_STOP.json"
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend(Path(a.runs), a.exp)
        if s["total"] >= a.cap:
            stop.write_text(json.dumps({"at": _now(), "spend": s, "cap": a.cap}, indent=1))
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(a.pid), sig)
                except OSError:
                    pass
                time.sleep(10)
            print(f"CAP REACHED: ${s['total']:.2f} >= ${a.cap:.2f}; runner stopped", flush=True)
            return 1
        print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} spend ${s['total']:.3f} ({s['records']} records)",
              flush=True)
        time.sleep(a.every)


def projection(runs: Path, exp: str = "e12") -> dict[str, Any]:
    """The smoke's mean $ per episode x 200 x 1.5 plus the spend so far must fit the cap, else no live episode."""
    import explore_e6a_alfworld as E6A

    cap = EXPS[exp]["cap"]
    sm = runs / EXPS[exp]["runs"][0]
    eps = []
    for p in sorted(sm.glob("*.jsonl")):
        for t in E6A._latest(p).values():
            eps.append(E6A.episode_spend(t)["total"])
    spent = total_spend(runs, exp)["total"]
    if not eps:
        return {"decision": "stop: no smoke episode", "spent": spent}
    per = sum(eps) / len(eps)
    proj = spent + per * LIVE_N * PROJ_FACTOR
    return {"smoke_episodes": len(eps), "usd_per_episode": per, "spent": spent, "projected": proj, "cap": cap,
            "decision": f"run all {LIVE_N}" if proj <= cap else "stop: projection above the cap"}


# ============================================================================== the serving-path check
def latest(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in _read(path):
        out[str(r["id"])] = r
    return out


def rehost(run: Path, logged: Path = ROOT / "runs" / "e1_bfcl_when2call_rl" / "s1_jevagent_rl.jsonl") -> dict[str, Any]:
    """The frozen checkpoint re-served on the E12 host answers E1 again: its answers must agree with its logged E1
    answers on >= 99 % of the decisions (E3 / E5's rule), and its serving time gives the same-host cost reference."""
    import explore_e9_frontier as E9

    new, old = latest(run / "s1_frozen_rehost.jsonl"), latest(logged)
    ids = sorted(set(new) & set(old))
    agree = sum(1 for i in ids if new[i].get("pred") == old[i].get("pred") and not new[i].get("error"))
    res = {"n": len(ids), "missing": len(set(old) - set(new)), "agree": agree,
           "agreement": agree / len(ids) if ids else NAN, "ok": bool(ids) and agree / len(ids) >= 0.99,
           "acc_rehost": sum(E9.is_correct(new[i]) for i in ids) / len(ids) if ids else NAN,
           "acc_logged": sum(E9.is_correct(old[i]) for i in ids) / len(ids) if ids else NAN,
           "usd_per_decision_rehost": sum(E9.gpu_cost(new[i]) for i in ids) / len(ids) if ids else NAN,
           "usd_per_decision_logged": sum(E9.gpu_cost(old[i]) for i in ids) / len(ids) if ids else NAN}
    return res


# ============================================================================== readout
def _pt(x: Any, d: int = 1) -> str:
    return f"{100 * x:.{d}f}" if isinstance(x, (int, float)) and math.isfinite(x) else "--"


def _pm(x: Any, d: int = 1) -> str:
    return f"{100 * x:+.{d}f}" if isinstance(x, (int, float)) and math.isfinite(x) else "--"


def e1_block(box: Path, runs: Path) -> dict[str, Any]:
    """E1's 3,991 decisions: E12 vs the frozen checkpoint's logged answers (E9's records), per family, paired item
    bootstrap (E9's paired_gap); the frontier with the E12 point is E9's (``explore_e9_frontier.e12_point``)."""
    import explore_e9_frontier as E9

    run = runs / "e1_bfcl_when2call_e12"
    f = run / "s1_e12_mixed.jsonl"
    if not f.exists():
        return {"available": False}
    new = latest(f)
    old = latest(runs / E9.MODELS["rl"][1])
    ids = sorted(old)
    if set(new) != set(old):
        return {"available": False, "reason": f"E12 answered {len(new)} of {len(old)} E1 decisions"}
    out: dict[str, Any] = {"available": True, "n": len(ids), "models": {}}
    for k, d in (("e12", new), ("frozen", old)):
        out["models"][k] = E9.model_stats([d[i] for i in ids], "gpu")
    out["delta"] = {"all": E9.paired_gap([E9.is_correct(new[i]) for i in ids], [E9.is_correct(old[i]) for i in ids])}
    for fam in E9.FAMILIES:
        fi = [i for i in ids if E9.family_of(old[i]) == fam]
        out["delta"][fam] = E9.paired_gap([E9.is_correct(new[i]) for i in fi], [E9.is_correct(old[i]) for i in fi])
    w2c = defaultdict(lambda: [0, 0])
    for i in ids:
        if E9.family_of(old[i]) == "w2c":
            lab = (old[i].get("gold") or ["?"])[0]
            w2c[lab][0] += int(E9.is_correct(new[i]))
            w2c[lab][1] += 1
    out["w2c_by_label"] = {k: {"n": v[1], "acc": v[0] / v[1]} for k, v in w2c.items()}
    if (run / "s1_frozen_rehost.jsonl").exists():
        out["rehost"] = rehost(run)
    return out


def c_block(box: Path) -> dict[str, Any]:
    """Corpus C (the 5,314 common states): accuracy of both checkpoints and the paired task bootstrap of the delta;
    outcome (ii): non-inferior at a 1-point margin when the interval's lower end is >= -1 point."""
    import ablation_score as AS
    import e5_analysis as EA

    corpus = ROOT / "results" / "stage2" / "s23"
    f = box / "s23" / "answers_e12_mixed.jsonl"
    if not f.exists():
        return {"available": False}
    occ, eligible = AS.load_corpus(corpus)
    res: dict[str, Any] = {"available": True}
    per = {}
    for m, p in (("frozen", corpus / "answers_rl.jsonl"), ("e12", f)):
        cm, pt = EA.c_rows(occ, eligible, AS.load_answers(p))
        res[m] = {k: cm.get(k) for k in ("c_acc", "c_acc_offplan", "c_acc_ask", "c_tasks", "c_missing", "c_errors")}
        per[m] = pt
    res["occurrences"] = len(occ)
    res["delta"] = {s: AS.paired_boot(per["e12"][s], per["frozen"][s], EA.N_BOOT, 0) for s in ("all", "offplan")}
    res["ni_margin_pt"] = 1.0
    res["non_inferior"] = res["delta"]["all"]["lo"] >= -0.01
    return res


def offline_block(box: Path) -> dict[str, Any]:
    """eval_decisions on the held-out JevGym dev rows (v1 dev, dev_recovery, dev_knockout) and the public dev split,
    both checkpoints on the E12 host."""
    out: dict[str, Any] = {}
    for m in ("e12", "frozen"):
        row: dict[str, Any] = {}
        for part in ("v1", "pubdev"):
            p = box / "eval" / f"{m}_{part}.json"
            if not p.exists():
                continue
            ev = json.loads(p.read_text())
            files = ev.get("files") or ev.get("by_file") or {}
            row[part] = {Path(k).name.split(".")[0]: {"n": (v.get("overall") or v).get("n"),
                                                      "acc": (v.get("overall") or v).get("acc"),
                                                      "ece": (v.get("overall") or v).get("ece")}
                         for k, v in files.items()}
        out[m] = row
    return out


def live_block(runs: Path) -> dict[str, Any]:
    import explore_common as C
    import explore_startlux_live as SL

    tasks = set(json.loads(SL.TASKS.read_text())["task_ids"][:LIVE_N])
    p = runs / RUN_LIVE / "e12_ts_replace_f.jsonl"
    if not p.exists():
        return {"available": False}
    rows: dict[str, Any] = {"available": True}
    ys = {}
    for k, path in (("e12", p), ("frozen", runs / "stage2_rl_states" / "rl_ts_replace_f.jsonl"),
                    ("jev", runs / "stage2_rl_states" / "jev_ts_replace_f.jsonl")):
        r, y = SL.live_row(path, tasks)
        rows[k], ys[k] = r, y
    for ref in ("frozen", "jev"):
        common = sorted(set(ys["e12"]) & set(ys[ref]))
        if common:
            nc = C.newcombe([ys["e12"][t] for t in common], [ys[ref][t] for t in common])
            rows[f"vs_{ref}"] = {"n": len(common), "diff_pt": nc["diff_pt"], "ci_pt": nc["ci_pt"],
                                 "wins": sum(1 for t in common if ys["e12"][t] > ys[ref][t]),
                                 "losses": sum(1 for t in common if ys["e12"][t] < ys[ref][t])}
    return rows


# ============================================================================== E12c and the same-engine readout
BOX_C = OUT / "box_c"
FAST_MODELS = {"frozen": ("AgentDecision-4B", "e1_bfcl_when2call_rl/s1_jevagent_rl.jsonl"),
               "e12": (NEW_NAME, "e1_bfcl_when2call_e12/s1_e12_mixed.jsonl"),
               "e12c": ("StartLux-4B (mixed)", "e1_bfcl_when2call_e12c/s1_e12c_sl4b.jsonl"),
               "startlux4b": ("StartLux-4B", "e1_bfcl_when2call_startlux/s1_startlux_4b.jsonl")}


def e12c_block(box_c: Path, runs: Path) -> dict[str, Any]:
    """E12c's four outcomes: E1 against StartLux-4B and Jev (paired item bootstrap; "beats" iff the lower end > 0),
    corpus C against AgentDecision-4B (1-point margin), live success; plus the checkpoint selection and offline rows."""
    import ablation_score as AS
    import e5_analysis as EA
    import explore_common as C
    import explore_e9_frontier as E9
    import explore_startlux_live as SL

    out: dict[str, Any] = {"available": False}
    for k, f in (("selection", box_c / "sel" / "selection.json"), ("base_choice", box_c / "base_choice.json"),
                 ("devprefix", box_c / "devprefix.json"), ("host", box_c / "host.json")):
        if f.exists():
            out[k] = json.loads(f.read_text())
    for k, f in (("smoke", box_c / "smoke.jsonl"), ("train_runs", box_c / "train_runs.jsonl")):
        if f.exists():
            out[k] = _read(f)
    f = runs / FAST_MODELS["e12c"][1]
    if f.exists():
        new = latest(f)
        refs = {"sl4b": latest(runs / E9.MODELS["sl4b"][1]), "jev": latest(runs / E9.MODELS["jev"][1]),
                "frozen": latest(runs / E9.MODELS["rl"][1])}
        ids = sorted(refs["frozen"])
        if set(new) == set(ids):
            out["available"] = True
            e1: dict[str, Any] = {"n": len(ids), "models": {"e12c": E9.model_stats([new[i] for i in ids], "gpu")}}
            for r, d in refs.items():
                e1["models"][r] = E9.model_stats([d[i] for i in ids], "api" if r == "jev" else "gpu")
                e1[f"vs_{r}"] = {"all": E9.paired_gap([E9.is_correct(new[i]) for i in ids],
                                                       [E9.is_correct(d[i]) for i in ids])}
                for fam in E9.FAMILIES:
                    fi = [i for i in ids if E9.family_of(d[i]) == fam]
                    e1[f"vs_{r}"][fam] = E9.paired_gap([E9.is_correct(new[i]) for i in fi], [E9.is_correct(d[i]) for i in fi])
            e1["beats_sl4b"] = bool(e1["vs_sl4b"]["all"]["ci"][0] > 0)
            out["e1"] = e1
        else:
            out["e1_reason"] = f"E12c answered {len(set(new) & set(ids))} of {len(ids)} E1 decisions"
    corpus = ROOT / "results" / "stage2" / "s23"
    fc = box_c / "s23" / "answers_e12c_sl4b.jsonl"
    if fc.exists():
        occ, eligible = AS.load_corpus(corpus)
        per, c = {}, {}
        for m, pth in (("frozen", corpus / "answers_rl.jsonl"), ("sl4b", corpus / "answers_sl4b.jsonl"), ("e12c", fc)):
            if pth.exists():
                cm, pt = EA.c_rows(occ, eligible, AS.load_answers(pth))
                c[m] = {k: cm.get(k) for k in ("c_acc", "c_acc_offplan", "c_acc_ask", "c_tasks", "c_missing", "c_errors")}
                per[m] = pt
        c["delta_vs_frozen"] = AS.paired_boot(per["e12c"]["all"], per["frozen"]["all"], EA.N_BOOT, 0)
        if "sl4b" in per:
            c["delta_vs_sl4b"] = AS.paired_boot(per["e12c"]["all"], per["sl4b"]["all"], EA.N_BOOT, 0)
        c["non_inferior_to_frozen"] = c["delta_vs_frozen"]["lo"] >= -0.01
        out["c"] = c
    lp = runs / "e12c_live" / "e12c_ts_replace_f.jsonl"
    if lp.exists():
        tasks = set(json.loads(SL.TASKS.read_text())["task_ids"][:LIVE_N])
        lv: dict[str, Any] = {}
        ys = {}
        for k, path in (("e12c", lp), ("frozen", runs / "stage2_rl_states" / "rl_ts_replace_f.jsonl"),
                        ("jev", runs / "stage2_rl_states" / "jev_ts_replace_f.jsonl"),
                        ("sl4b", runs / "stage2_startlux_states" / "sl4b_ts_replace_f.jsonl")):
            r, y = SL.live_row(path, tasks)
            if r:
                lv[k], ys[k] = r, y
        for ref in ("frozen", "jev", "sl4b"):
            common = sorted(set(ys.get("e12c", {})) & set(ys.get(ref, {})))
            if common:
                nc = C.newcombe([ys["e12c"][t] for t in common], [ys[ref][t] for t in common])
                lv[f"vs_{ref}"] = {"n": len(common), "diff_pt": nc["diff_pt"], "ci_pt": nc["ci_pt"]}
        out["live"] = lv
    for part in ("v1", "pubdev"):
        pth = box_c / "eval" / f"e12c_{part}.json"
        if pth.exists():
            files = json.loads(pth.read_text())["files"]
            out.setdefault("offline", {})[part] = {Path(k).name.split(".")[0]: {"n": v["overall"]["n"],
                                                                               "acc": v["overall"]["acc"]}
                                                   for k, v in files.items()}
    sp = total_spend(runs, "e12c")
    out["spend"] = {"total": sp["total"], "records": sp["records"], "cap": EXPS["e12c"]["cap"]}
    return out


def fast_block(box_c: Path, runs: Path) -> dict[str, Any]:
    """The same-engine readout: each model's vLLM timing (wall-clock GPU time per decision and E9's rule) beside its E9
    cost from its own serving stack, and the agreement of the engine's answers with the served ones."""
    import explore_e9_frontier as E9

    out: dict[str, Any] = {}
    vf = box_c / "fast" / "vllm_version.txt"
    if vf.exists():
        out["engine"] = vf.read_text().strip()
    for m, (label, rec) in FAST_MODELS.items():
        f = box_c / "fast" / f"time_{m}.json"
        row: dict[str, Any] = {"label": label}
        if f.exists():
            d = json.loads(f.read_text())
            row.update({k: v for k, v in d.items() if k not in ("answers", "label")})
            served = runs / rec
            if d.get("answers") and served.exists():
                s = latest(served)
                ids = sorted(set(d["answers"]) & set(s))
                row["agreement_with_served"] = (sum(1 for i in ids if d["answers"][i] == s[i].get("pred")) / len(ids)
                                                if ids else None)
                row["e9_rule_usd_per_1000_served"] = 1000 * sum(E9.gpu_cost(s[i]) for i in ids) / len(ids) if ids else None
        out[m] = row
    return out


def frontier_block() -> dict[str, Any]:
    import explore_e9_frontier as E9

    p = E9.OUT / "summary.json"
    if not p.exists():
        return {"available": False}
    return (json.loads(p.read_text()).get("e12") or {"available": False})


def score(box: Path, runs: Path = ROOT / "runs") -> dict[str, Any]:
    d: dict[str, Any] = {"label": LABEL, "generated": _now(), "box": str(box.relative_to(ROOT)) if box.is_absolute()
                         and ROOT in box.parents else str(box)}
    bi = OUT / "build_info.json"
    if bi.exists():
        b = json.loads(bi.read_text())
        d["data"] = {k: b[k] for k in ("rows", "train_by_family", "dev_by_family", "units", "audit", "replay",
                                       "train_args_other", "when2call_train_ids_in_sftv1_replay_file") if k in b}
    for k, f in (("train", box / "train_record.json"), ("host", box / "host.json"), ("windows", box / "windows.json")):
        if f.exists():
            d[k] = json.loads(f.read_text())
    d["e1"] = e1_block(box, runs)
    d["frontier"] = frontier_block()
    d["c"] = c_block(box)
    d["offline"] = offline_block(box)
    d["live"] = live_block(runs)
    d["e12b"] = score_b(box)
    d["e12c"] = e12c_block(BOX_C, runs)
    d["fast"] = fast_block(box, runs)      # the E12 host (Deviations "E12c: same-engine timing on the E12 host")
    sp = total_spend(runs)
    d["spend"] = {"gemini": sp["gemini"], "total": sp["total"], "records": sp["records"], "cap": CAP_USD}
    for k, f in (("projection", box / "projection.json"),):
        if f.exists():
            d[k] = json.loads(f.read_text())
    return d


def tex_keys(d: Mapping[str, Any]) -> dict[str, str]:
    import explore_common as C

    v: dict[str, str] = {}
    data = d.get("data") or {}
    if data:
        rows = data["rows"]
        v["e12/data/new"] = C.fmt(int(rows["public_train"]))
        v["e12/data/replay"] = C.fmt(int(data["replay"]["rows"]))
        v["e12/data/total"] = C.fmt(int(rows["train_mix"]))
        v["e12/data/dev"] = C.fmt(int(rows["public_dev"]))
        for f, n in data["train_by_family"].items():
            v[f"e12/data/{f.replace('_', '')}"] = C.fmt(int(n))
        a = data["audit"]
        v["e12/audit/units"] = C.fmt(int(a["units_with_hit"]))
        v["e12/audit/items"] = C.fmt(int(a["items_dropped"]))
        for t, kinds in a["by_target"].items():
            for kind, n in kinds.items():
                v[f"e12/audit/{t.replace('_', '')}/{kind}"] = C.fmt(int(n))
        v["e12/audit/unitsall"] = C.fmt(int(data["units"]["total"]))
    e1 = d.get("e1") or {}
    if e1.get("available"):
        for m, s in e1["models"].items():
            v[f"e12/{m}/e1/acc"] = _pt(s["acc"])
            v[f"e12/{m}/e1/usdk"] = f"{1000 * s['usd']:.4f}"
            for fam, x in s["families"].items():
                v[f"e12/{m}/e1/acc/{fam}"] = _pt(x["acc"])
        for k, g in e1["delta"].items():
            if g.get("est") is not None:
                v[f"e12/e1/delta/{k}"] = f"{g['est']:+.1f}"
                v[f"e12/e1/delta/{k}/lo"] = f"{g['ci'][0]:+.1f}"
                v[f"e12/e1/delta/{k}/hi"] = f"{g['ci'][1]:+.1f}"
        for lab, x in (e1.get("w2c_by_label") or {}).items():
            v[f"e12/e12/e1/w2c/{lab[:4]}"] = _pt(x["acc"])
        rh = e1.get("rehost") or {}
        if rh:
            v["e12/rehost/agree"] = _pt(rh["agreement"])
            v["e12/rehost/usdk"] = f"{1000 * rh['usd_per_decision_rehost']:.4f}"
    fr = d.get("frontier") or {}
    if fr.get("available"):
        v["e12/front"] = "yes" if fr.get("on_frontier") else "no"
        v["e12/front/changed"] = "yes" if fr.get("frontier_changed") else "no"
        v["e12/front/hostadj"] = "yes" if fr.get("on_frontier_hostadj") else "no"
        if fr.get("usd_hostadj") is not None:
            v["e12/e12/e1/usdkadj"] = f"{1000 * fr['usd_hostadj']:.4f}"
    c = d.get("c") or {}
    if c.get("available"):
        for m in ("e12", "frozen"):
            v[f"e12/{m}/c/acc"] = _pt(c[m]["c_acc"])
            v[f"e12/{m}/c/ask"] = _pt(c[m]["c_acc_ask"])            # = cov/ask/<m>/ask_now/acc's stratum
            v[f"e12/{m}/c/offplan"] = _pt(c[m]["c_acc_offplan"])
        do = c["delta"]["offplan"]
        v["e12/c/offplan/delta"], v["e12/c/offplan/lo"], v["e12/c/offplan/hi"] = _pm(do["delta"]), _pm(do["lo"]), _pm(do["hi"])
        da = c["delta"]["all"]
        v["e12/c/delta"] = _pm(da["delta"])
        v["e12/c/delta/lo"] = _pm(da["lo"])
        v["e12/c/delta/hi"] = _pm(da["hi"])
        v["e12/c/ni"] = "yes" if c["non_inferior"] else "no"
        v["e12/c/n"] = C.fmt(int(c["occurrences"]))
    off = d.get("offline") or {}
    for m, row in off.items():
        for part, files in row.items():
            for name, x in files.items():
                if x.get("acc") is not None:
                    v[f"e12/{m}/{part}/{name.replace('_', '')}/acc"] = _pt(x["acc"])
    lv = d.get("live") or {}
    if lv.get("available"):
        for m in ("e12", "frozen", "jev"):
            x = lv.get(m) or {}
            if x.get("live_success") is not None:
                v[f"e12/{m}/live/success"] = _pt(x["live_success"])
                v[f"e12/{m}/live/n"] = C.fmt(int(x["live_n"]))
                v[f"e12/{m}/live/own"] = _pt(x.get("own_acc"))
                v[f"e12/{m}/live/usdep"] = f"{x['api_cost']:.4f}" if x.get("api_cost") is not None else "--"
        for ref in ("frozen", "jev"):
            x = lv.get(f"vs_{ref}")
            if x:
                v[f"e12/live/vs/{ref}"] = f"{x['diff_pt']:+.1f}"
                v[f"e12/live/vs/{ref}/lo"] = f"{x['ci_pt'][0]:+.1f}"
                v[f"e12/live/vs/{ref}/hi"] = f"{x['ci_pt'][1]:+.1f}"
    b = d.get("e12b") or {}
    for m in ("e12b", "frozen"):
        for s, x in (b.get(m) or {}).items():
            if x and x.get("acc") is not None:
                v[f"e12b/{m}/heldout/{s.replace('_', '')}/acc"] = _pt(x["acc"])
                v[f"e12b/{m}/heldout/{s.replace('_', '')}/n"] = str(x.get("n"))
    for m, row in (b.get("e7d") or {}).items():
        for s, x in (row or {}).items():
            if x and x.get("acc") is not None:
                v[f"e12b/e7d{m}/heldout/{s.replace('_', '')}/acc"] = _pt(x["acc"])
    if (b.get("train") or {}).get("minutes") is not None:
        v["e12b/train/min"] = str(b["train"]["minutes"])
    c = d.get("e12c") or {}
    e = c.get("e1") or {}
    for m, s in (e.get("models") or {}).items():
        v[f"e12c/{m}/e1/acc"] = _pt(s["acc"])
        v[f"e12c/{m}/e1/usdk"] = f"{1000 * s['usd']:.4f}"
        for fam, x in s["families"].items():
            v[f"e12c/{m}/e1/acc/{fam}"] = _pt(x["acc"])
    for ref in ("sl4b", "jev", "frozen"):
        for fam, g in (e.get(f"vs_{ref}") or {}).items():
            if g.get("est") is not None:
                v[f"e12c/e1/vs/{ref}/{fam}"] = f"{g['est']:+.1f}"
                v[f"e12c/e1/vs/{ref}/{fam}/lo"] = f"{g['ci'][0]:+.1f}"
                v[f"e12c/e1/vs/{ref}/{fam}/hi"] = f"{g['ci'][1]:+.1f}"
    if e:
        v["e12c/e1/beatssl4b"] = "yes" if e["beats_sl4b"] else "no"
    cc = c.get("c") or {}
    for m in ("e12c", "frozen", "sl4b"):
        if cc.get(m):
            v[f"e12c/{m}/c/acc"] = _pt(cc[m]["c_acc"])
            v[f"e12c/{m}/c/ask"] = _pt(cc[m]["c_acc_ask"])
            v[f"e12c/{m}/c/offplan"] = _pt(cc[m]["c_acc_offplan"])
    if cc.get("delta_vs_sl4b"):
        x = cc["delta_vs_sl4b"]
        v["e12c/c/vssl4b"], v["e12c/c/vssl4b/lo"], v["e12c/c/vssl4b/hi"] = _pm(x["delta"]), _pm(x["lo"]), _pm(x["hi"])
    if cc.get("delta_vs_frozen"):
        x = cc["delta_vs_frozen"]
        v["e12c/c/delta"], v["e12c/c/delta/lo"], v["e12c/c/delta/hi"] = _pm(x["delta"]), _pm(x["lo"]), _pm(x["hi"])
        v["e12c/c/ni"] = "yes" if cc["non_inferior_to_frozen"] else "no"
    lv = c.get("live") or {}
    for m in ("e12c", "frozen", "jev", "sl4b"):
        x = lv.get(m) or {}
        if x.get("live_success") is not None:
            v[f"e12c/{m}/live/success"] = _pt(x["live_success"])
            v[f"e12c/{m}/live/n"] = str(x["live_n"])
    for ref in ("frozen", "jev", "sl4b"):
        x = lv.get(f"vs_{ref}")
        if x:
            v[f"e12c/live/vs/{ref}"] = f"{x['diff_pt']:+.1f}"
            v[f"e12c/live/vs/{ref}/lo"] = f"{x['ci_pt'][0]:+.1f}"
            v[f"e12c/live/vs/{ref}/hi"] = f"{x['ci_pt'][1]:+.1f}"
    sel = (c.get("selection") or {}).get("selected")
    if sel:
        v["e12c/sel/epoch"] = str(sel["epoch"])
    for cand in (c.get("selection") or {}).get("candidates") or []:
        v[f"e12c/sel/ep{cand['epoch']}/devprefix"] = _pt(cand["acc"].get("e1_devprefix"))
        v[f"e12c/sel/ep{cand['epoch']}/v1dev"] = _pt(cand["acc"].get("dev"))
    for part, files in (c.get("offline") or {}).items():
        for name, x in files.items():
            v[f"e12c/e12c/{part}/{name.replace('_', '')}/acc"] = _pt(x["acc"])
    hc = c.get("host") or {}
    if hc.get("gpu_hours") is not None:
        v["e12c/gpuh"] = f"{hc['gpu_hours']:.1f}"
    bc = c.get("base_choice") or {}
    if bc.get("s_per_step") is not None:
        v["e12c/smoke/sperstep"] = f"{bc['s_per_step']:.1f}"
    tr = [r for r in (c.get("train_runs") or []) if r.get("exit") == 0]
    if tr:
        v["e12c/train/min"] = str(tr[-1]["minutes"])
    if c.get("spend"):
        v["e12c/spend"] = f"{c['spend']['total']:.2f}"
    for m, row in (d.get("fast") or {}).items():
        if not isinstance(row, dict):
            continue
        if row.get("usd_per_1000_wall") is not None:
            v[f"e12/fast/{m}/usdkwall"] = f"{row['usd_per_1000_wall']:.4f}"
            v[f"e12/fast/{m}/usdke9"] = f"{row['usd_per_1000_e9_rule']:.4f}"
            v[f"e12/fast/{m}/msdec"] = f"{1000 * row['gpu_s_per_decision']:.1f}"
            for w, s in (row.get("by_workers") or {}).items():          # c = requests (states) in flight
                v[f"e12/fast/{m}/c{w}/p50"] = f"{s['latency_ms_median']:.0f}"
                v[f"e12/fast/{m}/c{w}/msdec"] = f"{1000 * s['gpu_s_per_decision']:.1f}"
                v[f"e12/fast/{m}/c{w}/usdkwall"] = f"{s['usd_per_1000_wall']:.4f}"
            if row.get("weights_gb") is not None:
                v[f"e12/fast/{m}/gb"] = f"{row['weights_gb']:.1f}"
        if row.get("agreement_with_served") is not None:
            v[f"e12/fast/{m}/agree"] = _pt(row["agreement_with_served"])
        if row.get("e9_rule_usd_per_1000_served") is not None:
            v[f"e12/fast/{m}/usdkserved"] = f"{row['e9_rule_usd_per_1000_served']:.4f}"
    sp = d.get("spend") or {}
    if sp:
        v["e12/spend"] = f"{sp['total']:.2f}"
    tr = d.get("train") or {}
    if tr.get("minutes") is not None:
        v["e12/train/min"] = str(tr["minutes"])
    host = d.get("host") or {}
    if host.get("gpu_hours") is not None:
        v["e12/gpuh"] = f"{host['gpu_hours']:.1f}"
    return v


def markdown(d: Mapping[str, Any]) -> str:
    v = tex_keys(d)
    g = lambda k: v.get(k, "--")  # noqa: E731
    L = [f"# E12: {NEW_NAME} (more training on public tool-use data)", "", f"_{d['label']}_", "",
         "## Pre-registered outcomes", "",
         f"(i) **Fixed questions (E1, 3,991 decisions):** {NEW_NAME} {g('e12/e12/e1/acc')} % vs {OLD_NAME} "
         f"{g('e12/frozen/e1/acc')} % ({g('e12/e1/delta/all')} points, paired item bootstrap 95 % "
         f"[{g('e12/e1/delta/all/lo')}, {g('e12/e1/delta/all/hi')}]).",
         f"(ii) **Common states C (5,314):** {g('e12/e12/c/acc')} % vs {g('e12/frozen/c/acc')} % "
         f"({g('e12/c/delta')} points, paired task bootstrap [{g('e12/c/delta/lo')}, {g('e12/c/delta/hi')}]); "
         f"non-inferior at the 1-point margin: {g('e12/c/ni')}.",
         f"(iii) **On E9's frontier:** {g('e12/front')} (frontier changed by the new point: {g('e12/front/changed')}; "
         f"with the same-host cost adjustment: {g('e12/front/hostadj')}).",
         f"(iv) **Live success (replace mode, Flash, seeds 300000-300199):** {g('e12/e12/live/success')} % "
         f"(n {g('e12/e12/live/n')}) vs {OLD_NAME} {g('e12/frozen/live/success')} % ({g('e12/live/vs/frozen')} points, "
         f"Newcombe [{g('e12/live/vs/frozen/lo')}, {g('e12/live/vs/frozen/hi')}]).", "",
         "## E1 by family", "", f"| | {NEW_NAME} | {OLD_NAME} | delta [95 %] |", "|---|---|---|---|"]
    for fam, lab in (("toolchoice", "BFCL tool choice"), ("args", "BFCL arguments"), ("w2c", "When2Call")):
        L.append(f"| {lab} | {g(f'e12/e12/e1/acc/{fam}')} | {g(f'e12/frozen/e1/acc/{fam}')} | "
                 f"{g(f'e12/e1/delta/{fam}')} [{g(f'e12/e1/delta/{fam}/lo')}, {g(f'e12/e1/delta/{fam}/hi')}] |")
    L += [f"| all | {g('e12/e12/e1/acc')} | {g('e12/frozen/e1/acc')} | {g('e12/e1/delta/all')} |",
          f"| $ per 1,000 decisions (E9's rule) | {g('e12/e12/e1/usdk')} | {g('e12/frozen/e1/usdk')} | |", ""]
    e1 = d.get("e1") or {}
    if e1.get("w2c_by_label"):
        L.append("When2Call by gold label (E12): " + ", ".join(
            f"{k} {_pt(x['acc'])} % (n {x['n']})" for k, x in sorted(e1["w2c_by_label"].items())) + ".")
    rh = e1.get("rehost") or {}
    if rh:
        L.append(f"Serving-path check: the frozen checkpoint re-served on the E12 host agrees with its logged E1 answers "
                 f"on {_pt(rh['agreement'])} % of {rh['n']} decisions (rule >= 99 %: {'ok' if rh['ok'] else 'FAILED'}); "
                 f"its same-host cost is ${1000 * rh['usd_per_decision_rehost']:.4f} per 1,000 decisions (logged on "
                 f"the Stage-2 host: ${1000 * rh['usd_per_decision_logged']:.4f}).")
    fr = d.get("frontier") or {}
    if fr.get("available"):
        L += ["", f"Frontier without the new point: {' -> '.join(fr.get('frontier_without') or [])}; with it: "
                  f"{' -> '.join(fr.get('frontier_with') or [])}."]
    off = d.get("offline") or {}
    if off:
        L += ["", "## Offline (E12 host)", "", f"| file | {NEW_NAME} | {OLD_NAME} |", "|---|---|---|"]
        names = sorted({(p, n) for m in off.values() for p, fs in m.items() for n in fs})
        for p, n in names:
            L.append(f"| {p}/{n} | {_pt(((off.get('e12') or {}).get(p) or {}).get(n, {}).get('acc'))} | "
                     f"{_pt(((off.get('frozen') or {}).get(p) or {}).get(n, {}).get('acc'))} |")
    lv = d.get("live") or {}
    if lv.get("available"):
        L += ["", "## Live (tool_select replace, Flash, the first 200 fresh Stage-2 tasks)", "",
              "| row | success % | n | own-state acc % | steps | API $ / ep. |", "|---|---|---|---|---|---|"]
        for m, lab in (("e12", NEW_NAME), ("frozen", OLD_NAME + " (A7.10 row)"), ("jev", "Jev (A7.10 row)")):
            x = lv.get(m) or {}
            L.append(f"| {lab} | {_pt(x.get('live_success'))} | {x.get('live_n', '--')} | {_pt(x.get('own_acc'))} | "
                     f"{x.get('steps') if x.get('steps') is None else round(x['steps'], 1)} | "
                     f"{'--' if x.get('api_cost') is None else round(x['api_cost'], 4)} |")
    data = d.get("data") or {}
    if data:
        a = data["audit"]
        L += ["", "## Data and leakage audit", "",
              f"New items {data['rows']['public_train']:,} (by family {json.dumps(data['train_by_family'])}), replay "
              f"{data['replay']['rows']:,} SFT v1 rows, public dev {data['rows']['public_dev']:,}. Units: "
              f"{json.dumps(data['units'])}.",
              f"Audit ({a['rule']}): {a['units_with_hit']} units ({a['items_dropped']} items) dropped; by target "
              f"{json.dumps(a['by_target'])}; by source {json.dumps(a['by_source'])}."]
    b = d.get("e12b") or {}
    if b.get("available"):
        L += ["", "## E12b: reading the order rule (held-out `tier` states, zero-shot for E12b)", "",
              "| stratum | AgentDecision-4B (random rules, E12b) | AgentDecision-4B (this host) | AgentDecision-4B (E7d) | "
              "AgentDecision-4B (tier, E7d: trained on tier) |", "|---|---|---|---|---|"]
        for s, lab in (("op", "cancel/ship tool choice"), ("tsother", "other tool choices"), ("tp", "arguments")):
            L.append(f"| {lab} | {g(f'e12b/e12b/heldout/{s}/acc')} | {g(f'e12b/frozen/heldout/{s}/acc')} | "
                     f"{g(f'e12b/e7dfrozen/heldout/{s}/acc')} | {g(f'e12b/e7dtier/heldout/{s}/acc')} |")
        if b.get("data"):
            L.append(f"E12b data: {json.dumps(b['data'])}.")
    c = d.get("e12c") or {}
    if c:
        L += ["", "## E12c: the recipe on StartLux-Decision-4B (CC BY-NC 4.0)", ""]
        if c.get("base_choice"):
            L.append(f"Base and memory setting: {json.dumps(c['base_choice'])}.")
        if c.get("selection"):
            L.append(f"Checkpoint selection (E1 dev prefix + v1 dev only): {json.dumps(c['selection']['selected'])}.")
        L += ["", "| outcome | value |", "|---|---|",
              f"| (i) E1: StartLux-4B (mixed) vs StartLux-4B | {g('e12c/e12c/e1/acc')} vs {g('e12c/sl4b/e1/acc')} % "
              f"({g('e12c/e1/vs/sl4b/all')} [{g('e12c/e1/vs/sl4b/all/lo')}, {g('e12c/e1/vs/sl4b/all/hi')}]); beats: "
              f"{g('e12c/e1/beatssl4b')} |",
              f"| (ii) E1: gap to Jev | {g('e12c/e1/vs/jev/all')} [{g('e12c/e1/vs/jev/all/lo')}, {g('e12c/e1/vs/jev/all/hi')}] "
              f"(Jev {g('e12c/jev/e1/acc')} %) |",
              f"| (iii) C vs AgentDecision-4B | {g('e12c/e12c/c/acc')} vs {g('e12c/frozen/c/acc')} % ({g('e12c/c/delta')} "
              f"[{g('e12c/c/delta/lo')}, {g('e12c/c/delta/hi')}]); non-inferior at 1 point: {g('e12c/c/ni')} |",
              f"| (iv) live success | {g('e12c/e12c/live/success')} % (n {g('e12c/e12c/live/n')}) vs AgentDecision-4B "
              f"{g('e12c/frozen/live/success')} %, Jev {g('e12c/jev/live/success')} %, StartLux-4B "
              f"{g('e12c/sl4b/live/success')} % |",
              f"| E1 $ per 1,000 (E9's rule, own stack) | {g('e12c/e12c/e1/usdk')} (Jev {g('e12c/jev/e1/usdk')}, "
              f"StartLux-4B {g('e12c/sl4b/e1/usdk')}) |"]
        for fam, lab in (("toolchoice", "BFCL tool choice"), ("args", "BFCL arguments"), ("w2c", "When2Call")):
            L.append(f"| E1 {lab} | {g(f'e12c/e12c/e1/acc/{fam}')} (StartLux-4B {g(f'e12c/sl4b/e1/acc/{fam}')}, "
                     f"Jev {g(f'e12c/jev/e1/acc/{fam}')}) |")
    fa = d.get("fast") or {}
    if any(isinstance(x, dict) and x.get("usd_per_1000_wall") is not None for x in fa.values()):
        L += ["", f"## Same-engine serving cost ({fa.get('engine', 'vLLM')}, merged LoRA, E1's 3,991 decisions, 4 states in flight)", "",
              "| model | GPU ms / decision (wall) | $ / 1,000 (wall) | $ / 1,000 (E9's rule, this engine) | "
              "$ / 1,000 (E9's rule, own stack) | agreement with served answers |", "|---|---|---|---|---|---|"]
        for m, row in fa.items():
            if not isinstance(row, dict) or "label" not in row:
                continue
            if row.get("usd_per_1000_wall") is None:
                L.append(f"| {row['label']} | {row.get('state', 'not run')} | | | | |")
                continue
            L.append(f"| {row['label']} | {1000 * row['gpu_s_per_decision']:.1f} | {row['usd_per_1000_wall']:.4f} | "
                     f"{row['usd_per_1000_e9_rule']:.4f} | {g(f'e12/fast/{m}/usdkserved')} | {g(f'e12/fast/{m}/agree')} |")
        L.append("StartLux-4B is timed on serve_jev's prompt (not its vendor's), so its agreement is not an accuracy.")
    sp = d.get("spend") or {}
    L += ["", f"Spend: ${sp.get('total', 0):.2f} of API (Gemini, live smoke + live), cap ${CAP_USD:.0f}.",
          f"GPU: {g('e12/gpuh')} host hours; training {g('e12/train/min')} min.", "",
          f"Generated {d.get('generated', '')}."]
    return "\n".join(L) + "\n"


def write_tex(vals: Mapping[str, str], path: Path = TEX) -> Path:
    import explore_common as C

    lines = ["% AUTO-GENERATED by scripts/explore_e12_train.py -- do not edit by hand.",
             "% Exploratory rows E12 / E12b, more training on public tool-use data / reading the order rule "
             "(descriptive; docs/PREREG_A7_STAGE2.md Deviations 2026-10-07), outside both stages.",
             f"% last written {_now()} at git {(C.git_head() or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(vals.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + [r"\makeatother", ""]))
    return path


def write(d: Mapping[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(d, indent=1, default=str) + "\n")
    (OUT / "summary.md").write_text(markdown(d))
    write_tex(tex_keys(d))


# ============================================================================== cli
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--toolace", required=True)
    b.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--data", required=True)
    v.add_argument("--record", default=str(OUT / "build_info.json"))
    dp = sub.add_parser("devprefix")
    dp.add_argument("--out", required=True)
    s = sub.add_parser("spend")
    s.add_argument("--exp", default="e12", choices=sorted(EXPS))
    s.add_argument("--runs", default="runs")
    s.add_argument("--short", action="store_true")
    w = sub.add_parser("watch")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--cap", type=float, default=CAP_USD)
    w.add_argument("--exp", default="e12", choices=sorted(EXPS))
    w.add_argument("--runs", default="runs")
    w.add_argument("--every", type=int, default=30)
    p = sub.add_parser("project")
    p.add_argument("--runs", default="runs")
    p.add_argument("--exp", default="e12", choices=sorted(EXPS))
    p.add_argument("--out", required=True)
    r = sub.add_parser("rehost")
    r.add_argument("--run", default="runs/e1_bfcl_when2call_e12")
    r.add_argument("--out", default=None)
    hb = sub.add_parser("harvest-b")
    hb.add_argument("--out", required=True)
    hb.add_argument("--n", type=int, default=None)
    hb.add_argument("--workers", type=int, default=16)
    bb = sub.add_parser("build-b")
    bb.add_argument("--harvest", required=True)
    bb.add_argument("--d100", required=True)
    bb.add_argument("--v1-dev", default=str(V1_DEV))
    bb.add_argument("--e7d-data", required=True)
    bb.add_argument("--out", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--box", default=str(OUT / "box"))
    sc.add_argument("--runs", default=str(ROOT / "runs"))
    a = ap.parse_args(argv)
    if a.cmd == "build":
        info = build(Path(a.toolace), Path(a.out))
        print(json.dumps({k: info[k] for k in ("rows", "train_by_family", "dev_by_family", "units", "audit")}, indent=1))
    elif a.cmd == "harvest-b":
        print(harvest_b(Path(a.out), a.n, a.workers))
    elif a.cmd == "build-b":
        info = build_b(Path(a.harvest), Path(a.d100), Path(a.v1_dev), Path(a.e7d_data), Path(a.out))
        print(json.dumps({k: info[k] for k in ("rows", "strata", "tasks", "rule_kinds")}, indent=1))
    elif a.cmd == "verify":
        return verify(Path(a.data), Path(a.record))
    elif a.cmd == "devprefix":
        print(json.dumps(devprefix(Path(a.out))))
    elif a.cmd == "spend":
        sp = total_spend(Path(a.runs), a.exp)
        print(f"${sp['total']:.3f} ({sp['records']} records)" if a.short else json.dumps(sp, indent=1))
    elif a.cmd == "watch":
        return cmd_watch(a)
    elif a.cmd == "project":
        pr = projection(Path(a.runs), a.exp)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(pr, indent=1) + "\n")
        print(json.dumps(pr, indent=1))
    elif a.cmd == "rehost":
        res = rehost(Path(a.run))
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    elif a.cmd == "score":
        d = score(Path(a.box), Path(a.runs))
        write(d)
        print(markdown(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
