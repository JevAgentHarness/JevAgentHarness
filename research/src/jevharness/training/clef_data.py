"""Decision rows (``training/data.py`` format) as Clef records: the training input of a scoring-head decision model.

Exploratory row E14 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E14: a sub-1B decision model with a scoring
head"). A Clef model (Cloudflare's joint schema head, trained here with Unsloth) reads one *record* = a state plus one
or more typed questions and returns one logit per allowed option of every question; it is served behind the Jev wire
API, so a record must look exactly like the request the harness sends (``decision/jev.question_to_wire``):

* **Choice**: ``{"type": "choice", "instructions": <question>, "criteria": {label: description | None}}``. A training
  row stores the rendered option lines (``data.option_text``): ``"name -- description"`` for tool choices and the bare
  value for arguments and When2Call. :func:`options_to_criteria` inverts that rendering: when every option carries
  ``" -- "`` it is split at the first one (tool names never contain it), otherwise each option is its own label.
* **Noul**: ``{"type": "noul", "instructions": <question>}`` (the harness's gates send no criteria).
* **Gold**: the row's target distribution mapped onto the labels (``probabilities``; soft where the row is soft) and its
  argmax (``label``), the form Unsloth's ``FastDecisionModel.build_dataset`` reads. Gold only ever goes into training
  targets: a served request never carries it.

One record per decision, except that the argument questions of one agent step (rows ``<task>/<step>/tool_params``,
``...#1``, ``...#2`` on the same state) form one record with questions ``q0``, ``q1``, ... in their logged order, as the
final harness batches them per step (``batch_s1``). Question ids are ``q<k>`` as on the wire (Clef reads the id).

Pure Python: no model, no tokenizer, no Unsloth.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

SEP = " -- "
_TRIAL = re.compile(r"#(\d+)$")


def options_to_criteria(options: Sequence[str]) -> dict[str, str | None]:
    """Rendered option lines -> wire Choice criteria ``{label: description | None}`` (order kept)."""
    split = bool(options) and all(SEP in o for o in options)
    out: dict[str, str | None] = {}
    for o in options:
        label, desc = (o.split(SEP, 1) if split else (o, None))
        if label in out:
            raise ValueError(f"duplicate option label {label!r}")
        out[label] = desc if desc else None
    return out


def row_question(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """A training row -> (wire question, rendered option -> wire key). Noul keys are ``true`` / ``false``."""
    kind = row.get("kind", "choice")
    opts = list(row["options"])
    if kind == "noul":
        if [o.split(SEP, 1)[0] for o in opts] != ["yes", "no"]:
            raise ValueError(f"noul row {row.get('id')!r} has options {opts!r}")
        return {"type": "noul", "instructions": row["question"]}, {opts[0]: "true", opts[1]: "false"}
    if kind != "choice":
        raise ValueError(f"unsupported row kind {kind!r} ({row.get('id')!r})")
    crit = options_to_criteria(opts)
    return ({"type": "choice", "instructions": row["question"], "criteria": crit},
            dict(zip(opts, crit.keys())))


def row_gold(row: Mapping[str, Any], keymap: Mapping[str, str]) -> dict[str, Any]:
    """The row's target over the wire keys (normalised) plus its argmax label."""
    tgt = {keymap[o]: max(0.0, float(row["target"].get(o, 0.0))) for o in row["options"]}
    z = sum(tgt.values())
    if not z > 0:
        raise ValueError(f"row {row.get('id')!r} has an empty target")
    probs = {k: v / z for k, v in tgt.items()}
    return {"label": max(probs, key=probs.__getitem__), "probabilities": probs}


def group_key(row: Mapping[str, Any]) -> str | None:
    """The step key of an argument row (its id without the ``#k`` suffix); None for every other row."""
    if row.get("dp") != "tool_params" or row.get("kind", "choice") != "choice":
        return None
    return _TRIAL.sub("", str(row["id"]))


def _trial(row_id: str) -> int:
    m = _TRIAL.search(row_id)
    return int(m.group(1)) if m else 0


def rows_to_records(rows: Iterable[Mapping[str, Any]], group_args: bool = True) -> list[dict[str, Any]]:
    """Training rows -> Clef records ``{"id", "state", "questions", "gold", "meta"}``, ordered by each record's first
    row in the input (a mixed file need not keep a step's argument rows together); inside a step the questions follow
    the logged argument order (``#k``). ``meta`` keeps ``dp``, ``kind``, the row ids, the source and ``after_error``."""
    slots: list[list[Mapping[str, Any]]] = []
    where: dict[tuple[str, str], int] = {}
    for r in rows:
        key = group_key(r) if group_args else None
        if key is not None and (key, r["state"]) in where:
            slots[where[(key, r["state"])]].append(r)
            continue
        if key is not None:
            where[(key, r["state"])] = len(slots)
        slots.append([r])
    out: list[dict[str, Any]] = []
    for members in slots:
        members = sorted(members, key=lambda x: _trial(str(x["id"])))
        first = members[0]
        m = first.get("meta") or {}
        rec: dict[str, Any] = {"id": str(first["id"]), "state": first["state"], "questions": {}, "gold": {},
                               "meta": {"dp": first.get("dp"), "kind": first.get("kind", "choice"), "row_ids": [],
                                        "source": first.get("source"), "after_error": m.get("after_error"),
                                        "jevgym_seed": m.get("jevgym_seed")}}
        for k, r in enumerate(members):
            q, keymap = row_question(r)
            rec["questions"][f"q{k}"] = q
            rec["gold"][f"q{k}"] = row_gold(r, keymap)
            rec["meta"]["row_ids"].append(r["id"])
        out.append(rec)
    return out


def record_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts for the build record: records, questions, by decision point, questions per record, options."""
    nq = Counter(len(r["questions"]) for r in records)
    by_dp = Counter(str(r["meta"].get("dp")) for r in records)
    q_by_dp = Counter()
    n_opts = []
    soft = 0
    for r in records:
        for qid, q in r["questions"].items():
            q_by_dp[str(r["meta"].get("dp"))] += 1
            n_opts.append(2 if q["type"] == "noul" else len(q["criteria"]))
            soft += max(r["gold"][qid]["probabilities"].values()) < 0.999
    return {"records": len(records), "questions": sum(q_by_dp.values()), "records_by_dp": dict(by_dp),
            "questions_by_dp": dict(q_by_dp), "questions_per_record": {str(k): v for k, v in sorted(nq.items())},
            "max_options": max(n_opts) if n_opts else 0, "soft_targets": soft}


def wire_request(record: Mapping[str, Any], model: str = "e14") -> dict[str, Any]:
    """The Jev wire request of a record: state and questions only -- never its gold or meta."""
    return {"model": model, "state": record["state"],
            "questions": {k: dict(v) for k, v in record["questions"].items()}}


# ------------------------------------------------------------------ exploratory row E18-F: the deferral option
DEFER_LABEL = "__defer__"
DEFER_DESC = "Hand this decision to the agent LLM (the decision model gives no answer)."


def add_defer_option(record: Mapping[str, Any], defer_qids: Iterable[str] = ()) -> dict[str, Any]:
    """A copy of a Clef record whose choice questions carry the ``__defer__`` criterion (``clef_engine.DEFER_LABEL``):
    its gold probability is 1 (and the label ``__defer__``) for the question ids in ``defer_qids``, else 0; noul
    questions are unchanged (the final harness runs no yes/no gate). Pure Python (E18 (c) 4)."""
    want = set(defer_qids)
    out = json.loads(json.dumps(record))
    for qid, q in out["questions"].items():
        if q.get("type") != "choice":
            if qid in want:
                raise ValueError(f"{record.get('id')!r}: {qid} is not a choice question")
            continue
        if DEFER_LABEL in q["criteria"]:
            raise ValueError(f"{record.get('id')!r}: {qid} already carries {DEFER_LABEL}")
        q["criteria"] = {**q["criteria"], DEFER_LABEL: DEFER_DESC}
        g = out["gold"][qid]
        if qid in want:
            g["probabilities"] = {**{k: 0.0 for k in g["probabilities"]}, DEFER_LABEL: 1.0}
            g["label"] = DEFER_LABEL
        else:
            g["probabilities"] = {**g["probabilities"], DEFER_LABEL: 0.0}
    out.setdefault("meta", {})["defer_qids"] = sorted(want)
    return out
