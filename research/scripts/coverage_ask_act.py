#!/usr/bin/env python3
r"""Ask-vs-act states on the common-state corpus C -- DESCRIPTIVE / POST HOC, offline (no model is called).

The question (coverage of decision-model roles, results/explore/small_models/coverage.md §3.1): "ask the user or act?"
is not a separate decision point in this project; it is an option (``ask_user``) of the tool choice. This script reads
how each decision model answered that choice on the recorded agent states of S2-3's corpus C (PREREG_A7_STAGE2 A7.10
(c)), split by what the oracle says about asking.

    python scripts/coverage_ask_act.py
    # reads results/stage2/s23/{corpus.json, occurrences.jsonl.gz, answers_<backend>.jsonl, replay_<backend>.json}
    # and paper/generated/{stage2,difficulty}.tex (the published C rows, for the reproduction check);
    # writes results/explore/coverage/ask_act.{json,md} and paper/generated/coverage.tex (\res{cov/ask/...})

**Scoring** is S2-3's, unchanged: a backend's latest replay record per question decides; an answer is correct iff it is
in the occurrence's logged acceptable tool set (``bench.stage2_corpus.answer_correct``; a missing, failed or invalid
answer is incorrect); accuracy is the mean over a task's occurrences, then the mean over tasks
(``bench.stage2_corpus.task_accuracy``) on the corpus's eligible tasks. The ``all`` and ``after_off_plan`` strata must
reproduce the published C rows (``stage2_rl``, ``stage2_s1``, ``stage2_ladder``, ``stage2_v2`` keys of stage2.tex), and
``all``, ``after_off_plan`` and ``ask_now`` the matching columns of the difficulty appendix (``diff/c/{all,after_off_plan,
ask_gold}/<backend>/acc`` of difficulty.tex, Table tab:hardstates), to the printed digit; otherwise the script exits 1
and writes nothing.

**Strata** (fixed in code before any answer is read; the labels are the oracle's, logged at decision time):

* ``all`` -- every occurrence (the published row); ``after_off_plan`` -- the published off-plan stratum;
* ``ask_now`` -- the oracle's answer is ``ask_user`` (the right move is to ask the user now; another read may also be
  acceptable, as logged);
* ``ask_only`` -- the acceptable set is exactly {``ask_user``};
* ``ask_optional`` -- ``ask_user`` is acceptable but the oracle's answer is another tool;
* ``act`` -- ``ask_user`` is not acceptable (the right move is to act or read).

**Answer breakdown** (pooled occurrence counts, not task-weighted): on ``ask_now`` states, how many answers asked,
picked another acceptable read, *acted* (a state-changing tool: JevGym's ``STATE_CHANGING``), finished, picked another
wrong tool, or failed; on ``act`` states, the *over-ask* rate (answer ``ask_user``, task-weighted like accuracy).

**What C can and cannot show.** C holds the pre-action tool-choice states of four source agents on the first 200
fresh Stage-2 tasks, so the ask states are those of JevGym's ``clarify`` tasks only (one missing slot, asked once).
Replay answers are one decision per state: C cannot show whether a wrong answer would have been recovered later in a
live episode, and it never shows the cost of a needless question to a real user. JevAgentRL was trained on the same
generator and oracle (in-generator, like S2-3)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import coverage_tex as CT  # noqa: E402
from jevharness.bench import stage2_corpus as C  # noqa: E402
from jevharness.bench.suites.jevgym import JevGymEnv  # noqa: E402

S23 = ROOT / "results/stage2/s23"
PUBLISHED = (ROOT / "paper/generated/stage2.tex", ROOT / "paper/generated/difficulty.tex")
ASK = "ask_user"
FINISH = JevGymEnv.FINAL_TOOL
STATE_CHANGING = frozenset(JevGymEnv.STATE_CHANGING)
LABEL = "descriptive, post hoc (coverage of decision-model roles); S2-3's scoring on corpus C; not a registered analysis"

# replay name -> (display name, published key prefixes of its C row: <prefix>/corpus_acc[_offplan])
BACKENDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "rl": ("JevAgentRL-4B (frozen)", ("stage2_rl/frozen", "stage2_s1/frozen")),
    "v2": ("JevAgentRL-v2 (exploratory)", ("stage2_v2",)),
    "sft_v1": ("SFT v1 (JevAgentRL recipe before RL)", ("stage2_rl/sft_v1",)),
    "base": ("untrained base of JevAgentRL-4B", ("stage2_rl/base",)),
    "jev": ("Jev", ("stage2_rl/jev", "stage2_s1/jev")),
    "intern": ("Intern-Decision-4B", ("stage2_rl/intern", "stage2_s1/intern")),
    "sl0.8b": ("StartLux-Decision 0.8B", ("stage2_ladder/sl0.8b",)),
    "sl2b": ("StartLux-Decision 2B", ("stage2_ladder/sl2b",)),
    "sl4b": ("StartLux-Decision 4B", ("stage2_s1/sl4b",)),
    "sl9b": ("StartLux-Decision 9B", ("stage2_ladder/sl9b",)),
    "sl27b": ("StartLux-Decision 27B", ("stage2_s1/sl27b",)),
}


def acceptable(o: Mapping[str, Any]) -> set[str]:
    aux = o.get("gold_aux") or {}
    acc = aux.get("acceptable_tools") if isinstance(aux, Mapping) else None
    return set(acc) if acc else ({str(o["gold"])} if o.get("gold") is not None else set())


STRATA: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "all": lambda o: True,
    "after_off_plan": lambda o: o.get("stratum") == "after_off_plan",
    "ask_now": lambda o: o.get("gold") == ASK,
    "ask_only": lambda o: acceptable(o) == {ASK},
    "ask_optional": lambda o: ASK in acceptable(o) and o.get("gold") != ASK,
    "act": lambda o: ASK not in acceptable(o),
}
PUBLISHED_STRATA = {"all": "corpus_acc", "after_off_plan": "corpus_acc_offplan"}
DIFFICULTY_STRATA = {"all": "all", "after_off_plan": "after_off_plan", "ask_now": "ask_gold"}   # diff/c/<x>/<backend>/acc


def load_answers(path: Path) -> dict[str, dict[str, Any]]:
    """qhash -> the latest replay record of that question (as stage2_analysis.py reads it)."""
    recs: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["qhash"]] = r
    return recs


def answer_of(a: Mapping[str, Any] | None) -> str | None:
    a = a or {}
    return None if a.get("error") or not a.get("valid", True) else a.get("answer")


def mean_over_tasks(per_task: Mapping[str, float]) -> float:
    return float(np.mean(list(per_task.values()))) if per_task else float("nan")


def over_ask(occ: Sequence[Mapping[str, Any]], ans: Mapping[str, Mapping[str, Any]],
             tasks: Sequence[str]) -> float:
    """Task-weighted share of ``act`` occurrences answered with ``ask_user``."""
    keep, per = set(tasks), {}
    for o in occ:
        if o["task"] in keep:
            per.setdefault(o["task"], []).append(float(answer_of(ans.get(o["qhash"])) == ASK))
    return mean_over_tasks({t: sum(v) / len(v) for t, v in per.items()})


def breakdown(occ: Sequence[Mapping[str, Any]], ans: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Pooled answer kinds on ``ask_now`` occurrences (each counted once, first matching kind)."""
    c: Counter[str] = Counter()
    for o in occ:
        a = answer_of(ans.get(o["qhash"]))
        if a is None:
            c["failed"] += 1
        elif a == ASK:
            c["asked"] += 1
        elif a in acceptable(o):
            c["other_acceptable"] += 1
        elif a in STATE_CHANGING:
            c["acted"] += 1
        elif a == FINISH:
            c["finished"] += 1
        else:
            c["other_wrong"] += 1
    return {k: int(c.get(k, 0)) for k in ("asked", "other_acceptable", "acted", "finished", "other_wrong", "failed")}


def read_published(paths: Sequence[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        if Path(path).exists():
            for m in re.finditer(r"\\defres\{([^}]*)\}\{([^}]*)\}", Path(path).read_text()):
                out[m.group(1)] = m.group(2)
    return out


def analyse(s23: Path = S23, published: Sequence[Path] = PUBLISHED) -> dict[str, Any]:
    meta = json.loads((s23 / "corpus.json").read_text())
    occ = C.read_jsonl_gz(s23 / "occurrences.jsonl.gz")
    tasks = [str(t) for t in meta.get("eligible_tasks") or []]
    keep = set(tasks)
    occ = [o for o in occ if o["task"] in keep]
    strata = {name: [o for o in occ if f(o)] for name, f in STRATA.items()}
    sizes = {name: {"occurrences": len(sub), "tasks": len({o["task"] for o in sub})} for name, sub in strata.items()}
    pub = read_published(published)
    rows: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    for name, (display, prefixes) in BACKENDS.items():
        ap = s23 / f"answers_{name}.jsonl"
        if not ap.exists():
            continue
        ans = load_answers(ap)
        rp = s23 / f"replay_{name}.json"
        row: dict[str, Any] = {"display": display, "answers": CT.rel(ap), "n_answers": len(ans),
                               "backend": (json.loads(rp.read_text()).get("backend") if rp.exists() else None),
                               "acc": {}, "tasks": {}}
        for sname, sub in strata.items():
            per = C.task_accuracy(sub, ans, tasks)
            row["acc"][sname] = mean_over_tasks(per)
            row["tasks"][sname] = len(per)
        row["ask_now_answers"] = breakdown(strata["ask_now"], ans)
        row["ask_only_answers"] = breakdown(strata["ask_only"], ans)
        row["over_ask"] = over_ask(strata["act"], ans, tasks)
        keys = [(sname, f"{pre}/{field}") for sname, field in PUBLISHED_STRATA.items() for pre in prefixes]
        keys += [(sname, f"diff/c/{sub}/{name}/acc") for sname, sub in DIFFICULTY_STRATA.items()]
        for sname, key in keys:
            if key in pub:
                mine = CT.pct(row["acc"][sname])
                checks.append({"backend": name, "stratum": sname, "key": key, "published": pub[key],
                               "reproduced": mine, "ok": pub[key] == mine})
        rows[name] = row
    return {"meta": {"generated": CT.now(), "git": CT.git_head(), "label": LABEL, "corpus": CT.rel(s23),
                     "registered_corpus": meta.get("registered"), "published": [CT.rel(p) for p in published],
                     "n_eligible_tasks": len(tasks), "state_changing": sorted(STATE_CHANGING)},
            "strata": sizes, "rows": rows, "checks": checks,
            "reproduced": bool(checks) and all(c["ok"] for c in checks),
            "limits": LIMITS}


LIMITS = [
    "C holds pre-action tool-choice states of four source agents on the first 200 fresh Stage-2 tasks; the ask states "
    "come from JevGym's clarify tasks only (one missing slot, asked once).",
    "Each state is scored as one replayed decision: C cannot show whether a wrong answer would have been recovered "
    "later in a live episode, nor the cost of a needless question to a real user.",
    "JevAgentRL (and SFT v1, v2) were trained on the same generator and oracle: their rows are in-generator.",
    "An 'acted' answer on an ask_now state is a state-changing call before the required question; JevGym's strict mode "
    "ends the episode at such a call, so these are irreversible errors in a live run (a replace-mode agent would make "
    "them; a cascade defers the low-confidence ones).",
    "The ask_now accuracy is the 'ask_user next' column of the difficulty appendix (Table tab:hardstates; same states, "
    "same scoring, checked above); new here are the answer kinds, the ask-only / ask-optional split and the over-ask "
    "rate.",
    "Descriptive and post hoc: no test, no interval; the strata were fixed in code before any answer was read.",
]


def tex_map(res: Mapping[str, Any]) -> dict[str, str]:
    tex: dict[str, str] = {}
    for sname, s in res["strata"].items():
        CT.put(tex, f"cov/ask/{sname}/occurrences", CT.count(s["occurrences"]))
        CT.put(tex, f"cov/ask/{sname}/tasks", CT.count(s["tasks"]))
    for name, r in res["rows"].items():
        for sname, v in r["acc"].items():
            CT.put(tex, f"cov/ask/{name}/{sname}/acc", CT.pct(v))
        for part in ("ask_now", "ask_only"):
            for k, n in r[f"{part}_answers"].items():
                CT.put(tex, f"cov/ask/{name}/{part}/{k}", CT.count(n))
        CT.put(tex, f"cov/ask/{name}/act/over_ask", CT.pct(r["over_ask"]))
    return tex


def to_markdown(res: Mapping[str, Any]) -> str:
    m, s = res["meta"], res["strata"]
    L = ["# Ask-vs-act states on the common-state corpus C (descriptive, post hoc)", "",
         f"Generated {m['generated']} at git `{str(m['git'])[:12]}` by `scripts/coverage_ask_act.py` from "
         f"`{m['corpus']}` ({m['registered_corpus']}); {m['n_eligible_tasks']} eligible tasks. {LABEL}.", "",
         "Accuracy = S2-3's score (answer in the logged acceptable tool set; mean over a task's occurrences, then over "
         "tasks). Strata: `ask_now` = the oracle's answer is `ask_user`; `ask_only` = `ask_user` is the only "
         "acceptable tool; `ask_optional` = `ask_user` acceptable, the oracle's answer another tool; `act` = "
         "`ask_user` not acceptable. Over-ask = share of `act` states answered `ask_user` (task-weighted).", "",
         "| backend | " + " | ".join(f"{k} ({s[k]['occurrences']:,} / {s[k]['tasks']} tasks)" for k in STRATA)
         + " | over-ask on act |",
         "|---|" + "---|" * (len(STRATA) + 1)]
    for r in res["rows"].values():
        L.append(f"| {r['display']} | " + " | ".join(CT.pct(r["acc"][k]) or "–" for k in STRATA)
                 + f" | {CT.pct(r['over_ask']) or '–'} |")
    for part in ("ask_now", "ask_only"):
        L += ["", f"## Answers on the {s[part]['occurrences']:,} `{part}` states (pooled counts)", "",
              "acted = a state-changing call (" + ", ".join(f"`{t}`" for t in m["state_changing"])
              + ") instead of asking.", "",
              "| backend | asked | other acceptable read | acted | finished | other wrong tool | failed |",
              "|---|---|---|---|---|---|---|"]
        for r in res["rows"].values():
            b = r[f"{part}_answers"]
            L.append(f"| {r['display']} | " + " | ".join(str(b[k]) for k in
                                                        ("asked", "other_acceptable", "acted", "finished",
                                                         "other_wrong", "failed")) + " |")
    ok = sum(c["ok"] for c in res["checks"])
    L += ["", f"## Reproduction of the published C rows: {ok} / {len(res['checks'])} keys match", "",
          "| backend | stratum | key | published | reproduced |", "|---|---|---|---|---|"]
    for c in res["checks"]:
        L.append(f"| {c['backend']} | {c['stratum']} | `{c['key']}` | {c['published']} | {c['reproduced']}"
                 + ("" if c["ok"] else " **MISMATCH**") + " |")
    L += ["", "## What these states can and cannot show", ""] + [f"- {x}" for x in res["limits"]] + [""]
    return "\n".join(L)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s23", default=str(S23))
    ap.add_argument("--published", nargs="*", default=[str(p) for p in PUBLISHED],
                    help="generated TeX files with the published C rows (none = skip the check)")
    ap.add_argument("--out-dir", default=str(CT.OUT_DIR))
    ap.add_argument("--tex", default=str(CT.TEX_OUT))
    a = ap.parse_args(argv)
    res = analyse(Path(a.s23), [Path(p) for p in a.published])
    if a.published and not res["reproduced"]:
        bad = [c for c in res["checks"] if not c["ok"]] or ["no published key found"]
        print(f"ERROR: the published C rows are not reproduced: {bad}", file=sys.stderr)
        return 1
    res["tex"] = tex_map(res)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ask_act.json").write_text(json.dumps(res, indent=1, sort_keys=False) + "\n")
    (out / "ask_act.md").write_text(to_markdown(res))
    tex = CT.write_tex(out, Path(a.tex))
    r = res["rows"]
    print(f"ask_now ({res['strata']['ask_now']['occurrences']} occurrences, {res['strata']['ask_now']['tasks']} tasks): "
          + ", ".join(f"{k} {CT.pct(v['acc']['ask_now'])}" for k, v in r.items()))
    print(f"reproduced {len(res['checks'])} published keys; wrote {CT.rel(out / 'ask_act.json')}, "
          f"{CT.rel(out / 'ask_act.md')}, {CT.rel(tex)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
