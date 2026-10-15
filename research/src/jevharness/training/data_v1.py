"""Training data v1: decision states from the harness's *own* state distribution on the evaluation setting, with Jev
soft labels (``scripts/train/make_data_v1.sh``; plan and gap diagnosis: docs/TRAINING_RECIPES.md §18).

Why (results/gpu_round2/NOTES.md §1-§3, results/training/train_v1/MANIFEST.md): Intern-Decision-4B scores 0.862 on the
210 static gate decisions of ``data/train_v0/dev.jsonl`` but 0.76 tool_select accuracy in shadow on the states the
harness visits and 0.41 when its own choices drive the agent (replace). ``data/train_v0`` cannot close that gap: its
states come from Gemini 3.5 Flash episodes that all succeeded (no state after a wrong action), 47 % of its tool_select
rows come from the v1 families / 30 mixed distractors / no noise (the frozen evaluation suite is v2 clarify + multi,
60 near distractors, 20 noise lines), its states are ~3.4x shorter at the median than in-harness states, it has no
tool-guard question and its soft labels come from the Gemini stand-in, not Jev.

v1 therefore collects decisions with the real harness (``bench.runner.run_experiment``, ``JevAgent``) on JevGym
TRAINING seeds in the evaluation setting, under three behaviour policies (the executor is always Gemini 3.5
Flash-Lite, the cheap System-2):

  lite_shadow       Flash-Lite acts alone; System-1 is asked in shadow on tool_select / tool_params / stop
                    -> natural states of a weaker executor, including its own mistakes (17 % failed episodes in v1)
  nearmiss_replace  System-1 = :class:`NearMissBackend` in replace on tool_select + tool_params (stop in shadow):
                    the JevGym oracle with *near-miss* errors injected at rate eps -- skipping the lookup step,
                    acting instead of asking, the near-duplicate tool, a confusable ID -- the error modes of
                    open deciders in the harness. DAgger-style: the states AFTER such errors are what an S1 in
                    replace / cascade reaches and what no teacher-driven log contains
  lite_guard        Flash-Lite proposes, System-1 (oracle with eps flips) guards every tool call and the stop
                    -> the guard Noul questions on real proposals, wrong ones included, and post-rejection states

``NearMissBackend`` is a *simulated* component: it reads the oracle gold from ``Question.meta`` in-process. It only
chooses which states get visited; it never produces a label. Labels are the JevGym oracle gold (always) and Jev's
distribution at the same state (``teacher_label.label_file``, real backend: ``Question.meta`` is never on the wire),
mixed q = alpha * gold + (1 - alpha) * p_Jev; rows where Jev's argmax disagrees with gold keep the gold one-hot.

Guards (``check_seed_plan``, ``verify_exclusions``; tests in tests/test_training_data_v1.py): every JevGym seed is
>= 10000 and outside the evaluation (200000-200499), retired (100000-100499) and calibration (5000-5999) ranges; no
task of the 210-decision gate set (``data/train_v0/dev.jsonl``) enters v1; no row carries an id of the E1 dev prefix
(``configs/eval/e1_dev_prefix_ids.json``) and no external replay row shares an 8-gram (or its whole short text) with
a BFCL or When2Call TEST question.

Outputs (``data/train_v1``, gzipped like v0's replay files; every loader reads ``.jsonl.gz`` transparently; ~12 MB,
small enough to ship in git as v0 was): ``train.jsonl.gz`` (Jev-soft targets), ``dev.jsonl.gz`` (gold; 15 % of the
tasks, natural distribution), ``dev_recovery.jsonl.gz`` (the dev rows that follow an off-plan step),
``dev_knockout.jsonl.gz`` (the knockout-final variants of the dev tool_select rows), ``replay_when2call.jsonl.gz``,
``train.jsonl.summary.json`` / ``dev_teacher.summary.json`` (teacher Ledgers and metrics), ``build_info.json``,
``MANIFEST.json``. Bulky intermediates stay out of git under ``data/cache/`` (ignored): the trajectories
(``data/cache/train_v1/train_v1_jevgym``), the gold train rows and per-arm records (``data/cache/train_v1/_work``) and
the teacher cache (``data/cache/teacher_v1.sqlite``); ``MANIFEST.md`` gives the command that regenerates them.

  python -m jevharness.training.data_v1 collect --out data/cache/train_v1 --n 600      # episodes ($: Flash-Lite)
  python -m jevharness.training.data_v1 build --runs data/cache/train_v1/train_v1_jevgym --out data/train_v1
  python -m jevharness.training.data_v1 label --out data/train_v1 --teacher jev-openrouter   # Jev soft labels ($)
  python -m jevharness.training.data_v1 manifest --out data/train_v1 --md results/training/train_v1/MANIFEST.md
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..bench.registry import TRAIN_MIN_SEED, in_ranges
from ..decision.base import DecisionBackend
from ..decision.types import DecisionResult, QKind, Question, Usage, answer_from_distribution
from ..harness.tools import OTHER
from .data import harvest_trajectories, open_text, read_jsonl, split_by_task, stats, write_jsonl
from .harvest import jevgym_seed, tool_descriptions
from .seed_guard import TRAIN_FORBIDDEN

# the frozen evaluation setting (configs/eval/jevgym_eval_tasks.json "suite"), minus its seeds
EVAL_SETTING: dict[str, Any] = {"name": "jevgym", "families": "v2", "distractor_style": "near", "n_distractors": 60,
                                "noise_lines": 20, "strict": True}
V1_SEED0 = 30000          # v0 used 10000-10029 / 12000-12059; round-4 OPD rollouts use 40000+ (12_sft_v1.sh)
EXECUTOR = {"provider": "gemini", "model": "gemini-3.5-flash-lite", "temperature": 0, "reasoning_effort": "minimal"}
GATE_DEV = "data/train_v0/dev.jsonl"
E1_DEV_PREFIX = "configs/eval/e1_dev_prefix_ids.json"
# evaluation + retired + calibration + the Stage-2 confirmation range (training/seed_guard.py)
FORBIDDEN_RANGES: tuple[tuple[int, int], ...] = TRAIN_FORBIDDEN

FINISH = "finish"
# the step an open decider skips to (round 2: "skips get_orders after find_user, calls cancel_order where gold is
# ask_user"): gold -> the tempting later / premature choices
SKIP_TO: dict[str, tuple[str, ...]] = {
    "find_user": ("get_orders", "cancel_order", "ship_order", "refund"),
    "get_orders": ("cancel_order", "ship_order", "refund", FINISH),
    "ask_user": ("cancel_order", "ship_order", "refund", "find_user", "get_orders"),
    "check_stock": (FINISH,),
    "cancel_order": (FINISH,), "ship_order": (FINISH,), "refund": (FINISH,), "send_message": (FINISH,),
}
TOOL_MODES = {"skip": 0.4, "near": 0.35, "over_ask": 0.1, "random": 0.15}
PARAM_MODES = {"confusable": 0.6, "other": 0.2, "random": 0.2}


# =============================================================================== seed guard
def check_seed_plan(seed0: int, n: int, min_seed: int = TRAIN_MIN_SEED,
                    forbidden: Sequence[tuple[int, int]] = FORBIDDEN_RANGES) -> None:
    """Raise ``ValueError`` unless every seed of ``seed0 .. seed0 + n - 1`` is a training seed: >= ``min_seed`` and
    outside the evaluation, retired-evaluation and calibration ranges (bench/registry.py)."""
    if n <= 0:
        raise ValueError(f"empty seed plan (n={n})")
    if seed0 < min_seed:
        raise ValueError(f"training data needs JevGym seeds >= {min_seed}; got seed0={seed0}")
    last = seed0 + n - 1
    hit = [(lo, hi) for lo, hi in forbidden if seed0 <= hi and last >= lo]
    if hit:
        raise ValueError(f"seeds {seed0}-{last} overlap reserved range(s) {hit} (evaluation / retired / calibration)")


def seed_ok(seed: int | None, min_seed: int = TRAIN_MIN_SEED,
            forbidden: Sequence[tuple[int, int]] = FORBIDDEN_RANGES) -> bool:
    """A row may enter v1 iff it is not a JevGym row (None) or its seed passes ``check_seed_plan``'s rule."""
    if seed is None:
        return True
    return seed >= min_seed and not in_ranges(seed, tuple(forbidden))


# =============================================================================== behaviour policy
def near_table() -> dict[str, list[str]]:
    """core tool -> the JevGym near-duplicate distractors that shadow it (``NEAR_TOOLS``)."""
    out: dict[str, list[str]] = defaultdict(list)
    try:
        from ..bench.suites.jevgym import NEAR_TOOLS
    except Exception:  # noqa: BLE001 - callers fall back to string similarity
        return out
    for t in NEAR_TOOLS:
        out[t.shadows].append(t.name)
    return out


def _shape(v: str) -> str:
    """Coarse lexical shape of an argument value: ORD-4759 -> A-9, U8166 -> A9, Mateo Okafor -> Aa Aa."""
    s = re.sub(r"[A-Z]+", "A", v)
    s = re.sub(r"[a-z]+", "a", s)
    return re.sub(r"\d+", "9", s)


class NearMissBackend(DecisionBackend):
    """Behaviour policy for DAgger-style state collection (simulation component; never a label source, never a
    result). Answers every typed question with the JevGym oracle gold that the harness puts into
    ``Question.meta['gold']`` (in-process only), except that with probability ``eps_*`` it makes a *near miss*:

      tool_select  skip     a later / premature step (SKIP_TO: find_user -> get_orders / an action, ask_user -> act)
                   near     a near-duplicate of the gold tool (JevGym NEAR_TOOLS: get_order for get_orders, ...)
                   over_ask ask_user when the gold is another tool
                   random   any other tool
      tool_params  confusable another option of the same lexical shape (another order id, user id, name, enum value)
                   other    the OTHER option (or a candidate when the gold is OTHER)
                   random   any other option
      Noul gates   the flipped verdict

    Answers are deterministic per (seed, state, question, gold). Questions without a gold label get a random option
    (Choice) or 'no' (Noul). The answered distribution puts ``conf`` on the chosen option; the harness acts on its
    argmax (replace) or on p_yes >= tau (guard), so the confidence value does not matter here.
    ``counts`` records how often each error mode fired (written to the run's manifest)."""

    name = "nearmiss"
    batched_questions = True
    max_options = 255   # no tournament: the policy sees every option at once (rows are capped at harvest time)

    def __init__(self, eps_tool: float = 0.2, eps_param: float = 0.15, eps_gate: float = 0.1, conf: float = 0.9,
                 seed: int = 0, tool_modes: dict[str, float] | None = None,
                 param_modes: dict[str, float] | None = None):
        for k, v in (("eps_tool", eps_tool), ("eps_param", eps_param), ("eps_gate", eps_gate)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{k} must be in [0, 1]; got {v}")
        self.eps_tool, self.eps_param, self.eps_gate = eps_tool, eps_param, eps_gate
        self.conf = conf
        self.seed = seed
        self.tool_modes = dict(tool_modes or TOOL_MODES)
        self.param_modes = dict(param_modes or PARAM_MODES)
        self.model = f"nearmiss(t={eps_tool},p={eps_param},g={eps_gate},s={seed})"
        self.counts: Counter = Counter()
        self._near = near_table()

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "eps_tool": self.eps_tool, "eps_param": self.eps_param,
                "eps_gate": self.eps_gate, "seed": self.seed, "tool_modes": self.tool_modes,
                "param_modes": self.param_modes}

    # ------------------------------------------------------------------ helpers
    def _rng(self, state: str, q: Question) -> random.Random:
        key = json.dumps([self.seed, state, q.kind.value, q.text, q.options, str(q.meta.get("gold"))], default=str)
        return random.Random(hashlib.sha256(key.encode()).hexdigest())

    @staticmethod
    def _pick_mode(modes: dict[str, float], cands: dict[str, list[str]], rng: random.Random) -> str | None:
        live = [(m, w) for m, w in modes.items() if w > 0 and cands.get(m)]
        if not live:
            return None
        r = rng.random() * sum(w for _, w in live)
        for m, w in live:
            r -= w
            if r <= 0:
                return m
        return live[-1][0]

    def _tool_candidates(self, gold: str, labels: list[str]) -> dict[str, list[str]]:
        avail = set(labels) - {gold}
        near = [t for t in self._near.get(gold, []) if t in avail]
        if not near:  # fallback: lexically closest names
            near = [t for t in difflib.get_close_matches(gold, sorted(avail), n=3, cutoff=0.6)]
        return {"skip": [t for t in SKIP_TO.get(gold, ()) if t in avail],
                "near": near,
                "over_ask": ["ask_user"] if "ask_user" in avail else [],
                "random": sorted(avail)}

    @staticmethod
    def _param_candidates(gold: str, labels: list[str]) -> dict[str, list[str]]:
        avail = [o for o in labels if o != gold]
        if gold == OTHER:
            return {"confusable": [], "other": [o for o in avail if o != OTHER], "random": avail}
        same = [o for o in avail if o != OTHER and _shape(o) == _shape(gold)]
        return {"confusable": same, "other": [OTHER] if OTHER in avail else [], "random": avail}

    def _choose(self, state: str, q: Question) -> tuple[str, str]:
        """(label to answer, what happened: 'gold' | mode | 'no_gold')."""
        labels = q.labels()
        rng = self._rng(state, q)
        gold = q.meta.get("gold")
        if q.kind == QKind.NOUL:
            if not isinstance(gold, bool):
                return "no", "no_gold"
            g = "yes" if gold else "no"
            if rng.random() < self.eps_gate:
                return ("no" if gold else "yes"), "flip"
            return g, "gold"
        gold = None if gold is None else str(gold)
        if gold not in labels:
            return rng.choice(labels), "no_gold"
        if q.decision_point == "tool_params":
            eps, modes, cands = self.eps_param, self.param_modes, self._param_candidates(gold, labels)
        else:
            eps, modes, cands = self.eps_tool, self.tool_modes, self._tool_candidates(gold, labels)
        if rng.random() >= eps:
            return gold, "gold"
        mode = self._pick_mode(modes, cands, rng)
        if mode is None:
            return gold, "gold"
        return rng.choice(cands[mode]), mode

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        answers = []
        for q in questions:
            pick, what = self._choose(state, q)
            self.counts[f"{q.decision_point}:{what}"] += 1
            labels = q.labels()
            rest = (1.0 - self.conf) / max(1, len(labels) - 1)
            answers.append(answer_from_distribution(q, {lab: (self.conf if lab == pick else rest) for lab in labels}))
        return DecisionResult(answers=answers, usage=Usage(backend=self.name, model=self.model, input_tokens=0,
                                                           cost_usd=0.0, latency_ms=0.0, priced=True))


def behaviour_arms(stop_mode: str = "shadow") -> list[dict[str, Any]]:
    """The three behaviour arms (module docstring). All bind the experiment's default System-1 (the NearMiss
    policy passed to ``run_experiment``); ``lite_shadow`` never acts on it."""
    return [
        {"name": "lite_shadow", "tool_select": {"mode": "shadow"}, "tool_params": {"mode": "shadow"},
         "stop": {"mode": "shadow"}},
        {"name": "nearmiss_replace", "tool_select": {"mode": "replace"}, "tool_params": {"mode": "replace"},
         "stop": {"mode": stop_mode}},
        {"name": "lite_guard", "tool_select": {"mode": "guard", "tau": 0.5}, "stop": {"mode": "guard", "tau": 0.5}},
    ]


def build_experiment(name: str = "train_v1_jevgym", seed0: int = V1_SEED0, n: int = 300,
                     n_by_arm: dict[str, int] | None = None, llm: dict[str, Any] | None = None, workers: int = 8,
                     suite: dict[str, Any] | None = None, arms: list[dict[str, Any]] | None = None
                     ) -> dict[str, Any]:
    """The runner experiment of the v1 episodes (``bench.runner`` YAML schema). ``n_by_arm`` sets per-arm task
    counts (the first k tasks of the suite; arms share their prefix, so step-0 states are deduplicated later)."""
    check_seed_plan(seed0, n)
    arms = [dict(a) for a in (arms or behaviour_arms())]
    for a in arms:
        if n_by_arm and a["name"] in n_by_arm:
            a["n_tasks"] = int(min(n, n_by_arm[a["name"]]))
    return {"name": name, "suite": {**EVAL_SETTING, **(suite or {}), "n": int(n), "seed0": int(seed0)},
            "llm": dict(llm or EXECUTOR), "s1": {"type": "nearmiss (in-process, data_v1.NearMissBackend)"},
            "s1_backends": {}, "workers": int(workers), "arms": arms, "sweeps": [], "retry_errors": True}


def collect(exp: dict[str, Any], out_dir: str | Path, s1: DecisionBackend | None = None,
            max_tasks: int | None = None, progress: bool = True) -> dict[str, Path]:
    """Run the behaviour arms (resumable: finished tasks are skipped). Returns arm -> trajectory JSONL."""
    from ..bench.runner import run_experiment

    suite = exp["suite"]
    check_seed_plan(int(suite["seed0"]), int(suite["n"]))
    s1 = s1 or NearMissBackend()
    paths = run_experiment(exp, out_dir, s1=s1, max_tasks=max_tasks, progress=progress)
    out = Path(out_dir) / exp["name"]
    if isinstance(s1, NearMissBackend):
        prev = {}
        f = out / "nearmiss_counts.json"
        if f.exists():
            prev = json.loads(f.read_text())
        merged = Counter(prev) + s1.counts
        f.write_text(json.dumps(dict(sorted(merged.items())), indent=1))
    return paths


# =============================================================================== harvest
def _latest_ok(path: Path) -> list[dict[str, Any]]:
    """Latest record per task (the runner appends retried episodes), episodes that ended in an error dropped."""
    from ..harness.trajectory import load_trajectories

    return [t for t in load_trajectories(path) if not t.get("error")]


def off_plan_steps(traj: dict[str, Any]) -> list[int]:
    """Steps whose executed tool differs from the oracle's canonical tool at that state (``StepLog.oracle_tool``)."""
    return [st["step"] for st in traj.get("steps", [])
            if st.get("oracle_tool") and st.get("tool") and st["tool"] != st["oracle_tool"]]


SYN_GUARD_ARMS = ("nearmiss_replace", "lite_shadow")


def wrong_proposal_rows(traj: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    """Guard questions about the WRONG calls an episode actually executed: a tool outside the oracle's acceptable
    set AT(s) (``gold_aux.acceptable_tools`` of the step's tool_select record) becomes a tool-guard Noul labelled
    'no' (``q_tool_guard`` with the executed arguments: AT(s) holds every tool whose call can be progress, so the
    proposal is not), and a premature ``finish`` (finish outside AT(s)) a stop-guard Noul labelled 'no'
    (``q_stop_guard`` with the executed answer). State = the step's System-1 state. The guard arms log almost only
    correct proposals (v1: 32 'no' of 1,199 tool-guard rows), so without these rows a guard learns "always yes"."""
    from ..harness.prompts import q_stop_guard, q_tool_guard

    out = []
    for st in traj.get("steps", []):
        tool = st.get("tool")
        d = next((d for d in st.get("decisions", []) if d.get("dp") == "tool_select" and d.get("options")
                  and d.get("state")), None)
        if not tool or d is None:
            continue
        acc = (d.get("gold_aux") or {}).get("acceptable_tools")
        if not acc or tool in acc:
            continue
        args = st.get("args") or {}
        if tool == FINISH:
            q, dp, tag = q_stop_guard(args.get("answer", args.get("final_answer"))), "stop", "stop:guard_syn"
        else:
            q, dp, tag = q_tool_guard(tool, args), "tool_select", "tool_select:guard_syn"
        out.append({"id": f"{traj['task_id']}/{arm}:{st['step']}/{tag}", "source": f"syn_guard:{arm}", "dp": dp,
                    "kind": "noul", "state": d["state"], "question": q, "options": ["yes", "no"],
                    "target": {"yes": 0.0, "no": 1.0}, "q_values": None,
                    "meta": {"label_source": "gold", "mode": "guard_syn", "executed": tool,
                             "acceptable_tools": acc, "label_rule": "executed call outside AT(s) -> not progress"}})
    return out


def harvest_runs(run_dir: str | Path, arms: Iterable[str] | None = None, max_options: int = 52, seed: int = 0,
                 work_dir: str | Path | None = None, syn_guard: bool = True
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gold-labelled decision rows from the behaviour arms of one run directory (``<arm>.jsonl`` +
    ``experiment.yaml``). Row ids are ``<task>/<arm>:<step>/<dp>[...]`` (task first: ``split_by_task`` and
    ``jevgym_seed`` read it). Each row's meta gets the arm, its JevGym seed, the number of off-plan steps executed
    before it (``after_error``: the recovery states) and the episode outcome. Returns (rows, episode summary)."""
    run_dir = Path(run_dir)
    exp_path = run_dir / "experiment.yaml"
    files = sorted(p for p in run_dir.glob("*.jsonl") if not p.name.endswith(".latest.jsonl"))
    if arms is not None:
        files = [p for p in files if p.stem in set(arms)]
    work = Path(work_dir or run_dir)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"run": str(run_dir), "arms": {}}
    for p in files:
        trajs = _latest_ok(p)
        n_err = sum(1 for line in open_text(p) if line.strip()) - len(trajs)
        latest = work / f"{p.stem}.latest.jsonl"
        write_jsonl(trajs, latest)
        tids = {t["task_id"] for t in trajs}
        desc = tool_descriptions(exp_path, tids) if exp_path.exists() else {}
        got = harvest_trajectories([latest], label_source="gold", descriptions=desc, max_options=max_options,
                                   seed=seed)
        n_nat = len(got)
        if syn_guard and p.stem in SYN_GUARD_ARMS:
            got += [r for t in trajs for r in wrong_proposal_rows(t, p.stem)]
        off = {t["task_id"]: off_plan_steps(t) for t in trajs}
        succ = {t["task_id"]: bool(t["result"].get("success")) for t in trajs}
        cost = sum(float(((t.get("usage") or {}).get("total") or {}).get("cost_usd", 0.0)) for t in trajs)
        for r in got[:n_nat]:
            task, rest = r["id"].split("/", 1)
            r["id"] = f"{task}/{p.stem}:{rest}"
            r["source"] = f"traj:{p.stem}"
        for r in got:
            task, rest = r["id"].split("/", 1)
            step = int(rest.split("/", 1)[0].split(":")[-1])
            r["meta"] = {**r["meta"], "arm": p.stem, "jevgym_seed": jevgym_seed(task), "step": step,
                         "after_error": sum(1 for s in off.get(task, []) if s < step),
                         "episode_success": succ.get(task)}
            rows.append(r)
        specs = Counter(json.dumps(((t.get("meta") or {}).get("s1") or {}).get("describe"), sort_keys=True)
                        for t in trajs)
        summary["arms"][p.stem] = {"episodes": len(trajs), "superseded_or_errored_records": max(0, n_err),
                                   "system1_behaviour": [json.loads(k) for k in specs],
                                   "executor": sorted({str(t.get("llm")) for t in trajs}),
                                   "success": sum(succ.values()), "cost_usd": round(cost, 6),
                                   "decisions": n_nat, "wrong_proposal_rows": len(got) - n_nat,
                                   "off_plan_episodes": sum(1 for v in off.values() if v)}
    summary["cost_usd"] = round(sum(a["cost_usd"] for a in summary["arms"].values()), 6)
    f = run_dir / "nearmiss_counts.json"
    if f.exists():  # every System-1 question the behaviour policy answered (shadow questions included)
        summary["nearmiss_counts"] = json.loads(f.read_text())
    return rows, summary


# =============================================================================== dedupe / exclusions / balance
def row_key(r: dict[str, Any]) -> str:
    """Identity of a decision for deduplication: same decision point, state, question, option set and gold."""
    t = r.get("target") or {}
    gold = max(t, key=t.get) if t else None
    key = json.dumps([r.get("dp"), r.get("kind"), r["state"], r["question"], sorted(r["options"]), gold])
    return hashlib.sha1(key.encode()).hexdigest()


def dedupe(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the first row of every ``row_key``; the kept row lists every arm that produced it (``meta.arms``)."""
    seen: dict[str, dict[str, Any]] = {}
    out = []
    for r in rows:
        k = row_key(r)
        if k in seen:
            arms = seen[k]["meta"].setdefault("arms", [seen[k]["meta"].get("arm")])
            a = (r.get("meta") or {}).get("arm")
            if a not in arms:
                arms.append(a)
            continue
        seen[k] = r
        out.append(r)
    return out, len(rows) - len(out)


def task_of(row_id: str) -> str:
    return row_id.split("/", 1)[0]


def e1_parent(item_id: str) -> str:
    """Source item of an E1 / training row id: ``irrelevance_14/tool`` -> ``irrelevance_14``,
    ``w2c/<uuid>[/...]`` -> ``w2c/<uuid>`` (When2Call test ids carry a namespace segment)."""
    parts = item_id.split("/")
    return "/".join(parts[:2]) if parts[0] == "w2c" and len(parts) > 1 else parts[0]


def gate_dev_tasks(path: str | Path = GATE_DEV) -> set[str]:
    """Task ids of the 210-decision competence-gate dev set (A5.1). They never enter v1 train."""
    if not Path(path).exists():
        return set()
    return {task_of(r["id"]) for r in read_jsonl(path)}


def e1_dev_prefix_ids(path: str | Path = E1_DEV_PREFIX) -> set[str]:
    if not Path(path).exists():
        return set()
    return set(json.loads(Path(path).read_text()).get("ids") or [])


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    t = _tokens(text)
    if len(t) < n:
        return {tuple(t)} if t else set()
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def query_text(row: dict[str, Any]) -> str:
    """The free-text user request of a row (When2Call / xLAM: after ``USER:``; JevGym: the ``TASK:`` line)."""
    s = row.get("state") or ""
    if "\nUSER: " in s or s.startswith("USER: "):
        return s.rpartition("USER: ")[2]
    m = re.match(r"TASK:\s*(.*)", s)
    return m.group(1) if m else s[:500]


def eval_texts(data_dir: str | Path = "data/raw") -> list[str]:
    """User questions of every evaluation item that could collide with replay data: BFCL (all six tool categories,
    every item -- E1 and C8 use subsets) and When2Call TEST. Read from the local copies (no download)."""
    from ..bench.suites.static import BFCL_TOOL_CATS, _query_text

    out: list[str] = []
    d = Path(data_dir)
    for cat in BFCL_TOOL_CATS:
        p = d / "bfcl" / f"BFCL_v4_{cat}.json"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    out.append(_query_text(json.loads(line)["question"]))
    p = d / "when2call" / "test" / "when2call_test_mcq.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line)["question"])
    return out


class Decontaminator:
    """8-gram overlap test against the evaluation questions (TRAINING_RECIPES §3). A text shorter than n tokens
    matches only an evaluation n-gram set equal to its own tokens (exact short-text match)."""

    def __init__(self, texts: Iterable[str], n: int = 8):
        self.n = n
        self.grams: set[tuple[str, ...]] = set()
        self.n_texts = 0
        for t in texts:
            self.grams |= ngrams(t, n)
            self.n_texts += 1

    def hit(self, text: str) -> bool:
        return bool(ngrams(text, self.n) & self.grams)


def verify_exclusions(rows: Iterable[dict[str, Any]], *, gate_tasks: set[str], e1_ids: set[str],
                      decon: Decontaminator | None = None, check_text: bool = True,
                      ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop every row that breaks a hold-out rule and count why. Rules: (1) JevGym seed passes ``seed_ok``;
    (2) its task is not a gate-dev task; (3) its id (or its task id) is not an E1 dev-prefix id; (4) for non-JevGym
    rows, its user text shares no 8-gram with a BFCL / When2Call test question (``decon``). ``check_text`` applies
    rule 4 to JevGym rows as well (their TASK line)."""
    kept: list[dict[str, Any]] = []
    why: Counter = Counter()
    e1_tasks = {e1_parent(i) for i in e1_ids}
    for r in rows:
        rid = r["id"]
        seed = jevgym_seed(rid)
        if not seed_ok(seed):
            why["reserved_or_low_seed"] += 1
            continue
        if task_of(rid) in gate_tasks:
            why["gate_dev_task"] += 1
            continue
        if rid in e1_ids or e1_parent(rid) in e1_tasks:
            why["e1_dev_prefix_id"] += 1
            continue
        if decon is not None and (seed is None or check_text) and decon.hit(query_text(r)):
            why["ngram_overlap_eval"] += 1
            continue
        kept.append(r)
    return kept, {"dropped": dict(why), "kept": len(kept)}


def assert_clean(rows: Iterable[dict[str, Any]], *, gate_tasks: set[str], e1_ids: set[str]) -> None:
    """Raise ``AssertionError`` if any row breaks a seed / gate-dev / E1-id rule (the final check before writing)."""
    e1_tasks = {e1_parent(i) for i in e1_ids}
    for r in rows:
        seed = jevgym_seed(r["id"])
        assert seed_ok(seed), f"reserved or low JevGym seed in {r['id']}"
        assert task_of(r["id"]) not in gate_tasks, f"gate-dev task in {r['id']}"
        assert r["id"] not in e1_ids and e1_parent(r["id"]) not in e1_tasks, f"E1 dev-prefix id {r['id']}"


def bucket(r: dict[str, Any]) -> str:
    """Balancing bucket: decision point x question type x gold label for the binary gates."""
    t = r.get("target") or {}
    gold = max(t, key=t.get) if t else "?"
    if r.get("kind") == "noul":
        q = r["question"]
        if q.startswith("The agent proposes to call"):
            what = "tool_guard"
        elif q.startswith("The agent wants to stop"):
            what = "stop_guard"
        else:
            what = "stop_trigger" if r.get("dp") == "stop" else f"{r.get('dp')}_gate"
        return f"{what}:{gold}"
    if r.get("dp") == "tool_params" and gold == OTHER:
        return "tool_params:OTHER"
    return str(r.get("dp"))


def balance(rows: list[dict[str, Any]], *, max_no_per_yes: float = 4.0, max_other_frac: float = 0.25,
            seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cap the majority classes that teach little: stop-trigger 'no' rows to ``max_no_per_yes`` x the 'yes' rows
    (every step asks the trigger, almost always 'no'), and tool_params rows whose gold is OTHER to
    ``max_other_frac`` of the tool_params rows. Recovery rows (``meta.after_error`` > 0) are always kept.
    Guard / stop-guard rows and tool_select rows are never subsampled."""
    rng = random.Random(seed)
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[bucket(r)].append(r)

    def cap(name: str, k: int) -> None:
        pool = by.get(name, [])
        if len(pool) <= k:
            return
        keep = [r for r in pool if (r.get("meta") or {}).get("after_error")]
        rest = [r for r in pool if not (r.get("meta") or {}).get("after_error")]
        rng.shuffle(rest)
        by[name] = keep + rest[: max(0, k - len(keep))]

    before = {k: len(v) for k, v in by.items()}
    n_yes = len(by.get("stop_trigger:yes", []))
    cap("stop_trigger:no", int(math.ceil(max_no_per_yes * max(n_yes, 1))))
    n_par = sum(len(v) for k, v in by.items() if k.startswith("tool_params"))
    n_other_cap = int(round(max_other_frac / max(1e-9, 1 - max_other_frac) * len(by.get("tool_params", []))))
    if n_par:
        cap("tool_params:OTHER", n_other_cap)
    keep_ids = {id(r) for v in by.values() for r in v}
    out = [r for r in rows if id(r) in keep_ids]
    return out, {"before": before, "after": {k: len(v) for k, v in by.items()}}


def _tool_name(option: str) -> str:
    return option.split(" -- ", 1)[0]


def knockout_final_variant(r: dict[str, Any], k: int = 4, near: dict[str, list[str]] | None = None
                           ) -> dict[str, Any] | None:
    """The final round of the served knockout for a tool_select row with many options (``serve_jev``: chunks of
    52, the top 2 of each re-compete, so a 69-tool Choice ends on ~4 finalists): the gold plus up to ``k - 1`` of
    its hardest negatives present among the row's options -- the near-duplicates that shadow it, the step it is
    skipped to (SKIP_TO), ``ask_user`` / ``finish`` -- in the row's option order. None when the row is not a
    tool_select Choice with >= 17 options or has no hard negative. The gold label carries over; the teacher labels
    the variant like any other row."""
    if r.get("dp") != "tool_select" or r.get("kind", "choice") != "choice" or len(r["options"]) < 17:
        return None
    t = r.get("target") or {}
    if not t:
        return None
    gold_opt = max(t, key=t.get)
    if t[gold_opt] < 1.0 - 1e-9:
        return None
    gold = _tool_name(gold_opt)
    by_name = {_tool_name(o): o for o in r["options"]}
    near = near if near is not None else near_table()
    hard: list[str] = []
    for n in list(near.get(gold, [])) + list(SKIP_TO.get(gold, ())) + ["ask_user", FINISH]:
        if n != gold and n in by_name and n not in hard:
            hard.append(n)
    if not hard:
        return None
    keep = {gold, *hard[: k - 1]}
    opts = [o for o in r["options"] if _tool_name(o) in keep]
    return {**r, "id": f"{r['id']}#ko", "options": opts, "target": {o: float(o == gold_opt) for o in opts},
            "meta": {**(r.get("meta") or {}), "augment": "knockout_final", "n_options_source": len(r["options"])}}


def add_knockout_variants(rows: list[dict[str, Any]], frac: float = 0.5, seed: int = 0
                          ) -> tuple[list[dict[str, Any]], int]:
    """``rows`` plus a knockout-final variant (``knockout_final_variant``) of a seeded ``frac`` of the eligible
    tool_select rows."""
    rng = random.Random(seed)
    near = near_table()
    extra = []
    for r in rows:
        v = knockout_final_variant(r, near=near)
        if v is not None and rng.random() < frac:
            extra.append(v)
    return rows + extra, len(extra)


# =============================================================================== teacher labels
def label_rows(rows: list[dict[str, Any]], teacher: str, *, alpha: float = 0.3, cache: str | None = None,
               workers: int = 8, max_questions: int = 16) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Soft targets q = alpha * gold + (1 - alpha) * p_teacher for every row (teacher_label.label_file with one
    teacher; rows where the teacher's argmax is not the gold keep the gold one-hot)."""
    from .teacher_label import label_file, parse_teacher

    name, spec = parse_teacher(teacher, 0)
    return label_file(rows, [(name, spec)], alpha=alpha, min_agree=1, gold_filter="gold", on_disagree="gold",
                      cache=cache, workers=workers, max_questions=max_questions)


def teacher_dev_metrics(rows: list[dict[str, Any]], teacher: str, *, cache: str | None = None, workers: int = 8,
                        max_questions: int = 1) -> dict[str, Any]:
    """The teacher's accuracy / Brier / ECE vs gold on the dev rows, overall and per decision point and bucket
    (one question per request by default: the protocol of scripts/gate_s1_dev.py). Dev targets stay gold."""
    from ..accounting import Ledger
    from ..bench.metrics import ece
    from .teacher_label import (SIMULATED_TYPES, backend_distributions, gold_of, ledger_summary, make_teacher,
                                parse_teacher)

    name, spec = parse_teacher(teacher, 0)
    ledger = Ledger()
    backend = make_teacher(spec, cache)
    dists, reasons = backend_distributions(rows, backend, ledger, expose_gold=spec["type"] in SIMULATED_TYPES,
                                           workers=workers, max_questions=max_questions)
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r, d in zip(rows, dists):
        y = gold_of(r)
        if d is None or y is None:
            continue
        k = max(range(len(d)), key=lambda j: d[j])
        hit = float(r["options"][k] == y)
        for g in ("overall", f"dp:{r['dp']}", f"bucket:{bucket(r)}",
                  f"after_error:{int(bool((r.get('meta') or {}).get('after_error')))}"):
            groups[g].append((d[k], hit))

    def m(v: list[tuple[float, float]]) -> dict[str, Any]:
        conf, corr = [c for c, _ in v], [h for _, h in v]
        return {"n": len(v), "acc": round(sum(corr) / len(v), 4),
                "brier": round(sum((c - h) ** 2 for c, h in v) / len(v), 4), "ece": round(ece(conf, corr), 4)}

    return {"teacher": name, "spec": spec, "metrics": {g: m(v) for g, v in sorted(groups.items())},
            "skipped": dict(reasons), "ledger": ledger_summary(ledger)}


# =============================================================================== manifest
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stats(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    s = stats(rows)
    seeds = [x for x in (jevgym_seed(r["id"]) for r in rows) if x is not None]
    return {"rows": len(rows), "bytes": path.stat().st_size, "sha256": _sha256(path),
            "tasks": s["tasks"], "by_source": dict(s["by_source"]), "by_dp": dict(s["by_dp"]),
            "by_bucket": dict(Counter(bucket(r) for r in rows)),
            "by_arm": dict(Counter(str((r.get("meta") or {}).get("arm", "-")) for r in rows)),
            "after_error_rows": sum(1 for r in rows if (r.get("meta") or {}).get("after_error")),
            "label_sources": dict(Counter(str((r.get("meta") or {}).get("label_source")) for r in rows)),
            "soft_frac": round(s["soft_frac"], 4), "by_options": dict(s["by_options"]),
            "jevgym_seed_range": [min(seeds), max(seeds)] if seeds else None,
            "state_chars_median": sorted(len(r["state"]) for r in rows)[len(rows) // 2] if rows else 0}


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _tbl(head: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join("–" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def manifest_markdown(man: dict[str, Any], out_dir: Path) -> str:
    """A committable summary of ``MANIFEST.json`` (counts per decision point / bucket / source / arm, seed ranges,
    exclusions verified, cost)."""
    b = man.get("build") or {}
    files = man.get("files") or {}
    lines = [f"# Training data v1 manifest (`{out_dir}`)", "",
             f"Generated {man.get('created')} at commit `{man.get('git_commit')}` by "
             "`scripts/train/make_data_v1.sh` (`python -m jevharness.training.data_v1`). The data files are not "
             f"committed ({man.get('total_mb')} MB, regenerated by that script from the cached runs / teacher calls); "
             "this page and `manifest.json` next to it are. Plan and gap diagnosis: `docs/TRAINING_RECIPES.md` §18.",
             "", "## Files", ""]
    fr = []
    for name, f in files.items():
        if "rows" in f:
            fr.append([f"`{name}`", f["rows"], f.get("tasks"), f.get("after_error_rows"), f.get("soft_frac"),
                       f.get("state_chars_median"), f.get("jevgym_seed_range"), f["sha256"][:12]])
    lines += [_tbl(["file", "rows", "tasks", "after-error rows", "soft frac", "state chars (median)",
                    "JevGym seeds", "sha256[:12]"], fr), ""]
    for name in ("train.jsonl.gz", "dev.jsonl.gz"):
        f = files.get(name)
        if not f:
            continue
        lines += [f"### `{name}` by decision point / bucket / arm / label source", "",
                  _tbl(["decision point", "rows"], [[k, v] for k, v in sorted(f["by_dp"].items())]), "",
                  _tbl(["bucket", "rows"], [[k, v] for k, v in sorted(f["by_bucket"].items())]), "",
                  _tbl(["arm (behaviour policy)", "rows"], [[k, v] for k, v in sorted(f["by_arm"].items())]), "",
                  _tbl(["label source", "rows"], [[k, v] for k, v in sorted(f["label_sources"].items())]), ""]
    eps = b.get("episodes") or {}
    if eps:
        lines += ["## Episodes (real harness, Gemini 3.5 Flash-Lite executor)", ""]
        er = []
        for run, s in eps.items():
            for arm, x in (s.get("arms") or {}).items():
                er.append([f"`{Path(run).name}`", arm, x["episodes"], x["success"], x["off_plan_episodes"],
                           x["decisions"], x.get("wrong_proposal_rows", 0), f"{x['cost_usd']:.4f}"])
        lines += [_tbl(["run", "arm", "episodes", "success", "episodes with off-plan steps", "logged decisions",
                        "wrong-proposal guard rows", "$"], er), ""]
        for run, s in eps.items():
            beh = {json.dumps(x, sort_keys=True) for a in (s.get("arms") or {}).values()
                   for x in a.get("system1_behaviour") or []}
            exe = sorted({e for a in (s.get("arms") or {}).values() for e in a.get("executor") or []})
            lines += [f"- `{Path(run).name}`: executor {exe}; System-1 behaviour policy "
                      f"(simulation component, never a label) {sorted(beh)}."]
            if s.get("nearmiss_counts"):
                lines += [f"- near-miss policy answers (all System-1 questions, shadow ones included): "
                          f"`{json.dumps(s['nearmiss_counts'], sort_keys=True)}`."]
        ko = b.get("knockout_final_variants") or {}
        lines += [f"- knockout-final variants added to train: {ko.get('rows')} (share {ko.get('frac')} of the "
                  "eligible tool_select rows; id suffix `#ko`); wrong-proposal guard rows: the executed call lay "
                  "outside the oracle's acceptable set AT(s) (label 'no').", ""]
    lines += ["## Regenerate", "",
              "```bash",
              "bash scripts/train/make_data_v1.sh      # collect (resumable) -> build -> label (cached) -> manifest",
              "# from the kept intermediates only (no API call): SKIP_EPISODES=1 bash scripts/train/make_data_v1.sh",
              "```", "",
              "Intermediates (gitignored, not committed): trajectories `data/cache/train_v1/train_v1_jevgym/` "
              "(`<arm>.jsonl` + `experiment.yaml`), gold train rows `data/cache/train_v1/_work/`, Jev answers "
              "`data/cache/teacher_v1.sqlite` (a re-run over them costs $0). Every step is deterministic (seed 0): "
              "rebuilding from the same intermediates reproduces the sha256 values above.", ""]
    lines += ["## Hold-out rules verified", "",
              f"- JevGym seeds: every row >= 10000 and outside {list(FORBIDDEN_RANGES)} (evaluation, retired, "
              "calibration); `data_v1.assert_clean` re-checks the written files.",
              f"- Gate dev set (`{GATE_DEV}`, 210 decisions): its {b.get('gate_dev_tasks_excluded')} tasks are "
              "excluded from v1 (v1 episodes use other seeds; v0 train rows are disjoint by task).",
              f"- E1 dev prefix (`{E1_DEV_PREFIX}`): {b.get('e1_dev_prefix_ids')} ids, none in v1.",
              f"- 8-gram decontamination of every row's user text against {(b.get('decontamination') or {}).get('eval_texts')} "
              "BFCL (six tool categories, all items) + When2Call test questions; BFCL items never enter v1.",
              f"- Dropped by these rules (JevGym rows): `{json.dumps((b.get('exclusions') or {}).get('dropped'))}`; "
              f"v0 train: `{json.dumps(((b.get('extra_train') or {}).get('v0_train') or {}).get('dropped'))}`; "
              f"When2Call replay: `{json.dumps((b.get('replay') or {}).get('dropped'))}`.",
              f"- Duplicates removed (identical state / question / options / gold across arms): "
              f"{b.get('duplicates_removed')} of {b.get('rows_raw')} raw rows.", ""]
    cost = man.get("cost") or {}
    if cost:
        lines += ["## Cost (Ledger)", "", _tbl(["item", "$"], [[k, v] for k, v in cost.items()]), ""]
    tm = man.get("teacher_dev") or {}
    if tm:
        lines += ["## Teacher (Jev) on v1 dev (gold targets; one question per request)", "",
                  _tbl(["group", "n", "acc", "Brier", "ECE"],
                       [[g, x["n"], x["acc"], x["brier"], x["ece"]] for g, x in tm.items()]), ""]
    return "\n".join(lines) + "\n"


def write_manifest(out: str | Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = Path(out)
    files = {}
    for p in sorted(out.glob("*.jsonl")) + sorted(out.glob("*.jsonl.gz")):
        files[p.name] = file_stats(p)
    for p in sorted(out.glob("*.summary.json")):
        files[p.name] = {"bytes": p.stat().st_size, "sha256": _sha256(p)}
    prev = json.loads((out / "MANIFEST.json").read_text()) if (out / "MANIFEST.json").exists() else {}
    man = {**prev, **(extra or {}), "name": out.name, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "git_commit": _git_commit(), "files": files,
           "total_mb": round(sum(f.get("bytes", 0) for f in files.values()) / 1e6, 3)}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str) + "\n")
    return man


# =============================================================================== go / no-go (round 4)
GATE_VERDICT = "results/gpu_round2/gate_verdict.json"
# Intern-Decision-4B on the states the harness visits (round 2, n = 10 dev tasks, Flash System-2, shadow): the proxy
# used when its v1-dev run (12_sft_v1.sh stage `intern`) is not available
INTERN_SHADOW_PROXY = {"tool_select": 0.760, "tool_params": 0.872, "stop": 0.933}


def _file_metrics(ev: dict[str, Any], suffix: str) -> dict[str, Any] | None:
    """The metrics of the evaluated file whose path ends with ``suffix`` (or ``suffix`` + ``.gz``)."""
    for f, m in (ev.get("files") or {}).items():
        if f.endswith(suffix) or f.endswith(suffix + ".gz"):
            return m
    return None


def _gate_ref_metrics(path: str | Path) -> dict[str, Any] | None:
    """A ``scripts/gate_s1_dev.py`` output -> {"overall": {"acc"}, "by_dp": {dp: {"acc", "n"}}}."""
    p = Path(path)
    if not p.exists():
        return None
    g = json.loads(p.read_text())
    return {"overall": {"acc": g.get("acc"), "n": g.get("n")},
            "by_dp": {k: {"acc": v.get("acc"), "n": v.get("n")} for k, v in (g.get("by_dp") or {}).items()}}


def gonogo(ev: dict[str, Any], *, jev_v1: dict[str, Any] | None, intern_v1: dict[str, Any] | None,
           gate_verdict: dict[str, Any] | None, baseline: dict[str, Any] | None = None,
           ratio: float = 0.8, ts_margin: float = 0.05, overall_margin: float = 0.02, ece_max: float = 0.10
           ) -> dict[str, Any]:
    """The round-4 decision for a trained decider (docs/TRAINING_RECIPES.md §18.4). ``ev`` is an
    ``eval_decisions`` output over the gate set (data/train_v0/dev.jsonl) and v1 dev (data/train_v1/dev.jsonl,
    + dev_recovery.jsonl; ``.gz`` accepted); ``jev_v1`` the teacher summary of v1 dev (``dev_teacher.summary.json``), ``intern_v1``
    Intern-Decision-4B on v1 dev (``_gate_ref_metrics``); ``gate_verdict`` the committed A5.1 verdict (Jev 0.971,
    Intern 0.862 on the gate set); ``baseline`` an earlier checkpoint's ``eval_decisions`` output (R2 on v0).

      G1 minimum      gate-set acc >= ratio x Jev's gate acc (A5.1's competence bar, 0.8 x 0.971 = 0.777)
      G2 static       gate-set acc >= Intern-4B's gate acc - 0.01 (does not lose the static ground Intern holds)
      G3 in-harness   v1-dev tool_select acc >= Intern-4B's + ts_margin and v1-dev overall acc >= Intern-4B's +
                      overall_margin (Intern on v1 dev; the round-2 shadow accuracies are the proxy if missing)
      G4 vs teacher   v1-dev acc >= ratio x Jev's v1-dev acc on each of tool_select, tool_params, stop
      G5 calibration  v1-dev ECE <= max(ece_max, Intern-4B's v1-dev ECE when known) (raw, before temperature)
    GO = G1 and G2 and G3 and G4 and G5: the checkpoint is the better open System-1 for the exploratory in-harness
    rows (it never replaces the registered second family, A5.1), confirmed next on the E2 dev slice (dev seeds,
    12_sft_v1.sh stage `e2dev`)."""
    gate = _file_metrics(ev, "train_v0/dev.jsonl") or {}
    dev = _file_metrics(ev, "train_v1/dev.jsonl") or {}
    rec = _file_metrics(ev, "train_v1/dev_recovery.jsonl") or {}
    gv = gate_verdict or {}
    jev_gate = gv.get("jev_acc", 0.9714)
    intern_gate = ((gv.get("candidates") or {}).get("intern_4B") or {}).get("acc", 0.8619)

    def acc(m: dict[str, Any], dp: str | None = None) -> float | None:
        x = (m.get("by_dp") or {}).get(dp) if dp else m.get("overall")
        return None if not x else x.get("acc")

    jm = (jev_v1 or {}).get("metrics") or {}
    jev = {"overall": (jm.get("overall") or {}).get("acc"),
           **{dp: (jm.get(f"dp:{dp}") or {}).get("acc") for dp in ("tool_select", "tool_params", "stop")}}
    if intern_v1:
        intern = {"overall": acc(intern_v1), **{dp: acc(intern_v1, dp) for dp in ("tool_select", "tool_params",
                                                                                    "stop")}, "source": "v1 dev"}
    else:
        intern = {"overall": None, **INTERN_SHADOW_PROXY, "source": "proxy: round-2 shadow s1_acc (n = 10 tasks)"}
    g_acc, d_acc = acc(gate), acc(dev)
    checks: dict[str, Any] = {}
    checks["G1_min_gate"] = {"value": g_acc, "threshold": round(ratio * jev_gate, 4),
                             "pass": g_acc is not None and g_acc >= ratio * jev_gate}
    checks["G2_static_vs_intern"] = {"value": g_acc, "threshold": round(intern_gate - 0.01, 4),
                                     "pass": g_acc is not None and g_acc >= intern_gate - 0.01}
    ts, it_ts = acc(dev, "tool_select"), intern.get("tool_select")
    ok3 = ts is not None and it_ts is not None and ts >= it_ts + ts_margin
    if intern.get("overall") is not None:
        ok3 = ok3 and d_acc is not None and d_acc >= intern["overall"] + overall_margin
    checks["G3_in_harness_vs_intern"] = {"tool_select": ts, "intern_tool_select": it_ts, "overall": d_acc,
                                         "intern_overall": intern.get("overall"), "pass": bool(ok3)}
    per = {}
    for dp in ("tool_select", "tool_params", "stop"):
        v, j = acc(dev, dp), jev.get(dp)
        per[dp] = {"value": v, "jev": j, "threshold": None if j is None else round(ratio * j, 4),
                   "pass": v is not None and j is not None and v >= ratio * j}
    checks["G4_vs_teacher"] = {**per, "pass": all(x["pass"] for x in per.values())}
    ece = (dev.get("overall") or {}).get("ece")
    it_ece = ((intern_v1 or {}).get("overall") or {}).get("ece")
    lim = max(ece_max, it_ece) if it_ece is not None else ece_max
    checks["G5_calibration"] = {"value": ece, "threshold": lim, "pass": ece is not None and ece <= lim}
    go = all(c["pass"] for c in checks.values())
    out = {"go": go, "checks": checks, "references": {"jev_gate": jev_gate, "intern_gate": intern_gate,
                                                      "jev_v1_dev": jev, "intern_v1_dev": intern},
           "recovery_states": {"acc": acc(rec), "n": (rec.get("overall") or {}).get("n")},
           "decision": ("GO: better open System-1 than Intern-Decision-4B on the harness's states; run the E2 dev "
                        "slice (stage e2dev) before reporting it" if go else
                        "NO-GO: keep Intern-Decision-4B as the open System-1; report the trained row as is")}
    if baseline:
        bg, bd = _file_metrics(baseline, "train_v0/dev.jsonl") or {}, _file_metrics(baseline, "train_v1/dev.jsonl") or {}
        out["vs_baseline"] = {"gate_acc_delta": None if acc(bg) is None or g_acc is None else round(g_acc - acc(bg), 4),
                              "v1_dev_acc_delta": None if acc(bd) is None or d_acc is None else round(d_acc - acc(bd), 4)}
    return out


def merge_evals(paths: Sequence[str | Path]) -> dict[str, Any] | None:
    """Several ``eval_decisions`` outputs of one checkpoint -> one (their ``files`` merged); missing paths skipped."""
    files: dict[str, Any] = {}
    head: dict[str, Any] = {}
    for p in paths:
        if Path(p).exists():
            ev = json.loads(Path(p).read_text())
            head = head or {k: v for k, v in ev.items() if k != "files"}
            files.update(ev.get("files") or {})
    return {**head, "files": files} if files else None


def _cmd_gonogo(a: argparse.Namespace) -> None:
    ev = merge_evals(a.eval)
    if ev is None:
        raise SystemExit(f"no eval_decisions output among {a.eval}")
    jev = json.loads(Path(a.jev_v1).read_text()) if a.jev_v1 and Path(a.jev_v1).exists() else None
    intern = _gate_ref_metrics(a.intern_v1) if a.intern_v1 else None
    gv = json.loads(Path(a.gate_verdict).read_text()) if Path(a.gate_verdict).exists() else None
    base = merge_evals(a.baseline) if a.baseline else None
    res = gonogo(ev, jev_v1=jev, intern_v1=intern, gate_verdict=gv, baseline=base)
    res["inputs"] = {"eval": a.eval, "jev_v1": a.jev_v1, "intern_v1": a.intern_v1, "gate_verdict": a.gate_verdict,
                     "baseline": a.baseline}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


# =============================================================================== CLI
def _cmd_collect(a: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    n_by_arm = json.loads(a.n_by_arm) if a.n_by_arm else None
    exp = build_experiment(a.name, a.seed0, a.n, n_by_arm=n_by_arm, workers=a.workers,
                           llm=json.loads(a.llm) if a.llm else None)
    if a.dry_run:
        print(json.dumps(exp, indent=1))
        return
    s1 = NearMissBackend(eps_tool=a.eps_tool, eps_param=a.eps_param, eps_gate=a.eps_gate, seed=a.seed)
    paths = collect(exp, a.out, s1=s1, max_tasks=a.max_tasks)
    work = Path(a.out) / exp["name"] / "_harvest"
    work.mkdir(parents=True, exist_ok=True)
    _, summ = harvest_runs(Path(a.out) / exp["name"], work_dir=work)
    print(json.dumps({"runs": {k: str(v) for k, v in paths.items()}, "nearmiss_counts": dict(s1.counts),
                      **summ}, indent=1))


def _cmd_build(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    episodes: dict[str, Any] = {}
    for run in a.runs:
        got, summ = harvest_runs(run, max_options=52, seed=a.seed, work_dir=work)
        rows += got
        episodes[str(run)] = summ
    n_raw = len(rows)
    rows, n_dup = dedupe(rows)
    gate = gate_dev_tasks(a.gate_dev)
    e1 = e1_dev_prefix_ids(a.e1_ids)
    decon = Decontaminator(eval_texts(a.raw_dir), n=a.ngram)
    rows, excl = verify_exclusions(rows, gate_tasks=gate, e1_ids=e1, decon=decon, check_text=True)
    bal, bal_info = balance(rows, max_no_per_yes=a.max_no_per_yes, max_other_frac=a.max_other_frac, seed=a.seed)
    train, dev = split_by_task(bal, a.dev_frac, a.seed)
    # dev keeps the natural distribution of its tasks (no balancing): rebuild it from the unbalanced rows
    dev_tasks = {task_of(r["id"]) for r in dev}
    dev = [r for r in rows if task_of(r["id"]) in dev_tasks]
    extra_train: dict[str, Any] = {}
    if a.v0_train:
        v0 = [r for r in read_jsonl(a.v0_train)]
        v0, v0_excl = verify_exclusions(v0, gate_tasks=gate, e1_ids=e1, decon=decon, check_text=True)
        for r in v0:
            r["meta"] = {**(r.get("meta") or {}), "arm": "v0_flash_shadow"}
        train += v0
        extra_train["v0_train"] = {"path": a.v0_train, "rows": len(v0), **v0_excl}
    train, n_ko = add_knockout_variants(train, a.ko_frac, a.seed) if a.ko_frac > 0 else (train, 0)
    replay_info: dict[str, Any] = {}
    if a.replay:
        rep = read_jsonl(a.replay)
        rep, rep_excl = verify_exclusions(rep, gate_tasks=gate, e1_ids=e1, decon=decon)
        write_jsonl(rep, out / "replay_when2call.jsonl.gz")
        replay_info = {"path": a.replay, **rep_excl}
    for part in (train, dev):
        assert_clean(part, gate_tasks=gate, e1_ids=e1)
    assert not ({task_of(r["id"]) for r in train} & dev_tasks), "train / dev share a task"
    rng = random.Random(a.seed)
    rng.shuffle(train)
    write_jsonl(train, work / "train_gold.jsonl")
    write_jsonl(dev, out / "dev.jsonl.gz")
    write_jsonl([r for r in dev if (r.get("meta") or {}).get("after_error")], out / "dev_recovery.jsonl.gz")
    write_jsonl(add_knockout_variants(dev, 1.0, a.seed)[0][len(dev):], out / "dev_knockout.jsonl.gz")
    info = {"build": {"runs": [str(r) for r in a.runs], "episodes": episodes, "rows_raw": n_raw,
                      "duplicates_removed": n_dup, "exclusions": excl, "balance": bal_info,
                      "dev_frac": a.dev_frac, "seed": a.seed, "train_rows_gold": len(train), "dev_rows": len(dev),
                      "dev_tasks": len(dev_tasks), "extra_train": extra_train, "replay": replay_info,
                      "knockout_final_variants": {"frac": a.ko_frac, "rows": n_ko},
                      "decontamination": {"ngram": a.ngram, "eval_texts": decon.n_texts,
                                          "sources": "BFCL v4 six tool categories (all items) + When2Call test"},
                      "gate_dev_tasks_excluded": len(gate), "e1_dev_prefix_ids": len(e1)}}
    (out / "build_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: v for k, v in info["build"].items() if k != "episodes"}, indent=1, default=str))


def _cmd_label(a: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    out = Path(a.out)
    train = read_jsonl(Path(a.work) / "train_gold.jsonl")
    if a.limit:
        train = train[: a.limit]
    labelled, summ = label_rows(train, a.teacher, alpha=a.alpha, cache=a.cache or None, workers=a.workers)
    write_jsonl(labelled, out / "train.jsonl.gz")
    (out / "train.jsonl.summary.json").write_text(json.dumps(summ, indent=1, default=str))
    dev = read_jsonl(out / "dev.jsonl.gz")
    dm = teacher_dev_metrics(dev, a.teacher, cache=a.cache or None, workers=a.workers)
    (out / "dev_teacher.summary.json").write_text(json.dumps(dm, indent=1, default=str))
    total = summ["cost_usd_fresh"] + dm["ledger"]["cost_usd_fresh"]
    print(json.dumps({"train_outcomes": summ["outcomes"], "train_teacher_acc": {
        n: t.get("acc") for n, t in summ["teachers"].items()}, "dev_teacher": dm["metrics"].get("overall"),
        "teacher_cost_usd_fresh": round(total, 4)}, indent=1))


def cost_and_teacher(out: Path, info: dict[str, Any]) -> dict[str, Any]:
    """Ledger totals (episodes per run + teacher on train and dev; cached teacher replays carry their original cost)
    and the teacher's dev metrics, for the manifest."""
    cost: dict[str, Any] = {}
    for run, s in ((info.get("build") or {}).get("episodes") or {}).items():
        cost[f"episodes {Path(run).name} (System-2 Flash-Lite; System-1 behaviour policy $0)"] = round(s["cost_usd"], 4)
    extra: dict[str, Any] = {}
    f = out / "train.jsonl.summary.json"
    if f.exists():
        s = json.loads(f.read_text())
        for n, t in s["teachers"].items():
            cost[f"teacher {n} on train ({t['ledger']['calls']} requests)"] = round(t["ledger"]["cost_usd"], 4)
        extra["teacher_train"] = {"outcomes": s.get("outcomes"), "config": s.get("config"),
                                  "teachers": {n: {k: t.get(k) for k in ("acc", "nll", "ece", "answered", "skipped")}
                                               for n, t in s["teachers"].items()}}
    f = out / "dev_teacher.summary.json"
    if f.exists():
        s = json.loads(f.read_text())
        cost[f"teacher {s['teacher']} on dev ({s['ledger']['calls']} requests)"] = round(s["ledger"]["cost_usd"], 4)
        extra["teacher_dev"] = s["metrics"]
    cost["total"] = round(sum(v for v in cost.values()), 4)
    return {"cost": cost, **extra}


def _cmd_manifest(a: argparse.Namespace) -> None:
    extra = {}
    if a.info:
        extra = json.loads(Path(a.info).read_text())
    extra.update(cost_and_teacher(Path(a.out), extra))
    man = write_manifest(a.out, extra)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(manifest_markdown(man, Path(a.out)))
    print(json.dumps({k: man[k] for k in ("name", "total_mb")} | {"files": list(man["files"])}, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="training data v1 (on-policy-style JevGym states + Jev soft labels)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="run the behaviour arms on training seeds (Flash-Lite executor: $)")
    c.add_argument("--out", default="data/cache/train_v1")
    c.add_argument("--name", default="train_v1_jevgym")
    c.add_argument("--seed0", type=int, default=V1_SEED0)
    c.add_argument("--n", type=int, default=600)
    c.add_argument("--n-by-arm", default=None, help='JSON {arm: n_tasks}, e.g. {"lite_guard": 120}')
    c.add_argument("--max-tasks", type=int, default=None)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--llm", default=None, help="executor spec JSON (default: Gemini 3.5 Flash-Lite, T=0)")
    c.add_argument("--eps-tool", type=float, default=0.2)
    c.add_argument("--eps-param", type=float, default=0.15)
    c.add_argument("--eps-gate", type=float, default=0.1)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--dry-run", action="store_true")
    b = sub.add_parser("build", help="harvest + dedupe + exclusions + balance + split (no API call)")
    b.add_argument("--runs", nargs="+", required=True, help="run directories (<arm>.jsonl + experiment.yaml)")
    b.add_argument("--out", default="data/train_v1")
    b.add_argument("--work", default="data/cache/train_v1/_work", help="large intermediates (gold train rows before the "
                   "teacher pass, per-arm latest records): keep them out of the data dir")
    b.add_argument("--v0-train", default="data/train_v0/train.jsonl", help="v0 train rows added to train ('' = no)")
    b.add_argument("--replay", default="data/train_v0/replay_when2call.jsonl.gz", help="When2Call-train replay")
    b.add_argument("--gate-dev", default=GATE_DEV)
    b.add_argument("--e1-ids", default=E1_DEV_PREFIX)
    b.add_argument("--raw-dir", default="data/raw")
    b.add_argument("--ngram", type=int, default=8)
    b.add_argument("--dev-frac", type=float, default=0.15)
    b.add_argument("--max-no-per-yes", type=float, default=4.0)
    b.add_argument("--max-other-frac", type=float, default=0.25)
    b.add_argument("--ko-frac", type=float, default=0.5,
                   help="share of tool_select train rows (>= 17 options) that also get a knockout-final variant")
    b.add_argument("--seed", type=int, default=0)
    lab = sub.add_parser("label", help="Jev soft labels on train, Jev reference metrics on dev ($)")
    lab.add_argument("--out", default="data/train_v1")
    lab.add_argument("--work", default="data/cache/train_v1/_work")
    lab.add_argument("--teacher", default="jev=jev-openrouter")
    lab.add_argument("--alpha", type=float, default=0.3)
    lab.add_argument("--cache", default="data/cache/teacher_v1.sqlite")
    lab.add_argument("--workers", type=int, default=8)
    lab.add_argument("--limit", type=int, default=None)
    m = sub.add_parser("manifest", help="MANIFEST.json (+ a committable markdown copy)")
    m.add_argument("--out", default="data/train_v1")
    m.add_argument("--info", default=None, help="JSON merged into the manifest (e.g. build_info.json)")
    m.add_argument("--md", default=None)
    g = sub.add_parser("gonogo", help="round-4 go / no-go of a trained decider (TRAINING_RECIPES §18.4)")
    g.add_argument("--eval", nargs="+", required=True,
                   help="eval_decisions JSON(s) of the checkpoint over the gate set + v1 dev (+ recovery)")
    g.add_argument("--jev-v1", default="data/train_v1/dev_teacher.summary.json")
    g.add_argument("--intern-v1", default=None, help="scripts/gate_s1_dev.py JSON of Intern-Decision-4B on v1 dev")
    g.add_argument("--gate-verdict", default=GATE_VERDICT)
    g.add_argument("--baseline", nargs="+", default=None,
                   help="eval_decisions JSON(s) of an earlier checkpoint (e.g. R2 on v0)")
    g.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    {"collect": _cmd_collect, "build": _cmd_build, "label": _cmd_label, "manifest": _cmd_manifest,
     "gonogo": _cmd_gonogo}[a.cmd](a)


if __name__ == "__main__":
    main()
