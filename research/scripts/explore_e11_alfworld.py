#!/usr/bin/env python3
r"""E11 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 "E11: the training recipe on
ALFWorld"): the paper's training recipe (DecisionAgentRL's SFT step: a small decision model trained on the states the
agent itself visits) applied to ALFWorld's action decision point. Harvest on ALFWorld TRAIN games only (Flash LLM-only
episodes plus a scripted executor that follows the gold action and injects errors), Qwen3-4B + LoRA SFT on the gold
next action (one-hot), temperatures on held-out development games, the CASCADE threshold fitted with the registered
calibration-arm estimator on E6a's 48 calibration train games, then live on E6a's 140 valid_seen games beside a
re-run of LLM-only. No test; outside every Holm table; revises nothing.

    python scripts/explore_e11_alfworld.py select --json-zip J.zip --twpddl-zip T.zip   # 600 harvest games (box)
    python scripts/explore_e11_alfworld.py harvest-scripted --out runs                  # $0, scripted executor (box)
    python -m jevharness.bench.runner configs/explore/e11_alfworld/harvest_flash.yaml --out runs   # Flash (box)
    python scripts/explore_e11_alfworld.py build --runs runs --out DIR                  # SFT rows -> DIR/{train,dev}*.jsonl
    python scripts/explore_e11_alfworld.py fit                                          # tau-hat (calibration shadow arm)
    python scripts/explore_e11_alfworld.py set-tau                                      # tau-hat -> eval.yaml / smoke.yaml
    python scripts/explore_e11_alfworld.py spend | watch --pid PID --cap 60 | project
    python scripts/explore_e11_alfworld.py summarize [--box results/explore/e11_alfworld/box]

Readout: E6a's (``explore_e6a_alfworld.arm_block``: paired complete case over the eval list against ``llm_only``;
success, Delta with the paired task bootstrap B = 10,000 and the Newcombe interval, wins / losses, cost per episode as
logged and the saving 1 - C_arm / C_ref, wall time, steps, calls, coverage and accuracy against the single gold action),
plus the GPU tariff q = host $/h x evaluation window / completed GPU-arm episodes (E7's convention after A7.13 (iii)),
with q = 0 and S2-4's frozen q as sensitivity rows."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import re
import shutil
import sys
import threading
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_common as C  # noqa: E402
import explore_e6a_alfworld as E6A  # noqa: E402

from jevharness.bench.suites.alfworld import TASK_TYPES, collect_game_files  # noqa: E402
from jevharness.llm.base import ChatResponse  # noqa: E402
from jevharness.llm.scripted import ScriptedLLM  # noqa: E402

ROOT = C.ROOT
RUNS = ROOT / "runs"
CFG_DIR = ROOT / "configs/explore/e11_alfworld"
OUT_DIR = ROOT / "results/explore/e11_alfworld"
BOX_DIR = OUT_DIR / "box"
TEX = ROOT / "paper/generated/e11_alfworld.tex"
HARVEST_LIST = CFG_DIR / "harvest_games.json"
TAU_FILE = CFG_DIR / "tauhat_action.json"
CAL_LIST = E6A.TRAIN_LIST                 # E6a's 48 calibration train games (never harvested)
EVAL_LIST = E6A.EVAL_LIST                 # E6a's 140 valid_seen games
DATA_SCRIPTED = ROOT / "data/raw/alfworld_e11"          # the 600 harvest games (gitignored)
DATA_FLASH = ROOT / "data/raw/alfworld_e11_flash"       # the 360 Flash harvest games (copies; gitignored)
RUN_FLASH, RUN_SCRIPTED = "e11_harvest_flash", "e11_harvest_scripted"
RUN_CAL, RUN_SMOKE, RUN_EVAL = "e11_alfworld_cal", "e11_alfworld_smoke", "e11_alfworld_eval"
FLASH_ARM, SCRIPTED_ARM = "llm_only", "expert_noise"
CAL_ARM = "e11_action_shadow_cal"
REF = "llm_only"
ARMS = {"cascade": "e11_action_cascade_tauhat", "replace": "e11_action_replace"}
GPU_ARMS = tuple(ARMS.values())
LABEL = "exploratory (E11)"
SERVED = "e11-alfworld-4b"
SALT = "jev-explore-E11-alfworld-harvest-2026-10-07"
N_PER_TYPE, N_FLASH_PER_TYPE, DEV_EVERY = 100, 60, 10
EPS = 0.2
ERROR_KINDS = {"near": 0.5, "random": 0.3, "noop": 0.2}
SCRIPTED_SEED = 0
MAX_OPTIONS = 52
MAX_TRAIN_ROWS = 10_000
BUILD_SEED = 0
QUESTION = "Which action should the agent take next?"
FALLBACK = "look"          # ALFWorld's AlfredExpert wrapper: the plan when the hand-coded expert has no admissible action
CAP_USD, SAFETY, N_EVAL, MIN_EVAL_GAMES = 60.0, 1.5, 140, 100
# on-demand $/h, us-east-1 (the prices E7 records): the host that served the evaluation sets the tariff
HOST_USD_PER_HOUR = {"g6e.2xlarge": 2.24208, "g5.8xlarge": 2.448, "g5.4xlarge": 1.624}
S24_TARIFF = ROOT / "results/stage2/tariff.json"
PENDING_RE = re.compile(r"action: \{mode: cascade, tau: [0-9.]+, calibration: pending\}")
NAN = float("nan")
_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ============================================================================== harvest-game selection
def task_type_of(relp: str) -> str:
    task_dir = relp.split("/")[0]
    return next((t for t in TASK_TYPES.values() if task_dir.startswith(t + "-")), "unknown")


def order_key(relp: str, salt: str = SALT) -> str:
    return _sha(f"{salt}\n{relp}")


def split_games(by_type: Mapping[str, Sequence[str]], n_flash: int = N_FLASH_PER_TYPE,
                dev_every: int = DEV_EVERY) -> dict[str, list[str]]:
    """Per task type (games in the salted order): the first ``n_flash`` also get a Flash episode; every
    ``dev_every``-th game (1-indexed positions dev_every, 2 dev_every, ...) is a development game."""
    out: dict[str, list[str]] = {"all": [], "flash": [], "dev": []}
    for t in TASK_TYPES.values():
        games = list(by_type.get(t) or [])
        out["all"] += games
        out["flash"] += games[:n_flash]
        out["dev"] += [g for i, g in enumerate(games) if (i + 1) % dev_every == 0]
    return out


def select_harvest(json_zip: Path, twpddl_zip: Path, exclude: Iterable[str], n_per_type: int = N_PER_TYPE,
                   salt: str = SALT) -> dict[str, list[str]]:
    """E6a's train-split filters (no ``movable`` / ``Sliced``, the six task types, ``solvable``), the 48 calibration
    games excluded, ordered by sha256(salt + newline + "<task dir>/<trial>"), the first ``n_per_type`` per type."""
    zj, zt = zipfile.ZipFile(json_zip), zipfile.ZipFile(twpddl_zip)
    pre = "json_2.1.1/train/"
    tw = {n[: -len("/game.tw-pddl")]: n for n in zt.namelist() if n.startswith(pre) and n.endswith("/game.tw-pddl")}
    js = {n[: -len("/traj_data.json")]: n for n in zj.namelist() if n.startswith(pre) and n.endswith("/traj_data.json")}
    excl = set(exclude)
    cands = []
    for d in sorted(set(tw) & set(js)):
        relp = d[len(pre):]
        parts = relp.split("/")
        if len(parts) != 2 or not all(_SAFE.match(p) for p in parts) or "movable" in d or "Sliced" in d:
            continue
        if relp in excl:
            continue
        cands.append(relp)
    cands.sort(key=lambda r: order_key(r, salt))
    out: dict[str, list[str]] = {t: [] for t in TASK_TYPES.values()}
    for relp in cands:
        if all(len(v) >= n_per_type for v in out.values()):
            break
        typ = json.loads(zj.read(js[pre + relp])).get("task_type")
        if typ not in out or len(out[typ]) >= n_per_type:
            continue
        if not json.loads(zt.read(tw[pre + relp])).get("solvable", False):
            continue
        out[typ].append(relp)
    return out


def cal_games() -> list[str]:
    rec = json.loads(CAL_LIST.read_text())
    return [g for t in TASK_TYPES.values() for g in rec["games"][t]]


def cmd_select(a: argparse.Namespace) -> int:
    rec6 = json.loads(CAL_LIST.read_text())
    shas = {k: C.sha256_file(Path(p)) for k, p in (("json", a.json_zip), ("twpddl", a.twpddl_zip))}
    bad = [k for k, s in shas.items() if s != rec6["archives"][k]["sha256"]]
    if bad:
        raise SystemExit(f"release archive sha256 differs from E6a's record for {bad}")
    by_type = select_harvest(Path(a.json_zip), Path(a.twpddl_zip), cal_games(), a.n_per_type)
    short = {t: len(v) for t, v in by_type.items() if len(v) < a.n_per_type}
    if short:
        raise SystemExit(f"too few eligible train games for {short}")
    sp = split_games(by_type)
    for root, games in ((Path(a.data_dir), sp["all"]), (Path(a.flash_dir), sp["flash"])):
        if root.exists():
            shutil.rmtree(root)
        E6A.extract_train(Path(a.json_zip), Path(a.twpddl_zip), games, root)
        listed = collect_game_files("train", data_dir=root)
        got = sorted(str(Path(p).parent.relative_to(root / "json_2.1.1" / "train")) for p in listed)
        if got != sorted(games):
            raise SystemExit(f"{root} lists {len(got)} train games, expected the {len(games)} selected ones")
    rec = {"row": LABEL, "rule": "ALFWorld train-split filters (no movable / Sliced, six task types, solvable) minus "
           "E6a's 48 calibration games; order by sha256(salt + newline + '<task dir>/<trial>'); first "
           f"{a.n_per_type} per task type; the first {N_FLASH_PER_TYPE} per type also get a Flash episode; every "
           f"{DEV_EVERY}th per type (1-indexed) is a development game",
           "salt": SALT, "n_per_type": a.n_per_type, "n": len(sp["all"]), "n_flash": len(sp["flash"]),
           "n_dev": len(sp["dev"]), "archives": {k: {"url": rec6["archives"][k]["url"], "sha256": shas[k]} for k in shas},
           "excluded": {"list": C.rel(CAL_LIST), "n": len(cal_games())},
           "data_dir": C.rel(Path(a.data_dir)), "flash_dir": C.rel(Path(a.flash_dir)), "games": by_type,
           "flash": sp["flash"], "dev": sp["dev"], "list_sha256": _sha("\n".join(sp["all"]))}
    HARVEST_LIST.parent.mkdir(parents=True, exist_ok=True)
    HARVEST_LIST.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {C.rel(HARVEST_LIST)}: {rec['n']} harvest games ({rec['n_flash']} with Flash, {rec['n_dev']} "
          f"development; list sha256 {rec['list_sha256'][:16]})")
    return 0


# ============================================================================== scripted executor ($0)
VERBS = ("go to", "take", "move", "put", "open", "close", "toggle", "clean", "heat", "cool", "use", "slice",
         "examine", "look", "inventory")


def verb(action: str) -> str:
    for v in VERBS:
        if action == v or action.startswith(v + " "):
            return v
    return action.split(" ")[0] if action else ""


def scripted_error(gold: str, actions: Sequence[str], prev: str | None, rng: random.Random,
                   kinds: Mapping[str, float] = ERROR_KINDS) -> tuple[str, str]:
    """A scripted mistake at a state whose gold action is ``gold``: kind drawn by ``kinds`` (near: another admissible
    action with the gold's verb; random: any other admissible action; noop: repeat the previous action, or look /
    examine), falling back to ``random`` when the drawn kind has no candidate. Returns (action, kind)."""
    others = [x for x in actions if x != gold]
    if not others:
        return gold, "none"
    names = list(kinds)
    kind = rng.choices(names, weights=[float(kinds[k]) for k in names], k=1)[0]
    cand = {"near": [x for x in others if verb(x) == verb(gold)],
            "random": others,
            "noop": [x for x in others if x == prev or verb(x) in ("look", "examine")]}.get(kind) or []
    if not cand:
        kind, cand = "random", others
    return rng.choice(cand), kind


class ExpertNoiseLLM(ScriptedLLM):
    """The scripted executor of E11's harvest (registration (c) 2): in an action-only env it answers ``Action: <a>``
    with the gold next action (``env.oracle_action()``; if that is not admissible, the hand-coded expert's first planned
    action when admissible), and with probability ``eps`` a scripted mistake (``scripted_error``). States without a gold
    action get a random admissible action. Per-episode RNG seeded by (seed, task id); thread-safe counters."""

    provider = "scripted"

    def __init__(self, eps: float = EPS, seed: int = SCRIPTED_SEED, kinds: Mapping[str, float] | None = None,
                 model: str = "e11-expert-noise"):
        super().__init__(model=model, policy=None)
        self.eps, self.seed, self.kinds = float(eps), int(seed), dict(kinds or ERROR_KINDS)
        self.policy = self._policy
        self._local = threading.local()
        self._lock = threading.Lock()
        self.counts: Counter[str] = Counter()

    def bind_env(self, env: Any) -> None:
        self._local.env = env
        self._local.rng = random.Random(f"{self.seed}-{getattr(env, 'task_id', '')}")
        self._local.prev = None

    @staticmethod
    def gold_of(env: Any, actions: Sequence[str]) -> str | None:
        """The gold next action, or None. ``look`` is the expert wrapper's fallback (no admissible expert action;
        no walkthrough and no hand-coded expert action is ever ``look``), so it counts as no gold (addendum)."""
        o = env.oracle_action()
        g = (o[1] or {}).get("action") if o else None
        if g in actions and g != FALLBACK:
            return g
        plan = list(getattr(env, "_expert_plan", None) or [])
        return plan[0] if plan and plan[0] in actions and plan[0] != FALLBACK else None

    def _policy(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ChatResponse:
        env = getattr(self._local, "env", None)
        rng: random.Random = getattr(self._local, "rng", random.Random(0))
        actions = list(env.admissible_actions() or []) if env is not None else []
        if not actions:
            return ChatResponse(text="Action: look")
        g = self.gold_of(env, actions)
        if g is None:
            a, kind = rng.choice(actions), "no_gold"
        elif rng.random() < self.eps:
            a, kind = scripted_error(g, actions, self._local.prev, rng, self.kinds)
        else:
            a, kind = g, "gold"
        self._local.prev = a
        with self._lock:
            self.counts[kind] += 1
        return ChatResponse(text=f"Action: {a}")


def patch_handcoded_move() -> bool:
    """Harvest labels only (E11 addendum): ALFWorld 0.4.2's hand-coded expert tracks its inventory from the executed
    commands but recognises only ``put``, while 0.4.2 places objects with ``move X to Y``; after any ``move`` its
    inventory goes stale and its next actions are wrong or not admissible (the wrapper then falls back to ``look``).
    This makes its tracker read ``move X to Y`` as ``put X to Y``. Idempotent; affects this process only (the
    calibration and evaluation runs use the unpatched oracle, as E6a). Returns True when the patch is active."""
    from alfworld.agents.expert import handcoded_expert as H

    if getattr(H.BasePolicy.update_state_tracking, "_e11_move_patch", False):
        return True
    orig = H.BasePolicy.update_state_tracking

    def update_state_tracking(self: Any, game_state: Any, last_action: str) -> Any:
        if isinstance(last_action, str) and last_action.startswith("move ") and " to " in last_action:
            last_action = "put " + last_action[len("move "):]
        return orig(self, game_state, last_action)

    update_state_tracking._e11_move_patch = True  # type: ignore[attr-defined]
    H.BasePolicy.update_state_tracking = update_state_tracking
    return True


def cmd_harvest_flash(a: argparse.Namespace) -> int:
    """The Flash harvest (harvest_flash.yaml) in this process, with the harvest-label patch of the expert."""
    import yaml

    from jevharness.bench.runner import run_experiment

    patch_handcoded_move()
    exp = yaml.safe_load(Path(a.config).read_text())
    paths = run_experiment(exp, a.out, max_tasks=a.max_tasks, progress=True)
    print(json.dumps({k: str(v) for k, v in paths.items()}))
    return 0


def scripted_experiment(data_dir: Path = DATA_SCRIPTED, workers: int = 8) -> dict[str, Any]:
    return {"name": RUN_SCRIPTED,
            "suite": {"name": "alfworld", "split": "train", "data_dir": C.rel(data_dir), "stratify": True,
                      "max_episode_steps": 50},
            # documentary: the runner gets the ExpertNoiseLLM instance (registration (c) 2)
            "llm": {"provider": "scripted", "model": "e11-expert-noise", "eps": EPS, "kinds": dict(ERROR_KINDS),
                    "seed": SCRIPTED_SEED},
            "harness_defaults": {"max_steps": 50, "history_window": 12},
            "workers": int(workers), "arms": [{"name": SCRIPTED_ARM}]}


def cmd_harvest_scripted(a: argparse.Namespace) -> int:
    from jevharness.bench.runner import run_experiment

    patch_handcoded_move()
    exp = scripted_experiment(Path(a.data_dir), a.workers)
    llm = ExpertNoiseLLM()
    paths = run_experiment(exp, a.out, llm=llm, max_tasks=a.max_tasks, progress=True)
    out = Path(a.out) / RUN_SCRIPTED
    prev = json.loads((out / "error_counts.json").read_text()) if (out / "error_counts.json").exists() else {}
    merged = Counter(prev) + llm.counts
    (out / "error_counts.json").write_text(json.dumps(dict(sorted(merged.items())), indent=1) + "\n")
    print(f"scripted harvest: {paths}; executor actions by kind {dict(sorted(merged.items()))}")
    return 0


# ============================================================================== SFT rows
def _latest_ok(path: Path) -> list[dict[str, Any]]:
    return [t for t in E6A._latest(path).values() if not t.get("error")]


def game_of(task_id: str) -> str:
    """'alfworld/train/<task dir>/<trial>' -> '<task dir>/<trial>'."""
    return "/".join(task_id.split("/")[2:4])


def off_plan_steps(traj: Mapping[str, Any]) -> dict[int, bool]:
    """step -> True when an earlier executed action of the episode differed from that earlier step's gold action
    (the state follows an off-plan step); steps without a logged gold do not count as off-plan."""
    out: dict[int, bool] = {}
    off = False
    for st in traj.get("steps") or []:
        out[int(st["step"])] = off
        dec = [d for d in st.get("decisions") or [] if d.get("dp") == "action"]
        g = dec[0].get("gold") if dec else None
        act = (st.get("args") or {}).get("action")
        if g is not None and act is not None and act != g:
            off = True
    return out


def lost_from(traj: Mapping[str, Any]) -> int | None:
    """The first step whose logged gold is the expert wrapper's fallback ``look`` (the expert has lost track), or None.
    That step and every later step of the episode give no training row (addendum)."""
    for st in traj.get("steps") or []:
        dec = [d for d in st.get("decisions") or [] if d.get("dp") == "action"]
        if dec and dec[0].get("gold") == FALLBACK:
            return int(st["step"])
    return None


DROPPED: Counter[str] = Counter()


def rows_from_runs(sources: Mapping[str, Path], seed: int = BUILD_SEED) -> list[dict[str, Any]]:
    """Training rows from the harvest trajectories (latest non-errored record per game): every logged action decision
    with a state, its admissible actions and a gold label among them (``training.data.harvest_trajectories``, options
    capped at 52 with the gold kept), labelled one-hot with the gold; meta: source, game, task type, step, off-plan.
    Rows from the first fallback label of an episode on are dropped (``lost_from``; counted in ``DROPPED``)."""
    import tempfile

    from jevharness.training.data import harvest_trajectories

    rows: list[dict[str, Any]] = []
    for src, path in sources.items():
        trajs = _latest_ok(Path(path))
        if not trajs:
            continue
        offp = {t["task_id"]: off_plan_steps(t) for t in trajs}
        lost = {t["task_id"]: lost_from(t) for t in trajs}
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for t in trajs:
                f.write(json.dumps(t) + "\n")
            tmp = f.name
        try:
            got = harvest_trajectories([tmp], label_source="gold", max_options=MAX_OPTIONS, seed=seed)
        finally:
            Path(tmp).unlink(missing_ok=True)
        for r in got:
            if r.get("dp") != "action" or r.get("kind") != "choice":
                continue
            task_id, step, _dp = r["id"].rsplit("/", 2)
            if lost.get(task_id) is not None and int(step) >= int(lost[task_id]):
                DROPPED[f"{src}:after_fallback"] += 1
                continue
            r["id"] = f"{src}:{r['id']}"
            r["source"] = f"e11:{src}"
            r["question"] = QUESTION
            r["meta"] = {**(r.get("meta") or {}), "src": src, "task_id": task_id, "game": game_of(task_id),
                         "task_type": task_type_of(game_of(task_id)), "step": int(step),
                         "off_plan": bool(offp.get(task_id, {}).get(int(step), False))}
            rows.append(r)
    return rows


def check_rows(rows: Iterable[Mapping[str, Any]], cal: Iterable[str], eval_ids: Iterable[str]) -> None:
    """Registration (b): every row comes from an ALFWorld TRAIN game that is not a calibration game and names no
    valid_seen trial."""
    cal = set(cal)
    trials = {t.split("/")[-1] for t in eval_ids}
    for r in rows:
        tid = r["meta"]["task_id"]
        if not tid.startswith("alfworld/train/"):
            raise SystemExit(f"row {r['id']}: not an ALFWorld train game ({tid})")
        if r["meta"]["game"] in cal:
            raise SystemExit(f"row {r['id']}: a calibration game ({r['meta']['game']})")
        if tid.split("/")[-1] in trials:
            raise SystemExit(f"row {r['id']}: names a valid_seen trial")


def dedupe(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    out = []
    for r in rows:
        k = _sha(r["state"] + "\x00" + "\x01".join(sorted(r["options"])))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out, len(rows) - len(out)


def _bucket(n: int) -> str:
    return "2-5" if n <= 5 else "6-16" if n <= 16 else "17-52" if n <= 52 else ">52"


def row_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    m = [r["meta"] for r in rows]
    return {"n": len(rows), "by_source": dict(Counter(x["src"] for x in m)),
            "off_plan_share": float(np.mean([bool(x["off_plan"]) for x in m])),
            "games": len({x["game"] for x in m}),
            "by_task_type": dict(sorted(Counter(x["task_type"] for x in m).items())),
            "options_logged": dict(sorted(Counter(_bucket(int(x.get("n_options_logged") or len(r["options"])))
                                                  for x, r in zip(m, rows)).items())),
            "options_median": float(np.median([len(r["options"]) for r in rows])),
            "state_chars_median": float(np.median([len(r["state"]) for r in rows]))}


def build(sources: Mapping[str, Path], dev_games: Iterable[str], cal: Iterable[str], eval_ids: Iterable[str],
          max_train: int = MAX_TRAIN_ROWS, seed: int = BUILD_SEED) -> dict[str, Any]:
    """Rows -> development rows (the development games; never trained on) and train rows (deduplicated on (state,
    option set), first kept; a seeded uniform sample of ``max_train`` if more)."""
    DROPPED.clear()
    rows = rows_from_runs(sources, seed)
    check_rows(rows, cal, eval_ids)
    dev_set = set(dev_games)
    dev = [r for r in rows if r["meta"]["game"] in dev_set]
    train = [r for r in rows if r["meta"]["game"] not in dev_set]
    dev, dev_dup = dedupe(dev)
    train, train_dup = dedupe(train)
    n_before_cap = len(train)
    if len(train) > max_train:
        keep = set(random.Random(seed).sample(range(len(train)), max_train))
        train = [r for i, r in enumerate(train) if i in keep]
    return {"train": train, "dev": dev,
            "stats": {"harvested": row_stats(rows), "train": row_stats(train), "dev": row_stats(dev),
                      "duplicates_dropped": {"train": train_dup, "dev": dev_dup}, "train_before_cap": n_before_cap,
                      "fallback_dropped": dict(DROPPED),
                      "cap": max_train, "seed": seed}}


def episode_stats(path: Path) -> dict[str, Any]:
    trajs = _latest_ok(path)
    if not trajs:
        return {"episodes": 0}
    return {"episodes": len(trajs), "success": float(np.mean([C.success(t) for t in trajs])),
            "steps_mean": float(np.mean([len(t.get("steps") or []) for t in trajs])),
            "cost_usd": float(sum(E6A.episode_spend(t)["total"] for t in trajs))}


def cmd_build(a: argparse.Namespace) -> int:
    hv = json.loads(HARVEST_LIST.read_text())
    runs = Path(a.runs)
    sources = {"flash": runs / RUN_FLASH / f"{FLASH_ARM}.jsonl", "scripted": runs / RUN_SCRIPTED / f"{SCRIPTED_ARM}.jsonl"}
    eval_ids = json.loads(EVAL_LIST.read_text())["task_ids"]
    res = build(sources, hv["dev"], cal_games(), eval_ids, a.max_train)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from jevharness.training.data import write_jsonl
    files = {"train": res["train"], "dev": res["dev"],
             "dev_off_plan": [r for r in res["dev"] if r["meta"]["off_plan"]],
             "dev_flash": [r for r in res["dev"] if r["meta"]["src"] == "flash"]}
    for name, rows in files.items():
        write_jsonl(rows, out / f"{name}.jsonl")
    counts = runs / RUN_SCRIPTED / "error_counts.json"
    man = {"row": LABEL, "generated": _now(), "git": C.git_head(), "harvest_list_sha256": hv["list_sha256"],
           "sources": {k: {"path": C.rel(p), "sha256": C.sha256_file(p) if p.exists() else None, **episode_stats(p)}
                       for k, p in sources.items()},
           "scripted_executor": {"eps": EPS, "kinds": ERROR_KINDS, "seed": SCRIPTED_SEED,
                                 "actions_by_kind": json.loads(counts.read_text()) if counts.exists() else None},
           "label": "the environment's gold next action, one-hot (no teacher); the expert wrapper's fallback 'look' "
                    "is no label (rows from an episode's first fallback on are dropped); harvest-time move tracking of "
                    "the hand-coded expert (patch_handcoded_move)", "max_options": MAX_OPTIONS,
           **res["stats"], "files": {n: {"rows": len(r), "sha256": C.sha256_file(out / f"{n}.jsonl")}
                                     for n, r in files.items()}}
    (out / "build_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("sources", "train", "dev", "duplicates_dropped", "train_before_cap")},
                     indent=1))
    return 0


# ============================================================================== threshold
def cmd_fit(a: argparse.Namespace) -> int:
    run = Path(a.run)
    res = E6A.fit_tau(run, CAL_ARM)
    trajs = list(E6A._latest(run / f"{CAL_ARM}.jsonl").values())
    res["cal_run"] = {"dir": C.rel(run), "arm": CAL_ARM, "episodes": len(trajs),
                      "errored": sum(bool(t.get("error")) for t in trajs),
                      "success": float(np.mean([C.success(t) for t in trajs])) if trajs else None,
                      "sha256": C.sha256_file(run / f"{CAL_ARM}.jsonl")}
    res["cal_states"] = cal_state_accuracy(trajs)
    TAU_FILE.parent.mkdir(parents=True, exist_ok=True)
    TAU_FILE.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"wrote {C.rel(TAU_FILE)}: n_paired {res.get('n_paired')}, rule {(res.get('tau_rule') or {}).get('rule')}, "
          f"tau_hat {res.get('tau_hat')} -> deploy {res['deploy']}; calibration states {res['cal_states']}")
    return 0


def cal_state_accuracy(trajs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """On the calibration shadow arm's states: the trained model's answer and Flash's executed action against the
    single gold action (same states; lower bounds), and the trained model's mean confidence."""
    recs = [d for t in trajs for st in t.get("steps") or [] for d in st.get("decisions") or []
            if d.get("dp") == "action" and d.get("gold") is not None]
    if not recs:
        return {"n": 0}
    s1 = [d for d in recs if d.get("s1_value") is not None]
    s2 = [d for d in recs if d.get("s2_value") is not None]
    return {"n": len(recs), "model_acc": float(np.mean([d["s1_value"] == d["gold"] for d in s1])) if s1 else None,
            "flash_acc": float(np.mean([d["s2_value"] == d["gold"] for d in s2])) if s2 else None,
            "model_mean_conf": float(np.mean([float(d["s1_conf"]) for d in s1 if d.get("s1_conf") is not None]))
            if s1 else None}


def set_tau_text(text: str, deploy: Mapping[str, Any]) -> str:
    """The config with the cascade arm's pending threshold replaced by the fitted one (or, when the fit always defers,
    the arm commented out: it is not run). Exactly one pending line must be present; nothing else changes."""
    lines = text.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if PENDING_RE.search(ln)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one pending cascade line, found {len(hits)}")
    i = hits[0]
    if deploy.get("always_defer"):
        lines[i] = "  # " + lines[i].strip() + "  # the fit always defers: not run\n"
    else:
        tau = float(deploy["tau"])
        lines[i] = PENDING_RE.sub(f"action: {{mode: cascade, tau: {tau!r}}}", lines[i])
    return "".join(lines)


def cmd_set_tau(a: argparse.Namespace) -> int:
    res = json.loads(TAU_FILE.read_text())
    deploy = res["deploy"]
    rec = {"tau_file_sha256": C.sha256_file(TAU_FILE), "deploy": deploy, "at": _now(), "configs": {}}
    for name in ("smoke.yaml", "eval.yaml"):
        p = CFG_DIR / name
        before = p.read_text()
        after = set_tau_text(before, deploy)
        diff = [(x, y) for x, y in zip(before.splitlines(), after.splitlines()) if x != y]
        if len(diff) != 1 or len(before.splitlines()) != len(after.splitlines()):
            raise SystemExit(f"{name}: set-tau must change exactly one line")
        p.write_text(after)
        rec["configs"][name] = {"sha256_before": _sha(before), "sha256_after": _sha(after), "line": diff[0][1].strip()}
    out = Path(a.record)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


# ============================================================================== spend, watchdog, projection
def total_spend(runs: Path = RUNS) -> dict[str, Any]:
    per = {name: E6A.run_spend(runs / name) for name in (RUN_FLASH, RUN_SCRIPTED, RUN_CAL, RUN_SMOKE, RUN_EVAL)
           if (runs / name).is_dir()}
    tot = {k: sum(v[k] for v in per.values()) for k in ("gemini", "jev", "total", "records")}
    return {"by_run": per, **tot}


def cmd_spend(a: argparse.Namespace) -> int:
    s = total_spend(Path(a.runs))
    print(f"${s['total']:.3f} ({s['records']} records)" if a.short else json.dumps(s, indent=1))
    return 0


def cmd_watch(a: argparse.Namespace) -> int:
    """E6a's watchdog over E11's runs: at the cap, SIGTERM / SIGKILL to the runner's process group and
    ``<runs>/e11_STOP.json``."""
    import os
    import signal
    import time

    stop = Path(a.runs) / "e11_STOP.json"
    while True:
        try:
            os.kill(a.pid, 0)
        except OSError:
            print(f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} runner {a.pid} gone; watchdog exits", flush=True)
            return 0
        s = total_spend(Path(a.runs))
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


def projection(runs: Path = RUNS, n_games: int = N_EVAL, safety: float = SAFETY, cap: float = CAP_USD) -> dict[str, Any]:
    """Registration (k): the smoke's mean $ per episode per arm, summed over the arms that ran, x the games x
    ``safety``, plus the spend so far, against the cap; else the first ``MIN_EVAL_GAMES`` (same check); else stop."""
    arms = (REF, *ARMS.values())
    sm = {arm: list(E6A._latest(runs / RUN_SMOKE / f"{arm}.jsonl").values()) for arm in arms}
    per = {arm: (float(np.mean([E6A.episode_spend(t)["total"] for t in ts])) if ts else None) for arm, ts in sm.items()}
    errored = {arm: sum(bool(t.get("error")) for t in ts) for arm, ts in sm.items()}
    so_far = total_spend(runs)["total"]
    ep = sum(v for v in per.values() if v is not None)
    out: dict[str, Any] = {"smoke_per_episode": per, "smoke_n": {k: len(v) for k, v in sm.items()},
                           "smoke_errored": errored, "spent_so_far": so_far, "safety": safety, "cap": cap}
    for n in (n_games, MIN_EVAL_GAMES):
        out[f"projected_{n}"] = ep * n * safety
        out[f"fits_{n}"] = so_far + ep * n * safety <= cap
    if not sm[REF]:
        out["decision"] = "stop after the smoke (no llm_only smoke episode)"
    else:
        out["decision"] = (f"run all {n_games}" if out[f"fits_{n_games}"] else
                           f"run the first {MIN_EVAL_GAMES}" if out[f"fits_{MIN_EVAL_GAMES}"] else "stop after the smoke")
    return out


def cmd_project(a: argparse.Namespace) -> int:
    p = projection(Path(a.runs))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p, indent=1) + "\n")
    print(json.dumps(p, indent=1))
    return 0


# ============================================================================== readout
def tariff(runs: Path, window: Path) -> dict[str, Any]:
    """E7's convention (A7.13 (iii)): q = host $/h x evaluation window / completed episodes of the GPU arms."""
    s24 = json.loads(S24_TARIFF.read_text())["q_usd_per_episode"] if S24_TARIFF.exists() else NAN
    if not window.exists():
        return {"q_usd_per_episode": NAN, "state": "no window record", "sensitivity": {"s2_4": s24, "zero": 0.0}}
    w = json.loads(window.read_text())
    it = str(w.get("instance_type") or "")
    if it not in HOST_USD_PER_HOUR:
        raise SystemExit(f"no on-demand price recorded for instance type {it!r}")
    t0 = dt.datetime.fromisoformat(str(w["window_start"]).replace("Z", "+00:00"))
    t1 = dt.datetime.fromisoformat(str(w["window_end"]).replace("Z", "+00:00"))
    hours = (t1 - t0).total_seconds() / 3600
    done = 0
    for arm in GPU_ARMS:
        p = runs / RUN_EVAL / f"{arm}.jsonl"
        if p.exists():
            done += sum(1 for t in E6A._latest(p).values() if not t.get("error"))
    q = HOST_USD_PER_HOUR[it] * hours / done if done and hours > 0 else NAN
    return {"q_usd_per_episode": q, "host_usd_per_hour": HOST_USD_PER_HOUR[it], "instance_type": it,
            "serving_hours": hours, "episodes": done, "window": {"start": w["window_start"], "end": w["window_end"]},
            "state": "ok", "convention": "E7 / S2-4 provisioned tariff (A7.13 (iii)): host $/h x evaluation window / "
                                         "completed GPU-arm episodes; on-demand price, us-east-1",
            "sensitivity": {"s2_4": s24, "zero": 0.0}}


def with_tariff(arm_t: Mapping[str, Mapping[str, Any]], ref_t: Mapping[str, Mapping[str, Any]], ids: Sequence[str],
                q: float) -> dict[str, Any]:
    """Saving 1 - (C_arm + q) / C_ref on the paired complete case (task bootstrap)."""
    keys = [t for t in ids if t in arm_t and t in ref_t and not arm_t[t].get("error") and not ref_t[t].get("error")]
    if not keys or not math.isfinite(q):
        return {"q": q, "saving": None, "saving_ci": [None, None]}
    ca = [E6A.episode_spend(arm_t[t])["total"] + q for t in keys]
    cr = [E6A.episode_spend(ref_t[t])["total"] for t in keys]
    sv = C.paired_boot(ca, cr, ratio=True)
    return {"q": q, "cost_usd": float(np.mean(ca)), "saving": sv["est"], "saving_ci": sv["ci"]}


def offline_block(box: Path) -> dict[str, Any]:
    """eval_decisions outputs of the box (development rows; trained model with temperatures, base model)."""
    out: dict[str, Any] = {}
    for tag in ("trained", "base"):
        p = box / "eval" / f"dev_{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        files = d.get("files") or d.get("results") or {}
        if isinstance(files, dict):
            for name, v in files.items():
                key = Path(str(name)).name.replace(".jsonl", "")
                o = (v or {}).get("overall") or {}
                out.setdefault(tag, {})[key] = {k: o.get(k) for k in ("n", "acc", "ece", "mean_conf", "auroc")}
    return out


def summarize(runs: Path = RUNS, box: Path = BOX_DIR) -> dict[str, Any]:
    ev = json.loads(EVAL_LIST.read_text())
    run = runs / RUN_EVAL
    exp_n = None
    if (run / "experiment.yaml").exists():
        import yaml
        exp = yaml.safe_load((run / "experiment.yaml").read_text())
        exp_n = int((exp.get("suite") or {}).get("n") or ev["n"])
    ids = ev["task_ids"][: exp_n or ev["n"]]
    tt = {name: E6A._latest(run / f"{name}.jsonl") for name in (REF, *ARMS.values())}
    tau = json.loads(TAU_FILE.read_text()) if TAU_FILE.exists() else {}
    tf = tariff(runs, box / "eval_window.json")
    man = json.loads((box / "data" / "build_manifest.json").read_text()) if (box / "data" / "build_manifest.json").exists() else {}
    train_rec = json.loads((box / "train_record.json").read_text()) if (box / "train_record.json").exists() else {}
    res: dict[str, Any] = {
        "row": LABEL, "label": "descriptive, revises no registered result; a new checkpoint, not AgentDecision-4B",
        "generated": _now(), "git": C.git_head(),
        "eval": {"split": "valid_seen", "n_games": len(ids), "list_sha256": _sha("\n".join(ids)), "run": C.rel(run),
                 "files": {n: {"sha256": C.sha256_file(run / f"{n}.jsonl"),
                               "records": sum(1 for _ in open(run / f"{n}.jsonl"))}
                           for n in tt if (run / f"{n}.jsonl").exists()}},
        "settings": {"agent_llm": "gemini-3.5-flash (temperature 0, reasoning_effort minimal)",
                     "decision_model": f"{SERVED} (Qwen3-4B + LoRA, SFT on ALFWorld train states; serve_jev)",
                     "max_steps": 50, "history_window": 12, "schedule": "task_major, seed 732"},
        "data": {k: man.get(k) for k in ("sources", "scripted_executor", "harvested", "train", "dev",
                                         "duplicates_dropped", "fallback_dropped", "train_before_cap", "files")
                 if k in man},
        "training": train_rec,
        "tau": {"tau_hat": tau.get("tau_hat"), "deploy": tau.get("deploy"), "rule": (tau.get("tau_rule") or {}).get("rule"),
                "n_paired": tau.get("n_paired"), "cal_games": (tau.get("cal_run") or {}).get("episodes"),
                "cal_success": (tau.get("cal_run") or {}).get("success"), "cal_states": tau.get("cal_states"),
                "V": (tau.get("primary") or {}).get("V"), "delta_c": (tau.get("primary") or {}).get("delta_c")},
        "offline": offline_block(box), "tariff": tf, "reference": E6A.ref_block(tt[REF], ids), "arms": {}}
    for mode, name in ARMS.items():
        if not tt[name]:
            always = mode == "cascade" and bool((tau.get("deploy") or {}).get("always_defer"))
            res["arms"][mode] = {"arm": name, "status": "always defers (not run)" if always else "not run"}
            continue
        blk = {"arm": name, "status": "run", **E6A.arm_block(tt[name], tt[REF], ids, mode)}
        blk["saving_api"], blk["saving_api_ci"] = blk.get("saving"), blk.get("saving_ci")
        q = tf.get("q_usd_per_episode", NAN)
        wt = with_tariff(tt[name], tt[REF], ids, q)
        blk["tariff"] = wt
        blk["saving"], blk["saving_ci"] = wt["saving"], wt["saving_ci"]
        blk["sensitivity"] = {k: with_tariff(tt[name], tt[REF], ids, v)
                              for k, v in (tf.get("sensitivity") or {}).items()}
        res["arms"][mode] = blk
    res["by_task_type"] = E6A.by_type({"llm_only": tt[REF], **{m: tt[n] for m, n in ARMS.items()}}, ids)
    res["spend"] = total_spend(runs)
    e6a = E6A.OUT_DIR / "summary.json"
    if e6a.exists():
        d = json.loads(e6a.read_text())
        res["e6a"] = {"llm_only": {k: d["reference"].get(k) for k in ("success", "cost_usd", "wall_mean_s")},
                      **{m: {k: (d["arms"].get(m) or {}).get(k) for k in ("success", "delta_pt", "saving",
                                                                           "wall_mean_s")}
                         for m in ("cascade", "replace")}}
    return res


def _f(x: Any, d: int = 1, sign: bool = False, scale: float = 1.0) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    y = x * scale
    return f"{y:+.{d}f}" if sign else f"{y:.{d}f}"


def to_markdown(r: Mapping[str, Any]) -> str:
    ref, t, tf = r["reference"], r["tau"], r["tariff"]
    tr, dat = r.get("training") or {}, r.get("data") or {}
    L = [f"# E11: the training recipe on ALFWorld ({r['row']})", "",
         f"- **{r['label']}.** Outside every Holm table; no test. Generated {r['generated']} at git "
         f"`{(r['git'] or '?')[:12]}` by `scripts/explore_e11_alfworld.py`.",
         f"- Data (ALFWorld train games only): {_f((dat.get('train') or {}).get('n'), 0)} train rows / "
         f"{_f((dat.get('dev') or {}).get('n'), 0)} development rows; sources "
         + ", ".join(f"{k} {v.get('episodes')} episodes (success {_f(100 * v['success'] if v.get('success') is not None else None)} %)"
                     for k, v in (dat.get("sources") or {}).items())
         + f"; train rows after an off-plan step {_f(100 * (dat.get('train') or {}).get('off_plan_share', NAN))} %; "
         f"rows dropped from the expert's first fallback on {sum((dat.get('fallback_dropped') or {}).values())}, "
         f"duplicates dropped {(dat.get('duplicates_dropped') or {}).get('train')}.",
         f"- Training: {tr.get('summary', '--')}.",
         f"- CASCADE threshold on {t.get('cal_games')} calibration train games: tau-hat = {_f(t.get('tau_hat'), 4)} "
         f"(rule `{t.get('rule')}`, {t.get('n_paired')} paired decisions); on those states the trained model matched the "
         f"gold at {_f(100 * ((t.get('cal_states') or {}).get('model_acc') or NAN))} %, Flash at "
         f"{_f(100 * ((t.get('cal_states') or {}).get('flash_acc') or NAN))} %.",
         f"- GPU tariff q = ${_f(tf.get('q_usd_per_episode'), 5)} per GPU-arm episode ({tf.get('instance_type')}, "
         f"{_f(tf.get('serving_hours'), 3)} h / {tf.get('episodes')} episodes); sensitivity: q = 0 and S2-4's q.",
         f"- LLM only: success {_f(ref.get('success'), scale=100)} % (n {ref['n']}, errored {ref['n_errored']}), "
         f"${_f(ref.get('cost_usd'), 4)} per episode, wall mean / median {_f(ref.get('wall_mean_s'))} / "
         f"{_f(ref.get('wall_median_s'))} s, {_f(ref.get('steps'))} steps; its executed action matched the single gold "
         f"action at {_f(((ref.get('decisions') or {}).get('llm_acc') or NAN), scale=100)} % of its steps (a lower bound).", "",
         "| arm | n | success % | Δ pt [95 % bootstrap] | wins / losses | $ / episode API (+ q) | saving % with q "
         "[95 %] | saving % API only | wall s mean / median | steps | LLM calls | coverage % | acc. on taken % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode, a in r["arms"].items():
        if a.get("status") != "run" or not a.get("n"):
            L.append(f"| {mode} (`{a['arm']}`) | -- | {a.get('status')} | | | | | | | | | | |")
            continue
        d = a["decisions"]
        L.append(f"| {mode} (`{a['arm']}`) | {a['n']} | {_f(a['success'], scale=100)} | {_f(a['delta_pt'], sign=True)} "
                 f"[{_f(a['delta_ci_pt'][0], sign=True)}, {_f(a['delta_ci_pt'][1], sign=True)}] | {a['wins']} / {a['losses']} | "
                 f"{_f(a['cost_usd'], 4)} (+ {_f(a['tariff']['q'], 4)}) | {_f(a['saving'], scale=100)} "
                 f"[{_f(a['saving_ci'][0], scale=100)}, {_f(a['saving_ci'][1], scale=100)}] | {_f(a['saving_api'], scale=100)} | "
                 f"{_f(a['wall_mean_s'])} / {_f(a['wall_median_s'])} | {_f(a['steps'])} | {_f(a['llm_calls'])} | "
                 f"{_f(d.get('coverage'), scale=100)} | {_f(d.get('acc'), scale=100)} |")
    off = r.get("offline") or {}
    if off:
        L += ["", "## Offline, development rows (held-out train games)", "",
              "| model | file | n | acc. % | ECE | mean conf. |", "|---|---|---|---|---|---|"]
        for tag, files in off.items():
            for name, m in files.items():
                L.append(f"| {tag} | {name} | {m.get('n')} | {_f(m.get('acc'), scale=100)} | {_f(m.get('ece'), 3)} | "
                         f"{_f(m.get('mean_conf'), 3)} |")
    L += ["", "## Success by task type (descriptive, small n)", "",
          "| task type | n | LLM only | cascade | replace |", "|---|---|---|---|---|"]
    for typ, v in r["by_task_type"].items():
        L.append(f"| {typ} | {v['n']} | " + " | ".join(_f(v.get(k), 0, scale=100) for k in ("llm_only", "cascade", "replace"))
                 + " |")
    if r.get("e6a"):
        e = r["e6a"]
        L += ["", f"E6a (Jev, same games, earlier run): LLM only {_f(e['llm_only']['success'], scale=100)} %; Jev cascade "
              f"{_f(e['cascade']['success'], scale=100)} % ({_f(e['cascade']['delta_pt'], sign=True)} pt, saving "
              f"{_f(e['cascade']['saving'], scale=100)} %); Jev alone {_f(e['replace']['success'], scale=100)} %."]
    s = r["spend"]
    L += ["", "## Spend (logged, list prices)", "",
          f"Gemini ${s['gemini']:.2f} ({s['records']} records incl. errored attempts) over "
          + ", ".join(f"`{k}` ${v['total']:.2f}" for k, v in s["by_run"].items()) + f"; cap ${CAP_USD:.0f}.", ""]
    return "\n".join(L)


def tex_keys(r: Mapping[str, Any]) -> dict[str, str]:
    v: dict[str, str] = {}

    def put(k: str, x: Any, d: int = 1, sign: bool = False, scale: float = 1.0) -> None:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return
        y = x * scale
        v[k] = f"{y:+.{d}f}" if sign else f"{y:.{d}f}"

    def put_int(k: str, x: Any) -> None:
        if isinstance(x, (int, float)) and math.isfinite(x):
            v[k] = f"{int(x):,}".replace(",", "{,}")

    v["e11/label"] = LABEL
    put_int("e11/n_games", r["eval"]["n_games"])
    dat = r.get("data") or {}
    put_int("e11/data/train_rows", (dat.get("train") or {}).get("n"))
    put_int("e11/data/dev_rows", (dat.get("dev") or {}).get("n"))
    put("e11/data/off_plan_share", (dat.get("train") or {}).get("off_plan_share"), 1, scale=100)
    srcs = dat.get("sources") or {}
    put_int("e11/data/flash_games", (srcs.get("flash") or {}).get("episodes"))
    put_int("e11/data/scripted_games", (srcs.get("scripted") or {}).get("episodes"))
    put_int("e11/data/train_games", (dat.get("train") or {}).get("games"))
    tr = r.get("training") or {}
    put("e11/train/minutes", tr.get("minutes"), 0)
    put("e11/train/gpu_hours", tr.get("gpu_hours_total"), 1)
    t = r["tau"]
    put("e11/tau", t.get("tau_hat"), 2)
    put_int("e11/cal_games", t.get("cal_games"))
    put_int("e11/cal_paired", t.get("n_paired"))
    cs = t.get("cal_states") or {}
    put("e11/cal/model_acc", cs.get("model_acc"), 1, scale=100)
    put("e11/cal/flash_acc", cs.get("flash_acc"), 1, scale=100)
    for tag, files in (r.get("offline") or {}).items():
        for name, m in files.items():
            put(f"e11/offline/{tag}/{name}/acc", m.get("acc"), 1, scale=100)
            put(f"e11/offline/{tag}/{name}/ece", m.get("ece"), 3)
    tf = r["tariff"]
    put("e11/tariff/q", tf.get("q_usd_per_episode"), 5)
    put("e11/tariff/host_usd_h", tf.get("host_usd_per_hour"), 3)
    ref = r["reference"]
    put("e11/llm_only/success", ref.get("success"), 1, scale=100)
    put("e11/llm_only/cost_k", ref.get("cost_usd"), 1, scale=1000)
    put("e11/llm_only/wall_mean", ref.get("wall_mean_s"), 1)
    put("e11/llm_only/wall_median", ref.get("wall_median_s"), 1)
    put("e11/llm_only/steps", ref.get("steps"), 1)
    put("e11/llm_only/llm_acc", (ref.get("decisions") or {}).get("llm_acc"), 1, scale=100)
    put_int("e11/data/fallback_dropped", sum((dat.get("fallback_dropped") or {}).values()) if dat.get("fallback_dropped") else None)
    put_int("e11/llm_only/n", ref.get("n"))
    for mode, a in r["arms"].items():
        p = f"e11/{mode}"
        if a.get("status") != "run" or not a.get("n"):
            v[f"{p}/status"] = str(a.get("status"))
            continue
        v[f"{p}/status"] = "run"
        put_int(f"{p}/n", a["n"])
        put(f"{p}/success", a["success"], 1, scale=100)
        put(f"{p}/delta", a["delta_pt"], 1, sign=True)
        put(f"{p}/delta_ci/lo", a["delta_ci_pt"][0], 1, sign=True)
        put(f"{p}/delta_ci/hi", a["delta_ci_pt"][1], 1, sign=True)
        v[f"{p}/wins"], v[f"{p}/losses"] = str(a["wins"]), str(a["losses"])
        put(f"{p}/cost_k", a["cost_usd"], 1, scale=1000)
        put(f"{p}/saving", a["saving"], 1, scale=100)
        put(f"{p}/saving_ci/lo", (a.get("saving_ci") or [None])[0], 1, scale=100)
        put(f"{p}/saving_ci/hi", (a.get("saving_ci") or [None, None])[1], 1, scale=100)
        put(f"{p}/saving_api", a.get("saving_api"), 1, scale=100)
        for k, sens in (a.get("sensitivity") or {}).items():
            put(f"{p}/saving_q_{k}", sens.get("saving"), 1, scale=100)
        put(f"{p}/wall_mean", a["wall_mean_s"], 1)
        put(f"{p}/wall_median", a["wall_median_s"], 1)
        put(f"{p}/steps", a["steps"], 1)
        put(f"{p}/llm_calls", a["llm_calls"], 1)
        put(f"{p}/coverage", a["decisions"].get("coverage"), 1, scale=100)
        put(f"{p}/acc", a["decisions"].get("acc"), 1, scale=100)
        put(f"{p}/deferred_llm_acc", a["decisions"].get("deferred_llm_acc"), 1, scale=100)
    s = r["spend"]
    put("e11/spend/gemini", s["gemini"], 2)
    # alias for the paper's hook (sections/08_results.tex, \ebArm picks the first of rl, trained, ...): e11/trained/* is
    # the registered primary arm, the action cascade with the trained model
    for k in [k for k in v if k.startswith("e11/cascade/")]:
        v["e11/trained/" + k[len("e11/cascade/"):]] = v[k]
    return v


def to_tex(r: Mapping[str, Any]) -> str:
    lines = ["% AUTO-GENERATED by scripts/explore_e11_alfworld.py -- do not edit by hand.",
             "% E11: exploratory, descriptive row (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-07 'E11').",
             f"% generated {r['generated']} at git {(r['git'] or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(tex_keys(r).items())]
    return "\n".join(lines + [r"\makeatother", ""])


def cmd_summarize(a: argparse.Namespace) -> int:
    r = summarize(Path(a.runs), Path(a.box))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (OUT_DIR / "summary.md").write_text(to_markdown(r) + "\n")
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(to_tex(r))
    print(to_markdown(r))
    print(f"wrote {C.rel(OUT_DIR / 'summary.json')}, {C.rel(OUT_DIR / 'summary.md')}, {C.rel(TEX)}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--json-zip", required=True)
    s.add_argument("--twpddl-zip", required=True)
    s.add_argument("--data-dir", default=str(DATA_SCRIPTED))
    s.add_argument("--flash-dir", default=str(DATA_FLASH))
    s.add_argument("--n-per-type", type=int, default=N_PER_TYPE)
    s = sub.add_parser("harvest-scripted")
    s.add_argument("--out", default=str(RUNS))
    s.add_argument("--data-dir", default=str(DATA_SCRIPTED))
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--max-tasks", type=int, default=None)
    s = sub.add_parser("harvest-flash")
    s.add_argument("--config", default=str(CFG_DIR / "harvest_flash.yaml"))
    s.add_argument("--out", default=str(RUNS))
    s.add_argument("--max-tasks", type=int, default=None)
    s = sub.add_parser("build")
    s.add_argument("--runs", default=str(RUNS))
    s.add_argument("--out", required=True)
    s.add_argument("--max-train", type=int, default=MAX_TRAIN_ROWS)
    s = sub.add_parser("fit")
    s.add_argument("--run", default=str(RUNS / RUN_CAL))
    s = sub.add_parser("set-tau")
    s.add_argument("--record", default=str(BOX_DIR / "set_tau.json"))
    s = sub.add_parser("spend")
    s.add_argument("--runs", default=str(RUNS))
    s.add_argument("--short", action="store_true")
    s = sub.add_parser("project")
    s.add_argument("--runs", default=str(RUNS))
    s.add_argument("--out", default=str(BOX_DIR / "projection.json"))
    s = sub.add_parser("watch")
    s.add_argument("--runs", default=str(RUNS))
    s.add_argument("--pid", type=int, required=True)
    s.add_argument("--cap", type=float, default=CAP_USD)
    s.add_argument("--every", type=float, default=30.0)
    s = sub.add_parser("summarize")
    s.add_argument("--runs", default=str(RUNS))
    s.add_argument("--box", default=str(BOX_DIR))
    a = ap.parse_args(argv)
    return {"select": cmd_select, "harvest-scripted": cmd_harvest_scripted, "harvest-flash": cmd_harvest_flash,
            "build": cmd_build, "fit": cmd_fit,
            "set-tau": cmd_set_tau, "spend": cmd_spend, "watch": cmd_watch, "project": cmd_project,
            "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
