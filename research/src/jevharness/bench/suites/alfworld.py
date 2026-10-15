"""ALFWorld (text / TextWorld modality) as an action-only :class:`~jevharness.bench.env.Env`.

Wraps the ``alfworld`` package (https://github.com/alfworld/alfworld, tested with alfworld 0.4.2 on
PyPI == master @ aaba687 for the files used here, textworld 1.7.0). The official batch interface is::

    env = alfworld.agents.environment.get_environment("AlfredTWEnv")(config, train_eval=split)
    env = env.init_env(batch_size=1)          # a textworld.gym batch env cycling over *all* games
    obs, infos = env.reset()                  # infos["admissible_commands"][0], infos["won"][0]

It cycles through the split's games in ``os.walk`` order, which is neither per-task addressable nor
stable across machines. For per-task determinism we mirror ``AlfredTWEnv.collect_game_files``
(same filters: skip ``movable``/``Sliced`` trajectories, task-type filter, ``solvable`` flag), sort the
game files, and give every task its own single-game TextWorld env built exactly like
``AlfredTWEnv.init_env`` does (``textworld.gym.register_games`` with the ``AlfredDemangler`` and
``AlfredInfos`` wrappers, ``EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])``,
no domain randomisation for eval splits).

Splits (``base_config.yaml``):  train -> json_2.1.1/train (3553 games),
eval_in_distribution -> json_2.1.1/valid_seen (140), eval_out_of_distribution -> json_2.1.1/valid_unseen (134).

Gold next actions (``oracle_action``; never shown to the agent, used for decision-level metrics and
the simulated System-1 backends) come from the game file's planner ``walkthrough`` while the agent is
still on it, and from ALFWorld's hand-coded expert (``AlfredExpert(handcoded)`` wrapper, which
re-plans from the current state) once it has deviated. Note that many states admit several equally
good actions, so decision accuracy against this single gold action is a lower bound.

``alfworld`` / ``textworld`` are imported lazily (inside :meth:`AlfWorldEnv.reset`), so this module
and :func:`load_alfworld`'s game listing work without them. Setup: ``docs/ALFWORLD.md``.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import yaml

from ..env import Env, StepResult

TASK_TYPES = {1: "pick_and_place_simple",
              2: "look_at_obj_in_light",
              3: "pick_clean_then_place_in_recep",
              4: "pick_heat_then_place_in_recep",
              5: "pick_cool_then_place_in_recep",
              6: "pick_two_obj_and_place"}

# split name (AlfredTWEnv train_eval) -> base_config.yaml dataset key and default directory
SPLITS = {"train": ("data_path", "json_2.1.1/train"),
          "eval_in_distribution": ("eval_id_data_path", "json_2.1.1/valid_seen"),
          "eval_out_of_distribution": ("eval_ood_data_path", "json_2.1.1/valid_unseen")}
SPLIT_ALIASES = {"valid_seen": "eval_in_distribution", "id": "eval_in_distribution",
                 "valid_unseen": "eval_out_of_distribution", "ood": "eval_out_of_distribution"}

WELCOME = "-= Welcome to TextWorld, ALFRED! =-"
TASK_PREFIX = "Your task is to:"
DEFAULT_EXCLUDE = ("help", "inventory")
ORACLES = (None, "none", "walkthrough", "handcoded", "auto")

# TextWorld is not thread-safe: PDDL games are parsed with one module-level tatsu parser
# (textworld/envs/pddl/logic/__init__.py: _PARSER) and textworld.gym.register_games mutates a global registry.
# The runner executes episodes in threads, so every TextWorld call is serialised with this lock (env steps take
# milliseconds; the LLM / System-1 calls around them still run in parallel).
_TW_LOCK = threading.RLock()


def alfworld_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    """``data_dir`` > ``$ALFWORLD_DATA`` > ``~/.cache/alfworld`` (the default of ``alfworld-download``)."""
    return Path(data_dir or os.environ.get("ALFWORLD_DATA") or Path.home() / ".cache" / "alfworld").expanduser()


def _canon_split(split: str) -> str:
    split = SPLIT_ALIASES.get(split, split)
    if split not in SPLITS:
        raise ValueError(f"unknown ALFWorld split {split!r}; expected one of {sorted(SPLITS)} "
                         f"or aliases {sorted(SPLIT_ALIASES)}")
    return split


def _task_type_names(task_types: list[int | str] | None) -> set[str]:
    if not task_types:
        return set(TASK_TYPES.values())
    return {TASK_TYPES[t] if isinstance(t, int) else str(t) for t in task_types}


def split_root(split: str, data_dir: str | os.PathLike | None = None, config: str | dict | None = None) -> Path:
    """Directory holding a split's games. With ``config`` (an ALFWorld ``base_config.yaml`` path or dict)
    the ``dataset.*_path`` entry is used (``$ALFWORLD_DATA`` is expanded), else ``<data_dir>/json_2.1.1/...``."""
    split = _canon_split(split)
    key, rel = SPLITS[split]
    if config is not None:
        cfg = yaml.safe_load(Path(config).read_text()) if not isinstance(config, dict) else config
        path = (cfg.get("dataset") or {}).get(key)
        if path:
            os.environ.setdefault("ALFWORLD_DATA", str(alfworld_data_dir(data_dir)))
            return Path(os.path.expandvars(path)).expanduser()
    return alfworld_data_dir(data_dir) / rel


def collect_game_files(split: str = "eval_out_of_distribution", data_dir: str | os.PathLike | None = None,
                       task_types: list[int | str] | None = None, config: str | dict | None = None) -> list[str]:
    """Sorted list of playable ``game.tw-pddl`` files; same filters as ``AlfredTWEnv.collect_game_files``.
    The task type is read from ``traj_data.json`` exactly like ALFWorld does."""
    if config is not None and task_types is None:
        cfg = yaml.safe_load(Path(config).read_text()) if not isinstance(config, dict) else config
        task_types = (cfg.get("env") or {}).get("task_types")
    wanted = _task_type_names(task_types)
    root = split_root(split, data_dir, config)
    games: list[str] = []
    for dirpath, _dirs, files in os.walk(root):
        if "traj_data.json" not in files:
            continue
        if "movable" in dirpath or "Sliced" in dirpath:
            continue
        game = os.path.join(dirpath, "game.tw-pddl")
        if not os.path.exists(game):
            continue
        with open(os.path.join(dirpath, "traj_data.json")) as f:
            if json.load(f).get("task_type") not in wanted:
                continue
        with open(game) as f:
            if not json.load(f).get("solvable", False):
                continue
        games.append(game)
    return sorted(games)


def make_textworld_env(game_file: str, max_episode_steps: int = 50, expert: bool = False) -> Any:
    """Single-game batch env (batch_size=1) built like ``AlfredTWEnv.init_env`` for an eval split."""
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import (AlfredDemangler, AlfredExpert, AlfredExpertType,
                                                           AlfredInfos)

    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    wrappers: list[Any] = [AlfredDemangler(shuffle=False), AlfredInfos]
    if expert:
        wrappers.append(AlfredExpert(AlfredExpertType.HANDCODED))
        request_infos.extras.append("expert_plan")
    with _TW_LOCK:
        env_id = textworld.gym.register_games([game_file], request_infos, batch_size=1, asynchronous=False,
                                              max_episode_steps=max_episode_steps, wrappers=wrappers,
                                              name=f"jev-{uuid.uuid4().hex[:12]}")
        env = textworld.gym.make(env_id)
        getattr(textworld.gym, "registry", {}).pop(env_id, None)
    return env


class AlfWorldEnv(Env):
    """One ALFWorld game. Action-only: ``admissible_actions()`` + ``step("act", {"action": a})``.

    ``evaluate()`` finalises the episode and releases the TextWorld env (the harness never calls
    ``close()``); ``reset()`` may be called again afterwards.
    """

    def __init__(self, game_file: str, split: str = "eval_out_of_distribution", max_episode_steps: int = 50,
                 oracle: str | None = "auto", exclude_commands: tuple[str, ...] | list[str] = DEFAULT_EXCLUDE,
                 tw_factory: Any = None):
        if oracle not in ORACLES:
            raise ValueError(f"oracle must be one of {ORACLES}")
        self.game_file = str(game_file)
        self.split = _canon_split(split)
        self.max_episode_steps = max_episode_steps
        self.oracle = None if oracle in (None, "none") else oracle
        self.exclude = set(exclude_commands or ())
        self._tw_factory = tw_factory or make_textworld_env   # injectable for tests
        p = Path(self.game_file)
        trial, task_dir = p.parent.name, p.parent.parent.name
        self.task_id = f"alfworld/{SPLITS[self.split][1].split('/')[-1]}/{task_dir}/{trial}"
        self.task_type = next((t for t in TASK_TYPES.values() if task_dir.startswith(t + "-")), "unknown")
        self.instruction = ""
        self.gold = {"task_type": self.task_type}
        self._tw: Any = None
        self._admissible: list[str] = []
        self._expert_plan: list[str] = []
        self._walkthrough: list[str] | None = None
        self._history: list[str] = []
        self._won = False
        self._done = False
        self._score = 0.0

    # ------------------------------------------------------------------ lifecycle
    def _load_walkthrough(self) -> list[str]:
        if self._walkthrough is None:
            try:
                with open(self.game_file) as f:
                    self._walkthrough = list(json.load(f).get("walkthrough") or [])
            except (OSError, ValueError):
                self._walkthrough = []
        return self._walkthrough

    def _update(self, infos: dict[str, Any]) -> None:
        self._admissible = [str(a) for a in (infos.get("admissible_commands") or [[]])[0]]
        self._won = bool((infos.get("won") or [False])[0])
        plan = infos.get("extra.expert_plan")
        self._expert_plan = list(plan[0] or []) if plan else []

    @staticmethod
    def _split_intro(text: str) -> tuple[str, str]:
        """Initial observation -> (room description, task sentence)."""
        body = text.replace(WELCOME, "").strip()
        task = ""
        keep = []
        for line in body.splitlines():
            if line.strip().startswith(TASK_PREFIX):
                task = line.strip()
            else:
                keep.append(line)
        return "\n".join(keep).strip(), task

    def reset(self) -> str:
        self.close()
        want_expert = self.oracle in ("handcoded", "auto")
        with _TW_LOCK:
            self._tw = self._tw_factory(self.game_file, self.max_episode_steps, want_expert)
            obs, infos = self._tw.reset()
        self._history, self._done, self._score = [], False, 0.0
        self._update(infos)
        room, task = self._split_intro(obs[0])
        self.instruction = task or self.instruction
        if self.oracle in ("walkthrough", "auto"):
            self.gold = {"task_type": self.task_type, "walkthrough": self._load_walkthrough()}
        return room or obs[0]

    def close(self) -> None:
        # NB: TextWorld's PddlEnv loads a private copy of libdownward.so per game and never unloads it
        # (~5MB RSS per episode). Unloading it ourselves (fast_downward.close_lib -> dlclose) segfaults in
        # threaded runs, so we only drop our reference.
        if self._tw is not None:
            with _TW_LOCK:
                try:
                    self._tw.close()
                except Exception:  # pragma: no cover
                    pass
            self._tw = None

    # ------------------------------------------------------------------ Env protocol
    def admissible_actions(self) -> list[str]:
        seen: set[str] = set()
        out = []
        for a in self._admissible:
            if a in self.exclude or a in seen:
                continue
            seen.add(a)
            out.append(a)
        return out

    def step(self, tool: str, args: dict[str, Any]) -> StepResult:
        if tool != "act":
            return StepResult(f"Error: unknown tool {tool!r}; this environment only accepts act(action=...).")
        if self._tw is None or self._done:
            return StepResult("The episode is over.", done=True, info={"won": self._won})
        action = str(args.get("action", "")).strip()
        try:
            with _TW_LOCK:
                obs, scores, dones, infos = self._tw.step([action])
        except Exception as e:  # e.g. the hand-coded expert wrapper raising "Timeout" after its 200-step budget
            self._done = True
            self._history.append(action)
            return StepResult(f"Error: environment failure ({type(e).__name__}: {e})", done=True,
                              info={"won": self._won, "error": str(e)})
        self._history.append(action)
        self._update(infos)
        self._score = float(scores[0] or 0.0)
        self._done = bool(dones[0]) or self._won
        return StepResult(str(obs[0]), done=self._done, info={"won": self._won, "score": self._score})

    def on_walkthrough(self) -> bool:
        wt = self._load_walkthrough()
        return self._history == wt[: len(self._history)]

    def oracle_action(self) -> tuple[str, dict[str, Any]] | None:
        if self.oracle is None or self._tw is None or self._done:
            return None
        gold = None
        if self.oracle in ("walkthrough", "auto") and self.on_walkthrough():
            wt = self._load_walkthrough()
            if len(self._history) < len(wt):
                gold = wt[len(self._history)]
        if gold is None and self.oracle in ("handcoded", "auto") and self._expert_plan:
            cand = self._expert_plan[0]   # right after reset the hand-coded expert always says "look"
            gold = cand if cand in self._admissible else None
        return ("act", {"action": gold}) if gold else None

    def oracle_done(self) -> bool | None:
        return self._won

    def evaluate(self, final_answer: str | None) -> dict[str, Any]:
        res = {"success": bool(self._won), "score": 1.0 if self._won else 0.0, "steps": len(self._history),
               "task_type": self.task_type}
        if self._walkthrough is not None:
            res["expert_steps"] = len(self._walkthrough)
        self.close()
        return res


def load_alfworld(split: str = "eval_out_of_distribution", n: int | None = None, **kw: Any) -> list[Any]:
    """Zero-arg factories for :class:`AlfWorldEnv`, one per game, in a stable (sorted) order.

    Keyword args: ``data_dir`` (default ``$ALFWORLD_DATA`` or ``~/.cache/alfworld``), ``config`` (an ALFWorld
    ``base_config.yaml`` for data paths / task types), ``task_types`` (ids 1-6 or names), ``offset``,
    ``seed`` (shuffle before taking ``n``), ``stratify`` (interleave the task types round-robin, in
    ``TASK_TYPES`` order, so any prefix -- e.g. a 6-game pilot -- covers the types evenly; combine with
    ``seed`` to shuffle within each type), ``max_episode_steps`` (50), ``oracle`` ("auto" | "walkthrough" |
    "handcoded" | None), ``exclude_commands`` (default: help, inventory).
    """
    split = _canon_split(split)
    games = collect_game_files(split, kw.get("data_dir"), kw.get("task_types"), kw.get("config"))
    if not games:
        root = split_root(split, kw.get("data_dir"), kw.get("config"))
        raise FileNotFoundError(
            f"no ALFWorld games found under {root}. Install `alfworld` and run `alfworld-download` "
            "(or set ALFWORLD_DATA / pass data_dir=...); see docs/ALFWORLD.md")
    if kw.get("seed") is not None:
        import random
        random.Random(kw["seed"]).shuffle(games)
    if kw.get("stratify"):
        by_type: dict[str, list[str]] = {}
        for g in games:  # keeps the (sorted or shuffled) order within each type
            task_dir = Path(g).parent.parent.name
            by_type.setdefault(next((t for t in TASK_TYPES.values() if task_dir.startswith(t + "-")), "unknown"),
                               []).append(g)
        order = [t for t in TASK_TYPES.values() if t in by_type] + [t for t in by_type if t not in TASK_TYPES.values()]
        queues = [by_type[t] for t in order]
        games = [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]
    off = int(kw.get("offset", 0))
    games = games[off: off + n if n else None]
    env_kw = {k: kw[k] for k in ("max_episode_steps", "oracle", "exclude_commands") if k in kw}
    return [(lambda g=g: AlfWorldEnv(g, split=split, **env_kw)) for g in games]


__all__ = ["AlfWorldEnv", "TASK_TYPES", "alfworld_data_dir", "collect_game_files", "load_alfworld",
           "make_textworld_env"]
