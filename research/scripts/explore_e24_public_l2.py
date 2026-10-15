#!/usr/bin/env python3
r"""E24 (exploratory, descriptive; docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 "E24: agent-state accuracy (L2) on
public benchmarks"): an agent-state corpus built from the stored live logs of the agent LLM alone on a PUBLIC benchmark,
labelled by the benchmark's own expert, and every decision model's offline answers on the same states.

Part 1, L2-ALFWorld: the states of Flash LLM-only on ALFWorld valid_seen (E11's evaluation run) and valid_unseen (E23's
evaluation run). Each logged episode is replayed action by action in a fresh TextWorld game (deterministic: the
replay must reproduce every logged option list and observation), and every state is labelled by ALFWorld's own experts:
the game's planner walkthrough while the agent is still on it, else the hand-coded expert, which re-plans from the
current state (E11's harvest labeler: its tracker reads ``move X to Y`` as ``put X to Y``). The strict label is that
single action; the lenient set adds what ALFWorld's experts would equally have done (the hand-coded expert's own
action, the object instance it draws at random, the receptacle it picks by list order). No model chose these states.

    python scripts/explore_e24_public_l2.py build-alfworld          # replay + labels -> results/explore/e24_public_l2/alfworld
    python scripts/explore_e24_public_l2.py replay --corpus DIR --slug SLUG [--s1 JSON] [--cache F] [--workers 4] [--cap-usd X]
    python scripts/explore_e24_public_l2.py quick --corpus DIR --slug SLUG
    python scripts/explore_e24_public_l2.py summarize                # summary.{md,json} + paper/generated/e24_public_l2.tex

Score (registration): per game, the mean correctness over its labelled states; every game weighs the same; a missing,
failed, refused or invalid answer is wrong. Labels never reach a backend: the question file holds state and options only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/explore/e24_public_l2"
ALF_DIR = OUT / "alfworld"
TEX = ROOT / "paper/generated/e24_public_l2.tex"
CFG = ROOT / "configs/explore/e24_public_l2"
LABEL = ("exploratory (E24), descriptive; PREREG_A7_STAGE2 Deviations 2026-10-08 'E24: agent-state accuracy (L2) on "
         "public benchmarks'; outside every Holm table")
QUESTION = "Which action should the agent take next?"     # the live harness's action question (harness/agent.py)
DP = "action"
ALF_DATA = ROOT / "data/raw/alfworld/json_2.1.1"
# the logged Flash LLM-only episodes (no decision model chose a state); split -> (run file, row that produced it)
ALF_SOURCES = {"seen": (ROOT / "runs/e11_alfworld_eval/llm_only.jsonl", "E11 (valid_seen)"),
               "unseen": (ROOT / "runs/e23_alfworld_unseen_eval/llm_only.jsonl", "E23 (valid_unseen)")}
FALLBACK = "look"            # ALFWorld's AlfredExpert wrapper: the plan when the hand-coded expert has no admissible action
REPLAY_SEED = "jev-explore-E24-alfworld-replay-2026-10-08"
N_BOOT, BOOT_SEED = 2000, 0
TASK_TYPES = ("pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
              "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place")
TT_SHORT = {"pick_and_place_simple": "place", "look_at_obj_in_light": "look", "pick_clean_then_place_in_recep": "clean",
            "pick_heat_then_place_in_recep": "heat", "pick_cool_then_place_in_recep": "cool",
            "pick_two_obj_and_place": "pick2"}
# slug -> (name in the paper, group, serving); group: trained-on-ALFWorld / trained elsewhere (transfer) / off-the-shelf
MODELS: dict[str, tuple[str, str, str]] = {
    "e11": ("ALFWorld-trained 4B (E11)", "alfworld", "serve_jev, E11's adapter and temperatures"),
    "qwen4b": ("Qwen3-4B (untrained base)", "offshelf", "serve_jev, no adapter"),
    "e12": ("AgentDecision-4B (released)", "transfer", "serve_jev, E12's adapter and temperatures"),
    "e14a": ("0.8B scoring head", "transfer", "serve_jev --backend clef (E14 arm A)"),
    "sl4b": ("StartLux-4B", "offshelf", "the vendor's server"),
    "sl9b": ("StartLux-9B", "offshelf", "the vendor's server"),
    "sl27b": ("StartLux-27B (Q8_0)", "offshelf", "the vendor's GGUF server"),
    "jev": ("Jev", "offshelf", "hosted (OpenRouter, Jev-only key)"),
    "oaidec": ("OpenAI Decisions API", "offshelf", "hosted (POST /v1/decisions, gpt-6-luna)"),
    "e22": ("tau^2-trained 0.8B head (E22)", "tau2", "serve_jev --backend clef, E22's final checkpoint"),
}
GPU_SLUGS = ("e11", "qwen4b", "e12", "e14a", "sl4b", "sl9b", "sl27b", "e22")
HOSTED = ("jev", "oaidec")
FLASH_ROW = "flash"          # reference: the agent LLM's own executed action on its own states (in sample)
NAN = float("nan")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def jl(p: Path) -> list[dict[str, Any]]:
    op = gzip.open if str(p).endswith(".gz") else open
    out = []
    with op(p, "rt") as f:
        for x in f:
            if x.strip():
                try:
                    out.append(json.loads(x))
                except json.JSONDecodeError:
                    continue
    return out


def write_jl_gz(rows: Iterable[Mapping[str, Any]], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode())


def question_hash(state: str, options: Sequence[str], text: str = QUESTION) -> str:
    """As ``stage2_corpus.question_hash`` (state, options, descriptions, question text; no label enters it)."""
    blob = json.dumps({"state": state, "options": list(options), "descriptions": [], "text": text}, sort_keys=True,
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def latest_by_task(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in jl(path):
        if isinstance(r, dict) and "task_id" in r:
            out[str(r["task_id"])] = r
    return out


# ================================================================================================ ALFWorld labels
def install_label_patches() -> None:
    """E11's harvest labeler (``patch_handcoded_move``: the hand-coded expert's tracker reads ``move X to Y`` as
    ``put X to Y``) plus read-only instrumentation of the hand-coded expert, which changes none of its actions:
    after every ``BasePolicy.act`` the policy records its subgoal and, for a search ``go to`` or a GOTO subgoal, the
    receptacles it could equally have picked (``_e24_info``); ``HandCodedAgent.act`` is re-stated verbatim except that
    it records when it returned its random fallback (the policy proposed the agent's last action again), which is no
    label. Idempotent; this process only."""
    import explore_e11_alfworld as E11
    from alfworld.agents.expert import handcoded_expert as H

    E11.patch_handcoded_move()
    if getattr(H.BasePolicy.act, "_e24", False):
        return
    orig_act = H.BasePolicy.act

    def act(self: Any, game_state: Any, last_action: str) -> Any:
        self._e24_info = {"ret": None, "kind": None, "tier": [], "random_fallback": False}
        ret = orig_act(self, game_state, last_action)
        info: dict[str, Any] = {"ret": ret, "kind": None, "tier": [], "random_fallback": False}
        try:
            sa, sp, _objs = self.get_next_subgoal()
        except Exception:  # noqa: BLE001 - past the last subgoal
            sa, sp = None, None
        info["subgoal"] = [sa, sp]
        tc = list(self.receptacles_to_check)
        if sa == "find" and tc and ret == "go to {}".format(tc[-1]):
            if sp in self.obj_cls_to_receptacle_map:          # seen before: one known place, no choice
                tier = [tc[-1]]
            else:
                prior = set(self.get_list_of_receptacles_of_type(sp) if sp in self.RECEPTACLES
                            else self.get_list_of_receptacles_to_search_for_object_cls(sp))
                tier = [r for r in tc if (r in prior) == (tc[-1] in prior)]
            info.update(kind="search", tier=tier)
        elif sa == "goto" and isinstance(ret, str) and ret.startswith("go to "):
            info.update(kind="goto", tier=list(self.get_list_of_receptacles_of_type(sp)))
        self._e24_info = info
        return ret

    act._e24 = True  # type: ignore[attr-defined]
    H.BasePolicy.act = act

    def hact(self: Any, game_state: Any, reward: Any, done: Any, last_action: str = "") -> Any:
        action = self.policy.act(game_state, last_action)
        if action == last_action or (last_action == "look" and action not in game_state["admissible_commands"]):
            info = dict(getattr(self.policy, "_e24_info", {}) or {})
            info["random_fallback"] = True
            self.policy._e24_info = info
            return H.random.choice(game_state["admissible_commands"])
        return action

    hact._e24 = True  # type: ignore[attr-defined]
    H.HandCodedAgent.act = hact


def _expert_policy(env: Any) -> Any:
    """The hand-coded expert's policy object inside an AlfWorldEnv's TextWorld env (the AlfredExpert wrapper)."""
    seen: set[int] = set()
    stack = [getattr(env, "_tw", None)]
    while stack:
        o = stack.pop()
        if o is None or id(o) in seen:
            continue
        seen.add(id(o))
        he = getattr(o, "_handcoded_expert", None)
        if he is not None:
            return getattr(he, "policy", None)
        for name in ("batch_env", "envs", "env", "_env", "_wrapped_env", "unwrapped"):
            v = getattr(o, name, None)
            if isinstance(v, (list, tuple)):
                stack.extend(v)
            elif v is not None:
                stack.append(v)
    return None


_TAKE = re.compile(r"^take (\S+) (\d+) from (.+)$")
_USE = re.compile(r"^use (\S+) (\d+)$")


def instance_equivalents(action: str | None, options: Sequence[str]) -> set[str]:
    """``take O from R`` -> every offered ``take O' from R`` with O' of O's class; ``use O`` -> every offered ``use O'``
    of O's class. The hand-coded expert draws that instance at random among the visible objects of the class
    (``random.choice(objs_of_interest)``), and every ALFWorld goal names an object class, not an instance."""
    if not action:
        return set()
    m = _TAKE.match(action)
    if m:
        cls, recep = m.group(1), m.group(3)
        return {o for o in options if (mm := _TAKE.match(o)) and mm.group(1) == cls and mm.group(3) == recep}
    m = _USE.match(action)
    if m:
        cls = m.group(1)
        return {o for o in options if (mm := _USE.match(o)) and mm.group(1) == cls}
    return {action} if action in options else set()


def lenient_set(strict: str, hand: str | None, info: Mapping[str, Any] | None, options: Sequence[str]) -> list[str]:
    """Registration (b) 3: the strict label, the hand-coded expert's own action (when it is a label), the instance
    equivalents of both, and for the hand-coded expert's search ``go to`` (or GOTO subgoal) every offered ``go to`` of a
    receptacle it could equally have picked (its search list's current priority tier, or every receptacle of the
    subgoal's type)."""
    opts = set(options)
    acc = {strict} | instance_equivalents(strict, options)
    if hand:
        acc |= {hand} | instance_equivalents(hand, options)
        if info and info.get("kind") in ("search", "goto") and info.get("ret") == hand:
            acc |= {f"go to {r}" for r in info.get("tier") or []}
    return sorted(a for a in acc if a in opts)


def _game_file(task_id: str, data_dir: Path = ALF_DATA) -> Path:
    _, split, tdir, trial = task_id.split("/")
    return data_dir / split / tdir / trial / "game.tw-pddl"


def _seed_for(task_id: str) -> int:
    return int(hashlib.sha256((REPLAY_SEED + "\n" + task_id).encode()).hexdigest()[:8], 16)


def relabel_episode(traj: Mapping[str, Any], game_file: Path, env_factory: Callable[..., Any] | None = None
                    ) -> dict[str, Any]:
    """Replay one logged episode action by action and label every state (registration (b)). Returns
    {states: [...], checks: {...}}; every state carries the logged state string and options, the strict label (or
    None with the reason), the lenient set, the hand-coded expert's own action and the logged gold."""
    from alfworld.agents.expert import handcoded_expert as H

    from jevharness.bench.suites.alfworld import AlfWorldEnv

    task_id = str(traj["task_id"])
    H.random.seed(_seed_for(task_id))      # the expert's random instance draws, reproducible per game
    random.seed(_seed_for(task_id))
    env = (env_factory or AlfWorldEnv)(str(game_file), split=task_id.split("/")[1], oracle="auto")
    env.reset()
    states: list[dict[str, Any]] = []
    checks = {"steps": 0, "options_mismatch": 0, "obs_mismatch": 0, "desync_from": None}
    lost = False
    for st in traj.get("steps") or []:
        dec = [d for d in st.get("decisions") or [] if d.get("dp") == DP]
        if not dec:
            continue
        d = dec[0]
        step = int(st.get("step", len(states)))
        checks["steps"] += 1
        logged_opts = list(d.get("options") or [])
        if checks["desync_from"] is None and sorted(env.admissible_actions()) != sorted(logged_opts):
            checks["options_mismatch"] += 1
            checks["desync_from"] = step
        on_wt = env.on_walkthrough()
        wt = env._load_walkthrough()
        plan = list(env._expert_plan or [])
        pol = _expert_policy(env)
        info = dict(getattr(pol, "_e24_info", {}) or {}) if (pol is not None and step > 0) else {}
        hand_raw = plan[0] if plan else None
        reason, fallback_here = None, False
        if step == 0:
            hand = None                       # right after reset the hand-coded expert always says "look"
        elif hand_raw == FALLBACK:
            hand, reason, fallback_here = None, "expert_fallback", True
        elif info.get("random_fallback"):
            hand, reason = None, "expert_random_fallback"
        else:
            hand = hand_raw if hand_raw in logged_opts else None
            reason = None if hand else "expert_not_offered"
        strict, source = None, None
        if on_wt and len(env._history) < len(wt):
            strict, source = wt[len(env._history)], "walkthrough"
        else:
            lost = lost or fallback_here      # E11's lost_from: off the walkthrough, the expert's first fallback on
            if hand is not None and not lost:
                strict, source = hand, "handcoded"
        drop = None
        if checks["desync_from"] is not None:
            drop = "replay_desync"
        elif lost:
            drop = "after_expert_fallback"
        elif strict is None:
            drop = reason or "no_label"
        elif strict not in logged_opts:
            drop = "label_not_offered"
        rec = {"step": step, "state": d.get("state") or "", "options": logged_opts, "strict": None, "lenient": [],
               "source": source, "hand": hand if not lost else None, "hand_kind": info.get("kind"),
               "logged_gold": d.get("gold"), "executed": (st.get("args") or {}).get("action"), "drop": drop,
               "on_walkthrough": bool(on_wt)}
        if drop is None:
            rec["strict"] = strict
            rec["lenient"] = lenient_set(strict, rec["hand"], info, logged_opts)
            take = _TAKE.match(strict or "") or _USE.match(strict or "")
            rec["strict_instances"] = len(instance_equivalents(strict, logged_opts)) if take else 1
        states.append(rec)
        res = env.step("act", {"action": rec["executed"]})
        if checks["desync_from"] is None and str(res.observation).strip() != str(st.get("observation") or "").strip():
            checks["obs_mismatch"] += 1
            checks["desync_from"] = step + 1
        if res.done:
            break
    env.evaluate(None)
    return {"states": states, "checks": checks}


def off_plan_flags(states: Sequence[Mapping[str, Any]]) -> list[bool]:
    """state i -> True when an earlier executed action of the episode differed from that earlier state's strict label
    (E11's off-plan definition; states without a strict label do not count)."""
    out, off = [], False
    for s in states:
        out.append(off)
        if s.get("strict") is not None and s.get("executed") is not None and s["executed"] != s["strict"]:
            off = True
    return out


def relabel_games(split: str, path: Path, task_ids: Sequence[str] | None = None, data_dir: Path = ALF_DATA,
                  env_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Replay + labels for the given games of one source run (all when ``task_ids`` is None), in this process."""
    if env_factory is None:
        install_label_patches()
    trajs = latest_by_task(Path(path))
    ids = sorted(trajs) if task_ids is None else list(task_ids)
    out: dict[str, Any] = {}
    for tid in ids:
        t = trajs[tid]
        if t.get("error"):
            out[tid] = {"errored": True}
            continue
        res = relabel_episode(t, _game_file(tid, data_dir), env_factory)
        res["task_type"] = str((t.get("result") or {}).get("task_type") or tid.split("/")[2].split("-")[0])
        res["success"] = bool((t.get("result") or {}).get("success"))
        out[tid] = res
    return out


def relabel_chunked(split: str, path: Path, chunk: int = 25, max_games: int | None = None) -> dict[str, Any]:
    """``relabel_games`` in child processes of ``chunk`` games each: TextWorld loads a private 31 MB copy of
    libdownward.so per game into a temporary directory and frees it only when the process exits (alfworld.py's note),
    which this container's disk cannot hold for 274 games. Temporary files go to /dev/shm when it exists."""
    import os
    import tempfile

    ids = sorted(latest_by_task(Path(path)))
    if max_games is not None:
        ids = ids[:max_games]
    env = dict(os.environ)
    if Path("/dev/shm").is_dir():
        env["TMPDIR"] = "/dev/shm"
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as td:
        for i in range(0, len(ids), chunk):
            part = Path(td) / f"part_{i}.json"
            idf = Path(td) / f"ids_{i}.json"
            idf.write_text(json.dumps(ids[i:i + chunk]))
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "relabel-chunk", "--split", split,
                                "--ids", str(idf), "--out", str(part)], env=env, capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"relabel-chunk {split} {i}: exit {r.returncode}: {r.stderr[-2000:]}")
            out.update(json.loads(part.read_text()))
            print(f"{split}: {min(i + chunk, len(ids))} / {len(ids)} games relabelled", file=sys.stderr, flush=True)
    return out


def cmd_relabel_chunk(a: argparse.Namespace) -> int:
    path = ALF_SOURCES[a.split][0]
    res = relabel_games(a.split, path, json.loads(Path(a.ids).read_text()))
    Path(a.out).write_text(json.dumps(res))
    return 0


def assemble(relabelled: Mapping[str, Mapping[str, Any]], sources: Mapping[str, tuple[Path, str]] = ALF_SOURCES
             ) -> dict[str, Any]:
    """Registration (b): occurrences (one per labelled state; label fields), distinct questions (state + options only)."""
    occ: list[dict[str, Any]] = []
    questions: dict[str, dict[str, Any]] = {}
    dropped: Counter[str] = Counter()
    games: dict[str, dict[str, Any]] = {}
    checks_tot: Counter[str] = Counter()
    errored: list[str] = []
    for split in sources:
        row = sources[split][1]
        for tid, res in sorted((relabelled.get(split) or {}).items()):
            if res.get("errored"):
                errored.append(tid)
                continue
            ch = res["checks"]
            checks_tot["steps"] += ch["steps"]
            checks_tot["options_mismatch"] += ch["options_mismatch"]
            checks_tot["obs_mismatch"] += ch["obs_mismatch"]
            checks_tot["games_desynced"] += int(ch["desync_from"] is not None)
            tt = res["task_type"]
            games[tid] = {"split": split, "task_type": tt, "success": res["success"], "steps": len(res["states"]),
                          "desync_from": ch["desync_from"]}
            offp = off_plan_flags(res["states"])
            for s, op in zip(res["states"], offp):
                if s["drop"]:
                    dropped[s["drop"]] += 1
                    continue
                qh = question_hash(s["state"], s["options"])
                questions.setdefault(qh, {"qhash": qh, "state": s["state"], "options": list(s["options"])})
                occ.append({"task": tid, "split": split, "task_type": tt, "step": s["step"], "qhash": qh,
                            "strict": s["strict"], "lenient": s["lenient"], "source": s["source"],
                            "hand": s["hand"], "hand_kind": s["hand_kind"], "logged_gold": s["logged_gold"],
                            "flash": s["executed"], "off_plan": bool(op), "n_options": len(s["options"]),
                            "strict_instances": s.get("strict_instances", 1), "row": row})
    return {"occurrences": occ, "questions": questions, "dropped": dict(dropped), "games": games,
            "checks": dict(checks_tot), "errored": errored}


def build_alfworld(sources: Mapping[str, tuple[Path, str]] = ALF_SOURCES, data_dir: Path = ALF_DATA,
                   env_factory: Callable[..., Any] | None = None, max_games: int | None = None,
                   chunked: bool = False) -> dict[str, Any]:
    """Every state of the latest non-errored record of every game of the two LLM-only runs (registration (b))."""
    rel_: dict[str, Any] = {}
    for split, (path, _row) in sources.items():
        if chunked:
            rel_[split] = relabel_chunked(split, Path(path), max_games=max_games)
        else:
            ids = sorted(latest_by_task(Path(path)))[:max_games] if max_games is not None else None
            rel_[split] = relabel_games(split, Path(path), ids, data_dir, env_factory)
    return assemble(rel_, sources)


def corpus_stats(c: Mapping[str, Any]) -> dict[str, Any]:
    occ = c["occurrences"]
    by_split: dict[str, Any] = {}
    for sp in sorted({o["split"] for o in occ}):
        oo = [o for o in occ if o["split"] == sp]
        gl = [g for g in c["games"].values() if g["split"] == sp]
        by_split[sp] = {"games": len(gl), "games_with_states": len({o["task"] for o in oo}), "states": len(oo),
                        "unique_questions": len({o["qhash"] for o in oo}),
                        "off_plan_share": float(np.mean([o["off_plan"] for o in oo])) if oo else NAN,
                        "lenient_gt1_share": float(np.mean([len(o["lenient"]) > 1 for o in oo])) if oo else NAN,
                        "lenient_mean_size": float(np.mean([len(o["lenient"]) for o in oo])) if oo else NAN,
                        "options_median": float(np.median([o["n_options"] for o in oo])) if oo else NAN,
                        "source_walkthrough_share": float(np.mean([o["source"] == "walkthrough" for o in oo])) if oo else NAN,
                        "flash_strict_acc_states": float(np.mean([o["flash"] == o["strict"] for o in oo])) if oo else NAN,
                        "logged_equals_strict_share": float(np.mean([o["logged_gold"] == o["strict"] for o in oo])) if oo else NAN,
                        "by_task_type": dict(sorted(Counter(o["task_type"] for o in oo).items()))}
    return {"games": len(c["games"]), "states": len(occ), "unique_questions": len(c["questions"]),
            "dropped": c["dropped"], "checks": c["checks"], "errored_games": len(c["errored"]), "by_split": by_split,
            "strict_random_instance_states": sum(1 for o in occ if int(o.get("strict_instances") or 1) > 1),
            "hand_kinds": dict(Counter(str(o.get("hand_kind")) for o in occ))}


def write_corpus(c: Mapping[str, Any], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    occ_p, q_p = out / "occurrences.jsonl.gz", out / "questions.jsonl.gz"
    write_jl_gz(c["occurrences"], occ_p)
    write_jl_gz((c["questions"][h] for h in sorted(c["questions"])), q_p)
    meta = {"registered": "PREREG_A7_STAGE2 Deviations 2026-10-08 'E24' (b)", "label": LABEL, "question": QUESTION,
            "decision_point": DP, "generated": _now(), "git": git_head(),
            "sources": {sp: {"path": rel(p), "row": row, "sha256": sha256_file(p) if Path(p).exists() else None}
                        for sp, (p, row) in ALF_SOURCES.items()},
            "replay_seed": REPLAY_SEED, "stats": corpus_stats(c), "games": c["games"], "errored": c["errored"],
            "files": {p.name: sha256_file(p) for p in (occ_p, q_p)}}
    (out / "corpus.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


def cmd_build_alfworld(a: argparse.Namespace) -> int:
    c = build_alfworld(max_games=a.max_games, chunked=True)
    meta = write_corpus(c, Path(a.out))
    print(json.dumps(meta["stats"], indent=1))
    return 0


# ================================================================================================ consequence labels
# Amendment "E24: consequence labels" (Deviations 2026-10-08): the lead label is the acceptable set A(s) = the lenient set
# plus every action after which ALFWorld's hand-coded expert needs fewer steps to win than from s.
EXPERT_BUDGET = 60           # most expert steps rolled out from a state (the game's own limit is 50 steps)
SAMPLE_PER_SPLIT = 40        # states per split whose every option is evaluated (descriptive set sizes)
SAMPLE_SALT = "jev-explore-E24-alfworld-setsize-2026-10-08"


def _find(o: Any, cls_name: str) -> Any:
    seen: set[int] = set()
    stack = [o]
    while stack:
        x = stack.pop()
        if x is None or id(x) in seen:
            continue
        seen.add(id(x))
        if type(x).__name__ == cls_name:
            return x
        for name in ("batch_env", "envs", "env", "_env", "_wrapped_env", "unwrapped"):
            v = getattr(x, name, None)
            if isinstance(v, (list, tuple)):
                stack.extend(v)
            elif v is not None:
                stack.append(v)
    return None


class Brancher:
    """One TextWorld game (one libdownward copy), reset for every branch: replay a prefix of logged actions, then try an
    action and roll ALFWorld's hand-coded expert out."""

    def __init__(self, game_file: str | Path, factory: Callable[..., Any] | None = None):
        from jevharness.bench.suites.alfworld import make_textworld_env

        self.tw = (factory or make_textworld_env)(str(game_file), 200, True)
        self.pddl = _find(self.tw, "PddlEnv")
        self.infos: dict[str, Any] = {}
        self.won = False
        self.n_hist = 0

    def _set(self, infos: Mapping[str, Any]) -> None:
        self.infos = dict(infos)
        self.won = bool((infos.get("won") or [False])[0])

    def goto(self, prefix: Sequence[str], seed: int) -> None:
        try:
            from alfworld.agents.expert import handcoded_expert as H
            H.random.seed(seed)
        except ImportError:      # tests with a scripted game
            pass
        random.seed(seed)
        _obs, infos = self.tw.reset()
        self.pddl = _find(self.tw, "PddlEnv")   # the gym wrapper builds its env on reset
        self._set(infos)
        self.n_hist = 0
        self.expert_broken = False
        for a in prefix:
            self.step(a)

    def step(self, action: str) -> None:
        """One action. When ALFWorld's expert wrapper raises after the game applied the action (e.g. the hand-coded
        expert's ``random.choice`` on an empty list), the expert is broken for this branch (no label from it); the
        game state stays valid, so the planner and the goal check still read it."""
        self.n_hist += 1
        try:
            _obs, _sc, _dn, infos = self.tw.step([action])
        except Exception:  # noqa: BLE001
            self.expert_broken = True
            self.won = bool(self.pddl._pddl_state.check_goal()) if self.pddl is not None else False
            if self.pddl is not None:
                self.infos = {"admissible_commands": [sorted(set(self.pddl.state.get("admissible_commands") or []))]}
            return
        self._set(infos)

    def expert_action(self) -> str | None:
        """The hand-coded expert's next action, or None when it is no label (fallback ``look`` after the first step,
        its random fallback, nothing offered)."""
        if getattr(self, "expert_broken", False):
            return None
        plan = (self.infos.get("extra.expert_plan") or [[]])[0] or []
        a = plan[0] if plan else None
        if a is None or (a == FALLBACK and self.n_hist > 0):
            return None
        pol = _expert_policy_from(self.tw)       # the policy object is new after every reset
        info = getattr(pol, "_e24_info", {}) if pol is not None else {}
        if info and info.get("random_fallback") and self.n_hist > 0:
            return None
        return a

    def rollout(self, budget: int) -> int | None:
        """Steps the hand-coded expert needs from here to win (at most ``budget``); None if it does not win in time."""
        n = 0
        while not self.won:
            if n >= budget:
                return None
            a = self.expert_action()
            if a is None:
                return None
            self.step(a)
            n += 1
        return n


def _expert_policy_from(tw: Any) -> Any:
    seen: set[int] = set()
    stack = [tw]
    while stack:
        o = stack.pop()
        if o is None or id(o) in seen:
            continue
        seen.add(id(o))
        he = getattr(o, "_handcoded_expert", None)
        if he is not None:
            return getattr(he, "policy", None)
        for name in ("batch_env", "envs", "env", "_env", "_wrapped_env", "unwrapped"):
            v = getattr(o, name, None)
            if isinstance(v, (list, tuple)):
                stack.extend(v)
            elif v is not None:
                stack.append(v)
    return None


def _branch_seed(task: str, step: int, action: str | None) -> int:
    return int(hashlib.sha256(f"{REPLAY_SEED}\n{task}\n{step}\n{action}".encode()).hexdigest()[:8], 16)


def _forked(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run ``fn`` in a forked child and return its JSON result: the child inherits the game exactly as it stands (the
    TextWorld / libdownward state lives in process memory), so a branch costs no replay of the prefix. Linux only."""
    import os

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                   # child: one branch, then exit without cleanup
        os.close(r)
        try:
            data = json.dumps(fn())
        except Exception as e:  # noqa: BLE001
            data = json.dumps({"error": f"{type(e).__name__}: {str(e)[:200]}"})
        b = data.encode()
        while b:
            b = b[os.write(w, b):]
        os.close(w)
        os._exit(0)
    os.close(w)
    chunks = []
    while True:
        c = os.read(r, 65536)
        if not c:
            break
        chunks.append(c)
    os.close(r)
    os.waitpid(pid, 0)
    return json.loads(b"".join(chunks).decode() or "{}")


def _reseed(seed: int) -> None:
    try:
        from alfworld.agents.expert import handcoded_expert as H
        H.random.seed(seed)
    except ImportError:      # tests with a scripted game
        pass
    random.seed(seed)


def consequence_game(task: str, actions: Sequence[str], todo: Mapping[int, Sequence[str]], game_file: str | Path,
                     factory: Callable[..., Any] | None = None) -> list[dict[str, Any]]:
    """Amendment rule for one game. The logged actions are replayed once, in order; at every state s to label the game
    is forked: H(s) = the steps ALFWorld's hand-coded expert needs from s to win (at most 60; the opening ``look`` at
    the first state not counted, it only re-reads the room), and per action a the expert's steps to win after a, rolled
    out with budget H(s) - 1. a is accepted iff it wins the game or H(s + a) < H(s). Every branch reseeds the expert's
    random draws from (game, state, action)."""
    br = Brancher(game_file, factory)
    br.goto([], _branch_seed(task, -1, None))
    out: list[dict[str, Any]] = []

    def base(step: int) -> dict[str, Any]:
        _reseed(_branch_seed(task, step, None))
        opening = br.n_hist == 0 and br.expert_action() == FALLBACK      # ALFWorld's expert opens with `look`
        h = br.rollout(EXPERT_BUDGET)
        return {"h": h - 1 if (h is not None and opening) else h}

    def branch(step: int, a: str, h_s: int | None) -> dict[str, Any]:
        _reseed(_branch_seed(task, step, a))
        if a not in set((br.infos.get("admissible_commands") or [[]])[0] or []):
            return {"accepted": False, "why": "not_admissible"}
        br.step(a)
        if br.won:
            return {"accepted": True, "why": "won", "h": 0}
        h_a = br.rollout(h_s - 1) if (h_s is not None and h_s >= 1) else None
        ok = h_a is not None and h_s is not None and h_a < h_s
        return {"accepted": bool(ok), "why": "shrinks" if ok else ("no_expert_baseline" if h_s is None else "no_shrink"),
                "h": h_a}

    for step in sorted(todo):
        while br.n_hist < step:
            br.step(actions[br.n_hist])
        b = _forked(lambda: base(step))
        h_s = b.get("h")
        out.append({"task": task, "step": step, "kind": "base", "h": h_s, **({"error": b["error"]} if "error" in b else {})})
        for a in todo[step]:
            r = _forked(lambda: branch(step, a, h_s))
            if "error" in r:
                r = {"accepted": False, "why": "error", "error": r["error"]}
            out.append({"task": task, "step": step, "kind": "action", "action": a, **r})
    return out


def _consequence_worker(job: tuple[str, list[str], dict[str, list[str]], str]) -> list[dict[str, Any]]:
    install_label_patches()
    task, actions, todo, gf = job
    try:
        return consequence_game(task, actions, {int(k): v for k, v in todo.items()}, gf)
    except Exception as e:  # noqa: BLE001 - a failed game is recorded, its pairs stay unlabelled (counted)
        return [{"task": task, "kind": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"}]


def episode_actions(sources: Mapping[str, tuple[Path, str]] = ALF_SOURCES) -> dict[str, list[str]]:
    """task -> the logged executed actions, in step order."""
    out: dict[str, list[str]] = {}
    for _sp, (path, _row) in sources.items():
        for tid, t in latest_by_task(Path(path)).items():
            out[tid] = [str((st.get("args") or {}).get("action")) for st in t.get("steps") or []]
    return out


def sample_states(occ: Sequence[Mapping[str, Any]], n: int = SAMPLE_PER_SPLIT) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for sp in ("seen", "unseen"):
        oo = sorted((o for o in occ if o["split"] == sp),
                    key=lambda o: hashlib.sha256(f"{SAMPLE_SALT}\n{o['task']}\n{o['step']}".encode()).hexdigest())
        out |= {(o["task"], int(o["step"])) for o in oo[:n]}
    return out


def consequence_todo(cdir: Path, occ: Sequence[Mapping[str, Any]], done: set[tuple[str, int, str]] | None = None
                     ) -> dict[str, dict[int, list[str]]]:
    """task -> step -> actions to evaluate: every answer of every row (answer files in ``cdir``, and Flash's own action)
    that is not in the lenient set, plus every option of the sampled states; minus pairs already evaluated."""
    qs = {q["qhash"]: q for q in jl(Path(cdir) / "questions.jsonl.gz")}
    answers: dict[str, set[str]] = defaultdict(set)
    for p in sorted(Path(cdir).glob("answers_*.jsonl")):
        for r in jl(p):
            v = answer_value(r)
            if v is not None:
                answers[r["qhash"]].add(v)
    samp = sample_states(occ)
    done = done or set()
    todo: dict[str, dict[int, list[str]]] = defaultdict(dict)
    for o in occ:
        key = (o["task"], int(o["step"]))
        want = set(answers.get(o["qhash"], set()))
        if o.get("flash"):
            want.add(o["flash"])
        if key in samp:
            want |= set(qs[o["qhash"]]["options"])
        want = {a for a in want if a not in set(o["lenient"]) or key in samp}
        want = {a for a in want if (o["task"], int(o["step"]), a) not in done}
        if want or key in samp:
            todo[o["task"]][int(o["step"])] = sorted(want)
    return todo


def cmd_consequence(a: argparse.Namespace) -> int:
    """Evaluate the amendment's consequence rule for every pending (state, action) pair, ``workers`` games at a time;
    appends to ``consequence.jsonl`` in the corpus directory (resumes)."""
    import multiprocessing as mp
    import os

    cdir = Path(a.corpus)
    occ = jl(cdir / "occurrences.jsonl.gz")
    outp = cdir / "consequence.jsonl"
    done_rows = jl(outp) if outp.exists() else []
    done = {(r["task"], int(r["step"]), r["action"]) for r in done_rows if r.get("kind") == "action"}
    base_done = {(r["task"], int(r["step"])) for r in done_rows if r.get("kind") == "base"}
    todo = consequence_todo(cdir, occ, done)
    todo = {t: {s: acts for s, acts in steps.items() if acts or (t, s) not in base_done} for t, steps in todo.items()}
    todo = {t: v for t, v in todo.items() if v}
    acts = episode_actions()
    jobs = [(t, acts[t], {str(k): v for k, v in todo[t].items()}, str(_game_file(t))) for t in sorted(todo)]
    jobs.sort(key=lambda j: -sum(len(v) + 1 for v in j[2].values()))      # the longest games first
    if a.max_games is not None:          # smoke only (development; never reported)
        jobs = jobs[: a.max_games]
        todo = {t: todo[t] for t, *_ in jobs}
    n_pairs = sum(len(v) for t in todo for v in todo[t].values())
    print(f"consequence: {len(jobs)} games, {sum(len(v) for v in todo.values())} states, {n_pairs} pairs pending",
          file=sys.stderr, flush=True)
    if Path("/dev/shm").is_dir():
        os.environ["TMPDIR"] = "/dev/shm"
    f = open(outp, "a")
    t0 = time.time()
    with mp.get_context("spawn").Pool(processes=a.workers, maxtasksperchild=8) as pool:
        for i, rows in enumerate(pool.imap_unordered(_consequence_worker, jobs)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"  {i + 1} / {len(jobs)} games ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
    f.close()
    rows = jl(outp)
    print(f"consequence.jsonl: {sum(1 for r in rows if r.get('kind') == 'action')} pairs, "
          f"{sum(1 for r in rows if r.get('kind') == 'error')} game errors", file=sys.stderr)
    return 0


def load_consequence(cdir: Path) -> dict[str, Any]:
    p = Path(cdir) / "consequence.jsonl"
    rows = jl(p) if p.exists() else []
    acc = {(r["task"], int(r["step"]), r["action"]): bool(r["accepted"]) for r in rows if r.get("kind") == "action"}
    base = {(r["task"], int(r["step"])): r for r in rows if r.get("kind") == "base"}
    return {"acc": acc, "base": base, "errors": [r for r in rows if r.get("kind") == "error"]}


# ================================================================================================ Part 2: L2-tau^2
# Deviations 2026-10-09 "E24 Part 2: L2-τ² from E22's test episodes": states rebuilt from E22's committed LLM-only test
# episodes with the adapter's own rendering, labelled where the reference write action is due.
TAU2_DIR = OUT / "tau2"
TAU2_SRC = ROOT / "results/explore/e22_tau2/test_episodes"
TAU2_DOMAINS = ("retail", "airline")
TAU2_ERR_TERMS = ("agent_error", "user_error", "infrastructure_error")


def tau2_question_hash(state: str, options: Sequence[str], descriptions: Sequence[str] | None, text: str) -> str:
    blob = json.dumps({"state": state, "options": list(options), "descriptions": list(descriptions or []),
                       "text": text}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def tau2_episode_states(domain: str, sim: Mapping[str, Any], rec: Mapping[str, Any], env_info: Mapping[str, Any],
                        task: Any, cfg: Any, stats: Counter) -> list[dict[str, Any]]:
    """Registration (a): the labelled states of one simulation (tool-choice question, argument questions, labels,
    Flash's own call), rebuilt with ``JevTau2Agent._view`` from the messages before each logged turn."""
    import types

    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

    from jevharness.bench.suites import tau2 as jt
    from jevharness.harness.tools import OTHER, params_of, tool_desc

    cls = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    agent = jt.JevTau2Agent(tools=env_info["tools"], domain_policy=env_info["policy"], llm="replay", jev_config=cfg,
                            task=task)
    ep = rec.get("episode")
    msgs = list(sim.get("messages") or [])
    idx: dict[int, int] = {}
    for i, m in enumerate(msgs):
        j = ((m.get("raw_data") or {}).get("jev") or {}) if m.get("role") == "assistant" else {}
        if j.get("episode") == ep and j.get("step") is not None:
            idx[int(j["step"])] = i
    names = list(agent.by_name) + [jt.RESPOND]
    descs = [tool_desc(agent.s1_schemas[n]) for n in agent.by_name] + [jt.RESPOND_DESC]
    out: list[dict[str, Any]] = []
    for st in rec.get("steps") or []:
        d = next((x for x in st.get("decisions") or [] if x.get("dp") == "tool_select"), None)
        if d is None or int(st["step"]) not in idx:
            stats["unjoined"] += 1
            continue
        i = idx[int(st["step"])]
        view = agent._view(types.SimpleNamespace(messages=[cls[m["role"]].model_validate(m) for m in msgs[:i]]))
        stats["turns"] += 1
        if view.gold_tool != d.get("gold"):
            stats["gold_mismatch"] += 1
            continue
        if list(d.get("options") or []) != names:
            stats["options_mismatch"] += 1
            continue
        if agent.kinds.get(view.gold_tool) != "write":
            stats["not_write_due"] += 1
            continue
        stats["write_due"] += 1
        tq = {"state": view.state, "text": jt.TOOL_SELECT_Q, "options": names, "descriptions": descs, "dp": "tool_select"}
        tq["qhash"] = tau2_question_hash(view.state, names, descs, jt.TOOL_SELECT_Q)
        call = (msgs[i].get("tool_calls") or [None])[0]
        flash_tool = call["name"] if call else jt.RESPOND
        flash_args = (call or {}).get("arguments") or {}
        args_q = []
        n_max = agent._n_param_options()
        for pinfo in (pp for pp in params_of(agent.s1_schemas[view.gold_tool]) if pp.required):
            gold = (view.gold_args or {}).get(pinfo.name)
            po = agent._param_options(pinfo, view, gold, view.gold_args, n_max)
            if po is None or len(po[0]) < 2:
                stats["arg_not_asked"] += 1
                continue
            opts, _bounded, gold_s = po
            q = agent._param_question(view.gold_tool, pinfo, opts, gold_s)
            aq = {"state": view.state, "text": q.text, "options": list(opts), "descriptions": None, "dp": "tool_params"}
            aq["qhash"] = tau2_question_hash(view.state, opts, None, q.text)
            fv = jt._label(flash_args.get(pinfo.name)) if flash_tool == view.gold_tool else None
            args_q.append({"param": pinfo.name, "q": aq, "label": gold_s,
                           "flash": (fv if fv in opts else OTHER) if fv is not None else None})
            stats["arg_questions"] += 1
            stats["arg_other_label"] += int(gold_s == OTHER)
        out.append({"domain": domain, "task": f"{domain}/{sim.get('task_id')}", "trial": int(sim.get("trial") or 0),
                    "step": int(st["step"]), "tq": tq, "label_tool": view.gold_tool, "label_args": view.gold_args,
                    "args": args_q, "flash_tool": flash_tool, "write_tools": sorted(n for n, k in agent.kinds.items()
                                                                                   if k == "write")})
    return out


def build_tau2(src: Path = TAU2_SRC, domains: Sequence[str] = TAU2_DOMAINS, max_sims: int | None = None) -> dict[str, Any]:
    from tau2.runner import build_environment, get_tasks

    from jevharness.bench.suites import tau2 as jt

    occ: list[dict[str, Any]] = []
    questions: dict[str, dict[str, Any]] = {}
    stats: dict[str, Counter] = {}
    files: dict[str, str] = {}
    for dom in domains:
        base = Path(src) / dom / "llm_only"
        for fn in ("results.json.gz", "jev_decisions.jsonl.gz"):
            files[f"{dom}/{fn}"] = sha256_file(base / fn)
        res = json.load(gzip.open(base / "results.json.gz", "rt"))
        side = jl(base / "jev_decisions.jsonl.gz")
        by_ep = {r["episode"]: r for r in side}
        env = build_environment(dom)
        info = {"tools": env.get_tools(), "policy": env.get_policy()}
        tasks = {str(t.id): t for t in get_tasks(dom)}
        cfg = jt.Tau2JevConfig.model_validate(side[0]["config_dump"])
        st: Counter = Counter()
        sims = list(res.get("simulations") or [])[: max_sims]
        for sim in sims:
            term = str(sim.get("termination_reason") or "")
            rec = by_ep.get(jt._episode_of(sim))
            if rec is None or rec.get("error") or any(t in term.lower() for t in TAU2_ERR_TERMS):
                st["sim_dropped"] += 1
                continue
            st["sims"] += 1
            for o in tau2_episode_states(dom, sim, rec, info, tasks[str(sim["task_id"])], cfg, st):
                for q in [o["tq"]] + [a["q"] for a in o["args"]]:
                    questions.setdefault(q["qhash"], dict(q))
                occ.append({"domain": o["domain"], "task": o["task"], "trial": o["trial"], "step": o["step"],
                            "qhash": o["tq"]["qhash"], "label_tool": o["label_tool"], "label_args": o["label_args"],
                            "args": [{"param": a["param"], "qhash": a["q"]["qhash"], "label": a["label"],
                                      "flash": a["flash"]} for a in o["args"]],
                            "flash": o["flash_tool"], "write_tools": o["write_tools"]})
        stats[dom] = st
    return {"occurrences": occ, "questions": questions, "stats": {d: dict(v) for d, v in stats.items()},
            "source_sha256": files}


def cmd_build_tau2(a: argparse.Namespace) -> int:
    c = build_tau2(max_sims=a.max_sims)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    occ_p, q_p = out / "occurrences.jsonl.gz", out / "questions.jsonl.gz"
    write_jl_gz(c["occurrences"], occ_p)
    write_jl_gz((c["questions"][h] for h in sorted(c["questions"])), q_p)
    occ = c["occurrences"]
    meta = {"registered": "PREREG_A7_STAGE2 Deviations 2026-10-09 'E24 Part 2' (a)", "label": LABEL, "generated": _now(),
            "git": git_head(), "sources": c["source_sha256"], "stats": {
                "by_domain": c["stats"], "states": len(occ), "tasks": len({o["task"] for o in occ}),
                "questions": len(c["questions"]),
                "tool_questions": len({o["qhash"] for o in occ}),
                "arg_questions": len({a["qhash"] for o in occ for a in o["args"]}),
                "by_label_tool": dict(sorted(Counter(o["label_tool"] for o in occ).items()))},
            "files": {p.name: sha256_file(p) for p in (occ_p, q_p)}}
    (out / "corpus.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps(meta["stats"], indent=1))
    return 0


def tau2_score(occ: Sequence[Mapping[str, Any]], ans: Mapping[str, Mapping[str, Any]] | None,
               flash: bool = False) -> dict[str, Any]:
    """Registration (c): task means of the tool choice, the complete call, the wrong-write rate and the argument
    accuracy; retail, airline and pooled, with the task bootstrap on the tool choice."""
    def val(qh: str) -> str | None:
        return answer_value((ans or {}).get(qh)) if ans is not None else None

    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for o in occ:
        tool = o["flash"] if flash else val(o["qhash"])
        tool_ok = float(tool is not None and tool == o["label_tool"])
        arg_ok = [float((a["flash"] if flash else val(a["qhash"])) == a["label"]) for a in o["args"]]
        g = per[o["task"]]
        g["tool"].append(tool_ok)
        g["call"].append(float(tool_ok and all(arg_ok)))
        g["wrong_write"].append(float(tool is not None and tool != o["label_tool"] and tool in set(o["write_tools"])))
        if not flash or tool_ok:      # Flash's argument values exist only where it called the label tool
            g["args"].extend(arg_ok)
    out: dict[str, Any] = {}
    for sp in ("retail", "airline", "pooled"):
        tasks = {t: v for t, v in per.items() if sp == "pooled" or t.startswith(sp + "/")}
        blk: dict[str, Any] = {"n_tasks": len(tasks), "n_states": sum(len(v["tool"]) for v in tasks.values())}
        for k in ("tool", "call", "wrong_write", "args"):
            m = {t: float(np.mean(v[k])) for t, v in tasks.items() if v[k]}
            blk[k] = float(np.mean(list(m.values()))) if m else None
            if k == "tool" and m:
                blk["tool_ci"] = list(boot_ci(m))
        out[sp] = blk
    return out


# ================================================================================================ answers
def backends() -> dict[str, dict[str, Any]]:
    import yaml

    return dict((yaml.safe_load((CFG / "backends.yaml").read_text()) or {}).get("backends") or {})


def answers_path(cdir: Path, slug: str) -> Path:
    return Path(cdir) / f"answers_{slug}.jsonl"


def replay_questions(questions: Sequence[Mapping[str, Any]], backend: Any, retries: int = 2,
                     on_answer: Callable[[dict[str, Any]], None] | None = None,
                     stop: Callable[[], bool] | None = None) -> list[dict[str, Any]]:
    """Registration (c): every question through the live harness's option handling at the action decision point
    (``JevAgent._choice_tournament``, one question per request, the harness knockout above the backend's read width),
    the action question, no label (``gold=None``); a failure is retried ``retries`` times, then recorded with ``error``
    (scored wrong); a refusal is no answer."""
    from jevharness.accounting import Ledger
    from jevharness.harness.agent import JevAgent
    from jevharness.harness.config import DPConfig, HarnessConfig, Mode

    shell = JevAgent(llm=None, cfg=HarnessConfig(name="e24_replay", action=DPConfig(mode=Mode.REPLACE)), s1=backend)
    out: list[dict[str, Any]] = []
    for q in questions:
        if stop is not None and stop():
            break
        rec: dict[str, Any] = {"qhash": q["qhash"], "answer": None, "confidence": None, "error": None, "attempts": 0}
        for attempt in range(retries + 1):
            led = Ledger()
            rec["attempts"] = attempt + 1
            t0 = time.perf_counter()
            try:
                a = shell._choice_tournament(q["state"], q.get("text") or QUESTION, list(q["options"]),
                                             q.get("dp") or DP, led, list(q.get("descriptions") or []) or None,
                                             gold=None)
                rec.update({"answer": a.value, "confidence": a.confidence, "error": None,
                            "valid": a.value in q["options"]})
                if (a.raw or {}).get("refusal"):
                    rec.update({"answer": None, "confidence": 0.0, "valid": False, "refusal": True})
            except Exception as e:  # noqa: BLE001 - a failed answer is counted, never a crash
                rec.update({"answer": None, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            rec.update({"cost_usd": sum(r.cost_usd for r in led.records),
                        "latency_ms": sum(r.latency_ms for r in led.records),
                        "wall_ms": (time.perf_counter() - t0) * 1000, "requests": len(led.records)})
            if rec["error"] is None:
                break
        out.append(rec)
        if on_answer is not None:
            on_answer(rec)
    return out


def cmd_replay(a: argparse.Namespace) -> int:
    """Every distinct question, sorted by hash and sharded over ``workers``; resumes from the answer file; ``--cap-usd``
    stops a hosted backend once its logged cost over its E24 answer files reaches the cap (checked per answer)."""
    from jevharness.decision import make_backend

    cdir = Path(a.corpus)
    qs = jl(cdir / "questions.jsonl.gz")
    out = answers_path(cdir, a.slug)
    done = {r["qhash"] for r in jl(out) if not r.get("error")} if out.exists() else set()
    spec = dict(json.loads(a.s1) if a.s1 else backends()[a.slug])
    spec.pop("cache", None)
    backend = make_backend(spec, cache=a.cache)
    (cdir / f"replay_{a.slug}.json").write_text(json.dumps({
        "registered": "PREREG_A7_STAGE2 Deviations 2026-10-08 'E24' (c)", "backend": backend.describe(),
        "cache": a.cache, "workers": a.workers, "question": QUESTION, "decision_point": DP,
        "order": "sorted question hash, sharded over workers", "batching": "one question per request (the harness "
        "tournament for Choices wider than the backend's max_options)", "retries": 2, "n_questions": len(qs),
        "resumed_with": len(done), "cap_usd": a.cap_usd, "written": _now()}, indent=1, default=str))
    todo = sorted((q for q in qs if q["qhash"] not in done), key=lambda x: x["qhash"])
    files = {p.resolve() for p in Path(OUT).glob(f"*/answers_{a.slug}.jsonl")} | ({out.resolve()} if out.exists() else set())
    spent = sum(float(r.get("cost_usd") or 0) for p in files for r in jl(p))
    lock = threading.Lock()
    state = {"spent": spent, "stop": a.cap_usd is not None and spent >= a.cap_usd}
    f = open(out, "a")

    def write(rec: dict[str, Any]) -> None:
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            state["spent"] += float(rec.get("cost_usd") or 0.0)
            if a.cap_usd is not None and state["spent"] >= a.cap_usd:
                state["stop"] = True

    shards = [todo[i::max(1, a.workers)] for i in range(max(1, a.workers))]
    ths = [threading.Thread(target=replay_questions, args=(sh, backend),
                            kwargs={"retries": 2, "on_answer": write, "stop": lambda: state["stop"]})
           for sh in shards if sh]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    f.close()
    recs = {r["qhash"]: r for r in jl(out)}
    n_err = sum(1 for q in qs if (recs.get(q["qhash"]) or {}).get("error") or q["qhash"] not in recs)
    print(f"answers -> {rel(out)}: {len(recs)} of {len(qs)}; errors / missing {n_err}; logged ${state['spent']:.4f}"
          + ("; STOPPED at the cap" if state["stop"] else ""), file=sys.stderr)
    return 0 if n_err == 0 else (3 if state["stop"] else 2)


# ================================================================================================ score
def answer_value(a: Mapping[str, Any] | None) -> str | None:
    """A missing, failed, refused or invalid answer is no answer (wrong)."""
    a = a or {}
    if a.get("error") or a.get("refusal") or not a.get("valid", True):
        return None
    return a.get("answer")


def occ_correct(o: Mapping[str, Any], val: str | None, rule: str = "strict") -> float | None:
    """Correctness of ``val`` at occurrence ``o`` under a label rule; None when the rule has no label there."""
    if rule == "strict":
        return float(val is not None and val == o["strict"])
    if rule == "lenient":
        return float(val is not None and val in set(o["lenient"]))
    if rule == "acceptable":       # amendment: the lenient set plus the consequence check
        if val is None:
            return 0.0
        if val in set(o["lenient"]):
            return 1.0
        return float(bool((o.get("cons") or {}).get(val, False)))
    if rule == "logged":
        g = o.get("logged_gold")
        if g is None or o.get("logged_lost"):
            return None
        return float(val is not None and val == g)
    raise ValueError(rule)


def mark_logged_lost(occ: Sequence[dict[str, Any]]) -> None:
    """The logged-gold sensitivity (registration (d) 2): the logged fallback ``look`` and every later state of that
    episode carry no logged label (E11's rule)."""
    lost: dict[str, int] = {}
    for o in occ:
        if o.get("logged_gold") == FALLBACK:
            lost[o["task"]] = min(lost.get(o["task"], 10 ** 9), int(o["step"]))
    for o in occ:
        o["logged_lost"] = o["task"] in lost and int(o["step"]) >= lost[o["task"]]


def game_means(occ: Sequence[Mapping[str, Any]], values: Mapping[str, str | None] | Callable[[Mapping[str, Any]], Any],
               rule: str = "strict", keep: Callable[[Mapping[str, Any]], bool] | None = None) -> dict[str, float]:
    """game -> mean correctness over its kept, labelled occurrences."""
    per: dict[str, list[float]] = defaultdict(list)
    for o in occ:
        if keep is not None and not keep(o):
            continue
        v = values(o) if callable(values) else values.get(o["qhash"])
        c = occ_correct(o, v, rule)
        if c is not None:
            per[o["task"]].append(c)
    return {g: float(np.mean(v)) for g, v in per.items() if v}


def boot_ci(per_game: Mapping[str, float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    v = np.array(list(per_game.values()), dtype=float)
    if len(v) < 2:
        return (NAN, NAN)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    m = v[idx].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall's tau-b (ties in either variable corrected); None with fewer than 2 pairs or no untied pair."""
    n = len(x)
    if n < 2:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den > 0 else None


def load_corpus(cdir: Path) -> dict[str, Any]:
    occ = jl(Path(cdir) / "occurrences.jsonl.gz")
    mark_logged_lost(occ)
    meta = json.loads((Path(cdir) / "corpus.json").read_text())
    cons = load_consequence(cdir)
    by_state: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    for (t, st, a), ok in cons["acc"].items():
        by_state[(t, st)][a] = ok
    for o in occ:
        o["cons"] = by_state.get((o["task"], int(o["step"])), {})
        b = cons["base"].get((o["task"], int(o["step"])))
        o["h"] = b.get("h") if b else None
    return {"occ": occ, "meta": meta, "cons": cons}


def missing_pairs(occ: Sequence[Mapping[str, Any]], values: Any) -> int:
    """Answered actions outside the lenient set whose consequence check has not been evaluated (should be 0)."""
    n = 0
    for o in occ:
        v = values(o) if callable(values) else values.get(o["qhash"])
        if v is not None and v not in set(o["lenient"]) and v not in (o.get("cons") or {}):
            n += 1
    return n


def set_sizes(occ: Sequence[Mapping[str, Any]], cdir: Path) -> dict[str, Any]:
    """Amendment (b): on the sampled states (every option evaluated), the mean size of the strict, lenient and
    acceptable sets and the mean number of options."""
    qs = {q["qhash"]: q for q in jl(Path(cdir) / "questions.jsonl.gz")}
    samp = sample_states(occ)
    out: dict[str, Any] = {}
    for sp in ("seen", "unseen"):
        oo = [o for o in occ if o["split"] == sp and (o["task"], int(o["step"])) in samp]
        full = [o for o in oo if all(a in set(o["lenient"]) or a in (o.get("cons") or {}) for a in qs[o["qhash"]]["options"])]
        if not full:
            out[sp] = {"n": 0}
            continue
        acc_sizes = [sum(1 for a in qs[o["qhash"]]["options"] if occ_correct(o, a, "acceptable")) for o in full]
        out[sp] = {"n": len(full), "options": float(np.mean([len(qs[o["qhash"]]["options"]) for o in full])),
                   "lenient": float(np.mean([len(o["lenient"]) for o in full])),
                   "acceptable": float(np.mean(acc_sizes)),
                   "acceptable_share": float(np.mean([a / len(qs[o["qhash"]]["options"]) for a, o in zip(acc_sizes, full)]))}
    return out


def cmd_quick(a: argparse.Namespace) -> int:
    cdir = Path(a.corpus)
    ap_ = answers_path(cdir, a.slug)
    if not ap_.exists():
        print("no answers")
        return 1
    ans = {r["qhash"]: r for r in jl(ap_)}
    occ0 = jl(cdir / "occurrences.jsonl.gz")
    if occ0 and "label_tool" in occ0[0]:          # Part 2 (tau^2)
        sc = tau2_score(occ0, ans)
        err = sum(1 for r in ans.values() if r.get("error"))
        print("tau2 tool acc " + "; ".join(f"{sp} {100 * (sc[sp]['tool'] or 0):.1f}" for sp in ("retail", "airline", "pooled"))
              + f"; {len(ans)} answers, {err} errors, ${sum(float(r.get('cost_usd') or 0) for r in ans.values()):.4f}")
        return 0
    c = load_corpus(cdir)
    vals = {k: answer_value(r) for k, r in ans.items()}
    parts = []
    for sp in ("seen", "unseen"):
        g = game_means(c["occ"], vals, "strict", lambda o, sp=sp: o["split"] == sp)
        if g:
            parts.append(f"{sp} {100 * np.mean(list(g.values())):.1f}")
    err = sum(1 for r in ans.values() if r.get("error"))
    print(f"expert-agreement acc {'; '.join(parts)}; {len(ans)} answers, {err} errors, "
          f"{sum(1 for r in ans.values() if r.get('refusal'))} refusals, ${sum(float(r.get('cost_usd') or 0) for r in ans.values()):.4f}")
    return 0


# ================================================================================================ readout
LIVE = {"e11": ROOT / "results/explore/e11_alfworld/summary.json",
        "e23": ROOT / "results/explore/e23_alfworld_unseen/summary.json",
        "e6a": ROOT / "results/explore/e6a_alfworld/summary.json"}
# (model slug, split) -> (live summary, arm) of the model alone at the action point (REPLACE), quoted not recomputed
LIVE_REPLACE = {("e11", "seen"): ("e11", "replace"), ("e11", "unseen"): ("e23", "replace"), ("jev", "seen"): ("e6a", "replace")}
LIVE_CASCADE = {("e11", "seen"): ("e11", "cascade"), ("e11", "unseen"): ("e23", "cascade"), ("jev", "seen"): ("e6a", "cascade")}
LIVE_FLASH = {"seen": "e11", "unseen": "e23"}
SPLITS = ("seen", "unseen", "pooled")


def _in(split: str) -> Callable[[Mapping[str, Any]], bool]:
    return (lambda o: True) if split == "pooled" else (lambda o, sp=split: o["split"] == sp)


def _acc_block(occ: Sequence[Mapping[str, Any]], values: Any, rule: str,
               keep: Callable[[Mapping[str, Any]], bool]) -> dict[str, Any]:
    g = game_means(occ, values, rule, keep)
    n_states = sum(1 for o in occ if keep(o) and occ_correct(o, None, rule) is not None)
    if not g:
        return {"acc": None, "n_games": 0, "n_states": n_states}
    lo, hi = boot_ci(g)
    return {"acc": float(np.mean(list(g.values()))), "ci": [lo, hi], "n_games": len(g), "n_states": n_states}


def _occ_weighted(occ: Sequence[Mapping[str, Any]], values: Any, keep: Callable[[Mapping[str, Any]], bool]) -> float | None:
    v = [occ_correct(o, values(o) if callable(values) else values.get(o["qhash"]), "acceptable") for o in occ if keep(o)]
    return float(np.mean(v)) if v else None


def score_row(occ: Sequence[Mapping[str, Any]], values: Any) -> dict[str, Any]:
    """Every registered reading of one row (registration (d) 1-2)."""
    out: dict[str, Any] = {}
    for rule in ("acceptable", "lenient", "strict", "logged"):
        out[rule] = {sp: _acc_block(occ, values, rule, _in(sp)) for sp in SPLITS}
    out["occ_weighted"] = {sp: _occ_weighted(occ, values, _in(sp)) for sp in SPLITS}
    for rule in ("acceptable", "strict"):
        sfx = "" if rule == "acceptable" else "_strict"
        out["by_task_type" + sfx] = {sp: {tt: _acc_block(occ, values, rule, lambda o, sp=sp, tt=tt: _in(sp)(o) and
                                                         o["task_type"] == tt)["acc"] for tt in TASK_TYPES}
                                     for sp in SPLITS}
        out["off_plan" + sfx] = {sp: {k: _acc_block(occ, values, rule, lambda o, sp=sp, k=k: _in(sp)(o) and
                                                    o["off_plan"] == (k == "after"))["acc"] for k in ("on", "after")}
                                 for sp in SPLITS}
    out["missing_pairs"] = missing_pairs(occ, values)
    return out


def live_rows() -> dict[str, Any]:
    """Live success where a row exists (quoted from the rows' summaries, not recomputed)."""
    S = {k: json.loads(p.read_text()) for k, p in LIVE.items() if p.exists()}
    out: dict[str, Any] = {"replace": {}, "cascade": {}, "flash": {}, "by_task_type": {}}
    for (slug, sp), (src, arm) in LIVE_REPLACE.items():
        if src in S:
            out["replace"].setdefault(slug, {})[sp] = {"success": S[src]["arms"][arm]["success"],
                                                      "n": S[src]["arms"][arm]["n"], "row": src.upper()}
            out["by_task_type"].setdefault(slug, {})[sp] = {tt: (S[src]["by_task_type"].get(tt) or {}).get(arm)
                                                           for tt in TASK_TYPES}
    for (slug, sp), (src, arm) in LIVE_CASCADE.items():
        if src in S:
            out["cascade"].setdefault(slug, {})[sp] = {"success": S[src]["arms"][arm]["success"],
                                                      "n": S[src]["arms"][arm]["n"], "row": src.upper()}
    for sp, src in LIVE_FLASH.items():
        if src in S:
            out["flash"][sp] = {"success": S[src]["reference"]["success"], "n": S[src]["reference"]["n"],
                                "row": src.upper()}
    return out


def kendall_block(models: Mapping[str, Any], live: Mapping[str, Any], rule: str = "acceptable") -> dict[str, Any]:
    """Registration (d) 4: (i) across the (model, split) cells with a live REPLACE row; (ii) within a model over its
    (split x task type) cells, and pooled; on the lead label (acceptable) or the strict one."""
    tt_key = "by_task_type" if rule == "acceptable" else "by_task_type_strict"
    cells = []
    for slug, per in (live.get("replace") or {}).items():
        for sp, v in per.items():
            l2 = (((models.get(slug) or {}).get("score") or {}).get(rule) or {}).get(sp, {}).get("acc")
            if l2 is not None and v.get("success") is not None:
                cells.append({"model": slug, "split": sp, "l2": l2, "live": v["success"]})
    out: dict[str, Any] = {"across": {"n": len(cells), "cells": cells,
                                      "tau_b": kendall_tau_b([c["l2"] for c in cells], [c["live"] for c in cells])}}
    within: dict[str, Any] = {}
    pooled_x: list[float] = []
    pooled_y: list[float] = []
    for slug, per in (live.get("by_task_type") or {}).items():
        sc = ((models.get(slug) or {}).get("score") or {}).get(tt_key) or {}
        xs, ys, cl = [], [], []
        for sp, tts in per.items():
            for tt, y in tts.items():
                x = (sc.get(sp) or {}).get(tt)
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
                    cl.append([sp, tt])
        within[slug] = {"n": len(xs), "tau_b": kendall_tau_b(xs, ys), "cells": cl}
        pooled_x += xs
        pooled_y += ys
    within["pooled"] = {"n": len(pooled_x), "tau_b": kendall_tau_b(pooled_x, pooled_y)}
    out["within"] = within
    return out


def box_records(box: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if (box / "host.json").exists():
        out["host"] = json.loads((box / "host.json").read_text().strip().splitlines()[-1])
    if (box / "model_runs.jsonl").exists():
        out["model_runs"] = jl(box / "model_runs.jsonl")
    if (box / "host_hours.jsonl").exists():
        out["host_hours"] = jl(box / "host_hours.jsonl")
    if (box / "ckpt_sha256.txt").exists():
        out["ckpt_sha256"] = (box / "ckpt_sha256.txt").read_text().strip().splitlines()
    return out


def summarize(cdir: Path = ALF_DIR, box: Path = OUT / "box") -> dict[str, Any]:
    c = load_corpus(cdir)
    occ = c["occ"]
    models: dict[str, Any] = {}
    for slug, (name, group, serving) in MODELS.items():
        ap_ = answers_path(cdir, slug)
        if not ap_.exists():
            models[slug] = {"name": name, "group": group, "serving": serving, "run": False}
            continue
        ans = {r["qhash"]: r for r in jl(ap_)}
        vals = {k: answer_value(r) for k, r in ans.items()}
        qset = {o["qhash"] for o in occ}
        ok = [r for k, r in ans.items() if k in qset and not r.get("error")]
        conf = [float(r["confidence"]) for r in ok if r.get("confidence") is not None and not r.get("refusal")]
        models[slug] = {"name": name, "group": group, "serving": serving, "run": True,
                        "n_questions": len(qset), "n_answers": len(qset & set(ans)),
                        "n_errors": sum(1 for k in qset if (ans.get(k) or {}).get("error") or k not in ans),
                        "n_refusals": sum(1 for k in qset if (ans.get(k) or {}).get("refusal")),
                        "n_invalid": sum(1 for k in qset if (ans.get(k) or {}).get("valid") is False
                                         and not (ans.get(k) or {}).get("refusal")),
                        "cost_usd": float(sum(float(r.get("cost_usd") or 0) for r in ans.values())),
                        "mean_conf": float(np.mean(conf)) if conf else None,
                        "latency_ms_median": float(np.median([float(r.get("latency_ms") or 0) for r in ok])) if ok else None,
                        "score": score_row(occ, vals)}
    flash = {"name": "Flash (the agent LLM's own action; in sample)", "score": score_row(occ, lambda o: o.get("flash"))}
    live = live_rows()
    kd = kendall_block(models, live, "acceptable")
    kd_strict = kendall_block(models, live, "strict")
    ran = {k: v for k, v in models.items() if v.get("run")}
    pooled = {k: v["score"]["acceptable"]["pooled"]["acc"] for k, v in ran.items()
              if v["score"]["acceptable"]["pooled"]["acc"] is not None}
    reading: dict[str, Any] = {}
    if pooled:
        best = max(pooled, key=pooled.get)
        worst = min(pooled, key=pooled.get)
        reading = {"spread_pt": 100 * (pooled[best] - pooled[worst]), "best": best, "worst": worst,
                   "ranking": sorted(pooled, key=lambda k: -pooled[k])}
        for grp in ("alfworld", "transfer", "offshelf"):
            g = {k: x for k, x in pooled.items() if MODELS[k][1] == grp}
            if g:
                b = max(g, key=g.get)
                reading[f"best_{grp}"] = {"slug": b, "acc": g[b]}
        if "best_alfworld" in reading:
            for grp in ("transfer", "offshelf"):
                if f"best_{grp}" in reading:
                    reading[f"lead_over_{grp}_pt"] = 100 * (reading["best_alfworld"]["acc"] - reading[f"best_{grp}"]["acc"])
        reading["seen_to_unseen_pt"] = {k: 100 * (v["score"]["acceptable"]["unseen"]["acc"] - v["score"]["acceptable"]["seen"]["acc"])
                                        for k, v in ran.items() if v["score"]["acceptable"]["unseen"]["acc"] is not None
                                        and v["score"]["acceptable"]["seen"]["acc"] is not None}
    spend = {s: float(sum(float(r.get("cost_usd") or 0) for p in Path(OUT).glob(f"*/answers_{s}.jsonl") for r in jl(p)))
             for s in HOSTED}
    tau2 = tau2_block(models)
    return {"row": "E24", "label": LABEL, "generated": _now(), "git": git_head(), "corpus": c["meta"]["stats"],
            "tau2": tau2,
            "corpus_files": c["meta"]["files"], "corpus_sources": c["meta"]["sources"],
            "models": models, "flash": flash, "live": live, "kendall": kd, "kendall_strict": kd_strict,
            "reading": reading, "spend": spend, "set_sizes": set_sizes(occ, cdir),
            "consequence": {"pairs": len(c["cons"]["acc"]), "accepted": sum(c["cons"]["acc"].values()),
                            "states_with_baseline": sum(1 for b in c["cons"]["base"].values() if b.get("h") is not None),
                            "states_evaluated": len(c["cons"]["base"]), "game_errors": len(c["cons"]["errors"])},
            "box": box_records(box)}


def tau2_block(models: Mapping[str, Any], tdir: Path = TAU2_DIR) -> dict[str, Any]:
    """Part 2 readout (registration (c)): every row with answers on the tau^2 corpus, Flash in sample, and Kendall's
    tau-b between L2-ALFWorld (acceptable, pooled) and L2-tau^2 (tool choice, pooled) over the rows with both."""
    if not (Path(tdir) / "occurrences.jsonl.gz").exists():
        return {"run": False}
    occ = jl(Path(tdir) / "occurrences.jsonl.gz")
    meta = json.loads((Path(tdir) / "corpus.json").read_text())
    rows: dict[str, Any] = {}
    for slug, (name, group, _serving) in MODELS.items():
        ap_ = answers_path(tdir, slug)
        if not ap_.exists():
            continue
        ans = {r["qhash"]: r for r in jl(ap_)}
        qset = {o["qhash"] for o in occ} | {a["qhash"] for o in occ for a in o["args"]}
        rows[slug] = {"name": name, "group": group, "score": tau2_score(occ, ans),
                      "n_errors": sum(1 for k in qset if (ans.get(k) or {}).get("error") or k not in ans),
                      "n_refusals": sum(1 for k in qset if (ans.get(k) or {}).get("refusal")),
                      "cost_usd": float(sum(float(r.get("cost_usd") or 0) for r in ans.values()))}
    both = [(models[k]["score"]["acceptable"]["pooled"]["acc"], rows[k]["score"]["pooled"]["tool"]) for k in rows
            if (models.get(k) or {}).get("run") and models[k]["score"]["acceptable"]["pooled"]["acc"] is not None
            and rows[k]["score"]["pooled"]["tool"] is not None]
    return {"run": True, "corpus": meta["stats"], "files": meta["files"], "sources": meta["sources"], "rows": rows,
            "flash": tau2_score(occ, None, flash=True),
            "kendall_alf_tau2": {"n": len(both), "tau_b": kendall_tau_b([x for x, _ in both], [y for _, y in both])}}


def _pct(x: Any, d: int = 1) -> str:
    return "--" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * x:.{d}f}"


def to_markdown(r: Mapping[str, Any]) -> str:
    st = r["corpus"]
    ss = r.get("set_sizes") or {}
    cq = r.get("consequence") or {}
    L = [f"# E24: agent-state accuracy (L2) on public benchmarks -- Part 1, L2-ALFWorld ({LABEL.split(';')[0]})", "",
         f"- **descriptive; no test; revises nothing.** Generated {r['generated']} at git `{(r['git'] or '?')[:12]}` by "
         f"`scripts/explore_e24_public_l2.py`.",
         f"- Corpus: {st['states']} labelled states ({st['unique_questions']} distinct questions) from {st['games']} games of "
         f"Flash alone (valid_seen: E11's run, {st['by_split'].get('seen', {}).get('states')} states; valid_unseen: E23's run, "
         f"{st['by_split'].get('unseen', {}).get('states')} states). Replay check: {st['checks']}; dropped states by reason: "
         f"{st['dropped']}.",
         "- Labels: **acceptable** (lead, amendment) = the lenient set plus every action after which ALFWorld's hand-coded "
         "expert needs fewer steps to win (consequence check by replay); lenient = the strict action plus the expert's equally "
         "valid choices; **strict** = ALFWorld's single expert action (agreement with the expert; a lower bound); logged = the "
         "gold the live runs logged (unpatched expert). Score: game means (every game weighs the same); bootstrap 95 % over games.",
         f"- Consequence check: {cq.get('pairs')} (state, action) pairs evaluated, {cq.get('accepted')} accepted; expert baseline "
         f"on {cq.get('states_with_baseline')} of {cq.get('states_evaluated')} evaluated states; game errors {cq.get('game_errors')}.",
         "- Set sizes on the sampled states (every option evaluated): " + "; ".join(
             f"{sp}: n {v.get('n')}, options {v.get('options', float('nan')):.1f}, lenient {v.get('lenient', float('nan')):.2f}, "
             f"acceptable {v.get('acceptable', float('nan')):.2f} ({_pct(v.get('acceptable_share'))} % of options)"
             for sp, v in ss.items() if v.get("n")), ""]
    L += ["## L2-ALFWorld accuracy %, game means", "",
          "| model | group | acceptable seen | acceptable unseen | acceptable pooled [95 %] | lenient seen / unseen | "
          "strict (expert agreement) seen / unseen | logged seen / unseen | after off-plan / on plan | errors | missing pairs | $ |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(name: str, grp: str, sc: Mapping[str, Any], m: Mapping[str, Any] | None = None) -> str:
        a, le, s_, lg = sc["acceptable"], sc["lenient"], sc["strict"], sc["logged"]
        ci = a["pooled"].get("ci") or [None, None]
        op = sc["off_plan"]["pooled"]
        return (f"| {name} | {grp} | {_pct(a['seen']['acc'])} | {_pct(a['unseen']['acc'])} | {_pct(a['pooled']['acc'])} "
                f"[{_pct(ci[0])}, {_pct(ci[1])}] | {_pct(le['seen']['acc'])} / {_pct(le['unseen']['acc'])} | "
                f"{_pct(s_['seen']['acc'])} / {_pct(s_['unseen']['acc'])} | {_pct(lg['seen']['acc'])} / {_pct(lg['unseen']['acc'])} | "
                f"{_pct(op['after'])} / {_pct(op['on'])} | {(m or {}).get('n_errors', '--')} | {sc.get('missing_pairs', '--')} | "
                f"{('%.3f' % m['cost_usd']) if m and m.get('cost_usd') is not None else '--'} |")

    for slug, m in r["models"].items():
        if m.get("run"):
            L.append(row(m["name"], m["group"], m["score"], m))
        else:
            L.append(f"| {m['name']} | {m['group']} | not run | | | | | | | | | |")
    L.append(row(r["flash"]["name"], "reference", r["flash"]["score"]))
    L += ["", "## By task type (acceptable, pooled seen + unseen)", "",
          "| model | " + " | ".join(TT_SHORT[t] for t in TASK_TYPES) + " |", "|---|" + "---|" * len(TASK_TYPES)]
    for slug, m in list(r["models"].items()) + [("flash", r["flash"])]:
        if slug != "flash" and not m.get("run"):
            continue
        bt = m["score"]["by_task_type"]["pooled"]
        L.append(f"| {m['name']} | " + " | ".join(_pct(bt.get(t)) for t in TASK_TYPES) + " |")
    lv = r["live"]
    L += ["", "## Live (quoted from E11, E23, E6a; not recomputed)", "",
          "| model | split | L2 acceptable | L2 strict | live alone (REPLACE) | live cascade |", "|---|---|---|---|---|---|"]
    for slug, per in lv.get("replace", {}).items():
        for sp, v in per.items():
            sc = ((r["models"].get(slug) or {}).get("score") or {})
            cas = (lv.get("cascade", {}).get(slug) or {}).get(sp, {}).get("success")
            L.append(f"| {MODELS[slug][0]} | {sp} | {_pct((sc.get('acceptable') or {}).get(sp, {}).get('acc'))} | "
                     f"{_pct((sc.get('strict') or {}).get(sp, {}).get('acc'))} | {_pct(v['success'])} ({v['row']}) | {_pct(cas)} |")
    for sp, v in lv.get("flash", {}).items():
        L.append(f"| Flash alone | {sp} | {_pct(r['flash']['score']['acceptable'][sp]['acc'])} (own states) | "
                 f"{_pct(r['flash']['score']['strict'][sp]['acc'])} | {_pct(v['success'])} ({v['row']}) | -- |")
    L += ["", "## Kendall tau-b, L2 vs live alone (descriptive)", ""]
    for lab, kd in (("acceptable", r["kendall"]), ("strict", r["kendall_strict"])):
        L.append(f"- {lab}: across (model, split) cells with a live row: tau-b {('%.2f' % kd['across']['tau_b']) if kd['across']['tau_b'] is not None else '--'} (n = {kd['across']['n']}); "
                 + "; ".join(f"within {k} (split x task type): "
                             f"{('%.2f' % v['tau_b']) if v.get('tau_b') is not None else '--'} (n = {v['n']})"
                             for k, v in kd["within"].items()))
    rd = r.get("reading") or {}
    if rd:
        L += ["", "## Reading rules (registration (d) 5, on the acceptable label)", "",
              f"- spread (pooled, best - worst): {rd['spread_pt']:.1f} points ({rd['best']} vs {rd['worst']}); "
              f"ranking {rd['ranking']}"]
        for k in ("best_alfworld", "best_transfer", "best_offshelf"):
            if k in rd:
                L.append(f"- {k}: {rd[k]['slug']} {_pct(rd[k]['acc'])}")
        for k in ("lead_over_transfer_pt", "lead_over_offshelf_pt"):
            if k in rd:
                L.append(f"- {k}: {rd[k]:+.1f}")
        L.append("- seen-to-unseen change (points): " + ", ".join(f"{k} {v:+.1f}" for k, v in
                                                                 rd.get("seen_to_unseen_pt", {}).items()))
    t2 = r.get("tau2") or {}
    if t2.get("run"):
        cs = t2["corpus"]
        L += ["", "## Part 2: L2-tau^2 (E22's test episodes; write-due states; task means)", "",
              f"- {cs['states']} labelled states of {cs['tasks']} tasks ({cs['tool_questions']} tool questions, "
              f"{cs['arg_questions']} argument questions); by domain {json.dumps(cs['by_domain'])}", "",
              "| model | tool retail | tool airline | tool pooled [95 %] | complete call pooled | wrong write pooled | "
              "arguments pooled | errors | $ |", "|---|---|---|---|---|---|---|---|---|"]
        for slug, m in list(t2["rows"].items()) + [("flash", {"name": "Flash (own action; in sample)", "score": t2["flash"]})]:
            sc = m["score"]
            ci = sc["pooled"].get("tool_ci") or [None, None]
            L.append(f"| {m['name']} | {_pct(sc['retail']['tool'])} | {_pct(sc['airline']['tool'])} | "
                     f"{_pct(sc['pooled']['tool'])} [{_pct(ci[0])}, {_pct(ci[1])}] | {_pct(sc['pooled']['call'])} | "
                     f"{_pct(sc['pooled']['wrong_write'])} | {_pct(sc['pooled']['args'])} | {m.get('n_errors', '--')} | "
                     f"{('%.3f' % m['cost_usd']) if m.get('cost_usd') is not None else '--'} |")
        kd = t2["kendall_alf_tau2"]
        L.append(f"\n- Kendall tau-b, L2-ALFWorld vs L2-tau^2 over the rows with both: "
                 f"{('%.2f' % kd['tau_b']) if kd.get('tau_b') is not None else '--'} (n = {kd['n']})")
    L += ["", "## Spend and host", "", f"- hosted: {r['spend']} (caps: Jev $3, OpenAI $3); Gemini $0.",
          f"- host: {json.dumps(r['box'].get('host'))}; model runs: {json.dumps(r['box'].get('model_runs'))}",
          f"- host hours: {json.dumps(r['box'].get('host_hours'))}"]
    return "\n".join(L) + "\n"


def tex_keys(r: Mapping[str, Any]) -> dict[str, str]:
    v: dict[str, str] = {}

    def put(k: str, x: Any, d: int = 1, scale: float = 100.0) -> None:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return
        v[k] = f"{x * scale:.{d}f}"

    def put_int(k: str, x: Any) -> None:
        if isinstance(x, (int, float)) and math.isfinite(x):
            v[k] = f"{int(x):,}".replace(",", "{,}")

    st = r["corpus"]
    put_int("e24/alf/n/states", st["states"])
    put_int("e24/alf/n/questions", st["unique_questions"])
    put_int("e24/alf/n/games", st["games"])
    for sp in ("seen", "unseen"):
        b = st["by_split"].get(sp) or {}
        put_int(f"e24/alf/n/{sp}/states", b.get("states"))
        put_int(f"e24/alf/n/{sp}/games", b.get("games"))
        put(f"e24/alf/{sp}/off_plan_share", b.get("off_plan_share"))
        put(f"e24/alf/{sp}/options_median", b.get("options_median"), 0, 1.0)
        z = (r.get("set_sizes") or {}).get(sp) or {}
        put(f"e24/alf/{sp}/set/options", z.get("options"), 1, 1.0)
        put(f"e24/alf/{sp}/set/lenient", z.get("lenient"), 1, 1.0)
        put(f"e24/alf/{sp}/set/acceptable", z.get("acceptable"), 1, 1.0)
        put(f"e24/alf/{sp}/set/acceptable_share", z.get("acceptable_share"))
    put_int("e24/alf/n/dropped", sum((st.get("dropped") or {}).values()))
    put_int("e24/alf/n/steps", (st.get("checks") or {}).get("steps"))
    rows = {k: m for k, m in r["models"].items() if m.get("run")}
    rows["flash"] = r["flash"]
    for slug, m in rows.items():
        sc = m["score"]
        for rule in ("acceptable", "lenient", "strict", "logged"):
            for sp in SPLITS:
                b = sc[rule][sp]
                put(f"e24/{slug}/{sp}/{rule}", b.get("acc"))
                if rule == "acceptable" and b.get("ci"):
                    put(f"e24/{slug}/{sp}/acceptable/lo", b["ci"][0])
                    put(f"e24/{slug}/{sp}/acceptable/hi", b["ci"][1])
        for sp in SPLITS:
            for tt in TASK_TYPES:
                put(f"e24/{slug}/{sp}/tt/{TT_SHORT[tt]}", sc["by_task_type"][sp].get(tt))
            put(f"e24/{slug}/{sp}/after_off_plan", sc["off_plan"][sp]["after"])
            put(f"e24/{slug}/{sp}/on_plan", sc["off_plan"][sp]["on"])
        if slug != "flash":
            put_int(f"e24/{slug}/errors", m.get("n_errors"))
            put(f"e24/{slug}/mean_conf", m.get("mean_conf"))
            a = sc["acceptable"]
            if a["seen"]["acc"] is not None and a["unseen"]["acc"] is not None:
                v[f"e24/{slug}/seen_to_unseen"] = f"{100 * (a['unseen']['acc'] - a['seen']['acc']):+.1f}"
    lv = r["live"]
    for kind in ("replace", "cascade"):
        for slug, per in lv.get(kind, {}).items():
            for sp, x in per.items():
                put(f"e24/live/{slug}/{sp}/{kind}", x.get("success"))
    for sp, x in lv.get("flash", {}).items():
        put(f"e24/live/flash/{sp}", x.get("success"))
    for lab, kd in (("acceptable", r["kendall"]), ("strict", r["kendall_strict"])):
        if kd["across"].get("tau_b") is not None:
            v[f"e24/kendall/{lab}/across"] = f"{kd['across']['tau_b']:.2f}"
        put_int(f"e24/kendall/{lab}/across/n", kd["across"]["n"])
        for k, x in kd["within"].items():
            if x.get("tau_b") is not None:
                v[f"e24/kendall/{lab}/within/{k}"] = f"{x['tau_b']:.2f}"
            put_int(f"e24/kendall/{lab}/within/{k}/n", x.get("n"))
    rd = r.get("reading") or {}
    if rd:
        put("e24/reading/spread", rd["spread_pt"], 1, 1.0)
        for k in ("lead_over_transfer_pt", "lead_over_offshelf_pt"):
            if k in rd:
                v[f"e24/reading/{k[:-3]}"] = f"{rd[k]:+.1f}"
    t2 = r.get("tau2") or {}
    if t2.get("run"):
        cs = t2["corpus"]
        put_int("e24/tau2/n/states", cs["states"])
        put_int("e24/tau2/n/tasks", cs["tasks"])
        put_int("e24/tau2/n/questions", cs["questions"])
        for slug, m in list(t2["rows"].items()) + [("flash", {"score": t2["flash"]})]:
            for sp in ("retail", "airline", "pooled"):
                b = m["score"][sp]
                for k in ("tool", "call", "wrong_write", "args"):
                    put(f"e24/{slug}/tau2/{k}/{sp}", b.get(k))
                if b.get("tool_ci"):
                    put(f"e24/{slug}/tau2/tool/{sp}/lo", b["tool_ci"][0])
                    put(f"e24/{slug}/tau2/tool/{sp}/hi", b["tool_ci"][1])
        kd = t2["kendall_alf_tau2"]
        if kd.get("tau_b") is not None:
            v["e24/kendall/alf_tau2"] = f"{kd['tau_b']:.2f}"
        put_int("e24/kendall/alf_tau2/n", kd["n"])
    for s_, x in (r.get("spend") or {}).items():
        if x is not None:
            v[f"e24/spend/{s_}"] = f"{x:.2f}"
    hh = (r.get("box") or {}).get("host_hours") or []
    if hh:
        put("e24/host_hours", hh[-1].get("used_h"), 1, 1.0)
    return v


def to_tex(r: Mapping[str, Any]) -> str:
    lines = ["% AUTO-GENERATED by scripts/explore_e24_public_l2.py -- do not edit by hand.",
             "% E24: exploratory, descriptive row (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-08 'E24').",
             f"% generated {r['generated']} at git {(r['git'] or '?')[:12]}",
             r"\makeatletter",
             r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
             r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    lines += [f"\\defres{{{k}}}{{{x}}}" for k, x in sorted(tex_keys(r).items())]
    return "\n".join(lines + [r"\makeatother", ""])


def cmd_summarize(a: argparse.Namespace) -> int:
    r = summarize(Path(a.corpus), Path(a.box))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (OUT / "summary.md").write_text(to_markdown(r))
    if not a.no_tex:
        TEX.parent.mkdir(parents=True, exist_ok=True)
        TEX.write_text(to_tex(r))
    print(to_markdown(r))
    return 0


# ================================================================================================ main
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build-alfworld")
    p.add_argument("--out", default=str(ALF_DIR))
    p.add_argument("--max-games", type=int, default=None)
    p.set_defaults(fn=cmd_build_alfworld)
    p = sp.add_parser("relabel-chunk")
    p.add_argument("--split", required=True, choices=sorted(ALF_SOURCES))
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_relabel_chunk)
    p = sp.add_parser("build-tau2")
    p.add_argument("--out", default=str(TAU2_DIR))
    p.add_argument("--max-sims", type=int, default=None)
    p.set_defaults(fn=cmd_build_tau2)
    p = sp.add_parser("replay")
    p.add_argument("--corpus", default=str(ALF_DIR))
    p.add_argument("--slug", required=True)
    p.add_argument("--s1", default=None, help="backend JSON (default: configs/explore/e24_public_l2/backends.yaml)")
    p.add_argument("--cache", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cap-usd", type=float, default=None)
    p.set_defaults(fn=cmd_replay)
    p = sp.add_parser("quick")
    p.add_argument("--corpus", default=str(ALF_DIR))
    p.add_argument("--slug", required=True)
    p.set_defaults(fn=cmd_quick)
    p = sp.add_parser("consequence")
    p.add_argument("--corpus", default=str(ALF_DIR))
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--max-games", type=int, default=None)
    p.set_defaults(fn=cmd_consequence)
    p = sp.add_parser("summarize")
    p.add_argument("--corpus", default=str(ALF_DIR))
    p.add_argument("--box", default=str(OUT / "box"))
    p.add_argument("--no-tex", action="store_true")
    p.set_defaults(fn=cmd_summarize)
    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
