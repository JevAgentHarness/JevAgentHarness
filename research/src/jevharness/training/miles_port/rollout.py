"""Miles rollout functions of the decision chain (``--rollout-function-path``) and a stateless data source.

  sft_rollout       data v1 rows (+ the seeded When2Call replay fraction) -> one decision sample per row; the epoch
                    order and every row's option permutation are pure functions of (seed, epoch, row), so a resumed
                    run sees the same stream (``--data-source-path ...NullDataSource``: nothing to save).
  harness_rollout   OPD / GRPO: real ``JevAgent`` episodes on JevGym training seeds (``RolloutCfg``; the task stream
                    is a seeded permutation of the pool, ``tasks_per_iter`` tasks per rollout id, never reused), arms
                    round-robin over the tasks, ``group_size`` episodes per task, System-2 = the configured executor,
                    System-1 = ``SGLangDecider`` on the Miles engine (sampling at ``temperature``).
                      OPD   every visited decision is a candidate; ``fresh_rows`` are asked to the teacher (Jev) at the
                            same state with the same scored pool; the target is the teacher distribution with
                            ``gold_alpha`` of the oracle gold and the gold one-hot where the teacher's argmax is not
                            the gold (``opd.visit_example``); + ``replay_rows`` of the previous iteration; a fresh
                            option permutation per row. Fixed row count per rollout id (shortfalls filled by
                            resampling) so Miles' schedule (``global_batch_size`` rows per step) never trims.
                      GRPO  ``grpo_online.compute_advantages`` (group-normalised episode reward + GiGPO anchors, error
                            and zero-signal episodes dropped) on the configured reward; one sample per trainable
                            decision event, every decision of an episode a sibling of one Miles rollout; an episode
                            without a trainable event contributes one masked placeholder so every rollout id has the
                            same number of rollouts (``tasks_per_iter x group_size``).

Per rollout id one JSON row of statistics goes to ``<out>/rollout_metrics.jsonl`` (and the teacher ledger to
``<out>/teacher_ledger.jsonl``); labelled OPD rows to ``<out>/labelled/iter-XXXXX.jsonl``.
"""
from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import StageConfig, load_stage_config

try:  # Miles is only present in its image; the tests use the light stand-ins below
    from miles.rollout.data_source import DataSource as _DataSourceBase  # type: ignore
except Exception:  # noqa: BLE001
    _DataSourceBase = object


# =============================================================================== samples
@dataclass
class SampleLite:
    """The fields of ``miles.utils.types.Sample`` the chain sets (used when Miles is not importable)."""
    group_index: int | None = None
    index: int | None = None
    rollout_id: int | None = None
    prompt: str = ""
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    reward: float | None = None
    loss_mask: list[int] | None = None
    train_metadata: dict | None = None
    metadata: dict = field(default_factory=dict)
    status: str = "completed"


def make_sample(prompt_ids: list[int], response_token: int, meta: dict[str, Any], *, index: int, group_index: int,
                rollout_id: int, reward: float = 0.0, train: bool = True) -> Any:
    tokens = list(map(int, prompt_ids)) + [int(response_token)]
    try:
        from miles.utils.types import Sample  # type: ignore

        s = Sample(group_index=group_index, index=index, rollout_id=rollout_id, prompt="", tokens=tokens,
                   response="", response_length=1, reward=float(reward), loss_mask=[1 if train else 0],
                   train_metadata=meta)
        s.status = Sample.Status.COMPLETED
        return s
    except ImportError:
        return SampleLite(group_index=group_index, index=index, rollout_id=rollout_id, tokens=tokens,
                          response_length=1, reward=float(reward), loss_mask=[1 if train else 0], train_metadata=meta)


class NullDataSource(_DataSourceBase):  # type: ignore[misc, valid-type]
    """The rollout functions derive their data from the rollout id; Miles' data-source calls are no-ops."""

    def __init__(self, args: Any = None):
        self.args = args

    def get_samples(self, num_samples: int) -> list:
        return []

    def add_samples(self, samples: Any) -> None:
        return None

    def save(self, rollout_id: Any) -> None:
        return None

    def load(self, rollout_id: Any = None) -> None:
        return None

    def get_buffer_length(self) -> int | None:
        return None


def _out(cfg: StageConfig) -> Path:
    p = Path(os.environ.get("JEV_MILES_OUT") or cfg.out)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _log(cfg: StageConfig, name: str, row: dict[str, Any]) -> None:
    def clean(v: Any) -> Any:
        return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
    with open(_out(cfg) / name, "a") as f:
        f.write(json.dumps({k: clean(v) for k, v in row.items()}, default=str) + "\n")


def load_tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


# =============================================================================== SFT
_SFT: dict[str, Any] = {}
_LOCK = threading.Lock()


def sft_rows(cfg: StageConfig) -> list[dict[str, Any]]:
    from ..data import read_jsonl

    assert cfg.sft is not None
    rows: list[dict[str, Any]] = []
    for f in cfg.sft.train:
        rows += read_jsonl(f)
    if cfg.sft.replay:
        rep = read_jsonl(cfg.sft.replay)
        k = int(round(cfg.sft.replay_frac * len(rep)))
        rows += random.Random(cfg.seed).sample(rep, k)
    return [r for r in rows if 2 <= len(r["options"]) <= 52]


def sft_num_rollouts(n_rows: int, epochs: int, batch_rows: int) -> int:
    return math.ceil(n_rows * epochs / batch_rows)


def _epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    idx = list(range(n))
    random.Random(seed * 1000003 + epoch).shuffle(idx)
    return idx


def sft_batch(cfg: StageConfig, collator: Any, rows: list[dict[str, Any]], rollout_id: int) -> list[list[Any]]:
    from ..data import permuted

    assert cfg.sft is not None
    n, B = len(rows), cfg.sft.batch_rows
    total = n * cfg.sft.epochs
    orders: dict[int, list[int]] = {}
    groups = []
    for j in range(B):
        p = (rollout_id * B + j) % total
        e, i = divmod(p, n)
        order = orders.setdefault(e, _epoch_order(n, cfg.seed, e))
        ridx = order[i]
        row = rows[ridx]
        rng = random.Random(f"{cfg.seed}-{e}-{ridx}") if cfg.sft.permute else None
        prompt, tgt, _ = permuted(row, rng)
        ids = collator.encode(prompt)
        codes = collator.codes[:len(tgt)]
        meta = {"stage": "sft", "codes": codes, "target": tgt, "brier": cfg.sft.brier_weight}
        k = rollout_id * B + j
        best = max(range(len(tgt)), key=lambda a: tgt[a])
        groups.append([make_sample(ids, codes[best], meta, index=k, group_index=k, rollout_id=k)])
    return groups


def sft_rollout(args: Any, rollout_id: int, data_source: Any = None, evaluation: bool = False) -> list[list[Any]]:
    if evaluation:
        return {}  # type: ignore[return-value]  # no eval rollouts (gates run on the exported checkpoints)
    cfg = load_stage_config()
    with _LOCK:
        if "rows" not in _SFT:
            from ..decision_model import DecisionCollator

            tok = load_tokenizer(cfg.init or cfg.base)
            _SFT["collator"] = DecisionCollator(tok, max_len=cfg.max_len)
            _SFT["rows"] = sft_rows(cfg)
            _log(cfg, "rollout_metrics.jsonl", {"event": "sft_data", "rows": len(_SFT["rows"]),
                                                "num_rollouts": sft_num_rollouts(len(_SFT["rows"]), cfg.sft.epochs,
                                                                                 cfg.sft.batch_rows)})
    return sft_batch(cfg, _SFT["collator"], _SFT["rows"], rollout_id)


# =============================================================================== harness rollouts (OPD / GRPO)
class Harness:
    """Everything a rollout id needs, built once per rollout-manager process."""

    def __init__(self, cfg: StageConfig, url: str, client: Any = None, executor: Any = None):
        from ...bench.suites import load_suite
        from ...harness.config import HarnessConfig
        from ..grpo_online import RewardCfg
        from .sglang_decider import OPDSGLangDecider, SGLangDecider

        assert cfg.rollout is not None
        r = cfg.rollout
        self.cfg = cfg
        tok = load_tokenizer(cfg.init or cfg.base)
        self.arms = [HarnessConfig(**{"max_steps": r.max_steps, **{k: v for k, v in a.items() if k != "llm"}})
                     .validate_modes() for a in r.arms]
        for arm in self.arms:
            if not arm.uses_s1():
                raise ValueError(f"arm {arm.name!r} routes no decision point to System-1")
        self.arm_llm = {a.get("name"): a.get("llm") for a in r.arms}
        cls = OPDSGLangDecider if cfg.stage == "opd" else SGLangDecider
        self.decider = cls(tok, url, self.arms[0], client=client, temperature=max(r.temperature, 1e-6),
                           greedy=r.temperature <= 0, max_len=cfg.max_len, batch_size=r.decider_batch,
                           name=f"jevagentrl-v2-{cfg.stage}")
        self.collator = self.decider.collator
        if executor is not None:
            self.executor, self.named = executor, {k: executor for k in r.executors}
        else:
            from ...llm import make_llm

            self.executor = make_llm(r.executor)
            self.named = {k: make_llm(v) for k, v in r.executors.items()}
        self.tasks = load_suite("jevgym", n=r.n_tasks, seed0=r.seed0, **r.suite)
        self.order = list(range(len(self.tasks)))
        random.Random(cfg.seed).shuffle(self.order)
        self.reward_cfg = cfg.grpo.reward if cfg.grpo is not None else RewardCfg()
        self.replay: list[dict[str, Any]] = []
        self.teacher = None
        self.expose_gold = False
        if cfg.stage == "opd":
            from ...accounting import Ledger
            from ..teacher_label import SIMULATED_TYPES, make_teacher, parse_teacher

            assert cfg.opd is not None
            _, spec = parse_teacher(cfg.opd.teacher, 0)
            self.teacher = make_teacher(spec, cfg.opd.teacher_cache)
            self.expose_gold = spec["type"] in SIMULATED_TYPES
            self.ledger = Ledger()
            self.teacher_cost_prev = self._ledger_total_from_log()

    def _ledger_total_from_log(self) -> float:
        p = _out(self.cfg) / "teacher_ledger.jsonl"
        if not p.exists():
            return 0.0
        tot = 0.0
        for line in p.read_text().splitlines():
            try:
                tot = max(tot, float(json.loads(line).get("teacher_cost_total", 0.0)))
            except (ValueError, json.JSONDecodeError):
                continue
        return tot

    def make_agent(self, arm: Any, s1: Any) -> Any:
        from ...harness.agent import JevAgent

        name = self.arm_llm.get(arm.name)
        ex = self.named[name] if name else self.executor
        assert self.cfg.rollout is not None
        return JevAgent(ex, arm, s1=s1, system_prompt=self.cfg.rollout.system_prompt)

    def task_ids(self, rollout_id: int) -> list[int]:
        assert self.cfg.rollout is not None
        k = self.cfg.rollout.tasks_per_iter
        return [self.order[(rollout_id * k + j) % len(self.order)] for j in range(k)]

    def run(self, rollout_id: int) -> tuple[list[Any], dict[str, Any]]:
        from ..grpo_online import rollout_stats, run_episodes

        assert self.cfg.rollout is not None
        r = self.cfg.rollout
        picked = self.task_ids(rollout_id)
        jobs = [(gi, self.tasks[ti], g, self.arms[gi % len(self.arms)]) for gi, ti in enumerate(picked)
                for g in range(r.group_size)]
        t0 = time.perf_counter()
        episodes = run_episodes(jobs, self.decider, self.make_agent, self.reward_cfg, r.workers,
                                (self.cfg.seed, rollout_id))
        row: dict[str, Any] = {"rollout_id": rollout_id, "tasks": len(picked), "task_seeds": [r.seed0 + t for t in picked],
                               **rollout_stats(episodes), "time_rollout_s": time.perf_counter() - t0,
                               "sglang_requests": getattr(self.decider.client, "requests", None)}
        for arm in self.arms:
            ea = [e for e in episodes if e.arm == arm.name]
            if ea:
                sa = rollout_stats(ea)
                row.update({f"arm/{arm.name}/{k}": sa[k] for k in ("success", "dec_acc", "dec_acc_greedy",
                                                                    "reward_mean", "llm_calls")})
        errors = [e.error for e in episodes if e.error]
        if errors:
            row["first_error"] = str(errors[0])[:500]
            if len(errors) == len(episodes):
                raise RuntimeError(f"every episode of rollout {rollout_id} failed; first error:\n{errors[0]}")
        return episodes, row


_H: dict[str, Harness] = {}


def _harness(args: Any, cfg: StageConfig) -> Harness:
    with _LOCK:
        if "h" not in _H:
            url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
            _H["h"] = Harness(cfg, url)
        return _H["h"]


def opd_samples(h: Harness, episodes: list[Any], rollout_id: int, row: dict[str, Any]) -> list[list[Any]]:
    from ..data import permuted, write_jsonl
    from ..opd import label_stats, label_visits, visit_example
    from ..teacher_label import ledger_summary

    cfg = h.cfg
    assert cfg.opd is not None and h.teacher is not None
    o = cfg.opd
    rng = random.Random(f"{cfg.seed}-opd-{rollout_id}")
    visits = []
    for e in episodes:
        for v in e.records:
            v.task_id = e.task_id
            visits.append(v)
    asked = rng.sample(visits, o.fresh_rows) if len(visits) > o.fresh_rows else list(visits)
    spent = h.teacher_cost_prev + float(ledger_summary(h.ledger)["cost_usd"])
    reasons: Counter = Counter()
    n0 = len(h.ledger.records)
    t1 = time.perf_counter()
    if o.budget_usd > 0 and spent >= o.budget_usd:
        reasons["budget"] = len(asked)
        asked = []
    elif asked:
        reasons = label_visits(asked, h.teacher, h.ledger, expose_gold=h.expose_gold, workers=o.teacher_workers,
                               per_request=o.questions_per_request)
    fresh_rec = h.ledger.records[n0:]
    fresh = [visit_example(v, rollout_id, k, gold_alpha=o.gold_alpha, teacher_temperature=o.teacher_temperature,
                           gold_fallback=o.gold_fallback) for k, v in enumerate(asked) if v.teacher is not None]
    total = h.teacher_cost_prev + float(ledger_summary(h.ledger)["cost_usd"])
    row.update({"visits": len(visits), "asked": len(asked), **label_stats(asked), "teacher_skipped": dict(reasons),
                "teacher_calls_iter": len(fresh_rec), "teacher_cost_iter": float(sum(x.cost_usd for x in fresh_rec)),
                "teacher_cost_total": total, "time_label_s": time.perf_counter() - t1,
                "gold_fallback_n": sum(x["meta"]["label_source"] == "gold_fallback" for x in fresh)})
    _log(cfg, "teacher_ledger.jsonl", {"rollout_id": rollout_id, "teacher_cost_total": total,
                                       **{k: v for k, v in ledger_summary(h.ledger).items()
                                          if k != "unpriced_models"}})
    lab_dir = _out(cfg) / "labelled"
    lab_dir.mkdir(exist_ok=True)
    write_jsonl(fresh, lab_dir / f"iter-{rollout_id:05d}.jsonl")
    # fixed row count: fresh (up to fresh_rows), replay (up to replay_rows), shortfall resampled from what exists
    want = o.fresh_rows + o.replay_rows
    replay = rng.sample(h.replay, min(len(h.replay), o.replay_rows)) if h.replay else []
    batch = fresh[:o.fresh_rows] + replay
    pool = fresh + h.replay
    if not pool:
        raise RuntimeError(f"OPD rollout {rollout_id}: no labelled row and no replay (teacher skipped: {dict(reasons)})")
    while len(batch) < want:
        batch.append(rng.choice(pool))
    row.update({"fresh_n": min(len(fresh), o.fresh_rows), "replay_n": len(replay), "train_n": len(batch),
                "resampled_n": want - min(len(fresh), o.fresh_rows) - len(replay)})
    h.replay = fresh if fresh else h.replay
    groups = []
    base = rollout_id * want
    for j, ex in enumerate(batch):
        prompt, tgt, _ = permuted(ex, rng)
        ids = h.collator.encode(prompt)
        codes = h.collator.codes[:len(tgt)]
        meta = {"stage": "opd", "codes": codes, "target": tgt, "loss": o.loss}
        best = max(range(len(tgt)), key=lambda a: tgt[a])
        groups.append([make_sample(ids, codes[best], meta, index=base + j, group_index=base + j,
                                   rollout_id=base + j)])
    return groups


def grpo_samples(h: Harness, episodes: list[Any], rollout_id: int, row: dict[str, Any]) -> list[list[Any]]:
    from ..grpo_online import compute_advantages

    cfg = h.cfg
    assert cfg.grpo is not None and cfg.rollout is not None
    A = cfg.grpo.algo
    row.update(compute_advantages(episodes, A))
    T = max(cfg.rollout.temperature, 1e-6)
    per_task = cfg.rollout.group_size
    groups: list[list[Any]] = []
    idx = 0
    n_train = 0
    base = rollout_id * len(episodes)
    for ei, e in enumerate(episodes):
        rid = base + ei
        gi = rollout_id * cfg.rollout.tasks_per_iter + ei // per_task
        sibs = []
        for rec in e.records:
            if not rec.train:
                continue
            ids = h.collator.encode(rec.prompt)
            codes = h.collator.codes[:rec.n]
            meta = {"stage": "grpo", "codes": codes, "event": list(rec.event), "ev": rec.ev,
                    "old_logp": float(rec.old_logp), "adv": float(rec.adv), "temperature": T,
                    "clip": A.clip_eps, "clip_high": A.clip_eps_high if A.clip_eps_high is not None else A.clip_eps,
                    "dp": rec.dp}
            sibs.append(make_sample(ids, codes[rec.sampled], meta, index=rollout_id * 1_000_000 + idx, group_index=gi,
                                    rollout_id=rid, reward=e.reward))
            idx += 1
            n_train += 1
        if not sibs:  # no trainable event: one masked placeholder keeps the episode's rollout slot
            codes = h.collator.codes[:2]
            meta = {"stage": "grpo", "codes": codes, "event": [0], "ev": "choice", "old_logp": math.log(0.5),
                    "adv": 0.0, "temperature": 1.0, "clip": A.clip_eps, "clip_high": A.clip_eps, "dp": None}
            ids = h.collator.encode("placeholder")[-8:]
            sibs.append(make_sample(ids, codes[0], meta, index=rollout_id * 1_000_000 + idx, group_index=gi,
                                    rollout_id=rid,
                                    reward=e.reward, train=False))
            idx += 1
        groups.append(sibs)
    row.update({"samples_train": n_train, "placeholders": sum(1 for g in groups if not g[0].loss_mask[0])})
    return groups


def harness_rollout(args: Any, rollout_id: int, data_source: Any = None, evaluation: bool = False) -> Any:
    if evaluation:
        return {}
    cfg = load_stage_config()
    h = _harness(args, cfg)
    episodes, row = h.run(rollout_id)
    groups = opd_samples(h, episodes, rollout_id, row) if cfg.stage == "opd" else grpo_samples(h, episodes,
                                                                                               rollout_id, row)
    _log(cfg, "rollout_metrics.jsonl", row)
    if cfg.stage == "grpo":
        with open(_out(cfg) / "rollouts.jsonl", "a") as f:
            for e in episodes:
                f.write(json.dumps({"rollout_id": rollout_id, "task_id": e.task_id, "arm": e.arm, "reward": e.reward,
                                    "success": e.success, "llm_calls": e.llm_calls, "steps": e.n_steps,
                                    "decisions": len(e.records), "adv": e.adv, "error": e.error}) + "\n")
    return groups
