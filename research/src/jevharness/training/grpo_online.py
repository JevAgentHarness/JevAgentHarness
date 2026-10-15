"""Episode-level GRPO for a System-1 decision model *inside* the agent harness (recipe R8).

The trainable decider runs in-process as a ``DecisionBackend`` (:class:`TorchDecider`). Every harness
question is rendered exactly as ``serve_jev.py`` renders the corresponding Jev wire question
(``question_to_wire`` -> options), scored with ONE prefill (restricted softmax over the option-code
tokens, ``decision_model.option_logits``), and an option is *sampled* at temperature T (training) or
taken greedily (evaluation). A policy trained here is therefore drop-in for ``serve_jev.py``.

Every sampled decision is recorded with the log-probability of the *event it caused in the harness*:

  replace           the sampled option a                                  log pi(a)
  cascade           commit a       (pi_1(a) >= tau and a != OTHER)          log pi(a)
                    defer          (sampled option fails the gate)         log sum_{a' in D} pi(a')
                                   D = every option that would have deferred: the deferral is an action
                                   of its own -- its LLM cost flows into the reward, its credit into log P(D)
  assist (top-k)    ordered top-k subset, Plackett-Luce / Gumbel-top-k     sum_i log pi(s_i)/(1-sum_{j<i} pi(s_j))
  Noul gates        sampled yes/no (guard / trigger / stop / memory / routing); the harness gate follows
                    the sample during training, and p_yes >= tau at evaluation

Per iteration: sample ``tasks_per_iter`` tasks, run G rollouts of each (same executor LLM, JevGym
strict mode supported), reward every episode

  linear  R = 1[success] - lambda_cost * cost_usd - lambda_calls * n_llm_calls
  gated   R = 1[success] * (1 - lambda_gate * min(1, cost / cost_scale))   (failing cheaply earns nothing)

(minus ``irreversible_penalty`` when the env reports an irreversible error), then

  A_E = (R - mean_G) / (std_G + eps)                  GRPO group normalisation over the G rollouts
  A_S = (g_t - mean_s) / (std_s + eps)                optional GiGPO step level: decisions whose prompt is
                                                      identical across the group's rollouts (anchor states),
                                                      g_t = gamma^(T-1-t) R
  A   = A_E + omega * A_S                             broadcast to every sampled decision of the episode

  L = mean_d [ max(-A r_d, -A clip(r_d, 1-eps, 1+eps_high)) + beta KL(pi_theta || pi_ref)(s_d) - c_H H(s_d) ]
      r_d = exp(log P_theta(event_d) - log P_old(event_d)),  mu inner epochs over the rollout buffer,
      exact KL over the option set; reference = LoRA adapter disabled (``ref: base``) or a frozen copy of
      the ``--init`` adapter (``ref: init``; default when ``init`` is set).

Which decisions are updated: records of episodes that raised an exception (``drop_error_episodes``) get no credit
and no blame; records with no signal at either level -- identical rewards in their task group AND (with GiGPO) no
return spread in their anchor group -- are dropped (``drop_zero_std_groups``, DAPO-style dynamic sampling). GiGPO
thus still learns "reach success in fewer steps" from groups whose episodes all succeeded.

Practical notes
  * cascade needs a warm start: a decision whose options all sit below tau defers with probability 1, which carries
    no gradient (``cascade_forced_defer_frac`` in the metrics). Start from an SFT adapter (``init``) or train
    replace / assist / gate arms first; a warning is printed when an iteration has no learning signal.
  * the simulated executor seeds its RNG per task, so the G rollouts of a task share executor noise (common random
    numbers): the group advantage then isolates the decider's choices. Real executors sample (temperature > 0).
  * decision accuracy vs the JevGym oracle (``dec_acc``, per decision point) is logged but never rewarded.

  python -m jevharness.training.grpo_online --config configs/train/grpo_online_jevgym.yaml \\
      --set model=Qwen/Qwen3-4B init=checkpoints/r4 out=checkpoints/r8

Outputs (``out``): ``metrics.jsonl`` (one row per iteration: reward, success, cost, calls, deferral,
decision accuracy vs gold, KL, entropy, clip fraction, ... and ``eval/<arm>/*`` greedy evaluations),
``ckpt-XXXXX/`` adapters (+ optimizer state for ``--resume``), and the final adapter in ``out`` itself
(``serve_jev.py --adapter out`` / ``calibrate.py --adapter out`` work unchanged).

Scaling (single process, no accelerate, no vLLM needed: the decider never generates):
  * rollouts run in ``workers`` threads; their System-1 prefills are micro-batched across threads into
    one forward pass (``decider.batch_size`` / ``decider.max_wait_ms``), so GPU utilisation grows with the
    number of concurrent episodes; executor calls (API or a separately hosted vLLM) overlap with them;
  * the update is a plain batched forward/backward over the decision buffer (``minibatch_size`` per
    optimizer step, ``micro_batch_size`` per forward, optional gradient checkpointing for long states);
  * more GPUs: run independent seeds / ablations one per GPU (``scripts/train/70_grpo_online.sh``).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import queue
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..accounting import default_prices
from ..decision.base import DecisionBackend
from ..decision.jev import question_to_wire
from ..decision.types import JEV_MAX_OPTIONS, OPTION_LABELS, Answer, DecisionResult, QKind, Question, Usage
from ..harness.config import DP, HarnessConfig, Mode
from ..harness.tools import OTHER
from .data import render
from .decision_model import reference_context
from .decision_model import use_adapter as _adapter  # noqa: F401  (re-exported for callers)

MAX_PASS = len(OPTION_LABELS)  # options scored in one prefill; more -> knockout (as serve_jev.py)


# =============================================================================== config
class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoraCfg(_Cfg):
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    target_modules: str | list[str] = "all-linear"


class DeciderCfg(_Cfg):
    temperature: float = 1.0          # sampling temperature T during rollouts (KL/confidence use T=1)
    max_len: int = 4096               # prompt tokens kept (tail), same convention as train_decision
    batch_size: int = 16              # prompts per forward pass (rollout inference, reference pass)
    max_wait_ms: float = 5.0          # cross-thread micro-batching window during rollouts
    price_model: str | None = None    # prices-file key used to cost System-1 input tokens (e.g. Qwen/Qwen3-4B)
    name: str = "grpo-policy"


class GigpoCfg(_Cfg):
    enabled: bool = False
    omega: float = 1.0                # weight of the step-level advantage
    gamma: float = 0.95               # discount for step returns g_t = gamma^(T-1-t) R
    scale: Literal["std", "none"] = "std"


class AlgoCfg(_Cfg):
    iterations: int = 100
    tasks_per_iter: int = 8
    group_size: int = 8               # G rollouts per task
    inner_epochs: int = 1             # mu passes over the rollout buffer
    minibatch_size: int = 64          # decisions per optimizer step
    micro_batch_size: int = 8         # decisions per forward/backward
    max_batch_tokens: int = 0         # > 0: length-sorted micro-batches of <= this many padded tokens instead
    clip_eps: float = 0.2
    clip_eps_high: float | None = 0.28   # DAPO clip-higher; None -> symmetric
    kl_beta: float = 0.02
    entropy_coef: float = 0.0
    adv_scale: Literal["group", "batch", "none"] = "group"
    adv_eps: float = 1e-6
    drop_zero_std_groups: bool = True   # groups with identical rewards carry no signal (DAPO dynamic sampling)
    drop_error_episodes: bool = True    # harness/executor exceptions are not the decider's fault: no credit, no blame
    loss_agg: Literal["decision", "episode"] = "decision"
    gigpo: GigpoCfg = Field(default_factory=GigpoCfg)


class RewardCfg(_Cfg):
    kind: Literal["linear", "gated"] = "gated"
    lambda_cost: float = 0.0          # linear: per USD of episode cost (System-1 + System-2)
    lambda_calls: float = 0.0         # linear: per System-2 (LLM) call
    lambda_gate: float = 0.5          # gated: success * (1 - lambda_gate * cost_norm)
    cost_unit: Literal["usd", "calls"] = "calls"
    cost_scale: float = 10.0          # gated: cost_norm = min(1, cost / cost_scale) in cost_unit
    irreversible_penalty: float = 0.0


class OptimCfg(_Cfg):
    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_steps: int = 0             # linear warmup (optimizer steps), then constant
    betas: tuple[float, float] = (0.9, 0.999)


class Config(_Cfg):
    name: str = "grpo_online"
    model: str
    init: str | None = None           # SFT adapter to start from (e.g. checkpoints/r3)
    ref: Literal["auto", "base", "init"] = "auto"   # auto: init if init else base (adapter disabled)
    out: str = "checkpoints/grpo_online"
    lora: LoraCfg = Field(default_factory=LoraCfg)
    suite: dict[str, Any] = Field(default_factory=lambda: {"name": "jevgym", "n": 64, "n_distractors": 10,
                                                           "seed0": 1000})
    eval_suite: dict[str, Any] | None = Field(default_factory=lambda: {"name": "jevgym", "n": 16,
                                                                       "n_distractors": 10, "seed0": 5000})  # A6.4: calibration/dev range, never 0..999 or 100000+
    # seed rules (training/seed_guard.py). `registered`: the defaults above (A6.4); only the never-run ranges
    # (evaluation, retired, Stage-2 300000-300999) are refused. `strict` (JevAgentRL, round 5): the suite must be
    # training seeds (>= 10000 minus every reserved range) and the eval suite development seeds (0-999 / 1000-4999),
    # disjoint from the training pool; the calibration seeds 5000-5149 stay for the tau-hat* fit of a frozen model.
    seed_policy: Literal["registered", "strict"] = "registered"
    executor: dict[str, Any] = Field(default_factory=lambda: {"provider": "simulated", "model": "scripted",
                                                              "error_rate": 0.1})
    executors: dict[str, dict[str, Any]] = Field(default_factory=dict)   # named extra System-2s: arm key `llm: <name>`
    eval_arms: list[str] | None = None   # arm names evaluated greedily (None = all)
    small_executor: dict[str, Any] | None = None   # only for routing=trigger arms
    harness_defaults: dict[str, Any] = Field(default_factory=dict)
    arms: list[dict[str, Any]] = Field(default_factory=lambda: [
        {"name": "ts_cascade", "tool_select": {"mode": "cascade", "tau": 0.8}}])
    system_prompt: str | None = None
    decider: DeciderCfg = Field(default_factory=DeciderCfg)
    grpo: AlgoCfg = Field(default_factory=AlgoCfg)
    reward: RewardCfg = Field(default_factory=RewardCfg)
    optim: OptimCfg = Field(default_factory=OptimCfg)
    workers: int = 8
    eval_every: int = 10              # iterations; 0 = never (an eval also runs before training and at the end)
    save_every: int = 25
    seed: int = 0
    device: str | None = None
    bf16: bool | None = None          # default: True on CUDA
    gradient_checkpointing: bool = False
    log_rollouts: bool = False        # also write rollouts.jsonl (one row per episode)

    def harness_arms(self) -> list[HarnessConfig]:
        arms = [HarnessConfig(**{**self.harness_defaults, **{k: v for k, v in a.items() if k != "llm"}}).validate_modes()
                for a in self.arms]
        if not arms:
            raise ValueError("config needs at least one arm")
        for a in arms:
            if not a.uses_s1():
                raise ValueError(f"arm {a.name!r} routes no decision point to System-1: nothing to train")
        for a in self.arms:
            if a.get("llm") is not None and a["llm"] not in self.executors:
                raise ValueError(f"arm {a.get('name')!r}: llm {a['llm']!r} is not a key of `executors`")
        unknown = sorted(set(self.eval_arms or []) - {a.name for a in arms})
        if unknown:
            raise ValueError(f"eval_arms {unknown}: not among the arms {[a.name for a in arms]}")
        return arms

    def check_seeds(self) -> None:
        """training/seed_guard.py. Always: no suite touches the evaluation / retired / Stage-2 ranges. With
        ``seed_policy: strict``: the training suite draws training seeds only and the greedy evaluation suite
        development seeds only, disjoint from the training pool."""
        from ..bench.registry import overlaps
        from .seed_guard import NEVER_RUN, check_dev_pool, check_disjoint, check_train_pool

        suites = [("suite", self.suite)] + ([("eval_suite", self.eval_suite)] if self.eval_suite else [])
        for what, su in suites:
            if su.get("name", "jevgym") == "jevgym":
                hit = overlaps(int(su.get("seed0", 0) or 0), int(su.get("n", 0) or 0), NEVER_RUN)
                if hit:
                    raise ValueError(f"{what}: seeds overlap the never-run range(s) {hit} (evaluation / retired / "
                                     "Stage-2)")
        if self.seed_policy != "strict" or self.suite.get("name", "jevgym") != "jevgym":
            return
        s0, n = int(self.suite.get("seed0", 0)), int(self.suite.get("n", 0))
        check_train_pool(s0, n, "suite")
        if self.eval_suite and self.eval_suite.get("name", "jevgym") == "jevgym":
            e0, en = int(self.eval_suite.get("seed0", 0)), int(self.eval_suite.get("n", 0))
            check_dev_pool(e0, en, "eval_suite")
            check_disjoint((e0, en), (s0, n), "eval_suite vs suite")


def _set_dotted(d: dict[str, Any], key: str, value: Any) -> None:
    node = d
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def _seed_dict_default(raw: dict[str, Any], key: str) -> None:
    """Before a dotted override into a plain-dict field (suite, executor, ...) that the YAML does not set,
    copy the field's default in, so ``suite.n=4`` edits the default suite instead of replacing it."""
    top = key.split(".", 1)[0]
    if "." not in key or top in raw or top not in Config.model_fields:
        return
    d = Config.model_fields[top].get_default(call_default_factory=True)
    if isinstance(d, dict):
        raw[top] = copy.deepcopy(d)


def load_config(path: str | Path | None = None, sets: Sequence[str] = (), **overrides: Any) -> Config:
    """YAML file + ``dotted.key=value`` overrides (values parsed as YAML) + keyword overrides."""
    raw: dict[str, Any] = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    for kv in sets:
        if "=" not in kv:
            raise ValueError(f"override {kv!r} is not KEY=VALUE")
        k, v = kv.split("=", 1)
        _seed_dict_default(raw, k)
        _set_dotted(raw, k, yaml.safe_load(v))
    for k, v in overrides.items():
        if v is not None:
            _seed_dict_default(raw, k)
            _set_dotted(raw, k, v)
    return Config.model_validate(raw)


# =============================================================================== decisions
def question_to_example(state: str, q: Question) -> dict[str, Any]:
    """Harness question -> training-format decision (state, question, options) + answer labels.

    Mirrors ``serve_jev.Engine.answer`` applied to ``question_to_wire(q)``, so in-process scoring and
    scoring behind the Jev API see byte-identical prompts."""
    w = question_to_wire(q)
    instr = w.get("instructions") or ""
    if not isinstance(instr, str):
        instr = json.dumps(instr)
    crit = w.get("criteria")
    if w["type"] == "choice":
        labels = list(crit.keys())
        opts = [lab if not crit[lab] else
                f"{lab} -- {crit[lab] if isinstance(crit[lab], str) else json.dumps(crit[lab])}" for lab in labels]
        kind = "choice"
    elif w["type"] == "noul":
        crit = crit or {}
        opts = ["yes" + (f" -- {crit['true']}" if crit.get("true") else ""),
                "no" + (f" -- {crit['false']}" if crit.get("false") else "")]
        labels = ["yes", "no"]
        kind = "noul"
    else:
        levels = [c if isinstance(c, str) else json.dumps(c) for c in crit]
        opts = [f"level {k}: {d}" for k, d in enumerate(levels)]
        labels = q.labels()
        kind = "score"
    return {"state": state, "question": instr, "options": opts, "labels": labels, "kind": kind}


def _gold_label(q: Question) -> str | None:
    g = q.meta.get("gold")
    if g is None:
        return None
    if isinstance(g, bool):
        return "yes" if g else "no"
    return str(g)


def _sample(p: Sequence[float], rng: random.Random) -> int:
    u, c = rng.random(), 0.0
    for i, v in enumerate(p):
        c += v
        if u < c:
            return i
    return max(range(len(p)), key=lambda i: p[i])  # numerical slack


def _gumbel_topk(logp: Sequence[float], k: int, rng: random.Random) -> list[int]:
    """k options without replacement from softmax(logp) (Plackett-Luce), in sampled order."""
    keys = [lp - math.log(-math.log(min(max(rng.random(), 1e-300), 1 - 1e-16))) for lp in logp]
    return sorted(range(len(keys)), key=lambda i: -keys[i])[:k]


def event_logp(logp: torch.Tensor, event: Sequence[int], ev: str) -> torch.Tensor:
    """log P(event) for one decision; ``logp`` = log-probs over options (1-D, may contain -inf padding)."""
    if ev == "subset":  # Plackett-Luce: sum_i log p(s_i) - log sum_{a not in s_<i} p(a)
        total = logp.new_zeros(())
        keep = torch.ones_like(logp, dtype=torch.bool)
        for i in event:
            total = total + logp[i] - torch.logsumexp(logp.masked_fill(~keep, float("-inf")), -1)
            keep = keep.clone()
            keep[i] = False
        return total
    idx = torch.as_tensor(list(event), dtype=torch.long, device=logp.device)
    return torch.logsumexp(logp.index_select(0, idx), -1)


@dataclass
class DecisionRecord:
    prompt: str                   # user prompt (training/data.render); the collator applies the chat template
    n: int                        # number of options in the scored pass
    event: list[int]              # option indices forming the realised event
    ev: str                       # choice | commit | defer | subset | gate | greedy | noop
    old_logp: float               # log P(event) under the behaviour policy (temperature T)
    dp: str | None
    step: int | None              # harness step (GiGPO discounting)
    key: str                      # anchor-state key: hash(dp, prompt)
    sampled: int
    greedy: int
    gold: int | None              # gold option index when the env exposes an oracle
    conf: float                   # pi_1(sampled)
    entropy: float                # H(pi_1) at decision time
    cascade: bool = False
    deferred: bool = False
    train: bool = True
    adv: float = 0.0
    ep: int = -1                  # index of the episode in the rollout batch
    ref: list[float] | None = None   # reference log-probs over the options (filled before the update)


class _Batcher:
    """Collects scoring requests from rollout threads and runs them as batched forward passes."""

    def __init__(self, fn: Callable[[list[tuple[str, int]]], list[Any]], max_batch: int, max_wait_ms: float):
        self.fn = fn
        self.max_batch = max(1, max_batch)
        self.max_wait = max(0.0, max_wait_ms) / 1000.0
        self.q: queue.Queue = queue.Queue()
        self.t = threading.Thread(target=self._loop, name="decider-batcher", daemon=True)
        self.t.start()

    def submit(self, items: list[tuple[str, int]]) -> list[Any]:
        fut: Future = Future()
        self.q.put((items, fut))
        return fut.result()

    def close(self) -> None:
        self.q.put(None)
        self.t.join(timeout=30)

    def _loop(self) -> None:
        while True:
            first = self.q.get()
            if first is None:
                return
            batch, n, stop = [first], len(first[0]), False
            deadline = time.perf_counter() + self.max_wait
            while n < self.max_batch:
                rem = deadline - time.perf_counter()
                if rem <= 0:
                    break
                try:
                    nxt = self.q.get(timeout=rem)
                except queue.Empty:
                    break
                if nxt is None:
                    stop = True
                    break
                batch.append(nxt)
                n += len(nxt[0])
            items = [it for its, _ in batch for it in its]
            try:
                res = self.fn(items)
            except BaseException as e:  # noqa: BLE001 - propagate to every waiting caller
                for _, f in batch:
                    f.set_exception(e)
            else:
                i = 0
                for its, f in batch:
                    f.set_result(res[i:i + len(its)])
                    i += len(its)
            if stop:
                return


class TorchDecider(DecisionBackend):
    """A trainable causal-LM decider living in the training process (sampling or greedy).

    ``begin_episode`` / ``end_episode`` scope the decision buffer to the calling thread, so several
    rollouts can run concurrently against one decider (and one model)."""

    name = "torch_decider"
    batched_questions = True
    max_options = JEV_MAX_OPTIONS  # > 52 options: knockout inside, exactly like serve_jev.py

    def __init__(self, net: Any, tokenizer: Any, harness_cfg: HarnessConfig | None = None, *,
                 temperature: float = 1.0, greedy: bool = False, max_len: int = 4096, batch_size: int = 16,
                 max_wait_ms: float = 5.0, price_model: str | None = None, name: str = "grpo-policy"):
        from .decision_model import DecisionCollator

        self.net = net
        self.tok = tokenizer
        self.cfg = harness_cfg
        self.temperature = float(temperature)
        self.greedy = greedy
        self.collator = DecisionCollator(tokenizer, max_len=max_len)
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.price = default_prices().get(price_model) if price_model else None
        self.model = name
        self._local = threading.local()
        self._lock = threading.Lock()
        self._batcher: _Batcher | None = None
        self._fallback_rng = random.Random(0)

    # ---------------------------------------------------------------- episode scope
    def begin_episode(self, seed: int = 0, harness_cfg: HarnessConfig | None = None) -> None:
        self._local.records = []
        self._local.rng = random.Random(seed)
        self._local.cfg = harness_cfg

    def end_episode(self) -> list[DecisionRecord]:
        recs = getattr(self._local, "records", None) or []
        self._local.records = None
        self._local.cfg = None
        return recs

    def enable_batching(self, on: bool = True) -> None:
        if on and self._batcher is None:
            self._batcher = _Batcher(self._forward, max(self.batch_size, 1) * 4, self.max_wait_ms)
        elif not on and self._batcher is not None:
            self._batcher.close()
            self._batcher = None

    # ---------------------------------------------------------------- scoring
    @torch.inference_mode()
    def _forward(self, items: list[tuple[str, int]]) -> list[tuple[list[float], int]]:
        """(user prompt, n options) -> (option logits, prompt tokens). Thread-safe (serialised)."""
        from .decision_model import option_logits

        out: list[Any] = [None] * len(items)
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        with self._lock:
            dev = next(self.net.parameters()).device
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                enc = self.collator([{"prompt": items[i][0], "target": [0.0] * items[i][1]} for i in idx])
                z = option_logits(self.net, enc["input_ids"].to(dev), enc["attention_mask"].to(dev),
                                  enc["code_ids"].to(dev)).float().cpu()
                lens = enc["attention_mask"].sum(-1).tolist()
                for j, i in enumerate(idx):
                    out[i] = (z[j, :items[i][1]].tolist(), int(lens[j]))
        return out

    def _logits(self, items: list[tuple[str, int]]) -> list[tuple[list[float], int]]:
        return self._batcher.submit(items) if self._batcher is not None else self._forward(items)

    def _knockout(self, ex: dict[str, Any]) -> tuple[list[int], str, list[float], int]:
        opts = ex["options"]
        pool, ntok = list(range(len(opts))), 0

        def score(idx: list[int]) -> tuple[str, list[float], int]:
            prompt = render({**ex, "options": [opts[k] for k in idx]})[0]
            (z, nt), = self._logits([(prompt, len(idx))])
            return prompt, z, nt

        while len(pool) > MAX_PASS:
            nxt: list[int] = []
            for j in range(0, len(pool), MAX_PASS):
                idx = pool[j:j + MAX_PASS]
                if len(idx) == 1:
                    nxt += idx
                    continue
                _, z, nt = score(idx)
                ntok += nt
                nxt += [idx[r] for r in sorted(range(len(idx)), key=lambda r: -z[r])[:2]]
            pool = nxt
        prompt, z, nt = score(pool)
        return pool, prompt, z, ntok + nt

    # ---------------------------------------------------------------- acting
    def _gate(self, dp: str | None) -> tuple[Mode, float, int]:
        cfg = getattr(self._local, "cfg", None) or self.cfg
        if cfg is None or dp is None:
            return Mode.REPLACE, 0.0, 1
        try:
            c = cfg.dp(DP(dp))
        except ValueError:
            return Mode.REPLACE, 0.0, 1
        return c.mode, c.tau, c.k

    def _act(self, q: Question, ex: dict[str, Any], pool: list[int], prompt: str,
             z: list[float]) -> tuple[Answer, DecisionRecord]:
        labels = ex["labels"]
        plabels = [labels[i] for i in pool]
        m = len(pool)
        zt = torch.tensor(z, dtype=torch.float64)
        logp1 = torch.log_softmax(zt, -1)
        p1 = logp1.exp().tolist()
        logpT = torch.log_softmax(zt / self.temperature, -1)
        pT = logpT.exp().tolist()
        greedy = max(range(m), key=lambda i: z[i])
        mode, tau, k = self._gate(q.decision_point)
        gate = q.kind == QKind.NOUL
        cascade = mode == Mode.CASCADE and not gate
        gold = _gold_label(q)
        gold_i = plabels.index(gold) if gold in plabels else None
        rng = getattr(self._local, "rng", None) or self._fallback_rng

        def commit(i: int) -> bool:
            return p1[i] >= tau and plabels[i] != OTHER

        dist = {lab: float(p) for lab, p in zip(plabels, p1)}
        if self.greedy:
            a, event, ev, train = greedy, [greedy], "greedy", False
            deferred = cascade and not commit(a)
        elif gate:
            a = _sample(pT, rng)
            event, ev, train, deferred = [a], "gate", True, False
            dist = {"yes": 1.0 if a == 0 else 0.0, "no": 0.0 if a == 0 else 1.0}  # the harness gate follows the sample
        elif mode == Mode.ASSIST:
            if m > k:
                sub = _gumbel_topk(logpT.tolist(), k, rng)
                a, event, ev, train = sub[0], sub, "subset", True
                dist = {lab: 1e-3 * p1[i] for i, lab in enumerate(plabels)}  # top_k(k) == sampled subset, in order
                for r, i in enumerate(sub):
                    dist[plabels[i]] = float(k - r)
                zsum = sum(dist.values())
                dist = {lab: v / zsum for lab, v in dist.items()}
            else:  # everything is kept: the decision has no effect
                a, event, ev, train = greedy, [greedy], "noop", False
            deferred = False
        else:
            a = _sample(pT, rng)
            if cascade:
                deferred = not commit(a)
                event = [i for i in range(m) if not commit(i)] if deferred else [a]
                ev = "defer" if deferred else "commit"
                train = len(event) < m  # an event of probability 1 carries no gradient
            else:
                event, ev, train, deferred = [a], "choice", True, False
        old = float(event_logp(logpT, event, ev)) if train else 0.0
        if len(pool) < len(labels):  # knockout: eliminated options get probability 0 (as serve_jev)
            dist = {**{lab: 0.0 for lab in labels}, **dist}
        if q.kind == QKind.NOUL:
            value: Any = bool(dist["yes"] >= 0.5) if not self.greedy else bool(p1[0] >= 0.5)
        elif q.kind == QKind.SCORE:
            value = float(plabels[a])
        else:
            value = plabels[a]
        ans = Answer(question_id=q.id, kind=q.kind, distribution=dist, value=value, confidence=float(p1[a]),
                     raw={"policy": {lab: float(p) for lab, p in zip(plabels, p1)}, "event": ev})
        ent = -sum(p * lp for p, lp in zip(p1, logp1.tolist()) if p > 0)
        rec = DecisionRecord(prompt=prompt, n=m, event=list(event), ev=ev, old_logp=old, dp=q.decision_point,
                             step=getattr(self._local, "step", None),
                             key=hashlib.sha1(f"{q.decision_point}\x00{prompt}".encode()).hexdigest()[:20],
                             sampled=a, greedy=greedy, gold=gold_i, conf=float(p1[a]), entropy=float(ent),
                             cascade=cascade, deferred=deferred, train=train)
        return ans, rec

    def decide(self, state: Any, questions: list[Question], ledger: Any = None,
               decision_point: str | None = None) -> DecisionResult:
        self._local.step = getattr(ledger, "step", None)
        return super().decide(state, questions, ledger=ledger, decision_point=decision_point)

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        t0 = time.perf_counter()
        exs = [question_to_example(state, q) for q in questions]
        small = [i for i, ex in enumerate(exs) if len(ex["options"]) <= MAX_PASS]
        prompts = {i: render(exs[i])[0] for i in small}
        outs = self._logits([(prompts[i], len(exs[i]["options"])) for i in small]) if small else []
        scored: dict[int, tuple[list[int], str, list[float], int]] = {
            i: (list(range(len(exs[i]["options"]))), prompts[i], z, nt) for i, (z, nt) in zip(small, outs)}
        for i, ex in enumerate(exs):
            if i not in scored:
                scored[i] = self._knockout(ex)
        recs = getattr(self._local, "records", None)
        answers, n_in = [], 0
        for i, q in enumerate(questions):
            pool, prompt, z, nt = scored[i]
            n_in += nt
            ans, rec = self._act(q, exs[i], pool, prompt, z)
            answers.append(ans)
            if recs is not None:
                recs.append(rec)
        cost = self.price.cost(n_in, 0) if self.price else 0.0
        return DecisionResult(answers=answers, usage=Usage(
            backend=self.name, model=self.model, input_tokens=n_in, cost_usd=cost,
            latency_ms=(time.perf_counter() - t0) * 1000))

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model, "greedy": self.greedy, "temperature": self.temperature}


# =============================================================================== rewards & advantages
@dataclass
class Episode:
    task_id: str
    group: int
    rollout: int
    arm: str
    reward: float
    success: bool
    cost_usd: float
    llm_calls: int
    s1_calls: int
    n_steps: int
    irreversible: bool
    error: str | None
    records: list[DecisionRecord] = field(default_factory=list)
    adv: float = 0.0
    zero_std: bool = False


def episode_reward(traj: Any, rc: RewardCfg) -> float:
    """Scalar reward of a harness Trajectory (see module docstring)."""
    usage = traj.usage or {}
    success = 1.0 if traj.success else 0.0
    cost = float(usage.get("total", {}).get("cost_usd", 0.0))
    calls = float(usage.get("s2", {}).get("calls", 0))
    if rc.kind == "linear":
        r = success - rc.lambda_cost * cost - rc.lambda_calls * calls
    else:
        c = cost if rc.cost_unit == "usd" else calls
        norm = min(1.0, c / rc.cost_scale) if rc.cost_scale > 0 else 0.0
        r = success * (1.0 - rc.lambda_gate * norm)
    if (traj.result or {}).get("irreversible_error"):
        r -= rc.irreversible_penalty
    return float(r)


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def compute_advantages(episodes: list[Episode], cfg: AlgoCfg) -> dict[str, float]:
    """GRPO group advantages (+ optional GiGPO anchor-state advantages), written onto every record.

    Records get ``train=False`` when their episode errored (``drop_error_episodes``) or when they carry no
    signal at all: their episode group has identical rewards AND (with GiGPO) their anchor group has identical
    discounted returns (``drop_zero_std_groups``, DAPO-style dynamic sampling)."""
    groups: dict[int, list[Episode]] = defaultdict(list)
    n_err = 0
    for e in episodes:
        if cfg.drop_error_episodes and e.error:
            n_err += 1
            e.adv = 0.0
            for r in e.records:
                r.train = False
            continue
        groups[e.group].append(e)
    kept = [e for g in groups.values() for e in g]
    bstd = _std([e.reward for e in kept])
    n_zero = 0
    for eps_ in groups.values():
        rs = [e.reward for e in eps_]
        mu, sd = sum(rs) / len(rs), _std(rs)
        zero = sd < 1e-8
        n_zero += int(zero)
        for e in eps_:
            if zero:
                e.adv = 0.0
            elif cfg.adv_scale == "group":
                e.adv = (e.reward - mu) / (sd + cfg.adv_eps)
            elif cfg.adv_scale == "batch":
                e.adv = (e.reward - mu) / (bstd + cfg.adv_eps)
            else:
                e.adv = e.reward - mu
            e.zero_std = zero
            for r in e.records:
                r.adv = e.adv
    step_signal: set[int] = set()  # id() of records that received a non-zero step-level advantage
    n_anchor = n_cand = 0
    if cfg.gigpo.enabled:
        for eps_ in groups.values():
            anchors: dict[str, list[tuple[DecisionRecord, float]]] = defaultdict(list)
            for e in eps_:
                T = max(e.n_steps, 1)
                for r in e.records:
                    if r.train:
                        n_cand += 1
                        t = r.step if r.step is not None else 0
                        anchors[r.key].append((r, cfg.gigpo.gamma ** max(T - 1 - t, 0) * e.reward))
            for items in anchors.values():
                if len(items) < 2:
                    continue
                g = [x for _, x in items]
                mu, sd = sum(g) / len(g), _std(g)
                n_anchor += len(items)
                if sd < 1e-8:
                    continue
                for r, x in items:
                    a_s = (x - mu) / (sd + cfg.adv_eps) if cfg.gigpo.scale == "std" else x - mu
                    r.adv += cfg.gigpo.omega * a_s
                    step_signal.add(id(r))
    if cfg.drop_zero_std_groups:
        for e in kept:
            if e.zero_std:
                for r in e.records:
                    if id(r) not in step_signal:
                        r.train = False
    n_train = sum(r.train for e in episodes for r in e.records)
    return {"frac_zero_std_groups": n_zero / max(1, len(groups)), "error_episodes": n_err,
            "anchor_frac": n_anchor / max(1, n_cand),
            "updated_decisions": n_train,
            "adv_abs_mean": (sum(abs(r.adv) for e in episodes for r in e.records if r.train) / max(1, n_train))}


# =============================================================================== loss
def grpo_decision_loss(z: torch.Tensor, events: Sequence[Sequence[int]], evs: Sequence[str], old_logp: torch.Tensor,
                       adv: torch.Tensor, ref_logp: torch.Tensor | None, *, temperature: float = 1.0,
                       clip_eps: float = 0.2, clip_eps_high: float | None = None, kl_beta: float = 0.0,
                       entropy_coef: float = 0.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO-clipped surrogate on event log-probs + exact KL / entropy over the option set.

    z [B, K]: option logits, -inf beyond each row's option count. Returns (per-decision loss [B], stats)."""
    mask = torch.isfinite(z)
    logpT = F.log_softmax(z / temperature, -1)
    new = torch.stack([event_logp(logpT[i], events[i], evs[i]) for i in range(z.size(0))])
    ratio = torch.exp(new - old_logp)
    hi = clip_eps if clip_eps_high is None else clip_eps_high
    pg = torch.maximum(-adv * ratio, -adv * ratio.clamp(1.0 - clip_eps, 1.0 + hi))
    logp1 = F.log_softmax(z, -1).masked_fill(~mask, 0.0)
    p1 = logp1.exp() * mask
    ent = -(p1 * logp1).sum(-1)
    if ref_logp is not None:
        kl = (p1 * (logp1 - ref_logp.masked_fill(~mask, 0.0))).sum(-1)
    else:
        kl = torch.zeros_like(ent)
    loss = pg + kl_beta * kl - entropy_coef * ent
    clipped = ((ratio < 1.0 - clip_eps) & (adv < 0)) | ((ratio > 1.0 + hi) & (adv > 0))
    return loss, {"pg": pg.detach(), "kl": kl.detach(), "entropy": ent.detach(), "ratio": ratio.detach(),
                  "clipped": clipped.float(), "approx_kl_old": (old_logp - new).detach()}


# =============================================================================== model


def load_policy(cfg: Config, device: str, dtype: torch.dtype, policy_adapter: str | None = None
                ) -> tuple[Any, Any, str]:
    """LoRA policy (fresh, or continued from ``init`` / a checkpoint) + tokenizer + reference mode."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer

    from .decision_model import add_reference_adapter
    from .train_decision import load_causal_lm

    ref_mode = cfg.ref if cfg.ref != "auto" else ("init" if cfg.init else "base")
    if ref_mode == "init" and not cfg.init:
        raise ValueError("ref=init needs `init` (the adapter the policy starts from)")
    start = policy_adapter or cfg.init
    tok = AutoTokenizer.from_pretrained(start or cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = load_causal_lm(cfg.model, dtype)
    if start:
        model = PeftModel.from_pretrained(base, start, is_trainable=True)
        if ref_mode == "init":
            add_reference_adapter(model, cfg.init)
    else:
        model = get_peft_model(base, LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha,
                                                lora_dropout=cfg.lora.dropout,
                                                target_modules=cfg.lora.target_modules, task_type="CAUSAL_LM"))
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model.to(device), tok, ref_mode


@torch.no_grad()
def fill_reference(model: Any, collator: Any, records: list[DecisionRecord], ref_mode: str,
                   batch_size: int) -> None:
    from .decision_model import option_logits

    model.eval()
    dev = next(model.parameters()).device
    order = sorted(range(len(records)), key=lambda i: len(records[i].prompt))
    with reference_context(model, ref_mode):
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = collator([{"prompt": records[i].prompt, "target": [0.0] * records[i].n} for i in idx])
            z = option_logits(model, enc["input_ids"].to(dev), enc["attention_mask"].to(dev),
                              enc["code_ids"].to(dev)).float()
            lp = F.log_softmax(z, -1).cpu()
            for j, i in enumerate(idx):
                records[i].ref = lp[j, :records[i].n].tolist()


def update_policy(model: Any, collator: Any, records: list[DecisionRecord], n_by_ep: dict[int, int],
                  cfg: Config, opt: torch.optim.Optimizer, sched: Any, rng: random.Random) -> dict[str, float]:
    """mu epochs of PPO-clip + KL over the decision buffer. Returns averaged statistics."""
    from .decision_model import option_logits

    model.train()
    A = cfg.grpo
    dev = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    w_of = {i: (1.0 if A.loss_agg == "decision" else 1.0 / max(1, n_by_ep.get(r.ep, 1)))
            for i, r in enumerate(records)}
    acc: dict[str, list[float]] = defaultdict(list)
    gnorms: list[float] = []
    n_skip = 0
    idx = list(range(len(records)))
    tok_len = ({i: len(collator.encode(records[i].prompt)) for i in idx} if A.max_batch_tokens > 0 else {})
    for _ in range(A.inner_epochs):
        rng.shuffle(idx)
        for s in range(0, len(idx), A.minibatch_size):
            mb = sorted(idx[s:s + A.minibatch_size], key=lambda i: len(records[i].prompt))
            W = sum(w_of[i] for i in mb)
            opt.zero_grad(set_to_none=True)
            if A.max_batch_tokens > 0:
                from .decision_model import token_budget_batches

                micros = [[mb[j] for j in b] for b in token_budget_batches(
                    [tok_len[i] for i in mb], A.max_batch_tokens, max(1, A.micro_batch_size) * 8)]
            else:
                micros = [mb[u:u + A.micro_batch_size] for u in range(0, len(mb), A.micro_batch_size)]
            for micro in micros:
                rs = [records[i] for i in micro]
                enc = collator([{"prompt": r.prompt, "target": [0.0] * r.n} for r in rs])
                z = option_logits(model, enc["input_ids"].to(dev), enc["attention_mask"].to(dev),
                                  enc["code_ids"].to(dev))
                ref = torch.zeros_like(z)
                for j, r in enumerate(rs):
                    ref[j, :r.n] = torch.tensor(r.ref, device=dev)
                loss, st = grpo_decision_loss(
                    z, [r.event for r in rs], [r.ev for r in rs],
                    torch.tensor([r.old_logp for r in rs], device=dev, dtype=z.dtype),
                    torch.tensor([r.adv for r in rs], device=dev, dtype=z.dtype), ref,
                    temperature=cfg.decider.temperature, clip_eps=A.clip_eps, clip_eps_high=A.clip_eps_high,
                    kl_beta=A.kl_beta, entropy_coef=A.entropy_coef)
                w = torch.tensor([w_of[i] for i in micro], device=dev, dtype=z.dtype)
                total = (loss * w).sum() / W
                if not torch.isfinite(total):
                    n_skip += 1
                    continue
                total.backward()
                acc["loss"] += loss.detach().float().cpu().tolist()
                for k, v in st.items():
                    acc[k] += v.float().cpu().tolist()
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
            if torch.isfinite(gn):
                opt.step()
            else:
                n_skip += 1
            sched.step()
            gnorms.append(float(gn))

    def mean(k: str) -> float:
        return float(sum(acc[k]) / len(acc[k])) if acc[k] else float("nan")

    return {"loss": mean("loss"), "pg_loss": mean("pg"), "kl": mean("kl"), "entropy": mean("entropy"),
            "clip_frac": mean("clipped"), "ratio_mean": mean("ratio"), "approx_kl_old": mean("approx_kl_old"),
            "grad_norm": float(sum(gnorms) / len(gnorms)) if gnorms else float("nan"),
            "optimizer_steps": len(gnorms), "nonfinite_skipped": n_skip, "lr": float(sched.get_last_lr()[0])}


def save_adapter(model: Any, tok: Any, path: str | Path, cfg: Config, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), selected_adapters=["default"])
    tok.save_pretrained(str(path))
    (path / "decision_config.json").write_text(json.dumps(
        {"base_model": cfg.model, "loss": "grpo_online", "lora": True, "max_len": cfg.decider.max_len,
         "init": cfg.init, "arms": [a.get("name") for a in cfg.arms], **(extra or {})}, indent=1))


# =============================================================================== rollouts
def _seed(*parts: Any) -> int:
    return random.Random("-".join(map(str, parts))).getrandbits(32)


def run_episodes(jobs: list[tuple[int, Callable[[], Any], int, HarnessConfig]], decider: TorchDecider,
                 make_agent: Callable[[HarnessConfig, TorchDecider], Any], rc: RewardCfg, workers: int,
                 seed: Any) -> list[Episode]:
    """jobs: (group index, env factory, rollout index, arm). One harness episode per job."""

    def one(job: tuple[int, Callable[[], Any], int, HarnessConfig]) -> Episode:
        gi, factory, g, arm = job
        env = factory()
        decider.begin_episode(_seed(seed, gi, g), arm)
        try:
            traj = make_agent(arm, decider).run(env)
        finally:
            recs = decider.end_episode()
        usage = traj.usage or {}
        return Episode(task_id=traj.task_id, group=gi, rollout=g, arm=arm.name, reward=episode_reward(traj, rc),
                       success=traj.success, cost_usd=float(usage.get("total", {}).get("cost_usd", 0.0)),
                       llm_calls=int(usage.get("s2", {}).get("calls", 0)),
                       s1_calls=int(usage.get("s1", {}).get("calls", 0)), n_steps=len(traj.steps),
                       irreversible=bool((traj.result or {}).get("irreversible_error")),
                       error=traj.error, records=recs)

    decider.enable_batching(workers > 1)
    try:
        if workers <= 1:
            episodes = [one(j) for j in jobs]
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                episodes = list(ex.map(one, jobs))
    finally:
        decider.enable_batching(False)
    for i, e in enumerate(episodes):
        for r in e.records:
            r.ep = i
    return episodes


def _clean(v: float) -> float | None:
    return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v


def rollout_stats(episodes: list[Episode]) -> dict[str, float]:
    def mean(xs: Sequence[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    recs = [r for e in episodes for r in e.records]
    gold = [r for r in recs if r.gold is not None]
    out = {"episodes": len(episodes), "decisions": len(recs), "trainable_decisions": sum(r.train for r in recs),
           "reward_mean": mean([e.reward for e in episodes]), "reward_std": _std([e.reward for e in episodes]),
           "success": mean([float(e.success) for e in episodes]), "cost_usd": mean([e.cost_usd for e in episodes]),
           "llm_calls": mean([e.llm_calls for e in episodes]), "s1_calls": mean([e.s1_calls for e in episodes]),
           "steps": mean([e.n_steps for e in episodes]),
           "irreversible_rate": mean([float(e.irreversible) for e in episodes]),
           "error_rate": mean([float(e.error is not None) for e in episodes]),
           "defer_rate": mean([float(r.deferred) for r in recs if r.cascade]),
           # cascade decisions where no option reached tau: deferral had probability 1 -> no gradient (cold start)
           "cascade_forced_defer_frac": mean([float(r.ev == "defer" and len(r.event) == r.n) for r in recs if r.cascade]),
           "dec_acc": mean([float(r.sampled == r.gold) for r in gold]),
           "dec_acc_greedy": mean([float(r.greedy == r.gold) for r in gold]),
           "conf_mean": mean([r.conf for r in recs]), "entropy_rollout": mean([r.entropy for r in recs])}
    by_dp: dict[str, list[DecisionRecord]] = defaultdict(list)
    for r in gold:
        by_dp[str(r.dp)].append(r)
    for dp, rs in sorted(by_dp.items()):
        out[f"dec_acc/{dp}"] = mean([float(r.sampled == r.gold) for r in rs])
    return out


# =============================================================================== training loop
def train(cfg: Config, resume: str | None = None) -> Path:
    from dotenv import load_dotenv

    from ..bench.suites import load_suite
    from ..harness.agent import JevAgent
    from ..llm import make_llm

    load_dotenv()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = cfg.bf16 if cfg.bf16 is not None else device.startswith("cuda")
    dtype = torch.bfloat16 if bf16 else torch.float32
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grpo_config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))

    arms = cfg.harness_arms()
    try:
        cfg.check_seeds()
    except ValueError as e:
        raise SystemExit(f"[grpo_online] {e}") from None
    model, tok, ref_mode = load_policy(cfg, device, dtype, policy_adapter=resume)
    dk = cfg.decider
    common = dict(max_len=dk.max_len, batch_size=dk.batch_size, max_wait_ms=dk.max_wait_ms,
                  price_model=dk.price_model, name=dk.name)
    decider = TorchDecider(model, tok, arms[0], temperature=dk.temperature, greedy=False, **common)
    evaluator = TorchDecider(model, tok, arms[0], greedy=True, **common)
    executor = make_llm(cfg.executor)
    named = {name: make_llm(spec) for name, spec in cfg.executors.items()}
    arm_llm = {a.get("name", "llm_only"): a.get("llm") for a in cfg.arms}
    small = make_llm(cfg.small_executor) if cfg.small_executor else None

    def make_agent(arm: HarnessConfig, s1: TorchDecider) -> Any:  # per-arm executor (`llm: <name of executors>`)
        ex = named[arm_llm[arm.name]] if arm_llm.get(arm.name) else executor
        return JevAgent(ex, arm, s1=s1, small_llm=small, system_prompt=cfg.system_prompt)

    eval_arms = [a for a in arms if not cfg.eval_arms or a.name in cfg.eval_arms]

    suite = dict(cfg.suite)
    train_tasks = load_suite(suite.pop("name"), **suite)
    eval_tasks: list[Any] = []
    if cfg.eval_suite:
        es = dict(cfg.eval_suite)
        eval_tasks = load_suite(es.pop("name"), **es)
        overlap = {f().task_id for f in train_tasks} & {f().task_id for f in eval_tasks}
        if overlap:
            print(f"[grpo_online] WARNING: {len(overlap)} eval tasks also in the training suite")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay, betas=cfg.optim.betas)
    warm = max(1, cfg.optim.warmup_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    rng = random.Random(cfg.seed)
    order = list(range(len(train_tasks)))
    rng.shuffle(order)
    cursor, start_it = 0, 0
    if resume:
        st = torch.load(Path(resume) / "trainer_state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        start_it, cursor, order = st["iteration"], st["cursor"], st["order"]
        rng.setstate(st["rng"])
    metrics_path = out / "metrics.jsonl"

    def log(row: dict[str, Any]) -> None:
        row = {k: _clean(v) if isinstance(v, float) else v for k, v in row.items()}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        keys = ["iter", "reward_mean", "success", "cost_usd", "llm_calls", "defer_rate", "dec_acc", "kl", "loss"]
        print("[grpo_online] " + " ".join(f"{k}={row[k]:.4g}" if isinstance(row.get(k), float) else
                                          f"{k}={row.get(k)}" for k in keys if k in row), flush=True)

    def evaluate(it: int) -> dict[str, float]:
        res: dict[str, float] = {}
        if not eval_tasks:
            return res
        model.eval()
        for arm in eval_arms:
            jobs = [(gi, f, 0, arm) for gi, f in enumerate(eval_tasks)]
            eps_ = run_episodes(jobs, evaluator, make_agent, cfg.reward, cfg.workers, ("eval", it))
            st = rollout_stats(eps_)
            for k in ("reward_mean", "success", "cost_usd", "llm_calls", "s1_calls", "steps", "defer_rate",
                      "dec_acc_greedy", "irreversible_rate", "error_rate"):
                res[f"eval/{arm.name}/{k.replace('dec_acc_greedy', 'dec_acc')}"] = st[k]
        return res

    if cfg.eval_every and start_it == 0:
        log({"iter": 0, **evaluate(0)})
    it = start_it
    for it in range(start_it + 1, cfg.grpo.iterations + 1):
        picked = []
        for _ in range(min(cfg.grpo.tasks_per_iter, len(order))):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            picked.append(order[cursor])
            cursor += 1
        jobs = [(gi, train_tasks[ti], g, arms[gi % len(arms)]) for gi, ti in enumerate(picked)
                for g in range(cfg.grpo.group_size)]
        t0 = time.perf_counter()
        model.eval()
        episodes = run_episodes(jobs, decider, make_agent, cfg.reward, cfg.workers, (cfg.seed, it))
        t_roll = time.perf_counter() - t0
        errors = [e.error for e in episodes if e.error]
        if errors and len(errors) == len(episodes):
            raise RuntimeError(f"every rollout failed; first error:\n{errors[0]}")
        row: dict[str, Any] = {"iter": it, **rollout_stats(episodes), **compute_advantages(episodes, cfg.grpo),
                               "time_rollout_s": t_roll}
        for arm in arms:  # per-arm reward / success / cost of the sampled policy
            ea = [e for e in episodes if e.arm == arm.name]
            if ea:
                sa = rollout_stats(ea)
                row.update({f"arm/{arm.name}/{k}": sa[k] for k in ("reward_mean", "success", "cost_usd", "llm_calls",
                                                                   "defer_rate", "dec_acc_greedy")})
        records = [r for e in episodes for r in e.records if r.train]
        t1 = time.perf_counter()
        if records:
            n_by_ep: dict[int, int] = defaultdict(int)
            for r in records:
                n_by_ep[r.ep] += 1
            fill_reference(model, decider.collator, records, ref_mode, dk.batch_size)
            row.update(update_policy(model, decider.collator, records, n_by_ep, cfg, opt, sched, rng))
        else:
            print(f"[grpo_online] iter {it}: no decision carries a learning signal (zero-variance groups: "
                  f"{row['frac_zero_std_groups']:.2f}, cascade decisions with P(defer)=1: "
                  f"{row.get('cascade_forced_defer_frac')}). Warm-start from an SFT adapter (`init`), raise "
                  "group_size, use a replace/assist arm, or a linear reward with a cost term.", flush=True)
        row["time_update_s"] = time.perf_counter() - t1
        if cfg.log_rollouts:
            with open(out / "rollouts.jsonl", "a") as f:
                for e in episodes:
                    f.write(json.dumps({"iter": it, "task_id": e.task_id, "group": e.group, "rollout": e.rollout,
                                        "arm": e.arm, "reward": e.reward, "adv": e.adv, "success": e.success,
                                        "cost_usd": e.cost_usd, "llm_calls": e.llm_calls, "steps": e.n_steps,
                                        "decisions": len(e.records), "error": e.error}) + "\n")
        if errors:
            row["first_error"] = errors[0][:500]
        if cfg.eval_every and (it % cfg.eval_every == 0 or it == cfg.grpo.iterations):
            row.update(evaluate(it))
        log(row)
        if cfg.save_every and it % cfg.save_every == 0 and it < cfg.grpo.iterations:
            ck = out / f"ckpt-{it:05d}"
            save_adapter(model, tok, ck, cfg, {"iteration": it})
            torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "iteration": it,
                        "cursor": cursor, "order": order, "rng": rng.getstate()}, ck / "trainer_state.pt")
    save_adapter(model, tok, out, cfg, {"iteration": it})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None, help="YAML config (see configs/train/grpo_online_jevgym.yaml)")
    ap.add_argument("--set", nargs="*", default=[], action="extend", metavar="KEY=VALUE",
                    help="dotted overrides parsed as YAML, e.g. grpo.kl_beta=0.05 reward.kind=linear "
                         "'arms=[{name: a, tool_select: {mode: replace}}]'")
    ap.add_argument("--model", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--resume", default=None, help="checkpoint dir (ckpt-XXXXX) to continue from")
    ap.add_argument("--print-config", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set, model=a.model, init=a.init, out=a.out,
                      **({"grpo.iterations": a.iterations} if a.iterations is not None else {}))
    if a.print_config:
        print(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
        return
    print(json.dumps({"out": str(train(cfg, resume=a.resume))}))


if __name__ == "__main__":
    main()
