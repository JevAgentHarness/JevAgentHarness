"""On-policy distillation (OPD) of a System-1 decider *inside* the agent harness (stage 65, recipe R9).

Loop, repeated for ``--iterations``:

  1. rollout        the STUDENT acts as System-1 in ``JevAgent`` on JevGym training tasks (seeds >= 10000), sampling
                    its option at temperature T (``--rollout-temperature 0`` = greedy). The visited decision states
                    are therefore drawn from the student's OWN state distribution (the on-policy part), including the
                    states its mistakes lead to.
  2. teacher label  the TEACHER -- any ``DecisionBackend`` of ``make_backend`` (Jev, Intern-Decision, the Gemini
                    verbalised distribution, a served checkpoint) or a trained checkpoint read out in-process by
                    ``serve_jev.Engine`` (``{"type": "local", ...}``) -- answers the SAME typed questions at the SAME
                    states with a full distribution p_T over the options. Every call goes through a ``Ledger``;
                    ``--budget-usd`` stops the run when the teacher spend passes it.
  3. train          one pass of ``--epochs-per-iter`` over the fresh rows (+ a replay of the previous iteration's rows,
                    ``--replay-frac``) with a per-question divergence over the option codes:

                      forward_kl   KL(p_T || pi_theta) = sum_a p_T(a) log p_T(a)/pi(a)   (= soft CE - H(p_T): the
                                   gradient of the stages-20/40/80 soft-CE objective; mean-seeking)
                      reverse_kl   KL(pi_theta || p_T)                                      (mode-seeking; what
                                   on-policy distillation of generators usually minimises)
                      jsd          1/2 KL(p_T || m) + 1/2 KL(pi || m), m = (p_T + pi)/2    (bounded, symmetric)

No reward is designed and no gold label is used for learning (``--gold-alpha`` can mix the JevGym oracle one-hot in
for ablations; it is 0 by default). Decision accuracy vs the oracle is logged, never optimised.

Prompts are ``data.render`` + ``decision_model.chat_text`` byte for byte in all three steps: rollout scoring and the
update share one ``DecisionCollator``; a local teacher scores the identical rendered example. The saved checkpoint
loads unchanged in ``serve_jev.py`` / ``calibrate.py`` / ``eval_decisions.py``.

  python -m jevharness.training.opd --config configs/train/opd_jevgym.yaml --student Qwen/Qwen3-4B \\
      --init checkpoints/r3 --teacher jev-openrouter --out checkpoints/r9-opd --budget-usd 5

Outputs (``--out``): ``metrics.jsonl`` (one row per iteration: rollout success / cost / calls, states visited,
labelled rows, student-vs-teacher agreement, KL(p_T || pi) on the fresh states before the update, training loss,
teacher $ for the iteration and in total, ``eval/<arm>/*`` greedy dev rollouts), ``labelled/iter-XXXXX.jsonl``
(the teacher-labelled rows in the training format: reusable off-policy by ``10_sft.sh`` / ``80_distill_small.sh``),
``ckpt-XXXXX/`` (adapter + optimizer state; ``--resume auto`` continues from the latest), ``summary.json`` and the
final adapter in ``--out`` itself.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
import yaml

from ..accounting import Ledger
from ..decision.types import Choice, Noul, QKind, Question, Score
from ..harness.config import HarnessConfig
from .data import permuted, read_jsonl, render, write_jsonl
from .decision_model import DecisionCollator, option_logits
from .grpo_online import DecisionRecord, Episode, RewardCfg, TorchDecider, _clean, rollout_stats, run_episodes
from .teacher_label import SIMULATED_TYPES, LocalTeacher, _norm, ledger_summary, make_teacher, parse_teacher

LOSSES = ("forward_kl", "reverse_kl", "jsd")
MIN_TRAIN_SEED = 10000       # JevGym training seeds (TRAINING_AGENT_BRIEF hard rule 1); 5000+ = dev, 0..199 = E2 eval
MIN_DEV_SEED = 1000
DEFAULT_SUITE = {"n_distractors": 30, "distractor_style": "mixed", "families": "all", "strict": True}
DEFAULT_ARMS = [{"name": "opd_replace", "tool_select": {"mode": "replace"}, "tool_params": {"mode": "replace"}}]
SECRET_KEYS = re.compile(r"(api_key|token|secret|password)", re.I)


# =============================================================================== loss
def opd_loss(z: torch.Tensor, target: torch.Tensor, loss: str = "forward_kl", eps: float = 1e-6) -> torch.Tensor:
    """Per-row divergence between the teacher distribution ``target`` [B, K] and the student's restricted softmax
    of ``z`` [B, K] (``-inf`` beyond each row's option count, as ``option_logits`` returns it). Returns [B].

    ``forward_kl`` equals the ``decision_model.decision_loss('soft_ce')`` objective minus the teacher entropy, so
    both have the same gradient; it is kept as a divergence so the logged value is 0 when student == teacher."""
    if loss not in LOSSES:
        raise ValueError(f"unknown OPD loss {loss!r}; choose from {LOSSES}")
    mask = torch.isfinite(z)
    logp = F.log_softmax(z, dim=-1).masked_fill(~mask, 0.0)
    p = logp.exp() * mask
    t = target.to(z.dtype).clamp(min=0.0) * mask
    t = t / t.sum(-1, keepdim=True).clamp(min=eps)
    logt = torch.log(t.clamp(min=eps)).masked_fill(~mask, 0.0)
    zero = torch.zeros_like(p)
    if loss == "forward_kl":
        return torch.where(t > 0, t * (logt - logp), zero).sum(-1)
    if loss == "reverse_kl":
        return torch.where(mask, p * (logp - logt), zero).sum(-1)
    m = 0.5 * (p + t)
    logm = torch.log(m.clamp(min=eps)).masked_fill(~mask, 0.0)
    return 0.5 * torch.where(t > 0, t * (logt - logm), zero).sum(-1) + \
        0.5 * torch.where(mask, p * (logp - logm), zero).sum(-1)


def forward_kl(teacher: Sequence[float], z: Sequence[float]) -> float:
    """KL(p_T || softmax(z)) for one decision (plain floats; the pre-update diagnostic on fresh states)."""
    zt = torch.tensor(list(z), dtype=torch.float64)
    return float(opd_loss(zt[None], torch.tensor(list(teacher), dtype=torch.float64)[None], "forward_kl")[0])


# =============================================================================== the student as System-1
@dataclass
class Visit(DecisionRecord):
    """A decision the student made during a rollout, with everything the teacher needs to be asked the same
    question: the rendered state, the question text, the scored option pool (its rendered strings = the student's
    prompt lines, and the question labels the teacher answers with) and the harness ``Question`` itself."""
    task_id: str = ""
    state: str = ""
    qtext: str = ""
    kind: str = "choice"
    options: list[str] = field(default_factory=list)   # rendered option strings of the scored pool (prompt order)
    labels: list[str] = field(default_factory=list)    # question labels of the pool (names / yes,no / levels)
    pool: list[int] = field(default_factory=list)      # indices into the harness question's options
    question: Any = None                               # the harness Question (its meta never reaches a real teacher)
    z: list[float] = field(default_factory=list)       # student option logits over the pool at rollout time
    teacher: list[float] | None = None                 # teacher distribution over the pool (filled by label_visits)


class OPDDecider(TorchDecider):
    """``TorchDecider`` whose records are :class:`Visit` objects (same sampling / cascade / assist / gate semantics,
    same knockout for > 52 options, same prompts)."""

    name = "opd_student"

    def _act(self, q: Question, ex: dict[str, Any], pool: list[int], prompt: str,
             z: list[float]) -> tuple[Any, DecisionRecord]:
        ans, rec = super()._act(q, ex, pool, prompt, z)
        v = Visit(**vars(rec), state=ex["state"], qtext=ex["question"], kind=ex["kind"],
                  options=[ex["options"][k] for k in pool], labels=[ex["labels"][k] for k in pool],
                  pool=list(pool), question=q, z=[float(x) for x in z])
        return ans, v


# =============================================================================== teacher
def teacher_question(v: Visit, expose_gold: bool) -> Question:
    """The harness question restricted to the student's scored pool. ``meta`` carries the oracle gold ONLY for
    simulated teachers (``expose_gold``); for every real backend it is empty (gold never goes on the wire)."""
    q: Question = v.question
    meta: dict[str, Any] = {}
    if q.kind == QKind.CHOICE:
        assert q.options is not None
        opts = [q.options[k] for k in v.pool]
        desc = [q.option_descriptions[k] for k in v.pool] if q.option_descriptions else None
        if expose_gold and q.meta.get("gold") in opts:
            meta["gold"] = q.meta["gold"]
        return Choice(q.text, opts, desc, decision_point=q.decision_point, **meta)
    if q.kind == QKind.NOUL:
        if isinstance(q.meta.get("criteria"), dict):
            meta["criteria"] = q.meta["criteria"]  # part of the wire question (see decision.jev.question_to_wire)
        if expose_gold and q.meta.get("gold") is not None:
            meta["gold"] = q.meta["gold"]
        return Noul(q.text, decision_point=q.decision_point, **meta)
    if expose_gold and q.meta.get("gold") is not None:
        meta["gold"] = q.meta["gold"]
    return Score(q.text, rubric=q.rubric, scale=q.scale, levels=q.levels, decision_point=q.decision_point, **meta)


def label_visits(visits: list[Visit], teacher: Any, ledger: Ledger, *, expose_gold: bool, workers: int = 8,
                 per_request: int = 16, max_errors: int = 50) -> Counter:
    """Ask the teacher about every visit (``Visit.teacher`` is filled in place; None when it failed).
    Visits that share a state are asked in one request (up to ``per_request``) when the backend batches."""
    reasons: Counter = Counter()
    if isinstance(teacher, LocalTeacher):
        exs = [{"state": v.state, "question": v.qtext, "options": v.options, "kind": v.kind} for v in visits]
        for v, d in zip(visits, teacher.distributions(exs, ledger)):
            v.teacher = d
        return reasons
    groups: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(visits):
        if len(v.options) > getattr(teacher, "max_options", 52):
            reasons["too_many_options"] += 1
            continue
        groups[v.state].append(i)
    per = max(1, per_request) if getattr(teacher, "batched_questions", False) else 1
    jobs = [idx[j:j + per] for idx in groups.values() for j in range(0, len(idx), per)]
    lock = threading.Lock()
    errors: list[str] = []
    abort = threading.Event()

    def run(job: list[int]) -> None:
        if abort.is_set():
            return
        qs = [teacher_question(visits[i], expose_gold) for i in job]
        try:
            res = teacher.decide(visits[job[0]].state, qs, ledger=ledger, decision_point="teacher")
        except Exception as e:  # noqa: BLE001 - network / parse failures are counted, not fatal
            with lock:
                errors.append(f"{type(e).__name__}: {str(e)[:300]}")
                reasons["error"] += len(job)
                ok = sum(1 for v in visits if v.teacher is not None)
                if len(errors) >= max_errors or (len(errors) >= min(20, len(jobs)) and ok == 0):
                    abort.set()
            return
        for i, a in zip(job, res.answers):
            if (a.raw or {}).get("parse_ok") is False:  # verbalised teachers: unparseable -> skip the row
                reasons["unparseable"] += 1
                continue
            visits[i].teacher = _norm([a.distribution.get(lab, 0.0) for lab in visits[i].labels])

    if workers <= 1 or len(jobs) <= 1:
        for j in jobs:
            run(j)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, jobs))
    if abort.is_set():
        raise RuntimeError(f"teacher {getattr(teacher, 'name', '?')} failed {len(errors)} requests; first error: "
                           f"{errors[0]}")
    if errors:
        print(f"[opd] {len(errors)} failed teacher requests; first: {errors[0]}", flush=True)
    return reasons


def visit_example(v: Visit, it: int, k: int, *, gold_alpha: float = 0.0, teacher_temperature: float = 1.0,
                  gold_fallback: bool = False) -> dict[str, Any]:
    """Teacher-labelled visit -> a row of the ``training/data.py`` format. ``render(row)[0]`` is the prompt the
    student was scored with during the rollout, byte for byte (``Visit.prompt``). ``gold_fallback``: where the
    teacher's argmax is not the JevGym oracle gold, the target is the gold one-hot (data v1's rule, §18.2), so the
    teacher's mistakes never become the argmax of a target; elsewhere ``gold_alpha`` mixes the gold in."""
    assert v.teacher is not None
    p = list(v.teacher)
    if teacher_temperature != 1.0:
        m = max(p)
        p = _norm([(x / m) ** (1.0 / teacher_temperature) for x in p])
    gold = v.labels[v.gold] if v.gold is not None else None
    fallback = bool(gold_fallback and v.gold is not None and max(range(len(p)), key=lambda j: p[j]) != v.gold)
    if fallback:
        p = [1.0 if j == v.gold else 0.0 for j in range(len(p))]
    elif gold_alpha > 0 and v.gold is not None:
        p = [(1 - gold_alpha) * x + (gold_alpha if j == v.gold else 0.0) for j, x in enumerate(p)]
    dp = v.dp or "decision"
    return {"id": f"{v.task_id}/{v.step if v.step is not None else 0}/{dp}#{k}", "source": "opd", "dp": dp,
            "kind": v.kind, "state": v.state, "question": v.qtext, "options": list(v.options),
            "target": dict(zip(v.options, p)), "q_values": None,
            "meta": {"label_source": "gold_fallback" if fallback else "opd_teacher", "iter": it, "gold": gold,
                     "event": v.ev,
                     "student_sampled": v.labels[v.sampled], "student_greedy": v.labels[v.greedy],
                     "student_conf": round(v.conf, 5), "teacher_dist": [round(x, 5) for x in v.teacher],
                     "n_options_logged": len(v.question.options) if v.question.options else len(v.options)}}


# =============================================================================== model
def _resolve(student: str, adapter: str | None) -> tuple[str, str | None]:
    from .serve_jev import resolve_checkpoint

    return resolve_checkpoint(student, adapter)


def load_student(student: str, init: str | None, *, lora_r: int, lora_alpha: int, device: str, dtype: torch.dtype,
                 gradient_checkpointing: bool = False) -> tuple[Any, Any, bool]:
    """(model, tokenizer, is_lora). ``init``: a LoRA adapter (continued, trainable) or a full-FT checkpoint dir
    (loaded as the model; a fresh LoRA goes on top unless ``lora_r == 0``). ``lora_r == 0`` = full fine-tuning."""
    from transformers import AutoTokenizer

    from .train_decision import load_causal_lm

    model_path, adapter = _resolve(student, init)
    tok = AutoTokenizer.from_pretrained(adapter or model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = load_causal_lm(model_path, dtype)
    lora = bool(adapter) or lora_r > 0
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    elif lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(base, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                                                target_modules="all-linear", task_type="CAUSAL_LM"))
    else:
        model = base
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if lora:
            model.enable_input_require_grads()
    return model.to(device), tok, lora


def redact(spec: Any) -> Any:
    """A teacher spec safe to write to disk (keys that look like credentials are dropped)."""
    if isinstance(spec, dict):
        return {k: redact(v) for k, v in spec.items() if not SECRET_KEYS.search(k)}
    if isinstance(spec, list):
        return [redact(v) for v in spec]
    return spec


def save_student(model: Any, tok: Any, path: str | Path, lora: bool, cfg: dict[str, Any],
                 extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if lora:
        model.save_pretrained(str(path), selected_adapters=["default"])
    else:
        model.save_pretrained(str(path))
    tok.save_pretrained(str(path))
    (path / "decision_config.json").write_text(json.dumps(
        {"base_model": cfg["student"], "loss": "opd", "opd_loss": cfg["loss"], "lora": lora,
         "max_len": cfg["max_len"], "init": cfg["init"], "teacher": redact(cfg["teacher_spec"]),
         "arms": [a.get("name") for a in cfg["arms"]], **(extra or {})}, indent=1))


# =============================================================================== update
def update_student(model: Any, collator: DecisionCollator, examples: list[dict[str, Any]], *, loss: str,
                   epochs: int, minibatch: int, micro_batch: int, permute: bool, grad_clip: float,
                   opt: torch.optim.Optimizer, sched: Any, rng: random.Random, max_batch_tokens: int = 0) -> dict[str, float]:
    """``epochs`` passes over ``examples`` (fresh + replay rows); ``minibatch`` rows per optimizer step,
    ``micro_batch`` rows per forward/backward (or, with ``max_batch_tokens``, length-sorted micro-batches of at most
    that many padded tokens); a fresh option permutation per row and pass when ``permute``."""
    from .decision_model import token_budget_batches

    model.train()
    dev = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    losses: list[float] = []
    gnorms: list[float] = []
    n_skip = 0
    idx = list(range(len(examples)))
    for _ in range(max(1, epochs)):
        rng.shuffle(idx)
        for s in range(0, len(idx), max(1, minibatch)):
            mb = idx[s:s + max(1, minibatch)]
            opt.zero_grad(set_to_none=True)
            prepared = [permuted(examples[i], rng if permute else None) for i in mb]
            if max_batch_tokens > 0:
                lens = [len(collator.encode(pr[0])) for pr in prepared]
                micros = token_budget_batches(lens, max_batch_tokens, max(1, micro_batch) * 8)
            else:
                micros = [list(range(u, min(u + max(1, micro_batch), len(mb))))
                          for u in range(0, len(mb), max(1, micro_batch))]
            for mi in micros:
                rows = [{"prompt": prepared[j][0], "target": prepared[j][1]} for j in mi]
                enc = collator(rows)
                z = option_logits(model, enc["input_ids"].to(dev), enc["attention_mask"].to(dev),
                                  enc["code_ids"].to(dev))
                per = opd_loss(z, enc["target"].to(dev), loss)
                total = per.sum() / len(mb)
                if not torch.isfinite(total):
                    n_skip += 1
                    continue
                total.backward()
                losses += per.detach().float().cpu().tolist()
            gn = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            if torch.isfinite(gn):
                opt.step()
            else:
                n_skip += 1
            sched.step()
            gnorms.append(float(gn))
    return {"loss": float(sum(losses) / len(losses)) if losses else float("nan"),
            "grad_norm": float(sum(gnorms) / len(gnorms)) if gnorms else float("nan"),
            "optimizer_steps": len(gnorms), "nonfinite_skipped": n_skip, "lr": float(sched.get_last_lr()[0])}


# =============================================================================== metrics
def label_stats(visits: list[Visit]) -> dict[str, float]:
    """Student-vs-teacher agreement, teacher / student accuracy vs the oracle (logged only) and KL(p_T || pi)
    on the fresh states at rollout time (before the update)."""
    def mean(xs: Sequence[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    lab = [v for v in visits if v.teacher is not None]
    am = {id(v): max(range(len(v.teacher)), key=lambda j: v.teacher[j]) for v in lab}  # type: ignore[arg-type]
    gold = [v for v in lab if v.gold is not None]
    out = {"labelled": len(lab), "label_frac": len(lab) / max(1, len(visits)),
           "agree_greedy": mean([float(am[id(v)] == v.greedy) for v in lab]),
           "agree_sampled": mean([float(am[id(v)] == v.sampled) for v in lab]),
           "teacher_acc": mean([float(am[id(v)] == v.gold) for v in gold]),
           "student_acc_greedy_labelled": mean([float(v.greedy == v.gold) for v in gold]),
           "teacher_conf": mean([v.teacher[am[id(v)]] for v in lab]),  # type: ignore[index]
           "kl_teacher_student": mean([forward_kl(v.teacher, v.z) for v in lab])}  # type: ignore[arg-type]
    by_dp: dict[str, list[Visit]] = defaultdict(list)
    for v in lab:
        by_dp[str(v.dp)].append(v)
    for dp, vs in sorted(by_dp.items()):
        out[f"agree_greedy/{dp}"] = mean([float(am[id(v)] == v.greedy) for v in vs])
        out[f"labelled/{dp}"] = len(vs)
    return out


# =============================================================================== config / CLI
def _yaml_arg(v: Any) -> Any:
    return yaml.safe_load(v) if isinstance(v, str) else v


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None, help="YAML whose keys are these flags' names (CLI flags override)")
    ap.add_argument("--student", default=None, help="base model (HF id / dir) or a full fine-tuned checkpoint dir")
    ap.add_argument("--init", default=None, help="LoRA adapter to start from ('' = fresh LoRA / full FT)")
    ap.add_argument("--teacher", default="jev-openrouter", metavar="[NAME=]SPEC",
                    help="teacher_label alias or JSON make_backend spec, or {\"type\": \"local\", \"model\": ..., "
                         "\"adapter\": ...} (serve_jev.Engine in-process)")
    ap.add_argument("--teacher-cache", default="", help="SQLite decision cache for real teachers ('' = none)")
    ap.add_argument("--executor", choices=["sim", "jevgym"], default="sim",
                    help="sim = simulated System-2 (pipeline checks only); jevgym = the --llm System-2")
    ap.add_argument("--llm", default=None, help="System-2 spec (JSON/YAML) for --executor jevgym")
    ap.add_argument("--llms", default=None, help="named extra System-2 specs (JSON/YAML dict); an arm with "
                    "`llm: <name>` runs its episodes with that executor (e.g. Flash next to the Flash-Lite default)")
    ap.add_argument("--suite", default=None, help=f"jevgym options (JSON/YAML), default {DEFAULT_SUITE}")
    ap.add_argument("--seeds", type=int, default=MIN_TRAIN_SEED, help=f"first JevGym training seed (>= {MIN_TRAIN_SEED})")
    ap.add_argument("--n-tasks", type=int, default=256, help="training task pool: seeds .. seeds + n - 1")
    ap.add_argument("--arms", default=None, help=f"harness arms (JSON/YAML list), default {DEFAULT_ARMS}")
    ap.add_argument("--max-steps", type=int, default=20, help="harness max_steps per episode")
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--tasks-per-iter", type=int, default=16)
    ap.add_argument("--rollouts-per-task", type=int, default=1)
    ap.add_argument("--rollout-temperature", type=float, default=1.0, help="0 = greedy rollouts")
    ap.add_argument("--loss", choices=list(LOSSES), default="forward_kl")
    ap.add_argument("--gold-alpha", type=float, default=0.0, help="ablation: mix the oracle one-hot into p_T")
    ap.add_argument("--gold-fallback", action="store_true", help="target = the oracle one-hot where the teacher's "
                    "argmax is not the gold (data v1's rule); gold_alpha mixes the gold in elsewhere")
    ap.add_argument("--teacher-temperature", type=float, default=1.0, help="temper p_T (T < 1 sharpens)")
    ap.add_argument("--replay-frac", type=float, default=0.5,
                    help="replay rows from the previous iteration, as a fraction of this iteration's fresh rows")
    ap.add_argument("--max-questions", type=int, default=0, help="label at most N visited decisions per iteration "
                    "(seeded subsample; 0 = all)")
    ap.add_argument("--questions-per-request", type=int, default=16, help="questions per request (batched teachers)")
    ap.add_argument("--teacher-workers", type=int, default=8)
    ap.add_argument("--budget-usd", type=float, default=0.0, help="stop when the teacher ledger passes this (0 = off)")
    ap.add_argument("--lora-r", type=int, default=32, help="0 = full fine-tuning")
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--epochs-per-iter", type=int, default=1)
    ap.add_argument("--minibatch", type=int, default=32, help="rows per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=8, help="rows per forward/backward")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-batch-tokens", type=int, default=0, help="update micro-batches by padded-token budget "
                    "(length-sorted; 0 = --micro-batch rows); use without --gradient-checkpointing")
    ap.add_argument("--no-permute", action="store_true", help="fixed option order in the update")
    ap.add_argument("--workers", type=int, default=8, help="concurrent rollout episodes")
    ap.add_argument("--decider-batch", type=int, default=16, help="prompts per student forward during rollouts")
    ap.add_argument("--price-model", default=None, help="prices-file key for the student's input tokens")
    ap.add_argument("--eval-every", type=int, default=0, help="greedy dev rollouts every N iterations (0 = never)")
    ap.add_argument("--eval-n", type=int, default=16)
    ap.add_argument("--eval-arms", default=None, help="comma-separated arm names evaluated greedily (default: all)")
    ap.add_argument("--eval-seed0", type=int, default=MIN_DEV_SEED,
                    help=f"dev seeds (>= {MIN_DEV_SEED}, disjoint from training; 1000-4999 = training-side dev range; "
                         "never the evaluation / retired ranges; avoid the calibration range 5000-5999)")
    ap.add_argument("--strict-seeds", action="store_true", help="refuse greedy dev rollouts outside the development "
                    "seeds 0-999 / 1000-4999 (round-5 rule; default: the calibration range is only warned about)")
    ap.add_argument("--save-every", type=int, default=1, help="checkpoint (+ optimizer state) every N iterations")
    ap.add_argument("--out", default="checkpoints/r9-opd")
    ap.add_argument("--resume", default=None, help="ckpt-XXXXX dir, or 'auto' = latest under --out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None, help="default: on CUDA")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--behaviour-frozen", action="store_true", help="control (A7.13 (iii)): the rollouts are driven "
                    "by a frozen copy of --init (off-policy teacher labels on the init policy's states); the trained "
                    "adapter only learns, the greedy dev evaluations use it")
    ap.add_argument("--log-rollouts", action="store_true", help="also write rollouts.jsonl (one row per episode)")
    ap.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = build_parser()
    pre, _ = ap.parse_known_args(argv)
    if pre.config:
        raw = yaml.safe_load(Path(pre.config).read_text()) or {}
        known = {a.dest for a in ap._actions}
        bad = sorted(set(raw) - known)
        if bad:
            raise SystemExit(f"{pre.config}: unknown keys {bad}; valid keys are the flags of opd.py")
        ap.set_defaults(**raw)
    a = ap.parse_args(argv)
    if not a.student:
        raise SystemExit("--student is required (or `student:` in --config)")
    a.init = a.init or None
    a.suite = {**DEFAULT_SUITE, **(_yaml_arg(a.suite) or {})}
    if {"n", "seed0"} & set(a.suite):
        raise SystemExit("--suite must not set n / seed0: use --n-tasks and --seeds (training seeds >= 10000)")
    a.arms = _yaml_arg(a.arms) or DEFAULT_ARMS
    a.llm = _yaml_arg(a.llm)
    if a.executor == "jevgym" and not a.llm:
        raise SystemExit("--executor jevgym needs --llm (the System-2 spec)")
    if a.executor == "sim":
        a.llm = {"provider": "simulated", "model": "scripted", "error_rate": 0.1, "simulated_latency_ms": 0,
                 **(a.llm or {})}
    from ..bench.registry import CALIBRATION_RANGE, overlaps
    from .seed_guard import NEVER_RUN, check_dev_pool, check_disjoint, check_train_pool

    try:  # training/seed_guard.py: train >= 10000 minus evaluation / retired / calibration / Stage-2 ranges
        check_train_pool(a.seeds, a.n_tasks, "--seeds/--n-tasks")
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if a.eval_every:
        if a.eval_seed0 < MIN_DEV_SEED or overlaps(a.eval_seed0, a.eval_n, ((a.seeds, a.seeds + a.n_tasks - 1),)):
            raise SystemExit(f"--eval-seed0 {a.eval_seed0}: dev seeds must be >= {MIN_DEV_SEED} and disjoint from the "
                             f"training seeds {a.seeds}..{a.seeds + a.n_tasks - 1}")
        if overlaps(a.eval_seed0, a.eval_n, NEVER_RUN):
            raise SystemExit(f"--eval-seed0 {a.eval_seed0}: the greedy dev rollouts would touch the evaluation / retired "
                             f"/ Stage-2 seeds {overlaps(a.eval_seed0, a.eval_n, NEVER_RUN)}")
        if a.strict_seeds:  # JevAgentRL (round 5): greedy evaluations on the development seeds 1000-4999 only
            try:
                check_disjoint((a.eval_seed0, a.eval_n), (a.seeds, a.n_tasks), "--eval-seed0 vs the training seeds")
                check_dev_pool(a.eval_seed0, a.eval_n, "--eval-seed0/--eval-n")
            except ValueError as e:
                raise SystemExit(str(e)) from None
        elif overlaps(a.eval_seed0, a.eval_n, (CALIBRATION_RANGE,)):
            import sys

            print(f"[opd] WARNING: --eval-seed0 {a.eval_seed0} puts the greedy dev rollouts in the calibration range "
                  f"{CALIBRATION_RANGE} (A6.5 shadow arms use 5000-5149); use the training-side dev range 1000-4999, "
                  "e.g. --eval-seed0 1000", file=sys.stderr)
    a.llms = _yaml_arg(a.llms) or {}
    if isinstance(a.eval_arms, str):
        a.eval_arms = [x.strip() for x in a.eval_arms.split(",") if x.strip()]
    a.eval_arms = list(a.eval_arms or [])
    unknown = sorted(set(a.eval_arms) - {arm.get("name") for arm in a.arms})
    if unknown:
        raise SystemExit(f"--eval-arms {unknown}: not among the arms {[arm.get('name') for arm in a.arms]}")
    for arm in a.arms:
        name = arm.get("llm")
        if name is not None and name not in a.llms:
            raise SystemExit(f"arm {arm.get('name')!r}: llm {name!r} is not a key of --llms {sorted(a.llms)}")
    name, spec = parse_teacher(a.teacher, 0)
    a.teacher_name, a.teacher_spec = name, spec
    return a


def _latest_ckpt(out: Path) -> Path | None:
    cks = sorted(p for p in out.glob("ckpt-*") if (p / "trainer_state.pt").exists())
    return cks[-1] if cks else None


# =============================================================================== training loop
def train(a: argparse.Namespace) -> Path:
    from dotenv import load_dotenv

    from ..bench.suites import load_suite
    from ..harness.agent import JevAgent
    from ..llm import make_llm

    load_dotenv()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = a.bf16 if a.bf16 is not None else device.startswith("cuda")
    dtype = torch.bfloat16 if bf16 else torch.float32
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "labelled").mkdir(exist_ok=True)
    cfg = {k: v for k, v in vars(a).items() if k not in ("teacher", "teacher_spec", "llm", "llms")}
    cfg.update(teacher=f"{a.teacher_name}={json.dumps(redact(a.teacher_spec))}", teacher_spec=a.teacher_spec,
               llm=redact(a.llm), llms=redact(a.llms))  # raw --teacher / --llm strings may carry credentials
    (out / "opd_config.yaml").write_text(yaml.safe_dump(redact(cfg), sort_keys=False, default_flow_style=False))

    resume = a.resume
    if resume == "auto":
        ck = _latest_ckpt(out)
        resume = str(ck) if ck else None
        print(f"[opd] resume auto -> {resume}", flush=True)
    arms = [HarnessConfig(**{"max_steps": a.max_steps, **arm}).validate_modes() for arm in a.arms]
    for arm in arms:
        if not arm.uses_s1():
            raise SystemExit(f"arm {arm.name!r} routes no decision point to System-1: the student never acts")
    model, tok, lora = load_student(a.student, resume or a.init, lora_r=a.lora_r, lora_alpha=a.lora_alpha,
                                    device=device, dtype=dtype, gradient_checkpointing=a.gradient_checkpointing)
    if a.behaviour_frozen:  # frozen-policy distillation control (A7.13 (iii)): rollouts by a frozen copy of --init
        if not (lora and a.init):
            raise SystemExit("--behaviour-frozen needs a LoRA --init adapter (the behaviour policy held fixed)")
        model.load_adapter(a.init, adapter_name="behaviour", is_trainable=False)
        model.set_adapter("default")
    common = dict(max_len=a.max_len, batch_size=a.decider_batch, price_model=a.price_model, name="opd-student")
    greedy_rollouts = a.rollout_temperature <= 0
    student = OPDDecider(model, tok, arms[0], temperature=max(a.rollout_temperature, 1e-6), greedy=greedy_rollouts,
                         **common)
    evaluator = OPDDecider(model, tok, arms[0], greedy=True, **common)
    collator = student.collator
    executor = make_llm(a.llm)
    named = {name: make_llm(spec) for name, spec in a.llms.items()}
    arm_llm = {arm.get("name", "llm_only"): arm.get("llm") for arm in a.arms}

    def make_agent(arm: HarnessConfig, s1: TorchDecider) -> Any:  # per-arm executor (`llm: <name of --llms>`)
        ex = named[arm_llm[arm.name]] if arm_llm.get(arm.name) else executor
        return JevAgent(ex, arm, s1=s1, system_prompt=a.system_prompt)

    eval_arms = [arm for arm in arms if not a.eval_arms or arm.name in a.eval_arms]

    train_tasks = load_suite("jevgym", n=a.n_tasks, seed0=a.seeds, **a.suite)
    eval_tasks = load_suite("jevgym", n=a.eval_n, seed0=a.eval_seed0, **a.suite) if a.eval_every else []
    teacher = make_teacher(a.teacher_spec, a.teacher_cache or None)
    expose_gold = a.teacher_spec["type"] in SIMULATED_TYPES
    ledger = Ledger()
    reward_cfg = RewardCfg()  # episode rewards are logged by run_episodes, never used for learning

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("the student has no trainable parameters")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    warm = max(1, a.warmup_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    rng = random.Random(a.seed)
    order = list(range(len(train_tasks)))
    rng.shuffle(order)
    cursor, start_it = 0, 0
    cost_prev, calls_prev = 0.0, 0
    replay: list[dict[str, Any]] = []
    if resume:
        st = torch.load(Path(resume) / "trainer_state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        start_it, cursor, order = st["iteration"], st["cursor"], st["order"]
        rng.setstate(st["rng"])
        cost_prev, calls_prev = float(st.get("teacher_cost_total", 0.0)), int(st.get("teacher_calls_total", 0))
        rf = st.get("replay_file")
        if rf and Path(rf).exists():
            replay = read_jsonl(rf)
        print(f"[opd] resumed from {resume}: iteration {start_it}, teacher spend so far ${cost_prev:.4f}", flush=True)
    metrics_path = out / "metrics.jsonl"

    def teacher_totals() -> tuple[float, int]:
        s = ledger_summary(ledger)
        return cost_prev + float(s["cost_usd"]), calls_prev + int(s["calls"])

    def log(row: dict[str, Any]) -> None:
        row = {k: _clean(v) if isinstance(v, float) else v for k, v in row.items()}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        keys = ["iter", "success", "cost_usd", "decisions", "labelled", "agree_greedy", "kl_teacher_student",
                "loss", "teacher_cost_iter", "teacher_cost_total"]
        print("[opd] " + " ".join(f"{k}={row[k]:.4g}" if isinstance(row.get(k), float) else f"{k}={row.get(k)}"
                                  for k in keys if k in row), flush=True)

    def evaluate(it: int) -> dict[str, float]:
        res: dict[str, float] = {}
        if not eval_tasks:
            return res
        model.eval()
        for arm in eval_arms:
            jobs = [(gi, f, 0, arm) for gi, f in enumerate(eval_tasks)]
            eps_ = run_episodes(jobs, evaluator, make_agent, reward_cfg, a.workers, ("eval", it))
            st = rollout_stats(eps_)
            for k in ("success", "cost_usd", "llm_calls", "s1_calls", "steps", "dec_acc_greedy", "irreversible_rate",
                      "error_rate"):
                res[f"eval/{arm.name}/{k.replace('dec_acc_greedy', 'dec_acc')}"] = st[k]
        return res

    stop_reason = "iterations"
    cost_total, calls_total = teacher_totals()
    it = start_it
    todo = range(start_it + 1, a.iterations + 1)
    if a.budget_usd > 0 and cost_total >= a.budget_usd:
        print(f"[opd] teacher budget ${a.budget_usd} already spent (${cost_total:.4f}): nothing to do", flush=True)
        stop_reason, todo = "budget", range(0)
    elif a.eval_every and start_it == 0:
        log({"iter": 0, **evaluate(0)})
    for it in todo:
        picked = []
        for _ in range(min(a.tasks_per_iter, len(order))):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            picked.append(order[cursor])
            cursor += 1
        jobs = [(gi, train_tasks[ti], g, arms[gi % len(arms)]) for gi, ti in enumerate(picked)
                for g in range(a.rollouts_per_task)]
        # 1. rollout: the student's own policy drives the agent
        t0 = time.perf_counter()
        model.eval()
        if a.behaviour_frozen:
            from .decision_model import use_adapter

            with use_adapter(model, "behaviour"):
                episodes: list[Episode] = run_episodes(jobs, student, make_agent, reward_cfg, a.workers, (a.seed, it))
        else:
            episodes = run_episodes(jobs, student, make_agent, reward_cfg, a.workers, (a.seed, it))
        t_roll = time.perf_counter() - t0
        errors = [e.error for e in episodes if e.error]
        if errors and len(errors) == len(episodes):
            raise RuntimeError(f"every rollout failed; first error:\n{errors[0]}")
        visits: list[Visit] = []
        for e in episodes:
            for v in e.records:
                v.task_id = e.task_id
                visits.append(v)  # type: ignore[arg-type]
        row: dict[str, Any] = {"iter": it, "tasks": len(picked), **rollout_stats(episodes), "time_rollout_s": t_roll}
        for arm in arms:  # per-arm rollout success / decision accuracy of the sampled policy
            ea = [e for e in episodes if e.arm == arm.name]
            if ea:
                sa = rollout_stats(ea)
                row.update({f"arm/{arm.name}/{k}": sa[k] for k in ("success", "dec_acc", "dec_acc_greedy", "cost_usd")})
        # 2. teacher: the same questions at the same states
        t1 = time.perf_counter()
        asked = visits
        if a.max_questions and len(visits) > a.max_questions:
            asked = rng.sample(visits, a.max_questions)
        n0 = len(ledger.records)
        reasons = label_visits(asked, teacher, ledger, expose_gold=expose_gold, workers=a.teacher_workers,
                               per_request=a.questions_per_request) if asked else Counter()
        fresh_rec = ledger.records[n0:]
        cost_total, calls_total = teacher_totals()
        row.update({"asked": len(asked), **label_stats(asked), "teacher_skipped": dict(reasons),
                    "teacher_calls_iter": len(fresh_rec), "teacher_calls_total": calls_total,
                    "teacher_cost_iter": float(sum(r.cost_usd for r in fresh_rec)), "teacher_cost_total": cost_total,
                    "teacher_cost_cached": float(sum(r.cost_usd for r in fresh_rec if r.cached)),
                    "teacher_input_tokens_iter": int(sum(r.input_tokens for r in fresh_rec)),
                    "teacher_unpriced": sorted({str(r.model) for r in fresh_rec if not r.priced}),
                    "time_label_s": time.perf_counter() - t1})
        fresh = [visit_example(v, it, k, gold_alpha=a.gold_alpha, teacher_temperature=a.teacher_temperature,
                               gold_fallback=a.gold_fallback)
                 for k, v in enumerate(asked) if v.teacher is not None]
        row["gold_fallback_n"] = sum(r["meta"]["label_source"] == "gold_fallback" for r in fresh)
        labelled_file = out / "labelled" / f"iter-{it:05d}.jsonl"
        write_jsonl(fresh, labelled_file)
        # 3. update on fresh rows + a replay of the previous iteration's rows
        n_rep = min(len(replay), int(round(a.replay_frac * len(fresh)))) if fresh else 0
        batch = fresh + (rng.sample(replay, n_rep) if n_rep else [])
        row.update({"fresh_n": len(fresh), "replay_n": n_rep, "train_n": len(batch)})
        t2 = time.perf_counter()
        if batch:
            row.update(update_student(model, collator, batch, loss=a.loss, epochs=a.epochs_per_iter,
                                      minibatch=a.minibatch, micro_batch=a.micro_batch, permute=not a.no_permute,
                                      grad_clip=a.grad_clip, opt=opt, sched=sched, rng=rng,
                                      max_batch_tokens=a.max_batch_tokens))
        else:
            print(f"[opd] iter {it}: no labelled decision (teacher skipped: {dict(reasons)})", flush=True)
        row["time_update_s"] = time.perf_counter() - t2
        replay = fresh
        if a.log_rollouts:
            with open(out / "rollouts.jsonl", "a") as f:
                for e in episodes:
                    f.write(json.dumps({"iter": it, "task_id": e.task_id, "arm": e.arm, "success": e.success,
                                        "cost_usd": e.cost_usd, "llm_calls": e.llm_calls, "steps": e.n_steps,
                                        "decisions": len(e.records), "error": e.error}) + "\n")
        if errors:
            row["first_error"] = errors[0][:500]
        if a.budget_usd > 0 and cost_total >= a.budget_usd:
            stop_reason = "budget"
            row["stop_reason"] = "budget"
            print(f"[opd] teacher budget reached: ${cost_total:.4f} >= ${a.budget_usd}; stopping after iteration "
                  f"{it}", flush=True)
        last = it == a.iterations or stop_reason == "budget"
        if a.eval_every and (it % a.eval_every == 0 or last):
            row.update(evaluate(it))
        log(row)
        if a.save_every and (it % a.save_every == 0 or last):
            ck = out / f"ckpt-{it:05d}"
            save_student(model, tok, ck, lora, cfg, {"iteration": it})
            torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "iteration": it,
                        "cursor": cursor, "order": order, "rng": rng.getstate(), "teacher_cost_total": cost_total,
                        "teacher_calls_total": calls_total, "replay_file": str(labelled_file)},
                       ck / "trainer_state.pt")
        if stop_reason == "budget":
            break
    cost_total, calls_total = teacher_totals()
    save_student(model, tok, out, lora, cfg, {"iteration": it, "stop_reason": stop_reason})
    summary = {"out": str(out), "iterations_done": it, "stop_reason": stop_reason, "student": a.student,
               "init": a.init, "teacher": redact(a.teacher_spec), "executor": redact(a.llm), "loss": a.loss,
               "teacher_ledger": {**ledger_summary(ledger), "cost_usd_total_incl_resumed": cost_total,
                                  "calls_total_incl_resumed": calls_total}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("out", "iterations_done", "stop_reason")}
                     | {"teacher_cost_usd": cost_total}), flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    if a.print_config:
        print(yaml.safe_dump({**{k: v for k, v in vars(a).items() if k not in ("print_config", "teacher_spec")},
                              "teacher_spec": redact(a.teacher_spec)}, sort_keys=False))
        return
    train(a)


if __name__ == "__main__":
    main()
