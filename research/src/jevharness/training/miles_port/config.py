"""Stage configuration of the Miles chain (``configs/train/miles_*.yaml``).

One YAML file per stage (SFT, OPD, GRPO). ``launch.py`` resolves ``base`` / ``init`` ("auto" -> the base rule's choice
and the chain's retained checkpoint), writes the resolved copy next to the stage outputs and points
``JEV_MILES_CONFIG`` at it; the rollout functions (rollout manager process) and the loss (trainer ranks) read it from
there. Seeds are checked against ``training/seed_guard.py`` when the config is loaded.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..grpo_online import AlgoCfg, RewardCfg

CFG_ENV = "JEV_MILES_CONFIG"


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MilesCfg(_Cfg):
    """Trainer / engine flags of the Miles command line (``launch.miles_args``)."""
    actor_gpus: int = 4
    rollout_gpus: int = 0               # 0 = no engine (SFT, --debug-train-only)
    colocate: bool = False
    lr: float = 1e-5
    min_lr: float = 0.0
    lr_decay_style: Literal["cosine", "constant", "linear"] = "cosine"
    warmup_fraction: float = 0.03
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    clip_grad: float = 1.0
    global_batch_size: int = 64         # rollouts (= decisions for SFT / OPD, episodes for GRPO) per optimizer step
    max_tokens_per_gpu: int = 8192
    gradient_checkpointing: bool = True
    attn_implementation: str = "flash_attention_2"
    sglang_mem_fraction: float = 0.6
    sglang_attention_backend: str = "flashinfer"
    save_interval: int = 0              # rollouts between checkpoints (0 = only at the end)
    keep_steps: list[int] = Field(default_factory=list)   # rollouts whose checkpoints are exported (continuations)
    seed: int = 1234
    extra_args: list[str] = Field(default_factory=list)


class SFTCfg(_Cfg):
    train: list[str]
    replay: str | None = None
    replay_frac: float = 0.45           # seeded fraction of the replay file mixed into train (v1: 0.45 x When2Call)
    epochs: int = 2
    batch_rows: int = 64                # rows per rollout (= one optimizer step)
    permute: bool = True
    brier_weight: float = 0.5


class RolloutCfg(_Cfg):
    suite: dict[str, Any]
    seed0: int
    n_tasks: int
    tasks_per_iter: int
    group_size: int = 1
    iterations: int
    arms: list[dict[str, Any]]
    executor: dict[str, Any]
    executors: dict[str, dict[str, Any]] = Field(default_factory=dict)
    max_steps: int = 20
    workers: int = 16
    temperature: float = 1.0
    decider_batch: int = 32
    system_prompt: str | None = None


class OPDCfg(_Cfg):
    teacher: str = "jev-openrouter"
    teacher_cache: str | None = None
    teacher_temperature: float = 1.0
    gold_alpha: float = 0.3
    gold_fallback: bool = True
    loss: Literal["forward_kl", "reverse_kl", "jsd"] = "forward_kl"
    fresh_rows: int = 512               # labelled visits per iteration (seeded subsample; shortfall filled from replay)
    replay_rows: int = 256              # rows of the previous iteration's labelled set
    questions_per_request: int = 16
    teacher_workers: int = 8
    budget_usd: float = 8.0


class GRPOCfg(_Cfg):
    algo: AlgoCfg = Field(default_factory=AlgoCfg)
    reward: RewardCfg = Field(default_factory=RewardCfg)


class StageConfig(_Cfg):
    name: str
    stage: Literal["sft", "opd", "grpo"]
    base: str                           # HF dir / id of the base model (or "auto": the base rule)
    init: str | None = None             # HF dir the stage starts from (None = base; "auto" = the chain's choice)
    out: str
    max_len: int = 4096
    seed: int = 0
    miles: MilesCfg = Field(default_factory=MilesCfg)
    sft: SFTCfg | None = None
    rollout: RolloutCfg | None = None
    opd: OPDCfg | None = None
    grpo: GRPOCfg | None = None

    def check(self) -> "StageConfig":
        if self.stage == "sft" and self.sft is None:
            raise ValueError("stage sft needs an `sft` section")
        if self.stage in ("opd", "grpo"):
            if self.rollout is None:
                raise ValueError(f"stage {self.stage} needs a `rollout` section")
            if self.stage == "opd" and self.opd is None:
                raise ValueError("stage opd needs an `opd` section")
            if self.stage == "grpo" and self.grpo is None:
                raise ValueError("stage grpo needs a `grpo` section")
            from ..seed_guard import check_train_pool

            check_train_pool(self.rollout.seed0, self.rollout.n_tasks, f"{self.name}: rollout pool")
            need = self.rollout.tasks_per_iter * self.rollout.iterations
            if need > self.rollout.n_tasks:
                raise ValueError(f"{self.name}: {need} tasks needed, pool has {self.rollout.n_tasks} (no task reuse)")
        return self

    def rollouts_per_iter(self) -> int:
        if self.stage == "sft":
            assert self.sft is not None
            return self.sft.batch_rows
        assert self.rollout is not None
        if self.stage == "grpo":
            return self.rollout.tasks_per_iter * self.rollout.group_size
        assert self.opd is not None
        return self.opd.fresh_rows + self.opd.replay_rows


def load_stage_config(path: str | Path | None = None) -> StageConfig:
    p = path or os.environ.get(CFG_ENV)
    if not p:
        raise RuntimeError(f"no stage config: pass a path or set {CFG_ENV}")
    raw = yaml.safe_load(Path(p).read_text())
    return StageConfig.model_validate(raw).check()
