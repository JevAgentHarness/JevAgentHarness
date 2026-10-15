"""Carry ``Sample.train_metadata`` from Miles' rollout side to the custom loss on the FSDP trainer ranks.

Miles builds ``train_data["metadata"]`` from ``Sample.train_metadata`` (``convert_samples_to_train_data``), but two
fixed key lists drop it on the way to the loss:

1. ``miles.ray.rollout.train_data_conversion._package_shards`` (rollout executor) copies only a fixed list of keys into
   each data-parallel shard -- ``metadata`` is not among them;
2. ``miles.backends.training_utils.torch_native.actor.TRAIN_KEYS`` (FSDP / torchtitan trainer) lists the keys fetched
   into each micro-batch -- ``metadata`` is not among them either.

``install()`` registers an import hook that patches both modules the moment they are imported (and at once if they
already are): the shard packer also partitions ``metadata`` like every per-sample key, and ``"metadata"`` is appended
to ``TRAIN_KEYS``. It is installed in every Python process of the Miles container by ``_startup/sitecustomize.py`` on
PYTHONPATH (Ray workers inherit it) and by a ``.pth`` line (``launch.install_keys_patch``). Nothing else changes: one
more list of dicts travels with the samples.
"""
from __future__ import annotations

import importlib.abc
import sys
from typing import Any, Callable

TARGET = "miles.backends.training_utils.torch_native.actor"
CONVERSION = "miles.ray.rollout.train_data_conversion"
EXTRA_KEYS = ("metadata",)                          # partitioned into every data-parallel shard
TRAINER_KEYS = ("metadata", "rollout_mask_sums")    # fetched into every micro-batch


def full_parallel_config(ps: Any) -> dict[str, int]:
    """The schedule fields Miles' rollout side needs to plan compact rollouts (``dp_schedule.SCHEDULE_CONFIG_KEYS``)."""
    return {"dp_size": int(ps.intra_dp.size), "cp_size": int(getattr(ps.cp, "size", 1) or 1),
            "vpp_size": int(getattr(ps, "vpp_size", 1) or 1),
            "microbatch_group_size_per_vp_stage": int(getattr(ps, "microbatch_group_size_per_vp_stage", 1) or 1)}


def patch_actor(mod: Any) -> None:
    """(1) ``metadata`` (the decision's codes / target / event / behaviour log-prob) and ``rollout_mask_sums`` (the
    per-rollout denominators of ``sum_of_sample_mean``: a GRPO episode's decisions average to one episode term, as the
    Megatron backend fetches them) join the micro-batch keys. (2) The FSDP backend advertises the full parallel
    config, so Miles' rollout side plans the compact-rollout schedule (global batch counted in rollouts, an episode's
    decisions kept in one step) as it does for Megatron; without it the trainer's sample-count schedule fails on
    episodes of unequal length. SFT / OPD samples are their own rollouts, so nothing changes for them."""
    keys = getattr(mod, "TRAIN_KEYS", None)
    if isinstance(keys, list):
        for k in TRAINER_KEYS:
            if k not in keys:
                keys.append(k)
    cls0 = getattr(mod, "TorchNativeTrainRayActor", None)
    gps = getattr(mod, "get_parallel_state", None)
    if cls0 is not None and gps is not None and not getattr(cls0, "_jev_parallel_config", False):
        cls0.train_parallel_config = property(lambda self: full_parallel_config(gps()))
        cls0._jev_parallel_config = True
    # The decision loss never reads the trainer's "old" full-vocabulary log-probabilities of the response token (SFT /
    # OPD need none; GRPO's ratio uses the stored behaviour log-probability of the restricted event), so the separate
    # forward-only pass over every sample that precedes each update is skipped for the custom loss.
    cls = getattr(mod, "TorchNativeTrainRayActor", None)
    orig = getattr(cls, "_log_probs", None) if cls is not None else None
    if orig is not None and not getattr(orig, "_jev_patched", False):
        def _log_probs(self: Any, runner: Any, data_iterator: Any, num_microbatches: Any, store_prefix: str = "") -> Any:
            if getattr(self.args, "loss_type", None) == "custom_loss" and not store_prefix:
                return {}
            return orig(self, runner, data_iterator, num_microbatches, store_prefix)

        _log_probs._jev_patched = True  # type: ignore[attr-defined]
        cls._log_probs = _log_probs


def patch_conversion(mod: Any) -> None:
    orig = getattr(mod, "_package_shards", None)
    if orig is None or getattr(orig, "_jev_patched", False):
        return

    def _package_shards(args: Any, data: dict, partitions: Any) -> list:
        shards = orig(args, data, partitions)
        for key in EXTRA_KEYS:
            if key in data and data[key] is not None:
                for shard, part in zip(shards, partitions):
                    shard[key] = [data[key][j] for j in part]
        return shards

    _package_shards._jev_patched = True  # type: ignore[attr-defined]
    mod._package_shards = _package_shards


FSDP_ACTOR = "miles.backends.fsdp_utils.actor"


def decision_positions(total_lengths: Any, response_lengths: Any) -> list[int]:
    """Packed (thd) positions whose logits predict each sample's first response token."""
    out, end = [], 0
    for L, R in zip(total_lengths, response_lengths):
        end += int(L)
        out.append(end - int(R) - 1)
    return out


def patch_fsdp_actor(mod: Any) -> None:
    """For the custom decision loss, the language-model head runs only at the decision positions
    (``logits_to_keep`` = those packed positions): the loss reads one row per sample, and full-vocabulary logits of
    every token of a micro-batch dominated the activation memory on 24 GB GPUs. The batch is flagged so the loss
    knows the logits are already one row per sample (``loss.decision_rows``)."""
    cls = getattr(mod, "FSDPTrainRayActor", None)
    orig = getattr(cls, "_get_model_inputs_args", None) if cls is not None else None
    if orig is None or getattr(orig, "_jev_patched", False):
        return
    import os

    def _get_model_inputs_args(self: Any, batch: dict) -> dict:
        args = orig(self, batch)
        if (getattr(self.args, "loss_type", None) == "custom_loss" and getattr(self.args, "qkv_format", "thd") == "thd"
                and os.environ.get("JEV_MILES_ROWS_ONLY", "1") == "1" and batch.get("total_lengths") is not None):
            import torch

            idx = decision_positions(batch["total_lengths"], batch["response_lengths"])
            args["logits_to_keep"] = torch.tensor(idx, dtype=torch.long, device=batch["tokens"].device)
            batch["_jev_rows_only"] = True
        return args

    _get_model_inputs_args._jev_patched = True  # type: ignore[attr-defined]
    cls._get_model_inputs_args = _get_model_inputs_args


PATCHES: dict[str, Callable[[Any], None]] = {TARGET: patch_actor, CONVERSION: patch_conversion,
                                              FSDP_ACTOR: patch_fsdp_actor}


def patch_module(mod: Any) -> None:  # kept for callers of the first version
    PATCHES.get(getattr(mod, "__name__", ""), patch_actor)(mod)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        fn = PATCHES.get(fullname)
        if fn is None:
            return None
        for finder in sys.meta_path:
            if finder is self or isinstance(finder, _Finder):
                continue
            spec = getattr(finder, "find_spec", lambda *a: None)(fullname, path, target)
            if spec is not None and spec.loader is not None:
                loader = spec.loader
                exec_module = loader.exec_module

                def _exec(module: Any, _orig: Any = exec_module, _fn: Any = fn) -> None:
                    _orig(module)
                    _fn(module)

                loader.exec_module = _exec  # type: ignore[method-assign]
                return spec
        return None


def install() -> None:
    for name, fn in PATCHES.items():
        if name in sys.modules:
            fn(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


def pth_line() -> str:
    return "import jevharness.training.miles_port.keys_patch as _jk; _jk.install()\n"
