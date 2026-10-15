"""Run one stage of the Miles chain inside the Miles image (``scripts/aws/gpu_round6_stages.sh`` calls it).

  python -m jevharness.training.miles_port.launch run --config configs/train/miles_sft_v2.yaml [--dry-run]
  python -m jevharness.training.miles_port.launch args --config ...          # print the Miles command line only

``run``: resolve ``base`` / ``init`` ("auto": ``<chain>/base.json`` written by the base rule, ``<chain>/init_<stage>.txt``
written by the chain), write the resolved config to ``<out>/stage_config.yaml``, install the trainer key patch
(``keys_patch``, a ``.pth`` line in site-packages), start a local Ray head with every GPU (the API keys and
``JEV_MILES_*`` variables are in this process's environment, which Ray workers inherit), run Miles ``train.py`` /
``train_async.py`` with the FSDP2 backend, the decision loss and the stage's rollout function, then export the kept
checkpoints to HF directories (``export.py``) that ``training/serve_jev.py`` loads unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import site
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .config import CFG_ENV, StageConfig, load_stage_config

MILES_ROOT = os.environ.get("MILES_ROOT", "/root/miles")
CHAIN_DIR = os.environ.get("JEV_MILES_CHAIN", "runs/gpu_round6_miles")
LOSS_PATH = "jevharness.training.miles_port.loss.jev_decision_loss"
ROLLOUT = {"sft": "jevharness.training.miles_port.rollout.sft_rollout",
           "opd": "jevharness.training.miles_port.rollout.harness_rollout",
           "grpo": "jevharness.training.miles_port.rollout.harness_rollout"}
DATA_SOURCE = "jevharness.training.miles_port.rollout.NullDataSource"


def resolve(cfg: StageConfig, chain: str = CHAIN_DIR) -> StageConfig:
    raw = cfg.model_dump()
    if raw["base"] == "auto":
        raw["base"] = json.loads((Path(chain) / "base.json").read_text())["path"]
    if raw.get("init") == "auto":
        raw["init"] = (Path(chain) / f"init_{cfg.stage}.txt").read_text().strip()
    if not raw.get("init"):
        raw["init"] = raw["base"]
    return StageConfig.model_validate(raw).check()


def num_rollouts(cfg: StageConfig) -> int:
    if cfg.stage == "sft":
        from .rollout import sft_num_rollouts, sft_rows

        assert cfg.sft is not None
        return sft_num_rollouts(len(sft_rows(cfg)), cfg.sft.epochs, cfg.sft.batch_rows)
    assert cfg.rollout is not None
    return cfg.rollout.iterations


def miles_args(cfg: StageConfig, n_rollout: int) -> list[str]:
    m = cfg.miles
    out = Path(cfg.out)
    if cfg.stage == "grpo":
        assert cfg.rollout is not None
        rbs, nsp = cfg.rollout.tasks_per_iter, cfg.rollout.group_size
    else:
        rbs, nsp = cfg.rollouts_per_iter(), 1
    if (rbs * nsp) % m.global_batch_size:
        raise ValueError(f"{cfg.name}: {rbs * nsp} rollouts per iteration is not a multiple of global_batch_size "
                         f"{m.global_batch_size}")
    if m.global_batch_size % m.actor_gpus:
        raise ValueError(f"{cfg.name}: global_batch_size {m.global_batch_size} is not a multiple of the "
                         f"{m.actor_gpus} data-parallel ranks")
    a = ["--train-backend", "fsdp", "--hf-checkpoint", str(cfg.init),
         "--rollout-function-path", ROLLOUT[cfg.stage], "--data-source-path", DATA_SOURCE,
         "--loss-type", "custom_loss", "--custom-loss-function-path", LOSS_PATH,
         "--disable-compute-advantages-and-returns", "--disable-rewards-normalization",
         "--num-rollout", str(n_rollout), "--rollout-batch-size", str(rbs), "--n-samples-per-prompt", str(nsp),
         "--global-batch-size", str(m.global_batch_size),
         "--optimizer", "adam", "--lr", str(m.lr), "--min-lr", str(m.min_lr), "--lr-decay-style", m.lr_decay_style,
         "--lr-warmup-fraction", str(m.warmup_fraction), "--weight-decay", str(m.weight_decay),
         "--adam-beta1", str(m.adam_beta1), "--adam-beta2", str(m.adam_beta2), "--clip-grad", str(m.clip_grad),
         "--use-dynamic-batch-size", "--max-tokens-per-gpu", str(m.max_tokens_per_gpu),
         "--attn-implementation", m.attn_implementation,
         "--actor-num-nodes", "1", "--actor-num-gpus-per-node", str(m.actor_gpus),
         "--num-gpus-per-node", str(m.actor_gpus + (0 if m.colocate else m.rollout_gpus)),
         "--save", str(out / "dcp"), "--load", str(out / "dcp"), "--seed", str(m.seed)]
    # Miles saves only when a save interval is set (then always at the last rollout as well): 0 = the end only
    if os.environ.get("JEV_MILES_NO_SAVE") == "1":   # timing runs only
        a = [x for i, x in enumerate(a) if x != "--save" and (i == 0 or a[i - 1] != "--save")]
    else:
        a += ["--save-interval", str(m.save_interval or n_rollout)]
    if m.gradient_checkpointing:
        a += ["--gradient-checkpointing"]
    if cfg.stage == "sft":
        a += ["--debug-train-only"]
    else:
        a += ["--rollout-num-gpus-per-engine", "1", "--sglang-mem-fraction-static", str(m.sglang_mem_fraction),
              "--sglang-attention-backend", m.sglang_attention_backend]
        if m.colocate:
            a += ["--colocate"]          # engines on the training GPUs; Miles offloads the trainer between phases
        else:
            if m.rollout_gpus < 1:
                raise ValueError(f"{cfg.name}: a disaggregated rollout stage needs rollout_gpus >= 1")
            a += ["--rollout-num-gpus", str(m.rollout_gpus)]
    return a + list(m.extra_args)


STARTUP_DIR = Path(__file__).resolve().parent / "_startup"


def install_keys_patch() -> list[str]:
    """Two independent routes, then a check in a fresh interpreter: (1) a ``.pth`` line in this interpreter's
    purelib (Debian-patched venvs list a non-processed ``local/.../dist-packages`` first in
    ``site.getsitepackages()``, which is why purelib is used); (2) ``_startup/sitecustomize.py`` first on PYTHONPATH,
    which Ray workers inherit."""
    import sysconfig

    from .keys_patch import CONVERSION, TARGET, pth_line

    written = []
    for d in {sysconfig.get_paths()["purelib"], *[p for p in site.getsitepackages() if "site-packages" in p]}:
        if Path(d).is_dir():
            (Path(d) / "jev_miles_keys.pth").write_text(pth_line())
            written.append(d)
    pp = os.environ.get("PYTHONPATH", "")
    if str(STARTUP_DIR) not in pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join([str(STARTUP_DIR)] + ([pp] if pp else []))
    code = (f"import {TARGET} as a, {CONVERSION} as c; "
            "print('metadata' in a.TRAIN_KEYS and 'rollout_mask_sums' in a.TRAIN_KEYS "
            "and getattr(a.TorchNativeTrainRayActor, '_jev_parallel_config', False) "
            "and getattr(c._package_shards, '_jev_patched', False))")
    probe = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy())
    ok = probe.stdout.strip().endswith("True")
    print(f"[launch] key patch: .pth in {written}; sitecustomize {STARTUP_DIR}; fresh-interpreter check: "
          f"{'ok' if ok else 'FAILED ' + (probe.stdout + probe.stderr)[-500:]}", flush=True)
    if not ok:
        raise SystemExit("the trainer key patch is not active in a fresh interpreter")
    return written


def sh(cmd: str, check: bool = True) -> int:
    print(f"[launch] $ {cmd}", flush=True)
    rc = subprocess.call(cmd, shell=True)
    if check and rc:
        raise SystemExit(f"command failed ({rc}): {cmd}")
    return rc


def apply_overrides(cfg: StageConfig) -> StageConfig:
    """Infrastructure overrides from the environment (manifest §3 / §9: no objective value changes):
    ``JEV_MILES_MAX_TOKENS`` (micro-batch token budget, e.g. after an out-of-memory retry)."""
    tok = os.environ.get("JEV_MILES_MAX_TOKENS")
    if tok:
        raw = cfg.model_dump()
        raw["miles"]["max_tokens_per_gpu"] = int(tok)
        cfg = StageConfig.model_validate(raw).check()
    return cfg


def alloc_env(cfg: StageConfig, env: dict[str, str]) -> None:
    """Colocated stages run SGLang under TorchMemorySaver, which refuses PyTorch's expandable segments; the
    trainer-only stages keep them (less fragmentation)."""
    if cfg.miles.colocate:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def run(config: str, dry_run: bool = False, chain: str = CHAIN_DIR) -> int:
    cfg = apply_overrides(resolve(load_stage_config(config), chain))
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    resolved = out / "stage_config.yaml"
    resolved.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    os.environ[CFG_ENV] = str(resolved.resolve())
    os.environ["JEV_MILES_OUT"] = str(out.resolve())
    os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "1"
    alloc_env(cfg, os.environ)  # type: ignore[arg-type]
    n = int(os.environ.get("JEV_MILES_NUM_ROLLOUT") or num_rollouts(cfg))   # override: the 4-step smoke only
    args = miles_args(cfg, n)
    script = "train_async.py" if cfg.stage == "sft" else "train.py"
    cmd = f"python3 {MILES_ROOT}/{script} " + " ".join(shlex.quote(x) for x in args)
    (out / "miles_cmd.txt").write_text(cmd + "\n")
    print(f"[launch] {cfg.name}: {n} rollouts; {cmd}", flush=True)
    if dry_run:
        return 0
    install_keys_patch()
    ngpu = cfg.miles.actor_gpus + (0 if cfg.miles.colocate else cfg.miles.rollout_gpus)
    sh("ray stop --force > /dev/null 2>&1; pkill -9 -f sglang > /dev/null 2>&1; sleep 2", check=False)
    sh(f"ray start --head --node-ip-address 127.0.0.1 --num-gpus {ngpu} --disable-usage-stats")
    t0 = time.time()
    env = {**os.environ, "RAY_ADDRESS": "127.0.0.1:6379", "MASTER_ADDR": "127.0.0.1", "no_proxy": "127.0.0.1"}
    try:
        rc = subprocess.call(cmd, shell=True, env=env, cwd=os.getcwd())
    finally:
        sh("ray stop --force > /dev/null 2>&1", check=False)
    (out / "train_done.json").write_text(json.dumps({"rc": rc, "minutes": round((time.time() - t0) / 60, 1),
                                                     "rollouts": n}))
    return rc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "args"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--chain", default=CHAIN_DIR)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "args":
        cfg = resolve(load_stage_config(a.config), a.chain)
        print(" ".join(shlex.quote(x) for x in miles_args(cfg, num_rollouts(cfg))))
        return
    sys.exit(run(a.config, a.dry_run, a.chain))


if __name__ == "__main__":
    main()
