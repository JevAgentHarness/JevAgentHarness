"""Shared helpers of the exploratory analyses outside both registered stages (scripts/explore_telecom_rl.py, E1;
scripts/explore_startlux_live.py, E2): one-record-per-task loading, paired task bootstrap, Newcombe intervals, and
the merged key file paper/generated/explore.tex (every script owns its own key prefix; writing one prefix keeps the
others' keys). Descriptive only: no test, no Holm table."""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench import metrics as M  # noqa: E402

TEX = ROOT / "paper/generated/explore.tex"
TRIAL = re.compile(r"#\d+$")
N_BOOT, BOOT_SEED = 10_000, 0


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_latest(paths: Iterable[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """task id -> the record that decides the task (the latest record without an error, else the latest record; the
    runner and `convert` append re-runs), and per-task record counts (attempts). ``.gz`` files are read decompressed."""
    by_task: dict[str, dict[str, Any]] = {}
    n: dict[str, int] = {}
    for p in paths:
        text = gzip.decompress(Path(p).read_bytes()).decode() if str(p).endswith(".gz") else Path(p).read_text()
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                t = json.loads(line)
            except ValueError:
                continue
            if not isinstance(t, dict) or "task_id" not in t:
                continue
            k = TRIAL.sub("", str(t["task_id"]))  # tau2 records carry the trial ("telecom/<id>#0"); one per task
            n[k] = n.get(k, 0) + 1
            prev = by_task.get(k)
            if prev is None or not t.get("error") or prev.get("error"):
                by_task[k] = t
    return by_task, n


def success(t: Mapping[str, Any]) -> bool:
    return bool((t.get("result") or {}).get("success"))


def cost(t: Mapping[str, Any]) -> float:
    return float(((t.get("usage") or {}).get("total") or {}).get("cost_usd") or 0.0)


def latency_ms(t: Mapping[str, Any]) -> float:
    return float(((t.get("usage") or {}).get("total") or {}).get("latency_ms") or 0.0)


def s_calls(t: Mapping[str, Any], comp: str) -> int:
    return int(((t.get("usage") or {}).get(comp) or {}).get("calls") or 0)


def newcombe(ya: Sequence[float], yb: Sequence[float]) -> dict[str, Any]:
    """Newcombe interval of mean(ya) - mean(yb) for paired binary outcomes (percentage points)."""
    a, b = np.asarray(ya, bool), np.asarray(yb, bool)
    r = M.newcombe_interval(int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum()))
    return {"diff_pt": 100 * r["diff"], "ci_pt": [100 * r["ci"][0], 100 * r["ci"][1]],
            "cells": {"both": int((a & b).sum()), "only_arm": int((a & ~b).sum()), "only_ref": int((~a & b).sum()),
                      "neither": int((~a & ~b).sum())}}


def paired_boot(arm: Sequence[float], ref: Sequence[float], *, ratio: bool = False, n_boot: int = N_BOOT,
                seed: int = BOOT_SEED) -> dict[str, Any]:
    """Paired task bootstrap (tasks resampled with replacement): mean(arm) - mean(ref), or with ``ratio`` the savings
    1 - sum(arm) / sum(ref); point estimate and percentile 95 % interval."""
    a, b = np.asarray(arm, float), np.asarray(ref, float)
    if len(a) == 0:
        return {"est": None, "ci": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    if ratio:
        est = 1.0 - a.sum() / b.sum() if b.sum() else float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            boot = 1.0 - a[idx].sum(1) / b[idx].sum(1)
    else:
        est = float(a.mean() - b.mean())
        boot = a[idx].mean(1) - b[idx].mean(1)
    boot = boot[np.isfinite(boot)]
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if len(boot) else [None, None]
    return {"est": float(est), "ci": ci, "n": int(len(a)), "n_boot": n_boot, "seed": seed}


# ------------------------------------------------------------------------------------------------ TeX keys
def _tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
            .replace("#", r"\#").replace("$", r"\$"))


def fmt(v: Any, digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return r"\tbd"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}".replace(",", "{,}")
    if isinstance(v, float):
        s = f"{v:.{digits}f}"
        return s.replace("-", "$-$", 1) if s.startswith("-") else s
    return _tex_escape(str(v))


def write_tex(prefix: str, keys: Mapping[str, str], source: str, path: Path = TEX) -> Path:
    """Merge ``keys`` (already formatted) under ``explore/<prefix>/`` into the shared key file; keys of other prefixes
    are kept verbatim, every key of this prefix is replaced."""
    keep: list[str] = []
    pat = re.compile(r"^\\defres\{explore/([^/}]+)/")
    if path.exists():
        for line in path.read_text().splitlines():
            m = pat.match(line)
            if m and m.group(1) != prefix:
                keep.append(line)
    own = [f"\\defres{{explore/{prefix}/{k}}}{{{v}}}" for k, v in sorted(keys.items())]
    head = ["% AUTO-GENERATED by scripts/explore_telecom_rl.py and scripts/explore_startlux_live.py -- do not edit by hand.",
            "% Exploratory / descriptive rows outside both registered stages (docs/PREREG_A7_STAGE2.md, Deviations).",
            f"% last written {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} by {source} "
            f"at git {(git_head() or '?')[:12]}",
            r"\makeatletter",
            r"\providecommand{\res}[1]{\ifcsname jevres@#1\endcsname\csname jevres@#1\endcsname\else\tbd\fi}",
            r"\providecommand{\defres}[2]{\expandafter\def\csname jevres@#1\endcsname{#2}}"]
    body = sorted(set(keep) | set(own))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(head + body + [r"\makeatother", ""]))
    return path
