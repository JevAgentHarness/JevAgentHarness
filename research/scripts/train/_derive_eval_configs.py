"""Derive evaluation configs that point an existing E1 (``jevbench decide``) or E2 (``jevbench run``) config at a
served checkpoint (``training/serve_jev.py``). Used by ``scripts/train/90_calibrate_serve_eval.sh``.

  python scripts/train/_derive_eval_configs.py decide configs/decision/bfcl_when2call.yaml OUT.yaml \\
      --spec '{"type":"jev","name":"ours","base_url":"http://localhost:8081","model":"ours","api_key":""}' --name ours
  python scripts/train/_derive_eval_configs.py run configs/experiments/jevgym_tool_select.yaml OUT.yaml --spec ... --name ours

decide: ``backends`` = {name: spec} (plus the original ones with --keep-backends; FC LLM arms only with --keep-fc).
run:    ``s1`` = spec; arms / sweeps bound to other System-1 backends are dropped (``llm_only`` and every arm
        using the default ``s1`` are kept, so the trained decider is compared on the same arms); ``s1_backends``
        is cleared; the run name gets a ``_<name>`` suffix. The System-1 cache is disabled for the new backend
        (a served checkpoint changes between runs under the same name).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def derive_decide(exp: dict[str, Any], spec: dict[str, Any], name: str, keep_backends: bool = False,
                  keep_fc: bool = False) -> dict[str, Any]:
    out = dict(exp)
    out["name"] = f"{exp.get('name', 'decide')}_{name}"
    out["backends"] = {**((exp.get("backends") or {}) if keep_backends else {}), name: spec}
    if not keep_fc:
        out.pop("fc_llms", None)
    out.pop("s1_cache", None)
    return out


def derive_run(exp: dict[str, Any], spec: dict[str, Any], name: str) -> dict[str, Any]:
    out = dict(exp)
    out["name"] = f"{exp.get('name', 'exp')}_{name}"
    out["s1"] = spec
    out.pop("s1_cache", None)
    out["s1_backends"] = {}
    out["arms"] = [a for a in (exp.get("arms") or []) if "s1" not in a]
    out["sweeps"] = [sw for sw in (exp.get("sweeps") or []) if "s1" not in (sw.get("base") or {})]
    if not out["arms"] and not out["sweeps"]:
        raise SystemExit("no arm left that uses the default System-1 backend")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=["decide", "run"])
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--spec", required=True, help="JSON make_backend spec of the served checkpoint")
    ap.add_argument("--name", required=True)
    ap.add_argument("--keep-backends", action="store_true")
    ap.add_argument("--keep-fc", action="store_true")
    a = ap.parse_args(argv)
    exp = yaml.safe_load(Path(a.base).read_text())
    spec = json.loads(a.spec)
    cfg = (derive_decide(exp, spec, a.name, a.keep_backends, a.keep_fc) if a.kind == "decide"
           else derive_run(exp, spec, a.name))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    arms = [x.get("name") for x in cfg.get("arms") or []]
    print(json.dumps({"out": a.out, "name": cfg["name"], "backends": list(cfg.get("backends") or {}), "arms": arms}))


if __name__ == "__main__":
    main()
