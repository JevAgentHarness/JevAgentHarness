#!/usr/bin/env python3
"""The derived profile of the second System-2 family (docs/PREREG_A7_STAGE2.md A7.12 (e)): a deterministic rendering of
the committed best-practice profile for System-2 X. It reads the committed selection artefact
(configs/harness/jevagent_best.yaml; it never re-runs the selection or re-reads Stage-1 outputs) and X's calibration
fits (A7.12 (d)), and writes a profile whose only section is X:

* the **Flash** section's decision points, every cascade τ replaced by X's fit for that decision point (a fit that
  accepts no level -> ``always_defer: true``, A6.11 (c).4); guard / trigger τ and assist k transferred zero-shot;
* a selected routing group routes between X-small (the experiment's ``small_llm``, the default) and X (the main LLM);
* ``skip`` = "not tested (the fitted profile delegates nothing on X)" when no decision point can intervene: every
  cascade always defers (a deferred tool choice leaves tool_params unreachable, A6.22 (a) 4) and no other group is
  active (A7.12 (g));
* the parents' paths and sha256 (profile and fits) in ``derived_from``.

    python scripts/render_profile_x.py                                   # DeepSeek-V4-Pro (configs/experiments/stage2_x.yaml)
    python scripts/render_profile_x.py --s2 gpt-6.1-sol --fit-arm sol_shadow_cal --out configs/harness/jevagent_best_sol.yaml
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.harness import profile as HP  # noqa: E402

FLASH = "gemini-3.5-flash"
CASCADE_DPS = ("tool_select", "tool_params")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fitted_tau(fit: Mapping[str, Any]) -> float | None:
    """The deployed τ̂* of a calibration-arm fit (``primary.tau_hat``); None when no level is accepted (never)."""
    t = (fit.get("primary") or {}).get("tau_hat", fit.get("tau_hat"))
    if t is None:
        return None
    t = float(t)
    return None if not math.isfinite(t) or t > 1.0 else t


def render(parent: Mapping[str, Any], fits: Mapping[str, Mapping[str, Any] | None], s2: str) -> dict[str, Any]:
    if FLASH not in (parent.get("system2") or {}):
        raise SystemExit("the parent profile has no Flash section")
    flash = parent["system2"][FLASH]
    dps = copy.deepcopy(dict(flash.get("decision_points") or {}))
    notes = []
    for dp in CASCADE_DPS:
        c = dps.get(dp)
        if not c or str(c.get("mode")) != "cascade":
            continue
        fit = fits.get(dp)
        if fit is None:
            raise SystemExit(f"the profile has a {dp} cascade but no X fit for it (A7.12 (d))")
        t = fitted_tau(fit)
        c.pop("always_defer", None)
        if t is None:
            c.update({"tau": 0.99, "always_defer": True})
            notes.append(f"{dp}: X's fit accepts no level -> always defer")
        else:
            c["tau"] = t
            notes.append(f"{dp}: tau {t:g} (X's fit)")
    sec: dict[str, Any] = {"decision_points": dps, "harness": dict(flash.get("harness") or {}), "notes": notes}
    ts = dps.get("tool_select") or {}
    tool_can_act = str(ts.get("mode", "off")) not in ("off", "shadow") and not (
        str(ts.get("mode")) == "cascade" and ts.get("always_defer"))
    others = [g for g in HP.active_groups(dps) if g != "tool"]
    if not tool_can_act and not others:
        sec["skip"] = "not tested (the fitted profile delegates nothing on X) -- A7.12 (g)"
    return {"kind": "jevagent_profile", "version": 1, "name": f"jevagent_best_{s2.split('/')[-1]}",
            "status": parent.get("status", "development"),
            "description": "Derived from the committed best-practice profile for System-2 X (A7.12 (e)).",
            "system2": {s2: sec}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parent", default="configs/harness/jevagent_best.yaml")
    ap.add_argument("--s2", default="deepseek/deepseek-v4-pro")
    ap.add_argument("--fit-arm", default="jev_shadow_cal_x")
    ap.add_argument("--fits-dir", default="results/calibration")
    ap.add_argument("--out", default="configs/harness/jevagent_best_x.yaml")
    ap.add_argument("--allow-development", action="store_true", help="DRY RUNS: accept a development parent")
    a = ap.parse_args()
    pp = ROOT / a.parent
    parent = HP.load_profile(pp)
    if parent["status"] != "selected" and not a.allow_development:
        raise SystemExit(f"{a.parent} has status {parent['status']!r}: the selection must be committed first (A7.12 (e))")
    fits, parents = {}, {"profile": {"path": a.parent, "sha256": _sha(pp)}}
    for dp in CASCADE_DPS:
        fp = ROOT / a.fits_dir / f"tauhat_{a.fit_arm}_{dp}.json"
        fits[dp] = json.loads(fp.read_text()) if fp.exists() else None
        if fp.exists():
            parents[f"fit_{dp}"] = {"path": str(fp.relative_to(ROOT)), "sha256": _sha(fp)}
    prof = render(parent, fits, a.s2)
    prof["derived_from"] = parents
    out = ROOT / a.out
    out.write_text("# Derived profile for the second System-2 family, WRITTEN BY scripts/render_profile_x.py "
                   "(docs/PREREG_A7_STAGE2.md A7.12 (e)); do not edit by hand.\n"
                   + yaml.safe_dump(prof, sort_keys=False, allow_unicode=True))
    HP.load_profile(out)
    print(f"wrote {out}: {prof['system2'][a.s2].get('notes')}{' SKIP' if prof['system2'][a.s2].get('skip') else ''}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
