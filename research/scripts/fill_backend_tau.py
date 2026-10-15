#!/usr/bin/env python3
"""Write a System-1 backend's fitted cascade thresholds into the selected profile's `backends.<name>` block
(docs/PREREG_A7_STAGE2.md A7.1 (f) for JevAgentRL, A7.18 (e) 4 for StartLux-Decision-4B). Deterministic: it reads the
committed profile and the backend's calibration fits (`results/calibration/tauhat_<fit-arm>_<dp>.json`, written by
`jevharness.bench.threshold_theory`); it never re-runs the selection or reads an evaluation output.

    python scripts/fill_backend_tau.py --backend startlux_4b --fit-arm sl4b_shadow_cal
    python scripts/fill_backend_tau.py --backend jevagent_rl --fit-arm rl_shadow_cal

Every decision point of the block that is `calibration: pending` gets `tau` = the fit's τ̂* (a fit that accepts no
level -> `tau: 0.99, always_defer: true`, A6.11 (c) 4); guard / trigger τ and assist k are not in the block (they carry
over unchanged). When no decision point of the section can then intervene with this backend (the tool choice always
defers, so tool_params is unreachable, A6.22 (a) 4, and no other group is active), the block gets `skip` and every arm
with `profile_backend: <name>` is skipped: the test it serves is not tested (A7.12 (g), A7.18 (f)). The fits' paths
and sha256 go into `fitted_from`.

    python scripts/fill_backend_tau.py --backend jevagentrl_v2 --add-pending       # A7.19 (f), before v2's calibration

`--add-pending` adds the block of a System-1 that was not in scripts/select_profile.py's CALIBRATED_BACKENDS when the
profile was written (JevAgentRL-v2, A7.19 (f)) by that script's own rule: `calibration: pending` on every cascade decision
point of the section's tool group (none -> nothing to calibrate, refused). Until it is filled the runner skips every arm
with `profile_backend: <name>` (an arm with no block would silently take the section's thresholds)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jevharness.harness import profile as HP  # noqa: E402
from render_profile_x import fitted_tau  # noqa: E402

FLASH = "gemini-3.5-flash"


def fill(prof: Mapping[str, Any], backend: str, fits: Mapping[str, Mapping[str, Any] | None],
         s2: str = FLASH) -> dict[str, Any]:
    out = json.loads(json.dumps({k: v for k, v in prof.items() if not k.startswith("_")}))
    sec = (out.get("system2") or {}).get(s2)
    if sec is None:
        raise SystemExit(f"the profile has no {s2} section")
    block = (sec.get("backends") or {}).get(backend)
    if block is None:
        raise SystemExit(f"the profile's {s2} section has no backends.{backend} block (nothing to calibrate: the "
                         "selected tool group is not a cascade, so the backend runs with the section's settings)")
    notes = []
    for dp, cfg in list(block.items()):
        if dp in HP.BACKEND_META_KEYS or not isinstance(cfg, dict) or cfg.get("calibration") != "pending":
            continue
        fit = fits.get(dp)
        if fit is None:
            raise SystemExit(f"backends.{backend}.{dp} is pending but no fit was given for it")
        t = fitted_tau(fit)
        block[dp] = {"tau": 0.99, "always_defer": True} if t is None else {"tau": t}
        notes.append(f"{dp}: " + ("no accepted level -> always defer" if t is None else f"tau {t:g}"))
    dps = {k: dict(v) for k, v in (sec.get("decision_points") or {}).items()}
    for dp, upd in block.items():
        if dp in dps and isinstance(upd, dict):
            dps[dp].update(upd)
    ts = dps.get("tool_select") or {}
    tool_can_act = str(ts.get("mode", "off")) not in ("off", "shadow") and not (
        str(ts.get("mode")) == "cascade" and ts.get("always_defer"))
    others = [g for g in HP.active_groups(dps) if g != "tool"]
    if not tool_can_act and not others:
        block["skip"] = f"not tested (the fitted profile delegates nothing with {backend})"
    block["notes"] = notes
    return out


def add_pending(prof: Mapping[str, Any], backend: str, s2: str = FLASH) -> dict[str, Any]:
    """The profile with a new `backends.<backend>` block, `calibration: pending` on every cascade decision point of the
    section's tool group: scripts/select_profile.py's rule for its CALIBRATED_BACKENDS, applied to a System-1 added later
    (A7.19 (f)). Refuses a skipped section, a section without a tool cascade and an existing block."""
    out = json.loads(json.dumps({k: v for k, v in prof.items() if not k.startswith("_")}))
    sec = (out.get("system2") or {}).get(s2)
    if sec is None:
        raise SystemExit(f"the profile has no {s2} section")
    if sec.get("skip"):
        raise SystemExit(f"the profile's {s2} section is skipped ({sec['skip']}): nothing to calibrate")
    tool = {k: v for k, v in (sec.get("decision_points") or {}).items() if k in HP.DP_GROUPS["tool"]}
    pending = {dp: {"calibration": "pending"} for dp, v in tool.items() if str((v or {}).get("mode")) == "cascade"}
    if not pending:
        raise SystemExit(f"the profile's {s2} tool group has no cascade: nothing to calibrate (the backend runs with the "
                         "section's settings)")
    backends = sec.setdefault("backends", {})
    if backend in backends:
        raise SystemExit(f"backends.{backend} exists already")
    backends[backend] = pending
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", default="configs/harness/jevagent_best.yaml")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--fit-arm", default=None)
    ap.add_argument("--fits-dir", default="results/calibration")
    ap.add_argument("--s2", default=FLASH)
    ap.add_argument("--add-pending", action="store_true",
                    help="add the backend's block with `calibration: pending` (select_profile.py's rule), fill nothing")
    a = ap.parse_args(argv)
    pp = Path(a.profile) if Path(a.profile).is_absolute() else ROOT / a.profile
    prof = HP.load_profile(pp)
    if prof["status"] != "selected":
        raise SystemExit(f"{a.profile} has status {prof['status']!r}: thresholds are filled into a selected profile only")
    head = [ln for ln in pp.read_text().splitlines() if ln.startswith("#")]
    if a.add_pending:
        out = add_pending(prof, a.backend, a.s2)
        dps = sorted(out["system2"][a.s2]["backends"][a.backend])
        pp.write_text("\n".join(head + [f"# backends.{a.backend} added with calibration: pending by "
                                         f"scripts/fill_backend_tau.py --add-pending (A7.19 (f); select_profile.py's "
                                         f"rule): {', '.join(dps)}"]) + "\n"
                      + yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
        HP.load_profile(pp)
        print(f"{a.profile}: backends.{a.backend}: pending {dps}", file=sys.stderr)
        return 0
    if not a.fit_arm:
        ap.error("--fit-arm is required (unless --add-pending)")
    fd = Path(a.fits_dir) if Path(a.fits_dir).is_absolute() else ROOT / a.fits_dir
    fits, parents = {}, {}
    for dp in ("tool_select", "tool_params"):
        fp = fd / f"tauhat_{a.fit_arm}_{dp}.json"
        fits[dp] = json.loads(fp.read_text()) if fp.exists() else None
        if fp.exists():
            parents[dp] = {"path": str(fp), "sha256": hashlib.sha256(fp.read_bytes()).hexdigest()}
    out = fill(prof, a.backend, fits, a.s2)
    blk = out["system2"][a.s2]["backends"][a.backend]
    notes = blk.pop("notes")
    blk["fitted_from"] = {**parents, "fit_arm": a.fit_arm}
    pp.write_text("\n".join(head + [f"# backends.{a.backend} filled by scripts/fill_backend_tau.py "
                                     f"(A7.1 (f) / A7.18 (e) 4): {'; '.join(notes)}"]) + "\n"
                  + yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    HP.load_profile(pp)
    print(f"{a.profile}: backends.{a.backend}: {'; '.join(notes)}{' -> SKIP' if blk.get('skip') else ''}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
