#!/usr/bin/env python3
r"""E6d (exploratory, descriptive; outside both registered stages -- docs/PREREG_A7_STAGE2.md, Deviations entry
"Exploratory row E6d" of 2026-10-06): the seven module-map cells added by E6d (harness/config.py E6D_CELLS; semantics:
results/explore/e6d_modes/cells.md) -- "advise" assists at memory write, reflection, stop and routing, guards at
reflection and routing, and the memory-read guard -- each in its own arm beside the LLM-only reference (routing: Flash-Lite
alone) on fresh JevGym seeds 301000-301199. The readout is E6b's (scripts/explore_e6b_modes.py, row "e6d"): the same
paired comparison, coverage / activity / accuracy per mode, the module map's fixed verdict rule; keys ``e6d/...`` in
paper/generated/e6d.tex. No test; outside every Holm table.

    python scripts/explore_e6d_modes.py spend | watch --pid PID [--cap 120] | project | summarize
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_e6b_modes as E  # noqa: E402

E.configure("e6d")

if __name__ == "__main__":
    raise SystemExit(E.main())
