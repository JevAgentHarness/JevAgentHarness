#!/usr/bin/env python3
r"""E6e (exploratory, descriptive; outside both registered stages -- docs/PREREG_A7_STAGE2.md, Deviations entry
"Exploratory row E6e" of 2026-10-06): the five REPLACE cells of the module map at memory write, memory read,
reflection, stop and routing (harness/config.py E6E_CELLS; semantics: results/explore/e6e_modes/cells.md), each run as
its own arm although it behaves as another cell at a fixed threshold (EQUIVALENT_NOTES), beside the LLM-only reference
(routing: Flash-Lite alone) on fresh JevGym seeds 301400-301599. The readout is E6b's (scripts/explore_e6b_modes.py,
row "e6e"): the same paired comparison, coverage / activity / accuracy per mode, the module map's fixed verdict rule;
keys ``e6e/...`` in paper/generated/e6e.tex. No test; outside every Holm table.

    python scripts/explore_e6e_modes.py spend | watch --pid PID [--cap 60] | project | summarize
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_e6b_modes as E  # noqa: E402

E.configure("e6e")

if __name__ == "__main__":
    raise SystemExit(E.main())
