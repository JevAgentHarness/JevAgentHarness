"""jevbench CLI: repeated -o flags accumulate instead of keeping only the last one."""

from __future__ import annotations

import sys
from unittest import mock

import jevharness.bench.cli as cli


def test_repeated_override_flags_accumulate(monkeypatch):
    seen = {}

    def fake_run(exp, out, **kw):
        seen["exp"] = exp
        return {}

    monkeypatch.setattr(sys, "argv", ["jevbench", "run", "configs/pilot/jevgym_gemini.yaml",
                                      "-o", "workers=2", "-o", "suite.n=3", "name=x"])
    with mock.patch("jevharness.bench.runner.run_experiment", fake_run), \
            mock.patch.object(cli, "run_experiment", fake_run, create=True):
        cli.main()
    exp = seen["exp"]
    assert (exp["workers"], exp["suite"]["n"], exp["name"]) == (2, 3, "x")
