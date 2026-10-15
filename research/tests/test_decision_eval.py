"""Tests of jevharness.bench.decision_eval (offline)."""
from __future__ import annotations

import json

from jevharness.bench import decision_eval as DE


def test_exclusion_is_applied_by_parent_item_and_counts_are_stable(tmp_path):
    """A6.16 (l): a source item (BFCL row) with any record in the exclusion list leaves the population entirely."""
    from jevharness.bench.suites.static import DecisionItem

    def it(i, src, dp):
        return DecisionItem(id=i, source=src, dp=dp, state="", question=None, gold=["x"])

    items = [it("row_1/tool", "bfcl:simple_python", "tool_select"), it("row_1/param/a", "bfcl:simple_python", "tool_params"),
             it("row_2/tool", "bfcl:simple_python", "tool_select"), it("w2c_7", "when2call:test", "tool_select")]
    f = tmp_path / "excl.json"
    f.write_text(json.dumps({"ids": ["row_1/param/a"]}))
    kept = DE.exclude_by_parent(items, f)
    assert [x.id for x in kept] == ["row_2/tool", "w2c_7"]
    c = DE.population_counts(kept)
    assert c["n_items"] == 2 and c["n_source_items"] == 2 and c["by_source"] == {"bfcl:simple_python": 1, "when2call:test": 1}
    assert c["by_source_dp"] == {"bfcl:simple_python|tool_select": 1, "when2call:test|tool_select": 1}
    assert DE.parent_id("row_1/param/a") == "row_1" == DE.parent_id("row_1/tool") and DE.parent_id("w2c_7") == "w2c_7"
    assert all(DE.parent_id(x.id) == x.unit for x in items)
