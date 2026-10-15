"""E7e (Deviations 2026-10-06 "E7e: itemized final answer"): JevGym-Hard's answer_style option."""
import pytest

from jevharness.bench.suites.jevgym_hard import ANSWER_STYLES, JevGymHardEnv

KW = dict(families="multi", multi_customers=12, multi_rule="tier")


def test_default_wording_unchanged():
    env = JevGymHardEnv(seed=7272, **KW)
    assert env.instruction.endswith("When all are done, answer with the total amount of the orders you processed, "
                                    "rounded to 2 decimals.")


def test_itemized_changes_only_the_last_sentence():
    a = JevGymHardEnv(seed=7272, **KW)
    b = JevGymHardEnv(seed=7272, answer_style="itemized", **KW)
    head = a.instruction[: -len(a.answer_text())]
    assert b.instruction.startswith(head)
    assert "one line per customer" in b.instruction
    assert a.multi_total == b.multi_total and a._plan() == b._plan()


def test_unknown_style_rejected():
    assert ANSWER_STYLES == ("total", "itemized")
    with pytest.raises(ValueError):
        JevGymHardEnv(seed=7272, answer_style="bullets", **KW)
