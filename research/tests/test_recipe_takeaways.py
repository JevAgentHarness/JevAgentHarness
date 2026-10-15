"""scripts/recipe_takeaways.py: formatting helpers, the derived recipe/ keys on the committed records, no clash with
another generator's keys, every takeaway key resolves, the paper drafts use only defined keys, and the committed
paper/generated/recipe.tex is in sync with the records."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import recipe_takeaways as RT  # noqa: E402

DRAFTS = (ROOT / "results" / "training" / "recipe_paper_draft.tex", ROOT / "paper" / "sections" / "app_recipe.tex")
KEY = re.compile(r"\\res(?:m|na|mna|num|pend|pct)?\{([^}#]+)\}")


def test_helpers():
    assert RT.signed(7.59) == "+7.6" and RT.signed(-0.76) == "-0.8" and RT.signed(-0.04) == "+0.0"
    assert RT.signed(None) is None and RT.pct(0.30859) == "30.9" and RT.pct(None) is None
    assert RT.num(15694) == "15{,}694" and RT.num(77) == "77" and RT.usd(1234.5) == "1{,}234.50"
    assert RT.plain("1{,}213") == "1,213" and RT.plain("$-$1.5") == "-1.5"
    # on-plan accuracy = (all hits - recovery hits) / (all - recovery)
    assert RT.onplan_acc(0.75, 4, 0.5, 2) == 1.0
    status = ("10-02T20:26:43 stage sft exit=143 (11 min)\n10-02T23:29:53 stage sft exit=0 (134 min)\n"
              "10-03T04:14:04 stage opd exit=0 (222 min)\n")
    assert RT.stage_minutes(status) == {"sft": 134, "opd": 222}          # failed attempts are not counted
    assert RT.parse_defres("\\defres{a/b}{1{,}2}\n% x\n\\defres{c}{$-$0.1}\n") == {"a/b": "1{,}2", "c": "$-$0.1"}


@pytest.fixture(scope="module")
def built():
    gen = RT.generated_keys()
    values, raw = RT.build(gen)
    return gen, values, raw


def test_derived_keys_on_records(built):
    _, v, _ = built
    assert v["recipe/data/v1/rows"] == "15{,}694" and v["recipe/data/v1/after_error_share"] == "30.9"
    assert v["recipe/data/v0/rows"] == "1{,}225" and v["recipe/data/v1/gold_fallback_rows"] == "864"
    assert v["recipe/states/delta_recovery"] == "+7.6"                  # 99.835 - 92.244 (development)
    assert v["recipe/stages/opd_i12/delta_v1_acc"] == "-0.8"
    assert v["recipe/stages/grpo_i15/delta_reward_live"] == "+0.0034"
    assert v["recipe/stage/sft_v1/gpu_min"] == "134" and v["recipe/stage/rl_extra/gpu_h"] == "8.2"
    assert v["recipe/cost/sft_only_usd"] == "11.99"                     # 5.283 + 134 / 60 x 3.004
    assert v["recipe/deploy/tauhat/jev_flash_ts"] == "1.00" and v["recipe/deploy/tauhat/rl_flash_ts"] == "0.50"
    assert v["recipe/deploy/tauhat/rl_flash_ts_wrong"] == "0"
    assert v["recipe/calib/sft_v1/temp_global"] == "0.40"
    if (RT.DATA_V1 / "train.jsonl.gz").exists():                       # the mixing rule keeps gold as every argmax
        assert v["recipe/data/v1/target_argmax_gold_share"] == "100.0"
    assert all(k.startswith("recipe/") for k in v)


def test_no_clash_with_other_generators(built):
    gen, v, _ = built
    assert not set(v) & set(gen)
    with pytest.raises(SystemExit):
        RT.build({**gen, "recipe/data/v1/rows": "x"})


def test_takeaway_keys_resolve_and_render(built):
    gen, v, _ = built
    allk = {**gen, **v}
    assert not [k for k in RT.referenced_keys() if k not in allk]
    md = RT.render_md(v, gen)
    assert "MISSING" not in md and "None" not in md
    for t in RT.TAKEAWAYS:
        assert f"## {t['id']}. {t['topic']}" in md
        assert t["label"] and t["support"] and t["sources"]


def test_paper_drafts_use_defined_keys(built):
    gen, v, _ = built
    allk = {**gen, **v}
    for p in DRAFTS:
        if not p.exists():
            continue
        keys = set(KEY.findall(p.read_text()))
        assert keys, p
        assert not sorted(k for k in keys if k not in allk), p


def test_committed_tex_in_sync(built, tmp_path):
    _, v, _ = built
    committed = RT.OUT_TEX
    if not committed.exists():
        pytest.skip("paper/generated/recipe.tex not generated yet")
    assert RT.parse_defres(committed.read_text()) == v, "regenerate: python scripts/recipe_takeaways.py"
    out = RT.main(["--out-tex", str(tmp_path / "r.tex"), "--out-md", str(tmp_path / "r.md")])
    assert out == v and (tmp_path / "r.md").read_text().startswith("# Recipe takeaways")
