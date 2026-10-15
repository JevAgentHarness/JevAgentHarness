"""scripts/paper_prompts.py -> paper/sections/app_prompts.tex and results/prompts/prompts.md (App. app:prompts-full): the
committed catalogue is what the code renders today (no drift), every prompt text of harness/prompts.py, of the
mini-SWE-agent hooks that ran and of the tau2-bench adapter is in it, no placeholder sentinel or gold label leaks into it,
and its examples come from development seeds. Offline (no API call, no GPU); the trained-model prompt plan imports torch."""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paper_prompts.py"
COMMITTED_TEX = ROOT / "paper" / "sections" / "app_prompts.tex"
COMMITTED_MD = ROOT / "results" / "prompts" / "prompts.md"
sys.path.insert(0, str(ROOT / "scripts"))

import paper_prompts as PP  # noqa: E402

from jevharness.harness import prompts as PR  # noqa: E402


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    d = tmp_path_factory.mktemp("prompts")
    r = subprocess.run([sys.executable, str(SCRIPT), "--out-tex", str(d / "p.tex"), "--out-md", str(d / "p.md")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return (d / "p.tex").read_text(), (d / "p.md").read_text()


@pytest.fixture(scope="module")
def built() -> tuple[str, str]:
    return PP.build()          # in-process: fills PP.PROMPTS_PY_COVERED


def test_committed_copies_match_the_code(rendered: tuple[str, str], built: tuple[str, str]) -> None:
    tex, md = rendered
    assert (tex, md) == built, "the generator is not deterministic"
    msg = "run python scripts/paper_prompts.py and commit paper/sections/app_prompts.tex and results/prompts/prompts.md"
    assert COMMITTED_TEX.read_text() == tex, msg
    assert COMMITTED_MD.read_text() == md, msg


def test_every_harness_prompt_is_catalogued(built: tuple[str, str]) -> None:
    _, md = built
    for name, v in vars(PR).items():
        if name.startswith("_") or getattr(v, "__module__", PR.__name__) != PR.__name__:
            continue
        if isinstance(v, str) and (name.startswith("Q_") or name.endswith(("_PROMPT", "_NOTE"))):
            assert v in md, f"prompts.{name} is not in the catalogue"
        if inspect.isfunction(v) or isinstance(v, str) or (isinstance(v, (int, dict)) and name.isupper()):
            assert name in PP.PROMPTS_PY_COVERED, f"prompts.{name} is new: show it in scripts/paper_prompts.py"


def _fixed_parts(template: str) -> list[str]:
    return [p for p in re.split(r"\{\w+(?::[^{}]*)?\}", template) if p.strip()]


def test_adapter_prompts_that_ran_are_catalogued(built: tuple[str, str]) -> None:
    _, md = built
    swe = PP.module_constants(PP.P_SWE)
    prompt_names = {n for n in swe if re.match(r"^(Q|MSG|CRIT)_", n)}
    assert PP.SWE_NOT_RUN <= prompt_names, "a hook listed as not run no longer exists"
    for name in sorted(prompt_names - PP.SWE_NOT_RUN):
        texts = list(swe[name].values()) if isinstance(swe[name], dict) else [swe[name]]
        for t in texts:
            for part in _fixed_parts(t):
                assert part in md, f"mini_swe.{name} is not in the catalogue"
    tau2 = PP.module_constants(PP.P_TAU2)
    for name in ("TOOL_SELECT_Q", "ROUTING_Q", "STOP_Q", "RESPOND_DESC"):
        assert tau2[name] in md, f"tau2.{name} is not in the catalogue"


def test_no_sentinel_gold_or_evaluation_seed(built: tuple[str, str]) -> None:
    tex, md = built
    assert "@@" not in tex + md
    assert '"gold"' not in md and "gold_" not in md                 # Question.meta never reaches a rendered prompt
    for seed in (PP.DEV_SEED, PP.HARD_SEED):
        PP.refuse_reserved(seed)                                     # development seeds: no exit
    for seed in (200000, 100499, 300000, 310599):
        with pytest.raises(SystemExit):
            PP.refuse_reserved(seed)


def test_tex_structure(built: tuple[str, str]) -> None:
    tex, _ = built
    assert r"\subsection{Prompts}" in tex and r"\label{app:prompts-full}" in tex
    lines = [ln for ln in tex.splitlines() if ln.startswith(r"\promptline{")]
    assert len(lines) > 200
    for ln in lines:                                                 # braces balance once the escaped ones are gone
        bare = ln.replace(r"\{", "").replace(r"\}", "")
        assert bare.count("{") == bare.count("}"), ln
    n_boxes = len(re.findall(r"\\textbf\{P\d+\. ", tex))
    assert n_boxes == sum(len(g.boxes) for g in PP.catalogue()[1])
