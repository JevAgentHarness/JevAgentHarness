"""S2-3's common-state corpus and replay (docs/PREREG_A7_STAGE2.md A7.10 (c), (d); bench/stage2_corpus.py) on
simulated development-seed runs: occurrences per source and step, canonical question hashes without labels or source
answers, exclusion of tasks without four valid sources, one replay path per backend, failures scored incorrect."""
from __future__ import annotations

import json
from pathlib import Path

from jevharness.bench import stage2_corpus as C
from jevharness.bench.meta_gold import suite_factories
from jevharness.bench.runner import run_experiment
from jevharness.decision import make_backend

SUITE = {"name": "jevgym", "n": 3, "seed0": 0, "families": "v2", "distractor_style": "near", "n_distractors": 60,
         "noise_lines": 20, "strict": True}
LLM = {"provider": "simulated", "model": "sim", "error_rate": 0.0, "simulated_latency_ms": 1}
S1 = {"type": "simulated", "accuracy": 0.8, "latency_ms": 1}


def _runs(root: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    base = {"suite": SUITE, "llm": LLM, "s1": S1, "workers": 1}
    run_experiment({**base, "name": "stage2_jevgym", "arms": [{"name": "llm_only"}]}, root, progress=False)
    run_experiment({**base, "name": "stage2_rl_states", "s1_backends": {"intern": S1, "jevagent_rl": S1},
                    "arms": [{"name": "jev_ts_replace_f", "tool_select": {"mode": "replace"}},
                             {"name": "intern_ts_replace_f", "s1": "intern", "tool_select": {"mode": "replace"}},
                             {"name": "rl_ts_replace_f", "s1": "jevagent_rl", "tool_select": {"mode": "replace"}}]},
                   root, progress=False)
    dirs = {"stage2": root / "stage2_jevgym", "states": root / "stage2_rl_states"}
    tasks = [json.loads(x)["task_id"] for x in (dirs["stage2"] / "llm_only.jsonl").read_text().splitlines()]
    return [(arm, dirs[d] / f"{arm}.jsonl") for d, arm in C.SOURCES], tasks


class Boom:
    """A backend that always fails: every answer is scored incorrect after the retries (A7.10 (d))."""
    name, model, max_options, batched_questions = "boom", "boom", 255, False

    def ask(self, *a, **k):
        raise RuntimeError("down")

    def decide(self, *a, **k):
        raise RuntimeError("down")

    def describe(self):
        return {"backend": "boom"}


def test_corpus_build_replay_and_score(tmp_path: Path):
    sources, tasks = _runs(tmp_path)
    facs = suite_factories(tmp_path / "stage2_jevgym")
    corpus = C.build_corpus(sources, tasks, facs)
    occ, qs = corpus["occurrences"], corpus["questions"]
    assert corpus["summary"]["eligible_tasks"] == 3 and not corpus["excluded"]
    assert {o["source"] for o in occ} == {a for _, a in C.SOURCES}
    # one occurrence per labelled step of every source trajectory; the hash covers state, options and descriptions only
    o = occ[0]
    assert o["qhash"] == C.question_hash(o["state"], o["options"], o["descriptions"])
    assert all(any(o["descriptions"]) for o in occ) and set(qs) == {o["qhash"] for o in occ}
    assert {o["stratum"] for o in occ} <= {"on_plan", "after_off_plan"}
    paths = C.write_corpus(corpus, tmp_path / "s23")
    assert len(C.read_jsonl_gz(Path(paths["questions"]))) == len(qs)
    stored = C.read_jsonl_gz(Path(paths["occurrences"]))
    assert "state" not in stored[0] and stored[0]["label_provenance"].startswith("oracle")
    # replay: a simulated backend answers every distinct question once; a failing backend is retried, then incorrect
    qlist = C.read_jsonl_gz(Path(paths["questions"]))
    ans = {r["qhash"]: r for r in C.replay(qlist, make_backend(S1))}
    assert set(ans) == set(qs) and all(r["error"] is None and r["valid"] for r in ans.values())
    bad = {r["qhash"]: r for r in C.replay(qlist[:2], Boom())}
    assert all(r["attempts"] == 3 and r["error"] for r in bad.values())
    acc = C.task_accuracy(stored, ans)
    assert set(acc) == set(tasks) and all(0.0 <= v <= 1.0 for v in acc.values())
    assert all(v == 0.0 for v in C.task_accuracy(stored, bad).values())
    # a task missing from one source leaves the corpus, listed with the failing source
    short = [(a, p) if a != "rl_ts_replace_f" else (a, tmp_path / "nope.jsonl") for a, p in sources]
    ex = C.build_corpus(short, tasks, facs)
    assert len(ex["excluded"]) == 3 and ex["excluded"][0]["failing_sources"] == {"rl_ts_replace_f": "missing"}
