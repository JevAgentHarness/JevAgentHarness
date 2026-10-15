#!/usr/bin/env python3
"""S2-3's common-state corpus (docs/PREREG_A7_STAGE2.md A7.10 (c), (d)): build it from the four source arms, then let
every frozen backend answer every distinct question through one replay path. Library: `jevharness.bench.stage2_corpus`.

    # after the three passes of configs/experiments/stage2_rl_states.yaml and of stage2_jevgym.yaml's llm_only
    python scripts/stage2_corpus.py build --runs runs --out results/stage2/s23
    # one backend at a time (Intern-Decision in-process and JevAgentRL served are loaded sequentially on the L4);
    # the replay cache is new and separate from every rollout cache (A7.10 (c)); order = sorted question hash
    python scripts/stage2_corpus.py replay --corpus results/stage2/s23 --name jev --s1 '{"type": "jev", "provider": "openrouter"}'
    python scripts/stage2_corpus.py replay --corpus results/stage2/s23 --name intern \
        --s1 '{"type": "intern_decision", "model": "internlm/Intern-Decision-4B", "device": "auto", "max_length": 8192}'
    python scripts/stage2_corpus.py replay --corpus results/stage2/s23 --name rl \
        --s1 '{"type": "jev", "name": "jevagent-rl", "base_url": "http://localhost:8081", "model": "jevagent-rl", "api_key": ""}'
    # A7.18 (e) 2: StartLux-Decision (the vendor's server, scripts/aws/serve_startlux.sh; its own knockout beyond 26
    # options); names sl4b, sl27b and the ladder sl0.8b, sl2b, sl9b; A7.19 (f): v2
    python scripts/stage2_corpus.py replay --corpus results/stage2/s23 --name sl4b --cache cache/s1_stage2_startlux_replay.sqlite \
        --s1 '{"type": "jev", "name": "startlux-4b", "base_url": "http://localhost:8090", "model": "StartLux-Decision-4B", "api_key": "", "timeout": 60.0}'

`build` lists every task without four valid source trajectories (excluded, with the failing source; run the three
registered passes first so errored tasks have had their retries); `replay` resumes (answers already in answers_<name>.jsonl are kept) and writes the replay order,
cache path and backend description to replay_<name>.json before the first question.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevharness.bench import stage2_corpus as C  # noqa: E402

STAGE2_TASKS = ROOT / "configs/eval/jevgym_stage2_tasks.json"
N_TASKS = 200          # A7.10 (c): the first 200 tasks of the Stage-2 list


def cmd_build(a: argparse.Namespace) -> int:
    runs = Path(a.runs)
    dirs = {"stage2": runs / "stage2_jevgym", "states": runs / "stage2_rl_states"}
    sources = [(arm, dirs[d] / f"{arm}.jsonl") for d, arm in C.SOURCES]
    tasks = json.loads(STAGE2_TASKS.read_text())["task_ids"][: a.n_tasks]
    from jevharness.bench.meta_gold import suite_factories

    facs = suite_factories(dirs["stage2"]) if (dirs["stage2"] / "experiment.yaml").exists() else {}
    corpus = C.build_corpus(sources, tasks, facs)
    paths = C.write_corpus(corpus, Path(a.out))
    print(json.dumps({"summary": corpus["summary"], "files": paths}, indent=1))
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    from jevharness.decision import make_backend

    cdir = Path(a.corpus)
    qs = C.read_jsonl_gz(cdir / "questions.jsonl.gz")
    out = cdir / f"answers_{a.name}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(x)["qhash"] for x in out.read_text().splitlines() if x.strip()}
    spec = json.loads(a.s1)
    backend = make_backend(spec, cache=a.cache)
    (cdir / f"replay_{a.name}.json").write_text(json.dumps({
        "registered": "PREREG_A7_STAGE2 A7.10 (c)", "backend": backend.describe(), "cache": a.cache,
        "order": "sorted question hash", "batching": "one question per request (the harness tournament for Choices "
        "wider than the backend's max_options)", "retries": 2, "n_questions": len(qs), "resumed_with": len(done)},
        indent=1, default=str))
    with open(out, "a") as f:
        def write(rec: dict) -> None:
            f.write(json.dumps(rec) + "\n")
            f.flush()

        C.replay(qs, backend, retries=2, on_answer=write, done=done)
    print(f"answers -> {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--runs", default="runs")
    b.add_argument("--out", default="results/stage2/s23")
    b.add_argument("--n-tasks", type=int, default=N_TASKS)
    b.set_defaults(fn=cmd_build)
    r = sub.add_parser("replay")
    r.add_argument("--corpus", default="results/stage2/s23")
    r.add_argument("--name", required=True, help="jev | intern | rl | a stage row of A7.13 (base, sft_v1, ...)")
    r.add_argument("--s1", required=True, help="make_backend spec (JSON)")
    r.add_argument("--cache", default="cache/s1_stage2_replay.sqlite", help="the replay cache (never a rollout cache)")
    r.set_defaults(fn=cmd_replay)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
