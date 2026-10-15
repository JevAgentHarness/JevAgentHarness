"""Offline tests for the extra suites: long-term-memory QA (Jev-for-memory) and ALFWorld (action-only).

No downloads, no API keys: synthetic / hand-written tiny datasets, scripted System-2 LLMs and the
simulated / oracle System-1 backends. ALFWorld runs against a fake TextWorld batch env; a real-data
smoke test runs only when ``alfworld`` is installed and ``$ALFWORLD_DATA`` holds the eval games.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from jevharness.accounting import Ledger
from jevharness.bench.env import Env, StepResult
from jevharness.bench.metrics import decision_records, decision_table, episode_table
from jevharness.bench.suites import memory_qa as mq
from jevharness.bench.suites.alfworld import AlfWorldEnv, collect_game_files, load_alfworld
from jevharness.decision.simulated import OracleBackend, SimulatedBackend
from jevharness.harness.agent import JevAgent
from jevharness.harness.config import DPConfig, HarnessConfig, Mode
from jevharness.harness.memory import MemoryStore
from jevharness.llm.base import ChatResponse
from jevharness.llm.scripted import ScriptedLLM

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "memory"


# ====================================================================== memory QA
def _reader() -> ScriptedLLM:
    return ScriptedLLM("scripted", policy=mq.lexical_reader_policy)


def _judge() -> ScriptedLLM:
    return ScriptedLLM("scripted-small", policy=mq.containment_judge_policy)


def test_synthetic_conversations_and_candidates():
    convs = mq.synthetic_conversations(n=2, sessions=3, filler=5)
    assert [c.conv_id for c in convs] == ["syn0", "syn1"]
    c = convs[0]
    uids = {t.uid for t in c.turns()}
    assert c.questions and all(q.evidence and set(q.evidence[0]) <= uids for q in c.questions)
    assert any(q.category == "temporal" for q in c.questions)
    for gran, n_expected in (("turn", len(c.turns())), ("session", len(c.sessions))):
        cands = mq.make_candidates(c, gran)
        assert len(cands) == n_expected
        assert {u for x in cands for u in x.units} == uids
    rounds = mq.make_candidates(c, "round")
    assert all(1 <= len(x.units) <= 2 for x in rounds) and {u for x in rounds for u in x.units} == uids
    assert rounds[0].text.startswith(f"({c.sessions[0].date})")
    assert mq.synthetic_conversations(n=1)[0].questions == mq.synthetic_conversations(n=1)[0].questions  # deterministic
    with pytest.raises(ValueError):
        mq.make_candidates(c, "paragraph")


def test_gold_hint_backend_drives_oracle_through_memory_store():
    s1 = mq.GoldHintBackend(OracleBackend())
    store = MemoryStore(DPConfig(mode=Mode.TRIGGER, tau=0.5), DPConfig(mode=Mode.ASSIST, k=1), s1=s1,
                        max_items=mq.UNLIMITED)
    ledger = Ledger()
    kept = []
    for i in range(160):
        s1.set_write_gold(i % 3 == 0)
        kept.append(store.offer(f"memory number {i}", "ctx", ledger=ledger, step=i, units=[f"u{i}"], order=i))
    assert kept == [i % 3 == 0 for i in range(160)]
    assert len(store.items) == 54                         # > 51 -> the read goes through a Choice tournament
    for target in (store.items[7], store.items[53]):
        s1.set_read_gold({target.id})
        n0 = len(ledger.records)
        got = store.recall("QUESTION: which one?", ledger=ledger)
        assert [m.id for m in got] == [target.id]
        # 2 chunks; the chunk without the target answers "none" and is pruned, so the single
        # surviving winner needs no final round
        assert len(ledger.records) - n0 == 2
    # nothing relevant -> the oracle ranks the "none" option first -> empty recall
    s1.set_read_gold(set())
    assert store.recall("QUESTION: which one?", ledger=ledger) == []
    dps = ledger.summary()["by_decision_point"]
    assert dps["memory_write"]["calls"] == 160 and dps["memory_read"]["calls"] >= 6
    assert s1.describe() == OracleBackend().describe()


def _exp(**over):
    exp = {
        "name": "syn",
        "dataset": {"name": "synthetic", "n": 2, "sessions": 4, "filler": 8},
        "granularity": "turn",
        "s1_backends": {"sim": {"type": "simulated", "accuracy": 0.8, "seed": 1}},
        "embedder": {"type": "hashing"},
        "workers": 2,
        "arms": [
            {"name": "full_context", "read": {"policy": "full_context"}},
            {"name": "no_memory", "read": {"policy": "none"}},
            {"name": "all_recency_k2", "write": {"mode": "always"}, "read": {"policy": "recency", "k": 2}},
            {"name": "all_bm25_k3", "write": {"mode": "always"}, "read": {"policy": "bm25", "k": 3}},
            {"name": "all_embed_k3", "write": {"mode": "always"}, "read": {"policy": "embedding", "k": 3}},
            {"name": "all_s1read_k3", "write": {"mode": "always"}, "read": {"policy": "s1", "k": 3}},
            {"name": "s1write_s1read_k3", "write": {"mode": "trigger", "tau": 0.5}, "read": {"policy": "s1", "k": 3}},
            {"name": "simwrite_bm25_k3", "s1": "sim", "write": {"mode": "trigger", "tau": 0.5},
             "read": {"policy": "bm25", "k": 3}},
            {"name": "llmwrite_all", "write": {"mode": "llm"}, "read": {"policy": "all"}},
            {"name": "rerank_k2", "write": {"mode": "always"},
             "read": {"policy": "s1_rerank", "k": 2, "prefilter": {"policy": "embedding", "n": 10}}},
        ],
    }
    exp.update(over)
    return exp


def test_memory_qa_pipeline_offline(tmp_path):
    exp = _exp()
    summary = mq.run_memory_experiment(exp, tmp_path, llm=_reader(), s1=OracleBackend(), judge_llm=_judge(),
                                       progress=False)
    n_q = sum(len(c.questions) for c in mq.load_dataset(exp["dataset"]))
    assert set(summary) == {a["name"] for a in exp["arms"]}
    for name, s in summary.items():
        assert s["n"] == n_q and s["errors"] == 0, name
        assert s["accuracy_metric"] == "judge" and s["judge_n"] == n_q
        assert s["cost_usd_per_q"] > 0 and s["llm_calls_per_q"] == 1.0 and s["judge_cost_usd"] > 0

    full, none = summary["full_context"], summary["no_memory"]
    assert full["evidence_recall"] == 1.0 and none["evidence_recall"] == 0.0 and none["accuracy"] == 0.0
    oracle_rw = summary["s1write_s1read_k3"]
    assert oracle_rw["gate_precision"] == 1.0 and oracle_rw["gate_recall"] == 1.0
    assert oracle_rw["evidence_stored"] == 1.0 and oracle_rw["evidence_recall"] == 1.0
    assert oracle_rw["stored_fraction"] < 0.5
    assert oracle_rw["memory_tokens"] < summary["all_s1read_k3"]["memory_tokens"]
    assert oracle_rw["accuracy"] >= 0.9 and oracle_rw["accuracy"] > summary["all_recency_k2"]["accuracy"]
    assert summary["all_s1read_k3"]["evidence_recall"] == 1.0
    assert summary["all_recency_k2"]["evidence_recall"] < 0.5
    assert summary["all_s1read_k3"]["s1_calls_per_q"] >= 1 and summary["all_bm25_k3"]["s1_calls_per_q"] == 0
    sim = summary["simwrite_bm25_k3"]
    assert 0 < sim["stored_fraction"] < 1 and sim["write_s1_calls_per_conv"] > 0
    llmw = summary["llmwrite_all"]
    assert llmw["write_llm_calls_per_conv"] > 0 and llmw["write_cost_usd_per_conv"] > 0

    run = tmp_path / "syn"
    for f in ("summary.json", "summary.md", "experiment.yaml", "all_bm25_k3.jsonl",
              "all_bm25_k3.conversations.jsonl"):
        assert (run / f).exists(), f
    # one S1 write-gate call per offered candidate; question ledgers never contain judge or write calls
    convs = [json.loads(x) for x in (run / "s1write_s1read_k3.conversations.jsonl").read_text().splitlines()]
    for c in convs:
        assert c["usage"]["by_decision_point"]["memory_write"]["calls"] == c["offered"] > c["stored"]
    qrecs = [json.loads(x) for x in (run / "s1write_s1read_k3.jsonl").read_text().splitlines()]
    assert all(set(r["usage"]["by_decision_point"]) <= {"memory_read", "answer"} for r in qrecs)
    assert all(r["judge_usage"]["calls"] == 1 for r in qrecs)

    # resumable: a rerun adds nothing
    lines = (run / "all_bm25_k3.jsonl").read_text().count("\n")
    mq.run_memory_experiment(exp, tmp_path, llm=_reader(), s1=OracleBackend(), judge_llm=_judge(), progress=False)
    assert (run / "all_bm25_k3.jsonl").read_text().count("\n") == lines


def test_llm_write_gate_gets_enough_tokens():
    """Pilot regression: MemoryStore's LLM gate asks with max_tokens=4, at which Gemini 3.x returns an empty
    text, so every candidate was rejected. The suite wraps the write LLM (MinTokensLLM)."""
    from jevharness.llm.base import LLMClient

    class TinyBudgetLLM(LLMClient):  # Gemini-like: nothing visible below 8 output tokens
        provider = "scripted"

        def _chat(self, messages, tools=None, **kw):
            text = "Yes" if (kw.get("max_tokens") or 1000) >= 8 else ""
            return ChatResponse(text=text, input_tokens=50, output_tokens=len(text) and 1, model=self.model)

    gate = DPConfig(mode=Mode.LLM)
    raw = MemoryStore(gate, DPConfig(mode=Mode.OFF), llm=TinyBudgetLLM("tiny"))
    # fixed at the source too (MemoryStore now asks with max_tokens=16), so even the raw store stores it
    assert raw.offer("Caroline adopted a dog on 3 May.", "TASK: remember facts") is True
    arm = mq.parse_arm({"name": "llmgate", "write": {"mode": "llm"}, "read": {"policy": "bm25", "k": 3}})
    engine = mq.MemoryQAEngine(arm, llm=_reader(), write_llm=TinyBudgetLLM("tiny"))
    assert isinstance(engine.write_llm, mq.MinTokensLLM) and engine.write_llm.min_tokens == 16
    fixed = MemoryStore(gate, DPConfig(mode=Mode.OFF), llm=engine.write_llm)
    ledger = Ledger()
    assert fixed.offer("Caroline adopted a dog on 3 May.", "TASK: remember facts", ledger=ledger) is True
    assert ledger.summary()["s2"]["calls"] == 1 and ledger.summary()["by_decision_point"]["memory_write"]
    off = mq.MemoryQAEngine(arm, llm=_reader(), write_llm=TinyBudgetLLM("tiny"), opts={"write_gate_min_tokens": 0})
    assert not isinstance(off.write_llm, mq.MinTokensLLM)


def test_memory_qa_simulated_s1_from_config_and_validation(tmp_path):
    exp = _exp(s1={"type": "simulated", "accuracy": 0.7, "seed": 0},
               arms=[{"name": "simgate", "write": {"mode": "trigger", "tau": 0.5}, "read": {"policy": "s1", "k": 3}}])
    s = mq.run_memory_experiment(exp, tmp_path, llm=_reader(), progress=False)["simgate"]
    assert s["accuracy_metric"] == "f1" and s["judge_n"] == 0          # no judge configured
    assert 0 < s["gate_recall"] < 1 or 0 < s["gate_precision"] < 1
    assert s["latency_ms_per_q"] >= 100.0                               # simulated S1 latency on the read path
    with pytest.raises(ValueError):
        mq.parse_arm({"name": "bad", "write": {"mode": "off"}, "read": {"policy": "bm25"}})
    with pytest.raises(ValueError):
        mq.parse_arm({"name": "bad", "read": {"policy": "magic"}})
    with pytest.raises(ValueError):   # k must leave room for the tournament to shrink
        mq.MemoryQAEngine(mq.parse_arm({"name": "k", "read": {"policy": "s1", "k": 60}}), llm=_reader(),
                          s1=SimulatedBackend())
    with pytest.raises(ValueError):
        mq.MemoryQAEngine(mq.parse_arm({"name": "no_s1", "read": {"policy": "s1"}}), llm=_reader())
    arms = mq.expand_arms({"arms": [], "sweeps": [{"base": {"name": "g", "write": {"mode": "trigger"},
                                                            "read": {"policy": "bm25"}},
                                                   "set": "write.tau", "values": [0.3, 0.7]}]})
    assert [a["name"] for a in arms] == ["g_tau0.3", "g_tau0.7"] and arms[1]["write"]["tau"] == 0.7


def test_scoring_follows_locomo_and_longmemeval_rules():
    assert mq.exact_match("The Pixel.", "pixel") == 1.0
    assert mq.f1_score("dogs", "dog") == (1.0 if mq.STEMMER == "porter" else 0.0)   # LoCoMo stems (NLTK Porter)
    assert mq.f1_multi("Oslo, Berlin", "Berlin, Oslo") == 1.0
    assert mq.f1_multi("Oslo", "Berlin, Oslo") == 0.5
    q3 = mq.QAItem("q", "?", "psychology; counseling", "open-domain", meta={"category_id": 3})
    assert mq.score_answer("psychology", q3)["f1"] == 1.0
    q1 = mq.QAItem("q", "?", "Oslo, Berlin", "multi-hop", meta={"category_id": 1})
    assert mq.score_answer("Berlin", q1)["f1"] == 0.5
    adv = mq.QAItem("q", "What did A realize?", "", "adversarial", abstention=True,
                    meta={"category_id": 5, "adversarial_answer": "self-care"})
    assert mq.score_answer("Not mentioned in the conversation", adv)["f1"] == 1.0
    assert mq.score_answer("self-care", adv)["f1"] == 0.0
    q, key = mq._cat5_question(adv)
    assert "Select the correct answer: (a)" in q and set(key.values()) == {"self-care", mq.NOT_MENTIONED}
    assert mq._cat5_answer("(a)", key) == key["a"] and mq._cat5_answer("b", key) == key["b"]
    lme_abs = mq.QAItem("x_abs", "?", "never discussed", "abstention", abstention=True)
    assert mq.score_answer("I don't know, you never told me.", lme_abs)["em"] == 1.0
    # judge prompts: LongMemEval's templates are chosen by question type
    base = dict(qid="x", question="Q?", answer="A", evidence=[])
    assert "off-by-one" in mq.judge_prompt(mq.QAItem(category="t", meta={"question_type": "temporal-reasoning"},
                                                     **base), "r")
    assert "updated answer" in mq.judge_prompt(mq.QAItem(category="k", meta={"question_type": "knowledge-update"},
                                                         **base), "r")
    assert "Rubric: A" in mq.judge_prompt(mq.QAItem(category="p", meta={"question_type":
                                                                        "single-session-preference"}, **base), "r")
    assert "unanswerable" in mq.judge_prompt(adv, "r") and "self-care" in mq.judge_prompt(adv, "r")
    j = mq.containment_judge_policy
    assert j([{"role": "user", "content": mq.judge_prompt(q3, "It is psychology; counseling.")}], None).text == "yes"
    assert j([{"role": "user", "content": mq.judge_prompt(q3, "biology")}], None).text == "no"


def test_prompts_budgets_and_rendering():
    conv = mq.synthetic_conversations(n=1, sessions=3, filler=4)[0]
    text, covered = mq.render_full(conv, None)
    assert covered == {t.uid for t in conv.turns()} and text.count("DATE:") == 3
    short, cov_short = mq.render_full(conv, 30)          # budget keeps (the tail of) the newest session only
    assert cov_short == {t.uid for t in conv.sessions[-1].turns}
    qa = conv.questions[0]
    p, key = mq.build_prompt(conv, qa, "ctx", "embedding", "short", "locomo")
    assert key is None and p.startswith("Below are memories") and p.endswith("Short answer:")
    temporal = next(q for q in conv.questions if q.category == "temporal")
    assert "Use DATE of CONVERSATION" in mq.build_prompt(conv, temporal, "", "full_context", "short", "locomo")[0]
    lme = mq.QAItem("q", "Where?", "Oslo", "single-session-user", question_date="2023/05/01 (Mon) 10:00")
    p2, _ = mq.build_prompt(conv, lme, "hist", "s1", "cot", "longmemeval")
    assert "History Chats:\n\nhist" in p2 and "Current Date: 2023/05/01" in p2 and p2.endswith("(step by step):")


def _locomo_sample() -> dict:
    turn = lambda d, s, t, **kw: {"speaker": s, "dia_id": d, "text": t, **kw}  # noqa: E731
    return {"sample_id": "conv-x", "conversation": {
        "speaker_a": "Ana", "speaker_b": "Ben",
        "session_1_date_time": "1:00 pm on 1 May, 2023",
        "session_1": [turn("D1:1", "Ana", "Hi Ben!"),
                      turn("D1:2", "Ben", "Look at my dog", img_url=["x"], blip_caption="a photo of a beagle")],
        "session_2_date_time": "2:00 pm on 8 May, 2023",
        "session_2": [turn("D2:1", "Ana", "I painted a sunrise in 2022.")],
        "session_10_date_time": "3:00 pm on 1 June, 2023",
        "session_10": [turn("D10:1", "Ben", "We moved to Oslo.")],
        "session_11_date_time": "orphan date without a session"},
        "qa": [{"question": "When did Ana paint a sunrise?", "answer": 2022, "evidence": ["D2:1"], "category": 2},
               {"question": "Dog and city?", "answer": "beagle, Oslo", "evidence": ["D:10:1; D1:2", "D9:9"],
                "category": 1},
               {"question": "What did Ana realize?", "evidence": ["D1:1"], "category": 5,
                "adversarial_answer": "self-care"}]}


def test_load_locomo_from_local_file(tmp_path):
    p = tmp_path / "locomo10.json"
    p.write_text(json.dumps([_locomo_sample()]))
    (conv,) = mq.load_locomo(str(p), download=False)
    assert conv.speakers == ("Ana", "Ben") and [s.session_id for s in conv.sessions] == ["S1", "S2", "S10"]
    assert conv.sessions[0].turns[1].text.endswith("[shares an image: a photo of a beagle]")
    assert [q.category for q in conv.questions] == ["temporal", "multi-hop"]       # adversarial off by default
    assert conv.questions[0].answer == "2022"
    assert conv.questions[1].evidence == [["D10:1"], ["D1:2"]]                    # malformed + unknown ids handled
    adv = mq.load_locomo(str(p), include_adversarial=True, download=False)[0].questions[-1]
    assert adv.abstention and adv.category == "adversarial" and adv.meta["adversarial_answer"] == "self-care"
    only = mq.load_locomo(str(p), categories=[1], download=False)[0].questions
    assert [q.category for q in only] == ["multi-hop"]
    with pytest.raises(FileNotFoundError):
        mq.load_locomo(str(tmp_path / "missing.json"), download=False)


def test_load_longmemeval_from_local_file(tmp_path):
    def sess(*turns):
        return [{"role": r, "content": c, **({"has_answer": True} if h else {})} for r, c, h in turns]
    data = [
        {"question_id": "q1", "question_type": "knowledge-update", "question": "Where do I live now?",
         "answer": "Porto", "question_date": "2023/06/01 (Thu) 10:00",
         "haystack_session_ids": ["s_b", "s_a", "s_a"],
         "haystack_dates": ["2023/05/20 (Sat) 09:00", "2023/05/01 (Mon) 09:00", "2023/05/02 (Tue) 09:00"],
         "haystack_sessions": [sess(("user", "I moved to Porto.", True), ("assistant", "Nice!", False)),
                               sess(("user", "I live in Lisbon.", True), ("assistant", "Cool.", False)),
                               sess(("user", "Recipes please", False))],
         "answer_session_ids": ["s_b", "s_a"]},
        {"question_id": "q2_abs", "question_type": "single-session-user", "question": "What is my cat's name?",
         "answer": "You never mentioned a cat.", "question_date": "2023/06/01 (Thu) 10:00",
         "haystack_session_ids": ["s_c"], "haystack_dates": ["2023/05/03 (Wed) 09:00"],
         "haystack_sessions": [sess(("user", "I have a dog.", False))], "answer_session_ids": ["s_c"]},
    ]
    p = tmp_path / "longmemeval_oracle.json"
    p.write_text(json.dumps(data))
    c1, c2 = mq.load_longmemeval(str(p))
    assert [s.session_id for s in c1.sessions] == ["s_a", "s_a#2", "s_b"]         # sorted by date, dup ids kept
    q = c1.questions[0]
    assert q.category == "knowledge-update" and q.question_date.startswith("2023/06/01")
    assert q.evidence == [["s_a:0"], ["s_b:0"]]
    assert c2.questions[0].abstention and c2.questions[0].evidence == []
    assert [c.conv_id for c in mq.load_longmemeval(str(p), include_abstention=False)] == ["q1"]
    assert mq.load_longmemeval(str(p), question_types=["abstention"])[0].conv_id == "q2_abs"
    with pytest.raises(FileNotFoundError, match="docs/MEMORY.md"):
        mq.load_longmemeval(str(tmp_path / "nope.json"))
    # end to end with LongMemEval prompts / judge templates
    exp = {"name": "lme", "dataset": {"name": "longmemeval", "path": str(p)}, "granularity": "session",
           "answer": {"style": "cot"}, "embedder": {"type": "hashing"}, "workers": 1,
           "arms": [{"name": "embed_k1", "write": {"mode": "always"}, "read": {"policy": "embedding", "k": 1}}]}
    s = mq.run_memory_experiment(exp, tmp_path, llm=_reader(), judge_llm=_judge(), progress=False)["embed_k1"]
    assert s["n"] == 2 and s["errors"] == 0 and set(s["by_category"]) == {"knowledge-update", "abstention"}


def test_cli_main_on_synthetic_config(tmp_path, capsys):
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text((CONFIGS / "synthetic_smoke.yaml").read_text())
    mq.main(["--config", str(cfg), "--out", str(tmp_path / "runs"), "--max-conversations", "1",
             "--max-questions", "3", "--arms", "full_context,simwrite_simread_k3,llmwrite_bm25_k3"])
    out = capsys.readouterr().out
    assert "| arm |" in out and "simwrite_simread_k3" in out and "all_bm25_k3" not in out
    mq.main(["--config", str(cfg), "--out", str(tmp_path / "runs"), "--report-only"])
    assert "llmwrite_bm25_k3" in capsys.readouterr().out
    for f in ("locomo.yaml", "longmemeval.yaml", "synthetic_smoke.yaml"):   # every shipped arm parses
        import yaml
        exp = yaml.safe_load((CONFIGS / f).read_text())
        assert all(mq.parse_arm(a) for a in mq.expand_arms(exp))


# ====================================================================== ALFWorld
class FakeTextWorld:
    """Mimics ``textworld.gym`` batch env (batch_size=1) with ALFWorld wrappers for a tiny game:
    put the apple (in the fridge) on the countertop."""

    ROOM = ("You are in the middle of a room. Looking quickly around you, you see a countertop 1, a fridge 1, "
            "and a sinkbasin 1.")

    def __init__(self, game_file, max_steps=50, expert=False):
        self.game_file, self.max_steps, self.expert = game_file, max_steps, expert
        self.closed = False
        self.loc, self.open, self.holding, self.won, self.steps = None, False, False, False, 0

    def _adm(self):
        adm = ["go to countertop 1", "go to fridge 1", "go to sinkbasin 1", "help", "inventory", "look", "look"]
        if self.loc == "fridge" and not self.open:
            adm.append("open fridge 1")
        if self.loc == "fridge" and self.open and not self.holding:
            adm.append("take apple 1 from fridge 1")
        if self.loc == "countertop" and self.holding:
            adm.append("put apple 1 in/on countertop 1")
        return adm

    def _expert_next(self):
        if not self.holding:
            return "go to fridge 1" if self.loc != "fridge" else ("open fridge 1" if not self.open
                                                                   else "take apple 1 from fridge 1")
        return "go to countertop 1" if self.loc != "countertop" else "put apple 1 in/on countertop 1"

    def _infos(self, first=False):
        infos = {"admissible_commands": [self._adm()], "won": [self.won], "extra.gamefile": [self.game_file]}
        if self.expert:
            infos["extra.expert_plan"] = [["look"] if first else [self._expert_next()]]
        return infos

    def reset(self):
        self.loc, self.open, self.holding, self.won, self.steps = None, False, False, False, 0
        obs = f"-= Welcome to TextWorld, ALFRED! =-\n\n{self.ROOM}\n\nYour task is to: put a apple in countertop."
        return (obs,), self._infos(first=True)

    def step(self, actions):
        a = actions[0]
        obs = "Nothing happens."
        if a.startswith("go to "):
            self.loc = a.split()[2]
            obs = f"You arrive at {a[6:]}."
        elif a == "open fridge 1" and a in self._adm():
            self.open, obs = True, "You open the fridge 1. In it, you see a apple 1."
        elif a == "take apple 1 from fridge 1" and a in self._adm():
            self.holding, obs = True, "You pick up the apple 1 from the fridge 1."
        elif a == "put apple 1 in/on countertop 1" and a in self._adm():
            self.holding, self.won, obs = False, True, "You put the apple 1 in/on the countertop 1."
        self.steps += 1
        done = self.won or self.steps >= self.max_steps
        return (obs,), (1 if self.won else 0,), (done,), self._infos()

    def close(self):
        self.closed = True


WALKTHROUGH = ["go to fridge 1", "open fridge 1", "take apple 1 from fridge 1", "go to countertop 1",
               "put apple 1 in/on countertop 1"]


@pytest.fixture
def alf_data(tmp_path):
    root = tmp_path / "json_2.1.1" / "valid_unseen"

    def game(task_dir, trial, task_type, solvable=True, with_game=True):
        d = root / task_dir / trial
        d.mkdir(parents=True)
        (d / "traj_data.json").write_text(json.dumps({"task_type": task_type}))
        if with_game:
            (d / "game.tw-pddl").write_text(json.dumps({"solvable": solvable, "walkthrough": WALKTHROUGH}))
        return d / "game.tw-pddl"

    main = game("pick_and_place_simple-Apple-None-CounterTop-1", "trial_T1", "pick_and_place_simple")
    game("pick_and_place_simple-AppleSliced-None-CounterTop-1", "trial_T2", "pick_and_place_simple")  # Sliced
    game("pick_and_place_simple-Mug-None-CounterTop-2", "trial_T3", "pick_and_place_simple", solvable=False)
    game("look_at_obj_in_light-Book-None-DeskLamp-3", "trial_T4", "look_at_obj_in_light")
    game("pick_two_obj_and_place-CD-None-Safe-5", "trial_T5", "pick_two_obj_and_place", with_game=False)
    return tmp_path, str(main)


def test_alfworld_game_collection_and_factories(alf_data, monkeypatch):
    data_dir, main = alf_data
    games = collect_game_files("eval_out_of_distribution", data_dir=data_dir)
    assert [Path(g).parent.parent.name for g in games] == ["look_at_obj_in_light-Book-None-DeskLamp-3",
                                                           "pick_and_place_simple-Apple-None-CounterTop-1"]
    assert collect_game_files("valid_unseen", data_dir=data_dir, task_types=[1]) == [main]
    monkeypatch.setenv("ALFWORLD_DATA", str(data_dir))
    facs = load_alfworld(n=1, task_types=["pick_and_place_simple"])
    env = facs[0]()
    assert isinstance(env, AlfWorldEnv) and env._tw is None                      # lazy: nothing loaded yet
    assert env.task_id == "alfworld/valid_unseen/pick_and_place_simple-Apple-None-CounterTop-1/trial_T1"
    assert env.task_type == "pick_and_place_simple" and env.admissible_actions() == []
    with pytest.raises(FileNotFoundError, match="alfworld-download"):
        load_alfworld(split="eval_in_distribution", data_dir=data_dir)
    with pytest.raises(ValueError):
        load_alfworld(split="test", data_dir=data_dir)


def test_alfworld_env_protocol_with_fake_textworld(alf_data):
    _, main = alf_data
    env = AlfWorldEnv(main, tw_factory=FakeTextWorld)
    obs = env.reset()
    assert obs == FakeTextWorld.ROOM and env.instruction == "Your task is to: put a apple in countertop."
    acts = env.admissible_actions()
    assert "help" not in acts and "inventory" not in acts and acts.count("look") == 1
    assert env.oracle_action() == ("act", {"action": "go to fridge 1"})           # planner walkthrough
    assert env.step("finish", {}).observation.startswith("Error")
    env.step("act", {"action": "go to sinkbasin 1"})                              # leave the walkthrough ...
    assert not env.on_walkthrough()
    assert env.oracle_action() == ("act", {"action": "go to fridge 1"})           # ... expert re-plans
    for _ in range(4):
        gold = env.oracle_action()[1]["action"]
        assert gold in env.admissible_actions()
        env.step("act", {"action": gold})
    res = env.step("act", {"action": env.oracle_action()[1]["action"]})
    assert res.done and res.info["won"] and env.oracle_done()
    tw = env._tw
    out = env.evaluate(None)
    assert out["success"] and out["score"] == 1.0 and out["steps"] == 6 and out["expert_steps"] == 5
    assert tw.closed and env._tw is None                                          # evaluate() releases the env
    assert env.step("act", {"action": "look"}).done
    walk_only = AlfWorldEnv(main, oracle="walkthrough", tw_factory=FakeTextWorld)
    walk_only.reset()
    walk_only.step("act", {"action": "look"})
    assert walk_only.oracle_action() is None                                      # off-path, no expert
    with pytest.raises(ValueError):
        AlfWorldEnv(main, oracle="psychic")


def _action_llm(plan=None):
    """Scripted ReAct-style LLM: follows `plan` when possible, else picks the first listed action."""
    state = {"i": 0}

    def policy(messages, tools):
        listed = [ln[2:] for ln in str(messages[-1]["content"]).splitlines() if ln.startswith("- ")]
        want = plan[state["i"]] if plan and state["i"] < len(plan) else None
        pick = want if want in listed else listed[0]
        if pick == want:
            state["i"] += 1
        return ChatResponse(text=f"Thought: next step.\nAction: {pick}")
    return ScriptedLLM("scripted", policy=policy)


def test_alfworld_action_only_agent_paths(alf_data):
    _, main = alf_data

    def run(cfg, s1=None, llm=None, n=3):
        agent = JevAgent(llm or ScriptedLLM("scripted"), cfg, s1=s1)
        return [agent.run(AlfWorldEnv(main, tw_factory=FakeTextWorld)).to_dict() for _ in range(n)]

    # Jev replaces the LLM entirely
    trajs = run(HarnessConfig(name="replace", max_steps=10, action=DPConfig(mode=Mode.REPLACE)), s1=OracleBackend())
    ep = episode_table(trajs)
    assert ep["success"] == 1.0 and ep["llm_calls"] == 0 and ep["steps"] == 5, [t["error"] for t in trajs]
    assert decision_table(decision_records(trajs))["action/replace"]["s1_acc"] == 1.0
    # LLM-only ReAct baseline (action parsed from "Action: ...")
    trajs = run(HarnessConfig(name="llm", max_steps=10), llm=_action_llm(WALKTHROUGH), n=1)
    assert trajs[0]["result"]["success"] and trajs[0]["error"] is None
    # assist: S1 prunes to top-k, the LLM picks among them
    seen = []

    def spy(messages, tools):
        seen.append([ln for ln in str(messages[-1]["content"]).splitlines() if ln.startswith("- ")])
        return ChatResponse(text=f"Action: {seen[-1][0][2:]}")
    trajs = run(HarnessConfig(name="assist", max_steps=10, action=DPConfig(mode=Mode.ASSIST, k=2)),
                s1=OracleBackend(), llm=ScriptedLLM("scripted", policy=spy), n=1)
    assert trajs[0]["result"]["success"] and all(len(s) <= 2 for s in seen)
    # cascade with a noisy S1: runs, logs deferrals, decisions scored against the oracle
    trajs = run(HarnessConfig(name="cascade", max_steps=15, action=DPConfig(mode=Mode.CASCADE, tau=0.8)),
                s1=SimulatedBackend(accuracy=0.8, seed=2), llm=_action_llm(), n=4)
    assert all(t["error"] is None for t in trajs)
    row = decision_table(decision_records(trajs))["action/cascade"]
    assert row["n_gold"] == row["n"] and 0.0 <= row["defer_rate"] <= 1.0


class CorridorEnv(Env):
    """Minimal discrete-action env: walk right `n` cells; 'wait' is a distractor."""

    def __init__(self, n=4, seed=0):
        self.n, self.task_id, self.instruction = n, f"corridor-{seed}", f"Walk to cell {n}."
        self.pos = 0

    def reset(self):
        self.pos = 0
        return "You are at cell 0."

    def admissible_actions(self):
        return ["wait", "step right"] + (["step left"] if self.pos > 0 else [])

    def step(self, tool, args):
        a = args.get("action")
        self.pos += {"step right": 1, "step left": -1}.get(a, 0)
        return StepResult(f"You are at cell {self.pos}.", done=self.pos >= self.n)

    def oracle_action(self):
        return ("act", {"action": "step right"})

    def evaluate(self, final_answer):
        return {"success": self.pos >= self.n, "score": float(self.pos >= self.n)}


def test_action_only_agent_on_fake_discrete_env():
    cfg = HarnessConfig(name="cascade", max_steps=12, action=DPConfig(mode=Mode.CASCADE, tau=0.95),
                        memory_write=DPConfig(mode=Mode.TRIGGER, tau=0.5), memory_read=DPConfig(mode=Mode.ASSIST, k=2))
    agent = JevAgent(_action_llm(["step right"] * 10), cfg, s1=SimulatedBackend(accuracy=0.9, seed=0))
    trajs = [agent.run(CorridorEnv(seed=i)).to_dict() for i in range(5)]
    assert all(t["error"] is None for t in trajs), [t["error"] for t in trajs]
    assert episode_table(trajs)["success"] == 1.0
    assert all(t["meta"]["memory"]["offered"] == len(t["steps"]) for t in trajs)


def _real_alfworld_games() -> list[str]:
    try:
        import alfworld  # noqa: F401
        import textworld  # noqa: F401
    except ImportError:
        return []
    if not os.environ.get("ALFWORLD_DATA"):
        return []
    try:
        return collect_game_files("eval_out_of_distribution")
    except OSError:
        return []


@pytest.mark.skipif(not _real_alfworld_games(), reason="alfworld not installed or $ALFWORLD_DATA has no games")
def test_real_alfworld_game_with_oracle_s1():
    games = _real_alfworld_games()
    assert len(games) == 134
    facs = load_alfworld(n=1)
    agent = JevAgent(ScriptedLLM("scripted"), HarnessConfig(name="replace", max_steps=40,
                                                            action=DPConfig(mode=Mode.REPLACE)), s1=OracleBackend())
    t = agent.run(facs[0]()).to_dict()
    assert t["error"] is None and t["result"]["success"]
