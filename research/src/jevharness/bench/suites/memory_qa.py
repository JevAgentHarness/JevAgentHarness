"""Long-term-memory QA for the *Jev-for-memory* paradigm (LoCoMo, LongMemEval).

Pipeline per conversation and harness arm::

    stream memory candidates (turns | user/assistant rounds | whole sessions) in time order
        -> MemoryStore.offer()             WRITE policy under test: always | trigger(tau) [S1 Noul gate] | llm
    for every question:
        -> recall                           READ policy: recency(k) | s1(k) [MemoryStore.recall, S1 Choice
                                             tournament] | embedding top-k | bm25 top-k | s1_rerank (lexical/
                                             dense prefilter -> S1) | all stored | full_context | none
        -> System-2 LLM answers from the recalled memories (dataset-native prompt)
        -> score: LoCoMo EM/F1 (their normalisation, stemming and per-category rules) and optionally an
           LLM judge (LongMemEval's ``evaluate_qa.py`` prompts)

Every model call goes to a jevharness :class:`~jevharness.accounting.Ledger` -- one per conversation for
the write phase (``memory_write`` gate calls, ``memory_index`` embeddings) and one per question for the
read/answer phase (``memory_read``, ``answer``); judge calls go to a separate evaluation ledger so they
never count as system cost. Per arm we report accuracy, EM/F1, judge accuracy, evidence recall (of the
stored memory and of the recalled set), stored fraction, memory tokens, cost, latency and call counts.

Data
  LoCoMo      data/locomo10.json from https://github.com/snap-research/locomo, pinned commit
              ``LOCOMO_COMMIT``; downloaded from raw.githubusercontent.com on first use.
  LongMemEval longmemeval_{s_cleaned,m_cleaned,oracle}.json from the HF dataset
              ``xiaowu0162/longmemeval-cleaned`` -- must be downloaded manually (see docs/MEMORY.md).
  synthetic   a tiny deterministic conversation generator for offline tests / smoke runs.

CLI::

    python -m jevharness.bench.suites.memory_qa --config configs/memory/locomo.yaml --out runs/memory
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import string
import threading
import time
import traceback
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from ...accounting import CallRecord, Ledger, default_prices, estimate_tokens
from ...decision.base import DecisionBackend
from ...decision.types import DecisionResult, QKind, Question
from ...harness.config import DP, DPConfig, Mode
from ...harness.memory import MemoryItem, MemoryStore
from ...llm.base import ChatResponse, LLMClient

DATA_DIR = Path("data/raw")
LOCOMO_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_URL = f"https://raw.githubusercontent.com/snap-research/locomo/{LOCOMO_COMMIT}/data/locomo10.json"
# numeric categories of locomo10.json; names follow the mapping used by most follow-up work (Mem0, A-Mem, Zep)
LOCOMO_CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
LME_HF = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/{file}"
LME_FILES = {"s": "longmemeval_s_cleaned.json", "m": "longmemeval_m_cleaned.json", "oracle": "longmemeval_oracle.json"}

READ_POLICIES = ("full_context", "none", "all", "recency", "s1", "embedding", "bm25", "s1_rerank")
NONE_OPTION = "(none of these memories is relevant)"   # must match MemoryStore._select
UNLIMITED = 10 ** 9


# ====================================================================== data model
@dataclass
class Turn:
    uid: str
    session_id: str
    speaker: str
    text: str
    date: str = ""


@dataclass
class Session:
    session_id: str
    date: str
    turns: list[Turn]


@dataclass
class QAItem:
    qid: str
    question: str
    answer: str
    category: str
    # evidence *groups* of unit (turn) ids: a group counts as covered if any of its units is present
    evidence: list[list[str]] = field(default_factory=list)
    question_date: str | None = None
    abstention: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    conv_id: str
    source: str                       # locomo | longmemeval | synthetic
    sessions: list[Session]
    questions: list[QAItem]
    speakers: tuple[str, str] = ("user", "assistant")

    def turns(self) -> list[Turn]:
        return [t for s in self.sessions for t in s.turns]

    def evidence_units(self) -> set[str]:
        return {u for q in self.questions for g in q.evidence for u in g}


# ====================================================================== loaders
def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 - pinned public URL
            dest.write_bytes(r.read())
    return dest


def _dedup_groups(groups: list[list[str]]) -> list[list[str]]:
    seen, out = set(), []
    for g in groups:
        key = tuple(sorted(g))
        if g and key not in seen:
            seen.add(key)
            out.append(g)
    return out


def load_locomo(path: str | None = None, n_conversations: int | None = None,
                categories: Sequence[int | str] | None = None, include_adversarial: bool = False,
                download: bool = True) -> list[Conversation]:
    """LoCoMo-10. Images are represented by their BLIP caption (as in LoCoMo's own QA scripts).
    Adversarial questions (category 5, no gold answer, only ``adversarial_answer``) are excluded by default,
    as in most follow-up work; with ``include_adversarial`` they are asked in LoCoMo's multiple-choice form."""
    p = Path(path) if path else DATA_DIR / "locomo" / "locomo10.json"
    if not p.exists():
        if not download:
            raise FileNotFoundError(f"{p} not found (LoCoMo: {LOCOMO_URL})")
        _download(LOCOMO_URL, p)
    data = json.loads(p.read_text())
    want = {LOCOMO_CATEGORIES.get(c, str(c)) if isinstance(c, int) else str(c) for c in (categories or [])}
    convs = []
    for sample in data[:n_conversations]:
        c = sample["conversation"]
        nums = sorted(int(m.group(1)) for k in c if (m := re.fullmatch(r"session_(\d+)", k)))
        sessions = []
        for i in nums:
            date = str(c.get(f"session_{i}_date_time", ""))
            turns = []
            for t in c[f"session_{i}"]:
                text = str(t.get("text", ""))
                if t.get("blip_caption"):
                    text += f" [shares an image: {t['blip_caption']}]"
                turns.append(Turn(uid=str(t["dia_id"]), session_id=f"S{i}", speaker=str(t["speaker"]), text=text,
                                  date=date))
            sessions.append(Session(f"S{i}", date, turns))
        units = {t.uid for s in sessions for t in s.turns}
        qs = []
        for j, qa in enumerate(sample["qa"]):
            cat = int(qa["category"])
            name = LOCOMO_CATEGORIES.get(cat, str(cat))
            if cat == 5 and not include_adversarial:
                continue
            if want and name not in want:
                continue
            ev = []
            for e in qa.get("evidence") or []:   # tolerate "D8:6; D9:17", "D:11:26", space-separated lists
                for a, b in re.findall(r"D:?(\d+):(\d+)", str(e)):
                    if f"D{a}:{b}" in units:
                        ev.append([f"D{a}:{b}"])
            ans = qa.get("answer")
            qs.append(QAItem(qid=f"{sample['sample_id']}/q{j}", question=str(qa["question"]),
                             answer="" if ans is None else str(ans), category=name, evidence=_dedup_groups(ev),
                             abstention=cat == 5,
                             meta={"category_id": cat, "adversarial_answer": qa.get("adversarial_answer")}))
        convs.append(Conversation(str(sample["sample_id"]), "locomo", sessions, qs,
                                  (str(c["speaker_a"]), str(c["speaker_b"]))))
    return convs


def load_longmemeval(path: str | None = None, variant: str = "s", n: int | None = None,
                     question_types: Sequence[str] | None = None, include_abstention: bool = True,
                     seed: int | None = None) -> list[Conversation]:
    """LongMemEval: one :class:`Conversation` (= the question's own haystack) per question.
    Evidence groups are the ``has_answer`` turns; an evidence session without such turns counts as one group."""
    p = Path(path) if path else DATA_DIR / "longmemeval" / LME_FILES[variant]
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. LongMemEval is distributed only via Hugging Face "
            f"({LME_HF.format(file=LME_FILES.get(variant, 'longmemeval_s_cleaned.json'))}); download it manually "
            "(see docs/MEMORY.md) and pass dataset.path.")
    data = json.loads(p.read_text())
    if seed is not None:
        random.Random(seed).shuffle(data)
    convs = []
    for e in data:
        qtype = str(e["question_type"])
        abst = str(e["question_id"]).endswith("_abs")
        if question_types and qtype not in question_types and not (abst and "abstention" in question_types):
            continue
        if abst and not include_abstention:
            continue
        raw = sorted(zip(e["haystack_dates"], e["haystack_session_ids"], e["haystack_sessions"]),
                     key=lambda x: str(x[0]))   # "YYYY/MM/DD (Dow) HH:MM" sorts chronologically
        seen: Counter[str] = Counter()
        sessions, groups = [], []
        by_sid: dict[str, list[str]] = defaultdict(list)       # original id -> all turn uids (dup ids merged)
        for date, sid, sess in raw:
            seen[sid] += 1
            sid2 = sid if seen[sid] == 1 else f"{sid}#{seen[sid]}"
            turns = [Turn(f"{sid2}:{i}", sid2, str(t.get("role", "")), str(t.get("content", "")), str(date))
                     for i, t in enumerate(sess)]
            groups.extend([f"{sid2}:{i}"] for i, t in enumerate(sess) if t.get("has_answer"))
            by_sid[sid].extend(t.uid for t in turns)
            sessions.append(Session(sid2, str(date), turns))
        with_answer_turns = {u.rsplit(":", 1)[0].split("#")[0] for g in groups for u in g}
        for sid in e.get("answer_session_ids") or []:   # evidence session without has_answer turns: one group
            if sid not in with_answer_turns and by_sid.get(sid):
                groups.append(list(by_sid[sid]))
        qa = QAItem(qid=str(e["question_id"]), question=str(e["question"]), answer=str(e["answer"]),
                    category="abstention" if abst else qtype, evidence=[] if abst else _dedup_groups(groups),
                    question_date=str(e.get("question_date") or "") or None, abstention=abst,
                    meta={"question_type": qtype})
        convs.append(Conversation(str(e["question_id"]), "longmemeval", sessions, [qa]))
        if n and len(convs) >= n:
            break
    return convs


_SYN_FACTS = [
    ("A", "I finally adopted a dog, his name is Pixel.", "What is the name of {A}'s dog?", "Pixel"),
    ("B", "My sister just moved to Oslo for her new job.", "Where did {B}'s sister move?", "Oslo"),
    ("A", "I started learning the cello last month.", "Which instrument is {A} learning?", "cello"),
    ("B", "I ran my first marathon in Berlin!", "In which city did {B} run his first marathon?", "Berlin"),
    ("A", "My favourite dessert is pistachio gelato.", "What is {A}'s favourite dessert?", "pistachio gelato"),
    ("B", "We booked our honeymoon trip to Kyoto.", "Where is {B} going for his honeymoon?", "Kyoto"),
    ("A", "I got a new job as a nurse at the city hospital.", "What is {A}'s new job?", "nurse"),
    ("B", "My daughter turned seven on Sunday.", "How old is {B}'s daughter?", "seven"),
]
_SYN_FILLER = ["Hey! How are you doing?", "Pretty good, thanks. Busy week.", "Haha, same here.",
               "Did you watch the game last night?", "Not really, I was too tired.", "Let's catch up again soon.",
               "Sure, talk later!", "The weather has been so nice lately.", "Yeah, perfect for a walk.",
               "Anyway, what's new with you?", "Nothing much, just work.", "Oh nice, glad to hear that."]
_SYN_NAMES = [("Ana", "Ben"), ("Chloe", "Dev"), ("Eli", "Fay")]


def synthetic_conversations(n: int = 1, sessions: int = 4, filler: int = 6, seed: int = 0) -> list[Conversation]:
    """Deterministic toy conversations: facts are scattered among filler turns; one question per fact
    (single-hop) plus a temporal question per conversation."""
    convs = []
    for ci in range(n):
        rng = random.Random(f"{seed}-{ci}")
        a, b = _SYN_NAMES[ci % len(_SYN_NAMES)]
        facts = list(_SYN_FACTS)
        rng.shuffle(facts)
        per = math.ceil(len(facts) / sessions)
        sess_list, qs = [], []
        for si in range(sessions):
            date = f"{1 + 7 * si} May 2023"
            lines = [(a if k % 2 == 0 else b, rng.choice(_SYN_FILLER)) for k in range(filler)]
            chunk = facts[si * per:(si + 1) * per]
            for who, text, _q, _a in chunk:
                lines.insert(rng.randrange(1, len(lines) + 1), (a if who == "A" else b, text))
            turns = [Turn(f"c{ci}-D{si + 1}:{k + 1}", f"S{si + 1}", sp, tx, date) for k, (sp, tx) in enumerate(lines)]
            sess_list.append(Session(f"S{si + 1}", date, turns))
            for who, text, q, ans in chunk:
                uid = next(t.uid for t in turns if t.text == text)
                qs.append(QAItem(f"syn{ci}/q{len(qs)}", q.format(A=a, B=b), ans, "single-hop", [[uid]],
                                 meta={"category_id": 4}))
                if "marathon" in text:
                    qs.append(QAItem(f"syn{ci}/q{len(qs)}", f"When did {b} run his first marathon?", date,
                                     "temporal", [[uid]], meta={"category_id": 2}))
        convs.append(Conversation(f"syn{ci}", "synthetic", sess_list, qs, (a, b)))
    return convs


def load_dataset(spec: dict[str, Any]) -> list[Conversation]:
    """``spec``: ``name`` (locomo | longmemeval | synthetic), the loader's keyword args, and optionally
    ``max_questions`` (per conversation) and ``question_seed``: shuffle each conversation's questions with
    ``random.Random(f"{question_seed}/{conv_id}")`` before truncating, so a small pilot sees every category
    (LoCoMo lists its questions grouped by category)."""
    spec = dict(spec)
    name = spec.pop("name")
    max_q = spec.pop("max_questions", None)
    q_seed = spec.pop("question_seed", None)
    if name == "locomo":
        convs = load_locomo(**spec)
    elif name == "longmemeval":
        convs = load_longmemeval(**spec)
    elif name == "synthetic":
        convs = synthetic_conversations(**spec)
    else:
        raise ValueError(f"unknown memory-QA dataset {name!r}")
    if q_seed is not None:
        for c in convs:
            random.Random(f"{q_seed}/{c.conv_id}").shuffle(c.questions)
    if max_q:
        for c in convs:
            c.questions = c.questions[:max_q]
    return convs


# ====================================================================== memory candidates
@dataclass
class Candidate:
    text: str
    units: list[str]
    session_id: str
    date: str
    order: int


def _turn_line(t: Turn, with_date: bool = True) -> str:
    return f"({t.date}) {t.speaker}: {t.text}" if with_date and t.date else f"{t.speaker}: {t.text}"


def _clip(s: str, n: int | None) -> str:
    return s if not n or len(s) <= n else s[: n - 14] + " ...[clipped]"


def make_candidates(conv: Conversation, granularity: str = "turn", max_chars: int | None = None) -> list[Candidate]:
    """turn: one memory per utterance; round: consecutive utterance pairs (user+assistant); session: one per
    session. Texts carry the session date so both the S1 gate/reader and the answerer can reason about time."""
    out: list[Candidate] = []
    for s in conv.sessions:
        if granularity == "session":
            body = "\n".join(_turn_line(t, False) for t in s.turns)
            out.append(Candidate(_clip(f"[Session {s.session_id} | {s.date}]\n{body}", max_chars),
                                 [t.uid for t in s.turns], s.session_id, s.date, len(out)))
            continue
        step = 2 if granularity == "round" else 1
        if granularity not in ("turn", "round"):
            raise ValueError(f"granularity must be turn|round|session, got {granularity!r}")
        for i in range(0, len(s.turns), step):
            ts = s.turns[i:i + step]
            text = "\n".join(_turn_line(t) if k == 0 else _turn_line(t, False) for k, t in enumerate(ts))
            out.append(Candidate(_clip(text, max_chars), [t.uid for t in ts], s.session_id, s.date, len(out)))
    return out


# ====================================================================== gold hints for simulated S1 backends
class GoldHintBackend(DecisionBackend):
    """Transparent proxy that attaches ``Question.meta['gold']`` to the memory questions produced by
    :class:`MemoryStore` (which does not know the gold). Only simulated/oracle backends read it; real
    backends ignore ``meta`` and the decision cache key excludes it, so wrapping is always safe.

    write gold (Noul):   the candidate contains an evidence unit of some question of this conversation.
    read gold (Choice):  the first option whose memory contains evidence for the current question,
                         else the "none" option.
    Hints are thread-local (conversations run in parallel threads)."""

    def __init__(self, inner: DecisionBackend):
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.batched_questions = inner.batched_questions
        self.max_options = inner.max_options
        self._local = threading.local()

    def set_write_gold(self, gold: bool | None) -> None:
        self._local.write = gold

    def set_read_gold(self, item_ids: set[int] | None) -> None:
        self._local.read = item_ids

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        for q in questions:
            if q.meta.get("gold") is not None:
                continue
            if q.decision_point == DP.MEMORY_WRITE.value and q.kind == QKind.NOUL:
                g = getattr(self._local, "write", None)
                if g is not None:
                    q.meta["gold"] = bool(g)
            elif q.decision_point == DP.MEMORY_READ.value and q.kind == QKind.CHOICE:
                ids = getattr(self._local, "read", None)
                if ids is not None:
                    gold = None
                    for o in q.options or []:
                        m = re.match(r"\[m(\d+)\]", o)
                        if m and int(m.group(1)) in ids:
                            gold = o
                            break
                    if gold is None and NONE_OPTION in (q.options or []):
                        gold = NONE_OPTION
                    if gold is not None:
                        q.meta["gold"] = gold
        return self.inner._decide(state, questions)

    def describe(self) -> dict[str, Any]:
        return self.inner.describe()


# ====================================================================== retrieval baselines
_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("a an the and or but of to in on at for with by from is are was were be been being am i you he she it "
            "we they me him her us them my your his its our their this that these those what which who whom whose "
            "when where why how do does did done have has had not no yes so as if then than there here about into "
            "over after before up down out just also very can could would should will shall may might must".split())


def _content_tokens(s: str) -> list[str]:
    return [w for w in _TOK.findall(s.lower()) if w not in _STOP]


class BM25:
    """Okapi BM25 over memory texts (LongMemEval's ``flat-bm25`` style lexical baseline). Zero cost."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(_content_tokens(d)) for d in docs]
        self.len = np.array([sum(d.values()) for d in self.docs], float)
        self.avg = float(self.len.mean()) if len(self.docs) else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.docs))
        for w in set(_content_tokens(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            tf = np.array([d.get(w, 0) for d in self.docs], float)
            s += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len / max(self.avg, 1e-9)))
        return s


class HashingEmbedder:
    """Offline, deterministic bag-of-words (+bigrams) hashing embedder: a zero-cost stand-in for tests."""

    name = "hashing"

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.model = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        toks = _content_tokens(text)
        for t in toks + [f"{x}_{y}" for x, y in zip(toks, toks[1:])]:
            h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def embed(self, texts: Sequence[str], ledger: Ledger | None = None, dp: str | None = None) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class OpenAIEmbedder:
    """OpenAI(-compatible) embeddings (``text-embedding-3-small`` by default; any OpenAI-compatible server via
    ``base_url``, e.g. vLLM/TEI serving an open embedding model). Calls are ledgered as component ``embed``."""

    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small", base_url: str | None = None,
                 api_key: str | None = None, batch_size: int = 64, max_chars: int = 24000):
        from openai import OpenAI

        self.model = model
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str], ledger: Ledger | None = None, dp: str | None = None) -> np.ndarray:
        texts = [t[: self.max_chars] or " " for t in texts]
        with self._lock:
            missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i:i + self.batch_size]
            t0 = time.perf_counter()
            r = self.client.embeddings.create(model=self.model, input=batch)
            ms = (time.perf_counter() - t0) * 1000
            n_tok = int(getattr(getattr(r, "usage", None), "total_tokens", 0) or sum(estimate_tokens(t) for t in batch))
            with self._lock:
                for t, d in zip(batch, r.data):
                    v = np.asarray(d.embedding, np.float32)
                    self._cache[t] = v / (np.linalg.norm(v) + 1e-9)
            if ledger is not None:
                price = default_prices().get(self.model)
                ledger.add(CallRecord(component="embed", backend="openai", model=self.model, decision_point=dp,
                                      input_tokens=n_tok, cost_usd=price.cost(n_tok, 0) if price else 0.0,
                                      latency_ms=ms, priced=price is not None, meta={"n_texts": len(batch)}))
        with self._lock:
            return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)


def make_embedder(spec: dict[str, Any] | str | None) -> Any:
    spec = {"type": spec} if isinstance(spec, str) else dict(spec or {"type": "hashing"})
    kind = spec.pop("type", "hashing")
    if kind == "hashing":
        return HashingEmbedder(**spec)
    if kind == "openai":
        return OpenAIEmbedder(**spec)
    raise ValueError(f"unknown embedder type {kind!r}")


# ====================================================================== scoring (LoCoMo task_eval/evaluation.py)
try:  # LoCoMo stems with NLTK's Porter stemmer; fall back to no stemming (recorded in the summary)
    from nltk.stem import PorterStemmer as _Porter

    _stem: Callable[[str], str] = _Porter().stem
    STEMMER = "porter"
except Exception:  # pragma: no cover - depends on the environment
    def _stem(w: str) -> str:
        return w
    STEMMER = "none"


def normalize_answer(s: Any) -> str:
    s = str(s).replace(",", "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(set(normalize_answer(prediction).split()) == set(normalize_answer(gold).split()))


def f1_score(prediction: str, gold: str) -> float:
    p = [_stem(w) for w in normalize_answer(prediction).split()]
    g = [_stem(w) for w in normalize_answer(gold).split()]
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_multi(prediction: str, gold: str) -> float:
    """LoCoMo multi-hop: split both into comma-separated sub-answers; mean over gold of best match."""
    preds = [x.strip() for x in str(prediction).split(",")]
    golds = [x.strip() for x in str(gold).split(",")]
    return float(np.mean([max(f1_score(p, g) for p in preds) for g in golds]))


_ABSTAIN = ("not mentioned", "no information available", "no information", "not available", "don't know",
            "do not know", "cannot be determined", "can't be determined", "not provided", "unknown", "no record")


def score_answer(prediction: str, qa: QAItem) -> dict[str, float]:
    """EM / F1 with LoCoMo's per-category rules. Abstention-type questions (LoCoMo adversarial, LongMemEval
    ``_abs``) score 1 iff the prediction says the information is not available."""
    pred = str(prediction or "")
    if qa.abstention:
        low = pred.lower()
        hit = ("not mentioned" in low or "no information available" in low) if qa.meta.get("category_id") == 5 \
            else any(k in low for k in _ABSTAIN)
        return {"em": float(hit), "f1": float(hit)}
    gold = qa.answer
    cat = qa.meta.get("category_id")
    if cat == 3:
        gold = gold.split(";")[0].strip()
    f1 = f1_multi(pred, gold) if cat == 1 else f1_score(pred, gold)
    return {"em": exact_match(pred, gold), "f1": f1}


def judge_prompt(qa: QAItem, response: str) -> str:
    """LongMemEval's ``get_anscheck_prompt`` (src/evaluation/evaluate_qa.py @ 9e0b455), verbatim. LoCoMo (and
    synthetic) questions use the generic template; LoCoMo adversarial ones the abstention template."""
    task = qa.meta.get("question_type", "single-session-user")
    question, answer = qa.question, qa.answer
    if qa.abstention:
        if qa.meta.get("category_id") == 5:
            answer = (f"The conversation never mentions this; the tempting answer "
                      f"'{qa.meta.get('adversarial_answer')}' is not supported by the conversation.")
        template = ("I will give you an unanswerable question, an explanation, and a response from a model. Please "
                    "answer yes if the model correctly identifies the question as unanswerable. The model could say "
                    "that the information is incomplete, or some other information is given but the asked "
                    "information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model "
                    "correctly identify the question as unanswerable? Answer yes or no only.")
    elif task == "temporal-reasoning":
        template = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if "
                    "the response contains the correct answer. Otherwise, answer no. If the response is equivalent to "
                    "the correct answer or contains all the intermediate steps to get the correct answer, you should "
                    "also answer yes. If the response only contains a subset of the information required by the "
                    "answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the "
                    "question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
                    "(e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
                    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? "
                    "Answer yes or no only.")
    elif task == "knowledge-update":
        template = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if "
                    "the response contains the correct answer. Otherwise, answer no. If the response contains some "
                    "previous information along with an updated answer, the response should be considered as correct "
                    "as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
                    "Model Response: {}\n\nIs the model response correct? Answer yes or no only.")
    elif task == "single-session-preference":
        template = ("I will give you a question, a rubric for desired personalized response, and a response from a "
                    "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                    "The model does not need to reflect all the points in the rubric. The response is correct as long "
                    "as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\n"
                    "Rubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.")
    else:  # single-session-user / single-session-assistant / multi-session (and LoCoMo, synthetic)
        template = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if "
                    "the response contains the correct answer. Otherwise, answer no. If the response is equivalent to "
                    "the correct answer or contains all the intermediate steps to get the correct answer, you should "
                    "also answer yes. If the response only contains a subset of the information required by the "
                    "answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model "
                    "response correct? Answer yes or no only.")
    return template.format(question, answer, response)


# ====================================================================== answer prompts
LOCOMO_FULL_START = ("Below is a conversation between two people: {a} and {b}. The conversation takes place over "
                     "multiple days and the date of each conversation is written at the beginning of the "
                     "conversation.\n\n")
LOCOMO_MEM_START = ("Below are memories retrieved from a long conversation between two people: {a} and {b}. The "
                    "conversation took place over multiple days; each memory starts with the date of its "
                    "session.\n\n")
LOCOMO_NONE_START = ("You are answering a question about a long conversation between two people: {a} and {b}, but "
                     "none of the conversation is available to you.\n\n")
LOCOMO_QA = ("\n\nBased on the above context, write an answer in the form of a short phrase for the following "
             "question. Answer with exact words from the context whenever possible.\n\nQuestion: {q} Short answer:")
LOCOMO_QA_CAT5 = "\n\nBased on the above context, answer the following question.\n\nQuestion: {q} Short answer:"
LME_DIRECT = ("I will give you several history chats between you and a user. Please answer the question based on the "
              "relevant chat history.\n\n\nHistory Chats:\n\n{history}\n\nCurrent Date: {date}\nQuestion: {q}\n"
              "Answer:")
LME_COT = ("I will give you several history chats between you and a user. Please answer the question based on the "
           "relevant chat history. Answer the question step by step: first extract all the relevant information, and "
           "then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{history}\n\nCurrent Date: "
           "{date}\nQuestion: {q}\nAnswer (step by step):")
NOT_MENTIONED = "Not mentioned in the conversation"


def _cat5_question(qa: QAItem) -> tuple[str, dict[str, str]]:
    """LoCoMo's multiple-choice form for adversarial questions (order randomised deterministically per qid)."""
    adv = str(qa.meta.get("adversarial_answer") or "")
    if random.Random(qa.qid).random() < 0.5:
        key = {"a": NOT_MENTIONED, "b": adv}
    else:
        key = {"a": adv, "b": NOT_MENTIONED}
    return f"{qa.question} Select the correct answer: (a) {key['a']} (b) {key['b']}. ", key


def _cat5_answer(pred: str, key: dict[str, str]) -> str:
    p = pred.strip().lower()
    if len(p) == 1:
        return key["a"] if "a" in p else key["b"]
    if len(p) == 3:
        return key["a"] if "(a)" in p else key["b"]
    return pred


def render_full(conv: Conversation, max_tokens: int | None) -> tuple[str, set[str]]:
    """Whole conversation, session by session; if over budget keep the most recent turns."""
    blocks: list[tuple[str, list[str]]] = []
    for s in conv.sessions:
        lines = [f"{t.speaker}: {t.text}" for t in s.turns]
        blocks.append((f"DATE: {s.date}\nCONVERSATION:\n" + "\n".join(lines), [t.uid for t in s.turns]))
    kept, covered, used = [], set(), 0
    for text, uids in reversed(blocks):
        n = estimate_tokens(text)
        if max_tokens and used + n > max_tokens:
            if not kept:  # keep the tail of the most recent session
                text = text[-max_tokens * 4:]
                kept.append(text)
                covered.update(uids)  # approximate: partially kept
            break
        kept.append(text)
        covered.update(uids)
        used += n
    return "\n\n".join(reversed(kept)), covered


def render_memories(items: Sequence[MemoryItem], max_tokens: int | None) -> tuple[str, list[MemoryItem]]:
    """Chronological rendering; if over budget drop the *lowest-ranked* items (input order = rank)."""
    chosen, used = [], 0
    for it in items:
        n = estimate_tokens(it.text)
        if max_tokens and used + n > max_tokens and chosen:
            continue
        chosen.append(it)
        used += n
    chosen.sort(key=lambda m: m.meta.get("order", m.id))
    return "\n".join(m.text for m in chosen), chosen


def build_prompt(conv: Conversation, qa: QAItem, context: str, policy: str, style: str, prompt: str) -> tuple[str, Any]:
    """Returns (prompt, cat5 answer key or None)."""
    if prompt == "longmemeval":
        tmpl = LME_COT if style == "cot" else LME_DIRECT
        hist = context if context else "(no history available)"
        return tmpl.format(history=hist, date=qa.question_date or "", q=qa.question), None
    a, b = conv.speakers
    start = {"full_context": LOCOMO_FULL_START, "none": LOCOMO_NONE_START}.get(policy, LOCOMO_MEM_START)
    head = start.format(a=a, b=b) + context
    if qa.meta.get("category_id") == 5:
        q, key = _cat5_question(qa)
        return head + LOCOMO_QA_CAT5.format(q=q), key
    q = qa.question
    if qa.meta.get("category_id") == 2 or qa.category == "temporal":
        q += " Use DATE of CONVERSATION to answer with an approximate date."
    if style == "cot":
        return head + LOCOMO_QA.format(q=q).replace("Short answer:", "Think step by step, then give the final "
                                                                    "short answer after 'Answer:'."), None
    return head + LOCOMO_QA.format(q=q), None


# ====================================================================== offline stand-in LLMs
def lexical_reader_policy(messages: list[dict[str, Any]], tools: Any = None) -> ChatResponse:
    """Offline System-2 stand-in (tests / smoke runs only): answers the LLM write-gate prompt with a
    first-person-fact heuristic and QA prompts by copying the non-question words of the context line with
    the highest content-word overlap with the question."""
    prompt = str(messages[-1]["content"])
    if "Should this be stored in long-term memory" in prompt:
        cand = prompt.split("CANDIDATE MEMORY:", 1)[-1].split("\n\nShould this", 1)[0]
        says = cand.split(":", 1)[-1].lower()
        return ChatResponse(text="yes" if re.search(r"\b(i|my|we|our)\b", says) and len(says) > 25 else "no")
    m = re.search(r"Question: (.*?)(?: Short answer:| Select the| Use DATE|\nAnswer)", prompt, re.S)
    question = m.group(1) if m else prompt[-300:]
    context = prompt[: m.start()] if m else prompt
    qtok = set(_content_tokens(question))
    best, best_s = None, 0
    for line in context.splitlines():
        if ":" not in line or line.startswith(("Below ", "You are answering", "I will give you", "DATE:")):
            continue
        s = len(qtok & set(_content_tokens(line)))
        if s > best_s:
            best, best_s = line, s
    if best is None:
        return ChatResponse(text=NOT_MENTIONED)
    date = re.match(r"\((.*?)\)", best)
    if "when" in question.lower() and date:
        return ChatResponse(text=date.group(1))
    said = best.split(":", 1)[-1]
    words = [w for w in re.findall(r"[A-Za-z0-9']+", said) if w.lower() not in qtok and w.lower() not in _STOP]
    return ChatResponse(text=" ".join(words[:6]) or said.strip())


def containment_judge_policy(messages: list[dict[str, Any]], tools: Any = None) -> ChatResponse:
    """Offline judge stand-in: 'yes' iff the normalised gold answer occurs in the normalised response."""
    prompt = str(messages[-1]["content"])
    if "correctly identify the question as unanswerable" in prompt:
        resp = prompt.split("Model Response:", 1)[-1].lower()
        return ChatResponse(text="yes" if any(k in resp for k in _ABSTAIN) else "no")
    gold = re.search(r"(?:Correct Answer|Rubric): (.*?)\n\nModel Response: (.*?)\n\n", prompt, re.S)
    if not gold:
        return ChatResponse(text="no")
    g, r = normalize_answer(gold.group(1)), normalize_answer(gold.group(2))
    return ChatResponse(text="yes" if g and g in r else "no")


OFFLINE_POLICIES = {"lexical-reader": lexical_reader_policy, "containment-judge": containment_judge_policy}


class MinTokensLLM(LLMClient):
    """Raises ``max_tokens`` to at least ``min_tokens`` (ledger accounting unchanged).

    ``harness/memory.py`` asks the LLM write gate for a yes/no with ``max_tokens=4``. Gemini 3.x (Flash and
    Flash-Lite, minimal thinking) returns an *empty* text with ``finish_reason=length`` at that budget, so the
    gate silently rejected every candidate: in the LoCoMo pilot, 0 of 419 turns were stored
    (docs/PILOT_RESULTS.md). At 16 tokens both answer "Yes"/"No" in one token. The suite reuses
    ``MemoryStore`` unchanged, so the write LLM is wrapped here (config key ``write_gate_min_tokens``, 0 = off)."""

    def __init__(self, inner: LLMClient, min_tokens: int = 16):
        super().__init__(inner.model, **inner.defaults)
        self.inner = inner
        self.min_tokens = int(min_tokens)
        self.provider = inner.provider

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              **kw: Any) -> ChatResponse:
        if kw.get("max_tokens") is not None:
            kw["max_tokens"] = max(int(kw["max_tokens"]), self.min_tokens)
        return self.inner._chat(messages, tools=tools, **kw)

    def describe(self) -> dict[str, Any]:
        return {**self.inner.describe(), "min_tokens": self.min_tokens}


def make_llm_any(spec: dict[str, Any] | str) -> LLMClient:
    """``make_llm`` plus ``{"provider": "offline", "model": "lexical-reader" | "containment-judge"}``."""
    from ...llm import make_llm
    from ...llm.scripted import ScriptedLLM

    if isinstance(spec, dict) and spec.get("provider") == "offline":
        return ScriptedLLM(spec.get("price_as", "scripted"), policy=OFFLINE_POLICIES[spec["model"]])
    return make_llm(spec)


# ====================================================================== arms
@dataclass
class Arm:
    name: str
    write: DPConfig
    read: dict[str, Any]
    s1: str | None = None          # key of exp["s1_backends"]
    llm: str | None = None         # key of exp["llms"] (answerer)
    write_llm: str | None = None   # key of exp["llms"] (LLM write gate); default: exp["write_llm"] or the answerer

    @property
    def policy(self) -> str:
        return self.read["policy"]

    def needs_store(self) -> bool:
        return self.policy not in ("full_context", "none")

    def needs_s1(self) -> bool:
        return self.needs_store() and (self.write.mode == Mode.TRIGGER or self.policy in ("s1", "s1_rerank"))

    def prefilter(self) -> dict[str, Any]:
        return dict(self.read.get("prefilter") or {"policy": "bm25", "n": 50})

    def needs_embedder(self) -> bool:
        return self.policy == "embedding" or (self.policy == "s1_rerank" and self.prefilter()["policy"] == "embedding")

    def read_dp(self) -> DPConfig:
        k = int(self.read.get("k", 10))
        if self.policy == "recency":
            return DPConfig(mode=Mode.ALWAYS, k=k, extra={"recency_k": k})
        if self.policy in ("s1", "s1_rerank"):
            return DPConfig(mode=Mode.ASSIST, k=k)
        return DPConfig(mode=Mode.OFF, k=k)

    def short(self) -> str:
        w = self.write.mode.value + (f"({self.write.tau})" if self.write.mode == Mode.TRIGGER else "")
        k = f"@{self.read.get('k', 10)}" if self.policy not in ("full_context", "none", "all") else ""
        return f"write={w} read={self.policy}{k}" if self.needs_store() else f"read={self.policy}"


def expand_arms(exp: dict[str, Any]) -> list[dict[str, Any]]:
    """``arms`` plus ``sweeps`` ({base, set: dotted.path, values}) -- same convention as bench/runner.py."""
    arms = [copy.deepcopy(a) for a in exp.get("arms", [])]
    for sw in exp.get("sweeps", []) or []:
        path = sw["set"].split(".")
        for v in sw["values"]:
            arm = copy.deepcopy(sw["base"])
            node = arm
            for p in path[:-1]:
                node = node.setdefault(p, {})
            node[path[-1]] = v
            arm["name"] = f"{sw['base']['name']}_{path[-1]}{v}"
            arms.append(arm)
    return arms


def parse_arm(d: dict[str, Any]) -> Arm:
    d = dict(d)
    read = dict(d.get("read") or {"policy": "full_context"})
    if read.get("policy") not in READ_POLICIES:
        raise ValueError(f"arm {d.get('name')!r}: read.policy must be one of {READ_POLICIES}")
    write = DPConfig(**(d.get("write") or {"mode": "always" if read["policy"] not in ("full_context", "none")
                                            else "off"}))
    arm = Arm(name=d["name"], write=write, read=read, s1=d.get("s1"), llm=d.get("llm"), write_llm=d.get("write_llm"))
    if arm.needs_store() and write.mode == Mode.OFF:
        raise ValueError(f"arm {arm.name!r}: write.mode=off stores nothing; use read.policy full_context/none")
    if write.mode not in (Mode.OFF, Mode.ALWAYS, Mode.TRIGGER, Mode.LLM):
        raise ValueError(f"arm {arm.name!r}: write.mode must be off|always|trigger|llm")
    return arm


# ====================================================================== engine
def _groups_covered(groups: list[list[str]], units: set[str]) -> float | None:
    if not groups:
        return None
    return sum(1 for g in groups if units.intersection(g)) / len(groups)


def _agg(records: Sequence[CallRecord], pred: Callable[[CallRecord], bool] = lambda r: True) -> dict[str, float]:
    rs = [r for r in records if pred(r)]
    return {"calls": len(rs), "input_tokens": sum(r.input_tokens for r in rs),
            "output_tokens": sum(r.output_tokens for r in rs), "cost_usd": sum(r.cost_usd for r in rs),
            "latency_ms": sum(r.latency_ms for r in rs)}


@dataclass
class Built:
    store: MemoryStore | None
    record: dict[str, Any]
    vecs: np.ndarray | None = None
    bm25: BM25 | None = None


class MemoryQAEngine:
    """Runs one arm over conversations. Thread-safe across conversations."""

    def __init__(self, arm: Arm, llm: LLMClient, s1: DecisionBackend | None = None,
                 write_llm: LLMClient | None = None, embedder: Any = None, judge_llm: LLMClient | None = None,
                 opts: dict[str, Any] | None = None):
        if arm.needs_s1() and s1 is None:
            raise ValueError(f"arm {arm.name!r} needs a System-1 backend (exp['s1'] or an s1_backends binding)")
        if arm.needs_embedder() and embedder is None:
            raise ValueError(f"arm {arm.name!r} needs an embedder (exp['embedder'])")
        self.arm = arm
        self.llm = llm
        self.s1 = GoldHintBackend(s1) if s1 is not None else None
        min_tokens = int((opts or {}).get("write_gate_min_tokens", 16))
        self.write_llm = MinTokensLLM(write_llm or llm, min_tokens) if min_tokens > 0 else (write_llm or llm)
        self.embedder = embedder
        self.judge_llm = judge_llm
        o = dict(opts or {})
        self.granularity = o.get("granularity", "turn")
        self.write_context_turns = int(o.get("write_context_turns", 3))
        self.max_item_chars = o.get("max_item_chars")
        ans = dict(o.get("answer") or {})
        self.style = ans.get("style", "short")
        self.prompt = ans.get("prompt")
        self.max_context_tokens = ans.get("max_context_tokens", 120000)
        self.max_answer_tokens = int(ans.get("max_tokens", 512 if self.style == "cot" else 64))
        self.max_memory_tokens = ans.get("max_memory_tokens", self.max_context_tokens)
        if self.s1 is not None and arm.policy in ("s1", "s1_rerank"):
            chunk = self.s1.max_options - 1
            if int(arm.read.get("k", 10)) >= chunk:
                raise ValueError(f"arm {arm.name!r}: read.k must be < {chunk} (S1 max_options - 1)")

    # ---------------------------------------------------------------- write phase
    def _write_context(self, conv: Conversation, prev: list[str], date: str) -> str:
        a, b = conv.speakers
        who = "a chat assistant and its user" if conv.source == "longmemeval" else f"{a} and {b}"
        parts = [f"TASK: You are the long-term memory of a conversational agent ({who}). Keep what may be asked "
                 f"about later: facts about people, events and dates, plans, preferences, and their updates.",
                 f"SESSION DATE: {date}"]
        if prev:
            parts.append("RECENT CONVERSATION:\n" + "\n".join(prev))
        return "\n".join(parts)

    def build(self, conv: Conversation) -> Built:
        arm = self.arm
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"conv_id": conv.conv_id, "arm": arm.name, "source": conv.source,
                               "n_questions": len(conv.questions)}
        if not arm.needs_store():
            full = "\n".join(_turn_line(t) for t in conv.turns())
            rec.update({"offered": 0, "stored": 0, "stored_fraction": None,
                        "memory_tokens": estimate_tokens(full) if arm.policy == "full_context" else 0,
                        "offered_tokens": estimate_tokens(full), "gate": None,
                        "usage": Ledger().summary(), "wall_ms": 0.0})
            return Built(None, rec)
        ledger = Ledger()
        store = MemoryStore(arm.write, arm.read_dp(), s1=self.s1, llm=self.write_llm, max_items=UNLIMITED)
        cands = make_candidates(conv, self.granularity, self.max_item_chars)
        ev = conv.evidence_units()
        gate = Counter()
        window: list[str] = []
        last_sid = None
        for c in cands:
            if c.session_id != last_sid:
                window, last_sid = [], c.session_id
            is_ev = bool(ev.intersection(c.units))
            if self.s1 is not None:
                self.s1.set_write_gold(is_ev)
            ctx = self._write_context(conv, window[-self.write_context_turns:] if self.write_context_turns else [],
                                      c.date)
            kept = store.offer(c.text, ctx, ledger=ledger, step=c.order, units=c.units, session_id=c.session_id,
                               date=c.date, order=c.order)
            gate[("tp" if kept else "fn") if is_ev else ("fp" if kept else "tn")] += 1
            window.append(_clip(c.text, 400))
        built = Built(store, rec)
        if arm.needs_embedder() and store.items:
            built.vecs = self.embedder.embed([it.text for it in store.items], ledger=ledger, dp="memory_index")
        rec.update({"offered": store.n_offered, "stored": len(store.items),
                    "stored_fraction": len(store.items) / max(store.n_offered, 1),
                    "memory_tokens": sum(estimate_tokens(it.text) for it in store.items),
                    "offered_tokens": sum(estimate_tokens(c.text) for c in cands),
                    "evidence_units": len(ev),
                    "evidence_stored": _groups_covered([[u] for u in sorted(ev)],
                                                       {u for it in store.items for u in it.meta["units"]}),
                    "gate": dict(gate), "usage": ledger.summary(), "wall_ms": (time.perf_counter() - t0) * 1000})
        return built

    # ---------------------------------------------------------------- read phase
    def _topk(self, built: Built, policy: str, query: str, k: int, ledger: Ledger) -> list[MemoryItem]:
        items = built.store.items if built.store else []
        if not items:
            return []
        if policy == "bm25":
            if built.bm25 is None:
                built.bm25 = BM25([it.text for it in items])
            s = built.bm25.scores(query)
        elif policy == "embedding":
            assert built.vecs is not None
            q = self.embedder.embed([query], ledger=ledger, dp=DP.MEMORY_READ.value)[0]
            s = built.vecs @ q
        else:
            raise ValueError(f"unknown prefilter/retrieval policy {policy!r}")
        order = np.argsort(-s, kind="stable")[:k]
        return [items[i] for i in order]

    def recall(self, built: Built, query: str, ledger: Ledger) -> list[MemoryItem]:
        """Ranked list (best first) for top-k policies; chronological for recency/all."""
        arm, store = self.arm, built.store
        assert store is not None
        k = int(arm.read.get("k", 10))
        pol = arm.policy
        if pol == "all":
            return list(store.items)
        if pol in ("recency", "s1"):
            return store.recall(query, ledger=ledger)
        if pol in ("embedding", "bm25"):
            return self._topk(built, pol, query, k, ledger)
        if pol == "s1_rerank":
            pf = arm.prefilter()
            pre = self._topk(built, pf["policy"], query, int(pf.get("n", 50)), ledger)
            sub = MemoryStore(store.write_cfg, arm.read_dp(), s1=store.s1, max_items=UNLIMITED)
            sub.items = sorted(pre, key=lambda m: m.id)   # chronological: no retrieval-rank position bias
            return sub.recall(query, ledger=ledger)
        raise ValueError(pol)

    def _query(self, qa: QAItem) -> str:
        q = f"CURRENT STEP: answer the following question using long-term memory.\nQUESTION: {qa.question}"
        return q + (f"\nQUESTION DATE: {qa.question_date}" if qa.question_date else "")

    def answer(self, conv: Conversation, built: Built, qa: QAItem) -> dict[str, Any]:
        arm = self.arm
        t0 = time.perf_counter()
        ledger = Ledger()
        rec: dict[str, Any] = {"arm": arm.name, "conv_id": conv.conv_id, "qid": qa.qid, "category": qa.category,
                               "question": qa.question, "answer": qa.answer, "abstention": qa.abstention,
                               "evidence_groups": len(qa.evidence), "error": None}
        try:
            recalled: list[MemoryItem] = []
            if arm.policy == "full_context":
                context, covered = render_full(conv, self.max_context_tokens)
                stored_units: set[str] | None = {t.uid for t in conv.turns()}
            elif arm.policy == "none":
                context, covered, stored_units = "", set(), None
            else:
                assert built.store is not None
                stored_units = {u for it in built.store.items for u in it.meta["units"]}
                if self.s1 is not None:
                    self.s1.set_read_gold({it.id for it in built.store.items
                                           if any(set(g) & set(it.meta["units"]) for g in qa.evidence)})
                t_r = time.perf_counter()
                recalled = self.recall(built, self._query(qa), ledger)
                rec["read_wall_ms"] = (time.perf_counter() - t_r) * 1000
                if arm.policy in ("all", "recency"):   # chronological -> newest first, so a budget drops the oldest
                    recalled = recalled[::-1]
                context, recalled = render_memories(recalled, self.max_memory_tokens)
                covered = {u for it in recalled for u in it.meta["units"]}
            style_prompt = self.prompt or ("longmemeval" if conv.source == "longmemeval" else "locomo")
            prompt, key = build_prompt(conv, qa, context, arm.policy, self.style, style_prompt)
            resp = self.llm.chat([{"role": "user", "content": prompt}], ledger=ledger, decision_point="answer",
                                 temperature=0.0, max_tokens=self.max_answer_tokens)
            raw = (resp.text or "").strip()
            pred = raw
            if self.style == "cot" and "Answer:" in raw and style_prompt == "locomo":
                pred = raw.rsplit("Answer:", 1)[-1].strip()
            if key is not None:
                pred = _cat5_answer(pred, key)
            rec.update({"prediction": pred, "raw_prediction": raw if raw != pred else None,
                        **score_answer(pred, qa),
                        "evidence_recalled": _groups_covered(qa.evidence, covered),
                        "evidence_stored": _groups_covered(qa.evidence, stored_units) if stored_units is not None
                        else None,
                        "n_recalled": len(recalled), "recalled_ids": [m.id for m in recalled],
                        "context_tokens": estimate_tokens(prompt)})
        except Exception as e:  # failures are data; not counted as done on resume
            rec.update({"prediction": "", "em": 0.0, "f1": 0.0, "evidence_recalled": None, "evidence_stored": None,
                        "n_recalled": 0, "recalled_ids": [], "context_tokens": 0,
                        "error": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"})
        rec["wall_ms"] = (time.perf_counter() - t0) * 1000
        rec["usage"] = ledger.summary()
        rec["judge"] = None
        if self.judge_llm is not None and rec["error"] is None:
            jl = Ledger()
            try:
                jr = self.judge_llm.chat([{"role": "user", "content": judge_prompt(qa, rec["prediction"])}],
                                         ledger=jl, decision_point="judge", temperature=0.0, max_tokens=10)
                rec["judge"] = "yes" in (jr.text or "").lower()
                rec["judge_response"] = (jr.text or "").strip()
            except Exception as e:
                rec["judge_error"] = f"{type(e).__name__}: {e}"
            rec["judge_usage"] = _agg(jl.records)
        return rec

    def run_conversation(self, conv: Conversation, skip: set[str] | None = None,
                         emit: Callable[[str, dict[str, Any]], None] | None = None) -> tuple[dict, list[dict]]:
        todo = [q for q in conv.questions if not skip or q.qid not in skip]
        try:
            built = self.build(conv)
        except Exception as e:
            rec = {"conv_id": conv.conv_id, "arm": self.arm.name, "error": f"{type(e).__name__}: {e}\n"
                   f"{traceback.format_exc(limit=3)}"}
            if emit:
                emit("conv", rec)
            return rec, []
        if emit:
            emit("conv", built.record)
        out = []
        for qa in todo:
            r = self.answer(conv, built, qa)
            out.append(r)
            if emit:
                emit("q", r)
        return built.record, out


# ====================================================================== summaries
def _mean(xs: Sequence[float | None]) -> float:
    v = [float(x) for x in xs if x is not None]
    return float(np.mean(v)) if v else float("nan")


def summarize_arm(qrecs: list[dict[str, Any]], crecs: list[dict[str, Any]]) -> dict[str, Any]:
    q_by = {r["qid"]: r for r in qrecs}           # last record per question wins (resume)
    c_by = {r["conv_id"]: r for r in crecs if not r.get("error")}
    qs = list(q_by.values())
    ok = [r for r in qs if not r.get("error")]
    n = len(ok)
    judged = [r["judge"] for r in ok if r.get("judge") is not None]
    out: dict[str, Any] = {"n": n, "errors": len(qs) - n, "stemmer": STEMMER}
    out["f1"] = _mean([r["f1"] for r in ok])
    out["em"] = _mean([r["em"] for r in ok])
    out["judge_acc"] = _mean([float(j) for j in judged]) if judged else float("nan")
    out["judge_n"] = len(judged)
    out["accuracy"], out["accuracy_metric"] = (out["judge_acc"], "judge") if judged else (out["f1"], "f1")
    cats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        cats[r["category"]].append(r)
    out["by_category"] = {c: {"n": len(rs), "f1": _mean([r["f1"] for r in rs]), "em": _mean([r["em"] for r in rs]),
                              "judge_acc": _mean([float(r["judge"]) for r in rs if r.get("judge") is not None])}
                          for c, rs in sorted(cats.items())}
    out["evidence_recall"] = _mean([r.get("evidence_recalled") for r in ok])
    out["evidence_stored"] = _mean([r.get("evidence_stored") for r in ok])
    convs = list(c_by.values())
    offered = sum(c.get("offered", 0) for c in convs)
    stored = sum(c.get("stored", 0) for c in convs)
    out["n_conversations"] = len(convs)
    out["stored_fraction"] = stored / offered if offered else float("nan")
    out["memory_tokens"] = _mean([c.get("memory_tokens") for c in convs])
    out["offered_tokens"] = _mean([c.get("offered_tokens") for c in convs])
    gate: Counter[str] = Counter()
    for c in convs:
        gate.update(c.get("gate") or {})
    if gate:
        out["gate_precision"] = gate["tp"] / max(gate["tp"] + gate["fp"], 1)
        out["gate_recall"] = gate["tp"] / max(gate["tp"] + gate["fn"], 1)
    out["context_tokens"] = _mean([r.get("context_tokens") for r in ok])

    def u(r: dict[str, Any], comp: str, key: str) -> float:
        return float((r.get("usage") or {}).get(comp, {}).get(key, 0.0))

    def dp(r: dict[str, Any], d: str, key: str) -> float:
        return float((r.get("usage") or {}).get("by_decision_point", {}).get(d, {}).get(key, 0.0))

    write_cost = sum(u(c, "total", "cost_usd") for c in convs)
    q_cost = sum(u(r, "total", "cost_usd") for r in qs)
    nq = max(len(qs), 1)
    out["cost_usd_per_q"] = (write_cost + q_cost) / nq                     # write cost amortised over questions
    out["write_cost_usd_per_conv"] = write_cost / max(len(convs), 1)
    out["read_cost_usd_per_q"] = sum(dp(r, DP.MEMORY_READ.value, "cost_usd") for r in qs) / nq
    out["answer_cost_usd_per_q"] = sum(dp(r, "answer", "cost_usd") for r in qs) / nq
    out["answer_input_tokens_per_q"] = sum(dp(r, "answer", "input_tokens") for r in qs) / nq
    out["latency_ms_per_q"] = _mean([u(r, "total", "latency_ms") for r in ok])       # read + answer (critical path)
    out["read_latency_ms_per_q"] = _mean([dp(r, DP.MEMORY_READ.value, "latency_ms") for r in ok])
    out["wall_ms_per_q"] = _mean([r.get("wall_ms") for r in ok])
    out["write_latency_ms_per_candidate"] = sum(u(c, "total", "latency_ms") for c in convs) / max(offered, 1)
    out["s1_calls_per_q"] = _mean([u(r, "s1", "calls") for r in ok])
    out["llm_calls_per_q"] = _mean([u(r, "s2", "calls") for r in ok])
    out["write_s1_calls_per_conv"] = _mean([u(c, "s1", "calls") for c in convs])
    out["write_llm_calls_per_conv"] = _mean([u(c, "s2", "calls") for c in convs])
    out["judge_cost_usd"] = sum(float((r.get("judge_usage") or {}).get("cost_usd", 0.0)) for r in qs)
    out["unpriced_models"] = sorted({m for r in qs + convs for m in (r.get("usage") or {}).get("unpriced_models", [])})
    return out


TABLE_COLS = [("n", "n"), ("accuracy", "acc"), ("f1", "F1"), ("em", "EM"), ("judge_acc", "judge"),
              ("evidence_recall", "ev.recall"), ("evidence_stored", "ev.stored"), ("stored_fraction", "stored"),
              ("memory_tokens", "mem.tok"), ("context_tokens", "ctx.tok"), ("cost_usd_per_q", "$/q"),
              ("write_cost_usd_per_conv", "write$/conv"), ("latency_ms_per_q", "lat.ms/q"),
              ("s1_calls_per_q", "s1/q"), ("llm_calls_per_q", "llm/q")]


def format_table(summary: dict[str, dict[str, Any]]) -> str:
    def f(v: Any) -> str:
        if isinstance(v, float):
            if math.isnan(v):
                return "-"
            return f"{v:.2e}" if 0 < abs(v) < 1e-3 else (f"{v:.4f}" if abs(v) < 10 else f"{v:.0f}")
        return str(v)
    lines = ["| arm | " + " | ".join(h for _, h in TABLE_COLS) + " |",
             "|" + "---|" * (len(TABLE_COLS) + 1)]
    for arm, s in summary.items():
        lines.append(f"| {arm} | " + " | ".join(f(s.get(k, float("nan"))) for k, _ in TABLE_COLS) + " |")
    return "\n".join(lines)


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def summarize_dir(run_dir: str | Path, arms: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    run_dir = Path(run_dir)
    names = list(arms) if arms else sorted(p.name[:-6] for p in run_dir.glob("*.jsonl")
                                           if not p.name.endswith(".conversations.jsonl"))
    return {a: summarize_arm(_read_jsonl(run_dir / f"{a}.jsonl"), _read_jsonl(run_dir / f"{a}.conversations.jsonl"))
            for a in names}


# ====================================================================== experiment runner
def run_memory_experiment(exp: dict[str, Any], out_dir: str | Path, llm: LLMClient | None = None,
                          s1: DecisionBackend | None = None, judge_llm: LLMClient | None = None,
                          embedder: Any = None, max_conversations: int | None = None,
                          max_questions: int | None = None, arms: Sequence[str] | None = None,
                          progress: bool = True) -> dict[str, dict[str, Any]]:
    """Run every arm of a memory-QA experiment YAML (see configs/memory/). Resumable: questions already in
    ``<out>/<name>/<arm>.jsonl`` without an error are skipped. Returns {arm: summary}."""
    from ...decision import make_backend

    out = Path(out_dir) / exp.get("name", "memory_qa")
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))
    ds = dict(exp["dataset"])
    if max_questions:
        ds["max_questions"] = max_questions
    convs = load_dataset(ds)[:max_conversations] if max_conversations else load_dataset(ds)
    arm_list = [parse_arm(a) for a in expand_arms(exp)]
    if arms:
        arm_list = [a for a in arm_list if a.name in set(arms)]
    llm = llm or make_llm_any(exp["llm"])
    if judge_llm is None and exp.get("judge"):
        judge_llm = make_llm_any(exp["judge"])
    llm_pool = {name: make_llm_any(spec) for name, spec in (exp.get("llms") or {}).items()
                if any(name in (a.llm, a.write_llm) for a in arm_list)}
    default_write_llm = make_llm_any(exp["write_llm"]) if exp.get("write_llm") and any(
        a.write.mode == Mode.LLM and not a.write_llm for a in arm_list) else None
    if s1 is None and any(a.needs_s1() and not a.s1 for a in arm_list):
        s1 = make_backend(exp["s1"], cache=exp.get("s1_cache"))
    s1_pool = {name: make_backend(spec, cache=exp.get("s1_cache"))
               for name, spec in (exp.get("s1_backends") or {}).items() if any(a.s1 == name for a in arm_list)}
    if embedder is None and any(a.needs_embedder() for a in arm_list):
        embedder = make_embedder(exp.get("embedder"))
    opts = {k: exp[k] for k in ("granularity", "write_context_turns", "max_item_chars", "answer",
                                "write_gate_min_tokens") if k in exp}
    workers = int(exp.get("workers", os.environ.get("JEV_WORKERS", 4)))
    lock = threading.Lock()
    for arm in arm_list:
        engine = MemoryQAEngine(arm, llm=llm_pool[arm.llm] if arm.llm else llm,
                                s1=s1_pool[arm.s1] if arm.s1 else s1,
                                write_llm=(llm_pool[arm.write_llm] if arm.write_llm else default_write_llm),
                                embedder=embedder, judge_llm=judge_llm, opts=opts)
        qpath, cpath = out / f"{arm.name}.jsonl", out / f"{arm.name}.conversations.jsonl"
        done = {r["qid"] for r in _read_jsonl(qpath) if not r.get("error")}
        jobs = [c for c in convs if any(q.qid not in done for q in c.questions)]
        if not jobs:
            continue

        def emit(kind: str, rec: dict[str, Any], qpath: Path = qpath, cpath: Path = cpath) -> None:
            with lock, open(qpath if kind == "q" else cpath, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

        bar = None
        if progress:
            from tqdm import tqdm
            bar = tqdm(total=sum(len([q for q in c.questions if q.qid not in done]) for c in jobs),
                       desc=f"{arm.name} [{arm.short()}]")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = [ex.submit(engine.run_conversation, c, done, emit) for c in jobs]
            for fut in as_completed(futs):
                _, qrecs = fut.result()
                if bar is not None:
                    bar.update(len(qrecs))
        if bar is not None:
            bar.close()
    summary = summarize_dir(out, [a.name for a in arm_list])
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "summary.md").write_text(format_table(summary) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # pragma: no cover
        pass
    p = argparse.ArgumentParser("python -m jevharness.bench.suites.memory_qa",
                                description="Long-term-memory QA (LoCoMo / LongMemEval) for Jev-for-memory arms.")
    p.add_argument("--config", required=True, help="experiment YAML (configs/memory/*.yaml)")
    p.add_argument("--out", default="runs/memory")
    p.add_argument("--max-conversations", type=int, default=None)
    p.add_argument("--max-questions", type=int, default=None, help="per conversation")
    p.add_argument("--arms", default=None, help="comma-separated subset of arm names")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--report-only", action="store_true", help="only summarise existing results")
    p.add_argument("--override", "-o", nargs="*", default=[], help="dotted.key=value overrides of the YAML")
    a = p.parse_args(argv)
    exp = yaml.safe_load(Path(a.config).read_text())
    for kv in a.override:
        k, v = kv.split("=", 1)
        node = exp
        parts = k.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(v)
    if a.no_judge:
        exp["judge"] = None
    arms = a.arms.split(",") if a.arms else None
    if a.report_only:
        summary = summarize_dir(Path(a.out) / exp.get("name", "memory_qa"), arms)
    else:
        summary = run_memory_experiment(exp, a.out, max_conversations=a.max_conversations,
                                        max_questions=a.max_questions, arms=arms)
    print(format_table(summary))


if __name__ == "__main__":
    main()
