"""Lexical (sparse) retrieval baselines for CHOICE questions: BM25 and TF-IDF cosine.

These are the classic non-learned rankers used as tool-retrieval baselines (Gorilla's BM25 retriever,
ToolLLM's API retriever comparison, the ToolRet benchmark) and as memory rankers. Each option is a
document (``option`` + its description, when given); the query is the rendered state plus the question
text. Scores become a distribution with a softmax at ``temperature``, so the backends plug into every
Choice decision point and every mode (replace / cascade / assist; tool_select, tool_params, memory_read,
action) exactly like Jev. They are free and run locally in well under a millisecond per decision.
Abstain options ("none of these", OTHER) carry no lexical evidence: they score 0 (as a document without
any query term) unless ``abstain_score`` sets a retrieval threshold for them.

They cannot answer NOUL / SCORE questions. Pass ``fallback`` (any ``make_backend`` spec, for example
``{"type": "heuristic"}``) to route those to another backend; without it they raise
``NotImplementedError``, as :class:`~jevharness.decision.embedding.EmbeddingBackend` does.

Tokenisation is built for tool names: ``getUserOrders``, ``get_user_orders``, ``get-user-orders`` and
``get.user.orders`` all become ``get user order``. It lowercases, drops English stopwords and
single letters, keeps digit runs (IDs), and applies a light, deterministic suffix stemmer
(``orders``->``order``, ``shipped``/``shipping``->``ship``, ``cancelled``->``cancel``).

References: Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond*;
Salton & Buckley (1988), *Term-weighting approaches in automatic text retrieval*.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any

import numpy as np

from .base import DecisionBackend
from .types import JEV_MAX_OPTIONS, Answer, DecisionResult, QKind, Question, Usage, answer_from_distribution

# Part of describe() and hence of the decision-cache key: bump it when tokenisation or scoring changes.
# 2: abstain options ("none" / OTHER) score 0 by default instead of their lexical score, see ``abstain_score``
VERSION = 2

# ------------------------------------------------------------------ tokenisation

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours yourself yourselves
please shall may might must also us
""".split())

_CAMEL_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")      # HTTPServer -> HTTP Server
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")                   # getUser -> get User
_WORD = re.compile(r"[A-Za-z]+|\d+")                        # splits on _ - . / : and letter|digit changes
# rendering artefacts of the harness state (section headers) are not query terms
_HEADERS = re.compile(r"^(TASK|MEMORY|HISTORY|CURRENT OBSERVATION|CANDIDATE MEMORY):", re.M)


def stem(tok: str) -> str:
    """Light suffix stripper (S-stemmer + -ing/-ed). Deterministic and consistent: every surface form
    goes through the same rules, and a final doubled consonant is collapsed, so ``ship``/``shipped``/
    ``shipping`` share the stem ``ship`` and ``cancel``/``cancelled`` share ``cancel``."""
    if not tok.isalpha() or len(tok) <= 3:
        return tok
    t = tok
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("ied") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("sses"):
        t = t[:-2]
    elif t.endswith("ing") and len(t) > 5:
        t = t[:-3]
    elif t.endswith("ed") and len(t) > 4:
        t = t[:-2]
    elif t.endswith(("ches", "shes", "xes", "zes")):
        t = t[:-2]
    elif t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    if len(t) > 3 and t[-1] == t[-2] and t[-1] not in "aeiou":
        t = t[:-1]
    return t


def tokenize(text: str, do_stem: bool = True, stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    s = _CAMEL_ACRONYM.sub(r"\1 \2", text or "")
    s = _CAMEL.sub(r"\1 \2", s)
    out = []
    for w in _WORD.findall(s):
        w = w.lower()
        if w in stopwords or (len(w) == 1 and not w.isdigit()):
            continue
        out.append(stem(w) if do_stem else w)
    return out


def split_sections(state: str) -> dict[str, str]:
    """Split a harness-rendered state (``JevAgent._s1_state``) into its sections:
    ``TASK``, ``MEMORY``, ``HISTORY``, ``CURRENT OBSERVATION`` (+ ``CANDIDATE MEMORY`` for the memory
    write gate). Returns ``{}`` for states in any other format (static suites, the wire server)."""
    marks = list(_HEADERS.finditer(state or ""))
    out: dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(state)
        out[m.group(1)] = state[m.end():end].strip()
    return out


# options that mean "none of these" / "defer" (BFCL NONE_TOOL, the harness OTHER option, the empty-recall
# option of memory_read); a lexical scorer has no evidence *for* them, see ``abstain_score``. Scoring them
# lexically matches boilerplate instead: on BFCL the constant question ("Which function should be called to
# handle the user's request?") shares "function" and "request" with NONE_TOOL's text, so v1 BM25 / TF-IDF
# answered "none" on 45% of the BFCL rows that need a tool (E1 dev shakedown, runs/dev/e1_dev/NOTES.md).
ABSTAIN_RE = re.compile(r"^\W*(none\b|no\b|<other\b|other\b)", re.I)


# ------------------------------------------------------------------ scorers

def bm25_scores(query: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75,
                k3: float | None = 8.0) -> np.ndarray:
    """Okapi BM25 of every doc for the query. IDF uses the non-negative (Lucene) form
    ``log(1 + (N - df + 0.5) / (df + 0.5))`` computed over the candidate set; query term frequency
    is saturated with ``k3`` (``None`` = linear, i.e. every query occurrence counts)."""
    n = len(docs)
    if n == 0:
        return np.zeros(0)
    avgdl = (sum(len(d) for d in docs) / n) or 1.0
    df = Counter(t for d in docs for t in set(d))
    qtf = Counter(query)
    out = np.zeros(n)
    for i, d in enumerate(docs):
        tf = Counter(d)
        norm = k1 * (1.0 - b + b * len(d) / avgdl)
        s = 0.0
        for t, qf in qtf.items():
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5))
            wq = qf if k3 is None else (k3 + 1.0) * qf / (k3 + qf)
            s += idf * f * (k1 + 1.0) / (f + norm) * wq
        out[i] = s
    return out


def tfidf_cosine(query: list[str], docs: list[list[str]], sublinear_tf: bool = True) -> np.ndarray:
    """Cosine similarity of TF-IDF vectors (smoothed IDF over the candidate set, as scikit-learn's
    ``TfidfVectorizer(smooth_idf=True, sublinear_tf=True)``). Query terms absent from every doc are ignored."""
    n = len(docs)
    if n == 0:
        return np.zeros(0)
    df = Counter(t for d in docs for t in set(d))
    vocab = {t: j for j, t in enumerate(sorted(df))}
    idf = np.array([math.log((1.0 + n) / (1.0 + df[t])) + 1.0 for t in vocab])

    def vec(toks: list[str]) -> np.ndarray:
        v = np.zeros(len(vocab))
        for t, c in Counter(toks).items():
            j = vocab.get(t)
            if j is not None:
                v[j] = (1.0 + math.log(c)) if sublinear_tf else c
        v *= idf
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v

    qv = vec(query)
    return np.array([float(qv @ vec(d)) for d in docs])


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / max(float(temperature), 1e-9)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ------------------------------------------------------------------ backends

class LexicalBackend(DecisionBackend):
    """Shared plumbing: query/doc construction, softmax, NOUL/SCORE fallback, zero-cost accounting."""

    name = "lexical"
    batched_questions = True
    max_options = JEV_MAX_OPTIONS
    default_temperature = 1.0

    def __init__(self, temperature: float | None = None, query_mode: str = "full", include_question: bool = True,
                 include_descriptions: bool = True, stem: bool = True,
                 abstain_score: float | str | None = None, fallback: dict[str, Any] | str | None = None):
        """
        temperature           softmax temperature on the raw scores (confidence scale; ``calibrated`` can refit it).
        query_mode            "full": the whole rendered state (what every System-1 backend sees);
                              "task_obs": only the TASK and CURRENT OBSERVATION sections of harness states
                              (the usual tool-retrieval query: instruction + latest context), falling back to
                              the full state for states without those sections.
        include_question      append the question text to the query.
        include_descriptions  documents are ``option + description`` (else option names only).
        abstain_score         how "none / other" options (``ABSTAIN_RE``) are scored. ``None`` (default): 0, the
                              score of a document with no lexical evidence, so they never beat a real option
                              that matches the query (and tie, as the last options, when nothing matches).
                              A number: this fixed score (a retrieval threshold for irrelevance detection; tune
                              it on a dev split). ``"lexical"``: their own lexical score (v1, for ablations).
        fallback              backend spec answering NOUL / SCORE questions (else NotImplementedError).
        """
        if query_mode not in ("full", "task_obs"):
            raise ValueError("query_mode must be 'full' or 'task_obs'")
        self.temperature = float(self.default_temperature if temperature is None else temperature)
        self.query_mode = query_mode
        self.include_question = include_question
        self.include_descriptions = include_descriptions
        self.stem = stem
        if isinstance(abstain_score, str) and abstain_score != "lexical":
            raise ValueError("abstain_score must be None, a number or 'lexical'")
        self.abstain_score = abstain_score if abstain_score is None or isinstance(abstain_score, str) \
            else float(abstain_score)
        self.fallback_spec = fallback
        self._fallback: DecisionBackend | None = None
        if fallback is not None:
            from . import make_backend  # late import: the package imports this module lazily
            self._fallback = make_backend(fallback)
        self.model = self.name

    # -- scoring ----------------------------------------------------------------------
    def score(self, query: list[str], docs: list[list[str]]) -> np.ndarray:
        raise NotImplementedError

    def query_text(self, state: str, q: Question | None = None) -> str:
        text = state
        if self.query_mode == "task_obs":
            sec = split_sections(state)
            parts = [sec[k] for k in ("TASK", "CANDIDATE MEMORY", "CURRENT OBSERVATION") if sec.get(k)]
            text = "\n".join(parts) if parts else state
        text = _HEADERS.sub(" ", text)
        if q is not None and self.include_question:
            text = f"{text}\n{q.text}"
        return text

    def doc_texts(self, q: Question) -> list[str]:
        assert q.options is not None
        descs = q.option_descriptions if self.include_descriptions and q.option_descriptions else None
        descs = descs or [None] * len(q.options)
        return [o if not d else f"{o} {d}" for o, d in zip(q.options, descs)]

    def rank_scores(self, query_text: str, docs: list[str]) -> np.ndarray:
        return self.score(tokenize(query_text, self.stem), [tokenize(d, self.stem) for d in docs])

    def _choice(self, state: str, q: Question) -> Answer:
        assert q.options is not None
        scores = self.rank_scores(self.query_text(state, q), self.doc_texts(q))
        abstain = np.array([bool(ABSTAIN_RE.match(o)) for o in q.options])
        if self.abstain_score != "lexical" and abstain.any() and not abstain.all():
            # default: the score of a document without lexical evidence (0 for BM25 and TF-IDF cosine)
            scores[abstain] = 0.0 if self.abstain_score is None else float(self.abstain_score)
        probs = softmax(scores, self.temperature)
        return answer_from_distribution(q, dict(zip(q.options, probs.tolist())),
                                        raw={"scores": [round(float(s), 4) for s in scores]})

    # -- DecisionBackend ----------------------------------------------------------------
    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        t0 = time.perf_counter()
        answers: list[Answer | None] = [None] * len(questions)
        other = [i for i, q in enumerate(questions) if q.kind != QKind.CHOICE]
        if other and self._fallback is None:
            raise NotImplementedError(f"{type(self).__name__} only answers CHOICE questions; "
                                      "set `fallback` (e.g. {'type': 'heuristic'}) for NOUL/SCORE")
        for i, q in enumerate(questions):
            if q.kind == QKind.CHOICE:
                answers[i] = self._choice(state, q)
        usage = Usage(backend=self.name, model=self.model)
        if other:
            assert self._fallback is not None
            fb = self._fallback._decide(state, [questions[i] for i in other])
            for i, a in zip(other, fb.answers):
                answers[i] = a
            usage.input_tokens, usage.output_tokens = fb.usage.input_tokens, fb.usage.output_tokens
            usage.cost_usd = fb.usage.cost_usd
            usage.latency_ms = fb.usage.latency_ms
            usage.n_calls = 1 + fb.usage.n_calls
            usage.priced = getattr(fb.usage, "priced", None)
        usage.latency_ms += (time.perf_counter() - t0) * 1000
        return DecisionResult(answers=[a for a in answers if a is not None], usage=usage)

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"backend": self.name, "model": self.model, "version": VERSION,
                             "temperature": self.temperature,
                             "query_mode": self.query_mode, "include_question": self.include_question,
                             "include_descriptions": self.include_descriptions, "stem": self.stem}
        if self.abstain_score is not None:
            d["abstain_score"] = self.abstain_score
        if self._fallback is not None:
            d["fallback"] = self._fallback.describe()
        return d


class BM25Backend(LexicalBackend):
    """Okapi BM25 over option documents (k1=1.2, b=0.75, the defaults of Robertson & Zaragoza 2009).
    Raw BM25 scores are additive log-odds-like evidence, so the default temperature is 1."""

    name = "bm25"
    default_temperature = 1.0

    def __init__(self, k1: float = 1.2, b: float = 0.75, k3: float | None = 8.0, **kw: Any):
        self.k1, self.b, self.k3 = k1, b, k3
        super().__init__(**kw)

    def score(self, query: list[str], docs: list[list[str]]) -> np.ndarray:
        return bm25_scores(query, docs, self.k1, self.b, self.k3)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "k1": self.k1, "b": self.b, "k3": self.k3}


class TfidfBackend(LexicalBackend):
    """TF-IDF cosine similarity (sublinear TF, smoothed IDF). Cosines lie in [0, 1], so the default
    temperature (0.05) matches the dense-embedding baseline's."""

    name = "tfidf"
    default_temperature = 0.05

    def __init__(self, sublinear_tf: bool = True, **kw: Any):
        self.sublinear_tf = sublinear_tf
        super().__init__(**kw)

    def score(self, query: list[str], docs: list[list[str]]) -> np.ndarray:
        return tfidf_cosine(query, docs, self.sublinear_tf)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "sublinear_tf": self.sublinear_tf}
