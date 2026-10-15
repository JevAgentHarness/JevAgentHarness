"""Intern-Decision (InternLM, 2026-09-26) as an in-process System-1 backend.

``internlm/Intern-Decision-{0.8B,2B,4B}`` (Apache-2.0, decision-tuned from Qwen3.5-{0.8B,2B,4B}) answer ALL typed
questions about a shared state in ONE causal forward pass, with no generation -- within the reference limits of 16
fields per prompt and 62 options per field. More fields take one pass per 16; a wider Choice is a knockout over
several passes (``_knockout``, decision/tournament.py), and the harness tournament splits a wider tool set into
<= 62-option chunks before it reaches this backend (69 JevGym tools: two passes). ``Usage.n_passes`` / ``n_rounds``
and the ledger record's ``meta`` carry the actual counts:

1. every question's options are mapped, in order, to single-token symbols ``A..Z a..z 0..9`` (<= 62 options);
2. one prompt is compiled: a fixed system prompt, a user message ``## State <json> ## Decision schema
   <field: instructions / symbol = value: description>`` and a complete *assistant* JSON skeleton with one
   ``<decision>`` placeholder per field (``{"team": "<decision>", ...}``), rendered with the checkpoint's chat
   template and its empty ``<think>`` block (``enable_thinking=False``);
3. the logits at the position immediately BEFORE each placeholder (``logits_to_keep=positions``) are softmaxed over
   that field's allowed symbols and temperature-scaled with the checkpoint's fitted calibration (0.8B: T=2.748,
   2B: 2.101, 4B: 1.992; it never changes the argmax);
4. symbols map back to the option values: ``choice``, ``noul`` = P(yes), ``score`` = expected value, each with a
   ``probabilities`` dict and ``confidence`` = max probability -- the Jev response shape.

This differs from Jev-wire backends (``jev.py``: one HTTP request, model-side batching) and from our trained deciders
(``training/serve_jev.py``: one prompt per question, logits at the last position): a multi-field read-out at
in-sequence placeholder positions, so a request with k questions costs one prefill, not k.

Prompt compilation and the probability arithmetic below (``_json``, ``_options``, ``compile_row``, ``argmax``,
``validate_temperature``, ``scale_probabilities``) are vendored from ``inference.py`` of
``internlm/Intern-Decision-0.8B`` (revision 85a0cc5, 2026-09-26; identical in the 2B / 4B repos apart from
``MODEL_NAME`` / ``DEFAULT_TEMPERATURE``), Copyright (c) InternLM, licensed under the Apache License 2.0 --
re-stated here so the prompt is byte-identical to the reference without importing their file at runtime (it needs
Python >= 3.12 f-string syntax and imports Pillow + the vision processor). Deviations are marked ``# jevharness:``.
The model is loaded text-only (``Qwen3_5ForCausalLM`` via ``training.train_decision.load_causal_lm``: no vision
tower, no processor, no Pillow); images are not supported.

Spec::

    {"type": "intern_decision", "model": "internlm/Intern-Decision-2B"}                       # defaults below
    {"type": "intern_decision", "model": "internlm/Intern-Decision-0.8B", "device": "cuda", "dtype": "bfloat16",
     "max_length": 8192, "temperature": 2.747760550703, "max_questions_per_request": 16, "max_options": 62,
     "truncate_state": true, "state_json": true, "knockout_probs": "linked"}

``knockout_probs`` composes the reads of a knockout (Choice wider than ``max_options``) into the full-set
distribution: ``linked`` (default) or ``finalists`` (the output before 2026-10-02; decision/tournament.py).
``temperature`` defaults to the checkpoint's fitted value (parsed from its ``inference.py``, else the vendored table);
``temperature: 1`` gives the uncalibrated probabilities. ``device: auto`` = cuda if available else cpu; ``dtype``
defaults to bfloat16 on cuda and float32 on cpu. Instances with the same checkpoint / device / dtype share one copy
of the weights (``describe()['shared_weights']``), forward passes on one device are serialised across instances,
and a CUDA out-of-memory error is retried once after freeing the caching allocator (``n_oom_retries``).
Prompts longer than ``max_length`` tokens drop the oldest part of
the state (``truncate_state: true``; the reference rejects them) and flag ``raw["truncated_state"]``. Every call
is booked on the ledger: ``input_tokens`` = prompt tokens per forward pass, ``output_tokens`` = scored fields (the
reference's convention), ``latency_ms`` = forward-pass time, cost from ``configs/prices/models.yaml``.
"""

from __future__ import annotations

import gc
import json
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..accounting import default_prices
from .base import DecisionBackend
from .jev import question_to_wire
from .tournament import DEFAULT_KNOCKOUT, check_method, full_set_distribution
from .types import Answer, DecisionResult, QKind, Question, Usage, answer_from_distribution

# ---------------------------------------------------------------------------------------------------------------
# Vendored from internlm/Intern-Decision-0.8B inference.py (Apache-2.0, InternLM). Keep byte-for-byte semantics.
# ---------------------------------------------------------------------------------------------------------------

DECISION_TOKEN = "<decision>"
ANSWER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
SYSTEM_PROMPT = (
    "You are a careful decision assistant. Use the state and decision schema in the user message to make the "
    "requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed "
    "options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names "
    "and symbols exactly as given. Do not include explanations, Markdown, or extra text."
)
MAX_FIELDS = 16  # validate_request(): 1-16 questions per request
MAX_SYMBOLS = len(ANSWER_SYMBOLS)

# DEFAULT_TEMPERATURE of each checkpoint's inference.py (NLL-fitted on 1,728 calibration cases per the model card),
# used when the file is not available locally. ``checkpoint_temperature`` prefers the shipped file.
KNOWN_TEMPERATURES: dict[str, float] = {
    "internlm/Intern-Decision-0.8B": 2.747760550703,
    "internlm/Intern-Decision-2B": 2.100509348278,
    "internlm/Intern-Decision-4B": 1.99241824,
}


@dataclass(frozen=True)
class CompiledExample:
    messages: list[dict[str, Any]]
    fields: tuple[str, ...]
    symbols: dict[str, tuple[str, ...]]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


def _options(question: dict[str, Any]) -> list[tuple[str, str]]:
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict):
            raise ValueError("choice criteria must be an object")
        return [(str(key), str(value)) for key, value in criteria.items()]
    if kind == "score":
        if isinstance(criteria, list):
            return [(str(index), str(value)) for index, value in enumerate(criteria)]
        if isinstance(criteria, dict):
            return [(str(key), str(value)) for key, value in criteria.items()]
        raise ValueError("score criteria must be a list or object")
    if kind == "noul":
        descriptions = criteria if isinstance(criteria, dict) else {}
        yes = next((str(descriptions[k]) for k in descriptions if str(k).lower() in {"yes", "true", "1"}),
                   "The answer is yes (affirmative, or align with the claim).")
        no = next((str(descriptions[k]) for k in descriptions if str(k).lower() in {"no", "false", "0"}),
                  "The answer is no (negative, or disagree with the claim).")
        return [("no", no), ("yes", yes)]
    raise ValueError(f"unsupported question type: {kind!r}")


def compile_row(row: dict[str, Any], raw_state: bool = False) -> CompiledExample:
    """Reference prompt compiler (text-only). ``raw_state`` is a jevharness addition: embed a string state verbatim
    under ``## State`` instead of as a JSON string literal (the reference always JSON-encodes the state)."""
    questions = row.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError(f"{row.get('id', '<missing id>')}: questions must be a non-empty object")
    fields: list[str] = []
    symbols: dict[str, tuple[str, ...]] = {}
    schema_lines: list[str] = []
    for field, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(f"{row.get('id', '<missing id>')}: question {field!r} is not an object")
        options = _options(question)
        if not options:
            raise ValueError(f"{row.get('id', '<missing id>')}: question {field!r} has no options")
        if len(options) > len(ANSWER_SYMBOLS):
            raise ValueError(f"At most {len(ANSWER_SYMBOLS)} single-token answer symbols are supported")
        field_name = str(field)
        fields.append(field_name)
        field_symbols = tuple(ANSWER_SYMBOLS[: len(options)])
        symbols[field_name] = field_symbols
        schema_lines.append(f"{field_name}: {question.get('instructions', '')}")
        for symbol, (value, description) in zip(field_symbols, options):
            schema_lines.append(f"    {symbol} = {value}: {description}")
    state_value = row.get("state")
    state = state_value if raw_state and isinstance(state_value, str) else _json(state_value)  # jevharness: raw_state
    user_text = ("Return one answer for every field using the supplied answer symbols.\n\n## State\n"
                 f"{state}\n## Decision schema\n" + "\n".join(schema_lines))
    if DECISION_TOKEN in user_text:
        raise ValueError("Reserved decision marker appears in input evidence")
    skeleton = json.dumps(dict.fromkeys(fields, DECISION_TOKEN), ensure_ascii=False, indent=4)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text},
                {"role": "assistant", "content": skeleton}]
    return CompiledExample(messages=messages, fields=tuple(fields), symbols=symbols)


def argmax(probs: dict[str, float]) -> str:
    return min(probs, key=lambda label: (-probs[label], label))


def validate_temperature(temperature: Any) -> float:
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return temperature


def scale_probabilities(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """``softmax(log(p) / T)``: candidate-probability calibration that preserves the argmax."""
    temperature = validate_temperature(temperature)
    if not probs or any((not math.isfinite(p) or p < 0 or p > 1 for p in probs.values())):
        raise ValueError("Invalid probability distribution")
    if abs(sum(probs.values()) - 1.0) > 0.001:
        raise ValueError("Probabilities must sum to one")
    if temperature == 1.0:
        return dict(probs)
    logs = {label: math.log(p) if p else -math.inf for label, p in probs.items()}
    maximum = max(logs.values())
    weights = {label: math.exp((logp - maximum) / temperature) for label, logp in logs.items()}
    total = sum(weights.values())
    result = {label: p / total for label, p in weights.items()}
    if argmax(result) != argmax(probs):
        raise ArithmeticError("Floating-point temperature scaling changed argmax")
    return result


def read_fields(logits: Any, compiled: CompiledExample, questions: dict[str, dict[str, Any]],
                symbol_ids: dict[str, int], temperature: float) -> dict[str, dict[str, Any]]:
    """Reference read-out: ``logits[i]`` is the vocabulary row BEFORE field ``i``'s placeholder. Softmax over that
    field's allowed symbols (in float32), temperature-scale, then derive the typed answer from the scaled
    probabilities exactly as ``DecisionEngine.predict`` + ``scale_result`` do."""
    import torch

    answers: dict[str, dict[str, Any]] = {}
    for index, field in enumerate(compiled.fields):
        question = questions[field]
        options = _options(question)
        values = [value for value, _ in options]
        ids = [symbol_ids[s] for s in compiled.symbols[field]]
        probabilities = torch.softmax(logits[index, ids].float(), dim=-1).cpu().tolist()
        probs = scale_probabilities(dict(zip(values, probabilities)), temperature)
        best = argmax(probs)
        kind = question["type"]
        answer: dict[str, Any] = {"type": kind, "probabilities": probs, "confidence": probs[best], "decision": best}
        if kind == "noul":
            answer["noul"] = probs["yes"]
        elif kind == "score":
            answer["score"] = sum(float(value) * probs[value] for value in values)
            answer["legend"] = dict(options)
        else:
            answer["choice"] = best
        answers[field] = answer
    return answers


# ---------------------------------------------------------------------------------------------------------------
# jevharness glue: our typed questions <-> the reference request / answer dicts
# ---------------------------------------------------------------------------------------------------------------

# Weights are shared between backend instances of the same (checkpoint, device, dtype, attention) -- an experiment
# binds e.g. ``intern4b`` and ``intern4b_uncal`` (temperature 1) to the same checkpoint, and loading it twice is
# what pushed GPU round 2 (L4, 24 GB: 4B x2 + 2B + 0.8B) into CUDA OOM. The forward-pass lock is per device, so
# instances sharing one GPU never overlap their prefills (activation memory is bounded by one pass at a time).
_SHARED: dict[tuple[str, str, str, str | None], Any] = {}
_DEVICE_LOCKS: dict[str, threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


def shared_weights(model: str, device: Any, dtype_name: str, attn: str | None, loader: Any) -> tuple[Any, bool]:
    """``(weights, reused)``: the cached module for the key, or ``loader()`` once (under a lock) and cached."""
    key = (model, str(device), dtype_name, attn)
    with _REGISTRY_LOCK:
        m = _SHARED.get(key)
        if m is not None:
            return m, True
        m = loader()
        _SHARED[key] = m
        return m, False


def device_lock(device: Any) -> threading.Lock:
    with _REGISTRY_LOCK:
        return _DEVICE_LOCKS.setdefault(str(device), threading.Lock())


def release_shared_weights() -> int:
    """Drop every cached checkpoint (tests / notebooks); returns how many were held."""
    with _REGISTRY_LOCK:
        n = len(_SHARED)
        _SHARED.clear()
    return n


def _oom_error() -> type[BaseException]:
    import torch

    return getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


_TEMP_RE = re.compile(r"^DEFAULT_TEMPERATURE\s*=\s*([0-9][0-9.eE+-]*)\s*$", re.M)


def checkpoint_temperature(model: str) -> tuple[float | None, str | None]:
    """The calibration temperature fitted for ``model``: parsed (never executed) from the ``inference.py`` shipped
    with the checkpoint -- a local directory, or the Hub (served from the HF cache when offline) -- else the vendored
    table. ``(None, None)`` when neither knows the checkpoint."""
    text: str | None = None
    p = Path(model)
    if p.is_dir():
        f = p / "inference.py"
        text = f.read_text() if f.exists() else None
    else:
        try:
            from huggingface_hub import hf_hub_download

            text = Path(hf_hub_download(model, "inference.py")).read_text()
        except Exception:  # offline and not cached, not a hub id, no such file: fall back to the table
            text = None
    if text:
        m = _TEMP_RE.search(text)
        if m:
            return validate_temperature(m.group(1)), "checkpoint inference.py"
    for key, t in KNOWN_TEMPERATURES.items():
        if model == key or model.rstrip("/").endswith("/" + key.split("/")[-1]):
            return t, "vendored table"
    return None, None


def question_to_intern(q: Question) -> dict[str, Any]:
    """Our typed question -> an Intern-Decision (Jev-style) question dict. Choice and Noul are the Jev wire form
    (``jev.question_to_wire``: option -> description | "", Noul ``criteria`` from ``meta["criteria"]`` only); Score
    uses the object form keyed by OUR scale values (``{"1": "lowest", ...}``), so the reference's expected ``score``
    is already on our scale and ``probabilities`` are keyed by our labels. Nothing else from ``Question.meta`` is
    used (gold labels never reach the prompt)."""
    w = question_to_wire(q)
    if q.kind == QKind.CHOICE:
        w["criteria"] = {k: ("" if v is None else v) for k, v in w["criteria"].items()}
    elif q.kind == QKind.SCORE:
        lo, _ = q.scale  # type: ignore[misc]
        w["criteria"] = {str(lo + i): d for i, d in enumerate(w["criteria"])}
    return w


def field_names(questions: list[Question]) -> list[str]:
    """JSON field names for the skeleton: the wire client's own field name (``meta["wire_id"]``, set by
    ``decision/server.py``), else the harness decision point (``tool_select``, ``stop``, ...), else ``q<i>``;
    de-duplicated with ``_2``, ``_3``, ... so the skeleton stays a valid object."""
    names: list[str] = []
    used: set[str] = set()
    for i, q in enumerate(questions):
        wire_id = q.meta.get("wire_id")
        base = wire_id if isinstance(wire_id, str) and wire_id else (q.decision_point or f"q{i}")
        base = base.replace(DECISION_TOKEN, "decision")
        name, k = base, 2
        while name in used:
            name, k = f"{base}_{k}", k + 1
        used.add(name)
        names.append(name)
    return names


def compile_request(state: Any, questions: list[Question], names: list[str] | None = None,
                    raw_state: bool = False) -> tuple[CompiledExample, dict[str, dict[str, Any]]]:
    """Compile our questions about ``state`` into the reference prompt. Returns the compiled example and the
    reference-format question dicts by field (what ``read_fields`` needs)."""
    names = names or field_names(questions)
    qs = {n: question_to_intern(q) for n, q in zip(names, questions)}
    return compile_row({"state": state, "questions": qs}, raw_state=raw_state), qs


def answer_from_intern(q: Question, a: dict[str, Any], **raw_extra: Any) -> Answer:
    """Reference answer dict -> our ``Answer``. ``probabilities`` are keyed by our labels already (see
    ``question_to_intern``); ``raw`` keeps the exact calibrated probabilities and the reference fields."""
    raw = {k: v for k, v in a.items() if k != "legend"}
    raw.update(raw_extra)
    ans = answer_from_distribution(q, a["probabilities"], confidence=float(a["confidence"]), raw=raw)
    if q.kind == QKind.CHOICE:
        ans.value = a["choice"]  # reference tie-break (highest probability, then lexical), not dict order
    elif q.kind == QKind.SCORE:
        ans.value = float(a["score"])  # reference expected value (ours differs only by the 1e-6 floor)
    return ans


class InternDecisionBackend(DecisionBackend):
    name = "intern_decision"
    batched_questions = True

    def __init__(self, model: str = "internlm/Intern-Decision-0.8B", device: str = "auto", dtype: str | None = None,
                 max_length: int = 8192, temperature: float | None = None,
                 max_questions_per_request: int = MAX_FIELDS, max_options: int = MAX_SYMBOLS,
                 truncate_state: bool = True, state_json: bool = True, attn_implementation: str | None = None,
                 price_model: str | None = None, name: str | None = None, knockout_probs: str = DEFAULT_KNOCKOUT):
        import torch
        from transformers import AutoTokenizer

        from ..training.train_decision import load_causal_lm

        if name:
            self.name = name
        self.model = model
        self.price_model = price_model or model
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.max_length = int(max_length)
        self.max_q = max(1, min(int(max_questions_per_request), MAX_FIELDS))
        self.max_options = max(2, min(int(max_options), MAX_SYMBOLS))
        self.truncate_state = bool(truncate_state)
        self.state_json = bool(state_json)
        self.knockout_probs = check_method(knockout_probs)
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.dtype_name = dtype or ("bfloat16" if self.device.type == "cuda" else "float32")
        if self.dtype_name not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16 or float32")
        self.tok = AutoTokenizer.from_pretrained(model)
        if DECISION_TOKEN not in self.tok.get_added_vocab():
            raise ValueError(f"{model}: checkpoint must ship its trained decision tokenizer (no {DECISION_TOKEN} token)")
        self.marker_id = int(self.tok.convert_tokens_to_ids(DECISION_TOKEN))
        self.symbol_ids: dict[str, int] = {}
        for s in ANSWER_SYMBOLS:  # the reference checks per request; the vocabulary is fixed, so check once
            enc = self.tok.encode(s, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"{model}: answer symbol {s!r} is not a single token ({enc})")
            self.symbol_ids[s] = int(enc[0])
        extra = {"attn_implementation": attn_implementation} if attn_implementation else {}
        self._model, self.shared_weights = shared_weights(
            model, self.device, self.dtype_name, attn_implementation,
            lambda: load_causal_lm(model, getattr(torch, self.dtype_name), **extra).to(self.device).eval())
        if self.marker_id >= self._model.get_input_embeddings().weight.shape[0]:
            raise ValueError("Decision marker exceeds checkpoint vocabulary")
        if temperature is not None:
            self.temperature, self.temperature_source = validate_temperature(temperature), "spec"
        else:
            t, src = checkpoint_temperature(model)
            if t is None:
                print(f"[intern_decision] no calibration temperature known for {model!r}: using T=1 (uncalibrated)",
                      file=sys.stderr)
                t, src = 1.0, "uncalibrated default"
            self.temperature, self.temperature_source = t, src
        self._lock = device_lock(self.device)  # one forward pass per device at a time (worker threads, all instances)
        self.n_truncated = 0
        self.n_oom_retries = 0

    # --- one forward pass over <= max_q fields ------------------------------------------------------------------
    def _encode(self, state: str, questions: list[Question], names: list[str]) -> tuple[CompiledExample, dict, Any, Any, bool]:
        truncated = False
        for _ in range(12):
            compiled, qs = compile_request(state, questions, names, raw_state=not self.state_json)
            text = self.tok.apply_chat_template(compiled.messages, tokenize=False, add_generation_prompt=False,
                                                enable_thinking=False, add_vision_id=True)
            ids = self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
            length = int(ids.shape[-1])
            if length <= self.max_length:
                break
            if not self.truncate_state:
                raise ValueError(f"Example has {length} tokens, above {self.max_length}; truncation is forbidden")
            # drop the oldest part of the state (the harness state ends with the newest observation)
            state_ids = self.tok(state, add_special_tokens=False)["input_ids"]
            drop = length - self.max_length + 16
            if not state_ids or drop >= len(state_ids):
                raise ValueError(f"questions alone exceed max_length={self.max_length} tokens")
            state = self.tok.decode(state_ids[drop:])
            truncated = True
        else:
            raise ValueError(f"could not fit the request into max_length={self.max_length} tokens")
        positions = (ids[0] == self.marker_id).nonzero().flatten() - 1
        if len(positions) != len(compiled.fields) or bool((positions < 0).any()):
            raise ValueError("Decision marker count or position mismatch")
        if truncated:
            self.n_truncated += 1
        return compiled, qs, ids, positions, truncated

    def _forward(self, ids: Any, positions: Any) -> tuple[Any, float]:
        """Logits [n_fields, vocab] at ``positions`` and the forward-pass time in ms (the latency we book)."""
        import torch

        with self._lock, torch.inference_mode():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t = time.perf_counter()
            batch = ids.to(self.device)
            kw = {"input_ids": batch, "attention_mask": torch.ones_like(batch), "use_cache": False}

            def run() -> Any:
                try:
                    return self._model(**kw, logits_to_keep=positions.to(self.device)).logits[0]
                except TypeError:  # older transformers without logits_to_keep
                    return self._model(**kw).logits[0, positions.to(self.device)]

            try:
                out = run()
            except _oom_error():  # a long state after other instances' passes: release cached blocks, retry once
                self.n_oom_retries += 1
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                out = run()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            ms = (time.perf_counter() - t) * 1000
            return out.float().cpu(), ms

    def _score(self, state: str, questions: list[Question], names: list[str], usage: Usage) -> list[dict[str, Any]]:
        """Reference answer dicts for <= max_q questions in one forward pass; books tokens / latency on ``usage``."""
        compiled, qs, ids, positions, truncated = self._encode(state, questions, names)
        logits, ms = self._forward(ids, positions)
        out = read_fields(logits, compiled, qs, self.symbol_ids, self.temperature)
        usage.input_tokens += int(ids.shape[-1])
        usage.output_tokens += len(compiled.fields)
        usage.latency_ms += ms  # forward passes are sequential
        usage.n_calls += 1
        answers = []
        for n in compiled.fields:
            a = dict(out[n])
            a["field"] = n
            if truncated:
                a["truncated_state"] = True
            answers.append(a)
        return answers

    def _score_many(self, state: str, questions: list[Question], usage: Usage) -> list[dict[str, Any]]:
        names = field_names(questions)
        out: list[dict[str, Any]] = []
        for i in range(0, len(questions), self.max_q):
            out += self._score(state, questions[i:i + self.max_q], names[i:i + self.max_q], usage)
        return out

    # --- Choice beyond max_options: knockout as training/serve_jev.py (chunks, top-2 advance), full-set composition --
    def _knockout(self, state: str, q: Question, usage: Usage) -> tuple[Answer, int]:
        """(answer, elimination rounds). Every round's chunks are read in one multi-field prompt (one forward pass per
        <= ``max_questions_per_request`` chunks); the reads are composed into a distribution over every option with
        ``knockout_probs`` (decision/tournament.py: ``linked`` by default, ``finalists`` = the pre-2026-10-02 output:
        the final read, zero elsewhere). ``raw["knockout"]`` records rounds, reads, forward passes and finalists."""
        from .types import Choice

        options = list(q.options or [])
        descs = q.option_descriptions
        m = self.max_options
        calls0 = usage.n_calls

        def sub(idx: list[int]) -> Question:
            return Choice(q.text, [options[k] for k in idx], [descs[k] for k in idx] if descs else None,
                          decision_point=q.decision_point)

        pool = list(range(len(options)))
        rounds: list[list[tuple[list[str], dict[str, float], list[str]]]] = []
        reads = 0
        while len(pool) > m:
            chunks = [pool[j:j + m] for j in range(0, len(pool), m)]
            scored = [idx for idx in chunks if len(idx) > 1]
            outs = iter(self._score_many(state, [sub(idx) for idx in scored], usage))  # chunks share one prefill
            reads += len(scored)
            nxt: list[int] = []
            rnd: list[tuple[list[str], dict[str, float], list[str]]] = []
            for idx in chunks:
                if len(idx) == 1:
                    nxt += idx
                    continue
                probs = next(outs)["probabilities"]
                p = [probs[options[k]] for k in idx]
                surv = [idx[k] for k in sorted(range(len(idx)), key=lambda k: -p[k])[:2]]
                nxt += surv
                rnd.append(([options[k] for k in idx], {options[k]: float(probs[options[k]]) for k in idx},
                            [options[k] for k in surv]))
            rounds.append(rnd)
            pool = nxt
        final = self._score_many(state, [sub(pool)], usage)[0]
        fin = {options[k]: float(final["probabilities"][options[k]]) for k in pool}
        full = full_set_distribution(options, rounds, fin, self.knockout_probs)
        info = {"rounds": len(rounds) + 1, "reads": reads + 1, "passes": usage.n_calls - calls0,
                "finalists": [options[k] for k in pool], "full_set": self.knockout_probs}
        a = {**final, "probabilities": full, "confidence": full[final["choice"]], "knockout": info,
             "finalists": info["finalists"], "finalist_probabilities": fin}
        return answer_from_intern(q, a, calibration={"method": "temperature-scaling", "temperature": self.temperature,
                                                     "applied_to": "each read"}), len(rounds)

    def _decide(self, state: str, questions: list[Question]) -> DecisionResult:
        usage = Usage(backend=self.name, model=self.model, n_calls=0)
        answers: list[Answer | None] = [None] * len(questions)
        pending: list[tuple[int, Question]] = []
        elim = 0
        for i, q in enumerate(questions):
            if q.kind == QKind.CHOICE and q.options and len(q.options) > self.max_options:
                answers[i], r = self._knockout(state, q, usage)
                elim = max(elim, r)
            else:
                pending.append((i, q))
        if pending:
            outs = self._score_many(state, [q for _, q in pending], usage)
            for (i, q), a in zip(pending, outs):
                answers[i] = answer_from_intern(q, a, calibration={"method": "temperature-scaling",
                                                                   "temperature": self.temperature})
        usage.cost_usd = default_prices().cost(self.price_model, usage.input_tokens, usage.output_tokens)
        usage.priced = default_prices().get(self.price_model) is not None
        usage.n_passes = usage.n_calls       # every call is one forward pass of the checkpoint
        usage.n_rounds = 1 + elim            # logical depth of the widest Choice (several knockouts run one after
                                             # another here: n_passes counts every pass on the critical path)
        return DecisionResult(answers=[a for a in answers if a is not None], usage=usage)

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model, "dtype": self.dtype_name, "device": str(self.device),
                "max_length": self.max_length, "temperature": self.temperature, "batch": False,
                "batched_questions": True, "max_questions_per_request": self.max_q, "max_options": self.max_options,
                "truncate_state": self.truncate_state, "state_json": self.state_json,
                "shared_weights": self.shared_weights,
                # only a non-default composition is part of the spec id (backend_id): existing ids stay valid
                **({"knockout_probs": self.knockout_probs} if self.knockout_probs != DEFAULT_KNOCKOUT else {})}


__all__ = ["ANSWER_SYMBOLS", "DECISION_TOKEN", "KNOWN_TEMPERATURES", "MAX_FIELDS", "SYSTEM_PROMPT", "CompiledExample",
           "device_lock", "release_shared_weights", "shared_weights",
           "InternDecisionBackend", "answer_from_intern", "argmax", "checkpoint_temperature", "compile_request",
           "compile_row", "field_names", "question_to_intern", "read_fields", "scale_probabilities",
           "validate_temperature"]
