"""A causal LM used as a single-pass typed decider (the open, trainable counterpart of Jev).

Readout: render state + question + coded options (A, B, C, ...), run ONE forward pass, take the
logits of the next token restricted to the option-code tokens -> softmax / T. This is the
architecture of current open Jev-class models (JevK5, Imajev, decider-4b; see
docs/TRAINING_RECIPES.md) and it is what ``train_decision.py`` optimises.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from ..decision.types import OPTION_LABELS
from .data import SYSTEM, render  # noqa: F401  (render re-exported for callers)


def chat_text(tokenizer: Any, prompt: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    kw: dict[str, Any] = {}
    name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    # Answer immediately, no <think> block (Qwen3-style hybrid templates). Decided from the TEMPLATE, not the
    # path: a tokenizer re-loaded from a checkpoint dir (serve_jev / calibrate / grpo --init) has no "qwen3"
    # in its name_or_path but must render byte-identical prompts to training.
    if "qwen3" in name or "enable_thinking" in str(getattr(tokenizer, "chat_template", "") or ""):
        kw["enable_thinking"] = False
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    return f"{SYSTEM}\n\n{prompt}\n\nAnswer:"


def token_budget_batches(lengths: list[int], max_tokens: int, max_batch: int = 64,
                         order: list[int] | None = None) -> list[list[int]]:
    """Split ``order`` (default: all indices, longest first) into micro-batches whose padded size
    ``len(batch) * max(length in batch)`` stays <= ``max_tokens`` (a single over-long item gets a batch of its own).
    Items should arrive sorted by length (descending) so each batch pads little. Used without activation
    checkpointing: memory then grows with the padded token count, not with the number of rows."""
    idx = list(order) if order is not None else sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in idx:
        m = max(cur_max, lengths[i])
        if cur and (m * (len(cur) + 1) > max_tokens or len(cur) >= max_batch):
            out.append(cur)
            cur, m = [], lengths[i]
        cur.append(i)
        cur_max = m
    if cur:
        out.append(cur)
    return out


def code_token_ids(tokenizer: Any, n: int = len(OPTION_LABELS)) -> list[int]:
    """Token id of each option code. Raises if a code is not a single token."""
    ids = []
    for c in OPTION_LABELS[:n]:
        enc = tokenizer.encode(c, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"option code {c!r} is not a single token for this tokenizer: {enc}")
        ids.append(enc[0])
    return ids


class DecisionCollator:
    """Left-pads prompts so the decision position is always the last one (index -1)."""

    def __init__(self, tokenizer: Any, max_len: int = 4096):
        self.tok = tokenizer
        self.max_len = max_len
        self.codes = code_token_ids(tokenizer)

    def encode(self, prompt: str) -> list[int]:
        ids = self.tok(chat_text(self.tok, prompt), add_special_tokens=False).input_ids
        return ids[-self.max_len:]  # keep the tail: question + options + generation prompt

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        seqs = [self.encode(b["prompt"]) for b in batch]
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        input_ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, L - len(s):] = torch.tensor(s)
            attn[i, L - len(s):] = 1
        K = max(len(b["target"]) for b in batch)
        code_ids = torch.full((len(batch), K), -1, dtype=torch.long)
        target = torch.zeros((len(batch), K))
        ref_logits = torch.zeros((len(batch), K))
        q = torch.full((len(batch), K), float("nan"))
        for i, b in enumerate(batch):
            k = len(b["target"])
            code_ids[i, :k] = torch.tensor(self.codes[:k])
            target[i, :k] = torch.tensor(b["target"])
            if b.get("ref_logits") is not None:
                ref_logits[i, :k] = torch.tensor(b["ref_logits"])
            if b.get("q") is not None:
                q[i, :k] = torch.tensor([float("nan") if v is None else v for v in b["q"]])
        return {"input_ids": input_ids, "attention_mask": attn, "code_ids": code_ids, "target": target,
                "ref_logits": ref_logits, "q": q}


def option_logits(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  code_ids: torch.Tensor) -> torch.Tensor:
    """Logits over option codes at the last position; padded options are -inf."""
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1)
    except TypeError:  # older transformers
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    last = out.logits[:, -1, :].float()
    idx = code_ids.clamp(min=0)
    z = torch.gather(last, 1, idx)
    return z.masked_fill(code_ids < 0, float("-inf"))


def decision_loss(z: torch.Tensor, batch: dict[str, torch.Tensor], loss: str = "soft_ce_brier",
                  brier_weight: float = 0.5, beta: float = 0.1, kl_weight: float = 0.02) -> torch.Tensor:
    """
    soft_ce         -sum_k q_k log p_k   (q = one-hot, teacher-soft, or MC target p*)
    soft_ce_brier   + brier_weight * sum_k (p_k - q_k)^2     (proper scoring rule; helps calibration)
    dpo             pairwise DPO between argmax-Q and argmin-Q options, reference = ref_logits
    pg              exact expected-reward policy gradient: -sum_k p_k Q_k + kl_weight * KL(p || p_ref)
    """
    mask = batch["code_ids"] >= 0
    logp = F.log_softmax(z, dim=-1).masked_fill(~mask, 0.0)
    p = logp.exp() * mask
    tgt = batch["target"]
    if loss in ("soft_ce", "soft_ce_brier"):
        per = -(tgt * logp).sum(-1)
        if loss == "soft_ce_brier":
            per = per + brier_weight * ((p - tgt) ** 2 * mask).sum(-1)
        return per.mean()
    ref = batch["ref_logits"].masked_fill(~mask, float("-inf"))
    ref_logp = F.log_softmax(ref, dim=-1).masked_fill(~mask, 0.0)
    q = batch["q"]
    has_q = ~torch.isnan(q) & mask
    if loss == "dpo":
        qf = q.masked_fill(~has_q, float("-inf"))
        w = qf.argmax(-1)
        lq = q.masked_fill(~has_q, float("inf"))
        lo = lq.argmin(-1)
        dz = (logp.gather(1, w[:, None]) - ref_logp.gather(1, w[:, None])) - \
             (logp.gather(1, lo[:, None]) - ref_logp.gather(1, lo[:, None]))
        valid = (w != lo).float()[:, None]
        return (-(F.logsigmoid(beta * dz)) * valid).sum() / valid.sum().clamp(min=1)
    if loss == "pg":
        qz = q.masked_fill(~has_q, 0.0)
        expected = (p * qz).sum(-1)
        kl = (p * (logp - ref_logp)).sum(-1)
        return (-expected + kl_weight * kl).mean()
    raise ValueError(loss)


# ------------------------------------------------------------------ reference policy (KL / DPO anchors)
@contextmanager
def use_adapter(model: Any, name: str) -> Iterator[None]:
    """Temporarily activate LoRA adapter ``name`` (frozen), then restore the trainable "default" adapter."""
    model.set_adapter(name, inference_mode=True)
    try:
        yield
    finally:
        model.set_adapter("default")


def reference_context(model: Any, ref_mode: str) -> Any:
    """``base``: LoRA disabled (the pre-trained model). ``init``: the frozen "ref" adapter, a copy of the
    ``--init`` adapter loaded with ``add_reference_adapter`` (the SFT policy the run started from)."""
    return model.disable_adapter() if ref_mode == "base" else use_adapter(model, "ref")


def add_reference_adapter(model: Any, init: str) -> None:
    model.load_adapter(init, adapter_name="ref", is_trainable=False)
    model.set_adapter("default")


# ------------------------------------------------------------------ incremental (prefix-cache) scoring
# One agent state is read many times: by k questions in one request (they share the whole ``STATE:`` block)
# and by the following steps of the episode (the state at t+1 is the state at t plus a delta). The helpers below
# score a prompt as ``prefix (cached) + suffix`` so the shared tokens are prefilled once. The prompt text itself is
# unchanged (``data.render`` + ``chat_text``, byte for byte); only computation is reused. Correctness relies on
# causality: a token's hidden states never depend on later tokens, so prefilling the prefix separately, in chunks,
# and right-padding the suffix batch are all exact (up to floating-point summation order).

def _dynamic_layer_types() -> tuple[type, ...]:
    """Cache layers whose ``update`` is out of place (``torch.cat``) and whose tensors are therefore safe to share
    between a snapshot and a working copy. Everything else (linear-attention / SSM layers update with ``copy_``)
    is cloned tensor by tensor."""
    from transformers import cache_utils as cu

    return tuple(t for t in (getattr(cu, "DynamicLayer", None), getattr(cu, "DynamicSlidingWindowLayer", None))
                 if t is not None)


def _map_tensors(obj: Any, fn: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _map_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_tensors(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_map_tensors(v, fn) for v in obj)
    return obj


def clone_cache(cache: Any, deep: bool = False) -> Any:
    """A private copy of a KV / recurrent cache: later ``forward`` calls on the copy leave the original untouched.
    Plain dynamic layers share their tensors unless ``deep`` (their update is out of place); other layers are
    cloned tensor by tensor. Container attributes (dicts of states / flags) are always copied."""
    import copy

    share = _dynamic_layer_types()
    new = copy.copy(cache)
    layers = []
    for layer in getattr(cache, "layers", []):
        nl = copy.copy(layer)
        keep = isinstance(layer, share) and not deep
        for k, v in vars(layer).items():
            if isinstance(v, torch.Tensor):
                setattr(nl, k, v if keep else v.clone())
            elif isinstance(v, (dict, list, tuple)):
                setattr(nl, k, _map_tensors(v, (lambda t: t) if keep else (lambda t: t.clone())))
        layers.append(nl)
    new.layers = layers
    return new


def expand_cache(cache: Any, n: int) -> Any:
    """Repeat a batch-1 cache ``n`` times along the batch dimension (a new cache; the input is not modified)."""
    if n == 1:
        return clone_cache(cache)
    new = clone_cache(cache)
    for layer in new.layers:
        for k, v in list(vars(layer).items()):
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.numel() > 0:
                setattr(layer, k, v.repeat_interleave(n, dim=0))
            elif isinstance(v, (dict, list, tuple)):
                setattr(layer, k, _map_tensors(
                    v, lambda t: t.repeat_interleave(n, dim=0) if t.dim() >= 1 and t.numel() > 0 else t))
    return new


def cache_length(cache: Any) -> int:
    return 0 if cache is None else int(cache.get_seq_length())


def crop_cache(cache: Any, length: int) -> Any:
    """Shorten ``cache`` (in place) to its first ``length`` tokens. Raises when the cache cannot be rolled back
    (linear-attention layers without past recording: Qwen3.5 / Qwen3-Next Gated DeltaNet)."""
    n = cache_length(cache) - length
    if n < 0:
        raise ValueError(f"cannot crop a {cache_length(cache)}-token cache to {length}")
    if n > 0:
        cache.crop(-n)  # negative = relative (transformers 4.x and 5.x); positive is the deprecated absolute form
        if cache_length(cache) != length:
            raise RuntimeError(f"crop left {cache_length(cache)} tokens, expected {length}")
    return cache


def cache_bytes(cache: Any) -> int:
    total = 0
    for layer in getattr(cache, "layers", []):
        for v in vars(layer).values():
            for t in _flat_tensors(v):
                total += t.numel() * t.element_size()
    return total


def _flat_tensors(obj: Any) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    _map_tensors(obj, lambda t: out.append(t))
    return out


@torch.no_grad()
def prefill(model: Any, ids: list[int], cache: Any = None, chunk: int = 4096) -> tuple[Any, torch.Tensor | None]:
    """Run ``ids`` through the model on top of ``cache`` (batch 1), in chunks of at most ``chunk`` tokens.
    Returns (cache, last-position logits or None when ``ids`` is empty). ``cache`` is modified in place."""
    device = next(model.parameters()).device
    last = None
    for a in range(0, len(ids), max(1, chunk)):
        x = torch.tensor([ids[a:a + chunk]], dtype=torch.long, device=device)
        kw: dict[str, Any] = {"input_ids": x, "use_cache": True}
        if cache is not None:
            kw["past_key_values"] = cache
        try:
            out = model(**kw, logits_to_keep=1)
        except TypeError:  # older transformers
            out = model(**kw)
        cache = out.past_key_values
        last = out.logits[:, -1, :].float()
    return cache, last


@torch.no_grad()
def option_logits_cached(model: Any, cache: Any, suffixes: list[list[int]], code_ids: torch.Tensor,
                         pad_id: int, gather: bool = False) -> torch.Tensor:
    """``option_logits`` for prompts ``prefix + suffix_i`` where the prefix is already in ``cache`` (batch 1).
    Suffixes are right-padded: real tokens keep the positions they have without padding and, being causal, never
    see the pads, so the result equals the stateless left-padded path for every architecture (full, sliding and
    linear attention). Each row's logits are read at its own last real token. ``cache`` is not modified.
    ``gather``: the LM head runs only at the rows' read-out positions (``logits_to_keep`` as an index tensor) instead
    of the whole window from the shortest row's read-out to the end -- same logits, far less head compute when the
    suffix lengths differ."""
    if not suffixes or any(len(s) == 0 for s in suffixes):
        raise ValueError("every suffix needs at least one token (the read-out position)")
    device = next(model.parameters()).device
    B, S = len(suffixes), max(len(s) for s in suffixes)
    P = cache_length(cache)
    lens = [len(s) for s in suffixes]
    input_ids = torch.full((B, S), pad_id, dtype=torch.long, device=device)
    attn = torch.ones((B, P + S), dtype=torch.long, device=device)
    for i, s in enumerate(suffixes):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        attn[i, P + len(s):] = 0
    kw: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attn, "use_cache": True}
    if cache is not None:
        kw["past_key_values"] = expand_cache(cache, B)
    if gather:
        uniq = sorted(set(L - 1 for L in lens))           # the distinct read-out positions in the suffix window
        try:
            out = model(**kw, logits_to_keep=torch.tensor(uniq, dtype=torch.long, device=device))
            if out.logits.shape[1] == len(uniq):
                col = torch.tensor([uniq.index(L - 1) for L in lens], device=device)
                last = out.logits[torch.arange(B, device=device), col].float()
                idx = code_ids.to(device).clamp(min=0)
                z = torch.gather(last, 1, idx)
                return z.masked_fill(code_ids.to(device) < 0, float("-inf"))
        except TypeError:  # a model without index-tensor ``logits_to_keep``: the window path below
            pass
        kw["past_key_values"] = expand_cache(cache, B) if cache is not None else None
        if kw["past_key_values"] is None:
            kw.pop("past_key_values")
    keep = S - min(lens) + 1  # the window of last positions that contains every row's read-out position
    try:
        out = model(**kw, logits_to_keep=keep)
        logits = out.logits
    except TypeError:
        out = model(**kw)
        logits = out.logits[:, -keep:, :]
    pos = torch.tensor([L - min(lens) for L in lens], device=device)  # row i reads index len_i - 1 of the suffix
    last = logits[torch.arange(B, device=device), pos].float()
    idx = code_ids.to(device).clamp(min=0)
    z = torch.gather(last, 1, idx)
    return z.masked_fill(code_ids.to(device) < 0, float("-inf"))


def common_prefix_len(seqs: list[list[int]]) -> int:
    """Length of the longest common token prefix of ``seqs``."""
    if not seqs:
        return 0
    n = min(len(s) for s in seqs)
    first = seqs[0]
    for i in range(n):
        t = first[i]
        for s in seqs[1:]:
            if s[i] != t:
                return i
    return n
