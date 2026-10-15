"""Serve a (fine-tuned) causal-LM decision model behind the Jev wire API.

    python -m jevharness.training.serve_jev --model Qwen/Qwen3-4B --adapter checkpoints/jev4b-sft \
        --calibration checkpoints/jev4b-sft/calibration.json --port 8081

Then use it anywhere a Jev backend is accepted, e.g. ``{"type": "jev", "name": "ours-4b",
"base_url": "http://localhost:8081", "model": "ours-4b", "api_key": ""}`` -- the harness, the
hook adapters and pi extensions all speak this protocol, so trained checkpoints are drop-in.

Endpoints: POST /v1/systemone, GET /v1/models, GET /health.
Choice with > 52 options uses a knockout (chunks of 52, top-2 advance, a final read over the finalists): several
forward passes in two or more rounds, composed into a distribution over every option with ``--knockout-probs``
(``linked``, default; ``finalists`` = the output before 2026-10-02: the final read, zero elsewhere) -- see
``decision/tournament.py``. Every response's ``usage`` reports ``n_passes`` (logical forward passes this request's
answers used: a prefill segment, a stateless batch or a batched suffix pass each count once, a chunked prefill of one
segment once) and ``n_rounds`` (1 + elimination rounds of its widest Choice); a knockout answer carries
``knockout = {rounds, reads, finalists, full_set}``.

Incremental serving (docs/TRAINING_RECIPES.md "Incremental decision serving"): the prompt text is the training
prompt byte for byte, but computation is reused -- the k questions of one request share one prefill of their
common token prefix (the whole ``STATE:`` block), and a session cache keeps prefix snapshots across requests so
the next step of an episode prefills only its delta. ``JEV_SERVE_NO_CACHE=1`` forces the stateless path.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

try:  # module-level so FastAPI can resolve the (string) annotation under `from __future__ import annotations`
    from fastapi import Request
except ImportError:  # pragma: no cover - fastapi is an optional extra
    Request = Any  # type: ignore[misc,assignment]

from ..decision.tournament import DEFAULT_KNOCKOUT, check_method, full_set_distribution
from ..decision.types import OPTION_LABELS
from .data import render

MAX = len(OPTION_LABELS)
NO_CACHE_ENV = "JEV_SERVE_NO_CACHE"
PROFILE_ENV = "JEV_SERVE_PROFILE"
STAT_KEYS = ("requests", "questions", "prefilled_tokens", "stateless_tokens", "cache_hits", "cache_hit_tokens",
             "snapshots_stored", "evictions", "passes")


def _bucket(k: int) -> str:
    return "2" if k == 2 else "3-5" if k <= 5 else "6-16" if k <= 16 else "17-52"


def resolve_checkpoint(model: str, adapter: str | None) -> tuple[str, str | None]:
    """(model, adapter) to load. ``train_decision --lora-r 0`` (full fine-tuning, e.g. the 0.6B student of
    ``80_distill_small.sh``) saves a complete model, not a PEFT adapter: passed as ``adapter`` it is loaded as the
    model itself, so ``MODEL=<base> ADAPTER=<checkpoint>`` works for LoRA and full checkpoints alike."""
    from pathlib import Path

    if adapter and not Path(adapter, "adapter_config.json").exists() and Path(adapter, "config.json").exists():
        return adapter, None
    return model, adapter


def _lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    eq = a[:n] == b[:n]
    return n if eq.all() else int(np.argmin(eq))


class SessionCache:
    """LRU of (token prefix, cache snapshot) pairs, bounded by total cached tokens and entry count.

    ``lookup`` returns the stored snapshot whose tokens are the longest usable prefix of the query: an entry that is
    an exact prefix always qualifies; when the cache can be cropped (dense / sliding-window KV), a longer entry
    qualifies too and is cropped to the match. Snapshots are private copies; callers receive copies as well."""

    def __init__(self, max_tokens: int = 65536, max_entries: int = 64, croppable: bool = False):
        self.max_tokens = int(max_tokens)
        self.max_entries = int(max_entries)
        self.croppable = croppable
        self._e: OrderedDict[int, tuple[np.ndarray, Any]] = OrderedDict()
        self._next = 0
        self.tokens = 0
        self.evictions = 0
        self.stored = 0

    def __len__(self) -> int:
        return len(self._e)

    def clear(self) -> None:
        self._e.clear()
        self.tokens = 0

    def lookup(self, tokens: list[int]) -> tuple[Any, int, int, np.ndarray | None]:
        """-> (cache copy or None, usable match length, longest raw match with any entry, that entry's tokens)."""
        from .decision_model import clone_cache, crop_cache

        q = np.asarray(tokens, dtype=np.int32)
        best_key, best_m, any_m, any_toks = None, 0, 0, None
        for key, (toks, _) in self._e.items():
            m = _lcp(toks, q)
            if m >= any_m and m > 0:
                any_m, any_toks = m, toks
            usable = m if (m == len(toks) or self.croppable) else 0
            if usable >= best_m and usable > 0:  # ties -> most recently used (later in the OrderedDict)
                best_key, best_m = key, usable
        if best_key is None:
            return None, 0, any_m, any_toks
        toks, cache = self._e[best_key]
        self._e.move_to_end(best_key)
        copy = clone_cache(cache)
        if len(toks) > best_m:
            try:
                crop_cache(copy, best_m)
            except Exception:  # the probe said croppable but this cache is not (e.g. a sliding window reached)
                self.croppable = False
                return None, 0, any_m, any_toks
        return copy, best_m, any_m, any_toks

    def store(self, tokens: list[int], cache: Any) -> bool:
        """Keep ``cache`` (a private copy; not modified afterwards) as the snapshot of ``tokens``."""
        n = len(tokens)
        if n == 0 or n > self.max_tokens or self.max_entries <= 0:
            return False
        q = np.asarray(tokens, dtype=np.int32)
        for key, (toks, _) in list(self._e.items()):
            m = _lcp(toks, q)
            if m == n and len(toks) >= n and (self.croppable or len(toks) == n):
                self._e.move_to_end(key)  # already covered (exactly, or by a longer croppable entry)
                return False
            if m == len(toks) and n > m and self.croppable:
                self._drop(key)  # the new entry extends this one; cropping recovers it
        while self._e and (self.tokens + n > self.max_tokens or len(self._e) >= self.max_entries):
            self._drop(next(iter(self._e)))
            self.evictions += 1
        self._e[self._next] = (q, cache)
        self._next += 1
        self.tokens += n
        self.stored += 1
        return True

    def _drop(self, key: int) -> None:
        toks, _ = self._e.pop(key)
        self.tokens -= len(toks)

    def describe(self) -> dict[str, Any]:
        return {"entries": len(self._e), "cached_tokens": self.tokens, "max_tokens": self.max_tokens,
                "max_entries": self.max_entries, "croppable": self.croppable, "evictions": self.evictions,
                "snapshots_stored": self.stored}


class Engine:
    def __init__(self, model: str, adapter: str | None = None, calibration: str | None = None,
                 name: str = "ours", device: str | None = None, max_len: int = 4096, batch_size: int = 16,
                 session_cache: bool = False, cache_tokens: int = 65536, cache_entries: int = 64,
                 prefill_chunk: int = 4096, batch_suffixes: bool = True, suffix_batch_tokens: int = 32768,
                 stateless: bool = False, profile: bool = False, knockout_probs: str = DEFAULT_KNOCKOUT,
                 fuse_multi: bool = False, suffix_pad_slack: float | None = None, readout_gather: bool = False,
                 preencode: bool = False):
        """``session_cache`` keeps prefix snapshots across requests (the server turns it on; in-process batch scorers
        such as ``teacher_label`` / ``eval_decisions`` leave it off). ``stateless`` (or ``JEV_SERVE_NO_CACHE=1``)
        disables every reuse: one full prefill per question, exactly the training collator's path.
        ``suffix_batch_tokens`` bounds prefix tokens x questions scored in one batched suffix pass (the prefix cache
        is repeated per question); longer prefixes score their questions one by one. ``profile`` (or
        ``JEV_SERVE_PROFILE=1``) accumulates wall time per phase of the shared-prefix path in ``phase_ms`` (encode,
        lookup, prefill, store, suffix; with a device sync around each, so only for benchmarks). ``knockout_probs``
        composes the reads of a > 52-option Choice (``linked`` | ``finalists``, decision/tournament.py).

        Exact fast paths of the final harness (docs/JEVAGENT_HARNESS.md §8; all off by default = the Stage-2 server):
        ``fuse_multi`` -- on a croppable cache, several questions on one state run the prefix delta together with the
        LONGEST question's suffix in one forward pass (as the single-question fused path does), crop the cache back to
        the prefix, and score the remaining suffixes on it; ``suffix_pad_slack`` -- the batched suffix passes group
        suffixes by length so that padded positions stay within (1 + slack) x the real ones (a 52-option knockout chunk
        and a 17-option one are no longer padded to the same length); ``readout_gather`` -- the language-model head is
        applied only at the read-out positions instead of every position of the last-token window; ``preencode`` -- a
        request's prompts are tokenized before the engine lock is taken, so tokenizing overlaps other requests' passes."""
        from transformers import AutoTokenizer

        from .decision_model import DecisionCollator
        from .train_decision import load_causal_lm

        model, adapter = resolve_checkpoint(model, adapter)
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(adapter or model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        m = load_causal_lm(model, dtype)
        if adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        self.model = m.to(self.device).eval()
        self.dtype = str(dtype).replace("torch.", "")
        self.collator = DecisionCollator(self.tok, max_len=max_len)
        self.batch_size = batch_size
        self.cal = json.load(open(calibration)) if calibration else {}
        self.prefill_chunk = max(1, int(prefill_chunk))
        self.batch_suffixes = batch_suffixes
        self.suffix_batch_tokens = max(1, int(suffix_batch_tokens))
        self.stateless = stateless
        self.knockout_probs = check_method(knockout_probs)
        self.fuse_multi = bool(fuse_multi)
        self.suffix_pad_slack = None if suffix_pad_slack is None else max(0.0, float(suffix_pad_slack))
        self.readout_gather = bool(readout_gather)
        self.preencode = bool(preencode)
        self._enc_memo: dict[str, list[int]] = {}
        self.lock = threading.RLock()
        self.stats: dict[str, int] = {k: 0 for k in STAT_KEYS}
        self.profile = bool(profile) or os.environ.get(PROFILE_ENV, "") not in ("", "0", "false", "no")
        self.phase_ms: dict[str, float] = {}
        self.cache_info = self._probe_cache()
        self.session: SessionCache | None = (SessionCache(cache_tokens, cache_entries, self.cache_info["croppable"])
                                             if session_cache else None)

    # ------------------------------------------------------------------ cache capability probe
    @torch.no_grad()
    def _probe_cache(self) -> dict[str, Any]:
        """Learn once what the model's cache supports: whether it can be cropped (rolled back to a shorter prefix,
        checked numerically against a fresh prefill) and how many bytes it holds per token."""
        from .decision_model import cache_bytes, cache_length, clone_cache, crop_cache, prefill

        ids = self.collator.encode("probe " * 4)
        info: dict[str, Any] = {"croppable": False, "cache_class": None, "layer_types": {}, "bytes_per_token": 0}
        try:
            cache, _ = prefill(self.model, ids, None, self.prefill_chunk)
        except Exception as e:  # pragma: no cover - a model without a usable cache falls back to stateless
            info["error"] = f"{type(e).__name__}: {e}"
            self.stateless = True
            return info
        info["cache_class"] = type(cache).__name__
        types: dict[str, int] = {}
        for layer in getattr(cache, "layers", []):
            types[type(layer).__name__] = types.get(type(layer).__name__, 0) + 1
        info["layer_types"] = types
        info["bytes_per_token"] = cache_bytes(cache) // max(1, len(ids))
        k = min(3, max(1, len(ids) - 1))
        try:
            short, z_ref = prefill(self.model, ids[:-k], None, self.prefill_chunk)
            cropped = crop_cache(clone_cache(cache), len(ids) - k)
            ok = cache_length(cropped) == len(ids) - k and cache_length(cache) == len(ids)
            _, z_crop = prefill(self.model, ids[-k:], cropped, self.prefill_chunk)
            _, z_full = prefill(self.model, ids[-k:], short, self.prefill_chunk)
            if ok and self.dtype == "float32":
                ok = bool(torch.allclose(z_crop, z_full, atol=1e-3, rtol=1e-3))
            info["croppable"] = bool(ok)
        except Exception as e:
            info["croppable"] = False
            info["crop_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return info

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "device": self.device, "dtype": self.dtype,
                             "stateless": self._is_stateless(), "session_cache": self.session is not None,
                             "prefill_chunk": self.prefill_chunk, "batch_suffixes": self.batch_suffixes,
                             "suffix_batch_tokens": self.suffix_batch_tokens, "batch_size": self.batch_size,
                             "knockout_probs": self.knockout_probs, "fuse_multi": self.fuse_multi,
                             "suffix_pad_slack": self.suffix_pad_slack, "readout_gather": self.readout_gather,
                             "preencode": self.preencode,
                             **self.cache_info, "counters": dict(self.stats)}
        if self.session is not None:
            d["session"] = self.session.describe()
        if self.profile:
            d["phase_ms"] = dict(self.phase_ms)
        return d

    class _Phase:
        """``with engine._phase("prefill"):`` adds the block's wall time (device-synchronised) to ``phase_ms``."""

        def __init__(self, engine: "Engine", name: str):
            self.engine, self.name = engine, name

        def __enter__(self) -> None:
            if self.engine.device == "cuda":
                torch.cuda.synchronize()
            self.t0 = time.perf_counter()

        def __exit__(self, *exc: Any) -> None:
            if self.engine.device == "cuda":
                torch.cuda.synchronize()
            self.engine.phase_ms[self.name] = self.engine.phase_ms.get(self.name, 0.0) + (time.perf_counter() - self.t0) * 1000

    class _NoPhase:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: Any) -> None:
            return None

    def _phase(self, name: str) -> Any:
        return self._Phase(self, name) if self.profile else self._NoPhase()

    def temperature(self, kind: str, k: int) -> float:
        b = self.cal.get("buckets", {})
        return float(b.get(f"{kind}:{_bucket(k)}", b.get(_bucket(k), self.cal.get("global", 1.0))))

    def _is_stateless(self) -> bool:
        return self.stateless or os.environ.get(NO_CACHE_ENV, "") not in ("", "0", "false", "no")

    def _new_pass(self, items: list[int]) -> int:
        """Book one logical forward pass that the given items' read-outs depend on (review P1-4 accounting)."""
        self._pass_seq = getattr(self, "_pass_seq", 0) + 1
        self.stats["passes"] = self.stats.get("passes", 0) + 1
        for i in items:
            self._item_stats[i]["passes"].add(self._pass_seq)
        return self._pass_seq

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def _probs(self, items: list[tuple[str, str, list[str], str]]) -> list[list[float]]:
        """items: (state, question, options, kind) -> probability lists (<= 52 options each)."""
        z = self._option_logits(items)
        out: list[list[float]] = []
        for (_, _, o, kind), zi in zip(items, z):
            t = self.temperature(kind, len(o))
            out.append(torch.softmax(torch.tensor(zi[: len(o)]) / t, -1).tolist())
        return out

    @torch.no_grad()
    def _option_logits(self, items: list[tuple[str, str, list[str], str]]) -> list[list[float]]:
        """Raw option-code logits per item (``option_logits`` numbers), via the stateless or the shared-prefix
        path. Fills ``self._item_stats`` (per item: prefilled / stateless tokens, cache-hit tokens) for the usage
        report."""
        with self.lock:
            self.stats["questions"] += len(items)
            self._item_stats = [{"prefilled": 0, "stateless": 0, "hit": 0, "passes": set()} for _ in items]
            if self._is_stateless():
                return self._logits_stateless(list(range(len(items))), items)
            out: list[list[float] | None] = [None] * len(items)
            groups: dict[str, list[int]] = {}
            for i, it in enumerate(items):
                groups.setdefault(it[0], []).append(i)
            singles = [g[0] for g in groups.values() if len(g) == 1]
            if singles and self.session is None:  # distinct states, nothing to share: one padded batch
                for i, z in zip(singles, self._logits_stateless(singles, items)):
                    out[i] = z
            for idxs in groups.values():
                if out[idxs[0]] is not None:
                    continue
                for i, z in zip(idxs, self._logits_shared(idxs, items)):
                    out[i] = z
            return out  # type: ignore[return-value]

    def _logits_stateless(self, idxs: list[int], items: list[tuple[str, str, list[str], str]]) -> list[list[float]]:
        from .decision_model import option_logits

        out: list[list[float]] = []
        for a in range(0, len(idxs), self.batch_size):
            chunk = [items[i] for i in idxs[a:a + self.batch_size]]
            batch = [{"prompt": render({"state": s, "question": q, "options": o})[0], "target": [0.0] * len(o)}
                     for s, q, o, _ in chunk]
            enc = self.collator(batch)
            n = enc["attention_mask"].sum(1).tolist()
            for i, t in zip(idxs[a:a + self.batch_size], n):
                self._item_stats[i].update(prefilled=int(t), stateless=int(t))
                self.stats["prefilled_tokens"] += int(t)
                self.stats["stateless_tokens"] += int(t)
            z = option_logits(self.model, enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device),
                              enc["code_ids"].to(self.device)).float().cpu()
            self._new_pass(idxs[a:a + self.batch_size])
            out += z.tolist()
        return out

    def _logits_shared(self, idxs: list[int], items: list[tuple[str, str, list[str], str]]) -> list[list[float]]:
        """All ``idxs`` share one state: prefill the common token prefix once (from the session cache when it holds a
        snapshot of a prefix of it), then score every question's suffix on a copy of that cache."""
        from .decision_model import clone_cache, common_prefix_len, crop_cache, option_logits_cached, prefill

        with self._phase("encode"):
            seqs = [self._encode(render({"state": s, "question": q, "options": o})[0])
                    for s, q, o, _ in (items[i] for i in idxs)]
        L = min(common_prefix_len(seqs), min(len(s) for s in seqs) - 1)  # every suffix keeps >= 1 token
        C = seqs[0][:L]
        cache, m, any_m, any_toks = (None, 0, 0, None)
        if self.session is not None and L > 0:
            with self._phase("lookup"):
                cache, m, any_m, any_toks = self.session.lookup(C)
        # split points of the prefill: where this state diverged from the most similar stored snapshot (so the next
        # step, which shares that part, gets an exact-prefix hit even when the cache cannot be cropped), and where the
        # same divergence marker (e.g. "\n\nCURRENT OBSERVATION:") appears again later in this state
        splits: set[int] = set()
        if self.session is not None and any_toks is not None and not self.session.croppable:
            # a croppable cache needs no intermediate snapshots: the next step crops the full-prefix entry to its
            # match, so the prefix is prefilled in one segment (one forward pass with the suffix, see below)
            if m < any_m < L:
                splits.add(any_m)
            p = self._predict_divergence(C, any_toks, any_m)
            if p is not None and m < p < L:
                splits.add(p)
        # one question on a croppable cache (dense / sliding-window KV): the last prefill segment and the suffix go
        # through ONE forward pass (the read-out is the last token), then the cache is cropped back to the prefix
        # for the snapshot. Two passes (prefix, then suffix on a copy) are only needed when several questions share
        # the prefix or the cache cannot be rolled back (recurrent / linear-attention state).
        croppable = self.session is not None and self.session.croppable and self.batch_suffixes
        fused = len(idxs) == 1 and croppable and len(seqs[0]) > L
        # fuse_multi: several questions -- the prefix delta and the LONGEST suffix in one pass (positions ``fj``)
        fj = (max(range(len(idxs)), key=lambda j: len(seqs[j])) if (self.fuse_multi and croppable and len(idxs) > 1
                                                                     and m < L) else None)
        if fused:
            fj = 0
        pos = m
        fused_logits = None
        for stop in sorted(splits | {L}):
            if stop <= pos:
                continue
            if fj is not None and stop == L:
                with self._phase("prefill"):
                    cache, last = prefill(self.model, C[pos:stop] + seqs[fj][L:], cache, self.prefill_chunk)
                with self._phase("store"):
                    crop_cache(cache, L)
                fused_logits = last
            else:
                with self._phase("prefill"):
                    cache, _ = prefill(self.model, C[pos:stop], cache, self.prefill_chunk)
            self._new_pass(idxs)  # every question of the group reads off this prefix
            pos = stop
            if self.session is not None and (stop in splits or stop == L):
                with self._phase("store"):
                    if self.session.store(C[:stop], clone_cache(cache)):
                        self.stats["snapshots_stored"] += 1
        self.stats["evictions"] = self.session.evictions if self.session is not None else 0
        if m > 0:
            self.stats["cache_hits"] += 1
        self.stats["cache_hit_tokens"] += m
        self.stats["prefilled_tokens"] += L - m
        self._item_stats[idxs[0]].update(prefilled=L - m, hit=m)
        for i, s in zip(idxs, seqs):
            self._item_stats[i]["prefilled"] += len(s) - L
            self._item_stats[i]["stateless"] = len(s)
            self.stats["prefilled_tokens"] += len(s) - L
            self.stats["stateless_tokens"] += len(s)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        K = max(len(items[i][2]) for i in idxs)
        out: list[list[float]] = []
        rows: dict[int, list[float]] = {}
        if fused_logits is not None:  # the fused pass already read the option logits at the suffix's last token
            with self._phase("suffix"):
                k = len(items[idxs[fj]][2])
                code_ids = torch.tensor(self.collator.codes[:k], dtype=torch.long, device=fused_logits.device)
                z = torch.gather(fused_logits[0], 0, code_ids.clamp(min=0))
                z = z.masked_fill(code_ids < 0, float("-inf"))
                rows[fj] = z.float().cpu().tolist() + [float("-inf")] * (K - k)
            if len(idxs) == 1:
                return [rows[0]]
        step = min(self.batch_size, max(1, self.suffix_batch_tokens // max(1, L))) if self.batch_suffixes else 1
        rest = [j for j in range(len(idxs)) if j not in rows]
        for grp in self._suffix_groups(rest, [len(s) - L for s in seqs], step):
            sub = [(idxs[j], seqs[j]) for j in grp]
            code_ids = torch.full((len(sub), K), -1, dtype=torch.long)
            for r, (i, _) in enumerate(sub):
                k = len(items[i][2])
                code_ids[r, :k] = torch.tensor(self.collator.codes[:k])
            with self._phase("suffix"):
                z = option_logits_cached(self.model, cache, [s[L:] for _, s in sub], code_ids, pad,
                                         gather=self.readout_gather)
                for j, zr in zip(grp, z.float().cpu().tolist()):
                    rows[j] = zr
            self._new_pass([i for i, _ in sub])
        out += [rows[j] for j in range(len(idxs))]
        return out

    def _suffix_groups(self, js: list[int], lens: list[int], step: int) -> list[list[int]]:
        """Batches of suffix indices for the cached suffix passes: in order, ``step`` at a time (the Stage-2 server),
        or -- with ``suffix_pad_slack`` -- longest first, a batch closed when one more suffix would make the padded
        positions exceed (1 + slack) x the real ones."""
        if self.suffix_pad_slack is None:
            return [js[a:a + step] for a in range(0, len(js), step)]
        groups: list[list[int]] = []
        cur: list[int] = []
        for j in sorted(js, key=lambda j: -lens[j]):
            if cur:
                n, mx = len(cur) + 1, lens[cur[0]]
                real = sum(lens[x] for x in cur) + lens[j]
                if n > step or n * mx > (1.0 + self.suffix_pad_slack) * real:
                    groups.append(cur)
                    cur = []
            cur.append(j)
        if cur:
            groups.append(cur)
        return groups

    def _encode(self, prompt: str) -> list[int]:
        """``collator.encode`` (chat template + tokenizer), served from the pre-encoded prompts of the request when
        ``preencode`` tokenized them before the engine lock was taken (same ids; the work only moves off the lock)."""
        ids = self._enc_memo.pop(prompt, None) if self.preencode else None
        return ids if ids is not None else self.collator.encode(prompt)

    def _preencode(self, requests: list[tuple[str, dict[str, Any]]], plans: list[dict[str, Any]]) -> None:
        """Tokenize every prompt the first round of these requests will read, outside the engine lock (other
        requests' forward passes run meanwhile): direct questions and the first knockout round's chunks."""
        if len(self._enc_memo) > 1024:   # leftovers of failed requests never accumulate
            self._enc_memo.clear()
        for (state, _), ps in zip(requests, plans):
            for kind, instr, opts, _ in ps.values():
                chunks = [opts] if len(opts) <= MAX else [opts[a:a + MAX] for a in range(0, len(opts), MAX)]
                for o in chunks:
                    if len(o) > 1 or len(opts) <= MAX:
                        prompt = render({"state": state, "question": instr, "options": o})[0]
                        self._enc_memo[prompt] = self.collator.encode(prompt)

    @staticmethod
    def _predict_divergence(C: list[int], old: np.ndarray, m: int, width: int = 8) -> int | None:
        """Where will the *next* state diverge from this one? The previous step's tail started, right after the
        divergence point ``m``, with a marker such as ``\\n\\nCURRENT OBSERVATION:\\n`` that reappears before
        this state's own volatile tail: the last occurrence of the longest such marker (>= 2 tokens) in ``C``
        after ``m``."""
        tail = [int(t) for t in old[m:m + width]]
        arr = np.asarray(C, dtype=np.int32)
        n = len(arr)
        for w in range(len(tail), 1, -1):
            if n - w <= m:
                continue
            hit = np.ones(n - w + 1, dtype=bool)
            for i, t in enumerate(tail[:w]):
                hit &= arr[i:n - w + 1 + i] == t
            js = np.flatnonzero(hit[m + 1:])
            if len(js):
                return int(js[-1]) + m + 1
        return None

    # ------------------------------------------------------------------ questions
    def choice_probs(self, state: str, question: str, options: list[str], kind: str) -> list[float]:
        if len(options) <= MAX:
            return self._probs([(state, question, options, kind)])[0]
        return self._knockout([(state, question, options, kind)])[0]

    def _knockout(self, specs: list[tuple[str, str, list[str], str]],
                  info: list[dict[str, Any]] | None = None) -> list[list[float]]:
        """Choices with > 52 options: chunks of 52, top-2 advance (a one-option chunk advances as is); every round
        scores the chunks of every spec in one ``_probs`` call. Same pool order as the single-question path. The reads
        are composed into one distribution over every option with ``knockout_probs`` (decision/tournament.py);
        ``info`` (optional) receives per spec ``{"rounds", "reads", "finalists" (option indices), "full_set"}``."""
        pools = [list(range(len(o))) for _, _, o, _ in specs]
        acc: list[dict[str, Any]] = [{"prefilled": 0, "stateless": 0, "hit": 0, "passes": set()} for _ in specs]
        rounds: list[list[list[tuple[list[int], dict[int, float], list[int]]]]] = [[] for _ in specs]
        reads = [0 for _ in specs]

        def tally(owners: list[int]) -> None:
            for j, st in zip(owners, self._item_stats):
                for k in st:
                    if k == "passes":
                        acc[j]["passes"] |= st["passes"]
                    else:
                        acc[j][k] += st[k]

        while True:
            todo = [j for j, p in enumerate(pools) if len(p) > MAX]
            if not todo:
                break
            calls = [(j, pools[j][a:a + MAX]) for j in todo for a in range(0, len(pools[j]), MAX)
                     if len(pools[j][a:a + MAX]) > 1]
            probs = iter(self._probs([(specs[j][0], specs[j][1], [specs[j][2][k] for k in idx], specs[j][3])
                                      for j, idx in calls]))
            tally([j for j, _ in calls])
            for j in todo:
                nxt: list[int] = []
                rnd: list[tuple[list[int], dict[int, float], list[int]]] = []
                for a in range(0, len(pools[j]), MAX):
                    idx = pools[j][a:a + MAX]
                    if len(idx) == 1:
                        nxt += idx
                        continue
                    p = next(probs)
                    surv = [idx[k] for k in sorted(range(len(idx)), key=lambda k: -p[k])[:2]]
                    nxt += surv
                    rnd.append((idx, {k: float(v) for k, v in zip(idx, p)}, surv))
                    reads[j] += 1
                rounds[j].append(rnd)
                pools[j] = nxt
        probs_f = self._probs([(s, q, [o[k] for k in pool], kind) for (s, q, o, kind), pool in zip(specs, pools)])
        tally(list(range(len(specs))))
        self._item_stats = acc  # one record per spec, summed over all rounds
        method = getattr(self, "knockout_probs", DEFAULT_KNOCKOUT)
        out = []
        for j, ((_, _, o, _), pool, p) in enumerate(zip(specs, pools, probs_f)):
            full = full_set_distribution(list(range(len(o))), rounds[j], dict(zip(pool, p)), method)
            out.append([full[k] for k in range(len(o))])
            if info is not None:
                info.append({"rounds": len(rounds[j]) + 1, "reads": reads[j] + 1, "finalists": list(pool),
                             "full_set": method})
        return out

    @staticmethod
    def _plan(q: dict[str, Any]) -> tuple[str, str, list[str], Any]:
        """A wire question -> (kind, instruction, option strings, labels/levels for the answer)."""
        t = q.get("type")
        instr = q.get("instructions") or ""
        if not isinstance(instr, str):
            instr = json.dumps(instr)
        crit = q.get("criteria")
        if t == "choice":
            labels = list(crit.keys())
            opts = [lab if not crit[lab] else f"{lab} -- {crit[lab] if isinstance(crit[lab], str) else json.dumps(crit[lab])}"
                    for lab in labels]
            return "choice", instr, opts, labels
        if t == "noul":
            crit = crit or {}
            opts = ["yes" + (f" -- {crit['true']}" if crit.get("true") else ""),
                    "no" + (f" -- {crit['false']}" if crit.get("false") else "")]
            return "noul", instr, opts, None
        if t == "score":
            levels = [c if isinstance(c, str) else json.dumps(c) for c in crit]
            return "score", instr, [f"level {k}: {d}" for k, d in enumerate(levels)], levels
        raise ValueError(f"unsupported question type {t!r}")

    @staticmethod
    def _finish(kind: str, p: list[float], meta: Any) -> dict[str, Any]:
        if kind == "choice":
            best = max(range(len(meta)), key=lambda k: p[k])
            return {"type": "choice", "choice": meta[best], "confidence": p[best],
                    "probabilities": dict(zip(meta, p))}
        if kind == "noul":
            return {"type": "noul", "noul": p[0]}
        best = max(range(len(p)), key=lambda k: p[k])
        return {"type": "score", "score": sum(k * v for k, v in enumerate(p)), "confidence": p[best],
                "legend": {str(k): d for k, d in enumerate(meta)},
                "probabilities": {str(k): v for k, v in enumerate(p)}}

    def answer(self, state: str, q: dict[str, Any]) -> dict[str, Any]:
        """One wire question (``answer_all`` is the multi-question path). Only ``choice_probs`` / ``_probs`` are
        called on ``self`` so a stand-in with just those two methods can mirror the rendering (tests)."""
        kind, instr, opts, meta = Engine._plan(q)
        if kind == "choice":
            p = self.choice_probs(state, instr, opts, "choice")
        else:
            p = self._probs([(state, instr, opts, kind)])[0]
        return Engine._finish(kind, p, meta)

    def answer_batch(self, requests: list[tuple[str, dict[str, Any]]]
                     ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Several wire requests ``(state, {qid: question})`` at once -> per request (answers, usage extras).
        All questions of all requests that fit one pass are scored in one ``_probs`` call: questions on the same
        state share its prefix, and distinct states are batched when nothing is shared."""
        plans = [{qid: self._plan(q) for qid, q in qs.items()} for _, qs in requests]
        if getattr(self, "preencode", False) and not self._is_stateless():
            self._preencode(requests, plans)
        t0 = time.perf_counter()
        with self.lock:
            t1 = time.perf_counter()
            out = self._answer_planned(requests, plans)
            t2 = time.perf_counter()
        for _, u in out:  # serving bookkeeping: waiting for the engine (other requests) vs computing this batch
            u["lock_wait_ms"] = round((t1 - t0) * 1000, 3)
            u["compute_ms"] = round((t2 - t1) * 1000, 3)
        return out

    def _answer_planned(self, requests: list[tuple[str, dict[str, Any]]], plans: list[dict[str, Any]]
                        ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        items: list[tuple[str, str, list[str], str]] = []
        where: list[tuple[int, str]] = []
        big: list[tuple[int, str]] = []
        for r, ((state, _), ps) in enumerate(zip(requests, plans)):
            for qid, (kind, instr, opts, _) in ps.items():
                if len(opts) > MAX:
                    big.append((r, qid))
                else:
                    items.append((state, instr, opts, kind))
                    where.append((r, qid))
        usage: list[dict[str, Any]] = [{"prefilled_tokens": 0, "stateless_tokens": 0, "cache_hit_tokens": 0,
                                        "questions": len(ps), "_passes": set(), "n_rounds": 1} for ps in plans]
        probs: dict[tuple[int, str], list[float]] = {}
        ko: dict[tuple[int, str], dict[str, Any]] = {}
        if items:
            for (r, qid), p in zip(where, self._probs(items)):
                probs[(r, qid)] = p
            self._account(usage, [r for r, _ in where])
        if big:
            specs = [(requests[r][0], plans[r][qid][1], plans[r][qid][2], plans[r][qid][0]) for r, qid in big]
            info: list[dict[str, Any]] = []
            for (r, qid), p in zip(big, self._knockout(specs, info)):
                probs[(r, qid)] = p
            for (r, qid), inf in zip(big, info):
                ko[(r, qid)] = inf
                usage[r]["n_rounds"] = max(usage[r]["n_rounds"], int(inf["rounds"]))
            self._account(usage, [r for r, _ in big])
        out = []
        for r, ps in enumerate(plans):
            answers = {qid: self._finish(kind, probs[(r, qid)], meta) for qid, (kind, _, _, meta) in ps.items()}
            for qid, (kind, _, _, meta) in ps.items():
                inf = ko.get((r, qid))
                if inf and kind == "choice":
                    answers[qid]["knockout"] = {**inf, "finalists": [meta[k] for k in inf["finalists"]]}
            u = usage[r]
            u["cache_hit"] = u["cache_hit_tokens"] > 0
            u["n_passes"] = len(u.pop("_passes"))
            out.append((answers, u))
        self.stats["requests"] += len(requests)
        return out

    def _account(self, usage: list[dict[str, Any]], owners: list[int]) -> None:
        """Add the per-item statistics of the last ``_probs`` / ``_knockout`` call to their requests' usage."""
        for r, st in zip(owners, getattr(self, "_item_stats", [])):
            usage[r]["prefilled_tokens"] += st["prefilled"]
            usage[r]["stateless_tokens"] += st["stateless"]
            usage[r]["cache_hit_tokens"] += st["hit"]
            usage[r]["_passes"] |= st.get("passes", set())

    def answer_all(self, state: str, questions: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """One wire request: all its questions in one pass -> (answers by qid, usage extras)."""
        return self.answer_batch([(state, questions)])[0]


def build_app(engine: Engine):  # noqa: ANN201
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="jevharness decision server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **engine.stats}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"data": [{"id": engine.name, "object": "model", "backend": engine.describe()}]}

    @app.post("/v1/systemone")
    async def systemone(req: Request) -> JSONResponse:
        body = await req.json()
        state = body.get("state")
        if state is None or not body.get("questions"):
            raise HTTPException(422, "state and questions are required")
        state_s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        t = time.perf_counter()
        try:  # off the event loop: the engine serialises forwards with its own lock
            answers, extra = await run_in_threadpool(engine.answer_all, state_s, body["questions"])
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        n_in = len(engine.tok(state_s, add_special_tokens=False).input_ids)
        ms = (time.perf_counter() - t) * 1000
        return JSONResponse({"model": engine.name, "answers": answers,
                             "usage": {"input_tokens": n_in, "output_tokens": 0, **extra}},
                            headers={"server-timing": f"model;dur={ms:.1f}"})

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--name", default="ours")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--session-cache", action=argparse.BooleanOptionalAction, default=True,
                    help="keep prefix snapshots across requests (episode cache)")
    ap.add_argument("--cache-tokens", type=int, default=65536, help="session cache budget in cached tokens")
    ap.add_argument("--cache-entries", type=int, default=64)
    ap.add_argument("--prefill-chunk", type=int, default=4096)
    ap.add_argument("--batch-suffixes", action=argparse.BooleanOptionalAction, default=True,
                    help="score the questions of one request in one batched pass over the shared prefix")
    ap.add_argument("--suffix-batch-tokens", type=int, default=32768,
                    help="max prefix tokens x questions per batched suffix pass (the prefix cache is repeated)")
    ap.add_argument("--knockout-probs", default=DEFAULT_KNOCKOUT, choices=["linked", "finalists"],
                    help="full-set distribution of a > 52-option Choice (decision/tournament.py)")
    ap.add_argument("--fuse-multi", action=argparse.BooleanOptionalAction, default=False,
                    help="several questions: the prefix delta and the longest suffix in one pass (croppable caches)")
    ap.add_argument("--suffix-pad-slack", type=float, default=None,
                    help="group batched suffixes by length (padded <= (1 + slack) x real positions); default: in order")
    ap.add_argument("--readout-gather", action=argparse.BooleanOptionalAction, default=False,
                    help="apply the LM head only at the read-out positions")
    ap.add_argument("--preencode", action=argparse.BooleanOptionalAction, default=False,
                    help="tokenize a request's prompts before taking the engine lock")
    ap.add_argument("--fast", action="store_true",
                    help="the final harness's server: --fuse-multi --suffix-pad-slack 0.25 --readout-gather --preencode")
    ap.add_argument("--backend", default="lm", choices=["lm", "clef"],
                    help="lm: a causal LM read out at option codes (default); clef: a Clef scoring-head checkpoint "
                         "(--model = its saved folder; training/clef_engine.py, exploratory row E14)")
    ap.add_argument("--clef-max-len", type=int, default=16384, help="Clef: tokens per encoded request (no cut below)")
    ap.add_argument("--clef-coalesce", action="store_true",
                    help="Clef engine v2: requests queued behind a running pass are scored together in the next one")
    ap.add_argument("--clef-warmup", type=int, default=0, help="Clef engine v2: warm-up rounds before serving")
    ap.add_argument("--clef-defer", action="store_true",
                    help="E18-F: score a __defer__ option on every choice question; confidence = 1 - P(defer)")
    ap.add_argument("--clef-defer-head", default=None,
                    help="E18: a logistic defer head (JSON with 'weights') on the served top probability; "
                         "confidence = 1 - P(defer)")
    a = ap.parse_args()
    import uvicorn

    if a.backend == "clef":
        from .clef_engine import ClefEngine

        from .clef_engine import DeferOption

        defer = None
        if a.clef_defer_head:
            defer = DeferOption.from_file(a.clef_defer_head)
        elif a.clef_defer:
            defer = DeferOption(mode="option")
        clef = ClefEngine.load(a.model, name=a.name, max_len=a.clef_max_len, coalesce=a.clef_coalesce, defer=defer)
        if a.clef_warmup:
            clef.info["warmup"] = clef.warmup(a.clef_warmup)
        print(json.dumps({"engine": clef.describe()}, default=str))
        uvicorn.run(build_app(clef), host=a.host, port=a.port)
        return

    eng = Engine(a.model, a.adapter, a.calibration, a.name, device=a.device, max_len=a.max_len,
                 batch_size=a.batch_size, session_cache=a.session_cache, cache_tokens=a.cache_tokens,
                 cache_entries=a.cache_entries, prefill_chunk=a.prefill_chunk, batch_suffixes=a.batch_suffixes,
                 suffix_batch_tokens=a.suffix_batch_tokens, knockout_probs=a.knockout_probs,
                 fuse_multi=a.fuse_multi or a.fast,
                 suffix_pad_slack=a.suffix_pad_slack if a.suffix_pad_slack is not None else (0.25 if a.fast else None),
                 readout_gather=a.readout_gather or a.fast, preencode=a.preencode or a.fast)
    print(json.dumps({"engine": eng.describe()}))
    uvicorn.run(build_app(eng), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
