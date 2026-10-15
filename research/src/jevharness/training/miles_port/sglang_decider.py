"""v1's in-harness decider with its forward pass served by the Miles SGLang engine.

``SGLangDecider`` subclasses ``grpo_online.TorchDecider`` and replaces only ``_forward`` (user prompt, n options) ->
(option scores, prompt tokens). Prompt rendering (``DecisionCollator.encode`` = ``chat_text`` + tail ``max_len``), the
option codes, the > 52-option knockout, the samplers (replace: categorical; cascade: stochastic commit / defer; assist:
Plackett-Luce top-k; guards: yes / no) and the stored behaviour log-probabilities are v1's code, unchanged. The scores
are log-softmax values over the full vocabulary, which equal the logits up to one constant per prompt, so every
restricted softmax ``softmax(z / T)`` over the options is the one the logits give.

``OPDSGLangDecider`` is the same for on-policy distillation: like ``opd.OPDDecider`` it keeps every visited decision
(state, question, scored option pool) so the teacher can be asked at the same state.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..grpo_online import TorchDecider
from ..opd import OPDDecider
from .sglang_client import SGLangClient


class _SGLangForward:
    client: SGLangClient
    collator: Any
    batch_size: int

    def enable_batching(self, on: bool = True) -> None:  # noqa: ARG002
        """No client-side micro-batching: every rollout thread sends its own request, so the router spreads them over
        the engines and SGLang batches (and prefix-caches) them server-side."""
        self._batcher = None

    def _forward(self, items: list[tuple[str, int]]) -> list[tuple[list[float], int]]:  # type: ignore[override]
        ids = [self.collator.encode(p) for p, _ in items]
        codes = [self.collator.codes[:n] for _, n in items]
        out: list[tuple[list[float], int]] = []
        bs = max(1, int(self.batch_size))
        for s in range(0, len(items), bs):
            lps = self.client.option_logprobs(ids[s:s + bs], codes[s:s + bs])
            out += [(lp, len(t)) for lp, t in zip(lps, ids[s:s + bs])]
        return out


class SGLangDecider(_SGLangForward, TorchDecider):
    name = "miles_sglang_decider"

    def __init__(self, tokenizer: Any, url: str, harness_cfg: Any = None, *, client: SGLangClient | None = None,
                 temperature: float = 1.0, greedy: bool = False, max_len: int = 4096, batch_size: int = 32,
                 max_wait_ms: float = 5.0, price_model: str | None = None, name: str = "jevagentrl-v2"):
        TorchDecider.__init__(self, None, tokenizer, harness_cfg, temperature=temperature, greedy=greedy,
                              max_len=max_len, batch_size=batch_size, max_wait_ms=max_wait_ms,
                              price_model=price_model, name=name)
        self.client = client or SGLangClient(url)


class OPDSGLangDecider(_SGLangForward, OPDDecider):
    name = "miles_sglang_opd_decider"

    def __init__(self, tokenizer: Any, url: str, harness_cfg: Any = None, *, client: SGLangClient | None = None,
                 temperature: float = 1.0, greedy: bool = False, max_len: int = 4096, batch_size: int = 32,
                 max_wait_ms: float = 5.0, price_model: str | None = None, name: str = "jevagentrl-v2-opd"):
        OPDDecider.__init__(self, None, tokenizer, harness_cfg, temperature=temperature, greedy=greedy,
                            max_len=max_len, batch_size=batch_size, max_wait_ms=max_wait_ms,
                            price_model=price_model, name=name)
        self.client = client or SGLangClient(url)


def records_prompt_ids(collator: Any, prompts: Sequence[str]) -> list[list[int]]:
    """The token ids the engine scored for each prompt (the Miles sample = these + one option-code token)."""
    return [collator.encode(p) for p in prompts]
