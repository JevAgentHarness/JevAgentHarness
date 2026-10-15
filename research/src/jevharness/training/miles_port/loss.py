"""The Miles custom loss of the decision model (``--loss-type custom_loss --custom-loss-function-path
jevharness.training.miles_port.loss.jev_decision_loss``).

Miles hands the loss the logits of a packed micro-batch (``[1, T, V]`` for ``qkv_format=thd``, ``[B, S, V]`` for
``bshd``). Every sample is one decision whose response is ONE option-code token, so the decision position is the
last prompt token: ``decision_rows`` slices exactly the row Miles' ``_iter_response_chunks`` gives for the first
response token (context parallelism 1). The row is restricted to the option-code ids of the presented order
(``metadata["codes"]``), which is v1's readout (``decision_model.option_logits``), and the stage's objective is v1's:

  sft    soft cross-entropy + ``brier`` x Brier against the target over the presented options
         (``decision_model.decision_loss("soft_ce_brier")``)
  opd    divergence between the teacher target and the student's restricted softmax (``opd.opd_loss``)
  grpo   PPO-clip on the realised event's log-probability under softmax(z / T) against the STORED behaviour
         log-probability of the SGLang rollout engine (``grpo_online.grpo_decision_loss`` with no KL / entropy term)

Per-sample values are reduced with Miles' ``sum_of_sample_mean`` (each sample has one response token, so a sample's
value enters once, divided by its rollout's mask sum: a GRPO episode's decisions share one rollout and average to one
episode term). Metrics are reduced the same way (Miles divides by the sample count when it logs them).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from ..grpo_online import event_logp
from ..opd import opd_loss


def _cp_size() -> int:
    try:
        from miles.backends.training_utils.parallel import get_parallel_state
    except Exception:  # noqa: BLE001 - tests run without Miles
        return 1
    try:
        return int(get_parallel_state().cp.size)
    except Exception:  # noqa: BLE001
        return 1


def decision_rows(logits: torch.Tensor, total_lengths: Sequence[int], response_lengths: Sequence[int],
                  qkv_format: str = "thd", max_seq_lens: Sequence[int] | None = None) -> list[torch.Tensor]:
    """One ``[V]`` row per sample: the logits that predict the sample's first response token."""
    if _cp_size() != 1:
        raise RuntimeError("the decision loss supports context parallelism 1 only")
    if logits.dim() != 3:
        raise ValueError(f"logits must be 3-D, got {tuple(logits.shape)}")
    rows: list[torch.Tensor] = []
    if qkv_format == "thd":
        flat = logits.squeeze(0)
        end = 0
        for L, R in zip(total_lengths, response_lengths):
            end += int(L)
            rows.append(flat[end - int(R) - 1])
    else:
        if max_seq_lens is None:
            raise ValueError("bshd needs max_seq_lens")
        flat = logits.reshape(-1, logits.size(-1))
        for i, (L, R) in enumerate(zip(total_lengths, response_lengths)):
            end = int(max_seq_lens[i]) * i + int(L)
            rows.append(flat[end - int(R) - 1])
    return rows


def _codes(m: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.as_tensor([int(c) for c in m["codes"]], dtype=torch.long, device=device)


def sample_terms(z: torch.Tensor, m: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Loss (0-d, with grad) and detached statistics of ONE decision; ``z`` = restricted option logits [K] (fp32)."""
    stage = m["stage"]
    if stage == "sft":
        t = torch.as_tensor(m["target"], dtype=z.dtype, device=z.device)
        logp = F.log_softmax(z, -1)
        p = logp.exp()
        ce = -(t * logp).sum()
        brier = ((p - t) ** 2).sum()
        loss = ce + float(m.get("brier", 0.5)) * brier
        return loss, {"ce": ce.detach(), "brier": brier.detach(),
                      "acc": (z.argmax() == t.argmax()).float().detach()}
    if stage == "opd":
        t = torch.as_tensor(m["target"], dtype=z.dtype, device=z.device)
        div = opd_loss(z[None], t[None], m.get("loss", "forward_kl"))[0]
        return div, {"div": div.detach(), "agree": (z.argmax() == t.argmax()).float().detach()}
    if stage == "grpo":
        T = float(m.get("temperature", 1.0))
        logpT = F.log_softmax(z / T, -1)
        new = event_logp(logpT, [int(i) for i in m["event"]], str(m["ev"]))
        old = torch.as_tensor(float(m["old_logp"]), dtype=z.dtype, device=z.device)
        adv = torch.as_tensor(float(m["adv"]), dtype=z.dtype, device=z.device)
        ratio = torch.exp(new - old)
        lo, hi = float(m.get("clip", 0.2)), float(m.get("clip_high", m.get("clip", 0.2)))
        pg = torch.maximum(-adv * ratio, -adv * ratio.clamp(1.0 - lo, 1.0 + hi))
        logp1 = F.log_softmax(z, -1)
        ent = -(logp1.exp() * logp1).sum()
        clipped = ((ratio < 1.0 - lo) & (adv < 0)) | ((ratio > 1.0 + hi) & (adv > 0))
        return pg, {"pg": pg.detach(), "ratio": ratio.detach(), "clipped": clipped.float().detach(),
                    "logp_gap": (new.detach() - old).abs(), "entropy": ent.detach()}
    raise ValueError(f"unknown stage {stage!r}")


def jev_decision_loss(args: Any, batch: dict[str, Any], logits: torch.Tensor,
                      sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor]
                      ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metas = batch.get("metadata")
    if metas is None:
        raise RuntimeError("batch has no 'metadata': the Miles key patch is not installed in the trainer process "
                           "(jevharness.training.miles_port.keys_patch)")
    if any(int(r) != 1 for r in batch["response_lengths"]):
        raise ValueError("every decision sample has exactly one response token")
    if batch.get("_jev_rows_only"):  # the LM head ran only at the decision positions (keys_patch.patch_fsdp_actor)
        if logits.dim() != 3 or logits.size(1) != len(metas):
            raise ValueError(f"rows-only logits {tuple(logits.shape)} for {len(metas)} samples")
        rows = [logits[0, i] for i in range(len(metas))]
    else:
        rows = decision_rows(logits, batch["total_lengths"], batch["response_lengths"],
                             getattr(args, "qkv_format", "thd"), batch.get("max_seq_lens"))
    losses: list[torch.Tensor] = []
    stats: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row, m in zip(rows, metas):
        z = row.float().index_select(0, _codes(m, row.device))
        loss, st = sample_terms(z, m)
        losses.append(loss)
        for k, v in st.items():
            stats[k].append(v.reshape(()))
    per = torch.stack(losses)
    loss = sum_of_sample_mean(per)
    out = {"loss": loss.detach()}
    for k, vs in stats.items():
        if len(vs) == len(losses):
            out[k] = sum_of_sample_mean(torch.stack(vs).to(per.dtype)).detach()
    if per.numel() == 0:
        loss = loss + 0 * logits.sum(dtype=torch.float32)
    return loss, out
