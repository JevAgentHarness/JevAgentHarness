"""JevGym-Hard (exploratory row E7, docs/PREREG_A7_STAGE2.md Deviations 2026-10-06): ``multi`` tasks whose operations
follow an order policy instead of being named per customer.

``JevGymHardEnv`` is :class:`~.jevgym.JevGymEnv` with one more option, ``multi_rule``. It lives in its own module so the
frozen generator ``jevgym.py`` (sha256-locked by ``scripts/prereg_check.py``, A6.16 (l)) stays byte-identical:

  given   (default) the parent's task, unchanged: each customer's operation is named in the instruction
  amount  the instruction names only the customers; cancel a pending order above $450.00, ship it otherwise
  tier    cancel it if the customer is basic tier and the amount is above $300.00, or silver tier and the amount is
          above $600.00; never cancel a gold-tier customer's order; ship it otherwise
  random  (E12b, docs/PREREG_A7_STAGE2.md Deviations 2026-10-07) one rule per seed, drawn from its own stream
          (``random_rule``; the parent's draws are unchanged): cancel above $T, cancel below $T, or a per-tier rule with
          its own thresholds. Its thresholds never take tier's values ($300 / $600), and a per-tier draw that would read
          as tier's rule is redrawn, so ``tier`` stays unseen by a model trained on ``random``

Under ``amount`` / ``tier`` the agent must derive each operation from what the episode reveals (the tier from
``find_user``, the amount from ``get_orders``). The world, the customers, their order and the orders of a seed are those
of its ``given`` task (the parent's random draws are made unchanged, then the operations are replaced by the rule's),
and the gold plan keeps its shape (find_user, get_orders, cancel / ship per customer, finish with the total), so every
oracle and gold of the parent applies unchanged. The rule changes ``multi`` tasks only. ``load_suite("jevgym", ...)``
builds this class when the suite block carries ``multi_rule``.
"""
from __future__ import annotations

import random
from typing import Any, Callable

from .jevgym import ENV_KWARGS as BASE_ENV_KWARGS
from .jevgym import JevGymEnv

MULTI_RULES = ("given", "amount", "tier", "random")
MULTI_AMOUNT_CANCEL_ABOVE = 450.0                                  # amount: cancel strictly above this
MULTI_TIER_CANCEL_ABOVE = {"basic": 300.0, "silver": 600.0}       # tier: gold tier is never cancelled
TIERS = ("basic", "silver", "gold")
RANDOM_GRID = tuple(float(x) for x in range(150, 801, 50) if x not in (300, 600))   # never tier's thresholds


def random_rule(seed: int) -> dict[str, Any]:
    """E12b's rule for one seed, from its own stream: {"kind": "above" | "below", "t": T} (cancel strictly above /
    below $T) or {"kind": "tiered", "t": {tier: T or None}} (None = never cancel that tier). Thresholds come from
    RANDOM_GRID; a tiered draw always has a threshold for at least one tier and is never tier's rule."""
    rng = random.Random(f"jevgym-hard-random-rule:{int(seed)}")
    kind = rng.choice(("above", "below", "tiered"))
    if kind != "tiered":
        return {"kind": kind, "t": rng.choice(RANDOM_GRID)}
    while True:
        th = {tr: (None if rng.random() < 0.3 else rng.choice(RANDOM_GRID)) for tr in TIERS}
        if any(v is not None for v in th.values()) and th != {"basic": 300.0, "silver": 600.0, "gold": None}:
            return {"kind": "tiered", "t": th}
ANSWER_STYLES = ("total", "itemized")
ENV_KWARGS = BASE_ENV_KWARGS + ("multi_rule", "answer_style")


class JevGymHardEnv(JevGymEnv):
    def __init__(self, seed: int, *args: Any, multi_rule: str = "given", answer_style: str = "total", **kw: Any):
        if multi_rule not in MULTI_RULES:
            raise ValueError(f"multi_rule must be one of {MULTI_RULES}, got {multi_rule!r}")
        if answer_style not in ANSWER_STYLES:
            raise ValueError(f"answer_style must be one of {ANSWER_STYLES}, got {answer_style!r}")
        self.multi_rule = multi_rule          # read by _build_multi, which the parent's __init__ calls
        self.rule = random_rule(seed) if multi_rule == "random" else None
        self.answer_style = answer_style      # E7e: "itemized" asks for one line per customer before the total
        super().__init__(seed, *args, **kw)

    def rule_cancels(self, user: dict[str, Any], order: dict[str, Any]) -> bool:
        """Is this customer's pending order cancelled (else shipped) under the rule? Strictly above the threshold."""
        if self.multi_rule == "amount":
            return float(order["amount"]) > MULTI_AMOUNT_CANCEL_ABOVE
        if self.multi_rule == "random":
            r, amt = self.rule, float(order["amount"])
            if r["kind"] == "above":
                return amt > r["t"]
            if r["kind"] == "below":
                return amt < r["t"]
            lim = r["t"].get(user["tier"])
            return lim is not None and amt > lim
        lim = MULTI_TIER_CANCEL_ABOVE.get(user["tier"])
        return lim is not None and float(order["amount"]) > lim

    def policy_text(self) -> str:
        if self.multi_rule == "amount":
            return f"cancel it if its amount is above ${MULTI_AMOUNT_CANCEL_ABOVE:.2f}, otherwise ship it"
        if self.multi_rule == "random":
            r = self.rule
            if r["kind"] in ("above", "below"):
                return f"cancel it if its amount is {r['kind']} ${r['t']:.2f}, otherwise ship it"
            parts = [f"{tr} tier and the amount is above ${v:.2f}" for tr, v in r["t"].items() if v is not None]
            never = [tr for tr, v in r["t"].items() if v is None]
            txt = "cancel it if the customer is " + ", or ".join(parts)
            if never:
                txt += " (never cancel a " + " or ".join(f"{tr}-tier" for tr in never) + " customer's order)"
            return txt + "; otherwise ship it"
        return (f"cancel it if the customer is basic tier and the amount is above "
                f"${MULTI_TIER_CANCEL_ABOVE['basic']:.2f}, or silver tier and the amount is above "
                f"${MULTI_TIER_CANCEL_ABOVE['silver']:.2f} (never cancel a gold-tier customer's order); otherwise ship it")

    def _build_multi(self, rng: random.Random) -> None:
        super()._build_multi(rng)            # the parent's draws (customers, operations) are made unchanged
        if self.multi_rule == "given":
            return
        self.jobs = [("cancel" if self.rule_cancels(u, o) else "ship", u, o) for _, u, o in self.jobs]
        names = "; ".join(f"({i + 1}) {u['name']}" for i, (_, u, _) in enumerate(self.jobs))
        self.instruction = (f"Handle the pending order of each of these customers, in order: {names}. For each "
                            f"customer, look up their pending order and {self.policy_text()}. Cancellations use the "
                            "reason customer_request; use express shipping for gold-tier customers and standard "
                            "shipping otherwise. " + self.answer_text())

    def answer_text(self) -> str:
        """The instruction's last sentence. ``total`` (E7-E7c) is the original wording; ``itemized`` (E7e, Deviations
        2026-10-06 "E7e") asks for the per-customer lines first: in E7c every failure was a wrong final total, and a
        bare-number answer almost never matched (0 of 31 LLM-only and 3 of 74 Jev-cascade bare answers did)."""
        if self.answer_style == "itemized":
            return ("When all are done, answer with one line per customer giving the action you took and the order "
                    "amount, then the total amount of the orders you processed, rounded to 2 decimals.")
        return "When all are done, answer with the total amount of the orders you processed, rounded to 2 decimals."


def task_factories(n: int = 50, seed0: int = 0, **kw: Any) -> list[Callable[[], JevGymHardEnv]]:
    """Zero-argument factories, as ``jevgym.task_factories`` (same options plus ``multi_rule``)."""
    unknown = sorted(set(kw) - set(ENV_KWARGS))
    if unknown:
        raise ValueError(f"unknown jevgym suite option(s) {unknown}; valid: n, seed0, {', '.join(ENV_KWARGS)}")
    JevGymHardEnv(seed=seed0, **kw)  # validate the options once, eagerly
    return [(lambda i=i: JevGymHardEnv(seed=seed0 + i, **kw)) for i in range(n)]
