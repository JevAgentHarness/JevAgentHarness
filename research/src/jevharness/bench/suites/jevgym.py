"""JevGym: a small, deterministic, fully-controllable tool-use world for *diagnostic* studies.

Why a synthetic suite in addition to public benchmarks?  It exposes an exact oracle policy at
every state, so we can (i) score every System-1 decision against gold, (ii) sweep the factors
that determine whether fast decisions are safe -- tool-set size (distractors) and how confusable
they are, horizon length, fraction of copyable vs. generated arguments, observation noise, missing
information -- and (iii) run end-to-end offline with simulated backends for CI.

Task families (all need ID copying across steps; some need enum args or long-horizon recall):

  v1 (default round-robin, unchanged since the pilots)
  cancel      find_user -> get_orders(pending) -> cancel_order(reason) -> finish
  refund      find_user -> get_orders(delivered) -> refund(amount, method) -> finish
  stock       check_stock -> finish(count)
  ship        find_user -> get_orders(pending) -> ship_order(speed depends on tier) -> finish
  total       find_user -> get_orders(delivered) -> finish(sum of amounts)

  v2 (``families="v2"`` / ``"all"`` or ``family=...``)
  clarify     ask / abstain: information needed to act is missing or ambiguous, so the right action is
              ``ask_user(missing=...)`` *before* any state-changing call; four variants --
                order   the customer has several pending orders ("cancel their pending order")
                method  refund without a refund method
                speed   ship without a shipping-speed policy
                name    the customer never said who they are
              guessing (acting without asking) fails even when the guess happens to be right
  multi       long horizon: ``multi_customers`` (default 3) customers, each with a cancel or ship request,
              then ``finish(total amount of the processed orders)`` -- the amounts appear only in the
              earlier ``get_orders`` observations, so the final answer needs memory of them

Distractor styles (``distractor_style``):

  random      unrelated tools with a single ``query`` argument (v1 default)
  near        near-duplicates of the core tools with plausible descriptions and typed parameters
              (``get_order`` vs ``get_orders``, ``cancel_subscription``, ``refund_shipping_fee``,
              ``find_customer_by_email``, ``ship_order_international``, ``check_stock_warehouse``, ...).
              They are functional: read-only ones return plausible (sometimes tempting) observations,
              state-changing ones apply a *side effect* that fails the episode (and ends it in strict mode)
  mixed       ceil(n/2) near-duplicates + random ones

The oracle (``oracle_action``) is exact for every family: it returns the first step of the canonical
gold plan that has not been executed yet; ``evaluate`` checks the final world state, the answer and
(for ``clarify``) that the right question was asked before acting.

Meta-decision golds (docs/META_GOLD_SPEC.md §2) are pure functions of the environment state, computed on
independent copies (``oracle_snapshot``) so the episode itself never changes: ``oracle_cost_to_go`` (the
early-exit canonical rollout D(s, B)), ``oracle_verdict`` (progress / detour / fatal / premature / wrong_answer of
a proposed call), ``oracle_finish_ok``, ``oracle_world_complete``, ``oracle_need`` (the evidence atoms the
remaining plan consumes), ``oracle_acceptable_tools`` (the information-feasible acceptable tool set) and the
memory golds ``oracle_mem_gold`` / ``oracle_memory_read_gold``. The harness reads them via ``getattr`` after the
System-1 answer of a decision is recorded; they never reach a backend.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..env import Env, StepResult

INF = float("inf")
FAILING = frozenset({"fatal", "premature", "wrong_answer"})       # executing the action makes success impossible
# token-boundary literal match of identifiers / tiers / names (META_GOLD_SPEC §2.4); the left boundary excludes the
# ``#123456`` heartbeat noise. Right boundary: the spec's (?![\w.]), relaxed to admit a sentence-final period
# ("my name is Dana Kim.") while still rejecting a continuation such as "U123.4".
_ID_L, _ID_R = r"(?<![\w.#])", r"(?!\w|\.\w)"
_NUM_TOKEN = re.compile(r"(?<![\w.#-])(\d+(?:\.\d+)?)(?![\w.])")

FIRST = ["Alice", "Bob", "Chen", "Dana", "Emeka", "Farah", "Goro", "Hana", "Ivan", "Jia", "Kofi", "Lena",
         "Mateo", "Nia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tariq"]
LAST = ["Smith", "Garcia", "Wang", "Kim", "Okafor", "Haddad", "Tanaka", "Novak", "Silva", "Ali"]
ITEMS = ["laptop", "headphones", "keyboard", "monitor", "webcam", "desk lamp", "router", "tablet", "charger", "mouse"]
STATUSES = ["pending", "shipped", "delivered", "cancelled"]
TIERS = ["basic", "silver", "gold"]
WAREHOUSES = ["north", "south", "east", "west"]
REASONS = ["customer_request", "out_of_stock", "fraud"]
METHODS = ["original_payment", "store_credit"]
SPEEDS = ["standard", "express"]
MISSING = ["customer_name", "order_id", "refund_method", "shipping_speed"]


def _fn(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


CORE_TOOLS = [
    _fn("find_user", "Look up a customer by full name; returns user_id and tier.",
        {"name": {"type": "string", "description": "customer full name"}}, ["name"]),
    _fn("get_orders", "List a customer's orders filtered by status.",
        {"user_id": {"type": "string"},
         "status": {"type": "string", "enum": ["all", "pending", "shipped", "delivered", "cancelled"]}},
        ["user_id", "status"]),
    _fn("cancel_order", "Cancel an order.",
        {"order_id": {"type": "string"},
         "reason": {"type": "string", "enum": ["customer_request", "out_of_stock", "fraud"]}},
        ["order_id", "reason"]),
    _fn("refund", "Refund an order.",
        {"order_id": {"type": "string"}, "amount": {"type": "number"},
         "method": {"type": "string", "enum": ["original_payment", "store_credit"]}},
        ["order_id", "amount", "method"]),
    _fn("check_stock", "Units in stock for an item.", {"item": {"type": "string", "enum": ITEMS}}, ["item"]),
    _fn("ship_order", "Ship a pending order.",
        {"order_id": {"type": "string"}, "speed": {"type": "string", "enum": ["standard", "express"]}},
        ["order_id", "speed"]),
    _fn("send_message", "Send a free-text message to a customer.",
        {"user_id": {"type": "string"}, "text": {"type": "string"}}, ["user_id", "text"]),
]

# v2: a dedicated clarification tool (present for v2 families, or with ``ask_tool=True``)
ASK_TOOL = _fn(
    "ask_user", "Ask the customer a clarifying question and wait for the reply. Use it only when information "
    "needed to act is missing or ambiguous.",
    {"missing": {"type": "string", "enum": MISSING, "description": "which piece of information is missing"},
     "question": {"type": "string", "description": "the question to put to the customer"}}, ["missing"])

ASK_POLICY = (" Policy: never guess -- if a detail you need to act (who the customer is, which order, the "
              "refund method, the shipping speed) is not given or is ambiguous, ask the customer with ask_user "
              "before taking any action.")

DISTRACTOR_NAMES = [
    ("get_weather", "Current weather for a city."), ("convert_currency", "Convert between currencies."),
    ("search_docs", "Search the internal knowledge base."), ("create_ticket", "Open an IT support ticket."),
    ("list_warehouses", "List warehouse locations."), ("get_exchange_rate", "Exchange rate between currencies."),
    ("schedule_meeting", "Schedule a calendar meeting."), ("translate_text", "Translate text."),
    ("get_supplier", "Supplier details for a product line."), ("update_price", "Change an item's list price."),
    ("audit_log", "Fetch audit-log entries."), ("reset_password", "Reset an employee password."),
    ("get_holidays", "Public holidays for a country."), ("compute_tax", "Compute sales tax."),
    ("list_promotions", "Active promotions."), ("get_shipping_zones", "Shipping zones for a region."),
    ("archive_user", "Archive an inactive customer account."), ("merge_accounts", "Merge two customer accounts."),
    ("generate_invoice", "Generate an invoice PDF."), ("check_fraud_score", "Fraud score for a transaction."),
    ("get_return_policy", "Return policy text."), ("set_reminder", "Set a reminder."),
    ("list_employees", "List employees in a team."), ("forecast_demand", "Demand forecast for an item."),
    ("rate_limit_status", "API rate-limit status."), ("get_order_history_export", "Export order history as CSV."),
    ("update_address", "Update a customer's shipping address."), ("verify_identity", "Start ID verification."),
    ("apply_coupon", "Apply a coupon to an order."), ("get_loyalty_points", "Customer loyalty points."),
]


def distractor_tools(n: int) -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        name, desc = DISTRACTOR_NAMES[i % len(DISTRACTOR_NAMES)]
        if i >= len(DISTRACTOR_NAMES):
            name = f"{name}_v{i // len(DISTRACTOR_NAMES) + 1}"
        out.append(_fn(name, desc, {"query": {"type": "string"}}, ["query"]))
    return out


# ---------------------------------------------------------------------- near-duplicate distractors
_S = {"type": "string"}
_ORDER = {"type": "string", "description": "order id, e.g. ORD-1234"}
_USER = {"type": "string", "description": "customer user_id"}


@dataclass(frozen=True)
class NearTool:
    """A distractor that shadows core tool ``shadows``. ``write`` tools change the world: calling one is a
    harmful side effect (episode failure; immediate failure in strict mode)."""
    name: str
    shadows: str
    desc: str
    props: dict[str, Any]
    required: tuple[str, ...]
    write: bool = False

    def spec(self) -> dict[str, Any]:
        return _fn(self.name, self.desc, self.props, list(self.required))


NEAR_TOOLS: list[NearTool] = [
    # find_user
    NearTool("find_customer_by_email", "find_user", "Look up a customer by e-mail address; returns user_id and tier.",
             {"email": {"type": "string", "description": "customer e-mail address"}}, ("email",)),
    NearTool("find_user_by_phone", "find_user", "Look up a customer by phone number; returns user_id and tier.",
             {"phone": {"type": "string", "description": "phone number in E.164 format"}}, ("phone",)),
    NearTool("lookup_employee", "find_user", "Look up a staff member (employee) by full name; returns "
             "employee_id and team.", {"name": {"type": "string", "description": "employee full name"}}, ("name",)),
    NearTool("search_users", "find_user", "List customers whose last name starts with a prefix (display names "
             "only, no ids).", {"prefix": {"type": "string", "description": "last-name prefix"}}, ("prefix",)),
    # (``prefix`` is classified free text by ``JevGymEnv.param_kinds``: its description mentions a "name")
    # get_orders
    NearTool("get_order", "get_orders", "Get one order by its order id.", {"order_id": _ORDER}, ("order_id",)),
    NearTool("get_order_status", "get_orders", "Get only the status field of one order.", {"order_id": _ORDER},
             ("order_id",)),
    NearTool("list_orders_by_item", "get_orders", "Count store-wide orders of an item with a given status (all "
             "customers, no details).", {"item": {"type": "string", "enum": ITEMS},
                                         "status": {"type": "string", "enum": STATUSES}}, ("item", "status")),
    NearTool("get_orders_archive", "get_orders", "List a customer's orders from the archive of a past calendar "
             "year (orders older than 12 months only).",
             {"user_id": _USER, "year": {"type": "integer", "minimum": 2015, "maximum": 2025}}, ("user_id", "year")),
    NearTool("get_orders_summary", "get_orders", "Number of orders per status for a customer (counts only, no "
             "ids or amounts).", {"user_id": _USER}, ("user_id",)),
    # cancel_order
    NearTool("cancel_subscription", "cancel_order", "Cancel a customer's recurring subscription plan (not an "
             "individual order).", {"user_id": _USER, "reason": {"type": "string", "enum": REASONS}},
             ("user_id", "reason"), write=True),
    NearTool("cancel_order_item", "cancel_order", "Remove one item line from a multi-item order; the rest of the "
             "order stays active.", {"order_id": _ORDER, "item": {"type": "string", "enum": ITEMS},
                                     "reason": {"type": "string", "enum": REASONS}}, ("order_id", "item", "reason"),
             write=True),
    NearTool("cancel_shipment", "cancel_order", "Recall a shipment that is already in transit (orders with status "
             "shipped).", {"order_id": _ORDER, "reason": {"type": "string", "enum": REASONS}},
             ("order_id", "reason"), write=True),
    NearTool("void_payment", "cancel_order", "Void the card authorisation of an order's payment (payments team; "
             "the order itself stays open).", {"order_id": _ORDER}, ("order_id",), write=True),
    # refund
    NearTool("refund_shipping_fee", "refund", "Refund only the shipping fee of an order (not the item amount).",
             {"order_id": _ORDER, "method": {"type": "string", "enum": METHODS}}, ("order_id", "method"), write=True),
    NearTool("issue_store_credit", "refund", "Add goodwill store credit to a customer's account; not linked to any "
             "order and not a refund.", {"user_id": _USER, "amount": {"type": "number"}}, ("user_id", "amount"),
             write=True),
    NearTool("partial_refund", "refund", "Refund part of an order's amount for a damaged item; requires a damage "
             "report id.", {"order_id": _ORDER, "amount": {"type": "number"}, "damage_report_id": _S},
             ("order_id", "amount", "damage_report_id"), write=True),
    NearTool("reverse_chargeback", "refund", "Respond to a bank-initiated card chargeback dispute on an order.",
             {"order_id": _ORDER, "accept": {"type": "boolean"}}, ("order_id", "accept"), write=True),
    # check_stock
    NearTool("check_stock_warehouse", "check_stock", "Units of an item held at ONE warehouse (the company-wide "
             "stock is the sum over all warehouses).", {"item": {"type": "string", "enum": ITEMS},
                                                        "warehouse": {"type": "string", "enum": WAREHOUSES}},
             ("item", "warehouse")),
    NearTool("check_stock_incoming", "check_stock", "Units of an item on order from suppliers (not yet in stock).",
             {"item": {"type": "string", "enum": ITEMS}}, ("item",)),
    NearTool("get_item_price", "check_stock", "Current list price of an item.",
             {"item": {"type": "string", "enum": ITEMS}}, ("item",)),
    NearTool("reserve_stock", "check_stock", "Reserve units of an item for an order (reduces available stock).",
             {"item": {"type": "string", "enum": ITEMS}, "units": {"type": "integer"}}, ("item", "units"), write=True),
    NearTool("restock_item", "check_stock", "Order more units of an item from the supplier.",
             {"item": {"type": "string", "enum": ITEMS}, "units": {"type": "integer"}}, ("item", "units"), write=True),
    # ship_order
    NearTool("ship_order_international", "ship_order", "Ship a pending order to an address outside the home "
             "country (adds customs paperwork and duties).",
             {"order_id": _ORDER, "speed": {"type": "string", "enum": SPEEDS}, "country": _S},
             ("order_id", "speed", "country"), write=True),
    NearTool("ship_replacement", "ship_order", "Ship a free replacement for a damaged, already delivered order.",
             {"order_id": _ORDER, "speed": {"type": "string", "enum": SPEEDS}}, ("order_id", "speed"), write=True),
    NearTool("upgrade_shipping", "ship_order", "Change the shipping speed of an order that has already shipped.",
             {"order_id": _ORDER, "speed": {"type": "string", "enum": SPEEDS}}, ("order_id", "speed"), write=True),
    NearTool("get_shipping_quote", "ship_order", "Quote the price of shipping an order at a given speed (does not "
             "ship it).", {"order_id": _ORDER, "speed": {"type": "string", "enum": SPEEDS}}, ("order_id", "speed")),
    NearTool("schedule_pickup", "ship_order", "Schedule a courier pickup of a customer's return parcel.",
             {"order_id": _ORDER, "date": {"type": "string", "description": "YYYY-MM-DD"}}, ("order_id", "date"),
             write=True),
    # send_message / ask_user
    NearTool("send_email", "send_message", "E-mail a customer (no reply is collected).",
             {"user_id": _USER, "subject": _S, "body": _S}, ("user_id", "subject", "body")),
    NearTool("notify_warehouse", "send_message", "Send an internal note about an order to the warehouse team.",
             {"order_id": _ORDER, "text": _S}, ("order_id", "text")),
    NearTool("request_callback", "ask_user", "Schedule a phone call-back to the customer for a later day (no "
             "immediate reply).", {"user_id": _USER, "date": {"type": "string", "description": "YYYY-MM-DD"}},
             ("user_id", "date")),
    NearTool("escalate_to_supervisor", "ask_user", "Hand the case over to a human supervisor (fraud, legal or "
             "abuse cases only).", {"reason": {"type": "string", "enum": ["fraud", "legal", "abuse"]}}, ("reason",),
             write=True),
]
NEAR_BY_NAME = {t.name: t for t in NEAR_TOOLS}
DISTRACTOR_STYLES = ("random", "near", "mixed")


def _interleaved_near() -> list[NearTool]:
    """Near-duplicates ordered round-robin over the core tool they shadow, so any prefix covers every core tool
    as evenly as possible."""
    groups: dict[str, list[NearTool]] = {}
    for t in NEAR_TOOLS:
        groups.setdefault(t.shadows, []).append(t)
    out: list[NearTool] = []
    for i in range(max(len(g) for g in groups.values())):
        out += [g[i] for g in groups.values() if i < len(g)]
    return out


def near_distractor_tools(n: int) -> list[dict[str, Any]]:
    """``n`` near-duplicate distractors; beyond the pool (``len(NEAR_TOOLS)``) the rest are random ones."""
    pool = _interleaved_near()
    return [t.spec() for t in pool[:n]] + distractor_tools(max(0, n - len(pool)))


def make_distractors(n: int, style: str = "random") -> list[dict[str, Any]]:
    if style == "random":
        return distractor_tools(n)
    if style == "near":
        return near_distractor_tools(n)
    if style == "mixed":
        k = min(math.ceil(n / 2), len(NEAR_TOOLS))
        return near_distractor_tools(k) + distractor_tools(n - k)
    raise ValueError(f"distractor_style must be one of {DISTRACTOR_STYLES}, got {style!r}")


# ---------------------------------------------------------------------- world
@dataclass
class World:
    users: dict[str, dict[str, Any]] = field(default_factory=dict)      # user_id -> record
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)     # order_id -> record
    stock: dict[str, int] = field(default_factory=dict)


def make_world(rng: random.Random, n_users: int = 12) -> World:
    w = World()
    names = rng.sample([f"{f} {s}" for f in FIRST for s in LAST], n_users)
    oid = rng.randint(1000, 5000)
    for i, nm in enumerate(names):
        uid = f"U{rng.randint(100, 999)}{i}"
        w.users[uid] = {"user_id": uid, "name": nm, "tier": rng.choice(TIERS)}
        for s in STATUSES:
            for _ in range(rng.randint(0, 1) if s != "pending" else 1):
                oid += rng.randint(1, 9)
                w.orders[f"ORD-{oid}"] = {"order_id": f"ORD-{oid}", "user_id": uid, "item": rng.choice(ITEMS),
                                          "status": s, "amount": round(rng.uniform(15, 900), 2)}
        # guarantee at least one delivered order for refund/total tasks
        if not any(o["user_id"] == uid and o["status"] == "delivered" for o in w.orders.values()):
            oid += 3
            w.orders[f"ORD-{oid}"] = {"order_id": f"ORD-{oid}", "user_id": uid, "item": rng.choice(ITEMS),
                                      "status": "delivered", "amount": round(rng.uniform(15, 900), 2)}
    w.stock = {it: rng.randint(0, 250) for it in ITEMS}
    return w


def _speed(user: dict[str, Any]) -> str:
    return "express" if user["tier"] == "gold" else "standard"


def _resolve_families(families: str | Sequence[str] | None) -> list[str]:
    if families is None:
        return list(JevGymEnv.FAMILIES)
    if isinstance(families, str):
        if families in FAMILY_SETS:
            return list(FAMILY_SETS[families])
        families = [f.strip() for f in families.split(",") if f.strip()]
    fams = list(families)
    bad = [f for f in fams if f not in ALL_FAMILIES]
    if bad or not fams:
        raise ValueError(f"unknown JevGym families {bad or fams}; choose from {ALL_FAMILIES} or {list(FAMILY_SETS)}")
    return fams


class JevGymEnv(Env):
    FAMILIES = ["cancel", "refund", "stock", "ship", "total"]          # v1: the default round-robin
    V2_FAMILIES = ["clarify", "multi"]
    CLARIFY_VARIANTS = ["order", "method", "speed", "name"]

    STATE_CHANGING = {"cancel_order", "refund", "ship_order"}
    FINAL_TOOL = "finish"            # the harness's final tool (``HarnessConfig.final_tool`` default)
    # free-text policy (META_GOLD_SPEC §7.1 rule 1, set by the suite): parameters the schema rules would misclassify.
    # Kept out of the JSON schema itself so no provider ever sees a non-standard schema key.
    param_kinds: dict[str, str] = {"search_users.prefix": "free"}

    def __init__(self, seed: int, family: str | None = None, n_distractors: int = 10, noise_lines: int = 0,
                 strict: bool = False, distractor_style: str = "random",
                 families: str | Sequence[str] | None = None, ask_tool: bool | None = None,
                 multi_customers: int = 3):
        """strict=True makes state-changing calls irreversible: any cancel/refund/ship that deviates
        from the gold call (or, for ``clarify``, precedes the right question), and any state-changing
        near-duplicate distractor, ends the episode as a failure (large V in Prop. 1).

        families         round-robin family set when ``family`` is None: ``"v1"`` (default), ``"v2"``,
                         ``"all"``, or a list / comma-separated string of family names
        distractor_style ``"random"`` (default), ``"near"`` or ``"mixed"`` (see module docstring)
        ask_tool         offer ``ask_user``; default: iff a v2 family is in play (so within one suite every
                         task has the same tool set and its presence does not reveal the family)
        multi_customers  number of customers in a ``multi`` task (clipped to 2..12)"""
        fams = _resolve_families(families)
        if family is not None and family not in ALL_FAMILIES:
            raise ValueError(f"unknown JevGym family {family!r}; choose from {ALL_FAMILIES}")
        if distractor_style not in DISTRACTOR_STYLES:
            raise ValueError(f"distractor_style must be one of {DISTRACTOR_STYLES}, got {distractor_style!r}")
        self.seed = seed
        self.rng = random.Random(seed)
        self.family = family or fams[seed % len(fams)]
        self._rr = 1 if family else len(fams)      # round-robin period (balances the clarify variants)
        self.n_distractors = n_distractors
        self.noise_lines = noise_lines
        self.strict = strict
        self.distractor_style = distractor_style
        in_play = [family] if family else fams
        self.ask_tool = bool(ask_tool) if ask_tool is not None else any(f in self.V2_FAMILIES for f in in_play)
        self.multi_customers = min(12, max(2, int(multi_customers)))   # make_world has 12 customers
        self.task_id = (f"jevgym-{self.family}-{seed}-d{n_distractors}"
                        + (f"-{distractor_style}" if distractor_style != "random" else "")
                        + ("-strict" if strict else ""))
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        rng = self.rng
        self.world = make_world(rng)
        uid = rng.choice(list(self.world.users))
        self.user = self.world.users[uid]
        self.target_item = rng.choice(ITEMS)
        pend = [o for o in self.world.orders.values() if o["user_id"] == uid and o["status"] == "pending"]
        deliv = [o for o in self.world.orders.values() if o["user_id"] == uid and o["status"] == "delivered"]
        self.pending_order = pend[0]
        self.delivered_order = sorted(deliv, key=lambda o: o["order_id"])[-1]
        self.delivered_total = round(sum(o["amount"] for o in deliv), 2)
        self.variant: str | None = None
        self.missing: str | None = None           # clarify: what the customer must be asked for
        self.reply: str = ""                      # clarify: the customer's answer
        self.method, self.ship_speed = "store_credit", _speed(self.user)
        self.jobs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []  # multi: (op, user, order)
        nm = self.user["name"]
        f = self.family
        if f == "cancel":
            self.instruction = f"The customer {nm} asked us to cancel their pending order. Please do it."
        elif f == "refund":
            self.instruction = (f"Refund the full amount of {nm}'s most recent delivered order (highest order id) "
                                "as store credit.")
        elif f == "stock":
            self.instruction = f"How many units of '{self.target_item}' do we have in stock? Answer with the number."
        elif f == "ship":
            self.instruction = (f"Ship {nm}'s pending order. Use express shipping if the customer is gold tier, "
                                "otherwise standard.")
        elif f == "total":
            self.instruction = (f"What is the total amount of all of {nm}'s delivered orders? Answer with the "
                                "number rounded to 2 decimals.")
        elif f == "clarify":
            self._build_clarify(rng)
        else:
            self._build_multi(rng)
        tools = CORE_TOOLS + ([ASK_TOOL] if self.ask_tool else []) + make_distractors(self.n_distractors,
                                                                                        self.distractor_style)
        self._tools = tools[:]
        random.Random(self.seed + 7).shuffle(self._tools)
        self._tool_names = frozenset(t["function"]["name"] for t in self._tools)
        self._near_offered = self._tool_names & set(NEAR_BY_NAME)
        self.gold = {"family": f, "user_id": uid}
        if self.variant:
            self.gold.update({"variant": self.variant, "missing": self.missing})

    def _new_order_id(self, rng: random.Random) -> str:
        top = max(int(o.split("-")[1]) for o in self.world.orders)
        return f"ORD-{top + rng.randint(1, 9)}"

    def _build_clarify(self, rng: random.Random) -> None:
        u, nm = self.user, self.user["name"]
        self.variant = self.CLARIFY_VARIANTS[(self.seed // self._rr) % len(self.CLARIFY_VARIANTS)]
        if self.variant == "order":        # several pending orders: which one?
            for _ in range(rng.randint(1, 2)):
                oid = self._new_order_id(rng)
                self.world.orders[oid] = {"order_id": oid, "user_id": u["user_id"], "item": rng.choice(ITEMS),
                                          "status": "pending", "amount": round(rng.uniform(15, 900), 2)}
            pend = sorted((o for o in self.world.orders.values()
                           if o["user_id"] == u["user_id"] and o["status"] == "pending"), key=lambda o: o["order_id"])
            self.pending_order = rng.choice(pend)
            self.missing = "order_id"
            self.reply = (f"Sorry for the confusion -- please cancel order {self.pending_order['order_id']} "
                          f"(the {self.pending_order['item']}).")
            self.instruction = f"The customer {nm} asked us to cancel their pending order. Please do it."
        elif self.variant == "method":     # refund without a refund method
            self.method = rng.choice(METHODS)
            self.missing = "refund_method"
            self.reply = ("Please refund it as store credit." if self.method == "store_credit"
                          else "Please refund it to my original payment method.")
            self.instruction = f"Refund the full amount of {nm}'s most recent delivered order (highest order id)."
        elif self.variant == "speed":      # ship without a speed policy
            self.ship_speed = rng.choice(SPEEDS)
            self.missing = "shipping_speed"
            self.reply = f"{self.ship_speed.capitalize()} shipping, please."
            self.instruction = f"Ship {nm}'s pending order."
        else:                              # who is the customer?
            self.missing = "customer_name"
            self.reply = f"Oh, sorry -- my name is {nm}."
            self.instruction = ("A customer wrote in asking us to ship their pending order (express if they are "
                                "gold tier, otherwise standard), but did not say who they are.")

    def _build_multi(self, rng: random.Random) -> None:
        others = [x for x in self.world.users if x != self.user["user_id"]]
        uids = [self.user["user_id"]] + rng.sample(others, self.multi_customers - 1)
        parts = []
        for uid in uids:
            u = self.world.users[uid]
            order = next(o for o in self.world.orders.values() if o["user_id"] == uid and o["status"] == "pending")
            op = rng.choice(["ship", "cancel"])
            self.jobs.append((op, u, order))
            parts.append(f"cancel {u['name']}'s pending order (the customer requested it)" if op == "cancel"
                         else f"ship {u['name']}'s pending order")
        self.multi_total = round(sum(o["amount"] for _, _, o in self.jobs), 2)
        steps = "; ".join(f"({i + 1}) {p}" for i, p in enumerate(parts))
        speed = (" Use express shipping for gold-tier customers and standard shipping otherwise."
                 if any(op == "ship" for op, _, _ in self.jobs) else "")
        self.instruction = (f"Process these requests in order: {steps}.{speed} When all are done, answer with the "
                            "total amount of the orders you processed, rounded to 2 decimals.")

    def reset(self) -> str:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.effects: dict[str, Any] = {}
        self.failed = False
        self.revealed_user = False
        self.revealed_orders: set[str] = set()
        self.asked = False                 # clarify: the right question was asked ...
        self.acted_before_ask = False      # ... and nothing was changed before it
        self.n_asks = 0
        self.side_effects: list[tuple[str, dict[str, Any]]] = []
        obs = "You have access to the customer-operations tools." + (ASK_POLICY if self.ask_tool else "")
        self._seen: list[str] = [obs]      # every observation so far (information feasibility of the oracle sets)
        return obs

    def tools(self) -> list[dict[str, Any]]:
        return self._tools

    # ------------------------------------------------------------------ dynamics
    def _noise(self) -> str:
        if not self.noise_lines:
            return ""
        return "\n" + "\n".join(f"log: heartbeat ok #{self.rng.randint(1, 10**6)}" for _ in range(self.noise_lines))

    @staticmethod
    def _args_match(tool: str, args: dict[str, Any], gold: dict[str, Any]) -> bool:
        """Tolerant argument matching: strip, case-insensitive, refund amounts within +-0.01."""
        for k, v in gold.items():
            if k == "amount":
                try:
                    if abs(float(args.get("amount") or 0) - float(v)) > 0.01:
                        return False
                except (TypeError, ValueError):
                    return False
            elif str(args.get(k)).strip().lower() != str(v).strip().lower():
                return False
        return True

    def _write_is_gold(self, tool: str, args: dict[str, Any]) -> bool:
        """A state-changing call is acceptable iff it equals one of the gold calls of that tool (and, for
        ``clarify``, the right question has already been asked)."""
        if self.family == "clarify" and not self.asked:
            return False
        return any(t == tool and self._args_match(tool, args, a) for t, a in self._plan())

    def _missing_of(self, args: dict[str, Any]) -> str | None:
        m = str(args.get("missing") or "").strip().lower()
        if m in MISSING:
            return m
        q = f"{m} {args.get('question') or ''}".lower()   # robustness: infer the slot from free text
        for slot, pat in (("refund_method", r"refund|credit|payment"), ("shipping_speed", r"express|standard|speed"),
                          ("customer_name", r"\bname\b|who are you"), ("order_id", r"\border\b")):
            if re.search(pat, q):
                return slot
        return None

    def _ask(self, args: dict[str, Any]) -> StepResult:
        self.n_asks += 1
        if self.family == "clarify" and self._missing_of(args) == self.missing:
            if not self.asked and (any(self.effects.get(k) for k in ("cancelled", "refunds", "shipped"))
                                   or self.side_effects):
                self.acted_before_ask = True   # the world was already changed before the question
            self.asked = True
            return StepResult(f"Customer: {self.reply}")
        if self.family == "clarify":
            return StepResult("Customer: I think I already gave you everything you need on that point.")
        return StepResult("Customer: I have nothing to add -- please go ahead with the request as described.")

    def step(self, tool: str, args: dict[str, Any]) -> StepResult:
        res = self._step(tool, args)
        if hasattr(self, "_seen"):
            self._seen.append(res.observation)
        return res

    def _step(self, tool: str, args: dict[str, Any]) -> StepResult:
        near = NEAR_BY_NAME.get(tool) if tool in self._near_offered else None
        if self.strict and ((tool in self.STATE_CHANGING and not self._write_is_gold(tool, args))
                            or (near is not None and near.write)):
            self.calls.append((tool, dict(args)))
            self.failed = True
            return StepResult(f"Irreversible error: {tool}({args}) was applied incorrectly.", done=True)
        self.calls.append((tool, dict(args)))
        w = self.world
        if tool == "find_user":
            nm = str(args.get("name", "")).strip().lower()
            for u in w.users.values():
                if u["name"].lower() == nm:
                    if u["user_id"] == self.user["user_id"]:
                        self.revealed_user = True
                    return StepResult(json.dumps(u) + self._noise())
            return StepResult("No user found." + self._noise())
        if tool == "get_orders":
            uid, st = args.get("user_id"), args.get("status", "all")
            rows = [o for o in w.orders.values() if o["user_id"] == uid and (st == "all" or o["status"] == st)]
            if uid == self.user["user_id"]:
                self.revealed_orders |= {st}
            return StepResult(json.dumps(rows) + self._noise() if rows else "[]")
        if tool == "cancel_order":
            o = w.orders.get(str(args.get("order_id")))
            if not o or o["status"] != "pending":
                return StepResult("Error: order not found or not pending.")
            o["status"] = "cancelled"
            self.effects.setdefault("cancelled", []).append((o["order_id"], args.get("reason")))
            return StepResult(f"Order {o['order_id']} cancelled.")
        if tool == "refund":
            o = w.orders.get(str(args.get("order_id")))
            if not o or o["status"] != "delivered":
                return StepResult("Error: order not found or not delivered.")
            self.effects.setdefault("refunds", []).append((o["order_id"], args.get("amount"), args.get("method")))
            return StepResult(f"Refunded {args.get('amount')} via {args.get('method')}.")
        if tool == "check_stock":
            it = args.get("item")
            return StepResult(f"{it}: {w.stock.get(it, 0)} units" if it in w.stock else "Unknown item.")
        if tool == "ship_order":
            o = w.orders.get(str(args.get("order_id")))
            if not o or o["status"] != "pending":
                return StepResult("Error: order not found or not pending.")
            o["status"] = "shipped"
            self.effects.setdefault("shipped", []).append((o["order_id"], args.get("speed")))
            return StepResult(f"Order {o['order_id']} shipped ({args.get('speed')}).")
        if tool == "send_message":
            return StepResult("Message sent.")
        if tool == "ask_user" and self.ask_tool:
            return self._ask(args)
        if near is not None:
            return self._near_step(near, args)
        return StepResult(f"{tool}: no relevant result.")

    def _near_step(self, t: NearTool, args: dict[str, Any]) -> StepResult:
        """Plausible behaviour of the near-duplicate distractors. Read-only ones answer (some answers are
        tempting but wrong for the task, e.g. a single warehouse's stock); write ones apply a side effect."""
        w, a = self.world, args
        if t.write:
            self.side_effects.append((t.name, dict(a)))
            done = {"cancel_subscription": "Subscription cancelled.", "cancel_order_item": "Item line removed.",
                    "cancel_shipment": "Shipment recall requested.", "void_payment": "Payment authorisation voided.",
                    "refund_shipping_fee": "Shipping fee refunded.", "issue_store_credit": "Store credit issued.",
                    "partial_refund": "Partial refund issued.", "reverse_chargeback": "Chargeback response filed.",
                    "reserve_stock": "Units reserved.", "restock_item": "Supplier order placed.",
                    "ship_order_international": "International shipment created.",
                    "ship_replacement": "Replacement shipped.", "upgrade_shipping": "Shipping speed updated.",
                    "schedule_pickup": "Pickup scheduled.", "escalate_to_supervisor": "Case escalated."}
            return StepResult(done.get(t.name, "Done."))
        order = w.orders.get(str(a.get("order_id")))
        item = a.get("item")
        if t.name in ("find_customer_by_email", "find_user_by_phone"):
            return StepResult("No customer found with that contact detail." + self._noise())
        if t.name == "lookup_employee":
            return StepResult("No employee found." + self._noise())
        if t.name == "search_users":
            p = str(a.get("prefix", "")).strip().lower()
            names = sorted(u["name"] for u in w.users.values() if p and u["name"].split()[-1].lower().startswith(p))
            return StepResult(json.dumps(names) + self._noise())
        if t.name == "get_order":
            return StepResult((json.dumps(order) if order else "Order not found.") + self._noise())
        if t.name == "get_order_status":
            return StepResult(order["status"] if order else "Order not found.")
        if t.name == "list_orders_by_item":
            n = sum(1 for o in w.orders.values() if o["item"] == item and o["status"] == a.get("status"))
            return StepResult(f"{n} orders")
        if t.name == "get_orders_archive":
            return StepResult("[]")
        if t.name == "get_orders_summary":
            uid = a.get("user_id")
            counts = {s: sum(1 for o in w.orders.values() if o["user_id"] == uid and o["status"] == s)
                      for s in STATUSES}
            return StepResult(json.dumps(counts) if uid in w.users else "No such customer.")
        if t.name == "check_stock_warehouse":
            if item not in w.stock:
                return StepResult("Unknown item.")
            wh = str(a.get("warehouse"))
            share = self._warehouse_split(item)
            return StepResult(f"{item} @ {wh}: {share.get(wh, 0)} units")
        if t.name == "check_stock_incoming":
            return StepResult(f"{item}: {(w.stock.get(item, 0) * 7 + 13) % 60} units incoming"
                              if item in w.stock else "Unknown item.")
        if t.name == "get_item_price":
            return StepResult(f"{item}: ${19 + 37 * ITEMS.index(item)}.99" if item in ITEMS else "Unknown item.")
        if t.name == "get_shipping_quote":
            return StepResult(f"Quote: ${'14.90' if a.get('speed') == 'express' else '4.90'}" if order
                              else "Order not found.")
        if t.name in ("send_email", "notify_warehouse"):
            return StepResult("Message sent.")
        if t.name == "request_callback":
            return StepResult("Call-back scheduled.")
        return StepResult(f"{t.name}: no relevant result.")

    def _warehouse_split(self, item: str) -> dict[str, int]:
        """Deterministic split of an item's stock over the warehouses (sums to the company-wide stock)."""
        total = self.world.stock[item]
        r = random.Random(f"{self.seed}-{item}")
        cuts = sorted(r.randint(0, total) for _ in range(len(WAREHOUSES) - 1))
        bounds = [0, *cuts, total]
        return {wh: bounds[i + 1] - bounds[i] for i, wh in enumerate(WAREHOUSES)}

    # ------------------------------------------------------------------ oracle
    def _plan(self) -> list[tuple[str, dict[str, Any]]]:
        u, f = self.user, self.family
        find = ("find_user", {"name": u["name"]})
        if f == "cancel":
            return [find, ("get_orders", {"user_id": u["user_id"], "status": "pending"}),
                    ("cancel_order", {"order_id": self.pending_order["order_id"], "reason": "customer_request"}),
                    ("finish", {"answer": "done"})]
        if f == "refund":
            o = self.delivered_order
            return [find, ("get_orders", {"user_id": u["user_id"], "status": "delivered"}),
                    ("refund", {"order_id": o["order_id"], "amount": o["amount"], "method": "store_credit"}),
                    ("finish", {"answer": "done"})]
        if f == "stock":
            return [("check_stock", {"item": self.target_item}),
                    ("finish", {"answer": str(self.world.stock[self.target_item])})]
        if f == "ship":
            speed = "express" if u["tier"] == "gold" else "standard"
            return [find, ("get_orders", {"user_id": u["user_id"], "status": "pending"}),
                    ("ship_order", {"order_id": self.pending_order["order_id"], "speed": speed}),
                    ("finish", {"answer": "done"})]
        if f == "total":
            return [find, ("get_orders", {"user_id": u["user_id"], "status": "delivered"}),
                    ("finish", {"answer": f"{self.delivered_total:.2f}"})]
        if f == "clarify":
            ask = ("ask_user", {"missing": self.missing})
            pending = ("get_orders", {"user_id": u["user_id"], "status": "pending"})
            done = ("finish", {"answer": "done"})
            if self.variant == "order":    # the ambiguity only shows once the pending orders are listed
                return [find, pending, ask, ("cancel_order", {"order_id": self.pending_order["order_id"],
                                                              "reason": "customer_request"}), done]
            if self.variant == "method":   # known to be missing from the instruction: ask first
                o = self.delivered_order
                return [ask, find, ("get_orders", {"user_id": u["user_id"], "status": "delivered"}),
                        ("refund", {"order_id": o["order_id"], "amount": o["amount"], "method": self.method}), done]
            # speed / name: missing from the instruction itself -> ask first
            return [ask, find, pending, ("ship_order", {"order_id": self.pending_order["order_id"],
                                                        "speed": self.ship_speed}), done]
        plan: list[tuple[str, dict[str, Any]]] = []
        for op, user, order in self.jobs:
            plan += [("find_user", {"name": user["name"]}),
                     ("get_orders", {"user_id": user["user_id"], "status": "pending"}),
                     ("cancel_order", {"order_id": order["order_id"], "reason": "customer_request"}) if op == "cancel"
                     else ("ship_order", {"order_id": order["order_id"], "speed": _speed(user)})]
        return plan + [("finish", {"answer": f"{self.multi_total:.2f}"})]

    def _served(self, tool: str, a: dict[str, Any], args: dict[str, Any]) -> bool:
        """Did the executed call ``a`` of ``tool`` do what plan step ``args`` needs, *as ``_step`` served it*? Reads
        match the way ``_step`` answers them: ``find_user`` strips and lower-cases the name; ``get_orders`` compares
        ``user_id`` and ``status`` exactly (a padded or lower-cased id, or status "ALL", returned "[]"), and a
        status-filtered read is also satisfied by the information-equivalent ``get_orders(uid, "all")``;
        ``check_stock`` compares the item exactly. Writes keep the tolerant ``_args_match`` (a refund whose amount is
        the string "458.240" is done) but must have taken effect (``_step`` looks the order id up exactly, so a
        case-variant id was "Error: order not found" and changed nothing)."""
        if tool == "find_user":
            return str(a.get("name", "")).strip().lower() == str(args.get("name", "")).strip().lower()
        if tool == "get_orders":
            served = a.get("status", "all")
            return a.get("user_id") == args.get("user_id") and (served == args.get("status", "all") or served == "all")
        if tool == "check_stock":
            return a.get("item") == args.get("item")
        if not self._args_match(tool, a, args):
            return False
        eff = {"cancel_order": "cancelled", "refund": "refunds", "ship_order": "shipped"}.get(tool)
        if eff is None:
            return True
        return any(str(e[0]) == str(args.get("order_id")) for e in self.effects.get(eff, []))

    def _done_step(self, tool: str, args: dict[str, Any]) -> bool:
        """Has plan step (tool, args) been executed? A call counts only as ``_step`` actually served it
        (:meth:`_served`; round-3 review: tolerant read matching marked a ``get_orders`` with a lower-cased user id --
        answered "[]" -- as done, the rollout then reached ``finish`` without the amounts and labelled the
        recoverable state "doomed"). Writes keep tolerant argument matching (META_GOLD_SPEC §2.1, P0 oracle fixes)."""
        return any(t == tool and self._served(tool, a, args) for t, a in self.calls)

    def oracle_action(self) -> tuple[str, dict[str, Any]] | None:
        for tool, args in self._plan():
            if tool == "ask_user":
                if not getattr(self, "asked", False):
                    return tool, args
                continue
            if tool == "finish" or not self._done_step(tool, args):
                return tool, args
        return None

    def oracle_done(self) -> bool | None:
        nxt = self.oracle_action()
        return nxt is not None and nxt[0] == "finish"

    def oracle_accepts(self, tool: str, args: dict[str, Any]) -> bool | None:
        """Exact gold verdict on a *proposed* call in the current state (for guard decisions): is (tool, args)
        the oracle's canonical next action? Unlike a tool-name comparison, the arguments must match the gold
        ones (case-insensitive; refund amounts within 0.01; extra optional arguments such as
        ``ask_user.question`` are ignored); ``ask_user`` must ask for the missing slot; ``finish`` must contain
        the gold number when the family is judged on its answer (stock, total, multi)."""
        nxt = self.oracle_action()
        if nxt is None:
            return None
        gold_tool, gold_args = nxt
        if tool != gold_tool:
            return False
        if tool == "finish":
            return (self.family not in ("stock", "total", "multi")
                    or str(gold_args["answer"]) in str(args.get("answer", "")).replace(",", ""))
        if tool == "ask_user":
            return self._missing_of(args) == self.missing
        return self._args_match(tool, args, gold_args)

    # ------------------------------------------------------------------ meta-decision golds (META_GOLD_SPEC §2)
    def oracle_snapshot(self) -> "JevGymEnv":
        """An independent copy of the episode state; every gold computation steps copies, never the episode.
        Static parts (tool specs, customers, stock, the task) are shared; everything a step can change is copied --
        the RNG included, so stepping a copy never changes the real episode's heartbeat noise."""
        c = copy.copy(self)
        c.world = World(users=self.world.users, orders={k: dict(v) for k, v in self.world.orders.items()},
                        stock=self.world.stock)
        c.rng = random.Random()
        c.rng.setstate(self.rng.getstate())
        if hasattr(self, "calls"):
            c.calls = [(t, dict(a)) for t, a in self.calls]
            c.effects = {k: list(v) for k, v in self.effects.items()}
            c.revealed_orders = set(self.revealed_orders)
            c.side_effects = list(self.side_effects)
            c._seen = list(getattr(self, "_seen", []))
        return c

    def _gold_answer(self) -> str:
        return str(self._plan()[-1][1].get("answer", "done"))

    def _complete(self) -> bool:
        """complete(s): only the answer remains -- finishing now with the plan's answer succeeds AND the evidence the
        answer is computed from (the finish step's atoms, §2.4) is visible. The visibility clause is a documented
        clarification of META_GOLD_SPEC §2.2: without it the answer-only families (``total``, ``stock``) would be
        "complete" at step 0 (``evaluate`` checks only the answer there), so the oracle's own information-gathering
        steps would score as detours."""
        if getattr(self, "failed", False) or not self.evaluate(self._gold_answer())["success"]:
            return False
        return all(self._visible(k, v) for k, v in self._step_atoms(self.FINAL_TOOL, {}))

    @staticmethod
    def _rollout_reason(env: "JevGymEnv", budget: int) -> tuple[float, str | None]:
        """(D(s, B), why it is inf): ``budget_steps`` = the canonical plan needs more than B steps from here (the
        oracle-relative route: the canonical plan is an upper bound on the optimum, A6 / P0-8), ``plan_stuck`` = the
        canonical plan has no next action or only a finish that would fail, ``plan_fails`` = the canonical plan's own
        step ended the episode unsuccessfully; None when D is finite."""
        c = env.oracle_snapshot()
        for k in range(max(0, int(budget))):
            if c._complete():
                return k + 1, None
            a = c.oracle_action()
            if a is None or a[0] == c.FINAL_TOOL:
                return INF, "plan_stuck"
            r = c.step(*a)
            if r.done:
                return (k + 1, None) if c.evaluate(None)["success"] else (INF, "plan_fails")
        return INF, "budget_steps"

    @staticmethod
    def _rollout(env: "JevGymEnv", budget: int) -> float:
        """D(s, B): the early-exit canonical cost-to-go (steps incl. the finish; inf = not recoverable within B).
        The early exit returns as soon as only the answer remains, so a state reached by a valid shortcut (e.g.
        ask -> cancel in clarify/order) is not charged for plan steps it no longer needs."""
        return JevGymEnv._rollout_reason(env, budget)[0]

    def oracle_cost_to_go(self, budget: int = 30) -> float:
        return self._rollout(self, budget)

    def oracle_world_complete(self) -> bool:
        return self._complete()

    def oracle_finish_ok(self, answer: Any) -> bool:
        """Stopping now with this answer succeeds (read-only ``evaluate``)."""
        return bool(self.evaluate(None if answer is None else str(answer))["success"])

    def oracle_verdict(self, tool: str, args: dict[str, Any] | None, budget: int = 30,
                       d_before: float | None = None) -> dict[str, Any]:
        """verdict(s, a) of a proposed call (META_GOLD_SPEC §2.3): ``progress`` (the cost-to-go drops), ``detour``
        (it does not: read-only distractor, wrong ask slot, repeated call, unknown tool), ``fatal`` (success becomes
        impossible: irreversible error, side effect, acting before the required ask, budget exhaustion), and for
        ``finish(x)`` ``progress`` / ``wrong_answer`` (only the answer remained) / ``premature``. A doomed state
        (D = inf) has no verdict (``status: doomed``). ``canonical`` is the legacy ``oracle_accepts``.

        A ``fatal`` verdict carries its kind (A6 / P0-8): ``fatal_kind = "terminal"`` when the proposed step itself
        ends the episode unsuccessfully (proven in the strict world: ``r.done and not success``), ``"budget"`` when
        success is only unreachable for the *canonical* plan within the remaining budget (D(s') = inf by rollout:
        oracle-relative, since the canonical plan is an upper bound on the optimum); ``fatal_detail`` refines the
        budget kind (``budget_steps`` / ``plan_stuck`` / ``plan_fails``, see :meth:`_rollout_reason`)."""
        args = args if isinstance(args, dict) else {}
        d0 = self._rollout(self, budget) if d_before is None else d_before
        out: dict[str, Any] = {"verdict": None, "status": "doomed", "canonical": None,
                               "d_before": None if d0 == INF else int(d0), "d_after": None}
        try:
            out["canonical"] = self.oracle_accepts(tool, args)
        except Exception:  # noqa: BLE001 - a legacy label never blocks the verdict
            pass
        if d0 == INF:
            return out
        if tool == self.FINAL_TOOL:
            if self.oracle_finish_ok(args.get("answer", "")):
                v = "progress"
            else:
                v = "wrong_answer" if self._complete() else "premature"
            out.update(verdict=v, status="ok")
            return out
        c = self.oracle_snapshot()
        kind = detail = None
        if tool in self._tool_names:
            r = c.step(tool, args)
            if r.done and not c.evaluate(None)["success"]:
                d1, kind, detail = INF, "terminal", "terminal"
            else:
                d1, detail = self._rollout_reason(c, budget - 1)
        else:  # unknown tool: the harness answers "Error: unknown tool"; the world is unchanged, one step is spent
            d1, detail = self._rollout_reason(c, budget - 1)
        v = "fatal" if d1 == INF else ("progress" if d1 < d0 else "detour")
        out.update(verdict=v, status="ok", d_after=None if d1 == INF else int(d1))
        if v == "fatal":
            out.update(fatal_kind=kind or "budget", fatal_detail=detail)
        return out

    # ---- evidence atoms (dataflow map, §2.4)
    def _undone(self) -> list[tuple[str, dict[str, Any]]]:
        out = []
        for tool, args in self._plan():
            if tool == "ask_user":
                if not getattr(self, "asked", False):
                    out.append((tool, args))
                continue
            if tool == self.FINAL_TOOL or not self._done_step(tool, args):
                out.append((tool, args))
        return out

    def _step_atoms(self, tool: str, args: dict[str, Any]) -> list[tuple[str, Any]]:
        """The evidence atoms plan step (tool, args) consumes: values the agent must have seen to make the call."""
        f, var = self.family, self.variant
        atoms: list[tuple[str, Any]] = []
        if tool == "find_user":
            if f == "clarify" and var == "name":          # the name is not in the instruction: the reply supplies it
                atoms += [("name", args.get("name")), ("reply", self.reply)]
        elif tool == "get_orders":
            atoms.append(("user_id", args.get("user_id")))
        elif tool in ("cancel_order", "ship_order", "refund"):
            atoms.append(("order_id", args.get("order_id")))
            if tool == "refund":
                atoms.append(("amount", args.get("amount")))
            if tool == "ship_order":
                if f == "clarify" and var == "speed":
                    atoms.append(("reply", self.reply))
                else:
                    o = self.world.orders.get(str(args.get("order_id")))
                    u = self.world.users.get(o["user_id"]) if o else None
                    if u:
                        atoms.append(("tier", u["tier"]))
            if f == "clarify" and var in ("order", "method"):
                atoms.append(("reply", self.reply))
        elif tool == self.FINAL_TOOL:
            if f == "multi":
                atoms += [("amount", o["amount"]) for _, _, o in self.jobs]
            elif f == "total":
                atoms += [("amount", o["amount"]) for o in self.world.orders.values()
                          if o["user_id"] == self.user["user_id"] and o["status"] == "delivered"]
            elif f == "stock":
                atoms.append(("count", self.world.stock[self.target_item]))
        return [(k, v) for k, v in atoms if v is not None and v != ""]

    @staticmethod
    def atom_in(kind: str, value: Any, text: str) -> bool:
        """Does ``text`` carry the atom? Identifiers and tiers: token-boundary literal match (the ``#123456``
        heartbeat noise never matches); names: the same, case-insensitive; amounts and counts: any number token
        within +-0.005; the clarify reply: substring."""
        if value is None or not text:
            return False
        if kind in ("amount", "count"):
            try:
                x = float(value)
            except (TypeError, ValueError):
                return False
            return any(abs(float(m.group(1)) - x) <= 0.005 for m in _NUM_TOKEN.finditer(text))
        if kind == "reply":
            return str(value) in text
        flags = re.I if kind in ("name", "tier") else 0
        return re.search(_ID_L + re.escape(str(value)) + _ID_R, text, flags) is not None

    def _visible(self, kind: str, value: Any) -> bool:
        return self.atom_in(kind, value, "\n".join([self.instruction, *getattr(self, "_seen", [])]))

    def oracle_need(self) -> list[dict[str, Any]]:
        """need(s): the atoms the undone plan steps still consume, with the distances j of their consumers
        (1 = the next undone step): ``d_first`` = min j, ``d_last`` = max j."""
        atoms: dict[tuple[str, str], dict[str, Any]] = {}
        for j, (tool, args) in enumerate(self._undone(), 1):
            for kind, val in self._step_atoms(tool, args):
                a = atoms.setdefault((kind, str(val)), {"kind": kind, "value": val, "d_first": j, "d_last": j})
                a["d_last"] = max(a["d_last"], j)
        return list(atoms.values())

    def oracle_acceptable_tool_sets(self, budget: int = 30) -> tuple[list[str], list[str]]:
        """(AT(s), unfiltered): tools of undone plan steps whose verdict is progress -- AT keeps only steps whose
        every input atom is visible in the instruction or an observation so far (a write tool is not "acceptable"
        merely because the oracle knows its arguments) -- plus ``finish`` when only the answer remains."""
        d0 = self._rollout(self, budget)
        if d0 == INF:
            return [], []
        feas: list[str] = []
        unf: list[str] = []
        for tool, args in self._undone():
            if tool == self.FINAL_TOOL or tool in feas:      # feas is a subset of unf
                continue
            visible = all(self._visible(k, v) for k, v in self._step_atoms(tool, args))
            if tool in unf and not visible:
                continue
            if self.oracle_verdict(tool, args, budget, d_before=d0)["verdict"] != "progress":
                continue
            if tool not in unf:
                unf.append(tool)
            if visible and tool not in feas:
                feas.append(tool)
        if self._complete():
            feas.append(self.FINAL_TOOL)
            unf.append(self.FINAL_TOOL)
        return feas, unf

    def oracle_acceptable_tools(self, budget: int = 30, feasible: bool = True) -> list[str]:
        f, u = self.oracle_acceptable_tool_sets(budget)
        return f if feasible else u

    def oracle_mem_gold(self, text: str, known: Sequence[str] = ()) -> dict[str, Any]:
        """memory_write gold of an offered text, on the env AFTER the step (META_GOLD_SPEC §3.6): ``mem_need`` = the
        text carries an atom the remaining plan consumes; ``far`` = such an atom must survive >= 2 intervening oracle
        steps (d_last >= 3); ``new`` = a matched atom is in neither the instruction nor ``known`` (the store)."""
        matched = [a for a in self.oracle_need() if self.atom_in(a["kind"], a["value"], text)]
        known_text = "\n".join([self.instruction, *known])
        return {"mem_need": bool(matched), "far": any(a["d_last"] >= 3 for a in matched),
                "new": any(not self.atom_in(a["kind"], a["value"], known_text) for a in matched),
                "atoms": [dict(a) for a in matched]}

    def oracle_memory_read_gold(self, texts: Sequence[str]) -> dict[str, list[int]]:
        """memory_read gold (§3.7): indices of the texts holding an input of the next undone step (``strict``) or
        of any remaining step (``lenient``)."""
        need = self.oracle_need()
        nxt = [a for a in need if a["d_first"] == 1]
        return {"strict": [i for i, t in enumerate(texts) if any(self.atom_in(a["kind"], a["value"], t) for a in nxt)],
                "lenient": [i for i, t in enumerate(texts)
                            if any(self.atom_in(a["kind"], a["value"], t) for a in need)]}

    def evaluate(self, final_answer: str | None) -> dict[str, Any]:
        f, u = self.family, self.user
        ok = False
        v2 = self.ask_tool or self.distractor_style != "random" or f in self.V2_FAMILIES
        extra: dict[str, Any] = ({"side_effects": len(getattr(self, "side_effects", [])),
                                  "n_asks": getattr(self, "n_asks", 0)} if v2 else {})
        if f == "clarify":
            extra.update({"asked": bool(getattr(self, "asked", False)),
                          "acted_before_ask": bool(getattr(self, "acted_before_ask", False))})
        if getattr(self, "failed", False):
            return {"success": False, "score": 0.0, "n_calls": len(self.calls), "optimal_calls": len(self._plan()) - 1,
                    "irreversible_error": True, **extra}
        if f == "cancel":
            ok = self.effects.get("cancelled") == [(self.pending_order["order_id"], "customer_request")]
        elif f == "refund":
            ok = self._refund_ok("store_credit")
        elif f == "stock":
            ok = final_answer is not None and str(self.world.stock[self.target_item]) in final_answer
        elif f == "ship":
            speed = "express" if u["tier"] == "gold" else "standard"
            ok = self.effects.get("shipped") == [(self.pending_order["order_id"], speed)]
        elif f == "total":
            ok = final_answer is not None and f"{self.delivered_total:.2f}" in final_answer.replace(",", "")
        elif f == "clarify":
            if self.variant == "order":
                ok = self.effects.get("cancelled") == [(self.pending_order["order_id"], "customer_request")]
            elif self.variant == "method":
                ok = self._refund_ok(self.method)
            else:
                ok = self.effects.get("shipped") == [(self.pending_order["order_id"], self.ship_speed)]
            expected = {"order": "cancelled", "method": "refunds"}.get(self.variant, "shipped")
            ok = (ok and self.asked and not self.acted_before_ask
                  and not any(self.effects.get(k) for k in ("cancelled", "refunds", "shipped") if k != expected))
        else:
            gold_c = sorted((o["order_id"], "customer_request") for op, _, o in self.jobs if op == "cancel")
            gold_s = sorted((o["order_id"], _speed(us)) for op, us, o in self.jobs if op == "ship")
            ok = (sorted(self.effects.get("cancelled") or []) == gold_c
                  and sorted(self.effects.get("shipped") or []) == gold_s and not self.effects.get("refunds")
                  and final_answer is not None and f"{self.multi_total:.2f}" in final_answer.replace(",", ""))
        ok = ok and not getattr(self, "side_effects", [])
        return {"success": bool(ok), "score": float(ok), "n_calls": len(self.calls),
                "optimal_calls": len(self._plan()) - 1, **extra}

    def _refund_ok(self, method: str) -> bool:
        r = self.effects.get("refunds") or []
        o = self.delivered_order
        try:
            return (len(r) == 1 and r[0][0] == o["order_id"] and abs(float(r[0][1] or 0) - o["amount"]) < 0.01
                    and r[0][2] == method)
        except (TypeError, ValueError):
            return False


ALL_FAMILIES = JevGymEnv.FAMILIES + JevGymEnv.V2_FAMILIES
FAMILY_SETS: dict[str, list[str]] = {"v1": JevGymEnv.FAMILIES, "v2": JevGymEnv.V2_FAMILIES, "all": ALL_FAMILIES}
ENV_KWARGS = ("family", "n_distractors", "noise_lines", "strict", "distractor_style", "families", "ask_tool",
              "multi_customers")


def make_tasks(n: int = 50, n_distractors: int = 10, noise_lines: int = 0, seed0: int = 0,
               strict: bool = False, **kw: Any) -> list[JevGymEnv]:
    """``n`` tasks with seeds ``seed0 .. seed0 + n - 1``; ``kw`` are further :class:`JevGymEnv` arguments
    (``family``, ``families``, ``distractor_style``, ``ask_tool``, ``multi_customers``)."""
    return [JevGymEnv(seed=seed0 + i, n_distractors=n_distractors, noise_lines=noise_lines, strict=strict, **kw)
            for i in range(n)]


def task_factories(n: int = 50, seed0: int = 0, **kw: Any) -> list[Callable[[], JevGymEnv]]:
    """Zero-argument factories (fresh env per run) -- what ``load_suite("jevgym", ...)`` returns."""
    unknown = sorted(set(kw) - set(ENV_KWARGS))
    if unknown:
        raise ValueError(f"unknown jevgym suite option(s) {unknown}; valid: n, seed0, {', '.join(ENV_KWARGS)}")
    JevGymEnv(seed=seed0, **kw)  # validate the options once, eagerly
    return [(lambda i=i: JevGymEnv(seed=seed0 + i, **kw)) for i in range(n)]
