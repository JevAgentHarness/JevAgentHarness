"""DecisionGym-Shift (exploratory row E21, docs/PREREG_A7_STAGE2.md Deviations 2026-10-08 "E21: a shifted DecisionGym
split"): a harder, shifted split of the JevGym generator for re-measuring decision models off their training
distribution. Three registered axes, each switchable, all on by default:

  surface      every core tool and the ask tool get a consistent alias (``ALIASES``: find_user -> lookup_customer, ...)
               with a paraphrased description; the request texts, the policy line of the first observation and the
               customer's replies use new paraphrased templates. The env maps every alias to its canonical tool
               internally, so the answer key is the generator's, expressed in the names the agent sees (a canonical
               core name is not offered and is an unknown tool to the oracle, as to the harness).
  distraction  twice the distractors: the Stage-2 set (near-duplicates of the core tools plus unrelated ones) plus
               ``SHIFT_NEAR`` (near-duplicates of the alias names: functional, read-only ones answer, state-changing
               ones are side effects that fail the episode in strict mode) and new unrelated tools; twice the heartbeat
               noise lines per observation.
  structure    a schedule of task kinds (``KINDS``, by seed mod 10) with more must-ask states and three plan shapes the
               training generator never produces: ``clarify.ask_twice`` (two details missing -- who the customer is and
               the shipping speed -- so the agent must ask twice), ``clarify.ask_after_lookup`` (refund one of several
               delivered orders: the request becomes ambiguous only after the lookup) and ``notify`` (ship, then
               message the customer in the support chat: ``send_message`` is never on a training plan). The other
               kinds are the generator's own clarify variants and ``multi`` (3 customers).

Composition: :class:`ShiftCore` (a :class:`~.jevgym.JevGymEnv` subclass, canonical tool names inside) holds the world,
the kinds and the oracle; :class:`JevGymShiftEnv` (the Env the harness sees) translates tool names at the boundary
(``tools``, ``step``, every ``oracle_*``). The frozen generator ``jevgym.py`` (sha256-locked, A6.16 (l)) and
``jevgym_hard.py`` are unchanged; with every axis off, a task is the generator's own Stage-2 task of that seed (same
instruction, plan and tool set), apart from the task id. ``load_suite("jevgym", ...)`` builds this class when the suite
block carries ``shift`` (``true`` or a mapping of the three flags). Gold labels never leave the oracle methods.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..env import Env, StepResult
from .jevgym import (ASK_POLICY, ENV_KWARGS as BASE_ENV_KWARGS, ITEMS, METHODS, NEAR_BY_NAME, REASONS, SPEEDS,
                     WAREHOUSES, JevGymEnv, _fn, _speed, make_distractors)

AXES = ("surface", "distraction", "structure")
DISTRACTION_FACTOR = 2            # distractors and noise lines per observation, relative to the suite block
ASK = "ask_user"

# ------------------------------------------------------------------------------------------------ (a) surface
# canonical name -> (alias, paraphrased description, {param: paraphrased description})
ALIASES: dict[str, tuple[str, str, dict[str, str]]] = {
    "find_user": ("lookup_customer", "Find a customer's record from their full name; gives back the user_id and the "
                  "loyalty tier.", {"name": "the customer's first and last name"}),
    "get_orders": ("list_customer_orders", "Return the orders on a customer's account that have a given status.", {}),
    "cancel_order": ("cancel_purchase", "Call off one order so that it is never fulfilled.", {}),
    "refund": ("reimburse_order", "Pay an order's amount back to the customer.", {}),
    "check_stock": ("inventory_count", "How many units of an item are available right now.", {}),
    "ship_order": ("dispatch_order", "Send a pending order out to the customer at the chosen delivery speed.", {}),
    "send_message": ("message_customer", "Write a free-text note to a customer in the support chat.", {}),
    ASK: ("request_clarification", "Put a question to the customer and wait for their answer. Only for when a detail "
          "you need in order to act is missing or unclear.",
          {"missing": "the detail you still need", "question": "what to ask the customer"}),
}

# paraphrased templates (surface on); the generator's own wording is used when the axis is off
POLICY_SHIFT = (" Rule: do not guess. If something you need in order to act is missing or unclear (which customer, "
                "which order, the refund method, the delivery speed), ask the customer with {ask} before you change "
                "anything.")
OPENING = {False: "You have access to the customer-operations tools.",
           True: "You can use the customer-service tools listed below."}
REPEAT_REPLY = {False: "Customer: I think I already gave you everything you need on that point.",
                True: "Customer: I believe you already have everything you need on that."}
NOTHING_REPLY = {False: "Customer: I have nothing to add -- please go ahead with the request as described.",
                 True: "Customer: Nothing more from my side -- please just do what I asked."}


def _name_reply(nm: str, surface: bool) -> str:
    return f"Sorry about that, I'm {nm}." if surface else f"Oh, sorry -- my name is {nm}."


def _speed_reply(speed: str, surface: bool) -> str:
    return f"I'd like {speed} delivery, please." if surface else f"{speed.capitalize()} shipping, please."


def _order_reply(order: Mapping[str, Any], verb: str, surface: bool) -> str:
    if surface:
        return f"Apologies, I meant order {order['order_id']}, the {order['item']}. Please {verb} that one."
    return f"Sorry for the confusion -- please {verb} order {order['order_id']} (the {order['item']})."


# ------------------------------------------------------------------------------------------------ (b) distraction
_S = {"type": "string"}
_ORDER = {"type": "string", "description": "order id, e.g. ORD-1234"}
_USER = {"type": "string", "description": "customer user_id"}


@dataclass(frozen=True)
class ShiftNear:
    """A near-duplicate of an alias name (it shadows canonical tool ``shadows``); ``write`` = a harmful side effect."""
    name: str
    shadows: str
    desc: str
    props: dict[str, Any]
    required: tuple[str, ...]
    write: bool = False

    def spec(self) -> dict[str, Any]:
        return _fn(self.name, self.desc, self.props, list(self.required))


SHIFT_NEAR: list[ShiftNear] = [
    ShiftNear("lookup_customer_by_email", "find_user", "Find a customer's record from their e-mail address; gives back "
              "the user_id and the loyalty tier.", {"email": {"type": "string", "description": "e-mail address"}},
              ("email",)),
    ShiftNear("list_customer_orders_archived", "get_orders", "Return the orders a customer placed more than a year ago "
              "(archived orders only).", {"user_id": _USER, "year": {"type": "integer", "minimum": 2015,
                                                                     "maximum": 2025}}, ("user_id", "year")),
    ShiftNear("cancel_purchase_line", "cancel_order", "Remove a single item line from a purchase; the rest of the order "
              "goes ahead.", {"order_id": _ORDER, "item": {"type": "string", "enum": ITEMS},
                              "reason": {"type": "string", "enum": REASONS}}, ("order_id", "item", "reason"), write=True),
    ShiftNear("reimburse_shipping", "refund", "Pay back only the delivery fee of an order, not its amount.",
              {"order_id": _ORDER, "method": {"type": "string", "enum": METHODS}}, ("order_id", "method"), write=True),
    ShiftNear("inventory_count_by_site", "check_stock", "Units of an item held at ONE warehouse (the total is the sum "
              "over all sites).", {"item": {"type": "string", "enum": ITEMS},
                                   "warehouse": {"type": "string", "enum": WAREHOUSES}}, ("item", "warehouse")),
    ShiftNear("dispatch_order_abroad", "ship_order", "Send a pending order to an address in another country (customs "
              "forms and duties).", {"order_id": _ORDER, "speed": {"type": "string", "enum": SPEEDS}, "country": _S},
              ("order_id", "speed", "country"), write=True),
    ShiftNear("message_team", "send_message", "Post an internal note about an order to the warehouse team.",
              {"order_id": _ORDER, "text": _S}, ("order_id", "text")),
    ShiftNear("request_clarification_later", ASK, "Book a call with the customer for a later day (no answer now).",
              {"user_id": _USER, "date": {"type": "string", "description": "YYYY-MM-DD"}}, ("user_id", "date")),
    ShiftNear("lookup_customer_by_postcode", "find_user", "List the customers registered at a postcode (display names "
              "only, no ids).", {"postcode": {"type": "string", "description": "postal code"}}, ("postcode",)),
    ShiftNear("list_customer_returns", "get_orders", "Return the return requests on a customer's account (not its "
              "orders).", {"user_id": _USER}, ("user_id",)),
    ShiftNear("cancel_purchase_plan", "cancel_order", "Stop a customer's recurring purchase plan (not a single order).",
              {"user_id": _USER, "reason": {"type": "string", "enum": REASONS}}, ("user_id", "reason"), write=True),
    ShiftNear("reimburse_partial", "refund", "Pay back part of an order's amount for a damaged item; needs a damage "
              "report id.", {"order_id": _ORDER, "amount": {"type": "number"}, "damage_report_id": _S},
              ("order_id", "amount", "damage_report_id"), write=True),
    ShiftNear("inventory_count_reserved", "check_stock", "Units of an item already reserved for open orders.",
              {"item": {"type": "string", "enum": ITEMS}}, ("item",)),
    ShiftNear("dispatch_replacement", "ship_order", "Send a free replacement for a damaged order that was already "
              "delivered.", {"order_id": _ORDER, "speed": {"type": "string", "enum": SPEEDS}}, ("order_id", "speed"),
              write=True),
    ShiftNear("message_supplier", "send_message", "E-mail a supplier about a product line.",
              {"supplier": _S, "text": _S}, ("supplier", "text")),
    ShiftNear("request_review", ASK, "Pass the case to a human supervisor (fraud, legal or abuse cases only).",
              {"reason": {"type": "string", "enum": ["fraud", "legal", "abuse"]}}, ("reason",), write=True),
    ShiftNear("lookup_supplier", "find_user", "Find a supplier's record from its company name; gives back the "
              "supplier_id.", {"company": {"type": "string", "description": "company name"}}, ("company",)),
    ShiftNear("count_customer_orders", "get_orders", "Number of orders on a customer's account per status (counts "
              "only, no ids or amounts).", {"user_id": _USER}, ("user_id",)),
    ShiftNear("cancel_purchase_delivery", "cancel_order", "Call back a delivery that is already on its way (orders with "
              "status shipped).", {"order_id": _ORDER, "reason": {"type": "string", "enum": REASONS}},
              ("order_id", "reason"), write=True),
    ShiftNear("reimburse_voucher", "refund", "Give the customer a goodwill voucher; not linked to an order and not a "
              "refund.", {"user_id": _USER, "amount": {"type": "number"}}, ("user_id", "amount"), write=True),
    ShiftNear("inventory_adjust", "check_stock", "Correct the recorded stock level of an item.",
              {"item": {"type": "string", "enum": ITEMS}, "units": {"type": "integer"}}, ("item", "units"), write=True),
    ShiftNear("dispatch_quote", "ship_order", "Price of sending an order at a given speed (does not send it).",
              {"order_id": _ORDER, "speed": {"type": "string", "enum": SPEEDS}}, ("order_id", "speed")),
]
SHIFT_NEAR_BY_NAME = {t.name: t for t in SHIFT_NEAR}
SHIFT_RANDOM = [
    ("get_store_hours", "Opening hours of a retail store."), ("list_job_openings", "Open positions at the company."),
    ("get_share_price", "Share price of a listed company."), ("book_meeting_room", "Reserve a meeting room."),
    ("get_news_headlines", "Latest company news headlines."), ("submit_expense", "File an expense report."),
    ("get_cafeteria_menu", "Today's cafeteria menu."), ("check_vpn_status", "VPN connection status."),
    ("list_parking_spots", "Free spots in the staff car park."), ("get_air_quality", "Air-quality index for a city."),
    ("order_office_supplies", "Order pens, paper and other office supplies."),
    ("get_training_courses", "Internal training courses."), ("request_laptop", "Request a new work laptop."),
    ("get_org_chart", "The organisation chart of a department."), ("list_webinars", "Upcoming public webinars."),
    ("get_carbon_report", "The company's quarterly carbon report."), ("search_patents", "Search the patent database."),
    ("get_tax_forms", "Download payroll tax forms."), ("check_build_status", "Status of a software build."),
    ("get_server_metrics", "CPU and memory metrics of a server."), ("rotate_api_key", "Rotate a service API key."),
    ("list_feature_flags", "Feature flags of a web service."), ("get_sla_report", "Monthly service-level report."),
    ("summarize_survey", "Summarise an employee survey."), ("get_brand_guidelines", "The brand style guide."),
    ("plan_trip", "Plan a business trip itinerary."), ("get_visa_requirements", "Visa rules for a country."),
    ("list_open_incidents", "Open IT incidents."), ("get_marketing_budget", "Marketing budget of a quarter."),
    ("schedule_backup", "Schedule a database backup."), ("check_badge_access", "Check an employee's building access."),
    ("list_printers", "Printers on an office floor."), ("get_time_zone", "Time zone of a city."),
    ("get_product_reviews", "Public reviews of a product."), ("translate_document", "Translate an uploaded document."),
    ("get_parking_rates", "Parking rates near an office."), ("list_suppliers_onboarding", "Suppliers being onboarded."),
    ("get_weather_forecast", "Ten-day weather forecast for a city."),
    ("compute_commission", "Sales commission of an employee."), ("list_meeting_notes", "Notes of past meetings."),
]


def shift_distractors(n_base: int, style: str) -> list[dict[str, Any]]:
    """The distraction axis: the suite block's own ``n_base`` distractors (the Stage-2 set for ``near``), then the
    alias near-duplicates and new unrelated tools, ``DISTRACTION_FACTOR * n_base`` in all."""
    extra = (DISTRACTION_FACTOR - 1) * int(n_base)
    near = [t.spec() for t in SHIFT_NEAR[:extra]]
    rand = []
    for i in range(extra - len(near)):
        name, desc = SHIFT_RANDOM[i % len(SHIFT_RANDOM)]
        if i >= len(SHIFT_RANDOM):
            name = f"{name}_{i // len(SHIFT_RANDOM) + 1}"
        rand.append(_fn(name, desc, {"query": {"type": "string"}}, ["query"]))
    return make_distractors(n_base, style) + near + rand


# ------------------------------------------------------------------------------------------------ (c) structure
# seed mod 10 -> (family, variant); multi 30 %, the generator's clarify variants 40 %, the new kinds 30 %
KINDS: tuple[tuple[str, str | None], ...] = (
    ("clarify", "order"), ("multi", None), ("clarify", "method"), ("multi", None), ("clarify", "speed"),
    ("notify", None), ("clarify", "name"), ("multi", None), ("clarify", "ask_twice"), ("clarify", "ask_after_lookup"))
NEW_KINDS = (("notify", None), ("clarify", "ask_twice"), ("clarify", "ask_after_lookup"))


def kind_of(seed: int) -> tuple[str, str | None]:
    return KINDS[int(seed) % len(KINDS)]


def kind_label(family: str, variant: str | None) -> str:
    return family + (f".{variant}" if variant else "")


ENV_KWARGS = BASE_ENV_KWARGS + AXES


class ShiftCore(JevGymEnv):
    """The shifted task with canonical tool names inside (the harness never sees this object directly)."""

    def __init__(self, seed: int, *args: Any, surface: bool = True, distraction: bool = True, structure: bool = True,
                 **kw: Any):
        self.surface, self.distraction, self.structure = bool(surface), bool(distraction), bool(structure)
        self.kind = kind_of(seed) if self.structure else None
        self.asked_slots: set[str] = set()
        base_n = int(kw.get("n_distractors", 10))
        self.base_noise = int(kw.get("noise_lines", 0))
        if self.distraction:
            kw["noise_lines"] = DISTRACTION_FACTOR * self.base_noise
        self.base_n_distractors = base_n
        super().__init__(seed, *args, **kw)
        if self.distraction:
            self.n_distractors = DISTRACTION_FACTOR * base_n
        axes = "".join(c for c, on in zip("sdt", (self.surface, self.distraction, self.structure)) if on) or "none"
        self.task_id = (f"jevgym-shift-{axes}-{kind_label(self.family, self.variant)}-{seed}-d{self.n_distractors}"
                        + (f"-{self.distractor_style}" if self.distractor_style != "random" else "")
                        + ("-strict" if self.strict else ""))

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        if self.kind is not None:
            self.family = self.kind[0]          # read by the parent's _build ("notify" goes to _build_multi)
        super()._build()
        if self.kind is not None and self.kind[1] is None:
            self.variant = None
        if self.variant in ("ask_twice",):
            self.gold.update({"variant": self.variant, "missing": "+".join(self.slots())})
        if self.distraction:                     # (b): the tool list with twice the distractors, reshuffled
            tools = [t for t in self._tools if t["function"]["name"] in self._core_names()]
            tools += shift_distractors(self.base_n_distractors, self.distractor_style)
            random.Random(f"jevgym-shift-tools:{self.seed}").shuffle(tools)
            self._tools = tools
            self._tool_names = frozenset(t["function"]["name"] for t in self._tools)
            self._near_offered = self._tool_names & set(NEAR_BY_NAME)
        self._shift_near_offered = self._tool_names & set(SHIFT_NEAR_BY_NAME)

    def _core_names(self) -> set[str]:
        return {"find_user", "get_orders", "cancel_order", "refund", "check_stock", "ship_order", "send_message", ASK}

    def _build_clarify(self, rng: random.Random) -> None:
        var = self.kind[1] if self.kind is not None else None
        if var == "ask_twice":
            self._build_ask_twice(rng)
            return
        if var == "ask_after_lookup":
            self._build_ask_after_lookup(rng)
            return
        if var is not None:                      # the generator's variant, fixed by the kind schedule
            self.CLARIFY_VARIANTS = [var]        # instance attribute: the parent picks variants[(seed // rr) % 1]
        super()._build_clarify(rng)
        if self.surface:
            self._paraphrase_clarify()

    def _paraphrase_clarify(self) -> None:
        nm, v = self.user["name"], self.variant
        if v == "order":
            self.reply = _order_reply(self.pending_order, "cancel", True)
            self.instruction = f"{nm} has asked for their pending order to be cancelled. Please take care of it."
        elif v == "method":
            self.reply = ("Store credit works for me, thanks." if self.method == "store_credit"
                          else "I'd like it back on my original payment method, please.")
            self.instruction = (f"{nm} should get back the full amount of their latest delivered order (the one with "
                                "the highest order id).")
        elif v == "speed":
            self.reply = _speed_reply(self.ship_speed, True)
            self.instruction = f"Please send out {nm}'s pending order."
        elif v == "name":
            self.reply = _name_reply(nm, True)
            self.instruction = ("Someone contacted us to have their pending order sent out (express for gold-tier "
                                "customers, standard for everyone else), but they did not give their name.")

    def _build_ask_twice(self, rng: random.Random) -> None:
        """Two details missing (who the customer is and the shipping speed): the agent must ask twice."""
        self.variant = "ask_twice"
        self.ship_speed = rng.choice(SPEEDS)
        self.missing = "customer_name"           # the first slot (the parent's single-slot fields stay meaningful)
        self.replies = {"customer_name": _name_reply(self.user["name"], self.surface),
                        "shipping_speed": _speed_reply(self.ship_speed, self.surface)}
        self.reply = self.replies["customer_name"]
        self.instruction = ("Someone contacted us to have their pending order sent out, but they gave neither their "
                            "name nor the delivery speed they want." if self.surface else
                            "A customer wrote in asking us to ship their pending order, but did not say who they are "
                            "or which shipping speed they want.")

    def _build_ask_after_lookup(self, rng: random.Random) -> None:
        """Refund one of several delivered orders: the request is ambiguous only once the orders are listed."""
        u, nm = self.user, self.user["name"]
        self.variant = "ask_after_lookup"
        for _ in range(rng.randint(1, 2)):
            oid = self._new_order_id(rng)
            self.world.orders[oid] = {"order_id": oid, "user_id": u["user_id"], "item": rng.choice(ITEMS),
                                      "status": "delivered", "amount": round(rng.uniform(15, 900), 2)}
        deliv = sorted((o for o in self.world.orders.values()
                        if o["user_id"] == u["user_id"] and o["status"] == "delivered"), key=lambda o: o["order_id"])
        self.delivered_order = rng.choice(deliv)
        self.method = "store_credit"
        self.missing = "order_id"
        self.reply = _order_reply(self.delivered_order, "refund", self.surface)
        self.instruction = (f"{nm} would like their delivered order refunded in full as store credit." if self.surface
                            else f"Refund the full amount of {nm}'s delivered order as store credit.")

    def _build_multi(self, rng: random.Random) -> None:
        if self.family == "notify":
            self._build_notify()
            return
        super()._build_multi(rng)
        if not self.surface:
            return
        parts = [f"cancel {u['name']}'s pending order (at the customer's request)" if op == "cancel"
                 else f"send out {u['name']}'s pending order" for op, u, _ in self.jobs]
        steps = "; ".join(f"({i + 1}) {p}" for i, p in enumerate(parts))
        speed = (" Gold-tier customers get express delivery, everyone else standard."
                 if any(op == "ship" for op, _, _ in self.jobs) else "")
        self.instruction = (f"Please work through these requests one after another: {steps}.{speed} When everything "
                            "is done, reply with the total amount of the orders you handled, rounded to 2 decimals.")

    def _build_notify(self) -> None:
        """Ship, then message the customer in the support chat (a plan shape the training generator never produces)."""
        nm = self.user["name"]
        self.instruction = (f"Please send out {nm}'s pending order (express for a gold-tier customer, standard "
                            "otherwise), then write to the customer in the support chat (not by e-mail) to confirm "
                            "that it has shipped." if self.surface else
                            f"Ship {nm}'s pending order (express if the customer is gold tier, otherwise standard), "
                            "then send the customer a message in the support chat (not by e-mail) confirming that it "
                            "has shipped.")

    # ------------------------------------------------------------------ dynamics
    def reset(self) -> str:
        super().reset()
        self.asked_slots = set()
        policy = ""
        if self.ask_tool:
            policy = POLICY_SHIFT.format(ask=ALIASES[ASK][0]) if self.surface else ASK_POLICY
        obs = OPENING[self.surface] + policy
        self._seen = [obs]
        return obs

    def slots(self) -> list[str]:
        """The details the customer must be asked for (clarify only), in plan order."""
        if self.family != "clarify":
            return []
        if self.variant == "ask_twice":
            return ["customer_name", "shipping_speed"]
        return [self.missing] if self.missing else []

    def _reply_of(self, slot: str) -> str:
        return getattr(self, "replies", {}).get(slot, self.reply)

    def _ask(self, args: dict[str, Any]) -> StepResult:
        """The generator's rule per slot: the right slot gets its reply (again, if asked again); acting before the
        first question of a slot counts as acting before asking."""
        self.n_asks += 1
        if self.family == "clarify":
            slot, slots = self._missing_of(args), self.slots()
            if slot in slots:
                if slot not in self.asked_slots and (any(self.effects.get(k) for k in ("cancelled", "refunds",
                                                                                         "shipped"))
                                                     or self.side_effects):
                    self.acted_before_ask = True
                self.asked_slots.add(slot)
                self.asked = set(slots) <= self.asked_slots
                return StepResult(f"Customer: {self._reply_of(slot)}")
            return StepResult(REPEAT_REPLY[self.surface])
        return StepResult(NOTHING_REPLY[self.surface])

    def _step(self, tool: str, args: dict[str, Any]) -> StepResult:
        t = SHIFT_NEAR_BY_NAME.get(tool) if tool in getattr(self, "_shift_near_offered", ()) else None
        if t is None:
            return super()._step(tool, args)
        self.calls.append((tool, dict(args)))
        if t.write:
            if self.strict:
                self.failed = True
                return StepResult(f"Irreversible error: {tool}({args}) was applied incorrectly.", done=True)
            self.side_effects.append((tool, dict(args)))
            return StepResult("Done.")
        w, item = self.world, args.get("item")
        if tool == "lookup_customer_by_email":
            return StepResult("No customer found with that e-mail address." + self._noise())
        if tool == "lookup_customer_by_postcode":
            return StepResult("[]" + self._noise())
        if tool == "lookup_supplier":
            return StepResult("No supplier found." + self._noise())
        if tool in ("list_customer_orders_archived", "list_customer_returns"):
            return StepResult("[]")
        if tool == "count_customer_orders":
            uid = args.get("user_id")
            if uid not in w.users:
                return StepResult("No such customer.")
            counts = {s: sum(1 for o in w.orders.values() if o["user_id"] == uid and o["status"] == s)
                      for s in ("pending", "shipped", "delivered", "cancelled")}
            return StepResult(json.dumps(counts))
        if tool == "inventory_count_by_site":
            if item not in w.stock:
                return StepResult("Unknown item.")
            wh = str(args.get("warehouse"))
            return StepResult(f"{item} @ {wh}: {self._warehouse_split(item).get(wh, 0)} units")
        if tool == "inventory_count_reserved":
            return StepResult(f"{item}: {(w.stock.get(item, 0) * 3 + 5) % 40} units reserved" if item in w.stock
                              else "Unknown item.")
        if tool == "dispatch_quote":
            ok = str(args.get("order_id")) in w.orders
            return StepResult(f"Quote: ${'15.40' if args.get('speed') == 'express' else '5.20'}" if ok
                              else "Order not found.")
        if tool in ("message_team", "message_supplier"):
            return StepResult("Note posted.")
        if tool == "request_clarification_later":
            return StepResult("Call booked.")
        return StepResult(f"{tool}: no relevant result.")

    # ------------------------------------------------------------------ oracle (canonical names)
    def _plan(self) -> list[tuple[str, dict[str, Any]]]:
        u, var = self.user, self.variant
        find = ("find_user", {"name": u["name"]})
        done = ("finish", {"answer": "done"})
        if self.family == "notify":
            return [find, ("get_orders", {"user_id": u["user_id"], "status": "pending"}),
                    ("ship_order", {"order_id": self.pending_order["order_id"], "speed": _speed(u)}),
                    ("send_message", {"user_id": u["user_id"]}), done]
        if self.family == "clarify" and var == "ask_twice":
            # who first (nothing can be looked up before), the speed when the write needs it; the labels (AT) do not
            # depend on the order of the two questions, only the canonical next action does
            return [(ASK, {"missing": "customer_name"}), find,
                    ("get_orders", {"user_id": u["user_id"], "status": "pending"}), (ASK, {"missing": "shipping_speed"}),
                    ("ship_order", {"order_id": self.pending_order["order_id"], "speed": self.ship_speed}), done]
        if self.family == "clarify" and var == "ask_after_lookup":
            o = self.delivered_order
            return [find, ("get_orders", {"user_id": u["user_id"], "status": "delivered"}), (ASK, {"missing": "order_id"}),
                    ("refund", {"order_id": o["order_id"], "amount": o["amount"], "method": "store_credit"}), done]
        return super()._plan()

    def _ask_done(self, args: Mapping[str, Any]) -> bool:
        return str(args.get("missing")) in self.asked_slots

    def oracle_action(self) -> tuple[str, dict[str, Any]] | None:
        """The parent's rule with one question per slot (identical for single-slot tasks)."""
        for tool, args in self._plan():
            if tool == ASK:
                if not self._ask_done(args):
                    return tool, args
                continue
            if tool == "finish" or not self._done_step(tool, args):
                return tool, args
        return None

    def _undone(self) -> list[tuple[str, dict[str, Any]]]:
        out = []
        for tool, args in self._plan():
            if tool == ASK:
                if not self._ask_done(args):
                    out.append((tool, args))
                continue
            if tool == self.FINAL_TOOL or not self._done_step(tool, args):
                out.append((tool, args))
        return out

    def oracle_accepts(self, tool: str, args: dict[str, Any]) -> bool | None:
        nxt = self.oracle_action()
        if nxt is None:
            return None
        if tool == ASK and nxt[0] == ASK:
            return self._missing_of(args) == nxt[1].get("missing")
        return super().oracle_accepts(tool, args)

    def _ship_call_index(self) -> int | None:
        gold = {"order_id": self.pending_order["order_id"], "speed": _speed(self.user)}
        shipped = any(str(e[0]) == str(gold["order_id"]) for e in self.effects.get("shipped", []))
        if not shipped:
            return None
        return next((i for i, (t, a) in enumerate(self.calls) if t == "ship_order" and self._args_match(t, a, gold)),
                    None)

    def _messaged_after_ship(self) -> bool:
        i = self._ship_call_index()
        uid = str(self.user["user_id"])
        return i is not None and any(t == "send_message" and str(a.get("user_id")).strip() == uid
                                     for t, a in self.calls[i + 1:])

    def _done_step(self, tool: str, args: dict[str, Any]) -> bool:
        if self.family == "notify" and tool == "send_message":
            return self._messaged_after_ship()
        return super()._done_step(tool, args)

    def _step_atoms(self, tool: str, args: dict[str, Any]) -> list[tuple[str, Any]]:
        f, var = self.family, self.variant
        if f == "notify":
            atoms: list[tuple[str, Any]] = []
            if tool in ("get_orders", "send_message"):
                atoms.append(("user_id", args.get("user_id")))
            elif tool == "ship_order":
                atoms.append(("order_id", args.get("order_id")))
                o = self.world.orders.get(str(args.get("order_id")))
                u = self.world.users.get(o["user_id"]) if o else None
                if u:
                    atoms.append(("tier", u["tier"]))
            return [(k, v) for k, v in atoms if v is not None and v != ""]
        if f == "clarify" and var == "ask_twice":
            atoms = []
            if tool == "find_user":
                atoms += [("name", args.get("name")), ("reply", self.replies["customer_name"])]
            elif tool == "get_orders":
                atoms.append(("user_id", args.get("user_id")))
            elif tool == "ship_order":
                atoms += [("order_id", args.get("order_id")), ("reply", self.replies["shipping_speed"])]
            return [(k, v) for k, v in atoms if v is not None and v != ""]
        if f == "clarify" and var == "ask_after_lookup":
            atoms = []
            if tool == "get_orders":
                atoms.append(("user_id", args.get("user_id")))
            elif tool == "refund":
                atoms += [("order_id", args.get("order_id")), ("amount", args.get("amount")), ("reply", self.reply)]
            return [(k, v) for k, v in atoms if v is not None and v != ""]
        return super()._step_atoms(tool, args)

    def oracle_snapshot(self) -> "ShiftCore":
        c = super().oracle_snapshot()
        c.asked_slots = set(getattr(self, "asked_slots", set()))
        return c

    def evaluate(self, final_answer: str | None) -> dict[str, Any]:
        f, var = self.family, self.variant
        if not (f == "notify" or (f == "clarify" and var in ("ask_twice", "ask_after_lookup"))):
            return super().evaluate(final_answer)
        extra: dict[str, Any] = {"side_effects": len(getattr(self, "side_effects", [])),
                                 "n_asks": getattr(self, "n_asks", 0)}
        if f == "clarify":
            extra.update({"asked": bool(getattr(self, "asked", False)),
                          "acted_before_ask": bool(getattr(self, "acted_before_ask", False))})
        if getattr(self, "failed", False):
            return {"success": False, "score": 0.0, "n_calls": len(self.calls), "optimal_calls": len(self._plan()) - 1,
                    "irreversible_error": True, **extra}
        eff = self.effects
        if f == "notify":
            ok = (eff.get("shipped") == [(self.pending_order["order_id"], _speed(self.user))]
                  and self._messaged_after_ship() and not eff.get("cancelled") and not eff.get("refunds"))
            extra["messaged_after_ship"] = self._messaged_after_ship()
        elif var == "ask_twice":
            ok = (eff.get("shipped") == [(self.pending_order["order_id"], self.ship_speed)]
                  and bool(getattr(self, "asked", False)) and not getattr(self, "acted_before_ask", False)
                  and not eff.get("cancelled") and not eff.get("refunds"))
        else:   # ask_after_lookup
            ok = (self._refund_ok("store_credit") and bool(getattr(self, "asked", False))
                  and not getattr(self, "acted_before_ask", False) and not eff.get("cancelled")
                  and not eff.get("shipped"))
        ok = ok and not getattr(self, "side_effects", [])
        return {"success": bool(ok), "score": float(ok), "n_calls": len(self.calls),
                "optimal_calls": len(self._plan()) - 1, **extra}


# ------------------------------------------------------------------------------------------------ the Env the agent sees
HIDDEN = "__not_offered__:"       # a canonical core name under the surface axis: an unknown tool for the oracle


class JevGymShiftEnv(Env):
    """DecisionGym-Shift as the harness sees it: the core's tools under their aliases; every name crossing the
    boundary is translated (alias <-> canonical); everything else is the core's."""

    FINAL_TOOL = "finish"

    def __init__(self, seed: int, *args: Any, surface: bool = True, distraction: bool = True, structure: bool = True,
                 **kw: Any):
        self.core = ShiftCore(seed, *args, surface=surface, distraction=distraction, structure=structure, **kw)
        self._names(surface)

    def _names(self, surface: bool) -> None:
        self.alias = {c: a for c, (a, _, _) in ALIASES.items()} if surface else {}
        self.canon = {a: c for c, a in self.alias.items()}

    @classmethod
    def _wrap(cls, core: ShiftCore) -> "JevGymShiftEnv":
        obj = cls.__new__(cls)
        obj.core = core
        obj._names(core.surface)
        return obj

    def __getattr__(self, k: str) -> Any:      # seed, family, variant, world, jobs, ... (read-only view)
        if k == "core":
            raise AttributeError(k)
        return getattr(self.core, k)

    # names
    def to_canon(self, tool: str) -> str:
        if tool in self.canon:
            return self.canon[tool]
        return HIDDEN + tool if tool in self.alias else tool

    def to_ext(self, tool: str) -> str:
        return self.alias.get(tool, tool)

    def _spec(self, t: dict[str, Any]) -> dict[str, Any]:
        name = t["function"]["name"]
        if name not in self.alias:
            return t
        alias, desc, pdesc = ALIASES[name]
        s = copy.deepcopy(t)
        s["function"]["name"], s["function"]["description"] = alias, desc
        for p, d in pdesc.items():
            if p in s["function"]["parameters"]["properties"]:
                s["function"]["parameters"]["properties"][p]["description"] = d
        return s

    # Env
    @property
    def task_id(self) -> str:   # type: ignore[override]
        return self.core.task_id

    @property
    def instruction(self) -> str:   # type: ignore[override]
        return self.core.instruction

    @property
    def gold(self) -> dict[str, Any]:   # type: ignore[override]
        return self.core.gold

    @property
    def param_kinds(self) -> dict[str, str]:
        out = {}
        for k, v in self.core.param_kinds.items():
            t, _, p = k.partition(".")
            out[f"{self.to_ext(t)}.{p}"] = v
        return out

    def reset(self) -> str:
        return self.core.reset()

    def tools(self) -> list[dict[str, Any]]:
        return [self._spec(t) for t in self.core.tools()]

    def step(self, tool: str, args: dict[str, Any]) -> StepResult:
        return self.core.step(self.to_canon(tool), args)

    def evaluate(self, final_answer: str | None) -> dict[str, Any]:
        return self.core.evaluate(final_answer)

    # oracle (outputs in the names the agent sees; inputs translated)
    def oracle_action(self) -> tuple[str, dict[str, Any]] | None:
        a = self.core.oracle_action()
        return None if a is None else (self.to_ext(a[0]), a[1])

    def oracle_done(self) -> bool | None:
        return self.core.oracle_done()

    def oracle_accepts(self, tool: str, args: dict[str, Any]) -> bool | None:
        return self.core.oracle_accepts(self.to_canon(tool), args)

    def oracle_verdict(self, tool: str, args: dict[str, Any] | None, budget: int = 30,
                       d_before: float | None = None) -> dict[str, Any]:
        return self.core.oracle_verdict(self.to_canon(tool), args, budget, d_before)

    def oracle_acceptable_tool_sets(self, budget: int = 30) -> tuple[list[str], list[str]]:
        f, u = self.core.oracle_acceptable_tool_sets(budget)
        return [self.to_ext(t) for t in f], [self.to_ext(t) for t in u]

    def oracle_acceptable_tools(self, budget: int = 30, feasible: bool = True) -> list[str]:
        f, u = self.oracle_acceptable_tool_sets(budget)
        return f if feasible else u

    def oracle_cost_to_go(self, budget: int = 30) -> float:
        return self.core.oracle_cost_to_go(budget)

    def oracle_world_complete(self) -> bool:
        return self.core.oracle_world_complete()

    def oracle_finish_ok(self, answer: Any) -> bool:
        return self.core.oracle_finish_ok(answer)

    def oracle_need(self) -> list[dict[str, Any]]:
        return self.core.oracle_need()

    def oracle_mem_gold(self, text: str, known: Any = ()) -> dict[str, Any]:
        return self.core.oracle_mem_gold(text, known)

    def oracle_memory_read_gold(self, texts: Any) -> dict[str, list[int]]:
        return self.core.oracle_memory_read_gold(texts)

    def oracle_snapshot(self) -> "JevGymShiftEnv":
        return self._wrap(self.core.oracle_snapshot())


def shift_flags(shift: Any) -> dict[str, bool]:
    """The suite block's ``shift`` value -> the three axis flags (``true`` = all on; a mapping may switch any off)."""
    if shift is True or shift is None:
        return {a: True for a in AXES}
    if shift is False:
        raise ValueError("shift: false is not a shifted suite; drop the key to run the generator itself")
    if not isinstance(shift, Mapping):
        raise ValueError(f"shift must be true or a mapping of {AXES}, got {shift!r}")
    unknown = sorted(set(shift) - set(AXES))
    if unknown:
        raise ValueError(f"unknown shift axis {unknown}; valid: {', '.join(AXES)}")
    return {a: bool(shift.get(a, True)) for a in AXES}


def task_factories(n: int = 50, seed0: int = 0, shift: Any = True, **kw: Any) -> list[Callable[[], JevGymShiftEnv]]:
    """Zero-argument factories, as ``jevgym.task_factories`` (same options plus ``shift``)."""
    unknown = sorted(set(kw) - set(BASE_ENV_KWARGS))
    if unknown:
        raise ValueError(f"unknown jevgym suite option(s) {unknown}; valid: n, seed0, shift, {', '.join(BASE_ENV_KWARGS)}")
    flags = shift_flags(shift)
    JevGymShiftEnv(seed=seed0, **flags, **kw)  # validate the options once, eagerly
    return [(lambda i=i: JevGymShiftEnv(seed=seed0 + i, **flags, **kw)) for i in range(n)]
