"""Surface augmentation of agent-state training rows (exploratory row E27, docs/PREREG_A7_STAGE2.md, Deviations
2026-10-09 "E27: a 0.8B decision model that holds under shift" (c) 1-3).

E21 showed that the 0.8B scoring head (E14 arm A) learned the generator's tool NAMES more than the decisions: renaming
the tools cost it +21.9 points of open-loop accuracy. E27 trains the same recipe on states whose surface varies. For a
seeded half of the JevGym states (:func:`state_selected`), every row on the state gets the same :class:`Aug`:

* **rename** the 8 core tools to procedural ``verb_noun`` names (per-role word lists ``VERBS`` / ``NOUNS``), in the state
  text (history calls, the policy line, error lines), the option labels, the question text and the target;
* **reword** the renamed tools' descriptions (``DESCS``), the request (the ``TASK:`` line, when it is one of the
  generator's request forms: :func:`parse_task`), the opening and policy lines of the first observation and the
  customer's replies, by templates;
* **add near-copy distractors** (2-6 per tool choice): a renamed core name plus a qualifier word, with the role's
  paraphrase plus a clause that changes what the tool does (``NEAR``), target 0.

Parameter names, enum values, observation formats and argument values stay, as on E21's surface axis. Every template
field (``{nm}``, ``{oid}``, ``{F_status}``, ...) is filled from the state's own world values, field names or enum values,
never from the augmentation's vocabulary.

**No leakage** ((c) 3): :func:`aug_vocabulary` is every word of the word lists and template literals (fields removed)
minus ``FUNCTION_WORDS``; :func:`e21_vocabulary` is every word of the string literals of ``jevgym_shift.py`` and
``jevgym.py`` plus E21's L2 corpus texts in all renderings; :func:`disjointness` requires an empty overlap on tokens and
on light stems. Pure Python: no model, no tokenizer, no E21 seed.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import random
import re
import string
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SALT = "jev-e27-aug"
CORE = ("find_user", "get_orders", "cancel_order", "refund", "check_stock", "ship_order", "send_message", "ask_user")
SEP = " -- "
N_NEAR = (2, 6)

# ------------------------------------------------------------------------------------------------ vocabulary
# procedural names: verb_noun, the verb from per-role word lists, the noun the state's customer / order word (or a
# per-role noun); 8 distinct names per state
VERBS: dict[str, tuple[str, ...]] = {
    "find_user": ("retrieve", "pinpoint", "discover", "trace", "seek", "unearth"),
    "get_orders": ("enumerate", "browse", "tabulate", "itemise", "scan", "collate"),
    "cancel_order": ("annul", "revoke", "abort", "rescind", "scrap", "retract"),
    "refund": ("repay", "remit", "compensate", "rebate", "recompense", "indemnify"),
    "check_stock": ("tally", "gauge", "quantify", "assay", "appraise", "reckon"),
    "ship_order": ("forward", "convey", "transport", "consign", "freight", "haul"),
    "send_message": ("inform", "ping", "alert", "apprise", "brief", "signal"),
    "ask_user": ("inquire", "probe", "consult", "canvass", "solicit", "poll"),
}
NOUNS: dict[str, tuple[str, ...]] = {      # roles whose noun is not the state's customer / order word
    "check_stock": ("wares", "goods", "merchandise", "supply"),
    "send_message": ("memo", "missive", "bulletin", "dispatchnote"),
}
# the words the reworded texts use for the world's things (one draw per state, so the state reads consistently)
WHO = ("client", "buyer", "patron", "guest")
WHAT = ("docket", "requisition", "basket", "cart")
PACE = ("tempo", "pace")
# descriptions: {c} who, {o} what (singular), {os} what (plural), {p} pace word; F_ = a field name, V_ = an enum value
DESCS: dict[str, tuple[str, ...]] = {
    "find_user": ("Pinpoint a {c} from forename and surname; yields {F_user_id} alongside {F_tier}.",
                  "Discover which {c} a forename and surname belong to; outputs {F_user_id} together with {F_tier}."),
    "get_orders": ("Enumerate every {o} tied to one {c}, narrowed down to one {F_status}.",
                   "Browse the {os} of one {c}, restricted to one {F_status}."),
    "cancel_order": ("Annul one {o} outright, so nobody packs it.",
                     "Revoke one {o} prior to packing."),
    "refund": ("Repay to its {c} the money spent on a {o}.",
               "Remit to the {c} the money spent on one {o}."),
    "check_stock": ("Tally how much of some merchandise lies on our shelves at present.",
                    "Gauge how much of some merchandise lies on our shelves."),
    "ship_order": ("Forward an outstanding {o} to its {c} at a selected {p}.",
                   "Convey an outstanding {o} to its {c}; the {p} is picked by the invoker."),
    "send_message": ("Compose a memo for a {c} within the conversation window.",
                     "Craft a freeform memo for one {c} within the conversation window."),
    "ask_user": ("Pose an inquiry to the {c}, then await their feedback. Intended for situations where an essential "
                 "fact is absent or vague.",
                 "Pose an inquiry to the {c} and pause for their feedback; intended solely for moments when an essential "
                 "fact is absent or vague."),
}
# near-copy distractors: (qualifier appended to the renamed name, clause appended to the role's paraphrase)
NEAR: tuple[tuple[str, str], ...] = (
    ("preview", "Dry rehearsal: alters no data whatsoever."),
    ("bulk", "Operates upon an entire group simultaneously, not upon one {o}."),
    ("draft", "Prepares a draft awaiting approval by a manager."),
    ("overseas", "Solely for {c}s living in foreign lands."),
    ("vintage", "Solely for {os} from bygone seasons."),
    ("projection", "Yields a projected figure; alters no data."),
    ("crew", "Intended for crew logins, not for a {c}."),
    ("weekly", "Operates on a weekly timetable instead of instantly."),
)
OPENINGS = ("A toolkit for {c} {os} is ready.", "These utilities manage {c} {os}.")
POLICIES = (" Principle: no assumptions. Whenever an essential fact (the {c}'s moniker, which {o}, the repayment "
            "channel, the {p} of conveyance) is absent or vague, consult the {c} with {ask} prior to altering {os}.",
            " Principle: no guesswork. If an essential fact (the {c}'s moniker, which {o}, the repayment channel, "
            "the {p} of conveyance) is absent or vague, consult the {c} with {ask} prior to altering {os}.")
# the request forms of the generator (bench/suites/jevgym.py) -> templates with the same content
REQUESTS: dict[str, tuple[str, ...]] = {
    "cancel": ("{nm} wishes their outstanding {o} annulled; attend to it.",
               "Annul the outstanding {o} belonging to {nm}; this was the {c}'s wish."),
    "refund_credit": ("Repay entirely the newest {V_delivered} {o} of {nm} (top serial), as {V_store_credit}.",
                      "{nm} deserves the entire outlay of their newest {V_delivered} {o} (top serial) as "
                      "{V_store_credit}."),
    "stock": ("What quantity of '{item}' lies on our shelves? Tell us the figure.",
              "Tell us the figure: what quantity of '{item}' lies on our shelves?"),
    "ship_policy": ("Forward {nm}'s outstanding {o}: {V_express} {p} when the {c}'s {F_tier} is {V_gold}, "
                    "{V_standard} {p} for everybody besides.",
                    "Convey the outstanding {o} of {nm}, at {V_express} {p} for a {V_gold} {F_tier} and at "
                    "{V_standard} {p} for everybody besides."),
    "total": ("Combine the outlays of every {V_delivered} {o} of {nm}; tell us that figure to the nearest cent.",
              "Tell us the combined outlay of all {V_delivered} {os} of {nm}, to the nearest cent."),
    "refund_method": ("Repay entirely the newest {V_delivered} {o} of {nm} (top serial).",
                      "{nm} deserves the entire outlay of their newest {V_delivered} {o} (top serial)."),
    "ship_speed": ("Forward {nm}'s outstanding {o}.", "Convey the outstanding {o} of {nm}."),
    "ship_name": ("An anonymous {c} wishes their outstanding {o} forwarded ({V_express} {p} for a {V_gold} {F_tier}, "
                  "{V_standard} {p} for everybody besides).",
                  "Somebody anonymous wishes their outstanding {o} conveyed ({V_express} {p} for a {V_gold} {F_tier}, "
                  "{V_standard} {p} for everybody besides)."),
    "multi": ("Tackle these chores sequentially: {steps}.{speed} Afterwards, tell us the combined outlay of those {os}, "
              "to the nearest cent.",
              "Sequentially tackle: {steps}.{speed} Afterwards, tell us the combined outlay of those {os}, to the "
              "nearest cent."),
    "multi_cancel": ("annul {nm}'s outstanding {o} (the {c}'s wish)",),
    "multi_ship": ("forward {nm}'s outstanding {o}",),
    "multi_speed": (" {V_express} {p} for {V_gold} {F_tier} {c}s, {V_standard} {p} for everybody besides.",),
}
REPLIES: dict[str, tuple[str, ...]] = {
    "order": ("My mistake -- annul {o} {oid}, the {item}.", "Oops, I mean {o} {oid} (the {item}); annul that one."),
    "method_store_credit": ("Opt for {V_store_credit}.",),
    "method_original_payment": ("Opt for {V_original_payment}.",),
    "speed": ("{V_speed} {p} is fine by me.",),
    "name": ("My bad -- I am {nm}.",),
    "repeat": ("I mentioned that previously.",),
    "nothing": ("No remarks from me; proceed as outlined.",),
}
# words that carry no content (articles, pronouns, prepositions, conjunctions, auxiliaries, quantifiers, the
# possessive / plural s): left out of the disjointness check on both sides
FUNCTION_WORDS = frozenset("""
a an the this that these those it its they them their theirs he she his her him we us our ours i me my mine you
your yours who whom whose which what when where why how there here s
of in on at to from by with for about into onto upon over under within without between among per as than then
and or but nor so yet if because while although though whether either neither both
is are was were be been being am do does did done has have had having will would shall should can could may might
must ought not no all any some each every few many much more most other such own same only just also too very
one two
""".split())

# ------------------------------------------------------------------------------------------------ the generator's forms
TASK_FORMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cancel", re.compile(r"The customer (?P<nm>.+?) asked us to cancel their pending order\. Please do it\.")),
    ("refund_credit", re.compile(r"Refund the full amount of (?P<nm>.+?)'s most recent delivered order "
                                 r"\(highest order id\) as store credit\.")),
    ("refund_method", re.compile(r"Refund the full amount of (?P<nm>.+?)'s most recent delivered order "
                                 r"\(highest order id\)\.")),
    ("stock", re.compile(r"How many units of '(?P<item>.+?)' do we have in stock\? Answer with the number\.")),
    ("ship_policy", re.compile(r"Ship (?P<nm>.+?)'s pending order\. Use express shipping if the customer is gold "
                               r"tier, otherwise standard\.")),
    ("ship_speed", re.compile(r"Ship (?P<nm>.+?)'s pending order\.")),
    ("total", re.compile(r"What is the total amount of all of (?P<nm>.+?)'s delivered orders\? Answer with the number "
                         r"rounded to 2 decimals\.")),
    ("ship_name", re.compile(r"A customer wrote in asking us to ship their pending order \(express if they are gold "
                             r"tier, otherwise standard\), but did not say who they are\.")),
    ("multi", re.compile(r"Process these requests in order: (?P<steps>.+?)\.(?P<speed> Use express shipping for "
                         r"gold-tier customers and standard shipping otherwise\.)? When all are done, answer with the "
                         r"total amount of the orders you processed, rounded to 2 decimals\.")),
)
MULTI_PART = re.compile(r"\((?P<i>\d+)\) (?:(?P<cancel>cancel (?P<nc>.+?)'s pending order \(the customer requested it\))"
                        r"|(?P<ship>ship (?P<ns>.+?)'s pending order))$")
GEN_OPENING = "You have access to the customer-operations tools."
GEN_POLICY = (" Policy: never guess -- if a detail you need to act (who the customer is, which order, the refund method, "
              "the shipping speed) is not given or is ambiguous, ask the customer with ask_user before taking any "
              "action.")
GEN_REPLIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("order", re.compile(r"Customer: Sorry for the confusion -- please cancel order (?P<oid>ORD-\d+) "
                         r"\(the (?P<item>[^)]+)\)\.")),
    ("method_store_credit", re.compile(r"Customer: Please refund it as store credit\.")),
    ("method_original_payment", re.compile(r"Customer: Please refund it to my original payment method\.")),
    ("speed", re.compile(r"Customer: (?P<speed>Standard|Express) shipping, please\.")),
    ("name", re.compile(r"Customer: Oh, sorry -- my name is (?P<nm>[A-Z][a-z]+ [A-Z][a-z]+)\.")),
    ("repeat", re.compile(r"Customer: I think I already gave you everything you need on that point\.")),
    ("nothing", re.compile(r"Customer: I have nothing to add -- please go ahead with the request as described\.")),
)
# template fields filled from the world, never from the augmentation's vocabulary: field names and enum values
FIELDS = {"F_user_id": "user_id", "F_tier": "tier", "F_status": "status", "V_delivered": "delivered",
          "V_store_credit": "store_credit", "V_original_payment": "original_payment", "V_express": "express",
          "V_standard": "standard", "V_gold": "gold"}


def _h(*parts: str) -> int:
    return int(hashlib.sha256("\n".join(parts).encode()).hexdigest(), 16)


def state_selected(state: str) -> bool:
    """Registration (c) 1: a state is augmented iff sha256("jev-e27-aug" newline state) is even."""
    return _h(SALT, state) % 2 == 0


class Aug:
    """One augmented state: the names, descriptions, reworded texts and near copies (seeded by the state)."""

    def __init__(self, state: str):
        self.rng = random.Random(_h(SALT, "rng", state))
        r = self.rng
        self.c, self.o, self.p = r.choice(WHO), r.choice(WHAT), r.choice(PACE)
        self.names: dict[str, str] = {}
        used: set[str] = set()
        noun = {"find_user": self.c, "ask_user": self.c, "get_orders": self.o + "s", "cancel_order": self.o,
                "refund": self.o, "ship_order": self.o}
        for t in CORE:     # the verb lists are disjoint across roles, so the 8 names are distinct
            n = f"{r.choice(VERBS[t])}_{noun.get(t) or r.choice(NOUNS[t])}"
            assert n not in used
            used.add(n)
            self.names[t] = n
        self.descs = {t: self.fill(r.choice(DESCS[t])) for t in CORE}
        self.task_pick = r.random()
        self.reply_pick = r.random()
        self.n_near = r.randint(*N_NEAR)
        self.near_roles = [r.choice([t for t in CORE if t != "ask_user"] + ["ask_user"]) for _ in range(self.n_near)]
        self.near_quals = r.sample(NEAR, k=min(self.n_near, len(NEAR)))
        self.near_pos = [r.random() for _ in range(self.n_near)]

    def fill(self, tpl: str, **kw: Any) -> str:
        return tpl.format(c=self.c, o=self.o, os=self.o + "s", p=self.p, ask=self.names.get("ask_user", "ask_user"),
                          **FIELDS, **kw)

    def pick(self, options: Sequence[str], u: float) -> str:
        return options[min(len(options) - 1, int(u * len(options)))]

    # ------------------------------------------------------------------ texts
    def task(self, text: str) -> tuple[str, str]:
        """The reworded request and the form it matched ('kept' if none: the text stays)."""
        for form, pat in TASK_FORMS:
            m = pat.fullmatch(text)
            if not m:
                continue
            if form == "multi":
                parts = []
                for p in m.group("steps").split("; "):
                    pm = MULTI_PART.fullmatch(p)
                    if not pm:
                        return text, "kept"
                    if pm.group("cancel"):
                        parts.append(f"({pm.group('i')}) " + self.fill(REQUESTS["multi_cancel"][0], nm=pm.group("nc")))
                    else:
                        parts.append(f"({pm.group('i')}) " + self.fill(REQUESTS["multi_ship"][0], nm=pm.group("ns")))
                speed = self.fill(REQUESTS["multi_speed"][0]) if m.group("speed") else ""
                return self.fill(self.pick(REQUESTS["multi"], self.task_pick), steps="; ".join(parts), speed=speed), form
            kw = {k: v for k, v in m.groupdict().items() if v is not None}
            return self.fill(self.pick(REQUESTS[form], self.task_pick), **kw), form
        return text, "kept"

    def reply(self, kind: str, m: re.Match[str]) -> str:
        kw = dict(m.groupdict())
        if "speed" in kw:
            kw["V_speed"] = kw.pop("speed").lower()
            kw = {k: v for k, v in kw.items()}
            return "Customer: " + self.pick(REPLIES[kind], self.reply_pick).format(
                c=self.c, o=self.o, os=self.o + "s", p=self.p, **{**FIELDS, **kw})
        return "Customer: " + self.fill(self.pick(REPLIES[kind], self.reply_pick), **kw)

    def rename_text(self, s: str) -> str:
        """Whole-identifier renames; ``refund`` (also an English word) only as a call or a quoted name."""
        for t in CORE:
            n = self.names[t]
            if t == "refund":
                s = re.sub(r"(?<![\w])refund(?=\()", n, s)
                s = s.replace("`refund`", f"`{n}`")
            else:
                s = re.sub(rf"(?<![\w]){re.escape(t)}(?![\w])", n, s)
        return s

    def state(self, s: str) -> tuple[str, dict[str, Any]]:
        info: dict[str, Any] = {"task": "none", "replies": 0, "policy": False}
        m = re.match(r"TASK: (.*?)\n", s)
        if m:
            new, form = self.task(m.group(1))
            info["task"] = form
            s = "TASK: " + new + "\n" + s[m.end():]
        if GEN_OPENING in s:
            pol = self.fill(self.pick(POLICIES, self.reply_pick))
            s = s.replace(GEN_OPENING + GEN_POLICY, self.fill(self.pick(OPENINGS, self.task_pick)) + pol)
            s = s.replace(GEN_OPENING, self.fill(self.pick(OPENINGS, self.task_pick)))
            info["policy"] = True
        for kind, pat in GEN_REPLIES:
            s, n = pat.subn(lambda mm, k=kind: self.reply(k, mm), s)
            info["replies"] += n
        return self.rename_text(s), info

    def near_options(self, taken: set[str]) -> list[str]:
        out = []
        for role, (qual, clause) in zip(self.near_roles, self.near_quals):
            name = f"{self.names[role]}_{qual}"
            if name in taken:
                continue
            taken.add(name)
            out.append(f"{name}{SEP}{self.descs[role]} {self.fill(clause)}")
        return out


def is_agent_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("id", "")).startswith("jevgym-")


def augment_row(row: Mapping[str, Any], aug: Aug | None = None) -> dict[str, Any]:
    """The row on its augmented state (same id, same answer key moved to the renamed option)."""
    aug = aug or Aug(row["state"])
    out = json.loads(json.dumps(row))
    out["state"], info = aug.state(row["state"])
    out["question"] = aug.rename_text(row["question"])
    opts = list(row["options"])
    tool_choice = row.get("dp") == "tool_select" and row.get("kind", "choice") == "choice" and opts \
        and all(SEP in o for o in opts)
    new_opts, omap = [], {}
    for o in opts:
        n = o
        if tool_choice:
            label, desc = o.split(SEP, 1)
            if label in aug.names:
                n = f"{aug.names[label]}{SEP}{aug.descs[label]}"
        new_opts.append(n)
        omap[o] = n
    near: list[str] = []
    if tool_choice:
        taken = {o.split(SEP, 1)[0] for o in new_opts}
        near = aug.near_options(taken)
        for pos, nopt in zip(aug.near_pos, near):
            new_opts.insert(int(pos * (len(new_opts) + 1)), nopt)
    if len(set(new_opts)) != len(new_opts):
        raise ValueError(f"{row.get('id')}: augmented options collide")
    tgt = {omap[o]: float(v) for o, v in (row.get("target") or {}).items()}
    for nopt in near:
        tgt[nopt] = 0.0
    out["options"] = new_opts
    out["target"] = tgt
    meta = dict(out.get("meta") or {})
    if isinstance(meta.get("gold"), str) and meta["gold"] in omap:
        meta["gold"] = omap[meta["gold"]]
    meta["e27_aug"] = {"names": dict(aug.names), "near": len(near), **info}
    out["meta"] = meta
    return out


def augment_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Registration (c) 1: the agent rows on a selected state augmented (one :class:`Aug` per state), the rest as they
    are; plus counts."""
    cache: dict[str, Aug] = {}
    out: list[dict[str, Any]] = []
    st = {"rows": 0, "agent_rows": 0, "augmented_rows": 0, "states": 0, "augmented_states": 0, "task_forms": {},
          "near_added": 0, "replies_reworded": 0, "policy_reworded": 0}
    seen: set[str] = set()
    for r in rows:
        st["rows"] += 1
        if not is_agent_row(r):
            out.append(dict(r))
            continue
        st["agent_rows"] += 1
        s = r["state"]
        new_state = s not in seen
        seen.add(s)
        st["states"] += new_state
        if not state_selected(s):
            out.append(dict(r))
            continue
        if s not in cache:
            cache[s] = Aug(s)
        a = augment_row(r, cache[s])
        out.append(a)
        st["augmented_rows"] += 1
        info = a["meta"]["e27_aug"]
        st["near_added"] += info["near"]
        if new_state:
            st["augmented_states"] += 1
            st["task_forms"][info["task"]] = st["task_forms"].get(info["task"], 0) + 1
            st["replies_reworded"] += info["replies"]
            st["policy_reworded"] += bool(info["policy"])
    return out, st


# ------------------------------------------------------------------------------------------------ disjointness (c) 3
_FIELD = re.compile(r"\{[^{}]*\}")


def words(text: str) -> list[str]:
    """Word tokens: lowercase letter runs (split at every non-letter, underscores included)."""
    return re.findall(r"[a-z]+", text.lower())


def stem(w: str) -> str:
    """A light stem: one common suffix stripped, a doubled final consonant undone, a final i read as y."""
    for suf in ("ings", "ing", "edly", "ers", "ied", "ies", "ed", "es", "er", "ly", "s", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)] + ("y" if suf in ("ied", "ies") else "")
            break
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    if w.endswith("i"):
        w = w[:-1] + "y"
    return w


def template_strings() -> list[str]:
    """Every vocabulary-bearing literal of the augmentation (word lists and templates)."""
    out: list[str] = []
    for d in (VERBS, NOUNS):
        out += [w for ws in d.values() for w in ws]
    out += list(WHO) + list(WHAT) + list(PACE)
    out += [t for ts in DESCS.values() for t in ts]
    out += [q for q, _ in NEAR] + [c for _, c in NEAR]
    out += list(OPENINGS) + list(POLICIES)
    out += [t for ts in REQUESTS.values() for t in ts]
    out += [t for ts in REPLIES.values() for t in ts]
    return out


def aug_vocabulary() -> set[str]:
    v: set[str] = set()
    for s in template_strings():
        v.update(words(_FIELD.sub(" ", s)))
    return v - FUNCTION_WORDS


def module_literals(path: Path) -> list[str]:
    tree = ast.parse(Path(path).read_text())
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def corpus_texts(cdir: Path) -> Iterable[str]:
    for p in sorted(Path(cdir).glob("questions*.jsonl.gz")):
        with gzip.open(p, "rt") as f:
            for line in f:
                if not line.strip():
                    continue
                q = json.loads(line)
                yield str(q.get("state") or "")
                yield str(q.get("question") or "")
                yield from (str(o) for o in q.get("options") or [])
                yield from (str(d or "") for d in q.get("descriptions") or [])


def e21_vocabulary(modules: Sequence[Path], corpus_dir: Path | None) -> tuple[set[str], dict[str, Any]]:
    v: set[str] = set()
    info: dict[str, Any] = {"modules": [], "corpus_files": []}
    for m in modules:
        lits = module_literals(m)
        for s in lits:
            v.update(words(s))
        info["modules"].append({"path": str(m), "literals": len(lits), "sha256": _sha_file(Path(m))})
    if corpus_dir is not None:
        n = 0
        for t in corpus_texts(corpus_dir):
            v.update(words(t))
            n += 1
        info["corpus_files"] = [{"path": str(p), "sha256": _sha_file(p)}
                                for p in sorted(Path(corpus_dir).glob("questions*.jsonl.gz"))]
        info["corpus_texts"] = n
    return v - FUNCTION_WORDS, info


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def vocab_sha(v: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(v)).encode()).hexdigest()


def disjointness(aug: set[str], e21: set[str]) -> dict[str, Any]:
    """Registration (c) 3: the overlap on tokens and on light stems (both must be empty)."""
    tok = sorted(aug & e21)
    sa = {stem(w): w for w in aug}
    se: dict[str, str] = {}
    for w in e21:
        se.setdefault(stem(w), w)
    st = sorted(set(sa) & set(se))
    return {"aug_words": len(aug), "e21_words": len(e21), "aug_sha256": vocab_sha(aug), "e21_sha256": vocab_sha(e21),
            "overlap_tokens": tok, "overlap_stems": [[sa[s], se[s]] for s in st],
            "overlap_count": len(tok) + len(st), "ok": not tok and not st}


def template_fields() -> set[str]:
    """Every field name the templates use (a test checks each can be filled)."""
    out: set[str] = set()
    for s in template_strings():
        out.update(f for _, f, _, _ in string.Formatter().parse(s) if f)
    return out
