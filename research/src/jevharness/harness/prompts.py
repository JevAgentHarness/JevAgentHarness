"""Question wording and harness notes shared by every mode that asks them (a shadow arm must ask exactly the
question its live counterpart asks). Changing a text here changes the System-1 decision-cache keys, so old answers
are never replayed for a new wording.

Wording amendment A4 (docs/META_GOLD_SPEC.md §3.1, §3.3): the reflection question is forward-looking (the dev
positives were all pending wrong actions, which the retrospective v1 wording never asked about), the stop guard
shows the proposed answer, and the stop-guard rejection note no longer invites an extra (in strict mode fatal) write.
"""

from __future__ import annotations

import json
from typing import Any

Q_ROUTING = ("Does the next step require careful multi-step reasoning, planning, or recovery from an error (i.e., a "
             "strong model) rather than a routine action?")
Q_REFLECTION = ("Should the agent pause and re-check before its next action: is it stuck or off track, or likely to "
                "make a wrong action or give a wrong final answer next?")
Q_REFLECTION_V1 = "Is the agent stuck, repeating itself, or did the last action fail or move away from the goal?"
REFLECTION_WORDINGS = {"v2": Q_REFLECTION, "v1": Q_REFLECTION_V1}
Q_STOP_TRIGGER = "Has the task already been fully completed, so the agent should stop now?"
Q_MEMORY_WRITE = ("Will this information likely be needed later to complete the task "
                  "(facts, IDs, constraints, user preferences, results)?")
Q_TOOL_SELECT = "Which tool should the agent call next?"
# E6b (memory_read trigger): asked on the System-1 state, which shows the recent history but not the stored memories
Q_MEMORY_READ_TRIGGER = ("Does the agent need information it stored earlier (facts, IDs, results from steps no longer "
                         "in the recent history) to choose its next step correctly?")

REFLECT_PROMPT = ("Pause. In <=4 sentences: assess progress toward the task, identify any mistake so far, and state "
                  "the best next step. Do not call tools.")
STOP_REJECT_NOTE = ("[Verifier] The task may not be complete yet, or the final answer may be wrong. Re-check the "
                    "requests and recompute the answer before calling finish.")
EARLY_STOP_PROMPT = "The task appears complete. Call finish with the final answer."
ANSWER_CHARS = 800      # stop-guard answer budget; longer answers keep their head and (mostly) their tail
ANSWER_HEAD_CHARS = 200


def clip_answer(answer: Any) -> str:
    """The proposed answer as shown to the stop guard: whole up to ``ANSWER_CHARS``, else its first
    ``ANSWER_HEAD_CHARS`` and its last ``ANSWER_CHARS - ANSWER_HEAD_CHARS`` characters. The tail is kept because agents
    end long summaries with the value the task asks for (Jev pilot 2026-10-01: a 300-character head cut that value
    off and Jev rejected 84 of 90 correct finishes)."""
    a = "" if answer is None else str(answer)
    if len(a) <= ANSWER_CHARS:
        return a
    return a[:ANSWER_HEAD_CHARS] + " [...] " + a[-(ANSWER_CHARS - ANSWER_HEAD_CHARS):]


def q_stop_guard(answer: Any) -> str:
    """The stop-guard question with the proposed answer, clipped by ``clip_answer`` and JSON-quoted (quoting keeps
    keywords inside the answer from steering over-the-wire decision-point inference, ``controls.infer_dp``)."""
    return (f"The agent wants to stop and submit the final answer {json.dumps(clip_answer(answer))}. Is the task "
            "actually complete, and is this answer correct?")


def q_tool_guard(name: str, args: Any) -> str:
    return (f"The agent proposes to call `{name}` with arguments {json.dumps(args)[:500]}. "
            "Is this the right next step toward completing the task?")


def tool_reject_note(name: str, args: Any) -> str:
    return (f"[Verifier] Your proposed call `{name}({json.dumps(args)[:300]})` was judged likely "
            "wrong. Reconsider and choose the correct next tool call.")


def q_action_guard(action: str) -> str:
    """E6c (action guard): the tool-guard question for a proposed admissible action of an action-only env."""
    return (f"The agent proposes to take the action `{str(action)[:500]}`. "
            "Is this the right next step toward completing the task?")


def action_reject_note(action: str) -> str:
    """E6c (action guard): the tool-guard rejection note for a proposed action (a guard only rejects; no hint)."""
    return (f"[Verifier] Your proposed action `{str(action)[:300]}` was judged likely wrong. Reconsider and choose "
            "the correct next action.")


# E6f: the trigger at tool choice, tool arguments and action. System-1 is asked whether the LLM's proposal needs a
# second look; if so, the LLM runs one verification call with the matching note (a check of its own proposal, not a
# verdict: the note says nothing about whether the proposal is right).
def q_verify_tool(name: str, args: Any) -> str:
    """E6f (tool-choice trigger): does the proposed choice of tool need the LLM's verification pass?"""
    return (f"The agent proposes to call `{name}` with arguments {json.dumps(args)[:500]}. Should the agent re-check "
            "this choice of tool before the call runs: could another tool be the right next step?")


def q_verify_args(name: str, args: Any) -> str:
    """E6f (argument trigger): do the proposed argument values need the LLM's verification pass?"""
    return (f"The agent proposes to call `{name}` with arguments {json.dumps(args)[:500]}. Should the agent re-check "
            "these argument values before the call runs: could a value be wrong or missing?")


def q_verify_action(action: str) -> str:
    """E6f (action trigger): does the proposed admissible action need the LLM's verification pass?"""
    return (f"The agent proposes to take the action `{str(action)[:500]}`. Should the agent re-check this action "
            "before it runs: could another action be the right next step?")


def verify_tool_note(name: str, args: Any) -> str:
    """E6f: the LLM's verification call at tool choice (every tool offered; the call it makes is executed)."""
    return (f"[Check] Before your proposed call `{name}({json.dumps(args)[:300]})` runs, re-read the task and the "
            "observations so far and check that this is the right tool to call now. If it is, make the same call "
            "again; if not, make the right call instead.")


def verify_args_note(name: str, args: Any) -> str:
    """E6f: the LLM's verification call at tool arguments (the call is forced to the proposed tool)."""
    return (f"[Check] Before your proposed call `{name}({json.dumps(args)[:300]})` runs, re-read the task and the "
            f"observations so far and check every argument value. Call `{name}` again, with the same arguments if "
            "they are right or with corrected arguments if not.")


def verify_action_note(action: str) -> str:
    """E6f: the LLM's verification call at the action point (after the admissible list)."""
    return (f"[Check] Before your proposed action `{str(action)[:300]}` runs, re-read the task and the observations "
            "so far and check that it is the right next action. If it is, repeat it; if not, choose the right action "
            "instead. End with a line `Action: <action>` copied verbatim from the list.")


def arg_reject_note(name: str, args: Any, param: str) -> str:
    """E6b (tool_params guard): the veto names the argument, never a replacement value (a guard only rejects)."""
    return (f"[Verifier] In your proposed call `{name}({json.dumps(args)[:300]})`, the value of the argument "
            f"`{param}` was judged likely wrong. Reconsider and choose the correct next tool call.")


def llm_gate_prompt(state: str, question: str, advice: str | None = None) -> str:
    """E6b: the LLM answers a yes / no meta question itself (the deferral of a reflection / routing cascade), on the
    same state System-1 saw -- the pattern of the memory-write LLM gate (``memory.py``). E6d (assist): ``advice``
    (:func:`advice_note`) is shown before the question; without it the prompt is E6b's, byte for byte."""
    if advice:
        return f"{state}\n\n{advice}\n\n{question} Answer yes or no."
    return f"{state}\n\n{question} Answer yes or no."


def advice_note(yes: bool, confidence: float) -> str:
    """E6d (assist at a yes / no point): System-1's answer to the question that follows, with its confidence, as a hint
    the LLM may override."""
    return (f"[Advisor] A fast decision model answered the question below: {'yes' if yes else 'no'} (confidence "
            f"{confidence:.2f}). It can be wrong; decide yourself.")


def stop_advice_note(yes: bool, confidence: float) -> str:
    """E6d (stop assist): System-1's answer to the stop-trigger question, shown in the LLM's turn before its call."""
    return (f"[Advisor] A fast decision model's answer to \"{Q_STOP_TRIGGER}\": {'yes' if yes else 'no'} (confidence "
            f"{confidence:.2f}). It can be wrong; decide yourself whether to call finish now.")


MEMORY_SELECT_PROMPT = ("Which of the stored memories above does the agent's next step need? Answer with their ids "
                        "separated by commas (for example: m3, m7), or with none.")
MEMORY_REJECT_NOTE = ("[Verifier] Your memory selection was judged likely wrong (a needed memory missing, or the wrong "
                      "ones). Reconsider and select again.")


def q_memory_read_guard(labels: list[str]) -> str:
    """E6d (memory-read guard): the tool-guard-style question about the LLM's memory selection."""
    sel = ", ".join(labels) if labels else "none of them"
    return (f"The agent proposes to recall these stored memories for its next step: {sel}. Are these the memories it "
            "needs, with no needed memory missing?")


def assist_args_hint(name: str, shortlists: list[tuple[str, list[Any], bool]], fixed: dict[str, Any]) -> str:
    """E6b (tool_params assist): the forced call's instruction with System-1's shortlist per argument (best first)."""
    lines = [f"- `{p}`: one of {json.dumps(vals)}" + (" (or another value)" if open_ else "")
             for p, vals, open_ in shortlists]
    hint = f"You must now call `{name}`."
    if lines:
        hint += " Choose these arguments from the shortlists:\n" + "\n".join(lines)
    if fixed:
        hint += f"\nAlready-determined arguments: {json.dumps(fixed)}."
    return hint + "\nProvide all arguments."


def append_note(messages: list[dict[str, Any]], text: str) -> None:
    """Keep a harness note (a reflection, a stop-verifier verdict) in the LLM's context for the rest of the
    episode. After a tool result the note is appended to that tool message instead of following it as a new
    ``user`` message: Gemini 3 answers a tool-calling request whose history reads ``tool -> user(text) ->
    assistant(call) -> tool`` with an EMPTY response (no text, no call; finish_reason None), which the loop
    turns into an empty ``finish`` and the episode fails. Reproduced on gemini-3.5-flash (E2b dev run,
    2026-10-01: jev_reflect_trigger failed every episode whose reflection fired at step 1 but not at step 2,
    5/10). Without a preceding tool result (step 0, action environments) the note is a ``user`` message.
    ``messages[-1]`` is replaced, never mutated, so shallow copies of the list (counterfactual snapshots) stay
    valid."""
    if messages and messages[-1].get("role") == "tool":
        last = dict(messages[-1])
        last["content"] = f"{last.get('content') or ''}\n\n{text}"
        messages[-1] = last
    else:
        messages.append({"role": "user", "content": text})
