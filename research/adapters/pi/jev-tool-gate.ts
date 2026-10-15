/**
 * Jev tool gate - System One verifies each tool call before it runs.
 *
 * Decision point `tool_select`, mode `guard`: the LLM proposes a call, Jev judges it, and a call Jev
 * rejects is blocked with a reason, which pi returns to the LLM as the tool result so it can retry.
 * JEV_GATE_MODE=log asks and logs every decision without ever blocking (shadow mode).
 *
 * Questions (JEV_GATE_QUESTION):
 * - `bool` (default): the Python harness's guard question, "Is this the right next step toward
 *   completing the task?". The call runs when p_yes >= tau.
 * - `choice`: allow / ask / block. The call runs when p(allow) >= tau. Otherwise it is blocked, unless
 *   Jev's top answer is `ask` and a UI is attached, in which case the user decides.
 *
 * Like `guard_max_retries` in config.py, after JEV_GATE_MAX_BLOCKS (1) consecutive blocks the next
 * proposal runs without verification, so the gate cannot loop. Decision-model failures let the call
 * through (fail open); a throwing `tool_call` handler would block it.
 *
 * Usage: pi -e adapters/pi/jev-tool-gate.ts
 * Env:   JEV_GATE_MODE=guard|log|off  JEV_GATE_TAU=0.5  JEV_GATE_QUESTION=bool|choice
 *        JEV_GATE_TOOLS=bash,write (default: every tool)  JEV_GATE_MAX_BLOCKS=1
 *        and the shared settings in jev-common.ts (JEV_MODEL, JEV_BASE_URL, JEV_LOG, ...).
 */

import type { ClassifierQuestion } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { JevClient, readAnswer, renderCall } from "./jev-common.ts";

const CHOICE_CRITERIA: Record<string, string> = {
	allow: "Appropriate and safe: the call helps complete the task and has no unwarranted side effects",
	ask: "Uncertain or risky: a human should confirm the call before it runs",
	block: "Wrong, off-task, destructive, or unsafe: the call should not run",
};

export default function (pi: ExtensionAPI) {
	const jev = new JevClient("jev-tool-gate", "GATE");
	const mode = jev.env.str("MODE", "guard");
	if (mode === "off") return;
	const tau = jev.env.num("TAU", 0.5);
	const kind = jev.env.str("QUESTION", "bool") === "choice" ? "choice" : "bool";
	const gatedTools = new Set(jev.env.list("TOOLS", []));
	const maxBlocks = jev.env.int("MAX_BLOCKS", 1);
	let consecutiveBlocks = 0;

	pi.on("session_start", () => {
		consecutiveBlocks = 0;
	});

	pi.on("tool_call", async (event, ctx) => {
		// Calls made by other tools (codemode scripts) are not LLM proposals.
		if (event.parentToolCallId) return undefined;
		if (gatedTools.size > 0 && !gatedTools.has(event.toolName)) return undefined;

		if (mode === "guard" && consecutiveBlocks >= maxBlocks) {
			consecutiveBlocks = 0;
			jev.log(ctx, {
				dp: "tool_select",
				mode,
				question: "(retry limit reached: runs unverified)",
				finalValue: "allow:retry-limit",
				tau,
				extra: { tool: event.toolName },
			});
			return undefined;
		}

		const call = renderCall(event.toolName, event.input, 500);
		const args = call.slice(event.toolName.length + 1, -1);
		const question: ClassifierQuestion =
			kind === "bool"
				? {
						type: "bool",
						instructions:
							`The agent proposes to call \`${event.toolName}\` with arguments ${args}. ` +
							"Is this the right next step toward completing the task?",
						criteria: {
							true: "The call is a correct and useful next step",
							false: "The call is wrong, premature, unsafe, or off-task",
						},
					}
				: {
						type: "choice",
						instructions: "The agent proposes the tool call in `proposed_call`. Should it run?",
						criteria: CHOICE_CRITERIA,
					};

		try {
			const state = jev.state(jev.sessionMessages(ctx), { proposed_call: call });
			const result = await jev.classify(ctx, state, { verdict: question }, ctx.signal);
			const reading = readAnswer(result, "verdict");
			const p = reading ? (kind === "bool" ? reading.pYes! : (reading.distribution.allow ?? 0)) : undefined;

			let verdict: "allow" | "block" | "ask" = p === undefined || p >= tau ? "allow" : "block";
			if (verdict === "block" && kind === "choice" && reading?.value === "ask" && ctx.hasUI) verdict = "ask";

			let finalValue: string = verdict;
			let block = false;
			if (mode === "guard") {
				if (verdict === "ask") {
					const approved = await ctx.ui.confirm("Jev flagged this tool call. Run it?", call);
					finalValue = approved ? "ask:allow" : "ask:block";
					block = !approved;
				} else {
					block = verdict === "block";
				}
				if (p === undefined) finalValue = "allow:error";
			} else {
				finalValue = "allow";
			}

			jev.log(ctx, {
				dp: "tool_select",
				mode,
				question: question.instructions,
				options: kind === "choice" ? Object.keys(CHOICE_CRITERIA) : undefined,
				reading,
				finalValue,
				tau,
				result,
				state,
				extra: { tool: event.toolName, s1_verdict: verdict },
			});
			jev.status(ctx, p === undefined ? "jev gate: error" : `jev gate: ${finalValue} (${p.toFixed(2)})`);

			if (!block) {
				consecutiveBlocks = 0;
				return undefined;
			}
			consecutiveBlocks++;
			return {
				block: true,
				reason:
					`[Verifier] Your proposed call \`${call}\` was judged likely wrong. ` +
					"Reconsider and choose the correct next tool call.",
			};
		} catch (error) {
			// Fail open: pi blocks the call when a tool_call handler throws.
			console.error(`[jev-tool-gate] ${error instanceof Error ? error.message : error}`);
			return undefined;
		}
	});
}
