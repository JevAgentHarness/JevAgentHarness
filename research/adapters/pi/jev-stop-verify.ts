/**
 * Jev stop verifier - System One checks that the task is done before the agent stops.
 *
 * Decision point `stop`:
 * - `guard` (default): when a run is about to settle after a final answer (`agent_before_settle`), Jev
 *   answers "Is the task actually complete?". Below tau the extension appends a verification nudge and
 *   returns `continue: true`, so pi makes one more request. At most JEV_STOP_MAX_NUDGES (2) nudges per
 *   user prompt, so the agent always stops eventually.
 * - `trigger`: after each turn with tool calls (`turn_end`), Jev answers "Has the task already been
 *   fully completed, so the agent should stop now?". At or above tau the extension tells the LLM to stop
 *   calling tools and answer (early stop), once per prompt.
 * - `log`: asks the guard question when the agent settles and logs it, without nudging.
 *
 * Decision-model failures accept the stop (fail open).
 *
 * Usage: pi -e adapters/pi/jev-stop-verify.ts
 * Env:   JEV_STOP_MODE=guard|trigger|log|off  JEV_STOP_TAU=0.5  JEV_STOP_MAX_NUDGES=2
 *        JEV_STOP_NUDGE="<custom nudge text>"  and the shared settings in jev-common.ts.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ClassifierQuestion } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { JevClient, readAnswer, textOf, truncate } from "./jev-common.ts";

const GUARD_QUESTION: ClassifierQuestion = {
	type: "bool",
	instructions: "The agent wants to stop and submit. Is the task actually complete?",
	criteria: {
		true: "Every requirement of the task is met, and the result was checked",
		false: "Work remains, or the result is wrong or unverified",
	},
};

const TRIGGER_QUESTION: ClassifierQuestion = {
	type: "bool",
	instructions: "Has the task already been fully completed, so the agent should stop now?",
	criteria: {
		true: "The task is complete; further tool calls are unnecessary",
		false: "Work remains",
	},
};

const DEFAULT_NUDGE =
	"[Verifier] The task does not look complete yet. Re-read the task, check every requirement against " +
	"the current state (run the tests or inspect the outputs), and continue working before you finish.";

const EARLY_STOP = "[Verifier] The task appears complete. Stop calling tools and give your final answer.";

export default function (pi: ExtensionAPI) {
	const jev = new JevClient("jev-stop-verify", "STOP");
	const mode = jev.env.str("MODE", "guard");
	if (mode === "off") return;
	const tau = jev.env.num("TAU", 0.5);
	const maxNudges = jev.env.int("MAX_NUDGES", 2);
	const nudge = jev.env.str("NUDGE", DEFAULT_NUDGE);
	let nudges = 0;
	let earlyStopSent = false;

	// Budgets are per user prompt.
	pi.on("before_agent_start", () => {
		nudges = 0;
		earlyStopSent = false;
	});

	pi.on("agent_before_settle", async (event, ctx) => {
		if (mode !== "guard" && mode !== "log") return;
		// Only a run that ended with a final answer; another handler may already continue it.
		if (event.outcome !== "completed" || event.continue) return;
		const messages = event.context.contextMessages;
		const last = messages.at(-1);
		if (last?.role !== "assistant") return;

		if (mode === "guard" && nudges >= maxNudges) {
			jev.log(ctx, {
				dp: "stop",
				mode,
				question: GUARD_QUESTION.instructions,
				finalValue: "stop:nudge-limit",
				tau,
				extra: { nudges },
			});
			return;
		}

		const state = jev.state(messages, { final_response: truncate(textOf(last.content), 2000) });
		const result = await jev.classify(ctx, state, { done: GUARD_QUESTION }, ctx.signal);
		const reading = readAnswer(result, "done");
		const reject = mode === "guard" && reading !== undefined && reading.pYes! < tau;
		jev.log(ctx, {
			dp: "stop",
			mode,
			question: GUARD_QUESTION.instructions,
			reading,
			finalValue: reject ? "continue" : reading ? "stop" : "stop:error",
			tau,
			result,
			state,
			extra: { nudges },
		});
		jev.status(ctx, reading ? `jev stop: p_done=${reading.pYes!.toFixed(2)}` : "jev stop: error");
		if (!reject) return;

		nudges++;
		return {
			entries: [
				...event.entries,
				{ type: "custom_message", customType: "jev-stop-verify", content: nudge, display: true },
			],
			continue: true,
		};
	});

	pi.on("turn_end", async (event, ctx) => {
		if (mode !== "trigger" || earlyStopSent || event.toolResults.length === 0) return;
		const messages: readonly AgentMessage[] = event.context.contextMessages;
		const state = jev.state(messages);
		const result = await jev.classify(ctx, state, { done: TRIGGER_QUESTION }, ctx.signal);
		const reading = readAnswer(result, "done");
		const stop = reading !== undefined && reading.pYes! >= tau;
		jev.log(ctx, {
			dp: "stop",
			mode,
			question: TRIGGER_QUESTION.instructions,
			reading,
			finalValue: stop ? "stop" : reading ? "continue" : "continue:error",
			tau,
			result,
			state,
			extra: { turn: event.turnIndex },
		});
		if (!stop) return;

		earlyStopSent = true;
		return {
			entries: [
				...event.entries,
				{ type: "custom_message", customType: "jev-stop-trigger", content: EARLY_STOP, display: true },
			],
		};
	});
}
