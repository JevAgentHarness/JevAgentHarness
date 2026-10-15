/**
 * Jev reflection trigger - System One decides when the agent should pause and reflect.
 *
 * Decision point `reflection`, mode `trigger`: after a turn in which a tool call failed or repeated an
 * earlier call (`turn_end`), Jev answers the Python harness's question, "Is the agent stuck, repeating
 * itself, or did the last action fail or move away from the goal?". At p_yes >= tau the extension
 * appends a reflection prompt, which the LLM answers at the start of its next response.
 *
 * The Python harness spends an extra LLM call on the reflection; here the working LLM reflects inline,
 * so a reflection costs output tokens but no extra request. The prompt asks it to continue afterwards,
 * because a response without tool calls would end the run.
 *
 * - JEV_REFLECT_EVERY_TURN=1 asks after every turn with tool calls, not only after a failure or repeat.
 * - JEV_REFLECT_MODE=always reflects on every signal without asking (the paper's `always` reference);
 *   `log` asks and logs without injecting.
 * - JEV_REFLECT_COOLDOWN turns (default 2) must pass between reflections.
 * - Decision-model failures inject nothing.
 *
 * Usage: pi -e adapters/pi/jev-reflect.ts
 * Env:   JEV_REFLECT_MODE=trigger|always|log|off  JEV_REFLECT_TAU=0.5  JEV_REFLECT_EVERY_TURN=0
 *        JEV_REFLECT_REPEAT_WINDOW=6  JEV_REFLECT_COOLDOWN=2  JEV_REFLECT_PROMPT="<custom text>"
 *        and the shared settings in jev-common.ts.
 */

import type { ClassifierQuestion } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { JevClient, readAnswer, renderCall } from "./jev-common.ts";

const QUESTION: ClassifierQuestion = {
	type: "bool",
	instructions: "Is the agent stuck, repeating itself, or did the last action fail or move away from the goal?",
	criteria: {
		true: "Stuck, looping, failing, or drifting from the goal",
		false: "Making steady progress toward the goal",
	},
};

const DEFAULT_PROMPT =
	"[Reflection] Pause before your next tool call. In at most 4 sentences: assess progress toward the task, " +
	"identify any mistake so far, and state the best next step. Then continue working.";

export default function (pi: ExtensionAPI) {
	const jev = new JevClient("jev-reflect", "REFLECT");
	const mode = jev.env.str("MODE", "trigger");
	if (mode === "off") return;
	const tau = jev.env.num("TAU", 0.5);
	const everyTurn = jev.env.bool("EVERY_TURN", false);
	const repeatWindow = jev.env.int("REPEAT_WINDOW", 6);
	const cooldown = jev.env.int("COOLDOWN", 2);
	const prompt = jev.env.str("PROMPT", DEFAULT_PROMPT);
	/** Recent calls as `tool(args)`, to detect repeats. */
	let recentCalls: string[] = [];
	let turnsSinceReflection = Number.POSITIVE_INFINITY;

	pi.on("before_agent_start", () => {
		recentCalls = [];
		turnsSinceReflection = Number.POSITIVE_INFINITY;
	});

	pi.on("turn_end", async (event, ctx) => {
		if (event.toolResults.length === 0) return;
		turnsSinceReflection++;

		const calls =
			event.message.role === "assistant"
				? event.message.content.flatMap((block) =>
						block.type === "toolCall" ? [renderCall(block.name, block.arguments, 1000)] : [],
					)
				: [];
		const failed = event.toolResults.some((result) => result.isError);
		const repeated = calls.some((call) => recentCalls.includes(call));
		recentCalls = [...recentCalls, ...calls].slice(-repeatWindow);

		if (!(failed || repeated || everyTurn) || turnsSinceReflection <= cooldown) return;
		const signals = { failed, repeated, turn: event.turnIndex };

		let fire = mode === "always";
		if (mode !== "always") {
			const state = jev.state(event.context.contextMessages);
			const result = await jev.classify(ctx, state, { stuck: QUESTION }, ctx.signal);
			const reading = readAnswer(result, "stuck");
			fire = mode === "trigger" && reading !== undefined && reading.pYes! >= tau;
			jev.log(ctx, {
				dp: "reflection",
				mode,
				question: QUESTION.instructions,
				reading,
				finalValue: fire,
				tau,
				result,
				state,
				extra: signals,
			});
		} else {
			jev.log(ctx, { dp: "reflection", mode, question: QUESTION.instructions, finalValue: true, extra: signals });
		}
		if (!fire) return;

		turnsSinceReflection = 0;
		return {
			entries: [
				...event.entries,
				{ type: "custom_message", customType: "jev-reflect", content: prompt, display: true },
			],
		};
	});
}
