/**
 * Jev active tools - System One narrows the tools declared to the LLM.
 *
 * Decision point `tool_select`, mode `assist`: before each LLM request Jev ranks the candidate tools and
 * pi declares only the top k plus an always-keep list, so the LLM chooses among fewer tools. Candidates
 * are the tools active when the session starts, or JEV_TOOLS_CANDIDATES.
 *
 * - `choice` (default): one Choice question, "Which tool should the agent call next?", over the
 *   candidates with their descriptions; the k most probable are kept (the Python harness's assist mode).
 * - `score`: one Score question per tool ("How useful is `tool` for the next step?"); the k highest
 *   expected scores are kept.
 *
 * pi reads the active tools when it prepares a request, so the selection runs at `before_agent_start`
 * (the first request of a prompt) and at `turn_end` (every following request). Changing the declared
 * tools appends a tool change to the transcript; providers that cannot express it get a full transcript
 * checkpoint, which can cost a prompt-cache miss, so the set is only replaced when it differs.
 * JEV_TOOLS_MODE=log ranks and logs without changing the tools. Decision-model failures restore every
 * candidate (fail open).
 *
 * Usage: pi -e adapters/pi/jev-active-tools.ts
 * Env:   JEV_TOOLS_MODE=assist|log|off  JEV_TOOLS_K=3  JEV_TOOLS_KEEP=bash  JEV_TOOLS_METHOD=choice|score
 *        JEV_TOOLS_CANDIDATES=read,bash,edit,write,grep,find,ls  and the shared settings in jev-common.ts.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ClassifierQuestion } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { JevClient, type Reading, ranked, readAnswer, truncate } from "./jev-common.ts";

const QUESTION = "Which tool should the agent call next?";
const SCORE_LEVELS = ["not useful", "might help", "likely needed", "needed now"];

export default function (pi: ExtensionAPI) {
	const jev = new JevClient("jev-active-tools", "TOOLS");
	const mode = jev.env.str("MODE", "assist");
	if (mode === "off") return;
	const k = jev.env.int("K", 3);
	const keep = new Set(jev.env.list("KEEP", []));
	const method = jev.env.str("METHOD", "choice") === "score" ? "score" : "choice";
	let candidates: string[] = [];

	pi.on("session_start", () => {
		const configured = jev.env.list("CANDIDATES", []);
		candidates = configured.length > 0 ? configured : pi.getActiveTools();
	});

	function describe(name: string): string {
		const description = pi.getAllTools().find((tool) => tool.name === name)?.description ?? "";
		return truncate(description.split("\n")[0] ?? "", 200) || name;
	}

	/** Rank the candidates: a Choice distribution, or expected scores keyed by tool name. */
	async function rank(ctx: ExtensionContext, messages: readonly AgentMessage[]) {
		const state = jev.state(messages);
		const questions: Record<string, ClassifierQuestion> = {};
		if (method === "choice") {
			questions.tool = {
				type: "choice",
				instructions: QUESTION,
				criteria: Object.fromEntries(candidates.map((name) => [name, describe(name)])),
			};
		} else {
			candidates.forEach((name, i) => {
				questions[`t${i}`] = {
					type: "score",
					instructions: `How useful is the tool \`${name}\` for the agent's next step? ${describe(name)}`,
					criteria: SCORE_LEVELS,
				};
			});
		}
		const result = await jev.classify(ctx, state, questions, ctx.signal);
		let reading: Reading | undefined;
		if (result.ok && method === "choice") {
			reading = readAnswer(result, "tool");
		} else if (result.ok) {
			const scores = Object.fromEntries(
				candidates.map((name, i) => [name, readAnswer(result, `t${i}`)?.value as number]),
			);
			reading = { value: ranked(scores)[0] ?? "", confidence: 1, distribution: scores };
		}
		return { state, result, reading };
	}

	async function select(ctx: ExtensionContext, messages: readonly AgentMessage[], trigger: string): Promise<void> {
		// Nothing to prune when every candidate fits in the top k plus the always-keep list.
		if (candidates.filter((name) => !keep.has(name)).length <= k) return;
		const { state, result, reading } = await rank(ctx, messages);
		const top = new Set(reading ? ranked(reading.distribution).slice(0, k) : candidates);
		const next = candidates.filter((name) => keep.has(name) || top.has(name));
		const active = pi.getActiveTools();
		const changed = next.length !== active.length || next.some((name) => !active.includes(name));
		if (mode === "assist" && changed) pi.setActiveTools(next);

		jev.log(ctx, {
			dp: "tool_select",
			mode,
			question: method === "choice" ? QUESTION : "How useful is each tool for the agent's next step?",
			options: candidates,
			reading,
			finalValue: mode === "assist" ? next : active,
			result,
			state,
			extra: { k, method, trigger, s1_top_k: [...top] },
		});
		jev.status(ctx, `jev tools: ${next.join(",")}`);
	}

	pi.on("before_agent_start", async (event, ctx) => {
		const prompt: AgentMessage = { role: "user", content: event.prompt, timestamp: Date.now() };
		await select(ctx, [...jev.sessionMessages(ctx), prompt], "prompt");
	});

	pi.on("turn_end", async (event, ctx) => {
		// A turn without tool calls ends the run; its next request (if any) keeps the current tools.
		if (event.toolResults.length === 0) return;
		await select(ctx, event.context.contextMessages, "turn");
	});
}
