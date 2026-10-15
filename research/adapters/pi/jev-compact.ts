/**
 * Jev compaction - System One decides which messages survive compaction.
 *
 * On `session_before_compact`, Jev scores each message pi is about to summarize (both
 * `messagesToSummarize` and, for a split turn such as a single long task prompt, `turnPrefixMessages`):
 * "How important is message `mN` for continuing the task?" on four levels (drop, minor, useful,
 * critical). Messages with an expected score below JEV_COMPACT_MIN_SCORE are dropped. Messages the user
 * wrote are always kept. The decision point is logged as `compaction`, an addition to the paper's list.
 *
 * - `prefilter` (default): the dropped messages are removed from the preparation's message lists, and
 *   pi's default LLM summarizer runs on the rest, so the summary request is smaller and focused on what
 *   Jev kept. This edits the preparation object in place, which pi 0.99 passes on to its default
 *   compaction (agent-session.ts) but does not document as an extension contract.
 * - `extractive`: no LLM call. The summary is the task, the previous summary, and excerpts of the kept
 *   messages; System One replaces the summarizer.
 * - `log`: scores and logs, and lets pi compact as usual.
 *
 * Messages are scored in batches of JEV_COMPACT_BATCH per request, each shown in the state truncated to
 * JEV_COMPACT_MESSAGE_CHARS. A decision-model failure falls back to pi's default compaction.
 *
 * Usage: pi -e adapters/pi/jev-compact.ts   (runs on /compact and on automatic compaction)
 * Env:   JEV_COMPACT_MODE=prefilter|extractive|log|off  JEV_COMPACT_MIN_SCORE=1.0  JEV_COMPACT_BATCH=32
 *        JEV_COMPACT_MESSAGE_CHARS=800  JEV_COMPACT_MIN_MESSAGES=4  and the shared settings in jev-common.ts.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ClassifierQuestion, JsonObject } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { JevClient, lastUserText, readAnswer, renderCall, textOf, truncate } from "./jev-common.ts";

const LEVELS = [
	"drop: not needed to continue",
	"minor: a detail that is unlikely to matter",
	"useful: context worth keeping",
	"critical: needed to continue the task",
];

/** One line per message, as the summarizer would read it. */
function renderMessage(message: AgentMessage, max: number): string {
	switch (message.role) {
		case "user":
			return `[user] ${truncate(textOf(message.content), max)}`;
		case "assistant": {
			const text = textOf(message.content);
			const calls = message.content.flatMap((block) =>
				block.type === "toolCall" ? [renderCall(block.name, block.arguments)] : [],
			);
			return truncate(`[assistant] ${[text, ...calls].filter(Boolean).join(" ; ")}`, max);
		}
		case "toolResult":
			return `[tool result ${message.toolName}${message.isError ? ", error" : ""}] ${truncate(textOf(message.content), max)}`;
		case "custom":
			return `[note ${message.customType}] ${truncate(textOf(message.content), max)}`;
		case "bashExecution":
			return truncate(`[user ran ${message.command}] ${message.output}`, max);
		case "branchSummary":
		case "compactionSummary":
			return `[summary] ${truncate(message.summary, max)}`;
		default:
			return `[${message.role}]`;
	}
}

export default function (pi: ExtensionAPI) {
	const jev = new JevClient("jev-compact", "COMPACT");
	const mode = jev.env.str("MODE", "prefilter");
	if (mode === "off") return;
	const minScore = jev.env.num("MIN_SCORE", 1.0);
	const batch = Math.max(1, jev.env.int("BATCH", 32));
	const messageChars = jev.env.int("MESSAGE_CHARS", 800);
	const minMessages = jev.env.int("MIN_MESSAGES", 4);

	pi.on("session_before_compact", async (event, ctx) => {
		const { preparation, signal } = event;
		// With a single long prompt (one user-message span), pi summarizes a split turn: the messages
		// are in turnPrefixMessages and messagesToSummarize is empty. Score both.
		const lists = [preparation.messagesToSummarize, preparation.turnPrefixMessages];
		const messages = lists.flat();
		if (messages.length < minMessages) return;
		const task = lastUserText(jev.sessionMessages(ctx)) ?? lastUserText(messages) ?? "";

		const scores: (number | undefined)[] = [];
		for (let start = 0; start < messages.length; start += batch) {
			const window = messages.slice(start, start + batch);
			const shown: JsonObject = {};
			const questions: Record<string, ClassifierQuestion> = {};
			window.forEach((message, i) => {
				const id = `m${start + i}`;
				shown[id] = renderMessage(message, messageChars);
				questions[id] = {
					type: "score",
					instructions: `How important is message \`${id}\` in \`messages\` for continuing the task after the conversation is compacted?`,
					criteria: LEVELS,
				};
			});
			const state: JsonObject = {
				task: truncate(task, 4000),
				...(preparation.previousSummary ? { previous_summary: truncate(preparation.previousSummary, 3000) } : {}),
				messages: shown,
			};
			const result = await jev.classify(ctx, state, questions, signal);
			if (!result.ok) {
				jev.log(ctx, {
					dp: "compaction",
					mode,
					question: "(batch failed: default compaction)",
					finalValue: "default",
					result,
				});
				return;
			}
			window.forEach((message, i) => {
				const id = `m${start + i}`;
				const reading = readAnswer(result, id);
				const score = reading?.value as number | undefined;
				scores.push(score);
				const keep = message.role === "user" || score === undefined || score >= minScore;
				jev.log(ctx, {
					dp: "compaction",
					mode,
					question: questions[id].instructions,
					options: LEVELS,
					reading,
					finalValue: keep ? "keep" : "drop",
					tau: minScore,
					result,
					state: i === 0 ? state : undefined,
					extra: { role: message.role, message: shown[id] },
				});
			});
		}

		const keeps = messages.map(
			(message, i) => message.role === "user" || scores[i] === undefined || scores[i]! >= minScore,
		);
		const kept = messages.filter((_, i) => keeps[i]);
		jev.status(ctx, `jev compact: kept ${kept.length}/${messages.length}`);
		if (mode === "log" || kept.length === 0) return;

		if (mode === "prefilter") {
			let offset = 0;
			for (const list of lists) {
				const survivors = list.filter((_, i) => keeps[offset + i]);
				offset += list.length;
				list.splice(0, list.length, ...survivors);
			}
			return;
		}

		const summary = [
			"## Goal",
			truncate(task, 2000),
			...(preparation.previousSummary ? ["", "## Earlier summary", preparation.previousSummary] : []),
			"",
			"## Kept context (selected by the System-One model)",
			...kept.map((message) => `- ${renderMessage(message, messageChars).replace(/\n/gu, " ")}`),
		].join("\n");
		return {
			compaction: {
				summary,
				firstKeptEntryId: preparation.firstKeptEntryId,
				tokensBefore: preparation.tokensBefore,
				details: { jev: { kept: kept.length, dropped: messages.length - kept.length, minScore } },
			},
		};
	});
}
