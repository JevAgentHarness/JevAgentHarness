/**
 * Jev memory - System One decides what to remember and what to recall.
 *
 * - `memory_write`, mode `trigger`: after each turn with tool results (`turn_end`), Jev answers for each
 *   result "Will this information likely be needed later to complete the task?". Results with
 *   p_yes >= tau become notes (`tool(args) -> output`). Notes are stored as `jev-memory` session
 *   entries, so they follow the session tree, and with JEV_MEMORY_FILE also in a JSONL file that later
 *   sessions load.
 * - `memory_read`, mode `assist`: before each LLM request (`context`), Jev picks up to k notes relevant
 *   to the current step: a Choice over the notes plus "none", keeping the most probable k (a tournament
 *   when there are more notes than Jev accepts options). The notes are appended to that request only,
 *   as a `[Memory]` message, so they never accumulate in the transcript.
 *
 * Reference modes for the paper's baselines: JEV_MEMORY_WRITE_MODE=always stores every result, and
 * JEV_MEMORY_READ_MODE=always injects the JEV_MEMORY_RECENCY_K most recent notes. `off` disables a side;
 * `log` asks and logs without storing or injecting. Decision-model failures store and inject nothing.
 * The LLM already sees its full history until compaction, so memory matters most on long tasks.
 *
 * Usage: pi -e adapters/pi/jev-memory.ts
 * Env:   JEV_MEMORY_WRITE_MODE=trigger|always|log|off  JEV_MEMORY_READ_MODE=assist|always|log|off
 *        JEV_MEMORY_TAU=0.5  JEV_MEMORY_K=3  JEV_MEMORY_RECENCY_K=20  JEV_MEMORY_MAX_NOTES=500
 *        JEV_MEMORY_FILE=/path/notes.jsonl  and the shared settings in jev-common.ts.
 */

import { appendFileSync, existsSync, readFileSync } from "node:fs";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ClassifierQuestion } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, SessionEntry } from "@earendil-works/pi-coding-agent";
import { JevClient, ranked, readAnswer, renderCall, textOf, truncate } from "./jev-common.ts";

const ENTRY_TYPE = "jev-memory";
const RECALL_TYPE = "jev-memory-recall";
const WRITE_QUESTION =
	"Will this information likely be needed later to complete the task " +
	"(facts, IDs, constraints, user preferences, results)?";
const READ_QUESTION = "Which stored memory is most relevant to the current step?";
const NONE = "none";
/** Jev accepts at most 255 options; one is reserved for "none". */
const CHUNK = 254;

interface Note {
	id: number;
	text: string;
	tool: string;
	session: string | null;
	ts: string;
}

export default function (pi: ExtensionAPI) {
	const jev = new JevClient("jev-memory", "MEMORY");
	const writeMode = jev.env.str("WRITE_MODE", "trigger");
	const readMode = jev.env.str("READ_MODE", "assist");
	if (writeMode === "off" && readMode === "off") return;
	const tau = jev.env.num("TAU", 0.5);
	const k = jev.env.int("K", 3);
	const recencyK = jev.env.int("RECENCY_K", 20);
	const maxNotes = jev.env.int("MAX_NOTES", 500);
	const file = jev.env.optional("FILE");

	let notes: Note[] = [];
	let nextId = 0;
	/** Recall cache, so a retried request does not ask again. */
	let recallCache: { key: string; notes: Note[] } | undefined;

	function add(note: Note): void {
		notes.push(note);
		nextId = Math.max(nextId, note.id + 1);
		if (notes.length > maxNotes) notes.shift();
	}

	function restore(ctx: ExtensionContext): void {
		notes = [];
		nextId = 0;
		recallCache = undefined;
		const seen = new Set<string>();
		const fromFile: Note[] = [];
		if (file && existsSync(file)) {
			for (const line of readFileSync(file, "utf8").split("\n")) {
				try {
					if (line.trim()) fromFile.push(JSON.parse(line) as Note);
				} catch {
					// Skip malformed lines.
				}
			}
		}
		const fromBranch = ctx.sessionManager
			.getBranch()
			.flatMap((entry: SessionEntry) =>
				entry.type === "custom" && entry.customType === ENTRY_TYPE ? [entry.data as Note] : [],
			);
		for (const note of [...fromFile, ...fromBranch]) {
			if (!note?.text || seen.has(note.text)) continue;
			seen.add(note.text);
			add({ ...note, id: nextId });
		}
	}

	pi.on("session_start", (_event, ctx) => restore(ctx));
	pi.on("session_tree", (_event, ctx) => restore(ctx));

	// ------------------------------------------------------------------ memory_write
	pi.on("turn_end", async (event, ctx) => {
		if (writeMode === "off" || event.toolResults.length === 0) return;
		const message = event.message;
		const calls = new Map<string, string>();
		if (message.role === "assistant") {
			for (const block of message.content) {
				if (block.type === "toolCall") calls.set(block.id, renderCall(block.name, block.arguments));
			}
		}
		const candidates = event.toolResults.map((result) => ({
			tool: result.toolName,
			text: `${calls.get(result.toolCallId) ?? `${result.toolName}(...)`} -> ${truncate(textOf(result.content), 600)}`,
		}));

		let keep: boolean[];
		if (writeMode === "always") {
			keep = candidates.map(() => true);
			for (const candidate of candidates) {
				jev.log(ctx, {
					dp: "memory_write",
					mode: writeMode,
					question: WRITE_QUESTION,
					finalValue: true,
					extra: { candidate: candidate.text },
				});
			}
		} else {
			const state = jev.state(event.context.contextMessages);
			const questions: Record<string, ClassifierQuestion> = {};
			candidates.forEach((candidate, i) => {
				questions[`c${i}`] = {
					type: "bool",
					instructions: `CANDIDATE MEMORY:\n${candidate.text}\n\n${WRITE_QUESTION}`,
					criteria: { true: "Worth remembering", false: "Not needed later" },
				};
			});
			const result = await jev.classify(ctx, state, questions, ctx.signal);
			keep = candidates.map((candidate, i) => {
				const reading = readAnswer(result, `c${i}`);
				const store = writeMode === "trigger" && reading !== undefined && reading.pYes! >= tau;
				jev.log(ctx, {
					dp: "memory_write",
					mode: writeMode,
					question: WRITE_QUESTION,
					reading,
					finalValue: store,
					tau,
					result,
					state: i === 0 ? state : undefined,
					extra: { candidate: candidate.text },
				});
				return store;
			});
		}

		const drafts = candidates.flatMap((candidate, i) => {
			if (!keep[i]) return [];
			const note: Note = {
				id: nextId,
				text: candidate.text,
				tool: candidate.tool,
				session: ctx.sessionManager.getSessionId(),
				ts: new Date().toISOString(),
			};
			add(note);
			if (file) {
				try {
					appendFileSync(file, `${JSON.stringify(note)}\n`);
				} catch (error) {
					console.error(`[jev-memory] cannot write ${file}: ${error instanceof Error ? error.message : error}`);
				}
			}
			return [{ type: "custom" as const, customType: ENTRY_TYPE, data: note }];
		});
		if (drafts.length === 0) return;
		return { entries: [...event.entries, ...drafts] };
	});

	// ------------------------------------------------------------------ memory_read
	async function choose(ctx: ExtensionContext, messages: readonly AgentMessage[], pool: Note[]): Promise<Note[]> {
		if (pool.length <= k) return pool;
		const state = jev.state(messages);
		const criteria: Record<string, string> = Object.fromEntries(
			pool.map((note) => [`m${note.id}`, truncate(note.text, 280)]),
		);
		criteria[NONE] = "None of these memories is relevant";
		const result = await jev.classify(
			ctx,
			state,
			{ memory: { type: "choice", instructions: READ_QUESTION, criteria } },
			ctx.signal,
		);
		const reading = readAnswer(result, "memory");
		const byLabel = new Map(pool.map((note) => [`m${note.id}`, note]));
		const picked = reading
			? ranked(reading.distribution)
					.slice(0, k + 1)
					.filter((label) => label !== NONE)
					.slice(0, k)
					.flatMap((label) => byLabel.get(label) ?? [])
			: [];
		jev.log(ctx, {
			dp: "memory_read",
			mode: readMode,
			question: READ_QUESTION,
			options: Object.keys(criteria),
			reading,
			finalValue: picked.map((note) => `m${note.id}`),
			result,
			state,
			extra: { k, pool: pool.length },
		});
		return picked;
	}

	async function recall(ctx: ExtensionContext, messages: readonly AgentMessage[]): Promise<Note[]> {
		if (readMode === "always") return notes.slice(-recencyK);
		let pool = notes;
		while (pool.length > CHUNK) {
			const winners: Note[] = [];
			for (let i = 0; i < pool.length; i += CHUNK)
				winners.push(...(await choose(ctx, messages, pool.slice(i, i + CHUNK))));
			if (winners.length >= pool.length) break;
			pool = winners;
		}
		return choose(ctx, messages, pool);
	}

	pi.on("context", async (event, ctx) => {
		if (readMode === "off" || notes.length === 0) return;
		const key = `${event.messages.length}:${notes.length}:${nextId}`;
		const recalled = recallCache?.key === key ? recallCache.notes : await recall(ctx, event.messages);
		recallCache = { key, notes: recalled };
		if (readMode === "log" || recalled.length === 0) return;
		const content = `[Memory]\n${recalled.map((note) => `- [m${note.id}] ${note.text}`).join("\n")}`;
		const memory: AgentMessage = {
			role: "custom",
			customType: RECALL_TYPE,
			content,
			display: false,
			timestamp: Date.now(),
		};
		return { messages: [...event.messages, memory] };
	});
}
