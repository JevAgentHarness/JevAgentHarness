/**
 * Shared helpers for the Jev System-One extensions in this directory. Not an extension itself:
 * it has no default export, so load the `jev-*.ts` files next to it, not this one.
 *
 * Each extension asks a System-One decision model typed questions (`choice`, `bool`, `score`) about a
 * JSON rendering of the agent's state, acts on the answer at one decision point of the paper's
 * design space (see `src/jevharness/harness/config.py`), and logs the decision as one JSONL record.
 *
 * The decision model is swappable without code changes:
 *
 * - `JEV_MODEL=provider/id` (default `typesafe/jev-latest`) resolves a classifier model in pi's model
 *   registry and calls `ctx.modelRegistry.classify()`. Any classifier pi knows works: TypeSafe Jev
 *   (`TYPESAFE_API_KEY`), Jev through OpenRouter or another gateway, or a llama.cpp chat model read
 *   through logprobs (`llama.cpp/<model-id>`, API `llama-cpp-classify`).
 * - `JEV_BASE_URL=http://host:port` bypasses the registry and POSTs the Jev wire format to
 *   `{base}/v1/systemone`, for OpenJev, openjev-server, or our own trained model (`serve_jev.py`).
 *   The wire `model` is `JEV_WIRE_MODEL`, else the id part of `JEV_MODEL`.
 *
 * Every setting `JEV_<NAME>` can be overridden per extension as `JEV_<SCOPE>_<NAME>`, for example
 * `JEV_GATE_TAU=0.3` next to a shared `JEV_TAU=0.5`.
 */

import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ClassifierAnswer, ClassifierQuestion, JsonObject } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";

/** Decision points of the paper (config.py `DP`), plus `compaction` for session_before_compact. */
export type DecisionPoint =
	| "tool_select"
	| "tool_params"
	| "memory_write"
	| "memory_read"
	| "reflection"
	| "stop"
	| "routing"
	| "compaction";

// ============================================================================
// Environment
// ============================================================================

export interface JevEnv {
	str(name: string, fallback: string): string;
	optional(name: string): string | undefined;
	num(name: string, fallback: number): number;
	int(name: string, fallback: number): number;
	bool(name: string, fallback: boolean): boolean;
	list(name: string, fallback: readonly string[]): string[];
}

/** Reads `JEV_<SCOPE>_<NAME>`, then `JEV_<NAME>`. Empty values count as unset. */
export function jevEnv(scope: string): JevEnv {
	const read = (name: string): string | undefined => {
		for (const key of [`JEV_${scope}_${name}`, `JEV_${name}`]) {
			const value = process.env[key]?.trim();
			if (value) return value;
		}
		return undefined;
	};
	const num = (name: string, fallback: number): number => {
		const raw = read(name);
		const value = raw === undefined ? Number.NaN : Number(raw);
		return Number.isFinite(value) ? value : fallback;
	};
	return {
		str: (name, fallback) => read(name) ?? fallback,
		optional: read,
		num,
		int: (name, fallback) => Math.trunc(num(name, fallback)),
		bool: (name, fallback) => {
			const raw = read(name)?.toLowerCase();
			return raw === undefined ? fallback : ["1", "true", "yes", "on"].includes(raw);
		},
		list: (name, fallback) => {
			const raw = read(name);
			if (raw === undefined) return [...fallback];
			return raw
				.split(",")
				.map((item) => item.trim())
				.filter(Boolean);
		},
	};
}

// ============================================================================
// State rendering
// ============================================================================

export interface StateOptions {
	/** Steps of history rendered into the state (config.py `history_window`). */
	historyWindow: number;
	/** Characters kept of the latest tool result (config.py `obs_truncate_chars`). */
	obsChars: number;
	/** Upper bound of the serialized state (config.py `s1_state_max_chars`). */
	maxChars: number;
}

/** Head and tail of long text, like the Python harness's `_trunc`. */
export function truncate(text: string, max: number): string {
	if (text.length <= max) return text;
	const half = Math.max(1, Math.floor(max / 2));
	return `${text.slice(0, half)}\n...[truncated]...\n${text.slice(-half)}`;
}

/** Plain text of message content: strings as-is, text blocks joined, images as a placeholder. */
export function textOf(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.map((block) => {
			if (block?.type === "text") return String(block.text ?? "");
			if (block?.type === "image") return "[image]";
			return "";
		})
		.filter(Boolean)
		.join("\n");
}

/** `name({"arg":...})` with the arguments cut to `max` characters. */
export function renderCall(name: string, args: unknown, max = 200): string {
	let json: string;
	try {
		json = JSON.stringify(args ?? {});
	} catch {
		json = String(args);
	}
	return `${name}(${json.length > max ? `${json.slice(0, max)}...` : json})`;
}

/** Text of the latest message the user wrote, or undefined. */
export function lastUserText(messages: readonly AgentMessage[]): string | undefined {
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i];
		if (message.role === "user") return textOf(message.content);
	}
	return undefined;
}

/**
 * The agent state a System-One model reads, mirroring the Python harness's `_s1_state`:
 * the task (latest user message), the latest compaction summary, the last steps as
 * `[i] tool(args) -> result`, and the current observation. `extra` fields are added last
 * (decision-point specifics such as `proposed_call`). Oldest history goes first when the
 * state exceeds `maxChars`.
 */
export function renderState(
	messages: readonly AgentMessage[],
	options: StateOptions,
	extra: JsonObject = {},
): JsonObject {
	let task = "";
	let summary: string | undefined;
	let observation: string | undefined;
	const steps: string[] = [];
	const stepOfCall = new Map<string, number>();
	const pending = new Set<number>(); // steps whose tool call has no result yet

	for (const message of messages) {
		switch (message.role) {
			case "user":
				task = textOf(message.content);
				break;
			case "compactionSummary":
				summary = message.summary;
				break;
			case "branchSummary":
				steps.push(`branch summary: ${truncate(message.summary, 400)}`);
				break;
			case "custom":
				steps.push(`note: ${truncate(textOf(message.content), 300)}`);
				break;
			case "bashExecution":
				steps.push(`user ran \`${message.command}\` -> ${truncate(message.output, 400)}`);
				break;
			case "assistant": {
				const text = textOf(message.content);
				let calls = 0;
				for (const block of message.content) {
					if (block.type !== "toolCall") continue;
					calls++;
					stepOfCall.set(block.id, steps.length);
					pending.add(steps.length);
					steps.push(renderCall(block.name, block.arguments));
				}
				if (calls === 0 && text) steps.push(`said: ${truncate(text, 300)}`);
				break;
			}
			case "toolResult": {
				const output = textOf(message.content);
				observation = output;
				const result = `${message.isError ? "[error] " : ""}${truncate(output, 400)}`;
				const index = stepOfCall.get(message.toolCallId);
				if (index === undefined) steps.push(`${message.toolName}(...) -> ${result}`);
				else {
					steps[index] = `${steps[index]} -> ${result}`;
					pending.delete(index);
				}
				break;
			}
		}
	}

	// Calls still waiting for their result are in flight: pi appends the assistant message before it runs
	// its tool calls, so at `tool_call` the call being judged is already in the session. Leave such calls
	// out, as the Python harness asks before a step enters the history (the gate passes the call itself as
	// `proposed_call`). Pilot finding: rendered as a result-less last step, models read the proposed call as
	// already done and rejected correct calls with p_yes ~ 0 (docs/PILOT_RESULTS.md).
	const done = steps.filter((_, i) => !pending.has(i));
	const start = Math.max(0, done.length - options.historyWindow);
	const history = done.slice(start).map((step, i) => `[${start + i}] ${step}`);
	const build = (taskText: string): JsonObject => ({
		task: taskText,
		...(summary ? { summary: truncate(summary, 3000) } : {}),
		history,
		...(observation !== undefined ? { current_observation: truncate(observation, options.obsChars) } : {}),
		...extra,
	});

	let state = build(truncate(task, 4000));
	while (history.length > 1 && JSON.stringify(state).length > options.maxChars) {
		history.shift();
		state = build(truncate(task, 4000));
	}
	if (JSON.stringify(state).length > options.maxChars) state = build(truncate(task, 1000));
	return state;
}

// ============================================================================
// Classification
// ============================================================================

export interface JevConfig {
	/** `provider/id` of a classifier model in pi's registry, e.g. `typesafe/jev-latest`. */
	model: string;
	/** Jev-compatible server. When set, requests bypass the registry. */
	baseUrl?: string;
	/** Model id sent on the wire to `baseUrl`. */
	wireModel: string;
	apiKey?: string;
	timeoutMs: number;
	maxRetries: number;
	/** Softens logprob classifiers (llama-cpp-classify); System One ignores it. */
	temperature?: number;
	/** Extra request fields, e.g. OpenJev's `{"samples": 8, "steps": 4}`. */
	extraBody?: Record<string, unknown>;
	logPath?: string;
	/** Also log the rendered state (for training-data harvest). */
	logState: boolean;
	state: StateOptions;
}

export interface JevResult {
	ok: boolean;
	answers: Record<string, ClassifierAnswer>;
	/** Score-level distributions, when the backend returns them (direct HTTP only). */
	scoreProbabilities: Record<string, Record<string, number>>;
	/** Model that answered. */
	model: string;
	latencyMs: number;
	inputTokens?: number;
	error?: string;
	/** Correlates the log records of one request that answered several questions. */
	requestId: string;
	questionCount: number;
}

/** One answer read the way the Python harness reads it (`answer_from_distribution`). */
export interface Reading {
	/** bool: `p_yes >= 0.5`; choice: the chosen label; score: the expected level. */
	value: boolean | string | number;
	confidence: number;
	distribution: Record<string, number>;
	/** bool only: probability of `true`. */
	pYes?: number;
}

export function readAnswer(result: JevResult, id: string): Reading | undefined {
	const answer = result.answers[id];
	if (!answer) return undefined;
	switch (answer.type) {
		case "bool": {
			const p = answer.probability;
			return { value: p >= 0.5, confidence: Math.max(p, 1 - p), distribution: { yes: p, no: 1 - p }, pYes: p };
		}
		case "choice":
			return { value: answer.choice, confidence: answer.confidence, distribution: answer.probabilities };
		case "score":
			return {
				value: answer.score,
				confidence: answer.confidence,
				distribution: result.scoreProbabilities[id] ?? {},
			};
	}
}

/** Labels of a distribution, most probable first. */
export function ranked(distribution: Record<string, number>): string[] {
	return Object.entries(distribution)
		.sort((a, b) => b[1] - a[1])
		.map(([label]) => label);
}

function splitModel(spec: string): { provider: string; id: string } {
	const slash = spec.indexOf("/");
	if (slash <= 0) return { provider: "typesafe", id: spec };
	return { provider: spec.slice(0, slash), id: spec.slice(slash + 1) };
}

function newRequestId(): string {
	return Math.random().toString(36).slice(2, 10);
}

function combineSignals(signal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
	const timeout = AbortSignal.timeout(timeoutMs);
	return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

const RETRY_STATUS = new Set([408, 429, 500, 502, 503, 504, 529]);

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise((resolve) => {
		const timer = setTimeout(resolve, ms);
		signal?.addEventListener(
			"abort",
			() => {
				clearTimeout(timer);
				resolve();
			},
			{ once: true },
		);
	});
}

/** Public `bool` questions become wire-level `noul` questions, as in pi's TypeSafe adapter and jev.py. */
function toWire(question: ClassifierQuestion): Record<string, unknown> {
	return question.type === "bool" ? { ...question, type: "noul" } : { ...question };
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
	return typeof value === "object" && value !== null && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: undefined;
}

function numbers(value: unknown): Record<string, number> {
	const record = asRecord(value) ?? {};
	return Object.fromEntries(
		Object.entries(record).flatMap(([key, v]) => (typeof v === "number" && Number.isFinite(v) ? [[key, v]] : [])),
	);
}

/** Jev wire answer -> pi's `ClassifierAnswer`, tolerating missing optional fields like jev.py. */
function fromWire(question: ClassifierQuestion, raw: Record<string, unknown>): ClassifierAnswer {
	if (question.type === "choice") {
		const choice = typeof raw.choice === "string" ? raw.choice : undefined;
		const probabilities = numbers(raw.probabilities);
		if (Object.keys(probabilities).length === 0 && choice) probabilities[choice] = 1;
		const best = choice ?? ranked(probabilities)[0];
		if (!best) throw new Error("choice answer without a choice");
		const confidence = typeof raw.confidence === "number" ? raw.confidence : (probabilities[best] ?? 1);
		return { type: "choice", choice: best, probabilities, confidence };
	}
	if (question.type === "bool") {
		const p = typeof raw.noul === "number" ? raw.noul : typeof raw.probability === "number" ? raw.probability : 0.5;
		return { type: "bool", probability: p };
	}
	if (typeof raw.score !== "number") throw new Error("score answer without a score");
	return { type: "score", score: raw.score, confidence: typeof raw.confidence === "number" ? raw.confidence : 0 };
}

/**
 * Client for one extension: reads its settings, calls the decision model, and writes the decision log.
 * Classification never throws; failures come back as `ok: false` so callers can fail open.
 */
export class JevClient {
	readonly extension: string;
	readonly env: JevEnv;
	readonly config: JevConfig;
	private logDirReady = false;

	constructor(extension: string, scope: string) {
		this.extension = extension;
		this.env = jevEnv(scope);
		const env = this.env;
		const model = env.str("MODEL", "typesafe/jev-latest");
		const extraBody = env.optional("EXTRA_BODY");
		let parsedExtraBody: Record<string, unknown> | undefined;
		if (extraBody) {
			try {
				parsedExtraBody = asRecord(JSON.parse(extraBody));
			} catch {
				console.error(`[${extension}] ignoring JEV_EXTRA_BODY: not a JSON object`);
			}
		}
		const temperature = env.num("TEMPERATURE", Number.NaN);
		this.config = {
			model,
			baseUrl: env.optional("BASE_URL"),
			wireModel: env.str("WIRE_MODEL", splitModel(model).id),
			apiKey: env.optional("API_KEY") ?? firstEnv("OPENJEV_API_KEY", "TYPESAFE_API_KEY"),
			timeoutMs: env.int("TIMEOUT_MS", 10_000),
			maxRetries: env.int("MAX_RETRIES", 2),
			temperature: Number.isFinite(temperature) ? temperature : undefined,
			extraBody: parsedExtraBody,
			logPath: env.optional("LOG"),
			logState: env.bool("LOG_STATE", false),
			state: {
				historyWindow: env.int("HISTORY_WINDOW", 8),
				obsChars: env.int("OBS_CHARS", 2000),
				maxChars: env.int("STATE_MAX_CHARS", 12_000),
			},
		};
	}

	/** State of the current session branch, for handlers whose event carries no messages. */
	sessionMessages(ctx: ExtensionContext): AgentMessage[] {
		return ctx.sessionManager.buildSessionProjection().messages;
	}

	state(messages: readonly AgentMessage[], extra: JsonObject = {}): JsonObject {
		return renderState(messages, this.config.state, extra);
	}

	async classify(
		ctx: ExtensionContext,
		state: JsonObject,
		questions: Record<string, ClassifierQuestion>,
		signal?: AbortSignal,
	): Promise<JevResult> {
		const started = performance.now();
		const base: JevResult = {
			ok: false,
			answers: {},
			scoreProbabilities: {},
			model: this.config.baseUrl ? this.config.wireModel : this.config.model,
			latencyMs: 0,
			requestId: newRequestId(),
			questionCount: Object.keys(questions).length,
		};
		try {
			const result = this.config.baseUrl
				? await this.classifyHttp(base, state, questions, signal)
				: await this.classifyRegistry(ctx, base, state, questions, signal);
			return { ...result, latencyMs: performance.now() - started };
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			return { ...base, latencyMs: performance.now() - started, error: message };
		}
	}

	private async classifyRegistry(
		ctx: ExtensionContext,
		base: JevResult,
		state: JsonObject,
		questions: Record<string, ClassifierQuestion>,
		signal: AbortSignal | undefined,
	): Promise<JevResult> {
		const { provider, id } = splitModel(this.config.model);
		const model = ctx.modelRegistry.findOfType("classifier", provider, id);
		if (!model) return { ...base, error: `classifier model ${this.config.model} is not in pi's catalog` };
		const extraBody = this.config.extraBody;
		const result = await ctx.modelRegistry.classify(
			model,
			{ state, questions },
			{
				signal,
				timeoutMs: this.config.timeoutMs,
				maxRetries: this.config.maxRetries,
				temperature: this.config.temperature,
				onPayload: extraBody ? (payload) => ({ ...(asRecord(payload) ?? {}), ...extraBody }) : undefined,
			},
		);
		const served = `${result.provider}/${result.model}`;
		if (result.stopReason !== "stop") {
			return {
				...base,
				model: served,
				inputTokens: result.usage?.input,
				error: result.errorMessage ?? result.stopReason,
			};
		}
		return { ...base, ok: true, answers: result.answers, model: served, inputTokens: result.usage?.input };
	}

	/** POST {base}/v1/systemone with jev.py's wire format and retry policy. */
	private async classifyHttp(
		base: JevResult,
		state: JsonObject,
		questions: Record<string, ClassifierQuestion>,
		signal: AbortSignal | undefined,
	): Promise<JevResult> {
		const root = this.config.baseUrl!.replace(/\/+$/u, "");
		const url = `${/\/v1$/u.test(root) ? root : `${root}/v1`}/systemone`;
		const body = JSON.stringify({
			state,
			model: this.config.wireModel,
			questions: Object.fromEntries(Object.entries(questions).map(([id, q]) => [id, toWire(q)])),
			...this.config.extraBody,
		});
		const headers: Record<string, string> = {
			"content-type": "application/json",
			"x-typesafe-sdk": "jevharness-pi/0.1",
		};
		if (this.config.apiKey) headers.authorization = `Bearer ${this.config.apiKey}`;

		let delay = 500;
		for (let attempt = 0; ; attempt++) {
			let response: Response;
			try {
				response = await fetch(url, {
					method: "POST",
					headers: { ...headers, "x-typesafe-retry-count": String(attempt) },
					body,
					signal: combineSignals(signal, this.config.timeoutMs),
				});
			} catch (error) {
				if (signal?.aborted || attempt >= this.config.maxRetries) throw error;
				await sleep(Math.min(5000, delay) + Math.random() * 250, signal);
				delay *= 2;
				continue;
			}
			if (response.ok) {
				const data = asRecord(await response.json()) ?? {};
				const wireAnswers = asRecord(data.answers) ?? {};
				const answers: Record<string, ClassifierAnswer> = {};
				const scoreProbabilities: Record<string, Record<string, number>> = {};
				for (const [id, question] of Object.entries(questions)) {
					const raw = asRecord(wireAnswers[id]);
					if (!raw) throw new Error(`Jev server returned no answer for ${id}`);
					answers[id] = fromWire(question, raw);
					if (question.type === "score") scoreProbabilities[id] = numbers(raw.probabilities);
				}
				const usage = asRecord(data.usage);
				return {
					...base,
					ok: true,
					answers,
					scoreProbabilities,
					model: typeof data.model === "string" ? data.model : this.config.wireModel,
					inputTokens: typeof usage?.input_tokens === "number" ? usage.input_tokens : undefined,
				};
			}
			const text = await response.text();
			if (!RETRY_STATUS.has(response.status) || attempt >= this.config.maxRetries) {
				throw new Error(`Jev API error ${response.status}: ${text.slice(0, 500)}`);
			}
			const retryAfterMs = Number(response.headers.get("retry-after-ms"));
			const retryAfter = Number(response.headers.get("retry-after"));
			const wait =
				Number.isFinite(retryAfterMs) && retryAfterMs > 0
					? retryAfterMs
					: retryAfter > 0
						? retryAfter * 1000
						: delay;
			await sleep(Math.min(5000, wait) + Math.random() * 250, signal);
			delay *= 2;
		}
	}

	// ==========================================================================
	// Decision log
	// ==========================================================================

	/**
	 * Append one decision to `$JEV_LOG` with the fields shared by all harness adapters:
	 * `{ts, harness, session, dp, mode, question, options, s1_value, s1_conf, s1_dist, final_value,
	 * latency_ms, input_tokens, model}`, plus `extension`, `tau`, `request_id`, `n_questions`, and
	 * `error` when the model failed. Records of one request share `request_id`, `latency_ms`, and
	 * `input_tokens`; deduplicate by `request_id` before summing cost.
	 */
	log(
		ctx: ExtensionContext,
		record: {
			dp: DecisionPoint;
			mode: string;
			question: string;
			options?: readonly string[];
			reading?: Reading;
			finalValue: unknown;
			tau?: number;
			result?: JevResult;
			state?: JsonObject;
			extra?: Record<string, unknown>;
		},
	): void {
		const path = this.config.logPath;
		if (!path) return;
		const { result, reading } = record;
		const line = {
			...record.extra,
			ts: new Date().toISOString(),
			harness: "pi",
			session: safeSessionId(ctx),
			extension: this.extension,
			dp: record.dp,
			mode: record.mode,
			question: record.question,
			options: record.options ?? null,
			s1_value: reading?.value ?? null,
			s1_conf: reading?.confidence ?? null,
			s1_dist: reading?.distribution ?? null,
			final_value: record.finalValue ?? null,
			tau: record.tau ?? null,
			latency_ms: result ? Math.round(result.latencyMs * 10) / 10 : null,
			input_tokens: result?.inputTokens ?? null,
			model: result?.model ?? null,
			request_id: result?.requestId ?? null,
			n_questions: result?.questionCount ?? null,
			...(result?.error ? { error: result.error } : {}),
			...(this.config.logState && record.state ? { state: record.state } : {}),
		};
		try {
			if (!this.logDirReady) {
				mkdirSync(dirname(path), { recursive: true });
				this.logDirReady = true;
			}
			appendFileSync(path, `${JSON.stringify(line)}\n`);
		} catch (error) {
			console.error(`[${this.extension}] cannot write ${path}: ${error instanceof Error ? error.message : error}`);
		}
	}

	/** Footer status in interactive mode; no-op elsewhere. */
	status(ctx: ExtensionContext, text: string | undefined): void {
		if (ctx.hasUI) ctx.ui.setStatus(this.extension, text);
	}
}

function firstEnv(...names: string[]): string | undefined {
	for (const name of names) {
		const value = process.env[name]?.trim();
		if (value) return value;
	}
	return undefined;
}

function safeSessionId(ctx: ExtensionContext): string | null {
	try {
		return ctx.sessionManager.getSessionId();
	} catch {
		return null;
	}
}
