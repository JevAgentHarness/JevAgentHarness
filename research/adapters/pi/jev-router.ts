/**
 * Jev router - a virtual model that routes each request between a strong and a cheap model.
 *
 * Decision point `routing`, mode `trigger`: before each request of the agent loop, Jev answers the
 * Python harness's routing question, "Does the next step require careful multi-step reasoning,
 * planning, or recovery from an error (i.e., a strong model) rather than a routine action?". At
 * p >= tau the request goes to the strong model, otherwise to the cheap one. Recent tool errors
 * escalate without asking: JEV_ROUTER_ERROR_ESCALATE failed results among the last
 * JEV_ROUTER_ERROR_WINDOW tool results route to the strong model.
 *
 * This extends pi's official `examples/extensions/jev-router.ts`, which classifies the first prompt
 * once and switches models once per session. Routing per request trades prompt-cache hits for
 * cheaper steps; after a switch, JEV_ROUTER_MIN_DWELL keeps the new model for that many responses
 * before Jev is asked again, and JEV_ROUTER_GRANULARITY=user asks only on messages the user wrote.
 *
 * - `direct` requests (compaction summaries, nested model calls) go to the cheap model.
 * - A retry after a failed cheap request escalates to the strong model; other retries stay put.
 * - The tier is router state: pi stores it on the session branch, so it survives compaction and
 *   follows /tree navigation.
 * - JEV_ROUTER_MODE=log always routes to JEV_ROUTER_FALLBACK and logs what Jev would have chosen.
 * - Decision-model failures route to JEV_ROUTER_FALLBACK (default strong).
 * - Gemini 3.x rejects a function call in the current turn that carries no thought signature (HTTP 400), and pi-ai
 *   keeps signatures only on calls the SAME model made, so the first switch between two Gemini models ended the run
 *   (found in E25's Terminal-Bench evaluation). A `before_provider_request` handler therefore gives every function-call
 *   part without a signature Google's documented placeholder (JEV_ROUTER_SIGNATURE_FIX=1, default; the same placeholder
 *   jevharness.llm.openai_compat uses for calls the decision model made). Requests whose function calls all carry
 *   their own signatures, as every request of an unrouted run does, are left unchanged.
 *
 * The model is listed as `s1/router`, not under `jev/`: Harbor reads `<PROVIDER>_BASE_URL` as the LLM
 * endpoint of an unknown provider, so `jev/...` would collide with JEV_BASE_URL (the decision model).
 *
 * Usage: pi -e adapters/pi/jev-router.ts --model s1/router
 * Env:   JEV_ROUTER_STRONG=google/gemini-3.5-flash  JEV_ROUTER_CHEAP=google/gemini-3.5-flash-lite
 *        JEV_ROUTER_MODE=trigger|log|off  JEV_ROUTER_TAU=0.5  JEV_ROUTER_GRANULARITY=request|user
 *        JEV_ROUTER_MIN_DWELL=1  JEV_ROUTER_ERROR_WINDOW=4  JEV_ROUTER_ERROR_ESCALATE=2
 *        JEV_ROUTER_FALLBACK=strong|cheap  JEV_ROUTER_PROVIDER=s1  JEV_ROUTER_ID=router
 *        JEV_ROUTER_CONTEXT_WINDOW=200000
 *        JEV_ROUTER_MAX_TOKENS=64000  and the shared settings in jev-common.ts.
 */

import type { Api, ClassifierQuestion, Message, Model } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, ModelRoute, ModelRouteRequest } from "@earendil-works/pi-coding-agent";
import { JevClient, readAnswer } from "./jev-common.ts";

type Tier = "strong" | "cheap";

interface RouterState {
	tier: Tier;
	/** Assistant responses in the transcript when the tier was chosen. */
	since: number;
}

type RouterRequest = ModelRouteRequest<RouterState>;

const QUESTION: ClassifierQuestion = {
	type: "bool",
	instructions:
		"Does the next step require careful multi-step reasoning, planning, or recovery from an error " +
		"(i.e., a strong model) rather than a routine action?",
	criteria: {
		true: "Hard step: planning, subtle reasoning, debugging, or recovering from a failure",
		false: "Routine step: an obvious command, a simple edit, or reading output",
	},
};

/** Google's documented thought-signature placeholder for function calls the current model did not produce. */
export const GEMINI_DUMMY_SIGNATURE = "skip_thought_signature_validator";

/**
 * Give every Gemini function-call part without a thought signature the placeholder (in place). Returns the number of
 * parts changed. Works on the @google/genai request payload ({contents: [{role, parts: [...]}], ...}); any other
 * payload shape is left alone.
 */
export function fillMissingThoughtSignatures(payload: unknown): number {
	const contents = (payload as { contents?: unknown })?.contents;
	if (!Array.isArray(contents)) return 0;
	let changed = 0;
	for (const content of contents) {
		const parts = (content as { parts?: unknown })?.parts;
		if (!Array.isArray(parts)) continue;
		for (const part of parts as Record<string, unknown>[]) {
			if (part && typeof part === "object" && part.functionCall && !part.thoughtSignature) {
				part.thoughtSignature = GEMINI_DUMMY_SIGNATURE;
				changed++;
			}
		}
	}
	return changed;
}

function splitSpec(spec: string): [string, string] {
	const slash = spec.indexOf("/");
	if (slash <= 0) throw new Error(`Model "${spec}" must be provider/id`);
	return [spec.slice(0, slash), spec.slice(slash + 1)];
}

/** Failed results among the last `window` tool results. */
function recentToolErrors(messages: readonly Message[], window: number): number {
	const results = messages.filter((message) => message.role === "toolResult").slice(-window);
	return results.filter((message) => message.isError).length;
}

export default function (pi: ExtensionAPI) {
	const jev = new JevClient("jev-router", "ROUTER");
	const mode = jev.env.str("MODE", "trigger");
	if (mode === "off") return;
	const models: Record<Tier, string> = {
		strong: jev.env.str("STRONG", "google/gemini-3.5-flash"),
		cheap: jev.env.str("CHEAP", "google/gemini-3.5-flash-lite"),
	};
	const tau = jev.env.num("TAU", 0.5);
	const perUser = jev.env.str("GRANULARITY", "request") === "user";
	const minDwell = jev.env.int("MIN_DWELL", 1);
	const errorWindow = jev.env.int("ERROR_WINDOW", 4);
	const errorEscalate = jev.env.int("ERROR_ESCALATE", 2);
	const fallback: Tier = jev.env.str("FALLBACK", "strong") === "cheap" ? "cheap" : "strong";

	function modelFor(ctx: ExtensionContext, tier: Tier): Model<Api> {
		const [provider, id] = splitSpec(models[tier]);
		const model = ctx.modelRegistry.find(provider, id);
		if (!model) throw new Error(`Model ${models[tier]} is not in the catalog`);
		return model;
	}

	function isTier(model: Model<Api>, tier: Tier): boolean {
		return `${model.provider}/${model.id}` === models[tier];
	}

	/** Ask Jev (or apply the error rule) for the tier of the next request. */
	async function decide(request: RouterRequest, ctx: ExtensionContext): Promise<Tier> {
		const errors = recentToolErrors(request.messages, errorWindow);
		if (errors >= errorEscalate) {
			jev.log(ctx, {
				dp: "routing",
				mode,
				question: "(recent tool errors: escalate)",
				finalValue: mode === "log" ? fallback : "strong",
				tau,
				extra: { reason: request.reason, tool_errors: errors, rule: "error-escalate" },
			});
			return mode === "log" ? fallback : "strong";
		}

		// The session projection keeps extension messages (nudges, reflections) apart from user prompts;
		// in `request.messages` pi has already turned them into user messages.
		const state = jev.state(jev.sessionMessages(ctx), { recent_tool_errors: errors });
		const result = await jev.classify(ctx, state, { hard: QUESTION }, request.signal);
		const reading = readAnswer(result, "hard");
		const s1Tier: Tier | undefined = reading ? (reading.pYes! >= tau ? "strong" : "cheap") : undefined;
		const tier = mode === "log" ? fallback : (s1Tier ?? fallback);
		jev.log(ctx, {
			dp: "routing",
			mode,
			question: QUESTION.instructions,
			reading,
			finalValue: tier,
			tau,
			result,
			state,
			extra: { reason: request.reason, tool_errors: errors, s1_tier: s1Tier ?? null, routed_model: models[tier] },
		});
		jev.status(ctx, reading ? `jev route: ${tier} (${reading.pYes!.toFixed(2)})` : `jev route: ${tier} (error)`);
		return tier;
	}

	if (jev.env.str("SIGNATURE_FIX", "1") !== "0") {
		pi.on("before_provider_request", (event) => {
			fillMissingThoughtSignatures(event.payload);
			return event.payload;
		});
	}

	pi.registerVirtualModel<RouterState>({
		provider: jev.env.str("PROVIDER", "s1"),
		id: jev.env.str("ID", "router"),
		name: "Router (Jev System One)",
		thinkingLevels: ["off", "minimal", "low", "medium", "high", "xhigh"],
		contextWindow: jev.env.int("CONTEXT_WINDOW", 200_000),
		maxTokens: jev.env.int("MAX_TOKENS", 64_000),
		async route(request, ctx): Promise<ModelRoute<RouterState>> {
			const route = (tier: Tier, state?: RouterState): ModelRoute<RouterState> => ({
				model: modelFor(ctx, tier),
				thinkingLevel: request.thinkingLevel,
				state,
			});
			if (request.reason === "direct") return route("cheap");

			const current = request.state;
			const responses = request.messages.filter((message) => message.role === "assistant").length;
			const switchTo = (tier: Tier) => route(tier, tier === current?.tier ? current : { tier, since: responses });

			if (request.reason === "retry") {
				if (request.failed && isTier(request.failed.model, "cheap") && mode !== "log") return switchTo("strong");
				return route(current?.tier ?? fallback);
			}
			if (current && perUser && request.reason !== "user") return route(current.tier);
			if (current && responses - current.since < minDwell) return route(current.tier);
			return switchTo(await decide(request, ctx));
		},
	});
}
