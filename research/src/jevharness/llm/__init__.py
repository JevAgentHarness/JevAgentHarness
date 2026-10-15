from .base import ChatResponse, LLMClient, ToolCall


def make_llm(spec: dict | str) -> LLMClient:
    """Build an LLM client from a config dict, e.g.
    ``{"provider": "deepseek", "model": "deepseek-chat", "temperature": 0}`` or a string
    ``"openai:gpt-4.1-mini"`` / ``"scripted"``.

    ``openai`` is the Chat Completions path (OpenAI-compatible client, logprobs on the gpt-4.1/4o families);
    ``openai_responses`` is OpenAI's current Responses API (``llm/openai_responses.py``: native function calling
    for the GPT-5.x/6 reasoning models, reasoning-item echo, optional ``previous_response_id`` chaining)."""
    if isinstance(spec, str):
        provider, _, model = spec.partition(":")
        spec = {"provider": provider, "model": model}
    spec = dict(spec)
    provider = spec.pop("provider")
    model = spec.pop("model", "")
    if provider in ("openai_responses", "openai-responses"):
        from .openai_responses import OpenAIResponsesClient
        return OpenAIResponsesClient(model, **spec)
    if provider == "simulated":
        from .simulated import SimulatedLLM
        return SimulatedLLM(model or "scripted", **spec)
    if provider == "scripted":
        from .scripted import ScriptedLLM
        return ScriptedLLM(model or "scripted", **spec)
    from .openai_compat import OpenAICompatClient
    return OpenAICompatClient(model, provider=provider, **spec)


__all__ = ["ChatResponse", "LLMClient", "ToolCall", "make_llm"]
