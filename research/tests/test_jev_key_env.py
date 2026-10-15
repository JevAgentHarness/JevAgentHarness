"""The author's Jev-only OpenRouter key (OPENROUTER_API_KEY_JEV) is read by the Jev backend only (2026-10-06)."""
from jevharness.decision.jev import JevBackend
from jevharness.llm import openai_compat


def test_jev_backend_prefers_jev_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "shared-dummy")
    monkeypatch.setenv("OPENROUTER_API_KEY_JEV", "jev-dummy")
    assert JevBackend(provider="openrouter").api_key == "jev-dummy"


def test_jev_backend_falls_back_to_shared_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "shared-dummy")
    monkeypatch.delenv("OPENROUTER_API_KEY_JEV", raising=False)
    assert JevBackend(provider="openrouter").api_key == "shared-dummy"


def test_llm_route_never_reads_jev_key():
    assert all(env != "OPENROUTER_API_KEY_JEV" for _, env in openai_compat.PROVIDERS.values()) \
        if hasattr(openai_compat, "PROVIDERS") else "OPENROUTER_API_KEY_JEV" not in open(openai_compat.__file__).read()
