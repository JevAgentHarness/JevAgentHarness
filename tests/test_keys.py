"""A key pool in one environment variable: each session sticks to one key, sessions spread over the pool."""

from __future__ import annotations

from jevagentharness.config import Config
from jevagentharness.upstreams import Upstreams


def test_key_pool_is_sticky_and_spread(monkeypatch):
    monkeypatch.setenv("POOL_KEY", "k-a, k-b ,k-c")
    cfg = Config.from_dict({"upstreams": {"g": {"type": "gemini", "base_url": "http://x", "key_env": "POOL_KEY"}},
                            "tiers": {"strong": {"upstream": "g", "model": "m"},
                                      "fast": {"upstream": "g", "model": "m"}}})
    up = cfg.upstreams["g"]
    assert up.keys() == ["k-a", "k-b", "k-c"]
    sessions = [f"jah-auto/trial-{i}" for i in range(60)]
    picks = [up.key(s) for s in sessions]
    assert picks == [up.key(s) for s in sessions]                 # sticky
    assert set(picks) == {"k-a", "k-b", "k-c"}                     # spread
    ups = Upstreams(cfg)
    by_key = {up.key(s): ups.cache_for("g", s) for s in sessions}
    assert len({id(c) for c in by_key.values()}) == 3              # one cache manager per key


def test_single_key_unchanged(monkeypatch):
    monkeypatch.setenv("ONE_KEY", "only")
    cfg = Config.from_dict({"upstreams": {"g": {"type": "gemini", "base_url": "http://x", "key_env": "ONE_KEY"}},
                            "tiers": {"strong": {"upstream": "g", "model": "m"},
                                      "fast": {"upstream": "g", "model": "m"}}})
    up = cfg.upstreams["g"]
    assert up.key() == up.key("any") == "only" and up.key_index("any") == 0
