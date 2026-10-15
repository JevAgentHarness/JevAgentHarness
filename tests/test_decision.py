"""The Jev backend against a local fake System-One server: parsing, cost, and which failures are retried."""

from __future__ import annotations

from aiohttp import web
from aiohttp.test_utils import TestServer

from jevagentharness.decision import JevBackend
from jevagentharness.router import QUESTION


def test_jev_backend_retries_a_provider_499_and_not_a_400(run):
    calls: list[int] = []
    statuses = [499, 200, 400]

    async def handle(request: web.Request) -> web.Response:
        body = await request.json()
        assert body["questions"]["next_step"]["type"] == "choice" and body["state"] == "S"
        status = statuses[min(len(calls), len(statuses) - 1)]
        calls.append(status)
        if status != 200:
            return web.json_response({"error": {"message": "Unable to complete this request."}}, status=status)
        return web.json_response({"model": "typesafe/jev-test", "usage": {"input_tokens": 2000, "cost": 8.4e-05},
                                  "answers": {"next_step": {"probabilities": {"routine": 0.3, "hard": 0.7}}}})

    async def go():
        app = web.Application()
        app.router.add_post("/v1/systemone", handle)
        server = TestServer(app)
        await server.start_server()
        be = JevBackend(provider="local", base_url=str(server.make_url("")), key_env="", retries=2)
        try:
            d = await be.decide("S", QUESTION)
            assert d.p == {"routine": 0.3, "hard": 0.7} and d.cost_usd == 8.4e-05 and d.model == "typesafe/jev-test"
            assert calls == [499, 200]                          # the 499 was retried once
            try:
                await be.decide("S", QUESTION)
                raise AssertionError("a 400 must raise")
            except RuntimeError as e:
                assert "HTTP 400" in str(e)
            assert calls == [499, 200, 400]                     # ... and the 400 was not
        finally:
            await be.close()
            await server.close()

    run(go())
