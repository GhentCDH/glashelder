import asyncio

import httpx
import pytest
from artscraper.client import Stats, fetch_json

from artscraper import client
from common.throttle import Throttle


def test_fetch_json_retries_server_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "INITIAL_BACKOFF", 0.0)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        # a content type `Response.json()` would not accept
        return httpx.Response(
            200, content=b'{"d": {"ok": true}}', headers={"content-type": "text/xml"}
        )

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await fetch_json(
                http,
                Throttle(1, 0),
                stats,
                "POST",
                "https://example.org/api",
                json={"q": 1},
            )

    stats = Stats()
    assert asyncio.run(run()) == {"d": {"ok": True}}
    assert len(calls) == 2
    assert calls[0].method == "POST"
    assert calls[0].content == b'{"q":1}'
    assert (stats.requests, stats.retries, stats.errors) == (2, 1, 0)


def test_fetch_json_raises_on_client_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await fetch_json(
                http, Throttle(1, 0), stats, "DELETE", "https://example.org/x"
            )

    stats = Stats()
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run())
    assert (stats.requests, stats.retries, stats.errors) == (1, 0, 1)
