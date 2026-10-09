import asyncio
from collections.abc import AsyncIterator

import httpx
from artscraper.backends import Context, bare, take, urls_to_artworks
from artscraper.client import Stats
from artscraper.model import Artwork

from common.throttle import Throttle


def context(limit: int = 0, with_images: bool = True) -> Context:
    return Context(
        client=httpx.AsyncClient(),
        throttle=Throttle(1, 0),
        stats=Stats(),
        limit=limit,
        with_images=with_images,
        workers=2,
    )


def numbered(closed: list[bool]) -> AsyncIterator[Artwork]:
    async def gen() -> AsyncIterator[Artwork]:
        try:
            for n in range(5):
                yield Artwork(source="test", url=str(n), image_url=None)
        finally:
            closed.append(True)

    return gen()


def collect(artworks: AsyncIterator[Artwork]) -> list[str]:
    async def run() -> list[str]:
        return [artwork.url async for artwork in artworks]

    return asyncio.run(run())


def test_take_stops_at_limit_and_closes_source() -> None:
    closed: list[bool] = []
    assert collect(take(context(limit=2), numbered(closed))) == ["0", "1"]
    assert closed == [True]


def test_take_without_limit_yields_everything() -> None:
    closed: list[bool] = []
    assert collect(take(context(), numbered(closed))) == ["0", "1", "2", "3", "4"]
    assert closed == [True]


def test_urls_to_artworks() -> None:
    async def fetch_one(ctx: Context, url: str) -> Artwork | None:
        if url == "bad":
            raise ValueError(url)
        if url == "none":
            return None
        return Artwork(source="test", url=url, image_url=f"{url}.jpg")

    urls = ["a", "bad", "none", "b"]

    fetched = collect(urls_to_artworks(context(), urls, fetch_one, bare("t")))
    assert sorted(fetched) == ["a", "b"]

    # without images nothing is fetched and the url-only artworks come back in order
    url_only = collect(
        urls_to_artworks(context(with_images=False), urls, fetch_one, bare("t"))
    )
    assert url_only == urls
