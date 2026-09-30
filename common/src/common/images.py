"""Reading images that live either on disk or behind a URL.

An image reference is just a string: a path for local files, an `http(s)` URL for
scraped artworks. Nothing else in the pipeline has to care which it is.

Fetching is IO bound and the compute that follows is not, so `stream` pulls many
images at once and hands them over as they land, which keeps the GPU fed without
ever writing to disk. Each image comes with a hash of its bytes, so an embedding
can later be checked against the image it was made from. Caching is opt-in: a
full reference set is tens of gigabytes and every image is normally read once.
"""

import asyncio
import hashlib
import io
import logging
import os
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image

from .config import config
from .throttle import Throttle

USER_AGENT = "glashelder/2.0 (sysadmin.ghentcdh@ugent.be)"
TIMEOUT = 30.0
MAX_ATTEMPTS = 3

logger = logging.getLogger(__name__)

# httpx logs every request at INFO, which buries any progress output
logging.getLogger("httpx").setLevel(logging.WARNING)


def is_url(ref: str | Path) -> bool:
    return str(ref).startswith(("http://", "https://"))


def cache_name(url: str) -> str:
    """Stable filename for a URL: the same URL always maps to the same file."""
    suffix = Path(str(url).split("?")[0]).suffix
    return hashlib.sha256(url.encode()).hexdigest()[:16] + (suffix or ".jpg")


def cache_path(url: str, cache_dir: Path | None = None) -> Path:
    directory = (
        Path(cache_dir) if cache_dir is not None else config().paths.images.cache
    )
    return directory / cache_name(url)


def decode(raw: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(raw))
    image.load()  # decode now, while we still hold the bytes
    return image.convert("RGB")


def store(url: str, raw: bytes, cache_dir: Path | None = None) -> Path:
    """Write a fetched image into the cache, atomically."""
    path = cache_path(url, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    return path


def load(ref: str | Path, *, cache: bool | None = None) -> Image.Image:
    """Read one image, from disk or from the network.

    Use this where a single image is needed at a time. For a whole set, `stream`
    overlaps the fetching with whatever comes next.
    """
    if not is_url(ref):
        return Image.open(ref).convert("RGB")

    url = str(ref)
    caching = config().images.cache if cache is None else cache
    cached = cache_path(url) if caching else None
    if cached is not None and cached.is_file():
        return Image.open(cached).convert("RGB")

    with httpx.Client(
        timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        raw = _fetch_sync(client, url)

    if cached is not None:
        store(url, raw)
    return decode(raw)


def _fetch_sync(client: httpx.Client, url: str) -> bytes:
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.content
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == MAX_ATTEMPTS - 1:
                raise
    raise RuntimeError(f"{url}: out of attempts")


def digest(raw: bytes) -> str:
    """Hash of an image's bytes, to tell whether it changed since it was embedded."""
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def host(url: str) -> str:
    return urlsplit(url).hostname or ""


async def stream(
    refs: Iterable[str | Path],
    *,
    concurrency: int | None = None,
    per_host: int | None = None,
    rate: float | None = None,
    cache: bool | None = None,
    buffer: int | None = None,
) -> AsyncIterator[tuple[str, Image.Image, str]]:
    """Yield `(reference, image, digest)` as they become available.

    Order is not preserved: the point is to keep fetches in flight so the consumer
    never waits on the network. Every host gets its own workers, `per_host` of them
    at `rate` requests per second (or what `[images.hosts]` sets for it), so a slow
    or strict server is neither hammered nor holds up the others. `concurrency`
    caps the connections over all hosts, and the workers for local files.

    At most `buffer` decoded images are held, so memory does not grow with the size
    of the input. References that fail to load are logged and skipped, the same way
    the dataloader drops unreadable images.
    """
    settings = config().images
    connections = concurrency if concurrency is not None else settings.concurrency
    caching = settings.cache if cache is None else cache
    queue: asyncio.Queue[tuple[str, Image.Image, str] | None] = asyncio.Queue(
        maxsize=buffer if buffer is not None else 2 * connections
    )
    stop = asyncio.Event()

    groups: dict[str, list[str | Path]] = {}
    for ref in refs:
        groups.setdefault(host(str(ref)) if is_url(ref) else "", []).append(ref)

    async def worker(
        client: httpx.AsyncClient, todo: Iterator[str | Path], throttle: Throttle
    ) -> None:
        for ref in todo:  # a shared iterator is safe between coroutines
            if stop.is_set():
                return
            try:
                raw = await _read_async(client, ref, caching, throttle)
                image, hashed = await asyncio.to_thread(_decode_hashed, raw)
            except Exception as e:  # one bad image must not kill the run
                logger.warning("skipping %s: %s", ref, e)
                continue
            await queue.put((str(ref), image, hashed))

    def workers(client: httpx.AsyncClient, name: str, group: list[str | Path]):
        count, limit = (
            (connections, 0.0) if name == "" else settings.limits(name, per_host, rate)
        )
        todo, throttle = iter(group), Throttle(count, limit)
        return [worker(client, todo, throttle) for _ in range(count)]

    async def run() -> None:
        async with httpx.AsyncClient(
            # waiting for a free connection is expected, not a timeout
            timeout=httpx.Timeout(TIMEOUT, pool=None),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            limits=httpx.Limits(max_connections=connections),
        ) as client:
            await asyncio.gather(
                *(
                    w
                    for name, group in groups.items()
                    for w in workers(client, name, group)
                )
            )
        await queue.put(None)  # sentinel: nothing more is coming

    runner = asyncio.create_task(run())
    try:
        while (item := await queue.get()) is not None:
            yield item
    finally:
        stop.set()
        runner.cancel()


def _decode_hashed(raw: bytes) -> tuple[Image.Image, str]:
    return decode(raw), digest(raw)


async def _read_async(
    client: httpx.AsyncClient,
    ref: str | Path,
    caching: bool,
    throttle: Throttle,
) -> bytes:
    """The raw bytes behind a reference; file IO is kept off the event loop."""
    if not is_url(ref):
        return await asyncio.to_thread(Path(ref).read_bytes)

    url = str(ref)
    cached = cache_path(url) if caching else None
    if cached is not None and cached.is_file():
        return await asyncio.to_thread(cached.read_bytes)

    raw = await _fetch_async(client, url, throttle)
    if cached is not None:
        await asyncio.to_thread(store, url, raw)
    return raw


async def _fetch_async(
    client: httpx.AsyncClient, url: str, throttle: Throttle
) -> bytes:
    backoff = 1.0
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            await asyncio.sleep(backoff)
            backoff *= 2
        try:
            async with throttle:
                response = await client.get(url)
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == MAX_ATTEMPTS - 1:
                raise
            continue
        if response.is_success:
            return response.content
        if not response.is_server_error and response.status_code != 429:
            response.raise_for_status()
    raise RuntimeError(f"{url}: out of attempts")


def as_paths(refs: Iterable[str | Path]) -> list[Path]:
    """Local paths for references, fetching and caching the URLs among them.

    For the few consumers that genuinely need a file on disk, such as the Qwen
    embedder's `file://` messages.
    """
    return [Path(ref) if not is_url(ref) else _ensure_cached(str(ref)) for ref in refs]


def _ensure_cached(url: str) -> Path:
    path = cache_path(url)
    if not path.is_file():
        with httpx.Client(
            timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        ) as client:
            store(url, _fetch_sync(client, url))
    return path
