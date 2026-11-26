"""Reference photos of buildings, with their style, from Onroerend Erfgoed.

The Inventaris Onroerend Erfgoed (https://inventaris.onroerenderfgoed.be) describes
Flemish heritage objects. Each object's `kenmerkgroepen` link it to thesaurus terms:
its type (`typologie`), dating (`datering`) and architectural style (`stijl`). Its
`primaire_foto` points at an image in the Beeldbank, which serves it in several sizes.

The HTTP layer (throttling, retries, user agent) is artscraper's; the records are our
own, since artscraper's `Artwork` has no place for style or dating.

Fair use (https://developers.onroerenderfgoed.be/docs/fair-use): there is no fixed
request limit, but no high parallel load, cache on our side, and wait before
retrying. So the defaults are a few requests at a time, every API answer is kept
under `cache/` and never asked for twice, and artscraper's client backs off
(and honours `Retry-After`) before it retries.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

import httpx
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from artscraper.client import Stats, fetch_bytes, fetch_json, make_client
from common.throttle import Throttle

INVENTARIS = "https://inventaris.onroerenderfgoed.be/erfgoedobjecten"
OBJECT_URI = "https://id.erfgoed.net/erfgoedobjecten/{}"
IMAGE_URI = "https://id.erfgoed.net/afbeeldingen/{}"
IMAGE_CONTENT = "https://beeldbank.onroerenderfgoed.be/images/{}/content/{}"
STYLES = "https://thesaurus.onroerenderfgoed.be/conceptschemes/STIJLEN_EN_CULTUREN/c"
STYLE_URI = "https://id.erfgoed.net/thesauri/stijlen_en_culturen/{}"
STYLE_COLLECTION = "60"
"""The `stijlen` collection; its sibling `culturen` (61) holds prehistoric cultures."""
BUILDINGS = "2"
"""The `Bouwkundig` discipline: 1 is archaeological, 3 landscape, 4 maritime."""

USER_AGENT = (
    "glashelder-archstil/0.1 (research, Ghent University; sysadmin.ghentcdh@ugent.be)"
)
CONCURRENCY = 4
RATE = 2.0
"""Requests per second per host; the fair use policy asks for no heavy load."""

PAGE_SIZE = 50
"""The inventaris answers at most 50 objects per request, whatever range is asked."""
SIZES = ("medium", "large", "full")
"""Beeldbank sizes: 500 px, 1024 px, and the original resolution."""
JSON = {"Accept": "application/json"}

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Term:
    """A thesaurus term, such as a style or a dating."""

    label: str
    uri: str


@dataclass(slots=True)
class Reference:
    """One heritage object and its primary photo."""

    object_id: int
    object_uri: str
    name: str
    location: str | None
    styles: list[Term]
    datings: list[Term]
    types: list[Term]
    image_id: int
    image_uri: str
    image_file: str
    """Path of the downloaded photo, relative to the output directory."""
    image_title: str | None = None
    image_date: str | None = None
    attribution: str | None = None
    """Credit line the licence asks for, e.g. "Beeldbank Onroerend Erfgoed, ..."."""
    licenses: list[str] = field(default_factory=list)
    query: dict[str, str] = field(default_factory=dict)
    """The filter this object was found with."""


@dataclass
class Session:
    client: httpx.AsyncClient
    api: Throttle
    images: Throttle
    stats: Stats
    cache: Path | None = None
    """Directory for the JSON of every object and image record fetched."""


async def get_json(session: Session, url: str, **headers: str) -> Any:
    return await fetch_json(
        session.client,
        session.api,
        session.stats,
        "GET",
        url,
        headers=JSON | headers,
    )


async def cached_json(session: Session, url: str, key: str) -> Any:
    """`get_json`, answered from `session.cache` when it was fetched before."""
    path = session.cache / f"{key}.json" if session.cache else None
    if path is not None and path.exists():
        return json.loads(path.read_text())
    data = await get_json(session, url)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        tmp.replace(path)
    return data


async def styles(session: Session) -> list[dict]:
    """Every style in the thesaurus, as `{"id", "uri", "label"}`."""
    found = await get_json(session, STYLES, Range="items=0-999")
    return sorted(found, key=lambda s: s["label"].casefold())


@dataclass
class Style:
    """A style, the number of objects filed under it, and its narrower styles."""

    id: str
    label: str
    count: int | None
    """Objects with this style or a narrower one: the filter includes narrower
    styles."""
    children: list["Style"] = field(default_factory=list)


async def count(session: Session, query: dict[str, str]) -> int | None:
    """How many objects match `query`, from the `Content-Range` of a 1-item page;
    `None` when the inventaris keeps failing.

    artscraper's `fetch_json` hides the headers, so this retries by itself: the
    inventaris now and then answers a valid filter with a 404.
    """
    url = httpx.URL(INVENTARIS, params=query)
    for attempt in range(4):
        if attempt:
            session.stats.retries += 1
            await asyncio.sleep(2**attempt)
        try:
            async with session.api:
                response = await session.client.get(
                    url, headers=JSON | {"Range": "items=0-0"}
                )
        except httpx.TransportError:
            continue
        finally:
            session.stats.requests += 1
        if response.status_code == 416:
            return 0
        if response.is_success:
            total = response.headers.get("Content-Range", "").rsplit("/", 1)[-1]
            return int(total) if total.isdigit() else 0  # no header: an empty page
    session.stats.errors += 1
    logger.warning("could not count %s", url)
    return None


async def style_tree(session: Session, query: dict[str, str]) -> list[Style]:
    """The architectural styles as a tree, each with its number of objects that
    also match `query` (e.g. `{"discipline": BUILDINGS}`)."""

    async def node(concept: dict) -> Style:
        data = await get_json(session, f"{STYLES}/{concept['id']}")
        narrower, objects = await asyncio.gather(
            asyncio.gather(*(node(c) for c in data.get("narrower") or [])),
            count(session, query | {"stijl": concept["id"]}),
        )
        return Style(
            concept["id"],
            data["label"],
            objects,
            sorted(narrower, key=lambda s: -(s.count or 0)),
        )

    collection = await get_json(session, f"{STYLES}/{STYLE_COLLECTION}")
    tree = await asyncio.gather(*(node(m) for m in collection["members"]))
    return sorted(tree, key=lambda s: -(s.count or 0))


def resolve_style(value: str, known: list[dict]) -> str:
    """A style id from an id, a URI or a label such as "gotiek"."""
    value = value.strip()
    if value.isdigit():
        return value
    for style in known:
        if value in (style["uri"], style["id"]) or (
            value.casefold() == style["label"].casefold()
        ):
            return style["id"]
    raise ValueError(f"unknown style {value!r}; see `archstil styles`")


async def list_objects(session: Session, query: dict[str, str]) -> AsyncIterator[dict]:
    """Every object matching `query`, a page of 50 at a time."""
    url = httpx.URL(INVENTARIS, params=query)
    start = 0
    while True:
        try:
            page = await get_json(
                session, str(url), Range=f"items={start}-{start + PAGE_SIZE - 1}"
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 416:  # asked past the last object
                return
            raise
        for item in page:
            yield item
        if len(page) < PAGE_SIZE:
            return
        start += PAGE_SIZE


def terms(record: dict, kind: str) -> list[Term]:
    """The thesaurus terms of one kind (`stijl`, `datering`, `typologie`)."""
    found: dict[str, Term] = {}
    for group in record.get("kenmerkgroepen") or []:
        for term in group.get("thesaurus") or []:
            if (term.get("kenmerk") or {}).get("naam") == kind:
                found.setdefault(term["uri"], Term(term["label"], term["uri"]))
    return list(found.values())


def image_id(record: dict) -> int | None:
    photo = record.get("primaire_foto")
    if not photo:
        return None
    tail = photo.rstrip("/").rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else None


async def fetch_reference(
    session: Session,
    object_id: int,
    output: Path,
    size: str,
    query: dict[str, str],
) -> Reference | None:
    """The full record, its photo's metadata and the photo itself."""
    record = await cached_json(
        session, OBJECT_URI.format(object_id), f"objects/{object_id}"
    )
    photo = image_id(record)
    if photo is None:
        return None

    meta = await cached_json(session, IMAGE_URI.format(photo), f"images/{photo}")
    if meta.get("visibility") not in (None, "publiek"):
        return None

    image_file = Path("images") / f"{photo}.jpg"
    target = output / image_file
    if not target.exists():
        data = await fetch_bytes(
            session.client,
            IMAGE_CONTENT.format(photo, size),
            session.images,
            session.stats,
        )
        tmp = target.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(target)

    return Reference(
        object_id=record["id"],
        object_uri=record.get("uri") or OBJECT_URI.format(object_id),
        name=record.get("naam") or "",
        location=record.get("locatie_samenvatting"),
        styles=terms(record, "stijl"),
        datings=terms(record, "datering"),
        types=terms(record, "typologie"),
        image_id=photo,
        image_uri=IMAGE_URI.format(photo),
        image_file=str(image_file),
        image_title=meta.get("title"),
        image_date=(meta.get("dates") or {}).get("original"),
        attribution=meta.get("naamsvermelding"),
        licenses=[lic["uri"] for lic in meta.get("licenses") or [] if lic.get("uri")],
        query=query,
    )


def saved(path: Path) -> list[dict]:
    """References already in a references file, to resume a run."""
    if not path.exists():
        return []
    found = []
    for line in path.read_text().splitlines():
        try:
            found.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a line cut short by an interrupted run
    return found


@dataclass
class Run:
    """What every query of one scrape shares."""

    session: Session
    output: Path
    size: str
    limit: int
    require_style: bool
    concurrency: int
    out: TextIO
    seen: set[int]
    progress: Progress


async def save_batch(run: Run, ids: list[int], query: dict[str, str]) -> int:
    """Fetch a batch of objects at once, write the usable ones; how many."""

    async def one(object_id: int) -> Reference | None:
        try:
            return await fetch_reference(
                run.session, object_id, run.output, run.size, query
            )
        except Exception as e:
            logger.warning("skipping object %s: %s", object_id, e)
            return None

    kept = 0
    for ref in await asyncio.gather(*(one(i) for i in ids)):
        if ref is None or (run.require_style and not ref.styles):
            continue
        run.out.write(json.dumps(asdict(ref), ensure_ascii=False) + "\n")
        run.seen.add(ref.object_id)
        kept += 1
    run.out.flush()
    return kept


async def scrape_query(run: Run, query: dict[str, str], kept: int) -> None:
    """Save objects matching `query` until `run.limit` of them are saved."""
    label = ", ".join(f"{k}={v.rsplit('/', 1)[-1]}" for k, v in query.items()) or "all"
    task = run.progress.add_task(label)
    stats = run.session.stats

    def report() -> None:
        run.progress.update(
            task,
            description=f"{label}: {kept} saved, {stats.requests} requests, "
            f"{stats.errors} errors",
        )

    def room() -> int:
        """Objects in the next batch, so `limit` is not passed."""
        return min(run.concurrency, run.limit - kept) if run.limit else run.concurrency

    report()
    if run.limit and kept >= run.limit:
        return
    batch: list[int] = []
    async for item in list_objects(run.session, query):
        if item["id"] in run.seen or not item.get("primaire_foto"):
            continue
        batch.append(item["id"])
        if len(batch) >= room():
            kept += await save_batch(run, batch, query)
            batch = []
            report()
            if run.limit and kept >= run.limit:
                return
    if batch:
        kept += await save_batch(run, batch, query)
        report()


async def scrape(
    output: Path,
    style_names: tuple[str, ...],
    typologie: str | None,
    limit: int,
    size: str,
    concurrency: int,
    rate: float,
    require_style: bool,
    buildings_only: bool = True,
) -> None:
    (output / "images").mkdir(parents=True, exist_ok=True)
    references = output / "references.jsonl"
    done = saved(references)

    async with make_client(concurrency) as client:
        client.headers["User-Agent"] = USER_AGENT
        session = Session(
            client,
            api=Throttle(concurrency, rate),
            images=Throttle(concurrency, rate),
            stats=Stats(workers=concurrency),
            cache=output / "cache",
        )

        if style_names:
            known = await styles(session)
            queries = [
                {"stijl": STYLE_URI.format(resolve_style(s, known))}
                for s in style_names
            ]
        else:
            queries = [{}]
        if typologie:
            queries = [q | {"typologie": typologie} for q in queries]
        if buildings_only:
            queries = [q | {"discipline": BUILDINGS} for q in queries]

        with (
            references.open("a") as out,
            Progress(
                SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn()
            ) as progress,
        ):
            run = Run(
                session,
                output,
                size,
                limit,
                require_style,
                concurrency,
                out,
                seen={r["object_id"] for r in done},
                progress=progress,
            )
            for query in queries:
                # a resumed run counts what this query saved before
                before = sum(r.get("query") == query for r in done)
                await scrape_query(run, query, before)


def standalone_session(client: httpx.AsyncClient, concurrency: int) -> Session:
    client.headers["User-Agent"] = USER_AGENT
    throttle = Throttle(concurrency, RATE)
    return Session(client, throttle, throttle, Stats())


async def list_styles() -> list[dict]:
    """Every term in the thesaurus, cultures included."""
    async with make_client(1) as client:
        return await styles(standalone_session(client, 1))


async def building_styles(buildings_only: bool) -> list[Style]:
    """The architectural style tree, counted over buildings or every object."""
    async with make_client(CONCURRENCY) as client:
        session = standalone_session(client, CONCURRENCY)
        return await style_tree(
            session, {"discipline": BUILDINGS} if buildings_only else {}
        )
