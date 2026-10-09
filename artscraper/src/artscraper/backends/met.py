"""Metropolitan Museum of Art collection backend (metmuseum.org).

Uses the MET Collection API (RESTful JSON at
collectionapi.metmuseum.org).

All data is CC0 / Open Access. The search endpoint returns Object IDs;
each object's details (title, artist, image URL) are fetched individually.

Filters applied (matching the user's website search):
  - hasImages=true
  - dateEnd=1950
  - medium per-material

The API's ``medium`` parameter accepts pipe-separated values but certain
combinations return 0 results, so we search per material and deduplicate.

The v1.1 paginated search endpoint caps at 10,000 results per query.
For materials exceeding that, the date range is recursively split until
each sub-query fits under the cap.

The website's ``yearTo=1950`` is open-ended on the lower end, so we use a
negative ``dateBegin`` to include BCE / ancient objects.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from urllib.parse import urlencode

import httpx

from ..client import fetch_json
from ..model import Artwork
from . import Backend, Context, enrich_pool

BASE_URL = "https://collectionapi.metmuseum.org"
SEARCH_URL = f"{BASE_URL}/public/collection/v1.1/search"
OBJECT_URL = f"{BASE_URL}/public/collection/v1/objects"
OBJECT_PAGE_URL = "https://www.metmuseum.org/art/collection/search"

# Website filters mapped to API parameters.
MATERIALS = [
    "Terracotta",
    "Metal",
    "Prints",
    "Clay",
    "Fragments",
    "Ink",
    "Ceramics",
    "Vases",
]

DATE_BEGIN = -10000
DATE_END = 1950

PAGE_SIZE = 500
SEARCH_CAP = 10000

FAILED_OBJECTS_NAME = "metmuseum_failed_objects.jsonl"
"""Objects that failed, kept for the retry pass at the end of a run."""

logger = logging.getLogger(__name__)


def _failed_path(ctx: Context) -> Path:
    """Where failed objects go: next to the output, or the working directory without one."""
    if ctx.output is None:
        return Path(FAILED_OBJECTS_NAME)
    return ctx.output.with_name(f"{ctx.output.stem}.{FAILED_OBJECTS_NAME}")


# --------------------------------------------------------------------------- #
#  API helpers
# --------------------------------------------------------------------------- #


async def _fetch_json(ctx: Context, url: str) -> dict:
    """GET a URL and parse JSON; the client retries transient failures."""
    data = await fetch_json(ctx.client, ctx.throttle, ctx.stats, "GET", url)

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected JSON response from {url}: {type(data).__name__}")

    return data


def _search_params(
    medium: str,
    date_begin: int,
    date_end: int,
    limit: int = 1,
    offset: int = 0,
) -> dict[str, str]:
    return {
        "q": "*",
        "hasImages": "true",
        "dateBegin": str(date_begin),
        "dateEnd": str(date_end),
        "medium": medium,
        "limit": str(limit),
        "offset": str(offset),
    }


async def _search_total(
    ctx: Context,
    medium: str,
    date_begin: int,
    date_end: int,
) -> int:
    url = f"{SEARCH_URL}?{urlencode(_search_params(medium, date_begin, date_end))}"

    try:
        data = await _fetch_json(ctx, url)
    except Exception as exc:
        logger.warning(
            "metmuseum: search total failed for %s (%d-%d): %s",
            medium,
            date_begin,
            date_end,
            exc,
        )
        return 0

    try:
        return max(0, int(data.get("total", 0)))
    except (TypeError, ValueError):
        logger.warning(
            "metmuseum: invalid search total for %s: %r",
            medium,
            data.get("total"),
        )
        return 0


async def _search_page(
    ctx: Context,
    medium: str,
    date_begin: int,
    date_end: int,
    offset: int,
) -> list[int]:
    params = _search_params(
        medium,
        date_begin,
        date_end,
        PAGE_SIZE,
        offset,
    )

    url = f"{SEARCH_URL}?{urlencode(params)}"

    try:
        data = await _fetch_json(ctx, url)
    except Exception as exc:
        logger.warning(
            "metmuseum: search page failed for %s (%d-%d, offset=%d): %s",
            medium,
            date_begin,
            date_end,
            offset,
            exc,
        )
        return []

    object_ids = data.get("objectIDs") or []

    if not isinstance(object_ids, list):
        logger.warning(
            "metmuseum: invalid objectIDs response for %s (offset=%d)",
            medium,
            offset,
        )
        return []

    valid_ids: list[int] = []

    for object_id in object_ids:
        try:
            valid_ids.append(int(object_id))
        except (TypeError, ValueError):
            logger.warning(
                "metmuseum: ignoring invalid object ID %r",
                object_id,
            )

    return valid_ids


async def _search_ids(
    ctx: Context,
    medium: str,
    date_begin: int,
    date_end: int,
) -> AsyncIterator[list[int]]:
    """Yield batches of object IDs for a medium and date range.

    If the total exceeds the 10,000-result cap, the date range is split
    recursively until each sub-query fits.
    """
    total = await _search_total(
        ctx,
        medium,
        date_begin,
        date_end,
    )

    if total == 0:
        return

    if total > SEARCH_CAP:
        mid = (date_begin + date_end) // 2

        if date_begin < mid < date_end:
            async for batch in _search_ids(
                ctx,
                medium,
                date_begin,
                mid,
            ):
                yield batch

            async for batch in _search_ids(
                ctx,
                medium,
                mid + 1,
                date_end,
            ):
                yield batch

            return

        logger.warning(
            "metmuseum: %s in %d-%d has %d results (cap %d); truncating to first %d",
            medium,
            date_begin,
            date_end,
            total,
            SEARCH_CAP,
            SEARCH_CAP,
        )

    maximum = min(total, SEARCH_CAP)
    offset = 0

    while offset < maximum:
        ids = await _search_page(
            ctx,
            medium,
            date_begin,
            date_end,
            offset,
        )

        if not ids:
            break

        yield ids

        # Advance by the actual number returned, rather than assuming
        # that the API always returns PAGE_SIZE records.
        offset += len(ids)

        if len(ids) < PAGE_SIZE:
            break


# --------------------------------------------------------------------------- #
#  Failed-object persistence
# --------------------------------------------------------------------------- #


def _append_failed_object_sync(
    path: Path,
    object_id: str,
    error: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "object_id": object_id,
        "error": error,
    }

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


async def _record_failed_object(
    path: Path,
    object_id: str,
    error: str,
) -> None:
    """Append a failed object to the retry file."""
    await asyncio.to_thread(
        _append_failed_object_sync,
        path,
        object_id,
        error,
    )


def _read_failed_ids_sync(path: Path) -> list[str]:
    if not path.exists():
        return []

    object_ids: list[str] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
                object_id = record["object_id"]
                object_ids.append(str(object_id))
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
            ):
                logger.warning(
                    "metmuseum: ignoring malformed failed-object record on line %d",
                    line_number,
                )

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(object_ids))


async def _read_failed_ids(path: Path) -> list[str]:
    return await asyncio.to_thread(_read_failed_ids_sync, path)


def _remove_resolved_ids(
    path: Path,
    resolved: set[str],
) -> None:
    """Rewrite the retry file to hold only the objects that still fail, once each."""
    if not path.exists():
        return

    remaining_lines: list[str] = []
    kept: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            try:
                record = json.loads(line)
                object_id = str(record["object_id"])
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
            ):
                # Preserve malformed lines rather than deleting them.
                remaining_lines.append(line)
                continue

            if object_id not in resolved and object_id not in kept:
                kept.add(object_id)
                remaining_lines.append(line)

    if remaining_lines:
        path.write_text(
            "".join(remaining_lines),
            encoding="utf-8",
        )
    else:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
#  Object detail fetcher
# --------------------------------------------------------------------------- #


def _is_permanent_invalid_object(
    data: dict,
) -> bool:
    """Return True for an explicit invalid-object API response."""
    message = data.get("message")

    if not message:
        return False

    normalized = str(message).lower()

    return "not a valid object" in normalized or "invalid object" in normalized


def _error_body(response: httpx.Response) -> dict:
    try:
        data = json.loads(response.text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


async def _fetch_object(
    ctx: Context,
    object_id: str,
) -> Artwork | None:
    """Fetch one object.

    The client already retries transient failures, so anything raised here is
    worth trying again in a later pass. Objects the API calls invalid (a 404
    with an explanation) are skipped for good.
    """
    url = f"{OBJECT_URL}/{object_id}"

    try:
        data = await _fetch_json(ctx, url)
    except httpx.HTTPStatusError as e:
        body = _error_body(e.response)
        if e.response.status_code == 404 and _is_permanent_invalid_object(body):
            logger.debug(
                "metmuseum: object %s is invalid: %s", object_id, body.get("message")
            )
            return None
        raise

    if _is_permanent_invalid_object(data):
        logger.debug(
            "metmuseum: object %s is invalid: %s", object_id, data.get("message")
        )
        return None

    if data.get("message"):
        raise RuntimeError(str(data["message"]))

    if not data.get("isPublicDomain"):
        return None

    image = data.get("primaryImage")

    if not image:
        images = data.get("additionalImages") or []
        image = images[0] if images else None

    if not image:
        return None

    object_url = (
        data.get("objectURL")
        or data.get("linkResource")
        or f"{OBJECT_PAGE_URL}/{object_id}"
    )

    return Artwork(
        source="metmuseum",
        url=object_url,
        image_url=image,
        title=data.get("title") or None,
        artist=data.get("artistDisplayName") or None,
        year=data.get("objectDate") or None,
    )


# --------------------------------------------------------------------------- #
#  Backend
# --------------------------------------------------------------------------- #


async def _scrape_materials(
    ctx: Context,
    fetch_one: Callable[[Context, str], Awaitable[Artwork | None]],
) -> AsyncIterator[Artwork]:
    seen_ids: set[int] = set()
    total_yielded = 0

    for medium in MATERIALS:
        if ctx.limit and total_yielded >= ctx.limit:
            return

        logger.info(
            "metmuseum: searching medium %s",
            medium,
        )

        async for batch in _search_ids(
            ctx,
            medium,
            DATE_BEGIN,
            DATE_END,
        ):
            # Build the to-fetch list, deduplicating across materials
            # and skipping URLs already scraped in earlier runs.
            todo: list[str] = []

            for object_id in batch:
                if object_id in seen_ids:
                    continue

                seen_ids.add(object_id)

                object_page = f"{OBJECT_PAGE_URL}/{object_id}"

                if object_page in ctx.seen:
                    continue

                todo.append(str(object_id))

            if not todo:
                continue

            remaining = ctx.limit - total_yielded if ctx.limit else 0

            async for artwork in enrich_pool(
                ctx,
                todo,
                fetch_one,
                remaining,
            ):
                yield artwork
                total_yielded += 1

                if ctx.limit and total_yielded >= ctx.limit:
                    return

        logger.info(
            "metmuseum: %s done, %d IDs seen, %d artworks yielded",
            medium,
            len(seen_ids),
            total_yielded,
        )


class MetMuseum(Backend):
    name = "metmuseum"

    async def stream(
        self,
        ctx: Context,
    ) -> AsyncIterator[Artwork]:
        failed_path = _failed_path(ctx)
        # objects fetched without failing this run, they leave the retry file
        resolved: set[str] = set()

        async def fetch(ctx: Context, object_id: str) -> Artwork | None:
            artwork = await _fetch_object(ctx, object_id)
            resolved.add(object_id)
            return artwork

        async def fetch_or_record(ctx: Context, object_id: str) -> Artwork | None:
            try:
                return await fetch(ctx, object_id)
            except Exception as exc:
                logger.error("metmuseum: object %s failed: %s", object_id, exc)
                await _record_failed_object(failed_path, object_id, repr(exc))
                return None

        try:
            yielded = 0
            async for artwork in _scrape_materials(ctx, fetch_or_record):
                yield artwork
                yielded += 1

            if ctx.limit and yielded >= ctx.limit:
                return

            # Failures from this and earlier runs get one more go; ones that
            # fail again stay in the file without being appended twice.
            failed_ids = await _read_failed_ids(failed_path)
            resolved.update(
                object_id
                for object_id in failed_ids
                if f"{OBJECT_PAGE_URL}/{object_id}" in ctx.seen
            )
            todo = [object_id for object_id in failed_ids if object_id not in resolved]
            if not todo:
                return

            logger.info("metmuseum: retrying %d previously failed objects", len(todo))
            remaining = ctx.limit - yielded if ctx.limit else 0
            async for artwork in enrich_pool(ctx, todo, fetch, remaining):
                yield artwork
        finally:
            if resolved:
                _remove_resolved_ids(failed_path, resolved)
