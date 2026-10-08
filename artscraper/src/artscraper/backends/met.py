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
import random
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import urlencode

from ..client import fetch_text
from ..model import Artwork, merge
from . import Backend, Context, enrich_pool, physical

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

MAX_ATTEMPTS = 8

# Failed requests are retained for a later retry pass.
FAILED_OBJECTS_PATH = Path("metmuseum_failed_objects.jsonl")

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  API helpers
# --------------------------------------------------------------------------- #


async def _fetch_json(ctx: Context, url: str) -> dict:
    """GET a URL via the project's HTTP client and parse JSON.

    Exceptions are deliberately allowed to propagate. The object fetcher
    handles retries because it needs to distinguish temporary failures from
    permanent invalid-object responses.
    """
    text = await fetch_text(
        ctx.client,
        url,
        ctx.throttle,
        ctx.stats,
    )

    data = json.loads(text)

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
    object_id: str,
    error: str,
) -> None:
    FAILED_OBJECTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "object_id": object_id,
        "error": error,
    }

    with FAILED_OBJECTS_PATH.open(
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
    object_id: str,
    error: str,
) -> None:
    """Append a permanently failed request to the retry file."""
    await asyncio.to_thread(
        _append_failed_object_sync,
        object_id,
        error,
    )


def _read_failed_ids_sync() -> list[str]:
    if not FAILED_OBJECTS_PATH.exists():
        return []

    object_ids: list[str] = []

    with FAILED_OBJECTS_PATH.open(
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


async def _read_failed_ids() -> list[str]:
    return await asyncio.to_thread(_read_failed_ids_sync)


def _remove_successful_failed_ids_sync(
    successful_ids: set[str],
) -> None:
    if not FAILED_OBJECTS_PATH.exists():
        return

    remaining_lines: list[str] = []

    with FAILED_OBJECTS_PATH.open(
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

            if object_id not in successful_ids:
                remaining_lines.append(line)

    if remaining_lines:
        FAILED_OBJECTS_PATH.write_text(
            "".join(remaining_lines),
            encoding="utf-8",
        )
    else:
        FAILED_OBJECTS_PATH.unlink(missing_ok=True)


async def _remove_successful_failed_ids(
    successful_ids: set[str],
) -> None:
    await asyncio.to_thread(
        _remove_successful_failed_ids_sync,
        successful_ids,
    )


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


async def _fetch_object(
    ctx: Context,
    object_id: str,
) -> Artwork | None:
    """Fetch one object with retries.

    Temporary HTTP failures, connection errors, timeouts, and malformed
    responses are retried. Explicit invalid-object responses are not retried
    indefinitely.
    """
    url = f"{OBJECT_URL}/{object_id}"

    for attempt in range(MAX_ATTEMPTS):
        try:
            data = await _fetch_json(ctx, url)

            if _is_permanent_invalid_object(data):
                logger.warning(
                    "metmuseum: object %s is invalid: %s",
                    object_id,
                    data.get("message"),
                )
                return None

            if data.get("message"):
                raise RuntimeError(str(data["message"]))

            break

        except Exception as exc:
            final_attempt = attempt == MAX_ATTEMPTS - 1

            if final_attempt:
                logger.error(
                    "metmuseum: object %s failed after %d attempts: %s",
                    object_id,
                    MAX_ATTEMPTS,
                    exc,
                )

                await _record_failed_object(
                    object_id,
                    repr(exc),
                )

                return None

            # Exponential backoff with jitter.
            # This does not alter enrich_pool or concurrency.
            delay = min(
                300.0,
                (2**attempt) + random.uniform(0.5, 2.0),
            )

            logger.warning(
                "metmuseum: object %s failed "
                "(attempt %d/%d): %s; "
                "retrying in %.1f seconds",
                object_id,
                attempt + 1,
                MAX_ATTEMPTS,
                exc,
                delay,
            )

            await asyncio.sleep(delay)

    else:
        return None

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
        physical_description=physical(
            data.get("objectName"), data.get("medium"), data.get("dimensions")
        ),
    )


# --------------------------------------------------------------------------- #
#  Backend
# --------------------------------------------------------------------------- #


class MetMuseum(Backend):
    name = "metmuseum"

    async def stream(
        self,
        ctx: Context,
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

                # This is intentionally unchanged.
                # Concurrency is controlled by the existing enrich_pool.
                remaining = ctx.limit - total_yielded if ctx.limit else 0

                async for artwork in enrich_pool(
                    ctx,
                    todo,
                    _fetch_object,
                    remaining,
                ):
                    if artwork is None:
                        continue

                    yield artwork
                    total_yielded += 1

                    if ctx.limit and total_yielded >= ctx.limit:
                        return

            logger.info(
                "metmuseum: %s done — %d IDs seen, %d artworks yielded",
                medium,
                len(seen_ids),
                total_yielded,
            )

        # Retry objects that still failed after all normal attempts.
        failed_ids = await _read_failed_ids()

        if not failed_ids:
            return

        logger.info(
            "metmuseum: retrying %d previously failed objects",
            len(failed_ids),
        )

        successful_retry_ids: set[str] = set()

        remaining = ctx.limit - total_yielded if ctx.limit else 0

        async for artwork in enrich_pool(
            ctx,
            failed_ids,
            _fetch_object,
            remaining,
        ):
            if artwork is None:
                continue

            successful_retry_ids.add(artwork.url.rsplit("/", 1)[-1])

            yield artwork
            total_yielded += 1

            if ctx.limit and total_yielded >= ctx.limit:
                break

        if successful_retry_ids:
            await _remove_successful_failed_ids(
                successful_retry_ids,
            )

    async def complete(
        self, ctx: Context, artworks: list[Artwork]
    ) -> AsyncIterator[Artwork]:
        async def complete_one(ctx: Context, artwork: Artwork) -> Artwork | None:
            # the object page url ends in the API's object id
            found = await _fetch_object(ctx, artwork.url.rstrip("/").rsplit("/", 1)[-1])
            return merge(artwork, found) if found else None

        async for artwork in enrich_pool(ctx, artworks, complete_one):
            yield artwork
