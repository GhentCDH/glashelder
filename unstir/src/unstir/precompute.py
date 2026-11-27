"""`unstir precompute`: the ALIKED features of every image of a collection, once.

`unstir lookup` then reads the features of its candidates from the match store
instead of reading each image and running ALIKED on it, which is most of the time
a lookup takes on a CPU. Only images the store has no features for yet are done.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

from common.progress import progress

from .features import Writer, location
from .match import Matcher, Settings
from .pipeline import IO_THREADS, prefetch, readable
from .store import relative

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Collection:
    key: str
    """Directory name under `STORE/features`, as `unstir lookup` looks it up."""
    embeddings: Path
    """The images are the ones in this embedding store."""
    images: Path


def run(store: Path, collections: list[Collection], settings: Settings, batch: int):
    from embed.device import get_device
    from embed.io import load_embeddings

    device = get_device()
    matcher = Matcher(settings, device)
    read = readable(settings.resize)
    with ThreadPoolExecutor(IO_THREADS) as pool:
        for collection in collections:
            path = location(store, collection.key)
            writer = Writer(path, settings)
            refs = load_embeddings(collection.embeddings, root=collection.images)[
                "images"
            ]
            todo = [
                (ref, key)
                for ref in refs
                if (key := relative(ref, collection.images)) not in writer.known
            ]
            logger.info(
                f"{path}: {len(writer.refs):,} images done, {len(todo):,} to do "
                f"on {device}"
            )
            failed = 0
            with progress(stderr=True) as bar:
                task = bar.add_task(collection.key, total=len(todo))
                images = prefetch(pool, lambda item: read(item[0]), todo)
                for chunk in batched(images, batch, strict=False):
                    rows = []
                    for (_, key), image in chunk:
                        if image is None:
                            failed += 1
                            continue
                        rows.append((key, matcher.extract(image), image.shape[:2]))
                    writer.add(rows)
                    bar.update(task, advance=len(chunk))
            logger.info(
                f"{path}: features of {len(writer.refs):,} images"
                + (
                    f"; {failed:,} could not be read, tried again next run"
                    if failed
                    else ""
                )
            )
