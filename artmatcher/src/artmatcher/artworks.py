"""Loading scraped artwork metadata and resolving it against glashelder images."""

import logging
from pathlib import Path

import click

from artscraper import Artwork

logger = logging.getLogger(__name__)


def load_artworks(path: str | Path | None) -> dict[str, Artwork]:
    """The artwork records a scrape produced, keyed by the reference they describe."""
    if path is None or not Path(path).is_file():
        if path is not None:
            logger.warning(f"no artworks at {path}, results will carry no metadata")
        return {}

    import artscraper

    found = artscraper.index(path)
    logger.info(f"loaded metadata for {len(found)} artwork references from {path}")
    return found


def artwork_of(reference: str, artworks: dict[str, Artwork] | None) -> Artwork | None:
    """The scraped record a matched reference describes, if we have it.

    The reference is the join key: a URL for scraped artworks, a path for local
    files (looked up by name, since the directory may differ per machine).
    """
    if not artworks or not reference:
        return None
    return artworks.get(reference) or artworks.get(Path(reference).name)


def resolve_subset(path: Path, source, root: str | Path | None) -> list[int]:
    """The glashelder indices of the images a `--subset` file lists.

    A line matches an image by its path relative to `root`, by its full path, or
    by its file name, in that order. Lines that match nothing are reported, so a
    typo or a list for another image directory does not go unnoticed.
    """
    served = list(source.indices)
    by_key: dict[str, int] = {}
    for index in served:
        image = Path(source.glashelder_path(index) or "")
        by_key.setdefault(image.name, index)
        by_key[str(image)] = index
        if root is not None and image.is_relative_to(root):
            by_key[str(image.relative_to(root))] = index

    lines = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
    wanted = [line for line in lines if line]
    found = {by_key[line] for line in wanted if line in by_key}
    missing = [line for line in wanted if line not in by_key]

    if not found:
        raise click.ClickException(
            f"none of the {len(wanted):,} images in {path} are among the "
            f"{len(served):,} glashelder images being served"
        )
    logger.info(f"labeling a subset: {len(found):,} images from {path}")
    if missing:
        logger.warning(
            f"{len(missing):,} lines in {path} match no glashelder image, "
            f"for example {missing[:3]}"
        )
    return sorted(found)
