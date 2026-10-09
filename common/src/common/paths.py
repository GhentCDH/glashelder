"""Moving image paths between machines.

Embeddings and clusterings record where their images were. Local paths are kept
relative to a root, so the same files can be found under another directory on
another machine. URLs are portable already and pass through untouched.
"""

import logging
import os.path
from pathlib import Path

from .images import is_url

logger = logging.getLogger(__name__)


def _common_root(image_paths: list[str]) -> str | None:
    """Infer the directory all image paths live under, or None if not derivable."""
    absolute = [p for p in image_paths if p and os.path.isabs(p) and not is_url(p)]
    if not absolute:
        return None
    if len(absolute) == 1:
        return os.path.dirname(absolute[0])
    return os.path.commonpath(absolute)


def _relative_to(path: str, root: Path) -> str:
    # "" is a placeholder for a missing image; Path("") would become "."
    # a URL is already portable and must not be mangled into a path
    if path == "" or is_url(path):
        return path
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(root)
        except ValueError:
            logger.warning(f"Image path {path} is not under root {root}")
    return str(candidate)


def relativize_image_paths(
    image_paths: list[str], root: str | Path | None
) -> tuple[list[str], str | None]:
    """Strip `root` from image paths so they can be rebased on another machine.

    Falls back to the common ancestor of the given paths when `root` is not
    provided, which is what makes legacy pickles (absolute paths, no stored
    root) portable as well.

    Returns:
        Tuple of (relative paths, the root they are relative to).
    """
    root_str = str(root) if root is not None else _common_root(image_paths)
    if root_str is None:
        return list(image_paths), None

    root_path = Path(root_str)
    return [_relative_to(p, root_path) for p in image_paths], root_str


def rebase_paths(image_paths: list[str], root: str | Path | None) -> list[str]:
    """Re-root image paths recorded on another machine onto `root`.

    This is the single place where "paths were stored somewhere else" is fixed:
    the stored paths are made relative to their common ancestor and re-joined
    with the local image directory. Empty entries (used as placeholders) and
    already-relative paths are preserved. Returns the paths unchanged when
    `root` is None.
    """
    if root is None:
        return list(image_paths)

    root_path = Path(root)
    relative, stored_root = relativize_image_paths(image_paths, None)
    if stored_root is not None and str(root_path) != stored_root:
        logger.info(f"Rebasing image paths from {stored_root} onto {root_path}")

    return [str(root_path / p) if p and not is_url(p) else p for p in relative]
