"""Reading embeddings off disk, a store or a legacy pickle."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from common.errors import UserError
from common.images import is_url
from common.paths import rebase_paths, relativize_image_paths

from .models import CLIP_MODEL_ID

if TYPE_CHECKING:
    from rich.progress import Progress

logger = logging.getLogger(__name__)

MAKE_EMBEDDINGS = (
    "Point it at an embedding store, made with `glashelder embed artworks` "
    "or `glashelder embed images`."
)


class EmbeddingsError(UserError, ValueError):
    """An embeddings path that does not hold embeddings."""


def not_embeddings(path: str | Path, data: object) -> EmbeddingsError:
    if not isinstance(data, dict):
        held = f"a {type(data).__name__}"
    else:
        held = "the keys " + ", ".join(map(str, data))
    hint = (
        " It looks like the artmatcher classifier (paths.ml_model)."
        if isinstance(data, dict) and {"threshold", "feature_order"} <= data.keys()
        else ""
    )
    return EmbeddingsError(
        f"{path} does not hold embeddings: it is a pickle with {held}.{hint} "
        f"{MAKE_EMBEDDINGS}"
    )


def read_embeddings(
    path: str | Path, root: str | Path | None = None, progress: Progress | None = None
) -> dict:
    """What an embeddings file holds, a store or a legacy pickle, as stored.

    Local image paths come back relative to `data["root"]`: the root recorded in
    the file, or `root`, or for old pickles without one, the directory all their
    absolute paths share. This is the raw form, for converting between formats;
    `load_embeddings` is what gives paths usable on this machine.

    A `progress` gets a bar for reading a pickle, which has to be read whole and
    can take minutes from a network mount.

    Returns:
        Dictionary containing 'images', 'embeddings', 'root', 'model',
        'model_id' and 'embedding_dim'. For a store the embeddings are a
        memory-mapped tensor, nothing is read until it is used.
    """
    from .store import Store, is_store

    if is_store(path):
        import torch

        try:
            store = Store.open(path)
        except ValueError as error:
            raise EmbeddingsError(str(error)) from error
        return {
            "model": store.model,
            "model_id": store.model_id,
            "embedding_dim": store.dim,
            "root": store.root,
            "images": store.refs(),
            "embeddings": torch.from_numpy(store.vectors()),
        }

    import pickle

    if not Path(path).exists():
        raise EmbeddingsError(f"{path} does not exist. {MAKE_EMBEDDINGS}")
    if Path(path).is_dir():
        raise EmbeddingsError(
            f"{path} is a directory, but not an embedding store. {MAKE_EMBEDDINGS}"
        )

    opened = (
        progress.open(path, "rb", description=f"reading {Path(path).name}")
        if progress is not None
        else open(path, "rb")
    )
    with opened as f:
        try:
            data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as error:
            raise EmbeddingsError(
                f"{path} is not an embedding store or an embeddings pickle. "
                f"{MAKE_EMBEDDINGS}"
            ) from error
    if not isinstance(data, dict) or not {"images", "embeddings"} <= data.keys():
        raise not_embeddings(path, data)

    # Handle legacy format without model metadata
    if "model" not in data:
        logger.warning(
            "Loading embeddings in legacy format (no model metadata). "
            "Assuming CLIP embeddings."
        )
        data["model"] = "clip"
        data["model_id"] = CLIP_MODEL_ID
        data["embedding_dim"] = 768

    stored_root = root if root is not None else data.get("root") or None
    data["images"], data["root"] = relativize_image_paths(data["images"], stored_root)
    return data


def load_embeddings(path: str | Path, root: str | Path | None = None) -> dict:
    """Load embeddings from a store or a legacy pickle.

    Image paths are always returned under `root`, which is the single place
    where the mismatch between the machine that generated the embeddings and
    the machine reading them gets resolved: the file keeps paths relative to
    the root it was made with, and here they are joined onto this one.

    Args:
        path: Path to the store directory or the pickle file.
        root: Directory the images live in on *this* machine. When omitted, the
            root recorded in the file (or inferred from it) is used, which
            only works if the images are in the same place as when embedded.

    Returns:
        Dictionary containing 'images' (absolute paths under `root`, or URLs),
        'embeddings', 'root', and model metadata.
    """
    data = read_embeddings(path)

    resolved_root = root if root is not None else data["root"]
    if resolved_root is None:
        if any(p and not is_url(p) for p in data["images"]):
            logger.warning(
                f"Embeddings at {path} have no image root and none was given; "
                "using image paths as stored."
            )
        return data

    data["root"] = str(resolved_root)
    data["images"] = rebase_paths(data["images"], resolved_root)

    return data
