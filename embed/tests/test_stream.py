"""Checks for incremental embedding into a store, with a stand-in model.

Run with `uv run python embed/tests/test_stream.py`.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from embed import EmbeddingModel, Store, encode_into, pending


class FakeEmbedder(EmbeddingModel):
    """Embeds an image as its mean colour, so no model has to be loaded."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model_id(self) -> str:
        return "fake/1"

    @property
    def embedding_dim(self) -> int:
        return 3

    @property
    def supports_text(self) -> bool:
        return False

    def load(self, device: str) -> None: ...

    def encode_images(self, image_paths, batch_size=64):
        raise NotImplementedError

    def encode_pil_batch(self, images: list[Image.Image]) -> torch.Tensor:
        self.calls += len(images)
        pixels = [
            torch.from_numpy(np.asarray(image, dtype=np.float32)) for image in images
        ]
        return torch.stack([p.mean(dim=(0, 1)) / 255 for p in pixels])


def images(directory: Path, colours: dict[str, tuple[int, int, int]]) -> dict[str, str]:
    for name, colour in colours.items():
        Image.new("RGB", (4, 4), colour).save(directory / name)
    return {str(directory / name): name for name in colours}


def run(store: Store, refs: dict[str, str], verify: bool = False):
    embedder = FakeEmbedder()
    todo = pending(store, refs, verify=verify)
    return encode_into(store, embedder, refs, todo, batch_size=2), embedder.calls


def test_only_new_or_changed_images_are_embedded() -> None:
    directory = Path(tempfile.mkdtemp())
    refs = images(directory, {"a.png": (255, 0, 0), "b.png": (0, 255, 0)})
    store = Store.create_or_open(
        directory / "s.store", model="fake", model_id="fake/1", root=directory
    )

    done, calls = run(store, refs)
    assert (done.embedded, calls) == (2, 2)
    assert store.refs() == ["a.png", "b.png"] or store.refs() == ["b.png", "a.png"]

    # nothing new: nothing is even read
    done, calls = run(store, refs)
    assert (done.total, calls) == (0, 0)

    refs |= images(directory, {"c.png": (0, 0, 255)})
    done, calls = run(store, refs)
    assert (done.embedded, calls) == (1, 1) and len(store) == 3

    # verify reads everything but only embeds what changed
    images(directory, {"a.png": (10, 10, 10)})
    row = store.refs().index("a.png")
    done, calls = run(store, refs, verify=True)
    assert (done.loaded, done.embedded, calls) == (3, 1, 1)
    assert store.refs().index("a.png") == row  # updated in place
    assert abs(float(store.vectors()[row][0]) - 10 / 255) < 1e-3


def test_rows_without_hash_get_one_without_embedding() -> None:
    directory = Path(tempfile.mkdtemp())
    refs = images(directory, {"a.png": (255, 0, 0)})
    store = Store.create_or_open(
        directory / "s.store", model="fake", model_id="fake/1", root=directory
    )
    # as `embed convert` leaves a legacy row: a vector but no hash
    store.put(["a.png"], [None], torch.zeros(1, 3).numpy())

    done, calls = run(store, refs, verify=True)
    assert (done.hashed, done.embedded, calls) == (1, 0, 0)
    assert Store.open(store.path).hashes()["a.png"] is not None


if __name__ == "__main__":
    test_only_new_or_changed_images_are_embedded()
    test_rows_without_hash_get_one_without_embedding()
    print("ok")
