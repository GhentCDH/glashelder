"""Checks for the embedding store.

Run with `uv run python embed/tests/test_store.py`.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

from embed import Store, is_store, load_embeddings, read_embeddings

HASH_A, HASH_B = "aa" * 16, "bb" * 16


def fresh(**kwargs) -> Store:
    path = Path(tempfile.mkdtemp()) / "test.store"
    return Store.create_or_open(path, model="clip", model_id="m", **kwargs)


def vectors(n: int, dim: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def test_append_update_reopen() -> None:
    store = fresh(root="/images")
    store.put(["a.jpg", "b.jpg"], [HASH_A, None], vectors(2))
    store.put(["c.jpg", "a.jpg"], [None, HASH_B], vectors(2, seed=1))

    again = Store.open(store.path)
    assert again.refs() == ["a.jpg", "b.jpg", "c.jpg"]
    assert again.hashes() == {"a.jpg": HASH_B, "b.jpg": None, "c.jpg": None}
    # a.jpg was overwritten in place, keeping its row
    np.testing.assert_allclose(again.vectors()[0], vectors(2, seed=1)[1], atol=1e-2)
    np.testing.assert_allclose(again.vectors()[2], vectors(2, seed=1)[0], atol=1e-2)
    assert again.dim == 4 and again.dtype == np.float16 and again.nbytes == 3 * 4 * 2


def test_other_model_is_refused() -> None:
    store = fresh()
    try:
        Store.create_or_open(store.path, model="dinov2", model_id="other")
    except ValueError as e:
        assert "other" in str(e)
    else:
        raise AssertionError("a store must not mix models")


def test_interrupted_write_is_cut_off() -> None:
    store = fresh()
    store.put(["a", "b"], [None, None], vectors(2))
    # a crash after the vector and hash were written, halfway through the ref line
    with (store.path / "vectors.bin").open("ab") as file:
        file.write(vectors(1).astype(np.float16).tobytes())
    with (store.path / "hashes.bin").open("ab") as file:
        file.write(bytes(16))
    with (store.path / "refs.txt").open("a") as file:
        file.write("hal")

    again = Store.open(store.path)
    assert again.refs() == ["a", "b"]
    assert again.vectors().shape == (2, 4)
    again.put(["c"], [None], vectors(1))
    assert Store.open(store.path).refs() == ["a", "b", "c"]


def test_prune() -> None:
    store = fresh()
    store.put(["a", "b", "c"], [HASH_A, None, HASH_B], vectors(3))
    assert store.prune({"a", "c"}) == 1
    again = Store.open(store.path)
    assert again.refs() == ["a", "c"]
    assert again.hashes() == {"a": HASH_A, "c": HASH_B}
    np.testing.assert_allclose(again.vectors()[1], vectors(3)[2], atol=1e-2)


def test_load_rebases_onto_the_local_root() -> None:
    store = fresh(root="/on/the/gpu/machine")
    store.put(["x.jpg", "https://example.org/y.jpg"], [None, None], vectors(2))
    data = load_embeddings(store.path, root="/here")
    assert data["images"] == ["/here/x.jpg", "https://example.org/y.jpg"]
    assert isinstance(data["embeddings"], torch.Tensor)
    assert data["embeddings"].shape == (2, 4) and data["model_id"] == "m"


def test_legacy_pickle_without_root() -> None:
    path = Path(tempfile.mkdtemp()) / "legacy.pkl"
    with path.open("wb") as file:
        pickle.dump(
            {
                "images": ["/mnt/a/1.jpg", "/mnt/a/2.jpg"],
                "embeddings": torch.zeros(2, 4),
            },
            file,
        )
    assert not is_store(path)
    raw = read_embeddings(path)
    assert raw["root"] == "/mnt/a" and raw["images"] == ["1.jpg", "2.jpg"]
    assert load_embeddings(path, root="/b")["images"] == ["/b/1.jpg", "/b/2.jpg"]


if __name__ == "__main__":
    test_append_update_reopen()
    test_other_model_is_refused()
    test_interrupted_write_is_cut_off()
    test_prune()
    test_load_rebases_onto_the_local_root()
    test_legacy_pickle_without_root()
    print("ok")
