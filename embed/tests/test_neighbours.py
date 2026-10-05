"""Checks that saved neighbours stay exact as the stores change.

Run with `uv run python embed/tests/test_neighbours.py`.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from embed.neighbours import load, neighbours, path_for
from embed.search import topk
from embed.store import Store

DIM, K = 16, 5


def vectors(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, DIM)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def stores() -> tuple[Store, Store]:
    directory = Path(tempfile.mkdtemp())
    make = lambda name: Store.create_or_open(  # noqa: E731
        directory / name, model="m", model_id="m/1", dtype="float32"
    )
    queries, references = make("q.store"), make("r.store")
    queries.put([f"q{i}" for i in range(40)], [None] * 40, vectors(40, 1))
    references.put([f"r{i}" for i in range(300)], [None] * 300, vectors(300, 2))
    return queries, references


def search(queries: Store, references: Store):
    """Through the saved result, reopening the stores like a new process would."""
    queries, references = Store.open(queries.path), Store.open(references.path)
    q = torch.from_numpy(np.asarray(queries.vectors()))
    r = torch.from_numpy(np.asarray(references.vectors()))
    return neighbours(queries, references, q, r, K), (q, r)


def assert_exact(result, matrices) -> None:
    (similarities, indices), (q, r) = result, matrices
    expected_similarities, expected_indices = topk(q, r, K)
    assert np.array_equal(indices, expected_indices.numpy())
    assert np.allclose(similarities, expected_similarities.numpy(), atol=1e-6)


def saved_rows(queries: Store, references: Store) -> tuple[int, int]:
    saved = load(path_for(references, queries))
    assert saved is not None
    return saved.queries, saved.rows


def test_first_search_is_saved() -> None:
    queries, references = stores()
    assert_exact(*search(queries, references))
    assert saved_rows(queries, references) == (40, 300)


def test_new_references_are_merged_exactly() -> None:
    queries, references = stores()
    search(queries, references)
    # some of these are near copies of the queries, so they must win the top-k
    near = vectors(40, 1)[:10] + 0.01 * vectors(10, 3)
    references.put([f"new{i}" for i in range(10)], [None] * 10, near)
    references.put([f"more{i}" for i in range(50)], [None] * 50, vectors(50, 4))
    result = search(queries, references)
    assert_exact(*result)
    assert saved_rows(queries, references) == (40, 360)
    assert result[0][1][0, 0] == 300  # query 0 found its near copy


def test_new_queries_are_searched() -> None:
    queries, references = stores()
    search(queries, references)
    queries.put([f"q{i}" for i in range(40, 55)], [None] * 15, vectors(15, 5))
    assert_exact(*search(queries, references))
    assert saved_rows(queries, references) == (55, 300)


def test_rows_changed_in_place_start_over() -> None:
    queries, references = stores()
    search(queries, references)
    # what --verify does for an image whose bytes changed
    references.put(["r7"], [None], vectors(40, 1)[3:4])
    assert Store.open(references.path).edits == 1
    result = search(queries, references)
    assert_exact(*result)
    assert result[0][1][3, 0] == 7


def test_pruned_references_start_over() -> None:
    queries, references = stores()
    search(queries, references)
    references.prune({f"r{i}" for i in range(0, 300, 2)})
    assert_exact(*search(queries, references))
    assert saved_rows(queries, references) == (40, 150)


if __name__ == "__main__":
    test_first_search_is_saved()
    test_new_references_are_merged_exactly()
    test_new_queries_are_searched()
    test_rows_changed_in_place_start_over()
    test_pruned_references_start_over()
    print("ok")
