"""Checks for serving `batch-classify` results that left rows out.

Run with `uv run python artmatcher/tests/test_precomputed.py`.
"""

from artmatcher.web.source import PrecomputedSource


def source() -> PrecomputedSource:
    """Images 0, 2 and 5 of a run made with `--min-confidence`; 5 is CLIP rank 2."""
    results = [
        {
            "glashelder_index": index,
            "glashelder_path": f"/elsewhere/plates/{index}.jpg",
            "artwork_index": 10 + index,
            "artwork_path": f"https://example.org/{index}.jpg",
            "confidence": confidence,
            "rank": rank,
            "similarities": [0.9, 0.8],
        }
        for index, confidence, rank in [(0, 0.9, 0), (2, 0.4, 0), (5, 0.95, 2)]
    ]
    return PrecomputedSource(results, "/here", None)


def test_keyed_by_glashelder_index() -> None:
    found = source()
    assert list(found.indices) == [0, 2, 5]
    assert found.total_images == 6
    assert found.classify(1, 0.5) is None and found.glashelder_path(3) is None

    item = found.classify(5, 0.5)
    assert item is not None
    assert item.glashelder_index == 5 and item.is_match
    assert item.glashelder_path == "/here/5.jpg"
    assert found.glashelder_path(5) == "/here/5.jpg"
    assert (item.rank, item.best_index) == (2, 15)


if __name__ == "__main__":
    test_keyed_by_glashelder_index()
    print("ok")
