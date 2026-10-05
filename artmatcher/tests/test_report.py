"""Checks for `artmatcher report`, on a run and labels with known answers.

Run with `uv run python artmatcher/tests/test_report.py`.
"""

import math

from artmatcher.report import build, render, summary_lines, wilson
from artmatcher.web.session import LabelDataset

THRESHOLD = 0.82


def run(candidates: int = 2) -> dict:
    """1000 images: the first 100 are matches, half wikiart and half kikirpa."""
    results = [
        {
            "glashelder_index": i,
            "glashelder_path": f"/plates/{i}.jpg",
            "artwork_index": i,
            "artwork_path": f"https://example.org/{i}.jpg",
            "confidence": 0.9 if i < 100 else 0.1,
            "rank": 1 if i < 10 else 0,
            "similarities": [0.9, 0.8],
            "artwork": {"source": "wikiart" if i % 2 else "kikirpa"},
        }
        for i in range(1000)
    ]
    metadata = {
        "threshold": THRESHOLD,
        "candidates": candidates,
        "min_confidence": None,
    }
    return {"metadata": metadata, "results": results}


def labels() -> LabelDataset:
    """10 labeled matches (9 right) and 100 labeled non-matches (1 a true match)."""

    def entry(i: int, artwork: str | None = None) -> dict:
        return {
            "glashelder_index": i,
            "glashelder_path": f"/plates/{i}.jpg",
            "artwork_path": artwork or f"https://example.org/{i}.jpg",
        }

    dataset = LabelDataset()
    dataset.positives = [entry(i) for i in range(9)]
    dataset.false_positives = [entry(9)]
    dataset.false_negatives = [entry(100)]
    dataset.negatives = [entry(i) for i in range(101, 200)]
    # not usable: labeled for another artwork, and an image the run does not have
    dataset.negatives += [entry(500, "https://example.org/other.jpg"), entry(5000)]
    return dataset


def test_counts() -> None:
    report = build(run(), [], threshold=None)
    assert (report.images, report.matches, report.near_threshold) == (1000, 100, 0)
    sources = {g.name: (g.matches, g.non_matches) for g in report.by_source}
    assert sources == {"wikiart": (50, 450), "kikirpa": (50, 450)}
    assert report.by_rank == [(1, 90), (2, 10)]
    assert report.accuracy is None


def test_one_candidate_has_no_rank_table() -> None:
    data = run(candidates=1)
    for result in data["results"]:
        result["rank"] = 0
    assert build(data, []).by_rank == []


def test_accuracy_scales_each_verdict_by_its_size() -> None:
    accuracy = build(run(), [labels()]).accuracy
    assert accuracy is not None
    assert (accuracy.labels, accuracy.comparable) == (112, 110)
    assert (accuracy.other_artwork, accuracy.not_in_results) == (1, 1)
    assert (accuracy.precision.hits, accuracy.precision.total) == (9, 10)
    assert (accuracy.miss_rate.hits, accuracy.miss_rate.total) == (1, 100)

    assert accuracy.true_matches is not None and accuracy.recall is not None
    low, estimate, high = accuracy.true_matches
    assert math.isclose(estimate, 100 * 0.9 + 900 * 0.01) and low < estimate < high
    assert math.isclose(accuracy.recall[1], 90 / 99)

    # the curve at the run's threshold agrees with the strata
    _, precision, recall = next(
        p for p in accuracy.curve if math.isclose(p[0], THRESHOLD)
    )
    assert precision is not None and recall is not None
    assert math.isclose(precision, 0.9) and math.isclose(recall, 90 / 99)


def test_later_label_replaces_earlier() -> None:
    relabel = LabelDataset()
    relabel.false_positives = [
        {
            "glashelder_index": 0,
            "glashelder_path": "/plates/0.jpg",
            "artwork_path": "https://example.org/0.jpg",
        }
    ]
    accuracy = build(run(), [labels(), relabel]).accuracy
    assert accuracy is not None and accuracy.precision.hits == 8


def test_wilson() -> None:
    low, high = wilson(9, 10)
    assert 0.59 < low < 0.6 and 0.98 < high < 0.99
    assert wilson(0, 0) == (0.0, 1.0)


def test_render_and_summary() -> None:
    report = build(run(), [labels()])
    page = render(report)
    assert "Accuracy, from the labels" in page and "data:image/png;base64," in page
    assert any("recall" in line for line in summary_lines(report))


if __name__ == "__main__":
    test_counts()
    test_one_candidate_has_no_rank_table()
    test_accuracy_scales_each_verdict_by_its_size()
    test_later_label_replaces_earlier()
    test_wilson()
    test_render_and_summary()
    print("ok")
