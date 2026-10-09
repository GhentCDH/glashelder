"""What the four label buckets mean."""

import pytest

from artmatcher.web.session import (
    BUCKETS,
    LabelDataset,
    bucket_for,
    truth,
    was_match,
)
from artmatcher.web.store import normalize_dataset


@pytest.mark.parametrize(
    ("bucket", "said_match", "is_match"),
    [
        ("positives", True, True),
        ("negatives", False, False),
        ("false_positives", True, False),
        ("false_negatives", False, True),
    ],
)
def test_meaning(bucket: str, said_match: bool, is_match: bool) -> None:
    assert was_match(bucket) is said_match
    assert truth(bucket) is is_match
    assert bucket_for(said_match == is_match, said_match) == bucket


def test_every_bucket_is_a_field() -> None:
    assert BUCKETS == tuple(LabelDataset().as_dict())


def test_labeled_and_counts() -> None:
    dataset = LabelDataset(
        positives=[{"glashelder_index": 0}],
        negatives=[{"glashelder_index": 1}, {"glashelder_index": 2}],
        false_positives=[{"glashelder_index": 3}],
        false_negatives=[{"glashelder_index": 4}],
    )
    assert [(e["glashelder_index"], t) for e, t in dataset.labeled()] == [
        (0, True),
        (1, False),
        (2, False),
        (3, False),
        (4, True),
    ]
    assert dataset.counts() == {
        "positives": 1,
        "negatives": 2,
        "false_positives": 1,
        "false_negatives": 1,
        "total": 5,
    }


def test_forget() -> None:
    dataset = LabelDataset(
        positives=[{"glashelder_index": 0}], negatives=[{"glashelder_index": 1}]
    )
    dataset.forget(0)
    assert dataset.entries() == [{"glashelder_index": 1}]


def test_import_fills_is_match() -> None:
    dataset = normalize_dataset(
        {"false_positives": [{"glashelder_index": 0}], "negatives": [{}]}
    )
    assert dataset.false_positives[0]["is_match"] is True
    assert dataset.negatives[0]["is_match"] is False
