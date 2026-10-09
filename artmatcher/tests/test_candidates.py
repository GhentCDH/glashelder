"""Checks for trying more than the top CLIP match with the ML classifier.

Run with `uv run python artmatcher/tests/test_candidates.py`.
"""

from typing import Any

import numpy as np

from artmatcher.batch import choose, score_round, shifted

# what the fake classifier says for each reference
CONFIDENCE = {"r0": 0.3, "r1": 0.6, "r2": 0.95, "r3": 0.4}


class FakeModel:
    def predict_proba(self, matrix):
        return np.stack([1 - matrix[:, 0], matrix[:, 0]], axis=1)


class FakeClassifier:
    _model = FakeModel()

    def _features_to_array(self, features):
        return np.array([features["confidence"]])


# stands in for an MLClassifier: only these two members are used
CLASSIFIER: Any = FakeClassifier()


def run(candidates: int, threshold: float, tried=None):
    """Score image 0, whose top matches are r0..r3, like serve and batch-classify."""
    tried = {} if tried is None else tried
    seen = []

    def extract(work_items):
        seen.extend((reference, sims) for _, _, reference, sims, _ in work_items)
        return {n: {"confidence": CONFIDENCE[ref]} for n, _, ref, _, _ in work_items}

    while score_round(
        tried,
        [0],
        threshold=threshold,
        candidates=candidates,
        glashelder_paths=["g0"],
        artwork_paths=["r0", "r1", "r2", "r3"],
        top_indices=np.array([[0, 1, 2, 3]]),
        top_similarities=np.array([[0.9, 0.8, 0.7, 0.6]]),
        classifier=CLASSIFIER,
        extract=extract,
    ):
        pass
    return choose(tried[0], threshold, candidates), tried, seen


def test_shifted() -> None:
    assert shifted([0.9, 0.8, 0.7, 0.6], 0) == [0.9, 0.8, 0.7, 0.6]
    assert shifted([0.9, 0.8, 0.7, 0.6], 2) == [0.7, 0.6, 0.6, 0.6]


def test_one_candidate_is_the_old_behaviour() -> None:
    rank, tried, seen = run(candidates=1, threshold=0.82)
    assert rank == 0 and len(tried[0]) == 1 and seen[0][0] == "r0"


def test_no_match_keeps_the_most_confident() -> None:
    rank, tried, _ = run(candidates=2, threshold=0.82)
    assert (rank, len(tried[0])) == (1, 2)


def test_stops_at_the_first_match() -> None:
    rank, tried, seen = run(candidates=4, threshold=0.82)
    assert rank == 2 and len(tried[0]) == 3  # r3 is never scored
    # the classifier saw r2's similarities as if it were the top match
    assert seen[2] == ("r2", [0.7, 0.6, 0.6, 0.6])


def test_lower_threshold_stops_earlier() -> None:
    rank, tried, _ = run(candidates=4, threshold=0.5)
    assert (rank, len(tried[0])) == (1, 2)


def test_threshold_changes_reuse_what_was_scored() -> None:
    _, tried, _ = run(candidates=4, threshold=0.82)
    # lower: r1 is already a match, though r2 was scored with more confidence
    rank, _, seen = run(candidates=4, threshold=0.5, tried=tried)
    assert rank == 1 and seen == []
    # higher than anything: the last candidate is scored, the best one kept
    rank, tried, seen = run(candidates=4, threshold=0.99, tried=tried)
    assert rank == 2 and [ref for ref, _ in seen] == ["r3"] and len(tried[0]) == 4


def test_a_score_added_meanwhile_is_kept() -> None:
    """Another request scored the same candidate while this one extracted."""
    tried: dict = {}

    def extract(work_items):
        tried.setdefault(0, []).append((0.1, {"confidence": 0.1}))
        return {n: {"confidence": CONFIDENCE[ref]} for n, _, ref, _, _ in work_items}

    score_round(
        tried,
        [0],
        threshold=0.82,
        candidates=4,
        glashelder_paths=["g0"],
        artwork_paths=["r0", "r1", "r2", "r3"],
        top_indices=np.array([[0, 1, 2, 3]]),
        top_similarities=np.array([[0.9, 0.8, 0.7, 0.6]]),
        classifier=CLASSIFIER,
        extract=extract,
    )
    assert tried[0] == [(0.1, {"confidence": 0.1})]


if __name__ == "__main__":
    test_shifted()
    test_one_candidate_is_the_old_behaviour()
    test_no_match_keeps_the_most_confident()
    test_stops_at_the_first_match()
    test_lower_threshold_stops_earlier()
    test_threshold_changes_reuse_what_was_scored()
    test_a_score_added_meanwhile_is_kept()
    print("ok")
