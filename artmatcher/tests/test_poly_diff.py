"""The poly diff shared by the simple classifier and the ML features."""

import json

import numpy as np
import pytest

from artmatcher.features import distribution_features
from artmatcher.similarity_classifier import SimpleClassifier, poly_diff


def reference(similarities: list[float]) -> float:
    """The formula written out: a line fitted through the rest with Gaussian
    weights that fall to 0.3, compared with the top similarity at x = 0."""
    rest = similarities[1:]
    n = len(rest)
    sigma = (n - 1) / np.sqrt(-2 * np.log(0.3))
    weights = np.exp(-0.5 * (np.arange(n) / sigma) ** 2)
    coeffs = np.polyfit(np.arange(n), rest, 1, w=weights / weights.sum())
    return similarities[0] - coeffs[-1]


@pytest.mark.parametrize(
    "similarities",
    [
        [0.82, 0.51, 0.43, 0.41, 0.38, 0.35, 0.33, 0.30, 0.28, 0.25],
        [0.9, 0.85, 0.2],
        [0.5, 0.5, 0.5, 0.5],
        list(np.linspace(0.95, 0.1, 50)),
    ],
)
def test_matches_formula(similarities):
    assert poly_diff(similarities) == reference(similarities)
    assert distribution_features(similarities)["poly_diff"] == poly_diff(similarities)
    assert SimpleClassifier().classify(similarities).diff == poly_diff(similarities)


def test_known_value():
    similarities = [0.82, 0.51, 0.43, 0.41, 0.38, 0.35, 0.33, 0.30, 0.28, 0.25]
    assert poly_diff(similarities) == pytest.approx(0.3368, abs=1e-4)


@pytest.mark.parametrize("similarities", [[], [0.7]])
def test_too_short(similarities):
    assert poly_diff(similarities) == 0.0


def test_verdict_is_plain_python():
    """Labels go into the session file verbatim, so numpy's scalars would not do."""
    result = SimpleClassifier().classify([0.82, 0.51, 0.43, 0.41])
    assert type(result.is_match) is bool
    assert type(result.diff) is float
    json.dumps({"is_match": result.is_match, "diff": result.diff})
