import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def fit_poly(
    values: list[float],
    inputs: list[float | int],
    weights: None | list[float] | NDArray = None,
    degree: int = 1,
) -> tuple[float, ...]:
    """
    Fit a polynomial of given degree to data points.

    Parameters
    ----------
    values : list[float]
        Y-coordinates of the data points
    inputs : list[float]
        X-coordinates of the data points
    weights : list[float] | None
        Optional weights for each point (higher = more important)
    degree : int
        Degree of polynomial (1=line, 2=parabola, etc.)

    Returns
    -------
    tuple[float, ...]
        Polynomial coefficients from highest to lowest degree.
        For degree=1: (m, q) where y = m*x + q
        For degree=2: (a, b, c) where y = a*x² + b*x + c
    """
    coeffs = np.polyfit(inputs, values, degree, w=weights)
    return tuple(coeffs)


def gaussian_weights(
    length: int, center: int = 0, sigma: float = 1.0
) -> NDArray[np.float64]:
    """
    Generate normalized Gaussian weights.

    Parameters
    ----------
    length : int
        Number of weights to generate
    center : int
        Index of the peak (highest value)
    sigma : float
        Standard deviation of the Gaussian (controls spread)

    Returns
    -------
    NDArray[np.float64]
        Normalized weights that sum to 1.0

    Examples
    --------
    >>> gaussian_weights(5, center=2)  # Peak in middle
    array([0.10..., 0.23..., 0.32..., 0.23..., 0.10...])

    >>> gaussian_weights(10, center=0)  # Descending from left
    array([0.24..., 0.19..., 0.13..., ..., 0.00...])
    """
    indices = np.arange(length)
    weights = np.exp(-0.5 * ((indices - center) / sigma) ** 2)
    return weights / weights.sum()


def gaussian_weights_min(
    length: int, center: int = 0, min_value: float = 0.1
) -> NDArray[np.float64]:
    """
    Generate normalized Gaussian weights with specified minimum value.

    Parameters
    ----------
    length : int
        Number of weights to generate
    center : int
        Index of the peak (highest value)
    min_value : float
        The minimum weight value desired (before normalization).
        Actual minimum after normalization will be proportionally adjusted.

    Returns
    -------
    NDArray[np.float64]
        Normalized weights that sum to 1.0

    Examples
    --------
    >>> weights = gaussian_weights_min(10, center=0, min_value=0.1)
    >>> weights[0]  # Peak value
    >>> weights[-1]  # Should be close to min_value (proportionally)
    """
    if length == 1:
        return np.array([1.0])

    # Find the farthest index from center
    max_distance = max(abs(0 - center), abs(length - 1 - center))

    if max_distance == 0:
        # All points are at center
        return np.ones(length) / length

    # Solve for sigma: exp(-0.5 * (max_distance/sigma)^2) = min_value
    # => -0.5 * (max_distance/sigma)^2 = ln(min_value)
    # => sigma = max_distance / sqrt(-2 * ln(min_value))
    sigma = max_distance / np.sqrt(-2 * np.log(min_value))

    return gaussian_weights(length, center, sigma)


class SimpleClassifier:
    """Classifies whether or not the top n similarity scores indicate that index 0 (the most similar) is an actual match"""

    def __init__(self):
        pass

    def classify(self, similarities: list[float]) -> bool:
        weights = gaussian_weights_min(len(similarities[1:]), 0, 0.3)
        fitted = fit_poly(
            similarities[1:], list(range(len(similarities[1:]))), weights, 1
        )

        at_zero = fitted[-1]

        logger.info(f"Similarity[0] = {similarities[0]}")
        logger.info(f"Fitted @ 0 = {at_zero}")
        logger.info(f"Diff = {similarities[0] - at_zero}")

        return similarities[0] - at_zero > 0.2
