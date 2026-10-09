"""Distribution-shape features of the CLIP similarity scores of a pair."""

import numpy as np

from ..similarity_classifier import poly_diff


def _sanitize_features(features: dict[str, float]) -> dict[str, float]:
    """Replace NaN and Inf values with safe defaults."""
    sanitized = {}
    for key, value in features.items():
        if np.isnan(value) or np.isinf(value):
            # Use 0 as default for most features, 1.0 for ratios and warp_stretch
            if "ratio" in key or "warp_stretch" in key:
                sanitized[key] = 1.0
            elif "hash_distance" in key:
                sanitized[key] = 64.0  # Max hamming distance for 64-bit hash
            else:
                sanitized[key] = 0.0
        else:
            # Clip to reasonable range to avoid sklearn issues
            sanitized[key] = float(np.clip(value, -1e6, 1e6))
    return sanitized


def distribution_features(similarities: list[float]) -> dict[str, float]:
    """Distribution shape features of the top-k similarity scores (descending)."""
    features = {}
    sims = np.array(similarities)
    n = len(sims)

    features["sim_0"] = float(sims[0]) if n > 0 else 0.0
    features["sim_1"] = float(sims[1]) if n > 1 else 0.0
    features["sim_2"] = float(sims[2]) if n > 2 else 0.0

    features["gap_01"] = features["sim_0"] - features["sim_1"]
    features["gap_02"] = features["sim_0"] - features["sim_2"]
    features["gap_12"] = features["sim_1"] - features["sim_2"]

    rest = sims[1:10] if n > 1 else np.array([0.0])
    features["mean_1to9"] = float(np.mean(rest))
    features["std_1to9"] = float(np.std(rest)) if len(rest) > 1 else 0.0

    # Z-score with minimum std threshold to avoid huge values
    std_threshold = 1e-6
    if features["std_1to9"] > std_threshold:
        features["sim_0_zscore"] = (
            features["sim_0"] - features["mean_1to9"]
        ) / features["std_1to9"]
    else:
        features["sim_0_zscore"] = 0.0

    if len(rest) > 1:
        x = np.arange(len(rest))
        try:
            coeffs = np.polyfit(x, rest, 1)
            features["decay_rate"] = float(coeffs[0])
        except (np.linalg.LinAlgError, ValueError):
            features["decay_rate"] = 0.0
    else:
        features["decay_rate"] = 0.0

    features["sim_min"] = float(np.min(sims[:10])) if n > 0 else 0.0
    features["sim_range"] = features["sim_0"] - features["sim_min"]

    features["poly_diff"] = poly_diff(similarities)

    # Ratio with minimum denominator threshold
    min_denom = 1e-6
    features["sim_ratio_01"] = (
        features["sim_0"] / max(features["sim_1"], min_denom)
        if features["sim_1"] > min_denom
        else 1.0
    )

    return features
