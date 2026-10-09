"""ML-based classifier for artwork matching using Random Forest."""

import logging
import pickle
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from rich.console import Console

from .features.image import NO_IMAGE_FEATURES, _pair_features, make_orb
from .features.similarity import _sanitize_features, distribution_features
from .training import extract_training_features, read_labeled_entries

logger = logging.getLogger(__name__)
console = Console()


class MLClassifier:
    """ML-based classifier using Random Forest with calibrated probabilities."""

    FEATURE_ORDER: ClassVar[list[str]] = [
        "sim_0",
        "sim_1",
        "sim_2",
        "gap_01",
        "gap_02",
        "gap_12",
        "orb_num_matches",
        "orb_num_inliers",
        "orb_confidence",
        "orb_warp_stretch",
        "mean_1to9",
        "std_1to9",
        "sim_0_zscore",
        "decay_rate",
        "sim_min",
        "sim_range",
        "poly_diff",
        "sim_ratio_01",
        "inlier_match_ratio",
        "phash_distance",
        "dhash_distance",
        "whash_distance",
        "ahash_distance",
        "aligned_phash_distance",
        "aligned_dhash_distance",
        "aligned_whash_distance",
        "aligned_ahash_distance",
        "aligned_ncc",
        "aligned_ssim",
    ]

    def __init__(self, threshold: float = 0.5):
        """Initialize the classifier.

        Args:
            threshold: Confidence threshold for is_match decision
        """
        self.threshold = threshold
        self._model = None
        self._orb: tuple | None = None

    def _get_orb(self) -> tuple:
        """The ORB detector and matcher, made on first use."""
        if self._orb is None:
            self._orb = make_orb()
        return self._orb

    def _features_to_array(self, features: dict[str, float]) -> NDArray:
        """Convert feature dict to ordered numpy array."""
        return np.array([features.get(f, 0.0) for f in self.FEATURE_ORDER])

    def extract_features(
        self,
        similarities: list[float],
        glashelder_path: str | None = None,
        artwork_path: str | None = None,
    ) -> dict[str, float]:
        """All features for a pair: distribution features plus, when both image
        paths are given, ORB and hash features."""
        image_features = (
            _pair_features(glashelder_path, artwork_path, *self._get_orb()).as_dict()
            if glashelder_path and artwork_path
            else NO_IMAGE_FEATURES
        )
        return _sanitize_features(distribution_features(similarities) | image_features)

    def predict_proba(self, features: list[dict[str, float]]) -> NDArray:
        """Match-confidence prediction for a batch of feature dicts."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        matrix = np.array([self._features_to_array(f) for f in features])
        return self._model.predict_proba(matrix)[:, 1].astype(float)

    def train(
        self,
        dataset_path: str,
        n_estimators: int = 100,
        cv_folds: int = 5,
        extract_image_features: bool = True,
        workers: int = 1,
    ) -> None:
        """Train the classifier from a labeled JSON dataset.

        Args:
            dataset_path: Path to labeling_dataset.json from the web UI
            n_estimators: Number of trees in the Random Forest
            cv_folds: Number of cross-validation folds
            extract_image_features: Whether to extract ORB/hash features from image paths
            workers: Number of parallel workers for feature extraction (1=sequential)
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        entries_with_labels = read_labeled_entries(dataset_path)
        n_entries = len(entries_with_labels)
        console.print(f"[cyan]Found {n_entries} labeled samples[/cyan]")

        features_array, labels_array = extract_training_features(
            entries_with_labels,
            extract_image_features,
            workers,
            self._features_to_array,
        )

        n_samples = len(labels_array)
        n_positives = int(np.sum(labels_array))
        n_negatives = n_samples - n_positives

        console.print(
            f"[green]Training on {n_samples} samples "
            f"({n_positives} positives, {n_negatives} negatives)[/green]"
        )

        if n_samples < 10:
            raise ValueError(f"Need at least 10 samples for training, got {n_samples}")

        # Train Random Forest with cross-validation
        console.print("[cyan]Training Random Forest classifier...[/cyan]")
        base_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # Cross-validation with progress
        console.print(f"[cyan]Running {cv_folds}-fold cross-validation...[/cyan]")
        n_cv_folds = min(cv_folds, n_samples)
        cv_scores = cross_val_score(
            base_rf, features_array, labels_array, cv=n_cv_folds, scoring="accuracy"
        )
        cv_accuracy = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        console.print(
            f"[green]Cross-validation accuracy: {cv_accuracy:.3f} (+/- {cv_std:.3f})[/green]"
        )

        # Train final model with probability calibration
        console.print("[cyan]Calibrating probabilities...[/cyan]")
        self._model = CalibratedClassifierCV(
            base_rf,
            method="sigmoid",
            cv=min(3, n_samples),
        )
        self._model.fit(features_array, labels_array)

        # Get feature importance from the base estimator
        console.print("[cyan]Computing feature importance...[/cyan]")
        base_rf.fit(features_array, labels_array)
        importances = base_rf.feature_importances_
        feature_importance = {
            name: float(imp)
            for name, imp in zip(self.FEATURE_ORDER, importances, strict=True)
        }

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        console.print("\n[bold green]Top 5 features by importance:[/bold green]")
        for i, (name, imp) in enumerate(list(feature_importance.items())[:5]):
            console.print(f"  {i + 1}. [yellow]{name}[/yellow]: {imp:.4f}")

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk.

        Args:
            path: Path to save the model (.pkl file)
        """
        if self._model is None:
            raise RuntimeError("No model to save. Call train() first.")

        data = {
            "model": self._model,
            "threshold": self.threshold,
            "feature_order": self.FEATURE_ORDER,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path, threshold: float | None = None) -> "MLClassifier":
        """Load a trained model from disk.

        Args:
            path: Path to the saved model (.pkl file)
            threshold: Optional threshold override (uses saved threshold if None)

        Returns:
            MLClassifier with loaded model
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        if data.get("feature_order") != cls.FEATURE_ORDER:
            raise ValueError(
                f"{path} was trained on other features than this version computes, "
                "retrain it with `glashelder artmatcher train-classifier`"
            )
        classifier = cls(
            threshold=threshold if threshold is not None else data.get("threshold", 0.5)
        )
        classifier._model = data["model"]

        logger.info(f"Model loaded from {path}")
        return classifier
