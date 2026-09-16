"""Local Climate Zone (LCZ) classification.

Implements two interchangeable strategies:
  1. A Random Forest classifier trained on morphometric/spectral features
     sampled at fishnet-cell centroids with labeled training points.
  2. A deterministic rule-based fallback usable when no labeled training
     data is available for a given year, based on published morphometric
     thresholds (Stewart & Oke, 2012; Bechtel et al., 2015).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.exceptions import ModelTrainingError
from core.logger import get_logger

logger = get_logger(__name__)

# Human-readable names for numeric LCZ codes used by this module. Kept
# local (rather than derived from config.constants.LCZ_CLASSES, which is
# a flat string enum list) because the classifier operates on integer
# codes, including the synonym codes -1/-2 for LCZ A/D.
LCZ_CODE_NAMES: dict[int, str] = {
    1: "LCZ 1: Compact high-rise",
    2: "LCZ 2: Compact midrise",
    3: "LCZ 3: Compact low-rise",
    4: "LCZ 4: Open high-rise",
    5: "LCZ 5: Open midrise",
    6: "LCZ 6: Open low-rise",
    8: "LCZ 8: Large low-rise",
    10: "LCZ 10: Heavy industry",
}


@dataclass
class LCZTrainingResult:
    """Result of a Random Forest LCZ training run.

    Attributes:
        model: The fitted ``sklearn`` classifier.
        feature_names: Ordered feature column names used for training.
        accuracy: Held-out accuracy on the internal validation split.
        classes: Class labels the model was trained on.
    """

    model: Any
    feature_names: list[str]
    accuracy: float
    classes: list[int]


class LCZClassifier:
    """Random-Forest-based LCZ classifier with a rule-based fallback."""

    def __init__(self, n_estimators: int = 300, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state

    def train_random_forest(
        self, features_df: Any, label_column: str = "lcz_class", test_size: float = 0.2
    ) -> LCZTrainingResult:
        """Train a Random Forest LCZ classifier on labeled fishnet features.

        Args:
            features_df: A pandas DataFrame of fishnet-cell features
                including ``label_column``.
            label_column: Column containing integer LCZ class codes.
            test_size: Held-out fraction for validation accuracy.

        Returns:
            A populated :class:`LCZTrainingResult`.

        Raises:
            ModelTrainingError: If there are too few labeled samples.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        labeled = features_df.dropna(subset=[label_column])
        if len(labeled) < 20:
            raise ModelTrainingError(
                f"Need at least 20 labeled samples to train LCZ classifier, got {len(labeled)}."
            )

        feature_cols = [c for c in labeled.columns if c != label_column]
        X = labeled[feature_cols].fillna(0.0)
        y = labeled[label_column].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y if y.nunique() > 1 else None
        )

        model = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state, class_weight="balanced"
        )
        model.fit(X_train, y_train)
        accuracy = float(accuracy_score(y_test, model.predict(X_test)))

        logger.info("LCZ Random Forest trained: accuracy=%.3f, n_train=%d", accuracy, len(X_train))

        return LCZTrainingResult(
            model=model,
            feature_names=feature_cols,
            accuracy=accuracy,
            classes=sorted(y.unique().tolist()),
        )

    def predict(self, model_result: LCZTrainingResult, features_df: Any) -> np.ndarray:
        """Predict LCZ classes for unlabeled fishnet cells.

        Args:
            model_result: Output of :meth:`train_random_forest`.
            features_df: DataFrame containing at least
                ``model_result.feature_names`` columns.

        Returns:
            Array of predicted integer LCZ class codes.
        """
        X = features_df[model_result.feature_names].fillna(0.0)
        return model_result.model.predict(X)

    def classify_rule_based(
        self,
        building_density: float,
        building_spacing_m: float,
        ndvi: float,
    ) -> int:
        """Deterministic LCZ assignment from morphometric thresholds.

        This is a coarse, published-threshold fallback (not a substitute
        for a properly trained classifier) used when labeled data is
        unavailable for a study year.

        Args:
            building_density: Built-up fraction in ``[0, 1]``.
            building_spacing_m: Mean nearest-neighbor building spacing in
                meters.
            ndvi: Mean NDVI of the cell.

        Returns:
            An integer LCZ class code from
            :data:`config.constants.LCZ_CLASSES` (or ``-1`` for
            unclassified natural/vegetated cells, represented separately
            as LCZ A/D).
        """
        if ndvi > 0.5:
            return -1  # LCZ A (dense trees) — handled outside numeric LCZ codes
        if ndvi > 0.3 and building_density < 0.1:
            return -2  # LCZ D (low plants)

        if building_density >= 0.7:
            return 1 if building_spacing_m < 15 else 3  # compact high-rise vs low-rise
        if building_density >= 0.4:
            if building_spacing_m < 20:
                return 2  # compact midrise
            return 5  # open midrise
        if building_density >= 0.2:
            return 6  # open low-rise
        return 8  # large low-rise / sparse industrial-scale footprint

    def class_name(self, code: int) -> str:
        """Human-readable LCZ class name for a code (including A/D synonyms).

        Args:
            code: LCZ integer code (may be the synonym codes ``-1``/``-2``
                used by :meth:`classify_rule_based`).

        Returns:
            The class name string.
        """
        if code == -1:
            return "LCZ A: Dense trees"
        if code == -2:
            return "LCZ D: Low plants"
        return LCZ_CODE_NAMES.get(code, f"Unknown LCZ ({code})")
