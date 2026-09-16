"""Dataset preparation for the ML classification stage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.constants import RANDOM_SEED
from core.exceptions import ValidationError
from core.logger import get_logger
from core.validation import validate_dataframe_not_empty

logger = get_logger(__name__)

DEFAULT_FEATURE_COLUMNS: list[str] = [
    "NDVI",
    "NDBI",
    "ISF",
    "MNDWI",
    "BSI",
    "UI",
    "building_density",
    "building_spacing_m",
    "patch_density",
    "edge_density",
    "contrast",
    "entropy",
    "homogeneity",
]


@dataclass
class MLDataset:
    """Container for a train/test-ready feature matrix and target vector.

    Attributes:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target vector.
        y_test: Test target vector.
        feature_names: Ordered feature column names.
        class_names: Mapping of encoded label -> original class value.
    """

    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    feature_names: list[str]
    class_names: dict[int, Any]


class DatasetBuilder:
    """Builds a clean, encoded ML-ready dataset from fishnet features."""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        target_column: str = "morphology_class",
        test_size: float = 0.2,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def build(self, fishnet_df: Any) -> MLDataset:
        """Build a stratified train/test split from fishnet features.

        Args:
            fishnet_df: A (Geo)DataFrame containing feature columns and
                ``self.target_column``.

        Returns:
            A populated :class:`MLDataset`.

        Raises:
            ValidationError: If required columns are missing or too few
                labeled rows remain after cleaning.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        validate_dataframe_not_empty(fishnet_df, "fishnet feature dataset")

        missing_cols = [c for c in self.feature_columns if c not in fishnet_df.columns]
        if missing_cols:
            raise ValidationError(f"Fishnet dataset missing feature columns: {missing_cols}")
        if self.target_column not in fishnet_df.columns:
            raise ValidationError(f"Fishnet dataset missing target column '{self.target_column}'.")

        df = fishnet_df[[*self.feature_columns, self.target_column]].dropna(
            subset=[self.target_column]
        )
        df[self.feature_columns] = df[self.feature_columns].fillna(0.0)

        if len(df) < 20:
            raise ValidationError(
                f"Only {len(df)} labeled rows available; need at least 20 to train."
            )

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(df[self.target_column])
        class_names = {int(i): cls for i, cls in enumerate(encoder.classes_)}

        X = df[self.feature_columns]

        stratify = y_encoded if len(set(y_encoded)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )

        logger.info(
            "Built ML dataset: %d train / %d test rows, %d features, %d classes.",
            len(X_train), len(X_test), len(self.feature_columns), len(class_names),
        )

        return MLDataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=self.feature_columns,
            class_names=class_names,
        )
