"""XGBoost morphological classification model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.constants import RANDOM_SEED
from core.exceptions import ModelTrainingError
from core.logger import get_logger
from ml.dataset import MLDataset

logger = get_logger(__name__)


@dataclass
class TrainedModel:
    """A fitted XGBoost model plus the dataset context it was trained on.

    Attributes:
        model: The fitted ``xgboost.XGBClassifier``.
        dataset: The :class:`ml.dataset.MLDataset` used for training.
        best_params: Hyperparameters used (post-tuning if applicable).
    """

    model: Any
    dataset: MLDataset
    best_params: dict[str, Any]


class XGBoostMorphologyClassifier:
    """Trains and persists an XGBoost classifier for morphology classes."""

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def train(self, dataset: MLDataset, tune: bool = False) -> TrainedModel:
        """Fit the XGBoost classifier, optionally with grid-search tuning.

        Args:
            dataset: A prepared :class:`ml.dataset.MLDataset`.
            tune: If ``True``, run a small grid search over key
                hyperparameters before final fitting.

        Returns:
            A populated :class:`TrainedModel`.

        Raises:
            ModelTrainingError: If training fails.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ModelTrainingError(
                "xgboost is required for this stage: pip install xgboost"
            ) from exc

        base_params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
        )

        try:
            if tune:
                best_params = self._tune(dataset, base_params)
            else:
                best_params = base_params

            model = XGBClassifier(**best_params)
            model.fit(dataset.X_train, dataset.y_train)
        except Exception as exc:  # noqa: BLE001
            raise ModelTrainingError(f"XGBoost training failed: {exc}") from exc

        logger.info("XGBoost model trained with params: %s", best_params)
        return TrainedModel(model=model, dataset=dataset, best_params=best_params)

    def _tune(self, dataset: MLDataset, base_params: dict[str, Any]) -> dict[str, Any]:
        """Small grid search over depth/learning-rate/estimators."""
        from sklearn.model_selection import GridSearchCV
        from xgboost import XGBClassifier

        param_grid = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [200, 400],
        }
        search_base = {k: v for k, v in base_params.items() if k not in param_grid}
        search = GridSearchCV(
            XGBClassifier(**search_base),
            param_grid=param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
        )
        search.fit(dataset.X_train, dataset.y_train)
        logger.info("Grid search best score: %.3f, params: %s", search.best_score_, search.best_params_)
        return {**search_base, **search.best_params_}

    def save(self, trained_model: TrainedModel, path: Path) -> Path:
        """Persist a trained model to disk (JSON format).

        Args:
            trained_model: A :class:`TrainedModel`.
            path: Destination file path (``.json`` recommended).

        Returns:
            The path written to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        trained_model.model.save_model(str(path))
        logger.info("Saved XGBoost model: %s", path)
        return path

    def load(self, path: Path) -> Any:
        """Load a persisted XGBoost model from disk.

        Args:
            path: Path to a ``.json`` model file written by :meth:`save`.

        Returns:
            A fitted ``xgboost.XGBClassifier``.
        """
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(str(path))
        return model
