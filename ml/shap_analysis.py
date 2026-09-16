"""Explainable AI: SHAP-based global and local feature importance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.exceptions import ValidationError
from core.logger import get_logger
from ml.xgboost_model import TrainedModel

logger = get_logger(__name__)


@dataclass
class SHAPResult:
    """Container for SHAP explanation artifacts.

    Attributes:
        explainer: The fitted ``shap.TreeExplainer``.
        shap_values: Raw SHAP values (list of arrays for multiclass, or a
            single array for binary/regression).
        feature_names: Ordered feature names matching SHAP value columns.
        global_importance: Mean absolute SHAP value per feature
            (aggregated across classes for multiclass).
    """

    explainer: Any
    shap_values: Any
    feature_names: list[str]
    global_importance: dict[str, float]


class SHAPAnalyzer:
    """Generates global/local SHAP explanations for a trained XGBoost model."""

    def explain(self, trained_model: TrainedModel) -> SHAPResult:
        """Compute SHAP values for the test split of a trained model.

        Args:
            trained_model: A :class:`ml.xgboost_model.TrainedModel`.

        Returns:
            A populated :class:`SHAPResult`.

        Raises:
            ValidationError: If ``shap`` is not installed.
        """
        try:
            import shap
        except ImportError as exc:
            raise ValidationError("shap is required for this stage: pip install shap") from exc

        import numpy as np

        dataset = trained_model.dataset
        explainer = shap.TreeExplainer(trained_model.model)
        shap_values = explainer.shap_values(dataset.X_test)

        if isinstance(shap_values, list):
            stacked = np.abs(np.stack(shap_values, axis=0))
            mean_abs = stacked.mean(axis=(0, 1))
        else:
            arr = np.abs(shap_values)
            mean_abs = arr.mean(axis=tuple(range(arr.ndim - 1))) if arr.ndim > 2 else arr.mean(axis=0)

        global_importance = {
            name: float(value) for name, value in zip(dataset.feature_names, mean_abs)
        }
        global_importance = dict(
            sorted(global_importance.items(), key=lambda kv: kv[1], reverse=True)
        )

        logger.info("Computed SHAP values for %d test samples.", len(dataset.X_test))

        return SHAPResult(
            explainer=explainer,
            shap_values=shap_values,
            feature_names=dataset.feature_names,
            global_importance=global_importance,
        )

    def save_summary_plot(self, result: SHAPResult, trained_model: TrainedModel, path: Path) -> Path:
        """Render and save a SHAP summary (beeswarm) plot.

        Args:
            result: Output of :meth:`explain`.
            trained_model: The trained model (for the test feature matrix).
            path: Destination ``.png`` path.

        Returns:
            The path written to.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        shap.summary_plot(
            result.shap_values, trained_model.dataset.X_test, feature_names=result.feature_names, show=False
        )
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        logger.info("Saved SHAP summary plot: %s", path)
        return path

    def save_dependence_plot(
        self, result: SHAPResult, trained_model: TrainedModel, feature: str, path: Path
    ) -> Path:
        """Render and save a SHAP dependence plot for one feature.

        Args:
            result: Output of :meth:`explain`.
            trained_model: The trained model (for the test feature matrix).
            feature: Feature name to plot dependence for.
            path: Destination ``.png`` path.

        Returns:
            The path written to.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        shap_values = result.shap_values[0] if isinstance(result.shap_values, list) else result.shap_values

        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        shap.dependence_plot(
            feature, shap_values, trained_model.dataset.X_test, feature_names=result.feature_names, show=False
        )
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        logger.info("Saved SHAP dependence plot for '%s': %s", feature, path)
        return path

    def save_global_importance_json(self, result: SHAPResult, path: Path) -> Path:
        """Serialize global SHAP feature importance to JSON.

        Args:
            result: Output of :meth:`explain`.
            path: Destination ``.json`` path.

        Returns:
            The path written to.
        """
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.global_importance, f, indent=2)
        return path
