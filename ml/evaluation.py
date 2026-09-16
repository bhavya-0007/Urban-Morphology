"""Model evaluation: accuracy, precision/recall/F1, confusion matrix."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.logger import get_logger
from ml.xgboost_model import TrainedModel

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    """Standard classification evaluation results.

    Attributes:
        accuracy: Overall accuracy on the test split.
        precision_macro: Macro-averaged precision.
        recall_macro: Macro-averaged recall.
        f1_macro: Macro-averaged F1 score.
        per_class_report: Full ``sklearn`` classification report dict.
        confusion_matrix: Confusion matrix as a nested list.
        class_labels: Ordered class label names matching the confusion
            matrix rows/columns.
    """

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class_report: dict[str, Any] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    class_labels: list[str] = field(default_factory=list)


class ModelEvaluator:
    """Computes and persists standard evaluation metrics for a trained model."""

    def evaluate(self, trained_model: TrainedModel) -> EvaluationReport:
        """Evaluate a trained model on its held-out test split.

        Args:
            trained_model: A :class:`ml.xgboost_model.TrainedModel`.

        Returns:
            A populated :class:`EvaluationReport`.
        """
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        dataset = trained_model.dataset
        y_pred = trained_model.model.predict(dataset.X_test)
        y_true = dataset.y_test

        class_labels = [str(dataset.class_names[i]) for i in sorted(dataset.class_names)]

        report = EvaluationReport(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            per_class_report=classification_report(
                y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0
            ),
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
            class_labels=class_labels,
        )

        logger.info(
            "Evaluation: accuracy=%.3f, f1_macro=%.3f", report.accuracy, report.f1_macro
        )
        return report

    def save_report(self, report: EvaluationReport, path: Path) -> Path:
        """Serialize an evaluation report to JSON.

        Args:
            report: The report to serialize.
            path: Destination ``.json`` path.

        Returns:
            The path written to.
        """
        import json
        from dataclasses import asdict

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("Saved evaluation report: %s", path)
        return path
