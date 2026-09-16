"""Machine learning: dataset prep, XGBoost classification, evaluation, SHAP."""
from __future__ import annotations

from ml.dataset import DatasetBuilder, MLDataset
from ml.evaluation import EvaluationReport, ModelEvaluator
from ml.shap_analysis import SHAPAnalyzer, SHAPResult
from ml.xgboost_model import TrainedModel, XGBoostMorphologyClassifier

__all__ = [
    "DatasetBuilder",
    "MLDataset",
    "XGBoostMorphologyClassifier",
    "TrainedModel",
    "ModelEvaluator",
    "EvaluationReport",
    "SHAPAnalyzer",
    "SHAPResult",
]
